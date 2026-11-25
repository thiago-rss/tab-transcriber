use std::collections::VecDeque;
use std::env;
use std::path::PathBuf;
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

use audio_capture::{
    find_process_by_pid, find_processes_by_query, list_audio_processes, AudioCapturePort,
    CaptureTarget, MacOSLoopbackCapture,
};
use dsp_pitch::{
    midi_to_note_name, HpssSeparator, InstrumentFilter, InstrumentFocus, NoteStabilizer,
    StereoMode, YinDetector,
};
use poly_transcriber::{
    BasicPitchEngine, DecodeConfig, LinearResampler, MODEL_SAMPLE_RATE, MODEL_WINDOW_SAMPLES,
};
use tab_solver::{
    render_ascii_chord_tab, render_ascii_tab, solve_polyphonic_with_config,
    solve_tablature_with_config, NeckRegion, SolverConfig,
};

const YIN_WINDOW_SIZE: usize = 2048;
const YIN_HOP_SIZE: usize = 512;
const MAX_HISTORY: usize = 12;

fn select_tuning(arg: Option<&str>) -> (&'static str, Vec<u8>, Vec<&'static str>) {
    match arg.unwrap_or("standard").to_lowercase().as_str() {
        "dadgad" => (
            "DADGAD",
            vec![38, 45, 50, 55, 57, 62],
            vec!["D", "A", "D", "G", "A", "d"],
        ),
        "dropd" => (
            "Drop D",
            vec![38, 45, 50, 55, 59, 64],
            vec!["D", "A", "D", "G", "B", "e"],
        ),
        "bstandard" => (
            "B Standard (6 cordas)",
            vec![35, 40, 45, 50, 54, 59],
            vec!["B", "E", "A", "D", "F#", "b"],
        ),
        _ => (
            "Standard (E-A-D-G-B-e)",
            vec![40, 45, 50, 55, 59, 64],
            vec!["E", "A", "D", "G", "B", "e"],
        ),
    }
}

fn print_audio_processes(show_all: bool) {
    let procs = list_audio_processes(!show_all);
    if procs.is_empty() {
        println!(
            "Nenhum aplicativo {} encontrado.",
            if show_all { "de áudio" } else { "emitindo áudio no momento" }
        );
        return;
    }
    println!(
        "{:<8} {:<8} {:<10} {:<26} {}",
        "OBJ_ID", "PID", "STATUS", "APLICATIVO", "BUNDLE ID"
    );
    println!("{}", "-".repeat(86));
    for p in procs {
        let status = if p.is_running_output { "TOCANDO" } else { "inativo" };
        println!(
            "{:<8} {:<8} {:<10} {:<26} {}",
            p.object_id, p.pid, status, p.name, p.bundle_id
        );
    }
}

fn resolve_target(
    app_filter: Option<&str>,
    pid_filter: Option<i32>,
) -> Result<(CaptureTarget, String), String> {
    if let Some(pid) = pid_filter {
        let proc = find_process_by_pid(pid)
            .ok_or_else(|| format!("PID {} não encontrado na lista de áudio.", pid))?;
        return Ok((
            CaptureTarget::ProcessObjects(vec![proc.object_id]),
            format!("PID {} ({})", proc.pid, proc.name),
        ));
    }
    if let Some(query) = app_filter {
        let matches = find_processes_by_query(query);
        if matches.is_empty() {
            return Err(format!("Nenhum processo corresponde a '{}'.", query));
        }
        let ids: Vec<u32> = matches.iter().map(|p| p.object_id).collect();
        return Ok((CaptureTarget::ProcessObjects(ids), format!("Filtro '{}'", query)));
    }
    Ok((CaptureTarget::SystemGlobal, "Áudio Global".to_string()))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().skip(1).collect();
    let mode = args.first().map(|s| s.as_str()).unwrap_or("poly");

    if mode == "list" {
        print_audio_processes(args.iter().any(|a| a == "--all"));
        return Ok(());
    }

    let mut tuning_arg: Option<String> = None;
    let mut app_filter: Option<String> = None;
    let mut pid_filter: Option<i32> = None;
    let mut focus = InstrumentFocus::FullMix;
    let mut stereo_mode = StereoMode::MonoMix;
    let mut region = NeckRegion::Auto;
    let mut enable_hpss = true;
    let mut model_path = PathBuf::from("models/nmp.onnx");

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--app" if i + 1 < args.len() => { app_filter = Some(args[i + 1].clone()); i += 2; }
            "--pid" if i + 1 < args.len() => { pid_filter = Some(args[i + 1].parse::<i32>()?); i += 2; }
            "--focus" if i + 1 < args.len() => { focus = InstrumentFocus::from_str_id(&args[i + 1]); i += 2; }
            "--stereo" if i + 1 < args.len() => { stereo_mode = StereoMode::from_str_id(&args[i + 1]); i += 2; }
            "--region" if i + 1 < args.len() => { region = NeckRegion::from_str_id(&args[i + 1]); i += 2; }
            "--no-hpss" => { enable_hpss = false; i += 1; }
            "--model" if i + 1 < args.len() => { model_path = PathBuf::from(&args[i + 1]); i += 2; }
            other if !other.starts_with("--") && tuning_arg.is_none() => {
                tuning_arg = Some(other.to_string());
                i += 1;
            }
            _ => i += 1,
        }
    }

    let (tuning_name, tuning_midi, string_labels) = select_tuning(tuning_arg.as_deref());
    let (target, target_label) = resolve_target(app_filter.as_deref(), pid_filter)?;
    let solver_cfg = SolverConfig {
        region,
        ..SolverConfig::default()
    };

    let (loopback, mut consumer) = MacOSLoopbackCapture::start_with_target(96_000, target)?;
    let info = loopback.stream_info();
    let mut filter = InstrumentFilter::new(info.sample_rate as f32, focus);
    let mut hpss = HpssSeparator::new(1.3);

    println!(
        "Modo: {} | {} | {} | {} | HPSS: {} | Região: {} | Fonte: {}",
        mode,
        tuning_name,
        stereo_mode.label(),
        focus.label(),
        if enable_hpss { "ON" } else { "OFF" },
        region.label(),
        target_label
    );

    if mode == "mono" {
        let mut detector = YinDetector::new(info.sample_rate as f32, YIN_WINDOW_SIZE);
        let (min_hz, max_hz) = focus.yin_freq_bounds();
        detector.set_freq_range(min_hz, max_hz);

        let mut stabilizer = NoteStabilizer::new(3, 6);
        let mut window = [0.0f32; YIN_WINDOW_SIZE];
        let mut hop_buf = [0.0f32; YIN_HOP_SIZE];
        let mut read_chunk = [[0.0f32; 2]; 512];
        let mut mono_chunk = Vec::with_capacity(512);
        let mut hpss_chunk = Vec::with_capacity(768);
        let mut filtered_chunk = Vec::with_capacity(768);
        let mut hop_filled = 0usize;
        let mut phrase: Vec<u8> = Vec::with_capacity(MAX_HISTORY);
        let mut warmths: Vec<f32> = Vec::with_capacity(MAX_HISTORY);

        loop {
            let (popped, _) = consumer.pop_partial_slice(&mut read_chunk);
            if popped.is_empty() {
                thread::sleep(Duration::from_millis(2));
                continue;
            }
            stereo_mode.extract_slice(popped, &mut mono_chunk);
            let pre = if enable_hpss {
                hpss.process_slice(&mono_chunk, &mut hpss_chunk);
                &hpss_chunk[..]
            } else {
                &mono_chunk[..]
            };
            filter.process_slice(pre, &mut filtered_chunk);

            for &sample in &filtered_chunk {
                hop_buf[hop_filled] = sample;
                hop_filled += 1;
                if hop_filled == YIN_HOP_SIZE {
                    window.copy_within(YIN_HOP_SIZE.., 0);
                    window[YIN_WINDOW_SIZE - YIN_HOP_SIZE..].copy_from_slice(&hop_buf);
                    hop_filled = 0;
                    if let Some(ev) = stabilizer.push(detector.detect(&window)) {
                        if phrase.len() == MAX_HISTORY {
                            phrase.remove(0);
                            warmths.remove(0);
                        }
                        phrase.push(ev.midi_note);
                        warmths.push(ev.timbral_warmth);
                        let tab = solve_tablature_with_config(
                            &phrase,
                            Some(&warmths),
                            &tuning_midi,
                            24,
                            &solver_cfg,
                        );
                        println!(
                            "[Nota] {:>3} ({:6.1} Hz | calor {:.0}%)",
                            midi_to_note_name(ev.midi_note),
                            ev.freq_hz,
                            ev.timbral_warmth * 100.0
                        );
                        println!("{}", render_ascii_tab(&tab, &string_labels));
                    }
                }
            }
        }
    } else {
        let mut engine = BasicPitchEngine::load(&model_path)?;
        let (tx, rx) = mpsc::sync_channel::<Vec<f32>>(2);
        let tuning_clone = tuning_midi.clone();
        let labels_clone = string_labels.clone();
        let (min_midi, max_midi) = focus.poly_midi_bounds();

        thread::spawn(move || {
            let mut cfg = DecodeConfig::default();
            cfg.min_midi = min_midi;
            cfg.max_midi = max_midi;
            let mut chord_history: Vec<Vec<u8>> = Vec::with_capacity(MAX_HISTORY);
            let mut warmth_history: Vec<f32> = Vec::with_capacity(MAX_HISTORY);

            while let Ok(window_22k) = rx.recv() {
                let t0 = Instant::now();
                if let Ok(res) = engine.transcribe_window(&window_22k, &cfg) {
                    let mut added = false;
                    for ch in res.chords.into_iter().filter(|c| c.frame >= 86) {
                        if chord_history.last() == Some(&ch.notes) {
                            continue;
                        }
                        if chord_history.len() == MAX_HISTORY {
                            chord_history.remove(0);
                            warmth_history.remove(0);
                        }
                        chord_history.push(ch.notes);
                        warmth_history.push(ch.timbral_warmth);
                        added = true;
                    }
                    if added {
                        let tab = solve_polyphonic_with_config(
                            &chord_history,
                            Some(&warmth_history),
                            &tuning_clone,
                            24,
                            &solver_cfg,
                        );
                        if let Some(last) = chord_history.last() {
                            let names: Vec<String> =
                                last.iter().map(|&m| midi_to_note_name(m)).collect();
                            println!(
                                "[Poly | {} ms] [{}]",
                                t0.elapsed().as_millis(),
                                names.join(", ")
                            );
                            println!("{}", render_ascii_chord_tab(&tab, &labels_clone));
                        }
                    }
                }
            }
        });

        let mut resampler = LinearResampler::new(info.sample_rate, MODEL_SAMPLE_RATE);
        let mut read_chunk = [[0.0f32; 2]; 1024];
        let mut mono_chunk = Vec::with_capacity(1024);
        let mut hpss_chunk = Vec::with_capacity(1280);
        let mut filtered_chunk = Vec::with_capacity(1280);
        let mut resampled_chunk = Vec::with_capacity(1024);
        let mut rolling_22k: VecDeque<f32> = VecDeque::with_capacity(MODEL_WINDOW_SAMPLES + 2048);
        let hop_22k = (MODEL_SAMPLE_RATE * 0.75) as usize;
        let mut samples_since_infer = 0usize;

        loop {
            let (popped, _) = consumer.pop_partial_slice(&mut read_chunk);
            if popped.is_empty() {
                thread::sleep(Duration::from_millis(4));
                continue;
            }
            stereo_mode.extract_slice(popped, &mut mono_chunk);
            let pre = if enable_hpss {
                hpss.process_slice(&mono_chunk, &mut hpss_chunk);
                &hpss_chunk[..]
            } else {
                &mono_chunk[..]
            };
            filter.process_slice(pre, &mut filtered_chunk);
            resampled_chunk.clear();
            resampler.push(&filtered_chunk, &mut resampled_chunk);

            for &s in &resampled_chunk {
                if rolling_22k.len() == MODEL_WINDOW_SAMPLES {
                    rolling_22k.pop_front();
                }
                rolling_22k.push_back(s);
                samples_since_infer += 1;
            }

            if rolling_22k.len() == MODEL_WINDOW_SAMPLES && samples_since_infer >= hop_22k {
                samples_since_infer = 0;
                let energy: f32 =
                    rolling_22k.iter().map(|&s| s * s).sum::<f32>() / MODEL_WINDOW_SAMPLES as f32;
                if energy.sqrt() > 0.008 {
                    let _ = tx.try_send(rolling_22k.iter().copied().collect());
                }
            }
        }
    }
}
