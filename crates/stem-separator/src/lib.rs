pub mod wav;

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use dsp_pitch::{
    compute_percussiveness, detect_instrument_profile, estimate_timbral_warmth, DcBlocker,
    InstrumentFilter, InstrumentFocus, NoteStabilizer, YinDetector,
};
use music_theory::{
    build_quantized_score, midi_to_note_name_for_key, score_to_alphatex, ClefKind, QuantizedScore,
    TimeSignature, TimedEvent,
};
use poly_transcriber::{
    BasicPitchEngine, DecodeConfig, SincResampler, FRAME_DURATION_MS, MODEL_SAMPLE_RATE,
};
use serde::{Deserialize, Serialize};
use tab_solver::{
    solve_polyphonic_with_config, solve_tablature_with_config, ChordVoicing, SolverConfig,
};

pub use wav::{read_wav_mono_f32, write_stereo_wav_pcm16, write_stereo_wav_pcm24, WavWriteReport};

const MAX_OFFLINE_EVENTS: usize = 256;

#[derive(Debug, Clone, Serialize)]
pub struct BackendStatus {
    pub ready: bool,
    pub engine: String,
    pub hint: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct StemActivity {
    pub stem: String,
    pub label: String,
    pub rms_db: f32,
    pub active: bool,
}

#[derive(Debug, Clone)]
pub struct StemBundle {
    pub session_dir: PathBuf,
    pub mix_wav: PathBuf,
    pub stems: BTreeMap<String, PathBuf>,
    pub activities: Vec<StemActivity>,
    pub engine: String,
    pub separation_ms: u128,
}

impl StemBundle {
    pub fn wav_for_stem(&self, stem: &str) -> Option<&PathBuf> {
        if stem == "mix" {
            return Some(&self.mix_wav);
        }
        self.stems.get(stem)
    }

    pub fn available_stems(&self) -> Vec<String> {
        let mut order = Vec::new();
        for preferred in ["guitar", "bass", "piano", "vocals", "other", "drums"] {
            if self.stems.contains_key(preferred) {
                order.push(preferred.to_string());
            }
        }
        order.push("mix".to_string());
        order
    }

    /// Retorna o stem melódico mais forte caso o escolhido esteja silencioso (< -42 dBFS).
    pub fn best_melodic_stem_or(&self, preferred: &str) -> String {
        if preferred == "mix" {
            return "mix".to_string();
        }
        if let Some(act) = self.activities.iter().find(|a| a.stem == preferred) {
            if act.active {
                return preferred.to_string();
            }
        }
        self.activities
            .iter()
            .filter(|a| a.stem != "drums" && a.stem != "mix" && a.active)
            .max_by(|a, b| a.rms_db.total_cmp(&b.rms_db))
            .map(|a| a.stem.clone())
            .unwrap_or_else(|| preferred.to_string())
    }
}

#[derive(Debug, Deserialize)]
struct CheckJson {
    ready: bool,
    engine: String,
}

#[derive(Debug, Deserialize)]
struct SeparateJson {
    ok: bool,
    engine: Option<String>,
    stems: Option<BTreeMap<String, String>>,
    error: Option<String>,
}

fn resolve_python_and_script(workspace_root: &Path) -> Result<(PathBuf, PathBuf), String> {
    let script = workspace_root.join("scripts/separate_stems.py");
    if !script.exists() {
        return Err(format!(
            "Script de separação não encontrado em '{}'.",
            script.display()
        ));
    }

    let venv_py = workspace_root.join(".venv-stems/bin/python");
    let python = if venv_py.exists() {
        venv_py
    } else {
        PathBuf::from("python3")
    };

    Ok((python, script))
}

pub fn check_stems_backend(workspace_root: &Path) -> BackendStatus {
    let Ok((python, script)) = resolve_python_and_script(workspace_root) else {
        return BackendStatus {
            ready: false,
            engine: "indisponível".to_string(),
            hint: "Script scripts/separate_stems.py ausente.".to_string(),
        };
    };

    let output = Command::new(&python)
        .arg(&script)
        .arg("--input")
        .arg("_")
        .arg("--out-dir")
        .arg("_")
        .arg("--check")
        .output();

    if let Ok(out) = output {
        let stdout = String::from_utf8_lossy(&out.stdout);
        if let Some(line) = stdout.lines().rev().find(|l| l.trim().starts_with('{')) {
            if let Ok(parsed) = serde_json::from_str::<CheckJson>(line) {
                if parsed.ready {
                    return BackendStatus {
                        ready: true,
                        engine: parsed.engine,
                        hint: "Pronto para separar 6 stems localmente".to_string(),
                    };
                }
            }
        }
    }

    BackendStatus {
        ready: false,
        engine: "não instalado".to_string(),
        hint: "Instale demucs-mlx em ~/tab-transcriber/.venv-stems".to_string(),
    }
}

fn analyze_stem_activity(stem: &str, path: &Path) -> StemActivity {
    let base_label = match stem {
        "guitar" => "Guitarra",
        "bass" => "Baixo",
        "piano" => "Piano",
        "vocals" => "Voz",
        "drums" => "Bateria",
        "other" => "Outros",
        _ => "Mix Original",
    };

    let Ok((sr, mono)) = read_wav_mono_f32(path) else {
        return StemActivity {
            stem: stem.to_string(),
            label: base_label.to_string(),
            rms_db: -120.0,
            active: false,
        };
    };

    if mono.is_empty() {
        return StemActivity {
            stem: stem.to_string(),
            label: base_label.to_string(),
            rms_db: -120.0,
            active: false,
        };
    }

    let sum_sq: f64 = mono.iter().map(|&s| (s as f64) * (s as f64)).sum();
    let rms = (sum_sq / mono.len() as f64).sqrt() as f32;
    let rms_db = 20.0 * rms.max(1e-7).log10();
    let active = rms_db > -40.0;

    // Classifica acusticamente o conteúdo de "other" (synth, cordas, metais, oud/dedilhados)
    let label = if stem == "other" && active {
        let win = 4096usize.min(mono.len());
        let start = mono.len().saturating_sub(win) / 2;
        let slice = &mono[start..start + win];
        let warmth = estimate_timbral_warmth(slice, sr as f32, 330.0);
        let peak = slice.iter().map(|&x| x.abs()).fold(0.0f32, f32::max);
        let seg_rms = (slice.iter().map(|&x| x * x).sum::<f32>() / slice.len() as f32).sqrt();
        let crest = if seg_rms > 1e-6 { peak / seg_rms } else { 1.0 };

        let sub = if crest > 5.5 {
            "Dedilhados / Oud / Perc."
        } else if warmth > 0.58 {
            "Cordas / Pad / Órgão"
        } else {
            "Synth / Metais / Sopros"
        };
        format!("Outros ({})", sub)
    } else {
        base_label.to_string()
    };

    StemActivity {
        stem: stem.to_string(),
        label,
        rms_db,
        active,
    }
}

pub fn separate_with_htdemucs_6s(
    mix_wav: &Path,
    out_dir: &Path,
    workspace_root: &Path,
    quality: &str,
) -> Result<StemBundle, String> {
    let (python, script) = resolve_python_and_script(workspace_root)?;
    let t0 = Instant::now();
    let q_flag = if quality == "fast" { "fast" } else { "high" };

    let output = Command::new(&python)
        .arg(&script)
        .arg("--input")
        .arg(mix_wav)
        .arg("--out-dir")
        .arg(out_dir)
        .arg("--model")
        .arg("htdemucs_6s")
        .arg("--quality")
        .arg(q_flag)
        .output()
        .map_err(|e| format!("Falha ao iniciar Python ({}): {}", python.display(), e))?;

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    let json_line = stdout
        .lines()
        .rev()
        .find(|l| l.trim().starts_with('{'))
        .ok_or_else(|| {
            format!(
                "Saída inesperada do HTDemucs.\nSTDERR: {}\nSTDOUT: {}",
                stderr.trim(),
                stdout.trim()
            )
        })?;

    let parsed: SeparateJson = serde_json::from_str(json_line)
        .map_err(|e| format!("Erro ao ler resposta JSON do HTDemucs: {} ({})", e, json_line))?;

    if !parsed.ok {
        return Err(parsed
            .error
            .unwrap_or_else(|| format!("HTDemucs falhou: {}", stderr.trim())));
    }

    let raw_stems = parsed.stems.unwrap_or_default();
    let mut stems = BTreeMap::new();
    for (k, v) in raw_stems {
        stems.insert(k, PathBuf::from(v));
    }

    let mut activities = Vec::new();
    for key in ["guitar", "bass", "piano", "vocals", "other", "drums"] {
        if let Some(p) = stems.get(key) {
            activities.push(analyze_stem_activity(key, p));
        }
    }

    Ok(StemBundle {
        session_dir: out_dir.to_path_buf(),
        mix_wav: mix_wav.to_path_buf(),
        stems,
        activities,
        engine: parsed.engine.unwrap_or_else(|| "htdemucs_6s".to_string()),
        separation_ms: t0.elapsed().as_millis(),
    })
}

pub struct OfflineTranscribeParams {
    pub stem_name: String,
    pub mode: String,
    pub tuning_midi: Vec<u8>,
    pub tuning_label: String,
    pub alphatex_tuning: String,
    pub midi_program: u8,
    pub show_tab_staff: bool,
    pub clef: ClefKind,
    pub allow_bends: bool,
    pub solver_cfg: SolverConfig,
    pub time_signature: TimeSignature,
    pub bpm_override: Option<u16>,
    pub focus: InstrumentFocus,
    pub model_path: Option<PathBuf>,
}

pub struct OfflineTranscribeOutput {
    pub score: QuantizedScore,
    pub alphatex: String,
    pub latest_notes: Vec<String>,
    pub latest_chord: Option<String>,
    pub detected_instrument: String,
    pub event_count: usize,
    pub estimated_a4_hz: f32,
    pub pruned_overtones: usize,
    pub max_bend: f32,
    pub elapsed_ms: u128,
}

fn stem_display_name(stem: &str) -> &'static str {
    match stem {
        "guitar" => "Stem Guitarra",
        "bass" => "Stem Baixo",
        "piano" => "Stem Piano",
        "vocals" => "Stem Voz",
        "drums" => "Stem Bateria",
        "other" => "Stem Outros",
        _ => "Mix Original",
    }
}

fn stem_midi_bounds(stem: &str, focus: InstrumentFocus) -> (u8, u8) {
    match stem {
        "bass" => (23, 64),
        "vocals" => (45, 84),
        "piano" => (21, 108),
        "guitar" => {
            if focus == InstrumentFocus::GuitarSolo {
                (55, 93)
            } else {
                (40, 88)
            }
        }
        _ => focus.poly_midi_bounds(),
    }
}

fn stem_yin_bounds(stem: &str, focus: InstrumentFocus) -> (f32, f32) {
    match stem {
        "bass" => (29.0, 380.0),
        "vocals" => (95.0, 1300.0),
        "piano" => (40.0, 3200.0),
        "guitar" if focus == InstrumentFocus::GuitarSolo => (196.0, 2400.0),
        "guitar" => (75.0, 1800.0),
        _ => focus.yin_freq_bounds(),
    }
}

fn condition_stem_audio(samples: &mut [f32], sample_rate: u32) {
    let mut dc = DcBlocker::new(sample_rate as f32);
    dc.process_in_place(samples);

    let peak = samples.iter().map(|&s| s.abs()).fold(0.0f32, f32::max);
    if peak > 0.004 {
        let target_peak = 0.8414f32; // -1.5 dBFS
        let gain = (target_peak / peak).min(10.0);
        for s in samples.iter_mut() {
            *s *= gain;
        }
    }
}

/// Transcreve integralmente um stem isolado, reutilizando um BasicPitchEngine em cache se fornecido.
pub fn transcribe_stem_offline_with_engine(
    wav_path: &Path,
    params: &OfflineTranscribeParams,
    cached_engine: Option<&mut BasicPitchEngine>,
) -> Result<OfflineTranscribeOutput, String> {
    let t0 = Instant::now();
    let (sample_rate, mut audio_mono) = read_wav_mono_f32(wav_path)?;

    if audio_mono.len() < 2048 {
        return Err("Stem gravado é curto demais para transcrição.".to_string());
    }

    condition_stem_audio(&mut audio_mono, sample_rate);

    if matches!(
        params.focus,
        InstrumentFocus::GuitarSolo
            | InstrumentFocus::Bass
            | InstrumentFocus::OudLute
            | InstrumentFocus::BowedStrings
            | InstrumentFocus::WindsBrass
    ) {
        let mut filter = InstrumentFilter::new(sample_rate as f32, params.focus);
        let mut filtered = Vec::with_capacity(audio_mono.len());
        filter.process_slice(&audio_mono, &mut filtered);
        audio_mono = filtered;
    }

    let mut timed_history: Vec<TimedEvent> = Vec::new();
    let mut warmth_history: Vec<f32> = Vec::new();
    let mut estimated_a4_hz: f32 = 440.0;
    let mut total_pruned: usize = 0;
    let mut max_bend: f32 = 0.0;

    if params.mode == "mono" {
        const WIN: usize = 2048;
        const HOP: usize = 512;

        let mut detector = YinDetector::new(sample_rate as f32, WIN);
        let (min_hz, max_hz) = stem_yin_bounds(&params.stem_name, params.focus);
        detector.set_freq_range(min_hz, max_hz);

        let mut stabilizer = NoteStabilizer::new(3, 6);
        let mut a4_samples: Vec<f32> = Vec::new();
        let mut pos = 0usize;

        while pos + WIN <= audio_mono.len() {
            if let Some(ev) = stabilizer.push(detector.detect(&audio_mono[pos..pos + WIN])) {
                let start_ms = ((pos as f64 / sample_rate as f64) * 1000.0).round() as u64;
                if ev.cents.abs() <= 45.0 {
                    a4_samples.push(440.0 * 2.0f32.powf(ev.cents / 1200.0));
                }
                if let Some(prev) = timed_history.last_mut() {
                    prev.duration_ms = start_ms.saturating_sub(prev.start_ms).clamp(80, 2000);
                }
                timed_history.push(TimedEvent {
                    start_ms,
                    duration_ms: 250,
                    notes: vec![ev.midi_note],
                    bend_semitones: 0.0,
                });
                warmth_history.push(ev.timbral_warmth);
                if timed_history.len() >= MAX_OFFLINE_EVENTS {
                    break;
                }
            }
            pos += HOP;
        }
        if !a4_samples.is_empty() {
            a4_samples.sort_by(|a, b| a.total_cmp(b));
            estimated_a4_hz = a4_samples[a4_samples.len() / 2];
        }
    } else {
        let mut owned_engine: Option<BasicPitchEngine> = None;
        let engine: &mut BasicPitchEngine = match cached_engine {
            Some(eng) => eng,
            None => {
                let model_path = params
                    .model_path
                    .as_ref()
                    .ok_or_else(|| "Caminho do modelo nmp.onnx não informado.".to_string())?;
                owned_engine = Some(
                    BasicPitchEngine::load(model_path)
                        .map_err(|e| format!("Falha ao carregar Basic Pitch ONNX: {}", e))?,
                );
                owned_engine.as_mut().unwrap()
            }
        };

        let mut resampler = SincResampler::new(sample_rate as f64, MODEL_SAMPLE_RATE);
        let mut audio_22k = Vec::with_capacity(
            ((audio_mono.len() as f64) * (MODEL_SAMPLE_RATE / sample_rate as f64)) as usize + 256,
        );
        resampler.push(&audio_mono, &mut audio_22k);
        resampler.flush(&mut audio_22k);

        let mut cfg = DecodeConfig::default();
        let (min_m, max_m) = stem_midi_bounds(&params.stem_name, params.focus);
        cfg.min_midi = min_m;
        cfg.max_midi = max_m;
        cfg.max_polyphony = match params.stem_name.as_str() {
            "vocals" | "bass" => 1,
            "piano" => 6,
            _ if matches!(params.focus, InstrumentFocus::Vocals | InstrumentFocus::Bass | InstrumentFocus::WindsBrass) => 1,
            _ => 6,
        };

        // Ajusta limiar e poda de oitavas conforme família do instrumento
        if params.stem_name == "piano" || params.focus.preserve_wide_octaves() {
            cfg.enable_overtone_pruning = false;
            cfg.onset_threshold = 0.42;
        if !params.allow_bends { cfg.bend_min_cents = 999.0; }
            cfg.frame_threshold = 0.26;
        } else if matches!(params.focus, InstrumentFocus::BowedStrings | InstrumentFocus::Vocals)
            || params.stem_name == "vocals"
        {
            // Ataques suaves (arco / voz) precisam de limiar de onset menor
            cfg.onset_threshold = 0.36;
        if !params.allow_bends { cfg.bend_min_cents = 999.0; }
            cfg.frame_threshold = 0.25;
        } else {
            cfg.onset_threshold = 0.44;
        if !params.allow_bends { cfg.bend_min_cents = 999.0; }
            cfg.frame_threshold = 0.27;
        }

        let res = engine
            .transcribe_audio_full(&audio_22k, &cfg)
            .map_err(|e| format!("Erro na inferência Basic Pitch: {}", e))?;

        total_pruned = res.pruned_overtones;
        if let Some(a4) = res.estimated_a4_hz {
            estimated_a4_hz = a4;
        }

        for ch in res.chords.into_iter().take(MAX_OFFLINE_EVENTS) {
            let start_ms = ((ch.frame as f64) * FRAME_DURATION_MS).round() as u64;
            let duration_ms =
                (((ch.duration_frames as f64) * FRAME_DURATION_MS).round() as u64).max(90);
            max_bend = max_bend.max(ch.max_bend_semitones);
            timed_history.push(TimedEvent {
                start_ms,
                duration_ms,
                notes: ch.notes,
                bend_semitones: ch.max_bend_semitones,
            });
            warmth_history.push(ch.timbral_warmth);
        }
    }

    if timed_history.is_empty() {
        return Err(format!(
            "Nenhuma nota detectada em {}. Tente outro stem ativo.",
            stem_display_name(&params.stem_name)
        ));
    }

    let first_ms = timed_history[0].start_ms;
    for ev in &mut timed_history {
        ev.start_ms = ev.start_ms.saturating_sub(first_ms);
    }

    // Classifica o instrumento detectado no trecho
    let mut all_midi: Vec<u8> = timed_history
        .iter()
        .flat_map(|e| e.notes.iter().copied())
        .collect();
    all_midi.sort_unstable();
    let median_midi = all_midi.get(all_midi.len() / 2).copied().unwrap_or(60) as f32;
    let mean_poly = (all_midi.len() as f32) / (timed_history.len().max(1) as f32);
    let mean_warmth = if warmth_history.is_empty() {
        0.5
    } else {
        warmth_history.iter().sum::<f32>() / (warmth_history.len() as f32)
    };
    let perc = compute_percussiveness(&audio_mono);
    let detected_instrument = if params.stem_name == "other" || params.stem_name == "mix" {
        detect_instrument_profile(median_midi, mean_poly, mean_warmth, perc)
            .label
            .to_string()
    } else {
        stem_display_name(&params.stem_name).to_string()
    };

    let voicings: Vec<Option<ChordVoicing>> = if !params.show_tab_staff {
        vec![None; timed_history.len()]
    } else if params.mode == "mono" {
        let notes: Vec<u8> = timed_history
            .iter()
            .filter_map(|e| e.notes.first().copied())
            .collect();
        solve_tablature_with_config(
            &notes,
            Some(&warmth_history),
            &params.tuning_midi,
            24,
            &params.solver_cfg,
        )
        .into_iter()
        .map(|opt| opt.map(|p| vec![p]))
        .collect()
    } else {
        let chords: Vec<Vec<u8>> = timed_history.iter().map(|e| e.notes.clone()).collect();
        solve_polyphonic_with_config(
            &chords,
            Some(&warmth_history),
            &params.tuning_midi,
            24,
            &params.solver_cfg,
        )
    };

    let q_score = build_quantized_score(
        "Tab Transcriber",
        &timed_history,
        &voicings,
        &params.tuning_midi,
        params.bpm_override,
        params.time_signature,
    )
    .with_instrument(params.midi_program, params.show_tab_staff, params.clef);

    let key_obj = q_score.key;
    let subtitle = format!(
        "{} · {} — {} · {} · {} BPM",
        detected_instrument,
        params.tuning_label,
        key_obj.display_name(),
        params.time_signature.display_label(),
        q_score.bpm
    );
    let alphatex = score_to_alphatex(&q_score, &params.alphatex_tuning, &subtitle);

    let latest_notes = timed_history
        .last()
        .map(|e| {
            e.notes
                .iter()
                .map(|&m| midi_to_note_name_for_key(m, key_obj))
                .collect()
        })
        .unwrap_or_default();

    let latest_chord = q_score
        .measures
        .iter()
        .rev()
        .flat_map(|m| m.beats.iter().rev())
        .find_map(|b| b.chord.as_ref().map(|c| c.label.clone()));

    Ok(OfflineTranscribeOutput {
        score: q_score,
        alphatex,
        latest_notes,
        latest_chord,
        detected_instrument,
        event_count: timed_history.len(),
        estimated_a4_hz,
        pruned_overtones: total_pruned,
        max_bend,
        elapsed_ms: t0.elapsed().as_millis(),
    })
}

pub fn transcribe_stem_offline(
    wav_path: &Path,
    params: &OfflineTranscribeParams,
) -> Result<OfflineTranscribeOutput, String> {
    transcribe_stem_offline_with_engine(wav_path, params, None)
}
