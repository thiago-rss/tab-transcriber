mod alphatex;

use std::collections::{BTreeMap, VecDeque};
use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use audio_capture::{list_audio_processes, AudioCapturePort, CaptureTarget, MacOSLoopbackCapture};
use dsp_pitch::{
    compute_percussiveness, compute_phase_correlation, detect_instrument_profile, DcBlocker,
    HpssSeparator, InstrumentFilter, InstrumentFocus, NoteStabilizer, SignalLeveler, StereoMode,
    YinDetector,
};
use music_theory::{
    build_quantized_score, export_midi, export_musicxml, midi_to_note_name_for_key,
    score_to_alphatex, QuantizedScore, TimeSignature, TimedEvent,
};
use poly_transcriber::{
    BasicPitchEngine, DecodeConfig, SincResampler, FRAME_DURATION_MS, MODEL_SAMPLE_RATE,
    MODEL_WINDOW_SAMPLES,
};
use serde::{Deserialize, Serialize};
use stem_separator::{
    check_stems_backend as detect_stems_backend, separate_with_htdemucs_6s,
    transcribe_stem_offline_with_engine, write_stereo_wav_pcm24, BackendStatus,
    OfflineTranscribeParams, StemActivity, StemBundle,
};
use tab_solver::{
    solve_polyphonic_with_config, solve_tablature_with_config, ChordVoicing, NeckRegion,
    SolverConfig,
};
use tauri::ipc::Response;
use tauri::{AppHandle, Emitter, State};

use alphatex::{empty_score_tex, get_tuning_preset};

const RING_CAPACITY_FRAMES: usize = 192_000;
const YIN_WINDOW_SIZE: usize = 2048;
const YIN_HOP_SIZE: usize = 512;
const MAX_SCORE_EVENTS: usize = 64;
const MAX_SNIPPET_SECONDS: usize = 90;

fn default_focus() -> String {
    "full".to_string()
}
fn default_time_signature() -> String {
    "4/4".to_string()
}
fn default_stereo_mode() -> String {
    "mix".to_string()
}
fn default_neck_region() -> String {
    "auto".to_string()
}
fn default_stem() -> String {
    "guitar".to_string()
}
fn default_stem_quality() -> String {
    "high".to_string()
}
fn default_true() -> bool {
    true
}

#[derive(Debug, Clone, Serialize)]
pub struct AudioSourceDto {
    pub id: String,
    pub name: String,
    pub bundle_id: String,
    pub pids: Vec<i32>,
    pub object_ids: Vec<u32>,
    pub is_running_output: bool,
}

#[derive(Debug, Clone, Deserialize)]
pub struct StartConfig {
    pub mode: String,
    pub tuning: String,
    pub target_ids: Vec<u32>,
    #[serde(default)]
    pub source_id: Option<String>,
    pub bpm: Option<u16>,
    #[serde(default = "default_focus")]
    pub focus: String,
    #[serde(default = "default_time_signature")]
    pub time_signature: String,
    #[serde(default = "default_stereo_mode")]
    pub stereo_mode: String,
    #[serde(default)]
    pub enable_hpss: bool,
    #[serde(default = "default_neck_region")]
    pub neck_region: String,
    #[serde(default)]
    pub prefer_closed: bool,
    #[serde(default = "default_true")]
    pub auto_level: bool,
}

#[derive(Debug, Clone, Deserialize)]
pub struct StemTranscribeConfig {
    #[serde(default = "default_stem")]
    pub stem: String,
    pub mode: String,
    pub tuning: String,
    pub bpm: Option<u16>,
    #[serde(default = "default_focus")]
    pub focus: String,
    #[serde(default = "default_time_signature")]
    pub time_signature: String,
    #[serde(default = "default_neck_region")]
    pub neck_region: String,
    #[serde(default)]
    pub prefer_closed: bool,
    #[serde(default = "default_stem_quality")]
    pub quality: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct StemSeparationResultDto {
    pub engine: String,
    pub available_stems: Vec<String>,
    pub stem_activities: Vec<StemActivity>,
    pub active_stem: String,
    pub separation_ms: u128,
    pub transcription_ms: u128,
    pub event_count: usize,
    pub session_dir: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct SnippetProgressPayload {
    pub elapsed_sec: f64,
    pub max_sec: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct TranscriptionUpdatePayload {
    pub alphatex: String,
    pub latest_notes: Vec<String>,
    pub latest_chord: Option<String>,
    pub detected_key: String,
    pub detected_instrument: String,
    pub time_signature: String,
    pub event_count: usize,
    pub latency_ms: u128,
    pub detected_bpm: u16,
    pub estimated_a4_hz: f32,
    pub pruned_overtones: usize,
    pub bend_semitones: f32,
    pub is_reattack: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct AudioLevelPayload {
    pub rms_db: f32,
    pub peak_db: f32,
    pub silence_warning: bool,
    pub dropped_samples: u64,
    pub device_switches: u64,
    pub clip_events: u64,
    pub clock_glitches: u64,
    pub phase_corr: f32,
    pub sample_rate: u32,
    pub format_label: String,
}

struct RunningSession {
    stop_flag: Arc<AtomicBool>,
    clear_flag: Arc<AtomicBool>,
    worker: Option<JoinHandle<()>>,
}

struct SnippetRecorderSession {
    stop_flag: Arc<AtomicBool>,
    worker: Option<JoinHandle<Result<(u32, Vec<[f32; 2]>), String>>>,
}

struct ScoreSnapshot {
    events: Vec<TimedEvent>,
    voicings: Vec<Option<ChordVoicing>>,
    tuning_id: String,
    time_sig: TimeSignature,
    detected_instrument: String,
    score: QuantizedScore,
}

struct AppState {
    session: Arc<Mutex<Option<RunningSession>>>,
    snippet_recorder: Arc<Mutex<Option<SnippetRecorderSession>>>,
    engine_cache: Arc<Mutex<Option<BasicPitchEngine>>>,
    active_stems: Arc<Mutex<Option<StemBundle>>>,
    active_tuning: Mutex<String>,
    active_time_signature: Mutex<TimeSignature>,
    latest_snapshot: Arc<Mutex<Option<ScoreSnapshot>>>,
}

impl Default for AppState {
    fn default() -> Self {
        Self {
            session: Arc::new(Mutex::new(None)),
            snippet_recorder: Arc::new(Mutex::new(None)),
            engine_cache: Arc::new(Mutex::new(None)),
            active_stems: Arc::new(Mutex::new(None)),
            active_tuning: Mutex::new("standard".to_string()),
            active_time_signature: Mutex::new(TimeSignature::default()),
            latest_snapshot: Arc::new(Mutex::new(None)),
        }
    }
}

fn workspace_root_dir() -> PathBuf {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest_dir
        .join("../../..")
        .canonicalize()
        .unwrap_or_else(|_| manifest_dir.join("../../.."))
}

fn exports_dir() -> Result<PathBuf, String> {
    let home = std::env::var("HOME")
        .map_err(|_| "Variável HOME não encontrada para definir pasta de exportação.".to_string())?;
    let dir = PathBuf::from(home).join("Documents").join("TabTranscriber");
    fs::create_dir_all(&dir).map_err(|e| format!("Falha ao criar pasta de exportação: {}", e))?;
    Ok(dir)
}

fn resolve_model_path() -> Result<PathBuf, String> {
    let root = workspace_root_dir();
    let candidates = [
        root.join("models/nmp.onnx"),
        PathBuf::from("models/nmp.onnx"),
        PathBuf::from("../../models/nmp.onnx"),
        PathBuf::from("../../../models/nmp.onnx"),
    ];

    for path in candidates {
        if path.exists() {
            return Ok(path);
        }
    }

    Err(
        "Modelo ONNX 'models/nmp.onnx' não encontrado. Confirme que ele foi baixado em ~/tab-transcriber/models/nmp.onnx."
            .to_string(),
    )
}

/// Carrega o BasicPitchEngine uma única vez e mantém em cache no AppState.
fn ensure_engine_loaded(cache: &Arc<Mutex<Option<BasicPitchEngine>>>) -> Result<(), String> {
    let mut guard = cache.lock().unwrap();
    if guard.is_none() {
        let path = resolve_model_path()?;
        let eng = BasicPitchEngine::load(&path)
            .map_err(|e| format!("Falha ao carregar ONNX Runtime: {}", e))?;
        *guard = Some(eng);
    }
    Ok(())
}

/// Para qualquer captura ativa (Ao Vivo ou Gravação de Trecho) fora do lock e aguarda liberação do HAL.
fn teardown_all_captures_blocking(
    session_arc: &Arc<Mutex<Option<RunningSession>>>,
    recorder_arc: &Arc<Mutex<Option<SnippetRecorderSession>>>,
) {
    let mut stopped_any = false;

    if let Some(mut sess) = session_arc.lock().unwrap().take() {
        sess.stop_flag.store(true, Ordering::SeqCst);
        if let Some(handle) = sess.worker.take() {
            let _ = handle.join();
        }
        stopped_any = true;
    }

    if let Some(mut rec) = recorder_arc.lock().unwrap().take() {
        rec.stop_flag.store(true, Ordering::SeqCst);
        if let Some(handle) = rec.worker.take() {
            let _ = handle.join();
        }
        stopped_any = true;
    }

    if stopped_any {
        thread::sleep(Duration::from_millis(60));
    }
}

/// Re-resolve os ObjectIDs atuais no Core Audio a partir do bundle_id (evita IDs vencidos na troca de modo).
fn resolve_fresh_capture_target(source_id: Option<&str>, fallback_ids: &[u32]) -> CaptureTarget {
    if let Some(id) = source_id {
        if id.is_empty() || id == "__global__" {
            return CaptureTarget::SystemGlobal;
        }
        let fresh: Vec<u32> = list_audio_processes(false)
            .into_iter()
            .filter(|p| {
                let key = if !p.bundle_id.is_empty() {
                    p.bundle_id.as_str()
                } else {
                    p.name.as_str()
                };
                key.eq_ignore_ascii_case(id)
            })
            .map(|p| p.object_id)
            .collect();

        if !fresh.is_empty() {
            return CaptureTarget::ProcessObjects(fresh);
        }
    }

    if fallback_ids.is_empty() {
        CaptureTarget::SystemGlobal
    } else {
        CaptureTarget::ProcessObjects(fallback_ids.to_vec())
    }
}

fn latest_chord_label(score: &QuantizedScore) -> Option<String> {
    score
        .measures
        .iter()
        .rev()
        .flat_map(|m| m.beats.iter().rev())
        .find_map(|b| b.chord.as_ref().map(|c| c.label.clone()))
}

fn resolve_auto_stem_preset_id(stem: &str) -> &'static str {
    match stem {
        "bass" => "bass4",
        "piano" => "piano",
        "vocals" => "vocals",
        "other" => "strings",
        _ => "standard",
    }
}

fn build_offline_params(
    cfg: &StemTranscribeConfig,
    model_path: Option<PathBuf>,
) -> OfflineTranscribeParams {
    let eff_tuning = if cfg.tuning == "auto" {
        resolve_auto_stem_preset_id(&cfg.stem)
    } else {
        cfg.tuning.as_str()
    };
    let preset = get_tuning_preset(eff_tuning);
    let time_sig = TimeSignature::from_str_id(&cfg.time_signature);
    let focus = InstrumentFocus::from_str_id(&cfg.focus);
    let region = NeckRegion::from_str_id(&cfg.neck_region);

    OfflineTranscribeParams {
        stem_name: cfg.stem.clone(),
        mode: cfg.mode.clone(),
        tuning_midi: preset.midi_low_to_high,
        tuning_label: preset.label.to_string(),
        alphatex_tuning: preset.alphatex_high_to_low.to_string(),
        midi_program: preset.midi_program,
        show_tab_staff: preset.show_tab_staff,
        clef: preset.clef,
        allow_bends: preset.allow_bends,
        solver_cfg: SolverConfig {
            region,
            prefer_closed_positions: cfg.prefer_closed,
            ..SolverConfig::default()
        },
        time_signature: time_sig,
        bpm_override: cfg.bpm,
        focus,
        model_path,
    }
}

#[tauri::command]
fn check_stems_backend() -> BackendStatus {
    detect_stems_backend(&workspace_root_dir())
}

#[tauri::command]
fn load_stem_audio(state: State<'_, AppState>, stem: String) -> Result<Response, String> {
    let wav_path = {
        let guard = state.active_stems.lock().unwrap();
        let bundle = guard
            .as_ref()
            .ok_or_else(|| "Nenhum trecho separado ainda.".to_string())?;
        bundle
            .wav_for_stem(&stem)
            .cloned()
            .ok_or_else(|| format!("Áudio do stem '{}' não encontrado.", stem))?
    };

    let bytes =
        fs::read(&wav_path).map_err(|e| format!("Falha ao ler '{}': {}", wav_path.display(), e))?;
    Ok(Response::new(bytes))
}

#[tauri::command]
fn list_audio_sources(only_active: bool) -> Vec<AudioSourceDto> {
    let raw = list_audio_processes(only_active);
    let mut grouped: BTreeMap<String, AudioSourceDto> = BTreeMap::new();

    for p in raw {
        let key = if !p.bundle_id.is_empty() {
            p.bundle_id.clone()
        } else {
            p.name.to_lowercase()
        };

        grouped
            .entry(key.clone())
            .and_modify(|dto| {
                if !dto.pids.contains(&p.pid) {
                    dto.pids.push(p.pid);
                }
                if !dto.object_ids.contains(&p.object_id) {
                    dto.object_ids.push(p.object_id);
                }
                dto.is_running_output |= p.is_running_output;
            })
            .or_insert_with(|| AudioSourceDto {
                id: key,
                name: p.name,
                bundle_id: p.bundle_id,
                pids: vec![p.pid],
                object_ids: vec![p.object_id],
                is_running_output: p.is_running_output,
            });
    }

    let mut list: Vec<AudioSourceDto> = grouped.into_values().collect();
    list.sort_by(|a, b| {
        b.is_running_output
            .cmp(&a.is_running_output)
            .then_with(|| a.name.to_lowercase().cmp(&b.name.to_lowercase()))
    });
    list
}

#[tauri::command]
fn get_initial_tex(tuning: String, time_signature: Option<String>) -> String {
    let preset = get_tuning_preset(&tuning);
    let ts = time_signature
        .as_deref()
        .map(TimeSignature::from_str_id)
        .unwrap_or_default();
    empty_score_tex(&preset, ts)
}

#[tauri::command]
async fn stop_transcription(state: State<'_, AppState>) -> Result<(), String> {
    let session_arc = Arc::clone(&state.session);
    let recorder_arc = Arc::clone(&state.snippet_recorder);
    tauri::async_runtime::spawn_blocking(move || {
        teardown_all_captures_blocking(&session_arc, &recorder_arc);
    })
    .await
    .map_err(|e| format!("Falha ao encerrar captura: {}", e))?;
    Ok(())
}

#[tauri::command]
async fn cancel_snippet_recording(state: State<'_, AppState>) -> Result<(), String> {
    let session_arc = Arc::clone(&state.session);
    let recorder_arc = Arc::clone(&state.snippet_recorder);
    tauri::async_runtime::spawn_blocking(move || {
        teardown_all_captures_blocking(&session_arc, &recorder_arc);
    })
    .await
    .map_err(|e| format!("Falha ao cancelar gravação: {}", e))?;
    Ok(())
}

#[tauri::command]
fn clear_transcription(app: AppHandle, state: State<'_, AppState>) -> Result<(), String> {
    let tuning = state.active_tuning.lock().unwrap().clone();
    let ts = *state.active_time_signature.lock().unwrap();
    let preset = get_tuning_preset(&tuning);

    if let Some(sess) = state.session.lock().unwrap().as_ref() {
        sess.clear_flag.store(true, Ordering::SeqCst);
    }
    *state.latest_snapshot.lock().unwrap() = None;

    let _ = app.emit(
        "transcription-update",
        TranscriptionUpdatePayload {
            alphatex: empty_score_tex(&preset, ts),
            latest_notes: vec![],
            latest_chord: None,
            detected_key: "—".to_string(),
            detected_instrument: "—".to_string(),
            time_signature: ts.display_label(),
            event_count: 0,
            latency_ms: 0,
            detected_bpm: 100,
            estimated_a4_hz: 440.0,
            pruned_overtones: 0,
            bend_semitones: 0.0,
            is_reattack: false,
        },
    );
    Ok(())
}

/// Re-quantiza imediatamente a partitura atual em ½× ou 2× BPM sem regravar.
#[tauri::command]
fn rescale_score_bpm(
    app: AppHandle,
    state: State<'_, AppState>,
    factor: f32,
) -> Result<u16, String> {
    let mut guard = state.latest_snapshot.lock().unwrap();
    let snap = guard
        .as_mut()
        .ok_or_else(|| "Nenhuma transcrição ativa para ajustar o BPM.".to_string())?;

    let new_bpm = ((snap.score.bpm as f32) * factor).round().clamp(45.0, 220.0) as u16;
    let preset = get_tuning_preset(&snap.tuning_id);

    let q_score = build_quantized_score(
        "Tab Transcriber",
        &snap.events,
        &snap.voicings,
        &preset.midi_low_to_high,
        Some(new_bpm),
        snap.time_sig,
    )
    .with_instrument(preset.midi_program, preset.show_tab_staff, preset.clef);

    let detected_key = q_score.key.display_name();
    let latest_chord = latest_chord_label(&q_score);
    let key_obj = q_score.key;
    let subtitle = format!(
        "{} · {} — {} · {} · {} BPM",
        snap.detected_instrument,
        preset.label,
        detected_key,
        snap.time_sig.display_label(),
        new_bpm
    );
    let alphatex = score_to_alphatex(&q_score, preset.alphatex_high_to_low, &subtitle);

    let latest_notes = snap
        .events
        .last()
        .map(|e| {
            e.notes
                .iter()
                .map(|&m| midi_to_note_name_for_key(m, key_obj))
                .collect()
        })
        .unwrap_or_default();

    snap.score = q_score;

    let _ = app.emit(
        "transcription-update",
        TranscriptionUpdatePayload {
            alphatex,
            latest_notes,
            latest_chord,
            detected_key,
            detected_instrument: snap.detected_instrument.clone(),
            time_signature: snap.time_sig.display_label(),
            event_count: snap.events.len(),
            latency_ms: 0,
            detected_bpm: new_bpm,
            estimated_a4_hz: 440.0,
            pruned_overtones: 0,
            bend_semitones: 0.0,
            is_reattack: false,
        },
    );

    Ok(new_bpm)
}

#[tauri::command]
fn export_score(state: State<'_, AppState>, format: String) -> Result<String, String> {
    let score = state
        .latest_snapshot
        .lock()
        .unwrap()
        .as_ref()
        .map(|s| s.score.clone())
        .ok_or_else(|| "Nenhuma nota transcrita ainda para exportar.".to_string())?;

    let dir = exports_dir()?;
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);

    match format.to_lowercase().as_str() {
        "musicxml" | "xml" => {
            let path = dir.join(format!("transcricao_{}.musicxml", ts));
            let xml = export_musicxml(&score);
            fs::write(&path, xml).map_err(|e| format!("Erro ao gravar MusicXML: {}", e))?;
            Ok(path.display().to_string())
        }
        "midi" | "mid" => {
            let path = dir.join(format!("transcricao_{}.mid", ts));
            let bytes = export_midi(&score);
            fs::write(&path, bytes).map_err(|e| format!("Erro ao gravar MIDI: {}", e))?;
            Ok(path.display().to_string())
        }
        other => Err(format!("Formato não suportado: {}", other)),
    }
}

#[tauri::command]
fn open_exports_folder() -> Result<(), String> {
    let dir = exports_dir()?;
    #[cfg(target_os = "macos")]
    {
        Command::new("open")
            .arg(&dir)
            .spawn()
            .map_err(|e| format!("Não foi possível abrir o Finder: {}", e))?;
    }
    Ok(())
}

// ============================================================================
// CAPTURA DE TRECHO + HTDEMUCS 6-STEMS (ASSÍNCRONO, SEM TRAVAR TROCA DE MODO)
// ============================================================================

#[tauri::command]
async fn start_snippet_recording(
    app: AppHandle,
    state: State<'_, AppState>,
    target_ids: Vec<u32>,
    source_id: Option<String>,
) -> Result<String, String> {
    let session_arc = Arc::clone(&state.session);
    let recorder_arc = Arc::clone(&state.snippet_recorder);

    tauri::async_runtime::spawn_blocking(move || {
        teardown_all_captures_blocking(&session_arc, &recorder_arc);

        let target = resolve_fresh_capture_target(source_id.as_deref(), &target_ids);
        let stop_flag = Arc::new(AtomicBool::new(false));
        let stop_clone = Arc::clone(&stop_flag);
        let (ready_tx, ready_rx) = mpsc::channel::<Result<(u32, String), String>>();

        let worker = thread::spawn(move || -> Result<(u32, Vec<[f32; 2]>), String> {
            let (loopback, mut consumer) =
                match MacOSLoopbackCapture::start_with_target(RING_CAPACITY_FRAMES, target) {
                    Ok(pair) => pair,
                    Err(e) => {
                        let _ = ready_tx.send(Err(e.clone()));
                        return Err(e);
                    }
                };

            let sample_rate = loopback.current_sample_rate().round() as u32;
            let fmt_label = loopback.format_label();
            let _ = ready_tx.send(Ok((sample_rate, fmt_label)));

            let max_frames = (sample_rate as usize) * MAX_SNIPPET_SECONDS;
            let mut recorded: Vec<[f32; 2]> = Vec::with_capacity((sample_rate as usize) * 20);
            let mut read_chunk = [[0.0f32; 2]; 1024];
            let mut zero_chunks = 0u32;
            let mut last_ui_tick = Instant::now();

            while !stop_clone.load(Ordering::Relaxed) && recorded.len() < max_frames {
                let (popped, _) = consumer.pop_partial_slice(&mut read_chunk);
                if popped.is_empty() {
                    thread::sleep(Duration::from_millis(3));
                    continue;
                }

                let remaining = max_frames - recorded.len();
                let take = popped.len().min(remaining);
                recorded.extend_from_slice(&popped[..take]);

                let mut sum_sq = 0.0f64;
                let mut peak = 0.0f32;
                for fr in &popped[..take] {
                    let m = 0.5 * (fr[0] + fr[1]);
                    peak = peak.max(fr[0].abs()).max(fr[1].abs());
                    sum_sq += (m as f64) * (m as f64);
                }

                if peak == 0.0 {
                    zero_chunks = zero_chunks.saturating_add(1);
                } else {
                    zero_chunks = 0;
                }

                if last_ui_tick.elapsed() >= Duration::from_millis(90) {
                    last_ui_tick = Instant::now();
                    let rms = (sum_sq / take.max(1) as f64).sqrt() as f32;
                    let rms_db = 20.0 * rms.max(1e-7).log10();
                    let peak_db = 20.0 * peak.max(1e-7).log10();
                    let _ = app.emit(
                        "audio-level",
                        AudioLevelPayload {
                            rms_db,
                            peak_db,
                            silence_warning: zero_chunks >= 45,
                            dropped_samples: loopback.dropped_samples(),
                            device_switches: loopback.device_switches(),
                            clip_events: loopback.clip_events(),
                            clock_glitches: loopback.clock_glitches(),
                            phase_corr: compute_phase_correlation(&popped[..take]),
                            sample_rate: loopback.current_sample_rate().round() as u32,
                            format_label: loopback.format_label(),
                        },
                    );
                    let _ = app.emit(
                        "snippet-progress",
                        SnippetProgressPayload {
                            elapsed_sec: (recorded.len() as f64) / (sample_rate as f64),
                            max_sec: MAX_SNIPPET_SECONDS,
                        },
                    );
                }
            }

            let final_sr = loopback.current_sample_rate().round() as u32;
            Ok((final_sr, recorded))
        });

        match ready_rx.recv() {
            Ok(Ok((sr, fmt))) => {
                *recorder_arc.lock().unwrap() = Some(SnippetRecorderSession {
                    stop_flag,
                    worker: Some(worker),
                });
                Ok(format!("Gravando 24-bit ({} Hz · {})…", sr, fmt))
            }
            Ok(Err(err)) => {
                let _ = worker.join();
                Err(err)
            }
            Err(_) => {
                let _ = worker.join();
                Err("Gravador encerrou inesperadamente.".to_string())
            }
        }
    })
    .await
    .map_err(|e| format!("Falha ao iniciar gravação de trecho: {}", e))?
}

#[tauri::command]
async fn stop_and_separate_snippet(
    app: AppHandle,
    state: State<'_, AppState>,
    config: StemTranscribeConfig,
) -> Result<StemSeparationResultDto, String> {
    let rec_opt = { state.snippet_recorder.lock().unwrap().take() };
    let mut rec = rec_opt.ok_or_else(|| "Nenhuma gravação de trecho em andamento.".to_string())?;
    rec.stop_flag.store(true, Ordering::SeqCst);

    *state.active_tuning.lock().unwrap() = config.tuning.clone();
    *state.active_time_signature.lock().unwrap() = TimeSignature::from_str_id(&config.time_signature);

    let root_dir = workspace_root_dir();
    let base_exports = exports_dir()?;
    let active_stems_arc = Arc::clone(&state.active_stems);
    let latest_snap_arc = Arc::clone(&state.latest_snapshot);
    let engine_cache_arc = Arc::clone(&state.engine_cache);
    let app_clone = app.clone();

    tauri::async_runtime::spawn_blocking(move || {
        let handle = rec
            .worker
            .take()
            .ok_or_else(|| "Worker de gravação ausente.".to_string())?;
        let (sample_rate, frames) = handle
            .join()
            .map_err(|_| "Falha interna ao finalizar gravação.".to_string())??;

        let min_frames = (sample_rate as usize) / 2;
        if frames.len() < min_frames {
            return Err("Trecho muito curto (grave pelo menos 1 segundo de áudio).".to_string());
        }

        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let session_dir = base_exports.join("stems").join(format!("sess_{}", ts));
        fs::create_dir_all(&session_dir)
            .map_err(|e| format!("Falha ao criar pasta da sessão de stems: {}", e))?;

        let mix_wav = session_dir.join("mix.wav");
        let _ = app_clone.emit("stem-status", "Salvando mix.wav 24-bit estéreo…");
        let _report = write_stereo_wav_pcm24(&mix_wav, sample_rate, &frames)?;

        let _ = app_clone.emit(
            "stem-status",
            "Separando 6 stems com HTDemucs (guitar, bass, piano, vocals, other)…",
        );
        let bundle = separate_with_htdemucs_6s(&mix_wav, &session_dir, &root_dir, &config.quality)?;

        // Se o stem escolhido estiver silencioso, escolhe automaticamente o stem melódico ativo mais forte
        let target_stem = bundle.best_melodic_stem_or(&config.stem);
        let stem_wav = bundle
            .wav_for_stem(&target_stem)
            .cloned()
            .ok_or_else(|| format!("Stem '{}' não encontrado.", target_stem))?;

        let available_stems = bundle.available_stems();
        let stem_activities = bundle.activities.clone();
        let engine_name = bundle.engine.clone();
        let separation_ms = bundle.separation_ms;
        let session_dir_str = bundle.session_dir.display().to_string();

        *active_stems_arc.lock().unwrap() = Some(bundle);

        let _ = app_clone.emit(
            "stem-status",
            format!("Transcrevendo stem '{}' em modo offline…", target_stem),
        );

        let mut cfg_eff = config;
        cfg_eff.stem = target_stem.clone();
        let params = build_offline_params(&cfg_eff, None);

        let out = if cfg_eff.mode == "poly" {
            ensure_engine_loaded(&engine_cache_arc)?;
            let mut eng_guard = engine_cache_arc.lock().unwrap();
            transcribe_stem_offline_with_engine(&stem_wav, &params, eng_guard.as_mut())?
        } else {
            transcribe_stem_offline_with_engine(&stem_wav, &params, None)?
        };

        let detected_key = out.score.key.display_name();
        let detected_bpm = out.score.bpm;
        let ts_label = out.score.time_signature.display_label();
        let time_sig = out.score.time_signature;

        // Reconstrói lista simples de eventos/voicings para permitir ½× / 2× instantâneo
        let snap_events: Vec<TimedEvent> = out
            .score
            .measures
            .iter()
            .flat_map(|m| m.beats.iter())
            .filter(|b| !b.is_rest() && !b.tie_from_prev)
            .enumerate()
            .map(|(idx, b)| TimedEvent {
                start_ms: (idx as u64) * 250,
                duration_ms: ((b.duration_ticks as u64) * 60_000) / ((detected_bpm as u64) * 4),
                notes: b.midi_notes.clone(),
                bend_semitones: b.bend_semitones,
            })
            .collect();
        let snap_voicings: Vec<Option<ChordVoicing>> = out
            .score
            .measures
            .iter()
            .flat_map(|m| m.beats.iter())
            .filter(|b| !b.is_rest() && !b.tie_from_prev)
            .map(|b| b.voicing.clone())
            .collect();

        *latest_snap_arc.lock().unwrap() = Some(ScoreSnapshot {
            events: snap_events,
            voicings: snap_voicings,
            tuning_id: cfg_eff.tuning.clone(),
            time_sig,
            detected_instrument: out.detected_instrument.clone(),
            score: out.score,
        });

        let _ = app_clone.emit(
            "transcription-update",
            TranscriptionUpdatePayload {
                alphatex: out.alphatex,
                latest_notes: out.latest_notes,
                latest_chord: out.latest_chord,
                detected_key,
                detected_instrument: out.detected_instrument,
                time_signature: ts_label,
                event_count: out.event_count,
                latency_ms: out.elapsed_ms,
                detected_bpm,
                estimated_a4_hz: out.estimated_a4_hz,
                pruned_overtones: out.pruned_overtones,
                bend_semitones: out.max_bend,
                is_reattack: false,
            },
        );

        Ok(StemSeparationResultDto {
            engine: engine_name,
            available_stems,
            stem_activities,
            active_stem: target_stem,
            separation_ms,
            transcription_ms: out.elapsed_ms,
            event_count: out.event_count,
            session_dir: session_dir_str,
        })
    })
    .await
    .map_err(|e| format!("Falha ao executar tarefa de separação: {}", e))?
}

#[tauri::command]
async fn transcribe_existing_stem(
    app: AppHandle,
    state: State<'_, AppState>,
    config: StemTranscribeConfig,
) -> Result<String, String> {
    let bundle = {
        state
            .active_stems
            .lock()
            .unwrap()
            .clone()
            .ok_or_else(|| "Nenhum trecho separado ainda. Grave um trecho primeiro.".to_string())?
    };

    let stem_wav = bundle
        .wav_for_stem(&config.stem)
        .cloned()
        .ok_or_else(|| format!("Stem '{}' não disponível nesta sessão.", config.stem))?;

    *state.active_tuning.lock().unwrap() = config.tuning.clone();
    *state.active_time_signature.lock().unwrap() = TimeSignature::from_str_id(&config.time_signature);

    let latest_snap_arc = Arc::clone(&state.latest_snapshot);
    let engine_cache_arc = Arc::clone(&state.engine_cache);
    let app_clone = app.clone();
    let stem_label = config.stem.clone();

    tauri::async_runtime::spawn_blocking(move || {
        let _ = app_clone.emit("stem-status", format!("Transcrevendo stem '{}'…", stem_label));
        let params = build_offline_params(&config, None);

        let out = if config.mode == "poly" {
            ensure_engine_loaded(&engine_cache_arc)?;
            let mut eng_guard = engine_cache_arc.lock().unwrap();
            transcribe_stem_offline_with_engine(&stem_wav, &params, eng_guard.as_mut())?
        } else {
            transcribe_stem_offline_with_engine(&stem_wav, &params, None)?
        };

        let detected_key = out.score.key.display_name();
        let detected_bpm = out.score.bpm;
        let ts_label = out.score.time_signature.display_label();
        let time_sig = out.score.time_signature;

        let snap_events: Vec<TimedEvent> = out
            .score
            .measures
            .iter()
            .flat_map(|m| m.beats.iter())
            .filter(|b| !b.is_rest() && !b.tie_from_prev)
            .enumerate()
            .map(|(idx, b)| TimedEvent {
                start_ms: (idx as u64) * 250,
                duration_ms: ((b.duration_ticks as u64) * 60_000) / ((detected_bpm as u64) * 4),
                notes: b.midi_notes.clone(),
                bend_semitones: b.bend_semitones,
            })
            .collect();
        let snap_voicings: Vec<Option<ChordVoicing>> = out
            .score
            .measures
            .iter()
            .flat_map(|m| m.beats.iter())
            .filter(|b| !b.is_rest() && !b.tie_from_prev)
            .map(|b| b.voicing.clone())
            .collect();

        *latest_snap_arc.lock().unwrap() = Some(ScoreSnapshot {
            events: snap_events,
            voicings: snap_voicings,
            tuning_id: config.tuning.clone(),
            time_sig,
            detected_instrument: out.detected_instrument.clone(),
            score: out.score,
        });

        let _ = app_clone.emit(
            "transcription-update",
            TranscriptionUpdatePayload {
                alphatex: out.alphatex,
                latest_notes: out.latest_notes,
                latest_chord: out.latest_chord,
                detected_key,
                detected_instrument: out.detected_instrument,
                time_signature: ts_label,
                event_count: out.event_count,
                latency_ms: out.elapsed_ms,
                detected_bpm,
                estimated_a4_hz: out.estimated_a4_hz,
                pruned_overtones: out.pruned_overtones,
                bend_semitones: out.max_bend,
                is_reattack: false,
            },
        );

        Ok(format!(
            "Stem '{}' transcrito ({} eventos em {} ms)",
            stem_label, out.event_count, out.elapsed_ms
        ))
    })
    .await
    .map_err(|e| format!("Falha ao transcrever stem: {}", e))?
}

// ============================================================================
// MODO AO VIVO (ASSÍNCRONO + ENGINE EM CACHE + RE-RESOLUÇÃO DE APP)
// ============================================================================

#[tauri::command]
async fn start_transcription(
    app: AppHandle,
    state: State<'_, AppState>,
    config: StartConfig,
) -> Result<String, String> {
    let time_sig = TimeSignature::from_str_id(&config.time_signature);
    *state.active_tuning.lock().unwrap() = config.tuning.clone();
    *state.active_time_signature.lock().unwrap() = time_sig;

    let session_arc = Arc::clone(&state.session);
    let recorder_arc = Arc::clone(&state.snippet_recorder);
    let engine_cache_arc = Arc::clone(&state.engine_cache);
    let latest_snap_arc = Arc::clone(&state.latest_snapshot);

    tauri::async_runtime::spawn_blocking(move || {
        teardown_all_captures_blocking(&session_arc, &recorder_arc);

        let preset = get_tuning_preset(&config.tuning);
        let focus = InstrumentFocus::from_str_id(&config.focus);
        let stereo_mode = StereoMode::from_str_id(&config.stereo_mode);
        let enable_hpss = config.enable_hpss;
        let auto_level = config.auto_level;
        let region = NeckRegion::from_str_id(&config.neck_region);
        let solver_cfg = SolverConfig {
            region,
            prefer_closed_positions: config.prefer_closed,
            ..SolverConfig::default()
        };
        let bpm_override = config.bpm;

        if config.mode == "poly" {
            ensure_engine_loaded(&engine_cache_arc)?;
        }

        let target = resolve_fresh_capture_target(config.source_id.as_deref(), &config.target_ids);
        let stop_flag = Arc::new(AtomicBool::new(false));
        let clear_flag = Arc::new(AtomicBool::new(false));

        let stop_clone = Arc::clone(&stop_flag);
        let clear_clone = Arc::clone(&clear_flag);
        let mode = config.mode.clone();
        let tuning_id = config.tuning.clone();

        let (ready_tx, ready_rx) = mpsc::channel::<Result<(f64, String), String>>();

        let worker = thread::spawn(move || {
            let capture_result =
                MacOSLoopbackCapture::start_with_target(RING_CAPACITY_FRAMES, target);
            let (loopback, mut consumer) = match capture_result {
                Ok(pair) => pair,
                Err(e) => {
                    let _ = ready_tx.send(Err(e));
                    return;
                }
            };

            let mut active_sr = loopback.current_sample_rate();
            let fmt_label = loopback.format_label();
            let session_start = Instant::now();
            let mut dc_blocker = DcBlocker::new(active_sr as f32);
            let mut leveler = SignalLeveler::new(active_sr as f32);
            let mut filter = InstrumentFilter::new(active_sr as f32, focus);
            let mut hpss = HpssSeparator::new(1.05);

            if mode == "poly" {
                let _ = ready_tx.send(Ok((active_sr, fmt_label)));

                let (win_tx, win_rx) = mpsc::sync_channel::<(Vec<f32>, u64)>(2);
                let app_infer = app.clone();
                let preset_infer = preset.clone();
                let clear_infer = Arc::clone(&clear_clone);
                let snap_infer = Arc::clone(&latest_snap_arc);
                let solver_infer = solver_cfg.clone();
                let engine_infer = Arc::clone(&engine_cache_arc);
                let tuning_id_infer = tuning_id.clone();

                let infer_handle = thread::spawn(move || {
                    let mut cfg = DecodeConfig::default();
                    let (min_m, max_m) = focus.poly_midi_bounds();
                    cfg.min_midi = min_m;
                    cfg.max_midi = max_m;
                    if focus.preserve_wide_octaves() {
                        cfg.enable_overtone_pruning = false;
                    }
                    if matches!(focus, InstrumentFocus::BowedStrings | InstrumentFocus::Vocals) {
                        cfg.onset_threshold = 0.36;
                        cfg.frame_threshold = 0.25;
                    }
                    if !preset_infer.allow_bends {
                        cfg.bend_min_cents = 999.0;
                    }

                    let mut timed_history: Vec<TimedEvent> = Vec::with_capacity(MAX_SCORE_EVENTS);
                    let mut warmth_history: Vec<f32> = Vec::with_capacity(MAX_SCORE_EVENTS);
                    let mut smoothed_a4_hz: f32 = 440.0;
                    let mut total_pruned: usize = 0;

                    while let Ok((window_22k, window_end_ms)) = win_rx.recv() {
                        if clear_infer.swap(false, Ordering::SeqCst) {
                            timed_history.clear();
                            warmth_history.clear();
                            smoothed_a4_hz = 440.0;
                            total_pruned = 0;
                        }

                        let t0 = Instant::now();
                        let infer_res = {
                            let mut guard = engine_infer.lock().unwrap();
                            match guard.as_mut() {
                                Some(eng) => eng.transcribe_window(&window_22k, &cfg).map_err(|e| e.to_string()),
                                None => Err("Engine Basic Pitch indisponível.".to_string()),
                            }
                        };

                        match infer_res {
                            Ok(res) => {
                                let latency_ms = t0.elapsed().as_millis();
                                total_pruned += res.pruned_overtones;
                                if let Some(a4) = res.estimated_a4_hz {
                                    smoothed_a4_hz += 0.25 * (a4 - smoothed_a4_hz);
                                }

                                let mut changed = false;
                                let mut latest_bend = 0.0f32;

                                for ch in res.chords.into_iter().filter(|c| c.frame >= 86) {
                                    let frames_before_end = 172usize.saturating_sub(ch.frame);
                                    let raw_start_ms = window_end_ms.saturating_sub(
                                        (frames_before_end as f64 * FRAME_DURATION_MS) as u64,
                                    );
                                    let duration_ms = ((ch.duration_frames as f64)
                                        * FRAME_DURATION_MS)
                                        .round() as u64;

                                    if let Some(last) = timed_history.last() {
                                        if last.notes == ch.notes
                                            && raw_start_ms.saturating_sub(last.start_ms) < 350
                                        {
                                            continue;
                                        }
                                    }

                                    let start_ms = match timed_history.last() {
                                        Some(last) => raw_start_ms.max(last.start_ms + 70),
                                        None => raw_start_ms,
                                    };

                                    if timed_history.len() == MAX_SCORE_EVENTS {
                                        timed_history.remove(0);
                                        warmth_history.remove(0);
                                    }
                                    latest_bend = ch.max_bend_semitones;
                                    timed_history.push(TimedEvent {
                                        start_ms,
                                        duration_ms: duration_ms.max(100),
                                        notes: ch.notes,
                                        bend_semitones: ch.max_bend_semitones,
                                    });
                                    warmth_history.push(ch.timbral_warmth);
                                    changed = true;
                                }

                                if changed {
                                    let mut all_m: Vec<u8> = timed_history
                                        .iter()
                                        .flat_map(|e| e.notes.iter().copied())
                                        .collect();
                                    all_m.sort_unstable();
                                    let med_m = all_m.get(all_m.len() / 2).copied().unwrap_or(60) as f32;
                                    let mean_poly =
                                        (all_m.len() as f32) / (timed_history.len().max(1) as f32);
                                    let mean_w = warmth_history.iter().sum::<f32>()
                                        / (warmth_history.len().max(1) as f32);
                                    let perc = compute_percussiveness(&window_22k);
                                    let detected_prof =
                                        detect_instrument_profile(med_m, mean_poly, mean_w, perc);
                                    let detected_instrument = detected_prof.label.to_string();

                                    let active_preset = if tuning_id_infer == "auto" {
                                        get_tuning_preset(detected_prof.preset_id)
                                    } else {
                                        preset_infer.clone()
                                    };

                                    let voicings = if active_preset.show_tab_staff {
                                        let raw_chords: Vec<Vec<u8>> =
                                            timed_history.iter().map(|e| e.notes.clone()).collect();
                                        solve_polyphonic_with_config(
                                            &raw_chords,
                                            Some(&warmth_history),
                                            &active_preset.midi_low_to_high,
                                            24,
                                            &solver_infer,
                                        )
                                    } else {
                                        vec![None; timed_history.len()]
                                    };

                                    let q_score = build_quantized_score(
                                        "Tab Transcriber",
                                        &timed_history,
                                        &voicings,
                                        &active_preset.midi_low_to_high,
                                        bpm_override,
                                        time_sig,
                                    )
                                    .with_instrument(
                                        active_preset.midi_program,
                                        active_preset.show_tab_staff,
                                        active_preset.clef,
                                    );

                                    let detected_bpm = q_score.bpm;
                                    let detected_key = q_score.key.display_name();
                                    let latest_chord = latest_chord_label(&q_score);
                                    let key_obj = q_score.key;

                                    let subtitle = format!(
                                        "{} · {} — {} · {} · {} BPM",
                                        detected_instrument,
                                        active_preset.label,
                                        detected_key,
                                        time_sig.display_label(),
                                        detected_bpm
                                    );
                                    let alphatex = score_to_alphatex(
                                        &q_score,
                                        active_preset.alphatex_high_to_low,
                                        &subtitle,
                                    );

                                    *snap_infer.lock().unwrap() = Some(ScoreSnapshot {
                                        events: timed_history.clone(),
                                        voicings,
                                        tuning_id: tuning_id_infer.clone(),
                                        time_sig,
                                        detected_instrument: detected_instrument.clone(),
                                        score: q_score,
                                    });

                                    let latest_notes = timed_history
                                        .last()
                                        .map(|e| {
                                            e.notes
                                                .iter()
                                                .map(|&m| midi_to_note_name_for_key(m, key_obj))
                                                .collect()
                                        })
                                        .unwrap_or_default();

                                    let _ = app_infer.emit(
                                        "transcription-update",
                                        TranscriptionUpdatePayload {
                                            alphatex,
                                            latest_notes,
                                            latest_chord,
                                            detected_key,
                                            detected_instrument,
                                            time_signature: time_sig.display_label(),
                                            event_count: timed_history.len(),
                                            latency_ms,
                                            detected_bpm,
                                            estimated_a4_hz: smoothed_a4_hz,
                                            pruned_overtones: total_pruned,
                                            bend_semitones: latest_bend,
                                            is_reattack: false,
                                        },
                                    );
                                }
                            }
                            Err(e) => {
                                let _ = app_infer.emit("transcription-error", e);
                            }
                        }
                    }
                });

                let mut resampler = SincResampler::new(active_sr, MODEL_SAMPLE_RATE);
                let mut read_chunk = [[0.0f32; 2]; 1024];
                let mut mono_chunk: Vec<f32> = Vec::with_capacity(1024);
                let mut hpss_chunk: Vec<f32> = Vec::with_capacity(1536);
                let mut filtered_chunk: Vec<f32> = Vec::with_capacity(1536);
                let mut resampled_chunk: Vec<f32> = Vec::with_capacity(1024);
                let mut rolling_22k: VecDeque<f32> =
                    VecDeque::with_capacity(MODEL_WINDOW_SAMPLES + 2048);

                let hop_22k = (MODEL_SAMPLE_RATE * 0.75) as usize;
                let mut samples_since_infer = 0usize;
                let mut zero_chunks = 0u32;
                let mut last_meter = Instant::now();

                while !stop_clone.load(Ordering::Relaxed) {
                    let live_sr = loopback.current_sample_rate();
                    if (live_sr - active_sr).abs() > 1.0 {
                        active_sr = live_sr;
                        dc_blocker = DcBlocker::new(active_sr as f32);
                        leveler = SignalLeveler::new(active_sr as f32);
                        filter = InstrumentFilter::new(active_sr as f32, focus);
                        resampler = SincResampler::new(active_sr, MODEL_SAMPLE_RATE);
                    }

                    let (popped, _) = consumer.pop_partial_slice(&mut read_chunk);
                    if popped.is_empty() {
                        thread::sleep(Duration::from_millis(3));
                        continue;
                    }

                    let mut raw_peak = 0.0f32;
                    for fr in popped.iter() {
                        raw_peak = raw_peak.max(fr[0].abs()).max(fr[1].abs());
                    }
                    if raw_peak == 0.0 {
                        zero_chunks = zero_chunks.saturating_add(1);
                    } else {
                        zero_chunks = 0;
                    }

                    stereo_mode.extract_slice(popped, &mut mono_chunk);
                    dc_blocker.process_in_place(&mut mono_chunk);
                    if auto_level {
                        leveler.process_in_place(&mut mono_chunk);
                    }

                    let pre_filter = if enable_hpss {
                        hpss.process_slice(&mono_chunk, &mut hpss_chunk);
                        &hpss_chunk[..]
                    } else {
                        &mono_chunk[..]
                    };
                    filter.process_slice(pre_filter, &mut filtered_chunk);

                    if !filtered_chunk.is_empty() && last_meter.elapsed() >= Duration::from_millis(80)
                    {
                        last_meter = Instant::now();
                        let sum_sq: f64 = filtered_chunk
                            .iter()
                            .map(|&s| (s as f64) * (s as f64))
                            .sum();
                        let rms = (sum_sq / filtered_chunk.len() as f64).sqrt() as f32;
                        let rms_db = 20.0 * rms.max(1e-7).log10();
                        let peak_db = 20.0 * raw_peak.max(1e-7).log10();
                        let _ = app.emit(
                            "audio-level",
                            AudioLevelPayload {
                                rms_db,
                                peak_db,
                                silence_warning: zero_chunks >= 45,
                                dropped_samples: loopback.dropped_samples(),
                                device_switches: loopback.device_switches(),
                                clip_events: loopback.clip_events(),
                                clock_glitches: loopback.clock_glitches(),
                                phase_corr: compute_phase_correlation(popped),
                                sample_rate: active_sr.round() as u32,
                                format_label: loopback.format_label(),
                            },
                        );
                    }

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
                        let energy: f32 = rolling_22k.iter().map(|&s| s * s).sum::<f32>()
                            / MODEL_WINDOW_SAMPLES as f32;
                        if energy.sqrt() > 0.008 {
                            let snapshot: Vec<f32> = rolling_22k.iter().copied().collect();
                            let now_ms = session_start.elapsed().as_millis() as u64;
                            let _ = win_tx.try_send((snapshot, now_ms));
                        }
                    }
                }

                drop(win_tx);
                let _ = infer_handle.join();
            } else {
                let _ = ready_tx.send(Ok((active_sr, fmt_label)));

                let mut detector = YinDetector::new(active_sr as f32, YIN_WINDOW_SIZE);
                let (min_hz, max_hz) = focus.yin_freq_bounds();
                detector.set_freq_range(min_hz, max_hz);

                let mut stabilizer = NoteStabilizer::new(3, 6);

                let mut window = [0.0f32; YIN_WINDOW_SIZE];
                let mut hop_buf = [0.0f32; YIN_HOP_SIZE];
                let mut read_chunk = [[0.0f32; 2]; 512];
                let mut mono_chunk: Vec<f32> = Vec::with_capacity(512);
                let mut hpss_chunk: Vec<f32> = Vec::with_capacity(1024);
                let mut filtered_chunk: Vec<f32> = Vec::with_capacity(1024);
                let mut hop_filled = 0usize;
                let mut timed_history: Vec<TimedEvent> = Vec::with_capacity(MAX_SCORE_EVENTS);
                let mut warmth_history: Vec<f32> = Vec::with_capacity(MAX_SCORE_EVENTS);
                let mut smoothed_a4_hz: f32 = 440.0;
                let mut zero_chunks = 0u32;
                let mut last_meter = Instant::now();

                while !stop_clone.load(Ordering::Relaxed) {
                    let live_sr = loopback.current_sample_rate();
                    if (live_sr - active_sr).abs() > 1.0 {
                        active_sr = live_sr;
                        dc_blocker = DcBlocker::new(active_sr as f32);
                        leveler = SignalLeveler::new(active_sr as f32);
                        filter = InstrumentFilter::new(active_sr as f32, focus);
                        detector = YinDetector::new(active_sr as f32, YIN_WINDOW_SIZE);
                        detector.set_freq_range(min_hz, max_hz);
                    }

                    if clear_clone.swap(false, Ordering::SeqCst) {
                        timed_history.clear();
                        warmth_history.clear();
                        smoothed_a4_hz = 440.0;
                    }

                    let (popped, _) = consumer.pop_partial_slice(&mut read_chunk);
                    if popped.is_empty() {
                        thread::sleep(Duration::from_millis(2));
                        continue;
                    }

                    let mut raw_peak = 0.0f32;
                    for fr in popped.iter() {
                        raw_peak = raw_peak.max(fr[0].abs()).max(fr[1].abs());
                    }
                    if raw_peak == 0.0 {
                        zero_chunks = zero_chunks.saturating_add(1);
                    } else {
                        zero_chunks = 0;
                    }

                    stereo_mode.extract_slice(popped, &mut mono_chunk);
                    dc_blocker.process_in_place(&mut mono_chunk);
                    if auto_level {
                        leveler.process_in_place(&mut mono_chunk);
                    }

                    let pre_filter = if enable_hpss {
                        hpss.process_slice(&mono_chunk, &mut hpss_chunk);
                        &hpss_chunk[..]
                    } else {
                        &mono_chunk[..]
                    };
                    filter.process_slice(pre_filter, &mut filtered_chunk);

                    if !filtered_chunk.is_empty() && last_meter.elapsed() >= Duration::from_millis(80)
                    {
                        last_meter = Instant::now();
                        let sum_sq: f64 = filtered_chunk
                            .iter()
                            .map(|&s| (s as f64) * (s as f64))
                            .sum();
                        let rms = (sum_sq / filtered_chunk.len() as f64).sqrt() as f32;
                        let rms_db = 20.0 * rms.max(1e-7).log10();
                        let peak_db = 20.0 * raw_peak.max(1e-7).log10();
                        let _ = app.emit(
                            "audio-level",
                            AudioLevelPayload {
                                rms_db,
                                peak_db,
                                silence_warning: zero_chunks >= 60,
                                dropped_samples: loopback.dropped_samples(),
                                device_switches: loopback.device_switches(),
                                clip_events: loopback.clip_events(),
                                clock_glitches: loopback.clock_glitches(),
                                phase_corr: compute_phase_correlation(popped),
                                sample_rate: active_sr.round() as u32,
                                format_label: loopback.format_label(),
                            },
                        );
                    }

                    for &sample in &filtered_chunk {
                        hop_buf[hop_filled] = sample;
                        hop_filled += 1;

                        if hop_filled == YIN_HOP_SIZE {
                            window.copy_within(YIN_HOP_SIZE.., 0);
                            window[YIN_WINDOW_SIZE - YIN_HOP_SIZE..].copy_from_slice(&hop_buf);
                            hop_filled = 0;

                            let t0 = Instant::now();
                            if let Some(event) = stabilizer.push(detector.detect(&window)) {
                                let now_ms = session_start.elapsed().as_millis() as u64;

                                if event.cents.abs() <= 45.0 {
                                    let inst_a4 = 440.0 * 2.0f32.powf(event.cents / 1200.0);
                                    smoothed_a4_hz += 0.15 * (inst_a4 - smoothed_a4_hz);
                                }

                                if let Some(prev) = timed_history.last_mut() {
                                    prev.duration_ms = now_ms
                                        .saturating_sub(prev.start_ms)
                                        .clamp(80, 2000);
                                }

                                if timed_history.len() == MAX_SCORE_EVENTS {
                                    timed_history.remove(0);
                                    warmth_history.remove(0);
                                }
                                timed_history.push(TimedEvent {
                                    start_ms: now_ms,
                                    duration_ms: 250,
                                    notes: vec![event.midi_note],
                                    bend_semitones: 0.0,
                                });
                                warmth_history.push(event.timbral_warmth);

                                let voicings: Vec<Option<ChordVoicing>> = if preset.show_tab_staff {
                                    let midi_seq: Vec<u8> = timed_history
                                        .iter()
                                        .filter_map(|e| e.notes.first().copied())
                                        .collect();
                                    solve_tablature_with_config(
                                        &midi_seq,
                                        Some(&warmth_history),
                                        &preset.midi_low_to_high,
                                        24,
                                        &solver_cfg,
                                    )
                                    .into_iter()
                                    .map(|opt| opt.map(|p| vec![p]))
                                    .collect()
                                } else {
                                    vec![None; timed_history.len()]
                                };

                                let q_score = build_quantized_score(
                                    "Tab Transcriber",
                                    &timed_history,
                                    &voicings,
                                    &preset.midi_low_to_high,
                                    bpm_override,
                                    time_sig,
                                )
                                .with_instrument(preset.midi_program, preset.show_tab_staff, preset.clef);

                                let mut all_m: Vec<u8> = timed_history
                                    .iter()
                                    .filter_map(|e| e.notes.first().copied())
                                    .collect();
                                all_m.sort_unstable();
                                let med_m = all_m.get(all_m.len() / 2).copied().unwrap_or(60) as f32;
                                let mean_w = warmth_history.iter().sum::<f32>()
                                    / (warmth_history.len().max(1) as f32);
                                let detected_instrument =
                                    detect_instrument_profile(med_m, 1.0, mean_w, compute_percussiveness(&window)).label
                                        .to_string();

                                let detected_bpm = q_score.bpm;
                                let detected_key = q_score.key.display_name();
                                let key_obj = q_score.key;

                                let subtitle = format!(
                                    "{} · {} — {} · {} · {} BPM",
                                    detected_instrument,
                                    preset.label,
                                    detected_key,
                                    time_sig.display_label(),
                                    detected_bpm
                                );
                                let alphatex = score_to_alphatex(
                                    &q_score,
                                    preset.alphatex_high_to_low,
                                    &subtitle,
                                );

                                *latest_snap_arc.lock().unwrap() = Some(ScoreSnapshot {
                                    events: timed_history.clone(),
                                    voicings,
                                    tuning_id: tuning_id.clone(),
                                    time_sig,
                                    detected_instrument: detected_instrument.clone(),
                                    score: q_score,
                                });

                                let latency_ms = t0.elapsed().as_millis();
                                let _ = app.emit(
                                    "transcription-update",
                                    TranscriptionUpdatePayload {
                                        alphatex,
                                        latest_notes: vec![midi_to_note_name_for_key(
                                            event.midi_note,
                                            key_obj,
                                        )],
                                        latest_chord: None,
                                        detected_key,
                                        detected_instrument,
                                        time_signature: time_sig.display_label(),
                                        event_count: timed_history.len(),
                                        latency_ms,
                                        detected_bpm,
                                        estimated_a4_hz: smoothed_a4_hz,
                                        pruned_overtones: 0,
                                        bend_semitones: 0.0,
                                        is_reattack: event.is_reattack,
                                    },
                                );
                            }
                        }
                    }
                }
            }
        });

        match ready_rx.recv() {
            Ok(Ok((sr, fmt))) => {
                *session_arc.lock().unwrap() = Some(RunningSession {
                    stop_flag,
                    clear_flag,
                    worker: Some(worker),
                });
                Ok(format!(
                    "Ao Vivo {:.0} Hz ({}) · {} · {}",
                    sr,
                    fmt,
                    stereo_mode.label(),
                    focus.label()
                ))
            }
            Ok(Err(err)) => {
                let _ = worker.join();
                Err(err)
            }
            Err(_) => {
                let _ = worker.join();
                Err("Thread de captura encerrou inesperadamente.".to_string())
            }
        }
    })
    .await
    .map_err(|e| format!("Falha ao iniciar transcrição ao vivo: {}", e))?
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .manage(AppState::default())
        .invoke_handler(tauri::generate_handler![
            check_stems_backend,
            load_stem_audio,
            list_audio_sources,
            get_initial_tex,
            start_transcription,
            stop_transcription,
            cancel_snippet_recording,
            clear_transcription,
            rescale_score_bpm,
            export_score,
            open_exports_folder,
            start_snippet_recording,
            stop_and_separate_snippet,
            transcribe_existing_stem,
        ])
        .run(tauri::generate_context!())
        .expect("erro ao executar aplicação Tauri");
}
