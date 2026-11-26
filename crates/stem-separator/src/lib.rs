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
    build_quantized_score, infer_optimal_clef, midi_to_note_name_for_key, score_to_alphatex,
    ClefKind, QuantizedScore, TimeSignature, TimedEvent, NeuralBeatGrid, build_score_with_beat_grid,};
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

#[derive(Debug, Clone, Serialize, Deserialize)]
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

fn exe_bundle_resources_dir() -> Option<PathBuf> {
    let exe = std::env::current_exe().ok()?;
    let contents = exe.parent()?.parent()?;
    let res = contents.join("Resources");
    if res.is_dir() {
        Some(res)
    } else {
        None
    }
}

fn enriched_env_path(python: &Path) -> String {
    let mut parts: Vec<String> = Vec::new();
    if let Some(py_dir) = python.parent() {
        let s = py_dir.display().to_string();
        if !s.is_empty() {
            parts.push(s);
        }
    }
    if let Ok(home) = std::env::var("HOME") {
        parts.push(format!("{}/.local/bin", home));
    }
    parts.push("/opt/homebrew/bin".to_string());
    parts.push("/usr/local/bin".to_string());
    if let Ok(curr) = std::env::var("PATH") {
        if !curr.is_empty() {
            parts.push(curr);
        }
    } else {
        parts.push("/usr/bin:/bin:/usr/sbin:/sbin".to_string());
    }
    parts.join(":")
}

fn resolve_python_and_script(primary_root: &Path) -> Result<(PathBuf, PathBuf), String> {
    let home_dir = std::env::var("HOME").ok().map(PathBuf::from);
    let bundle_res = exe_bundle_resources_dir();

    // 1. Localiza scripts/separate_stems.py (env -> primary_root -> bundle -> workspace)
    let mut script_candidates: Vec<PathBuf> = Vec::new();
    if let Ok(env_script) = std::env::var("TAB_TRANSCRIBER_STEMS_SCRIPT") {
        if !env_script.trim().is_empty() {
            script_candidates.push(PathBuf::from(env_script));
        }
    }
    script_candidates.push(primary_root.join("scripts/separate_stems.py"));
    script_candidates.push(primary_root.join("separate_stems.py"));
    script_candidates.push(primary_root.join("_up_/_up_/_up_/scripts/separate_stems.py"));
    if let Some(ref res) = bundle_res {
        script_candidates.push(res.join("scripts/separate_stems.py"));
        script_candidates.push(res.join("separate_stems.py"));
    }
    if let Some(ref home) = home_dir {
        script_candidates.push(
            home.join("Library/Application Support/TabTranscriber/scripts/separate_stems.py"),
        );
        script_candidates.push(home.join("tab-transcriber/scripts/separate_stems.py"));
    }

    let script = script_candidates
        .into_iter()
        .find(|p| p.is_file())
        .ok_or_else(|| {
            format!(
                "Script 'separate_stems.py' não encontrado a partir de '{}'.",
                primary_root.display()
            )
        })?;

    // 2. Localiza Python da .venv-stems (env -> primary_root -> App Support -> workspace -> Homebrew)
    let mut py_candidates: Vec<PathBuf> = Vec::new();
    if let Ok(env_py) = std::env::var("TAB_TRANSCRIBER_PYTHON") {
        if !env_py.trim().is_empty() {
            py_candidates.push(PathBuf::from(env_py));
        }
    }
    py_candidates.push(primary_root.join(".venv-stems/bin/python"));
    py_candidates.push(primary_root.join(".venv-stems/bin/python3"));
    if let Some(ref res) = bundle_res {
        py_candidates.push(res.join(".venv-stems/bin/python"));
    }
    if let Some(ref home) = home_dir {
        py_candidates
            .push(home.join("Library/Application Support/TabTranscriber/venv-stems/bin/python"));
        py_candidates.push(home.join("tab-transcriber/.venv-stems/bin/python"));
    }
    py_candidates.push(PathBuf::from("/opt/homebrew/bin/python3"));
    py_candidates.push(PathBuf::from("/usr/local/bin/python3"));

    let python = py_candidates
        .into_iter()
        .find(|p| p.is_file())
        .unwrap_or_else(|| PathBuf::from("python3"));

    Ok((python, script))
}

/// Retorna `(python_bin, script_path)` atualmente resolvidos para diagnóstico da UI/bundle.
#[derive(Debug, Clone, Serialize)]
pub struct NeuralBackendStatus {
    pub muscriptor_ready: bool,
    pub beat_this_ready: bool,
    pub mps_available: bool,
    pub device: String,
    pub hint: String,
}

#[derive(Debug, Deserialize)]
struct NeuralCheckJson {
    muscriptor_ready: bool,
    beat_this_ready: bool,
    mps_available: bool,
    device: String,
    hint: String,
}

#[derive(Debug, Deserialize)]
struct NeuralNoteJson {
    start_ms: u64,
    duration_ms: u64,
    midi_note: u8,
    #[serde(default)]
    velocity: u8,
}

#[derive(Debug, Deserialize)]
struct NeuralTranscribeJson {
    ok: bool,
    #[serde(default)]
    muscriptor_used: bool,
    #[serde(default)]
    muscriptor_engine: String,
    #[serde(default)]
    muscriptor_error: Option<String>,
    #[serde(default)]
    beat_engine: String,
    #[serde(default)]
    beats_sec: Vec<f64>,
    #[serde(default)]
    downbeats_sec: Vec<f64>,
    #[serde(default)]
    notes: Vec<NeuralNoteJson>,
    #[serde(default)]
    error: Option<String>,
}

fn resolve_neural_python_and_script(primary_root: &Path) -> Result<(PathBuf, PathBuf), String> {
    let home_dir = std::env::var("HOME").ok().map(PathBuf::from);
    let bundle_res = exe_bundle_resources_dir();

    let mut script_candidates: Vec<PathBuf> = Vec::new();
    if let Ok(env_script) = std::env::var("TAB_TRANSCRIBER_NEURAL_SCRIPT") {
        if !env_script.trim().is_empty() {
            script_candidates.push(PathBuf::from(env_script));
        }
    }
    script_candidates.push(primary_root.join("scripts/transcribe_neural.py"));
    script_candidates.push(primary_root.join("transcribe_neural.py"));
    if let Some(ref res) = bundle_res {
        script_candidates.push(res.join("scripts/transcribe_neural.py"));
        script_candidates.push(res.join("transcribe_neural.py"));
    }
    if let Some(ref home) = home_dir {
        script_candidates.push(
            home.join("Library/Application Support/TabTranscriber/scripts/transcribe_neural.py"),
        );
        script_candidates.push(home.join("tab-transcriber/scripts/transcribe_neural.py"));
    }

    let script = script_candidates
        .into_iter()
        .find(|p| p.is_file())
        .ok_or_else(|| "Script 'transcribe_neural.py' não encontrado.".to_string())?;

    let mut py_candidates: Vec<PathBuf> = Vec::new();
    if let Ok(env_py) = std::env::var("TAB_TRANSCRIBER_NEURAL_PYTHON") {
        if !env_py.trim().is_empty() {
            py_candidates.push(PathBuf::from(env_py));
        }
    }
    py_candidates.push(primary_root.join(".venv-neural/bin/python"));
    if let Some(ref res) = bundle_res {
        py_candidates.push(res.join(".venv-neural/bin/python"));
    }
    if let Some(ref home) = home_dir {
        py_candidates
            .push(home.join("Library/Application Support/TabTranscriber/venv-neural/bin/python"));
        py_candidates.push(home.join("tab-transcriber/.venv-neural/bin/python"));
        py_candidates.push(home.join("tab-transcriber/.venv-stems/bin/python"));
    }
    py_candidates.push(primary_root.join(".venv-stems/bin/python"));

    let python = py_candidates
        .into_iter()
        .find(|p| p.is_file())
        .unwrap_or_else(|| PathBuf::from("python3"));

    Ok((python, script))
}

pub fn check_neural_backend(primary_root: &Path) -> NeuralBackendStatus {
    let Ok((python, script)) = resolve_neural_python_and_script(primary_root) else {
        return NeuralBackendStatus {
            muscriptor_ready: false,
            beat_this_ready: false,
            mps_available: false,
            device: "cpu".to_string(),
            hint: "Script scripts/transcribe_neural.py ausente.".to_string(),
        };
    };

    let output = Command::new(&python)
        .env("PATH", enriched_env_path(&python))
        .arg(&script)
        .arg("--check")
        .output();

    if let Ok(out) = output {
        let stdout = String::from_utf8_lossy(&out.stdout);
        if let Some(line) = stdout.lines().rev().find(|l| l.trim().starts_with('{')) {
            if let Ok(parsed) = serde_json::from_str::<NeuralCheckJson>(line) {
                return NeuralBackendStatus {
                    muscriptor_ready: parsed.muscriptor_ready,
                    beat_this_ready: parsed.beat_this_ready,
                    mps_available: parsed.mps_available,
                    device: parsed.device,
                    hint: parsed.hint,
                };
            }
        }
    }

    NeuralBackendStatus {
        muscriptor_ready: false,
        beat_this_ready: false,
        mps_available: false,
        device: "cpu".to_string(),
        hint: "Instale beat-this e muscriptor em ~/tab-transcriber/.venv-neural".to_string(),
    }
}

fn run_neural_worker(
    primary_root: &Path,
    stem_wav: &Path,
    mix_wav: &Path,
    stem_name: &str,
    engine_mode: &str,
) -> Result<NeuralTranscribeJson, String> {
    let (python, script) = resolve_neural_python_and_script(primary_root)?;
    let output = Command::new(&python)
        .env("PATH", enriched_env_path(&python))
        .arg(&script)
        .arg("--audio")
        .arg(stem_wav)
        .arg("--mix")
        .arg(mix_wav)
        .arg("--stem")
        .arg(stem_name)
        .arg("--engine")
        .arg(engine_mode)
        .arg("--device")
        .arg("auto")
        .output()
        .map_err(|e| format!("Falha ao executar worker neural: {}", e))?;

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let json_line = stdout
        .lines()
        .rev()
        .find(|l| l.trim().starts_with('{'))
        .ok_or_else(|| format!("Saída inesperada do worker neural: {}", stderr.trim()))?;

    let parsed: NeuralTranscribeJson = serde_json::from_str(json_line)
        .map_err(|e| format!("Erro JSON do worker neural: {} ({})", e, json_line))?;
    if !parsed.ok {
        return Err(parsed.error.unwrap_or_else(|| stderr.trim().to_string()));
    }
    Ok(parsed)
}

fn group_neural_notes_to_events(
    raw_notes: &[NeuralNoteJson],
    audio_mono: &[f32],
    sample_rate: u32,
    min_midi: u8,
    max_midi: u8,
    max_polyphony: usize,
) -> (Vec<TimedEvent>, Vec<f32>) {
    let mut filtered: Vec<&NeuralNoteJson> = raw_notes
        .iter()
        .filter(|n| n.midi_note >= min_midi && n.midi_note <= max_midi && n.duration_ms >= 40)
        .collect();
    filtered.sort_by_key(|n| (n.start_ms, std::cmp::Reverse(n.velocity), n.midi_note));

    let mut groups: Vec<Vec<&NeuralNoteJson>> = Vec::new();
    for n in filtered {
        if let Some(last) = groups.last_mut() {
            if n.start_ms <= last[0].start_ms.saturating_add(45) {
                last.push(n);
                continue;
            }
        }
        groups.push(vec![n]);
    }

    let mut events = Vec::with_capacity(groups.len());
    let mut warmths = Vec::with_capacity(groups.len());
    let cap = max_polyphony.max(1);

    for mut g in groups.into_iter().take(MAX_OFFLINE_EVENTS) {
        g.sort_by(|a, b| b.velocity.cmp(&a.velocity));
        g.truncate(cap);

        let start_ms = g.iter().map(|n| n.start_ms).min().unwrap_or(0);
        let duration_ms = g.iter().map(|n| n.duration_ms).max().unwrap_or(100).max(90);
        let mut midi_notes: Vec<u8> = g.iter().map(|n| n.midi_note).collect();
        midi_notes.sort_unstable();
        midi_notes.dedup();

        let s_idx = (((start_ms as f64) / 1000.0) * (sample_rate as f64)) as usize;
        let e_idx = ((((start_ms + duration_ms) as f64) / 1000.0) * (sample_rate as f64)) as usize;
        let s_clamped = s_idx.min(audio_mono.len());
        let e_clamped = e_idx.min(audio_mono.len());
        let w = if e_clamped > s_clamped + 64 {
            let ref_midi = midi_notes[midi_notes.len() / 2];
            let f0 = 440.0 * 2.0f32.powf(((ref_midi as f32) - 69.0) / 12.0);
            estimate_timbral_warmth(&audio_mono[s_clamped..e_clamped], sample_rate as f32, f0)
        } else {
            0.5
        };

        events.push(TimedEvent {
            start_ms,
            duration_ms,
            notes: midi_notes,
            bend_semitones: 0.0,
        });
        warmths.push(w);
    }

    (events, warmths)
}

pub fn describe_stems_runtime(primary_root: &Path) -> (Option<PathBuf>, Option<PathBuf>) {
    match resolve_python_and_script(primary_root) {
        Ok((py, sc)) => (Some(py), Some(sc)),
        Err(_) => (None, None),
    }
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
        .env("PATH", enriched_env_path(&python))
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
        .env("PATH", enriched_env_path(&python))
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

/// Reconstrói um `StemBundle` a partir de uma pasta de sessão já separada em disco.
/// Usa `cached_activities` se fornecidas (evita reler todos os WAVs) ou calcula sob demanda.
pub fn load_stem_bundle_from_dir(
    session_dir: &Path,
    engine_hint: Option<&str>,
    cached_activities: Option<Vec<StemActivity>>,
) -> Result<StemBundle, String> {
    if !session_dir.is_dir() {
        return Err(format!(
            "Pasta de sessão não encontrada: '{}'",
            session_dir.display()
        ));
    }

    let mut stems = BTreeMap::new();
    for key in ["guitar", "bass", "piano", "vocals", "other", "drums"] {
        let p = session_dir.join(format!("{}.wav", key));
        if p.exists() {
            stems.insert(key.to_string(), p);
        }
    }

    let mix_wav = session_dir.join("mix.wav");
    if stems.is_empty() && !mix_wav.exists() {
        return Err(format!(
            "Nenhum arquivo de stem (.wav) encontrado em '{}'",
            session_dir.display()
        ));
    }

    let activities = match cached_activities {
        Some(acts) if !acts.is_empty() => acts,
        _ => {
            let mut computed = Vec::new();
            for key in ["guitar", "bass", "piano", "vocals", "other", "drums"] {
                if let Some(p) = stems.get(key) {
                    computed.push(analyze_stem_activity(key, p));
                }
            }
            computed
        }
    };

    let effective_mix = if mix_wav.exists() {
        mix_wav
    } else {
        stems
            .values()
            .next()
            .cloned()
            .unwrap_or_else(|| session_dir.join("mix.wav"))
    };

    Ok(StemBundle {
        session_dir: session_dir.to_path_buf(),
        mix_wav: effective_mix,
        stems,
        activities,
        engine: engine_hint.unwrap_or("cache local").to_string(),
        separation_ms: 0,
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
    pub clef_override: Option<ClefKind>,
    pub allow_bends: bool,
    pub solver_cfg: SolverConfig,
    pub time_signature: TimeSignature,
    pub bpm_override: Option<u16>,
    pub focus: InstrumentFocus,
    pub model_path: Option<PathBuf>,
    pub neural_engine: String,
    pub neural_root: Option<PathBuf>,
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
    pub engine_used: String,
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
        let mut filter = if params.mode == "mono" {
            InstrumentFilter::new(sample_rate as f32, params.focus)
        } else {
            InstrumentFilter::new_for_poly(sample_rate as f32, params.focus)
        };
        let mut filtered = Vec::with_capacity(audio_mono.len());
        filter.process_slice(&audio_mono, &mut filtered);
        audio_mono = filtered;
    }

    let mut timed_history: Vec<TimedEvent> = Vec::new();
    let mut warmth_history: Vec<f32> = Vec::new();
    let mut estimated_a4_hz: f32 = 440.0;
    let mut total_pruned: usize = 0;
    let mut max_bend: f32 = 0.0;
    let mut neural_beat_grid = NeuralBeatGrid::default();
    let mut engine_tags: Vec<String> = Vec::new();
    let mut used_muscriptor = false;

    if let Some(ref root) = params.neural_root {
        let mix_for_beats = wav_path
            .parent()
            .map(|d| d.join("mix.wav"))
            .filter(|p| p.is_file())
            .unwrap_or_else(|| wav_path.to_path_buf());

        let worker_mode = if params.mode == "mono" || params.neural_engine == "basic_pitch" {
            "beats_only"
        } else if params.neural_engine == "muscriptor" {
            "muscriptor"
        } else {
            "auto"
        };

        if let Ok(nr) = run_neural_worker(root, wav_path, &mix_for_beats, &params.stem_name, worker_mode) {
            if nr.beats_sec.len() >= 4 {
                neural_beat_grid.beats_ms = nr
                    .beats_sec
                    .iter()
                    .map(|&sec| (sec.max(0.0) * 1000.0).round() as u64)
                    .collect();
                neural_beat_grid.downbeats_ms = nr
                    .downbeats_sec
                    .iter()
                    .map(|&sec| (sec.max(0.0) * 1000.0).round() as u64)
                    .collect();
                if !nr.beat_engine.is_empty() && nr.beat_engine != "none" {
                    engine_tags.push(nr.beat_engine);
                }
            }

            if nr.muscriptor_used && !nr.notes.is_empty() && params.mode != "mono" {
                let (min_m, max_m) = stem_midi_bounds(&params.stem_name, params.focus);
                let max_poly = match params.stem_name.as_str() {
                    "vocals" | "bass" => 1,
                    _ if matches!(
                        params.focus,
                        InstrumentFocus::Vocals | InstrumentFocus::Bass | InstrumentFocus::WindsBrass
                    ) => 1,
                    _ => 6,
                };
                let (evs, ws) = group_neural_notes_to_events(
                    &nr.notes,
                    &audio_mono,
                    sample_rate,
                    min_m,
                    max_m,
                    max_poly,
                );
                if !evs.is_empty() {
                    timed_history = evs;
                    warmth_history = ws;
                    used_muscriptor = true;
                    engine_tags.insert(0, nr.muscriptor_engine);
                }
            } else if worker_mode != "beats_only" && !nr.muscriptor_used {
                if let Some(ref err) = nr.muscriptor_error {
                    let short_err: String = err.chars().take(72).collect();
                    engine_tags.push(format!("MuScriptor indisponível: {}", short_err));
                }
            }
        }
    }

    if used_muscriptor {
        // Notas já extraídas em alta precisão pelo MuScriptor
    } else if params.mode == "mono" {
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
        let mut owned_engine;
        let engine: &mut BasicPitchEngine = match cached_engine {
            Some(eng) => eng,
            None => {
                let model_path = params
                    .model_path
                    .as_ref()
                    .ok_or_else(|| "Caminho do modelo nmp.onnx não informado.".to_string())?;
                owned_engine = BasicPitchEngine::load(model_path)
                    .map_err(|e| format!("Falha ao carregar Basic Pitch ONNX: {}", e))?;
                &mut owned_engine
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

    if !neural_beat_grid.is_usable() {
        let first_ms = timed_history[0].start_ms;
        for ev in &mut timed_history {
            ev.start_ms = ev.start_ms.saturating_sub(first_ms);
        }
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

    let eff_clef = params.clef_override.unwrap_or_else(|| {
        infer_optimal_clef(&timed_history, params.show_tab_staff, params.clef)
    });
    let eff_show_tab = if eff_clef == ClefKind::GrandStaff {
        false
    } else {
        params.show_tab_staff
    };

    let q_score = (if neural_beat_grid.is_usable() {
        build_score_with_beat_grid(
            "",
        &timed_history,
        &voicings,
        &params.tuning_midi,
        params.bpm_override,
        params.time_signature,
            &neural_beat_grid,
        )
    } else {
        build_quantized_score(
            "",
        &timed_history,
        &voicings,
        &params.tuning_midi,
        params.bpm_override,
        params.time_signature,
        )
    })
    .with_instrument(params.midi_program, eff_show_tab, eff_clef);

    let key_obj = q_score.key;
    let alphatex = score_to_alphatex(&q_score, &params.alphatex_tuning, "");

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
        engine_used: if engine_tags.is_empty() {
            if params.mode == "mono" { "YIN".to_string() } else { "Basic Pitch (ONNX)".to_string() }
        } else if !used_muscriptor {
            let base = if params.mode == "mono" { "YIN" } else { "Basic Pitch" };
            format!("{} + {}", base, engine_tags.join(" + "))
        } else {
            engine_tags.join(" + ")
        },
        elapsed_ms: t0.elapsed().as_millis(),
    })
}

pub fn transcribe_stem_offline(
    wav_path: &Path,
    params: &OfflineTranscribeParams,
) -> Result<OfflineTranscribeOutput, String> {
    transcribe_stem_offline_with_engine(wav_path, params, None)
}
