use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use stem_separator::{StemActivity, StemBundle};

const MAX_LISTED_SESSIONS: usize = 30;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionManifest {
    pub label: String,
    pub created_at: u64,
    pub engine: String,
    pub source_kind: String, // "snippet" | "file"
    #[serde(default)]
    pub activities: Vec<StemActivity>,
}

#[derive(Debug, Clone, Serialize)]
pub struct StoredStemSessionDto {
    pub id: String,
    pub label: String,
    pub created_at: u64,
    pub session_dir: String,
    pub engine: String,
    pub active_stems: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct UserPreferences {
    pub mode: String,
    pub stereo_mode: String,
    pub auto_level: bool,
    pub enable_hpss: bool,
    pub focus: String,
    pub tuning: String,
    pub neck_region: String,
    pub prefer_closed: bool,
    pub time_signature: String,
    pub bpm: String,
    pub view: String,
    pub stem_quality: String,
    pub stem: String,
    pub playback_speed: f32,
    pub metronome_enabled: bool,
}

impl Default for UserPreferences {
    fn default() -> Self {
        Self {
            mode: "poly".to_string(),
            stereo_mode: "mix".to_string(),
            auto_level: true,
            enable_hpss: false,
            focus: "full".to_string(),
            tuning: "auto".to_string(),
            neck_region: "auto".to_string(),
            prefer_closed: false,
            time_signature: "4/4".to_string(),
            bpm: "auto".to_string(),
            view: "scoretab".to_string(),
            stem_quality: "high".to_string(),
            stem: "guitar".to_string(),
            playback_speed: 1.0,
            metronome_enabled: false,
        }
    }
}

pub fn preferences_path(base_exports: &Path) -> PathBuf {
    base_exports.join("preferences.json")
}

pub fn load_preferences(base_exports: &Path) -> UserPreferences {
    let path = preferences_path(base_exports);
    let Ok(raw) = fs::read_to_string(&path) else {
        return UserPreferences::default();
    };
    serde_json::from_str(&raw).unwrap_or_default()
}

pub fn save_preferences(base_exports: &Path, prefs: &UserPreferences) -> Result<(), String> {
    fs::create_dir_all(base_exports)
        .map_err(|e| format!("Falha ao preparar pasta de preferências: {}", e))?;
    let path = preferences_path(base_exports);
    let json = serde_json::to_string_pretty(prefs)
        .map_err(|e| format!("Falha ao serializar preferências: {}", e))?;
    fs::write(&path, json).map_err(|e| format!("Falha ao gravar '{}': {}", path.display(), e))
}

pub fn save_session_manifest(
    bundle: &StemBundle,
    label: &str,
    source_kind: &str,
) -> Result<(), String> {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);

    let manifest = SessionManifest {
        label: label.to_string(),
        created_at: now,
        engine: bundle.engine.clone(),
        source_kind: source_kind.to_string(),
        activities: bundle.activities.clone(),
    };

    let json = serde_json::to_string_pretty(&manifest)
        .map_err(|e| format!("Falha ao serializar session.json: {}", e))?;
    fs::write(bundle.session_dir.join("session.json"), json)
        .map_err(|e| format!("Falha ao gravar session.json: {}", e))
}

pub fn read_session_manifest(session_dir: &Path) -> Option<SessionManifest> {
    let raw = fs::read_to_string(session_dir.join("session.json")).ok()?;
    serde_json::from_str(&raw).ok()
}

fn dir_timestamp(dir: &Path) -> u64 {
    if let Some(name) = dir.file_name().and_then(|n| n.to_str()) {
        if let Some(ts_str) = name.rsplit('_').next() {
            if let Ok(ts) = ts_str.parse::<u64>() {
                return ts;
            }
        }
    }
    fs::metadata(dir)
        .and_then(|m| m.modified())
        .ok()
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

pub fn list_saved_sessions(base_exports: &Path) -> Vec<StoredStemSessionDto> {
    let stems_root = base_exports.join("stems");
    let Ok(entries) = fs::read_dir(&stems_root) else {
        return Vec::new();
    };

    let mut sessions = Vec::new();

    for entry in entries.flatten() {
        let dir = entry.path();
        if !dir.is_dir() {
            continue;
        }

        let id = dir
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("sessao")
            .to_string();

        // Garante que a pasta realmente contém pelo menos um stem ou mix.wav
        let has_wav = ["guitar", "bass", "piano", "vocals", "other", "drums", "mix"]
            .iter()
            .any(|k| dir.join(format!("{}.wav", k)).exists());
        if !has_wav {
            continue;
        }

        if let Some(manifest) = read_session_manifest(&dir) {
            let active_stems: Vec<String> = manifest
                .activities
                .iter()
                .filter(|a| a.active && a.stem != "drums")
                .map(|a| a.stem.clone())
                .collect();

            sessions.push(StoredStemSessionDto {
                id,
                label: manifest.label,
                created_at: manifest.created_at,
                session_dir: dir.display().to_string(),
                engine: manifest.engine,
                active_stems,
            });
        } else {
            // Compatibilidade com sessões gravadas antes da criação do session.json
            let created_at = dir_timestamp(&dir);
            let kind_prefix = if id.starts_with("file_") {
                "Arquivo"
            } else {
                "Trecho"
            };
            let existing: Vec<String> = ["guitar", "bass", "piano", "vocals", "other"]
                .iter()
                .filter(|k| dir.join(format!("{}.wav", k)).exists())
                .map(|k| (*k).to_string())
                .collect();

            sessions.push(StoredStemSessionDto {
                id: id.clone(),
                label: format!("{} ({})", kind_prefix, id),
                created_at,
                session_dir: dir.display().to_string(),
                engine: "cache local".to_string(),
                active_stems: existing,
            });
        }
    }

    sessions.sort_by(|a, b| b.created_at.cmp(&a.created_at).then_with(|| b.id.cmp(&a.id)));
    sessions.truncate(MAX_LISTED_SESSIONS);
    sessions
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    #[test]
    fn roundtrips_user_preferences_json() {
        let ts = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
        let dir = std::env::temp_dir().join(format!("tab_transcriber_prefs_{}", ts));

        let mut prefs = UserPreferences::default();
        prefs.tuning = "oud_syrian".to_string();
        prefs.time_signature = "7/8".to_string();
        prefs.playback_speed = 0.75;
        prefs.metronome_enabled = true;

        save_preferences(&dir, &prefs).expect("save_preferences");
        let loaded = load_preferences(&dir);
        let _ = fs::remove_dir_all(&dir);

        assert_eq!(loaded, prefs);
    }

    #[test]
    fn saves_manifest_and_lists_sessions_newest_first() {
        let ts = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
        let base = std::env::temp_dir().join(format!("tab_transcriber_sess_{}", ts));
        let s1 = base.join("stems").join("sess_100");
        let s2 = base.join("stems").join("file_200");
        fs::create_dir_all(&s1).unwrap();
        fs::create_dir_all(&s2).unwrap();
        fs::write(s1.join("guitar.wav"), b"dummy").unwrap();
        fs::write(s2.join("piano.wav"), b"dummy").unwrap();

        let bundle2 = StemBundle {
            session_dir: s2.clone(),
            mix_wav: s2.join("mix.wav"),
            stems: BTreeMap::from([("piano".to_string(), s2.join("piano.wav"))]),
            activities: vec![StemActivity {
                stem: "piano".to_string(),
                label: "Piano".to_string(),
                rms_db: -15.0,
                active: true,
            }],
            engine: "demucs-mlx (HQ 24-bit)".to_string(),
            separation_ms: 1200,
        };
        save_session_manifest(&bundle2, "solo_piano.wav", "file").expect("save_manifest");

        let list = list_saved_sessions(&base);
        let _ = fs::remove_dir_all(&base);

        assert_eq!(list.len(), 2);
        assert_eq!(list[0].label, "solo_piano.wav");
        assert_eq!(list[0].active_stems, vec!["piano".to_string()]);
        assert_eq!(list[1].id, "sess_100");
    }
}
