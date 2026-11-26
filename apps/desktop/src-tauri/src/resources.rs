use std::path::PathBuf;
use std::sync::OnceLock;

use serde::Serialize;

static TAURI_RESOURCE_DIR: OnceLock<PathBuf> = OnceLock::new();

#[derive(Debug, Clone, Serialize)]
pub struct RuntimeDiagnosticsDto {
    pub runtime_mode: String,
    pub resource_dir: Option<String>,
    pub model_path: Option<String>,
    pub stems_script: Option<String>,
    pub python_bin: Option<String>,
}

/// Registra o diretório de recursos resolvido pelo Tauri v2 no startup (`.setup()`).
pub fn register_resource_dir(dir: PathBuf) {
    let _ = TAURI_RESOURCE_DIR.set(dir);
}

pub fn dev_workspace_root() -> PathBuf {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest_dir
        .join("../../..")
        .canonicalize()
        .unwrap_or_else(|_| manifest_dir.join("../../.."))
}

pub fn bundle_resources_dir() -> Option<PathBuf> {
    if let Some(dir) = TAURI_RESOURCE_DIR.get() {
        if dir.is_dir() {
            return Some(dir.clone());
        }
    }
    let exe = std::env::current_exe().ok()?;
    let res = exe.parent()?.parent()?.join("Resources");
    if res.is_dir() {
        Some(res)
    } else {
        None
    }
}

pub fn is_bundled_runtime() -> bool {
    let Some(res) = bundle_resources_dir() else {
        return false;
    };
    res.join("models/nmp.onnx").is_file()
        || res.join("scripts/separate_stems.py").is_file()
        || res.join("_up_/_up_/_up_/models/nmp.onnx").is_file()
}

/// Resolve o diretório raiz de recursos para o separador de stems (Bundle primeiro, Dev como fallback).
pub fn resolve_stems_root() -> PathBuf {
    if let Some(res) = bundle_resources_dir() {
        if res.join("scripts/separate_stems.py").is_file()
            || res.join("separate_stems.py").is_file()
            || res.join("_up_/_up_/_up_/scripts/separate_stems.py").is_file()
        {
            return res;
        }
    }

    let dev_root = dev_workspace_root();
    if dev_root.join("scripts/separate_stems.py").is_file() {
        return dev_root;
    }

    if let Ok(home) = std::env::var("HOME") {
        let home_ws = PathBuf::from(home).join("tab-transcriber");
        if home_ws.join("scripts/separate_stems.py").is_file() {
            return home_ws;
        }
    }

    dev_root
}

/// Resolve o modelo Basic Pitch (`nmp.onnx`) de forma portável entre dev, bundle `.app` e App Support.
pub fn resolve_model_path() -> Result<PathBuf, String> {
    let mut candidates: Vec<PathBuf> = Vec::new();

    if let Ok(env_model) = std::env::var("TAB_TRANSCRIBER_MODEL") {
        if !env_model.trim().is_empty() {
            candidates.push(PathBuf::from(env_model));
        }
    }

    if let Some(res) = bundle_resources_dir() {
        candidates.push(res.join("models/nmp.onnx"));
        candidates.push(res.join("nmp.onnx"));
        candidates.push(res.join("_up_/_up_/_up_/models/nmp.onnx"));
    }

    let dev_root = dev_workspace_root();
    candidates.push(dev_root.join("models/nmp.onnx"));

    if let Ok(home) = std::env::var("HOME") {
        let home_path = PathBuf::from(home);
        candidates.push(home_path.join("Library/Application Support/TabTranscriber/models/nmp.onnx"));
        candidates.push(home_path.join("tab-transcriber/models/nmp.onnx"));
    }

    candidates.push(PathBuf::from("models/nmp.onnx"));
    candidates.push(PathBuf::from("../../models/nmp.onnx"));
    candidates.push(PathBuf::from("../../../models/nmp.onnx"));

    for path in candidates {
        if path.is_file() {
            return Ok(path);
        }
    }

    Err(
        "Modelo ONNX 'nmp.onnx' não encontrado no bundle nem em ~/tab-transcriber/models/nmp.onnx."
            .to_string(),
    )
}

pub fn collect_diagnostics() -> RuntimeDiagnosticsDto {
    let stems_root = resolve_stems_root();
    let (py, sc) = stem_separator::describe_stems_runtime(&stems_root);

    RuntimeDiagnosticsDto {
        runtime_mode: if is_bundled_runtime() {
            "bundle".to_string()
        } else {
            "dev".to_string()
        },
        resource_dir: bundle_resources_dir().map(|p| p.display().to_string()),
        model_path: resolve_model_path().ok().map(|p| p.display().to_string()),
        stems_script: sc.map(|p| p.display().to_string()),
        python_bin: py.map(|p| p.display().to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_stems_script_in_dev_workspace() {
        let root = resolve_stems_root();
        assert!(
            root.join("scripts/separate_stems.py").is_file(),
            "separate_stems.py deveria existir em '{}'",
            root.display()
        );
    }

    #[test]
    fn diagnostics_reports_valid_runtime_mode() {
        let diag = collect_diagnostics();
        assert!(diag.runtime_mode == "dev" || diag.runtime_mode == "bundle");
        assert!(diag.stems_script.is_some());
    }
}
