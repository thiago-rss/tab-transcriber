use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Abre o seletor nativo de arquivos do macOS filtrado para formatos de áudio.
/// Retorna `Ok(None)` caso o usuário cancele o diálogo.
pub fn pick_audio_file_macos() -> Result<Option<PathBuf>, String> {
    #[cfg(target_os = "macos")]
    {
        let script = r#"
try
    set chosen to choose file with prompt "Selecione um arquivo de áudio (WAV, MP3, M4A, FLAC, AIFF)" of type {"public.audio"}
    return POSIX path of chosen
on error number -128
    return ""
end try
"#;

        let output = Command::new("osascript")
            .arg("-e")
            .arg(script)
            .output()
            .map_err(|e| format!("Falha ao abrir seletor de arquivos do macOS: {}", e))?;

        if !output.status.success() {
            let err = String::from_utf8_lossy(&output.stderr);
            return Err(format!("Erro no seletor de arquivos: {}", err.trim()));
        }

        let path_str = String::from_utf8_lossy(&output.stdout).trim().to_string();
        if path_str.is_empty() {
            return Ok(None);
        }
        Ok(Some(PathBuf::from(path_str)))
    }

    #[cfg(not(target_os = "macos"))]
    {
        Err("Seletor nativo disponível apenas no macOS.".to_string())
    }
}

/// Converte qualquer áudio suportado pelo macOS (WAV, MP3, M4A, FLAC, AIFF, CAF)
/// para WAV PCM 24-bit limpo em `dst_wav` usando `afconvert` (Core Audio) com fallback `ffmpeg`.
pub fn prepare_mix_wav_24bit(src: &Path, dst_wav: &Path) -> Result<String, String> {
    if !src.exists() {
        return Err(format!("Arquivo de áudio não encontrado: {}", src.display()));
    }

    // 1. Conversão nativa via Core Audio (sem dependências externas)
    #[cfg(target_os = "macos")]
    {
        let af_out = Command::new("/usr/bin/afconvert")
            .arg("-f")
            .arg("WAVE")
            .arg("-d")
            .arg("LEI24")
            .arg(src)
            .arg(dst_wav)
            .output();

        if let Ok(out) = af_out {
            if out.status.success()
                && dst_wav
                    .metadata()
                    .map(|m| m.len() > 44)
                    .unwrap_or(false)
            {
                return Ok("Core Audio (WAV 24-bit)".to_string());
            }
        }
    }

    // 2. Fallback via ffmpeg (checa caminhos completos do Homebrew para funcionar fora do terminal)
    let ffmpeg_bin = ["/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg"]
        .into_iter()
        .find(|p| Path::new(p).is_file())
        .unwrap_or("ffmpeg");
    let ff_out = Command::new(ffmpeg_bin)
        .arg("-y")
        .arg("-i")
        .arg(src)
        .arg("-acodec")
        .arg("pcm_s24le")
        .arg(dst_wav)
        .output();

    if let Ok(out) = ff_out {
        if out.status.success()
            && dst_wav
                .metadata()
                .map(|m| m.len() > 44)
                .unwrap_or(false)
        {
            return Ok("FFmpeg (WAV 24-bit)".to_string());
        }
    }

    // 3. Se já for WAV válido, copia diretamente
    let is_wav = src
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.eq_ignore_ascii_case("wav"))
        .unwrap_or(false);

    if is_wav {
        fs::copy(src, dst_wav)
            .map_err(|e| format!("Falha ao copiar arquivo WAV: {}", e))?;
        return Ok("WAV original".to_string());
    }

    Err(format!(
        "Não foi possível converter '{}'. Formatos nativos: WAV, MP3, M4A, FLAC, AIFF.",
        src.display()
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_missing_audio_file_cleanly() {
        let missing = PathBuf::from("/tmp/nao_existe_tab_transcriber_12345.mp3");
        let dst = PathBuf::from("/tmp/out_tab_transcriber_12345.wav");
        let res = prepare_mix_wav_24bit(&missing, &dst);
        assert!(res.is_err());
    }
}
