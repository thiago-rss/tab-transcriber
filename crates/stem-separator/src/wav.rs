use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

#[derive(Debug, Clone, Copy)]
pub struct WavWriteReport {
    pub peak_linear: f32,
    pub applied_gain: f32,
}

/// Grava frames estéreo `[L, R]` em WAV PCM 24-bit (144 dB de faixa dinâmica).
/// Aplica remoção leve de DC e proteção de headroom se o sinal ultrapassar 0.99 (zero clipping).
pub fn write_stereo_wav_pcm24(
    path: &Path,
    sample_rate: u32,
    frames: &[[f32; 2]],
) -> Result<WavWriteReport, String> {
    let channels: u16 = 2;
    let bits_per_sample: u16 = 24;
    let block_align: u16 = channels * (bits_per_sample / 8); // 6 bytes/frame
    let byte_rate: u32 = sample_rate * (block_align as u32);
    let data_bytes: u32 = (frames.len() * (block_align as usize))
        .try_into()
        .map_err(|_| "Áudio muito longo para cabeçalho WAV 32-bit.".to_string())?;
    let riff_size: u32 = 36 + data_bytes;

    // Mede DC médio e pico real antes de quantizar
    let mut sum_l = 0.0f64;
    let mut sum_r = 0.0f64;
    for fr in frames {
        sum_l += fr[0] as f64;
        sum_r += fr[1] as f64;
    }
    let inv_n = 1.0 / (frames.len().max(1) as f64);
    let dc_l = (sum_l * inv_n) as f32;
    let dc_r = (sum_r * inv_n) as f32;

    let mut peak = 0.0f32;
    for fr in frames {
        peak = peak.max((fr[0] - dc_l).abs()).max((fr[1] - dc_r).abs());
    }

    // Só atenua se o fluxo Core Audio ultrapassar 0.99 (-0.1 dBFS), evitando clipping digital
    let gain = if peak > 0.99 { 0.985 / peak } else { 1.0 };

    let mut file =
        File::create(path).map_err(|e| format!("Erro ao criar '{}': {}", path.display(), e))?;

    file.write_all(b"RIFF").map_err(|e| e.to_string())?;
    file.write_all(&riff_size.to_le_bytes()).map_err(|e| e.to_string())?;
    file.write_all(b"WAVE").map_err(|e| e.to_string())?;

    file.write_all(b"fmt ").map_err(|e| e.to_string())?;
    file.write_all(&16u32.to_le_bytes()).map_err(|e| e.to_string())?;
    file.write_all(&1u16.to_le_bytes()).map_err(|e| e.to_string())?; // PCM
    file.write_all(&channels.to_le_bytes()).map_err(|e| e.to_string())?;
    file.write_all(&sample_rate.to_le_bytes()).map_err(|e| e.to_string())?;
    file.write_all(&byte_rate.to_le_bytes()).map_err(|e| e.to_string())?;
    file.write_all(&block_align.to_le_bytes()).map_err(|e| e.to_string())?;
    file.write_all(&bits_per_sample.to_le_bytes()).map_err(|e| e.to_string())?;

    file.write_all(b"data").map_err(|e| e.to_string())?;
    file.write_all(&data_bytes.to_le_bytes()).map_err(|e| e.to_string())?;

    let mut pcm = Vec::with_capacity(frames.len() * 6);
    for fr in frames {
        let l = (((fr[0] - dc_l) * gain).clamp(-1.0, 1.0) * 8_388_607.0).round() as i32;
        let r = (((fr[1] - dc_r) * gain).clamp(-1.0, 1.0) * 8_388_607.0).round() as i32;
        let lb = l.to_le_bytes();
        let rb = r.to_le_bytes();
        pcm.extend_from_slice(&[lb[0], lb[1], lb[2], rb[0], rb[1], rb[2]]);
    }

    file.write_all(&pcm).map_err(|e| e.to_string())?;
    Ok(WavWriteReport {
        peak_linear: peak,
        applied_gain: gain,
    })
}

/// Mantido por retrocompatibilidade (grava PCM 16-bit).
pub fn write_stereo_wav_pcm16(
    path: &Path,
    sample_rate: u32,
    frames: &[[f32; 2]],
) -> Result<(), String> {
    let channels: u16 = 2;
    let bits_per_sample: u16 = 16;
    let block_align: u16 = channels * (bits_per_sample / 8);
    let byte_rate: u32 = sample_rate * (block_align as u32);
    let data_bytes: u32 = (frames.len() * (block_align as usize))
        .try_into()
        .map_err(|_| "Áudio muito longo para cabeçalho WAV 32-bit.".to_string())?;
    let riff_size: u32 = 36 + data_bytes;

    let mut file =
        File::create(path).map_err(|e| format!("Erro ao criar '{}': {}", path.display(), e))?;

    file.write_all(b"RIFF").map_err(|e| e.to_string())?;
    file.write_all(&riff_size.to_le_bytes()).map_err(|e| e.to_string())?;
    file.write_all(b"WAVE").map_err(|e| e.to_string())?;

    file.write_all(b"fmt ").map_err(|e| e.to_string())?;
    file.write_all(&16u32.to_le_bytes()).map_err(|e| e.to_string())?;
    file.write_all(&1u16.to_le_bytes()).map_err(|e| e.to_string())?;
    file.write_all(&channels.to_le_bytes()).map_err(|e| e.to_string())?;
    file.write_all(&sample_rate.to_le_bytes()).map_err(|e| e.to_string())?;
    file.write_all(&byte_rate.to_le_bytes()).map_err(|e| e.to_string())?;
    file.write_all(&block_align.to_le_bytes()).map_err(|e| e.to_string())?;
    file.write_all(&bits_per_sample.to_le_bytes()).map_err(|e| e.to_string())?;

    file.write_all(b"data").map_err(|e| e.to_string())?;
    file.write_all(&data_bytes.to_le_bytes()).map_err(|e| e.to_string())?;

    let mut pcm = Vec::with_capacity(frames.len() * 4);
    for fr in frames {
        let l = (fr[0].clamp(-1.0, 1.0) * 32767.0).round() as i16;
        let r = (fr[1].clamp(-1.0, 1.0) * 32767.0).round() as i16;
        pcm.extend_from_slice(&l.to_le_bytes());
        pcm.extend_from_slice(&r.to_le_bytes());
    }

    file.write_all(&pcm).map_err(|e| e.to_string())?;
    Ok(())
}

pub fn read_wav_mono_f32(path: &Path) -> Result<(u32, Vec<f32>), String> {
    let mut file =
        File::open(path).map_err(|e| format!("Erro ao abrir '{}': {}", path.display(), e))?;
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes).map_err(|e| e.to_string())?;

    if bytes.len() < 12 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err(format!("Arquivo '{}' não é um WAV RIFF válido.", path.display()));
    }

    let mut pos = 12usize;
    let mut audio_format: u16 = 0;
    let mut channels: u16 = 0;
    let mut sample_rate: u32 = 0;
    let mut bits_per_sample: u16 = 0;
    let mut data_slice: Option<&[u8]> = None;

    while pos + 8 <= bytes.len() {
        let tag = &bytes[pos..pos + 4];
        let chunk_size = u32::from_le_bytes(bytes[pos + 4..pos + 8].try_into().unwrap()) as usize;
        let body_start = pos + 8;
        let body_end = (body_start + chunk_size).min(bytes.len());

        if tag == b"fmt " && body_end - body_start >= 16 {
            let fmt = &bytes[body_start..body_end];
            audio_format = u16::from_le_bytes(fmt[0..2].try_into().unwrap());
            channels = u16::from_le_bytes(fmt[2..4].try_into().unwrap());
            sample_rate = u32::from_le_bytes(fmt[4..8].try_into().unwrap());
            bits_per_sample = u16::from_le_bytes(fmt[14..16].try_into().unwrap());

            if audio_format == 0xFFFE && fmt.len() >= 26 {
                audio_format = u16::from_le_bytes(fmt[24..26].try_into().unwrap());
            }
        } else if tag == b"data" {
            data_slice = Some(&bytes[body_start..body_end]);
        }

        pos = body_start + chunk_size;
        if chunk_size % 2 == 1 && pos < bytes.len() {
            pos += 1;
        }
    }

    let raw = data_slice.ok_or_else(|| "Chunk 'data' não encontrado no WAV.".to_string())?;
    if channels == 0 || sample_rate == 0 {
        return Err("Cabeçalho 'fmt ' inválido no WAV.".to_string());
    }

    let bytes_per_sample = (bits_per_sample / 8) as usize;
    let frame_stride = bytes_per_sample * (channels as usize);
    if frame_stride == 0 {
        return Err("Bit depth inválido no WAV.".to_string());
    }

    let num_frames = raw.len() / frame_stride;
    let mut mono = Vec::with_capacity(num_frames);
    let inv_ch = 1.0f32 / (channels as f32);

    for f in 0..num_frames {
        let mut sum = 0.0f32;
        let base = f * frame_stride;

        for c in 0..(channels as usize) {
            let off = base + c * bytes_per_sample;
            let s = match (audio_format, bits_per_sample) {
                (1, 16) => {
                    let v = i16::from_le_bytes(raw[off..off + 2].try_into().unwrap());
                    (v as f32) / 32768.0
                }
                (1, 24) => {
                    let b0 = raw[off] as i32;
                    let b1 = (raw[off + 1] as i32) << 8;
                    let b2 = (raw[off + 2] as i8 as i32) << 16;
                    ((b0 | b1 | b2) as f32) / 8_388_608.0
                }
                (1, 32) => {
                    let v = i32::from_le_bytes(raw[off..off + 4].try_into().unwrap());
                    (v as f32) / 2_147_483_648.0
                }
                (3, 32) => f32::from_le_bytes(raw[off..off + 4].try_into().unwrap()),
                _ => {
                    return Err(format!(
                        "Formato WAV não suportado (tag={}, bits={}).",
                        audio_format, bits_per_sample
                    ));
                }
            };
            sum += s;
        }

        mono.push(sum * inv_ch);
    }

    Ok((sample_rate, mono))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn pcm24_roundtrips_and_protects_against_hot_signals() {
        let ts = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
        let path = std::env::temp_dir().join(format!("tab_transcriber_wav24_{}.wav", ts));
        let frames = [[1.5f32, 1.5], [-1.5, -1.5], [0.0, 0.0], [0.0, 0.0]];

        let report = write_stereo_wav_pcm24(&path, 48_000, &frames).expect("escrita WAV 24-bit");
        let (sr, mono) = read_wav_mono_f32(&path).expect("leitura WAV 24-bit");
        let _ = std::fs::remove_file(&path);

        assert_eq!(sr, 48_000);
        assert!(report.applied_gain < 0.7);
        assert!(mono[0] <= 0.99 && mono[0] > 0.97);
        assert!(mono[1] >= -0.99 && mono[1] < -0.97);
    }
}
