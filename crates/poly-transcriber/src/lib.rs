use std::collections::BTreeSet;
use std::f64::consts::PI;
use std::path::Path;

use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Tensor;

pub const MODEL_SAMPLE_RATE: f64 = 22_050.0;
pub const MODEL_WINDOW_SAMPLES: usize = 43_844;
pub const FFT_HOP: usize = 256;
pub const FRAME_DURATION_MS: f64 = (FFT_HOP as f64 / MODEL_SAMPLE_RATE) * 1000.0;
pub const MIDI_OFFSET: u8 = 21;
pub const CONTOUR_BINS_PER_SEMITONE: usize = 3;

#[derive(Debug, Clone)]
pub struct PolyNoteEvent {
    pub midi_note: u8,
    pub start_frame: usize,
    pub end_frame: usize,
    pub confidence: f32,
    pub onset_confidence: f32,
    pub bend_semitones: f32,
    pub median_cents: f32,
    pub timbral_warmth: f32,
}

#[derive(Debug, Clone)]
pub struct ChordEvent {
    pub frame: usize,
    pub duration_frames: usize,
    pub notes: Vec<u8>,
    pub max_bend_semitones: f32,
    pub timbral_warmth: f32,
}

#[derive(Debug, Clone)]
pub struct WindowTranscription {
    pub chords: Vec<ChordEvent>,
    pub estimated_a4_hz: Option<f32>,
    pub pruned_overtones: usize,
}

#[derive(Debug, Clone)]
pub struct DecodeConfig {
    pub onset_threshold: f32,
    pub frame_threshold: f32,
    pub sustain_threshold: f32,
    pub legato_frame_threshold: f32,
    pub min_note_frames: usize,
    pub merge_gap_frames: usize,
    pub chord_window_frames: usize,
    pub min_midi: u8,
    pub max_midi: u8,
    pub max_polyphony: usize,
    pub relative_salience_ratio: f32,
    pub enable_overtone_pruning: bool,
    pub enable_semitone_bleed_pruning: bool,
    pub overtone_octave_ratio: f32,
    pub overtone_fifth_ratio: f32,
    pub bend_min_cents: f32,
}

impl Default for DecodeConfig {
    fn default() -> Self {
        Self {
            onset_threshold: 0.48,
            frame_threshold: 0.34,
            sustain_threshold: 0.24,
            legato_frame_threshold: 0.64,
            min_note_frames: 6,
            merge_gap_frames: 3,
            chord_window_frames: 4,
            min_midi: 28,
            max_midi: 96,
            max_polyphony: 6,
            relative_salience_ratio: 0.54,
            enable_overtone_pruning: true,
            enable_semitone_bleed_pruning: true,
            overtone_octave_ratio: 0.78,
            overtone_fifth_ratio: 0.72,
            bend_min_cents: 70.0,
        }
    }
}

// ============================================================================
// RESAMPLER POLYPHASE WINDOWED-SINC (64 TAPS, BLACKMAN-HARRIS, ZERO ALIASING)
// ============================================================================
const SINC_TAPS: usize = 64;
const SINC_HALF: usize = SINC_TAPS / 2;
const SINC_PHASES: usize = 256;

pub struct SincResampler {
    step: f64,
    read_pos: f64,
    buf: Vec<f32>,
    table: Vec<[f32; SINC_TAPS]>,
}

pub type LinearResampler = SincResampler;

impl SincResampler {
    pub fn new(input_rate: f64, output_rate: f64) -> Self {
        let step = (input_rate / output_rate).max(0.05);
        let fc = 0.44 * (output_rate / input_rate).min(1.0);

        let mut table = Vec::with_capacity(SINC_PHASES + 1);
        for p in 0..=SINC_PHASES {
            let frac = (p as f64) / (SINC_PHASES as f64);
            let mut kernel = [0.0f32; SINC_TAPS];
            let mut sum = 0.0f64;

            for t in 0..SINC_TAPS {
                let k = (t as isize) - ((SINC_HALF as isize) - 1);
                let x = (k as f64) - frac;
                let u = ((x + SINC_HALF as f64) / (SINC_TAPS as f64)).clamp(0.0, 1.0);

                let w = 0.35875
                    - 0.48829 * (2.0 * PI * u).cos()
                    + 0.14128 * (4.0 * PI * u).cos()
                    - 0.01168 * (6.0 * PI * u).cos();

                let arg = 2.0 * PI * fc * x;
                let sinc = if arg.abs() < 1e-9 {
                    1.0
                } else {
                    arg.sin() / arg
                };

                let val = 2.0 * fc * sinc * w;
                kernel[t] = val as f32;
                sum += val;
            }

            if sum.abs() > 1e-9 {
                let inv = (1.0 / sum) as f32;
                for v in &mut kernel {
                    *v *= inv;
                }
            }
            table.push(kernel);
        }

        Self {
            step,
            read_pos: SINC_HALF as f64,
            buf: vec![0.0; SINC_HALF],
            table,
        }
    }

    pub fn push(&mut self, input: &[f32], output: &mut Vec<f32>) {
        if input.is_empty() {
            return;
        }
        self.buf.extend_from_slice(input);

        while (self.read_pos.floor() as usize) + SINC_HALF < self.buf.len() {
            let center = self.read_pos.floor() as usize;
            let frac = self.read_pos - (center as f64);
            let pf = frac * (SINC_PHASES as f64);
            let p0 = (pf.floor() as usize).min(SINC_PHASES - 1);
            let alpha = (pf - (p0 as f64)) as f32;

            let k0 = &self.table[p0];
            let k1 = &self.table[p0 + 1];
            let base = center - (SINC_HALF - 1);

            let mut acc = 0.0f32;
            for t in 0..SINC_TAPS {
                let coeff = k0[t] + alpha * (k1[t] - k0[t]);
                acc += self.buf[base + t] * coeff;
            }
            output.push(acc);
            self.read_pos += self.step;
        }

        let pos_floor = self.read_pos.floor() as usize;
        if pos_floor > SINC_HALF + 512 {
            let drop_n = pos_floor - SINC_HALF;
            self.buf.drain(..drop_n);
            self.read_pos -= drop_n as f64;
        }
    }

    pub fn flush(&mut self, output: &mut Vec<f32>) {
        let pad = vec![0.0f32; SINC_HALF];
        self.push(&pad, output);
    }
}

fn estimate_slice_warmth(samples: &[f32], sample_rate: f64, midi_note: u8) -> f32 {
    if samples.len() < 32 {
        return 0.5;
    }
    let f0 = 440.0 * 2.0f64.powf(((midi_note as f64) - 69.0) / 12.0);
    let mut energy = 0.0f64;
    let mut diff_energy = 0.0f64;
    for pair in samples.windows(2) {
        let x0 = pair[0] as f64;
        let d = (pair[1] as f64) - x0;
        energy += x0 * x0;
        diff_energy += d * d;
    }
    if energy <= 1e-12 || f0 <= 20.0 {
        return 0.5;
    }
    let arg = (0.5 * (diff_energy / energy).sqrt()).clamp(0.0, 1.0);
    let centroid = (sample_rate / PI) * arg.asin();
    let ratio = (centroid / f0) as f32;
    ((3.6 - ratio) / 2.2).clamp(0.0, 1.0)
}

pub struct BasicPitchEngine {
    session: Session,
}

impl BasicPitchEngine {
    pub fn load(model_path: impl AsRef<Path>) -> Result<Self, Box<dyn std::error::Error>> {
        let env = ort::init().build()?;
        let session = Session::builder(&env)?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(4)?
            .commit_from_file(model_path)?;

        Ok(Self { session })
    }

    pub fn transcribe_window(
        &mut self,
        window_22k: &[f32],
        cfg: &DecodeConfig,
    ) -> Result<WindowTranscription, Box<dyn std::error::Error>> {
        if window_22k.len() != MODEL_WINDOW_SAMPLES {
            return Err(format!(
                "Tamanho de janela inválido: esperado {}, recebido {}",
                MODEL_WINDOW_SAMPLES,
                window_22k.len()
            )
            .into());
        }

        let input =
            Tensor::<f32>::from_array(([1usize, MODEL_WINDOW_SAMPLES, 1], window_22k.to_vec()))?;
        let outputs = self.session.run(ort::inputs![input])?;

        let (note_shape, note_data) = outputs
            .get("StatefulPartitionedCall:1")
            .ok_or("Saída 'StatefulPartitionedCall:1' (note) não encontrada no modelo")?
            .try_extract_tensor::<f32>()?;

        let (onset_shape, onset_data) = outputs
            .get("StatefulPartitionedCall:2")
            .ok_or("Saída 'StatefulPartitionedCall:2' (onset) não encontrada no modelo")?
            .try_extract_tensor::<f32>()?;

        let n_frames = note_shape[1] as usize;
        let n_bins = note_shape[2] as usize;

        if onset_shape[1] as usize != n_frames || onset_shape[2] as usize != n_bins {
            return Err("Dimensões divergentes entre tensores note e onset".into());
        }

        let smoothed_notes = temporal_smooth_notes(note_data, n_frames, n_bins);
        let mut notes =
            decode_polyphonic_notes(&smoothed_notes, onset_data, n_frames, n_bins, cfg);

        for ev in &mut notes {
            let s_start = (ev.start_frame * FFT_HOP).min(window_22k.len());
            let s_end = (ev.end_frame * FFT_HOP).min(window_22k.len());
            if s_end > s_start + 32 {
                ev.timbral_warmth =
                    estimate_slice_warmth(&window_22k[s_start..s_end], MODEL_SAMPLE_RATE, ev.midi_note);
            }
        }

        let estimated_a4_hz = if let Some(contour_val) = outputs.get("StatefulPartitionedCall:0") {
            if let Ok((c_shape, c_data)) = contour_val.try_extract_tensor::<f32>() {
                let c_frames = c_shape[1] as usize;
                let c_bins = c_shape[2] as usize;
                analyze_contour_for_events(&mut notes, c_data, c_frames, c_bins, cfg)
            } else {
                None
            }
        } else {
            None
        };

        let mut pruned_overtones = 0;
        if cfg.enable_semitone_bleed_pruning {
            pruned_overtones += prune_semitone_bleed(&mut notes);
        }
        if cfg.enable_overtone_pruning {
            pruned_overtones += prune_phantom_overtones(&mut notes, cfg);
        }

        let chords = group_notes_into_chords(&notes, cfg);
        Ok(WindowTranscription {
            chords,
            estimated_a4_hz,
            pruned_overtones,
        })
    }

    pub fn transcribe_audio_full(
        &mut self,
        audio_22k: &[f32],
        cfg: &DecodeConfig,
    ) -> Result<WindowTranscription, Box<dyn std::error::Error>> {
        if audio_22k.is_empty() {
            return Ok(WindowTranscription {
                chords: Vec::new(),
                estimated_a4_hz: None,
                pruned_overtones: 0,
            });
        }

        let hop_frames: usize = 86;
        let hop_samples: usize = hop_frames * FFT_HOP;

        let mut full_notes: Vec<f32> = Vec::new();
        let mut full_onsets: Vec<f32> = Vec::new();
        let mut full_contour: Vec<f32> = Vec::new();
        let mut n_note_bins: usize = 0;
        let mut n_contour_bins: usize = 0;
        let mut total_stitched_frames: usize = 0;

        let mut start_sample = 0usize;
        let mut win_idx = 0usize;

        while start_sample < audio_22k.len() {
            let end_sample = (start_sample + MODEL_WINDOW_SAMPLES).min(audio_22k.len());
            let valid_samples = end_sample - start_sample;
            let mut window_buf = vec![0.0f32; MODEL_WINDOW_SAMPLES];
            window_buf[..valid_samples].copy_from_slice(&audio_22k[start_sample..end_sample]);

            let input =
                Tensor::<f32>::from_array(([1usize, MODEL_WINDOW_SAMPLES, 1], window_buf))?;
            let outputs = self.session.run(ort::inputs![input])?;

            let (note_shape, note_data) = outputs
                .get("StatefulPartitionedCall:1")
                .ok_or("Saída 'StatefulPartitionedCall:1' (note) não encontrada")?
                .try_extract_tensor::<f32>()?;
            let (onset_shape, onset_data) = outputs
                .get("StatefulPartitionedCall:2")
                .ok_or("Saída 'StatefulPartitionedCall:2' (onset) não encontrada")?
                .try_extract_tensor::<f32>()?;

            let n_frames = note_shape[1] as usize;
            let bins = note_shape[2] as usize;
            if onset_shape[1] as usize != n_frames || onset_shape[2] as usize != bins {
                return Err("Dimensões divergentes entre tensores note e onset".into());
            }
            n_note_bins = bins;

            let contour_extracted = outputs
                .get("StatefulPartitionedCall:0")
                .and_then(|v| v.try_extract_tensor::<f32>().ok());
            if let Some((c_shape, _)) = &contour_extracted {
                n_contour_bins = c_shape[2] as usize;
            }

            let is_first = win_idx == 0;
            let is_last = end_sample >= audio_22k.len();
            let valid_frames = ((valid_samples + FFT_HOP - 1) / FFT_HOP).min(n_frames);

            let keep_start = if is_first { 0 } else { (hop_frames / 2).min(n_frames) };
            let keep_end = if is_last {
                valid_frames
            } else {
                ((hop_frames / 2) + hop_frames).min(n_frames)
            };

            if keep_end > keep_start {
                let row_start = keep_start * bins;
                let row_end = keep_end * bins;
                full_notes.extend_from_slice(&note_data[row_start..row_end]);
                full_onsets.extend_from_slice(&onset_data[row_start..row_end]);

                if let Some((c_shape, c_data)) = &contour_extracted {
                    let cb = c_shape[2] as usize;
                    let cf = c_shape[1] as usize;
                    let c_end = keep_end.min(cf);
                    let c_start = keep_start.min(c_end);
                    full_contour.extend_from_slice(&c_data[c_start * cb..c_end * cb]);
                }

                total_stitched_frames += keep_end - keep_start;
            }

            if is_last {
                break;
            }
            start_sample += hop_samples;
            win_idx += 1;
        }

        if total_stitched_frames == 0 || n_note_bins == 0 {
            return Ok(WindowTranscription {
                chords: Vec::new(),
                estimated_a4_hz: None,
                pruned_overtones: 0,
            });
        }

        let smoothed_notes = temporal_smooth_notes(&full_notes, total_stitched_frames, n_note_bins);
        let mut notes = decode_polyphonic_notes(
            &smoothed_notes,
            &full_onsets,
            total_stitched_frames,
            n_note_bins,
            cfg,
        );

        for ev in &mut notes {
            let s_start = (ev.start_frame * FFT_HOP).min(audio_22k.len());
            let s_end = (ev.end_frame * FFT_HOP).min(audio_22k.len());
            if s_end > s_start + 32 {
                ev.timbral_warmth =
                    estimate_slice_warmth(&audio_22k[s_start..s_end], MODEL_SAMPLE_RATE, ev.midi_note);
            }
        }

        let estimated_a4_hz =
            if n_contour_bins > 0 && full_contour.len() >= total_stitched_frames * n_contour_bins {
                analyze_contour_for_events(
                    &mut notes,
                    &full_contour,
                    total_stitched_frames,
                    n_contour_bins,
                    cfg,
                )
            } else {
                None
            };

        let mut pruned_overtones = 0;
        if cfg.enable_semitone_bleed_pruning {
            pruned_overtones += prune_semitone_bleed(&mut notes);
        }
        if cfg.enable_overtone_pruning {
            pruned_overtones += prune_phantom_overtones(&mut notes, cfg);
        }

        let chords = group_notes_into_chords(&notes, cfg);
        Ok(WindowTranscription {
            chords,
            estimated_a4_hz,
            pruned_overtones,
        })
    }
}

/// Suavização mediana temporal de 3 frames no tensor `note` (elimina picos espúrios de 1 frame sem atrasar ataques).
fn temporal_smooth_notes(notes: &[f32], n_frames: usize, n_bins: usize) -> Vec<f32> {
    if n_frames < 3 || n_bins == 0 {
        return notes.to_vec();
    }
    let mut out = notes.to_vec();
    for t in 1..(n_frames - 1) {
        let prev = (t - 1) * n_bins;
        let curr = t * n_bins;
        let next = (t + 1) * n_bins;
        for p in 0..n_bins {
            let mut a = notes[prev + p];
            let mut b = notes[curr + p];
            let mut c = notes[next + p];
            if a > b {
                std::mem::swap(&mut a, &mut b);
            }
            if b > c {
                std::mem::swap(&mut b, &mut c);
            }
            if a > b {
                std::mem::swap(&mut a, &mut b);
            }
            out[curr + p] = b;
        }
    }
    out
}

fn decode_polyphonic_notes(
    notes: &[f32],
    onsets: &[f32],
    n_frames: usize,
    n_bins: usize,
    cfg: &DecodeConfig,
) -> Vec<PolyNoteEvent> {
    let mut events = Vec::new();
    let idx = |t: usize, p: usize| -> usize { t * n_bins + p };
    let sustain_thr = cfg.sustain_threshold.min(cfg.frame_threshold);

    for p in 0..n_bins {
        let midi = MIDI_OFFSET + p as u8;
        if midi < cfg.min_midi || midi > cfg.max_midi {
            continue;
        }

        let mut pitch_events: Vec<PolyNoteEvent> = Vec::new();
        let mut t = 1usize;

        while t + 1 < n_frames {
            let onset_val = onsets[idx(t, p)];
            let note_val = notes[idx(t, p)];
            let prev_note = notes[idx(t - 1, p)];

            let is_onset_peak = onset_val >= cfg.onset_threshold
                && onset_val >= onsets[idx(t - 1, p)]
                && onset_val > onsets[idx(t + 1, p)]
                && note_val >= cfg.frame_threshold;

            let is_legato_entry = !is_onset_peak
                && note_val >= cfg.legato_frame_threshold
                && prev_note < cfg.legato_frame_threshold;

            if is_onset_peak || is_legato_entry {
                let start = t;
                let mut end = t + 1;
                let mut peak_conf = note_val;
                let mut quiet_streak = 0usize;

                while end < n_frames {
                    let frame_val = notes[idx(end, p)];
                    if frame_val > peak_conf {
                        peak_conf = frame_val;
                    }
                    // Histerese: exige nível mais alto para abrir a nota e nível menor para sustentá-la
                    if frame_val < sustain_thr {
                        quiet_streak += 1;
                        if quiet_streak >= 3 {
                            end = end.saturating_sub(2);
                            break;
                        }
                    } else {
                        quiet_streak = 0;
                    }
                    if end > start + 3 && onsets[idx(end, p)] >= cfg.onset_threshold {
                        break;
                    }
                    end += 1;
                }

                let min_len = if is_onset_peak {
                    cfg.min_note_frames
                } else {
                    cfg.min_note_frames + 2
                };

                if end.saturating_sub(start) >= min_len {
                    let eff_onset = if is_onset_peak {
                        onset_val
                    } else {
                        peak_conf * 0.78
                    };
                    let merged = if let Some(prev) = pitch_events.last_mut() {
                        if !is_onset_peak && start <= prev.end_frame + cfg.merge_gap_frames {
                            prev.end_frame = prev.end_frame.max(end);
                            prev.confidence = prev.confidence.max(peak_conf);
                            true
                        } else {
                            false
                        }
                    } else {
                        false
                    };

                    if !merged {
                        pitch_events.push(PolyNoteEvent {
                            midi_note: midi,
                            start_frame: start,
                            end_frame: end,
                            confidence: peak_conf,
                            onset_confidence: eff_onset,
                            bend_semitones: 0.0,
                            median_cents: 0.0,
                            timbral_warmth: 0.5,
                        });
                    }
                }
                t = end.max(t + 1);
            } else {
                t += 1;
            }
        }

        events.extend(pitch_events);
    }

    events.sort_by_key(|e| (e.start_frame, e.midi_note));
    events
}

fn median_f32(vals: &mut [f32]) -> Option<f32> {
    if vals.is_empty() {
        return None;
    }
    vals.sort_by(|a, b| a.total_cmp(b));
    let mid = vals.len() / 2;
    if vals.len() % 2 == 1 {
        Some(vals[mid])
    } else {
        Some((vals[mid - 1] + vals[mid]) * 0.5)
    }
}

pub fn analyze_contour_for_events(
    events: &mut [PolyNoteEvent],
    contour: &[f32],
    n_frames: usize,
    n_contour_bins: usize,
    cfg: &DecodeConfig,
) -> Option<f32> {
    if n_frames == 0 || n_contour_bins == 0 {
        return None;
    }

    let cents_per_bin = 100.0 / (CONTOUR_BINS_PER_SEMITONE as f32);
    let mut steady_tuning_samples: Vec<f32> = Vec::new();

    for ev in events.iter_mut() {
        let center_bin =
            (ev.midi_note.saturating_sub(MIDI_OFFSET) as usize) * CONTOUR_BINS_PER_SEMITONE;
        if center_bin >= n_contour_bins {
            continue;
        }

        let lo_bin = center_bin.saturating_sub(2);
        let hi_bin = (center_bin + 6).min(n_contour_bins - 1);

        let f_start = ev.start_frame.min(n_frames);
        let f_end = ev.end_frame.min(n_frames);
        let mut frame_cents: Vec<f32> = Vec::with_capacity(f_end.saturating_sub(f_start));

        for t in f_start..f_end {
            let row = t * n_contour_bins;
            let mut best_b = center_bin;
            let mut best_val = 0.0f32;

            for b in lo_bin..=hi_bin {
                let v = contour[row + b];
                if v > best_val {
                    best_val = v;
                    best_b = b;
                }
            }

            if best_val < 0.15 {
                continue;
            }

            let refined_bin = if best_b > lo_bin && best_b < hi_bin {
                let s0 = contour[row + best_b - 1];
                let s1 = contour[row + best_b];
                let s2 = contour[row + best_b + 1];
                let denom = 2.0 * (2.0 * s1 - s2 - s0);
                if denom.abs() > 1e-6 {
                    let delta = ((s2 - s0) / denom).clamp(-0.5, 0.5);
                    (best_b as f32) + delta
                } else {
                    best_b as f32
                }
            } else {
                best_b as f32
            };

            frame_cents.push((refined_bin - (center_bin as f32)) * cents_per_bin);
        }

        if frame_cents.len() < 3 {
            continue;
        }

        let mut attack_slice: Vec<f32> = frame_cents.iter().take(3).copied().collect();
        let attack_cents = median_f32(&mut attack_slice).unwrap_or(0.0);

        let mut sorted = frame_cents.clone();
        sorted.sort_by(|a, b| a.total_cmp(b));
        let p90_idx = ((sorted.len() - 1) * 9) / 10;
        let peak_cents = sorted[p90_idx];
        let med_cents = median_f32(&mut sorted).unwrap_or(0.0);

        ev.median_cents = med_cents;

        let bend_rise = peak_cents - attack_cents;
        if bend_rise >= cfg.bend_min_cents {
            let semitones = ((bend_rise / 100.0) * 2.0).round() * 0.5;
            ev.bend_semitones = semitones.clamp(0.5, 2.0);
        } else if med_cents.abs() <= 45.0 {
            steady_tuning_samples.push(med_cents);
        }
    }

    let global_cents = median_f32(&mut steady_tuning_samples)?;
    let estimated_a4 = 440.0 * 2.0f32.powf(global_cents / 1200.0);
    Some(estimated_a4.clamp(415.0, 466.0))
}

/// Remove vazamentos espectrais de semitom adjacente (`±1` semitom) causados por vibrato, distorção ou largura de banda CQT.
pub fn prune_semitone_bleed(events: &mut Vec<PolyNoteEvent>) -> usize {
    if events.len() < 2 {
        return 0;
    }

    let mut removed = vec![false; events.len()];
    for i in 0..events.len() {
        if removed[i] {
            continue;
        }
        for j in (i + 1)..events.len() {
            if removed[j] {
                continue;
            }
            let a = &events[i];
            let b = &events[j];
            if a.midi_note.abs_diff(b.midi_note) != 1 {
                continue;
            }
            let start_close = a.start_frame.abs_diff(b.start_frame) <= 4;
            let overlap = a.end_frame.min(b.end_frame) > a.start_frame.max(b.start_frame) + 1;
            if !start_close && !overlap {
                continue;
            }

            let score_a = a.confidence * 0.6 + a.onset_confidence * 0.4;
            let score_b = b.confidence * 0.6 + b.onset_confidence * 0.4;

            if score_b < score_a * 0.76 && b.onset_confidence < 0.72 {
                removed[j] = true;
            } else if score_a < score_b * 0.76 && a.onset_confidence < 0.72 {
                removed[i] = true;
                break;
            }
        }
    }

    let pruned = removed.iter().filter(|&&r| r).count();
    let mut idx = 0usize;
    events.retain(|_| {
        let keep = !removed[idx];
        idx += 1;
        keep
    });
    pruned
}

/// Remove harmônicos fantasma acima (+12, +19, +24, +28) e sub-oitavas fantasma abaixo (-12).
pub fn prune_phantom_overtones(events: &mut Vec<PolyNoteEvent>, cfg: &DecodeConfig) -> usize {
    if events.len() < 2 {
        return 0;
    }

    events.sort_by_key(|e| (e.midi_note, e.start_frame));
    let mut removed = vec![false; events.len()];

    for i in 0..events.len() {
        if removed[i] {
            continue;
        }
        let low = &events[i];

        for j in (i + 1)..events.len() {
            if removed[j] {
                continue;
            }
            let high = &events[j];
            let interval = high.midi_note.saturating_sub(low.midi_note);
            if interval > 28 {
                break;
            }

            let overlap_start = low.start_frame.max(high.start_frame);
            let overlap_end = low.end_frame.min(high.end_frame);
            let overlaps = overlap_end > overlap_start + 1
                || high.start_frame.abs_diff(low.start_frame) <= 3;
            if !overlaps {
                continue;
            }

            // 1. Sub-oitava fantasma (nota mais grave muito mais fraca que a nota 1 oitava acima)
            if interval == 12
                && low.onset_confidence < high.onset_confidence * 0.56
                && low.confidence < high.confidence * 0.65
            {
                removed[i] = true;
                break;
            }

            // 2. Sobretons fantasma acima da fundamental
            let is_phantom_overtone = match interval {
                12 => {
                    high.onset_confidence < low.onset_confidence * cfg.overtone_octave_ratio
                        && high.confidence < low.confidence * 0.90
                }
                19 => high.onset_confidence < low.onset_confidence * cfg.overtone_fifth_ratio,
                24 | 28 => high.onset_confidence < low.onset_confidence * 0.65,
                _ => false,
            };

            if is_phantom_overtone {
                removed[j] = true;
            }
        }
    }

    let pruned = removed.iter().filter(|&&r| r).count();
    let mut idx = 0usize;
    events.retain(|_| {
        let keep = !removed[idx];
        idx += 1;
        keep
    });
    events.sort_by_key(|e| (e.start_frame, e.midi_note));
    pruned
}

fn group_notes_into_chords(events: &[PolyNoteEvent], cfg: &DecodeConfig) -> Vec<ChordEvent> {
    let mut groups: Vec<Vec<&PolyNoteEvent>> = Vec::new();

    for ev in events {
        if let Some(last_g) = groups.last_mut() {
            let base_frame = last_g[0].start_frame;
            if ev.start_frame <= base_frame + cfg.chord_window_frames {
                last_g.push(ev);
                continue;
            }
        }
        groups.push(vec![ev]);
    }

    let max_voices = cfg.max_polyphony.max(1);
    let mut out: Vec<ChordEvent> = Vec::with_capacity(groups.len());

    for mut g in groups {
        let peak_conf = g.iter().map(|e| e.confidence).fold(0.0f32, f32::max);
        let min_keep_conf = peak_conf * cfg.relative_salience_ratio;

        g.retain(|e| e.confidence >= min_keep_conf);
        g.sort_by(|a, b| b.confidence.total_cmp(&a.confidence));
        g.truncate(max_voices);

        if g.is_empty() {
            continue;
        }

        let start_frame = g.iter().map(|e| e.start_frame).min().unwrap_or(0);
        let duration_frames = g
            .iter()
            .map(|e| e.end_frame.saturating_sub(e.start_frame).max(1))
            .max()
            .unwrap_or(1);
        let max_bend = g.iter().map(|e| e.bend_semitones).fold(0.0f32, f32::max);
        let mean_warmth = g.iter().map(|e| e.timbral_warmth).sum::<f32>() / (g.len() as f32);

        let unique_notes: Vec<u8> = g
            .iter()
            .map(|e| e.midi_note)
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();

        out.push(ChordEvent {
            frame: start_frame,
            duration_frames,
            notes: unique_notes,
            max_bend_semitones: max_bend,
            timbral_warmth: mean_warmth,
        });
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sinc_resampler_preserves_passband_and_rejects_aliasing_tones() {
        let sr_in = 48_000.0;
        let sr_out = MODEL_SAMPLE_RATE;
        let n = 4800;

        let pass_in: Vec<f32> = (0..n)
            .map(|i| (2.0 * PI * 1000.0 * (i as f64) / sr_in).sin() as f32)
            .collect();
        let mut r_pass = SincResampler::new(sr_in, sr_out);
        let mut pass_out = Vec::new();
        r_pass.push(&pass_in, &mut pass_out);
        let mid_p = &pass_out[pass_out.len() / 4..(pass_out.len() * 3) / 4];
        let rms_pass = (mid_p.iter().map(|&x| x * x).sum::<f32>() / mid_p.len() as f32).sqrt();
        assert!(rms_pass > 0.65, "1 kHz deveria passar sem perda (rms={})", rms_pass);

        let stop_in: Vec<f32> = (0..n)
            .map(|i| (2.0 * PI * 18_000.0 * (i as f64) / sr_in).sin() as f32)
            .collect();
        let mut r_stop = SincResampler::new(sr_in, sr_out);
        let mut stop_out = Vec::new();
        r_stop.push(&stop_in, &mut stop_out);
        let mid_s = &stop_out[stop_out.len() / 4..(stop_out.len() * 3) / 4];
        let rms_stop = (mid_s.iter().map(|&x| x * x).sum::<f32>() / mid_s.len() as f32).sqrt();
        assert!(rms_stop < 0.005, "18 kHz deveria ser rejeitado pelo anti-aliasing (rms={})", rms_stop);
    }

    #[test]
    fn prunes_adjacent_semitone_bleed_and_sub_octave_ghosts() {
        let mut evs = vec![
            PolyNoteEvent {
                midi_note: 52, // sub-oitava fraca fantasma de 64
                start_frame: 10,
                end_frame: 26,
                confidence: 0.42,
                onset_confidence: 0.35,
                bend_semitones: 0.0,
                median_cents: 0.0,
                timbral_warmth: 0.5,
            },
            PolyNoteEvent {
                midi_note: 64, // nota real E4
                start_frame: 10,
                end_frame: 30,
                confidence: 0.90,
                onset_confidence: 0.88,
                bend_semitones: 0.0,
                median_cents: 0.0,
                timbral_warmth: 0.5,
            },
            PolyNoteEvent {
                midi_note: 65, // vazamento de semitom adjacente F4
                start_frame: 11,
                end_frame: 25,
                confidence: 0.48,
                onset_confidence: 0.45,
                bend_semitones: 0.0,
                median_cents: 0.0,
                timbral_warmth: 0.5,
            },
        ];

        let bleed = prune_semitone_bleed(&mut evs);
        let ov = prune_phantom_overtones(&mut evs, &DecodeConfig::default());
        assert_eq!(bleed, 1);
        assert_eq!(ov, 1);
        assert_eq!(evs.len(), 1);
        assert_eq!(evs[0].midi_note, 64);
    }
}
