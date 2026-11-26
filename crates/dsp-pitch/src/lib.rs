use std::collections::VecDeque;
use std::f32::consts::PI;
use std::sync::Arc;

use rustfft::{num_complex::Complex, Fft, FftPlanner};

/// Remove deslocamento DC (offset contínuo) preservando todas as frequências musicais (> 15 Hz).
#[derive(Debug, Clone, Copy)]
pub struct DcBlocker {
    r: f32,
    x_prev: f32,
    y_prev: f32,
}

impl DcBlocker {
    pub fn new(sample_rate: f32) -> Self {
        let sr = sample_rate.max(8000.0);
        let r = 1.0 - (2.0 * PI * 12.0 / sr).clamp(0.0005, 0.05);
        Self {
            r,
            x_prev: 0.0,
            y_prev: 0.0,
        }
    }

    #[inline]
    pub fn process_sample(&mut self, x: f32) -> f32 {
        let y = x - self.x_prev + self.r * self.y_prev;
        self.x_prev = x;
        self.y_prev = y;
        y
    }

    pub fn process_in_place(&mut self, samples: &mut [f32]) {
        for s in samples.iter_mut() {
            *s = self.process_sample(*s);
        }
    }
}

/// Modo de extração espacial a partir do áudio estéreo [L, R].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StereoMode {
    MonoMix,
    SideCancelCenter,
    LeftOnly,
    RightOnly,
}

impl StereoMode {
    pub fn from_str_id(id: &str) -> Self {
        match id.to_lowercase().as_str() {
            "side" | "lateral" => Self::SideCancelCenter,
            "left" | "l" => Self::LeftOnly,
            "right" | "r" => Self::RightOnly,
            _ => Self::MonoMix,
        }
    }

    pub fn label(&self) -> &'static str {
        match self {
            Self::MonoMix => "Mix Estéreo",
            Self::SideCancelCenter => "Lateral (L−R)",
            Self::LeftOnly => "Canal Esquerdo",
            Self::RightOnly => "Canal Direito",
        }
    }

    #[inline]
    pub fn extract_sample(&self, frame: [f32; 2]) -> f32 {
        match self {
            Self::MonoMix => 0.5 * (frame[0] + frame[1]),
            Self::SideCancelCenter => 0.5 * (frame[0] - frame[1]),
            Self::LeftOnly => frame[0],
            Self::RightOnly => frame[1],
        }
    }

    pub fn extract_slice(&self, frames: &[[f32; 2]], out: &mut Vec<f32>) {
        out.clear();
        out.reserve(frames.len());
        for &fr in frames {
            out.push(self.extract_sample(fr));
        }
    }
}

/// Perfis de foco espectral ampliados para múltiplos instrumentos.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InstrumentFocus {
    FullMix,
    GuitarLead,
    GuitarSolo,
    AcousticBright,
    OudLute,
    Bass,
    PianoKeys,
    Vocals,
    BowedStrings,
    WindsBrass,
}

impl InstrumentFocus {
    pub fn from_str_id(id: &str) -> Self {
        match id.to_lowercase().as_str() {
            "guitar" | "guitarlead" => Self::GuitarLead,
            "solo" | "guitarsolo" => Self::GuitarSolo,
            "acoustic" => Self::AcousticBright,
            "oud" | "lute" => Self::OudLute,
            "bass" => Self::Bass,
            "piano" | "keys" | "synth" => Self::PianoKeys,
            "vocals" | "voice" => Self::Vocals,
            "strings" | "violin" | "cello" => Self::BowedStrings,
            "winds" | "brass" | "sax" => Self::WindsBrass,
            _ => Self::FullMix,
        }
    }

    pub fn label(&self) -> &'static str {
        match self {
            Self::FullMix => "Mix Completo",
            Self::GuitarLead => "Guitarra Base/Lead",
            Self::GuitarSolo => "Solo / Agudo",
            Self::AcousticBright => "Violão Aço/Nylon",
            Self::OudLute => "Oud / Alaúde",
            Self::Bass => "Baixo 4/5c",
            Self::PianoKeys => "Piano / Teclado",
            Self::Vocals => "Voz Principal",
            Self::BowedStrings => "Cordas (Violino/Cello)",
            Self::WindsBrass => "Sopros / Metais",
        }
    }

    pub fn yin_freq_bounds(&self) -> (f32, f32) {
        match self {
            Self::FullMix => (55.0, 1600.0),
            Self::GuitarLead => (75.0, 1700.0),
            Self::GuitarSolo => (196.0, 2400.0),
            Self::AcousticBright => (70.0, 2000.0),
            Self::OudLute => (60.0, 1800.0),
            Self::Bass => (29.0, 380.0),
            Self::PianoKeys => (40.0, 3200.0),
            Self::Vocals => (95.0, 1300.0),
            Self::BowedStrings => (62.0, 2800.0),
            Self::WindsBrass => (105.0, 2600.0),
        }
    }

    pub fn poly_midi_bounds(&self) -> (u8, u8) {
        match self {
            Self::FullMix => (28, 96),
            Self::GuitarLead => (40, 88),
            Self::GuitarSolo => (55, 93),
            Self::AcousticBright => (40, 88),
            Self::OudLute => (36, 84),
            Self::Bass => (23, 64),
            Self::PianoKeys => (21, 108),
            Self::Vocals => (45, 84),
            Self::BowedStrings => (36, 96),
            Self::WindsBrass => (46, 93),
        }
    }

    /// Indica se harmônicos em oitava devem ser podados com menos agressividade (ex.: Piano/Teclado).
    pub fn preserve_wide_octaves(&self) -> bool {
        matches!(self, Self::PianoKeys | Self::BowedStrings)
    }
}

// ============================================================================
// HPSS DE ALTA RESOLUÇÃO (FFT 2048 / HOP 512 / median7 + median11)
// ============================================================================
const HPSS_FFT_SIZE: usize = 2048;
const HPSS_HOP_SIZE: usize = 512;
const HPSS_HALF_BINS: usize = HPSS_FFT_SIZE / 2 + 1;
const HPSS_TIME_FRAMES: usize = 7;
const HPSS_CENTER: usize = HPSS_TIME_FRAMES / 2;

#[inline]
fn median7(mut vals: [f32; 7]) -> f32 {
    vals.sort_by(|a, b| a.total_cmp(b));
    vals[3]
}

#[inline]
fn median11(mut vals: [f32; 11]) -> f32 {
    vals.sort_by(|a, b| a.total_cmp(b));
    vals[5]
}

pub struct HpssSeparator {
    fft_fwd: Arc<dyn Fft<f32>>,
    fft_inv: Arc<dyn Fft<f32>>,
    window: [f32; HPSS_FFT_SIZE],
    in_window: [f32; HPSS_FFT_SIZE],
    in_hop: [f32; HPSS_HOP_SIZE],
    hop_pos: usize,
    ola_buf: [f32; HPSS_FFT_SIZE],
    scratch: Vec<Complex<f32>>,
    spec_history: VecDeque<Vec<Complex<f32>>>,
    mag_history: VecDeque<Vec<f32>>,
    percussive_beta: f32,
}

impl HpssSeparator {
    pub fn new(percussive_beta: f32) -> Self {
        let mut planner = FftPlanner::<f32>::new();
        let fft_fwd = planner.plan_fft_forward(HPSS_FFT_SIZE);
        let fft_inv = planner.plan_fft_inverse(HPSS_FFT_SIZE);

        let mut window = [0.0f32; HPSS_FFT_SIZE];
        for (i, w) in window.iter_mut().enumerate() {
            *w = 0.5 * (1.0 - (2.0 * PI * (i as f32) / (HPSS_FFT_SIZE as f32)).cos());
        }

        Self {
            fft_fwd,
            fft_inv,
            window,
            in_window: [0.0; HPSS_FFT_SIZE],
            in_hop: [0.0; HPSS_HOP_SIZE],
            hop_pos: 0,
            ola_buf: [0.0; HPSS_FFT_SIZE],
            scratch: vec![Complex::new(0.0, 0.0); HPSS_FFT_SIZE],
            spec_history: VecDeque::with_capacity(HPSS_TIME_FRAMES + 1),
            mag_history: VecDeque::with_capacity(HPSS_TIME_FRAMES + 1),
            percussive_beta: percussive_beta.clamp(0.5, 2.0),
        }
    }

    pub fn process_slice(&mut self, input: &[f32], output: &mut Vec<f32>) {
        output.clear();
        output.reserve(input.len() + HPSS_HOP_SIZE);

        for &s in input {
            self.in_hop[self.hop_pos] = s;
            self.hop_pos += 1;

            if self.hop_pos == HPSS_HOP_SIZE {
                self.in_window.copy_within(HPSS_HOP_SIZE.., 0);
                self.in_window[HPSS_FFT_SIZE - HPSS_HOP_SIZE..].copy_from_slice(&self.in_hop);
                self.hop_pos = 0;

                self.process_hop(output);
            }
        }
    }

    fn process_hop(&mut self, output: &mut Vec<f32>) {
        for i in 0..HPSS_FFT_SIZE {
            self.scratch[i] = Complex::new(self.in_window[i] * self.window[i], 0.0);
        }
        self.fft_fwd.process(&mut self.scratch);

        let mut mags = vec![0.0f32; HPSS_HALF_BINS];
        for k in 0..HPSS_HALF_BINS {
            mags[k] = self.scratch[k].norm();
        }

        if self.spec_history.len() == HPSS_TIME_FRAMES {
            self.spec_history.pop_front();
            self.mag_history.pop_front();
        }
        self.spec_history.push_back(self.scratch.clone());
        self.mag_history.push_back(mags);

        if self.spec_history.len() < HPSS_TIME_FRAMES {
            return;
        }

        let center_spec = &self.spec_history[HPSS_CENTER];
        let center_mags = &self.mag_history[HPSS_CENTER];
        self.scratch.copy_from_slice(center_spec);

        let beta = self.percussive_beta;
        for k in 0..HPSS_HALF_BINS {
            let mut t_slice = [0.0f32; 7];
            for t in 0..HPSS_TIME_FRAMES {
                t_slice[t] = self.mag_history[t][k];
            }
            let h = median7(t_slice);

            let mut f_slice = [0.0f32; 11];
            for df in 0..11usize {
                let bin = (k as isize + df as isize - 5).clamp(0, (HPSS_HALF_BINS - 1) as isize)
                    as usize;
                f_slice[df] = center_mags[bin];
            }
            let p = median11(f_slice) * beta;

            let h2 = h * h;
            let p2 = p * p;
            let mask = if h2 + p2 > 1e-12 {
                h2 / (h2 + p2)
            } else {
                0.0
            };

            self.scratch[k] *= mask;
            if k > 0 && k < HPSS_HALF_BINS - 1 {
                self.scratch[HPSS_FFT_SIZE - k] *= mask;
            }
        }

        self.fft_inv.process(&mut self.scratch);

        let norm = 1.0 / (1.5 * HPSS_FFT_SIZE as f32);
        for i in 0..HPSS_FFT_SIZE {
            self.ola_buf[i] += self.scratch[i].re * self.window[i] * norm;
        }

        output.extend_from_slice(&self.ola_buf[..HPSS_HOP_SIZE]);
        self.ola_buf.copy_within(HPSS_HOP_SIZE.., 0);
        self.ola_buf[HPSS_FFT_SIZE - HPSS_HOP_SIZE..].fill(0.0);
    }
}

#[derive(Debug, Clone, Copy)]
struct Biquad {
    b0: f32,
    b1: f32,
    b2: f32,
    a1: f32,
    a2: f32,
    z1: f32,
    z2: f32,
}

impl Biquad {
    fn bypass() -> Self {
        Self {
            b0: 1.0,
            b1: 0.0,
            b2: 0.0,
            a1: 0.0,
            a2: 0.0,
            z1: 0.0,
            z2: 0.0,
        }
    }

    fn highpass(sample_rate: f32, cutoff_hz: f32, q: f32) -> Self {
        let fc = cutoff_hz.clamp(10.0, sample_rate * 0.45);
        let w0 = 2.0 * PI * fc / sample_rate;
        let cos_w0 = w0.cos();
        let sin_w0 = w0.sin();
        let alpha = sin_w0 / (2.0 * q);
        let a0 = 1.0 + alpha;

        Self {
            b0: ((1.0 + cos_w0) * 0.5) / a0,
            b1: -(1.0 + cos_w0) / a0,
            b2: ((1.0 + cos_w0) * 0.5) / a0,
            a1: (-2.0 * cos_w0) / a0,
            a2: (1.0 - alpha) / a0,
            z1: 0.0,
            z2: 0.0,
        }
    }

    fn lowpass(sample_rate: f32, cutoff_hz: f32, q: f32) -> Self {
        let fc = cutoff_hz.clamp(20.0, sample_rate * 0.45);
        let w0 = 2.0 * PI * fc / sample_rate;
        let cos_w0 = w0.cos();
        let sin_w0 = w0.sin();
        let alpha = sin_w0 / (2.0 * q);
        let a0 = 1.0 + alpha;

        Self {
            b0: ((1.0 - cos_w0) * 0.5) / a0,
            b1: (1.0 - cos_w0) / a0,
            b2: ((1.0 - cos_w0) * 0.5) / a0,
            a1: (-2.0 * cos_w0) / a0,
            a2: (1.0 - alpha) / a0,
            z1: 0.0,
            z2: 0.0,
        }
    }

    #[inline]
    fn process(&mut self, x: f32) -> f32 {
        let y = self.b0 * x + self.z1;
        self.z1 = self.b1 * x - self.a1 * y + self.z2;
        self.z2 = self.b2 * x - self.a2 * y;
        y
    }
}

pub struct InstrumentFilter {
    focus: InstrumentFocus,
    hpf1: Biquad,
    hpf2: Biquad,
    lpf1: Biquad,
    lpf2: Biquad,
    use_lpf: bool,
}

impl InstrumentFilter {
    pub fn new(sample_rate: f32, focus: InstrumentFocus) -> Self {
        const Q1: f32 = 0.541_196_1;
        const Q2: f32 = 1.306_563;

        let (hp_hz, lp_opt) = match focus {
            InstrumentFocus::FullMix => (30.0, None),
            InstrumentFocus::GuitarLead => (80.0, Some(1800.0)),
            InstrumentFocus::GuitarSolo => (220.0, Some(2800.0)),
            InstrumentFocus::AcousticBright => (75.0, Some(3000.0)),
            InstrumentFocus::OudLute => (62.0, Some(2600.0)),
            InstrumentFocus::Bass => (30.0, Some(360.0)),
            InstrumentFocus::PianoKeys => (35.0, Some(4500.0)),
            InstrumentFocus::Vocals => (100.0, Some(1600.0)),
            InstrumentFocus::BowedStrings => (62.0, Some(3800.0)),
            InstrumentFocus::WindsBrass => (105.0, Some(3400.0)),
        };

        let (lpf1, lpf2, use_lpf) = match lp_opt {
            Some(lp_hz) => (
                Biquad::lowpass(sample_rate, lp_hz, Q1),
                Biquad::lowpass(sample_rate, lp_hz, Q2),
                true,
            ),
            None => (Biquad::bypass(), Biquad::bypass(), false),
        };

        Self {
            focus,
            hpf1: Biquad::highpass(sample_rate, hp_hz, Q1),
            hpf2: Biquad::highpass(sample_rate, hp_hz, Q2),
            lpf1,
            lpf2,
            use_lpf,
        }
    }

    /// Filtro otimizado para redes CQT (Basic Pitch): aplica apenas HPF subsônico para não cortar
    /// os harmônicos superiores que o empilhamento harmônico da rede usa para estimar a fundamental.
    pub fn new_for_poly(sample_rate: f32, focus: InstrumentFocus) -> Self {
        const Q1: f32 = 0.541_196_1;
        const Q2: f32 = 1.306_563;

        let hp_hz = match focus {
            InstrumentFocus::FullMix | InstrumentFocus::PianoKeys => 25.0,
            InstrumentFocus::Bass => 26.0,
            InstrumentFocus::OudLute | InstrumentFocus::BowedStrings => 55.0,
            InstrumentFocus::GuitarLead | InstrumentFocus::AcousticBright => 70.0,
            InstrumentFocus::Vocals | InstrumentFocus::WindsBrass => 90.0,
            InstrumentFocus::GuitarSolo => 180.0,
        };

        Self {
            focus,
            hpf1: Biquad::highpass(sample_rate, hp_hz, Q1),
            hpf2: Biquad::highpass(sample_rate, hp_hz, Q2),
            lpf1: Biquad::bypass(),
            lpf2: Biquad::bypass(),
            use_lpf: false,
        }
    }

    pub fn focus(&self) -> InstrumentFocus {
        self.focus
    }

    #[inline]
    pub fn process_sample(&mut self, x: f32) -> f32 {
        let mut y = self.hpf2.process(self.hpf1.process(x));
        if self.use_lpf {
            y = self.lpf2.process(self.lpf1.process(y));
        }
        y
    }

    pub fn process_slice(&mut self, input: &[f32], output: &mut Vec<f32>) {
        output.clear();
        output.reserve(input.len());
        for &x in input {
            output.push(self.process_sample(x));
        }
    }
}

pub fn estimate_timbral_warmth(samples: &[f32], sample_rate: f32, fundamental_hz: f32) -> f32 {
    if samples.len() < 16 || fundamental_hz <= 20.0 {
        return 0.5;
    }

    let mut energy = 0.0f64;
    let mut diff_energy = 0.0f64;
    for pair in samples.windows(2) {
        let x0 = pair[0] as f64;
        let d = (pair[1] as f64) - x0;
        energy += x0 * x0;
        diff_energy += d * d;
    }

    if energy <= 1e-12 {
        return 0.5;
    }

    let mobility = (diff_energy / energy).sqrt();
    let arg = (0.5 * mobility).clamp(0.0, 1.0);
    let centroid_hz = ((sample_rate as f64) / std::f64::consts::PI) * arg.asin();
    let ratio = (centroid_hz as f32) / fundamental_hz;

    ((3.6 - ratio) / 2.2).clamp(0.0, 1.0)
}

#[derive(Debug, Clone, Copy)]
pub struct PitchFrame {
    pub freq_hz: f32,
    pub midi_note: u8,
    pub cents: f32,
    pub clarity: f32,
    pub rms_db: f32,
    pub transient_flux: f32,
    pub timbral_warmth: f32,
    pub is_reattack: bool,
}

pub struct YinDetector {
    sample_rate: f32,
    window_size: usize,
    threshold: f32,
    min_freq: f32,
    max_freq: f32,
    noise_gate_rms: f32,
    reference_a4: f32,
    diff_buf: Vec<f32>,
}

impl YinDetector {
    pub fn new(sample_rate: f32, window_size: usize) -> Self {
        let half = window_size / 2;
        Self {
            sample_rate,
            window_size,
            threshold: 0.15,
            min_freq: 65.0,
            max_freq: 1400.0,
            noise_gate_rms: 0.008,
            reference_a4: 440.0,
            diff_buf: vec![0.0; half + 1],
        }
    }

    pub fn set_freq_range(&mut self, min_freq: f32, max_freq: f32) {
        self.min_freq = min_freq.max(25.0);
        self.max_freq = max_freq.max(self.min_freq + 20.0);
    }

    pub fn set_reference_a4(&mut self, a4_hz: f32) {
        self.reference_a4 = a4_hz.clamp(415.0, 466.0);
    }

    pub fn detect(&mut self, samples: &[f32]) -> Option<PitchFrame> {
        if samples.len() < self.window_size {
            return None;
        }

        let slice = &samples[..self.window_size];

        let sum_sq: f32 = slice.iter().map(|&s| s * s).sum();
        let rms = (sum_sq / self.window_size as f32).sqrt();
        if rms < self.noise_gate_rms {
            return None;
        }
        let rms_db = 20.0 * rms.max(1e-7).log10();

        let tail_start = self.window_size.saturating_sub((self.window_size / 4).max(64));
        let tail = &slice[tail_start..];
        let mut flux_sum = 0.0f32;
        for pair in tail.windows(2) {
            flux_sum += (pair[1] - pair[0]).abs();
        }
        let transient_flux = flux_sum / (tail.len().saturating_sub(1).max(1) as f32);

        let half = self.window_size / 2;
        let min_tau = ((self.sample_rate / self.max_freq).floor() as usize).max(2);
        let max_tau = ((self.sample_rate / self.min_freq).ceil() as usize).min(half - 1);
        if min_tau >= max_tau {
            return None;
        }

        self.diff_buf[0] = 1.0;
        let mut running_sum = 0.0f32;
        for tau in 1..=max_tau {
            let mut sum = 0.0f32;
            for j in 0..half {
                let delta = slice[j] - slice[j + tau];
                sum += delta * delta;
            }
            running_sum += sum;
            self.diff_buf[tau] = if running_sum > 1e-12 {
                sum * (tau as f32) / running_sum
            } else {
                1.0
            };
        }

        let mut tau_est = None;
        let mut tau = min_tau;
        while tau <= max_tau {
            if self.diff_buf[tau] < self.threshold {
                while tau + 1 <= max_tau && self.diff_buf[tau + 1] < self.diff_buf[tau] {
                    tau += 1;
                }
                tau_est = Some(tau);
                break;
            }
            tau += 1;
        }

        let tau = tau_est?;

        let refined_tau = if tau > 1 && tau < max_tau {
            let s0 = self.diff_buf[tau - 1];
            let s1 = self.diff_buf[tau];
            let s2 = self.diff_buf[tau + 1];
            let denom = 2.0 * (2.0 * s1 - s2 - s0);
            if denom.abs() > 1e-12 {
                (tau as f32) + (s2 - s0) / denom
            } else {
                tau as f32
            }
        } else {
            tau as f32
        };

        let freq_hz = self.sample_rate / refined_tau;
        let midi_float = 69.0 + 12.0 * (freq_hz / self.reference_a4).log2();
        if !(0.0..=127.0).contains(&midi_float) {
            return None;
        }

        let midi_round = midi_float.round();
        let cents = (midi_float - midi_round) * 100.0;
        let clarity = (1.0 - self.diff_buf[tau]).clamp(0.0, 1.0);
        let timbral_warmth = estimate_timbral_warmth(slice, self.sample_rate, freq_hz);

        Some(PitchFrame {
            freq_hz,
            midi_note: midi_round as u8,
            cents,
            clarity,
            rms_db,
            transient_flux,
            timbral_warmth,
            is_reattack: false,
        })
    }
}

pub struct NoteStabilizer {
    min_stable_frames: usize,
    silence_reset_frames: usize,
    reattack_cooldown_frames: usize,
    reattack_db_delta: f32,
    candidate_note: Option<u8>,
    candidate_count: usize,
    active_note: Option<u8>,
    silence_count: usize,
    frames_since_trigger: usize,
    env_rms_db: f32,
    env_flux: f32,
}

impl NoteStabilizer {
    pub fn new(min_stable_frames: usize, silence_reset_frames: usize) -> Self {
        Self {
            min_stable_frames,
            silence_reset_frames,
            reattack_cooldown_frames: 8,
            reattack_db_delta: 3.5,
            candidate_note: None,
            candidate_count: 0,
            active_note: None,
            silence_count: 0,
            frames_since_trigger: 0,
            env_rms_db: -60.0,
            env_flux: 0.0,
        }
    }

    pub fn push(&mut self, frame: Option<PitchFrame>) -> Option<PitchFrame> {
        let Some(mut f) = frame else {
            self.silence_count += 1;
            self.candidate_count = 0;
            self.candidate_note = None;
            if self.silence_count >= self.silence_reset_frames {
                self.active_note = None;
                self.env_rms_db = -60.0;
                self.env_flux = 0.0;
            }
            return None;
        };

        self.silence_count = 0;
        self.frames_since_trigger = self.frames_since_trigger.saturating_add(1);

        if self.active_note == Some(f.midi_note) {
            let db_jump = f.rms_db - self.env_rms_db;
            let flux_ratio = if self.env_flux > 1e-6 {
                f.transient_flux / self.env_flux
            } else {
                1.0
            };

            let triggered_reattack = self.frames_since_trigger >= self.reattack_cooldown_frames
                && (db_jump >= self.reattack_db_delta
                    || (db_jump >= 1.8 && flux_ratio >= 1.55));

            let rms_alpha = if f.rms_db < self.env_rms_db { 0.30 } else { 0.12 };
            self.env_rms_db += rms_alpha * (f.rms_db - self.env_rms_db);
            self.env_flux += 0.20 * (f.transient_flux - self.env_flux);

            if triggered_reattack {
                self.frames_since_trigger = 0;
                self.env_rms_db = f.rms_db;
                self.env_flux = f.transient_flux;
                f.is_reattack = true;
                return Some(f);
            }
            return None;
        }

        if self.candidate_note == Some(f.midi_note) {
            self.candidate_count += 1;
        } else {
            self.candidate_note = Some(f.midi_note);
            self.candidate_count = 1;
        }

        if self.candidate_count == self.min_stable_frames {
            self.active_note = Some(f.midi_note);
            self.frames_since_trigger = 0;
            self.env_rms_db = f.rms_db;
            self.env_flux = f.transient_flux;
            f.is_reattack = false;
            return Some(f);
        }

        None
    }
}

pub fn midi_to_note_name(midi: u8) -> String {
    const NAMES: [&str; 12] = [
        "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
    ];
    let octave = (midi as i32 / 12) - 1;
    format!("{}{}", NAMES[(midi % 12) as usize], octave)
}

/// Mede a correlação de fase L/R (-1.0 = fase invertida, 0.0 = descorrelacionado, +1.0 = mono/em fase).
pub fn compute_phase_correlation(frames: &[[f32; 2]]) -> f32 {
    if frames.is_empty() {
        return 1.0;
    }
    let mut sum_lr = 0.0f64;
    let mut sum_ll = 0.0f64;
    let mut sum_rr = 0.0f64;
    for fr in frames {
        let l = fr[0] as f64;
        let r = fr[1] as f64;
        sum_lr += l * r;
        sum_ll += l * l;
        sum_rr += r * r;
    }
    let denom = (sum_ll * sum_rr).sqrt();
    if denom <= 1e-12 {
        return 1.0;
    }
    (sum_lr / denom).clamp(-1.0, 1.0) as f32
}

/// Auto-ganho suave (AGC) com proteção de piso de ruído e soft-clipping para alimentar DSP/IA em nível ótimo.
pub struct SignalLeveler {
    target_rms: f32,
    noise_floor_rms: f32,
    min_gain: f32,
    max_gain: f32,
    env_sq: f32,
    gain: f32,
    env_alpha: f32,
    attack_alpha: f32,
    release_alpha: f32,
}

impl SignalLeveler {
    pub fn new(sample_rate: f32) -> Self {
        let sr = sample_rate.max(8000.0);
        Self {
            target_rms: 0.16,       // ~ -16 dBFS
            noise_floor_rms: 0.004, // ~ -48 dBFS
            min_gain: 0.25,
            max_gain: 8.0,          // até +18 dB
            env_sq: 0.0,
            gain: 1.0,
            env_alpha: 1.0 - (-1.0 / (0.06 * sr)).exp(),
            attack_alpha: 1.0 - (-1.0 / (0.025 * sr)).exp(),
            release_alpha: 1.0 - (-1.0 / (0.15 * sr)).exp(),
        }
    }

    pub fn current_gain_db(&self) -> f32 {
        20.0 * self.gain.max(1e-4).log10()
    }

    #[inline]
    pub fn process_sample(&mut self, x: f32) -> f32 {
        self.env_sq += self.env_alpha * (x * x - self.env_sq);
        let rms = self.env_sq.max(0.0).sqrt();

        let desired_gain = if rms > self.noise_floor_rms {
            (self.target_rms / rms).clamp(self.min_gain, self.max_gain)
        } else {
            1.0
        };

        let alpha = if desired_gain < self.gain {
            self.attack_alpha
        } else {
            self.release_alpha
        };
        self.gain += alpha * (desired_gain - self.gain);

        let y = x * self.gain;
        if y.abs() <= 0.88 {
            y
        } else {
            let excess = (y.abs() - 0.88) / 0.12;
            y.signum() * (0.88 + 0.115 * excess.tanh())
        }
    }

    pub fn process_in_place(&mut self, samples: &mut [f32]) {
        for s in samples.iter_mut() {
            *s = self.process_sample(*s);
        }
    }
}

/// Classificação acústica leve do perfil instrumental predominante a partir de registro, brilho e polifonia.
pub fn classify_instrument_signature(
    median_midi: f32,
    mean_polyphony: f32,
    mean_warmth: f32,
    percussiveness: f32,
) -> &'static str {
    if median_midi <= 47.0 {
        return "Baixo / Grave";
    }
    if mean_polyphony >= 2.3 {
        if median_midi >= 56.0 && mean_warmth < 0.48 {
            return "Piano / Teclado";
        }
        if mean_warmth >= 0.60 {
            return "Guitarra / Violão (Acordes)";
        }
        return "Harmonia (Guitarra/Teclado)";
    }
    if percussiveness < 0.14 && mean_warmth >= 0.55 {
        return if median_midi >= 69.0 {
            "Cordas / Violino / Synth"
        } else {
            "Voz / Cello / Pad"
        };
    }
    if percussiveness < 0.22 && mean_warmth < 0.45 && median_midi >= 60.0 {
        return "Sopros / Metais / Lead";
    }
    if median_midi >= 68.0 {
        return "Guitarra Solo / Agudo";
    }
    if mean_warmth >= 0.58 {
        return "Oud / Violão / Guitarra";
    }
    "Guitarra / Melodia"
}


/// Perfil instrumental detectado automaticamente com preset correspondente.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DetectedInstrumentProfile {
    pub label: &'static str,
    pub preset_id: &'static str,
}

/// Mede a percussividade real do sinal (0.0 = ataque suave/arco/voz, 1.0 = ataque muito percussivo/palheta).
pub fn compute_percussiveness(samples: &[f32]) -> f32 {
    if samples.len() < 64 {
        return 0.25;
    }
    let sum_sq: f64 = samples.iter().map(|&x| (x as f64) * (x as f64)).sum();
    let rms = (sum_sq / samples.len() as f64).sqrt() as f32;
    if rms <= 1e-6 {
        return 0.0;
    }
    let peak = samples.iter().map(|&x| x.abs()).fold(0.0f32, f32::max);
    let crest = peak / rms; // senoide ~1.41, voz/arco ~2.2-3.5, guitarra/piano ~4.0-7.0
    ((crest - 2.0) / 4.5).clamp(0.0, 1.0)
}

/// Detecta automaticamente o instrumento e mapeia para o preset ideal (clave, pauta e timbre GM).
pub fn detect_instrument_profile(
    median_midi: f32,
    mean_polyphony: f32,
    mean_warmth: f32,
    percussiveness: f32,
) -> DetectedInstrumentProfile {
    if median_midi <= 47.0 {
        return if percussiveness < 0.28 {
            DetectedInstrumentProfile {
                label: "Violoncelo / Contrabaixo",
                preset_id: "cello",
            }
        } else {
            DetectedInstrumentProfile {
                label: "Baixo Elétrico/Acústico",
                preset_id: "bass4",
            }
        };
    }

    if mean_polyphony >= 2.2 {
        if median_midi >= 56.0 && mean_warmth < 0.50 {
            return DetectedInstrumentProfile {
                label: "Piano / Teclado",
                preset_id: "piano",
            };
        }
        return DetectedInstrumentProfile {
            label: "Guitarra / Violão (Acordes)",
            preset_id: "standard",
        };
    }

    if percussiveness < 0.22 {
        if mean_warmth >= 0.54 {
            return if median_midi >= 69.0 {
                DetectedInstrumentProfile {
                    label: "Cordas / Violino",
                    preset_id: "strings",
                }
            } else {
                DetectedInstrumentProfile {
                    label: "Voz Principal",
                    preset_id: "vocals",
                }
            };
        }
        if median_midi >= 58.0 {
            return DetectedInstrumentProfile {
                label: "Sopros / Sax / Metais",
                preset_id: "winds",
            };
        }
    }

    if median_midi >= 68.0 {
        return DetectedInstrumentProfile {
            label: "Guitarra Solo / Agudo",
            preset_id: "standard",
        };
    }

    if mean_warmth >= 0.62 && percussiveness >= 0.35 {
        return DetectedInstrumentProfile {
            label: "Oud / Violão Nylon",
            preset_id: "oud_syrian",
        };
    }

    DetectedInstrumentProfile {
        label: "Guitarra / Melodia",
        preset_id: "standard",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dc_blocker_removes_constant_offset_cleanly() {
        let mut dc = DcBlocker::new(48_000.0);
        let mut buf = vec![0.4f32; 8192];
        dc.process_in_place(&mut buf);
        assert!(buf.last().unwrap().abs() < 0.01);
    }

    #[test]
    fn hpss_2048_preserves_harmonic_and_attenuates_percussive_clicks() {
        let sr = 48_000.0;
        let n = 24_576;

        let sine: Vec<f32> = (0..n)
            .map(|i| 0.5 * (2.0 * PI * 440.0 * (i as f32) / sr).sin())
            .collect();
        let mut hpss_s = HpssSeparator::new(1.1);
        let mut out_sine = Vec::new();
        hpss_s.process_slice(&sine, &mut out_sine);
        let tail_s = &out_sine[out_sine.len() / 2..];
        let rms_sine = (tail_s.iter().map(|&x| x * x).sum::<f32>() / tail_s.len() as f32).sqrt();
        assert!(rms_sine > 0.30, "senoide deveria passar pelo HPSS 2048 (rms={})", rms_sine);

        let mut clicks = vec![0.0f32; n];
        for pos in (2048..n).step_by(4000) {
            clicks[pos] = 1.0;
            if pos + 1 < n {
                clicks[pos + 1] = -0.8;
            }
        }
        let mut hpss_c = HpssSeparator::new(1.1);
        let mut out_clicks = Vec::new();
        hpss_c.process_slice(&clicks, &mut out_clicks);
        let peak_out = out_clicks.iter().map(|&x| x.abs()).fold(0.0f32, f32::max);
        assert!(peak_out < 0.25, "cliques deveriam ser atenuados (peak={})", peak_out);
    }

    #[test]
    fn signal_leveler_boosts_quiet_signal_without_boosting_silence() {
        let sr = 48_000.0;
        let mut lvl = SignalLeveler::new(sr);
        let mut quiet: Vec<f32> = (0..24_000)
            .map(|i| 0.025 * (2.0 * PI * 440.0 * (i as f32) / sr).sin())
            .collect();
        lvl.process_in_place(&mut quiet);
        let tail = &quiet[quiet.len() / 2..];
        let rms = (tail.iter().map(|&x| x * x).sum::<f32>() / tail.len() as f32).sqrt();
        assert!(rms > 0.11 && rms < 0.22, "RMS nivelado fora do alvo: {}", rms);

        let mut silent = vec![0.0005f32; 24_000];
        lvl.process_in_place(&mut silent);
        assert!(lvl.current_gain_db() < 3.0, "não deveria amplificar ruído de fundo");
    }

    #[test]
    fn phase_correlation_measures_mono_and_inverted_stereo() {
        let mono = [[0.5, 0.5], [-0.3, -0.3]];
        let inv = [[0.5, -0.5], [-0.3, 0.3]];
        assert!((compute_phase_correlation(&mono) - 1.0).abs() < 1e-4);
        assert!((compute_phase_correlation(&inv) + 1.0).abs() < 1e-4);
    }
}
