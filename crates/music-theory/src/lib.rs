/// Clave musical para renderização correta por família instrumental.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClefKind {
    /// Clave de Sol em altura real (Piano, Voz, Violino, Flauta, Sax, Synth)
    Treble,
    /// Clave de Sol oitavada abaixo (Guitarra, Violão, Oud)
    Treble8vb,
    /// Clave de Fá (Baixo, Violoncelo, Fagote, Trombone)
    Bass,
}

impl Default for ClefKind {
    fn default() -> Self {
        Self::Treble8vb
    }
}

impl ClefKind {
    pub fn alphatex_clef(&self) -> &'static str {
        match self {
            Self::Treble | Self::Treble8vb => "G2",
            Self::Bass => "F4",
        }
    }
}

use tab_solver::{ChordVoicing, FretPos};

/// Fórmula de compasso suportada pelo quantizador e exportadores.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TimeSignature {
    pub numerator: u8,
    pub denominator: u8,
}

impl Default for TimeSignature {
    fn default() -> Self {
        Self {
            numerator: 4,
            denominator: 4,
        }
    }
}

impl TimeSignature {
    pub const fn new(numerator: u8, denominator: u8) -> Self {
        Self {
            numerator,
            denominator,
        }
    }

    pub fn from_str_id(id: &str) -> Self {
        match id.trim() {
            "3/4" => Self::new(3, 4),
            "6/8" => Self::new(6, 8),
            "5/4" => Self::new(5, 4),
            "7/8" => Self::new(7, 8),
            _ => Self::new(4, 4),
        }
    }

    /// Total de semicolcheias (ticks) em um compasso completo.
    pub fn measure_ticks(&self) -> u8 {
        let ticks_per_beat = 16u8 / self.denominator.max(1);
        self.numerator.saturating_mul(ticks_per_beat).max(4)
    }

    pub fn display_label(&self) -> String {
        format!("{}/{}", self.numerator, self.denominator)
    }

    pub fn alphatex_ts(&self) -> String {
        format!("({} {})", self.numerator, self.denominator)
    }

    /// Potência de 2 do denominador para o meta-evento MIDI 0xFF 0x58.
    pub fn midi_denominator_pow2(&self) -> u8 {
        match self.denominator {
            1 => 0,
            2 => 1,
            4 => 2,
            8 => 3,
            16 => 4,
            _ => 2,
        }
    }
}

/// Modo tonal detectado pelo classificador Krumhansl-Schmuckler.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeyMode {
    Major,
    Minor,
}

/// Tonalidade inferida com posição no ciclo de quintas (-7 a +7).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DetectedKey {
    pub tonic_pc: u8,
    pub mode: KeyMode,
    pub fifths: i8,
    pub confidence: f32,
}

impl Default for DetectedKey {
    fn default() -> Self {
        Self {
            tonic_pc: 0,
            mode: KeyMode::Major,
            fifths: 0,
            confidence: 0.0,
        }
    }
}

impl DetectedKey {
    pub fn prefer_flats(&self) -> bool {
        self.fifths < 0
    }

    pub fn display_name(&self) -> String {
        let tonic = pitch_class_name(self.tonic_pc, self.prefer_flats());
        match self.mode {
            KeyMode::Major => format!("{} maior", tonic),
            KeyMode::Minor => format!("{} menor", tonic),
        }
    }

    pub fn alphatex_ks(&self) -> &'static str {
        match self.fifths {
            -7 => "Cb",
            -6 => "Gb",
            -5 => "Db",
            -4 => "Ab",
            -3 => "Eb",
            -2 => "Bb",
            -1 => "F",
            0 => "C",
            1 => "G",
            2 => "D",
            3 => "A",
            4 => "E",
            5 => "B",
            6 => "F#",
            7 => "C#",
            _ => "C",
        }
    }

    pub fn is_diatonic(&self, pc: u8) -> bool {
        let interval = (pc + 12 - (self.tonic_pc % 12)) % 12;
        match self.mode {
            KeyMode::Major => matches!(interval, 0 | 2 | 4 | 5 | 7 | 9 | 11),
            KeyMode::Minor => matches!(interval, 0 | 2 | 3 | 5 | 7 | 8 | 10),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChordQuality {
    Major,
    Minor,
    Power,
    Sus2,
    Sus4,
    Diminished,
    Augmented,
    Dominant7,
    Major7,
    Minor7,
    HalfDim7,
}

impl ChordQuality {
    pub fn suffix(&self) -> &'static str {
        match self {
            Self::Major => "",
            Self::Minor => "m",
            Self::Power => "5",
            Self::Sus2 => "sus2",
            Self::Sus4 => "sus4",
            Self::Diminished => "dim",
            Self::Augmented => "aug",
            Self::Dominant7 => "7",
            Self::Major7 => "maj7",
            Self::Minor7 => "m7",
            Self::HalfDim7 => "m7b5",
        }
    }

    pub fn musicxml_kind(&self) -> &'static str {
        match self {
            Self::Major => "major",
            Self::Minor => "minor",
            Self::Power => "power",
            Self::Sus2 => "suspended-second",
            Self::Sus4 => "suspended-fourth",
            Self::Diminished => "diminished",
            Self::Augmented => "augmented",
            Self::Dominant7 => "dominant",
            Self::Major7 => "major-seventh",
            Self::Minor7 => "minor-seventh",
            Self::HalfDim7 => "half-diminished",
        }
    }

    fn intervals(&self) -> &'static [u8] {
        match self {
            Self::Major => &[0, 4, 7],
            Self::Minor => &[0, 3, 7],
            Self::Power => &[0, 7],
            Self::Sus2 => &[0, 2, 7],
            Self::Sus4 => &[0, 5, 7],
            Self::Diminished => &[0, 3, 6],
            Self::Augmented => &[0, 4, 8],
            Self::Dominant7 => &[0, 4, 7, 10],
            Self::Major7 => &[0, 4, 7, 11],
            Self::Minor7 => &[0, 3, 7, 10],
            Self::HalfDim7 => &[0, 3, 6, 10],
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChordSymbol {
    pub root_pc: u8,
    pub quality: ChordQuality,
    pub bass_pc: Option<u8>,
    pub label: String,
}

/// Evento musical capturado com estampa temporal, duração e expressividade (bend).
#[derive(Debug, Clone)]
pub struct TimedEvent {
    pub start_ms: u64,
    pub duration_ms: u64,
    pub notes: Vec<u8>,
    pub bend_semitones: f32,
}

/// Unidade rítmica quantizada em semicolcheias (ticks).
/// Valores válidos: 1 (16th), 2 (8th), 3 (8th.), 4 (quarter), 6 (quarter.), 8 (half), 12 (half.), 16 (whole).
#[derive(Debug, Clone)]
pub struct QuantizedBeat {
    pub duration_ticks: u8,
    pub midi_notes: Vec<u8>,
    pub voicing: Option<ChordVoicing>,
    pub bend_semitones: f32,
    pub chord: Option<ChordSymbol>,
    pub tie_from_prev: bool,
    pub tie_to_next: bool,
}

impl QuantizedBeat {
    pub fn is_rest(&self) -> bool {
        self.midi_notes.is_empty()
    }

    pub fn is_dotted(&self) -> bool {
        matches!(self.duration_ticks, 3 | 6 | 12)
    }

    pub fn alphatex_div(&self) -> u8 {
        match self.duration_ticks {
            1 => 16,
            2 | 3 => 8,
            4 | 6 => 4,
            8 | 12 => 2,
            _ => 1,
        }
    }

    pub fn musicxml_type(&self) -> &'static str {
        match self.duration_ticks {
            1 => "16th",
            2 | 3 => "eighth",
            4 | 6 => "quarter",
            8 | 12 => "half",
            _ => "whole",
        }
    }
}

#[derive(Debug, Clone)]
pub struct QuantizedMeasure {
    pub beats: Vec<QuantizedBeat>,
}

#[derive(Debug, Clone)]
pub struct QuantizedScore {
    pub title: String,
    pub bpm: u16,
    pub time_signature: TimeSignature,
    pub key: DetectedKey,
    pub tuning_low_to_high: Vec<u8>,
    pub measures: Vec<QuantizedMeasure>,
    pub midi_program: u8,
    pub show_tab_staff: bool,
    pub clef: ClefKind,
}

impl QuantizedScore {
    pub fn with_instrument(mut self, midi_program: u8, show_tab_staff: bool, clef: ClefKind) -> Self {
        self.midi_program = midi_program;
        self.show_tab_staff = show_tab_staff;
        self.clef = clef;
        self
    }
}

// ============================================================================
// ENARMONIA, TONALIDADE (KRUMHANSL-SCHMUCKLER) E CIFRAGEM
// ============================================================================

pub fn pitch_class_name(pc: u8, prefer_flats: bool) -> &'static str {
    const SHARPS: [&str; 12] = [
        "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
    ];
    const FLATS: [&str; 12] = [
        "C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B",
    ];
    let idx = (pc % 12) as usize;
    if prefer_flats {
        FLATS[idx]
    } else {
        SHARPS[idx]
    }
}

pub fn midi_to_note_name_for_key(midi: u8, key: DetectedKey) -> String {
    let octave = (midi as i32 / 12) - 1;
    format!("{}{}", pitch_class_name(midi % 12, key.prefer_flats()), octave)
}

fn pitch_class_to_xml_step(pc: u8, prefer_flats: bool) -> (&'static str, i8) {
    const SHARPS: [(&str, i8); 12] = [
        ("C", 0), ("C", 1), ("D", 0), ("D", 1), ("E", 0), ("F", 0),
        ("F", 1), ("G", 0), ("G", 1), ("A", 0), ("A", 1), ("B", 0),
    ];
    const FLATS: [(&str, i8); 12] = [
        ("C", 0), ("D", -1), ("D", 0), ("E", -1), ("E", 0), ("F", 0),
        ("G", -1), ("G", 0), ("A", -1), ("A", 0), ("B", -1), ("B", 0),
    ];
    let idx = (pc % 12) as usize;
    if prefer_flats {
        FLATS[idx]
    } else {
        SHARPS[idx]
    }
}

fn midi_to_xml_pitch(midi: u8, prefer_flats: bool) -> (&'static str, i8, i32) {
    let (step, alter) = pitch_class_to_xml_step(midi % 12, prefer_flats);
    let octave = (midi as i32 / 12) - 1;
    (step, alter, octave)
}

fn tonic_mode_to_fifths(tonic_pc: u8, mode: KeyMode) -> i8 {
    match mode {
        KeyMode::Major => match tonic_pc % 12 {
            0 => 0,
            1 => -5,
            2 => 2,
            3 => -3,
            4 => 4,
            5 => -1,
            6 => 6,
            7 => 1,
            8 => -4,
            9 => 3,
            10 => -2,
            11 => 5,
            _ => 0,
        },
        KeyMode::Minor => match tonic_pc % 12 {
            0 => -3,
            1 => 4,
            2 => -1,
            3 => -6,
            4 => 1,
            5 => -4,
            6 => 3,
            7 => -2,
            8 => 5,
            9 => 0,
            10 => -5,
            11 => 2,
            _ => 0,
        },
    }
}

fn pearson_correlation(a: &[f64; 12], b: &[f64; 12]) -> f64 {
    let mean_a = a.iter().sum::<f64>() / 12.0;
    let mean_b = b.iter().sum::<f64>() / 12.0;

    let mut num = 0.0;
    let mut den_a = 0.0;
    let mut den_b = 0.0;

    for i in 0..12 {
        let da = a[i] - mean_a;
        let db = b[i] - mean_b;
        num += da * db;
        den_a += da * da;
        den_b += db * db;
    }

    let denom = (den_a * den_b).sqrt();
    if denom <= 1e-12 {
        0.0
    } else {
        num / denom
    }
}

pub fn detect_key(events: &[TimedEvent]) -> DetectedKey {
    const MAJOR_PROFILE: [f64; 12] = [
        6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88,
    ];
    const MINOR_PROFILE: [f64; 12] = [
        6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17,
    ];

    let mut hist = [0.0f64; 12];
    let mut total_weight = 0.0f64;

    for ev in events {
        if ev.notes.is_empty() {
            continue;
        }
        let dur_weight = (ev.duration_ms.max(80) as f64) / 250.0;
        let lowest = ev.notes.iter().copied().min();

        for &midi in &ev.notes {
            let bass_boost = if Some(midi) == lowest { 1.35 } else { 1.0 };
            let w = dur_weight * bass_boost;
            hist[(midi % 12) as usize] += w;
            total_weight += w;
        }
    }

    if total_weight <= 0.0 {
        return DetectedKey::default();
    }

    let mut best_key = DetectedKey::default();
    let mut best_score = f64::NEG_INFINITY;

    for tonic in 0..12u8 {
        let mut rotated_maj = [0.0f64; 12];
        let mut rotated_min = [0.0f64; 12];
        for i in 0..12 {
            let pc = (tonic as usize + i) % 12;
            rotated_maj[pc] = MAJOR_PROFILE[i];
            rotated_min[pc] = MINOR_PROFILE[i];
        }

        let r_maj = pearson_correlation(&hist, &rotated_maj);
        if r_maj > best_score {
            best_score = r_maj;
            best_key = DetectedKey {
                tonic_pc: tonic,
                mode: KeyMode::Major,
                fifths: tonic_mode_to_fifths(tonic, KeyMode::Major),
                confidence: r_maj as f32,
            };
        }

        let r_min = pearson_correlation(&hist, &rotated_min);
        if r_min > best_score {
            best_score = r_min;
            best_key = DetectedKey {
                tonic_pc: tonic,
                mode: KeyMode::Minor,
                fifths: tonic_mode_to_fifths(tonic, KeyMode::Minor),
                confidence: r_min as f32,
            };
        }
    }

    best_key
}

pub fn recognize_chord(midi_notes: &[u8], key: DetectedKey) -> Option<ChordSymbol> {
    if midi_notes.len() < 2 {
        return None;
    }

    let lowest_midi = *midi_notes.iter().min()?;
    let bass_pc = lowest_midi % 12;

    let mut pc_mask: u16 = 0;
    for &m in midi_notes {
        pc_mask |= 1u16 << (m % 12);
    }

    let unique_count = pc_mask.count_ones();
    if unique_count < 2 {
        return None;
    }

    if unique_count == 2 {
        let mut pcs = Vec::with_capacity(2);
        for pc in 0..12u8 {
            if (pc_mask & (1u16 << pc)) != 0 {
                pcs.push(pc);
            }
        }
        let other = if pcs[0] == bass_pc { pcs[1] } else { pcs[0] };
        if (other + 12 - bass_pc) % 12 == 7 {
            let root_name = pitch_class_name(bass_pc, key.prefer_flats());
            return Some(ChordSymbol {
                root_pc: bass_pc,
                quality: ChordQuality::Power,
                bass_pc: None,
                label: format!("{}5", root_name),
            });
        }
        return None;
    }

    const QUALITIES: [ChordQuality; 10] = [
        ChordQuality::Major,
        ChordQuality::Minor,
        ChordQuality::Dominant7,
        ChordQuality::Major7,
        ChordQuality::Minor7,
        ChordQuality::Sus2,
        ChordQuality::Sus4,
        ChordQuality::Diminished,
        ChordQuality::Augmented,
        ChordQuality::HalfDim7,
    ];

    let mut best: Option<(f32, u8, ChordQuality)> = None;

    for root in 0..12u8 {
        if (pc_mask & (1u16 << root)) == 0 {
            continue;
        }

        for quality in QUALITIES {
            let intervals = quality.intervals();
            let mut tmpl_mask: u16 = 0;
            for &iv in intervals {
                tmpl_mask |= 1u16 << ((root + iv) % 12);
            }

            let matched = (pc_mask & tmpl_mask).count_ones();
            let missing = (tmpl_mask & !pc_mask).count_ones();
            let extra = (pc_mask & !tmpl_mask).count_ones();

            if matched < 3 || missing > 1 {
                continue;
            }
            if intervals.len() == 3 && missing > 0 {
                continue;
            }
            if intervals.len() == 4 && missing == 1 {
                let fifth_bit = 1u16 << ((root + 7) % 12);
                if (tmpl_mask & !pc_mask) != fifth_bit {
                    continue;
                }
            }

            let mut score = (matched as f32) * 3.0
                - (missing as f32) * 1.8
                - (extra as f32) * 2.6;

            if root == bass_pc {
                score += 2.2;
            } else if (tmpl_mask & (1u16 << bass_pc)) != 0 {
                score += 0.6;
            } else {
                score -= 1.2;
            }

            if key.is_diatonic(root) {
                score += 0.55;
            }

            if matches!(quality, ChordQuality::Major | ChordQuality::Minor) {
                score += 0.25;
            }

            if best.map_or(true, |(best_s, _, _)| score > best_s) {
                best = Some((score, root, quality));
            }
        }
    }

    let (score, root_pc, quality) = best?;
    if score < 5.5 {
        return None;
    }

    let prefer_flats = key.prefer_flats();
    let root_name = pitch_class_name(root_pc, prefer_flats);
    let slash_bass = if bass_pc != root_pc {
        Some(bass_pc)
    } else {
        None
    };

    let label = match slash_bass {
        Some(b) => format!("{}{}/{}", root_name, quality.suffix(), pitch_class_name(b, prefer_flats)),
        None => format!("{}{}", root_name, quality.suffix()),
    };

    Some(ChordSymbol {
        root_pc,
        quality,
        bass_pc: slash_bass,
        label,
    })
}

// ============================================================================
// ESTIMATIVA DE BPM POR AUTOCORRELAÇÃO NORMALIZADA + PROTEÇÃO HALF-TIME
// ============================================================================

/// Estima o BPM via autocorrelação normalizada por sobreposição (sem viés para lags curtos),
/// com saliência musical (acordes/mudanças/duração) e proteção explícita contra dobramento (half-time).
pub fn estimate_bpm(events: &[TimedEvent], fallback_bpm: u16) -> u16 {
    if events.len() < 4 {
        return fallback_bpm;
    }

    const STEP_MS: u64 = 10; // 100 Hz
    const MAX_WINDOW_MS: u64 = 24_000;
    const MIN_LAG: usize = 35;  // ~171 BPM máx.
    const MAX_LAG: usize = 108; // ~55.5 BPM mín.

    let end_ms = events.last().map(|e| e.start_ms).unwrap_or(0);
    let start_cutoff = end_ms.saturating_sub(MAX_WINDOW_MS);

    // Filtra micro-aglomerados (< 85 ms) que vêm de palhetada dedilhada/arpejada
    let mut active: Vec<&TimedEvent> = Vec::new();
    for ev in events.iter().filter(|e| e.start_ms >= start_cutoff) {
        if let Some(prev) = active.last() {
            if ev.start_ms.saturating_sub(prev.start_ms) < 85 {
                continue;
            }
        }
        active.push(ev);
    }

    if active.len() < 4 {
        return fallback_bpm;
    }

    let t0 = active[0].start_ms;
    let span_ms = active.last().unwrap().start_ms.saturating_sub(t0);
    if span_ms < 900 {
        return fallback_bpm;
    }

    let len = ((span_ms / STEP_MS) as usize) + 12;
    if len <= MIN_LAG + 10 {
        return fallback_bpm;
    }

    // Curva de novidade ponderada por saliência musical (acordes, notas longas e mudança de baixo)
    let mut env = vec![0.0f64; len];
    const KERNEL: [(isize, f64); 5] = [(-2, 0.20), (-1, 0.65), (0, 1.0), (1, 0.65), (2, 0.20)];
    let mut prev_bass: Option<u8> = None;

    for ev in &active {
        let center = ((ev.start_ms.saturating_sub(t0)) / STEP_MS) as isize + 2;
        let bass = ev.notes.iter().copied().min();
        let dur_bonus = if ev.duration_ms >= 360 {
            0.55
        } else if ev.duration_ms >= 220 {
            0.25
        } else {
            0.0
        };
        let chord_bonus = 0.25 * (ev.notes.len().saturating_sub(1).min(3) as f64);
        let bass_change_bonus = if prev_bass.is_some() && bass != prev_bass {
            0.30
        } else {
            0.0
        };
        prev_bass = bass;

        let weight = 1.0 + dur_bonus + chord_bonus + bass_change_bonus;
        for &(offset, kw) in &KERNEL {
            let idx = center + offset;
            if idx >= 0 && (idx as usize) < len {
                env[idx as usize] += kw * weight;
            }
        }
    }

    let max_lag = MAX_LAG.min(len.saturating_sub(10));
    if max_lag <= MIN_LAG {
        return fallback_bpm;
    }

    // Autocorrelação normalizada localmente (cosine similarity sobre a região sobreposta):
    // impede que lags curtos (150-180 BPM) vençam apenas por terem mais amostras na soma.
    let unbiased_autocorr = |lag: usize| -> f64 {
        if lag == 0 || lag + 8 >= len {
            return 0.0;
        }
        let n = len - lag;
        let mut dot = 0.0;
        let mut e1 = 0.0;
        let mut e2 = 0.0;
        for i in 0..n {
            let a = env[i];
            let b = env[i + lag];
            dot += a * b;
            e1 += a * a;
            e2 += b * b;
        }
        let denom = (e1 * e2).sqrt();
        if denom <= 1e-9 {
            0.0
        } else {
            let overlap_ratio = (n as f64) / (len as f64);
            let conf = (overlap_ratio / 0.30).min(1.0);
            (dot / denom) * conf
        }
    };

    let mut raw_r = vec![0.0f64; max_lag + 1];
    let mut scores = vec![0.0f64; max_lag + 1];
    let mut best_lag = 60usize; // 100 BPM padrão
    let mut best_score = 0.0f64;

    for lag in MIN_LAG..=max_lag {
        let r1 = unbiased_autocorr(lag);
        raw_r[lag] = r1;

        let mut num = r1;
        let mut w_sum = 1.0;
        if lag * 2 + 12 < len {
            let r2 = unbiased_autocorr(lag * 2);
            num += 0.28 * r2;
            w_sum += 0.28;
        }
        let harmonic_r = num / w_sum;

        // Prior perceptual centrado em 96 BPM (desencoraja leituras aceleradas 150-175 BPM)
        let bpm = 60_000.0 / ((lag as f64) * (STEP_MS as f64));
        let log_dist = (bpm / 96.0).ln() / 0.36;
        let prior = (-0.5 * log_dist * log_dist).exp();

        let score = harmonic_r * (0.25 + 0.75 * prior);
        scores[lag] = score;

        if score > best_score {
            best_score = score;
            best_lag = lag;
        }
    }

    if best_score < 0.06 {
        return fallback_bpm;
    }

    // Proteção explícita contra erro de oitava rítmica (subdivisão de colcheias lida como pulso)
    let cand_bpm = 60_000.0 / ((best_lag as f64) * (STEP_MS as f64));
    if cand_bpm > 130.0 {
        let double_lag = best_lag * 2;
        if double_lag <= max_lag {
            let r_fast = raw_r[best_lag];
            let r_half = unbiased_autocorr(double_lag);
            if r_fast > 0.0 && r_half >= r_fast * 0.72 {
                best_lag = double_lag;
            }
        }
    }

    let refined_lag = if best_lag > MIN_LAG && best_lag < max_lag {
        let s0 = scores[best_lag - 1];
        let s1 = scores[best_lag];
        let s2 = scores[best_lag + 1];
        let denom = 2.0 * (2.0 * s1 - s2 - s0);
        if denom.abs() > 1e-9 {
            let delta = ((s2 - s0) / denom).clamp(-0.5, 0.5);
            (best_lag as f64) + delta
        } else {
            best_lag as f64
        }
    } else {
        best_lag as f64
    };

    let bpm = (60_000.0 / (refined_lag * (STEP_MS as f64))).round() as u16;
    bpm.clamp(56, 172)
}

// ============================================================================
// QUANTIZAÇÃO COM FIGURAS PONTUADAS, LIGADURAS E COMPASSOS FLEXÍVEIS
// ============================================================================

const VALID_FIGURES: [u8; 8] = [16, 12, 8, 6, 4, 3, 2, 1];

fn quantize_note_ticks(raw_ticks: f64, max_ticks: u8) -> u8 {
    let clamped = raw_ticks.clamp(1.0, max_ticks as f64);

    let mut best_fig = None;
    let mut best_err = 0.36f64;
    for &fig in &VALID_FIGURES {
        if fig > max_ticks {
            continue;
        }
        let err = (clamped - (fig as f64)).abs();
        if err < best_err {
            best_err = err;
            best_fig = Some(fig);
        }
    }

    best_fig.unwrap_or_else(|| clamped.round().clamp(1.0, max_ticks as f64) as u8)
}

fn push_rest_ticks(
    measures: &mut Vec<QuantizedMeasure>,
    current_beats: &mut Vec<QuantizedBeat>,
    current_ticks: &mut u8,
    measure_ticks: u8,
    mut remaining: u8,
) {
    while remaining > 0 {
        let space = measure_ticks.saturating_sub(*current_ticks);
        let d = VALID_FIGURES
            .into_iter()
            .find(|&cand| cand <= remaining && cand <= space)
            .unwrap_or(1);

        current_beats.push(QuantizedBeat {
            duration_ticks: d,
            midi_notes: Vec::new(),
            voicing: None,
            bend_semitones: 0.0,
            chord: None,
            tie_from_prev: false,
            tie_to_next: false,
        });
        *current_ticks += d;
        remaining -= d;

        if *current_ticks >= measure_ticks {
            measures.push(QuantizedMeasure {
                beats: std::mem::take(current_beats),
            });
            *current_ticks = 0;
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn push_tied_note_ticks(
    measures: &mut Vec<QuantizedMeasure>,
    current_beats: &mut Vec<QuantizedBeat>,
    current_ticks: &mut u8,
    measure_ticks: u8,
    total_ticks: u8,
    midi_notes: &[u8],
    voicing: &Option<ChordVoicing>,
    bend_semitones: f32,
    chord: &Option<ChordSymbol>,
) {
    let mut segments: Vec<u8> = Vec::new();
    let mut sim_tick = *current_ticks;
    let mut remaining = total_ticks;

    while remaining > 0 {
        let space = measure_ticks.saturating_sub(sim_tick);
        let d = VALID_FIGURES
            .into_iter()
            .find(|&cand| cand <= remaining && cand <= space)
            .unwrap_or(1);
        segments.push(d);
        sim_tick += d;
        remaining -= d;
        if sim_tick >= measure_ticks {
            sim_tick = 0;
        }
    }

    let seg_count = segments.len();
    for (idx, d) in segments.into_iter().enumerate() {
        let is_first = idx == 0;
        let is_last = idx + 1 == seg_count;

        current_beats.push(QuantizedBeat {
            duration_ticks: d,
            midi_notes: midi_notes.to_vec(),
            voicing: voicing.clone(),
            bend_semitones: if is_first { bend_semitones } else { 0.0 },
            chord: if is_first { chord.clone() } else { None },
            tie_from_prev: !is_first,
            tie_to_next: !is_last,
        });
        *current_ticks += d;

        if *current_ticks >= measure_ticks {
            measures.push(QuantizedMeasure {
                beats: std::mem::take(current_beats),
            });
            *current_ticks = 0;
        }
    }
}

pub fn build_empty_score(
    title: &str,
    tuning_low_to_high: &[u8],
    bpm: u16,
    time_signature: TimeSignature,
) -> QuantizedScore {
    let measure_ticks = time_signature.measure_ticks();
    let mut measures = Vec::new();
    let mut current_beats = Vec::new();
    let mut current_ticks = 0u8;

    push_rest_ticks(
        &mut measures,
        &mut current_beats,
        &mut current_ticks,
        measure_ticks,
        measure_ticks,
    );

    QuantizedScore {
        title: title.to_string(),
        bpm,
        time_signature,
        key: DetectedKey::default(),
        tuning_low_to_high: tuning_low_to_high.to_vec(),
        measures,
        midi_program: 27,
        show_tab_staff: true,
        clef: ClefKind::Treble8vb,
    }
}

pub fn build_quantized_score(
    title: &str,
    events: &[TimedEvent],
    voicings: &[Option<ChordVoicing>],
    tuning_low_to_high: &[u8],
    bpm_override: Option<u16>,
    time_signature: TimeSignature,
) -> QuantizedScore {
    let bpm = bpm_override.unwrap_or_else(|| estimate_bpm(events, 100));
    let key = detect_key(events);
    let measure_ticks = time_signature.measure_ticks();
    let sixteenth_ms = 60_000.0 / ((bpm as f64) * 4.0);

    if events.is_empty() {
        return build_empty_score(title, tuning_low_to_high, bpm, time_signature);
    }

    // 1. Ancoragem absoluta na grade de semicolcheias (zero deriva acumulada)
    let t0 = events[0].start_ms;
    let mut grid_starts: Vec<u32> = Vec::with_capacity(events.len());
    for ev in events {
        let rel_ms = ev.start_ms.saturating_sub(t0) as f64;
        let mut tick = (rel_ms / sixteenth_ms).round() as u32;
        if let Some(&prev) = grid_starts.last() {
            if tick <= prev {
                tick = prev + 1;
            }
        }
        grid_starts.push(tick);
    }

    let mut measures: Vec<QuantizedMeasure> = Vec::new();
    let mut current_beats: Vec<QuantizedBeat> = Vec::new();
    let mut current_ticks: u8 = 0;
    let mut cursor_abs: u32 = 0;

    for (i, ev) in events.iter().enumerate() {
        let start_tick = grid_starts[i];

        // Preenche qualquer silêncio real até o ataque ancorado (limitado a 2 compassos)
        if start_tick > cursor_abs {
            let mut rest_rem = (start_tick - cursor_abs).min((measure_ticks as u32) * 2);
            while rest_rem > 0 {
                let chunk = rest_rem.min(measure_ticks as u32) as u8;
                push_rest_ticks(
                    &mut measures,
                    &mut current_beats,
                    &mut current_ticks,
                    measure_ticks,
                    chunk,
                );
                rest_rem -= chunk as u32;
            }
            cursor_abs = start_tick;
        }

        let max_slot = grid_starts
            .get(i + 1)
            .map(|&next_t| next_t.saturating_sub(start_tick).max(1))
            .unwrap_or(measure_ticks as u32)
            .min(measure_ticks as u32) as u8;

        let raw_ticks = (ev.duration_ms as f64) / sixteenth_ms;
        let note_ticks = quantize_note_ticks(raw_ticks, max_slot);

        let voicing = voicings.get(i).cloned().flatten();
        let chord = recognize_chord(&ev.notes, key);

        push_tied_note_ticks(
            &mut measures,
            &mut current_beats,
            &mut current_ticks,
            measure_ticks,
            note_ticks,
            &ev.notes,
            &voicing,
            ev.bend_semitones,
            &chord,
        );
        cursor_abs += note_ticks as u32;
    }

    if current_ticks > 0 {
        let pad = measure_ticks.saturating_sub(current_ticks);
        push_rest_ticks(
            &mut measures,
            &mut current_beats,
            &mut current_ticks,
            measure_ticks,
            pad,
        );
    }

    if measures.is_empty() {
        return build_empty_score(title, tuning_low_to_high, bpm, time_signature);
    }

    QuantizedScore {
        title: title.to_string(),
        bpm,
        time_signature,
        key,
        tuning_low_to_high: tuning_low_to_high.to_vec(),
        measures,
        midi_program: 27,
        show_tab_staff: true,
        clef: ClefKind::Treble8vb,
    }
}

// ============================================================================
// SERIALIZAÇÃO ALPHATEX (Pauta Pura ou Partitura+Tab, \instrument, Pontos, Ties)
// ============================================================================

fn format_alphatex_pos(pos: FretPos, num_strings: usize, is_tied: bool) -> String {
    let atex_string = num_strings.saturating_sub(pos.string);
    if is_tied {
        format!("-.{}", atex_string)
    } else {
        format!("{}.{}", pos.fret, atex_string)
    }
}

pub fn score_to_alphatex(
    score: &QuantizedScore,
    alphatex_tuning_high_to_low: &str,
    subtitle: &str,
) -> String {
    let num_strings = score.tuning_low_to_high.len();
    let staff_block = if score.show_tab_staff {
        format!(
            "\\staff {{ score tabs }}\n\\tuning ({})\n",
            alphatex_tuning_high_to_low
        )
    } else {
        "\\staff { score }\n".to_string()
    };

    let header = format!(
        "\\title \"{}\"\n\
         \\subtitle \"{}\"\n\
         \\tempo {}\n\
         \\track \"Instrumento\"\n\
         \\instrument {}\n\
         {}\
         \\clef {}\n\
         \\ts {}\n\
         \\ks {}\n",
        score.title.replace('"', "'"),
        subtitle.replace('"', "'"),
        score.bpm,
        score.midi_program,
        staff_block,
        score.clef.alphatex_clef(),
        score.time_signature.alphatex_ts(),
        score.key.alphatex_ks()
    );

    let mut bar_strings = Vec::with_capacity(score.measures.len());
    let mut last_rendered_chord: Option<String> = None;

    for measure in &score.measures {
        let mut tokens: Vec<String> = Vec::new();
        let mut active_div: Option<u8> = None;

        for beat in &measure.beats {
            let div = beat.alphatex_div();
            if active_div != Some(div) {
                tokens.push(format!(":{}", div));
                active_div = Some(div);
            }

            if beat.is_rest() {
                if beat.is_dotted() {
                    tokens.push("r { d }".to_string());
                } else {
                    tokens.push("r".to_string());
                }
                continue;
            }

            let chord_prop = if !beat.tie_from_prev {
                match &beat.chord {
                    Some(ch) if last_rendered_chord.as_deref() != Some(ch.label.as_str()) => {
                        last_rendered_chord = Some(ch.label.clone());
                        Some(format!("ch \"{}\"", ch.label.replace('"', "'")))
                    }
                    _ => None,
                }
            } else {
                None
            };

            if !score.show_tab_staff {
                let mut pitched: Vec<String> = beat
                    .midi_notes
                    .iter()
                    .map(|&m| {
                        let name = midi_to_note_name_for_key(m, score.key);
                        if beat.tie_from_prev {
                            format!("{} {{ t }}", name)
                        } else {
                            name
                        }
                    })
                    .collect();
                if pitched.is_empty() {
                    tokens.push(if beat.is_dotted() {
                        "r { d }".to_string()
                    } else {
                        "r".to_string()
                    });
                    continue;
                }
                let body = if pitched.len() == 1 {
                    pitched.remove(0)
                } else {
                    format!("({})", pitched.join(" "))
                };
                let mut beat_props: Vec<String> = Vec::new();
                if beat.is_dotted() {
                    beat_props.push("d".to_string());
                }
                if let Some(ch_p) = chord_prop {
                    beat_props.push(ch_p);
                }
                if beat_props.is_empty() {
                    tokens.push(body);
                } else {
                    tokens.push(format!("{} {{ {} }}", body, beat_props.join(" ")));
                }
                continue;
            }

            match &beat.voicing {
                Some(v) if v.len() == 1 => {
                    let base = format_alphatex_pos(v[0], num_strings, beat.tie_from_prev);
                    let mut props: Vec<String> = Vec::new();

                    if !beat.tie_from_prev && beat.bend_semitones >= 0.45 {
                        let qtones = (beat.bend_semitones * 4.0).round().clamp(2.0, 8.0) as u8;
                        props.push(format!("b (0 {})", qtones));
                    }
                    if beat.is_dotted() {
                        props.push("d".to_string());
                    }
                    if let Some(ch_p) = chord_prop {
                        props.push(ch_p);
                    }

                    if props.is_empty() {
                        tokens.push(base);
                    } else {
                        tokens.push(format!("{} {{ {} }}", base, props.join(" ")));
                    }
                }
                Some(v) if v.len() > 1 => {
                    let chord_notes: Vec<String> = v
                        .iter()
                        .map(|&p| format_alphatex_pos(p, num_strings, beat.tie_from_prev))
                        .collect();
                    let body = format!("({})", chord_notes.join(" "));
                    let mut beat_props: Vec<String> = Vec::new();
                    if beat.is_dotted() {
                        beat_props.push("d".to_string());
                    }
                    if let Some(ch_p) = chord_prop {
                        beat_props.push(ch_p);
                    }

                    if beat_props.is_empty() {
                        tokens.push(body);
                    } else {
                        tokens.push(format!("{} {{ {} }}", body, beat_props.join(" ")));
                    }
                }
                _ => {
                    if beat.is_dotted() {
                        tokens.push("r { d }".to_string());
                    } else {
                        tokens.push("r".to_string());
                    }
                }
            }
        }

        bar_strings.push(tokens.join(" "));
    }

    format!("{}{}\n", header, bar_strings.join(" |\n"))
}

// ============================================================================
// EXPORTAÇÃO MUSICXML 4.0
// ============================================================================

fn escape_xml(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

fn write_pitch_xml(out: &mut String, midi: u8, prefer_flats: bool) {
    let (step, alter, octave) = midi_to_xml_pitch(midi, prefer_flats);
    out.push_str("        <pitch>\n");
    out.push_str(&format!("          <step>{}</step>\n", step));
    if alter != 0 {
        out.push_str(&format!("          <alter>{}</alter>\n", alter));
    }
    out.push_str(&format!("          <octave>{}</octave>\n", octave));
    out.push_str("        </pitch>\n");
}

fn write_harmony_xml(out: &mut String, chord: &ChordSymbol, prefer_flats: bool) {
    let (root_step, root_alter) = pitch_class_to_xml_step(chord.root_pc, prefer_flats);
    out.push_str("      <harmony>\n");
    out.push_str("        <root>\n");
    out.push_str(&format!("          <root-step>{}</root-step>\n", root_step));
    if root_alter != 0 {
        out.push_str(&format!("          <root-alter>{}</root-alter>\n", root_alter));
    }
    out.push_str("        </root>\n");
    out.push_str(&format!(
        "        <kind text=\"{}\">{}</kind>\n",
        escape_xml(chord.quality.suffix()),
        chord.quality.musicxml_kind()
    ));
    if let Some(bass_pc) = chord.bass_pc {
        let (bass_step, bass_alter) = pitch_class_to_xml_step(bass_pc, prefer_flats);
        out.push_str("        <bass>\n");
        out.push_str(&format!("          <bass-step>{}</bass-step>\n", bass_step));
        if bass_alter != 0 {
            out.push_str(&format!("          <bass-alter>{}</bass-alter>\n", bass_alter));
        }
        out.push_str("        </bass>\n");
    }
    out.push_str("      </harmony>\n");
}

pub fn export_musicxml(score: &QuantizedScore) -> String {
    let num_strings = score.tuning_low_to_high.len();
    let measure_ticks = score.time_signature.measure_ticks();
    let prefer_flats = score.key.prefer_flats();
    let mode_xml = match score.key.mode {
        KeyMode::Major => "major",
        KeyMode::Minor => "minor",
    };

    let mut xml = String::with_capacity(4096);

    xml.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
    xml.push_str("<!DOCTYPE score-partwise PUBLIC \"-//Recordare//DTD MusicXML 4.0 Partwise//EN\" \"http://www.musicxml.org/dtds/partwise.dtd\">\n");
    xml.push_str("<score-partwise version=\"4.0\">\n");
    xml.push_str("  <work>\n");
    xml.push_str(&format!(
        "    <work-title>{}</work-title>\n",
        escape_xml(&score.title)
    ));
    xml.push_str("  </work>\n");
    xml.push_str("  <identification>\n");
    xml.push_str("    <encoding><software>Tab Transcriber</software></encoding>\n");
    xml.push_str("  </identification>\n");
    xml.push_str("  <part-list>\n");
    xml.push_str("    <score-part id=\"P1\">\n");
    xml.push_str("      <part-name>Instrumento</part-name>\n");
    xml.push_str("    </score-part>\n");
    xml.push_str("  </part-list>\n");
    xml.push_str("  <part id=\"P1\">\n");

    let mut last_harmony: Option<String> = None;

    for (m_idx, measure) in score.measures.iter().enumerate() {
        xml.push_str(&format!("    <measure number=\"{}\">\n", m_idx + 1));

        if m_idx == 0 {
            xml.push_str("      <attributes>\n");
            xml.push_str("        <divisions>4</divisions>\n");
            xml.push_str("        <key>\n");
            xml.push_str(&format!("          <fifths>{}</fifths>\n", score.key.fifths));
            xml.push_str(&format!("          <mode>{}</mode>\n", mode_xml));
            xml.push_str("        </key>\n");
            xml.push_str(&format!(
                "        <time><beats>{}</beats><beat-type>{}</beat-type></time>\n",
                score.time_signature.numerator, score.time_signature.denominator
            ));
            let (clef_sign, clef_line, oct_change) = match score.clef {
                ClefKind::Treble => ("G", 2, None),
                ClefKind::Treble8vb => ("G", 2, Some(-1)),
                ClefKind::Bass => ("F", 4, if score.show_tab_staff { Some(-1) } else { None }),
            };
            if score.show_tab_staff {
                xml.push_str("        <staves>2</staves>\n");
            }
            xml.push_str("        <clef number=\"1\">\n");
            xml.push_str(&format!("          <sign>{}</sign>\n", clef_sign));
            xml.push_str(&format!("          <line>{}</line>\n", clef_line));
            if let Some(oc) = oct_change {
                xml.push_str(&format!("          <clef-octave-change>{}</clef-octave-change>\n", oc));
            }
            xml.push_str("        </clef>\n");
            if score.show_tab_staff {
                xml.push_str("        <clef number=\"2\">\n");
                xml.push_str("          <sign>TAB</sign>\n");
                xml.push_str("          <line>5</line>\n");
                xml.push_str("        </clef>\n");
                xml.push_str("        <staff-details number=\"2\">\n");
                xml.push_str(&format!("          <staff-lines>{}</staff-lines>\n", num_strings));
                for (idx, &open_midi) in score.tuning_low_to_high.iter().enumerate() {
                    let (step, alter, octave) = midi_to_xml_pitch(open_midi, prefer_flats);
                    xml.push_str(&format!("          <staff-tuning line=\"{}\">\n", idx + 1));
                    xml.push_str(&format!("            <tuning-step>{}</tuning-step>\n", step));
                    if alter != 0 {
                        xml.push_str(&format!("            <tuning-alter>{}</tuning-alter>\n", alter));
                    }
                    xml.push_str(&format!(
                        "            <tuning-octave>{}</tuning-octave>\n",
                        octave
                    ));
                    xml.push_str("          </staff-tuning>\n");
                }
                xml.push_str("        </staff-details>\n");
            }
            xml.push_str("      </attributes>\n");
            xml.push_str("      <direction placement=\"above\">\n");
            xml.push_str("        <direction-type>\n");
            xml.push_str("          <metronome>\n");
            xml.push_str("            <beat-unit>quarter</beat-unit>\n");
            xml.push_str(&format!("            <per-minute>{}</per-minute>\n", score.bpm));
            xml.push_str("          </metronome>\n");
            xml.push_str("        </direction-type>\n");
            xml.push_str(&format!("        <sound tempo=\"{}\"/>\n", score.bpm));
            xml.push_str("      </direction>\n");
        }

        for beat in &measure.beats {
            if !beat.tie_from_prev {
                if let Some(ch) = &beat.chord {
                    if last_harmony.as_deref() != Some(ch.label.as_str()) {
                        write_harmony_xml(&mut xml, ch, prefer_flats);
                        last_harmony = Some(ch.label.clone());
                    }
                }
            }

            if beat.is_rest() {
                xml.push_str("      <note>\n");
                xml.push_str("        <rest/>\n");
                xml.push_str(&format!("        <duration>{}</duration>\n", beat.duration_ticks));
                xml.push_str("        <voice>1</voice>\n");
                xml.push_str(&format!("        <type>{}</type>\n", beat.musicxml_type()));
                if beat.is_dotted() {
                    xml.push_str("        <dot/>\n");
                }
                xml.push_str("        <staff>1</staff>\n");
                xml.push_str("      </note>\n");
            } else {
                for (n_idx, &midi) in beat.midi_notes.iter().enumerate() {
                    xml.push_str("      <note>\n");
                    if n_idx > 0 {
                        xml.push_str("        <chord/>\n");
                    }
                    write_pitch_xml(&mut xml, midi, prefer_flats);
                    xml.push_str(&format!("        <duration>{}</duration>\n", beat.duration_ticks));
                    if beat.tie_from_prev {
                        xml.push_str("        <tie type=\"stop\"/>\n");
                    }
                    if beat.tie_to_next {
                        xml.push_str("        <tie type=\"start\"/>\n");
                    }
                    xml.push_str("        <voice>1</voice>\n");
                    xml.push_str(&format!("        <type>{}</type>\n", beat.musicxml_type()));
                    if beat.is_dotted() {
                        xml.push_str("        <dot/>\n");
                    }
                    xml.push_str("        <staff>1</staff>\n");
                    if beat.tie_from_prev || beat.tie_to_next {
                        xml.push_str("        <notations>\n");
                        if beat.tie_from_prev {
                            xml.push_str("          <tied type=\"stop\"/>\n");
                        }
                        if beat.tie_to_next {
                            xml.push_str("          <tied type=\"start\"/>\n");
                        }
                        xml.push_str("        </notations>\n");
                    }
                    xml.push_str("      </note>\n");
                }
            }
        }

        if !score.show_tab_staff {
            xml.push_str("    </measure>\n");
            continue;
        }

        xml.push_str(&format!("      <backup><duration>{}</duration></backup>\n", measure_ticks));

        for beat in &measure.beats {
            let tab_notes: Vec<(u8, FretPos)> = match &beat.voicing {
                Some(v) if !v.is_empty() => v
                    .iter()
                    .filter_map(|&pos| {
                        score
                            .tuning_low_to_high
                            .get(pos.string)
                            .map(|&open| (open.saturating_add(pos.fret), pos))
                    })
                    .collect(),
                _ => Vec::new(),
            };

            if tab_notes.is_empty() {
                xml.push_str("      <note>\n");
                xml.push_str("        <rest/>\n");
                xml.push_str(&format!("        <duration>{}</duration>\n", beat.duration_ticks));
                xml.push_str("        <voice>2</voice>\n");
                xml.push_str(&format!("        <type>{}</type>\n", beat.musicxml_type()));
                if beat.is_dotted() {
                    xml.push_str("        <dot/>\n");
                }
                xml.push_str("        <staff>2</staff>\n");
                xml.push_str("      </note>\n");
            } else {
                let note_count = tab_notes.len();
                for (n_idx, (midi, pos)) in tab_notes.into_iter().enumerate() {
                    let xml_string = num_strings.saturating_sub(pos.string);
                    xml.push_str("      <note>\n");
                    if n_idx > 0 {
                        xml.push_str("        <chord/>\n");
                    }
                    write_pitch_xml(&mut xml, midi, prefer_flats);
                    xml.push_str(&format!("        <duration>{}</duration>\n", beat.duration_ticks));
                    if beat.tie_from_prev {
                        xml.push_str("        <tie type=\"stop\"/>\n");
                    }
                    if beat.tie_to_next {
                        xml.push_str("        <tie type=\"start\"/>\n");
                    }
                    xml.push_str("        <voice>2</voice>\n");
                    xml.push_str(&format!("        <type>{}</type>\n", beat.musicxml_type()));
                    if beat.is_dotted() {
                        xml.push_str("        <dot/>\n");
                    }
                    xml.push_str("        <staff>2</staff>\n");
                    xml.push_str("        <notations>\n");
                    if beat.tie_from_prev {
                        xml.push_str("          <tied type=\"stop\"/>\n");
                    }
                    if beat.tie_to_next {
                        xml.push_str("          <tied type=\"start\"/>\n");
                    }
                    xml.push_str("          <technical>\n");
                    xml.push_str(&format!("            <string>{}</string>\n", xml_string));
                    xml.push_str(&format!("            <fret>{}</fret>\n", pos.fret));
                    if !beat.tie_from_prev && beat.bend_semitones >= 0.45 && n_idx + 1 == note_count {
                        xml.push_str(&format!(
                            "            <bend><bend-alter>{:.1}</bend-alter></bend>\n",
                            beat.bend_semitones
                        ));
                    }
                    xml.push_str("          </technical>\n");
                    xml.push_str("        </notations>\n");
                    xml.push_str("      </note>\n");
                }
            }
        }

        xml.push_str("    </measure>\n");
    }

    xml.push_str("  </part>\n");
    xml.push_str("</score-partwise>\n");
    xml
}

// ============================================================================
// EXPORTAÇÃO STANDARD MIDI FILE (SMF FORMAT 0)
// ============================================================================

fn write_vlq(buf: &mut Vec<u8>, mut value: u32) {
    let mut stack = [0u8; 5];
    let mut len = 0usize;
    stack[len] = (value & 0x7F) as u8;
    len += 1;
    value >>= 7;

    while value > 0 {
        stack[len] = ((value & 0x7F) as u8) | 0x80;
        len += 1;
        value >>= 7;
    }

    for i in (0..len).rev() {
        buf.push(stack[i]);
    }
}

pub fn export_midi(score: &QuantizedScore) -> Vec<u8> {
    const TPQ: u16 = 480;
    const TICKS_PER_SIXTEENTH: u32 = (TPQ as u32) / 4;

    let mut track: Vec<u8> = Vec::with_capacity(1024);

    let us_per_quarter: u32 = 60_000_000 / (score.bpm.max(30) as u32);
    write_vlq(&mut track, 0);
    track.extend_from_slice(&[
        0xFF,
        0x51,
        0x03,
        ((us_per_quarter >> 16) & 0xFF) as u8,
        ((us_per_quarter >> 8) & 0xFF) as u8,
        (us_per_quarter & 0xFF) as u8,
    ]);

    let clocks_per_click = if score.time_signature.denominator == 8
        && score.time_signature.numerator % 3 == 0
    {
        36u8
    } else {
        24u8
    };
    write_vlq(&mut track, 0);
    track.extend_from_slice(&[
        0xFF,
        0x58,
        0x04,
        score.time_signature.numerator,
        score.time_signature.midi_denominator_pow2(),
        clocks_per_click,
        8,
    ]);

    let mi = match score.key.mode {
        KeyMode::Major => 0u8,
        KeyMode::Minor => 1u8,
    };
    write_vlq(&mut track, 0);
    track.extend_from_slice(&[0xFF, 0x59, 0x02, score.key.fifths as u8, mi]);

    write_vlq(&mut track, 0);
    track.extend_from_slice(&[0xC0, score.midi_program & 0x7F]);

    let mut pending_delta: u32 = 0;
    let mut held_notes: Vec<u8> = Vec::new();
    let mut held_ticks: u32 = 0;

    for measure in &score.measures {
        for beat in &measure.beats {
            let dur_ticks = (beat.duration_ticks as u32) * TICKS_PER_SIXTEENTH;

            if beat.is_rest() {
                if !held_notes.is_empty() {
                    for (idx, &midi) in held_notes.iter().enumerate() {
                        let delta = if idx == 0 { std::mem::take(&mut held_ticks) } else { 0 };
                        write_vlq(&mut track, delta);
                        track.extend_from_slice(&[0x80, midi & 0x7F, 0]);
                    }
                    held_notes.clear();
                }
                pending_delta += dur_ticks;
                continue;
            }

            if beat.tie_from_prev && !held_notes.is_empty() {
                held_ticks += dur_ticks;
                if !beat.tie_to_next {
                    for (idx, &midi) in held_notes.iter().enumerate() {
                        let delta = if idx == 0 { std::mem::take(&mut held_ticks) } else { 0 };
                        write_vlq(&mut track, delta);
                        track.extend_from_slice(&[0x80, midi & 0x7F, 0]);
                    }
                    held_notes.clear();
                }
                continue;
            }

            if !held_notes.is_empty() {
                for (idx, &midi) in held_notes.iter().enumerate() {
                    let delta = if idx == 0 { std::mem::take(&mut held_ticks) } else { 0 };
                    write_vlq(&mut track, delta);
                    track.extend_from_slice(&[0x80, midi & 0x7F, 0]);
                }
                held_notes.clear();
            }

            for (idx, &midi) in beat.midi_notes.iter().enumerate() {
                let delta = if idx == 0 {
                    std::mem::take(&mut pending_delta)
                } else {
                    0
                };
                write_vlq(&mut track, delta);
                track.extend_from_slice(&[0x90, midi & 0x7F, 96]);
            }

            if beat.tie_to_next {
                held_notes = beat.midi_notes.clone();
                held_ticks = dur_ticks;
            } else {
                for (idx, &midi) in beat.midi_notes.iter().enumerate() {
                    let delta = if idx == 0 { dur_ticks } else { 0 };
                    write_vlq(&mut track, delta);
                    track.extend_from_slice(&[0x80, midi & 0x7F, 0]);
                }
            }
        }
    }

    if !held_notes.is_empty() {
        for (idx, &midi) in held_notes.iter().enumerate() {
            let delta = if idx == 0 { std::mem::take(&mut held_ticks) } else { 0 };
            write_vlq(&mut track, delta);
            track.extend_from_slice(&[0x80, midi & 0x7F, 0]);
        }
    }

    write_vlq(&mut track, pending_delta);
    track.extend_from_slice(&[0xFF, 0x2F, 0x00]);

    let mut smf: Vec<u8> = Vec::with_capacity(14 + 8 + track.len());
    smf.extend_from_slice(b"MThd");
    smf.extend_from_slice(&6u32.to_be_bytes());
    smf.extend_from_slice(&0u16.to_be_bytes());
    smf.extend_from_slice(&1u16.to_be_bytes());
    smf.extend_from_slice(&TPQ.to_be_bytes());

    smf.extend_from_slice(b"MTrk");
    smf.extend_from_slice(&(track.len() as u32).to_be_bytes());
    smf.extend_from_slice(&track);

    smf
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn autocorrelation_estimates_bpm_reliably_without_doubling() {
        // 1. Pulso a 120 BPM (500 ms) com micro-timing humano
        let steady: Vec<TimedEvent> = [0, 505, 995, 1502, 2000, 2494, 3008, 3500]
            .into_iter()
            .map(|t| TimedEvent {
                start_ms: t,
                duration_ms: 420,
                notes: vec![64],
                bend_semitones: 0.0,
            })
            .collect();
        let bpm_120 = estimate_bpm(&steady, 90);
        assert!(bpm_120.abs_diff(120) <= 2, "esperado ~120, obteve {}", bpm_120);

        // 2. Pulso mais lento a 80 BPM (750 ms) — não deve dobrar para 160 BPM
        let slow: Vec<TimedEvent> = (0..8)
            .map(|k| TimedEvent {
                start_ms: k * 750,
                duration_ms: 600,
                notes: vec![50, 57],
                bend_semitones: 0.0,
            })
            .collect();
        let bpm_80 = estimate_bpm(&slow, 120);
        assert!(bpm_80.abs_diff(80) <= 2, "esperado ~80, obteve {}", bpm_80);

        // 3. Riff contínuo de colcheias a 90 BPM (IOI = 333 ms) — não deve dobrar para 180 BPM
        let eighths_90: Vec<TimedEvent> = (0..12)
            .map(|k| TimedEvent {
                start_ms: (k as u64) * 333,
                duration_ms: 280,
                notes: if k % 2 == 0 { vec![45, 52] } else { vec![57] },
                bend_semitones: 0.0,
            })
            .collect();
        let bpm_90 = estimate_bpm(&eighths_90, 120);
        assert!(bpm_90.abs_diff(90) <= 3, "colcheias a 90 BPM não devem dobrar (obteve {})", bpm_90);
    }

    #[test]
    fn supports_dotted_figures_and_ties_across_barlines() {
        let tuning = [40, 45, 50, 55, 59, 64];
        let events = vec![
            TimedEvent { start_ms: 0, duration_ms: 750, notes: vec![64], bend_semitones: 0.0 },
            TimedEvent { start_ms: 750, duration_ms: 750, notes: vec![67], bend_semitones: 0.0 },
            TimedEvent { start_ms: 1500, duration_ms: 750, notes: vec![69], bend_semitones: 0.0 },
        ];
        let voicings = vec![
            Some(vec![FretPos { string: 5, fret: 0 }]),
            Some(vec![FretPos { string: 5, fret: 3 }]),
            Some(vec![FretPos { string: 5, fret: 5 }]),
        ];

        let score = build_quantized_score(
            "Sincope",
            &events,
            &voicings,
            &tuning,
            Some(120),
            TimeSignature::new(4, 4),
        );

        assert_eq!(score.measures.len(), 2);
        for m in &score.measures {
            let sum: u8 = m.beats.iter().map(|b| b.duration_ticks).sum();
            assert_eq!(sum, 16);
        }

        assert!(score.measures[0].beats[0].is_dotted());
        assert!(score.measures[0].beats[2].tie_to_next);
        assert!(score.measures[1].beats[0].tie_from_prev);

        let tex = score_to_alphatex(&score, "E4 B3 G3 D3 A2 E2", "Standard");
        assert!(tex.contains("0.1 { d }"));
        assert!(tex.contains("-.1"));
    }
}
