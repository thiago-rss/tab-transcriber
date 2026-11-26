use std::collections::BTreeSet;

use tab_solver::{ChordVoicing, FretPos};

/// Clave musical para renderização correta por família instrumental.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClefKind {
    /// Clave de Sol em altura real (Voz, Violino, Flauta, Sax)
    Treble,
    /// Clave de Sol oitavada abaixo (Guitarra, Violão, Oud)
    Treble8vb,
    /// Clave de Fá (Baixo, Violoncelo, Fagote, Trombone)
    Bass,
    /// Sistema duplo separado: Clave de Sol (mão direita/agudo) + Clave de Fá (mão esquerda/grave)
    GrandStaff,
    /// Clave de Dó na 3ª linha (Viola / registro médio)
    Alto,
}

impl Default for ClefKind {
    fn default() -> Self {
        Self::Treble8vb
    }
}

impl ClefKind {
    pub fn from_str_id(id: &str) -> Option<Self> {
        match id.trim().to_lowercase().as_str() {
            "grand" | "grandstaff" | "piano" | "dual" => Some(Self::GrandStaff),
            "treble" | "g2" | "sol" => Some(Self::Treble),
            "treble8vb" | "guitar" => Some(Self::Treble8vb),
            "bass" | "f4" | "fa" => Some(Self::Bass),
            _ => None, // "auto"
        }
    }

    pub fn alphatex_clef(&self) -> &'static str {
        match self {
            Self::Bass => "F4",
            Self::Alto => "C3",
            _ => "G2",
        }
    }
}

/// Escolhe automaticamente a melhor clave (incluindo Grand Staff) a partir da tessitura real das notas.
pub fn infer_optimal_clef(events: &[TimedEvent], show_tab_staff: bool, fallback: ClefKind) -> ClefKind {
    let mut all_notes: Vec<u8> = events
        .iter()
        .flat_map(|e| e.notes.iter().copied())
        .collect();
    if all_notes.is_empty() {
        return fallback;
    }
    all_notes.sort_unstable();

    let p10 = all_notes[all_notes.len() / 10];
    let med = all_notes[all_notes.len() / 2];
    let p90 = all_notes[(all_notes.len() * 9) / 10];

    if show_tab_staff {
        return if med <= 48 && p90 <= 62 {
            ClefKind::Bass
        } else {
            ClefKind::Treble8vb
        };
    }

    // Se cobre tanto a região grave (< B3) quanto a aguda (>= E4) com amplitude >= 13 semitons -> Grand Staff
    if p10 <= 57 && p90 >= 64 && p90.saturating_sub(p10) >= 13 {
        ClefKind::GrandStaff
    } else if med <= 55 && p90 <= 64 {
        ClefKind::Bass
    } else {
        ClefKind::Treble
    }
}

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeyMode {
    Major,
    Minor,
}

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

#[derive(Debug, Clone)]
pub struct TimedEvent {
    pub start_ms: u64,
    pub duration_ms: u64,
    pub notes: Vec<u8>,
    pub bend_semitones: f32,
}

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
    /// Ponto de divisão Clave de Sol / Clave de Fá no modo GrandStaff (padrão: MIDI 60 = C4).
    pub split_midi: u8,
}

impl QuantizedScore {
    pub fn with_instrument(mut self, midi_program: u8, show_tab_staff: bool, clef: ClefKind) -> Self {
        self.midi_program = midi_program;
        self.show_tab_staff = show_tab_staff;
        self.clef = clef;
        self
    }

    pub fn with_split_midi(mut self, split_midi: u8) -> Self {
        self.split_midi = split_midi.clamp(48, 72);
        self
    }
}

/// Divide as notas de um beat entre Clave de Sol (treble) e Clave de Fá (bass) com histerese
/// suave em torno de `split_midi` para evitar saltos constantes entre pautas em melodias centrais.
fn split_beat_notes(
    notes: &[u8],
    split_midi: u8,
    prev_single_was_treble: &mut bool,
) -> (Vec<u8>, Vec<u8>) {
    if notes.is_empty() {
        return (Vec::new(), Vec::new());
    }

    if notes.len() == 1 {
        let m = notes[0];
        let go_treble = if m >= split_midi + 2 {
            true
        } else if m <= split_midi.saturating_sub(3) {
            false
        } else {
            // Zona de histerese (A3..C#4): mantém a pauta da nota anterior
            *prev_single_was_treble
        };
        *prev_single_was_treble = go_treble;
        return if go_treble {
            (vec![m], Vec::new())
        } else {
            (Vec::new(), vec![m])
        };
    }

    // Em acordes, divide estritamente pelo ponto de corte (C4 = 60 por padrão)
    let mut treble = Vec::new();
    let mut bass = Vec::new();
    for &m in notes {
        if m >= split_midi {
            treble.push(m);
        } else {
            bass.push(m);
        }
    }
    if !treble.is_empty() && bass.is_empty() {
        *prev_single_was_treble = true;
    } else if treble.is_empty() && !bass.is_empty() {
        *prev_single_was_treble = false;
    }
    (treble, bass)
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

pub fn estimate_bpm(events: &[TimedEvent], fallback_bpm: u16) -> u16 {
    if events.len() < 4 {
        return fallback_bpm;
    }

    const STEP_MS: u64 = 10;
    const MAX_WINDOW_MS: u64 = 24_000;
    const MIN_LAG: usize = 35;
    const MAX_LAG: usize = 108;

    let end_ms = events.last().map(|e| e.start_ms).unwrap_or(0);
    let start_cutoff = end_ms.saturating_sub(MAX_WINDOW_MS);

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
    let mut best_lag = 60usize;
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
// QUANTIZAÇÃO COM FASE ÓTIMA DE GRADE, FUSÃO DE JITTER, PONTOS E LIGADURAS
// ============================================================================

const VALID_FIGURES: [u8; 8] = [16, 12, 8, 6, 4, 3, 2, 1];

/// Encontra o deslocamento de fase (em ms) que melhor alinha todos os ataques à grade de semicolcheias.
fn find_best_grid_phase(events: &[TimedEvent], t0: u64, sixteenth_ms: f64) -> f64 {
    if events.len() < 3 || sixteenth_ms <= 10.0 {
        return 0.0;
    }

    let mut best_phase = 0.0f64;
    let mut best_err = f64::INFINITY;

    for step in 0..20 {
        let phase = (step as f64 / 20.0) * sixteenth_ms;
        let mut total_err = 0.0f64;

        for ev in events {
            let rel = (ev.start_ms.saturating_sub(t0) as f64) - phase;
            let snapped = (rel / sixteenth_ms).round() * sixteenth_ms;
            let diff = rel - snapped;
            // Acordes e notas longas pesam mais no alinhamento de fase
            let w = 1.0 + 0.25 * (ev.notes.len().saturating_sub(1) as f64);
            total_err += w * diff * diff;
        }

        if total_err < best_err {
            best_err = total_err;
            best_phase = phase;
        }
    }

    best_phase
}

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
        split_midi: 60,
    }
}

struct MergedGridEvent {
    grid_tick: u32,
    duration_ms: u64,
    notes: Vec<u8>,
    voicing: Option<ChordVoicing>,
    bend_semitones: f32,
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

    // 1. Otimização de fase sub-tick e fusão de colisões simultâneas em acordes limpos
    let t0 = events[0].start_ms;
    let phase_ms = find_best_grid_phase(events, t0, sixteenth_ms);
    let mut merged: Vec<MergedGridEvent> = Vec::with_capacity(events.len());

    for (i, ev) in events.iter().enumerate() {
        let rel_ms = (ev.start_ms.saturating_sub(t0) as f64) - phase_ms;
        let tick = (rel_ms / sixteenth_ms).round().max(0.0) as u32;
        let v = voicings.get(i).cloned().flatten();

        if let Some(last) = merged.last_mut() {
            if tick <= last.grid_tick {
                let mut set: BTreeSet<u8> = last.notes.iter().copied().collect();
                set.extend(ev.notes.iter().copied());
                last.notes = set.into_iter().collect();
                last.duration_ms = last.duration_ms.max(ev.duration_ms);
                last.bend_semitones = last.bend_semitones.max(ev.bend_semitones);
                if last.voicing.is_none() {
                    last.voicing = v;
                }
                continue;
            }
        }

        merged.push(MergedGridEvent {
            grid_tick: tick,
            duration_ms: ev.duration_ms,
            notes: ev.notes.clone(),
            voicing: v,
            bend_semitones: ev.bend_semitones,
        });
    }

    let first_tick = merged.first().map(|m| m.grid_tick).unwrap_or(0);
    for m in &mut merged {
        m.grid_tick = m.grid_tick.saturating_sub(first_tick);
    }

    let mut measures: Vec<QuantizedMeasure> = Vec::new();
    let mut current_beats: Vec<QuantizedBeat> = Vec::new();
    let mut current_ticks: u8 = 0;
    let mut cursor_abs: u32 = 0;

    for (i, ev) in merged.iter().enumerate() {
        let start_tick = ev.grid_tick;

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

        let max_slot = merged
            .get(i + 1)
            .map(|next| next.grid_tick.saturating_sub(start_tick).max(1))
            .unwrap_or(measure_ticks as u32)
            .min(measure_ticks as u32) as u8;

        let raw_ticks = (ev.duration_ms as f64) / sixteenth_ms;
        let note_ticks = quantize_note_ticks(raw_ticks, max_slot);
        let chord = recognize_chord(&ev.notes, key);

        push_tied_note_ticks(
            &mut measures,
            &mut current_beats,
            &mut current_ticks,
            measure_ticks,
            note_ticks,
            &ev.notes,
            &ev.voicing,
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
        split_midi: 60,
    }
}

/// Grade rítmica neural gerada pelo Beat This! (tempos reais de batidas e cabeças de compasso).
#[derive(Debug, Clone, Default)]
pub struct NeuralBeatGrid {
    pub beats_ms: Vec<u64>,
    pub downbeats_ms: Vec<u64>,
}

impl NeuralBeatGrid {
    pub fn is_usable(&self) -> bool {
        self.beats_ms.len() >= 4
    }

    /// Calcula o BPM mediano a partir dos intervalos reais entre batidas consecutivas.
    pub fn median_bpm(&self) -> Option<u16> {
        if self.beats_ms.len() < 2 {
            return None;
        }
        let mut ibis: Vec<u64> = self
            .beats_ms
            .windows(2)
            .map(|w| w[1].saturating_sub(w[0]))
            .filter(|&d| (250..=1500).contains(&d))
            .collect();
        if ibis.is_empty() {
            return None;
        }
        ibis.sort_unstable();
        let med_ms = ibis[ibis.len() / 2] as f64;
        Some(((60_000.0 / med_ms).round() as u16).clamp(45, 220))
    }

    /// Infere o número de batidas por compasso comparando downbeats consecutivos com a grade de batidas.
    pub fn inferred_beats_per_bar(&self) -> Option<u8> {
        if self.beats_ms.len() < 4 || self.downbeats_ms.len() < 2 {
            return None;
        }
        let nearest_idx = |t: u64| -> usize {
            self.beats_ms
                .iter()
                .enumerate()
                .min_by_key(|(_, &b)| b.abs_diff(t))
                .map(|(i, _)| i)
                .unwrap_or(0)
        };
        let mut counts: Vec<usize> = self
            .downbeats_ms
            .windows(2)
            .map(|w| nearest_idx(w[1]).saturating_sub(nearest_idx(w[0])))
            .filter(|&c| (2..=7).contains(&c))
            .collect();
        if counts.is_empty() {
            return None;
        }
        counts.sort_unstable();
        Some(counts[counts.len() / 2] as u8)
    }

    /// Converte um timestamp em milissegundos para posição contínua em batidas (acompanha rubato/variação de tempo).
    fn ms_to_beat_pos(&self, ms: u64) -> f64 {
        let n = self.beats_ms.len();
        if n < 2 {
            return 0.0;
        }
        let first = self.beats_ms[0];
        if ms <= first {
            let ibi = self.beats_ms[1].saturating_sub(first).max(200) as f64;
            return -((first - ms) as f64) / ibi;
        }
        let last = self.beats_ms[n - 1];
        if ms >= last {
            let ibi = last.saturating_sub(self.beats_ms[n - 2]).max(200) as f64;
            return ((n - 1) as f64) + ((ms - last) as f64) / ibi;
        }
        let idx = match self.beats_ms.binary_search(&ms) {
            Ok(i) => return i as f64,
            Err(ins) => ins.saturating_sub(1),
        };
        let b0 = self.beats_ms[idx];
        let b1 = self.beats_ms[idx + 1];
        let span = b1.saturating_sub(b0).max(1) as f64;
        (idx as f64) + ((ms - b0) as f64) / span
    }
}

/// Constrói a partitura quantizada usando a grade neural de batidas e cabeças de compasso do Beat This!.
/// Acompanha flutuações humanas de andamento sem deriva acumulada e alinha cada downbeat ao tempo 1.
pub fn build_score_with_beat_grid(
    title: &str,
    events: &[TimedEvent],
    voicings: &[Option<ChordVoicing>],
    tuning_low_to_high: &[u8],
    bpm_override: Option<u16>,
    time_signature: TimeSignature,
    beat_grid: &NeuralBeatGrid,
) -> QuantizedScore {
    if !beat_grid.is_usable() || events.is_empty() {
        return build_quantized_score(
            title,
            events,
            voicings,
            tuning_low_to_high,
            bpm_override,
            time_signature,
        );
    }

    let bpm = bpm_override
        .or_else(|| beat_grid.median_bpm())
        .unwrap_or_else(|| estimate_bpm(events, 100));
    let key = detect_key(events);
    let measure_ticks = time_signature.measure_ticks();

    // Ticks de semicolcheia por batida detectada (ex.: 4 em 4/4 e 3/4; 6 em 6/8 com pulso de semínima pontuada)
    let beats_per_bar = beat_grid
        .inferred_beats_per_bar()
        .unwrap_or(time_signature.numerator)
        .max(1);
    let ticks_per_beat: f64 = if time_signature.denominator == 4 {
        4.0
    } else if time_signature.numerator == 6 && time_signature.denominator == 8 && beats_per_bar == 2 {
        6.0
    } else {
        (measure_ticks as f64) / (beats_per_bar as f64)
    };

    // Alinha o início do compasso 1 a uma cabeça de compasso (downbeat) real
    let first_ev_beat = beat_grid.ms_to_beat_pos(events[0].start_ms);
    let origin_beat = if !beat_grid.downbeats_ms.is_empty() {
        let db_beats: Vec<f64> = beat_grid
            .downbeats_ms
            .iter()
            .map(|&db| beat_grid.ms_to_beat_pos(db))
            .collect();
        if let Some(&prev_db) = db_beats.iter().rfind(|&&b| b <= first_ev_beat + 0.18) {
            prev_db
        } else {
            let mut b = db_beats[0];
            while b > first_ev_beat + 0.18 {
                b -= beats_per_bar as f64;
            }
            b
        }
    } else {
        first_ev_beat.floor()
    };

    let mut grid_starts: Vec<u32> = Vec::with_capacity(events.len());
    for ev in events {
        let rel_beats = (beat_grid.ms_to_beat_pos(ev.start_ms) - origin_beat).max(0.0);
        let mut tick = (rel_beats * ticks_per_beat).round() as u32;
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

        let b_start = beat_grid.ms_to_beat_pos(ev.start_ms);
        let b_end = beat_grid.ms_to_beat_pos(ev.start_ms.saturating_add(ev.duration_ms));
        let raw_ticks = ((b_end - b_start) * ticks_per_beat).max(1.0);
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
        split_midi: 60,
    }
}

// ============================================================================
// SERIALIZAÇÃO ALPHATEX (Topo Limpo + Separação de Claves Sol/Fá + Tab)
// ============================================================================

fn format_alphatex_pos(pos: FretPos, num_strings: usize, is_tied: bool) -> String {
    let atex_string = num_strings.saturating_sub(pos.string);
    if is_tied {
        format!("-.{}", atex_string)
    } else {
        format!("{}.{}", pos.fret, atex_string)
    }
}

fn format_pitched_alphatex_beat(
    midi_notes: &[u8],
    beat: &QuantizedBeat,
    key: DetectedKey,
    chord_prop: Option<String>,
) -> String {
    if midi_notes.is_empty() {
        return if beat.is_dotted() {
            "r { d }".to_string()
        } else {
            "r".to_string()
        };
    }

    let mut pitched: Vec<String> = midi_notes
        .iter()
        .map(|&m| {
            let name = midi_to_note_name_for_key(m, key);
            if beat.tie_from_prev {
                format!("{} {{ t }}", name)
            } else {
                name
            }
        })
        .collect();

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
        body
    } else {
        format!("{} {{ {} }}", body, beat_props.join(" "))
    }
}

fn ticks_to_alphatex_dur_v2(ticks: u8) -> (&'static str, bool) {
    match ticks {
        16..=u8::MAX => ("1", false),
        12..=15 => ("2", true),
        8..=11 => ("2", false),
        6..=7 => ("4", true),
        4..=5 => ("4", false),
        3 => ("8", true),
        2 => ("8", false),
        _ => ("16", false),
    }
}

fn is_valid_single_dur_v2(ticks: u8) -> bool {
    matches!(ticks, 1 | 2 | 3 | 4 | 6 | 8 | 12 | 16)
}

fn format_pitched_beat_v2(
    notes: &[u8],
    ticks: u8,
    tie_from_prev: bool,
    chord_effect: Option<String>,
    key: DetectedKey,
) -> String {
    let (dur, dotted) = ticks_to_alphatex_dur_v2(ticks);
    let mut effects: Vec<String> = Vec::new();
    if dotted {
        effects.push("d".to_string());
    }
    if let Some(ch) = chord_effect {
        effects.push(ch);
    }
    let eff_suffix = if effects.is_empty() {
        String::new()
    } else {
        format!(" {{ {} }}", effects.join(" "))
    };

    if notes.is_empty() {
        return format!("r.{}{}", dur, eff_suffix);
    }

    let note_tokens: Vec<String> = notes
        .iter()
        .map(|&m| {
            let name = midi_to_note_name_for_key(m, key);
            if tie_from_prev {
                format!("{} {{ t }}", name)
            } else {
                name
            }
        })
        .collect();

    if note_tokens.len() == 1 {
        format!("{}.{}{}", note_tokens[0], dur, eff_suffix)
    } else {
        format!("({}).{}{}", note_tokens.join(" "), dur, eff_suffix)
    }
}

fn format_fretted_beat_v2(
    beat: &QuantizedBeat,
    num_strings: usize,
    key: DetectedKey,
) -> String {
    let (dur, dotted) = ticks_to_alphatex_dur_v2(beat.duration_ticks);
    let mut effects: Vec<String> = Vec::new();
    if dotted {
        effects.push("d".to_string());
    }
    if let Some(ch) = beat.chord.as_ref().map(|c| format!("ch \"{}\"", c.label.as_str().replace('"', "'"))) {
        effects.push(ch);
    }
    let eff_suffix = if effects.is_empty() {
        String::new()
    } else {
        format!(" {{ {} }}", effects.join(" "))
    };

    if beat.is_rest() {
        return format!("r.{}{}", dur, eff_suffix);
    }

    if let Some(ref voicing) = beat.voicing {
        if !voicing.is_empty() {
            let bend_q = ((beat.bend_semitones * 2.0).round() as i32).clamp(0, 8);
            let mut tokens = Vec::with_capacity(voicing.len());
            for (idx, fp) in voicing.iter().enumerate() {
                let at_string = num_strings.saturating_sub(fp.string).max(1);
                let base = if beat.tie_from_prev {
                    format!("-.{}", at_string)
                } else {
                    format!("{}.{}", fp.fret, at_string)
                };
                if !beat.tie_from_prev && idx == 0 && bend_q >= 2 {
                    tokens.push(format!("{} {{ b (0 {} {}) }}", base, bend_q, bend_q));
                } else {
                    tokens.push(base);
                }
            }
            if tokens.len() == 1 {
                return format!("{}.{}{}", tokens[0], dur, eff_suffix);
            }
            return format!("({}).{}{}", tokens.join(" "), dur, eff_suffix);
        }
    }

    format_pitched_beat_v2(
        &beat.midi_notes,
        beat.duration_ticks,
        beat.tie_from_prev,
        beat.chord.as_ref().map(|c| format!("ch \"{}\"", c.label.as_str().replace('"', "'"))),
        key,
    )
}

fn render_pitched_hand_measures(
    score: &QuantizedScore,
    clef_tag: &str,
    keep_note: impl Fn(u8) -> bool,
    include_chords: bool,
) -> String {
    let mut bars: Vec<String> = Vec::with_capacity(score.measures.len());

    for (m_idx, measure) in score.measures.iter().enumerate() {
        let mut items: Vec<(Vec<u8>, u8, bool, Option<String>)> = Vec::new();
        for beat in &measure.beats {
            let filtered: Vec<u8> = beat
                .midi_notes
                .iter()
                .copied()
                .filter(|&m| keep_note(m))
                .collect();
            let ch = if include_chords { beat.chord.as_ref().map(|c| format!("ch \"{}\"", c.label.as_str().replace('"', "'"))) } else { None };

            if filtered.is_empty() {
                if let Some(last) = items.last_mut() {
                    if last.0.is_empty() && last.3.is_none() && ch.is_none() {
                        let sum = last.1.saturating_add(beat.duration_ticks);
                        if is_valid_single_dur_v2(sum) {
                            last.1 = sum;
                            continue;
                        }
                    }
                }
                items.push((Vec::new(), beat.duration_ticks, false, ch));
            } else {
                items.push((filtered, beat.duration_ticks, beat.tie_from_prev, ch));
            }
        }

        let rendered_beats: Vec<String> = items
            .into_iter()
            .map(|(notes, ticks, tie, ch)| format_pitched_beat_v2(&notes, ticks, tie, ch, score.key))
            .collect();

        let body = if rendered_beats.is_empty() {
            "r.1".to_string()
        } else {
            rendered_beats.join(" ")
        };

        if m_idx == 0 {
            bars.push(format!(
                "\\clef {} \\ts {} \\ks {} {}",
                clef_tag,
                score.time_signature.alphatex_ts(),
                score.key.alphatex_ks(),
                body
            ));
        } else {
            bars.push(body);
        }
    }

    bars.join(" |\n")
}

pub fn score_to_alphatex(score: &QuantizedScore, tuning_high_to_low: &str, subtitle: &str) -> String {
    let title = score.title.trim().replace('"', "'");
    let sub = subtitle.trim().replace('"', "'");

    let mut header = String::new();
    if !title.is_empty() {
        header.push_str(&format!("\\title \"{}\"\n", title));
    }
    if !sub.is_empty() {
        header.push_str(&format!("\\subtitle \"{}\"\n", sub));
    }
    header.push_str(&format!("\\tempo {}\n.\n", score.bpm));

    // 1. Pauta Dupla (Grand Staff: Sol na mão direita + Fá na mão esquerda)
    if !score.show_tab_staff && score.clef == ClefKind::GrandStaff {
        let split = score.split_midi;
        let rh_bars = render_pitched_hand_measures(score, "G2", |m| m >= split, true);
        let lh_bars = render_pitched_hand_measures(score, "F4", |m| m < split, false);

        return format!(
            "{}\\track \"Piano\"\n\\staff {{score}} \\tuning piano \\instrument {}\n{}\n\\staff {{score}} \\tuning piano\n{}\n",
            header, score.midi_program, rh_bars, lh_bars
        );
    }

    // 2. Pauta Única sem Tablatura (altura real: Clave de Sol, Fá ou Dó)
    if !score.show_tab_staff {
        let clef_tag = score.clef.alphatex_clef();
        let bars = render_pitched_hand_measures(score, clef_tag, |_| true, true);

        return format!(
            "{}\\track \"Partitura\"\n\\staff {{score}} \\tuning piano \\instrument {}\n{}\n",
            header, score.midi_program, bars
        );
    }

    // 3. Instrumento com Tablatura (Guitarra, Violão, Oud, Baixo) — clave no 1º compasso pós-'.'
    let num_strings = score.tuning_low_to_high.len().max(1);
    let clef_tag = score.clef.alphatex_clef();
    let mut bars: Vec<String> = Vec::with_capacity(score.measures.len());

    for (m_idx, measure) in score.measures.iter().enumerate() {
        let rendered_beats: Vec<String> = measure
            .beats
            .iter()
            .map(|b| format_fretted_beat_v2(b, num_strings, score.key))
            .collect();
        let body = if rendered_beats.is_empty() {
            "r.1".to_string()
        } else {
            rendered_beats.join(" ")
        };
        if m_idx == 0 {
            bars.push(format!(
                "\\clef {} \\ts {} \\ks {} {}",
                clef_tag,
                score.time_signature.alphatex_ts(),
                score.key.alphatex_ks(),
                body
            ));
        } else {
            bars.push(body);
        }
    }

    format!(
        "{}\\track \"Tab\"\n\\staff {{score tabs}} \\tuning {} \\instrument {}\n{}\n",
        header,
        tuning_high_to_low,
        score.midi_program,
        bars.join(" |\n")
    )
}

// ============================================================================
// EXPORTAÇÃO MUSICXML 4.0 (Inclui Grand Staff Sol + Fá)
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

fn write_pitched_notes_xml(
    xml: &mut String,
    notes: &[u8],
    beat: &QuantizedBeat,
    voice: u8,
    staff: u8,
    prefer_flats: bool,
) {
    if notes.is_empty() {
        xml.push_str("      <note>\n");
        xml.push_str("        <rest/>\n");
        xml.push_str(&format!("        <duration>{}</duration>\n", beat.duration_ticks));
        xml.push_str(&format!("        <voice>{}</voice>\n", voice));
        xml.push_str(&format!("        <type>{}</type>\n", beat.musicxml_type()));
        if beat.is_dotted() {
            xml.push_str("        <dot/>\n");
        }
        xml.push_str(&format!("        <staff>{}</staff>\n", staff));
        xml.push_str("      </note>\n");
        return;
    }

    for (n_idx, &midi) in notes.iter().enumerate() {
        xml.push_str("      <note>\n");
        if n_idx > 0 {
            xml.push_str("        <chord/>\n");
        }
        write_pitch_xml(xml, midi, prefer_flats);
        xml.push_str(&format!("        <duration>{}</duration>\n", beat.duration_ticks));
        if beat.tie_from_prev {
            xml.push_str("        <tie type=\"stop\"/>\n");
        }
        if beat.tie_to_next {
            xml.push_str("        <tie type=\"start\"/>\n");
        }
        xml.push_str(&format!("        <voice>{}</voice>\n", voice));
        xml.push_str(&format!("        <type>{}</type>\n", beat.musicxml_type()));
        if beat.is_dotted() {
            xml.push_str("        <dot/>\n");
        }
        xml.push_str(&format!("        <staff>{}</staff>\n", staff));
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

pub fn export_musicxml(score: &QuantizedScore) -> String {
    let num_strings = score.tuning_low_to_high.len();
    let measure_ticks = score.time_signature.measure_ticks();
    let prefer_flats = score.key.prefer_flats();
    let is_grand_staff = !score.show_tab_staff && score.clef == ClefKind::GrandStaff;
    let mode_xml = match score.key.mode {
        KeyMode::Major => "major",
        KeyMode::Minor => "minor",
    };

    let mut xml = String::with_capacity(4096);

    xml.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
    xml.push_str("<!DOCTYPE score-partwise PUBLIC \"-//Recordare//DTD MusicXML 4.0 Partwise//EN\" \"http://www.musicxml.org/dtds/partwise.dtd\">\n");
    xml.push_str("<score-partwise version=\"4.0\">\n");
    if !score.title.trim().is_empty() {
        xml.push_str("  <work>\n");
        xml.push_str(&format!(
            "    <work-title>{}</work-title>\n",
            escape_xml(&score.title)
        ));
        xml.push_str("  </work>\n");
    }
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
    let mut prev_single_treble = true;

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

            if is_grand_staff {
                xml.push_str("        <staves>2</staves>\n");
                xml.push_str("        <clef number=\"1\">\n");
                xml.push_str("          <sign>G</sign>\n");
                xml.push_str("          <line>2</line>\n");
                xml.push_str("        </clef>\n");
                xml.push_str("        <clef number=\"2\">\n");
                xml.push_str("          <sign>F</sign>\n");
                xml.push_str("          <line>4</line>\n");
                xml.push_str("        </clef>\n");
            } else {
                let (clef_sign, clef_line, oct_change) = match score.clef {
                    ClefKind::Treble | ClefKind::GrandStaff => ("G", 2, None),
                    ClefKind::Treble8vb => ("G", 2, Some(-1)),
                    ClefKind::Bass => ("F", 4, if score.show_tab_staff { Some(-1) } else { None }),
                    ClefKind::Alto => ("C", 3, None),
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

        if is_grand_staff {
            let mut split_cache: Vec<(Vec<u8>, Vec<u8>)> = Vec::with_capacity(measure.beats.len());
            for beat in &measure.beats {
                split_cache.push(split_beat_notes(
                    &beat.midi_notes,
                    score.split_midi,
                    &mut prev_single_treble,
                ));
            }

            // Staff 1: Clave de Sol
            for (beat, (tr_notes, _)) in measure.beats.iter().zip(split_cache.iter()) {
                if !beat.tie_from_prev {
                    if let Some(ch) = &beat.chord {
                        if last_harmony.as_deref() != Some(ch.label.as_str()) {
                            write_harmony_xml(&mut xml, ch, prefer_flats);
                            last_harmony = Some(ch.label.clone());
                        }
                    }
                }
                write_pitched_notes_xml(&mut xml, tr_notes, beat, 1, 1, prefer_flats);
            }

            xml.push_str(&format!("      <backup><duration>{}</duration></backup>\n", measure_ticks));

            // Staff 2: Clave de Fá
            for (beat, (_, bs_notes)) in measure.beats.iter().zip(split_cache.iter()) {
                write_pitched_notes_xml(&mut xml, bs_notes, beat, 2, 2, prefer_flats);
            }

            xml.push_str("    </measure>\n");
            continue;
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
            write_pitched_notes_xml(&mut xml, &beat.midi_notes, beat, 1, 1, prefer_flats);
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

        let eighths_90: Vec<TimedEvent> = (0..12)
            .map(|k| TimedEvent {
                start_ms: (k as u64) * 333,
                duration_ms: 280,
                notes: if k % 2 == 0 { vec![45, 52] } else { vec![57] },
                bend_semitones: 0.0,
            })
            .collect();
        let bpm_90 = estimate_bpm(&eighths_90, 120);
        assert!(bpm_90.abs_diff(90) <= 3, "obteve {}", bpm_90);
    }

    #[test]
    fn grand_staff_splits_treble_and_bass_cleanly_without_header_text() {
        let tuning = [40, 45, 50, 55, 59, 64];
        // Piano: C3 (48) na mão esquerda + E4 (64) e G4 (67) na mão direita
        let events = vec![
            TimedEvent { start_ms: 0, duration_ms: 1000, notes: vec![48, 64, 67], bend_semitones: 0.0 },
            TimedEvent { start_ms: 1000, duration_ms: 1000, notes: vec![43, 62, 71], bend_semitones: 0.0 },
        ];
        let voicings = vec![None, None];
        let score = build_quantized_score("", &events, &voicings, &tuning, Some(120), TimeSignature::new(4, 4))
            .with_instrument(0, false, ClefKind::GrandStaff);

        let tex = score_to_alphatex(&score, "E4 B3 G3 D3 A2 E2", "");
        assert!(!tex.contains("\\title"), "não deve inserir \\title quando vazio");
        assert!(!tex.contains("\\subtitle"), "não deve inserir \\subtitle quando vazio");
        assert!(tex.contains("\\clef G2"));
        assert!(tex.contains("\\clef F4"));

        let xml = export_musicxml(&score);
        assert!(xml.contains("<staves>2</staves>"));
        assert!(xml.contains("<sign>G</sign>"));
        assert!(xml.contains("<sign>F</sign>"));
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

        let score = build_quantized_score("", &events, &voicings, &tuning, Some(120), TimeSignature::new(4, 4));
        assert_eq!(score.measures.len(), 2);
        assert!(score.measures[0].beats[0].is_dotted());
        assert!(score.measures[0].beats[2].tie_to_next);
        assert!(score.measures[1].beats[0].tie_from_prev);
    }

    #[test]
    fn neural_beat_grid_tracks_tempo_drift_and_aligns_downbeats() {
        let tuning = [40, 45, 50, 55, 59, 64];
        // Andamento acelera gradualmente (600 ms/batida -> 500 ms/batida), 2 compassos de 4/4
        let beats_ms = vec![0, 600, 1180, 1740, 2280, 2800, 3300, 3800, 4300];
        let grid = NeuralBeatGrid {
            beats_ms: beats_ms.clone(),
            downbeats_ms: vec![0, 2280, 4300],
        };

        // 8 semínimas tocadas exatamente em cada batida real apesar da aceleração
        let events: Vec<TimedEvent> = (0..8)
            .map(|i| TimedEvent {
                start_ms: beats_ms[i],
                duration_ms: beats_ms[i + 1] - beats_ms[i],
                notes: vec![64],
                bend_semitones: 0.0,
            })
            .collect();
        let voicings = vec![Some(vec![FretPos { string: 5, fret: 0 }]); 8];

        let score = build_score_with_beat_grid(
            "Rubato",
            &events,
            &voicings,
            &tuning,
            None,
            TimeSignature::new(4, 4),
            &grid,
        );

        assert_eq!(score.measures.len(), 2);
        assert_eq!(score.measures[0].beats.len(), 4);
        assert_eq!(score.measures[1].beats.len(), 4);
        assert!(score.measures[0].beats.iter().all(|b| b.duration_ticks == 4));
        assert!(score.measures[1].beats.iter().all(|b| b.duration_ticks == 4));
        assert_eq!(grid.inferred_beats_per_bar(), Some(4));
    }

    #[test]
    fn alphatex_places_bass_clef_in_bar_scope_and_renders_grand_staff() {
        let tuning = [28, 33, 38, 43];
        let bass_events = vec![
            TimedEvent { start_ms: 0, duration_ms: 500, notes: vec![36], bend_semitones: 0.0 },
            TimedEvent { start_ms: 500, duration_ms: 500, notes: vec![43], bend_semitones: 0.0 },
        ];
        let mut score_bass = build_quantized_score(
            "Bass",
            &bass_events,
            &[None, None],
            &tuning,
            Some(120),
            TimeSignature::new(4, 4),
        );
        score_bass.midi_program = 33;
        score_bass.show_tab_staff = false;
        score_bass.clef = ClefKind::Bass;

        let tex_bass = score_to_alphatex(&score_bass, "G2 D2 A1 E1", "Baixo");
        let dot_pos = tex_bass.find("\n.\n").expect("separador '.' ausente");
        let clef_pos = tex_bass.find(r"\clef F4").expect("clave F4 ausente");
        assert!(clef_pos > dot_pos, r"\clef F4 deve vir depois de '.' (escopo de compasso)");
        assert!(tex_bass.contains(r"\tuning piano"));

        let piano_events = vec![
            TimedEvent { start_ms: 0, duration_ms: 500, notes: vec![48, 64, 67], bend_semitones: 0.0 },
            TimedEvent { start_ms: 500, duration_ms: 500, notes: vec![43, 62, 71], bend_semitones: 0.0 },
        ];
        assert_eq!(infer_optimal_clef(&piano_events, false, ClefKind::Treble), ClefKind::GrandStaff);

        let mut score_piano = build_quantized_score(
            "Piano",
            &piano_events,
            &[None, None],
            &tuning,
            Some(120),
            TimeSignature::new(4, 4),
        );
        score_piano.midi_program = 0;
        score_piano.show_tab_staff = false;
        score_piano.clef = ClefKind::GrandStaff;

        let tex_piano = score_to_alphatex(&score_piano, "E4 B3 G3 D3 A2 E2", "Piano");
        assert!(tex_piano.contains(r"\clef G2"));
        assert!(tex_piano.contains(r"\clef F4"));
        assert_eq!(tex_piano.matches(r"\staff {score}").count(), 2);
    }
}
