#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FretPos {
    pub string: usize,
    pub fret: u8,
}

pub type ChordVoicing = Vec<FretPos>;

/// Região preferencial do braço para guiar o posicionamento ergonômico.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NeckRegion {
    /// Infere automaticamente a região natural a partir do registro da frase e do timbre
    Auto,
    /// Posição aberta (casas 0 a 5)
    Open,
    /// Região média (casas 5 a 10)
    Mid,
    /// Região aguda (casas 9 a 15)
    High,
    /// Região de solo extremo (casas 12 a 24)
    Solo,
}

impl NeckRegion {
    pub fn from_str_id(id: &str) -> Self {
        match id.to_lowercase().as_str() {
            "open" | "aberta" => Self::Open,
            "mid" | "media" => Self::Mid,
            "high" | "aguda" => Self::High,
            "solo" => Self::Solo,
            _ => Self::Auto,
        }
    }

    pub fn label(&self) -> &'static str {
        match self {
            Self::Auto => "Auto (Registro + Timbre)",
            Self::Open => "Aberta (0–5)",
            Self::Mid => "Média (5–10)",
            Self::High => "Aguda (9–15)",
            Self::Solo => "Solo (12–24)",
        }
    }

    /// Faixa [min, max] e centro preferencial quando a região é fixa.
    fn fixed_bounds(&self) -> Option<(u8, u8, f32)> {
        match self {
            Self::Auto => None,
            Self::Open => Some((0, 5, 2.0)),
            Self::Mid => Some((5, 10, 7.5)),
            Self::High => Some((9, 15, 12.0)),
            Self::Solo => Some((12, 24, 16.0)),
        }
    }
}

#[derive(Debug, Clone)]
pub struct CostWeights {
    pub fret_distance_weight: f32,
    pub string_skip_weight: f32,
    pub region_deviation_weight: f32,
    pub open_string_bonus: f32,
    pub closed_open_penalty: f32,
    pub stretch_threshold: u8,
    pub stretch_penalty: f32,
}

impl Default for CostWeights {
    fn default() -> Self {
        Self {
            fret_distance_weight: 1.5,
            string_skip_weight: 0.75,
            region_deviation_weight: 0.38,
            open_string_bonus: 0.25,
            closed_open_penalty: 3.4,
            stretch_threshold: 4,
            stretch_penalty: 3.0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SolverConfig {
    pub weights: CostWeights,
    pub region: NeckRegion,
    pub prefer_closed_positions: bool,
}

impl Default for SolverConfig {
    fn default() -> Self {
        Self {
            weights: CostWeights::default(),
            region: NeckRegion::Auto,
            prefer_closed_positions: false,
        }
    }
}

pub fn generate_candidates(midi_note: u8, tuning: &[u8], max_fret: u8) -> Vec<FretPos> {
    let mut candidates = Vec::new();
    for (string_idx, &open_midi) in tuning.iter().enumerate() {
        if midi_note >= open_midi {
            let fret = midi_note - open_midi;
            if fret <= max_fret {
                candidates.push(FretPos {
                    string: string_idx,
                    fret,
                });
            }
        }
    }
    candidates
}

/// Calcula o centro preferencial do braço (em casas) para a frase inteira.
fn resolve_region_target(
    midi_samples: &[u8],
    mean_warmth: f32,
    tuning: &[u8],
    max_fret: u8,
    cfg: &SolverConfig,
) -> (u8, u8, f32, bool) {
    if let Some((min_f, max_f, center)) = cfg.region.fixed_bounds() {
        let avoid_open = cfg.prefer_closed_positions || min_f >= 5;
        return (min_f, max_f.min(max_fret), center, avoid_open);
    }

    // Modo Auto: estima a região natural com base no registro da frase + calor tímbrico
    let mut valid: Vec<u8> = midi_samples
        .iter()
        .copied()
        .filter(|&m| tuning.first().map_or(false, |&low| m >= low))
        .collect();

    if valid.is_empty() || tuning.is_empty() {
        return (0, max_fret, 3.0, cfg.prefer_closed_positions);
    }

    valid.sort_unstable();
    let median_midi = valid[valid.len() / 2] as f32;
    let p75_midi = valid[(valid.len() * 3) / 4] as f32;
    let ref_pitch = 0.65 * median_midi + 0.35 * p75_midi;

    // Referência neutra: média entre a corda central e a mais aguda da afinação
    let mid_open = tuning[tuning.len() / 2] as f32;
    let top_open = *tuning.last().unwrap() as f32;
    let upper_mid_open = 0.5 * (mid_open + top_open);

    // Se o timbre é encorpado/quente (warmth > 0.5), desloca a preferência para casas mais altas
    let timbre_shift = (mean_warmth.clamp(0.0, 1.0) - 0.45) * 6.0;
    let est_center = (ref_pitch - upper_mid_open + timbre_shift).clamp(2.0, (max_fret as f32) - 3.0);

    let min_f = (est_center - 4.0).max(0.0) as u8;
    let max_f = (est_center + 4.5).min(max_fret as f32) as u8;
    let avoid_open = cfg.prefer_closed_positions || est_center >= 5.5;

    (min_f, max_f, est_center, avoid_open)
}

fn emission_cost_with_region(
    pos: FretPos,
    min_region: u8,
    max_region: u8,
    region_center: f32,
    avoid_open: bool,
    note_warmth: f32,
    num_strings: usize,
    w: &CostWeights,
) -> f32 {
    if pos.fret == 0 {
        if avoid_open {
            // Penaliza corda solta quando a mão está em região média/aguda ou em modo fechado
            return w.closed_open_penalty + (region_center * 0.18);
        }
        // Mesmo em posição aberta, timbre muito quente reduz o bônus da corda solta
        return w.open_string_bonus + note_warmth * 0.45;
    }

    let fret_f = pos.fret as f32;
    let dist_center = (fret_f - region_center).abs();
    let mut cost = dist_center * w.region_deviation_weight;

    // Penalidade forte quando sai da janela da região escolhida
    if pos.fret < min_region {
        cost += ((min_region - pos.fret) as f32) * 1.15;
    } else if pos.fret > max_region {
        cost += ((pos.fret - max_region) as f32) * 0.95;
    }

    // Ajuste tímbrico sutil: timbre mais quente prefere cordas ligeiramente mais grossas / casas mais altas
    let string_norm = if num_strings > 1 {
        (pos.string as f32) / ((num_strings - 1) as f32)
    } else {
        0.5
    };
    let warmth_bias = (note_warmth - 0.5) * 1.4;
    if warmth_bias > 0.0 {
        // Quente -> leve penalidade para corda mais fina em casa muito baixa
        cost += warmth_bias * string_norm * (1.0 - (fret_f / 18.0).clamp(0.0, 1.0));
    } else {
        // Brilhante -> leve preferência para cordas mais finas
        cost += (-warmth_bias) * (1.0 - string_norm) * 0.5;
    }

    cost
}

// ============================================================================
// VITERBI MONOFÔNICO COM ESTADO DE ÂNCORA DE MÃO (hand_anchor_fret)
// ============================================================================

#[derive(Debug, Clone, Copy)]
struct AnchoredState {
    pos: FretPos,
    anchor_fret: u8,
}

fn anchored_transition_cost(prev: AnchoredState, curr: AnchoredState, w: &CostWeights) -> f32 {
    let string_diff = prev.pos.string.abs_diff(curr.pos.string) as f32;
    let string_cost = string_diff * w.string_skip_weight;

    if curr.pos.fret == 0 {
        // Ao tocar corda solta, mantém a âncora da mão onde ela estava
        let anchor_drift = prev.anchor_fret.abs_diff(curr.anchor_fret) as f32;
        return string_cost + anchor_drift * 2.5 + 0.15;
    }

    // Distância medida a partir da âncora real da mão (mesmo se a nota anterior foi corda solta!)
    let fret_diff = prev.anchor_fret.abs_diff(curr.pos.fret);
    let mut fret_cost = (fret_diff as f32) * w.fret_distance_weight;

    if fret_diff > w.stretch_threshold {
        fret_cost += w.stretch_penalty * ((fret_diff - w.stretch_threshold) as f32);
    }

    fret_cost + string_cost
}

/// Solver monofônico com âncora de mão, região do braço e pista tímbrica opcional.
pub fn solve_tablature_with_config(
    notes: &[u8],
    timbral_warmths: Option<&[f32]>,
    tuning: &[u8],
    max_fret: u8,
    cfg: &SolverConfig,
) -> Vec<Option<FretPos>> {
    let n = notes.len();
    if n == 0 {
        return vec![];
    }

    let mean_warmth = match timbral_warmths {
        Some(ws) if !ws.is_empty() => ws.iter().sum::<f32>() / (ws.len() as f32),
        _ => 0.5,
    };

    let (min_reg, max_reg, reg_center, avoid_open) =
        resolve_region_target(notes, mean_warmth, tuning, max_fret, cfg);
    let default_anchor = reg_center.round().clamp(1.0, max_fret as f32) as u8;

    let mut layers: Vec<Vec<AnchoredState>> = Vec::with_capacity(n);
    let mut prev_anchors: Vec<u8> = vec![default_anchor];

    for &note in notes {
        let raw_cands = generate_candidates(note, tuning, max_fret);
        let mut states = Vec::with_capacity(raw_cands.len() * 2);

        for pos in raw_cands {
            if pos.fret > 0 {
                states.push(AnchoredState {
                    pos,
                    anchor_fret: pos.fret,
                });
            } else {
                for &anc in &prev_anchors {
                    states.push(AnchoredState {
                        pos,
                        anchor_fret: anc,
                    });
                }
            }
        }

        let mut next_anchors: Vec<u8> = states.iter().map(|s| s.anchor_fret).collect();
        next_anchors.sort_unstable();
        next_anchors.dedup();
        if !next_anchors.is_empty() {
            prev_anchors = next_anchors;
        }
        layers.push(states);
    }

    let mut dp: Vec<Vec<f32>> = Vec::with_capacity(n);
    let mut backpointers: Vec<Vec<Option<usize>>> = Vec::with_capacity(n);

    let w0 = timbral_warmths.and_then(|s| s.first().copied()).unwrap_or(0.5);
    dp.push(
        layers[0]
            .iter()
            .map(|&st| {
                emission_cost_with_region(
                    st.pos,
                    min_reg,
                    max_reg,
                    reg_center,
                    avoid_open,
                    w0,
                    tuning.len(),
                    &cfg.weights,
                )
            })
            .collect(),
    );
    backpointers.push(vec![None; layers[0].len()]);

    for t in 1..n {
        let curr_layer = &layers[t];
        let prev_layer = &layers[t - 1];
        let wt = timbral_warmths.and_then(|s| s.get(t).copied()).unwrap_or(0.5);

        let mut curr_dp = vec![f32::INFINITY; curr_layer.len()];
        let mut curr_bp = vec![None; curr_layer.len()];

        for (j, &curr_st) in curr_layer.iter().enumerate() {
            let emit = emission_cost_with_region(
                curr_st.pos,
                min_reg,
                max_reg,
                reg_center,
                avoid_open,
                wt,
                tuning.len(),
                &cfg.weights,
            );

            if prev_layer.is_empty() {
                curr_dp[j] = emit;
                continue;
            }

            let mut best_cost = f32::INFINITY;
            let mut best_prev = None;
            for (i, &prev_st) in prev_layer.iter().enumerate() {
                let total =
                    dp[t - 1][i] + anchored_transition_cost(prev_st, curr_st, &cfg.weights) + emit;
                if total < best_cost {
                    best_cost = total;
                    best_prev = Some(i);
                }
            }
            curr_dp[j] = best_cost;
            curr_bp[j] = best_prev;
        }

        dp.push(curr_dp);
        backpointers.push(curr_bp);
    }

    let mut result = vec![None; n];
    let mut current_idx = dp[n - 1]
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(idx, _)| idx);

    for t in (0..n).rev() {
        if let Some(idx) = current_idx {
            if let Some(&st) = layers[t].get(idx) {
                result[t] = Some(st.pos);
                current_idx = backpointers[t][idx];
            }
        } else if t > 0 && !dp[t - 1].is_empty() {
            current_idx = dp[t - 1]
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| a.total_cmp(b))
                .map(|(idx, _)| idx);
        }
    }

    result
}

/// Wrapper retrocompatível (usa `NeckRegion::Auto`).
pub fn solve_tablature(
    notes: &[u8],
    tuning: &[u8],
    max_fret: u8,
    weights: &CostWeights,
) -> Vec<Option<FretPos>> {
    let cfg = SolverConfig {
        weights: weights.clone(),
        ..SolverConfig::default()
    };
    solve_tablature_with_config(notes, None, tuning, max_fret, &cfg)
}

// ============================================================================
// EXTENSÃO POLIFÔNICA (ACORDES COM REGIÃO DO BRAÇO E TIMBRE)
// ============================================================================

fn voicing_centroid(voicing: &[FretPos]) -> Option<f32> {
    let mut sum = 0.0f32;
    let mut count = 0usize;
    for p in voicing {
        if p.fret > 0 {
            sum += p.fret as f32;
            count += 1;
        }
    }
    if count == 0 {
        None
    } else {
        Some(sum / count as f32)
    }
}

fn voicing_emission_cost(
    voicing: &[FretPos],
    min_reg: u8,
    max_reg: u8,
    reg_center: f32,
    avoid_open: bool,
    warmth: f32,
    num_strings: usize,
    w: &CostWeights,
) -> f32 {
    let mut min_f = u8::MAX;
    let mut max_f = 0u8;
    let mut base = 0.0f32;

    for &p in voicing {
        base += emission_cost_with_region(
            p,
            min_reg,
            max_reg,
            reg_center,
            avoid_open,
            warmth,
            num_strings,
            w,
        );
        if p.fret > 0 {
            min_f = min_f.min(p.fret);
            max_f = max_f.max(p.fret);
        }
    }

    let span = if min_f <= max_f { max_f - min_f } else { 0 };
    base + (span as f32) * 0.9
}

fn voicing_transition_cost(
    prev: &[FretPos],
    curr: &[FretPos],
    reg_center: f32,
    avoid_open: bool,
    w: &CostWeights,
) -> f32 {
    match (voicing_centroid(prev), voicing_centroid(curr)) {
        (Some(c1), Some(c2)) => {
            let diff = (c1 - c2).abs();
            let mut cost = diff * w.fret_distance_weight;
            if diff > w.stretch_threshold as f32 {
                cost += (diff - w.stretch_threshold as f32) * w.stretch_penalty;
            }
            cost
        }
        _ => {
            if avoid_open {
                1.8 + reg_center * 0.15
            } else {
                0.35
            }
        }
    }
}

fn dfs_voicings(
    idx: usize,
    per_note_candidates: &[Vec<FretPos>],
    used_strings: u32,
    min_pressed: u8,
    max_pressed: u8,
    max_span: u8,
    current: &mut Vec<FretPos>,
    out: &mut Vec<ChordVoicing>,
) {
    if out.len() >= 192 {
        return;
    }
    if idx == per_note_candidates.len() {
        out.push(current.clone());
        return;
    }

    for &cand in &per_note_candidates[idx] {
        let bit = 1u32 << cand.string;
        if (used_strings & bit) != 0 {
            continue;
        }

        let (new_min, new_max) = if cand.fret == 0 {
            (min_pressed, max_pressed)
        } else {
            (min_pressed.min(cand.fret), max_pressed.max(cand.fret))
        };

        if new_min <= new_max && (new_max - new_min) > max_span {
            continue;
        }

        current.push(cand);
        dfs_voicings(
            idx + 1,
            per_note_candidates,
            used_strings | bit,
            new_min,
            new_max,
            max_span,
            current,
            out,
        );
        current.pop();
    }
}

pub fn generate_chord_candidates_in_region(
    chord_notes: &[u8],
    tuning: &[u8],
    max_fret: u8,
    min_reg: u8,
    max_reg: u8,
    reg_center: f32,
    avoid_open: bool,
    warmth: f32,
    w: &CostWeights,
) -> Vec<ChordVoicing> {
    if chord_notes.is_empty() {
        return vec![];
    }

    let mut playable: Vec<u8> = chord_notes
        .iter()
        .copied()
        .filter(|&n| !generate_candidates(n, tuning, max_fret).is_empty())
        .collect();
    playable.sort_unstable();
    playable.dedup();
    if playable.len() > tuning.len() {
        // Quando em região aguda/solo, prioriza as notas mais agudas do acorde
        if reg_center >= 8.0 {
            let start = playable.len() - tuning.len();
            playable = playable[start..].to_vec();
        } else {
            playable.truncate(tuning.len());
        }
    }
    if playable.is_empty() {
        return vec![];
    }

    let per_note: Vec<Vec<FretPos>> = playable
        .iter()
        .map(|&n| {
            let mut cands = generate_candidates(n, tuning, max_fret);
            // Ordena candidatos próximos da região alvo primeiro para o DFS não saturar em casas 0-4
            cands.sort_by(|a, b| {
                let da = ((a.fret as f32) - reg_center).abs();
                let db = ((b.fret as f32) - reg_center).abs();
                da.total_cmp(&db)
            });
            cands
        })
        .collect();

    let mut voicings = Vec::new();
    let mut current = Vec::with_capacity(playable.len());

    dfs_voicings(
        0,
        &per_note,
        0,
        u8::MAX,
        0,
        w.stretch_threshold,
        &mut current,
        &mut voicings,
    );
    if voicings.is_empty() {
        dfs_voicings(
            0,
            &per_note,
            0,
            u8::MAX,
            0,
            w.stretch_threshold + 1,
            &mut current,
            &mut voicings,
        );
    }

    voicings.sort_by(|a, b| {
        voicing_emission_cost(
            a,
            min_reg,
            max_reg,
            reg_center,
            avoid_open,
            warmth,
            tuning.len(),
            w,
        )
        .total_cmp(&voicing_emission_cost(
            b,
            min_reg,
            max_reg,
            reg_center,
            avoid_open,
            warmth,
            tuning.len(),
            w,
        ))
    });
    voicings.truncate(32);
    voicings
}

/// Solver polifônico consciente de região do braço, posição fechada e calor tímbrico.
pub fn solve_polyphonic_with_config(
    chords: &[Vec<u8>],
    timbral_warmths: Option<&[f32]>,
    tuning: &[u8],
    max_fret: u8,
    cfg: &SolverConfig,
) -> Vec<Option<ChordVoicing>> {
    let n = chords.len();
    if n == 0 {
        return vec![];
    }

    // Se todos os eventos forem notas únicas, delega para o solver monofônico com âncora de mão!
    if chords.iter().all(|c| c.len() == 1) {
        let single_notes: Vec<u8> = chords.iter().map(|c| c[0]).collect();
        return solve_tablature_with_config(&single_notes, timbral_warmths, tuning, max_fret, cfg)
            .into_iter()
            .map(|opt| opt.map(|p| vec![p]))
            .collect();
    }

    let flat_midi: Vec<u8> = chords.iter().flat_map(|c| c.iter().copied()).collect();
    let mean_warmth = match timbral_warmths {
        Some(ws) if !ws.is_empty() => ws.iter().sum::<f32>() / (ws.len() as f32),
        _ => 0.5,
    };
    let (min_reg, max_reg, reg_center, avoid_open) =
        resolve_region_target(&flat_midi, mean_warmth, tuning, max_fret, cfg);

    let layers: Vec<Vec<ChordVoicing>> = chords
        .iter()
        .enumerate()
        .map(|(t, c)| {
            let wt = timbral_warmths.and_then(|s| s.get(t).copied()).unwrap_or(0.5);
            generate_chord_candidates_in_region(
                c,
                tuning,
                max_fret,
                min_reg,
                max_reg,
                reg_center,
                avoid_open,
                wt,
                &cfg.weights,
            )
        })
        .collect();

    let mut dp: Vec<Vec<f32>> = Vec::with_capacity(n);
    let mut bp: Vec<Vec<Option<usize>>> = Vec::with_capacity(n);

    let w0 = timbral_warmths.and_then(|s| s.first().copied()).unwrap_or(0.5);
    dp.push(
        layers[0]
            .iter()
            .map(|v| {
                voicing_emission_cost(
                    v,
                    min_reg,
                    max_reg,
                    reg_center,
                    avoid_open,
                    w0,
                    tuning.len(),
                    &cfg.weights,
                )
            })
            .collect(),
    );
    bp.push(vec![None; layers[0].len()]);

    for t in 1..n {
        let curr_layer = &layers[t];
        let prev_layer = &layers[t - 1];
        let wt = timbral_warmths.and_then(|s| s.get(t).copied()).unwrap_or(0.5);

        let mut curr_dp = vec![f32::INFINITY; curr_layer.len()];
        let mut curr_bp = vec![None; curr_layer.len()];

        for (j, curr_v) in curr_layer.iter().enumerate() {
            let emit = voicing_emission_cost(
                curr_v,
                min_reg,
                max_reg,
                reg_center,
                avoid_open,
                wt,
                tuning.len(),
                &cfg.weights,
            );
            if prev_layer.is_empty() {
                curr_dp[j] = emit;
                continue;
            }
            let mut best_cost = f32::INFINITY;
            let mut best_prev = None;
            for (i, prev_v) in prev_layer.iter().enumerate() {
                let total = dp[t - 1][i]
                    + voicing_transition_cost(prev_v, curr_v, reg_center, avoid_open, &cfg.weights)
                    + emit;
                if total < best_cost {
                    best_cost = total;
                    best_prev = Some(i);
                }
            }
            curr_dp[j] = best_cost;
            curr_bp[j] = best_prev;
        }

        dp.push(curr_dp);
        bp.push(curr_bp);
    }

    let mut result = vec![None; n];
    let mut current_idx = dp[n - 1]
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(idx, _)| idx);

    for t in (0..n).rev() {
        if let Some(idx) = current_idx {
            if let Some(v) = layers[t].get(idx) {
                result[t] = Some(v.clone());
                current_idx = bp[t][idx];
            }
        } else if t > 0 && !dp[t - 1].is_empty() {
            current_idx = dp[t - 1]
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| a.total_cmp(b))
                .map(|(idx, _)| idx);
        }
    }

    result
}

/// Wrapper retrocompatível (usa `NeckRegion::Auto`).
pub fn solve_polyphonic_tablature(
    chords: &[Vec<u8>],
    tuning: &[u8],
    max_fret: u8,
    weights: &CostWeights,
) -> Vec<Option<ChordVoicing>> {
    let cfg = SolverConfig {
        weights: weights.clone(),
        ..SolverConfig::default()
    };
    solve_polyphonic_with_config(chords, None, tuning, max_fret, &cfg)
}

pub fn render_ascii_tab(positions: &[Option<FretPos>], string_labels: &[&str]) -> String {
    let mut out = String::new();
    for s in (0..string_labels.len()).rev() {
        out.push_str(&format!("{:<2}|", string_labels[s]));
        for pos in positions {
            match pos {
                Some(p) if p.string == s => out.push_str(&format!("-{:->2}-", p.fret)),
                _ => out.push_str("----"),
            }
        }
        out.push_str("|\n");
    }
    out
}

pub fn render_ascii_chord_tab(voicings: &[Option<ChordVoicing>], string_labels: &[&str]) -> String {
    let mut out = String::new();
    for s in (0..string_labels.len()).rev() {
        out.push_str(&format!("{:<2}|", string_labels[s]));
        for voicing in voicings {
            let fret_opt = voicing
                .as_ref()
                .and_then(|v| v.iter().find(|p| p.string == s).map(|p| p.fret));
            match fret_opt {
                Some(f) => out.push_str(&format!("-{:->2}-", f)),
                None => out.push_str("----"),
            }
        }
        out.push_str("|\n");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn upper_neck_solo_stays_in_high_box_without_collapsing_to_open_position() {
        let standard = [40, 45, 50, 55, 59, 64];
        // Frase aguda em Mi menor na região da casa 12: E5(76), D5(74), B4(71), A4(69), G4(67), E4(64), G4(67), B4(71)
        let solo = [76, 74, 71, 69, 67, 64, 67, 71];
        let warmths = [0.75; 8]; // Timbre encorpado de guitarra solo
        let cfg = SolverConfig::default(); // Auto

        let tab = solve_tablature_with_config(&solo, Some(&warmths), &standard, 24, &cfg);
        for (i, pos) in tab.iter().enumerate() {
            let p = pos.expect("nota tocável");
            assert!(
                p.fret >= 9,
                "nota índice {} (MIDI {}) caiu para casa {} em vez de permanecer na região aguda",
                i,
                solo[i],
                p.fret
            );
        }
    }

    #[test]
    fn high_neck_region_places_triad_around_12th_fret() {
        let standard = [40, 45, 50, 55, 59, 64];
        // Tríade de Em aguda: E4(64), G4(67), B4(71)
        let chords = vec![vec![64, 67, 71]];
        let cfg = SolverConfig {
            region: NeckRegion::High,
            prefer_closed_positions: true,
            ..SolverConfig::default()
        };

        let res = solve_polyphonic_with_config(&chords, None, &standard, 24, &cfg);
        let voicing = res[0].as_ref().expect("deveria encontrar voicing na região aguda");
        assert!(
            voicing.iter().all(|p| p.fret >= 9),
            "voicing esperado na região aguda, obteve {:?}",
            voicing
        );
    }
}
