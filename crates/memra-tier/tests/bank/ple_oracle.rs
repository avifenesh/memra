// Preserve the pinned native oracle operation/index order, not a lint rewrite.
#![allow(clippy::needless_range_loop)]
// Test-only verbatim native host ID oracle from qwen4exp_gpu.rs at c5a33b14.
// Only visibility and lint attributes changed; no runtime dependency or new model path.
#[allow(clippy::too_many_arguments)]
pub(super) fn shift_right_ignore_eos(history: &[i64], shift: usize, eos: i64) -> Vec<i64> {
    if shift == 0 {
        return history.to_vec();
    }
    let mut last_eos_inclusive: i64 = -1;
    let mut output = Vec::with_capacity(history.len());
    for (position, &token) in history.iter().enumerate() {
        let previous_eos = last_eos_inclusive;
        if token == eos {
            last_eos_inclusive = position as i64;
        }
        let segment_start = previous_eos + 1;
        let position_in_segment = position as i64 - segment_start;
        let source = position as i64 - shift as i64;
        let valid = position_in_segment >= shift as i64 && source >= 0;
        output.push(if valid { history[source as usize] } else { eos });
    }
    output
}

#[allow(clippy::too_many_arguments)]
pub(super) fn host_ngram_ids_cached(
    cache_ids: &mut Vec<i64>,
    cache_history: &mut Vec<i64>,
    cache_last_eos: &mut i64,
    token_ids: &[u32],
    multipliers: &[i64],
    sizes: &[i64],
    offsets: &[i64],
    max_ngram: usize,
    heads_per_ngram: usize,
    eos_token_id: u32,
) {
    let context = max_ngram - 1;
    let eos = eos_token_id as i64;
    let total_heads = (max_ngram - 1) * heads_per_ngram;
    if cache_history.is_empty() {
        cache_history.extend(std::iter::repeat_n(eos, context));
        *cache_last_eos = context as i64 - 1; // every prefix row IS an eos
        cache_ids.clear();
    }
    let cached_tokens = (cache_history.len() - context).min(cache_ids.len() / total_heads);
    // Longest common prefix of the cached tokens and the requested ones.
    let mut keep = cached_tokens.min(token_ids.len());
    for i in 0..keep {
        if cache_history[context + i] != token_ids[i] as i64 {
            keep = i;
            break;
        }
    }
    if keep < cached_tokens {
        // Rewind: drop the diverged tail and rebuild the eos scan over what survives.
        cache_history.truncate(context + keep);
        cache_ids.truncate(keep * total_heads);
        *cache_last_eos = cache_history
            .iter()
            .rposition(|&v| v == eos)
            .map(|p| p as i64)
            .unwrap_or(-1);
    }
    for &token in &token_ids[keep..] {
        let position = cache_history.len();
        let value = token as i64;
        cache_history.push(value);
        // `shift_right_ignore_eos`: `previous_eos` is read BEFORE this position updates it.
        let previous_eos = *cache_last_eos;
        if value == eos {
            *cache_last_eos = position as i64;
        }
        let segment_start = previous_eos + 1;
        let position_in_segment = position as i64 - segment_start;
        let shifted_at = |shift: usize| -> i64 {
            if shift == 0 {
                return cache_history[position];
            }
            let source = position as i64 - shift as i64;
            if position_in_segment >= shift as i64 && source >= 0 {
                cache_history[source as usize]
            } else {
                eos
            }
        };
        // Same op order as the twin: shift 0 multiply, then xor the higher shifts in order.
        let mut row = vec![0i64; total_heads];
        for ngram in 2..=max_ngram {
            let head_start = (ngram - 2) * heads_per_ngram;
            let mut mixed = shifted_at(0).wrapping_mul(multipliers[0]);
            for shift in 1..ngram {
                mixed ^= shifted_at(shift).wrapping_mul(multipliers[shift]);
            }
            for head in 0..heads_per_ngram {
                let index = head_start + head;
                row[index] = mixed.rem_euclid(sizes[index]) + offsets[index];
            }
        }
        cache_ids.extend_from_slice(&row);
    }
    debug_assert_eq!(cache_ids.len(), token_ids.len() * total_heads);
    // Returns nothing on purpose: the caller reads the tail of `cache_ids` in place. Handing
    // back a `Vec` would clone the whole history's ids on every decode step (19 MB at a
    // 150,000-token fill), which is the O(context) cost this seam exists to delete.
}

#[allow(clippy::too_many_arguments)]
pub(super) fn host_ngram_ids(
    token_ids: &[u32],
    multipliers: &[i64],
    sizes: &[i64],
    offsets: &[i64],
    max_ngram: usize,
    heads_per_ngram: usize,
    eos_token_id: u32,
) -> Vec<i64> {
    let context = max_ngram - 1;
    let eos = eos_token_id as i64;
    let total_heads = (max_ngram - 1) * heads_per_ngram;
    let mut history = Vec::with_capacity(context + token_ids.len());
    history.extend(std::iter::repeat_n(eos, context));
    history.extend(token_ids.iter().map(|&token| token as i64));
    let shifted: Vec<Vec<i64>> = (0..max_ngram)
        .map(|shift| shift_right_ignore_eos(&history, shift, eos))
        .collect();
    let tokens = token_ids.len();
    let mut ids = vec![0i64; tokens * total_heads];
    for ngram in 2..=max_ngram {
        let head_start = (ngram - 2) * heads_per_ngram;
        for token in 0..tokens {
            let position = context + token;
            let mut mixed = shifted[0][position].wrapping_mul(multipliers[0]);
            for (shift, row) in shifted.iter().enumerate().take(ngram).skip(1) {
                mixed ^= row[position].wrapping_mul(multipliers[shift]);
            }
            for head in 0..heads_per_ngram {
                let index = head_start + head;
                ids[token * total_heads + index] = mixed.rem_euclid(sizes[index]) + offsets[index];
            }
        }
    }
    ids
}
