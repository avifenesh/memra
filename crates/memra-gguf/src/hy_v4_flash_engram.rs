//! Hy4-flash Engram hashing: the integer half of transformers `HYV4FlashEngramLayer`.
//!
//! The semantic source is transformers `modeling_hy_v4_flash.py` (PR #50903 at b865850d0532):
//! `HYV4FlashTextModel.build_ngram_tokens`, `HYV4FlashModel.build_engram_inputs`,
//! `HYV4FlashEngramLayer.draw_hash_multipliers` (redrawn by `_init_weights` from the loaded
//! lookup table) and `HYV4FlashEngramLayer.hash_ngrams`. Every function takes its program from
//! the typed [`HyV4EngramPlan`]; the table moduli (the config's `engram_layer_head_vocab_sizes`
//! for the layer, order-major) and the checkpoint's `lookup_table` are data. Gated bit for bit
//! against the pinned code on the artifact's config by memra-reference
//! `tests/hy_v4_flash_engram.rs`.

use crate::dsv41_forward::numpy_odd_multipliers;
use crate::model_plan::{HyV4EngramMultiplierBound, HyV4EngramPlan};

/// `HYV4FlashTextModel.build_ngram_tokens`: per position the `max_ngram` token ids ending there,
/// newest first (`[tokens, max_ngram]` row-major). A slot `k` back reads `pad` where the
/// position is less than `k` into its document (`positions` restart at 0 on each document of a
/// packed row), so an n-gram never spans two documents.
pub fn ngram_tokens(plan: &HyV4EngramPlan, ids: &[u32], positions: &[u32]) -> Vec<u32> {
    assert_eq!(ids.len(), positions.len(), "one position per token");
    let n = plan.max_ngram as usize;
    let mut out = Vec::with_capacity(ids.len() * n);
    for (t, &position) in positions.iter().enumerate() {
        for k in 0..n {
            out.push(if (position as usize) < k || t < k {
                plan.pad_token_id
            } else {
                ids[t - k]
            });
        }
    }
    out
}

/// `HYV4FlashModel.build_engram_inputs`' mask on text: a position owns an n-gram unless its
/// token is the pad id. (Image and video placeholder ids are excluded too in transformers; this
/// program refuses image parts.)
pub fn engram_mask(plan: &HyV4EngramPlan, ids: &[u32]) -> Vec<bool> {
    ids.iter().map(|id| *id != plan.pad_token_id).collect()
}

/// The size the multipliers are bounded by, per the plan: `max(lookup_table) + 1`.
pub fn compressed_vocab(plan: &HyV4EngramPlan, lookup_table: &[i32]) -> Result<u64, String> {
    if lookup_table.len() != plan.lookup_table_size as usize {
        return Err(format!(
            "engram lookup_table has {} entries, the plan {}",
            lookup_table.len(),
            plan.lookup_table_size
        ));
    }
    match plan.multiplier_bound {
        HyV4EngramMultiplierBound::LookupTableMaxPlusOne => {
            let max = lookup_table
                .iter()
                .copied()
                .max()
                .ok_or("empty lookup_table")?;
            if lookup_table.iter().any(|v| *v < 0) {
                return Err("engram lookup_table holds a negative id".into());
            }
            Ok(max as u64 + 1)
        }
    }
}

/// `HYV4FlashEngramLayer.draw_hash_multipliers`: numpy `default_rng(hash_seed)` (the plan's
/// seed is `engram_seed + 10007 * layer_id`, 1-based), `max_ngram` odd multipliers bounded by
/// the compressed vocabulary.
pub fn hash_multipliers(plan: &HyV4EngramPlan, compressed_vocab: u64) -> Result<Vec<i64>, String> {
    numpy_odd_multipliers(plan.hash_seed, plan.max_ngram as usize, compressed_vocab)
}

/// The row each hash head starts at in the layer's one table: the running sum of the moduli.
pub fn head_offsets(moduli: &[u64]) -> Vec<u64> {
    moduli
        .iter()
        .scan(0u64, |acc, m| {
            let start = *acc;
            *acc += m;
            Some(start)
        })
        .collect()
}

/// `HYV4FlashEngramLayer.hash_ngrams`: `[tokens, heads]` hash ids (without the head offsets).
/// Each token id maps through `lookup_table`; an id past the table reads the pad id's entry.
/// The mapped ids are multiplied by `multipliers` slot by slot; order `n` (2..=max_ngram) XORs
/// the first `n` products, and each of its heads takes that value modulo its own prime
/// (`moduli` order-major, `plan.heads / (max_ngram - 1)` per order).
pub fn hash_ids(
    plan: &HyV4EngramPlan,
    lookup_table: &[i32],
    multipliers: &[i64],
    moduli: &[u64],
    ngram_tokens: &[u32],
) -> Result<Vec<u64>, String> {
    let n = plan.max_ngram as usize;
    let heads = plan.heads as usize;
    let orders = n - 1;
    if multipliers.len() != n {
        return Err(format!("{} multipliers for {n}-grams", multipliers.len()));
    }
    if moduli.len() != heads || !heads.is_multiple_of(orders) {
        return Err(format!(
            "{} moduli for {heads} hash heads over {orders} orders",
            moduli.len()
        ));
    }
    if moduli.iter().sum::<u64>() != plan.table_rows {
        return Err(format!(
            "engram moduli sum to {}, the table has {} rows",
            moduli.iter().sum::<u64>(),
            plan.table_rows
        ));
    }
    if lookup_table.len() != plan.lookup_table_size as usize
        || plan.pad_token_id as usize >= lookup_table.len()
    {
        return Err("engram lookup_table does not cover the pad id".into());
    }
    if !ngram_tokens.len().is_multiple_of(n) {
        return Err(format!("n-gram tokens are [tokens, {n}]"));
    }
    let per_order = heads / orders;
    let pad_entry = i64::from(lookup_table[plan.pad_token_id as usize]);
    let mut out = Vec::with_capacity(ngram_tokens.len() / n * heads);
    for row in ngram_tokens.chunks(n) {
        let mixed: Vec<i64> = row
            .iter()
            .zip(multipliers)
            .map(|(&token, &m)| {
                let compressed = lookup_table
                    .get(token as usize)
                    .map_or(pad_entry, |v| i64::from(*v));
                // The bound keeps the product inside int64 for any compressed id.
                compressed.wrapping_mul(m)
            })
            .collect();
        let mut mix = mixed[0];
        for order in 0..orders {
            mix ^= mixed[order + 1];
            for &modulus in &moduli[order * per_order..(order + 1) * per_order] {
                // torch `%` on int64 is the floored remainder; the mix is non-negative here.
                out.push(mix.rem_euclid(modulus as i64) as u64);
            }
        }
    }
    Ok(out)
}
