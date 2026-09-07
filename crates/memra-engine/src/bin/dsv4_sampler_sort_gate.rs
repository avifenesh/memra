//! CPU-only sampler phase attribution and exact radix-order gate.
//! Production defaults to radix; comparison is the oracle. No CUDA context is created here.
use memra_engine::dsv4_gpu::{
    Dsv4SampleCfg, Dsv4SamplerOrder, dsv4_candidate_order, dsv4_pos_uniform, dsv4_sample_row,
    dsv4_sample_row_ordered,
};
use sha2::{Digest, Sha256};
use std::{path::Path, time::Instant};

fn ordered(logits: &[f32], radix: bool) -> Vec<u32> {
    if radix {
        return dsv4_candidate_order(logits, Dsv4SamplerOrder::Radix);
    }
    // Frozen pre-rewrite comparator remains independent of the production helper.
    let mut indices: Vec<u32> = (0..logits.len() as u32).collect();
    indices.sort_by(|&a, &b| {
        logits[b as usize]
            .partial_cmp(&logits[a as usize])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    indices
}

/// Same arithmetic as production; only sorting can vary. Each return is checked
/// against the actual production function outside the timed region.
fn profile(logits: &[f32], pos: usize, cfg: &Dsv4SampleCfg, radix: bool) -> (u32, [u128; 3]) {
    let start = Instant::now();
    let mut idx = ordered(logits, radix);
    let sorted = Instant::now();
    let k = if cfg.top_k == 0 || cfg.top_k > logits.len() {
        logits.len()
    } else {
        cfg.top_k
    };
    idx.truncate(k);
    let m = logits[idx[0] as usize] as f64;
    let t = cfg.temperature as f64;
    let mut probs: Vec<f64> = idx
        .iter()
        .map(|&i| (((logits[i as usize] as f64) - m) / t).exp())
        .collect();
    let z: f64 = probs.iter().sum();
    for p in &mut probs {
        *p /= z;
    }
    let mut cum = 0.0f64;
    let mut keep = probs.len();
    for (i, p) in probs.iter().enumerate() {
        cum += p;
        if cum >= cfg.top_p as f64 {
            keep = i + 1;
            break;
        }
    }
    probs.truncate(keep);
    idx.truncate(keep);
    let z2: f64 = probs.iter().sum();
    let u = dsv4_pos_uniform(cfg.seed, pos) * z2;
    let mut acc = 0.0f64;
    let mut token = idx[keep - 1];
    for (i, p) in probs.iter().enumerate() {
        acc += p;
        if u < acc {
            token = idx[i];
            break;
        }
    }
    let end = Instant::now();
    (
        token,
        [
            (sorted - start).as_nanos(),
            (end - sorted).as_nanos(),
            (end - start).as_nanos(),
        ],
    )
}

fn main() {
    let args: Vec<_> = std::env::args().collect();
    assert_eq!(
        args.len(),
        2,
        "usage: dsv4_sampler_sort_gate <frozen-rows-dir>"
    );
    let root = Path::new(&args[1]);
    let mut compared = 0usize;
    println!(
        "PROTOCOL cpu_only=true numerical_program=descending_value_ascending_id_ordered_f64 ABBAx3=true timings_exclude_oracle_and_order_checks=true"
    );
    for window in 0..3 {
        for program in ["reference", "matrix"] {
            let path = root.join(format!("window-{window}.{program}.f32le"));
            let raw = std::fs::read(&path).expect("frozen rows");
            assert_eq!(raw.len(), 64 * 129280 * 4);
            println!(
                "INPUT window={window} program={program} sha256={:x}",
                Sha256::digest(&raw)
            );
            let values: Vec<f32> = raw
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect();
            assert!(values.iter().all(|v| v.is_finite()));
            for (row, logits) in values.chunks_exact(129280).enumerate() {
                assert_eq!(
                    ordered(logits, false),
                    ordered(logits, true),
                    "full candidate order"
                );
                compared += logits.len();
                for (temperature, top_p, top_k) in
                    [(1.0, 1.0, 0), (0.2, 0.95, 0), (2.0, 0.5, 32), (1.0, 1.0, 1)]
                {
                    let cfg = Dsv4SampleCfg {
                        temperature,
                        top_p,
                        top_k,
                        seed: 20260906 + row as u64,
                    };
                    let want = dsv4_sample_row(logits, row, &cfg).expect("production oracle");
                    assert_eq!(
                        want,
                        dsv4_sample_row_ordered(logits, row, &cfg, Dsv4SamplerOrder::Radix)
                            .unwrap(),
                        "integrated radix sampler"
                    );
                    assert_eq!(
                        want,
                        profile(logits, row, &cfg, false).0,
                        "instrument oracle"
                    );
                    assert_eq!(
                        want,
                        profile(logits, row, &cfg, true).0,
                        "radix sampled identity"
                    );
                }
            }
            let cfg = Dsv4SampleCfg {
                temperature: 1.0,
                top_p: 1.0,
                top_k: 0,
                seed: 20260906,
            };
            let rows = &values[..8 * 129280];
            // Warm both modes, then use the same eight captured rows per timed pass.
            for radix in [false, true] {
                for (pos, logits) in rows.chunks_exact(129280).enumerate() {
                    std::hint::black_box(profile(logits, pos, &cfg, radix));
                }
            }
            for (ordinal, radix) in [false, true, true, false].repeat(3).into_iter().enumerate() {
                let mut times = [0u128; 3];
                let mut hash = Sha256::new();
                for (pos, logits) in rows.chunks_exact(129280).enumerate() {
                    let (token, row_times) =
                        profile(std::hint::black_box(logits), pos, &cfg, radix);
                    hash.update(token.to_le_bytes());
                    for (sum, value) in times.iter_mut().zip(row_times) {
                        *sum += value;
                    }
                }
                println!(
                    "MEASURE window={window} program={program} ordinal={ordinal} radix={radix} rows=8 order_ns={} probability_draw_ns={} total_ns={} output_sha256={:x}",
                    times[0],
                    times[1],
                    times[2],
                    hash.finalize()
                );
            }
        }
    }
    println!(
        "PASS full candidate order elements={compared} and 1536 sampled comparisons; comparison/radix identity"
    );
}

#[cfg(test)]
mod tests {
    use super::ordered;
    #[test]
    fn signed_zero_ties_infinities_and_all_tail_sizes_preserve_order() {
        let mut values = vec![0.0, -0.0, f32::INFINITY, f32::NEG_INFINITY, -1.0, 1.0];
        let mut state = 7u32;
        for _ in 0..129280 {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            let value = f32::from_bits(state);
            values.push(if value.is_nan() { 0.0 } else { value });
        }
        for n in [0, 1, 2, 3, 6, 31, 32, 33, 127, 128, 129, 1025, 129280] {
            assert_eq!(
                ordered(&values[..n], false),
                ordered(&values[..n], true),
                "length {n}"
            );
        }
        assert_eq!(&ordered(&[0.0, -0.0, 0.0, -0.0], true), &[0, 1, 2, 3]);
    }
}
