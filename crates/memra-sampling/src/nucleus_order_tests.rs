// Frozen comparison oracle from 809376789, followed by independent prefix/draw gates.

use super::*;

fn probabilities(s: &Sampler, logits: &[f32]) -> Vec<(u32, f32)> {
    let mut cand: Vec<_> = logits
        .iter()
        .enumerate()
        .map(|(i, &v)| (i as u32, v))
        .collect();
    s.apply_penalties_dense(&mut cand);
    if s.cfg.temperature > 0.0 && s.cfg.temperature != 1.0 {
        let inv = 1.0 / s.cfg.temperature;
        for c in &mut cand {
            c.1 *= inv;
        }
    }
    if s.cfg.top_k > 0 && s.cfg.top_k < cand.len() {
        cand.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
        cand.truncate(s.cfg.top_k);
    }
    oracle_softmax(&mut cand);
    cand
}

fn truncate_nucleus(cand: &mut Vec<(u32, f32)>, top_p: f32) {
    let mut cum = 0.0f32;
    let mut keep = 0;
    for (i, c) in cand.iter().enumerate() {
        cum += c.1;
        keep = i + 1;
        if cum >= top_p {
            break;
        }
    }
    cand.truncate(keep.max(1));
}

fn bits(cand: &[(u32, f32)]) -> Vec<(u32, u32)> {
    cand.iter().map(|&(id, p)| (id, p.to_bits())).collect()
}

fn check_case(logits: &[f32], cfg: SamplerConfig, history: &[u32], draws: usize) {
    let mut actual = Sampler::with_nucleus_ordering(cfg.clone(), NucleusOrdering::PrefixRadix);
    let mut expected = Sampler::new(cfg);
    for &t in history {
        actual.accept(t);
        expected.accept(t);
    }
    for step in 0..draws {
        if !actual.is_greedy() && actual.cfg.top_p < 1.0 {
            let mut a = probabilities(&actual, logits);
            let mut b = a.clone();
            NucleusOrderScratch::default().sort(&mut a, actual.cfg.top_p);
            b.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
            truncate_nucleus(&mut a, actual.cfg.top_p);
            truncate_nucleus(&mut b, actual.cfg.top_p);
            assert_eq!(
                bits(&a),
                bits(&b),
                "candidate prefix at step {step}, {:?}",
                actual.cfg
            );
        }
        let want = reference_sample(&mut expected, logits);
        let got = actual.sample(logits);
        assert_eq!(got, want, "draw {step}, {:?}", actual.cfg);
        assert_eq!(actual.rng.state, expected.rng.state, "RNG advancement");
        actual.accept(got);
        expected.accept(want);
    }
}

fn vendor_row(n: usize) -> Vec<f32> {
    let mut row = vec![f32::NEG_INFINITY; n];
    for i in 0..128.min(n) {
        row[(i * 73 + 19) % n] = 10.0 - i as f32 * 0.17;
    }
    row
}

fn vendor_finite_row(n: usize) -> Vec<f32> {
    let mut row: Vec<_> = random_row(n, 91)
        .into_iter()
        .map(|v| -40.0 + v * 0.25)
        .collect();
    for i in 0..128.min(n) {
        row[(i * 73 + 19) % n] = 10.0 - i as f32 * 0.17;
    }
    row
}

fn random_row(n: usize, seed: u64) -> Vec<f32> {
    let mut rng = SplitMix64::new(seed);
    (0..n).map(|_| rng.next_f32() * 40.0 - 20.0).collect()
}

#[test]
fn radix_keys_match_total_cmp_including_signed_zero() {
    let mut a = [
        f32::MIN,
        -100.0,
        -f32::MIN_POSITIVE,
        -0.0,
        0.0,
        f32::MIN_POSITIVE,
        1.0,
        f32::MAX,
    ];
    let mut b = a;
    a.sort_by(|a, b| b.total_cmp(a));
    b.sort_by_key(|v| descending_probability_key(*v));
    assert_eq!(a.map(f32::to_bits), b.map(f32::to_bits));
}

#[test]
fn discarded_zero_tail_does_not_disable_radix() {
    let row = vendor_row(154_880);
    let cfg = SamplerConfig {
        temperature: 1.0,
        top_p: 0.95,
        ..Default::default()
    };
    let mut sampler = Sampler::with_nucleus_ordering(cfg.clone(), NucleusOrdering::PrefixRadix);
    sampler.sample(&row);
    assert_eq!(sampler.nucleus_sort_counts(), (1, 0));
    check_case(&row, cfg, &[], 4);
}

#[test]
fn ties_inside_and_across_nucleus_preserve_legacy_order() {
    for top_p in [0.1, 0.5, 0.95, 0.99999994] {
        let mut row = vendor_row(4096);
        row[19] = 10.0;
        row[92] = 10.0;
        row[165] = 10.0;
        let cfg = SamplerConfig {
            temperature: 1.0,
            top_p,
            ..Default::default()
        };
        let mut sampler = Sampler::with_nucleus_ordering(cfg.clone(), NucleusOrdering::PrefixRadix);
        sampler.sample(&row);
        assert_eq!(
            sampler.nucleus_sort_counts().0,
            0,
            "tie must use legacy comparator"
        );
        check_case(&row, cfg, &[19, 92, 165], 5);
    }
    let row = vec![0.0; 4096];
    check_case(
        &row,
        SamplerConfig {
            temperature: 1.0,
            top_p: 0.95,
            ..Default::default()
        },
        &[],
        8,
    );
}

#[test]
fn exact_cumulative_boundary_and_next_float_match() {
    let row = vendor_row(4096);
    let s = Sampler::new(SamplerConfig {
        temperature: 1.0,
        ..Default::default()
    });
    let mut p = probabilities(&s, &row);
    p.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
    let boundary = p.iter().take(7).map(|x| x.1).sum::<f32>();
    for top_p in [
        f32::from_bits(boundary.to_bits() - 1),
        boundary,
        f32::from_bits(boundary.to_bits() + 1),
    ] {
        check_case(
            &row,
            SamplerConfig {
                temperature: 1.0,
                top_p,
                ..Default::default()
            },
            &[],
            8,
        );
    }
}

#[test]
fn nonfinite_extremes_underflow_and_signed_zero_keep_existing_behavior() {
    let base = random_row(4096, 77);
    for values in [
        vec![0.0, -0.0, f32::MIN_POSITIVE, -f32::MIN_POSITIVE],
        vec![f32::MAX, f32::MIN, 80.0, -1000.0],
        vec![f32::INFINITY],
        vec![f32::NEG_INFINITY],
        vec![f32::NAN],
        vec![f32::from_bits(0xffc00001)],
    ] {
        let mut row = base.clone();
        for (i, v) in values.into_iter().enumerate() {
            row[i * 73 + 19] = v;
        }
        for temperature in [0.01, 1.0, 2.0] {
            for top_p in [0.0, f32::MIN_POSITIVE, 0.95, 1.0] {
                check_case(
                    &row,
                    SamplerConfig {
                        temperature,
                        top_p,
                        seed: 9,
                        ..Default::default()
                    },
                    &[],
                    3,
                );
            }
        }
    }
    check_case(
        &vec![f32::NEG_INFINITY; 4096],
        SamplerConfig {
            temperature: 1.0,
            top_p: 0.95,
            ..Default::default()
        },
        &[],
        3,
    );
}

#[test]
fn differential_filter_penalty_seed_matrix() {
    let rows = [vendor_row(4096), random_row(4096, 73)];
    for row in &rows {
        for seed in [0, 20260908, u64::MAX] {
            for temperature in [0.0, 0.7, 1.0, 2.0] {
                for top_p in [-0.1, 0.1, 0.95, 0.99999994, 1.0] {
                    for top_k in [0, 1, 17, 4096, 4097] {
                        for min_p in [0.0, 0.05] {
                            for penalized in [false, true] {
                                let cfg = SamplerConfig {
                                    temperature,
                                    top_p,
                                    top_k,
                                    min_p,
                                    seed,
                                    penalty_last_n: if penalized { 8 } else { 0 },
                                    penalty_repeat: if penalized { 1.1 } else { 1.0 },
                                    penalty_freq: if penalized { 0.2 } else { 0.0 },
                                    penalty_present: if penalized { 0.3 } else { 0.0 },
                                };
                                check_case(
                                    row,
                                    cfg,
                                    &[19, 92, 19, 165, 19, 92, 3, 3, 19, 7, 11, 19],
                                    2,
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn randomized_vocab_shapes_and_masked_rows_match() {
    for n in [1, 17, 1023, 1024, 1025, 8192, 154880] {
        for seed in [7, 29, 99] {
            let mut row = random_row(n, seed);
            for i in (0..n).step_by(11) {
                row[i] = f32::NEG_INFINITY;
            }
            if n == 1 {
                row[0] = 0.0;
            }
            check_case(
                &row,
                SamplerConfig {
                    temperature: 1.0,
                    top_p: 0.95,
                    seed,
                    ..Default::default()
                },
                &[],
                3,
            );
        }
    }
}

#[test]
fn radix_draws_have_reference_support_and_expected_mass() {
    let mut row = vec![f32::NEG_INFINITY; 1024];
    row[513] = 0.6f32.ln();
    row[3] = 0.25f32.ln();
    row[701] = 0.15f32.ln();
    let mut a = Sampler::for_glm5_plain_tp2(SamplerConfig {
        temperature: 1.0,
        top_p: 0.8,
        seed: 42,
        ..Default::default()
    });
    let mut b = Sampler::new(SamplerConfig {
        temperature: 1.0,
        top_p: 0.8,
        seed: 42,
        ..Default::default()
    });
    a.nucleus_ordering = NucleusOrdering::PrefixRadix;
    let mut high = 0;
    for _ in 0..4096 {
        let x = a.sample(&row);
        assert_eq!(x, reference_sample(&mut b, &row));
        assert!(x == 513 || x == 3);
        high += usize::from(x == 513);
    }
    assert!(
        high > 2700 && high < 3100,
        "expected about 2891 top-token draws, got {high}"
    );
    assert_eq!(a.nucleus_sort_counts(), (4096, 0));
}

#[test]
#[ignore = "remote CPU measurement, run explicitly without compilation overlap"]
fn benchmark_pinned_rows() {
    use std::{hint::black_box, time::Instant};
    let out = std::path::Path::new("target/nucleus-bench");
    std::fs::create_dir_all(out).unwrap();
    for (name, row) in [
        ("vendor_peaked", vendor_row(154880)),
        ("vendor_finite_tail", vendor_finite_row(154880)),
        ("broad", random_row(154880, 20260908)),
    ] {
        let bytes: Vec<u8> = row.iter().flat_map(|x| x.to_le_bytes()).collect();
        std::fs::write(out.join(format!("{name}.f32")), bytes).unwrap();
        let cfg = SamplerConfig {
            temperature: 1.0,
            top_p: 0.95,
            seed: 20260908,
            ..Default::default()
        };
        let mut a = Sampler::with_nucleus_ordering(cfg.clone(), NucleusOrdering::PrefixRadix);
        let mut b = Sampler::new(cfg);
        for _ in 0..3 {
            assert_eq!(a.sample(&row), reference_sample(&mut b, &row));
        }
        let mut optimized = Vec::new();
        let mut comparison = Vec::new();
        for arm in [false, true, true, false, false, true, true, false] {
            let start = Instant::now();
            for _ in 0..20 {
                if arm {
                    black_box(a.sample(black_box(&row)));
                } else {
                    black_box(reference_sample(&mut b, black_box(&row)));
                }
            }
            let ms = start.elapsed().as_secs_f64() * 1000.0 / 20.0;
            if arm {
                optimized.push(ms);
            } else {
                comparison.push(ms);
            }
        }
        println!(
            "NUCLEUS_BENCH name={name} vocab={} comparison_ms={comparison:?} radix_ms={optimized:?} counts={:?}",
            row.len(),
            a.nucleus_sort_counts()
        );
    }
}

fn reference_sample(s: &mut Sampler, logits: &[f32]) -> u32 {
    // Greedy fast path: argmax over RAW logits (penalties don't change the argmax direction
    // enough to matter for the reference path; llama greedy is also pre-penalty argmax only
    // when no penalties set — but to stay correct under penalties we still apply them first).
    if s.is_greedy()
        && s.cfg.penalty_repeat == 1.0
        && s.cfg.penalty_freq == 0.0
        && s.cfg.penalty_present == 0.0
    {
        return argmax_u32(logits);
    }

    // Work on (id, logit) candidates.
    let mut cand: Vec<(u32, f32)> = logits
        .iter()
        .enumerate()
        .map(|(i, &l)| (i as u32, l))
        .collect();

    // 1. Penalties (operate on logits, over the last-n history window).
    // `cand` is DENSE and INDEX-ALIGNED here by construction (built from
    // `logits.iter().enumerate()` immediately above, nothing has filtered it yet), so the
    // penalty pass indexes straight into it instead of hashing every candidate. See
    // `apply_penalties_dense`.
    s.apply_penalties_dense(&mut cand);

    // Greedy-with-penalties: argmax after penalties, no sampling.
    if s.is_greedy() {
        let mut best = cand[0];
        for &c in &cand[1..] {
            if c.1 > best.1 {
                best = c;
            }
        }
        return best.0;
    }

    // 2. Temperature scale.
    if s.cfg.temperature > 0.0 && s.cfg.temperature != 1.0 {
        let inv = 1.0 / s.cfg.temperature;
        for c in cand.iter_mut() {
            c.1 *= inv;
        }
    }

    // 3. top-k: keep the k highest-logit candidates (partial sort by logit desc).
    if s.cfg.top_k > 0 && s.cfg.top_k < cand.len() {
        cand.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
        cand.truncate(s.cfg.top_k);
    }

    // softmax over the surviving candidates (numerically stable).
    oracle_softmax(&mut cand);

    // 4. top-p (nucleus): smallest set whose cumulative prob >= top_p. Needs desc-by-prob order.
    if s.cfg.top_p < 1.0 {
        cand.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
        let mut cum = 0.0f32;
        let mut keep = 0usize;
        for (i, c) in cand.iter().enumerate() {
            cum += c.1;
            keep = i + 1;
            if cum >= s.cfg.top_p {
                break;
            }
        }
        cand.truncate(keep.max(1));
    }

    // 5. min-p: keep candidates with prob >= min_p * max_prob.
    if s.cfg.min_p > 0.0 {
        let maxp = cand.iter().map(|c| c.1).fold(0.0f32, f32::max);
        let thresh = s.cfg.min_p * maxp;
        cand.retain(|c| c.1 >= thresh);
        if cand.is_empty() {
            return argmax_u32(logits);
        } // safety
    }

    // renormalize the surviving probs and draw.
    let sum: f32 = cand.iter().map(|c| c.1).sum();
    let r = s.rng.next_f32() * sum;
    let mut acc = 0.0f32;
    for c in &cand {
        acc += c.1;
        if acc >= r {
            return c.0;
        }
    }
    cand.last().unwrap().0
}

fn oracle_softmax(cand: &mut [(u32, f32)]) {
    let maxl = cand.iter().map(|c| c.1).fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for c in cand.iter_mut() {
        let e = (c.1 - maxl).exp();
        c.1 = e;
        sum += e;
    }
    let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
    for c in cand.iter_mut() {
        c.1 *= inv;
    }
}

#[test]
fn shared_default_skips_radix_scratch_even_for_broad_rows() {
    let row = random_row(154_880, 20260908);
    let cfg = SamplerConfig {
        temperature: 1.0,
        top_p: 0.95,
        ..Default::default()
    };
    let mut sampler = Sampler::new(cfg.clone());
    let mut reference = Sampler::new(cfg);
    for _ in 0..4 {
        assert_eq!(sampler.sample(&row), reference_sample(&mut reference, &row));
        assert_eq!(sampler.rng.state, reference.rng.state);
    }
    assert_eq!(sampler.nucleus_sort_counts(), (0, 4));
    assert!(sampler.nucleus_order.keys.is_empty());
    assert!(sampler.nucleus_order.order.is_empty());
    assert!(sampler.nucleus_order.scratch.is_empty());
    assert!(sampler.nucleus_order.sorted.is_empty());
}

#[test]
fn glm_plain_constructor_limits_ordering_to_qualified_sampling_shape() {
    let row = vendor_finite_row(154_880);
    let cfg = SamplerConfig {
        temperature: 1.0,
        top_p: 0.95,
        ..Default::default()
    };
    let mut qualified = Sampler::for_glm5_plain_tp2(cfg.clone());
    let mut reference = Sampler::new(cfg.clone());
    assert_eq!(qualified.sample(&row), reference.sample(&row));
    assert_eq!(qualified.nucleus_sort_counts(), (1, 0));
    for changed in [
        SamplerConfig {
            temperature: 0.8,
            ..cfg.clone()
        },
        SamplerConfig {
            top_p: 0.9,
            ..cfg.clone()
        },
        SamplerConfig {
            top_k: 40,
            ..cfg.clone()
        },
        SamplerConfig {
            min_p: 0.01,
            ..cfg.clone()
        },
        SamplerConfig {
            penalty_last_n: 32,
            penalty_repeat: 1.1,
            ..cfg.clone()
        },
        SamplerConfig {
            penalty_last_n: 32,
            penalty_freq: 0.1,
            ..cfg.clone()
        },
        SamplerConfig {
            penalty_last_n: 32,
            penalty_present: 0.1,
            ..cfg.clone()
        },
    ] {
        let mut actual = Sampler::for_glm5_plain_tp2(changed.clone());
        let mut expected = Sampler::new(changed);
        actual.accept(19);
        expected.accept(19);
        assert_eq!(actual.sample(&row), expected.sample(&row));
        assert_eq!(actual.rng.state, expected.rng.state);
        assert_eq!(actual.nucleus_sort_counts(), (0, 1));
        assert!(actual.nucleus_order.keys.is_empty());
    }
}
