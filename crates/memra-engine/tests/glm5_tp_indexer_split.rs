//! Pool-split membership, emission order, arithmetic and two-device transport gates.
//! GPU tests are ignored and must run on the lane's non-serving pair under its GPU lock.
use memra_engine::{Engine, tp_ar::ArLink};

fn select(mut candidates: Vec<(f32, usize)>, k: usize) -> Vec<(f32, usize)> {
    candidates.retain(|(s, _)| s.is_finite());
    candidates.sort_by(|(a, p), (b, q)| b.partial_cmp(a).unwrap().then(p.cmp(q)));
    candidates.truncate(k);
    candidates.sort_by_key(|(_, p)| *p);
    candidates
}

fn split_merge(scores: &[f32], k: usize) -> Vec<usize> {
    let mid = scores.len().div_ceil(2);
    let local = |lo, hi| select((lo..hi).map(|p| (scores[p], p)).collect(), k);
    let mut candidates = local(0, mid);
    candidates.extend(local(mid, scores.len()));
    select(candidates, k).into_iter().map(|(_, p)| p).collect()
}

#[test]
fn merge_ties_across_ranks_uses_global_index() {
    assert_eq!(split_merge(&[1.0; 9], 5), [0, 1, 2, 3, 4]);
    assert_eq!(split_merge(&[-0.0, 0.0, -0.0, 0.0], 2), [0, 1]);
    assert_eq!(split_merge(&[3.0, 2.0, 2.0, 3.0, 2.0], 3), [0, 1, 3]);
}

#[test]
fn merge_k_larger_than_candidates_and_empty_ranks() {
    assert_eq!(split_merge(&[1.0, 2.0, 3.0], 8), [0, 1, 2]);
    assert_eq!(split_merge(&[f32::NAN, f32::INFINITY, -1.0], 8), [2]);
    assert!(split_merge(&[f32::NEG_INFINITY; 7], 3).is_empty());
    assert!(split_merge(&[], 4).is_empty());
    assert!(split_merge(&[1.0; 7], 0).is_empty());
}

fn scores(n: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    (0..n)
        .map(|p| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            match p % 37 {
                0 => f32::NEG_INFINITY,
                1 => f32::NAN,
                2 => f32::INFINITY,
                3 => -0.0,
                _ => ((state >> 32) as i32 % 31) as f32 / 8.0,
            }
        })
        .collect()
}

#[test]
fn local_topk_union_contains_global_topk() {
    for n in [2, 3, 17, 513, 1025] {
        let s = scores(n, 91);
        for k in [0, 1, 7, 512, 2048] {
            let want: Vec<_> = select(
                s.iter().copied().enumerate().map(|(p, s)| (s, p)).collect(),
                k,
            )
            .into_iter()
            .map(|(_, p)| p)
            .collect();
            assert_eq!(split_merge(&s, k), want, "n={n}, k={k}");
        }
    }
}

#[test]
#[ignore = "needs the lane's two peer-access CUDA devices"]
fn gpu_pool_split_matches_replicated_selection_and_cpu() {
    let a = Engine::new(0).expect("rank 0");
    let b = Engine::new(1).expect("rank 1");
    memra_engine::tp::grant_peer_access(&a, &b, "indexer split gate").unwrap();
    memra_engine::tp::grant_peer_access(&b, &a, "indexer split gate").unwrap();
    let devs = [&a, &b];
    let mut link = ArLink::new(&devs).unwrap();
    // Includes the single/multi-CTA crossover on each HALF, odd counts, ragged k,
    // the 1M pool plane, fewer finite candidates than k and all-equal cross-rank ties.
    for (n, k, rows) in [
        (3usize, 3usize, 4usize),
        (1025, 512, 4),
        (32213, 512, 4),
        (131073, 512, 1),
        (250483, 512, 1),
        (8193, 2048, 4),
    ] {
        for pattern in 0..3 {
            let pool = 4;
            let width = k * pool + pool - 1;
            let first_pos = n * pool - rows;
            let mut full_scores = Vec::new();
            for row in 0..rows {
                let mut s = if pattern == 0 {
                    scores(n, 999 + row as u64)
                } else if pattern == 1 {
                    vec![0.0; n]
                } else {
                    vec![f32::NEG_INFINITY; n]
                };
                for (p, score) in s.iter_mut().enumerate() {
                    if p >= (first_pos + row + 1) / pool {
                        *score = f32::NEG_INFINITY;
                    }
                }
                full_scores.extend(s);
            }
            let mid = n.div_ceil(2);
            let mut parts = Vec::new();
            for (r, (lo, hi)) in [(0, mid), (mid, n)].into_iter().enumerate() {
                let local: Vec<_> = (0..rows)
                    .flat_map(|q| full_scores[q * n + lo..q * n + hi].iter().copied())
                    .collect();
                let sd = devs[r].htod(&local).unwrap();
                parts.push(
                    devs[r]
                        .mla_kpool_candidates(&sd, rows, hi - lo, lo, k)
                        .unwrap(),
                );
            }
            let whole = a.htod(&full_scores).unwrap();
            let mut expected = a.uninit_i32(rows * width).unwrap();
            a.mla_kpool_select(
                &whole,
                &mut expected,
                rows,
                n,
                pool,
                k,
                width,
                first_pos,
                true,
            )
            .unwrap();
            let expected = a.dtoh_i32(&expected).unwrap();
            for q in 0..rows {
                let selected = split_merge(&full_scores[q * n..(q + 1) * n], k);
                let mut want: Vec<i32> = selected
                    .iter()
                    .flat_map(|p| (p * pool..(p + 1) * pool).map(|i| i as i32))
                    .collect();
                let visible = first_pos + q + 1;
                want.extend((visible - visible % pool..visible).map(|i| i as i32));
                want.resize(width, -1);
                assert_eq!(&expected[q * width..(q + 1) * width], want);
            }
            // Reuse the SAME signal blocks while changing launch sizes and candidate contents.
            for _ in 0..3 {
                let span = 2 * rows * k;
                let mut x = a.uninit_i32(2 * span).unwrap();
                let mut y = b.uninit_i32(2 * span).unwrap();
                link.gather_i32(&devs, &[&parts[0], &parts[1]], &mut [&mut x, &mut y], span)
                    .unwrap();
                let merged = [
                    a.mla_kpool_merge(&x, rows, k, pool, width, first_pos)
                        .unwrap(),
                    b.mla_kpool_merge(&y, rows, k, pool, width, first_pos)
                        .unwrap(),
                ];
                for r in 0..2 {
                    assert_eq!(
                        devs[r].dtoh_i32(&merged[r]).unwrap(),
                        expected,
                        "rank={r}, pools={n}, k={k}, pattern={pattern}"
                    );
                }
                assert_eq!(link.barrier_errors(&devs).unwrap(), [0, 0]);
            }
        }
    }
}

#[test]
#[ignore = "needs the lane's CUDA device"]
fn gpu_range_scores_are_bit_identical_including_causal_boundary() {
    let e = Engine::new(0).unwrap();
    // Same caller dispatch and RP environment as the OFF/ON production-shape cells.
    // The early prime queries precede the peer's entire pool range.
    for (rows, n, first_pos) in [
        (1usize, 32213usize, 128851usize),
        (1, 250483, 1001931),
        (128, 257, 900),
        (128, 33, 4),
        (8, 129, 508),
    ] {
        let heads = 32;
        let d = 128;
        let finite = |n, seed| {
            scores(n, seed)
                .into_iter()
                .map(|v| if v.is_finite() { v } else { 0.0 })
                .collect::<Vec<_>>()
        };
        let q = e.htod(&finite(rows * heads * d, 71)).unwrap();
        let keys = e.htod(&finite(n * d, 79)).unwrap();
        let hw = e.htod(&finite(rows * heads, 89)).unwrap();
        let mut all = e.uninit(rows * n).unwrap();
        e.mla_kpool_score(
            &q,
            &keys,
            &hw,
            &mut all,
            rows,
            heads,
            d,
            n,
            4,
            first_pos,
            (d as f32).powf(-0.5),
            (heads as f32).powf(-0.5),
        )
        .unwrap();
        let all = e.dtoh(&all).unwrap();
        let mid = n.div_ceil(2);
        for (lo, hi) in [(0, mid), (mid, n)] {
            let mut part = e.uninit(rows * (hi - lo)).unwrap();
            e.mla_kpool_score_range(
                &q,
                &keys,
                &hw,
                &mut part,
                rows,
                heads,
                d,
                hi - lo,
                lo,
                4,
                first_pos,
                (d as f32).powf(-0.5),
                (heads as f32).powf(-0.5),
            )
            .unwrap();
            let part = e.dtoh(&part).unwrap();
            for row in 0..rows {
                for p in lo..hi {
                    assert_eq!(
                        part[row * (hi - lo) + p - lo].to_bits(),
                        all[row * n + p].to_bits(),
                        "row={row} pool={p} rows={rows} pools={n}"
                    );
                }
            }
        }
    }
}
