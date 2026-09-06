//! Gate for the fused MoE router (lane/router-fused-20260906): the GEMV with the sigmoid top-k
//! run by the last block to finish, against the two-launch pair it replaces
//! (`router_gemv` then `moe_router_sigmoid_topk`).
//!
//! BITWISE on all three outputs: the logits (the GEMV body is verbatim), the selected expert ids,
//! and the selection weights, at the served shape (288 experts, top-8, hidden 4096) and at a shape
//! that exercises the epilogue's second candidate slot on every thread (512 experts). Ties are
//! planted on purpose: with a zero correction bias and repeated logit rows two experts carry the
//! same score, and lowest-index-wins must hold in both kernels.
//!
//! RED ARM: perturbing one correction-bias entry must move the selection. Without it a bitwise
//! pass could be a top-k that never read the bias.
use memra_engine::Engine;

fn vecf(n: usize, seed: u64, amp: f32) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 2.0 * amp
        })
        .collect()
}

#[allow(clippy::type_complexity)]
fn both(
    e: &Engine,
    n_embd: usize,
    n_expert: usize,
    n_used: usize,
    t: usize,
    bias_perturb: Option<usize>,
    tie_rows: bool,
) -> (
    (Vec<f32>, Vec<i32>, Vec<f32>),
    (Vec<f32>, Vec<i32>, Vec<f32>),
) {
    let mut wh = vecf(n_expert * n_embd, 7, 0.05);
    if tie_rows {
        // expert 5 and expert 9 get identical rows: identical logits, a planted tie
        let (a, b) = (5 * n_embd, 9 * n_embd);
        let row: Vec<f32> = wh[a..a + n_embd].to_vec();
        wh[b..b + n_embd].copy_from_slice(&row);
    }
    let x = vecf(t * n_embd, 11, 1.0);
    let mut bias = vec![0.0f32; n_expert];
    if let Some(i) = bias_perturb {
        bias[i] = 5.0;
    }
    let active = vec![1u8; n_expert];
    let w_d = e.htod(&wh).unwrap();
    let x_d = e.htod(&x).unwrap();
    let b_d = e.htod(&bias).unwrap();
    let a_d = e.htod_bytes(&active).unwrap();
    let (sf, rn) = (2.5f32, true);
    // two launches
    let lg = e.router_gemv(&w_d, &x_d, n_embd, n_expert, t).unwrap();
    let (si, sw) = e
        .moe_router_sigmoid_topk(&lg, t, n_expert, n_used, n_expert, &b_d, &a_d, sf, rn)
        .unwrap();
    let pair = (
        e.dtoh_view(&lg.slice(0..t * n_expert)).unwrap(),
        e.dtoh_i32(&si).unwrap(),
        e.dtoh_view(&sw.slice(0..t * n_used)).unwrap(),
    );
    // one launch
    let (lg2, si2, sw2) = e
        .router_sigmoid_topk_fused(
            &w_d, &x_d, t, n_embd, n_expert, n_used, n_expert, &b_d, &a_d, sf, rn,
        )
        .unwrap();
    let fused = (
        e.dtoh_view(&lg2.slice(0..t * n_expert)).unwrap(),
        e.dtoh_i32(&si2).unwrap(),
        e.dtoh_view(&sw2.slice(0..t * n_used)).unwrap(),
    );
    (pair, fused)
}

#[test]
fn fused_router_is_bitwise_the_two_launch_pair() {
    let Ok(e) = Engine::new(0) else {
        eprintln!("no CUDA device; skipping");
        return;
    };
    for &(n_expert, n_used, t, ties) in &[
        (288usize, 8usize, 1usize, false),
        (288, 8, 3, true),
        (512, 8, 1, true),
        (64, 4, 2, false),
    ] {
        let ((lg, si, sw), (lg2, si2, sw2)) = both(&e, 4096, n_expert, n_used, t, None, ties);
        for i in 0..t * n_expert {
            assert_eq!(
                lg[i].to_bits(),
                lg2[i].to_bits(),
                "logit differs n_expert={n_expert} t={t} i={i}"
            );
        }
        assert_eq!(
            si, si2,
            "selection differs n_expert={n_expert} t={t} ties={ties}"
        );
        for i in 0..t * n_used {
            assert_eq!(
                sw[i].to_bits(),
                sw2[i].to_bits(),
                "weight differs n_expert={n_expert} t={t} i={i}"
            );
        }
    }
}

#[test]
fn perturbing_the_bias_moves_the_selection() {
    let Ok(e) = Engine::new(0) else {
        eprintln!("no CUDA device; skipping");
        return;
    };
    let (_, (_, base, _)) = both(&e, 4096, 288, 8, 1, None, false);
    // a +5 bias on an expert that was not selected must pull it in
    let outsider = (0..288).find(|i| !base.contains(&(*i as i32))).unwrap();
    let (_, (_, moved, _)) = both(&e, 4096, 288, 8, 1, Some(outsider), false);
    assert!(
        moved.contains(&(outsider as i32)),
        "red arm: the bias never reached the top-k"
    );
}
