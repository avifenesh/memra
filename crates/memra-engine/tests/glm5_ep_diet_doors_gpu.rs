//! glm5 EP DISPATCH DIET doors (lane/glm5-ep-diet, 2026-08-31) — rig exactness gates.
//!
//! Door `MEMRA_GLM5_EP_DIET`: the TP-2 EP walk's per-slot host round-trips fold into ONE
//! bulk peer return and the t*n_used sequential `axpy_f32` combine launches fold into ONE
//! `moe_pairs_scatter` launch over a compact-packed pair slab (root rows head, peer rows
//! tail) with a slot-ordered id table. The bar here is BIT identity of that combine against
//! the host fma chain it claims to reproduce, with reds proving the id table and the weight
//! placement are both load-bearing. The walk-level identity (decode byte, prime band, map
//! skew, reds) is `glm5-tp-gate`'s job — arms B2/B3/M2/R2D/R3D.
//!
//! Door `MEMRA_GLM5_EP_GROUPED_PRIME`: the plain grouped-prefill program split by expert
//! ownership. The rig gate for the NVFP4 numeric class: per-expert grouped-GEMM rows must
//! be BIT-identical between the plain (one CSR over all pairs) and split (two rank CSRs)
//! compositions — grouping is per expert, so splitting ranks must not move a row's bytes —
//! and the ONE reassociation (per-token root+peer partial add) must sit inside a CALIBRATED
//! band (measured green 1.360e-4 on raw cancellation-exposed layer outputs; band 1e-3),
//! with the dropped-peer-partial red landing orders louder.
//!
//! Every gate here is exactness or counters only — the rig cannot produce a timing row
//! (rig law); the box window prices both doors.

use memra_engine::Engine;

static GPU: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn gpu_guard() -> std::sync::MutexGuard<'static, ()> {
    GPU.lock().unwrap_or_else(|p| p.into_inner())
}

fn force_true_f32() {
    unsafe {
        std::env::set_var("NVIDIA_TF32_OVERRIDE", "0");
    }
}

fn varied(len: usize, seed: u64, spread: f32) -> Vec<f32> {
    let mut s = seed | 1;
    (0..len)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = ((s >> 33) as f32) / ((1u64 << 31) as f32);
            (u - 0.5) * spread
        })
        .collect()
}

fn bit_diffs(a: &[f32], b: &[f32]) -> usize {
    assert_eq!(a.len(), b.len(), "compared buffers differ in length");
    a.iter()
        .zip(b)
        .filter(|(x, y)| x.to_bits() != y.to_bits())
        .count()
}

fn max_rel(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(&x, &y)| {
            let d = (x as f64 - y as f64).abs();
            let s = (x as f64).abs().max((y as f64).abs()).max(1e-20);
            d / s
        })
        .fold(0.0, f64::max)
}

// -------------------------------------------------------------------------------------------
// Door EP_DIET — the compact-slab slot-ordered combine.
// -------------------------------------------------------------------------------------------

/// The dieted walk's combine: pack pair rows compact (root head, peer tail), build the
/// slab-position id table and slab-position weights, ONE `moe_pairs_scatter` launch. The
/// reference is the HOST fma chain in slot order — `f32::mul_add` is the same
/// round-once-per-term contract as the kernel's `__fmaf_rn`, and the v1 walk's
/// zeros + sequential `axpy_f32` chain compiles to exactly that FMA sequence (the A2
/// slot-scheme contract quoted in the kernel header). Bitwise, every (token, col).
#[test]
#[ignore = "needs a CUDA device, run under flock /tmp/memra-5090.lock"]
fn gpu_diet_compact_scatter_matches_the_host_fma_chain_bitwise() {
    let _g = gpu_guard();
    force_true_f32();
    let e = Engine::new(0).expect("engine");
    let n_embd = 257usize; // deliberately odd: a stride slip cannot hide on a round size
    for t in [1usize, 2, 5, 8] {
        for n_used in [1usize, 3, 8] {
            let n_pairs = t * n_used;
            // Ownership plant: pseudo-random peer/root mix, including all-root and
            // all-peer tokens at the larger shapes.
            let owner: Vec<u8> = (0..n_pairs)
                .map(|p| ((p * 7 + p / 3) % 3 == 1) as u8)
                .collect();
            let n_root = owner.iter().filter(|&&o| o == 0).count();
            let rows: Vec<Vec<f32>> = (0..n_pairs)
                .map(|p| varied(n_embd, 0xE9_D1E7 ^ (p as u64) << 9, 4.0))
                .collect();
            let w: Vec<f32> = varied(n_pairs, 0x5CA7_7E12, 1.5);

            // Compact packing + the slab-position tables, exactly the walk's arithmetic.
            let mut ids = vec![0i32; n_pairs];
            let (mut kr, mut kp) = (0usize, 0usize);
            for (p, id) in ids.iter_mut().enumerate() {
                if owner[p] == 0 {
                    *id = kr as i32;
                    kr += 1;
                } else {
                    *id = (n_root + kp) as i32;
                    kp += 1;
                }
            }
            let mut slab = vec![0f32; n_pairs * n_embd];
            let mut wd = vec![0f32; n_pairs];
            for p in 0..n_pairs {
                let pos = ids[p] as usize;
                slab[pos * n_embd..(pos + 1) * n_embd].copy_from_slice(&rows[p]);
                wd[pos] = w[p];
            }
            let y_all = e.htod(&slab).expect("slab");
            let pw = e.htod(&wd).expect("wd");
            let toff: Vec<i32> = (0..=t).map(|tok| (tok * n_used) as i32).collect();
            let toff_d = e.htod_i32(&toff).expect("toff");
            let ids_d = e.htod_i32(&ids).expect("ids");
            let mut out = e.uninit(t * n_embd).expect("out");
            e.moe_pairs_scatter(&y_all, &pw, &toff_d, &ids_d, &mut out, t, n_embd)
                .expect("scatter");
            let got = e.dtoh(&out).expect("dtoh");

            // Host reference: the v1 slot-ordered chain, one fma per slot, acc starts 0.0.
            let mut re = vec![0f32; t * n_embd];
            for tok in 0..t {
                for c in 0..n_embd {
                    let mut acc = 0.0f32;
                    for j in 0..n_used {
                        let p = tok * n_used + j;
                        acc = w[p].mul_add(rows[p][c], acc);
                    }
                    re[tok * n_embd + c] = acc;
                }
            }
            assert_eq!(
                bit_diffs(&re, &got),
                0,
                "compact scatter diverged from the host fma chain at t={t} n_used={n_used}"
            );

            if n_used < 2 {
                continue;
            }
            // RED 1: two slots of token 0 swapped in the ID TABLE — the chain ORDER moves,
            // so bytes must move (fma reassociation), or the id table is not load-bearing.
            let mut ids_red = ids.clone();
            ids_red.swap(0, 1);
            let ids_red_d = e.htod_i32(&ids_red).expect("ids red");
            let mut out_r = e.uninit(t * n_embd).expect("out red");
            e.moe_pairs_scatter(&y_all, &pw, &toff_d, &ids_red_d, &mut out_r, t, n_embd)
                .expect("scatter red");
            let got_r = e.dtoh(&out_r).expect("dtoh red");
            assert!(
                bit_diffs(&re, &got_r) > 0,
                "swapped-slot red did not move bytes at t={t} n_used={n_used} — the id \
                 table would be vacuous"
            );

            // RED 2: weights placed at PAIR positions instead of slab positions — the
            // weight placement is the other half of the compact-packing arithmetic.
            let pw_red = e.htod(&w).expect("wd red");
            let mut out_w = e.uninit(t * n_embd).expect("out wred");
            e.moe_pairs_scatter(&y_all, &pw_red, &toff_d, &ids_d, &mut out_w, t, n_embd)
                .expect("scatter wred");
            let got_w = e.dtoh(&out_w).expect("dtoh wred");
            assert!(
                bit_diffs(&re, &got_w) > 0,
                "pair-position weight red did not move bytes at t={t} n_used={n_used}"
            );
        }
    }
    println!("[ep-diet-doors] compact scatter == host fma chain, 12 shapes, both reds bite");
}

/// The bulk-return landing helper: `htod_f32_into_at` lands at the element offset and
/// refuses an out-of-range window (the slab-tail landing is the ONE new transfer helper the
/// diet adds, so its geometry is gated here rather than assumed).
#[test]
#[ignore = "needs a CUDA device, run under flock /tmp/memra-5090.lock"]
fn gpu_htod_f32_into_at_lands_at_offset_and_fails_closed() {
    let _g = gpu_guard();
    force_true_f32();
    let e = Engine::new(0).expect("engine");
    let base = varied(96, 0xBA5E, 2.0);
    let tail = varied(32, 0x7A11, 2.0);
    let mut dst = e.htod(&base).expect("dst");
    e.htod_f32_into_at(&tail, &mut dst, 64).expect("landing");
    let got = e.dtoh(&dst).expect("dtoh");
    assert_eq!(bit_diffs(&got[..64], &base[..64]), 0, "head disturbed");
    assert_eq!(bit_diffs(&got[64..], &tail), 0, "tail not landed");
    assert!(
        e.htod_f32_into_at(&tail, &mut dst, 65).is_err(),
        "out-of-range landing must refuse"
    );
}

// -------------------------------------------------------------------------------------------
// Door EP_GROUPED_PRIME — split-vs-plain grouped composition on minted NVFP4 slabs.
// -------------------------------------------------------------------------------------------

fn nvfp4_slab_padded(
    e: &Engine,
    n_expert: usize,
    out_f: usize,
    in_f: usize,
    pad: usize,
    seed: u64,
) -> (cudarc::driver::CudaSlice<u8>, usize, usize) {
    let mut bytes = Vec::new();
    let mut row_bytes = 0usize;
    for ex in 0..n_expert {
        for o in 0..out_f {
            let row = varied(in_f, seed ^ ((ex * out_f + o) as u64) << 8, 2.0);
            let rb = memra_gguf::nvfp4_repack::f32_to_nvfp4(&row);
            row_bytes = rb.len();
            bytes.extend_from_slice(&rb);
        }
    }
    let stride = out_f * row_bytes;
    // The grouped GEMM's ragged-k visitor may overread past the LAST row — the same
    // 144-byte tail slack the resident-slab builder and the EP rank slabs carry.
    let slab = e.htod_bytes_padded(&bytes, pad).expect("nvfp4 slab upload");
    (slab, row_bytes, stride)
}

/// One grouped-pairs program over a pair subset — the composition
/// `moe_ffn_glm5_ep_grouped_prime`'s rank pass makes, driven through the same public
/// Engine calls: CSR -> f16 act gather -> grouped gate/up -> pre-clamped SwiGLU ->
/// grouped down -> CSR->local permute -> slot-ordered scatter. `take` filters the pairs
/// (all = the plain arm; an ownership filter = one rank of the split arm).
#[allow(clippy::too_many_arguments)]
fn grouped_pairs_program(
    e: &Engine,
    tables: &cudarc::driver::CudaSlice<u64>,
    n_expert: usize,
    sel: &[u32],
    w: &[f32],
    z: &cudarc::driver::CudaSlice<f32>,
    t: usize,
    n_used: usize,
    (n_embd, n_ff): (usize, usize),
    (rb_gu, rb_d): (usize, usize),
    limit: f32,
    take: &dyn Fn(usize) -> bool,
) -> (Vec<f32>, Vec<f32>, Vec<i32>) {
    let n_pairs = t * n_used;
    let mut buckets: Vec<Vec<i32>> = vec![Vec::new(); n_expert];
    let mut local_tok = Vec::new();
    let mut local_wd = Vec::new();
    let mut local_pair = Vec::new(); // global pair id of local row l (for cross-arm row compare)
    let mut per_tok = vec![0i32; t];
    for p in 0..n_pairs {
        let ex = sel[p] as usize;
        if !take(ex) {
            continue;
        }
        let l = local_tok.len() as i32;
        buckets[ex].push(l);
        local_tok.push((p / n_used) as i32);
        local_wd.push(w[p]);
        local_pair.push(p as i32);
        per_tok[p / n_used] += 1;
    }
    let n_owned = local_tok.len();
    assert!(n_owned > 0, "planted routing must touch this arm");
    let mut ex_ids: Vec<i32> = Vec::new();
    let mut ex_off: Vec<i32> = vec![0];
    let mut ex_pairs: Vec<i32> = Vec::new();
    let mut csr_tok: Vec<i32> = Vec::new();
    for (e_id, b) in buckets.iter().enumerate() {
        if !b.is_empty() {
            ex_ids.push(e_id as i32);
            for &l in b {
                ex_pairs.push(l);
                csr_tok.push(local_tok[l as usize]);
            }
            ex_off.push(ex_pairs.len() as i32);
        }
    }
    let n_active = ex_ids.len();
    let exi = e.htod_i32(&ex_ids).expect("exi");
    let exo = e.htod_i32(&ex_off).expect("exo");
    let exp_d = e.htod_i32(&ex_pairs).expect("exp");
    let csr_tok_d = e.htod_i32(&csr_tok).expect("csr tok");
    let (z16, zs) = e
        .moe_f16g_act(z, Some(&csr_tok_d), n_embd, n_owned)
        .expect("act gather");
    let g = e
        .moe_f16_grouped(
            tables,
            0,
            n_expert,
            &exi,
            &ex_off,
            &exo,
            &z16,
            &zs,
            n_embd,
            n_ff,
            n_active,
            n_owned,
            memra_engine::QT_NVFP4,
            rb_gu,
        )
        .expect("gate gemm");
    let u = e
        .moe_f16_grouped(
            tables,
            1,
            n_expert,
            &exi,
            &ex_off,
            &exo,
            &z16,
            &zs,
            n_embd,
            n_ff,
            n_active,
            n_owned,
            memra_engine::QT_NVFP4,
            rb_gu,
        )
        .expect("up gemm");
    let mut act = e.uninit(n_owned * n_ff).expect("act");
    e.swiglu_preclamped_mul_scaled(&g, &u, 1.0, 1.0, limit, &mut act, n_owned * n_ff)
        .expect("epilogue");
    let (a16, a_s) = e.moe_f16g_act(&act, None, n_ff, n_owned).expect("act f16");
    let d_csr = e
        .moe_f16_grouped(
            tables,
            2,
            n_expert,
            &exi,
            &ex_off,
            &exo,
            &a16,
            &a_s,
            n_ff,
            n_embd,
            n_active,
            n_owned,
            memra_engine::QT_NVFP4,
            rb_d,
        )
        .expect("down gemm");
    let y_local = e
        .rows_permute(&d_csr, &exp_d, n_owned, n_embd)
        .expect("permute");
    let mut toff: Vec<i32> = Vec::with_capacity(t + 1);
    let mut acc = 0i32;
    toff.push(0);
    for &c in &per_tok {
        acc += c;
        toff.push(acc);
    }
    let tids: Vec<i32> = (0..n_owned as i32).collect();
    let pw = e.htod(&local_wd).expect("pw");
    let toff_d = e.htod_i32(&toff).expect("toff");
    let tids_d = e.htod_i32(&tids).expect("tids");
    let mut partial = e.uninit(t * n_embd).expect("partial");
    e.moe_pairs_scatter(&y_local, &pw, &toff_d, &tids_d, &mut partial, t, n_embd)
        .expect("scatter");
    (
        e.dtoh(&partial).expect("partial dtoh"),
        e.dtoh(&y_local).expect("rows dtoh"),
        local_pair,
    )
}

/// Split-vs-plain grouped prime on minted NVFP4: per-pair down rows BIT-identical between
/// the one-CSR plain composition and the two-rank split (an expert's grouped GEMM cannot
/// depend on which OTHER experts share its launch); the per-token partial add sits inside
/// the chunked-prime band; the dropped-peer-partial red lands orders louder.
#[test]
#[ignore = "needs a CUDA device, run under flock /tmp/memra-5090.lock"]
fn gpu_grouped_prime_split_matches_plain_rows_bitwise_and_combine_in_band() {
    let _g = gpu_guard();
    force_true_f32();
    let e = Engine::new(0).expect("engine");
    let (n_expert, n_embd, n_ff) = (8usize, 128usize, 192usize); // NVFP4: 64-value blocks
    let (t, n_used) = (6usize, 4usize);
    let limit = 7.0f32; // a live PRE clamp (the vrest gate-4 shape class)
    let (gate, rb_gu, s_gu) = nvfp4_slab_padded(&e, n_expert, n_ff, n_embd, 8, 0x6A7E);
    let (up, rb_gu2, _s_u) = nvfp4_slab_padded(&e, n_expert, n_ff, n_embd, 8, 0x0B57);
    let (down, rb_d, s_d) = nvfp4_slab_padded(&e, n_expert, n_embd, n_ff, 144, 0xD003);
    assert_eq!(rb_gu, rb_gu2);

    // Pointer tables: full (plain), rank 0 (experts 0..4), rank 1 (experts 4..8) — the
    // arm_moe_ep shape: global-expert-id indexing, zeros for non-owned entries.
    use cudarc::driver::DevicePtr;
    let (pg, pu, pd) = {
        let s = e.stream();
        let (pg, _a) = gate.device_ptr(&s);
        let (pu, _b) = up.device_ptr(&s);
        let (pd, _c) = down.device_ptr(&s);
        (pg, pu, pd)
    };
    let mk_table = |owned: &dyn Fn(usize) -> bool| -> cudarc::driver::CudaSlice<u64> {
        let mut host = vec![0u64; 3 * n_expert];
        for ex in 0..n_expert {
            if !owned(ex) {
                continue;
            }
            host[ex] = pg + (ex * s_gu) as u64;
            host[n_expert + ex] = pu + (ex * s_gu) as u64;
            host[2 * n_expert + ex] = pd + (ex * s_d) as u64;
        }
        e.htod_u64(&host).expect("table")
    };
    let tab_full = mk_table(&|_| true);
    let tab_r0 = mk_table(&|ex| ex < 4);
    let tab_r1 = mk_table(&|ex| ex >= 4);

    // Planted routing: every token mixes both ranks, repeated experts included, expert 0
    // exercised (the base + 0*stride term).
    let sel: Vec<u32> = (0..t * n_used)
        .map(|p| ((p * 5 + p / n_used) % n_expert) as u32)
        .collect();
    let w: Vec<f32> = varied(t * n_used, 0x1D1E7, 1.2);
    let z = e.htod(&varied(t * n_embd, 0x2E_57A6E, 3.0)).expect("z");

    let dims = (n_embd, n_ff);
    let rbs = (rb_gu, rb_d);
    let (plain, plain_rows, plain_pairs) = grouped_pairs_program(
        &e,
        &tab_full,
        n_expert,
        &sel,
        &w,
        &z,
        t,
        n_used,
        dims,
        rbs,
        limit,
        &|_| true,
    );
    let (part0, rows0, pairs0) = grouped_pairs_program(
        &e,
        &tab_r0,
        n_expert,
        &sel,
        &w,
        &z,
        t,
        n_used,
        dims,
        rbs,
        limit,
        &|ex| ex < 4,
    );
    let (part1, rows1, pairs1) = grouped_pairs_program(
        &e,
        &tab_r1,
        n_expert,
        &sel,
        &w,
        &z,
        t,
        n_used,
        dims,
        rbs,
        limit,
        &|ex| ex >= 4,
    );

    // Gate A — per-pair down rows bit-identical plain vs split: the grouped GEMM's row
    // bytes must not depend on launch-set composition, or the split is a new kernel class
    // and the band below would be measuring the wrong thing.
    let row_of = |pairs: &[i32], rows: &[f32], p: i32| -> Option<Vec<f32>> {
        pairs
            .iter()
            .position(|&q| q == p)
            .map(|l| rows[l * n_embd..(l + 1) * n_embd].to_vec())
    };
    let mut rows_checked = 0usize;
    for &p in &plain_pairs {
        let a = row_of(&plain_pairs, &plain_rows, p).expect("plain row");
        let b = row_of(&pairs0, &rows0, p)
            .or_else(|| row_of(&pairs1, &rows1, p))
            .expect("split row");
        assert_eq!(
            bit_diffs(&a, &b),
            0,
            "pair {p}: split grouped-GEMM row differs from the plain composition"
        );
        rows_checked += 1;
    }
    assert_eq!(rows_checked, t * n_used);

    // Gate B — the ONE reassociation: out = part0 + part1 (fma alpha=1.0, the walk's
    // combine) vs the plain single chain. CALIBRATED BAND, per the calibration law
    // (measure first, then set the bar from the measurement): the first run on this rig
    // measured green max_rel = 1.360e-4 — RAW layer outputs expose cancellation (a
    // token's 4-term sum can land near zero while its terms are O(1), so a 1-ulp
    // reassociation shift reads as ~1e-4 RELATIVE; the tp-gate's 2e-4 logit band sits on
    // network-smoothed values and does not transfer to this surface). Band = 1e-3
    // (7.4x margin over the measured green); the dropped-peer red below must land
    // ORDERS above it, which is what keeps the band honest.
    let combined: Vec<f32> = part0
        .iter()
        .zip(&part1)
        .map(|(&a, &b)| 1.0f32.mul_add(b, a))
        .collect();
    let rel = max_rel(&plain, &combined);
    assert!(
        rel <= 1e-3,
        "split partial-add reassociation left the calibrated band: max_rel={rel:.3e} \
         (green measured 1.360e-4; band 1e-3)"
    );

    // RED — the dropped peer partial (the skip-peer-combine class) must land orders
    // above the band, or the band gate could pass on a combine that never adds the peer.
    let red_rel = max_rel(&plain, &part0);
    assert!(
        red_rel > 1e-1,
        "dropped-peer-partial red is not loud: max_rel={red_rel:.3e} (band 1e-3)"
    );
    println!(
        "[ep-diet-doors] grouped prime split: {rows_checked} rows bitwise, combine \
         max_rel={rel:.3e} (calibrated band 1e-3, green measured 1.360e-4), \
         dropped-peer red {red_rel:.3e}"
    );
}
