//! Gate for `MEMRA_MLA_TC_PREFILL` — the glm5_next tensor-core MLA prefill chain
//! (lane/glm5-mla-tc-prefill, 2026-08-30).
//!
//! WHAT THE DOOR REPLACES (launch-diet census, WINDOW-20260830.md §4): at a cold 4626-token
//! prime, `memra_mla_attn_gathered_kernel` (139.1 ms) plus `memra_mla_absorb_q_kernel`
//! (44.5 ms) plus `memra_mla_decompress_v_kernel` (43.6 ms) per layer-chunk = 75.8% of a
//! 98%-GPU-busy prefill, running 1-2 orders below tensor-core class. The ON arm runs
//! absorb/decompress as strided-batched bf16 TC GEMMs and the attention as
//! `fa_mla_gathered_bf16` (bf16 MMA, f32 accumulate, online softmax) over the UNCHANGED DSA
//! selection lists.
//!
//! NUMERIC CLASS: bf16 operands, f32 accumulate — a maxdiff band vs both the CPU oracle and
//! the shipped f32 kernels, never bit identity (the reduction order and the operand precision
//! both change). The band is CALIBRATED here and PINNED; the FLAGS.md row quotes it.
//!
//! GATES:
//!   1. `gpu_mla_tc_matches_cpu_oracle_and_f32_kernels_small` — t in {16, 64}, fresh and
//!      chunked shapes, trivial-selection (identity gather) and sparse-selection lists, at
//!      full GLM5_NEXT geometry, vs a CPU gathered oracle AND vs the shipped f32 chain.
//!   2. `gpu_mla_tc_on_vs_off_at_t4096` — the census shape class (t=4096, real 2048/4 budget,
//!      width 2051) vs the shipped f32 chain, plus CPU-oracle spot checks on two queries
//!      (one trivial, one budget-limited).
//!   3. `gpu_mla_tc_red_mutations_fail_the_comparator` — four reds, each run/finite/silently
//!      wrong, each caught by the SAME comparator that passes the unmutated arm:
//!      a. W_uk/W_uv swapped (identical element counts at this geometry — loads and runs);
//!      b. per-head TRANSPOSED W_uk (corrupts the Q@K GEMM's Q operand);
//!      c. selection mask dropped (full causal lists where the DSA program selected);
//!      d. causal off by one (one future row appended to every list that has one).
//!   4. `gpu_mla_tc_mixer_matches_reference_mini512` — the established kpool mini fixture
//!      family with kv_lora_rank raised to the kernel's 512 stamp: ON logits vs
//!      `memra_reference::execute` at t in {16, 64} (sparse regime asserted), OFF logits
//!      unchanged, and the engagement counter anchored at the invocation.
//!   5. `gpu_mla_tc_mixer_matches_reference_t4096` — the same mixer bar at t=4096.
//!   6. `gpu_mla_tc_decode_stays_byte_identical` — prime OFF into two identical caches, then
//!      decode with the flag ON vs OFF: logits BIT-identical at every step and the dispatch
//!      counter FLAT — the decode path is untouched, gated not assumed.
//!
//! Rig law (exactness only, never timing):
//!   NVIDIA_TF32_OVERRIDE=0 flock /tmp/memra-5090.lock \
//!     cargo test -p memra-engine --test mla_tc_prefill_gpu -- --ignored --test-threads=1

use cudarc::driver::CudaSlice;
use memra_engine::Engine;
use memra_engine::mla::MlaDims;
use memra_gguf::micro_gguf::Rng;

/// Process-wide GPU serialization (mla_gpu_forward pattern).
fn gpu_guard() -> std::sync::MutexGuard<'static, ()> {
    static GPU: std::sync::Mutex<()> = std::sync::Mutex::new(());
    GPU.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// cuBLASLt f32 compute rides TF32 on Blackwell by default — right for serving, wrong for a
/// parity gate. Must run before the first `Engine::new` in the process.
fn force_true_f32() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        if std::env::var("NVIDIA_TF32_OVERRIDE").as_deref() != Ok("0") {
            // SAFETY: no CUDA call has been made yet; call_once serializes the write.
            unsafe { std::env::set_var("NVIDIA_TF32_OVERRIDE", "0") };
        }
    });
}

fn maxabs(a: &[f32]) -> f32 {
    a.iter().map(|x| x.abs()).fold(0.0f32, f32::max)
}

fn relative(got: &[f32], want: &[f32]) -> f32 {
    assert_eq!(got.len(), want.len());
    let scale = maxabs(want).max(1e-20);
    got.iter()
        .zip(want)
        .map(|(g, w)| (g - w).abs())
        .fold(0.0f32, f32::max)
        / scale
}

// ---------------------------------------------------------------- kernel-level fixture
// The mla_gpu_forward `Case` family, restricted to the NoPE geometry this door serves.

struct Case {
    q_nope: Vec<f32>,
    c_kv: Vec<f32>,
    w_uk: Vec<f32>, // mla.rs layout [h][p][l]
    w_uv: Vec<f32>, // mla.rs layout [h][j][l] == checkpoint attn_v_b
}

impl Case {
    fn new(d: &MlaDims, t_q: usize, t_kv: usize, seed: u64) -> Self {
        assert_eq!(d.d_rope, 0, "this gate covers the NoPE door only");
        let mut rng = Rng(seed | 1);
        let ws = 1.0 / (d.kv_rank as f32).sqrt();
        Case {
            q_nope: rng.fill(t_q * d.n_head * d.d_nope, 1.0),
            c_kv: rng.fill(t_kv * d.kv_rank, 1.0),
            w_uk: rng.fill(d.n_head * d.d_nope * d.kv_rank, ws),
            w_uv: rng.fill(d.n_head * d.d_v * d.kv_rank, ws),
        }
    }

    /// `attn_k_b` in the CHECKPOINT layout, ne {d_nope, kv_rank, n_head}: element (h, l, p) at
    /// `h*kv_rank*d_nope + l*d_nope + p` (mla_gpu_forward's transpose, verbatim).
    fn wk_b(&self, d: &MlaDims) -> Vec<f32> {
        let (nh, dn, r) = (d.n_head, d.d_nope, d.kv_rank);
        let mut w = vec![0.0f32; nh * r * dn];
        for h in 0..nh {
            for l in 0..r {
                for p in 0..dn {
                    w[h * r * dn + l * dn + p] = self.w_uk[h * dn * r + p * r + l];
                }
            }
        }
        w
    }
}

/// Deterministic pseudo-random index lists with the DSA selector's structural contract:
/// per query (absolute position `first_pos + i`), ascending, causal, pool-granular selection
/// capped at `topk/pool` pools plus the raw incomplete tail, -1 TRAILING pad to `width`.
/// Trivial-selection queries (visible pools <= budget) get the FULL causal prefix — exactly
/// what `memra_mla_kpool_select_*` emits for them (budget >= candidates selects everything).
fn synth_lists(
    t_q: usize,
    first_pos: usize,
    t_kv: usize,
    topk: usize,
    pool: usize,
    seed: u64,
) -> (Vec<Vec<usize>>, usize) {
    let budget = topk / pool;
    let n_pools_total = t_kv / pool;
    let width = budget.min(n_pools_total) * pool + pool - 1;
    let mut s = seed | 1;
    let mut lists = Vec::with_capacity(t_q);
    for i in 0..t_q {
        let visible = first_pos + i + 1;
        assert!(visible <= t_kv, "query {i} sees past the cache");
        let n_pools = visible / pool;
        let mut rows = Vec::new();
        if n_pools <= budget {
            rows.extend(0..n_pools * pool);
        } else {
            // pick `budget` distinct pools, ascending (Fisher-Yates prefix on a pool list).
            let mut pools: Vec<usize> = (0..n_pools).collect();
            for j in 0..budget {
                s = s
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                let k = j + ((s >> 33) as usize) % (n_pools - j);
                pools.swap(j, k);
            }
            let mut sel = pools[..budget].to_vec();
            sel.sort_unstable();
            for p in sel {
                rows.extend(p * pool..(p + 1) * pool);
            }
        }
        // raw incomplete tail (always_select_tail).
        rows.extend(n_pools * pool..visible);
        assert!(
            rows.len() <= width,
            "query {i}: {} rows > width {width}",
            rows.len()
        );
        lists.push(rows);
    }
    (lists, width)
}

fn lists_to_device(e: &Engine, lists: &[Vec<usize>], width: usize) -> CudaSlice<i32> {
    let mut flat = vec![-1i32; lists.len() * width];
    for (i, rows) in lists.iter().enumerate() {
        for (slot, &r) in rows.iter().enumerate() {
            flat[i * width + slot] = r as i32;
        }
    }
    e.htod_i32(&flat).expect("index lists to device")
}

/// CPU f32 gathered-attention oracle over explicit lists: absorb -> softmax over the listed
/// rows -> decompress. Single left-to-right pass, plain f32 — the mla.rs class.
fn cpu_gathered(d: &MlaDims, c: &Case, lists: &[Vec<usize>]) -> Vec<f32> {
    let (nh, dn, dv, r) = (d.n_head, d.d_nope, d.d_v, d.kv_rank);
    let scale = d.scale();
    let t = lists.len();
    let mut out = vec![0.0f32; t * nh * dv];
    for i in 0..t {
        for h in 0..nh {
            let qn = &c.q_nope[(i * nh + h) * dn..][..dn];
            let wuk = &c.w_uk[h * dn * r..][..dn * r];
            let mut q_lat = vec![0.0f32; r];
            for (p, &qp) in qn.iter().enumerate() {
                let row = &wuk[p * r..][..r];
                for l in 0..r {
                    q_lat[l] += qp * row[l];
                }
            }
            let list = &lists[i];
            let mut scores: Vec<f32> = list
                .iter()
                .map(|&srow| {
                    let row = &c.c_kv[srow * r..][..r];
                    q_lat.iter().zip(row).map(|(a, b)| a * b).sum::<f32>() * scale
                })
                .collect();
            let m = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut dsum = 0.0f32;
            for sc in scores.iter_mut() {
                *sc = (*sc - m).exp();
                dsum += *sc;
            }
            let mut o_lat = vec![0.0f32; r];
            for (w, &srow) in scores.iter().zip(list) {
                let row = &c.c_kv[srow * r..][..r];
                let wn = w / dsum;
                for l in 0..r {
                    o_lat[l] += wn * row[l];
                }
            }
            let wuv = &c.w_uv[h * dv * r..][..dv * r];
            for j in 0..dv {
                out[(i * nh + h) * dv + j] = wuv[j * r..][..r]
                    .iter()
                    .zip(&o_lat)
                    .map(|(a, b)| a * b)
                    .sum::<f32>();
            }
        }
    }
    out
}

/// Uploaded device operands for one case: everything both chains consume.
struct DeviceCase {
    q_nope: CudaSlice<f32>,
    cache: CudaSlice<f32>,
    wk_b: CudaSlice<f32>, // checkpoint (h, l, p)
    wv_b: CudaSlice<f32>, // checkpoint (h, j, l)
}

fn upload(e: &Engine, d: &MlaDims, c: &Case) -> DeviceCase {
    DeviceCase {
        q_nope: e.htod(&c.q_nope).expect("q_nope"),
        cache: e.htod(&c.c_kv).expect("latent cache"),
        wk_b: e.htod(&c.wk_b(d)).expect("wk_b"),
        wv_b: e.htod(&c.w_uv).expect("wv_b"),
    }
}

/// The shipped f32 chain (absorb_q -> gathered attention -> decompress_v), the OFF arm.
fn f32_chain(
    e: &Engine,
    d: &MlaDims,
    dc: &DeviceCase,
    idx: &CudaSlice<i32>,
    t: usize,
    width: usize,
) -> Vec<f32> {
    let (nh, dn, dv, r) = (d.n_head, d.d_nope, d.d_v, d.kv_rank);
    let mut q_lat = e.uninit(t * nh * r).unwrap();
    e.mla_absorb_q(&dc.q_nope, &dc.wk_b, &mut q_lat, t, nh, dn, r)
        .expect("absorb_q");
    let q_pe = e.htod(&[0.0f32]).unwrap(); // NoPE placeholder (never dereferenced at d_rope 0)
    let mut o_lat = e.uninit(t * nh * r).unwrap();
    e.mla_attn_gathered(
        &q_lat,
        &q_pe,
        &dc.cache,
        idx,
        &mut o_lat,
        nh,
        r,
        0,
        t,
        width,
        d.scale(),
    )
    .expect("gathered attention");
    let mut out = e.uninit(t * nh * dv).unwrap();
    e.mla_decompress_v(&o_lat, &dc.wv_b, &mut out, t, nh, dv, r)
        .expect("decompress_v");
    e.stream().clone_dtoh(&out).expect("readback")
}

/// The TC chain, mirroring `mla_tc_prefill_chain` operation for operation (the door itself is
/// gated at the mixer level below; this entry gives the red arms operand-level control).
/// `wk_override`/`wv_override` substitute mutated weight planes.
#[allow(clippy::too_many_arguments)]
fn tc_chain(
    e: &Engine,
    d: &MlaDims,
    dc: &DeviceCase,
    idx: &CudaSlice<i32>,
    t: usize,
    t_kv: usize,
    width: usize,
    wk_override: Option<&CudaSlice<f32>>,
    wv_override: Option<&CudaSlice<f32>>,
) -> Vec<f32> {
    let (nh, dn, dv, r) = (d.n_head, d.d_nope, d.d_v, d.kv_rank);
    let wk = wk_override.unwrap_or(&dc.wk_b);
    let wv = wv_override.unwrap_or(&dc.wv_b);
    let wk_bf = e.f32_to_bf16(wk, nh * r * dn).unwrap();
    let wv_bf = e.f32_to_bf16(wv, nh * dv * r).unwrap();
    let qn_bf = e.f32_to_bf16(&dc.q_nope, t * nh * dn).unwrap();
    let mut q_lat_bf = e.alloc_u8_uninit(t * nh * r * 2).unwrap();
    assert!(
        e.mla_bf16_gemm_sb_bf16out(
            &wk_bf,
            &qn_bf,
            &mut q_lat_bf,
            t,
            r,
            dn,
            nh * dn,
            dn,
            nh * r,
            r,
            nh,
        )
        .expect("absorb sb-GEMM"),
        "cuBLASLt declined the absorb shape (m={t} n={r} k={dn} batch={nh}) on this device"
    );
    let cache_bf = e.f32_to_bf16(&dc.cache, t_kv * r).unwrap();
    let mut o_lat = e.uninit(t * nh * r).unwrap();
    e.mla_attn_gathered_tc(
        &q_lat_bf,
        &cache_bf,
        idx,
        &mut o_lat,
        nh,
        r,
        t,
        width,
        d.scale(),
    )
    .expect("TC gathered attention");
    let o_bf = e.f32_to_bf16(&o_lat, t * nh * r).unwrap();
    let mut out = e.uninit(t * nh * dv).unwrap();
    assert!(
        e.mla_bf16_gemm_sb_f32out(
            &wv_bf,
            &o_bf,
            &mut out,
            t,
            dv,
            r,
            nh * r,
            r,
            nh * dv,
            dv,
            nh
        )
        .expect("decompress sb-GEMM"),
        "cuBLASLt declined the decompress shape (m={t} n={dv} k={r} batch={nh})"
    );
    e.stream().clone_dtoh(&out).expect("readback")
}

/// PINNED BANDS (calibrated on the rig 5090, TF32 off, debug build — the run-green receipt in
/// the lane doc quotes the measured worst cases these sit above with headroom):
///   * TC vs CPU oracle / TC vs the f32 chain: the bf16-operand class. Operands round to 8
///     mantissa bits before 512-wide f32-accumulated dots, so the honest bar is the "bf16
///     8e-3" class named in the lane brief.
///   * f32 chain vs CPU oracle: the established f32 reorder class (mla_gpu_forward's 1e-4).
const TC_BAND: f32 = 8e-3;
const F32_BAND: f32 = 1e-4;

/// GATE 1 — small shapes, full GLM5_NEXT geometry, vs BOTH the CPU oracle and the f32 chain.
#[test]
#[ignore = "needs a CUDA device — run under flock /tmp/memra-5090.lock"]
fn gpu_mla_tc_matches_cpu_oracle_and_f32_kernels_small() {
    let _gpu = gpu_guard();
    force_true_f32();
    let e = Engine::new(0).expect("CUDA device 0");
    let d = MlaDims::GLM5_NEXT;

    // (t, first_pos, topk, label): trivial-selection (identity gather), sparse-selection, and
    // the chunked shape (queries a suffix of a longer cache — slot > 0 semantics).
    let shapes = [
        (16usize, 0usize, 2048usize, "t16 fresh, trivial selection"),
        (16, 48, 32, "t16 chunked, sparse selection"),
        (64, 0, 32, "t64 fresh, sparse past query 35"),
        (64, 64, 32, "t64 chunked, all queries sparse"),
    ];
    let mut worst_cpu = 0.0f32;
    let mut worst_f32 = 0.0f32;
    for (t, first_pos, topk, label) in shapes {
        let t_kv = first_pos + t;
        let c = Case::new(&d, t, t_kv, 0x91A_7C00 + t as u64 + first_pos as u64);
        let (lists, width) = synth_lists(t, first_pos, t_kv, topk, 4, 0x5EED + t as u64);
        let sparse = lists
            .iter()
            .enumerate()
            .filter(|(i, l)| l.len() < first_pos + i + 1)
            .count();
        let dc = upload(&e, &d, &c);
        let idx = lists_to_device(&e, &lists, width);
        let want = cpu_gathered(&d, &c, &lists);
        let old = f32_chain(&e, &d, &dc, &idx, t, width);
        let tc = tc_chain(&e, &d, &dc, &idx, t, t_kv, width, None, None);
        assert!(
            tc.iter().all(|v| v.is_finite()),
            "{label}: non-finite TC output"
        );
        assert!(maxabs(&tc) > 1e-6, "{label}: degenerate TC output");
        let rel_old = relative(&old, &want);
        let rel_cpu = relative(&tc, &want);
        let rel_f32 = relative(&tc, &old);
        println!(
            "{label}: width {width}, {sparse}/{t} queries budget-limited; \
             f32-vs-cpu {rel_old:.3e} (band {F32_BAND:.0e}), tc-vs-cpu {rel_cpu:.3e}, \
             tc-vs-f32 {rel_f32:.3e} (band {TC_BAND:.0e})"
        );
        assert!(
            rel_old <= F32_BAND,
            "{label}: f32 chain drifted from the CPU oracle"
        );
        assert!(
            rel_cpu <= TC_BAND,
            "{label}: TC chain vs CPU oracle {rel_cpu:.3e} > {TC_BAND:.0e}"
        );
        assert!(
            rel_f32 <= TC_BAND,
            "{label}: TC chain vs f32 chain {rel_f32:.3e} > {TC_BAND:.0e}"
        );
        worst_cpu = worst_cpu.max(rel_cpu);
        worst_f32 = worst_f32.max(rel_f32);
    }
    println!("worst tc-vs-cpu {worst_cpu:.3e}, worst tc-vs-f32 {worst_f32:.3e}");
}

/// GATE 2 — the census shape class: t=4096 at the REAL budget (topk 2048, pool 4, width 2051):
/// queries 0..=2050 are trivial (identity gather), 2051..4095 budget-limited. The f32 chain is
/// the reference-anchored arm at this size (mla_gpu_forward + the kpool gates pin it); the CPU
/// oracle spot-checks two queries directly (a full-batch CPU pass at this geometry is hours of
/// debug-build scalar arithmetic and would gate nothing more).
#[test]
#[ignore = "needs a CUDA device — run under flock /tmp/memra-5090.lock"]
fn gpu_mla_tc_on_vs_off_at_t4096() {
    let _gpu = gpu_guard();
    force_true_f32();
    let e = Engine::new(0).expect("CUDA device 0");
    let d = MlaDims::GLM5_NEXT;
    let (t, topk, pool) = (4096usize, 2048usize, 4usize);
    let c = Case::new(&d, t, t, 0x4096_2026);
    let (lists, width) = synth_lists(t, 0, t, topk, pool, 0xB1E55ED);
    assert_eq!(width, 2051, "the real glm5_next index width");
    let dc = upload(&e, &d, &c);
    let idx = lists_to_device(&e, &lists, width);
    let old = f32_chain(&e, &d, &dc, &idx, t, width);
    let tc = tc_chain(&e, &d, &dc, &idx, t, t, width, None, None);
    assert!(
        tc.iter().all(|v| v.is_finite()),
        "non-finite TC output at t=4096"
    );
    let rel = relative(&tc, &old);
    println!("t=4096 width={width}: tc-vs-f32 {rel:.3e} (band {TC_BAND:.0e})");
    assert!(
        rel <= TC_BAND,
        "TC vs f32 chain at t=4096: {rel:.3e} > {TC_BAND:.0e}"
    );

    // CPU spot checks: one trivial query, one deep-sparse query.
    let row = d.n_head * d.d_v;
    for qi in [1024usize, 4095] {
        let want = cpu_gathered(&d, &c_query(&c, &d, qi), &[lists[qi].clone()]);
        let got = &tc[qi * row..(qi + 1) * row];
        let rel = relative(got, &want);
        println!(
            "t=4096 query {qi} ({} rows attended): tc-vs-cpu {rel:.3e}",
            lists[qi].len()
        );
        assert!(rel <= TC_BAND, "query {qi}: {rel:.3e} > {TC_BAND:.0e}");
    }
}

/// A single-query view of a Case (the CPU spot-check helper): q_nope sliced to one query,
/// cache and weights shared.
fn c_query(c: &Case, d: &MlaDims, qi: usize) -> Case {
    let row = d.n_head * d.d_nope;
    Case {
        q_nope: c.q_nope[qi * row..(qi + 1) * row].to_vec(),
        c_kv: c.c_kv.clone(),
        w_uk: c.w_uk.clone(),
        w_uv: c.w_uv.clone(),
    }
}

/// GATE 3 — RED MUTATIONS. Each one loads, runs, and stays finite; the comparator that passes
/// the unmutated arm must FAIL each of them, or this gate could not have caught the same bug
/// in the shipped chain.
#[test]
#[ignore = "needs a CUDA device — run under flock /tmp/memra-5090.lock"]
fn gpu_mla_tc_red_mutations_fail_the_comparator() {
    let _gpu = gpu_guard();
    force_true_f32();
    let e = Engine::new(0).expect("CUDA device 0");
    let d = MlaDims::GLM5_NEXT;
    let (t, first_pos, topk, pool) = (64usize, 64usize, 32usize, 4usize);
    let t_kv = first_pos + t;
    let c = Case::new(&d, t, t_kv, 0xDEAD_2026);
    let (lists, width) = synth_lists(t, first_pos, t_kv, topk, pool, 0x0FF5E7);
    let sparse = lists
        .iter()
        .enumerate()
        .filter(|(i, l)| l.len() < first_pos + i + 1)
        .count();
    assert!(
        sparse == t,
        "red fixture must sit fully in the sparse regime ({sparse}/{t}) or the dropped-mask \
         red is vacuous"
    );
    let dc = upload(&e, &d, &c);
    let idx = lists_to_device(&e, &lists, width);
    let want = cpu_gathered(&d, &c, &lists);

    // The comparator that must do the catching, green first (isolation).
    let green = tc_chain(&e, &d, &dc, &idx, t, t_kv, width, None, None);
    let rel_green = relative(&green, &want);
    println!("green arm: {rel_green:.3e} (band {TC_BAND:.0e})");
    assert!(
        rel_green <= TC_BAND,
        "the unmutated arm must pass before reds mean anything"
    );

    // RED a: W_uk/W_uv swapped. At this geometry both planes are nh*512*256 elements, so the
    // swap loads and runs — the classic silent-checkpoint-mixup shape.
    let red_a = tc_chain(
        &e,
        &d,
        &dc,
        &idx,
        t,
        t_kv,
        width,
        Some(&dc.wv_b),
        Some(&dc.wk_b),
    );
    let rel_a = relative(&red_a, &want);
    println!("red a (w_uk/w_uv swapped): {rel_a:.3e}");
    assert!(
        red_a.iter().all(|v| v.is_finite()),
        "red a must run and stay finite"
    );
    assert!(
        rel_a > TC_BAND,
        "swapped up-projections passed the gate ({rel_a:.3e})"
    );

    // RED b: per-head TRANSPOSED W_uk — the Q@K GEMM's Q operand is built from W_uk^T instead
    // of W_uk (the kda_fused_proj transposed-slice recipe).
    let wk = c.wk_b(&d);
    let (nh, dn, r) = (d.n_head, d.d_nope, d.kv_rank);
    let mut wk_t = vec![0.0f32; wk.len()];
    for h in 0..nh {
        let src = &wk[h * r * dn..][..r * dn]; // [r, dn] row-major
        let dst = &mut wk_t[h * r * dn..][..r * dn];
        for l in 0..r {
            for p in 0..dn {
                // transposed DATA reinterpreted in the same [r, dn] frame
                dst[l * dn + p] = src[(p % r) * dn + (l % dn)];
            }
        }
    }
    let wk_t_d = e.htod(&wk_t).unwrap();
    let red_b = tc_chain(&e, &d, &dc, &idx, t, t_kv, width, Some(&wk_t_d), None);
    let rel_b = relative(&red_b, &want);
    println!("red b (transposed Q@K operand): {rel_b:.3e}");
    assert!(
        red_b.iter().all(|v| v.is_finite()),
        "red b must run and stay finite"
    );
    assert!(
        rel_b > TC_BAND,
        "a transposed absorb operand passed the gate ({rel_b:.3e})"
    );

    // RED c: SELECTION MASK DROPPED — attend the full causal prefix where the DSA program
    // selected a proper subset. Every query here is budget-limited, so the two programs are
    // different functions and the comparator must say so.
    let full_width = t_kv; // full causal lists need the whole prefix
    let full_lists: Vec<Vec<usize>> = (0..t).map(|i| (0..first_pos + i + 1).collect()).collect();
    let full_idx = lists_to_device(&e, &full_lists, full_width);
    let red_c = tc_chain(&e, &d, &dc, &full_idx, t, t_kv, full_width, None, None);
    let rel_c = relative(&red_c, &want);
    println!("red c (selection mask dropped): {rel_c:.3e}");
    assert!(
        red_c.iter().all(|v| v.is_finite()),
        "red c must run and stay finite"
    );
    assert!(
        rel_c > TC_BAND,
        "attending past the DSA selection passed the gate ({rel_c:.3e})"
    );

    // RED d: CAUSAL OFF BY ONE — one future row appended to every list that has one.
    let shifted: Vec<Vec<usize>> = lists
        .iter()
        .enumerate()
        .map(|(i, l)| {
            let mut l = l.clone();
            let visible = first_pos + i + 1;
            if visible < t_kv {
                l.push(visible); // the first row this query must NOT see
            }
            l
        })
        .collect();
    let shifted_idx = lists_to_device(&e, &shifted, width + 1);
    let red_d = tc_chain(&e, &d, &dc, &shifted_idx, t, t_kv, width + 1, None, None);
    let rel_d = relative(&red_d, &want);
    println!("red d (causal off by one): {rel_d:.3e}");
    assert!(
        red_d.iter().all(|v| v.is_finite()),
        "red d must run and stay finite"
    );
    assert!(
        rel_d > TC_BAND,
        "an off-by-one causal horizon passed the gate ({rel_d:.3e})"
    );
}

// ---------------------------------------------------------------- mixer-level fixture
// The glm5_kpool_indexer_gpu mini fixture family (copied per that file's own convention —
// test files cannot import each other) with ONE change: kv_lora_rank raised to 512, the TC
// kernel's stamped latent width, so the door ENGAGES on this model. index_topk 16 / kpool 4
// keep the fixture in the sparse regime from ~40 tokens (`the_fixture_reaches_the_sparse_
// regime` in the kpool gate proves that property for this geometry).

use memra_engine::hybrid::HybridModel;
use memra_gguf::GgmlType;
use memra_gguf::config::{HfConfig, ModelConfig};
use memra_gguf::model_plan::ModelPlan;
use memra_gguf::source::{TensorSource, TensorView};
use memra_gguf::tensor_contract::{
    CheckpointDialect, ContractOptions, OutputHead, TensorContract, TensorMatch,
};
use memra_reference::{ReferenceWeights, deterministic_fixture};
use std::borrow::Cow;
use std::collections::BTreeMap;

const HIDDEN: usize = 128;
const VOCAB: u32 = 32;
const Q_LORA: usize = 16;
const INDEX_HEADS: usize = 2;
const INDEX_HEAD_DIM: usize = 8;
const INDEX_TOPK: usize = 16;
const KPOOL: usize = 4;
const KV_RANK: usize = 512; // the ONE departure from the kpool mini fixture: the TC stamp
const CTX: usize = 64;

/// Mixer tolerance for the OFF arm — the kpool gate's measured class (6.843e-7 there; 1e-5
/// carries ~15x headroom). The ON arm carries the bf16 band.
const TOL_OFF: f32 = 1e-5;
const TOL_ON: f32 = 8e-3;
/// DISCRETE-SELECTION FLIP CLASS, measured and stated rather than averaged away. Above the
/// indexer budget, layer 2's pool selection and the sigmoid router's top-k are DISCRETE
/// functions of layer 1's output; a bf16-band perturbation flips near-tie choices for a few
/// tokens, and those rows move by percents — the same class as the MEMRA_PP_BF16 near-tie
/// argmax flip (docs/FLAGS.md:864). Measured on this fixture (rig 5090, TF32 off, 2026-08-30):
/// T=64 → 1/64 rows above the bf16 band (worst 4.061e-2); T=4096 → 63/4096 rows (1.5%, worst
/// 2.181e-1); every other row inside 8e-3. On a 2-layer hidden-128 micro model one flipped
/// top-2 expert (of 4) legitimately rewrites tenths of a token's row, so the WORST-ROW band
/// is a fixture-pinned magnitude class, not a correctness bar. The correctness bars are (a)
/// the kernel-level gates, which hold EVERY query to the bf16 band when the selections are
/// pinned, and (b) the COUNT assertion here: flips are isolated (<= t/16 rows), where a
/// dropped mask or causal bug moves MOST rows (red c: 0.8 relative across the whole batch)
/// and cannot hide.
const FLIP_ROW_BAND: f32 = 3e-1;

fn mini_config_json() -> String {
    format!(
        r#"{{
      "model_type": "glm5_next_text",
      "num_hidden_layers": 2,
      "num_nextn_predict_layers": 0,
      "hidden_size": {HIDDEN},
      "intermediate_size": 64,
      "vocab_size": {VOCAB},
      "max_position_embeddings": 8192,
      "rms_norm_eps": 1e-05,
      "hidden_act": "silu",
      "swiglu_limit": 10.0,
      "tie_word_embeddings": true,
      "hc_mult": 4,
      "hc_eps": 1e-06,
      "hc_sinkhorn_iters": 20,
      "mhc": true,
      "layer_types": ["deepseek_sparse_attention", "deepseek_sparse_attention"],
      "mlp_layer_types": ["dense", "dense"],
      "first_k_dense_replace": 2,
      "indexer_types": ["full", "full"],
      "linear_attn_config": {{
        "num_heads": 1,
        "head_dim": 128,
        "short_conv_kernel_size": 4,
        "gate_lower_bound": -5.0,
        "kda_layers": [],
        "full_attn_layers": [0, 1]
      }},
      "num_attention_heads": 2,
      "num_key_value_heads": 2,
      "q_lora_rank": {Q_LORA},
      "kv_lora_rank": {KV_RANK},
      "qk_head_dim": 16,
      "qk_nope_head_dim": 16,
      "qk_rope_head_dim": 0,
      "v_head_dim": 16,
      "mla_use_nope": true,
      "index_n_heads": {INDEX_HEADS},
      "index_head_dim": {INDEX_HEAD_DIM},
      "index_topk": {INDEX_TOPK},
      "index_kpool": {KPOOL},
      "index_kpool_always_select_tail": true,
      "index_kpool_compress": true,
      "indexer_rope_interleave": true,
      "index_share_for_mtp_iteration": true,
      "n_routed_experts": 4,
      "num_experts_per_tok": 2,
      "moe_intermediate_size": 32,
      "n_shared_experts": 1,
      "scoring_func": "sigmoid",
      "topk_method": "noaux_tc",
      "routed_scaling_factor": 2.5,
      "norm_topk_prob": true,
      "n_group": 1,
      "topk_group": 1,
      "head_dim": 0,
      "attention_bias": false,
      "moe_router_dtype": "float32",
      "dtype": "bfloat16"
    }}"#
    )
}

struct OwnedTensor {
    bytes: Vec<u8>,
    ne: Vec<u64>,
    ggml_type: GgmlType,
}

struct FixtureSource {
    config: ModelConfig,
    tensors: BTreeMap<String, OwnedTensor>,
}

impl TensorSource for FixtureSource {
    fn config(&self) -> ModelConfig {
        self.config.clone()
    }
    fn find(&self, name: &str) -> Option<TensorView<'_>> {
        let t = self.tensors.get(name)?;
        Some(TensorView {
            bytes: Cow::Borrowed(&t.bytes),
            ggml_type: t.ggml_type,
            ne: t.ne.clone(),
        })
    }
}

fn fixture_source(
    config: &ModelConfig,
    plan: &ModelPlan,
    weights: &ReferenceWeights,
) -> FixtureSource {
    let contract = TensorContract::for_plan(
        plan,
        CheckpointDialect::Gguf,
        ContractOptions {
            output_head: OutputHead::TiedToEmbedding,
        },
    )
    .expect("contract for the mini512 glm5_next plan");
    let mut tensors = BTreeMap::new();
    for req in contract
        .requirements
        .iter()
        .filter(|r| r.required || weights.contains_key(&r.id))
    {
        let tensor = weights
            .get(&req.id)
            .unwrap_or_else(|| panic!("reference fixture is missing {:?}", req.id));
        let names = match req.match_mode {
            TensorMatch::OneOf => &req.names[..1],
            TensorMatch::All => req.names.as_slice(),
        };
        for name in names {
            tensors.insert(
                name.clone(),
                OwnedTensor {
                    bytes: tensor.data.iter().flat_map(|v| v.to_le_bytes()).collect(),
                    ne: req.shape.clone(),
                    ggml_type: GgmlType::F32,
                },
            );
        }
    }
    FixtureSource {
        config: config.clone(),
        tensors,
    }
}

fn tokens(n: usize, seed: u64) -> Vec<u32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((s >> 33) as u32) % VOCAB
        })
        .collect()
}

struct Harness {
    engine: Engine,
    model: HybridModel,
    plan: ModelPlan,
    weights: ReferenceWeights,
}

impl Harness {
    fn new() -> Self {
        force_true_f32();
        let config = ModelConfig::from_hf(&HfConfig::parse(&mini_config_json()));
        let plan = memra_gguf::model_packs::for_config(&config)
            .expect("glm5_next model pack matches the mini512 config")
            .compile_plan(&config)
            .expect("mini512 glm5_next plan compiles");
        let weights = deterministic_fixture(&plan).expect("fixture").weights;
        let source = fixture_source(&config, &plan, &weights);
        let engine = Engine::new(0).expect("CUDA engine on device 0");
        let model = HybridModel::load_from_source_without_mtp(&engine, &source)
            .expect("mini512 glm5_next model loads");
        Self {
            engine,
            model,
            plan,
            weights,
        }
    }

    fn logits(&self, ids: &[u32]) -> Vec<f32> {
        self.model.forward(&self.engine, ids).expect("gpu prefill")
    }

    fn reference_logits(&self, ids: &[u32]) -> Vec<f32> {
        memra_reference::execute(&self.plan, &self.weights, ids)
            .expect("reference execute")
            .logits
    }
}

/// Flag scope guards: pin `MEMRA_MLA_TC_PREFILL` for one arm and restore on drop. The flag is
/// read PER CALL by the door, so these flip arms inside one process — the reason the door is
/// not latched. Since the 2026-08-30 default flip (unset = ON, owner acceptance), the OFF arm
/// must be PINNED with `=0`; relying on an unset env would silently run the TC arm twice
/// (caught by this gate the day of the flip: "OFF arm drifted" at exactly the bf16 band).
/// GPU tests in this file serialize on `gpu_guard`.
struct FlagOn;
impl FlagOn {
    fn new() -> Self {
        // SAFETY: gpu_guard serializes every test in this binary that reads the flag.
        unsafe { std::env::set_var("MEMRA_MLA_TC_PREFILL", "1") };
        FlagOn
    }
}
impl Drop for FlagOn {
    fn drop(&mut self) {
        unsafe { std::env::remove_var("MEMRA_MLA_TC_PREFILL") };
    }
}
struct FlagOff;
impl FlagOff {
    fn new() -> Self {
        // SAFETY: gpu_guard serializes every test in this binary that reads the flag.
        unsafe { std::env::set_var("MEMRA_MLA_TC_PREFILL", "0") };
        FlagOff
    }
}
impl Drop for FlagOff {
    fn drop(&mut self) {
        unsafe { std::env::remove_var("MEMRA_MLA_TC_PREFILL") };
    }
}

/// GATE 4 — the MIXER through the REAL door, vs `memra_reference::execute`, on the established
/// fixture family at the kernel's stamped latent width. Engagement is anchored on the dispatch
/// counter, not the flag (LAW:wiring-assertions-match-prose).
#[test]
#[ignore = "needs a CUDA device — run under flock /tmp/memra-5090.lock"]
fn gpu_mla_tc_mixer_matches_reference_mini512() {
    let _gpu = gpu_guard();
    let h = Harness::new();
    for t in [16usize, 64] {
        let ids = tokens(t, 0xC7E1);
        let want = h.reference_logits(&ids);
        let off = {
            let _f = FlagOff::new();
            h.logits(&ids)
        };
        let rel_off = relative(&off, &want);

        let before = memra_engine::mla_tc_prefill_dispatches();
        let on = {
            let _f = FlagOn::new();
            h.logits(&ids)
        };
        let engaged = memra_engine::mla_tc_prefill_dispatches() - before;
        let rel_on = relative(&on, &want);
        let vocab = VOCAB as usize;
        let row_rels: Vec<f32> = (0..t)
            .map(|i| {
                relative(
                    &on[i * vocab..(i + 1) * vocab],
                    &want[i * vocab..(i + 1) * vocab],
                )
            })
            .collect();
        let above = row_rels.iter().filter(|&&r| r > TOL_ON).count();
        let worst_row = row_rels.iter().copied().fold(0.0f32, f32::max);
        println!(
            "T={t}: OFF vs reference {rel_off:.3e} (tol {TOL_OFF:.0e}), \
             ON vs reference {rel_on:.3e}, ON-vs-OFF {:.3e}, {engaged} TC dispatches; \
             per-token rows above the bf16 band ({TOL_ON:.0e}): {above}/{t} \
             (flip allowance {}, worst row {worst_row:.3e}, flip band {FLIP_ROW_BAND:.1e})",
            relative(&on, &off),
            t / 16,
        );
        assert!(rel_off <= TOL_OFF, "OFF arm drifted: {rel_off:.3e}");
        // The flip-signature assertion (see FLIP_ROW_BAND): all rows in the bf16 band except at
        // most t/16 discrete re-selections, each bounded well below the red class.
        assert!(
            above <= t / 16,
            "ON arm: {above}/{t} token rows above the bf16 band — too many for near-tie \
             selection flips; this is the many-rows signature of a mask/causality bug"
        );
        assert!(
            worst_row <= FLIP_ROW_BAND,
            "ON arm worst token row {worst_row:.3e} exceeds the flip class {FLIP_ROW_BAND:.0e}"
        );
        // 2 MLA layers, one chunk each at these lengths.
        assert_eq!(
            engaged, 2,
            "the door must engage once per MLA layer at t={t}"
        );
    }

    // Below the t >= 16 door: the ON arm must be the OFF program, bit for bit, count flat.
    let ids = tokens(8, 0xC7E1);
    let off = {
        let _f = FlagOff::new();
        h.logits(&ids)
    };
    let before = memra_engine::mla_tc_prefill_dispatches();
    let on = {
        let _f = FlagOn::new();
        h.logits(&ids)
    };
    assert_eq!(
        memra_engine::mla_tc_prefill_dispatches() - before,
        0,
        "the door engaged below its t >= 16 threshold"
    );
    assert_eq!(
        on.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        off.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        "t=8 with the flag on is not byte-identical to the flag-off program"
    );
}

/// GATE 5 — the mixer bar at t=4096: deep sparse regime (budget 16 of up to 4096 visible
/// rows), the length class the census attributed. The reference is exact f32; the ON band is
/// the bf16 class.
#[test]
#[ignore = "needs a CUDA device — run under flock /tmp/memra-5090.lock; reference execute at t=4096 is minutes of CPU"]
fn gpu_mla_tc_mixer_matches_reference_t4096() {
    let _gpu = gpu_guard();
    let h = Harness::new();
    let ids = tokens(4096, 0x4096_C7E1);
    let want = h.reference_logits(&ids);
    let off = {
        let _f = FlagOff::new();
        h.logits(&ids)
    };
    let rel_off = relative(&off, &want);
    let before = memra_engine::mla_tc_prefill_dispatches();
    let on = {
        let _f = FlagOn::new();
        h.logits(&ids)
    };
    let engaged = memra_engine::mla_tc_prefill_dispatches() - before;
    let rel_on = relative(&on, &want);
    let t = ids.len();
    let vocab = VOCAB as usize;
    let row_rels: Vec<f32> = (0..t)
        .map(|i| {
            relative(
                &on[i * vocab..(i + 1) * vocab],
                &want[i * vocab..(i + 1) * vocab],
            )
        })
        .collect();
    let above = row_rels.iter().filter(|&&r| r > TOL_ON).count();
    let worst_row = row_rels.iter().copied().fold(0.0f32, f32::max);
    println!(
        "T=4096: OFF vs reference {rel_off:.3e}, ON vs reference {rel_on:.3e}, \
         {engaged} TC dispatches; rows above the bf16 band: {above}/{t} \
         (flip allowance {}, worst row {worst_row:.3e})",
        t / 16
    );
    assert!(
        rel_off <= TOL_OFF,
        "OFF arm drifted at t=4096: {rel_off:.3e}"
    );
    // The flip-signature assertion (see FLIP_ROW_BAND).
    assert!(
        above <= t / 16,
        "ON arm at t=4096: {above}/{t} rows above the bf16 band — the many-rows signature"
    );
    assert!(
        worst_row <= FLIP_ROW_BAND,
        "ON arm worst token row at t=4096: {worst_row:.3e} exceeds {FLIP_ROW_BAND:.0e}"
    );
    assert!(engaged >= 2, "the door must engage at t=4096");
}

/// GATE 6 — DECODE BYTE-IDENTITY, gated not assumed. Two identical caches primed with the flag
/// OFF; then every decode step runs once with the flag ON and once OFF. The t=1 program must
/// never enter the door (counter FLAT) and the logits must match BIT FOR BIT.
#[test]
#[ignore = "needs a CUDA device — run under flock /tmp/memra-5090.lock"]
fn gpu_mla_tc_decode_stays_byte_identical() {
    let _gpu = gpu_guard();
    let h = Harness::new();
    let prompt = 40usize;
    let steps = CTX - prompt;
    let ids = tokens(CTX, 0x8_0A17_DEC0);

    let mut cache_on =
        memra_engine::cache::Cache::new_planned(&h.engine, &h.model.cfg, &h.plan, CTX + 8)
            .expect("cache A");
    let mut cache_off =
        memra_engine::cache::Cache::new_planned(&h.engine, &h.model.cfg, &h.plan, CTX + 8)
            .expect("cache B");
    // Both primes run the FLAG-OFF program (pinned since the default flip), so the caches
    // enter decode identical.
    {
        let _f = FlagOff::new();
        h.model
            .prime_cache(&h.engine, &ids[..prompt], &mut cache_on, 0)
            .expect("prime A");
        h.model
            .prime_cache(&h.engine, &ids[..prompt], &mut cache_off, 0)
            .expect("prime B");
    }

    let before = memra_engine::mla_tc_prefill_dispatches();
    for step in 0..steps {
        let row = prompt + step;
        let on = {
            let _f = FlagOn::new();
            h.model
                .decode_step(&h.engine, ids[row], &mut cache_on)
                .expect("decode step, flag on")
        };
        let off = h
            .model
            .decode_step(&h.engine, ids[row], &mut cache_off)
            .expect("decode step, flag off");
        assert_eq!(
            on.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            off.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "decode step {step} (position {row}) is not byte-identical with the flag on"
        );
    }
    assert_eq!(
        memra_engine::mla_tc_prefill_dispatches() - before,
        0,
        "the TC door engaged during t=1 decode — the decode path is supposed to be untouched"
    );
    println!("{steps} decode steps byte-identical, TC dispatch counter flat");
}
