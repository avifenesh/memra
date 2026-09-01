//! glm5 VERIFY-BATCH kernel bit-gates (lane/glm5-verify-batch).
//!
//! The walk-level gates (`glm5_tparallel_verify_gpu`) hold the end-to-end claim; the gates
//! here localize the three NEW shape changes the batched walk stands on, at the kernel
//! seam, so a future kernel edit that breaks one class fails HERE by name instead of as an
//! undiagnosed walk divergence (the `_vl`-twin discipline: every batched twin carries its
//! own bit-gate, red-armed):
//!
//! 1. `matvec_bf16_f32acc_x4_tcols` — the t-column bf16 matvec twin (weight read once for
//!    all t rows) vs the t=1 program per row. Red arm: a shifted activation row must bite.
//! 2. `memra_kda_scan_s128` at T=t vs the chained T=1 launches — the in-kernel sequential
//!    recurrence IS the per-row chain. Red arm: swapped input rows must bite.
//! 3. The KDA conv PREFILL arm (+ ring roll) at T=t vs the chained DECODE arm — same
//!    ascending taps over the same window values, per token. Plus the rollback re-roll:
//!    ring restore + roll(T=keep) == the chain's ring after row keep-1, for every keep.
//! 4. (lane/glm5-vrest) The verify-rows MoE pairs twins (`moe_gate_up_preclamp8_q8_rows` +
//!    `moe_down8_fma_q8_rows`) vs the sequential per-(token,expert) slab chain
//!    (quantize_q8_1_view + qmatvec_expert_q8 + swiglu_preclamped + quantize_q8_1 +
//!    qmatvec_expert_q8 + slot-ordered axpy), on minted NVFP4 expert banks with a LIVE
//!    macro plane. Red arms: swapped pair rows (row isolation) and dropped macro scales.
//!
//! Rig law (exactness only, never timing):
//!   NVIDIA_TF32_OVERRIDE=0 flock /tmp/memra-5090.lock \
//!     cargo test -p memra-engine --test glm5_verify_batch_gpu -- --ignored --test-threads=1

use memra_engine::Engine;

fn gpu_guard() -> std::sync::MutexGuard<'static, ()> {
    static GPU: std::sync::Mutex<()> = std::sync::Mutex::new(());
    GPU.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn force_true_f32() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        if std::env::var("NVIDIA_TF32_OVERRIDE").as_deref() != Ok("0") {
            // SAFETY: no CUDA call yet in this process; call_once serializes test threads.
            unsafe { std::env::set_var("NVIDIA_TF32_OVERRIDE", "0") };
        }
    });
}

/// Deterministic non-trivial f32s (varied signs and magnitudes; an all-ones operand cannot
/// catch a swapped index).
fn varied(len: usize, seed: u64, spread: f32) -> Vec<f32> {
    (0..len)
        .map(|i| {
            let x = (i as u64)
                .wrapping_mul(6364136223846793005)
                .wrapping_add(seed)
                .rotate_left(17) as f64
                / u64::MAX as f64;
            spread * (x as f32 - 0.5)
        })
        .collect()
}

/// f32 -> bf16 bytes (truncation — the kernels reinterpret raw u16 patterns, so any valid
/// bf16 encoding exercises them).
fn bf16_bytes(v: &[f32]) -> Vec<u8> {
    v.iter()
        .flat_map(|x| ((x.to_bits() >> 16) as u16).to_le_bytes())
        .collect()
}

fn bit_diffs(a: &[f32], b: &[f32]) -> usize {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .filter(|(x, y)| x.to_bits() != y.to_bits())
        .count()
}

// ---------------------------------------------------------------------------------------------
// Gate 1 — the tcols bf16 matvec twin: bit-identical per (row, token) to the t=1 program.
// ---------------------------------------------------------------------------------------------

#[test]
#[ignore = "needs a CUDA device, run under flock /tmp/memra-5090.lock"]
fn gpu_tcols_matches_per_row_t1_program_bitwise() {
    let _gpu = gpu_guard();
    force_true_f32();
    let e = Engine::new(0).expect("CUDA engine on device 0");
    // out_f = 133: a ragged 4-row block tail; in_f multiple of 8 (the kernels' bar).
    let (in_f, out_f) = (256usize, 133usize);
    let w_host = varied(out_f * in_f, 0xBF16, 2.0);
    let w = e.htod_bytes(&bf16_bytes(&w_host)).expect("weight upload");

    for t in 2..=8usize {
        let x_host = varied(t * in_f, 0xAC70 + t as u64, 2.0);
        let x = e.htod(&x_host).expect("activation upload");

        // Batched twin: one launch, all t rows.
        let mut y_t = e.uninit(t * out_f).expect("y_t");
        e.matvec_bf16_tcols_into(&w, &x, &mut y_t, in_f, out_f, t)
            .expect("tcols launch");
        let got = e.dtoh(&y_t).expect("tcols readback");

        // Reference: the t=1 program per row (grid.y=1 rows kernel — the decode class).
        let mut want = vec![0f32; t * out_f];
        for r in 0..t {
            let xr = e.htod(&x_host[r * in_f..(r + 1) * in_f]).expect("row");
            let mut yr = e.uninit(out_f).expect("yr");
            e.matvec_bf16_rows_into(&w, &xr, &mut yr, in_f, out_f, 1)
                .expect("t=1 launch");
            want[r * out_f..(r + 1) * out_f].copy_from_slice(&e.dtoh(&yr).expect("row back"));
        }
        let diffs = bit_diffs(&got, &want);
        assert_eq!(
            diffs,
            0,
            "t={t}: tcols twin diverged from the per-row t=1 program in {diffs}/{} outputs",
            t * out_f
        );
        println!("gate 1 PASS t={t}: {} outputs bit-identical", t * out_f);
    }

    // RED ARM: shifting the activation by one row must bite in (almost) every output —
    // proves the compare above can see a row-addressing bug.
    let t = 4usize;
    let x_host = varied((t + 1) * in_f, 0x5EED, 2.0);
    let x_ok = e.htod(&x_host[..t * in_f]).expect("x ok");
    let x_shift = e.htod(&x_host[in_f..(t + 1) * in_f]).expect("x shifted");
    let mut y_ok = e.uninit(t * out_f).expect("y ok");
    let mut y_shift = e.uninit(t * out_f).expect("y shifted");
    e.matvec_bf16_tcols_into(&w, &x_ok, &mut y_ok, in_f, out_f, t)
        .expect("ok launch");
    e.matvec_bf16_tcols_into(&w, &x_shift, &mut y_shift, in_f, out_f, t)
        .expect("shifted launch");
    let diffs = bit_diffs(
        &e.dtoh(&y_ok).expect("ok back"),
        &e.dtoh(&y_shift).expect("shifted back"),
    );
    assert!(
        diffs > 0,
        "shifted-row red arm produced identical outputs — the gate cannot see row addressing"
    );
    println!("gate 1 RED bites: {diffs} outputs differ with a one-row activation shift");
}

// ---------------------------------------------------------------------------------------------
// Gate 2 — the KDA scan at T=t vs the chained T=1 launches: outputs AND final state.
// ---------------------------------------------------------------------------------------------

#[test]
#[ignore = "needs a CUDA device, run under flock /tmp/memra-5090.lock"]
fn gpu_kda_scan_t_walk_matches_chained_t1_bitwise() {
    let _gpu = gpu_guard();
    force_true_f32();
    let e = Engine::new(0).expect("CUDA engine on device 0");
    const D: usize = 128; // the only instantiated head width
    let heads = 2usize;
    let qkv = heads * D;
    let t = 5usize;
    let scale = 1.0 / (D as f32).sqrt();
    let state_w = heads * D * D;

    let q_h = varied(t * qkv, 0x91, 1.0);
    let k_h = varied(t * qkv, 0x92, 1.0);
    let v_h = varied(t * qkv, 0x93, 1.0);
    // Raw log-gates: negative (the scan applies expf — keep decay factors in (0,1)).
    let g_h: Vec<f32> = varied(t * qkv, 0x94, 1.0)
        .iter()
        .map(|x| -x.abs())
        .collect();
    // Betas in (0,1).
    let b_h: Vec<f32> = varied(t * heads, 0x95, 1.0)
        .iter()
        .map(|x| 0.5 + 0.4 * x)
        .collect();
    let s0_h = varied(state_w, 0x96, 1.0);

    // Arm A: ONE T=t launch.
    let (q, k, v, g, b) = (
        e.htod(&q_h).unwrap(),
        e.htod(&k_h).unwrap(),
        e.htod(&v_h).unwrap(),
        e.htod(&g_h).unwrap(),
        e.htod(&b_h).unwrap(),
    );
    let s0 = e.htod(&s0_h).unwrap();
    let mut s_out_a = e.zeros(state_w).unwrap();
    let mut o_a = e.uninit(t * qkv).unwrap();
    e.kda_scan(
        &q,
        &k,
        &v,
        &g,
        &b,
        &s0,
        &mut s_out_a,
        &mut o_a,
        heads,
        t,
        scale,
    )
    .expect("T=t scan");
    let out_a = e.dtoh(&o_a).unwrap();
    let state_a = e.dtoh(&s_out_a).unwrap();

    // Arm B: t chained T=1 launches (state round-trips through the ping-pong, exactly the
    // per-row walk's shape).
    let mut state_in = e.htod(&s0_h).unwrap();
    let mut state_out = e.zeros(state_w).unwrap();
    let mut out_b = vec![0f32; t * qkv];
    for r in 0..t {
        let qr = e.htod(&q_h[r * qkv..(r + 1) * qkv]).unwrap();
        let kr = e.htod(&k_h[r * qkv..(r + 1) * qkv]).unwrap();
        let vr = e.htod(&v_h[r * qkv..(r + 1) * qkv]).unwrap();
        let gr = e.htod(&g_h[r * qkv..(r + 1) * qkv]).unwrap();
        let br = e.htod(&b_h[r * heads..(r + 1) * heads]).unwrap();
        let mut or = e.uninit(qkv).unwrap();
        e.kda_scan(
            &qr,
            &kr,
            &vr,
            &gr,
            &br,
            &state_in,
            &mut state_out,
            &mut or,
            heads,
            1,
            scale,
        )
        .expect("T=1 scan");
        out_b[r * qkv..(r + 1) * qkv].copy_from_slice(&e.dtoh(&or).unwrap());
        std::mem::swap(&mut state_in, &mut state_out);
    }
    let state_b = e.dtoh(&state_in).unwrap();

    assert_eq!(
        bit_diffs(&out_a, &out_b),
        0,
        "scan T={t} readouts diverged from the chained T=1 walk"
    );
    assert_eq!(
        bit_diffs(&state_a, &state_b),
        0,
        "scan T={t} final state diverged from the chained T=1 walk"
    );
    println!("gate 2 PASS: T={t} scan == chained T=1 (readouts + state, bit-for-bit)");

    // RED ARM: swap rows 1 and 2 of g in arm A — the recurrence must propagate the swap
    // into different readouts AND a different final state.
    let mut g_swapped = g_h.clone();
    for c in 0..qkv {
        g_swapped.swap(qkv + c, 2 * qkv + c);
    }
    let g_bad = e.htod(&g_swapped).unwrap();
    let mut s_out_red = e.zeros(state_w).unwrap();
    let mut o_red = e.uninit(t * qkv).unwrap();
    e.kda_scan(
        &q,
        &k,
        &v,
        &g_bad,
        &b,
        &s0,
        &mut s_out_red,
        &mut o_red,
        heads,
        t,
        scale,
    )
    .expect("red scan");
    let d_out = bit_diffs(&e.dtoh(&o_red).unwrap(), &out_a);
    let d_state = bit_diffs(&e.dtoh(&s_out_red).unwrap(), &state_a);
    assert!(
        d_out > 0 && d_state > 0,
        "swapped-row red arm changed nothing (out {d_out}, state {d_state}) — the gate \
         cannot see row order"
    );
    println!("gate 2 RED bites: {d_out} readout / {d_state} state diffs with swapped g rows");
}

// ---------------------------------------------------------------------------------------------
// Gate 3 — conv PREFILL arm vs the chained DECODE arm at T=t, plus the rollback re-roll.
// ---------------------------------------------------------------------------------------------

#[test]
#[ignore = "needs a CUDA device, run under flock /tmp/memra-5090.lock"]
fn gpu_kda_conv_prefill_arm_matches_decode_chain_and_reroll() {
    let _gpu = gpu_guard();
    force_true_f32();
    let e = Engine::new(0).expect("CUDA engine on device 0");
    let qkv = 192usize;
    let kernel = 4usize;
    let pad = kernel - 1;
    let t = 6usize;

    let w_h = varied(3 * qkv * kernel, 0xC0, 1.0);
    let ring0_h = varied(3 * qkv * pad, 0xC1, 1.0);
    let w = e.htod(&w_h).unwrap();
    // Per-plane raw projection rows [t, qkv].
    let raws_h: Vec<Vec<f32>> = (0..3)
        .map(|p| varied(t * qkv, 0xD0 + p as u64, 1.0))
        .collect();
    let raws: Vec<_> = raws_h.iter().map(|r| e.htod(r).unwrap()).collect();

    // Arm A: the batched walk's shape — conv all planes over the OLD ring, then roll.
    let mut ring_a = e.htod(&ring0_h).unwrap();
    let mut y_a: Vec<Vec<f32>> = Vec::new();
    for (p, raw) in raws.iter().enumerate() {
        let mut y = e.uninit(t * qkv).unwrap();
        e.kda_conv_silu(raw, &w, &ring_a, &mut y, qkv, t, kernel, p)
            .expect("prefill conv");
        y_a.push(e.dtoh(&y).unwrap());
    }
    for (p, raw) in raws.iter().enumerate() {
        e.kda_conv_ring_roll(raw, &mut ring_a, qkv, t, kernel, p)
            .expect("roll");
    }
    let ring_a_final = e.dtoh(&ring_a).unwrap();

    // Arm B: the per-row walk's shape — t chained decode-arm launches per plane, ring
    // rolled in-kernel; ring snapshots banked after every row (the re-roll references).
    let mut ring_b = e.htod(&ring0_h).unwrap();
    let mut y_b: Vec<Vec<f32>> = vec![vec![0f32; t * qkv]; 3];
    let mut ring_after: Vec<Vec<f32>> = Vec::new(); // ring bytes after row r
    for r in 0..t {
        for (p, raw_h) in raws_h.iter().enumerate() {
            let xr = e.htod(&raw_h[r * qkv..(r + 1) * qkv]).unwrap();
            let mut yr = e.uninit(qkv).unwrap();
            e.kda_conv_silu_decode(&xr, &mut ring_b, &w, &mut yr, qkv, kernel, p)
                .expect("decode conv");
            y_b[p][r * qkv..(r + 1) * qkv].copy_from_slice(&e.dtoh(&yr).unwrap());
        }
        ring_after.push(e.dtoh(&ring_b).unwrap());
    }

    for p in 0..3 {
        assert_eq!(
            bit_diffs(&y_a[p], &y_b[p]),
            0,
            "plane {p}: prefill conv arm diverged from the chained decode arm"
        );
    }
    assert_eq!(
        bit_diffs(&ring_a_final, &ring_after[t - 1]),
        0,
        "post-roll ring diverged from the decode chain's ring"
    );
    println!("gate 3 PASS: prefill conv + roll == chained decode arm (3 planes, t={t})");

    // Rollback re-roll: ring := snapshot, roll(T=keep) — must equal the chain's ring
    // after row keep-1, for every partial keep.
    for keep in 1..t {
        let mut ring_c = e.htod(&ring0_h).unwrap();
        for (p, raw) in raws.iter().enumerate() {
            e.kda_conv_ring_roll(raw, &mut ring_c, qkv, keep, kernel, p)
                .expect("re-roll");
        }
        assert_eq!(
            bit_diffs(&e.dtoh(&ring_c).unwrap(), &ring_after[keep - 1]),
            0,
            "keep={keep}: re-rolled ring diverged from the chain's ring after row {}",
            keep - 1
        );
    }
    println!("gate 3 PASS: re-roll(keep) == chain ring, every keep in 1..{t}");

    // RED ARM: re-roll from a CORRUPTED snapshot at keep < pad (the window still reads
    // snapshot slots there) must diverge — proves the re-roll compare sees the base.
    // Slot 1 of channel 0: at keep=1 the re-roll copies old slots 1..pad into 0..pad-1,
    // so a slot-1 corruption lands in the rebuilt ring (slot 0 would be dropped).
    let mut bad0 = ring0_h.clone();
    bad0[1] += 1.0;
    let mut ring_red = e.htod(&bad0).unwrap();
    for (p, raw) in raws.iter().enumerate() {
        e.kda_conv_ring_roll(raw, &mut ring_red, qkv, 1, kernel, p)
            .expect("red re-roll");
    }
    let d = bit_diffs(&e.dtoh(&ring_red).unwrap(), &ring_after[0]);
    assert!(
        d > 0,
        "corrupted-snapshot red arm produced the chain ring — the re-roll gate is blind"
    );
    println!("gate 3 RED bites: {d} ring slots differ from a corrupted snapshot base");
}

// ---------------------------------------------------------------------------------------------
// Gate 4 — the verify-rows MoE pairs twins vs the sequential per-(token,expert) slab chain
// (lane/glm5-vrest). The batched arm's bar is per-row BIT identity with the chain the per-row
// verify walk (and plain decode) runs on the serving expert class: NVFP4 banks, live
// per-expert macro scales, PRE-clamped SwiGLU, slot-ordered down accumulation.
// ---------------------------------------------------------------------------------------------

/// Mint a uniform NVFP4 expert slab: `n_expert` experts x `out_f` rows x `in_f` cols.
/// Returns (device slab, row_bytes, expert_stride_bytes).
fn nvfp4_slab(
    e: &Engine,
    n_expert: usize,
    out_f: usize,
    in_f: usize,
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
    let slab = e.htod_bytes(&bytes).expect("nvfp4 slab upload");
    (slab, row_bytes, stride)
}

/// The sequential slab chain, verbatim from `moe_ffn_sequential_zq8`'s per-expert arm:
/// per (token, slot): quantize the token row, gate/up expert matvecs, PRE-clamp epilogue
/// with the gate/up macro folds, activation quantize, down matvec, slot-ordered axpy with
/// the down macro folded into the weight. `out` rows are zeroed first (the loop's class).
#[allow(clippy::too_many_arguments)]
fn moe_sequential_chain(
    e: &Engine,
    slabs: &(cudarc::driver::CudaSlice<u8>, usize, usize),
    slabs_d: &(cudarc::driver::CudaSlice<u8>, usize, usize),
    macros: &(Vec<f32>, Vec<f32>, Vec<f32>),
    z: &[f32],
    sel: &[u32],
    w: &[f32],
    t: usize,
    n_used: usize,
    (in_f, n_ff): (usize, usize),
    limit: f32,
) -> Vec<f32> {
    let (slab_gu, rb_gu, stride_gu) = slabs;
    let (slab_d, rb_d, stride_d) = slabs_d;
    let z_d = e.htod(z).expect("z upload");
    let mut out = e.zeros(t * in_f).expect("out");
    for tok in 0..t {
        let zt = z_d.slice(tok * in_f..(tok + 1) * in_f);
        let (zq, zd) = e.quantize_q8_1_view(&zt, 1, in_f).expect("tok quantize");
        for j in 0..n_used {
            let ex = sel[tok * n_used + j] as usize;
            let g0 = ex * stride_gu;
            let gate = e
                .qmatvec_expert_q8(
                    slab_gu,
                    g0..g0 + stride_gu,
                    &zq,
                    &zd,
                    1,
                    in_f,
                    n_ff,
                    memra_engine::QT_NVFP4,
                    *rb_gu,
                )
                .expect("gate matvec");
            // One slab serves gate and up in this gate (distinct macro folds keep the two
            // dots distinguishable); the kernels only see row pointers either way.
            let up = e
                .qmatvec_expert_q8(
                    slab_gu,
                    g0..g0 + stride_gu,
                    &zq,
                    &zd,
                    1,
                    in_f,
                    n_ff,
                    memra_engine::QT_NVFP4,
                    *rb_gu,
                )
                .expect("up matvec");
            let mut act = e.uninit(n_ff).expect("act");
            e.swiglu_preclamped_mul_scaled(
                &gate,
                &up,
                macros.0[ex],
                macros.1[ex],
                limit,
                &mut act,
                n_ff,
            )
            .expect("preclamp epilogue");
            let (aq2, ad2) = e.quantize_q8_1(&act, 1, n_ff).expect("act quantize");
            let d0 = ex * stride_d;
            let y = e
                .qmatvec_expert_q8(
                    slab_d,
                    d0..d0 + stride_d,
                    &aq2,
                    &ad2,
                    1,
                    n_ff,
                    in_f,
                    memra_engine::QT_NVFP4,
                    *rb_d,
                )
                .expect("down matvec");
            let mut dst = out.slice_mut(tok * in_f..(tok + 1) * in_f);
            e.axpy_into(&y, w[tok * n_used + j] * macros.2[ex], &mut dst, in_f)
                .expect("axpy");
        }
    }
    e.dtoh(&out).expect("sequential readback")
}

/// The batched pairs program: build the plane-major pointer/scale tables from the slab
/// bases (the dispatch arm's exact host build), then the two rows launches.
#[allow(clippy::too_many_arguments)]
fn moe_pairs_program(
    e: &Engine,
    slabs: &(cudarc::driver::CudaSlice<u8>, usize, usize),
    slabs_d: &(cudarc::driver::CudaSlice<u8>, usize, usize),
    macros: &(Vec<f32>, Vec<f32>, Vec<f32>),
    z: &[f32],
    sel: &[u32],
    w: &[f32],
    t: usize,
    n_used: usize,
    (in_f, n_ff): (usize, usize),
    limit: f32,
    mutate: impl Fn(&mut Vec<u64>, &mut Vec<f32>, usize),
) -> Vec<f32> {
    use cudarc::driver::DevicePtr;
    let (slab_gu, rb_gu, stride_gu) = slabs;
    let (slab_d, rb_d, stride_d) = slabs_d;
    let stream = e.stream();
    let (base_gu, _g0) = slab_gu.device_ptr(&stream);
    let (base_d, _g1) = slab_d.device_ptr(&stream);
    let n_pairs = t * n_used;
    let mut ptrs = vec![0u64; 3 * n_pairs];
    let mut scl = vec![0f32; 3 * n_pairs];
    for (p, (&ex, &wj)) in sel.iter().zip(w).enumerate() {
        let ex = ex as usize;
        ptrs[p] = base_gu + (ex * stride_gu) as u64;
        ptrs[n_pairs + p] = base_gu + (ex * stride_gu) as u64;
        ptrs[2 * n_pairs + p] = base_d + (ex * stride_d) as u64;
        scl[p] = macros.0[ex];
        scl[n_pairs + p] = macros.1[ex];
        scl[2 * n_pairs + p] = wj * macros.2[ex];
    }
    mutate(&mut ptrs, &mut scl, n_pairs);
    let ptrs_d = e.htod_u64(&ptrs).expect("ptr table");
    let scl_d = e.htod(&scl).expect("scale table");
    let z_d = e.htod(z).expect("z upload");
    let (zq, zd) = e.quantize_q8_1(&z_d, t, in_f).expect("batched quantize");
    let act = e
        .moe_gate_up_preclamp8_q8_rows(
            &ptrs_d,
            &scl_d,
            &zq,
            &zd,
            limit,
            in_f,
            n_ff,
            n_used,
            n_pairs,
            memra_engine::QT_NVFP4,
            memra_engine::QT_NVFP4,
            *rb_gu,
            *rb_gu,
        )
        .expect("gate/up rows launch");
    let (aq2, ad2) = e
        .quantize_q8_1(&act, n_pairs, n_ff)
        .expect("pair act quantize");
    let mut out = e.uninit(t * in_f).expect("out");
    e.moe_down8_fma_q8_rows(
        &ptrs_d,
        &scl_d,
        &aq2,
        &ad2,
        &mut out,
        n_ff,
        in_f,
        n_used,
        n_pairs,
        memra_engine::QT_NVFP4,
        *rb_d,
    )
    .expect("down rows launch");
    e.dtoh(&out).expect("pairs readback")
}

#[test]
#[ignore = "needs a CUDA device, run under flock /tmp/memra-5090.lock"]
fn gpu_moe_vrows_pairs_match_sequential_chain_bitwise() {
    let _gpu = gpu_guard();
    force_true_f32();
    let e = Engine::new(0).expect("CUDA engine on device 0");
    // NVFP4 block = 64: in_f 128 (gate/up), down in = n_ff 64 — the glm5 shape class shrunk.
    let (in_f, n_ff) = (128usize, 64usize);
    let (n_expert, n_used) = (16usize, 8usize);
    let slabs = nvfp4_slab(&e, n_expert, n_ff, in_f, 0x6A7E);
    let slabs_d = nvfp4_slab(&e, n_expert, in_f, n_ff, 0xD003);
    // LIVE macro plane (all != 1.0) — dropping any fold must be visible.
    let macros = (
        (0..n_expert)
            .map(|i| 0.5 + 0.07 * i as f32)
            .collect::<Vec<_>>(),
        (0..n_expert)
            .map(|i| 1.6 - 0.05 * i as f32)
            .collect::<Vec<_>>(),
        (0..n_expert)
            .map(|i| 0.8 + 0.04 * i as f32)
            .collect::<Vec<_>>(),
    );
    // limit small enough that real rows cross it in both signs (the epilogue-gate lesson:
    // at the shipped 10.0 the clamp never fires on a small fixture and the FORM is untested).
    let limit = 0.75f32;

    for t in 2..=8usize {
        let z = varied(t * in_f, 0x2A + t as u64, 2.0);
        // Distinct expert sets per token (near-disjoint, like real routing) and varied
        // weights; token r's slot pattern differs from token r+1's so a row swap bites.
        let sel: Vec<u32> = (0..t * n_used)
            .map(|p| ((p * 5 + p / n_used) % n_expert) as u32)
            .collect();
        let w: Vec<f32> = (0..t * n_used)
            .map(|p| 0.1 + 0.03 * (p % 11) as f32)
            .collect();

        let want = moe_sequential_chain(
            &e,
            &slabs,
            &slabs_d,
            &macros,
            &z,
            &sel,
            &w,
            t,
            n_used,
            (in_f, n_ff),
            limit,
        );
        let got = moe_pairs_program(
            &e,
            &slabs,
            &slabs_d,
            &macros,
            &z,
            &sel,
            &w,
            t,
            n_used,
            (in_f, n_ff),
            limit,
            |_, _, _| {},
        );
        let diffs = bit_diffs(&got, &want);
        assert_eq!(
            diffs,
            0,
            "t={t}: pairs twins diverged from the sequential chain in {diffs}/{} outputs",
            t * in_f
        );
        println!("gate 4 PASS t={t}: {} outputs bit-identical", t * in_f);
    }

    // RED ARM 1 — row isolation: swapping token 0's and token 1's slot-0 pair entries
    // (pointer + scales) must bite; proves the compare sees cross-row mixing.
    let t = 4usize;
    let z = varied(t * in_f, 0x2A + t as u64, 2.0);
    let sel: Vec<u32> = (0..t * n_used)
        .map(|p| ((p * 5 + p / n_used) % n_expert) as u32)
        .collect();
    let w: Vec<f32> = (0..t * n_used)
        .map(|p| 0.1 + 0.03 * (p % 11) as f32)
        .collect();
    let want = moe_sequential_chain(
        &e,
        &slabs,
        &slabs_d,
        &macros,
        &z,
        &sel,
        &w,
        t,
        n_used,
        (in_f, n_ff),
        limit,
    );
    let swapped = moe_pairs_program(
        &e,
        &slabs,
        &slabs_d,
        &macros,
        &z,
        &sel,
        &w,
        t,
        n_used,
        (in_f, n_ff),
        limit,
        |ptrs, scl, n_pairs| {
            let (a, b) = (0usize, n_used); // (tok0, slot0) <-> (tok1, slot0)
            for plane in 0..3 {
                ptrs.swap(plane * n_pairs + a, plane * n_pairs + b);
                scl.swap(plane * n_pairs + a, plane * n_pairs + b);
            }
        },
    );
    let d = bit_diffs(&swapped, &want);
    assert!(
        d > 0,
        "swapped-pair red arm produced the sequential chain — the gate cannot see row mixing"
    );
    println!("gate 4 RED 1 bites: {d} outputs differ with swapped pair rows");

    // RED ARM 2 — dropped macro plane: gate/up scales forced to 1.0 must bite (the ~3e4x
    // fluent-garbage class, shrunk to the fixture's live plane).
    let dropped = moe_pairs_program(
        &e,
        &slabs,
        &slabs_d,
        &macros,
        &z,
        &sel,
        &w,
        t,
        n_used,
        (in_f, n_ff),
        limit,
        |_, scl, n_pairs| {
            for s in scl[..2 * n_pairs].iter_mut() {
                *s = 1.0;
            }
        },
    );
    let d = bit_diffs(&dropped, &want);
    assert!(
        d > 0,
        "dropped-macro red arm produced the sequential chain — the macro fold is untested"
    );
    println!("gate 4 RED 2 bites: {d} outputs differ with the macro plane dropped");
}
