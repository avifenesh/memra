//! glm5 MATVEC-EFFICIENCY door bit-gates (lane/glm5-matvec, 2026-08-31).
//!
//! The diet-battery census named the matvec classes (moe + bf16-mmv = 65% of ship-shape
//! GPU) and the spec loop's alloc churn as the remaining levers toward the 100 tok/s bar.
//! This file localizes each door's exactness claim at the kernel seam, red-armed (the
//! `_vl`-twin discipline):
//!
//! T. `MEMRA_BF16_TCOLS_WIDE` — `matvec_bf16_f32acc_x4_tcols16` (t=9..=16, the DFlash2
//!    drafter-head shape) vs the t=1 per-row program, bitwise; the rows_into route A/B
//!    (flag ON == flag OFF bytes, dispatch counter anchors engagement). Red: a shifted
//!    activation row must bite.
//! X. `MEMRA_BF16_TCOLS_X1` — the one-row-per-block tcols grid twin vs the x4 form,
//!    bitwise at t=2..=8. Red: swapped weight rows must bite.
//! R. `MEMRA_BF16_TCOLS_RED_FUSED` (lane/glm5-door-r) — the `_rf` fused-reduce-tail twins
//!    vs the standing tcols kernels, bitwise: both grid forms at t=2..=8, the tcols16 form
//!    at t=9..=16, and the t=1 degenerate bounds vs the per-row t=1 program. Red: the
//!    shifted-pairing twin (ascending shuffle offsets — a different association of the
//!    same 32 partials) must bite.
//! M. `MEMRA_MOE_VROWS_PACK` — the `_w4` warp-packed verify-rows MoE pair vs the unpacked
//!    pair AND vs the sequential per-(token,expert) slab chain, on minted NVFP4 banks
//!    with a LIVE macro plane. Reds: the vrest gate-4 swapped-pair and dropped-macro
//!    arms, re-bitten THROUGH the packed door.
//! K. `MEMRA_TOPK_SHARDS` — the exact two-launch shard split vs the standing
//!    `topk_rows_f32`, values bitwise + indices equal, on fixtures with PLANTED TIES
//!    across shard boundaries (the tie rule is the exactness claim). Red: a one-column
//!    logit shift must bite.
//!
//! Door W (`MEMRA_GLM5_VERIFY_WS`) is gated at the walk level in
//! `glm5_spec_session_gpu.rs` (`gpu_verify_ws_...`) — byte identity needs the real
//! session walk plus the `SCRATCH_ALLOC_CALLS` receipt, not a kernel seam.
//!
//! Rig law (exactness only, never timing):
//!   NVIDIA_TF32_OVERRIDE=0 flock /tmp/memra-5090.lock \
//!     cargo test -p memra-engine --test glm5_matvec_doors_gpu -- --ignored --test-threads=1

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

/// Run `f` with a door env var set to "1" (the doors read per call — the rollback seam —
/// so an in-process A/B is exact). Callers hold `gpu_guard`, which serializes the mutation.
fn with_flag<T>(key: &str, f: impl FnOnce() -> T) -> T {
    // SAFETY: serialized behind gpu_guard by every caller.
    unsafe { std::env::set_var(key, "1") };
    let r = f();
    unsafe { std::env::remove_var(key) };
    r
}

/// Run `f` with a door env var PINNED to "0" — the explicit OFF arm.
///
/// Doors T/X/K/W read `!= Ok("0")`, so they are DEFAULT ON since the 2026-08-31 mv-battery
/// flip: an OFF arm expressed by leaving the variable UNSET silently becomes an ON arm.
/// That is not hypothetical — the flip turned this file's three reference arms into
/// door-vs-itself comparisons: door T's route A/B failed loudly on its own dispatch-counter
/// assertion, while doors X and K passed VACUOUSLY (x1-vs-x1, sharded-vs-sharded). Door W's
/// gate in `glm5_spec_session_gpu` already pinned "0" and was unaffected.
///
/// The law this restores (owner, 2026-08-25 new-flags rule): the OFF arm of any flag is
/// pinned `=0`, never merely unset. A reference arm must name the program it is referencing.
fn without_flag<T>(key: &str, f: impl FnOnce() -> T) -> T {
    // SAFETY: serialized behind gpu_guard by every caller.
    unsafe { std::env::set_var(key, "0") };
    let r = f();
    unsafe { std::env::remove_var(key) };
    r
}

/// Run `f` with a door PINNED to "1" (`on`) or "0" (`!on`) — both arms explicit, per the
/// same law [`without_flag`] restores.
fn pinned<T>(key: &str, on: bool, f: impl FnOnce() -> T) -> T {
    if on {
        with_flag(key, f)
    } else {
        without_flag(key, f)
    }
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
// Door T — the wide-t tcols twin and its rows_into route.
// ---------------------------------------------------------------------------------------------

#[test]
#[ignore = "needs a CUDA device, run under flock /tmp/memra-5090.lock"]
fn gpu_tcols16_matches_per_row_t1_program_bitwise() {
    let _gpu = gpu_guard();
    force_true_f32();
    let e = Engine::new(0).expect("CUDA engine on device 0");
    // out_f = 133: a ragged 4-row block tail; in_f multiple of 8 (the kernels' bar).
    let (in_f, out_f) = (256usize, 133usize);
    let w_host = varied(out_f * in_f, 0xBF16, 2.0);
    let w = e.htod_bytes(&bf16_bytes(&w_host)).expect("weight upload");

    for t in 9..=16usize {
        let x_host = varied(t * in_f, 0x7C01 + t as u64, 2.0);
        let x = e.htod(&x_host).expect("activation upload");

        let mut y_t = e.uninit(t * out_f).expect("y_t");
        e.matvec_bf16_tcols16_into(&w, &x, &mut y_t, in_f, out_f, t)
            .expect("tcols16 launch");
        let got = e.dtoh(&y_t).expect("tcols16 readback");

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
            "t={t}: tcols16 twin diverged from the per-row t=1 program in {diffs}/{} outputs",
            t * out_f
        );
        println!("door T PASS t={t}: {} outputs bit-identical", t * out_f);
    }

    // RED ARM: shifting the activation by one row must bite — proves the compare above can
    // see a row-addressing bug in the 16-wide accumulator array.
    let t = 12usize;
    let x_host = varied((t + 1) * in_f, 0x0DD5, 2.0);
    let x_ok = e.htod(&x_host[..t * in_f]).expect("x ok");
    let x_shift = e.htod(&x_host[in_f..(t + 1) * in_f]).expect("x shifted");
    let mut y_ok = e.uninit(t * out_f).expect("y ok");
    let mut y_shift = e.uninit(t * out_f).expect("y shifted");
    e.matvec_bf16_tcols16_into(&w, &x_ok, &mut y_ok, in_f, out_f, t)
        .expect("ok launch");
    e.matvec_bf16_tcols16_into(&w, &x_shift, &mut y_shift, in_f, out_f, t)
        .expect("shifted launch");
    let diffs = bit_diffs(
        &e.dtoh(&y_ok).expect("ok back"),
        &e.dtoh(&y_shift).expect("shifted back"),
    );
    assert!(
        diffs > 0,
        "shifted-row red arm produced identical outputs — the gate cannot see row addressing"
    );
    println!("door T RED bites: {diffs} outputs differ with a one-row activation shift");

    // ROUTE A/B: matvec_bf16_rows_into with the door ON must produce the flag-off bytes
    // (the drafter-head route), and the engagement counter must move ONLY on the ON arm.
    let t = 15usize; // the DFlash2 block-head shape (block_size 16, nd = 15)
    let x_host = varied(t * in_f, 0xD00F, 2.0);
    let x = e.htod(&x_host).expect("route x");
    let mut y_off = e.uninit(t * out_f).expect("y off");
    let d0 = memra_engine::bf16_tcols_wide_dispatches();
    // OFF arm PINNED =0 (never merely unset): door T is default ON, so an unset var would
    // make this "reference" the door itself and the A/B below vacuous.
    without_flag("MEMRA_BF16_TCOLS_WIDE", || {
        e.matvec_bf16_rows_into(&w, &x, &mut y_off, in_f, out_f, t)
            .expect("off-arm launch");
    });
    assert_eq!(
        memra_engine::bf16_tcols_wide_dispatches(),
        d0,
        "flag-off arm moved the wide-tcols dispatch counter"
    );
    let mut y_on = e.uninit(t * out_f).expect("y on");
    with_flag("MEMRA_BF16_TCOLS_WIDE", || {
        e.matvec_bf16_rows_into(&w, &x, &mut y_on, in_f, out_f, t)
            .expect("on-arm launch");
    });
    assert!(
        memra_engine::bf16_tcols_wide_dispatches() > d0,
        "ON arm did not take the wide-tcols door (engagement counter flat)"
    );
    let diffs = bit_diffs(&e.dtoh(&y_on).expect("on"), &e.dtoh(&y_off).expect("off"));
    assert_eq!(
        diffs, 0,
        "MEMRA_BF16_TCOLS_WIDE route diverged from the _rows kernel in {diffs} outputs"
    );
    println!("door T route A/B PASS: t=15 ON == OFF bytes, counter engaged");
}

// ---------------------------------------------------------------------------------------------
// Door X — the one-row-per-block tcols grid twin.
// ---------------------------------------------------------------------------------------------

#[test]
#[ignore = "needs a CUDA device, run under flock /tmp/memra-5090.lock"]
fn gpu_x1_tcols_matches_x4_bitwise() {
    let _gpu = gpu_guard();
    force_true_f32();
    let e = Engine::new(0).expect("CUDA engine on device 0");
    let (in_f, out_f) = (256usize, 133usize);
    let w_host = varied(out_f * in_f, 0x0F1D, 2.0);
    let w = e.htod_bytes(&bf16_bytes(&w_host)).expect("weight upload");

    for t in 2..=8usize {
        let x_host = varied(t * in_f, 0x0A11 + t as u64, 2.0);
        let x = e.htod(&x_host).expect("activation upload");
        let mut y_x4 = e.uninit(t * out_f).expect("y x4");
        // REFERENCE arm PINNED =0: door X is default ON, so leaving the var unset would
        // compute this "x4 reference" through the x1 grid and compare x1 against x1.
        without_flag("MEMRA_BF16_TCOLS_X1", || {
            e.matvec_bf16_tcols_into(&w, &x, &mut y_x4, in_f, out_f, t)
                .expect("x4 launch");
        });
        let mut y_x1 = e.uninit(t * out_f).expect("y x1");
        let d0 = memra_engine::bf16_tcols_x1_dispatches();
        with_flag("MEMRA_BF16_TCOLS_X1", || {
            e.matvec_bf16_tcols_into(&w, &x, &mut y_x1, in_f, out_f, t)
                .expect("x1 launch");
        });
        assert!(
            memra_engine::bf16_tcols_x1_dispatches() > d0,
            "t={t}: the x1 door did not engage"
        );
        let diffs = bit_diffs(&e.dtoh(&y_x1).expect("x1"), &e.dtoh(&y_x4).expect("x4"));
        assert_eq!(
            diffs,
            0,
            "t={t}: x1 grid twin diverged from the x4 form in {diffs}/{} outputs",
            t * out_f
        );
        println!("door X PASS t={t}: {} outputs bit-identical", t * out_f);
    }

    // RED ARM: swapping two weight rows must bite in exactly those output rows — proves the
    // compare sees per-row addressing through the reshaped grid.
    let t = 4usize;
    let x_host = varied(t * in_f, 0x0A11, 2.0);
    let x = e.htod(&x_host).expect("red x");
    let mut w_swap = w_host.clone();
    for i in 0..in_f {
        w_swap.swap(i, in_f + i); // rows 0 <-> 1
    }
    let w2 = e.htod_bytes(&bf16_bytes(&w_swap)).expect("swapped weight");
    let mut y_a = e.uninit(t * out_f).expect("y a");
    let mut y_b = e.uninit(t * out_f).expect("y b");
    with_flag("MEMRA_BF16_TCOLS_X1", || {
        e.matvec_bf16_tcols_into(&w, &x, &mut y_a, in_f, out_f, t)
            .expect("a launch");
        e.matvec_bf16_tcols_into(&w2, &x, &mut y_b, in_f, out_f, t)
            .expect("b launch");
    });
    let diffs = bit_diffs(&e.dtoh(&y_a).expect("a"), &e.dtoh(&y_b).expect("b"));
    assert!(
        diffs > 0,
        "swapped-weight-row red arm produced identical outputs — the gate is vacuous"
    );
    println!("door X RED bites: {diffs} outputs differ with weight rows 0/1 swapped");
}

// ---------------------------------------------------------------------------------------------
// Door R — the fused-t reduce-tail twins (lane/glm5-door-r).
// ---------------------------------------------------------------------------------------------

#[test]
#[ignore = "needs a CUDA device, run under flock /tmp/memra-5090.lock"]
fn gpu_red_fused_tcols_matches_standing_tree_bitwise() {
    let _gpu = gpu_guard();
    force_true_f32();
    let e = Engine::new(0).expect("CUDA engine on device 0");
    // in_f = 2048: in_f/8 = 256 >= the default 128-thread block, so EVERY lane's partial is
    // nonzero and the pairing bar is live across the whole tree — with a short in_f the top
    // lanes hold +0.0f and a shifted association over zeros cannot round differently (a+0
    // is exact), which would make the red arm below vacuous. out_f = 133: ragged 4-row tail.
    let (in_f, out_f) = (2048usize, 133usize);
    let w_host = varied(out_f * in_f, 0xD00E, 2.0);
    let w = e.htod_bytes(&bf16_bytes(&w_host)).expect("weight upload");

    // Both grid forms at t=2..=8 through the REAL launcher: door R composes with door X
    // (grid choice first, tail twin second), so each form is compared against ITS OWN
    // standing tree. Every OFF/reference arm is PINNED =0, never merely unset (§without_flag).
    for t in 2..=8usize {
        let x_host = varied(t * in_f, 0x5EED + t as u64, 2.0);
        let x = e.htod(&x_host).expect("activation upload");
        for (x1_arm, form) in [(true, "x1"), (false, "x4")] {
            let mut y_ref = e.uninit(t * out_f).expect("y ref");
            let d0 = memra_engine::bf16_tcols_red_fused_dispatches();
            pinned("MEMRA_BF16_TCOLS_X1", x1_arm, || {
                without_flag("MEMRA_BF16_TCOLS_RED_FUSED", || {
                    e.matvec_bf16_tcols_into(&w, &x, &mut y_ref, in_f, out_f, t)
                        .expect("standing launch");
                });
            });
            assert_eq!(
                memra_engine::bf16_tcols_red_fused_dispatches(),
                d0,
                "t={t} {form}: the R=0 arm moved the fused-tail dispatch counter"
            );
            let mut y_rf = e.uninit(t * out_f).expect("y rf");
            pinned("MEMRA_BF16_TCOLS_X1", x1_arm, || {
                with_flag("MEMRA_BF16_TCOLS_RED_FUSED", || {
                    e.matvec_bf16_tcols_into(&w, &x, &mut y_rf, in_f, out_f, t)
                        .expect("rf launch");
                });
            });
            assert!(
                memra_engine::bf16_tcols_red_fused_dispatches() > d0,
                "t={t} {form}: the fused-tail door did not engage"
            );
            let diffs = bit_diffs(&e.dtoh(&y_rf).expect("rf"), &e.dtoh(&y_ref).expect("ref"));
            assert_eq!(
                diffs,
                0,
                "t={t} {form}: the _rf fused tail diverged from the standing tree in {diffs}/{} outputs",
                t * out_f
            );
            println!(
                "door R PASS t={t} {form}: {} outputs bit-identical",
                t * out_f
            );
        }
    }

    // The wide-t twin at t=9..=16 (the drafter head's t=15 = 135 barriers is in-range).
    for t in 9..=16usize {
        let x_host = varied(t * in_f, 0x16ED + t as u64, 2.0);
        let x = e.htod(&x_host).expect("wide activation upload");
        let mut y_ref = e.uninit(t * out_f).expect("y ref16");
        without_flag("MEMRA_BF16_TCOLS_RED_FUSED", || {
            e.matvec_bf16_tcols16_into(&w, &x, &mut y_ref, in_f, out_f, t)
                .expect("standing tcols16 launch");
        });
        let mut y_rf = e.uninit(t * out_f).expect("y rf16");
        let d0 = memra_engine::bf16_tcols_red_fused_dispatches();
        with_flag("MEMRA_BF16_TCOLS_RED_FUSED", || {
            e.matvec_bf16_tcols16_into(&w, &x, &mut y_rf, in_f, out_f, t)
                .expect("rf tcols16 launch");
        });
        assert!(
            memra_engine::bf16_tcols_red_fused_dispatches() > d0,
            "t={t}: the fused-tail door did not engage on the tcols16 route"
        );
        let diffs = bit_diffs(
            &e.dtoh(&y_rf).expect("rf16"),
            &e.dtoh(&y_ref).expect("ref16"),
        );
        assert_eq!(
            diffs,
            0,
            "t={t}: the _rf tcols16 tail diverged from the standing tree in {diffs}/{} outputs",
            t * out_f
        );
        println!(
            "door R PASS t={t} tcols16: {} outputs bit-identical",
            t * out_f
        );
    }

    // t=1 — the routed launchers refuse t<2, but the door-R bar covers t=1..=16: the three
    // _rf twins at the degenerate column-loop bounds vs the per-row t=1 program (the same
    // reference the door T gate names). Gate-only launcher; never a serving route.
    {
        let t = 1usize;
        let x_host = varied(in_f, 0x0071, 2.0);
        let x = e.htod(&x_host).expect("t1 activation");
        let mut y_ref = e.uninit(out_f).expect("y t1 ref");
        without_flag("MEMRA_BF16_TCOLS_WIDE", || {
            e.matvec_bf16_rows_into(&w, &x, &mut y_ref, in_f, out_f, 1)
                .expect("t=1 rows launch");
        });
        let want = e.dtoh(&y_ref).expect("t1 ref back");
        for kernel in [
            "matvec_bf16_f32acc_x1_tcols_rf",
            "matvec_bf16_f32acc_x4_tcols_rf",
            "matvec_bf16_f32acc_x4_tcols16_rf",
        ] {
            let mut y_rf = e.uninit(out_f).expect("y t1 rf");
            e.matvec_bf16_tcols_gate_kernel_into(kernel, &w, &x, &mut y_rf, in_f, out_f, t)
                .expect("t=1 rf gate launch");
            let got = e.dtoh(&y_rf).expect("t1 rf back");
            let diffs = bit_diffs(&got, &want);
            assert_eq!(
                diffs, 0,
                "t=1 {kernel}: diverged from the per-row t=1 program in {diffs}/{out_f} outputs"
            );
        }
        println!("door R PASS t=1: all three _rf twins match the per-row t=1 program");
    }

    // RED ARM — the pairing bar itself: the `_rf_redshift` twin runs the warp phase with
    // ASCENDING shuffle offsets (1,2,4,8,16) — the same 32 partials under a DIFFERENT
    // association. If the bit compare above could not see a pairing change, this arm would
    // pass silently; it must bite.
    let t = 4usize;
    let x_host = varied(t * in_f, 0x4ED0, 2.0);
    let x = e.htod(&x_host).expect("red x");
    let mut y_a = e.uninit(t * out_f).expect("y pair");
    let mut y_b = e.uninit(t * out_f).expect("y shifted");
    e.matvec_bf16_tcols_gate_kernel_into(
        "matvec_bf16_f32acc_x1_tcols_rf",
        &w,
        &x,
        &mut y_a,
        in_f,
        out_f,
        t,
    )
    .expect("rf launch");
    e.matvec_bf16_tcols_gate_kernel_into(
        "matvec_bf16_f32acc_x1_tcols_rf_redshift",
        &w,
        &x,
        &mut y_b,
        in_f,
        out_f,
        t,
    )
    .expect("redshift launch");
    let diffs = bit_diffs(&e.dtoh(&y_a).expect("a"), &e.dtoh(&y_b).expect("b"));
    assert!(
        diffs > 0,
        "shifted-pairing red arm produced identical outputs — the door-R bit bar cannot \
         see an association change and every PASS above is unproven"
    );
    println!("door R RED bites: {diffs} outputs differ under the shifted pairing");
}

// ---------------------------------------------------------------------------------------------
// Door M — the warp-packed verify-rows MoE pair (fixture family lifted from the vrest gate 4).
// ---------------------------------------------------------------------------------------------

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

/// The batched pairs program THROUGH THE LAUNCHERS (which carry door M): build the
/// plane-major tables from the slab bases, then the two rows launches. `mutate` plants the
/// red-arm corruptions.
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
fn gpu_moe_pack_matches_unpacked_pair_bitwise() {
    let _gpu = gpu_guard();
    force_true_f32();
    let e = Engine::new(0).expect("CUDA engine on device 0");
    // The vrest gate-4 shape class: NVFP4 block 64, live macro plane, biting clamp.
    let (in_f, n_ff) = (128usize, 64usize);
    let (n_expert, n_used) = (16usize, 8usize);
    let slabs = nvfp4_slab(&e, n_expert, n_ff, in_f, 0x6A7E);
    let slabs_d = nvfp4_slab(&e, n_expert, in_f, n_ff, 0xD003);
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
    let limit = 0.75f32;

    let mk_sel = |t: usize| -> (Vec<u32>, Vec<f32>) {
        (
            (0..t * n_used)
                .map(|p| ((p * 5 + p / n_used) % n_expert) as u32)
                .collect(),
            (0..t * n_used)
                .map(|p| 0.1 + 0.03 * (p % 11) as f32)
                .collect(),
        )
    };

    for t in 2..=8usize {
        let z = varied(t * in_f, 0x2A + t as u64, 2.0);
        let (sel, w) = mk_sel(t);
        let want = moe_pairs_program(
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
        let d0 = memra_engine::moe_vrows_pack_dispatches();
        let got = with_flag("MEMRA_MOE_VROWS_PACK", || {
            moe_pairs_program(
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
            )
        });
        assert!(
            memra_engine::moe_vrows_pack_dispatches() > d0,
            "t={t}: the pack door did not engage"
        );
        let diffs = bit_diffs(&got, &want);
        assert_eq!(
            diffs,
            0,
            "t={t}: the _w4 packed pair diverged from the unpacked pair in {diffs}/{} outputs",
            t * in_f
        );
        println!("door M PASS t={t}: {} outputs bit-identical", t * in_f);
    }

    // RED ARMS through the packed door — the vrest gate-4 corruptions must still bite with
    // the packing on (row isolation + macro plane), so the identity above is not vacuous.
    let t = 4usize;
    let z = varied(t * in_f, 0x2A + t as u64, 2.0);
    let (sel, w) = mk_sel(t);
    let want = with_flag("MEMRA_MOE_VROWS_PACK", || {
        moe_pairs_program(
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
        )
    });
    let swapped = with_flag("MEMRA_MOE_VROWS_PACK", || {
        moe_pairs_program(
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
        )
    });
    let d = bit_diffs(&swapped, &want);
    assert!(
        d > 0,
        "packed swapped-pair red arm produced identical outputs — row isolation untested"
    );
    println!("door M RED 1 bites: {d} outputs differ with swapped pair rows");
    let dropped = with_flag("MEMRA_MOE_VROWS_PACK", || {
        moe_pairs_program(
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
        )
    });
    let d = bit_diffs(&dropped, &want);
    assert!(
        d > 0,
        "packed dropped-macro red arm produced identical outputs — the macro fold is untested"
    );
    println!("door M RED 2 bites: {d} outputs differ with the macro plane dropped");
}

// ---------------------------------------------------------------------------------------------
// Door K — the sharded exact top-k.
// ---------------------------------------------------------------------------------------------

#[test]
#[ignore = "needs a CUDA device, run under flock /tmp/memra-5090.lock"]
fn gpu_topk_shards_match_standing_kernel_exactly() {
    let _gpu = gpu_guard();
    force_true_f32();
    let e = Engine::new(0).expect("CUDA engine on device 0");
    let (n_rows, n_cols, k) = (15usize, 20_000usize, 16usize);

    // Fixture with PLANTED HAZARDS — the tie rule is the exactness claim:
    //  row 0: a duplicated maximum straddling the shard boundary (idx 900 == idx 17_311);
    //  row 1: k+4 equal maxima scattered across shards (top-k must be the LOWEST indices);
    //  row 2: all-equal row (top-k must be columns 0..k-1 in order);
    //  row 3: -inf everywhere except k-1 finite values (fewer finite than k — the -inf
    //         entries themselves become tie-selected fillers at the LOWEST indices, the
    //         standing kernel's insertion semantics);
    //  rows 4..: varied noise.
    let mut l = varied(n_rows * n_cols, 0x70D0, 2.0);
    l[900] = 9.5;
    l[17_311] = 9.5;
    for j in 0..(k + 4) {
        l[n_cols + (j * 997 + 13)] = 7.25;
    }
    for c in 0..n_cols {
        l[2 * n_cols + c] = 1.0;
    }
    for c in 0..n_cols {
        l[3 * n_cols + c] = f32::NEG_INFINITY;
    }
    for j in 0..(k - 1) {
        l[3 * n_cols + j * 1000 + 500] = 0.5 + j as f32;
    }
    let l_d = e.htod(&l).expect("logits upload");

    // STANDING-KERNEL reference PINNED =0: door K is default ON and n_cols=20000 is above
    // its 16384 engagement threshold, so an unset var would route this "standing" reference
    // through the shard split and compare the sharded path against itself.
    let (v_off, i_off) = without_flag("MEMRA_TOPK_SHARDS", || {
        e.topk_rows(&l_d, n_rows, n_cols, k).expect("standing topk")
    });
    let d0 = memra_engine::topk_shards_dispatches();
    let (v_on, i_on) = with_flag("MEMRA_TOPK_SHARDS", || {
        e.topk_rows(&l_d, n_rows, n_cols, k).expect("sharded topk")
    });
    assert!(
        memra_engine::topk_shards_dispatches() > d0,
        "the shard door did not engage at n_cols=20000"
    );
    let vals_off = e.dtoh(&v_off).expect("v off");
    let vals_on = e.dtoh(&v_on).expect("v on");
    let idx_off = e.dtoh_u32(&i_off).expect("i off");
    let idx_on = e.dtoh_u32(&i_on).expect("i on");
    let vdiffs = bit_diffs(&vals_on, &vals_off);
    assert_eq!(vdiffs, 0, "sharded top-k values diverged in {vdiffs} slots");
    assert_eq!(idx_on, idx_off, "sharded top-k indices diverged");
    // Spot-assert the planted rows so the fixture itself is proven live:
    assert_eq!(idx_off[0], 900, "row-0 tie must resolve to the lower index");
    assert_eq!(
        &idx_off[2 * k..2 * k + 4],
        &[0, 1, 2, 3],
        "the all-equal row must select ascending leading columns"
    );
    assert_eq!(
        idx_off[k], 13,
        "row-1 scattered ties must start at the lowest index"
    );
    assert_eq!(
        idx_off[3 * k + k - 1],
        0,
        "the sparse -inf row's 16th slot must be the -inf tie at column 0"
    );
    assert_eq!(
        vals_off[3 * k + k - 1],
        f32::NEG_INFINITY,
        "the sparse -inf row's 16th value must be -inf"
    );
    println!(
        "door K PASS: {} rows x {k} slots value+index identical",
        n_rows
    );

    // Small-column fall-through: below the shard threshold the flag must NOT divert.
    let small = e.htod(&l[..n_rows * 1024]).expect("small logits");
    let d1 = memra_engine::topk_shards_dispatches();
    let _ = with_flag("MEMRA_TOPK_SHARDS", || {
        e.topk_rows(&small, n_rows, 1024, k).expect("small topk")
    });
    assert_eq!(
        memra_engine::topk_shards_dispatches(),
        d1,
        "the shard door engaged below its column threshold"
    );

    // RED ARM: rotating row 4's logits by one column must move its indices — proves the
    // equality compare above can see a column-addressing bug.
    let mut l_rot = l.clone();
    l_rot[4 * n_cols..5 * n_cols].rotate_right(1);
    let lr_d = e.htod(&l_rot).expect("rotated upload");
    let (_, i_rot) = with_flag("MEMRA_TOPK_SHARDS", || {
        e.topk_rows(&lr_d, n_rows, n_cols, k).expect("rotated topk")
    });
    let idx_rot = e.dtoh_u32(&i_rot).expect("i rot");
    assert_ne!(
        &idx_rot[4 * k..5 * k],
        &idx_off[4 * k..5 * k],
        "rotated-row red arm left row-4 indices unchanged — the gate is vacuous"
    );
    println!("door K RED bites: row-4 indices moved under a one-column rotation");
}

/// Door I (`MEMRA_MOE_VROWS_ILP`): the `_ilp` twins (loads of four, then two, groups per lane
/// hoisted ahead of their math) vs the shipped pair, BITWISE, at the served nsb classes — gate/up
/// at in_f 4096 (nsb 128: one four-deep round per lane) and down at in_f = n_ff 2048 (nsb 64: one
/// two-deep round), plus the small vrest shape (nsb 4: the single tail only) — alone and composed
/// with door M. Engagement is counted, and the swapped-pair red arm bites THROUGH the door.
#[test]
#[ignore = "needs a CUDA device, run under flock /tmp/memra-5090.lock"]
fn gpu_moe_ilp_matches_shipped_pair_bitwise() {
    let _gpu = gpu_guard();
    force_true_f32();
    let e = Engine::new(0).expect("CUDA engine on device 0");
    let (n_expert, n_used) = (16usize, 8usize);
    let limit = 0.75f32;
    let mk_sel = |t: usize| -> (Vec<u32>, Vec<f32>) {
        (
            (0..t * n_used)
                .map(|p| ((p * 5 + p / n_used) % n_expert) as u32)
                .collect(),
            (0..t * n_used)
                .map(|p| 0.1 + 0.03 * (p % 11) as f32)
                .collect(),
        )
    };
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
    for &(in_f, n_ff) in &[(128usize, 64usize), (4096, 2048)] {
        let slabs = nvfp4_slab(&e, n_expert, n_ff, in_f, 0x1D0 + in_f as u64);
        let slabs_d = nvfp4_slab(&e, n_expert, in_f, n_ff, 0xD1D + n_ff as u64);
        for t in [1usize, 3, 8] {
            let z = varied(t * in_f, 0x11 + t as u64 + in_f as u64, 2.0);
            let (sel, w) = mk_sel(t);
            let want = moe_pairs_program(
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
            for pack in [false, true] {
                let d0 = memra_engine::moe_vrows_ilp_dispatches();
                let got = with_flag("MEMRA_MOE_VROWS_ILP", || {
                    if pack {
                        with_flag("MEMRA_MOE_VROWS_PACK", || {
                            moe_pairs_program(
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
                            )
                        })
                    } else {
                        moe_pairs_program(
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
                        )
                    }
                });
                assert!(
                    memra_engine::moe_vrows_ilp_dispatches() >= d0 + 2,
                    "in_f={in_f} t={t} pack={pack}: the ILP door did not engage both launches"
                );
                let nan = got.iter().filter(|v| v.is_nan()).count();
                assert_eq!(
                    nan, 0,
                    "in_f={in_f} t={t} pack={pack}: the ILP twin poisoned {nan} outputs (qtype wiring)"
                );
                let diffs = bit_diffs(&got, &want);
                assert_eq!(
                    diffs,
                    0,
                    "in_f={in_f} n_ff={n_ff} t={t} pack={pack}: the _ilp pair diverged from the shipped pair in {diffs}/{} outputs",
                    t * in_f
                );
                println!(
                    "door I PASS in_f={in_f} n_ff={n_ff} t={t} pack={pack}: {} outputs bit-identical",
                    t * in_f
                );
            }
        }
        // RED ARM through the ILP door: swapped pair rows must still change the output.
        let t = 3usize;
        let z = varied(t * in_f, 0x33 + in_f as u64, 2.0);
        let (sel, w) = mk_sel(t);
        let want = with_flag("MEMRA_MOE_VROWS_ILP", || {
            moe_pairs_program(
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
            )
        });
        let swapped = with_flag("MEMRA_MOE_VROWS_ILP", || {
            moe_pairs_program(
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
                    let (a, b) = (0usize, n_used);
                    for plane in 0..3 {
                        ptrs.swap(plane * n_pairs + a, plane * n_pairs + b);
                        scl.swap(plane * n_pairs + a, plane * n_pairs + b);
                    }
                },
            )
        });
        let d = bit_diffs(&swapped, &want);
        assert!(
            d > 0,
            "in_f={in_f}: ILP swapped-pair red arm produced identical outputs"
        );
        println!("door I RED bites in_f={in_f}: {d} outputs differ with swapped pair rows");
    }
}
