//! Kernel gate for `moe_vrows_tables_from_sel` — the DEVICE build of the verify-rows MoE pair's
//! pointer/scale tables (door D, and the `MEMRA_GLM5_DECODE_GRAPH` T=1 arm that makes the walk
//! capturable at all).
//!
//! WHY IT EXISTS. Box run 5 (2026-09-02, 2x B200, real GLM-5.3-Flash artifact) traced the
//! decode-graph door's corruption to a segment the trace labels `eager-run` — i.e. with the door
//! ON but the capture NOT in use. That rules out capture/replay mechanics and leaves the door's
//! other enabler, this table build, which replaces the host router readback. The trace shows
//! absmax climbing to 1.36e19 at the first routed-MoE run and NaN five layers later, which is
//! what reading the wrong weight bytes looks like.
//!
//! WHAT THIS PROVES, and what it deliberately does NOT. It compares the DEVICE-built tables
//! against the HOST arithmetic they claim to reproduce (`hybrid_forward.rs::moe_vrows_pairs_q8`'s
//! `VrowsSel::Host` arm), pointer for pointer and scale BIT for bit. It needs no model and no
//! weights: the kernel only computes addresses, so the bases are synthetic and are never
//! dereferenced. That makes it runnable on the exactness-only rig, which is the whole point —
//! the pair is the only place the real artifact exists and it is not this lane's to burn.
//!
//! THE CASE THAT MATTERS IS THE BIG ONE. The serving posture is `MEMRA_MOE_RESIDENT_GB=130`, so
//! a per-projection expert slab spans tens of GB and `expert_stride * (n_expert - 1)` runs far
//! past 4 GiB. Any 32-bit step in that product wraps and the kernel reads a garbage row, which
//! is exactly the observed failure class — so the strides here are chosen to cross 4 GiB and
//! 8 GiB rather than to look realistic at fixture scale.
//!
//! Rig law: correctness-only, run under `flock /tmp/memra-5090.lock`,
//! `-- --ignored --test-threads=1`.

use memra_engine::Engine;

/// The host arithmetic the device kernel must reproduce, lifted verbatim from
/// `moe_vrows_pairs_q8`'s `VrowsSel::Host` arm so the two cannot drift apart silently.
fn host_tables(
    sel: &[i32],
    selw: &[f32],
    (pg, pu, pd): (u64, u64, u64),
    (sg, su, sd): (usize, usize, usize),
    macros: Option<(&[f32], &[f32], &[f32])>,
) -> (Vec<u64>, Vec<f32>) {
    let n_pairs = sel.len();
    let mut ptrs = vec![0u64; 3 * n_pairs];
    let mut scl = vec![0f32; 3 * n_pairs];
    for (p, (&ex, &w)) in sel.iter().zip(selw).enumerate() {
        let ex = ex as usize;
        ptrs[p] = pg + (ex * sg) as u64;
        ptrs[n_pairs + p] = pu + (ex * su) as u64;
        ptrs[2 * n_pairs + p] = pd + (ex * sd) as u64;
        let (mg, mu, md) = match macros {
            Some((g, u, d)) => (g[ex], u[ex], d[ex]),
            None => (1.0, 1.0, 1.0),
        };
        scl[p] = mg;
        scl[n_pairs + p] = mu;
        scl[2 * n_pairs + p] = w * md;
    }
    (ptrs, scl)
}

#[allow(clippy::too_many_arguments)] // allow: one case knob per axis the gate varies; a struct would hide them at the call site
fn check_case(
    e: &Engine,
    label: &str,
    n_expert: usize,
    sel: &[i32],
    selw: &[f32],
    bases: (u64, u64, u64),
    strides: (usize, usize, usize),
    with_macros: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let n_pairs = sel.len();
    // Macro planes: distinct per plane and per expert, so a plane mix-up or an index slip cannot
    // land on the right value by accident.
    let mg: Vec<f32> = (0..n_expert).map(|i| 1.0 + i as f32 * 1e-3).collect();
    let mu: Vec<f32> = (0..n_expert).map(|i| 2.0 + i as f32 * 1e-3).collect();
    let md: Vec<f32> = (0..n_expert).map(|i| 3.0 + i as f32 * 1e-3).collect();
    let macros = with_macros.then_some((mg.as_slice(), mu.as_slice(), md.as_slice()));

    let (want_ptrs, want_scl) = host_tables(sel, selw, bases, strides, macros);

    let sel_d = e.htod_i32(sel)?;
    let selw_d = e.htod(selw)?;
    let mut ptrs_d = e.htod_u64(&vec![0u64; 3 * n_pairs])?;
    let mut scl_d = e.htod(&vec![0f32; 3 * n_pairs])?;
    // `il` differs per case so the launcher's per-(layer, plane) macro mirror cache is exercised
    // rather than reused across cases with different plane contents.
    let il = if with_macros { 7u16 } else { 9u16 };
    e.moe_vrows_tables_from_sel(
        &sel_d,
        &selw_d,
        il,
        macros,
        bases,
        strides,
        n_pairs,
        &mut ptrs_d,
        &mut scl_d,
    )?;
    let got_ptrs = e.dtoh_u64(&ptrs_d)?;
    let got_scl = e.dtoh(&scl_d)?;

    let mut bad = 0usize;
    for i in 0..3 * n_pairs {
        if got_ptrs[i] != want_ptrs[i] {
            if bad < 6 {
                let plane = ["gate", "up", "down"][i / n_pairs];
                let p = i % n_pairs;
                println!(
                    "  [{label}] PTR MISMATCH plane={plane} pair={p} expert={} host=0x{:x} dev=0x{:x} (delta={})",
                    sel[p],
                    want_ptrs[i],
                    got_ptrs[i],
                    got_ptrs[i].wrapping_sub(want_ptrs[i]) as i64
                );
            }
            bad += 1;
        }
        // BITS, not approximate equality: a scale that differs in the last ulp is a different
        // numeric program, and the door's whole claim is that these two arms are one program.
        if got_scl[i].to_bits() != want_scl[i].to_bits() {
            if bad < 6 {
                let plane = ["gate", "up", "down"][i / n_pairs];
                println!(
                    "  [{label}] SCL MISMATCH plane={plane} pair={} host={:e} dev={:e}",
                    i % n_pairs,
                    want_scl[i],
                    got_scl[i]
                );
            }
            bad += 1;
        }
    }
    if bad > 0 {
        return Err(format!("{label}: {bad}/{} table entries differ", 6 * n_pairs).into());
    }
    println!(
        "  [{label}] OK ({} pointers + {} scales identical)",
        3 * n_pairs,
        3 * n_pairs
    );
    Ok(())
}

#[test]
#[ignore = "needs a CUDA device — run under flock /tmp/memra-5090.lock"]
fn dev_built_vrows_tables_match_the_host_arithmetic() -> Result<(), Box<dyn std::error::Error>> {
    let e = Engine::new(0)?;
    println!("[vrows-dev-tables] GPU0: {}", e.ctx().name()?);
    let n_expert = 256usize;
    // Experts spread to the top of the bank: the last one is what makes `ex * stride` large.
    let sel: Vec<i32> = vec![0, 1, 17, 63, 128, 200, 254, 255];
    let selw: Vec<f32> = vec![0.5, 0.25, 0.125, 1.0, 0.0625, 0.75, 0.375, 0.875];
    let bases = (
        0x0000_7f00_0000_0000u64,
        0x0000_7f40_0000_0000u64,
        0x0000_7f80_0000_0000u64,
    );

    // 1. Fixture-scale strides: everything fits in 32 bits, so this arm passes even on a build
    //    with a 32-bit product. It is the control, not the finding.
    check_case(
        &e,
        "small-strides",
        n_expert,
        &sel,
        &selw,
        bases,
        (860_160, 860_160, 1_114_112),
        true,
    )?;

    // 2. SERVING-SCALE strides. `MEMRA_MOE_RESIDENT_GB=130` puts tens of GB of experts on each
    //    stage: at 24 MiB per expert row the last expert sits 6 GiB into the gate slab and
    //    12 GiB into the down slab. Any 32-bit step in `base + ex*stride` wraps here.
    let big = (24 << 20, 24 << 20, 48 << 20);
    check_case(
        &e,
        "serving-strides-past-4GiB",
        n_expert,
        &sel,
        &selw,
        bases,
        big,
        true,
    )?;

    // 3. The macro-free bank: `macro_scale` answers 1.0 and the kernel must not dereference the
    //    aliased placeholder planes.
    check_case(&e, "no-macros", n_expert, &sel, &selw, bases, big, false)?;

    // 4. Every expert selected once, at serving strides — catches an index slip that a short
    //    hand-picked selection can miss.
    let sel_all: Vec<i32> = (0..8i32).map(|j| j * 32 + 31).collect();
    let selw_all: Vec<f32> = (0..8).map(|j| 1.0 / (j as f32 + 3.0)).collect();
    check_case(
        &e,
        "spread-selection",
        n_expert,
        &sel_all,
        &selw_all,
        bases,
        big,
        true,
    )?;
    Ok(())
}
