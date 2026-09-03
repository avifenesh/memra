//! Gate/bench for `MEMRA_B200_DSA_SELECT`, the exact multi-CTA DSA k-pool selector
//! (lane/b200-dsa-select-20260903, research/b200-dsa-select-20260903/LANE.md).
//!
//! The shipped `memra_mla_kpool_select_kernel` grids `t_q` blocks, so plain decode runs it on
//! ONE CTA. The door replaces it with a six-launch parallel pipeline that computes the SAME
//! `select_k`-th smallest order key and runs the SAME membership test, so the emitted `idx`
//! plane must be BYTE-IDENTICAL, not merely close. This gate holds three bars:
//!
//! 1. **EXACTNESS.** For every score shape x context x `t_q`, the door's `idx` plane must equal
//!    the shipped kernel's bit for bit. The shapes are chosen to hit the paths that actually
//!    decide the answer, not just the easy one: heavy exact-0.0 ties (ReLU makes those ORDINARY,
//!    which is the whole reason the order key carries the pool index), a plane where EVERY
//!    finite score ties, a sparse plane with fewer finite pools than the budget (the rank clamp),
//!    and a fully `-INFINITY` plane (nothing selected, tail only).
//! 2. **RED ARM.** Before any of that counts, `memra_mla_kpool_select_dsa_redarm_f32` runs the
//!    same pipeline with the resolved threshold deliberately bumped and MUST produce a different
//!    plane. A gate that cannot fail is not a gate; this proves the byte comparison above is
//!    actually looking at the selection. If the red arm matches, the gate exits 1 no matter how
//!    green everything else is.
//! 3. **ANCHOR.** At a small shape the shipped radix kernel is also checked against the in-tree
//!    reference selector, so the chain the door is byte-compared to is itself pinned to the
//!    order definition rather than to itself.
//!
//! Timing is interleaved and, per the rig law, a hard REGRESSION bar only on an sm_100a build --
//! the class this door targets and the only hardware a 100a binary runs on. Elsewhere it prints
//! as a DIAGNOSTIC; the correctness bars stay hard on every device.
//!
//! usage: dsa-select-gate [device_ordinal] [rounds (default 5)]

use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use memra_engine::Engine;
use memra_engine::mla_ffi::{
    MLA_DSA_REGRESSION_MARGIN, MLA_DSA_SELECT_MIN_POOLS, memra_mla_kpool_select_ctas,
    memra_mla_kpool_select_dsa_f32, memra_mla_kpool_select_dsa_redarm_f32,
    memra_mla_kpool_select_f32, memra_mla_kpool_select_ref_f32, memra_mla_kpool_select_ws_ints,
    mla_dsa_select_engages,
};
use std::os::raw::c_void;
use std::sync::Arc;
use std::time::Instant;

const POOL: usize = 4;
const SELECT_K: usize = 512;
const WIDTH: usize = SELECT_K * POOL + POOL - 1;

/// Pool counts swept: `n_pools = t_kv / pool`, so these are 16k, 32k, 128k, 256k and 1M context.
const POOLS: [usize; 5] = [4_096, 8_192, 32_768, 65_536, 262_144];
const T_QS: [usize; 2] = [1, 4];

/// Times each exactness cell is re-run. The pipeline synchronises through last-CTA arrivals, so
/// a barrier bug there is a race whose failure mode is a silent wrong selection; repeating widens
/// the window. A net, not a proof -- see `memra_sel_last_arrival` in cu/mla_attn.cu.
const EXACT_REPEATS: usize = 20;

/// Is the TIMING bar a hard check on this build? Only on the class the door targets: `100a`
/// SASS cannot run anywhere but sm_100, so the build arch is an exact proxy. The sibling
/// `dsa-decode-gate` carries the same rule, and for the same reason -- the two machines this
/// programme has measured disagreed in both directions on arm choice.
const TIMING_IS_BINDING: bool = cfg!(memra_sm100_tcgen05);

/// Score-plane shapes. Each names a path that decides the answer.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Shape {
    /// ~40% exact +0.0 (the ReLU floor), the rest positive, a causal `-INFINITY` tail.
    Mixed,
    /// Every finite score is exactly +0.0: the threshold is decided entirely by pool index.
    AllTies,
    /// Fewer finite pools than `select_k`: exercises the rank clamp.
    Sparse,
    /// Nothing causally visible: selects nothing, emits only the tail.
    Empty,
}

impl Shape {
    const ALL: [Shape; 4] = [Shape::Mixed, Shape::AllTies, Shape::Sparse, Shape::Empty];
    fn name(self) -> &'static str {
        match self {
            Shape::Mixed => "mixed(40% zero ties)",
            Shape::AllTies => "all-ties(every finite score +0.0)",
            Shape::Sparse => "sparse(n_fin < select_k)",
            Shape::Empty => "empty(all -inf)",
        }
    }
}

fn scores(shape: Shape, t_q: usize, n_pools: usize, first_pos: usize, seed: u64) -> Vec<f32> {
    let mut s = seed | 1;
    let mut next = || {
        s = s
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (s >> 33) as u32
    };
    let mut v = vec![0f32; t_q * n_pools];
    for t in 0..t_q {
        // The scorer marks every pool whose last token is not visible with -INFINITY; reproduce
        // that horizon so the selectors see the plane they actually get in serving.
        let vis = ((first_pos + t + 1) / POOL).min(n_pools);
        for p in 0..n_pools {
            let cell = &mut v[t * n_pools + p];
            if p >= vis {
                *cell = f32::NEG_INFINITY;
                continue;
            }
            *cell = match shape {
                Shape::Empty => f32::NEG_INFINITY,
                Shape::AllTies => 0.0,
                Shape::Sparse => {
                    if next() % 512 == 0 {
                        (next() % 1000) as f32 / 1000.0
                    } else {
                        f32::NEG_INFINITY
                    }
                }
                Shape::Mixed => {
                    if next() % 5 < 2 {
                        0.0
                    } else {
                        (next() % 100_000) as f32 / 100_000.0
                    }
                }
            };
        }
    }
    v
}

type Stream = Arc<cudarc::driver::CudaStream>;

fn dp(s: &CudaSlice<f32>, stream: &Stream) -> *const f32 {
    s.device_ptr(stream).0 as *const f32
}
fn dpmi(s: &mut CudaSlice<i32>, stream: &Stream) -> *mut i32 {
    s.device_ptr_mut(stream).0 as *mut i32
}

fn bench(
    label: &str,
    rounds: usize,
    stream: &Stream,
    arms: &mut [&mut dyn FnMut() -> i32],
) -> Vec<f64> {
    for (i, a) in arms.iter_mut().enumerate() {
        for _ in 0..2 {
            let rc = a();
            assert_eq!(rc, 0, "{label} arm {i}: warmup rc={rc}");
            stream.synchronize().expect("sync");
        }
    }
    let mut acc = vec![0f64; arms.len()];
    for _ in 0..rounds {
        for (i, a) in arms.iter_mut().enumerate() {
            let t0 = Instant::now();
            let rc = a();
            stream.synchronize().expect("sync");
            acc[i] += t0.elapsed().as_secs_f64() * 1e6;
            assert_eq!(rc, 0, "{label} arm {i}: rc={rc}");
        }
    }
    acc.into_iter().map(|a| a / rounds as f64).collect()
}

#[allow(clippy::too_many_lines)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let dev: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(0);
    let rounds: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(5).max(1);
    let e = Engine::new(dev)?;
    let stream = e.stream();
    let cu = stream.cu_stream() as *mut c_void;

    println!(
        "dsa-select-gate: device {dev}, pool={POOL} select_k={SELECT_K} width={WIDTH}, \
         rounds={rounds}, engages at n_pools >= {MLA_DSA_SELECT_MIN_POOLS}, timing bar {}",
        if TIMING_IS_BINDING {
            "BINDING (sm_100a build)"
        } else {
            "DIAGNOSTIC ONLY (non-target build; correctness bars still hard)"
        }
    );

    let mut failures: Vec<String> = Vec::new();
    let red_arm_proved;
    let t_max = *T_QS.iter().max().unwrap();
    let pools_max = *POOLS.iter().max().unwrap();
    let ctas_max = unsafe { memra_mla_kpool_select_ctas(pools_max as i32) };
    let ws_stride_max = unsafe { memra_mla_kpool_select_ws_ints(ctas_max) } as usize;

    let mut idx_ship = e.uninit_i32(t_max * WIDTH)?;
    let mut idx_arm = e.uninit_i32(t_max * WIDTH)?;
    let mut idx_ref = e.uninit_i32(t_max * WIDTH)?;
    let mut ws = e.uninit_i32(t_max * ws_stride_max)?;

    // ---------------------------------------------------------------- ANCHOR: shipped vs ref
    {
        let n_pools = 4_096usize;
        let t = 1usize;
        let first_pos = n_pools * POOL - t;
        let h = scores(Shape::Mixed, t, n_pools, first_pos, 0x5E1E_0001);
        let sc = e.htod(&h)?;
        let (sp, ip, rp) = (
            dp(&sc, &stream),
            dpmi(&mut idx_ship, &stream),
            dpmi(&mut idx_ref, &stream),
        );
        let a = unsafe {
            memra_mla_kpool_select_f32(
                sp,
                ip,
                t as i32,
                n_pools as i32,
                POOL as i32,
                SELECT_K as i32,
                WIDTH as i32,
                first_pos as i32,
                1,
                cu,
            )
        };
        let b = unsafe {
            memra_mla_kpool_select_ref_f32(
                sp,
                rp,
                t as i32,
                n_pools as i32,
                POOL as i32,
                SELECT_K as i32,
                WIDTH as i32,
                first_pos as i32,
                1,
                cu,
            )
        };
        assert_eq!((a, b), (0, 0), "anchor launch");
        stream.synchronize()?;
        let (x, y) = (e.dtoh_i32(&idx_ship)?, e.dtoh_i32(&idx_ref)?);
        let same = x[..t * WIDTH] == y[..t * WIDTH];
        println!(
            "  anchor: shipped radix vs in-tree reference selector at {n_pools} pools: {}",
            if same { "IDENTICAL" } else { "MISMATCH" }
        );
        if !same {
            failures.push("ANCHOR shipped radix != reference selector".into());
        }
    }

    // ---------------------------------------------------------------- RED ARM, before anything
    {
        let n_pools = 65_536usize;
        let t = 1usize;
        let first_pos = n_pools * POOL - t;
        let h = scores(Shape::Mixed, t, n_pools, first_pos, 0x5E1E_0002);
        let sc = e.htod(&h)?;
        let sp = dp(&sc, &stream);
        let ip = dpmi(&mut idx_ship, &stream);
        let ap = dpmi(&mut idx_arm, &stream);
        let wp = dpmi(&mut ws, &stream);
        unsafe {
            memra_mla_kpool_select_f32(
                sp,
                ip,
                t as i32,
                n_pools as i32,
                POOL as i32,
                SELECT_K as i32,
                WIDTH as i32,
                first_pos as i32,
                1,
                cu,
            );
            memra_mla_kpool_select_dsa_redarm_f32(
                sp,
                ap,
                wp,
                t as i32,
                n_pools as i32,
                POOL as i32,
                SELECT_K as i32,
                WIDTH as i32,
                first_pos as i32,
                1,
                -1,
                cu,
            );
        }
        stream.synchronize()?;
        let (x, y) = (e.dtoh_i32(&idx_ship)?, e.dtoh_i32(&idx_arm)?);
        let red = x[..t * WIDTH] != y[..t * WIDTH];
        println!(
            "  RED ARM (threshold lowered by 1, always drops the threshold pool): {} -- the byte \
             comparison {} detect a wrong \
             selection",
            if red { "DIFFERS" } else { "MATCHED" },
            if red { "DOES" } else { "DOES NOT" }
        );
        red_arm_proved = red;
        if !red {
            failures.push(
                "RED ARM did not fail: a deliberately wrong threshold produced the shipped \
                 plane, so every exactness check below is vacuous"
                    .into(),
            );
        }
    }

    // ---------------------------------------------------------------- exactness + timing
    for &n_pools in &POOLS {
        println!("\n== n_pools {n_pools} (kv {} tokens) ==", n_pools * POOL);
        for &t in &T_QS {
            let first_pos = n_pools * POOL - t;
            for shape in Shape::ALL {
                let h = scores(
                    shape,
                    t,
                    n_pools,
                    first_pos,
                    0x5E1E_1000 + n_pools as u64 + t as u64,
                );
                let sc = e.htod(&h)?;
                let sp = dp(&sc, &stream);
                let ip = dpmi(&mut idx_ship, &stream);
                let ap = dpmi(&mut idx_arm, &stream);
                let wp = dpmi(&mut ws, &stream);
                let rc1 = unsafe {
                    memra_mla_kpool_select_f32(
                        sp,
                        ip,
                        t as i32,
                        n_pools as i32,
                        POOL as i32,
                        SELECT_K as i32,
                        WIDTH as i32,
                        first_pos as i32,
                        1,
                        cu,
                    )
                };
                assert_eq!(rc1, 0, "shipped launch rc at {n_pools}/{t}");
                stream.synchronize()?;
                let x = e.dtoh_i32(&idx_ship)?;
                // REPEATS. The pipeline synchronises its passes through last-CTA arrivals, so a
                // barrier bug there is a RACE whose failure mode is a silent wrong selection,
                // not a crash. One clean comparison says very little about that; repeating the
                // cell widens the window. It does NOT prove absence -- on an idle device the
                // window stays small and `racecheck` is shared-memory only. The ordering
                // argument written on `memra_sel_last_arrival` is the actual evidence; this is
                // a net, not a proof.
                let mut same = true;
                let mut diff = 0usize;
                for _ in 0..EXACT_REPEATS {
                    let rc2 = unsafe {
                        memra_mla_kpool_select_dsa_f32(
                            sp,
                            ap,
                            wp,
                            t as i32,
                            n_pools as i32,
                            POOL as i32,
                            SELECT_K as i32,
                            WIDTH as i32,
                            first_pos as i32,
                            1,
                            cu,
                        )
                    };
                    assert_eq!(rc2, 0, "door launch rc at {n_pools}/{t}");
                    stream.synchronize()?;
                    let y = e.dtoh_i32(&idx_arm)?;
                    let d = x[..t * WIDTH]
                        .iter()
                        .zip(&y[..t * WIDTH])
                        .filter(|(a, b)| a != b)
                        .count();
                    if d != 0 {
                        same = false;
                        diff = diff.max(d);
                    }
                }
                println!(
                    "  t_q={t} {:34} {}",
                    shape.name(),
                    if same {
                        "EXACT".to_string()
                    } else {
                        format!("MISMATCH (worst {diff}/{} slots)", t * WIDTH)
                    }
                );
                if !same {
                    failures.push(format!(
                        "EXACTNESS n_pools={n_pools} t_q={t} shape={} : {diff} slots differ \
                         (worst of {EXACT_REPEATS} repeats)",
                        shape.name()
                    ));
                }
            }

            // Timing on the realistic shape only.
            let h = scores(Shape::Mixed, t, n_pools, first_pos, 0x5E1E_2000);
            let sc = e.htod(&h)?;
            let sp = dp(&sc, &stream);
            let ip = dpmi(&mut idx_ship, &stream);
            let ap = dpmi(&mut idx_arm, &stream);
            let wp = dpmi(&mut ws, &stream);
            let mut ship = || unsafe {
                memra_mla_kpool_select_f32(
                    sp,
                    ip,
                    t as i32,
                    n_pools as i32,
                    POOL as i32,
                    SELECT_K as i32,
                    WIDTH as i32,
                    first_pos as i32,
                    1,
                    cu,
                )
            };
            let mut arm = || unsafe {
                memra_mla_kpool_select_dsa_f32(
                    sp,
                    ap,
                    wp,
                    t as i32,
                    n_pools as i32,
                    POOL as i32,
                    SELECT_K as i32,
                    WIDTH as i32,
                    first_pos as i32,
                    1,
                    cu,
                )
            };
            let us = bench("select", rounds, &stream, &mut [&mut ship, &mut arm]);
            let (a, b) = (us[0], us[1]);
            println!(
                "  t_q={t} us: shipped {a:.1}  parallel {b:.1}  ({:.2}x)   (N={rounds} interleaved)",
                a / b
            );
            if mla_dsa_select_engages(t, n_pools) && b > a * MLA_DSA_REGRESSION_MARGIN {
                let line = format!(
                    "kpool_select n_pools={n_pools} t_q={t}: {b:.1} us vs shipped {a:.1} us \
                     ({:.3}x, margin {MLA_DSA_REGRESSION_MARGIN:.2}x)",
                    b / a
                );
                if TIMING_IS_BINDING {
                    failures.push(format!("REGRESSION {line}"));
                } else {
                    println!("  DIAGNOSTIC (non-target build, not a failure): {line}");
                }
            }
            if !mla_dsa_select_engages(t, n_pools) {
                println!(
                    "  note: the policy does NOT engage here (n_pools < {MLA_DSA_SELECT_MIN_POOLS} \
                     or t_q > 8), so the row above is information, not a bar"
                );
            }
        }
    }

    println!();
    if failures.is_empty() && red_arm_proved {
        println!(
            "dsa-select-gate PASS: red arm failed as required, the shipped radix kernel matches \
             the in-tree reference, and the parallel selector is EXACT (byte-identical `idx`) at \
             every shape in {{mixed, all-ties, sparse, empty}} x n_pools {POOLS:?} x t_q {T_QS:?}. \
             Timing bar was {}.",
            if TIMING_IS_BINDING {
                "BINDING"
            } else {
                "diagnostic only"
            }
        );
        return Ok(());
    }
    for f in &failures {
        println!("{f}");
    }
    println!("dsa-select-gate FAIL: {} failing check(s)", failures.len());
    std::process::exit(1);
}
