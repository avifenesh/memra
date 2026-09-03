//! Gate/bench for `MEMRA_B200_DSA_DECODE`, the t<=8 depth-oriented MLA/DSA decode door
//! (lane/b200-dsa-decode-20260902, research/b200-dsa-decode-20260902/ROOFLINE.md).
//!
//! It sweeps CONTEXT, not just query width, because the two stages this door rewrites fail in
//! opposite ways with depth: `attn_gathered` is depth-FLAT (n_slots is pinned at the DSA top-k
//! budget) and `kpool_score` is depth-LINEAR (`n_pools = t_kv / pool`, and no score survives a
//! decode step because `q_t` is new every token). A gate that only measured one context would
//! miss the whole reason the 1M window slides from 31.1 to 22.7 tok/s.
//!
//! Three checks, all hard:
//!
//! 1. BIT-IDENTITY, for the arms that claim it. `memra_mla_attn_gathered_dsa_f32` and
//!    `memra_mla_kpool_score_dsa_f32` must equal the shipped kernels bit for bit at every
//!    (context, t_q). Both are constructions (same fold order / same six-step rounding
//!    sequence, see the cu/mla_attn.cu headers), and this asserts them rather than trusting
//!    them.
//! 2. NUMERIC CLASS `dsa-warp-online-f32`, for the warp-online gathered arm, which is NOT
//!    bit-identical and never claims to be: an ARGMAX gate over every (token, head) latent row
//!    against the shipped kernel, plus a reported maxdiff and max relative error. A single
//!    argmax move at a width where the door CAN serve the class
//!    (`t_q <= MLA_DSA_NAMED_CLASS_T_MAX`) fails the gate. Above that width the arm is still
//!    measured and any move is still printed, but it is not a failure, because
//!    `mla_dsa_attn_arm_effective` demotes arm >= 2 to the shipped kernel there -- that width
//!    rule exists precisely BECAUSE the 2026-09-03 B200 run moved 1 of 256 rows at
//!    kv=131072 / t_q=4. Keeping the measurement is how a future proposal to raise the rule
//!    arrives with evidence instead of an inference from t_q=1.
//! 3. REGRESSION. At every (context, t_q) the arm the SERVING POLICY would select
//!    (`mla_ffi::mla_dsa_attn_arm` for the gathered stage, `MLA_DSA_SCORE_MIN_POOLS` for the
//!    scorer) may not be slower than the kernel it replaces by more than
//!    `MLA_DSA_REGRESSION_MARGIN`. Policy and gate read the same constants, so they cannot
//!    drift apart.
//!
//! Timing is interleaved: round `r` runs every arm back to back, `rounds` rounds, so clock
//! drift lands on every arm equally. Each sample is one launch bracketed by
//! `stream.synchronize()` (launch-to-completion), with every buffer preallocated outside the
//! timed region.
//!
//! RIG LAW (docs/PERFORMANCE.md): this bin's timing is a serving-relevant receipt only on the
//! B200 class the door targets; anywhere else it is a diagnostic. The correctness checks are
//! valid on any CUDA device -- the gate calls the arms through their raw FFI, bypassing the
//! sm_100a-gated door, so a 120a build still proves the KERNELS.
//!
//! usage: dsa-decode-gate [device_ordinal] [rounds (default 5)]

use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use memra_engine::Engine;
use memra_engine::mla_ffi::{
    MLA_DSA_ATTN_CHUNK_SWEEP, MLA_DSA_NAMED_CLASS_T_MAX, MLA_DSA_REGRESSION_MARGIN,
    MLA_DSA_SCORE_MIN_POOLS, memra_mla_attn_gathered_dsa_f32, memra_mla_attn_gathered_f32,
    memra_mla_dsa_attn_split_f32, memra_mla_kpool_score_dsa_f32, memra_mla_kpool_score_f32,
    mla_dsa_attn_arm, mla_dsa_attn_arm_effective,
};
use std::os::raw::c_void;
use std::sync::Arc;
use std::time::Instant;

// glm5_next / GLM-5.3-Flash serving geometry, read off the checkout (see ROOFLINE.md §0).
const N_HEAD: usize = 64;
const KV_RANK: usize = 512;
const D_ROPE: usize = 0;
const N_SLOTS: usize = 2048;
const IDX_HEADS: usize = 32;
const IDX_D: usize = 128;
const POOL: usize = 4;
const SCALE: f32 = 1.0 / 23.323_808; // 1/sqrt(d_nope + d_rope), d_nope=256 / d_rope=0

/// Contexts swept, in tokens.
const KVS: [usize; 5] = [2_048, 32_768, 131_072, 262_144, 1_048_576];
/// Query widths: plain decode and the DFlash2 spec-verify shape.
const T_QS: [usize; 2] = [1, 4];

/// Latent-cache rows actually allocated. The gathered attention is depth-FLAT by construction
/// (it always walks exactly `n_slots` gathered rows), so what a deeper cache would change for
/// it is only the TLB/L2 footprint the gather misses into -- and 262144 rows is already
/// 512 MB, several times the die's L2, so every gather is a cold stream at every context on
/// this ladder. Both arms see the identical window, so the comparison is unbiased. The
/// SCORER, which is the genuinely depth-linear stage, gets its real `n_pools` at every context
/// up to 1M (262144 pools, a 134 MB key plane). Overridable as argv[3].
const CACHE_ROWS_DEFAULT: usize = 262_144;

fn randf(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((s >> 33) as u32 as f32) / (u32::MAX as f32 / 2.0) - 1.0
        })
        .collect()
}

fn bits_equal(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

/// Per-row argmax over `row_len`-wide rows: the decisive check for the split arm, since the
/// latent row is what `decompress_v` and everything downstream consume.
fn argmax_rows(v: &[f32], row_len: usize) -> Vec<usize> {
    v.chunks(row_len)
        .map(|r| {
            let mut best = 0usize;
            for (i, x) in r.iter().enumerate() {
                if *x > r[best] {
                    best = i;
                }
            }
            best
        })
        .collect()
}

fn maxdiff(a: &[f32], b: &[f32]) -> (f64, f64) {
    let mut abs = 0f64;
    let mut rel = 0f64;
    for (x, y) in a.iter().zip(b) {
        let d = (*x as f64 - *y as f64).abs();
        abs = abs.max(d);
        let den = (*x as f64).abs().max((*y as f64).abs());
        if den > 0.0 {
            rel = rel.max(d / den);
        }
    }
    (abs, rel)
}

type Stream = Arc<cudarc::driver::CudaStream>;

fn dp(s: &CudaSlice<f32>, stream: &Stream) -> *const f32 {
    s.device_ptr(stream).0 as *const f32
}
fn dpm(s: &mut CudaSlice<f32>, stream: &Stream) -> *mut f32 {
    s.device_ptr_mut(stream).0 as *mut f32
}
fn dpi(s: &CudaSlice<i32>, stream: &Stream) -> *const i32 {
    s.device_ptr(stream).0 as *const i32
}

/// Warm every arm twice (unmeasured), then `rounds` INTERLEAVED rounds: round `r` runs arm 0,
/// arm 1, ... back to back, so box clock drift lands on every arm equally (the interleaved-A/B
/// protocol law). Returns the mean microseconds per launch, one entry per arm, in order.
fn bench_interleaved(
    labels: &[&str],
    rounds: usize,
    stream: &Stream,
    arms: &mut [&mut dyn FnMut() -> i32],
) -> Vec<f64> {
    for (i, arm) in arms.iter_mut().enumerate() {
        for _ in 0..2 {
            let rc = arm();
            assert_eq!(rc, 0, "{}: warmup launch rc={rc}", labels[i]);
            stream.synchronize().expect("sync");
        }
    }
    let mut acc = vec![0f64; arms.len()];
    for _ in 0..rounds {
        for (i, arm) in arms.iter_mut().enumerate() {
            let t0 = Instant::now();
            let rc = arm();
            stream.synchronize().expect("sync");
            acc[i] += t0.elapsed().as_secs_f64() * 1e6;
            assert_eq!(rc, 0, "{}: launch rc={rc}", labels[i]);
        }
    }
    acc.into_iter().map(|a| a / rounds as f64).collect()
}

/// Is the TIMING bar a hard check on this build? Only on the class the door targets. `100a`
/// SASS cannot run anywhere but sm_100, so the build arch is an exact proxy for "this process is
/// on a B200-class device". Elsewhere a regression line still prints, as a DIAGNOSTIC, because a
/// laptop 5090's microseconds were never a receipt for a B200-only door and the two machines have
/// already disagreed in BOTH directions on this lane (the single-pass arm is a 30% loss on the
/// 5090 and a 3.5-7.4% win on the B200; 16 chunks beat 32 on the 5090 and lost to it on the
/// B200). CORRECTNESS bars -- bit identity and the argmax gate -- stay hard on every device.
const TIMING_IS_BINDING: bool = cfg!(memra_sm100_tcgen05);

/// One failed check, collected and printed together at the end.
struct Failure(String);

#[allow(clippy::too_many_lines)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let dev: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(0);
    let rounds: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(5).max(1);
    let cache_rows: usize = args
        .next()
        .and_then(|s| s.parse().ok())
        .unwrap_or(CACHE_ROWS_DEFAULT)
        .max(N_SLOTS);

    let e = Engine::new(dev)?;
    let stream = e.stream();
    let width = KV_RANK + D_ROPE;
    let t_max = *T_QS.iter().max().unwrap();
    let n_pools_max = KVS.iter().max().unwrap() / POOL;

    println!(
        "dsa-decode-gate: device {dev}, geometry nh={N_HEAD} kv_rank={KV_RANK} d_rope={D_ROPE} \
         n_slots={N_SLOTS} idx_heads={IDX_HEADS} idx_d={IDX_D} pool={POOL}, cache_rows={cache_rows} \
         ({} MB), rounds={rounds}",
        cache_rows * width * 4 / (1024 * 1024)
    );
    println!(
        "  policy: attn_gathered arm table {}  (0 = shipped, 1 = single-pass bit-identical, \
          n>=2 = warp-online with n chunks; a cell in parentheses is demoted by the width rule); \
         named class admissible only at t_q <= {MLA_DSA_NAMED_CLASS_T_MAX}; scorer engages at \
         n_pools >= {MLA_DSA_SCORE_MIN_POOLS}; regression margin \
         {MLA_DSA_REGRESSION_MARGIN:.2}x, {}",
        (1..=8)
            .map(|t| {
                let (raw, eff) = (mla_dsa_attn_arm(t), mla_dsa_attn_arm_effective(t));
                if raw == eff {
                    format!("t{t}={eff}")
                } else {
                    format!("t{t}={eff}(was {raw})")
                }
            })
            .collect::<Vec<_>>()
            .join(" "),
        if TIMING_IS_BINDING {
            "TIMING BINDING (sm_100a build, the door's target class)"
        } else {
            "timing DIAGNOSTIC ONLY (non-target build; correctness bars still hard)"
        }
    );

    // ---------------------------------------------------------------- shared device state
    let cache = e.htod(&randf(cache_rows * width, 0xD5A0_0001))?;
    let pool_keys = e.htod(&randf(n_pools_max * IDX_D, 0xD5A0_0002))?;
    let q_lat = e.htod(&randf(t_max * N_HEAD * KV_RANK, 0xD5A0_0003))?;
    let q_pe = e.htod(&randf((t_max * N_HEAD * D_ROPE).max(1), 0xD5A0_0004))?;
    let q_index = e.htod(&randf(t_max * IDX_HEADS * IDX_D, 0xD5A0_0005))?;
    // Head weights are a softmax-shaped non-negative plane in serving; keep them non-negative so
    // the ReLU tie structure the selection depends on is exercised the way it is in production.
    let hw_h: Vec<f32> = randf(t_max * IDX_HEADS, 0xD5A0_0006)
        .into_iter()
        .map(f32::abs)
        .collect();
    let hw = e.htod(&hw_h)?;

    let mut out_ship = e.uninit(t_max * N_HEAD * KV_RANK)?;
    let mut out_arm = e.uninit(t_max * N_HEAD * KV_RANK)?;
    let chunks_max = *MLA_DSA_ATTN_CHUNK_SWEEP.iter().max().unwrap() as usize;
    let mut part_m = e.uninit(t_max * N_HEAD * chunks_max)?;
    let mut part_d = e.uninit(t_max * N_HEAD * chunks_max)?;
    let mut part_acc = e.uninit(t_max * N_HEAD * chunks_max * KV_RANK)?;
    let mut score_ship = e.uninit(t_max * n_pools_max)?;
    let mut score_arm = e.uninit(t_max * n_pools_max)?;

    let cache_p = dp(&cache, &stream);
    let pool_keys_p = dp(&pool_keys, &stream);
    let q_lat_p = dp(&q_lat, &stream);
    let q_pe_p = dp(&q_pe, &stream);
    let q_index_p = dp(&q_index, &stream);
    let hw_p = dp(&hw, &stream);
    let out_ship_p = dpm(&mut out_ship, &stream);
    let out_arm_p = dpm(&mut out_arm, &stream);
    let part_m_p = dpm(&mut part_m, &stream);
    let part_d_p = dpm(&mut part_d, &stream);
    let part_acc_p = dpm(&mut part_acc, &stream);
    let score_ship_p = dpm(&mut score_ship, &stream);
    let score_arm_p = dpm(&mut score_arm, &stream);
    let cu = stream.cu_stream() as *mut c_void;

    let mut failures: Vec<Failure> = Vec::new();

    for &kv in &KVS {
        let n_pools = kv / POOL;
        // Queries are the last t_q rows of the cache; the gathered list is drawn from the
        // allocated window, which both arms share.
        let window = cache_rows.min(kv);
        println!("\n== context {kv} tokens (n_pools {n_pools}, gather window {window} rows) ==");

        for &t in &T_QS {
            // A shape-faithful gathered list: ascending, distinct, spread across the window,
            // with a tail of -1 padding on the last query the way the selector emits it.
            let mut idx_h = vec![-1i32; t * N_SLOTS];
            for (qi, row) in idx_h.chunks_mut(N_SLOTS).enumerate() {
                let stride = (window / N_SLOTS).max(1);
                let filled = N_SLOTS - qi; // one fewer valid slot per query: exercises the -1 path
                for (s, cell) in row.iter_mut().enumerate().take(filled) {
                    *cell = ((s * stride + qi) % window) as i32;
                }
            }
            let idx = e.htod_i32(&idx_h)?;
            let idx_p = dpi(&idx, &stream);
            let first_pos = kv.saturating_sub(t);

            // ------------------------------------------------------------ attn_gathered
            let ship = || unsafe {
                memra_mla_attn_gathered_f32(
                    q_lat_p,
                    q_pe_p,
                    cache_p,
                    idx_p,
                    out_ship_p,
                    N_HEAD as i32,
                    KV_RANK as i32,
                    D_ROPE as i32,
                    t as i32,
                    N_SLOTS as i32,
                    SCALE,
                    cu,
                )
            };
            let dsa = || unsafe {
                memra_mla_attn_gathered_dsa_f32(
                    q_lat_p,
                    q_pe_p,
                    cache_p,
                    idx_p,
                    out_arm_p,
                    N_HEAD as i32,
                    KV_RANK as i32,
                    D_ROPE as i32,
                    t as i32,
                    N_SLOTS as i32,
                    SCALE,
                    cu,
                )
            };
            assert_eq!(ship(), 0, "attn_gathered shipped launch");
            stream.synchronize()?;
            let ref_out = e.dtoh(&out_ship)?[..t * N_HEAD * KV_RANK].to_vec();
            let rc = dsa();
            assert_eq!(rc, 0, "attn_gathered dsa launch rc={rc}");
            stream.synchronize()?;
            let dsa_out = e.dtoh(&out_arm)?[..t * N_HEAD * KV_RANK].to_vec();
            let identical = bits_equal(&ref_out, &dsa_out);
            println!(
                "  t_q={t} attn_gathered single-pass: {}",
                if identical {
                    "BIT-IDENTICAL"
                } else {
                    "MISMATCH"
                }
            );
            if !identical {
                failures.push(Failure(format!(
                    "MISMATCH attn_gathered single-pass kv={kv} t_q={t}: claims bit identity"
                )));
            }

            let ref_arg = argmax_rows(&ref_out, KV_RANK);
            let mut split_us: Vec<(i32, f64)> = Vec::new();
            let mut split_chunks: Vec<i32> = Vec::new();
            for &c in &MLA_DSA_ATTN_CHUNK_SWEEP {
                if c <= 1 {
                    continue;
                }
                let run = || unsafe {
                    memra_mla_dsa_attn_split_f32(
                        q_lat_p,
                        q_pe_p,
                        cache_p,
                        idx_p,
                        out_arm_p,
                        part_m_p,
                        part_d_p,
                        part_acc_p,
                        N_HEAD as i32,
                        KV_RANK as i32,
                        D_ROPE as i32,
                        t as i32,
                        N_SLOTS as i32,
                        c,
                        SCALE,
                        cu,
                    )
                };
                let rc = run();
                assert_eq!(rc, 0, "attn_gathered warp-online chunks={c} launch rc={rc}");
                stream.synchronize()?;
                let arm_out = e.dtoh(&out_arm)?[..t * N_HEAD * KV_RANK].to_vec();
                let arg = argmax_rows(&arm_out, KV_RANK);
                let moved = ref_arg.iter().zip(&arg).filter(|(a, b)| a != b).count();
                let (abs, rel) = maxdiff(&ref_out, &arm_out);
                println!(
                    "  t_q={t} attn_gathered warp-online chunks={c} [dsa-warp-online-f32]: \
                     argmax {} ({moved}/{} rows moved), maxdiff {abs:.3e}, max-rel {rel:.3e}",
                    if moved == 0 { "MATCH" } else { "MOVED" },
                    ref_arg.len()
                );
                // The argmax bar is HARD exactly where the class can be served. Above
                // MLA_DSA_NAMED_CLASS_T_MAX the door demotes arm >= 2 to the shipped kernel
                // (`mla_dsa_attn_arm_effective`), so a move there is a recorded property of the
                // class, not a shipping defect -- and it is the reason the width rule exists:
                // the 2026-09-03 B200 run moved 1 of 256 rows at kv=131072 / t_q=4 for every
                // swept chunk count. The arm keeps being MEASURED at those widths so the day
                // someone proposes raising the rule, the evidence is already on the table.
                if moved != 0 {
                    if t <= MLA_DSA_NAMED_CLASS_T_MAX {
                        failures.push(Failure(format!(
                            "ARGMAX attn_gathered warp-online kv={kv} t_q={t} chunks={c}: \
                             {moved} rows moved at a width where the class IS servable"
                        )));
                    } else {
                        println!(
                            "    INFO: not a failure -- t_q={t} is above the named-class width \
                             rule ({MLA_DSA_NAMED_CLASS_T_MAX}), so the door demotes this arm to \
                             the shipped kernel here. This is why that rule exists."
                        );
                    }
                }
                split_chunks.push(c);
            }

            // One interleaved sweep over shipped, single-pass and every swept chunk count.
            let mut ship_m = ship;
            let mut dsa_m = dsa;
            let mut split_runs: Vec<Box<dyn FnMut() -> i32>> = split_chunks
                .iter()
                .map(|&c| {
                    let f: Box<dyn FnMut() -> i32> = Box::new(move || unsafe {
                        memra_mla_dsa_attn_split_f32(
                            q_lat_p,
                            q_pe_p,
                            cache_p,
                            idx_p,
                            out_arm_p,
                            part_m_p,
                            part_d_p,
                            part_acc_p,
                            N_HEAD as i32,
                            KV_RANK as i32,
                            D_ROPE as i32,
                            t as i32,
                            N_SLOTS as i32,
                            c,
                            SCALE,
                            cu,
                        )
                    });
                    f
                })
                .collect();
            let mut arms: Vec<&mut dyn FnMut() -> i32> = vec![&mut ship_m, &mut dsa_m];
            for r in split_runs.iter_mut() {
                arms.push(r.as_mut());
            }
            let mut labels: Vec<String> =
                vec!["gathered-shipped".into(), "gathered-single-pass".into()];
            labels.extend(split_chunks.iter().map(|c| format!("gathered-chunks{c}")));
            let label_refs: Vec<&str> = labels.iter().map(String::as_str).collect();
            let us = bench_interleaved(&label_refs, rounds, &stream, &mut arms);
            drop(arms);
            let ship_us = us[0];
            let dsa_us = us[1];
            for (i, &c) in split_chunks.iter().enumerate() {
                split_us.push((c, us[2 + i]));
            }
            let splits = split_us
                .iter()
                .map(|(c, u)| format!("chunks={c} {u:.1}"))
                .collect::<Vec<_>>()
                .join("  ");
            println!(
                "  t_q={t} attn_gathered us: shipped {ship_us:.1}  single-pass {dsa_us:.1}  \
                 {splits}   (N={rounds} interleaved)"
            );

            // Regression against the arm the policy selects at this width.
            let policy_arm = mla_dsa_attn_arm_effective(t);
            let policy_us = match policy_arm {
                0 => ship_us,
                1 => dsa_us,
                n => split_us
                    .iter()
                    .find(|(c, _)| *c == n)
                    .map(|(_, u)| *u)
                    .unwrap_or(ship_us),
            };
            if policy_us > ship_us * MLA_DSA_REGRESSION_MARGIN {
                let line = format!(
                    "attn_gathered kv={kv} t_q={t} arm={policy_arm}: {policy_us:.1} us vs \
                     shipped {ship_us:.1} us ({:.3}x, margin {MLA_DSA_REGRESSION_MARGIN:.2}x)",
                    policy_us / ship_us
                );
                if TIMING_IS_BINDING {
                    failures.push(Failure(format!("REGRESSION {line}")));
                } else {
                    println!("  DIAGNOSTIC (non-target build, not a failure): {line}");
                }
            }
            // The winner this run would justify putting in the table -- and it only ever names
            // an arm the table is ALLOWED to hold at this width, so a note can never talk the
            // next editor into an illegal cell.
            let mut best = (0i32, ship_us);
            if dsa_us < best.1 {
                best = (1, dsa_us);
            }
            if t <= MLA_DSA_NAMED_CLASS_T_MAX {
                for (c, u) in &split_us {
                    if *u < best.1 {
                        best = (*c, *u);
                    }
                }
            }
            if best.0 != policy_arm {
                println!(
                    "  note: kv={kv} t_q={t} fastest ADMISSIBLE gathered arm is {} ({:.1} us), \
                     table says {policy_arm} -- an arm-table edit this run would justify (cite it \
                     in mla_ffi.rs)",
                    best.0, best.1
                );
            }

            // ---------------------------------------------------------------- kpool_score
            let qk_scale = (IDX_D as f32).powf(-0.5);
            let head_scale = (IDX_HEADS as f32).powf(-0.5);
            let sship = || unsafe {
                memra_mla_kpool_score_f32(
                    q_index_p,
                    pool_keys_p,
                    hw_p,
                    score_ship_p,
                    t as i32,
                    IDX_HEADS as i32,
                    IDX_D as i32,
                    n_pools as i32,
                    POOL as i32,
                    first_pos as i32,
                    qk_scale,
                    head_scale,
                    cu,
                )
            };
            let sdsa = || unsafe {
                memra_mla_kpool_score_dsa_f32(
                    q_index_p,
                    pool_keys_p,
                    hw_p,
                    score_arm_p,
                    t as i32,
                    IDX_HEADS as i32,
                    IDX_D as i32,
                    n_pools as i32,
                    POOL as i32,
                    first_pos as i32,
                    qk_scale,
                    head_scale,
                    cu,
                )
            };
            assert_eq!(sship(), 0, "kpool_score shipped launch");
            stream.synchronize()?;
            let sref = e.dtoh(&score_ship)?[..t * n_pools].to_vec();
            let rc = sdsa();
            if rc != 0 {
                println!("  t_q={t} kpool_score dsa: REFUSED geometry (rc {rc}), shipped path");
                continue;
            }
            stream.synchronize()?;
            let sarm = e.dtoh(&score_arm)?[..t * n_pools].to_vec();
            let sid = bits_equal(&sref, &sarm);
            println!(
                "  t_q={t} kpool_score head-blocked: {}",
                if sid { "BIT-IDENTICAL" } else { "MISMATCH" }
            );
            if !sid {
                failures.push(Failure(format!(
                    "MISMATCH kpool_score head-blocked kv={kv} t_q={t}: claims bit identity"
                )));
            }
            let mut sship_m = sship;
            let mut sdsa_m = sdsa;
            let sus = bench_interleaved(
                &["score-shipped", "score-head-blocked"],
                rounds,
                &stream,
                &mut [&mut sship_m, &mut sdsa_m],
            );
            let (sship_us, sdsa_us) = (sus[0], sus[1]);
            println!(
                "  t_q={t} kpool_score us: shipped {sship_us:.1}  head-blocked {sdsa_us:.1}  \
                 ({:.2}x)   (N={rounds} interleaved)",
                sship_us / sdsa_us
            );
            if n_pools >= MLA_DSA_SCORE_MIN_POOLS && sdsa_us > sship_us * MLA_DSA_REGRESSION_MARGIN
            {
                let line = format!(
                    "kpool_score kv={kv} t_q={t}: {sdsa_us:.1} us vs shipped {sship_us:.1} us \
                     ({:.3}x, margin {MLA_DSA_REGRESSION_MARGIN:.2}x)",
                    sdsa_us / sship_us
                );
                if TIMING_IS_BINDING {
                    failures.push(Failure(format!("REGRESSION {line}")));
                } else {
                    println!("  DIAGNOSTIC (non-target build, not a failure): {line}");
                }
            }
        }
    }

    println!();
    if failures.is_empty() {
        println!(
            "dsa-decode-gate PASS: every bit-identical arm matched bytewise, every \
             dsa-warp-online-f32 arm held argmax at every width where the door can serve it \
             (t_q <= {MLA_DSA_NAMED_CLASS_T_MAX}), and no policy-selected arm regressed beyond \
             {MLA_DSA_REGRESSION_MARGIN:.2}x at any context in {KVS:?} or t_q in {T_QS:?}. \
             Timing bar was {}.",
            if TIMING_IS_BINDING {
                "BINDING"
            } else {
                "diagnostic only (non-target build)"
            }
        );
        return Ok(());
    }
    for f in &failures {
        println!("{}", f.0);
    }
    println!("dsa-decode-gate FAIL: {} failing check(s)", failures.len());
    std::process::exit(1);
}
