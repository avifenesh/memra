//! Gate/bench for `MEMRA_B200_MLA_DECODE_ARM`, the t<=8 head-parallel MLA/DSA decode arm
//! (lane/b200-mla-decode-20260902, then lane/b200-mla-depth-20260902 for the depth axis;
//! research/b200-mla-decode-20260902/LANE.md and research/b200-mla-depth-20260902/LANE.md).
//!
//! Synthetic, shape-faithful glm5_next geometry (64 heads, kv_rank 512, head dim 256, d_rope 0
//! / NoPE, DSA top-k-shaped 2048-slot gather over a k-pool of `kv` rows). Two axes and two
//! checks, both hard:
//!
//! Axes: t_q in {1,2,4,8} (plain decode and the DFlash2 spec-verify widths) x kv in
//! {2k, 32k, 128k, 256k} cached positions (the pool the gathered rows come from; the gathered
//! set stays the DSA top-k width at every depth, as it does in serving). absorb_q and
//! decompress_v never read the pool, so their depth axis is the L2 state serving leaves them
//! in, which the scrub below reproduces. Depth is a MEASUREMENT axis only: the serving policy
//! keys on t_q alone (`mla_ffi::mla_b200_arm_table_split`), so a split that scales badly
//! with depth shows up here as a REGRESSION, not as a ceiling that hides it. The 2026-09-02
//! end-to-end depth A/B on the pair (arm ON 15.58 tok/s vs OFF 32.08 at 256k) was later found
//! bimodal on the spec route with every B200 door off (15.42 tok/s), so the MLA-arm
//! attribution is NOT established and no depth key ships until the plain-route arms say
//! otherwise (research/b200-mla-depth-20260902/LANE.md).
//!
//! 1. BIT-IDENTITY. At every (t_q, kv) and every split in {2,4,8}, each split twin's output
//!    must equal the shipped kernel's output bit for bit, for all three kernels. THE ONE
//!    CHANGE UNDER TEST is launch geometry (see cu/mla_attn.cu headers on
//!    `memra_mla_absorb_q_split_kernel` / `memra_mla_decompress_v_split_kernel` /
//!    `memra_mla_attn_gathered_split_kernel`): every kept output element is computed by the
//!    SAME sequence of floating-point operations as the unsplit kernel, so bit identity is a
//!    construction argument, not a tolerance. This gate asserts it rather than trusting it.
//!
//! 2. REGRESSION. The arm the SERVING TABLE selects (`mla_ffi::mla_b200_arm_table_split`, the
//!    same constants the wrappers read, so this check and the policy cannot drift apart) may
//!    not be slower than the shipped kernel by more than `MLA_B200_ARM_REGRESSION_MARGIN` at
//!    any measured (t_q, kv). A failing cell prints one `REGRESSION ...` line and the gate
//!    exits 1. A table cell of 1 selects the shipped launcher itself, so it can never regress.
//!
//! Timing: every split in {1,2,4,8} is timed at every (t_q, kv) for all three kernels. Split 1
//! is the shipped launcher (`memra_mla_*_f32`), never the twin at split=1, matching what the
//! wrapper runs for a selection of 1. Samples are taken in INTERLEAVED rounds (round r times
//! split 1, 2, 4, 8 back to back; N rounds) so clock drift lands on every arm equally. Before
//! EVERY timed launch (all arms, all depths) a scrub streams `SCRUB_ROWS` x kv_rank floats
//! (512 MB, above any current L2) through the SMs, so each arm starts from the cold-L2 state
//! serving leaves these kernels in at any depth (the MoE expert stream between two MLA layers
//! evicts everything). `scrub=0` reproduces the first cut's warm-L2 methodology. Each sample is
//! one launch bracketed by `stream.synchronize()` (launch-to-completion), output buffers
//! preallocated outside the timed region. The per-(t, kv) winner table printed at the end is
//! what a box run reads to edit the table constants, or to justify a depth key if one is
//! ever warranted.
//!
//! RIG LAW (docs/PERFORMANCE.md, "5090 laptop throttles; correctness gates OK, timing numbers
//! never"): the timing is a serving-relevant receipt only on the B200 class the door targets;
//! elsewhere it is a diagnostic. The bit-identity check is valid on any CUDA device (the gate
//! calls the twins through their raw FFI, bypassing the door, so a 120a build proves the
//! KERNELS).
//!
//! usage: mla-decode-arm-gate [device_ordinal] [rounds=5] [max_kv=262144] [scrub=1]

use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use memra_engine::Engine;
use memra_engine::mla_ffi::{
    MLA_B200_ARM_REGRESSION_MARGIN, MLA_B200_ARM_T_MAX, MlaB200Kernel, memra_mla_absorb_q_f32,
    memra_mla_absorb_q_split_f32, memra_mla_attn_gathered_f32, memra_mla_attn_gathered_split_f32,
    memra_mla_decompress_v_f32, memra_mla_decompress_v_split_f32, mla_b200_arm_table_split,
};
use std::os::raw::c_void;
use std::sync::Arc;
use std::time::Instant;

const N_HEAD: usize = 64;
const KV_RANK: usize = 512;
const D_NOPE: usize = 256;
const D_V: usize = 256;
const D_ROPE: usize = 0;
/// The DSA gathered width: fixed by top-k, not by depth (min'd with the pool at 2k, where it
/// is the whole pool).
const N_SLOTS: usize = 2048;
const SCALE: f32 = 1.0 / 23.323_808; // 1/sqrt(d_nope + d_rope), d_nope=256/d_rope=0 shape

/// Split factors timed at every cell. Index 0 (split=1) is the shipped launcher.
const SPLITS: [i32; 4] = [1, 2, 4, 8];
/// Query widths measured: plain decode (1) and the DFlash2 spec-verify shape (4..8).
const T_QS: [usize; 4] = [1, 2, 4, 8];
/// Cached-position depths swept: the 2k floor, the first cut's 32k pool, and the 128k/256k
/// depths the box serves the 1M tier at (the 256k end-to-end receipt is where the arm lost).
const KV_LENS: [usize; 4] = [2048, 32_768, 131_072, 262_144];
/// Rows streamed through L2 before every timed launch: 262144 x 512 x 4 B = 512 MB read plus
/// the same written, above any current L2 (B200 126 MB), so the launch that follows is cold.
const SCRUB_ROWS: usize = 262_144;

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

fn randidx(n: usize, seed: u64, max_excl: i32) -> Vec<i32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((s >> 33) as u32 % max_excl as u32) as i32
        })
        .collect()
}

fn bits_equal(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
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

/// One measured (kernel, t_q, kv) cell: mean us per split, indexed like `SPLITS`.
struct Cell {
    kernel: MlaB200Kernel,
    t_q: usize,
    kv: usize,
    us: [f64; 4],
}

impl Cell {
    fn shipped_us(&self) -> f64 {
        self.us[0]
    }
    fn best(&self) -> (i32, f64) {
        let mut k = 0;
        for (i, &u) in self.us.iter().enumerate() {
            if u < self.us[k] {
                k = i;
            }
        }
        (SPLITS[k], self.us[k])
    }
    fn us_for(&self, split: i32) -> Option<f64> {
        SPLITS.iter().position(|&s| s == split).map(|i| self.us[i])
    }
}

/// Warm each split twice (unmeasured), then `rounds` interleaved rounds over every split,
/// `scrub()` before every launch (warm-ups included) so each sample starts cold.
fn bench_splits(
    label: &str,
    rounds: usize,
    stream: &Stream,
    scrub: &mut impl FnMut(),
    mut launch: impl FnMut(i32) -> i32,
) -> [f64; 4] {
    for &split in &SPLITS {
        for _ in 0..2 {
            scrub();
            stream.synchronize().expect("sync");
            let rc = launch(split);
            assert_eq!(rc, 0, "{label} split={split}: warmup launch rc={rc}");
            stream.synchronize().expect("sync");
        }
    }
    let mut acc = [0f64; 4];
    for _ in 0..rounds {
        for (k, &split) in SPLITS.iter().enumerate() {
            scrub();
            stream.synchronize().expect("sync");
            let t0 = Instant::now();
            let rc = launch(split);
            stream.synchronize().expect("sync");
            acc[k] += t0.elapsed().as_secs_f64() * 1e6;
            assert_eq!(rc, 0, "{label} split={split}: launch rc={rc}");
        }
    }
    let mut us = [0f64; 4];
    for (u, a) in us.iter_mut().zip(acc) {
        *u = a / rounds as f64;
    }
    println!(
        "  {label} timing: split=1(shipped) {:.1}  split=2 {:.1}  split=4 {:.1}  split=8 {:.1} us \
         (N={rounds} interleaved rounds, mean)",
        us[0], us[1], us[2], us[3]
    );
    us
}

/// Bit-identity of every split twin against the shipped output: one line when all agree,
/// one line per mismatching split otherwise.
fn check_identity(
    label: &str,
    shipped_out: &[f32],
    arms: &[(i32, Vec<f32>)],
    mismatches: &mut usize,
) {
    let bad: Vec<i32> = arms
        .iter()
        .filter(|(_, out)| !bits_equal(shipped_out, out))
        .map(|(split, _)| *split)
        .collect();
    if bad.is_empty() {
        let all: Vec<String> = arms.iter().map(|(s, _)| s.to_string()).collect();
        println!("  {label} split={{{}}}: BIT-IDENTICAL", all.join(","));
    } else {
        for split in bad {
            println!("  {label} split={split}: MISMATCH");
            *mismatches += 1;
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let arg = |i: usize| std::env::args().nth(i);
    let dev: usize = arg(1).and_then(|s| s.parse().ok()).unwrap_or(0);
    let rounds: usize = arg(2).and_then(|s| s.parse().ok()).unwrap_or(5).max(1);
    let max_kv: usize = arg(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or(KV_LENS[KV_LENS.len() - 1]);
    let scrub_on: bool = arg(4).map(|s| s != "0").unwrap_or(true);
    let e = Engine::new(dev)?;
    let stream = e.stream();

    println!(
        "mla-decode-arm-gate: device {dev}, geometry nh={N_HEAD} kv_rank={KV_RANK} \
         d_nope={D_NOPE} d_v={D_V} d_rope={D_ROPE} n_slots={N_SLOTS}, t_q in {T_QS:?}, \
         kv in {KV_LENS:?} (max_kv={max_kv}), rounds={rounds}, scrub={}",
        if scrub_on {
            format!("{} rows before every launch", SCRUB_ROWS)
        } else {
            "OFF (warm-L2 methodology)".to_string()
        }
    );
    for kernel in MlaB200Kernel::ALL {
        let cells: Vec<String> = (1..=MLA_B200_ARM_T_MAX)
            .map(|t| format!("t{t}={}", mla_b200_arm_table_split(kernel, t)))
            .collect();
        println!(
            "  serving table {:<13} {}  (1 = shipped kernel; keyed on t_q only, no depth key)",
            kernel.name(),
            cells.join(" ")
        );
    }

    // The scrub pair: an rms_norm over SCRUB_ROWS x KV_RANK reads 512 MB and writes 512 MB
    // through the SMs, which is exactly the "everything evicted" state a served MLA layer sees.
    let scrub_src = e.htod(&randf(SCRUB_ROWS * KV_RANK, 0xB200_00A0))?;
    let scrub_w = e.htod(&vec![1.0f32; KV_RANK])?;
    let mut scrub_dst = e.uninit(SCRUB_ROWS * KV_RANK)?;
    let mut scrub = || {
        if scrub_on {
            e.rms_norm(
                &scrub_src,
                &scrub_w,
                &mut scrub_dst,
                KV_RANK,
                SCRUB_ROWS,
                1e-6,
            )
            .expect("scrub rms_norm");
        }
    };

    let mut mismatches = 0usize;
    let mut cells: Vec<Cell> = Vec::new();

    for &kv in KV_LENS.iter().filter(|&&kv| kv <= max_kv) {
        // The pool at this depth: the only input whose size follows kv.
        let cache = e.htod(&randf(kv * (KV_RANK + D_ROPE), 0xB200_0007 ^ kv as u64))?;
        let n_slots = N_SLOTS.min(kv);

        for &t in &T_QS {
            println!("== kv={kv} t_q={t} ==");

            // ------------------------------------------------------------ absorb_q
            {
                let q_nope = randf(t * N_HEAD * D_NOPE, 0xB200_0001 ^ t as u64);
                let wk_b = randf(N_HEAD * KV_RANK * D_NOPE, 0xB200_0002 ^ t as u64);
                let q = e.htod(&q_nope)?;
                let wk = e.htod(&wk_b)?;
                let mut out = e.uninit(t * N_HEAD * KV_RANK)?;
                let out_ptr = dpm(&mut out, &stream);

                let launch = |split: i32| unsafe {
                    if split == 1 {
                        memra_mla_absorb_q_f32(
                            dp(&q, &stream),
                            dp(&wk, &stream),
                            out_ptr,
                            t as i32,
                            N_HEAD as i32,
                            D_NOPE as i32,
                            KV_RANK as i32,
                            stream.cu_stream() as *mut c_void,
                        )
                    } else {
                        memra_mla_absorb_q_split_f32(
                            dp(&q, &stream),
                            dp(&wk, &stream),
                            out_ptr,
                            t as i32,
                            N_HEAD as i32,
                            D_NOPE as i32,
                            KV_RANK as i32,
                            split,
                            stream.cu_stream() as *mut c_void,
                        )
                    }
                };

                assert_eq!(launch(1), 0, "absorb_q shipped launch");
                stream.synchronize()?;
                let shipped_out = e.dtoh(&out)?;
                let mut arms = Vec::new();
                for &split in &SPLITS[1..] {
                    assert_eq!(launch(split), 0, "absorb_q split={split} launch");
                    stream.synchronize()?;
                    arms.push((split, e.dtoh(&out)?));
                }
                check_identity("absorb_q", &shipped_out, &arms, &mut mismatches);
                let us = bench_splits("absorb_q", rounds, &stream, &mut scrub, launch);
                cells.push(Cell {
                    kernel: MlaB200Kernel::AbsorbQ,
                    t_q: t,
                    kv,
                    us,
                });
            }

            // --------------------------------------------------------- decompress_v
            {
                let o_lat_h = randf(t * N_HEAD * KV_RANK, 0xB200_0003 ^ t as u64);
                let wv_b = randf(N_HEAD * D_V * KV_RANK, 0xB200_0004 ^ t as u64);
                let o_lat = e.htod(&o_lat_h)?;
                let wv = e.htod(&wv_b)?;
                let mut out = e.uninit(t * N_HEAD * D_V)?;
                let out_ptr = dpm(&mut out, &stream);

                let launch = |split: i32| unsafe {
                    if split == 1 {
                        memra_mla_decompress_v_f32(
                            dp(&o_lat, &stream),
                            dp(&wv, &stream),
                            out_ptr,
                            t as i32,
                            N_HEAD as i32,
                            D_V as i32,
                            KV_RANK as i32,
                            stream.cu_stream() as *mut c_void,
                        )
                    } else {
                        memra_mla_decompress_v_split_f32(
                            dp(&o_lat, &stream),
                            dp(&wv, &stream),
                            out_ptr,
                            t as i32,
                            N_HEAD as i32,
                            D_V as i32,
                            KV_RANK as i32,
                            split,
                            stream.cu_stream() as *mut c_void,
                        )
                    }
                };

                assert_eq!(launch(1), 0, "decompress_v shipped launch");
                stream.synchronize()?;
                let shipped_out = e.dtoh(&out)?;
                let mut arms = Vec::new();
                for &split in &SPLITS[1..] {
                    assert_eq!(launch(split), 0, "decompress_v split={split} launch");
                    stream.synchronize()?;
                    arms.push((split, e.dtoh(&out)?));
                }
                check_identity("decompress_v", &shipped_out, &arms, &mut mismatches);
                let us = bench_splits("decompress_v", rounds, &stream, &mut scrub, launch);
                cells.push(Cell {
                    kernel: MlaB200Kernel::DecompressV,
                    t_q: t,
                    kv,
                    us,
                });
            }

            // -------------------------------------------------------- attn_gathered
            {
                let q_lat_h = randf(t * N_HEAD * KV_RANK, 0xB200_0005 ^ t as u64);
                // D_ROPE is 0 on this NoPE geometry; allocate a dummy positive-length plane
                // the kernel never dereferences (d_rope==0 makes its rope loop bound zero).
                let q_pe_h = randf(t * N_HEAD, 0xB200_0006 ^ t as u64);
                let idx_h = randidx(t * n_slots, 0xB200_0008 ^ t as u64 ^ kv as u64, kv as i32);

                let q_lat = e.htod(&q_lat_h)?;
                let q_pe = e.htod(&q_pe_h)?;
                let idx = e.htod_i32(&idx_h)?;
                let mut out = e.uninit(t * N_HEAD * KV_RANK)?;
                let out_ptr = dpm(&mut out, &stream);

                let launch = |split: i32| unsafe {
                    if split == 1 {
                        memra_mla_attn_gathered_f32(
                            dp(&q_lat, &stream),
                            dp(&q_pe, &stream),
                            dp(&cache, &stream),
                            dpi(&idx, &stream),
                            out_ptr,
                            N_HEAD as i32,
                            KV_RANK as i32,
                            D_ROPE as i32,
                            t as i32,
                            n_slots as i32,
                            SCALE,
                            stream.cu_stream() as *mut c_void,
                        )
                    } else {
                        memra_mla_attn_gathered_split_f32(
                            dp(&q_lat, &stream),
                            dp(&q_pe, &stream),
                            dp(&cache, &stream),
                            dpi(&idx, &stream),
                            out_ptr,
                            N_HEAD as i32,
                            KV_RANK as i32,
                            D_ROPE as i32,
                            t as i32,
                            n_slots as i32,
                            SCALE,
                            split,
                            stream.cu_stream() as *mut c_void,
                        )
                    }
                };

                assert_eq!(launch(1), 0, "attn_gathered shipped launch");
                stream.synchronize()?;
                let shipped_out = e.dtoh(&out)?;
                let mut arms = Vec::new();
                for &split in &SPLITS[1..] {
                    assert_eq!(launch(split), 0, "attn_gathered split={split} launch");
                    stream.synchronize()?;
                    arms.push((split, e.dtoh(&out)?));
                }
                check_identity("attn_gathered", &shipped_out, &arms, &mut mismatches);
                let us = bench_splits("attn_gathered", rounds, &stream, &mut scrub, launch);
                cells.push(Cell {
                    kernel: MlaB200Kernel::AttnGathered,
                    t_q: t,
                    kv,
                    us,
                });
            }
        }
        drop(cache);
    }

    // ------------------------------------------- per-(t, kv) winner table + regression
    println!(
        "== per-(t, kv) winner table (mean us, N={rounds} interleaved rounds, scrub={}; split=1 \
         is the shipped kernel; 'sel' is the serving table's cell, 'arm' its time) ==",
        if scrub_on { "on" } else { "off" }
    );
    println!(
        "  {:<13} {:>3} {:>7} {:>10} {:>10} {:>10} {:>10} {:>5} {:>4} {:>11}",
        "kernel",
        "t_q",
        "kv",
        "split=1",
        "split=2",
        "split=4",
        "split=8",
        "best",
        "sel",
        "arm/shipped"
    );
    let mut regressions = 0usize;
    for c in &cells {
        let (best_split, _) = c.best();
        let sel = mla_b200_arm_table_split(c.kernel, c.t_q);
        let (arm_us, ratio) = match c.us_for(sel) {
            Some(us) => (us, us / c.shipped_us()),
            None => {
                println!(
                    "REGRESSION {} t_q={} kv={}: serving table split={sel} is not in the timed \
                     set {SPLITS:?}; extend SPLITS or fix the table",
                    c.kernel.name(),
                    c.t_q,
                    c.kv
                );
                regressions += 1;
                continue;
            }
        };
        println!(
            "  {:<13} {:>3} {:>7} {:>10.1} {:>10.1} {:>10.1} {:>10.1} {:>5} {:>4} {:>10.3}x",
            c.kernel.name(),
            c.t_q,
            c.kv,
            c.us[0],
            c.us[1],
            c.us[2],
            c.us[3],
            best_split,
            sel,
            ratio
        );
        if ratio > MLA_B200_ARM_REGRESSION_MARGIN {
            println!(
                "REGRESSION {} t_q={} kv={}: table split={sel} {arm_us:.1} us vs shipped \
                 {:.1} us ({:+.1}%), margin {:.0}%",
                c.kernel.name(),
                c.t_q,
                c.kv,
                c.shipped_us(),
                (ratio - 1.0) * 100.0,
                (MLA_B200_ARM_REGRESSION_MARGIN - 1.0) * 100.0
            );
            regressions += 1;
        }
    }
    for c in &cells {
        let (best_split, best_us) = c.best();
        let sel = mla_b200_arm_table_split(c.kernel, c.t_q);
        if best_split != sel {
            println!(
                "  note {} t_q={} kv={}: fastest measured split={best_split} ({best_us:.1} us), \
                 serving table has {sel} ({:.1} us); edit the table only on a B200-class run",
                c.kernel.name(),
                c.t_q,
                c.kv,
                c.us_for(sel).unwrap_or(f64::NAN)
            );
        }
    }

    if mismatches == 0 && regressions == 0 {
        println!(
            "mla-decode-arm-gate PASS: every split arm BIT-IDENTICAL to its shipped kernel at \
             t_q in {T_QS:?} x kv in {:?} (absorb_q, decompress_v, attn_gathered), and the \
             serving table's arm is within {:.0}% of shipped or faster at every measured (t_q, kv)",
            KV_LENS
                .iter()
                .filter(|&&kv| kv <= max_kv)
                .collect::<Vec<_>>(),
            (MLA_B200_ARM_REGRESSION_MARGIN - 1.0) * 100.0
        );
        Ok(())
    } else {
        println!(
            "mla-decode-arm-gate FAIL: {mismatches} split-vs-shipped mismatches, {regressions} \
             table cells slower than shipped by more than {:.0}%",
            (MLA_B200_ARM_REGRESSION_MARGIN - 1.0) * 100.0
        );
        std::process::exit(1);
    }
}
