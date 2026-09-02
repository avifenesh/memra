//! Gate/bench for `MEMRA_B200_MLA_DECODE_ARM`, the t<=8 head-parallel MLA/DSA decode arm
//! (lane/b200-mla-decode-20260902, research/b200-mla-decode-20260902/LANE.md).
//!
//! Synthetic, shape-faithful glm5_next geometry (64 heads, kv_rank 512, head dim 256, d_rope 0
//! / NoPE, DSA top-k-shaped 2048-slot gather over a 32k-row k-pool). Two checks, both hard:
//!
//! 1. BIT-IDENTITY. At every t_q in {1,2,4,8} and every split in {2,4,8}, each split twin's
//!    output must equal the shipped kernel's output bit for bit, for all three kernels
//!    (absorb_q, decompress_v, attn_gathered). THE ONE CHANGE UNDER TEST is launch geometry
//!    (see cu/mla_attn.cu headers on `memra_mla_absorb_q_split_kernel` /
//!    `memra_mla_decompress_v_split_kernel` / `memra_mla_attn_gathered_split_kernel`): every
//!    kept output element is computed by the SAME sequence of floating-point operations as the
//!    unsplit kernel produces for that element, so bit identity is a construction argument,
//!    not a tolerance. This gate asserts it directly rather than trusting the argument.
//!
//! 2. REGRESSION. The arm the SERVING TABLE selects (`mla_ffi::mla_b200_arm_table_split`, the
//!    same constants the wrappers read, so this check and the policy cannot drift apart) may
//!    not be slower than the shipped kernel by more than `MLA_B200_ARM_REGRESSION_MARGIN` at
//!    any measured t_q. A cell that fails prints one `REGRESSION ...` line and the gate exits
//!    1. A table cell of 1 selects the shipped launcher itself, so it can never regress.
//!
//! Timing: every split in {1,2,4,8} is timed at every t_q for all three kernels. Split 1 is
//! the shipped launcher (`memra_mla_*_f32`), never the twin at split=1, matching what the
//! wrapper runs for a table cell of 1. Samples are taken in INTERLEAVED rounds (round r times
//! split 1, 2, 4, 8 back to back; N rounds) so clock drift lands on every arm equally; each
//! sample is one launch bracketed by `stream.synchronize()` (launch-to-completion, not
//! enqueue latency), with output buffers preallocated outside the timed region. The per-t
//! winner table printed at the end is what a box run reads to edit the table constants.
//!
//! RIG LAW (docs/PERFORMANCE.md, "5090 laptop throttles; correctness gates OK, timing numbers
//! never"): this bin's timing is a serving-relevant receipt only when the device is the B200
//! class the door targets; on any other device it is a diagnostic. The bit-identity check is
//! valid on any CUDA device (the gate calls the twins through their raw FFI, bypassing the
//! door, so a 120a build still proves the KERNELS).
//!
//! usage: mla-decode-arm-gate [device_ordinal] [rounds (default 5)]

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
const N_SLOTS: usize = 2048;
const POOL_ROWS: usize = 32_768;
const SCALE: f32 = 1.0 / 23.323_808; // 1/sqrt(d_nope + d_rope), d_nope=256/d_rope=0 shape

/// Split factors timed at every t_q. Index 0 (split=1) is the shipped launcher.
const SPLITS: [i32; 4] = [1, 2, 4, 8];
/// Query widths measured: plain decode (1) and the DFlash2 spec-verify shape (4..8).
const T_QS: [usize; 4] = [1, 2, 4, 8];

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

/// One measured (kernel, t_q) cell: mean us per split, indexed like `SPLITS`.
struct Cell {
    kernel: MlaB200Kernel,
    t_q: usize,
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

/// Warm each split twice (unmeasured), then `rounds` interleaved rounds over every split.
fn bench_splits(
    label: &str,
    rounds: usize,
    stream: &Stream,
    mut launch: impl FnMut(i32) -> i32,
) -> [f64; 4] {
    for &split in &SPLITS {
        for _ in 0..2 {
            let rc = launch(split);
            assert_eq!(rc, 0, "{label} split={split}: warmup launch rc={rc}");
            stream.synchronize().expect("sync");
        }
    }
    let mut acc = [0f64; 4];
    for _ in 0..rounds {
        for (k, &split) in SPLITS.iter().enumerate() {
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

/// Bit-identity of one split twin against the shipped output, printed and counted.
fn check_identity(
    label: &str,
    split: i32,
    shipped_out: &[f32],
    arm_out: &[f32],
    mismatches: &mut usize,
) {
    let ok = bits_equal(shipped_out, arm_out);
    println!(
        "  {label} split={split}: {}",
        if ok { "BIT-IDENTICAL" } else { "MISMATCH" }
    );
    if !ok {
        *mismatches += 1;
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dev: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let rounds: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(5)
        .max(1);
    let e = Engine::new(dev)?;
    let stream = e.stream();

    println!(
        "mla-decode-arm-gate: device {dev}, geometry nh={N_HEAD} kv_rank={KV_RANK} \
         d_nope={D_NOPE} d_v={D_V} d_rope={D_ROPE} n_slots={N_SLOTS} pool_rows={POOL_ROWS}, \
         rounds={rounds}"
    );
    for kernel in MlaB200Kernel::ALL {
        let cells: Vec<String> = (1..=MLA_B200_ARM_T_MAX)
            .map(|t| format!("t{t}={}", mla_b200_arm_table_split(kernel, t)))
            .collect();
        println!(
            "  serving table {:<13} {}  (1 = shipped kernel)",
            kernel.name(),
            cells.join(" ")
        );
    }

    let mut mismatches = 0usize;
    let mut cells: Vec<Cell> = Vec::new();

    for &t in &T_QS {
        println!("== t_q={t} ==");

        // ---------------------------------------------------------------- absorb_q
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
            for &split in &SPLITS[1..] {
                assert_eq!(launch(split), 0, "absorb_q split={split} launch");
                stream.synchronize()?;
                let arm_out = e.dtoh(&out)?;
                check_identity("absorb_q", split, &shipped_out, &arm_out, &mut mismatches);
            }
            let us = bench_splits("absorb_q", rounds, &stream, launch);
            cells.push(Cell {
                kernel: MlaB200Kernel::AbsorbQ,
                t_q: t,
                us,
            });
        }

        // ------------------------------------------------------------- decompress_v
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
            for &split in &SPLITS[1..] {
                assert_eq!(launch(split), 0, "decompress_v split={split} launch");
                stream.synchronize()?;
                let arm_out = e.dtoh(&out)?;
                check_identity(
                    "decompress_v",
                    split,
                    &shipped_out,
                    &arm_out,
                    &mut mismatches,
                );
            }
            let us = bench_splits("decompress_v", rounds, &stream, launch);
            cells.push(Cell {
                kernel: MlaB200Kernel::DecompressV,
                t_q: t,
                us,
            });
        }

        // ------------------------------------------------------------- attn_gathered
        {
            let q_lat_h = randf(t * N_HEAD * KV_RANK, 0xB200_0005 ^ t as u64);
            // D_ROPE is 0 on this NoPE geometry; allocate a dummy positive-length plane the
            // kernel never dereferences (d_rope==0 makes its rope loop bound zero iterations).
            let q_pe_h = randf(t * N_HEAD, 0xB200_0006 ^ t as u64);
            let cache_h = randf(POOL_ROWS * (KV_RANK + D_ROPE), 0xB200_0007);
            let idx_h = randidx(t * N_SLOTS, 0xB200_0008 ^ t as u64, POOL_ROWS as i32);

            let q_lat = e.htod(&q_lat_h)?;
            let q_pe = e.htod(&q_pe_h)?;
            let cache = e.htod(&cache_h)?;
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
                        N_SLOTS as i32,
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
                        N_SLOTS as i32,
                        SCALE,
                        split,
                        stream.cu_stream() as *mut c_void,
                    )
                }
            };

            assert_eq!(launch(1), 0, "attn_gathered shipped launch");
            stream.synchronize()?;
            let shipped_out = e.dtoh(&out)?;
            for &split in &SPLITS[1..] {
                assert_eq!(launch(split), 0, "attn_gathered split={split} launch");
                stream.synchronize()?;
                let arm_out = e.dtoh(&out)?;
                check_identity(
                    "attn_gathered",
                    split,
                    &shipped_out,
                    &arm_out,
                    &mut mismatches,
                );
            }
            let us = bench_splits("attn_gathered", rounds, &stream, launch);
            cells.push(Cell {
                kernel: MlaB200Kernel::AttnGathered,
                t_q: t,
                us,
            });
        }
    }

    // ------------------------------------------------------ per-t winner table + regression
    println!(
        "== per-t winner table (mean us, N={rounds} interleaved rounds; split=1 is the shipped \
         kernel; 'table' is the serving cell, 'arm' its time) =="
    );
    println!(
        "  {:<13} {:>3} {:>10} {:>10} {:>10} {:>10} {:>5} {:>5} {:>11}",
        "kernel", "t_q", "split=1", "split=2", "split=4", "split=8", "best", "table", "arm/shipped"
    );
    let mut regressions = 0usize;
    for c in &cells {
        let (best_split, _) = c.best();
        let table = mla_b200_arm_table_split(c.kernel, c.t_q);
        let (arm_us, ratio) = match c.us_for(table) {
            Some(us) => (us, us / c.shipped_us()),
            None => {
                println!(
                    "REGRESSION {} t_q={}: serving table split={table} is not in the timed set \
                     {SPLITS:?}; extend SPLITS or fix the table",
                    c.kernel.name(),
                    c.t_q
                );
                regressions += 1;
                continue;
            }
        };
        println!(
            "  {:<13} {:>3} {:>10.1} {:>10.1} {:>10.1} {:>10.1} {:>5} {:>5} {:>10.3}x",
            c.kernel.name(),
            c.t_q,
            c.us[0],
            c.us[1],
            c.us[2],
            c.us[3],
            best_split,
            table,
            ratio
        );
        if ratio > MLA_B200_ARM_REGRESSION_MARGIN {
            println!(
                "REGRESSION {} t_q={}: table split={table} {arm_us:.1} us vs shipped {:.1} us \
                 ({:+.1}%), margin {:.0}%",
                c.kernel.name(),
                c.t_q,
                c.shipped_us(),
                (ratio - 1.0) * 100.0,
                (MLA_B200_ARM_REGRESSION_MARGIN - 1.0) * 100.0
            );
            regressions += 1;
        }
    }
    for c in &cells {
        let (best_split, best_us) = c.best();
        let table = mla_b200_arm_table_split(c.kernel, c.t_q);
        if best_split != table {
            println!(
                "  note {} t_q={}: fastest measured split={best_split} ({best_us:.1} us), \
                 serving table has {table} ({:.1} us); edit the table only on a B200-class run",
                c.kernel.name(),
                c.t_q,
                c.us_for(table).unwrap_or(f64::NAN)
            );
        }
    }

    if mismatches == 0 && regressions == 0 {
        println!(
            "mla-decode-arm-gate PASS: every split arm BIT-IDENTICAL to its shipped kernel at \
             t_q in {T_QS:?} (absorb_q, decompress_v, attn_gathered), and the serving table's arm \
             is within {:.0}% of shipped or faster at every measured t_q",
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
