//! Gate/bench for `MEMRA_B200_MLA_DECODE_ARM` — the t<=8 head-parallel, launch-light MLA/DSA
//! decode arm (lane/b200-mla-decode-20260902, research/b200-mla-decode-20260902/LANE.md).
//!
//! Runs shipped-vs-arm on synthetic, shape-faithful glm5_next geometry (64 heads, kv_rank 512,
//! head dim 256, d_rope 0 / NoPE, DSA top-k-shaped 2048-slot gather over a 32k-row k-pool) at
//! t_q in {1, 4}, asserting BIT-IDENTICAL outputs between the shipped kernel and its split
//! twin at every tested split factor, then printing N=5 per-launch wall times for both arms.
//!
//! THE ONE CHANGE UNDER TEST is launch geometry (see cu/mla_attn.cu headers on
//! `memra_mla_absorb_q_split_kernel` / `memra_mla_decompress_v_split_kernel` /
//! `memra_mla_attn_gathered_split_kernel`): every kept output element is computed by the SAME
//! sequence of floating-point operations as the unsplit kernel produces for that element, so
//! bit identity is a construction argument, not a tolerance — this gate asserts it directly
//! rather than trusting the argument.
//!
//! RIG LAW (docs/PERFORMANCE.md, "5090 laptop throttles; correctness gates OK, timing numbers
//! never"): this bin's timing output is a diagnostic (relative shipped-vs-arm shape on
//! whatever device runs it), never a serving/perf claim, unless the printed device name is the
//! B200 box the door targets. The B200 A/B receipt is a separate, PENDING deliverable (see
//! LANE.md) — this gate exists so that A/B can be run and to prove correctness on any CUDA
//! device in the meantime.
//!
//! usage: mla-decode-arm-gate [device_ordinal]

use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use memra_engine::Engine;
use memra_engine::mla_ffi::{
    memra_mla_absorb_q_f32, memra_mla_absorb_q_split_f32, memra_mla_attn_gathered_f32,
    memra_mla_attn_gathered_split_f32, memra_mla_decompress_v_f32,
    memra_mla_decompress_v_split_f32,
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

fn dp(s: &CudaSlice<f32>, stream: &Arc<cudarc::driver::CudaStream>) -> *const f32 {
    s.device_ptr(stream).0 as *const f32
}
fn dpm(s: &mut CudaSlice<f32>, stream: &Arc<cudarc::driver::CudaStream>) -> *mut f32 {
    s.device_ptr_mut(stream).0 as *mut f32
}
fn dpi(s: &CudaSlice<i32>, stream: &Arc<cudarc::driver::CudaStream>) -> *const i32 {
    s.device_ptr(stream).0 as *const i32
}

/// Warm up twice (unmeasured, includes JIT/plan-cache settling), then time N calls with a
/// `stream.synchronize()` bracketing each so the printed number is launch-to-completion, not
/// launch-enqueue latency.
fn bench(
    label: &str,
    n: usize,
    stream: &Arc<cudarc::driver::CudaStream>,
    mut f: impl FnMut() -> i32,
) {
    for _ in 0..2 {
        let rc = f();
        assert_eq!(rc, 0, "{label}: warmup launch rc={rc}");
        stream.synchronize().expect("sync");
    }
    let mut us = Vec::with_capacity(n);
    for _ in 0..n {
        let t0 = Instant::now();
        let rc = f();
        stream.synchronize().expect("sync");
        us.push(t0.elapsed().as_secs_f64() * 1e6);
        assert_eq!(rc, 0, "{label}: launch rc={rc}");
    }
    let parts: Vec<String> = us.iter().map(|v| format!("{v:.1}")).collect();
    let mean = us.iter().sum::<f64>() / us.len() as f64;
    println!(
        "  {label}: [{}] us  (N={n}, mean={mean:.1}us)",
        parts.join(", ")
    );
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dev: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let e = Engine::new(dev)?;
    let stream = e.stream();

    println!(
        "mla-decode-arm-gate: device {dev}, geometry nh={N_HEAD} kv_rank={KV_RANK} \
         d_nope={D_NOPE} d_v={D_V} d_rope={D_ROPE} n_slots={N_SLOTS} pool_rows={POOL_ROWS}"
    );

    let mut mismatches = 0usize;

    for &t in &[1usize, 4] {
        println!("== t_q={t} ==");

        // ---------------------------------------------------------------- absorb_q
        let q_nope = randf(t * N_HEAD * D_NOPE, 0xB200_0001 ^ t as u64);
        let wk_b = randf(N_HEAD * KV_RANK * D_NOPE, 0xB200_0002 ^ t as u64);
        let q = e.htod(&q_nope)?;
        let wk = e.htod(&wk_b)?;

        let mut shipped = e.uninit(t * N_HEAD * KV_RANK)?;
        let rc = unsafe {
            memra_mla_absorb_q_f32(
                dp(&q, &stream),
                dp(&wk, &stream),
                dpm(&mut shipped, &stream),
                t as i32,
                N_HEAD as i32,
                D_NOPE as i32,
                KV_RANK as i32,
                stream.cu_stream() as *mut c_void,
            )
        };
        assert_eq!(rc, 0, "absorb_q shipped launch");
        let shipped_out = e.dtoh(&shipped)?;

        for &split in &[2i32, 4, 8] {
            let mut arm = e.uninit(t * N_HEAD * KV_RANK)?;
            let rc = unsafe {
                memra_mla_absorb_q_split_f32(
                    dp(&q, &stream),
                    dp(&wk, &stream),
                    dpm(&mut arm, &stream),
                    t as i32,
                    N_HEAD as i32,
                    D_NOPE as i32,
                    KV_RANK as i32,
                    split,
                    stream.cu_stream() as *mut c_void,
                )
            };
            assert_eq!(rc, 0, "absorb_q split={split} launch");
            let arm_out = e.dtoh(&arm)?;
            let ok = bits_equal(&shipped_out, &arm_out);
            println!(
                "  absorb_q split={split}: {}",
                if ok { "BIT-IDENTICAL" } else { "MISMATCH" }
            );
            if !ok {
                mismatches += 1;
            }
        }

        bench("absorb_q shipped", 5, &stream, || unsafe {
            let mut o = e.uninit(t * N_HEAD * KV_RANK).unwrap();
            memra_mla_absorb_q_f32(
                dp(&q, &stream),
                dp(&wk, &stream),
                dpm(&mut o, &stream),
                t as i32,
                N_HEAD as i32,
                D_NOPE as i32,
                KV_RANK as i32,
                stream.cu_stream() as *mut c_void,
            )
        });
        bench("absorb_q split=4 (arm)", 5, &stream, || unsafe {
            let mut o = e.uninit(t * N_HEAD * KV_RANK).unwrap();
            memra_mla_absorb_q_split_f32(
                dp(&q, &stream),
                dp(&wk, &stream),
                dpm(&mut o, &stream),
                t as i32,
                N_HEAD as i32,
                D_NOPE as i32,
                KV_RANK as i32,
                4,
                stream.cu_stream() as *mut c_void,
            )
        });

        // ------------------------------------------------------------- decompress_v
        let o_lat_h = randf(t * N_HEAD * KV_RANK, 0xB200_0003 ^ t as u64);
        let wv_b = randf(N_HEAD * D_V * KV_RANK, 0xB200_0004 ^ t as u64);
        let o_lat = e.htod(&o_lat_h)?;
        let wv = e.htod(&wv_b)?;

        let mut shipped = e.uninit(t * N_HEAD * D_V)?;
        let rc = unsafe {
            memra_mla_decompress_v_f32(
                dp(&o_lat, &stream),
                dp(&wv, &stream),
                dpm(&mut shipped, &stream),
                t as i32,
                N_HEAD as i32,
                D_V as i32,
                KV_RANK as i32,
                stream.cu_stream() as *mut c_void,
            )
        };
        assert_eq!(rc, 0, "decompress_v shipped launch");
        let shipped_out = e.dtoh(&shipped)?;

        for &split in &[2i32, 4, 8] {
            let mut arm = e.uninit(t * N_HEAD * D_V)?;
            let rc = unsafe {
                memra_mla_decompress_v_split_f32(
                    dp(&o_lat, &stream),
                    dp(&wv, &stream),
                    dpm(&mut arm, &stream),
                    t as i32,
                    N_HEAD as i32,
                    D_V as i32,
                    KV_RANK as i32,
                    split,
                    stream.cu_stream() as *mut c_void,
                )
            };
            assert_eq!(rc, 0, "decompress_v split={split} launch");
            let arm_out = e.dtoh(&arm)?;
            let ok = bits_equal(&shipped_out, &arm_out);
            println!(
                "  decompress_v split={split}: {}",
                if ok { "BIT-IDENTICAL" } else { "MISMATCH" }
            );
            if !ok {
                mismatches += 1;
            }
        }

        bench("decompress_v shipped", 5, &stream, || unsafe {
            let mut o = e.uninit(t * N_HEAD * D_V).unwrap();
            memra_mla_decompress_v_f32(
                dp(&o_lat, &stream),
                dp(&wv, &stream),
                dpm(&mut o, &stream),
                t as i32,
                N_HEAD as i32,
                D_V as i32,
                KV_RANK as i32,
                stream.cu_stream() as *mut c_void,
            )
        });
        bench("decompress_v split=4 (arm)", 5, &stream, || unsafe {
            let mut o = e.uninit(t * N_HEAD * D_V).unwrap();
            memra_mla_decompress_v_split_f32(
                dp(&o_lat, &stream),
                dp(&wv, &stream),
                dpm(&mut o, &stream),
                t as i32,
                N_HEAD as i32,
                D_V as i32,
                KV_RANK as i32,
                4,
                stream.cu_stream() as *mut c_void,
            )
        });

        // ------------------------------------------------------------- attn_gathered
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

        let mut shipped = e.uninit(t * N_HEAD * KV_RANK)?;
        let rc = unsafe {
            memra_mla_attn_gathered_f32(
                dp(&q_lat, &stream),
                dp(&q_pe, &stream),
                dp(&cache, &stream),
                dpi(&idx, &stream),
                dpm(&mut shipped, &stream),
                N_HEAD as i32,
                KV_RANK as i32,
                D_ROPE as i32,
                t as i32,
                N_SLOTS as i32,
                SCALE,
                stream.cu_stream() as *mut c_void,
            )
        };
        assert_eq!(rc, 0, "attn_gathered shipped launch");
        let shipped_out = e.dtoh(&shipped)?;

        for &split in &[2i32, 4] {
            let mut arm = e.uninit(t * N_HEAD * KV_RANK)?;
            let rc = unsafe {
                memra_mla_attn_gathered_split_f32(
                    dp(&q_lat, &stream),
                    dp(&q_pe, &stream),
                    dp(&cache, &stream),
                    dpi(&idx, &stream),
                    dpm(&mut arm, &stream),
                    N_HEAD as i32,
                    KV_RANK as i32,
                    D_ROPE as i32,
                    t as i32,
                    N_SLOTS as i32,
                    SCALE,
                    split,
                    stream.cu_stream() as *mut c_void,
                )
            };
            assert_eq!(rc, 0, "attn_gathered split={split} launch");
            let arm_out = e.dtoh(&arm)?;
            let ok = bits_equal(&shipped_out, &arm_out);
            println!(
                "  attn_gathered split={split}: {}",
                if ok { "BIT-IDENTICAL" } else { "MISMATCH" }
            );
            if !ok {
                mismatches += 1;
            }
        }

        bench("attn_gathered shipped", 5, &stream, || unsafe {
            let mut o = e.uninit(t * N_HEAD * KV_RANK).unwrap();
            memra_mla_attn_gathered_f32(
                dp(&q_lat, &stream),
                dp(&q_pe, &stream),
                dp(&cache, &stream),
                dpi(&idx, &stream),
                dpm(&mut o, &stream),
                N_HEAD as i32,
                KV_RANK as i32,
                D_ROPE as i32,
                t as i32,
                N_SLOTS as i32,
                SCALE,
                stream.cu_stream() as *mut c_void,
            )
        });
        bench("attn_gathered split=2 (arm)", 5, &stream, || unsafe {
            let mut o = e.uninit(t * N_HEAD * KV_RANK).unwrap();
            memra_mla_attn_gathered_split_f32(
                dp(&q_lat, &stream),
                dp(&q_pe, &stream),
                dp(&cache, &stream),
                dpi(&idx, &stream),
                dpm(&mut o, &stream),
                N_HEAD as i32,
                KV_RANK as i32,
                D_ROPE as i32,
                t as i32,
                N_SLOTS as i32,
                SCALE,
                2,
                stream.cu_stream() as *mut c_void,
            )
        });
    }

    if mismatches == 0 {
        println!(
            "mla-decode-arm-gate PASS: every split arm BIT-IDENTICAL to its shipped kernel at \
             t_q in {{1,4}} (absorb_q, decompress_v, attn_gathered)"
        );
        Ok(())
    } else {
        println!("mla-decode-arm-gate FAIL: {mismatches} split-vs-shipped mismatches");
        std::process::exit(1);
    }
}
