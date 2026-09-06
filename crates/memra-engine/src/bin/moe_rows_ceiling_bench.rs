//! Ceiling probe for the verify-rows MoE gate/up kernel (lane/moe-rows-ceiling-20260906).
//!
//! The served kernel reaches ~19% of the B200 pair's HBM peak and ncu cannot run inside the vast
//! container (ERR_NVGPUCTRPERM), so this answers the memory-vs-issue question with three arms on
//! the SAME buffers at the served shape (in_f 4096, n_ff 2048, n_used 8, t = 1, interleaved
//! NVFP4): the served kernel, a load-only twin (same addresses, arithmetic deleted) and a
//! math-only twin (one group loaded, the served arithmetic repeated). Rotating slab sets keeps
//! the loads real DRAM traffic. Prints per-launch us and the implied GB/s for each arm.
//!
//! Usage: `moe-rows-ceiling-bench [iters=7] [reps=100] [copies=4]`.
use cudarc::driver::{CudaSlice, DevicePtr};
use memra_engine::{Engine, QT_NVFP4, QT_NVFP4_V2};
use std::time::Instant;

struct Lcg(u32);
impl Lcg {
    fn byte(&mut self) -> u8 {
        self.0 = self.0.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (self.0 >> 24) as u8
    }
}
fn synth_rows(out_f: usize, in_f: usize, seed: u32) -> (Vec<u8>, usize) {
    let nsb64 = in_f / 64;
    let row_bytes = nsb64 * 36;
    let mut w = vec![0u8; out_f * row_bytes];
    let mut r = Lcg(seed);
    for chunk in w.chunks_exact_mut(36) {
        for d in &mut chunk[0..4] {
            *d = (r.byte() & 0x07) | 0x38;
        }
        for q in &mut chunk[4..36] {
            *q = r.byte();
        }
    }
    (w, row_bytes)
}
fn vecf(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 2.0
        })
        .collect()
}
fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}
fn setenv(k: &str, v: &str) {
    unsafe { std::env::set_var(k, v) };
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let iters: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(7);
    let reps: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(100);
    let copies: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(4);
    let e = Engine::new(0)?;
    let (n_used, t) = (8usize, 1usize);
    let n_pairs = t * n_used;
    let stream = e.stream();
    let (gin, n_ff) = (4096usize, 2048usize);
    // the served posture for this kernel: ILP on, every other rows door off
    for (k, v) in [
        ("MEMRA_MOE_VROWS_ILP", "1"),
        ("MEMRA_MOE_VROWS_ORD", "0"),
        ("MEMRA_MOE_VROWS_DEDUP_ORDER", "0"),
        ("MEMRA_MOE_VROWS_PACK", "0"),
    ] {
        setenv(k, v);
    }
    let (grows, grb) = synth_rows(n_ff, gin, 0x8181);
    let mut sets = Vec::new();
    let mut keep = Vec::new();
    for c in 0..copies {
        let mut ptrs = vec![0u64; 3 * n_pairs];
        for j in 0..n_used {
            for plane in 0..2 {
                let mut d = grows.clone();
                d[5] ^= (c * 16 + j * 2 + plane + 1) as u8;
                let buf = e.htod_bytes(&d)?;
                let p = {
                    let (p, _g) = buf.device_ptr(&stream);
                    p
                };
                ptrs[plane * n_pairs + j] = p;
                keep.push(buf);
            }
        }
        sets.push(e.htod_u64(&ptrs)?);
    }
    // the same rows repacked slot-major, so the V2 arms read identical VALUES with the
    // contiguous scale stream (that is the whole point of the comparison)
    let mut sets_v2 = Vec::new();
    let mut keep_v2 = Vec::new();
    for c in 0..copies {
        let mut ptrs = vec![0u64; 3 * n_pairs];
        for j in 0..n_used {
            for plane in 0..2 {
                let mut d = grows.clone();
                d[5] ^= (c * 16 + j * 2 + plane + 1) as u8;
                let v1 = e.htod_bytes(&d)?;
                let buf = e.nvfp4_expert_split_repack(&v1, 1, n_ff, gin / 64)?;
                let p = {
                    let (p, _g) = buf.device_ptr(&stream);
                    p
                };
                ptrs[plane * n_pairs + j] = p;
                keep_v2.push(buf);
            }
        }
        sets_v2.push(e.htod_u64(&ptrs)?);
    }
    let scl: Vec<f32> = (0..3 * n_pairs).map(|pr| 0.5 + 0.01 * pr as f32).collect();
    let scl_d = e.htod(&scl)?;
    let x_d = e.htod(&vecf(t * gin, 91))?;
    let (aq, ad) = e.quantize_q8_1(&x_d, t, gin)?;

    let served = |set: usize| {
        e.moe_gate_up_preclamp8_q8_rows(
            &sets[set], &scl_d, &aq, &ad, 7.0, gin, n_ff, n_used, n_pairs, QT_NVFP4, QT_NVFP4, grb,
            grb,
        )
    };
    let probe = |arm: &str, set: usize| {
        e.moe_gate_up_rows_ceiling_probe(
            arm, &sets[set], &scl_d, &aq, &ad, 7.0, gin, n_ff, n_used, n_pairs, grb, grb,
        )
    };
    // the V2 row is 18 bytes per group (16 quant + 2 scale) against the interleaved 36 per two
    let grb2 = (gin / 32) * 18;
    let probe_v2 = |arm: &str, set: usize| {
        e.moe_gate_up_rows_ceiling_probe(
            arm,
            &sets_v2[set],
            &scl_d,
            &aq,
            &ad,
            7.0,
            gin,
            n_ff,
            n_used,
            n_pairs,
            grb2,
            grb2,
        )
    };
    let served_v2 = |set: usize| {
        e.moe_gate_up_preclamp8_q8_rows(
            &sets_v2[set],
            &scl_d,
            &aq,
            &ad,
            7.0,
            gin,
            n_ff,
            n_used,
            n_pairs,
            QT_NVFP4_V2,
            QT_NVFP4_V2,
            grb2,
            grb2,
        )
    };
    // lane-major arms (lane/moe-rows-lanemajor-20260906): the served activation permuted once,
    // the V2 rows read as 16 B words. Bit-identity against the served interleaved kernel is
    // asserted on set 0 before any timing.
    let aq_lm = e.q8_to_lane_major(&aq, t, gin)?;
    let lm_v2 = |set: usize, pack: bool| {
        e.moe_gate_up_preclamp8_q8_rows_lm(
            &sets_v2[set],
            &scl_d,
            &aq_lm,
            &ad,
            7.0,
            gin,
            n_ff,
            n_used,
            n_pairs,
            QT_NVFP4_V2,
            QT_NVFP4_V2,
            grb2,
            grb2,
            pack,
        )
    };
    {
        let h_s = e.dtoh(&served(0)?)?;
        let h_v2 = e.dtoh(&served_v2(0)?)?;
        let h_lm = e.dtoh(&lm_v2(0, false)?)?;
        let h_lm4 = e.dtoh(&lm_v2(0, true)?)?;
        let mism = |a: &[f32], b: &[f32]| {
            a.iter()
                .zip(b)
                .filter(|(x, y)| x.to_bits() != y.to_bits())
                .count()
        };
        println!(
            "[rows-ceiling] gate/up bit-identity vs served interleaved: V2 served mism={} | lane-major mism={} | lane-major w4 mism={}",
            mism(&h_s, &h_v2),
            mism(&h_s, &h_lm),
            mism(&h_s, &h_lm4)
        );
    }
    // down: one slot-major plane of gin rows over n_ff inputs (V2 row = n_ff/32 * 18 bytes), the
    // per-pair activation quantized from a synthetic swiglu output, served `_ilp` vs `_lm`.
    let (drows, _drb) = synth_rows(gin, n_ff, 0x4242);
    let drb2 = (n_ff / 32) * 18;
    let mut down_sets = Vec::new();
    let mut keep_d = Vec::new();
    for c in 0..copies {
        let mut ptrs = vec![0u64; 3 * n_pairs];
        for j in 0..n_used {
            let mut d = drows.clone();
            d[7] ^= (c * 16 + j + 3) as u8;
            let v1 = e.htod_bytes(&d)?;
            let buf = e.nvfp4_expert_split_repack(&v1, 1, gin, n_ff / 64)?;
            let p = {
                let (p, _g) = buf.device_ptr(&stream);
                p
            };
            ptrs[2 * n_pairs + j] = p;
            keep_d.push(buf);
        }
        down_sets.push(e.htod_u64(&ptrs)?);
    }
    let x2_d = e.htod(&vecf(n_pairs * n_ff, 1234))?;
    let (aq2, ad2) = e.quantize_q8_1(&x2_d, n_pairs, n_ff)?;
    let aq2_lm = e.q8_to_lane_major(&aq2, n_pairs, n_ff)?;
    let served_down = |set: usize| -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        let mut y = e.zeros(t * gin)?;
        e.moe_down8_fma_q8_rows(
            &down_sets[set],
            &scl_d,
            &aq2,
            &ad2,
            &mut y,
            n_ff,
            gin,
            n_used,
            n_pairs,
            QT_NVFP4_V2,
            drb2,
        )?;
        Ok(y)
    };
    let lm_down = |set: usize, pack: bool| -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        let mut y = e.zeros(t * gin)?;
        e.moe_down8_fma_q8_rows_lm(
            &down_sets[set],
            &scl_d,
            &aq2_lm,
            &ad2,
            &mut y,
            n_ff,
            gin,
            n_used,
            n_pairs,
            QT_NVFP4_V2,
            drb2,
            pack,
        )?;
        Ok(y)
    };
    {
        let h_s = e.dtoh(&served_down(0)?)?;
        let h_lm = e.dtoh(&lm_down(0, false)?)?;
        let h_lm4 = e.dtoh(&lm_down(0, true)?)?;
        let mism = |a: &[f32], b: &[f32]| {
            a.iter()
                .zip(b)
                .filter(|(x, y)| x.to_bits() != y.to_bits())
                .count()
        };
        println!(
            "[rows-ceiling] down bit-identity vs served V2 _ilp: lane-major mism={} | lane-major w4 mism={}",
            mism(&h_s, &h_lm),
            mism(&h_s, &h_lm4)
        );
    }
    // warm
    for _ in 0..10 {
        let _ = served(0)?;
        let _ = probe("loadonly", 0)?;
        let _ = probe("mathonly", 0)?;
        let _ = served_v2(0)?;
        let _ = probe_v2("loadonly_v2", 0)?;
        let _ = lm_v2(0, false)?;
        let _ = lm_v2(0, true)?;
        let _ = served_down(0)?;
        let _ = lm_down(0, false)?;
        let _ = lm_down(0, true)?;
    }
    e.stream().synchronize()?;
    let mut t_s = Vec::new();
    let mut t_l = Vec::new();
    let mut t_m = Vec::new();
    let mut t_sv = Vec::new();
    let mut t_lv = Vec::new();
    let mut t_lm = Vec::new();
    let mut t_lm4 = Vec::new();
    let mut t_ds = Vec::new();
    let mut t_dlm = Vec::new();
    let mut t_dlm4 = Vec::new();
    for i in 0..iters {
        for arm in 0..10 {
            let t0 = Instant::now();
            for r in 0..reps {
                let set = (i * reps + r) % copies;
                match arm {
                    0 => {
                        let _ = served(set)?;
                    }
                    1 => {
                        let _ = probe("loadonly", set)?;
                    }
                    2 => {
                        let _ = probe("mathonly", set)?;
                    }
                    3 => {
                        let _ = served_v2(set)?;
                    }
                    4 => {
                        let _ = probe_v2("loadonly_v2", set)?;
                    }
                    5 => {
                        let _ = lm_v2(set, false)?;
                    }
                    6 => {
                        let _ = lm_v2(set, true)?;
                    }
                    7 => {
                        let _ = served_down(set)?;
                    }
                    8 => {
                        let _ = lm_down(set, false)?;
                    }
                    _ => {
                        let _ = lm_down(set, true)?;
                    }
                }
            }
            e.stream().synchronize()?;
            let us = t0.elapsed().as_secs_f64() * 1e6 / reps as f64;
            match arm {
                0 => t_s.push(us),
                1 => t_l.push(us),
                2 => t_m.push(us),
                3 => t_sv.push(us),
                4 => t_lv.push(us),
                5 => t_lm.push(us),
                6 => t_lm4.push(us),
                7 => t_ds.push(us),
                8 => t_dlm.push(us),
                _ => t_dlm4.push(us),
            }
        }
    }
    let (s, l, m) = (median(&mut t_s), median(&mut t_l), median(&mut t_m));
    let (sv, lv) = (median(&mut t_sv), median(&mut t_lv));
    // weights only: both planes, n_used experts, one row each per output column
    let bytes = (2 * n_used * n_ff * grb) as f64;
    let bytes2 = (2 * n_used * n_ff * grb2) as f64;
    println!(
        "[rows-ceiling] t=1 n_used=8 {gin}->{n_ff} rb={grb}x2 ({iters} rounds x {reps}, {copies} slab sets): \
         served {s:.2} us ({:.0} GB/s) | loadonly {l:.2} us ({:.0} GB/s, {:.2}x served) | \
         mathonly {m:.2} us ({:.2}x served) | V2 served {sv:.2} us ({:.0} GB/s, {:+.1}%) | \
         V2 loadonly {lv:.2} us ({:.0} GB/s, {:+.1}% vs V1 loadonly)",
        bytes / s / 1e3,
        bytes / l / 1e3,
        s / l,
        s / m,
        bytes2 / sv / 1e3,
        100.0 * (sv / s - 1.0),
        bytes2 / lv / 1e3,
        100.0 * (lv / l - 1.0)
    );
    println!(
        "[rows-ceiling] reading: loadonly close to served => the ACCESS PATTERN is the wall; \
         mathonly close to served => the PRMT/dp4a INSTRUCTION STREAM is."
    );
    let (lm, lm4) = (median(&mut t_lm), median(&mut t_lm4));
    let (ds, dlm, dlm4) = (median(&mut t_ds), median(&mut t_dlm), median(&mut t_dlm4));
    let dbytes = (n_used * gin * drb2) as f64;
    println!(
        "[rows-lanemajor] gate/up V2: served _ilp {sv:.2} us ({:.0} GB/s) | lane-major {lm:.2} us ({:.0} GB/s, {:.3}x) | lane-major w4 {lm4:.2} us ({:.0} GB/s, {:.3}x)",
        bytes2 / sv / 1e3,
        bytes2 / lm / 1e3,
        sv / lm,
        bytes2 / lm4 / 1e3,
        sv / lm4
    );
    println!(
        "[rows-lanemajor] down V2 {n_ff}->{gin}: served _ilp {ds:.2} us ({:.0} GB/s) | lane-major {dlm:.2} us ({:.0} GB/s, {:.3}x) | lane-major w4 {dlm4:.2} us ({:.0} GB/s, {:.3}x)",
        dbytes / ds / 1e3,
        dbytes / dlm / 1e3,
        ds / dlm,
        dbytes / dlm4 / 1e3,
        ds / dlm4
    );
    Ok(())
}
