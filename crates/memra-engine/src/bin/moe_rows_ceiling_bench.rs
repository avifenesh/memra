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
use cudarc::driver::DevicePtr;
use memra_engine::{Engine, QT_NVFP4};
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
        ("MEMRA_MOE_GATEUP_ILP2", "0"),
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
    // warm
    for _ in 0..10 {
        let _ = served(0)?;
        let _ = probe("loadonly", 0)?;
        let _ = probe("mathonly", 0)?;
    }
    e.stream().synchronize()?;
    let mut t_s = Vec::new();
    let mut t_l = Vec::new();
    let mut t_m = Vec::new();
    for i in 0..iters {
        for arm in 0..3 {
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
                    _ => {
                        let _ = probe("mathonly", set)?;
                    }
                }
            }
            e.stream().synchronize()?;
            let us = t0.elapsed().as_secs_f64() * 1e6 / reps as f64;
            match arm {
                0 => t_s.push(us),
                1 => t_l.push(us),
                _ => t_m.push(us),
            }
        }
    }
    let (s, l, m) = (median(&mut t_s), median(&mut t_l), median(&mut t_m));
    // weights only: both planes, n_used experts, one row each per output column
    let bytes = (2 * n_used * n_ff * grb) as f64;
    println!(
        "[rows-ceiling] t=1 n_used=8 {gin}->{n_ff} rb={grb}x2 ({iters} rounds x {reps}, {copies} slab sets): \
         served {s:.2} us ({:.0} GB/s) | loadonly {l:.2} us ({:.0} GB/s, {:.2}x served) | \
         mathonly {m:.2} us ({:.2}x served)",
        bytes / s / 1e3,
        bytes / l / 1e3,
        s / l,
        s / m
    );
    println!(
        "[rows-ceiling] reading: loadonly close to served => the ACCESS PATTERN is the wall; \
         mathonly close to served => the PRMT/dp4a INSTRUCTION STREAM is."
    );
    Ok(())
}
