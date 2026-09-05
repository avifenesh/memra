//! Box instrument for `MEMRA_MOE_GATEUP_ILP2` (lane/moe-gateup-ilp2-20260905): the served gate/up
//! shape (t=1, n_used 8, in_f 4096 -> n_ff 2048, NVFP4 v1 rows, distinct expert slabs so the reads
//! are real DRAM traffic across copies), `_ilp` vs `_ilp2` back-to-back, interleaved, medians of
//! `iters` rounds of `reps` launches each, plus the bitwise check. Usage:
//! `moe-gateup-ilp2-bench [iters=7] [reps=100] [copies=4]`. (The down half this file carried, the
//! down door's instrument, was removed 2026-09-06 on that door's negative model-scale row.)
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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let iters: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(7);
    let reps: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(100);
    let copies: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(4);
    let e = Engine::new(0)?;
    let (n_used, t) = (8usize, 1usize);
    let n_pairs = t * n_used;
    let stream = e.stream();
    // ---- gate/up: two pairs per warp (memra PR: MEMRA_MOE_GATEUP_ILP2) at the served shape
    let (gin, n_ff) = (4096usize, 2048usize);
    let (grows, grb) = synth_rows(n_ff, gin, 0x8181);
    let mut gsets = Vec::new();
    let mut gkeep = Vec::new();
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
                gkeep.push(buf);
            }
        }
        gsets.push(e.htod_u64(&ptrs)?);
    }
    let gscl: Vec<f32> = (0..3 * n_pairs).map(|pr| 0.5 + 0.01 * pr as f32).collect();
    let gscl_d = e.htod(&gscl)?;
    let x_d = e.htod(&vecf(t * gin, 91))?;
    let (aq, ad) = e.quantize_q8_1(&x_d, t, gin)?;
    unsafe {
        std::env::set_var("MEMRA_MOE_VROWS_ORD", "0");
    }
    let glaunch = |ilp2: bool, set: usize| {
        unsafe {
            std::env::set_var("MEMRA_MOE_GATEUP_ILP2", if ilp2 { "1" } else { "0" });
        }
        e.moe_gate_up_preclamp8_q8_rows(
            &gsets[set],
            &gscl_d,
            &aq,
            &ad,
            7.0,
            gin,
            n_ff,
            n_used,
            n_pairs,
            QT_NVFP4,
            QT_NVFP4,
            grb,
            grb,
        )
    };
    let g0 = glaunch(false, 0)?;
    let g1 = glaunch(true, 0)?;
    e.stream().synchronize()?;
    let (h0, h1) = (e.dtoh(&g0)?, e.dtoh(&g1)?);
    let gmism = h0
        .iter()
        .zip(&h1)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    for _ in 0..10 {
        let _ = glaunch(false, 0)?;
        let _ = glaunch(true, 0)?;
    }
    e.stream().synchronize()?;
    let mut t_g = Vec::new();
    let mut t_g2 = Vec::new();
    for i in 0..iters {
        for (ilp2, acc) in [(false, &mut t_g), (true, &mut t_g2)] {
            let t0 = Instant::now();
            for r in 0..reps {
                let _ = glaunch(ilp2, (i * reps + r) % copies)?;
            }
            e.stream().synchronize()?;
            acc.push(t0.elapsed().as_secs_f64() * 1e6 / reps as f64);
        }
    }
    let (a, b) = (median(&mut t_g), median(&mut t_g2));
    let gbytes = (2 * n_used * n_ff * grb) as f64;
    println!(
        "[gateup-ilp2-timing] t=1 n_used=8 {gin}->{n_ff} rb={grb}x2 back-to-back x{reps} ({iters} rounds, {copies} slab sets): \
         ilp {a:.2} us ({:.0} GB/s), ilp2 {b:.2} us ({:.0} GB/s) ({:+.1}%) bitwise_mismatch={gmism}",
        gbytes / a / 1e3,
        gbytes / b / 1e3,
        100.0 * (b / a - 1.0)
    );
    Ok(())
}
