//! Box instrument for `MEMRA_MOE_DOWN_ILP2` (lane/moe-down-ilp2-20260905): the served down shape
//! (t=1, n_used 8, in_f 2048 -> out_f 4096, NVFP4 v1 rows, 8 distinct expert slabs so the reads
//! are real DRAM traffic across copies), `_ilp` vs `_ilp2` back-to-back, interleaved, medians of
//! `iters` rounds of `reps` launches each, plus the bitwise check. Usage:
//! `moe-down-ilp2-bench [iters=7] [reps=100] [copies=4]`.
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
    let (in_f, out_f, n_used, t) = (2048usize, 4096usize, 8usize, 1usize);
    let n_pairs = t * n_used;
    let (rows, rb) = synth_rows(out_f, in_f, 0x7171);
    let stream = e.stream();
    // copies x 8 distinct slabs (37.7 MB per copy) so successive rounds walk different bytes
    let mut ptr_sets = Vec::new();
    let mut keep = Vec::new();
    for c in 0..copies {
        let mut ptrs = vec![0u64; 3 * n_pairs];
        for j in 0..n_used {
            let mut d = rows.clone();
            d[5] ^= (c * 8 + j + 1) as u8;
            let buf = e.htod_bytes(&d)?;
            let p = {
                let (p, _g) = buf.device_ptr(&stream);
                p
            };
            ptrs[2 * n_pairs + j] = p;
            keep.push(buf);
        }
        ptr_sets.push(e.htod_u64(&ptrs)?);
    }
    let scl: Vec<f32> = (0..3 * n_pairs)
        .map(|pr| 0.125 + 0.01 * pr as f32)
        .collect();
    let scl_d = e.htod(&scl)?;
    let act_d = e.htod(&vecf(n_pairs * in_f, 77))?;
    let (aq2, ad2) = e.quantize_q8_1(&act_d, n_pairs, in_f)?;
    unsafe {
        std::env::set_var("MEMRA_MOE_VROWS_ILP", "1");
        std::env::set_var("MEMRA_MOE_VROWS_PACK", "0");
    }
    let launch = |ilp2: bool, set: usize, dst: &mut cudarc::driver::CudaSlice<f32>| {
        unsafe {
            std::env::set_var("MEMRA_MOE_DOWN_ILP2", if ilp2 { "1" } else { "0" });
        }
        e.moe_down8_fma_q8_rows(
            &ptr_sets[set],
            &scl_d,
            &aq2,
            &ad2,
            dst,
            in_f,
            out_f,
            n_used,
            n_pairs,
            QT_NVFP4,
            rb,
        )
    };
    // bitwise
    let mut y0 = e.zeros(t * out_f)?;
    let mut y1 = e.zeros(t * out_f)?;
    launch(false, 0, &mut y0)?;
    launch(true, 0, &mut y1)?;
    e.stream().synchronize()?;
    let (h0, h1) = (e.dtoh(&y0)?, e.dtoh(&y1)?);
    let mism = h0
        .iter()
        .zip(&h1)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    // warm
    for _ in 0..10 {
        launch(false, 0, &mut y0)?;
        launch(true, 0, &mut y1)?;
    }
    e.stream().synchronize()?;
    let mut t_ilp = Vec::new();
    let mut t_ilp2 = Vec::new();
    for i in 0..iters {
        for (ilp2, acc) in [(false, &mut t_ilp), (true, &mut t_ilp2)] {
            let t0 = Instant::now();
            for r in 0..reps {
                launch(ilp2, (i * reps + r) % copies, &mut y0)?;
            }
            e.stream().synchronize()?;
            acc.push(t0.elapsed().as_secs_f64() * 1e6 / reps as f64);
        }
    }
    let (a, b) = (median(&mut t_ilp), median(&mut t_ilp2));
    let bytes = (n_used * out_f * rb) as f64;
    println!(
        "[down-ilp2-timing] t=1 n_used=8 {in_f}->{out_f} rb={rb} back-to-back x{reps} ({iters} rounds, {copies} slab sets): \
         ilp {a:.2} us ({:.0} GB/s), ilp2 {b:.2} us ({:.0} GB/s) ({:+.1}%) bitwise_mismatch={mism}",
        bytes / a / 1e3,
        bytes / b / 1e3,
        100.0 * (b / a - 1.0)
    );
    Ok(())
}
