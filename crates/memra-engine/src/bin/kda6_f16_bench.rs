//! kda6-f16-bench: the served e4m3 KDA six against its f16-class twins on one card, interleaved,
//! three copies of the six planes so nothing sits in L2 (6 x 25.5 MB x 3 = 458 MB). Prints
//! us/launch and GB/s over the six planes' bytes. Usage: kda6-f16-bench [iters=15] [copies=3].
use memra_engine::Engine;
use std::time::Instant;

struct Lcg(u32);
impl Lcg {
    fn byte(&mut self) -> u8 {
        self.0 = self.0.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (self.0 >> 24) as u8
    }
}
fn synth_e4m3(n: usize, seed: u32) -> Vec<u8> {
    let mut r = Lcg(seed);
    (0..n)
        .map(|_| {
            let b = r.byte();
            let v = (b & 0x80) | ((4 + (b >> 3 & 0x7)) << 3) | (b & 7);
            if (v & 0x7F) == 0x7F {
                (b & 0x80) | 0x30
            } else {
                v
            }
        })
        .collect()
}
fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let iters: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(15);
    let copies: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(3);
    let e = Engine::new(0)?;
    let in_f = 4096usize;
    let dims = [8192usize, 8192, 8192, 128, 128, 64];
    let ws = [0.0123f32, 0.0234, 0.0345, 0.0456, 0.0567, 0.0678];
    let bytes: usize = dims.iter().map(|o| o * in_f).sum();
    let mut sets = Vec::new();
    for c in 0..copies {
        let planes: Vec<_> = dims
            .iter()
            .enumerate()
            .map(|(i, &o)| {
                e.htod_bytes(&synth_e4m3(
                    o * in_f,
                    0x1111 * (i as u32 + 1) + c as u32 * 7,
                ))
            })
            .collect::<Result<_, _>>()?;
        sets.push(planes);
    }
    let x: Vec<f32> = (0..in_f)
        .map(|i| ((i * 7919) % 1000) as f32 / 500.0 - 1.0)
        .collect();
    let x_d = e.htod(&x)?;
    let (aq, ad) = e.quantize_q8_1(&x_d, 1, in_f)?;
    let acth = e.f32_to_f16_lane_major_nat(&x_d, 1, in_f)?;
    let mut outs = [
        e.zeros(dims[0])?,
        e.zeros(dims[1])?,
        e.zeros(dims[2])?,
        e.zeros(dims[3])?,
        e.zeros(dims[4])?,
        e.zeros(dims[5])?,
    ];
    let run = |arm: usize,
               set: usize,
               outs: &mut [cudarc::driver::CudaSlice<f32>; 6]|
     -> Result<(), Box<dyn std::error::Error>> {
        let p = &sets[set];
        let w: [&_; 6] = [&p[0], &p[1], &p[2], &p[3], &p[4], &p[5]];
        match arm {
            0 => e.e4m3_fused6_into_arm(w, &aq, &ad, in_f, dims, in_f, ws, outs, 0),
            1 => e.e4m3_fused6_f16_into(w, &acth, in_f, dims, in_f, ws, outs, false),
            _ => e.e4m3_fused6_f16_into(w, &acth, in_f, dims, in_f, ws, outs, true),
        }
    };
    for _ in 0..5 {
        for arm in 0..3 {
            run(arm, 0, &mut outs)?;
        }
    }
    e.stream().synchronize()?;
    let mut t = [Vec::new(), Vec::new(), Vec::new()];
    for i in 0..iters {
        for (arm, tv) in t.iter_mut().enumerate() {
            let t0 = Instant::now();
            for r in 0..20 {
                run(arm, (i * 20 + r) % copies, &mut outs)?;
            }
            e.stream().synchronize()?;
            tv.push(t0.elapsed().as_secs_f64() * 1e6 / 20.0);
        }
    }
    let (s, f, fa) = (median(&mut t[0]), median(&mut t[1]), median(&mut t[2]));
    println!(
        "[kda6-f16] served e4m3 six {s:.2} us ({:.0} GB/s) | f16 {f:.2} us ({:.0} GB/s, {:.3}x) | f16 acc32 {fa:.2} us ({:.0} GB/s, {:.3}x)  [{} B over the six planes, {iters} x 20 interleaved, {copies} copies]",
        bytes as f64 / s / 1e3,
        bytes as f64 / f / 1e3,
        s / f,
        bytes as f64 / fa / 1e3,
        s / fa,
        bytes
    );
    Ok(())
}
