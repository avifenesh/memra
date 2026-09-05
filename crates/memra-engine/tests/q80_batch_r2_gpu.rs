//! `MEMRA_Q80_BATCH_R2`: the two-rows-per-warp twins of the batched Q8_0 matvec
//! (`qmatvec_q8_0_mmvq_b{2,4,8}_r2` and `_r2_rp`) against the shipped one-row-per-warp kernels,
//! BITWISE, through the real launcher (`qmatvec_batched_raw`: quantize + dispatch), at the MLA
//! projection shapes, an odd out_f tail, both layouts, m = 2..8 (m = 9 and 16 fall through with
//! the counter flat). Engagement is counted, and a perturbed-activation red arm bites THROUGH
//! the door.
//!
//! Run: `flock /tmp/memra-5090.lock cargo test --release -p memra-engine --test q80_batch_r2_gpu -- --ignored --nocapture`
use memra_engine::{Engine, QT_Q8_0};

fn varied(len: usize, seed: u64, spread: f32) -> Vec<f32> {
    let mut s = seed | 1;
    (0..len)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = ((s >> 33) as f32) / ((1u64 << 31) as f32);
            (u - 0.5) * spread
        })
        .collect()
}

/// GGUF-layout Q8_0 rows: per 32-element block an f16 scale then 32 int8 quants (34 B).
fn q8_0_bytes(out_f: usize, in_f: usize, seed: u64) -> Vec<u8> {
    let nblk = in_f / 32;
    let mut s = seed | 1;
    let mut v = Vec::with_capacity(out_f * nblk * 34);
    for _ in 0..out_f * nblk {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        // an f16 scale built from its bit pattern: exponent field 7 (2^-8) with a random
        // mantissa, i.e. values in [0.0039, 0.0078); no f32 -> f16 conversion needed
        let scale_bits: u16 = 0x1C00 | (((s >> 40) & 0x03ff) as u16);
        v.extend_from_slice(&scale_bits.to_le_bytes());
        for _ in 0..32 {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            v.push(((s >> 40) & 0xff) as u8);
        }
    }
    v
}

/// The split-plane (rp) permutation of the same rows: the quant plane (32 B per block) then the
/// f16 scale plane, the byte order `q8_0_split_rp_build` writes.
fn q8_0_rp_bytes(base: &[u8], out_f: usize, in_f: usize) -> Vec<u8> {
    let nblk = in_f / 32;
    let n = out_f * nblk;
    let mut v = vec![0u8; n * 34];
    for i in 0..n {
        let b = &base[i * 34..(i + 1) * 34];
        v[i * 32..(i + 1) * 32].copy_from_slice(&b[2..34]);
        v[n * 32 + i * 2] = b[0];
        v[n * 32 + i * 2 + 1] = b[1];
    }
    v
}

fn set(key: &str, v: &str) {
    unsafe { std::env::set_var(key, v) };
}

fn bit_diffs(a: &[f32], b: &[f32]) -> usize {
    a.iter()
        .zip(b)
        .filter(|(x, y)| x.to_bits() != y.to_bits())
        .count()
}

#[test]
#[ignore = "needs a CUDA device, run under flock /tmp/memra-5090.lock"]
fn gpu_q80_batch_r2_matches_one_row_per_warp_bitwise() {
    let e = Engine::new(0).expect("CUDA engine on device 0");
    // the MLA/DSA projection shapes (q_a 4096->1536, kv_a 4096->576-class, o_proj 16384->4096,
    // q_b 1536->12288-class) plus an ODD out_f so the last warp owns one row
    let shapes = [
        (4096usize, 1536usize),
        (4096, 576),
        (16384, 4096),
        (1536, 2048),
        (4096, 133),
    ];
    let ms = [2usize, 3, 4, 5, 8, 9, 16];
    let mut cells = 0usize;
    for (si, &(in_f, out_f)) in shapes.iter().enumerate() {
        let base = q8_0_bytes(out_f, in_f, 0x80 + si as u64);
        let rpb = q8_0_rp_bytes(&base, out_f, in_f);
        let row_bytes = in_f / 32 * 34;
        let w_base = e.htod_bytes(&base).expect("base upload");
        let w_rp = e.htod_bytes(&rpb).expect("rp upload");
        for &m in &ms {
            let x = e
                .htod(&varied(m * in_f, 0xB0 + m as u64, 2.0))
                .expect("activation upload");
            let mcols = Engine::batched_mcols(m);
            for (rp, w) in [(false, &w_base), (true, &w_rp)] {
                set("MEMRA_Q80_BATCH_R2", "0");
                let y0 = e
                    .qmatvec_batched_raw(w, &x, m, in_f, out_f, QT_Q8_0, row_bytes, mcols, rp)
                    .expect("shipped batched");
                let d0 = memra_engine::q80_batch_r2_dispatches();
                set("MEMRA_Q80_BATCH_R2", "1");
                let y1 = e
                    .qmatvec_batched_raw(w, &x, m, in_f, out_f, QT_Q8_0, row_bytes, mcols, rp)
                    .expect("r2 batched");
                set("MEMRA_Q80_BATCH_R2", "0");
                let d1 = memra_engine::q80_batch_r2_dispatches();
                if m <= 8 {
                    assert!(
                        d1 > d0,
                        "the r2 door did not engage at {in_f}x{out_f} m={m} rp={rp}"
                    );
                } else {
                    assert_eq!(d1, d0, "the r2 door engaged at m={m} (b16 has no r2 twin)");
                }
                let h0 = e.dtoh(&y0).expect("y0");
                let h1 = e.dtoh(&y1).expect("y1");
                assert_eq!(h0.len(), m * out_f);
                let nz = h0.iter().filter(|v| **v != 0.0 && v.is_finite()).count();
                assert!(
                    nz > h0.len() / 2,
                    "vacuous fixture at {in_f}x{out_f} m={m} rp={rp}: nz={nz}"
                );
                let bad = bit_diffs(&h0, &h1);
                assert_eq!(
                    bad,
                    0,
                    "{in_f}x{out_f} m={m} rp={rp}: {bad}/{} words differ",
                    h0.len()
                );
                cells += 1;
            }
        }
    }
    println!(
        "q8_0 batch r2 PASS: {cells} (shape, m, layout) cells bitwise against the one-row-per-warp kernels"
    );

    // RED ARM through the door: one activation value moved must move row 0's outputs only.
    let (in_f, out_f, m) = (4096usize, 1536usize, 4usize);
    let base = q8_0_bytes(out_f, in_f, 0x80);
    let w = e.htod_bytes(&base).expect("weight upload");
    let row_bytes = in_f / 32 * 34;
    let mut xs = varied(m * in_f, 0xB4, 2.0);
    let x = e.htod(&xs).expect("x");
    set("MEMRA_Q80_BATCH_R2", "1");
    let y_ref = e
        .qmatvec_batched_raw(&w, &x, m, in_f, out_f, QT_Q8_0, row_bytes, 4, false)
        .expect("r2 ref");
    xs[7] += 1.0;
    let xp = e.htod(&xs).expect("x perturbed");
    let y_red = e
        .qmatvec_batched_raw(&w, &xp, m, in_f, out_f, QT_Q8_0, row_bytes, 4, false)
        .expect("r2 red");
    set("MEMRA_Q80_BATCH_R2", "0");
    let (hr, hp) = (e.dtoh(&y_ref).expect("ref"), e.dtoh(&y_red).expect("red"));
    let moved = bit_diffs(&hr[..out_f], &hp[..out_f]);
    assert!(
        moved > out_f / 2,
        "red arm: only {moved}/{out_f} row-0 outputs moved"
    );
    let same = bit_diffs(&hr[out_f..], &hp[out_f..]);
    assert_eq!(
        same, 0,
        "red arm: rows 1.. moved ({same}) although only row 0 changed"
    );
    println!("q8_0 batch r2 RED bites: {moved}/{out_f} row-0 outputs moved, rows 1.. unchanged");
}
