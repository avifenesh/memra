//! `MEMRA_E4M3_BATCH_R2`: the two-rows-per-warp twins of the batched F8-E4M3 matvec
//! (`qmatvec_e4m3_mmvq_b{2,4,8}_r2`) against the shipped one-row-per-warp kernels, BITWISE,
//! through the real launcher (`qmatvec_batched_raw`: quantize + dispatch), at the KDA six
//! shapes, an odd out_f tail, and m = 2..8 (m = 9 and 16 fall through to the shipped b16 and
//! must stay identical with the counter flat). Engagement is counted, and a perturbed
//! activation red arm bites THROUGH the door.
//!
//! Run: `flock /tmp/memra-5090.lock cargo test --release -p memra-engine --test e4m3_batch_r2_gpu -- --ignored --nocapture`
use memra_engine::{Engine, QT_F8_E4M3};

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

/// Random e4m3 bytes with the two NaN codes (0x7f, 0xff) mapped to +0, so every product is
/// finite and a bit-compare is a compare of numbers.
fn e4m3_bytes(len: usize, seed: u64) -> Vec<u8> {
    let mut s = seed | 1;
    (0..len)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let b = (s >> 40) as u8;
            if b & 0x7f == 0x7f { 0 } else { b }
        })
        .collect()
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
fn gpu_e4m3_batch_r2_matches_one_row_per_warp_bitwise() {
    let e = Engine::new(0).expect("CUDA engine on device 0");
    // the KDA six's shapes (q/k/v 8192, f_a/g_a 128, b 64 at in_f 4096), a narrower in_f, and an
    // ODD out_f so the last warp owns one row
    let shapes = [
        (4096usize, 8192usize),
        (4096, 128),
        (4096, 64),
        (2048, 1024),
        (4096, 133),
    ];
    let ms = [2usize, 3, 4, 5, 8, 9, 16];
    let mut cells = 0usize;
    for (si, &(in_f, out_f)) in shapes.iter().enumerate() {
        let w = e
            .htod_bytes(&e4m3_bytes(in_f * out_f, 0xE4 + si as u64))
            .expect("weight upload");
        for &m in &ms {
            let x = e
                .htod(&varied(m * in_f, 0xA0 + m as u64, 2.0))
                .expect("activation upload");
            let mcols = Engine::batched_mcols(m);
            set("MEMRA_E4M3_BATCH_R2", "0");
            let y0 = e
                .qmatvec_batched_raw(&w, &x, m, in_f, out_f, QT_F8_E4M3, in_f, mcols, false)
                .expect("shipped batched");
            let d0 = memra_engine::e4m3_batch_r2_dispatches();
            set("MEMRA_E4M3_BATCH_R2", "1");
            let y1 = e
                .qmatvec_batched_raw(&w, &x, m, in_f, out_f, QT_F8_E4M3, in_f, mcols, false)
                .expect("r2 batched");
            set("MEMRA_E4M3_BATCH_R2", "0");
            let d1 = memra_engine::e4m3_batch_r2_dispatches();
            if m <= 8 {
                assert!(
                    d1 > d0,
                    "the r2 door did not engage at {in_f}x{out_f} m={m}"
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
                "vacuous fixture at {in_f}x{out_f} m={m}: nz={nz}"
            );
            let bad = bit_diffs(&h0, &h1);
            assert_eq!(
                bad,
                0,
                "{in_f}x{out_f} m={m}: {bad}/{} words differ",
                h0.len()
            );
            cells += 1;
        }
    }
    println!(
        "e4m3 batch r2 PASS: {cells} (shape, m) cells bitwise against the one-row-per-warp kernels"
    );

    // RED ARM through the door: one activation value moved must move row 0's outputs only.
    let (in_f, out_f, m) = (4096usize, 8192usize, 4usize);
    let w = e
        .htod_bytes(&e4m3_bytes(in_f * out_f, 0xE4))
        .expect("weight upload");
    let mut xs = varied(m * in_f, 0xA4, 2.0);
    let x = e.htod(&xs).expect("x");
    set("MEMRA_E4M3_BATCH_R2", "1");
    let y_ref = e
        .qmatvec_batched_raw(&w, &x, m, in_f, out_f, QT_F8_E4M3, in_f, 4, false)
        .expect("r2 ref");
    xs[7] += 1.0;
    let xp = e.htod(&xs).expect("x perturbed");
    let y_red = e
        .qmatvec_batched_raw(&w, &xp, m, in_f, out_f, QT_F8_E4M3, in_f, 4, false)
        .expect("r2 red");
    set("MEMRA_E4M3_BATCH_R2", "0");
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
    println!("e4m3 batch r2 RED bites: {moved}/{out_f} row-0 outputs moved, rows 1.. unchanged");
}
