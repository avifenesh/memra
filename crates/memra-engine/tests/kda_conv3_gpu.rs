//! `memra_kda_conv_silu_decode3_f32` (MEMRA_KDA_CONV3) vs three `memra_kda_conv_silu_decode_f32`
//! launches, BITWISE on the three outputs AND the rolled ring, over several steps (so the ring
//! that the fused form rolls feeds its own next step exactly as the per-plane form's does).
use memra_engine::Engine;

fn gpu_guard() -> std::sync::MutexGuard<'static, ()> {
    static M: std::sync::Mutex<()> = std::sync::Mutex::new(());
    M.lock().unwrap_or_else(|p| p.into_inner())
}

fn lcg(n: usize, seed: u64, scale: f32) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (((s >> 40) as f32) / ((1u64 << 24) as f32) * 2.0 - 1.0) * scale
        })
        .collect()
}

#[test]
#[ignore = "needs a CUDA device, run under flock /tmp/memra-5090.lock"]
fn conv3_matches_three_launches_bitwise() {
    let _g = gpu_guard();
    let e = Engine::new(0).expect("CUDA engine on device 0");
    let (qkv, kernel) = (8192usize, 4usize); // the glm5_next geometry: 64 heads x 128, conv 4
    let pad = kernel - 1;
    let w = e.htod(&lcg(3 * qkv * kernel, 0xC0FFEE, 0.7)).expect("w");
    let ring0 = lcg(3 * qkv * pad, 0x51DE, 1.3);
    let mut ring_a = e.htod(&ring0).expect("ring a");
    let mut ring_b = e.htod(&ring0).expect("ring b");
    let mut mismatches = 0usize;
    for step in 0..6u64 {
        let xs: Vec<_> = (0..3)
            .map(|p| e.htod(&lcg(qkv, 0x7E57 + step * 7 + p, 2.1)).expect("x"))
            .collect();
        let mut ya: Vec<_> = (0..3)
            .map(|_| e.htod(&vec![0.0f32; qkv]).unwrap())
            .collect();
        let mut yb: Vec<_> = (0..3)
            .map(|_| e.htod(&vec![0.0f32; qkv]).unwrap())
            .collect();
        for p in 0..3 {
            e.kda_conv_silu_decode(&xs[p], &mut ring_a, &w, &mut ya[p], qkv, kernel, p)
                .expect("per-plane");
        }
        let d0 = memra_engine::kda::kda_conv3_dispatches();
        let [y0, y1, y2] = yb.as_mut_slice() else {
            unreachable!()
        };
        e.kda_conv_silu_decode3(
            [&xs[0], &xs[1], &xs[2]],
            &mut ring_b,
            &w,
            [y0, y1, y2],
            qkv,
            kernel,
        )
        .expect("fused");
        assert!(
            memra_engine::kda::kda_conv3_dispatches() > d0,
            "the conv3 launcher did not count"
        );
        for p in 0..3 {
            let (a, b) = (e.dtoh(&ya[p]).unwrap(), e.dtoh(&yb[p]).unwrap());
            let nz = a.iter().filter(|v| **v != 0.0).count();
            assert!(
                nz > qkv / 2,
                "step {step} plane {p}: vacuous output ({nz} nonzero)"
            );
            mismatches += a
                .iter()
                .zip(&b)
                .filter(|(x, y)| x.to_bits() != y.to_bits())
                .count();
        }
        let (ra, rb) = (e.dtoh(&ring_a).unwrap(), e.dtoh(&ring_b).unwrap());
        mismatches += ra
            .iter()
            .zip(&rb)
            .filter(|(x, y)| x.to_bits() != y.to_bits())
            .count();
        assert_eq!(
            mismatches, 0,
            "step {step}: conv3 diverged from the per-plane launches"
        );
    }
    println!("[kda-conv3] 6 steps x 3 planes x {qkv} channels + the ring: bit-identical");
}
