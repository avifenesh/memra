//! `MEMRA_GLM5_SPEC_DEV_IO` primitives: the launched device words (`u32_words_dev`,
//! `f32_words_dev`, `i32_iota_dev`) land the exact bits a pageable upload would, and the
//! sampled accept's batched drafter-side gather (one `softmax_gather_filtered` launch over k
//! contiguous rows with launched statistics) is bitwise the per-draft launch it replaces.

use memra_engine::Engine;

fn vecf(n: usize, seed: u64) -> Vec<f32> {
    let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..n)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            ((x >> 11) as f64 / (1u64 << 53) as f64) as f32 * 12.0 - 6.0
        })
        .collect()
}

#[test]
fn launched_words_land_exact_bits() {
    let Ok(e) = Engine::new(0) else {
        eprintln!("no CUDA device; skipping");
        return;
    };
    let u: Vec<u32> = vec![0, 7, 4_294_967_295, 151_329, 1, 2, 3, 9];
    let f: Vec<f32> = vec![
        0.0,
        -0.0,
        1.5,
        f32::MIN_POSITIVE,
        f32::NAN,
        1e30,
        -3.25,
        0.1,
    ];
    let ud = e.u32_words_dev(&u).unwrap();
    let fd = e.f32_words_dev(&f).unwrap();
    let id = e.i32_iota_dev(4_000_003, 6).unwrap();
    e.stream().synchronize().unwrap();
    assert_eq!(e.dtoh_u32(&ud).unwrap(), u);
    let fb: Vec<u32> = e.dtoh(&fd).unwrap().iter().map(|v| v.to_bits()).collect();
    let fe: Vec<u32> = f.iter().map(|v| v.to_bits()).collect();
    assert_eq!(
        fb, fe,
        "f32 words must land bit-exact (including NaN and -0.0)"
    );
    assert_eq!(
        e.dtoh_i32(&id).unwrap(),
        vec![
            4_000_003, 4_000_004, 4_000_005, 4_000_006, 4_000_007, 4_000_008
        ]
    );
    println!("launched words: u32/f32/iota bit-exact");
}

#[test]
fn batched_drafter_gather_matches_per_draft_bitwise() {
    let Ok(e) = Engine::new(0) else {
        eprintln!("no CUDA device; skipping");
        return;
    };
    let (k, d_vocab, temp) = (4usize, 4096usize, 0.7f32);
    let logits: Vec<Vec<f32>> = (0..k).map(|j| vecf(d_vocab, 100 + j as u64)).collect();
    let ids: Vec<u32> = vec![17, 4095, 2048, 3];
    let th: Vec<f32> = vec![1.25, -0.5, 3.0, 0.125];
    let z: Vec<f32> = vec![100.5, 42.0, 7.75, 1234.5];
    // per-draft launches, as the shipped accept walk issues them (uploads per draft)
    let mut per: Vec<u32> = Vec::new();
    for j in 0..k {
        let dl = e.htod(&logits[j]).unwrap();
        let idsd = e.htod_u32_v(&[ids[j]]).unwrap();
        let rows0 = e.htod_i32(&[0]).unwrap();
        let thd = e.htod(&[th[j]]).unwrap();
        let zd = e.htod(&[z[j]]).unwrap();
        let mut outd = e.zeros(1).unwrap();
        e.softmax_gather_filtered(
            &dl, d_vocab, &idsd, &rows0, &thd, &zd, &mut outd, d_vocab, 1, temp,
        )
        .unwrap();
        e.stream().synchronize().unwrap();
        per.push(e.dtoh(&outd).unwrap()[0].to_bits());
    }
    // the batched launch over k contiguous rows with launched statistics
    let mut dl_all = e.uninit(k * d_vocab).unwrap();
    for (j, row) in logits.iter().enumerate() {
        let dl = e.htod(row).unwrap();
        e.copy_into(&mut dl_all, j * d_vocab, &dl, d_vocab).unwrap();
    }
    let idq = e.u32_words_dev(&ids).unwrap();
    let rows = e.i32_iota_dev(0, k).unwrap();
    let thq = e.f32_words_dev(&th).unwrap();
    let zq = e.f32_words_dev(&z).unwrap();
    let mut q_d = e.zeros(k).unwrap();
    e.softmax_gather_filtered(
        &dl_all, d_vocab, &idq, &rows, &thq, &zq, &mut q_d, d_vocab, k, temp,
    )
    .unwrap();
    e.stream().synchronize().unwrap();
    let bat: Vec<u32> = e.dtoh(&q_d).unwrap().iter().map(|v| v.to_bits()).collect();
    assert_eq!(
        bat, per,
        "batched drafter gather differs from the per-draft launches"
    );
    assert!(
        per.iter().any(|&b| f32::from_bits(b) > 0.0),
        "vacuous: every q is zero"
    );
    println!("batched drafter gather: k={k} pairs bitwise = per-draft launches");
}
