//! Gate for `MEMRA_MOE_DOWN_ILP2` (lane/moe-down-ilp2-20260905): the two-experts-in-flight down
//! twins against the shipped `_ilp` twins, BITWISE, at the served down shape (in_f 2048 -> out_f
//! 4096, NVFP4 v1 rows of 1152 B), t = 1..4, n_used 8 (even) and 7 (odd tail), both the plain and
//! the `_w4` packed launch; a red arm (two experts swapped in the slot order must bite, since the
//! chain is slot-ordered); and the door's non-vacuity (its counter moves).
use cudarc::driver::DevicePtr;
use memra_engine::{Engine, QT_NVFP4};
use std::sync::atomic::Ordering;

struct Lcg(u32);
impl Lcg {
    fn byte(&mut self) -> u8 {
        self.0 = self.0.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (self.0 >> 24) as u8
    }
}
fn safe_e4m3(b: u8) -> u8 {
    // keep the UE4M3 scale finite and moderate: exponent bits 0..=8
    (b & 0x07) | 0x38
}
fn synth_rows(out_f: usize, in_f: usize, seed: u32) -> (Vec<u8>, usize) {
    let nsb64 = in_f / 64;
    let row_bytes = nsb64 * 36;
    let mut w = vec![0u8; out_f * row_bytes];
    let mut r = Lcg(seed);
    for chunk in w.chunks_exact_mut(36) {
        for d in &mut chunk[0..4] {
            *d = safe_e4m3(r.byte());
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
fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

#[allow(clippy::too_many_arguments)]
fn run(
    e: &Engine,
    experts: &[cudarc::driver::CudaSlice<u8>],
    order: &[usize],
    t: usize,
    n_used: usize,
    in_f: usize,
    out_f: usize,
    rb: usize,
    ilp2: bool,
    packed: bool,
) -> Vec<f32> {
    let n_pairs = t * n_used;
    let stream = e.stream();
    let mut ptrs = vec![0u64; 3 * n_pairs];
    let mut scl = vec![0f32; 3 * n_pairs];
    for pr in 0..n_pairs {
        let (p, _g) = experts[order[pr % order.len()]].device_ptr(&stream);
        ptrs[2 * n_pairs + pr] = p;
        scl[2 * n_pairs + pr] = 0.125 + 0.01 * (pr as f32);
    }
    let ptrs_d = e.htod_u64(&ptrs).unwrap();
    let scl_d = e.htod(&scl).unwrap();
    let act = vecf(n_pairs * in_f, 77);
    let act_d = e.htod(&act).unwrap();
    let (aq2, ad2) = e.quantize_q8_1(&act_d, n_pairs, in_f).unwrap();
    let mut dst = e.zeros(t * out_f).unwrap();
    unsafe {
        std::env::set_var("MEMRA_MOE_VROWS_ILP", "1");
        std::env::set_var("MEMRA_MOE_VROWS_PACK", if packed { "1" } else { "0" });
        std::env::set_var("MEMRA_MOE_DOWN_ILP2", if ilp2 { "1" } else { "0" });
    }
    e.moe_down8_fma_q8_rows(
        &ptrs_d, &scl_d, &aq2, &ad2, &mut dst, in_f, out_f, n_used, n_pairs, QT_NVFP4, rb,
    )
    .unwrap();
    unsafe {
        std::env::set_var("MEMRA_MOE_DOWN_ILP2", "0");
    }
    e.stream().synchronize().unwrap();
    e.dtoh(&dst).unwrap()
}

#[test]
fn moe_down_ilp2_matches_ilp_bitwise() {
    let Ok(e) = Engine::new(0) else {
        eprintln!("no CUDA device; skipping");
        return;
    };
    let (in_f, out_f) = (2048usize, 4096usize);
    let (rows, rb) = synth_rows(out_f, in_f, 0x5151);
    let experts: Vec<_> = (0..8u32)
        .map(|j| {
            let mut d = rows.clone();
            d[5] ^= (j + 1) as u8;
            d[rb * 7 + 9] ^= (3 * j + 1) as u8;
            e.htod_bytes(&d).unwrap()
        })
        .collect();
    let order: Vec<usize> = (0..8).collect();
    let c0 = memra_engine::MOE_DOWN_ILP2_DISPATCHES.load(Ordering::Relaxed);
    for packed in [false, true] {
        for n_used in [8usize, 7] {
            for t in 1..=4usize {
                let a = run(
                    &e, &experts, &order, t, n_used, in_f, out_f, rb, false, packed,
                );
                let b = run(
                    &e, &experts, &order, t, n_used, in_f, out_f, rb, true, packed,
                );
                assert!(
                    a.iter().all(|v| v.is_finite()),
                    "non-finite output in the ilp arm"
                );
                assert_eq!(
                    bits(&a),
                    bits(&b),
                    "ilp2 differs from ilp (packed={packed} n_used={n_used} t={t})"
                );
            }
        }
    }
    let c1 = memra_engine::MOE_DOWN_ILP2_DISPATCHES.load(Ordering::Relaxed);
    assert!(
        c1 > c0,
        "VACUOUS: the ilp2 door never dispatched ({c0} -> {c1})"
    );
    // red arm: swapping two experts in the slot order changes the slot-ordered chain
    let mut swapped = order.clone();
    swapped.swap(2, 5);
    let a = run(&e, &experts, &order, 2, 8, in_f, out_f, rb, true, false);
    let r = run(&e, &experts, &swapped, 2, 8, in_f, out_f, rb, true, false);
    assert_ne!(
        bits(&a),
        bits(&r),
        "red arm: swapped slot order did not bite"
    );
    println!("moe_down ilp2: bitwise = ilp at t=1..4, n_used 8 and 7, plain and w4; red arm bites");
}
