//! Gate for `MEMRA_MOE_GATEUP_ILP2` (lane/moe-gateup-ilp2-20260905): the two-pairs-per-warp gate/up
//! twins against the shipped `_ilp` twins, BITWISE, at the served gate/up shape (in_f 4096 -> n_ff
//! 2048, NVFP4 v1 rows of 2304 B per plane), t = 1..4, n_used 8 (even) and 7 (odd: pairs straddle
//! tokens and the last pair runs the `_ilp` warp), both the plain and the `_w4` packed launch; a red
//! arm (a perturbed up plane must bite; the preclamp limit is 1e30 here so the synthetic rows'
//! large dot products are not clamped to a constant that would hide the perturbation, and the
//! ops are the same whatever the limit); and the door's non-vacuity (its counter moves).
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
    gates: &[cudarc::driver::CudaSlice<u8>],
    ups: &[cudarc::driver::CudaSlice<u8>],
    t: usize,
    n_used: usize,
    in_f: usize,
    n_ff: usize,
    rb: usize,
    ilp2: bool,
    packed: bool,
) -> Vec<f32> {
    let n_pairs = t * n_used;
    let stream = e.stream();
    let mut ptrs = vec![0u64; 3 * n_pairs];
    let mut scl = vec![0f32; 3 * n_pairs];
    for pr in 0..n_pairs {
        let (pg, _g) = gates[pr % gates.len()].device_ptr(&stream);
        let (pu, _u) = ups[pr % ups.len()].device_ptr(&stream);
        ptrs[pr] = pg;
        ptrs[n_pairs + pr] = pu;
        scl[pr] = 0.5 + 0.01 * (pr as f32);
        scl[n_pairs + pr] = 0.25 + 0.02 * (pr as f32);
    }
    let ptrs_d = e.htod_u64(&ptrs).unwrap();
    let scl_d = e.htod(&scl).unwrap();
    let act = vecf(t * in_f, 91);
    let act_d = e.htod(&act).unwrap();
    let (aq, ad) = e.quantize_q8_1(&act_d, t, in_f).unwrap();
    unsafe {
        std::env::set_var("MEMRA_MOE_VROWS_ILP", "1");
        std::env::set_var("MEMRA_MOE_VROWS_PACK", if packed { "1" } else { "0" });
        std::env::set_var("MEMRA_MOE_VROWS_ORD", "0");
        std::env::set_var("MEMRA_MOE_GATEUP_ILP2", if ilp2 { "1" } else { "0" });
    }
    let out = e
        .moe_gate_up_preclamp8_q8_rows(
            &ptrs_d, &scl_d, &aq, &ad, 1e30, in_f, n_ff, n_used, n_pairs, QT_NVFP4, QT_NVFP4, rb,
            rb,
        )
        .unwrap();
    unsafe {
        std::env::set_var("MEMRA_MOE_GATEUP_ILP2", "0");
    }
    e.stream().synchronize().unwrap();
    e.dtoh(&out).unwrap()
}

#[test]
fn moe_gateup_ilp2_matches_ilp_bitwise() {
    let Ok(e) = Engine::new(0) else {
        eprintln!("no CUDA device; skipping");
        return;
    };
    let (in_f, n_ff) = (4096usize, 2048usize);
    let (rows, rb) = synth_rows(n_ff, in_f, 0x6161);
    let mk = |salt: u8| -> Vec<cudarc::driver::CudaSlice<u8>> {
        (0..8u32)
            .map(|j| {
                let mut d = rows.clone();
                d[5] ^= (j + 1) as u8 ^ salt;
                d[rb * 7 + 9] ^= (3 * j + 1) as u8;
                e.htod_bytes(&d).unwrap()
            })
            .collect()
    };
    let gates = mk(0x10);
    let ups = mk(0x20);
    let c0 = memra_engine::MOE_GATEUP_ILP2_DISPATCHES.load(Ordering::Relaxed);
    for packed in [false, true] {
        for n_used in [8usize, 7] {
            for t in 1..=4usize {
                let a = run(&e, &gates, &ups, t, n_used, in_f, n_ff, rb, false, packed);
                let b = run(&e, &gates, &ups, t, n_used, in_f, n_ff, rb, true, packed);
                assert!(
                    a.iter().all(|v| v.is_finite()),
                    "non-finite output in the ilp arm"
                );
                assert_eq!(
                    bits(&a),
                    bits(&b),
                    "gate/up ilp2 differs from ilp (packed={packed} n_used={n_used} t={t})"
                );
            }
        }
    }
    let c1 = memra_engine::MOE_GATEUP_ILP2_DISPATCHES.load(Ordering::Relaxed);
    assert!(
        c1 > c0,
        "VACUOUS: the gate/up ilp2 door never dispatched ({c0} -> {c1})"
    );
    // red arm: a perturbed up plane changes the output
    let ups2 = mk(0x21);
    let a = run(&e, &gates, &ups, 2, 8, in_f, n_ff, rb, true, false);
    let r = run(&e, &gates, &ups2, 2, 8, in_f, n_ff, rb, true, false);
    assert_ne!(
        bits(&a),
        bits(&r),
        "red arm: perturbed up plane did not bite"
    );
    println!(
        "moe_gate_up ilp2: bitwise = ilp at t=1..4, n_used 8 and 7, plain and w4; red arm bites"
    );
}
