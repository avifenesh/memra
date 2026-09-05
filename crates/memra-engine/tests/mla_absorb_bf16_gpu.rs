//! Gate for `MEMRA_MLA_ABSORB_BF16` (lane/mla-absorb-bf16-20260905): the BF16-plane `_wp`
//! decode kernels are BIT-IDENTICAL to the f32 `_wp` kernels when the f32 plane is an exact
//! widening of BF16 (what the loader builds the copy from), at the served split and at split 1,
//! for both absorb_q and decompress_v; red arm: a perturbed BF16 plane moves the output.
use cudarc::driver::{DevicePtr, DevicePtrMut};
use memra_engine::Engine;
use memra_engine::mla_ffi as k;
use std::os::raw::c_void;

const NH: usize = 8;
const DN: usize = 256;
const KR: usize = 512;
const DV: usize = 256;

fn bf16_bits(n: usize, seed: u64) -> Vec<u16> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let frac = (s >> 40) as f32 / (1u64 << 24) as f32 - 0.5;
            let kexp = ((s >> 8) % 9) as i32 - 4;
            let v = frac * (2.0f32).powi(kexp);
            (v.to_bits() >> 16) as u16
        })
        .collect()
}

fn widen(bits: &[u16]) -> Vec<f32> {
    bits.iter()
        .map(|b| f32::from_bits((*b as u32) << 16))
        .collect()
}

fn vecf(n: usize, seed: u64) -> Vec<f32> {
    widen(&bf16_bits(n, seed))
        .iter()
        .map(|v| v * 1.37 + 0.001)
        .collect()
}

#[test]
fn bf16_absorb_planes_match_the_f32_wp_kernels_bitwise() {
    let Ok(e) = Engine::new(0) else {
        eprintln!("no CUDA device; skipping");
        return;
    };
    let stream = e.stream();
    let sp = stream.cu_stream() as *mut c_void;
    let wk_bits = bf16_bits(NH * KR * DN, 21);
    let wv_bits = bf16_bits(NH * DV * KR, 22);
    let wk32 = e.htod(&widen(&wk_bits)).unwrap();
    let wv32 = e.htod(&widen(&wv_bits)).unwrap();
    let wk16 = e.htod_u16(&wk_bits).unwrap();
    let wv16 = e.htod_u16(&wv_bits).unwrap();
    let q_nope = e.htod(&vecf(NH * DN, 23)).unwrap();
    let o_lat = e.htod(&vecf(NH * KR, 24)).unwrap();
    for split in [1i32, 16] {
        let mut a = e.uninit(NH * KR).unwrap();
        let mut b = e.uninit(NH * KR).unwrap();
        let mut c = e.uninit(NH * DV).unwrap();
        let mut d = e.uninit(NH * DV).unwrap();
        unsafe {
            assert_eq!(
                0,
                k::memra_mla_absorb_q_wp_f32(
                    q_nope.device_ptr(&stream).0 as *const f32,
                    wk32.device_ptr(&stream).0 as *const f32,
                    a.device_ptr_mut(&stream).0 as *mut f32,
                    1,
                    NH as i32,
                    DN as i32,
                    KR as i32,
                    split,
                    sp
                )
            );
            assert_eq!(
                0,
                k::memra_mla_absorb_q_wp_bf16(
                    q_nope.device_ptr(&stream).0 as *const f32,
                    wk16.device_ptr(&stream).0 as *const u16,
                    b.device_ptr_mut(&stream).0 as *mut f32,
                    1,
                    NH as i32,
                    DN as i32,
                    KR as i32,
                    split,
                    sp
                )
            );
            assert_eq!(
                0,
                k::memra_mla_decompress_v_wp_f32(
                    o_lat.device_ptr(&stream).0 as *const f32,
                    wv32.device_ptr(&stream).0 as *const f32,
                    c.device_ptr_mut(&stream).0 as *mut f32,
                    1,
                    NH as i32,
                    DV as i32,
                    KR as i32,
                    split,
                    sp
                )
            );
            assert_eq!(
                0,
                k::memra_mla_decompress_v_wp_bf16(
                    o_lat.device_ptr(&stream).0 as *const f32,
                    wv16.device_ptr(&stream).0 as *const u16,
                    d.device_ptr_mut(&stream).0 as *mut f32,
                    1,
                    NH as i32,
                    DV as i32,
                    KR as i32,
                    split,
                    sp
                )
            );
        }
        stream.synchronize().unwrap();
        let (a, b, c, d) = (
            e.dtoh(&a).unwrap(),
            e.dtoh(&b).unwrap(),
            e.dtoh(&c).unwrap(),
            e.dtoh(&d).unwrap(),
        );
        let bad_q = a
            .iter()
            .zip(&b)
            .filter(|(x, y)| x.to_bits() != y.to_bits())
            .count();
        let bad_v = c
            .iter()
            .zip(&d)
            .filter(|(x, y)| x.to_bits() != y.to_bits())
            .count();
        assert_eq!(
            (bad_q, bad_v),
            (0, 0),
            "split {split}: absorb {bad_q}/{} decompress {bad_v}/{} words differ",
            a.len(),
            c.len()
        );
        assert!(
            a.iter().any(|v| v.abs() > 1e-6) && c.iter().any(|v| v.abs() > 1e-6),
            "vacuous"
        );
    }
    // red arm: perturb one BF16 word of the plane
    let mut wk2 = wk_bits.clone();
    wk2[5] ^= 0x0040;
    let wk16b = e.htod_u16(&wk2).unwrap();
    let mut b0 = e.uninit(NH * KR).unwrap();
    let mut b1 = e.uninit(NH * KR).unwrap();
    unsafe {
        assert_eq!(
            0,
            k::memra_mla_absorb_q_wp_bf16(
                q_nope.device_ptr(&stream).0 as *const f32,
                wk16.device_ptr(&stream).0 as *const u16,
                b0.device_ptr_mut(&stream).0 as *mut f32,
                1,
                NH as i32,
                DN as i32,
                KR as i32,
                16,
                sp
            )
        );
        assert_eq!(
            0,
            k::memra_mla_absorb_q_wp_bf16(
                q_nope.device_ptr(&stream).0 as *const f32,
                wk16b.device_ptr(&stream).0 as *const u16,
                b1.device_ptr_mut(&stream).0 as *mut f32,
                1,
                NH as i32,
                DN as i32,
                KR as i32,
                16,
                sp
            )
        );
    }
    stream.synchronize().unwrap();
    let (b0, b1) = (e.dtoh(&b0).unwrap(), e.dtoh(&b1).unwrap());
    assert!(
        b0.iter().zip(&b1).any(|(x, y)| x.to_bits() != y.to_bits()),
        "red arm: a perturbed plane did not move the output"
    );
}
