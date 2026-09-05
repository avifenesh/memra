//! Gate for `MEMRA_KDA_NARROW_Q8` (lane/kda-narrow-q8-20260905): the narrow in-kernel-quantize
//! Q8_0 MMVQ (`qmatvec_q8_0_mmvq_f32in_narrow`) against `matmul` on the same plain-layout Q8_0
//! tensor (which runs `quantize_q8_1` then `qmatvec_q8_0_mmvq`), BITWISE, at the KDA low-rank
//! shapes (in_f 128 -> out_f 8192, and 256 -> 64) at m = 1 (the MMVQ fast path is m = 1 only, as
//! `mmvq_fast_eligible` says; the kernel's m > 1 rows are the raw launcher's, unreferenced here); a red arm (a perturbed weight row
//! must bite); shape refusal (in_f 512 and m 17 return None and launch nothing); and the door's
//! non-vacuity through the raw launcher counter is the kda site's job (covered by the served tape).
use memra_engine::model::GpuTensor;
use memra_engine::{Engine, QT_Q8_0};

fn vecf(n: usize, seed: u64, amp: f32) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * amp
        })
        .collect()
}
fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}
fn q8_tensor(e: &Engine, w: &[f32], in_f: usize, out_f: usize) -> GpuTensor {
    let bytes = memra_gguf::nvfp4_repack::f32_to_q8_0(w);
    assert_eq!(bytes.len(), out_f * in_f / 32 * 34);
    GpuTensor::Quant {
        bytes: e.htod_bytes(&bytes).unwrap(),
        qtype: QT_Q8_0,
        row_bytes: in_f / 32 * 34,
        ne: vec![in_f as u64, out_f as u64],
        scale: 1.0,
        rp: false,
        #[cfg(memra_cutlass)]
        cutlass: None,
        fp8: None,
        rp4: None,
        blk: None,
        f16: None,
    }
}

#[test]
fn q8_narrow_f32in_matches_quantize_then_mmvq_bitwise() {
    let Ok(e) = Engine::new(0) else {
        eprintln!("no CUDA device; skipping");
        return;
    };
    unsafe {
        std::env::set_var("MEMRA_FAST", "1");
    }
    for (in_f, out_f) in [(128usize, 8192usize), (256, 64)] {
        let wf = vecf(out_f * in_f, 41, 2.0);
        let w = q8_tensor(&e, &wf, in_f, out_f);
        {
            let m = 1usize;
            let x = e.htod(&vecf(m * in_f, 7 + m as u64, 8.0)).unwrap();
            let a = e.matmul(&w, &x, m).unwrap();
            let b = e
                .matmul_q8_narrow_f32in(&w, &x, m)
                .unwrap()
                .expect("narrow twin refused a fitting shape");
            e.stream().synchronize().unwrap();
            let (ha, hb) = (e.dtoh(&a).unwrap(), e.dtoh(&b).unwrap());
            assert!(ha.iter().all(|v| v.is_finite()));
            assert_eq!(
                bits(&ha),
                bits(&hb),
                "narrow twin differs ({in_f}->{out_f} m={m})"
            );
        }
        // red arm
        let mut wf2 = wf.clone();
        wf2[3 * in_f + 5] += 1.0;
        let w2 = q8_tensor(&e, &wf2, in_f, out_f);
        let x = e.htod(&vecf(in_f, 8, 8.0)).unwrap();
        let b = e.matmul_q8_narrow_f32in(&w, &x, 1).unwrap().unwrap();
        let r = e.matmul_q8_narrow_f32in(&w2, &x, 1).unwrap().unwrap();
        e.stream().synchronize().unwrap();
        assert_ne!(
            bits(&e.dtoh(&b).unwrap()),
            bits(&e.dtoh(&r).unwrap()),
            "red arm did not bite"
        );
    }
    // refusals
    let w = q8_tensor(&e, &vecf(8 * 512, 3, 1.0), 512, 8);
    let x = e.htod(&vecf(512, 4, 1.0)).unwrap();
    assert!(e.matmul_q8_narrow_f32in(&w, &x, 1).unwrap().is_none());
    let w = q8_tensor(&e, &vecf(8 * 128, 3, 1.0), 128, 8);
    let x = e.htod(&vecf(17 * 128, 4, 1.0)).unwrap();
    assert!(e.matmul_q8_narrow_f32in(&w, &x, 17).unwrap().is_none());
    println!(
        "q8 narrow f32in: bitwise = quantize_q8_1 + qmatvec_q8_0_mmvq at 128->8192 and 256->64 (m=1); red arm bites"
    );
}
