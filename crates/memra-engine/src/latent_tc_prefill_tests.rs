use crate::Engine;
use crate::latent_nvfp4_ffi::{Nvfp4LatentOps, Nvfp4LatentStorage};
use memra_kv::latent_layout::{LatentBasis, rht512_forward, rht512_inverse};
use memra_kv::latent_nvfp4::PackedRow;

#[test]
#[ignore = "requires an admitted non-production CUDA device and canonical GPU lock"]
fn tensor_core_prefill_chain_uses_compressed_history_with_existing_prefix() {
    use crate::hybrid::HybridModel;
    let e = Engine::new(0).expect("CUDA required");
    let (r, nh, dn, dv, t, prefix, slots) = (512, 4, 32, 32, 16, 19, 24);
    let rows = prefix + t;
    let values = |n: usize, salt: usize| {
        (0..n)
            .map(|i| ((i * 43 + salt) % 997) as f32 / 997.0 - 0.5)
            .collect::<Vec<_>>()
    };
    let history = values(rows * r, 17);
    let decoded: Vec<_> = history
        .chunks_exact(r)
        .flat_map(|row| {
            PackedRow::encode(&rht512_forward(row).unwrap())
                .unwrap()
                .decode()
                .unwrap()
        })
        .collect();
    let mut plane = Nvfp4LatentStorage::new_with_basis(&e, r, rows, LatentBasis::Rht512V1).unwrap();
    plane
        .append(&e, &e.htod(&history[..prefix * r]).unwrap(), 0)
        .unwrap();
    plane
        .append(&e, &e.htod(&history[prefix * r..]).unwrap(), prefix)
        .unwrap();
    let wk = e.htod(&values(nh * r * dn, 29)).unwrap();
    let wv = e.htod(&values(nh * dv * r, 31)).unwrap();
    let q = e.htod(&values(t * nh * dn, 37)).unwrap();
    let ids: Vec<i32> = (0..t)
        .flat_map(|i| (0..slots).map(move |s| if s <= prefix + i { s as i32 } else { -1 }))
        .collect();
    let idx = e.htod_i32(&ids).unwrap();
    let before = crate::MLA_TC_PREFILL_DISPATCHES.load(std::sync::atomic::Ordering::Relaxed);
    let actual = HybridModel::mla_tc_prefill_chain(
        &e,
        &wk,
        &wv,
        &q,
        &e.uninit(0).unwrap(),
        Some(&mut plane),
        &idx,
        slots,
        t,
        rows,
        nh,
        dn,
        dv,
        r,
        0.0625,
    )
    .unwrap()
    .expect("TC compressed path must engage");
    plane.check(&e).unwrap();
    // Independent basis plumbing around the unchanged GEMM/attention primitives.
    // Absorption already rounds to BF16; rotate that widened buffer on CPU,
    // then round once more to BF16. Do NOT use chain(None, rotated_history):
    // it would leave the query and output in the wrong basis.
    let wk_bf = e.f32_to_bf16(&wk, nh * r * dn).unwrap();
    let wv_bf = e.f32_to_bf16(&wv, nh * dv * r).unwrap();
    let q_bf = e.f32_to_bf16(&q, t * nh * dn).unwrap();
    let mut absorbed = e.alloc_u8_uninit(t * nh * r * 2).unwrap();
    assert!(
        e.mla_bf16_gemm_sb_bf16out(
            &wk_bf,
            &q_bf,
            &mut absorbed,
            t,
            r,
            dn,
            nh * dn,
            dn,
            nh * r,
            r,
            nh
        )
        .unwrap()
    );
    let absorbed_host: Vec<f32> = e
        .stream()
        .clone_dtoh(&absorbed)
        .unwrap()
        .chunks_exact(2)
        .map(|b| f32::from_bits((u16::from_le_bytes([b[0], b[1]]) as u32) << 16))
        .collect();
    let rotated_query: Vec<f32> = absorbed_host
        .chunks_exact(r)
        .flat_map(|row| rht512_forward(row).unwrap())
        .collect();
    let query_bf = e
        .f32_to_bf16(&e.htod(&rotated_query).unwrap(), t * nh * r)
        .unwrap();
    let cache_bf = e.f32_to_bf16(&e.htod(&decoded).unwrap(), rows * r).unwrap();
    let mut physical_output = e.uninit(t * nh * r).unwrap();
    e.mla_attn_gathered_tc(
        &query_bf,
        &cache_bf,
        &idx,
        &mut physical_output,
        nh,
        r,
        t,
        slots,
        0.0625,
    )
    .unwrap();
    let physical_host = e.dtoh(&physical_output).unwrap();
    let inverse: Vec<f32> = physical_host
        .chunks_exact(r)
        .flat_map(|row| rht512_inverse(row).unwrap())
        .collect();
    let inverse_bf = e
        .f32_to_bf16(&e.htod(&inverse).unwrap(), t * nh * r)
        .unwrap();
    let mut expected = e.uninit(t * nh * dv).unwrap();
    assert!(
        e.mla_bf16_gemm_sb_f32out(
            &wv_bf,
            &inverse_bf,
            &mut expected,
            t,
            dv,
            r,
            nh * r,
            r,
            nh * dv,
            dv,
            nh
        )
        .unwrap()
    );
    assert!(
        crate::MLA_TC_PREFILL_DISPATCHES.load(std::sync::atomic::Ordering::Relaxed) >= before + 1
    );
    let a = e.dtoh(&actual).unwrap();
    let b = e.dtoh(&expected).unwrap();
    assert!(a.iter().all(|x| x.is_finite()) && a.iter().any(|x| *x != 0.));
    assert_eq!(
        a.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
        b.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
    );
    // Omitting the inverse before BF16/Wv must be observable at the final output.
    let wrong_bf = e.f32_to_bf16(&physical_output, t * nh * r).unwrap();
    let mut wrong = e.uninit(t * nh * dv).unwrap();
    assert!(
        e.mla_bf16_gemm_sb_f32out(
            &wv_bf,
            &wrong_bf,
            &mut wrong,
            t,
            dv,
            r,
            nh * r,
            r,
            nh * dv,
            dv,
            nh
        )
        .unwrap()
    );
    assert!(
        b.iter()
            .zip(e.dtoh(&wrong).unwrap())
            .any(|(x, y)| (*x - y).abs() > 1e-4)
    );
}
