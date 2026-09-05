use crate::Engine;
use crate::latent_nvfp4_ffi::{Nvfp4LatentOps, Nvfp4LatentStorage};
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
        .flat_map(|row| PackedRow::encode(row).unwrap().decode().unwrap())
        .collect();
    let mut plane = Nvfp4LatentStorage::new(&e, r, rows).unwrap();
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
    let expected = HybridModel::mla_tc_prefill_chain(
        &e,
        &wk,
        &wv,
        &q,
        &e.htod(&decoded).unwrap(),
        None,
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
    .expect("TC reference path must engage");
    assert!(
        crate::MLA_TC_PREFILL_DISPATCHES.load(std::sync::atomic::Ordering::Relaxed) >= before + 2
    );
    let a = e.dtoh(&actual).unwrap();
    let b = e.dtoh(&expected).unwrap();
    assert!(a.iter().all(|x| x.is_finite()) && a.iter().any(|x| *x != 0.));
    assert_eq!(
        a.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
        b.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
    );
}
