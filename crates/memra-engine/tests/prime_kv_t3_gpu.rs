//! Byte gate for wider query sharing. Run alone with MEMRA_PRIME_KV_T3=1 under the GPU lock.
use memra_engine::Engine;

fn values(n: usize, seed: u64) -> Vec<f32> {
    let mut x = seed;
    (0..n)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            ((x >> 40) as f32 / 16777216.0 - 0.5) * 0.4
        })
        .collect()
}

#[test]
#[ignore = "requires isolated CUDA device and MEMRA_PRIME_KV_T3=1"]
fn three_plane_staging_preserves_quantized_attention_bytes() {
    assert_eq!(std::env::var("MEMRA_PRIME_KV_T3").as_deref(), Ok("1"));
    let e = Engine::new(0).unwrap();
    let hd = 256;
    for (t, tkv, nh, nkv) in [
        (128, 128, 6, 1),
        (129, 161, 6, 1),
        (192, 4097, 6, 1),
        (992, 32768, 6, 1),
        (1024, 131070, 6, 1),
        (4096, 8192, 24, 4),
    ] {
        let dim = hd * nkv;
        let kb = dim / 32 * 34;
        let vb = dim / 32 * 24;
        let q = e.htod(&values(t * nh * hd, 19)).unwrap();
        let k = values(tkv * dim, 29);
        let v = values(tkv * dim, 37);
        let mut kc = e.alloc_u8(tkv * kb).unwrap();
        let mut vc = e.alloc_u8(tkv * vb).unwrap();
        // The append kernel has a CUDA grid-y limit; preserve rows across bounded uploads.
        for at in (0..tkv).step_by(32768) {
            let end = (at + 32768).min(tkv);
            let kd = e.htod(&k[at * dim..end * dim]).unwrap();
            let vd = e.htod(&v[at * dim..end * dim]).unwrap();
            e.append_kv_quantized_rows(
                &kd,
                &vd,
                &mut kc,
                &mut vc,
                at,
                end - at,
                dim,
                dim,
                kb,
                vb,
                false,
            )
            .unwrap();
        }
        let kv = e.view_u8(&kc, tkv * kb);
        let vv = e.view_u8(&vc, tkv * vb);
        for causal in [true, false] {
            let mut reference = e.zeros(t * nh * hd).unwrap();
            let mut candidate = e.zeros(t * nh * hd).unwrap();
            e.fa_prefill_view(
                &q,
                &kv,
                &vv,
                &mut reference,
                hd,
                nh,
                nkv,
                t,
                tkv,
                0.0625,
                causal,
                kb,
                vb,
                false,
            )
            .unwrap();
            e.fa_prefill_view_ws(
                &q,
                &kv,
                &vv,
                &mut candidate,
                hd,
                nh,
                nkv,
                t,
                tkv,
                0.0625,
                causal,
                kb,
                vb,
                false,
            )
            .unwrap();
            let a = e.dtoh(&reference).unwrap();
            let b = e.dtoh(&candidate).unwrap();
            assert!(a.iter().all(|x| x.is_finite()));
            let differences = a
                .iter()
                .zip(&b)
                .filter(|(x, y)| x.to_bits() != y.to_bits())
                .count();
            println!(
                "kv-t3 t={t} tkv={tkv} nh={nh} nkv={nkv} causal={causal} differences={differences}"
            );
            assert_eq!(differences, 0);
        }
    }
}
