use super::*;
use crate::safetensors::StShard;

fn f32s(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Mints every case of a golden file (lane F's quantizer output) and counts differing bytes.
fn check_golden(path: &Path) -> Vec<String> {
    let st = StShard::open(path).expect("golden file");
    let mut cases: Vec<String> = st
        .names()
        .filter_map(|n| n.strip_suffix(".codes").map(str::to_string))
        .collect();
    cases.sort();
    assert!(!cases.is_empty(), "no cases in {}", path.display());
    let mut lines = Vec::new();
    for case in cases {
        let get = |p: &str| {
            st.raw(&format!("{case}.{p}"))
                .map(|(i, b)| (i.shape.clone(), b))
        };
        let per_expert = get("per_expert").expect("per_expert").1[0] == 1;
        let (shape, wbytes) = match get("w") {
            Some((s, b)) => (s.iter().map(|v| *v as usize).collect::<Vec<_>>(), f32s(b)),
            None => {
                let (s, b) = get("w_bf16").expect("w or w_bf16");
                (s.iter().map(|v| *v as usize).collect(), bf16_values(b))
            }
        };
        let k = *shape.last().unwrap();
        let groups = if per_expert { shape[0] } else { 1 };
        let per = wbytes.len() / groups;
        let want_codes = get("codes").unwrap().1;
        let want_scales = get("scales").unwrap().1;
        let want_t = f32s(get("t").unwrap().1);
        let mut codes = vec![0u8; wbytes.len() / 2];
        let mut scales = vec![0u8; wbytes.len() / 16];
        let mut ts = Vec::new();
        for g in 0..groups {
            let w = &wbytes[g * per..(g + 1) * per];
            let t = tensor_scale(amax(w));
            ts.push(t);
            quantize_rows(
                w,
                k,
                t,
                &mut codes[g * per / 2..(g + 1) * per / 2],
                &mut scales[g * per / 16..(g + 1) * per / 16],
                4,
            );
        }
        let dc = codes.iter().zip(want_codes).filter(|(a, b)| a != b).count();
        let ds = scales
            .iter()
            .zip(want_scales)
            .filter(|(a, b)| a != b)
            .count();
        let dt = ts
            .iter()
            .zip(&want_t)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        let mut dd = 0;
        if let Some((_, deq)) = get("deq") {
            let want = f32s(deq);
            for g in 0..groups {
                let got = dequant_rows(
                    &codes[g * per / 2..(g + 1) * per / 2],
                    &scales[g * per / 16..(g + 1) * per / 16],
                    k,
                    ts[g],
                );
                dd += got
                    .iter()
                    .zip(&want[g * per..(g + 1) * per])
                    .filter(|(a, b)| a.to_bits() != b.to_bits())
                    .count();
            }
        }
        lines.push(format!(
            "{case}: {} elements, code bytes differ {dc}, scale bytes differ {ds}, tensor scales differ {dt}, dequantized values differ {dd}",
            wbytes.len()
        ));
        assert_eq!((dc, ds, dt, dd), (0, 0, 0, 0), "{}", lines.last().unwrap());
    }
    lines
}

#[test]
fn quantizer_matches_lane_f_goldens() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("src/model_packs/inkling2/fixtures/nvfp4-rig-golden.bin");
    for line in check_golden(&path) {
        eprintln!("{line}");
    }
}

/// A larger golden file outside the repository (`INK2_RIG_GOLDEN=path`).
#[test]
#[ignore]
fn quantizer_matches_external_golden() {
    let path = std::env::var("INK2_RIG_GOLDEN").expect("INK2_RIG_GOLDEN");
    for line in check_golden(Path::new(&path)) {
        eprintln!("{line}");
    }
}

#[test]
fn e4m3_index_is_round_to_nearest_even() {
    for b in 0..=126u8 {
        let v = e4m3_scale_value(b);
        assert_eq!(e4m3_rne_index(v), b);
    }
    // midpoint of bytes 8 and 9 goes to the even byte, of 9 and 10 to 10
    let mid = |a: u8| ((e4m3_magnitude(a) + e4m3_magnitude(a + 1)) / 2.0) as f32;
    assert_eq!(e4m3_rne_index(mid(8)), 8);
    assert_eq!(e4m3_rne_index(mid(9)), 10);
    assert_eq!(e4m3_rne_index(448.0), 126);
    assert_eq!(e4m3_rne_index(0.0), 0);
}

#[test]
fn nibbles_round_trip_and_keep_negative_zero() {
    for n in 0..16u8 {
        assert_eq!(e2m1_nibble(e2m1_value(n)), n);
    }
    assert_eq!(e2m1_nibble(e2m1_rne(-0.1)), 8);
    assert_eq!(e2m1_rne(0.75), 1.0);
    assert_eq!(e2m1_rne(0.25), 0.0);
    assert_eq!(e2m1_rne(2.5), 2.0);
    assert_eq!(e2m1_rne(5.0), 4.0);
    assert_eq!(e2m1_rne(9.0), 6.0);
}

#[test]
fn quantize_rows_is_thread_count_invariant() {
    let mut s = 1u64;
    let w: Vec<f32> = (0..64 * 256)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 3.0
        })
        .collect();
    let t = tensor_scale(amax(&w));
    let run = |threads| {
        let mut c = vec![0u8; w.len() / 2];
        let mut sc = vec![0u8; w.len() / 16];
        quantize_rows(&w, 256, t, &mut c, &mut sc, threads);
        (c, sc)
    };
    assert_eq!(run(1), run(7));
}

#[test]
fn class_tensor_set_of_the_shipped_config() {
    let mut config = crate::config::ModelConfig::from_hf(&crate::config::HfConfig::parse(
        include_str!("../fixtures/config.json"),
    ));
    config
        .attach_quant_groups(include_str!("../fixtures/hf_quant_config.json"))
        .unwrap();
    let ink = config.inkling2.as_ref().unwrap();
    let set = rig_tensors(ink).unwrap();
    // 24 layers x 6 projections, 4 layers x 2 banks, the unembed
    assert_eq!(set.len(), 24 * 6 + 4 * 2 + 1);
    // the measurement subset leaves the loop core's 12 x 6 attention projections as stored
    let outside = class_tensors(ink, false).unwrap();
    assert_eq!(outside.len(), 12 * 6 + 4 * 2 + 1);
    assert!(
        !outside
            .iter()
            .any(|t| t.name().starts_with("model.llm.layers.6.attn"))
    );
    assert_eq!(class_tensors(ink, true).unwrap(), set);
    let experts: Vec<_> = set
        .iter()
        .filter(|t| matches!(t, RigTensor::Experts { .. }))
        .map(|t| t.name().to_string())
        .collect();
    assert_eq!(experts[0], "model.llm.layers.20.mlp.experts.w13_weight");
    assert_eq!(experts.len(), 8);
    assert!(
        set.iter()
            .any(|t| t.name() == "model.llm.layers.5.attn.wr_du.weight")
    );
    assert!(
        set.iter()
            .any(|t| t.name() == "model.llm.layers.18.attn.wq_du.weight")
    );
    assert!(
        set.iter()
            .any(|t| t.name() == "model.llm.layers.6.attn.wq_du.weight")
    );
    assert!(
        set.iter()
            .any(|t| t.name() == "model.llm.layers.17.attn.wr_du.weight")
    );
}

/// A tiny checkpoint directory (the reference crate'\''s tiny image and the pack'\''s tiny configs) and
/// its minted overlay, in a fresh scratch directory.
fn tiny_dirs(tag: &str) -> (std::path::PathBuf, std::path::PathBuf) {
    let root = std::env::temp_dir().join(format!("ink2-rig-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    let ckpt = root.join("tiny");
    std::fs::create_dir_all(&ckpt).unwrap();
    std::fs::write(
        ckpt.join("model.safetensors"),
        include_bytes!("../../../../../memra-reference/src/fixtures/inkling2-tiny-checkpoint.bin"),
    )
    .unwrap();
    std::fs::write(ckpt.join("config.json"), super::super::TINY_CONFIG).unwrap();
    std::fs::write(
        ckpt.join("hf_quant_config.json"),
        super::super::TINY_QUANT_CONFIG,
    )
    .unwrap();
    let out = default_overlay_dir(&ckpt);
    mint_overlay_dir(&ckpt, &out, "tiny@test", 2, &mut |_| {}).unwrap();
    (ckpt, out)
}

fn bind_tiny(ckpt: &Path, out: &Path) -> Result<BoundOverlay, String> {
    let config = super::super::tiny_model_config().unwrap();
    let base = crate::safetensors::StModel::open(ckpt).unwrap();
    let sha = sha256_hex(super::super::TINY_QUANT_CONFIG.as_bytes());
    bind_overlay(
        ckpt,
        out,
        &base,
        config.inkling2.as_ref().unwrap(),
        &sha,
        true,
        2,
    )
}

fn rewrite(path: &Path, f: impl FnOnce(&mut Vec<u8>)) {
    let mut perm = std::fs::metadata(path).unwrap().permissions();
    #[allow(clippy::permissions_set_readonly_false)]
    // allow: the test tampers with its own read-only file
    perm.set_readonly(false);
    std::fs::set_permissions(path, perm).unwrap();
    let mut bytes = std::fs::read(path).unwrap();
    f(&mut bytes);
    std::fs::write(path, bytes).unwrap();
}

#[test]
fn overlay_binds_its_checkpoint_and_refuses_every_mismatch() {
    let (ckpt, out) = tiny_dirs("bind");
    let bound = bind_tiny(&ckpt, &out).unwrap();
    assert_eq!(bound.manifest.tensors.len(), 8 * 6 + 2 * 2 + 1);
    // the outside-loop measurement class binds the same overlay and maps the prelude and coda
    let config = super::super::tiny_model_config().unwrap();
    let base = crate::safetensors::StModel::open(&ckpt).unwrap();
    let sha = sha256_hex(super::super::TINY_QUANT_CONFIG.as_bytes());
    let outside = bind_overlay(
        &ckpt,
        &out,
        &base,
        config.inkling2.as_ref().unwrap(),
        &sha,
        false,
        2,
    )
    .unwrap();
    assert!(
        !outside
            .planes
            .contains_key("model.llm.layers.2.attn.wq_du.weight")
    );
    assert!(
        outside
            .planes
            .contains_key("model.llm.layers.1.attn.wq_du.weight")
    );
    assert!(
        bound
            .planes
            .contains_key("model.llm.layers.2.attn.wq_du.weight")
    );
    drop((outside, base));
    // read-only, and a second mint into the same directory refuses
    assert!(
        std::fs::metadata(out.join(OVERLAY_FILE))
            .unwrap()
            .permissions()
            .readonly()
    );
    assert!(mint_overlay_dir(&ckpt, &out, "tiny@test", 2, &mut |_| {}).is_err());
    // a re-mint elsewhere is byte-identical (deterministic, any thread count)
    let again = out.with_file_name("again");
    let m2 = mint_overlay_dir(&ckpt, &again, "tiny@test", 7, &mut |_| {}).unwrap();
    assert_eq!(m2.overlay_sha256, bound.manifest.overlay_sha256);
    drop(bound);

    // one flipped overlay byte (the last byte of the data: a code byte)
    rewrite(&out.join(OVERLAY_FILE), |b| *b.last_mut().unwrap() ^= 0x11);
    let e = bind_tiny(&ckpt, &out)
        .err()
        .expect("a flipped overlay byte binds");
    assert!(e.contains("sha256"), "{e}");
    rewrite(&out.join(OVERLAY_FILE), |b| *b.last_mut().unwrap() ^= 0x11);
    assert!(bind_tiny(&ckpt, &out).is_ok());

    // one flipped byte of a source tensor the overlay replaces
    let base_path = ckpt.join("model.safetensors");
    let st = crate::safetensors::StShard::open(&base_path).unwrap();
    let (info, _) = st.raw("model.llm.unembed.weight").unwrap();
    let header =
        u64::from_le_bytes(std::fs::read(&base_path).unwrap()[..8].try_into().unwrap()) as usize;
    let at = 8 + header + info.data_offsets[0] + 5;
    drop(st);
    rewrite(&base_path, |b| b[at] ^= 0x01);
    let e = bind_tiny(&ckpt, &out)
        .err()
        .expect("a changed source tensor binds");
    assert!(e.contains("base model.llm.unembed.weight"), "{e}");
    rewrite(&base_path, |b| b[at] ^= 0x01);

    // another quantization config, and a manifest missing a tensor
    let config = super::super::tiny_model_config().unwrap();
    let base = crate::safetensors::StModel::open(&ckpt).unwrap();
    assert!(
        bind_overlay(
            &ckpt,
            &out,
            &base,
            config.inkling2.as_ref().unwrap(),
            "00",
            true,
            2
        )
        .is_err()
    );
    rewrite(&out.join(MANIFEST_FILE), |b| {
        let mut m = Manifest::parse(std::str::from_utf8(b).unwrap()).unwrap();
        m.tensors.pop();
        *b = m.to_json().into_bytes();
    });
    let e = bind_tiny(&ckpt, &out)
        .err()
        .expect("a short manifest binds");
    assert!(e.contains("tensor list"), "{e}");
    let _ = std::fs::remove_dir_all(ckpt.parent().unwrap());
}
