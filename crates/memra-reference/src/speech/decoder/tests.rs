use super::*;
use crate::speech::encoder::tests::tiny_plan;
use std::path::Path;

fn floats(path: &Path) -> Vec<f32> {
    std::fs::read(path)
        .unwrap()
        .chunks_exact(4)
        .map(|x| f32::from_le_bytes(x.try_into().unwrap()))
        .collect()
}

fn close(actual: &[f32], expected: &[f32], tolerance: f32, stage: &str) {
    assert_eq!(actual.len(), expected.len(), "{stage}");
    let delta = actual
        .iter()
        .zip(expected)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    assert!(
        delta <= tolerance,
        "{stage}: max_abs={delta}, tolerance={tolerance}"
    );
}

#[test]
fn cached_steps_hidden_logits_and_kv_match_independent_hf() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/speech/fixtures");
    let source = StModel::open(&root.join("tiny-decoder.safetensors")).unwrap();
    let plan = tiny_plan();
    let encoded =
        ReferenceTensor::new(vec![8, 16], floats(&root.join("tiny-f32-encoder.f32"))).unwrap();
    for (numeric, label, tolerance) in [
        (WhisperNumeric::F32, "f32", 1e-5),
        (WhisperNumeric::F16, "f16", 2e-3),
    ] {
        let decoder = WhisperDecoder::load(&plan, &source, numeric).unwrap();
        let mut session = decoder.start(&encoded).unwrap();
        let cross0: Vec<_> = (0..2)
            .map(|i| {
                let (k, v) = session.cross_kv(i).unwrap();
                (k.to_vec(), v.to_vec())
            })
            .collect();
        for (step, token) in [1, 7, 12, 3, 2].into_iter().enumerate() {
            let output = session.step(token).unwrap();
            let prefix = format!("tiny-decoder-{label}-{step}");
            close(
                &output.hidden,
                &floats(&root.join(format!("{prefix}-hidden.f32"))),
                tolerance,
                &prefix,
            );
            close(
                &output.hidden,
                &floats(&root.join(format!("{prefix}-full-hidden.f32"))),
                tolerance,
                "cached versus full prefix",
            );
            close(
                &output.logits,
                &floats(&root.join(format!("{prefix}-logits.f32"))),
                tolerance,
                "tied output logits",
            );
            assert_eq!(output.position, step);
            assert_eq!(session.position(), step + 1);
            for (layer, (old_key, old_value)) in cross0.iter().enumerate() {
                let (key, value) = session.self_kv(layer).unwrap();
                close(
                    key,
                    &floats(&root.join(format!("{prefix}-{layer}-self-keys.f32"))),
                    tolerance,
                    "self key",
                );
                close(
                    value,
                    &floats(&root.join(format!("{prefix}-{layer}-self-values.f32"))),
                    tolerance,
                    "self value",
                );
                let (key, value) = session.cross_kv(layer).unwrap();
                assert_eq!(key, old_key);
                assert_eq!(value, old_value);
                close(
                    key,
                    &floats(&root.join(format!("tiny-decoder-{label}-0-{layer}-cross-keys.f32"))),
                    tolerance,
                    "cross key",
                );
                close(
                    value,
                    &floats(&root.join(format!("tiny-decoder-{label}-0-{layer}-cross-values.f32"))),
                    tolerance,
                    "cross value",
                );
            }
        }
    }
}

#[test]
fn sessions_are_independent_and_invalid_steps_do_not_advance_caches() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/speech/fixtures");
    let source = StModel::open(&root.join("tiny-decoder.safetensors")).unwrap();
    let plan = tiny_plan();
    let encoded =
        ReferenceTensor::new(vec![8, 16], floats(&root.join("tiny-f32-encoder.f32"))).unwrap();
    let decoder = WhisperDecoder::load(&plan, &source, WhisperNumeric::F32).unwrap();
    let mut a = decoder.start(&encoded).unwrap();
    let mut b = decoder.start(&encoded).unwrap();
    let a0 = a.step(1).unwrap();
    assert_eq!(b.position(), 0);
    assert!(b.self_kv(0).unwrap().0.is_empty());
    assert_eq!(b.step(1).unwrap().logits, a0.logits);
    let saved = a.self_kv(0).unwrap().0.to_vec();
    assert!(a.step(plan.vocab_size).is_err());
    assert_eq!(a.position(), 1);
    assert_eq!(a.self_kv(0).unwrap().0, saved);
    assert_eq!(a.step(7).unwrap().logits, b.step(7).unwrap().logits);
    while a.position() < plan.target_positions as usize {
        a.step(3).unwrap();
    }
    let saved = a.self_kv(0).unwrap().0.to_vec();
    assert!(a.step(3).is_err());
    assert_eq!(a.position(), plan.target_positions as usize);
    assert_eq!(a.self_kv(0).unwrap().0, saved);
    let mut invalid = encoded.clone();
    invalid.data[0] = f32::NAN;
    assert!(decoder.start(&invalid).is_err());
}
