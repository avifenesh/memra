use super::*;
use std::path::Path;

fn tiny_plan() -> WhisperPlan {
    let mut plan = memra_gguf::model_packs::whisper::PACK
        .compile_plan(
            include_str!("../fixtures/whisper-source-config.json"),
            include_str!("../fixtures/whisper-source-frontend.json"),
        )
        .unwrap()
        .speech
        .unwrap();
    plan.hidden_size = 16;
    plan.encoder_heads = 2;
    plan.decoder_heads = 2;
    plan.encoder_layers = 2;
    plan.decoder_layers = 2;
    plan.encoder_ffn = 32;
    plan.decoder_ffn = 32;
    plan.source_positions = 8;
    plan.target_positions = 16;
    plan.vocab_size = 64;
    plan.frontend.max_frames = 16;
    plan.frontend.max_samples = 2560;
    plan.conv_stem[0].output_channels = 16;
    plan.conv_stem[1].input_channels = 16;
    plan.conv_stem[1].output_channels = 16;
    plan.attention.qk_scale = 1.0 / 8.0f32.sqrt();
    plan
}

fn floats(path: &Path) -> Vec<f32> {
    std::fs::read(path)
        .unwrap()
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}

#[test]
fn every_encoder_stage_matches_independent_hf_f32_and_f16_fixtures() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/speech/fixtures");
    let source = StModel::open(&root.join("tiny-encoder.safetensors")).unwrap();
    let plan = tiny_plan();
    let input = ReferenceTensor::new(vec![128, 16], floats(&root.join("tiny-mel.f32"))).unwrap();
    for (numeric, label, tolerance) in [
        (WhisperNumeric::F32, "f32", 1e-5),
        (WhisperNumeric::F16, "f16", 1e-2),
    ] {
        let encoder = WhisperEncoder::load(&plan, &source, numeric).unwrap();
        let mut visited = Vec::new();
        let output = encoder
            .encode_with_trace(&input, |name, got| {
                let expected = floats(&root.join(format!("tiny-{label}-{name}.f32")));
                assert_eq!(got.data.len(), expected.len());
                let max_abs = got
                    .data
                    .iter()
                    .zip(expected)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                assert!(
                    max_abs <= tolerance,
                    "{label} {name} max_abs={max_abs} > {tolerance}"
                );
                visited.push(name.to_string());
                Ok(())
            })
            .unwrap();
        assert_eq!(
            visited,
            ["conv1", "conv2", "layer-00", "layer-01", "encoder"]
        );
        assert_eq!(output.shape, [8, 16]);
        assert!(output.data.iter().any(|x| x.abs() > 0.5));
        assert!(
            encoder
                .encode(&ReferenceTensor {
                    shape: vec![128, 15],
                    data: vec![0.0; 128 * 15],
                    ints: None
                })
                .is_err()
        );
        let mut invalid = input.clone();
        invalid.data[0] = f32::INFINITY;
        assert!(encoder.encode(&invalid).is_err());
    }
}

#[test]
fn half_rounding_is_nearest_even_and_erf_gelu_is_not_tanh_gelu() {
    let half = WhisperNumeric::F16;
    assert_eq!(half.round(1.0 + 2f32.powi(-11)), 1.0);
    assert_eq!(half.round(1.0 + 3.0 * 2f32.powi(-11)), 1.0 + 2f32.powi(-9));
    assert_eq!(half.round(2f32.powi(-24)), 2f32.powi(-24));
    assert_eq!(half.round(-0.0).to_bits(), (-0.0f32).to_bits());
    for (x, expected) in [
        (-3.0, -0.004049694),
        (-1.0, -0.15865526),
        (0.0, 0.0),
        (1.0, 0.8413447),
        (3.0, 2.9959502),
    ] {
        assert!(
            (gelu_erf(x) - expected).abs() < 3e-7,
            "GELU({x})={}",
            gelu_erf(x)
        );
    }
}

#[test]
fn changed_encoder_programs_fail_before_loading_weights() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/speech/fixtures");
    let source = StModel::open(&root.join("tiny-encoder.safetensors")).unwrap();
    let mut plan = tiny_plan();
    plan.attention.key_bias = true;
    assert!(WhisperEncoder::load(&plan, &source, WhisperNumeric::F16).is_err());
    let mut plan = tiny_plan();
    plan.conv_stem[1].stride = 1;
    assert!(WhisperEncoder::load(&plan, &source, WhisperNumeric::F16).is_err());
    let mut plan = tiny_plan();
    plan.encoder_layers = 1;
    assert!(
        WhisperEncoder::load(&plan, &source, WhisperNumeric::F16).is_err(),
        "unclaimed encoder-layer tensors must fail census"
    );
}

#[test]
fn gelu_matches_the_pinned_hf_vector_numeric_program() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/speech/fixtures");
    let input = floats(&root.join("gelu-hf-vector-input.f32"));
    let expected = floats(&root.join("gelu-hf-vector-expected.f32"));
    assert_eq!(input.len(), 4096);
    assert_eq!(input.len(), expected.len());
    let errors: Vec<_> = input
        .into_iter()
        .zip(expected)
        .map(|(x, y)| (gelu_erf(x) - y).abs())
        .collect();
    let max = errors.iter().copied().fold(0.0f32, f32::max);
    let mean = errors.iter().map(|&x| f64::from(x)).sum::<f64>() / errors.len() as f64;
    assert!(
        max <= 1e-6 && mean <= 1e-8,
        "GELU numeric program: max={max} mean={mean}"
    );
}
