//! Compare Memra's combined audio encoder with independent publisher inputs.
//! This is a diagnostic; a printed code mismatch remains a qualification gap.

use std::path::Path;

use memra_gguf::model_packs::mimo_v2::audio_tokenizer::{LFS_WEIGHT_SHA256, SOURCE};
use sha2::{Digest, Sha256};

use super::{Engine, Fail, MEL_CHANNELS, MiMoAudioCodecEncoderWeights};

const TOKENS: usize = 3;
const WIDTH: usize = 1_024;
const DEPTHS: usize = 20;
const CPU_IDS: [u16; TOKENS * DEPTHS] = [
    892, 542, 112, 40, 86, 47, 38, 118, 18, 53, 99, 124, 43, 24, 87, 75, 69, 26, 58, 91, 54, 61,
    119, 28, 59, 86, 90, 38, 102, 35, 72, 17, 29, 37, 55, 24, 123, 127, 17, 18, 992, 285, 49, 126,
    107, 96, 114, 107, 88, 83, 89, 5, 48, 24, 37, 56, 16, 64, 29, 44,
];
const PUBLISHER_GPU_IDS: [u16; TOKENS * DEPTHS] = [
    892, 542, 112, 0, 86, 47, 38, 118, 18, 91, 99, 124, 43, 24, 24, 5, 69, 26, 58, 91, 54, 61, 119,
    28, 59, 86, 90, 38, 102, 35, 72, 17, 29, 37, 55, 24, 123, 127, 17, 18, 992, 285, 49, 126, 107,
    96, 114, 107, 88, 83, 89, 5, 48, 24, 37, 56, 16, 64, 29, 72,
];
const PUBLISHER_GPU_LINEAR_CONTROL_IDS: [u16; TOKENS * DEPTHS] = [
    892, 542, 112, 40, 86, 47, 38, 118, 18, 53, 99, 124, 43, 24, 87, 2, 64, 26, 58, 91, 54, 61,
    119, 28, 59, 86, 90, 38, 102, 35, 72, 17, 29, 37, 55, 24, 123, 32, 84, 18, 992, 285, 49, 126,
    107, 96, 114, 107, 88, 83, 89, 5, 48, 24, 37, 56, 16, 64, 29, 72,
];
const CPU_FEATURES: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../memra-reference/src/fixtures/mimo-v26-publisher-cpu-pre-rvq.bf16"
));
const PUBLISHER_GPU_FEATURES: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../memra-reference/src/fixtures/mimo-v26-publisher-pro6000-pre-rvq.bf16"
));
const PUBLISHER_GPU_FRONTEND: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../memra-reference/src/fixtures/mimo-v26-publisher-pro6000-audio-frontend.bf16"
));
const PUBLISHER_GPU_CONV1: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../memra-reference/src/fixtures/mimo-v26-publisher-pro6000-audio-conv1.bf16"
));
const PUBLISHER_GPU_CONV1_PRE_GELU: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../memra-reference/src/fixtures/mimo-v26-publisher-pro6000-audio-conv1-pre-gelu.bf16"
));
const PUBLISHER_GPU_CONV1_LINEAR: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../memra-reference/src/fixtures/mimo-v26-publisher-pro6000-audio-conv1-linear.bf16"
));
const PUBLISHER_GPU_CONV1_LINEAR_CONTROL: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../memra-reference/src/fixtures/mimo-v26-publisher-pro6000-audio-conv1-linear-control.bf16"
));
const PUBLISHER_GPU_CONV2_PRE_GELU: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../memra-reference/src/fixtures/mimo-v26-publisher-pro6000-audio-conv2-pre-gelu.bf16"
));
const PUBLISHER_GPU_CONV2_PRE_GELU_LINEAR_CONTROL: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../memra-reference/src/fixtures/mimo-v26-publisher-pro6000-audio-conv2-pre-gelu-linear-control.bf16"
));
const PUBLISHER_GPU_LAYER0_CUDNN: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../memra-reference/src/fixtures/mimo-v26-publisher-pro6000-audio-layer0-cudnn.bf16"
));
const PUBLISHER_GPU_LAYER0_LINEAR_CONTROL: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../memra-reference/src/fixtures/mimo-v26-publisher-pro6000-audio-layer0-linear-control.bf16"
));
const PUBLISHER_GPU_LAYER0_ATTN_NORM_LINEAR_CONTROL: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../memra-reference/src/fixtures/mimo-v26-publisher-pro6000-audio-layer0-attn-norm-linear-control.bf16"
));
const PUBLISHER_GPU_LAYER0_ATTENTION_LINEAR_CONTROL: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../memra-reference/src/fixtures/mimo-v26-publisher-pro6000-audio-layer0-attention-linear-control.bf16"
));
const PUBLISHER_GPU_LAYER0_MLP_NORM_LINEAR_CONTROL: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../memra-reference/src/fixtures/mimo-v26-publisher-pro6000-audio-layer0-mlp-norm-linear-control.bf16"
));
const PUBLISHER_GPU_LAYER0_FC2_LINEAR_CONTROL: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../memra-reference/src/fixtures/mimo-v26-publisher-pro6000-audio-layer0-fc2-linear-control.bf16"
));
const PUBLISHER_GPU_LAYER0_FC2_INPUT_LINEAR_CONTROL: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../memra-reference/src/fixtures/mimo-v26-publisher-pro6000-audio-layer0-fc2-input-linear-control.bf16"
));
const PUBLISHER_GPU_FRONTEND_LINEAR_CONTROL: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../memra-reference/src/fixtures/mimo-v26-publisher-pro6000-audio-frontend-linear-control.bf16"
));
const PUBLISHER_GPU_STACK_LINEAR_CONTROL: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../memra-reference/src/fixtures/mimo-v26-publisher-pro6000-audio-stack-linear-control.bf16"
));
const PUBLISHER_GPU_PRE_RVQ_LINEAR_CONTROL: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../memra-reference/src/fixtures/mimo-v26-publisher-pro6000-audio-pre-rvq-linear-control.bf16"
));
const PUBLISHER_GPU_STACK: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../memra-reference/src/fixtures/mimo-v26-publisher-pro6000-audio-stack.bf16"
));
const MEL: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../memra-reference/src/fixtures/mimo-pcm-mel-voiced-2048.logmel.f32"
));

fn decode_bf16(bytes: &[u8], rows: usize) -> Result<Vec<f32>, Fail> {
    decode_bf16_width(bytes, rows, WIDTH)
}

fn decode_bf16_width(bytes: &[u8], rows: usize, width: usize) -> Result<Vec<f32>, Fail> {
    if width == 0 || bytes.len() != rows * width * 2 {
        return Err("MiMo BF16 feature fixture extent changed".into());
    }
    Ok(bytes
        .chunks_exact(2)
        .map(|pair| {
            let bits = u16::from_le_bytes([pair[0], pair[1]]);
            f32::from_bits(u32::from(bits) << 16)
        })
        .collect())
}

fn feature_stats(label: &str, actual: &[f32], reference: &[f32], rows: usize) -> Result<(), Fail> {
    if actual.len() != rows * WIDTH || reference.len() != actual.len() {
        return Err("MiMo pre-RVQ feature comparison extent changed".into());
    }
    let mut numerator = 0.0f64;
    let mut denominator = 0.0f64;
    let mut max_abs = 0.0f32;
    let mut matching_bits = 0usize;
    for (&got, &want) in actual.iter().zip(reference) {
        if !got.is_finite() || !want.is_finite() {
            return Err("MiMo pre-RVQ feature comparison was non-finite".into());
        }
        let difference = (got - want).abs();
        numerator += f64::from(difference).powi(2);
        denominator += f64::from(want).powi(2);
        max_abs = max_abs.max(difference);
        matching_bits += usize::from(got.to_bits() == want.to_bits());
    }
    println!(
        "mimo_audio_features\t{label}\trel_l2={:.9e}\tmax_abs={max_abs:.9e}\tmatching_bits={matching_bits}/{}",
        (numerator / denominator).sqrt(),
        actual.len()
    );
    Ok(())
}

fn code_diff(label: &str, actual: &[u16], reference: &[u16]) -> Result<(), Fail> {
    if actual.len() != TOKENS * DEPTHS || reference.len() != actual.len() {
        return Err("MiMo RVQ code comparison extent changed".into());
    }
    let mut differences = 0;
    for (index, (&got, &want)) in actual.iter().zip(reference).enumerate() {
        if got != want {
            differences += 1;
            println!(
                "mimo_audio_code_diff\t{label}\trow={}\tdepth={}\tactual={got}\treference={want}",
                index / DEPTHS,
                index % DEPTHS
            );
        }
    }
    println!(
        "mimo_audio_codes\t{label}\tmatching={}/{}",
        actual.len() - differences,
        actual.len()
    );
    Ok(())
}

#[test]
#[ignore = "requires pinned bundled weights and a dedicated target GPU"]
fn prepared_mel_target_features_and_rvq_diagnostic() -> Result<(), Fail> {
    assert_eq!(
        SOURCE,
        "XiaomiMiMo/MiMo-V2.6-Flash-RL@3b38d063180c3e4aed9691fdc735f3d10b266ee4"
    );
    assert_eq!(
        LFS_WEIGHT_SHA256,
        "077033345d80eef3a315e8d394e0589667e80e4cdaba9bc5a7488410c6657265"
    );
    assert_eq!(
        format!("{:x}", Sha256::digest(CPU_FEATURES)),
        "8ca5d8fcb8e8cc3c07043f458f3a3e0c57d609609638a2522a3365ec4890d174"
    );
    assert_eq!(
        format!("{:x}", Sha256::digest(PUBLISHER_GPU_FEATURES)),
        "103eab4231e8acef815884e438df7fcb63ab908071fd7a99abff59d1544c90db"
    );
    assert_eq!(
        format!("{:x}", Sha256::digest(PUBLISHER_GPU_FRONTEND)),
        "ede628d13a85d9be7e4522aa0b03b365a815b90b78eaca07c985fff78c96a15c"
    );
    assert_eq!(
        format!("{:x}", Sha256::digest(PUBLISHER_GPU_CONV1)),
        "5a8828316406c5702487789ed75a08ef49b7c9a8ed2d97de1763082c718afbe2"
    );
    assert_eq!(
        format!("{:x}", Sha256::digest(PUBLISHER_GPU_CONV1_PRE_GELU)),
        "d22f4ee4205209c45c70695a3dbd53dd45cf2cf9c3675475dac8bea63a749c78"
    );
    assert_eq!(
        format!("{:x}", Sha256::digest(PUBLISHER_GPU_CONV1_LINEAR)),
        "06d8460ae269a1b7bbb57f449c6126ad900537b86fd1e86b68debe5c8acc267d"
    );
    for (bytes, expected) in [
        (
            PUBLISHER_GPU_CONV1_LINEAR_CONTROL,
            "bb12e6a3a33b537114e666c21ae5ead6051891973e44a24b4cedcef5fb785847",
        ),
        (
            PUBLISHER_GPU_FRONTEND_LINEAR_CONTROL,
            "d9032c925246ca9df4aa7b815d3aee54219eb2a80513aeb3d2d0d7e44aa32bfc",
        ),
        (
            PUBLISHER_GPU_STACK_LINEAR_CONTROL,
            "cf54a19070fe8d8b1dc8d8c08118409b4b280184aed22c2ece31231c39690117",
        ),
        (
            PUBLISHER_GPU_PRE_RVQ_LINEAR_CONTROL,
            "b6a3ebda9b9ead1390d71239d15fe0589f23c0e13f61471bfb99783e0e475500",
        ),
    ] {
        assert_eq!(format!("{:x}", Sha256::digest(bytes)), expected);
    }
    assert_eq!(
        format!("{:x}", Sha256::digest(PUBLISHER_GPU_CONV2_PRE_GELU)),
        "0291a89751356add18f82512b93e03d6f3c34d74d72f26065cedb60f4f846a92"
    );
    assert_eq!(
        format!(
            "{:x}",
            Sha256::digest(PUBLISHER_GPU_CONV2_PRE_GELU_LINEAR_CONTROL)
        ),
        "c224a2b9204945c41f67fc8cd753f49a9ceed0942017ab2d7c3d5e60ee1f6d7f"
    );
    assert_eq!(
        format!("{:x}", Sha256::digest(PUBLISHER_GPU_LAYER0_CUDNN)),
        "6f17d647d78f434d3b23cef1553d6bf95631ad5332ca7d749ad6a6d29eaf8d65"
    );
    assert_eq!(
        format!("{:x}", Sha256::digest(PUBLISHER_GPU_LAYER0_LINEAR_CONTROL)),
        "70624fd0c6dc490801117559883ab75707667d3bb0326e300463405b317d324d"
    );
    for (bytes, expected) in [
        (
            PUBLISHER_GPU_LAYER0_ATTN_NORM_LINEAR_CONTROL,
            "6ca25de5a479365ba9d7cdd0b172e10487abf61e2d9bc9d220383f1ab3e0d089",
        ),
        (
            PUBLISHER_GPU_LAYER0_ATTENTION_LINEAR_CONTROL,
            "abb16d8d8e2f3528062b90869267a77c233310f6aaaffd154bbeb43effcd351f",
        ),
        (
            PUBLISHER_GPU_LAYER0_MLP_NORM_LINEAR_CONTROL,
            "d3ace5bb650a597bb959c7ae687e091d810fba939293ad696465ced282133a70",
        ),
        (
            PUBLISHER_GPU_LAYER0_FC2_LINEAR_CONTROL,
            "699708e10dcb9ddb6d8b5ebac04268f7e7c26da0e63e9fbd8c3c7937c817913b",
        ),
        (
            PUBLISHER_GPU_LAYER0_FC2_INPUT_LINEAR_CONTROL,
            "0d26edb25c9814cf4db4709f5995ec55cdbce777209158470a23508a5320d07a",
        ),
    ] {
        assert_eq!(format!("{:x}", Sha256::digest(bytes)), expected);
    }
    assert_eq!(
        format!("{:x}", Sha256::digest(PUBLISHER_GPU_STACK)),
        "fc95bf4a5f0bc569ce8fb0d31c19824395e92f2e4c5e9271aa06a2926a6aafd2"
    );
    assert_eq!(MEL.len(), 9 * MEL_CHANNELS * 4);
    let mel = MEL
        .chunks_exact(4)
        .map(|word| f32::from_le_bytes(word.try_into().expect("four-byte F32")))
        .collect::<Vec<_>>();
    let root = std::env::var("MIMO_PINNED_SOURCE_ROOT")?;
    let gpu = std::env::var("MEMRA_MIMO_COMPONENT_GPU")
        .unwrap_or_else(|_| "0".into())
        .parse()?;
    let engine = Engine::new(gpu)?;
    let weights = MiMoAudioCodecEncoderWeights::load(&engine, Path::new(&root))?;
    let preactivation =
        weights.encode_prepared_mel_conv1_preact(&engine, &engine.htod(&mel)?, 9)?;
    let publisher_preactivation = decode_bf16(PUBLISHER_GPU_CONV1_PRE_GELU, 9)?;
    feature_stats(
        "memra_vs_gpu_publisher_conv1_pre_gelu",
        &preactivation,
        &publisher_preactivation,
        9,
    )?;
    let publisher_linear = decode_bf16(PUBLISHER_GPU_CONV1_LINEAR, 9)?;
    feature_stats(
        "memra_vs_gpu_publisher_conv1_linear",
        &preactivation,
        &publisher_linear,
        9,
    )?;
    feature_stats(
        "gpu_publisher_linear_vs_cudnn_conv1",
        &publisher_linear,
        &publisher_preactivation,
        9,
    )?;
    let first_conv = weights.encode_prepared_mel_conv1(&engine, &engine.htod(&mel)?, 9)?;
    let first_conv_values = engine.dtoh(&first_conv)?;
    let publisher_gpu_conv1 = decode_bf16(PUBLISHER_GPU_CONV1, 9)?;
    feature_stats(
        "memra_vs_gpu_publisher_conv1",
        &first_conv_values,
        &publisher_gpu_conv1,
        9,
    )?;
    let publisher_linear_conv1 = decode_bf16(PUBLISHER_GPU_CONV1_LINEAR_CONTROL, 9)?;
    feature_stats(
        "memra_vs_gpu_publisher_conv1_linear_control",
        &first_conv_values,
        &publisher_linear_conv1,
        9,
    )?;
    let source_gelu = MiMoAudioCodecEncoderWeights::encode_gelu_from_preact(
        &engine,
        &engine.htod(&publisher_preactivation)?,
        9,
    )?;
    let source_gelu = engine.dtoh(&source_gelu)?;
    feature_stats(
        "memra_gelu_on_gpu_publisher_preact_vs_post",
        &source_gelu,
        &publisher_gpu_conv1,
        9,
    )?;
    let second_preact = weights.encode_prepared_mel_conv2_preact(&engine, &first_conv, 9)?;
    let publisher_gpu_conv2_preact = decode_bf16(PUBLISHER_GPU_CONV2_PRE_GELU, 5)?;
    let publisher_linear_conv2_preact =
        decode_bf16(PUBLISHER_GPU_CONV2_PRE_GELU_LINEAR_CONTROL, 5)?;
    feature_stats(
        "memra_vs_gpu_publisher_conv2_preact",
        &second_preact,
        &publisher_gpu_conv2_preact,
        5,
    )?;
    feature_stats(
        "memra_vs_gpu_publisher_conv2_linear_control",
        &second_preact,
        &publisher_linear_conv2_preact,
        5,
    )?;
    let fused_bias =
        weights.encode_prepared_mel_conv2_fused_bias_diagnostic(&engine, &first_conv, 9)?;
    feature_stats(
        "memra_fused_bias_vs_gpu_publisher_conv2_linear_control",
        &fused_bias,
        &publisher_linear_conv2_preact,
        5,
    )?;
    feature_stats(
        "memra_fused_bias_vs_plain_conv2_preact",
        &fused_bias,
        &second_preact,
        5,
    )?;
    let source_conv2_gelu = MiMoAudioCodecEncoderWeights::encode_gelu_from_preact(
        &engine,
        &engine.htod(&publisher_linear_conv2_preact)?,
        5,
    )?;
    let publisher_linear_frontend = decode_bf16(PUBLISHER_GPU_FRONTEND_LINEAR_CONTROL, 5)?;
    feature_stats(
        "memra_gelu_on_gpu_publisher_conv2_linear_preact_vs_post",
        &engine.dtoh(&source_conv2_gelu)?,
        &publisher_linear_frontend,
        5,
    )?;
    let fused_frontend = MiMoAudioCodecEncoderWeights::encode_gelu_from_preact(
        &engine,
        &engine.htod(&fused_bias)?,
        5,
    )?;
    let fused_frontend_values = engine.dtoh(&fused_frontend)?;
    let layer0_capture = crate::mimo_audio_codec_layer::start_layer0_stage_capture();
    let fused_layer0 = weights.encode_one_transformer_layer(&engine, &fused_frontend, 5, 0)?;
    let layer0_stages = layer0_capture.finish();
    for ((label, actual), (expected_label, fixture)) in layer0_stages.iter().zip([
        (
            "layer0_attn_norm",
            PUBLISHER_GPU_LAYER0_ATTN_NORM_LINEAR_CONTROL,
        ),
        (
            "layer0_attention",
            PUBLISHER_GPU_LAYER0_ATTENTION_LINEAR_CONTROL,
        ),
        (
            "layer0_mlp_norm",
            PUBLISHER_GPU_LAYER0_MLP_NORM_LINEAR_CONTROL,
        ),
        ("layer0_fc2", PUBLISHER_GPU_LAYER0_FC2_LINEAR_CONTROL),
        ("layer0", PUBLISHER_GPU_LAYER0_LINEAR_CONTROL),
    ]) {
        assert_eq!(*label, expected_label);
        feature_stats(
            &format!("memra_fused_bias_{label}_vs_gpu_publisher_linear_control"),
            actual,
            &decode_bf16(fixture, 5)?,
            5,
        )?;
    }
    assert_eq!(layer0_stages.len(), 5);
    let publisher_fc2_input =
        decode_bf16_width(PUBLISHER_GPU_LAYER0_FC2_INPUT_LINEAR_CONTROL, 5, 4_096)?;
    let memra_fc2_from_source =
        weights.project_layer0_fc2_source_input(&engine, &engine.htod(&publisher_fc2_input)?, 5)?;
    feature_stats(
        "memra_fc2_from_source_input_vs_gpu_publisher_linear_control",
        &engine.dtoh(&memra_fc2_from_source)?,
        &decode_bf16(PUBLISHER_GPU_LAYER0_FC2_LINEAR_CONTROL, 5)?,
        5,
    )?;
    let fused_layer0_values = engine.dtoh(&fused_layer0)?;
    let fused_stack = weights.encode_transformer_stack(&engine, &fused_frontend, 5)?;
    let fused_stack_values = engine.dtoh(&fused_stack)?;
    let fused_features = weights.downsample_post_stack(&engine, &fused_stack, 5)?;
    let fused_feature_values = engine.dtoh(&fused_features)?;
    let fused_ids = weights
        .encode_20_rvq(&engine, &fused_features, TOKENS)?
        .code_ids;
    let first = weights.encode_prepared_mel_conv(&engine, &engine.htod(&mel)?, 9)?;
    let frontend_values = engine.dtoh(&first)?;
    let default_layer0 = weights.encode_one_transformer_layer(&engine, &first, 5, 0)?;
    let default_layer0_values = engine.dtoh(&default_layer0)?;
    let stack = weights.encode_transformer_stack(&engine, &first, 5)?;
    let stack_values = engine.dtoh(&stack)?;
    let features = weights.downsample_post_stack(&engine, &stack, 5)?;
    let memra_features = engine.dtoh(&features)?;
    let cpu_features = decode_bf16(CPU_FEATURES, TOKENS)?;
    let publisher_gpu_features = decode_bf16(PUBLISHER_GPU_FEATURES, TOKENS)?;
    let publisher_gpu_frontend = decode_bf16(PUBLISHER_GPU_FRONTEND, 5)?;
    let publisher_gpu_stack = decode_bf16(PUBLISHER_GPU_STACK, 5)?;
    let publisher_linear_frontend = decode_bf16(PUBLISHER_GPU_FRONTEND_LINEAR_CONTROL, 5)?;
    let publisher_linear_stack = decode_bf16(PUBLISHER_GPU_STACK_LINEAR_CONTROL, 5)?;
    let publisher_linear_pre_rvq = decode_bf16(PUBLISHER_GPU_PRE_RVQ_LINEAR_CONTROL, TOKENS)?;
    let publisher_cudnn_layer0 = decode_bf16(PUBLISHER_GPU_LAYER0_CUDNN, 5)?;
    let publisher_linear_layer0 = decode_bf16(PUBLISHER_GPU_LAYER0_LINEAR_CONTROL, 5)?;
    feature_stats(
        "memra_default_layer0_vs_gpu_publisher_cudnn",
        &default_layer0_values,
        &publisher_cudnn_layer0,
        5,
    )?;
    feature_stats(
        "memra_fused_bias_layer0_vs_gpu_publisher_linear_control",
        &fused_layer0_values,
        &publisher_linear_layer0,
        5,
    )?;
    feature_stats(
        "memra_fused_bias_frontend_vs_gpu_publisher_linear_control",
        &fused_frontend_values,
        &publisher_linear_frontend,
        5,
    )?;
    feature_stats(
        "memra_fused_bias_stack_vs_gpu_publisher_linear_control",
        &fused_stack_values,
        &publisher_linear_stack,
        5,
    )?;
    feature_stats(
        "memra_fused_bias_pre_rvq_vs_gpu_publisher_linear_control",
        &fused_feature_values,
        &publisher_linear_pre_rvq,
        TOKENS,
    )?;
    feature_stats(
        "memra_vs_gpu_publisher_frontend",
        &frontend_values,
        &publisher_gpu_frontend,
        5,
    )?;
    feature_stats(
        "memra_vs_gpu_publisher_stack",
        &stack_values,
        &publisher_gpu_stack,
        5,
    )?;
    feature_stats(
        "memra_vs_gpu_publisher_frontend_linear_control",
        &frontend_values,
        &publisher_linear_frontend,
        5,
    )?;
    feature_stats(
        "memra_vs_gpu_publisher_stack_linear_control",
        &stack_values,
        &publisher_linear_stack,
        5,
    )?;
    feature_stats(
        "memra_vs_gpu_publisher_pre_rvq_linear_control",
        &memra_features,
        &publisher_linear_pre_rvq,
        TOKENS,
    )?;
    feature_stats(
        "memra_vs_cpu_publisher",
        &memra_features,
        &cpu_features,
        TOKENS,
    )?;
    feature_stats(
        "memra_vs_gpu_publisher",
        &memra_features,
        &publisher_gpu_features,
        TOKENS,
    )?;
    feature_stats(
        "cpu_vs_gpu_publisher",
        &cpu_features,
        &publisher_gpu_features,
        TOKENS,
    )?;

    let memra_ids = weights.encode_20_rvq(&engine, &features, TOKENS)?.code_ids;
    code_diff("memra_vs_cpu_publisher", &memra_ids, &CPU_IDS)?;
    code_diff("memra_vs_gpu_publisher", &memra_ids, &PUBLISHER_GPU_IDS)?;
    code_diff(
        "memra_vs_gpu_publisher_linear_control",
        &memra_ids,
        &PUBLISHER_GPU_LINEAR_CONTROL_IDS,
    )?;
    code_diff(
        "memra_fused_bias_vs_gpu_publisher_linear_control",
        &fused_ids,
        &PUBLISHER_GPU_LINEAR_CONTROL_IDS,
    )?;
    code_diff(
        "memra_fused_bias_vs_gpu_publisher_cudnn",
        &fused_ids,
        &PUBLISHER_GPU_IDS,
    )?;
    for (label, reference_features, expected) in [
        ("gpu_rvq_on_cpu_features", &cpu_features, &CPU_IDS),
        (
            "gpu_rvq_on_gpu_publisher_features",
            &publisher_gpu_features,
            &PUBLISHER_GPU_IDS,
        ),
    ] {
        let input = engine.htod(reference_features)?;
        let ids = weights.encode_20_rvq(&engine, &input, TOKENS)?.code_ids;
        code_diff(label, &ids, expected)?;
    }
    Ok(())
}
