//! Layer-streamed, two-device MiMo source text-token diagnostic.
//! This is an offline arithmetic rung, not resident serving or model support.

use std::fmt::Write as _;
use std::io::Write as _;
use std::path::Path;
use std::time::Instant;

use cudarc::driver::CudaSlice;
use memra_engine::Engine;
use memra_engine::QT_F8_E4M3_BLK;
use memra_engine::mimo_source_moe::{PinnedMiMoSource, ResidentMiMoMoeLayer, source_moe_token};
use memra_engine::model::GpuTensor;
use memra_gguf::config::{HfConfig, ModelConfig};
use memra_gguf::model_packs;
use memra_gguf::model_plan::{
    ActivationPlan, AttentionPlan, FullAttentionPlan, LayerPlan, MlpPlan, ModelPlan, NormKind,
    ResidualTopology, RopeFactors, StatePlan, WeightTransform,
};
use memra_gguf::safetensors::StModel;
use memra_gguf::source::{SafetensorsSource, TensorSource};
use sha2::{Digest, Sha256};

type Fail = Box<dyn std::error::Error>;

const CONFIG_SHA256: &str = "61bea4a0f7a0dd8969f8cae528761e26b697dd12ff63e98804c3f0945492e621";
const REVISION: &str = "3b38d063180c3e4aed9691fdc735f3d10b266ee4";
const HIDDEN: usize = 4096;
const VOCAB: usize = 152576;
const LAYERS: usize = 48;
const STAGE_CUT: usize = 24;
const QK: usize = 192;
const VALUE: usize = 128;
const GLOBAL_LAYERS: [usize; 9] = [0, 5, 11, 17, 23, 29, 35, 41, 47];

fn decode_bf16(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|pair| f32::from_bits(u32::from(u16::from_le_bytes([pair[0], pair[1]])) << 16))
        .collect()
}

fn read_vector(model: &StModel, name: &str, length: usize) -> Result<Vec<f32>, Fail> {
    let (info, bytes) = model.raw(name).ok_or_else(|| format!("{name}: missing"))?;
    if info.shape != [length as u64] {
        return Err(format!("{name}: wrong shape {:?}", info.shape).into());
    }
    let values = match info.dtype.as_str() {
        "BF16" if bytes.len() == length * 2 => decode_bf16(bytes),
        "F32" if bytes.len() == length * 4 => bytes
            .chunks_exact(4)
            .map(|v| f32::from_le_bytes([v[0], v[1], v[2], v[3]]))
            .collect(),
        _ => return Err(format!("{name}: wrong dtype or byte length").into()),
    };
    if values.iter().any(|value| !value.is_finite()) {
        return Err(format!("{name}: non-finite value").into());
    }
    Ok(values)
}

fn embedding_row(model: &StModel, token: usize) -> Result<Vec<f32>, Fail> {
    let name = "model.embed_tokens.weight";
    let (info, bytes) = model.raw(name).ok_or("source embedding is missing")?;
    if info.dtype != "BF16"
        || info.shape != [VOCAB as u64, HIDDEN as u64]
        || bytes.len() != VOCAB * HIDDEN * 2
        || token >= VOCAB
    {
        return Err("source embedding storage or token id changed".into());
    }
    let start = token * HIDDEN * 2;
    let values = decode_bf16(&bytes[start..start + HIDDEN * 2]);
    if values.iter().any(|value| !value.is_finite()) {
        return Err("source embedding contains a non-finite value".into());
    }
    Ok(values)
}

fn bf16_matrix(
    engine: &Engine,
    model: &StModel,
    name: &str,
    rows: usize,
    cols: usize,
) -> Result<GpuTensor, Fail> {
    let (info, bytes) = model.raw(name).ok_or_else(|| format!("{name}: missing"))?;
    if info.dtype != "BF16"
        || info.shape != [rows as u64, cols as u64]
        || bytes.len() != rows * cols * 2
    {
        return Err(format!("{name}: BF16 matrix geometry changed").into());
    }
    Ok(GpuTensor::FloatBf16 {
        data: engine.htod_bytes(bytes)?,
        ne: vec![cols as u64, rows as u64],
    })
}

fn normalized(
    engine: &Engine,
    model: &StModel,
    input: &CudaSlice<f32>,
    name: &str,
    norm: &memra_gguf::model_plan::NormPlan,
) -> Result<CudaSlice<f32>, Fail> {
    if norm.kind != NormKind::Rms
        || norm.weight_transform != WeightTransform::Identity
        || input.len() != HIDDEN
    {
        return Err(format!("{name}: unsupported MiMo norm").into());
    }
    let weight = engine.htod(&read_vector(model, name, HIDDEN)?)?;
    let mut output = engine.uninit(HIDDEN)?;
    engine.rms_norm(input, &weight, &mut output, HIDDEN, 1, norm.epsilon)?;
    Ok(output)
}

fn checked_attention(plan: &LayerPlan) -> Result<(&FullAttentionPlan, bool), Fail> {
    let layer = plan.index as usize;
    let global = GLOBAL_LAYERS.contains(&layer);
    let attention = match (&plan.attention, global) {
        (AttentionPlan::Full(full), true) if full.kv_heads == 4 => full,
        (AttentionPlan::SlidingWindow { attention, window }, false)
            if *window == 128 && attention.kv_heads == 8 =>
        {
            attention
        }
        _ => return Err(format!("layer {layer}: unsupported MiMo attention").into()),
    };
    let state_is_pinned = matches!(
        (&plan.state, global),
        (
            StatePlan::KvCache {
                key_width: 768,
                value_width: 512,
            },
            true,
        ) | (
            StatePlan::SlidingKvCache {
                key_width: 1536,
                value_width: 1024,
                window: 128,
            },
            false,
        )
    );
    if attention.query_heads != 64
        || attention.key_head_dim as usize != QK
        || attention.value_head_dim as usize != VALUE
        || attention.rope.dimensions != 64
        || !matches!(attention.rope.factors, RopeFactors::None)
        || !state_is_pinned
        || plan.residual != ResidualTopology::Serial
        || plan.sparse_overlay.is_some()
        || plan.ple.is_some()
    {
        return Err(format!("layer {layer}: changed attention geometry or RoPE").into());
    }
    Ok((attention, global))
}

struct KvState {
    key: CudaSlice<f32>,
    value: CudaSlice<f32>,
    tokens: usize,
}

fn append_kv(
    engine: &Engine,
    slot: &mut Option<KvState>,
    position: usize,
    key: CudaSlice<f32>,
    value: CudaSlice<f32>,
    kv_heads: usize,
) -> Result<usize, Fail> {
    let key_width = kv_heads * QK;
    let value_width = kv_heads * VALUE;
    let device = engine.stream().context().ordinal();
    if position > 1
        || key.len() != key_width
        || value.len() != value_width
        || key.ordinal() != device
        || value.ordinal() != device
    {
        return Err("MiMo two-token KV append geometry changed".into());
    }
    if position == 0 {
        if slot.is_some() {
            return Err("MiMo KV slot already holds a prior token".into());
        }
        *slot = Some(KvState {
            key,
            value,
            tokens: 1,
        });
        return Ok(0);
    }
    let prior = slot.take().ok_or("MiMo continuation has no cached KV")?;
    if prior.tokens != position
        || prior.key.len() != position * key_width
        || prior.value.len() != position * value_width
        || prior.key.ordinal() != device
        || prior.value.ordinal() != device
    {
        return Err("MiMo continuation KV position or shape differs".into());
    }
    let mut keys = engine.uninit((position + 1) * key_width)?;
    let mut values = engine.uninit((position + 1) * value_width)?;
    engine.dtod_copy_into(&prior.key, &mut keys, 0)?;
    engine.dtod_copy_into(&key, &mut keys, position * key_width)?;
    engine.dtod_copy_into(&prior.value, &mut values, 0)?;
    engine.dtod_copy_into(&value, &mut values, position * value_width)?;
    *slot = Some(KvState {
        key: keys,
        value: values,
        tokens: position + 1,
    });
    Ok(position)
}

fn attention_token(
    engine: &Engine,
    model: &StModel,
    source: &SafetensorsSource,
    plan: &LayerPlan,
    hidden: &CudaSlice<f32>,
    position: usize,
    kv_slot: &mut Option<KvState>,
) -> Result<(CudaSlice<f32>, usize), Fail> {
    let layer = plan.index as usize;
    let (attention, global) = checked_attention(plan)?;
    let kv_heads = attention.kv_heads as usize;
    let prefix = format!("model.layers.{layer}");
    let norm = normalized(
        engine,
        model,
        hidden,
        &format!("{prefix}.input_layernorm.weight"),
        &plan.pre_attention_norm,
    )?;
    let qkv_name = format!("{prefix}.self_attn.qkv_proj.weight");
    let views = source
        .find_fp8_mimo_qkv_shards(&qkv_name)
        .ok_or_else(|| format!("{qkv_name}: missing four native views"))?;
    if views.len() != 4 {
        return Err(format!("{qkv_name}: wrong source shard count").into());
    }
    let expected_rows = 3072 + kv_heads / 4 * (QK + VALUE);
    let mut projections = Vec::with_capacity(4);
    for (index, view) in views.iter().enumerate() {
        if view.out_f != expected_rows || view.in_f != HIDDEN {
            return Err(format!("{qkv_name} shard {index}: changed geometry").into());
        }
        let weight = GpuTensor::load_mimo_fp8_qkv_shard(engine, view)?;
        projections.push(engine.matmul(&weight, &norm, 1)?);
    }
    let mut qkv = engine.mimo_gather_qkv(
        [
            &projections[0],
            &projections[1],
            &projections[2],
            &projections[3],
        ],
        1,
        attention,
    )?;
    drop((projections, norm));
    let position_gpu = engine.htod_i32(&[position as i32])?;
    engine.rope_neox(
        &mut qkv.query,
        &position_gpu,
        QK,
        64,
        64,
        1,
        attention.rope.base,
        1.0,
    )?;
    engine.rope_neox(
        &mut qkv.key,
        &position_gpu,
        QK,
        64,
        kv_heads,
        1,
        attention.rope.base,
        1.0,
    )?;
    let sink = if !global {
        Some(engine.htod(&read_vector(
            model,
            &format!("{prefix}.self_attn.attention_sink_bias"),
            64,
        )?)?)
    } else {
        None
    };
    let cached_before = append_kv(engine, kv_slot, position, qkv.key, qkv.value, kv_heads)?;
    let cache = kv_slot.as_ref().ok_or("MiMo KV append lost its state")?;
    let context = engine.mimo_sink_decode(
        &qkv.query,
        &cache.key,
        &cache.value,
        sink.as_ref(),
        position + 1,
        &plan.attention,
    )?;
    drop((qkv.query, sink));
    let output = bf16_matrix(
        engine,
        model,
        &format!("{prefix}.self_attn.o_proj.weight"),
        HIDDEN,
        64 * VALUE,
    )?;
    Ok((engine.matmul(&output, &context, 1)?, cached_before))
}

fn native_fp8_matrix(
    engine: &Engine,
    source: &SafetensorsSource,
    name: &str,
    rows: usize,
    cols: usize,
) -> Result<GpuTensor, Fail> {
    let view = source
        .find_fp8_native(name)
        .ok_or_else(|| format!("{name}: source FP8 native view missing"))?;
    if view.in_f != cols || view.out_f != rows || view.blk.is_none() {
        return Err(format!("{name}: changed source block-FP8 shape or grid").into());
    }
    let weight = GpuTensor::load_from_source(engine, source, name)?;
    if weight.in_features() != cols
        || weight.out_features() != rows
        || !matches!(
            &weight,
            GpuTensor::Quant {
                qtype: QT_F8_E4M3_BLK,
                ..
            }
        )
    {
        return Err(format!("{name}: did not retain source block-FP8 residency").into());
    }
    Ok(weight)
}

fn dense_mlp_token(
    engine: &Engine,
    source: &SafetensorsSource,
    input: &CudaSlice<f32>,
    plan: &memra_gguf::model_plan::DenseMlpPlan,
) -> Result<CudaSlice<f32>, Fail> {
    if plan.intermediate_size != 16384 || plan.activation != ActivationPlan::Silu {
        return Err("MiMo dense layer 0 MLP contract changed".into());
    }
    let gate = native_fp8_matrix(engine, source, "blk.0.ffn_gate.weight", 16384, HIDDEN)?;
    let gate_out = engine.matmul(&gate, input, 1)?;
    drop(gate);
    let up = native_fp8_matrix(engine, source, "blk.0.ffn_up.weight", 16384, HIDDEN)?;
    let up_out = engine.matmul(&up, input, 1)?;
    drop(up);
    let mut activated = engine.uninit(16384)?;
    engine.silu_mul(&gate_out, &up_out, &mut activated, 16384)?;
    drop((gate_out, up_out));
    let down = native_fp8_matrix(engine, source, "blk.0.ffn_down.weight", HIDDEN, 16384)?;
    engine.matmul(&down, &activated, 1)
}

fn validate_plan(plan: &ModelPlan) -> Result<(), Fail> {
    if plan.layers.len() != LAYERS
        || plan.hidden_size as usize != HIDDEN
        || plan.vocab_size as usize != VOCAB
        || plan.embedding_scale.to_bits() != 1.0f32.to_bits()
        || !plan.logits.is_empty()
        || plan.output_norm.kind != NormKind::Rms
        || plan.output_norm.weight_transform != WeightTransform::Identity
        || !plan.partition_boundaries.contains(&STAGE_CUT)
    {
        return Err("MiMo text plan geometry changed".into());
    }
    for (index, layer) in plan.layers.iter().enumerate() {
        if layer.index as usize != index
            || layer.pre_attention_norm.kind != NormKind::Rms
            || layer.pre_attention_norm.weight_transform != WeightTransform::Identity
            || layer.pre_mlp_norm.kind != NormKind::Rms
            || layer.pre_mlp_norm.weight_transform != WeightTransform::Identity
        {
            return Err(format!("MiMo layer {index}: norm or index changed").into());
        }
        checked_attention(layer)?;
        match (&layer.mlp, index) {
            (MlpPlan::Dense(dense), 0)
                if dense.intermediate_size == 16384 && dense.activation == ActivationPlan::Silu => {
            }
            (MlpPlan::Moe(moe), 1..)
                if moe.expert_count == 256
                    && moe.experts_per_token == 8
                    && moe.expert_intermediate_size == 2048
                    && moe.activation == ActivationPlan::Silu
                    && moe.shared.is_none() => {}
            _ => return Err(format!("MiMo layer {index}: MLP plan changed").into()),
        }
    }
    Ok(())
}

fn run() -> Result<(), Fail> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 5 || args.len() > 7 {
        return Err(
            "usage: mimo_source_gpu_token <source_dir> <gpu0> <gpu1> <token_id> <report.tsv> [--continue-one] [--resident-moe]"
                .into(),
        );
    }
    let mut continue_one = false;
    let mut resident_moe = false;
    for option in args.iter().skip(5) {
        match option.as_str() {
            "--continue-one" if !continue_one => continue_one = true,
            "--resident-moe" if !resident_moe => resident_moe = true,
            _ => return Err(format!("unknown or repeated MiMo token option: {option}").into()),
        }
    }
    let dir = Path::new(&args[0]);
    let gpu0: usize = args[1].parse()?;
    let gpu1: usize = args[2].parse()?;
    let token: usize = args[3].parse()?;
    let report_path = Path::new(&args[4]);
    let staged_path = report_path.with_extension("tsv.writing");
    if gpu0 == gpu1 || report_path.exists() || staged_path.exists() {
        return Err("MiMo stage devices must differ and report paths must not exist".into());
    }
    let started = Instant::now();
    let config_bytes = std::fs::read(dir.join("config.json"))?;
    let digest = format!("{:x}", Sha256::digest(&config_bytes));
    if digest != CONFIG_SHA256 {
        return Err("MiMo source config differs from pinned revision".into());
    }
    let config_text = String::from_utf8(config_bytes)?;
    let config = ModelConfig::from_hf(&HfConfig::try_parse(&config_text)?);
    let pack = model_packs::by_alias("mimo_v2_source")
        .ok_or("MiMo source inspection profile unavailable")?;
    let plan = pack.compile_plan(&config)?;
    validate_plan(&plan)?;
    if !memra_engine::fp8_ffi::st_e4m3_blk_enabled() {
        return Err("MiMo source token requires native block-FP8 residency".into());
    }
    let model = StModel::open(dir)?;
    let pinned = PinnedMiMoSource::bind(&model, &config, &plan)?;
    if pinned.semantic_tensors() != 36922 {
        return Err("MiMo source semantic binding count changed".into());
    }
    let source = SafetensorsSource::open(dir)?;
    for layer in &plan.layers {
        let index = layer.index as usize;
        let (attention, _) = checked_attention(layer)?;
        let name = format!("model.layers.{index}.self_attn.qkv_proj.weight");
        let views = source
            .find_fp8_mimo_qkv_shards(&name)
            .ok_or_else(|| format!("{name}: missing native QKV shards"))?;
        let expected_rows = 3072 + (attention.kv_heads as usize / 4) * (QK + VALUE);
        if views.len() != 4
            || views
                .iter()
                .any(|view| view.out_f != expected_rows || view.in_f != HIDDEN)
        {
            return Err(format!("{name}: native QKV shard geometry changed").into());
        }
    }
    for (name, rows, cols) in [
        ("blk.0.ffn_gate.weight", 16384, HIDDEN),
        ("blk.0.ffn_up.weight", 16384, HIDDEN),
        ("blk.0.ffn_down.weight", HIDDEN, 16384),
    ] {
        let view = source
            .find_fp8_native(name)
            .ok_or_else(|| format!("{name}: native block-FP8 view missing"))?;
        if view.out_f != rows || view.in_f != cols || view.blk.is_none() {
            return Err(format!("{name}: native block-FP8 geometry changed").into());
        }
    }
    embedding_row(&model, token)?;
    let engines = [Engine::new(gpu0)?, Engine::new(gpu1)?];
    let mut resident_layers: Vec<Option<ResidentMiMoMoeLayer>> =
        std::iter::repeat_with(|| None).take(LAYERS).collect();
    let mut resident_bytes = [0usize; 2];
    let mut resident_load_ms = 0.0f64;
    if resident_moe {
        let load_start = Instant::now();
        for (index, layer) in plan.layers.iter().enumerate().skip(1) {
            let stage = usize::from(index >= STAGE_CUT);
            let engine = &engines[stage];
            engine.gpu.ctx.bind_to_thread()?;
            let MlpPlan::Moe(moe) = &layer.mlp else {
                return Err(format!("MiMo resident layer {index} has no MoE plan").into());
            };
            let before = Instant::now();
            let resident = ResidentMiMoMoeLayer::load(engine, &pinned, index, moe)?;
            resident_bytes[stage] += resident.resident_bytes();
            resident_layers[index] = Some(resident);
            eprintln!(
                "MiMo resident layer {index} stage {stage} loaded in {:.3}s; cumulative stage bytes {}",
                before.elapsed().as_secs_f64(),
                resident_bytes[stage]
            );
        }
        resident_load_ms = load_start.elapsed().as_secs_f64() * 1000.0;
    }
    let mut kv: Vec<Option<KvState>> = std::iter::repeat_with(|| None).take(LAYERS).collect();
    let turns = if continue_one { 2 } else { 1 };
    let mut report = if continue_one {
        String::from("format\tmemra-mimo-source-gpu-token-v2\n")
    } else {
        String::from("format\tmemra-mimo-source-gpu-token-v1\n")
    };
    writeln!(report, "model\tXiaomiMiMo/MiMo-V2.6-Flash-RL@{REVISION}")?;
    writeln!(report, "payload_verification\texternal_hf_verify_required")?;
    writeln!(report, "config_sha256\t{CONFIG_SHA256}")?;
    writeln!(
        report,
        "source_header_digest\t5ebbdd27e45716b805fc2bdf115c8345b4bfc03c6012b860f76b6b222c758aee"
    )?;
    writeln!(
        report,
        "numeric_class\tmemra_block_fp8_q8_1_act_and_mxfp4_pow2_e4m3_moe_act"
    )?;
    writeln!(report, "stage_cut_before_layer\t{STAGE_CUT}")?;
    writeln!(report, "stage_transfer\thost_bounce")?;
    writeln!(
        report,
        "weight_residency\t{}",
        if resident_moe {
            "source_mxfp4_moe_resident"
        } else {
            "layer_streamed"
        }
    )?;
    writeln!(report, "resident_moe_bytes_stage0\t{}", resident_bytes[0])?;
    writeln!(report, "resident_moe_bytes_stage1\t{}", resident_bytes[1])?;
    writeln!(report, "resident_moe_load_ms\t{resident_load_ms:.3}")?;
    writeln!(report, "kv_format\tf32_contiguous_component")?;
    writeln!(report, "kv_append\tdevice_copy_of_prior_plus_current")?;
    writeln!(report, "kv_sequence_length\t{turns}")?;
    writeln!(report, "gpu0_ordinal\t{gpu0}")?;
    writeln!(report, "gpu1_ordinal\t{gpu1}")?;
    writeln!(report, "token_id\t{token}")?;
    writeln!(report, "generated_continuation\t{continue_one}")?;
    let mut current_token = token;
    let mut last_logits: Option<Vec<f32>> = None;
    let mut last_argmax: Option<usize> = None;
    for turn in 0..turns {
        if turn > 0
            && kv
                .iter()
                .any(|slot| slot.as_ref().is_none_or(|cached| cached.tokens != turn))
        {
            return Err("MiMo continuation did not retain every layer's native KV".into());
        }
        engines[0].gpu.ctx.bind_to_thread()?;
        let initial = embedding_row(&model, current_token)?;
        let mut hidden = engines[0].htod(&initial)?;
        writeln!(report, "processed_token\t{turn}\t{current_token}")?;
        for (index, layer) in plan.layers.iter().enumerate() {
            let stage = usize::from(index >= STAGE_CUT);
            if index == STAGE_CUT {
                let values = engines[0].dtoh(&hidden)?;
                if values.len() != HIDDEN || values.iter().any(|value| !value.is_finite()) {
                    return Err("MiMo stage transfer carried invalid hidden values".into());
                }
                engines[1].gpu.ctx.bind_to_thread()?;
                hidden = engines[1].htod(&values)?;
            }
            let engine = &engines[stage];
            engine.gpu.ctx.bind_to_thread()?;
            let before = Instant::now();
            let (attention, cached_before) = attention_token(
                engine,
                &model,
                &source,
                layer,
                &hidden,
                turn,
                &mut kv[index],
            )?;
            writeln!(report, "kv_reuse\t{turn}\t{index}\t{cached_before}\t1")?;
            let mut after_attention = engine.uninit(HIDDEN)?;
            engine.add(&hidden, &attention, &mut after_attention, HIDDEN)?;
            drop((hidden, attention));
            let post_norm = normalized(
                engine,
                &model,
                &after_attention,
                &format!("model.layers.{index}.post_attention_layernorm.weight"),
                &layer.pre_mlp_norm,
            )?;
            let mlp = match &layer.mlp {
                MlpPlan::Dense(dense) if index == 0 => {
                    dense_mlp_token(engine, &source, &post_norm, dense)?
                }
                MlpPlan::Moe(moe) if index > 0 => {
                    let result = if resident_moe {
                        resident_layers[index]
                            .as_ref()
                            .ok_or("MiMo resident MoE layer was not loaded")?
                            .token(engine, &post_norm, moe, &pinned)?
                    } else {
                        source_moe_token(engine, &pinned, index, &post_norm, moe)?
                    };
                    writeln!(
                        report,
                        "selected_experts\t{turn}\t{index}\t{:?}",
                        result.selected
                    )?;
                    writeln!(
                        report,
                        "routing_weights\t{turn}\t{index}\t{:?}",
                        result.weights
                    )?;
                    result.output
                }
                _ => return Err(format!("layer {index}: unsupported MiMo MLP plan").into()),
            };
            let mut after_mlp = engine.uninit(HIDDEN)?;
            engine.add(&after_attention, &mlp, &mut after_mlp, HIDDEN)?;
            hidden = after_mlp;
            let observed = engine.dtoh(&hidden)?;
            if observed.len() != HIDDEN || observed.iter().any(|value| !value.is_finite()) {
                return Err(format!("turn {turn} layer {index}: non-finite hidden row").into());
            }
            writeln!(
                report,
                "layer_ms\t{turn}\t{index}\t{stage}\t{:.3}",
                before.elapsed().as_secs_f64() * 1000.0
            )?;
            eprintln!("MiMo GPU turn {turn} layer {index} stage {stage} complete");
        }
        let last = &engines[1];
        let final_norm = normalized(
            last,
            &model,
            &hidden,
            "model.norm.weight",
            &plan.output_norm,
        )?;
        let head = bf16_matrix(last, &model, "lm_head.weight", VOCAB, HIDDEN)?;
        let logits_gpu = last.matmul(&head, &final_norm, 1)?;
        let logits = last.dtoh(&logits_gpu)?;
        if logits.len() != VOCAB || logits.iter().any(|value| !value.is_finite()) {
            return Err("MiMo GPU output logits are non-finite or wrong width".into());
        }
        let mut argmax = 0usize;
        for index in 1..VOCAB {
            if logits[index].total_cmp(&logits[argmax]).is_gt() {
                argmax = index;
            }
        }
        writeln!(report, "argmax_turn\t{turn}\t{argmax}")?;
        current_token = argmax;
        last_argmax = Some(argmax);
        last_logits = Some(logits);
    }
    let logits = last_logits.ok_or("MiMo token runner produced no logit row")?;
    let argmax = last_argmax.ok_or("MiMo token runner produced no argmax")?;
    writeln!(report, "vocab\t{VOCAB}")?;
    writeln!(report, "argmax\t{argmax}")?;
    writeln!(report, "wall_s\t{:.3}", started.elapsed().as_secs_f64())?;
    for (index, value) in logits.iter().enumerate() {
        writeln!(report, "logit\t{index}\t{:08x}", value.to_bits())?;
    }
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&staged_path)?;
    file.write_all(report.as_bytes())?;
    file.sync_all()?;
    drop(file);
    std::fs::rename(staged_path, report_path)?;
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("mimo_source_gpu_token: {error}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pinned_text_plan_passes_before_gpu_and_changed_state_refuses() {
        let fixture = include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../memra-gguf/src/model_packs/mimo_v2/fixtures/config.json"
        ));
        let config = ModelConfig::from_hf(&HfConfig::parse(fixture));
        let pack = model_packs::by_alias("mimo_v2_source").unwrap();
        let mut plan = pack.compile_plan(&config).unwrap();
        validate_plan(&plan).unwrap();
        plan.layers[1].state = StatePlan::KvCache {
            key_width: 1536,
            value_width: 1024,
        };
        assert!(validate_plan(&plan).is_err());
        plan = pack.compile_plan(&config).unwrap();
        plan.partition_boundaries.retain(|&cut| cut != STAGE_CUT);
        assert!(validate_plan(&plan).is_err());
    }
}
