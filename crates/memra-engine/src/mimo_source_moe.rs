//! One-token, source-MXFP4 MiMo MoE diagnostic. Expert weights are streamed
//! from safetensors for each call; this is neither a resident nor a serving path.

use std::collections::BTreeMap;
use std::error::Error;
use std::ffi::c_void;

use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use memra_gguf::config::{Arch, ModelConfig};
use memra_gguf::model_packs::mimo_v2::inspect_pinned_source_headers;
use memra_gguf::model_plan::{ActivationPlan, MlpPlan, ModelPlan, MoeMlpPlan, RouterPlan};
use memra_gguf::safetensors::StModel;

use crate::Engine;
use crate::dsv4_ffi::{memra_dsv4_act_quant_fp8, memra_dsv4_fp4_gemm};

type Fail = Box<dyn Error>;
const HIDDEN: usize = 4096;
const EXPERTS: usize = 256;
const TOP_K: usize = 8;
const EXPERT_WIDTH: usize = 2048;

pub struct MiMoMoeToken {
    pub output: CudaSlice<f32>,
    pub selected: Vec<u32>,
    pub weights: Vec<f32>,
}

fn validate_contract(config: &ModelConfig, plan: &MoeMlpPlan, layer: usize) -> Result<(), Fail> {
    let mimo = config
        .mimo
        .as_ref()
        .ok_or("source MoE requires MiMo configuration")?;
    if config.arch != Arch::MiMoV2
        || config.n_layer != 48
        || layer >= 48
        || config.n_embd as usize != HIDDEN
        || mimo.n_group != Some(1)
        || mimo.topk_group != Some(1)
        || mimo.topk_method.as_deref() != Some("noaux_tc")
        || mimo.scoring_func.as_deref() != Some("sigmoid")
        || mimo.norm_topk_prob != Some(true)
        || mimo.routed_scaling_factor.is_some_and(|scale| scale != 1.0)
        || mimo.moe_router_dtype.as_deref() != Some("bfloat16")
        || config.hidden_act.as_deref() != Some("silu")
        || config.m3.is_some()
        || plan.expert_count as usize != EXPERTS
        || plan.experts_per_token as usize != TOP_K
        || plan.expert_intermediate_size as usize != EXPERT_WIDTH
        || plan.shared.is_some()
        || plan.activation != ActivationPlan::Silu
        || !matches!(
            plan.router,
            RouterPlan::Sigmoid {
                normalize_selected: true,
                scaling_factor: 1.0,
                selection_bias: true
            }
        )
    {
        return Err("unsupported MiMo source MoE contract".into());
    }
    let compiled = ModelPlan::compile(config)?;
    let Some(layer_plan) = compiled.layers.get(layer) else {
        return Err("MiMo source MoE layer is out of range".into());
    };
    if !matches!(&layer_plan.mlp, MlpPlan::Moe(actual) if actual == plan) {
        return Err("MiMo source MoE plan does not match the requested layer".into());
    }
    Ok(())
}

fn float_tensor(
    dtype: &str,
    shape: &[u64],
    bytes: &[u8],
    expected: &[u64],
) -> Result<Vec<f32>, Fail> {
    if shape != expected {
        return Err(format!("router shape {shape:?} != {expected:?}").into());
    }
    let element_bytes = match dtype {
        "BF16" => 2,
        "F32" => 4,
        _ => return Err(format!("unsupported router dtype {dtype}").into()),
    };
    let count = expected.iter().product::<u64>() as usize;
    if bytes.len() != count * element_bytes {
        return Err("router byte extent does not match dtype and shape".into());
    }
    let values: Vec<f32> = match dtype {
        "BF16" => bytes
            .chunks_exact(2)
            .map(|part| f32::from_bits(u32::from(u16::from_le_bytes([part[0], part[1]])) << 16))
            .collect(),
        "F32" => bytes
            .chunks_exact(4)
            .map(|part| f32::from_le_bytes([part[0], part[1], part[2], part[3]]))
            .collect(),
        _ => unreachable!(),
    };
    if values.iter().any(|value| !value.is_finite()) {
        return Err("router contains a non-finite value".into());
    }
    Ok(values)
}

fn source_float(source: &StModel, name: &str, shape: &[u64]) -> Result<Vec<f32>, Fail> {
    let (info, bytes) = source
        .raw(name)
        .ok_or_else(|| format!("missing MiMo router tensor {name}"))?;
    float_tensor(&info.dtype, &info.shape, bytes, shape)
        .map_err(|error| format!("{name}: {error}").into())
}

fn source_mxfp4<'a>(
    source: &'a StModel,
    name: &str,
    rows: usize,
    cols: usize,
) -> Result<(&'a [u8], &'a [u8]), Fail> {
    let weight_name = format!("{name}.weight");
    let scale_name = format!("{name}.weight_scale");
    let (weight_info, weight) = source
        .raw(&weight_name)
        .ok_or_else(|| format!("missing {weight_name}"))?;
    let (scale_info, scales) = source
        .raw(&scale_name)
        .ok_or_else(|| format!("missing {scale_name}"))?;
    validate_mxfp4(
        &weight_info.dtype,
        &weight_info.shape,
        weight,
        &scale_info.dtype,
        &scale_info.shape,
        scales,
        rows,
        cols,
    )
    .map_err(|error| -> Fail { format!("{name}: {error}").into() })?;
    Ok((weight, scales))
}

#[allow(clippy::too_many_arguments)]
fn validate_mxfp4(
    weight_dtype: &str,
    weight_shape: &[u64],
    weight: &[u8],
    scale_dtype: &str,
    scale_shape: &[u64],
    scales: &[u8],
    rows: usize,
    cols: usize,
) -> Result<(), Fail> {
    if weight_dtype != "U8"
        || weight_shape != [rows as u64, (cols / 2) as u64]
        || weight.len() != rows * cols / 2
        || scale_dtype != "U8"
        || scale_shape != [rows as u64, (cols / 32) as u64]
        || scales.len() != rows * cols / 32
        || scales.contains(&0xff)
    {
        return Err("changed MXFP4 weight or E8M0 scale grid".into());
    }
    Ok(())
}

fn quantize(
    engine: &Engine,
    input: &CudaSlice<f32>,
    width: usize,
) -> Result<(CudaSlice<u8>, CudaSlice<f32>), Fail> {
    let stream = engine.stream();
    let mut codes = stream.alloc_zeros::<u8>(width)?;
    let mut scales = stream.alloc_zeros::<f32>(width / 128)?;
    let (input_ptr, _input_guard) = input.device_ptr(&stream);
    let (codes_ptr, _codes_guard) = codes.device_ptr_mut(&stream);
    let (scales_ptr, _scales_guard) = scales.device_ptr_mut(&stream);
    let rc = unsafe {
        memra_dsv4_act_quant_fp8(
            input_ptr as *const f32,
            codes_ptr as *mut c_void,
            scales_ptr as *mut f32,
            1,
            width as i32,
            stream.cu_stream() as *mut c_void,
        )
    };
    drop((_input_guard, _codes_guard, _scales_guard));
    if rc != 0 {
        return Err(format!("MiMo activation FP8 quantization failed: {rc}").into());
    }
    Ok((codes, scales))
}

#[allow(clippy::too_many_arguments)]
fn project(
    engine: &Engine,
    source: &StModel,
    stem: &str,
    codes: &CudaSlice<u8>,
    activation_scales: &CudaSlice<f32>,
    rows: usize,
    cols: usize,
) -> Result<CudaSlice<f32>, Fail> {
    let (weight, weight_scales) = source_mxfp4(source, stem, rows, cols)?;
    let weight_gpu = engine.htod_bytes(weight)?;
    let weight_scales_gpu = engine.htod_bytes(weight_scales)?;
    let stream = engine.stream();
    let mut output = stream.alloc_zeros::<f32>(rows)?;
    let (codes_ptr, _codes_guard) = codes.device_ptr(&stream);
    let (activation_scales_ptr, _activation_scales_guard) = activation_scales.device_ptr(&stream);
    let (weight_ptr, _weight_guard) = weight_gpu.device_ptr(&stream);
    let (weight_scales_ptr, _weight_scales_guard) = weight_scales_gpu.device_ptr(&stream);
    let (output_ptr, _output_guard) = output.device_ptr_mut(&stream);
    let rc = unsafe {
        memra_dsv4_fp4_gemm(
            codes_ptr as *const c_void,
            activation_scales_ptr as *const f32,
            weight_ptr as *const c_void,
            weight_scales_ptr as *const c_void,
            0.0,
            1,
            output_ptr as *mut f32,
            1,
            rows as i32,
            cols as i32,
            stream.cu_stream() as *mut c_void,
        )
    };
    drop((
        _codes_guard,
        _activation_scales_guard,
        _weight_guard,
        _weight_scales_guard,
        _output_guard,
    ));
    if rc != 0 {
        return Err(format!("{stem}: MXFP4 GEMM failed: {rc}").into());
    }
    Ok(output)
}

fn validate_selection(ids: &[i32], weights: &[f32]) -> Result<Vec<u32>, Fail> {
    if ids.len() != TOP_K
        || weights.len() != TOP_K
        || weights
            .iter()
            .any(|weight| !weight.is_finite() || *weight < 0.0)
        || (weights.iter().sum::<f32>() - 1.0).abs() > 1e-4
    {
        return Err("MiMo router returned invalid top-8 weights".into());
    }
    let mut seen = [false; EXPERTS];
    ids.iter()
        .map(|&id| {
            let index = usize::try_from(id).map_err(|_| "MiMo router returned negative expert")?;
            let slot = seen
                .get_mut(index)
                .ok_or("MiMo router returned expert out of range")?;
            if *slot {
                return Err("MiMo router returned duplicate expert");
            }
            *slot = true;
            Ok(id as u32)
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(Into::into)
}

/// Evaluate one pre-MLP-normalized 4096-element token. The output is the
/// weighted routed-expert sum before the layer's residual addition.
pub fn source_moe_token(
    engine: &Engine,
    source: &StModel,
    layer: usize,
    x: &CudaSlice<f32>,
    config: &ModelConfig,
    plan: &MoeMlpPlan,
) -> Result<MiMoMoeToken, Box<dyn Error>> {
    validate_contract(config, plan, layer)?;
    if x.len() != HIDDEN || x.ordinal() != engine.stream().context().ordinal() {
        return Err("MiMo source MoE input must be one 4096-vector on the engine device".into());
    }
    // The exact source header digest binds the full official MXFP4 checkpoint
    // structure. Payload provenance and numerical parity remain caller gates.
    let headers = source
        .names()
        .map(|name| {
            source
                .info(name)
                .map(|info| (name.clone(), info.clone()))
                .ok_or_else(|| format!("missing MiMo source header {name}"))
        })
        .collect::<Result<BTreeMap<_, _>, _>>()?;
    inspect_pinned_source_headers(config, &headers)?;
    engine.gpu.ctx.bind_to_thread()?;
    let router = format!("model.layers.{layer}.mlp.gate");
    let matrix = source_float(source, &format!("{router}.weight"), &[256, 4096])?;
    let bias = source_float(source, &format!("{router}.e_score_correction_bias"), &[256])?;
    let matrix_gpu = engine.htod(&matrix)?;
    let bias_gpu = engine.htod(&bias)?;
    let active_gpu = engine.htod_bytes(&[1u8; EXPERTS])?;
    let logits = engine.linear(x, &matrix_gpu, 1, HIDDEN, EXPERTS)?;
    let (ids_gpu, weights_gpu) = engine.moe_router_sigmoid_topk(
        &logits,
        1,
        EXPERTS,
        TOP_K,
        EXPERTS,
        &bias_gpu,
        &active_gpu,
        1.0,
        true,
    )?;
    let ids = engine.dtoh_i32(&ids_gpu)?;
    let weights = engine.dtoh(&weights_gpu)?;
    let selected = validate_selection(&ids, &weights)?;

    let (input_codes, input_scales) = quantize(engine, x, HIDDEN)?;
    let mut output = engine.zeros(HIDDEN)?;
    for (&expert, &weight) in selected.iter().zip(&weights) {
        let stem = format!("model.layers.{layer}.mlp.experts.{expert}");
        let gate = project(
            engine,
            source,
            &format!("{stem}.gate_proj"),
            &input_codes,
            &input_scales,
            EXPERT_WIDTH,
            HIDDEN,
        )?;
        let up = project(
            engine,
            source,
            &format!("{stem}.up_proj"),
            &input_codes,
            &input_scales,
            EXPERT_WIDTH,
            HIDDEN,
        )?;
        let mut activation = engine.uninit(EXPERT_WIDTH)?;
        engine.silu_mul(&gate, &up, &mut activation, EXPERT_WIDTH)?;
        let (activation_codes, activation_scales) = quantize(engine, &activation, EXPERT_WIDTH)?;
        let down = project(
            engine,
            source,
            &format!("{stem}.down_proj"),
            &activation_codes,
            &activation_scales,
            HIDDEN,
            EXPERT_WIDTH,
        )?;
        engine.axpy_into(&down, weight, &mut output.slice_mut(..), HIDDEN)?;
    }
    Ok(MiMoMoeToken {
        output,
        selected,
        weights,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use memra_gguf::config::HfConfig;

    #[test]
    fn source_contract_rejects_drift_and_wrong_layer_plan() {
        let fixture = include_str!("../../memra-gguf/src/model_packs/mimo_v2/fixtures/config.json");
        let config = ModelConfig::from_hf(&HfConfig::parse(fixture));
        let compiled = ModelPlan::compile(&config).unwrap();
        let MlpPlan::Moe(plan) = &compiled.layers[1].mlp else {
            panic!("fixture layer 1 must be MoE");
        };
        validate_contract(&config, plan, 1).unwrap();
        assert!(validate_contract(&config, plan, 0).is_err());
        let mut changed = config.clone();
        changed.mimo.as_mut().unwrap().topk_group = Some(2);
        assert!(validate_contract(&changed, plan, 1).is_err());
        let mut changed = config.clone();
        changed.hidden_act = Some("gelu".into());
        assert!(validate_contract(&changed, plan, 1).is_err());
        let mut changed_plan = plan.clone();
        changed_plan.shared = Some(memra_gguf::model_plan::SharedMlpPlan {
            intermediate_size: 2048,
            gated: false,
        });
        assert!(validate_contract(&config, &changed_plan, 1).is_err());
    }

    #[test]
    fn source_tensor_extents_and_scale_codes_fail_closed() {
        let bf16 = [0x80, 0x3f, 0x00, 0x40];
        assert_eq!(float_tensor("BF16", &[2], &bf16, &[2]).unwrap(), [1.0, 2.0]);
        assert!(float_tensor("BF16", &[2], &bf16[..2], &[2]).is_err());
        assert!(float_tensor("BF16", &[1, 2], &bf16, &[2]).is_err());
        assert!(float_tensor("U8", &[2], &bf16, &[2]).is_err());
        assert!(float_tensor("F32", &[1], &f32::NAN.to_le_bytes(), &[1]).is_err());

        let weight = [0u8; 32];
        let scale = [1u8; 2];
        validate_mxfp4("U8", &[1, 32], &weight, "U8", &[1, 2], &scale, 1, 64).unwrap();
        assert!(
            validate_mxfp4("U8", &[1, 32], &weight[..31], "U8", &[1, 2], &scale, 1, 64).is_err()
        );
        assert!(validate_mxfp4("U8", &[1, 32], &weight, "U8", &[1, 1], &scale, 1, 64).is_err());
        assert!(validate_mxfp4("U8", &[1, 32], &weight, "U8", &[1, 2], &[1, 255], 1, 64).is_err());
    }

    #[test]
    fn router_ids_must_be_unique_and_in_range() {
        let ids = [0, 255, 2, 3, 4, 5, 6, 7];
        let weights = [0.125; TOP_K];
        assert_eq!(
            validate_selection(&ids, &weights).unwrap(),
            ids.map(|id| id as u32)
        );
        let mut bad = ids;
        bad[7] = -1;
        assert!(validate_selection(&bad, &weights).is_err());
        bad[7] = 256;
        assert!(validate_selection(&bad, &weights).is_err());
        bad[7] = 0;
        assert!(validate_selection(&bad, &weights).is_err());
        let mut bad_weights = weights;
        bad_weights[0] = f32::NAN;
        assert!(validate_selection(&ids, &bad_weights).is_err());
        bad_weights = weights;
        bad_weights[0] = 0.0;
        assert!(validate_selection(&ids, &bad_weights).is_err());
    }
}
