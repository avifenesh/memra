//! One-token, source-MXFP4 MiMo MoE diagnostics. The streamed path reads
//! selected experts per call; the resident path holds all experts on one GPU.
//! Neither path is a serving admission gate.

use std::collections::BTreeMap;
use std::error::Error;
use std::ffi::c_void;

use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use memra_gguf::config::{Arch, ModelConfig};
use memra_gguf::model_packs::mimo_v2::inspect_pinned_source_headers;
use memra_gguf::model_plan::{ActivationPlan, MlpPlan, ModelPlan, MoeMlpPlan, RouterPlan};
use memra_gguf::safetensors::StModel;

use crate::Engine;
use crate::dsv4_ffi::{memra_dsv4_act_quant_fp8, memra_dsv4_fp4_gemm, memra_dsv4_fp4_gemm_sel};

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

/// A source whose complete pinned header census and model plan were checked
/// once before any GPU work. Its private fields prevent bypassing that check.
pub struct PinnedMiMoSource<'a> {
    model: &'a StModel,
    config: &'a ModelConfig,
    plan: &'a ModelPlan,
    semantic_tensors: usize,
}

impl<'a> PinnedMiMoSource<'a> {
    pub fn bind(
        model: &'a StModel,
        config: &'a ModelConfig,
        plan: &'a ModelPlan,
    ) -> Result<Self, Fail> {
        if &ModelPlan::compile(config)? != plan {
            return Err("MiMo source plan does not match its configuration".into());
        }
        let headers = model
            .names()
            .map(|name| {
                model
                    .info(name)
                    .map(|info| (name.clone(), info.clone()))
                    .ok_or_else(|| format!("missing MiMo source header {name}"))
            })
            .collect::<Result<BTreeMap<_, _>, _>>()?;
        let bound = inspect_pinned_source_headers(config, &headers)?;
        Ok(Self {
            model,
            config,
            plan,
            semantic_tensors: bound.tensors.len(),
        })
    }

    pub fn semantic_tensors(&self) -> usize {
        self.semantic_tensors
    }
}

fn validate_contract(
    config: &ModelConfig,
    compiled: &ModelPlan,
    plan: &MoeMlpPlan,
    layer: usize,
) -> Result<(), Fail> {
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
    quantize_rows(engine, input, width, 1)
}

fn quantize_rows(
    engine: &Engine,
    input: &CudaSlice<f32>,
    width: usize,
    rows: usize,
) -> Result<(CudaSlice<u8>, CudaSlice<f32>), Fail> {
    if rows == 0
        || !width.is_multiple_of(128)
        || input.len() != rows * width
        || input.ordinal() != engine.stream().context().ordinal()
    {
        return Err("MiMo FP8 activation row geometry changed".into());
    }
    let stream = engine.stream();
    let mut codes = stream.alloc_zeros::<u8>(rows * width)?;
    let mut scales = stream.alloc_zeros::<f32>(rows * width / 128)?;
    let (input_ptr, _input_guard) = input.device_ptr(&stream);
    let (codes_ptr, _codes_guard) = codes.device_ptr_mut(&stream);
    let (scales_ptr, _scales_guard) = scales.device_ptr_mut(&stream);
    let rc = unsafe {
        memra_dsv4_act_quant_fp8(
            input_ptr as *const f32,
            codes_ptr as *mut c_void,
            scales_ptr as *mut f32,
            rows as i32,
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

struct ResidentProjection {
    weight: CudaSlice<u8>,
    scales: CudaSlice<u8>,
    weight_stride: usize,
    scale_stride: usize,
    rows: usize,
    cols: usize,
}

impl ResidentProjection {
    fn load(
        engine: &Engine,
        source: &StModel,
        layer: usize,
        projection: &str,
        rows: usize,
        cols: usize,
    ) -> Result<Self, Fail> {
        let weight_stride = rows * cols / 2;
        let scale_stride = rows * cols / 32;
        let mut weight = engine.alloc_u8_uninit(EXPERTS * weight_stride)?;
        let mut scales = engine.alloc_u8_uninit(EXPERTS * scale_stride)?;
        let stream = engine.stream();
        for expert in 0..EXPERTS {
            let stem = format!("model.layers.{layer}.mlp.experts.{expert}.{projection}");
            let (source_weight, source_scales) = source_mxfp4(source, &stem, rows, cols)?;
            stream.memcpy_htod(
                source_weight,
                &mut weight.slice_mut(expert * weight_stride..(expert + 1) * weight_stride),
            )?;
            stream.memcpy_htod(
                source_scales,
                &mut scales.slice_mut(expert * scale_stride..(expert + 1) * scale_stride),
            )?;
        }
        Ok(Self {
            weight,
            scales,
            weight_stride,
            scale_stride,
            rows,
            cols,
        })
    }

    fn run(
        &self,
        engine: &Engine,
        expert: u32,
        codes: &CudaSlice<u8>,
        activation_scales: &CudaSlice<f32>,
    ) -> Result<CudaSlice<f32>, Fail> {
        let expert = expert as usize;
        if expert >= EXPERTS
            || codes.len() != self.cols
            || activation_scales.len() != self.cols / 128
            || self.weight.ordinal() != engine.stream().context().ordinal()
            || self.scales.ordinal() != engine.stream().context().ordinal()
        {
            return Err("MiMo resident projection shape or device changed".into());
        }
        let stream = engine.stream();
        let weight_view = self
            .weight
            .slice(expert * self.weight_stride..(expert + 1) * self.weight_stride);
        let scales_view = self
            .scales
            .slice(expert * self.scale_stride..(expert + 1) * self.scale_stride);
        let mut output = stream.alloc_zeros::<f32>(self.rows)?;
        let (codes_ptr, codes_guard) = codes.device_ptr(&stream);
        let (activation_scales_ptr, activation_scales_guard) =
            activation_scales.device_ptr(&stream);
        let (weight_ptr, weight_guard) = weight_view.device_ptr(&stream);
        let (weight_scales_ptr, weight_scales_guard) = scales_view.device_ptr(&stream);
        let (output_ptr, output_guard) = output.device_ptr_mut(&stream);
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
                self.rows as i32,
                self.cols as i32,
                stream.cu_stream() as *mut c_void,
            )
        };
        drop((
            codes_guard,
            activation_scales_guard,
            weight_guard,
            weight_scales_guard,
            output_guard,
        ));
        if rc != 0 {
            return Err(format!("MiMo resident MXFP4 GEMM failed: {rc}").into());
        }
        Ok(output)
    }
}

/// The original source codes and E8M0 scales in [expert][gate,down,up]
/// order. All three projections have equal byte and scale strides.
struct InterleavedExperts {
    weight: CudaSlice<u8>,
    scales: CudaSlice<u8>,
    scale2_dummy: CudaSlice<f32>,
    weight_stride: usize,
    scale_stride: usize,
}

impl InterleavedExperts {
    fn load(engine: &Engine, source: &StModel, layer: usize) -> Result<Self, Fail> {
        let weight_stride = EXPERT_WIDTH * HIDDEN / 2;
        let scale_stride = EXPERT_WIDTH * HIDDEN / 32;
        let mut weight = engine.alloc_u8_uninit(EXPERTS * 3 * weight_stride)?;
        let mut scales = engine.alloc_u8_uninit(EXPERTS * 3 * scale_stride)?;
        let stream = engine.stream();
        for expert in 0..EXPERTS {
            for (projection, name, rows, cols) in [
                (0, "gate_proj", EXPERT_WIDTH, HIDDEN),
                (1, "down_proj", HIDDEN, EXPERT_WIDTH),
                (2, "up_proj", EXPERT_WIDTH, HIDDEN),
            ] {
                let stem = format!("model.layers.{layer}.mlp.experts.{expert}.{name}");
                let (source_weight, source_scales) = source_mxfp4(source, &stem, rows, cols)?;
                if source_weight.len() != weight_stride || source_scales.len() != scale_stride {
                    return Err(format!("{stem}: interleaved stride changed").into());
                }
                let index = expert * 3 + projection;
                stream.memcpy_htod(
                    source_weight,
                    &mut weight.slice_mut(index * weight_stride..(index + 1) * weight_stride),
                )?;
                stream.memcpy_htod(
                    source_scales,
                    &mut scales.slice_mut(index * scale_stride..(index + 1) * scale_stride),
                )?;
            }
        }
        let scale2_dummy = engine.htod(&[0.0f32; EXPERTS * 3])?;
        Ok(Self {
            weight,
            scales,
            scale2_dummy,
            weight_stride,
            scale_stride,
        })
    }

    fn project(
        &self,
        engine: &Engine,
        codes: &CudaSlice<u8>,
        activation_scales: &CudaSlice<f32>,
        selected: &CudaSlice<i32>,
        shape: (i32, usize, usize, bool),
    ) -> Result<CudaSlice<f32>, Fail> {
        let (projection, rows, cols, per_slot) = shape;
        let expected = if per_slot { TOP_K } else { 1 };
        let ordinal = engine.stream().context().ordinal();
        if !(0..=2).contains(&projection)
            || rows * cols / 2 != self.weight_stride
            || rows * cols / 32 != self.scale_stride
            || codes.len() != expected * cols
            || activation_scales.len() != expected * cols / 128
            || selected.len() != TOP_K
            || [
                codes.ordinal(),
                activation_scales.ordinal(),
                selected.ordinal(),
                self.weight.ordinal(),
                self.scales.ordinal(),
                self.scale2_dummy.ordinal(),
            ]
            .into_iter()
            .any(|device| device != ordinal)
        {
            return Err("MiMo grouped expert projection shape or GPU changed".into());
        }
        let stream = engine.stream();
        let mut output = stream.alloc_zeros::<f32>(TOP_K * rows)?;
        let (codes_ptr, codes_guard) = codes.device_ptr(&stream);
        let (activation_scales_ptr, activation_scales_guard) =
            activation_scales.device_ptr(&stream);
        let (weight_ptr, weight_guard) = self.weight.device_ptr(&stream);
        let (weight_scales_ptr, weight_scales_guard) = self.scales.device_ptr(&stream);
        let (scale2_ptr, scale2_guard) = self.scale2_dummy.device_ptr(&stream);
        let (selected_ptr, selected_guard) = selected.device_ptr(&stream);
        let (output_ptr, output_guard) = output.device_ptr_mut(&stream);
        let rc = unsafe {
            memra_dsv4_fp4_gemm_sel(
                codes_ptr as *const c_void,
                activation_scales_ptr as *const f32,
                weight_ptr as *const c_void,
                weight_scales_ptr as *const c_void,
                scale2_ptr as *const f32,
                selected_ptr as *const i32,
                projection,
                i32::from(per_slot),
                1,
                output_ptr as *mut f32,
                TOP_K as i32,
                rows as i32,
                cols as i32,
                self.weight_stride as i64,
                self.scale_stride as i64,
                stream.cu_stream() as *mut c_void,
            )
        };
        drop((
            codes_guard,
            activation_scales_guard,
            weight_guard,
            weight_scales_guard,
            scale2_guard,
            selected_guard,
            output_guard,
        ));
        if rc != 0 {
            return Err(format!("MiMo selected MXFP4 GEMM failed: {rc}").into());
        }
        Ok(output)
    }

    fn resident_bytes(&self) -> usize {
        self.weight.len() + self.scales.len() + self.scale2_dummy.len() * size_of::<f32>()
    }
}

/// The pinned source's full expert bank for one MoE layer on one device.
/// Projection slabs use the original E2M1 codes and E8M0 scales byte-for-byte.
pub struct ResidentMiMoMoeLayer {
    layer: usize,
    matrix: CudaSlice<f32>,
    bias: CudaSlice<f32>,
    active: CudaSlice<u8>,
    gate: ResidentProjection,
    up: ResidentProjection,
    down: ResidentProjection,
}

impl ResidentMiMoMoeLayer {
    pub fn load(
        engine: &Engine,
        pinned: &PinnedMiMoSource<'_>,
        layer: usize,
        plan: &MoeMlpPlan,
    ) -> Result<Self, Fail> {
        validate_contract(pinned.config, pinned.plan, plan, layer)?;
        engine.gpu.ctx.bind_to_thread()?;
        let source = pinned.model;
        let router = format!("model.layers.{layer}.mlp.gate");
        let matrix = engine.htod(&source_float(
            source,
            &format!("{router}.weight"),
            &[EXPERTS as u64, HIDDEN as u64],
        )?)?;
        let bias = engine.htod(&source_float(
            source,
            &format!("{router}.e_score_correction_bias"),
            &[EXPERTS as u64],
        )?)?;
        let active = engine.htod_bytes(&[1u8; EXPERTS])?;
        let gate =
            ResidentProjection::load(engine, source, layer, "gate_proj", EXPERT_WIDTH, HIDDEN)?;
        let up = ResidentProjection::load(engine, source, layer, "up_proj", EXPERT_WIDTH, HIDDEN)?;
        let down =
            ResidentProjection::load(engine, source, layer, "down_proj", HIDDEN, EXPERT_WIDTH)?;
        Ok(Self {
            layer,
            matrix,
            bias,
            active,
            gate,
            up,
            down,
        })
    }

    pub fn token(
        &self,
        engine: &Engine,
        x: &CudaSlice<f32>,
        plan: &MoeMlpPlan,
        pinned: &PinnedMiMoSource<'_>,
    ) -> Result<MiMoMoeToken, Fail> {
        validate_contract(pinned.config, pinned.plan, plan, self.layer)?;
        let device = engine.stream().context().ordinal();
        if x.len() != HIDDEN
            || x.ordinal() != device
            || self.matrix.ordinal() != device
            || self.bias.ordinal() != device
            || self.active.ordinal() != device
        {
            return Err("MiMo resident MoE input or layer belongs to another device".into());
        }
        engine.gpu.ctx.bind_to_thread()?;
        let logits = engine.linear(x, &self.matrix, 1, HIDDEN, EXPERTS)?;
        let (ids_gpu, weights_gpu) = engine.moe_router_sigmoid_topk(
            &logits,
            1,
            EXPERTS,
            TOP_K,
            EXPERTS,
            &self.bias,
            &self.active,
            1.0,
            true,
        )?;
        let ids = engine.dtoh_i32(&ids_gpu)?;
        let weights = engine.dtoh(&weights_gpu)?;
        let selected = validate_selection(&ids, &weights)?;
        let (input_codes, input_scales) = quantize(engine, x, HIDDEN)?;
        let mut output = engine.zeros(HIDDEN)?;
        for (&expert, &weight) in selected.iter().zip(&weights) {
            let gate = self.gate.run(engine, expert, &input_codes, &input_scales)?;
            let up = self.up.run(engine, expert, &input_codes, &input_scales)?;
            let mut activation = engine.uninit(EXPERT_WIDTH)?;
            engine.silu_mul(&gate, &up, &mut activation, EXPERT_WIDTH)?;
            let (activation_codes, activation_scales) =
                quantize(engine, &activation, EXPERT_WIDTH)?;
            let down = self
                .down
                .run(engine, expert, &activation_codes, &activation_scales)?;
            engine.axpy_into(&down, weight, &mut output.slice_mut(..), HIDDEN)?;
        }
        Ok(MiMoMoeToken {
            output,
            selected,
            weights,
        })
    }

    pub fn resident_bytes(&self) -> usize {
        self.matrix.len() * size_of::<f32>()
            + self.bias.len() * size_of::<f32>()
            + self.active.len()
            + [&self.gate, &self.up, &self.down]
                .iter()
                .map(|projection| projection.weight.len() + projection.scales.len())
                .sum::<usize>()
    }
}

/// Source MXFP4 experts laid out for Memra's existing selected-expert GPU
/// projection. Router and accumulation order match the streamed control.
pub struct GroupedMiMoMoeLayer {
    layer: usize,
    matrix: CudaSlice<f32>,
    bias: CudaSlice<f32>,
    active: CudaSlice<u8>,
    experts: InterleavedExperts,
}

impl GroupedMiMoMoeLayer {
    pub fn load(
        engine: &Engine,
        pinned: &PinnedMiMoSource<'_>,
        layer: usize,
        plan: &MoeMlpPlan,
    ) -> Result<Self, Fail> {
        validate_contract(pinned.config, pinned.plan, plan, layer)?;
        engine.gpu.ctx.bind_to_thread()?;
        let source = pinned.model;
        let router = format!("model.layers.{layer}.mlp.gate");
        let matrix = engine.htod(&source_float(
            source,
            &format!("{router}.weight"),
            &[EXPERTS as u64, HIDDEN as u64],
        )?)?;
        let bias = engine.htod(&source_float(
            source,
            &format!("{router}.e_score_correction_bias"),
            &[EXPERTS as u64],
        )?)?;
        let active = engine.htod_bytes(&[1u8; EXPERTS])?;
        let experts = InterleavedExperts::load(engine, source, layer)?;
        Ok(Self {
            layer,
            matrix,
            bias,
            active,
            experts,
        })
    }

    pub fn token(
        &self,
        engine: &Engine,
        x: &CudaSlice<f32>,
        plan: &MoeMlpPlan,
        pinned: &PinnedMiMoSource<'_>,
    ) -> Result<MiMoMoeToken, Fail> {
        validate_contract(pinned.config, pinned.plan, plan, self.layer)?;
        let ordinal = engine.stream().context().ordinal();
        if x.len() != HIDDEN
            || [
                x.ordinal(),
                self.matrix.ordinal(),
                self.bias.ordinal(),
                self.active.ordinal(),
            ]
            .into_iter()
            .any(|device| device != ordinal)
        {
            return Err("MiMo grouped MoE input or layer belongs to another GPU".into());
        }
        engine.gpu.ctx.bind_to_thread()?;
        let logits = engine.linear(x, &self.matrix, 1, HIDDEN, EXPERTS)?;
        let (ids_gpu, weights_gpu) = engine.moe_router_sigmoid_topk(
            &logits,
            1,
            EXPERTS,
            TOP_K,
            EXPERTS,
            &self.bias,
            &self.active,
            1.0,
            true,
        )?;
        let ids = engine.dtoh_i32(&ids_gpu)?;
        let weights = engine.dtoh(&weights_gpu)?;
        let selected = validate_selection(&ids, &weights)?;

        let (codes, scales) = quantize(engine, x, HIDDEN)?;
        let gate = self.experts.project(
            engine,
            &codes,
            &scales,
            &ids_gpu,
            (0, EXPERT_WIDTH, HIDDEN, false),
        )?;
        let up = self.experts.project(
            engine,
            &codes,
            &scales,
            &ids_gpu,
            (2, EXPERT_WIDTH, HIDDEN, false),
        )?;
        let mut activated = engine.uninit(TOP_K * EXPERT_WIDTH)?;
        engine.silu_mul(&gate, &up, &mut activated, TOP_K * EXPERT_WIDTH)?;
        let (activation_codes, activation_scales) =
            quantize_rows(engine, &activated, EXPERT_WIDTH, TOP_K)?;
        let down = self.experts.project(
            engine,
            &activation_codes,
            &activation_scales,
            &ids_gpu,
            (1, HIDDEN, EXPERT_WIDTH, true),
        )?;
        let mut output = engine.uninit(HIDDEN)?;
        engine.axpy_rows_seq_into(&down, &weights_gpu, &mut output, HIDDEN, TOP_K)?;
        Ok(MiMoMoeToken {
            output,
            selected,
            weights,
        })
    }

    pub fn resident_bytes(&self) -> usize {
        self.matrix.len() * size_of::<f32>()
            + self.bias.len() * size_of::<f32>()
            + self.active.len()
            + self.experts.resident_bytes()
    }
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
    pinned: &PinnedMiMoSource<'_>,
    layer: usize,
    x: &CudaSlice<f32>,
    plan: &MoeMlpPlan,
) -> Result<MiMoMoeToken, Box<dyn Error>> {
    validate_contract(pinned.config, pinned.plan, plan, layer)?;
    if x.len() != HIDDEN || x.ordinal() != engine.stream().context().ordinal() {
        return Err("MiMo source MoE input must be one 4096-vector on the engine device".into());
    }
    // `PinnedMiMoSource::bind` checked the full source header digest. Payload
    // provenance and numerical parity remain separate caller gates.
    let source = pinned.model;
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
        validate_contract(&config, &compiled, plan, 1).unwrap();
        assert!(validate_contract(&config, &compiled, plan, 0).is_err());
        let mut changed = config.clone();
        changed.mimo.as_mut().unwrap().topk_group = Some(2);
        assert!(validate_contract(&changed, &compiled, plan, 1).is_err());
        let mut changed = config.clone();
        changed.hidden_act = Some("gelu".into());
        assert!(validate_contract(&changed, &compiled, plan, 1).is_err());
        let mut changed_plan = plan.clone();
        changed_plan.shared = Some(memra_gguf::model_plan::SharedMlpPlan {
            intermediate_size: 2048,
            gated: false,
        });
        assert!(validate_contract(&config, &compiled, &changed_plan, 1).is_err());
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
