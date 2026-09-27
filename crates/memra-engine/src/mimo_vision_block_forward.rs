//! One pinned MiMo ViT block over projected, BF16-valued patch rows.
//! The input and output use row order. Column blocks reorder intact 2x2
//! merge units around attention, as the publisher's tower does.

use core::ffi::c_void;
use std::error::Error;

use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use memra_gguf::config::ModelConfig;
use memra_gguf::model_packs::mimo_v2::vision::{
    MiMoPatchOrder, MiMoVisionAttentionPlan, MiMoVisionGrid, MiMoVisionLayout,
    pinned_attention_plan, pinned_vision_layout,
};
use memra_reference::mimo_vision_rope::ordered_positions;

use crate::Engine;
use crate::mimo_vision_load::{MiMoVisionBlock, MiMoVisionWeights};
use crate::model::GpuTensor;

type Fail = Box<dyn Error>;

const HIDDEN: usize = 1_280;
const Q: usize = 32 * 64;
const KV: usize = 8 * 64;
const FUSED_QKV: usize = Q + 2 * KV;
const FF: usize = 4_608;
// The pinned vision config omits rms_norm_eps; MiMoVisionTransformer defaults
// it to 1e-6 when constructing each nn.RMSNorm.
const RMS_EPSILON: f32 = 1e-6;

#[derive(Clone, Copy)]
#[repr(i32)]
enum Epilogue {
    Round = 0,
    Bias = 1,
    Residual = 2,
    SwiGlu = 3,
}

unsafe extern "C" {
    fn memra_mimo_vision_block_epilogue(
        a: *const f32,
        b: *const f32,
        output: *mut f32,
        elements: i32,
        width: i32,
        operation: i32,
        stream: *mut c_void,
    ) -> i32;
    fn memra_mimo_vision_block_split_qkv(
        fused: *const f32,
        query: *mut f32,
        key: *mut f32,
        value: *mut f32,
        patches: i32,
        stream: *mut c_void,
    ) -> i32;
    fn memra_mimo_vision_block_check_bf16(
        values: *const f32,
        elements: i32,
        fault: *mut i32,
        stream: *mut c_void,
    ) -> i32;
}

struct BlockRequest {
    layout: MiMoVisionLayout,
    lengths: Vec<usize>,
    patches: usize,
}

struct ProjectedQkv {
    query: CudaSlice<f32>,
    key: CudaSlice<f32>,
    value: CudaSlice<f32>,
}

fn request(
    config: &ModelConfig,
    plan: &MiMoVisionAttentionPlan,
    layer: usize,
    grids: &[MiMoVisionGrid],
    hidden_elements: usize,
) -> Result<BlockRequest, Fail> {
    let layer = u32::try_from(layer)?;
    if pinned_attention_plan(config, layer)? != *plan {
        return Err("MiMo vision block plan differs from pinned source".into());
    }
    let layout = pinned_vision_layout(config, grids)?;
    let patches = layout.row_positions.len();
    // This also checks the merge-unit permutation and the component's
    // 256-patch frame bound before any GPU work.
    ordered_positions(plan, &layout)?;
    if patches > 1_024
        || layout.frame_ends.len() > 32
        || patches.checked_mul(HIDDEN) != Some(hidden_elements)
    {
        return Err("MiMo vision block frame count or hidden extent changed".into());
    }
    let mut previous = 0usize;
    let mut lengths = Vec::with_capacity(layout.frame_ends.len());
    for &end in &layout.frame_ends {
        let end = end as usize;
        lengths.push(end - previous);
        previous = end;
    }
    Ok(BlockRequest {
        layout,
        lengths,
        patches,
    })
}

fn check_matrix(
    matrix: &GpuTensor,
    input: usize,
    output: usize,
    device: usize,
) -> Result<(), Fail> {
    if !matches!(matrix, GpuTensor::FloatBf16 { .. })
        || matrix.in_features() != input
        || matrix.out_features() != output
        || matrix.ordinal() != device
    {
        return Err("MiMo vision block matrix dtype, shape, or GPU changed".into());
    }
    Ok(())
}

fn check_vector(vector: &CudaSlice<f32>, width: usize, device: usize) -> Result<(), Fail> {
    if vector.len() != width || vector.ordinal() != device {
        return Err("MiMo vision block vector shape or GPU changed".into());
    }
    Ok(())
}

fn check_block(block: &MiMoVisionBlock, device: usize) -> Result<(), Fail> {
    check_vector(&block.norm1, HIDDEN, device)?;
    check_vector(&block.qkv_bias, FUSED_QKV, device)?;
    check_vector(&block.attention_output_bias, HIDDEN, device)?;
    check_vector(&block.norm2, HIDDEN, device)?;
    check_vector(&block.mlp_gate_bias, FF, device)?;
    check_vector(&block.mlp_up_bias, FF, device)?;
    check_vector(&block.mlp_down_bias, HIDDEN, device)?;
    if block.first_key_bias.is_some() != block.plan.sink_first_key {
        return Err("MiMo vision first-key bias presence changed".into());
    }
    if let Some(sink) = &block.first_key_bias {
        check_vector(sink, 32, device)?;
    }
    check_matrix(&block.qkv, HIDDEN, FUSED_QKV, device)?;
    check_matrix(&block.attention_output, Q, HIDDEN, device)?;
    check_matrix(&block.mlp_gate, HIDDEN, FF, device)?;
    check_matrix(&block.mlp_up, HIDDEN, FF, device)?;
    check_matrix(&block.mlp_down, FF, HIDDEN, device)?;
    Ok(())
}

fn epilogue(
    engine: &Engine,
    a: &CudaSlice<f32>,
    b: Option<&CudaSlice<f32>>,
    width: usize,
    operation: Epilogue,
) -> Result<CudaSlice<f32>, Fail> {
    engine.gpu.ctx.bind_to_thread()?;
    let stream = engine.stream();
    let device = stream.context().ordinal();
    if a.is_empty()
        || width == 0
        || !a.len().is_multiple_of(width)
        || a.len() > i32::MAX as usize
        || width > i32::MAX as usize
        || a.ordinal() != device
    {
        return Err("MiMo vision BF16 epilogue input extent or GPU changed".into());
    }
    let expected_b = match operation {
        Epilogue::Round => None,
        Epilogue::Bias => Some(width),
        Epilogue::Residual | Epilogue::SwiGlu => Some(a.len()),
    };
    if b.map(CudaSlice::len) != expected_b || b.is_some_and(|rhs| rhs.ordinal() != device) {
        return Err("MiMo vision BF16 epilogue second operand changed".into());
    }
    let mut output = engine.uninit(a.len())?;
    let (a_ptr, a_guard) = a.device_ptr(&stream);
    let (b_ptr, b_guard) = if let Some(rhs) = b {
        let (ptr, guard) = rhs.device_ptr(&stream);
        (ptr as *const f32, Some(guard))
    } else {
        (std::ptr::null(), None)
    };
    let (out_ptr, out_guard) = output.device_ptr_mut(&stream);
    let rc = unsafe {
        memra_mimo_vision_block_epilogue(
            a_ptr as *const f32,
            b_ptr,
            out_ptr as *mut f32,
            a.len() as i32,
            width as i32,
            operation as i32,
            stream.cu_stream() as *mut c_void,
        )
    };
    drop((a_guard, b_guard, out_guard));
    if rc != 0 {
        return Err(format!("MiMo vision BF16 epilogue CUDA refusal {rc}").into());
    }
    Ok(output)
}

fn normalized(
    engine: &Engine,
    input: &CudaSlice<f32>,
    weight: &CudaSlice<f32>,
    patches: usize,
) -> Result<CudaSlice<f32>, Fail> {
    let mut result = engine.uninit(patches * HIDDEN)?;
    engine.rms_norm(input, weight, &mut result, HIDDEN, patches, RMS_EPSILON)?;
    epilogue(engine, &result, None, HIDDEN, Epilogue::Round)
}

fn linear_bias(
    engine: &Engine,
    weight: &GpuTensor,
    bias: &CudaSlice<f32>,
    input: &CudaSlice<f32>,
    patches: usize,
) -> Result<CudaSlice<f32>, Fail> {
    let projected = engine.matmul(weight, input, patches)?;
    epilogue(
        engine,
        &projected,
        Some(bias),
        weight.out_features(),
        Epilogue::Bias,
    )
}

fn split_qkv(
    engine: &Engine,
    fused: &CudaSlice<f32>,
    patches: usize,
) -> Result<ProjectedQkv, Fail> {
    engine.gpu.ctx.bind_to_thread()?;
    let stream = engine.stream();
    if fused.len() != patches * FUSED_QKV || fused.ordinal() != stream.context().ordinal() {
        return Err("MiMo vision fused QKV extent or GPU changed".into());
    }
    let mut query = engine.uninit(patches * Q)?;
    let mut key = engine.uninit(patches * KV)?;
    let mut value = engine.uninit(patches * KV)?;
    let (fused_ptr, fused_guard) = fused.device_ptr(&stream);
    let (query_ptr, query_guard) = query.device_ptr_mut(&stream);
    let (key_ptr, key_guard) = key.device_ptr_mut(&stream);
    let (value_ptr, value_guard) = value.device_ptr_mut(&stream);
    let rc = unsafe {
        memra_mimo_vision_block_split_qkv(
            fused_ptr as *const f32,
            query_ptr as *mut f32,
            key_ptr as *mut f32,
            value_ptr as *mut f32,
            patches as i32,
            stream.cu_stream() as *mut c_void,
        )
    };
    drop((fused_guard, query_guard, key_guard, value_guard));
    if rc != 0 {
        return Err(format!("MiMo vision QKV split CUDA refusal {rc}").into());
    }
    Ok(ProjectedQkv { query, key, value })
}

fn check_bf16(engine: &Engine, values: &CudaSlice<f32>) -> Result<(), Fail> {
    engine.gpu.ctx.bind_to_thread()?;
    let stream = engine.stream();
    if values.is_empty()
        || values.len() > 1_024 * FF
        || values.ordinal() != stream.context().ordinal()
    {
        return Err("MiMo vision BF16 value extent or GPU changed".into());
    }
    let mut fault = engine.htod_i32(&[0])?;
    let (values_ptr, values_guard) = values.device_ptr(&stream);
    let (fault_ptr, fault_guard) = fault.device_ptr_mut(&stream);
    let rc = unsafe {
        memra_mimo_vision_block_check_bf16(
            values_ptr as *const f32,
            values.len() as i32,
            fault_ptr as *mut i32,
            stream.cu_stream() as *mut c_void,
        )
    };
    drop((values_guard, fault_guard));
    if rc != 0 {
        return Err(format!("MiMo vision BF16 check CUDA refusal {rc}").into());
    }
    let fault = engine.dtoh_i32(&fault)?[0];
    if fault != 0 {
        return Err(format!("MiMo vision value is non-finite or not BF16-valued ({fault})").into());
    }
    Ok(())
}

fn gather_indices(layout: &MiMoVisionLayout, reverse: bool) -> Vec<i32> {
    let groups = if reverse {
        &layout.reverse_column_groups
    } else {
        &layout.column_groups
    };
    groups
        .iter()
        .flat_map(|&group| (0..4).map(move |offset| (group * 4 + offset) as i32))
        .collect()
}

fn reorder(
    engine: &Engine,
    hidden: &CudaSlice<f32>,
    indices: &[i32],
) -> Result<CudaSlice<f32>, Fail> {
    let mut output = engine.uninit(hidden.len())?;
    let index = engine.htod_i32(indices)?;
    engine.gather_rows(hidden, &index, &mut output, HIDDEN, indices.len())?;
    Ok(output)
}

fn forward_block(
    block: &MiMoVisionBlock,
    engine: &Engine,
    input: &CudaSlice<f32>,
    request: &BlockRequest,
) -> Result<CudaSlice<f32>, Fail> {
    let device = engine.stream().context().ordinal();
    check_block(block, device)?;
    if input.ordinal() != device {
        return Err("MiMo vision hidden crossed GPU devices".into());
    }
    check_bf16(engine, input)?;
    let reordered = if block.plan.patch_order == MiMoPatchOrder::Column {
        Some(reorder(
            engine,
            input,
            &gather_indices(&request.layout, false),
        )?)
    } else {
        None
    };
    let hidden = reordered.as_ref().unwrap_or(input);
    let norm1 = normalized(engine, hidden, &block.norm1, request.patches)?;
    let qkv = linear_bias(engine, &block.qkv, &block.qkv_bias, &norm1, request.patches)?;
    let qkv = split_qkv(engine, &qkv, request.patches)?;
    let rotated =
        engine.mimo_vision_axial_rope(&block.plan, &request.layout, &qkv.query, &qkv.key)?;
    let attention = engine.mimo_vision_preprojected_attention(
        &block.plan,
        &rotated.query,
        &rotated.key,
        &qkv.value,
        &request.lengths,
        block.first_key_bias.as_ref(),
    )?;
    let attention = epilogue(engine, &attention, None, Q, Epilogue::Round)?;
    let projected = linear_bias(
        engine,
        &block.attention_output,
        &block.attention_output_bias,
        &attention,
        request.patches,
    )?;
    let after_attention = epilogue(engine, hidden, Some(&projected), HIDDEN, Epilogue::Residual)?;
    let norm2 = normalized(engine, &after_attention, &block.norm2, request.patches)?;
    let gate = linear_bias(
        engine,
        &block.mlp_gate,
        &block.mlp_gate_bias,
        &norm2,
        request.patches,
    )?;
    let up = linear_bias(
        engine,
        &block.mlp_up,
        &block.mlp_up_bias,
        &norm2,
        request.patches,
    )?;
    let activated = epilogue(engine, &gate, Some(&up), FF, Epilogue::SwiGlu)?;
    let mlp = linear_bias(
        engine,
        &block.mlp_down,
        &block.mlp_down_bias,
        &activated,
        request.patches,
    )?;
    let result = epilogue(
        engine,
        &after_attention,
        Some(&mlp),
        HIDDEN,
        Epilogue::Residual,
    )?;
    let result = if block.plan.patch_order == MiMoPatchOrder::Column {
        reorder(engine, &result, &gather_indices(&request.layout, true))?
    } else {
        result
    };
    check_bf16(engine, &result)?;
    Ok(result)
}

impl MiMoVisionWeights {
    /// Run exactly one of the 28 source ViT blocks on already projected
    /// BF16-valued `[patches,1280]` rows in source row order. The result is
    /// also row ordered. This does not run patchification, merger, or serving.
    pub fn forward_one_block(
        &self,
        engine: &Engine,
        config: &ModelConfig,
        layer: usize,
        grids: &[MiMoVisionGrid],
        hidden: &CudaSlice<f32>,
    ) -> Result<CudaSlice<f32>, Fail> {
        if self.blocks.len() != 28 {
            return Err("MiMo vision needs all 28 bound source blocks".into());
        }
        let block = self
            .blocks
            .get(layer)
            .ok_or("MiMo vision block index is out of range")?;
        let request = request(config, &block.plan, layer, grids, hidden.len())?;
        forward_block(block, engine, hidden, &request)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use memra_gguf::config::{HfConfig, ModelConfig};
    use memra_reference::mimo_vision::preprojected_attention;
    use memra_reference::mimo_vision_rope::rotate_qk;

    fn config() -> ModelConfig {
        ModelConfig::from_hf(&HfConfig::parse(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../memra-gguf/src/model_packs/mimo_v2/fixtures/config.json"
        ))))
    }

    #[test]
    fn pinned_rows_frames_and_block_plans_gate_component() {
        let config = config();
        let grids = [
            MiMoVisionGrid {
                frames: 2,
                height: 4,
                width: 6,
            },
            MiMoVisionGrid {
                frames: 1,
                height: 2,
                width: 2,
            },
        ];
        for layer in [0, 1, 5, 9, 27] {
            let plan = pinned_attention_plan(&config, layer).unwrap();
            let admitted = request(&config, &plan, layer as usize, &grids, 52 * HIDDEN).unwrap();
            assert_eq!(admitted.lengths, [24, 24, 4]);
            assert_eq!(admitted.patches, 52);
            assert_eq!(admitted.layout.frame_ends, [24, 48, 52]);
        }
        let mut changed = pinned_attention_plan(&config, 5).unwrap();
        changed.patch_order = MiMoPatchOrder::Row;
        assert!(request(&config, &changed, 5, &grids, 52 * HIDDEN).is_err());
        let plan = pinned_attention_plan(&config, 5).unwrap();
        assert!(request(&config, &plan, 5, &grids, 52 * HIDDEN - 1).is_err());
        assert!(request(&config, &plan, 28, &grids, 52 * HIDDEN).is_err());
        assert!(
            request(
                &config,
                &plan,
                5,
                &[MiMoVisionGrid {
                    frames: 1,
                    height: 2,
                    width: 258,
                }],
                516 * HIDDEN,
            )
            .is_err()
        );
        assert!(
            request(
                &config,
                &plan,
                5,
                &[MiMoVisionGrid {
                    frames: 5,
                    height: 16,
                    width: 16,
                }],
                1_280 * HIDDEN,
            )
            .is_err()
        );
        assert!(
            request(
                &config,
                &plan,
                5,
                &[MiMoVisionGrid {
                    frames: 33,
                    height: 2,
                    width: 2,
                }],
                132 * HIDDEN,
            )
            .is_err()
        );
    }

    #[test]
    fn column_block_round_trip_preserves_each_merge_unit_and_frame() {
        let config = config();
        let grids = [
            MiMoVisionGrid {
                frames: 1,
                height: 4,
                width: 6,
            },
            MiMoVisionGrid {
                frames: 1,
                height: 2,
                width: 2,
            },
        ];
        let plan = pinned_attention_plan(&config, 5).unwrap();
        let admitted = request(&config, &plan, 5, &grids, 28 * HIDDEN).unwrap();
        let forward = gather_indices(&admitted.layout, false);
        let backward = gather_indices(&admitted.layout, true);
        assert_eq!(&forward[0..8], &[0, 1, 2, 3, 12, 13, 14, 15]);
        assert_eq!(&forward[24..28], &[24, 25, 26, 27]);
        let restored = backward
            .iter()
            .map(|&index| forward[index as usize])
            .collect::<Vec<_>>();
        assert_eq!(restored, (0..28).collect::<Vec<_>>());
    }

    fn bf16(value: f32) -> f32 {
        let bits = value.to_bits();
        let bias = 0x7fff + ((bits >> 16) & 1);
        f32::from_bits(bits.wrapping_add(bias) & 0xffff_0000)
    }

    fn sparse_matrix(
        engine: &Engine,
        input: usize,
        output: usize,
        entries: &[(usize, usize, f32)],
    ) -> Result<GpuTensor, Fail> {
        let mut bytes = vec![0u8; input * output * 2];
        for &(row, column, value) in entries {
            let offset = (row * input + column) * 2;
            bytes[offset..offset + 2]
                .copy_from_slice(&((bf16(value).to_bits() >> 16) as u16).to_le_bytes());
        }
        Ok(GpuTensor::FloatBf16 {
            data: engine.htod_bytes(&bytes)?,
            ne: vec![input as u64, output as u64],
        })
    }

    fn sparse_block(
        engine: &Engine,
        plan: MiMoVisionAttentionPlan,
    ) -> Result<MiMoVisionBlock, Fail> {
        let mut qkv_bias = vec![0.0; FUSED_QKV];
        qkv_bias[0] = 0.125;
        qkv_bias[Q] = -0.0625;
        qkv_bias[Q + KV] = 0.25;
        let mut output_bias = vec![0.0; HIDDEN];
        output_bias[0] = 0.125;
        let mut gate_bias = vec![0.0; FF];
        gate_bias[0] = 0.0625;
        let mut up_bias = vec![0.0; FF];
        up_bias[0] = -0.125;
        let mut down_bias = vec![0.0; HIDDEN];
        down_bias[0] = 0.0625;
        let mut sink = vec![0.0; 32];
        sink[0] = 0.5;
        Ok(MiMoVisionBlock {
            plan,
            norm1: engine.htod(&vec![1.0; HIDDEN])?,
            qkv: sparse_matrix(
                engine,
                HIDDEN,
                FUSED_QKV,
                &[(0, 0, 0.0625), (Q, 0, 0.0625), (Q + KV, 0, 0.0625)],
            )?,
            qkv_bias: engine.htod(&qkv_bias)?,
            first_key_bias: plan
                .sink_first_key
                .then(|| engine.htod(&sink))
                .transpose()?,
            attention_output: sparse_matrix(engine, Q, HIDDEN, &[(0, 0, 0.5)])?,
            attention_output_bias: engine.htod(&output_bias)?,
            norm2: engine.htod(&vec![1.0; HIDDEN])?,
            mlp_gate: sparse_matrix(engine, HIDDEN, FF, &[(0, 0, 0.125)])?,
            mlp_gate_bias: engine.htod(&gate_bias)?,
            mlp_up: sparse_matrix(engine, HIDDEN, FF, &[(0, 0, 0.25)])?,
            mlp_up_bias: engine.htod(&up_bias)?,
            mlp_down: sparse_matrix(engine, FF, HIDDEN, &[(0, 0, 0.5)])?,
            mlp_down_bias: engine.htod(&down_bias)?,
        })
    }

    fn sparse_reference(
        plan: &MiMoVisionAttentionPlan,
        layout: &MiMoVisionLayout,
        input: &[f32],
    ) -> Result<Vec<f32>, Fail> {
        let patches = layout.row_positions.len();
        let mut hidden = if plan.patch_order == MiMoPatchOrder::Column {
            gather_indices(layout, false)
                .iter()
                .flat_map(|&row| input[row as usize * HIDDEN..(row as usize + 1) * HIDDEN].iter())
                .copied()
                .collect::<Vec<_>>()
        } else {
            input.to_vec()
        };
        let mut q = vec![0.0; patches * Q];
        let mut k = vec![0.0; patches * KV];
        let mut v = vec![0.0; patches * KV];
        for patch in 0..patches {
            let a = hidden[patch * HIDDEN];
            let b = hidden[patch * HIDDEN + 1];
            let normalized = bf16(a / ((a * a + b * b) / HIDDEN as f32 + RMS_EPSILON).sqrt());
            q[patch * Q] = bf16(normalized * 0.0625 + 0.125);
            k[patch * KV] = bf16(normalized * 0.0625 - 0.0625);
            v[patch * KV] = bf16(normalized * 0.0625 + 0.25);
        }
        let (q, k) = rotate_qk(plan, layout, &q, &k)?;
        let sink = plan.sink_first_key.then(|| {
            let mut values = vec![0.0; 32];
            values[0] = 0.5;
            values
        });
        let mut previous = 0usize;
        let lengths = layout
            .frame_ends
            .iter()
            .map(|&end| {
                let length = end as usize - previous;
                previous = end as usize;
                length
            })
            .collect::<Vec<_>>();
        let attended = preprojected_attention(plan, &q, &k, &v, &lengths, sink.as_deref())?;
        for patch in 0..patches {
            let at = patch * HIDDEN;
            let projection = bf16(bf16(attended[patch * Q]) * 0.5 + 0.125);
            let after = bf16(hidden[at] + projection);
            let other = hidden[at + 1];
            let normalized = bf16(
                after / ((after * after + other * other) / HIDDEN as f32 + RMS_EPSILON).sqrt(),
            );
            let gate = bf16(normalized * 0.125 + 0.0625);
            let up = bf16(normalized * 0.25 - 0.125);
            let activated = bf16(bf16(gate / (1.0 + (-gate).exp())) * up);
            let down = bf16(activated * 0.5 + 0.0625);
            hidden[at] = bf16(after + down);
        }
        if plan.patch_order == MiMoPatchOrder::Column {
            Ok(gather_indices(layout, true)
                .iter()
                .flat_map(|&row| hidden[row as usize * HIDDEN..(row as usize + 1) * HIDDEN].iter())
                .copied()
                .collect())
        } else {
            Ok(hidden)
        }
    }

    #[test]
    fn portable_sparse_block_preserves_bf16_boundaries_and_uses_first_key_bias() {
        let config = config();
        let grids = [MiMoVisionGrid {
            frames: 1,
            height: 2,
            width: 2,
        }];
        let layout = pinned_vision_layout(&config, &grids).unwrap();
        let mut input = vec![0.0; 4 * HIDDEN];
        for patch in 0..4 {
            input[patch * HIDDEN] = 0.5 + patch as f32 * 0.25;
            input[patch * HIDDEN + 1] = 0.25;
        }
        let full =
            sparse_reference(&pinned_attention_plan(&config, 0).unwrap(), &layout, &input).unwrap();
        let windowed =
            sparse_reference(&pinned_attention_plan(&config, 1).unwrap(), &layout, &input).unwrap();
        assert_eq!(full.len(), input.len());
        assert_eq!(windowed.len(), input.len());
        assert!(full.iter().all(|&value| value == bf16(value)));
        assert!(windowed.iter().all(|&value| value == bf16(value)));
        assert!((0..4).all(|patch| full[patch * HIDDEN + 1] == 0.25));
        let q = vec![0.0; 4 * Q];
        let k = vec![0.0; 4 * KV];
        let mut v = vec![0.0; 4 * KV];
        for patch in 0..4 {
            v[patch * KV] = patch as f32;
        }
        let full_attention = preprojected_attention(
            &pinned_attention_plan(&config, 0).unwrap(),
            &q,
            &k,
            &v,
            &[4],
            None,
        )
        .unwrap();
        let mut first_key_bias = vec![0.0; 32];
        first_key_bias[0] = bf16(2.0f32.ln());
        let local_attention = preprojected_attention(
            &pinned_attention_plan(&config, 1).unwrap(),
            &q,
            &k,
            &v,
            &[4],
            Some(&first_key_bias),
        )
        .unwrap();
        assert_eq!(full_attention[0], 1.5);
        let first_key_weight = first_key_bias[0].exp();
        assert!((local_attention[0] - 6.0 / (first_key_weight + 3.0)).abs() < 1e-6);
    }

    #[test]
    #[ignore = "requires a dedicated MiMo GPU component lane"]
    fn gpu_one_block_matches_sparse_bf16_source_reference() -> Result<(), Fail> {
        let gpu: usize = std::env::var("MEMRA_MIMO_COMPONENT_GPU")
            .unwrap_or_else(|_| "0".into())
            .parse()?;
        let engine = Engine::new(gpu)?;
        let config = config();
        let grids = [
            MiMoVisionGrid {
                frames: 1,
                height: 4,
                width: 6,
            },
            MiMoVisionGrid {
                frames: 1,
                height: 2,
                width: 2,
            },
        ];
        let mut input = vec![0.0; 28 * HIDDEN];
        for patch in 0..28 {
            input[patch * HIDDEN] = 0.5 + (patch % 7) as f32 * 0.25;
            input[patch * HIDDEN + 1] = 0.25;
        }
        for layer in [1, 5] {
            let plan = pinned_attention_plan(&config, layer)?;
            let block = sparse_block(&engine, plan)?;
            let admitted = request(&config, &plan, layer as usize, &grids, input.len())?;
            let expected = sparse_reference(&plan, &admitted.layout, &input)?;
            let got = forward_block(&block, &engine, &engine.htod(&input)?, &admitted)?;
            let got = engine.dtoh(&got)?;
            let max_abs = got
                .iter()
                .zip(&expected)
                .map(|(actual, wanted)| (actual - wanted).abs())
                .fold(0.0f32, f32::max);
            eprintln!("MiMo vision one-block GPU={gpu} layer={layer} max_abs={max_abs}");
            assert!(max_abs <= 0.0625, "MiMo vision one-block parity changed");
        }
        Ok(())
    }
}
