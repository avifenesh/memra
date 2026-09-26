//! Model-owned MiMo text KV storage and one-token attention component.
//! Global layers keep Q8_0 K and GGUF NVFP4 V; local layers keep a 128-token
//! f32 ring. This has no text-forward dispatch, modality path, or serving door.

use core::ffi::c_void;
use std::error::Error;

use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use memra_gguf::config::ModelConfig;
use memra_gguf::model_plan::ModelPlan;

use crate::Engine;
use crate::mimo_attn_load::MiMoAttentionGeometry;
use crate::mimo_mixed_attn_ffi::MiMoMixedAttentionWorkspace;
use crate::mimo_text_forward::{validate_forward_plan, validate_residency};
use crate::mimo_text_weights::{MiMoTextWeights, stage_for_layer};

type Fail = Box<dyn Error>;
const LAYERS: usize = 48;
const STAGE_CUT: usize = 24;
const HEADS: usize = 64;
const QK: usize = 192;
const VALUE: usize = 128;
const GLOBAL_KV_HEADS: usize = 4;
const LOCAL_KV_HEADS: usize = 8;
const SWA: usize = 128;
const GLOBAL_K_BYTES: usize = GLOBAL_KV_HEADS * (QK / 32) * 34;
const GLOBAL_V_BYTES: usize = GLOBAL_KV_HEADS * (VALUE / 64) * 36;
const DUMMY_Q5_BYTES: usize = GLOBAL_KV_HEADS * (VALUE / 32) * 24;
const LOCAL_K_ELEMENTS: usize = SWA * LOCAL_KV_HEADS * QK;
const LOCAL_V_ELEMENTS: usize = SWA * LOCAL_KV_HEADS * VALUE;
const MIN_FREE_AFTER_BYTES: usize = 4 * 1024 * 1024 * 1024;
const ALLOCATION_SLACK_BYTES: usize = 64 * 1024 * 1024;

/// The pinned source context includes both input and generated tokens.
pub const MAX_COMPRESSED_CONTEXT_TOKENS: usize = 1_048_576;

unsafe extern "C" {
    fn memra_mimo_kv_nvfp4_encode_f32(
        input: *const f32,
        input_elements: usize,
        output: *mut u8,
        output_bytes: usize,
        rows: usize,
        width: i32,
        stream: *mut c_void,
    ) -> i32;
    fn memra_mimo_swa_ring_decode_f32(
        q: *const f32,
        k: *const f32,
        v: *const f32,
        sink: *const f32,
        output: *mut f32,
        position: i32,
        heads: i32,
        kv_heads: i32,
        qk_dim: i32,
        v_dim: i32,
        window: i32,
        stream: *mut c_void,
    ) -> i32;
}

/// Exact persistent allocations, before allocator granularity and transient
/// token projections. `reserved_headroom` is enforced separately at runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MiMoCompressedKvBudget {
    pub cache_bytes: [usize; 2],
    pub workspace_bytes_per_card: usize,
    pub total_bytes: [usize; 2],
}

impl MiMoCompressedKvBudget {
    pub fn for_plan(
        config: &ModelConfig,
        plan: &ModelPlan,
        max_tokens: usize,
    ) -> Result<Self, Fail> {
        validate_forward_plan(config, plan)?;
        if !(1..=MAX_COMPRESSED_CONTEXT_TOKENS).contains(&max_tokens)
            || !plan.partition_boundaries.contains(&STAGE_CUT)
        {
            return Err("MiMo compressed KV needs the pinned context and stage cut".into());
        }
        let mut cache_bytes = [0usize; 2];
        for index in 0..LAYERS {
            let geometry = MiMoAttentionGeometry::from_plan(plan, index)?;
            let stage = stage_for_layer(index)?;
            let bytes = if geometry.window == 0 {
                max_tokens
                    .checked_mul(GLOBAL_K_BYTES + GLOBAL_V_BYTES)
                    .ok_or("MiMo global KV extent overflowed")?
            } else {
                (LOCAL_K_ELEMENTS + LOCAL_V_ELEMENTS)
                    .checked_mul(size_of::<f32>())
                    .ok_or("MiMo local KV extent overflowed")?
            };
            cache_bytes[stage] = cache_bytes[stage]
                .checked_add(bytes)
                .ok_or("MiMo stage KV extent overflowed")?;
        }
        let workspace_bytes_per_card =
            MiMoMixedAttentionWorkspace::native_vscale_bytes(max_tokens)?;
        let total_bytes = cache_bytes.map(|bytes| {
            bytes
                .checked_add(workspace_bytes_per_card)
                .and_then(|sum| sum.checked_add(GLOBAL_K_BYTES + DUMMY_Q5_BYTES))
        });
        Ok(Self {
            cache_bytes,
            workspace_bytes_per_card,
            total_bytes: [
                total_bytes[0].ok_or("MiMo stage 0 allocation extent overflowed")?,
                total_bytes[1].ok_or("MiMo stage 1 allocation extent overflowed")?,
            ],
        })
    }
}

enum LayerCache {
    Global {
        key: CudaSlice<u8>,
        value: CudaSlice<u8>,
        tokens: usize,
    },
    Local {
        key: CudaSlice<f32>,
        value: CudaSlice<f32>,
        tokens: usize,
    },
}

impl LayerCache {
    fn tokens(&self) -> usize {
        match self {
            Self::Global { tokens, .. } | Self::Local { tokens, .. } => *tokens,
        }
    }

    fn validate(&self, geometry: MiMoAttentionGeometry, max_tokens: usize, device: usize) -> bool {
        match self {
            Self::Global { key, value, .. } => {
                geometry.window == 0
                    && geometry.kv_heads == GLOBAL_KV_HEADS
                    && key.len() == max_tokens * GLOBAL_K_BYTES
                    && value.len() == max_tokens * GLOBAL_V_BYTES
                    && key.ordinal() == device
                    && value.ordinal() == device
            }
            Self::Local { key, value, .. } => {
                geometry.window == SWA
                    && geometry.kv_heads == LOCAL_KV_HEADS
                    && key.len() == LOCAL_K_ELEMENTS
                    && value.len() == LOCAL_V_ELEMENTS
                    && key.ordinal() == device
                    && value.ordinal() == device
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Cursor {
    position: usize,
    next_layer: usize,
    failed: bool,
}

impl Cursor {
    fn check(&self, layer: usize, max_tokens: usize, cached: usize) -> Result<(), &'static str> {
        if self.failed {
            return Err("MiMo compressed KV sequence was poisoned by a failed GPU step");
        }
        if self.position >= max_tokens {
            return Err("MiMo compressed KV reached its admitted context cap");
        }
        if layer != self.next_layer || layer >= LAYERS || cached != self.position {
            return Err("MiMo compressed KV layer or token position drifted");
        }
        Ok(())
    }

    fn commit(&mut self) {
        self.next_layer += 1;
        if self.next_layer == LAYERS {
            self.next_layer = 0;
            self.position += 1;
        }
        self.failed = false;
    }
}

/// One continuing text sequence. Calls must visit layers 0..47 in order for
/// each token. An error after a GPU operation poisons the instance because a
/// partial token cannot be rolled back.
pub struct MiMoCompressedKv<'a> {
    weights: &'a MiMoTextWeights,
    engines: [&'a Engine; 2],
    caches: Vec<LayerCache>,
    workspaces: [MiMoMixedAttentionWorkspace; 2],
    key_staging: [CudaSlice<u8>; 2],
    dummy_q5: [CudaSlice<u8>; 2],
    cursor: Cursor,
    max_tokens: usize,
    min_free_after: [usize; 2],
    budget: MiMoCompressedKvBudget,
}

fn require_memory(
    engines: [&Engine; 2],
    min_free: [usize; 2],
    allocations: [usize; 2],
) -> Result<(), Fail> {
    for stage in 0..2 {
        engines[stage].gpu.ctx.bind_to_thread()?;
        let free = engines[stage].ctx().mem_get_info()?.0;
        let required = min_free[stage]
            .checked_add(allocations[stage])
            .ok_or("MiMo memory guard extent overflowed")?;
        if free < required {
            return Err(format!(
                "MiMo stage {stage} has {free} free bytes, needs {required} including headroom"
            )
            .into());
        }
    }
    Ok(())
}

fn encode_value_at(
    engine: &Engine,
    input: &CudaSlice<f32>,
    output: &mut CudaSlice<u8>,
    position: usize,
    max_tokens: usize,
) -> Result<(), Fail> {
    let stream = engine.stream();
    let device = stream.context().ordinal();
    if position >= max_tokens
        || input.len() != GLOBAL_KV_HEADS * VALUE
        || output.len() != max_tokens * GLOBAL_V_BYTES
        || input.ordinal() != device
        || output.ordinal() != device
    {
        return Err("MiMo NVFP4 V append geometry, position, or device changed".into());
    }
    let byte_offset = position * GLOBAL_V_BYTES;
    let mut row = output.slice_mut(byte_offset..byte_offset + GLOBAL_V_BYTES);
    let row_bytes = row.len();
    let rc = unsafe {
        memra_mimo_kv_nvfp4_encode_f32(
            input.device_ptr(&stream).0 as *const f32,
            input.len(),
            row.device_ptr_mut(&stream).0 as *mut u8,
            row_bytes,
            GLOBAL_KV_HEADS,
            VALUE as i32,
            stream.cu_stream() as *mut c_void,
        )
    };
    if rc != 0 {
        return Err(format!("MiMo NVFP4 V append returned {rc}").into());
    }
    Ok(())
}

fn decode_local(
    engine: &Engine,
    query: &CudaSlice<f32>,
    key: &CudaSlice<f32>,
    value: &CudaSlice<f32>,
    sink: &CudaSlice<f32>,
    position: usize,
) -> Result<CudaSlice<f32>, Fail> {
    let stream = engine.stream();
    let device = stream.context().ordinal();
    if position >= MAX_COMPRESSED_CONTEXT_TOKENS
        || query.len() != HEADS * QK
        || key.len() != LOCAL_K_ELEMENTS
        || value.len() != LOCAL_V_ELEMENTS
        || sink.len() != HEADS
        || [
            query.ordinal(),
            key.ordinal(),
            value.ordinal(),
            sink.ordinal(),
        ]
        .iter()
        .any(|&ordinal| ordinal != device)
    {
        return Err("MiMo local ring attention extent, position, or device changed".into());
    }
    let mut output = engine.uninit(HEADS * VALUE)?;
    let rc = unsafe {
        memra_mimo_swa_ring_decode_f32(
            query.device_ptr(&stream).0 as *const f32,
            key.device_ptr(&stream).0 as *const f32,
            value.device_ptr(&stream).0 as *const f32,
            sink.device_ptr(&stream).0 as *const f32,
            output.device_ptr_mut(&stream).0 as *mut f32,
            position as i32,
            HEADS as i32,
            LOCAL_KV_HEADS as i32,
            QK as i32,
            VALUE as i32,
            SWA as i32,
            stream.cu_stream() as *mut c_void,
        )
    };
    if rc != 0 {
        return Err(format!("MiMo local ring attention returned {rc}").into());
    }
    stream.synchronize()?;
    Ok(output)
}

impl MiMoTextWeights {
    /// Admit a separate two-card compressed KV sequence. Caller supplies
    /// minimum free headroom per card after cache and workspace allocations;
    /// each value must reserve at least 4 GiB for other model components.
    pub fn compressed_text_kv<'a>(
        &'a self,
        engines: [&'a Engine; 2],
        max_tokens: usize,
        min_free_after: [usize; 2],
    ) -> Result<MiMoCompressedKv<'a>, Fail> {
        MiMoCompressedKv::new(self, engines, max_tokens, min_free_after)
    }
}

impl<'a> MiMoCompressedKv<'a> {
    fn new(
        weights: &'a MiMoTextWeights,
        engines: [&'a Engine; 2],
        max_tokens: usize,
        min_free_after: [usize; 2],
    ) -> Result<Self, Fail> {
        validate_residency(weights, engines)?;
        if min_free_after
            .iter()
            .any(|&bytes| bytes < MIN_FREE_AFTER_BYTES)
        {
            return Err("MiMo compressed KV requires at least 4 GiB free per card".into());
        }
        let budget = MiMoCompressedKvBudget::for_plan(&weights.config, &weights.plan, max_tokens)?;
        let preflight = budget.total_bytes.map(|bytes| {
            bytes
                .checked_add(ALLOCATION_SLACK_BYTES)
                .ok_or("MiMo KV preflight extent overflowed")
        });
        require_memory(engines, min_free_after, [preflight[0]?, preflight[1]?])?;

        let mut caches = Vec::with_capacity(LAYERS);
        for index in 0..LAYERS {
            let geometry = MiMoAttentionGeometry::from_plan(&weights.plan, index)?;
            let stage = stage_for_layer(index)?;
            let engine = engines[stage];
            engine.gpu.ctx.bind_to_thread()?;
            if geometry.window == 0 {
                caches.push(LayerCache::Global {
                    key: engine.alloc_u8_uninit(max_tokens * GLOBAL_K_BYTES)?,
                    value: engine.alloc_u8_uninit(max_tokens * GLOBAL_V_BYTES)?,
                    tokens: 0,
                });
            } else {
                caches.push(LayerCache::Local {
                    key: engine.uninit(LOCAL_K_ELEMENTS)?,
                    value: engine.uninit(LOCAL_V_ELEMENTS)?,
                    tokens: 0,
                });
            }
        }
        let new_workspace = |stage: usize| {
            engines[stage].gpu.ctx.bind_to_thread()?;
            MiMoMixedAttentionWorkspace::new_dp4a_native_vscale(engines[stage], max_tokens)
        };
        let workspaces = [new_workspace(0)?, new_workspace(1)?];
        let new_key_staging = |stage: usize| {
            engines[stage].gpu.ctx.bind_to_thread()?;
            engines[stage].alloc_u8_uninit(GLOBAL_K_BYTES)
        };
        let key_staging = [new_key_staging(0)?, new_key_staging(1)?];
        let new_dummy = |stage: usize| {
            engines[stage].gpu.ctx.bind_to_thread()?;
            engines[stage].alloc_u8_uninit(DUMMY_Q5_BYTES)
        };
        let dummy_q5 = [new_dummy(0)?, new_dummy(1)?];
        for engine in engines {
            engine.gpu.ctx.bind_to_thread()?;
            engine.stream().synchronize()?;
        }
        require_memory(engines, min_free_after, [0, 0])?;
        Ok(Self {
            weights,
            engines,
            caches,
            workspaces,
            key_staging,
            dummy_q5,
            cursor: Cursor {
                position: 0,
                next_layer: 0,
                failed: false,
            },
            max_tokens,
            min_free_after,
            budget,
        })
    }

    /// Number of complete 48-layer token steps.
    pub fn position(&self) -> usize {
        self.cursor.position
    }

    pub fn budget(&self) -> MiMoCompressedKvBudget {
        self.budget
    }

    /// Append one post-RoPE key and pre-scaled value, then attend with the
    /// current query. Global K uses GGUF Q8_0 bytes; V uses the source-probed
    /// GGUF NVFP4 codec. Local KV stays f32 in a fixed circular window.
    ///
    /// Only the current row is written. The existing Q8 kernel quantizes into
    /// one staged row at offset zero, then copies that row to the cache slot.
    /// The global workspace is shared across layers on each card, so this API
    /// is serial by construction.
    pub fn append_and_attend(
        &mut self,
        layer: usize,
        query: &CudaSlice<f32>,
        key: &CudaSlice<f32>,
        value: &CudaSlice<f32>,
    ) -> Result<CudaSlice<f32>, Fail> {
        let cache = self
            .caches
            .get(layer)
            .ok_or("MiMo compressed KV layer is out of range")?;
        self.cursor.check(layer, self.max_tokens, cache.tokens())?;
        let geometry = MiMoAttentionGeometry::from_plan(&self.weights.plan, layer)?;
        let stage = stage_for_layer(layer)?;
        let engine = self.engines[stage];
        engine.gpu.ctx.bind_to_thread()?;
        let device = engine.stream().context().ordinal();
        if !cache.validate(geometry, self.max_tokens, device)
            || self.key_staging[stage].len() != GLOBAL_K_BYTES
            || self.key_staging[stage].ordinal() != device
            || self.dummy_q5[stage].len() != DUMMY_Q5_BYTES
            || self.dummy_q5[stage].ordinal() != device
            || query.len() != HEADS * QK
            || key.len() != geometry.kv_heads * QK
            || value.len() != geometry.kv_heads * VALUE
            || [query.ordinal(), key.ordinal(), value.ordinal()]
                .iter()
                .any(|&ordinal| ordinal != device)
        {
            return Err("MiMo compressed KV input or residency drifted".into());
        }
        let sink = self.weights.layers[layer].attention.sink.as_ref();
        if sink.is_some() != (geometry.window == SWA)
            || sink.is_some_and(|row| row.len() != HEADS || row.ordinal() != device)
        {
            return Err("MiMo compressed KV learned sink drifted".into());
        }
        if layer == 0 {
            require_memory(self.engines, self.min_free_after, [0, 0])?;
            engine.gpu.ctx.bind_to_thread()?;
        }

        self.cursor.failed = true;
        let output = match &mut self.caches[layer] {
            LayerCache::Global {
                key: keys,
                value: values,
                tokens,
            } => {
                engine.append_kv_quantized(
                    key,
                    value,
                    &mut self.key_staging[stage],
                    &mut self.dummy_q5[stage],
                    0,
                    GLOBAL_KV_HEADS * QK,
                    GLOBAL_KV_HEADS * VALUE,
                    GLOBAL_K_BYTES,
                    DUMMY_Q5_BYTES,
                    false,
                )?;
                let start = self.cursor.position * GLOBAL_K_BYTES;
                engine.stream().memcpy_dtod(
                    &self.key_staging[stage],
                    &mut keys.slice_mut(start..start + GLOBAL_K_BYTES),
                )?;
                encode_value_at(engine, value, values, self.cursor.position, self.max_tokens)?;
                let context = engine.mimo_global_q8_nvfp4_decode(
                    query,
                    keys,
                    values,
                    self.cursor.position + 1,
                    &mut self.workspaces[stage],
                )?;
                *tokens += 1;
                context
            }
            LayerCache::Local {
                key: keys,
                value: values,
                tokens,
            } => {
                let slot = self.cursor.position % SWA;
                engine.dtod_copy_into(key, keys, slot * LOCAL_KV_HEADS * QK)?;
                engine.dtod_copy_into(value, values, slot * LOCAL_KV_HEADS * VALUE)?;
                let context = decode_local(
                    engine,
                    query,
                    keys,
                    values,
                    sink.ok_or("MiMo local layer lost its learned sink")?,
                    self.cursor.position,
                )?;
                *tokens += 1;
                context
            }
        };
        self.cursor.commit();
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use memra_gguf::config::HfConfig;
    use memra_gguf::model_packs::mimo_v2::SOURCE_PROFILE;
    use memra_gguf::model_plan::AttentionPlan;

    fn pinned_plan() -> (ModelConfig, ModelPlan) {
        let config = ModelConfig::from_hf(&HfConfig::parse(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../memra-gguf/src/model_packs/mimo_v2/fixtures/config.json"
        ))));
        let plan = SOURCE_PROFILE.compile_plan(&config).unwrap();
        (config, plan)
    }

    #[test]
    fn two_card_budget_matches_global_and_local_layer_census() {
        let (config, plan) = pinned_plan();
        let cap = MAX_COMPRESSED_CONTEXT_TOKENS;
        let budget = MiMoCompressedKvBudget::for_plan(&config, &plan, cap).unwrap();
        let local_bytes = (LOCAL_K_ELEMENTS + LOCAL_V_ELEMENTS) * size_of::<f32>();
        assert_eq!(
            budget.cache_bytes,
            [
                5 * cap * (GLOBAL_K_BYTES + GLOBAL_V_BYTES) + 19 * local_bytes,
                4 * cap * (GLOBAL_K_BYTES + GLOBAL_V_BYTES) + 20 * local_bytes,
            ]
        );
        assert!(budget.workspace_bytes_per_card > 0);
        assert_eq!(
            budget.total_bytes[0],
            budget.cache_bytes[0]
                + budget.workspace_bytes_per_card
                + GLOBAL_K_BYTES
                + DUMMY_Q5_BYTES
        );
        assert!(MiMoCompressedKvBudget::for_plan(&config, &plan, 0).is_err());
        assert!(MiMoCompressedKvBudget::for_plan(&config, &plan, cap + 1).is_err());
    }

    #[test]
    fn changed_window_stage_and_value_scale_fail_closed() {
        let (config, plan) = pinned_plan();
        let mut changed = plan.clone();
        if let AttentionPlan::SlidingWindow { window, .. } = &mut changed.layers[1].attention {
            *window = 127;
        }
        assert!(MiMoCompressedKvBudget::for_plan(&config, &changed, 256).is_err());
        let mut changed = plan.clone();
        changed.partition_boundaries.clear();
        assert!(MiMoCompressedKvBudget::for_plan(&config, &changed, 256).is_err());
        let mut changed = plan;
        if let AttentionPlan::Full(attention) = &mut changed.layers[0].attention {
            attention
                .mimo_math
                .as_mut()
                .unwrap()
                .value_scale_before_cache = 1.0;
        }
        assert!(MiMoCompressedKvBudget::for_plan(&config, &changed, 256).is_err());
    }

    #[test]
    fn cursor_requires_all_layers_before_advancing_and_refuses_capacity() {
        let mut cursor = Cursor {
            position: 0,
            next_layer: 0,
            failed: false,
        };
        assert!(cursor.check(1, 2, 0).is_err());
        for layer in 0..LAYERS {
            assert!(cursor.check(layer, 2, 0).is_ok());
            cursor.failed = true;
            assert!(cursor.check(layer, 2, 0).is_err());
            cursor.commit();
        }
        assert_eq!((cursor.position, cursor.next_layer), (1, 0));
        assert!(cursor.check(0, 2, 0).is_err());
        assert!(cursor.check(0, 2, 1).is_ok());
        for _ in 0..LAYERS {
            cursor.commit();
        }
        assert!(cursor.check(0, 2, 2).is_err());
    }

    fn ring_value(ring: &[f32; SWA], position: usize) -> f32 {
        let first = (position + 1).saturating_sub(SWA);
        let mut numerator = 0.0;
        let mut denominator = 1.0; // learned sink logit 0
        for token in first..=position {
            let weight = (token as f32 / 128.0).exp();
            numerator += weight * ring[token % SWA];
            denominator += weight;
        }
        numerator / denominator
    }

    #[test]
    fn ring_chronology_matches_contiguous_window_across_wraps() {
        let mut ring = [0.0; SWA];
        let mut history = Vec::new();
        for position in 0..(SWA * 3 + 5) {
            let value = ((position * 17) % 97) as f32 / 31.0;
            ring[position % SWA] = value;
            history.push(value);
            if matches!(position, 0 | 126 | 127 | 128 | 255 | 256 | 388) {
                let first = (position + 1).saturating_sub(SWA);
                let mut numerator = 0.0;
                let mut denominator = 1.0;
                for (token, value) in history.iter().enumerate().take(position + 1).skip(first) {
                    let weight = (token as f32 / 128.0).exp();
                    numerator += weight * value;
                    denominator += weight;
                }
                assert!((ring_value(&ring, position) - numerator / denominator).abs() < 1e-6);
            }
        }
    }

    #[test]
    #[ignore = "requires a dedicated MiMo GPU component lane"]
    fn gpu_local_ring_matches_contiguous_sink_oracle() -> Result<(), Fail> {
        let gpu: usize = std::env::var("MEMRA_MIMO_COMPONENT_GPU")
            .unwrap_or_else(|_| "0".into())
            .parse()?;
        let engine = Engine::new(gpu)?;
        engine.gpu.ctx.bind_to_thread()?;
        let (_, plan) = pinned_plan();
        let mut query = vec![0.0f32; HEADS * QK];
        for head in 0..HEADS {
            query[head * QK] = 10.0;
        }
        let query = engine.htod(&query)?;
        let sink = engine.htod(&vec![0.25f32; HEADS])?;
        let mut ring_k = engine.uninit(LOCAL_K_ELEMENTS)?;
        let mut ring_v = engine.uninit(LOCAL_V_ELEMENTS)?;
        let mut full_k = Vec::new();
        let mut full_v = Vec::new();
        for position in 0..=256 {
            let mut k = vec![0.0f32; LOCAL_KV_HEADS * QK];
            let mut v = vec![0.0f32; LOCAL_KV_HEADS * VALUE];
            for kv_head in 0..LOCAL_KV_HEADS {
                k[kv_head * QK] = (position % 13) as f32;
                for dim in 0..VALUE {
                    v[kv_head * VALUE + dim] =
                        ((position * 17 + kv_head * 7 + dim) % 101) as f32 / 31.0;
                }
            }
            let gpu_k = engine.htod(&k)?;
            let gpu_v = engine.htod(&v)?;
            engine.dtod_copy_into(&gpu_k, &mut ring_k, (position % SWA) * LOCAL_KV_HEADS * QK)?;
            engine.dtod_copy_into(
                &gpu_v,
                &mut ring_v,
                (position % SWA) * LOCAL_KV_HEADS * VALUE,
            )?;
            full_k.extend_from_slice(&k);
            full_v.extend_from_slice(&v);
            if matches!(position, 0 | 127 | 128 | 255 | 256) {
                let actual = decode_local(&engine, &query, &ring_k, &ring_v, &sink, position)?;
                let expected = engine.mimo_sink_decode(
                    &query,
                    &engine.htod(&full_k)?,
                    &engine.htod(&full_v)?,
                    Some(&sink),
                    position + 1,
                    &plan.layers[1].attention,
                )?;
                let actual = engine.dtoh(&actual)?;
                let expected = engine.dtoh(&expected)?;
                assert_eq!(actual.len(), expected.len());
                for (index, (got, want)) in actual.iter().zip(expected.iter()).enumerate() {
                    assert!(
                        (got - want).abs() < 1e-4,
                        "MiMo local ring differs at position {position}, element {index}: {got} vs {want}"
                    );
                }
            }
        }
        Ok(())
    }
}
