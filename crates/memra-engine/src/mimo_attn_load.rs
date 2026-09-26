//! Plan-bound MiMo source attention weights. Loading a layer is not serving support.

use cudarc::driver::CudaSlice;
use memra_gguf::GgmlType;
use memra_gguf::checkpoint_binding::CheckpointBinding;
use memra_gguf::config::{Arch, AttentionGateKind};
use memra_gguf::model_plan::{
    AttentionPlan, AttentionScale, ModelPlan, ResidualTopology, RopeFactors, StatePlan,
    TensorPresence, ValueNorm, ValueProjection,
};
use memra_gguf::source::TensorSource;
use memra_gguf::tensor_contract::{LayerTensor, TensorId};

use crate::Engine;
use crate::model::GpuTensor;

type Fail = Box<dyn std::error::Error>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MiMoAttentionGeometry {
    pub kv_heads: usize,
    pub window: usize,
    pub shard_rows: usize,
    pub output_width: usize,
}

impl MiMoAttentionGeometry {
    pub fn from_plan(plan: &ModelPlan, index: usize) -> Result<Self, &'static str> {
        if plan.arch != Arch::MiMoV2 || plan.hidden_size != 4096 || plan.layers.len() != 48 {
            return Err("MiMo attention loader requires the pinned 48-layer text trunk");
        }
        let layer = plan
            .layers
            .get(index)
            .ok_or("MiMo attention layer is out of range")?;
        if layer.index as usize != index
            || layer.residual != ResidualTopology::Serial
            || layer.sparse_overlay.is_some()
            || layer.ple.is_some()
        {
            return Err("MiMo layer plan differs from the pinned serial trunk");
        }
        let (attention, window) = match &layer.attention {
            AttentionPlan::Full(attention) => (attention, 0),
            AttentionPlan::SlidingWindow { attention, window } => (attention, *window as usize),
            _ => return Err("MiMo attention loader needs full or sliding attention"),
        };
        let math = attention
            .mimo_math
            .ok_or("MiMo attention plan has no family math")?;
        let expects_global = index == 0 || (index + 1).is_multiple_of(6);
        if (window == 0) != expects_global {
            return Err("MiMo global attention layer position differs from pinned source");
        }
        let expected_rope_base = if expects_global {
            10_000_000.0f32
        } else {
            10_000.0f32
        };
        if math.fused_qkv_checkpoint_shards != Some(4)
            || math.value_scale_before_cache.to_bits() != 0.707f32.to_bits()
            || attention.query_heads != 64
            || attention.key_head_dim != 192
            || attention.value_head_dim != 128
            || attention.rope.dimensions != 64
            || attention.rope.base.to_bits() != expected_rope_base.to_bits()
            || !matches!(attention.rope.factors, RopeFactors::None)
            || attention.qk_norm != TensorPresence::Absent
            || attention.output_gate != AttentionGateKind::None
            || attention.scale != AttentionScale::InverseSqrtKeyDim
            || attention.value_projection != ValueProjection::Separate
            || attention.value_norm != ValueNorm::None
        {
            return Err("MiMo attention math differs from the pinned source");
        }
        let (kv_heads, key_width, value_width) =
            match (attention.kv_heads, window, math.sink, layer.state) {
                (
                    4,
                    0,
                    TensorPresence::Absent,
                    StatePlan::KvCache {
                        key_width: 768,
                        value_width: 512,
                    },
                ) => (4usize, 768usize, 512usize),
                (
                    8,
                    128,
                    TensorPresence::Required,
                    StatePlan::SlidingKvCache {
                        key_width: 1536,
                        value_width: 1024,
                        window: 128,
                    },
                ) => (8usize, 1536usize, 1024usize),
                _ => return Err("MiMo KV state, sink, or window differs from pinned source"),
            };
        let output_width = 64 * 128;
        let shard_rows = (64 * 192 + key_width + value_width) / 4;
        Ok(Self {
            kv_heads,
            window,
            shard_rows,
            output_width,
        })
    }
}

pub struct MiMoSourceAttention {
    pub qkv: Vec<GpuTensor>,
    pub output: GpuTensor,
    pub sink: Option<CudaSlice<f32>>,
    pub geometry: MiMoAttentionGeometry,
}

impl MiMoSourceAttention {
    /// Load one exact MiMo source mixer through the bound, recorded source.
    /// The caller owns the model-wide binding and later consumption audit.
    pub fn load(
        engine: &Engine,
        source: &dyn TensorSource,
        binding: &CheckpointBinding,
        plan: &ModelPlan,
        index: usize,
    ) -> Result<Self, Fail> {
        let geometry = MiMoAttentionGeometry::from_plan(plan, index)?;
        if binding.family() != "mimo_v2_source" {
            return Err("MiMo source attention needs the source pack binding".into());
        }
        engine.gpu.ctx.bind_to_thread()?;
        let layer = index as u32;
        let qkv_name = binding.require_ggml(&TensorId::Layer {
            index: layer,
            tensor: LayerTensor::FusedQkv,
        })?;
        let views = source
            .find_mimo_fp8_qkv_ggml(&qkv_name)
            .ok_or_else(|| format!("{qkv_name}: native MiMo QKV shards unavailable"))?;
        if views.len() != 4
            || views.iter().any(|view| {
                view.out_f != geometry.shard_rows || view.in_f != plan.hidden_size as usize
            })
        {
            return Err(format!("{qkv_name}: source QKV shard geometry changed").into());
        }
        let qkv = views
            .iter()
            .map(|view| GpuTensor::load_mimo_fp8_qkv_shard(engine, view))
            .collect::<Result<Vec<_>, _>>()?;

        let output_name = binding.require_ggml(&TensorId::Layer {
            index: layer,
            tensor: LayerTensor::AttentionOutput,
        })?;
        let output_view = source
            .find_mimo_bf16_ggml(&output_name)
            .ok_or_else(|| format!("{output_name}: MiMo output projection missing"))?;
        if output_view.ggml_type != GgmlType::BF16
            || output_view.ne != [geometry.output_width as u64, plan.hidden_size as u64]
            || output_view.bytes.len()
                != geometry.output_width * plan.hidden_size as usize * size_of::<u16>()
        {
            return Err(format!("{output_name}: source BF16 output geometry changed").into());
        }
        let output = GpuTensor::FloatBf16 {
            data: engine.htod_bytes(&output_view.bytes)?,
            ne: output_view.ne,
        };

        let sink_id = TensorId::Layer {
            index: layer,
            tensor: LayerTensor::AttentionSink,
        };
        let sink = if geometry.window == 0 {
            if binding.has(&sink_id) {
                return Err("MiMo global attention unexpectedly bound a sink".into());
            }
            None
        } else {
            let name = binding.require_ggml(&sink_id)?;
            let view = source
                .find_mimo_bf16_ggml(&name)
                .ok_or_else(|| format!("{name}: MiMo sliding sink missing"))?;
            if view.ggml_type != GgmlType::BF16
                || view.ne != [64]
                || view.bytes.len() != 64 * size_of::<u16>()
            {
                return Err(format!("{name}: source BF16 sink geometry changed").into());
            }
            let values: Vec<f32> = view
                .bytes
                .chunks_exact(2)
                .map(|pair| f32::from_bits(u32::from(u16::from_le_bytes([pair[0], pair[1]])) << 16))
                .collect();
            if values.iter().any(|value| !value.is_finite()) {
                return Err(format!("{name}: source sink is not finite").into());
            }
            Some(engine.htod(&values)?)
        };
        Ok(Self {
            qkv,
            output,
            sink,
            geometry,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use memra_gguf::config::{HfConfig, ModelConfig};

    #[test]
    fn pinned_global_and_sliding_geometry_refuse_changed_math() {
        let config = ModelConfig::from_hf(&HfConfig::parse(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../memra-gguf/src/model_packs/mimo_v2/fixtures/config.json"
        ))));
        let plan = ModelPlan::compile(&config).unwrap();
        assert_eq!(
            MiMoAttentionGeometry::from_plan(&plan, 0),
            Ok(MiMoAttentionGeometry {
                kv_heads: 4,
                window: 0,
                shard_rows: 3392,
                output_width: 8192,
            })
        );
        assert_eq!(
            MiMoAttentionGeometry::from_plan(&plan, 1),
            Ok(MiMoAttentionGeometry {
                kv_heads: 8,
                window: 128,
                shard_rows: 3712,
                output_width: 8192,
            })
        );
        let mut wrong = plan.clone();
        if let AttentionPlan::SlidingWindow { window, .. } = &mut wrong.layers[1].attention {
            *window = 127;
        }
        assert!(MiMoAttentionGeometry::from_plan(&wrong, 1).is_err());
        wrong = plan;
        if let AttentionPlan::Full(attention) = &mut wrong.layers[0].attention {
            attention
                .mimo_math
                .as_mut()
                .unwrap()
                .value_scale_before_cache = 1.0;
        }
        assert!(MiMoAttentionGeometry::from_plan(&wrong, 0).is_err());
        let mut wrong = ModelPlan::compile(&config).unwrap();
        let misplaced = wrong.layers[1].attention.clone();
        wrong.layers[0].attention = misplaced;
        assert!(MiMoAttentionGeometry::from_plan(&wrong, 0).is_err());
    }
}
