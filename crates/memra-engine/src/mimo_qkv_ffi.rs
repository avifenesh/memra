//! Device gather for MiMo's four checkpoint-sharded QKV projections.
//! This is a typed component rewrite, not an admitted MiMo serving backend.

use std::ffi::c_void;

use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use memra_gguf::model_plan::{FullAttentionPlan, TensorPresence};

use crate::Engine;

unsafe extern "C" {
    fn memra_mimo_qkv_gather_f32(
        s0: *const f32,
        s1: *const f32,
        s2: *const f32,
        s3: *const f32,
        q: *mut f32,
        k: *mut f32,
        v: *mut f32,
        tokens: i32,
        q_per_shard: i32,
        k_per_shard: i32,
        v_per_shard: i32,
        stream: *mut c_void,
    ) -> i32;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Geometry {
    q_per_shard: usize,
    k_per_shard: usize,
    v_per_shard: usize,
}

pub struct MiMoQkvOutput {
    pub query: CudaSlice<f32>,
    pub key: CudaSlice<f32>,
    pub value: CudaSlice<f32>,
}

fn geometry(plan: &FullAttentionPlan) -> Result<Geometry, &'static str> {
    let math = plan.mimo_math.ok_or("MiMo QKV plan has no family math")?;
    if math.fused_qkv_checkpoint_shards != Some(4)
        || math.value_scale_before_cache.to_bits() != 0.707f32.to_bits()
        || plan.query_heads != 64
        || plan.key_head_dim != 192
        || plan.value_head_dim != 128
        || !matches!(
            (plan.kv_heads, math.sink),
            (4, TensorPresence::Absent) | (8, TensorPresence::Required)
        )
    {
        return Err("MiMo QKV plan differs from pinned full or sliding geometry");
    }
    Ok(Geometry {
        q_per_shard: 64 / 4 * 192,
        k_per_shard: plan.kv_heads as usize / 4 * 192,
        v_per_shard: plan.kv_heads as usize / 4 * 128,
    })
}

impl Engine {
    /// Assemble four exact checkpoint shard projections on-device. The V output
    /// is scaled by 0.707 before any caller can write it into a KV cache.
    pub fn mimo_gather_qkv(
        &self,
        shards: [&CudaSlice<f32>; 4],
        tokens: usize,
        plan: &FullAttentionPlan,
    ) -> Result<MiMoQkvOutput, Box<dyn std::error::Error>> {
        if !(1..=128).contains(&tokens) {
            return Err("MiMo QKV component accepts 1..=128 tokens".into());
        }
        let shape = geometry(plan)?;
        let shard_width = shape.q_per_shard + shape.k_per_shard + shape.v_per_shard;
        let input_elements = tokens
            .checked_mul(shard_width)
            .ok_or("MiMo shard size overflow")?;
        if shards.iter().any(|shard| shard.len() != input_elements) {
            return Err("MiMo QKV shard projection output length mismatch".into());
        }
        let mut q = self.uninit(tokens * 4 * shape.q_per_shard)?;
        let mut k = self.uninit(tokens * 4 * shape.k_per_shard)?;
        let mut v = self.uninit(tokens * 4 * shape.v_per_shard)?;
        self.gpu.ctx.bind_to_thread()?;
        let stream = self.stream();
        let device = stream.context().ordinal();
        if shards.iter().any(|shard| shard.ordinal() != device) {
            return Err("MiMo QKV shard projection crossed GPU devices".into());
        }
        let rc = unsafe {
            memra_mimo_qkv_gather_f32(
                shards[0].device_ptr(&stream).0 as *const f32,
                shards[1].device_ptr(&stream).0 as *const f32,
                shards[2].device_ptr(&stream).0 as *const f32,
                shards[3].device_ptr(&stream).0 as *const f32,
                q.device_ptr_mut(&stream).0 as *mut f32,
                k.device_ptr_mut(&stream).0 as *mut f32,
                v.device_ptr_mut(&stream).0 as *mut f32,
                tokens as i32,
                shape.q_per_shard as i32,
                shape.k_per_shard as i32,
                shape.v_per_shard as i32,
                stream.cu_stream() as *mut c_void,
            )
        };
        if rc != 0 {
            return Err(format!("MiMo QKV GPU gather returned {rc}").into());
        }
        Ok(MiMoQkvOutput {
            query: q,
            key: k,
            value: v,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use memra_gguf::config::{HfConfig, ModelConfig};
    use memra_gguf::model_plan::{AttentionPlan, ModelPlan};

    #[test]
    fn pinned_plan_has_two_valid_shard_geometries() {
        let config = ModelConfig::from_hf(&HfConfig::parse(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../memra-gguf/src/model_packs/mimo_v2/fixtures/config.json"
        ))));
        let plan = ModelPlan::compile(&config).unwrap();
        let AttentionPlan::Full(full) = &plan.layers[0].attention else {
            panic!("layer 0 must be full");
        };
        assert_eq!(
            geometry(full).unwrap(),
            Geometry {
                q_per_shard: 3072,
                k_per_shard: 192,
                v_per_shard: 128,
            }
        );
        let AttentionPlan::SlidingWindow {
            attention: sliding, ..
        } = &plan.layers[1].attention
        else {
            panic!("layer 1 must slide");
        };
        assert_eq!(
            geometry(sliding).unwrap(),
            Geometry {
                q_per_shard: 3072,
                k_per_shard: 384,
                v_per_shard: 256,
            }
        );
        let mut changed = sliding.clone();
        changed.mimo_math.as_mut().unwrap().value_scale_before_cache = 1.0;
        assert!(geometry(&changed).is_err());
    }
}
