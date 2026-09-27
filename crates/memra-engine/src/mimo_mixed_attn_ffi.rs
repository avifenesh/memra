//! Source-inspection split attention for MiMo global layers with Q8_0 K
//! and either default NVFP4 V or explicit experimental S5 group16 V.
//! Admission to Memra serving requires separate model and request qualification.

use core::ffi::c_void;

use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};

use crate::Engine;

const HEADS: usize = 64;
const KV_HEADS: usize = 4;
const QK: usize = 192;
const VALUE: usize = 128;
const Q8_ROW_BYTES: usize = QK / 32 * 34;
const NVFP4_ROW_BYTES: usize = VALUE / 64 * 36;
const S5_ROW_BYTES: usize = crate::mimo_s5_g16_codec::HEAD_BYTES;
const BASE_TILE: usize = 256;
const GROUPED_TILE: usize = 64;
const DEEP_SPLIT: usize = 512;
const REDUCE: usize = 128;
const PARTIAL: usize = VALUE + 2;
const MAX_SEQ: usize = 1_048_576;

unsafe extern "C" {
    fn memra_mimo_global_q8_nvfp4_decode(
        q: *const f32,
        k: *const u8,
        v: *const u8,
        output: *mut f32,
        scratch1: *mut f32,
        scratch2: *mut f32,
        scratch3: *mut f32,
        seq: i32,
        heads: i32,
        kv_heads: i32,
        qk_dim: i32,
        v_dim: i32,
        scratch1_floats: usize,
        scratch2_floats: usize,
        scratch3_floats: usize,
        stream: *mut c_void,
    ) -> i32;
    fn memra_mimo_global_q8_nvfp4_decode_grouped(
        q: *const f32,
        k: *const u8,
        v: *const u8,
        output: *mut f32,
        scratch1: *mut f32,
        scratch2: *mut f32,
        scratch3: *mut f32,
        seq: i32,
        heads: i32,
        kv_heads: i32,
        qk_dim: i32,
        v_dim: i32,
        scratch1_floats: usize,
        scratch2_floats: usize,
        scratch3_floats: usize,
        stream: *mut c_void,
    ) -> i32;
    fn memra_mimo_global_q8_nvfp4_decode_deep(
        q: *const f32,
        k: *const u8,
        v: *const u8,
        output: *mut f32,
        scratch1: *mut f32,
        scratch2: *mut f32,
        scratch3: *mut f32,
        seq: i32,
        heads: i32,
        kv_heads: i32,
        qk_dim: i32,
        v_dim: i32,
        scratch1_floats: usize,
        scratch2_floats: usize,
        scratch3_floats: usize,
        stream: *mut c_void,
    ) -> i32;
    fn memra_mimo_global_q8_nvfp4_decode_dp4a(
        q: *const f32,
        k: *const u8,
        v: *const u8,
        output: *mut f32,
        scratch1: *mut f32,
        scratch2: *mut f32,
        scratch3: *mut f32,
        seq: i32,
        heads: i32,
        kv_heads: i32,
        qk_dim: i32,
        v_dim: i32,
        scratch1_floats: usize,
        scratch2_floats: usize,
        scratch3_floats: usize,
        stream: *mut c_void,
    ) -> i32;
    fn memra_mimo_global_q8_nvfp4_decode_dp4a_native_vscale(
        q: *const f32,
        k: *const u8,
        v: *const u8,
        output: *mut f32,
        scratch1: *mut f32,
        scratch2: *mut f32,
        scratch3: *mut f32,
        seq: i32,
        heads: i32,
        kv_heads: i32,
        qk_dim: i32,
        v_dim: i32,
        scratch1_floats: usize,
        scratch2_floats: usize,
        scratch3_floats: usize,
        stream: *mut c_void,
    ) -> i32;
    fn memra_mimo_global_q8_s5_g16_decode_dp4a(
        q: *const f32,
        k: *const u8,
        v: *const u8,
        output: *mut f32,
        scratch1: *mut f32,
        scratch2: *mut f32,
        scratch3: *mut f32,
        seq: i32,
        heads: i32,
        kv_heads: i32,
        qk_dim: i32,
        v_dim: i32,
        scratch1_floats: usize,
        scratch2_floats: usize,
        scratch3_floats: usize,
        stream: *mut c_void,
    ) -> i32;
}

fn extents(seq: usize, program: u8) -> Result<(usize, usize, usize), &'static str> {
    if !(1..=MAX_SEQ).contains(&seq) {
        return Err("MiMo mixed attention accepts 1..=1048576 tokens");
    }
    let tile_size = match program {
        0 => BASE_TILE,
        1 => GROUPED_TILE,
        2..=5 => DEEP_SPLIT,
        _ => return Err("MiMo mixed attention program is unavailable"),
    };
    let tiles = seq.div_ceil(tile_size);
    let groups = tiles.div_ceil(REDUCE);
    if groups > REDUCE {
        return Err("MiMo mixed attention reduction grid is too large");
    }
    Ok((
        HEADS * tiles * PARTIAL,
        HEADS * groups * PARTIAL,
        HEADS * PARTIAL,
    ))
}

pub struct MiMoMixedAttentionWorkspace {
    scratch1: CudaSlice<f32>,
    scratch2: CudaSlice<f32>,
    scratch3: CudaSlice<f32>,
    max_seq: usize,
    program: u8,
}

impl MiMoMixedAttentionWorkspace {
    pub(crate) fn native_vscale_bytes(max_seq: usize) -> Result<usize, &'static str> {
        let (one, two, three) = extents(max_seq, 4)?;
        one.checked_add(two)
            .and_then(|bytes| bytes.checked_add(three))
            .and_then(|floats| floats.checked_mul(size_of::<f32>()))
            .ok_or("MiMo mixed attention workspace extent overflowed")
    }

    pub fn new(engine: &Engine, max_seq: usize) -> Result<Self, Box<dyn std::error::Error>> {
        Self::new_for(engine, max_seq, 0)
    }

    pub fn new_grouped(
        engine: &Engine,
        max_seq: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        Self::new_for(engine, max_seq, 1)
    }

    pub fn new_deep(engine: &Engine, max_seq: usize) -> Result<Self, Box<dyn std::error::Error>> {
        Self::new_for(engine, max_seq, 2)
    }

    pub fn new_dp4a(engine: &Engine, max_seq: usize) -> Result<Self, Box<dyn std::error::Error>> {
        Self::new_for(engine, max_seq, 3)
    }

    pub fn new_dp4a_native_vscale(
        engine: &Engine,
        max_seq: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        Self::new_for(engine, max_seq, 4)
    }

    /// Workspace for the explicit MiMo S5 group16 V reader.
    pub fn new_dp4a_s5_g16(
        engine: &Engine,
        max_seq: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        Self::new_for(engine, max_seq, 5)
    }

    fn new_for(
        engine: &Engine,
        max_seq: usize,
        program: u8,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let (one, two, three) = extents(max_seq, program)?;
        engine.gpu.ctx.bind_to_thread()?;
        Ok(Self {
            scratch1: engine.uninit(one)?,
            scratch2: engine.uninit(two)?,
            scratch3: engine.uninit(three)?,
            max_seq,
            program,
        })
    }
}

impl Engine {
    /// Decode one query against a contiguous, current-token-inclusive packed
    /// global cache. K rows are q8_0 and V rows use Memra's GGUF NVFP4 layout.
    /// Q and K already have RoPE applied; V already includes MiMo's 0.707 scale.
    pub fn mimo_global_q8_nvfp4_decode(
        &self,
        query: &CudaSlice<f32>,
        key: &CudaSlice<u8>,
        value: &CudaSlice<u8>,
        seq: usize,
        workspace: &mut MiMoMixedAttentionWorkspace,
    ) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        if workspace.program == 5 {
            return Err("MiMo NVFP4 V decode received an S5 workspace".into());
        }
        self.mimo_global_decode(query, key, value, seq, workspace)
    }

    /// Decode one query from the explicit packed S5 group16 V cache.
    pub fn mimo_global_q8_s5_g16_decode(
        &self,
        query: &CudaSlice<f32>,
        key: &CudaSlice<u8>,
        value: &CudaSlice<u8>,
        seq: usize,
        workspace: &mut MiMoMixedAttentionWorkspace,
    ) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        if workspace.program != 5 {
            return Err("MiMo S5 V decode requires its dedicated workspace".into());
        }
        self.mimo_global_decode(query, key, value, seq, workspace)
    }

    fn mimo_global_decode(
        &self,
        query: &CudaSlice<f32>,
        key: &CudaSlice<u8>,
        value: &CudaSlice<u8>,
        seq: usize,
        workspace: &mut MiMoMixedAttentionWorkspace,
    ) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        let (one, two, three) = extents(seq, workspace.program)?;
        let v_row_bytes = if workspace.program == 5 {
            S5_ROW_BYTES
        } else {
            NVFP4_ROW_BYTES
        };
        if seq > workspace.max_seq
            || query.len() != HEADS * QK
            || key.len() < seq * KV_HEADS * Q8_ROW_BYTES
            || value.len() < seq * KV_HEADS * v_row_bytes
            || workspace.scratch1.len() < one
            || workspace.scratch2.len() < two
            || workspace.scratch3.len() < three
        {
            return Err("MiMo mixed attention tensor or workspace extent mismatch".into());
        }
        self.gpu.ctx.bind_to_thread()?;
        let stream = self.stream();
        let device = stream.context().ordinal();
        if query.ordinal() != device
            || key.ordinal() != device
            || value.ordinal() != device
            || workspace.scratch1.ordinal() != device
            || workspace.scratch2.ordinal() != device
            || workspace.scratch3.ordinal() != device
        {
            return Err("MiMo mixed attention inputs crossed GPU devices".into());
        }
        let mut output = self.uninit(HEADS * VALUE)?;
        let scratch1_floats = workspace.scratch1.len();
        let scratch2_floats = workspace.scratch2.len();
        let scratch3_floats = workspace.scratch3.len();
        let launch = match workspace.program {
            0 => memra_mimo_global_q8_nvfp4_decode,
            1 => memra_mimo_global_q8_nvfp4_decode_grouped,
            2 => memra_mimo_global_q8_nvfp4_decode_deep,
            3 => memra_mimo_global_q8_nvfp4_decode_dp4a,
            4 => memra_mimo_global_q8_nvfp4_decode_dp4a_native_vscale,
            5 => memra_mimo_global_q8_s5_g16_decode_dp4a,
            _ => return Err("MiMo mixed attention program is unavailable".into()),
        };
        let rc = unsafe {
            launch(
                query.device_ptr(&stream).0 as *const f32,
                key.device_ptr(&stream).0 as *const u8,
                value.device_ptr(&stream).0 as *const u8,
                output.device_ptr_mut(&stream).0 as *mut f32,
                workspace.scratch1.device_ptr_mut(&stream).0 as *mut f32,
                workspace.scratch2.device_ptr_mut(&stream).0 as *mut f32,
                workspace.scratch3.device_ptr_mut(&stream).0 as *mut f32,
                seq as i32,
                HEADS as i32,
                KV_HEADS as i32,
                QK as i32,
                VALUE as i32,
                scratch1_floats,
                scratch2_floats,
                scratch3_floats,
                stream.cu_stream() as *mut c_void,
            )
        };
        if rc != 0 {
            return Err(format!("MiMo mixed attention GPU decode returned {rc}").into());
        }
        stream.synchronize()?;
        Ok(output)
    }
}

#[cfg(test)]
mod s5_tests {
    use super::*;
    use crate::mimo_s5_g16_codec::{decode_reference, encode_reference};
    use memra_gguf::GgmlType;
    use memra_gguf::dequant::dequantize;
    use memra_gguf::nvfp4_repack::f32_to_q8_0;

    fn fixture(seq: usize) -> (Vec<f32>, Vec<u8>, Vec<f32>, Vec<u8>) {
        let query = (0..HEADS * QK)
            .map(|i| ((i * 17 % 79) as f32 - 39.0) / 72.0)
            .collect::<Vec<_>>();
        let key = (0..seq * KV_HEADS * QK)
            .map(|i| ((i * 11 % 89) as f32 - 44.0) / 80.0)
            .collect::<Vec<_>>();
        let value = (0..seq * KV_HEADS * VALUE)
            .map(|i| ((i * 23 % 97) as f32 - 48.0) / 61.0)
            .collect::<Vec<_>>();
        let key_bytes = f32_to_q8_0(&key);
        let value_bytes = encode_reference(&value).unwrap();
        (query, key_bytes, value, value_bytes)
    }

    fn packed_oracle(query: &[f32], key_bytes: &[u8], value_bytes: &[u8], seq: usize) -> Vec<f32> {
        let key = dequantize(GgmlType::Q8_0, key_bytes, seq * KV_HEADS * QK);
        let value = decode_reference(value_bytes).unwrap();
        let mut output = vec![0.0f32; HEADS * VALUE];
        for head in 0..HEADS {
            let kv_head = head / (HEADS / KV_HEADS);
            let mut scaled_query = [0.0f32; QK];
            for block in 0..QK / 32 {
                let begin = head * QK + block * 32;
                let part = &query[begin..begin + 32];
                let maximum = part.iter().fold(0.0f32, |a, &x| a.max(x.abs()));
                let step = if maximum > 0.0 {
                    maximum / (QK as f32).sqrt() / 127.0
                } else {
                    1.0
                };
                for dim in 0..32 {
                    let code = ((part[dim] / (QK as f32).sqrt() / step).round_ties_even() as i32)
                        .clamp(-127, 127);
                    scaled_query[block * 32 + dim] = code as f32 * step;
                }
            }
            let scores = (0..seq)
                .map(|token| {
                    let base = (token * KV_HEADS + kv_head) * QK;
                    (0..QK)
                        .map(|dim| f64::from(scaled_query[dim]) * f64::from(key[base + dim]))
                        .sum::<f64>()
                })
                .collect::<Vec<_>>();
            let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let weights = scores
                .iter()
                .map(|&score| (score - maximum).exp())
                .collect::<Vec<_>>();
            let denominator = weights.iter().sum::<f64>();
            for (token, weight) in weights.into_iter().enumerate() {
                let base = (token * KV_HEADS + kv_head) * VALUE;
                for dim in 0..VALUE {
                    output[head * VALUE + dim] +=
                        (weight / denominator * f64::from(value[base + dim])) as f32;
                }
            }
        }
        output
    }

    #[test]
    fn packed_oracle_respects_causal_prefix_and_gqa_heads() {
        let query = vec![0.0f32; HEADS * QK];
        let key_bytes = f32_to_q8_0(&vec![0.0f32; 3 * KV_HEADS * QK]);
        let values = (0..3 * KV_HEADS * VALUE)
            .map(|i| {
                let token = i / (KV_HEADS * VALUE);
                let head = i / VALUE % KV_HEADS;
                (token as f32 - 1.0) * (head as f32 + 1.0)
            })
            .collect::<Vec<_>>();
        let value_bytes = encode_reference(&values).unwrap();
        let decoded = decode_reference(&value_bytes).unwrap();
        let first = packed_oracle(
            &query,
            &key_bytes[..KV_HEADS * Q8_ROW_BYTES],
            &value_bytes[..KV_HEADS * S5_ROW_BYTES],
            1,
        );
        let last = packed_oracle(&query, &key_bytes, &value_bytes, 3);
        for head in 0..HEADS {
            let kv_head = head / (HEADS / KV_HEADS);
            for dim in 0..VALUE {
                assert_eq!(first[head * VALUE + dim], decoded[kv_head * VALUE + dim]);
                let mean = (0..3)
                    .map(|token| decoded[(token * KV_HEADS + kv_head) * VALUE + dim])
                    .sum::<f32>()
                    / 3.0;
                assert!((last[head * VALUE + dim] - mean).abs() < 1e-6);
            }
        }
    }

    #[test]
    #[ignore = "requires an explicitly assigned GPU; S5 packed attention is unqualified"]
    fn gpu_s5_writer_and_deep_attention_match_packed_oracle()
    -> Result<(), Box<dyn std::error::Error>> {
        let gpu: usize = std::env::var("MEMRA_MIMO_COMPONENT_GPU")?.parse()?;
        let engine = Engine::new(gpu)?;
        for seq in [1, 7, 33, 128, 129, 256] {
            let (query, key_bytes, value, value_bytes) = fixture(seq);
            let query_gpu = engine.htod(&query)?;
            let key_gpu = engine.htod_bytes(&key_bytes)?;
            let value_gpu = engine.mimo_s5_g16_encode_rows(&engine.htod(&value)?)?;
            assert_eq!(engine.dtoh_u8(&value_gpu)?, value_bytes, "seq={seq}");
            let mut s5_workspace = MiMoMixedAttentionWorkspace::new_dp4a_s5_g16(&engine, seq)?;
            let mut nvfp4_workspace =
                MiMoMixedAttentionWorkspace::new_dp4a_native_vscale(&engine, seq)?;
            assert!(
                engine
                    .mimo_global_q8_nvfp4_decode(
                        &query_gpu,
                        &key_gpu,
                        &value_gpu,
                        seq,
                        &mut s5_workspace
                    )
                    .is_err()
            );
            assert!(
                engine
                    .mimo_global_q8_s5_g16_decode(
                        &query_gpu,
                        &key_gpu,
                        &value_gpu,
                        seq,
                        &mut nvfp4_workspace
                    )
                    .is_err()
            );
            let actual = engine.dtoh(&engine.mimo_global_q8_s5_g16_decode(
                &query_gpu,
                &key_gpu,
                &value_gpu,
                seq,
                &mut s5_workspace,
            )?)?;
            let expected = packed_oracle(&query, &key_bytes, &value_bytes, seq);
            let max_abs = actual
                .iter()
                .zip(expected)
                .map(|(got, want)| (got - want).abs())
                .fold(0.0f32, f32::max);
            assert!(actual.iter().all(|x| x.is_finite()));
            assert!(max_abs <= 0.001, "seq={seq} packed max_abs={max_abs}");
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires an explicitly assigned GPU; S5 split reduction is unqualified"]
    fn gpu_s5_second_deep_split_contributes_to_attention() -> Result<(), Box<dyn std::error::Error>>
    {
        const SEQ: usize = DEEP_SPLIT + 1;
        let gpu: usize = std::env::var("MEMRA_MIMO_COMPONENT_GPU")?.parse()?;
        let engine = Engine::new(gpu)?;
        let query = engine.htod(&vec![0.0f32; HEADS * QK])?;
        let zero_key = f32_to_q8_0(&vec![0.0f32; KV_HEADS * QK]);
        let key = engine.htod_bytes(&zero_key.repeat(SEQ))?;
        let first = (0..KV_HEADS * VALUE)
            .map(|i| ((i * 17 % 47) as f32 - 23.0) / 29.0)
            .collect::<Vec<_>>();
        let last = (0..KV_HEADS * VALUE)
            .map(|i| ((i * 13 % 61) as f32 - 30.0) / 17.0)
            .collect::<Vec<_>>();
        let first_bytes = encode_reference(&first)?;
        let last_bytes = encode_reference(&last)?;
        let mut packed = first_bytes.repeat(DEEP_SPLIT);
        packed.extend_from_slice(&last_bytes);
        let value = engine.htod_bytes(&packed)?;
        let first = decode_reference(&first_bytes)?;
        let last = decode_reference(&last_bytes)?;
        let mut workspace = MiMoMixedAttentionWorkspace::new_dp4a_s5_g16(&engine, SEQ)?;
        let output = engine.dtoh(&engine.mimo_global_q8_s5_g16_decode(
            &query,
            &key,
            &value,
            SEQ,
            &mut workspace,
        )?)?;
        for head in 0..HEADS {
            let kv_head = head / (HEADS / KV_HEADS);
            for dim in 0..VALUE {
                let index = kv_head * VALUE + dim;
                let want = (f64::from(first[index]) * DEEP_SPLIT as f64 + f64::from(last[index]))
                    / SEQ as f64;
                let got = f64::from(output[head * VALUE + dim]);
                assert!(
                    (got - want).abs() <= 1e-4,
                    "head={head} dim={dim} got={got} want={want}"
                );
            }
        }
        Ok(())
    }
}
