//! MiMo speech-table gather and ordered BF16 accumulation on one GPU.
//! This is a patch-encoder component, not a raw-audio or serving path.

use core::ffi::c_void;

use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};

use crate::Engine;
use crate::mimo_audio_patch_load::MiMoAudioPatchWeights;

unsafe extern "C" {
    fn memra_mimo_audio_embed_sum_bf16(
        codes: *const u16,
        table_ptrs: *const u64,
        output: *mut f32,
        positions: i32,
        channels: i32,
        width: i32,
        vocab: i32,
        stream: *mut c_void,
    ) -> i32;
}

fn positions_for_codes(codes: &[u16], groups: usize) -> Result<usize, &'static str> {
    if !(1..=1_500).contains(&groups) {
        return Err("MiMo audio code groups are outside 1..=1500");
    }
    let positions = groups * 4;
    if codes.len() != positions * 20 || codes.iter().any(|&code| code >= 1_280) {
        return Err("MiMo audio grouped codes have wrong extent or vocabulary");
    }
    Ok(positions)
}

fn sum_with_pointer_row(
    engine: &Engine,
    speech_ptrs: &CudaSlice<u64>,
    codes: &[u16],
    groups: usize,
) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
    let positions = positions_for_codes(codes, groups)?;
    engine.gpu.ctx.bind_to_thread()?;
    let stream = engine.stream();
    if speech_ptrs.len() != 20 || speech_ptrs.ordinal() != stream.context().ordinal() {
        return Err("MiMo speech table pointer row changed size or GPU".into());
    }
    let codes_dev = engine.htod_u16(codes)?;
    let mut output = engine.uninit(positions * 1_024)?;
    let (code_ptr, code_guard) = codes_dev.device_ptr(&stream);
    let (table_ptr, table_guard) = speech_ptrs.device_ptr(&stream);
    let (output_ptr, output_guard) = output.device_ptr_mut(&stream);
    let rc = unsafe {
        memra_mimo_audio_embed_sum_bf16(
            code_ptr as *const u16,
            table_ptr as *const u64,
            output_ptr as *mut f32,
            positions as i32,
            20,
            1_024,
            1_280,
            stream.cu_stream() as *mut c_void,
        )
    };
    drop((code_guard, table_guard, output_guard));
    if rc != 0 {
        return Err(format!("MiMo audio BF16 speech gather returned {rc}").into());
    }
    stream.synchronize()?;
    Ok(output)
}

impl MiMoAudioPatchWeights {
    /// Compute `[groups,4,1024]` f32 values whose elements are BF16-rounded
    /// after each source channel addition. This synchronizes before dropping
    /// its temporary code buffer and returning the result.
    pub fn sum_grouped_codes(
        &self,
        engine: &Engine,
        codes: &[u16],
        groups: usize,
    ) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        if self.plan.code_channels != 20
            || self.plan.group_size != 4
            || self.plan.code_vocab != 1_280
            || self.plan.local_hidden != 1_024
            || self.speech_embeddings.len() != 20
            || self.speech_ptrs.len() != 20
        {
            return Err("MiMo audio source tables differ from pinned patch plan".into());
        }
        sum_with_pointer_row(engine, &self.speech_ptrs, codes, groups)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use memra_gguf::config::{HfConfig, ModelConfig};
    use memra_gguf::model_packs::mimo_v2::audio::pinned_patch_plan;
    use memra_reference::mimo_audio::{group_audio_codes, sum_speech_embeddings_bf16};

    #[test]
    fn grouped_code_extent_and_vocabulary_fail_closed() {
        assert_eq!(positions_for_codes(&[0; 80], 1), Ok(4));
        assert_eq!(
            positions_for_codes(&[1_279; 120], 1),
            Err("MiMo audio grouped codes have wrong extent or vocabulary")
        );
        let mut codes = [0u16; 80];
        codes[19] = 1_280;
        assert!(positions_for_codes(&codes, 1).is_err());
        assert!(positions_for_codes(&[], 0).is_err());
        assert!(positions_for_codes(&[], 1_501).is_err());
    }

    #[test]
    #[ignore = "requires a dedicated MiMo GPU component lane"]
    fn gpu_speech_sum_matches_ordered_bf16_reference() -> Result<(), Box<dyn std::error::Error>> {
        let gpu: usize = std::env::var("MEMRA_MIMO_COMPONENT_GPU")
            .unwrap_or_else(|_| "0".into())
            .parse()?;
        let engine = Engine::new(gpu)?;
        let config = ModelConfig::from_hf(&HfConfig::parse(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../memra-gguf/src/model_packs/mimo_v2/fixtures/config.json"
        ))));
        let plan = pinned_patch_plan(&config)?;
        let code_rows = (0..3 * plan.code_channels)
            .map(|index| ((index * 17 + 9) % plan.code_vocab) as i64)
            .collect::<Vec<_>>();
        let grouped = group_audio_codes(&plan, &code_rows, plan.code_channels)?;
        assert_eq!(grouped.groups, 1);
        let pattern = [0x0000u16, 0x3b80, 0x3e80, 0x3f00, 0x3f80, 0xbf00, 0xbf80];
        let mut tables = vec![vec![0u16; plan.code_vocab * plan.local_hidden]; plan.code_channels];
        for (channel, table) in tables.iter_mut().enumerate() {
            for (index, value) in table.iter_mut().enumerate() {
                *value =
                    pattern[(index * 13 + channel * 5 + index / plan.local_hidden) % pattern.len()];
            }
        }
        let cpu_tables = tables.iter().map(Vec::as_slice).collect::<Vec<_>>();
        let expected = sum_speech_embeddings_bf16(&plan, &grouped, &cpu_tables, plan.code_vocab)?;
        let gpu_tables = tables
            .iter()
            .map(|table| engine.htod_u16(table))
            .collect::<Result<Vec<_>, _>>()?;
        let stream = engine.stream();
        let pointers = gpu_tables
            .iter()
            .map(|table| table.device_ptr(&stream).0)
            .collect::<Vec<_>>();
        let pointer_row = engine.htod_u64(&pointers)?;
        let output = sum_with_pointer_row(&engine, &pointer_row, &grouped.codes, grouped.groups)?;
        let actual = engine.dtoh(&output)?;
        assert_eq!(actual.len(), expected.len());
        for (index, (&got, &want)) in actual.iter().zip(&expected).enumerate() {
            assert_eq!(
                got.to_bits(),
                (u32::from(want) << 16),
                "MiMo BF16 speech sum differs at element {index}"
            );
        }
        println!(
            "mimo_audio_gpu_parity gpu={gpu} grouped_positions={} exact_elements={}",
            grouped.groups * plan.group_size,
            actual.len()
        );
        Ok(())
    }
}
