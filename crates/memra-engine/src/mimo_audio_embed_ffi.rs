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
        let positions = positions_for_codes(codes, groups)?;
        if self.plan.code_channels != 20
            || self.plan.group_size != 4
            || self.plan.code_vocab != 1_280
            || self.plan.local_hidden != 1_024
            || self.speech_embeddings.len() != 20
            || self.speech_ptrs.len() != 20
        {
            return Err("MiMo audio source tables differ from pinned patch plan".into());
        }
        engine.gpu.ctx.bind_to_thread()?;
        let stream = engine.stream();
        if self.speech_ptrs.ordinal() != stream.context().ordinal() {
            return Err("MiMo speech table pointer row crossed GPU devices".into());
        }
        let codes_dev = engine.htod_u16(codes)?;
        let mut output = engine.uninit(positions * self.plan.local_hidden)?;
        let (code_ptr, code_guard) = codes_dev.device_ptr(&stream);
        let (table_ptr, table_guard) = self.speech_ptrs.device_ptr(&stream);
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
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
