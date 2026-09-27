//! MiMo NVFP4 cache row codec for source-inspection qualification.

use core::ffi::c_void;

use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};

use crate::Engine;

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
    fn memra_mimo_kv_nvfp4_decode_f32(
        input: *const u8,
        input_bytes: usize,
        output: *mut f32,
        output_elements: usize,
        rows: usize,
        width: i32,
        stream: *mut c_void,
    ) -> i32;
}

fn packed_extent(rows: usize, width: usize) -> Result<usize, &'static str> {
    if rows == 0 || !matches!(width, 128 | 192) {
        return Err("MiMo NVFP4 rows require nonzero count and width 128 or 192");
    }
    rows.checked_mul(width / 64 * 36)
        .ok_or("MiMo NVFP4 packed extent overflowed")
}

impl Engine {
    /// Encode complete contiguous f32 K or V rows to Memra GGUF NVFP4 bytes.
    /// The caller supplies already rotated K or already MiMo-scaled V.
    pub fn mimo_nvfp4_encode_rows(
        &self,
        input: &CudaSlice<f32>,
        width: usize,
    ) -> Result<CudaSlice<u8>, Box<dyn std::error::Error>> {
        if !matches!(width, 128 | 192) || !input.len().is_multiple_of(width) {
            return Err("MiMo NVFP4 encode input is not complete rows".into());
        }
        let rows = input.len() / width;
        let output_bytes = packed_extent(rows, width)?;
        self.gpu.ctx.bind_to_thread()?;
        let stream = self.stream();
        if input.ordinal() != stream.context().ordinal() {
            return Err("MiMo NVFP4 encode input crossed GPU devices".into());
        }
        let mut output = self.alloc_u8_uninit(output_bytes)?;
        let rc = unsafe {
            memra_mimo_kv_nvfp4_encode_f32(
                input.device_ptr(&stream).0 as *const f32,
                input.len(),
                output.device_ptr_mut(&stream).0 as *mut u8,
                output_bytes,
                rows,
                width as i32,
                stream.cu_stream() as *mut c_void,
            )
        };
        if rc != 0 {
            return Err(format!("MiMo NVFP4 GPU encode returned {rc}").into());
        }
        stream.synchronize()?;
        Ok(output)
    }

    /// Decode a complete NVFP4 row set for deterministic byte/number gates.
    pub fn mimo_nvfp4_decode_rows(
        &self,
        input: &CudaSlice<u8>,
        rows: usize,
        width: usize,
    ) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        let packed_bytes = packed_extent(rows, width)?;
        if input.len() != packed_bytes {
            return Err("MiMo NVFP4 decode input extent mismatch".into());
        }
        self.gpu.ctx.bind_to_thread()?;
        let stream = self.stream();
        if input.ordinal() != stream.context().ordinal() {
            return Err("MiMo NVFP4 decode input crossed GPU devices".into());
        }
        let output_elements = rows * width;
        let mut output = self.uninit(output_elements)?;
        let rc = unsafe {
            memra_mimo_kv_nvfp4_decode_f32(
                input.device_ptr(&stream).0 as *const u8,
                packed_bytes,
                output.device_ptr_mut(&stream).0 as *mut f32,
                output_elements,
                rows,
                width as i32,
                stream.cu_stream() as *mut c_void,
            )
        };
        if rc != 0 {
            return Err(format!("MiMo NVFP4 GPU decode returned {rc}").into());
        }
        stream.synchronize()?;
        Ok(output)
    }
}
