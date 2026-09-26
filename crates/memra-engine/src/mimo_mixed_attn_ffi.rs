//! Source-inspection split attention for MiMo global layers with q8_0 K and NVFP4 V.
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
}

fn extents(seq: usize, program: u8) -> Result<(usize, usize, usize), &'static str> {
    if !(1..=MAX_SEQ).contains(&seq) {
        return Err("MiMo mixed attention accepts 1..=1048576 tokens");
    }
    let tile_size = match program {
        0 => BASE_TILE,
        1 => GROUPED_TILE,
        2..=4 => DEEP_SPLIT,
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
        let (one, two, three) = extents(seq, workspace.program)?;
        if seq > workspace.max_seq
            || query.len() != HEADS * QK
            || key.len() < seq * KV_HEADS * Q8_ROW_BYTES
            || value.len() < seq * KV_HEADS * NVFP4_ROW_BYTES
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
