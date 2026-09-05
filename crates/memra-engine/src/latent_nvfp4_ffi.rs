//! Native compressed latent storage primitives. No environment-selected dispatch.
use crate::Engine;
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
pub use memra_kv::latent_nvfp4::DevicePlane as Nvfp4LatentStorage;
use std::ffi::c_void;

type Res<T> = Result<T, Box<dyn std::error::Error>>;

unsafe extern "C" {
    fn memra_latent_nvfp4_append_live(
        payload: *mut u8,
        scales: *mut u8,
        macros: *mut f32,
        rows: *const f32,
        error: *mut i32,
        pos: *const i32,
        count: i32,
        width: i32,
        capacity: i32,
        stream: *mut c_void,
    ) -> i32;
    fn memra_mla_attn_gathered_nvfp4_live(
        query: *const f32,
        payload: *const u8,
        scales: *const u8,
        macros: *const f32,
        error: *mut i32,
        indices: *const i32,
        slots: *const i32,
        pos: *const i32,
        out: *mut f32,
        heads: i32,
        rank: i32,
        queries: i32,
        stride: i32,
        capacity: i32,
        scale: f32,
        stream: *mut c_void,
    ) -> i32;
    fn memra_latent_nvfp4_to_bf16(
        payload: *const u8,
        scales: *const u8,
        macros: *const f32,
        out: *mut u16,
        error: *mut i32,
        visible: i32,
        width: i32,
        stream: *mut c_void,
    ) -> i32;
    fn memra_mla_attn_gathered_nvfp4(
        q_lat: *const f32,
        q_pe: *const f32,
        payload: *const u8,
        scales: *const u8,
        macros: *const f32,
        error: *mut i32,
        positions: *const i32,
        out: *mut f32,
        heads: i32,
        rank: i32,
        queries: i32,
        slots: i32,
        visible: i32,
        scale: f32,
        stream: *mut c_void,
    ) -> i32;
    fn memra_latent_nvfp4_append(
        payload: *mut u8,
        scales: *mut u8,
        macros: *mut f32,
        rows: *const f32,
        error: *mut i32,
        slot: i32,
        count: i32,
        width: i32,
        capacity: i32,
        stream: *mut c_void,
    ) -> i32;
    fn memra_latent_nvfp4_gather(
        payload: *const u8,
        scales: *const u8,
        macros: *const f32,
        positions: *const i32,
        out: *mut f32,
        error: *mut i32,
        count: i32,
        width: i32,
        visible: i32,
        stream: *mut c_void,
    ) -> i32;
}

pub trait Nvfp4LatentOps {
    fn append_live(&mut self, e: &Engine, rows: &CudaSlice<f32>, pos: &CudaSlice<i32>) -> Res<()>;
    #[allow(clippy::too_many_arguments)] // mirrors native live attention geometry
    fn attend_live(
        &mut self,
        e: &Engine,
        query: &CudaSlice<f32>,
        positions: &CudaSlice<i32>,
        slots: &CudaSlice<i32>,
        pos: &CudaSlice<i32>,
        heads: usize,
        queries: usize,
        scale: f32,
    ) -> Res<CudaSlice<f32>>;
    fn bf16_history(&mut self, e: &Engine, visible: usize) -> Res<CudaSlice<u8>>;
    #[allow(clippy::too_many_arguments)] // mirrors native attention geometry
    fn attend(
        &mut self,
        e: &Engine,
        query: &CudaSlice<f32>,
        positions: &CudaSlice<i32>,
        heads: usize,
        queries: usize,
        slots: usize,
        visible: usize,
        scale: f32,
    ) -> Res<CudaSlice<f32>>;
    fn append(&mut self, e: &Engine, rows: &CudaSlice<f32>, slot: usize) -> Res<()>;
    fn gather(
        &mut self,
        e: &Engine,
        positions: &CudaSlice<i32>,
        visible: usize,
    ) -> Res<CudaSlice<f32>>;
    fn check(&self, e: &Engine) -> Res<()>;
}

impl Nvfp4LatentOps for Nvfp4LatentStorage {
    fn append_live(&mut self, e: &Engine, rows: &CudaSlice<f32>, pos: &CudaSlice<i32>) -> Res<()> {
        self.validate()?;
        let (width, capacity) = (self.width(), self.capacity());
        if pos.is_empty() || !rows.len().is_multiple_of(width) || rows.len() / width > capacity {
            return Err("invalid live NVFP4 append geometry".into());
        }
        let count = i32::try_from(rows.len() / width)?;
        let s = e.stream();
        let mut status = self.error.write_buffer()?;
        let rc = unsafe {
            memra_latent_nvfp4_append_live(
                self.payload.device_ptr_mut(&s).0 as *mut u8,
                self.scales.device_ptr_mut(&s).0 as *mut u8,
                self.macros.device_ptr_mut(&s).0 as *mut f32,
                rows.device_ptr(&s).0 as *const f32,
                status.buffer.device_ptr_mut(&s).0 as *mut i32,
                pos.device_ptr(&s).0 as *const i32,
                count,
                width as i32,
                capacity as i32,
                s.cu_stream() as *mut c_void,
            )
        };
        if rc != 0 {
            return Err(format!("live NVFP4 append launch failed: {rc}").into());
        }
        Ok(())
    }
    fn attend_live(
        &mut self,
        e: &Engine,
        query: &CudaSlice<f32>,
        positions: &CudaSlice<i32>,
        slots: &CudaSlice<i32>,
        pos: &CudaSlice<i32>,
        heads: usize,
        queries: usize,
        scale: f32,
    ) -> Res<CudaSlice<f32>> {
        self.validate()?;
        let (width, capacity) = (self.width(), self.capacity());
        let elems = queries
            .checked_mul(heads)
            .and_then(|n| n.checked_mul(width))
            .ok_or("live attention size overflow")?;
        if heads == 0
            || queries == 0
            || query.len() != elems
            || positions.is_empty()
            || !positions.len().is_multiple_of(queries)
            || slots.len() != 1
            || pos.is_empty()
            || !scale.is_finite()
        {
            return Err("invalid live NVFP4 attention geometry".into());
        }
        let (heads, queries, stride) = (
            i32::try_from(heads)?,
            i32::try_from(queries)?,
            i32::try_from(positions.len() / queries)?,
        );
        let mut out = e.uninit(elems)?;
        let s = e.stream();
        let mut status = self.error.write_buffer()?;
        let rc = unsafe {
            memra_mla_attn_gathered_nvfp4_live(
                query.device_ptr(&s).0 as *const f32,
                self.payload.device_ptr(&s).0 as *const u8,
                self.scales.device_ptr(&s).0 as *const u8,
                self.macros.device_ptr(&s).0 as *const f32,
                status.buffer.device_ptr_mut(&s).0 as *mut i32,
                positions.device_ptr(&s).0 as *const i32,
                slots.device_ptr(&s).0 as *const i32,
                pos.device_ptr(&s).0 as *const i32,
                out.device_ptr_mut(&s).0 as *mut f32,
                heads,
                width as i32,
                queries,
                stride,
                capacity as i32,
                scale,
                s.cu_stream() as *mut c_void,
            )
        };
        if rc != 0 {
            return Err(format!("live NVFP4 attention launch failed: {rc}").into());
        }
        Ok(out)
    }
    fn bf16_history(&mut self, e: &Engine, visible: usize) -> Res<CudaSlice<u8>> {
        self.validate()?;
        if visible > self.capacity() {
            return Err("latent BF16 operand exceeds capacity".into());
        }
        let width = self.width();
        let bytes = visible
            .checked_mul(width)
            .and_then(|n| n.checked_mul(2))
            .ok_or("latent BF16 operand overflow")?;
        let mut out = e.alloc_u8_uninit(bytes)?;
        let s = e.stream();
        let mut status = self.error.write_buffer()?;
        let rc = unsafe {
            memra_latent_nvfp4_to_bf16(
                self.payload.device_ptr(&s).0 as *const u8,
                self.scales.device_ptr(&s).0 as *const u8,
                self.macros.device_ptr(&s).0 as *const f32,
                out.device_ptr_mut(&s).0 as *mut u16,
                status.buffer.device_ptr_mut(&s).0 as *mut i32,
                visible as i32,
                width as i32,
                s.cu_stream() as *mut c_void,
            )
        };
        if rc != 0 {
            return Err(format!("NVFP4 latent BF16 conversion failed: {rc}").into());
        }
        Ok(out)
    }
    fn attend(
        &mut self,
        e: &Engine,
        query: &CudaSlice<f32>,
        positions: &CudaSlice<i32>,
        heads: usize,
        queries: usize,
        slots: usize,
        visible: usize,
        scale: f32,
    ) -> Res<CudaSlice<f32>> {
        self.validate()?;
        let width = self.width();
        let expected = queries
            .checked_mul(heads)
            .and_then(|n| n.checked_mul(width))
            .ok_or("latent attention shape overflow")?;
        let selected = queries
            .checked_mul(slots)
            .ok_or("latent index shape overflow")?;
        if heads == 0
            || queries == 0
            || slots == 0
            || visible == 0
            || visible > self.capacity()
            || query.len() != expected
            || positions.len() != selected
            || !scale.is_finite()
        {
            return Err("invalid NVFP4 latent attention geometry".into());
        }
        let (heads, queries, slots, visible) = (
            i32::try_from(heads)?,
            i32::try_from(queries)?,
            i32::try_from(slots)?,
            i32::try_from(visible)?,
        );
        let mut out = e.uninit(expected)?;
        let s = e.stream();
        let mut status = self.error.write_buffer()?;
        let rc = unsafe {
            memra_mla_attn_gathered_nvfp4(
                query.device_ptr(&s).0 as *const f32,
                std::ptr::null(),
                self.payload.device_ptr(&s).0 as *const u8,
                self.scales.device_ptr(&s).0 as *const u8,
                self.macros.device_ptr(&s).0 as *const f32,
                status.buffer.device_ptr_mut(&s).0 as *mut i32,
                positions.device_ptr(&s).0 as *const i32,
                out.device_ptr_mut(&s).0 as *mut f32,
                heads,
                width as i32,
                queries,
                slots,
                visible,
                scale,
                s.cu_stream() as *mut c_void,
            )
        };
        if rc != 0 {
            return Err(format!("NVFP4 latent attention launch failed: {rc}").into());
        }
        Ok(out)
    }
    fn append(&mut self, e: &Engine, rows: &CudaSlice<f32>, slot: usize) -> Res<()> {
        self.validate()?;
        if !rows.len().is_multiple_of(self.width()) {
            return Err("partial latent input row".into());
        }
        let count = rows.len() / self.width();
        if slot > self.capacity() || count > self.capacity() - slot {
            return Err("latent append exceeds capacity".into());
        }
        let (width, capacity) = (self.width() as i32, self.capacity() as i32);
        let s = e.stream();
        let mut status = self.error.write_buffer()?;
        let rc = unsafe {
            memra_latent_nvfp4_append(
                self.payload.device_ptr_mut(&s).0 as *mut u8,
                self.scales.device_ptr_mut(&s).0 as *mut u8,
                self.macros.device_ptr_mut(&s).0 as *mut f32,
                rows.device_ptr(&s).0 as *const f32,
                status.buffer.device_ptr_mut(&s).0 as *mut i32,
                i32::try_from(slot)?,
                i32::try_from(count)?,
                width,
                capacity,
                s.cu_stream() as *mut c_void,
            )
        };
        if rc != 0 {
            return Err(format!("NVFP4 latent append launch failed: {rc}").into());
        }
        Ok(())
    }

    fn gather(
        &mut self,
        e: &Engine,
        positions: &CudaSlice<i32>,
        visible: usize,
    ) -> Res<CudaSlice<f32>> {
        self.validate()?;
        if visible > self.capacity() {
            return Err("latent visible length exceeds capacity".into());
        }
        let count = i32::try_from(positions.len())?;
        let width = self.width() as i32;
        let mut out = e.uninit(
            positions
                .len()
                .checked_mul(self.width())
                .ok_or("latent gather overflow")?,
        )?;
        let s = e.stream();
        let mut status = self.error.write_buffer()?;
        let rc = unsafe {
            memra_latent_nvfp4_gather(
                self.payload.device_ptr(&s).0 as *const u8,
                self.scales.device_ptr(&s).0 as *const u8,
                self.macros.device_ptr(&s).0 as *const f32,
                positions.device_ptr(&s).0 as *const i32,
                out.device_ptr_mut(&s).0 as *mut f32,
                status.buffer.device_ptr_mut(&s).0 as *mut i32,
                count,
                width,
                visible as i32,
                s.cu_stream() as *mut c_void,
            )
        };
        if rc != 0 {
            return Err(format!("NVFP4 latent gather launch failed: {rc}").into());
        }
        Ok(out)
    }

    fn check(&self, _e: &Engine) -> Res<()> {
        self.validate()?;
        self.error.invalidate()?;
        self.error.check()
    }
}
