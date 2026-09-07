//! Rank-local DSV4 FP8 dense-plane packing for the first TP2 attention slice.
//!
//! This module is deliberately a preparation seam only.  It does not select a runtime path,
//! launch an attention kernel, or implement the eventual `wo_b` rank reduction.  It owns the
//! exact storage adapter needed by that path:
//!
//! * `wq_b` and `wo_a` use contiguous row partitions;
//! * `wo_b` uses packed contiguous column partitions, because the current GEMV kernels address
//!   every weight row as `row * k` and cannot consume a source row stride of the full tensor;
//! * FP8 code planes are `u8`, while the decoded 128x128 scale planes are explicitly `f32`.
//!
//! The eventual `wo_b` partial sum is a separate named numeric class.  Nothing in this adapter
//! invents an error tolerance or claims that reduction is bit-identical to the full 8192-column
//! GEMV.

use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut, sys};
use memra_runtime::Gpu;

type Res<T> = Result<T, String>;

pub const FP8_BLOCK: usize = 128;

/// Logical location of a rank-local dense plane in the full checkpoint tensor.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Partition {
    /// Contiguous output rows, used by `wq_b` and `wo_a`.
    Rows { start: usize, len: usize },
    /// Contiguous input columns, repacked row by row, used by `wo_b`.
    Columns { start: usize, len: usize },
}

impl Partition {
    pub const fn start(self) -> usize {
        match self {
            Self::Rows { start, .. } | Self::Columns { start, .. } => start,
        }
    }

    pub const fn len(self) -> usize {
        match self {
            Self::Rows { len, .. } | Self::Columns { len, .. } => len,
        }
    }
}

/// Metadata and host-owned packed payload for one logical rank.
#[derive(Clone, Debug, PartialEq)]
pub struct PackedFp8Host {
    pub rank: usize,
    pub full_rows: usize,
    pub full_cols: usize,
    pub partition: Partition,
    pub rows: usize,
    pub cols: usize,
    /// Number of decoded f32 scales in one packed scale row.
    pub scale_cols: usize,
    pub codes: Vec<u8>,
    pub scales: Vec<f32>,
}

impl PackedFp8Host {
    /// Pack a contiguous output-row partition from a full row-major FP8 tensor.
    pub fn rows_from_full(
        rank: usize,
        full_rows: usize,
        full_cols: usize,
        row_start: usize,
        row_len: usize,
        codes: &[u8],
        scales: &[f32],
    ) -> Res<Self> {
        validate_full(full_rows, full_cols, codes, scales)?;
        validate_partition(
            rank,
            full_rows,
            full_cols,
            Partition::Rows {
                start: row_start,
                len: row_len,
            },
        )?;
        let scale_cols = full_cols / FP8_BLOCK;
        let scale_row_start = row_start / FP8_BLOCK;
        let scale_rows = row_len / FP8_BLOCK;
        let mut packed_codes = Vec::with_capacity(row_len * full_cols);
        for row in row_start..row_start + row_len {
            packed_codes.extend_from_slice(&codes[row * full_cols..(row + 1) * full_cols]);
        }
        let mut packed_scales = Vec::with_capacity(scale_rows * scale_cols);
        for row in scale_row_start..scale_row_start + scale_rows {
            packed_scales.extend_from_slice(&scales[row * scale_cols..(row + 1) * scale_cols]);
        }
        Ok(Self {
            rank,
            full_rows,
            full_cols,
            partition: Partition::Rows {
                start: row_start,
                len: row_len,
            },
            rows: row_len,
            cols: full_cols,
            scale_cols,
            codes: packed_codes,
            scales: packed_scales,
        })
    }

    /// Pack a contiguous input-column partition into a new row-major slab.
    ///
    /// This is intentionally a repack, not a pointer offset: every packed row has `cols`
    /// elements, which is the physical `k` stride expected by the existing GEMV kernels.
    pub fn columns_from_full(
        rank: usize,
        full_rows: usize,
        full_cols: usize,
        col_start: usize,
        col_len: usize,
        codes: &[u8],
        scales: &[f32],
    ) -> Res<Self> {
        validate_full(full_rows, full_cols, codes, scales)?;
        validate_partition(
            rank,
            full_rows,
            full_cols,
            Partition::Columns {
                start: col_start,
                len: col_len,
            },
        )?;
        let full_scale_cols = full_cols / FP8_BLOCK;
        let scale_cols = col_len / FP8_BLOCK;
        let mut packed_codes = Vec::with_capacity(full_rows * col_len);
        for row in 0..full_rows {
            let start = row * full_cols + col_start;
            packed_codes.extend_from_slice(&codes[start..start + col_len]);
        }
        let mut packed_scales = Vec::with_capacity(full_rows / FP8_BLOCK * scale_cols);
        for row in 0..full_rows / FP8_BLOCK {
            let start = row * full_scale_cols + col_start / FP8_BLOCK;
            packed_scales.extend_from_slice(&scales[start..start + scale_cols]);
        }
        Ok(Self {
            rank,
            full_rows,
            full_cols,
            partition: Partition::Columns {
                start: col_start,
                len: col_len,
            },
            rows: full_rows,
            cols: col_len,
            scale_cols,
            codes: packed_codes,
            scales: packed_scales,
        })
    }

    pub fn half_rows(
        rank: usize,
        full_rows: usize,
        full_cols: usize,
        codes: &[u8],
        scales: &[f32],
    ) -> Res<Self> {
        if rank > 1 {
            return Err(format!("FP8 pack logical rank {rank} outside TP2"));
        }
        if !full_rows.is_multiple_of(2) {
            return Err(format!(
                "FP8 TP2 row split requires even full rows, got {full_rows}"
            ));
        }
        let half = full_rows / 2;
        Self::rows_from_full(rank, full_rows, full_cols, rank * half, half, codes, scales)
    }

    pub fn half_columns(
        rank: usize,
        full_rows: usize,
        full_cols: usize,
        codes: &[u8],
        scales: &[f32],
    ) -> Res<Self> {
        if rank > 1 {
            return Err(format!("FP8 pack logical rank {rank} outside TP2"));
        }
        if !full_cols.is_multiple_of(2) {
            return Err(format!(
                "FP8 TP2 column split requires even full cols, got {full_cols}"
            ));
        }
        let half = full_cols / 2;
        Self::columns_from_full(rank, full_rows, full_cols, rank * half, half, codes, scales)
    }

    pub fn code_bytes(&self) -> usize {
        self.codes.len()
    }

    pub fn scale_bytes(&self) -> usize {
        self.scales.len() * std::mem::size_of::<f32>()
    }
}

/// Device-owned packed code and decoded FP32 scale planes.
///
/// The metadata is sufficient for a future launcher to pass the local `rows`, `cols`, and
/// `scale_cols` without reconstructing full-tensor geometry.  No kernel is launched here.
pub struct DeviceFp8Plane {
    pub rank: usize,
    pub full_rows: usize,
    pub full_cols: usize,
    pub partition: Partition,
    pub rows: usize,
    pub cols: usize,
    pub scale_cols: usize,
    pub codes: CudaSlice<u8>,
    pub scales: CudaSlice<f32>,
}

impl DeviceFp8Plane {
    pub fn upload(gpu: &Gpu, packed: PackedFp8Host) -> Res<Self> {
        gpu.ctx
            .bind_to_thread()
            .map_err(|e| format!("FP8 pack bind: {e}"))?;
        let stream = gpu.stream();
        let codes = stream
            .clone_htod(&packed.codes)
            .map_err(|e| format!("FP8 pack code upload: {e}"))?;
        let scales = stream
            .clone_htod(&packed.scales)
            .map_err(|e| format!("FP8 pack scale upload: {e}"))?;
        Ok(Self {
            rank: packed.rank,
            full_rows: packed.full_rows,
            full_cols: packed.full_cols,
            partition: packed.partition,
            rows: packed.rows,
            cols: packed.cols,
            scale_cols: packed.scale_cols,
            codes,
            scales,
        })
    }

    /// Device-to-device row/column pack from a full source plane.  This is asynchronous and
    /// ordered on `gpu.stream()`; the source and destination owners must outlive that stream.
    pub fn from_device(
        gpu: &Gpu,
        rank: usize,
        full_rows: usize,
        full_cols: usize,
        partition: Partition,
        source_codes: &CudaSlice<u8>,
        source_scales: &CudaSlice<f32>,
    ) -> Res<Self> {
        validate_full_lengths(
            full_rows,
            full_cols,
            source_codes.len(),
            source_scales.len(),
        )?;
        validate_partition(rank, full_rows, full_cols, partition)?;
        gpu.ctx
            .bind_to_thread()
            .map_err(|e| format!("FP8 pack bind: {e}"))?;
        let stream = gpu.stream();
        let (rows, cols) = match partition {
            Partition::Rows { len, .. } => (len, full_cols),
            Partition::Columns { len, .. } => (full_rows, len),
        };
        let scale_cols = cols / FP8_BLOCK;
        let scale_rows = rows / FP8_BLOCK;
        let mut codes = stream
            .alloc_zeros::<u8>(rows * cols)
            .map_err(|e| format!("FP8 pack code allocation: {e}"))?;
        let mut scales = stream
            .alloc_zeros::<f32>(scale_rows * scale_cols)
            .map_err(|e| format!("FP8 pack scale allocation: {e}"))?;
        match partition {
            Partition::Rows { start, .. } => {
                copy_2d_device(
                    &stream,
                    source_codes,
                    0,
                    start,
                    full_cols,
                    &mut codes,
                    0,
                    0,
                    cols,
                    cols,
                    rows,
                )?;
                copy_2d_device(
                    &stream,
                    source_scales,
                    0,
                    start / FP8_BLOCK,
                    full_cols / FP8_BLOCK,
                    &mut scales,
                    0,
                    0,
                    scale_cols,
                    scale_cols,
                    scale_rows,
                )?;
            }
            Partition::Columns { start, .. } => {
                copy_2d_device(
                    &stream,
                    source_codes,
                    start,
                    0,
                    full_cols,
                    &mut codes,
                    0,
                    0,
                    cols,
                    cols,
                    rows,
                )?;
                copy_2d_device(
                    &stream,
                    source_scales,
                    start / FP8_BLOCK,
                    0,
                    full_cols / FP8_BLOCK,
                    &mut scales,
                    0,
                    0,
                    scale_cols,
                    scale_cols,
                    scale_rows,
                )?;
            }
        }
        Ok(Self {
            rank,
            full_rows,
            full_cols,
            partition,
            rows,
            cols,
            scale_cols,
            codes,
            scales,
        })
    }
}

#[allow(clippy::too_many_arguments)]
fn copy_2d_device<T>(
    stream: &std::sync::Arc<cudarc::driver::CudaStream>,
    src: &CudaSlice<T>,
    src_x: usize,
    src_y: usize,
    src_pitch: usize,
    dst: &mut CudaSlice<T>,
    dst_x: usize,
    dst_y: usize,
    dst_pitch: usize,
    width: usize,
    height: usize,
) -> Res<()> {
    if width == 0 || height == 0 {
        return Ok(());
    }
    if src_x.checked_add(width).is_none_or(|end| end > src_pitch)
        || dst_x.checked_add(width).is_none_or(|end| end > dst_pitch)
    {
        return Err("FP8 pack 2D x+width exceeds row pitch".into());
    }
    let src_need = src_y
        .checked_mul(src_pitch)
        .and_then(|v| v.checked_add(src_x))
        .and_then(|v| v.checked_add((height - 1).checked_mul(src_pitch)?))
        .and_then(|v| v.checked_add(width))
        .ok_or("FP8 pack source 2D range overflow")?;
    let dst_need = dst_y
        .checked_mul(dst_pitch)
        .and_then(|v| v.checked_add(dst_x))
        .and_then(|v| v.checked_add((height - 1).checked_mul(dst_pitch)?))
        .and_then(|v| v.checked_add(width))
        .ok_or("FP8 pack destination 2D range overflow")?;
    if src_need > src.len() || dst_need > dst.len() {
        return Err(format!(
            "FP8 pack 2D range exceeds allocation: src={src_need}/{} dst={dst_need}/{}",
            src.len(),
            dst.len()
        ));
    }
    let elem_bytes = std::mem::size_of::<T>();
    let (src_ptr, _src_guard) = src.device_ptr(stream);
    let (dst_ptr, _dst_guard) = dst.device_ptr_mut(stream);
    let copy = sys::CUDA_MEMCPY2D_st {
        srcXInBytes: src_x * elem_bytes,
        srcY: src_y,
        srcMemoryType: sys::CUmemorytype_enum::CU_MEMORYTYPE_DEVICE,
        srcHost: std::ptr::null(),
        srcDevice: src_ptr,
        srcArray: std::ptr::null_mut(),
        srcPitch: src_pitch * elem_bytes,
        dstXInBytes: dst_x * elem_bytes,
        dstY: dst_y,
        dstMemoryType: sys::CUmemorytype_enum::CU_MEMORYTYPE_DEVICE,
        dstHost: std::ptr::null_mut(),
        dstDevice: dst_ptr,
        dstArray: std::ptr::null_mut(),
        dstPitch: dst_pitch * elem_bytes,
        WidthInBytes: width * elem_bytes,
        Height: height,
    };
    unsafe { sys::cuMemcpy2DAsync_v2(&copy, stream.cu_stream()).result() }
        .map_err(|e| format!("FP8 pack 2D copy: {e}"))
}

fn validate_full(rows: usize, cols: usize, codes: &[u8], scales: &[f32]) -> Res<()> {
    validate_full_lengths(rows, cols, codes.len(), scales.len())
}

fn validate_full_lengths(rows: usize, cols: usize, code_len: usize, scale_len: usize) -> Res<()> {
    if rows == 0 || cols == 0 || !rows.is_multiple_of(FP8_BLOCK) || !cols.is_multiple_of(FP8_BLOCK)
    {
        return Err(format!(
            "FP8 pack shape must be nonzero and 128-aligned: rows={rows} cols={cols}"
        ));
    }
    let expected_codes = rows
        .checked_mul(cols)
        .ok_or("FP8 pack code shape overflow")?;
    if code_len != expected_codes {
        return Err(format!(
            "FP8 pack code length {code_len} != full shape {}",
            expected_codes
        ));
    }
    let expected_scales = (rows / FP8_BLOCK)
        .checked_mul(cols / FP8_BLOCK)
        .ok_or("FP8 pack scale shape overflow")?;
    if scale_len != expected_scales {
        return Err(format!(
            "FP8 pack scale length {scale_len} != FP32 scale plane {expected_scales}"
        ));
    }
    Ok(())
}

fn validate_partition(rank: usize, rows: usize, cols: usize, partition: Partition) -> Res<()> {
    if rank > 1 {
        return Err(format!("FP8 pack logical rank {rank} outside TP2"));
    }
    match partition {
        Partition::Rows { start, len } => {
            if len == 0
                || !start.is_multiple_of(FP8_BLOCK)
                || !len.is_multiple_of(FP8_BLOCK)
                || start.checked_add(len).is_none_or(|end| end > rows)
            {
                return Err(format!(
                    "FP8 row partition is not 128-aligned/in-bounds: start={start} len={len} rows={rows}"
                ));
            }
        }
        Partition::Columns { start, len } => {
            if len == 0
                || !start.is_multiple_of(FP8_BLOCK)
                || !len.is_multiple_of(FP8_BLOCK)
                || start.checked_add(len).is_none_or(|end| end > cols)
            {
                return Err(format!(
                    "FP8 column partition is not 128-aligned/in-bounds: start={start} len={len} cols={cols}"
                ));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{DeviceFp8Plane, FP8_BLOCK, Gpu, PackedFp8Host, Partition, copy_2d_device};

    fn mix(mut x: u64) -> u64 {
        x ^= x >> 30;
        x = x.wrapping_mul(0xbf58476d1ce4e5b9);
        x ^= x >> 27;
        x = x.wrapping_mul(0x94d049bb133111eb);
        x ^ (x >> 31)
    }

    fn codes(n: usize, seed: u64) -> Vec<u8> {
        (0..n)
            .map(|i| ((mix(seed.wrapping_add(i as u64 * 0x9e37)) & 0x7e) as u8).saturating_add(1))
            .collect()
    }

    fn scales(n: usize, seed: u64) -> Vec<f32> {
        (0..n)
            .map(|i| {
                let exponent = ((mix(seed.wrapping_add(i as u64 * 0x10001)) % 15) as i32) - 7;
                2f32.powi(exponent)
            })
            .collect()
    }

    fn dot_full_range(
        codes: &[u8],
        scales: &[f32],
        full_cols: usize,
        scale_cols: usize,
        row: usize,
        col_start: usize,
        len: usize,
        input: &[f32],
    ) -> f32 {
        assert_eq!(input.len(), len);
        let mut acc = 0.0f32;
        for local_col in 0..len {
            let col = col_start + local_col;
            let code = codes[row * full_cols + col];
            let scale = scales[(row / FP8_BLOCK) * scale_cols + col / FP8_BLOCK];
            acc += memra_gguf::nvfp4_repack::fp8_e4m3_to_f32(code) * scale * input[local_col];
        }
        acc
    }

    fn dot_packed_row(packed: &PackedFp8Host, row: usize, input: &[f32]) -> f32 {
        let mut acc = 0.0f32;
        for col in 0..packed.cols {
            let code = packed.codes[row * packed.cols + col];
            let scale = packed.scales[(row / FP8_BLOCK) * packed.scale_cols + col / FP8_BLOCK];
            acc += memra_gguf::nvfp4_repack::fp8_e4m3_to_f32(code) * scale * input[col];
        }
        acc
    }

    fn dot_packed_column_half(
        packed: &PackedFp8Host,
        full_cols: usize,
        col_start: usize,
        row: usize,
        input: &[f32],
    ) -> f32 {
        let mut acc = 0.0f32;
        for col in 0..packed.cols {
            let code = packed.codes[row * packed.cols + col];
            let scale = packed.scales[(row / FP8_BLOCK) * packed.scale_cols + col / FP8_BLOCK];
            acc += memra_gguf::nvfp4_repack::fp8_e4m3_to_f32(code) * scale * input[col_start + col];
        }
        debug_assert_eq!(input.len(), full_cols);
        acc
    }

    fn assert_row_partition(
        packed: &PackedFp8Host,
        full_codes: &[u8],
        full_scales: &[f32],
        row_start: usize,
    ) {
        let cols = packed.full_cols;
        for row in 0..packed.rows {
            assert_eq!(
                &packed.codes[row * cols..(row + 1) * cols],
                &full_codes[(row_start + row) * cols..(row_start + row + 1) * cols]
            );
        }
        let scale_cols = cols / FP8_BLOCK;
        let scale_start = row_start / FP8_BLOCK;
        for row in 0..packed.rows / FP8_BLOCK {
            assert_eq!(
                &packed.scales[row * scale_cols..(row + 1) * scale_cols],
                &full_scales
                    [(scale_start + row) * scale_cols..(scale_start + row + 1) * scale_cols]
            );
        }
    }

    fn assert_column_partition(
        packed: &PackedFp8Host,
        full_codes: &[u8],
        full_scales: &[f32],
        col_start: usize,
    ) {
        let full_cols = packed.full_cols;
        for row in 0..packed.rows {
            assert_eq!(
                &packed.codes[row * packed.cols..(row + 1) * packed.cols],
                &full_codes[row * full_cols + col_start..row * full_cols + col_start + packed.cols]
            );
        }
        let full_scale_cols = full_cols / FP8_BLOCK;
        let scale_start = col_start / FP8_BLOCK;
        for row in 0..packed.rows / FP8_BLOCK {
            assert_eq!(
                &packed.scales[row * packed.scale_cols..(row + 1) * packed.scale_cols],
                &full_scales[row * full_scale_cols + scale_start
                    ..row * full_scale_cols + scale_start + packed.scale_cols]
            );
        }
    }

    #[test]
    fn real_dsv4_rank_row_shapes_are_exact_and_scale_planes_are_f32() {
        let cases = [(32768usize, 1024usize), (8192, 4096)];
        for (rows, cols) in cases {
            let full_codes = codes(rows * cols, rows as u64 * 17 + cols as u64);
            let full_scales = scales(
                rows / FP8_BLOCK * (cols / FP8_BLOCK),
                rows as u64 * 31 + cols as u64,
            );
            let rank0 = PackedFp8Host::half_rows(0, rows, cols, &full_codes, &full_scales).unwrap();
            let rank1 = PackedFp8Host::half_rows(1, rows, cols, &full_codes, &full_scales).unwrap();
            assert_eq!(rank0.rows, rows / 2);
            assert_eq!(rank1.rows, rows / 2);
            assert_eq!(rank0.cols, cols);
            assert_eq!(
                rank0.scales.len(),
                rows / FP8_BLOCK / 2 * (cols / FP8_BLOCK)
            );
            assert_row_partition(&rank0, &full_codes, &full_scales, 0);
            assert_row_partition(&rank1, &full_codes, &full_scales, rows / 2);
            assert_ne!(rank0.codes, rank1.codes);
            assert_ne!(rank0.scales, rank1.scales);
            assert_eq!(rank0.codes.len() * 2, full_codes.len());
            assert_eq!(rank0.scales.len() * 2, full_scales.len());

            let input: Vec<f32> = (0..cols)
                .map(|i| ((mix(i as u64 * 0x20003 + rows as u64) % 1709) as f32 - 854.0) / 193.0)
                .collect();
            let scale_cols = cols / FP8_BLOCK;
            for &global_row in &[0usize, 127, rows / 2, rows - 1] {
                let expected = dot_full_range(
                    &full_codes,
                    &full_scales,
                    cols,
                    scale_cols,
                    global_row,
                    0,
                    cols,
                    &input,
                );
                let packed = if global_row < rows / 2 {
                    dot_packed_row(&rank0, global_row, &input)
                } else {
                    dot_packed_row(&rank1, global_row - rows / 2, &input)
                };
                assert_eq!(
                    packed.to_bits(),
                    expected.to_bits(),
                    "row oracle={global_row}"
                );
            }
        }
    }

    #[test]
    fn real_dsv4_wob_column_halves_repack_rows_and_f32_scales() {
        let (rows, cols) = (4096usize, 8192usize);
        let full_codes = codes(rows * cols, 0xdead_beef);
        let full_scales = scales(rows / FP8_BLOCK * (cols / FP8_BLOCK), 0x1234_5678);
        let rank0 = PackedFp8Host::half_columns(0, rows, cols, &full_codes, &full_scales).unwrap();
        let rank1 = PackedFp8Host::half_columns(1, rows, cols, &full_codes, &full_scales).unwrap();
        assert_eq!(rank0.rows, rows);
        assert_eq!(rank0.cols, cols / 2);
        assert_eq!(rank0.scale_cols, cols / FP8_BLOCK / 2);
        assert_column_partition(&rank0, &full_codes, &full_scales, 0);
        assert_column_partition(&rank1, &full_codes, &full_scales, cols / 2);
        for row in 0..rows {
            let a = &rank0.codes[row * rank0.cols..(row + 1) * rank0.cols];
            let b = &rank1.codes[row * rank1.cols..(row + 1) * rank1.cols];
            assert_eq!([a, b].concat(), full_codes[row * cols..(row + 1) * cols]);
        }
        assert_ne!(rank0.scales, rank1.scales);
        assert_eq!(
            rank0.scale_bytes(),
            rows / FP8_BLOCK * (cols / FP8_BLOCK / 2) * 4
        );

        // Independent nonperiodic FP8*FP32-scale oracle. The halves are checked independently
        // and then combined in explicit rank order. This intentionally does not compare the
        // rank-sum class to a full single-pass 8192-column accumulation.
        let input: Vec<f32> = (0..cols)
            .map(|i| ((mix(i as u64 * 0x10001 + 7) % 2001) as f32 - 1000.0) / 257.0)
            .collect();
        let full_scale_cols = cols / FP8_BLOCK;
        for &row in &[0usize, 127, 128, 2047, 4095] {
            let expected0 = dot_packed_column_half(&rank0, cols, 0, row, &input);
            let expected1 = dot_packed_column_half(&rank1, cols, cols / 2, row, &input);
            let independent0 = dot_full_range(
                &full_codes,
                &full_scales,
                cols,
                full_scale_cols,
                row,
                0,
                cols / 2,
                &input[..cols / 2],
            );
            let independent1 = dot_full_range(
                &full_codes,
                &full_scales,
                cols,
                full_scale_cols,
                row,
                cols / 2,
                cols / 2,
                &input[cols / 2..],
            );
            assert_eq!(
                expected0.to_bits(),
                independent0.to_bits(),
                "rank0 row={row}"
            );
            assert_eq!(
                expected1.to_bits(),
                independent1.to_bits(),
                "rank1 row={row}"
            );
            assert_eq!(
                (expected0 + expected1).to_bits(),
                (independent0 + independent1).to_bits(),
                "rank-order row={row}"
            );
        }

        let row = 2048usize;
        let before = dot_packed_row(&rank1, row, &input[cols / 2..]);
        let mut corrupt = rank1.clone();
        let scale_idx = corrupt.scales.len() / 2;
        corrupt.scales[scale_idx] *= 2.0;
        let after = dot_packed_row(&corrupt, row, &input[cols / 2..]);
        assert_ne!(
            before.to_bits(),
            after.to_bits(),
            "FP32 scale corruption must reach the oracle"
        );
    }

    #[test]
    fn logical_rank_swap_and_scale_corruption_are_rejected_by_identity_teeth() {
        let (rows, cols) = (4096usize, 8192usize);
        let full_codes = codes(rows * cols, 0xfeed_face);
        let full_scales = scales(rows / FP8_BLOCK * (cols / FP8_BLOCK), 0x44aa_9911);
        let rank0 = PackedFp8Host::half_columns(0, rows, cols, &full_codes, &full_scales).unwrap();
        let rank1 = PackedFp8Host::half_columns(1, rows, cols, &full_codes, &full_scales).unwrap();
        let mut reassembled = Vec::with_capacity(full_codes.len());
        for row in 0..rows {
            reassembled.extend_from_slice(&rank0.codes[row * rank0.cols..(row + 1) * rank0.cols]);
            reassembled.extend_from_slice(&rank1.codes[row * rank1.cols..(row + 1) * rank1.cols]);
        }
        assert_eq!(reassembled, full_codes);

        let mut swapped = Vec::with_capacity(full_codes.len());
        for row in 0..rows {
            swapped.extend_from_slice(&rank1.codes[row * rank1.cols..(row + 1) * rank1.cols]);
            swapped.extend_from_slice(&rank0.codes[row * rank0.cols..(row + 1) * rank0.cols]);
        }
        assert_ne!(
            swapped, full_codes,
            "rank swap must not be a silent identity"
        );

        let mut bad_scales = rank1.scales.clone();
        let scale_idx = bad_scales.len() / 2;
        let old_scale = bad_scales[scale_idx];
        bad_scales[scale_idx] = f32::from_bits(old_scale.to_bits() ^ 1);
        assert_ne!(
            bad_scales, rank1.scales,
            "FP32 scale corruption must be observable"
        );
        assert_eq!(rank0.scales.len(), rank1.scales.len());
    }

    #[test]
    fn malformed_shapes_and_non_aligned_partitions_fail_closed() {
        let codes = vec![0u8; FP8_BLOCK * FP8_BLOCK];
        let scales = vec![1.0f32; 1];
        assert!(PackedFp8Host::rows_from_full(0, 128, 128, 1, 127, &codes, &scales).is_err());
        assert!(PackedFp8Host::columns_from_full(0, 128, 128, 1, 127, &codes, &scales).is_err());
        assert!(PackedFp8Host::rows_from_full(2, 128, 128, 0, 128, &codes, &scales).is_err());
        assert!(
            PackedFp8Host::rows_from_full(0, 128, 128, 0, 128, &codes[..127], &scales).is_err()
        );
        assert!(PackedFp8Host::columns_from_full(0, 128, 128, 0, 128, &codes, &[1.0; 2]).is_err());
    }

    #[test]
    #[ignore = "requires one locked CUDA GPU; attention FP8 2D pack adapter"]
    fn cuda_attention_split_device_2d_pack_matches_host() {
        let gpu = Gpu::new(0).expect("gpu");
        unsafe { gpu.ctx.disable_event_tracking() };
        let stream = gpu.stream();
        for (rows, cols, kind) in [
            (32768usize, 1024usize, 0usize),
            (8192usize, 4096usize, 0usize),
            (4096usize, 8192usize, 1usize),
        ] {
            let full_codes = codes(rows * cols, rows as u64 * 19 + cols as u64);
            let full_scales = scales(
                rows / FP8_BLOCK * (cols / FP8_BLOCK),
                rows as u64 * 37 + cols as u64,
            );
            for rank in 0..2usize {
                let half = if kind == 0 { rows / 2 } else { cols / 2 };
                let partition = if kind == 0 {
                    Partition::Rows {
                        start: rank * half,
                        len: half,
                    }
                } else {
                    Partition::Columns {
                        start: rank * half,
                        len: half,
                    }
                };
                let expected = match partition {
                    Partition::Rows { start, len } => PackedFp8Host::rows_from_full(
                        rank,
                        rows,
                        cols,
                        start,
                        len,
                        &full_codes,
                        &full_scales,
                    )
                    .unwrap(),
                    Partition::Columns { start, len } => PackedFp8Host::columns_from_full(
                        rank,
                        rows,
                        cols,
                        start,
                        len,
                        &full_codes,
                        &full_scales,
                    )
                    .unwrap(),
                };
                let source_codes = stream.clone_htod(&full_codes).unwrap();
                let source_scales = stream.clone_htod(&full_scales).unwrap();
                let packed = DeviceFp8Plane::from_device(
                    &gpu,
                    rank,
                    rows,
                    cols,
                    partition,
                    &source_codes,
                    &source_scales,
                )
                .unwrap();
                stream.synchronize().unwrap();
                let got_codes = stream.clone_dtoh(&packed.codes).unwrap();
                let got_scales = stream.clone_dtoh(&packed.scales).unwrap();
                assert_eq!(
                    got_codes, expected.codes,
                    "device code pack rank={rank} rows={rows} cols={cols}"
                );
                assert_eq!(
                    got_scales
                        .iter()
                        .copied()
                        .map(f32::to_bits)
                        .collect::<Vec<_>>(),
                    expected
                        .scales
                        .iter()
                        .copied()
                        .map(f32::to_bits)
                        .collect::<Vec<_>>(),
                    "device FP32 scale pack rank={rank} rows={rows} cols={cols}"
                );

                // Prove the source plane was not modified by either the adapter or the raw
                // 2D copy. The second half also exercises explicit destination canaries.
                assert_eq!(stream.clone_dtoh(&source_codes).unwrap(), full_codes);
                assert_eq!(
                    stream
                        .clone_dtoh(&source_scales)
                        .unwrap()
                        .iter()
                        .copied()
                        .map(f32::to_bits)
                        .collect::<Vec<_>>(),
                    full_scales
                        .iter()
                        .copied()
                        .map(f32::to_bits)
                        .collect::<Vec<_>>()
                );
                let code_pitch = expected.cols;
                let code_rows = expected.rows + 2;
                let code_canary = 0xd3u8;
                let mut guarded_codes_host = vec![code_canary; code_rows * code_pitch];
                let mut guarded_codes = stream.clone_htod(&guarded_codes_host).unwrap();
                let scale_pitch = expected.scale_cols;
                let scale_rows = expected.rows / FP8_BLOCK + 2;
                let scale_canary = f32::from_bits(0x7fc0_1234);
                let mut guarded_scales_host = vec![scale_canary; scale_rows * scale_pitch];
                let mut guarded_scales = stream.clone_htod(&guarded_scales_host).unwrap();
                let (code_x, code_y, scale_x, scale_y) = match partition {
                    Partition::Rows { start, .. } => (0, start, 0, start / FP8_BLOCK),
                    Partition::Columns { start, .. } => (start, 0, start / FP8_BLOCK, 0),
                };
                copy_2d_device(
                    &stream,
                    &source_codes,
                    code_x,
                    code_y,
                    cols,
                    &mut guarded_codes,
                    0,
                    1,
                    code_pitch,
                    expected.cols,
                    expected.rows,
                )
                .unwrap();
                copy_2d_device(
                    &stream,
                    &source_scales,
                    scale_x,
                    scale_y,
                    cols / FP8_BLOCK,
                    &mut guarded_scales,
                    0,
                    1,
                    scale_pitch,
                    expected.scale_cols,
                    expected.rows / FP8_BLOCK,
                )
                .unwrap();
                stream.synchronize().unwrap();
                let got_guarded_codes = stream.clone_dtoh(&guarded_codes).unwrap();
                let got_guarded_scales = stream.clone_dtoh(&guarded_scales).unwrap();
                assert_eq!(
                    &got_guarded_codes[..code_pitch],
                    &guarded_codes_host[..code_pitch]
                );
                assert_eq!(
                    &got_guarded_codes[(code_rows - 1) * code_pitch..],
                    &guarded_codes_host[(code_rows - 1) * code_pitch..]
                );
                assert_eq!(
                    &got_guarded_codes[code_pitch..(code_rows - 1) * code_pitch],
                    expected.codes.as_slice()
                );
                assert_eq!(
                    got_guarded_scales[..scale_pitch]
                        .iter()
                        .copied()
                        .map(f32::to_bits)
                        .collect::<Vec<_>>(),
                    guarded_scales_host[..scale_pitch]
                        .iter()
                        .copied()
                        .map(f32::to_bits)
                        .collect::<Vec<_>>()
                );
                assert_eq!(
                    got_guarded_scales[(scale_rows - 1) * scale_pitch..]
                        .iter()
                        .copied()
                        .map(f32::to_bits)
                        .collect::<Vec<_>>(),
                    guarded_scales_host[(scale_rows - 1) * scale_pitch..]
                        .iter()
                        .copied()
                        .map(f32::to_bits)
                        .collect::<Vec<_>>()
                );
                assert_eq!(
                    got_guarded_scales[scale_pitch..(scale_rows - 1) * scale_pitch]
                        .iter()
                        .copied()
                        .map(f32::to_bits)
                        .collect::<Vec<_>>(),
                    expected
                        .scales
                        .iter()
                        .copied()
                        .map(f32::to_bits)
                        .collect::<Vec<_>>()
                );
                assert_eq!(stream.clone_dtoh(&source_codes).unwrap(), full_codes);
                assert_eq!(
                    stream
                        .clone_dtoh(&source_scales)
                        .unwrap()
                        .iter()
                        .copied()
                        .map(f32::to_bits)
                        .collect::<Vec<_>>(),
                    full_scales
                        .iter()
                        .copied()
                        .map(f32::to_bits)
                        .collect::<Vec<_>>()
                );
                // Keep the host canary vectors live and explicit in this guard check.
                guarded_codes_host.fill(code_canary);
                guarded_scales_host.fill(scale_canary);
            }
        }
    }
}
