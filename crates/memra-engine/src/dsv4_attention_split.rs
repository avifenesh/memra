//! Rank-local DSV4 FP8 dense planes for TP2 head-parallel attention.
//!
//! `Dsv4Gpu::load` builds these for every TP/EP load with attention TP, the served
//! two-card default. Each rank packs a contiguous output-row partition of `wq_b`, `wo_a` and
//! `wo_b` from the checkpoint plane on its own device ([`DeviceFp8Plane::from_device`]), so
//! every packed row is the one-card row and the join is an exact row gather
//! (`dsv4_attention_tp::ATTENTION_TP_NUMERIC_CLASS`). FP8 code planes are `u8`; the decoded
//! 128x128 scale planes are `f32`.

use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut, sys};
use memra_runtime::Gpu;

type Res<T> = Result<T, String>;

pub const FP8_BLOCK: usize = 128;

/// Logical location of a rank-local dense plane in the full checkpoint tensor.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Partition {
    /// Contiguous output rows, used by `wq_b`, `wo_a` and `wo_b`.
    Rows { start: usize, len: usize },
}

/// Metadata and host-owned packed payload for one logical rank: the host oracle the device
/// pack gates compare [`DeviceFp8Plane::from_device`] against.
#[cfg(test)]
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

#[cfg(test)]
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
}

/// Device-owned packed code and decoded FP32 scale planes.
///
/// The metadata gives the attention TP launchers the local `rows`, `cols` and `scale_cols`
/// without reconstructing full-tensor geometry.
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
    /// Upload a host-packed plane (the gates' oracle path).
    #[cfg(test)]
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

    /// Device-to-device row pack from a full source plane. This is asynchronous on
    /// `gpu.stream()`. Both sources must be allocated on that same stream, so their ordered
    /// frees remain after these copies even when CUDA event tracking is disabled.
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
        validate_partition(rank, full_rows, partition)?;
        gpu.ctx
            .bind_to_thread()
            .map_err(|e| format!("FP8 pack bind: {e}"))?;
        let stream = gpu.stream();
        if !std::sync::Arc::ptr_eq(source_codes.stream(), &stream)
            || !std::sync::Arc::ptr_eq(source_scales.stream(), &stream)
        {
            return Err("FP8 pack sources must belong to the copy allocation stream".into());
        }
        let Partition::Rows { start, len: rows } = partition;
        let cols = full_cols;
        let scale_cols = cols / FP8_BLOCK;
        let scale_rows = rows / FP8_BLOCK;
        let mut codes = stream
            .alloc_zeros::<u8>(rows * cols)
            .map_err(|e| format!("FP8 pack code allocation: {e}"))?;
        let mut scales = stream
            .alloc_zeros::<f32>(scale_rows * scale_cols)
            .map_err(|e| format!("FP8 pack scale allocation: {e}"))?;
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

#[allow(clippy::too_many_arguments)] // The two pitched regions and copy extent are the CUDA ABI.
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
    // SAFETY: checked element extents remain inside both allocations. The pointer guards
    // span enqueue, and callers use each allocation's own stream for ordered use/free.
    unsafe { sys::cuMemcpy2DAsync_v2(&copy, stream.cu_stream()).result() }
        .map_err(|e| format!("FP8 pack 2D copy: {e}"))
}

#[cfg(test)]
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

fn validate_partition(rank: usize, rows: usize, partition: Partition) -> Res<()> {
    if rank > 1 {
        return Err(format!("FP8 pack logical rank {rank} outside TP2"));
    }
    let Partition::Rows { start, len } = partition;
    if len == 0
        || !start.is_multiple_of(FP8_BLOCK)
        || !len.is_multiple_of(FP8_BLOCK)
        || start.checked_add(len).is_none_or(|end| end > rows)
    {
        return Err(format!(
            "FP8 row partition is not 128-aligned/in-bounds: start={start} len={len} rows={rows}"
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{DeviceFp8Plane, FP8_BLOCK, Gpu, PackedFp8Host, Partition, copy_2d_device};
    use cudarc::driver::{DevicePtr, DevicePtrMut};
    use std::ffi::c_void;

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

    #[allow(clippy::too_many_arguments)] // Independent full-plane oracle keeps source geometry explicit.
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
        for (local_col, &value) in input.iter().enumerate() {
            let col = col_start + local_col;
            let code = codes[row * full_cols + col];
            let scale = scales[(row / FP8_BLOCK) * scale_cols + col / FP8_BLOCK];
            acc += memra_gguf::nvfp4_repack::fp8_e4m3_to_f32(code) * scale * value;
        }
        acc
    }

    fn dot_packed_row(packed: &PackedFp8Host, row: usize, input: &[f32]) -> f32 {
        assert_eq!(input.len(), packed.cols);
        let mut acc = 0.0f32;
        for (col, &value) in input.iter().enumerate() {
            let code = packed.codes[row * packed.cols + col];
            let scale = packed.scales[(row / FP8_BLOCK) * packed.scale_cols + col / FP8_BLOCK];
            acc += memra_gguf::nvfp4_repack::fp8_e4m3_to_f32(code) * scale * value;
        }
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

    #[test]
    fn real_dsv4_rank_row_shapes_are_exact_and_scale_planes_are_f32() {
        let cases = [(32768usize, 1024usize), (8192, 4096), (4096, 8192)];
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
    fn malformed_shapes_and_non_aligned_partitions_fail_closed() {
        let codes = vec![0u8; FP8_BLOCK * FP8_BLOCK];
        let scales = vec![1.0f32; 1];
        assert!(PackedFp8Host::rows_from_full(0, 128, 128, 1, 127, &codes, &scales).is_err());
        assert!(PackedFp8Host::rows_from_full(2, 128, 128, 0, 128, &codes, &scales).is_err());
        assert!(
            PackedFp8Host::rows_from_full(0, 128, 128, 0, 128, &codes[..127], &scales).is_err()
        );
        assert!(PackedFp8Host::rows_from_full(0, 128, 128, 0, 128, &codes, &[1.0; 2]).is_err());
    }

    fn bf16_input(n: usize, seed: u64) -> Vec<u16> {
        (0..n)
            .map(|i| {
                let value =
                    ((mix(seed.wrapping_add(i as u64 * 0x10001)) % 4097) as f32 - 2048.0) / 257.0;
                (value.to_bits() >> 16) as u16
            })
            .collect()
    }

    fn launch_existing_fp8_gemv(
        gpu: &Gpu,
        plane: &DeviceFp8Plane,
        input: &[u16],
        n: usize,
        k: usize,
        label: &str,
    ) -> Vec<f32> {
        assert_eq!(plane.rows, n, "{label}: row shape");
        assert_eq!(plane.cols, k, "{label}: column shape");
        let stream = gpu.stream();
        let input_dev = stream.clone_htod(input).unwrap();
        let canary = f32::from_bits(0x7fc5_4321);
        let guard = 8usize;
        let mut output = stream.clone_htod(&vec![canary; n + 2 * guard]).unwrap();
        gpu.ctx.bind_to_thread().unwrap();
        let rc_bind = unsafe { crate::mmq_ffi::memra_bind_device(gpu.ctx.ordinal() as i32) };
        assert_eq!(rc_bind, 0, "{label}: runtime device bind");
        {
            let (w, _w_guard) = plane.codes.device_ptr(&stream);
            let (sc, _sc_guard) = plane.scales.device_ptr(&stream);
            let (x, _x_guard) = input_dev.device_ptr(&stream);
            let (y, _y_guard) = output.device_ptr_mut(&stream);
            let rc = unsafe {
                crate::dsv4_ffi::memra_dsv4_gemv_fp8_m(
                    w as *const c_void,
                    sc as *const f32,
                    plane.scale_cols as i32,
                    x as *const c_void,
                    (y as *mut f32).add(guard),
                    1,
                    n as i32,
                    k as i32,
                    0,
                    0,
                    stream.cu_stream().cast(),
                )
            };
            crate::dsv4_ffi::ck(label, rc).unwrap();
        }
        stream.synchronize().unwrap();
        let got = stream.clone_dtoh(&output).unwrap();
        assert!(
            got[..guard].iter().all(|v| v.to_bits() == canary.to_bits()),
            "{label}: prefix"
        );
        assert!(
            got[n + guard..]
                .iter()
                .all(|v| v.to_bits() == canary.to_bits()),
            "{label}: suffix"
        );
        assert_eq!(
            stream.clone_dtoh(&input_dev).unwrap(),
            input,
            "{label}: input mutated"
        );
        assert!(
            got[guard..guard + n].iter().all(|v| v.is_finite()),
            "{label}: non-finite output"
        );
        got[guard..guard + n].to_vec()
    }

    fn assert_plane_unchanged(gpu: &Gpu, plane: &DeviceFp8Plane, codes: &[u8], scales: &[f32]) {
        let stream = gpu.stream();
        stream.synchronize().unwrap();
        assert_eq!(stream.clone_dtoh(&plane.codes).unwrap(), codes);
        assert_eq!(
            stream
                .clone_dtoh(&plane.scales)
                .unwrap()
                .iter()
                .copied()
                .map(f32::to_bits)
                .collect::<Vec<_>>(),
            scales.iter().copied().map(f32::to_bits).collect::<Vec<_>>()
        );
    }

    #[test]
    #[ignore = "requires one locked CUDA GPU; existing FP8 GEMV attention component"]
    fn cuda_attention_split_existing_fp8_gemv_component_gate() {
        let gpu = Gpu::new(0).expect("gpu");
        unsafe { gpu.ctx.disable_event_tracking() };

        for (rows, cols, seed) in [
            (32768usize, 1024usize, 0x10u64),
            (8192, 4096, 0x20),
            (4096, 8192, 0x30),
        ] {
            let full_codes = codes(rows * cols, seed);
            let full_scales = scales(rows / FP8_BLOCK * (cols / FP8_BLOCK), seed + 1);
            if rows == 8192 {
                let full_pack = PackedFp8Host::rows_from_full(
                    0,
                    rows,
                    cols,
                    0,
                    rows,
                    &full_codes,
                    &full_scales,
                )
                .unwrap();
                let full_plane = DeviceFp8Plane::upload(&gpu, full_pack).unwrap();
                let mut full_groups = Vec::with_capacity(8);
                for group in 0..8usize {
                    let input = bf16_input(cols, seed + group as u64 + 2);
                    let group_plane = DeviceFp8Plane::from_device(
                        &gpu,
                        group / 4,
                        rows,
                        cols,
                        Partition::Rows {
                            start: group * 1024,
                            len: 1024,
                        },
                        &full_plane.codes,
                        &full_plane.scales,
                    )
                    .unwrap();
                    full_groups.push(launch_existing_fp8_gemv(
                        &gpu,
                        &group_plane,
                        &input,
                        1024,
                        cols,
                        "wo_a full group",
                    ));
                }
                for rank in 0..2usize {
                    let rank_plane = DeviceFp8Plane::from_device(
                        &gpu,
                        rank,
                        rows,
                        cols,
                        Partition::Rows {
                            start: rank * (rows / 2),
                            len: rows / 2,
                        },
                        &full_plane.codes,
                        &full_plane.scales,
                    )
                    .unwrap();
                    for local_group in 0..4usize {
                        let group = rank * 4 + local_group;
                        let input = bf16_input(cols, seed + group as u64 + 2);
                        let group_plane = DeviceFp8Plane::from_device(
                            &gpu,
                            rank,
                            rows / 2,
                            cols,
                            Partition::Rows {
                                start: local_group * 1024,
                                len: 1024,
                            },
                            &rank_plane.codes,
                            &rank_plane.scales,
                        )
                        .unwrap();
                        let local = launch_existing_fp8_gemv(
                            &gpu,
                            &group_plane,
                            &input,
                            1024,
                            cols,
                            "wo_a rank group",
                        );
                        for (i, (got, want)) in local.iter().zip(&full_groups[group]).enumerate() {
                            assert_eq!(
                                got.to_bits(),
                                want.to_bits(),
                                "wo_a exact group rank={rank} group={group} i={i}"
                            );
                        }
                    }
                }
                assert_plane_unchanged(&gpu, &full_plane, &full_codes, &full_scales);
                continue;
            }
            let full_pack =
                PackedFp8Host::rows_from_full(0, rows, cols, 0, rows, &full_codes, &full_scales)
                    .unwrap();
            let full_plane = DeviceFp8Plane::upload(&gpu, full_pack).unwrap();
            let input = bf16_input(cols, seed + 2);
            let name = if rows == 32768 { "Q_b" } else { "wo_b" };
            let full = launch_existing_fp8_gemv(
                &gpu,
                &full_plane,
                &input,
                rows,
                cols,
                &format!("{name} full"),
            );
            for rank in 0..2usize {
                let half = rows / 2;
                let partition = Partition::Rows {
                    start: rank * half,
                    len: half,
                };
                let local = DeviceFp8Plane::from_device(
                    &gpu,
                    rank,
                    rows,
                    cols,
                    partition,
                    &full_plane.codes,
                    &full_plane.scales,
                )
                .unwrap();
                let local_output = launch_existing_fp8_gemv(
                    &gpu,
                    &local,
                    &input,
                    half,
                    cols,
                    &format!("{name} half"),
                );
                for (i, got) in local_output.iter().enumerate() {
                    assert_eq!(
                        got.to_bits(),
                        full[rank * half + i].to_bits(),
                        "existing GEMV exact row half rank={rank} rows={rows} i={i}"
                    );
                }
                assert_plane_unchanged(&gpu, &full_plane, &full_codes, &full_scales);
            }
        }

        println!("PASS existing GEMV Q_b/wo_a/wo_b exact row halves; GPU join not exercised");
    }

    #[test]
    #[ignore = "requires one locked CUDA GPU; attention FP8 2D pack adapter"]
    fn cuda_attention_split_device_2d_pack_matches_host() {
        let gpu = Gpu::new(0).expect("gpu");
        unsafe { gpu.ctx.disable_event_tracking() };
        let stream = gpu.stream();
        let other_stream = gpu.ctx.new_stream().expect("foreign allocation stream");
        let local_codes = stream.alloc_zeros::<u8>(FP8_BLOCK * FP8_BLOCK).unwrap();
        let local_scales = stream.alloc_zeros::<f32>(1).unwrap();
        let foreign_codes = other_stream
            .alloc_zeros::<u8>(FP8_BLOCK * FP8_BLOCK)
            .unwrap();
        let foreign_scales = other_stream.alloc_zeros::<f32>(1).unwrap();
        for (source_codes, source_scales) in [
            (&foreign_codes, &local_scales),
            (&local_codes, &foreign_scales),
        ] {
            let error = DeviceFp8Plane::from_device(
                &gpu,
                0,
                FP8_BLOCK,
                FP8_BLOCK,
                Partition::Rows {
                    start: 0,
                    len: FP8_BLOCK,
                },
                source_codes,
                source_scales,
            )
            .err()
            .expect("foreign allocation stream must be refused");
            assert!(error.contains("copy allocation stream"), "{error}");
        }
        other_stream.synchronize().unwrap();
        for (rows, cols) in [(32768usize, 1024usize), (8192, 4096), (4096, 8192)] {
            let full_codes = codes(rows * cols, rows as u64 * 19 + cols as u64);
            let full_scales = scales(
                rows / FP8_BLOCK * (cols / FP8_BLOCK),
                rows as u64 * 37 + cols as u64,
            );
            for rank in 0..2usize {
                let half = rows / 2;
                let partition = Partition::Rows {
                    start: rank * half,
                    len: half,
                };
                let expected = PackedFp8Host::rows_from_full(
                    rank,
                    rows,
                    cols,
                    rank * half,
                    half,
                    &full_codes,
                    &full_scales,
                )
                .unwrap();
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
                let (code_x, code_y, scale_x, scale_y) =
                    (0, rank * half, 0, rank * half / FP8_BLOCK);
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
