//! Load-time layout contract for DSV4's two-rank ModelOpt NVFP4 split-bank arm.
//!
//! This module owns the load-time bank adapter, while leaving the serving walk/topology to the
//! compose lane.  The byte geometry is explicit so a loader cannot claim a split bank without
//! packing the down K halves.  The intended full-layer topology is replicated attention/router state on
//! both ranks, every rank owning both halves of every expert, gate/up split by output rows, and
//! down split by input columns.  Down produces a named rank-order f32 partial-sum class.
//!
//! ModelOpt's stored planes are row-major NVFP4 codes plus one E4M3 scale byte per 16 input
//! values.  GU row halves are contiguous in the source.  Down K halves are strided by output
//! row and therefore require a pack operation; pretending that a pointer offset alone makes a
//! down bank contiguous is an ownership/layout bug.  Each rank owns one half of every expert,
//! not a disjoint half of the expert IDs.

use std::ops::Range;

use cudarc::driver::{CudaSlice, CudaStream, DevicePtr, DevicePtrMut};
use std::sync::Arc;

type Res<T> = Result<T, String>;

pub const MODEL_OPT_SPLIT_NUMERIC_CLASS: &str =
    "dsv4_modelopt_tp2_gu_rows_down_cols_f32_rank_reduce";
pub const MODEL_OPT_SPLIT_WORLD: usize = 2;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Projection {
    Gate,
    Up,
    Down,
}
impl Projection {
    pub const ALL: [Self; 3] = [Self::Gate, Self::Up, Self::Down];

    pub const fn plane(self) -> usize {
        match self {
            Self::Gate => 0,
            Self::Up => 2,
            Self::Down => 1,
        }
    }

    pub const fn is_row_split(self) -> bool {
        !matches!(self, Self::Down)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SplitAxis {
    OutputRows,
    InputColumns,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DownReduction {
    /// Rank 0's partial is added first, then rank 1's partial.  This is a named numeric class,
    /// not a claim of bit identity with the unsplit full-K dot.
    RankOrderF32 { ranks: usize },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProjectionShape {
    pub projection: Projection,
    pub axis: SplitAxis,
    pub input: usize,
    pub output: usize,
    /// Bytes in one destination output row of the packed NVFP4 plane.
    pub destination_row_bytes: usize,
    /// Bytes in one destination scale row.  One E4M3 byte covers 16 input values.
    pub destination_scale_row_bytes: usize,
    /// Bytes in one source output row of the full model plane.
    pub source_row_bytes: usize,
    pub source_scale_row_bytes: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StridedCopy {
    /// Offset in the full source projection plane, in bytes.
    pub source_offset: usize,
    /// Offset in the packed destination projection plane, in bytes.
    pub destination_offset: usize,
    /// Bytes copied for each row.
    pub row_bytes: usize,
    /// Number of rows in the copy run.
    pub rows: usize,
    /// Source and destination row strides, in bytes.
    pub source_row_stride: usize,
    pub destination_row_stride: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PlaneSlice {
    pub expert: usize,
    pub projection: Projection,
    pub rank: usize,
    pub axis: SplitAxis,
    pub source_rows: Range<usize>,
    pub source_columns: Range<usize>,
    pub destination_rows: Range<usize>,
    pub destination_columns: Range<usize>,
    pub weights: StridedCopy,
    pub scales: StridedCopy,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ModelOptSplitPlan {
    pub rank: usize,
    pub world: usize,
    pub experts: usize,
    pub hidden: usize,
    pub inter: usize,
    pub reduction: DownReduction,
}

/// Device-resident packed split bank and the six-plane ModelOpt pointer table.  The pack is
/// load-time only.  It uses the CUDA driver's 2D device copy so down's strided K halves are
/// materialized without a host bounce or one launch per output row.
pub struct DeviceSplitBank {
    pub plan: ModelOptSplitPlan,
    pub weights: CudaSlice<u8>,
    pub scales: CudaSlice<u8>,
    pub table: CudaSlice<u64>,
}

fn copy_2d(
    stream: &CudaStream,
    source: &CudaSlice<u8>,
    destination: &mut CudaSlice<u8>,
    copy: StridedCopy,
    label: &str,
) -> Res<()> {
    let source_end = copy
        .rows
        .checked_sub(1)
        .and_then(|rows| rows.checked_mul(copy.source_row_stride))
        .and_then(|offset| offset.checked_add(copy.row_bytes))
        .and_then(|end| copy.source_offset.checked_add(end));
    let destination_end = copy
        .rows
        .checked_sub(1)
        .and_then(|rows| rows.checked_mul(copy.destination_row_stride))
        .and_then(|offset| offset.checked_add(copy.row_bytes))
        .and_then(|end| copy.destination_offset.checked_add(end));
    if copy.row_bytes == 0
        || copy.rows == 0
        || copy.row_bytes > copy.source_row_stride
        || copy.row_bytes > copy.destination_row_stride
        || source_end.is_none_or(|end| end > source.len())
        || destination_end.is_none_or(|end| end > destination.len())
    {
        return Err(format!("{label}: invalid 2D copy bounds"));
    }
    stream
        .context()
        .bind_to_thread()
        .map_err(|e| format!("{label}: bind CUDA context: {e}"))?;
    // Keep both cudarc guards alive until the raw driver call returns.  With event tracking
    // enabled, dropping either temporary before `cuMemcpy2DAsync_v2` would release its
    // stream-order dependency before the enqueue is recorded.
    let (src_base, _src_guard) = source.device_ptr(stream);
    let (dst_base, _dst_guard) = destination.device_ptr_mut(stream);
    let src = src_base + copy.source_offset as u64;
    let dst = dst_base + copy.destination_offset as u64;
    let params = cudarc::driver::sys::CUDA_MEMCPY2D {
        srcXInBytes: 0,
        srcY: 0,
        srcMemoryType: cudarc::driver::sys::CUmemorytype::CU_MEMORYTYPE_DEVICE,
        srcHost: std::ptr::null(),
        srcDevice: src,
        srcArray: std::ptr::null_mut(),
        srcPitch: copy.source_row_stride,
        dstXInBytes: 0,
        dstY: 0,
        dstMemoryType: cudarc::driver::sys::CUmemorytype::CU_MEMORYTYPE_DEVICE,
        dstHost: std::ptr::null_mut(),
        dstDevice: dst,
        dstArray: std::ptr::null_mut(),
        dstPitch: copy.destination_row_stride,
        WidthInBytes: copy.row_bytes,
        Height: copy.rows,
    };
    let rc = unsafe { cudarc::driver::sys::cuMemcpy2DAsync_v2(&params, stream.cu_stream()) };
    if rc != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
        return Err(format!("{label}: cuMemcpy2DAsync rc={rc:?}"));
    }
    Ok(())
}

impl ModelOptSplitPlan {
    pub fn new(rank: usize, experts: usize, hidden: usize, inter: usize) -> Res<Self> {
        if rank >= MODEL_OPT_SPLIT_WORLD {
            return Err(format!(
                "ModelOpt TP split rank {rank} outside world {}",
                MODEL_OPT_SPLIT_WORLD
            ));
        }
        if experts == 0 || experts > 512 {
            return Err(format!("ModelOpt expert count {experts} outside 1..=512"));
        }
        if hidden == 0 || inter == 0 || !hidden.is_multiple_of(128) || !inter.is_multiple_of(128) {
            return Err(format!(
                "ModelOpt split dimensions hidden={hidden} inter={inter} require nonzero multiples of 128"
            ));
        }
        if !hidden.is_multiple_of(MODEL_OPT_SPLIT_WORLD)
            || !inter.is_multiple_of(MODEL_OPT_SPLIT_WORLD)
            || (inter / MODEL_OPT_SPLIT_WORLD) % 64 != 0
        {
            return Err(format!(
                "ModelOpt TP2 split hidden={hidden} inter={inter} is not packed-byte/kernel aligned"
            ));
        }
        let area = hidden
            .checked_mul(inter)
            .ok_or("ModelOpt split dimension overflow")?;
        if experts
            .checked_mul(3)
            .and_then(|planes| planes.checked_mul(area))
            .is_none()
        {
            return Err("ModelOpt split bank byte geometry overflows usize".into());
        }
        Ok(Self {
            rank,
            world: MODEL_OPT_SPLIT_WORLD,
            experts,
            hidden,
            inter,
            reduction: DownReduction::RankOrderF32 {
                ranks: MODEL_OPT_SPLIT_WORLD,
            },
        })
    }

    pub const fn local_hidden(self) -> usize {
        self.hidden / MODEL_OPT_SPLIT_WORLD
    }

    pub const fn local_inter(self) -> usize {
        self.inter / MODEL_OPT_SPLIT_WORLD
    }

    pub const fn full_weight_row_bytes(self, projection: Projection) -> usize {
        match projection {
            Projection::Gate | Projection::Up => self.hidden / 2,
            Projection::Down => self.inter / 2,
        }
    }

    pub const fn full_scale_row_bytes(self, projection: Projection) -> usize {
        match projection {
            Projection::Gate | Projection::Up => self.hidden / 16,
            Projection::Down => self.inter / 16,
        }
    }

    pub const fn local_weight_row_bytes(self, projection: Projection) -> usize {
        match projection {
            Projection::Gate | Projection::Up => self.hidden / 2,
            Projection::Down => self.local_inter() / 2,
        }
    }

    pub const fn local_scale_row_bytes(self, projection: Projection) -> usize {
        match projection {
            Projection::Gate | Projection::Up => self.hidden / 16,
            Projection::Down => self.local_inter() / 16,
        }
    }

    pub const fn local_input(self, projection: Projection) -> usize {
        match projection {
            Projection::Gate | Projection::Up => self.hidden,
            Projection::Down => self.local_inter(),
        }
    }

    pub const fn local_output(self, projection: Projection) -> usize {
        match projection {
            Projection::Gate | Projection::Up => self.local_inter(),
            Projection::Down => self.hidden,
        }
    }

    pub const fn shape(self, projection: Projection) -> ProjectionShape {
        ProjectionShape {
            projection,
            axis: if projection.is_row_split() {
                SplitAxis::OutputRows
            } else {
                SplitAxis::InputColumns
            },
            input: self.local_input(projection),
            output: self.local_output(projection),
            destination_row_bytes: self.local_weight_row_bytes(projection),
            destination_scale_row_bytes: self.local_scale_row_bytes(projection),
            source_row_bytes: self.full_weight_row_bytes(projection),
            source_scale_row_bytes: self.full_scale_row_bytes(projection),
        }
    }

    pub const fn projection_destination_bytes(self, projection: Projection) -> usize {
        self.experts * self.local_output(projection) * self.local_weight_row_bytes(projection)
    }

    pub const fn projection_destination_scale_bytes(self, projection: Projection) -> usize {
        self.experts * self.local_output(projection) * self.local_scale_row_bytes(projection)
    }

    pub const fn projection_source_bytes(self, projection: Projection) -> usize {
        let output_rows = match projection {
            Projection::Gate | Projection::Up => self.inter,
            Projection::Down => self.hidden,
        };
        self.experts * self.full_weight_row_bytes(projection) * output_rows
    }

    pub const fn projection_source_scale_bytes(self, projection: Projection) -> usize {
        let output_rows = match projection {
            Projection::Gate | Projection::Up => self.inter,
            Projection::Down => self.hidden,
        };
        self.experts * self.full_scale_row_bytes(projection) * output_rows
    }

    pub const fn full_bank_weight_bytes(self) -> usize {
        self.experts * 3 * self.inter * self.hidden / 2
    }

    pub const fn full_bank_scale_bytes(self) -> usize {
        self.experts * 3 * self.inter * self.hidden / 16
    }

    pub const fn split_bank_weight_bytes(self) -> usize {
        self.experts
            * (2 * self.local_inter() * self.hidden / 2 + self.hidden * self.local_inter() / 2)
    }

    pub const fn split_bank_scale_bytes(self) -> usize {
        self.experts
            * (2 * self.local_inter() * self.hidden / 16 + self.hidden * self.local_inter() / 16)
    }

    /// `scale_2` is one pow2 f32 per expert/projection and stays replicated with the route
    /// metadata; it is not part of the split code/scale byte banks.
    pub const fn replicated_scale2_elements(self) -> usize {
        self.experts * 3
    }

    pub const fn replicated_scale2_bytes(self) -> usize {
        self.replicated_scale2_elements() * std::mem::size_of::<f32>()
    }

    pub fn validate_source_bank(self, weight_bytes: usize, scale_bytes: usize) -> Res<()> {
        if weight_bytes != self.full_bank_weight_bytes()
            || scale_bytes != self.full_bank_scale_bytes()
        {
            return Err(format!(
                "ModelOpt full bank length mismatch: weights={weight_bytes}/{} scales={scale_bytes}/{}",
                self.full_bank_weight_bytes(),
                self.full_bank_scale_bytes()
            ));
        }
        Ok(())
    }

    pub fn validate_split_bank(self, weight_bytes: usize, scale_bytes: usize) -> Res<()> {
        if weight_bytes != self.split_bank_weight_bytes()
            || scale_bytes != self.split_bank_scale_bytes()
        {
            return Err(format!(
                "ModelOpt split bank length mismatch: weights={weight_bytes}/{} scales={scale_bytes}/{}",
                self.split_bank_weight_bytes(),
                self.split_bank_scale_bytes()
            ));
        }
        Ok(())
    }

    /// Return the six-plane pointer-table offsets expected by the existing ModelOpt visitor:
    /// `[w1[experts], sc1[experts], w2[experts], sc2[experts], w3[experts], sc3[experts]]`.
    /// The caller adds the device base address of the packed weight/scale allocations.  The
    /// table intentionally uses the same expert-major `[expert][projection]` bank order as the
    /// current `modelopt_table`; only the per-projection rows/strides differ for the split bank.
    pub fn pointer_offsets(self) -> Vec<u64> {
        let weight_plane = self.projection_destination_bytes(Projection::Gate) / self.experts;
        let scale_plane = self.projection_destination_scale_bytes(Projection::Gate) / self.experts;
        let mut result = vec![0u64; self.experts * 6];
        for expert in 0..self.experts {
            for projection in Projection::ALL {
                let plane = projection.plane();
                let weight_offset = (expert * 3 + plane) * weight_plane;
                let scale_offset = (expert * 3 + plane) * scale_plane;
                result[2 * plane * self.experts + expert] = weight_offset as u64;
                result[(2 * plane + 1) * self.experts + expert] = scale_offset as u64;
            }
        }
        result
    }

    /// Pack one full ModelOpt bank into this rank's split bank and build the matching pointer
    /// table.  This does not synchronize the stream; callers must retain the returned buffers
    /// and use the same stream/event ordering before the first expert visitor reads `table`.
    pub fn pack_device(
        self,
        stream: &Arc<CudaStream>,
        source_weights: &CudaSlice<u8>,
        source_scales: &CudaSlice<u8>,
    ) -> Res<DeviceSplitBank> {
        self.validate_source_bank(source_weights.len(), source_scales.len())?;
        let mut weights = stream
            .alloc_zeros::<u8>(self.split_bank_weight_bytes())
            .map_err(|e| format!("ModelOpt split weight allocation: {e}"))?;
        let mut scales = stream
            .alloc_zeros::<u8>(self.split_bank_scale_bytes())
            .map_err(|e| format!("ModelOpt split scale allocation: {e}"))?;
        for expert in 0..self.experts {
            for projection in Projection::ALL {
                let slice = self.slice(expert, projection)?;
                copy_2d(
                    stream,
                    source_weights,
                    &mut weights,
                    slice.weights,
                    "ModelOpt split weights",
                )?;
                copy_2d(
                    stream,
                    source_scales,
                    &mut scales,
                    slice.scales,
                    "ModelOpt split scales",
                )?;
            }
        }
        let weight_base = weights.device_ptr(stream).0;
        let scale_base = scales.device_ptr(stream).0;
        let mut pointers = self.pointer_offsets();
        for plane in 0..3 {
            for expert in 0..self.experts {
                let wi = 2 * plane * self.experts + expert;
                let si = (2 * plane + 1) * self.experts + expert;
                pointers[wi] = weight_base
                    .checked_add(pointers[wi])
                    .ok_or("ModelOpt split weight pointer overflow")?;
                pointers[si] = scale_base
                    .checked_add(pointers[si])
                    .ok_or("ModelOpt split scale pointer overflow")?;
            }
        }
        let table = stream
            .clone_htod(&pointers)
            .map_err(|e| format!("ModelOpt split pointer table: {e}"))?;
        Ok(DeviceSplitBank {
            plan: self,
            weights,
            scales,
            table,
        })
    }

    /// Build the weight and scale copy descriptors for one expert/projection.  For GU the
    /// selected rows are one contiguous run.  For down, each output row contributes one
    /// strided half-K run, so a pack implementation must honor `rows` and both strides.
    pub fn slice(self, expert: usize, projection: Projection) -> Res<PlaneSlice> {
        if expert >= self.experts {
            return Err("ModelOpt split expert outside full replicated bank".into());
        }
        let full_w = self.full_weight_row_bytes(projection);
        let full_s = self.full_scale_row_bytes(projection);
        let local_w = self.local_weight_row_bytes(projection);
        let local_s = self.local_scale_row_bytes(projection);
        let out = self.local_output(projection);
        let full_out = match projection {
            Projection::Gate | Projection::Up => self.inter,
            Projection::Down => self.hidden,
        };
        let projection_offset_w = projection.plane() * self.inter * self.hidden / 2;
        let projection_offset_s = projection.plane() * self.inter * self.hidden / 16;
        let expert_offset_w = expert * 3 * self.inter * self.hidden / 2;
        let expert_offset_s = expert * 3 * self.inter * self.hidden / 16;
        let (
            source_rows,
            source_columns,
            destination_rows,
            destination_columns,
            source_w,
            source_s,
        ) = if projection.is_row_split() {
            let start = self.rank * out;
            (
                start..start + out,
                0..self.hidden,
                0..out,
                0..self.hidden,
                expert_offset_w + projection_offset_w + start * full_w,
                expert_offset_s + projection_offset_s + start * full_s,
            )
        } else {
            let start = self.rank * self.local_inter();
            (
                0..self.hidden,
                start..start + self.local_inter(),
                0..self.hidden,
                0..self.local_inter(),
                expert_offset_w + projection_offset_w + start / 2,
                expert_offset_s + projection_offset_s + start / 16,
            )
        };
        Ok(PlaneSlice {
            expert,
            projection,
            rank: self.rank,
            axis: if projection.is_row_split() {
                SplitAxis::OutputRows
            } else {
                SplitAxis::InputColumns
            },
            source_rows,
            source_columns,
            destination_rows,
            destination_columns,
            weights: StridedCopy {
                source_offset: source_w,
                destination_offset: (expert * 3 + projection.plane())
                    * self.local_output(projection)
                    * self.local_weight_row_bytes(projection),
                row_bytes: local_w,
                rows: if projection.is_row_split() {
                    out
                } else {
                    full_out
                },
                source_row_stride: full_w,
                destination_row_stride: local_w,
            },
            scales: StridedCopy {
                source_offset: source_s,
                destination_offset: (expert * 3 + projection.plane())
                    * self.local_output(projection)
                    * self.local_scale_row_bytes(projection),
                row_bytes: local_s,
                rows: if projection.is_row_split() {
                    out
                } else {
                    full_out
                },
                source_row_stride: full_s,
                destination_row_stride: local_s,
            },
        })
    }

    pub const fn replicated_route_domain(self) -> Range<usize> {
        0..self.experts
    }

    pub const fn uses_full_route_on_each_rank(self) -> bool {
        true
    }

    pub const fn numeric_class(self) -> &'static str {
        MODEL_OPT_SPLIT_NUMERIC_CLASS
    }
}

#[cfg(test)]
mod tests {
    use super::{DownReduction, ModelOptSplitPlan, Projection, SplitAxis, StridedCopy, copy_2d};

    #[test]
    fn dsv4_shape_matches_modelopt_4096x2048_geometry() {
        let plan = ModelOptSplitPlan::new(0, 256, 4096, 2048).unwrap();
        assert_eq!(plan.world, 2);
        assert_eq!(plan.local_hidden(), 2048);
        assert_eq!(plan.local_inter(), 1024);
        assert_eq!(plan.shape(Projection::Gate).axis, SplitAxis::OutputRows);
        assert_eq!(plan.shape(Projection::Down).axis, SplitAxis::InputColumns);
        assert_eq!(plan.shape(Projection::Gate).input, 4096);
        assert_eq!(plan.shape(Projection::Gate).output, 1024);
        assert_eq!(plan.shape(Projection::Down).input, 1024);
        assert_eq!(plan.shape(Projection::Down).output, 4096);
        assert_eq!(plan.shape(Projection::Down).source_row_bytes, 1024);
        assert_eq!(plan.shape(Projection::Down).destination_row_bytes, 512);
        assert_eq!(plan.shape(Projection::Down).source_scale_row_bytes, 128);
        assert_eq!(plan.shape(Projection::Down).destination_scale_row_bytes, 64);
        assert_eq!(plan.full_bank_weight_bytes(), 3 * 2048 * 4096 * 256 / 2);
        assert_eq!(plan.full_bank_scale_bytes(), 3 * 128 * 4096 * 256);
        assert_eq!(
            plan.split_bank_weight_bytes(),
            plan.full_bank_weight_bytes() / 2
        );
        assert_eq!(
            plan.split_bank_scale_bytes(),
            plan.full_bank_scale_bytes() / 2
        );
        assert_eq!(plan.replicated_scale2_elements(), 256 * 3);
        assert_eq!(plan.replicated_scale2_bytes(), 256 * 3 * 4);
        plan.validate_source_bank(plan.full_bank_weight_bytes(), plan.full_bank_scale_bytes())
            .unwrap();
        plan.validate_split_bank(
            plan.split_bank_weight_bytes(),
            plan.split_bank_scale_bytes(),
        )
        .unwrap();
        assert_eq!(plan.reduction, DownReduction::RankOrderF32 { ranks: 2 });
        assert!(plan.uses_full_route_on_each_rank());
        assert_eq!(plan.replicated_route_domain(), 0..256);
    }

    #[test]
    fn row_and_column_copy_descriptors_preserve_source_strides() {
        let rank0 = ModelOptSplitPlan::new(0, 256, 4096, 2048).unwrap();
        let rank1 = ModelOptSplitPlan::new(1, 256, 4096, 2048).unwrap();
        let gu0 = rank0.slice(7, Projection::Gate).unwrap();
        let gu1 = rank1.slice(7, Projection::Gate).unwrap();
        assert_eq!(gu0.source_rows, 0..1024);
        assert_eq!(gu1.source_rows, 1024..2048);
        assert_eq!(gu0.weights.rows, 1024);
        assert_eq!(gu0.weights.source_row_stride, 2048);
        assert_eq!(gu0.weights.destination_row_stride, 2048);
        assert_eq!(gu0.scales.source_row_stride, 256);
        assert_eq!(gu0.scales.destination_row_stride, 256);
        assert_eq!(gu0.weights.destination_offset, (7 * 3) * 1024 * 2048);
        assert_eq!(
            rank0
                .slice(7, Projection::Up)
                .unwrap()
                .weights
                .destination_offset,
            (7 * 3 + 2) * 1024 * 2048
        );
        let table = rank0.pointer_offsets();
        assert_eq!(table.len(), 256 * 6);
        assert_eq!(table[7], (7 * 3) as u64 * 1024 * 2048);
        assert_eq!(table[2 * 256 + 7], (7 * 3 + 1) as u64 * 1024 * 2048);
        assert_eq!(table[4 * 256 + 7], (7 * 3 + 2) as u64 * 1024 * 2048);
        assert_eq!(table[256 + 7], (7 * 3) as u64 * 1024 * 256);

        let down0 = rank0.slice(7, Projection::Down).unwrap();
        let down1 = rank1.slice(7, Projection::Down).unwrap();
        assert_eq!(down0.source_columns, 0..1024);
        assert_eq!(down1.source_columns, 1024..2048);
        assert_eq!(down0.weights.rows, 4096);
        assert_eq!(down0.weights.row_bytes, 512);
        assert_eq!(down0.weights.source_row_stride, 1024);
        assert_eq!(down0.weights.destination_row_stride, 512);
        assert_eq!(down0.scales.row_bytes, 64);
        assert_eq!(down0.scales.source_row_stride, 128);
        assert_eq!(down0.scales.destination_row_stride, 64);
        assert_eq!(down0.weights.destination_offset, (7 * 3 + 1) * 4096 * 512);
        assert_eq!(
            down0.weights.source_offset + 512,
            down1.weights.source_offset
        );
        assert_eq!(down0.scales.source_offset + 64, down1.scales.source_offset);
    }

    #[test]
    fn invalid_split_shapes_fail_closed() {
        for (rank, experts, hidden, inter) in [
            (2, 256, 4096, 2048),
            (0, 0, 4096, 2048),
            (0, 513, 4096, 2048),
            (0, 256, 4096, 2049),
            (0, 256, 4096, 192),
        ] {
            assert!(
                ModelOptSplitPlan::new(rank, experts, hidden, inter).is_err(),
                "unexpected admission rank={rank} experts={experts} hidden={hidden} inter={inter}"
            );
        }
        let plan = ModelOptSplitPlan::new(0, 256, 4096, 2048).unwrap();
        assert!(plan.slice(256, Projection::Gate).is_err());
        assert!(
            plan.validate_source_bank(
                plan.full_bank_weight_bytes() - 1,
                plan.full_bank_scale_bytes()
            )
            .is_err()
        );
        assert!(
            plan.validate_split_bank(
                plan.split_bank_weight_bytes(),
                plan.split_bank_scale_bytes() + 1
            )
            .is_err()
        );
    }

    #[test]
    #[ignore = "requires an exclusively locked CUDA device; ModelOpt split-bank copy identity"]
    fn cuda_pack_split_bank_preserves_codes_scales_and_pointer_table() {
        use cudarc::driver::DevicePtr;
        use memra_runtime::Gpu;

        let gpu = Gpu::new(0).expect("GPU required by ignored split-bank gate");
        let stream = gpu.stream();
        fn source_pattern(bytes: usize, seed: usize) -> Vec<u8> {
            (0..bytes)
                .map(|i| {
                    // SplitMix64-style index mixing: no small byte period at the real plane,
                    // expert, or rank-half offsets used below.
                    let mut z =
                        (i as u64).wrapping_add((seed as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
                    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
                    (z ^ (z >> 31)) as u8
                })
                .collect()
        }

        fn independent_expected(
            plan: ModelOptSplitPlan,
            source_w: &[u8],
            source_s: &[u8],
        ) -> (Vec<u8>, Vec<u8>) {
            let full_weight_plane = plan.inter * plan.hidden / 2;
            let full_scale_plane = plan.inter * plan.hidden / 16;
            let local_inter = plan.inter / 2;
            let local_weight_plane = local_inter * plan.hidden / 2;
            let local_scale_plane = local_inter * plan.hidden / 16;
            let mut expected_w = vec![0xA5; plan.split_bank_weight_bytes()];
            let mut expected_s = vec![0x5A; plan.split_bank_scale_bytes()];
            for expert in 0..plan.experts {
                for (projection, plane) in [
                    (Projection::Gate, 0usize),
                    (Projection::Up, 2),
                    (Projection::Down, 1),
                ] {
                    let row_count = if projection.is_row_split() {
                        local_inter
                    } else {
                        plan.hidden
                    };
                    let source_row_bytes = if projection.is_row_split() {
                        plan.hidden / 2
                    } else {
                        plan.inter / 2
                    };
                    let source_scale_row_bytes = if projection.is_row_split() {
                        plan.hidden / 16
                    } else {
                        plan.inter / 16
                    };
                    let destination_row_bytes = if projection.is_row_split() {
                        plan.hidden / 2
                    } else {
                        local_inter / 2
                    };
                    let destination_scale_row_bytes = if projection.is_row_split() {
                        plan.hidden / 16
                    } else {
                        local_inter / 16
                    };
                    let source_row_start = if projection.is_row_split() {
                        plan.rank * local_inter
                    } else {
                        0
                    };
                    let source_col_start = if projection.is_row_split() {
                        0
                    } else {
                        plan.rank * local_inter
                    };
                    let source_weight_base =
                        expert * 3 * full_weight_plane + plane * full_weight_plane;
                    let source_scale_base =
                        expert * 3 * full_scale_plane + plane * full_scale_plane;
                    let destination_weight_base = (expert * 3 + plane) * local_weight_plane;
                    let destination_scale_base = (expert * 3 + plane) * local_scale_plane;
                    for row in 0..row_count {
                        let source_row = source_row_start + row;
                        let source_weight = source_weight_base
                            + source_row * source_row_bytes
                            + source_col_start / 2;
                        let destination_weight =
                            destination_weight_base + row * destination_row_bytes;
                        expected_w[destination_weight..destination_weight + destination_row_bytes]
                            .copy_from_slice(
                                &source_w[source_weight..source_weight + destination_row_bytes],
                            );
                        let source_scale = source_scale_base
                            + source_row * source_scale_row_bytes
                            + source_col_start / 16;
                        let destination_scale =
                            destination_scale_base + row * destination_scale_row_bytes;
                        expected_s
                            [destination_scale..destination_scale + destination_scale_row_bytes]
                            .copy_from_slice(
                                &source_s[source_scale..source_scale + destination_scale_row_bytes],
                            );
                    }
                }
            }
            (expected_w, expected_s)
        }

        fn run_case(gpu: &Gpu, plan: ModelOptSplitPlan, seed: usize) {
            let stream = gpu.stream();
            let source_w = source_pattern(plan.full_bank_weight_bytes(), seed);
            let source_s = source_pattern(plan.full_bank_scale_bytes(), seed ^ 0x5A);
            let weights = stream.clone_htod(&source_w).unwrap();
            let scales = stream.clone_htod(&source_s).unwrap();
            let bank = plan.pack_device(&stream, &weights, &scales).unwrap();
            stream.synchronize().unwrap();
            let (expected_w, expected_s) = independent_expected(plan, &source_w, &source_s);
            assert_eq!(stream.clone_dtoh(&bank.weights).unwrap(), expected_w);
            assert_eq!(stream.clone_dtoh(&bank.scales).unwrap(), expected_s);
            assert_eq!(
                stream.clone_dtoh(&weights).unwrap(),
                source_w,
                "source weights mutated"
            );
            assert_eq!(
                stream.clone_dtoh(&scales).unwrap(),
                source_s,
                "source scales mutated"
            );
            let actual_table = stream.clone_dtoh(&bank.table).unwrap();
            let weight_base = bank.weights.device_ptr(&stream).0;
            let scale_base = bank.scales.device_ptr(&stream).0;
            let weight_plane = plan.split_bank_weight_bytes() / (plan.experts * 3);
            let scale_plane = plan.split_bank_scale_bytes() / (plan.experts * 3);
            for plane in 0..3 {
                for expert in 0..plan.experts {
                    let wi = 2 * plane * plan.experts + expert;
                    let si = (2 * plane + 1) * plan.experts + expert;
                    assert_eq!(
                        actual_table[wi],
                        weight_base + ((expert * 3 + plane) * weight_plane) as u64
                    );
                    assert_eq!(
                        actual_table[si],
                        scale_base + ((expert * 3 + plane) * scale_plane) as u64
                    );
                }
            }
        }

        fn assert_real_red_teeth() {
            let plan0 = ModelOptSplitPlan::new(0, 2, 4096, 2048).unwrap();
            let plan1 = ModelOptSplitPlan::new(1, 2, 4096, 2048).unwrap();
            let source_w = source_pattern(plan0.full_bank_weight_bytes(), 0xC0FFEE);
            let source_s = source_pattern(plan0.full_bank_scale_bytes(), 0xBAD5EED);
            let plane_w = plan0.inter * plan0.hidden / 2;
            let plane_s = plan0.inter * plan0.hidden / 16;
            let gu_half_w = plan0.local_inter() * plan0.hidden / 2;
            let gu_half_s = plan0.local_inter() * plan0.hidden / 16;
            assert_ne!(
                &source_w[..plane_w],
                &source_w[plane_w..2 * plane_w],
                "wrong projection red tooth is periodic"
            );
            assert_ne!(
                &source_w[..plane_w],
                &source_w[3 * plane_w..4 * plane_w],
                "wrong expert red tooth is periodic"
            );
            assert_ne!(
                &source_w[..gu_half_w],
                &source_w[gu_half_w..2 * gu_half_w],
                "opposite-rank GU half red tooth is periodic"
            );
            assert_ne!(
                &source_s[..plane_s],
                &source_s[plane_s..2 * plane_s],
                "scale projection red tooth is periodic"
            );
            assert_ne!(
                &source_s[..gu_half_s],
                &source_s[gu_half_s..2 * gu_half_s],
                "scale GU half red tooth is periodic"
            );
            let (expected0, expected_s0) = independent_expected(plan0, &source_w, &source_s);
            let (expected1, expected_s1) = independent_expected(plan1, &source_w, &source_s);
            assert_ne!(
                expected0, expected1,
                "rank GU/down mapping did not move bytes"
            );
            assert_ne!(
                expected_s0, expected_s1,
                "rank scale mapping did not move bytes"
            );
        }

        assert_real_red_teeth();
        // Both logical ranks on the actual DSV4 geometry, plus a tiny one-expert tail/control.
        for rank in 0..2 {
            run_case(
                &gpu,
                ModelOptSplitPlan::new(rank, 2, 4096, 2048).unwrap(),
                rank + 1,
            );
            run_case(
                &gpu,
                ModelOptSplitPlan::new(rank, 1, 128, 128).unwrap(),
                rank + 11,
            );
        }

        // Guard canaries and pre-enqueue bounds refusals exercise the raw 2D seam independently
        // of the plan oracle.
        let source = stream.clone_htod(&(0u8..32).collect::<Vec<_>>()).unwrap();
        let mut guarded = stream.clone_htod(&vec![0xC7u8; 48]).unwrap();
        copy_2d(
            &stream,
            &source,
            &mut guarded,
            StridedCopy {
                source_offset: 0,
                destination_offset: 8,
                row_bytes: 8,
                rows: 2,
                source_row_stride: 16,
                destination_row_stride: 16,
            },
            "canary",
        )
        .unwrap();
        stream.synchronize().unwrap();
        let guarded_host = stream.clone_dtoh(&guarded).unwrap();
        assert!(guarded_host[..8].iter().all(|&v| v == 0xC7));
        assert_eq!(&guarded_host[8..16], &[0, 1, 2, 3, 4, 5, 6, 7]);
        assert_eq!(&guarded_host[24..32], &[16, 17, 18, 19, 20, 21, 22, 23]);
        assert!(guarded_host[16..24].iter().all(|&v| v == 0xC7));
        assert!(guarded_host[32..].iter().all(|&v| v == 0xC7));
        let tiny = ModelOptSplitPlan::new(0, 1, 128, 128).unwrap();
        let short_w = stream
            .clone_htod(&vec![0u8; tiny.full_bank_weight_bytes() - 1])
            .unwrap();
        let full_s = stream
            .clone_htod(&vec![0u8; tiny.full_bank_scale_bytes()])
            .unwrap();
        assert!(tiny.pack_device(&stream, &short_w, &full_s).is_err());
        let full_w = stream
            .clone_htod(&vec![0u8; tiny.full_bank_weight_bytes()])
            .unwrap();
        let short_s = stream
            .clone_htod(&vec![0u8; tiny.full_bank_scale_bytes() - 1])
            .unwrap();
        assert!(tiny.pack_device(&stream, &full_w, &short_s).is_err());
    }
}
