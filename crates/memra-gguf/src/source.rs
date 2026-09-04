//! Source-agnostic weight access: a `TensorSource` trait the engine loads from, implemented by
//! BOTH the GGUF reader and the safetensors reader. The engine only ever asks for ggml-style
//! names + gets back `{ggml_type, ne, &[u8]}`; the source hides where the bytes come from.
//!
//! This trait lives in memra-gguf (not memra-engine) because it returns memra-gguf types
//! (`GgmlType`, `ModelConfig`) and both readers live here. memra-engine already depends on
//! memra-gguf, so `GpuTensor::load_from_source(&dyn TensorSource, ...)` introduces no new dep.

use crate::config::{Arch, JsonObj, ModelConfig};
use crate::safetensors::{StInfo, StModel};
use crate::tensor_contract::{
    CheckpointDialect, FloatType, IntegerType, QuantLayout, StorageLayout, TensorCensusEntry,
};
use crate::{GgmlType, GgufFile};
use memmap2::Mmap;
use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
#[cfg(unix)]
use std::os::fd::AsRawFd;
#[cfg(unix)]
use std::os::unix::fs::{MetadataExt, OpenOptionsExt};
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Whole-map access advice for expert slab files. `Random` preserves the original spill behavior;
/// `Normal` lets Linux use its ordinary mmap readahead for each multi-megabyte expert access.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExpertMmapAdvice {
    Random,
    Normal,
}

/// Pure parser kept separate from the cached environment lookup so invalid/default behavior is
/// unit-testable without mutating process-global environment state.
pub fn parse_expert_mmap_advice(value: Option<&str>) -> Result<ExpertMmapAdvice, &'static str> {
    match value.unwrap_or("random") {
        "random" => Ok(ExpertMmapAdvice::Random),
        "normal" => Ok(ExpertMmapAdvice::Normal),
        _ => Err("expected random or normal"),
    }
}

pub fn expert_mmap_advice() -> ExpertMmapAdvice {
    static MODE: std::sync::OnceLock<ExpertMmapAdvice> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| {
        let raw = std::env::var("MEMRA_MOE_MMAP_ADVICE").ok();
        match parse_expert_mmap_advice(raw.as_deref()) {
            Ok(mode) => mode,
            Err(reason) => {
                eprintln!(
                    "[spill] invalid MEMRA_MOE_MMAP_ADVICE={:?} ({reason}); using random",
                    raw.as_deref().unwrap_or("")
                );
                ExpertMmapAdvice::Random
            }
        }
    })
}

/// Apply the selected policy to one expert mmap. `MADV_NORMAL` explicitly clears a prior
/// `MADV_RANDOM` VMA policy, so this is safe for maps reused across loader paths.
pub fn apply_expert_mmap_advice(map: &Mmap) -> std::io::Result<()> {
    #[cfg(unix)]
    {
        let advice = match expert_mmap_advice() {
            ExpertMmapAdvice::Random => memmap2::Advice::Random,
            ExpertMmapAdvice::Normal => memmap2::Advice::Normal,
        };
        map.advise(advice)
    }
    #[cfg(not(unix))]
    {
        let _ = map;
        Ok(())
    }
}

/// Load-time page-cache population policy for expert slab files (`.memra-repack` tiers and the
/// GGUF spill tier). `Fits` populates a slab that leaves the host-RAM floor intact; `Off` keeps
/// the pure demand-fault behavior.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExpertSlabPopulate {
    Fits,
    Off,
}

/// Fraction of MemTotal that must remain available AFTER a slab is populated. The page cache the
/// slab lives in is reclaimable, so this floor only has to protect the allocations that are not:
/// engine host buffers, pinned staging, and the CUDA context.
pub const EXPERT_SLAB_RAM_FLOOR_FRAC: f64 = 0.20;

/// Pure parser kept separate from the cached environment lookup so invalid/default behavior is
/// unit-testable without mutating process-global environment state.
pub fn parse_expert_slab_populate(value: Option<&str>) -> Result<ExpertSlabPopulate, &'static str> {
    match value.unwrap_or("fits") {
        "fits" => Ok(ExpertSlabPopulate::Fits),
        "off" => Ok(ExpertSlabPopulate::Off),
        _ => Err("expected fits or off"),
    }
}

pub fn expert_slab_populate() -> ExpertSlabPopulate {
    static MODE: std::sync::OnceLock<ExpertSlabPopulate> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| {
        let raw = std::env::var("MEMRA_MOE_SLAB_POPULATE").ok();
        match parse_expert_slab_populate(raw.as_deref()) {
            Ok(mode) => mode,
            Err(reason) => {
                eprintln!(
                    "[slab-populate] invalid MEMRA_MOE_SLAB_POPULATE={:?} ({reason}); using fits",
                    raw.as_deref().unwrap_or("")
                );
                ExpertSlabPopulate::Fits
            }
        }
    })
}

/// Does populating `slab_len` bytes leave `floor_frac` of `mem_total` still available? Pure so the
/// guard is testable without a machine whose `/proc/meminfo` can be steered.
pub fn expert_slab_populate_fits(
    slab_len: usize,
    mem_available: usize,
    mem_total: usize,
    floor_frac: f64,
) -> bool {
    let floor = (mem_total as f64 * floor_frac) as usize;
    mem_available.saturating_sub(slab_len) >= floor
}

/// One `/proc/meminfo` field in bytes. `None` when the file or key is unreadable — callers treat
/// an unknown budget as "do not populate", so an unparseable meminfo degrades to today's behavior.
fn meminfo_bytes(key: &str) -> Option<usize> {
    let text = std::fs::read_to_string("/proc/meminfo").ok()?;
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix(key) {
            let kb: usize = rest
                .trim_start_matches(':')
                .split_whitespace()
                .next()?
                .parse()
                .ok()?;
            return Some(kb * 1024);
        }
    }
    None
}

/// Read `file` sequentially so its pages enter the page cache, and report the elapsed transfer.
///
/// WHY A READ AND NOT `MADV_POPULATE_READ`: the slab VMA carries the `MEMRA_MOE_MMAP_ADVICE`
/// policy, whose default `MADV_RANDOM` suppresses fault-around and readahead. Prefaulting through
/// the mapping under that policy issues one 4 KiB request per page; the file handle is not bound
/// by the VMA policy and gets ordinary sequential readahead.
///
/// The bytes are discarded: the goal is the page cache the slab mmap already points at, so this
/// adds no permanent host allocation and does not change any tensor's bytes.
fn populate_pages(file: &File, len: usize) -> std::io::Result<std::time::Duration> {
    use std::os::unix::fs::FileExt;
    let start = std::time::Instant::now();
    let mut scratch = vec![0u8; 8 << 20];
    let mut off = 0usize;
    while off < len {
        let want = scratch.len().min(len - off);
        match file.read_at(&mut scratch[..want], off as u64) {
            Ok(0) => break,
            Ok(n) => off += n,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
        }
    }
    Ok(start.elapsed())
}

/// Populate one expert slab's pages at load when the configured policy and the host-RAM floor
/// allow it. Returns whether the slab was read in.
///
/// The slab tiers map `MAP_SHARED` with no populate so an artifact LARGER than host RAM stays
/// loadable — that is the sizing case the disk tier exists for and the guard preserves it. When
/// the slab does fit, leaving it unpopulated is strictly worse than reading it once: the same
/// bytes are paid for regardless, but as demand faults on the single GPU-worker thread, at page
/// granularity, on the request's critical path.
pub fn populate_expert_slab(file: &File, len: usize, label: &str) -> bool {
    if len == 0 || expert_slab_populate() == ExpertSlabPopulate::Off {
        return false;
    }
    let (Some(avail), Some(total)) = (meminfo_bytes("MemAvailable"), meminfo_bytes("MemTotal"))
    else {
        return false;
    };
    if !expert_slab_populate_fits(len, avail, total, EXPERT_SLAB_RAM_FLOOR_FRAC) {
        eprintln!(
            "[slab-populate] skip {label}: {} MiB slab does not fit under the {:.0}% RAM floor \
             (MemAvailable {} MiB of {} MiB); demand-fault tier retained",
            len >> 20,
            EXPERT_SLAB_RAM_FLOOR_FRAC * 100.0,
            avail >> 20,
            total >> 20
        );
        return false;
    }
    match populate_pages(file, len) {
        Ok(elapsed) => {
            let secs = elapsed.as_secs_f64().max(1e-9);
            eprintln!(
                "[slab-populate] {label}: {} MiB in {:.1}s ({:.0} MiB/s)",
                len >> 20,
                elapsed.as_secs_f64(),
                (len >> 20) as f64 / secs
            );
            true
        }
        Err(err) => {
            eprintln!("[slab-populate] {label}: read failed ({err}); demand-fault tier retained");
            false
        }
    }
}

/// A view of one tensor's data, source-agnostic.
///
/// `bytes` is a `Cow`: the GGUF path and the zero-copy dense safetensors path BORROW the mmap
/// (no allocation); the hybrid-SSM transforms (`-exp(A_log)`, norm `+1`, conv1d squeeze, V-reorder)
/// produce an OWNED buffer (ST-MOE-PLAN §2.1) since they cannot be expressed as a borrow of the
/// on-disk bytes. All consumers read it as `&[u8]` via `&v.bytes`, so the fast path is untouched.
pub struct TensorView<'a> {
    pub bytes: Cow<'a, [u8]>,
    pub ggml_type: GgmlType,
    pub ne: Vec<u64>, // inner-fastest (ne[0] = in_features for a [in,out] weight)
}

/// Raw NVFP4-native (modelopt/Reza) weight view: the packed e2m1 codes + per-16 UE4M3 scales
/// exactly as they sit in the file, for the engine's DIRECT split-plane repack (A1 direct import).
/// Only returned for a PLAIN (untransformed) quantized Linear; anything needing a V-reorder
/// transform keeps the GGUF-block path. The per-tensor macro-scale still rides the `<stem>.scale`
/// sibling via `find` (identical to the GGUF NVFP4 path).
pub struct Nvfp4Native<'a> {
    pub wbytes: &'a [u8], // packed e2m1, [out_f, in_f/2] row-major, 2 codes/byte
    // UE4M3 per-16 scales, ALWAYS linear [out_f, in_f/16] row-major: borrowed from the mmap for a
    // linear checkpoint, owned for one that stores them swizzled (see `Nvfp4ScaleLayout`).
    pub wscale: Cow<'a, [u8]>,
    pub out_f: usize,
    pub in_f: usize,
}

/// Raw FP8-E4M3-native weight view (MEMRA_PP_FP8 prefill operand / MEMRA_ST_E4M3 resident copy):
/// the checkpoint's e4m3 codes `[out_f, in_f]` row-major + its weight scale(s). For a PLAIN
/// (untransformed) F8 Linear this borrows the mmap verbatim (EXACT bytes); for the hybrid
/// V-reordered F8 projections it is an OWNED buffer produced by the f32 transform round-trip —
/// exact too, since the V-reorder is a pure permutation and every f32 value is
/// `e4m3_code * scale`, which the nearest-e4m3 re-encode of `value/scale` recovers bit-for-bit
/// (grid spacing >> f32 rounding).
///
/// Scale carriage, two mutually exclusive forms:
///  * per-tensor: `scale` is the dequant multiplier, `blk == None` (modelopt / NVIDIA class).
///  * block-128: `blk == Some(grid)` and `scale == 1.0` (Qwen official FP8 class). The grid is
///    the checkpoint's `weight_scale_inv` decoded to f32, in its ON-DISK order — see
///    `F8BlockGrid` for the layout contract.
pub struct Fp8Native<'a> {
    pub bytes: Cow<'a, [u8]>,     // e4m3 codes, [out_f, in_f] row-major
    pub scale: f32, // per-tensor weight_scale (dequant multiplier); 1.0 when blk is Some
    pub blk: Option<F8BlockGrid>, // block-128 fine-grained scales (Qwen official FP8)
    pub out_f: usize,
    pub in_f: usize,
}

/// Raw block-128 E4M3 stacked expert bank. Step stores routed projections as
/// `[n_expert, out_f, in_f]` codes plus `[n_expert, ceil(out_f/128), ceil(in_f/128)]` scales.
/// Keeping the expert axis explicit prevents a 3-D bank from entering a 2-D linear path.
pub struct Fp8StackedNative<'a> {
    pub bytes: &'a [u8],
    pub scales: Vec<f32>,
    pub n_expert: usize,
    pub out_f: usize,
    pub in_f: usize,
    pub scale_rows: usize,
    pub scale_cols: usize,
}

/// Raw stacked modelopt NVFP4 expert bank. Step-3.7-Flash-NVFP4 stores each routed projection as
/// `[n_expert, out_f, in_f/2]` packed e2m1 codes (U8, 2 codes/byte) plus
/// `[n_expert, out_f, in_f/16]` per-16 UE4M3 scales (F8_E4M3 bytes) plus a per-EXPERT
/// `weight_scale_2` F32 macro. Dequant = `e2m1(code) * ue4m3(scale) * macros[e]` — the macros in
/// the official checkpoint run ~1e-5..1e-4 and are LOAD-BEARING; a consumer that drops them
/// produces garbage. Keeping the expert axis explicit prevents a 3-D bank from entering a 2-D
/// linear path, mirroring `Fp8StackedNative`.
pub struct Nvfp4StackedNative<'a> {
    pub codes: &'a [u8],  // [n_expert, out_f, in_f/2] packed e2m1, row-major
    pub scales: &'a [u8], // [n_expert, out_f, in_f/16] UE4M3 bytes, row-major
    pub macros: Vec<f32>, // [n_expert] weight_scale_2 dequant multipliers
    pub n_expert: usize,
    pub out_f: usize,
    pub in_f: usize,
}

/// Host-side block-128 scale grid, decoded to f32 in checkpoint order.
///
/// LAYOUT (the storage decision B1b pins, P1 consumes): row-major
/// `[rows = ceil(out_f/128), cols = ceil(in_f/128)]`, i.e. `scales[ob * cols + kb]` scales the
/// 128x128 tile at output-block `ob`, input-block `kb`. Equivalence note for the cuBLASLt
/// consumer: the weight `[out, in]` row-major is the GEMM's column-major `[k=in, n=out]` A^T
/// operand, and this linear order — `ob` outer, `kb` inner — is column-major `[kblk, nblk]`
/// with leading dimension `kblk` for that view. Whether cuBLASLt's
/// `CUBLASLT_MATMUL_MATRIX_SCALE_BLK128x128_32F` accepts this order on sm_120 at all is what
/// probe P1 (`probe/fp8_lt_blk_probe.cu`) answers; until then this struct is the single
/// canonical layout and any consumer-side reorder happens at that consumer's build step.
pub struct F8BlockGrid {
    pub scales: Vec<f32>,
    pub rows: usize, // ceil(out_f/128)
    pub cols: usize, // ceil(in_f/128)
}

/// One immutable expert extent backed by an opened file and its whole-file mmap. Retaining both
/// handles lets the engine choose explicit positioned I/O later while keeping the mmap bytes as the
/// permanent correctness fallback. `offset` is absolute within both the file and the whole mapping.
#[derive(Clone)]
pub struct DiskExtent {
    pub map: Arc<Mmap>,
    pub file: Arc<File>,
    pub offset: u64,
    pub len: usize,
}

/// One header-only census row for a source tensor.
///
/// `entry.physical_bytes` is the exact number of checkpoint bytes represented by the semantic
/// row. Safetensors quantization auxiliaries are folded into their owning weight row because the
/// tensor contract models them as one storage layout; GGUF auxiliaries remain independent rows.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorCensusRecord {
    pub physical_name: String,
    pub dtype: String,
    pub entry: TensorCensusEntry,
}

/// A complete source census and the naming dialect its semantic names use.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorCensus {
    pub dialect: CheckpointDialect,
    pub tensors: Vec<TensorCensusRecord>,
}

fn info_physical_bytes(name: &str, info: &StInfo) -> Result<u64, String> {
    let bytes = info.data_offsets[1]
        .checked_sub(info.data_offsets[0])
        .ok_or_else(|| format!("safetensors tensor {name} has reversed data offsets"))?;
    u64::try_from(bytes).map_err(|_| format!("safetensors tensor {name} byte length overflows u64"))
}

/// Normalize wrapper prefixes that do not change a tensor's semantic identity.
pub fn canonical_hf_name(name: &str) -> String {
    if let Some(suffix) = name.strip_prefix("model.language_model.") {
        return format!("model.{suffix}");
    }
    if let Some(suffix) = name.strip_prefix("language_model.model.") {
        return format!("model.{suffix}");
    }
    if let Some(suffix) = name.strip_prefix("language_model.lm_head.") {
        return format!("lm_head.{suffix}");
    }
    name.to_string()
}

fn ggml_storage(kind: GgmlType) -> StorageLayout {
    match kind {
        GgmlType::F32 => StorageLayout::Float(FloatType::F32),
        GgmlType::F16 => StorageLayout::Float(FloatType::F16),
        GgmlType::BF16 => StorageLayout::Float(FloatType::Bf16),
        GgmlType::I64 => StorageLayout::Integer(IntegerType::I64),
        other => {
            let (block, _) = other.block_and_type_size();
            StorageLayout::Quantized(QuantLayout {
                format: format!("{other:?}"),
                block_shape: vec![block as u32],
                auxiliaries: Vec::new(),
            })
        }
    }
}

/// Build an exact-byte GGUF census from tensor-table metadata only.
pub fn census_from_gguf(gguf: &GgufFile) -> TensorCensus {
    TensorCensus {
        dialect: CheckpointDialect::Gguf,
        tensors: gguf
            .tensors
            .iter()
            .map(|tensor| TensorCensusRecord {
                physical_name: tensor.name.clone(),
                dtype: format!("{:?}", tensor.ggml_type),
                entry: TensorCensusEntry {
                    name: tensor.name.clone(),
                    shape: tensor.ne.clone(),
                    storage: ggml_storage(tensor.ggml_type),
                    physical_bytes: tensor.n_bytes,
                },
            })
            .collect(),
    }
}

fn is_quant_auxiliary(name: &str, headers: &BTreeMap<String, StInfo>) -> bool {
    [
        ".weight_scale",
        ".weight_scale_inv",
        ".weight_scale_2",
        ".weight_global_scale",
        ".input_scale",
        ".scale",
    ]
    .into_iter()
    .any(|suffix| {
        name.strip_suffix(suffix).is_some_and(|stem| {
            headers.contains_key(&format!("{stem}.weight"))
                || headers.contains_key(&format!("{stem}.weight_packed"))
        })
    })
}

fn safetensors_storage(
    info: &StInfo,
    auxiliaries: &[String],
) -> Result<(Vec<u64>, StorageLayout), String> {
    let float = match info.dtype.as_str() {
        "F32" => Some(FloatType::F32),
        "F16" => Some(FloatType::F16),
        "BF16" => Some(FloatType::Bf16),
        "F8_E4M3" if auxiliaries.is_empty() => Some(FloatType::Fp8E4m3),
        _ => None,
    };
    if let Some(float) = float {
        return Ok((info.shape.clone(), StorageLayout::Float(float)));
    }
    if info.dtype == "I64" && auxiliaries.is_empty() {
        return Ok((info.shape.clone(), StorageLayout::Integer(IntegerType::I64)));
    }
    if auxiliaries.is_empty() {
        return Err(format!(
            "unsupported standalone safetensors dtype {}",
            info.dtype
        ));
    }
    let mut shape = info.shape.clone();
    let (format, block_shape) = match info.dtype.as_str() {
        "U8" => {
            let last = shape
                .last_mut()
                .ok_or("packed U8 weight has no dimensions")?;
            *last = last
                .checked_mul(2)
                .ok_or("packed U8 logical shape overflows")?;
            ("NVFP4", vec![16])
        }
        "I8" => {
            let last = shape
                .last_mut()
                .ok_or("packed I8 weight has no dimensions")?;
            *last = last
                .checked_mul(2)
                .ok_or("packed I8 logical shape overflows")?;
            ("MXFP4", vec![32])
        }
        "F8_E4M3" => ("FP8_E4M3", vec![128, 128]),
        other => return Err(format!("unsupported quantized weight dtype {other}")),
    };
    Ok((
        shape,
        StorageLayout::Quantized(QuantLayout {
            format: format.to_string(),
            block_shape,
            auxiliaries: Vec::new(),
        }),
    ))
}

/// Build an exact-byte safetensors census from parsed headers only.
///
/// Quantization auxiliaries are represented by the owning weight's storage layout, matching the
/// tensor contract, so their physical byte ranges are included in that weight's byte total.
pub fn census_from_safetensors_headers(
    headers: &BTreeMap<String, StInfo>,
) -> Result<TensorCensus, String> {
    let mut auxiliary_names = BTreeSet::new();
    let mut rows = Vec::new();
    for (physical_name, info) in headers {
        if auxiliary_names.contains(physical_name) || is_quant_auxiliary(physical_name, headers) {
            continue;
        }
        let (semantic_physical_name, stem) = match physical_name.strip_suffix(".weight_packed") {
            Some(stem) => (format!("{stem}.weight"), Some(stem)),
            None => (physical_name.clone(), physical_name.strip_suffix(".weight")),
        };
        let auxiliaries: Vec<String> = stem
            .map(|stem| {
                [
                    format!("{stem}.weight_scale"),
                    format!("{stem}.weight_scale_inv"),
                    format!("{stem}.weight_scale_2"),
                    format!("{stem}.weight_global_scale"),
                    format!("{stem}.input_scale"),
                    format!("{stem}.scale"),
                ]
                .into_iter()
                .filter(|name| headers.contains_key(name))
                .collect()
            })
            .unwrap_or_default();
        auxiliary_names.extend(auxiliaries.iter().cloned());
        let (shape, mut storage) = safetensors_storage(info, &auxiliaries)?;
        if let StorageLayout::Quantized(layout) = &mut storage {
            layout.auxiliaries = auxiliaries
                .iter()
                .map(|name| canonical_hf_name(name))
                .collect();
        }
        let mut physical_bytes = info_physical_bytes(physical_name, info)?;
        for auxiliary in &auxiliaries {
            physical_bytes = physical_bytes
                .checked_add(info_physical_bytes(auxiliary, &headers[auxiliary])?)
                .ok_or_else(|| {
                    format!("safetensors tensor {physical_name} byte total overflows")
                })?;
        }
        rows.push(TensorCensusRecord {
            physical_name: physical_name.clone(),
            dtype: info.dtype.clone(),
            entry: TensorCensusEntry {
                name: canonical_hf_name(&semantic_physical_name),
                shape,
                storage,
                physical_bytes,
            },
        });
    }
    rows.sort_by(|left, right| left.entry.name.cmp(&right.entry.name));
    let mut names = BTreeSet::new();
    for row in &rows {
        if !names.insert(&row.entry.name) {
            return Err(format!(
                "multiple physical tensors normalize to {}",
                row.entry.name
            ));
        }
    }
    Ok(TensorCensus {
        dialect: CheckpointDialect::HfSafetensors,
        tensors: rows,
    })
}

fn census_from_safetensors_model(model: &StModel) -> Result<TensorCensus, String> {
    let headers = model
        .names()
        .map(|name| {
            model
                .info(name)
                .cloned()
                .map(|info| (name.clone(), info))
                .ok_or_else(|| format!("safetensors header disappeared for {name}"))
        })
        .collect::<Result<BTreeMap<_, _>, _>>()?;
    census_from_safetensors_headers(&headers)
}

/// Artifact-level activation precision for routed expert linears.
///
/// This is deliberately source metadata, not an architecture switch: two checkpoints with the
/// same ModelPlan may carry different activation contracts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpertActivationPrecision {
    F32,
    Bf16,
    Quantized,
}

fn expert_activation_precision_from_quant_algo(
    quant_algo: Option<&str>,
) -> ExpertActivationPrecision {
    match quant_algo {
        Some("W4A16_NVFP4") => ExpertActivationPrecision::Bf16,
        Some("NVFP4" | "W4A4_NVFP4") => ExpertActivationPrecision::Quantized,
        _ => ExpertActivationPrecision::F32,
    }
}

/// A weight source the engine can load from. GGUF and safetensors both implement it.
pub trait TensorSource {
    /// The model configuration (from GGUF metadata or config.json).
    fn config(&self) -> ModelConfig;
    /// Fallible configuration boundary for untrusted model artifacts. Legacy callers retain the
    /// infallible config API, while serving loaders use this wrapper so malformed required
    /// metadata becomes a startup error rather than unwinding the worker thread.
    fn try_config(&self) -> Result<ModelConfig, String> {
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| self.config())).map_err(
            |payload| {
                payload
                    .downcast_ref::<String>()
                    .cloned()
                    .or_else(|| payload.downcast_ref::<&str>().map(|s| (*s).to_string()))
                    .unwrap_or_else(|| "model metadata parser panicked".into())
            },
        )
    }
    /// Enumerate tensor metadata and exact checkpoint byte ranges without reading or transforming
    /// payload bytes. Real GGUF, safetensors, and manifest sources implement this; synthetic test
    /// sources may leave it unavailable.
    fn tensor_census(&self) -> Result<TensorCensus, String> {
        Err("tensor source does not expose a metadata-only census".to_string())
    }
    /// Routed-expert activation precision declared by the artifact.
    fn expert_activation_precision(&self) -> ExpertActivationPrecision {
        ExpertActivationPrecision::F32
    }
    /// Find a tensor by its **ggml-style** name. Returns None if absent/unmapped.
    fn find(&self, ggml_name: &str) -> Option<TensorView<'_>>;
    /// Whether a ggml-named tensor is present.
    fn has(&self, ggml_name: &str) -> bool {
        self.find(ggml_name).is_some()
    }
    /// The backing GGUF file, if this source IS a GGUF (None for safetensors). Used by the
    /// disk-spill tier, which needs the on-disk file mmap (`g.path()` / per-expert byte ranges).
    fn gguf(&self) -> Option<&GgufFile> {
        None
    }
    /// NVFP4-native access (A1 direct import): the raw modelopt/Reza packed codes + scales for a
    /// plain (untransformed) NVFP4 weight, so the engine can repack straight into its split-plane
    /// resident layout without materializing GGUF 36B blocks. None for GGUF sources (already the
    /// import layout), for transformed weights, and for non-NVFP4 tensors.
    fn find_nvfp4_native(&self, _ggml_name: &str) -> Option<Nvfp4Native<'_>> {
        None
    }
    /// FP8-E4M3-native access (MEMRA_PP_FP8 prefill operand): the raw e4m3 codes + per-tensor f32
    /// `weight_scale` for an F8-sourced 2D projection, so the engine can stash the exact weight
    /// bytes for the cuBLASLt FP8 prefill GEMM alongside its Q8_0 re-encode. None for GGUF
    /// sources and for anything that is not an F8-E4M3 2D Linear.
    fn find_fp8_native(&self, _ggml_name: &str) -> Option<Fp8Native<'_>> {
        None
    }
    /// Native access for a stacked expert bank. This is deliberately distinct from
    /// `find_fp8_native`: expert and scale-grid strides are part of the checkpoint contract.
    fn find_fp8_stacked_native(&self, _ggml_name: &str) -> Option<Fp8StackedNative<'_>> {
        None
    }
    /// Native access for a stacked modelopt NVFP4 expert bank (Step-3.7-Flash-NVFP4 class).
    /// Deliberately distinct from `find_nvfp4_native` (2-D Linear) and `find_fp8_stacked_native`
    /// (E4M3 bank): the packed-code, per-16 scale-grid, and per-expert macro strides are part of
    /// the checkpoint contract.
    fn find_nvfp4_stacked_native(&self, _ggml_name: &str) -> Option<Nvfp4StackedNative<'_>> {
        None
    }
    /// The checkpoint directory, if this source is a safetensors HF dir (None for GGUF). Used by
    /// the ST expert disk-tier to place its repack cache next to the shards.
    fn st_dir(&self) -> Option<&std::path::Path> {
        None
    }
    /// A short tag naming the on-disk NVFP4 scale layout this source READ, for the expert repack
    /// cache key. Empty for the historical linear layout, so existing cache filenames are
    /// unchanged and every pre-2026-09-04 cache stays valid.
    ///
    /// THE TRAP THIS CLOSES. The repack cache lives inside the checkpoint directory and its
    /// freshness check is SIZE-ONLY (`repack_cache_is_fresh`). Reading the same checkpoint under a
    /// different scale layout produces a cache of exactly the same size, so a cache written by a
    /// binary that read a swizzled plane as linear would be silently reused by a binary that
    /// reads it correctly -- reinstating, from disk, the very corruption `Nvfp4ScaleLayout` exists
    /// to prevent, and doing it invisibly because the load path never re-reads the source. Putting
    /// the layout in the cache KEY means the two readings can never collide: a stale cache is
    /// simply not found, and is rebuilt.
    fn nvfp4_cache_tag(&self) -> &'static str {
        ""
    }
    /// Whether per-expert tensor encodings exposed by this source are an intentional artifact
    /// contract and must be retained even when every expert happens to use the same encoding.
    /// Sparse overlays use this so an all-Q4_K control does not get normalized back to F32.
    fn preserve_expert_encodings(&self) -> bool {
        false
    }
    /// Optional per-layer routed-expert mask. A false entry is physically absent from a pruned
    /// overlay and must be excluded before top-k routing. Keeping the original router width makes
    /// usage-driven pruning possible without rewriting router tensors or expert ids.
    fn active_experts(&self, _layer: u32) -> Option<&[bool]> {
        None
    }
    /// Zero-copy mmap window for an expert tensor (the disk-spill tier's byte source):
    /// `(shared file mmap, tensor byte offset, tensor bytes)`. This covers both stacked 3D expert
    /// slabs and v2 per-expert overlay entries. The engine then backs `HostExps` with
    /// `HostBuf::Mmap` directly (page cache = RAM tier, faults = NVMe tier) instead of copying a
    /// potentially >RAM expert set. None for sources that require gathering or repacking.
    fn find_expert_disk(&self, _ggml_name: &str) -> Option<DiskExtent> {
        None
    }
    /// Compatibility view for callers that only need mmap access. New disk-aware consumers should
    /// use `find_expert_disk` so the opened file remains available after the source is dropped.
    fn find_expert_mmap(&self, ggml_name: &str) -> Option<(Arc<Mmap>, usize, usize)> {
        let extent = self.find_expert_disk(ggml_name)?;
        let offset = usize::try_from(extent.offset).ok()?;
        Some((extent.map, offset, extent.len))
    }
}

/// GGUF-backed source (the existing path). Zero behavior change vs. direct GgufFile use.
pub struct GgufSource<'g>(pub &'g GgufFile);

impl<'g> TensorSource for GgufSource<'g> {
    fn config(&self) -> ModelConfig {
        ModelConfig::from_gguf(self.0)
    }
    fn tensor_census(&self) -> Result<TensorCensus, String> {
        Ok(census_from_gguf(self.0))
    }
    fn find(&self, name: &str) -> Option<TensorView<'_>> {
        let t = self.0.find(name)?;
        Some(TensorView {
            bytes: Cow::Borrowed(self.0.tensor_data(t)),
            ggml_type: t.ggml_type,
            ne: t.ne.clone(),
        })
    }
    fn gguf(&self) -> Option<&GgufFile> {
        Some(self.0)
    }
}

#[derive(Debug, Clone)]
struct RepackTensor {
    file: PathBuf,
    offset: usize,
    ggml_type: GgmlType,
    ne: Vec<u64>,
    bytes: usize,
    expert_stride: Option<usize>,
}

struct RepackFile {
    // Only expert files retain an fd. Dense-only manifest files keep their mmap but do not consume
    // the process fd budget because positioned I/O is never requested for them.
    file: Option<Arc<File>>,
    map: Arc<Mmap>,
}

#[allow(clippy::large_enum_variant)] // allow: variant size asymmetry is deliberate; these enums live in per-layer tables, not hot moves
enum RepackFallback {
    Safetensors(SafetensorsSource),
    Repack(Box<Hy3RepackSource>),
}

impl RepackFallback {
    fn config(&self) -> ModelConfig {
        match self {
            Self::Safetensors(source) => source.config(),
            Self::Repack(source) => source.config(),
        }
    }

    fn find(&self, name: &str) -> Option<TensorView<'_>> {
        match self {
            Self::Safetensors(source) => source.find(name),
            Self::Repack(source) => source.find(name),
        }
    }

    fn tensor_census(&self) -> Result<TensorCensus, String> {
        match self {
            Self::Safetensors(source) => source.tensor_census(),
            Self::Repack(source) => source.tensor_census(),
        }
    }

    fn expert_activation_precision(&self) -> ExpertActivationPrecision {
        match self {
            Self::Safetensors(source) => source.expert_activation_precision(),
            Self::Repack(source) => source.expert_activation_precision(),
        }
    }

    fn find_nvfp4_native(&self, name: &str) -> Option<Nvfp4Native<'_>> {
        match self {
            Self::Safetensors(source) => source.find_nvfp4_native(name),
            Self::Repack(source) => source.find_nvfp4_native(name),
        }
    }

    fn find_fp8_native(&self, name: &str) -> Option<Fp8Native<'_>> {
        match self {
            Self::Safetensors(source) => source.find_fp8_native(name),
            Self::Repack(source) => source.find_fp8_native(name),
        }
    }

    fn find_fp8_stacked_native(&self, name: &str) -> Option<Fp8StackedNative<'_>> {
        match self {
            Self::Safetensors(source) => source.find_fp8_stacked_native(name),
            Self::Repack(source) => source.find_fp8_stacked_native(name),
        }
    }

    fn find_nvfp4_stacked_native(&self, name: &str) -> Option<Nvfp4StackedNative<'_>> {
        match self {
            Self::Safetensors(source) => source.find_nvfp4_stacked_native(name),
            Self::Repack(source) => source.find_nvfp4_stacked_native(name),
        }
    }

    fn st_dir(&self) -> Option<&Path> {
        match self {
            Self::Safetensors(source) => source.st_dir(),
            Self::Repack(source) => source.st_dir(),
        }
    }
}

/// Manifest-backed source for memra repack directories and sparse per-expert overlays.
///
/// The transcoder writes one file per tensor (including stacked expert slabs) plus a manifest with
/// ggml-style names. This source presents those bytes directly to the existing loaders without a
/// single-file GGUF wrapper. Expert overlays may fall back to either an HF checkpoint or another
/// manifest-backed repack; the latter lets a multi-tier expert artifact reuse the established Hy3
/// dense/router repack without copying it. Files are mmap'd lazily by the OS; opening the 80G
/// repack maps address space but does not fault tensor pages into RAM. The public type retains its
/// historical Hy3 name for compatibility with existing callers.
pub struct Hy3RepackSource {
    cfg: ModelConfig,
    expert_activation_precision: ExpertActivationPrecision,
    dir: PathBuf,
    source_dir: Option<PathBuf>,
    tensors: BTreeMap<String, RepackTensor>,
    // Retain both handles so stacked expert slabs can be handed to the engine's disk-aware mmap tier
    // while `find` keeps borrowing the same mapping.
    files: BTreeMap<PathBuf, RepackFile>,
    // Expert overlays store only overridden tensors. Everything else resolves from either the
    // original HF checkpoint (v1) or a complete manifest repack (v2).
    fallback: Option<RepackFallback>,
    active_experts: BTreeMap<u32, Vec<bool>>,
}

impl Hy3RepackSource {
    pub fn open(path: &Path) -> std::io::Result<Self> {
        let manifest = if path.is_dir() {
            path.join("manifest.json")
        } else {
            path.to_path_buf()
        };
        let dir = manifest
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .to_path_buf();
        let txt = std::fs::read_to_string(&manifest)?;
        let top = JsonObj::parse(&txt);
        let tensors_obj = top
            .object("tensors")
            .ok_or_else(|| invalid_data("manifest missing tensors object"))?;
        let mut tensors = BTreeMap::new();
        for (name, raw) in tensors_obj.fields() {
            let obj = JsonObj::parse(raw);
            let file = obj
                .string("file")
                .ok_or_else(|| invalid_data(format!("manifest tensor {name} missing file")))?;
            let file = validated_manifest_file(&file).map_err(|reason| {
                invalid_data(format!(
                    "manifest tensor {name} has invalid file path: {reason}"
                ))
            })?;
            let qtype = obj
                .string("qtype")
                .ok_or_else(|| invalid_data(format!("manifest tensor {name} missing qtype")))?;
            let ne = obj
                .u64_array("ne")
                .ok_or_else(|| invalid_data(format!("manifest tensor {name} missing ne")))?;
            let raw_bytes = obj
                .u64("bytes")
                .ok_or_else(|| invalid_data(format!("manifest tensor {name} missing bytes")))?;
            let bytes = usize::try_from(raw_bytes).map_err(|_| {
                invalid_data(format!(
                    "manifest tensor {name} bytes={raw_bytes} does not fit this platform"
                ))
            })?;
            let raw_offset = obj.u64("offset").unwrap_or(0);
            let offset = usize::try_from(raw_offset).map_err(|_| {
                invalid_data(format!(
                    "manifest tensor {name} offset={raw_offset} does not fit this platform"
                ))
            })?;
            let ggml_type = manifest_qtype(&qtype).ok_or_else(|| {
                invalid_data(format!("manifest tensor {name} unsupported qtype {qtype}"))
            })?;
            let expected = ggml_type.checked_nbytes(&ne).ok_or_else(|| {
                invalid_data(format!(
                    "manifest tensor {name} has invalid or overflowing {ggml_type:?} geometry {ne:?}"
                ))
            })?;
            if expected != raw_bytes {
                return Err(invalid_data(format!(
                    "manifest tensor {name} declares {raw_bytes} bytes but {ggml_type:?} geometry {ne:?} encodes exactly {expected}"
                )));
            }
            tensors.insert(
                name.to_string(),
                RepackTensor {
                    file,
                    offset,
                    ggml_type,
                    ne,
                    bytes,
                    expert_stride: obj.u64("expert_stride").map(|x| x as usize),
                },
            );
        }

        let source_dir = top.string("source_dir").map(PathBuf::from).map(|path| {
            if path.is_absolute() {
                path
            } else {
                dir.join(path)
            }
        });
        let format = top.string("format");
        // "bw24-*" is the pre-rename spelling: published overlay artifacts (e.g. the Hy3
        // layer103.5 runtime manifest, sha-pinned on HF) carry it on disk and must keep
        // loading byte-identical after the memra rename.
        let is_overlay = matches!(
            format.as_deref(),
            Some(
                "memra-expert-overlay-v1"
                    | "memra-expert-overlay-v2"
                    | "bw24-expert-overlay-v1"
                    | "bw24-expert-overlay-v2"
            )
        );
        let fallback = if is_overlay {
            let source = source_dir
                .as_deref()
                .ok_or_else(|| invalid_data("expert overlay manifest missing source_dir"))?;
            if source.join("manifest.json").exists() {
                Some(RepackFallback::Repack(Box::new(Hy3RepackSource::open(
                    source,
                )?)))
            } else {
                Some(RepackFallback::Safetensors(SafetensorsSource::open(
                    source,
                )?))
            }
        } else {
            None
        };
        let mut cfg = if let Some(source) = &fallback {
            source.config()
        } else {
            let cfg_path = source_dir
                .clone()
                .map(|p| p.join("config.json"))
                .filter(|p| p.exists())
                .unwrap_or_else(|| dir.join("config.json"));
            ModelConfig::from_config_json(&cfg_path)?
        };
        // A complete repack can intentionally omit the appended MTP block. An expert overlay is
        // sparse by definition: tensors absent from its manifest resolve through the fallback, so
        // its highest overridden block says nothing about whether the fallback still has MTP.
        if fallback.is_none() {
            apply_stripped_mtp_override(&mut cfg, &tensors);
        }
        let expert_activation_precision = fallback
            .as_ref()
            .map(RepackFallback::expert_activation_precision)
            .unwrap_or_else(|| {
                let cfg_path = source_dir
                    .clone()
                    .map(|path| path.join("config.json"))
                    .filter(|path| path.exists())
                    .unwrap_or_else(|| dir.join("config.json"));
                let quant_algo = std::fs::read_to_string(cfg_path)
                    .ok()
                    .map(|json| crate::config::HfConfig::parse(&json))
                    .and_then(|config| config.quant_algo);
                expert_activation_precision_from_quant_algo(quant_algo.as_deref())
            });
        let mut active_experts = BTreeMap::new();
        if let Some(pruned) = top.object("pruned_experts") {
            let moe = cfg.moe.as_ref().ok_or_else(|| {
                invalid_data("pruned_experts is present but the model config has no MoE")
            })?;
            let n_expert = moe.expert_count as usize;
            let n_used = moe.expert_used_count as usize;
            for (layer, raw) in pruned.fields() {
                let layer: u32 = layer.parse().map_err(|_| {
                    invalid_data(format!("invalid pruned_experts layer key {layer:?}"))
                })?;
                let wrapper = JsonObj::parse(&format!("{{\"v\":{raw}}}"));
                let ids = wrapper.u64_array("v").ok_or_else(|| {
                    invalid_data(format!("pruned_experts.{layer} must be an integer array"))
                })?;
                let mut mask = vec![true; n_expert];
                for id in ids {
                    let id = id as usize;
                    if id >= n_expert {
                        return Err(invalid_data(format!(
                            "pruned_experts.{layer} contains {id}, expert_count={n_expert}"
                        )));
                    }
                    mask[id] = false;
                }
                if mask.iter().filter(|&&active| active).count() < n_used {
                    return Err(invalid_data(format!(
                        "pruned_experts.{layer} leaves fewer than top-k {n_used} experts"
                    )));
                }
                active_experts.insert(layer, mask);
            }
        }

        let expert_files: BTreeSet<PathBuf> = tensors
            .iter()
            .filter(|(name, _)| name.contains("_exps."))
            .map(|(_, tensor)| tensor.file.clone())
            .collect();
        let mut files = BTreeMap::new();
        let mut seen = BTreeSet::new();
        for t in tensors.values() {
            if !seen.insert(t.file.clone()) {
                continue;
            }
            let file = Arc::new(open_repack_shard(&dir, &t.file)?);
            let map = unsafe { Mmap::map(file.as_ref())? };
            let retain_file = t.expert_stride.is_some() || expert_files.contains(&t.file);
            // Expert slabs use the configured whole-map policy. Default random is the historical
            // behavior; normal restores Linux readahead for multi-megabyte expert reads.
            if retain_file {
                let _ = apply_expert_mmap_advice(&map);
            }
            files.insert(
                t.file.clone(),
                RepackFile {
                    file: retain_file.then_some(file),
                    map: Arc::new(map),
                },
            );
        }
        for (name, t) in &tensors {
            let len = files
                .get(&t.file)
                .ok_or_else(|| invalid_data(format!("manifest tensor {name} file not mapped")))?
                .map
                .len();
            let end = t.offset.checked_add(t.bytes).ok_or_else(|| {
                invalid_data(format!(
                    "manifest tensor {name} offset {} + {} bytes overflows this platform",
                    t.offset, t.bytes
                ))
            })?;
            if len < end {
                return Err(invalid_data(format!(
                    "manifest tensor {name} declares offset {} + {} bytes but {:?} has {len}",
                    t.offset, t.bytes, t.file
                )));
            }
        }

        Ok(Self {
            cfg,
            expert_activation_precision,
            dir,
            source_dir,
            tensors,
            files,
            fallback,
            active_experts,
        })
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// The original HF checkpoint dir recorded by the transcoder (tokenizer files live there —
    /// the repack dir carries only weights + manifest).
    pub fn source_dir(&self) -> Option<&Path> {
        match self.fallback.as_ref() {
            Some(RepackFallback::Safetensors(source)) => source.st_dir(),
            Some(RepackFallback::Repack(source)) => source.source_dir(),
            None => self.source_dir.as_deref(),
        }
    }

    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }

    pub fn expert_stride(&self, ggml_name: &str) -> Option<usize> {
        self.tensors.get(ggml_name).and_then(|t| t.expert_stride)
    }
}

impl TensorSource for Hy3RepackSource {
    fn config(&self) -> ModelConfig {
        self.cfg.clone()
    }

    fn tensor_census(&self) -> Result<TensorCensus, String> {
        let (dialect, mut rows) = match self.fallback.as_ref() {
            Some(fallback) => {
                let census = fallback.tensor_census()?;
                (census.dialect, census.tensors)
            }
            None => (CheckpointDialect::Gguf, Vec::new()),
        };
        let semantic_name = |name: &str| {
            if dialect == CheckpointDialect::HfSafetensors {
                use crate::hf_mapping::{HfTarget, resolve_ggml};
                if let Some(target) = resolve_ggml(name, &self.cfg) {
                    let hf = match target {
                        HfTarget::Plain(hf) | HfTarget::Transform { hf, .. } => hf,
                    };
                    return canonical_hf_name(&hf);
                }
            }
            name.to_string()
        };

        let mut by_name: BTreeMap<String, TensorCensusRecord> = rows
            .drain(..)
            .map(|row| (row.entry.name.clone(), row))
            .collect();
        if dialect == CheckpointDialect::Gguf {
            // A per-expert overlay masks the fallback's uniform stacked bank exactly as `find`
            // does. Exclude that unreachable fallback allocation from the effective census.
            for name in self.tensors.keys() {
                if let Some(prefix) = name.strip_suffix(".weight")
                    && let Some((bank, expert)) = prefix.rsplit_once('.')
                    && expert.parse::<usize>().is_ok()
                    && bank.ends_with("_exps")
                {
                    by_name.remove(&format!("{bank}.weight"));
                }
            }
        }
        for (name, tensor) in &self.tensors {
            let semantic = semantic_name(name);
            let physical_bytes = u64::try_from(tensor.bytes)
                .map_err(|_| format!("manifest tensor {name} byte length overflows u64"))?;
            let shape = if dialect == CheckpointDialect::HfSafetensors {
                tensor.ne.iter().rev().copied().collect()
            } else {
                tensor.ne.clone()
            };
            by_name.insert(
                semantic.clone(),
                TensorCensusRecord {
                    physical_name: name.clone(),
                    dtype: format!("{:?}", tensor.ggml_type),
                    entry: TensorCensusEntry {
                        name: semantic,
                        shape,
                        storage: ggml_storage(tensor.ggml_type),
                        physical_bytes,
                    },
                },
            );
        }
        Ok(TensorCensus {
            dialect,
            tensors: by_name.into_values().collect(),
        })
    }

    fn expert_activation_precision(&self) -> ExpertActivationPrecision {
        self.expert_activation_precision
    }

    fn find(&self, ggml_name: &str) -> Option<TensorView<'_>> {
        if let Some(t) = self.tensors.get(ggml_name) {
            let file = self.files.get(&t.file)?;
            let raw = &file.map[t.offset..t.offset + t.bytes];
            if t.ggml_type == GgmlType::BF16 && t.ne.len() == 1 {
                let n: u64 = t.ne.iter().product();
                let vals = crate::dequant::dequantize(t.ggml_type, raw, n as usize);
                let mut bytes = Vec::with_capacity(vals.len() * 4);
                for f in vals {
                    bytes.extend_from_slice(&f.to_le_bytes());
                }
                return Some(TensorView {
                    bytes: Cow::Owned(bytes),
                    ggml_type: GgmlType::F32,
                    ne: t.ne.clone(),
                });
            }
            return Some(TensorView {
                bytes: Cow::Borrowed(raw),
                ggml_type: t.ggml_type,
                ne: t.ne.clone(),
            });
        }
        // A v2 overlay stores mixed experts separately while its fallback may contain the old
        // uniform stacked slab. Do not let that slab bypass the per-expert override loader.
        if ggml_name.contains("_exps.weight") {
            let prefix = format!("{}.", ggml_name.strip_suffix(".weight")?);
            if self
                .tensors
                .range(prefix.clone()..)
                .next()
                .is_some_and(|(name, _)| name.starts_with(&prefix))
            {
                return None;
            }
        }
        self.fallback.as_ref()?.find(ggml_name)
    }

    fn find_nvfp4_native(&self, ggml_name: &str) -> Option<Nvfp4Native<'_>> {
        if self.tensors.contains_key(ggml_name) {
            return None;
        }
        self.fallback.as_ref()?.find_nvfp4_native(ggml_name)
    }

    fn find_fp8_native(&self, ggml_name: &str) -> Option<Fp8Native<'_>> {
        if self.tensors.contains_key(ggml_name) {
            return None;
        }
        self.fallback.as_ref()?.find_fp8_native(ggml_name)
    }

    fn find_fp8_stacked_native(&self, ggml_name: &str) -> Option<Fp8StackedNative<'_>> {
        if self.tensors.contains_key(ggml_name) {
            return None;
        }
        self.fallback.as_ref()?.find_fp8_stacked_native(ggml_name)
    }

    fn find_nvfp4_stacked_native(&self, ggml_name: &str) -> Option<Nvfp4StackedNative<'_>> {
        if self.tensors.contains_key(ggml_name) {
            return None;
        }
        self.fallback.as_ref()?.find_nvfp4_stacked_native(ggml_name)
    }

    fn st_dir(&self) -> Option<&Path> {
        self.fallback.as_ref().and_then(RepackFallback::st_dir)
    }

    fn preserve_expert_encodings(&self) -> bool {
        self.fallback.is_some()
    }

    fn active_experts(&self, layer: u32) -> Option<&[bool]> {
        self.active_experts.get(&layer).map(Vec::as_slice)
    }

    /// Hand stacked expert slabs and v2 per-expert overlay entries to the engine as shared mmap
    /// windows. Both layouts are already kernel-ready; copying them into Vec would make the 161 GB
    /// full-bank control impossible on a 124 GB host.
    fn find_expert_disk(&self, ggml_name: &str) -> Option<DiskExtent> {
        let t = self.tensors.get(ggml_name)?;
        let file = self.files.get(&t.file)?;
        Some(DiskExtent {
            map: file.map.clone(),
            file: file.file.as_ref()?.clone(),
            offset: t.offset as u64,
            len: t.bytes,
        })
    }
}

fn manifest_qtype(s: &str) -> Option<GgmlType> {
    Some(match s {
        "F32" => GgmlType::F32,
        "F16" => GgmlType::F16,
        "BF16" => GgmlType::BF16,
        "Q8_0" => GgmlType::Q8_0,
        "Q2_K" => GgmlType::Q2_K,
        "Q4_K" => GgmlType::Q4_K,
        "Q5_K" => GgmlType::Q5_K,
        "Q6_K" => GgmlType::Q6_K,
        "Q3_K" => GgmlType::Q3_K,
        "IQ4_XS" => GgmlType::IQ4_XS,
        "IQ3_S" => GgmlType::IQ3_S,
        "NVFP4" => GgmlType::NVFP4,
        _ => return None,
    })
}

fn apply_stripped_mtp_override(cfg: &mut ModelConfig, tensors: &BTreeMap<String, RepackTensor>) {
    if cfg.nextn_predict_layers == 0 {
        return;
    }
    let max_blk = tensors
        .keys()
        .filter_map(|name| {
            let rest = name.strip_prefix("blk.")?;
            let (il, _) = rest.split_once('.')?;
            il.parse::<u32>().ok()
        })
        .max();
    let Some(max_blk) = max_blk else {
        return;
    };
    let manifest_layers = max_blk + 1;
    let trunk_layers = cfg.n_layer.saturating_sub(cfg.nextn_predict_layers);
    let has_nextn_names = tensors.keys().any(|name| name.contains(".nextn."));
    if manifest_layers <= trunk_layers && !has_nextn_names {
        cfg.n_layer = manifest_layers;
        cfg.n_layer_total = manifest_layers;
        cfg.nextn_predict_layers = 0;
    }
}

fn invalid_data(msg: impl Into<String>) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidData, msg.into())
}

fn validated_manifest_file(raw: &str) -> Result<PathBuf, &'static str> {
    let path = Path::new(raw);
    if raw.is_empty() || path.is_absolute() {
        return Err("expected a non-empty relative path");
    }
    if !path
        .components()
        .all(|component| matches!(component, std::path::Component::Normal(_)))
    {
        return Err("absolute, parent, and current-directory components are forbidden");
    }
    Ok(path.to_path_buf())
}

fn repack_trusted_roots(dir: &Path) -> std::io::Result<Vec<PathBuf>> {
    let mut roots = vec![std::fs::canonicalize(dir)?];
    let Some(raw) = std::env::var_os("MEMRA_REPACK_EXTERNAL_ROOTS") else {
        return Ok(roots);
    };
    for root in std::env::split_paths(&raw) {
        if !root.is_absolute() {
            return Err(invalid_data(format!(
                "MEMRA_REPACK_EXTERNAL_ROOTS entry {} is not absolute",
                root.display()
            )));
        }
        roots.push(std::fs::canonicalize(&root).map_err(|error| {
            invalid_data(format!(
                "canonicalize MEMRA_REPACK_EXTERNAL_ROOTS entry {}: {error}",
                root.display()
            ))
        })?);
    }
    Ok(roots)
}

fn open_repack_shard(dir: &Path, relative: &Path) -> std::io::Result<File> {
    let path = dir.join(relative);
    let external_roots_configured = std::env::var_os("MEMRA_REPACK_EXTERNAL_ROOTS").is_some();
    let mut options = std::fs::OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    if !external_roots_configured {
        options.custom_flags(libc::O_NOFOLLOW);
    }
    let file = options.open(&path)?;
    let metadata = file.metadata()?;
    if !metadata.is_file() {
        return Err(invalid_data(format!(
            "repack shard {} is not a regular file",
            path.display()
        )));
    }
    #[cfg(unix)]
    if metadata.nlink() != 1 {
        return Err(invalid_data(format!(
            "repack shard {} has {} hard links; expected exactly one",
            path.display(),
            metadata.nlink()
        )));
    }

    #[cfg(target_os = "linux")]
    let opened_path = std::fs::canonicalize(format!("/proc/self/fd/{}", file.as_raw_fd()))?;
    #[cfg(not(target_os = "linux"))]
    let opened_path = std::fs::canonicalize(&path)?;
    let trusted = repack_trusted_roots(dir)?;
    if !trusted.iter().any(|root| opened_path.starts_with(root)) {
        return Err(invalid_data(format!(
            "repack shard {} resolves outside the artifact root and MEMRA_REPACK_EXTERNAL_ROOTS",
            path.display()
        )));
    }
    Ok(file)
}

/// How a checkpoint stores its NVFP4 per-16 FP8 scale plane.
///
/// The scale plane is the ONE part of an NVFP4 checkpoint whose on-disk order is not implied by
/// its declared shape. `tiyuvta/GLM-5.3-Flash-NVFP4-B200-hybrid` declares `weight_scale` as
/// `[out, in/16]` — exactly the shape a linear plane has — while storing the bytes in the
/// CUTLASS/vLLM `Swizzle32x4x4` tensor-core order. So the shape guard below CANNOT tell the two
/// apart, and reading swizzled bytes as linear is silent corruption rather than a load failure:
/// measured against the source BF16 (`zai-org/GLM-5.3-Flash-BF16@f12e0fe1`, layer 0
/// `mlp.gate_proj` rows 0..128, 2026-09-04) a linear read of this mint gives relative error
/// 0.570 / cosine 0.847 — wrong, but fluent — where the unswizzled read gives 0.094 / 0.9956,
/// which is ordinary e2m1 quantization error. A model that loads and speaks is not evidence.
///
/// Therefore the layout is DECLARED, never guessed, and an unrecognised declaration is a hard
/// load error rather than a fallback to `Linear`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Nvfp4ScaleLayout {
    /// Row-major `[out_f, in_f/16]`, one FP8 byte per 16 input elements. Every checkpoint that
    /// predates the B200 hybrid mint, and the default when no `LAYOUT.json` is present.
    Linear,
    /// CUTLASS / vLLM `swizzle_blockscale`: the linear plane padded to `out%128` and
    /// `(in/16)%4`, viewed as `(N/128, 4, 32, K4, 4)` and permuted to `(N/128, K4, 32, 4, 4)`.
    /// Undone once at load by `nvfp4_repack::unswizzle_blockscale`, after which every existing
    /// NVFP4 kernel reads the checkpoint unchanged.
    Swizzle32x4x4,
}

/// Read the checkpoint's declared NVFP4 scale layout from `LAYOUT.json` beside `config.json`.
///
/// Absent file or absent key => `Linear` (the historical behaviour, and correct for every
/// checkpoint minted before 2026-09-04). A PRESENT but unrecognised `nvfp4_scale` is an error:
/// a future layout must not be read as linear just because this build has not learned it yet.
fn read_nvfp4_scale_layout(dir: &std::path::Path) -> std::io::Result<Nvfp4ScaleLayout> {
    let Ok(text) = std::fs::read_to_string(dir.join("LAYOUT.json")) else {
        return Ok(Nvfp4ScaleLayout::Linear);
    };
    let Some(rest) = text.split("\"nvfp4_scale\"").nth(1) else {
        return Ok(Nvfp4ScaleLayout::Linear);
    };
    let value = rest
        .split_once(':')
        .and_then(|(_, v)| v.trim_start().strip_prefix('"'))
        .and_then(|v| v.split('"').next())
        .unwrap_or("")
        .trim();
    if value.starts_with("Swizzle32x4x4") {
        Ok(Nvfp4ScaleLayout::Swizzle32x4x4)
    } else if value.eq_ignore_ascii_case("linear") || value.is_empty() {
        Ok(Nvfp4ScaleLayout::Linear)
    } else {
        Err(invalid_data(format!(
            "LAYOUT.json declares nvfp4_scale {value:?}, which this build cannot read. \
             Reading it as a linear plane would load and generate fluent but WRONG text \
             (see Nvfp4ScaleLayout). Teach the loader this layout or re-mint linear."
        )))
    }
}

/// safetensors-backed source: an HF checkpoint (config.json + one/more .safetensors shards).
/// `find` translates the requested ggml name into the HF name, looks it up, and reverses the
/// shape into ggml `ne` order.
pub struct SafetensorsSource {
    model: StModel,
    cfg: ModelConfig,
    dir: std::path::PathBuf,
    modules_to_not_convert: Vec<String>,
    preserve_checkpoint_bf16: bool,
    quant_algo: Option<String>,
    nvfp4_scale_layout: Nvfp4ScaleLayout,
}

impl SafetensorsSource {
    /// Open an HF model directory: expects a `config.json` plus `model.safetensors`
    /// (single) or `model.safetensors.index.json` (+ shards). `dir` may also be a direct
    /// path to a single `.safetensors` file (config.json must then sit beside it).
    pub fn open(path: &std::path::Path) -> std::io::Result<Self> {
        let dir = if path.is_file() {
            path.parent().unwrap_or(std::path::Path::new("."))
        } else {
            path
        };
        let config = std::fs::read_to_string(dir.join("config.json"))?;
        let (hf, cfg) = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let hf = crate::config::HfConfig::parse(&config);
            let cfg = ModelConfig::from_hf(&hf);
            (hf, cfg)
        }))
        .map_err(|payload| {
            invalid_data(format!(
                "model config parser panicked: {}",
                payload
                    .downcast_ref::<String>()
                    .map(String::as_str)
                    .or_else(|| payload.downcast_ref::<&str>().copied())
                    .unwrap_or("unknown panic")
            ))
        })?;
        let model = StModel::open(path)?;
        let nvfp4_scale_layout = read_nvfp4_scale_layout(dir)?;
        Ok(Self {
            model,
            cfg,
            dir: dir.to_path_buf(),
            modules_to_not_convert: hf.modules_to_not_convert,
            preserve_checkpoint_bf16: hf.preserve_checkpoint_bf16,
            quant_algo: hf.quant_algo,
            nvfp4_scale_layout,
        })
    }

    /// Open with an explicitly-provided config (e.g. tests, or config.json elsewhere).
    pub fn open_with_config(path: &std::path::Path, cfg: ModelConfig) -> std::io::Result<Self> {
        let model = StModel::open(path)?;
        let dir = if path.is_file() {
            path.parent()
                .unwrap_or(std::path::Path::new("."))
                .to_path_buf()
        } else {
            path.to_path_buf()
        };
        let hf = std::fs::read_to_string(dir.join("config.json"))
            .ok()
            .map(|json| crate::config::HfConfig::parse(&json));
        let modules_to_not_convert = hf
            .as_ref()
            .map(|config| config.modules_to_not_convert.clone())
            .unwrap_or_default();
        let preserve_checkpoint_bf16 = hf
            .as_ref()
            .is_some_and(|config| config.preserve_checkpoint_bf16);
        let quant_algo = hf.as_ref().and_then(|config| config.quant_algo.clone());
        let nvfp4_scale_layout = read_nvfp4_scale_layout(&dir)?;
        Ok(Self {
            model,
            cfg,
            dir,
            modules_to_not_convert,
            preserve_checkpoint_bf16,
            quant_algo,
            nvfp4_scale_layout,
        })
    }

    pub fn arch(&self) -> &Arch {
        &self.cfg.arch
    }

    fn preserves_source_dtype(&self, hf_name: &str) -> bool {
        if self.preserve_checkpoint_bf16 {
            return true;
        }
        let matches = |candidate: &str| {
            self.modules_to_not_convert.iter().any(|module| {
                candidate == module
                    || candidate
                        .strip_prefix(module)
                        .is_some_and(|rest| rest.starts_with('.'))
            })
        };
        if matches(hf_name) {
            return true;
        }

        // Step's physical MTP namespaces omit some logical wrapper components used by the
        // quantization contract. Match both identities without weakening other model families.
        if let Some((layer, suffix)) = hf_name
            .strip_prefix("model.layers.")
            .and_then(|rest| rest.split_once('.'))
        {
            let logical = format!("model.layers.{layer}.mtp_block.{suffix}");
            if matches(&logical) {
                return true;
            }
        }
        let without_transformer = hf_name.replacen(".transformer.", ".", 1);
        matches(&without_transformer)
    }

    /// Direct HF-name access (zero-copy). Applies the prefix-fallback so a wrapper prefix like
    /// `model.language_model.` (qwen35 VLM) resolves against the plain `model.` namespace and vice
    /// versa (ST-MOE-PLAN §2.0). Returns a BORROWED view (no transform).
    /// MTP-block fused stacked experts (unsloth qwen3.6-35B-A3B ST class): the checkpoint
    /// stores `mtp.layers.{k}.mlp.experts.gate_up_proj` [E, 2*ff, in] and `...experts.down_proj`
    /// [E, out, ff] as single 3D BF16 stacks (transformers fused-MoE layout, gate rows first);
    /// the engine asks per-expert 2D `blk.{trunk+k}.ffn_{gate,up,down}_exps.{e}.weight`.
    /// Row-major means every slice is contiguous: expert block e, then gate = first ff rows,
    /// up = last ff rows. Slices retain BF16 when the checkpoint contract requires it; otherwise
    /// the generic large-matmul loader law may re-encode them to Q8_0.
    fn mtp_fused_expert_slice(&self, ggml_name: &str) -> Option<TensorView<'_>> {
        if self.cfg.nextn_predict_layers == 0 {
            return None;
        }
        let n_trunk = self.cfg.n_layer - self.cfg.nextn_predict_layers;
        let rest = ggml_name.strip_prefix("blk.")?;
        let (il, suffix) = rest.split_once('.')?;
        let il: u32 = il.parse().ok()?;
        if il < n_trunk {
            return None;
        }
        let (proj, e) = ["gate", "up", "down"].iter().find_map(|p| {
            suffix
                .strip_prefix(&format!("ffn_{p}_exps."))
                .and_then(|s| s.strip_suffix(".weight"))
                .and_then(|s| s.parse::<usize>().ok())
                .map(|e| (*p, e))
        })?;
        let fused = if proj == "down" {
            "down_proj"
        } else {
            "gate_up_proj"
        };
        let hf = format!("mtp.layers.{}.mlp.experts.{fused}", il - n_trunk);
        let (info, bytes) = self.lookup(&hf)?;
        if info.dtype != "BF16" || info.shape.len() != 3 {
            return None;
        }
        let (n_e, out, in_f) = (
            info.shape[0] as usize,
            info.shape[1] as usize,
            info.shape[2] as usize,
        );
        if e >= n_e {
            return None;
        }
        let block = out * in_f; // one expert's elements, contiguous
        let (row0, rows) = match proj {
            "gate" => (0, out / 2),
            "up" => (out / 2, out / 2),
            _ => (0, out),
        };
        let start = (e * block + row0 * in_f) * 2; // BF16 = 2 bytes
        let slice = &bytes[start..start + rows * in_f * 2];
        if self.preserves_source_dtype(&hf) {
            return Some(TensorView {
                bytes: Cow::Borrowed(slice),
                ggml_type: GgmlType::BF16,
                ne: vec![in_f as u64, rows as u64],
            });
        }
        let data: Vec<f32> = slice
            .chunks_exact(2)
            .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
            .collect();
        Some(TensorView {
            bytes: Cow::Owned(crate::nvfp4_repack::f32_to_q8_0(&data)),
            ggml_type: GgmlType::Q8_0,
            ne: vec![in_f as u64, rows as u64],
        })
    }

    /// Public f32 dequant access for offline oracles (the glm5 checkpoint-parity
    /// runner). Same path as the loader's value-transform producers: wrapper-prefix
    /// fallback (`model.` <-> `model.language_model.`), FP8 e4m3 sibling-scale block
    /// dequant, NVFP4, and plain float storage. Returns `(data, ne)` with `ne` in
    /// GGUF order (innermost dimension first).
    pub fn dequant_f32_hf(&self, hf_name: &str) -> Option<(Vec<f32>, Vec<u64>)> {
        self.deq_f32(hf_name)
    }

    pub fn raw_hf(&self, hf_name: &str) -> Option<TensorView<'_>> {
        let (info, bytes) = self.lookup(hf_name)?;
        let ggml_type = match info.ggml_type() {
            Ok(ggml_type) => ggml_type,
            Err(error) => {
                eprintln!("[safetensors] refusing generic tensor {hf_name:?}: {error}");
                return None;
            }
        };
        Some(TensorView {
            bytes: Cow::Borrowed(bytes),
            ggml_type,
            ne: info.ne(),
        })
    }

    /// Resolve one physical HF tensor spelling, trying it verbatim then with the qwen35
    /// multimodal wrapper prefix inserted/removed (`model.` <-> `model.language_model.`). The
    /// dense map and the SSM map share one `model.layers.{il}.` namespace this way
    /// (ST-MOE-PLAN §2.0).
    fn lookup_physical(&self, hf_name: &str) -> Option<(&crate::safetensors::StInfo, &[u8])> {
        if let Some(r) = self.model.raw(hf_name) {
            return Some(r);
        }
        // model.layers.* -> model.language_model.layers.*  (and the symmetric strip)
        if let Some(rest) = hf_name.strip_prefix("model.")
            && !rest.starts_with("language_model.")
            && !rest.starts_with("visual.")
        {
            let alt = format!("model.language_model.{rest}");
            if let Some(r) = self.model.raw(&alt) {
                return Some(r);
            }
        }
        if let Some(rest) = hf_name.strip_prefix("model.language_model.") {
            let alt = format!("model.{rest}");
            if let Some(r) = self.model.raw(&alt) {
                return Some(r);
            }
        }
        // MiniMax-M3-VL nests the OTHER way round: `language_model.model.layers.*` /
        // `language_model.lm_head.weight` (whole text model under a `language_model.` root).
        if hf_name.starts_with("model.") || hf_name == "lm_head.weight" {
            let alt = format!("language_model.{hf_name}");
            if let Some(r) = self.model.raw(&alt) {
                return Some(r);
            }
        }
        if let Some(rest) = hf_name.strip_prefix("language_model.")
            && let Some(r) = self.model.raw(rest)
        {
            return Some(r);
        }
        None
    }

    /// Resolve an HF tensor name while preserving Tencent's canonical Hy3 semantic map. NVIDIA
    /// ModelOpt exports the same router, correction bias, and shared MLP under flattened names;
    /// accept those physical aliases only for Hy3 and only after the canonical spelling misses.
    fn lookup(&self, hf_name: &str) -> Option<(&crate::safetensors::StInfo, &[u8])> {
        self.lookup_physical(hf_name).or_else(|| {
            self.cfg
                .arch
                .is_hy3()
                .then(|| hy3_modelopt_aliases(hf_name))?
                .into_iter()
                .find_map(|alias| self.lookup_physical(&alias))
        })
    }

    /// Ownership twin of `lookup`: return the exact same wrapper-prefix fallback as an owned
    /// whole-file mmap extent so a resident expert bank can outlive this source without copying.
    fn lookup_extent_physical(
        &self,
        hf_name: &str,
    ) -> Option<(Arc<Mmap>, Arc<File>, usize, usize)> {
        if let Some(extent) = self.model.raw_extent(hf_name) {
            return Some(extent);
        }
        if let Some(rest) = hf_name.strip_prefix("model.")
            && !rest.starts_with("language_model.")
            && !rest.starts_with("visual.")
        {
            let alt = format!("model.language_model.{rest}");
            if let Some(extent) = self.model.raw_extent(&alt) {
                return Some(extent);
            }
        }
        if let Some(rest) = hf_name.strip_prefix("model.language_model.") {
            let alt = format!("model.{rest}");
            if let Some(extent) = self.model.raw_extent(&alt) {
                return Some(extent);
            }
        }
        if hf_name.starts_with("model.") || hf_name == "lm_head.weight" {
            let alt = format!("language_model.{hf_name}");
            if let Some(extent) = self.model.raw_extent(&alt) {
                return Some(extent);
            }
        }
        if let Some(rest) = hf_name.strip_prefix("language_model.")
            && let Some(extent) = self.model.raw_extent(rest)
        {
            return Some(extent);
        }
        None
    }

    fn lookup_extent(&self, hf_name: &str) -> Option<(Arc<Mmap>, Arc<File>, usize, usize)> {
        self.lookup_extent_physical(hf_name).or_else(|| {
            self.cfg
                .arch
                .is_hy3()
                .then(|| hy3_modelopt_aliases(hf_name))?
                .into_iter()
                .find_map(|alias| self.lookup_extent_physical(&alias))
        })
    }

    /// FP8 weight-scale sibling lookup: `<stem>.weight_scale` (modelopt / compressed-tensors)
    /// OR `<stem>.weight_scale_inv` (Qwen official FP8 / DeepSeek-V3 lineage). Despite the
    /// `_inv` suffix, `weight_scale_inv` is the DEQUANT MULTIPLIER (dequant = code * scale —
    /// the "inv" names the inverse of the QUANT divide; DeepSeek-V3 reference kernel and vLLM
    /// both multiply by it), so both spellings feed the same downstream math.
    fn f8_scale_sibling(&self, stem: &str) -> Option<(&crate::safetensors::StInfo, &[u8])> {
        self.lookup(&format!("{stem}.weight_scale"))
            .or_else(|| self.lookup(&format!("{stem}.weight_scale_inv")))
    }

    /// Dequantize an HF tensor to f32 (used by the value-transform producers). Handles BOTH plain
    /// F32/F16/BF16 tensors AND modelopt (compressed-tensors) NVFP4 weights: a `<name>.weight` stored
    /// `U8` with a sibling `<name>.weight_scale` is dequantized through the NVFP4 path (per-16 UE4M3
    /// block scale × the per-tensor `weight_scale_2`), so the hybrid SSM V-reorder transforms (which
    /// operate on f32) work on an NVFP4 checkpoint exactly as on a BF16 one.
    fn deq_f32(&self, hf_name: &str) -> Option<(Vec<f32>, Vec<u64>)> {
        // NVFP4 weight (modelopt OR Reza)? Dequant through the NVFP4 path so the hybrid SSM V-reorder
        // transforms (which operate on f32) work on an NVFP4 checkpoint exactly as on a BF16 one.
        if hf_name.ends_with(".weight")
            && let Some((out_f, in_f, wbytes, wscale, macro_s)) = self.nvfp4_quant(hf_name)
        {
            use crate::nvfp4_repack::dequant_modelopt_row;
            let in_bytes = in_f / 2;
            let scl_bytes = in_f / 16;
            let mut data = vec![0f32; out_f * in_f];
            for o in 0..out_f {
                let row = dequant_modelopt_row(
                    &wbytes[o * in_bytes..(o + 1) * in_bytes],
                    &wscale[o * scl_bytes..(o + 1) * scl_bytes],
                    in_f,
                );
                for (e, v) in row.iter().enumerate() {
                    data[o * in_f + e] = v * macro_s; // fold the per-tensor macro-scale into f32
                }
            }
            return Some((data, vec![in_f as u64, out_f as u64]));
        }
        // FP8 E4M3 weight + scale sibling (per-tensor F32 scalar = NVIDIA 27B linear_attn
        // class; per-channel [out,1] F32/BF16 = unsloth compressed-tensors mixed-precision
        // class; block-128 2-D grid = Qwen official FP8 / DeepSeek-V3 lineage): dequant to
        // f32 here so the V-reorder transforms consume it like a BF16 tensor.
        if hf_name.ends_with(".weight")
            && let Some((info, bytes)) = self.lookup(hf_name)
            && info.dtype == "F8_E4M3"
            && info.shape.len() == 2
        {
            let stem = hf_name.strip_suffix(".weight").unwrap_or(hf_name);
            let out_f = info.shape[0] as usize;
            let in_f = info.shape[1] as usize;
            if let Some((sinfo, sbytes)) = self.f8_scale_sibling(stem)
                && let Some(scales) = f8_scales(sinfo, sbytes, out_f, in_f)
            {
                return Some((f8_deq_f32(bytes, out_f, in_f, &scales), info.ne()));
            }
        }
        let (info, bytes) = self.lookup(hf_name)?;
        let ne = info.ne();
        let n: u64 = ne.iter().product();
        Some((
            crate::dequant::dequantize(info.ggml_type().ok()?, bytes, n as usize),
            ne,
        ))
    }

    /// Validate an on-disk NVFP4 scale plane against the declared layout and return it in the
    /// LINEAR `[out_f, in_f/16]` form every memra NVFP4 reader consumes. Borrowed for a linear
    /// checkpoint (zero copy, the historical path); owned for a swizzled one, where the
    /// permutation is undone once here instead of in every kernel's inner loop.
    fn nvfp4_scale_linear<'a>(
        &self,
        shape: &[u64],
        sbytes: &'a [u8],
        out_f: usize,
        in_f: usize,
    ) -> Option<Cow<'a, [u8]>> {
        let k = in_f / 16;
        match self.nvfp4_scale_layout {
            Nvfp4ScaleLayout::Linear => {
                if shape != [out_f as u64, k as u64] || sbytes.len() != out_f.checked_mul(k)? {
                    return None;
                }
                Some(Cow::Borrowed(sbytes))
            }
            Nvfp4ScaleLayout::Swizzle32x4x4 => {
                // The swizzled plane is padded to out%128 / k%4. A mint whose shapes are already
                // multiples (every tensor in the B200 hybrid) declares the UNPADDED shape while
                // storing swizzled bytes, so accept either spelling and key off the byte count.
                let s_n = out_f.div_ceil(128) * 128;
                let s_k = k.div_ceil(4) * 4;
                let padded = s_n.checked_mul(s_k)?;
                if sbytes.len() != padded
                    || (shape != [out_f as u64, k as u64] && shape != [s_n as u64, s_k as u64])
                {
                    return None;
                }
                Some(Cow::Owned(crate::nvfp4_repack::unswizzle_blockscale(
                    sbytes, out_f, in_f,
                )))
            }
        }
    }

    /// Detect an HF NVFP4 quantized Linear under ANY on-disk encoding and return everything the
    /// repack needs: `(out_f, in_f, packed_bytes, per16_fp8_scale_bytes, macro_scale)`. All encodings
    /// store the SAME e2m1 weights + per-16 FP8(e4m3) scales — only names + macro-scale differ:
    ///   * modelopt: `<name>.weight`(U8 packed) + `<name>.weight_scale`(F8_E4M3) +
    ///     `<name>.weight_scale_2`(required scalar F32 per-tensor macro).
    ///   * compressed-tensors (llm-compressor): `<name>.weight_packed`(U8) +
    ///     `<name>.weight_scale`(F8_E4M3 per-16) + `<name>.weight_global_scale`(F32 per-tensor macro).
    ///     The plain `<name>.weight` coexists as a BF16 tensor (unused by us when packed is present).
    ///   * Reza "custom_nvfp4_e2m1_e4m3_scales": `<name>.weight.nvfp4_packed`(U8) +
    ///     `<name>.weight.nvfp4_scale_e4m3`(U8/FP8 bytes), NO macro-scale (=> 1.0).
    ///     `out_f`/`in_f` are the logical [out, in] dims (packed weight is [out, in/2] U8). `None` for a
    ///     plain (non-quantized) weight or missing siblings.
    #[allow(clippy::type_complexity)] // allow: one-shot composite type; naming it would hide the shape that matters at the call site
    fn nvfp4_quant<'a>(
        &'a self,
        hf_weight: &str,
    ) -> Option<(usize, usize, &'a [u8], Cow<'a, [u8]>, f32)> {
        // modelopt: the `.weight` itself is the U8 packed tensor with a `.weight_scale` sibling.
        if let Some((winfo, wbytes)) = self.lookup(hf_weight)
            && winfo.dtype == "U8"
            && winfo.shape.len() == 2
        {
            let stem = hf_weight.strip_suffix(".weight")?;
            if let Some((sinfo, sbytes)) = self.lookup(&format!("{stem}.weight_scale"))
                && sinfo.dtype == "F8_E4M3"
            {
                let out_f = winfo.shape[0] as usize; // HF row-major [out, in/2]
                let in_f = (winfo.shape[1] as usize) * 2; // U8 packs 2 codes/byte
                if !in_f.is_multiple_of(16) || wbytes.len() != out_f.checked_mul(in_f / 2)? {
                    return None;
                }
                let scale = self.nvfp4_scale_linear(&sinfo.shape, sbytes, out_f, in_f)?;
                let (macro_info, macro_bytes) = self.lookup(&format!("{stem}.weight_scale_2"))?;
                if macro_info.dtype != "F32"
                    || !(macro_info.shape.is_empty() || macro_info.shape == [1])
                    || macro_bytes.len() != 4
                {
                    return None;
                }
                let macro_s = f32::from_le_bytes(macro_bytes.try_into().ok()?);
                if !macro_s.is_finite() || macro_s <= 0.0 {
                    return None;
                }
                return Some((out_f, in_f, wbytes, scale, macro_s));
            }
        }
        // compressed-tensors (llm-compressor): `<name>.weight_packed` (U8) + `<name>.weight_scale`
        // (F8_E4M3 per-16) + `<name>.weight_global_scale` (F32 per-tensor). The plain
        // `<name>.weight` (BF16) coexists but is UNUSED when the packed sibling is present —
        // the packed representation IS the quantized model output. CRITICAL SEMANTICS DIFFERENCE:
        // compressed-tensors' `weight_global_scale` is a DIVISOR (dequant = code * micro / global),
        // whereas modelopt's `weight_scale_2` is a MULTIPLIER (dequant = code * micro * scale_2).
        // The packed bytes + micro scales are byte-identical between the two formats (verified on
        // the AxionML vs apolo13x pair), only the macro-scale semantics differ. We invert here so
        // the engine's post-matmul multiply stays unchanged.
        //   compressed-tensors: elem = e2m1_code * ue4m3_scale_per16 / weight_global_scale
        //   modelopt:           elem = e2m1_code * ue4m3_scale_per16 * weight_scale_2
        //   => macro_s = 1.0 / weight_global_scale
        let stem = hf_weight.strip_suffix(".weight")?;
        if let Some((winfo, wbytes)) = self.lookup(&format!("{stem}.weight_packed"))
            && winfo.dtype == "U8"
            && winfo.shape.len() == 2
            && let Some((sinfo, sbytes)) = self.lookup(&format!("{stem}.weight_scale"))
            && sinfo.dtype == "F8_E4M3"
        {
            let out_f = winfo.shape[0] as usize;
            let in_f = (winfo.shape[1] as usize) * 2;
            if !in_f.is_multiple_of(16) || wbytes.len() != out_f.checked_mul(in_f / 2)? {
                return None;
            }
            let scale = self.nvfp4_scale_linear(&sinfo.shape, sbytes, out_f, in_f)?;
            let (macro_info, macro_bytes) = self.lookup(&format!("{stem}.weight_global_scale"))?;
            if macro_info.dtype != "F32"
                || !(macro_info.shape.is_empty() || macro_info.shape == [1])
                || macro_bytes.len() != 4
            {
                return None;
            }
            let global = f32::from_le_bytes(macro_bytes.try_into().ok()?);
            if !global.is_finite() || global <= 0.0 {
                return None;
            }
            let macro_s = 1.0 / global;
            return Some((out_f, in_f, wbytes, scale, macro_s));
        }
        // Reza custom: `<name>.weight.nvfp4_packed` (U8) + `<name>.weight.nvfp4_scale_e4m3`. No macro.
        let (winfo, wbytes) = self.lookup(&format!("{hf_weight}.nvfp4_packed"))?;
        if winfo.dtype != "U8" || winfo.shape.len() != 2 {
            return None;
        }
        let (sinfo, sbytes) = self.lookup(&format!("{hf_weight}.nvfp4_scale_e4m3"))?;
        let out_f = winfo.shape[0] as usize;
        let in_f = (winfo.shape[1] as usize) * 2;
        if !in_f.is_multiple_of(16) {
            return None;
        }
        let scale = self.nvfp4_scale_linear(&sinfo.shape, sbytes, out_f, in_f)?;
        Some((out_f, in_f, wbytes, scale, 1.0))
    }
}

/// FP8 weight-scale granularity, decoded to f32. Every observed encoding:
///  * `PerTensor`: modelopt F32 scalar (NVIDIA 27B linear_attn class).
///  * `PerRow`: compressed-tensors per-channel `[out, 1]` in F32 or BF16 (unsloth
///    mixed-precision FP8 class).
///  * `Block128`: fine-grained 2-D grid `[ceil(out/128), ceil(in/128)]` row-major
///    (Qwen official FP8, DeepSeek-V3 lineage; Qwen3.6-27B-FP8 ships it as BF16 under
///    the `weight_scale_inv` name — e.g. gate_proj [17408,5120] -> scales [136,40],
///    verified from the HF shard header 2026-08-03). `scales[(o/128)*cols + (e/128)]`
///    applies to element W[o][e].
enum F8Scales {
    PerTensor(f32),
    PerRow(Vec<f32>),
    Block128 { scales: Vec<f32>, cols: usize },
}

impl F8Scales {
    /// Dequant multiplier for element W[o][e].
    #[inline]
    fn at(&self, o: usize, e: usize) -> f32 {
        match self {
            F8Scales::PerTensor(s) => *s,
            F8Scales::PerRow(v) => v[o],
            F8Scales::Block128 { scales, cols } => scales[(o >> 7) * cols + (e >> 7)],
        }
    }
}

/// Dequantize a full F8-E4M3 `[out_f, in_f]` weight to f32 with any scale granularity.
/// Block-128 note: within one row the scale changes only every 128 elements, so the inner
/// loop hoists the multiplier per 128-chunk — same result as `at()` per element.
fn f8_deq_f32(bytes: &[u8], out_f: usize, in_f: usize, scales: &F8Scales) -> Vec<f32> {
    let mut data = vec![0f32; out_f * in_f];
    for o in 0..out_f {
        let row = &bytes[o * in_f..(o + 1) * in_f];
        let drow = &mut data[o * in_f..(o + 1) * in_f];
        let mut e = 0;
        while e < in_f {
            let end = (e + 128 - (e & 127)).min(in_f); // next 128 boundary (or row end)
            let s = scales.at(o, e);
            for i in e..end {
                drow[i] = crate::nvfp4_repack::fp8_e4m3_to_f32(row[i]) * s;
            }
            e = end;
        }
    }
    data
}

/// `MEMRA_FP8_FOLD=1` (default OFF; ARM A, lane fp8-gemm-arm): fold a block-128 scale grid
/// into ONE per-tensor scale at load (global-amax re-encode) so the per-tensor e4m3
/// consumers (QT_F8_E4M3 resident arm + try_fp8_gemm) take block-128 checkpoints unchanged.
fn fp8_fold_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("MEMRA_FP8_FOLD")
            .map(|v| v == "1")
            .unwrap_or(false)
    })
}

/// Parse an FP8 `weight_scale` / `weight_scale_inv` sibling into its granularity.
/// F32 and BF16 payloads accepted for every granularity. None = unrecognized encoding
/// (element count matches neither 1, out_f, nor the ceil-128 block grid) — the caller
/// falls through and the raw-path assert names the tensor loudly.
fn f8_scales(
    sinfo: &crate::safetensors::StInfo,
    sbytes: &[u8],
    out_f: usize,
    in_f: usize,
) -> Option<F8Scales> {
    let n = sinfo.shape.iter().product::<u64>() as usize;
    let vals: Vec<f32> = match sinfo.dtype.as_str() {
        "F32" if sbytes.len() >= n * 4 => sbytes[..n * 4]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect(),
        "BF16" if sbytes.len() >= n * 2 => sbytes[..n * 2]
            .chunks_exact(2)
            .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
            .collect(),
        _ => return None,
    };
    if !vals.iter().all(|s| s.is_finite() && *s > 0.0) {
        return None;
    }
    if n == 1 {
        return Some(F8Scales::PerTensor(vals[0]));
    }
    if n == out_f {
        return Some(F8Scales::PerRow(vals));
    }
    // Block-128 grid: shape must be exactly [ceil(out/128), ceil(in/128)] (ceil handles
    // dims that are not multiples of 128; every Qwen3.6-27B dim happens to divide evenly).
    let (rows, cols) = (out_f.div_ceil(128), in_f.div_ceil(128));
    if sinfo.shape.len() == 2 && sinfo.shape[0] as usize == rows && sinfo.shape[1] as usize == cols
    {
        return Some(F8Scales::Block128 { scales: vals, cols });
    }
    None
}

fn hy3_modelopt_aliases(hf_name: &str) -> Vec<String> {
    let mut aliases = Vec::with_capacity(2);
    if hf_name.contains(".mlp.router.gate.") {
        aliases.push(hf_name.replace(".mlp.router.gate.", ".mlp.gate."));
    }
    if hf_name.ends_with(".mlp.expert_bias") {
        aliases.push(hf_name.replace(".mlp.expert_bias", ".mlp.e_score_correction_bias"));
        aliases.push(hf_name.replace(".mlp.expert_bias", ".mlp.router.expert_bias"));
    }
    if hf_name.contains(".mlp.shared_mlp.") {
        aliases.push(hf_name.replace(".mlp.shared_mlp.", ".mlp.shared_experts."));
    }
    aliases
}

impl TensorSource for SafetensorsSource {
    fn config(&self) -> ModelConfig {
        self.cfg.clone()
    }
    fn tensor_census(&self) -> Result<TensorCensus, String> {
        census_from_safetensors_model(&self.model)
    }
    fn expert_activation_precision(&self) -> ExpertActivationPrecision {
        expert_activation_precision_from_quant_algo(self.quant_algo.as_deref())
    }
    fn st_dir(&self) -> Option<&std::path::Path> {
        Some(&self.dir)
    }
    fn nvfp4_cache_tag(&self) -> &'static str {
        match self.nvfp4_scale_layout {
            Nvfp4ScaleLayout::Linear => "",
            Nvfp4ScaleLayout::Swizzle32x4x4 => "-swz32x4x4",
        }
    }
    /// Presence check without the repack: `find` on a plain NVFP4 weight materializes the whole
    /// repacked buffer just to answer `has` (then `load_opt_from_source` repacks AGAIN to load).
    /// The native lookup is header-only, so answer from it first.
    fn has(&self, ggml_name: &str) -> bool {
        self.find_nvfp4_native(ggml_name).is_some()
            || self.find_fp8_stacked_native(ggml_name).is_some()
            || self.find_nvfp4_stacked_native(ggml_name).is_some()
            || self.find(ggml_name).is_some()
    }
    /// A1 direct import: plain (untransformed) modelopt/Reza NVFP4 weights expose their raw file
    /// bytes so the engine repacks modelopt -> split-plane in ONE pass. Transform targets (the
    /// hybrid V-reorders) return None and keep the GGUF-block hop (`kind.apply_nvfp4`).
    fn find_nvfp4_native(&self, ggml_name: &str) -> Option<Nvfp4Native<'_>> {
        use crate::hf_mapping::{HfTarget, resolve_ggml};
        let hf = match resolve_ggml(ggml_name, &self.cfg)? {
            HfTarget::Plain(hf) => hf,
            HfTarget::Transform { .. } => return None,
        };
        let (out_f, in_f, wbytes, wscale, _macro) = self.nvfp4_quant(&hf)?;
        Some(Nvfp4Native {
            wbytes,
            wscale,
            out_f,
            in_f,
        })
    }
    /// FP8-E4M3-native access (MEMRA_PP_FP8 prefill operand / MEMRA_ST_E4M3 resident copy).
    /// Two arms, mirroring the Q8_0 re-encode arms in `find`:
    ///  * Plain: borrow the checkpoint's e4m3 bytes verbatim (zero copy, EXACT). Per-tensor and
    ///    block-128 scale classes both qualify — block-128 rides `blk` (grid decoded to f32,
    ///    on-disk order, F8BlockGrid layout contract).
    ///  * Transform (hybrid V-reorders): run the SAME `deq_f32` + `kind.apply` the Q8_0 arm runs,
    ///    then re-encode `value/scale` to nearest e4m3 — exact for a pure permutation (every value
    ///    is `code*scale`; the e4m3 grid spacing dwarfs the f32 divide rounding). PER-TENSOR ONLY:
    ///    a V-reorder permutes ROWS across 128-row scale-block boundaries, so the on-disk block
    ///    grid no longer maps to the permuted rows; re-deriving a permuted grid is P1-consumer
    ///    work, not loader work. Block-128 transform targets fall back to the Q8_0 arm (which
    ///    dequants with the pre-permutation grid, correctly).
    ///    Per-row (unsloth per-channel) stays excluded from BOTH arms as before: no kernel consumes
    ///    a per-row e4m3 operand.
    ///    Dim gates: 2D, in_f/out_f % 16 == 0 (cuBLASLt FP8 TN alignment), and the Transform arm
    ///    keeps the >=1M-element gate of its Q8_0 twin (small tensors stay F32 there).
    fn find_fp8_native(&self, ggml_name: &str) -> Option<Fp8Native<'_>> {
        use crate::hf_mapping::{HfTarget, resolve_ggml};
        let (hf, kind) = match resolve_ggml(ggml_name, &self.cfg)? {
            HfTarget::Plain(hf) => (hf, None),
            HfTarget::Transform { hf, kind } => (hf, Some(kind)),
        };
        let (info, bytes) = self.lookup(&hf)?;
        if info.dtype != "F8_E4M3" || info.shape.len() != 2 {
            return None;
        }
        let stem = hf.strip_suffix(".weight").unwrap_or(&hf);
        let (sinfo, sbytes) = self.f8_scale_sibling(stem)?;
        let (out_hf, in_hf) = (info.shape[0] as usize, info.shape[1] as usize);
        let (scale, blk) = match f8_scales(sinfo, sbytes, out_hf, in_hf)? {
            F8Scales::PerTensor(s) => (s, None),
            // ARM A scale-fold (MEMRA_FP8_FOLD=1, lane fp8-gemm-arm 2026-08-03): collapse the
            // 128x128 grid to ONE per-tensor scale at load — dequant each block by its own scale,
            // take the global amax, re-encode nearest-e4m3 with s = amax/448 (e4m3 max normal).
            // The result is a plain per-tensor operand: the existing QT_F8_E4M3 resident arm and
            // try_fp8_gemm consume it UNCHANGED (their blk-reject gates never see a grid). This is
            // LOSSY where the grid's dynamic range varies across blocks (that's the arm's whole
            // question — argmax + logit-maxdiff vs the Q8_0 floor arbitrate). Plain targets only:
            // a Transform (V-reorder) still falls to the Q8_0 arm below.
            F8Scales::Block128 { scales, cols } if kind.is_none() && fp8_fold_enabled() => {
                if in_hf % 16 != 0 || out_hf % 16 != 0 {
                    return None;
                }
                let grid = F8Scales::Block128 { scales, cols };
                let data = f8_deq_f32(bytes, out_hf, in_hf, &grid);
                let amax = data.iter().fold(0f32, |a, &v| a.max(v.abs()));
                let s = if amax > 0.0 && amax.is_finite() {
                    amax / 448.0
                } else {
                    1.0
                };
                let enc: Vec<u8> = data
                    .iter()
                    .map(|&v| crate::nvfp4_repack::f32_to_fp8_e4m3(v / s))
                    .collect();
                return Some(Fp8Native {
                    bytes: Cow::Owned(enc),
                    scale: s,
                    blk: None,
                    out_f: out_hf,
                    in_f: in_hf,
                });
            }
            F8Scales::Block128 { scales, cols } => {
                let rows = scales.len() / cols;
                (1.0, Some(F8BlockGrid { scales, rows, cols }))
            }
            // Per-row: no e4m3 kernel consumes a per-channel scale vector; the Q8_0
            // re-encode arm in `find` (which folds it host-side) stays that class's path.
            F8Scales::PerRow(_) => return None,
        };
        match kind {
            None => {
                let ne = info.ne();
                let (in_f, out_f) = (ne[0] as usize, ne[1] as usize);
                if in_f % 16 != 0 || out_f % 16 != 0 {
                    return None;
                }
                Some(Fp8Native {
                    bytes: Cow::Borrowed(bytes),
                    scale,
                    blk,
                    out_f,
                    in_f,
                })
            }
            Some(kind) => {
                // BLOCK-128 + a V-head permutation (lane/fp8-blk128-decode): permute the e4m3
                // codes and the scale grid TOGETHER, no dequant, EXACT — see
                // `TransformKind::apply_fp8_blk` for the alignment proof and for why refusing this
                // case left 144 of the 27B's 208 FP8 projections on the Q8_0 slab. A non-aligned
                // permutation still returns None here and falls to the Q8_0 arm below, which
                // dequants with the pre-permutation grid and is correct.
                if let Some(F8BlockGrid {
                    scales,
                    rows: _,
                    cols,
                }) = &blk
                {
                    let (out_hf, in_hf) = (info.shape[0] as usize, info.shape[1] as usize);
                    if let Some((ne, codes, ns, r2, c2)) =
                        kind.apply_fp8_blk(bytes, out_hf, in_hf, scales, *cols, &self.cfg)
                    {
                        let (in_f, out_f) = (ne[0] as usize, ne[1] as usize);
                        if in_f % 16 == 0 && out_f % 16 == 0 {
                            return Some(Fp8Native {
                                bytes: Cow::Owned(codes),
                                scale,
                                blk: Some(F8BlockGrid {
                                    scales: ns,
                                    rows: r2,
                                    cols: c2,
                                }),
                                out_f,
                                in_f,
                            });
                        }
                    }
                    return None; // not grid-aligned: the Q8_0 arm in `find` serves this tensor
                }
                let (mut data, ne_in) = self.deq_f32(&hf)?;
                let (ne, fbytes) = kind.apply(&mut data, ne_in, &self.cfg);
                if ne.len() != 2 || ne.iter().product::<u64>() < 1_000_000 {
                    return None;
                }
                let (in_f, out_f) = (ne[0] as usize, ne[1] as usize);
                if in_f % 16 != 0 || out_f % 16 != 0 {
                    return None;
                }
                let enc: Vec<u8> = fbytes
                    .chunks_exact(4)
                    .map(|c| {
                        crate::nvfp4_repack::f32_to_fp8_e4m3(
                            f32::from_le_bytes(c.try_into().unwrap()) / scale,
                        )
                    })
                    .collect();
                Some(Fp8Native {
                    bytes: Cow::Owned(enc),
                    scale,
                    blk: None,
                    out_f,
                    in_f,
                })
            }
        }
    }

    fn find_fp8_stacked_native(&self, ggml_name: &str) -> Option<Fp8StackedNative<'_>> {
        use crate::hf_mapping::{HfTarget, resolve_ggml};
        let hf = match resolve_ggml(ggml_name, &self.cfg)? {
            HfTarget::Plain(hf) => hf,
            HfTarget::Transform { .. } => return None,
        };
        let (info, bytes) = self.lookup(&hf)?;
        if info.dtype != "F8_E4M3" || info.shape.len() != 3 {
            return None;
        }
        let (n_expert, out_f, in_f) = (
            usize::try_from(info.shape[0]).ok()?,
            usize::try_from(info.shape[1]).ok()?,
            usize::try_from(info.shape[2]).ok()?,
        );
        if bytes.len() != n_expert.checked_mul(out_f)?.checked_mul(in_f)? {
            return None;
        }
        let stem = hf.strip_suffix(".weight").unwrap_or(&hf);
        let (scale_info, scale_bytes) = self.f8_scale_sibling(stem)?;
        let (scale_rows, scale_cols) = (out_f.div_ceil(128), in_f.div_ceil(128));
        if scale_info.shape != [n_expert as u64, scale_rows as u64, scale_cols as u64] {
            return None;
        }
        let n_scale = n_expert.checked_mul(scale_rows)?.checked_mul(scale_cols)?;
        let scales: Vec<f32> = match scale_info.dtype.as_str() {
            "F32" if scale_bytes.len() == n_scale * 4 => scale_bytes
                .chunks_exact(4)
                .map(|chunk| Some(f32::from_le_bytes(chunk.try_into().ok()?)))
                .collect::<Option<Vec<_>>>()?,
            "BF16" if scale_bytes.len() == n_scale * 2 => scale_bytes
                .chunks_exact(2)
                .map(|chunk| {
                    let bits = u16::from_le_bytes(chunk.try_into().ok()?);
                    Some(f32::from_bits(u32::from(bits) << 16))
                })
                .collect::<Option<Vec<_>>>()?,
            _ => return None,
        };
        if !scales.iter().all(|scale| scale.is_finite() && *scale > 0.0) {
            return None;
        }
        Some(Fp8StackedNative {
            bytes,
            scales,
            n_expert,
            out_f,
            in_f,
            scale_rows,
            scale_cols,
        })
    }

    fn find_nvfp4_stacked_native(&self, ggml_name: &str) -> Option<Nvfp4StackedNative<'_>> {
        use crate::hf_mapping::{HfTarget, resolve_ggml};
        let hf = match resolve_ggml(ggml_name, &self.cfg)? {
            HfTarget::Plain(hf) => hf,
            HfTarget::Transform { .. } => return None,
        };
        let (info, bytes) = self.lookup(&hf)?;
        if info.dtype != "U8" || info.shape.len() != 3 {
            return None;
        }
        let (n_expert, out_f, in_half) = (
            usize::try_from(info.shape[0]).ok()?,
            usize::try_from(info.shape[1]).ok()?,
            usize::try_from(info.shape[2]).ok()?,
        );
        let in_f = in_half.checked_mul(2)?;
        if in_f % 16 != 0 || bytes.len() != n_expert.checked_mul(out_f)?.checked_mul(in_half)? {
            return None;
        }
        let stem = hf.strip_suffix(".weight").unwrap_or(&hf);
        let (scale_info, scale_bytes) = self.lookup(&format!("{stem}.weight_scale"))?;
        let scale_cols = in_f / 16;
        if scale_info.dtype != "F8_E4M3"
            || scale_info.shape != [n_expert as u64, out_f as u64, scale_cols as u64]
            || scale_bytes.len() != n_expert.checked_mul(out_f)?.checked_mul(scale_cols)?
        {
            return None;
        }
        let macros: Vec<f32> = match self.lookup(&format!("{stem}.weight_scale_2")) {
            Some((macro_info, macro_bytes))
                if macro_info.dtype == "F32" && macro_bytes.len() == n_expert * 4 =>
            {
                macro_bytes
                    .chunks_exact(4)
                    .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
                    .collect()
            }
            // modelopt defaults an absent macro to 1.0, mirroring `nvfp4_quant`.
            None => vec![1.0; n_expert],
            _ => return None,
        };
        if !macros.iter().all(|value| value.is_finite() && *value > 0.0) {
            return None;
        }
        Some(Nvfp4StackedNative {
            codes: bytes,
            scales: scale_bytes,
            macros,
            n_expert,
            out_f,
            in_f,
        })
    }

    fn find_expert_disk(&self, ggml_name: &str) -> Option<DiskExtent> {
        use crate::hf_mapping::{HfTarget, resolve_ggml};
        let hf = match resolve_ggml(ggml_name, &self.cfg)? {
            HfTarget::Plain(hf) => hf,
            HfTarget::Transform { .. } => return None,
        };
        let (info, bytes) = self.lookup(&hf)?;
        if info.dtype != "F8_E4M3" || info.shape.len() != 3 {
            return None;
        }
        let (map, file, offset, len) = self.lookup_extent(&hf)?;
        if len != bytes.len() {
            return None;
        }
        Some(DiskExtent {
            map,
            file,
            offset: offset as u64,
            len,
        })
    }
    fn find(&self, ggml_name: &str) -> Option<TensorView<'_>> {
        use crate::hf_mapping::{HfTarget, resolve_ggml};
        // Router weights feed a discontinuous top-k decision. Preserve their checkpoint dtype
        // even outside Step's stricter whole-checkpoint BF16 contract.
        let is_float_router = ggml_name.ends_with(".ffn_gate_inp.weight");
        // MTP-block fused stacked experts (qwen3.6-35B-A3B ST class): mtp.layers.{k}.mlp.
        // experts.{gate_up,down}_proj are 3D BF16 stacks; the engine asks per-expert 2D names.
        if let Some(tv) = self.mtp_fused_expert_slice(ggml_name) {
            return Some(tv);
        }
        // NVFP4 per-tensor macro-scale sibling: the engine asks for `<stem>.scale` (model.rs) and
        // expects an F32 scalar. Map `<stem>.scale` -> modelopt `<hf>.weight_scale_2` OR
        // compressed-tensors `<hf>.weight_global_scale`. Returns None for non-quantized weights
        // (then the engine defaults the macro-scale to 1.0). Reza has no macro-scale at all.
        if let Some(stem) = ggml_name.strip_suffix(".scale") {
            let hf_weight = match resolve_ggml(&format!("{stem}.weight"), &self.cfg)? {
                HfTarget::Plain(hf) | HfTarget::Transform { hf, .. } => hf,
            };
            let hf_stem = hf_weight.strip_suffix(".weight")?;
            // Try modelopt `weight_scale_2` first (direct multiplier, borrow zero-copy).
            if let Some((info, bytes)) = self.lookup(&format!("{hf_stem}.weight_scale_2")) {
                return Some(TensorView {
                    bytes: Cow::Borrowed(bytes),
                    ggml_type: info.ggml_type().ok()?,
                    ne: vec![1],
                });
            }
            // compressed-tensors `weight_global_scale`: DIVISOR semantics, must INVERT to match
            // the engine's multiplier convention (engine does: result *= macro_scale).
            if let Some((info, bytes)) = self.lookup(&format!("{hf_stem}.weight_global_scale")) {
                if info.dtype != "F32"
                    || !(info.shape.is_empty() || info.shape == [1])
                    || bytes.len() != 4
                {
                    return None;
                }
                let global = f32::from_le_bytes(bytes.try_into().ok()?);
                if !global.is_finite() || global <= 0.0 {
                    return None;
                }
                return Some(TensorView {
                    bytes: Cow::Owned((1.0 / global).to_le_bytes().to_vec()),
                    ggml_type: GgmlType::F32,
                    ne: vec![1],
                });
            }
            return None;
        }
        match resolve_ggml(ggml_name, &self.cfg)? {
            // Zero-copy: a plain rename (dense path + most SSM matrices), borrow the mmap directly.
            // NVFP4 modelopt weights take the repack arm (owned GGUF block bytes); else borrow.
            HfTarget::Plain(mut hf) => {
                // Hy3 changed the correction-bias key between the preview and current releases.
                // Prefer the current mapper spelling, but keep old repacks/checkpoints loadable.
                if self.cfg.arch.is_hy3()
                    && ggml_name.ends_with(".exp_probs_b.bias")
                    && self.lookup(&hf).is_none()
                {
                    let legacy = hf.replace(".mlp.expert_bias", ".mlp.router.expert_bias");
                    if self.lookup(&legacy).is_some() {
                        hf = legacy;
                    }
                }
                // NVFP4 (modelopt OR Reza) -> repack to memra internal GGUF block_nvfp4 bytes (NO kernel
                // change). `nvfp4_quant` returns the packed bytes directly (in Reza the packed tensor
                // is `<hf>.nvfp4_packed`, not `<hf>` itself), so no second lookup.
                if let Some((out_f, in_f, wbytes, wscale, _macro)) = self.nvfp4_quant(&hf) {
                    let packed =
                        crate::nvfp4_repack::repack_modelopt_to_gguf(wbytes, &wscale, out_f, in_f);
                    return Some(TensorView {
                        bytes: Cow::Owned(packed),
                        ggml_type: GgmlType::NVFP4,
                        ne: vec![in_f as u64, out_f as u64], // ggml ne: [in, out]
                    });
                }
                // FP8 E4M3 weight (NVIDIA official 27B linear_attn projections): modelopt
                // per-TENSOR quant — F8 codes + scalar F32 `weight_scale`. Re-encode host-side to
                // GGUF Q8_0 blocks (per-32 fp16 scale + int8): rides the proven q8-fast/MMVQ/fused3
                // path at ~1.06B/elem instead of a 22GB f32 blow-up (OOM, measured). Accuracy: the
                // source is 4-mantissa-bit FP8 with ONE per-tensor scale; per-32 q8 re-quant is a
                // FINER grid — class-equal or better. FP8-native matvec = later perf rung.
                if let Some((info, bytes)) = self.lookup(&hf) {
                    // Large BF16 2D matmul weights -> Q8_0 re-encode (LOADER LAW, 2026-07-08),
                    // except modules the checkpoint explicitly keeps outside quantization:
                    // ANY such weight >= 1M elements that reaches the engine as Float/FloatBf16
                    // fails `uses_q8_1_fast` and rides the slow dot_kernel+reduce_1Block cuBLAS f32
                    // GEMV pairs (the "Float-poison" trap, occurrences 1-5). Q8_0 per-32 fp16 scale
                    // + int8 is a FINER grid than BF16 (7-bit mantissa int8*scale vs bf16's 7-bit
                    // significand) — same class, strictly no worse accuracy — and puts every tensor
                    // on the proven q8-fast/MMVQ/fused3 path. Covers: mtp.* (draft, same class as
                    // the GGUF Q8_0 twin), lm_head, embed_tokens, AND any unquantized attention
                    // projections in checkpoints where they remain BF16 (compressed-tensors/apolo:
                    // linear_attn.in_proj_qkv/z/out_proj + self_attn.q/k/v/o_proj, per recipe.yaml
                    // ignore list). MEMRA_FULL_PREC=1 bypasses this to surface raw BF16 (the engine
                    // keeps them FloatBf16-resident for the MTP-heal protocol).
                    let full_prec = std::env::var("MEMRA_FULL_PREC").as_deref() == Ok("1");
                    if !full_prec
                        && !is_float_router
                        && !self.preserves_source_dtype(&hf)
                        && info.dtype == "BF16"
                        && info.shape.len() == 2
                        && info.shape.iter().product::<u64>() >= 1_000_000
                    {
                        let ne = info.ne();
                        if (bytes.len() / 2) % 32 == 0 {
                            let data: Vec<f32> = bytes
                                .chunks_exact(2)
                                .map(|c| {
                                    f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16)
                                })
                                .collect();
                            // lm_head -> Q5_K (parity with the GGUF mint's head class): on the
                            // 248,320-token Qwen3.8 vocab the Q8_0 fallback reads 1.27 GB/token
                            // vs Q5_K's 833 MB — measured ~+2 ms/token on the head. Q5_K needs
                            // in_f % 256 == 0; everything else keeps the finer Q8_0 law.
                            if ggml_name == "output.weight" && data.len().is_multiple_of(256) {
                                let out = crate::nvfp4_repack::f32_to_q5_k(&data);
                                return Some(TensorView {
                                    bytes: Cow::Owned(out),
                                    ggml_type: GgmlType::Q5_K,
                                    ne,
                                });
                            }
                            let out = crate::nvfp4_repack::f32_to_q8_0(&data);
                            return Some(TensorView {
                                bytes: Cow::Owned(out),
                                ggml_type: GgmlType::Q8_0,
                                ne,
                            });
                        }
                    }
                    if info.dtype == "F8_E4M3" && info.shape.len() == 2 {
                        let stem = hf.strip_suffix(".weight").unwrap_or(&hf);
                        if let Some((sinfo, sbytes)) = self.f8_scale_sibling(stem) {
                            // Scale sibling: per-tensor F32 scalar (modelopt / NVIDIA),
                            // per-channel [out,1] F32/BF16 (unsloth compressed-tensors FP8),
                            // or block-128 [out/128, in/128] grid (Qwen official FP8,
                            // `weight_scale_inv`). The Q8_0 per-32 re-encode grid is FINER
                            // than all three source granularities — accuracy class-equal
                            // or better in every case (the block-128 arm's 128x128 tile is
                            // 4x coarser along in_f than one Q8_0 block).
                            let (out_f, in_f) = (info.shape[0] as usize, info.shape[1] as usize);
                            if let Some(scales) = f8_scales(sinfo, sbytes, out_f, in_f) {
                                let ne = info.ne();
                                assert!(
                                    bytes.len() % 32 == 0,
                                    "F8 tensor {hf} len {} not 32-aligned",
                                    bytes.len()
                                );
                                let data = f8_deq_f32(bytes, out_f, in_f, &scales);
                                // MEMRA_NV_W4=1: F8 attention weights -> NVFP4 (0.56 B/w vs Q8_0's
                                // 1.06) — decode is bandwidth-bound and these layers are 35-40% of
                                // per-token kernel time (nsys 2026-07-07). Real e4m3->e2m1 re-quant;
                                // same 4-bit class the daily GGUF runs on these layers. Opt-in until
                                // the acceptance/text battery proves the class locally.
                                if std::env::var("MEMRA_NV_W4")
                                    .map(|v| v == "1")
                                    .unwrap_or(false)
                                    && ne[0] % 64 == 0
                                {
                                    let out = crate::nvfp4_repack::f32_to_nvfp4(&data);
                                    return Some(TensorView {
                                        bytes: Cow::Owned(out),
                                        ggml_type: GgmlType::NVFP4,
                                        ne,
                                    });
                                }
                                let out = crate::nvfp4_repack::f32_to_q8_0(&data);
                                return Some(TensorView {
                                    bytes: Cow::Owned(out),
                                    ggml_type: GgmlType::Q8_0,
                                    ne,
                                });
                            }
                        }
                    }
                }
                self.raw_hf(&hf)
            }
            // A value transform. Two paths:
            //  (a) NVFP4-preserving: a modelopt NVFP4 weight + a pure structural V-head permutation
            //      (qkv/z/a/b row reorder, out_proj col reorder) -> repack then permute the PACKED
            //      bytes, keeping the weight NVFP4 (no ~8x f32 blow-up; the macro-scale rides the
            //      `<stem>.scale` sibling, applied post-matmul exactly like the Plain NVFP4 arm).
            //  (b) f32 fallback: value transforms (`-exp`, `+1`, conv1d squeeze, identity) operate on
            //      the tiny BF16 SSM tensors; `deq_f32` is NVFP4-aware for any NVFP4 weight here too.
            HfTarget::Transform { hf, kind } => {
                if let Some((out_f, in_f, wbytes, wscale, _macro)) = self.nvfp4_quant(&hf) {
                    let packed =
                        crate::nvfp4_repack::repack_modelopt_to_gguf(wbytes, &wscale, out_f, in_f);
                    if let Some((ne, bytes)) = kind.apply_nvfp4(&packed, out_f, in_f, &self.cfg) {
                        return Some(TensorView {
                            bytes: Cow::Owned(bytes),
                            ggml_type: GgmlType::NVFP4,
                            ne,
                        });
                    }
                    // fall through to f32 (value transform on an NVFP4 weight — rare/none in qwen35).
                }
                let (mut data, ne_in) = self.deq_f32(&hf)?;
                let cfg = &self.cfg;
                let (ne, bytes) = kind.apply(&mut data, ne_in, cfg);
                // F8-E4M3-sourced LARGE 2D projections (NVIDIA 27B linear_attn in_proj_qkv/z +
                // out_proj, V-reordered above): surfacing F32 is 461MB/linear-layer -> 22GB across
                // the 48 linear layers (the load-tail OOM). Re-encode the post-reorder f32 to GGUF
                // Q8_0 (same class as the Plain-arm F8 re-encode; per-32 fp16 scale is FINER than
                // the source's single per-tensor scale). Small norm-class tensors (ssm_a/dt/
                // conv1d/norms) stay F32 — the engine consumes them via float_data().
                // BF16-sourced in_proj_a/b [48,5120]: below the 1M-element BF16 gate, but leaving
                // them F32 breaks `mixer_in_q8_1_fast` (requires beta+alpha quant) for EVERY
                // linear layer -> unfused norm path + cuBLAS f32 GEMV pairs (nsys: 96 dot+reduce
                // launches/pass + rms_norm_f32 at 12.5us vs 1.8us fused). Q8_0 puts them on the
                // fused2/dual q8-fast chain like the GGUF twin's quantized a/b.
                // BF16-sourced LARGE 2D projections (compressed-tensors: in_proj_qkv/z/out_proj +
                // self_attn q/k/v/o_proj ALL BF16): same Float-poison loader law as the Plain arm.
                // Q8_0 per-32 is FINER than BF16's 7-bit significand. The in_proj_a/b gate below
                // catches the small-but-matmul-class a/b (below the 1M-element gate but still must
                // ride q8-fast for mixer_in_q8_1_fast). MEMRA_FULL_PREC=1 bypasses both.
                let full_prec = std::env::var("MEMRA_FULL_PREC").as_deref() == Ok("1");
                let is_bf16 = self.lookup(&hf).is_some_and(|(i, _)| i.dtype == "BF16");
                if !full_prec
                    && !is_float_router
                    && !self.preserves_source_dtype(&hf)
                    && is_bf16
                    && ne.len() == 2
                    && ne[0] % 32 == 0
                    && (ne.iter().product::<u64>() >= 1_000_000
                        || hf.ends_with("in_proj_a.weight")
                        || hf.ends_with("in_proj_b.weight"))
                {
                    let f32s: Vec<f32> = bytes
                        .chunks_exact(4)
                        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                        .collect();
                    let out = crate::nvfp4_repack::f32_to_q8_0(&f32s);
                    return Some(TensorView {
                        bytes: Cow::Owned(out),
                        ggml_type: GgmlType::Q8_0,
                        ne,
                    });
                }
                let is_f8 = self.lookup(&hf).is_some_and(|(i, _)| i.dtype == "F8_E4M3");
                if is_f8
                    && ne.len() == 2
                    && ne[0] % 32 == 0
                    && ne.iter().product::<u64>() >= 1_000_000
                {
                    let f32s: Vec<f32> = bytes
                        .chunks_exact(4)
                        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                        .collect();
                    // MEMRA_NV_W4: same F8->NVFP4 re-quant as the Plain arm (see there for the
                    // bandwidth math and class argument) — post-reorder, so the V-permutation is
                    // already baked into the f32 surface.
                    if std::env::var("MEMRA_NV_W4")
                        .map(|v| v == "1")
                        .unwrap_or(false)
                        && ne[0] % 64 == 0
                    {
                        let out = crate::nvfp4_repack::f32_to_nvfp4(&f32s);
                        return Some(TensorView {
                            bytes: Cow::Owned(out),
                            ggml_type: GgmlType::NVFP4,
                            ne,
                        });
                    }
                    let out = crate::nvfp4_repack::f32_to_q8_0(&f32s);
                    return Some(TensorView {
                        bytes: Cow::Owned(out),
                        ggml_type: GgmlType::Q8_0,
                        ne,
                    });
                }
                Some(TensorView {
                    bytes: Cow::Owned(bytes),
                    ggml_type: GgmlType::F32,
                    ne,
                })
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expert_activation_precision_is_artifact_metadata_not_family_identity() {
        assert_eq!(
            expert_activation_precision_from_quant_algo(Some("W4A16_NVFP4")),
            ExpertActivationPrecision::Bf16
        );
        assert_eq!(
            expert_activation_precision_from_quant_algo(Some("NVFP4")),
            ExpertActivationPrecision::Quantized
        );
        assert_eq!(
            expert_activation_precision_from_quant_algo(None),
            ExpertActivationPrecision::F32
        );
    }

    #[test]
    fn safetensors_census_is_header_only_and_counts_quant_auxiliaries() {
        let headers = BTreeMap::from([
            (
                "model.language_model.layers.0.self_attn.q_proj.weight".to_string(),
                StInfo {
                    dtype: "U8".to_string(),
                    shape: vec![2, 4],
                    data_offsets: [0, 8],
                },
            ),
            (
                "model.language_model.layers.0.self_attn.q_proj.weight_scale".to_string(),
                StInfo {
                    dtype: "F8_E4M3".to_string(),
                    shape: vec![2, 1],
                    data_offsets: [8, 10],
                },
            ),
            (
                "model.language_model.layers.0.self_attn.q_proj.weight_scale_2".to_string(),
                StInfo {
                    dtype: "F32".to_string(),
                    shape: vec![1],
                    data_offsets: [10, 14],
                },
            ),
            (
                "model.norm.weight".to_string(),
                StInfo {
                    dtype: "BF16".to_string(),
                    shape: vec![2],
                    data_offsets: [14, 18],
                },
            ),
            (
                "model.layers.1.self_attn.q_proj.weight_packed".to_string(),
                StInfo {
                    dtype: "U8".to_string(),
                    shape: vec![2, 4],
                    data_offsets: [18, 26],
                },
            ),
            (
                "model.layers.1.self_attn.q_proj.weight_scale".to_string(),
                StInfo {
                    dtype: "F8_E4M3".to_string(),
                    shape: vec![2, 1],
                    data_offsets: [26, 28],
                },
            ),
            (
                "model.layers.1.self_attn.q_proj.weight_global_scale".to_string(),
                StInfo {
                    dtype: "F32".to_string(),
                    shape: vec![1],
                    data_offsets: [28, 32],
                },
            ),
        ]);
        let census = census_from_safetensors_headers(&headers).unwrap();
        assert_eq!(census.dialect, CheckpointDialect::HfSafetensors);
        assert_eq!(census.tensors.len(), 3);
        let weight = census
            .tensors
            .iter()
            .find(|row| row.entry.name == "model.layers.0.self_attn.q_proj.weight")
            .unwrap();
        assert_eq!(weight.entry.shape, vec![2, 8]);
        assert_eq!(weight.entry.physical_bytes, 14);
        let StorageLayout::Quantized(layout) = &weight.entry.storage else {
            panic!("packed U8 weight must be quantized");
        };
        assert_eq!(layout.format, "NVFP4");
        assert_eq!(layout.auxiliaries.len(), 2);
        let norm = census
            .tensors
            .iter()
            .find(|row| row.entry.name == "model.norm.weight")
            .unwrap();
        assert_eq!(norm.entry.physical_bytes, 4);
        let packed = census
            .tensors
            .iter()
            .find(|row| row.entry.name == "model.layers.1.self_attn.q_proj.weight")
            .unwrap();
        assert_eq!(
            packed.physical_name,
            "model.layers.1.self_attn.q_proj.weight_packed"
        );
        assert_eq!(packed.entry.physical_bytes, 14);
    }

    #[test]
    fn repack_census_reads_manifest_lengths_without_tensor_views() {
        let dir = std::env::temp_dir().join(format!("memra-repack-census-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("weights.bin"), [0u8; 16]).unwrap();
        std::fs::write(
            dir.join("config.json"),
            r#"{"model_type":"qwen3","num_hidden_layers":1,"hidden_size":4,
                "num_attention_heads":1,"num_key_value_heads":1,"head_dim":4,
                "intermediate_size":8,"vocab_size":4,"max_position_embeddings":16}"#,
        )
        .unwrap();
        std::fs::write(
            dir.join("manifest.json"),
            r#"{"format":"memra-repack-v1","tensors":{
                "blk.0.attn_q.weight":{"file":"weights.bin","qtype":"F32",
                "ne":[4,1],"bytes":16}}}"#,
        )
        .unwrap();
        let source = Hy3RepackSource::open(&dir).unwrap();
        let census = source.tensor_census().unwrap();
        assert_eq!(census.dialect, CheckpointDialect::Gguf);
        assert_eq!(census.tensors.len(), 1);
        assert_eq!(census.tensors[0].entry.name, "blk.0.attn_q.weight");
        assert_eq!(census.tensors[0].entry.physical_bytes, 16);
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn repack_manifest_requires_exact_encoded_extent() {
        let dir = std::env::temp_dir().join(format!("memra-repack-extent-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("weights.bin"), [0u8; 32]).unwrap();
        std::fs::write(
            dir.join("config.json"),
            r#"{"model_type":"qwen3","num_hidden_layers":1,"hidden_size":4,
                "num_attention_heads":1,"num_key_value_heads":1,"head_dim":4,
                "intermediate_size":8,"vocab_size":4,"max_position_embeddings":16}"#,
        )
        .unwrap();
        std::fs::write(
            dir.join("manifest.json"),
            r#"{"format":"memra-repack-v1","tensors":{
                "blk.0.attn_q.weight":{"file":"weights.bin","qtype":"F32",
                "ne":[4,1],"bytes":8}}}"#,
        )
        .unwrap();
        let error = match Hy3RepackSource::open(&dir) {
            Ok(_) => panic!("mismatched manifest byte extent was accepted"),
            Err(error) => error.to_string(),
        };
        assert!(error.contains("encodes exactly 16"), "{error}");
        std::fs::remove_dir_all(dir).ok();
    }

    #[cfg(unix)]
    #[test]
    fn repack_shard_symlink_is_rejected_without_an_explicit_trusted_root() {
        use std::os::unix::fs::symlink;

        let dir = std::env::temp_dir().join(format!("memra-repack-link-{}", std::process::id()));
        let outside =
            std::env::temp_dir().join(format!("memra-repack-outside-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(&outside, [0u8; 16]).unwrap();
        symlink(&outside, dir.join("weights.bin")).unwrap();
        std::fs::write(
            dir.join("config.json"),
            r#"{"model_type":"qwen3","num_hidden_layers":1,"hidden_size":4,
                "num_attention_heads":1,"num_key_value_heads":1,"head_dim":4,
                "intermediate_size":8,"vocab_size":4,"max_position_embeddings":16}"#,
        )
        .unwrap();
        std::fs::write(
            dir.join("manifest.json"),
            r#"{"format":"memra-repack-v1","tensors":{
                "blk.0.attn_q.weight":{"file":"weights.bin","qtype":"F32",
                "ne":[4,1],"bytes":16}}}"#,
        )
        .unwrap();
        assert!(Hy3RepackSource::open(&dir).is_err());
        std::fs::remove_dir_all(dir).ok();
        std::fs::remove_file(outside).ok();
    }

    #[test]
    fn manifest_payload_paths_cannot_escape_artifact_root() {
        assert_eq!(
            validated_manifest_file("experts/layer-000.bin").unwrap(),
            PathBuf::from("experts/layer-000.bin")
        );
        for unsafe_path in [
            "",
            "/etc/passwd",
            "../outside",
            "experts/../../outside",
            "./x",
        ] {
            assert!(
                validated_manifest_file(unsafe_path).is_err(),
                "accepted unsafe path {unsafe_path:?}"
            );
        }
    }

    #[test]
    fn expert_mmap_advice_parser_preserves_random_default() {
        assert_eq!(parse_expert_mmap_advice(None), Ok(ExpertMmapAdvice::Random));
        assert_eq!(
            parse_expert_mmap_advice(Some("random")),
            Ok(ExpertMmapAdvice::Random)
        );
        assert_eq!(
            parse_expert_mmap_advice(Some("normal")),
            Ok(ExpertMmapAdvice::Normal)
        );
        assert!(parse_expert_mmap_advice(Some("sequential")).is_err());
        assert!(parse_expert_mmap_advice(Some("")).is_err());
    }

    #[test]
    fn safetensors_source_find_maps_names() {
        // Build a tiny HF-named safetensors file + config.json, open via SafetensorsSource,
        // and assert ggml-name lookups resolve through the mapper with reversed shape.
        let dir = std::env::temp_dir().join(format!("memra_src_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();

        // one F32 attn_q weight, HF shape [out=4, in=2] -> ne should be [2,4]
        let json = r#"{"model.layers.0.self_attn.q_proj.weight":{"dtype":"F32","shape":[4,2],"data_offsets":[0,32]}}"#;
        let mut buf = Vec::new();
        buf.extend_from_slice(&(json.len() as u64).to_le_bytes());
        buf.extend_from_slice(json.as_bytes());
        for v in 0..8u32 {
            buf.extend_from_slice(&(v as f32).to_le_bytes());
        }
        std::fs::write(dir.join("model.safetensors"), &buf).unwrap();

        let cfg_json = r#"{"model_type":"qwen3","num_hidden_layers":1,"hidden_size":4,"num_attention_heads":2,"intermediate_size":8,"vocab_size":10,"max_position_embeddings":128}"#;
        std::fs::write(dir.join("config.json"), cfg_json).unwrap();

        let src = SafetensorsSource::open(&dir).unwrap();
        assert_eq!(src.config().arch, Arch::Qwen3);
        assert_eq!(src.config().n_layer, 1);

        let v = src
            .find("blk.0.attn_q.weight")
            .expect("ggml name maps to HF and resolves");
        assert_eq!(v.ggml_type, GgmlType::F32);
        // shape-reversal assertion: HF [out=4,in=2] -> ne [in=2,out=4]
        assert_eq!(v.ne, vec![2, 4]);
        assert_eq!(v.bytes.len(), 32);
        assert!(src.has("blk.0.attn_q.weight"));
        // unmapped ggml name (no SSM tensors in this dense model)
        assert!(src.find("blk.0.ssm_a").is_none());

        std::fs::remove_dir_all(&dir).ok();
    }

    /// A checkpoint that stores its NVFP4 scales swizzled must be READ swizzled, an undeclared
    /// one must stay linear, and an unrecognised declaration must refuse to load.
    ///
    /// This gate exists because the failure it guards is invisible: the swizzled plane declares
    /// the same `[out, in/16]` shape a linear plane declares, so every existing validity check
    /// passes on the wrong reading and the model loads, serves, and speaks — measured on
    /// `tiyuvta/GLM-5.3-Flash-NVFP4-B200-hybrid` against its BF16 source, a linear read of a
    /// swizzled plane gives cosine 0.847 to the true weights. The RED ARM below (no LAYOUT.json,
    /// same bytes) asserts the corrupted reading actually differs, so a future change that
    /// silently unswizzles everything, or nothing, cannot leave this test green.
    #[test]
    fn nvfp4_scale_layout_is_declared_and_unrecognised_declarations_refuse_to_load() {
        use crate::nvfp4_repack::{repack_modelopt_to_gguf, swizzle_blockscale};

        let (out_f, in_f) = (256usize, 128usize);
        let (wlen, slen) = (out_f * in_f / 2, out_f * in_f / 16);
        // Distinct per (row, col) so ANY mis-permutation changes the output. All bytes are finite
        // ue4m3 codes well inside the normal range.
        let linear_scale: Vec<u8> = (0..slen)
            .map(|i| 0x30u8 + ((i * 7 + i / 8) % 32) as u8)
            .collect();
        let swizzled = swizzle_blockscale(&linear_scale, out_f, in_f);
        assert_eq!(swizzled.len(), slen, "no padding at these dims");
        assert_ne!(swizzled, linear_scale, "the permutation must move bytes");
        let weight: Vec<u8> = (0..wlen).map(|i| ((i * 31 + 5) % 251) as u8).collect();
        let want = repack_modelopt_to_gguf(&weight, &linear_scale, out_f, in_f);

        let dir = std::env::temp_dir().join(format!("memra_nvfp4_swz_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let header = format!(
            r#"{{
              "model.layers.1.self_attn.q_proj.weight":{{"dtype":"U8","shape":[{out_f},{}],"data_offsets":[0,{wlen}]}},
              "model.layers.1.self_attn.q_proj.weight_scale":{{"dtype":"F8_E4M3","shape":[{out_f},{}],"data_offsets":[{wlen},{}]}},
              "model.layers.1.self_attn.q_proj.weight_scale_2":{{"dtype":"F32","shape":[],"data_offsets":[{},{}]}}
            }}"#,
            in_f / 2,
            in_f / 16,
            wlen + slen,
            wlen + slen,
            wlen + slen + 4,
        );
        let mut file = Vec::new();
        file.extend_from_slice(&(header.len() as u64).to_le_bytes());
        file.extend_from_slice(header.as_bytes());
        file.extend_from_slice(&weight);
        file.extend_from_slice(&swizzled);
        file.extend_from_slice(&1.0f32.to_le_bytes());
        std::fs::write(dir.join("model.safetensors"), file).unwrap();
        std::fs::write(
            dir.join("config.json"),
            r#"{
              "model_type":"step3p5","num_hidden_layers":2,"hidden_size":128,
              "intermediate_size":256,"num_attention_heads":1,"num_attention_groups":1,
              "head_dim":128,"vocab_size":64,"max_position_embeddings":2048,
              "moe_num_experts":2,"moe_top_k":1,"moe_intermediate_size":128,
              "share_expert_dim":128,"moe_layers_enum":"1",
              "moe_router_activation":"sigmoid",
              "layer_types":["full_attention","sliding_attention"],
              "rope_theta":[5000000,10000],"partial_rotary_factors":[0.5,1],
              "sliding_window":512,
              "attention_other_setting":{"num_attention_heads":1,"num_attention_groups":1},
              "swiglu_limits":[0,0],"swiglu_limits_shared":[0,0]
            }"#,
        )
        .unwrap();
        let name = "blk.1.attn_q.weight";

        // ARM 1 — declared swizzled: the loader must undo the permutation.
        std::fs::write(
            dir.join("LAYOUT.json"),
            r#"{"nvfp4_scale":"Swizzle32x4x4 float8_e4m3fn padded N%128 K%4"}"#,
        )
        .unwrap();
        let src = SafetensorsSource::open(&dir).unwrap();
        assert_eq!(src.nvfp4_scale_layout, Nvfp4ScaleLayout::Swizzle32x4x4);
        let swz_tag = src.nvfp4_cache_tag();
        assert!(
            !swz_tag.is_empty(),
            "a swizzled read must key its repack cache"
        );
        let got = src.find(name).expect("NVFP4 weight").bytes.into_owned();
        drop(src);
        assert_eq!(
            got, want,
            "declared-swizzled read must equal the linear truth"
        );

        // ARM 2 (RED) — same bytes, no declaration: read linear, and therefore WRONG. If this
        // ever equals `want`, the unswizzle is a no-op and arm 1 proves nothing.
        std::fs::remove_file(dir.join("LAYOUT.json")).unwrap();
        let src = SafetensorsSource::open(&dir).unwrap();
        assert_eq!(src.nvfp4_scale_layout, Nvfp4ScaleLayout::Linear);
        // The expert repack cache lives in the checkpoint dir and is size-only-fresh, so the two
        // readings MUST NOT share a cache filename: a linear-read cache reused by a swizzled read
        // would restore the corruption from disk, invisibly.
        assert_ne!(
            src.nvfp4_cache_tag(),
            swz_tag,
            "linear and swizzled reads share a repack cache key"
        );
        let raw = src.find(name).expect("NVFP4 weight").bytes.into_owned();
        drop(src);
        assert_ne!(
            raw, want,
            "reading a swizzled plane as linear must differ — otherwise this gate is vacuous"
        );

        // ARM 3 — a layout this build does not know must FAIL the open, not fall back to linear.
        std::fs::write(
            dir.join("LAYOUT.json"),
            r#"{"nvfp4_scale":"SomeFutureOrder"}"#,
        )
        .unwrap();
        let err = match SafetensorsSource::open(&dir) {
            Err(e) => e,
            Ok(_) => panic!("an unrecognised nvfp4_scale layout must refuse to load"),
        };
        assert!(
            err.to_string().contains("SomeFutureOrder"),
            "the error must name the layout it could not read, got: {err}"
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn step_stacked_fp8_native_preserves_expert_scale_and_file_axes() {
        let dir = std::env::temp_dir().join(format!("memra_step_fp8_stack_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let weight_len = 2 * 128 * 128;
        let scale_len = 2 * 4;
        let header = format!(
            r#"{{
              "model.layers.1.moe.gate_proj.weight":{{"dtype":"F8_E4M3","shape":[2,128,128],"data_offsets":[0,{weight_len}]}},
              "model.layers.1.moe.gate_proj.weight_scale_inv":{{"dtype":"F32","shape":[2,1,1],"data_offsets":[{weight_len},{}]}}
            }}"#,
            weight_len + scale_len,
        );
        let mut file = Vec::new();
        file.extend_from_slice(&(header.len() as u64).to_le_bytes());
        file.extend_from_slice(header.as_bytes());
        file.extend(std::iter::repeat_n(0x38u8, weight_len));
        file.extend_from_slice(&0.25f32.to_le_bytes());
        file.extend_from_slice(&0.5f32.to_le_bytes());
        std::fs::write(dir.join("model.safetensors"), file).unwrap();
        std::fs::write(
            dir.join("config.json"),
            r#"{
              "model_type":"step3p5","num_hidden_layers":2,"hidden_size":128,
              "intermediate_size":256,"num_attention_heads":1,"num_attention_groups":1,
              "head_dim":128,"vocab_size":64,"max_position_embeddings":2048,
              "moe_num_experts":2,"moe_top_k":1,"moe_intermediate_size":128,
              "share_expert_dim":128,"moe_layers_enum":"1",
              "moe_router_activation":"sigmoid",
              "layer_types":["full_attention","sliding_attention"],
              "rope_theta":[5000000,10000],"partial_rotary_factors":[0.5,1],
              "sliding_window":512,
              "attention_other_setting":{"num_attention_heads":1,"num_attention_groups":1},
              "swiglu_limits":[0,0],"swiglu_limits_shared":[0,0]
            }"#,
        )
        .unwrap();

        let src = SafetensorsSource::open(&dir).unwrap();
        let name = "blk.1.ffn_gate_exps.weight";
        assert!(src.has(name));
        let fp8 = src.find_fp8_stacked_native(name).expect("stacked FP8 bank");
        assert_eq!((fp8.n_expert, fp8.out_f, fp8.in_f), (2, 128, 128));
        assert_eq!((fp8.scale_rows, fp8.scale_cols), (1, 1));
        assert_eq!(fp8.bytes.len(), weight_len);
        assert_eq!(fp8.scales, vec![0.25, 0.5]);

        let extent = src
            .find_expert_disk(name)
            .expect("stacked FP8 must expose an owned extent");
        assert_eq!(extent.len, weight_len);
        let offset = usize::try_from(extent.offset).unwrap();
        assert_eq!(&extent.map[offset..offset + extent.len], fp8.bytes);
        drop(fp8);
        drop(src);
        assert_eq!(
            &extent.map[offset..offset + extent.len],
            vec![0x38u8; weight_len]
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn exact_step37_fp8_preserves_every_bf16_checkpoint_surface() {
        let dir = std::env::temp_dir().join(format!("memra_step_bf16_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();

        let router_len = 288 * 4096 * 2;
        let q_len = 256 * 4096 * 2;
        let header = format!(
            r#"{{
              "model.layers.1.moe.gate.weight":{{"dtype":"BF16","shape":[288,4096],"data_offsets":[0,{router_len}]}},
              "model.layers.1.self_attn.q_proj.weight":{{"dtype":"BF16","shape":[256,4096],"data_offsets":[{router_len},{}]}}
            }}"#,
            router_len + q_len,
        );
        let mut file = Vec::with_capacity(8 + header.len() + router_len + q_len);
        file.extend_from_slice(&(header.len() as u64).to_le_bytes());
        file.extend_from_slice(header.as_bytes());
        file.resize(file.len() + router_len + q_len, 0);
        std::fs::write(dir.join("model.safetensors"), file).unwrap();
        std::fs::write(
            dir.join("config.json"),
            r#"{
              "model_type":"step3p7","num_hidden_layers":2,"hidden_size":4096,
              "intermediate_size":8192,"num_attention_heads":32,"num_key_value_heads":8,
              "head_dim":128,"vocab_size":64,"max_position_embeddings":2048,
              "moe_num_experts":288,"moe_top_k":8,"moe_intermediate_size":1280,
              "share_expert_dim":5120,"moe_layers_enum":"1",
              "layer_types":["full_attention","sliding_attention"],
              "rope_theta":[5000000,10000],"partial_rotary_factors":[0.5,1],
              "sliding_window":512,
              "attention_other_setting":{"num_attention_heads":32,"num_attention_groups":8},
              "swiglu_limits":[0,0],"swiglu_limits_shared":[0,0],
              "quantization_config":{
                "quant_method":"fp8","activation_scheme":"dynamic","fmt":"e4m3",
                "weight_block_size":[128,128],
                "modules_to_not_convert":["model.layers.1.moe.gate"]
              }
            }"#,
        )
        .unwrap();

        let src = SafetensorsSource::open(&dir).unwrap();
        assert!(src.preserve_checkpoint_bf16);
        assert!(src.preserves_source_dtype("model.layers.1.moe.gate.weight"));
        assert!(src.preserves_source_dtype("model.layers.1.self_attn.q_proj.weight"));

        let router = src
            .find("blk.1.ffn_gate_inp.weight")
            .expect("Step router must resolve");
        assert_eq!(router.ggml_type, GgmlType::BF16);
        assert_eq!(router.ne, vec![4096, 288]);
        assert_eq!(router.bytes.len(), router_len);

        let q = src
            .find("blk.1.attn_q.weight")
            .expect("Step Q projection must resolve");
        assert_eq!(q.ggml_type, GgmlType::BF16);
        assert_eq!(q.ne, vec![4096, 256]);
        assert_eq!(q.bytes.len(), q_len);

        std::fs::remove_dir_all(&dir).ok();
    }

    /// END TO END through the real source: a glm5_next MLA layer's `attn_k_b`/`attn_v_b` are
    /// produced from the checkpoint's FUSED `kv_b_proj` by the `SplitMlaKv` transform, and come
    /// out F32 with a 3-D `ne` — the only layout the absorbed MLA kernels can read.
    ///
    /// WHAT THIS PINS THAT THE UNIT TESTS CANNOT (glm53-flash lane, 2026-08-28). `hf_mapping`'s
    /// tests cover the name resolution and the split arithmetic in isolation. This one drives the
    /// Transform arm of `find` itself, where the outcome depends on a gate NOT firing: that arm
    /// re-encodes a BF16 source to Q8_0 when `ne.len() == 2 && ne[0] % 32 == 0` and the tensor has
    /// >= 1M elements. The fused weight here is 1,572,864 elements with `ne[0] == 128`, so it
    /// > satisfies every one of those conditions — and the ONLY thing keeping the split planes off
    /// > the quantized path is their 3-D `ne`. The contrast is asserted directly below: the fused
    /// > tensor, asked for by its own name, DOES come back Q8_0.
    ///
    /// The config carries NO `modules_to_not_convert`, which is exactly our own NVFP4 mint's
    /// situation (it writes its keep list under the compressed-tensors `ignore` key, which the
    /// engine does not read). So this fixture is the artifact we actually intend to load, not a
    /// friendly one.
    #[test]
    fn glm5_next_mla_kv_b_splits_to_f32_3d_planes_through_the_source() {
        let (heads, nope, v, rank) = (64usize, 128usize, 64usize, 128usize);
        let rows = heads * (nope + v);
        let elements = rows * rank;
        assert!(
            elements >= 1_000_000,
            "the fixture must cross the loader's >= 1M BF16 re-encode gate to be meaningful"
        );

        // Deterministic values, all integers < 256 so BF16 holds them EXACTLY — the comparison
        // below is then byte-exact and cannot pass on rounding slack.
        let value = |row: usize, r: usize| ((row * 7 + r * 3) % 251) as f32;
        let mut payload = Vec::with_capacity(elements * 2);
        for row in 0..rows {
            for r in 0..rank {
                let bits = value(row, r).to_bits();
                payload.extend_from_slice(&((bits >> 16) as u16).to_le_bytes());
            }
        }

        let dir = std::env::temp_dir().join(format!("memra_glm5_kvb_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let name = "model.layers.1.self_attn.kv_b_proj.weight";
        let header = format!(
            r#"{{"{name}":{{"dtype":"BF16","shape":[{rows},{rank}],"data_offsets":[0,{}]}}}}"#,
            payload.len()
        );
        let mut file = Vec::with_capacity(8 + header.len() + payload.len());
        file.extend_from_slice(&(header.len() as u64).to_le_bytes());
        file.extend_from_slice(header.as_bytes());
        file.extend_from_slice(&payload);
        std::fs::write(dir.join("model.safetensors"), file).unwrap();
        std::fs::write(
            dir.join("config.json"),
            format!(
                r#"{{
                  "model_type":"glm5_next_text","num_hidden_layers":2,
                  "num_nextn_predict_layers":0,"hidden_size":128,"intermediate_size":64,
                  "vocab_size":32,"max_position_embeddings":512,"rms_norm_eps":1e-05,
                  "hidden_act":"silu","swiglu_limit":1e30,"tie_word_embeddings":true,
                  "hc_mult":4,"hc_eps":1e-06,"hc_sinkhorn_iters":20,"mhc":true,
                  "layer_types":["linear_attention","deepseek_sparse_attention"],
                  "mlp_layer_types":["dense","dense"],"first_k_dense_replace":2,
                  "indexer_types":["full","full"],
                  "linear_attn_config":{{"num_heads":1,"head_dim":128,
                    "short_conv_kernel_size":4,"gate_lower_bound":-5.0,
                    "kda_layers":[0],"full_attn_layers":[1]}},
                  "num_attention_heads":{heads},"num_key_value_heads":1,
                  "q_lora_rank":4,"kv_lora_rank":{rank},"qk_head_dim":{nope},
                  "qk_nope_head_dim":{nope},"qk_rope_head_dim":0,"v_head_dim":{v},
                  "mla_use_nope":true,"index_n_heads":1,"index_head_dim":8,"index_topk":8,
                  "index_kpool":4,"index_kpool_always_select_tail":true,
                  "index_kpool_compress":true,"indexer_rope_interleave":true,
                  "index_share_for_mtp_iteration":true,
                  "n_routed_experts":4,"num_experts_per_tok":2,"moe_intermediate_size":32,
                  "n_shared_experts":1,"scoring_func":"sigmoid","topk_method":"noaux_tc",
                  "routed_scaling_factor":2.5,"norm_topk_prob":true,"n_group":1,"topk_group":1,
                  "head_dim":0,"attention_bias":false,"moe_router_dtype":"float32",
                  "dtype":"bfloat16"
                }}"#
            ),
        )
        .unwrap();

        let src = SafetensorsSource::open(&dir).unwrap();
        assert!(
            !src.preserves_source_dtype(name),
            "the fixture must NOT be dtype-preserved — that is the mint situation under test"
        );

        // CONTRAST: the fused tensor by its own name takes the Plain/Transform 2-D arm and IS
        // re-encoded. This is what the split planes are being exempted from.
        let fused = src
            .find("blk.1.attn_kv_b.weight")
            .expect("the fused kv_b_proj must resolve");
        assert_eq!(
            fused.ggml_type,
            GgmlType::Q8_0,
            "the fused 2-D weight must take the >= 1M BF16 re-encode law; if it does not, the \
             3-D exemption asserted below proves nothing"
        );

        let f32_le =
            |data: &[f32]| -> Vec<u8> { data.iter().flat_map(|x| x.to_le_bytes()).collect() };

        // attn_k_b: ne [nope, rank, head], row-major [head][rank][nope] — the TRANSPOSED slice.
        // Expected values are indexed here from the FUSED layout directly (the micro_gguf
        // formula), never by calling the split helper, so this cannot pass by agreeing with the
        // implementation's own mistake.
        let mut want_k = vec![0f32; heads * rank * nope];
        let mut want_v = vec![0f32; heads * v * rank];
        for h in 0..heads {
            for r in 0..rank {
                for p in 0..nope {
                    want_k[h * rank * nope + r * nope + p] = value(h * (nope + v) + p, r);
                }
            }
            for j in 0..v {
                for r in 0..rank {
                    want_v[h * v * rank + j * rank + r] = value(h * (nope + v) + nope + j, r);
                }
            }
        }

        let k_b = src
            .find("blk.1.attn_k_b.weight")
            .expect("attn_k_b must resolve through the SplitMlaKv transform");
        assert_eq!(
            k_b.ggml_type,
            GgmlType::F32,
            "attn_k_b must stay F32 — the absorb kernel takes a raw f32 slice"
        );
        assert_eq!(k_b.ne, vec![nope as u64, rank as u64, heads as u64]);
        assert_eq!(k_b.bytes.as_ref(), f32_le(&want_k).as_slice());

        let v_b = src
            .find("blk.1.attn_v_b.weight")
            .expect("attn_v_b must resolve through the SplitMlaKv transform");
        assert_eq!(v_b.ggml_type, GgmlType::F32);
        assert_eq!(v_b.ne, vec![rank as u64, v as u64, heads as u64]);
        assert_eq!(v_b.bytes.as_ref(), f32_le(&want_v).as_slice());

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn hy3_nvfp4_formats_preserve_deliberately_unquantized_bf16_linears() {
        for quant_method in ["modelopt", "compressed-tensors"] {
            let dir = std::env::temp_dir().join(format!(
                "memra_hy3_{quant_method}_bf16_{}",
                std::process::id()
            ));
            std::fs::create_dir_all(&dir).unwrap();
            let bytes = 1024 * 1024 * 2;
            let header = format!(
                r#"{{"model.layers.0.self_attn.q_proj.weight":{{"dtype":"BF16","shape":[1024,1024],"data_offsets":[0,{bytes}]}}}}"#
            );
            let mut file = Vec::with_capacity(8 + header.len() + bytes);
            file.extend_from_slice(&(header.len() as u64).to_le_bytes());
            file.extend_from_slice(header.as_bytes());
            file.resize(file.len() + bytes, 0);
            std::fs::write(dir.join("model.safetensors"), file).unwrap();
            std::fs::write(
                dir.join("config.json"),
                format!(
                    r#"{{
                      "model_type":"hy_v3","num_hidden_layers":1,"hidden_size":1024,
                      "intermediate_size":2048,"num_attention_heads":8,"num_key_value_heads":2,
                      "head_dim":128,"vocab_size":64,"max_position_embeddings":2048,
                      "quantization_config":{{
                        "quant_method":"{quant_method}","quant_algo":"MIXED_PRECISION"
                      }}
                    }}"#
                ),
            )
            .unwrap();

            let src = SafetensorsSource::open(&dir).unwrap();
            assert!(src.preserve_checkpoint_bf16, "{quant_method}");
            let q = src
                .find("blk.0.attn_q.weight")
                .unwrap_or_else(|| panic!("Hy3 {quant_method} BF16 Q projection must resolve"));
            assert_eq!(q.ggml_type, GgmlType::BF16, "{quant_method}");
            assert_eq!(q.ne, vec![1024, 1024], "{quant_method}");
            assert_eq!(q.bytes.len(), bytes, "{quant_method}");

            std::fs::remove_dir_all(&dir).ok();
        }
    }

    #[test]
    fn hy3_expert_bias_resolves_current_and_preview_keys() {
        for (tag, hf_name) in [
            ("current", "model.layers.1.mlp.expert_bias"),
            ("preview", "model.layers.1.mlp.router.expert_bias"),
            ("modelopt", "model.layers.1.mlp.e_score_correction_bias"),
        ] {
            let dir =
                std::env::temp_dir().join(format!("memra_hy3_bias_{tag}_{}", std::process::id()));
            std::fs::create_dir_all(&dir).unwrap();

            let header =
                format!(r#"{{"{hf_name}":{{"dtype":"F32","shape":[3],"data_offsets":[0,12]}}}}"#);
            let mut buf = Vec::new();
            buf.extend_from_slice(&(header.len() as u64).to_le_bytes());
            buf.extend_from_slice(header.as_bytes());
            for value in [0.25f32, -0.5, 0.75] {
                buf.extend_from_slice(&value.to_le_bytes());
            }
            std::fs::write(dir.join("model.safetensors"), buf).unwrap();
            std::fs::write(
                dir.join("config.json"),
                r#"{"model_type":"hy_v3","num_hidden_layers":2,"hidden_size":4,"num_attention_heads":1,"num_key_value_heads":1,"head_dim":4,"intermediate_size":8,"vocab_size":10,"max_position_embeddings":128,"num_experts":3,"num_experts_per_tok":1,"moe_intermediate_size":4,"first_k_dense_replace":1,"moe_router_use_sigmoid":true,"moe_router_enable_expert_bias":true}"#,
            ).unwrap();

            let src = SafetensorsSource::open(&dir).unwrap();
            let bias = src
                .find("blk.1.exp_probs_b.bias")
                .unwrap_or_else(|| panic!("Hy3 {tag} expert-bias key did not resolve"));
            assert_eq!(bias.ggml_type, GgmlType::F32);
            assert_eq!(bias.ne, vec![3]);
            assert_eq!(bias.bytes.len(), 12);

            std::fs::remove_dir_all(&dir).ok();
        }
    }

    #[test]
    fn hy3_modelopt_router_and_shared_expert_aliases_resolve() {
        let dir =
            std::env::temp_dir().join(format!("memra_hy3_modelopt_aliases_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();

        let router_bytes = 3 * 4 * 2;
        let shared_bytes = 4 * 4 * 2;
        let header = format!(
            r#"{{
              "model.layers.1.mlp.gate.weight":{{"dtype":"BF16","shape":[3,4],"data_offsets":[0,{router_bytes}]}},
              "model.layers.1.mlp.shared_experts.gate_proj.weight":{{"dtype":"BF16","shape":[4,4],"data_offsets":[{router_bytes},{}]}}
            }}"#,
            router_bytes + shared_bytes,
        );
        let mut buf = Vec::with_capacity(8 + header.len() + router_bytes + shared_bytes);
        buf.extend_from_slice(&(header.len() as u64).to_le_bytes());
        buf.extend_from_slice(header.as_bytes());
        buf.resize(buf.len() + router_bytes + shared_bytes, 0);
        std::fs::write(dir.join("model.safetensors"), buf).unwrap();
        std::fs::write(
            dir.join("config.json"),
            r#"{"model_type":"hy_v3","num_hidden_layers":2,"hidden_size":4,"num_attention_heads":1,"num_key_value_heads":1,"head_dim":4,"intermediate_size":8,"vocab_size":10,"max_position_embeddings":128,"num_experts":3,"num_experts_per_tok":1,"moe_intermediate_size":4,"num_shared_experts":1,"first_k_dense_replace":1,"moe_router_use_sigmoid":true,"moe_router_enable_expert_bias":true,"quantization_config":{"quant_method":"modelopt","quant_algo":"MIXED_PRECISION"}}"#,
        )
        .unwrap();

        let src = SafetensorsSource::open(&dir).unwrap();
        let router = src
            .find("blk.1.ffn_gate_inp.weight")
            .expect("ModelOpt-flattened Hy3 router must resolve");
        assert_eq!(router.ggml_type, GgmlType::BF16);
        assert_eq!(router.ne, vec![4, 3]);

        let shared = src
            .find("blk.1.ffn_gate_shexp.weight")
            .expect("ModelOpt-renamed Hy3 shared expert must resolve");
        assert_eq!(shared.ggml_type, GgmlType::BF16);
        assert_eq!(shared.ne, vec![4, 4]);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn expert_overlay_overrides_selected_tensor_and_falls_back_to_hf() {
        let root = std::env::temp_dir().join(format!("memra_overlay_test_{}", std::process::id()));
        let base = root.join("base");
        let overlay = root.join("overlay");
        std::fs::create_dir_all(&base).unwrap();
        std::fs::create_dir_all(overlay.join("experts")).unwrap();

        let json = r#"{"model.layers.0.self_attn.q_proj.weight":{"dtype":"F32","shape":[4,2],"data_offsets":[0,32]}}"#;
        let mut buf = Vec::new();
        buf.extend_from_slice(&(json.len() as u64).to_le_bytes());
        buf.extend_from_slice(json.as_bytes());
        for v in 0..8u32 {
            buf.extend_from_slice(&(v as f32).to_le_bytes());
        }
        std::fs::write(base.join("model.safetensors"), &buf).unwrap();
        let cfg_json = r#"{"model_type":"qwen3","num_hidden_layers":1,"hidden_size":4,"num_attention_heads":2,"intermediate_size":8,"vocab_size":10,"max_position_embeddings":128}"#;
        std::fs::write(base.join("config.json"), cfg_json).unwrap();

        let q4k = vec![0xa5u8; 144];
        std::fs::write(overlay.join("experts/e0.q4k"), &q4k).unwrap();
        let manifest = format!(
            r#"{{
            "format":"memra-expert-overlay-v1",
            "source_dir":"{}",
            "tensors":{{
                "blk.0.ffn_gate_exps.0.weight":{{
                    "file":"experts/e0.q4k","qtype":"Q4_K","ne":[256,1],"bytes":144
                }}
            }}
        }}"#,
            base.display()
        );
        std::fs::write(overlay.join("manifest.json"), manifest).unwrap();

        let src = Hy3RepackSource::open(&overlay).unwrap();
        assert!(src.preserve_expert_encodings());
        let selected = src.find("blk.0.ffn_gate_exps.0.weight").unwrap();
        assert_eq!(selected.ggml_type, GgmlType::Q4_K);
        assert_eq!(&*selected.bytes, &q4k);
        let (map, off, len) = src
            .find_expert_mmap("blk.0.ffn_gate_exps.0.weight")
            .unwrap();
        assert_eq!(&map[off..off + len], &q4k);
        let fallback = src.find("blk.0.attn_q.weight").unwrap();
        assert_eq!(fallback.ggml_type, GgmlType::F32);
        assert_eq!(fallback.ne, vec![2, 4]);
        assert_eq!(src.st_dir(), Some(base.as_path()));

        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn v2_overlay_supports_repack_fallback_offsets_and_prune_mask() {
        let root =
            std::env::temp_dir().join(format!("memra_overlay_v2_test_{}", std::process::id()));
        let base = root.join("base");
        let overlay = root.join("overlay");
        std::fs::create_dir_all(&base).unwrap();
        std::fs::create_dir_all(overlay.join("experts")).unwrap();
        let cfg = r#"{"model_type":"qwen3_moe","num_hidden_layers":1,"hidden_size":4,"num_attention_heads":2,"num_key_value_heads":1,"intermediate_size":8,"vocab_size":10,"max_position_embeddings":128,"num_experts":3,"num_experts_per_tok":2,"moe_intermediate_size":4}"#;
        std::fs::write(base.join("config.json"), cfg).unwrap();
        std::fs::write(base.join("dense.bin"), [9u8, 8, 7, 1, 2, 3, 4, 6]).unwrap();
        std::fs::write(
            base.join("manifest.json"),
            r#"{
            "format":"memra-test-repack-v1","source_dir":".",
            "tensors":{"blk.0.attn_norm.weight":{
                "file":"dense.bin","offset":3,"qtype":"F32","ne":[1],"bytes":4
            }}
        }"#,
        )
        .unwrap();

        let mut expert_blob = vec![0x55u8, 0x66];
        expert_blob.extend(vec![0x22u8; 84]);
        expert_blob.extend(vec![0x33u8; 34]);
        std::fs::write(overlay.join("experts/mixed.bin"), expert_blob).unwrap();
        let router: Vec<u8> = (0..12u32)
            .flat_map(|value| (value as f32 + 0.25).to_le_bytes())
            .collect();
        std::fs::write(overlay.join("router.bin"), &router).unwrap();
        let manifest = format!(
            r#"{{
            "format":"memra-expert-overlay-v2","source_dir":"{}",
            "pruned_experts":{{"0":[1]}},
            "tensors":{{"blk.0.ffn_gate_exps.0.weight":{{
                "file":"experts/mixed.bin","offset":2,"qtype":"Q2_K","ne":[256,1],"bytes":84
            }},"blk.0.ffn_gate_exps.2.weight":{{
                "file":"experts/mixed.bin","offset":86,"qtype":"Q8_0","ne":[32,1],"bytes":34
            }},"blk.0.ffn_gate_inp.weight":{{
                "file":"router.bin","offset":0,"qtype":"F32","ne":[4,3],"bytes":48
            }}}}
        }}"#,
            base.display()
        );
        std::fs::write(overlay.join("manifest.json"), manifest).unwrap();

        let src = Hy3RepackSource::open(&overlay).unwrap();
        assert_eq!(src.config().moe.as_ref().unwrap().expert_count, 3);
        assert_eq!(src.active_experts(0), Some(&[true, false, true][..]));
        let expert = src.find("blk.0.ffn_gate_exps.0.weight").unwrap();
        assert_eq!(expert.ggml_type, GgmlType::Q2_K);
        assert_eq!(&*expert.bytes, &[0x22u8; 84]);
        let q8 = src.find("blk.0.ffn_gate_exps.2.weight").unwrap();
        assert_eq!(q8.ggml_type, GgmlType::Q8_0);
        assert_eq!(&*q8.bytes, &[0x33u8; 34]);
        let (map, off, len) = src
            .find_expert_mmap("blk.0.ffn_gate_exps.0.weight")
            .unwrap();
        assert_eq!((off, len), (2, 84));
        assert_eq!(&map[off..off + len], &[0x22u8; 84]);
        let disk = src
            .find_expert_disk("blk.0.ffn_gate_exps.0.weight")
            .unwrap();
        assert_eq!((disk.offset, disk.len), (2, 84));
        assert!(std::sync::Arc::ptr_eq(&disk.map, &map));
        let dense = src.find("blk.0.attn_norm.weight").unwrap();
        assert_eq!(&*dense.bytes, &[1, 2, 3, 4]);
        let healed_router = src.find("blk.0.ffn_gate_inp.weight").unwrap();
        assert_eq!(healed_router.ggml_type, GgmlType::F32);
        assert_eq!(healed_router.ne, vec![4, 3]);
        assert_eq!(&*healed_router.bytes, &router);

        // The opened inode is part of the extent, not borrowed from the loader source.
        drop(src);
        #[cfg(unix)]
        {
            use std::os::unix::fs::FileExt;
            let expert_path = overlay.join("experts/mixed.bin");
            std::fs::remove_file(&expert_path).unwrap();
            let mut replacement = vec![0x99u8, 0x98];
            replacement.extend(vec![0x77u8; 118]);
            std::fs::write(&expert_path, &replacement).unwrap();

            let mut direct = vec![0u8; disk.len];
            assert_eq!(
                disk.file.read_at(&mut direct, disk.offset).unwrap(),
                disk.len
            );
            assert_eq!(direct, vec![0x22u8; 84]);
            assert_eq!(&std::fs::read(&expert_path).unwrap()[2..86], &[0x77u8; 84]);
        }

        std::fs::remove_dir_all(&root).ok();
    }
}

#[cfg(test)]
mod hy3_repack_probe {
    use super::*;
    use std::io::{Read, Seek, SeekFrom};

    fn repack_dir() -> Option<&'static Path> {
        let dir = Path::new("/data/ai-ml/hf-models/hy3-reap50-q4k-memra");
        if dir.join("manifest.json").exists() {
            Some(dir)
        } else {
            None
        }
    }

    fn tsv_row(source_name: &str) -> Option<(String, String)> {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../research/hy3-reap50-tensor-inventory.tsv");
        let txt = std::fs::read_to_string(path).ok()?;
        for line in txt.lines().skip(1) {
            let mut cols = line.split('\t');
            let name = cols.next()?;
            let dtype = cols.next()?;
            let shape = cols.next()?;
            if name == source_name {
                return Some((dtype.to_string(), shape.to_string()));
            }
        }
        None
    }

    #[test]
    fn hy3_manifest_offset_roundtrip() {
        let Some(dir) = repack_dir() else {
            eprintln!("SKIP: Hy3 repack absent");
            return;
        };
        let src = Hy3RepackSource::open(dir).unwrap();
        assert_eq!(src.dir(), dir);
        assert_eq!(src.tensor_count(), 1278);
        let cfg = src.config();
        assert_eq!(cfg.arch, Arch::Hy3);
        assert_eq!(
            cfg.n_layer, 80,
            "REAP manifest stripped the appended MTP block"
        );
        assert_eq!(cfg.nextn_predict_layers, 0);
        assert_eq!(cfg.n_layer_total, 80);
        assert_eq!(cfg.moe.as_ref().unwrap().expert_count, 96);
        assert_eq!(cfg.moe.as_ref().unwrap().expert_used_count, 8);
        assert_eq!(cfg.hy3.as_ref().unwrap().first_k_dense_replace, 1);

        let name = "blk.1.ffn_gate_exps.weight";
        let v = src.find(name).unwrap();
        assert_eq!(v.ggml_type, GgmlType::Q4_K);
        assert_eq!(v.ne, vec![4096, 1536, 96]);
        let stride = src.expert_stride(name).unwrap();
        assert_eq!(stride, 3_538_944);
        assert_eq!(v.bytes.len(), stride * 96);

        let expert = 37usize;
        let offset = expert * stride;
        let mut f = std::fs::File::open(dir.join("experts/blk1-gate-96x1536x4096.q4k")).unwrap();
        f.seek(SeekFrom::Start(offset as u64)).unwrap();
        let mut buf = [0u8; 64];
        f.read_exact(&mut buf).unwrap();
        assert_eq!(&v.bytes[offset..offset + buf.len()], &buf);
    }

    #[test]
    fn hy3_inventory_dtype_shape_assertions() {
        let Some(dir) = repack_dir() else {
            eprintln!("SKIP: Hy3 repack absent");
            return;
        };
        let src = Hy3RepackSource::open(dir).unwrap();
        let cases = [
            (
                "model.layers.0.self_attn.q_proj.weight",
                "U32",
                "8192x512",
                "blk.0.attn_q.weight",
                GgmlType::Q4_K,
                vec![4096, 8192],
            ),
            (
                "model.layers.0.mlp.down_proj.weight",
                "U32",
                "4096x1664",
                "blk.0.ffn_down.weight",
                GgmlType::Q4_K,
                vec![13312, 4096],
            ),
            (
                "model.layers.1.mlp.router.gate.weight",
                "U32",
                "96x1024",
                "blk.1.ffn_gate_inp.weight",
                GgmlType::F32,
                vec![4096, 96],
            ),
            (
                "model.layers.1.mlp.router.expert_bias",
                "F32",
                "96",
                "blk.1.exp_probs_b.bias",
                GgmlType::F32,
                vec![96],
            ),
            (
                "model.layers.1.mlp.shared_mlp.gate_proj.weight",
                "U32",
                "1536x512",
                "blk.1.ffn_gate_shexp.weight",
                GgmlType::Q4_K,
                vec![4096, 1536],
            ),
            (
                "model.layers.1.mlp.switch_mlp.gate_proj.weight",
                "U32",
                "96x1536x512",
                "blk.1.ffn_gate_exps.weight",
                GgmlType::Q4_K,
                vec![4096, 1536, 96],
            ),
            (
                "model.norm.weight",
                "BF16",
                "4096",
                "output_norm.weight",
                GgmlType::F32,
                vec![4096],
            ),
        ];
        for (source_name, dtype, shape, ggml_name, qtype, ne) in cases {
            let row =
                tsv_row(source_name).unwrap_or_else(|| panic!("missing TSV row {source_name}"));
            assert_eq!(row, (dtype.to_string(), shape.to_string()), "{source_name}");
            let v = src
                .find(ggml_name)
                .unwrap_or_else(|| panic!("missing manifest tensor {ggml_name}"));
            assert_eq!(v.ggml_type, qtype, "{ggml_name}");
            assert_eq!(v.ne, ne, "{ggml_name}");
        }
    }

    #[test]
    fn hy3_load_plan_dry_run_no_cuda() {
        let Some(dir) = repack_dir() else {
            eprintln!("SKIP: Hy3 repack absent");
            return;
        };
        let src = Hy3RepackSource::open(dir).unwrap();
        let cfg = src.config();
        let hy3 = cfg.hy3.as_ref().unwrap();

        for name in ["token_embd.weight", "output_norm.weight", "output.weight"] {
            assert!(src.find(name).is_some(), "missing {name}");
        }
        for il in 0..cfg.n_layer {
            let p = |s: &str| format!("blk.{il}.{s}");
            for name in [
                p("attn_norm.weight"),
                p("attn_q.weight"),
                p("attn_k.weight"),
                p("attn_v.weight"),
                p("attn_output.weight"),
                p("attn_q_norm.weight"),
                p("attn_k_norm.weight"),
                p("ffn_norm.weight"),
            ] {
                assert!(src.find(&name).is_some(), "missing load-plan tensor {name}");
            }
            if il < hy3.first_k_dense_replace {
                for name in [
                    p("ffn_gate.weight"),
                    p("ffn_up.weight"),
                    p("ffn_down.weight"),
                ] {
                    let v = src
                        .find(&name)
                        .unwrap_or_else(|| panic!("missing dense tensor {name}"));
                    assert_eq!(v.ggml_type, GgmlType::Q4_K, "{name}");
                    assert_eq!(v.ne.len(), 2, "{name}");
                }
            } else {
                for (name, qtype, rank) in [
                    (p("ffn_gate_inp.weight"), GgmlType::F32, 2usize),
                    (p("exp_probs_b.bias"), GgmlType::F32, 1),
                    (p("ffn_gate_shexp.weight"), GgmlType::Q4_K, 2),
                    (p("ffn_up_shexp.weight"), GgmlType::Q4_K, 2),
                    (p("ffn_down_shexp.weight"), GgmlType::Q4_K, 2),
                    (p("ffn_gate_exps.weight"), GgmlType::Q4_K, 3),
                    (p("ffn_up_exps.weight"), GgmlType::Q4_K, 3),
                    (p("ffn_down_exps.weight"), GgmlType::Q4_K, 3),
                ] {
                    let v = src
                        .find(&name)
                        .unwrap_or_else(|| panic!("missing MoE tensor {name}"));
                    assert_eq!(v.ggml_type, qtype, "{name}");
                    assert_eq!(v.ne.len(), rank, "{name}");
                }
            }
        }
    }
}

#[cfg(test)]
mod nv27b_probe {
    use super::*;

    /// NVIDIA-official 27B (mixed FP8/NVFP4 ckpt) dtype-routing regression: every BIG tensor class
    /// must surface memory-bounded (Q8_0 / NVFP4), NEVER F32 (461MB/linear-layer x 48 = the 22GB
    /// load-tail OOM, 2026-07-07). Skips when the checkpoint is absent.
    #[test]
    fn nvidia_27b_dtype_routing() {
        let dir = std::path::Path::new("/data/ai-ml/hf-models/nvidia-qwen36-27b-nvfp4");
        if !dir.join("model.safetensors.index.json").exists() {
            eprintln!("SKIP: ckpt absent");
            return;
        }
        let src = SafetensorsSource::open(dir).unwrap();
        let ty = |n: &str| {
            src.find(n)
                .unwrap_or_else(|| panic!("{n} unresolved"))
                .ggml_type
        };
        // linear_attn F8 projections (Transform arm, V-reordered) -> Q8_0, never F32.
        assert_eq!(ty("blk.0.attn_qkv.weight"), GgmlType::Q8_0);
        assert_eq!(ty("blk.0.attn_gate.weight"), GgmlType::Q8_0);
        assert_eq!(ty("blk.0.ssm_out.weight"), GgmlType::Q8_0);
        // full-attn F8 projections (Plain arm) -> Q8_0.
        assert_eq!(ty("blk.3.attn_q.weight"), GgmlType::Q8_0);
        assert_eq!(ty("blk.3.attn_output.weight"), GgmlType::Q8_0);
        // NVFP4 mlp + lm_head keep the native direct-import path.
        assert!(src.find_nvfp4_native("blk.0.ffn_gate.weight").is_some());
        assert!(src.find_nvfp4_native("output.weight").is_some());
        // ssm_alpha/beta (<- BF16 in_proj_a/b) are MATMUL-class: the Transform-arm gate re-encodes
        // them Q8_0 so mixer_in_q8_1_fast holds on every linear layer (the loader law; leaving
        // them Float unfused EVERY linear-attn mixer onto cuBLAS f32 GEMV pairs).
        assert_eq!(ty("blk.0.ssm_alpha.weight"), GgmlType::Q8_0);
        assert_eq!(ty("blk.0.ssm_beta.weight"), GgmlType::Q8_0);
        // norm-class SSM tensors stay F32 (engine consumes them via float_data()).
        assert_eq!(ty("blk.0.ssm_conv1d.weight"), GgmlType::F32);
        assert_eq!(ty("blk.0.ssm_a"), GgmlType::F32);
    }

    /// MTP head mapping: blk.64.* (GGUF NextN numbering) resolves into the HF `mtp.*` namespace
    /// with the exact ne the engine loads from the GGUF twin (blk.64 census 2026-07-07).
    /// nextn_predict_layers must come from `mtp_num_hidden_layers` (the 27B HF key).
    #[test]
    fn nvidia_27b_mtp_mapping() {
        let dir = std::path::Path::new("/data/ai-ml/hf-models/nvidia-qwen36-27b-nvfp4");
        if !dir.join("model.safetensors.index.json").exists() {
            eprintln!("SKIP: ckpt absent");
            return;
        }
        let src = SafetensorsSource::open(dir).unwrap();
        let cfg = src.config();
        assert_eq!(
            cfg.nextn_predict_layers, 1,
            "mtp_num_hidden_layers -> nextn"
        );
        assert_eq!(
            cfg.n_layer, 65,
            "n_layer includes the MTP block (GGUF block_count convention)"
        );
        let v = |n: &str| src.find(n).unwrap_or_else(|| panic!("{n} unresolved"));
        // glue (GGUF-twin ne reference: enorm/hnorm/shared_head_norm [5120], eh_proj [10240,5120])
        assert_eq!(v("blk.64.nextn.enorm.weight").ne, vec![5120]);
        assert_eq!(v("blk.64.nextn.hnorm.weight").ne, vec![5120]);
        assert_eq!(v("blk.64.nextn.eh_proj.weight").ne, vec![10240, 5120]);
        assert_eq!(v("blk.64.nextn.shared_head_norm.weight").ne, vec![5120]);
        assert!(
            src.find("blk.64.nextn.shared_head.weight").is_none(),
            "head reuses lm_head"
        );
        // block tensors (full-attn block: q [5120,12288], k/v [5120,1024], o [6144,5120])
        assert_eq!(v("blk.64.attn_q.weight").ne, vec![5120, 12288]);
        assert_eq!(v("blk.64.attn_k.weight").ne, vec![5120, 1024]);
        assert_eq!(v("blk.64.attn_v.weight").ne, vec![5120, 1024]);
        assert_eq!(v("blk.64.attn_output.weight").ne, vec![6144, 5120]);
        assert_eq!(v("blk.64.attn_q_norm.weight").ne, vec![256]);
        assert_eq!(v("blk.64.ffn_gate.weight").ne, vec![5120, 17408]);
        assert_eq!(v("blk.64.ffn_down.weight").ne, vec![17408, 5120]);
        assert_eq!(v("blk.64.attn_norm.weight").ne, vec![5120]);
        assert_eq!(v("blk.64.post_attention_norm.weight").ne, vec![5120]);
        // big BF16 mtp matrices -> Q8_0 (draft class), norms -> F32 with the +1 fold.
        assert_eq!(v("blk.64.attn_q.weight").ggml_type, GgmlType::Q8_0);
        assert_eq!(v("blk.64.nextn.eh_proj.weight").ggml_type, GgmlType::Q8_0);
        let en = v("blk.64.nextn.enorm.weight");
        assert_eq!(en.ggml_type, GgmlType::F32);
        // +1 fold check vs GGUF twin blk.64.nextn.enorm first value 0.4375 (raw bf16 -0.5625).
        let first = f32::from_le_bytes(en.bytes[..4].try_into().unwrap());
        assert!((first - 0.4375).abs() < 1e-3, "enorm +1 fold: got {first}");
    }
}

#[cfg(test)]
mod m3_probe {
    use super::*;

    /// MiniMax-M3 dtype-routing regression (loadersweep 2026-07-08): the REAP50 ckpt ships lm_head
    /// as the ONLY BF16 Linear (everything else NVFP4). Before the lm_head gate it surfaced F32 ->
    /// a 4.9GB GpuTensor::Float whose per-token decode matmul rode cuBLAS f32 GEMV (the loader-law
    /// Float-poison trap, occurrence #4). Must surface Q8_0. The router (ffn_gate_inp) is the
    /// AUDITED Float exception: selection-sensitive top-k, llama.cpp keeps every router F32, and it
    /// sits on no all-or-nothing predicate — it must STAY F32. Skips when the ckpt is absent.
    #[test]
    fn minimax_m3_lm_head_q8() {
        let dir = std::path::Path::new("/data/ai-ml/hf-models/minimax-m3-nvfp4-reap50");
        if !dir.join("model.safetensors.index.json").exists() {
            eprintln!("SKIP: ckpt absent");
            return;
        }
        let src = SafetensorsSource::open(dir).unwrap();
        // router: deliberately Float (audited exception — see model.rs float_2d_audited).
        let router = src
            .find("blk.3.ffn_gate_inp.weight")
            .expect("M3 router unresolved");
        assert_eq!(router.ggml_type, GgmlType::F32);
        assert_eq!(router.ne, vec![6144, 64]);
        // lm_head: BF16 on disk -> must re-encode Q8_0 (matmul-class, hot every decoded token).
        let head = src.find("output.weight").expect("M3 lm_head unresolved");
        assert_eq!(
            head.ggml_type,
            GgmlType::Q8_0,
            "BF16 lm_head must surface Q8_0, not Float"
        );
        assert_eq!(head.ne, vec![6144, 200064]);
    }
}

#[cfg(test)]
mod nv27b_twin_parity {
    use super::*;

    /// Full-tensor numeric parity vs the GGUF twin (converted from the same NVIDIA ckpt):
    /// every F32-surfaced tensor (norms incl +1 folds, ssm_a -exp, dt_bias, conv1d) must match
    /// the twin's dequant bit-for-bit (both go bf16 -> f32 exactly). Skips when either is absent.
    #[test]
    fn nvidia_27b_vs_gguf_twin_f32_parity() {
        let st_dir = std::path::Path::new("/data/ai-ml/hf-models/nvidia-qwen36-27b-nvfp4");
        let twin = std::path::Path::new(
            "/data/ai-ml/hf-models/qwen36-27b-nvfp4-mtp/Qwen3.6-27B-NVFP4-Q4_K_M-mtp.gguf",
        );
        if !st_dir.join("model.safetensors.index.json").exists() || !twin.exists() {
            eprintln!("SKIP: ckpt/twin absent");
            return;
        }
        let src = SafetensorsSource::open(st_dir).unwrap();
        let g = crate::GgufFile::open(twin.to_str().unwrap()).unwrap();

        // GEOMETRY parity comes first: identical bytes under a different rotary width is still a
        // different model. The twin ships `rope.dimension_count = 64`; the ST config ships
        // `partial_rotary_factor: 0.25` over `head_dim: 256`. Reading only a literal `rotary_dim`
        // on the HF side gave the ST route n_rot = 256 — full rope over the 192 dims that must
        // pass through — while every tensor compare below still passed bit-for-bit.
        let st_cfg = src.config();
        let twin_cfg = crate::config::ModelConfig::from_gguf(&g);
        assert_eq!(st_cfg.head_dim_k, twin_cfg.head_dim_k, "head_dim_k");
        assert_eq!(
            st_cfg.rope_dim_count, twin_cfg.rope_dim_count,
            "rotary width must match the twin (st {} vs gguf {})",
            st_cfg.rope_dim_count, twin_cfg.rope_dim_count
        );
        for il in [3u32, 7, 63] {
            let st_row = st_cfg.full_attention_geometry_at(il);
            let twin_row = twin_cfg.full_attention_geometry_at(il);
            assert_eq!(st_row.n_rot, twin_row.n_rot, "blk.{il} n_rot");
            assert_eq!(st_row.rope_base, twin_row.rope_base, "blk.{il} rope_base");
        }

        // NOTE: trunk pre-FFN norm resolves via the loader's ffn_norm fallback (the twin names it
        // post_attention_norm) — compare ST ffn_norm vs twin post_attention_norm below.
        let names = [
            "blk.0.attn_norm.weight",
            "blk.0.ssm_norm.weight",
            "blk.0.ssm_a",
            "blk.0.ssm_dt.bias",
            "blk.0.ssm_conv1d.weight",
            "blk.3.attn_q_norm.weight",
            "blk.3.attn_k_norm.weight",
            "blk.64.attn_norm.weight",
            "blk.64.nextn.enorm.weight",
            "blk.64.nextn.hnorm.weight",
            "blk.64.nextn.shared_head_norm.weight",
            "output_norm.weight",
        ];
        // (ST ggml name, twin ggml name) pairs where the two sources use different aliases.
        let pairs = [("blk.0.ffn_norm.weight", "blk.0.post_attention_norm.weight")];
        for (st_name, twin_name) in names.iter().map(|&n| (n, n)).chain(pairs) {
            let name = st_name;
            let sv = src
                .find(st_name)
                .unwrap_or_else(|| panic!("{st_name}: ST unresolved"));
            let gt = g
                .find(twin_name)
                .unwrap_or_else(|| panic!("{twin_name}: not in twin"));
            assert_eq!(sv.ne, gt.ne, "{name}: ne mismatch");
            let n: u64 = sv.ne.iter().product();
            let a = crate::dequant::dequantize(sv.ggml_type, &sv.bytes, n as usize);
            let b = crate::dequant::dequantize(gt.ggml_type, g.tensor_data(gt), n as usize);
            let md = a
                .iter()
                .zip(&b)
                .map(|(x, y)| (x - y).abs())
                .fold(0f32, f32::max);
            // ssm_a = -exp(A_log): Rust libm expf vs the converter's numpy exp differ by ULPs.
            // Everything else (renames, +1 folds, conv1d squeeze/reorder) must be bit-exact.
            let tol = if name.ends_with("ssm_a") { 1e-7 } else { 0.0 };
            assert!(md <= tol, "{name}: maxdiff {md} > tol {tol}");
            eprintln!("{name:40} n={n:6} maxdiff={md:.1e}");
        }
    }
}

#[cfg(test)]
mod compressed_tensors_roundtrip {
    use super::*;

    /// CPU roundtrip: build a synthetic compressed-tensors safetensors (weight_packed U8 +
    /// weight_scale F8_E4M3 + weight_global_scale F32 DIVISOR), open via SafetensorsSource, and
    /// verify: (1) nvfp4_quant returns macro_s = 1/global_scale, (2) the .scale sibling lookup
    /// returns the INVERTED value (multiplier, not divisor), (3) the repacked GGUF blocks
    /// dequantize to the correct magnitude range.
    #[test]
    fn ct_global_scale_inversion() {
        let dir = std::env::temp_dir().join(format!("memra_ct_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();

        // Shape: one layer, out_f=2, in_f=64 (minimum for NVFP4 64-elem blocks)
        let out_f: usize = 2;
        let in_f: usize = 64;
        let packed_bytes = out_f * in_f / 2; // 64 bytes (U8 [2,32])
        let scale_bytes = out_f * in_f / 16; // 8 bytes (F8_E4M3 [2,4])
        let global_scale: f32 = 9408.0; // typical DIVISOR value from real checkpoint

        // Fill packed with non-zero e2m1 codes (code=3 in each nibble -> magnitude 1.5)
        let weight_packed = vec![0x33u8; packed_bytes]; // code 3 in both nibbles
        // Fill scales with a known UE4M3 value (0x38 = exp=7 man=0 -> (1.0+0)*2^0 = 1.0 raw, *0.5 = 0.5)
        let weight_scale = vec![0x38u8; scale_bytes];
        // global_scale as F32 scalar
        let gs_bytes = global_scale.to_le_bytes();

        // Also need a BF16 placeholder for the ".weight" tensor (compressed-tensors has both).
        // We use shape [2,64] BF16 (256 bytes) but it should be IGNORED by the NVFP4 path.
        let bf16_weight = vec![0u8; out_f * in_f * 2]; // zeros

        // Build safetensors file:
        // tensors: model.layers.0.self_attn.q_proj.weight (BF16 [2,64])
        //          model.layers.0.self_attn.q_proj.weight_packed (U8 [2,32])
        //          model.layers.0.self_attn.q_proj.weight_scale (F8_E4M3 [2,4])
        //          model.layers.0.self_attn.q_proj.weight_global_scale (F32 [1])
        let data_len =
            bf16_weight.len() + weight_packed.len() + weight_scale.len() + gs_bytes.len();
        let mut off = 0usize;
        let bf16_start = off;
        off += bf16_weight.len();
        let packed_start = off;
        off += weight_packed.len();
        let scale_start = off;
        off += weight_scale.len();
        let gs_start = off;
        off += gs_bytes.len();
        assert_eq!(off, data_len);

        let json = format!(
            concat!(
                "{{",
                "\"model.layers.0.self_attn.q_proj.weight\":{{\"dtype\":\"BF16\",\"shape\":[2,64],\"data_offsets\":[{},{}]}},",
                "\"model.layers.0.self_attn.q_proj.weight_packed\":{{\"dtype\":\"U8\",\"shape\":[2,32],\"data_offsets\":[{},{}]}},",
                "\"model.layers.0.self_attn.q_proj.weight_scale\":{{\"dtype\":\"F8_E4M3\",\"shape\":[2,4],\"data_offsets\":[{},{}]}},",
                "\"model.layers.0.self_attn.q_proj.weight_global_scale\":{{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[{},{}]}}",
                "}}"
            ),
            bf16_start,
            bf16_start + bf16_weight.len(),
            packed_start,
            packed_start + weight_packed.len(),
            scale_start,
            scale_start + weight_scale.len(),
            gs_start,
            gs_start + gs_bytes.len(),
        );

        let mut buf = Vec::new();
        buf.extend_from_slice(&(json.len() as u64).to_le_bytes());
        buf.extend_from_slice(json.as_bytes());
        buf.extend_from_slice(&bf16_weight);
        buf.extend_from_slice(&weight_packed);
        buf.extend_from_slice(&weight_scale);
        buf.extend_from_slice(&gs_bytes);
        std::fs::write(dir.join("model.safetensors"), &buf).unwrap();

        let cfg_json = r#"{"model_type":"qwen3","num_hidden_layers":1,"hidden_size":64,"num_attention_heads":2,"intermediate_size":128,"vocab_size":10,"max_position_embeddings":128,"num_key_value_heads":2,"head_dim":32}"#;
        std::fs::write(dir.join("config.json"), cfg_json).unwrap();

        let src = SafetensorsSource::open(&dir).unwrap();

        // Test 1: nvfp4_quant returns correct inverted macro_s
        let hf_name = "model.layers.0.self_attn.q_proj.weight";
        let (o, i, _packed, _scales, macro_s) = src
            .nvfp4_quant(hf_name)
            .expect("nvfp4_quant must detect compressed-tensors format");
        assert_eq!(o, out_f);
        assert_eq!(i, in_f);
        let expected_macro = 1.0 / global_scale;
        assert!(
            (macro_s - expected_macro).abs() < 1e-10,
            "macro_s should be 1/global_scale = {expected_macro}, got {macro_s}"
        );

        // Test 2: the .scale sibling lookup returns the INVERTED value
        let scale_view = src
            .find("blk.0.attn_q.scale")
            .expect(".scale sibling must resolve for NVFP4 tensors");
        assert_eq!(scale_view.ggml_type, GgmlType::F32);
        assert_eq!(scale_view.ne, vec![1]);
        let returned_scale = f32::from_le_bytes(scale_view.bytes[..4].try_into().unwrap());
        assert!(
            (returned_scale - expected_macro).abs() < 1e-10,
            ".scale lookup should return 1/global_scale = {expected_macro}, got {returned_scale}"
        );

        // Test 3: the returned bytes are Owned (not borrowed raw divisor)
        assert!(
            matches!(scale_view.bytes, std::borrow::Cow::Owned(_)),
            ".scale for compressed-tensors must be Cow::Owned (inverted value)"
        );

        drop(scale_view);
        drop(src);
        let bad_json = json.replace("weight_global_scale", "weight_global_scale_missing");
        let mut bad = Vec::new();
        bad.extend_from_slice(&(bad_json.len() as u64).to_le_bytes());
        bad.extend_from_slice(bad_json.as_bytes());
        bad.extend_from_slice(&bf16_weight);
        bad.extend_from_slice(&weight_packed);
        bad.extend_from_slice(&weight_scale);
        bad.extend_from_slice(&gs_bytes);
        std::fs::write(dir.join("model.safetensors"), bad).unwrap();
        let bad_src = SafetensorsSource::open(&dir).unwrap();
        assert!(
            bad_src.nvfp4_quant(hf_name).is_none(),
            "compressed-tensors NVFP4 without weight_global_scale must fail closed"
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    /// Verify modelopt arm still borrows the raw weight_scale_2 (no inversion).
    #[test]
    fn modelopt_scale2_direct_borrow() {
        let dir = std::env::temp_dir().join(format!("memra_mo_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();

        let out_f: usize = 2;
        let in_f: usize = 64;
        let packed_bytes = out_f * in_f / 2;
        let scale_bytes = out_f * in_f / 16;
        let scale_2: f32 = 0.000106; // typical MULTIPLIER

        let weight = vec![0x33u8; packed_bytes];
        let wscale = vec![0x38u8; scale_bytes];
        let s2_bytes = scale_2.to_le_bytes();

        let mut off = 0usize;
        let w_start = off;
        off += weight.len();
        let s_start = off;
        off += wscale.len();
        let s2_start = off;

        let json = format!(
            concat!(
                "{{",
                "\"model.layers.0.self_attn.q_proj.weight\":{{\"dtype\":\"U8\",\"shape\":[2,32],\"data_offsets\":[{},{}]}},",
                "\"model.layers.0.self_attn.q_proj.weight_scale\":{{\"dtype\":\"F8_E4M3\",\"shape\":[2,4],\"data_offsets\":[{},{}]}},",
                "\"model.layers.0.self_attn.q_proj.weight_scale_2\":{{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[{},{}]}}",
                "}}"
            ),
            w_start,
            w_start + weight.len(),
            s_start,
            s_start + wscale.len(),
            s2_start,
            s2_start + s2_bytes.len(),
        );

        let mut buf = Vec::new();
        buf.extend_from_slice(&(json.len() as u64).to_le_bytes());
        buf.extend_from_slice(json.as_bytes());
        buf.extend_from_slice(&weight);
        buf.extend_from_slice(&wscale);
        buf.extend_from_slice(&s2_bytes);
        std::fs::write(dir.join("model.safetensors"), &buf).unwrap();

        let cfg_json = r#"{"model_type":"qwen3","num_hidden_layers":1,"hidden_size":64,"num_attention_heads":2,"intermediate_size":128,"vocab_size":10,"max_position_embeddings":128,"num_key_value_heads":2,"head_dim":32}"#;
        std::fs::write(dir.join("config.json"), cfg_json).unwrap();

        let src = SafetensorsSource::open(&dir).unwrap();

        // nvfp4_quant: macro_s should be the raw scale_2 value (direct multiplier)
        let (_, _, _, _, macro_s) = src
            .nvfp4_quant("model.layers.0.self_attn.q_proj.weight")
            .expect("nvfp4_quant must detect modelopt format");
        assert!(
            (macro_s - scale_2).abs() < 1e-10,
            "modelopt macro_s should be scale_2 directly = {scale_2}, got {macro_s}"
        );

        // .scale sibling: borrowed directly, value == scale_2
        let sv = src
            .find("blk.0.attn_q.scale")
            .expect(".scale sibling must resolve for modelopt NVFP4");
        let v = f32::from_le_bytes(sv.bytes[..4].try_into().unwrap());
        assert!(
            (v - scale_2).abs() < 1e-10,
            "modelopt .scale should be raw scale_2 = {scale_2}, got {v}"
        );
        assert!(
            matches!(sv.bytes, std::borrow::Cow::Borrowed(_)),
            "modelopt .scale should be Cow::Borrowed (zero-copy)"
        );

        drop(src);
        let bad_json = json.replace("weight_scale_2", "weight_scale_x");
        let mut bad = Vec::new();
        bad.extend_from_slice(&(bad_json.len() as u64).to_le_bytes());
        bad.extend_from_slice(bad_json.as_bytes());
        bad.extend_from_slice(&weight);
        bad.extend_from_slice(&wscale);
        bad.extend_from_slice(&s2_bytes);
        std::fs::write(dir.join("model.safetensors"), bad).unwrap();
        let bad_src = SafetensorsSource::open(&dir).unwrap();
        assert!(
            bad_src
                .nvfp4_quant("model.layers.0.self_attn.q_proj.weight")
                .is_none(),
            "modelopt NVFP4 without weight_scale_2 must fail closed"
        );

        std::fs::remove_dir_all(&dir).ok();
    }
}

#[cfg(test)]
mod f8_block128 {
    use super::*;

    /// Deterministic pseudo-random e4m3 byte, NaN codes remapped (0x7F/0xFF -> exp field 0xE).
    fn e4m3_byte(i: usize) -> u8 {
        let mut b = ((i.wrapping_mul(2654435761) ^ 0x9E37_79B9) >> 9) as u8;
        if b & 0x7F == 0x7F {
            b &= 0xF7;
        }
        b
    }

    /// f8_scales granularity detection: scalar, per-row, block-128 (F32 + BF16 payloads),
    /// and rejection of anything else.
    #[test]
    fn scale_shape_detection() {
        use crate::safetensors::StInfo;
        let (out_f, in_f) = (256usize, 384usize); // block grid [2, 3]
        let info = |dtype: &str, shape: Vec<u64>| StInfo {
            dtype: dtype.to_string(),
            shape,
            data_offsets: [0, 0],
        };
        let f32b = |v: &[f32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();

        // per-tensor scalar
        let s = f8_scales(&info("F32", vec![1]), &f32b(&[0.5]), out_f, in_f).unwrap();
        assert!(matches!(s, F8Scales::PerTensor(v) if v == 0.5));

        // per-row [out, 1]
        let rows: Vec<f32> = (0..out_f).map(|i| 1.0 + i as f32).collect();
        let s = f8_scales(
            &info("F32", vec![out_f as u64, 1]),
            &f32b(&rows),
            out_f,
            in_f,
        )
        .unwrap();
        assert!(matches!(&s, F8Scales::PerRow(v) if v.len() == out_f && v[255] == 256.0));

        // block-128 [2, 3] in F32
        let grid: Vec<f32> = (0..6).map(|i| 0.25 * (i + 1) as f32).collect();
        let s = f8_scales(&info("F32", vec![2, 3]), &f32b(&grid), out_f, in_f).unwrap();
        match &s {
            F8Scales::Block128 { scales, cols } => {
                assert_eq!(*cols, 3);
                assert_eq!(scales, &grid);
            }
            other => panic!(
                "expected Block128, got {}",
                match other {
                    F8Scales::PerTensor(_) => "PerTensor",
                    F8Scales::PerRow(_) => "PerRow",
                    _ => unreachable!(),
                }
            ),
        }
        // at(): element (o=130, e=200) -> block (1, 1) -> grid[1*3+1]
        assert_eq!(s.at(130, 200), grid[4]);
        assert_eq!(s.at(0, 0), grid[0]);
        assert_eq!(s.at(255, 383), grid[5]);

        // block-128 in BF16 (the Qwen3.6-27B on-disk encoding): bf16(0.5) = 0x3F00
        let bf: Vec<u8> = std::iter::repeat_n(0x3F00u16.to_le_bytes(), 6)
            .flatten()
            .collect();
        let s = f8_scales(&info("BF16", vec![2, 3]), &bf, out_f, in_f).unwrap();
        assert!(
            matches!(s, F8Scales::Block128 { ref scales, cols: 3 } if scales.iter().all(|&v| v == 0.5))
        );

        // ceil grid: out=200, in=300 -> [2, 3] must also be accepted
        assert!(f8_scales(&info("F32", vec![2, 3]), &f32b(&grid), 200, 300).is_some());

        // wrong grid shape -> None (falls to the loud raw-path assert downstream)
        assert!(f8_scales(&info("F32", vec![3, 2]), &f32b(&grid), out_f, in_f).is_none());
        // non-positive scale -> None
        assert!(f8_scales(&info("F32", vec![1]), &f32b(&[0.0]), out_f, in_f).is_none());
        // transposed count that coincidentally matches nothing
        assert!(f8_scales(&info("F32", vec![5]), &f32b(&[1.0; 5]), out_f, in_f).is_none());
    }

    /// f8_deq_f32 with block-128 scales is BIT-EXACT against the naive per-element reference
    /// (including non-multiple-of-128 dims where the last blocks are ragged).
    #[test]
    fn block128_dequant_bit_exact() {
        for (out_f, in_f) in [(256usize, 384usize), (200, 300)] {
            let (rows, cols) = (out_f.div_ceil(128), in_f.div_ceil(128));
            let codes: Vec<u8> = (0..out_f * in_f).map(e4m3_byte).collect();
            let grid: Vec<f32> = (0..rows * cols).map(|i| 0.03125 * (i + 1) as f32).collect();
            let scales = F8Scales::Block128 {
                scales: grid.clone(),
                cols,
            };

            let got = f8_deq_f32(&codes, out_f, in_f, &scales);

            // independent reference: no chunk-hoisting, straight (o,e) -> block indexing
            for o in 0..out_f {
                for e in 0..in_f {
                    let s = grid[(o / 128) * cols + (e / 128)];
                    let expect = crate::nvfp4_repack::fp8_e4m3_to_f32(codes[o * in_f + e]) * s;
                    let v = got[o * in_f + e];
                    assert!(
                        v.to_bits() == expect.to_bits(),
                        "({out_f}x{in_f}) mismatch at [{o}][{e}]: got {v} want {expect}"
                    );
                }
            }
        }
    }

    /// Build a synthetic Qwen-official-style FP8 checkpoint (F8_E4M3 weight + BF16
    /// `weight_scale_inv` [out/128, in/128]) and load it end-to-end through `find`:
    /// the loader must recognize the block grid, dequant, and re-encode to Q8_0 whose
    /// bytes are BIT-IDENTICAL to `f32_to_q8_0(reference dequant)`.
    #[test]
    fn qwen_official_fp8_find_q8_0_roundtrip() {
        let dir = std::env::temp_dir().join(format!("memra_f8blk_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();

        let (out_f, in_f) = (256usize, 256usize); // grid [2, 2]
        let codes: Vec<u8> = (0..out_f * in_f).map(e4m3_byte).collect();
        // BF16 scale grid: pick exactly-representable values so bf16->f32 is lossless here.
        let grid_f32 = [0.5f32, 0.25, 2.0, 1.5];
        let grid_bf16: Vec<u8> = grid_f32
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect();

        let w_len = codes.len();
        let s_start = w_len;
        let s_len = grid_bf16.len();
        let json = format!(
            concat!(
                "{{",
                "\"model.layers.0.self_attn.q_proj.weight\":{{\"dtype\":\"F8_E4M3\",\"shape\":[{o},{i}],\"data_offsets\":[0,{wl}]}},",
                "\"model.layers.0.self_attn.q_proj.weight_scale_inv\":{{\"dtype\":\"BF16\",\"shape\":[2,2],\"data_offsets\":[{ss},{se}]}}",
                "}}"
            ),
            o = out_f,
            i = in_f,
            wl = w_len,
            ss = s_start,
            se = s_start + s_len,
        );
        let mut buf = Vec::new();
        buf.extend_from_slice(&(json.len() as u64).to_le_bytes());
        buf.extend_from_slice(json.as_bytes());
        buf.extend_from_slice(&codes);
        buf.extend_from_slice(&grid_bf16);
        std::fs::write(dir.join("model.safetensors"), &buf).unwrap();
        std::fs::write(
            dir.join("config.json"),
            r#"{"model_type":"qwen3","num_hidden_layers":1,"hidden_size":256,"num_attention_heads":2,"num_key_value_heads":2,"head_dim":128,"intermediate_size":512,"vocab_size":16,"max_position_embeddings":128}"#,
        ).unwrap();

        let src = SafetensorsSource::open(&dir).unwrap();
        let v = src
            .find("blk.0.attn_q.weight")
            .expect("block-128 FP8 weight must load (was the st_dtype_to_ggml panic)");
        assert_eq!(v.ggml_type, GgmlType::Q8_0);
        assert_eq!(v.ne, vec![in_f as u64, out_f as u64]);

        // reference: dequant with the block grid, then the SAME Q8_0 encoder
        let mut reference = vec![0f32; out_f * in_f];
        for o in 0..out_f {
            for e in 0..in_f {
                let s = grid_f32[(o / 128) * 2 + (e / 128)];
                reference[o * in_f + e] =
                    crate::nvfp4_repack::fp8_e4m3_to_f32(codes[o * in_f + e]) * s;
            }
        }
        let expect = crate::nvfp4_repack::f32_to_q8_0(&reference);
        assert_eq!(
            v.bytes.as_ref(),
            expect.as_slice(),
            "Q8_0 bytes must be bit-identical"
        );

        // deq_f32 (the Transform-arm feeder) resolves the same block grid
        let (deq, ne) = src
            .deq_f32("model.layers.0.self_attn.q_proj.weight")
            .expect("deq_f32 must handle block-128 scales");
        assert_eq!(ne, vec![in_f as u64, out_f as u64]);
        assert!(
            deq.iter()
                .zip(&reference)
                .all(|(a, b)| a.to_bits() == b.to_bits())
        );

        // `has` must be true and not panic through the raw path
        assert!(src.has("blk.0.attn_q.weight"));

        std::fs::remove_dir_all(&dir).ok();
    }

    /// B1b loader side: find_fp8_native on a block-128 checkpoint must return the raw e4m3
    /// bytes BORROWED (zero copy) + the decoded block grid in checkpoint order, scale == 1.0.
    #[test]
    fn fp8_native_carries_block_grid() {
        let dir = std::env::temp_dir().join(format!("memra_f8nat_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();

        let (out_f, in_f) = (256usize, 256usize); // grid [2, 2]
        let codes: Vec<u8> = (0..out_f * in_f).map(e4m3_byte).collect();
        let grid_f32 = [0.5f32, 0.25, 2.0, 1.5];
        let grid_bf16: Vec<u8> = grid_f32
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        let w_len = codes.len();
        let json = format!(
            concat!(
                "{{",
                "\"model.layers.0.self_attn.q_proj.weight\":{{\"dtype\":\"F8_E4M3\",\"shape\":[{o},{i}],\"data_offsets\":[0,{wl}]}},",
                "\"model.layers.0.self_attn.q_proj.weight_scale_inv\":{{\"dtype\":\"BF16\",\"shape\":[2,2],\"data_offsets\":[{wl},{se}]}}",
                "}}"
            ),
            o = out_f,
            i = in_f,
            wl = w_len,
            se = w_len + grid_bf16.len(),
        );
        let mut buf = Vec::new();
        buf.extend_from_slice(&(json.len() as u64).to_le_bytes());
        buf.extend_from_slice(json.as_bytes());
        buf.extend_from_slice(&codes);
        buf.extend_from_slice(&grid_bf16);
        std::fs::write(dir.join("model.safetensors"), &buf).unwrap();
        std::fs::write(
            dir.join("config.json"),
            r#"{"model_type":"qwen3","num_hidden_layers":1,"hidden_size":256,"num_attention_heads":2,"num_key_value_heads":2,"head_dim":128,"intermediate_size":512,"vocab_size":16,"max_position_embeddings":128}"#,
        ).unwrap();

        let src = SafetensorsSource::open(&dir).unwrap();
        let f8 = src
            .find_fp8_native("blk.0.attn_q.weight")
            .expect("block-128 FP8 must surface through find_fp8_native");
        assert_eq!((f8.out_f, f8.in_f), (out_f, in_f));
        assert_eq!(
            f8.scale, 1.0,
            "block-128 class carries scale in blk, scalar must be 1.0"
        );
        assert!(
            matches!(f8.bytes, std::borrow::Cow::Borrowed(_)),
            "plain arm is zero-copy"
        );
        assert_eq!(
            f8.bytes.as_ref(),
            codes.as_slice(),
            "raw e4m3 codes verbatim"
        );
        let blk = f8.blk.expect("block grid must ride along");
        assert_eq!((blk.rows, blk.cols), (2, 2));
        assert_eq!(
            blk.scales, grid_f32,
            "grid decoded to f32 in checkpoint order"
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    /// The existing per-tensor F32 `weight_scale` class must still parse identically
    /// (regression guard for the f8_row_scales -> f8_scales refactor).
    #[test]
    fn per_tensor_scale_class_unchanged() {
        let dir = std::env::temp_dir().join(format!("memra_f8pt_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();

        let (out_f, in_f) = (64usize, 64usize);
        let codes: Vec<u8> = (0..out_f * in_f).map(e4m3_byte).collect();
        let scale = 0.125f32;
        let w_len = codes.len();
        let json = format!(
            concat!(
                "{{",
                "\"model.layers.0.self_attn.q_proj.weight\":{{\"dtype\":\"F8_E4M3\",\"shape\":[{o},{i}],\"data_offsets\":[0,{wl}]}},",
                "\"model.layers.0.self_attn.q_proj.weight_scale\":{{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[{wl},{se}]}}",
                "}}"
            ),
            o = out_f,
            i = in_f,
            wl = w_len,
            se = w_len + 4,
        );
        let mut buf = Vec::new();
        buf.extend_from_slice(&(json.len() as u64).to_le_bytes());
        buf.extend_from_slice(json.as_bytes());
        buf.extend_from_slice(&codes);
        buf.extend_from_slice(&scale.to_le_bytes());
        std::fs::write(dir.join("model.safetensors"), &buf).unwrap();
        std::fs::write(
            dir.join("config.json"),
            r#"{"model_type":"qwen3","num_hidden_layers":1,"hidden_size":64,"num_attention_heads":2,"num_key_value_heads":2,"head_dim":32,"intermediate_size":128,"vocab_size":16,"max_position_embeddings":128}"#,
        ).unwrap();

        let src = SafetensorsSource::open(&dir).unwrap();
        let v = src
            .find("blk.0.attn_q.weight")
            .expect("per-tensor FP8 still loads");
        assert_eq!(v.ggml_type, GgmlType::Q8_0);
        let reference: Vec<f32> = codes
            .iter()
            .map(|&c| crate::nvfp4_repack::fp8_e4m3_to_f32(c) * scale)
            .collect();
        let expect = crate::nvfp4_repack::f32_to_q8_0(&reference);
        assert_eq!(v.bytes.as_ref(), expect.as_slice());

        std::fs::remove_dir_all(&dir).ok();
    }
}

#[cfg(test)]
mod expert_slab_populate_tests {
    use super::{
        EXPERT_SLAB_RAM_FLOOR_FRAC, ExpertMmapAdvice, ExpertSlabPopulate,
        expert_slab_populate_fits, parse_expert_mmap_advice, parse_expert_slab_populate,
    };

    #[test]
    fn unset_advice_is_random_and_unset_populate_is_fits() {
        assert_eq!(parse_expert_mmap_advice(None), Ok(ExpertMmapAdvice::Random));
        assert_eq!(
            parse_expert_slab_populate(None),
            Ok(ExpertSlabPopulate::Fits)
        );
    }

    #[test]
    fn populate_policy_parses_both_arms_and_rejects_junk() {
        assert_eq!(
            parse_expert_slab_populate(Some("fits")),
            Ok(ExpertSlabPopulate::Fits)
        );
        assert_eq!(
            parse_expert_slab_populate(Some("off")),
            Ok(ExpertSlabPopulate::Off)
        );
        assert!(parse_expert_slab_populate(Some("yes")).is_err());
        assert!(parse_expert_slab_populate(Some("")).is_err());
    }

    /// The sizing case the disk tier exists for: an artifact larger than host RAM must keep the
    /// demand-fault tier, or populating it would thrash the very page cache it depends on.
    #[test]
    fn slab_larger_than_available_ram_does_not_populate() {
        let total = 64 << 30;
        assert!(!expert_slab_populate_fits(
            120 << 30,
            60 << 30,
            total,
            EXPERT_SLAB_RAM_FLOOR_FRAC
        ));
    }

    /// The GLM-5.3-Flash shape measured 2026-08-28: a 159.5 GiB expert slab on a 499 GiB host.
    #[test]
    fn slab_fitting_under_the_floor_populates() {
        let total = 499 << 30;
        assert!(expert_slab_populate_fits(
            160 << 30,
            490 << 30,
            total,
            EXPERT_SLAB_RAM_FLOOR_FRAC
        ));
    }

    /// The floor is what stops a slab that technically fits from consuming the last of RAM.
    #[test]
    fn floor_is_enforced_against_a_slab_that_would_just_fit() {
        let total = 100 << 30;
        // 85 GiB available, 80 GiB slab: 5 GiB left is under the 20 GiB floor.
        assert!(!expert_slab_populate_fits(
            80 << 30,
            85 << 30,
            total,
            EXPERT_SLAB_RAM_FLOOR_FRAC
        ));
        // 60 GiB slab leaves 25 GiB, above the floor.
        assert!(expert_slab_populate_fits(
            60 << 30,
            85 << 30,
            total,
            EXPERT_SLAB_RAM_FLOOR_FRAC
        ));
    }

    #[test]
    fn empty_or_unknown_budget_never_underflows() {
        assert!(expert_slab_populate_fits(
            0,
            0,
            0,
            EXPERT_SLAB_RAM_FLOOR_FRAC
        ));
        assert!(!expert_slab_populate_fits(
            1 << 30,
            0,
            64 << 30,
            EXPERT_SLAB_RAM_FLOOR_FRAC
        ));
    }
}
