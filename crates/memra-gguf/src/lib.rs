//! Minimal GGUF v3 reader, mmap-based, layout copied 1:1 from llama.cpp `ggml/src/gguf.cpp`.
//!
//! On-disk layout (little-endian):
//!   magic "GGUF" (4 bytes) | version u32 (==3) | n_tensors i64 | n_kv i64
//!   n_kv × { key: gguf_string | value_type: u32 | value }
//!   n_tensors × { name: gguf_string | n_dims: u32 | ne[n_dims]: i64 | ggml_type: u32 | offset: u64 }
//!   padding to `general.alignment` (default 32)
//!   tensor data blob (each tensor at data_start + offset)
//!
//! gguf_string = len: u64 | bytes[len]  (no NUL terminator)
//!
//! SPLIT (multi-shard) models: `llama-gguf-split` writes one complete GGUF per shard, each with
//! its own header, its own tensor-info table, and its own data blob, tagged by three KV keys —
//! `split.no` (u16, 0-based), `split.count` (u16), `split.tensors.count` (i32, the TOTAL across
//! all shards). Tensor `offset`s are relative to the OWNING shard's `data_start`. Shard 0 carries
//! the full architecture/tokenizer metadata; later shards carry only the three split keys.
//! `GgufFile::open` on any shard of such a set discovers its siblings by the standard
//! `-%05d-of-%05d.gguf` filename form and presents one merged tensor table, so every caller sees
//! a split model exactly as it sees a single-file one. Step-3.7-Flash IQ4_XS (97.78 GiB, 754
//! tensors over 3 shards) is the model that forced this: blocks 0..21 live in shard 1 and the
//! loader died at `blk.22` with `need post_attention_norm or ffn_norm`.

use memmap2::Mmap;
use std::collections::BTreeMap;
use std::fs::File;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;

pub mod config;
pub mod d2t;
pub mod dequant;
pub mod dsv4;
pub mod dsv4_decode;
pub mod dsv4_dspark;
pub mod dsv4_forward;
pub mod execution_manifest;
pub mod hf;
pub mod hf_mapping;
pub mod micro_gguf;
pub mod model_packs;
pub mod model_plan;
pub mod nvfp4_repack;
pub mod placement;
pub mod safetensors;
pub mod source;
pub mod spec_oracle;
pub mod tensor_contract;

pub const GGUF_MAGIC: u32 = 0x4655_4747; // "GGUF" little-endian
pub const GGUF_DEFAULT_ALIGNMENT: u64 = 32;
const MAX_SPLIT_SHARDS: usize = 1_024;
const MAX_GGUF_METADATA_BYTES: usize = 256 * 1024 * 1024;
const MAX_GGUF_STRING_BYTES: usize = 16 * 1024 * 1024;
const MAX_GGUF_ARRAY_ELEMENTS: usize = 2_000_000;
const MAX_GGUF_METADATA_DEPTH: usize = 64;
const MAX_GGUF_KV_ENTRIES: usize = 1_000_000;
const MAX_GGUF_TENSORS: usize = 1_000_000;
const MAX_GGUF_TENSOR_RANK: usize = 128;

/// ggml_type ids — values are the on-disk integers (ggml/include/ggml.h).
/// Variant names mirror ggml's C enum exactly (Q4_0, Q8_K, …) by design.
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum GgmlType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    Q8_K = 15,
    IQ2_XXS = 16,
    IQ2_XS = 17,
    IQ3_XXS = 18,
    IQ1_S = 19,
    IQ4_NL = 20,
    IQ3_S = 21,
    IQ2_S = 22,
    IQ4_XS = 23,
    I8 = 24,
    I16 = 25,
    I32 = 26,
    I64 = 27,
    F64 = 28,
    IQ1_M = 29,
    BF16 = 30,
    TQ1_0 = 34,
    TQ2_0 = 35,
    MXFP4 = 39,
    NVFP4 = 40,
    Q1_0 = 41,
}

impl GgmlType {
    pub fn from_u32(v: u32) -> Option<Self> {
        use GgmlType::*;
        Some(match v {
            0 => F32,
            1 => F16,
            2 => Q4_0,
            3 => Q4_1,
            6 => Q5_0,
            7 => Q5_1,
            8 => Q8_0,
            9 => Q8_1,
            10 => Q2_K,
            11 => Q3_K,
            12 => Q4_K,
            13 => Q5_K,
            14 => Q6_K,
            15 => Q8_K,
            16 => IQ2_XXS,
            17 => IQ2_XS,
            18 => IQ3_XXS,
            19 => IQ1_S,
            20 => IQ4_NL,
            21 => IQ3_S,
            22 => IQ2_S,
            23 => IQ4_XS,
            24 => I8,
            25 => I16,
            26 => I32,
            27 => I64,
            28 => F64,
            29 => IQ1_M,
            30 => BF16,
            34 => TQ1_0,
            35 => TQ2_0,
            39 => MXFP4,
            40 => NVFP4,
            41 => Q1_0,
            _ => return None,
        })
    }

    /// (block_size in elements, type_size in bytes) — from ggml.c type_traits.
    /// bytes_for_n_elems = n_elems / block_size * type_size.
    pub fn block_and_type_size(self) -> (u64, u64) {
        self.try_block_and_type_size()
            .unwrap_or_else(|| panic!("block_and_type_size not implemented for {self:?}"))
    }

    /// Exact encoded byte extent for a tensor geometry, or `None` on overflow, an unsupported
    /// storage layout, or a quantized element count that is not block-divisible. Manifest-backed
    /// sources use this same calculation as the GGUF parser so geometry and byte ranges cannot
    /// drift across formats before reaching native/CUDA consumers.
    pub fn checked_nbytes(self, ne: &[u64]) -> Option<u64> {
        let elements = ne
            .iter()
            .try_fold(1u64, |total, extent| total.checked_mul(*extent))?;
        let (block, type_size) = self.try_block_and_type_size()?;
        if elements % block != 0 {
            return None;
        }
        (elements / block).checked_mul(type_size)
    }

    fn try_block_and_type_size(self) -> Option<(u64, u64)> {
        use GgmlType::*;
        Some(match self {
            F32 => (1, 4),
            F16 => (1, 2),
            BF16 => (1, 2),
            F64 => (1, 8),
            I8 => (1, 1),
            I16 => (1, 2),
            I32 => (1, 4),
            I64 => (1, 8),
            Q4_0 => (32, 18), // 2 (d) + 16 (16 bytes for 32×4bit)
            Q4_1 => (32, 20), // 2 d + 2 m + 16
            Q5_0 => (32, 22), // 2 d + 4 qh + 16
            Q5_1 => (32, 24), // 2 d + 2 m + 4 qh + 16
            Q8_0 => (32, 34), // 2 d + 32 int8
            Q8_1 => (32, 36), // 4 (d,s as fp16×2) + 32
            // k-quants, super-block QK_K=256
            Q2_K => (256, 84),
            Q3_K => (256, 110),
            Q4_K => (256, 144),
            Q5_K => (256, 176),
            Q6_K => (256, 210),
            Q8_K => (256, 292),
            IQ4_NL => (32, 18),
            IQ4_XS => (256, 136),
            // i-quants (all QK_K=256 super-blocks) — sizes from ggml-common.h static_asserts
            IQ2_XXS => (256, 66),
            IQ2_XS => (256, 74),
            IQ2_S => (256, 82),
            IQ3_XXS => (256, 98),
            IQ3_S => (256, 110),
            IQ1_S => (256, 50),
            IQ1_M => (256, 56),
            MXFP4 => (32, 17), // 1 (E8M0 scale) + 16 (32×4bit e2m1)
            NVFP4 => (64, 36), // 4 (UE4M3 sub-scales, 1 per 16 elems) + 32 (64×4bit e2m1)
            TQ1_0 | TQ2_0 | Q1_0 => return None,
        })
    }
}

/// A metadata value. Arrays keep their element type + raw decoded values.
#[derive(Debug, Clone)]
pub enum MetaValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    U64(u64),
    I64(i64),
    F32(f32),
    F64(f64),
    Bool(bool),
    String(String),
    Array(Vec<MetaValue>),
}

impl MetaValue {
    pub fn as_u64(&self) -> Option<u64> {
        Some(match self {
            MetaValue::U8(v) => *v as u64,
            MetaValue::U16(v) => *v as u64,
            MetaValue::U32(v) => *v as u64,
            MetaValue::U64(v) => *v,
            MetaValue::I8(v) if *v >= 0 => *v as u64,
            MetaValue::I16(v) if *v >= 0 => *v as u64,
            MetaValue::I32(v) if *v >= 0 => *v as u64,
            MetaValue::I64(v) if *v >= 0 => *v as u64,
            MetaValue::Bool(v) => *v as u64,
            _ => return None,
        })
    }
    pub fn as_f32(&self) -> Option<f32> {
        Some(match self {
            MetaValue::F32(v) => *v,
            MetaValue::F64(v) => *v as f32,
            _ => return None,
        })
    }
    pub fn as_str(&self) -> Option<&str> {
        if let MetaValue::String(s) = self {
            Some(s)
        } else {
            None
        }
    }
    pub fn as_str_array(&self) -> Option<Vec<&str>> {
        if let MetaValue::Array(a) = self {
            a.iter().map(|v| v.as_str()).collect()
        } else {
            None
        }
    }
}

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub ne: Vec<u64>, // dimensions, ne[0] fastest
    pub ggml_type: GgmlType,
    pub offset: u64,  // relative to the OWNING shard's data_start
    pub n_bytes: u64, // computed size
    /// Index into `GgufFile::shards` of the file this tensor's bytes live in. Always 0 for a
    /// single-file GGUF, so `offset` keeps its historical meaning there (relative to `data_start`).
    pub shard: usize,
}

impl TensorInfo {
    pub fn n_elements(&self) -> u64 {
        self.ne.iter().product()
    }
}

/// One physical GGUF file. A single-file model has exactly one; a split model has `split.count`.
struct Shard {
    mmap: Mmap,
    /// The same opened inode backing `mmap`, retained for disk-tier positioned reads.
    file: Arc<File>,
    /// On-disk path, retained for diagnostics and adjacent artifact lookup.
    path: PathBuf,
    /// Where this shard's tensor-data blob begins — each shard has its OWN header, so each has
    /// its own `data_start`. A tensor's absolute offset is `shards[t.shard].data_start + t.offset`.
    data_start: u64,
}

pub struct GgufFile {
    /// Shard 0 first, then ascending `split.no`. Length 1 for a single-file model.
    shards: Vec<Shard>,
    pub version: u32,
    /// Merged metadata. Shard 0 carries the architecture/tokenizer KVs and wins every collision;
    /// later shards contribute only keys shard 0 lacks (in practice nothing but `split.no`).
    pub metadata: BTreeMap<String, MetaValue>,
    /// Every tensor across every shard, in shard-then-file order. `TensorInfo::shard` says where.
    pub tensors: Vec<TensorInfo>,
    /// Shard 0's data start. Kept public for back-compat; use `tensor_file_range` for a tensor's
    /// real absolute offset, which on a split model is relative to ITS OWN shard.
    pub data_start: u64,
    pub alignment: u64,
}

struct Cursor<'a> {
    buf: &'a [u8],
    path: &'a Path,
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(buf: &'a [u8], path: &'a Path) -> Self {
        Self { buf, path, pos: 0 }
    }

    fn remaining(&self) -> usize {
        self.buf.len().saturating_sub(self.pos)
    }

    fn take(&mut self, len: usize, what: &str) -> io::Result<&'a [u8]> {
        let start = self.pos;
        let remaining = self.remaining();
        let end = start.checked_add(len).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "GGUF {what} length {len} overflows the byte offset {start} in {}",
                    self.path.display()
                ),
            )
        })?;
        let bytes = self.buf.get(start..end).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::UnexpectedEof,
                format!(
                    "truncated GGUF {what} at byte offset {start} in {}: need {len} bytes, only \
                 {remaining} remain",
                    self.path.display()
                ),
            )
        })?;
        self.pos = end;
        Ok(bytes)
    }

    fn read<const N: usize>(&mut self) -> io::Result<[u8; N]> {
        let bytes = self.take(N, "field")?;
        let mut out = [0u8; N];
        out.copy_from_slice(bytes);
        Ok(out)
    }

    fn u32(&mut self) -> io::Result<u32> {
        Ok(u32::from_le_bytes(self.read::<4>()?))
    }
    fn i64(&mut self) -> io::Result<i64> {
        Ok(i64::from_le_bytes(self.read::<8>()?))
    }
    fn u64(&mut self) -> io::Result<u64> {
        Ok(u64::from_le_bytes(self.read::<8>()?))
    }

    fn string(&mut self) -> io::Result<String> {
        let length_offset = self.pos;
        let raw_len = self.u64()?;
        let len = usize::try_from(raw_len).map_err(|_| io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "GGUF string length {raw_len} at byte offset {length_offset} does not fit usize \
                 in {}",
                self.path.display()
            ),
        ))?;
        if len > MAX_GGUF_STRING_BYTES {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "GGUF string length {raw_len} at byte offset {length_offset} exceeds the \
                     {MAX_GGUF_STRING_BYTES}-byte limit in {}",
                    self.path.display()
                ),
            ));
        }
        let bytes = self.take(len, &format!("string length {raw_len}"))?;
        Ok(String::from_utf8_lossy(bytes).into_owned())
    }

    fn value_min_bytes(type_id: u32) -> Option<usize> {
        Some(match type_id {
            0 | 1 | 7 => 1,
            2 | 3 => 2,
            4..=6 => 4,
            8 => 8,
            9 => 12,
            10..=12 => 8,
            _ => return None,
        })
    }

    fn value(&mut self, type_id: u32) -> io::Result<MetaValue> {
        self.value_with_depth(type_id, 0)
    }

    fn value_with_depth(&mut self, type_id: u32, depth: usize) -> io::Result<MetaValue> {
        if depth > MAX_GGUF_METADATA_DEPTH {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "GGUF metadata nesting exceeds {MAX_GGUF_METADATA_DEPTH} levels at byte offset {} in {}",
                    self.pos,
                    self.path.display()
                ),
            ));
        }
        Ok(match type_id {
            0 => MetaValue::U8(self.read::<1>()?[0]),
            1 => MetaValue::I8(self.read::<1>()?[0] as i8),
            2 => MetaValue::U16(u16::from_le_bytes(self.read::<2>()?)),
            3 => MetaValue::I16(i16::from_le_bytes(self.read::<2>()?)),
            4 => MetaValue::U32(self.u32()?),
            5 => MetaValue::I32(i32::from_le_bytes(self.read::<4>()?)),
            6 => MetaValue::F32(f32::from_le_bytes(self.read::<4>()?)),
            7 => MetaValue::Bool(self.read::<1>()?[0] != 0),
            8 => MetaValue::String(self.string()?),
            9 => {
                let type_offset = self.pos;
                let elem_type = self.u32()?;
                let min_elem_bytes = Self::value_min_bytes(elem_type).ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!(
                            "unknown GGUF metadata array element type {elem_type} at byte offset \
                             {type_offset} in {}",
                            self.path.display()
                        ),
                    )
                })?;
                let length_offset = self.pos;
                let raw_n = self.u64()?;
                let n = usize::try_from(raw_n).map_err(|_| io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "GGUF metadata array length {raw_n} at byte offset {length_offset} does \
                         not fit usize in {}",
                        self.path.display()
                    ),
                ))?;
                if n > MAX_GGUF_ARRAY_ELEMENTS {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!(
                            "GGUF metadata array length {raw_n} at byte offset {length_offset} \
                             exceeds the {MAX_GGUF_ARRAY_ELEMENTS}-element limit in {}",
                            self.path.display()
                        ),
                    ));
                }
                let remaining = self.remaining();
                if n > remaining / min_elem_bytes {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!(
                            "GGUF metadata array length {raw_n} at byte offset {length_offset} in \
                             {} cannot fit in {remaining} remaining bytes (element type \
                             {elem_type} needs at least {min_elem_bytes} bytes)",
                            self.path.display()
                        ),
                    ));
                }
                let mut v = Vec::new();
                v.try_reserve_exact(n).map_err(|e| {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!(
                            "cannot allocate GGUF metadata array length {raw_n} at byte offset \
                         {length_offset} in {}: {e}",
                            self.path.display()
                        ),
                    )
                })?;
                for _ in 0..n {
                    v.push(self.value_with_depth(elem_type, depth + 1)?);
                }
                MetaValue::Array(v)
            }
            10 => MetaValue::U64(self.u64()?),
            11 => MetaValue::I64(self.i64()?),
            12 => MetaValue::F64(f64::from_le_bytes(self.read::<8>()?)),
            other => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "unknown GGUF metadata type {other} at byte offset {} in {}",
                        self.pos,
                        self.path.display()
                    ),
                ));
            }
        })
    }
}

fn checked_count(raw: i64, field: &str, byte_offset: usize, path: &Path) -> io::Result<usize> {
    usize::try_from(raw).map_err(|_| io::Error::new(
        io::ErrorKind::InvalidData,
        format!(
            "GGUF {field}={raw} at byte offset {byte_offset} in {} must be non-negative and fit \
             usize",
            path.display()
        ),
    ))
}

fn structural_u64(
    metadata: &BTreeMap<String, MetaValue>,
    key: &str,
    path: &Path,
) -> io::Result<Option<u64>> {
    let Some(value) = metadata.get(key) else {
        return Ok(None);
    };
    let signed = match value {
        MetaValue::U8(v) => i128::from(*v),
        MetaValue::I8(v) => i128::from(*v),
        MetaValue::U16(v) => i128::from(*v),
        MetaValue::I16(v) => i128::from(*v),
        MetaValue::U32(v) => i128::from(*v),
        MetaValue::I32(v) => i128::from(*v),
        MetaValue::U64(v) => return Ok(Some(*v)),
        MetaValue::I64(v) => i128::from(*v),
        MetaValue::Bool(v) => {
            if *v {
                1
            } else {
                0
            }
        }
        _ => return Ok(None),
    };
    if signed < 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "GGUF metadata {key}={signed} in {} must be non-negative",
                path.display()
            ),
        ));
    }
    Ok(Some(signed as u64))
}

fn metadata_alignment(metadata: &BTreeMap<String, MetaValue>, path: &Path) -> io::Result<u64> {
    let alignment =
        structural_u64(metadata, "general.alignment", path)?.unwrap_or(GGUF_DEFAULT_ALIGNMENT);
    if alignment == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "GGUF general.alignment=0 in {} must be non-zero",
                path.display()
            ),
        ));
    }
    Ok(alignment)
}

fn aligned_data_start(header_end: u64, alignment: u64, path: &Path) -> io::Result<u64> {
    let rounded = header_end.checked_add(alignment - 1).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "GGUF data-start overflow in {}: header_end={header_end}, alignment={alignment}",
                path.display()
            ),
        )
    })?;
    Ok(rounded / alignment * alignment)
}

/// Parse ONE physical GGUF file: `(shard, version, metadata, tensor infos with shard=usize::MAX)`.
/// `TensorInfo::shard` is patched by the caller once the shard's index is known.
#[allow(clippy::type_complexity)] // allow: one-shot composite type; naming it would hide the shape that matters at the call site
#[allow(clippy::manual_is_multiple_of)] // allow: divisor is runtime-derived; the modulo form keeps a zero divisor loud (a panic), where is_multiple_of would return false silently
fn parse_one(
    path: PathBuf,
) -> std::io::Result<(Shard, u32, BTreeMap<String, MetaValue>, Vec<TensorInfo>)> {
    let file = Arc::new(File::open(&path)?);
    let mmap = unsafe { Mmap::map(file.as_ref())? };
    let mut c = Cursor::new(&mmap, &path);

    // The header checks RETURN, they do not panic (lane/step-draft 2026-08-07). `parse_one` is
    // already `io::Result`, and "this file is not a GGUF" / "wrong GGUF version" is the most
    // ordinary caller error there is — an operator typo'ing a path onto some other file. As a
    // panic it crossed a thread boundary and came out the far side unrecognizable: the server's
    // worker thread caught it, printed `PANIC in the GPU worker thread`, spent a full respawn
    // attempt RELOADING EVERY WEIGHT, and then told the operator `worker init failed: worker died
    // during init` — with the drafter-attach refusal text, which names the offending path and
    // says what to do, unreachable because the error never came back as an Err. Found by this
    // lane's own arm C on a card that had finally freed up (raw/armC-refuse-baddraft-
    // 20260807T001704Z.log): the FIRST run of that arm, on a contended card, had reported a
    // trunk OOM and never reached here.
    //
    // Bytes are quoted, not summarized, and the expected value comes along: a wrong-magic file is
    // usually a wrong FILE, and the magic bytes are the fastest way to see what it actually is.
    let magic = c.u32()?;
    if magic != GGUF_MAGIC {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "bad GGUF magic: {magic:#010x} (expected {GGUF_MAGIC:#010x} = \"GGUF\") in {} — \
                 this file is not a GGUF",
                path.display()
            ),
        ));
    }
    let version = c.u32()?;
    if version != 3 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "only GGUF v3 supported, got v{version} in {} — requantize or re-export with a \
                 current converter",
                path.display()
            ),
        ));
    }
    let n_tensors_offset = c.pos;
    let raw_n_tensors = c.i64()?;
    let n_tensors = checked_count(raw_n_tensors, "n_tensors", n_tensors_offset, &path)?;
    let n_kv_offset = c.pos;
    let raw_n_kv = c.i64()?;
    let n_kv = checked_count(raw_n_kv, "n_kv", n_kv_offset, &path)?;
    if n_tensors > MAX_GGUF_TENSORS || n_kv > MAX_GGUF_KV_ENTRIES {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "GGUF header declares n_tensors={n_tensors}, n_kv={n_kv}; limits are \
             {MAX_GGUF_TENSORS} tensors and {MAX_GGUF_KV_ENTRIES} metadata entries"
            ),
        ));
    }

    const MIN_KV_BYTES: usize = 13; // empty key + type id + smallest scalar value
    let remaining = c.remaining();
    if n_kv > remaining / MIN_KV_BYTES {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "GGUF n_kv={raw_n_kv} at byte offset {n_kv_offset} in {} cannot fit in \
                 {remaining} remaining bytes (each entry needs at least {MIN_KV_BYTES} bytes)",
                path.display()
            ),
        ));
    }

    // --- metadata KV ---
    let metadata_start = c.pos;
    let mut metadata = BTreeMap::new();
    for _ in 0..n_kv {
        let key = c.string()?;
        let vtype = c.u32()?;
        let val = c.value(vtype)?;
        metadata.insert(key, val);
        if c.pos.saturating_sub(metadata_start) > MAX_GGUF_METADATA_BYTES {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "GGUF metadata exceeds the {MAX_GGUF_METADATA_BYTES}-byte cumulative limit in {}",
                    path.display()
                ),
            ));
        }
    }

    let alignment = metadata_alignment(&metadata, &path)?;

    // --- tensor infos ---
    const MIN_TENSOR_INFO_BYTES: usize = 24; // empty name + n_dims + type id + offset
    let remaining = c.remaining();
    if n_tensors > remaining / MIN_TENSOR_INFO_BYTES {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "GGUF n_tensors={raw_n_tensors} at byte offset {n_tensors_offset} in {} cannot \
                 fit in {remaining} remaining bytes (each tensor info needs at least \
                 {MIN_TENSOR_INFO_BYTES} bytes)",
                path.display()
            ),
        ));
    }
    let mut tensors = Vec::new();
    tensors.try_reserve_exact(n_tensors).map_err(|e| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "cannot allocate GGUF n_tensors={raw_n_tensors} in {}: {e}",
                path.display()
            ),
        )
    })?;
    for _ in 0..n_tensors {
        let name = c.string()?;
        let n_dims_offset = c.pos;
        let raw_n_dims = c.u32()?;
        let n_dims = usize::try_from(raw_n_dims).map_err(|_| io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "tensor {name} n_dims={raw_n_dims} at byte offset {n_dims_offset} in {} does not \
                 fit usize",
                path.display()
            ),
        ))?;
        if n_dims > MAX_GGUF_TENSOR_RANK {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "tensor {name} n_dims={n_dims} exceeds the rank limit {MAX_GGUF_TENSOR_RANK} in {}",
                    path.display()
                ),
            ));
        }
        let remaining = c.remaining();
        let min_tail_bytes = n_dims.checked_mul(8).and_then(|n| n.checked_add(12));
        if min_tail_bytes.is_none_or(|needed| needed > remaining) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "tensor {name} n_dims={raw_n_dims} at byte offset {n_dims_offset} in {} \
                     cannot fit in {remaining} remaining bytes",
                    path.display()
                ),
            ));
        }
        let mut ne = Vec::new();
        ne.try_reserve_exact(n_dims).map_err(|e| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "cannot allocate tensor {name} n_dims={raw_n_dims} in {}: {e}",
                    path.display()
                ),
            )
        })?;
        for dim in 0..n_dims {
            let dim_offset = c.pos;
            let raw_ne = c.i64()?;
            let extent = u64::try_from(raw_ne).map_err(|_| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "tensor {name} dimension {dim}={raw_ne} at byte offset {dim_offset} in {} \
                     must be non-negative",
                        path.display()
                    ),
                )
            })?;
            ne.push(extent);
        }
        let type_offset = c.pos;
        let type_id = c.u32()?;
        let ggml_type = GgmlType::from_u32(type_id).ok_or_else(|| io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "tensor {name} has unknown ggml_type {type_id} at byte offset {type_offset} in {}",
                path.display()
            ),
        ))?;
        let offset = c.u64()?;
        let mut n_elems = 1u64;
        for (dim, extent) in ne.iter().copied().enumerate() {
            n_elems = n_elems.checked_mul(extent).ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "tensor {name} element count overflow at dimension {dim}={extent} in {}",
                        path.display()
                    ),
                )
            })?;
        }
        let (blck, tsize) = ggml_type.try_block_and_type_size().ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "tensor {name} ggml_type {ggml_type:?} has no supported block layout in {}",
                    path.display()
                ),
            )
        })?;
        if n_elems % blck != 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "tensor {name} elements={n_elems} is not divisible by block={blck} for \
                     {ggml_type:?} in {}",
                    path.display()
                ),
            ));
        }
        let n_bytes = (n_elems / blck).checked_mul(tsize).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "tensor {name} byte size overflow: elements={n_elems}, block={blck}, \
                 type_size={tsize} in {}",
                    path.display()
                ),
            )
        })?;
        tensors.push(TensorInfo {
            name,
            ne,
            ggml_type,
            offset,
            n_bytes,
            shard: usize::MAX,
        });
    }

    // data section starts at the next `alignment` boundary after the header.
    let header_end = u64::try_from(c.pos).map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "GGUF header byte offset {} does not fit u64 in {}",
                c.pos,
                path.display()
            ),
        )
    })?;
    let data_start = aligned_data_start(header_end, alignment, &path)?;
    let file_len = u64::try_from(mmap.len()).map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "GGUF file length {} does not fit u64 in {}",
                mmap.len(),
                path.display()
            ),
        )
    })?;
    for tensor in &tensors {
        let start = data_start.checked_add(tensor.offset).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "tensor {} byte range overflow in {}: data_start={data_start} + offset={} \
                 overflows u64",
                    tensor.name,
                    path.display(),
                    tensor.offset
                ),
            )
        })?;
        let end = start.checked_add(tensor.n_bytes).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "tensor {} byte range overflow in {}: start={start} + n_bytes={} overflows u64",
                    tensor.name,
                    path.display(),
                    tensor.n_bytes
                ),
            )
        })?;
        if end > file_len {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "tensor {} byte range [{start}, {end}) exceeds shard file length {file_len} \
                     in {} (data_start={data_start}, offset={}, n_bytes={})",
                    tensor.name,
                    path.display(),
                    tensor.offset,
                    tensor.n_bytes
                ),
            ));
        }
    }
    Ok((
        Shard {
            mmap,
            file,
            path,
            data_start,
        },
        version,
        metadata,
        tensors,
    ))
}

/// Sibling paths of a split shard, in ascending `split.no`, given ANY member's path.
///
/// `llama-gguf-split` names shards `<prefix>-%05d-of-%05d.gguf` (gguf-split.cpp's
/// `SPLIT_PATH_FORMAT`). Rather than parse the number out of the name we rebuild every expected
/// name from `count` — so a shard whose filename disagrees with its own `split.no` cannot silently
/// map to the wrong bytes. Returns None if the name does not carry the standard suffix.
fn split_sibling_paths(path: &Path, count: usize) -> Option<Vec<PathBuf>> {
    let name = path.file_name()?.to_str()?;
    let stem = name.strip_suffix(".gguf")?;
    // trailing "-%05d-of-%05d"
    let (head, tail) = stem.rsplit_once("-of-")?;
    if tail.len() != 5 || !tail.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    if tail.parse::<usize>().ok()? != count {
        return None;
    }
    let (prefix, num) = head.rsplit_once('-')?;
    if num.len() != 5 || !num.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    let dir = path.parent()?;
    Some(
        (1..=count)
            .map(|i| dir.join(format!("{prefix}-{i:05}-of-{count:05}.gguf")))
            .collect(),
    )
}

impl GgufFile {
    /// Open a GGUF model. If `path` names a shard of a split (multi-file) model — detected by the
    /// `split.count` KV — every sibling shard is opened too and the result presents ONE merged
    /// tensor table. Callers cannot tell a split model from a single-file one.
    pub fn open<P: AsRef<Path>>(path: P) -> std::io::Result<Self> {
        let path = path.as_ref().to_path_buf();
        let (shard0, version, mut metadata, mut tensors) = parse_one(path.clone())?;
        let alignment = metadata_alignment(&metadata, &path)?;
        let data_start = shard0.data_start;

        let raw_split_count = structural_u64(&metadata, "split.count", &path)?.unwrap_or(0);
        let split_count = usize::try_from(raw_split_count).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "GGUF split.count={raw_split_count} does not fit usize in {}",
                    path.display()
                ),
            )
        })?;
        if split_count > MAX_SPLIT_SHARDS {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "GGUF split.count={split_count} exceeds maximum supported shard count {MAX_SPLIT_SHARDS}"
                ),
            ));
        }
        // count 0 or 1 = not split. Any member of the set is a valid entry point: the sibling
        // names come from the filename form, and shard 0's KVs (architecture, tokenizer) win the
        // merge below, so opening shard 3 yields the same model as opening shard 1.
        if split_count <= 1 {
            for t in &mut tensors {
                t.shard = 0;
            }
            return Ok(Self {
                shards: vec![shard0],
                version,
                metadata,
                tensors,
                data_start,
                alignment,
            });
        }

        let paths = split_sibling_paths(&path, split_count).ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!(
                    "{} declares split.count={split_count} but its name is not the \
                     -%05d-of-%05d.gguf split form; cannot find sibling shards",
                    path.display()
                ),
            )
        })?;

        let total_expected = structural_u64(&metadata, "split.tensors.count", &path)?.unwrap_or(0);

        let mut shards: Vec<Shard> = Vec::new();
        shards.try_reserve_exact(split_count).map_err(|e| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "cannot allocate GGUF split.count={split_count} in {}: {e}",
                    path.display()
                ),
            )
        })?;
        let mut all: Vec<TensorInfo> = Vec::new();
        let mut merged: BTreeMap<String, MetaValue> = BTreeMap::new();
        // Every shard is (re)parsed here, including the one we were handed — one extra header
        // parse + lazy mmap is nothing against a 105 GB model, and it keeps the merge loop
        // uniform. `tensors`/`shard0` from the probe above are dropped.
        drop(shard0);
        drop(tensors);
        for (i, p) in paths.iter().enumerate() {
            let (sh, ver, meta, mut ts) = parse_one(p.clone())?;
            if ver != version {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("shard {} GGUF version {ver} != {version}", p.display()),
                ));
            }
            let raw_no = structural_u64(&meta, "split.no", p)?.unwrap_or(0);
            let no = usize::try_from(raw_no).map_err(|_| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "shard {} declares split.no={raw_no}, which does not fit usize",
                        p.display()
                    ),
                )
            })?;
            if no != i {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "shard {} declares split.no={no}, expected {i} from its filename",
                        p.display()
                    ),
                ));
            }
            let raw_cnt = structural_u64(&meta, "split.count", p)?.unwrap_or(0);
            let cnt = usize::try_from(raw_cnt).map_err(|_| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "shard {} declares split.count={raw_cnt}, which does not fit usize",
                        p.display()
                    ),
                )
            })?;
            if cnt != split_count {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "shard {} declares split.count={cnt}, expected {split_count}",
                        p.display()
                    ),
                ));
            }
            for t in &mut ts {
                t.shard = i;
            }
            all.append(&mut ts);
            // Shard 0 wins: it holds the architecture + tokenizer KVs. Later shards may only ADD.
            for (k, v) in meta {
                merged.entry(k).or_insert(v);
            }
            shards.push(sh);
        }
        if total_expected > 0 {
            let actual = u64::try_from(all.len()).map_err(|_| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("split tensor count {} does not fit u64", all.len()),
                )
            })?;
            if actual != total_expected {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "split model has {actual} tensors across {split_count} shards but \
                         split.tensors.count={total_expected}"
                    ),
                ));
            }
        }
        metadata = merged;
        let alignment = metadata_alignment(&metadata, &paths[0])?;
        let data_start = shards[0].data_start;
        Ok(Self {
            shards,
            version,
            metadata,
            tensors: all,
            data_start,
            alignment,
        })
    }

    /// Raw bytes for a tensor (mmap'd, zero-copy slice) from its OWNING shard.
    pub fn tensor_data(&self, t: &TensorInfo) -> &[u8] {
        let sh = &self.shards[t.shard];
        let start = (sh.data_start + t.offset) as usize;
        &sh.mmap[start..start + t.n_bytes as usize]
    }

    /// On-disk path of shard 0 (the whole model for a single-file GGUF).
    pub fn path(&self) -> &Path {
        &self.shards[0].path
    }

    /// Number of physical files backing this model (1 unless split).
    pub fn n_shards(&self) -> usize {
        self.shards.len()
    }

    /// On-disk path of a given shard.
    pub fn shard_path(&self, i: usize) -> &Path {
        &self.shards[i].path
    }

    /// Opened inode backing shard 0's parsed mmap. Disk-tier consumers clone this handle instead
    /// of reopening `path`, so a path replacement cannot change the bytes behind a loaded model.
    /// SPLIT MODELS: use `shard_file(t.shard)` — shard 0's inode does not hold every tensor.
    pub fn opened_file(&self) -> &Arc<File> {
        &self.shards[0].file
    }

    /// Opened inode backing a given shard's parsed mmap.
    pub fn shard_file(&self, i: usize) -> &Arc<File> {
        &self.shards[i].file
    }

    /// Absolute byte range `[start, end)` of a tensor's data within **its own shard's** file.
    /// `start = shards[t.shard].data_start + t.offset`; the disk-tier `HostBuf::Mmap` slices the
    /// mmap of that same shard (pair this with `shard_mmap_of`/`shard_file`, never with shard 0's).
    pub fn tensor_file_range(&self, t: &TensorInfo) -> (usize, usize) {
        let start = (self.shards[t.shard].data_start + t.offset) as usize;
        (start, start + t.n_bytes as usize)
    }

    pub fn find(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.iter().find(|t| t.name == name)
    }

    pub fn arch(&self) -> Option<&str> {
        self.metadata
            .get("general.architecture")
            .and_then(|v| v.as_str())
    }

    /// Get a metadata value, trying `{arch}.{suffix}` then the literal key.
    pub fn meta_arch(&self, suffix: &str) -> Option<&MetaValue> {
        if let Some(arch) = self.arch()
            && let Some(v) = self.metadata.get(&format!("{arch}.{suffix}"))
        {
            return Some(v);
        }
        self.metadata.get(suffix)
    }
}

#[cfg(test)]
mod split_tests {
    use super::*;

    #[test]
    fn signed_metadata_cannot_become_a_large_unsigned_value() {
        assert_eq!(MetaValue::I8(-1).as_u64(), None);
        assert_eq!(MetaValue::I64(-1).as_u64(), None);
        assert_eq!(MetaValue::I32(7).as_u64(), Some(7));
    }

    fn put_test_string(h: &mut Vec<u8>, s: &str) {
        h.extend_from_slice(&(s.len() as u64).to_le_bytes());
        h.extend_from_slice(s.as_bytes());
    }

    fn raw_header(n_tensors: i64, n_kv: i64) -> Vec<u8> {
        let mut h = Vec::new();
        h.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        h.extend_from_slice(&3u32.to_le_bytes());
        h.extend_from_slice(&n_tensors.to_le_bytes());
        h.extend_from_slice(&n_kv.to_le_bytes());
        h
    }

    fn put_test_tensor(h: &mut Vec<u8>, name: &str, ne: &[i64], type_id: u32, offset: u64) {
        put_test_string(h, name);
        h.extend_from_slice(&(ne.len() as u32).to_le_bytes());
        for n in ne {
            h.extend_from_slice(&n.to_le_bytes());
        }
        h.extend_from_slice(&type_id.to_le_bytes());
        h.extend_from_slice(&offset.to_le_bytes());
    }

    fn write_raw_fixture(tag: &str, bytes: &[u8]) -> (PathBuf, PathBuf) {
        let dir = std::env::temp_dir().join(format!("memra-ggufhard-{tag}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(format!("{tag}.gguf"));
        std::fs::write(&path, bytes).unwrap();
        (dir, path)
    }

    fn open_error_without_panic(path: &Path) -> String {
        match std::panic::catch_unwind(|| GgufFile::open(path)) {
            Ok(Err(e)) => e.to_string(),
            Ok(Ok(_)) => panic!("corrupt fixture {} opened successfully", path.display()),
            Err(_) => panic!("opening corrupt fixture {} panicked", path.display()),
        }
    }

    fn assert_error_mentions(msg: &str, expected: &[&str]) {
        for needle in expected {
            assert!(
                msg.contains(needle),
                "error must contain {needle:?}, got: {msg}"
            );
        }
    }

    /// Serialize a minimal but REAL GGUF v3 file: header, KVs, tensor infos, aligned data blob.
    /// Every tensor is F32 with `ne = [n]` and its bytes are `fill` repeated, so a wrong-shard read
    /// is detectable by value rather than only by length.
    fn write_gguf(path: &Path, kv: &[(&str, MetaValue)], tensors: &[(&str, u64, u8)]) {
        let mut h: Vec<u8> = Vec::new();
        h.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        h.extend_from_slice(&3u32.to_le_bytes());
        h.extend_from_slice(&(tensors.len() as i64).to_le_bytes());
        h.extend_from_slice(&(kv.len() as i64).to_le_bytes());
        let put_str = |h: &mut Vec<u8>, s: &str| {
            h.extend_from_slice(&(s.len() as u64).to_le_bytes());
            h.extend_from_slice(s.as_bytes());
        };
        for (k, v) in kv {
            put_str(&mut h, k);
            match v {
                MetaValue::U16(x) => {
                    h.extend_from_slice(&2u32.to_le_bytes());
                    h.extend_from_slice(&x.to_le_bytes());
                }
                MetaValue::I32(x) => {
                    h.extend_from_slice(&5u32.to_le_bytes());
                    h.extend_from_slice(&x.to_le_bytes());
                }
                MetaValue::String(s) => {
                    h.extend_from_slice(&8u32.to_le_bytes());
                    put_str(&mut h, s);
                }
                other => panic!("test writer does not handle {other:?}"),
            }
        }
        // tensor infos: offsets are relative to THIS file's data_start, packed in order
        let mut off = 0u64;
        for (name, n, _) in tensors {
            put_str(&mut h, name);
            h.extend_from_slice(&1u32.to_le_bytes()); // n_dims
            h.extend_from_slice(&(*n as i64).to_le_bytes());
            h.extend_from_slice(&(GgmlType::F32 as u32).to_le_bytes());
            h.extend_from_slice(&off.to_le_bytes());
            off += n * 4;
        }
        let data_start = (h.len() as u64).div_ceil(GGUF_DEFAULT_ALIGNMENT) * GGUF_DEFAULT_ALIGNMENT;
        h.resize(data_start as usize, 0);
        for (_, n, fill) in tensors {
            h.extend(std::iter::repeat_n(*fill, (*n * 4) as usize));
        }
        std::fs::write(path, &h).unwrap();
    }

    /// A 2-shard split pair in a fresh temp dir. Shard 0 carries the arch KV; shard 1 carries only
    /// the split keys — exactly how llama-gguf-split writes them (verified against the real
    /// Step-3.7-Flash IQ4_XS headers, research/step37-bringup-20260802/raw/).
    fn write_split_pair(tag: &str) -> (PathBuf, PathBuf, PathBuf) {
        let dir = std::env::temp_dir().join(format!("memra-split-{tag}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p0 = dir.join("m-00001-of-00002.gguf");
        let p1 = dir.join("m-00002-of-00002.gguf");
        write_gguf(
            &p0,
            &[
                ("general.architecture", MetaValue::String("step35".into())),
                ("split.no", MetaValue::U16(0)),
                ("split.count", MetaValue::U16(2)),
                ("split.tensors.count", MetaValue::I32(3)),
            ],
            &[("blk.0.w", 8, 0xA1), ("blk.1.w", 4, 0xA2)],
        );
        write_gguf(
            &p1,
            &[
                ("split.no", MetaValue::U16(1)),
                ("split.count", MetaValue::U16(2)),
                ("split.tensors.count", MetaValue::I32(3)),
            ],
            &[("blk.2.w", 16, 0xB1)],
        );
        (dir, p0, p1)
    }

    #[test]
    fn split_model_presents_one_merged_tensor_table() {
        let (dir, p0, _p1) = write_split_pair("merge");
        let g = GgufFile::open(&p0).unwrap();
        assert_eq!(g.n_shards(), 2);
        assert_eq!(
            g.tensors.len(),
            3,
            "all three tensors must be visible from shard 0"
        );
        // The tensor that lives in shard 1 is the one the step37 boot died on.
        let t = g
            .find("blk.2.w")
            .expect("blk.2.w is in shard 1 and must be found");
        assert_eq!(t.shard, 1);
        assert_eq!(g.tensor_data(t), vec![0xB1u8; 64].as_slice());
        // ...and shard 0's tensors still read correctly.
        assert_eq!(
            g.tensor_data(g.find("blk.0.w").unwrap()),
            vec![0xA1u8; 32].as_slice()
        );
        assert_eq!(
            g.tensor_data(g.find("blk.1.w").unwrap()),
            vec![0xA2u8; 16].as_slice()
        );
        // Metadata comes from shard 0 (shard 1 has no architecture KV at all).
        assert_eq!(g.arch(), Some("step35"));
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn any_shard_is_a_valid_entry_point() {
        let (dir, p0, p1) = write_split_pair("entry");
        let from0 = GgufFile::open(&p0).unwrap();
        let from1 = GgufFile::open(&p1).unwrap();
        // Opening the LAST shard must yield the same model, including shard 0's metadata.
        assert_eq!(from1.arch(), Some("step35"));
        assert_eq!(from1.tensors.len(), from0.tensors.len());
        for t in &from0.tensors {
            let u = from1.find(&t.name).unwrap();
            assert_eq!(
                (u.shard, u.offset, u.n_bytes),
                (t.shard, t.offset, t.n_bytes)
            );
            assert_eq!(from1.tensor_data(u), from0.tensor_data(t));
        }
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn tensor_file_range_is_relative_to_the_owning_shard() {
        let (dir, p0, p1) = write_split_pair("range");
        let g = GgufFile::open(&p0).unwrap();
        // The shard-1 tensor's range must address shard 1's file, so it must reproduce those bytes
        // when applied to shard 1 — and must NOT be a global offset past the end of shard 0.
        let t = g.find("blk.2.w").unwrap();
        let (s, e) = g.tensor_file_range(t);
        let raw1 = std::fs::read(&p1).unwrap();
        assert_eq!(&raw1[s..e], vec![0xB1u8; 64].as_slice());
        assert_eq!(g.shard_path(1), p1.as_path());
        assert!(!std::sync::Arc::ptr_eq(g.shard_file(0), g.shard_file(1)));
        let raw0 = std::fs::read(&p0).unwrap();
        let (s0, e0) = g.tensor_file_range(g.find("blk.0.w").unwrap());
        assert_eq!(&raw0[s0..e0], vec![0xA1u8; 32].as_slice());
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn single_file_gguf_is_unchanged_one_shard() {
        let dir = std::env::temp_dir().join(format!("memra-split-single-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("solo.gguf");
        write_gguf(
            &p,
            &[("general.architecture", MetaValue::String("qwen35".into()))],
            &[("tok_embd.weight", 8, 0x5A)],
        );
        let g = GgufFile::open(&p).unwrap();
        assert_eq!(g.n_shards(), 1);
        let t = g.find("tok_embd.weight").unwrap();
        assert_eq!(t.shard, 0);
        assert_eq!(g.tensor_data(t), vec![0x5Au8; 32].as_slice());
        // data_start keeps its historical meaning for a single-file model.
        let (s, _e) = g.tensor_file_range(t);
        assert_eq!(s as u64, g.data_start + t.offset);
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn split_shard_with_a_nonstandard_filename_is_a_clear_error() {
        let dir = std::env::temp_dir().join(format!("memra-split-badname-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("renamed-by-hand.gguf");
        write_gguf(
            &p,
            &[
                ("split.no", MetaValue::U16(0)),
                ("split.count", MetaValue::U16(3)),
            ],
            &[("blk.0.w", 4, 0x11)],
        );
        let msg = match GgufFile::open(&p) {
            Ok(_) => panic!("a split shard with a non-standard filename must not open silently"),
            Err(e) => e.to_string(),
        };
        assert!(
            msg.contains("split.count=3") && msg.contains("sibling shards"),
            "error must name the split count and the missing siblings, got: {msg}"
        );
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn a_non_gguf_file_is_an_error_not_a_panic() {
        // Regression, lane/step-draft 2026-08-07. This used to `assert_eq!` on the magic, and a
        // panic here does not survive a thread boundary usefully: memra-server's worker caught
        // it, burned a full respawn attempt reloading every weight, and reported `worker died
        // during init` — while the caller's own error text, which names the offending path and
        // says what to do about it, was unreachable because nothing ever came back as an Err.
        let dir = std::env::temp_dir().join(format!("memra-badmagic-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("not-a-gguf.gguf");
        std::fs::write(&p, b"this is not a GGUF file").unwrap();
        let msg = match GgufFile::open(&p) {
            Ok(_) => panic!("a non-GGUF file must not open"),
            Err(e) => e.to_string(),
        };
        // The observed bytes AND the expected value: a wrong magic is usually a wrong FILE, and
        // the magic is the fastest way to see what the file actually is.
        assert!(msg.contains("bad GGUF magic"), "got: {msg}");
        assert!(
            msg.contains("0x73696874"),
            "the observed magic must be quoted, got: {msg}"
        );
        assert!(
            msg.contains("GGUF"),
            "the expected magic must be shown, got: {msg}"
        );
        assert!(
            msg.contains("not-a-gguf.gguf"),
            "the path must be named, got: {msg}"
        );

        // Same for a wrong VERSION: real header, v2 — the other panic on this path.
        let p2 = dir.join("v2.gguf");
        let mut h: Vec<u8> = Vec::new();
        h.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        h.extend_from_slice(&2u32.to_le_bytes());
        h.extend_from_slice(&0i64.to_le_bytes());
        h.extend_from_slice(&0i64.to_le_bytes());
        std::fs::write(&p2, &h).unwrap();
        let msg2 = match GgufFile::open(&p2) {
            Ok(_) => panic!("a GGUF v2 file must not open"),
            Err(e) => e.to_string(),
        };
        assert!(
            msg2.contains("v2") && msg2.contains("v3"),
            "the error must name both the version found and the one supported, got: {msg2}"
        );
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn truncated_fixed_fields_and_strings_are_errors_not_panics() {
        let (dir, path) = write_raw_fixture("truncated-version", &GGUF_MAGIC.to_le_bytes());
        let msg = open_error_without_panic(&path);
        assert_error_mentions(&msg, &["byte offset 4", "need 4 bytes"]);
        std::fs::remove_dir_all(dir).ok();

        let mut h = raw_header(0, 1);
        h.extend_from_slice(&20u64.to_le_bytes());
        h.extend_from_slice(b"abcde");
        let (dir, path) = write_raw_fixture("truncated-string", &h);
        let msg = open_error_without_panic(&path);
        assert_error_mentions(
            &msg,
            &["GGUF string length 20", "byte offset 32", "5 remain"],
        );
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn metadata_string_and_tensor_rank_limits_fail_before_allocation() {
        let mut h = raw_header(0, 1);
        h.extend_from_slice(&((MAX_GGUF_STRING_BYTES as u64) + 1).to_le_bytes());
        h.extend_from_slice(&[0u8; 5]);
        let (dir, path) = write_raw_fixture("oversize-string", &h);
        let msg = open_error_without_panic(&path);
        assert_error_mentions(&msg, &["string length", "exceeds", "limit"]);
        std::fs::remove_dir_all(dir).ok();

        let mut h = raw_header(1, 0);
        h.extend_from_slice(&0u64.to_le_bytes()); // empty tensor name
        h.extend_from_slice(&((MAX_GGUF_TENSOR_RANK as u32) + 1).to_le_bytes());
        h.extend_from_slice(&[0u8; 12]);
        let (dir, path) = write_raw_fixture("oversize-rank", &h);
        let msg = open_error_without_panic(&path);
        assert_error_mentions(&msg, &["n_dims=129", "rank limit"]);
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn every_prefix_truncation_of_a_valid_gguf_is_an_error_not_a_panic() {
        let dir =
            std::env::temp_dir().join(format!("memra-ggufhard-prefixes-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let complete = dir.join("complete.gguf");
        write_gguf(
            &complete,
            &[("general.architecture", MetaValue::String("qwen35".into()))],
            &[("weight", 8, 0x5A)],
        );
        let bytes = std::fs::read(&complete).unwrap();
        GgufFile::open(&complete).expect("the complete control fixture must open");

        let truncated = dir.join("truncated.gguf");
        for len in 0..bytes.len() {
            std::fs::write(&truncated, &bytes[..len]).unwrap();
            match std::panic::catch_unwind(|| GgufFile::open(&truncated)) {
                Ok(Err(_)) => {}
                Ok(Ok(_)) => panic!("GGUF prefix of {len}/{} bytes opened", bytes.len()),
                Err(_) => panic!("GGUF prefix of {len}/{} bytes panicked", bytes.len()),
            }
        }
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn impossible_counts_and_array_lengths_are_errors_not_panics() {
        let h = raw_header(i64::MAX, 0);
        let (dir, path) = write_raw_fixture("oversized-tensors", &h);
        let msg = open_error_without_panic(&path);
        assert_error_mentions(&msg, &["n_tensors=9223372036854775807", "limits are"]);
        std::fs::remove_dir_all(dir).ok();

        let mut h = raw_header(0, 1);
        put_test_string(&mut h, "deep");
        h.extend_from_slice(&9u32.to_le_bytes());
        for _ in 0..=MAX_GGUF_METADATA_DEPTH {
            h.extend_from_slice(&9u32.to_le_bytes());
            h.extend_from_slice(&1u64.to_le_bytes());
        }
        h.extend_from_slice(&0u32.to_le_bytes());
        h.extend_from_slice(&1u64.to_le_bytes());
        h.push(1);
        let (dir, path) = write_raw_fixture("deep-metadata", &h);
        let msg = open_error_without_panic(&path);
        assert_error_mentions(&msg, &["metadata nesting exceeds", "64 levels"]);
        std::fs::remove_dir_all(dir).ok();

        let h = raw_header(0, -1);
        let (dir, path) = write_raw_fixture("negative-kv-count", &h);
        let msg = open_error_without_panic(&path);
        assert_error_mentions(&msg, &["n_kv=-1", "non-negative"]);
        std::fs::remove_dir_all(dir).ok();

        let mut h = raw_header(0, 1);
        put_test_string(&mut h, "x");
        h.extend_from_slice(&9u32.to_le_bytes());
        h.extend_from_slice(&0u32.to_le_bytes());
        h.extend_from_slice(&u64::MAX.to_le_bytes());
        let (dir, path) = write_raw_fixture("oversized-array", &h);
        let msg = open_error_without_panic(&path);
        assert_error_mentions(
            &msg,
            &["array length 18446744073709551615", "element limit"],
        );
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn invalid_dimensions_and_tensor_sizes_are_errors_not_panics() {
        let mut h = raw_header(1, 0);
        put_test_string(&mut h, "dims");
        h.extend_from_slice(&100u32.to_le_bytes());
        h.extend_from_slice(&0u64.to_le_bytes());
        let (dir, path) = write_raw_fixture("oversized-dims", &h);
        let msg = open_error_without_panic(&path);
        assert_error_mentions(&msg, &["tensor dims", "n_dims=100", "8 remaining bytes"]);
        std::fs::remove_dir_all(dir).ok();

        let mut h = raw_header(1, 0);
        put_test_tensor(&mut h, "negative", &[-1], GgmlType::F32 as u32, 0);
        let (dir, path) = write_raw_fixture("negative-dimension", &h);
        let msg = open_error_without_panic(&path);
        assert_error_mentions(&msg, &["tensor negative", "dimension 0=-1", "non-negative"]);
        std::fs::remove_dir_all(dir).ok();

        let mut h = raw_header(1, 0);
        put_test_tensor(&mut h, "blocked", &[1], GgmlType::Q4_0 as u32, 0);
        let (dir, path) = write_raw_fixture("partial-block", &h);
        let msg = open_error_without_panic(&path);
        assert_error_mentions(&msg, &["tensor blocked", "elements=1", "block=32"]);
        std::fs::remove_dir_all(dir).ok();

        let mut h = raw_header(1, 0);
        put_test_tensor(&mut h, "overflow", &[i64::MAX, 3], GgmlType::F32 as u32, 0);
        let (dir, path) = write_raw_fixture("element-overflow", &h);
        let msg = open_error_without_panic(&path);
        assert_error_mentions(&msg, &["tensor overflow", "element count overflow"]);
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn unknown_types_and_zero_alignment_are_errors_not_panics() {
        let mut h = raw_header(0, 1);
        put_test_string(&mut h, "x");
        h.extend_from_slice(&99u32.to_le_bytes());
        let (dir, path) = write_raw_fixture("unknown-metadata-type", &h);
        let msg = open_error_without_panic(&path);
        assert_error_mentions(&msg, &["unknown GGUF metadata type 99", "byte offset"]);
        std::fs::remove_dir_all(dir).ok();

        let mut h = raw_header(1, 0);
        put_test_tensor(&mut h, "unknown", &[1], u32::MAX, 0);
        let (dir, path) = write_raw_fixture("unknown-tensor-type", &h);
        let msg = open_error_without_panic(&path);
        assert_error_mentions(&msg, &["tensor unknown", "unknown ggml_type 4294967295"]);
        std::fs::remove_dir_all(dir).ok();

        let mut h = raw_header(1, 0);
        put_test_tensor(&mut h, "unsupported", &[32], GgmlType::Q1_0 as u32, 0);
        let (dir, path) = write_raw_fixture("unsupported-tensor-type", &h);
        let msg = open_error_without_panic(&path);
        assert_error_mentions(
            &msg,
            &["tensor unsupported", "Q1_0", "no supported block layout"],
        );
        std::fs::remove_dir_all(dir).ok();

        let mut h = raw_header(0, 1);
        put_test_string(&mut h, "general.alignment");
        h.extend_from_slice(&4u32.to_le_bytes());
        h.extend_from_slice(&0u32.to_le_bytes());
        let (dir, path) = write_raw_fixture("zero-alignment", &h);
        let msg = open_error_without_panic(&path);
        assert_error_mentions(&msg, &["general.alignment=0", "must be non-zero"]);
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn tensor_ranges_are_rejected_during_open_not_deferred_to_slicing() {
        let mut h = raw_header(1, 0);
        put_test_tensor(&mut h, "past-eof", &[1], GgmlType::F32 as u32, 0);
        let (dir, path) = write_raw_fixture("tensor-past-eof", &h);
        let msg = open_error_without_panic(&path);
        assert_error_mentions(&msg, &["tensor past-eof", "byte range", "file length"]);
        std::fs::remove_dir_all(dir).ok();

        let mut h = raw_header(1, 0);
        put_test_tensor(
            &mut h,
            "offset-overflow",
            &[1],
            GgmlType::F32 as u32,
            u64::MAX,
        );
        let (dir, path) = write_raw_fixture("tensor-offset-overflow", &h);
        let msg = open_error_without_panic(&path);
        assert_error_mentions(
            &msg,
            &[
                "tensor offset-overflow",
                "offset=18446744073709551615",
                "overflow",
            ],
        );
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn split_metadata_mismatches_are_errors_not_panics() {
        let dir = std::env::temp_dir().join(format!("memra-split-errors-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p0 = dir.join("m-00001-of-00002.gguf");
        let p1 = dir.join("m-00002-of-00002.gguf");
        write_gguf(
            &p0,
            &[
                ("split.no", MetaValue::U16(0)),
                ("split.count", MetaValue::U16(2)),
                ("split.tensors.count", MetaValue::I32(2)),
            ],
            &[("a", 1, 0x11)],
        );
        write_gguf(
            &p1,
            &[
                ("split.no", MetaValue::U16(0)),
                ("split.count", MetaValue::U16(2)),
                ("split.tensors.count", MetaValue::I32(2)),
            ],
            &[("b", 1, 0x22)],
        );
        let msg = open_error_without_panic(&p0);
        assert_error_mentions(&msg, &["split.no=0", "expected 1", "m-00002-of-00002.gguf"]);

        write_gguf(
            &p1,
            &[
                ("split.no", MetaValue::U16(1)),
                ("split.count", MetaValue::U16(3)),
                ("split.tensors.count", MetaValue::I32(2)),
            ],
            &[("b", 1, 0x22)],
        );
        let msg = open_error_without_panic(&p0);
        assert_error_mentions(
            &msg,
            &["split.count=3", "expected 2", "m-00002-of-00002.gguf"],
        );

        write_gguf(
            &p1,
            &[
                ("split.no", MetaValue::U16(1)),
                ("split.count", MetaValue::U16(2)),
                ("split.tensors.count", MetaValue::I32(3)),
            ],
            &[("b", 1, 0x22)],
        );
        write_gguf(
            &p0,
            &[
                ("split.no", MetaValue::U16(0)),
                ("split.count", MetaValue::U16(2)),
                ("split.tensors.count", MetaValue::I32(3)),
            ],
            &[("a", 1, 0x11)],
        );
        let msg = open_error_without_panic(&p0);
        assert_error_mentions(&msg, &["2 tensors", "split.tensors.count=3"]);
        std::fs::remove_dir_all(dir).ok();
    }
}
