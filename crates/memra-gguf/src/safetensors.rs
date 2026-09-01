//! Minimal safetensors reader, mmap-based. Parallel to the GGUF reader.
//!
//! On-disk layout (little-endian):
//!   header_len: u64 (8 bytes) | header_len bytes of UTF-8 JSON | raw tensor byte buffer
//!
//! Header JSON: { tensor_name: { "dtype": str, "shape": [usize], "data_offsets": [begin, end] }, ... }
//! plus an optional free-form "__metadata__" object. `data_offsets` are byte ranges INTO the
//! post-header buffer (which begins at byte `8 + header_len`), row-major.
//!
//! Multi-shard: a `model.safetensors.index.json` maps each tensor name to its shard file:
//!   { "metadata": {...}, "weight_map": { tensor_name -> "model-0000X-of-0000N.safetensors" } }
//!
//! NOTE: this module hand-parses the small header JSON (no serde dependency in memra-gguf).
//! The grammar we accept is exactly what `safetensors` emits: objects, arrays, strings, integers.

use memmap2::Mmap;
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::Read;
use std::path::{Component, Path, PathBuf};
use std::sync::Arc;

use crate::GgmlType;

const N_LEN: usize = 8; // size_of::<u64>()
const MAX_HEADER: usize = 100_000_000; // DoS guard (matches safetensors crate)
const MAX_INDEX_BYTES: usize = 100_000_000;
const MAX_INDEX_ENTRIES: usize = 1_000_000;
const MAX_INDEX_SHARDS: usize = 1024;
const MAX_HEADER_TENSORS: usize = 1_000_000;
const MAX_TENSOR_RANK: usize = 128;
const MAX_JSON_DEPTH: usize = 128;

/// Return the number of bits occupied by one value of a safetensors dtype.
///
/// This is deliberately broader than `st_dtype_to_ggml`: modelopt and DSV4 artifacts use
/// byte-oriented dtypes (U8/F8/BOOL) that do not have a resident `GgmlType` equivalent, but their
/// declared shape still has to agree with the uploaded extent before any consumer can derive a
/// CUDA launch geometry from it.
fn st_dtype_bits(dtype: &str) -> Option<usize> {
    match dtype {
        "F4" => Some(4),
        "F6_E2M3" | "F6_E3M2" => Some(6),
        "BOOL" | "U8" | "I8" | "F8_E4M3" | "F8_E5M2" | "F8_E8M0" | "F8_E4M3FNUZ"
        | "F8_E5M2FNUZ" => Some(8),
        "U16" | "I16" | "F16" | "BF16" => Some(16),
        "U32" | "I32" | "F32" => Some(32),
        "U64" | "I64" | "F64" | "C64" => Some(64),
        _ => None,
    }
}

/// Validate one header entry against the post-header byte buffer.
///
/// `raw` is intentionally zero-copy, so this check is the last safe point before callers can
/// turn an artifact-controlled shape/offset pair into device dimensions. Requiring the exact
/// dtype-sized extent rejects both truncated tensors (which would otherwise slice out of bounds)
/// and oversized/reversed ranges (which could make a later upload consume bytes belonging to a
/// different tensor or an unrelated file tail).
fn validate_tensor_extent(name: &str, info: &StInfo, payload_len: usize) -> std::io::Result<()> {
    let Some(value_bits) = st_dtype_bits(&info.dtype) else {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "tensor {name:?} has unsupported safetensors dtype {:?}",
                info.dtype
            ),
        ));
    };
    let elements = info.shape.iter().try_fold(1usize, |total, &dim| {
        let dim = usize::try_from(dim).map_err(|_| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("tensor {name:?} shape dimension {dim} does not fit this platform"),
            )
        })?;
        total.checked_mul(dim).ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("tensor {name:?} shape element count overflows this platform"),
            )
        })
    })?;
    let expected_bits = elements.checked_mul(value_bits).ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("tensor {name:?} byte length overflows this platform"),
        )
    })?;
    if expected_bits % 8 != 0 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "tensor {name:?} {:?} shape {:?} is not byte-aligned ({expected_bits} bits)",
                info.dtype, info.shape
            ),
        ));
    }
    let expected = expected_bits / 8;
    let [start, end] = info.data_offsets;
    let actual = end.checked_sub(start).ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("tensor {name:?} has reversed data_offsets [{start}, {end}]"),
        )
    })?;
    if end > payload_len || actual != expected {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "tensor {name:?} extent [{start}, {end}) is invalid for {:?} shape {:?}: expected {expected} bytes within {payload_len}, found {actual}",
                info.dtype, info.shape
            ),
        ));
    }
    Ok(())
}

fn validate_tensor_extents(
    infos: &HashMap<String, StInfo>,
    payload_len: usize,
) -> std::io::Result<()> {
    let mut ordered: Vec<_> = infos.iter().collect();
    ordered.sort_unstable_by_key(|(name, info)| {
        (info.data_offsets[0], info.data_offsets[1], name.as_str())
    });
    let mut cursor = 0usize;
    for (name, info) in ordered {
        if info.data_offsets[0] != cursor {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "tensor {name:?} starts at {}, but exact safetensors coverage requires {cursor}",
                    info.data_offsets[0]
                ),
            ));
        }
        validate_tensor_extent(name, info, payload_len)?;
        cursor = info.data_offsets[1];
    }
    if cursor != payload_len {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "safetensors metadata covers {cursor} payload bytes, but the file contains {payload_len}"
            ),
        ));
    }
    Ok(())
}

/// safetensors dtype string -> memra GgmlType. FP8 is deferred (explicit panic, never silent).
/// Shape note handled by the caller: safetensors shape is row-major outer..inner, memra `ne` is
/// inner-fastest, so the caller reverses the shape.
pub fn st_dtype_to_ggml(s: &str) -> GgmlType {
    use GgmlType::*;
    match s {
        "F32" => F32,
        "F16" => F16,
        "BF16" => BF16,
        "F64" => F64,
        "I8" => I8,
        "I16" => I16,
        "I32" => I32,
        "I64" => I64,
        // FP8 deferred in v1 — explicit failure, NOT silent garbage.
        "F8_E4M3" | "F8_E5M2" | "F8_E8M0" => {
            panic!(
                "FP8 ({s}) safetensors not yet supported; use the GGUF twin or an F16/BF16 checkpoint"
            )
        }
        // U8 / BOOL have no GgmlType equivalent and never appear as model weights here.
        other => panic!("unsupported safetensors dtype {other}"),
    }
}

/// One tensor's header entry.
#[derive(Debug, Clone)]
pub struct StInfo {
    pub dtype: String,
    pub shape: Vec<u64>,          // row-major, outer..inner (as stored)
    pub data_offsets: [usize; 2], // [begin, end) into the post-header buffer
}

fn huggingface_blob_root(root: &Path) -> Option<PathBuf> {
    let snapshots = root.parent()?;
    if snapshots.file_name()?.to_str()? != "snapshots" {
        return None;
    }
    let blobs = snapshots.parent()?.join("blobs");
    std::fs::canonicalize(blobs)
        .ok()
        .filter(|path| path.is_dir())
}

fn safe_index_path(root: &Path, blob_root: Option<&Path>, name: &str) -> std::io::Result<PathBuf> {
    let relative = Path::new(name);
    if relative.as_os_str().is_empty()
        || relative
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("safetensors index shard name is not a relative normal path: {name:?}"),
        ));
    }
    let candidate = root.join(relative);
    let resolved = std::fs::canonicalize(&candidate)?;
    if !resolved.starts_with(root) && !blob_root.is_some_and(|blobs| resolved.starts_with(blobs)) {
        return Err(std::io::Error::new(
            std::io::ErrorKind::PermissionDenied,
            format!(
                "safetensors index path escapes the model snapshot and its repository blobs: {name:?}"
            ),
        ));
    }
    Ok(resolved)
}

impl StInfo {
    /// memra `ne` (inner-fastest) is the reverse of the safetensors shape.
    pub fn ne(&self) -> Vec<u64> {
        self.shape.iter().rev().cloned().collect()
    }
    pub fn ggml_type(&self) -> GgmlType {
        st_dtype_to_ggml(&self.dtype)
    }
}

/// A single mmap'd safetensors shard.
pub struct StShard {
    mmap: Arc<Mmap>,
    file: Arc<File>,
    data_base: usize, // 8 + header_len
    infos: HashMap<String, StInfo>,
}

impl StShard {
    pub fn open<P: AsRef<Path>>(p: P) -> std::io::Result<Self> {
        let path = p.as_ref();
        let metadata = std::fs::symlink_metadata(path)?;
        if !metadata.file_type().is_file() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "safetensors shard is not a regular non-symlink file: {}",
                    path.display()
                ),
            ));
        }
        #[cfg(unix)]
        use std::os::unix::fs::OpenOptionsExt as _;
        let mut options = std::fs::OpenOptions::new();
        options.read(true);
        #[cfg(unix)]
        options.custom_flags(libc::O_NOFOLLOW);
        let file = Arc::new(options.open(path)?);
        let mmap = Arc::new(unsafe { Mmap::map(file.as_ref())? });
        Self::from_mmap(mmap, file)
    }

    fn from_mmap(mmap: Arc<Mmap>, file: Arc<File>) -> std::io::Result<Self> {
        if mmap.len() < N_LEN {
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "safetensors file too small for header length",
            ));
        }
        let hlen = u64::from_le_bytes(
            mmap.get(..N_LEN)
                .and_then(|bytes| bytes.try_into().ok())
                .ok_or_else(|| {
                    std::io::Error::new(
                        std::io::ErrorKind::UnexpectedEof,
                        "safetensors header length is truncated",
                    )
                })?,
        );
        let hlen = usize::try_from(hlen).map_err(|_| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "safetensors header length overflows usize",
            )
        })?;
        if hlen > MAX_HEADER || hlen > mmap.len().saturating_sub(N_LEN) {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "bad/oversized safetensors header (len={hlen}, file={})",
                    mmap.len()
                ),
            ));
        }
        let json = std::str::from_utf8(&mmap[N_LEN..N_LEN + hlen]).map_err(|_| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "safetensors header is not valid UTF-8",
            )
        })?;
        if json_exceeds_depth(json, MAX_JSON_DEPTH) {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("safetensors header nesting exceeds maximum depth {MAX_JSON_DEPTH}"),
            ));
        }
        let infos = parse_header_json_checked(json)
            .map_err(|message| std::io::Error::new(std::io::ErrorKind::InvalidData, message))?;
        let data_base = N_LEN + hlen;
        validate_tensor_extents(&infos, mmap.len() - data_base)?;
        Ok(Self {
            mmap,
            file,
            data_base,
            infos,
        })
    }

    /// Zero-copy bytes for a tensor (mirrors GgufFile::tensor_data).
    pub fn raw(&self, name: &str) -> Option<(&StInfo, &[u8])> {
        let i = self.infos.get(name)?;
        let s = self.data_base + i.data_offsets[0];
        let e = self.data_base + i.data_offsets[1];
        Some((i, &self.mmap[s..e]))
    }

    /// Owned whole-file mapping plus the absolute tensor byte range. Resident expert loaders retain
    /// both handles so the checkpoint bytes outlive the short-lived source object without a copy.
    pub fn raw_extent(&self, name: &str) -> Option<(Arc<Mmap>, Arc<File>, usize, usize)> {
        let info = self.infos.get(name)?;
        let start = self.data_base.checked_add(info.data_offsets[0])?;
        let end = self.data_base.checked_add(info.data_offsets[1])?;
        let len = end.checked_sub(start)?;
        (end <= self.mmap.len()).then(|| (self.mmap.clone(), self.file.clone(), start, len))
    }

    pub fn names(&self) -> impl Iterator<Item = &String> {
        self.infos.keys()
    }

    pub fn len(&self) -> usize {
        self.infos.len()
    }
    pub fn is_empty(&self) -> bool {
        self.infos.is_empty()
    }
}

/// A whole safetensors model: one or more shards, routed by tensor name.
pub struct StModel {
    shards: Vec<StShard>,
    map: HashMap<String, usize>, // tensor_name -> shard index
}

impl StModel {
    /// Open a model from a directory. Prefers `model.safetensors.index.json` (multi-shard);
    /// falls back to a single `model.safetensors`. Also accepts an explicit file path.
    pub fn open(path: &Path) -> std::io::Result<Self> {
        if path.is_file() {
            // An explicit caller-selected file may itself be an HF cache symlink. Resolve that
            // one declared path, then retain StShard's no-follow rule for index-selected files.
            let sh = StShard::open(std::fs::canonicalize(path)?)?;
            let map = sh.names().map(|n| (n.clone(), 0)).collect();
            return Ok(Self {
                shards: vec![sh],
                map,
            });
        }
        let dir = path;
        let root = std::fs::canonicalize(dir)?;
        if !root.is_dir() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!(
                    "safetensors model root is not a directory: {}",
                    dir.display()
                ),
            ));
        }
        let blob_root = huggingface_blob_root(&root);
        let idx = root.join("model.safetensors.index.json");
        if idx.exists() {
            let idx = safe_index_path(&root, blob_root.as_deref(), "model.safetensors.index.json")?;
            let idx_meta = std::fs::symlink_metadata(&idx)?;
            if !idx_meta.file_type().is_file() {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!(
                        "safetensors index is not a regular non-symlink file: {}",
                        idx.display()
                    ),
                ));
            }
            let idx_len = usize::try_from(idx_meta.len()).map_err(|_| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "safetensors index length overflows usize",
                )
            })?;
            if idx_len > MAX_INDEX_BYTES {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!(
                        "safetensors index is too large ({idx_len} bytes; maximum {MAX_INDEX_BYTES})"
                    ),
                ));
            }
            #[cfg(unix)]
            use std::os::unix::fs::OpenOptionsExt as _;
            let mut index_options = std::fs::OpenOptions::new();
            index_options.read(true);
            #[cfg(unix)]
            index_options.custom_flags(libc::O_NOFOLLOW);
            let index_file = index_options.open(&idx)?;
            let index_meta = index_file.metadata()?;
            if !index_meta.file_type().is_file() {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("safetensors index is not a regular file: {}", idx.display()),
                ));
            }
            let mut txt = String::new();
            index_file
                .take((MAX_INDEX_BYTES as u64).saturating_add(1))
                .read_to_string(&mut txt)?;
            if txt.len() > MAX_INDEX_BYTES {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("safetensors index exceeds {MAX_INDEX_BYTES} bytes while reading"),
                ));
            }
            if json_exceeds_depth(&txt, MAX_JSON_DEPTH) {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("safetensors index nesting exceeds maximum depth {MAX_JSON_DEPTH}"),
                ));
            }
            let weight_map = parse_index_weight_map_json_checked(&txt)
                .map_err(|message| std::io::Error::new(std::io::ErrorKind::InvalidData, message))?;
            // distinct shard file names, in stable sorted order
            let mut files: Vec<String> = weight_map.values().cloned().collect();
            files.sort();
            files.dedup();
            if files.len() > MAX_INDEX_SHARDS {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!(
                        "safetensors index references too many shards ({}; maximum {MAX_INDEX_SHARDS})",
                        files.len()
                    ),
                ));
            }
            let pos: HashMap<&String, usize> =
                files.iter().enumerate().map(|(n, f)| (f, n)).collect();
            let shard_paths: Vec<PathBuf> = files
                .iter()
                .map(|file| safe_index_path(&root, blob_root.as_deref(), file))
                .collect::<Result<Vec<_>, _>>()?;
            let shards = shard_paths
                .iter()
                .map(StShard::open)
                .collect::<Result<Vec<_>, _>>()?;
            for (tensor, file) in &weight_map {
                let si = pos[file];
                // Index validation is metadata-only. Do not form a payload slice merely to prove
                // the owning shard header contains the indexed name.
                if !shards[si].infos.contains_key(tensor) {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!(
                            "safetensors index maps tensor {tensor:?} to {file:?}, but that shard \
                             does not contain it"
                        ),
                    ));
                }
            }
            let mut map: HashMap<String, usize> = weight_map
                .iter()
                .map(|(t, f)| (t.clone(), pos[f]))
                .collect();
            // Some checkpoints omit quantization sidecars from weight_map even though the shard
            // header owns them. Include those authoritative names, but reject cross-shard
            // ambiguity rather than silently choosing file order.
            for (si, shard) in shards.iter().enumerate() {
                for name in shard.names() {
                    match map.get(name).copied() {
                        Some(indexed) if indexed == si && weight_map.contains_key(name) => {}
                        Some(existing) => {
                            return Err(std::io::Error::new(
                                std::io::ErrorKind::InvalidData,
                                format!(
                                    "tensor {name:?} appears in multiple safetensors shards \
                                     ({:?} and {:?})",
                                    files[existing], files[si],
                                ),
                            ));
                        }
                        None => {
                            map.insert(name.clone(), si);
                        }
                    }
                }
            }
            Ok(Self { shards, map })
        } else {
            let single = safe_index_path(&root, blob_root.as_deref(), "model.safetensors")?;
            let sh = StShard::open(single)?;
            let map = sh.names().map(|n| (n.clone(), 0)).collect();
            Ok(Self {
                shards: vec![sh],
                map,
            })
        }
    }

    /// Zero-copy bytes + header info for a tensor, routed to the owning shard.
    pub fn raw(&self, name: &str) -> Option<(&StInfo, &[u8])> {
        let &si = self.map.get(name)?;
        self.shards[si].raw(name)
    }

    /// Header-only metadata for a tensor, routed to the owning shard.
    ///
    /// Unlike [`Self::raw`], this never forms a view into the tensor payload. Placement and
    /// inspection code use it to census exact physical byte ranges without faulting weight pages
    /// or invoking any source transform.
    pub fn info(&self, name: &str) -> Option<&StInfo> {
        let &si = self.map.get(name)?;
        self.shards[si].infos.get(name)
    }

    /// Owned extent for the same tensor selected by `raw`.
    pub fn raw_extent(&self, name: &str) -> Option<(Arc<Mmap>, Arc<File>, usize, usize)> {
        let &si = self.map.get(name)?;
        self.shards[si].raw_extent(name)
    }

    /// All tensor names across all shards.
    pub fn names(&self) -> impl Iterator<Item = &String> {
        self.map.keys()
    }

    pub fn n_tensors(&self) -> usize {
        self.map.len()
    }
}

fn json_exceeds_depth(text: &str, max: usize) -> bool {
    let mut depth = 0usize;
    let mut in_string = false;
    let mut escaped = false;
    for byte in text.bytes() {
        if in_string {
            if escaped {
                escaped = false;
            } else if byte == b'\\' {
                escaped = true;
            } else if byte == b'"' {
                in_string = false;
            }
            continue;
        }
        match byte {
            b'"' => in_string = true,
            b'{' | b'[' => {
                depth = depth.saturating_add(1);
                if depth > max {
                    return true;
                }
            }
            b'}' | b']' => depth = depth.saturating_sub(1),
            _ => {}
        }
    }
    false
}

// ============================ minimal JSON parsing (header + index) ============================
//
// We only parse the exact shapes the safetensors writer emits. This avoids adding a serde
// dependency to memra-gguf (which currently has only `memmap2`). Tolerant of whitespace.

struct Json<'a> {
    b: &'a [u8],
    i: usize,
    failed: bool,
}

impl<'a> Json<'a> {
    fn new(s: &'a str) -> Self {
        Self {
            b: s.as_bytes(),
            i: 0,
            failed: false,
        }
    }
    fn skip_ws(&mut self) {
        while self.i < self.b.len() && matches!(self.b[self.i], b' ' | b'\t' | b'\n' | b'\r') {
            self.i += 1;
        }
    }
    fn peek(&mut self) -> u8 {
        self.skip_ws();
        self.b.get(self.i).copied().unwrap_or_else(|| {
            self.failed = true;
            0
        })
    }
    fn eat(&mut self, c: u8) {
        self.skip_ws();
        if self.b.get(self.i).copied() != Some(c) {
            self.failed = true;
            return;
        }
        self.i = self.i.saturating_add(1);
    }
    /// Parse a JSON string (no escape handling beyond \" and \\, which suffices for tensor names).
    fn string(&mut self) -> String {
        self.skip_ws();
        if self.b.get(self.i).copied() != Some(b'"') {
            self.failed = true;
            return String::new();
        }
        self.i = self.i.saturating_add(1);
        let mut out = String::new();
        let mut terminated = false;
        while self.i < self.b.len() {
            let c = self.b[self.i];
            self.i += 1;
            match c {
                b'"' => {
                    terminated = true;
                    break;
                }
                b'\\' => {
                    let Some(&e) = self.b.get(self.i) else {
                        self.failed = true;
                        break;
                    };
                    self.i += 1;
                    out.push(match e {
                        b'"' => '"',
                        b'\\' => '\\',
                        b'/' => '/',
                        b'n' => '\n',
                        b't' => '\t',
                        b'r' => '\r',
                        other => other as char,
                    });
                }
                _ => out.push(c as char),
            }
        }
        if !terminated {
            self.failed = true;
        }
        out
    }
    /// Parse a non-negative integer (offsets/shape are always >= 0 in safetensors).
    fn u64(&mut self) -> u64 {
        self.skip_ws();
        let start = self.i;
        while self.i < self.b.len() && self.b[self.i].is_ascii_digit() {
            self.i += 1;
        }
        if self.i == start {
            self.failed = true;
            return 0;
        }
        match std::str::from_utf8(&self.b[start..self.i])
            .ok()
            .and_then(|value| value.parse().ok())
        {
            Some(value) => value,
            None => {
                self.failed = true;
                0
            }
        }
    }
    /// Skip an arbitrary JSON value (used for "__metadata__" and index "metadata").
    fn skip_value(&mut self) {
        self.skip_value_depth(0);
    }

    /// The safetensors `__metadata__` contract is string-to-string, not arbitrary JSON.
    fn skip_string_map(&mut self) {
        self.eat(b'{');
        let mut keys = HashSet::new();
        if self.peek() != b'}' {
            loop {
                let key = self.string();
                if !keys.insert(key) {
                    self.failed = true;
                    return;
                }
                self.eat(b':');
                let _ = self.string();
                if self.peek() == b',' {
                    self.eat(b',');
                } else {
                    break;
                }
            }
        }
        self.eat(b'}');
    }

    fn skip_value_depth(&mut self, depth: usize) {
        if self.failed {
            return;
        }
        if depth > MAX_JSON_DEPTH {
            self.failed = true;
            return;
        }
        match self.peek() {
            b'{' => {
                self.eat(b'{');
                if self.peek() != b'}' {
                    loop {
                        let _ = self.string();
                        self.eat(b':');
                        self.skip_value_depth(depth + 1);
                        if self.peek() == b',' {
                            self.eat(b',');
                        } else {
                            break;
                        }
                    }
                }
                self.eat(b'}');
            }
            b'[' => {
                self.eat(b'[');
                if self.peek() != b']' {
                    loop {
                        self.skip_value_depth(depth + 1);
                        if self.peek() == b',' {
                            self.eat(b',');
                        } else {
                            break;
                        }
                    }
                }
                self.eat(b']');
            }
            b'"' => {
                let _ = self.string();
            }
            _ => {
                // number, true, false, or null
                self.skip_ws();
                let start = self.i;
                while self.i < self.b.len()
                    && !matches!(
                        self.b[self.i],
                        b',' | b'}' | b']' | b' ' | b'\t' | b'\n' | b'\r'
                    )
                {
                    self.i += 1;
                }
                let token = &self.b[start..self.i];
                if token.is_empty()
                    || (token != b"true"
                        && token != b"false"
                        && token != b"null"
                        && !valid_json_number(token))
                {
                    self.failed = true;
                }
            }
        }
    }
}

fn valid_json_number(token: &[u8]) -> bool {
    let mut i = 0usize;
    if token.get(i) == Some(&b'-') {
        i += 1;
    }
    match token.get(i) {
        Some(b'0') => i += 1,
        Some(b'1'..=b'9') => {
            i += 1;
            while token.get(i).is_some_and(u8::is_ascii_digit) {
                i += 1;
            }
        }
        _ => return false,
    }
    if token.get(i) == Some(&b'.') {
        i += 1;
        let start = i;
        while token.get(i).is_some_and(u8::is_ascii_digit) {
            i += 1;
        }
        if i == start {
            return false;
        }
    }
    if matches!(token.get(i), Some(b'e' | b'E')) {
        i += 1;
        if matches!(token.get(i), Some(b'+' | b'-')) {
            i += 1;
        }
        let start = i;
        while token.get(i).is_some_and(u8::is_ascii_digit) {
            i += 1;
        }
        if i == start {
            return false;
        }
    }
    i == token.len()
}

/// Parse a safetensors header already obtained from a local file or an HTTP range request.
/// No tensor payload bytes are required. The public compatibility wrapper returns an empty map
/// for malformed input; file-backed callers use the checked variant below so malformed model
/// artifacts become ordinary load errors rather than panics.
pub fn parse_header_json(json: &str) -> HashMap<String, StInfo> {
    parse_header_json_checked(json).unwrap_or_default()
}

pub fn parse_header_json_checked(json: &str) -> Result<HashMap<String, StInfo>, String> {
    let mut p = Json::new(json);
    let mut out = HashMap::new();
    let mut top_keys = HashSet::new();
    p.eat(b'{');
    if p.peek() == b'}' {
        p.eat(b'}');
        p.skip_ws();
        return if p.failed || p.i != p.b.len() {
            Err("invalid safetensors header JSON".into())
        } else {
            Ok(out)
        };
    }
    loop {
        if out.len() >= MAX_HEADER_TENSORS {
            return Err(format!(
                "safetensors header has more than {MAX_HEADER_TENSORS} tensors"
            ));
        }
        let key = p.string();
        if !top_keys.insert(key.clone()) {
            return Err(format!("duplicate safetensors header key {key:?}"));
        }
        p.eat(b':');
        if key == "__metadata__" {
            p.skip_string_map();
            if p.failed {
                return Err("safetensors __metadata__ must be a string-to-string map".into());
            }
        } else {
            // { "dtype": "...", "shape": [...], "data_offsets": [a,b] } — fields in any order.
            p.eat(b'{');
            let mut dtype = None;
            let mut shape = None;
            let mut offsets = None;
            let mut fields = HashSet::new();
            loop {
                let field = p.string();
                if !fields.insert(field.clone()) {
                    return Err(format!(
                        "tensor {key:?} repeats safetensors field {field:?}"
                    ));
                }
                p.eat(b':');
                match field.as_str() {
                    "dtype" => dtype = Some(p.string()),
                    "shape" => {
                        p.eat(b'[');
                        let mut parsed = Vec::new();
                        if p.peek() != b']' {
                            loop {
                                if parsed.len() >= MAX_TENSOR_RANK {
                                    return Err(format!(
                                        "safetensors tensor rank exceeds {MAX_TENSOR_RANK}"
                                    ));
                                }
                                parsed.push(p.u64());
                                if p.peek() == b',' {
                                    p.eat(b',');
                                } else {
                                    break;
                                }
                            }
                        }
                        p.eat(b']');
                        shape = Some(parsed);
                    }
                    "data_offsets" => {
                        p.eat(b'[');
                        let start = usize::try_from(p.u64())
                            .map_err(|_| format!("tensor {key:?} start offset overflows usize"))?;
                        p.eat(b',');
                        let end = usize::try_from(p.u64())
                            .map_err(|_| format!("tensor {key:?} end offset overflows usize"))?;
                        p.eat(b']');
                        offsets = Some([start, end]);
                    }
                    _ => return Err(format!("tensor {key:?} has unknown field {field:?}")),
                }
                if p.peek() == b',' {
                    p.eat(b',');
                } else {
                    break;
                }
            }
            p.eat(b'}');
            if p.failed {
                return Err(format!("invalid metadata JSON for tensor {key:?}"));
            }
            let dtype = dtype.ok_or_else(|| format!("tensor {key:?} is missing dtype"))?;
            let shape = shape.ok_or_else(|| format!("tensor {key:?} is missing shape"))?;
            let data_offsets =
                offsets.ok_or_else(|| format!("tensor {key:?} is missing data_offsets"))?;
            out.insert(
                key,
                StInfo {
                    dtype,
                    shape,
                    data_offsets,
                },
            );
        }
        if p.peek() == b',' {
            p.eat(b',');
        } else {
            break;
        }
    }
    p.eat(b'}');
    p.skip_ws();
    if p.failed || p.i != p.b.len() {
        return Err("invalid safetensors header JSON".into());
    }
    Ok(out)
}

/// Parse `model.safetensors.index.json` into tensor-to-shard ownership.
pub fn parse_index_weight_map_json(json: &str) -> HashMap<String, String> {
    parse_index_weight_map_json_checked(json).unwrap_or_default()
}

pub fn parse_index_weight_map_json_checked(json: &str) -> Result<HashMap<String, String>, String> {
    let mut p = Json::new(json);
    let mut out = HashMap::new();
    let mut top_keys = HashSet::new();
    let mut saw_weight_map = false;
    p.eat(b'{');
    loop {
        let key = p.string();
        if !top_keys.insert(key.clone()) {
            return Err(format!("duplicate safetensors index key {key:?}"));
        }
        p.eat(b':');
        if key == "weight_map" {
            saw_weight_map = true;
            p.eat(b'{');
            if p.peek() != b'}' {
                loop {
                    let t = p.string();
                    p.eat(b':');
                    let f = p.string();
                    if out.len() >= MAX_INDEX_ENTRIES && !out.contains_key(&t) {
                        return Err(format!(
                            "safetensors index has more than {MAX_INDEX_ENTRIES} tensor entries"
                        ));
                    }
                    if out.contains_key(&t) {
                        return Err(format!("duplicate safetensors index tensor {t:?}"));
                    }
                    out.insert(t, f);
                    if p.peek() == b',' {
                        p.eat(b',');
                    } else {
                        break;
                    }
                }
            }
            p.eat(b'}');
        } else {
            p.skip_value();
            if p.failed {
                return Err("invalid safetensors index JSON".into());
            }
        }
        if p.peek() == b',' {
            p.eat(b',');
        } else {
            break;
        }
    }
    p.eat(b'}');
    p.skip_ws();
    if p.failed || p.i != p.b.len() {
        return Err("invalid safetensors index JSON".into());
    }
    if !saw_weight_map || out.is_empty() {
        return Err("safetensors index weight_map must contain at least one tensor".into());
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a tiny valid safetensors byte buffer by hand and parse it back.
    fn build_synthetic() -> Vec<u8> {
        // two tensors:
        //   "a.weight" F32 shape [2,3] -> 6 f32 = 24 bytes at [0,24)
        //   "b.norm"   BF16 shape [3]  -> 3 bf16 = 6 bytes  at [24,30)
        let json = r#"{"__metadata__":{"format":"pt"},"a.weight":{"dtype":"F32","shape":[2,3],"data_offsets":[0,24]},"b.norm":{"dtype":"BF16","shape":[3],"data_offsets":[24,30]}}"#;
        let jb = json.as_bytes();
        let mut buf = Vec::new();
        buf.extend_from_slice(&(jb.len() as u64).to_le_bytes());
        buf.extend_from_slice(jb);
        // tensor data: a.weight = [0,1,2,3,4,5] f32
        for v in 0..6u32 {
            buf.extend_from_slice(&(v as f32).to_le_bytes());
        }
        // b.norm = three bf16 (1.0, 2.0, -1.0) = 0x3F80, 0x4000, 0xBF80
        for v in [0x3F80u16, 0x4000, 0xBF80] {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        buf
    }

    fn write_temp(name: &str, bytes: &[u8]) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "memra_st_test_{}_{}.safetensors",
            std::process::id(),
            name
        ));
        std::fs::write(&p, bytes).unwrap();
        p
    }

    #[test]
    fn synthetic_header_roundtrip() {
        let bytes = build_synthetic();
        let p = write_temp("rt", &bytes);
        let sh = StShard::open(&p).unwrap();
        assert_eq!(sh.len(), 2, "two tensors (metadata dropped)");

        let (ia, ra) = sh.raw("a.weight").expect("a.weight");
        assert_eq!(ia.dtype, "F32");
        assert_eq!(ia.shape, vec![2, 3]);
        // shape-reversal: ne = [3,2] -> in_features=ne[0]=3, out_features=ne[1]=2
        assert_eq!(ia.ne(), vec![3, 2]);
        assert_eq!(ia.ggml_type(), GgmlType::F32);
        assert_eq!(ra.len(), 24);
        let f0 = f32::from_le_bytes(ra[0..4].try_into().unwrap());
        let f5 = f32::from_le_bytes(ra[20..24].try_into().unwrap());
        assert_eq!(f0, 0.0);
        assert_eq!(f5, 5.0);

        let (ib, rb) = sh.raw("b.norm").expect("b.norm");
        assert_eq!(ib.dtype, "BF16");
        assert_eq!(ib.shape, vec![3]);
        assert_eq!(ib.ne(), vec![3]);
        assert_eq!(rb.len(), 6);
        // bytes honored at the right offset
        assert_eq!(u16::from_le_bytes(rb[0..2].try_into().unwrap()), 0x3F80);

        std::fs::remove_file(&p).ok();
    }

    #[test]
    fn rejects_tensor_extents_that_disagree_with_shape_or_payload() {
        let cases = [
            (
                "short",
                r#"{"w":{"dtype":"F32","shape":[2],"data_offsets":[0,4]}}"#,
                4usize,
            ),
            (
                "long",
                r#"{"w":{"dtype":"F32","shape":[2],"data_offsets":[0,12]}}"#,
                12usize,
            ),
            (
                "reversed",
                r#"{"w":{"dtype":"F32","shape":[2],"data_offsets":[8,0]}}"#,
                8usize,
            ),
            (
                "oob",
                r#"{"w":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#,
                4usize,
            ),
        ];
        for (name, header, data_len) in cases {
            let mut bytes = Vec::new();
            bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
            bytes.extend_from_slice(header.as_bytes());
            bytes.extend(std::iter::repeat_n(0u8, data_len));
            let path = write_temp(&format!("extent_{name}"), &bytes);
            let error = match StShard::open(&path) {
                Ok(_) => panic!("invalid extent was accepted"),
                Err(error) => error,
            };
            assert_eq!(error.kind(), std::io::ErrorKind::InvalidData, "{error}");
            assert!(error.to_string().contains("tensor"), "{error}");
            std::fs::remove_file(path).ok();
        }
    }

    #[test]
    fn accepts_current_spec_subbyte_fnuz_and_complex_dtypes() {
        let json = r#"{"f4":{"dtype":"F4","shape":[2],"data_offsets":[0,1]},"f6":{"dtype":"F6_E2M3","shape":[4],"data_offsets":[1,4]},"fnuz":{"dtype":"F8_E4M3FNUZ","shape":[1],"data_offsets":[4,5]},"complex":{"dtype":"C64","shape":[1],"data_offsets":[5,13]}}"#;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(json.len() as u64).to_le_bytes());
        bytes.extend_from_slice(json.as_bytes());
        bytes.extend_from_slice(&[0u8; 13]);
        let path = write_temp("current_dtypes", &bytes);
        let shard = StShard::open(&path).unwrap();
        assert_eq!(shard.len(), 4);
        std::fs::remove_file(path).ok();

        let json = r#"{"misaligned":{"dtype":"F4","shape":[1],"data_offsets":[0,1]}}"#;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(json.len() as u64).to_le_bytes());
        bytes.extend_from_slice(json.as_bytes());
        bytes.push(0);
        let path = write_temp("misaligned_subbyte", &bytes);
        let error = match StShard::open(&path) {
            Ok(_) => panic!("misaligned sub-byte tensor was accepted"),
            Err(error) => error,
        };
        assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn rejects_overlaps_gaps_and_unclaimed_payload_tail() {
        let cases = [
            (
                "overlap",
                r#"{"a":{"dtype":"U8","shape":[2],"data_offsets":[0,2]},"b":{"dtype":"U8","shape":[2],"data_offsets":[1,3]}}"#,
                3usize,
            ),
            (
                "gap",
                r#"{"a":{"dtype":"U8","shape":[1],"data_offsets":[0,1]},"b":{"dtype":"U8","shape":[1],"data_offsets":[2,3]}}"#,
                3usize,
            ),
            (
                "tail",
                r#"{"a":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}"#,
                2usize,
            ),
        ];
        for (name, header, data_len) in cases {
            let mut bytes = Vec::new();
            bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
            bytes.extend_from_slice(header.as_bytes());
            bytes.extend(std::iter::repeat_n(0u8, data_len));
            let path = write_temp(&format!("coverage_{name}"), &bytes);
            let error = match StShard::open(&path) {
                Ok(_) => panic!("invalid {name} coverage was accepted"),
                Err(error) => error,
            };
            assert_eq!(error.kind(), std::io::ErrorKind::InvalidData, "{name}");
            std::fs::remove_file(path).ok();
        }
    }

    #[test]
    fn synthetic_dequant_through_path() {
        // round-trip the bytes back through the shared dequant path: F32 verbatim, BF16 known.
        let bytes = build_synthetic();
        let p = write_temp("dq", &bytes);
        let sh = StShard::open(&p).unwrap();
        let (ia, ra) = sh.raw("a.weight").unwrap();
        let av = crate::dequant::dequantize(ia.ggml_type(), ra, 6);
        assert_eq!(av, vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
        let (ib, rb) = sh.raw("b.norm").unwrap();
        let bv = crate::dequant::dequantize(ib.ggml_type(), rb, 3);
        assert_eq!(bv, vec![1.0, 2.0, -1.0]);
        std::fs::remove_file(&p).ok();
    }

    #[test]
    fn single_file_model_open() {
        let bytes = build_synthetic();
        let p = write_temp("single", &bytes);
        let m = StModel::open(&p).unwrap(); // explicit file path
        assert_eq!(m.n_tensors(), 2);
        let (_, ra) = m.raw("a.weight").expect("routed");
        assert_eq!(ra.len(), 24);
        assert!(m.raw("missing").is_none());
        std::fs::remove_file(&p).ok();
    }

    #[test]
    fn malformed_safetensors_headers_return_errors_without_panicking() {
        let cases = [
            Vec::new(),
            vec![0u8; 3],
            16u64.to_le_bytes().to_vec(),
            {
                let mut bytes = 1u64.to_le_bytes().to_vec();
                bytes.push(0xff);
                bytes
            },
            {
                let json = br#"{"broken":{"dtype":"F32""#;
                let mut bytes = (json.len() as u64).to_le_bytes().to_vec();
                bytes.extend_from_slice(json);
                bytes
            },
        ];
        for (index, bytes) in cases.into_iter().enumerate() {
            let path = write_temp(&format!("malformed-{index}"), &bytes);
            let result = std::panic::catch_unwind(|| StShard::open(&path));
            assert!(result.is_ok(), "malformed header panicked for case {index}");
            assert!(
                result.unwrap().is_err(),
                "malformed header was accepted for case {index}"
            );
            std::fs::remove_file(path).ok();
        }
    }

    #[test]
    fn checked_parsers_enforce_required_unique_and_metadata_contracts() {
        let bad_headers = [
            r#"{"w":{"dtype":"F32","data_offsets":[0,4]}}"#,
            r#"{"w":{"shape":[],"data_offsets":[0,4]}}"#,
            r#"{"w":{"dtype":"F32","shape":[]}}"#,
            r#"{"__metadata__":{"format":1},"w":{"dtype":"F32","shape":[],"data_offsets":[0,4]}}"#,
            r#"{"__metadata__":,"w":{"dtype":"F32","shape":[],"data_offsets":[0,4]}}"#,
            r#"{"w":{"dtype":"F32","shape":[],"data_offsets":[0,4]},"w":{"dtype":"F32","shape":[],"data_offsets":[0,4]}}"#,
            r#"{"w":{"dtype":"F32","dtype":"F16","shape":[],"data_offsets":[0,4]}}"#,
            r#"{"w":{"dtype":"F32","shape":[],"data_offsets":[0,4],"stride":1}}"#,
        ];
        for header in bad_headers {
            assert!(
                parse_header_json_checked(header).is_err(),
                "non-conforming header was accepted: {header}"
            );
        }

        let bad_indices = [
            r#"{"metadata":{"total_size":4}}"#,
            r#"{"weight_map":{}}"#,
            r#"{"metadata":,"weight_map":{"w":"model.safetensors"}}"#,
            r#"{"weight_map":{"w":"a.safetensors","w":"b.safetensors"}}"#,
            r#"{"weight_map":{"w":"a.safetensors"},"weight_map":{"x":"b.safetensors"}}"#,
        ];
        for index in bad_indices {
            assert!(
                parse_index_weight_map_json_checked(index).is_err(),
                "non-conforming index was accepted: {index}"
            );
        }
    }

    #[test]
    fn multi_shard_index_routing() {
        // Two shards, an index.json mapping each tensor to its file. Assert routing.
        let dir = std::env::temp_dir().join(format!("memra_st_idx_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();

        // shard 0: one F32 tensor
        let j0 = r#"{"x":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#;
        let mut s0 = Vec::new();
        s0.extend_from_slice(&(j0.len() as u64).to_le_bytes());
        s0.extend_from_slice(j0.as_bytes());
        s0.extend_from_slice(&7.0f32.to_le_bytes());
        s0.extend_from_slice(&8.0f32.to_le_bytes());
        std::fs::write(dir.join("model-00001-of-00002.safetensors"), &s0).unwrap();

        // shard 1: one F32 tensor
        let j1 = r#"{"y":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
        let mut s1 = Vec::new();
        s1.extend_from_slice(&(j1.len() as u64).to_le_bytes());
        s1.extend_from_slice(j1.as_bytes());
        s1.extend_from_slice(&9.0f32.to_le_bytes());
        std::fs::write(dir.join("model-00002-of-00002.safetensors"), &s1).unwrap();

        let idx = r#"{"metadata":{"total_size":12},"weight_map":{"x":"model-00001-of-00002.safetensors","y":"model-00002-of-00002.safetensors"}}"#;
        std::fs::write(dir.join("model.safetensors.index.json"), idx).unwrap();

        let m = StModel::open(&dir).unwrap();
        assert_eq!(m.n_tensors(), 2);
        let (_, rx) = m.raw("x").expect("x routes to shard 0");
        assert_eq!(f32::from_le_bytes(rx[0..4].try_into().unwrap()), 7.0);
        let (_, ry) = m.raw("y").expect("y routes to shard 1");
        assert_eq!(f32::from_le_bytes(ry[0..4].try_into().unwrap()), 9.0);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn model_index_rejects_absolute_parent_and_symlink_shard_paths() {
        let dir =
            std::env::temp_dir().join(format!("memra_st_index_escape_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let outside = dir.parent().unwrap().join(format!(
            "memra_st_escape_{}.safetensors",
            std::process::id()
        ));
        std::fs::write(&outside, build_synthetic()).unwrap();

        for name in [
            "../memra_st_escape_999.safetensors",
            outside.to_str().unwrap(),
        ] {
            std::fs::write(
                dir.join("model.safetensors.index.json"),
                format!(r#"{{"weight_map":{{"a.weight":{name:?}}}}}"#),
            )
            .unwrap();
            let error = match StModel::open(&dir) {
                Ok(_) => panic!("index path escaped the model root"),
                Err(error) => error,
            };
            assert!(
                error.to_string().contains("relative normal path")
                    || error.to_string().contains("escapes model root"),
                "unexpected error: {error}"
            );
        }

        #[cfg(unix)]
        {
            std::os::unix::fs::symlink(&outside, dir.join("linked.safetensors")).unwrap();
            std::fs::write(
                dir.join("model.safetensors.index.json"),
                r#"{"weight_map":{"a.weight":"linked.safetensors"}}"#,
            )
            .unwrap();
            let error = match StModel::open(&dir) {
                Ok(_) => panic!("symlink shard was accepted"),
                Err(error) => error,
            };
            assert!(
                error.to_string().contains("regular non-symlink")
                    || error.to_string().contains("escapes the model snapshot"),
                "{error}"
            );
        }
        std::fs::remove_file(outside).ok();
        std::fs::remove_dir_all(&dir).ok();
    }

    #[cfg(unix)]
    #[test]
    fn huggingface_snapshot_symlinks_are_confined_to_same_repository_blobs() {
        use std::os::unix::fs::symlink;

        let repo = std::env::temp_dir().join(format!(
            "models--memra--hf-symlink-fixture-{}",
            std::process::id()
        ));
        let snapshot = repo.join("snapshots").join("pinned-revision");
        let blobs = repo.join("blobs");
        std::fs::create_dir_all(&snapshot).unwrap();
        std::fs::create_dir_all(&blobs).unwrap();

        std::fs::write(blobs.join("shard-hash"), build_synthetic()).unwrap();
        std::fs::write(
            blobs.join("index-hash"),
            r#"{"weight_map":{"a.weight":"model-00001.safetensors","b.norm":"model-00001.safetensors"}}"#,
        )
        .unwrap();
        symlink(
            "../../blobs/index-hash",
            snapshot.join("model.safetensors.index.json"),
        )
        .unwrap();
        symlink(
            "../../blobs/shard-hash",
            snapshot.join("model-00001.safetensors"),
        )
        .unwrap();

        let model = StModel::open(&snapshot).unwrap();
        assert_eq!(model.n_tensors(), 2);
        assert!(model.raw("a.weight").is_some());
        assert!(model.raw("b.norm").is_some());
        std::fs::remove_dir_all(repo).ok();
    }

    #[test]
    fn multi_shard_header_sidecar_survives_an_incomplete_index() {
        let dir = std::env::temp_dir().join(format!("memra_st_sidecar_idx_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let header = r#"{
          "w":{"dtype":"F8_E4M3","shape":[1],"data_offsets":[0,1]},
          "w_scale_inv":{"dtype":"F32","shape":[1],"data_offsets":[1,5]}
        }"#;
        let mut shard = Vec::new();
        shard.extend_from_slice(&(header.len() as u64).to_le_bytes());
        shard.extend_from_slice(header.as_bytes());
        shard.push(0x38);
        shard.extend_from_slice(&0.25f32.to_le_bytes());
        std::fs::write(dir.join("model-00001.safetensors"), shard).unwrap();
        std::fs::write(
            dir.join("model.safetensors.index.json"),
            r#"{"metadata":{"total_size":5},"weight_map":{"w":"model-00001.safetensors"}}"#,
        )
        .unwrap();

        let model = StModel::open(&dir).unwrap();
        assert_eq!(model.n_tensors(), 2);
        let (info, raw) = model
            .raw("w_scale_inv")
            .expect("header-only sidecar must route");
        assert_eq!(info.dtype, "F32");
        assert_eq!(f32::from_le_bytes(raw.try_into().unwrap()), 0.25);
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn multi_shard_index_refuses_missing_and_duplicate_tensor_ownership() {
        fn shard(entries: &[(&str, f32)]) -> Vec<u8> {
            let mut offset = 0usize;
            let fields = entries
                .iter()
                .map(|(name, _)| {
                    let field = format!(
                        r#""{name}":{{"dtype":"F32","shape":[1],"data_offsets":[{offset},{}]}}"#,
                        offset + 4,
                    );
                    offset += 4;
                    field
                })
                .collect::<Vec<_>>()
                .join(",");
            let header = format!("{{{fields}}}");
            let mut bytes = Vec::new();
            bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
            bytes.extend_from_slice(header.as_bytes());
            for (_, value) in entries {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
            bytes
        }

        let missing =
            std::env::temp_dir().join(format!("memra_st_missing_indexed_{}", std::process::id()));
        std::fs::create_dir_all(&missing).unwrap();
        std::fs::write(
            missing.join("model-00001.safetensors"),
            shard(&[("actual", 1.0)]),
        )
        .unwrap();
        std::fs::write(
            missing.join("model.safetensors.index.json"),
            r#"{"weight_map":{"declared":"model-00001.safetensors"}}"#,
        )
        .unwrap();
        let error = match StModel::open(&missing) {
            Ok(_) => panic!("missing indexed tensor was accepted"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("does not contain it"), "{error}");
        std::fs::remove_dir_all(&missing).ok();

        let duplicate = std::env::temp_dir().join(format!(
            "memra_st_duplicate_cross_shard_{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&duplicate).unwrap();
        std::fs::write(
            duplicate.join("model-00001.safetensors"),
            shard(&[("x", 1.0), ("shared_scale", 0.25)]),
        )
        .unwrap();
        std::fs::write(
            duplicate.join("model-00002.safetensors"),
            shard(&[("y", 2.0), ("shared_scale", 0.5)]),
        )
        .unwrap();
        std::fs::write(
            duplicate.join("model.safetensors.index.json"),
            r#"{"weight_map":{
              "x":"model-00001.safetensors",
              "y":"model-00002.safetensors"
            }}"#,
        )
        .unwrap();
        let error = match StModel::open(&duplicate) {
            Ok(_) => panic!("cross-shard duplicate tensor was accepted"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("appears in multiple"), "{error}");
        std::fs::remove_dir_all(&duplicate).ok();
    }

    #[test]
    #[should_panic(expected = "FP8")]
    fn fp8_panics_explicitly() {
        st_dtype_to_ggml("F8_E4M3");
    }

    /// Real on-disk safetensors: parse a multi-shard HF checkpoint and assert known tensor
    /// shapes/dtypes/offsets round-trip. Skipped (not failed) when the model is absent.
    /// Point MEMRA_ST_TEST_DIR at an HF model dir (with model.safetensors.index.json) to run.
    /// Defaults to the Qwen3-1.7B snapshot present on this dev box.
    #[test]
    fn real_qwen3_17b_header() {
        let dir = std::env::var("MEMRA_ST_TEST_DIR").unwrap_or_else(|_| {
            "/data/ai-ml/hf-models/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e".to_string()
        });
        let dirp = std::path::Path::new(&dir);
        if !dirp.join("model.safetensors.index.json").exists() {
            eprintln!("SKIP real_qwen3_17b_header: no model at {dir}");
            return;
        }
        let m = StModel::open(dirp).expect("open multi-shard model");
        // Qwen3-1.7B has 311 weights spread over 2 shards (310 in shard 1 + lm_head in shard 2).
        assert_eq!(m.n_tensors(), 311, "tensor count");

        // embed_tokens: BF16 shape [vocab=151936, hidden=2048] -> ne [2048, 151936]
        let (e, eb) = m.raw("model.embed_tokens.weight").expect("embed");
        assert_eq!(e.dtype, "BF16");
        assert_eq!(e.shape, vec![151936, 2048]);
        assert_eq!(e.ne(), vec![2048, 151936]);
        assert_eq!(eb.len(), 151936 * 2048 * 2, "BF16 = 2 bytes/elem");

        // layer-0 q_proj: BF16 [2048,2048] -> ne [2048,2048], offsets honored.
        let (q, qb) = m
            .raw("model.layers.0.self_attn.q_proj.weight")
            .expect("q_proj");
        assert_eq!(q.dtype, "BF16");
        assert_eq!(q.shape, vec![2048, 2048]);
        assert_eq!(qb.len(), 2048 * 2048 * 2);

        // norm weight present and small (1-D, [2048]).
        let (nrm, nb) = m.raw("model.norm.weight").expect("final norm");
        assert_eq!(nrm.shape, vec![2048]);
        assert_eq!(nb.len(), 2048 * 2);

        // lm_head lives in shard 2 — proves cross-shard routing on a real file.
        let (lm, _) = m.raw("lm_head.weight").expect("lm_head (shard 2)");
        assert_eq!(lm.dtype, "BF16");
        assert_eq!(lm.shape, vec![151936, 2048]);

        // BF16 bytes dequant to finite f32 (spot-check the first row of the final norm).
        let nv = crate::dequant::dequantize(nrm.ggml_type(), nb, 2048);
        assert!(nv.iter().all(|v| v.is_finite()), "norm dequants finite");
        eprintln!(
            "real_qwen3_17b_header OK: {} tensors, norm[0]={}",
            m.n_tensors(),
            nv[0]
        );
    }

    /// Real config.json -> ModelConfig parse against the same on-disk model. Skipped if absent.
    #[test]
    fn real_qwen3_17b_config() {
        let dir = std::env::var("MEMRA_ST_TEST_DIR").unwrap_or_else(|_| {
            "/data/ai-ml/hf-models/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e".to_string()
        });
        let cfgp = std::path::Path::new(&dir).join("config.json");
        if !cfgp.exists() {
            eprintln!("SKIP real_qwen3_17b_config: no config at {cfgp:?}");
            return;
        }
        let mc = crate::config::ModelConfig::from_config_json(&cfgp).expect("parse config.json");
        assert_eq!(mc.arch, crate::config::Arch::Qwen3);
        assert_eq!(mc.n_layer, 28);
        assert_eq!(mc.n_embd, 2048);
        assert_eq!(mc.n_head, 16);
        assert_eq!(mc.n_head_kv, 8);
        assert_eq!(mc.head_dim_k, 128);
        assert_eq!(mc.n_ff, 6144);
        assert_eq!(mc.n_vocab, 151936);
        assert!(mc.moe.is_none() && mc.ssm.is_none());
    }
}
