//! Opened rank inputs: token order and provenance come from the same captured source.
use crate::source::TensorView;
use crate::{GgmlType, GgufFile};
use sha2::{Digest, Sha256};
use std::{collections::BTreeSet, fs::File, io::Read, path::Path, sync::Arc};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RankEncoding {
    Text,
    GgufI32,
    GgufI64,
}

/// Read-only source evidence, not a rewrite-admission capability.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RankIdentity {
    encoding: RankEncoding,
    source_sha256: String,
    order_sha256: String,
    artifact_sha256: String,
    source_bytes: u64,
    source_count: usize,
}
impl RankIdentity {
    pub fn encoding(&self) -> RankEncoding {
        self.encoding
    }
    /// Raw SHA256 for a single file (preserves existing diagnostic stamps); framed opened
    /// shard hashes for a split GGUF container. No pathname enters either identity.
    pub fn source_sha256(&self) -> &str {
        &self.source_sha256
    }
    pub fn order_sha256(&self) -> &str {
        &self.order_sha256
    }
    pub fn artifact_sha256(&self) -> &str {
        &self.artifact_sha256
    }
    pub fn source_bytes(&self) -> u64 {
        self.source_bytes
    }
    pub fn source_count(&self) -> usize {
        self.source_count
    }
}

/// Owns immutable ranks and their full provenance. Constructed only from an opened artifact;
/// callers cannot pair replacement IDs with an existing source digest.
#[derive(Debug, Clone)]
pub struct RankArtifact {
    ids: Arc<[u32]>,
    identity: RankIdentity,
}
impl RankArtifact {
    pub fn open(path: &Path) -> Result<Self, String> {
        let result = if path.as_os_str().as_encoded_bytes().ends_with(b".txt") {
            let mut file = File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
            let mut bytes = Vec::new();
            file.read_to_end(&mut bytes)
                .map_err(|e| format!("{}: {e}", path.display()))?;
            Self::from_text(bytes)
        } else {
            let gguf = GgufFile::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
            Self::from_gguf(&gguf)
        };
        result.map_err(|e| format!("rank artifact {}: {e}", path.display()))
    }
    fn from_text(bytes: Vec<u8>) -> Result<Self, String> {
        let text = std::str::from_utf8(&bytes).map_err(|e| format!("invalid UTF-8 ranks: {e}"))?;
        let ids = parse_text_ranks(text, "ranks")?;
        Self::finish(
            ids,
            RankEncoding::Text,
            format!("{:x}", Sha256::digest(&bytes)),
            bytes.len() as u64,
            1,
        )
    }
    fn from_gguf(gguf: &GgufFile) -> Result<Self, String> {
        Self::from_gguf_capture(capture_gguf(gguf)?)
    }
    fn from_gguf_capture(capture: GgufRankCapture) -> Result<Self, String> {
        let rows = capture.shape[0];
        let ids = decode_integer_ids(
            &TensorView {
                bytes: std::borrow::Cow::Borrowed(&capture.rank_bytes),
                ggml_type: capture.dtype,
                ne: capture.shape,
            },
            rows,
        )?;
        Self::finish(
            ids,
            capture.encoding,
            capture.source_sha256,
            capture.source_bytes,
            capture.source_count,
        )
    }
    fn finish(
        ids: Vec<u32>,
        encoding: RankEncoding,
        source_sha256: String,
        source_bytes: u64,
        source_count: usize,
    ) -> Result<Self, String> {
        validate_unique(&ids, "ranks")?;
        let mut order = Sha256::new();
        order.update(b"memra-rank-order-v1\0");
        order.update((ids.len() as u64).to_le_bytes());
        for id in &ids {
            order.update(id.to_le_bytes());
        }
        let order_sha256 = format!("{:x}", order.finalize());
        let mut artifact = Sha256::new();
        artifact.update(b"memra-rank-artifact-v1\0");
        for field in [
            format!("{encoding:?}"),
            source_sha256.clone(),
            order_sha256.clone(),
            source_bytes.to_string(),
            source_count.to_string(),
        ] {
            artifact.update((field.len() as u64).to_le_bytes());
            artifact.update(field.as_bytes());
        }
        Ok(Self {
            ids: ids.into(),
            identity: RankIdentity {
                encoding,
                source_sha256,
                order_sha256,
                artifact_sha256: format!("{:x}", artifact.finalize()),
                source_bytes,
                source_count,
            },
        })
    }
    pub fn ids(&self) -> &[u32] {
        &self.ids
    }
    pub fn identity(&self) -> &RankIdentity {
        &self.identity
    }
    pub fn validate_for_head(&self, rows: usize, what: &str) -> Result<(), String> {
        validate_ranks(&self.ids, rows, what)
    }
}

struct GgufRankCapture {
    rank_bytes: Vec<u8>,
    dtype: GgmlType,
    shape: Vec<u64>,
    encoding: RankEncoding,
    source_sha256: String,
    source_bytes: u64,
    source_count: usize,
}

// Hash and retain the rank bytes from the SAME read buffers. Unrelated model weights are
// streamed through the full-source hash, not copied into a giant temporary snapshot.
fn capture_gguf(gguf: &GgufFile) -> Result<GgufRankCapture, String> {
    use std::io::{Read, Seek, SeekFrom};
    let mut matches = gguf.tensors.iter().filter(|t| t.name == "d2t");
    let tensor = matches
        .next()
        .ok_or("GGUF rank container has no d2t tensor")?;
    if matches.next().is_some() {
        return Err("GGUF rank container has ambiguous d2t tensors".into());
    }
    let encoding = match tensor.ggml_type {
        GgmlType::I32 => RankEncoding::GgufI32,
        GgmlType::I64 => RankEncoding::GgufI64,
        other => return Err(format!("d2t must be I32/I64, got {other:?}")),
    };
    if tensor.ne.len() != 1 {
        return Err("d2t must have exactly one dimension".into());
    }
    let count = gguf.n_shards();
    let mut combined = Sha256::new();
    if count > 1 {
        combined.update(b"memra-rank-shards-v1\0");
        combined.update((count as u64).to_le_bytes());
    }
    let mut raw_single = None;
    let mut total = 0u64;
    let mut rank_bytes = Vec::new();
    for (index, shard) in gguf.shards.iter().enumerate() {
        let mut file = shard.file.try_clone().map_err(|e| e.to_string())?;
        file.seek(SeekFrom::Start(0)).map_err(|e| e.to_string())?;
        let file_len = shard.mmap.len() as u64;
        let data_start = shard.data_start;
        let range = (index == tensor.shard)
            .then(|| data_start + tensor.offset..data_start + tensor.offset + tensor.n_bytes);
        let mut header = Vec::new();
        let mut hash = Sha256::new();
        let mut offset = 0u64;
        let mut buffer = [0u8; 64 * 1024];
        loop {
            let n = file.read(&mut buffer).map_err(|e| e.to_string())?;
            if n == 0 {
                break;
            }
            let end = offset
                .checked_add(n as u64)
                .ok_or("rank source byte count overflow")?;
            if end > file_len {
                return Err("GGUF size changed during rank capture".into());
            }
            let bytes = &buffer[..n];
            hash.update(bytes);
            if offset < data_start {
                header.extend_from_slice(&bytes[..((data_start - offset).min(n as u64) as usize)]);
            }
            if let Some(range) = &range {
                let start = offset.max(range.start);
                let stop = end.min(range.end);
                if start < stop {
                    rank_bytes.extend_from_slice(
                        &bytes[(start - offset) as usize..(stop - offset) as usize],
                    );
                }
            }
            offset = end;
        }
        if offset != file_len {
            return Err("GGUF size changed during rank capture".into());
        }
        attest_header(gguf, index, &header)
            .map_err(|e| format!("GGUF header changed during rank capture: {e}"))?;
        total = total
            .checked_add(offset)
            .ok_or("rank source byte count overflow")?;
        let digest = hash.finalize();
        if count == 1 {
            raw_single = Some(format!("{digest:x}"));
        } else {
            combined.update(offset.to_le_bytes());
            combined.update(digest);
        }
    }
    Ok(GgufRankCapture {
        rank_bytes,
        dtype: tensor.ggml_type,
        shape: tensor.ne.clone(),
        encoding,
        source_sha256: raw_single.unwrap_or_else(|| format!("{:x}", combined.finalize())),
        source_bytes: total,
        source_count: count,
    })
}

// Reuse the GGUF reader's binary value decoder to attest that the captured header describes
// the already-opened tensor extents. A header edit cannot pair old offsets with a new digest.
fn attest_header(gguf: &GgufFile, shard: usize, header: &[u8]) -> std::io::Result<()> {
    use std::{
        collections::BTreeMap,
        io::{Error, ErrorKind},
    };
    let fail = || {
        Error::new(
            ErrorKind::InvalidData,
            "captured header differs from opened GGUF metadata",
        )
    };
    let path = gguf.shard_path(shard);
    let mut cursor = crate::Cursor::new(header, path);
    if cursor.u32()? != crate::GGUF_MAGIC || cursor.u32()? != gguf.version {
        return Err(fail());
    }
    let tensors: Vec<_> = gguf.tensors.iter().filter(|t| t.shard == shard).collect();
    let nt = crate::checked_count(cursor.i64()?, "n_tensors", 8, path)?;
    if nt != tensors.len() {
        return Err(fail());
    }
    let nk = crate::checked_count(cursor.i64()?, "n_kv", 16, path)?;
    if nk > crate::MAX_GGUF_KV_ENTRIES || nk > cursor.remaining() / 13 {
        return Err(fail());
    }
    let mut metadata = BTreeMap::new();
    for _ in 0..nk {
        let key = cursor.string()?;
        let ty = cursor.u32()?;
        metadata.insert(key, cursor.value(ty)?);
    }
    let alignment = crate::metadata_alignment(&metadata, path)?;
    let count = crate::structural_u64(&metadata, "split.count", path)?.unwrap_or(0);
    if (gguf.n_shards() == 1 && count > 1)
        || (gguf.n_shards() > 1 && count != gguf.n_shards() as u64)
    {
        return Err(fail());
    }
    if gguf.n_shards() > 1
        && crate::structural_u64(&metadata, "split.no", path)?.unwrap_or(0) != shard as u64
    {
        return Err(fail());
    }
    let total = crate::structural_u64(&metadata, "split.tensors.count", path)?.unwrap_or(0);
    if total > 0 && total != gguf.tensors.len() as u64 {
        return Err(fail());
    }
    for expected in tensors {
        if cursor.string()? != expected.name || cursor.u32()? as usize != expected.ne.len() {
            return Err(fail());
        }
        for &dim in &expected.ne {
            if cursor.i64()? as i128 != i128::from(dim) {
                return Err(fail());
            }
        }
        if GgmlType::from_u32(cursor.u32()?) != Some(expected.ggml_type)
            || cursor.u64()? != expected.offset
        {
            return Err(fail());
        }
    }
    if crate::aligned_data_start(cursor.pos as u64, alignment, path)?
        != gguf.shards[shard].data_start
    {
        return Err(fail());
    }
    Ok(())
}

pub fn parse_text_ranks(text: &str, what: &str) -> Result<Vec<u32>, String> {
    let mut out = Vec::new();
    for (lineno, raw) in text.lines().enumerate() {
        let line = raw.trim();
        if line.is_empty() {
            continue;
        }
        out.push(line.parse::<u32>().map_err(|_| format!("{what}: line {} is not a token id ({line:?}); a ranks .txt is one integer id per line in rank order",lineno+1))?);
    }
    Ok(out)
}
fn validate_unique(ids: &[u32], what: &str) -> Result<(), String> {
    if ids.is_empty() {
        return Err(format!(
            "{what}: the ranks artifact yields an EMPTY id list"
        ));
    }
    let mut seen = BTreeSet::new();
    for &id in ids {
        if !seen.insert(id) {
            return Err(format!(
                "{what}: token id {id} appears more than once: a ranks list is a set of distinct ids in rank order"
            ));
        }
    }
    Ok(())
}
pub fn validate_ranks(ids: &[u32], rows: usize, what: &str) -> Result<(), String> {
    if ids.len() > rows {
        return Err(format!(
            "{what}: {} ranks for a {rows}-row head: a ranks list wider than the vocabulary was minted for a different model",
            ids.len()
        ));
    }
    if let Some(id) = ids.iter().find(|&&id| id as usize >= rows) {
        return Err(format!(
            "{what}: token id {id} >= head rows {rows}: the ranks artifact was minted for a different vocabulary (wrong-model file refused at boot)"
        ));
    }
    validate_unique(ids, what)
}

/// Common signed-width decoder for external-draft d2t and independent ranks containers.
/// Vocabulary and uniqueness are checked by the caller's distinct semantic contract.
pub(super) fn decode_integer_ids(view: &TensorView<'_>, rows: u64) -> Result<Vec<u32>, String> {
    let width = match view.ggml_type {
        GgmlType::I32 => 4,
        GgmlType::I64 => 8,
        _ => return Err("d2t must be I32 or I64".into()),
    };
    if view.ne != [rows] || u64::try_from(view.bytes.len()).ok() != rows.checked_mul(width) {
        return Err("d2t payload differs from its bound shape".into());
    }
    view.bytes
        .chunks_exact(width as usize)
        .map(|chunk| {
            let value = if width == 4 {
                i64::from(i32::from_le_bytes(chunk.try_into().unwrap()))
            } else {
                i64::from_le_bytes(chunk.try_into().unwrap())
            };
            u32::try_from(value).map_err(|_| "d2t token ID is negative or exceeds u32".into())
        })
        .collect()
}

#[cfg(test)]
mod tests;
