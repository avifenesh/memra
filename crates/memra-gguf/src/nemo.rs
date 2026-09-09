//! Reader for NVIDIA NeMo `.nemo` archives.
//!
//! A `.nemo` file is a POSIX pax tar holding a YAML config, tokenizer artifacts and a
//! `model_weights.ckpt`, which is itself an uncompressed PyTorch zip. The zip's `data.pkl`
//! is a pickle: an ordered map from parameter name to a rebuilt tensor description whose
//! payload lives in a separate zip member.
//!
//! Nothing here executes checkpoint code. The pickle reader is a value machine over a fixed
//! opcode set with an explicit global allowlist, so a checkpoint that names any callable
//! other than `collections.OrderedDict`, `torch._utils._rebuild_tensor_v2` and the storage
//! classes is refused rather than interpreted. No tensor payload is read while taking a
//! census: this reads names, dtypes, shapes, strides and byte extents.

use std::collections::BTreeMap;
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NemoError {
    Archive(String),
    Zip(String),
    Pickle(String),
}

impl fmt::Display for NemoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Archive(m) => write!(f, "nemo archive: {m}"),
            Self::Zip(m) => write!(f, "nemo checkpoint zip: {m}"),
            Self::Pickle(m) => write!(f, "nemo checkpoint pickle: {m}"),
        }
    }
}

impl std::error::Error for NemoError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NemoMember {
    pub name: String,
    pub offset: usize,
    pub size: usize,
}

/// Members of a pax tar, in archive order, without reading any payload.
///
/// Long member names live in the pax extended header that precedes the entry, so the 100-byte
/// ustar name field is only a fallback. NeMo writes tokenizer names far past that limit.
pub fn tar_members(archive: &[u8]) -> Result<Vec<NemoMember>, NemoError> {
    const BLOCK: usize = 512;
    let mut members = Vec::new();
    let mut offset = 0usize;
    let mut pax_path: Option<String> = None;
    while offset + BLOCK <= archive.len() {
        let header = &archive[offset..offset + BLOCK];
        if header.iter().all(|&b| b == 0) {
            break;
        }
        if &header[257..262] != b"ustar" {
            return Err(NemoError::Archive(format!(
                "block at {offset} is not a ustar header"
            )));
        }
        let size = octal(&header[124..136])
            .ok_or_else(|| NemoError::Archive(format!("bad size field at {offset}")))?;
        let data = offset + BLOCK;
        let size = size as usize;
        if data + size > archive.len() {
            return Err(NemoError::Archive("member runs past the archive".into()));
        }
        let kind = header[156];
        let ustar_name = std::str::from_utf8(trim(&header[0..100]))
            .map_err(|_| NemoError::Archive("member name is not utf-8".into()))?
            .to_owned();
        match kind {
            b'x' | b'g' => {
                pax_path = pax_field(&archive[data..data + size], "path")?;
            }
            b'0' | 0 => {
                members.push(NemoMember {
                    name: pax_path.take().unwrap_or(ustar_name),
                    offset: data,
                    size,
                });
            }
            b'5' => {
                pax_path = None;
            }
            other => {
                return Err(NemoError::Archive(format!(
                    "unsupported tar entry type {} in {ustar_name}",
                    other as char
                )));
            }
        }
        offset = data + size.div_ceil(BLOCK) * BLOCK;
    }
    if members.is_empty() {
        return Err(NemoError::Archive(
            "archive holds no regular members".into(),
        ));
    }
    Ok(members)
}

fn trim(field: &[u8]) -> &[u8] {
    let end = field.iter().position(|&b| b == 0).unwrap_or(field.len());
    &field[..end]
}

fn octal(field: &[u8]) -> Option<u64> {
    let text = std::str::from_utf8(trim(field)).ok()?;
    let text = text.trim_matches(|c: char| c == ' ' || c == '\0');
    if text.is_empty() {
        return Some(0);
    }
    u64::from_str_radix(text, 8).ok()
}

/// One `LEN key=value\n` record per line, with LEN counting its own digits and the newline.
fn pax_field(block: &[u8], key: &str) -> Result<Option<String>, NemoError> {
    let mut offset = 0usize;
    while offset < block.len() {
        let space = block[offset..]
            .iter()
            .position(|&b| b == b' ')
            .ok_or_else(|| NemoError::Archive("pax record without a length".into()))?;
        let length: usize = std::str::from_utf8(&block[offset..offset + space])
            .ok()
            .and_then(|t| t.parse().ok())
            .ok_or_else(|| NemoError::Archive("pax record length is not a number".into()))?;
        if length < space + 2 || offset + length > block.len() {
            return Err(NemoError::Archive(
                "pax record length is out of range".into(),
            ));
        }
        let record = &block[offset + space + 1..offset + length - 1];
        if let Some(equals) = record.iter().position(|&b| b == b'=')
            && &record[..equals] == key.as_bytes()
        {
            return Ok(Some(
                std::str::from_utf8(&record[equals + 1..])
                    .map_err(|_| NemoError::Archive("pax value is not utf-8".into()))?
                    .to_owned(),
            ));
        }
        offset += length;
    }
    Ok(None)
}

/// Stored (uncompressed) zip entries and their payload extents.
///
/// PyTorch writes every member stored, so a compressed entry means this is not the artifact
/// the contract describes and the reader refuses it instead of pulling in an inflater.
pub fn zip_entries(zip: &[u8]) -> Result<BTreeMap<String, (usize, usize)>, NemoError> {
    const EOCD: [u8; 4] = [0x50, 0x4b, 0x05, 0x06];
    const CENTRAL: [u8; 4] = [0x50, 0x4b, 0x01, 0x02];
    const LOCAL: [u8; 4] = [0x50, 0x4b, 0x03, 0x04];
    let window = zip.len().saturating_sub(66_000);
    let eocd = (window..zip.len().saturating_sub(21))
        .rev()
        .find(|&i| zip[i..i + 4] == EOCD)
        .ok_or_else(|| NemoError::Zip("no end of central directory".into()))?;
    let count = u16(zip, eocd + 10)? as usize;
    let mut cursor = u32(zip, eocd + 16)? as usize;
    let mut entries = BTreeMap::new();
    for _ in 0..count {
        if cursor + 46 > zip.len() || zip[cursor..cursor + 4] != CENTRAL {
            return Err(NemoError::Zip(
                "central directory entry is malformed".into(),
            ));
        }
        let method = u16(zip, cursor + 10)?;
        if method != 0 {
            return Err(NemoError::Zip(format!(
                "entry uses compression method {method}, expected stored"
            )));
        }
        let size = u32(zip, cursor + 24)? as usize;
        let name_len = u16(zip, cursor + 28)? as usize;
        let extra_len = u16(zip, cursor + 30)? as usize;
        let comment_len = u16(zip, cursor + 32)? as usize;
        let local = u32(zip, cursor + 42)? as usize;
        let name = std::str::from_utf8(&zip[cursor + 46..cursor + 46 + name_len])
            .map_err(|_| NemoError::Zip("entry name is not utf-8".into()))?
            .to_owned();
        if local + 30 > zip.len() || zip[local..local + 4] != LOCAL {
            return Err(NemoError::Zip(format!("{name} has no local header")));
        }
        let start = local + 30 + u16(zip, local + 26)? as usize + u16(zip, local + 28)? as usize;
        if start + size > zip.len() {
            return Err(NemoError::Zip(format!("{name} runs past the zip")));
        }
        if entries.insert(name.clone(), (start, size)).is_some() {
            return Err(NemoError::Zip(format!("duplicate entry {name}")));
        }
        cursor += 46 + name_len + extra_len + comment_len;
    }
    Ok(entries)
}

fn u16(bytes: &[u8], at: usize) -> Result<u16, NemoError> {
    bytes
        .get(at..at + 2)
        .map(|b| u16::from_le_bytes([b[0], b[1]]))
        .ok_or_else(|| NemoError::Zip("read past the zip".into()))
}

fn u32(bytes: &[u8], at: usize) -> Result<u32, NemoError> {
    bytes
        .get(at..at + 4)
        .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .ok_or_else(|| NemoError::Zip("read past the zip".into()))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NemoDtype {
    F32,
    F16,
    BF16,
    I64,
    I32,
    U8,
}

impl NemoDtype {
    pub fn bytes(self) -> u64 {
        match self {
            Self::F32 | Self::I32 => 4,
            Self::F16 | Self::BF16 => 2,
            Self::I64 => 8,
            Self::U8 => 1,
        }
    }

    fn from_storage_class(name: &str) -> Option<Self> {
        match name {
            "FloatStorage" => Some(Self::F32),
            "HalfStorage" => Some(Self::F16),
            "BFloat16Storage" => Some(Self::BF16),
            "LongStorage" => Some(Self::I64),
            "IntStorage" => Some(Self::I32),
            "ByteStorage" => Some(Self::U8),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NemoTensor {
    pub name: String,
    pub dtype: NemoDtype,
    pub shape: Vec<u64>,
    pub stride: Vec<u64>,
    /// Element offset into the storage, not a byte offset.
    pub storage_offset: u64,
    /// Zip member key under `<prefix>/data/`; two tensors may name the same one.
    pub storage_key: String,
    /// Elements the storage declares, which can exceed this tensor's own extent.
    pub storage_elements: u64,
}

impl NemoTensor {
    pub fn elements(&self) -> u64 {
        self.shape.iter().product()
    }

    /// True when the tensor reads its storage as one dense ascending block from its offset.
    pub fn is_contiguous(&self) -> bool {
        let mut expected = 1u64;
        for (extent, stride) in self.shape.iter().zip(&self.stride).rev() {
            if *stride != expected {
                return false;
            }
            expected *= extent;
        }
        true
    }
}

#[derive(Debug, Clone, PartialEq)]
enum Value {
    None,
    Bool(bool),
    Int(i64),
    Str(String),
    Tuple(Vec<Value>),
    Dict(Vec<(Value, Value)>),
    Global(String, String),
    Storage {
        dtype: NemoDtype,
        key: String,
        elements: u64,
    },
    Tensor(Box<NemoTensor>),
    Mark,
}

/// Parameter name to tensor description, in checkpoint order.
///
/// The census is taken from `data.pkl` alone; storage members are never opened here.
pub fn pickle_census(pickle: &[u8]) -> Result<Vec<NemoTensor>, NemoError> {
    let mut stack: Vec<Value> = Vec::new();
    let mut memo: Vec<Value> = Vec::new();
    let mut cursor = 0usize;
    let error = |m: &str| NemoError::Pickle(m.to_owned());
    let mut result: Option<Value> = None;
    while cursor < pickle.len() {
        let opcode = pickle[cursor];
        cursor += 1;
        match opcode {
            0x80 => cursor += 1,                      // PROTO
            0x95 => cursor += 8,                      // FRAME
            b'(' => stack.push(Value::Mark),          // MARK
            b')' => stack.push(Value::Tuple(vec![])), // EMPTY_TUPLE
            b'}' => stack.push(Value::Dict(vec![])),  // EMPTY_DICT
            b'N' => stack.push(Value::None),
            0x88 => stack.push(Value::Bool(true)),
            0x89 => stack.push(Value::Bool(false)),
            b'K' => {
                stack.push(Value::Int(i64::from(read_u8(pickle, &mut cursor)?)));
            }
            b'M' => {
                stack.push(Value::Int(i64::from(read_u16(pickle, &mut cursor)?)));
            }
            b'J' => {
                stack.push(Value::Int(i64::from(read_i32(pickle, &mut cursor)?)));
            }
            b'X' => {
                let len = read_u32(pickle, &mut cursor)? as usize;
                stack.push(Value::Str(read_str(pickle, &mut cursor, len)?));
            }
            0x8c => {
                let len = read_u8(pickle, &mut cursor)? as usize;
                stack.push(Value::Str(read_str(pickle, &mut cursor, len)?));
            }
            b'q' => {
                let slot = read_u8(pickle, &mut cursor)? as usize;
                put(
                    &mut memo,
                    slot,
                    stack.last().cloned().ok_or(error("empty put"))?,
                );
            }
            b'r' => {
                let slot = read_u32(pickle, &mut cursor)? as usize;
                put(
                    &mut memo,
                    slot,
                    stack.last().cloned().ok_or(error("empty put"))?,
                );
            }
            b'h' => {
                let slot = read_u8(pickle, &mut cursor)? as usize;
                stack.push(get(&memo, slot)?);
            }
            b'j' => {
                let slot = read_u32(pickle, &mut cursor)? as usize;
                stack.push(get(&memo, slot)?);
            }
            b'c' => {
                let module = read_line(pickle, &mut cursor)?;
                let name = read_line(pickle, &mut cursor)?;
                allow_global(&module, &name)?;
                stack.push(Value::Global(module, name));
            }
            0x85..=0x87 => {
                let arity = (opcode - 0x84) as usize;
                let at = stack
                    .len()
                    .checked_sub(arity)
                    .ok_or(error("short tuple stack"))?;
                let items = stack.split_off(at);
                stack.push(Value::Tuple(items));
            }
            b't' => {
                let at = stack
                    .iter()
                    .rposition(|v| *v == Value::Mark)
                    .ok_or(error("tuple without a mark"))?;
                let items = stack.split_off(at + 1);
                stack.pop();
                stack.push(Value::Tuple(items));
            }
            b's' => {
                let value = stack.pop().ok_or(error("setitem without a value"))?;
                let key = stack.pop().ok_or(error("setitem without a key"))?;
                match stack.last_mut() {
                    Some(Value::Dict(items)) => items.push((key, value)),
                    _ => return Err(error("setitem onto a non-dictionary")),
                }
            }
            b'u' => {
                let at = stack
                    .iter()
                    .rposition(|v| *v == Value::Mark)
                    .ok_or(error("setitems without a mark"))?;
                let items = stack.split_off(at + 1);
                stack.pop();
                if !items.len().is_multiple_of(2) {
                    return Err(error("setitems with an odd run"));
                }
                match stack.last_mut() {
                    Some(Value::Dict(target)) => {
                        for pair in items.chunks_exact(2) {
                            target.push((pair[0].clone(), pair[1].clone()));
                        }
                    }
                    _ => return Err(error("setitems onto a non-dictionary")),
                }
            }
            b'b' => {
                // BUILD applies instance attributes, which for a torch state dict is the
                // `_metadata` map of per-module serialization versions. It carries no tensor,
                // so it is dropped. A slot state (a tuple) is not something this reader knows
                // how to apply and is refused rather than ignored.
                let state = stack.pop().ok_or(error("build without state"))?;
                if !matches!(state, Value::Dict(_)) {
                    return Err(error("unsupported object state"));
                }
            }
            b'Q' => {
                let id = stack.pop().ok_or(error("persistent id without a value"))?;
                stack.push(storage(id)?);
            }
            b'R' => {
                let args = stack.pop().ok_or(error("reduce without arguments"))?;
                let callable = stack.pop().ok_or(error("reduce without a callable"))?;
                stack.push(reduce(callable, args)?);
            }
            b'.' => {
                result = stack.pop();
                break;
            }
            other => {
                return Err(NemoError::Pickle(format!(
                    "unsupported opcode {other:#04x} at {}",
                    cursor - 1
                )));
            }
        }
    }
    let items = match result {
        Some(Value::Dict(items)) => items,
        _ => return Err(error("checkpoint is not a name to tensor map")),
    };
    let mut census = Vec::with_capacity(items.len());
    for (key, value) in items {
        match (key, value) {
            (Value::Str(name), Value::Tensor(mut tensor)) => {
                tensor.name = name;
                census.push(*tensor);
            }
            (Value::Str(name), _) => {
                return Err(NemoError::Pickle(format!("{name} is not a tensor")));
            }
            _ => return Err(error("checkpoint key is not a name")),
        }
    }
    Ok(census)
}

fn allow_global(module: &str, name: &str) -> Result<(), NemoError> {
    let allowed = matches!(
        (module, name),
        ("collections", "OrderedDict") | ("torch._utils", "_rebuild_tensor_v2")
    ) || (module == "torch" && NemoDtype::from_storage_class(name).is_some());
    if allowed {
        Ok(())
    } else {
        Err(NemoError::Pickle(format!(
            "checkpoint names disallowed global {module}.{name}"
        )))
    }
}

fn storage(id: Value) -> Result<Value, NemoError> {
    let items = match id {
        Value::Tuple(items) => items,
        _ => return Err(NemoError::Pickle("persistent id is not a tuple".into())),
    };
    match items.as_slice() {
        [
            Value::Str(kind),
            Value::Global(module, class),
            Value::Str(key),
            Value::Str(_location),
            Value::Int(elements),
        ] if kind == "storage" && module == "torch" => Ok(Value::Storage {
            dtype: NemoDtype::from_storage_class(class)
                .ok_or_else(|| NemoError::Pickle(format!("unsupported storage class {class}")))?,
            key: key.clone(),
            elements: *elements as u64,
        }),
        _ => Err(NemoError::Pickle("unsupported persistent id shape".into())),
    }
}

fn reduce(callable: Value, args: Value) -> Result<Value, NemoError> {
    let (module, name) = match callable {
        Value::Global(module, name) => (module, name),
        _ => return Err(NemoError::Pickle("reduce callable is not a global".into())),
    };
    let args = match args {
        Value::Tuple(items) => items,
        _ => return Err(NemoError::Pickle("reduce arguments are not a tuple".into())),
    };
    match (module.as_str(), name.as_str()) {
        ("collections", "OrderedDict") => Ok(Value::Dict(vec![])),
        ("torch._utils", "_rebuild_tensor_v2") => match args.as_slice() {
            [
                Value::Storage {
                    dtype,
                    key,
                    elements,
                },
                Value::Int(offset),
                Value::Tuple(shape),
                Value::Tuple(stride),
                Value::Bool(_requires_grad),
                ..,
            ] => Ok(Value::Tensor(Box::new(NemoTensor {
                name: String::new(),
                dtype: *dtype,
                shape: extents(shape)?,
                stride: extents(stride)?,
                storage_offset: *offset as u64,
                storage_key: key.clone(),
                storage_elements: *elements,
            }))),
            _ => Err(NemoError::Pickle("unsupported tensor rebuild shape".into())),
        },
        _ => Err(NemoError::Pickle(format!("cannot call {module}.{name}"))),
    }
}

fn extents(values: &[Value]) -> Result<Vec<u64>, NemoError> {
    values
        .iter()
        .map(|v| match v {
            Value::Int(x) if *x >= 0 => Ok(*x as u64),
            _ => Err(NemoError::Pickle("tensor extent is not a count".into())),
        })
        .collect()
}

fn put(memo: &mut Vec<Value>, slot: usize, value: Value) {
    if memo.len() <= slot {
        memo.resize(slot + 1, Value::None);
    }
    memo[slot] = value;
}

fn get(memo: &[Value], slot: usize) -> Result<Value, NemoError> {
    memo.get(slot)
        .cloned()
        .ok_or_else(|| NemoError::Pickle(format!("memo slot {slot} was never written")))
}

fn read_u8(bytes: &[u8], cursor: &mut usize) -> Result<u8, NemoError> {
    let value = *bytes
        .get(*cursor)
        .ok_or_else(|| NemoError::Pickle("truncated pickle".into()))?;
    *cursor += 1;
    Ok(value)
}

fn read_u16(bytes: &[u8], cursor: &mut usize) -> Result<u16, NemoError> {
    let slice = bytes
        .get(*cursor..*cursor + 2)
        .ok_or_else(|| NemoError::Pickle("truncated pickle".into()))?;
    *cursor += 2;
    Ok(u16::from_le_bytes([slice[0], slice[1]]))
}

fn read_u32(bytes: &[u8], cursor: &mut usize) -> Result<u32, NemoError> {
    let slice = bytes
        .get(*cursor..*cursor + 4)
        .ok_or_else(|| NemoError::Pickle("truncated pickle".into()))?;
    *cursor += 4;
    Ok(u32::from_le_bytes([slice[0], slice[1], slice[2], slice[3]]))
}

fn read_i32(bytes: &[u8], cursor: &mut usize) -> Result<i32, NemoError> {
    Ok(read_u32(bytes, cursor)? as i32)
}

/// A newline-terminated ASCII line, which is how protocol 2 spells a global's module and name.
fn read_line(bytes: &[u8], cursor: &mut usize) -> Result<String, NemoError> {
    let end = bytes[*cursor..]
        .iter()
        .position(|&b| b == b'\n')
        .ok_or_else(|| NemoError::Pickle("unterminated global name".into()))?;
    let text = std::str::from_utf8(&bytes[*cursor..*cursor + end])
        .map_err(|_| NemoError::Pickle("global name is not utf-8".into()))?
        .to_owned();
    *cursor += end + 1;
    Ok(text)
}

fn read_str(bytes: &[u8], cursor: &mut usize, len: usize) -> Result<String, NemoError> {
    let slice = bytes
        .get(*cursor..*cursor + len)
        .ok_or_else(|| NemoError::Pickle("truncated pickle string".into()))?;
    *cursor += len;
    std::str::from_utf8(slice)
        .map(str::to_owned)
        .map_err(|_| NemoError::Pickle("pickle string is not utf-8".into()))
}

/// Storage keys more than one tensor reads, with the readers named.
///
/// Aliasing is legal in a torch checkpoint and the upstream family documents flat LSTM gate
/// storage, so a loader must group before it copies. Substituting one aliased tensor for
/// another is silent at load and wrong at the first decode step.
pub fn shared_storage_groups(census: &[NemoTensor]) -> BTreeMap<String, Vec<String>> {
    let mut readers: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for tensor in census {
        readers
            .entry(tensor.storage_key.clone())
            .or_default()
            .push(tensor.name.clone());
    }
    readers.retain(|_, names| names.len() > 1);
    readers
}

#[cfg(test)]
mod tests;
