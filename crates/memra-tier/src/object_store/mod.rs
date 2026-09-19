//! Versioned immutable chunk store. Layout is a proposal pending lead freeze.
//! Root publication is last. Ephemeral publication is NOT restart durability.
use crate::contracts::{Digest, Error, ObjectKey, Result};
use sha2::{Digest as _, Sha256};
use std::fs::{self, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

pub const ALIGNMENT: usize = 4096;
pub const MAX_CHUNK: usize = 1024 * 1024;
pub const MAX_ROOT: usize = 1024 * 1024;
const MAGIC: &[u8; 8] = b"MREXT001";
const VERSION: u32 = 1;
pub fn digest(bytes: &[u8]) -> Digest {
    Sha256::digest(bytes).into()
}
pub fn hex(bytes: &Digest) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}
pub fn padded_len(valid: usize) -> Result<usize> {
    valid
        .checked_add(ALIGNMENT - 1)
        .map(|n| n & !(ALIGNMENT - 1))
        .ok_or(Error::Overflow)
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExtentHeader {
    pub valid_bytes: u64,
    pub storage_bytes: u64,
    pub payload_sha256: Digest,
}
impl ExtentHeader {
    pub fn encode(&self) -> Result<[u8; ALIGNMENT]> {
        if self.valid_bytes > MAX_CHUNK as u64
            || self.storage_bytes != padded_len(self.valid_bytes as usize)? as u64
        {
            return Err(Error::InvalidLayout);
        }
        let mut h = [0; ALIGNMENT];
        h[..8].copy_from_slice(MAGIC);
        h[8..12].copy_from_slice(&VERSION.to_le_bytes());
        h[12..16].copy_from_slice(&(ALIGNMENT as u32).to_le_bytes());
        h[16..24].copy_from_slice(&self.valid_bytes.to_le_bytes());
        h[24..32].copy_from_slice(&self.storage_bytes.to_le_bytes());
        h[32..64].copy_from_slice(&self.payload_sha256);
        let checksum = digest(&h[..ALIGNMENT - 32]);
        h[ALIGNMENT - 32..].copy_from_slice(&checksum);
        Ok(h)
    }
    pub fn decode(bytes: &[u8]) -> Result<Self> {
        if bytes.len() != ALIGNMENT || &bytes[..8] != MAGIC {
            return Err(Error::Corrupt);
        }
        let version = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
        if version != VERSION {
            return Err(Error::UnsupportedVersion(version));
        }
        if u32::from_le_bytes(bytes[12..16].try_into().unwrap()) != ALIGNMENT as u32
            || digest(&bytes[..ALIGNMENT - 32]) != bytes[ALIGNMENT - 32..]
            || bytes[64..ALIGNMENT - 32].iter().any(|&b| b != 0)
        {
            return Err(Error::Corrupt);
        }
        let h = Self {
            valid_bytes: u64::from_le_bytes(bytes[16..24].try_into().unwrap()),
            storage_bytes: u64::from_le_bytes(bytes[24..32].try_into().unwrap()),
            payload_sha256: bytes[32..64].try_into().unwrap(),
        };
        h.encode()?; // validate lengths before allocating or slicing
        Ok(h)
    }
}
pub fn encode_extent(payload: &[u8]) -> Result<Vec<u8>> {
    if payload.len() > MAX_CHUNK {
        return Err(Error::Capacity);
    }
    let storage = padded_len(payload.len())?;
    let header = ExtentHeader {
        valid_bytes: payload.len() as u64,
        storage_bytes: storage as u64,
        payload_sha256: digest(payload),
    };
    let mut encoded = Vec::with_capacity(ALIGNMENT + storage);
    encoded.extend_from_slice(&header.encode()?);
    encoded.extend_from_slice(payload);
    encoded.resize(ALIGNMENT + storage, 0);
    Ok(encoded)
}
pub fn decode_extent(encoded: &[u8]) -> Result<&[u8]> {
    if encoded.len() < ALIGNMENT {
        return Err(Error::Corrupt);
    }
    let h = ExtentHeader::decode(&encoded[..ALIGNMENT])?;
    if encoded.len() != ALIGNMENT + h.storage_bytes as usize {
        return Err(Error::ShortIo {
            expected: ALIGNMENT as u64 + h.storage_bytes,
            actual: encoded.len() as u64,
        });
    }
    let end = ALIGNMENT + h.valid_bytes as usize;
    let payload = &encoded[ALIGNMENT..end];
    if digest(payload) != h.payload_sha256 || encoded[end..].iter().any(|&b| b != 0) {
        return Err(Error::Corrupt);
    }
    Ok(payload)
}
fn key_bytes(key: &ObjectKey) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(104);
    bytes.extend_from_slice(&key.artifact);
    bytes.extend_from_slice(&key.semantic_id);
    bytes.extend_from_slice(&key.layout);
    bytes.extend_from_slice(&key.generation.to_le_bytes());
    bytes
}
fn root_id(key: &ObjectKey) -> Digest {
    digest(&key_bytes(key))
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChunkRef {
    pub encoded_sha256: Digest,
    pub valid_bytes: u64,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ObjectLease {
    pub key: ObjectKey,
    pub valid_bytes: u64,
    pub chunks: Vec<ChunkRef>,
}
impl ObjectLease {
    /// Explicit release of this metadata snapshot. Day-1 store has no deletion/GC,
    /// so bytes remain immutable. Frozen leases must also return governor charges.
    pub fn release(self) {
        drop(self);
    }
}
/// Private fields prevent clients publishing an incomplete transaction by construction.
pub struct StoreTxn {
    object: ObjectLease,
    written: u64,
}
pub trait ObjectStore {
    fn lookup(&self, key: &ObjectKey) -> Result<Option<ObjectLease>>;
    fn begin(&self, key: ObjectKey, valid_bytes: u64) -> Result<StoreTxn>;
    fn put(&mut self, txn: &mut StoreTxn, payload: &[u8]) -> Result<()>;
    fn commit(&mut self, txn: StoreTxn) -> Result<ObjectLease>;
    fn read(&self, object: &ObjectLease, chunk: usize) -> Result<Vec<u8>>;
}
/// Immutable insert, never overwrite. `get` must bound allocation by max_bytes.
/// Implementations distinguish root and chunk namespaces even for equal digests.
pub trait BlobBackend {
    fn get(&self, root: bool, id: Digest, max_bytes: usize) -> Result<Option<Vec<u8>>>;
    fn insert(&mut self, root: bool, id: Digest, bytes: &[u8]) -> Result<()>;
}
pub struct ExtentStore<B> {
    backend: B,
}
impl<B: BlobBackend> ExtentStore<B> {
    pub fn new(backend: B) -> Self {
        Self { backend }
    }
    pub fn backend(&self) -> &B {
        &self.backend
    }
    pub fn backend_mut(&mut self) -> &mut B {
        &mut self.backend
    }
    fn verify_chunk(&self, chunk: &ChunkRef) -> Result<Vec<u8>> {
        let encoded = self
            .backend
            .get(false, chunk.encoded_sha256, MAX_CHUNK + ALIGNMENT)?
            .ok_or(Error::NotFound)?;
        if digest(&encoded) != chunk.encoded_sha256 {
            return Err(Error::Corrupt);
        }
        let payload = decode_extent(&encoded)?;
        if payload.len() as u64 != chunk.valid_bytes {
            return Err(Error::Corrupt);
        }
        Ok(payload.to_vec())
    }
}
impl<B: BlobBackend> ObjectStore for ExtentStore<B> {
    fn lookup(&self, key: &ObjectKey) -> Result<Option<ObjectLease>> {
        let Some(encoded) = self.backend.get(true, root_id(key), MAX_ROOT + ALIGNMENT)? else {
            return Ok(None);
        };
        let root = decode_extent(&encoded)?;
        if root.len() < 120 || root[..104] != key_bytes(key) {
            return Err(Error::Corrupt);
        }
        let valid_bytes = u64::from_le_bytes(root[104..112].try_into().unwrap());
        let count = u64::from_le_bytes(root[112..120].try_into().unwrap());
        if count > ((MAX_ROOT - 120) / 40) as u64 || root.len() != 120 + count as usize * 40 {
            return Err(Error::Corrupt);
        }
        let chunks = root[120..]
            .chunks_exact(40)
            .map(|c| ChunkRef {
                encoded_sha256: c[..32].try_into().unwrap(),
                valid_bytes: u64::from_le_bytes(c[32..].try_into().unwrap()),
            })
            .collect::<Vec<_>>();
        let mut total = 0u64;
        for c in &chunks {
            if c.valid_bytes == 0 || c.valid_bytes > MAX_CHUNK as u64 {
                return Err(Error::Corrupt);
            }
            total = total.checked_add(c.valid_bytes).ok_or(Error::Overflow)?;
        }
        if total != valid_bytes {
            return Err(Error::Corrupt);
        }
        // Advisory only: each chunk is verified again by read; missing chunks never become ready.
        Ok(Some(ObjectLease {
            key: key.clone(),
            valid_bytes,
            chunks,
        }))
    }
    fn begin(&self, key: ObjectKey, valid_bytes: u64) -> Result<StoreTxn> {
        if valid_bytes > ((MAX_ROOT - 120) / 40 * MAX_CHUNK) as u64 {
            return Err(Error::Capacity);
        }
        Ok(StoreTxn {
            object: ObjectLease {
                key,
                valid_bytes,
                chunks: Vec::new(),
            },
            written: 0,
        })
    }
    fn put(&mut self, txn: &mut StoreTxn, payload: &[u8]) -> Result<()> {
        let next = txn
            .written
            .checked_add(payload.len() as u64)
            .ok_or(Error::Overflow)?;
        if payload.is_empty()
            || next > txn.object.valid_bytes
            || txn.object.chunks.len() >= (MAX_ROOT - 120) / 40
        {
            return Err(Error::InvalidLayout);
        }
        let encoded = encode_extent(payload)?;
        let chunk = ChunkRef {
            encoded_sha256: digest(&encoded),
            valid_bytes: payload.len() as u64,
        };
        self.backend.insert(false, chunk.encoded_sha256, &encoded)?;
        self.verify_chunk(&chunk)?;
        txn.object.chunks.push(chunk);
        txn.written = next;
        Ok(())
    }
    fn commit(&mut self, txn: StoreTxn) -> Result<ObjectLease> {
        if txn.written != txn.object.valid_bytes {
            return Err(Error::NotReady);
        }
        for c in &txn.object.chunks {
            self.verify_chunk(c)?;
        }
        let mut root = key_bytes(&txn.object.key);
        root.extend_from_slice(&txn.object.valid_bytes.to_le_bytes());
        root.extend_from_slice(&(txn.object.chunks.len() as u64).to_le_bytes());
        for c in &txn.object.chunks {
            root.extend_from_slice(&c.encoded_sha256);
            root.extend_from_slice(&c.valid_bytes.to_le_bytes());
        }
        let encoded = encode_extent(&root)?;
        self.backend
            .insert(true, root_id(&txn.object.key), &encoded)?;
        if self
            .backend
            .get(true, root_id(&txn.object.key), MAX_ROOT + ALIGNMENT)?
            .as_deref()
            != Some(&encoded)
        {
            return Err(Error::Corrupt);
        }
        Ok(txn.object)
    }
    fn read(&self, object: &ObjectLease, chunk: usize) -> Result<Vec<u8>> {
        self.verify_chunk(object.chunks.get(chunk).ok_or(Error::InvalidLayout)?)
    }
}
/// Ordinary filesystem EPHEMERAL cache. Atomic namespace publication via hard-link
/// after complete write; no clobber and no fsync durability claim. No direct I/O claim.
pub struct FileBackend {
    directory: PathBuf,
}
impl FileBackend {
    pub fn open(directory: &Path) -> Result<Self> {
        fs::create_dir_all(directory)?;
        Ok(Self {
            directory: directory.to_owned(),
        })
    }
    fn path(&self, root: bool, id: Digest) -> PathBuf {
        self.directory.join(format!(
            "{}-{}",
            if root { "root" } else { "chunk" },
            hex(&id)
        ))
    }
}
impl BlobBackend for FileBackend {
    fn get(&self, root: bool, id: Digest, max_bytes: usize) -> Result<Option<Vec<u8>>> {
        let file = match fs::File::open(self.path(root, id)) {
            Ok(f) => f,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(e) => return Err(e.into()),
        };
        if file.metadata()?.len() > max_bytes as u64 {
            return Err(Error::Capacity);
        }
        let mut bytes = Vec::new();
        file.take(max_bytes as u64 + 1).read_to_end(&mut bytes)?;
        if bytes.len() > max_bytes {
            return Err(Error::Capacity);
        }
        Ok(Some(bytes))
    }
    fn insert(&mut self, root: bool, id: Digest, bytes: &[u8]) -> Result<()> {
        use std::sync::atomic::{AtomicU64, Ordering};
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let tmp = self.directory.join(format!(
            ".pending-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let mut file = OpenOptions::new().write(true).create_new(true).open(&tmp)?;
        let result = (|| {
            file.write_all(bytes)?;
            drop(file);
            match fs::hard_link(&tmp, self.path(root, id)) {
                Ok(()) => Ok(()),
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                    if self.get(root, id, bytes.len())?.as_deref() == Some(bytes) {
                        Ok(())
                    } else {
                        Err(Error::Conflict)
                    }
                }
                Err(e) => Err(e.into()),
            }
        })();
        let cleanup = fs::remove_file(tmp).map_err(Error::from);
        result.and(cleanup)
    }
}
