//! Immutable bounded extents with canonical v1 roots and explicit durability.
use crate::contracts::{
    self, BudgetGovernor, BudgetRequest, CancelState, Digest, Durability, Error, ObjectKey, Result,
    Wire,
};
pub use crate::contracts::{ChunkRef, ObjectLease, ObjectManifest, ObjectStore};
use std::fs::{self, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::{cell::RefCell, collections::HashMap, rc::Rc};

pub mod catalog;
pub mod prepared;

pub const ALIGNMENT: usize = 4096;
pub const MAX_CHUNK: usize = 1024 * 1024;
pub const MAX_ROOT: usize = 1024 * 1024;
const MAGIC: &[u8; 8] = b"MREXT001";
const VERSION: u32 = 1;
pub fn digest(bytes: &[u8]) -> Digest {
    contracts::checksum(bytes)
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
        let checksum = contracts::digest("extent-header", &h[..ALIGNMENT - 32]);
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
            || contracts::digest("extent-header", &bytes[..ALIGNMENT - 32])
                != bytes[ALIGNMENT - 32..]
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
fn root_id(key: &ObjectKey) -> Result<Digest> {
    key.identity()
}
/// Transactions belong to exactly one store instance, and remain retryable on errors.
pub struct StoreTxn {
    owner: u64,
    object: ObjectManifest,
    written: u64,
    cancelled: bool,
    published: bool,
}
/// Immutable insert, never overwrite. `get` must bound allocation by max_bytes.
/// Implementations distinguish root and chunk namespaces even for equal digests.
/// Sealed GC authority, issued only after the store checks all live leases.
pub struct EvictionPermit(());
pub trait BlobBackend {
    fn persistent(&self) -> bool {
        false
    }
    /// Retain process-wide storage ownership through unresolved lease shutdown.
    fn ownership_guard(&self) -> Option<std::sync::Arc<fs::File>> {
        None
    }
    /// Backends without a process-safe deletion protocol remain read/write only.
    fn evict_root(&mut self, _id: Digest, _permit: EvictionPermit) -> Result<()> {
        Err(Error::Unsupported)
    }
    fn get(&self, root: bool, id: Digest, max_bytes: usize) -> Result<Option<Vec<u8>>>;
    fn insert(&mut self, root: bool, id: Digest, bytes: &[u8]) -> Result<()>;
}
/// A single verified extent reservation, not permission to read sibling chunks.
/// Whole-object `ObjectStore::lease` remains the frozen, eager-validation mode.
#[derive(Debug)]
pub struct ExtentLease {
    key: ObjectKey,
    chunk: u32,
    reference: ChunkRef,
    charge: contracts::ChargedLease,
}
impl ExtentLease {
    pub fn reference(&self) -> &ChunkRef {
        &self.reference
    }
    pub fn charge(&self) -> &contracts::ChargedLease {
        &self.charge
    }
}
pub struct ExtentStore<B, G> {
    backend: B,
    governor: Rc<RefCell<G>>,
    owner: u64,
    ownership_guard: Option<std::sync::Arc<fs::File>>,
    leases: HashMap<(u64, u64), ObjectManifest>,
    extent_leases: HashMap<(u64, u64), (ObjectKey, u32, ChunkRef)>,
}
impl<B: BlobBackend, G: BudgetGovernor> ExtentStore<B, G> {
    pub fn new(backend: B, governor: Rc<RefCell<G>>) -> Self {
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
        let ownership_guard = backend.ownership_guard();
        Self {
            backend,
            ownership_guard,
            governor,
            owner: NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            leases: HashMap::new(),
            extent_leases: HashMap::new(),
        }
    }
    pub fn backend(&self) -> &B {
        &self.backend
    }
    pub fn backend_mut(&mut self) -> &mut B {
        &mut self.backend
    }
    /// Bound payload validation and NVMe reservation to one selected extent plus
    /// its framed root. The caller's NVMe request is a ceiling; other dimensions
    /// remain caller-supplied. Metadata remains bounded by MAX_ROOT, not table size.
    pub fn lease_extent(
        &mut self,
        m: &ObjectManifest,
        chunk: u32,
        r: &BudgetRequest,
    ) -> Result<ExtentLease> {
        let lease = self.reserve_extent(m, chunk, r)?;
        if let Err(error) = self.verify_chunk(&lease.reference) {
            self.release_extent(&lease)?;
            return Err(error);
        }
        Ok(lease)
    }
    fn reserve_extent(
        &mut self,
        m: &ObjectManifest,
        chunk: u32,
        r: &BudgetRequest,
    ) -> Result<ExtentLease> {
        m.validate()?;
        if self.lookup(&m.key)?.as_ref() != Some(m) {
            return Err(Error::Conflict);
        }
        let reference = m
            .chunks
            .get(chunk as usize)
            .ok_or(Error::InvalidLayout)?
            .clone();
        let root_bytes = (ALIGNMENT + padded_len(m.encode()?.len())?) as u64;
        let size = reference
            .storage_bytes
            .checked_add(ALIGNMENT as u64)
            .and_then(|n| n.checked_add(root_bytes))
            .ok_or(Error::Overflow)?;
        if r.bytes.nvme < size {
            return Err(Error::Capacity);
        }
        let mut request = r.clone();
        request.bytes.nvme = size;
        let charge = self.governor.borrow_mut().reserve(&request)?;
        self.extent_leases
            .insert(charge.id(), (m.key.clone(), chunk, reference.clone()));
        Ok(ExtentLease {
            key: m.key.clone(),
            chunk,
            reference,
            charge,
        })
    }
    fn check_extent(&self, lease: &ExtentLease) -> Result<()> {
        if lease.charge.state()? == contracts::ChargeState::Released {
            return Err(Error::AlreadyReleased);
        }
        if self.extent_leases.get(&lease.charge.id())
            != Some(&(lease.key.clone(), lease.chunk, lease.reference.clone()))
        {
            return Err(Error::ForeignLease);
        }
        Ok(())
    }
    pub fn read_extent(&self, lease: &ExtentLease, destination: &mut [u8]) -> Result<u64> {
        self.check_extent(lease)?;
        if destination.len() < lease.reference.valid_bytes as usize {
            return Err(Error::Capacity);
        }
        // Immutable content address is revalidated on EVERY read; no unchecked cache.
        let payload = self.verify_chunk(&lease.reference)?;
        destination[..payload.len()].copy_from_slice(&payload);
        Ok(payload.len() as u64)
    }
    pub fn release_extent(&mut self, lease: &ExtentLease) -> Result<()> {
        self.check_extent(lease)?;
        self.governor.borrow_mut().release(&lease.charge)?;
        self.extent_leases.remove(&lease.charge.id());
        Ok(())
    }
    fn check_txn(&self, t: &StoreTxn) -> Result<()> {
        if t.owner != self.owner {
            return Err(Error::ForeignLease);
        }
        if t.cancelled {
            return Err(Error::Cancelled);
        }
        Ok(())
    }
    fn verify_chunk(&self, c: &ChunkRef) -> Result<Vec<u8>> {
        let encoded = self
            .backend
            .get(false, c.encoded_digest, MAX_CHUNK + ALIGNMENT)?
            .ok_or(Error::NotFound)?;
        if contracts::digest("extent", &encoded) != c.encoded_digest {
            return Err(Error::Corrupt);
        }
        let payload = decode_extent(&encoded)?;
        if payload.len() as u64 != c.valid_bytes
            || padded_len(payload.len())? as u64 != c.storage_bytes
            || digest(payload) != c.checksum
        {
            return Err(Error::Corrupt);
        }
        Ok(payload.to_vec())
    }
    fn check_lease(&self, l: &ObjectLease) -> Result<()> {
        if l.charge.state()? == contracts::ChargeState::Released {
            return Err(Error::AlreadyReleased);
        }
        if self.leases.get(&l.charge.id()) != Some(&l.manifest) {
            return Err(Error::ForeignLease);
        }
        if self.lookup(&l.manifest.key)?.as_ref() != Some(&l.manifest) {
            return Err(Error::Conflict);
        }
        Ok(())
    }
}
impl<B: BlobBackend, G: BudgetGovernor> ObjectStore for ExtentStore<B, G> {
    type Transaction = StoreTxn;
    fn lookup(&self, key: &ObjectKey) -> Result<Option<ObjectManifest>> {
        let Some(encoded) = self
            .backend
            .get(true, root_id(key)?, MAX_ROOT + ALIGNMENT)?
        else {
            return Ok(None);
        };
        let manifest = ObjectManifest::decode(decode_extent(&encoded)?)?;
        if &manifest.key != key {
            return Err(Error::Corrupt);
        }
        for c in &manifest.chunks {
            if c.valid_bytes > MAX_CHUNK as u64
                || c.storage_bytes != padded_len(c.valid_bytes as usize)? as u64
            {
                return Err(Error::Corrupt);
            }
        }
        Ok(Some(manifest))
    }
    fn begin(
        &mut self,
        key: ObjectKey,
        valid_bytes: u64,
        durability: Durability,
    ) -> Result<StoreTxn> {
        key.validate()?;
        if durability == Durability::Persistent && !self.backend.persistent() {
            return Err(Error::Unsupported);
        }
        Ok(StoreTxn {
            owner: self.owner,
            object: ObjectManifest {
                version: 1,
                key,
                valid_bytes,
                chunks: vec![],
                durability,
            },
            written: 0,
            cancelled: false,
            published: false,
        })
    }
    fn put(&mut self, txn: &mut StoreTxn, payload: &[u8]) -> Result<()> {
        self.check_txn(txn)?;
        if txn.published {
            return Err(Error::Conflict);
        }
        let next = txn
            .written
            .checked_add(payload.len() as u64)
            .ok_or(Error::Overflow)?;
        if payload.is_empty() || next > txn.object.valid_bytes {
            return Err(Error::InvalidLayout);
        }
        let encoded = encode_extent(payload)?;
        let chunk = ChunkRef {
            version: 1,
            encoded_digest: contracts::digest("extent", &encoded),
            valid_bytes: payload.len() as u64,
            storage_bytes: padded_len(payload.len())? as u64,
            checksum: digest(payload),
        };
        // Bound metadata growth before writing. Incomplete transactions cannot use Wire::encode.
        txn.object.chunks.push(chunk.clone());
        let fits = serde_json::to_vec(&txn.object)
            .map_err(|_| Error::Corrupt)?
            .len()
            <= MAX_ROOT;
        txn.object.chunks.pop();
        if !fits {
            return Err(Error::Capacity);
        }
        self.backend.insert(false, chunk.encoded_digest, &encoded)?;
        self.verify_chunk(&chunk)?;
        txn.object.chunks.push(chunk);
        txn.written = next;
        Ok(())
    }
    fn commit(&mut self, txn: &mut StoreTxn) -> Result<ObjectManifest> {
        self.check_txn(txn)?;
        if txn.written != txn.object.valid_bytes {
            return Err(Error::NotReady);
        }
        for c in &txn.object.chunks {
            self.verify_chunk(c)?;
        }
        let encoded = encode_extent(&txn.object.encode()?)?;
        self.backend
            .insert(true, root_id(&txn.object.key)?, &encoded)?;
        txn.published = true;
        if self.lookup(&txn.object.key)?.as_ref() != Some(&txn.object) {
            return Err(Error::Corrupt);
        }
        Ok(txn.object.clone())
    }
    fn cancel(&mut self, txn: &mut StoreTxn) -> Result<CancelState> {
        if txn.owner != self.owner {
            return Err(Error::ForeignLease);
        }
        if txn.published || self.lookup(&txn.object.key)?.as_ref() == Some(&txn.object) {
            txn.published = true;
            return Ok(CancelState::AlreadyPublished);
        }
        txn.cancelled = true;
        Ok(CancelState::PublicationRevoked)
    }
    fn lease(&mut self, m: &ObjectManifest, r: &BudgetRequest) -> Result<ObjectLease> {
        m.validate()?;
        if self.lookup(&m.key)?.as_ref() != Some(m) {
            return Err(Error::Conflict);
        }
        for c in &m.chunks {
            self.verify_chunk(c)?;
        }
        // Logical object reservation includes its framed root and all framed chunks.
        // Deduplicated physical backing can later use a shared backing pin; never undercharge today.
        let size =
            m.chunks
                .iter()
                .try_fold(encode_extent(&m.encode()?)?.len() as u64, |n, c| {
                    n.checked_add(c.storage_bytes + ALIGNMENT as u64)
                        .ok_or(Error::Overflow)
                })?;
        if r.bytes.nvme < size {
            return Err(Error::Capacity);
        }
        let charge = self.governor.borrow_mut().reserve(r)?;
        self.leases.insert(charge.id(), m.clone());
        Ok(ObjectLease {
            manifest: m.clone(),
            charge,
        })
    }
    fn read(&mut self, l: &ObjectLease, chunk: u32, destination: &mut [u8]) -> Result<u64> {
        self.check_lease(l)?;
        let c = l
            .manifest
            .chunks
            .get(chunk as usize)
            .ok_or(Error::InvalidLayout)?;
        if destination.len() < c.valid_bytes as usize {
            return Err(Error::Capacity);
        }
        let payload = self.verify_chunk(c)?;
        destination[..payload.len()].copy_from_slice(&payload);
        Ok(payload.len() as u64)
    }
    fn release(&mut self, l: &ObjectLease) -> Result<()> {
        if l.charge.state()? == contracts::ChargeState::Released {
            return Err(Error::AlreadyReleased);
        }
        if self.leases.get(&l.charge.id()) != Some(&l.manifest) {
            return Err(Error::ForeignLease);
        }
        self.governor.borrow_mut().release(&l.charge)?;
        self.leases.remove(&l.charge.id());
        Ok(())
    }
    fn evict(&mut self, key: &ObjectKey) -> Result<()> {
        if self.leases.values().any(|m| &m.key == key)
            || self.extent_leases.values().any(|(k, _, _)| k == key)
        {
            return Err(Error::Busy);
        }
        self.backend.evict_root(root_id(key)?, EvictionPermit(()))
    }
}
impl<B, G> Drop for ExtentStore<B, G> {
    fn drop(&mut self) {
        if !self.leases.is_empty() || !self.extent_leases.is_empty() {
            // Missing explicit retirement is not proof that an external reader/DMA
            // stopped. Keep process ownership (and unreleased governor charges).
            if let Some(guard) = self.ownership_guard.take() {
                std::mem::forget(guard);
            }
        }
    }
}
/// No-clobber atomic root publication. Persistent mode syncs file before publication
/// and directory after link/unlink; ephemeral mode makes no durability promise.
pub struct FileBackend {
    directory: PathBuf,
    ownership: std::sync::Arc<fs::File>,
    persistent: bool,
    read_mode: crate::io::direct::ReadMode,
    direct_writes: bool,
    io_bytes: std::cell::Cell<u64>,
    uncached_opener: Option<crate::io::direct::FileOpener>,
}
impl FileBackend {
    pub fn open(directory: &Path) -> Result<Self> {
        Self::open_mode(directory, Durability::Ephemeral)
    }
    pub fn open_mode(directory: &Path, durability: Durability) -> Result<Self> {
        // Parent must already exist so every new directory edge can be synced.
        if !directory.exists() {
            fs::create_dir(directory)?;
            if durability == Durability::Persistent {
                fs::File::open(
                    directory
                        .parent()
                        .filter(|p| !p.as_os_str().is_empty())
                        .unwrap_or(Path::new(".")),
                )?
                .sync_all()?;
            }
        }
        let ownership = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(directory.join(".ownership"))?;
        ownership.lock_shared()?;
        Ok(Self {
            ownership: std::sync::Arc::new(ownership),
            directory: directory.to_owned(),
            persistent: durability == Durability::Persistent,
            read_mode: crate::io::direct::ReadMode::Buffered,
            direct_writes: false,
            io_bytes: std::cell::Cell::new(0),
            uncached_opener: None,
        })
    }
    pub fn set_uncached_opener(&mut self, opener: crate::io::direct::FileOpener) {
        self.uncached_opener = Some(opener);
    }
    pub fn set_read_mode(&mut self, mode: crate::io::direct::ReadMode) {
        self.read_mode = mode;
    }
    /// Experimental explicit comparator, never a default or silent fallback.
    pub fn set_direct_writes(&mut self, enabled: bool) -> Result<()> {
        if enabled
            && !cfg!(all(
                target_os = "linux",
                any(target_arch = "x86_64", target_arch = "aarch64")
            ))
        {
            return Err(Error::Unsupported);
        }
        self.direct_writes = enabled;
        Ok(())
    }
    pub fn io_bytes(&self) -> u64 {
        self.io_bytes.get()
    }
    fn count(&self, n: u64) -> Result<()> {
        self.io_bytes
            .set(self.io_bytes.get().checked_add(n).ok_or(Error::Overflow)?);
        Ok(())
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
    fn ownership_guard(&self) -> Option<std::sync::Arc<fs::File>> {
        Some(self.ownership.clone())
    }
    fn evict_root(&mut self, id: Digest, _permit: EvictionPermit) -> Result<()> {
        // A different store/process keeps a shared lifetime lock even when it has
        // no lease yet. Upgrade only with exclusive ownership; never wait for GC.
        self.ownership.unlock()?;
        if self.ownership.try_lock().is_err() {
            self.ownership.lock_shared()?;
            return Err(Error::Busy);
        }
        let result = self.evict_exclusive(id);
        let relock = self.ownership.lock_shared().map_err(Error::from);
        result.and(relock)
    }
    fn persistent(&self) -> bool {
        self.persistent
    }
    fn get(&self, root: bool, id: Digest, max_bytes: usize) -> Result<Option<Vec<u8>>> {
        let file = match fs::File::open(self.path(root, id)) {
            Ok(f) => f,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(e) => return Err(e.into()),
        };
        if file.metadata()?.len() > max_bytes as u64 {
            return Err(Error::Capacity);
        }
        if self.read_mode == crate::io::direct::ReadMode::Uncached {
            let size = file.metadata()?.len() as usize;
            self.count(size as u64)?;
            return crate::io::direct::read_file(&self.path(root, id), size, self.uncached_opener)
                .map(Some);
        }
        let mut bytes = Vec::new();
        file.take(max_bytes as u64 + 1).read_to_end(&mut bytes)?;
        if bytes.len() > max_bytes {
            return Err(Error::Capacity);
        }
        self.count(bytes.len() as u64)?;
        Ok(Some(bytes))
    }
    fn insert(&mut self, root: bool, id: Digest, bytes: &[u8]) -> Result<()> {
        if root && self.directory.join(format!("tomb-{}", hex(&id))).exists() {
            return Err(Error::Conflict);
        }
        use std::sync::atomic::{AtomicU64, Ordering};
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let tmp = self.directory.join(format!(
            ".pending-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        // Allocate/validate before creating a temporary file. Direct files never
        // contain a buffered header followed by direct payload (or vice versa).
        let direct = if self.direct_writes {
            let mut buffer = crate::io::direct::AlignedBuffer::new(bytes.len())?;
            buffer.bytes_mut().copy_from_slice(bytes);
            Some(buffer)
        } else {
            None
        };
        let mut buffered;
        let direct_file;
        if direct.is_some() {
            direct_file = Some(crate::io::direct::AlignedFile::create_new(&tmp)?);
            buffered = None;
        } else {
            direct_file = None;
            buffered = Some(OpenOptions::new().write(true).create_new(true).open(&tmp)?);
        }
        let result = (|| {
            self.count(bytes.len() as u64)?;
            if let (Some(file), Some(buffer)) = (&direct_file, &direct) {
                file.write_aligned(0, buffer.bytes())?;
                if self.persistent {
                    file.sync_all()?;
                }
            } else {
                let file = buffered.as_mut().ok_or(Error::InvalidLayout)?;
                file.write_all(bytes)?;
                if self.persistent {
                    file.sync_all()?;
                }
            }
            drop(buffered);
            drop(direct_file);
            match fs::hard_link(&tmp, self.path(root, id)) {
                Ok(()) => {
                    if self.persistent {
                        fs::File::open(&self.directory)?.sync_all()?;
                    }
                    Ok(())
                }
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                    if self.get(root, id, bytes.len())?.as_deref() == Some(bytes) {
                        if self.persistent {
                            fs::File::open(self.path(root, id))?.sync_all()?;
                            fs::File::open(&self.directory)?.sync_all()?;
                        }
                        Ok(())
                    } else {
                        Err(Error::Conflict)
                    }
                }
                Err(e) => Err(e.into()),
            }
        })();
        let cleanup = fs::remove_file(tmp).map_err(Error::from);
        result.and(cleanup)?;
        if self.persistent {
            fs::File::open(&self.directory)?.sync_all()?;
        }
        Ok(())
    }
}

impl FileBackend {
    fn sync_directory(&self) -> Result<()> {
        if self.persistent {
            fs::File::open(&self.directory)?.sync_all()?;
        }
        Ok(())
    }
    fn evict_exclusive(&mut self, id: Digest) -> Result<()> {
        let root = self.path(true, id);
        let tomb = self.directory.join(format!("tomb-{}", hex(&id)));
        // Validate the entire deletion plan before revoking visibility. Recovery
        // uses the same validation for an already durable tombstone.
        let published = root.exists();
        if published && tomb.exists() {
            return Err(Error::Conflict);
        }
        if !published && !tomb.exists() {
            return Err(Error::NotFound);
        }
        let bytes = read_bounded(if published { &root } else { &tomb }, MAX_ROOT + ALIGNMENT)?;
        let removed = ObjectManifest::decode(decode_extent(&bytes)?)?;
        if removed.key.identity()? != id {
            return Err(Error::Corrupt);
        }
        // Count committed references under the exclusive process lock. Fail closed
        // on ANY malformed root; an unreadable reference must not become zero.
        let mut references = std::collections::HashMap::<Digest, u64>::new();
        for entry in fs::read_dir(&self.directory)? {
            let entry = entry?;
            if entry.file_name().to_string_lossy().starts_with("root-") && entry.path() != root {
                let bytes = read_bounded(&entry.path(), MAX_ROOT + ALIGNMENT)?;
                for c in ObjectManifest::decode(decode_extent(&bytes)?)?.chunks {
                    *references.entry(c.encoded_digest).or_default() += 1;
                }
            }
        }
        // Last irreversible publication step, after every verification above.
        if published {
            fs::rename(&root, &tomb)?;
            self.sync_directory()?;
        }
        for c in removed.chunks {
            if !references.contains_key(&c.encoded_digest) {
                remove_if_exists(&self.path(false, c.encoded_digest))?;
            }
        }
        self.sync_directory()?;
        remove_if_exists(&tomb)?;
        self.sync_directory()
    }
}
fn read_bounded(path: &Path, limit: usize) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    fs::File::open(path)?
        .take(limit as u64 + 1)
        .read_to_end(&mut bytes)?;
    if bytes.len() > limit {
        return Err(Error::Capacity);
    }
    Ok(bytes)
}
fn remove_if_exists(path: &Path) -> Result<()> {
    match fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(e.into()),
    }
}
