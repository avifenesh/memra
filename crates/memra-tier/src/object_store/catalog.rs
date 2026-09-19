//! Indexed, sharded metadata for model-scale immutable byte objects. This is an
//! additive catalog, not a larger frozen ObjectManifest. Lookup reads one small
//! root; leasing reads one fixed-size index row. No sibling payload is validated.
//! Directories are exclusively owned while open. An unresolved lease retains
//! ownership through Drop, so another process cannot collect unknown I/O.
use super::*;
use crate::contracts::{ChargedLease, LeasePin};
use serde::{Deserialize, Serialize};
use std::cell::Cell;
use std::fs::File;
use std::sync::Arc;

const RECORD: usize = 112;
const MAX_SHARDS: usize = 64;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CatalogHead {
    version: u32,
    key: ObjectKey,
    extents: u64,
    valid_bytes: u64,
    backing_bytes: u64,
    shards: usize,
}
impl CatalogHead {
    pub fn key(&self) -> &ObjectKey {
        &self.key
    }
    pub fn extents(&self) -> u64 {
        self.extents
    }
    pub fn valid_bytes(&self) -> u64 {
        self.valid_bytes
    }
    /// Conservative declared backing, including all index rows and framed data.
    pub fn backing_bytes(&self) -> u64 {
        self.backing_bytes
    }
}

struct Resident {
    head: CatalogHead,
    charge: ChargedLease,
}
struct Active {
    object: Digest,
    _pin: LeasePin,
}
/// Explicit release is mandatory. Dropping a handle is NOT an I/O completion.
pub struct CatalogLease {
    owner: u64,
    sequence: u64,
    object: Digest,
    index: u64,
    reference: ChunkRef,
}
impl CatalogLease {
    pub fn reference(&self) -> &ChunkRef {
        &self.reference
    }
}
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CatalogCounters {
    pub root_reads: u64,
    pub index_rows: u64,
    pub payload_reads: u64,
    pub metadata_bytes: u64,
}

pub struct CatalogStore<G: BudgetGovernor> {
    directories: Vec<PathBuf>,
    locks: Vec<Arc<File>>,
    governor: Rc<RefCell<G>>,
    owner: u64,
    sequence: u64,
    residents: HashMap<Digest, Resident>,
    active: HashMap<u64, Active>,
    counters: Cell<CatalogCounters>,
}
impl<G: BudgetGovernor> CatalogStore<G> {
    /// Shard order is configuration, not device detection. All directories must
    /// exist; aliases/duplicate paths are refused by the exclusive OS locks.
    pub fn open(directories: Vec<PathBuf>, governor: Rc<RefCell<G>>) -> Result<Self> {
        if directories.is_empty() || directories.len() > MAX_SHARDS {
            return Err(Error::InvalidLayout);
        }
        let mut locks = Vec::new();
        for directory in &directories {
            let file = OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .truncate(false)
                .open(directory.join(".catalog-owner"))?;
            file.try_lock().map_err(|_| Error::Busy)?;
            locks.push(Arc::new(file));
        }
        Ok(Self {
            directories,
            locks,
            governor,
            owner: contracts::DeviceOwner::new(0).issuer(),
            sequence: 0,
            residents: HashMap::new(),
            active: HashMap::new(),
            counters: Cell::new(CatalogCounters::default()),
        })
    }
    pub fn counters(&self) -> CatalogCounters {
        self.counters.get()
    }
    pub fn shard_for(&self, index: u64) -> usize {
        (index % self.directories.len() as u64) as usize
    }
    fn root(&self, id: Digest) -> PathBuf {
        self.directories[0].join(format!("catalog-{}.root", hex(&id)))
    }
    fn tomb(&self, id: Digest) -> PathBuf {
        self.directories[0].join(format!("catalog-{}.tomb", hex(&id)))
    }
    fn index_path(&self, id: Digest, shard: usize) -> PathBuf {
        self.directories[shard].join(format!("catalog-{}-{shard}.index", hex(&id)))
    }
    pub fn extent_path(&self, head: &CatalogHead, index: u64) -> Result<PathBuf> {
        if head.shards != self.directories.len() || index >= head.extents {
            return Err(Error::InvalidLayout);
        }
        Ok(self.directories[self.shard_for(index)].join(format!(
            "catalog-{}-{index}.extent",
            hex(&head.key.identity()?)
        )))
    }
    fn sync_all(&self) -> Result<()> {
        for path in &self.directories {
            File::open(path)?.sync_all()?;
        }
        Ok(())
    }
    fn read_head(&self, path: &Path, key: &ObjectKey) -> Result<CatalogHead> {
        let encoded = read_bounded(path, 2 * ALIGNMENT)?;
        let head: CatalogHead =
            serde_json::from_slice(decode_extent(&encoded)?).map_err(|_| Error::Corrupt)?;
        if head.version != 1
            || &head.key != key
            || head.shards != self.directories.len()
            || head.extents == 0
            || head.backing_bytes < head.valid_bytes
            || head.backing_bytes
                < head
                    .extents
                    .checked_mul(RECORD as u64)
                    .ok_or(Error::Overflow)?
        {
            return Err(Error::Corrupt);
        }
        head.key.validate()?;
        let mut c = self.counters.get();
        c.root_reads += 1;
        c.metadata_bytes += encoded.len() as u64;
        self.counters.set(c);
        Ok(head)
    }
    pub fn lookup(&self, key: &ObjectKey) -> Result<Option<CatalogHead>> {
        let id = key.identity()?;
        if self.tomb(id).exists() || !self.root(id).exists() {
            return Ok(None);
        }
        self.read_head(&self.root(id), key).map(Some)
    }
    fn charge(&mut self, head: &CatalogHead, request: &BudgetRequest) -> Result<()> {
        let id = head.key.identity()?;
        if let Some(existing) = self.residents.get(&id) {
            return if existing.head == *head {
                Ok(())
            } else {
                Err(Error::Conflict)
            };
        }
        if request.bytes.nvme < head.backing_bytes {
            return Err(Error::Capacity);
        }
        let mut request = request.clone();
        request.bytes.nvme = head.backing_bytes;
        let charge = self.governor.borrow_mut().reserve(&request)?;
        self.residents.insert(
            id,
            Resident {
                head: head.clone(),
                charge,
            },
        );
        Ok(())
    }
    /// Install a trusted immutable extent census, streaming metadata only. Missing
    /// payloads stay missing: this is useful for cold checkpoints and synthetic
    /// capacity tests, NOT proof that the declared object bytes exist.
    pub fn install_index(
        &mut self,
        key: ObjectKey,
        valid_bytes: u64,
        references: impl IntoIterator<Item = ChunkRef>,
        request: &BudgetRequest,
    ) -> Result<CatalogHead> {
        let id = key.identity()?;
        // Exclusive catalog ownership proves no installer is still writing.
        // A published root is immutable; only its abandoned staging link goes.
        // Without a root/tomb, an incomplete index is private and can be retried.
        if self.tomb(id).exists() {
            return Err(Error::Conflict); // collect owns tombstoned backing
        }
        remove_if_exists(&self.root(id).with_extension("pending"))?;
        if self.root(id).exists() {
            self.sync_all()?;
            return Err(Error::Conflict);
        }
        for shard in 0..self.directories.len() {
            remove_if_exists(&self.index_path(id, shard))?;
        }
        self.sync_all()?;
        // Charge the caller's conservative ceiling BEFORE any index write. This
        // remains charged through collection, not just while consumers hold it.
        let charge = self.governor.borrow_mut().reserve(request)?;
        let result = self.write_index(key, valid_bytes, references, request.bytes.nvme);
        match result {
            Ok(head) => {
                self.residents.insert(
                    id,
                    Resident {
                        head: head.clone(),
                        charge,
                    },
                );
                Ok(head)
            }
            Err(error) => {
                let cleanup = (|| -> Result<()> {
                    remove_if_exists(&self.root(id))?;
                    remove_if_exists(&self.root(id).with_extension("pending"))?;
                    for shard in 0..self.directories.len() {
                        remove_if_exists(&self.index_path(id, shard))?;
                    }
                    self.sync_all()
                })();
                // Failed cleanup is unknown persistent backing: retain the charge.
                if cleanup.is_ok() {
                    self.governor.borrow_mut().release(&charge)?;
                }
                Err(error)
            }
        }
    }
    fn write_index(
        &self,
        key: ObjectKey,
        valid_bytes: u64,
        references: impl IntoIterator<Item = ChunkRef>,
        ceiling: u64,
    ) -> Result<CatalogHead> {
        let id = key.identity()?;
        let mut files = Vec::new();
        for shard in 0..self.directories.len() {
            files.push(
                OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(self.index_path(id, shard))?,
            );
        }
        let mut head = CatalogHead {
            version: 1,
            key,
            extents: 0,
            valid_bytes: 0,
            backing_bytes: (2 * ALIGNMENT) as u64,
            shards: files.len(),
        };
        for reference in references {
            validate_reference(&reference)?;
            let row = encode_record(id, head.extents, &reference);

            head.extents = head.extents.checked_add(1).ok_or(Error::Overflow)?;
            head.valid_bytes = head
                .valid_bytes
                .checked_add(reference.valid_bytes)
                .ok_or(Error::Overflow)?;
            head.backing_bytes = head
                .backing_bytes
                .checked_add(reference.storage_bytes)
                .and_then(|n| n.checked_add((ALIGNMENT + RECORD) as u64))
                .ok_or(Error::Overflow)?;
            if head.backing_bytes > ceiling {
                return Err(Error::Capacity);
            }
            files[self.shard_for(head.extents - 1)].write_all(&row)?;
        }
        if head.extents == 0 || head.valid_bytes != valid_bytes {
            return Err(Error::InvalidLayout);
        }
        for file in &files {
            file.sync_all()?;
        }
        self.sync_all()?;
        let encoded = encode_extent(&serde_json::to_vec(&head).map_err(|_| Error::Corrupt)?)?;
        let pending = self.root(id).with_extension("pending");
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&pending)?;
        file.write_all(&encoded)?;
        file.sync_all()?;
        fs::hard_link(&pending, self.root(id))?;
        self.sync_all()?;
        remove_if_exists(&pending)?;
        self.sync_all()?;
        Ok(head)
    }
    fn reference(&self, head: &CatalogHead, index: u64) -> Result<ChunkRef> {
        if index >= head.extents {
            return Err(Error::InvalidLayout);
        }
        let id = head.key.identity()?;
        let file = File::open(self.index_path(id, self.shard_for(index)))?;
        let offset = (index / head.shards as u64)
            .checked_mul(RECORD as u64)
            .ok_or(Error::Overflow)?;
        let mut bytes = [0; RECORD];
        crate::io::read_exact_at(&file, &mut bytes, offset)?;
        let expected = record_checksum(id, index, &bytes[..80]);
        if bytes[80..] != expected {
            return Err(Error::Corrupt);
        }
        let reference = ChunkRef {
            version: 1,
            valid_bytes: u64::from_le_bytes(bytes[..8].try_into().unwrap()),
            storage_bytes: u64::from_le_bytes(bytes[8..16].try_into().unwrap()),
            encoded_digest: bytes[16..48].try_into().unwrap(),
            checksum: bytes[48..80].try_into().unwrap(),
        };
        validate_reference(&reference)?;
        let mut c = self.counters.get();
        c.index_rows += 1;
        c.metadata_bytes += RECORD as u64;
        self.counters.set(c);
        Ok(reference)
    }
    fn current(&self, head: &CatalogHead) -> Result<Digest> {
        if self.lookup(&head.key)?.as_ref() != Some(head) {
            return Err(Error::Conflict);
        }
        head.key.identity()
    }
    /// Charges backing once, irrespective of repeated leases. Only the selected
    /// index row is read here; checksum validation of payload happens on read.
    pub fn lease_extent(
        &mut self,
        head: &CatalogHead,
        index: u64,
        request: &BudgetRequest,
    ) -> Result<CatalogLease> {
        let id = self.current(head)?;
        let reference = self.reference(head, index)?;
        self.charge(head, request)?;
        self.sequence = self.sequence.checked_add(1).ok_or(Error::Overflow)?;
        let pin = self.residents[&id].charge.pin()?;
        self.active.insert(
            self.sequence,
            Active {
                object: id,
                _pin: pin,
            },
        );
        Ok(CatalogLease {
            owner: self.owner,
            sequence: self.sequence,
            object: id,
            index,
            reference,
        })
    }
    fn check_lease(&self, lease: &CatalogLease) -> Result<&CatalogHead> {
        if lease.owner != self.owner {
            return Err(Error::ForeignLease);
        }
        let active = self
            .active
            .get(&lease.sequence)
            .ok_or(Error::AlreadyReleased)?;
        if active.object != lease.object {
            return Err(Error::ForeignLease);
        }
        Ok(&self
            .residents
            .get(&lease.object)
            .ok_or(Error::ForeignLease)?
            .head)
    }
    pub fn read_extent(&self, lease: &CatalogLease, dst: &mut [u8]) -> Result<u64> {
        let head = self.check_lease(lease)?;
        if dst.len() < lease.reference.valid_bytes as usize {
            return Err(Error::Capacity);
        }
        let bytes = read_bounded(&self.extent_path(head, lease.index)?, MAX_CHUNK + ALIGNMENT)?;
        if contracts::digest("extent", &bytes) != lease.reference.encoded_digest {
            return Err(Error::Corrupt);
        }
        let payload = decode_extent(&bytes)?;
        if payload.len() as u64 != lease.reference.valid_bytes
            || digest(payload) != lease.reference.checksum
        {
            return Err(Error::Corrupt);
        }
        dst[..payload.len()].copy_from_slice(payload);
        let mut c = self.counters.get();
        c.payload_reads += 1;
        self.counters.set(c);
        Ok(payload.len() as u64)
    }
    pub fn release_extent(&mut self, lease: &CatalogLease) -> Result<()> {
        self.check_lease(lease)?;
        self.active.remove(&lease.sequence);
        Ok(())
    }
    pub fn put_extent(&mut self, head: &CatalogHead, index: u64, payload: &[u8]) -> Result<()> {
        self.current(head)?;
        if !self.residents.contains_key(&head.key.identity()?) {
            return Err(Error::NotReady);
        }
        let reference = self.reference(head, index)?;
        let bytes = encode_extent(payload)?;
        if contracts::digest("extent", &bytes) != reference.encoded_digest
            || payload.len() as u64 != reference.valid_bytes
            || digest(payload) != reference.checksum
        {
            return Err(Error::Corrupt);
        }
        let path = self.extent_path(head, index)?;
        let pending = path.with_extension("pending-extent");
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&pending)?;
        file.write_all(&bytes)?;
        file.sync_all()?;
        let result = fs::hard_link(&pending, &path).map_err(Error::from);
        remove_if_exists(&pending)?;
        result?;
        self.sync_all()
    }
    /// Durable visibility revocation, separately exposed for deterministic crash
    /// schedules. Charges remain held until collect finishes and acknowledges GC.
    pub fn tombstone(&mut self, head: &CatalogHead) -> Result<()> {
        let id = head.key.identity()?;
        if self.active.values().any(|a| a.object == id) {
            return Err(Error::Busy);
        }
        if self.tomb(id).exists() {
            return Ok(());
        }
        self.current(head)?;
        fs::rename(self.root(id), self.tomb(id))?;
        self.sync_all()
    }
    pub fn collect(&mut self, key: &ObjectKey) -> Result<()> {
        let id = key.identity()?;
        if self.active.values().any(|a| a.object == id) {
            return Err(Error::Busy);
        }
        if !self.tomb(id).exists() {
            return Err(Error::NotReady);
        }
        let head = self.read_head(&self.tomb(id), key)?;
        for index in 0..head.extents {
            remove_if_exists(&self.extent_path(&head, index)?)?;
        }
        for shard in 0..head.shards {
            remove_if_exists(&self.index_path(id, shard))?;
        }
        remove_if_exists(&self.root(id).with_extension("pending"))?;
        self.sync_all()?;
        // Keep a tombstone until the final governor release succeeds. Replaying
        // partial unlinks is safe and missing files do not resurrect visibility.
        if let Some(resident) = self.residents.get(&id) {
            self.governor.borrow_mut().release(&resident.charge)?;
        }
        self.residents.remove(&id);
        remove_if_exists(&self.tomb(id))?;
        self.sync_all()
    }
    pub fn evict(&mut self, head: &CatalogHead) -> Result<()> {
        self.tombstone(head)?;
        self.collect(&head.key)
    }
}
impl<G: BudgetGovernor> Drop for CatalogStore<G> {
    fn drop(&mut self) {
        if !self.active.is_empty() {
            for lock in self.locks.drain(..) {
                std::mem::forget(lock);
            }
            for (_, active) in self.active.drain() {
                std::mem::forget(active);
            }
        } else {
            // Ordinary close relinquishes this manager's residency accounting.
            // A subsequent owner must re-admit before leasing/materializing.
            for resident in self.residents.values() {
                let _ = self.governor.borrow_mut().release(&resident.charge);
            }
        }
    }
}
fn validate_reference(reference: &ChunkRef) -> Result<()> {
    if reference.version != 1 {
        return Err(Error::UnsupportedVersion(reference.version));
    }
    if reference.valid_bytes == 0
        || reference.valid_bytes > MAX_CHUNK as u64
        || reference.storage_bytes != padded_len(reference.valid_bytes as usize)? as u64
    {
        return Err(Error::InvalidLayout);
    }
    Ok(())
}
fn record_checksum(id: Digest, index: u64, bytes: &[u8]) -> Digest {
    let mut input = Vec::with_capacity(120);
    input.extend_from_slice(&id);
    input.extend_from_slice(&index.to_le_bytes());
    input.extend_from_slice(bytes);
    contracts::digest("catalog-row-v1", &input)
}
fn encode_record(id: Digest, index: u64, reference: &ChunkRef) -> [u8; RECORD] {
    let mut bytes = [0; RECORD];
    bytes[..8].copy_from_slice(&reference.valid_bytes.to_le_bytes());
    bytes[8..16].copy_from_slice(&reference.storage_bytes.to_le_bytes());
    bytes[16..48].copy_from_slice(&reference.encoded_digest);
    bytes[48..80].copy_from_slice(&reference.checksum);
    let checksum = record_checksum(id, index, &bytes[..80]);
    bytes[80..].copy_from_slice(&checksum);
    bytes
}
