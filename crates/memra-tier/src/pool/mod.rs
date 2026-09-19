//! CPU fake of startup-allocated aligned pinned slots. NOT CUDA-pinned memory.
//! Lease ownership travels to I/O and is returned only after completion. No per-lease
//! backing allocation. Admission must charge this physical pool once, not per caller.
use crate::contracts::{ChargedLease, Error, LeasePin, Result};
use std::sync::{Arc, Mutex};

struct Slot {
    backing: Vec<u8>,
    start: usize,
    bytes: usize,
}
impl Slot {
    fn new(bytes: usize, alignment: usize) -> Result<Self> {
        let capacity = bytes.checked_add(alignment - 1).ok_or(Error::Overflow)?;
        let mut backing = Vec::new();
        backing
            .try_reserve_exact(capacity)
            .map_err(|_| Error::Capacity)?;
        backing.resize(capacity, 0);
        let start = (alignment - (backing.as_ptr() as usize % alignment)) % alignment;
        Ok(Self {
            backing,
            start,
            bytes,
        })
    }
    fn data(&self) -> &[u8] {
        &self.backing[self.start..self.start + self.bytes]
    }
    fn data_mut(&mut self) -> &mut [u8] {
        &mut self.backing[self.start..self.start + self.bytes]
    }
}
struct Inner {
    pin: Option<LeasePin>,
    free: Vec<Slot>,
    quarantined: Vec<Slot>,
}
impl Drop for Inner {
    fn drop(&mut self) {
        if !self.quarantined.is_empty() {
            // No all-use retirement authority exists in this CPU fake. Unknown
            // shutdown must retain BOTH backing and its governor pin, even after
            // the last pool handle disappears. Intentionally retain until process
            // exit; never invent recovery from Drop, timeout, or a cancelled bit.
            let unknown = (std::mem::take(&mut self.quarantined), self.pin.take());
            std::mem::forget(unknown);
        }
    }
}
#[derive(Clone)]
pub struct FakePinnedPool {
    inner: Arc<Mutex<Inner>>,
    slots: usize,
    slot_bytes: usize,
    alignment: usize,
    reserved_demand_slots: usize,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Admission {
    Demand,
    Optional,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PoolAccounting {
    pub usable_bytes: usize,
    pub backing_bytes: usize,
    pub leased_bytes: usize,
    pub quarantined_bytes: usize,
    pub free_slots: usize,
}
impl FakePinnedPool {
    pub fn new(
        slots: usize,
        slot_bytes: usize,
        alignment: usize,
        reserved_demand_slots: usize,
        charge: &ChargedLease,
    ) -> Result<Self> {
        if slots == 0
            || slot_bytes == 0
            || !alignment.is_power_of_two()
            || alignment > u32::MAX as usize
            || !slot_bytes.is_multiple_of(alignment)
            || reserved_demand_slots > slots
        {
            return Err(Error::InvalidLayout);
        }
        slots
            .checked_mul(
                slot_bytes
                    .checked_add(alignment - 1)
                    .ok_or(Error::Overflow)?,
            )
            .ok_or(Error::Overflow)?;
        let backing_bytes = slots * (slot_bytes + alignment - 1);
        if charge.bytes().pageable < backing_bytes as u64 {
            return Err(Error::Capacity);
        }
        let pin = charge.pin()?;
        let free = (0..slots)
            .map(|_| Slot::new(slot_bytes, alignment))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            inner: Arc::new(Mutex::new(Inner {
                pin: Some(pin),
                free,
                quarantined: Vec::new(),
            })),
            slots,
            slot_bytes,
            alignment,
            reserved_demand_slots,
        })
    }
    pub fn acquire(&self, valid_bytes: usize, admission: Admission) -> Result<PinnedLease> {
        if valid_bytes > self.slot_bytes {
            return Err(Error::Capacity);
        }
        let mut inner = self.inner.lock().map_err(|_| Error::Quarantined)?;
        let floor = if admission == Admission::Optional {
            self.reserved_demand_slots
        } else {
            0
        };
        if inner.free.len() <= floor {
            return Err(Error::Capacity);
        }
        let mut slot = inner.free.pop().ok_or(Error::Capacity)?;
        slot.data_mut().fill(0xa5); // deterministic fake poison; never visible before initialization
        Ok(PinnedLease {
            lifetime: Arc::new(()),
            pool: self.clone(),
            slot: Some(slot),
            valid_bytes,
            initialized: false,
        })
    }
    pub fn accounting(&self) -> PoolAccounting {
        let inner = self.inner.lock().expect("fake pool lock poisoned");
        PoolAccounting {
            usable_bytes: self.slots * self.slot_bytes,
            backing_bytes: self.slots * (self.slot_bytes + self.alignment - 1),
            leased_bytes: (self.slots - inner.free.len() - inner.quarantined.len())
                * self.slot_bytes,
            quarantined_bytes: inner.quarantined.len() * self.slot_bytes,
            free_slots: inner.free.len(),
        }
    }
}
/// Exclusive, non-Clone lease; dropping an idle lease returns its slot, not backing quota.
/// GPU integration must move it into a retirement record BEFORE submitting DMA.
pub struct PinnedLease {
    lifetime: Arc<()>,
    pool: FakePinnedPool,
    slot: Option<Slot>,
    valid_bytes: usize,
    initialized: bool,
}
impl std::fmt::Debug for PinnedLease {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PinnedLease")
            .field("valid_bytes", &self.valid_bytes)
            .field("initialized", &self.initialized)
            .finish_non_exhaustive()
    }
}
impl PinnedLease {
    pub(crate) fn lifetime(&self) -> std::sync::Weak<()> {
        Arc::downgrade(&self.lifetime)
    }
    /// Explicit idle-lease release. In-flight owners must retain this value until
    /// disk/DMA/consumer retirement; cancel cannot call this behind their back.
    pub fn release(self) {
        drop(self);
    }
    pub fn valid_bytes(&self) -> usize {
        self.valid_bytes
    }
    pub fn storage_bytes(&self) -> usize {
        self.pool.slot_bytes
    }
    pub fn alignment(&self) -> usize {
        self.pool.alignment
    }
    pub fn write(&mut self, bytes: &[u8]) -> Result<()> {
        self.initialized = false;
        if bytes.len() != self.valid_bytes {
            return Err(Error::InvalidLayout);
        }
        let slot = self.slot.as_mut().ok_or(Error::Quarantined)?;
        slot.data_mut().fill(0);
        slot.data_mut()[..bytes.len()].copy_from_slice(bytes);
        self.initialized = true;
        Ok(())
    }
    pub fn bytes(&self) -> Result<&[u8]> {
        if !self.initialized {
            return Err(Error::NotReady);
        }
        Ok(&self.slot.as_ref().ok_or(Error::Quarantined)?.data()[..self.valid_bytes])
    }
    pub(crate) fn read_into(&mut self, read: impl FnOnce(&mut [u8]) -> Result<u64>) -> Result<u64> {
        self.initialized = false;
        let slot = self.slot.as_mut().ok_or(Error::Quarantined)?;
        slot.data_mut().fill(0);
        let n = read(&mut slot.data_mut()[..self.valid_bytes])?;
        if n != self.valid_bytes as u64 {
            return Err(Error::ShortIo {
                expected: self.valid_bytes as u64,
                actual: n,
            });
        }
        self.initialized = true;
        Ok(n)
    }
    /// Fake unknown-completion gate. Quarantined slots never re-enter the free list.
    pub fn quarantine(mut self) {
        if let Some(slot) = self.slot.take() {
            self.pool
                .inner
                .lock()
                .expect("fake pool lock poisoned")
                .quarantined
                .push(slot);
        }
    }
}
impl Drop for PinnedLease {
    fn drop(&mut self) {
        if let Some(slot) = self.slot.take() {
            self.pool
                .inner
                .lock()
                .expect("fake pool lock poisoned")
                .free
                .push(slot);
        }
    }
}

impl crate::contracts::PinnedLease for PinnedLease {
    fn storage_bytes(&self) -> u64 {
        self.storage_bytes() as u64
    }
    fn valid_bytes(&self) -> u64 {
        self.valid_bytes() as u64
    }
    fn alignment(&self) -> u32 {
        self.alignment() as u32
    }
    fn numa_node(&self) -> Option<u32> {
        None
    }
    fn bytes(&self) -> Result<&[u8]> {
        self.bytes()
    }
}
