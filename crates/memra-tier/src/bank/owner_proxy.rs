//! Send-only identities into an owner-thread registry. CUDA-owner services and
//! their Rc-backed leases never enter a cache/PP worker's Send/Sync object graph.
//! This is deliberately NOT an RPC implementation: a migrated worker refuses
//! WrongOwner instead of staging on an arbitrary thread or silently falling back.
use super::{ExpertDemand, ExpertDispatchBank, ExpertDispatchId};
use crate::contracts::{Error, Result};
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::marker::PhantomData;
use std::rc::Rc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread::{self, ThreadId};

static NEXT_OWNER: AtomicU64 = AtomicU64::new(1);
thread_local! {
    static OWNERS: RefCell<BTreeMap<u64, Entry>> = const { RefCell::new(BTreeMap::new()) };
}
struct Entry {
    bank: Option<Box<dyn ExpertDispatchBank>>,
    pending: BTreeMap<u64, ExpertDemand>,
    next_lease: u64,
    limit: usize,
}
impl Drop for Entry {
    fn drop(&mut self) {
        if !self.pending.is_empty() {
            // Thread exit or owner teardown is not proof of DMA retirement.
            // Retain backing AND governor charges on unknown completion.
            std::mem::forget(std::mem::take(&mut self.pending));
            std::mem::forget(self.bank.take());
        }
    }
}
/// Registration must outlive all cache users. The Rc marker forbids moving this
/// guard to a worker; only its proxy is transferable. Close refuses open leases.
pub struct ExpertBankOwner {
    proxy: ExpertBankProxy,
    _owner_only: PhantomData<Rc<()>>,
}
#[derive(Clone, Debug)]
pub struct ExpertBankProxy {
    owner: ThreadId,
    id: u64,
}
/// Host-ready token, NOT a device-ready fence. A token contains no pointer or Rc.
/// The cache must observe its CUDA copy completion before calling finish.
#[derive(Debug, PartialEq, Eq)]
pub struct ExpertLeaseToken {
    owner: u64,
    lease: u64,
}
impl ExpertBankOwner {
    pub fn register(bank: Box<dyn ExpertDispatchBank>, max_pending: usize) -> Result<Self> {
        if max_pending == 0 {
            return Err(Error::Capacity);
        }
        let id = NEXT_OWNER
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |v| v.checked_add(1))
            .map_err(|_| Error::Overflow)?;
        OWNERS.with(|owners| {
            owners.borrow_mut().insert(
                id,
                Entry {
                    bank: Some(bank),
                    pending: BTreeMap::new(),
                    next_lease: 1,
                    limit: max_pending,
                },
            );
        });
        Ok(Self {
            proxy: ExpertBankProxy {
                owner: thread::current().id(),
                id,
            },
            _owner_only: PhantomData,
        })
    }
    pub fn proxy(&self) -> ExpertBankProxy {
        self.proxy.clone()
    }
    pub fn close(&mut self) -> Result<()> {
        self.proxy.access(|entry| {
            if !entry.pending.is_empty() {
                return Err(Error::Busy);
            }
            Ok(())
        })?;
        OWNERS.with(|owners| {
            owners.borrow_mut().remove(&self.proxy.id);
        });
        Ok(())
    }
}
impl Drop for ExpertBankOwner {
    fn drop(&mut self) {
        // Entry's Drop deliberately retains unknown transfers, even at TLS exit.
        OWNERS.with(|owners| {
            owners.borrow_mut().remove(&self.proxy.id);
        });
    }
}
impl ExpertBankProxy {
    fn access<T>(&self, f: impl FnOnce(&mut Entry) -> Result<T>) -> Result<T> {
        if self.owner != thread::current().id() {
            return Err(Error::WrongOwner);
        }
        OWNERS.with(|owners| {
            let mut owners = owners.try_borrow_mut().map_err(|_| Error::Busy)?;
            f(owners.get_mut(&self.id).ok_or(Error::NotFound)?)
        })
    }
    pub fn validate(&self, id: ExpertDispatchId, bytes: usize) -> Result<()> {
        self.access(|e| e.bank.as_ref().ok_or(Error::NotFound)?.validate(id, bytes))
    }
    pub fn demand(&self, id: ExpertDispatchId, bytes: usize) -> Result<ExpertLeaseToken> {
        self.access(|e| {
            if e.pending.len() >= e.limit {
                return Err(Error::Capacity);
            }
            let lease = e.next_lease;
            let next = lease.checked_add(1).ok_or(Error::Overflow)?;
            let demand = e.bank.as_mut().ok_or(Error::NotFound)?.demand(id, bytes)?;
            e.pending.insert(lease, demand);
            e.next_lease = next;
            Ok(ExpertLeaseToken {
                owner: self.id,
                lease,
            })
        })
    }
    /// The closure may enqueue H2D but cannot extract a borrowed slice from here.
    /// The registry retains its lease until finish, including after closure error.
    pub fn with_bytes<T>(&self, token: &ExpertLeaseToken, f: impl FnOnce(&[u8]) -> T) -> Result<T> {
        if token.owner != self.id {
            return Err(Error::ForeignLease);
        }
        self.access(|e| {
            let demand = e.pending.get(&token.lease).ok_or(Error::UnknownTicket)?;
            let bytes = demand.lease.resource::<Vec<u8>>()?;
            Ok(f(&bytes))
        })
    }
    /// Explicit completion observation by the CUDA owner, never Drop. On error
    /// preserve the token/lease for retry; no byte backing is released early.
    pub fn finish(&self, token: &ExpertLeaseToken) -> Result<()> {
        if token.owner != self.id {
            return Err(Error::ForeignLease);
        }
        self.access(|e| {
            let demand = e.pending.get(&token.lease).ok_or(Error::UnknownTicket)?;
            // Retain the original alias if finish fails partway through retirement.
            let submitted = ExpertDemand {
                ticket: demand.ticket,
                lease: demand.lease.clone(),
            };
            e.bank.as_mut().ok_or(Error::NotFound)?.finish(submitted)?;
            e.pending.remove(&token.lease);
            Ok(())
        })
    }
}
