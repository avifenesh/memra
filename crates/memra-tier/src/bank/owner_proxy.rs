//! Send-only identities into an owner-thread registry. CUDA-owner services and
//! their Rc-backed leases never enter a cache/PP worker's Send/Sync object graph.
//! This is deliberately NOT an RPC implementation: a migrated worker refuses
//! WrongOwner instead of staging on an arbitrary thread or silently falling back.
use super::{ExpertDemand, ExpertDispatchBank, ExpertDispatchId, HostBuffer, dispatch_id};
use crate::contracts::{Digest, Epochs, Error, Result};
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
///
/// The token carries the identity the registry holds for its lease: the leased record's
/// artifact digest, the native dispatch key derived from its `BankId`, and the staging
/// ticket's epochs. `demand` refuses to mint a token for a lease that names another record
/// than the one demanded, and `with_bytes` / `finish` refuse a token whose identity does not
/// match the pending lease, so the CUDA-thread consumer can assert what it was leased before
/// it copies. All fields are `Copy`; the token stays `Send + Sync`.
#[derive(Debug, PartialEq, Eq)]
pub struct ExpertLeaseToken {
    owner: u64,
    lease: u64,
    record: ExpertDispatchId,
    artifact: Digest,
    epochs: Epochs,
}
impl ExpertLeaseToken {
    /// The `(layer, proj, expert)` key of the leased record, derived from its `BankId`.
    pub fn record(&self) -> ExpertDispatchId {
        self.record
    }
    /// The artifact digest of the leased record's tensor.
    pub fn artifact(&self) -> Digest {
        self.artifact
    }
    /// The epochs stamped on the lease's staging ticket.
    pub fn epochs(&self) -> Epochs {
        self.epochs
    }
}
/// The identity a pending lease carries, read from the registry's own `ExpertDemand`.
fn identity(demand: &ExpertDemand) -> Result<(ExpertDispatchId, Digest, Epochs)> {
    let id = demand.lease.id();
    Ok((
        dispatch_id(&id.record)?,
        id.tensor.artifact,
        demand.ticket.epochs,
    ))
}
/// A token that does not name the pending lease's identity is foreign, whatever its
/// owner and lease numbers say.
fn require(demand: &ExpertDemand, token: &ExpertLeaseToken) -> Result<()> {
    if identity(demand)? != (token.record, token.artifact, token.epochs) {
        return Err(Error::ForeignLease);
    }
    Ok(())
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
    /// The registered bank's stage clock line (`ExpertDispatchBank::stage_report`), read on
    /// the owner thread like every other call; `Ok(None)` when no clock is installed.
    pub fn stage_report(&self) -> Result<Option<String>> {
        self.access(|e| Ok(e.bank.as_ref().ok_or(Error::NotFound)?.stage_report()))
    }
    pub fn demand(&self, id: ExpertDispatchId, bytes: usize) -> Result<ExpertLeaseToken> {
        self.access(|e| {
            if e.pending.len() >= e.limit {
                return Err(Error::Capacity);
            }
            let lease = e.next_lease;
            let next = lease.checked_add(1).ok_or(Error::Overflow)?;
            let demand = e.bank.as_mut().ok_or(Error::NotFound)?.demand(id, bytes)?;
            let (record, artifact, epochs) = match identity(&demand) {
                Ok(identity) if identity.0 == id => identity,
                outcome => {
                    // The bank published a lease for a record other than the one demanded.
                    // Retire it through the bank before refusing, so no host use leaks; the
                    // demand never becomes a token.
                    e.bank.as_mut().ok_or(Error::NotFound)?.finish(demand)?;
                    return Err(match outcome {
                        Ok(_) => Error::ProgramMismatch,
                        Err(err) => err,
                    });
                }
            };
            e.pending.insert(lease, demand);
            e.next_lease = next;
            Ok(ExpertLeaseToken {
                owner: self.id,
                lease,
                record,
                artifact,
                epochs,
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
            require(demand, token)?;
            // A heap `Vec` (no buffer source) or a pooled buffer (day 47): the same bytes lent
            // the same way; the borrow ends before this returns.
            if let Ok(bytes) = demand.lease.resource::<Vec<u8>>() {
                return Ok(f(&bytes));
            }
            let buffer = demand.lease.resource::<Box<dyn HostBuffer>>()?;
            Ok(f(buffer.as_slice()))
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
            require(demand, token)?;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contracts::*;
    use std::cell::Cell;

    const EPOCHS: Epochs = Epochs {
        state: 3,
        src_gen: 5,
        dst_gen: 8,
    };
    fn tensor() -> TensorId {
        TensorId {
            version: WIRE_VERSION,
            artifact: [7; 32],
            name: "expert.weight".into(),
        }
    }
    fn layout(len: u64) -> RecordLayout {
        let segment = ByteSegment {
            version: WIRE_VERSION,
            group: 0,
            page: 0,
            owner: 0,
            role: Role::Payload,
            tensor: Some(tensor()),
            offset: 0,
            valid_bytes: len,
            storage_bytes: len,
            alignment: 1,
            encoding: EncodingId {
                version: WIRE_VERSION,
                program: digest("host-exps-qtype-v1", &2i32.to_le_bytes()),
                row_bytes: len,
            },
        };
        RecordLayout {
            version: WIRE_VERSION,
            segments: vec![segment],
            requirements: vec![GroupRequirement {
                version: WIRE_VERSION,
                group: 0,
                owner: 0,
                role: Role::Payload,
                page_count: 1,
                pages: PageRequirement::AllPages,
            }],
        }
    }
    fn record(layer: u32, projection: Projection, original_id: u32) -> RecordId {
        RecordId::Expert {
            layer,
            original_id,
            projection,
        }
    }
    /// A bank that leases the record it was built to serve, whatever the caller demanded.
    /// The lying arms below are exactly the case the token identity exists to refuse.
    struct Serving {
        record: RecordId,
        issuer: LeaseIssuer,
        finished: Rc<Cell<u32>>,
    }
    impl ExpertDispatchBank for Serving {
        fn validate(&self, _: ExpertDispatchId, _: usize) -> Result<()> {
            Ok(())
        }
        fn demand(&mut self, _: ExpertDispatchId, bytes: usize) -> Result<ExpertDemand> {
            let layout = layout(bytes as u64);
            let id = BankId {
                version: WIRE_VERSION,
                tensor: tensor(),
                record: self.record.clone(),
                layout: layout.identity()?,
            };
            let charge = self.issuer.issue(TierBudget::zero(1))?;
            let lease = BankLease::from_backend(
                id,
                layout,
                LayoutClass::Uniform,
                charge,
                Box::new(vec![0u8; bytes]),
            )
            .map_err(|rejected| rejected.error)?;
            Ok(ExpertDemand {
                ticket: TransferTicket {
                    issuer: 1,
                    sequence: 1,
                    epochs: EPOCHS,
                },
                lease,
            })
        }
        fn finish(&mut self, demand: ExpertDemand) -> Result<()> {
            self.finished.set(self.finished.get() + 1);
            drop(demand);
            Ok(())
        }
    }
    fn owner(record: RecordId) -> (ExpertBankOwner, Rc<Cell<u32>>) {
        let finished = Rc::new(Cell::new(0));
        let bank = Serving {
            record,
            issuer: LeaseIssuer::default(),
            finished: finished.clone(),
        };
        (
            ExpertBankOwner::register(Box::new(bank), 1).unwrap(),
            finished,
        )
    }

    #[test]
    fn token_carries_the_identity_the_registry_holds_for_its_lease() {
        let (mut owner, finished) = owner(record(2, Projection::Gate, 9));
        let proxy = owner.proxy();
        let token = proxy.demand((2, 0, 9), 16).unwrap();
        assert_eq!(token.record(), (2, 0, 9));
        assert_eq!(token.artifact(), [7; 32]);
        assert_eq!(token.epochs(), EPOCHS);
        assert_eq!(proxy.with_bytes(&token, |bytes| bytes.len()), Ok(16));
        proxy.finish(&token).unwrap();
        assert_eq!(finished.get(), 1);
        owner.close().unwrap();
    }
    #[test]
    fn a_lease_for_another_record_is_refused_before_a_token_exists_and_retired() {
        let (mut owner, finished) = owner(record(2, Projection::Gate, 8));
        let proxy = owner.proxy();
        assert_eq!(
            proxy.demand((2, 0, 9), 16).err(),
            Some(Error::ProgramMismatch)
        );
        assert_eq!(finished.get(), 1);
        // Nothing is pending: the registry never held the mismatched lease.
        owner.close().unwrap();
    }
    #[test]
    fn a_lease_whose_record_has_no_dispatch_id_is_refused_and_retired() {
        let (mut owner, finished) = owner(RecordId::Row(9));
        let proxy = owner.proxy();
        assert_eq!(
            proxy.demand((2, 0, 9), 16).err(),
            Some(Error::InvalidLayout)
        );
        assert_eq!(finished.get(), 1);
        owner.close().unwrap();
    }
    #[test]
    fn a_forged_token_with_the_right_numbers_and_the_wrong_identity_is_foreign() {
        let (mut owner, finished) = owner(record(2, Projection::Gate, 9));
        let proxy = owner.proxy();
        let token = proxy.demand((2, 0, 9), 16).unwrap();
        let forged = |record, artifact, epochs| ExpertLeaseToken {
            owner: token.owner,
            lease: token.lease,
            record,
            artifact,
            epochs,
        };
        for bad in [
            forged((2, 0, 8), token.artifact, token.epochs),
            forged(token.record, [8; 32], token.epochs),
            forged(
                token.record,
                token.artifact,
                Epochs {
                    state: 4,
                    ..token.epochs
                },
            ),
        ] {
            assert_eq!(proxy.with_bytes(&bad, |_| ()), Err(Error::ForeignLease));
            assert_eq!(proxy.finish(&bad), Err(Error::ForeignLease));
        }
        assert_eq!(finished.get(), 0);
        assert_eq!(proxy.with_bytes(&token, |bytes| bytes.len()), Ok(16));
        proxy.finish(&token).unwrap();
        assert_eq!(finished.get(), 1);
        owner.close().unwrap();
    }
    #[test]
    fn dispatch_id_is_the_native_key_and_refuses_what_it_cannot_name() {
        assert_eq!(
            dispatch_id(&record(u32::from(u16::MAX), Projection::Down, 255)),
            Ok((u16::MAX, 2, 255))
        );
        assert_eq!(dispatch_id(&record(3, Projection::Up, 1)), Ok((3, 1, 1)));
        assert_eq!(
            dispatch_id(&record(u32::from(u16::MAX) + 1, Projection::Gate, 0)),
            Err(Error::InvalidLayout)
        );
        assert_eq!(
            dispatch_id(&record(0, Projection::Gate, u32::from(u16::MAX) + 1)),
            Err(Error::InvalidLayout)
        );
        assert_eq!(
            dispatch_id(&record(0, Projection::Other("shared".into()), 0)),
            Err(Error::InvalidLayout)
        );
        assert_eq!(dispatch_id(&RecordId::Row(0)), Err(Error::InvalidLayout));
    }
}
