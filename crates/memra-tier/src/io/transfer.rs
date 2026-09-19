//! Bounded CPU host-read adapter. Default `drive` is synchronous; explicit
//! `enable_background` moves framed byte reads to bounded std workers. Owner-side
//! root lookup/open/admission remains synchronous, never a serving-latency claim.
//! CUDA/peer operations fail closed. No host completion can construct device readiness.
use super::{BoundedReader, ReadRequest};
use crate::contracts::*;
use crate::object_store::{
    ExtentLease,
    prepared::{BackgroundStore, PreparedExtent},
};
use crate::pool::PinnedLease as Host;
use std::collections::HashMap;
mod background;
struct Background<S> {
    reader: BoundedReader,
    pending: HashMap<TransferTicket, (TransferTicket, usize, ObjectKey, u32)>,
    prepare: fn(&mut S, &ObjectKey, u32, u64, &BudgetRequest) -> Result<PreparedExtent>,
    release: fn(&mut S, &ExtentLease) -> Result<()>,
}

struct Entry {
    ops: Vec<Option<ReadPlan<Host>>>,
    completion: Completion,
    cancelled: bool,
    published: bool,
    driven: bool,
    retired: bool,
    consumers: Vec<std::sync::Weak<()>>,
    extents: Vec<ExtentLease>,
}
pub struct CpuTransfers<S> {
    store: S,
    background: Option<Background<S>>,
    request: BudgetRequest,
    entries: HashMap<TransferTicket, Entry>,
    issuer: u64,
    sequence: u64,
    limit: usize,
    _queue_pin: LeasePin,
}
impl<S: ObjectStore> CpuTransfers<S> {
    pub fn new(
        store: S,
        request: BudgetRequest,
        limit: usize,
        queue_charge: &ChargedLease,
    ) -> Result<Self> {
        if limit == 0 || queue_charge.bytes().inflight < limit as u64 {
            return Err(Error::Capacity);
        }
        request.validate()?;
        Ok(Self {
            store,
            background: None,
            request,
            entries: HashMap::new(),
            issuer: DeviceOwner::new(0).issuer(),
            sequence: 0,
            limit,
            _queue_pin: queue_charge.pin()?,
        })
    }
    /// Explicit CPU work pump; never invent a completion before the actual byte read.
    pub fn drive(&mut self, ticket: &TransferTicket) -> Result<()> {
        if self.background.is_some() {
            return self.drive_background(ticket);
        }
        let e = self.entries.get_mut(ticket).ok_or(Error::UnknownTicket)?;
        if e.driven {
            return Ok(());
        }
        for (i, op) in e.ops.iter_mut().enumerate() {
            let Some(op) = op else {
                continue;
            };
            let s = &mut e.completion.items[i].segments[0];
            let result = if e.cancelled {
                Err(Error::Cancelled)
            } else {
                (|| {
                    let manifest = self.store.lookup(&op.object)?.ok_or(Error::NotFound)?;
                    let chunk = manifest
                        .chunks
                        .get(op.chunk as usize)
                        .ok_or(Error::InvalidLayout)?;
                    if chunk.valid_bytes != op.destination.valid_bytes() as u64 {
                        return Err(Error::InvalidLayout);
                    }
                    let expected = chunk.checksum;
                    let lease = self.store.lease(&manifest, &self.request)?;
                    let read = op
                        .destination
                        .read_into(|dst| self.store.read(&lease, op.chunk, dst));
                    let release = self.store.release(&lease);
                    let n = read?;
                    release?;
                    let actual = checksum(op.destination.bytes()?);
                    if actual != expected {
                        return Err(Error::Corrupt);
                    }
                    // Submitted framed chunk bytes, not whole-store validation I/O or physical traffic.
                    s.io_bytes = chunk.storage_bytes + crate::object_store::ALIGNMENT as u64;
                    s.valid_bytes = n;
                    s.checksum = Some(actual);
                    Ok(())
                })()
            };
            s.producer_done = true;
            match result {
                Ok(()) => s.status = ItemStatus::Complete,
                Err(error) => {
                    s.status = if error == Error::Cancelled {
                        ItemStatus::Cancelled
                    } else {
                        ItemStatus::Failed
                    };
                    s.error = Some(error);
                }
            }
        }
        e.driven = true;
        e.completion.producer_done = true;
        Ok(())
    }
    fn entry(&mut self, t: &TransferTicket) -> Result<&mut Entry> {
        self.entries.get_mut(t).ok_or(Error::UnknownTicket)
    }
}
#[allow(clippy::result_large_err)]
impl<S: ObjectStore> TransferEngine for CpuTransfers<S> {
    type Host = Host;
    fn h2d(
        &mut self,
        op: CopyOp<Host>,
    ) -> std::result::Result<TransferTicket, Rejected<CopyOp<Host>>> {
        Err(Rejected {
            op,
            error: Error::Unsupported,
        })
    }
    fn d2h(
        &mut self,
        op: CopyOp<Host>,
    ) -> std::result::Result<TransferTicket, Rejected<CopyOp<Host>>> {
        Err(Rejected {
            op,
            error: Error::Unsupported,
        })
    }
    fn p2p(
        &mut self,
        op: ContiguousCopy,
    ) -> std::result::Result<TransferTicket, Rejected<ContiguousCopy>> {
        Err(Rejected {
            op,
            error: Error::Unsupported,
        })
    }
    fn nvme_read(
        &mut self,
        op: ReadPlan<Host>,
    ) -> std::result::Result<TransferTicket, Rejected<ReadPlan<Host>>> {
        match self.submit_batch(vec![TransferOp::NvmeRead(op)]) {
            Ok(s) => Ok(s.ticket),
            Err(r) => {
                let TransferOp::NvmeRead(op) = r.op.into_iter().next().unwrap() else {
                    unreachable!()
                };
                Err(Rejected { op, error: r.error })
            }
        }
    }
    fn submit_batch(&mut self, ops: Vec<TransferOp<Host>>) -> Submission<TransferOp<Host>> {
        let epochs_of = |o: &TransferOp<Host>| match o {
            TransferOp::NvmeRead(o) => o.epochs,
            TransferOp::H2d(o) | TransferOp::D2h(o) => o.epochs,
            TransferOp::P2p(o) => o.epochs,
        };
        let error = if ops.is_empty() {
            Some(Error::EmptyBatch)
        } else if self.entries.len() >= self.limit
            || ops.len()
                > self.limit.saturating_sub(
                    self.entries
                        .values()
                        .map(|e| e.completion.items.len())
                        .sum::<usize>(),
                )
        {
            Some(Error::Capacity)
        } else if ops.iter().any(|o| epochs_of(o) != epochs_of(&ops[0])) {
            Some(Error::StaleEpoch)
        } else if !ops.iter().any(|o| matches!(o, TransferOp::NvmeRead(_))) {
            Some(Error::Unsupported)
        } else if self.sequence == u64::MAX {
            Some(Error::Overflow)
        } else {
            None
        };
        if let Some(error) = error {
            return Err(Rejected { op: ops, error });
        }
        self.sequence += 1;
        let ticket = TransferTicket {
            issuer: self.issuer,
            sequence: self.sequence,
            epochs: epochs_of(&ops[0]),
        };
        let mut owned = vec![];
        let mut items = vec![];
        let mut outcomes = vec![];
        for (i, op) in ops.into_iter().enumerate() {
            let accepted = matches!(op, TransferOp::NvmeRead(_));
            outcomes.push(ItemOutcome {
                item: i as u32,
                accepted,
                segments: vec![SegmentCompletion {
                    segment: 0,
                    status: if accepted {
                        ItemStatus::Pending
                    } else {
                        ItemStatus::Rejected
                    },
                    valid_bytes: 0,
                    io_bytes: 0,
                    checksum: None,
                    epochs: ticket.epochs,
                    producer_done: false,
                    consumer_fenced: false,
                    consumer_fence: None,
                    error: if accepted {
                        None
                    } else {
                        Some(Error::Unsupported)
                    },
                }],
            });
            match op {
                TransferOp::NvmeRead(op) => {
                    owned.push(Some(op));
                    items.push(ItemAcceptance::Accepted { item: i as u32 });
                }
                op => {
                    owned.push(None);
                    items.push(ItemAcceptance::Rejected {
                        item: i as u32,
                        op,
                        error: Error::Unsupported,
                    });
                }
            }
        }
        self.entries.insert(
            ticket,
            Entry {
                ops: owned,
                completion: Completion {
                    ticket,
                    items: outcomes,
                    producer_done: false,
                    consumer_fenced: false,
                },
                cancelled: false,
                published: false,
                driven: false,
                retired: false,
                consumers: vec![],
                extents: vec![],
            },
        );
        if self.background.is_some() {
            // Admission already accepted: preparation failures become item outcomes,
            // NEVER submission Err that would falsely return live inputs.
            self.drive_background(&ticket)
                .expect("newly registered ticket");
        }
        Ok(BatchSubmission { ticket, items })
    }
    fn poll(&mut self, t: &TransferTicket) -> Result<Completion> {
        self.progress()?;
        Ok(self.entry(t)?.completion.clone())
    }
    fn cancel(&mut self, t: &TransferTicket) -> Result<CancelState> {
        let e = self.entry(t)?;
        if e.published {
            return Ok(CancelState::AlreadyPublished);
        }
        e.cancelled = true;
        Ok(CancelState::PublicationRevoked)
    }
    fn ready_view(&mut self, t: &TransferTicket, _: u32, current: Epochs) -> Result<ReadyView<'_>> {
        let e = self.entry(t)?;
        t.epochs.require(current)?;
        if e.cancelled {
            return Err(Error::Cancelled);
        }
        Err(Error::Unsupported) // host bytes are NOT device-ready
    }
    fn take_destination(
        &mut self,
        t: &TransferTicket,
        item: u32,
        current: Epochs,
    ) -> Result<Destination<Host>> {
        let e = self.entry(t)?;
        t.epochs.require(current)?;
        if e.cancelled {
            return Err(Error::Cancelled);
        }
        if e.retired {
            return Err(Error::AlreadyReleased);
        }
        if !e.driven {
            return Err(Error::NotReady);
        }
        // Refuse publication of an incomplete/partially failed promised batch.
        for item in &e.completion.items {
            if !item.accepted {
                return Err(Error::Rejected);
            }
            for s in &item.segments {
                if s.status != ItemStatus::Complete {
                    return Err(s.error.clone().unwrap_or(Error::NotReady));
                }
            }
        }
        let op = e
            .ops
            .get_mut(item as usize)
            .ok_or(Error::InvalidLayout)?
            .take()
            .ok_or(Error::AlreadyReleased)?;
        e.consumers.push(op.destination.lifetime());
        e.published = true;
        Ok(Destination::Host(op.destination))
    }
    fn retire(&mut self, t: &TransferTicket, consumer_done: Option<FenceId>) -> Result<()> {
        if consumer_done.is_some() {
            return Err(Error::WrongOwner);
        } // no CUDA fence authority
        self.progress()?;
        let e = self.entries.get_mut(t).ok_or(Error::UnknownTicket)?;
        if !e.driven || e.consumers.iter().any(|w| w.strong_count() != 0) {
            return Err(Error::Busy);
        }
        if let Some(bg) = &self.background {
            while let Some(lease) = e.extents.last() {
                (bg.release)(&mut self.store, lease)?;
                e.extents.pop();
            }
        }
        e.ops.clear();
        e.retired = true;
        Ok(())
    }
    fn retired(&mut self, t: &TransferTicket) -> Result<bool> {
        Ok(self.entry(t)?.retired)
    }
    fn acknowledge(&mut self, t: &TransferTicket) -> Result<()> {
        if !self.retired(t)? {
            return Err(Error::Busy);
        }
        self.entries.remove(t);
        Ok(())
    }
}
