use super::support::*;
use memra_tier::contracts::*;
use std::collections::HashMap;
#[derive(Debug)]
pub struct Host {
    pub bytes: Vec<u8>,
    pub _pin: LeasePin,
}
impl PinnedLease for Host {
    fn storage_bytes(&self) -> u64 {
        self.bytes.len() as u64
    }
    fn valid_bytes(&self) -> u64 {
        3
    }
    fn alignment(&self) -> u32 {
        4
    }
    fn numa_node(&self) -> Option<u32> {
        Some(0)
    }
    fn bytes(&self) -> Result<&[u8]> {
        Ok(&self.bytes[..3])
    }
}
struct Entry {
    _ops: Vec<TransferOp<Host>>,
    source_retired: bool,
    source_consumer: bool,
    source_graph: bool,
    c: Completion,
    expected: Vec<Vec<SegmentExpectation>>,
    cancelled: bool,
    published: bool,
    unknown: bool,
    disk: bool,
    dma: bool,
    consumer: bool,
    graph: bool,
    target: Option<DeviceLease>,
    taken: bool,
}
pub struct Transfers {
    entries: HashMap<TransferTicket, Entry>,
    owner: DeviceOwner,
    next: u64,
    pub rejects: Vec<usize>,
    pub short: Option<usize>,
    pub gov: Shared,
}
impl Transfers {
    pub fn new(gov: Shared) -> Self {
        Self {
            entries: HashMap::new(),
            owner: DeviceOwner::new(0),
            next: 0,
            rejects: vec![],
            short: None,
            gov,
        }
    }
    pub fn read(&self) -> (ReadPlan<Host>, ChargedLease) {
        let mut r = request(0, Priority::Demand);
        r.bytes.pinned = 4;
        let charge = self.gov.borrow_mut().reserve(&r).unwrap();
        let host = Host {
            bytes: vec![3, 20, 37, 0],
            _pin: charge.pin().unwrap(),
        };
        (
            ReadPlan {
                object: key(),
                chunk: 0,
                destination: host,
                epochs: epochs(),
            },
            charge,
        )
    }
    fn submit(&mut self, ops: Vec<TransferOp<Host>>) -> Submission<TransferOp<Host>> {
        if ops.is_empty() || (0..ops.len()).all(|i| self.rejects.contains(&i)) {
            return Err(Rejected {
                op: ops,
                error: Error::Rejected,
            });
        }
        // All operations in this ticket share state and both allocation generations.
        let get_epoch = |o: &TransferOp<Host>| match o {
            TransferOp::H2d(o) | TransferOp::D2h(o) => o.epochs,
            TransferOp::P2p(o) => o.epochs,
            TransferOp::NvmeRead(o) => o.epochs,
        };
        let e = get_epoch(&ops[0]);
        if ops.iter().any(|o| get_epoch(o) != e) {
            return Err(Rejected {
                op: ops,
                error: Error::StaleEpoch,
            });
        }
        self.next += 1;
        let t = TransferTicket {
            issuer: 91,
            sequence: self.next,
            epochs: e,
        };
        let n = ops.len();
        let mut c = completion(t, n);
        let mut items = vec![];
        let mut owned = vec![];
        let mut target = None;
        for (i, op) in ops.into_iter().enumerate() {
            if self.rejects.contains(&i) {
                c.items[i].accepted = false;
                c.items[i].segments[0].status = ItemStatus::Rejected;
                items.push(ItemAcceptance::Rejected {
                    item: i as u32,
                    op,
                    error: Error::Capacity,
                });
            } else {
                if matches!(&op, TransferOp::NvmeRead(_)) {
                    c.items[i].segments[0].io_bytes =
                        (memra_tier::object_store::padded_len(3).unwrap()
                            + memra_tier::object_store::ALIGNMENT) as u64;
                }
                if let TransferOp::H2d(o) = &op {
                    c.items[i].segments[0]
                        .consumer_fence
                        .as_mut()
                        .unwrap()
                        .issuer = self.owner.issuer();
                    target = Some(self.owner.retain(&o.device).unwrap());
                    self.owner.bind_destination(t, &o.device).unwrap();
                }
                owned.push(op);
                items.push(ItemAcceptance::Accepted { item: i as u32 });
            }
        }
        if let Some(i) = self.short {
            c.items[i].segments[0].valid_bytes = 2;
        }
        self.entries.insert(
            t,
            Entry {
                _ops: owned,
                source_retired: false,
                source_consumer: true,
                source_graph: true,
                c,
                expected: expected(n),
                cancelled: false,
                published: false,
                unknown: false,
                disk: false,
                dma: false,
                consumer: false,
                graph: false,
                target,
                taken: false,
            },
        );
        Ok(BatchSubmission { ticket: t, items })
    }
    fn entry(&mut self, t: &TransferTicket) -> Result<&mut Entry> {
        self.entries.get_mut(t).ok_or(Error::UnknownTicket)
    }
    fn finish(&mut self, t: &TransferTicket) {
        let e = self.entries.get_mut(t).unwrap();
        e.disk = true;
        e.dma = true;
        e.consumer = true;
        e.graph = true;
        e.source_consumer = true;
        e.source_graph = true;
        e.unknown = false;
    }
}
impl TransferEngine for Transfers {
    type Host = Host;
    fn h2d(
        &mut self,
        o: CopyOp<Host>,
    ) -> std::result::Result<TransferTicket, Rejected<CopyOp<Host>>> {
        match self.submit(vec![TransferOp::H2d(o)]) {
            Ok(b) => Ok(b.ticket),
            Err(r) => {
                let TransferOp::H2d(op) = r.op.into_iter().next().unwrap() else {
                    unreachable!()
                };
                Err(Rejected { op, error: r.error })
            }
        }
    }
    fn d2h(
        &mut self,
        o: CopyOp<Host>,
    ) -> std::result::Result<TransferTicket, Rejected<CopyOp<Host>>> {
        match self.submit(vec![TransferOp::D2h(o)]) {
            Ok(b) => Ok(b.ticket),
            Err(r) => {
                let TransferOp::D2h(op) = r.op.into_iter().next().unwrap() else {
                    unreachable!()
                };
                Err(Rejected { op, error: r.error })
            }
        }
    }
    fn p2p(
        &mut self,
        o: ContiguousCopy,
    ) -> std::result::Result<TransferTicket, Rejected<ContiguousCopy>> {
        match self.submit(vec![TransferOp::P2p(o)]) {
            Ok(b) => Ok(b.ticket),
            Err(r) => {
                let TransferOp::P2p(op) = r.op.into_iter().next().unwrap() else {
                    unreachable!()
                };
                Err(Rejected { op, error: r.error })
            }
        }
    }
    fn nvme_read(
        &mut self,
        o: ReadPlan<Host>,
    ) -> std::result::Result<TransferTicket, Rejected<ReadPlan<Host>>> {
        match self.submit(vec![TransferOp::NvmeRead(o)]) {
            Ok(b) => Ok(b.ticket),
            Err(r) => {
                let TransferOp::NvmeRead(op) = r.op.into_iter().next().unwrap() else {
                    unreachable!()
                };
                Err(Rejected { op, error: r.error })
            }
        }
    }
    fn submit_batch(&mut self, ops: Vec<TransferOp<Host>>) -> Submission<TransferOp<Host>> {
        self.submit(ops)
    }
    fn poll(&mut self, t: &TransferTicket) -> Result<Completion> {
        let e = self.entry(t)?;
        if e.unknown {
            Err(Error::Quarantined)
        } else {
            Ok(e.c.clone())
        }
    }
    fn cancel(&mut self, t: &TransferTicket) -> Result<CancelState> {
        let e = self.entry(t)?;
        if e.published {
            Ok(CancelState::AlreadyPublished)
        } else {
            e.cancelled = true;
            Ok(CancelState::PublicationRevoked)
        }
    }
    fn ready_view(
        &mut self,
        t: &TransferTicket,
        _item: u32,
        current: Epochs,
    ) -> Result<ReadyView<'_>> {
        let e = self.entries.get_mut(t).ok_or(Error::UnknownTicket)?;
        if e.cancelled {
            return Err(Error::Cancelled);
        }
        if e.unknown {
            return Err(Error::Quarantined);
        }
        t.epochs.require(current)?;
        let ready = self.owner.ready_view(
            e.target.as_ref().ok_or(Error::NotReady)?,
            &e.c,
            &e.expected,
            current,
        )?;
        e.published = true;
        Ok(ready)
    }
    fn take_destination(
        &mut self,
        t: &TransferTicket,
        item: u32,
        current: Epochs,
    ) -> Result<Destination<Host>> {
        self.ready_view(t, item, current)?;
        let e = self.entries.get_mut(t).unwrap();
        if e.taken {
            return Err(Error::Busy);
        }
        e.taken = true;
        Ok(Destination::Device(
            self.owner.retain(e.target.as_ref().unwrap())?,
        ))
    }
    fn retire_source(&mut self, t: &TransferTicket) -> Result<()> {
        let e = self.entry(t)?;
        if e.source_retired {
            return Ok(());
        }
        if e.unknown {
            return Err(Error::Quarantined);
        }
        if !e.disk || !e.dma || !e.source_consumer || !e.source_graph {
            return Err(Error::Busy);
        }
        // This fake's additive binding is H2D only. Its retained target is the
        // independently charged destination; do not drop D2H/Read destinations.
        if !e._ops.iter().all(|op| matches!(op, TransferOp::H2d(_))) {
            return Err(Error::Unsupported);
        }
        e._ops.clear();
        e.source_retired = true;
        Ok(())
    }
    fn retire(&mut self, t: &TransferTicket, done: Option<FenceId>) -> Result<()> {
        let e = self.entry(t)?;
        if let Some(f) = done
            && (f.owner != 0 || f.generation != t.epochs.dst_gen)
        {
            return Err(Error::WrongOwner);
        }
        // Registering a last-use event is NOT observing completion.
        if !e.disk
            || !e.dma
            || !e.consumer
            || !e.graph
            || !e.source_consumer
            || !e.source_graph
            || e.unknown
        {
            return Err(Error::Busy);
        }
        Ok(())
    }
    fn retired(&mut self, t: &TransferTicket) -> Result<bool> {
        let e = self.entry(t)?;
        Ok(e.disk
            && e.dma
            && e.consumer
            && e.graph
            && e.source_consumer
            && e.source_graph
            && !e.unknown)
    }
    fn acknowledge(&mut self, t: &TransferTicket) -> Result<()> {
        if !self.retired(t)? {
            return Err(Error::Busy);
        }
        let e = self.entries.remove(t).unwrap();
        if e.target.is_some() {
            self.owner.retire_binding(t)?;
        }
        drop(e);
        Ok(())
    }
}
#[test]
fn transfer_partial_batch_and_broken_aggregate_are_detected() {
    let gov = shared();
    let mut f = Transfers::new(gov.clone());
    f.rejects = vec![2];
    f.short = Some(1);
    let mut charges = vec![];
    let ops = (0..3)
        .map(|_| {
            let (o, c) = f.read();
            charges.push(c);
            TransferOp::NvmeRead(o)
        })
        .collect();
    let b = f.submit_batch(ops).unwrap();
    b.validate(3).unwrap();
    let c = f.poll(&b.ticket).unwrap();
    assert_eq!(c.items.len(), 3);
    assert!(!c.items[2].accepted);
    assert!(matches!(
        c.require(&b.ticket, &expected(3), false),
        Err(Error::ShortIo { .. })
    ));
    let mut broken = c.clone();
    broken.items.remove(1);
    assert_eq!(
        broken.require(&b.ticket, &expected(3), false),
        Err(Error::Incomplete)
    );
    drop(b.items); // rejected input returned, never silently lost
    assert_eq!(gov.borrow_mut().release(&charges[0]), Err(Error::Busy));
    f.cancel(&b.ticket).unwrap();
    f.finish(&b.ticket);
    f.acknowledge(&b.ticket).unwrap();
    for c in charges {
        gov.borrow_mut().release(&c).unwrap();
    }
    assert_eq!(gov.borrow().used.pinned, 0);
}
#[test]
fn transfer_zero_accept_error_returns_every_input_and_empty_refuses() {
    let mut f = Transfers::new(shared());
    f.rejects = vec![0, 1];
    let (a, _) = f.read();
    let (b, _) = f.read();
    let e = f
        .submit_batch(vec![TransferOp::NvmeRead(a), TransferOp::NvmeRead(b)])
        .unwrap_err();
    assert_eq!(e.op.len(), 2);
    assert!(f.entries.is_empty());
    assert!(f.submit_batch(vec![]).is_err());
}
#[test]
fn unknown_cancel_and_delayed_fences_quarantine_until_retired() {
    let gov = shared();
    let mut f = Transfers::new(gov.clone());
    let (o, c) = f.read();
    let t = f.nvme_read(o).unwrap();
    f.entry(&t).unwrap().unknown = true;
    assert_eq!(f.poll(&t), Err(Error::Quarantined));
    f.cancel(&t).unwrap();
    assert!(!f.retired(&t).unwrap());
    assert_eq!(gov.borrow_mut().release(&c), Err(Error::Busy));
    f.entry(&t).unwrap().unknown = false;
    f.entry(&t).unwrap().disk = true;
    assert!(!f.retired(&t).unwrap());
    f.entry(&t).unwrap().dma = true;
    assert!(!f.retired(&t).unwrap());
    f.entry(&t).unwrap().consumer = true;
    assert!(!f.retired(&t).unwrap());
    f.entry(&t).unwrap().graph = true;
    assert!(f.retired(&t).unwrap());
    f.acknowledge(&t).unwrap();
    gov.borrow_mut().release(&c).unwrap();
    assert_eq!(f.retired(&t), Err(Error::UnknownTicket));
}
#[test]
fn transfer_cancel_after_completion_before_publication_revokes() {
    let mut f = Transfers::new(shared());
    let (o, _) = f.read();
    let t = f.nvme_read(o).unwrap();
    f.finish(&t);
    assert_eq!(f.cancel(&t), Ok(CancelState::PublicationRevoked));
    assert!(matches!(
        f.ready_view(&t, 0, epochs()),
        Err(Error::Cancelled)
    ));
}
#[test]
fn transfer_ready_take_is_owner_bound_and_post_publish_cannot_revoke() {
    let gov = shared();
    let mut f = Transfers::new(gov.clone());
    let (mut r, hc) = f.read();
    let mut request = request(0, Priority::MandatoryActive);
    request.bytes.device[0] = 4;
    let dc = gov.borrow_mut().reserve(&request).unwrap();
    let dev = f
        .owner
        .register(31, 4, Box::new(vec![3u8, 20, 37, 0]), &dc)
        .unwrap();
    r.epochs = epochs();
    let t = f
        .h2d(CopyOp {
            host: r.destination,
            device: dev,
            bytes: 4,
            epochs: epochs(),
            producer_fence: None,
        })
        .unwrap();
    for wrong in [
        Epochs {
            state: 8,
            ..epochs()
        },
        Epochs {
            src_gen: 20,
            ..epochs()
        },
        Epochs {
            dst_gen: 32,
            ..epochs()
        },
    ] {
        assert!(matches!(f.ready_view(&t, 0, wrong), Err(Error::StaleEpoch)));
    }
    assert_eq!(
        f.ready_view(&t, 0, epochs())
            .unwrap()
            .destination()
            .device(),
        0
    );
    assert_eq!(f.cancel(&t), Ok(CancelState::AlreadyPublished));
    let Destination::Device(taken) = f.take_destination(&t, 0, epochs()).unwrap() else {
        panic!()
    };
    assert!(matches!(
        f.take_destination(&t, 0, epochs()),
        Err(Error::Busy)
    ));
    assert_eq!(
        &*f.owner.resolve::<Vec<u8>>(&taken).unwrap(),
        &[3, 20, 37, 0]
    );
    assert_eq!(f.owner.release(&taken), Err(Error::Busy)); // accepted source/destination pins
    f.finish(&t);
    f.acknowledge(&t).unwrap();
    f.owner.release(&taken).unwrap();
    gov.borrow_mut().release(&dc).unwrap();
    gov.borrow_mut().release(&hc).unwrap();
}
#[test]
fn completion_refuses_each_epoch_bad_checksum_missing_fence_and_short_vector() {
    let t = TransferTicket {
        issuer: 91,
        sequence: 1,
        epochs: epochs(),
    };
    let c = completion(t, 1);
    for e in [
        Epochs {
            state: 8,
            ..epochs()
        },
        Epochs {
            src_gen: 20,
            ..epochs()
        },
        Epochs {
            dst_gen: 32,
            ..epochs()
        },
    ] {
        let mut bad = c.clone();
        bad.items[0].segments[0].epochs = e;
        assert_eq!(bad.require(&t, &expected(1), true), Err(Error::StaleEpoch));
    }
    let mut bad = c.clone();
    bad.items[0].segments[0].checksum = None;
    assert_eq!(bad.require(&t, &expected(1), true), Err(Error::Corrupt));
    let mut bad = c.clone();
    bad.items[0].segments[0].consumer_fence = None;
    assert_eq!(bad.require(&t, &expected(1), true), Err(Error::NotReady));
    let mut bad = c;
    bad.items[0].segments.clear();
    assert_eq!(bad.require(&t, &expected(1), true), Err(Error::Incomplete));
}
#[test]
fn reusable_transfer_schedule() {
    let mut t = Transfers::new(shared());
    let (read, _) = t.read();
    let ticket = t.nvme_read(read).unwrap();
    super::conformance::transfer_cancel(&mut t, ticket, |t, k| t.finish(k));
}
impl Drop for Transfers {
    fn drop(&mut self) {
        for (_, entry) in self.entries.drain() {
            if entry.unknown
                || !entry.disk
                || !entry.dma
                || !entry.consumer
                || !entry.graph
                || !entry.source_consumer
                || !entry.source_graph
            {
                // Deliberate test quarantine; the process owns recovery, not the caller.
                std::mem::forget(entry);
            }
        }
    }
}
#[test]
fn unknown_backend_shutdown_does_not_free_accepted_host_source() {
    let gov = shared();
    let mut transfers = Transfers::new(gov.clone());
    let (read, charge) = transfers.read();
    let ticket = transfers.nvme_read(read).unwrap();
    transfers.entry(&ticket).unwrap().unknown = true;
    drop(transfers);
    assert_eq!(gov.borrow_mut().release(&charge), Err(Error::Busy));
    assert_eq!(gov.borrow().used.pinned, 4);
}
#[test]
fn copy_bounds_and_consumer_context_generation_are_checked() {
    let gov = shared();
    let mut t = Transfers::new(gov.clone());
    let (read, _host_charge) = t.read();
    let mut req = request(0, Priority::MandatoryActive);
    req.bytes.device[0] = 4;
    let device_charge = gov.borrow_mut().reserve(&req).unwrap();
    let device = t
        .owner
        .register(31, 4, Box::new(bytes(4)), &device_charge)
        .unwrap();
    let mut op = CopyOp {
        host: read.destination,
        device,
        bytes: 5,
        epochs: epochs(),
        producer_fence: None,
    };
    assert_eq!(
        op.validate(CopyDirection::HostToDevice, epochs()),
        Err(Error::InvalidLayout)
    );
    op.bytes = 4;
    op.validate(CopyDirection::HostToDevice, epochs()).unwrap();
    let ticket = t.h2d(op).unwrap();
    t.entry(&ticket).unwrap().c.items[0].segments[0]
        .consumer_fence
        .as_mut()
        .unwrap()
        .issuer = u64::MAX;
    assert!(matches!(
        t.ready_view(&ticket, 0, epochs()),
        Err(Error::WrongOwner)
    ));
}

#[test]
fn revision_v11_complete_cancel_and_lifetime() {
    let gov = shared();
    let mut t = Transfers::new(gov.clone());
    let (read, charge) = t.read();
    let ticket = t.nvme_read(read).unwrap();
    super::conformance::transfer_complete_cancel(&mut t, ticket, &expected(1), |t, k| {
        t.entry(k).unwrap().disk = true;
        t.entry(k).unwrap().dma = true;
    });
    super::conformance::transfer_lifetime(
        &mut t,
        ticket,
        |t, k, step| {
            use super::conformance::LifetimeStep::*;
            let e = t.entry(k).unwrap();
            match step {
                Unknown => e.unknown = true,
                Producer => {
                    e.disk = true;
                    e.dma = true;
                }
                Recover => e.unknown = false,
                Consumer => e.consumer = true,
                Graph => e.graph = true,
            }
        },
        |t| t.gov.borrow().used(),
    );
    gov.borrow_mut().release(&charge).unwrap();
    assert_eq!(gov.borrow().used(), TierBudget::zero(2));
}
#[test]
fn revision_v11_indexed_acceptance() {
    for (rejected, short) in [(2, Some(1)), (0, None), (1, None)] {
        let gov = shared();
        let mut t = Transfers::new(gov.clone());
        t.rejects = vec![rejected];
        t.short = short;
        let mut charges = vec![];
        let ops = (0..3)
            .map(|_| {
                let (op, charge) = t.read();
                charges.push(charge);
                TransferOp::NvmeRead(op)
            })
            .collect();
        let b = t.submit_batch(ops).unwrap();
        let c = t.poll(&b.ticket).unwrap();
        super::conformance::acceptance(&b, &c, &expected(3), rejected, short);
        drop(b.items);
        t.cancel(&b.ticket).unwrap();
        t.finish(&b.ticket);
        t.acknowledge(&b.ticket).unwrap();
        for charge in charges {
            gov.borrow_mut().release(&charge).unwrap();
        }
        assert_eq!(gov.borrow().used(), TierBudget::zero(2));
    }
}

#[test]
fn revision_v11_zero_accept_preserves_owned_hosts() {
    let gov = shared();
    let mut t = Transfers::new(gov.clone());
    t.rejects = vec![0, 1];
    let (a, ca) = t.read();
    let (b, cb) = t.read();
    let returned = super::conformance::transfer_zero_accept(
        &mut t,
        vec![TransferOp::NvmeRead(a), TransferOp::NvmeRead(b)],
        |ops| {
            for op in ops {
                let TransferOp::NvmeRead(op) = op else {
                    panic!()
                };
                assert_eq!(op.destination.bytes().unwrap(), bytes(3));
            }
        },
    );
    assert!(t.entries.is_empty());
    drop(returned);
    gov.borrow_mut().release(&ca).unwrap();
    gov.borrow_mut().release(&cb).unwrap();
    assert_eq!(gov.borrow().used(), TierBudget::zero(2));
}

#[test]
fn revision_v12_transfer_framed_logical_bytes() {
    let gov = shared();
    let mut t = Transfers::new(gov.clone());
    let (read, charge) = t.read();
    let ticket = t.nvme_read(read).unwrap();
    assert!(t.poll(&ticket).unwrap().items[0].segments[0].io_bytes > 4);
    super::conformance::transfer_completion_bytes(&mut t, &ticket, &expected(1), false);
    t.cancel(&ticket).unwrap();
    t.finish(&ticket);
    t.acknowledge(&ticket).unwrap();
    gov.borrow_mut().release(&charge).unwrap();
    assert_eq!(gov.borrow().used, TierBudget::zero(2));
}

#[test]
fn revision_v13_source_retirement_preserves_taken_destination() {
    let gov = shared();
    let mut t = Transfers::new(gov.clone());
    let (read, hc) = t.read();
    let mut req = request(0, Priority::MandatoryActive);
    req.bytes.device[0] = 4;
    let dc = gov.borrow_mut().reserve(&req).unwrap();
    let device = t.owner.register(31, 4, Box::new(bytes(4)), &dc).unwrap();
    let ticket = t
        .h2d(CopyOp {
            host: read.destination,
            device,
            bytes: 4,
            epochs: epochs(),
            producer_fence: None,
        })
        .unwrap();
    t.entry(&ticket).unwrap().source_consumer = false;
    t.entry(&ticket).unwrap().source_graph = false;
    // Existing synchronous publication fixture; separate disk/dma flags model
    // pending lifetime observations. This is CPU ownership evidence, not CUDA.
    let Destination::Device(taken) = t.take_destination(&ticket, 0, epochs()).unwrap() else {
        panic!()
    };
    super::conformance::transfer_source_retirement(
        &mut t,
        ticket,
        |t, k, step| {
            use super::conformance::SourceStep::*;
            let e = t.entry(k).unwrap();
            match step {
                Unknown => e.unknown = true,
                Producer => {
                    e.disk = true;
                    e.dma = true;
                }
                Recover => e.unknown = false,
                SourceConsumer => e.source_consumer = true,
                SourceGraph => e.source_graph = true,
                DestinationConsumer => e.consumer = true,
                DestinationGraph => e.graph = true,
            }
        },
        |t| t.gov.borrow_mut().release(&hc),
        |t| {
            assert_eq!(t.gov.borrow().used.device[0], 4);
            assert_eq!(t.gov.borrow_mut().release(&dc), Err(Error::Busy));
            assert_eq!(*t.owner.resolve::<Vec<u8>>(&taken).unwrap(), bytes(4));
        },
    );
    t.owner.release(&taken).unwrap();
    gov.borrow_mut().release(&dc).unwrap();
    assert_eq!(gov.borrow().used, TierBudget::zero(2));
}
