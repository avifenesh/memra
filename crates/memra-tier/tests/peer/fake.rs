//! D's CPU byte-moving fake. These are NOT CUDA allocations, events, or P2P receipts.
use super::support::*;
use memra_tier::{
    contracts::*,
    peer::{topology::LinkHealth, validate_peer_charge},
};
use std::{
    cell::RefCell,
    collections::{HashMap, HashSet},
};

struct Entry {
    // Keep original item indices even when a middle operation was rejected.
    copies: Vec<Option<ContiguousCopy>>,
    completion: Completion,
    expected: Vec<Vec<SegmentExpectation>>,
    cancelled: bool,
    published: bool,
    dma: bool,
    consumer: bool,
    graph: bool,
    unknown: bool,
}
// Test-owned live route state, keyed by owner/source -> consumer/destination.
// These injected observations are not driver grants or hardware qualification.
struct RouteState {
    context_granted: bool,
    pool_granted: bool,
    link_health: LinkHealth,
}
impl RouteState {
    fn available(&self) -> bool {
        self.context_granted && self.pool_granted && self.link_health == LinkHealth::AtMaximum
    }
}
struct Peer {
    gov: Shared,
    owners: [DeviceOwner; 2],
    entries: HashMap<TransferTicket, Entry>,
    charges: HashMap<(u64, u64), bool>, // device released, governor release still retryable
    next: u64,
    reject: HashSet<usize>,
    routes: HashMap<(u32, u32), RouteState>,
    state: u64,
}
impl Peer {
    fn new(gov: Shared) -> Self {
        Self {
            gov,
            owners: [DeviceOwner::new(0), DeviceOwner::new(1)],
            entries: HashMap::new(),
            charges: HashMap::new(),
            next: 0,
            reject: HashSet::new(),
            routes: [(0, 1), (1, 0)]
                .into_iter()
                .map(|route| {
                    (
                        route,
                        RouteState {
                            context_granted: true,
                            pool_granted: true,
                            link_health: LinkHealth::AtMaximum,
                        },
                    )
                })
                .collect(),
            state: 7,
        }
    }
    fn route_available(&self, source: u32, destination: u32) -> bool {
        self.routes
            .get(&(source, destination))
            .is_some_and(RouteState::available)
    }
    fn route_fault(
        &mut self,
        route: (u32, u32),
        kind: super::conformance::RouteFault,
        denied: bool,
    ) {
        use super::conformance::RouteFault;
        let state = self.routes.get_mut(&route).unwrap();
        match kind {
            RouteFault::Context => state.context_granted = !denied,
            RouteFault::Pool => state.pool_granted = !denied,
            RouteFault::Downgrade => {
                state.link_health = if denied {
                    LinkHealth::Downgraded
                } else {
                    LinkHealth::AtMaximum
                };
            }
        }
    }
    fn plan(&self, device: u32, generation: u64) -> PeerPlan {
        let mut r = request(0, Priority::Demand);
        r.bytes.device[device as usize] = 4;
        r.bytes.peer[device as usize] = 4;
        PeerPlan {
            owner_device: device,
            consumer_device: 1 - device,
            bytes: 4,
            alignment: 4,
            epochs: Epochs {
                dst_gen: generation,
                ..epochs()
            },
            request: r,
        }
    }
    fn copy(&self, src: &PeerLease, dst: &PeerLease) -> ContiguousCopy {
        let s = src.device.device() as usize;
        let d = dst.device.device() as usize;
        ContiguousCopy::new(
            self.owners[s].retain(&src.device).unwrap(),
            self.owners[d].retain(&dst.device).unwrap(),
            ContiguousSpan::new(0, 4, 4).unwrap(),
            ContiguousSpan::new(0, 4, 4).unwrap(),
            Epochs {
                state: self.state,
                src_gen: src.device.generation(),
                dst_gen: dst.device.generation(),
            },
            FenceId {
                issuer: self.owners[s].issuer(),
                owner: s as u32,
                generation: src.device.generation(),
                sequence: 1,
            },
        )
        .unwrap()
    }
    fn finish(&mut self, t: &TransferTicket) {
        let e = self.entries.get_mut(t).unwrap();
        for (i, copy) in e.copies.iter().enumerate() {
            let Some(copy) = copy else { continue };
            let s = copy.source().device() as usize;
            let d = copy.destination().device() as usize;
            let bytes = self.owners[s]
                .resolve::<RefCell<Vec<u8>>>(copy.source())
                .unwrap()
                .borrow()
                .clone();
            let start = copy.source_span().offset() as usize;
            let n = copy.source_span().bytes() as usize;
            let out = self.owners[d]
                .resolve::<RefCell<Vec<u8>>>(copy.destination())
                .unwrap();
            let lo = copy.destination_span().offset() as usize;
            out.borrow_mut()[lo..lo + n].copy_from_slice(&bytes[start..start + n]);
            let segment = &mut e.completion.items[i].segments[0];
            segment.status = ItemStatus::Complete;
            segment.valid_bytes = n as u64;
            segment.io_bytes = n as u64;
            segment.checksum = Some(checksum(&out.borrow()[lo..lo + n]));
            segment.producer_done = true;
            segment.consumer_fenced = true;
        }
        e.completion.producer_done = true;
        e.completion.consumer_fenced = true;
        e.dma = true;
    }
    fn drain(&mut self, t: &TransferTicket) {
        self.finish(t);
        let e = self.entries.get_mut(t).unwrap();
        e.unknown = false;
        e.consumer = true;
        e.graph = true;
        self.acknowledge(t).unwrap();
    }
    fn fence(&self, t: &TransferTicket, device: u32) -> FenceId {
        FenceId {
            issuer: self.owners[device as usize].issuer(),
            owner: device,
            generation: t.epochs.dst_gen,
            sequence: 1,
        }
    }
}
impl PeerCapacity for Peer {
    fn reserve(&mut self, plan: PeerPlan) -> Result<PeerLease> {
        if !self.route_available(plan.owner_device, plan.consumer_device) {
            return Err(Error::Unsupported);
        }
        if plan.epochs.state != self.state {
            return Err(Error::StaleEpoch);
        }
        validate_peer_charge(&plan)?;
        let charge = self.gov.borrow_mut().reserve(&plan.request)?;
        let owner = plan.owner_device as usize;
        let result = self.owners[owner].register(
            plan.epochs.dst_gen,
            plan.bytes,
            Box::new(RefCell::new(if owner == 0 { bytes(4) } else { vec![0; 4] })),
            &charge,
        );
        match result {
            Ok(device) => {
                self.charges.insert(charge.id(), false);
                Ok(PeerLease {
                    plan,
                    charge,
                    device,
                })
            }
            Err(error) => {
                self.gov.borrow_mut().release(&charge)?;
                Err(error)
            }
        }
    }
    fn release(&mut self, lease: &PeerLease) -> Result<()> {
        let released = self
            .charges
            .get_mut(&lease.charge.id())
            .ok_or(Error::ForeignLease)?;
        if !matches!(
            lease.charge.state()?,
            ChargeState::Reserved | ChargeState::Retired
        ) {
            return Err(Error::Busy);
        }
        if !*released {
            self.owners[lease.device.device() as usize].release(&lease.device)?;
            *released = true;
        }
        self.gov.borrow_mut().release(&lease.charge)?;
        self.charges.remove(&lease.charge.id());
        Ok(())
    }
}
impl PeerBackend for Peer {
    fn submit(&mut self, copies: Vec<ContiguousCopy>) -> Submission<ContiguousCopy> {
        let error = if copies.is_empty() {
            Some(Error::EmptyBatch)
        } else if copies
            .iter()
            .any(|c| !self.route_available(c.source().device(), c.destination().device()))
        {
            Some(Error::Unsupported)
        } else if copies[0].epochs.state != self.state
            || copies.iter().any(|c| c.validate(copies[0].epochs).is_err())
        {
            Some(Error::StaleEpoch)
        } else if (0..copies.len()).all(|i| self.reject.contains(&i)) {
            Some(Error::Rejected)
        } else {
            None
        };
        if let Some(error) = error {
            return Err(Rejected { op: copies, error });
        }
        // Preflight all owners before accepting any operation.
        if copies.iter().any(|c| {
            self.owners
                .get(c.source().device() as usize)
                .is_none_or(|o| o.resolve::<RefCell<Vec<u8>>>(c.source()).is_err())
                || self
                    .owners
                    .get(c.destination().device() as usize)
                    .is_none_or(|o| o.resolve::<RefCell<Vec<u8>>>(c.destination()).is_err())
        }) {
            return Err(Rejected {
                op: copies,
                error: Error::WrongOwner,
            });
        }
        self.next += 1;
        let t = TransferTicket {
            issuer: self.owners[0].issuer(),
            sequence: self.next,
            epochs: copies[0].epochs,
        };
        let mut entry = Entry {
            copies: vec![],
            completion: Completion {
                ticket: t,
                items: vec![],
                producer_done: false,
                consumer_fenced: false,
            },
            expected: vec![],
            cancelled: false,
            published: false,
            dma: false,
            consumer: false,
            graph: false,
            unknown: false,
        };
        let mut items = vec![];
        for (i, c) in copies.into_iter().enumerate() {
            let reject = self.reject.contains(&i);
            let n = c.source_span().bytes();
            let s = self.owners[c.source().device() as usize]
                .resolve::<RefCell<Vec<u8>>>(c.source())
                .unwrap();
            let start = c.source_span().offset() as usize;
            let hash = checksum(&s.borrow()[start..start + n as usize]);
            drop(s);
            entry.expected.push(vec![SegmentExpectation {
                valid_bytes: n,
                io_bytes: n,
                checksum: hash,
            }]);
            let fence = self.fence(&t, c.destination().device());
            entry.completion.items.push(ItemOutcome {
                item: i as u32,
                accepted: !reject,
                segments: vec![SegmentCompletion {
                    segment: 0,
                    status: if reject {
                        ItemStatus::Rejected
                    } else {
                        ItemStatus::Pending
                    },
                    valid_bytes: 0,
                    io_bytes: 0,
                    checksum: None,
                    epochs: t.epochs,
                    producer_done: false,
                    consumer_fenced: false,
                    consumer_fence: Some(fence),
                    error: None,
                }],
            });
            if reject {
                entry.copies.push(None);
                items.push(ItemAcceptance::Rejected {
                    item: i as u32,
                    op: c,
                    error: Error::Capacity,
                });
            } else {
                self.owners[c.destination().device() as usize]
                    .bind_destination(t, c.destination())
                    .unwrap();
                entry.copies.push(Some(c));
                items.push(ItemAcceptance::Accepted { item: i as u32 });
            }
        }
        self.entries.insert(t, entry);
        Ok(BatchSubmission { ticket: t, items })
    }
    fn poll(&mut self, t: &TransferTicket) -> Result<Completion> {
        let e = self.entries.get(t).ok_or(Error::UnknownTicket)?;
        if e.unknown {
            Err(Error::Quarantined)
        } else {
            Ok(e.completion.clone())
        }
    }
    fn cancel(&mut self, t: &TransferTicket) -> Result<CancelState> {
        let e = self.entries.get_mut(t).ok_or(Error::UnknownTicket)?;
        if e.published {
            Ok(CancelState::AlreadyPublished)
        } else {
            e.cancelled = true;
            Ok(CancelState::PublicationRevoked)
        }
    }
    fn materialize_local(
        &mut self,
        t: &TransferTicket,
        item: u32,
        current: Epochs,
        consumer: u32,
    ) -> Result<ReadyView<'_>> {
        let e = self.entries.get_mut(t).ok_or(Error::UnknownTicket)?;
        if e.cancelled {
            return Err(Error::Cancelled);
        }
        if e.unknown {
            return Err(Error::Quarantined);
        }
        t.epochs.require(current)?;
        let copy = e
            .copies
            .get(item as usize)
            .ok_or(Error::NotFound)?
            .as_ref()
            .ok_or(Error::Rejected)?;
        if copy.destination().device() != consumer {
            return Err(Error::WrongOwner);
        }
        let ready = self.owners[consumer as usize].ready_view(
            copy.destination(),
            &e.completion,
            &e.expected,
            current,
        )?;
        e.published = true;
        Ok(ready)
    }
    fn retire_consumer(&mut self, t: &TransferTicket, f: FenceId) -> Result<()> {
        let e = self.entries.get_mut(t).ok_or(Error::UnknownTicket)?;
        if !e.copies.iter().flatten().any(|c| {
            c.destination().device() == f.owner
                && self.owners[f.owner as usize].issuer() == f.issuer
                && f.generation == t.epochs.dst_gen
        }) {
            return Err(Error::WrongOwner);
        }
        if !e.dma || !e.consumer || !e.graph || e.unknown {
            return Err(Error::Busy);
        }
        Ok(())
    }
    fn retired(&mut self, t: &TransferTicket) -> Result<bool> {
        let e = self.entries.get(t).ok_or(Error::UnknownTicket)?;
        Ok(e.dma && e.consumer && e.graph && !e.unknown)
    }
    fn acknowledge(&mut self, t: &TransferTicket) -> Result<()> {
        if !self.retired(t)? {
            return Err(Error::Busy);
        }
        let e = self.entries.remove(t).unwrap();
        let devices: HashSet<_> = e
            .copies
            .iter()
            .flatten()
            .map(|c| c.destination().device())
            .collect();
        for d in devices {
            self.owners[d as usize].retire_binding(t)?;
        }
        Ok(())
    }
}
#[test]
fn shared_schedule_runs_on_d_byte_fake() {
    let mut p = Peer::new(shared());
    let src = p.reserve(p.plan(0, 19)).unwrap();
    let dst = p.reserve(p.plan(1, 31)).unwrap();
    let copies = vec![p.copy(&src, &dst)];
    super::conformance::peer_cancel(&mut p, copies, 1);
    let t = *p.entries.keys().next().unwrap();
    p.drain(&t);
    p.release(&src).unwrap();
    p.release(&dst).unwrap();
    assert_eq!(p.gov.borrow().used.device, vec![0, 0]);
}
#[test]
fn partial_acceptance_and_zero_accept_return_owned_inputs() {
    let mut p = Peer::new(shared());
    let src = p.reserve(p.plan(0, 19)).unwrap();
    let dst = p.reserve(p.plan(1, 31)).unwrap();
    p.reject.insert(1);
    let b = p
        .submit(vec![
            p.copy(&src, &dst),
            p.copy(&src, &dst),
            p.copy(&src, &dst),
        ])
        .unwrap();
    b.validate(3).unwrap();
    p.finish(&b.ticket);
    assert_eq!(p.poll(&b.ticket).unwrap().items.len(), 3);
    assert!(matches!(
        p.materialize_local(&b.ticket, 1, epochs(), 1),
        Err(Error::Rejected)
    ));
    assert!(matches!(
        p.materialize_local(&b.ticket, 2, epochs(), 1),
        Err(Error::Rejected)
    ));
    drop(b.items);
    p.drain(&b.ticket);
    p.reject.insert(0);
    let rejected = p.submit(vec![p.copy(&src, &dst)]).unwrap_err();
    assert_eq!(rejected.op.len(), 1);
    drop(rejected);
    assert_eq!(p.submit(vec![]).unwrap_err().error, Error::EmptyBatch);
    p.release(&src).unwrap();
    p.release(&dst).unwrap();
}
#[test]
fn exact_bytes_epochs_cancel_and_fence_issuer() {
    let mut p = Peer::new(shared());
    let src = p.reserve(p.plan(0, 19)).unwrap();
    let dst = p.reserve(p.plan(1, 31)).unwrap();
    let t = p.submit(vec![p.copy(&src, &dst)]).unwrap().ticket;
    assert!(matches!(
        p.materialize_local(&t, 0, epochs(), 1),
        Err(Error::NotReady)
    ));
    p.finish(&t);
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
        assert!(matches!(
            p.materialize_local(&t, 0, e, 1),
            Err(Error::StaleEpoch)
        ));
    }
    assert!(matches!(
        p.materialize_local(&t, 0, epochs(), 0),
        Err(Error::WrongOwner)
    ));
    p.materialize_local(&t, 0, epochs(), 1).unwrap();
    assert_eq!(
        *p.owners[1]
            .resolve::<RefCell<Vec<u8>>>(&dst.device)
            .unwrap()
            .borrow(),
        bytes(4)
    );
    assert_eq!(p.cancel(&t), Ok(CancelState::AlreadyPublished));
    let mut f = p.fence(&t, 1);
    f.issuer += 100;
    assert_eq!(p.retire_consumer(&t, f), Err(Error::WrongOwner));
    assert_eq!(p.release(&src), Err(Error::Busy));
    p.drain(&t);
    let t = p.submit(vec![p.copy(&src, &dst)]).unwrap().ticket;
    p.finish(&t);
    p.cancel(&t).unwrap();
    assert!(matches!(
        p.materialize_local(&t, 0, epochs(), 1),
        Err(Error::Cancelled)
    ));
    p.drain(&t);
    p.release(&src).unwrap();
    p.release(&dst).unwrap();
}
#[test]
fn quarantine_graph_pins_foreign_tickets_and_shared_budget() {
    let gov = shared();
    let mut p = Peer::new(gov.clone());
    let src = p.reserve(p.plan(0, 19)).unwrap();
    let dst = p.reserve(p.plan(1, 31)).unwrap();
    let t = p.submit(vec![p.copy(&src, &dst)]).unwrap().ticket;
    p.entries.get_mut(&t).unwrap().unknown = true;
    gov.borrow_mut()
        .mark(&src.charge, ChargeState::Quarantined)
        .unwrap();
    gov.borrow_mut()
        .mark(&dst.charge, ChargeState::Quarantined)
        .unwrap();
    assert_eq!(p.poll(&t), Err(Error::Quarantined));
    p.finish(&t);
    assert!(!p.retired(&t).unwrap());
    assert_eq!(p.release(&dst), Err(Error::Busy));
    assert_eq!(gov.borrow().used.device, vec![4, 4]);
    assert_eq!(gov.borrow().used.peer, vec![4, 4]);
    assert_eq!(
        p.poll(&TransferTicket {
            issuer: t.issuer + 1,
            ..t
        }),
        Err(Error::UnknownTicket)
    );
    p.entries.get_mut(&t).unwrap().unknown = false;
    p.entries.get_mut(&t).unwrap().consumer = true;
    assert!(!p.retired(&t).unwrap());
    p.entries.get_mut(&t).unwrap().graph = true;
    assert!(p.retired(&t).unwrap());
    p.acknowledge(&t).unwrap();
    assert_eq!(p.retired(&t), Err(Error::UnknownTicket));
    let mut foreign = Peer::new(gov.clone());
    assert_eq!(foreign.release(&src), Err(Error::ForeignLease));
    p.state = 8;
    gov.borrow_mut()
        .mark(&src.charge, ChargeState::Retired)
        .unwrap();
    gov.borrow_mut()
        .mark(&dst.charge, ChargeState::Retired)
        .unwrap();
    p.release(&src).unwrap();
    p.release(&dst).unwrap();
    assert!(p.release(&dst).is_err());
    assert_eq!(gov.borrow().used.device, vec![0, 0]);
}
#[test]
fn capacity_grants_padding_and_homogeneous_ticket_refusal() {
    let mut p = Peer::new(shared());
    p.route_fault((0, 1), super::conformance::RouteFault::Context, true);
    assert!(matches!(p.reserve(p.plan(0, 19)), Err(Error::Unsupported)));
    p.route_fault((0, 1), super::conformance::RouteFault::Context, false);
    let mut under = p.plan(0, 19);
    under.request.bytes.peer[0] = 0;
    assert!(matches!(p.reserve(under), Err(Error::Capacity)));
    let mut invalid = p.plan(0, 19);
    invalid.alignment = 3;
    assert!(matches!(p.reserve(invalid), Err(Error::InvalidLayout)));
    let src = p.reserve(p.plan(0, 19)).unwrap();
    let dst = p.reserve(p.plan(1, 31)).unwrap();
    let other = p.reserve(p.plan(1, 32)).unwrap();
    let rejected = p
        .submit(vec![p.copy(&src, &dst), p.copy(&src, &other)])
        .unwrap_err();
    assert_eq!(rejected.error, Error::StaleEpoch);
    assert_eq!(rejected.op.len(), 2);
    drop(rejected);
    assert!(p.entries.is_empty());
    p.release(&src).unwrap();
    p.release(&dst).unwrap();
    p.release(&other).unwrap();
}
#[test]
fn short_corrupt_and_wrong_context_completion_refuse() {
    let mut p = Peer::new(shared());
    let src = p.reserve(p.plan(0, 19)).unwrap();
    let dst = p.reserve(p.plan(1, 31)).unwrap();
    let t = p.submit(vec![p.copy(&src, &dst)]).unwrap().ticket;
    p.finish(&t);
    let original = p.entries[&t].completion.clone();
    p.entries.get_mut(&t).unwrap().completion.items[0].segments[0].io_bytes = 3;
    assert!(matches!(
        p.materialize_local(&t, 0, epochs(), 1),
        Err(Error::ShortIo { .. })
    ));
    p.entries.get_mut(&t).unwrap().completion = original.clone();
    p.entries.get_mut(&t).unwrap().completion.items[0].segments[0].checksum = None;
    assert!(matches!(
        p.materialize_local(&t, 0, epochs(), 1),
        Err(Error::Corrupt)
    ));
    p.entries.get_mut(&t).unwrap().completion = original;
    p.entries.get_mut(&t).unwrap().completion.items[0].segments[0]
        .consumer_fence
        .as_mut()
        .unwrap()
        .issuer += 100;
    assert!(matches!(
        p.materialize_local(&t, 0, epochs(), 1),
        Err(Error::WrongOwner)
    ));
    p.drain(&t);
    p.release(&src).unwrap();
    p.release(&dst).unwrap();
}

#[test]
fn caller_source_drop_keeps_owned_bytes_until_acknowledged() {
    let mut p = Peer::new(shared());
    let src = p.reserve(p.plan(0, 19)).unwrap();
    let dst = p.reserve(p.plan(1, 31)).unwrap();
    let t = p.submit(vec![p.copy(&src, &dst)]).unwrap().ticket;
    let PeerLease {
        plan,
        charge,
        device,
    } = src;
    drop(device);
    assert_eq!(p.gov.borrow_mut().release(&charge), Err(Error::Busy));
    p.finish(&t);
    assert_eq!(
        *p.owners[1]
            .resolve::<RefCell<Vec<u8>>>(&dst.device)
            .unwrap()
            .borrow(),
        bytes(4)
    );
    let device = p.owners[0]
        .retain(p.entries[&t].copies[0].as_ref().unwrap().source())
        .unwrap();
    p.drain(&t);
    p.release(&PeerLease {
        plan,
        charge,
        device,
    })
    .unwrap();
    p.release(&dst).unwrap();
}

impl super::conformance::CapacityFixture for Peer {
    fn used(&self) -> TierBudget {
        self.gov.borrow().used()
    }
    fn non_peer(&mut self) -> ChargedLease {
        let mut r = request(4, Priority::MandatoryActive);
        r.bytes.device[0] = 4;
        self.gov.borrow_mut().reserve(&r).unwrap()
    }
    fn release_non_peer(&mut self, lease: &ChargedLease) -> Result<()> {
        self.gov.borrow_mut().release(lease)
    }
    fn plan(&self, owner: u32) -> PeerPlan {
        self.plan(owner, if owner == 0 { 19 } else { 31 })
    }
    fn retain(&self, lease: &PeerLease) -> DeviceLease {
        self.owners[lease.device.device() as usize]
            .retain(&lease.device)
            .unwrap()
    }
    fn next_state(&mut self) {
        self.state += 1;
    }
}
#[test]
fn revision_v11_peer_capacity() {
    super::conformance::peer_capacity(&mut Peer::new(shared()), &mut Peer::new(shared()));
}
#[test]
fn revision_v11_peer_complete_cancel_and_quarantine() {
    let mut p = Peer::new(shared());
    let src = p.reserve(p.plan(0, 19)).unwrap();
    let dst = p.reserve(p.plan(1, 31)).unwrap();
    let copies = vec![p.copy(&src, &dst)];
    let expect = vec![vec![SegmentExpectation {
        valid_bytes: 4,
        io_bytes: 4,
        checksum: checksum(&bytes(4)),
    }]];
    let t =
        super::conformance::peer_complete_cancel(&mut p, copies, 1, &expect, |p, t| p.finish(t));
    super::conformance::peer_lifetime(
        &mut p,
        t,
        |p, t, step| {
            use super::conformance::LifetimeStep::*;
            if matches!(step, Producer) {
                p.finish(t);
                return;
            }
            let e = p.entries.get_mut(t).unwrap();
            match step {
                Unknown => e.unknown = true,
                Recover => e.unknown = false,
                Consumer => e.consumer = true,
                Graph => e.graph = true,
                Producer => unreachable!(),
            }
        },
        |p| p.gov.borrow().used(),
    );
    p.release(&src).unwrap();
    p.release(&dst).unwrap();
    assert_eq!(p.gov.borrow().used(), TierBudget::zero(2));
}
#[test]
fn revision_v11_peer_indexed_acceptance() {
    for (rejected, short) in [(2, Some(1)), (0, None), (1, None)] {
        let mut p = Peer::new(shared());
        let src = p.reserve(p.plan(0, 19)).unwrap();
        let dst = p.reserve(p.plan(1, 31)).unwrap();
        p.reject.insert(rejected);
        let b = p
            .submit((0..3).map(|_| p.copy(&src, &dst)).collect())
            .unwrap();
        p.finish(&b.ticket);
        if let Some(i) = short {
            p.entries.get_mut(&b.ticket).unwrap().completion.items[i].segments[0].io_bytes = 3;
        }
        let c = p.poll(&b.ticket).unwrap();
        super::conformance::acceptance(&b, &c, &p.entries[&b.ticket].expected, rejected, short);
        assert!(
            p.materialize_local(&b.ticket, 2, b.ticket.epochs, 1)
                .is_err()
        );
        drop(b.items);
        p.cancel(&b.ticket).unwrap();
        p.drain(&b.ticket);
        p.release(&src).unwrap();
        p.release(&dst).unwrap();
        assert_eq!(p.gov.borrow().used(), TierBudget::zero(2));
    }
}
#[test]
fn revision_v11_directed_capacity() {
    for (source, destination) in [(0, 1), (1, 0)] {
        let mut p = Peer::new(shared());
        let forward = p.plan(source, 19);
        let reverse = p.plan(destination, 31);
        super::conformance::peer_directed_grants(
            &mut p,
            forward,
            reverse,
            |p, kind, denied| p.route_fault((source, destination), kind, denied),
            |p| p.gov.borrow().used(),
        );
    }
}

#[test]
fn directed_fault_controls_are_independent_and_missing_routes_refuse() {
    use super::conformance::RouteFault::{Context, Downgrade, Pool};
    let mut p = Peer::new(shared());
    let baseline = p.gov.borrow().used();
    for kind in [Context, Pool, Downgrade] {
        p.route_fault((0, 1), kind, true);
    }
    // Restoring one control cannot silently restore another denied control.
    for kind in [Context, Pool] {
        p.route_fault((0, 1), kind, false);
        assert!(matches!(p.reserve(p.plan(0, 19)), Err(Error::Unsupported)));
        assert_eq!(p.gov.borrow().used(), baseline);
    }
    p.route_fault((0, 1), Downgrade, false);
    // Do not impose a reverse-physical-health prerequisite on forward admission.
    p.route_fault((1, 0), Downgrade, true);
    let lease = p.reserve(p.plan(0, 19)).unwrap();
    p.release(&lease).unwrap();
    for health in [LinkHealth::Unknown, LinkHealth::IdleDeferred] {
        p.routes.get_mut(&(0, 1)).unwrap().link_health = health;
        assert!(matches!(p.reserve(p.plan(0, 19)), Err(Error::Unsupported)));
        assert_eq!(p.gov.borrow().used(), baseline);
    }
    p.routes.remove(&(0, 1));
    assert!(matches!(p.reserve(p.plan(0, 19)), Err(Error::Unsupported)));
    assert_eq!(p.gov.borrow().used(), baseline);
}

#[test]
fn directed_fault_after_reservation_returns_owned_copies_without_acceptance() {
    use super::conformance::RouteFault::{Context, Downgrade, Pool};
    let mut p = Peer::new(shared());
    let src = p.reserve(p.plan(0, 19)).unwrap();
    let dst = p.reserve(p.plan(1, 31)).unwrap();
    let charged = p.gov.borrow().used();
    for kind in [Context, Pool, Downgrade] {
        p.route_fault((0, 1), kind, true);
        let rejected = p.submit(vec![p.copy(&src, &dst)]).unwrap_err();
        assert_eq!(rejected.error, Error::Unsupported);
        assert_eq!(rejected.op.len(), 1);
        assert_eq!(
            rejected.op[0].source().allocation_id(),
            src.device.allocation_id()
        );
        assert!(p.entries.is_empty());
        assert_eq!(p.gov.borrow().used(), charged);
        drop(rejected);
        p.route_fault((0, 1), kind, false);
        let t = p.submit(vec![p.copy(&src, &dst)]).unwrap().ticket;
        p.drain(&t);
    }
    p.release(&src).unwrap();
    p.release(&dst).unwrap();
    assert_eq!(p.gov.borrow().used(), TierBudget::zero(2));
}

#[test]
fn revision_v11_peer_submit_epochs_and_zero_accept() {
    let mut p = Peer::new(shared());
    let src = p.reserve(p.plan(0, 19)).unwrap();
    let dst = p.reserve(p.plan(1, 31)).unwrap();
    super::conformance::peer_submit_epochs(
        &mut p,
        |p| vec![p.copy(&src, &dst), p.copy(&src, &dst)],
        |p, t| p.drain(t),
    );
    p.reject.extend([0, 1]);
    let copies = vec![p.copy(&src, &dst), p.copy(&src, &dst)];
    let returned = super::conformance::peer_zero_accept(&mut p, copies, |ops| {
        assert!(
            ops.iter()
                .all(|o| o.source().allocation_id() == src.device.allocation_id())
        );
    });
    assert!(p.entries.is_empty());
    drop(returned);
    p.release(&src).unwrap();
    p.release(&dst).unwrap();
    assert_eq!(p.gov.borrow().used(), TierBudget::zero(2));
}
