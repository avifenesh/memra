//! Test-only hooks expose fixture controls, never alternative runtime contracts.
//! Hooks drive/observe backend state; all expected outcomes live in these schedules.
use memra_tier::contracts::*;

pub fn stale_epochs(e: Epochs) -> [Epochs; 3] {
    [
        Epochs {
            state: e.state + 1,
            ..e
        },
        Epochs {
            src_gen: e.src_gen + 1,
            ..e
        },
        Epochs {
            dst_gen: e.dst_gen + 1,
            ..e
        },
    ]
}

/// Fresh governor: all dimensions cap=100, pageable mandatory headroom=25.
/// Sequential atomicity means a refused multidimensional request charges NOTHING;
/// concurrent admission and service fairness remain native/queue integration cells.
pub fn budget_governor<G: BudgetGovernor>(g: &mut G, foreign: &mut G) {
    let zero = TierBudget::zero(2);
    let request = |n, priority| {
        let mut bytes = zero.clone();
        bytes.pageable = n;
        BudgetRequest {
            bytes,
            priority,
            deadline: Deadline(100),
            tenant: [1; 32],
        }
    };
    assert_eq!(g.used(), zero);
    let optional = g.reserve(&request(75, Priority::OptionalPrefetch)).unwrap();
    for priority in [
        Priority::AdmittedRestore,
        Priority::Demand,
        Priority::OptionalPrefetch,
        Priority::Backup,
    ] {
        let mut denied = request(1, priority);
        denied.bytes.device[0] = 1;
        let before = g.used();
        assert!(matches!(g.reserve(&denied), Err(Error::Capacity)));
        assert_eq!(g.used(), before, "refused request partially charged");
    }
    let active = g.reserve(&request(25, Priority::MandatoryActive)).unwrap();
    assert_eq!(g.used().pageable, 100);
    let mut over = request(1, Priority::MandatoryActive);
    over.bytes.pinned = 1;
    assert!(matches!(g.reserve(&over), Err(Error::Capacity)));
    assert_eq!(
        g.used(),
        optional.bytes().checked_add(active.bytes()).unwrap()
    );
    assert_eq!(foreign.release(&optional), Err(Error::ForeignLease));
    assert_eq!(foreign.used(), zero);
    let producer = active.pin().unwrap();
    let consumer = active.pin().unwrap();
    let charged = g.used();
    assert_eq!(g.release(&active), Err(Error::Busy));
    drop(producer);
    assert_eq!(g.release(&active), Err(Error::Busy));
    drop(consumer);
    g.mark(&active, ChargeState::InUse).unwrap();
    g.mark(&active, ChargeState::Quarantined).unwrap();
    assert_eq!(g.release(&active), Err(Error::Busy));
    assert_eq!(g.used(), charged);
    g.mark(&active, ChargeState::Retired).unwrap();
    g.release(&active).unwrap();
    assert_eq!(g.release(&active), Err(Error::AlreadyReleased));
    assert_eq!(g.used(), optional.bytes().clone());
    g.release(&optional).unwrap();
    assert_eq!(g.used(), zero);
    // All ceilings are independent; overlapping device/peer/replica bytes are
    // not summed into device accounting. Multiple pins never charge backing twice.
    let mut r = request(1, Priority::Demand);
    r.bytes.device = vec![10; 2];
    r.bytes.peer = vec![10; 2];
    r.bytes.replicas = vec![10; 2];
    r.bytes.pinned = 10;
    r.bytes.nvme = 10;
    r.bytes.staging = 10;
    r.bytes.loaders = 10;
    r.bytes.inflight = 1;
    let lease = g.reserve(&r).unwrap();
    assert_eq!(g.used(), r.bytes);
    let pin = lease.pin().unwrap();
    assert_eq!(g.used(), r.bytes);
    drop(pin);
    g.release(&lease).unwrap();
    assert_eq!(g.used(), zero);
    // One over-quota dimension must not charge any of the other dimensions.
    for dimension in 0..12 {
        let mut denied = request(1, Priority::MandatoryActive);
        match dimension {
            0 | 1 => denied.bytes.device[dimension] = 101,
            2 | 3 => {
                denied.bytes.device[dimension - 2] = 100;
                denied.bytes.peer[dimension - 2] = 101;
            }
            4 | 5 => {
                denied.bytes.device[dimension - 4] = 100;
                denied.bytes.replicas[dimension - 4] = 101;
            }
            6 => denied.bytes.pinned = 101,
            7 => denied.bytes.pageable = 101,
            8 => denied.bytes.staging = 101,
            9 => denied.bytes.loaders = 101,
            10 => denied.bytes.nvme = 101,
            _ => denied.bytes.inflight = 101,
        }
        assert!(g.reserve(&denied).is_err());
        assert_eq!(g.used(), zero);
    }
    assert!(Priority::MandatoryActive < Priority::AdmittedRestore);
    assert!(Priority::AdmittedRestore < Priority::Demand);
    assert!(Priority::Demand < Priority::OptionalPrefetch);
    assert!(Priority::OptionalPrefetch < Priority::Backup);
}

/// Queue hook binds B's actual dispatcher, not a sort of a test-owned vector.
pub fn governor_priority<G: BudgetGovernor>(
    g: &mut G,
    mut enqueue: impl FnMut(&mut G, BudgetRequest) -> u64,
    mut dispatch: impl FnMut(&mut G) -> (u64, ChargedLease),
) {
    let order = [
        Priority::Backup,
        Priority::Demand,
        Priority::MandatoryActive,
        Priority::OptionalPrefetch,
        Priority::AdmittedRestore,
    ];
    let ids: Vec<_> = order
        .iter()
        .map(|&priority| {
            enqueue(
                g,
                BudgetRequest {
                    bytes: TierBudget::zero(2),
                    priority,
                    deadline: Deadline(100),
                    tenant: [1; 32],
                },
            )
        })
        .collect();
    for i in [2, 4, 1, 3, 0] {
        let (id, lease) = dispatch(g);
        assert_eq!(id, ids[i]);
        g.release(&lease).unwrap();
    }
    assert_eq!(g.used(), TierBudget::zero(2));
}

/// Two slots, one reserved for demand; backing is charged once before acquisition.
pub trait PinnedFixture {
    type Host: PinnedLease;
    fn acquire(&mut self, optional: bool) -> Result<Self::Host>;
    fn initialize(&mut self, host: &mut Self::Host, bytes: &[u8]) -> Result<()>;
    fn release(&mut self, host: Self::Host);
    fn quarantine(&mut self, host: Self::Host);
    fn free_slots(&self) -> usize;
    fn used(&self) -> TierBudget;
    fn release_backing(&mut self) -> Result<()>;
    fn close_pool(&mut self);
}
pub fn pinned_lease<F: PinnedFixture>(f: &mut F) {
    let charged = f.used();
    assert_eq!(f.free_slots(), 2);
    let mut producer = f.acquire(true).unwrap();
    assert!(producer.bytes().is_err(), "uninitialized bytes exposed");
    assert!(producer.alignment().is_power_of_two());
    assert!(producer.storage_bytes() >= producer.valid_bytes());
    let bytes = vec![0x39; producer.valid_bytes() as usize];
    f.initialize(&mut producer, &bytes).unwrap();
    assert_eq!(producer.bytes().unwrap(), bytes);
    assert!(matches!(f.acquire(true), Err(Error::Capacity)));
    let second = f.acquire(false).unwrap();
    assert!(matches!(f.acquire(false), Err(Error::Capacity)));
    assert_eq!(f.used(), charged);
    assert_eq!(f.release_backing(), Err(Error::Busy));
    // Ownership moves producer -> consumer, not copy-complete -> free.
    let consumer = producer;
    f.release(second);
    assert_eq!(f.free_slots(), 1);
    assert_eq!(consumer.bytes().unwrap(), bytes);
    assert_eq!(f.release_backing(), Err(Error::Busy));
    f.release(consumer);
    assert_eq!(f.free_slots(), 2);
    assert_eq!(f.used(), charged);
    f.close_pool();
    f.release_backing().unwrap();
    assert_eq!(f.used(), TierBudget::zero(charged.device.len()));
}
pub fn pinned_quarantine<F: PinnedFixture>(f: &mut F) {
    let before = f.used();
    let host = f.acquire(false).unwrap();
    f.quarantine(host);
    assert_eq!(f.free_slots(), 1);
    let other = f.acquire(false).unwrap();
    assert!(matches!(f.acquire(false), Err(Error::Capacity)));
    assert_eq!(f.release_backing(), Err(Error::Busy));
    f.release(other);
    assert_eq!(f.free_slots(), 1);
    assert_eq!(f.used(), before);
    // Unknown shutdown is not retirement: dropping the pool must not release
    // backing that the fixture explicitly marked as potentially in flight.
    f.close_pool();
    assert_eq!(f.release_backing(), Err(Error::Busy));
    assert_eq!(f.used(), before);
}

pub fn object_publish_release<S: ObjectStore>(s: &mut S, key: ObjectKey, request: BudgetRequest) {
    let mut txn = s.begin(key.clone(), 1, Durability::Ephemeral).unwrap();
    s.put(&mut txn, &[3]).unwrap();
    let m = s.commit(&mut txn).unwrap();
    assert_eq!(s.cancel(&mut txn), Ok(CancelState::AlreadyPublished));
    assert_eq!(s.lookup(&key).unwrap(), Some(m.clone()));
    let lease = s.lease(&m, &request).unwrap();
    assert_eq!(s.read(&lease, 0, &mut [0]), Ok(1));
    assert_eq!(s.evict(&key), Err(Error::Busy));
    s.release(&lease).unwrap();
    assert!(s.release(&lease).is_err());
    assert!(s.read(&lease, 0, &mut [0]).is_err());
}

pub fn transfer_complete_cancel<T: TransferEngine>(
    t: &mut T,
    ticket: TransferTicket,
    expected: &[Vec<SegmentExpectation>],
    complete: impl FnOnce(&mut T, &TransferTicket),
) {
    complete(t, &ticket);
    t.poll(&ticket)
        .unwrap()
        .require(&ticket, expected, false)
        .unwrap();
    assert_eq!(t.cancel(&ticket), Ok(CancelState::PublicationRevoked));
    assert!(matches!(
        t.take_destination(&ticket, 0, ticket.epochs),
        Err(Error::Cancelled)
    ));
    assert!(matches!(
        t.ready_view(&ticket, 0, ticket.epochs),
        Err(Error::Cancelled)
    ));
}

pub fn bank_complete_cancel<B: BankedResidency>(
    b: &mut B,
    batch: BankBatch,
    complete: impl FnOnce(&mut B, &TransferTicket) -> Completion,
) -> TransferTicket {
    let ticket = b.stage(batch).unwrap();
    let c = complete(b, &ticket);
    assert_eq!(c.ticket, ticket);
    assert!(c.producer_done);
    assert!(!c.items.is_empty());
    assert!(c.items.iter().all(|i| !i.segments.is_empty()));
    assert!(c.items.iter().all(|i| {
        i.accepted
            && i.segments
                .iter()
                .all(|s| s.producer_done && s.status == ItemStatus::Complete)
    }));
    assert_eq!(b.cancel(&ticket), Ok(CancelState::PublicationRevoked));
    assert!(matches!(
        b.publish(&ticket, ticket.epochs),
        Err(Error::Cancelled)
    ));
    ticket
}
pub fn rows_complete_cancel<R: RowService>(
    r: &mut R,
    batch: RowBatch,
    complete: impl FnOnce(&mut R, &TransferTicket) -> Completion,
) -> TransferTicket {
    let ticket = r.gather(batch).unwrap();
    let c = complete(r, &ticket);
    assert_eq!(c.ticket, ticket);
    assert!(c.producer_done);
    assert!(!c.items.is_empty());
    assert!(c.items.iter().all(|i| !i.segments.is_empty()));
    assert!(c.items.iter().all(|i| {
        i.accepted
            && i.segments
                .iter()
                .all(|s| s.producer_done && s.status == ItemStatus::Complete)
    }));
    assert_eq!(r.cancel(&ticket), Ok(CancelState::PublicationRevoked));
    assert!(matches!(
        r.publish(&ticket, ticket.epochs),
        Err(Error::Cancelled)
    ));
    ticket
}
pub fn peer_complete_cancel<P: PeerBackend>(
    p: &mut P,
    copies: Vec<ContiguousCopy>,
    consumer: u32,
    expected: &[Vec<SegmentExpectation>],
    complete: impl FnOnce(&mut P, &TransferTicket),
) -> TransferTicket {
    let n = copies.len();
    let b = p.submit(copies).unwrap();
    b.validate(n).unwrap();
    complete(p, &b.ticket);
    p.poll(&b.ticket)
        .unwrap()
        .require(&b.ticket, expected, true)
        .unwrap();
    assert_eq!(p.cancel(&b.ticket), Ok(CancelState::PublicationRevoked));
    assert!(matches!(
        p.materialize_local(&b.ticket, 0, b.ticket.epochs, consumer),
        Err(Error::Cancelled)
    ));
    b.ticket
}

pub fn tier_identity<T: TierStore>(t: &mut T, plan: TierAdmission) -> TierReservation {
    for index in 0..10 {
        let mut bad = plan.clone();
        let p = &mut bad.program;
        let fields = [
            &mut p.artifact,
            &mut p.serialized_plan,
            &mut p.numeric,
            &mut p.stream,
            &mut p.tokenizer,
            &mut p.template,
            &mut p.adapter,
            &mut p.modality,
            &mut p.position,
            &mut p.tenant_salt,
        ];
        fields.into_iter().nth(index).unwrap()[0] ^= 1;
        assert!(t.admit(bad).is_err(), "program field {index} admitted");
    }
    let mut parent = plan.clone();
    parent.id.parent[0] ^= 1;
    assert!(t.admit(parent).is_err());
    let mut high_water = plan.clone();
    high_water.committed_high_water = plan.id.end - 1;
    assert!(t.admit(high_water).is_err());
    for i in 0..plan.expected_layout.segments.len() {
        let mut missing = plan.clone();
        missing.expected_layout.segments.remove(i);
        assert!(t.admit(missing).is_err(), "missing plane {i} admitted");
    }
    let r = t.admit(plan.clone()).unwrap();
    t.prefetch(&r).unwrap();
    assert_eq!(t.advance(&r, plan.epochs), Ok(Phase::HostReady));
    t.load(&r).unwrap();
    assert_eq!(t.advance(&r, plan.epochs), Ok(Phase::Ready));
    for e in stale_epochs(plan.epochs) {
        assert!(t.ready(&r, &plan.program, e).is_err());
    }
    for index in 0..10 {
        let mut p = plan.program.clone();
        let fields = [
            &mut p.artifact,
            &mut p.serialized_plan,
            &mut p.numeric,
            &mut p.stream,
            &mut p.tokenizer,
            &mut p.template,
            &mut p.adapter,
            &mut p.modality,
            &mut p.position,
            &mut p.tenant_salt,
        ];
        fields.into_iter().nth(index).unwrap()[0] ^= 1;
        assert!(t.ready(&r, &p, plan.epochs).is_err());
    }
    assert_eq!(t.cancel(&r), Ok(CancelState::PublicationRevoked));
    assert!(t.ready(&r, &plan.program, plan.epochs).is_err());
    // Caller supplies retirement authority; no synthetic release proof here.
    r
}

pub fn bank_identity<B: BankedResidency>(b: &mut B, batch: BankBatch) {
    for field in 0..4 {
        let mut bad = batch.clone();
        let id = &mut bad.ids[0];
        match field {
            0 => id.tensor.artifact[0] ^= 1,
            1 => id.tensor.name.push_str(".wrong"),
            2 => id.record = RecordId::Row(u64::MAX),
            _ => id.layout[0] ^= 1,
        }
        assert!(b.stage(bad).is_err());
    }
}
pub fn bank_uniform<B: BankedResidency>(
    b: &mut B,
    batch: BankBatch,
    class: LayoutClass,
    complete: impl FnOnce(&mut B, &TransferTicket),
    retire: impl FnOnce(&mut B, &TransferTicket),
) {
    let ids = batch.ids.clone();
    let t = b.stage(batch).unwrap();
    complete(b, &t);
    for e in stale_epochs(t.epochs) {
        assert!(b.publish(&t, e).is_err());
    }
    let records = b.publish(&t, t.epochs).unwrap();
    assert_eq!(
        records.iter().map(|l| l.id().clone()).collect::<Vec<_>>(),
        ids
    );
    let records = match (class, UniformLease::try_new(records)) {
        (LayoutClass::Uniform, Ok(proof)) => proof.into_records(),
        (LayoutClass::PerRecord, Err(rejected)) => {
            assert_eq!(rejected.error, Error::MixedLayout);
            assert_eq!(
                rejected
                    .op
                    .iter()
                    .map(|l| l.id().clone())
                    .collect::<Vec<_>>(),
                ids
            );
            rejected.op
        }
        _ => panic!("source class was substituted"),
    };
    assert_eq!(b.cancel(&t), Ok(CancelState::AlreadyPublished));
    assert!(b.release(&records[0]).is_err());
    retire(b, &t);
    assert!(b.retire(&t).unwrap());
    let mut seen = std::collections::HashSet::new();
    for l in &records {
        if seen.insert(l.charge().id()) {
            b.release(l).unwrap();
        }
    }
}

/// Owner hooks separate producer, consumer, graph and unknown status. Usage is
/// observed externally, not inferred from a completion boolean.
#[derive(Clone, Copy)]
pub enum LifetimeStep {
    Unknown,
    Producer,
    Recover,
    Consumer,
    Graph,
}
pub fn peer_lifetime<P: PeerBackend>(
    p: &mut P,
    t: TransferTicket,
    mut drive: impl FnMut(&mut P, &TransferTicket, LifetimeStep),
    used: impl Fn(&P) -> TierBudget,
) {
    let before = used(p);
    let foreign = TransferTicket {
        issuer: t.issuer + 1,
        ..t
    };
    assert!(p.poll(&foreign).is_err());
    drive(p, &t, LifetimeStep::Unknown);
    assert!(p.poll(&t).is_err());
    assert!(!p.retired(&t).unwrap());
    assert!(p.acknowledge(&t).is_err());
    p.cancel(&t).unwrap();
    for step in [
        LifetimeStep::Producer,
        LifetimeStep::Recover,
        LifetimeStep::Consumer,
    ] {
        drive(p, &t, step);
        assert!(!p.retired(&t).unwrap());
        assert!(p.acknowledge(&t).is_err());
        assert_eq!(used(p), before);
    }
    drive(p, &t, LifetimeStep::Graph);
    assert!(p.retired(&t).unwrap());
    p.acknowledge(&t).unwrap();
    assert_eq!(p.poll(&t), Err(Error::UnknownTicket));
    assert_eq!(p.retired(&t), Err(Error::UnknownTicket));
}

/// All reservations use the injected governor, including non-peer occupancy.
pub trait CapacityFixture: PeerCapacity {
    fn used(&self) -> TierBudget;
    fn non_peer(&mut self) -> ChargedLease;
    fn release_non_peer(&mut self, lease: &ChargedLease) -> Result<()>;
    fn plan(&self, owner: u32) -> PeerPlan;
    fn retain(&self, lease: &PeerLease) -> DeviceLease;
    fn next_state(&mut self);
}
pub fn peer_capacity<P: CapacityFixture>(p: &mut P, foreign: &mut P) {
    let zero = p.used();
    assert_eq!(zero, TierBudget::zero(2));
    let non_peer = p.non_peer();
    let baseline = p.used();
    assert_eq!(baseline, non_peer.bytes().clone());
    let plan = p.plan(0);
    let mut invalid = plan.clone();
    invalid.request.bytes.peer[0] = 0;
    assert!(p.reserve(invalid).is_err());
    let mut invalid = plan.clone();
    invalid.alignment = 3;
    assert!(p.reserve(invalid).is_err());
    let mut stale = plan.clone();
    stale.epochs.state += 1;
    assert!(p.reserve(stale).is_err());
    let mut over = plan.clone();
    over.request.bytes.device[0] = u64::MAX;
    assert!(p.reserve(over).is_err());
    assert_eq!(p.used(), baseline);
    let lease = p.reserve(plan.clone()).unwrap();
    let charged = baseline.checked_add(&plan.request.bytes).unwrap();
    assert_eq!(p.used(), charged);
    assert!(foreign.release(&lease).is_err());
    assert_eq!(foreign.used(), zero);
    let borrowed = p.retain(&lease);
    for _ in 0..2 {
        assert_eq!(p.release(&lease), Err(Error::Busy));
        assert_eq!(p.used(), charged);
    }
    drop(borrowed);
    let graph = lease.charge.pin().unwrap();
    assert_eq!(p.release(&lease), Err(Error::Busy));
    assert_eq!(p.used(), charged);
    drop(graph);
    p.next_state();
    p.release(&lease).unwrap(); // Old state can reclaim after physical retirement.
    assert!(p.release(&lease).is_err());
    assert_eq!(p.used(), baseline);
    p.release_non_peer(&non_peer).unwrap();
    assert_eq!(p.used(), zero);
}

#[derive(Clone, Copy)]
pub enum RouteFault {
    Context,
    Pool,
    Downgrade,
}
/// Deny ONE direction. A global grants boolean cannot pass this schedule.
pub fn peer_directed_grants<P: PeerCapacity>(
    p: &mut P,
    forward: PeerPlan,
    reverse: PeerPlan,
    mut fault: impl FnMut(&mut P, RouteFault, bool),
    used: impl Fn(&P) -> TierBudget,
) {
    let before = used(p);
    for kind in [RouteFault::Context, RouteFault::Pool, RouteFault::Downgrade] {
        fault(p, kind, true);
        assert!(matches!(
            p.reserve(forward.clone()),
            Err(Error::Unsupported)
        ));
        assert_eq!(used(p), before);
        if !matches!(kind, RouteFault::Downgrade) {
            let backwards = p
                .reserve(reverse.clone())
                .expect("denying one directed route must not deny its reverse");
            p.release(&backwards).unwrap();
        }
        fault(p, kind, false);
        let lease = p.reserve(forward.clone()).unwrap();
        p.release(&lease).unwrap();
        assert_eq!(used(p), before);
    }
}

/// Shared original-index/short/rejected validation, used by transport fixtures.
/// Returned rejected inputs stay owned by the caller until it drops this result.
pub fn acceptance<T>(
    b: &BatchSubmission<T>,
    c: &Completion,
    expected: &[Vec<SegmentExpectation>],
    rejected: usize,
    short: Option<usize>,
) {
    b.validate(3).unwrap();
    assert_eq!(c.ticket, b.ticket);
    assert_eq!(c.items.len(), 3);
    assert_eq!(expected.len(), 3);
    for (i, item_expected) in expected.iter().enumerate() {
        assert_eq!(c.items[i].item, i as u32);
        assert_eq!(c.items[i].accepted, i != rejected);
        match &b.items[i] {
            ItemAcceptance::Accepted { item } => {
                assert_ne!(i, rejected);
                assert_eq!(*item, i as u32);
            }
            ItemAcceptance::Rejected { item, .. } => {
                assert_eq!(i, rejected);
                assert_eq!(*item, i as u32);
            }
        }
        if i != rejected {
            for (s, e) in c.items[i].segments.iter().zip(item_expected) {
                assert_eq!(s.epochs, b.ticket.epochs);
                if Some(i) == short {
                    assert!(s.valid_bytes < e.valid_bytes);
                } else {
                    assert_eq!(s.valid_bytes, e.valid_bytes);
                    assert_eq!(s.checksum, Some(e.checksum));
                }
            }
        }
    }
    assert!(c.require(&b.ticket, expected, false).is_err());
}
pub fn transfer_lifetime<T: TransferEngine>(
    t: &mut T,
    ticket: TransferTicket,
    mut drive: impl FnMut(&mut T, &TransferTicket, LifetimeStep),
    used: impl Fn(&T) -> TierBudget,
) {
    let before = used(t);
    assert!(
        t.poll(&TransferTicket {
            issuer: ticket.issuer + 1,
            ..ticket
        })
        .is_err()
    );
    drive(t, &ticket, LifetimeStep::Unknown);
    assert!(t.poll(&ticket).is_err());
    assert!(!t.retired(&ticket).unwrap());
    assert!(t.acknowledge(&ticket).is_err());
    t.cancel(&ticket).unwrap();
    for step in [
        LifetimeStep::Producer,
        LifetimeStep::Recover,
        LifetimeStep::Consumer,
    ] {
        drive(t, &ticket, step);
        assert!(!t.retired(&ticket).unwrap());
        assert!(t.acknowledge(&ticket).is_err());
        assert_eq!(used(t), before);
    }
    drive(t, &ticket, LifetimeStep::Graph);
    assert!(t.retired(&ticket).unwrap());
    t.acknowledge(&ticket).unwrap();
    assert_eq!(t.poll(&ticket), Err(Error::UnknownTicket));
}

pub fn rows_bytes<R: RowService>(
    r: &mut R,
    batch: RowBatch,
    expected: &[Vec<Vec<u8>>],
    inspect: impl Fn(&BankLease) -> Vec<Vec<u8>>,
    complete: impl FnOnce(&mut R, &TransferTicket),
    retire: impl FnOnce(&mut R, &TransferTicket),
) -> TransferTicket {
    let ids = batch.ids.clone();
    let t = r.gather(batch).unwrap();
    complete(r, &t);
    for e in stale_epochs(t.epochs) {
        assert!(r.publish(&t, e).is_err());
    }
    let lease = r.publish(&t, t.epochs).unwrap();
    assert_eq!(
        lease
            .records
            .iter()
            .map(|l| l.id().clone())
            .collect::<Vec<_>>(),
        ids
    );
    assert_eq!(lease.records.len(), expected.len());
    for (record, expected) in lease.records.iter().zip(expected) {
        assert_eq!(inspect(record), *expected);
    }
    for (i, id) in ids.iter().enumerate() {
        for (j, other) in ids.iter().enumerate() {
            if id == other {
                assert_eq!(
                    lease.records[i].charge().id(),
                    lease.records[j].charge().id()
                );
            }
        }
    }
    assert_eq!(r.cancel(&t), Ok(CancelState::AlreadyPublished));
    assert!(r.release(&lease).is_err());
    retire(r, &t);
    assert!(r.retire(&t).unwrap());
    r.release(&lease).unwrap();
    assert!(r.release(&lease).is_err());
    t
}
/// Source-bound completeness checks: consumers must call this with their actual
/// native envelope, not with a shadow model schema. Payload and every scale/tail
/// sibling are individually corrupted/removed.
pub fn bundle_integrity<A: StateBundleAdapter>(a: &mut A, bundle: &StateBundle, epochs: Epochs) {
    for index in 0..bundle.layout.segments.len() {
        let mut missing = bundle.clone();
        missing.layout.segments.remove(index);
        missing.checksums.remove(index);
        assert!(
            a.restore_committed(&missing, &bundle.program, epochs)
                .is_err()
        );
    }
    let mut high_water = bundle.clone();
    high_water.committed_high_water = bundle.id.end - 1;
    assert!(
        a.restore_committed(&high_water, &bundle.program, epochs)
            .is_err()
    );
}

/// A test fixture supplies a complete all/trailing envelope and real padded bytes.
/// This helper also feeds native TierStore/adapter fixtures without format conversion.
pub fn bundle_planes(bundle: &StateBundle, payloads: &[Vec<u8>]) {
    bundle.verify(payloads).unwrap();
    for i in 0..payloads.len() {
        let mut corrupt = payloads.to_vec();
        corrupt[i][0] ^= 1;
        assert_eq!(bundle.verify(&corrupt), Err(Error::Corrupt));
        let mut missing = bundle.clone();
        missing.layout.segments.remove(i);
        missing.checksums.remove(i);
        assert!(missing.validate().is_err());
    }
}

/// Validate zero acceptance/returned inputs without consuming retry ownership.
pub fn transfer_zero_accept<T: TransferEngine>(
    t: &mut T,
    ops: Vec<TransferOp<T::Host>>,
    inspect: impl FnOnce(&[TransferOp<T::Host>]),
) -> Vec<TransferOp<T::Host>> {
    let n = ops.len();
    let rejected = t.submit_batch(ops).unwrap_err();
    assert_eq!(rejected.op.len(), n);
    inspect(&rejected.op);
    assert!(t.submit_batch(vec![]).is_err());
    rejected.op
}
pub fn peer_zero_accept<P: PeerBackend>(
    p: &mut P,
    ops: Vec<ContiguousCopy>,
    inspect: impl FnOnce(&[ContiguousCopy]),
) -> Vec<ContiguousCopy> {
    let n = ops.len();
    let rejected = p.submit(ops).unwrap_err();
    assert_eq!(rejected.op.len(), n);
    inspect(&rejected.op);
    assert!(p.submit(vec![]).is_err());
    rejected.op
}

/// Wrong allocation generations are checked against the *existing* allocations,
/// not by minting new matching allocations (which would be legitimate new work).
pub fn peer_submit_epochs<P: PeerBackend>(
    p: &mut P,
    mut copies: impl FnMut(&P) -> Vec<ContiguousCopy>,
    mut drain: impl FnMut(&mut P, &TransferTicket),
) {
    let originals = copies(p);
    assert_eq!(originals.len(), 2);
    let e = originals[0].epochs;
    drop(originals);
    for wrong in stale_epochs(e) {
        let mut ops = copies(p);
        for op in &mut ops {
            op.epochs = wrong;
        }
        let returned = peer_zero_accept(p, ops, |ops| {
            assert!(ops.iter().all(|op| op.epochs == wrong));
        });
        drop(returned);
    }
    for field in 0..3 {
        let mut ops = copies(p);
        ops[1].epochs = stale_epochs(e)[field];
        let returned = peer_zero_accept(p, ops, |_| {});
        drop(returned);
    }
    let ops = copies(p);
    let mut tickets = vec![];
    for op in ops {
        let b = p.submit(vec![op]).unwrap();
        b.validate(1).unwrap();
        tickets.push(b.ticket);
    }
    assert_ne!(tickets[0], tickets[1]);
    for ticket in tickets {
        p.cancel(&ticket).unwrap();
        drain(p, &ticket);
    }
}

pub fn rows_identity<R: RowService>(r: &mut R, batch: RowBatch) {
    for field in 0..4 {
        let mut bad = batch.clone();
        let id = &mut bad.ids[0];
        match field {
            0 => id.tensor.artifact[0] ^= 1,
            1 => id.tensor.name.push_str(".wrong"),
            2 => id.record = RecordId::Row(u64::MAX),
            _ => id.layout[0] ^= 1,
        }
        assert!(r.gather(bad).is_err());
    }
}
pub fn bank_failed_sibling<B: BankedResidency>(
    b: &mut B,
    batch: BankBatch,
    observe: impl FnOnce(&mut B, &TransferTicket) -> Completion,
) -> TransferTicket {
    let t = b.stage(batch).unwrap();
    let c = observe(b, &t);
    assert_eq!(c.ticket, t);
    assert!(!c.items.is_empty());
    assert!(
        c.items
            .iter()
            .flat_map(|i| &i.segments)
            .any(|s| s.error.is_some() || s.status == ItemStatus::Failed)
    );
    assert!(b.publish(&t, t.epochs).is_err());
    b.cancel(&t).unwrap();
    t
}
pub fn rows_failed_sibling<R: RowService>(
    r: &mut R,
    batch: RowBatch,
    observe: impl FnOnce(&mut R, &TransferTicket) -> Completion,
) -> TransferTicket {
    let t = r.gather(batch).unwrap();
    let c = observe(r, &t);
    assert_eq!(c.ticket, t);
    assert!(!c.items.is_empty());
    assert!(
        c.items
            .iter()
            .flat_map(|i| &i.segments)
            .any(|s| s.error.is_some() || s.status == ItemStatus::Failed)
    );
    assert!(r.publish(&t, t.epochs).is_err());
    r.cancel(&t).unwrap();
    t
}

pub fn tier_unknown<T: TierStore>(
    t: &mut T,
    plan: TierAdmission,
    mut unknown: impl FnMut(&mut T, bool),
    retired: impl FnOnce(&mut T),
    used: impl Fn(&T) -> TierBudget,
) {
    let before = used(t);
    let r = t.admit(plan.clone()).unwrap();
    t.prefetch(&r).unwrap();
    let charged = used(t);
    unknown(t, true);
    assert!(t.advance(&r, plan.epochs).is_err());
    assert!(t.ready(&r, &plan.program, plan.epochs).is_err());
    assert!(!t.retire(&r).unwrap());
    assert!(t.release(&r).is_err());
    assert_eq!(used(t), charged);
    unknown(t, false);
    t.cancel(&r).unwrap();
    assert!(!t.retire(&r).unwrap());
    assert_eq!(used(t), charged);
    retired(t);
    assert!(t.retire(&r).unwrap());
    t.release(&r).unwrap();
    assert_eq!(used(t), before);
}
/// An optional lookup miss is not permission to discard sole admitted backing.
/// The owner hook removes backing before admission, without fabricating a cache hit.
pub fn tier_miss<T: TierStore>(t: &mut T, plan: TierAdmission) {
    assert!(
        t.lookup(&plan.id, &[], plan.target_device)
            .unwrap()
            .is_none()
    );
    assert!(t.admit(plan).is_err());
}

/// Lifecycle hook for asynchronous bank adapters; synchronous C host stages have
/// no unknown driver event to inject and must not claim to execute this GPU cell.
pub fn bank_lifetime<B: BankedResidency>(
    b: &mut B,
    t: TransferTicket,
    mut drive: impl FnMut(&mut B, &TransferTicket, LifetimeStep),
    used: impl Fn(&B) -> TierBudget,
) {
    let before = used(b);
    drive(b, &t, LifetimeStep::Unknown);
    assert!(b.publish(&t, t.epochs).is_err());
    assert!(!b.retire(&t).unwrap());
    assert_eq!(b.cancel(&t), Ok(CancelState::PublicationRevoked));
    for step in [
        LifetimeStep::Producer,
        LifetimeStep::Recover,
        LifetimeStep::Consumer,
    ] {
        drive(b, &t, step);
        assert!(!b.retire(&t).unwrap());
        assert!(b.publish(&t, t.epochs).is_err());
        assert_eq!(used(b), before);
    }
    drive(b, &t, LifetimeStep::Graph);
    assert!(b.retire(&t).unwrap());
}
pub fn rows_lifetime<R: RowService>(
    r: &mut R,
    t: TransferTicket,
    mut drive: impl FnMut(&mut R, &TransferTicket, LifetimeStep),
    used: impl Fn(&R) -> TierBudget,
) {
    let before = used(r);
    drive(r, &t, LifetimeStep::Unknown);
    assert!(r.publish(&t, t.epochs).is_err());
    assert!(!r.retire(&t).unwrap());
    assert_eq!(r.cancel(&t), Ok(CancelState::PublicationRevoked));
    for step in [
        LifetimeStep::Producer,
        LifetimeStep::Recover,
        LifetimeStep::Consumer,
    ] {
        drive(r, &t, step);
        assert!(!r.retire(&t).unwrap());
        assert!(r.publish(&t, t.epochs).is_err());
        assert_eq!(used(r), before);
    }
    drive(r, &t, LifetimeStep::Graph);
    assert!(r.retire(&t).unwrap());
}
