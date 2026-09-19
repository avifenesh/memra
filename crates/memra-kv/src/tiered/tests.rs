use super::materializer::*;
use super::*;
use memra_tier::tier::{Governor, QueueOutcome};
use std::cell::RefCell;
use std::rc::Rc;

#[allow(dead_code)]
#[path = "../../../memra-tier/tests/contracts/conformance.rs"]
mod conformance;

fn program() -> ProgramIdentity {
    ProgramIdentity {
        version: 1,
        artifact: [1; 32],
        serialized_plan: [2; 32],
        numeric: [3; 32],
        stream: [4; 32],
        tokenizer: [5; 32],
        template: [6; 32],
        adapter: [7; 32],
        modality: [8; 32],
        position: [9; 32],
        tenant_salt: [10; 32],
    }
}
fn epochs() -> Epochs {
    Epochs {
        state: 7,
        src_gen: 19,
        dst_gen: 31,
    }
}
fn payloads() -> Vec<Vec<u8>> {
    [34, 24]
        .iter()
        .map(|&n| {
            let mut b = vec![0; 64];
            for (i, v) in b[..n].iter_mut().enumerate() {
                *v = (i * 7 + n) as u8;
            }
            b
        })
        .collect()
}
fn bundle() -> StateBundle {
    let p = program();
    let segments: Vec<_> = [
        (Role::Key, 34, b"q8_0".as_slice()),
        (Role::Value, 24, b"q5_1".as_slice()),
    ]
    .iter()
    .enumerate()
    .map(|(i, (role, n, name))| ByteSegment {
        version: 1,
        group: 0,
        page: 0,
        owner: 0,
        role: *role,
        tensor: None,
        offset: i as u64 * 64,
        valid_bytes: *n,
        storage_bytes: 64,
        alignment: 64,
        encoding: EncodingId {
            version: 1,
            program: digest("native-kv-encoding", name),
            row_bytes: *n,
        },
    })
    .collect();
    StateBundle {
        version: 1,
        id: KvBlockId::new(&p, [0; 32], &[42], 0, 0, 0, 7).unwrap(),
        program: p,
        layout: RecordLayout {
            version: 1,
            requirements: segments
                .iter()
                .map(|s| GroupRequirement {
                    version: 1,
                    group: s.group,
                    owner: s.owner,
                    role: s.role,
                    page_count: 1,
                    pages: PageRequirement::AllPages,
                })
                .collect(),
            segments,
        },
        kind: StateKind::Active,
        committed_high_water: 1,
        owner_aliases: vec![],
        checksums: payloads()
            .iter()
            .zip([34, 24])
            .map(|(b, n)| checksum(&b[..n]))
            .collect(),
    }
}
fn budget(n: u64) -> TierBudget {
    TierBudget {
        version: 1,
        device: vec![n; 2],
        peer: vec![n; 2],
        replicas: vec![n; 2],
        pinned: n,
        pageable: n,
        staging: n,
        loaders: n,
        nvme: n,
        inflight: n,
    }
}
fn governor(cap: u64, head: TierBudget) -> Governor {
    Governor::new(budget(cap), head, 32, cap, Arc::new(|| 1)).unwrap()
}
fn request(n: u64, priority: Priority) -> BudgetRequest {
    let mut bytes = TierBudget::zero(2);
    bytes.pageable = n;
    BudgetRequest {
        bytes,
        priority,
        deadline: Deadline(100),
        tenant: program().tenant_salt,
    }
}
fn plan(b: &StateBundle) -> TierAdmission {
    let mut request = request(0, Priority::MandatoryActive);
    request.bytes.device[0] = 128;
    request.bytes.staging = 64;
    request.bytes.inflight = 2;
    TierAdmission {
        id: b.id.clone(),
        program: b.program.clone(),
        expected_layout: b.layout.clone(),
        source: Tier::Nvme,
        target_device: 0,
        epochs: epochs(),
        committed_high_water: 1,
        request,
    }
}
type Shared = Arc<Mutex<Governor>>;
#[derive(Debug)]
struct Host {
    bytes: Vec<u8>,
    valid: u64,
    _pin: LeasePin,
}
impl PinnedLease for Host {
    fn storage_bytes(&self) -> u64 {
        self.bytes.len() as u64
    }
    fn valid_bytes(&self) -> u64 {
        self.valid
    }
    fn alignment(&self) -> u32 {
        64
    }
    fn numa_node(&self) -> Option<u32> {
        None
    }
    fn bytes(&self) -> Result<&[u8]> {
        Ok(&self.bytes[..self.valid as usize])
    }
}
struct TransferEntry {
    ops: Vec<TransferOp<Host>>,
    completion: Completion,
    cancelled: bool,
    published: bool,
    retired: bool,
}
#[derive(Default)]
struct Controls {
    reject: Option<usize>,
    zero_accept: bool,
    fault: Option<usize>,
    retired: bool,
    fenced: bool,
    pending: bool,
    unknown: bool,
}
struct Transfers {
    owner: DeviceOwner,
    peer_owner: DeviceOwner,
    next: u64,
    entries: HashMap<TransferTicket, TransferEntry>,
    control: Rc<RefCell<Controls>>,
}
impl Transfers {
    fn new(c: Rc<RefCell<Controls>>) -> Self {
        Self {
            owner: DeviceOwner::new(0),
            peer_owner: DeviceOwner::new(1),
            next: 0,
            entries: HashMap::new(),
            control: c,
        }
    }
}
impl TransferEngine for Transfers {
    type Host = Host;
    fn h2d(
        &mut self,
        o: CopyOp<Host>,
    ) -> std::result::Result<TransferTicket, Rejected<CopyOp<Host>>> {
        Err(Rejected {
            op: o,
            error: Error::Unsupported,
        })
    }
    fn d2h(
        &mut self,
        o: CopyOp<Host>,
    ) -> std::result::Result<TransferTicket, Rejected<CopyOp<Host>>> {
        Err(Rejected {
            op: o,
            error: Error::Unsupported,
        })
    }
    fn p2p(
        &mut self,
        o: ContiguousCopy,
    ) -> std::result::Result<TransferTicket, Rejected<ContiguousCopy>> {
        Err(Rejected {
            op: o,
            error: Error::Unsupported,
        })
    }
    fn nvme_read(
        &mut self,
        o: ReadPlan<Host>,
    ) -> std::result::Result<TransferTicket, Rejected<ReadPlan<Host>>> {
        Err(Rejected {
            op: o,
            error: Error::Unsupported,
        })
    }
    fn submit_batch(&mut self, ops: Vec<TransferOp<Host>>) -> Submission<TransferOp<Host>> {
        if self.control.borrow().zero_accept {
            return Err(Rejected {
                op: ops,
                error: Error::Rejected,
            });
        }
        if ops.is_empty() {
            return Err(Rejected {
                op: ops,
                error: Error::EmptyBatch,
            });
        }
        self.next += 1;
        let t = TransferTicket {
            issuer: self.owner.issuer(),
            sequence: self.next,
            epochs: epochs(),
        };
        let mut items = vec![];
        for (i, o) in ops.iter().enumerate() {
            let (valid, storage, hash, device) = match o {
                TransferOp::NvmeRead(o) => (
                    o.destination.valid,
                    o.destination.storage_bytes(),
                    checksum(o.destination.bytes().unwrap()),
                    None,
                ),
                TransferOp::H2d(o) => (
                    o.host.valid,
                    o.host.storage_bytes(),
                    checksum(o.host.bytes().unwrap()),
                    Some(&o.device),
                ),
                TransferOp::P2p(o) => {
                    let owner = if o.source().device() == 0 {
                        &self.owner
                    } else {
                        &self.peer_owner
                    };
                    let bytes = owner.resolve::<Vec<u8>>(o.source()).unwrap();
                    let valid = bundle().layout.segments[i].valid_bytes;
                    // The fake actually compares the complete retained byte copy.
                    assert_eq!(
                        *bytes,
                        *self.owner.resolve::<Vec<u8>>(o.destination()).unwrap()
                    );
                    (
                        valid,
                        o.destination_span().bytes(),
                        checksum(&bytes[..valid as usize]),
                        Some(o.destination()),
                    )
                }
                _ => panic!(),
            };
            if let Some(d) = device {
                if self.control.borrow().reject != Some(i) {
                    self.owner.bind_destination(t, d).unwrap();
                }
            }
            items.push(ItemOutcome {
                item: i as u32,
                accepted: true,
                segments: vec![SegmentCompletion {
                    segment: 0,
                    status: ItemStatus::Complete,
                    valid_bytes: valid,
                    io_bytes: storage,
                    checksum: Some(hash),
                    epochs: t.epochs,
                    producer_done: true,
                    consumer_fenced: true,
                    consumer_fence: device.map(|d| FenceId {
                        issuer: self.owner.issuer(),
                        owner: 0,
                        generation: d.generation(),
                        sequence: 1,
                    }),
                    error: None,
                }],
            });
        }
        let mut accepted = vec![];
        let mut acceptance = vec![];
        for (i, op) in ops.into_iter().enumerate() {
            if self.control.borrow().reject == Some(i) {
                items[i].accepted = false;
                acceptance.push(ItemAcceptance::Rejected {
                    item: i as u32,
                    op,
                    error: Error::Capacity,
                });
            } else {
                acceptance.push(ItemAcceptance::Accepted { item: i as u32 });
                accepted.push(op);
            }
        }
        self.entries.insert(
            t,
            TransferEntry {
                ops: accepted,
                completion: Completion {
                    ticket: t,
                    items,
                    producer_done: true,
                    consumer_fenced: true,
                },
                cancelled: false,
                published: false,
                retired: false,
            },
        );
        Ok(BatchSubmission {
            ticket: t,
            items: acceptance,
        })
    }
    fn poll(&mut self, t: &TransferTicket) -> Result<Completion> {
        let ctl = self.control.borrow();
        if ctl.unknown {
            return Err(Error::Quarantined);
        }
        let mut c = self
            .entries
            .get(t)
            .ok_or(Error::UnknownTicket)?
            .completion
            .clone();
        if !ctl.fenced {
            c.consumer_fenced = false;
        }
        if ctl.pending {
            c.items[0].segments[0].status = ItemStatus::Pending;
        }
        match ctl.fault {
            Some(0) => {
                c.items.pop();
            }
            Some(1) => c.items[0].accepted = false,
            Some(2) => c.items[0].segments[0].io_bytes -= 1,
            Some(3) => c.items[0].segments[0].epochs.src_gen += 1,
            Some(4) => c.items[1].item = 0,
            Some(5) => c.items[0].segments[0].checksum = Some([99; 32]),
            Some(6) => c.items[0].segments[0].status = ItemStatus::Failed,
            _ => (),
        }
        Ok(c)
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
    fn ready_view(
        &mut self,
        t: &TransferTicket,
        item: u32,
        current: Epochs,
    ) -> Result<ReadyView<'_>> {
        let c = self.poll(t)?;
        let e = self.entries.get_mut(t).ok_or(Error::UnknownTicket)?;
        if e.cancelled {
            return Err(Error::Cancelled);
        }
        let device = match &e.ops[item as usize] {
            TransferOp::H2d(o) => &o.device,
            TransferOp::P2p(o) => o.destination(),
            _ => return Err(Error::NotReady),
        };
        let v = self
            .owner
            .ready_view(device, &c, &expected(&bundle()), current)?;
        e.published = true;
        Ok(v)
    }
    fn take_destination(
        &mut self,
        _: &TransferTicket,
        _: u32,
        _: Epochs,
    ) -> Result<Destination<Host>> {
        Err(Error::Unsupported)
    }
    fn retire(&mut self, t: &TransferTicket, _: Option<FenceId>) -> Result<()> {
        if self.retired(t)? {
            Ok(())
        } else {
            Err(Error::Busy)
        }
    }
    fn retired(&mut self, t: &TransferTicket) -> Result<bool> {
        let e = self.entries.get_mut(t).ok_or(Error::UnknownTicket)?;
        e.retired = self.control.borrow().retired;
        Ok(e.retired && !self.control.borrow().unknown)
    }
    fn acknowledge(&mut self, t: &TransferTicket) -> Result<()> {
        if !self.retired(t)? {
            return Err(Error::Busy);
        }
        let e = self.entries.remove(t).unwrap();
        if e.ops
            .iter()
            .any(|o| matches!(o, TransferOp::H2d(_) | TransferOp::P2p(_)))
        {
            self.owner.retire_binding(t)?;
        }
        for o in e.ops {
            match o {
                TransferOp::H2d(o) => self.owner.release(&o.device)?,
                TransferOp::P2p(o) => {
                    self.owner.release(o.destination())?;
                    if o.source().device() == 0 {
                        self.owner.release(o.source())?;
                    } else {
                        self.peer_owner.release(o.source())?;
                    }
                }
                _ => (),
            }
        }
        Ok(())
    }
}
struct Backing {
    stored: StateBundle,
    others: Vec<StateBundle>,
    route: Option<Tier>,
    gov: Shared,
    available: Rc<RefCell<bool>>,
}
impl Backing {
    fn host(&self, index: usize, block: &BlockLease) -> Host {
        Host {
            bytes: payloads()[index].clone(),
            valid: block.bundle.layout.segments[index].valid_bytes,
            _pin: block.charge.pin().unwrap(),
        }
    }
}
impl KvBacking<Transfers> for Backing {
    fn lookup(&self, id: &KvBlockId) -> Result<Vec<Lookup>> {
        Ok(
            if *self.available.borrow()
                && (id == &self.stored.id || self.others.iter().any(|b| &b.id == id))
            {
                [
                    Tier::Nvme,
                    Tier::PinnedHost,
                    Tier::PeerGpu(1),
                    Tier::LocalGpu(0),
                ]
                .iter()
                .filter(|&&tier| self.route.is_none_or(|r| r == tier))
                .map(|&tier| Lookup {
                    tier,
                    storage_bytes: 128,
                })
                .collect()
            } else {
                vec![]
            },
        )
    }
    fn acquire(&mut self, p: &TierAdmission) -> Result<BlockLease> {
        if !*self.available.borrow()
            || (p.id != self.stored.id && !self.others.iter().any(|b| b.id == p.id))
        {
            return Err(Error::NotFound);
        }
        let mut r = request(128, Priority::MandatoryActive);
        if let Tier::LocalGpu(d) | Tier::PeerGpu(d) = p.source {
            r.bytes.pageable = 0;
            r.bytes.device[d as usize] = 128;
            if matches!(p.source, Tier::PeerGpu(_)) {
                r.bytes.peer[d as usize] = 128;
            }
        }
        let charge = self.gov.lock().unwrap().reserve(&r)?;
        Ok(BlockLease {
            bundle: if p.id == self.stored.id {
                self.stored.clone()
            } else {
                self.others.iter().find(|b| b.id == p.id).unwrap().clone()
            },
            tier: p.source,
            charge,
        })
    }
    fn prepare_direct(
        &mut self,
        b: &BlockLease,
        p: &TierAdmission,
        r: &TierReservation,
        t: &mut Transfers,
    ) -> Result<Vec<TransferOp<Host>>> {
        let source = match b.tier {
            Tier::LocalGpu(d) | Tier::PeerGpu(d) => d,
            _ => return Err(Error::Unsupported),
        };
        let mut ops = vec![];
        for bytes in payloads() {
            let owner = if source == 0 {
                &mut t.owner
            } else {
                &mut t.peer_owner
            };
            let src = owner.register(p.epochs.src_gen, 64, Box::new(bytes.clone()), &b.charge)?;
            let producer = FenceId {
                issuer: owner.issuer(),
                owner: source,
                generation: p.epochs.src_gen,
                sequence: 1,
            };
            let dst = t
                .owner
                .register(p.epochs.dst_gen, 64, Box::new(bytes), &r.charge)?;
            ops.push(TransferOp::P2p(ContiguousCopy::new(
                src,
                dst,
                ContiguousSpan::new(0, 64, 64)?,
                ContiguousSpan::new(0, 64, 64)?,
                p.epochs,
                producer,
            )?));
        }
        Ok(ops)
    }
    fn prepare_prefetch(
        &mut self,
        b: &BlockLease,
        p: &TierAdmission,
        _: &TierReservation,
    ) -> Result<Vec<TransferOp<Host>>> {
        Ok((0..2)
            .map(|i| {
                TransferOp::NvmeRead(ReadPlan {
                    object: ObjectKey {
                        version: 1,
                        artifact: p.program.artifact,
                        semantic_id: p.id.identity().unwrap(),
                        layout: p.expected_layout.identity().unwrap(),
                        generation: p.epochs.src_gen,
                    },
                    chunk: i as u32,
                    destination: self.host(i, b),
                    epochs: p.epochs,
                })
            })
            .collect())
    }
    fn prepare_load(
        &mut self,
        b: &BlockLease,
        p: &TierAdmission,
        r: &TierReservation,
        _: &TransferTicket,
        t: &mut Transfers,
    ) -> Result<Vec<TransferOp<Host>>> {
        let mut ops = vec![];
        for i in 0..2 {
            let host = self.host(i, b);
            let d = t.owner.register(
                p.epochs.dst_gen,
                64,
                Box::new(host.bytes.clone()),
                &r.charge,
            )?;
            ops.push(TransferOp::H2d(CopyOp {
                host,
                device: d,
                bytes: 64,
                epochs: p.epochs,
                producer_fence: None,
            }));
        }
        Ok(ops)
    }
    fn reclaim_unsubmitted(&mut self, ops: Vec<TransferOp<Host>>, transfer: &mut Transfers) {
        for op in ops {
            match op {
                TransferOp::H2d(op) => transfer.owner.release(&op.device).unwrap(),
                TransferOp::P2p(op) => {
                    transfer.owner.release(op.destination()).unwrap();
                    if op.source().device() == 0 {
                        transfer.owner.release(op.source()).unwrap();
                    } else {
                        transfer.peer_owner.release(op.source()).unwrap();
                    }
                }
                _ => (),
            }
        }
    }
    fn release(&mut self, b: &BlockLease) -> Result<()> {
        self.gov.lock().unwrap().release(&b.charge)
    }
    fn evict(&mut self, _: &KvBlockId) -> Result<()> {
        *self.available.borrow_mut() = false;
        Ok(())
    }
}
type H = Hierarchy<Backing, Transfers, Governor>;
fn fixture() -> (H, Shared, Rc<RefCell<Controls>>, Rc<RefCell<bool>>) {
    let gov = Arc::new(Mutex::new(governor(4096, TierBudget::zero(2))));
    let c = Rc::new(RefCell::new(Controls {
        fenced: true,
        ..Default::default()
    }));
    let available = Rc::new(RefCell::new(true));
    (
        Hierarchy::new(
            Backing {
                stored: bundle(),
                others: vec![],
                route: None,
                gov: gov.clone(),
                available: available.clone(),
            },
            Transfers::new(c.clone()),
            gov.clone(),
        ),
        gov,
        c,
        available,
    )
}
fn finish(h: &mut H, r: &TierReservation, c: &Rc<RefCell<Controls>>) {
    h.cancel(r).unwrap();
    c.borrow_mut().retired = true;
    assert!(h.retire(r).unwrap());
    h.release(r).unwrap();
}
#[test]
fn frozen_tier_cancel_schedule_runs_against_hierarchy() {
    let (mut h, g, c, _) = fixture();
    let b = bundle();
    let r = h.admit(plan(&b)).unwrap();
    conformance::tier_cancel(&mut h, &r, &b.program, epochs());
    assert_eq!(g.lock().unwrap().used().device[0], 128);
    finish(&mut h, &r, &c);
    assert_eq!(g.lock().unwrap().used(), TierBudget::zero(2));
}
#[test]
fn advisory_lookup_complete_identity_and_full_operand_refusal() {
    let (mut h, g, c, available) = fixture();
    let b = bundle();
    assert_eq!(
        h.lookup(&b.id, &[1], 0).unwrap().unwrap().tier,
        Tier::LocalGpu(0)
    );
    assert_eq!(
        h.lookup(&b.id, &[1], 3).unwrap().unwrap().tier,
        Tier::PeerGpu(1)
    );
    assert_eq!(
        h.lookup(&b.id, &[], 3).unwrap().unwrap().tier,
        Tier::PinnedHost
    );
    *available.borrow_mut() = false;
    assert!(matches!(h.admit(plan(&b)), Err(Error::NotFound)));
    *available.borrow_mut() = true;
    for field in 0..10 {
        let mut p = plan(&b);
        let digests = [
            &mut p.program.artifact,
            &mut p.program.serialized_plan,
            &mut p.program.numeric,
            &mut p.program.stream,
            &mut p.program.tokenizer,
            &mut p.program.template,
            &mut p.program.adapter,
            &mut p.program.modality,
            &mut p.program.position,
            &mut p.program.tenant_salt,
        ];
        digests.into_iter().nth(field).unwrap()[0] ^= 1;
        assert!(matches!(h.admit(p), Err(Error::ProgramMismatch)));
    }
    let mut p = plan(&b);
    p.request.bytes.device[0] = 127;
    assert!(matches!(h.admit(p), Err(Error::Capacity)));
    let mut p = plan(&b);
    p.expected_layout.segments[0].encoding.program = [33; 32];
    assert!(matches!(h.admit(p), Err(Error::InvalidLayout)));
    assert_eq!(g.lock().unwrap().used(), TierBudget::zero(2));
    let r = h.admit(plan(&b)).unwrap();
    finish(&mut h, &r, &c);
}
#[test]
fn disk_not_device_and_publication_cancel_linearization() {
    let (mut h, g, c, _) = fixture();
    let b = bundle();
    let r = h.admit(plan(&b)).unwrap();
    h.prefetch(&r).unwrap();
    c.borrow_mut().pending = true;
    assert_eq!(h.advance(&r, epochs()), Ok(Phase::Prefetching));
    c.borrow_mut().pending = false;
    h.advance(&r, epochs()).unwrap();
    assert!(h.ready(&r, &b.program, epochs()).is_err());
    h.load(&r).unwrap();
    c.borrow_mut().fenced = false;
    assert_eq!(h.advance(&r, epochs()), Ok(Phase::Loading));
    c.borrow_mut().fenced = true;
    assert_eq!(h.advance(&r, epochs()), Ok(Phase::Ready));
    let mut wrong = b.program.clone();
    wrong.stream[0] ^= 1;
    assert!(matches!(
        h.ready(&r, &wrong, epochs()),
        Err(Error::ProgramMismatch)
    ));
    assert_eq!(h.ready(&r, &b.program, epochs()).unwrap().tier, Tier::Nvme);
    assert_eq!(h.cancel(&r), Ok(CancelState::AlreadyPublished));
    assert_eq!(h.release(&r), Err(Error::Busy));
    assert_eq!(g.lock().unwrap().release(&r.charge), Err(Error::Busy));
    finish(&mut h, &r, &c);
}
#[test]
fn every_bad_segment_and_unknown_completion_retains_until_drain() {
    for fault in 0..8 {
        let (mut h, g, c, _) = fixture();
        let r = h.admit(plan(&bundle())).unwrap();
        h.prefetch(&r).unwrap();
        if fault == 7 {
            c.borrow_mut().unknown = true;
        } else {
            c.borrow_mut().fault = Some(fault);
        }
        assert!(h.advance(&r, epochs()).is_err());
        assert!(h.release(&r).is_err());
        assert_eq!(g.lock().unwrap().used().device[0], 128);
        c.borrow_mut().unknown = false;
        finish(&mut h, &r, &c);
        assert_eq!(g.lock().unwrap().used(), TierBudget::zero(2));
    }
}
#[test]
fn each_epoch_independently_refuses_and_rollback_seal_keeps_prefix_stable() {
    for i in 0..3 {
        let (mut h, _, c, _) = fixture();
        let r = h.admit(plan(&bundle())).unwrap();
        h.prefetch(&r).unwrap();
        let mut e = epochs();
        match i {
            0 => e.state += 1,
            1 => e.src_gen += 1,
            _ => e.dst_gen += 1,
        };
        assert_eq!(h.advance(&r, e), Err(Error::StaleEpoch));
        assert_eq!(h.phase(&r), Ok(Phase::Cancelled));
        finish(&mut h, &r, &c);
    }
    let mut a = ActiveEpoch::new(epochs(), 1);
    a.require(&bundle()).unwrap();
    let sealed = bundle().seal().unwrap();
    let next = a.fork(3).unwrap();
    assert_eq!(a.require(&bundle()), Err(Error::StaleEpoch));
    a.commit(next, 2).unwrap();
    a.rollback(1).unwrap();
    assert_eq!(a.current().src_gen, 19);
    assert_eq!(a.current().dst_gen, 31);
    assert_eq!(sealed.id.epoch, 0);
    let mut b = bundle();
    b.id.epoch = 99;
    assert_eq!(b.seal().unwrap(), sealed);
    let mut bad = bundle();
    bad.committed_high_water = 0;
    assert!(bad.seal().is_err());
}
#[test]
fn canonical_wire_group_specific_counts_padding_and_opaque_fixtures() {
    let b = bundle();
    assert_eq!(StateBundle::decode(&b.encode().unwrap()).unwrap(), b);
    b.verify(&payloads()).unwrap();
    let mut bad = payloads();
    bad[0][63] = 1;
    assert_eq!(b.verify(&bad), Err(Error::Corrupt));
    for n in [132, 288, 584] {
        let mut b = b.clone();
        b.layout.segments[1].role = Role::Tail;
        b.layout.segments[1].group = 1;
        b.layout.segments[1].page = 2;
        b.layout.segments[1].valid_bytes = n;
        b.layout.segments[1].storage_bytes = 4096;
        b.layout.segments[1].encoding = EncodingId {
            version: 1,
            program: digest("fixture", b"opaque"),
            row_bytes: n,
        };
        b.layout.requirements[1].role = Role::Tail;
        b.layout.requirements[1].group = 1;
        b.layout.requirements[1].page_count = 3;
        b.layout.requirements[1].pages = PageRequirement::TrailingPages(1);
        b.validate().unwrap();
        b.layout.segments[1].page = 1;
        assert_eq!(b.validate(), Err(Error::Incomplete));
    }
}
#[test]
fn lead_governor_headroom_foreign_double_release_pin_schedule_replayed() {
    // Exact operation/assertion schedule from frozen services.rs, using B Governor.
    // That file exports no generic governor schedule; it is NOT modified or copied as a backend.
    let mut head = TierBudget::zero(2);
    head.pageable = 25;
    let mut g = governor(100, head);
    let optional = g.reserve(&request(75, Priority::OptionalPrefetch)).unwrap();
    assert!(matches!(
        g.reserve(&request(1, Priority::Backup)),
        Err(Error::Capacity)
    ));
    let active = g.reserve(&request(25, Priority::MandatoryActive)).unwrap();
    assert_eq!(g.used().pageable, 100);
    let pin = active.pin().unwrap();
    assert_eq!(g.release(&active), Err(Error::Busy));
    drop(pin);
    g.mark(&active, ChargeState::InUse).unwrap();
    g.mark(&active, ChargeState::Quarantined).unwrap();
    assert_eq!(g.release(&active), Err(Error::Busy));
    g.mark(&active, ChargeState::Retired).unwrap();
    g.release(&active).unwrap();
    assert_eq!(g.release(&active), Err(Error::AlreadyReleased));
    let mut foreign = governor(100, TierBudget::zero(2));
    assert_eq!(foreign.release(&optional), Err(Error::ForeignLease));
    assert_eq!(foreign.used().pageable, 0);
    g.release(&optional).unwrap();
    assert_eq!(g.used().pageable, 0);
}
#[test]
fn shared_dimensions_queue_fairness_headroom_and_dirty_backpressure() {
    let mut head = TierBudget::zero(2);
    head.device[0] = 25;
    let mut g = governor(100, head);
    let mut r = request(0, Priority::OptionalPrefetch);
    r.bytes.device[0] = 75;
    let a = g.reserve(&r).unwrap();
    r.bytes.device[0] = 1;
    assert!(matches!(g.reserve(&r), Err(Error::Capacity)));
    r.priority = Priority::MandatoryActive;
    r.bytes.device[0] = 25;
    let b = g.reserve(&r).unwrap();
    g.release(&a).unwrap();
    g.release(&b).unwrap();
    let mut charges = vec![];
    for priority in [
        Priority::Backup,
        Priority::Demand,
        Priority::MandatoryActive,
        Priority::OptionalPrefetch,
        Priority::AdmittedRestore,
    ] {
        charges.push((g.enqueue(request(1, priority)).unwrap(), priority));
    }
    charges.sort_by_key(|(_, p)| *p);
    for (expected, _) in charges {
        let Some(QueueOutcome::Admitted(id, l)) = g.dispatch().unwrap() else {
            panic!()
        };
        assert_eq!(id, expected);
        g.release(&l).unwrap();
    }
    let mut g = governor(100, TierBudget::zero(2));
    let r = request(1, Priority::Demand);
    let first = g.enqueue(r.clone()).unwrap();
    let last = g.enqueue(r.clone()).unwrap();
    let mut other = r;
    other.tenant = [22; 32];
    let middle = g.enqueue(other).unwrap();
    for want in [first, middle, last] {
        let Some(QueueOutcome::Admitted(id, l)) = g.dispatch().unwrap() else {
            panic!()
        };
        assert_eq!(id, want);
        g.release(&l).unwrap();
    }
    g.set_dirty([1; 32], 100).unwrap();
    assert_eq!(g.set_dirty([2; 32], 1), Err(Error::Capacity));
    g.set_dirty([1; 32], 0).unwrap();
    g.set_dirty([2; 32], 1).unwrap();
    let mut invalid = request(0, Priority::Demand);
    invalid.bytes.peer[0] = 1;
    assert!(matches!(g.reserve(&invalid), Err(Error::InvalidLayout)));
    let mut r = request(0, Priority::Demand);
    r.bytes = budget(100);
    let l = g.reserve(&r).unwrap();
    assert_eq!(g.used(), budget(100));
    g.release(&l).unwrap();
}
#[test]
fn native_qwen_materializer_exact_bytes_identity_and_stable_address() {
    let mut owner = DeviceOwner::new(0);
    let mut g = governor(4096, TierBudget::zero(2));
    let p = plan(&bundle());
    let charge = g.reserve(&p.request).unwrap();
    let b = bundle();
    let device = owner
        .register(
            31,
            128,
            Box::new(NativeKvImage {
                bundle: b.clone(),
                payloads: payloads(),
            }),
            &charge,
        )
        .unwrap();
    let ticket = TransferTicket {
        issuer: owner.issuer(),
        sequence: 1,
        epochs: epochs(),
    };
    owner.bind_destination(ticket, &device).unwrap();
    let mut completion = Completion {
        ticket,
        items: vec![],
        producer_done: true,
        consumer_fenced: true,
    };
    for (i, s) in b.layout.segments.iter().enumerate() {
        completion.items.push(ItemOutcome {
            item: i as u32,
            accepted: true,
            segments: vec![SegmentCompletion {
                segment: 0,
                status: ItemStatus::Complete,
                valid_bytes: s.valid_bytes,
                io_bytes: s.storage_bytes,
                checksum: Some(b.checksums[i]),
                epochs: epochs(),
                producer_done: true,
                consumer_fenced: true,
                consumer_fence: Some(FenceId {
                    issuer: owner.issuer(),
                    owner: 0,
                    generation: 31,
                    sequence: 1,
                }),
                error: None,
            }],
        });
    }
    let view = owner
        .ready_view(&device, &completion, &expected(&b), epochs())
        .unwrap();
    let geometry = NativeGeometry {
        group: 0,
        owner: 0,
        tokens: 1,
        k_token_bytes: 34,
        v_token_bytes: 24,
    };
    let mut materializer = QwenMaterializer::new(&owner, b.program.clone(), geometry);
    for i in 0..10 {
        let mut wrong = b.program.clone();
        let fields = [
            &mut wrong.artifact,
            &mut wrong.serialized_plan,
            &mut wrong.numeric,
            &mut wrong.stream,
            &mut wrong.tokenizer,
            &mut wrong.template,
            &mut wrong.adapter,
            &mut wrong.modality,
            &mut wrong.position,
            &mut wrong.tenant_salt,
        ];
        fields.into_iter().nth(i).unwrap()[0] ^= 1;
        assert!(matches!(
            materializer.materialize(&b, &wrong, &view, epochs()),
            Err(Error::ProgramMismatch)
        ));
    }
    let op = materializer
        .materialize(&b, &b.program, &view, epochs())
        .unwrap();
    assert_eq!(op.allocation_id(), device.allocation_id());
    let (k, v) = materializer.capture(&op).unwrap();
    assert_eq!(k, payloads()[0][..34]);
    assert_eq!(v, payloads()[1][..24]);
    assert_eq!(g.release(&charge), Err(Error::Busy));
    materializer
        .retire(
            &op,
            FenceId {
                issuer: owner.issuer(),
                owner: 0,
                generation: 31,
                sequence: 2,
            },
        )
        .unwrap();
    assert!(materializer.capture(&op).is_err());
    drop(op);
    drop(materializer);
    owner.retire_binding(&ticket).unwrap();
    owner.release(&device).unwrap();
    g.release(&charge).unwrap();
}
#[test]
fn native_format_substitution_partial_history_and_ring_layout_refuse() {
    let geo = NativeGeometry {
        group: 0,
        owner: 0,
        tokens: 1,
        k_token_bytes: 34,
        v_token_bytes: 24,
    };
    for name in [b"f16".as_slice(), b"fp8", b"q4_0"] {
        let mut b = bundle();
        b.layout.segments[0].encoding.program = digest("native-kv-encoding", name);
        assert!(geo.validate(&b).is_err());
    }
    let mut b = bundle();
    b.layout.requirements[0].pages = PageRequirement::TrailingPages(1);
    assert_eq!(geo.validate(&b), Err(Error::Unsupported));
}
#[test]
fn policy_frontier_fixture_table_and_mandatory_refusal() {
    use policy::*;
    for (tokens, bytes, gb_s, tps, fixed, mandatory, proof, want) in [
        (
            1000,
            1_000_000_000,
            1.0,
            1000.0,
            0.0,
            false,
            true,
            RestoreDecision::Recompute,
        ),
        (
            2000,
            1_000_000_000,
            1.0,
            1000.0,
            0.0,
            false,
            true,
            RestoreDecision::Load,
        ),
        (
            2000,
            1_000_000_000,
            1.0,
            1000.0,
            2.0,
            false,
            true,
            RestoreDecision::Recompute,
        ),
        (
            0,
            0,
            0.0,
            0.0,
            0.0,
            true,
            false,
            RestoreDecision::RequireState,
        ),
        (0, 0, 0.0, 0.0, 0.0, false, false, RestoreDecision::Load),
    ] {
        assert_eq!(
            recompute_vs_load(tokens, bytes, gb_s, tps, fixed, mandatory, proof).unwrap(),
            want
        );
    }
    for speed in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert!(recompute_vs_load(1, 1, speed, 1.0, 0.0, false, true).is_err());
    }
    assert_eq!(
        prefetch_decision(
            PrefetchPolicy::Timeout(Deadline(100)),
            false,
            true,
            100,
            true,
            true
        ),
        PrefetchDecision::Refuse
    );
    assert_eq!(
        prefetch_decision(
            PrefetchPolicy::Timeout(Deadline(100)),
            false,
            true,
            99,
            false,
            true
        ),
        PrefetchDecision::Wait
    );
}

#[test]
fn hostprefix_identity_lease_is_unbound_by_default_and_revoked_on_remove_or_purge() {
    use super::hostprefix::IdentitySlot;
    let mut slot = IdentitySlot::default();
    let generation = Arc::new(());
    let b = bundle().seal().unwrap();
    assert!(matches!(
        slot.lease(&b.program, &generation),
        Err(Error::Unsupported)
    ));
    assert!(
        slot.bind(b.clone(), &[43], &b.program, &b.layout, generation.clone())
            .is_err()
    );
    slot.bind(b.clone(), &[42], &b.program, &b.layout, generation.clone())
        .unwrap();
    let lease = slot.lease(&b.program, &generation).unwrap();
    assert_eq!(lease.require(&b.program, &generation).unwrap(), &b);
    assert!(matches!(
        slot.lease(&b.program, &Arc::new(())),
        Err(Error::StaleEpoch)
    ));
    let mut wrong = b.program.clone();
    wrong.tenant_salt[0] ^= 1;
    assert!(matches!(
        slot.lease(&wrong, &generation),
        Err(Error::ProgramMismatch)
    ));
    // HostPrefixCache::remove_at/replacement/tenant-purge drop the same entry sidecar.
    drop(slot);
    assert!(matches!(
        lease.require(&b.program, &generation),
        Err(Error::StaleEpoch)
    ));
}
#[test]
fn hostprefix_rebinding_revokes_old_lease_and_does_not_migrate_handoff_implicitly() {
    use super::hostprefix::IdentitySlot;
    let mut slot = IdentitySlot::default();
    let generation = Arc::new(());
    let b = bundle().seal().unwrap();
    slot.bind(b.clone(), &[42], &b.program, &b.layout, generation.clone())
        .unwrap();
    let old = slot.lease(&b.program, &generation).unwrap();
    slot.bind(b.clone(), &[42], &b.program, &b.layout, generation.clone())
        .unwrap();
    assert!(matches!(
        old.require(&b.program, &generation),
        Err(Error::StaleEpoch)
    ));
    assert!(slot.lease(&b.program, &generation).is_ok());
    let imported = IdentitySlot::default();
    assert!(matches!(
        imported.lease(&b.program, &generation),
        Err(Error::Unsupported)
    ));
}
#[test]
fn queue_bounds_deadline_expiry_no_bypass_and_atomic_concurrent_reservations() {
    use std::sync::atomic::{AtomicU64, Ordering};
    let clock = Arc::new(AtomicU64::new(1));
    let c = clock.clone();
    let mut g = Governor::new(
        budget(100),
        TierBudget::zero(2),
        1,
        100,
        Arc::new(move || c.load(Ordering::SeqCst)),
    )
    .unwrap();
    let id = g.enqueue(request(1, Priority::Demand)).unwrap();
    assert!(matches!(
        g.enqueue(request(1, Priority::Demand)),
        Err(Error::Capacity)
    ));
    assert!(matches!(
        g.reserve(&request(1, Priority::Demand)),
        Err(Error::Busy)
    ));
    clock.store(100, Ordering::SeqCst);
    assert!(matches!(g.dispatch().unwrap(),Some(QueueOutcome::Expired(i)) if i==id));
    let g = Arc::new(Mutex::new(governor(100, TierBudget::zero(2))));
    let barrier = Arc::new(std::sync::Barrier::new(8));
    let threads: Vec<_> = (0..8)
        .map(|_| {
            let g = g.clone();
            let barrier = barrier.clone();
            std::thread::spawn(move || {
                barrier.wait();
                g.lock().unwrap().reserve(&request(25, Priority::Demand))
            })
        })
        .collect();
    let results: Vec<_> = threads.into_iter().map(|j| j.join().unwrap()).collect();
    assert_eq!(results.iter().filter(|r| r.is_ok()).count(), 4);
    assert_eq!(g.lock().unwrap().used().pageable, 100);
    for charge in results.into_iter().flatten() {
        g.lock().unwrap().release(&charge).unwrap();
    }
}
#[test]
fn write_and_eviction_primitives_remain_byte_policy_only() {
    use policy::*;
    assert_eq!(
        write_action(WritePolicy::Through, true, false, 0, false),
        WriteAction::BackupBeforePublish
    );
    assert_eq!(
        write_action(WritePolicy::Back, true, false, 0, true),
        WriteAction::BackupBeforeEvict
    );
    assert_eq!(
        write_action(
            WritePolicy::Selective { reuse_threshold: 2 },
            true,
            false,
            1,
            false
        ),
        WriteAction::RetainDirty
    );
    let base = EvictionCandidate {
        id: 1,
        last_use: 1,
        protected: true,
        leased: false,
        inflight: false,
        dirty: false,
        has_backing: true,
        mandatory: true,
    };
    let newer = EvictionCandidate {
        id: 2,
        last_use: 2,
        protected: false,
        ..base
    };
    assert_eq!(
        eviction_victim(&[base, newer], EvictionPolicy::Lru),
        Some(1)
    );
    assert_eq!(
        eviction_victim(&[base, newer], EvictionPolicy::Slru),
        Some(2)
    );
    for c in [
        EvictionCandidate {
            leased: true,
            ..base
        },
        EvictionCandidate {
            inflight: true,
            ..base
        },
        EvictionCandidate {
            dirty: true,
            ..base
        },
        EvictionCandidate {
            has_backing: false,
            ..base
        },
    ] {
        assert_eq!(eviction_victim(&[c], EvictionPolicy::Lru), None);
    }
}

#[test]
fn calibrated_prefix_costs_include_suffix_and_materialization_and_never_discard_admitted_state() {
    use policy::*;
    let base = PrefixCosts {
        prompt_tokens: 1000,
        reused_tokens: 800,
        cold_prefill_ns: 1000,
        suffix_prefill_ns: 200,
        read_ns: 300,
        copy_ns: 100,
        materialize_ns: 100,
    };
    for (cost, want) in [
        (base, RestoreDecision::Load),
        (
            PrefixCosts {
                materialize_ns: 400,
                ..base
            },
            RestoreDecision::Recompute,
        ),
        (
            PrefixCosts {
                suffix_prefill_ns: 1100,
                ..base
            },
            RestoreDecision::Recompute,
        ),
    ] {
        assert_eq!(
            calibrated_recompute_vs_load(cost, false, false, true).unwrap(),
            want
        );
    }
    assert_eq!(
        calibrated_recompute_vs_load(base, true, true, false),
        Ok(RestoreDecision::RequireState)
    );
    assert_eq!(
        calibrated_recompute_vs_load(base, false, true, true),
        Err(Error::Busy)
    );
    assert_eq!(
        calibrated_recompute_vs_load(base, false, false, false),
        Ok(RestoreDecision::Load)
    );
    assert_eq!(
        calibrated_recompute_vs_load(
            PrefixCosts {
                read_ns: u64::MAX,
                ..base
            },
            false,
            false,
            true
        ),
        Err(Error::Overflow)
    );
}

#[test]
fn native_multi_page_odd_tail_is_not_reordered_and_full_history_is_required() {
    let mut b = bundle();
    b.id = KvBlockId::new(&b.program, [0; 32], &[1, 2, 3], 0, 0, 0, 7).unwrap();
    b.committed_high_water = 3;
    let templates = b.layout.segments.clone();
    b.layout.segments.clear();
    b.checksums.clear();
    let mut payloads = vec![];
    // Interleaved physical records; each role still follows exact token order.
    for (page, tokens) in [(0, 2), (1, 1)] {
        for template in &templates {
            let mut s = template.clone();
            s.page = page;
            s.valid_bytes = tokens * s.encoding.row_bytes;
            s.storage_bytes = s.valid_bytes.div_ceil(64) * 64;
            s.offset = 0;
            let mut bytes = vec![0; s.storage_bytes as usize];
            for (i, v) in bytes[..s.valid_bytes as usize].iter_mut().enumerate() {
                *v = (page as usize * 113 + i + 7) as u8;
            }
            b.checksums.push(checksum(&bytes[..s.valid_bytes as usize]));
            payloads.push(bytes);
            b.layout.segments.push(s);
        }
    }
    for r in &mut b.layout.requirements {
        r.page_count = 2;
    }
    let geo = NativeGeometry {
        group: 0,
        owner: 0,
        tokens: 3,
        k_token_bytes: 34,
        v_token_bytes: 24,
    };
    geo.validate(&b).unwrap();
    b.verify(&payloads).unwrap();
    let mut g = governor(4096, TierBudget::zero(2));
    let mut r = request(0, Priority::MandatoryActive);
    r.bytes.device[0] = b.layout.storage_bytes().unwrap();
    let charge = g.reserve(&r).unwrap();
    let mut owner = DeviceOwner::new(0);
    let device = owner
        .register(
            31,
            r.bytes.device[0],
            Box::new(NativeKvImage {
                bundle: b.clone(),
                payloads: payloads.clone(),
            }),
            &charge,
        )
        .unwrap();
    let ticket = TransferTicket {
        issuer: owner.issuer(),
        sequence: 1,
        epochs: epochs(),
    };
    owner.bind_destination(ticket, &device).unwrap();
    let c = Completion {
        ticket,
        producer_done: true,
        consumer_fenced: true,
        items: b
            .layout
            .segments
            .iter()
            .enumerate()
            .map(|(i, s)| ItemOutcome {
                item: i as u32,
                accepted: true,
                segments: vec![SegmentCompletion {
                    segment: 0,
                    status: ItemStatus::Complete,
                    valid_bytes: s.valid_bytes,
                    io_bytes: s.storage_bytes,
                    checksum: Some(b.checksums[i]),
                    epochs: epochs(),
                    producer_done: true,
                    consumer_fenced: true,
                    consumer_fence: Some(FenceId {
                        issuer: owner.issuer(),
                        owner: 0,
                        generation: 31,
                        sequence: 1,
                    }),
                    error: None,
                }],
            })
            .collect(),
    };
    let view = owner
        .ready_view(&device, &c, &expected(&b), epochs())
        .unwrap();
    let mut m = QwenMaterializer::new(&owner, b.program.clone(), geo.clone());
    let op = m.materialize(&b, &b.program, &view, epochs()).unwrap();
    let (k, v) = m.capture(&op).unwrap();
    assert_eq!(
        k,
        [payloads[0][..68].to_vec(), payloads[2][..34].to_vec()].concat()
    );
    assert_eq!(
        v,
        [payloads[1][..48].to_vec(), payloads[3][..24].to_vec()].concat()
    );
    let mut foreign = QwenMaterializer::new(&owner, b.program.clone(), geo.clone());
    let foreign_op = foreign
        .materialize(&b, &b.program, &view, epochs())
        .unwrap();
    let done = FenceId {
        issuer: owner.issuer(),
        owner: 0,
        generation: 31,
        sequence: 2,
    };
    assert_eq!(m.retire(&foreign_op, done), Err(Error::ForeignLease));
    foreign.retire(&foreign_op, done).unwrap();
    m.retire(&op, done).unwrap();
    drop(op);
    drop(foreign_op);
    drop(m);
    drop(foreign);
    owner.retire_binding(&ticket).unwrap();
    owner.release(&device).unwrap();
    g.release(&charge).unwrap();
    b.layout.segments.swap(0, 2);
    assert_eq!(geo.validate(&b), Err(Error::InvalidLayout));
}

#[test]
fn zero_accept_and_partial_submit_return_unaccepted_allocation_ownership() {
    for partial in [false, true] {
        let (mut h, g, c, _) = fixture();
        let r = h.admit(plan(&bundle())).unwrap();
        h.prefetch(&r).unwrap();
        h.advance(&r, epochs()).unwrap();
        if partial {
            c.borrow_mut().reject = Some(1);
        } else {
            c.borrow_mut().zero_accept = true;
        }
        assert_eq!(h.load(&r), Err(Error::Rejected));
        // The accepted subset and source remain owned until drain, not silently published.
        assert!(h.ready(&r, &program(), epochs()).is_err());
        finish(&mut h, &r, &c);
        assert_eq!(g.lock().unwrap().used(), TierBudget::zero(2));
    }
}
#[test]
fn unknown_shutdown_retains_source_and_reservation_pins() {
    let (mut h, g, _, _) = fixture();
    let r = h.admit(plan(&bundle())).unwrap();
    h.prefetch(&r).unwrap();
    drop(h);
    assert_eq!(g.lock().unwrap().release(&r.charge), Err(Error::Busy));
    assert_eq!(g.lock().unwrap().used().device[0], 128);
}

#[test]
fn revision_v11_generic_governor() {
    let mut head = TierBudget::zero(2);
    head.pageable = 25;
    conformance::budget_governor(&mut governor(100, head.clone()), &mut governor(100, head));
}
#[test]
fn revision_v11_governor_actual_dispatch_priority() {
    conformance::governor_priority(
        &mut governor(100, TierBudget::zero(2)),
        |g, r| g.enqueue(r).unwrap(),
        |g| match g.dispatch().unwrap().unwrap() {
            QueueOutcome::Admitted(id, lease) => (id, lease),
            QueueOutcome::Expired(_) => panic!("fixture deadline not expired"),
        },
    );
}
#[test]
fn revision_v11_tier_complete_cancel_identity_and_planes() {
    let (mut h, g, c, _) = fixture();
    let b = bundle();
    let r = conformance::tier_identity(&mut h, plan(&b));
    // Existing hierarchy keeps the cancelled reservation; drain using its owner hook.
    c.borrow_mut().retired = true;
    assert_eq!(g.lock().unwrap().used().device[0], 128);
    finish(&mut h, &r, &c);
    assert_eq!(g.lock().unwrap().used(), TierBudget::zero(2));
    conformance::bundle_planes(&b, &payloads());
}

#[test]
fn revision_v11_tier_unknown_and_missing_backing() {
    let (mut h, g, c, available) = fixture();
    conformance::tier_unknown(
        &mut h,
        plan(&bundle()),
        |_, unknown| c.borrow_mut().unknown = unknown,
        |_| c.borrow_mut().retired = true,
        |_| g.lock().unwrap().used(),
    );
    *available.borrow_mut() = false;
    conformance::tier_miss(&mut h, plan(&bundle()));
    assert_eq!(g.lock().unwrap().used(), TierBudget::zero(2));
}

#[test]
fn direct_local_and_peer_paths_skip_host_stage_but_not_fences() {
    for source in [Tier::LocalGpu(0), Tier::PeerGpu(1)] {
        let (mut h, g, c, _) = fixture();
        let mut p = plan(&bundle());
        p.source = source;
        let r = h.admit(p).unwrap();
        let ticket = h.prefetch(&r).unwrap();
        assert_eq!(h.phase(&r), Ok(Phase::Loading));
        assert_eq!(h.transfer.as_ref().unwrap().entries.len(), 1);
        assert!(
            h.transfer.as_ref().unwrap().entries[&ticket]
                .ops
                .iter()
                .all(|op| matches!(op, TransferOp::P2p(_)))
        );
        assert_eq!(h.load(&r), Err(Error::NotReady));
        c.borrow_mut().fenced = false;
        assert_eq!(h.advance(&r, epochs()), Ok(Phase::Loading));
        assert!(h.ready(&r, &program(), epochs()).is_err());
        c.borrow_mut().fenced = true;
        assert_eq!(h.advance(&r, epochs()), Ok(Phase::Ready));
        assert_eq!(h.ready(&r, &program(), epochs()).unwrap().tier, source);
        finish(&mut h, &r, &c);
        assert_eq!(g.lock().unwrap().used(), TierBudget::zero(2));
    }
}

#[test]
fn owned_prefix_seal_copies_committed_bytes_not_an_active_alias() {
    use hostprefix::SealedImage;
    let mut active = ActiveEpoch::new(epochs(), 0);
    let mut b = bundle();
    let next = active.fork(1).unwrap();
    b.id.epoch = next.state;
    assert!(SealedImage::copy_committed(&active, &b, &payloads()).is_err());
    assert_eq!(active.commit(next, 2), Err(Error::Incomplete));
    active.commit(next, 1).unwrap();
    let mut bytes = payloads();
    let sealed = SealedImage::copy_committed(&active, &b, &bytes).unwrap();
    assert_ne!(sealed.payloads()[0].as_ptr(), bytes[0].as_ptr());
    bytes[0][0] ^= 255;
    active.rollback(0).unwrap();
    assert!(active.require(&b).is_err());
    sealed.bundle().verify(sealed.payloads()).unwrap();
    assert_eq!(sealed.bundle().id.epoch, 0);
    assert_eq!(sealed.bundle().committed_high_water, 1);
    assert!(b.verify(&bytes).is_err());
}

#[test]
fn fake_scheduler_concurrent_churn_cancel_rollback_and_quota_no_double_charge() {
    use scheduler::*;
    let (mut h, g, c, _) = fixture();
    h.backing.as_mut().unwrap().route = Some(Tier::Nvme);
    let mut s = Scheduler::new(h, 3);
    let a = s.enqueue(plan(&bundle()), PrefetchPolicy::Wait).unwrap();
    let b = s.enqueue(plan(&bundle()), PrefetchPolicy::Wait).unwrap();
    let d = s.enqueue(plan(&bundle()), PrefetchPolicy::Wait).unwrap();
    assert_eq!(
        s.enqueue(plan(&bundle()), PrefetchPolicy::Wait),
        Err(Error::Capacity)
    );
    c.borrow_mut().pending = true;
    s.tick(1, &[], |_| epochs()).unwrap();
    assert_eq!(g.lock().unwrap().used().device[0], 128);
    s.tick(2, &[], |_| epochs()).unwrap();
    assert_eq!(g.lock().unwrap().used().device[0], 256);
    assert!(s.consume(a, epochs(), |_| ()).is_err());
    s.cancel(a).unwrap();
    assert_eq!(s.release(a), Err(Error::Busy));
    s.cancel(d).unwrap(); // uncharged queued cancel
    assert_eq!(g.lock().unwrap().used().device[0], 256);
    let changed = Epochs {
        state: epochs().state + 1,
        ..epochs()
    };
    let events = s.tick(3, &[], |_| changed).unwrap();
    assert!(
        events
            .iter()
            .any(|(id, result)| *id == b && *result == Err(Error::StaleEpoch))
    );
    c.borrow_mut().pending = false; // late success cannot resurrect either request
    assert!(s.consume(a, epochs(), |_| ()).is_err());
    assert!(s.consume(b, epochs(), |_| ()).is_err());
    c.borrow_mut().retired = true;
    s.release(a).unwrap();
    s.release(b).unwrap();
    assert_eq!(g.lock().unwrap().used(), TierBudget::zero(2));
    // Repeated ticks do not re-reserve. Pressure holds the head uncharged.
    let pressure = g
        .lock()
        .unwrap()
        .reserve(&BudgetRequest {
            bytes: budget(4096),
            ..request(0, Priority::MandatoryActive)
        })
        .unwrap();
    let id = s.enqueue(plan(&bundle()), PrefetchPolicy::Wait).unwrap();
    for _ in 0..10 {
        assert!(
            s.tick(4, &[], |_| epochs())
                .unwrap()
                .contains(&(id, Ok(Progress::Queued)))
        );
    }
    assert_eq!(g.lock().unwrap().used(), budget(4096));
    g.lock().unwrap().release(&pressure).unwrap();
    s.tick(5, &[], |_| epochs()).unwrap();
    s.tick(6, &[], |_| epochs()).unwrap();
    c.borrow_mut().fenced = false;
    s.tick(7, &[], |_| epochs()).unwrap();
    assert!(s.consume(id, epochs(), |_| ()).is_err());
    c.borrow_mut().fenced = true;
    s.tick(8, &[], |_| epochs()).unwrap();
    assert_eq!(s.consume(id, epochs(), |b| b.bundle.id.end), Ok(1));
    assert_eq!(s.cancel(id), Ok(CancelState::AlreadyPublished));
    s.release(id).unwrap();
    assert_eq!(g.lock().unwrap().used(), TierBudget::zero(2));
}

#[test]
fn scheduler_priority_deadline_fifo_bounded_fairness_and_timeout() {
    use scheduler::*;
    let (mut h, g, c, _) = fixture();
    h.backing.as_mut().unwrap().route = Some(Tier::Nvme);
    let mut s = Scheduler::new(h, 8);
    let mut ids = vec![];
    for priority in [
        Priority::Backup,
        Priority::Demand,
        Priority::MandatoryActive,
        Priority::OptionalPrefetch,
        Priority::AdmittedRestore,
    ] {
        let mut p = plan(&bundle());
        p.request.priority = priority;
        ids.push((priority, s.enqueue(p, PrefetchPolicy::Wait).unwrap()));
    }
    ids.sort();
    for (_, id) in ids {
        assert!(
            s.tick(1, &[], |_| epochs())
                .unwrap()
                .contains(&(id, Ok(Progress::Phase(Phase::Reserved))))
        );
        s.cancel(id).unwrap();
        c.borrow_mut().retired = true;
        s.release(id).unwrap();
    }
    let id = s
        .enqueue(plan(&bundle()), PrefetchPolicy::Timeout(Deadline(3)))
        .unwrap();
    s.tick(1, &[], |_| epochs()).unwrap();
    assert!(
        s.tick(3, &[], |_| epochs())
            .unwrap()
            .contains(&(id, Err(Error::Deadline)))
    );
    assert!(s.consume(id, epochs(), |_| ()).is_err());
    s.release(id).unwrap();
    let id = s.enqueue(plan(&bundle()), PrefetchPolicy::Wait).unwrap();
    assert!(
        s.tick(100, &[], |_| epochs())
            .unwrap()
            .contains(&(id, Err(Error::Deadline)))
    );
    assert_eq!(g.lock().unwrap().used(), TierBudget::zero(2));
}

#[test]
fn synchronous_residency_guard_charges_once_and_keeps_host_twin_on_promotion() {
    use hostprefix::ResidentCharge;
    let g = Arc::new(Mutex::new(governor(1024, TierBudget::zero(2))));
    let source = ResidentCharge::reserve(g.clone(), &request(128, Priority::Backup)).unwrap();
    assert_eq!(g.lock().unwrap().used().pageable, 128);
    let mut r = request(0, Priority::AdmittedRestore);
    r.bytes.device[0] = 128;
    let target = ResidentCharge::reserve(g.clone(), &r).unwrap();
    let pin = source.lease().pin().unwrap();
    assert_eq!(g.lock().unwrap().used().device[0], 128);
    assert_eq!(g.lock().unwrap().used().pageable, 128);
    drop(pin);
    drop(target);
    assert_eq!(g.lock().unwrap().used().device[0], 0);
    assert_eq!(g.lock().unwrap().used().pageable, 128);
    drop(source);
    assert_eq!(g.lock().unwrap().used(), TierBudget::zero(2));
}

#[test]
fn recompute_load_cross_product_fixture_is_not_a_runtime_default() {
    use policy::*;
    let csv = include_str!("../../../../research/spill-b-20260919/fixtures/recompute-load.csv");
    let mut count = 0;
    let mut load = 0;
    for row in csv.lines().skip(1) {
        let cells: Vec<_> = row.split(',').collect();
        let tokens: u64 = cells[0].parse().unwrap();
        let bpt: u64 = cells[1].parse().unwrap();
        let decision = recompute_vs_load(
            tokens,
            tokens * bpt,
            cells[2].parse().unwrap(),
            cells[3].parse().unwrap(),
            cells[4].parse().unwrap(),
            false,
            true,
        )
        .unwrap();
        let expected = match cells[5] {
            "load" => {
                load += 1;
                RestoreDecision::Load
            }
            "recompute" => RestoreDecision::Recompute,
            _ => panic!("unknown fixture verdict"),
        };
        assert_eq!(decision, expected, "{row}");
        count += 1;
    }
    assert_eq!(count, 48);
    assert!(load > 0 && load < count);
}

#[test]
fn scheduler_equal_priority_deadline_alternates_tenants_before_second_fifo_turn() {
    use scheduler::*;
    let (mut h, g, c, _) = fixture();
    let mut other = bundle();
    other.program.tenant_salt = [77; 32];
    other.id = KvBlockId::new(&other.program, [0; 32], &[42], 0, 0, 0, 7).unwrap();
    h.backing.as_mut().unwrap().others.push(other.clone());
    h.backing.as_mut().unwrap().route = Some(Tier::Nvme);
    let mut s = Scheduler::new(h, 3);
    let first = s.enqueue(plan(&bundle()), PrefetchPolicy::Wait).unwrap();
    let last = s.enqueue(plan(&bundle()), PrefetchPolicy::Wait).unwrap();
    let mut p = plan(&other);
    p.request.tenant = other.program.tenant_salt;
    let middle = s.enqueue(p, PrefetchPolicy::Wait).unwrap();
    for id in [first, middle, last] {
        assert!(
            s.tick(1, &[], |_| epochs())
                .unwrap()
                .contains(&(id, Ok(Progress::Phase(Phase::Reserved))))
        );
        s.cancel(id).unwrap();
        c.borrow_mut().retired = true;
        s.release(id).unwrap();
    }
    assert_eq!(g.lock().unwrap().used(), TierBudget::zero(2));
}

#[test]
fn scheduler_rearrival_cannot_starve_backlogged_tenant() {
    use scheduler::*;
    let (mut h, g, c, _) = fixture();
    let mut other = bundle();
    other.program.tenant_salt = [77; 32];
    other.id = KvBlockId::new(&other.program, [0; 32], &[42], 0, 0, 0, 7).unwrap();
    h.backing.as_mut().unwrap().others.push(other.clone());
    h.backing.as_mut().unwrap().route = Some(Tier::Nvme);
    let mut s = Scheduler::new(h, 3);
    let first = s.enqueue(plan(&bundle()), PrefetchPolicy::Wait).unwrap();
    let last = s.enqueue(plan(&bundle()), PrefetchPolicy::Wait).unwrap();
    let mut p = plan(&other);
    p.request.tenant = other.program.tenant_salt;
    let middle = s.enqueue(p.clone(), PrefetchPolicy::Wait).unwrap();
    for id in [first, middle] {
        assert!(
            s.tick(1, &[], |_| epochs())
                .unwrap()
                .contains(&(id, Ok(Progress::Phase(Phase::Reserved))))
        );
        s.cancel(id).unwrap();
        c.borrow_mut().retired = true;
        s.release(id).unwrap();
    }
    let again = s.enqueue(p, PrefetchPolicy::Wait).unwrap();
    for id in [last, again] {
        assert!(
            s.tick(1, &[], |_| epochs())
                .unwrap()
                .contains(&(id, Ok(Progress::Phase(Phase::Reserved))))
        );
        s.cancel(id).unwrap();
        c.borrow_mut().retired = true;
        s.release(id).unwrap();
    }
    assert_eq!(g.lock().unwrap().used(), TierBudget::zero(2));
}

#[test]
fn scheduler_eviction_floor_protects_backlog_under_priority_churn() {
    use scheduler::*;
    let (mut h, g, c, _) = fixture();
    h.backing.as_mut().unwrap().route = Some(Tier::Nvme);
    let others: Vec<_> = (100..110)
        .map(|tenant| {
            let mut b = bundle();
            b.program.tenant_salt = [tenant; 32];
            b.id = KvBlockId::new(&b.program, [0; 32], &[42], 0, 0, 0, 7).unwrap();
            h.backing.as_mut().unwrap().others.push(b.clone());
            b
        })
        .collect();
    let mut s = Scheduler::new(h, 3);
    let mut b = plan(&bundle());
    b.request.priority = Priority::Demand;
    let b1 = s.enqueue(b.clone(), PrefetchPolicy::Wait).unwrap();
    let b2 = s.enqueue(b, PrefetchPolicy::Wait).unwrap();
    let serve = |s: &mut Scheduler<_>, id| {
        let events = s.tick(1, &[], |_| epochs()).unwrap();
        assert!(
            events.contains(&(id, Ok(Progress::Phase(Phase::Reserved)))),
            "want {id}, got {events:?}"
        );
        s.cancel(id).unwrap();
        c.borrow_mut().retired = true;
        s.release(id).unwrap();
    };
    serve(&mut s, b1);
    for other in &others {
        let mut p = plan(other);
        p.request.tenant = other.program.tenant_salt;
        p.request.priority = Priority::MandatoryActive;
        let id = s.enqueue(p, PrefetchPolicy::Wait).unwrap();
        serve(&mut s, id);
    }
    let mut p = plan(&others[0]);
    p.request.tenant = others[0].program.tenant_salt;
    p.request.priority = Priority::Demand;
    let again = s.enqueue(p, PrefetchPolicy::Wait).unwrap();
    serve(&mut s, b2);
    serve(&mut s, again);
    assert_eq!(g.lock().unwrap().used(), TierBudget::zero(2));
}
