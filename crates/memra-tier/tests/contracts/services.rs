use super::support::*;
use memra_tier::contracts::*;
use std::collections::HashMap;

pub struct Objects {
    gov: Shared,
    roots: HashMap<ObjectKey, (ObjectManifest, Vec<Vec<u8>>)>,
    leases: HashMap<ObjectKey, usize>,
    pub interrupt: bool,
}
pub struct Txn {
    key: ObjectKey,
    size: u64,
    chunks: Vec<Vec<u8>>,
    cancelled: bool,
    published: bool,
    failed: bool,
}
impl Objects {
    fn new(gov: Shared) -> Self {
        Self {
            gov,
            roots: HashMap::new(),
            leases: HashMap::new(),
            interrupt: false,
        }
    }
}
impl ObjectStore for Objects {
    type Transaction = Txn;
    fn lookup(&self, key: &ObjectKey) -> Result<Option<ObjectManifest>> {
        key.validate()?;
        Ok(self.roots.get(key).map(|(m, _)| m.clone()))
    }
    fn begin(&mut self, key: ObjectKey, size: u64, d: Durability) -> Result<Txn> {
        key.validate()?;
        if d == Durability::Persistent {
            return Err(Error::Unsupported);
        }
        Ok(Txn {
            key,
            size,
            chunks: vec![],
            cancelled: false,
            published: false,
            failed: false,
        })
    }
    fn put(&mut self, t: &mut Txn, p: &[u8]) -> Result<()> {
        if t.cancelled || t.failed {
            return Err(Error::Cancelled);
        }
        if t.published {
            return Err(Error::Conflict);
        }
        if p.is_empty()
            || p.len() > 4
            || t.chunks.iter().map(|b| b.len() as u64).sum::<u64>() + p.len() as u64 > t.size
        {
            t.failed = true;
            return Err(Error::ShortIo {
                expected: 4,
                actual: p.len() as u64,
            });
        }
        t.chunks.push(p.to_vec());
        Ok(())
    }
    fn commit(&mut self, t: &mut Txn) -> Result<ObjectManifest> {
        if t.cancelled || t.failed {
            return Err(Error::Cancelled);
        }
        if t.chunks.iter().map(|b| b.len() as u64).sum::<u64>() != t.size {
            return Err(Error::Incomplete);
        }
        if self.interrupt {
            return Err(Error::Io {
                kind: std::io::ErrorKind::WriteZero,
                os_code: None,
            });
        }
        let m = ObjectManifest {
            version: 1,
            key: t.key.clone(),
            valid_bytes: t.size,
            durability: Durability::Ephemeral,
            chunks: t
                .chunks
                .iter()
                .map(|p| ChunkRef {
                    version: 1,
                    encoded_digest: digest("extent", p),
                    valid_bytes: p.len() as u64,
                    storage_bytes: 4,
                    checksum: checksum(p),
                })
                .collect(),
        };
        m.validate()?;
        if let Some((old, _)) = self.roots.get(&t.key)
            && old != &m
        {
            return Err(Error::Conflict);
        }
        self.roots
            .insert(t.key.clone(), (m.clone(), t.chunks.clone()));
        t.published = true;
        Ok(m)
    }
    fn cancel(&mut self, t: &mut Txn) -> Result<CancelState> {
        if t.published {
            Ok(CancelState::AlreadyPublished)
        } else {
            t.cancelled = true;
            Ok(CancelState::PublicationRevoked)
        }
    }
    fn lease(&mut self, m: &ObjectManifest, r: &BudgetRequest) -> Result<ObjectLease> {
        m.validate()?;
        let (actual, _) = self.roots.get(&m.key).ok_or(Error::NotFound)?;
        if m != actual {
            return Err(Error::Conflict);
        }
        let charge = self.gov.borrow_mut().reserve(r)?;
        *self.leases.entry(m.key.clone()).or_default() += 1;
        Ok(ObjectLease {
            manifest: m.clone(),
            charge,
        })
    }
    fn read(&mut self, l: &ObjectLease, chunk: u32, dst: &mut [u8]) -> Result<u64> {
        if l.charge.state()? == ChargeState::Released {
            return Err(Error::AlreadyReleased);
        }
        let (m, chunks) = self.roots.get(&l.manifest.key).ok_or(Error::NotFound)?;
        if m != &l.manifest {
            return Err(Error::Conflict);
        }
        let c = m.chunks.get(chunk as usize).ok_or(Error::NotFound)?;
        let b = &chunks[chunk as usize];
        if b.len() as u64 != c.valid_bytes || checksum(b) != c.checksum {
            return Err(Error::Corrupt);
        }
        if dst.len() < b.len() {
            return Err(Error::Capacity);
        }
        dst[..b.len()].copy_from_slice(b);
        Ok(b.len() as u64)
    }
    fn release(&mut self, l: &ObjectLease) -> Result<()> {
        self.gov.borrow_mut().release(&l.charge)?;
        *self
            .leases
            .get_mut(&l.manifest.key)
            .ok_or(Error::ForeignLease)? -= 1;
        Ok(())
    }
    fn evict(&mut self, k: &ObjectKey) -> Result<()> {
        if self.leases.get(k).copied().unwrap_or(0) > 0 {
            return Err(Error::Busy);
        }
        self.roots.remove(k).ok_or(Error::NotFound)?;
        Ok(())
    }
}
#[test]
fn object_partial_short_and_interrupted_root_never_publish() {
    let mut s = Objects::new(shared());
    let k = key();
    let mut t = s.begin(k.clone(), 12, Durability::Ephemeral).unwrap();
    s.put(&mut t, &[1, 2, 3, 4]).unwrap();
    assert!(s.put(&mut t, &[0; 5]).is_err());
    assert_eq!(s.commit(&mut t), Err(Error::Cancelled));
    assert!(s.lookup(&k).unwrap().is_none());
    let mut t = s.begin(k.clone(), 4, Durability::Ephemeral).unwrap();
    s.put(&mut t, &[0; 4]).unwrap();
    s.interrupt = true;
    assert!(s.commit(&mut t).is_err());
    assert!(s.lookup(&k).unwrap().is_none());
}
#[test]
fn object_cancel_before_and_after_commit_and_chunked_progress() {
    let gov = shared();
    let mut s = Objects::new(gov.clone());
    let k = key();
    let mut t = s.begin(k.clone(), 9, Durability::Ephemeral).unwrap();
    for chunk in bytes(9).chunks(4) {
        s.put(&mut t, chunk).unwrap();
    } // object > one staging slot
    assert_eq!(s.cancel(&mut t), Ok(CancelState::PublicationRevoked));
    assert_eq!(s.commit(&mut t), Err(Error::Cancelled));
    let mut t = s.begin(k.clone(), 9, Durability::Ephemeral).unwrap();
    for chunk in bytes(9).chunks(4) {
        s.put(&mut t, chunk).unwrap();
    }
    let m = s.commit(&mut t).unwrap();
    assert_eq!(s.cancel(&mut t), Ok(CancelState::AlreadyPublished));
    let l = s.lease(&m, &request(9, Priority::Demand)).unwrap();
    let mut out = vec![];
    for i in 0..3 {
        let mut slot = [0; 4];
        let n = s.read(&l, i, &mut slot).unwrap();
        out.extend_from_slice(&slot[..n as usize]);
    }
    assert_eq!(out, bytes(9));
    assert_eq!(gov.borrow().used.pageable, 9);
    assert_eq!(s.evict(&k), Err(Error::Busy));
    s.release(&l).unwrap();
    assert_eq!(s.release(&l), Err(Error::AlreadyReleased));
    assert_eq!(gov.borrow().used.pageable, 0);
    s.evict(&k).unwrap();
}
#[test]
fn object_full_key_collision_corruption_and_version_refusal() {
    let mut s = Objects::new(shared());
    let mut t = s.begin(key(), 4, Durability::Ephemeral).unwrap();
    s.put(&mut t, &bytes(4)).unwrap();
    let m = s.commit(&mut t).unwrap();
    let mut wrong = m.clone();
    wrong.chunks[0].checksum = [0; 32];
    assert!(matches!(
        s.lease(&wrong, &request(4, Priority::Demand)),
        Err(Error::Conflict)
    ));
    let mut wrongkey = key();
    wrongkey.artifact = [99; 32];
    assert!(s.lookup(&wrongkey).unwrap().is_none());
    wrongkey.version = 2;
    assert_eq!(s.lookup(&wrongkey), Err(Error::UnsupportedVersion(2)));
    let l = s.lease(&m, &request(4, Priority::Demand)).unwrap();
    s.roots.get_mut(&key()).unwrap().1[0][0] ^= 1;
    assert_eq!(s.read(&l, 0, &mut [0; 4]), Err(Error::Corrupt));
    assert!(matches!(
        s.begin(key(), 1, Durability::Persistent),
        Err(Error::Unsupported)
    ));
}

struct TierEntry {
    plan: TierAdmission,
    block: BlockLease,
    phase: Phase,
    retired: bool,
    published: bool,
}
struct Tiers {
    gov: Shared,
    stored: StateBundle,
    entries: HashMap<(u64, u64), TierEntry>,
    next: u64,
}
impl Tiers {
    fn new(gov: Shared) -> Self {
        Self {
            gov,
            stored: bundle(),
            entries: HashMap::new(),
            next: 0,
        }
    }
    fn plan(&self) -> TierAdmission {
        let mut r = request(0, Priority::MandatoryActive);
        r.bytes.device[0] = 4;
        r.bytes.staging = 4;
        r.bytes.inflight = 1;
        TierAdmission {
            id: self.stored.id.clone(),
            program: self.stored.program.clone(),
            expected_layout: self.stored.layout.clone(),
            source: Tier::Nvme,
            target_device: 0,
            epochs: epochs(),
            committed_high_water: 3,
            request: r,
        }
    }
    fn entry(&mut self, r: &TierReservation) -> Result<&mut TierEntry> {
        self.entries
            .get_mut(&r.charge.id())
            .ok_or(Error::ForeignLease)
    }
}
impl TierStore for Tiers {
    fn lookup(&self, id: &KvBlockId, _peers: &[u32], _local: u32) -> Result<Option<Lookup>> {
        Ok((id == &self.stored.id).then_some(Lookup {
            tier: Tier::Nvme,
            storage_bytes: 4,
        }))
    }
    fn admit(&mut self, p: TierAdmission) -> Result<TierReservation> {
        self.stored.require(
            &p.program,
            &p.expected_layout,
            p.epochs.state,
            p.committed_high_water,
        )?;
        if self.stored.id != p.id {
            return Err(Error::ProgramMismatch);
        }
        if p.request
            .bytes
            .device
            .get(p.target_device as usize)
            .copied()
            .unwrap_or(0)
            < p.expected_layout.storage_bytes()?
        {
            return Err(Error::Capacity);
        }
        let charge = self.gov.borrow_mut().reserve(&p.request)?;
        let block = BlockLease {
            bundle: self.stored.clone(),
            tier: p.source,
            charge: self
                .gov
                .borrow_mut()
                .reserve(&request(0, Priority::MandatoryActive))?,
        };
        self.entries.insert(
            charge.id(),
            TierEntry {
                plan: p,
                block,
                phase: Phase::Reserved,
                retired: false,
                published: false,
            },
        );
        Ok(TierReservation { charge })
    }
    fn prefetch(&mut self, r: &TierReservation) -> Result<TransferTicket> {
        self.next += 1;
        let n = self.next;
        let e = self.entry(r)?;
        e.phase = Phase::Prefetching;
        Ok(TransferTicket {
            issuer: 5,
            sequence: n,
            epochs: e.plan.epochs,
        })
    }
    fn load(&mut self, r: &TierReservation) -> Result<TransferTicket> {
        self.next += 1;
        let n = self.next;
        let e = self.entry(r)?;
        if e.phase != Phase::HostReady {
            return Err(Error::NotReady);
        }
        e.phase = Phase::Loading;
        Ok(TransferTicket {
            issuer: 5,
            sequence: n,
            epochs: e.plan.epochs,
        })
    }
    fn advance(&mut self, r: &TierReservation, current: Epochs) -> Result<Phase> {
        let e = self.entry(r)?;
        e.plan.epochs.require(current)?;
        e.phase = match e.phase {
            Phase::Prefetching => Phase::HostReady,
            Phase::Loading => Phase::Ready,
            p => p,
        };
        Ok(e.phase)
    }
    fn ready(
        &mut self,
        r: &TierReservation,
        p: &ProgramIdentity,
        current: Epochs,
    ) -> Result<&BlockLease> {
        let e = self.entry(r)?;
        e.plan.epochs.require(current)?;
        if e.phase != Phase::Ready {
            return Err(Error::NotReady);
        }
        if p != &e.block.bundle.program {
            return Err(Error::ProgramMismatch);
        }
        e.published = true;
        Ok(&e.block)
    }
    fn cancel(&mut self, r: &TierReservation) -> Result<CancelState> {
        let e = self.entry(r)?;
        if e.published {
            Ok(CancelState::AlreadyPublished)
        } else {
            e.phase = Phase::Cancelled;
            Ok(CancelState::PublicationRevoked)
        }
    }
    fn retire(&mut self, r: &TierReservation) -> Result<bool> {
        Ok(self.entry(r)?.retired)
    }
    fn release(&mut self, r: &TierReservation) -> Result<()> {
        if !self.retire(r)? {
            return Err(Error::Busy);
        }
        let e = self
            .entries
            .get(&r.charge.id())
            .ok_or(Error::ForeignLease)?;
        let charge = &e.block.charge;
        self.gov.borrow_mut().release(charge)?;
        self.gov.borrow_mut().release(&r.charge)?;
        self.entries.remove(&r.charge.id());
        Ok(())
    }
    fn evict(&mut self, id: &KvBlockId) -> Result<()> {
        if self.entries.values().any(|e| &e.block.bundle.id == id) {
            Err(Error::Busy)
        } else {
            Err(Error::NotFound)
        }
    }
}
#[test]
fn tier_advisory_lookup_identity_completeness_and_epochs() {
    let gov = shared();
    let mut s = Tiers::new(gov.clone());
    s.lookup(&s.stored.id, &[], 0).unwrap();
    assert_eq!(gov.borrow().used.device[0], 0);
    let mut p = s.plan();
    p.program.numeric = [88; 32];
    assert!(matches!(s.admit(p), Err(Error::ProgramMismatch)));
    let mut p = s.plan();
    p.expected_layout.segments.clear();
    assert!(matches!(s.admit(p), Err(Error::InvalidLayout)));
    let r = s.admit(s.plan()).unwrap();
    s.prefetch(&r).unwrap();
    assert_eq!(s.advance(&r, epochs()), Ok(Phase::HostReady));
    assert!(matches!(
        s.ready(&r, &program(), epochs()),
        Err(Error::NotReady)
    ));
    s.load(&r).unwrap();
    s.advance(&r, epochs()).unwrap();
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
        assert!(matches!(s.ready(&r, &program(), e), Err(Error::StaleEpoch)));
    }
    let mut wrong = program();
    wrong.stream = [99; 32];
    assert!(matches!(
        s.ready(&r, &wrong, epochs()),
        Err(Error::ProgramMismatch)
    ));
    s.ready(&r, &program(), epochs()).unwrap();
    assert_eq!(s.cancel(&r), Ok(CancelState::AlreadyPublished));
    assert_eq!(s.release(&r), Err(Error::Busy));
    s.entry(&r).unwrap().retired = true;
    s.release(&r).unwrap();
    assert_eq!(gov.borrow().used.device[0], 0);
}
#[test]
fn tier_cancel_after_load_before_ready_and_unknown_retirement() {
    let mut s = Tiers::new(shared());
    let r = s.admit(s.plan()).unwrap();
    s.prefetch(&r).unwrap();
    s.advance(&r, epochs()).unwrap();
    s.load(&r).unwrap();
    s.advance(&r, epochs()).unwrap();
    assert_eq!(s.cancel(&r), Ok(CancelState::PublicationRevoked));
    assert!(matches!(
        s.ready(&r, &program(), epochs()),
        Err(Error::NotReady)
    ));
    assert!(!s.retire(&r).unwrap());
    assert_eq!(s.release(&r), Err(Error::Busy));
}
#[test]
fn common_governor_headroom_foreign_double_release_and_pin_lifetime() {
    let mut g = Governor::new(100);
    let optional = g.reserve(&request(75, Priority::OptionalPrefetch)).unwrap();
    assert!(matches!(
        g.reserve(&request(1, Priority::Backup)),
        Err(Error::Capacity)
    ));
    let active = g.reserve(&request(25, Priority::MandatoryActive)).unwrap();
    assert_eq!(g.used.pageable, 100);
    let pin = active.pin().unwrap();
    assert_eq!(g.release(&active), Err(Error::Busy));
    drop(pin);
    g.mark(&active, ChargeState::InUse).unwrap();
    g.mark(&active, ChargeState::Quarantined).unwrap();
    assert_eq!(g.release(&active), Err(Error::Busy));
    g.mark(&active, ChargeState::Retired).unwrap();
    g.release(&active).unwrap();
    assert_eq!(g.release(&active), Err(Error::AlreadyReleased));
    let mut foreign = Governor::new(100);
    assert_eq!(foreign.release(&optional), Err(Error::ForeignLease));
    assert_eq!(foreign.used.pageable, 0);
    g.release(&optional).unwrap();
    assert_eq!(g.used.pageable, 0);
}

struct BankEntry {
    ids: Vec<BankId>,
    epochs: Epochs,
    cancelled: bool,
    published: bool,
    complete: bool,
    retired: bool,
}
struct Banks {
    gov: Shared,
    catalog: HashMap<BankId, Option<RecordLayout>>,
    pending: HashMap<TransferTicket, BankEntry>,
    next: u64,
    pub reads: usize,
    class: LayoutClass,
}
impl Banks {
    fn new(gov: Shared) -> Self {
        Self {
            gov,
            catalog: HashMap::new(),
            pending: HashMap::new(),
            next: 0,
            reads: 0,
            class: LayoutClass::Uniform,
        }
    }
    fn add(&mut self, n: u64, row: bool) -> BankId {
        let tensor = TensorId {
            version: 1,
            artifact: [1; 32],
            name: "model.experts".into(),
        };
        let mut l = layout();
        l.segments[0].tensor = Some(tensor.clone());
        let id = BankId {
            version: 1,
            tensor,
            record: if row {
                RecordId::Row(n)
            } else {
                RecordId::Expert {
                    layer: 0,
                    original_id: n as u32,
                    projection: Projection::Up,
                }
            },
            layout: l.identity().unwrap(),
        };
        self.catalog.insert(id.clone(), Some(l));
        id
    }
    fn enqueue(&mut self, ids: Vec<BankId>, e: Epochs) -> Result<TransferTicket> {
        if ids.is_empty() {
            return Err(Error::EmptyBatch);
        }
        for id in &ids {
            BankedResidency::layout(self, id)?;
        }
        self.next += 1;
        let t = TransferTicket {
            issuer: 12,
            sequence: self.next,
            epochs: e,
        };
        self.reads += ids.iter().collect::<std::collections::HashSet<_>>().len();
        self.pending.insert(
            t,
            BankEntry {
                ids,
                epochs: e,
                cancelled: false,
                published: false,
                complete: true,
                retired: false,
            },
        );
        Ok(t)
    }
    fn publish_records(&mut self, t: &TransferTicket, current: Epochs) -> Result<Vec<BankLease>> {
        let e = self.pending.get_mut(t).ok_or(Error::UnknownTicket)?;
        e.epochs.require(current)?;
        if e.cancelled {
            return Err(Error::Cancelled);
        }
        if !e.complete {
            return Err(Error::Incomplete);
        }
        if e.published {
            return Err(Error::Busy);
        }
        let mut unique = HashMap::new();
        let mut result = vec![];
        for id in &e.ids {
            if !unique.contains_key(id) {
                let l = self.catalog[id].as_ref().ok_or(Error::MaskedId)?.clone();
                let charge = self
                    .gov
                    .borrow_mut()
                    .reserve(&request(l.storage_bytes()?, Priority::Demand))?;
                unique.insert(
                    id.clone(),
                    BankLease::from_backend(
                        id.clone(),
                        l,
                        self.class,
                        charge,
                        Box::new(match id.record {
                            RecordId::Row(n) => vec![n as u8; 4],
                            _ => bytes(4),
                        }),
                    )
                    .map_err(|rejected| rejected.error)?,
                );
            }
            result.push(unique[id].clone());
        }
        e.published = true;
        Ok(result)
    }
}
impl BankedResidency for Banks {
    fn layout(&self, id: &BankId) -> Result<&RecordLayout> {
        self.catalog
            .get(id)
            .ok_or(Error::NotFound)?
            .as_ref()
            .ok_or(Error::MaskedId)
    }
    fn resident(&mut self, id: &BankId) -> Result<Option<BankLease>> {
        BankedResidency::layout(self, id)?;
        Ok(None)
    }
    fn stage(&mut self, b: BankBatch) -> Result<TransferTicket> {
        if b.ids.iter().any(|i| !ExpertDomain::accepts(&i.record)) {
            return Err(Error::InvalidLayout);
        }
        self.enqueue(b.ids, b.epochs)
    }
    fn publish(&mut self, t: &TransferTicket, e: Epochs) -> Result<Vec<BankLease>> {
        self.publish_records(t, e)
    }
    fn cancel(&mut self, t: &TransferTicket) -> Result<CancelState> {
        let e = self.pending.get_mut(t).ok_or(Error::UnknownTicket)?;
        if e.published {
            Ok(CancelState::AlreadyPublished)
        } else {
            e.cancelled = true;
            Ok(CancelState::PublicationRevoked)
        }
    }
    fn retire(&mut self, t: &TransferTicket) -> Result<bool> {
        Ok(self.pending.get(t).ok_or(Error::UnknownTicket)?.retired)
    }
    fn release(&mut self, l: &BankLease) -> Result<()> {
        if self
            .pending
            .values()
            .any(|e| e.published && !e.retired && e.ids.contains(l.id()))
        {
            return Err(Error::Busy);
        }
        l.retire_backing()?;
        self.gov.borrow_mut().release(l.charge())
    }
}
impl RowService for Banks {
    fn gather(&mut self, b: RowBatch) -> Result<TransferTicket> {
        if b.ids.iter().any(|i| !RowDomain::accepts(&i.record)) {
            return Err(Error::InvalidLayout);
        }
        self.enqueue(b.ids, b.epochs)
    }
    fn publish(&mut self, t: &TransferTicket, e: Epochs) -> Result<RowLease> {
        Ok(RowLease {
            records: self.publish_records(t, e)?,
        })
    }
    fn cancel(&mut self, t: &TransferTicket) -> Result<CancelState> {
        BankedResidency::cancel(self, t)
    }
    fn retire(&mut self, t: &TransferTicket) -> Result<bool> {
        BankedResidency::retire(self, t)
    }
    fn release(&mut self, l: &RowLease) -> Result<()> {
        let mut seen = std::collections::HashSet::new();
        for r in &l.records {
            if seen.insert(r.charge().id()) {
                BankedResidency::release(self, r)?;
            }
        }
        Ok(())
    }
}
#[test]
fn bank_masked_ids_partial_completion_and_uniform_mixed_refusal() {
    let gov = shared();
    let mut b = Banks::new(gov.clone());
    let ids = (0..3).map(|i| b.add(i, false)).collect::<Vec<_>>();
    b.catalog.insert(ids[1].clone(), None);
    let batch = BankBatch {
        ids: ids.clone(),
        epochs: epochs(),
        request: request(12, Priority::Demand),
    };
    assert_eq!(b.stage(batch.clone()), Err(Error::MaskedId));
    assert_eq!(b.reads, 0);
    let one = BankBatch {
        ids: vec![ids[0].clone()],
        ..batch
    };
    let t = b.stage(one.clone()).unwrap();
    b.pending.get_mut(&t).unwrap().complete = false;
    assert!(matches!(
        BankedResidency::publish(&mut b, &t, epochs()),
        Err(Error::Incomplete)
    ));
    b.pending.get_mut(&t).unwrap().complete = true;
    assert_eq!(
        BankedResidency::cancel(&mut b, &t),
        Ok(CancelState::PublicationRevoked)
    );
    assert!(matches!(
        BankedResidency::publish(&mut b, &t, epochs()),
        Err(Error::Cancelled)
    ));
    let t = b.stage(one.clone()).unwrap();
    b.class = LayoutClass::PerRecord;
    let leases = BankedResidency::publish(&mut b, &t, epochs()).unwrap();
    let rejected = UniformLease::try_new(leases).unwrap_err();
    assert_eq!(rejected.error, Error::MixedLayout);
    assert_eq!(
        BankedResidency::release(&mut b, &rejected.op[0]),
        Err(Error::Busy)
    );
    b.pending.get_mut(&t).unwrap().retired = true;
    BankedResidency::release(&mut b, &rejected.op[0]).unwrap();
    b.class = LayoutClass::Uniform;
    let t = b.stage(one).unwrap();
    let leases = BankedResidency::publish(&mut b, &t, epochs()).unwrap();
    assert_eq!(UniformLease::try_new(leases).unwrap().records().len(), 1);
}
#[test]
fn rows_duplicate_order_dedup_charge_and_delayed_consumer_retirement() {
    let gov = shared();
    let mut b = Banks::new(gov.clone());
    let five = b.add(5, true);
    let two = b.add(2, true);
    let nine = b.add(9, true);
    let ids = vec![five.clone(), two, five, nine];
    let t = b
        .gather(RowBatch {
            ids: ids.clone(),
            epochs: epochs(),
            request: request(16, Priority::Demand),
        })
        .unwrap();
    assert_eq!(b.reads, 3);
    let wrong = Epochs {
        state: 8,
        ..epochs()
    };
    assert!(matches!(
        RowService::publish(&mut b, &t, wrong),
        Err(Error::StaleEpoch)
    ));
    let l = RowService::publish(&mut b, &t, epochs()).unwrap();
    assert_eq!(
        l.records.iter().map(|l| l.id().clone()).collect::<Vec<_>>(),
        ids
    );
    assert_eq!(gov.borrow().used.pageable, 12);
    assert_eq!(
        l.records
            .iter()
            .map(|r| r.resource::<Vec<u8>>().unwrap()[0])
            .collect::<Vec<_>>(),
        vec![5, 2, 5, 9]
    );
    assert_eq!(
        RowService::cancel(&mut b, &t),
        Ok(CancelState::AlreadyPublished)
    );
    assert_eq!(RowService::release(&mut b, &l), Err(Error::Busy));
    b.pending.get_mut(&t).unwrap().retired = true;
    RowService::release(&mut b, &l).unwrap();
    assert_eq!(gov.borrow().used.pageable, 0);
}
#[test]
fn rows_rejected_read_and_cancel_never_publish_logical_subset() {
    let mut b = Banks::new(shared());
    let ids = vec![b.add(5, true), b.add(2, true)];
    let t = b
        .gather(RowBatch {
            ids,
            epochs: epochs(),
            request: request(8, Priority::Demand),
        })
        .unwrap();
    b.pending.get_mut(&t).unwrap().complete = false;
    assert!(matches!(
        RowService::publish(&mut b, &t, epochs()),
        Err(Error::Incomplete)
    ));
    b.pending.get_mut(&t).unwrap().complete = true;
    RowService::cancel(&mut b, &t).unwrap();
    assert!(matches!(
        RowService::publish(&mut b, &t, epochs()),
        Err(Error::Cancelled)
    ));
}
#[test]
fn reusable_object_tier_bank_row_schedules() {
    let mut s = Objects::new(shared());
    super::conformance::object_cancel(&mut s, key());
    let mut t = Tiers::new(shared());
    let r = t.admit(t.plan()).unwrap();
    super::conformance::tier_cancel(&mut t, &r, &program(), epochs());
    let mut bank = Banks::new(shared());
    let id = bank.add(0, false);
    super::conformance::bank_cancel(
        &mut bank,
        BankBatch {
            ids: vec![id],
            epochs: epochs(),
            request: request(4, Priority::Demand),
        },
    );
    let id = bank.add(5, true);
    super::conformance::rows_order(
        &mut bank,
        RowBatch {
            ids: vec![id.clone(), id],
            epochs: epochs(),
            request: request(8, Priority::Demand),
        },
    );
}
#[test]
fn bank_scale_plane_missing_is_not_a_complete_uniform_record() {
    let mut l = layout();
    let tensor = TensorId {
        version: 1,
        artifact: [1; 32],
        name: "expert.up".into(),
    };
    l.segments[0].tensor = Some(tensor);
    let mut scale = l.segments[0].clone();
    scale.role = Role::MacroScale;
    scale.offset = 4;
    l.segments.push(scale);
    let mut required = l.requirements[0].clone();
    required.role = Role::MacroScale;
    l.requirements.push(required);
    l.validate().unwrap();
    l.segments.pop();
    assert_eq!(l.validate(), Err(Error::Incomplete));
}
#[test]
fn budget_device_headroom_and_priority_deadline_order() {
    let mut g = Governor::new(100);
    g.headroom.device[0] = 25;
    let mut r = request(0, Priority::OptionalPrefetch);
    r.bytes.device[0] = 75;
    g.reserve(&r).unwrap();
    r.bytes.device[0] = 1;
    assert!(matches!(g.reserve(&r), Err(Error::Capacity)));
    r.priority = Priority::MandatoryActive;
    r.bytes.device[0] = 25;
    g.reserve(&r).unwrap();
    let mut queue = vec![
        (Priority::Backup, Deadline(1)),
        (Priority::Demand, Deadline(9)),
        (Priority::MandatoryActive, Deadline(99)),
        (Priority::OptionalPrefetch, Deadline(1)),
        (Priority::AdmittedRestore, Deadline(99)),
        (Priority::Demand, Deadline(2)),
    ];
    queue.sort();
    assert_eq!(
        queue,
        vec![
            (Priority::MandatoryActive, Deadline(99)),
            (Priority::AdmittedRestore, Deadline(99)),
            (Priority::Demand, Deadline(2)),
            (Priority::Demand, Deadline(9)),
            (Priority::OptionalPrefetch, Deadline(1)),
            (Priority::Backup, Deadline(1))
        ]
    );
}
#[test]
fn bank_lease_keeps_backing_pinned_and_explicit_retirement_invalidates_aliases() {
    let gov = shared();
    let mut b = Banks::new(gov.clone());
    let id = b.add(1, true);
    let t = b
        .gather(RowBatch {
            ids: vec![id],
            epochs: epochs(),
            request: request(4, Priority::Demand),
        })
        .unwrap();
    let l = RowService::publish(&mut b, &t, epochs()).unwrap();
    let alias = l.records[0].clone();
    assert_eq!(gov.borrow_mut().release(alias.charge()), Err(Error::Busy));
    b.pending.get_mut(&t).unwrap().retired = true;
    {
        let view = alias.resource::<Vec<u8>>().unwrap();
        assert_eq!(RowService::release(&mut b, &l), Err(Error::Busy));
        assert_eq!(view[0], 1);
    }
    RowService::release(&mut b, &l).unwrap();
    assert!(matches!(
        alias.resource::<Vec<u8>>(),
        Err(Error::AlreadyReleased)
    ));
}
#[test]
fn invalid_bank_publication_returns_charge_and_backing_for_explicit_release() {
    let gov = shared();
    let mut b = Banks::new(gov.clone());
    let id = b.add(0, false);
    let mut wrong = BankedResidency::layout(&b, &id).unwrap().clone();
    wrong.segments.clear();
    let charge = gov
        .borrow_mut()
        .reserve(&request(4, Priority::Demand))
        .unwrap();
    let rejected = BankLease::from_backend(
        id,
        wrong,
        LayoutClass::Uniform,
        charge,
        Box::new(vec![1u8, 2, 3, 4]),
    )
    .unwrap_err();
    assert_eq!(rejected.error, Error::Incomplete);
    assert_eq!(
        rejected.op.backing.downcast_ref::<Vec<u8>>().unwrap(),
        &[1, 2, 3, 4]
    );
    gov.borrow_mut().release(&rejected.op.charge).unwrap();
    assert_eq!(gov.borrow().used.pageable, 0);
}

#[test]
fn revision_v11_generic_governor_and_object() {
    super::conformance::budget_governor(&mut Governor::new(100), &mut Governor::new(100));
    let mut s = Objects::new(shared());
    super::conformance::object_publish_release(&mut s, key(), request(4, Priority::Demand));
}

#[test]
fn revision_v11_bank_and_rows_complete_before_cancel() {
    let mut b = Banks::new(shared());
    let id = b.add(0, false);
    super::conformance::bank_complete_cancel(
        &mut b,
        BankBatch {
            ids: vec![id],
            epochs: epochs(),
            request: request(4, Priority::Demand),
        },
        |b, t| {
            assert!(b.pending[t].complete);
            completion(*t, 1)
        },
    );
    let id = b.add(5, true);
    super::conformance::rows_complete_cancel(
        &mut b,
        RowBatch {
            ids: vec![id],
            epochs: epochs(),
            request: request(4, Priority::Demand),
        },
        |b, t| {
            assert!(b.pending[t].complete);
            completion(*t, 1)
        },
    );
}

#[test]
fn revision_v11_bank_row_unknown_retirement_hooks() {
    use super::conformance::{self, LifetimeStep};
    // The frozen fake has synchronous stage; completion is deliberately withheld
    // to model a future asynchronous backend. This is NOT C's host-only adapter.
    let mut b = Banks::new(shared());
    let id = b.add(0, false);
    let t = b
        .stage(BankBatch {
            ids: vec![id],
            epochs: epochs(),
            request: request(4, Priority::Demand),
        })
        .unwrap();
    conformance::bank_lifetime(
        &mut b,
        t,
        |b, t, step| {
            let e = b.pending.get_mut(t).unwrap();
            if matches!(step, LifetimeStep::Unknown) {
                e.complete = false;
            }
            if matches!(step, LifetimeStep::Graph) {
                e.retired = true;
            }
        },
        |b| b.gov.borrow().used(),
    );
    let id = b.add(5, true);
    let t = b
        .gather(RowBatch {
            ids: vec![id],
            epochs: epochs(),
            request: request(4, Priority::Demand),
        })
        .unwrap();
    conformance::rows_lifetime(
        &mut b,
        t,
        |b, t, step| {
            let e = b.pending.get_mut(t).unwrap();
            if matches!(step, LifetimeStep::Unknown) {
                e.complete = false;
            }
            if matches!(step, LifetimeStep::Graph) {
                e.retired = true;
            }
        },
        |b| b.gov.borrow().used(),
    );
}

#[test]
fn revision_v12_tier_logical_record_preserved() {
    let gov = shared();
    let mut tier = Tiers::new(gov.clone());
    let r = tier.admit(tier.plan()).unwrap();
    tier.prefetch(&r).unwrap();
    assert_eq!(tier.advance(&r, epochs()).unwrap(), Phase::HostReady);
    tier.load(&r).unwrap();
    assert_eq!(tier.advance(&r, epochs()).unwrap(), Phase::Ready);
    let expected = tier.stored.clone();
    super::conformance::tier_logical_bytes(&mut tier, &r, &expected, epochs());
    assert_eq!(tier.release(&r), Err(Error::Busy));
    tier.entry(&r).unwrap().retired = true;
    tier.release(&r).unwrap();
    assert_eq!(gov.borrow().used, TierBudget::zero(2));
}
