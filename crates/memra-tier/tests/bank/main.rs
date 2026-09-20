#[path = "../../../memra-engine/src/banked_residency.rs"]
mod engine_bridge;
mod host_bridge;
// API-shaped stand-in; this does NOT compile the actual CUDA HostExps type.
mod model {
    #[derive(Clone, Copy)]
    pub struct ExpertLayout {
        pub offset: usize,
        pub len: usize,
        pub qtype: i32,
        pub row_bytes: usize,
    }
    pub struct HostExpertFp8BlockScales {
        pub scales: Vec<f32>,
        pub expert_stride: usize,
    }
    pub struct HostExps {
        pub n_expert: usize,
        pub qtype: i32,
        pub row_bytes: usize,
        pub expert_stride: usize,
        pub layouts: Option<Vec<ExpertLayout>>,
        pub tiers: Option<Vec<()>>,
        pub macros: Option<Vec<f32>>,
        pub fp8_blk: Option<HostExpertFp8BlockScales>,
    }
    impl HostExps {
        pub fn is_uniform_layout(&self) -> bool {
            self.layouts.is_none()
        }
        pub fn max_expert_bytes(&self) -> usize {
            self.layouts
                .as_ref()
                .and_then(|ls| ls.iter().map(|l| l.len).max())
                .unwrap_or(self.expert_stride)
        }
        pub fn expert_layout(&self, e: usize) -> ExpertLayout {
            self.layouts
                .as_ref()
                .map(|ls| ls[e])
                .unwrap_or(ExpertLayout {
                    offset: e * self.expert_stride,
                    len: self.expert_stride,
                    qtype: self.qtype,
                    row_bytes: self.row_bytes,
                })
        }
    }
}
mod ple_oracle;
use memra_tier::conformance;
use memra_tier::{bank::*, contracts::*};
use std::{cell::RefCell, collections::BTreeMap, rc::Rc};
#[allow(dead_code)]
#[path = "../contracts/support.rs"]
mod support;
use support::{Governor, epochs, request};

#[derive(Default)]
struct Heat(BTreeMap<BankId, u64>);
impl<D: BankDomain> Hotness<D> for Heat {
    fn demand(&mut self, id: &BankId) {
        *self.0.entry(id.clone()).or_default() += 1;
    }
    fn score(&self, id: &BankId) -> u64 {
        self.0.get(id).copied().unwrap_or(0)
    }
}
#[derive(Default)]
struct Reader {
    calls: Vec<(TensorId, u64, usize)>,
    fail: Option<usize>,
    corrupt: bool,
}
fn byte(t: &TensorId, p: u64) -> u8 {
    ((p * 13 + t.name.len() as u64) % 251) as u8
}
impl ExactReader for Reader {
    fn storage_bytes(&self, _: &TensorId) -> Result<u64> {
        Ok(1 << 28)
    }
    fn read_exact(&mut self, t: &TensorId, o: u64, dst: &mut [u8]) -> Result<()> {
        self.calls.push((t.clone(), o, dst.len()));
        if self.fail == Some(self.calls.len()) {
            return Err(Error::ShortIo {
                expected: dst.len() as u64,
                actual: 0,
            });
        }
        for (i, b) in dst.iter_mut().enumerate() {
            *b = byte(t, o + i as u64) ^ u8::from(self.corrupt);
        }
        Ok(())
    }
}
fn tensor() -> TensorId {
    TensorId {
        version: 1,
        artifact: [7; 32],
        name: "expert.weight".into(),
    }
}
fn layout(n: u64, q: i32, len: u64) -> RecordLayout {
    let payload = ByteSegment {
        version: 1,
        group: 0,
        page: 0,
        owner: 0,
        role: Role::Payload,
        tensor: Some(tensor()),
        offset: n * 32768,
        valid_bytes: len,
        storage_bytes: len,
        alignment: 1,
        encoding: EncodingId {
            version: 1,
            program: digest("host-exps-qtype-v1", &q.to_le_bytes()),
            row_bytes: len,
        },
    };
    let scale = ByteSegment {
        group: 1,
        role: Role::MacroScale,
        tensor: Some(TensorId {
            name: "expert.macro".into(),
            ..tensor()
        }),
        offset: n * 4,
        valid_bytes: 4,
        storage_bytes: 4,
        encoding: EncodingId {
            row_bytes: 4,
            program: digest("encoding", b"f32-le"),
            version: 1,
        },
        ..payload.clone()
    };
    let segments = vec![payload, scale];
    RecordLayout {
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
    }
}
fn record(l: RecordLayout) -> CatalogRecord {
    let checksums = l
        .segments
        .iter()
        .map(|s| {
            checksum(
                &(s.offset..s.offset + s.valid_bytes)
                    .map(|p| byte(s.tensor.as_ref().unwrap(), p))
                    .collect::<Vec<_>>(),
            )
        })
        .collect();
    CatalogRecord {
        layout: l,
        checksums,
    }
}
fn bank_id(n: u32, l: &RecordLayout) -> BankId {
    BankId {
        version: 1,
        tensor: tensor(),
        record: RecordId::Expert {
            layer: 2,
            original_id: n,
            projection: Projection::Gate,
        },
        layout: l.identity().unwrap(),
    }
}
fn catalog(class: LayoutClass) -> (Catalog, Vec<BankId>) {
    let mut entries = Vec::new();
    let mut ids = Vec::new();
    for (n, q, len) in [
        (9, 2, 16),
        (47, if class == LayoutClass::Uniform { 2 } else { 3 }, 16),
        (
            83,
            if class == LayoutClass::Uniform { 2 } else { 4 },
            if class == LayoutClass::Uniform {
                16
            } else {
                32
            },
        ),
    ] {
        let l = layout(n as u64, q, len);
        let id = bank_id(n, &l);
        ids.push(id.clone());
        entries.push((id, Some(record(l))));
    }
    let masked = BankId {
        record: RecordId::Expert {
            layer: 2,
            original_id: 21,
            projection: Projection::Gate,
        },
        ..ids[0].clone()
    };
    ids.push(masked.clone());
    entries.push((masked, None));
    (Catalog::new(class, entries).unwrap(), ids)
}
fn gov() -> Rc<RefCell<Governor>> {
    Rc::new(RefCell::new(Governor::new(1_000_000)))
}
fn limits(cache: u64) -> BankLimits {
    BankLimits {
        cache_bytes: cache,
        batch_bytes: 100_000,
        items: 128,
        tickets: 4,
    }
}
type Banks = BankService<ExpertDomain, Heat, Reader>;
fn banks(class: LayoutClass, cache: u64, r: Reader, g: SharedBudget) -> (Banks, Vec<BankId>) {
    let (c, ids) = catalog(class);
    (
        BankService::new(
            c,
            g,
            Heat::default(),
            r,
            CoalescingPolicy::default(),
            limits(cache),
        )
        .unwrap(),
        ids,
    )
}
fn batch(ids: Vec<BankId>) -> BankBatch {
    BankBatch {
        ids,
        epochs: epochs(),
        request: request(0, Priority::Demand),
    }
}
fn finish<D: BankDomain, H: Hotness<D>, R: ExactReader>(
    b: &mut BankService<D, H, R>,
    t: &TransferTicket,
) {
    drain(b, t);
    b.finish_host_use(t).unwrap();
    assert!(b.retire(t).unwrap());
    b.acknowledge(t).unwrap();
}
fn expected(l: &RecordLayout) -> Vec<u8> {
    l.segments
        .iter()
        .flat_map(|s| {
            (s.offset..s.offset + s.valid_bytes).map(|p| byte(s.tensor.as_ref().unwrap(), p))
        })
        .collect()
}

#[test]
fn frozen_bank_cancel_schedule_runs_on_implementation() {
    let g = gov();
    let (mut b, ids) = banks(LayoutClass::PerRecord, 0, Reader::default(), g.clone());
    conformance::bank_cancel(&mut b, batch(vec![ids[0].clone()]));
    assert!(g.borrow().used.pageable > 0);
    for t in b.tickets() {
        finish(&mut b, &t);
    }
    assert_eq!(g.borrow().used, TierBudget::zero(2));
}
#[test]
fn cancel_retains_charge_until_retirement_and_tombstone_ack() {
    let g = gov();
    let (mut b, ids) = banks(LayoutClass::PerRecord, 0, Reader::default(), g.clone());
    let t = b.stage(batch(vec![ids[0].clone()])).unwrap();
    let used = g.borrow().used.clone();
    assert_eq!(b.cancel(&t), Ok(CancelState::PublicationRevoked));
    assert!(!b.retire(&t).unwrap());
    assert_eq!(g.borrow().used, used);
    assert_eq!(b.acknowledge(&t), Err(Error::Busy));
    assert!(matches!(
        publish_bank(&mut b, &t, epochs()),
        Err(Error::Cancelled)
    ));
    finish(&mut b, &t);
    assert_eq!(g.borrow().used, TierBudget::zero(2));
    assert_eq!(b.retire(&t), Err(Error::UnknownTicket));
}
#[test]
fn zero_small_full_cache_preserves_exact_bytes_and_scales() {
    for capacity in [0, 20, 40] {
        let g = gov();
        let (mut b, ids) = banks(
            LayoutClass::PerRecord,
            capacity,
            Reader::default(),
            g.clone(),
        );
        for _ in 0..5 {
            let order = vec![ids[1].clone(), ids[0].clone(), ids[1].clone()];
            let t = b.stage(batch(order.clone())).unwrap();
            let leases = publish_bank(&mut b, &t, epochs()).unwrap();
            assert_eq!(
                leases.iter().map(|r| r.id().clone()).collect::<Vec<_>>(),
                order
            );
            assert!(!b.completion(&t).unwrap().consumer_fenced); // HOST only
            for r in &leases {
                assert_eq!(&*r.resource::<Vec<u8>>().unwrap(), &expected(r.layout()));
            }
            assert_eq!(b.release(&leases[0]), Err(Error::Busy));
            assert_eq!(b.cancel(&t), Ok(CancelState::AlreadyPublished));
            finish(&mut b, &t);
            b.collect_evicted().unwrap();
            assert!(b.cache_bytes() <= capacity);
        }
        assert_eq!(
            b.reader().calls.len(),
            match capacity {
                0 => 15,
                20 => 11,
                _ => 3,
            }
        );
        for id in ids.iter().take(2) {
            b.evict_cached(id).unwrap();
        }
        b.collect_evicted().unwrap();
        assert_eq!(g.borrow().used, TierBudget::zero(2));
    }
}
#[test]
fn original_masks_foreign_catalog_and_wrong_domain_refuse_before_reads() {
    let g = gov();
    let (mut b, ids) = banks(LayoutClass::PerRecord, 0, Reader::default(), g.clone());
    assert_eq!(b.stage(batch(vec![ids[3].clone()])), Err(Error::MaskedId));
    let mut unknown = ids[0].clone();
    unknown.tensor.artifact = [99; 32];
    assert_eq!(b.stage(batch(vec![unknown])), Err(Error::NotFound));
    let mut row = ids[0].clone();
    row.record = RecordId::Row(9);
    assert_eq!(b.stage(batch(vec![row])), Err(Error::InvalidLayout));
    assert!(b.reader().calls.is_empty());
    assert_eq!(g.borrow().used, TierBudget::zero(2));
}
#[test]
fn all_three_epochs_cross_service_and_publish_replay_refuse() {
    let (mut b, ids) = banks(LayoutClass::Uniform, 0, Reader::default(), gov());
    let (mut other, _) = banks(LayoutClass::Uniform, 0, Reader::default(), gov());
    let t = b.stage(batch(vec![ids[0].clone()])).unwrap();
    assert_eq!(other.cancel(&t), Err(Error::UnknownTicket));
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
            publish_bank(&mut b, &t, e),
            Err(Error::StaleEpoch)
        ));
    }
    let l = publish_bank(&mut b, &t, epochs()).unwrap();
    assert!(matches!(
        publish_bank(&mut b, &t, epochs()),
        Err(Error::Busy)
    ));
    finish(&mut b, &t);
    b.release(&l[0]).unwrap();
}
#[test]
fn uniform_proof_checks_actual_source_even_homogeneous_subset() {
    for class in [LayoutClass::Uniform, LayoutClass::PerRecord] {
        let (mut b, ids) = banks(class, 0, Reader::default(), gov());
        let t = b.stage(batch(vec![ids[0].clone()])).unwrap();
        let leases = publish_bank(&mut b, &t, epochs()).unwrap();
        let result = UniformLease::try_new(leases);
        let leases = if class == LayoutClass::Uniform {
            let p = result.unwrap();
            with_uniform_experts(&p, |p| assert_eq!(p.records().len(), 1));
            p.into_records()
        } else {
            let rejected = result.unwrap_err();
            assert_eq!(rejected.error, Error::MixedLayout);
            rejected.op
        };
        finish(&mut b, &t);
        b.release(&leases[0]).unwrap();
    }
}
#[test]
fn shared_budget_atomic_refusal_and_tombstone_bound() {
    let g = gov();
    let (mut b, ids) = banks(LayoutClass::PerRecord, 0, Reader::default(), g.clone());
    let held = g
        .borrow_mut()
        .reserve(&request(749_999, Priority::Demand))
        .unwrap();
    assert_eq!(
        b.stage(batch(vec![ids[0].clone(), ids[1].clone()])),
        Err(Error::Capacity)
    );
    assert_eq!(g.borrow().used.pageable, 749_999);
    assert!(b.reader().calls.is_empty());
    g.borrow_mut().release(&held).unwrap();
    let mut tickets = vec![];
    for _ in 0..4 {
        tickets.push(b.stage(batch(vec![ids[0].clone()])).unwrap());
    }
    assert_eq!(b.stage(batch(vec![ids[0].clone()])), Err(Error::Capacity));
    for t in tickets {
        b.cancel(&t).unwrap();
        finish(&mut b, &t);
    }
    assert_eq!(g.borrow().used, TierBudget::zero(2));
}
#[test]
fn short_and_corrupt_reads_never_publish_partial_batch() {
    for reader in [
        Reader {
            fail: Some(2),
            ..Reader::default()
        },
        Reader {
            corrupt: true,
            ..Reader::default()
        },
    ] {
        let g = gov();
        let (mut b, ids) = banks(LayoutClass::PerRecord, 40, reader, g.clone());
        let t = b
            .stage(batch(vec![ids[0].clone(), ids[1].clone()]))
            .unwrap();
        assert!(publish_bank(&mut b, &t, epochs()).is_err());
        assert_eq!(b.cache_bytes(), 0);
        assert!(!b.retire(&t).unwrap());
        b.cancel(&t).unwrap();
        finish(&mut b, &t);
        assert_eq!(g.borrow().used, TierBudget::zero(2));
    }
}
#[test]
fn borrowed_view_and_alias_survive_busy_retirement() {
    let g = gov();
    let (mut b, ids) = banks(LayoutClass::Uniform, 20, Reader::default(), g.clone());
    let t = b.stage(batch(vec![ids[0].clone()])).unwrap();
    let leases = publish_bank(&mut b, &t, epochs()).unwrap();
    let alias = leases[0].clone();
    finish(&mut b, &t);
    {
        let view = alias.resource::<Vec<u8>>().unwrap();
        assert_eq!(b.release(&alias), Err(Error::Busy));
        assert_eq!(view.len(), 20);
    }
    b.release(&alias).unwrap();
    assert!(matches!(
        leases[0].resource::<Vec<u8>>(),
        Err(Error::AlreadyReleased)
    ));
    assert_eq!(b.release(&alias), Err(Error::AlreadyReleased));
    assert_eq!(g.borrow().used, TierBudget::zero(2));
}
#[test]
fn overlapping_cache_hit_tickets_hold_same_allocation_until_all_retire() {
    let g = gov();
    let (mut b, ids) = banks(LayoutClass::Uniform, 20, Reader::default(), g.clone());
    let t = b.stage(batch(vec![ids[0].clone()])).unwrap();
    let a = publish_bank(&mut b, &t, epochs()).unwrap();
    let u = b.stage(batch(vec![ids[0].clone()])).unwrap();
    let c = publish_bank(&mut b, &u, epochs()).unwrap();
    assert_eq!(a[0].charge().id(), c[0].charge().id());
    finish(&mut b, &t);
    assert_eq!(b.release(&a[0]), Err(Error::Busy));
    finish(&mut b, &u);
    b.release(&a[0]).unwrap();
    assert_eq!(g.borrow().used, TierBudget::zero(2));
}
#[test]
fn predictions_are_bounded_mask_checked_and_not_demand_heat() {
    struct Predictor(Vec<BankId>);
    impl PrefetchHook<ExpertDomain> for Predictor {
        type Context = ();
        fn predict(&self, _: &(), _: usize) -> Vec<BankId> {
            self.0.clone()
        }
    }
    let (c, ids) = catalog(LayoutClass::Uniform);
    assert!(matches!(
        prefetch_batch(
            &c,
            &Predictor(vec![ids[3].clone()]),
            &(),
            1,
            epochs(),
            request(0, Priority::Demand)
        ),
        Err(Error::MaskedId)
    ));
    assert!(matches!(
        prefetch_batch(
            &c,
            &Predictor(ids.clone()),
            &(),
            1,
            epochs(),
            request(0, Priority::Demand)
        ),
        Err(Error::Capacity)
    ));
    let p = prefetch_batch(
        &c,
        &Predictor(vec![ids[1].clone()]),
        &(),
        1,
        epochs(),
        request(0, Priority::Demand),
    )
    .unwrap()
    .unwrap();
    let (mut b, _) = banks(LayoutClass::Uniform, 20, Reader::default(), gov());
    let t = b.stage(batch(vec![ids[0].clone()])).unwrap();
    publish_bank(&mut b, &t, epochs()).unwrap();
    finish(&mut b, &t);
    let u = b.stage(p).unwrap();
    publish_bank(&mut b, &u, epochs()).unwrap();
    finish(&mut b, &u);
    assert!(b.resident(&ids[0]).unwrap().is_some());
    assert!(b.resident(&ids[1]).unwrap().is_none());
    b.collect_evicted().unwrap();
    b.evict_cached(&ids[0]).unwrap();
    b.collect_evicted().unwrap();
}
fn ple(encoding: PleEncoding) -> PleTable {
    PleTable {
        tensor: TensorId {
            name: "ple.table".into(),
            ..tensor()
        },
        rows: 1_000_000,
        head_dim: if encoding == PleEncoding::F32 {
            66
        } else {
            132
        },
        encoding,
    }
}
fn row_catalog(table: &PleTable, rows: &[u64]) -> Catalog {
    let entries = rows
        .iter()
        .map(|&n| (table.id(n).unwrap(), Some(record(table.layout(n).unwrap()))))
        .collect::<BTreeMap<_, _>>();
    Catalog::new(LayoutClass::PerRecord, entries.into_iter().collect()).unwrap()
}
fn rows(
    table: &PleTable,
    ns: &[u64],
    g: SharedBudget,
    policy: CoalescingPolicy,
) -> BoundedRowService<Heat, Reader> {
    BoundedRowService(
        BankService::new(
            row_catalog(table, ns),
            g,
            Heat::default(),
            Reader::default(),
            policy,
            limits(0),
        )
        .unwrap(),
    )
}
fn row_batch(table: &PleTable, ns: &[u64]) -> RowBatch {
    RowBatch {
        ids: ns.iter().map(|&n| table.id(n).unwrap()).collect(),
        epochs: epochs(),
        request: request(0, Priority::Demand),
    }
}
#[test]
fn frozen_row_order_schedule_runs_on_implementation() {
    let table = ple(PleEncoding::F32);
    let g = gov();
    let mut r = rows(&table, &[5, 2, 9], g.clone(), CoalescingPolicy::default());
    let leases = conformance::rows_order(&mut DrivenRows(&mut r), row_batch(&table, &[5, 2, 5, 9]));
    assert_eq!(
        leases.records[0].charge().id(),
        leases.records[2].charge().id()
    );
    for l in &leases.records {
        assert_eq!(&*l.resource::<Vec<u8>>().unwrap(), &expected(l.layout()));
    }
    for t in r.0.tickets() {
        finish(&mut r.0, &t);
    }
    r.release(&leases).unwrap();
    assert_eq!(g.borrow().used, TierBudget::zero(2));
}
#[test]
fn rows_straddles_duplicates_pool_smaller_than_batch_and_retirement() {
    let table = ple(PleEncoding::F32);
    let ns: Vec<_> = (0..48).rev().chain([0, 47, 16]).collect();
    for granularity in [512, 4096, 16384] {
        let g = gov();
        let mut r = rows(
            &table,
            &ns,
            g.clone(),
            CoalescingPolicy {
                granularity,
                slot_bytes: granularity,
            },
        );
        let t = r.gather(row_batch(&table, &ns)).unwrap();
        let l = publish_rows(&mut r, &t, epochs()).unwrap();
        assert_eq!(
            r.0.plan(&t).unwrap().io_bytes,
            12672u64.div_ceil(granularity) * granularity
        );
        assert_eq!(
            r.0.plan(&t).unwrap().straddling_segments,
            match granularity {
                512 => 24,
                4096 => 3,
                _ => 0,
            }
        );
        for lease in &l.records {
            assert_eq!(
                &*lease.resource::<Vec<u8>>().unwrap(),
                &expected(lease.layout())
            );
        }
        assert!(
            r.0.reader()
                .calls
                .iter()
                .all(|(_, _, n)| *n as u64 <= granularity)
        );
        assert_eq!(r.release(&l), Err(Error::Busy));
        finish(&mut r.0, &t);
        r.release(&l).unwrap();
        assert_eq!(g.borrow().used, TierBudget::zero(2));
    }
}
#[test]
fn sparse_amplification_and_default_policy_are_explicit_arithmetic_not_io_benchmarks() {
    let records: Vec<_> = (0..48)
        .map(|n| {
            let mut l = ple(PleEncoding::F32).layout(n).unwrap();
            l.segments[0].offset = n * 32768;
            (ple(PleEncoding::F32).id(n).unwrap(), record(l))
        })
        .collect();
    assert_eq!(CoalescingPolicy::default().granularity, 512);
    for g in [512, 4096, 16384] {
        let p = plan_reads(
            &records,
            12672,
            &Reader::default(),
            CoalescingPolicy {
                granularity: g,
                slot_bytes: g,
            },
        )
        .unwrap();
        assert_eq!(p.io_bytes, 48 * g);
        assert_eq!(p.unique_useful_bytes, 12672);
        println!(
            "sparse264x48 granularity={g} submitted={} useful={} amplification={}",
            p.io_bytes,
            p.unique_useful_bytes,
            p.amplification().unwrap()
        );
    }
}
#[test]
fn discontiguous_row_scale_planes_are_required_and_exact() {
    let mut l = layout(9, 2, 264);
    let table = ple(PleEncoding::F32);
    let id = BankId {
        record: RecordId::Row(9),
        tensor: table.tensor.clone(),
        layout: l.identity().unwrap(),
        version: 1,
    };
    let c = Catalog::new(
        LayoutClass::PerRecord,
        vec![(id.clone(), Some(record(l.clone())))],
    )
    .unwrap();
    let mut r = BoundedRowService(
        BankService::new(
            c,
            gov(),
            Heat::default(),
            Reader::default(),
            CoalescingPolicy::default(),
            limits(0),
        )
        .unwrap(),
    );
    let t = r
        .gather(RowBatch {
            ids: vec![id],
            epochs: epochs(),
            request: request(0, Priority::Demand),
        })
        .unwrap();
    let lease = publish_rows(&mut r, &t, epochs()).unwrap();
    assert_eq!(lease.records[0].resource::<Vec<u8>>().unwrap().len(), 268);
    finish(&mut r.0, &t);
    r.release(&lease).unwrap();
    l.segments.pop();
    assert_eq!(l.validate(), Err(Error::Incomplete));
}
#[test]
fn bounds_wrong_ids_overflow_and_device_request_fail_closed() {
    let table = ple(PleEncoding::F32);
    assert_eq!(table.id(table.rows), Err(Error::NotFound));
    assert!(matches!(
        table.last_chunk(&[-1], 1, 1, 1, epochs(), request(0, Priority::Demand)),
        Err(Error::NotFound)
    ));
    assert!(matches!(
        table.last_chunk(&[0], 1, 2, 1, epochs(), request(0, Priority::Demand)),
        Err(Error::InvalidLayout)
    ));
    let huge = PleTable {
        head_dim: u64::MAX,
        ..table
    };
    assert_eq!(huge.layout(0), Err(Error::Overflow));
    let (mut b, ids) = banks(LayoutClass::Uniform, 0, Reader::default(), gov());
    let mut q = batch(vec![ids[0].clone()]);
    q.request.bytes.device[0] = 20;
    assert_eq!(b.stage(q), Err(Error::Unsupported));
}

#[derive(Clone)]
struct HostFixture {
    uniform: bool,
    metadata: Vec<ExpertMetadata>,
}
impl HostExpsView for HostFixture {
    fn is_uniform_layout(&self) -> bool {
        self.uniform
    }
    fn n_expert(&self) -> usize {
        self.metadata.len()
    }
    fn max_expert_bytes(&self) -> u64 {
        self.metadata.iter().map(|m| m.len).max().unwrap_or(0)
    }
    fn expert_layout(&self, n: usize) -> Result<ExpertMetadata> {
        self.metadata.get(n).copied().ok_or(Error::NotFound)
    }
}
#[test]
fn host_exps_uniform_mixed_original_mask_and_split_offsets() {
    for uniform in [true, false] {
        for split in [true, false] {
            let fixture = HostFixture {
                uniform,
                metadata: vec![
                    ExpertMetadata {
                        offset: 0,
                        len: 16,
                        qtype: 2,
                        row_bytes: 8,
                    },
                    ExpertMetadata {
                        offset: 16,
                        len: 16,
                        qtype: 2,
                        row_bytes: 8,
                    },
                    ExpertMetadata {
                        offset: 64,
                        len: if uniform { 16 } else { 24 },
                        qtype: if uniform { 2 } else { 3 },
                        row_bytes: 8,
                    },
                ],
            };
            let sources = [0, 2]
                .into_iter()
                .map(|n| {
                    (
                        n,
                        ExpertSource {
                            tensor: if split {
                                TensorId {
                                    name: format!("expert.{n}.weight"),
                                    ..tensor()
                                }
                            } else {
                                tensor()
                            },
                            split,
                            scales: vec![],
                            checksums: vec![[1; 32]],
                        },
                    )
                })
                .collect();
            let mapped = host_exps_catalog(
                &fixture,
                tensor(),
                2,
                Projection::Up,
                &[true, false, true],
                &sources,
            )
            .unwrap();
            assert_eq!(mapped.ids.len(), 3);
            assert_eq!(mapped.max_expert_bytes, if uniform { 16 } else { 24 });
            assert!(matches!(
                mapped.ids[2].record,
                RecordId::Expert { original_id: 2, .. }
            ));
            assert_eq!(
                mapped
                    .catalog
                    .record(&mapped.ids[2])
                    .unwrap()
                    .layout
                    .segments[0]
                    .offset,
                if split { 0 } else { 64 }
            );
            assert!(matches!(
                mapped.catalog.record(&mapped.ids[1]),
                Err(Error::MaskedId)
            ));
            let mut missing = sources.clone();
            missing.remove(&2);
            assert!(matches!(
                host_exps_catalog(
                    &fixture,
                    tensor(),
                    2,
                    Projection::Up,
                    &[true, false, true],
                    &missing
                ),
                Err(Error::Incomplete)
            ));
        }
    }
}
#[test]
fn missing_or_extra_checksums_and_false_uniform_declarations_refuse() {
    let (c, ids) = catalog(LayoutClass::PerRecord);
    let mut r = c.record(&ids[0]).unwrap().clone();
    r.checksums.pop();
    assert!(matches!(
        Catalog::new(LayoutClass::Uniform, vec![(ids[0].clone(), Some(r))]),
        Err(Error::InvalidLayout)
    ));
    let entries = ids[..2]
        .iter()
        .map(|id| (id.clone(), Some(c.record(id).unwrap().clone())))
        .collect();
    assert!(matches!(
        Catalog::new(LayoutClass::Uniform, entries),
        Err(Error::MixedLayout)
    ));
}

#[test]
fn synthetic_ple_trace_preserves_chunk_rewind_eos_and_native_expansion_bits() {
    let trace: serde_json::Value = serde_json::from_str(include_str!(
        "../../../../research/spill-c-20260919/fixtures/ple-ngram-synthetic.json"
    ))
    .unwrap();
    assert_eq!(trace["synthetic"], true);
    let (mut cached, mut history, mut last_eos) = (vec![], vec![], -1);
    for step in trace["steps"].as_array().unwrap() {
        let tokens: Vec<u32> = step["tokens"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        let ids: Vec<i64> = step["ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_i64().unwrap())
            .collect();
        assert_eq!(
            ple_oracle::host_ngram_ids(&tokens, &[3, 5, 7], &[17, 19], &[0, 17], 3, 1, 99),
            ids
        );
        ple_oracle::host_ngram_ids_cached(
            &mut cached,
            &mut history,
            &mut last_eos,
            &tokens,
            &[3, 5, 7],
            &[17, 19],
            &[0, 17],
            3,
            1,
            99,
        );
        assert_eq!(cached, ids);
    }
    for encoding in [PleEncoding::F32, PleEncoding::Bf16] {
        let table = ple(encoding);
        for step in trace["steps"].as_array().unwrap() {
            let all: Vec<i64> = step["ids"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_i64().unwrap())
                .collect();
            let tokens = step["tokens"].as_array().unwrap().len();
            let chunk = step["chunk"].as_u64().unwrap() as usize;
            let b = table
                .last_chunk(
                    &all,
                    tokens,
                    chunk,
                    2,
                    epochs(),
                    request(0, Priority::Demand),
                )
                .unwrap();
            let ns: Vec<_> = b
                .ids
                .iter()
                .map(|id| match id.record {
                    RecordId::Row(n) => n,
                    _ => panic!(),
                })
                .collect();
            let g = gov();
            let mut r = rows(&table, &ns, g.clone(), CoalescingPolicy::default());
            let t = r.gather(b).unwrap();
            let l = publish_rows(&mut r, &t, epochs()).unwrap();
            for lease in &l.records {
                let raw = expected(lease.layout());
                let floats = table.expand_row(lease).unwrap();
                let bits: Vec<_> = match encoding {
                    PleEncoding::F32 => raw
                        .chunks_exact(4)
                        .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
                        .collect(),
                    PleEncoding::Bf16 => raw
                        .chunks_exact(2)
                        .map(|b| u32::from(u16::from_le_bytes([b[0], b[1]])) << 16)
                        .collect(),
                };
                assert_eq!(floats.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), bits);
            }
            finish(&mut r.0, &t);
            r.release(&l).unwrap();
            assert_eq!(g.borrow().used, TierBudget::zero(2));
        }
    }
}

#[test]
fn refused_publication_retains_owned_charge_until_explicit_retirement() {
    struct RefusePin(Rc<RefCell<Governor>>);
    impl BudgetGovernor for RefusePin {
        fn reserve(&mut self, r: &BudgetRequest) -> Result<ChargedLease> {
            let l = self.0.borrow_mut().reserve(r)?;
            if r.bytes.pageable > 0 && r.bytes.inflight == 0 {
                self.0.borrow_mut().mark(&l, ChargeState::Retired)?;
            }
            Ok(l)
        }
        fn used(&self) -> TierBudget {
            self.0.borrow().used.clone()
        }
        fn mark(&mut self, l: &ChargedLease, s: ChargeState) -> Result<()> {
            self.0.borrow_mut().mark(l, s)
        }
        fn release(&mut self, l: &ChargedLease) -> Result<()> {
            self.0.borrow_mut().release(l)
        }
    }
    let g = gov();
    let injected: SharedBudget = Rc::new(RefCell::new(RefusePin(g.clone())));
    let (mut b, ids) = banks(LayoutClass::Uniform, 0, Reader::default(), injected);
    let t = b.stage(batch(vec![ids[0].clone()])).unwrap();
    let before = g.borrow().used.clone();
    assert!(matches!(
        publish_bank(&mut b, &t, epochs()),
        Err(Error::Busy)
    ));
    assert_eq!(g.borrow().used, before); // rejected.op wasn't dropped/lost
    b.cancel(&t).unwrap();
    finish(&mut b, &t);
    assert_eq!(g.borrow().used, TierBudget::zero(2));
}

#[test]
fn row_release_busy_preserves_retry_after_partial_alias_retirement() {
    let table = ple(PleEncoding::Bf16);
    let g = gov();
    let mut r = rows(&table, &[2, 5], g.clone(), CoalescingPolicy::default());
    let t = r.gather(row_batch(&table, &[2, 5])).unwrap();
    let l = publish_rows(&mut r, &t, epochs()).unwrap();
    finish(&mut r.0, &t);
    {
        // Original-id order issues charge 2 after charge 1; release can retire the
        // first record before encountering this borrow. Retry must still work.
        let last = l.records[1].resource::<Vec<u8>>().unwrap();
        assert_eq!(r.release(&l), Err(Error::Busy));
        assert_eq!(last.len(), 264);
    }
    r.release(&l).unwrap();
    assert_eq!(g.borrow().used, TierBudget::zero(2));
    assert_eq!(r.release(&l), Err(Error::AlreadyReleased));
}

#[test]
fn actual_q2_q3_nvfp4_codes_are_preserved_without_kernel_promotion() {
    let native = include_str!("../../../memra-engine/src/lib.rs");
    for (name, value) in [("QT_Q2_K", 13), ("QT_Q3_K", 4), ("QT_NVFP4", 7)] {
        assert!(native.contains(&format!("pub const {name}: i32 = {value};")));
    }
    let f = HostFixture {
        uniform: false,
        metadata: vec![
            ExpertMetadata {
                offset: 0,
                len: 16,
                qtype: 13,
                row_bytes: 8,
            },
            ExpertMetadata {
                offset: 16,
                len: 24,
                qtype: 4,
                row_bytes: 8,
            },
            ExpertMetadata {
                offset: 40,
                len: 32,
                qtype: 7,
                row_bytes: 8,
            },
        ],
    };
    let sources = (0..3)
        .map(|n| {
            (
                n,
                ExpertSource {
                    tensor: tensor(),
                    split: false,
                    scales: vec![],
                    checksums: vec![[1; 32]],
                },
            )
        })
        .collect();
    let mapping =
        host_exps_catalog(&f, tensor(), 0, Projection::Down, &[true; 3], &sources).unwrap();
    for (id, (q, n)) in mapping.ids.iter().zip([(13i32, 16), (4, 24), (7, 32)]) {
        let r = mapping.catalog.record(id).unwrap();
        assert_eq!(
            r.layout.segments[0].encoding.program,
            digest("host-exps-qtype-v1", &q.to_le_bytes())
        );
        assert_eq!(r.layout.segments[0].valid_bytes, n);
    }
}

#[test]
fn pinned_ple_test_oracle_matches_native_function_bodies() {
    fn body<'a>(source: &'a str, name: &str) -> &'a str {
        let start = source.find(&format!("fn {name}(")).unwrap();
        let brace = start + source[start..].find('{').unwrap();
        let mut depth = 0;
        for (offset, ch) in source[brace..].char_indices() {
            if ch == '{' {
                depth += 1;
            } else if ch == '}' {
                depth -= 1;
            }
            if depth == 0 {
                return &source[start..brace + offset + 1];
            }
        }
        panic!("unclosed function")
    }
    let native = include_str!("../../../memra-engine/src/qwen4exp_gpu.rs");
    let fixture = include_str!("ple_oracle.rs");
    for name in [
        "host_ngram_ids",
        "host_ngram_ids_cached",
        "shift_right_ignore_eos",
    ] {
        let normalize = |s: &str| s.chars().filter(|c| !c.is_whitespace()).collect::<String>();
        assert_eq!(
            normalize(body(native, name)),
            normalize(body(fixture, name)),
            "native oracle drift: {name}"
        );
    }
}

#[test]
fn read_planning_clamps_tail_but_refuses_truncated_payload_and_invalid_granularity() {
    struct Tail;
    impl ExactReader for Tail {
        fn storage_bytes(&self, _: &TensorId) -> Result<u64> {
            Ok(264)
        }
        fn read_exact(&mut self, _: &TensorId, _: u64, _: &mut [u8]) -> Result<()> {
            panic!("planning only")
        }
    }
    let table = ple(PleEncoding::F32);
    let entries = vec![(table.id(0).unwrap(), record(table.layout(0).unwrap()))];
    let plan = plan_reads(&entries, 264, &Tail, CoalescingPolicy::default()).unwrap();
    assert_eq!(plan.extents[0].len, 264);
    assert_eq!(plan.io_bytes, 264);
    let mut truncated = entries.clone();
    truncated[0].1.layout.segments[0].offset = 1;
    assert_eq!(
        plan_reads(&truncated, 264, &Tail, CoalescingPolicy::default()),
        Err(Error::InvalidLayout)
    );
    assert_eq!(
        plan_reads(
            &entries,
            264,
            &Reader::default(),
            CoalescingPolicy {
                granularity: 3,
                slot_bytes: 4096
            }
        ),
        Err(Error::InvalidLayout)
    );
    let mut overflow = entries.clone();
    overflow[0].1.layout.segments[0].offset = u64::MAX;
    assert_eq!(
        plan_reads(
            &overflow,
            264,
            &Reader::default(),
            CoalescingPolicy::default()
        ),
        Err(Error::Overflow)
    );
}

#[test]
fn failed_transfer_completion_enumerates_all_records_and_segments() {
    let (mut b, ids) = banks(
        LayoutClass::PerRecord,
        0,
        Reader {
            fail: Some(2),
            ..Reader::default()
        },
        gov(),
    );
    let t = b
        .stage(batch(vec![ids[0].clone(), ids[1].clone()]))
        .unwrap();
    drain(&mut b, &t);
    let c = b.completion(&t).unwrap();
    assert_eq!(c.items.len(), 2);
    for (i, item) in c.items.iter().enumerate() {
        assert_eq!(item.item, i as u32);
        assert!(item.accepted);
        assert_eq!(item.segments.len(), 2);
        for s in &item.segments {
            assert_eq!(s.status, ItemStatus::Failed);
            assert!(s.error.is_some());
            assert_eq!(s.checksum, None);
        }
    }
    assert!(publish_bank(&mut b, &t, epochs()).is_err());
    b.cancel(&t).unwrap();
    finish(&mut b, &t);
}

#[test]
fn alternate_layout_cannot_resurrect_masked_original_identity() {
    let l = layout(9, 13, 16);
    let id = bank_id(9, &l);
    let alternative = layout(9, 4, 24);
    let other = bank_id(9, &alternative);
    assert_ne!(id.layout, other.layout);
    for first in [None, Some(record(l.clone()))] {
        assert!(matches!(
            Catalog::new(
                LayoutClass::PerRecord,
                vec![
                    (id.clone(), first),
                    (other.clone(), Some(record(alternative.clone())))
                ]
            ),
            Err(Error::Conflict)
        ));
    }
}

fn drain<D: BankDomain, H: Hotness<D>, R: ExactReader>(
    b: &mut BankService<D, H, R>,
    t: &TransferTicket,
) {
    for _ in 0..10000 {
        if b.progress(t).unwrap() {
            return;
        }
    }
    panic!("bounded CPU pump did not terminate");
}
fn publish_bank<D: BankDomain, H: Hotness<D>, R: ExactReader>(
    b: &mut BankService<D, H, R>,
    t: &TransferTicket,
    e: Epochs,
) -> Result<Vec<BankLease>> {
    drain(b, t);
    b.publish(t, e)
}
fn publish_rows<H: Hotness<RowDomain>, R: ExactReader>(
    r: &mut BoundedRowService<H, R>,
    t: &TransferTicket,
    e: Epochs,
) -> Result<RowLease> {
    drain(&mut r.0, t);
    r.publish(t, e)
}
// The frozen immediate-publication schedule needs an explicit test-only driver.
// Production RowService::publish never pumps I/O implicitly.
struct DrivenRows<'a, H: Hotness<RowDomain>, R: ExactReader>(&'a mut BoundedRowService<H, R>);
impl<H: Hotness<RowDomain>, R: ExactReader> RowService for DrivenRows<'_, H, R> {
    fn gather(&mut self, b: RowBatch) -> Result<TransferTicket> {
        self.0.gather(b)
    }
    fn publish(&mut self, t: &TransferTicket, e: Epochs) -> Result<RowLease> {
        publish_rows(self.0, t, e)
    }
    fn cancel(&mut self, t: &TransferTicket) -> Result<CancelState> {
        self.0.cancel(t)
    }
    fn retire(&mut self, t: &TransferTicket) -> Result<bool> {
        self.0.retire(t)
    }
    fn release(&mut self, l: &RowLease) -> Result<()> {
        self.0.release(l)
    }
}

mod integration;

// v1.1 completion/observation hooks drive the day-3 queued host reader.
// Keep stage/gather enqueue-only and publication/retirement separate from progress.
#[test]
fn revision_v11_bank_cancel_identity_uniform() {
    for class in [LayoutClass::Uniform, LayoutClass::PerRecord] {
        let g = gov();
        let (mut b, ids) = banks(class, 0, Reader::default(), g.clone());
        conformance::bank_identity(&mut b, batch(vec![ids[0].clone()]));
        let t = conformance::bank_complete_cancel(&mut b, batch(vec![ids[0].clone()]), |b, t| {
            drain(b, t);
            b.completion(t).unwrap().clone()
        });
        finish(&mut b, &t);
        // A homogeneous one-record subset of PerRecord is still not Uniform.
        conformance::bank_uniform(
            &mut b,
            batch(vec![ids[0].clone()]),
            class,
            |b, t| {
                drain(b, t);
                assert!(b.completion(t).unwrap().producer_done);
            },
            |b, t| {
                b.finish_host_use(t).unwrap();
            },
        );
        for t in b.tickets() {
            b.acknowledge(&t).unwrap();
        }
        assert_eq!(g.borrow().used(), TierBudget::zero(2));
    }
}
#[test]
fn revision_v11_rows_complete_cancel() {
    let table = ple(PleEncoding::F32);
    let g = gov();
    let mut r = rows(&table, &[5, 2, 9], g.clone(), CoalescingPolicy::default());
    let t = conformance::rows_complete_cancel(&mut r, row_batch(&table, &[5, 2, 5, 9]), |r, t| {
        drain(&mut r.0, t);
        r.0.completion(t).unwrap().clone()
    });
    finish(&mut r.0, &t);
    assert_eq!(g.borrow().used(), TierBudget::zero(2));
}
#[test]
fn revision_v11_rows_bytes_dedup_bounded_straddles() {
    let table = ple(PleEncoding::F32);
    let ns: Vec<_> = (0..48).rev().chain([0, 47, 16]).collect();
    let g = gov();
    let mut r = rows(
        &table,
        &ns,
        g.clone(),
        CoalescingPolicy {
            granularity: 4096,
            slot_bytes: 4096,
        },
    );
    let expected: Vec<_> = ns
        .iter()
        .map(|&n| vec![expected(&table.layout(n).unwrap())])
        .collect();
    let t = conformance::rows_bytes(
        &mut r,
        row_batch(&table, &ns),
        &expected,
        |l| vec![l.resource::<Vec<u8>>().unwrap().clone()],
        |r, t| {
            drain(&mut r.0, t);
            assert_eq!(r.0.plan(t).unwrap().straddling_segments, 3);
            assert_eq!(r.0.reader().calls.len(), 4);
        },
        |r, t| {
            r.0.finish_host_use(t).unwrap();
        },
    );
    assert!(r.0.reader().calls.iter().all(|(_, _, n)| *n <= 4096));
    r.0.acknowledge(&t).unwrap();
    assert_eq!(g.borrow().used(), TierBudget::zero(2));
}

#[test]
fn revision_v11_corrupt_sibling_and_row_namespace() {
    let g = gov();
    let (mut b, ids) = banks(
        LayoutClass::PerRecord,
        0,
        Reader {
            corrupt: true,
            ..Reader::default()
        },
        g.clone(),
    );
    let t = conformance::bank_failed_sibling(&mut b, batch(ids[..3].to_vec()), |b, t| {
        drain(b, t);
        b.completion(t).unwrap().clone()
    });
    finish(&mut b, &t);
    assert_eq!(g.borrow().used(), TierBudget::zero(2));
    let table = ple(PleEncoding::F32);
    let ns: Vec<_> = (0..48).collect();
    let mut r = BoundedRowService(
        BankService::new(
            row_catalog(&table, &ns),
            g.clone(),
            Heat::default(),
            Reader {
                fail: Some(2),
                ..Reader::default()
            },
            CoalescingPolicy {
                slot_bytes: 4096,
                granularity: 4096,
            },
            limits(0),
        )
        .unwrap(),
    );
    conformance::rows_identity(&mut r, row_batch(&table, &ns));
    let t = conformance::rows_failed_sibling(&mut r, row_batch(&table, &ns), |r, t| {
        drain(&mut r.0, t);
        r.0.completion(t).unwrap().clone()
    });
    assert_eq!(r.0.reader().calls.len(), 2);
    assert!(r.0.reader().calls.iter().all(|(_, _, n)| *n <= 4096));
    finish(&mut r.0, &t);
    assert_eq!(g.borrow().used(), TierBudget::zero(2));
}

mod day4;

mod ple_rows_native;

mod device_rows;

#[test]
fn expert_dispatch_slru_preserves_ids_bytes_hits_and_refuses_unknown() {
    let g = gov();
    let mut entries = vec![];
    let mut ids = BTreeMap::new();
    for n in [9, 47] {
        let mut l = layout(n as u64, 4, 16);
        l.segments.truncate(1);
        l.requirements.truncate(1);
        let id = bank_id(n, &l);
        entries.push((id.clone(), Some(record(l))));
        ids.insert((2, 0, n as u16), id);
    }
    let bank = BankService::new(
        Catalog::new(LayoutClass::Uniform, entries).unwrap(),
        g.clone(),
        Heat::default(),
        Reader::default(),
        CoalescingPolicy {
            granularity: 1,
            slot_bytes: 32,
        },
        limits(32),
    )
    .unwrap();
    let mut req = request(bank.slru_metadata_bytes(2).unwrap(), Priority::Demand);
    let metadata = g.borrow_mut().reserve(&req).unwrap();
    let bank = bank
        .with_slru(SlruPolicy::new(&[(16, 2)]).unwrap(), &metadata)
        .unwrap();
    req.bytes = TierBudget::zero(2);
    let mut dispatch = SlruExpertDispatch::new(bank, ids.clone(), req, epochs()).unwrap();
    assert_eq!(dispatch.validate((2, 0, 21), 16), Err(Error::NotFound));
    assert_eq!(dispatch.validate((2, 0, 9), 15), Err(Error::InvalidLayout));
    for n in [9, 47, 9, 47] {
        let demand = dispatch.demand((2, 0, n), 16).unwrap();
        assert_eq!(demand.lease.id(), &ids[&(2, 0, n)]);
        assert_eq!(
            &*demand.lease.resource::<Vec<u8>>().unwrap(),
            &expected(demand.lease.layout())
        );
        dispatch.finish(demand).unwrap();
    }
    assert_eq!(dispatch.bank().reader().calls.len(), 2);
    let mut bank = dispatch.into_bank();
    for id in ids.values() {
        bank.evict_cached(id).unwrap();
    }
    bank.collect_evicted().unwrap();
    drop(bank);
    g.borrow_mut().release(&metadata).unwrap();
    assert_eq!(g.borrow().used, TierBudget::zero(2));
}

mod owner_proxy;

mod day8;

mod day10;
