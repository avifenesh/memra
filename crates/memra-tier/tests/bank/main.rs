use memra_bank_prototype::*;
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

#[derive(Default)]
struct BudgetState {
    used: Charge,
    peak: Charge,
}
struct Governor {
    limit: Charge,
    state: Arc<Mutex<BudgetState>>,
}
struct Permit {
    charge: Charge,
    state: Arc<Mutex<BudgetState>>,
}
impl BudgetPermit for Permit {}
impl Drop for Permit {
    fn drop(&mut self) {
        let mut s = self.state.lock().unwrap();
        s.used.host_bytes -= self.charge.host_bytes;
        s.used.staging_bytes -= self.charge.staging_bytes;
        s.used.inflight -= self.charge.inflight;
    }
}
impl BudgetGovernor for Governor {
    fn reserve(&self, charge: Charge) -> Result<Box<dyn BudgetPermit>> {
        let mut s = self.state.lock().unwrap();
        let used = Charge {
            host_bytes: s
                .used
                .host_bytes
                .checked_add(charge.host_bytes)
                .ok_or(BankError::Overflow)?,
            staging_bytes: s
                .used
                .staging_bytes
                .checked_add(charge.staging_bytes)
                .ok_or(BankError::Overflow)?,
            inflight: s
                .used
                .inflight
                .checked_add(charge.inflight)
                .ok_or(BankError::Overflow)?,
        };
        if used.host_bytes > self.limit.host_bytes
            || used.staging_bytes > self.limit.staging_bytes
            || used.inflight > self.limit.inflight
        {
            return Err(BankError::Backpressure);
        }
        s.used = used;
        s.peak.host_bytes = s.peak.host_bytes.max(used.host_bytes);
        s.peak.staging_bytes = s.peak.staging_bytes.max(used.staging_bytes);
        s.peak.inflight = s.peak.inflight.max(used.inflight);
        Ok(Box::new(Permit {
            charge,
            state: self.state.clone(),
        }))
    }
}
fn governor(host: u64, staging: u64, inflight: u32) -> Arc<Governor> {
    Arc::new(Governor {
        limit: Charge {
            host_bytes: host,
            staging_bytes: staging,
            inflight,
        },
        state: Arc::default(),
    })
}
#[derive(Default)]
struct Heat(BTreeMap<BankId, u64>);
impl Hotness<ExpertDomain> for Heat {
    fn demand(&mut self, id: &BankId) {
        *self.0.entry(id.clone()).or_default() += 1;
    }
    fn score(&self, id: &BankId) -> u64 {
        self.0.get(id).copied().unwrap_or(0)
    }
}
fn tensor() -> TensorId {
    TensorId {
        artifact: [7; 32],
        name: "layer.2.experts.gate.weight".into(),
    }
}
fn id(original_id: u32) -> BankId {
    BankId {
        tensor: tensor(),
        record: RecordId::Expert {
            layer: 2,
            original_id,
            projection: Projection::Gate,
        },
        layout: [3; 32],
    }
}
fn layout(original: u64, encoding: &str, len: u64) -> RecordLayout {
    RecordLayout {
        encoding: encoding.into(),
        row_bytes: len,
        segments: vec![
            Segment {
                tensor: tensor(),
                offset: original * 64,
                len,
                role: "payload".into(),
            },
            Segment {
                tensor: TensorId {
                    name: "layer.2.experts.gate.scale".into(),
                    ..tensor()
                },
                offset: original * 4,
                len: 4,
                role: "macro_scale".into(),
            },
        ],
    }
}
fn catalog(class: LayoutClass) -> Catalog {
    let e = if class == LayoutClass::Uniform {
        "NVFP4"
    } else {
        "Q2_K"
    };
    Catalog::new(
        class,
        vec![
            (id(9), Some(layout(9, "NVFP4", 16))),
            (id(47), Some(layout(47, e, 16))),
            (id(21), None),
        ],
    )
    .unwrap()
}
fn batch(ids: &[u32]) -> BankBatch {
    catalog(LayoutClass::PerRecord)
        .batch(ids.iter().map(|i| id(*i)).collect())
        .unwrap()
}
struct Reader {
    calls: Vec<(TensorId, u64, usize)>,
    fail_at: Option<usize>,
}
impl Reader {
    fn new() -> Self {
        Self {
            calls: vec![],
            fail_at: None,
        }
    }
}
fn byte(t: &TensorId, position: u64) -> u8 {
    (position.wrapping_mul(13).wrapping_add(t.name.len() as u64) % 251) as u8
}
impl ExactReader for Reader {
    fn read_exact(&mut self, tensor: &TensorId, offset: u64, dst: &mut [u8]) -> Result<()> {
        self.calls.push((tensor.clone(), offset, dst.len()));
        if self.fail_at == Some(self.calls.len()) {
            return Err(BankError::ReadFailed);
        }
        for (i, b) in dst.iter_mut().enumerate() {
            *b = byte(tensor, offset + i as u64);
        }
        Ok(())
    }
}
struct Fence(bool);
impl ConsumerFence for Fence {
    fn ready_on_owner(&mut self, _: BankTicket) -> Result<bool> {
        Ok(self.0)
    }
}
fn service(budget: SharedBudget, cache: u64, queue: usize) -> BankService<ExpertDomain, Heat> {
    BankService::new(
        catalog(LayoutClass::PerRecord),
        budget,
        Heat::default(),
        cache,
        1024,
        64,
        queue,
    )
}
fn expected(l: &RecordLayout) -> Vec<u8> {
    l.segments
        .iter()
        .flat_map(|s| (s.offset..s.offset + s.len).map(|p| byte(&s.tensor, p)))
        .collect()
}

#[test]
fn masked_and_unknown_ids_fail_before_queue_or_cache() {
    let g = governor(1024, 0, 2);
    let mut s = service(g.clone(), 20, 2);
    assert!(matches!(s.resident(&id(21)), Err(BankError::MaskedId)));
    assert!(matches!(s.resident(&id(22)), Err(BankError::UnknownId)));
    assert!(matches!(
        catalog(LayoutClass::PerRecord).batch(vec![id(21)]),
        Err(BankError::MaskedId)
    ));
    let forged = Catalog::new(
        LayoutClass::Uniform,
        vec![(id(21), Some(layout(21, "Q3_K", 16)))],
    )
    .unwrap()
    .batch(vec![id(21)])
    .unwrap();
    assert_eq!(s.stage(forged), Err(BankError::MaskedId));
    assert_eq!(g.state.lock().unwrap().used, Charge::default());
}
#[test]
fn mixed_cannot_obtain_uniform_proof_even_for_one_record() {
    let c = catalog(LayoutClass::PerRecord);
    assert!(matches!(
        c.uniform(c.batch(vec![id(9)]).unwrap()),
        Err(BankError::MixedLayout)
    ));
    let c = catalog(LayoutClass::Uniform);
    assert_eq!(
        c.uniform(c.batch(vec![id(47)]).unwrap())
            .unwrap()
            .batch()
            .ids(),
        &[id(47)]
    );
    assert!(matches!(
        Catalog::new(
            LayoutClass::Uniform,
            vec![
                (id(9), Some(layout(9, "Q2_K", 16))),
                (id(47), Some(layout(47, "Q3_K", 16)))
            ]
        ),
        Err(BankError::MixedLayout)
    ));
}
#[test]
fn source_identity_keeps_artifact_projection_and_original_router_slot() {
    let mut a = id(9);
    let b = a.clone();
    a.tensor.artifact[0] += 1;
    assert_ne!(a, b);
    a = b.clone();
    a.record = RecordId::Expert {
        layer: 2,
        original_id: 9,
        projection: Projection::Down,
    };
    assert_ne!(a, b);
    assert!(matches!(
        catalog(LayoutClass::PerRecord).layout(&id(0)),
        Err(BankError::UnknownId)
    ));
}
#[test]
fn zero_small_full_cache_exact_bytes_scales_order_and_forced_misses() {
    for capacity in [0, 20, 40] {
        let g = governor(256, 0, 2);
        let mut s = service(g.clone(), capacity, 2);
        let mut r = Reader::new();
        for _ in 0..5 {
            let t = s.stage(batch(&[47, 9, 47])).unwrap();
            assert!(matches!(
                s.publish(t, &mut Fence(false)),
                Err(BankError::Pending)
            ));
            s.read(t, &mut r).unwrap();
            let leases = s.publish(t, &mut Fence(true)).unwrap();
            for (lease, original) in leases.iter().zip([47, 9, 47]) {
                assert_eq!(lease.id(), &id(original));
                assert_eq!(lease.bytes(), expected(lease.layout()));
            }
            assert!(s.cache_bytes() <= capacity);
        }
        assert_eq!(
            r.calls.len(),
            match capacity {
                0 => 20,
                20 => 12,
                _ => 4,
            }
        );
        drop(s);
        assert_eq!(g.state.lock().unwrap().used, Charge::default());
    }
}
#[test]
fn disk_ready_is_not_consumer_ready_and_queue_is_bounded() {
    let g = governor(256, 0, 4);
    let mut s = service(g.clone(), 40, 1);
    let t = s.stage(batch(&[9])).unwrap();
    assert_eq!(s.stage(batch(&[47])), Err(BankError::Backpressure));
    s.read(t, &mut Reader::new()).unwrap();
    assert!(matches!(
        s.publish(t, &mut Fence(false)),
        Err(BankError::Pending)
    ));
    assert!(s.resident(&id(9)).unwrap().is_none());
    let v = s.publish(t, &mut Fence(true)).unwrap();
    assert!(s.resident(&id(9)).unwrap().is_some());
    assert!(matches!(
        s.publish(t, &mut Fence(true)),
        Err(BankError::UnknownTicket)
    ));
    drop(s); // lease still pins budget after service/cache destruction
    assert_eq!(g.state.lock().unwrap().used.host_bytes, 20);
    drop(v);
    assert_eq!(g.state.lock().unwrap().used, Charge::default());
}
#[test]
fn cancellation_and_cross_service_ticket_do_not_publish() {
    let g = governor(256, 0, 4);
    let mut s = service(g.clone(), 0, 2);
    let mut other = service(g.clone(), 0, 2);
    let t = s.stage(batch(&[9])).unwrap();
    let u = other.stage(batch(&[9])).unwrap();
    assert_eq!(other.cancel(t), Err(BankError::UnknownTicket));
    s.read(t, &mut Reader::new()).unwrap();
    s.cancel(t).unwrap();
    assert!(matches!(
        s.publish(t, &mut Fence(true)),
        Err(BankError::UnknownTicket)
    ));
    other.cancel(u).unwrap();
    assert_eq!(g.state.lock().unwrap().used, Charge::default());
}
#[test]
fn partial_read_error_retires_whole_batch_and_all_permits() {
    let g = governor(256, 0, 2);
    let mut s = service(g.clone(), 40, 2);
    let t = s.stage(batch(&[9, 47])).unwrap();
    let mut r = Reader::new();
    r.fail_at = Some(3);
    assert_eq!(s.read(t, &mut r), Err(BankError::ReadFailed));
    assert!(matches!(
        s.publish(t, &mut Fence(true)),
        Err(BankError::UnknownTicket)
    ));
    assert_eq!(s.cache_bytes(), 0);
    assert_eq!(g.state.lock().unwrap().used, Charge::default());
}
#[test]
fn shared_governor_admission_is_atomic_and_releases_cancelled_wait() {
    let g = governor(30, 4096, 1);
    let mut s = service(g.clone(), 0, 2);
    assert_eq!(s.stage(batch(&[9, 47])), Err(BankError::Backpressure));
    assert_eq!(s.pending_count(), 0);
    assert_eq!(g.state.lock().unwrap().used, Charge::default());
    let kv_permit = g
        .reserve(Charge {
            host_bytes: 20,
            ..Charge::default()
        })
        .unwrap();
    assert_eq!(s.stage(batch(&[9])), Err(BankError::Backpressure));
    drop(kv_permit);
    let t = s.stage(batch(&[9])).unwrap();
    s.cancel(t).unwrap();
    assert_eq!(g.state.lock().unwrap().used, Charge::default());
}
fn table(stride: u64, rows: u64) -> RowTable {
    RowTable {
        tensor: tensor(),
        base: 0,
        rows,
        width: 264,
        stride,
        storage_bytes: rows * stride + 32768,
    }
}
#[test]
fn sparse_264_by_48_amplification_fixture() {
    let table = table(32768, 48);
    let batch = RowBatch {
        rows: (0..48).collect(),
    };
    for granularity in [512, 4096, 16384] {
        let p = table.plan(&batch, granularity).unwrap();
        assert_eq!(p.unique_useful_bytes, 12672);
        assert_eq!(p.logical_bytes, 12672);
        assert_eq!(p.physical_bytes, 48 * granularity);
        assert_eq!(p.extents.len(), 48);
        assert_eq!(p.straddling_rows, 0);
        assert_eq!(p.efficiency(), Some(264.0 / granularity as f64));
        assert_eq!(p.amplification(), Some(granularity as f64 / 264.0));
        println!(
            "fixture=sparse264x48 granularity={granularity} useful={} physical={} amplification={}",
            p.unique_useful_bytes,
            p.physical_bytes,
            p.amplification().unwrap()
        );
    }
}
#[test]
fn ordered_packed_rows_coalesce_and_straddles_count() {
    let table = table(264, 48);
    let batch = RowBatch {
        rows: (0..48).collect(),
    };
    for g in [512, 4096, 16384] {
        let p = table.plan(&batch, g).unwrap();
        assert_eq!(p.physical_bytes, 12672u64.div_ceil(g) * g);
        assert_eq!(p.extents.len(), 1);
        assert_eq!(
            p.straddling_rows,
            match g {
                512 => 24,
                4096 => 3,
                _ => 0,
            }
        );
    }
}
#[test]
fn duplicate_and_history_reordering_preserve_logical_output_not_io() {
    let table = table(264, 96);
    let a = table
        .plan(
            &RowBatch {
                rows: vec![30, 1, 30, 2, 0],
            },
            4096,
        )
        .unwrap();
    let b = table
        .plan(
            &RowBatch {
                rows: vec![0, 1, 2, 30],
            },
            4096,
        )
        .unwrap();
    assert_eq!(a.extents, b.extents);
    assert_eq!(a.unique_useful_bytes, 4 * 264);
    assert_eq!(a.logical_bytes, 5 * 264);
}
#[test]
fn row_bounds_overflow_padding_and_empty_are_explicit() {
    let mut t = table(264, 48);
    assert_eq!(
        t.plan(&RowBatch { rows: vec![48] }, 4096),
        Err(BankError::UnknownId)
    );
    assert_eq!(
        t.plan(&RowBatch { rows: vec![1] }, 3),
        Err(BankError::InvalidLayout)
    );
    t.base = u64::MAX;
    assert_eq!(
        t.plan(&RowBatch { rows: vec![1] }, 4096),
        Err(BankError::Overflow)
    );
    t.base = 0;
    t.storage_bytes = 12672;
    assert_eq!(
        t.plan(&RowBatch { rows: vec![47] }, 4096),
        Err(BankError::InvalidLayout)
    );
    let p = t.plan(&RowBatch { rows: vec![] }, 4096).unwrap();
    assert_eq!(p.efficiency(), None);
    assert_eq!(p.amplification(), None);
}
#[test]
fn gather_host_and_fake_nvme_exact_across_straddles_small_slot_and_duplicates() {
    for granularity in [512, 4096, 16384] {
        let t = table(264, 96);
        let g = governor(20000, granularity, 1);
        let mut s =
            BoundedRowService::new(t.clone(), g.clone(), granularity, granularity, 64, 20000)
                .unwrap();
        let batch = RowBatch {
            rows: (0..48).rev().chain([0, 47, 16]).collect(),
        };
        let mut nvme = Reader::new();
        let lease = s.gather(batch.clone(), &mut nvme).unwrap();
        let expected: Vec<_> = batch
            .rows
            .iter()
            .flat_map(|r| (*r * 264..(*r + 1) * 264).map(|p| byte(&t.tensor, p)))
            .collect();
        assert_eq!(lease.bytes(), expected);
        assert_eq!(
            nvme.calls.iter().map(|(_, _, n)| *n as u64).sum::<u64>(),
            lease.plan.physical_bytes
        );
        assert!(
            nvme.calls
                .iter()
                .all(|(_, offset, n)| offset % granularity == 0 && *n as u64 <= granularity)
        );
        assert!(g.state.lock().unwrap().peak.staging_bytes <= granularity);
        assert_eq!(g.state.lock().unwrap().used.staging_bytes, 0);
        drop(lease);
        assert_eq!(g.state.lock().unwrap().used, Charge::default());
    }
}
#[test]
fn row_read_error_and_output_pressure_leave_no_partial_lease() {
    let g = governor(4096, 4096, 1);
    let mut s = BoundedRowService::new(table(32768, 48), g.clone(), 4096, 4096, 64, 4096).unwrap();
    let mut r = Reader::new();
    r.fail_at = Some(2);
    assert!(matches!(
        s.gather(RowBatch { rows: vec![0, 1] }, &mut r),
        Err(BankError::ReadFailed)
    ));
    assert_eq!(g.state.lock().unwrap().used, Charge::default());
    let held = g
        .reserve(Charge {
            host_bytes: 4096,
            ..Charge::default()
        })
        .unwrap();
    assert!(matches!(
        s.gather(RowBatch { rows: vec![0] }, &mut Reader::new()),
        Err(BankError::Backpressure)
    ));
    drop(held);
    assert!(matches!(
        s.gather(RowBatch { rows: vec![0; 65] }, &mut Reader::new()),
        Err(BankError::TooLarge)
    ));
    assert!(matches!(
        s.gather(RowBatch { rows: vec![0; 16] }, &mut Reader::new()),
        Err(BankError::TooLarge)
    ));
    assert_eq!(g.state.lock().unwrap().used, Charge::default());
}

#[test]
fn uniform_compute_proof_is_bound_to_actual_leased_bytes() {
    let g = governor(256, 0, 4);
    let mut s = service(g.clone(), 0, 2);
    let ticket = s.stage(batch(&[9])).unwrap();
    s.read(ticket, &mut Reader::new()).unwrap();
    let lease = s.publish(ticket, &mut Fence(true)).unwrap();
    assert!(matches!(
        UniformLease::try_new(lease),
        Err(BankError::MixedLayout)
    ));
    let mut u: BankService<ExpertDomain, Heat> = BankService::new(
        catalog(LayoutClass::Uniform),
        g,
        Heat::default(),
        0,
        1024,
        64,
        2,
    );
    let t = u.stage(batch(&[9, 47])).unwrap();
    u.read(t, &mut Reader::new()).unwrap();
    let lease = UniformLease::try_new(u.publish(t, &mut Fence(true)).unwrap()).unwrap();
    assert_eq!(lease.records().len(), 2);
}

#[test]
fn prefetch_is_bounded_mask_checked_and_does_not_add_demand_heat() {
    struct Router(Vec<BankId>);
    impl PrefetchHook<ExpertDomain> for Router {
        type Context = ();
        fn predict(&self, _: &(), _: usize) -> Vec<BankId> {
            self.0.clone()
        }
    }
    let c = catalog(LayoutClass::PerRecord);
    assert!(matches!(
        prefetch_batch(&c, &Router(vec![id(9), id(47)]), &(), 1),
        Err(BankError::TooLarge)
    ));
    assert!(matches!(
        prefetch_batch(&c, &Router(vec![id(21)]), &(), 1),
        Err(BankError::MaskedId)
    ));
    assert!(
        prefetch_batch(&c, &Router(vec![]), &(), 1)
            .unwrap()
            .is_none()
    );
    let mut s = service(governor(256, 0, 2), 20, 2);
    let t = s.stage(batch(&[9])).unwrap();
    s.read(t, &mut Reader::new()).unwrap();
    drop(s.publish(t, &mut Fence(true)).unwrap());
    let b = prefetch_batch(&c, &Router(vec![id(47)]), &(), 1)
        .unwrap()
        .unwrap();
    let t = s.stage(b).unwrap();
    s.read(t, &mut Reader::new()).unwrap();
    drop(s.publish(t, &mut Fence(true)).unwrap());
    assert!(s.resident(&id(9)).unwrap().is_some());
    assert!(s.resident(&id(47)).unwrap().is_none());
}

#[test]
fn eviction_under_pressure_does_not_release_consumer_lease() {
    let g = governor(20, 0, 2);
    let mut s = service(g.clone(), 20, 2);
    let t = s.stage(batch(&[9])).unwrap();
    s.read(t, &mut Reader::new()).unwrap();
    let lease = s.publish(t, &mut Fence(true)).unwrap();
    s.evict_cached(&id(9)).unwrap();
    assert_eq!(s.stage(batch(&[47])), Err(BankError::Backpressure));
    drop(lease);
    let t = s.stage(batch(&[47])).unwrap();
    s.cancel(t).unwrap();
    assert_eq!(g.state.lock().unwrap().used, Charge::default());
}

#[test]
fn immutable_host_and_nvme_fake_backends_produce_identical_row_bytes() {
    struct Host {
        tensor: TensorId,
        bytes: Vec<u8>,
    }
    impl ExactReader for Host {
        fn read_exact(&mut self, tensor: &TensorId, offset: u64, dst: &mut [u8]) -> Result<()> {
            if tensor != &self.tensor {
                return Err(BankError::UnknownId);
            }
            let end = offset as usize + dst.len();
            let src = self
                .bytes
                .get(offset as usize..end)
                .ok_or(BankError::WrongLength)?;
            dst.copy_from_slice(src);
            Ok(())
        }
    }
    let table = table(264, 96);
    let mut host = Host {
        tensor: table.tensor.clone(),
        bytes: (0..table.storage_bytes)
            .map(|p| byte(&table.tensor, p))
            .collect(),
    };
    let mut s =
        BoundedRowService::new(table, governor(8192, 4096, 1), 4096, 4096, 16, 4096).unwrap();
    let rows = RowBatch {
        rows: vec![47, 0, 31, 1, 47],
    };
    let a = s.gather(rows.clone(), &mut host).unwrap();
    let b = s.gather(rows, &mut Reader::new()).unwrap();
    assert_eq!(a.bytes(), b.bytes());
}

#[test]
fn q2_q3_nvfp4_records_keep_distinct_lengths_encodings_and_macro_segments() {
    let entries: Vec<_> = [(9, "Q2_K", 16), (47, "Q3_K", 24), (83, "NVFP4", 32)]
        .into_iter()
        .map(|(e, q, n)| (id(e), Some(layout(e as u64, q, n))))
        .collect();
    let c = Catalog::new(LayoutClass::PerRecord, entries).unwrap();
    let batch = c.batch(vec![id(83), id(9), id(47)]).unwrap();
    let mut s: BankService<ExpertDomain, Heat> =
        BankService::new(c, governor(256, 0, 2), Heat::default(), 0, 256, 8, 2);
    let t = s.stage(batch).unwrap();
    s.read(t, &mut Reader::new()).unwrap();
    let leases = s.publish(t, &mut Fence(true)).unwrap();
    for (lease, (q, n)) in leases
        .iter()
        .zip([("NVFP4", 36), ("Q2_K", 20), ("Q3_K", 28)])
    {
        assert_eq!(lease.layout().encoding, q);
        assert_eq!(lease.bytes().len(), n);
        assert_eq!(lease.bytes(), expected(lease.layout()));
    }
}
