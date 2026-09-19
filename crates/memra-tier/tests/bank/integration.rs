use super::*;
use memra_tier::{
    object_store::{BlobBackend, ExtentStore},
    pool::FakePinnedPool,
    tier::Governor as SharedGovernor,
};
use std::{collections::HashMap, sync::Arc};

#[derive(Default)]
struct Memory(HashMap<(bool, Digest), Vec<u8>>, Rc<std::cell::Cell<bool>>);
impl BlobBackend for Memory {
    fn get(&self, root: bool, id: Digest, max: usize) -> Result<Option<Vec<u8>>> {
        if self.1.get() {
            return Err(Error::Corrupt);
        }
        let v = self.0.get(&(root, id));
        if v.is_some_and(|v| v.len() > max) {
            return Err(Error::Capacity);
        }
        Ok(v.cloned())
    }
    fn insert(&mut self, root: bool, id: Digest, bytes: &[u8]) -> Result<()> {
        if let Some(old) = self.0.get(&(root, id))
            && old != bytes
        {
            return Err(Error::Conflict);
        }
        self.0.insert((root, id), bytes.to_vec());
        Ok(())
    }
}
fn shared(cap: u64, reserve: u64) -> Rc<RefCell<SharedGovernor>> {
    let mut c = TierBudget::zero(2);
    c.pageable = cap;
    c.staging = cap;
    c.nvme = cap;
    c.inflight = 100;
    let mut h = TierBudget::zero(2);
    h.pageable = reserve;
    h.staging = reserve;
    h.inflight = 1;
    Rc::new(RefCell::new(
        SharedGovernor::new(c, h, 32, cap, Arc::new(|| 1)).unwrap(),
    ))
}

#[test]
fn object_store_transfer_rows_forced_misses_order_and_retirement() {
    let g = shared(2_000_000, 10_000);
    let table = ple(PleEncoding::F32);
    let ids: Vec<_> = [2, 5, 9].iter().map(|&n| table.id(n).unwrap()).collect();
    let entries = ids
        .iter()
        .map(|id| {
            let RecordId::Row(n) = id.record else {
                unreachable!()
            };
            let l = table.layout(n).unwrap();
            (id.clone(), Some(record(l)))
        })
        .collect();
    let cat = Catalog::new(LayoutClass::PerRecord, entries).unwrap();
    let t = cat.record(&ids[0]).unwrap().layout.segments[0]
        .tensor
        .clone()
        .unwrap();
    let key = ObjectKey {
        version: 1,
        artifact: t.artifact,
        semantic_id: t.identity().unwrap(),
        layout: [11; 32],
        generation: 0,
    };
    let mut store = ExtentStore::new(Memory::default(), g.clone());
    let payload: Vec<_> = (0..8192).map(|p| byte(&t, p)).collect();
    let mut txn = store
        .begin(key.clone(), payload.len() as u64, Durability::Ephemeral)
        .unwrap();
    // Objects are chunked to match the bounded source pool, never whole-table pinned.
    for part in payload.chunks(512) {
        store.put(&mut txn, part).unwrap();
    }
    store.commit(&mut txn).unwrap();
    let mut pool_req = request(2 * (512 + 511), Priority::MandatoryActive);
    pool_req.bytes.staging = 1024;
    let pool_charge = g.borrow_mut().reserve(&pool_req).unwrap();
    let pool = FakePinnedPool::new(2, 512, 512, 1, &pool_charge).unwrap();
    let mut queue_req = request(0, Priority::MandatoryActive);
    queue_req.bytes.inflight = 1;
    let queue_charge = g.borrow_mut().reserve(&queue_req).unwrap();
    let mut io_req = request(0, Priority::Demand);
    io_req.bytes.nvme = 200_000;
    let reader =
        ObjectReader::new(store, vec![(t, key)], pool.clone(), io_req, &queue_charge).unwrap();
    let mut rows = BoundedRowService(
        BankService::new(
            cat,
            g.clone(),
            Heat::default(),
            reader,
            CoalescingPolicy {
                granularity: 512,
                slot_bytes: 512,
            },
            limits(0),
        )
        .unwrap(),
    );
    let baseline = g.borrow().used();
    for _ in 0..2 {
        let ticket = rows
            .gather(RowBatch {
                ids: vec![
                    ids[2].clone(),
                    ids[0].clone(),
                    ids[2].clone(),
                    ids[1].clone(),
                ],
                epochs: epochs(),
                request: request(0, Priority::Demand),
            })
            .unwrap();
        let submitted = rows.0.reader().submitted;
        assert!(matches!(
            rows.publish(&ticket, epochs()),
            Err(Error::NotReady)
        ));
        assert_eq!(rows.0.reader().submitted, submitted); // publish is never an I/O pump
        while !rows.0.progress(&ticket).unwrap() {}
        assert!(rows.0.reader().submitted > submitted); // zero-cache forced misses twice
        assert_eq!(rows.0.reader().submitted, rows.0.reader().retired);
        let completion = rows.0.reader().last_completion.as_ref().unwrap();
        assert!(completion.producer_done);
        assert!(!completion.consumer_fenced);
        assert!(
            completion
                .items
                .iter()
                .all(|i| i.accepted && i.segments.iter().all(|s| s.status == ItemStatus::Complete))
        );
        let leases = rows.publish(&ticket, epochs()).unwrap();
        assert_eq!(leases.records[0].id(), &ids[2]);
        assert_eq!(
            leases.records[0].charge().id(),
            leases.records[2].charge().id()
        );
        for lease in &leases.records {
            assert_eq!(
                &*lease.resource::<Vec<u8>>().unwrap(),
                &expected(lease.layout())
            );
        }
        finish(&mut rows.0, &ticket);
        rows.release(&leases).unwrap();
        assert_eq!(g.borrow().used(), baseline);
        assert_eq!(pool.accounting().free_slots, 2);
    }
    // Cancel after one physical chunk but before the full logical batch completes.
    let ticket = rows
        .gather(RowBatch {
            ids: ids.clone(),
            epochs: epochs(),
            request: request(0, Priority::Demand),
        })
        .unwrap();
    assert!(!rows.0.progress(&ticket).unwrap());
    let submitted = rows.0.reader().submitted;
    rows.cancel(&ticket).unwrap();
    assert!(!rows.retire(&ticket).unwrap());
    rows.0.progress(&ticket).unwrap();
    assert_eq!(rows.0.reader().submitted, submitted);
    assert!(matches!(
        rows.publish(&ticket, epochs()),
        Err(Error::Cancelled)
    ));
    finish(&mut rows.0, &ticket);
    assert_eq!(g.borrow().used(), baseline);
    drop(rows);
    drop(pool);
    g.borrow_mut().release(&pool_charge).unwrap();
    g.borrow_mut().release(&queue_charge).unwrap();
    assert_eq!(g.borrow().used(), TierBudget::zero(2));
}

#[test]
fn shared_governor_hints_cannot_starve_mandatory_loads() {
    let g = shared(100_000, 40_000);
    let (cat, ids) = catalog(LayoutClass::PerRecord);
    let hint = RouterTopKHint::new(&cat, &ids[..3], 2, 40).unwrap();
    let context = vec![ids[0].clone(), ids[0].clone(), ids[2].clone()];
    assert_eq!(hint.predict(&context, 20), vec![ids[0].clone()]);
    assert!(RouterTopKHint::new(&cat, &ids, 2, 40).is_err()); // mask rejects
    let p = prefetch_batch::<ExpertDomain, _>(
        &cat,
        &hint,
        &context,
        20,
        epochs(),
        request(0, Priority::Demand),
    )
    .unwrap()
    .unwrap();
    assert_eq!(p.request.priority, Priority::OptionalPrefetch);
    let mut bank: Banks = BankService::new(
        cat,
        g.clone(),
        Heat::default(),
        Reader::default(),
        CoalescingPolicy::default(),
        limits(0),
    )
    .unwrap();
    let mut tickets = vec![];
    for _ in 0..3 {
        tickets.push(bank.stage(p.clone()).unwrap());
    }
    assert_eq!(bank.stage(p), Err(Error::Capacity)); // reserved demand ticket
    // Fill remaining optional capacity with another caller (e.g. KV prefix).
    let used = g.borrow().used().pageable;
    let other = g
        .borrow_mut()
        .reserve(&request(60_000 - used, Priority::OptionalPrefetch))
        .unwrap();
    assert_eq!(
        bank.stage(batch(vec![ids[1].clone()])),
        Err(Error::Capacity)
    );
    let mut mandatory = batch(vec![ids[1].clone()]);
    mandatory.request.priority = Priority::MandatoryActive;
    let t = bank.stage(mandatory).unwrap();
    drain(&mut bank, &t);
    let leases = bank.publish(&t, epochs()).unwrap();
    finish(&mut bank, &t);
    for l in leases {
        bank.release(&l).unwrap();
    }
    for t in tickets {
        bank.cancel(&t).unwrap();
        finish(&mut bank, &t);
    }
    g.borrow_mut().release(&other).unwrap();
    assert_eq!(g.borrow().used(), TierBudget::zero(2));
}

#[test]
fn ngram_hint_is_byte_bounded_deduplicated_and_domain_separate() {
    let table = ple(PleEncoding::Bf16);
    let ids: Vec<_> = [2, 5, 9].iter().map(|&n| table.id(n).unwrap()).collect();
    let entries = ids
        .iter()
        .map(|id| {
            let RecordId::Row(n) = id.record else {
                unreachable!()
            };
            (id.clone(), Some(record(table.layout(n).unwrap())))
        })
        .collect();
    let cat = Catalog::new(LayoutClass::PerRecord, entries).unwrap();
    let h = NgramLookaheadHint::new(&cat, &ids, 4, 264).unwrap();
    assert_eq!(
        h.predict(&vec![ids[1].clone(), ids[1].clone(), ids[0].clone()], 99),
        vec![ids[1].clone()]
    );
    assert_eq!(h.predict(&ids, 0), Vec::<BankId>::new());
    assert!(RouterTopKHint::new(&cat, &ids, 4, 1000).is_err());
}

#[test]
fn object_transfer_experts_partial_failure_never_publishes_successful_sibling() {
    for fail in [false, true] {
        let g = shared(1_000_000, 10_000);
        let mut ls = vec![];
        for n in 0..2 {
            let mut l = layout(n, 13, 16);
            l.segments[0].offset = n * 512;
            l.segments[1].tensor = Some(tensor());
            l.segments[1].offset = n * 512 + 16;
            ls.push(l);
        }
        let ids: Vec<_> = ls
            .iter()
            .enumerate()
            .map(|(n, l)| bank_id(n as u32, l))
            .collect();
        let cat = Catalog::new(
            LayoutClass::PerRecord,
            ids.iter()
                .cloned()
                .zip(ls.into_iter().map(|l| Some(record(l))))
                .collect(),
        )
        .unwrap();
        let key = ObjectKey {
            version: 1,
            artifact: tensor().artifact,
            semantic_id: tensor().identity().unwrap(),
            layout: [0; 32],
            generation: 0,
        };
        let backend = Memory::default();
        let fault = backend.1.clone();
        let mut store = ExtentStore::new(backend, g.clone());
        let mut txn = store
            .begin(key.clone(), 1024, Durability::Ephemeral)
            .unwrap();
        let data: Vec<_> = (0..1024).map(|p| byte(&tensor(), p)).collect();
        for c in data.chunks(512) {
            store.put(&mut txn, c).unwrap();
        }
        store.commit(&mut txn).unwrap();
        let pc = g
            .borrow_mut()
            .reserve(&request(1023, Priority::MandatoryActive))
            .unwrap();
        let pool = FakePinnedPool::new(1, 512, 512, 0, &pc).unwrap();
        let mut qr = request(0, Priority::MandatoryActive);
        qr.bytes.inflight = 1;
        let qc = g.borrow_mut().reserve(&qr).unwrap();
        let mut io = request(0, Priority::Demand);
        io.bytes.nvme = 32768;
        let reader = ObjectReader::new(store, vec![(tensor(), key)], pool, io, &qc).unwrap();
        let mut bank: BankService<ExpertDomain, _, _> = BankService::new(
            cat,
            g.clone(),
            Heat::default(),
            reader,
            CoalescingPolicy {
                granularity: 512,
                slot_bytes: 512,
            },
            limits(0),
        )
        .unwrap();
        let baseline = g.borrow().used();
        let t = bank
            .stage(batch(vec![ids[1].clone(), ids[0].clone(), ids[1].clone()]))
            .unwrap();
        assert_eq!(bank.reader().submitted, 0);
        assert!(!bank.progress(&t).unwrap());
        assert!(matches!(bank.publish(&t, epochs()), Err(Error::NotReady)));
        fault.set(fail);
        assert!(bank.progress(&t).unwrap());
        assert_eq!(bank.reader().submitted, 2);
        assert_eq!(bank.reader().retired, 2);
        if fail {
            assert!(matches!(bank.publish(&t, epochs()), Err(Error::Corrupt)));
            assert_eq!(bank.cache_bytes(), 0);
            let c = bank.completion(&t).unwrap();
            assert_eq!(c.items.len(), 2);
            assert!(
                c.items
                    .iter()
                    .all(|i| i.segments.iter().all(|s| s.status == ItemStatus::Failed))
            );
            let c = bank.reader().last_completion.as_ref().unwrap();
            assert_eq!(c.items[0].segments[0].error, Some(Error::Corrupt));
            bank.cancel(&t).unwrap();
            finish(&mut bank, &t);
        } else {
            let leases = bank.publish(&t, epochs()).unwrap();
            assert_eq!(leases[0].id(), &ids[1]);
            assert_eq!(leases[0].charge().id(), leases[2].charge().id());
            for l in &leases {
                assert_eq!(&*l.resource::<Vec<u8>>().unwrap(), &expected(l.layout()));
            }
            assert!(UniformLease::try_new(leases.clone()).is_err());
            finish(&mut bank, &t);
            for l in &leases[..2] {
                bank.release(l).unwrap();
            }
        }
        assert_eq!(g.borrow().used(), baseline);
        drop(bank);
        g.borrow_mut().release(&pc).unwrap();
        g.borrow_mut().release(&qc).unwrap();
        assert_eq!(g.borrow().used(), TierBudget::zero(2));
    }
}
