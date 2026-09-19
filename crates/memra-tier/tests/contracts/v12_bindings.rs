//! CPU implementation bindings for the additive v1.2 schedules.
use super::{conformance as schedule, support::*};
use memra_tier::{
    bank::*,
    contracts::*,
    object_store::{BlobBackend, ExtentStore},
    peer::{test_support::FakePeerCapacity, topology::LinkHealth},
    pool::FakePinnedPool,
};
use std::{cell::RefCell, collections::HashMap, rc::Rc};

#[test]
fn revision_v12_ready_owner_and_consumer_fence_identity() {
    let mut g = Governor::new(1000);
    let mut req = request(0, Priority::MandatoryActive);
    req.bytes.device[0] = 4;
    let charge = g.reserve(&req).unwrap();
    let mut owner = DeviceOwner::new(0);
    let foreign = DeviceOwner::new(0);
    let device = owner.register(31, 4, Box::new(bytes(4)), &charge).unwrap();
    let ticket = TransferTicket {
        issuer: owner.issuer(),
        sequence: 1,
        epochs: epochs(),
    };
    let mut c = completion(ticket, 2);
    for item in &mut c.items {
        item.segments[0].consumer_fence.as_mut().unwrap().issuer = owner.issuer();
    }
    schedule::ready_owner(&mut owner, &device, &foreign, &c, &expected(2));
    assert_eq!(g.release(&charge), Err(Error::Busy));
    owner.release(&device).unwrap();
    g.release(&charge).unwrap();
    assert_eq!(g.used, TierBudget::zero(2));
}

#[test]
fn revision_v12_directed_grants_bind_exported_d_capacity() {
    let g = Rc::new(RefCell::new(Governor::new(1000)));
    let mut p = FakePeerCapacity::new(g, 2, epochs().state);
    for (s, d) in [(0, 1), (1, 0)] {
        p.set_route(s, d, true, true, LinkHealth::AtMaximum)
            .unwrap();
    }
    let plan = |owner: u32| {
        let mut r = request(0, Priority::Demand);
        r.bytes.device[owner as usize] = 4;
        r.bytes.peer[owner as usize] = 4;
        PeerPlan {
            owner_device: owner,
            consumer_device: 1 - owner,
            bytes: 4,
            alignment: 4,
            epochs: epochs(),
            request: r,
        }
    };
    schedule::peer_directed_grants(
        &mut p,
        plan(0),
        plan(1),
        |p, kind, denied| {
            p.set_route(
                0,
                1,
                !(denied && matches!(kind, schedule::RouteFault::Context)),
                !(denied && matches!(kind, schedule::RouteFault::Pool)),
                if denied && matches!(kind, schedule::RouteFault::Downgrade) {
                    LinkHealth::Downgraded
                } else {
                    LinkHealth::AtMaximum
                },
            )
            .unwrap();
        },
        |p| p.gov.borrow().used.clone(),
    );
    assert_eq!(p.gov.borrow().used, TierBudget::zero(2));
}

#[derive(Default)]
struct Memory(HashMap<(bool, Digest), Vec<u8>>);
impl BlobBackend for Memory {
    fn get(&self, root: bool, id: Digest, max: usize) -> Result<Option<Vec<u8>>> {
        let value = self.0.get(&(root, id));
        if value.is_some_and(|v| v.len() > max) {
            return Err(Error::Capacity);
        }
        Ok(value.cloned())
    }
    fn insert(&mut self, root: bool, id: Digest, bytes: &[u8]) -> Result<()> {
        if self.0.get(&(root, id)).is_some_and(|v| v != bytes) {
            return Err(Error::Conflict);
        }
        self.0.insert((root, id), bytes.to_vec());
        Ok(())
    }
}
fn source_metadata(row: bool) -> (Catalog, BankId, RecordLayout, ObjectKey) {
    let tensor = TensorId {
        version: 1,
        artifact: [1; 32],
        name: "opaque.table".into(),
    };
    let mut l = layout();
    l.segments[0].tensor = Some(tensor.clone());
    let id = BankId {
        version: 1,
        tensor: tensor.clone(),
        record: if row {
            RecordId::Row(0)
        } else {
            RecordId::Expert {
                layer: 0,
                original_id: 0,
                projection: Projection::Up,
            }
        },
        layout: l.identity().unwrap(),
    };
    let catalog = Catalog::new(
        LayoutClass::PerRecord,
        vec![(
            id.clone(),
            Some(CatalogRecord {
                layout: l.clone(),
                checksums: vec![checksum(&bytes(3))],
            }),
        )],
    )
    .unwrap();
    let key = ObjectKey {
        version: 1,
        artifact: tensor.artifact,
        semantic_id: tensor.identity().unwrap(),
        layout: [22; 32],
        generation: 7,
    };
    (catalog, id, l, key)
}

/// Backend fault is limited to metadata lookup; read must not occur on install.
struct Lookup(Option<ObjectManifest>);
impl ObjectStore for Lookup {
    type Transaction = ();
    fn lookup(&self, _: &ObjectKey) -> Result<Option<ObjectManifest>> {
        Ok(self.0.clone())
    }
    fn begin(&mut self, _: ObjectKey, _: u64, _: Durability) -> Result<()> {
        Err(Error::Unsupported)
    }
    fn put(&mut self, _: &mut (), _: &[u8]) -> Result<()> {
        Err(Error::Unsupported)
    }
    fn commit(&mut self, _: &mut ()) -> Result<ObjectManifest> {
        Err(Error::Unsupported)
    }
    fn cancel(&mut self, _: &mut ()) -> Result<CancelState> {
        Err(Error::Unsupported)
    }
    fn lease(&mut self, _: &ObjectManifest, _: &BudgetRequest) -> Result<ObjectLease> {
        Err(Error::Unsupported)
    }
    fn read(&mut self, _: &ObjectLease, _: u32, _: &mut [u8]) -> Result<u64> {
        panic!("install must not read payload")
    }
    fn release(&mut self, _: &ObjectLease) -> Result<()> {
        Err(Error::Unsupported)
    }
    fn evict(&mut self, _: &ObjectKey) -> Result<()> {
        Err(Error::Unsupported)
    }
}
#[test]
fn revision_v12_bank_source_install_fail_closed() {
    schedule::bank_source_install(|case| {
        use schedule::InstallCase::*;
        let (catalog, id, _, key) = source_metadata(true);
        let mut specs = vec![BankSourceSpec {
            tensor: id.tensor.clone(),
            key: key.clone(),
            valid_bytes: 4,
        }];
        let mut supplied = vec![(id.tensor.clone(), key.clone())];
        let mut manifest = Some(ObjectManifest {
            version: 1,
            key,
            valid_bytes: 4,
            chunks: vec![ChunkRef {
                version: 1,
                encoded_digest: [2; 32],
                valid_bytes: 4,
                storage_bytes: 4096,
                checksum: checksum(&bytes(4)),
            }],
            durability: Durability::Ephemeral,
        });
        match case {
            Valid => (),
            MissingSource => supplied.clear(),
            AmbiguousSource => supplied.push(supplied[0].clone()),
            WrongLayout => supplied[0].1.layout[0] ^= 1,
            WrongGeneration => supplied[0].1.generation += 1,
            MissingObject => manifest = None,
            WrongManifest => manifest.as_mut().unwrap().key.layout[0] ^= 1,
            MissingSpec => specs.clear(),
            AmbiguousSpec => specs.push(specs[0].clone()),
            WrongLength => specs[0].valid_bytes += 1,
        }
        let mut g = Governor::new(10000);
        let pool_charge = g
            .reserve(&request(1023, Priority::MandatoryActive))
            .unwrap();
        let pool = FakePinnedPool::new(1, 512, 512, 0, &pool_charge).unwrap();
        let mut qr = request(0, Priority::MandatoryActive);
        qr.bytes.inflight = 1;
        let queue = g.reserve(&qr).unwrap();
        let result = BankSource::install(
            catalog,
            specs,
            Lookup(manifest),
            supplied,
            pool,
            request(0, Priority::Demand),
            &queue,
        )
        .map(drop);
        g.release(&pool_charge).unwrap();
        g.release(&queue).unwrap();
        assert_eq!(g.used, TierBudget::zero(2));
        result
    });
}
#[derive(Default)]
struct Heat;
impl<D: BankDomain> Hotness<D> for Heat {
    fn demand(&mut self, _: &BankId) {}
    fn score(&self, _: &BankId) -> u64 {
        0
    }
}

#[test]
fn revision_v12_object_bank_and_row_logical_completion_boundaries() {
    for row in [false, true] {
        let g = Rc::new(RefCell::new(Governor::new(1_000_000)));
        let (catalog, id, l, key) = source_metadata(row);
        let mut store = ExtentStore::new(Memory::default(), g.clone());
        let payload = [bytes(3), vec![0]].concat();
        let mut txn = store.begin(key.clone(), 4, Durability::Ephemeral).unwrap();
        store.put(&mut txn, &payload).unwrap();
        let manifest = store.commit(&mut txn).unwrap();
        assert!(manifest.chunks[0].storage_bytes > manifest.valid_bytes);
        let mut io = request(0, Priority::Demand);
        io.bytes.nvme = 100_000;
        schedule::object_logical_bytes(&mut store, &manifest, &io, std::slice::from_ref(&payload));
        let pc = g
            .borrow_mut()
            .reserve(&request(1023, Priority::MandatoryActive))
            .unwrap();
        let pool = FakePinnedPool::new(1, 512, 512, 0, &pc).unwrap();
        let mut qr = request(0, Priority::MandatoryActive);
        qr.bytes.inflight = 1;
        let qc = g.borrow_mut().reserve(&qr).unwrap();
        let source = BankSource::install(
            catalog,
            vec![BankSourceSpec {
                tensor: id.tensor.clone(),
                key: key.clone(),
                valid_bytes: 4,
            }],
            store,
            vec![(id.tensor.clone(), key)],
            pool,
            io,
            &qc,
        )
        .unwrap();
        let (catalog, reader) = source.into_parts();
        let limits = BankLimits {
            cache_bytes: 0,
            batch_bytes: 4096,
            items: 4,
            tickets: 4,
        };
        let policy = CoalescingPolicy {
            granularity: 4,
            slot_bytes: 4,
        };
        let expected = vec![vec![SegmentExpectation {
            valid_bytes: l.segments[0].valid_bytes,
            io_bytes: l.segments[0].valid_bytes,
            checksum: checksum(&bytes(3)),
        }]];
        if row {
            let mut service = BoundedRowService(
                BankService::<RowDomain, _, _>::new(
                    catalog,
                    g.clone(),
                    Heat,
                    reader,
                    policy,
                    limits,
                )
                .unwrap(),
            );
            let t = service
                .gather(RowBatch {
                    ids: vec![id],
                    epochs: epochs(),
                    request: request(0, Priority::Demand),
                })
                .unwrap();
            while !service.0.progress(&t).unwrap() {}
            schedule::row_completion_bytes(&service, &t, &expected, |s, t| {
                s.0.completion(t).unwrap().clone()
            });
            let lease = service.publish(&t, epochs()).unwrap();
            assert_eq!(&*lease.records[0].resource::<Vec<u8>>().unwrap(), &payload);
            assert_eq!(service.release(&lease), Err(Error::Busy));
            service.0.finish_host_use(&t).unwrap();
            assert!(service.retire(&t).unwrap());
            service.release(&lease).unwrap();
            service.0.acknowledge(&t).unwrap();
        } else {
            let mut service = BankService::<ExpertDomain, _, _>::new(
                catalog,
                g.clone(),
                Heat,
                reader,
                policy,
                limits,
            )
            .unwrap();
            let t = service
                .stage(BankBatch {
                    ids: vec![id],
                    epochs: epochs(),
                    request: request(0, Priority::Demand),
                })
                .unwrap();
            while !service.progress(&t).unwrap() {}
            schedule::bank_completion_bytes(&service, &t, &expected, |s, t| {
                s.completion(t).unwrap().clone()
            });
            let leases = service.publish(&t, epochs()).unwrap();
            assert_eq!(&*leases[0].resource::<Vec<u8>>().unwrap(), &payload);
            assert_eq!(service.release(&leases[0]), Err(Error::Busy));
            service.finish_host_use(&t).unwrap();
            assert!(service.retire(&t).unwrap());
            service.release(&leases[0]).unwrap();
            service.acknowledge(&t).unwrap();
        }
        g.borrow_mut().release(&pc).unwrap();
        g.borrow_mut().release(&qc).unwrap();
        assert_eq!(g.borrow().used, TierBudget::zero(2));
    }
}
