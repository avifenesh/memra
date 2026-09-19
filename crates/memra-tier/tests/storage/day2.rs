use super::*;
use memra_tier::contracts::{CancelState, ChargeState, Destination, ReadPlan, TransferEngine};
use memra_tier::io::transfer::CpuTransfers;

#[test]
fn frozen_object_schedule_real_files_and_transaction_owner() {
    let directory = OwnedDirectory::new();
    let mut s = new_store(FileBackend::open_mode(&directory.0, Durability::Persistent).unwrap());
    conformance::object_cancel(&mut s, key());
    let mut foreign = new_store(MemoryBackend::default());
    let mut txn = s.begin(key(), 1, Durability::Persistent).unwrap();
    assert_eq!(foreign.put(&mut txn, &[3]), Err(Error::ForeignLease));
    s.put(&mut txn, &[3]).unwrap();
    let m = s.commit(&mut txn).unwrap();
    assert_eq!(s.cancel(&mut txn), Ok(CancelState::AlreadyPublished));
    let mut request = support::request(0, Priority::MandatoryActive);
    assert!(matches!(s.lease(&m, &request), Err(Error::Capacity)));
    request.bytes.nvme = 32768;
    let lease = s.lease(&m, &request).unwrap();
    assert_eq!(s.evict(&key()), Err(Error::Busy));
    let mut altered = m.clone();
    altered.chunks[0].checksum[0] ^= 1;
    assert!(matches!(s.lease(&altered, &request), Err(Error::Conflict)));
    assert_eq!(foreign.release(&lease), Err(Error::ForeignLease));
    s.release(&lease).unwrap();
    assert_eq!(s.release(&lease), Err(Error::AlreadyReleased));
    assert_eq!(s.read(&lease, 0, &mut [0]), Err(Error::AlreadyReleased));
}
#[test]
fn persistent_restart_ignores_partial_pending_root_and_chunks() {
    let directory = OwnedDirectory::new();
    let mut s = new_store(FileBackend::open_mode(&directory.0, Durability::Persistent).unwrap());
    let mut txn = s.begin(key(), 8192, Durability::Persistent).unwrap();
    s.put(&mut txn, &payload(4096)).unwrap();
    // Simulate process loss after a short staging write, before the atomic link.
    let pending = directory.0.join(".pending-simulated-crash");
    std::fs::write(&pending, &encode_extent(&payload(264)).unwrap()[..97]).unwrap();
    std::fs::File::open(&pending).unwrap().sync_all().unwrap();
    drop(txn);
    drop(s);
    let mut reopened =
        new_store(FileBackend::open_mode(&directory.0, Durability::Persistent).unwrap());
    assert!(reopened.lookup(&key()).unwrap().is_none());
    let mut txn = reopened.begin(key(), 8192, Durability::Persistent).unwrap();
    reopened.put(&mut txn, &payload(4096)).unwrap();
    reopened.put(&mut txn, &payload(4096)).unwrap();
    let manifest = reopened.commit(&mut txn).unwrap();
    drop(reopened);
    let mut restart =
        new_store(FileBackend::open_mode(&directory.0, Durability::Persistent).unwrap());
    assert_eq!(restart.lookup(&key()).unwrap(), Some(manifest.clone()));
    assert_eq!(
        read_manifest(&mut restart, &manifest, 0).unwrap(),
        payload(4096)
    );
    assert_eq!(
        read_manifest(&mut restart, &manifest, 1).unwrap(),
        payload(4096)
    );
    // The completed root is never replaced by subsequent temporary-file garbage.
    std::fs::write(&pending, [0; 11]).unwrap();
    assert_eq!(restart.lookup(&key()).unwrap(), Some(manifest));
}
#[test]
fn charged_pool_pins_backing_once_until_last_slice_returns() {
    let gov = governor();
    let r = support::request(2 * (4096 + 4095), Priority::MandatoryActive);
    let charge = gov.borrow_mut().reserve(&r).unwrap();
    let pool = FakePinnedPool::new(2, 4096, 4096, 1, &charge).unwrap();
    assert_eq!(gov.borrow().used().pageable, r.bytes.pageable);
    let mut lease = pool.acquire(264, Admission::Demand).unwrap();
    lease.write(&payload(264)).unwrap();
    let contract: &dyn memra_tier::contracts::PinnedLease = &lease;
    assert_eq!(contract.storage_bytes(), 4096);
    assert_eq!(contract.valid_bytes(), 264);
    assert_eq!(contract.alignment(), 4096);
    assert_eq!(contract.numa_node(), None);
    assert_eq!(contract.bytes().unwrap(), payload(264));
    assert_eq!(gov.borrow_mut().release(&charge), Err(Error::Busy));
    drop(pool);
    assert_eq!(gov.borrow_mut().release(&charge), Err(Error::Busy));
    lease.release();
    gov.borrow_mut().release(&charge).unwrap();
    assert_eq!(gov.borrow().used().pageable, 0);
    assert_eq!(charge.state().unwrap(), ChargeState::Released);
}
fn transfers() -> (
    CpuTransfers<ExtentStore<MemoryBackend, memra_tier::tier::Governor>>,
    FakePinnedPool,
) {
    let mut s = new_store(MemoryBackend::default());
    let mut txn = s.begin(key(), 264, Durability::Ephemeral).unwrap();
    s.put(&mut txn, &payload(264)).unwrap();
    s.commit(&mut txn).unwrap();
    let mut request = support::request(0, Priority::MandatoryActive);
    request.bytes.nvme = 32768;
    let mut queue = support::request(0, Priority::MandatoryActive);
    queue.bytes.inflight = 4;
    let charge = governor().borrow_mut().reserve(&queue).unwrap();
    (
        CpuTransfers::new(s, request, 4, &charge).unwrap(),
        new_pool(4, 4096, 4096, 0).unwrap(),
    )
}
fn plan(pool: &FakePinnedPool) -> ReadPlan<memra_tier::pool::PinnedLease> {
    ReadPlan {
        object: key(),
        chunk: 0,
        destination: pool.acquire(264, Admission::Demand).unwrap(),
        epochs: epochs(7),
    }
}
#[test]
fn frozen_transfer_cancel_schedule_real_cpu_adapter() {
    let (mut engine, pool) = transfers();
    let ticket = engine.nvme_read(plan(&pool)).unwrap();
    conformance::transfer_cancel(&mut engine, ticket, |e, t| {
        e.drive(t).unwrap();
        e.retire(t, None).unwrap();
    });
    assert_eq!(pool.accounting().free_slots, 4);
}
#[test]
fn host_take_once_stale_triples_and_consumer_retirement() {
    let (mut engine, pool) = transfers();
    let ticket = engine.nvme_read(plan(&pool)).unwrap();
    assert!(matches!(
        engine.take_destination(&ticket, 0, ticket.epochs),
        Err(Error::NotReady)
    ));
    engine.drive(&ticket).unwrap();
    let expected = vec![vec![memra_tier::contracts::SegmentExpectation {
        valid_bytes: 264,
        io_bytes: 8192,
        checksum: digest(&payload(264)),
    }]];
    engine
        .poll(&ticket)
        .unwrap()
        .require(&ticket, &expected, false)
        .unwrap();
    assert!(
        engine
            .poll(&ticket)
            .unwrap()
            .require(&ticket, &expected, true)
            .is_err()
    );
    for e in [
        Epochs {
            state: 8,
            ..ticket.epochs
        },
        Epochs {
            src_gen: 20,
            ..ticket.epochs
        },
        Epochs {
            dst_gen: 32,
            ..ticket.epochs
        },
    ] {
        assert!(matches!(
            engine.take_destination(&ticket, 0, e),
            Err(Error::StaleEpoch)
        ));
    }
    let Destination::Host(host) = engine.take_destination(&ticket, 0, ticket.epochs).unwrap()
    else {
        panic!()
    };
    assert_eq!(host.bytes().unwrap(), payload(264));
    assert_eq!(engine.cancel(&ticket), Ok(CancelState::AlreadyPublished));
    assert!(matches!(
        engine.take_destination(&ticket, 0, ticket.epochs),
        Err(Error::AlreadyReleased)
    ));
    assert_eq!(engine.retire(&ticket, None), Err(Error::Busy));
    drop(host);
    engine.retire(&ticket, None).unwrap();
    engine.acknowledge(&ticket).unwrap();
    assert_eq!(pool.accounting().free_slots, 4);
}
#[test]
fn mixed_epoch_batch_returns_all_owned_inputs_and_missing_chunk_refuses_publish() {
    use memra_tier::contracts::TransferOp;
    let (mut engine, pool) = transfers();
    let a = plan(&pool);
    let mut b = plan(&pool);
    b.epochs.dst_gen += 1;
    let rejected = engine
        .submit_batch(vec![TransferOp::NvmeRead(a), TransferOp::NvmeRead(b)])
        .unwrap_err();
    assert_eq!(rejected.error, Error::StaleEpoch);
    assert_eq!(rejected.op.len(), 2);
    drop(rejected);
    let a = plan(&pool);
    let mut b = plan(&pool);
    b.chunk = 99;
    let submit = engine
        .submit_batch(vec![TransferOp::NvmeRead(a), TransferOp::NvmeRead(b)])
        .unwrap();
    submit.validate(2).unwrap();
    engine.drive(&submit.ticket).unwrap();
    assert_eq!(engine.poll(&submit.ticket).unwrap().items.len(), 2);
    assert!(
        engine
            .take_destination(&submit.ticket, 0, submit.ticket.epochs)
            .is_err()
    );
    engine.retire(&submit.ticket, None).unwrap();
    engine.acknowledge(&submit.ticket).unwrap();
    assert_eq!(pool.accounting().free_slots, 4);
}
#[test]
fn bench_real_roundtrip_reopen_restore_uncached_and_red_modes() {
    let dir = OwnedDirectory::new();
    let path = dir.0.join("objects");
    let mut args = vec![
        "storage-bench".into(),
        "roundtrip".into(),
        path.to_string_lossy().into_owned(),
        "1048840".into(),
        "uncached".into(),
    ];
    let a = bench::run(&args).unwrap();
    assert_eq!(a.valid_bytes, 1048840);
    assert_eq!(a.pinned_bytes, 0);
    assert!(a.physical_bytes.is_none());
    assert!(bench::run(&args).is_err());
    args[1] = "restore".into();
    let b = bench::run(&args).unwrap();
    assert_eq!(a.payload_checksum, b.payload_checksum);
    #[cfg(target_os = "macos")]
    {
        assert_eq!(a.backend_actual, "macos-f-nocache-development-fallback");
        assert_eq!(a.fallbacks, 1);
    }
    args[1] = "h2d".into();
    assert!(bench::run(&args).is_err());
}
#[test]
fn direct_alignment_and_short_eof_fail_closed() {
    use memra_tier::io::direct::AlignedFile;
    let dir = OwnedDirectory::new();
    let path = dir.0.join("aligned");
    std::fs::write(&path, payload(4096)).unwrap();
    #[cfg(target_os = "macos")]
    let file = AlignedFile::from_configured_file(bench::open_uncached(&path).unwrap());
    #[cfg(not(target_os = "macos"))]
    let file = AlignedFile::open(&path).unwrap();
    let mut bytes = vec![0; 8191];
    let start = (4096 - bytes.as_ptr() as usize % 4096) % 4096;
    let dst = &mut bytes[start..start + 4096];
    assert_eq!(file.read_aligned(1, dst), Err(Error::InvalidLayout));
    assert_eq!(
        file.read_aligned(0, &mut dst[..264]),
        Err(Error::InvalidLayout)
    );
    assert_eq!(file.read_aligned(0, dst), Ok(4096));
    assert_eq!(dst, payload(4096));
    assert_eq!(
        file.read_aligned(4096, dst),
        Err(Error::ShortIo {
            expected: 4096,
            actual: 0
        })
    );
}

#[test]
fn partial_acceptance_retains_rejected_device_and_refuses_batch_publication() {
    use memra_tier::contracts::{CopyOp, DeviceOwner, ItemAcceptance, TransferOp};
    let (mut engine, pool) = transfers();
    let g = governor();
    let mut r = support::request(0, Priority::MandatoryActive);
    r.bytes.device[0] = 4096;
    let charge = g.borrow_mut().reserve(&r).unwrap();
    let mut owner = DeviceOwner::new(0);
    let device = owner
        .register(31, 4096, Box::new(vec![0u8; 4096]), &charge)
        .unwrap();
    let mut host = pool.acquire(264, Admission::Demand).unwrap();
    host.write(&payload(264)).unwrap();
    let op = CopyOp {
        host,
        device,
        bytes: 264,
        epochs: epochs(7),
        producer_fence: None,
    };
    let submission = engine
        .submit_batch(vec![TransferOp::NvmeRead(plan(&pool)), TransferOp::H2d(op)])
        .unwrap();
    submission.validate(2).unwrap();
    let ticket = submission.ticket;
    for entry in submission.items {
        if let ItemAcceptance::Rejected {
            item,
            op: TransferOp::H2d(op),
            error,
        } = entry
        {
            assert_eq!(item, 1);
            assert_eq!(error, Error::Unsupported);
            assert_eq!(op.host.bytes().unwrap(), payload(264));
            owner.release(&op.device).unwrap();
            drop(op);
        }
    }
    g.borrow_mut().release(&charge).unwrap();
    engine.drive(&ticket).unwrap();
    let completion = engine.poll(&ticket).unwrap();
    assert!(completion.items[0].accepted);
    assert!(!completion.items[1].accepted);
    assert!(matches!(
        engine.take_destination(&ticket, 0, ticket.epochs),
        Err(Error::Rejected)
    ));
    engine.retire(&ticket, None).unwrap();
    engine.acknowledge(&ticket).unwrap();
    assert_eq!(pool.accounting().free_slots, 4);
}
#[test]
fn corrupt_backing_does_not_trap_explicit_charge_release() {
    let mut store = new_store(MemoryBackend::default());
    let mut txn = store.begin(key(), 264, Durability::Ephemeral).unwrap();
    store.put(&mut txn, &payload(264)).unwrap();
    let m = store.commit(&mut txn).unwrap();
    let mut request = support::request(0, Priority::MandatoryActive);
    request.bytes.nvme = 32768;
    let lease = store.lease(&m, &request).unwrap();
    store.backend_mut().blobs.retain(|(root, _), _| *root);
    assert_eq!(store.read(&lease, 0, &mut [0; 264]), Err(Error::NotFound));
    store.release(&lease).unwrap();
}
#[test]
fn root_metadata_requires_canonical_v1_and_distinct_digest_domains() {
    use memra_tier::contracts::digest as framed;
    assert_ne!(framed("extent", b"x"), framed("valid-bytes", b"x"));
    let mut s = new_store(MemoryBackend::default());
    let mut txn = s.begin(key(), 264, Durability::Ephemeral).unwrap();
    s.put(&mut txn, &payload(264)).unwrap();
    s.commit(&mut txn).unwrap();
    let root = s
        .backend_mut()
        .blobs
        .iter_mut()
        .find(|((r, _), _)| *r)
        .unwrap()
        .1;
    let mut metadata = decode_extent(root).unwrap().to_vec();
    metadata.push(b' ');
    *root = encode_extent(&metadata).unwrap();
    assert_eq!(s.lookup(&key()), Err(Error::Corrupt));
}

struct PinnedFixture {
    pool: Option<FakePinnedPool>,
    gov: SharedGovernor,
    charge: memra_tier::contracts::ChargedLease,
}
impl PinnedFixture {
    fn new() -> Self {
        let gov = governor();
        let charge = gov
            .borrow_mut()
            .reserve(&support::request(
                2 * (4096 + 4095),
                Priority::MandatoryActive,
            ))
            .unwrap();
        let pool = Some(FakePinnedPool::new(2, 4096, 4096, 1, &charge).unwrap());
        Self { pool, gov, charge }
    }
}
impl conformance::PinnedFixture for PinnedFixture {
    type Host = memra_tier::pool::PinnedLease;
    fn acquire(&mut self, optional: bool) -> memra_tier::contracts::Result<Self::Host> {
        self.pool.as_ref().unwrap().acquire(
            264,
            if optional {
                Admission::Optional
            } else {
                Admission::Demand
            },
        )
    }
    fn initialize(
        &mut self,
        host: &mut Self::Host,
        bytes: &[u8],
    ) -> memra_tier::contracts::Result<()> {
        host.write(bytes)
    }
    fn release(&mut self, host: Self::Host) {
        host.release();
    }
    fn quarantine(&mut self, host: Self::Host) {
        host.quarantine();
    }
    fn free_slots(&self) -> usize {
        self.pool.as_ref().unwrap().accounting().free_slots
    }
    fn used(&self) -> memra_tier::contracts::TierBudget {
        self.gov.borrow().used()
    }
    fn release_backing(&mut self) -> memra_tier::contracts::Result<()> {
        self.gov.borrow_mut().release(&self.charge)
    }
    fn close_pool(&mut self) {
        self.pool.take();
    }
}
#[test]
fn revision_v11_pinned_lease() {
    conformance::pinned_lease(&mut PinnedFixture::new());
}
#[test]
fn revision_v11_pinned_quarantine() {
    conformance::pinned_quarantine(&mut PinnedFixture::new());
}
#[test]
fn revision_v11_object_publish_release() {
    let directory = OwnedDirectory::new();
    let mut s = new_store(FileBackend::open_mode(&directory.0, Durability::Persistent).unwrap());
    let mut request = support::request(0, Priority::MandatoryActive);
    request.bytes.nvme = 32768;
    conformance::object_publish_release(&mut s, key(), request);
}
#[test]
fn revision_v11_transfer_complete_cancel() {
    let (mut engine, pool) = transfers();
    let ticket = engine.nvme_read(plan(&pool)).unwrap();
    conformance::transfer_complete_cancel(
        &mut engine,
        ticket,
        &[vec![memra_tier::contracts::SegmentExpectation {
            valid_bytes: 264,
            io_bytes: 8192,
            checksum: digest(&payload(264)),
        }]],
        |e, t| e.drive(t).unwrap(),
    );
    engine.retire(&ticket, None).unwrap();
    engine.acknowledge(&ticket).unwrap();
    assert_eq!(pool.accounting().free_slots, 4);
}
