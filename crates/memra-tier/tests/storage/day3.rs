use super::*;
use memra_tier::contracts::{BudgetRequest, Deadline, Digest, TierBudget};
use memra_tier::io::direct::{AlignedBuffer, AlignedFile};
use memra_tier::tier::Governor;

#[test]
fn common_governor_preserves_mandatory_pool_and_object_headroom() {
    let mut cap = TierBudget::zero(0);
    cap.pageable = 2 * 8191;
    cap.nvme = 32768;
    let mut headroom = TierBudget::zero(0);
    headroom.pageable = 8191;
    headroom.nvme = 16384;
    let g = std::rc::Rc::new(std::cell::RefCell::new(
        Governor::new(cap, headroom, 4, 0, Arc::new(|| 0)).unwrap(),
    ));
    let mut r = BudgetRequest {
        bytes: TierBudget::zero(0),
        priority: Priority::OptionalPrefetch,
        deadline: Deadline(100),
        tenant: [0; 32],
    };
    r.bytes.pageable = 8191;
    let optional = g.borrow_mut().reserve(&r).unwrap();
    let optional_pool = FakePinnedPool::new(1, 4096, 4096, 0, &optional).unwrap();
    assert!(matches!(g.borrow_mut().reserve(&r), Err(Error::Capacity)));
    r.priority = Priority::MandatoryActive;
    let mandatory = g.borrow_mut().reserve(&r).unwrap();
    let mandatory_pool = FakePinnedPool::new(1, 4096, 4096, 0, &mandatory).unwrap();
    assert!(matches!(g.borrow_mut().reserve(&r), Err(Error::Capacity)));
    assert_eq!(g.borrow_mut().release(&mandatory), Err(Error::Busy));
    drop(mandatory_pool);
    drop(optional_pool);
    g.borrow_mut().release(&mandatory).unwrap();
    g.borrow_mut().release(&optional).unwrap();
    let mut s = ExtentStore::new(MemoryBackend::default(), g.clone());
    let mut t = s.begin(key(), 264, Durability::Ephemeral).unwrap();
    s.put(&mut t, &payload(264)).unwrap();
    let m = s.commit(&mut t).unwrap();
    r.bytes.pageable = 0;
    r.bytes.nvme = 16384;
    r.priority = Priority::OptionalPrefetch;
    let opt_object = s.lease(&m, &r).unwrap();
    assert!(matches!(s.lease(&m, &r), Err(Error::Capacity)));
    r.priority = Priority::MandatoryActive;
    let active_object = s.lease(&m, &r).unwrap();
    assert!(matches!(s.lease(&m, &r), Err(Error::Capacity)));
    s.release(&active_object).unwrap();
    s.release(&opt_object).unwrap();
    assert_eq!(g.borrow().used(), TierBudget::zero(0));
}

#[test]
fn aligned_extent_io_rejects_misalignment_and_truncation() {
    let dir = OwnedDirectory::new();
    let path = dir.0.join("extent");
    std::fs::write(&path, vec![7; 8192]).unwrap();
    let file = AlignedFile::from_configured_file(
        std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(&path)
            .unwrap(),
    );
    let mut buffer = AlignedBuffer::new(8192).unwrap();
    assert_eq!(buffer.bytes().as_ptr() as usize % 4096, 0);
    assert_eq!(
        file.read_aligned(1, buffer.bytes_mut()),
        Err(Error::InvalidLayout)
    );
    assert_eq!(
        file.read_aligned(0, &mut buffer.bytes_mut()[1..]),
        Err(Error::InvalidLayout)
    );
    assert_eq!(
        file.write_aligned(1, buffer.bytes()),
        Err(Error::InvalidLayout)
    );
    assert_eq!(
        file.write_aligned(0, &buffer.bytes()[..4095]),
        Err(Error::InvalidLayout)
    );
    assert_eq!(file.read_aligned(0, buffer.bytes_mut()), Ok(8192));
    buffer.bytes_mut().fill(3);
    assert_eq!(file.write_aligned(0, buffer.bytes()), Ok(8192));
    file.sync_all().unwrap();
    assert_eq!(std::fs::read(&path).unwrap(), vec![3; 8192]);
    std::fs::OpenOptions::new()
        .write(true)
        .open(&path)
        .unwrap()
        .set_len(4097)
        .unwrap();
    assert_eq!(
        file.read_aligned(0, buffer.bytes_mut()),
        Err(Error::ShortIo {
            expected: 8192,
            actual: 4097
        })
    );
    assert!(matches!(AlignedBuffer::new(0), Err(Error::InvalidLayout)));
    assert!(matches!(
        AlignedBuffer::new(MAX_CHUNK + 8192),
        Err(Error::Capacity)
    ));
}

#[test]
fn direct_mode_is_explicit_and_never_silently_falls_back() {
    let dir = OwnedDirectory::new();
    let mut backend = FileBackend::open(&dir.0).unwrap();
    #[cfg(not(target_os = "linux"))]
    {
        assert_eq!(backend.set_direct_writes(true), Err(Error::Unsupported));
        assert!(matches!(
            AlignedFile::create_new(&dir.0.join("direct")),
            Err(Error::Unsupported)
        ));
    }
    #[cfg(target_os = "linux")]
    {
        backend.set_direct_writes(true).unwrap();
        backend.set_read_mode(memra_tier::io::direct::ReadMode::Uncached);
        let mut s = new_store(backend);
        let mut t = s.begin(key(), 4097, Durability::Ephemeral).unwrap();
        s.put(&mut t, &payload(4097)).unwrap();
        let m = s.commit(&mut t).unwrap();
        assert_eq!(read_manifest(&mut s, &m, 0).unwrap(), payload(4097));
        assert!(!std::fs::read_dir(&dir.0).unwrap().any(|p| {
            p.unwrap()
                .file_name()
                .to_string_lossy()
                .starts_with(".pending")
        }));
    }
}

#[derive(Default)]
struct CountedBackend {
    inner: MemoryBackend,
    reads: std::cell::RefCell<Vec<(bool, Digest)>>,
}
impl BlobBackend for CountedBackend {
    fn get(&self, root: bool, id: Digest, max: usize) -> Result<Option<Vec<u8>>> {
        self.reads.borrow_mut().push((root, id));
        self.inner.get(root, id, max)
    }
    fn insert(&mut self, root: bool, id: Digest, bytes: &[u8]) -> Result<()> {
        self.inner.insert(root, id, bytes)
    }
}
#[test]
fn extent_lease_is_bounded_lazy_charged_and_rechecks_corruption() {
    let g = governor();
    let mut store = ExtentStore::new(CountedBackend::default(), g.clone());
    let mut t = store.begin(key(), 3 * 4096, Durability::Ephemeral).unwrap();
    for b in [1, 2, 3] {
        store.put(&mut t, &vec![b; 4096]).unwrap();
    }
    let m = store.commit(&mut t).unwrap();
    store.backend().reads.borrow_mut().clear();
    let r = support::request(0, Priority::MandatoryActive);
    let mut r = r;
    r.bytes.nvme = 16384;
    assert_eq!(store.lookup(&key()).unwrap(), Some(m.clone()));
    assert!(store.backend().reads.borrow().iter().all(|(root, _)| *root));
    let lease = store.lease_extent(&m, 1, &r).unwrap();
    assert_eq!(g.borrow().used().nvme, 16384);
    let chunks: Vec<_> = store
        .backend()
        .reads
        .borrow()
        .iter()
        .filter(|(root, _)| !root)
        .map(|(_, id)| *id)
        .collect();
    assert_eq!(chunks, [m.chunks[1].encoded_digest]);
    assert!(matches!(store.lease(&m, &r), Err(Error::Capacity)));
    // An unrelated chunk can disappear without breaking the selected extent.
    store
        .backend_mut()
        .inner
        .blobs
        .remove(&(false, m.chunks[0].encoded_digest));
    let mut bytes = vec![0; 4096];
    assert_eq!(store.read_extent(&lease, &mut bytes), Ok(4096));
    assert_eq!(bytes, vec![2; 4096]);
    assert!(matches!(store.lease(&m, &r), Err(Error::NotFound)));
    assert_eq!(store.evict(&key()), Err(Error::Busy));
    store
        .backend_mut()
        .inner
        .blobs
        .get_mut(&(false, m.chunks[1].encoded_digest))
        .unwrap()[ALIGNMENT] ^= 1;
    assert_eq!(store.read_extent(&lease, &mut bytes), Err(Error::Corrupt));
    let used = g.borrow().used();
    assert!(matches!(store.lease_extent(&m, 1, &r), Err(Error::Corrupt)));
    assert_eq!(g.borrow().used(), used); // failed validation returns its admission
    store.release_extent(&lease).unwrap();
    assert_eq!(g.borrow().used().nvme, 0);
    assert_eq!(
        store.read_extent(&lease, &mut bytes),
        Err(Error::AlreadyReleased)
    );
    assert_eq!(store.release_extent(&lease), Err(Error::AlreadyReleased));
}

type FileTransfers = memra_tier::io::transfer::CpuTransfers<ExtentStore<FileBackend, Governor>>;
fn background_fixture(
    dir: &OwnedDirectory,
    limit: usize,
) -> (
    FileTransfers,
    FakePinnedPool,
    SharedGovernor,
    ObjectManifest,
) {
    use memra_tier::io::transfer::CpuTransfers;
    let g = governor();
    let mut store = ExtentStore::new(FileBackend::open(&dir.0).unwrap(), g.clone());
    let mut t = store.begin(key(), 264, Durability::Ephemeral).unwrap();
    store.put(&mut t, &payload(264)).unwrap();
    let manifest = store.commit(&mut t).unwrap();
    let mut request = support::request(16384, Priority::MandatoryActive);
    request.bytes.nvme = 16384;
    let mut qr = support::request(0, Priority::MandatoryActive);
    qr.bytes.inflight = limit as u64;
    let charge = g.borrow_mut().reserve(&qr).unwrap();
    let mut engine = CpuTransfers::new(store, request, limit, &charge).unwrap();
    engine.enable_background(1).unwrap();
    let pool_charge = g
        .borrow_mut()
        .reserve(&support::request(
            ((limit + 1) * 8191) as u64,
            Priority::MandatoryActive,
        ))
        .unwrap();
    let pool = FakePinnedPool::new(limit + 1, 4096, 4096, 0, &pool_charge).unwrap();
    (engine, pool, g, manifest)
}
fn read_plan(
    pool: &FakePinnedPool,
    chunk: u32,
) -> memra_tier::contracts::ReadPlan<memra_tier::pool::PinnedLease> {
    memra_tier::contracts::ReadPlan {
        object: key(),
        chunk,
        destination: pool.acquire(264, Admission::Demand).unwrap(),
        epochs: epochs(7),
    }
}
fn await_completion(
    engine: &mut FileTransfers,
    t: &TransferTicket,
) -> memra_tier::contracts::Completion {
    use memra_tier::contracts::TransferEngine;
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    loop {
        let c = engine.poll(t).unwrap();
        if c.producer_done {
            return c;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "bounded worker completion timeout"
        );
        std::thread::yield_now();
    }
}
#[test]
fn background_bytes_progress_without_drive_and_retire_after_consumer() {
    use memra_tier::contracts::{CancelState, Destination, TransferEngine};
    let dir = OwnedDirectory::new();
    let (mut engine, pool, g, _) = background_fixture(&dir, 1);
    let t = engine.nvme_read(read_plan(&pool, 0)).unwrap();
    assert_eq!(g.borrow().used().nvme, 16384);
    assert_eq!(
        g.borrow().used().pageable,
        12287 + pool.accounting().backing_bytes as u64
    ); // framed extent + alignment overhead
    assert!(!engine.retired(&t).unwrap());
    let rejected = engine.nvme_read(read_plan(&pool, 0)).unwrap_err();
    assert_eq!(rejected.error, Error::Capacity);
    drop(rejected);
    let c = await_completion(&mut engine, &t);
    c.require(
        &t,
        &[vec![memra_tier::contracts::SegmentExpectation {
            valid_bytes: 264,
            io_bytes: 8192,
            checksum: digest(&payload(264)),
        }]],
        false,
    )
    .unwrap();
    assert!(engine.ready_view(&t, 0, t.epochs).is_err());
    let Destination::Host(host) = engine.take_destination(&t, 0, t.epochs).unwrap() else {
        panic!()
    };
    assert_eq!(host.bytes().unwrap(), payload(264));
    assert_eq!(engine.cancel(&t), Ok(CancelState::AlreadyPublished));
    assert_eq!(engine.retire(&t, None), Err(Error::Busy));
    assert_eq!(g.borrow().used().nvme, 16384);
    host.release();
    engine.retire(&t, None).unwrap();
    assert_eq!(g.borrow().used().nvme, 0);
    assert_eq!(
        g.borrow().used().pageable,
        pool.accounting().backing_bytes as u64
    );
    assert!(engine.retired(&t).unwrap());
    assert_eq!(
        engine.nvme_read(read_plan(&pool, 0)).unwrap_err().error,
        Error::Capacity
    ); // tombstone still bounded
    engine.acknowledge(&t).unwrap();
    assert_eq!(engine.retired(&t), Err(Error::UnknownTicket));
    assert_eq!(pool.accounting().free_slots, 2);
}
#[test]
fn background_cancel_corrupt_and_failed_siblings_never_publish() {
    use memra_tier::contracts::{CancelState, ItemStatus, TransferEngine, TransferOp};
    let dir = OwnedDirectory::new();
    let (mut engine, pool, g, m) = background_fixture(&dir, 2);
    let t = engine.nvme_read(read_plan(&pool, 0)).unwrap();
    assert_eq!(engine.cancel(&t), Ok(CancelState::PublicationRevoked));
    assert!(!engine.retired(&t).unwrap());
    await_completion(&mut engine, &t);
    assert!(matches!(
        engine.take_destination(&t, 0, t.epochs),
        Err(Error::Cancelled)
    ));
    engine.retire(&t, None).unwrap();
    engine.acknowledge(&t).unwrap();
    assert_eq!(g.borrow().used().nvme, 0);
    let path = dir
        .0
        .join(format!("chunk-{}", hex(&m.chunks[0].encoded_digest)));
    let mut corrupt = std::fs::read(&path).unwrap();
    corrupt[ALIGNMENT] ^= 1;
    std::fs::write(path, corrupt).unwrap();
    let batch = engine
        .submit_batch(vec![
            TransferOp::NvmeRead(read_plan(&pool, 0)),
            TransferOp::NvmeRead(read_plan(&pool, 99)),
        ])
        .unwrap();
    batch.validate(2).unwrap();
    let c = await_completion(&mut engine, &batch.ticket);
    assert_eq!(c.items.len(), 2);
    assert_eq!(c.items[0].segments[0].error, Some(Error::Corrupt));
    assert_eq!(c.items[1].segments[0].error, Some(Error::InvalidLayout));
    assert!(
        c.items
            .iter()
            .all(|i| i.segments[0].status == ItemStatus::Failed)
    );
    assert!(
        engine
            .take_destination(&batch.ticket, 0, batch.ticket.epochs)
            .is_err()
    );
    engine.retire(&batch.ticket, None).unwrap();
    engine.acknowledge(&batch.ticket).unwrap();
    assert_eq!(g.borrow().used().nvme, 0);
    assert_eq!(pool.accounting().free_slots, 3);
}

#[test]
fn background_runs_frozen_cancellation_and_post_completion_schedule() {
    use memra_tier::contracts::{SegmentExpectation, TransferEngine};
    let dir = OwnedDirectory::new();
    let (mut engine, pool, _, _) = background_fixture(&dir, 1);
    let ticket = engine.nvme_read(read_plan(&pool, 0)).unwrap();
    conformance::transfer_cancel(&mut engine, ticket, |e, t| {
        await_completion(e, t);
        e.retire(t, None).unwrap();
    });
    let ticket = engine.nvme_read(read_plan(&pool, 0)).unwrap();
    conformance::transfer_complete_cancel(
        &mut engine,
        ticket,
        &[vec![SegmentExpectation {
            valid_bytes: 264,
            io_bytes: 8192,
            checksum: digest(&payload(264)),
        }]],
        |e, t| {
            await_completion(e, t);
        },
    );
    engine.retire(&ticket, None).unwrap();
    engine.acknowledge(&ticket).unwrap();
    assert_eq!(pool.accounting().free_slots, 2);
}
