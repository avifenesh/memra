#[allow(dead_code)]
#[path = "../contracts/conformance.rs"]
mod conformance;
mod retirement;
#[allow(dead_code)]
#[path = "../contracts/support.rs"]
mod support;
use memra_tier::contracts::{BudgetGovernor, Durability, Epochs, Priority, Wire};
fn epochs(state: u64) -> Epochs {
    Epochs {
        state,
        src_gen: 19,
        dst_gen: 31,
    }
}
type SharedGovernor = std::rc::Rc<std::cell::RefCell<memra_tier::tier::Governor>>;
fn governor() -> SharedGovernor {
    let mut cap = memra_tier::contracts::TierBudget::zero(2);
    cap.pageable = 1 << 30;
    cap.pinned = 1 << 30;
    cap.staging = 1 << 30;
    cap.device = vec![1 << 30; 2];
    cap.peer = vec![1 << 30; 2];
    cap.replicas = vec![1 << 30; 2];
    cap.nvme = 1 << 30;
    cap.inflight = 1 << 30;
    std::rc::Rc::new(std::cell::RefCell::new(
        memra_tier::tier::Governor::new(
            cap,
            memra_tier::contracts::TierBudget::zero(2),
            16,
            0,
            Arc::new(|| 0),
        )
        .unwrap(),
    ))
}
fn new_pool(slots: usize, size: usize, align: usize, reserved: usize) -> Result<FakePinnedPool> {
    let g = governor();
    let charge = g.borrow_mut().reserve(&support::request(
        (slots * (size + align - 1)) as u64,
        Priority::MandatoryActive,
    ))?;
    FakePinnedPool::new(slots, size, align, reserved, &charge)
}
fn new_store<B: BlobBackend>(backend: B) -> ExtentStore<B, memra_tier::tier::Governor> {
    ExtentStore::new(backend, governor())
}
fn read_manifest<B: BlobBackend>(
    s: &mut ExtentStore<B, memra_tier::tier::Governor>,
    m: &ObjectManifest,
    c: u32,
) -> Result<Vec<u8>> {
    let mut r = support::request(0, Priority::MandatoryActive);
    r.bytes.nvme = 1 << 26;
    let l = s.lease(m, &r)?;
    let mut out = vec![0; m.chunks[c as usize].valid_bytes as usize];
    let result = s.read(&l, c, &mut out);
    s.release(&l)?;
    result?;
    Ok(out)
}
use memra_tier::contracts::{Error, ObjectKey, Result, TransferTicket};
use memra_tier::io::{BoundedReader, ReadAt, ReadRequest, read_exact_at};
use memra_tier::object_store::*;
use memra_tier::pool::{Admission, FakePinnedPool};
use std::collections::HashMap;
use std::io;
use std::sync::{Arc, Mutex, mpsc};

fn payload(n: usize) -> Vec<u8> {
    (0..n)
        .map(|i| (i.wrapping_mul(17).wrapping_add(3) % 251) as u8)
        .collect()
}
fn key() -> ObjectKey {
    ObjectKey {
        version: 1,
        artifact: digest(b"fixture"),
        semantic_id: digest(b"record"),
        layout: digest(b"opaque-v1"),
        generation: 7,
    }
}
#[derive(Default)]
struct MemoryBackend {
    blobs: HashMap<(bool, [u8; 32]), Vec<u8>>,
    fail_chunk: bool,
    fail_root: bool,
    truncate_write: bool,
}
impl BlobBackend for MemoryBackend {
    fn get(&self, root: bool, id: [u8; 32], max_bytes: usize) -> Result<Option<Vec<u8>>> {
        let value = self.blobs.get(&(root, id));
        if value.is_some_and(|v| v.len() > max_bytes) {
            return Err(Error::Capacity);
        }
        Ok(value.cloned())
    }
    fn insert(&mut self, root: bool, id: [u8; 32], bytes: &[u8]) -> Result<()> {
        if self.fail_chunk && !root {
            return Err(io::Error::from_raw_os_error(28).into());
        } // ENOSPC Linux/macOS
        if self.fail_root && root {
            return Err(
                io::Error::new(io::ErrorKind::Interrupted, "injected before publication").into(),
            );
        }
        if let Some(old) = self.blobs.get(&(root, id)) {
            return if old == bytes {
                Ok(())
            } else {
                Err(Error::Conflict)
            };
        }
        self.blobs.insert(
            (root, id),
            if self.truncate_write {
                bytes[..bytes.len() - 1].to_vec()
            } else {
                bytes.to_vec()
            },
        );
        Ok(())
    }
}
#[test]
fn header_valid_vs_padded_and_full_integrity() {
    for n in [0, 1, 264, 288, 4095, 4096, 4097, MAX_CHUNK] {
        let input = payload(n);
        let encoded = encode_extent(&input).unwrap();
        assert_eq!(encoded.len(), ALIGNMENT + padded_len(n).unwrap());
        let header = ExtentHeader::decode(&encoded[..ALIGNMENT]).unwrap();
        assert_eq!(header.valid_bytes, n as u64);
        assert_eq!(header.payload_sha256, digest(&input));
        assert_eq!(decode_extent(&encoded).unwrap(), input);
    }
}
#[test]
fn corruption_header_payload_padding_truncation_and_future_version_refused() {
    let encoded = encode_extent(&payload(264)).unwrap();
    for at in [0, 16, 32, 100, ALIGNMENT - 1, ALIGNMENT, ALIGNMENT + 264] {
        let mut bad = encoded.clone();
        bad[at] ^= 1;
        assert!(decode_extent(&bad).is_err(), "mutation at {at} accepted");
    }
    assert!(decode_extent(&encoded[..encoded.len() - 1]).is_err());
    let mut future = encoded;
    future[8..12].copy_from_slice(&2u32.to_le_bytes());
    assert_eq!(decode_extent(&future), Err(Error::UnsupportedVersion(2)));
    assert_eq!(padded_len(usize::MAX), Err(Error::Overflow));
    assert_eq!(
        ExtentHeader {
            valid_bytes: 264,
            storage_bytes: 264,
            payload_sha256: [0; 32]
        }
        .encode(),
        Err(Error::InvalidLayout)
    );
}
#[test]
fn publication_is_last_and_streams_object_larger_than_pool() {
    let mut store = new_store(MemoryBackend::default());
    let mut txn = store.begin(key(), 4 * 264, Durability::Ephemeral).unwrap();
    let pool = new_pool(1, 4096, 4096, 1).unwrap();
    for i in 0..4 {
        store.put(&mut txn, &payload(264)).unwrap();
        assert!(
            store.lookup(&key()).unwrap().is_none(),
            "early root at chunk {i}"
        );
    }
    let object = store.commit(&mut txn).unwrap();
    assert_eq!(store.lookup(&key()).unwrap(), Some(object.clone()));
    for i in 0..4 {
        let mut lease = pool.acquire(264, Admission::Demand).unwrap();
        lease
            .write(&read_manifest(&mut store, &object, i).unwrap())
            .unwrap();
        assert_eq!(lease.bytes().unwrap(), payload(264));
    }
    assert_eq!(pool.accounting().free_slots, 1);
    // Byte object > pool usable bytes, using explicit chunk-at-a-time progress.
    let mut txn = store
        .begin(
            ObjectKey {
                generation: 8,
                ..key()
            },
            8192,
            Durability::Ephemeral,
        )
        .unwrap();
    for _ in 0..2 {
        store.put(&mut txn, &payload(4096)).unwrap();
    }
    let object = store.commit(&mut txn).unwrap();
    for i in 0..2 {
        let mut lease = pool.acquire(4096, Admission::Demand).unwrap();
        lease
            .write(&read_manifest(&mut store, &object, i).unwrap())
            .unwrap();
        assert_eq!(lease.bytes().unwrap(), payload(4096));
    }
}
#[test]
fn incomplete_enospc_short_write_and_interrupted_commit_never_publish() {
    let mut store = new_store(MemoryBackend::default());
    let mut txn = store.begin(key(), 264, Durability::Ephemeral).unwrap();
    assert_eq!(store.commit(&mut txn), Err(Error::NotReady));
    let mut txn = store.begin(key(), 264, Durability::Ephemeral).unwrap();
    store.backend_mut().fail_chunk = true;
    assert_eq!(
        store.put(&mut txn, &payload(264)),
        Err(Error::Io {
            kind: io::ErrorKind::StorageFull,
            os_code: Some(28)
        })
    );
    assert!(store.lookup(&key()).unwrap().is_none());
    store.backend_mut().fail_chunk = false;
    store.backend_mut().truncate_write = true;
    assert_eq!(store.put(&mut txn, &payload(264)), Err(Error::Corrupt));
    assert!(store.lookup(&key()).unwrap().is_none());
    // A failed write does not modify txn's high-water; a fresh backend can retry.
    *store.backend_mut() = MemoryBackend::default();
    store.put(&mut txn, &payload(264)).unwrap();
    store.backend_mut().fail_root = true;
    assert!(store.commit(&mut txn).is_err());
    assert!(store.lookup(&key()).unwrap().is_none());
}
#[test]
fn corrupt_or_missing_chunk_blocks_commit_and_read() {
    let mut store = new_store(MemoryBackend::default());
    let mut txn = store.begin(key(), 264, Durability::Ephemeral).unwrap();
    store.put(&mut txn, &payload(264)).unwrap();
    store.backend_mut().blobs.values_mut().next().unwrap()[ALIGNMENT] ^= 1;
    assert_eq!(store.commit(&mut txn), Err(Error::Corrupt));
    assert!(store.lookup(&key()).unwrap().is_none());
    *store.backend_mut() = MemoryBackend::default();
    let mut txn = store.begin(key(), 264, Durability::Ephemeral).unwrap();
    store.put(&mut txn, &payload(264)).unwrap();
    let obj = store.commit(&mut txn).unwrap();
    store.backend_mut().blobs.retain(|(root, _), _| *root);
    assert!(store.lookup(&key()).unwrap().is_some()); // advisory != readable
    assert_eq!(read_manifest(&mut store, &obj, 0), Err(Error::NotFound));
}
#[test]
fn root_collision_identity_and_immutable_conflicts_fail_closed() {
    let mut store = new_store(MemoryBackend::default());
    let mut txn = store.begin(key(), 264, Durability::Ephemeral).unwrap();
    store.put(&mut txn, &payload(264)).unwrap();
    store.commit(&mut txn).unwrap();
    let first = store
        .backend()
        .blobs
        .iter()
        .find(|((r, _), _)| *r)
        .unwrap()
        .1
        .clone();
    let wrong = ObjectKey {
        layout: [9; 32],
        ..key()
    };
    assert!(store.lookup(&wrong).unwrap().is_none());
    let mut empty = store
        .begin(wrong.clone(), 264, Durability::Ephemeral)
        .unwrap();
    store.put(&mut empty, &[8; 264]).unwrap();
    store.commit(&mut empty).unwrap();
    let second_root = store
        .backend_mut()
        .blobs
        .iter_mut()
        .find(|((r, _), b)| *r && **b != first)
        .unwrap()
        .1;
    *second_root = first;
    assert_eq!(store.lookup(&wrong), Err(Error::Corrupt));
    let mut conflict = store.begin(key(), 264, Durability::Ephemeral).unwrap();
    store.put(&mut conflict, &[7; 264]).unwrap();
    assert_eq!(store.commit(&mut conflict), Err(Error::Conflict));
}
#[test]
fn fake_pool_reserves_demand_headroom_and_charges_padding() {
    let pool = new_pool(2, 4096, 4096, 1).unwrap();
    let optional = pool.acquire(264, Admission::Optional).unwrap();
    assert!(matches!(
        pool.acquire(264, Admission::Optional),
        Err(Error::Capacity)
    ));
    let mut demand = pool.acquire(264, Admission::Demand).unwrap();
    assert_eq!(demand.bytes(), Err(Error::NotReady));
    demand.write(&payload(264)).unwrap();
    assert_eq!(demand.bytes().unwrap().as_ptr() as usize % 4096, 0);
    assert_eq!(pool.accounting().leased_bytes, 8192);
    assert_eq!(pool.accounting().backing_bytes, 2 * (4096 + 4095));
    assert_eq!(demand.storage_bytes(), 4096);
    drop(optional);
    drop(demand);
    let mut recycled = pool.acquire(264, Admission::Demand).unwrap();
    assert_eq!(recycled.bytes(), Err(Error::NotReady));
    assert_eq!(recycled.write(&[1; 263]), Err(Error::InvalidLayout));
    assert_eq!(recycled.bytes(), Err(Error::NotReady));
    recycled.write(&[2; 264]).unwrap();
    assert_eq!(recycled.bytes().unwrap(), &[2; 264]);
}
#[test]
fn unknown_completion_quarantines_not_reuses() {
    let pool = new_pool(1, 4096, 4096, 0).unwrap();
    pool.acquire(264, Admission::Demand).unwrap().quarantine();
    assert_eq!(pool.accounting().quarantined_bytes, 4096);
    assert_eq!(pool.accounting().free_slots, 0);
    assert!(matches!(
        pool.acquire(1, Admission::Demand),
        Err(Error::Capacity)
    ));
}
struct Fragmented {
    bytes: Vec<u8>,
    interrupt: Mutex<bool>,
}
impl ReadAt for Fragmented {
    fn read_at(&self, dst: &mut [u8], offset: u64) -> io::Result<usize> {
        let mut first = self.interrupt.lock().unwrap();
        if *first {
            *first = false;
            return Err(io::Error::from(io::ErrorKind::Interrupted));
        }
        let start = (offset as usize).min(self.bytes.len());
        let n = dst.len().min(7).min(self.bytes.len() - start);
        dst[..n].copy_from_slice(&self.bytes[start..start + n]);
        Ok(n)
    }
}
fn source(n: usize) -> Arc<dyn ReadAt> {
    Arc::new(Fragmented {
        bytes: payload(n),
        interrupt: Mutex::new(true),
    })
}
#[test]
fn exact_pread_eintr_partial_eof_overflow() {
    let source = source(97);
    let mut dst = [0; 37];
    assert_eq!(read_exact_at(source.as_ref(), &mut dst, 3), Ok(37));
    assert_eq!(&dst, &payload(97)[3..40]);
    assert_eq!(
        read_exact_at(source.as_ref(), &mut dst, 90),
        Err(Error::ShortIo {
            expected: 37,
            actual: 7
        })
    );
    assert_eq!(
        read_exact_at(source.as_ref(), &mut dst, u64::MAX),
        Err(Error::Overflow)
    );
}
#[test]
fn bounded_reader_partial_acceptance_and_short_read_visibility() {
    let pool = new_pool(3, 4096, 4096, 0).unwrap();
    let mut reader = BoundedReader::new(1, 2).unwrap();
    let request = |offset| ReadRequest {
        source: source(97),
        offset,
        lease: pool.acquire(37, Admission::Demand).unwrap(),
        epochs: epochs(3),
    };
    let a = reader.submit(request(3)).unwrap();
    let b = reader.submit(request(90)).unwrap();
    let rejected = reader.submit(request(0)).unwrap_err();
    assert_eq!(rejected.error, Error::Capacity);
    drop(rejected);
    let mut got = HashMap::new();
    for _ in 0..2 {
        let c = reader.wait().unwrap();
        got.insert(c.ticket, c);
    }
    assert_eq!(got[&a].result, Ok(37));
    assert_eq!(got[&a].lease.bytes().unwrap(), &payload(97)[3..40]);
    assert_eq!(
        got[&b].result,
        Err(Error::ShortIo {
            expected: 37,
            actual: 7
        })
    );
    assert_eq!(got[&b].lease.bytes(), Err(Error::NotReady));
    drop(got);
    assert_eq!(pool.accounting().free_slots, 3);
}
struct Delayed {
    entered: mpsc::Sender<()>,
    resume: Mutex<mpsc::Receiver<()>>,
}
impl ReadAt for Delayed {
    fn read_at(&self, dst: &mut [u8], _: u64) -> io::Result<usize> {
        self.entered.send(()).unwrap();
        self.resume.lock().unwrap().recv().unwrap();
        dst.fill(42);
        Ok(dst.len())
    }
}
#[test]
fn cancel_late_completion_keeps_source_and_slot_until_io_retires() {
    let pool = new_pool(1, 4096, 4096, 0).unwrap();
    let (entered, ack) = mpsc::channel();
    let (resume, wait) = mpsc::channel();
    let backend = Arc::new(Delayed {
        entered,
        resume: Mutex::new(wait),
    });
    let weak = Arc::downgrade(&backend);
    let mut reader = BoundedReader::new(1, 1).unwrap();
    let t = reader
        .submit(ReadRequest {
            source: backend,
            offset: 0,
            lease: pool.acquire(264, Admission::Demand).unwrap(),
            epochs: epochs(9),
        })
        .unwrap();
    ack.recv_timeout(std::time::Duration::from_secs(5)).unwrap();
    assert!(weak.upgrade().is_some());
    assert_eq!(
        reader.cancel(TransferTicket {
            epochs: epochs(10),
            ..t
        }),
        Err(Error::UnknownTicket)
    );
    reader.cancel(t).unwrap();
    assert!(reader.poll().unwrap().is_none());
    assert!(matches!(
        pool.acquire(1, Admission::Demand),
        Err(Error::Capacity)
    ));
    resume.send(()).unwrap();
    let completion = reader.wait().unwrap();
    assert!(completion.cancelled);
    assert_eq!(completion.result, Err(Error::Cancelled));
    assert_eq!(pool.accounting().free_slots, 0);
    drop(completion);
    assert_eq!(pool.accounting().free_slots, 1);
}
struct OwnedDirectory(std::path::PathBuf);
impl OwnedDirectory {
    fn new() -> Self {
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let p = std::env::temp_dir().join(format!(
            "memra-spill-a-test-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        std::fs::create_dir(&p).unwrap();
        Self(p)
    }
}
impl Drop for OwnedDirectory {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.0).unwrap();
    }
}
#[test]
fn real_filesystem_reopen_and_conflict_do_not_clobber() {
    let directory = OwnedDirectory::new();
    let mut store = new_store(FileBackend::open(&directory.0).unwrap());
    let mut t = store.begin(key(), 264, Durability::Ephemeral).unwrap();
    store.put(&mut t, &payload(264)).unwrap();
    store.commit(&mut t).unwrap();
    drop(store);
    let mut reopened = new_store(FileBackend::open(&directory.0).unwrap());
    let obj = reopened.lookup(&key()).unwrap().unwrap();
    assert_eq!(read_manifest(&mut reopened, &obj, 0).unwrap(), payload(264));
    let mut t = reopened.begin(key(), 264, Durability::Ephemeral).unwrap();
    reopened.put(&mut t, &[1; 264]).unwrap();
    assert_eq!(reopened.commit(&mut t), Err(Error::Conflict));
    assert_eq!(read_manifest(&mut reopened, &obj, 0).unwrap(), payload(264));
    assert!(std::fs::read_dir(&directory.0).unwrap().all(|p| {
        !p.unwrap()
            .file_name()
            .to_string_lossy()
            .starts_with(".pending")
    }));
}
#[test]
fn jsonl_escapes_strings_and_unknown_times_are_null() {
    let sample = memra_tier::telemetry::StorageSample {
        version: 1,
        fixture: "a\"b\nc".into(),
        backend_requested: "cpu-fixture".into(),
        backend_actual: "fake".into(),
        status: "exact".into(),
        valid_bytes: 264,
        padded_bytes: 4096,
        io_bytes: 8192,
        physical_bytes: None,
        queue_ns: None,
        io_ns: None,
        h2d_ns: None,
        d2h_ns: None,
        p2p_ns: None,
        total_ns: 42,
        inflight: 1,
        pinned_bytes: 0,
        pageable_bytes: Some(8191),
        fallbacks: 0,
        payload_checksum: digest(&payload(264)),
    };
    let line = String::from_utf8(sample.encode().unwrap()).unwrap();
    assert!(!line.contains('\n'));
    assert_eq!(
        serde_json::from_str::<memra_tier::telemetry::StorageSample>(&line).unwrap(),
        sample
    );
}

#[allow(dead_code)]
#[path = "../../../memra-engine/src/bin/storage_bench.rs"]
mod bench;
mod day2;

mod day3;

mod day4;

mod telemetry;
