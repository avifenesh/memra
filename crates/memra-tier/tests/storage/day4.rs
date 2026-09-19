use super::*;
use std::{cell::RefCell, rc::Rc};

use memra_tier::contracts::{BudgetRequest, TierBudget};
use memra_tier::object_store::catalog::CatalogStore;

fn request(bytes: u64) -> BudgetRequest {
    let mut r = support::request(0, Priority::MandatoryActive);
    r.bytes.nvme = bytes;
    r
}
fn reference(bytes: &[u8]) -> ChunkRef {
    ChunkRef {
        version: 1,
        valid_bytes: bytes.len() as u64,
        storage_bytes: padded_len(bytes.len()).unwrap() as u64,
        encoded_digest: memra_tier::contracts::digest("extent", &encode_extent(bytes).unwrap()),
        checksum: digest(bytes),
    }
}
fn large_governor() -> SharedGovernor {
    let mut cap = TierBudget::zero(2);
    cap.nvme = 1 << 40;
    Rc::new(RefCell::new(
        memra_tier::tier::Governor::new(cap, TierBudget::zero(2), 16, 0, Arc::new(|| 0)).unwrap(),
    ))
}
#[test]
fn sharded_catalog_200gb_metadata_only_constant_touch_cost() {
    let dirs: Vec<_> = (0..4).map(|_| OwnedDirectory::new()).collect();
    let paths = dirs.iter().map(|d| d.0.clone()).collect();
    let mut store = CatalogStore::open(paths, large_governor()).unwrap();
    let count = 200_000u64; // 209,715,200,000 logical bytes; NO payload files.
    let chunk = reference(&payload(MAX_CHUNK));
    let head = store
        .install_index(
            key(),
            count * MAX_CHUNK as u64,
            std::iter::repeat_n(chunk.clone(), count as usize),
            &request(300_000_000_000),
        )
        .unwrap();
    let baseline = store.counters();
    let found = store.lookup(&key()).unwrap().unwrap();
    assert_eq!(found, head);
    assert_eq!(found.extents(), count);
    for i in [0, 77, 100_000, count - 1] {
        assert_eq!(store.shard_for(i), (i % 4) as usize);
        let lease = store.lease_extent(&found, i, &request(0)).unwrap();
        assert_eq!(lease.reference(), &chunk);
        store.release_extent(&lease).unwrap();
    }
    let after = store.counters();
    assert_eq!(after.root_reads - baseline.root_reads, 5);
    assert_eq!(after.index_rows - baseline.index_rows, 4);
    assert_eq!(after.payload_reads, 0);
    assert_eq!(
        after.metadata_bytes - baseline.metadata_bytes,
        5 * 8192 + 4 * 112
    );
    let file_bytes: u64 = dirs
        .iter()
        .flat_map(|d| std::fs::read_dir(&d.0).unwrap())
        .map(|e| e.unwrap().metadata().unwrap().len())
        .sum();
    assert_eq!(file_bytes, count * 112 + 8192);
    // No huge payload allocation, no payload reads, and <=23 MB actual index.
    assert!(file_bytes < 23_000_000);
}
#[test]
fn sharded_catalog_lazy_payload_corruption_missing_sibling_and_index_binding() {
    let dirs: Vec<_> = (0..3).map(|_| OwnedDirectory::new()).collect();
    let mut store =
        CatalogStore::open(dirs.iter().map(|d| d.0.clone()).collect(), governor()).unwrap();
    let bytes = payload(264);
    let head = store
        .install_index(key(), 792, vec![reference(&bytes); 3], &request(65536))
        .unwrap();
    store.put_extent(&head, 2, &bytes).unwrap();
    let lease = store.lease_extent(&head, 2, &request(0)).unwrap();
    let mut out = vec![0; 264];
    assert_eq!(store.read_extent(&lease, &mut out), Ok(264));
    assert_eq!(out, bytes);
    assert_eq!(store.evict(&head), Err(Error::Busy));
    let mut bad = std::fs::read(store.extent_path(&head, 2).unwrap()).unwrap();
    bad[ALIGNMENT] ^= 1;
    std::fs::write(store.extent_path(&head, 2).unwrap(), bad).unwrap();
    assert_eq!(store.read_extent(&lease, &mut out), Err(Error::Corrupt));
    store.release_extent(&lease).unwrap();
    let missing = store.lease_extent(&head, 1, &request(0)).unwrap();
    assert!(store.read_extent(&missing, &mut out).is_err());
    store.release_extent(&missing).unwrap();
    // Swapping equal-looking records across indices is detected by row checksum.
    let id = hex(&key().identity().unwrap());
    let row0 = dirs[0].0.join(format!("catalog-{id}-0.index"));
    let row1 = dirs[1].0.join(format!("catalog-{id}-1.index"));
    std::fs::copy(row0, row1).unwrap();
    assert!(matches!(
        store.lease_extent(&head, 1, &request(0)),
        Err(Error::Corrupt)
    ));
    store.evict(&head).unwrap();
    assert!(store.lookup(&key()).unwrap().is_none());
}
#[test]
fn catalog_gc_reference_count_charges_tombstone_crash_replay() {
    let d = OwnedDirectory::new();
    let g = governor();
    let mut store = CatalogStore::open(vec![d.0.clone()], g.clone()).unwrap();
    let head = store
        .install_index(key(), 264, vec![reference(&payload(264))], &request(32768))
        .unwrap();
    store.put_extent(&head, 0, &payload(264)).unwrap();
    let a = store.lease_extent(&head, 0, &request(0)).unwrap();
    let b = store.lease_extent(&head, 0, &request(0)).unwrap();
    let cap = request((1 << 30) - 32768);
    let remaining = g.borrow_mut().reserve(&cap).unwrap();
    assert!(matches!(
        g.borrow_mut().reserve(&request(1)),
        Err(Error::Capacity)
    ));
    store.release_extent(&a).unwrap();
    assert_eq!(store.evict(&head), Err(Error::Busy));
    store.release_extent(&b).unwrap();
    assert_eq!(store.release_extent(&b), Err(Error::AlreadyReleased));
    store.tombstone(&head).unwrap();
    assert!(store.lookup(&key()).unwrap().is_none());
    assert!(store.extent_path(&head, 0).unwrap().exists());
    // Simulated crash boundary: tombstone durable, unlink has not run.
    drop(store);
    let mut reopened = CatalogStore::open(vec![d.0.clone()], g.clone()).unwrap();
    assert!(reopened.lookup(&key()).unwrap().is_none());
    reopened.collect(&key()).unwrap();
    assert!(!reopened.extent_path(&head, 0).unwrap().exists());
    let full = g.borrow_mut().reserve(&request(32768)).unwrap();
    g.borrow_mut().release(&full).unwrap();
    g.borrow_mut().release(&remaining).unwrap();
    assert_eq!(reopened.collect(&key()), Err(Error::NotReady));
}
#[test]
fn catalog_gc_unknown_completion_retains_lock_and_charge() {
    let d = OwnedDirectory::new();
    let g = governor();
    let mut store = CatalogStore::open(vec![d.0.clone()], g.clone()).unwrap();
    let head = store
        .install_index(key(), 264, vec![reference(&payload(264))], &request(32768))
        .unwrap();
    {
        let _lost = store.lease_extent(&head, 0, &request(0)).unwrap();
    } // Lost handle is not retirement.
    assert_eq!(store.evict(&head), Err(Error::Busy));
    drop(store);
    assert!(matches!(
        CatalogStore::open(vec![d.0.clone()], g.clone()),
        Err(Error::Busy)
    ));
    assert!(matches!(
        g.borrow_mut().reserve(&request(1 << 30)),
        Err(Error::Capacity)
    ));
}
#[test]
fn catalog_install_quota_failure_cleans_only_owned_files() {
    let d = OwnedDirectory::new();
    let g = governor();
    let mut store = CatalogStore::open(vec![d.0.clone()], g.clone()).unwrap();
    assert!(matches!(
        store.install_index(key(), 264, vec![reference(&payload(264))], &request(1)),
        Err(Error::Capacity)
    ));
    assert!(store.lookup(&key()).unwrap().is_none());
    let head = store
        .install_index(key(), 264, vec![reference(&payload(264))], &request(32768))
        .unwrap();
    store.evict(&head).unwrap();
    let full = g.borrow_mut().reserve(&request(1 << 30)).unwrap();
    g.borrow_mut().release(&full).unwrap();
}
#[test]
fn legacy_gc_dedup_other_store_and_live_lease_fences() {
    let d = OwnedDirectory::new();
    let mut store = new_store(FileBackend::open_mode(&d.0, Durability::Persistent).unwrap());
    let mut second = key();
    second.generation += 1;
    for k in [key(), second.clone()] {
        let mut txn = store.begin(k, 264, Durability::Persistent).unwrap();
        store.put(&mut txn, &payload(264)).unwrap();
        store.commit(&mut txn).unwrap();
    }
    let manifest = store.lookup(&key()).unwrap().unwrap();
    let lease = store.lease_extent(&manifest, 0, &request(32768)).unwrap();
    assert_eq!(store.evict(&key()), Err(Error::Busy));
    store.release_extent(&lease).unwrap();
    let other = new_store(FileBackend::open(&d.0).unwrap());
    assert_eq!(store.evict(&key()), Err(Error::Busy));
    drop(other);
    store.evict(&key()).unwrap();
    assert!(store.lookup(&key()).unwrap().is_none());
    let manifest = store.lookup(&second).unwrap().unwrap();
    assert_eq!(
        read_manifest(&mut store, &manifest, 0).unwrap(),
        payload(264)
    );
    let chunk_path =
        d.0.join(format!("chunk-{}", hex(&manifest.chunks[0].encoded_digest)));
    assert!(chunk_path.exists());
    store.evict(&second).unwrap();
    assert!(!chunk_path.exists());
}
#[test]
fn legacy_gc_replays_crash_after_tombstone_and_before_unlink() {
    let d = OwnedDirectory::new();
    let mut store = new_store(FileBackend::open_mode(&d.0, Durability::Persistent).unwrap());
    let mut txn = store.begin(key(), 264, Durability::Persistent).unwrap();
    store.put(&mut txn, &payload(264)).unwrap();
    store.commit(&mut txn).unwrap();
    let id = hex(&key().identity().unwrap());
    std::fs::rename(
        d.0.join(format!("root-{id}")),
        d.0.join(format!("tomb-{id}")),
    )
    .unwrap();
    std::fs::File::open(&d.0).unwrap().sync_all().unwrap();
    drop(store);
    let mut store = new_store(FileBackend::open_mode(&d.0, Durability::Persistent).unwrap());
    assert!(store.lookup(&key()).unwrap().is_none());
    assert_eq!(store.commit(&mut txn), Err(Error::ForeignLease));
    store.evict(&key()).unwrap();
    assert_eq!(std::fs::read_dir(&d.0).unwrap().count(), 2); // lifetime + GC gate files
}

#[test]
fn legacy_gc_unknown_shutdown_retains_shared_ownership() {
    let d = OwnedDirectory::new();
    let g = governor();
    let mut store = ExtentStore::new(FileBackend::open(&d.0).unwrap(), g.clone());
    let mut txn = store.begin(key(), 264, Durability::Ephemeral).unwrap();
    store.put(&mut txn, &payload(264)).unwrap();
    let head = store.commit(&mut txn).unwrap();
    let _unknown = store.lease_extent(&head, 0, &request(32768)).unwrap();
    drop(store);
    assert!(g.borrow().used().nvme > 0);
    let mut other = new_store(FileBackend::open(&d.0).unwrap());
    assert_eq!(other.evict(&key()), Err(Error::Busy));
    assert!(other.lookup(&key()).unwrap().is_some());
}

#[test]
fn catalog_gc_process_exit_between_tombstone_and_unlink() {
    // Actual child process termination without Rust destructors. This tests OS
    // lock release + persisted restart state, NOT power-loss/controller durability.
    if let Some(path) = std::env::var_os("SPILL_A_TEST_CRASH_DIR") {
        let mut store = CatalogStore::open(vec![path.into()], governor()).unwrap();
        let head = store
            .install_index(key(), 264, vec![reference(&payload(264))], &request(32768))
            .unwrap();
        store.put_extent(&head, 0, &payload(264)).unwrap();
        store.tombstone(&head).unwrap();
        std::process::exit(73);
    }
    let d = OwnedDirectory::new();
    let status = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "day4::catalog_gc_process_exit_between_tombstone_and_unlink",
        ])
        .env("SPILL_A_TEST_CRASH_DIR", &d.0)
        .stdout(std::process::Stdio::null())
        .status()
        .unwrap();
    assert_eq!(status.code(), Some(73));
    let id = hex(&key().identity().unwrap());
    assert!(d.0.join(format!("catalog-{id}.tomb")).exists());
    assert!(d.0.join(format!("catalog-{id}-0.extent")).exists());
    let mut recovered = CatalogStore::open(vec![d.0.clone()], governor()).unwrap();
    assert!(recovered.lookup(&key()).unwrap().is_none());
    recovered.collect(&key()).unwrap();
    assert_eq!(std::fs::read_dir(&d.0).unwrap().count(), 1);
}

#[test]
fn review_catalog_recovers_pending_with_and_without_published_root() {
    for published in [false, true] {
        let d = OwnedDirectory::new();
        let mut store = CatalogStore::open(vec![d.0.clone()], governor()).unwrap();
        let head = store
            .install_index(key(), 264, vec![reference(&payload(264))], &request(32768))
            .unwrap();
        let root =
            d.0.join(format!("catalog-{}.root", hex(&key().identity().unwrap())));
        let pending = root.with_extension("pending");
        std::fs::hard_link(&root, &pending).unwrap();
        if !published {
            std::fs::remove_file(&root).unwrap();
        }
        drop(store);
        let mut recovered = CatalogStore::open(vec![d.0.clone()], governor()).unwrap();
        let result =
            recovered.install_index(key(), 264, vec![reference(&payload(264))], &request(32768));
        if published {
            assert_eq!(result, Err(Error::Conflict)); // Immutable root is never replaced.
            assert_eq!(recovered.lookup(&key()).unwrap(), Some(head.clone()));
        } else {
            assert_eq!(result.unwrap(), head);
        }
        assert!(!pending.exists(), "stale pending must not wedge reuse");
        // Also exercise collect directly at the post-link crash boundary.
        std::fs::hard_link(&root, &pending).unwrap();
        recovered.tombstone(&head).unwrap();
        recovered.collect(&key()).unwrap();
        assert!(!pending.exists());
        recovered
            .install_index(key(), 264, vec![reference(&payload(264))], &request(32768))
            .unwrap();
    }
}

#[test]
fn review_gc_preserves_uncommitted_transaction_chunks() {
    let d = OwnedDirectory::new();
    let mut store = new_store(FileBackend::open(&d.0).unwrap());
    let mut victim = store.begin(key(), 264, Durability::Ephemeral).unwrap();
    store.put(&mut victim, &payload(264)).unwrap();
    store.commit(&mut victim).unwrap();
    let mut other = key();
    other.generation += 1;
    let mut pending = store
        .begin(other.clone(), 264, Durability::Ephemeral)
        .unwrap();
    store.put(&mut pending, &payload(264)).unwrap();
    store.evict(&key()).unwrap();
    let committed = store.commit(&mut pending).unwrap();
    assert_eq!(
        read_manifest(&mut store, &committed, 0).unwrap(),
        payload(264)
    );
    store.evict(&other).unwrap();
}

#[test]
fn review_gc_verification_failure_does_not_revoke_visibility() {
    let d = OwnedDirectory::new();
    let mut store = new_store(FileBackend::open(&d.0).unwrap());
    let mut txn = store.begin(key(), 264, Durability::Ephemeral).unwrap();
    store.put(&mut txn, &payload(264)).unwrap();
    let head = store.commit(&mut txn).unwrap();
    let corrupt = d.0.join(format!("root-{}", hex(&[99; 32])));
    std::fs::write(&corrupt, b"undecodable unrelated root").unwrap();
    assert_eq!(store.evict(&key()), Err(Error::Corrupt));
    assert_eq!(store.lookup(&key()).unwrap(), Some(head.clone()));
    assert_eq!(read_manifest(&mut store, &head, 0).unwrap(), payload(264));
    assert!(
        !d.0.join(format!("tomb-{}", hex(&key().identity().unwrap())))
            .exists()
    );
}

#[test]
fn review_gc_staged_transaction_process_exit_and_bounded_recovery() {
    // Child exits without destructors: the txn-scoped links must survive, but
    // their independent owner lock must not. No age/PID reuse heuristic is used.
    if let Some(path) = std::env::var_os("SPILL_A_TEST_TXN_CRASH_DIR") {
        let mut store = new_store(FileBackend::open(std::path::Path::new(&path)).unwrap());
        let mut other = key();
        other.generation += 1;
        let mut pending = store.begin(other, 265, Durability::Ephemeral).unwrap();
        store.put(&mut pending, &payload(265)).unwrap();
        std::process::exit(73);
    }
    let d = OwnedDirectory::new();
    let mut store = new_store(FileBackend::open(&d.0).unwrap());
    let mut txn = store.begin(key(), 264, Durability::Ephemeral).unwrap();
    store.put(&mut txn, &payload(264)).unwrap();
    let head = store.commit(&mut txn).unwrap();
    let status = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "day4::review_gc_staged_transaction_process_exit_and_bounded_recovery",
        ])
        .env("SPILL_A_TEST_TXN_CRASH_DIR", &d.0)
        .stdout(std::process::Stdio::null())
        .status()
        .unwrap();
    assert_eq!(status.code(), Some(73));
    let orphan = d.0.join(format!(
        "chunk-{}",
        hex(&reference(&payload(265)).encoded_digest)
    ));
    assert!(orphan.exists());
    // Include the earlier mkdir-before-owner crash, and more than one sweep's
    // quota. All empty abandoned dirs are safe; no filesystem clock involved.
    for i in 0..40 {
        std::fs::create_dir(d.0.join(format!(".txn-abandoned-{i}"))).unwrap();
    }
    let count = || {
        std::fs::read_dir(&d.0)
            .unwrap()
            .filter(|e| {
                e.as_ref()
                    .unwrap()
                    .file_name()
                    .to_string_lossy()
                    .starts_with(".txn-")
            })
            .count()
    };
    assert_eq!(count(), 41);
    store.evict(&key()).unwrap();
    assert_eq!(count(), 9); // at most 32 per collection, independent of order
    // A second real GC target allows the remaining bounded cleanup to run.
    let mut next = key();
    next.generation += 2;
    let mut txn = store
        .begin(next.clone(), 264, Durability::Ephemeral)
        .unwrap();
    store.put(&mut txn, &payload(264)).unwrap();
    store.commit(&mut txn).unwrap();
    store.evict(&next).unwrap();
    assert_eq!(count(), 0);
    assert!(!orphan.exists());
    assert!(store.lookup(&head.key).unwrap().is_none());
}

#[test]
fn review_gc_cancel_removes_staging_and_corrupt_staging_is_noop() {
    let d = OwnedDirectory::new();
    let mut store = new_store(FileBackend::open(&d.0).unwrap());
    let mut txn = store.begin(key(), 264, Durability::Ephemeral).unwrap();
    store.put(&mut txn, &payload(264)).unwrap();
    let head = store.commit(&mut txn).unwrap();
    let mut next = key();
    next.generation += 1;
    let mut pending = store.begin(next, 264, Durability::Ephemeral).unwrap();
    store.put(&mut pending, &payload(264)).unwrap();
    let dir = std::fs::read_dir(&d.0)
        .unwrap()
        .map(|e| e.unwrap().path())
        .find(|p| {
            p.file_name()
                .unwrap()
                .to_string_lossy()
                .starts_with(".txn-")
        })
        .unwrap();
    assert!(
        dir.join(format!("chunk-{}", hex(&head.chunks[0].encoded_digest)))
            .exists()
    );
    std::fs::write(dir.join("unknown-reference"), b"corrupt metadata").unwrap();
    assert_eq!(store.evict(&key()), Err(Error::Corrupt));
    assert_eq!(store.lookup(&key()).unwrap(), Some(head.clone()));
    assert_eq!(read_manifest(&mut store, &head, 0).unwrap(), payload(264));
    std::fs::remove_file(dir.join("unknown-reference")).unwrap();
    assert_eq!(
        store.cancel(&mut pending),
        Ok(memra_tier::contracts::CancelState::PublicationRevoked)
    );
    assert!(!dir.exists());
    store.evict(&key()).unwrap();
    assert!(
        !d.0.join(format!("chunk-{}", hex(&head.chunks[0].encoded_digest)))
            .exists()
    );
}

#[test]
fn review_gc_live_cross_process_transaction_commits_byte_exact() {
    use std::io::{BufRead, Write};
    if let Some(path) = std::env::var_os("SPILL_A_TEST_TXN_LIVE_DIR") {
        let mut store = new_store(FileBackend::open(std::path::Path::new(&path)).unwrap());
        let mut next = key();
        next.generation += 1;
        let mut txn = store.begin(next, 264, Durability::Ephemeral).unwrap();
        store.put(&mut txn, &payload(264)).unwrap();
        println!("STAGED");
        std::io::stdout().flush().unwrap();
        let mut line = String::new();
        std::io::stdin().read_line(&mut line).unwrap();
        assert_eq!(line, "COMMIT\n");
        let head = store.commit(&mut txn).unwrap();
        assert_eq!(read_manifest(&mut store, &head, 0).unwrap(), payload(264));
        return;
    }
    let d = OwnedDirectory::new();
    let mut store = new_store(FileBackend::open(&d.0).unwrap());
    let mut txn = store.begin(key(), 264, Durability::Ephemeral).unwrap();
    store.put(&mut txn, &payload(264)).unwrap();
    store.commit(&mut txn).unwrap();
    let mut child = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "day4::review_gc_live_cross_process_transaction_commits_byte_exact",
            "--nocapture",
        ])
        .env("SPILL_A_TEST_TXN_LIVE_DIR", &d.0)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .spawn()
        .unwrap();
    let mut output = std::io::BufReader::new(child.stdout.take().unwrap());
    let mut line = String::new();
    loop {
        line.clear();
        if output.read_line(&mut line).unwrap() == 0 || line == "STAGED\n" {
            break;
        }
    }
    let result = store.evict(&key());
    let sent = child.stdin.take().unwrap().write_all(b"COMMIT\n");
    let status = child.wait().unwrap();
    assert_eq!(line, "STAGED\n");
    sent.unwrap();
    assert!(status.success());
    assert_eq!(result, Err(Error::Busy));
    store.evict(&key()).unwrap();
    let mut next = key();
    next.generation += 1;
    let head = store.lookup(&next).unwrap().unwrap();
    assert_eq!(read_manifest(&mut store, &head, 0).unwrap(), payload(264));
    store.evict(&next).unwrap();
}
