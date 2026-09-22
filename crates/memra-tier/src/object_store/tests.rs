use super::*;
use crate::contracts::{Deadline, Priority, TierBudget};
use std::sync::Arc;

#[cfg(unix)]
#[test]
fn gc_gate_scope_ends_even_while_a_child_holds_copied_descriptors() {
    use std::process::{Command, Stdio};
    use std::time::Duration;

    let directory = std::env::temp_dir().join(format!("memra-gc-fd-copy-{}", std::process::id()));
    fs::create_dir(&directory).unwrap();
    let old_backend = FileBackend::open(&directory).unwrap();
    let gate = lock_gc_gate(&directory).unwrap();
    // A concurrent fork before exec temporarily copies all open descriptors.
    // Keep those same open-file descriptions in a bounded child deliberately,
    // making the otherwise tiny close-on-exec window deterministic and safe.
    let mut child = Command::new("/bin/sh")
        .args(["-c", "printf ready; exec sleep 30"])
        .stdin(Stdio::from(gate.try_clone().unwrap()))
        .stderr(Stdio::from(old_backend.ownership.try_clone().unwrap()))
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    let mut stdout = child.stdout.take().unwrap();
    let (ready_tx, ready_rx) = std::sync::mpsc::sync_channel(1);
    let reader = std::thread::spawn(move || {
        let mut bytes = [0; 5];
        let result = stdout.read_exact(&mut bytes).map(|()| bytes);
        let _ = ready_tx.send(result);
    });
    let ready = ready_rx.recv_timeout(Duration::from_secs(5));
    let blocked_during_scope = matches!(FileBackend::open(&directory), Err(Error::Busy));
    drop(gate);
    drop(old_backend);
    // The GC gate must end with its critical section. In contrast, a copied
    // lifetime-ownership descriptor MUST continue to fence garbage collection.
    let mut reopened = FileBackend::open(&directory);
    let admitted = reopened.is_ok();
    let error = reopened.as_ref().err().map(|error| format!("{error:?}"));
    let lifetime_fenced = reopened
        .as_mut()
        .ok()
        .map(|backend| backend.evict_root([0; 32], EvictionPermit(HashSet::new())));
    if child.try_wait().unwrap().is_none() {
        child.kill().unwrap();
    }
    child.wait().unwrap();
    reader.join().unwrap();
    let after_child = reopened
        .as_mut()
        .ok()
        .map(|backend| backend.evict_root([0; 32], EvictionPermit(HashSet::new())));
    drop(reopened);
    fs::remove_dir_all(&directory).unwrap();
    // Cleanup precedes assertions, including in the deliberately red control.
    assert_eq!(ready.unwrap().unwrap(), *b"ready");
    assert!(blocked_during_scope);
    assert!(
        admitted,
        "descriptor copy extended completed GC scope: {error:?}"
    );
    assert_eq!(lifetime_fenced, Some(Err(Error::Busy)));
    assert_eq!(after_child, Some(Err(Error::NotFound)));
}

#[test]
fn review_gc_upgrade_window_keeps_other_stores_lease_fenced() {
    let directory = std::env::temp_dir().join(format!("memra-gc-upgrade-{}", std::process::id()));
    fs::create_dir(&directory).unwrap();
    let key = ObjectKey {
        version: 1,
        artifact: [1; 32],
        semantic_id: [2; 32],
        layout: [3; 32],
        generation: 1,
    };
    let mut victim = key.clone();
    victim.generation += 1;
    let mut cap = TierBudget::zero(1);
    cap.nvme = 1 << 20;
    let governor = Rc::new(RefCell::new(
        crate::tier::Governor::new(cap.clone(), TierBudget::zero(1), 16, 0, Arc::new(|| 0))
            .unwrap(),
    ));
    let request = BudgetRequest {
        bytes: cap,
        priority: Priority::MandatoryActive,
        deadline: Deadline(100),
        tenant: [0; 32],
    };
    let mut first = ExtentStore::new(FileBackend::open(&directory).unwrap(), governor.clone());
    for k in [key.clone(), victim.clone()] {
        let mut txn = first.begin(k, 3, Durability::Ephemeral).unwrap();
        first.put(&mut txn, b"abc").unwrap();
        first.commit(&mut txn).unwrap();
    }
    let head = first.lookup(&key).unwrap().unwrap();
    let mut second = ExtentStore::new(FileBackend::open(&directory).unwrap(), governor);
    let lease = second.lease_extent(&head, 0, &request).unwrap();
    assert_eq!(first.evict(&key), Err(Error::Busy));
    let mut interleaved = None;
    let mut admission_blocked = false;
    let result = second.backend_mut().evict_with_hook(
        victim.identity().unwrap(),
        EvictionPermit(HashSet::new()),
        || {
            interleaved = Some(first.evict(&key));
            admission_blocked = matches!(FileBackend::open(&directory), Err(Error::Busy));
        },
    );
    let mut bytes = [0; 3];
    let read = second.read_extent(&lease, &mut bytes);
    // Assert after cleanup so the deliberately red run leaves no scratch files.
    second.release_extent(&lease).unwrap();
    drop(first);
    drop(second);
    fs::remove_dir_all(&directory).unwrap();
    assert_eq!(interleaved, Some(Err(Error::Busy)));
    assert_eq!(result, Err(Error::Busy));
    assert!(admission_blocked);
    assert_eq!(read, Ok(3));
    assert_eq!(&bytes, b"abc");
}
