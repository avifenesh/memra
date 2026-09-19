use super::*;
use crate::contracts::{Deadline, Priority, TierBudget};
use std::sync::Arc;

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
