use super::*;
use memra_tier::io::retirement::FakeTransferLease;

#[test]
fn cancelled_transfer_needs_disk_dma_and_consumer_retirement() {
    for missing in 0..3 {
        let pool = FakePinnedPool::new(1, 4096, 4096, 0).unwrap();
        let mut transfer = FakeTransferLease::new(pool.acquire(264, Admission::Demand).unwrap(), 7);
        transfer.cancel();
        assert_eq!(
            transfer.observe(6, true, true, true),
            Err(Error::StaleEpoch)
        );
        transfer
            .observe(7, missing != 0, missing != 1, missing != 2)
            .unwrap();
        assert!(!transfer.publication_allowed());
        assert!(!transfer.retired());
        assert_eq!(transfer.release(), Err(Error::NotReady));
        assert_eq!(pool.accounting().leased_bytes, 4096);
        transfer.observe(7, true, true, true).unwrap();
        assert!(transfer.retired());
        transfer.release().unwrap();
        assert_eq!(pool.accounting().free_slots, 1);
    }
}
#[test]
fn lost_transfer_completion_is_quarantined_on_drop() {
    let pool = FakePinnedPool::new(1, 4096, 4096, 0).unwrap();
    drop(FakeTransferLease::new(
        pool.acquire(264, Admission::Demand).unwrap(),
        7,
    ));
    assert_eq!(pool.accounting().quarantined_bytes, 4096);
    assert_eq!(pool.accounting().free_slots, 0);
}
