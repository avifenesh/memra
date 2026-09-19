use super::*;
use memra_tier::io::retirement::FakeTransferLease;

#[test]
fn cancelled_transfer_needs_disk_dma_consumer_and_graph_retirement() {
    for missing in 0..4 {
        let pool = new_pool(1, 4096, 4096, 0).unwrap();
        let mut transfer =
            FakeTransferLease::new(pool.acquire(264, Admission::Demand).unwrap(), epochs(7));
        transfer.cancel();
        assert_eq!(
            transfer.observe(epochs(6), true, true, true, true),
            Err(Error::StaleEpoch)
        );
        transfer
            .observe(
                epochs(7),
                missing != 0,
                missing != 1,
                missing != 2,
                missing != 3,
            )
            .unwrap();
        assert!(!transfer.publication_allowed());
        assert!(!transfer.retired());
        assert_eq!(transfer.release(), Err(Error::NotReady));
        assert_eq!(pool.accounting().leased_bytes, 4096);
        transfer.observe(epochs(7), true, true, true, true).unwrap();
        assert!(transfer.retired());
        transfer.release().unwrap();
        assert_eq!(pool.accounting().free_slots, 1);
    }
}
#[test]
fn lost_transfer_completion_is_quarantined_on_drop() {
    let pool = new_pool(1, 4096, 4096, 0).unwrap();
    drop(FakeTransferLease::new(
        pool.acquire(264, Admission::Demand).unwrap(),
        epochs(7),
    ));
    assert_eq!(pool.accounting().quarantined_bytes, 4096);
    assert_eq!(pool.accounting().free_slots, 0);
}
