//! Day-35 rule (WP-A, memra#536 Move 2, `DAY35.md` section 2, design M1): a D2H batch whose items'
//! completion checksums are DEFERRED to the caller (a hash helper reads each landed destination off the
//! owner thread). The D2H mirror of the day-34 rule. Additive and unversioned beside the frozen
//! schedules: every v1 to v1.3 schedule is byte-identical and `WIRE_VERSION` stays 1. Nothing here
//! qualifies a backend by itself.
//!
//! Rule, the deferred D2H checksum:
//!
//! 1. One landing. A deferred item lands only with its checksum supplied: the ticket is not
//!    `producer_done` until every accepted item's copy completed AND its checksum was supplied.
//! 2. No view before the copy. A D2H destination holds its bytes only once its copy completed, so a
//!    view of a deferred item's destination is handed out only after every deferred item's copy is
//!    observed complete (`NotReady` before); nothing hashes bytes still in flight.
//! 3. The destination stays owned while it is read. While a view is out, taking the destinations back
//!    and retiring the ticket are `Busy`; every view comes back exactly once (a second return is
//!    `AlreadyReleased`).
//! 4. The receipt is the supplied digest. Each landed item's completion checksum is the digest supplied
//!    for it, the same term an undeferred item's engine checksum is; the bind's re-hash compares against
//!    it (a wrong digest is a receipt the re-hash refuses; that comparison is the caller's, as today).
//!
//! The red arm: a binding that hands a view of a destination whose copy has not landed. A second red
//! arm: a binding that lands a deferred item on its copy alone.
use crate::contracts::*;

/// The fixture: `submit` issues the D2H batch with its checksums deferred; `views` asks for one view per
/// accepted item (the engine's `NotReady` before every copy landed); `copies_complete` fires every
/// item's copy event; `supply` hands back every view with a digest per item (`wrong` flips item 0's);
/// `supply_again` returns the views a second time; `digest_for(k)` is the digest `supply` hands for item
/// `k` when not `wrong`; `take_back` and `retire` are the engine's answers.
pub trait D2hDeferredChecksumFixture {
    fn submit(&mut self) -> TransferTicket;
    fn views(&mut self, ticket: &TransferTicket) -> Result<usize>;
    fn copies_complete(&mut self);
    fn poll(&mut self, ticket: &TransferTicket) -> Result<Completion>;
    fn supply(&mut self, ticket: &TransferTicket, wrong: bool) -> Result<()>;
    fn supply_again(&mut self, ticket: &TransferTicket) -> Result<()>;
    fn digest_for(&self, item: usize) -> [u8; 32];
    fn take_back(&mut self, ticket: &TransferTicket) -> Result<()>;
    fn retire(&mut self, ticket: &TransferTicket) -> Result<()>;
}

/// The schedule: no view before the copies, no landing before the supply, the supplied digests are the
/// receipts, and the destinations stay owned while a view is out.
pub fn d2h_deferred_checksum_lands_with_its_digests<F: D2hDeferredChecksumFixture>(f: &mut F) {
    let ticket = f.submit();
    // 2. Before the copies: no view.
    assert_eq!(
        f.views(&ticket),
        Err(Error::NotReady),
        "no view of a destination whose copy is in flight"
    );
    let c = f.poll(&ticket).unwrap();
    assert!(!c.producer_done, "nothing landed at submit");
    // 1. The copies landed; the checksums did not: not landed.
    f.copies_complete();
    let c = f.poll(&ticket).unwrap();
    assert!(
        !c.producer_done,
        "a deferred item does not land on its copy alone"
    );
    let n = f
        .views(&ticket)
        .expect("every copy landed: one view per item");
    assert!(n > 0, "one view per accepted item");
    // 3. A view is out: the destinations stay with the engine.
    assert_eq!(f.take_back(&ticket), Err(Error::Busy), "a view is out");
    assert_eq!(f.retire(&ticket), Err(Error::Busy));
    f.supply(&ticket, false).unwrap();
    assert_eq!(
        f.supply_again(&ticket),
        Err(Error::AlreadyReleased),
        "views come back exactly once"
    );
    let c = f.poll(&ticket).unwrap();
    assert!(c.producer_done, "copies and checksums: landed");
    // 4. The receipts are the supplied digests.
    for (k, item) in c.items.iter().enumerate() {
        assert_eq!(
            item.segments[0].checksum,
            Some(f.digest_for(k)),
            "item {k}'s receipt is its supplied digest"
        );
    }
    f.take_back(&ticket).unwrap();
    f.retire(&ticket).unwrap();
}

/// Rule 4 with a wrong digest: it becomes item 0's receipt as supplied, never the engine's own, so the
/// bind's re-hash of the bytes finds the difference.
pub fn d2h_deferred_checksum_wrong_digest_is_the_receipt<F: D2hDeferredChecksumFixture>(f: &mut F) {
    let ticket = f.submit();
    f.copies_complete();
    f.views(&ticket).unwrap();
    f.supply(&ticket, true).unwrap();
    let c = f.poll(&ticket).unwrap();
    assert!(c.producer_done);
    assert_ne!(
        c.items[0].segments[0].checksum,
        Some(f.digest_for(0)),
        "the wrong digest is item 0's receipt, as supplied"
    );
}

/// The red arm: a view asked for before any copy landed must be refused `NotReady`.
pub fn d2h_deferred_checksum_no_view_before_the_copy<F: D2hDeferredChecksumFixture>(f: &mut F) {
    let ticket = f.submit();
    assert_eq!(
        f.views(&ticket),
        Err(Error::NotReady),
        "a view of a destination whose copy has not landed was handed out"
    );
}

/// The second red arm: the copies landed and no digest was supplied; a binding that calls that landed
/// fails.
pub fn d2h_deferred_checksum_copy_alone_is_not_landed<F: D2hDeferredChecksumFixture>(f: &mut F) {
    let ticket = f.submit();
    f.copies_complete();
    let c = f.poll(&ticket).unwrap();
    assert!(
        !c.producer_done,
        "a deferred D2H item landed on its copy alone, with no checksum"
    );
}
