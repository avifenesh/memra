//! Day-34 rule (WP-A, memra#536 Move 2 owed item 1, `DAY34.md` design K): an H2D batch whose items'
//! completion checksums are DEFERRED to the caller (a hash helper reads each item's host source off the
//! owner thread). Additive and unversioned beside the frozen schedules: every v1 to v1.3 schedule is
//! byte-identical and `WIRE_VERSION` stays 1. Nothing here qualifies a backend by itself.
//!
//! Rule, the deferred checksum:
//!
//! 1. One landing. A deferred item lands only with its checksum supplied: the ticket is not
//!    `producer_done` until every accepted item's copy completed AND its checksum was supplied,
//!    whatever else completed.
//! 2. The receipt term is unchanged. A supplied checksum is the item's completion checksum and
//!    `Completion::require` against the demote-time receipts gates publication exactly as when the
//!    engine hashed the source itself: a supplied mismatch is `Corrupt`, nothing is published.
//! 3. The source stays owned while it is read. While a view of an item's host source is out, the
//!    engine keeps the source: retiring or recovering the ticket's sources and retiring the ticket are
//!    `Busy`. Every view comes back exactly once (a second return is `AlreadyReleased`).
//!
//! The red arm: a binding that lands a deferred item on its copy alone, with no checksum supplied.
use crate::contracts::*;

/// The fixture: `submit` issues the H2D batch; `defer` takes one view per accepted item;
/// `copies_complete` fires every item's copy event; `supply` hands back every view with the given
/// digest per item (`wrong` flips the digest of item 0); `supply_again` returns the views a second time;
/// `receipts` are the demote-time checksums; `retire_source` and `retire` are the engine's answers.
pub trait DeferredChecksumFixture {
    fn submit(&mut self) -> TransferTicket;
    fn defer(&mut self, ticket: &TransferTicket) -> Result<usize>;
    fn copies_complete(&mut self);
    fn poll(&mut self, ticket: &TransferTicket) -> Result<Completion>;
    fn supply(&mut self, ticket: &TransferTicket, wrong: bool) -> Result<()>;
    fn supply_again(&mut self, ticket: &TransferTicket) -> Result<()>;
    fn receipts(&self) -> Vec<Vec<SegmentExpectation>>;
    fn retire_source(&mut self, ticket: &TransferTicket) -> Result<()>;
    fn retire(&mut self, ticket: &TransferTicket) -> Result<()>;
}

/// The schedule, the matching digests: no landing before the supply, the gate open after it.
pub fn h2d_deferred_checksum_lands_with_its_digests<F: DeferredChecksumFixture>(f: &mut F) {
    let ticket = f.submit();
    let views = f
        .defer(&ticket)
        .expect("a live H2D batch defers its checksums");
    assert!(views > 0, "one view per accepted item");
    let c = f.poll(&ticket).unwrap();
    assert!(!c.producer_done, "nothing landed at submit");
    // 1. The copies landed; the checksums did not: not landed, and the sources stay owned.
    f.copies_complete();
    let c = f.poll(&ticket).unwrap();
    assert!(
        !c.producer_done,
        "a deferred item does not land on its copy alone"
    );
    assert_eq!(f.retire_source(&ticket), Err(Error::Busy), "a view is out");
    assert_eq!(f.retire(&ticket), Err(Error::Busy));
    // 2. Supplied: landed, and the demote-time receipts gate it as before.
    f.supply(&ticket, false).unwrap();
    assert_eq!(
        f.supply_again(&ticket),
        Err(Error::AlreadyReleased),
        "views come back exactly once"
    );
    let c = f.poll(&ticket).unwrap();
    assert!(c.producer_done, "copies and checksums: landed");
    assert!(
        c.require(&ticket, &f.receipts(), true).is_ok(),
        "matching digests pass the gate"
    );
    f.retire_source(&ticket).unwrap();
}

/// Rule 2: a supplied digest that differs from the demote-time receipt is `Corrupt` at the gate.
pub fn h2d_deferred_checksum_mismatch_is_corrupt<F: DeferredChecksumFixture>(f: &mut F) {
    let ticket = f.submit();
    f.defer(&ticket).unwrap();
    f.copies_complete();
    f.supply(&ticket, true).unwrap();
    let c = f.poll(&ticket).unwrap();
    assert!(c.producer_done);
    assert_eq!(
        c.require(&ticket, &f.receipts(), true),
        Err(Error::Corrupt),
        "a mismatching supplied digest is refused, never published"
    );
}

/// The red arm: the copies landed and no digest was supplied; a binding that calls that landed fails.
pub fn h2d_deferred_checksum_copy_alone_is_not_landed<F: DeferredChecksumFixture>(f: &mut F) {
    let ticket = f.submit();
    f.defer(&ticket).unwrap();
    f.copies_complete();
    let c = f.poll(&ticket).unwrap();
    assert!(
        !c.producer_done,
        "a deferred item landed on its copy alone, with no checksum"
    );
}
