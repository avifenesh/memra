//! Day-38 rule (WP-A, memra#536 Move 1 owed item 2's hash 1, `DAY38.md` design G): a D2H batch whose
//! receipt is taken on the DEVICE, over each item's source, by the copy stream (the receipt program
//! `checksum`, the framed SHA-256, byte for byte), instead of by the owner thread over the landed host
//! bytes. Additive and unversioned beside the frozen schedules: every v1 to v1.3 schedule is
//! byte-identical and `WIRE_VERSION` stays 1. Nothing here qualifies a backend by itself.
//!
//! Rule, the device receipt:
//!
//! 1. One landing. An item lands only when its copy AND the batch's receipt (every accepted item's
//!    source digest, sealed by one receipt event) are observed complete; the ticket is not
//!    `producer_done` before both, whatever else completed.
//! 2. The receipt names the source. A landed item's completion checksum is its SOURCE's digest and its
//!    expectation equals it, so the D2H `require` passes on the receipt alone, as it did when the
//!    owner thread hashed the landed bytes.
//! 3. The witness is the caller's. Before anything is published, the caller re-hashes the landed bytes
//!    with the same program and requires the result equal to the receipt; a byte that changed between
//!    the source digest and the copy (or after the landing) makes the two differ, and the caller
//!    refuses the image.
//!
//! The red arm: a binding that lands an item on its copy alone, with no receipt observed.
use crate::contracts::*;

/// The fixture: `submit` issues the D2H batch (with `flip`, one byte of item 0's source changes after
/// its digest and before its copy); `copies_complete` fires every item's copy event; `receipt_complete`
/// fires the batch's receipt event; `source_digest` is the program over an item's source as the digest
/// read it; `landed_digest` is the program over the item's landed host bytes (the caller's witness).
pub trait D2hDeviceReceiptFixture {
    fn submit(&mut self, flip: bool) -> TransferTicket;
    fn copies_complete(&mut self);
    fn receipt_complete(&mut self);
    fn poll(&mut self, ticket: &TransferTicket) -> Result<Completion>;
    fn source_digest(&self, item: usize) -> Digest;
    fn landed_digest(&self, item: usize) -> Digest;
    fn items(&self) -> usize;
}

/// The schedule, a clean batch: no landing before both events; after them every item's checksum is
/// its source digest, the gate passes on it, and the witness over the landed bytes agrees.
pub fn d2h_device_receipt_lands_with_the_source_digest<F: D2hDeviceReceiptFixture>(f: &mut F) {
    let ticket = f.submit(false);
    let c = f.poll(&ticket).unwrap();
    assert!(!c.producer_done, "nothing landed at submit");
    f.copies_complete();
    let c = f.poll(&ticket).unwrap();
    assert!(
        !c.producer_done,
        "a device-receipt item does not land on its copy alone"
    );
    f.receipt_complete();
    let c = f.poll(&ticket).unwrap();
    assert!(c.producer_done, "copies and receipt observed: landed");
    for i in 0..f.items() {
        let s = &c.items[i].segments[0];
        assert!(s.producer_done);
        assert_eq!(
            s.checksum,
            Some(f.source_digest(i)),
            "item {i}'s checksum is its source's digest"
        );
        assert_eq!(
            f.landed_digest(i),
            f.source_digest(i),
            "item {i}: the witness over the landed bytes agrees with the receipt"
        );
    }
    let receipts: Vec<Vec<SegmentExpectation>> = (0..f.items())
        .map(|i| {
            vec![SegmentExpectation {
                valid_bytes: c.items[i].segments[0].valid_bytes,
                io_bytes: c.items[i].segments[0].io_bytes,
                checksum: f.source_digest(i),
            }]
        })
        .collect();
    assert!(
        c.require(&ticket, &receipts, false).is_ok(),
        "the receipt alone passes the D2H gate"
    );
}

/// Rule 3: a byte flipped between the source digest and the copy lands, and the caller's witness over
/// the landed bytes differs from the receipt, so the caller refuses before publication.
pub fn d2h_device_receipt_flip_differs_at_the_witness<F: D2hDeviceReceiptFixture>(f: &mut F) {
    let ticket = f.submit(true);
    f.copies_complete();
    f.receipt_complete();
    let c = f.poll(&ticket).unwrap();
    assert!(c.producer_done);
    assert_eq!(c.items[0].segments[0].checksum, Some(f.source_digest(0)));
    assert_ne!(
        f.landed_digest(0),
        f.source_digest(0),
        "the witness sees the changed byte: the image is refused"
    );
    for i in 1..f.items() {
        assert_eq!(f.landed_digest(i), f.source_digest(i));
    }
}

/// The red arm: the copies landed and the receipt was not observed; a binding that calls that landed
/// fails.
pub fn d2h_device_receipt_copy_alone_is_not_landed<F: D2hDeviceReceiptFixture>(f: &mut F) {
    let ticket = f.submit(false);
    f.copies_complete();
    let c = f.poll(&ticket).unwrap();
    assert!(
        !c.producer_done,
        "a device-receipt item landed on its copy alone, with no receipt"
    );
}
