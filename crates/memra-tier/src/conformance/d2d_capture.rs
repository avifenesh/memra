//! Day-20 rule (WP-A, memra#536 Move 2, slice 1): the event-ordered publication of a D2D
//! capture issued off the owner stream. Additive and unversioned beside the frozen schedules, in
//! the day-11 and day-19 shape: every v1 to v1.3 schedule is byte-identical and `WIRE_VERSION`
//! stays 1. Native bindings observe real events; CPU bindings model the event as a flag. Nothing
//! here qualifies a backend by itself.
//!
//! The capture class. A prefix snapshot copies the KV rows `0..boundary` of a LIVE session cache
//! into fresh device planes; the session keeps decoding and appends rows past the boundary, so the
//! source rows are stable (append-only per position) and the recurrent state, which the next step
//! overwrites, is NOT in this class: it stays on the owner stream at the boundary. The copies ride
//! a side stream (the engine's receipt stream since WP-A day 38's design G''', its copy stream
//! before) behind a producer event recorded on the owner stream after the boundary chunk,
//! and the entry enters the device prefix index only after every copy's completion event has been
//! observed complete. Publication is the caller's (the LRU insert), not the engine's `ready_view`.
//!
//! Rule, the capture publish:
//!
//! 1. Not ready. Until every item's completion event is observed complete, the ticket is not
//!    `landed`, a publication into the index is refused `NotReady`, and `retire(ticket, None)`
//!    is `Busy`. A publish before the event is a schedule failure, never a tolerance.
//! 2. Landed. Once every item's event is complete the ticket is `landed`, each item's
//!    `valid_bytes` equals the bytes it was submitted with, and the publication succeeds exactly
//!    once. The destination's consumer is the index publication, host-ordered after the event
//!    like a D2H's host consumer; no reader wait on the owner stream is installed at publish and
//!    the ticket retires with `retire(ticket, None)` (the unpublished-in-the-engine's-sense,
//!    landed ticket), then acknowledges, then every destination comes back to its caller.
//! 3. The receipt term. Slice 1 carries the byte count, the epochs and the completion event and
//!    NO witnessed checksum, so `Completion::require` (the H2D and D2H publication gate) refuses
//!    such an item `Corrupt`: no path can publish a capture through the host-contract gate ahead of
//!    its receipt. Slice 3 (day 22, `d2d_receipt_witnessed`) added the witness: a destination
//!    digest taken on the copy's stream after the copy against a source digest taken behind the
//!    producer fence. This schedule's fixture stays receipt-less, so the clause here remains the
//!    receipt-less refusal by name; the witnessed arm is `d2d_receipt.rs`.
//!
//! Finding (lane A, day 20, recorded before any code): the pre-registered op shape
//! `TransferOp::D2d(ContiguousCopy)` takes owned `DeviceLease`s on BOTH sides and the engine's
//! registry admits only moved buffers, never a borrowed pointer; a capture's source is the live
//! session cache's plane, which the decoding session keeps. So the capture class is a same-device
//! copy from a BORROWED source span into an OWNED, registered destination lease, and its typed op
//! lives in the engine (`CudaTransfers::submit_d2d_capture`), not as a `TransferOp` variant; the
//! schedule here is over that class. `ContiguousCopy` stays the owned-to-owned (peer) shape.
use crate::contracts::*;

/// The capture fixture: one batch of same-device copies issued off the reader's stream, each
/// into a registered destination. `poll`, `retire`, `retired` and `acknowledge` are the engine's
/// own answers; `landed` is the engine's publication predicate (every item's completion event
/// observed complete); `publish` is the CALLER's publication into the device prefix index, which
/// must ask `landed` first and refuse `NotReady` otherwise; `require_receipt` runs
/// `Completion::require` as the host-contract gate would (`device = true`); `copy_completes`
/// fires every item's completion event; `destination_back` takes every destination out of the
/// engine after acknowledgement.
pub trait D2dCaptureFixture {
    fn submit(&mut self) -> TransferTicket;
    fn poll(&mut self, ticket: &TransferTicket) -> Result<Completion>;
    fn landed(&mut self, ticket: &TransferTicket) -> Result<bool>;
    fn publish(&mut self, ticket: &TransferTicket) -> Result<()>;
    fn require_receipt(&mut self, ticket: &TransferTicket) -> Result<()>;
    fn retire(&mut self, ticket: &TransferTicket, consumer_done: Option<FenceId>) -> Result<()>;
    fn retired(&mut self, ticket: &TransferTicket) -> Result<bool>;
    fn acknowledge(&mut self, ticket: &TransferTicket) -> Result<()>;
    fn copy_completes(&mut self);
    fn destination_back(&mut self, ticket: &TransferTicket) -> Result<usize>;
    /// The bytes each item was submitted with, in item order.
    fn submitted_bytes(&self) -> Vec<u64>;
}

/// The schedule. The fixture starts before `submit`.
pub fn d2d_capture_publish<F: D2dCaptureFixture>(f: &mut F) {
    let ticket = f.submit();
    let expected = f.submitted_bytes();
    assert!(
        !expected.is_empty(),
        "a capture batch has at least one item"
    );
    // 1. The copies are running: not landed, not publishable, not retirable.
    let c = f.poll(&ticket).unwrap();
    assert!(!c.producer_done, "the copies have not landed at submit");
    assert!(!f.landed(&ticket).unwrap(), "a running copy is not landed");
    assert!(
        matches!(f.publish(&ticket), Err(Error::NotReady)),
        "a publish before the event must be refused NotReady"
    );
    assert_eq!(f.retire(&ticket, None), Err(Error::Busy));
    assert!(!f.retired(&ticket).unwrap());
    // 3. The receipt term is not witnessed in this fixture: the host-contract gate refuses the item.
    assert!(
        matches!(
            f.require_receipt(&ticket),
            Err(Error::Corrupt | Error::NotReady)
        ),
        "a D2D item without a witnessed checksum is refused by require (day 22: d2d_receipt_witnessed)"
    );
    // 2. Every event completes: landed, bytes exact, published once, retired, acknowledged.
    f.copy_completes();
    let c = f.poll(&ticket).unwrap();
    assert!(c.producer_done);
    assert!(f.landed(&ticket).unwrap(), "every event observed complete");
    let delivered: Vec<u64> = c
        .items
        .iter()
        .filter(|i| i.accepted)
        .flat_map(|i| i.segments.iter().map(|s| s.valid_bytes))
        .collect();
    assert_eq!(delivered, expected, "each item delivered exactly its bytes");
    assert!(
        matches!(f.require_receipt(&ticket), Err(Error::Corrupt)),
        "landed or not, the unwitnessed checksum keeps the host-contract gate closed"
    );
    f.publish(&ticket).unwrap();
    assert!(
        matches!(f.publish(&ticket), Err(Error::AlreadyReleased)),
        "publication happens exactly once"
    );
    f.retire(&ticket, None).unwrap();
    assert!(f.retired(&ticket).unwrap());
    f.acknowledge(&ticket).unwrap();
    assert_eq!(
        f.destination_back(&ticket).unwrap(),
        expected.len(),
        "every destination comes back to its caller"
    );
}

// ---------------------------------------------------------------------------
// Day-24 rule (WP-A, memra#536 Move 2, owed item 2, the capture half): the DRAFT-BEARING capture.
// A spec session's boundary publish carries, beside the trunk KV rows `[0..pos)` of its live cache,
// the MTP draft plane: rows `[0..pos)` of its live draft scratch (`Role::Draft`, one K and one V
// span), append-only below the boundary for the session's lifetime as the trunk rows are. Both
// classes are borrowed live sources copied into fresh registered destinations behind ONE producer
// event recorded on the owner stream after the boundary's last trunk and draft-head write; the
// draft plane is two more items of the SAME batch: one ticket, one fence, one receipt, one
// publication. The pending entry owns the fresh planes of both classes and publishes both or
// neither. Nothing in the rule above changes for the trunk items.
//
// Rule, the draft-bearing capture publish:
//
// 1. One batch, both classes. The batch carries at least one `Role::Draft` item and at least one
//    trunk item (`Role::Key` or `Role::Value`); every item has a class.
// 2. Both or neither. The batch is `landed` only when EVERY item's completion event is observed
//    complete: a batch whose trunk items landed while a draft item is still running is not landed,
//    a publish then is refused `NotReady`, and `retire(ticket, None)` is `Busy`. A publish on the
//    trunk's landing alone is a schedule failure, never a tolerance (this schedule's red arm).
// 3. The receipt term covers both classes: after the landing every item, draft and trunk, carries
//    a witnessed checksum (the destination digest equal to the source digest), and the host-contract
//    gate admits the batch; an item of either class without its witness keeps the gate closed.
// 4. Published once, then retire with no consumer fence, acknowledge, and every destination of
//    both classes comes back to the caller (rule 2 above).

/// The draft-bearing fixture: the capture fixture plus the class of every item, a per-class landing
/// (`trunk_completes` fires the trunk items' events and leaves the draft items running) and the
/// witnessed receipt term per class (`witnessed_classes`: the classes of the items whose landed
/// completion carries a checksum equal to its expectation, in item order; empty before the landing).
pub trait D2dDraftCaptureFixture: D2dCaptureFixture {
    fn item_classes(&self) -> Vec<Role>;
    fn trunk_completes(&mut self);
    fn witnessed_classes(&mut self, ticket: &TransferTicket) -> Result<Vec<Role>>;
}

/// The schedule. The fixture starts before `submit`.
pub fn d2d_capture_draft_publish<F: D2dDraftCaptureFixture>(f: &mut F) {
    let ticket = f.submit();
    let classes = f.item_classes();
    let expected = f.submitted_bytes();
    assert_eq!(classes.len(), expected.len(), "every item has a class");
    assert!(
        classes.contains(&Role::Draft),
        "the batch carries the draft plane"
    );
    assert!(
        classes.iter().any(|r| matches!(r, Role::Key | Role::Value)),
        "and the trunk rows: one batch, one ticket, one receipt"
    );
    // 1. Running: not landed, not publishable, not retirable; nothing witnessed.
    let c = f.poll(&ticket).unwrap();
    assert!(!c.producer_done, "the copies have not landed at submit");
    assert!(!f.landed(&ticket).unwrap());
    assert!(matches!(f.publish(&ticket), Err(Error::NotReady)));
    assert_eq!(f.retire(&ticket, None), Err(Error::Busy));
    assert!(
        f.witnessed_classes(&ticket).unwrap().is_empty(),
        "no item is witnessed before its landing"
    );
    assert!(matches!(
        f.require_receipt(&ticket),
        Err(Error::NotReady | Error::Corrupt)
    ));
    // 2. The trunk landed, the draft still running: NOT landed, publish refused, retire Busy; the
    //    witnessed classes are the trunk's alone.
    f.trunk_completes();
    let c = f.poll(&ticket).unwrap();
    assert!(
        !c.producer_done,
        "a batch with a running draft item has not landed"
    );
    assert!(
        !f.landed(&ticket).unwrap(),
        "landing is every item's event: the trunk's alone is not it"
    );
    assert!(
        matches!(f.publish(&ticket), Err(Error::NotReady)),
        "a publish with the draft unlanded is refused: both or neither"
    );
    assert_eq!(f.retire(&ticket, None), Err(Error::Busy));
    let witnessed = f.witnessed_classes(&ticket).unwrap();
    assert!(
        !witnessed.is_empty() && !witnessed.contains(&Role::Draft),
        "the trunk items are witnessed, the draft items are not yet"
    );
    assert!(
        matches!(
            f.require_receipt(&ticket),
            Err(Error::NotReady | Error::Corrupt)
        ),
        "an unwitnessed draft item keeps the host-contract gate closed"
    );
    // 3. Every event completes: landed, bytes exact, both classes witnessed, the gate admits.
    f.copy_completes();
    let c = f.poll(&ticket).unwrap();
    assert!(c.producer_done);
    assert!(f.landed(&ticket).unwrap(), "every event observed complete");
    let delivered: Vec<u64> = c
        .items
        .iter()
        .filter(|i| i.accepted)
        .flat_map(|i| i.segments.iter().map(|s| s.valid_bytes))
        .collect();
    assert_eq!(delivered, expected, "each item delivered exactly its bytes");
    assert_eq!(
        f.witnessed_classes(&ticket).unwrap(),
        classes,
        "the receipt term is witnessed for every item of both classes"
    );
    assert_eq!(
        f.require_receipt(&ticket),
        Ok(()),
        "witnessed for both classes: the host-contract gate admits the batch"
    );
    // 4. Published once; retire with no consumer fence; acknowledge; both classes come back.
    f.publish(&ticket).unwrap();
    assert!(
        matches!(f.publish(&ticket), Err(Error::AlreadyReleased)),
        "publication happens exactly once"
    );
    f.retire(&ticket, None).unwrap();
    assert!(f.retired(&ticket).unwrap());
    f.acknowledge(&ticket).unwrap();
    assert_eq!(
        f.destination_back(&ticket).unwrap(),
        expected.len(),
        "every destination of both classes comes back to its caller"
    );
}

/// The forbidden order: a caller that publishes once the TRUNK items landed, with a draft item
/// still running. The schedule fails; the binding must report the batch not landed and the
/// publication already taken (the index names an entry whose draft plane is still being written).
pub fn d2d_capture_draft_published_with_the_draft_unlanded_fails<F: D2dDraftCaptureFixture>(
    f: &mut F,
) {
    let ticket = f.submit();
    assert!(f.item_classes().contains(&Role::Draft));
    f.trunk_completes();
    assert!(!f.landed(&ticket).unwrap());
    assert!(
        matches!(f.publish(&ticket), Err(Error::NotReady)),
        "the trunk's landing alone must not publish the entry"
    );
}
