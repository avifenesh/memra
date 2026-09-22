//! Day-30 rule (WP-A, memra#536 Move 2 owed item 1, the D2H half): the typed f32 SPANS of a demote
//! batch. Additive and unversioned beside the frozen schedules, in the day-20 shape: every v1 to
//! v1.3 schedule is byte-identical and `WIRE_VERSION` stays 1. Native bindings observe real events;
//! CPU bindings model the event as a flag. Nothing here qualifies a backend by itself.
//!
//! The span class. A demote's D2H batch carries the KV planes as `TransferOp::D2h` items with
//! registered device leases. The recurrent f32 state of the same entry (the conv and ssm planes,
//! about 99 percent of a 64-token image of the hybrid 27B) is not a registered plane: it is the
//! evicted entry's own device buffer, copied into a caller-supplied pinned host buffer. A span
//! attaches to the batch's ticket after the submission: one ticket, one landing, one settle. The
//! span's source and destination are OWNED by the engine from attach until the span is taken
//! back, because the device free of the source is not ordered behind a copy-stream read (the
//! engine context runs without cudarc's event tracking) and the host buffer is being written.
//!
//! Rule, the span batch:
//!
//! 1. Refusal before enqueue is whole. A span batch refused at attach (an unknown or retired
//!    ticket, a ticket that already carries spans, a zero-byte span, a destination whose length is
//!    not the source's) returns EVERY span to the caller and leaves the ticket as it was: nothing
//!    was enqueued, nothing is quarantined.
//! 2. One landing. The ticket is `producer_done` only when every KV item's AND every span's
//!    completion event is observed complete. While a span runs, `take_spans` is refused `NotReady`
//!    and `retire` is `Busy`, even when every KV item has landed: taking the spans on the KV
//!    items' landing alone is a schedule failure, never a tolerance (this rule's red arm).
//! 3. The receipt term. Each span taken back reports `valid_bytes` equal to the bytes it was
//!    attached with; its bytes become readable only by the take. The span carries no transfer-side
//!    checksum: its checksum is the host consumer's over the landed bytes (the hash helper's, which
//!    is also its bundle share), so the host-contract gate `Completion::require` over the batch's
//!    ITEMS is unchanged by the spans.
//! 4. Back exactly once, before retirement. Landed spans are taken back once (a second take is
//!    `AlreadyReleased`); `retire` stays `Busy` until they are; then the ticket retires and is
//!    acknowledged as before.
//! 5. Fail closed. An enqueue or event error on any span quarantines the whole ticket: `poll` and
//!    `take_spans` answer `Quarantined`, nothing is taken back, the caller latches and the engine
//!    keeps every span's source and destination.
use crate::contracts::*;

/// The span fixture: `submit` issues the KV batch; `attach` attaches the fixture's spans to it
/// (`Err` carries the refusal and the number of spans handed back); `attach_invalid` attaches a
/// batch that must be refused before enqueue; `items_complete` fires the KV items' events and
/// `spans_complete` the spans'; `fail_enqueue` makes the next attach's second span fail after the
/// first was enqueued; `take_spans` hands back the landed spans' `valid_bytes`, in span order.
pub trait D2hSpanFixture {
    fn submit(&mut self) -> TransferTicket;
    fn attach(&mut self, ticket: &TransferTicket) -> std::result::Result<(), (Error, usize)>;
    fn attach_invalid(&mut self, ticket: &TransferTicket) -> (Error, usize);
    fn poll(&mut self, ticket: &TransferTicket) -> Result<Completion>;
    fn take_spans(&mut self, ticket: &TransferTicket) -> Result<Vec<u64>>;
    fn retire(&mut self, ticket: &TransferTicket) -> Result<()>;
    fn acknowledge(&mut self, ticket: &TransferTicket) -> Result<()>;
    fn items_complete(&mut self);
    fn spans_complete(&mut self);
    fn fail_enqueue(&mut self);
    /// The bytes each span was attached with, in span order.
    fn span_bytes(&self) -> Vec<u64>;
}

/// The schedule. The fixture starts before `submit`.
pub fn d2h_span_batch<F: D2hSpanFixture>(f: &mut F) {
    let ticket = f.submit();
    let expected = f.span_bytes();
    assert!(!expected.is_empty(), "a span batch has at least one span");
    // 1. A refused attach hands every span back and leaves the ticket untouched.
    let (error, back) = f.attach_invalid(&ticket);
    assert_eq!(
        error,
        Error::InvalidLayout,
        "an invalid span is refused by name"
    );
    assert!(back > 0, "a refused attach hands every span back");
    assert!(
        f.poll(&ticket).is_ok(),
        "a refusal before enqueue quarantines nothing"
    );
    f.attach(&ticket).expect("the valid spans attach");
    let again = f.attach(&ticket);
    assert!(
        matches!(again, Err((Error::Busy, n)) if n == expected.len()),
        "a ticket carries one span batch; a second is refused whole"
    );
    // 2. Running: nothing landed.
    let c = f.poll(&ticket).unwrap();
    assert!(!c.producer_done, "the copies have not landed at attach");
    assert_eq!(f.take_spans(&ticket), Err(Error::NotReady));
    assert_eq!(f.retire(&ticket), Err(Error::Busy));
    // 2. The KV items landed, the spans still run: not landed, no take, no retire.
    f.items_complete();
    let c = f.poll(&ticket).unwrap();
    assert!(
        !c.producer_done,
        "a batch with a running span has not landed"
    );
    assert_eq!(
        f.take_spans(&ticket),
        Err(Error::NotReady),
        "the KV items' landing alone must not hand the spans back"
    );
    assert_eq!(f.retire(&ticket), Err(Error::Busy));
    // 2 and 4. Every event observed: landed; retire waits for the take.
    f.spans_complete();
    let c = f.poll(&ticket).unwrap();
    assert!(c.producer_done, "every item and span observed complete");
    assert_eq!(
        f.retire(&ticket),
        Err(Error::Busy),
        "landed spans not yet taken back keep the ticket unretired"
    );
    // 3. Each span delivered exactly its bytes.
    assert_eq!(f.take_spans(&ticket).unwrap(), expected);
    assert_eq!(
        f.take_spans(&ticket),
        Err(Error::AlreadyReleased),
        "spans come back exactly once"
    );
    f.retire(&ticket).unwrap();
    f.acknowledge(&ticket).unwrap();
}

/// Rule 5: an enqueue failure after the first span quarantines the ticket; nothing comes back.
pub fn d2h_span_enqueue_failure_quarantines<F: D2hSpanFixture>(f: &mut F) {
    let ticket = f.submit();
    f.fail_enqueue();
    f.attach(&ticket)
        .expect("a span batch that reached the stream is accepted, then quarantined");
    f.items_complete();
    f.spans_complete();
    assert_eq!(
        f.poll(&ticket).map(|c| c.producer_done),
        Err(Error::Quarantined)
    );
    assert_eq!(
        f.take_spans(&ticket),
        Err(Error::Quarantined),
        "a quarantined ticket hands no span back"
    );
    assert!(
        f.retire(&ticket).is_err(),
        "a quarantined ticket does not retire"
    );
}

/// The forbidden order: a caller that takes the spans once the KV items landed, with a span still
/// running. The schedule fails; the binding must report the take refused.
pub fn d2h_span_taken_on_the_items_landing_fails<F: D2hSpanFixture>(f: &mut F) {
    let ticket = f.submit();
    f.attach(&ticket).expect("the spans attach");
    f.items_complete();
    assert_eq!(
        f.take_spans(&ticket),
        Err(Error::NotReady),
        "the KV items' landing alone must not hand the spans back"
    );
}
