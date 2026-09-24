//! Day-32 rule (WP-A, memra#536 Move 2 owed item 1, the H2D half): the typed f32 SPANS of a promote
//! batch. Additive and unversioned beside the frozen schedules, in the day-30 shape: every v1 to
//! v1.3 schedule is byte-identical and `WIRE_VERSION` stays 1. Native bindings observe real events
//! and stream waits; CPU bindings model the events as flags and stream order as a record of reads.
//! Nothing here qualifies a backend by itself.
//!
//! The span class. A promote's H2D batch carries the KV planes as `TransferOp::H2d` items into
//! registered fresh device planes. The recurrent f32 state of the same entry (the conv and ssm
//! planes, about 99 percent of a 64-token image of the hybrid 27B) is not a registered plane: the
//! resident image holds it on the heap, a host helper fills an OWNED pinned staging buffer from it
//! off the owner thread, and the promote attaches one span per plane (the staging buffer as the
//! source, a fresh device buffer as the destination) to the batch's ticket right after the
//! submission: one ticket, one landing, one settle. The engine OWNS each span's source and
//! destination from attach until the span is taken back: the staging buffer must not be rewritten
//! or freed while the copy reads it, and the destination must not be read or freed while the copy
//! writes it (the engine context runs without cudarc's event tracking, so a caller-side free would
//! not be ordered behind the copy).
//!
//! Rule, the span batch:
//!
//! 1. Refusal before enqueue is whole. A span batch refused at attach (an unknown, retired or
//!    quarantined ticket, a ticket that is not an H2D batch, one that already carries spans, a
//!    zero-byte span, a source whose length is not the destination's) returns EVERY span to the
//!    caller and leaves the ticket as it was: nothing was enqueued, nothing is quarantined.
//! 2. One landing. The ticket is `producer_done` only when every KV item's AND every span's
//!    completion event is observed complete. While a span runs, `take_spans` is refused
//!    `NotReady` and `retire` is `Busy`, even when every KV item has landed.
//! 3. Readable only behind the reader wait (rule 3 of `reader_fence`, extended over the spans). A
//!    span's destination is read on the reader's stream (the CUDA owner stream: the restore, the
//!    prime, every kernel after it). The settle's install of the reader wait covers every span's
//!    completion event, and `take_spans` is refused `NotReady` until that wait is installed, landed
//!    or not, so no caller can name a destination before a read of it is ordered behind its copy.
//!    A read the reader's stream issues before the wait is UNORDERED with respect to the copy,
//!    whatever the copy's state at issue time (this rule's red arm): the contract orders through
//!    the fence, never through observation.
//! 4. Back exactly once, before retirement. Each span taken back reports `valid_bytes` equal to
//!    the bytes it was attached with (the receipt term: the span carries no transfer-side
//!    checksum, and the host-contract gate `Completion::require` over the batch's ITEMS is
//!    unchanged by the spans). A second take is `AlreadyReleased`; `retire` stays `Busy` until the
//!    spans are taken; then the ticket retires and is acknowledged as before.
//! 5. Fail closed. An enqueue or event error on any span quarantines the whole ticket: `poll`, the
//!    reader-wait install and `take_spans` answer `Quarantined`, nothing is taken back, the caller
//!    latches and the engine keeps every span's source and destination.
//!
//! Day-33 rule 6 (the fill on the copy stream, `DAY33.md` design F), additive beside rules 1 to 5:
//!
//! 6. The fill is ordered before its copies. A FILLED span batch attaches spans whose staging sources
//!    are not yet written; the copy stream runs one fill (a host function writing every source from
//!    its resident plane) ahead of the span copies. No span copy runs before the fill: until the fill
//!    has run the batch has not landed (`take_spans` `NotReady`) whatever else completed, and every
//!    copy that runs reads a filled source. A copy issued before its fill reads unfilled bytes (this
//!    rule's red arm); the contract orders the copy behind the fill by stream order, never by timing.
use crate::contracts::*;

/// The span fixture: `submit` issues the KV batch; `attach` attaches the fixture's spans to it
/// (`Err` carries the refusal and the number of spans handed back); `attach_invalid` attaches a
/// batch that must be refused before enqueue; `items_complete` fires the KV items' events and
/// `spans_complete` the spans'; `install_reader_wait` is the settle's install of the reader
/// stream's wait on every item's and span's completion event; `take_spans` hands back the taken
/// spans' `valid_bytes`, in span order; `reader_reads` is a read of every span destination handed
/// out so far on the reader's stream; `readers_ordered` is true only when every read issued so far
/// was issued after the reader wait existed; `fail_enqueue` makes the next attach's second span
/// fail after the first was enqueued.
pub trait H2dSpanFixture {
    fn submit(&mut self) -> TransferTicket;
    fn attach(&mut self, ticket: &TransferTicket) -> std::result::Result<(), (Error, usize)>;
    fn attach_invalid(&mut self, ticket: &TransferTicket) -> (Error, usize);
    fn poll(&mut self, ticket: &TransferTicket) -> Result<Completion>;
    fn install_reader_wait(&mut self, ticket: &TransferTicket) -> Result<()>;
    fn take_spans(&mut self, ticket: &TransferTicket) -> Result<Vec<u64>>;
    fn reader_reads(&mut self);
    fn readers_ordered(&self) -> bool;
    fn retire(&mut self, ticket: &TransferTicket) -> Result<()>;
    fn acknowledge(&mut self, ticket: &TransferTicket) -> Result<()>;
    fn items_complete(&mut self);
    fn spans_complete(&mut self);
    fn fail_enqueue(&mut self);
    /// The bytes each span was attached with, in span order.
    fn span_bytes(&self) -> Vec<u64>;
}

/// The schedule. The fixture starts before `submit`.
pub fn h2d_span_batch<F: H2dSpanFixture>(f: &mut F) {
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
    // 3. Every event observed: landed, and still not readable before the reader wait.
    f.spans_complete();
    let c = f.poll(&ticket).unwrap();
    assert!(c.producer_done, "every item and span observed complete");
    assert_eq!(
        f.take_spans(&ticket),
        Err(Error::NotReady),
        "landing is not a fence: no destination before the reader wait"
    );
    assert_eq!(
        f.retire(&ticket),
        Err(Error::Busy),
        "landed spans not yet taken back keep the ticket unretired"
    );
    f.install_reader_wait(&ticket)
        .expect("the reader wait installs over a landed batch");
    // 4. Each span delivered exactly its bytes, once; a read after the wait is ordered.
    assert_eq!(f.take_spans(&ticket).unwrap(), expected);
    assert_eq!(
        f.take_spans(&ticket),
        Err(Error::AlreadyReleased),
        "spans come back exactly once"
    );
    f.reader_reads();
    assert!(
        f.readers_ordered(),
        "a read after the installed wait is ordered"
    );
    f.retire(&ticket).unwrap();
    f.acknowledge(&ticket).unwrap();
}

/// Rule 5: an enqueue failure after the first span quarantines the ticket; nothing comes back.
pub fn h2d_span_enqueue_failure_quarantines<F: H2dSpanFixture>(f: &mut F) {
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
        f.install_reader_wait(&ticket),
        Err(Error::Quarantined),
        "a quarantined ticket installs no reader wait"
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

/// The forbidden order (rule 3's red arm): the batch landed, the reader wait is not installed,
/// and the reader's stream reads a destination. A binding that hands the spans back on the
/// landing alone lets that read happen, unordered; the schedule fails. The binding must refuse the
/// take until the wait exists, so the only read issued is the one after it.
pub fn h2d_span_read_before_its_wait_is_unordered<F: H2dSpanFixture>(f: &mut F) {
    let ticket = f.submit();
    f.attach(&ticket).expect("the spans attach");
    f.items_complete();
    f.spans_complete();
    assert!(f.poll(&ticket).unwrap().producer_done);
    let early = f.take_spans(&ticket);
    if early.is_ok() {
        // The destinations were handed out before the wait: the reader reads them now.
        f.reader_reads();
    }
    assert!(
        early == Err(Error::NotReady) && f.readers_ordered(),
        "an owner read before the reader wait is unordered, landed copy or not"
    );
    f.install_reader_wait(&ticket).unwrap();
    assert_eq!(f.take_spans(&ticket).unwrap(), f.span_bytes());
    f.reader_reads();
    assert!(f.readers_ordered());
}

/// Rule 6 fixture (day 33): one H2D batch with FILLED spans. `attach_filled` attaches spans whose
/// sources the copy stream fills ahead of the copies; `fill_runs` runs the fill; `copies_run` runs
/// every span copy the binding's stream order allows now; `copies_read_filled` is true only when every
/// span copy that ran read a filled source.
pub trait H2dFillFixture {
    fn submit(&mut self) -> TransferTicket;
    fn attach_filled(&mut self, ticket: &TransferTicket)
    -> std::result::Result<(), (Error, usize)>;
    fn poll(&mut self, ticket: &TransferTicket) -> Result<Completion>;
    fn install_reader_wait(&mut self, ticket: &TransferTicket) -> Result<()>;
    fn take_spans(&mut self, ticket: &TransferTicket) -> Result<Vec<u64>>;
    fn items_complete(&mut self);
    fn fill_runs(&mut self);
    fn copies_run(&mut self);
    fn copies_read_filled(&self) -> bool;
    fn span_bytes(&self) -> Vec<u64>;
}

/// Rule 6, the schedule: the copies are offered a chance to run before the fill; none may. The batch
/// lands only after the fill and the copies, and every copy read a filled source.
pub fn h2d_span_fill_ordered_before_its_copy<F: H2dFillFixture>(f: &mut F) {
    let ticket = f.submit();
    f.attach_filled(&ticket).expect("the filled spans attach");
    f.items_complete();
    f.copies_run();
    let c = f.poll(&ticket).unwrap();
    assert!(
        !c.producer_done,
        "no span copy runs before its fill: the batch has not landed"
    );
    assert_eq!(f.take_spans(&ticket), Err(Error::NotReady));
    assert!(
        f.copies_read_filled(),
        "a copy issued before its fill reads unfilled bytes"
    );
    f.fill_runs();
    f.copies_run();
    let c = f.poll(&ticket).unwrap();
    assert!(c.producer_done, "the fill, then the copies: landed");
    f.install_reader_wait(&ticket).unwrap();
    assert_eq!(f.take_spans(&ticket).unwrap(), f.span_bytes());
    assert!(f.copies_read_filled());
}
