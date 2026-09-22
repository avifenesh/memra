//! Day-21 rule (WP-A, memra#536 Move 2, slice 2): the readiness of a D2D restore issued off the
//! owner stream into a borrowed destination. Additive and unversioned beside the frozen schedules,
//! in the day-11, day-19 and day-20 shape: every v1 to v1.3 schedule is byte-identical and
//! `WIRE_VERSION` stays 1. Native bindings observe real events and stream waits; CPU bindings
//! model the event as a flag and the two streams as logs. Nothing here qualifies a backend by
//! itself.
//!
//! The restore class. A whole-entry prefix hit copies a PUBLISHED device entry's KV rows into the
//! admitted request's FRESH session cache. Neither side is the engine's: the destination is the
//! session's for its whole life and the source stays in the device LRU, servable to other hits
//! while the copy reads it. The copies ride the copy stream behind a producer event recorded on
//! the owner stream; the source is held by the device LRU's PIN from submit through acknowledge
//! (a pinned entry is out of the eviction index by construction), which is the contract's
//! producer-side guarantee for a borrowed source; the destination's consumer is the owner stream
//! (the request's first prime chunk and every kernel after it), so rule 3 applies: the items are
//! NOT fenced at submit, and the owner stream's wait on each completion event is installed at the
//! settle, before the request that owns the cache is re-admitted.
//!
//! Rule, the restore readiness:
//!
//! 1. Not landed. Until every item's completion event is observed complete the ticket is not
//!    `landed`, not `ready`, and `retire(ticket, None)` is `Busy`. The source is pinned.
//! 2. Landed, unfenced. Once every event is complete the ticket is `landed` and each item's
//!    `valid_bytes` equals the bytes it was submitted with, but the destination is STILL not
//!    `ready`: a landed copy without the installed reader wait is not a fence (rule 3), and a prime
//!    issued on the reader's stream now is UNORDERED whatever the copy's state. That prime is the
//!    schedule's red arm, never a tolerance.
//! 3. Ready. `install_reader_wait` installs the owner stream's wait on every item's event and
//!    fences the items; the ticket is `ready` exactly then; a prime issued after it is ordered.
//!    `retire(ticket, None)` succeeds (the destination leaves through its owner, not through the
//!    engine's publication), then `acknowledge`; the source pin is released only after that, and
//!    the caller's pin becomes the serving session's pin.
//! 4. The receipt term. As for the capture (`d2d_capture_publish` rule 3), a restore item has no
//!    witnessed checksum until slice 3: `Completion::require` refuses it, so no path publishes a
//!    restore through the host-contract gate.
use crate::contracts::*;

/// The restore fixture: one batch of same-device copies issued off the reader's stream into a
/// borrowed destination from a pinned borrowed source. `poll`, `landed`, `retire`, `retired` and
/// `acknowledge` are the engine's own answers; `ready` is the CALLER's predicate (landed AND every
/// item fenced by an installed reader wait); `require_receipt` runs `Completion::require` as the
/// host-contract gate would; `copy_completes` fires every item's completion event;
/// `install_reader_wait` is the settle-time install on the reader's stream; `prime_issues` is the
/// request's first prime chunk on the reader's stream; `primes_ordered` is true only when every
/// prime issued so far was issued after a wait on every item's event existed on the reader's
/// stream; `source_pinned` reads the device LRU's pin on the source entry; `release_source` is the
/// caller's hand-over of the pin after acknowledgement.
pub trait D2dRestoreFixture {
    fn submit(&mut self) -> TransferTicket;
    fn poll(&mut self, ticket: &TransferTicket) -> Result<Completion>;
    fn landed(&mut self, ticket: &TransferTicket) -> Result<bool>;
    fn ready(&mut self, ticket: &TransferTicket) -> Result<bool>;
    fn require_receipt(&mut self, ticket: &TransferTicket) -> Result<()>;
    fn retire(&mut self, ticket: &TransferTicket, consumer_done: Option<FenceId>) -> Result<()>;
    fn retired(&mut self, ticket: &TransferTicket) -> Result<bool>;
    fn acknowledge(&mut self, ticket: &TransferTicket) -> Result<()>;
    fn copy_completes(&mut self);
    fn install_reader_wait(&mut self, ticket: &TransferTicket);
    fn prime_issues(&mut self);
    fn primes_ordered(&self) -> bool;
    fn source_pinned(&self) -> bool;
    fn release_source(&mut self);
    /// The bytes each item was submitted with, in item order.
    fn submitted_bytes(&self) -> Vec<u64>;
}

fn fenced(c: &Completion) -> bool {
    c.consumer_fenced
        && c.items.iter().filter(|i| i.accepted).all(|i| {
            i.segments
                .iter()
                .all(|s| s.consumer_fenced && s.consumer_fence.is_some())
        })
}

/// The schedule. The fixture starts before `submit`; the source is pinned by the caller at submit.
pub fn d2d_restore_ready<F: D2dRestoreFixture>(f: &mut F) {
    let ticket = f.submit();
    let expected = f.submitted_bytes();
    assert!(
        !expected.is_empty(),
        "a restore batch has at least one item"
    );
    assert!(f.source_pinned(), "the source is pinned from submit");
    // 1. The copies are running: not landed, not ready, not fenced, not retirable.
    let c = f.poll(&ticket).unwrap();
    assert!(!c.producer_done, "the copies have not landed at submit");
    assert!(!fenced(&c), "a restore is not fenced at submit (rule 3)");
    assert!(!f.landed(&ticket).unwrap());
    assert!(!f.ready(&ticket).unwrap(), "a running copy is not ready");
    assert_eq!(f.retire(&ticket, None), Err(Error::Busy));
    assert!(!f.retired(&ticket).unwrap());
    // 4. No witnessed checksum term: the host-contract gate refuses the item.
    assert!(
        matches!(
            f.require_receipt(&ticket),
            Err(Error::Corrupt | Error::NotReady)
        ),
        "a D2D item has no witnessed checksum before slice 3; require must refuse"
    );
    // 2. Every event completes: landed, bytes exact, and STILL not ready without the wait.
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
    assert!(!fenced(&c), "landing is not a fence");
    assert!(
        !f.ready(&ticket).unwrap(),
        "a landed copy without the installed reader wait is not ready"
    );
    assert!(
        matches!(f.require_receipt(&ticket), Err(Error::Corrupt)),
        "landed or not, the unwitnessed checksum keeps the host-contract gate closed"
    );
    assert!(f.source_pinned(), "the source stays pinned while unfenced");
    // 3. The wait is installed: fenced, ready, the prime after it is ordered.
    f.install_reader_wait(&ticket);
    let c = f.poll(&ticket).unwrap();
    assert!(fenced(&c), "the wait is installed: every item is fenced");
    assert!(f.ready(&ticket).unwrap(), "landed and fenced: ready");
    f.prime_issues();
    assert!(
        f.primes_ordered(),
        "a prime issued after the installed wait is ordered behind the copy"
    );
    f.retire(&ticket, None).unwrap();
    assert!(f.retired(&ticket).unwrap());
    assert!(
        f.source_pinned(),
        "the pin outlives the ticket's retirement"
    );
    f.acknowledge(&ticket).unwrap();
    assert!(f.source_pinned(), "and its acknowledgement");
    f.release_source();
    assert!(
        !f.source_pinned(),
        "the caller hands the pin over only after acknowledge"
    );
    assert!(f.primes_ordered());
}

/// The forbidden order: a prime issued on the reader's stream after the copy landed but before
/// the reader wait is installed is UNORDERED, landed copy or not; a later wait does not order it.
/// The binding must report the early prime unordered and the destination not ready throughout.
pub fn d2d_restore_primed_before_its_wait_is_unordered<F: D2dRestoreFixture>(f: &mut F) {
    let ticket = f.submit();
    assert!(!f.ready(&ticket).unwrap());
    f.copy_completes();
    assert!(f.landed(&ticket).unwrap());
    assert!(!f.ready(&ticket).unwrap());
    assert!(f.primes_ordered(), "no prime has been issued");
    f.prime_issues();
    assert!(
        !f.primes_ordered(),
        "a prime before the installed wait is unordered, landed copy or not"
    );
    assert!(!f.ready(&ticket).unwrap());
    f.install_reader_wait(&ticket);
    assert!(f.ready(&ticket).unwrap());
    assert!(
        !f.primes_ordered(),
        "the early prime stays unordered; a later wait does not order it"
    );
}
