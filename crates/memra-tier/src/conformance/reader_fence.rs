//! Day-19 rule (WP-A, memra#536 Move 1, the first owed item): the reader fence of an H2D issued
//! off the owner stream. Additive and unversioned beside the frozen schedules, in the day-11
//! shape: every v1 to v1.3 schedule is byte-identical and `WIRE_VERSION` stays 1. Native bindings
//! observe real events and stream waits; CPU bindings model stream order only. Nothing here
//! qualifies a backend by itself.
//!
//! Rule 3, the reader fence. An H2D whose copy is issued on a stream other than the destination's
//! reader (the CUDA owner stream: the D2D restore and every kernel after it) is published only
//! behind a WAIT on the copy's completion event installed on the reader's stream. The contract's
//! `consumer_fenced` means exactly that installed wait, never a flag and never the copy's landing:
//!
//! 1. Readable. `ready_view` (and so `take_destination`) succeeds only when the item's producer is
//!    observed complete AND its reader wait is installed (`consumer_fenced` with a `consumer_fence`
//!    present). A completed copy with no wait installed is `NotReady`; an installed wait over a
//!    running copy is `NotReady`. Publication never runs ahead of either.
//! 2. Retire. The consumer fence is recorded on the reader's stream after the wait, so it orders
//!    behind the copy; `record_consumer` before publication is `NotReady`; `retire(ticket,
//!    Some(fence))` is `Busy` until that fence's event has completed, then succeeds once.
//! 3. Off-owner reads. A read the reader's stream issues before its wait is installed is
//!    UNORDERED with respect to the copy, whatever the copy's state at issue time: the contract
//!    orders through the fence, not through observation. The engine keeps the destination bound
//!    and unpublished until the wait exists, so no caller can name it before then; the install
//!    may sit at submit (the engine's day-18 program: the owner stream waits at submit, and the
//!    tenant's later kernels queue behind the copy's landing) or at settle (the owed program: the
//!    wait installed after the completion is observed, before `ready_view`), and the schedule is
//!    the same in both; only WHEN `consumer_fenced` turns true differs.
//!
//! Finding (lane A, day 18, verbatim): "The engine keeps the submit-time `owner.wait(item event)`
//! for an H2D, so kernels the tenant submits after the promote's submit queue behind the copy's
//! landing (bounded by the copy time, about 6 ms per 160 MB on the target card). An engine method
//! that installs the wait after `event_done` at the settle (the consumer fence recorded after it)
//! would remove that bound; it changes the engine's `consumer_fenced` semantics for a copy-stream
//! H2D and needs the tier crate's conformance to speak first."
use crate::contracts::*;

/// When the reader stream's wait on the item's completion event is installed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReaderWaitInstall {
    /// Inside `submit`: the engine's day-18 program for an H2D on the copy stream.
    AtSubmit,
    /// At the settle, after the completion is observed and before publication: the owed program.
    AtSettle,
}

/// Rule 3 fixture: one H2D issued off the reader's stream, its destination bound to the ticket.
/// `poll`, `ready_view`, `take_destination`, `record_consumer`, `retire`, `retired` and
/// `acknowledge` are the engine's own answers (a native binding forwards to its
/// `TransferEngine`; the CPU binding models them). The stream hooks drive and observe order:
/// `copy_completes` fires the item's completion event; `install_reader_wait` is the settle-time
/// install (an at-submit binding already installed it inside `submit` and treats this as a
/// no-op); `reader_issues` is a read of the destination on the reader's stream (the restore, a
/// kernel); `reader_completes` fires the consumer fence's event; `readers_ordered` is true only
/// when every read issued so far was issued after a wait on the item's completion event existed
/// on the reader's stream.
pub trait ReaderFenceFixture {
    fn submit(&mut self) -> TransferTicket;
    fn poll(&mut self, ticket: &TransferTicket) -> Result<Completion>;
    fn ready_view(&mut self, ticket: &TransferTicket) -> Result<()>;
    fn take_destination(&mut self, ticket: &TransferTicket) -> Result<()>;
    fn record_consumer(&mut self, ticket: &TransferTicket) -> Result<FenceId>;
    fn retire(&mut self, ticket: &TransferTicket, consumer_done: Option<FenceId>) -> Result<()>;
    fn retired(&mut self, ticket: &TransferTicket) -> Result<bool>;
    fn acknowledge(&mut self, ticket: &TransferTicket) -> Result<()>;
    fn copy_completes(&mut self);
    fn install_reader_wait(&mut self, ticket: &TransferTicket);
    fn reader_issues(&mut self);
    fn reader_completes(&mut self);
    fn readers_ordered(&self) -> bool;
}

fn fenced(c: &Completion) -> bool {
    c.consumer_fenced
        && c.items.iter().all(|i| {
            i.segments
                .iter()
                .all(|s| s.consumer_fenced && s.consumer_fence.is_some())
        })
}

/// Rule 3, the schedule, the same under both installs. The fixture starts before `submit`.
pub fn h2d_reader_fence<F: ReaderFenceFixture>(f: &mut F, install: ReaderWaitInstall) {
    let ticket = f.submit();
    // The copy is running: nothing is readable, whatever the fence says.
    let c = f.poll(&ticket).unwrap();
    assert!(!c.producer_done, "the copy has not landed at submit");
    assert_eq!(
        fenced(&c),
        install == ReaderWaitInstall::AtSubmit,
        "consumer_fenced is the installed wait: true at submit only under the at-submit install"
    );
    assert!(matches!(f.ready_view(&ticket), Err(Error::NotReady)));
    assert!(matches!(f.take_destination(&ticket), Err(Error::NotReady)));
    assert!(matches!(f.record_consumer(&ticket), Err(Error::NotReady)));
    assert_eq!(f.retire(&ticket, None), Err(Error::Busy));
    assert!(!f.retired(&ticket).unwrap());
    // The copy lands. Under the at-settle install the destination is STILL not readable: a
    // completed copy with no reader wait is not a fence.
    f.copy_completes();
    let c = f.poll(&ticket).unwrap();
    assert!(c.producer_done);
    if install == ReaderWaitInstall::AtSettle {
        assert!(!fenced(&c), "landing is not a fence");
        assert!(matches!(f.ready_view(&ticket), Err(Error::NotReady)));
        assert!(matches!(f.take_destination(&ticket), Err(Error::NotReady)));
        assert!(matches!(f.record_consumer(&ticket), Err(Error::NotReady)));
        // (`retire(ticket, None)` IS allowed here: an unpublished, landed ticket retires without
        // a consumer fence; that is the abort path, not a publication.)
        f.install_reader_wait(&ticket);
        let c = f.poll(&ticket).unwrap();
        assert!(c.producer_done);
    } else {
        f.install_reader_wait(&ticket); // a no-op for an at-submit binding, by its own statement
    }
    let c = f.poll(&ticket).unwrap();
    assert!(fenced(&c), "the wait is installed: the item is fenced");
    // Readable now, and the reader's work after this point is ordered behind the copy.
    f.ready_view(&ticket).unwrap();
    f.reader_issues();
    assert!(
        f.readers_ordered(),
        "a read after the installed wait is ordered"
    );
    // The consumer fence is recorded after the reader's work; retirement waits on its event.
    let consumer = f.record_consumer(&ticket).unwrap();
    assert_eq!(f.retire(&ticket, Some(consumer)), Err(Error::Busy));
    assert!(!f.retired(&ticket).unwrap());
    f.reader_completes();
    f.retire(&ticket, Some(consumer)).unwrap();
    assert!(f.retired(&ticket).unwrap());
    f.acknowledge(&ticket).unwrap();
    assert!(f.readers_ordered());
}

/// Rule 3, the forbidden order, under the at-settle install: a read issued on the reader's
/// stream between `submit` and the settle's `install_reader_wait` is unordered, and the engine
/// must not have let it read a published destination: `ready_view` is `NotReady` throughout,
/// before and after the copy lands, until the wait is installed. The binding reports the read
/// unordered even though the copy may have landed before it was issued: the contract orders
/// through the fence, not through observation.
pub fn h2d_reader_issued_before_its_wait_is_unordered<F: ReaderFenceFixture>(f: &mut F) {
    let ticket = f.submit();
    assert!(matches!(f.ready_view(&ticket), Err(Error::NotReady)));
    f.copy_completes();
    assert!(matches!(f.ready_view(&ticket), Err(Error::NotReady)));
    assert!(f.readers_ordered(), "no read has been issued");
    f.reader_issues();
    assert!(
        !f.readers_ordered(),
        "a read before the installed wait is unordered, landed copy or not"
    );
    assert!(matches!(f.ready_view(&ticket), Err(Error::NotReady)));
    f.install_reader_wait(&ticket);
    f.ready_view(&ticket).unwrap();
    assert!(
        !f.readers_ordered(),
        "the early read stays unordered; a later wait does not order it"
    );
}
