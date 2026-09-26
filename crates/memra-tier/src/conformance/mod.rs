//! Reusable conformance schedules. The CPU `contracts` test target and the
//! native qualification gates (`tier-transfer-gate`) run these exact functions against their own
//! backend fixtures. Do not duplicate expected outcomes in each lane. Nothing here qualifies a
//! production backend by itself.
use crate::contracts::*;

pub fn object_cancel<S: ObjectStore>(store: &mut S, key: ObjectKey) {
    let mut txn = store.begin(key.clone(), 1, Durability::Ephemeral).unwrap();
    store.put(&mut txn, &[3]).unwrap();
    assert_eq!(
        store.cancel(&mut txn).unwrap(),
        CancelState::PublicationRevoked
    );
    assert!(store.commit(&mut txn).is_err());
    assert!(store.lookup(&key).unwrap().is_none());
}
pub fn transfer_cancel<T: TransferEngine>(
    engine: &mut T,
    ticket: TransferTicket,
    complete: impl FnOnce(&mut T, &TransferTicket),
) {
    assert_eq!(
        engine.cancel(&ticket).unwrap(),
        CancelState::PublicationRevoked
    );
    assert!(!engine.retired(&ticket).unwrap());
    assert!(engine.acknowledge(&ticket).is_err());
    complete(engine, &ticket);
    assert!(engine.retired(&ticket).unwrap());
    assert!(matches!(
        engine.ready_view(&ticket, 0, ticket.epochs),
        Err(Error::Cancelled)
    ));
    engine.acknowledge(&ticket).unwrap();
    assert!(engine.retired(&ticket).is_err());
}
pub fn tier_cancel<T: TierStore>(
    store: &mut T,
    reservation: &TierReservation,
    program: &ProgramIdentity,
    epochs: Epochs,
) {
    store.prefetch(reservation).unwrap();
    assert_eq!(
        store.advance(reservation, epochs).unwrap(),
        Phase::HostReady
    );
    assert!(store.ready(reservation, program, epochs).is_err());
    store.load(reservation).unwrap();
    assert_eq!(store.advance(reservation, epochs).unwrap(), Phase::Ready);
    assert_eq!(
        store.cancel(reservation).unwrap(),
        CancelState::PublicationRevoked
    );
    assert!(store.ready(reservation, program, epochs).is_err());
    assert!(!store.retire(reservation).unwrap());
    assert!(store.release(reservation).is_err());
}
pub fn bank_cancel<B: BankedResidency>(bank: &mut B, batch: BankBatch) {
    let ticket = bank.stage(batch).unwrap();
    assert_eq!(
        bank.cancel(&ticket).unwrap(),
        CancelState::PublicationRevoked
    );
    assert!(matches!(
        bank.publish(&ticket, ticket.epochs),
        Err(Error::Cancelled)
    ));
    assert!(!bank.retire(&ticket).unwrap());
}
pub fn rows_order<R: RowService>(rows: &mut R, batch: RowBatch) -> RowLease {
    let ids = batch.ids.clone();
    let ticket = rows.gather(batch).unwrap();
    let lease = rows.publish(&ticket, ticket.epochs).unwrap();
    assert_eq!(
        lease
            .records
            .iter()
            .map(|r| r.id().clone())
            .collect::<Vec<_>>(),
        ids
    );
    assert_eq!(rows.cancel(&ticket).unwrap(), CancelState::AlreadyPublished);
    assert!(rows.release(&lease).is_err());
    lease
}
pub fn peer_cancel<P: PeerBackend>(peer: &mut P, copies: Vec<ContiguousCopy>, consumer: u32) {
    let count = copies.len();
    let submission = match peer.submit(copies) {
        Ok(s) => s,
        Err(_) => panic!("fixture submission rejected"),
    };
    submission.validate(count).unwrap();
    let ticket = submission.ticket;
    assert_eq!(
        peer.cancel(&ticket).unwrap(),
        CancelState::PublicationRevoked
    );
    assert!(matches!(
        peer.materialize_local(&ticket, 0, ticket.epochs, consumer),
        Err(Error::Cancelled)
    ));
    assert!(!peer.retired(&ticket).unwrap());
    assert!(peer.acknowledge(&ticket).is_err());
}

// v1.1 adds schedules only. Runtime traits and wire v1 are unchanged.
mod revision_v11;
pub use revision_v11::*;

// v1.2 remains test-only; persisted wire and runtime trait semantics are unchanged.
mod revision_v12;
pub use revision_v12::*;

// v1.3 adds optional source retirement and concrete-owner hand-back schedules.
mod revision_v13;
pub use revision_v13::*;

// Day-11 rules (lead ruling 9) sit beside the frozen schedules, unversioned: a cancelled
// restore recovers its source; a continuation over a suspended layer is refused.
mod recovery;
pub use recovery::*;

// Day-19 rule (WP-A, memra#536 Move 1): the reader fence of an H2D issued off the owner stream,
// beside the frozen schedules, unversioned: `consumer_fenced` is the installed reader wait.
mod reader_fence;
pub use reader_fence::*;

// Day-20 rule (WP-A, memra#536 Move 2 slice 1): the event-ordered publication of a D2D capture
// issued off the owner stream, beside the frozen schedules, unversioned: a captured entry is
// published only after every copy's completion event; the receipt term is slice 3's.
mod d2d_capture;
pub use d2d_capture::*;

// Day-21 rule (WP-A, memra#536 Move 2 slice 2): the readiness of a D2D restore into a borrowed
// destination from a pinned borrowed source, beside the frozen schedules, unversioned: landed is
// not ready; ready is the landing plus the installed reader wait; a prime before it is unordered.
mod d2d_restore;
pub use d2d_restore::*;

// Day-22 rule (WP-A, memra#536 Move 2 slice 3): the receipt term of both D2D classes, beside the
// frozen schedules, unversioned: the destination digest witnesses the source digest; a receipt-less
// or mismatching item is refused Corrupt, the caller latches, nothing is published or primed on.
mod d2d_receipt;
pub use d2d_receipt::*;

// Day-30 rule (WP-A, memra#536 Move 2 owed item 1, the D2H half): the typed f32 spans of a demote
// batch, beside the frozen schedules, unversioned: one ticket and one landing with the KV items; a
// refused attach hands every span back; a span error quarantines the ticket.
mod d2h_span;
pub use d2h_span::*;

// Day-32 rule (WP-A, memra#536 Move 2 owed item 1, the H2D half): the typed f32 spans of a promote
// batch, beside the frozen schedules, unversioned: one ticket and one landing with the KV items; a
// destination is handed out only behind the reader wait; a span error quarantines the ticket.
mod h2d_span;
pub use h2d_span::*;

// Day-34 rule (WP-A, memra#536 Move 2 owed item 1, `DAY34.md` design K): an H2D batch whose completion
// checksums are supplied by the caller's hash helper, beside the frozen schedules, unversioned: one
// landing with the supplied checksums; the demote-time receipts gate it as before; sources stay owned
// while a view is out.
mod h2d_deferred_checksum;
pub use h2d_deferred_checksum::*;

// Day-38 rule (WP-A, memra#536 Move 1 owed item 2's hash 1, `DAY38.md` design G): a D2H batch whose
// receipt is the framed SHA-256 of each item's DEVICE source, taken on the device, beside the frozen
// schedules, unversioned: one landing with the receipt observed; the checksum names the source; the
// caller's re-hash of the landed bytes is the witness before publication.
mod d2h_device_receipt;
pub use d2h_device_receipt::*;
mod span_receipt;
pub use span_receipt::*;
