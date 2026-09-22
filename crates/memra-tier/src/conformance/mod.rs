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
