//! Reusable schedules: WP test targets may path-import this exact file and supply
//! real backend fixtures. Do not duplicate expected outcomes in each lane.
use memra_tier::contracts::*;

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
