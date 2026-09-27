//! Day-11 rules (lead ruling 9): fault-arm findings became contracts, not patches. Two rules
//! beside the frozen schedules, additive and unversioned: every v1 to v1.3 schedule is
//! byte-identical and `WIRE_VERSION` stays 1. Native bindings observe real events and pins;
//! CPU bindings model ownership only. Nothing here qualifies a backend by itself.
//!
//! Rule 1, cancelled restore. A restore (H2D) that is cancelled before its consumer event
//! completes must either hand the untouched host source back to the caller as a typed lease
//! (recoverable) or refuse the cancel with a typed error while the source is still intact;
//! consuming the source and then cancelling is forbidden by the rule. Finding (lane D, day 11,
//! verbatim): "No H2D-source recovery after `cancel`: the demoted copy's only caller handle is
//! consumed at submission and released by `retire`; the D2H twin is take-once
//! (`AlreadyReleased`). A cancelled restore can only be drained."
//!
//! Rule 2, required-resident continuation. A continuation (decode or prime) over a cache with
//! any suspended layer must be refused at the continuation gate with a typed error naming the
//! suspended layers, unless the caller restores first. Finding (lane D, day 11, verbatim):
//! "`Cache::ensure_usable` returned `Ok(())` on the fully suspended cache (observed),
//! `decode_step_h` unwraps a suspended layer (`hybrid_forward.rs`),
//! `RestoreDecision::RequireState` is load-vs-recompute only."
use crate::contracts::*;

/// Owner hooks for rule 1. `Producer` completes the copy's DMA, `Unknown` loses the
/// observation, `Recover` re-observes the recorded event, `SourceConsumer` completes the last
/// reader of the source, `SourceGraph` releases the last source graph pin.
#[derive(Clone, Copy, Debug)]
pub enum RecoverStep {
    Unknown,
    Producer,
    Recover,
    SourceConsumer,
    SourceGraph,
}

/// Rule 1, the recoverable outcome. Fixture starts with the H2D submitted, its producer pending
/// and a live source consumer plus source graph pin; the destination side is idle (an
/// unpublished restore has no destination consumer). `source_held` asserts the source's charge
/// and registry entry are still the ticket's; it runs at every refusal. Returns the recovered
/// lease so the binding can check its bytes and release its charge.
pub fn transfer_cancel_recovers_source<T: TransferEngine>(
    engine: &mut T,
    ticket: TransferTicket,
    mut advance: impl FnMut(&mut T, &TransferTicket, RecoverStep),
    mut source_held: impl FnMut(&mut T),
) -> T::Host {
    use RecoverStep::*;
    // A live ticket keeps its source: recovery is offered only to a revoked publication.
    assert!(matches!(
        engine.recover_source(&ticket, 0),
        Err(Error::NotReady)
    ));
    let mut foreign = ticket;
    foreign.issuer ^= u64::MAX;
    assert!(engine.recover_source(&foreign, 0).is_err());
    source_held(engine);
    assert_eq!(
        engine.cancel(&ticket).unwrap(),
        CancelState::PublicationRevoked
    );
    assert!(matches!(
        engine.ready_view(&ticket, 0, ticket.epochs),
        Err(Error::Cancelled)
    ));
    assert!(matches!(
        engine.take_destination(&ticket, 0, ticket.epochs),
        Err(Error::Cancelled)
    ));
    for step in [
        None,
        Some(Unknown),
        Some(Producer),
        Some(Recover),
        Some(SourceConsumer),
    ] {
        if let Some(step) = step {
            advance(engine, &ticket, step);
        }
        // The copy or a source consumer may still read the source: no hand-back, no drain.
        assert!(matches!(
            engine.recover_source(&ticket, 0),
            Err(Error::Busy | Error::Quarantined)
        ));
        assert!(matches!(
            engine.retire(&ticket, None),
            Err(Error::Busy | Error::Quarantined)
        ));
        assert!(matches!(
            engine.retire_source(&ticket),
            Err(Error::Busy | Error::Quarantined)
        ));
        assert!(!engine.retired(&ticket).unwrap());
        assert_eq!(engine.acknowledge(&ticket), Err(Error::Busy));
        source_held(engine);
    }
    advance(engine, &ticket, SourceGraph);
    // Idle and revoked, still not recovered: the engine may not drain the source.
    assert_eq!(engine.retire(&ticket, None), Err(Error::Busy));
    assert_eq!(engine.retire_source(&ticket), Err(Error::Busy));
    assert!(!engine.retired(&ticket).unwrap());
    source_held(engine);
    let host = engine.recover_source(&ticket, 0).unwrap();
    assert!(matches!(
        engine.recover_source(&ticket, 0),
        Err(Error::AlreadyReleased)
    ));
    // The source left the ticket: a revocation can no longer be granted over it.
    assert_eq!(engine.cancel(&ticket), Err(Error::AlreadyReleased));
    engine.retire_source(&ticket).unwrap(); // idempotent: nothing is left on the source side
    engine.retire(&ticket, None).unwrap();
    assert!(engine.retired(&ticket).unwrap());
    engine.acknowledge(&ticket).unwrap();
    assert!(matches!(
        engine.recover_source(&ticket, 0),
        Err(Error::UnknownTicket)
    ));
    host
}

/// Rule 1, the forbidden order. `complete` observes the producer complete with the source idle.
/// Once the source has left the ticket through its owner's per-side retirement, `cancel` is
/// refused and no recovery is offered; the ticket retires and acknowledges normally.
pub fn transfer_cancel_refused_after_source_consumed<T: TransferEngine>(
    engine: &mut T,
    ticket: TransferTicket,
    complete: impl FnOnce(&mut T, &TransferTicket),
) {
    complete(engine, &ticket);
    engine.retire_source(&ticket).unwrap();
    assert_eq!(engine.cancel(&ticket), Err(Error::AlreadyReleased));
    assert!(matches!(
        engine.recover_source(&ticket, 0),
        Err(Error::NotReady)
    ));
    engine.retire(&ticket, None).unwrap();
    assert!(engine.retired(&ticket).unwrap());
    engine.acknowledge(&ticket).unwrap();
}

/// Rule 2 fixture: a continuation gate over a cache whose layers can leave residency. `suspend`
/// moves one layer's state out to a tier, `restore` brings it back, `continuation` asks the gate
/// whether a decode or prime may run and, on refusal, returns the suspended layers the typed
/// error named, ascending.
pub trait ContinuationGateFixture {
    fn suspend(&mut self, layer: u32);
    fn restore(&mut self, layer: u32);
    fn continuation(&mut self) -> std::result::Result<(), Vec<u32>>;
}

/// Rule 2. `layers` are distinct; at least two, so a partial restore is exercised. A whole cache
/// continues; one suspended layer refuses and is named; asking is not restoring; every suspended
/// layer is named; a partial restore still refuses, naming what is left; a restored cache
/// continues again.
pub fn required_resident_continuation<F: ContinuationGateFixture>(f: &mut F, layers: &[u32]) {
    assert!(layers.len() >= 2, "the rule needs a partial restore");
    let mut every = layers.to_vec();
    every.sort_unstable();
    every.dedup();
    assert_eq!(every.len(), layers.len(), "layers must be distinct");
    f.continuation().unwrap();
    f.suspend(layers[0]);
    assert_eq!(f.continuation(), Err(vec![layers[0]]));
    assert_eq!(f.continuation(), Err(vec![layers[0]])); // asking is not restoring
    for layer in &layers[1..] {
        f.suspend(*layer);
    }
    assert_eq!(f.continuation(), Err(every));
    f.restore(layers[0]);
    let mut rest = layers[1..].to_vec();
    rest.sort_unstable();
    assert_eq!(f.continuation(), Err(rest));
    for layer in &layers[1..] {
        f.restore(*layer);
    }
    f.continuation().unwrap();
    f.continuation().unwrap();
}
