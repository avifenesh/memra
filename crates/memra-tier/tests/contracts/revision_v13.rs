//! v1.3 additive schedules. Native bindings must observe real events, not set flags.
use memra_tier::contracts::*;

#[derive(Clone, Copy)]
pub enum SourceStep {
    Unknown,
    Producer,
    Recover,
    SourceConsumer,
    SourceGraph,
    DestinationConsumer,
    DestinationGraph,
}

/// Fixture starts with producer pending and live source/destination consumer + graph pins.
/// The release hook attempts the ORIGINAL source charge; it must not fabricate a balance.
pub fn transfer_source_retirement<T: TransferEngine>(
    engine: &mut T,
    ticket: TransferTicket,
    mut advance: impl FnMut(&mut T, &TransferTicket, SourceStep),
    mut release_source: impl FnMut(&mut T) -> Result<()>,
    mut destination_live: impl FnMut(&mut T),
) {
    use SourceStep::*;
    let mut foreign = ticket;
    foreign.issuer ^= u64::MAX;
    assert!(engine.retire_source(&foreign).is_err());
    let mut stale = ticket;
    stale.epochs.src_gen += 1;
    assert!(engine.retire_source(&stale).is_err());
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
        assert!(matches!(
            engine.retire_source(&ticket),
            Err(Error::Busy | Error::Quarantined)
        ));
        assert_eq!(release_source(engine), Err(Error::Busy));
        assert!(!engine.retired(&ticket).unwrap());
        assert_eq!(engine.acknowledge(&ticket), Err(Error::Busy));
        destination_live(engine);
    }
    advance(engine, &ticket, SourceGraph);
    engine.retire_source(&ticket).unwrap();
    engine.retire_source(&ticket).unwrap(); // idempotent, not a second release
    release_source(engine).unwrap();
    assert!(!engine.retired(&ticket).unwrap());
    assert_eq!(engine.acknowledge(&ticket), Err(Error::Busy));
    destination_live(engine);
    advance(engine, &ticket, DestinationConsumer);
    assert!(!engine.retired(&ticket).unwrap());
    advance(engine, &ticket, DestinationGraph);
    assert!(engine.retired(&ticket).unwrap());
    engine.acknowledge(&ticket).unwrap();
    assert_eq!(engine.retire_source(&ticket), Err(Error::UnknownTicket));
    // The taken destination must outlive even the whole-ticket tombstone.
    destination_live(engine);
}

#[derive(Clone, Copy)]
pub enum HandBackStep {
    Unknown,
    Producer,
    Recover,
    Consumer,
    Graph,
}

/// Concrete owner adapter, NOT a new TransferEngine associated CUDA type.
/// `take` returns native backing on CUDA; CPU binding returns its original Vec.
pub trait DeviceHandBackFixture {
    type Backing: std::fmt::Debug;
    fn take(&mut self) -> Result<Self::Backing>;
    fn advance(&mut self, step: HandBackStep);
    fn assert_registered_and_charged(&mut self);
    fn assert_returned_once(&mut self, backing: Self::Backing);
    fn reject_foreign(&mut self);
}

pub fn device_hand_back<F: DeviceHandBackFixture>(f: &mut F) {
    use HandBackStep::*;
    f.reject_foreign();
    for step in [
        None,
        Some(Unknown),
        Some(Producer),
        Some(Recover),
        Some(Consumer),
    ] {
        if let Some(step) = step {
            f.advance(step);
        }
        assert!(matches!(f.take(), Err(Error::Busy | Error::Quarantined)));
        f.assert_registered_and_charged();
    }
    f.advance(Graph);
    let backing = f.take().unwrap();
    assert!(f.take().is_err());
    f.assert_returned_once(backing);
}
