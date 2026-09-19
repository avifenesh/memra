use memra_tier::peer::{topology::*, *};
use std::{cell::RefCell, collections::BTreeMap, rc::Rc};

// Deliberately owner-thread-only fake. Not a second runtime allocator or transfer engine.
#[derive(Clone)]
struct Registration {
    device: u32,
    epoch: u64,
    data: Rc<RefCell<Vec<u8>>>,
}
impl PeerRegistration for Registration {
    fn device(&self) -> u32 {
        self.device
    }
    fn bytes(&self) -> u64 {
        self.data.borrow().len() as u64
    }
    fn epoch(&self) -> u64 {
        self.epoch
    }
}
#[derive(Debug)]
struct Ticket {
    id: u64,
    epoch: u64,
}
struct Ready {
    ticket: Ticket,
    destination: Registration,
}
struct Entry {
    copy: Option<ContiguousCopy<Registration>>,
    epoch: u64,
    progress: PeerProgress,
    cancelled: bool,
    published: bool,
}
#[derive(Default)]
struct FakePeer {
    next: u64,
    entries: BTreeMap<u64, Entry>,
}
impl FakePeer {
    fn entry(&mut self, ticket: &Ticket) -> Result<&mut Entry, PeerError> {
        let entry = self
            .entries
            .get_mut(&ticket.id)
            .ok_or(PeerError::ForeignLease)?;
        if entry.epoch != ticket.epoch {
            return Err(PeerError::StaleEpoch);
        }
        Ok(entry)
    }
    fn finish(&mut self, ticket: &Ticket) {
        let e = self.entry(ticket).unwrap();
        assert_eq!(e.progress, PeerProgress::Pending);
        if e.cancelled {
            e.copy = None;
            e.progress = PeerProgress::Retired;
            return;
        }
        let c = e.copy.as_ref().unwrap();
        let src = c.source.data.borrow();
        let start = c.source_span.offset() as usize;
        let end = start + c.source_span.bytes() as usize;
        let bytes = src[start..end].to_vec();
        drop(src);
        let mut dst = c.destination.data.borrow_mut();
        let start = c.destination_span.offset() as usize;
        dst[start..start + bytes.len()].copy_from_slice(&bytes);
        e.progress = PeerProgress::Complete {
            bytes: bytes.len() as u64,
            consumer_fence: 2,
        };
    }
}
impl PeerBackend for FakePeer {
    type Registration = Registration;
    type Ticket = Ticket;
    type LocalReady = Ready;
    fn submit(
        &mut self,
        epoch: u64,
        copies: Vec<ContiguousCopy<Registration>>,
    ) -> Vec<Result<Ticket, PeerError>> {
        copies
            .into_iter()
            .map(|copy| {
                copy.validate(epoch)?;
                if copy.producer_fence != 1 {
                    return Err(PeerError::Unavailable);
                }
                if self
                    .entries
                    .values()
                    .any(|e| e.progress != PeerProgress::Retired)
                {
                    return Err(PeerError::Capacity);
                }
                let id = self.next;
                self.next += 1;
                self.entries.insert(
                    id,
                    Entry {
                        copy: Some(copy),
                        epoch,
                        progress: PeerProgress::Pending,
                        cancelled: false,
                        published: false,
                    },
                );
                Ok(Ticket { id, epoch })
            })
            .collect()
    }
    fn poll(&mut self, ticket: &Ticket) -> Result<PeerProgress, PeerError> {
        let e = self.entry(ticket)?;
        Ok(if e.cancelled && e.progress == PeerProgress::Pending {
            PeerProgress::CancelPending
        } else {
            e.progress
        })
    }
    fn cancel(&mut self, ticket: &Ticket) -> Result<CancelState, PeerError> {
        let e = self.entry(ticket)?;
        if e.published {
            return Ok(CancelState::AlreadyPublished);
        }
        e.cancelled = true;
        if matches!(e.progress, PeerProgress::Complete { .. }) {
            e.progress = PeerProgress::Retired;
            e.copy = None;
        }
        Ok(CancelState::PublicationRevoked)
    }
    fn materialize_local(
        &mut self,
        ticket: &Ticket,
        current_epoch: u64,
        consumer: u32,
    ) -> Result<Ready, PeerError> {
        let e = self.entry(ticket)?;
        if current_epoch != e.epoch {
            return Err(PeerError::StaleEpoch);
        }
        if e.cancelled {
            return Err(PeerError::Cancelled);
        }
        if e.published {
            return Err(PeerError::Busy);
        }
        let PeerProgress::Complete { bytes, .. } = e.progress else {
            return Err(PeerError::Busy);
        };
        let c = e.copy.as_ref().unwrap();
        if c.destination.device != consumer {
            return Err(PeerError::Unavailable);
        }
        if bytes != c.destination_span.bytes() {
            return Err(PeerError::ShortCopy);
        }
        e.published = true;
        Ok(Ready {
            ticket: Ticket {
                id: ticket.id,
                epoch: ticket.epoch,
            },
            destination: c.destination.clone(),
        })
    }
    fn retire_consumer(&mut self, ready: &Ready) -> Result<(), PeerError> {
        let e = self.entry(&ready.ticket)?;
        e.progress = PeerProgress::Retired;
        e.copy = None;
        Ok(())
    }
}
fn copy() -> ContiguousCopy<Registration> {
    ContiguousCopy {
        source: Registration {
            device: 0,
            epoch: 7,
            data: Rc::new(RefCell::new(vec![1, 2, 3, 4])),
        },
        destination: Registration {
            device: 1,
            epoch: 7,
            data: Rc::new(RefCell::new(vec![0; 4])),
        },
        source_span: ContiguousSpan::new(0, 4, 4).unwrap(),
        destination_span: ContiguousSpan::new(0, 4, 4).unwrap(),
        producer_fence: 1,
    }
}

#[test]
fn per_item_acceptance_and_local_only_materialization() {
    let mut f = FakePeer::default();
    let items = f.submit(7, vec![copy(), copy()]);
    assert_eq!(items.len(), 2);
    assert_eq!(items[1].as_ref().unwrap_err(), &PeerError::Capacity);
    let ticket = items.into_iter().next().unwrap().unwrap();
    assert!(matches!(
        f.materialize_local(&ticket, 7, 1),
        Err(PeerError::Busy)
    ));
    f.finish(&ticket);
    assert!(matches!(
        f.materialize_local(&ticket, 7, 0),
        Err(PeerError::Unavailable)
    ));
    let ready = f.materialize_local(&ticket, 7, 1).unwrap();
    assert_eq!(*ready.destination.data.borrow(), [1, 2, 3, 4]);
    assert_eq!(f.cancel(&ticket), Ok(CancelState::AlreadyPublished));
    assert!(matches!(
        f.materialize_local(&ticket, 7, 1),
        Err(PeerError::Busy)
    ));
    f.retire_consumer(&ready).unwrap();
    assert_eq!(f.poll(&ticket), Ok(PeerProgress::Retired));
    assert!(f.submit(7, vec![copy()])[0].is_ok());
}

#[test]
fn late_completion_cancel_and_source_lifetime() {
    let mut f = FakePeer::default();
    let c = copy();
    let source = Rc::downgrade(&c.source.data);
    let destination = Rc::downgrade(&c.destination.data);
    let ticket = f.submit(7, vec![c]).remove(0).unwrap();
    assert_eq!(f.cancel(&ticket), Ok(CancelState::PublicationRevoked));
    assert_eq!(f.poll(&ticket), Ok(PeerProgress::CancelPending));
    assert!(source.upgrade().is_some() && destination.upgrade().is_some());
    assert_eq!(
        f.submit(7, vec![copy()])[0].as_ref().unwrap_err(),
        &PeerError::Capacity
    );
    f.finish(&ticket);
    assert!(source.upgrade().is_none() && destination.upgrade().is_none());
    assert!(matches!(
        f.materialize_local(&ticket, 7, 1),
        Err(PeerError::Cancelled)
    ));
    assert!(f.submit(7, vec![copy()])[0].is_ok());
}

#[test]
fn cancel_after_completion_before_publish_and_stale_epoch_refuse() {
    let mut f = FakePeer::default();
    let ticket = f.submit(7, vec![copy()]).remove(0).unwrap();
    f.finish(&ticket);
    assert!(matches!(
        f.materialize_local(&ticket, 8, 1),
        Err(PeerError::StaleEpoch)
    ));
    assert_eq!(
        f.poll(&Ticket {
            id: ticket.id,
            epoch: 8
        }),
        Err(PeerError::StaleEpoch)
    );
    assert_eq!(f.cancel(&ticket), Ok(CancelState::PublicationRevoked));
    assert!(matches!(
        f.materialize_local(&ticket, 7, 1),
        Err(PeerError::Cancelled)
    ));
    assert_eq!(
        f.submit(8, vec![copy()])[0].as_ref().unwrap_err(),
        &PeerError::StaleEpoch
    );
}

#[test]
fn short_and_unknown_completion_cannot_publish() {
    let mut f = FakePeer::default();
    let ticket = f.submit(7, vec![copy()]).remove(0).unwrap();
    f.entry(&ticket).unwrap().progress = PeerProgress::Quarantined;
    assert!(matches!(
        f.materialize_local(&ticket, 7, 1),
        Err(PeerError::Busy)
    ));
    assert_eq!(
        f.submit(7, vec![copy()])[0].as_ref().unwrap_err(),
        &PeerError::Capacity
    );
    f.entry(&ticket).unwrap().progress = PeerProgress::Complete {
        bytes: 3,
        consumer_fence: 2,
    };
    assert!(matches!(
        f.materialize_local(&ticket, 7, 1),
        Err(PeerError::ShortCopy)
    ));
}

#[test]
fn spans_cannot_encode_scatter_or_overflow_or_escape_registration() {
    assert_eq!(ContiguousSpan::new(3, 2, 4), Err(PeerError::InvalidRange));
    assert_eq!(
        ContiguousSpan::new(u64::MAX, 2, u64::MAX),
        Err(PeerError::InvalidRange)
    );
    assert_eq!(ContiguousSpan::new(0, 0, 4), Err(PeerError::InvalidRange));
    let mut c = copy();
    c.source_span = ContiguousSpan::new(0, 5, 5).unwrap();
    assert_eq!(c.validate(7), Err(PeerError::InvalidRange));
}

struct Lease {
    id: u64,
    issuer: u64,
    plan: PeerPlan,
}
impl PeerLease for Lease {
    fn plan(&self) -> PeerPlan {
        self.plan
    }
}
struct Governor {
    issuer: u64,
    epoch: u64,
    used: u64,
    next: u64,
    live: BTreeMap<u64, (u64, bool)>,
}
impl PeerCapacity for Governor {
    type Lease = Lease;
    fn reserve(&mut self, plan: PeerPlan) -> Result<Lease, PeerError> {
        if plan.epoch != self.epoch {
            return Err(PeerError::StaleEpoch);
        }
        if plan.device != 1 {
            return Err(PeerError::Unavailable);
        }
        if plan.bytes == 0 || plan.bytes > 8 - self.used {
            return Err(PeerError::Capacity);
        }
        let id = self.next;
        self.next += 1;
        self.used += plan.bytes;
        self.live.insert(id, (plan.bytes, false));
        Ok(Lease {
            id,
            issuer: self.issuer,
            plan,
        })
    }
    fn release(&mut self, lease: &Lease) -> Result<(), PeerError> {
        if lease.issuer != self.issuer {
            return Err(PeerError::ForeignLease);
        }
        let (bytes, busy) = self.live.get(&lease.id).ok_or(PeerError::AlreadyReleased)?;
        if *busy {
            return Err(PeerError::Busy);
        }
        self.used -= bytes;
        self.live.remove(&lease.id);
        Ok(())
    }
}
#[test]
fn global_lease_accounting_busy_release_and_epoch_reclamation() {
    let mut g = Governor {
        issuer: 9,
        epoch: 7,
        used: 0,
        next: 0,
        live: BTreeMap::new(),
    };
    let plan = PeerPlan {
        device: 1,
        bytes: 8,
        epoch: 7,
    };
    let lease = g.reserve(plan).unwrap();
    assert_eq!(lease.plan(), plan);
    assert!(matches!(g.reserve(plan), Err(PeerError::Capacity)));
    g.live.get_mut(&lease.id).unwrap().1 = true;
    assert_eq!(g.release(&lease), Err(PeerError::Busy));
    assert_eq!(g.used, 8);
    g.epoch = 8;
    assert!(matches!(g.reserve(plan), Err(PeerError::StaleEpoch)));
    g.live.get_mut(&lease.id).unwrap().1 = false;
    assert_eq!(g.release(&lease), Ok(())); // old epoch can reclaim, cannot publish
    assert_eq!(g.used, 0);
    assert_eq!(g.release(&lease), Err(PeerError::AlreadyReleased));
    assert_eq!(
        g.release(&Lease {
            id: 0,
            issuer: 10,
            plan
        }),
        Err(PeerError::ForeignLease)
    );
}

fn topology() -> TopologySnapshot {
    use Observation::Known;
    TopologySnapshot {
        devices: (0..4)
            .map(|device| DeviceTopology {
                device,
                numa_node: Known(device / 2),
                cpu_affinity: vec![device * 2],
                current_generation: Known(5),
                max_generation: Known(5),
                current_width: Known(16),
                max_width: Known(16),
                idle_p8: Known(false),
            })
            .collect(),
        edges: (0..4)
            .flat_map(|source| {
                (0..4)
                    .filter(move |&d| d != source)
                    .map(move |destination| PeerEdge {
                        source,
                        destination,
                        can_access: Known(true),
                        context_granted: Known(true),
                        pool_granted: Known(true),
                        direct_copy_qualified: Known(true),
                        shared_fabric: Some("synthetic-shared-uplink".into()),
                    })
            })
            .collect(),
    }
}
#[test]
fn topology_is_directed_grant_bound_and_never_host_bounce() {
    let mut t = topology();
    assert_eq!(t.edges.len(), 12);
    assert!(t.peer_available(0, 1));
    t.edges
        .iter_mut()
        .find(|e| e.source == 0 && e.destination == 1)
        .unwrap()
        .pool_granted = Observation::Known(false);
    assert!(!t.peer_available(0, 1));
    assert!(t.peer_available(1, 0));
    t.edges[0].pool_granted = Observation::Known(true);
    t.edges[0].direct_copy_qualified = Observation::Unknown;
    assert!(!t.peer_available(0, 1));
    assert!(!t.peer_available(0, 0));
    assert!(!t.peer_available(0, 7));
}
#[test]
fn idle_downshift_is_deferred_active_downgrade_refuses() {
    let mut t = topology();
    t.devices[0].current_generation = Observation::Known(1);
    t.devices[0].idle_p8 = Observation::Known(true);
    assert_eq!(t.devices[0].link_health(), LinkHealth::IdleDeferred);
    assert!(!t.peer_available(0, 1));
    t.devices[0].idle_p8 = Observation::Known(false);
    assert_eq!(t.devices[0].link_health(), LinkHealth::Downgraded);
    t.devices[0].current_generation = Observation::Unknown;
    assert_eq!(t.devices[0].link_health(), LinkHealth::Unknown);
}
