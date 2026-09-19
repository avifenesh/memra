use super::support::*;
use memra_tier::contracts::*;
use std::collections::HashMap;
struct Entry {
    copies: Vec<ContiguousCopy>,
    c: Completion,
    cancelled: bool,
    published: bool,
    done: bool,
    unknown: bool,
    graph: bool,
}
struct Peer {
    gov: Shared,
    owners: [DeviceOwner; 2],
    entries: HashMap<TransferTicket, Entry>,
    next: u64,
    grants: bool,
    reject: Option<usize>,
}
impl Peer {
    fn new(gov: Shared) -> Self {
        Self {
            gov,
            owners: [DeviceOwner::new(0), DeviceOwner::new(1)],
            entries: HashMap::new(),
            next: 0,
            grants: true,
            reject: None,
        }
    }
    fn plan(&self, device: u32, generation: u64) -> PeerPlan {
        let mut r = request(0, Priority::Demand);
        r.bytes.device[device as usize] = 4;
        r.bytes.peer[device as usize] = 4;
        PeerPlan {
            owner_device: device,
            consumer_device: 1 - device,
            bytes: 4,
            alignment: 4,
            epochs: Epochs {
                dst_gen: generation,
                ..epochs()
            },
            request: r,
        }
    }
    fn copy(&self, source: &PeerLease, dest: &PeerLease) -> ContiguousCopy {
        ContiguousCopy::new(
            self.owners[0].retain(&source.device).unwrap(),
            self.owners[1].retain(&dest.device).unwrap(),
            ContiguousSpan::new(0, 4, 4).unwrap(),
            ContiguousSpan::new(0, 4, 4).unwrap(),
            epochs(),
            FenceId {
                issuer: self.owners[0].issuer(),
                owner: 0,
                generation: 19,
                sequence: 1,
            },
        )
        .unwrap()
    }
}
impl PeerCapacity for Peer {
    fn reserve(&mut self, p: PeerPlan) -> Result<PeerLease> {
        if !self.grants {
            return Err(Error::Unsupported);
        }
        if p.epochs.state != 7 {
            return Err(Error::StaleEpoch);
        }
        if !p.alignment.is_power_of_two() || !p.bytes.is_multiple_of(u64::from(p.alignment)) {
            return Err(Error::InvalidLayout);
        }
        let charge = self.gov.borrow_mut().reserve(&p.request)?;
        let device = self.owners[p.owner_device as usize].register(
            p.epochs.dst_gen,
            p.bytes,
            Box::new(bytes(p.bytes as usize)),
            &charge,
        )?;
        Ok(PeerLease {
            plan: p,
            charge,
            device,
        })
    }
    fn release(&mut self, l: &PeerLease) -> Result<()> {
        self.owners[l.plan.owner_device as usize].release(&l.device)?;
        self.gov.borrow_mut().release(&l.charge)
    }
}
impl PeerBackend for Peer {
    fn submit(&mut self, copies: Vec<ContiguousCopy>) -> Submission<ContiguousCopy> {
        if copies.is_empty() || !self.grants {
            return Err(Rejected {
                op: copies,
                error: Error::Unsupported,
            });
        }
        let epoch = copies[0].epochs;
        if copies.iter().any(|c| c.validate(epoch).is_err()) {
            return Err(Rejected {
                op: copies,
                error: Error::StaleEpoch,
            });
        }
        if copies.len() == 1 && self.reject == Some(0) {
            return Err(Rejected {
                op: copies,
                error: Error::Rejected,
            });
        }
        self.next += 1;
        let t = TransferTicket {
            issuer: 45,
            sequence: self.next,
            epochs: epoch,
        };
        let mut c = completion(t, copies.len());
        let mut accepted = vec![];
        let mut items = vec![];
        for (i, copy) in copies.into_iter().enumerate() {
            c.items[i].segments[0]
                .consumer_fence
                .as_mut()
                .unwrap()
                .issuer = self.owners[copy.destination().device() as usize].issuer();
            c.items[i].segments[0]
                .consumer_fence
                .as_mut()
                .unwrap()
                .owner = copy.destination().device();
            if self.reject == Some(i) {
                c.items[i].accepted = false;
                c.items[i].segments[0].status = ItemStatus::Rejected;
                items.push(ItemAcceptance::Rejected {
                    item: i as u32,
                    op: copy,
                    error: Error::Capacity,
                });
            } else {
                self.owners[copy.destination().device() as usize]
                    .bind_destination(t, copy.destination())
                    .unwrap();
                accepted.push(copy);
                items.push(ItemAcceptance::Accepted { item: i as u32 });
            }
        }
        self.entries.insert(
            t,
            Entry {
                copies: accepted,
                c,
                cancelled: false,
                published: false,
                done: false,
                unknown: false,
                graph: false,
            },
        );
        Ok(BatchSubmission { ticket: t, items })
    }
    fn poll(&mut self, t: &TransferTicket) -> Result<Completion> {
        let e = self.entries.get(t).ok_or(Error::UnknownTicket)?;
        if e.unknown {
            Err(Error::Quarantined)
        } else {
            Ok(e.c.clone())
        }
    }
    fn cancel(&mut self, t: &TransferTicket) -> Result<CancelState> {
        let e = self.entries.get_mut(t).ok_or(Error::UnknownTicket)?;
        if e.published {
            Ok(CancelState::AlreadyPublished)
        } else {
            e.cancelled = true;
            Ok(CancelState::PublicationRevoked)
        }
    }
    fn materialize_local(
        &mut self,
        t: &TransferTicket,
        item: u32,
        current: Epochs,
        consumer: u32,
    ) -> Result<ReadyView<'_>> {
        let e = self.entries.get_mut(t).ok_or(Error::UnknownTicket)?;
        if e.cancelled {
            return Err(Error::Cancelled);
        }
        if e.unknown {
            return Err(Error::Quarantined);
        }
        t.epochs.require(current)?;
        let dst = e
            .copies
            .get(item as usize)
            .ok_or(Error::NotFound)?
            .destination();
        if dst.device() != consumer {
            return Err(Error::WrongOwner);
        }
        let ready = self.owners[consumer as usize].ready_view(
            dst,
            &e.c,
            &expected(e.c.items.len()),
            current,
        )?;
        e.published = true;
        Ok(ready)
    }
    fn retire_consumer(&mut self, t: &TransferTicket, f: FenceId) -> Result<()> {
        if f.owner != 1 || f.generation != t.epochs.dst_gen {
            return Err(Error::WrongOwner);
        }
        if !self.retired(t)? {
            return Err(Error::Busy);
        }
        Ok(())
    }
    fn retired(&mut self, t: &TransferTicket) -> Result<bool> {
        let e = self.entries.get(t).ok_or(Error::UnknownTicket)?;
        Ok(e.done && e.graph && !e.unknown)
    }
    fn acknowledge(&mut self, t: &TransferTicket) -> Result<()> {
        if !self.retired(t)? {
            return Err(Error::Busy);
        }
        let e = self.entries.remove(t).unwrap();
        let mut devices = std::collections::HashSet::new();
        for copy in &e.copies {
            devices.insert(copy.destination().device());
        }
        for d in devices {
            self.owners[d as usize].retire_binding(t)?;
        }
        drop(e);
        Ok(())
    }
}
#[test]
fn peer_capacity_grants_budget_source_lifetime_and_explicit_release() {
    let gov = shared();
    let mut p = Peer::new(gov.clone());
    p.grants = false;
    assert!(matches!(p.reserve(p.plan(0, 19)), Err(Error::Unsupported)));
    p.grants = true;
    let mut stale = p.plan(0, 19);
    stale.epochs.state = 6;
    assert!(matches!(p.reserve(stale), Err(Error::StaleEpoch)));
    let src = p.reserve(p.plan(0, 19)).unwrap();
    let dst = p.reserve(p.plan(1, 31)).unwrap();
    let batch = p.submit(vec![p.copy(&src, &dst)]).unwrap();
    let t = batch.ticket;
    assert_eq!(gov.borrow().used.device, vec![4, 4]);
    assert_eq!(p.release(&src), Err(Error::Busy));
    assert_eq!(p.release(&dst), Err(Error::Busy));
    p.entries.get_mut(&t).unwrap().unknown = true;
    assert_eq!(p.poll(&t), Err(Error::Quarantined));
    assert!(!p.retired(&t).unwrap());
    p.cancel(&t).unwrap();
    p.entries.get_mut(&t).unwrap().unknown = false;
    p.entries.get_mut(&t).unwrap().done = true;
    assert!(!p.retired(&t).unwrap());
    assert_eq!(p.acknowledge(&t), Err(Error::Busy));
    p.entries.get_mut(&t).unwrap().graph = true;
    p.acknowledge(&t).unwrap();
    p.release(&src).unwrap();
    p.release(&dst).unwrap();
    assert_eq!(gov.borrow().used.device, vec![0, 0]);
    assert!(p.release(&dst).is_err());
}
#[test]
fn peer_partial_vector_cannot_materialize_missing_sibling() {
    let mut p = Peer::new(shared());
    let src = p.reserve(p.plan(0, 19)).unwrap();
    let dst = p.reserve(p.plan(1, 31)).unwrap();
    p.reject = Some(1);
    let b = p
        .submit(vec![
            p.copy(&src, &dst),
            p.copy(&src, &dst),
            p.copy(&src, &dst),
        ])
        .unwrap();
    b.validate(3).unwrap();
    assert_eq!(p.poll(&b.ticket).unwrap().items.len(), 3);
    assert!(matches!(
        p.materialize_local(&b.ticket, 0, epochs(), 1),
        Err(Error::Rejected)
    ));
}
#[test]
fn peer_three_epochs_local_only_and_cancel_linearization() {
    let mut p = Peer::new(shared());
    let src = p.reserve(p.plan(0, 19)).unwrap();
    let dst = p.reserve(p.plan(1, 31)).unwrap();
    let t = p.submit(vec![p.copy(&src, &dst)]).unwrap().ticket;
    for e in [
        Epochs {
            state: 8,
            ..epochs()
        },
        Epochs {
            src_gen: 20,
            ..epochs()
        },
        Epochs {
            dst_gen: 32,
            ..epochs()
        },
    ] {
        assert!(matches!(
            p.materialize_local(&t, 0, e, 1),
            Err(Error::StaleEpoch)
        ));
    }
    assert!(matches!(
        p.materialize_local(&t, 0, epochs(), 0),
        Err(Error::WrongOwner)
    ));
    assert_eq!(
        p.materialize_local(&t, 0, epochs(), 1)
            .unwrap()
            .destination()
            .device(),
        1
    );
    assert_eq!(p.cancel(&t), Ok(CancelState::AlreadyPublished));
    let t2 = p.submit(vec![p.copy(&src, &dst)]).unwrap().ticket;
    assert_eq!(p.cancel(&t2), Ok(CancelState::PublicationRevoked));
    assert!(matches!(
        p.materialize_local(&t2, 0, epochs(), 1),
        Err(Error::Cancelled)
    ));
}
#[test]
fn contiguous_bounds_overflow_and_foreign_owner_fail_closed() {
    assert_eq!(ContiguousSpan::new(0, 0, 4), Err(Error::InvalidLayout));
    assert_eq!(
        ContiguousSpan::new(u64::MAX, 2, u64::MAX),
        Err(Error::InvalidLayout)
    );
    assert_eq!(ContiguousSpan::new(3, 2, 4), Err(Error::InvalidLayout));
    let mut p = Peer::new(shared());
    let src = p.reserve(p.plan(0, 19)).unwrap();
    assert!(matches!(
        p.owners[1].retain(&src.device),
        Err(Error::WrongOwner)
    ));
}
#[test]
fn reusable_peer_schedule() {
    let mut p = Peer::new(shared());
    let src = p.reserve(p.plan(0, 19)).unwrap();
    let dst = p.reserve(p.plan(1, 31)).unwrap();
    let copies = vec![p.copy(&src, &dst)];
    super::conformance::peer_cancel(&mut p, copies, 1);
}
