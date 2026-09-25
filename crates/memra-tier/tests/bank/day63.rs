//! Day 63 (`research/spill-c-20260919/DAY63.md`): the bank stage clock's split is log only (every write of a day-63
//! field, and of `retire_ns`, is an addition of a bracket's nanoseconds, and the line keeps the day-40 fields first,
//! in their order), and I13's changes keep what they replace.
use super::*;

const RESIDENCY: &str = include_str!("../../src/bank/residency.rs");

#[test]
fn the_split_only_adds_nanoseconds() {
    let code = &RESIDENCY[..RESIDENCY.find("#[cfg(test)]").unwrap_or(RESIDENCY.len())];
    for field in [
        "stage_lookup_ns",
        "stage_cache_ns",
        "stage_charge_ns",
        "publish_output_ns",
        "publish_policy_ns",
        "host_use_ns",
        "retire_only_ns",
        "ack_ns",
        "ack_release_ns",
    ] {
        let writes = code.matches(&format!("c.{field} ")).count();
        let adds = code.matches(&format!("c.{field} += ns")).count();
        assert!(writes >= 1, "{field}: written");
        assert_eq!(adds, writes, "{field}: every write adds a bracket");
    }
    // The retire side still counts into `retire_ns`, by additions only.
    let retire = code.matches("c.retire_ns ").count();
    assert!(retire >= 3);
    assert_eq!(code.matches("c.retire_ns += ns").count(), retire);
    let line = &code[code.find("pub fn line(&self)").unwrap()..];
    let format = &line[..line.find(")\n").unwrap()];
    assert!(format.contains(
        "stages={} stage_ns={} alloc_ns={} steps={} step_ns={} verified={} verify_ns={} publish_ns={} retire_ns={} collect_ns={} "
    ));
}

/// I13 change 1: the in-place add and subtract give the allocating form's value or its error, and an error leaves
/// the budget unchanged; randomized budgets with values near the limits, both device counts, and the invalid shapes.
#[test]
fn the_in_place_budget_arithmetic_is_the_allocating_form() {
    let mut s: u64 = 0x2545_F491_4F6C_DD1D;
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };
    let value = |next: &mut dyn FnMut() -> u64| match next() % 4 {
        0 => 0,
        1 => next() % 1000,
        2 => u64::MAX - next() % 1000,
        _ => next(),
    };
    for round in 0..20_000 {
        let devices = 1 + (round % 2) as usize;
        let budget = |next: &mut dyn FnMut() -> u64| {
            let mut b = TierBudget::zero(devices);
            for d in 0..devices {
                b.device[d] = value(next);
                b.peer[d] = value(next);
                b.replicas[d] = value(next);
            }
            b.pinned = value(next);
            b.pageable = value(next);
            b.staging = value(next);
            b.loaders = value(next);
            b.nvme = value(next);
            b.inflight = value(next);
            b
        };
        let a = budget(&mut next);
        let mut b = budget(&mut next);
        if round % 97 == 0 {
            b.version += 1;
        }
        if round % 89 == 0 {
            b.peer.pop();
        }
        for add in [true, false] {
            let want = if add {
                a.checked_add(&b)
            } else {
                a.checked_sub(&b)
            };
            let mut got = a.clone();
            let result = got.combine_in_place(&b, add);
            assert_eq!(
                a.check_combine(&b, add),
                want.as_ref().map(|_| ()).map_err(Clone::clone)
            );
            match want {
                Ok(v) => {
                    assert_eq!(result, Ok(()));
                    assert_eq!(got, v);
                }
                Err(e) => {
                    assert_eq!(result, Err(e));
                    assert_eq!(got, a, "an error changes nothing");
                }
            }
        }
    }
}

/// I13 change 2: a clone of a published lease is the same publication (one body) and reads what the lease reads;
/// a second ticket over the cached record lends the same publication; a clone outlives its source's drop.
#[test]
fn a_lease_clone_shares_its_body() {
    let g = gov();
    let (mut b, ids) = banks(LayoutClass::PerRecord, 1_000, Reader::default(), g.clone());
    let t = b
        .stage(batch(vec![ids[0].clone(), ids[1].clone()]))
        .unwrap();
    let leases = publish_bank(&mut b, &t, epochs()).unwrap();
    for lease in &leases {
        let clone = lease.clone();
        assert!(clone.same_publication(lease));
        assert_eq!(clone.id(), lease.id());
        assert_eq!(clone.layout(), lease.layout());
        assert_eq!(clone.charge().id(), lease.charge().id());
        assert_eq!(
            &*clone.resource::<Vec<u8>>().unwrap(),
            &expected(lease.layout())
        );
    }
    assert!(!leases[0].same_publication(&leases[1]));
    finish(&mut b, &t);
    // A second ticket over the cached record lends the same publication.
    let t2 = b.stage(batch(vec![ids[0].clone()])).unwrap();
    let again = publish_bank(&mut b, &t2, epochs()).unwrap();
    assert!(again[0].same_publication(&leases[0]));
    finish(&mut b, &t2);
    let clone = leases[1].clone();
    drop(leases);
    drop(again);
    assert_eq!(
        &*clone.resource::<Vec<u8>>().unwrap(),
        &expected(clone.layout())
    );
}

/// The three retire-side calls as `SlruExpertDispatch::finish` made them before day 63.
fn three_calls(b: &mut Banks, t: &TransferTicket) -> Result<()> {
    b.finish_host_use(t)?;
    if !b.retire(t)? {
        return Err(Error::NotReady);
    }
    b.acknowledge(t)
}

/// I13 change 3: `finish_ticket` answers as the three calls did on twin banks, case by case (unpublished, cancelled
/// before the producer finished, published, a second finish of the same ticket, a cache hit), and leaves the same
/// governor totals and the same cache behind.
#[test]
fn finish_ticket_is_the_three_calls() {
    let (ga, gb) = (gov(), gov());
    let (mut a, ids) = banks(LayoutClass::PerRecord, 1_000, Reader::default(), ga.clone());
    let (mut b, _) = banks(LayoutClass::PerRecord, 1_000, Reader::default(), gb.clone());
    let both = |a: &mut Banks, b: &mut Banks, ta: &TransferTicket, tb: &TransferTicket| {
        let ra = three_calls(a, ta);
        let rb = b.finish_ticket(tb);
        assert_eq!(ra, rb);
        ra
    };
    let totals = |g: &Rc<RefCell<Governor>>| g.borrow().used();
    // Unpublished: Busy, and nothing changes.
    let ta = a.stage(batch(vec![ids[0].clone()])).unwrap();
    let tb = b.stage(batch(vec![ids[0].clone()])).unwrap();
    assert_eq!(both(&mut a, &mut b, &ta, &tb), Err(Error::Busy));
    assert_eq!(totals(&ga), totals(&gb));
    // Cancelled before the producer finished: host use marked, then NotReady; after the pump, it finishes.
    a.cancel(&ta).unwrap();
    b.cancel(&tb).unwrap();
    assert_eq!(both(&mut a, &mut b, &ta, &tb), Err(Error::NotReady));
    drain(&mut a, &ta);
    drain(&mut b, &tb);
    assert_eq!(both(&mut a, &mut b, &ta, &tb), Ok(()));
    assert_eq!(both(&mut a, &mut b, &ta, &tb), Err(Error::UnknownTicket));
    assert_eq!(totals(&ga), totals(&gb));
    // Published misses, then a hit over the cached records.
    for order in [
        vec![ids[0].clone(), ids[1].clone()],
        vec![ids[1].clone()],
        vec![ids[2].clone(), ids[0].clone()],
    ] {
        let ta = a.stage(batch(order.clone())).unwrap();
        let tb = b.stage(batch(order)).unwrap();
        let la = publish_bank(&mut a, &ta, epochs()).unwrap();
        let lb = publish_bank(&mut b, &tb, epochs()).unwrap();
        assert_eq!(
            la.iter().map(|l| l.id().clone()).collect::<Vec<_>>(),
            lb.iter().map(|l| l.id().clone()).collect::<Vec<_>>()
        );
        assert_eq!(totals(&ga), totals(&gb));
        assert_eq!(both(&mut a, &mut b, &ta, &tb), Ok(()));
        assert_eq!(both(&mut a, &mut b, &ta, &tb), Err(Error::UnknownTicket));
        a.collect_evicted().unwrap();
        b.collect_evicted().unwrap();
        assert_eq!(totals(&ga), totals(&gb));
        assert_eq!(a.cached_records(), b.cached_records());
        assert_eq!(a.owned_leases(), b.owned_leases());
    }
}
