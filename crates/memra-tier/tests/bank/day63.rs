//! Day 63 (`research/spill-c-20260919/DAY63.md` section 1): the bank stage clock's split is log only. Every write of
//! a day-63 field is an addition of a bracket's nanoseconds, the retire side's three calls still add to `retire_ns`,
//! and the line keeps the day-40 fields first, in their order.
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
        assert_eq!(writes, 1, "{field}: one write");
        assert_eq!(adds, 1, "{field}: the write adds a bracket");
    }
    assert_eq!(code.matches("c.retire_ns += ns").count(), 3);
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
