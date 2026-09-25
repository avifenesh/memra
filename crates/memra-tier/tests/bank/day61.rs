//! Day 61 (`research/spill-c-20260919/DAY61.md`, I11): the lease protocol's repeated work
//! removed without changing what it charges, leases or refuses.
use super::*;

/// Change 1: the allowance `Catalog::new` memoizes equals the per-ticket formula it replaced
/// (the two canonical encodings plus 1024), for every retained record of both layout classes;
/// a masked record refuses as `record` does.
#[test]
fn the_memoized_allowance_is_the_recomputed_one() {
    for class in [LayoutClass::Uniform, LayoutClass::PerRecord] {
        let (catalog, ids) = catalog(class);
        let mut retained = 0;
        for id in &ids {
            match catalog.record(id) {
                Ok(record) => {
                    let formula = (id.encode().unwrap().len()
                        + record.layout.encode().unwrap().len()
                        + 1024) as u64;
                    assert_eq!(catalog.metadata_allowance(id).unwrap(), formula);
                    retained += 1;
                }
                Err(err) => {
                    assert_eq!(err, Error::MaskedId);
                    assert_eq!(catalog.metadata_allowance(id), Err(Error::MaskedId));
                }
            }
        }
        assert_eq!(retained, 3);
    }
}

/// Change 6's fixture: one door-shaped record `(2, 0, 9)` in an SLRU dispatch registered with a
/// pending bound of `limit`.
fn resident_owner(limit: usize) -> ExpertBankOwner {
    let g = gov();
    let mut l = layout(9, 4, 16);
    l.segments.truncate(1);
    l.requirements.truncate(1);
    let id = bank_id(9, &l);
    let bank = BankService::new(
        Catalog::new(LayoutClass::Uniform, vec![(id.clone(), Some(record(l)))]).unwrap(),
        g.clone(),
        Heat::default(),
        Reader::default(),
        CoalescingPolicy {
            granularity: 1,
            slot_bytes: 32,
        },
        limits(32),
    )
    .unwrap();
    let mut req = request(bank.slru_metadata_bytes(1).unwrap(), Priority::Demand);
    let metadata = g.borrow_mut().reserve(&req).unwrap();
    let bank = bank
        .with_slru(SlruPolicy::new(&[(16, 1)]).unwrap(), &metadata)
        .unwrap();
    req.bytes = TierBudget::zero(2);
    let bank =
        SlruExpertDispatch::new(bank, BTreeMap::from([((2, 0, 9), id)]), req, epochs()).unwrap();
    ExpertBankOwner::register(Box::new(bank), limit).unwrap()
}

/// Change 6: `demand_if_resident` answers in the order the three calls ran. Not resident: the
/// gate never runs and nothing is leased. Resident and the gate declines: nothing is leased.
/// Resident and the gate agrees: a lease with the gate's value, the same bytes `demand` lends.
/// Refused after the gate: the gate's value comes back and the registry holds no new lease.
#[test]
fn one_owner_call_keeps_the_prefetch_order() {
    let mut owner = resident_owner(1);
    let proxy = owner.proxy();
    let mut gated = 0;
    let outcome = proxy
        .demand_if_resident((2, 0, 9), 16, || {
            gated += 1;
            Some(())
        })
        .unwrap();
    assert!(matches!(outcome, ResidentDemand::NotResident));
    assert_eq!(gated, 0);
    // One demand makes the record host-resident.
    let token = proxy.demand((2, 0, 9), 16).unwrap();
    let lent = proxy.with_bytes(&token, <[u8]>::to_vec).unwrap();
    proxy.finish(&token).unwrap();
    assert!(proxy.host_resident((2, 0, 9)).unwrap());
    let outcome = proxy
        .demand_if_resident((2, 0, 9), 16, || None::<u32>)
        .unwrap();
    assert!(matches!(outcome, ResidentDemand::Declined));
    owner.close().unwrap();

    let mut owner = resident_owner(1);
    let proxy = owner.proxy();
    let token = proxy.demand((2, 0, 9), 16).unwrap();
    proxy.finish(&token).unwrap();
    let ResidentDemand::Leased(token, value) = proxy
        .demand_if_resident((2, 0, 9), 16, || Some(7u32))
        .unwrap()
    else {
        panic!("a resident record with an agreeing gate is leased");
    };
    assert_eq!(value, 7);
    assert_eq!(token.record(), (2, 0, 9));
    assert_eq!(proxy.with_bytes(&token, <[u8]>::to_vec).unwrap(), lent);
    // The pending bound is 1 and this lease is open: the next demand refuses after its gate.
    let mut gated = 0;
    let outcome = proxy
        .demand_if_resident((2, 0, 9), 16, || {
            gated += 1;
            Some(11u32)
        })
        .unwrap();
    assert!(matches!(
        outcome,
        ResidentDemand::Refused(Error::Capacity, 11)
    ));
    assert_eq!(gated, 1);
    assert_eq!(owner.close(), Err(Error::Busy));
    proxy.finish(&token).unwrap();
    owner.close().unwrap();
    assert_eq!(
        proxy.demand_if_resident((2, 0, 9), 16, || Some(())).err(),
        Some(Error::NotFound)
    );
}
