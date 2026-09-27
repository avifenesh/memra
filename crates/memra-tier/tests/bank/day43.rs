//! Day 43 (`research/spill-c-20260919/DAY43.md`): the O(1) host SLRU makes the VecDeque oracle's
//! decisions, and the bank's eviction bookkeeping stays exact at a large tier.
use super::*;
use crate::slru_oracle::SlruOracle;

/// xorshift64*: a fixed, dependency-free stream so the trace is the same on every run.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

fn id(n: u32) -> BankId {
    bank_id(n, &layout(n.into(), 13, 16))
}

#[test]
fn o1_host_slru_matches_the_vecdeque_oracle_on_a_randomized_trace() {
    let classes = [(32u64, 5usize), (64, 3), (128, 2)];
    let mut new = SlruPolicy::new(&classes).unwrap();
    let mut old = SlruOracle::new(&classes).unwrap();
    let pool: Vec<BankId> = (0..48).map(id).collect();
    let sizes = [16u64, 32, 48, 64, 100, 128];
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    let mut reserved: Vec<BankId> = Vec::new();
    let (mut reserves, mut evictions, mut hits, mut removes) = (0u64, 0u64, 0u64, 0u64);
    for step in 0..250_000u64 {
        match rng.below(10) {
            0..=3 => {
                let target = &pool[rng.below(pool.len() as u64) as usize];
                let required = sizes[rng.below(sizes.len() as u64) as usize];
                let keep: Vec<BankId> = (0..rng.below(3))
                    .map(|_| pool[rng.below(pool.len() as u64) as usize].clone())
                    .collect();
                let a = new.reserve(target, required, &keep);
                let b = old.reserve(target, required, &keep);
                match (&a, &b) {
                    (Ok(x), Ok(y)) => {
                        assert_eq!(
                            x.as_ref().map(|d| (d.slot, d.evicted.clone())),
                            y.as_ref().map(|d| (d.slot, d.evicted.clone())),
                            "reserve differs at step {step}"
                        );
                        if let Some(d) = x {
                            reserves += 1;
                            evictions += u64::from(d.evicted.is_some());
                            reserved.push(target.clone());
                        }
                    }
                    (Err(x), Err(y)) => assert_eq!(x, y, "reserve error differs at step {step}"),
                    _ => panic!("reserve outcome differs at step {step}: {a:?} vs {b:?}"),
                }
            }
            4..=5 if !reserved.is_empty() => {
                let target = reserved.swap_remove(rng.below(reserved.len() as u64) as usize);
                if rng.below(5) == 0 {
                    assert_eq!(new.abort_retired(&target), old.abort_retired(&target));
                } else {
                    assert_eq!(
                        new.publish(&target),
                        old.publish(&target),
                        "publish at {step}"
                    );
                }
            }
            6..=8 => {
                // Half the hits go to a 12-id hot set, so resident ids are hit often.
                let span = if rng.below(2) == 0 {
                    12
                } else {
                    pool.len() as u64
                };
                let target = &pool[rng.below(span) as usize];
                let (x, y) = (new.hit(target), old.hit(target));
                assert_eq!(x, y, "hit differs at step {step}");
                hits += u64::from(x);
            }
            _ => {
                let target = &pool[rng.below(pool.len() as u64) as usize];
                let (x, y) = (new.remove(target), old.remove(target));
                assert_eq!(x, y, "remove differs at step {step}");
                removes += u64::from(x);
            }
        }
        for target in pool.iter().take(4) {
            assert_eq!(new.resident(target), old.resident(target));
            assert_eq!(new.pending(target), old.pending(target));
        }
        if step % 1_000 == 0 {
            assert_eq!(new.orders(), old.orders(), "orders differ at step {step}");
        }
    }
    assert_eq!(new.orders(), old.orders());
    assert_eq!(new.is_empty(), old.is_empty());
    assert_eq!(new.capacity_bytes(), old.capacity_bytes());
    assert_eq!(new.slots(), old.slots());
    // The trace exercised every arm, not only the free-slot path (counts printed on failure).
    println!(
        "day43 trace: reserves={reserves} evictions={evictions} hits={hits} removes={removes}"
    );
    assert!(
        reserves > 20_000 && evictions > 10_000 && hits > 2_000 && removes > 1_000,
        "reserves={reserves} evictions={evictions} hits={hits} removes={removes}"
    );
}

/// A bank of 4,096 records behind a 256-slot SLRU, 20,000 demands with evictions: after every
/// `finish` each owned lease is either cached or held by an open ticket (evicted leases are
/// released, none leaks), and `cache_bytes` equals the sum it replaced.
#[test]
fn eviction_bookkeeping_is_exact_at_a_large_tier() {
    let g = Rc::new(RefCell::new(support::Governor::new(64_000_000)));
    let records = 4_096u32;
    let mut entries = Vec::new();
    let mut ids = Vec::new();
    for n in 0..records {
        let mut l = layout(u64::from(n), 13, 16);
        l.segments.truncate(1);
        l.requirements.truncate(1);
        let bid = bank_id(n, &l);
        ids.push(bid.clone());
        entries.push((bid, Some(record(l))));
    }
    let bank = BankService::<ExpertDomain, _, _>::new(
        Catalog::new(LayoutClass::Uniform, entries).unwrap(),
        g.clone(),
        Heat::default(),
        Reader::default(),
        CoalescingPolicy {
            granularity: 1,
            slot_bytes: 32,
        },
        BankLimits {
            cache_bytes: 256 * 16,
            batch_bytes: 1_000,
            items: 1,
            tickets: 1,
        },
    )
    .unwrap();
    let mut req = request(bank.slru_metadata_bytes(256).unwrap(), Priority::Demand);
    let metadata = g.borrow_mut().reserve(&req).unwrap();
    let bank = bank
        .with_slru(SlruPolicy::new(&[(16, 256)]).unwrap(), &metadata)
        .unwrap();
    req.bytes = TierBudget::zero(2);
    let map: BTreeMap<ExpertDispatchId, BankId> = ids
        .iter()
        .map(|b| (dispatch_id(&b.record).unwrap(), b.clone()))
        .collect();
    let mut dispatch = SlruExpertDispatch::new(bank, map.clone(), req, epochs()).unwrap();
    let mut rng = Rng(0x0123_4567_89ab_cdef);
    let keys: Vec<ExpertDispatchId> = map.keys().copied().collect();
    for _ in 0..20_000 {
        // A hot set of 200 plus a long tail, so both hits and evictions occur.
        let key = if rng.below(2) == 0 {
            keys[rng.below(200) as usize]
        } else {
            keys[rng.below(u64::from(records)) as usize]
        };
        let demand = dispatch.demand(key, 16).unwrap();
        dispatch.finish(demand).unwrap();
        let bank = dispatch.bank();
        assert!(bank.tickets().is_empty(), "a finished demand left a ticket");
        assert_eq!(
            bank.owned_leases(),
            bank.cached_records(),
            "an evicted lease was not released"
        );
        assert_eq!(bank.cache_bytes(), 16 * bank.cached_records() as u64);
        assert!(bank.cached_records() <= 256);
    }
}
