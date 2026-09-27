//! Day 85 (I22, `research/spill-c-20260919/DAY85.md`): the dispatch adapter, which stages by catalog position (the
//! host cache, the catalog entry and the SLRU read by position), against a twin bank driven through the hashed
//! `stage` with the batches the adapter built before I22: the same outcomes, leases, reads, SLRU orders and host cache
//! after every operation of a randomized trace of grouped and single demands and fill admissions, for demand and for
//! prefetch priority. Also `hit_at` against `hit` on the day-43 trace, `validate` by position against the id's layout,
//! and the catalog's `id_at`.
use super::*;

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

fn spread() -> (Vec<(BankId, CatalogRecord)>, Vec<BankId>) {
    let mut entries = Vec::new();
    let mut seed = 0u64;
    for layer in [0u32, 5, u32::from(u16::MAX)] {
        for expert in [1u32, 7, 30] {
            for projection in [Projection::Gate, Projection::Up, Projection::Down] {
                let mut l = layout(seed, 13, 16);
                seed += 1;
                l.segments.truncate(1);
                l.requirements.truncate(1);
                let id = BankId {
                    version: 1,
                    tensor: tensor(),
                    record: RecordId::Expert {
                        layer,
                        original_id: expert,
                        projection,
                    },
                    layout: l.identity().unwrap(),
                };
                entries.push((id, record(l)));
            }
        }
    }
    let ids = entries.iter().map(|(id, _)| id.clone()).collect();
    (entries, ids)
}

fn bank(slots: usize, priority: Priority) -> (Banks, BudgetRequest) {
    let (entries, _) = spread();
    let g = gov();
    let bank = BankService::<ExpertDomain, _, _>::new(
        Catalog::new(
            LayoutClass::Uniform,
            entries.into_iter().map(|(id, r)| (id, Some(r))).collect(),
        )
        .unwrap(),
        g.clone(),
        Heat::default(),
        Reader::default(),
        CoalescingPolicy {
            granularity: 1,
            slot_bytes: 32,
        },
        BankLimits {
            cache_bytes: 16 * slots as u64,
            batch_bytes: 100_000,
            items: MAX_GROUP,
            tickets: 4,
        },
    )
    .unwrap();
    let mut req = request(bank.slru_metadata_bytes(slots).unwrap(), priority);
    let metadata = g.borrow_mut().reserve(&req).unwrap();
    let bank = bank
        .with_slru(SlruPolicy::new(&[(16, slots)]).unwrap(), &metadata)
        .unwrap();
    req.bytes = TierBudget::zero(2);
    (bank, req)
}

/// The pre-I22 adapter's demand, on the hashed path: one batch of the demanded ids, pumped, published, finished.
fn hashed_demand(b: &mut Banks, ids: &[BankId], req: &BudgetRequest) -> Result<Vec<(BankId, u8)>> {
    let ticket = b.stage(BankBatch {
        ids: ids.to_vec(),
        epochs: epochs(),
        request: req.clone(),
    })?;
    while !b.progress(&ticket)? {}
    let leases = b.publish(&ticket, epochs())?;
    let seen = leases.iter().map(|l| (l.id().clone(), first(l))).collect();
    drop(leases);
    b.finish_ticket(&ticket)?;
    b.collect_evicted()?;
    Ok(seen)
}

/// A lease's first payload byte (the fixture reader's and the fill's bytes are heap vectors).
fn first(lease: &BankLease) -> u8 {
    lease.resource::<Vec<u8>>().map(|bytes| bytes[0]).unwrap()
}

fn state(b: &Banks, ids: &[BankId]) -> String {
    let policy = b.slru_policy().unwrap();
    let resident: Vec<_> = ids.iter().map(|id| policy.resident(id)).collect();
    format!(
        "orders={:?} resident={resident:?} cached={} bytes={} owned={} reads={}",
        policy.orders(),
        b.cached_records(),
        b.cache_bytes(),
        b.owned_leases(),
        b.reader().calls.len()
    )
}

fn equal_on_a_trace(priority: Priority, seed: u64) {
    let (_, ids) = spread();
    let (positioned, req) = bank(5, priority);
    let map = ids
        .iter()
        .map(|b| (dispatch_id(&b.record).unwrap(), b.clone()))
        .collect();
    let mut d = SlruExpertDispatch::new(positioned, map, req.clone(), epochs()).unwrap();
    let (mut h, hreq) = bank(5, priority);
    let mut rng = Rng(seed);
    let (mut groups, mut singles, mut fills, mut misses) = (0, 0, 0, 0);
    for step in 0..600 {
        let reads_before = h.reader().calls.len();
        match rng.below(20) {
            0..=11 => {
                // An expert's three blocks, one grouped demand.
                let start = 3 * rng.below(ids.len() as u64 / 3) as usize;
                let group = &ids[start..start + 3];
                let blocks: Vec<_> = group
                    .iter()
                    .map(|id| (dispatch_id(&id.record).unwrap(), 16))
                    .collect();
                let got = d.demand_many(&blocks).map(|demands| {
                    let seen: Vec<_> = demands
                        .leases
                        .iter()
                        .map(|l| (l.id().clone(), first(l)))
                        .collect();
                    d.finish_many(&demands).unwrap();
                    seen
                });
                assert_eq!(
                    got,
                    hashed_demand(&mut h, group, &hreq),
                    "group at step {step}"
                );
                groups += 1;
            }
            12..=16 => {
                let id = &ids[rng.below(ids.len() as u64) as usize];
                let local = dispatch_id(&id.record).unwrap();
                let got = d.demand(local, 16).map(|demand| {
                    let seen = vec![(demand.lease.id().clone(), first(&demand.lease))];
                    d.finish(&demand).unwrap();
                    seen
                });
                assert_eq!(
                    got,
                    hashed_demand(&mut h, std::slice::from_ref(id), &hreq),
                    "single at step {step}"
                );
                singles += 1;
            }
            _ => {
                let n = rng.below(ids.len() as u64) as usize;
                let (entries, _) = spread();
                let s = &entries[n].1.layout.segments[0];
                let bytes: Vec<u8> = (s.offset..s.offset + s.storage_bytes)
                    .map(|p| byte(s.tensor.as_ref().unwrap(), p))
                    .collect();
                let digest = checksum(&bytes);
                let local = dispatch_id(&ids[n].record).unwrap();
                assert_eq!(
                    d.admit_filled(local, HostBytes::Heap(bytes.clone()), digest),
                    h.admit_filled(&ids[n], HostBytes::Heap(bytes), digest, &hreq),
                    "fill at step {step}"
                );
                fills += 1;
            }
        }
        misses += usize::from(h.reader().calls.len() > reads_before);
        assert_eq!(
            state(d.bank(), &ids),
            state(&h, &ids),
            "state at step {step}"
        );
    }
    println!(
        "day85 {priority:?}: groups={groups} singles={singles} fills={fills} steps_with_reads={misses}"
    );
    assert!(groups > 200 && singles > 50 && fills > 50 && misses > 100);
}

#[test]
fn the_positioned_adapter_equals_the_hashed_bank_for_demand_tickets() {
    equal_on_a_trace(Priority::Demand, 0x243f_6a88_85a3_08d3);
}

#[test]
fn the_positioned_adapter_equals_the_hashed_bank_for_prefetch_tickets() {
    equal_on_a_trace(Priority::OptionalPrefetch, 0x1319_8a2e_0370_7344);
}

#[test]
fn hit_at_equals_hit_on_the_randomized_trace() {
    let classes = [(32u64, 5usize), (64, 3), (128, 2)];
    let mut a = SlruPolicy::new(&classes).unwrap();
    a.index_positions(48).unwrap();
    let mut b = SlruPolicy::new(&classes).unwrap();
    b.index_positions(48).unwrap();
    let pool: Vec<BankId> = (0..48)
        .map(|n| bank_id(n, &layout(n.into(), 13, 16)))
        .collect();
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    let mut reserved: Vec<usize> = Vec::new();
    let mut hits = 0u64;
    for step in 0..100_000u64 {
        match rng.below(10) {
            0..=3 => {
                let t = rng.below(48) as usize;
                let size = [16u64, 32, 64, 100][rng.below(4) as usize];
                let x = a.reserve(&pool[t], size, &[]);
                assert_eq!(x, b.reserve(&pool[t], size, &[]), "reserve at {step}");
                if let Ok(Some(_)) = x {
                    reserved.push(t);
                }
            }
            4..=5 if !reserved.is_empty() => {
                let t = reserved.swap_remove(rng.below(reserved.len() as u64) as usize);
                assert_eq!(a.publish_at(&pool[t], t), b.publish_at(&pool[t], t));
            }
            _ => {
                let span = if rng.below(2) == 0 { 12 } else { 48 };
                let t = rng.below(span) as usize;
                let (x, y) = (a.hit_at(t), b.hit(&pool[t]));
                assert_eq!(x, y, "hit at {step}");
                hits += u64::from(x);
            }
        }
        if step % 500 == 0 {
            assert_eq!(a.orders(), b.orders(), "orders at {step}");
        }
    }
    assert_eq!(a.orders(), b.orders());
    assert!(!a.hit_at(48));
    println!("day85 hit trace: hits={hits}");
    assert!(hits > 1_000, "hits={hits}");
}

#[test]
fn validate_by_position_answers_as_the_layout_did() {
    let (_, ids) = spread();
    let (b, req) = bank(4, Priority::Demand);
    let map = ids
        .iter()
        .map(|b| (dispatch_id(&b.record).unwrap(), b.clone()))
        .collect();
    let d = SlruExpertDispatch::new(b, map, req, epochs()).unwrap();
    let (entries, _) = spread();
    let catalog = Catalog::new(
        LayoutClass::Uniform,
        entries.into_iter().map(|(id, r)| (id, Some(r))).collect(),
    )
    .unwrap();
    let mut sorted = ids.clone();
    sorted.sort();
    for (position, id) in sorted.iter().enumerate() {
        assert_eq!(catalog.id_at(position), Ok(id));
        assert_eq!(d.bank().catalog_position(id), Ok(position));
        let local = dispatch_id(&id.record).unwrap();
        let want = d.bank().layout(id).map(|l| l.segments[0].valid_bytes);
        assert_eq!(want, Ok(16));
        assert_eq!(d.validate(local, 16), Ok(()));
        assert_eq!(d.validate(local, 15), Err(Error::InvalidLayout));
    }
    assert_eq!(catalog.id_at(ids.len()), Err(Error::NotFound));
    assert_eq!(d.validate((1, 0, 1), 16), Err(Error::NotFound));
}
