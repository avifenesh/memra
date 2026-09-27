//! Day 84 (I21, `research/spill-c-20260919/DAY84.md`): the SLRU policy's position view answers what its `table`
//! answers after every operation (the day-43 randomized trace and the day-4 synthetic fixture), a plain publication on
//! an indexed policy refuses, the catalog's positions are its `BankId` order, the SLRU metadata charge counts the view,
//! and the dispatch adapter's residency reads by position equal the map-based ones through fills, demands and
//! evictions.
use super::*;

/// xorshift64*, the day-43 stream.
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

fn pool_id(n: u32) -> BankId {
    bank_id(n, &layout(n.into(), 13, 16))
}

fn assert_view(p: &SlruPolicy, plain: &SlruPolicy, pool: &[BankId], step: u64) {
    for (position, id) in pool.iter().enumerate() {
        assert_eq!(
            p.resident_at(position),
            p.resident(id),
            "view differs at step {step}, position {position}"
        );
        assert_eq!(
            p.resident(id),
            plain.resident(id),
            "table differs at step {step}, position {position}"
        );
    }
}

#[test]
fn the_position_view_equals_the_table_on_the_randomized_trace() {
    let classes = [(32u64, 5usize), (64, 3), (128, 2)];
    let mut indexed = SlruPolicy::new(&classes).unwrap();
    indexed.index_positions(48).unwrap();
    let mut plain = SlruPolicy::new(&classes).unwrap();
    let pool: Vec<BankId> = (0..48).map(pool_id).collect();
    let sizes = [16u64, 32, 48, 64, 100, 128];
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    let mut reserved: Vec<usize> = Vec::new();
    let (mut publishes, mut evictions, mut removes) = (0u64, 0u64, 0u64);
    for step in 0..250_000u64 {
        match rng.below(10) {
            0..=3 => {
                let target = rng.below(pool.len() as u64) as usize;
                let required = sizes[rng.below(sizes.len() as u64) as usize];
                let keep: Vec<BankId> = (0..rng.below(3))
                    .map(|_| pool[rng.below(pool.len() as u64) as usize].clone())
                    .collect();
                let a = indexed.reserve(&pool[target], required, &keep);
                let b = plain.reserve(&pool[target], required, &keep);
                assert_eq!(a, b, "reserve differs at step {step}");
                if let Ok(Some(d)) = a {
                    evictions += u64::from(d.evicted.is_some());
                    reserved.push(target);
                }
            }
            4..=5 if !reserved.is_empty() => {
                let target = reserved.swap_remove(rng.below(reserved.len() as u64) as usize);
                if rng.below(5) == 0 {
                    assert_eq!(
                        indexed.abort_retired(&pool[target]),
                        plain.abort_retired(&pool[target])
                    );
                } else {
                    assert_eq!(
                        indexed.publish_at(&pool[target], target),
                        plain.publish(&pool[target]),
                        "publish at {step}"
                    );
                    publishes += 1;
                }
            }
            6..=8 => {
                let span = if rng.below(2) == 0 {
                    12
                } else {
                    pool.len() as u64
                };
                let target = &pool[rng.below(span) as usize];
                assert_eq!(
                    indexed.hit(target),
                    plain.hit(target),
                    "hit differs at step {step}"
                );
            }
            _ => {
                let target = &pool[rng.below(pool.len() as u64) as usize];
                let (x, y) = (indexed.remove(target), plain.remove(target));
                assert_eq!(x, y, "remove differs at step {step}");
                removes += u64::from(x);
            }
        }
        assert_view(&indexed, &plain, &pool, step);
        if step % 1_000 == 0 {
            assert_eq!(
                indexed.orders(),
                plain.orders(),
                "orders differ at step {step}"
            );
        }
    }
    assert_eq!(indexed.orders(), plain.orders());
    assert_eq!(indexed.resident_at(48), None);
    println!("day84 trace: publishes={publishes} evictions={evictions} removes={removes}");
    assert!(publishes > 10_000 && evictions > 10_000 && removes > 1_000);
}

#[test]
fn the_position_view_equals_the_table_on_the_day4_fixture() {
    let trace: serde_json::Value = serde_json::from_str(include_str!(
        "../../../../research/spill-c-20260919/fixtures/slru-synthetic.json"
    ))
    .unwrap();
    let ids: Vec<_> = (0..24)
        .map(|n| bank_id(n, &layout(n.into(), 13, 16)))
        .collect();
    let classes = [(32, 3), (64, 2), (128, 1)];
    let mut p = SlruPolicy::new(&classes).unwrap();
    p.index_positions(ids.len()).unwrap();
    let mut plain = SlruPolicy::new(&classes).unwrap();
    let rows = trace["rows"].as_array().unwrap();
    assert_eq!(rows.len(), 2_013);
    for (index, row) in rows.iter().enumerate() {
        let key = row["key"].as_u64().unwrap() as usize;
        let id = &ids[key];
        let op = row["op"].as_str().unwrap();
        let keep: Vec<_> = row["keep"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| ids[v.as_u64().unwrap() as usize].clone())
            .collect();
        // The day-4 replay's decisions, on both policies.
        if op == "abort" {
            if p.pending(id) {
                assert_eq!(p.abort_retired(id), plain.abort_retired(id));
            }
        } else if op == "complete" {
            if p.pending(id) {
                assert_eq!(p.publish_at(id, key), plain.publish(id));
            }
        } else if op == "demand" && p.resident(id).is_some() {
            assert_eq!(p.hit(id), plain.hit(id));
        } else if op == "demand" && p.pending(id) {
            assert_eq!(p.publish_at(id, key), plain.publish(id));
        } else if p.resident(id).is_none() && !p.pending(id) {
            let size = row["size"].as_u64().unwrap();
            let a = p.reserve(id, size, &keep).unwrap();
            assert_eq!(a, plain.reserve(id, size, &keep).unwrap(), "row {index}");
            if a.is_some() && op == "demand" {
                assert_eq!(p.publish_at(id, key), plain.publish(id));
            }
        }
        assert_view(&p, &plain, &ids, index as u64);
        assert_eq!(p.orders(), plain.orders(), "row {index}");
        assert_eq!(
            serde_json::to_value(p.orders()).unwrap(),
            row["orders"],
            "row {index}"
        );
    }
}

#[test]
fn an_indexed_policy_takes_positioned_publications_only() {
    let id = pool_id(3);
    let mut p = SlruPolicy::new(&[(32, 2)]).unwrap();
    assert!(!p.indexed());
    p.reserve(&id, 16, &[]).unwrap().unwrap();
    // Not indexed: `publish_at` refuses, and the view cannot be installed on a policy holding a reservation.
    assert_eq!(p.clone().publish_at(&id, 0), Err(Error::Unsupported));
    assert_eq!(p.clone().index_positions(4), Err(Error::Busy));
    let mut p = SlruPolicy::new(&[(32, 2)]).unwrap();
    p.index_positions(4).unwrap();
    assert!(p.indexed());
    assert_eq!(p.index_positions(4), Err(Error::Busy));
    p.reserve(&id, 16, &[]).unwrap().unwrap();
    // Indexed: a plain publication refuses and a position outside the view refuses, both before any change.
    assert_eq!(p.publish(&id), Err(Error::Unsupported));
    assert_eq!(p.publish_at(&id, 4), Err(Error::InvalidLayout));
    assert!(p.pending(&id));
    let slot = p.publish_at(&id, 2).unwrap();
    assert_eq!(p.resident_at(2), Some(slot));
    assert_eq!(p.resident(&id), Some(slot));
    assert!(p.remove(&id));
    assert_eq!(p.resident_at(2), None);
}

fn spread_id(
    position_seed: u32,
    layer: u32,
    projection: Projection,
    expert: u32,
) -> (BankId, CatalogRecord) {
    let mut l = layout(u64::from(position_seed), 13, 16);
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
    (id, record(l))
}

/// Records over two model layers and the MTP key (`layer = u16::MAX`), three projections and sparse expert ids, behind
/// a dispatch adapter with `slots` host slots.
fn spread_bank(
    slots: usize,
) -> (
    SlruExpertDispatch<Heat, Reader>,
    Vec<BankId>,
    Vec<CatalogRecord>,
) {
    let g = gov();
    let (mut entries, mut ids, mut recs) = (Vec::new(), Vec::new(), Vec::new());
    let mut seed = 0;
    for layer in [0u32, 5, u32::from(u16::MAX)] {
        for projection in [Projection::Gate, Projection::Up, Projection::Down] {
            for expert in [1u32, 7, 30] {
                let (id, r) = spread_id(seed, layer, projection.clone(), expert);
                seed += 1;
                ids.push(id.clone());
                recs.push(r.clone());
                entries.push((id, Some(r)));
            }
        }
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
        limits(16 * slots as u64),
    )
    .unwrap();
    let mut req = request(bank.slru_metadata_bytes(slots).unwrap(), Priority::Demand);
    let metadata = g.borrow_mut().reserve(&req).unwrap();
    let bank = bank
        .with_slru(SlruPolicy::new(&[(16, slots)]).unwrap(), &metadata)
        .unwrap();
    req.bytes = TierBudget::zero(2);
    let map = ids
        .iter()
        .map(|b| (dispatch_id(&b.record).unwrap(), b.clone()))
        .collect();
    (
        SlruExpertDispatch::new(bank, map, req, epochs()).unwrap(),
        ids,
        recs,
    )
}

fn assert_reads(d: &SlruExpertDispatch<Heat, Reader>, ids: &[BankId], step: usize) {
    let policy = d.bank().slru_policy().unwrap();
    for id in ids {
        let local = dispatch_id(&id.record).unwrap();
        let want = policy.resident(id);
        assert_eq!(
            d.resident_slot(local),
            Ok(want),
            "slot at step {step} for {local:?}"
        );
        assert_eq!(
            d.host_resident(local),
            Ok(want.is_some()),
            "residency at step {step} for {local:?}"
        );
    }
}

#[test]
fn the_dispatch_adapter_reads_residency_by_position() {
    let (mut d, ids, recs) = spread_bank(4);
    assert_reads(&d, &ids, 0);
    // Local ids the catalog does not hold: another layer, a projection or an expert outside the rows.
    for local in [
        (1u16, 0u8, 1u16),
        (0, 3, 1),
        (0, 0, 2),
        (0, 0, 31),
        (u16::MAX - 1, 0, 1),
    ] {
        assert_eq!(d.resident_slot(local), Err(Error::NotFound));
        assert_eq!(d.host_resident(local), Err(Error::NotFound));
    }
    // Two fills into free slots, then demands that hit, miss and evict.
    for n in [4usize, 22] {
        let local = dispatch_id(&ids[n].record).unwrap();
        let s = &recs[n].layout.segments[0];
        let bytes: Vec<u8> = (s.offset..s.offset + s.storage_bytes)
            .map(|p| byte(s.tensor.as_ref().unwrap(), p))
            .collect();
        let digest = checksum(&bytes);
        assert_eq!(
            d.admit_filled(local, HostBytes::Heap(bytes), digest),
            Ok(FillOutcome::Admitted)
        );
        assert_reads(&d, &ids, n);
    }
    let mut rng = Rng(0x51ed_270b_27a1_0f3d);
    let mut misses = 0;
    for step in 0..400 {
        let n = rng.below(ids.len() as u64) as usize;
        let local = dispatch_id(&ids[n].record).unwrap();
        let before = d.bank().slru_policy().unwrap().resident(&ids[n]);
        let demand = d.demand(local, 16).unwrap();
        d.finish(&demand).unwrap();
        misses += usize::from(before.is_none());
        assert_reads(&d, &ids, step + 100);
    }
    assert!(misses > 50, "the trace must miss and evict: {misses}");
}

#[test]
fn catalog_positions_are_the_bankid_order_and_the_charge_counts_the_view() {
    let (d, ids, _) = spread_bank(4);
    let mut sorted = ids.clone();
    sorted.sort();
    for (position, id) in sorted.iter().enumerate() {
        assert_eq!(d.bank().catalog_position(id), Ok(position));
    }
    let (unknown, _) = spread_id(99, 9, Projection::Gate, 1);
    assert_eq!(d.bank().catalog_position(&unknown), Err(Error::NotFound));
    // With no slot the formula's slot terms vanish: the constant plus one `u32` per catalog id (and, since day 85,
    // one 8-byte lease handle per catalog id for the host cache's view).
    assert_eq!(
        d.bank().slru_metadata_bytes(0),
        Ok(4096 + (4 + 8) * ids.len() as u64)
    );
    let per_slot =
        d.bank().slru_metadata_bytes(2).unwrap() - d.bank().slru_metadata_bytes(1).unwrap();
    let max_id = ids
        .iter()
        .map(|id| id.encode().unwrap().len() as u64)
        .max()
        .unwrap();
    assert_eq!(per_slot, max_id * 3 + 1024 + 4);
}
