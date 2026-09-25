//! Day 64 (`research/spill-c-20260919/DAY64.md`): I14 hashes the record lookups and keeps every answer, refusal and
//! order the ordered maps gave.
use super::*;

fn fixture_entries(class: LayoutClass) -> Vec<(BankId, Option<CatalogRecord>)> {
    let mut entries = Vec::new();
    for (n, q, len) in [
        (9u32, 2, 16u64),
        (47, 3, 16),
        (83, 4, 32),
        (5, 2, 16),
        (61, 3, 16),
    ] {
        let (q, len) = if class == LayoutClass::Uniform {
            (2, 16)
        } else {
            (q, len)
        };
        let l = layout(u64::from(n), q, len);
        entries.push((bank_id(n, &l), Some(record(l))));
    }
    let masked = BankId {
        record: RecordId::Expert {
            layer: 2,
            original_id: 21,
            projection: Projection::Gate,
        },
        ..entries[0].0.clone()
    };
    entries.push((masked, None));
    entries
}

/// I14 change 1: every id's `record` and `metadata_allowance`, and every refusal (an unknown id, a masked one, an id
/// that fails validation), equal what a `BTreeMap` of the same entries answers.
#[test]
fn the_hashed_catalog_answers_as_the_ordered_one() {
    for class in [LayoutClass::Uniform, LayoutClass::PerRecord] {
        let entries = fixture_entries(class);
        let reference: BTreeMap<BankId, Option<CatalogRecord>> = entries.iter().cloned().collect();
        let catalog = Catalog::new(class, entries.clone()).unwrap();
        for (id, want) in &reference {
            match want {
                Some(r) => {
                    let got = catalog.record(id).unwrap();
                    assert_eq!(got.layout, r.layout);
                    assert_eq!(got.checksums, r.checksums);
                    let formula = (id.encode().unwrap().len()
                        + r.layout.encode().unwrap().len()
                        + 1024) as u64;
                    assert_eq!(catalog.metadata_allowance(id).unwrap(), formula);
                }
                None => assert_eq!(catalog.record(id).err(), Some(Error::MaskedId)),
            }
        }
        let unknown = BankId {
            record: RecordId::Expert {
                layer: 3,
                original_id: 9,
                projection: Projection::Gate,
            },
            ..entries[0].0.clone()
        };
        assert_eq!(catalog.record(&unknown).err(), Some(Error::NotFound));
        let invalid = BankId {
            version: 99,
            ..entries[0].0.clone()
        };
        assert_eq!(catalog.record(&invalid).err(), invalid.validate().err());
        // Construction refuses a duplicate id as before.
        let mut twice = entries.clone();
        twice.push(entries[1].clone());
        assert_eq!(Catalog::new(class, twice).err(), Some(Error::Conflict));
    }
}

/// I14 change 2: without an SLRU the host cache trims to its byte limit by the lowest hotness score, ties to the
/// lowest id; over a randomized demand sequence with many ties, the cached set after every ticket equals a reference
/// that keeps the records in a `BTreeMap` and evicts its first minimum, as the ordered map did.
#[test]
fn trim_evicts_the_ordered_maps_victim() {
    let g = gov();
    let entries = fixture_entries(LayoutClass::Uniform);
    let ids: Vec<BankId> = entries
        .iter()
        .filter(|(_, r)| r.is_some())
        .map(|(id, _)| id.clone())
        .collect();
    let per_record = entries[0]
        .1
        .as_ref()
        .unwrap()
        .layout
        .storage_bytes()
        .unwrap();
    let limit = 2 * per_record;
    let mut bank = BankService::<ExpertDomain, _, _>::new(
        Catalog::new(LayoutClass::Uniform, entries).unwrap(),
        g.clone(),
        Heat::default(),
        Reader::default(),
        CoalescingPolicy::default(),
        limits(limit),
    )
    .unwrap();
    let mut heat: BTreeMap<BankId, u64> = BTreeMap::new();
    let mut cached: BTreeMap<BankId, ()> = BTreeMap::new();
    let mut s: u64 = 0x9E37_79B9_7F4A_7C15;
    for _ in 0..400 {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        let id = ids[(s % ids.len() as u64) as usize].clone();
        let t = bank.stage(batch(vec![id.clone()])).unwrap();
        let _ = publish_bank(&mut bank, &t, epochs()).unwrap();
        finish(&mut bank, &t);
        bank.collect_evicted().unwrap();
        *heat.entry(id.clone()).or_default() += 1;
        cached.insert(id, ());
        while cached.len() as u64 * per_record > limit {
            let victim = cached
                .keys()
                .min_by_key(|k| heat.get(*k).copied().unwrap_or(0))
                .cloned()
                .unwrap();
            cached.remove(&victim);
        }
        for id in &ids {
            assert_eq!(
                bank.resident(id).unwrap().is_some(),
                cached.contains_key(id),
                "{id:?}"
            );
        }
    }
}

/// A three-record SLRU dispatch of the fixture's retained records, and its `(ExpertDispatchId, bytes)` blocks.
fn group_bank() -> (
    SlruExpertDispatch<Heat, Reader>,
    Vec<(ExpertDispatchId, usize)>,
) {
    let g = gov();
    let mut entries = Vec::new();
    let mut map = BTreeMap::new();
    let mut blocks = Vec::new();
    for n in [9u32, 47, 83] {
        let mut l = layout(u64::from(n), 4, 16);
        l.segments.truncate(1);
        l.requirements.truncate(1);
        let id = bank_id(n, &l);
        let local = dispatch_id(&id.record).unwrap();
        map.insert(local, id.clone());
        blocks.push((local, 16));
        entries.push((id, Some(record(l))));
    }
    let bank = BankService::new(
        Catalog::new(LayoutClass::Uniform, entries).unwrap(),
        g.clone(),
        Heat::default(),
        Reader::default(),
        CoalescingPolicy {
            granularity: 1,
            slot_bytes: 32,
        },
        BankLimits {
            cache_bytes: 64,
            batch_bytes: 64,
            items: MAX_GROUP,
            tickets: 4,
        },
    )
    .unwrap();
    let mut req = request(bank.slru_metadata_bytes(3).unwrap(), Priority::Demand);
    let metadata = g.borrow_mut().reserve(&req).unwrap();
    let bank = bank
        .with_slru(SlruPolicy::new(&[(16, 3)]).unwrap(), &metadata)
        .unwrap();
    req.bytes = TierBudget::zero(2);
    (
        SlruExpertDispatch::new(bank, map, req, epochs()).unwrap(),
        blocks,
    )
}

/// A bank that answers a grouped demand with its leases in reverse order: the case the group identity refuses.
struct Reversed(SlruExpertDispatch<Heat, Reader>);
impl ExpertDispatchBank for Reversed {
    fn validate(&self, id: ExpertDispatchId, bytes: usize) -> Result<()> {
        self.0.validate(id, bytes)
    }
    fn demand(&mut self, id: ExpertDispatchId, bytes: usize) -> Result<ExpertDemand> {
        self.0.demand(id, bytes)
    }
    fn finish(&mut self, demand: ExpertDemand) -> Result<()> {
        self.0.finish(demand)
    }
    fn demand_many(&mut self, blocks: &[(ExpertDispatchId, usize)]) -> Result<ExpertDemands> {
        let mut d = self.0.demand_many(blocks)?;
        d.leases.reverse();
        Ok(d)
    }
    fn finish_many(&mut self, demands: ExpertDemands) -> Result<()> {
        self.0.finish_many(demands)
    }
}

/// I15: a grouped demand's token names every record in order; `with_bytes_at` lends each record's own bytes (the
/// same bytes one `demand` per record lends); `finish_group` retires the ticket once; a single token and a group
/// token do not stand for each other; the pending bound counts the group as one ticket; a bank that publishes the
/// records in another order is refused and its group finished through it.
#[test]
fn a_grouped_demand_leases_its_records_in_order() {
    let (bank, blocks) = group_bank();
    let mut owner = ExpertBankOwner::register(Box::new(bank), 1).unwrap();
    let proxy = owner.proxy();
    let mut singles = Vec::new();
    for &(local, bytes) in &blocks {
        let token = proxy.demand(local, bytes).unwrap();
        singles.push(proxy.with_bytes(&token, <[u8]>::to_vec).unwrap());
        proxy.finish(&token).unwrap();
    }
    let group = proxy.demand_many(&blocks).unwrap();
    assert_eq!(
        group.records(),
        blocks.iter().map(|b| b.0).collect::<Vec<_>>().as_slice()
    );
    for (index, want) in singles.iter().enumerate() {
        assert_eq!(
            &proxy.with_bytes_at(&group, index, <[u8]>::to_vec).unwrap(),
            want
        );
    }
    assert_eq!(
        proxy.with_bytes_at(&group, 3, |_| ()).err(),
        Some(Error::NotFound)
    );
    // The pending bound is one ticket: the open group holds it.
    assert_eq!(proxy.demand(blocks[0].0, 16).err(), Some(Error::Capacity));
    assert_eq!(proxy.demand_many(&blocks[..1]).err(), Some(Error::Capacity));
    assert_eq!(owner.close(), Err(Error::Busy));
    proxy.finish_group(&group).unwrap();
    assert_eq!(proxy.finish_group(&group).err(), Some(Error::UnknownTicket));
    // A single token is not a group's, and a group's is not a single one's.
    let single = proxy.demand(blocks[1].0, 16).unwrap();
    assert_eq!(
        proxy.with_bytes_at(&group, 0, |_| ()).err(),
        Some(Error::UnknownTicket)
    );
    proxy.finish(&single).unwrap();
    let group = proxy.demand_many(&blocks[1..]).unwrap();
    assert_eq!(group.records().len(), 2);
    proxy.finish_group(&group).unwrap();
    assert_eq!(proxy.demand_many(&[]).err(), Some(Error::EmptyBatch));
    let four = [blocks[0], blocks[1], blocks[2], blocks[0]];
    assert_eq!(proxy.demand_many(&four).err(), Some(Error::Capacity));
    owner.close().unwrap();

    let (bank, blocks) = group_bank();
    let mut owner = ExpertBankOwner::register(Box::new(Reversed(bank)), 1).unwrap();
    let proxy = owner.proxy();
    assert_eq!(
        proxy.demand_many(&blocks).err(),
        Some(Error::ProgramMismatch)
    );
    // The refused group was finished through the bank: nothing is pending.
    owner.close().unwrap();
}
