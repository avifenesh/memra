use super::*;

#[test]
fn slru_matches_recorded_native_semantics_synthetic_trace() {
    let trace: serde_json::Value = serde_json::from_str(include_str!(
        "../../../../research/spill-c-20260919/fixtures/slru-synthetic.json"
    ))
    .unwrap();
    use sha2::{Digest as _, Sha256};
    let actual = Sha256::digest(include_bytes!("../../../memra-engine/src/moe_cache.rs"));
    let hex: String = actual.iter().map(|b| format!("{b:02x}")).collect();
    assert_eq!(trace["source_sha256"].as_str().unwrap(), hex);
    let ids: Vec<_> = (0..24)
        .map(|n| bank_id(n, &layout(n.into(), 13, 16)))
        .collect();
    let mut p = SlruPolicy::new(&[(32, 3), (64, 2), (128, 1)]).unwrap();
    let mut outcomes = BTreeMap::new();
    let mut transfers = BTreeMap::new();
    let mut transferred = 0;
    assert_eq!(trace["rows"].as_array().unwrap().len(), 2_013);
    for (index, row) in trace["rows"].as_array().unwrap().iter().enumerate() {
        let id = &ids[row["key"].as_u64().unwrap() as usize];
        let op = row["op"].as_str().unwrap();
        let keep: Vec<_> = row["keep"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| ids[v.as_u64().unwrap() as usize].clone())
            .collect();
        let mut slot = None;
        let mut victim = None;
        let result = if op == "abort" {
            if p.pending(id) {
                // Recorded slot is observed via a clone; the real pending state
                // is returned only by explicit retired-abort, never on cancel.
                slot = Some(p.clone().publish(id).unwrap());
                p.abort_retired(id).unwrap();
                "aborted"
            } else {
                "noop"
            }
        } else if op == "complete" {
            if p.pending(id) {
                slot = Some(p.publish(id).unwrap());
                "published"
            } else {
                "noop"
            }
        } else if op == "demand" && p.resident(id).is_some() {
            slot = p.resident(id);
            assert!(p.hit(id));
            "hit"
        } else if op == "demand" && p.pending(id) {
            slot = Some(p.publish(id).unwrap());
            "published"
        } else if p.resident(id).is_none() && !p.pending(id) {
            match p.reserve(id, row["size"].as_u64().unwrap(), &keep).unwrap() {
                None => "capacity",
                Some(d) => {
                    slot = Some(d.slot);
                    victim = d
                        .evicted
                        .as_ref()
                        .map(|id| ids.iter().position(|i| i == id).unwrap());
                    if op == "demand" {
                        p.publish(id).unwrap();
                        "admitted"
                    } else {
                        "reserved"
                    }
                }
            }
        } else {
            "noop"
        };
        let key = row["key"].as_u64().unwrap() as u8;
        match result {
            "admitted" | "reserved" => {
                let upload =
                    device_rows::FrozenUpload::new(key, row["size"].as_u64().unwrap() as usize);
                transferred += 1;
                if result == "admitted" {
                    upload.finish(true);
                } else {
                    assert!(transfers.insert(key, upload).is_none());
                }
            }
            "published" => transfers.remove(&key).unwrap().finish(true),
            "aborted" => transfers.remove(&key).unwrap().finish(false),
            _ => {}
        }
        *outcomes.entry(result).or_insert(0) += 1;
        assert_eq!(result, row["result"].as_str().unwrap(), "step {index}");
        assert_eq!(
            slot,
            row["slot"].as_u64().map(|n| n as usize),
            "step {index}"
        );
        assert_eq!(
            victim,
            row["victim"].as_u64().map(|n| n as usize),
            "step {index}"
        );
        assert_eq!(
            serde_json::to_value(p.orders()).unwrap(),
            row["orders"],
            "step {index}"
        );
    }
    for arm in [
        "aborted",
        "published",
        "hit",
        "admitted",
        "reserved",
        "capacity",
        "noop",
    ] {
        assert!(outcomes[arm] > 0, "missing {arm}");
    }
    for (_, upload) in transfers {
        upload.finish(false);
    }
    assert!(transferred > 0);
    println!(
        "banked_residency frozen SLRU: 2013 decisions, {transferred} fake transfers; {outcomes:?}"
    );
}

#[test]
fn slru_bank_residency_charges_retained_aliases_and_metadata() {
    let g = integration::shared(1_000_000, 10_000);
    let (cat, ids) = catalog(LayoutClass::PerRecord);
    let bank: Banks = BankService::new(
        cat,
        g.clone(),
        Heat::default(),
        Reader::default(),
        CoalescingPolicy::default(),
        limits(1024),
    )
    .unwrap();
    let bytes = bank.slru_metadata_bytes(1).unwrap();
    let metadata = g
        .borrow_mut()
        .reserve(&request(bytes, Priority::MandatoryActive))
        .unwrap();
    let mut bank = bank
        .with_slru(SlruPolicy::new(&[(1024, 1)]).unwrap(), &metadata)
        .unwrap();
    assert_eq!(g.borrow_mut().release(&metadata), Err(Error::Busy));
    let t0 = bank.stage(batch(vec![ids[0].clone()])).unwrap();
    drain(&mut bank, &t0);
    let old = bank.publish(&t0, epochs()).unwrap().remove(0);
    let t1 = bank.stage(batch(vec![ids[1].clone()])).unwrap();
    drain(&mut bank, &t1);
    let next = bank.publish(&t1, epochs()).unwrap().remove(0);
    assert!(bank.resident(&ids[0]).unwrap().is_none());
    assert!(bank.resident(&ids[1]).unwrap().is_some());
    assert_eq!(bank.release(&old), Err(Error::Busy));
    // Reload same id while an old evicted allocation is still held. Releasing
    // the old generation must not erase the new policy occupant.
    let t2 = bank.stage(batch(vec![ids[0].clone()])).unwrap();
    drain(&mut bank, &t2);
    let newer = bank.publish(&t2, epochs()).unwrap().remove(0);
    assert_ne!(old.charge().id(), newer.charge().id());
    finish(&mut bank, &t0);
    bank.release(&old).unwrap();
    assert!(bank.slru_policy().unwrap().resident(&ids[0]).is_some());
    finish(&mut bank, &t1);
    bank.release(&next).unwrap();
    finish(&mut bank, &t2);
    bank.release(&newer).unwrap();
    assert_eq!(g.borrow().used().pageable, bytes);
    drop(bank);
    g.borrow_mut().release(&metadata).unwrap();
    assert_eq!(g.borrow().used(), TierBudget::zero(2));
}

struct LookupFake(Option<ObjectManifest>);
impl ObjectStore for LookupFake {
    type Transaction = ();
    fn lookup(&self, _: &ObjectKey) -> Result<Option<ObjectManifest>> {
        Ok(self.0.clone())
    }
    fn begin(&mut self, _: ObjectKey, _: u64, _: Durability) -> Result<()> {
        Err(Error::Unsupported)
    }
    fn put(&mut self, _: &mut (), _: &[u8]) -> Result<()> {
        Err(Error::Unsupported)
    }
    fn commit(&mut self, _: &mut ()) -> Result<ObjectManifest> {
        Err(Error::Unsupported)
    }
    fn cancel(&mut self, _: &mut ()) -> Result<CancelState> {
        Err(Error::Unsupported)
    }
    fn lease(&mut self, _: &ObjectManifest, _: &BudgetRequest) -> Result<ObjectLease> {
        Err(Error::Unsupported)
    }
    fn read(&mut self, _: &ObjectLease, _: u32, _: &mut [u8]) -> Result<u64> {
        panic!("installation must not dispatch bytes")
    }
    fn release(&mut self, _: &ObjectLease) -> Result<()> {
        Err(Error::Unsupported)
    }
    fn evict(&mut self, _: &ObjectKey) -> Result<()> {
        Err(Error::Unsupported)
    }
}

#[test]
fn bank_source_installation_refuses_missing_ambiguous_and_wrong_layout() {
    use memra_tier::pool::FakePinnedPool;
    for case in 0..10 {
        let g = integration::shared(1_000_000, 10_000);
        let table = ple(PleEncoding::F32);
        let l = table.layout(2).unwrap();
        let tensor = l.segments[0].tensor.clone().unwrap();
        let cat = Catalog::new(
            LayoutClass::PerRecord,
            vec![(table.id(2).unwrap(), Some(record(l)))],
        )
        .unwrap();
        let key = ObjectKey {
            version: 1,
            artifact: tensor.artifact,
            semantic_id: tensor.identity().unwrap(),
            layout: [11; 32],
            generation: 4,
        };
        let spec = BankSourceSpec {
            tensor: tensor.clone(),
            key: key.clone(),
            valid_bytes: 8192,
        };
        let mut expected = vec![spec.clone()];
        let mut supplied = vec![(tensor.clone(), key.clone())];
        let mut manifest = Some(ObjectManifest {
            version: 1,
            key,
            valid_bytes: 8192,
            chunks: vec![ChunkRef {
                version: 1,
                encoded_digest: [1; 32],
                valid_bytes: 8192,
                storage_bytes: 8192,
                checksum: [2; 32],
            }],
            durability: Durability::Ephemeral,
        });
        let error = match case {
            0 => None,
            1 => {
                supplied.clear();
                Some(Error::NotFound)
            }
            2 => {
                supplied.push(supplied[0].clone());
                Some(Error::Conflict)
            }
            3 => {
                supplied[0].1.layout = [99; 32];
                Some(Error::InvalidLayout)
            }
            4 => {
                supplied[0].1.generation += 1;
                Some(Error::InvalidLayout)
            }
            5 => {
                manifest = None;
                Some(Error::NotFound)
            }
            6 => {
                manifest.as_mut().unwrap().key.layout = [99; 32];
                Some(Error::InvalidLayout)
            }
            7 => {
                expected[0].valid_bytes = 1;
                Some(Error::InvalidLayout)
            }
            8 => {
                expected.push(spec);
                Some(Error::Conflict)
            }
            9 => {
                manifest.as_mut().unwrap().valid_bytes = 1;
                Some(Error::Incomplete)
            }
            _ => unreachable!(),
        };
        let pc = g
            .borrow_mut()
            .reserve(&request(1023, Priority::MandatoryActive))
            .unwrap();
        let pool = FakePinnedPool::new(1, 512, 512, 0, &pc).unwrap();
        let mut qr = request(0, Priority::MandatoryActive);
        qr.bytes.inflight = 1;
        let qc = g.borrow_mut().reserve(&qr).unwrap();
        let result = BankSource::install(
            cat,
            expected,
            LookupFake(manifest),
            supplied,
            pool,
            request(0, Priority::Demand),
            &qc,
        );
        match (result, error) {
            (Ok(source), None) => drop(source),
            (Err(actual), Some(expected)) => assert_eq!(actual, expected, "case {case}"),
            _ => panic!("unexpected outcome case {case}"),
        }
        g.borrow_mut().release(&pc).unwrap();
        g.borrow_mut().release(&qc).unwrap();
        assert_eq!(g.borrow().used(), TierBudget::zero(2));
    }
}

#[test]
fn patch_v2_source_bounds_preserve_split_offset_zero_without_substitution() {
    assert_eq!(validate_bank_source_extent(512, 16, 528, false), Ok(()));
    assert_eq!(validate_bank_source_extent(512, 16, 16, true), Ok(()));
    assert_eq!(
        validate_bank_source_extent(512, 16, 16, false),
        Err(Error::InvalidLayout)
    );
    assert_eq!(
        validate_bank_source_extent(512, 16, 32, true),
        Err(Error::InvalidLayout)
    );
    assert_eq!(
        validate_bank_source_extent(0, 0, 0, true),
        Err(Error::InvalidLayout)
    );
    assert_eq!(
        validate_bank_source_extent(u64::MAX, 16, u64::MAX, false),
        Err(Error::Overflow)
    );
}

#[test]
fn slru_banked_residency_serial_trace_matches_oracle_and_releases_charge() {
    let trace: serde_json::Value = serde_json::from_str(include_str!(
        "../../../../research/spill-c-20260919/fixtures/slru-synthetic.json"
    ))
    .unwrap();
    let mut ids = Vec::new();
    let mut entries = Vec::new();
    for n in 0..24 {
        let l = layout(n, 13, [16, 40, 80, 256][n as usize % 4] - 4);
        let id = bank_id(n as u32, &l);
        ids.push(id.clone());
        entries.push((id, Some(record(l))));
    }
    let g = integration::shared(1_000_000, 10_000);
    let bank: Banks = BankService::new(
        Catalog::new(LayoutClass::PerRecord, entries).unwrap(),
        g.clone(),
        Heat::default(),
        Reader::default(),
        CoalescingPolicy::default(),
        limits(352),
    )
    .unwrap();
    let bytes = bank.slru_metadata_bytes(6).unwrap();
    let metadata = g
        .borrow_mut()
        .reserve(&request(bytes, Priority::MandatoryActive))
        .unwrap();
    let mut bank = bank
        .with_slru(
            SlruPolicy::new(&[(32, 3), (64, 2), (128, 1)]).unwrap(),
            &metadata,
        )
        .unwrap();
    for (index, row) in trace["serial_rows"].as_array().unwrap().iter().enumerate() {
        let id = &ids[row["key"].as_u64().unwrap() as usize];
        let hit = bank.resident(id).unwrap().is_some();
        assert_eq!(hit, row["result"] == "hit", "step {index}");
        let t = bank.stage(batch(vec![id.clone()])).unwrap();
        drain(&mut bank, &t);
        let output = bank.publish(&t, epochs()).unwrap();
        assert_eq!(
            &*output[0].resource::<Vec<u8>>().unwrap(),
            &expected(output[0].layout())
        );
        assert_eq!(
            bank.slru_policy().unwrap().resident(id),
            row["slot"].as_u64().map(|s| s as usize)
        );
        assert_eq!(
            serde_json::to_value(bank.slru_policy().unwrap().orders()).unwrap(),
            row["orders"],
            "step {index}"
        );
        finish(&mut bank, &t);
        bank.collect_evicted().unwrap();
        assert!(bank.cache_bytes() <= 352);
    }
    for id in &ids {
        bank.evict_cached(id).unwrap();
    }
    bank.collect_evicted().unwrap();
    assert_eq!(g.borrow().used().pageable, bytes);
    drop(bank);
    g.borrow_mut().release(&metadata).unwrap();
    assert_eq!(g.borrow().used(), TierBudget::zero(2));
}

#[test]
fn banked_residency_budget_refuses_instead_of_rounding_up() {
    // Day 43: the host tier is planned per record size; the 16-record clamp and the 256 MiB
    // ceiling are gone, a budget above the machine ceiling or below one record is refused.
    use engine_bridge::{host_bank_ceiling, host_bank_plan};
    const G: u64 = 450_560;
    const D: u64 = 557_056;
    const M: u64 = 860_160;
    let sizes: Vec<u64> = std::iter::repeat_n(G, 200)
        .chain(std::iter::repeat_n(D, 100))
        .chain(std::iter::repeat_n(M, 4))
        .collect();
    let ceiling = 1 << 40;
    assert_eq!(
        host_bank_plan(0, &sizes, ceiling),
        Err("experts-via-tier host bank budget cannot hold one expert record")
    );
    assert_eq!(
        host_bank_plan(M - 1, &sizes, ceiling),
        host_bank_plan(0, &sizes, ceiling)
    );
    // Exactly one largest record: the largest class gets its one slot.
    let one = host_bank_plan(M, &sizes, ceiling).unwrap();
    assert_eq!(one.classes, vec![(M, 1)]);
    assert_eq!((one.planned_bytes, one.records_held), (M, 1));
    // The old 256 MiB default now holds records, not sixteen max-size slots.
    let default = host_bank_plan(256 * 1024 * 1024, &sizes, ceiling).unwrap();
    assert!(default.records_held > 16, "{default:?}");
    assert!(default.planned_bytes <= 256 * 1024 * 1024);
    assert!(default.classes.windows(2).all(|w| w[0].0 < w[1].0));
    assert!(default.classes.iter().any(|&(bytes, _)| bytes == M));
    // A budget over the whole bank holds every record, and no class exceeds its records.
    let all = host_bank_plan(1 << 36, &sizes, ceiling).unwrap();
    assert_eq!(all.classes, vec![(G, 200), (D, 100), (M, 4)]);
    assert_eq!(all.records_held, 304);
    assert_eq!(all.planned_bytes, 200 * G + 100 * D + 4 * M);
    // Above the machine ceiling: refused, never clamped.
    assert_eq!(
        host_bank_plan(ceiling + 1, &sizes, ceiling),
        Err("experts-via-tier host bank budget exceeds the machine ceiling")
    );
    assert_eq!(
        host_bank_plan(M, &[], ceiling),
        Err("experts-via-tier host bank has no expert record")
    );
    // The ceiling is three quarters of MemAvailable; anything else is unknown, not unlimited.
    let meminfo = "MemTotal:       62914560 kB\nMemFree:  1024 kB\nMemAvailable:   41943040 kB\n";
    assert_eq!(host_bank_ceiling(meminfo), Some(41_943_040 * 1024 / 4 * 3));
    assert_eq!(host_bank_ceiling("MemTotal: 1 kB\n"), None);
    assert_eq!(host_bank_ceiling("MemAvailable: x kB\n"), None);
    assert_eq!(host_bank_ceiling("MemAvailable: 12 MB\n"), None);
}
