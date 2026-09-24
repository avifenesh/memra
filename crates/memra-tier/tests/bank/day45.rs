//! Day 45 (`research/spill-c-20260919/DAY45.md`): the host fill's admission. A record offered
//! with the catalog's digest is published into a free slot and later demands are host hits with
//! no read; a wrong digest is refused; a resident record is dropped; a full tier reports `Full`;
//! the fill never evicts; layouts outside the door's single payload segment are unsupported.
use super::*;

fn single(n: u32) -> (BankId, CatalogRecord) {
    let mut l = layout(u64::from(n), 13, 16);
    l.segments.truncate(1);
    l.requirements.truncate(1);
    let id = bank_id(n, &l);
    (id, record(l))
}

fn fill_bank(
    slots: usize,
    records: u32,
) -> (
    SlruExpertDispatch<Heat, Reader>,
    Vec<BankId>,
    Vec<CatalogRecord>,
) {
    let g = gov();
    let mut entries = Vec::new();
    let mut ids = Vec::new();
    let mut recs = Vec::new();
    for n in 0..records {
        let (id, r) = single(n);
        ids.push(id.clone());
        recs.push(r.clone());
        entries.push((id, Some(r)));
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

/// The bytes a read of record `n` returns (the fixture reader's rule), and their digest.
fn filled(r: &CatalogRecord) -> (Vec<u8>, Digest) {
    let s = &r.layout.segments[0];
    let bytes: Vec<u8> = (s.offset..s.offset + s.storage_bytes)
        .map(|p| byte(s.tensor.as_ref().unwrap(), p))
        .collect();
    let digest = checksum(&bytes);
    (bytes, digest)
}

#[test]
fn a_filled_record_is_a_host_hit_without_a_read() {
    let (mut d, ids, recs) = fill_bank(4, 8);
    let local = dispatch_id(&ids[3].record).unwrap();
    let (bytes, digest) = filled(&recs[3]);
    assert_eq!(
        d.admit_filled(local, HostBytes::Heap(bytes.clone()), digest),
        Ok(FillOutcome::Admitted)
    );
    assert!(d.bank().slru_policy().unwrap().resident(&ids[3]).is_some());
    let reads = d.bank().reader().calls.len();
    let demand = d.demand(local, 16).unwrap();
    assert_eq!(
        d.bank().reader().calls.len(),
        reads,
        "a filled record was read again"
    );
    assert_eq!(&*demand.lease.resource::<Vec<u8>>().unwrap(), &bytes);
    d.finish(demand).unwrap();
    // Offered again while resident: dropped, never replaced.
    let (bytes, digest) = filled(&recs[3]);
    assert_eq!(
        d.admit_filled(local, HostBytes::Heap(bytes), digest),
        Ok(FillOutcome::Dropped)
    );
}

#[test]
fn a_wrong_digest_is_refused_and_nothing_is_published() {
    let (mut d, ids, recs) = fill_bank(4, 8);
    let local = dispatch_id(&ids[2].record).unwrap();
    let (mut bytes, digest) = filled(&recs[2]);
    bytes[0] ^= 1;
    // Changed bytes carry their own digest, which is not the catalog's; a flipped digest never
    // is either.
    assert_eq!(
        d.admit_filled(local, HostBytes::Heap(bytes.clone()), checksum(&bytes)),
        Ok(FillOutcome::Refused)
    );
    let mut wrong = digest;
    wrong[0] ^= 1;
    assert_eq!(
        d.admit_filled(local, HostBytes::Heap(bytes), wrong),
        Ok(FillOutcome::Refused)
    );
    assert!(d.bank().slru_policy().unwrap().resident(&ids[2]).is_none());
    assert_eq!(d.bank().cached_records(), 0);
    // Wrong length is refused too.
    let (bytes, digest) = filled(&recs[2]);
    assert_eq!(
        d.admit_filled(local, HostBytes::Heap(bytes[..8].to_vec()), digest),
        Ok(FillOutcome::Refused)
    );
}

#[test]
fn the_fill_never_evicts_and_reports_a_full_tier() {
    let (mut d, ids, recs) = fill_bank(3, 8);
    for n in 0..3 {
        let (bytes, digest) = filled(&recs[n]);
        let local = dispatch_id(&ids[n].record).unwrap();
        assert_eq!(
            d.admit_filled(local, HostBytes::Heap(bytes), digest),
            Ok(FillOutcome::Admitted)
        );
    }
    let before = d.bank().slru_policy().unwrap().orders();
    let (bytes, digest) = filled(&recs[5]);
    let local = dispatch_id(&ids[5].record).unwrap();
    assert_eq!(
        d.admit_filled(local, HostBytes::Heap(bytes), digest),
        Ok(FillOutcome::Full)
    );
    assert_eq!(
        d.bank().slru_policy().unwrap().orders(),
        before,
        "the fill moved a resident"
    );
    for id in &ids[..3] {
        assert!(d.bank().slru_policy().unwrap().resident(id).is_some());
    }
    assert!(d.bank().slru_policy().unwrap().resident(&ids[5]).is_none());
    // Every owned lease is cached (none leaked by the refusals).
    assert_eq!(d.bank().owned_leases(), d.bank().cached_records());
}

#[test]
fn only_the_doors_single_payload_segment_is_filled() {
    // The two-segment fixture layout (payload plus macro scale) is not a door record.
    let g = gov();
    let l = layout(9, 13, 16);
    let id = bank_id(9, &l);
    let r = record(l);
    let bank = BankService::<ExpertDomain, _, _>::new(
        Catalog::new(LayoutClass::Uniform, vec![(id.clone(), Some(r))]).unwrap(),
        g.clone(),
        Heat::default(),
        Reader::default(),
        CoalescingPolicy {
            granularity: 1,
            slot_bytes: 32,
        },
        limits(64),
    )
    .unwrap();
    let req = request(bank.slru_metadata_bytes(1).unwrap(), Priority::Demand);
    let metadata = g.borrow_mut().reserve(&req).unwrap();
    let mut bank = bank
        .with_slru(SlruPolicy::new(&[(32, 1)]).unwrap(), &metadata)
        .unwrap();
    assert_eq!(
        bank.admit_filled(&id, HostBytes::Heap(vec![0; 20]), [0; 32], &req),
        Err(Error::Unsupported)
    );
}
