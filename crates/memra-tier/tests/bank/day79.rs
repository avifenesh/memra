//! Day 79 (`research/spill-c-20260919/DAY79.md`): I18 keeps a ticket's records by position; every output lease is the
//! one the id-keyed map gave (the record whose id sits at that position), whether the group's records were all
//! missing, partly cached or all cached. Duplicated ids within one batch stay covered by the rows test in `main.rs`
//! (`rows_straddles_duplicates_pool_smaller_than_batch_and_retirement`).
use super::day64::group_bank;
use super::*;

fn singles() -> Vec<Vec<u8>> {
    let (bank, blocks) = group_bank();
    let mut owner = ExpertBankOwner::register(Box::new(bank), 1).unwrap();
    let proxy = owner.proxy();
    let out = blocks
        .iter()
        .map(|&(local, bytes)| {
            let token = proxy.demand(local, bytes).unwrap();
            let got = proxy.with_bytes(&token, <[u8]>::to_vec).unwrap();
            proxy.finish(&token).unwrap();
            got
        })
        .collect();
    owner.close().unwrap();
    out
}

#[test]
fn a_group_publishes_each_positions_own_record_missing_partly_or_fully_cached() {
    let want = singles();
    // All missing: a fresh bank's first grouped demand.
    let (bank, blocks) = group_bank();
    let mut owner = ExpertBankOwner::register(Box::new(bank), 1).unwrap();
    let proxy = owner.proxy();
    let group = proxy.demand_many(&blocks).unwrap();
    for (index, want) in want.iter().enumerate() {
        assert_eq!(
            &proxy.with_bytes_at(&group, index, <[u8]>::to_vec).unwrap(),
            want
        );
    }
    proxy.finish_group(&group).unwrap();
    // All cached now: the same records again, and a reversed order keeps each position's own record.
    let reversed: Vec<_> = blocks.iter().rev().copied().collect();
    let group = proxy.demand_many(&reversed).unwrap();
    for (index, want) in want.iter().rev().enumerate() {
        assert_eq!(
            &proxy.with_bytes_at(&group, index, <[u8]>::to_vec).unwrap(),
            want
        );
    }
    proxy.finish_group(&group).unwrap();
    owner.close().unwrap();
    // Partly cached: one record demanded alone first, then the group.
    let (bank, blocks) = group_bank();
    let mut owner = ExpertBankOwner::register(Box::new(bank), 1).unwrap();
    let proxy = owner.proxy();
    let token = proxy.demand(blocks[1].0, blocks[1].1).unwrap();
    proxy.finish(&token).unwrap();
    let group = proxy.demand_many(&blocks).unwrap();
    for (index, want) in want.iter().enumerate() {
        assert_eq!(
            &proxy.with_bytes_at(&group, index, <[u8]>::to_vec).unwrap(),
            want
        );
    }
    proxy.finish_group(&group).unwrap();
    owner.close().unwrap();
}
