//! Day 77 (`research/spill-c-20260919/DAY77.md`): I17, the grouped prefetch's residency and staging each in one
//! owner-registry entry.
use super::day64::group_bank;
use super::*;

/// I17: `host_resident_many` answers each record as `host_resident` does, in order, in one registry entry, and
/// refuses more than a group; `with_bytes_each` lends each record's bytes in order, as `with_bytes_at` lends them,
/// refuses a finished or foreign group token, stops at the first `Err` of the callback and leaves the group pending.
#[test]
fn the_group_calls_answer_as_the_per_record_calls() {
    let (bank, blocks) = group_bank();
    let mut owner = ExpertBankOwner::register(Box::new(bank), 1).unwrap();
    let proxy = owner.proxy();
    let ids: Vec<_> = blocks.iter().map(|b| b.0).collect();
    let singles: Vec<bool> = ids
        .iter()
        .map(|&id| proxy.host_resident(id).unwrap())
        .collect();
    let many = proxy.host_resident_many(&ids).unwrap();
    assert_eq!(&many[..ids.len()], singles.as_slice());
    assert!(many[ids.len()..].iter().all(|&r| !r));
    let four = [ids[0], ids[1], ids[2], ids[0]];
    assert_eq!(proxy.host_resident_many(&four).err(), Some(Error::Capacity));
    assert_eq!(proxy.host_resident_many(&[]).unwrap(), [false; MAX_GROUP]);

    let group = proxy.demand_many(&blocks).unwrap();
    let each: Vec<Vec<u8>> = (0..blocks.len())
        .map(|index| proxy.with_bytes_at(&group, index, <[u8]>::to_vec).unwrap())
        .collect();
    let mut seen = Vec::new();
    let walked = proxy
        .with_bytes_each(&group, |index, bytes| {
            seen.push((index, bytes.to_vec()));
            Ok::<(), ()>(())
        })
        .unwrap();
    assert_eq!(walked, Ok(()));
    assert_eq!(seen, each.iter().cloned().enumerate().collect::<Vec<_>>());
    // The first callback error stops the walk; the group stays pending and finishes as before.
    let mut calls = 0;
    let stopped = proxy
        .with_bytes_each(&group, |index, _| {
            calls += 1;
            if index == 1 { Err("stop") } else { Ok(()) }
        })
        .unwrap();
    assert_eq!((stopped, calls), (Err("stop"), 2));
    assert_eq!(owner.close(), Err(Error::Busy));
    proxy.finish_group(&group).unwrap();
    assert_eq!(
        proxy.with_bytes_each(&group, |_, _| Ok::<(), ()>(())).err(),
        Some(Error::UnknownTicket)
    );
    // A single lease's number is not a group.
    let single = proxy.demand(blocks[0].0, 16).unwrap();
    let group2 = proxy.demand_many(&blocks[1..]);
    assert_eq!(group2.err(), Some(Error::Capacity));
    proxy.finish(&single).unwrap();
    owner.close().unwrap();
}

/// I17: a proxy on another thread is refused, as every proxy call is.
#[test]
fn the_group_calls_refuse_another_thread() {
    let (bank, blocks) = group_bank();
    let mut owner = ExpertBankOwner::register(Box::new(bank), 1).unwrap();
    let proxy = owner.proxy();
    let ids: Vec<_> = blocks.iter().map(|b| b.0).collect();
    let moved = proxy.clone();
    let refused = std::thread::spawn(move || moved.host_resident_many(&ids).err())
        .join()
        .unwrap();
    assert_eq!(refused, Some(Error::WrongOwner));
    owner.close().unwrap();
}
