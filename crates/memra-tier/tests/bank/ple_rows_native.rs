// Actual native bridge source, compiled without CUDA; no model/GPU claim.
#[path = "../../../memra-engine/src/ple_rows_tier.rs"]
mod native;
use memra_tier::{contracts::*, tier::Governor};
use native::{HostTable, PleRowsTier};
use std::{cell::RefCell, rc::Rc, sync::Arc};

fn setup(cap: u64) -> (PleRowsTier, Rc<RefCell<Governor>>) {
    let mut budget = TierBudget::zero(0);
    budget.pageable = cap;
    budget.staging = cap;
    budget.inflight = 8;
    let g = Rc::new(RefCell::new(
        Governor::new(budget, TierBudget::zero(0), 8, 0, Arc::new(|| 0)).unwrap(),
    ));
    (PleRowsTier::new(g.clone()), g)
}
#[test]
fn native_ple_rows_bits_duplicates_forced_misses_and_drained_budget() {
    let (tier, g) = setup(1 << 24);
    let bits = [
        0u32, 0x80000000, 0x3f800000, 0x7fc10042, 0x7f800000, 0xff800000,
    ];
    let f32s: Vec<_> = bits.iter().copied().map(f32::from_bits).collect();
    let bf16: Vec<_> = bits
        .iter()
        .flat_map(|b| ((*b >> 16) as u16).to_le_bytes())
        .collect();
    for source in [HostTable::F32(&f32s), HostTable::Bf16(&bf16)] {
        for _ in 0..2 {
            let out = tier.gather(source, 2, 2, &[2, 0, 2, 1]).unwrap();
            let expected: Vec<_> = [2, 0, 2, 1]
                .iter()
                .flat_map(|r| bits[r * 2..r * 2 + 2].iter().copied())
                .collect();
            for (actual, mut expected) in out.iter().zip(expected) {
                if matches!(source, HostTable::Bf16(_)) {
                    expected &= 0xffff0000;
                }
                assert_eq!(actual.to_bits(), expected);
            }
            drop(out);
            assert_eq!(g.borrow().used(), TierBudget::zero(0));
        }
    }
    assert_eq!(tier.calls.get(), 4);
    assert_eq!(tier.reads.get(), 4); // coalesced complete tiny table, never cache hits
}
#[test]
fn native_ple_rows_refuses_invalid_bounds_and_budget_without_leak() {
    let (tier, g) = setup(1 << 24);
    for ids in [vec![-1], vec![3], vec![], vec![0; 4097]] {
        assert!(tier.gather(HostTable::F32(&[1.; 6]), 0, 2, &ids).is_err());
        assert_eq!(g.borrow().used(), TierBudget::zero(0));
    }
    assert!(tier.gather(HostTable::Bf16(&[0; 3]), 0, 2, &[0]).is_err());
    let (tiny, g) = setup(1);
    assert!(matches!(
        tiny.gather(HostTable::F32(&[1.; 6]), 0, 2, &[0]),
        Err(Error::Capacity)
    ));
    assert_eq!(g.borrow().used(), TierBudget::zero(0));
}

#[test]
fn native_ple_rows_legacy_output_and_partial_admission() {
    let legacy = native::GatheredRows::legacy(vec![1.]);
    assert_eq!(&*legacy, &[1.]);
    for cap in [8200, 8300, 8500, 9000, 10000] {
        let (tier, g) = setup(cap);
        let result = tier.gather(HostTable::F32(&[1.; 6]), 0, 2, &[0]);
        drop(result);
        assert_eq!(g.borrow().used(), TierBudget::zero(0));
    }
}
