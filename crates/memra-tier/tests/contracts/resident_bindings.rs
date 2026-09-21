//! Day-11 rule 2 bindings (lead ruling 9): the continuation gate over a suspended layer. CPU
//! ownership models only; the real gate is `memra_kv::Cache::ensure_usable` with its typed
//! `ContinuationRefused`, bound in `memra-kv`'s own tests.
use super::conformance::*;
use std::collections::BTreeSet;
use std::panic::{AssertUnwindSafe, catch_unwind};

/// The register the rule asks for: every suspended layer is named until it is restored.
#[derive(Default)]
struct Register {
    suspended: BTreeSet<u32>,
}
impl ContinuationGateFixture for Register {
    fn suspend(&mut self, layer: u32) {
        assert!(self.suspended.insert(layer));
    }
    fn restore(&mut self, layer: u32) {
        assert!(self.suspended.remove(&layer));
    }
    fn continuation(&mut self) -> std::result::Result<(), Vec<u32>> {
        if self.suspended.is_empty() {
            Ok(())
        } else {
            Err(self.suspended.iter().copied().collect())
        }
    }
}

/// The gate before the rule, the finding verbatim: "`Cache::ensure_usable` returned `Ok(())` on
/// the fully suspended cache (observed)". A one-way, pipeline-scoped taint flag knows nothing
/// about residency, so it lets a suspended cache continue.
#[derive(Default)]
struct TaintOnly {
    tainted: bool,
}
impl ContinuationGateFixture for TaintOnly {
    fn suspend(&mut self, _: u32) {}
    fn restore(&mut self, _: u32) {}
    fn continuation(&mut self) -> std::result::Result<(), Vec<u32>> {
        if self.tainted { Err(vec![]) } else { Ok(()) }
    }
}

#[test]
fn day11_suspended_layers_refuse_continuation_until_restored() {
    required_resident_continuation(&mut Register::default(), &[3, 0, 7]);
}

#[test]
fn day11_red_arm_taint_flag_lets_a_suspended_cache_continue() {
    let mut gate = TaintOnly::default();
    let schedule = catch_unwind(AssertUnwindSafe(|| {
        required_resident_continuation(&mut gate, &[3, 0]);
    }));
    assert!(schedule.is_err(), "a taint flag must not pass rule 2");
    gate.suspend(3);
    assert_eq!(gate.continuation(), Ok(())); // the observed answer on the suspended cache
}
