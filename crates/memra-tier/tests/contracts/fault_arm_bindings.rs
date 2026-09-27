//! Day 12 (lane D): the `cancel-restore` fault arm's rule-1 rows bound to the CPU fake
//! transport. The rows are the gate's own (`kv_tier_gate/fault_contract.rs`, included by path
//! so the names and expectations cannot drift from the native arm); the fixture is lane A's
//! day-11 transport with its `legacy` switch. `legacy = false` records every row green;
//! `legacy = true` is the red arm: the transport before the rule drains the cancelled restore,
//! the seam rows fail, and the verdict is a failed cell, never PASS and never the day-11 typed
//! refusal. No CUDA, no GPU claim.
use super::support::*;
use super::transfer::Transfers;
use memra_tier::contracts::*;

#[allow(dead_code)]
#[path = "../../../memra-engine/src/bin/kv_tier_gate/fault_contract.rs"]
mod fault_contract;
use fault_contract::{
    Arm, Check, cancel_restore_recover, cancel_restore_retire, cancel_restore_revoke, verdict,
};

/// The arm's precondition on the fake: an H2D restore submitted, its producer complete and the
/// source side idle, nothing published (`finish` observes every completion; nothing is taken).
fn restore_ready(
    t: &mut Transfers,
    gov: &Shared,
) -> (TransferTicket, ChargedLease, ChargedLease, DeviceLease) {
    let (ticket, hc, dc, keep) = super::transfer::pending_restore(t, gov);
    t.finish(&ticket);
    (ticket, hc, dc, keep)
}

fn names(checks: &[Check]) -> Vec<&'static str> {
    checks.iter().map(|c| c.name).collect()
}

#[test]
fn day12_cancel_restore_rows_hold_on_a_transport_with_the_seam() {
    let gov = shared();
    let mut t = Transfers::new(gov.clone());
    let (ticket, hc, dc, keep) = restore_ready(&mut t, &gov);
    let mut checks = vec![];
    cancel_restore_revoke(&mut t, &ticket, epochs(), &mut checks);
    let host = cancel_restore_recover(&mut t, &ticket, &mut checks)
        .expect("the seam hands the source back");
    // The hand-back is the untouched source, still carrying its own charge.
    assert_eq!(host.bytes().unwrap(), &[3, 20, 37]);
    assert_eq!(gov.borrow_mut().release(&hc), Err(Error::Busy));
    cancel_restore_retire(&mut t, &ticket, &mut checks);
    assert_eq!(
        names(&checks),
        [
            "cancel",
            "ready-view-after-cancel",
            "take-after-cancel",
            "retire-holds-source",
            "retire-source-holds",
            "recover-source",
            "recover-source-once",
            "cancel-after-recovery",
            "retire",
            "acknowledge"
        ]
    );
    for c in &checks {
        assert!(
            c.ok(),
            "{}: expected {} observed {}",
            c.name,
            c.expected,
            c.observed
        );
    }
    // Every row the generic sequence records is one the arm requires; the native arm adds its
    // backend-only rows (`with-destination-after-cancel`, bytes, accounting, `restored-identical`).
    for c in &checks {
        assert!(
            Arm::CancelRestore.required_checks().contains(&c.name),
            "{}",
            c.name
        );
    }
    drop(host);
    gov.borrow_mut().release(&hc).unwrap();
    t.owner.release(&keep).unwrap();
    gov.borrow_mut().release(&dc).unwrap();
    assert_eq!(gov.borrow().used, TierBudget::zero(2));
}

/// Red arm. Day-11 finding, verbatim: "No H2D-source recovery after `cancel`: the demoted
/// copy's only caller handle is consumed at submission and released by `retire`; the D2H twin
/// is take-once (`AlreadyReleased`). A cancelled restore can only be drained." On the transport
/// before the rule the hold rows fail (`retire` drains, `recover_source` is `Unsupported`), the
/// arm stops where the native arm leaves the layer out, and the verdict names the seam rows.
#[test]
fn day12_red_arm_legacy_transport_fails_the_seam_rows_and_is_never_a_refusal() {
    let gov = shared();
    let mut t = Transfers::new(gov.clone());
    t.legacy = true;
    let (ticket, hc, dc, keep) = restore_ready(&mut t, &gov);
    let mut checks = vec![];
    cancel_restore_revoke(&mut t, &ticket, epochs(), &mut checks);
    assert!(
        checks.iter().all(Check::ok),
        "revocation itself is granted by the legacy transport"
    );
    let recovered = cancel_restore_recover(&mut t, &ticket, &mut checks);
    assert!(
        recovered.is_none(),
        "the legacy transport must not hand a source back"
    );
    let row = |name: &str| {
        checks
            .iter()
            .find(|c| c.name == name)
            .unwrap()
            .observed
            .clone()
    };
    assert_eq!(row("retire-holds-source"), "Ok(())"); // drained, nothing held for the caller
    assert_eq!(row("recover-source"), "Err(Unsupported)");
    let v = verdict(Arm::CancelRestore, &checks);
    assert!(!v.pass);
    assert!(
        v.failed.contains(&"retire-holds-source") && v.failed.contains(&"recover-source"),
        "{:?}",
        v.failed
    );
    assert!(
        v.missing.contains(&"recovered-source-intact") && v.missing.contains(&"restored-identical")
    );
    assert!(
        v.line
            .starts_with("fault arm cancel-restore did not prove its contract"),
        "{}",
        v.line
    );
    assert!(
        !v.line.starts_with("REFUSED") && !v.line.contains("PASS"),
        "{}",
        v.line
    );
    // The day-11 gate printed this instead; it is the finding the failed rows now stand for.
    const DAY11: &str = "REFUSED: cancel-restore revoked publication, but the transfer contract has no seam to recover the H2D source after cancellation; no tokens, budget drained";
    assert_ne!(v.line, DAY11);
    // The drained entry acknowledges; the source and its charge went with it, nothing to hand back.
    t.acknowledge(&ticket).unwrap();
    gov.borrow_mut().release(&hc).unwrap();
    t.owner.release(&keep).unwrap();
    gov.borrow_mut().release(&dc).unwrap();
    assert_eq!(gov.borrow().used, TierBudget::zero(2));
}
