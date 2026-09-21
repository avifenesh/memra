//! Pure fault-arm contract for `kv-tier-gate --fault <arm>`; `rustc --test` exercises it
//! without CUDA. An arm passes only when every check the arm requires was recorded and every
//! recorded check observed exactly what the tier contract promises at the documented call. A
//! red arm (fault not injected, a contract answer that differs, or a check that was never
//! recorded) never prints the PASS line. Day 12: the two arms that ended in a typed refusal on
//! day 11 (no seam in the contracts as they were) drive lane A's rule seams,
//! `TransferEngine::recover_source` and `Cache::suspend_layer` / `resume_layer`; a transport or
//! a cache without the seam fails their required checks and is a failed cell, never a refusal
//! and never PASS. The day-11 refusal texts survive only as the red arms' recorded findings
//! (`crates/memra-tier/tests/reclaim/fault.rs`, `research/spill-d-20260919/verify-day11.py`).
use memra_tier::contracts::{Epochs, TransferEngine, TransferTicket};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Arm {
    CancelDemote,
    CancelRestore,
    CorruptHost,
    MissingHost,
    HostBudgetShort,
    DeviceShort,
    RequireResident,
}

impl Arm {
    pub const ALL: [Arm; 7] = [
        Arm::CancelDemote,
        Arm::CancelRestore,
        Arm::CorruptHost,
        Arm::MissingHost,
        Arm::HostBudgetShort,
        Arm::DeviceShort,
        Arm::RequireResident,
    ];
    pub const NAMES: &str = "cancel-demote, cancel-restore, corrupt-host, missing-host, host-budget-short, device-short or require-resident";

    pub fn name(self) -> &'static str {
        match self {
            Arm::CancelDemote => "cancel-demote",
            Arm::CancelRestore => "cancel-restore",
            Arm::CorruptHost => "corrupt-host",
            Arm::MissingHost => "missing-host",
            Arm::HostBudgetShort => "host-budget-short",
            Arm::DeviceShort => "device-short",
            Arm::RequireResident => "require-resident",
        }
    }
    pub fn parse(value: &str) -> Option<Arm> {
        Arm::ALL.into_iter().find(|arm| arm.name() == value)
    }
    /// The cache is whole again when the arm returns, so the gate captures it and compares
    /// the manifest hash with the suspended prefix (`restored-identical`). The two arms that
    /// lose a K plane (`corrupt-host`, `missing-host`) never restore it.
    pub fn restores_cache(self) -> bool {
        !matches!(self, Arm::CorruptHost | Arm::MissingHost)
    }
    /// A passing arm whose state is resident and bit-identical lets the same tokenwise
    /// continuation run; the offline verifier compares it with the frozen baseline bundle.
    pub fn continues(self) -> bool {
        self.restores_cache()
    }
    /// Every name here must be recorded, in any order, for the arm to pass. Extra recorded rows
    /// are evidence and must hold too (`verdict`).
    pub fn required_checks(self) -> &'static [&'static str] {
        match self {
            Arm::CancelDemote => &[
                "cancel",
                "take-after-cancel",
                "retire",
                "acknowledge",
                "pinned-after-cancel",
                "source-returned",
                "budget-zero",
                "restored-identical",
            ],
            // Rule 1 (lane A, day 11): the cancelled H2D holds its source for the caller,
            // hands it back exactly once as the demoted copy, then retires normally and the
            // same plane goes through the roundtrip's own `restore`.
            Arm::CancelRestore => &[
                "cancel",
                "ready-view-after-cancel",
                "with-destination-after-cancel",
                "take-after-cancel",
                "retire-holds-source",
                "retire-source-holds",
                "recover-source",
                "recovered-source-checksum",
                "recovered-source-intact",
                "recover-source-once",
                "cancel-after-recovery",
                "retire",
                "acknowledge",
                "pinned-held-by-recovered-lease",
                "retake-demoted-copy",
                "budget-zero",
                "restored-identical",
            ],
            Arm::CorruptHost => &[
                "write-under-live-ticket",
                "write-sole-owner",
                "restore-integrity",
                "device-registry-unchanged",
                "budget-zero",
            ],
            Arm::MissingHost => &[
                "completion-checksum",
                "remove",
                "pinned-released",
                "retake-removed-copy",
                "budget-zero",
            ],
            Arm::HostBudgetShort => &[
                "whole-state-admission",
                "layers-resident",
                "pinned-charged",
                "budget-zero",
                "restored-identical",
            ],
            Arm::DeviceShort => &[
                "restore-admission",
                "device-registry-unchanged",
                "host-copy-intact",
                "budget-zero",
                "restored-identical",
            ],
            // Rule 2 (lane A, day 11): the register names every suspended layer, the
            // continuation gate refuses with the typed error naming exactly them (asking is not
            // restoring, a partial resume still refuses), and answers `Ok(())` once every layer
            // is back.
            Arm::RequireResident => &[
                "suspended-register",
                "continuation-gate-on-suspended-cache",
                "continuation-gate-asked-twice",
                "continuation-gate-after-partial-resume",
                "register-empty-after-resume",
                "continuation-gate-after-resume",
                "budget-zero",
                "restored-identical",
            ],
        }
    }
}

/// One contract answer at one documented call: the frozen expectation and what the native
/// backend actually returned, both rendered with `Debug` so a typed error compares exactly.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Check {
    pub name: &'static str,
    pub expected: String,
    pub observed: String,
}
impl Check {
    pub fn new(
        name: &'static str,
        expected: impl Into<String>,
        observed: impl Into<String>,
    ) -> Self {
        Self {
            name,
            expected: expected.into(),
            observed: observed.into(),
        }
    }
    pub fn ok(&self) -> bool {
        self.expected == self.observed
    }
}

/// Rule 1 rows of the `cancel-restore` arm, generic over the transport so the native arm and
/// the CPU red arm (a transport without the seam) record the same names and expectations.
/// Precondition for `cancel_restore_revoke`: the H2D is submitted and its producer completion
/// observed; nothing is published. The native arm records its backend-only rows
/// (`with-destination-after-cancel`, the recovered bytes, accounting) between these calls.
pub fn cancel_restore_revoke<T: TransferEngine>(
    t: &mut T,
    ticket: &TransferTicket,
    epochs: Epochs,
    checks: &mut Vec<Check>,
) {
    checks.push(Check::new(
        "cancel",
        "Ok(PublicationRevoked)",
        format!("{:?}", t.cancel(ticket)),
    ));
    checks.push(Check::new(
        "ready-view-after-cancel",
        "Err(Cancelled)",
        format!("{:?}", t.ready_view(ticket, 0, epochs).map(|_| ())),
    ));
    checks.push(Check::new(
        "take-after-cancel",
        "Err(Cancelled)",
        format!("{:?}", t.take_destination(ticket, 0, epochs)),
    ));
}
/// The hold and the hand-back: a cancelled restore keeps its source for the caller (`retire`
/// and `retire_source` answer `Busy`) until `recover_source` returns it. `None` means the
/// transport did not hand the source back; the rows say what it answered instead.
pub fn cancel_restore_recover<T: TransferEngine>(
    t: &mut T,
    ticket: &TransferTicket,
    checks: &mut Vec<Check>,
) -> Option<T::Host> {
    checks.push(Check::new(
        "retire-holds-source",
        "Err(Busy)",
        format!("{:?}", t.retire(ticket, None)),
    ));
    checks.push(Check::new(
        "retire-source-holds",
        "Err(Busy)",
        format!("{:?}", t.retire_source(ticket)),
    ));
    let recovered = t.recover_source(ticket, 0);
    checks.push(Check::new(
        "recover-source",
        "Ok(host)",
        match &recovered {
            Ok(_) => "Ok(host)".to_owned(),
            Err(error) => format!("Err({error:?})"),
        },
    ));
    recovered.ok()
}
/// After the hand-back: once only, no revocation over a source that left, normal retirement.
pub fn cancel_restore_retire<T: TransferEngine>(
    t: &mut T,
    ticket: &TransferTicket,
    checks: &mut Vec<Check>,
) {
    checks.push(Check::new(
        "recover-source-once",
        "Err(AlreadyReleased)",
        format!("{:?}", t.recover_source(ticket, 0).map(|_| ())),
    ));
    checks.push(Check::new(
        "cancel-after-recovery",
        "Err(AlreadyReleased)",
        format!("{:?}", t.cancel(ticket)),
    ));
    checks.push(Check::new(
        "retire",
        "Ok(())",
        format!("{:?}", t.retire(ticket, None)),
    ));
    checks.push(Check::new(
        "acknowledge",
        "Ok(())",
        format!("{:?}", t.acknowledge(ticket)),
    ));
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Verdict {
    /// `true` only for `FAULT-ARM PASS <name>`.
    pub pass: bool,
    /// The final console line: the PASS line or the failure text (which `cli::diagnostic`
    /// prefixes, so the collector classifies a failed cell).
    pub line: String,
    pub missing: Vec<&'static str>,
    pub failed: Vec<&'static str>,
}

pub fn pass_line(arm: Arm) -> String {
    format!("FAULT-ARM PASS {}", arm.name())
}

/// A failed or missing check is a failed cell. There is no third outcome: a backend without a
/// seam the arm requires fails the seam's rows and is reported as exactly that.
pub fn verdict(arm: Arm, checks: &[Check]) -> Verdict {
    let missing: Vec<&'static str> = arm
        .required_checks()
        .iter()
        .copied()
        .filter(|name| !checks.iter().any(|c| c.name == *name))
        .collect();
    let mut failed: Vec<&'static str> = checks.iter().filter(|c| !c.ok()).map(|c| c.name).collect();
    // A required name recorded twice with different answers is one failure, not a pass.
    failed.dedup();
    if !missing.is_empty() || !failed.is_empty() {
        return Verdict {
            pass: false,
            line: format!(
                "fault arm {} did not prove its contract; missing=[{}] failed=[{}]",
                arm.name(),
                missing.join(","),
                failed.join(","),
            ),
            missing,
            failed,
        };
    }
    Verdict {
        pass: true,
        line: pass_line(arm),
        missing,
        failed,
    }
}

/// Receipt rows: `check\texpected\tobserved\tok`, one per recorded check, in recording order.
pub fn checks_tsv(checks: &[Check]) -> String {
    let mut table = String::from("check\texpected\tobserved\tok\n");
    for c in checks {
        table.push_str(&format!(
            "{}\t{}\t{}\t{}\n",
            c.name,
            c.expected.replace(['\t', '\n'], " "),
            c.observed.replace(['\t', '\n'], " "),
            c.ok()
        ));
    }
    table
}

#[cfg(test)]
mod tests {
    use super::*;

    fn green(arm: Arm) -> Vec<Check> {
        arm.required_checks()
            .iter()
            .map(|name| Check::new(name, "same", "same"))
            .collect()
    }

    #[test]
    fn names_round_trip_and_unknown_is_none() {
        for arm in Arm::ALL {
            assert_eq!(Arm::parse(arm.name()), Some(arm));
            assert!(Arm::NAMES.contains(arm.name()));
        }
        for junk in [
            "",
            "cancel",
            "Cancel-Demote",
            "cancel-demote ",
            "vmm",
            "--fault",
        ] {
            assert_eq!(Arm::parse(junk), None, "{junk:?}");
        }
    }

    #[test]
    fn every_arm_can_pass_and_the_two_holing_arms_never_continue() {
        let passing: Vec<Arm> = Arm::ALL
            .into_iter()
            .filter(|arm| verdict(*arm, &green(*arm)).pass)
            .collect();
        assert_eq!(passing, Arm::ALL);
        let continuing: Vec<Arm> = Arm::ALL.into_iter().filter(|a| a.continues()).collect();
        assert_eq!(
            continuing,
            [
                Arm::CancelDemote,
                Arm::CancelRestore,
                Arm::HostBudgetShort,
                Arm::DeviceShort,
                Arm::RequireResident
            ]
        );
        for arm in Arm::ALL {
            let v = verdict(arm, &green(arm));
            assert_eq!(v.line, format!("FAULT-ARM PASS {}", arm.name()));
            assert!(v.missing.is_empty() && v.failed.is_empty());
            assert!(!v.line.contains('\n'));
            // A continuing arm always restores the cache; the holing arms never continue.
            assert_eq!(arm.continues(), arm.restores_cache(), "{arm:?}");
            assert_eq!(
                arm.restores_cache(),
                !matches!(arm, Arm::CorruptHost | Arm::MissingHost)
            );
            // Every arm that restores must prove bit-identity.
            assert_eq!(
                arm.required_checks().contains(&"restored-identical"),
                arm.restores_cache(),
                "{arm:?}"
            );
            assert!(arm.required_checks().contains(&"budget-zero"), "{arm:?}");
        }
    }

    /// The day-11 findings, replayed as answers: a transport that drains a cancelled restore
    /// (`retire` `Ok(())`, `recover_source` `Unsupported`) and a cache whose gate accepts a
    /// suspended cache (`Ok(())`) fail the seam rows. That is a failed cell with the seam
    /// rows named, never a refusal and never PASS.
    #[test]
    fn a_backend_without_the_seam_is_a_failed_cell_not_a_refusal() {
        let mut checks = green(Arm::CancelRestore);
        for c in &mut checks {
            match c.name {
                "retire-holds-source" => c.observed = "Ok(())".into(),
                "retire-source-holds" => c.observed = "Ok(())".into(),
                "recover-source" => c.observed = "Err(Unsupported)".into(),
                _ => {}
            }
        }
        let v = verdict(Arm::CancelRestore, &checks);
        assert!(!v.pass);
        assert_eq!(
            v.failed,
            [
                "retire-holds-source",
                "retire-source-holds",
                "recover-source"
            ]
        );
        assert_eq!(
            v.line,
            "fault arm cancel-restore did not prove its contract; missing=[] failed=[retire-holds-source,retire-source-holds,recover-source]"
        );
        assert!(!v.line.starts_with("REFUSED") && !v.line.contains("PASS"));
        let mut checks = green(Arm::RequireResident);
        for c in &mut checks {
            if c.name.starts_with("continuation-gate-on") || c.name.ends_with("asked-twice") {
                c.observed = "Ok(())".into();
            }
        }
        let v = verdict(Arm::RequireResident, &checks);
        assert!(!v.pass);
        assert_eq!(
            v.failed,
            [
                "continuation-gate-on-suspended-cache",
                "continuation-gate-asked-twice"
            ]
        );
        assert!(!v.line.starts_with("REFUSED") && !v.line.contains("PASS"));
    }

    #[test]
    fn red_arm_fault_not_injected_is_a_failure_not_a_pass() {
        // cancel-demote: the backend answered AlreadyPublished, so nothing was revoked.
        let mut checks = green(Arm::CancelDemote);
        checks[0] = Check::new("cancel", "Ok(PublicationRevoked)", "Ok(AlreadyPublished)");
        let v = verdict(Arm::CancelDemote, &checks);
        assert!(!v.pass);
        assert_eq!(v.failed, ["cancel"]);
        assert!(
            v.line
                .starts_with("fault arm cancel-demote did not prove its contract")
        );
        assert!(!v.line.contains("FAULT-ARM PASS") && !v.line.starts_with("REFUSED"));
        // cancel-restore: the revocation was not granted, so the hold never existed.
        let mut checks = green(Arm::CancelRestore);
        checks[0] = Check::new("cancel", "Ok(PublicationRevoked)", "Ok(AlreadyPublished)");
        assert!(!verdict(Arm::CancelRestore, &checks).pass);
        // corrupt-host: the restore accepted the flipped byte.
        let mut checks = green(Arm::CorruptHost);
        checks[2] = Check::new("restore-integrity", "Corrupt", "Ok(())");
        assert!(!verdict(Arm::CorruptHost, &checks).pass);
        // missing-host: the removed copy was still takeable.
        let mut checks = green(Arm::MissingHost);
        checks[3] = Check::new("retake-removed-copy", "Err(UnknownTicket)", "Ok(Host(..))");
        assert!(!verdict(Arm::MissingHost, &checks).pass);
        // host-budget-short: the governor admitted the whole state.
        let mut checks = green(Arm::HostBudgetShort);
        checks[0] = Check::new("whole-state-admission", "Err(Capacity)", "Ok(())");
        assert!(!verdict(Arm::HostBudgetShort, &checks).pass);
        // device-short: the restore was admitted while the competitor held the budget.
        let mut checks = green(Arm::DeviceShort);
        checks[0] = Check::new("restore-admission", "Err(Capacity)", "Ok(lease)");
        assert!(!verdict(Arm::DeviceShort, &checks).pass);
        // require-resident: the register did not name the layers that left.
        let mut checks = green(Arm::RequireResident);
        checks[0] = Check::new("suspended-register", "[3, 4]", "[]");
        assert!(!verdict(Arm::RequireResident, &checks).pass);
    }

    #[test]
    fn a_missing_required_check_or_a_differing_restore_never_passes() {
        for arm in Arm::ALL {
            for drop in arm.required_checks() {
                let checks: Vec<Check> =
                    green(arm).into_iter().filter(|c| c.name != *drop).collect();
                let v = verdict(arm, &checks);
                assert!(!v.pass, "{arm:?} without {drop}");
                assert_eq!(v.missing, [*drop]);
                assert!(!v.line.contains("FAULT-ARM PASS"));
            }
        }
        for arm in Arm::ALL.into_iter().filter(|a| a.restores_cache()) {
            let mut checks = green(arm);
            let i = checks
                .iter()
                .position(|c| c.name == "restored-identical")
                .unwrap();
            checks[i] = Check::new("restored-identical", "true", "false");
            assert!(!verdict(arm, &checks).pass, "{arm:?}");
        }
        // Empty evidence passes nothing.
        for arm in Arm::ALL {
            assert!(!verdict(arm, &[]).pass);
        }
    }

    #[test]
    fn extra_checks_must_also_hold_and_the_tsv_is_flat() {
        let mut checks = green(Arm::CancelDemote);
        checks.push(Check::new("extra", "0", "0"));
        assert!(verdict(Arm::CancelDemote, &checks).pass);
        checks.push(Check::new("extra-red", "0", "1\twith\ttabs"));
        let v = verdict(Arm::CancelDemote, &checks);
        assert!(!v.pass);
        assert_eq!(v.failed, ["extra-red"]);
        let tsv = checks_tsv(&checks);
        let rows: Vec<&str> = tsv.lines().collect();
        assert_eq!(rows[0], "check\texpected\tobserved\tok");
        assert_eq!(rows.len(), checks.len() + 1);
        assert!(rows.iter().all(|r| r.split('\t').count() == 4), "{tsv}");
        assert!(rows.last().unwrap().ends_with("\tfalse"));
    }
}
