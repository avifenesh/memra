//! Pure fault-arm contract for `kv-tier-gate --fault <arm>`; `rustc --test` exercises it
//! without CUDA. An arm passes only when every check the arm requires was recorded and every
//! recorded check observed exactly what the frozen tier contract promises at the documented
//! call. A red arm (fault not injected, a contract answer that differs, or a check that was
//! never recorded) never prints the PASS line. Two arms have no seam in the contracts as they
//! are; their checks must still hold, and their outcome is a typed refusal, never PASS.

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
    /// the manifest hash with the suspended prefix (`restored-identical`).
    pub fn restores_cache(self) -> bool {
        !matches!(
            self,
            Arm::CancelRestore | Arm::CorruptHost | Arm::MissingHost
        )
    }
    /// A passing arm whose state is resident and bit-identical lets the same tokenwise
    /// continuation run; the offline verifier compares it with the frozen baseline bundle.
    pub fn continues(self) -> bool {
        self.restores_cache() && self.refusal().is_none()
    }
    /// Arms whose expectation has no seam in the frozen contracts. The typed refusal names
    /// the missing seam; recording it for the lead is the arm's deliverable.
    pub fn refusal(self) -> Option<&'static str> {
        match self {
            Arm::CancelRestore => Some(
                "REFUSED: cancel-restore revoked publication, but the transfer contract has no seam to recover the H2D source after cancellation; no tokens, budget drained",
            ),
            Arm::RequireResident => Some(
                "REFUSED: require-resident has no contract today: Cache::ensure_usable accepts a suspended cache, decode_step_h unwraps a suspended layer, and tier RestoreDecision::RequireState is a load-versus-recompute rule",
            ),
            _ => None,
        }
    }
    /// Every name here must be recorded, in any order, for the arm to pass.
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
            Arm::CancelRestore => &[
                "cancel",
                "ready-view-after-cancel",
                "with-destination-after-cancel",
                "take-after-cancel",
                "retire",
                "acknowledge",
                "retake-demoted-copy",
                "budget-zero",
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
            Arm::RequireResident => &["budget-zero", "restored-identical"],
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Verdict {
    /// `true` only for `FAULT-ARM PASS <name>`; a refusal and a failure are both `false`.
    pub pass: bool,
    /// The final console line: the PASS line, the arm's typed refusal, or the failure text
    /// (which `cli::diagnostic` prefixes, so the collector classifies a failed cell).
    pub line: String,
    pub missing: Vec<&'static str>,
    pub failed: Vec<&'static str>,
}

pub fn pass_line(arm: Arm) -> String {
    format!("FAULT-ARM PASS {}", arm.name())
}

/// A failed or missing check is a failed cell even for a refusal arm: the refusal states what
/// the contract lacks, and it is only meaningful when everything the contract does promise held.
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
    if let Some(refusal) = arm.refusal() {
        return Verdict {
            pass: false,
            line: refusal.to_owned(),
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
    fn only_the_five_bound_arms_can_pass_and_three_continue() {
        let passing: Vec<Arm> = Arm::ALL
            .into_iter()
            .filter(|arm| verdict(*arm, &green(*arm)).pass)
            .collect();
        assert_eq!(
            passing,
            [
                Arm::CancelDemote,
                Arm::CorruptHost,
                Arm::MissingHost,
                Arm::HostBudgetShort,
                Arm::DeviceShort
            ]
        );
        let continuing: Vec<Arm> = Arm::ALL.into_iter().filter(|a| a.continues()).collect();
        assert_eq!(
            continuing,
            [Arm::CancelDemote, Arm::HostBudgetShort, Arm::DeviceShort]
        );
        for arm in passing {
            let v = verdict(arm, &green(arm));
            assert_eq!(v.line, format!("FAULT-ARM PASS {}", arm.name()));
            assert!(v.missing.is_empty() && v.failed.is_empty());
            assert!(arm.refusal().is_none());
        }
        // A continuing arm always restores the cache; the incomplete arms never continue.
        for arm in Arm::ALL {
            assert!(!arm.continues() || arm.restores_cache(), "{arm:?}");
            assert_eq!(
                arm.restores_cache(),
                !matches!(
                    arm,
                    Arm::CancelRestore | Arm::CorruptHost | Arm::MissingHost
                )
            );
        }
    }

    #[test]
    fn refusal_arms_never_print_pass_even_when_every_check_holds() {
        for arm in [Arm::CancelRestore, Arm::RequireResident] {
            let v = verdict(arm, &green(arm));
            assert!(!v.pass);
            assert!(v.line.starts_with("REFUSED: "), "{}", v.line);
            assert_eq!(Some(v.line.as_str()), arm.refusal());
            assert!(!v.line.contains('\n'));
            assert!(!v.line.contains("PASS"));
        }
    }

    #[test]
    fn red_arm_fault_not_injected_is_a_failure_not_a_pass_and_not_a_refusal() {
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
        // A refusal arm whose contract half failed is a failed cell, not the typed refusal.
        let mut checks = green(Arm::CancelRestore);
        checks[0] = Check::new("cancel", "Ok(PublicationRevoked)", "Ok(AlreadyPublished)");
        let v = verdict(Arm::CancelRestore, &checks);
        assert!(!v.pass && !v.line.starts_with("REFUSED"));
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
