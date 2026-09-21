//! Day 11 (lane D) red arms for the fault door of `kv-tier-gate`: the CLI admits an arm only on
//! the active pooled roundtrip, and the pure verdict prints the PASS line only when every check
//! the arm requires was recorded and held. Where a committed target-card receipt exists, the
//! printed line is replayed from its `fault-checks.tsv`; these tests never produce a cell.
use super::cli;
use super::fault_contract::{Arm, Check, checks_tsv, pass_line, verdict};
use std::collections::BTreeMap;
use std::path::PathBuf;

fn active(extra: &[&str]) -> Vec<String> {
    let mut args: Vec<String> =
        "--artifact fixture.gguf --case active --context 8192 --tiers host --same-program --out receipts"
            .split_whitespace()
            .map(str::to_owned)
            .collect();
    args.extend(extra.iter().map(|s| (*s).to_owned()));
    args
}

#[test]
fn every_arm_is_a_door_on_the_active_pooled_roundtrip_only() {
    for arm in Arm::ALL {
        let parsed = cli::parse(active(&["--fault", arm.name()])).unwrap();
        assert_eq!(parsed.fault, Some(arm));
        assert_eq!(parsed.kv_allocator, cli::KvAllocator::Pooled);
        // Baseline and prefix are usage errors (failed cells), never refusals.
        for case in ["baseline", "prefix"] {
            let mut args = active(&["--fault", arm.name()]);
            args[3] = case.into();
            let error = cli::parse(args).unwrap_err();
            assert!(
                error.starts_with("--fault requires --case active"),
                "{error}"
            );
        }
        // The VMM door and the reclaim diagnostic refuse the arm explicitly.
        let error =
            cli::parse(active(&["--fault", arm.name(), "--kv-allocator", "vmm"])).unwrap_err();
        assert!(error.starts_with("REFUSED: fault arms are bound to the pooled allocator"));
        let error = cli::parse(active(&[
            "--fault",
            arm.name(),
            "--kv-allocator",
            "vmm",
            "--reclaim-diagnostic",
        ]))
        .unwrap_err();
        assert_eq!(
            error,
            "REFUSED: fault arms do not combine with the reclaim diagnostic"
        );
    }
    let error = cli::parse(active(&["--fault", "cancel-everything"])).unwrap_err();
    assert!(error.starts_with("REFUSED: unknown fault arm"), "{error}");
    assert!(!error.contains("cancel-everything"));
    // The diagnostic wrapper leaves refusals bare and prefixes usage errors.
    assert!(cli::diagnostic(&error).starts_with("REFUSED:"));
    assert!(cli::diagnostic("--fault requires --case active").starts_with("kv-tier-gate: "));
}

/// The fault the arm injects, as the backend would answer if it had NOT been injected.
fn not_injected(arm: Arm) -> Check {
    match arm {
        Arm::CancelDemote | Arm::CancelRestore => {
            Check::new("cancel", "Ok(PublicationRevoked)", "Ok(AlreadyPublished)")
        }
        Arm::CorruptHost => Check::new("restore-integrity", "Corrupt", "Ok(plane)"),
        Arm::MissingHost => Check::new(
            "retake-removed-copy",
            "Err(UnknownTicket)",
            "Ok(Host(CudaPinnedLease { bytes: 1 }))",
        ),
        Arm::HostBudgetShort => Check::new("whole-state-admission", "Err(Capacity)", "Ok(())"),
        Arm::DeviceShort => Check::new("restore-admission", "Err(Capacity)", "Ok(lease)"),
        Arm::RequireResident => Check::new("restored-identical", "true", "false"),
    }
}

#[test]
fn red_arm_without_its_fault_never_prints_the_pass_line() {
    for arm in Arm::ALL {
        let red = not_injected(arm);
        assert!(
            arm.required_checks().contains(&red.name),
            "{arm:?}: {} is not required",
            red.name
        );
        let checks: Vec<Check> = arm
            .required_checks()
            .iter()
            .map(|name| {
                if *name == red.name {
                    red.clone()
                } else {
                    Check::new(name, "held", "held")
                }
            })
            .collect();
        let v = verdict(arm, &checks);
        assert!(!v.pass, "{arm:?}");
        assert_eq!(v.failed, [red.name]);
        assert!(!v.line.contains("FAULT-ARM PASS"), "{arm:?}: {}", v.line);
        assert!(!v.line.starts_with("REFUSED"), "{arm:?}: {}", v.line);
        // The receipt keeps the red row, so a reader can see what the backend said.
        let tsv = checks_tsv(&checks);
        assert!(tsv.contains(&format!(
            "{}\t{}\t{}\tfalse",
            red.name, red.expected, red.observed
        )));
    }
}

fn day11(arm: Arm) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../research/spill-d-20260919/pro-single-day11")
        .join(arm.name())
        .join("receipt")
}

fn fields(path: PathBuf) -> BTreeMap<String, String> {
    std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("{}: {e}", path.display()))
        .lines()
        .filter_map(|line| line.split_once('='))
        .map(|(k, v)| (k.to_owned(), v.to_owned()))
        .collect()
}

/// Replays the committed target-card receipts: the pure verdict over the recorded checks must
/// be the line the gate printed and wrote, and the shape (continues, restores) must match.
#[test]
fn committed_target_card_receipts_replay_their_verdicts() {
    let mut replayed = 0;
    for arm in Arm::ALL {
        let receipt = day11(arm);
        let Ok(tsv) = std::fs::read_to_string(receipt.join("fault-checks.tsv")) else {
            continue;
        };
        let mut rows = tsv.lines();
        assert_eq!(rows.next(), Some("check\texpected\tobserved\tok"));
        let checks: Vec<Check> = rows
            .map(|row| {
                let f: Vec<&str> = row.split('\t').collect();
                assert_eq!(f.len(), 4, "{row}");
                let name = arm
                    .required_checks()
                    .iter()
                    .copied()
                    .find(|n| *n == f[0])
                    .unwrap_or_else(|| panic!("{arm:?}: unexpected check {}", f[0]));
                assert_eq!(f[3], (f[1] == f[2]).to_string(), "{row}");
                Check::new(name, f[1], f[2])
            })
            .collect();
        let v = verdict(arm, &checks);
        let recorded = fields(receipt.join("FAULT-ARM.txt"));
        assert_eq!(recorded["arm"], arm.name());
        assert_eq!(recorded["line"], v.line, "{arm:?}");
        assert_eq!(recorded["continues"], arm.continues().to_string());
        assert_eq!(recorded["restores_cache"], arm.restores_cache().to_string());
        let outcome = match (v.pass, arm.refusal()) {
            (true, _) => "PASS",
            (false, Some(_)) if v.failed.is_empty() && v.missing.is_empty() => "REFUSED",
            _ => "FAILED",
        };
        assert_eq!(recorded["verdict"], outcome, "{arm:?}");
        if v.pass {
            assert_eq!(v.line, pass_line(arm));
        } else {
            assert_eq!(Some(v.line.as_str()), arm.refusal(), "{arm:?}: {}", v.line);
        }
        // A continuing arm wrote the continuation; an incomplete arm wrote no token.
        assert_eq!(
            receipt.join("tokens.u32le").exists(),
            arm.continues(),
            "{arm:?}"
        );
        assert_eq!(
            receipt.join("ACTIVE.txt").exists(),
            arm.continues(),
            "{arm:?}"
        );
        assert_eq!(
            receipt.join("restored-prefix-state.tsv").exists(),
            arm.restores_cache(),
            "{arm:?}"
        );
        replayed += 1;
    }
    // Once the day-11 cells are committed all seven receipts must be present.
    if day11(Arm::CancelDemote).exists() {
        assert_eq!(
            replayed,
            Arm::ALL.len(),
            "committed day-11 receipts are incomplete"
        );
    }
}
