//! Day 11 and day 12 (lane D) red arms for the fault door of `kv-tier-gate`: the CLI admits an
//! arm only on the active pooled roundtrip, and the pure verdict prints the PASS line only when
//! every check the arm requires was recorded and held. Where a committed target-card receipt
//! exists, the printed line is replayed from its `fault-checks.tsv`; these tests never produce a
//! cell. Day 12 moved `cancel-restore` and `require-resident` onto lane A's rule seams: their
//! day-11 receipts (typed refusals, no seam) are kept as the record of the finding and are
//! failed cells under the day-12 rule, never PASS.
use super::cli;
use super::fault_contract::{Arm, Check, checks_tsv, pass_line, verdict};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

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

/// The fault the arm injects, as the backend would answer if it had NOT been injected. For the
/// two seam arms this is the day-11 observation itself: a transport that drains the cancelled
/// restore instead of holding its source, a gate that accepts a suspended cache.
fn not_injected(arm: Arm) -> Check {
    match arm {
        Arm::CancelDemote => Check::new("cancel", "Ok(PublicationRevoked)", "Ok(AlreadyPublished)"),
        Arm::CancelRestore => Check::new("retire-holds-source", "Err(Busy)", "Ok(())"),
        Arm::CorruptHost => Check::new("restore-integrity", "Corrupt", "Ok(plane)"),
        Arm::MissingHost => Check::new(
            "retake-removed-copy",
            "Err(UnknownTicket)",
            "Ok(Host(CudaPinnedLease { bytes: 1 }))",
        ),
        Arm::HostBudgetShort => Check::new("whole-state-admission", "Err(Capacity)", "Ok(())"),
        Arm::DeviceShort => Check::new("restore-admission", "Err(Capacity)", "Ok(lease)"),
        Arm::RequireResident => Check::new(
            "continuation-gate-on-suspended-cache",
            "Err(ContinuationRefused { path: \"kv-tier-gate continuation\", layers: [3, 4] })",
            "Ok(())",
        ),
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

fn receipt(root: &str, arm: Arm) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../research/spill-d-20260919")
        .join(root)
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

/// The recorded rows of a committed receipt, or `None` when the receipt is not committed.
/// Required names resolve to their static spelling; any other recorded row is extra evidence
/// (the gate may add rows, e.g. `holed-cache-refuses-continuation`) and is kept, still required
/// to hold. `verdict` insists on the required set.
fn rows(receipt: &Path, arm: Arm) -> Option<Vec<Check>> {
    let tsv = std::fs::read_to_string(receipt.join("fault-checks.tsv")).ok()?;
    let mut lines = tsv.lines();
    assert_eq!(lines.next(), Some("check\texpected\tobserved\tok"));
    Some(
        lines
            .map(|row| {
                let f: Vec<&str> = row.split('\t').collect();
                assert_eq!(f.len(), 4, "{row}");
                let name = arm
                    .required_checks()
                    .iter()
                    .copied()
                    .find(|n| *n == f[0])
                    .unwrap_or_else(|| -> &'static str {
                        Box::leak(f[0].to_string().into_boxed_str())
                    });
                assert_eq!(f[3], (f[1] == f[2]).to_string(), "{row}");
                Check::new(name, f[1], f[2])
            })
            .collect(),
    )
}

/// Replays the committed target-card receipts under `root` for `arms`: the pure verdict over
/// the recorded checks must be the PASS line the gate printed and wrote, and the shape
/// (continues, restores) must match. Returns how many receipts were present.
fn replay_green(root: &str, arms: &[Arm]) -> usize {
    let mut replayed = 0;
    for arm in arms.iter().copied() {
        let receipt = receipt(root, arm);
        let Some(checks) = rows(&receipt, arm) else {
            continue;
        };
        let v = verdict(arm, &checks);
        let recorded = fields(receipt.join("FAULT-ARM.txt"));
        assert_eq!(recorded["arm"], arm.name());
        assert_eq!(recorded["line"], v.line, "{root} {arm:?}");
        assert_eq!(recorded["continues"], arm.continues().to_string());
        assert_eq!(recorded["restores_cache"], arm.restores_cache().to_string());
        assert!(v.pass, "{root} {arm:?}: {}", v.line);
        assert_eq!(recorded["verdict"], "PASS", "{root} {arm:?}");
        assert_eq!(v.line, pass_line(arm));
        assert!(!recorded.contains_key("finding"), "{root} {arm:?}");
        // A continuing arm wrote the continuation; an incomplete arm wrote no token.
        assert_eq!(
            receipt.join("tokens.u32le").exists(),
            arm.continues(),
            "{root} {arm:?}"
        );
        assert_eq!(
            receipt.join("ACTIVE.txt").exists(),
            arm.continues(),
            "{root} {arm:?}"
        );
        assert_eq!(
            receipt.join("restored-prefix-state.tsv").exists(),
            arm.restores_cache(),
            "{root} {arm:?}"
        );
        if !arm.restores_cache() {
            // The day-11 hole is a typed refusal on the cache (revuto finding on #584).
            let holed = checks
                .iter()
                .find(|c| c.name == "holed-cache-refuses-continuation");
            assert!(
                holed.is_none_or(|c| c.observed == "Err"),
                "{root} {arm:?}: holed cache did not refuse"
            );
        }
        replayed += 1;
    }
    replayed
}

/// The five arms whose vocabulary did not move on day 12 keep replaying from their day-11 cells.
const DAY11_GREEN: [Arm; 5] = [
    Arm::CancelDemote,
    Arm::CorruptHost,
    Arm::MissingHost,
    Arm::HostBudgetShort,
    Arm::DeviceShort,
];

#[test]
fn committed_day11_receipts_replay_their_pass_lines() {
    // The receipts are tracked. Their absence is a failure, never a silent pass (#545: a test
    // that skips for a missing artifact must say so, and these have no reason to skip).
    let replayed = replay_green("pro-single-day11", &DAY11_GREEN);
    assert_eq!(
        replayed,
        DAY11_GREEN.len(),
        "committed day-11 receipts are incomplete"
    );
}

/// Day 12: every arm through the seams. Once the day-12 cells are committed all seven receipts
/// must be present and green, the two seam arms included.
#[test]
fn committed_day12_receipts_replay_their_pass_lines() {
    let replayed = replay_green("pro-single-day12", &Arm::ALL);
    assert_eq!(
        replayed,
        Arm::ALL.len(),
        "committed day-12 receipts are incomplete"
    );
    for arm in [Arm::CancelRestore, Arm::RequireResident] {
        let checks = rows(&receipt("pro-single-day12", arm), arm).unwrap();
        // The seam rows were recorded and held, not merely required.
        for name in [
            "recover-source",
            "recovered-source-checksum",
            "retire-holds-source",
        ] {
            assert!(
                arm != Arm::CancelRestore || checks.iter().any(|c| c.name == name && c.ok()),
                "{arm:?}: {name}"
            );
        }
        for name in [
            "continuation-gate-on-suspended-cache",
            "continuation-gate-after-partial-resume",
            "continuation-gate-after-resume",
        ] {
            assert!(
                arm != Arm::RequireResident || checks.iter().any(|c| c.name == name && c.ok()),
                "{arm:?}: {name}"
            );
        }
    }
}

/// Red arm, kept as the record of the finding. The day-11 cells of the two seam arms ended in
/// these typed refusals (verbatim) because the contracts had no seam; every row they recorded
/// held. Under the day-12 rule the same rows are a failed cell with the seam rows missing:
/// the absence of a seam is a failure the receipt names, never a refusal and never PASS.
#[test]
fn day11_refusal_receipts_are_failed_cells_under_the_day12_rule() {
    const DAY11: [(Arm, &str, &str); 2] = [
        (
            Arm::CancelRestore,
            "REFUSED: cancel-restore revoked publication, but the transfer contract has no seam to recover the H2D source after cancellation; no tokens, budget drained",
            "recover-source",
        ),
        (
            Arm::RequireResident,
            "REFUSED: require-resident has no contract today: Cache::ensure_usable accepts a suspended cache, decode_step_h unwraps a suspended layer, and tier RestoreDecision::RequireState is a load-versus-recompute rule",
            "continuation-gate-on-suspended-cache",
        ),
    ];
    for (arm, refusal, seam_row) in DAY11 {
        let receipt = receipt("pro-single-day11", arm);
        let Some(checks) = rows(&receipt, arm) else {
            continue;
        };
        let v = verdict(arm, &checks);
        assert!(!v.pass, "{arm:?}");
        assert!(
            v.failed.is_empty(),
            "{arm:?}: every day-11 row held: {:?}",
            v.failed
        );
        assert!(v.missing.contains(&seam_row), "{arm:?}: {:?}", v.missing);
        assert!(v.line.starts_with(&format!(
            "fault arm {} did not prove its contract",
            arm.name()
        )));
        assert!(!v.line.starts_with("REFUSED") && !v.line.contains("PASS"));
        let recorded = fields(receipt.join("FAULT-ARM.txt"));
        assert_eq!(recorded["verdict"], "REFUSED");
        assert_eq!(recorded["line"], refusal);
        assert_eq!(recorded["finding"], refusal);
        assert_eq!(
            std::fs::read_to_string(receipt.join("REFUSED.txt"))
                .unwrap()
                .trim_end(),
            refusal
        );
        // No token was written on either day-11 refusal.
        assert!(!receipt.join("tokens.u32le").exists());
        assert!(!receipt.join("ACTIVE.txt").exists());
        // The day-11 observation that became rule 2, verbatim from the receipt.
        if arm == Arm::RequireResident {
            assert_eq!(
                recorded["observation.continuation_gate_on_suspended_cache"],
                "Ok(())"
            );
        }
    }
}
