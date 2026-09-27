//! Day-ten CPU cells for the experts-via-tier budget door: the pure GPU slot budget,
//! the typed refusal, and the gate binaries' argv parse. No CUDA, no Engine.
use super::engine_bridge::{
    ExpertBankBudget, ExpertBankCli, ExpertBankMode, ExpertBankRefusal, Qualification,
    QualifiedArtifact, SLOT_TAIL_PAD_BYTES, expert_bank_cli, expert_bank_rollback, gpu_bank_budget,
    gpu_bank_slots, gpu_slot_bytes, host_bank_budget, refusal_reason,
};

const RECORD: u64 = 860_160;
const SLOT: u64 = RECORD + SLOT_TAIL_PAD_BYTES as u64;
const MINIMUM: u64 = 8 * SLOT;
const HARD: u64 = 64 * 1024 * 1024 * 1024;

#[test]
fn gpu_bank_slots_refuses_instead_of_clamping() {
    let below_minimum = Err("experts-via-tier GPU bank budget cannot hold the eight-slot minimum");
    let ceiling = Err("experts-via-tier GPU bank budget exceeds the hard VRAM ceiling");
    assert_eq!(gpu_slot_bytes(RECORD), Some(SLOT));
    assert_eq!(
        gpu_bank_slots(MINIMUM, 0, HARD),
        Err("experts-via-tier GPU bank budget has no expert record")
    );
    // Below one record, below one slot, seven slots exactly, one byte short of eight.
    assert_eq!(gpu_bank_slots(0, RECORD, HARD), below_minimum);
    assert_eq!(gpu_bank_slots(RECORD, RECORD, HARD), below_minimum);
    assert_eq!(gpu_bank_slots(7 * SLOT, RECORD, HARD), below_minimum);
    assert_eq!(gpu_bank_slots(MINIMUM - 1, RECORD, HARD), below_minimum);
    // Exact boundaries: the minimum, one byte over, one slot more.
    assert_eq!(gpu_bank_slots(MINIMUM, RECORD, HARD), Ok(8));
    assert_eq!(gpu_bank_slots(MINIMUM + 1, RECORD, HARD), Ok(8));
    assert_eq!(gpu_bank_slots(MINIMUM + SLOT, RECORD, HARD), Ok(9));
    // Ceiling: the ceiling itself is allowed, one byte over is refused, u64::MAX never saturates.
    assert_eq!(
        gpu_bank_slots(HARD, RECORD, HARD),
        Ok((HARD / SLOT) as usize)
    );
    assert_eq!(gpu_bank_slots(HARD + 1, RECORD, HARD), ceiling);
    assert_eq!(gpu_bank_slots(u64::MAX, RECORD, HARD), ceiling);
    // A ceiling below the minimum cannot be satisfied from either side.
    assert_eq!(
        gpu_bank_slots(MINIMUM - 1, RECORD, MINIMUM - 1),
        below_minimum
    );
    assert_eq!(gpu_bank_slots(MINIMUM, RECORD, MINIMUM - 1), ceiling);
    assert_eq!(gpu_bank_slots(MINIMUM, RECORD, MINIMUM), Ok(8));
    // Checked arithmetic: an absurd record overflows the slot or the minimum, never wraps.
    assert_eq!(
        gpu_bank_slots(u64::MAX, u64::MAX, u64::MAX),
        Err("experts-via-tier GPU bank budget arithmetic overflow")
    );
    assert_eq!(
        gpu_bank_slots(u64::MAX, u64::MAX / 4, u64::MAX),
        Err("experts-via-tier GPU bank budget arithmetic overflow")
    );
}

#[test]
fn typed_refusals_name_requested_minimum_and_ceiling() {
    assert_eq!(gpu_bank_budget(MINIMUM, RECORD, HARD), Ok(8));
    let seven = gpu_bank_budget(7 * SLOT, RECORD, HARD).unwrap_err();
    assert_eq!(
        seven.0,
        format!(
            "experts-via-tier GPU bank budget cannot hold the eight-slot minimum \
             (requested {}, minimum {MINIMUM}, ceiling {HARD})",
            7 * SLOT
        )
    );
    let over = gpu_bank_budget(HARD + 1, RECORD, HARD).unwrap_err();
    assert!(
        over.0
            .starts_with("experts-via-tier GPU bank budget exceeds the hard VRAM ceiling (")
    );
    assert!(
        over.0
            .ends_with(&format!("minimum {MINIMUM}, ceiling {HARD})"))
    );
    assert!(
        gpu_bank_budget(u64::MAX, u64::MAX, u64::MAX)
            .unwrap_err()
            .0
            .contains("minimum overflow")
    );
    // Day 43: the host plan's typed refusals carry the same suffix, with the machine ceiling.
    const CEILING: u64 = 1 << 34;
    let sizes = [RECORD; 16];
    assert_eq!(
        host_bank_budget(16 * RECORD, &sizes, CEILING).map(|plan| plan.records_held),
        Ok(16)
    );
    let host = host_bank_budget(1, &sizes, CEILING).unwrap_err();
    assert_eq!(
        host.0,
        format!(
            "experts-via-tier host bank budget cannot hold one expert record \
             (requested 1, minimum {RECORD}, ceiling {CEILING})"
        )
    );
    let over = host_bank_budget(CEILING + 1, &sizes, CEILING).unwrap_err();
    assert_eq!(
        over.0,
        format!(
            "experts-via-tier host bank budget exceeds the machine ceiling \
             (requested {}, minimum {RECORD}, ceiling {CEILING})",
            CEILING + 1
        )
    );
    // Only the typed refusal maps to the REFUSED contract; a plain error never does.
    let typed: Box<dyn std::error::Error> = Box::new(seven.clone());
    assert_eq!(refusal_reason(typed.as_ref()), Some(seven.0.as_str()));
    let plain: Box<dyn std::error::Error> = "experts-via-tier artifact SHA256 mismatch".into();
    assert_eq!(refusal_reason(plain.as_ref()), None);
    let wrapped: Box<dyn std::error::Error> = Box::new(ExpertBankRefusal(seven.0.clone()));
    assert_eq!(refusal_reason(wrapped.as_ref()), Some(seven.0.as_str()));
}

fn argv(args: &[&str]) -> Vec<String> {
    let mut v = vec!["run-gen".to_owned(), "model.gguf".to_owned()];
    v.extend(args.iter().map(|a| (*a).to_owned()));
    v
}

fn cli(mode: ExpertBankMode, budget: ExpertBankBudget, door_flags: bool) -> ExpertBankCli {
    ExpertBankCli {
        mode,
        budget,
        door_flags,
    }
}

/// Day 88 (`research/spill-c-20260919/DAY88.md`): the door is the default, so its budgets parse
/// without `--experts-via-tier` (which asks the door to install or refuse), the host budget's
/// default is the whole bank, and every malformed, repeated or look-alike key is a usage error.
#[test]
fn expert_bank_cli_parses_the_default_door_and_its_budgets() {
    use ExpertBankMode::{Default as Naked, Required};
    let default = ExpertBankBudget::default();
    assert_eq!(default.host_bytes, None);
    assert_eq!(default.gpu_bytes, None);
    assert!(!default.pool_allocated && !default.stage_clock);
    assert_eq!(
        expert_bank_cli(argv(&["55", "88", "13"])),
        Ok(cli(Naked, default, false))
    );
    assert_eq!(
        expert_bank_cli(argv(&["55", "88", "13", "--experts-via-tier"])),
        Ok(cli(Required, default, false))
    );
    assert_eq!(
        expert_bank_cli(argv(&["--experts-via-tier", "--expert-bank-host-bytes=1"])),
        Ok(cli(
            Required,
            ExpertBankBudget {
                host_bytes: Some(1),
                ..default
            },
            true
        ))
    );
    // Budgets without the door flag are the default door's budgets.
    assert_eq!(
        expert_bank_cli(argv(&["--expert-bank-gpu-bytes=6881344", "55"])),
        Ok(cli(
            Naked,
            ExpertBankBudget {
                gpu_bytes: Some(6_881_344),
                ..default
            },
            true
        ))
    );
    assert_eq!(
        expert_bank_cli(argv(&[
            "--experts-via-tier",
            "--expert-bank-host-bytes=268435456",
            "--expert-bank-gpu-bytes=6021176"
        ])),
        Ok(cli(
            Required,
            ExpertBankBudget {
                host_bytes: Some(268_435_456),
                gpu_bytes: Some(6_021_176),
                ..default
            },
            true
        ))
    );
    // Usage errors, never silent: bare flag, junk, negative, repeat.
    assert_eq!(
        expert_bank_cli(argv(&["--expert-bank-gpu-bytes"])),
        Err("--expert-bank-gpu-bytes expects --expert-bank-gpu-bytes=<bytes>".to_owned())
    );
    assert_eq!(
        expert_bank_cli(argv(&["--expert-bank-host-bytes=8MiB"])),
        Err("--expert-bank-host-bytes expects an unsigned byte count, got \"8MiB\"".to_owned())
    );
    assert!(expert_bank_cli(argv(&["--expert-bank-gpu-bytes=-1"])).is_err());
    assert!(expert_bank_cli(argv(&["--expert-bank-gpu-bytes="])).is_err());
    assert_eq!(
        expert_bank_cli(argv(&[
            "--expert-bank-gpu-bytes=1",
            "--expert-bank-gpu-bytes=2"
        ])),
        Err("--expert-bank-gpu-bytes given more than once".to_owned())
    );
    // A key that merely starts with a flag name is a usage error, never a silent ignore and never
    // the flag it resembles; the retired pool flags are unknown now.
    for junk in [
        "--expert-bank-gpu-bytesx=1",
        "--expert-bank-host-bytes-x=1",
        "--expert-bank-gpu-byte=1",
        "--expert-bank-gpu-bytes-",
        "--experts-via-tier=1",
        "--experts-via-tiers",
        "--expert-bank-pool-registered",
        "--expert-bank-pool-pageable",
        "--expert-bank-pool-chunk-bytes=268435456",
        "--expert-bank-pool-allocatedx",
    ] {
        let err = expert_bank_cli(argv(&[junk])).unwrap_err();
        assert!(
            err.contains("unknown expert bank flag") || err == "--experts-via-tier takes no value",
            "{junk}: {err}"
        );
        assert!(!err.contains("given more than once"), "{junk}: {err}");
    }
    assert_eq!(
        expert_bank_cli(argv(&["--expert-bank-host-bytes-x=1"])),
        Err(
            "unknown expert bank flag \"--expert-bank-host-bytes-x\"; expected \
             --experts-via-tier, --expert-bank-host-bytes=<bytes>, --expert-bank-gpu-bytes=<bytes>, \
             --expert-bank-pool-allocated or --expert-bank-stages"
                .to_owned()
        )
    );
    // The slot pad is shared.
    assert_eq!(SLOT_TAIL_PAD_BYTES, 8);
    assert_eq!(gpu_slot_bytes(RECORD), Some(RECORD + 8));
}

/// Day 88: `--expert-bank-pool-allocated` (the pool before the promotion, the rollback seam)
/// takes no value, refuses a repeat, and is a door flag.
#[test]
fn expert_bank_pool_allocated_is_the_pool_rollback() {
    assert_eq!(
        expert_bank_cli(argv(&["--expert-bank-pool-allocated"])),
        Ok(cli(
            ExpertBankMode::Default,
            ExpertBankBudget {
                pool_allocated: true,
                ..ExpertBankBudget::default()
            },
            true
        ))
    );
    assert_eq!(
        expert_bank_cli(argv(&["--expert-bank-pool-allocated=1"])),
        Err("--expert-bank-pool-allocated takes no value".to_owned())
    );
    assert_eq!(
        expert_bank_cli(argv(&[
            "--expert-bank-pool-allocated",
            "--expert-bank-pool-allocated"
        ])),
        Err("--expert-bank-pool-allocated given more than once".to_owned())
    );
}

/// Day 88: `MEMRA_EXPERTS_VIA_TIER=0` is the rollback; with `--experts-via-tier` or a door flag it is
/// a usage error; any other value, or none, keeps the default.
#[test]
fn the_rollback_seam_refuses_to_name_two_programs() {
    let naked = expert_bank_cli(argv(&[])).unwrap();
    assert_eq!(expert_bank_rollback(&naked, None), Ok(false));
    assert_eq!(expert_bank_rollback(&naked, Some("1")), Ok(false));
    assert_eq!(expert_bank_rollback(&naked, Some("")), Ok(false));
    assert_eq!(expert_bank_rollback(&naked, Some("0")), Ok(true));
    let required = expert_bank_cli(argv(&["--experts-via-tier"])).unwrap();
    assert_eq!(
        expert_bank_rollback(&required, Some("0")),
        Err("--experts-via-tier with MEMRA_EXPERTS_VIA_TIER=0 names two programs".to_owned())
    );
    assert_eq!(expert_bank_rollback(&required, None), Ok(false));
    for flags in [
        &["--expert-bank-host-bytes=1"][..],
        &["--expert-bank-pool-allocated"][..],
        &["--expert-bank-stages"][..],
        &["--expert-bank-gpu-bytes=6881344"][..],
    ] {
        let parsed = expert_bank_cli(argv(flags)).unwrap();
        assert!(parsed.door_flags, "{flags:?}");
        assert_eq!(
            expert_bank_rollback(&parsed, Some("0")),
            Err(
                "expert bank flags need the door, and MEMRA_EXPERTS_VIA_TIER=0 turns it off"
                    .to_owned()
            )
        );
    }
}

/// Day 88: the qualification types carry the digest as hex and compare by value.
#[test]
fn a_qualified_artifact_prints_its_digest() {
    let artifact = QualifiedArtifact {
        digest: [0xdf; 32],
        dev: 1,
        ino: 2,
        ceiling: 3,
        whole_bank: 4,
        fits: true,
    };
    assert_eq!(artifact.sha256_hex(), "df".repeat(32));
    assert_ne!(
        Qualification::Qualified(artifact),
        Qualification::Off("not a qualified artifact".to_owned())
    );
}

/// Day 40: `--expert-bank-stages` is the door's log-only stage clock. It takes no value, refuses a
/// repeat and a look-alike key, and leaves the budgets exactly as they were; day 88: it is a door flag.
#[test]
fn expert_bank_stage_clock_parses_as_a_door_flag() {
    assert_eq!(
        expert_bank_cli(argv(&["--experts-via-tier", "--expert-bank-stages"])),
        Ok(cli(
            ExpertBankMode::Required,
            ExpertBankBudget {
                stage_clock: true,
                ..ExpertBankBudget::default()
            },
            true
        ))
    );
    assert_eq!(
        expert_bank_cli(argv(&[
            "--expert-bank-stages",
            "--expert-bank-gpu-bytes=6881344"
        ])),
        Ok(cli(
            ExpertBankMode::Default,
            ExpertBankBudget {
                gpu_bytes: Some(6_881_344),
                stage_clock: true,
                ..ExpertBankBudget::default()
            },
            true
        ))
    );
    assert!(!ExpertBankBudget::default().stage_clock);
    assert_eq!(
        expert_bank_cli(argv(&["--expert-bank-stages=1"])),
        Err("--expert-bank-stages takes no value".to_owned())
    );
    assert_eq!(
        expert_bank_cli(argv(&["--expert-bank-stages", "--expert-bank-stages"])),
        Err("--expert-bank-stages given more than once".to_owned())
    );
    for junk in [
        "--expert-bank-stage",
        "--expert-bank-stagesx",
        "--expert-bank-stages-",
    ] {
        let err = expert_bank_cli(argv(&[junk])).unwrap_err();
        assert!(err.contains("unknown expert bank flag"), "{junk}: {err}");
    }
}
