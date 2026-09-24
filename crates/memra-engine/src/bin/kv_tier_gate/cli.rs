//! Pure argument contract; `rustc --test` can exercise this without CUDA.
use std::path::PathBuf;

#[derive(Debug, PartialEq, Eq)]
pub enum KvAllocator {
    Pooled,
    Vmm,
    /// WP-B day 37: on-demand VMM planes (`memra_kv::with_on_demand_kv`), `--case grow` only.
    VmmOnDemand,
}

#[derive(Debug, PartialEq, Eq)]
pub struct Args {
    pub artifact: PathBuf,
    pub kv_allocator: KvAllocator,
    pub reclaim_diagnostic: bool,
    /// Day 11: repeat the demote/restore program N >= 2 times in one process (diagnostic only).
    pub reclaim_cycles: Option<usize>,
    /// Day 11 (lane D): inject exactly one fault at a documented contract call of the active
    /// roundtrip. Pooled allocator only; usage error unless `--case active`.
    pub fault: Option<super::fault_contract::Arm>,
    pub case: String,
    pub context: usize,
    pub tiers: String,
    pub out: PathBuf,
}

pub const USAGE: &str = "kv-tier-gate --artifact <gguf> --case baseline|active|prefix|grow --context 8192|32768 --tiers host|host,nvme --same-program [--kv-allocator pooled|vmm|vmm-ondemand] [--reclaim-diagnostic [--reclaim-cycles N]] [--fault cancel-demote|cancel-restore|corrupt-host|missing-host|host-budget-short|device-short|require-resident] --out <new-directory>";
const CYCLES_REFUSAL: &str = "REFUSED: --reclaim-cycles requires an integer count >= 2";

pub fn parse(args: impl IntoIterator<Item = String>) -> Result<Args, String> {
    let mut iter = args.into_iter();
    let mut fields = std::collections::BTreeMap::new();
    let mut same = false;
    let mut reclaim_diagnostic = false;
    let mut reclaim_cycles = None;
    let mut fault = None;
    while let Some(key) = iter.next() {
        if key == "--fault" {
            if fault.is_some() {
                return Err("duplicate --fault".into());
            }
            let value = iter.next().ok_or("missing value for --fault")?;
            if value.is_empty() || value.starts_with("--") {
                return Err("missing value for --fault".into());
            }
            // The arm name is never echoed: the collector reads the refusal from the last line.
            fault = Some(super::fault_contract::Arm::parse(&value).ok_or_else(|| {
                format!(
                    "REFUSED: unknown fault arm (expected {})",
                    super::fault_contract::Arm::NAMES
                )
            })?);
            continue;
        }
        if key == "--reclaim-cycles" {
            if reclaim_cycles.is_some() {
                return Err("REFUSED: duplicate --reclaim-cycles".into());
            }
            // Junk is never echoed: the collector reads the refusal from the last console line.
            let value = iter.next().ok_or(CYCLES_REFUSAL)?;
            reclaim_cycles = Some(parse_cycles(&value)?);
            continue;
        }
        if key == "--reclaim-diagnostic" {
            if reclaim_diagnostic {
                return Err("duplicate --reclaim-diagnostic".into());
            }
            reclaim_diagnostic = true;
            continue;
        }
        if key == "--same-program" {
            if same {
                return Err("duplicate --same-program".into());
            }
            same = true;
            continue;
        }
        if ![
            "--artifact",
            "--case",
            "--context",
            "--tiers",
            "--out",
            "--kv-allocator",
        ]
        .contains(&key.as_str())
        {
            return Err(format!("unknown argument {key}; {USAGE}"));
        }
        let value = iter
            .next()
            .ok_or_else(|| format!("missing value for {key}"))?;
        if value.is_empty() || value.starts_with("--") {
            return Err(format!("missing value for {key}"));
        }
        if fields.insert(key.clone(), value).is_some() {
            return Err(format!("duplicate {key}"));
        }
    }
    if !same {
        return Err("--same-program is mandatory; no alternate numerical program permitted".into());
    }
    if reclaim_cycles.is_some() && !reclaim_diagnostic {
        return Err("REFUSED: --reclaim-cycles requires --reclaim-diagnostic".into());
    }
    let kv_allocator = match fields
        .remove("--kv-allocator")
        .as_deref()
        .unwrap_or("pooled")
    {
        "pooled" => KvAllocator::Pooled,
        "vmm" => KvAllocator::Vmm,
        "vmm-ondemand" => KvAllocator::VmmOnDemand,
        _ => {
            return Err(
                "REFUSED: unknown KV allocator (expected pooled, vmm or vmm-ondemand)".into(),
            );
        }
    };
    if reclaim_cycles.is_some() && kv_allocator != KvAllocator::Vmm {
        return Err(
            "REFUSED: --reclaim-cycles requires --kv-allocator vmm; a pooled cache releases no chunk"
                .into(),
        );
    }
    let mut get = |key: &str| fields.remove(key).ok_or_else(|| format!("missing {key}"));
    let artifact = get("--artifact")?.into();
    let case = get("--case")?;
    if !["baseline", "active", "prefix", "grow"].contains(&case.as_str()) {
        return Err("case must be baseline, active, prefix or grow".into());
    }
    // WP-B day 37: the grow series and the on-demand allocator come together, and only
    // together; the case runs its own pooled arm in the same process for the identity term.
    if (case == "grow") != (kv_allocator == KvAllocator::VmmOnDemand) {
        return Err(
            "REFUSED: --case grow requires --kv-allocator vmm-ondemand, and that allocator is bound to --case grow"
                .into(),
        );
    }
    let context = match get("--context")?.as_str() {
        "8192" => 8192,
        "32768" => 32768,
        "16384" if reclaim_diagnostic => 16384,
        _ => return Err("only fitting development contexts 8192 and 32768 are implemented".into()),
    };
    let tiers = get("--tiers")?;
    if !["host", "host,nvme"].contains(&tiers.as_str()) {
        return Err("tiers must be host or host,nvme (no substitute storage route)".into());
    }
    let out = get("--out")?.into();
    if reclaim_diagnostic
        && (kv_allocator != KvAllocator::Vmm || case != "active" || tiers != "host")
    {
        return Err("REFUSED: reclaim diagnostic requires active VMM host mode".into());
    }
    if case == "grow" && (reclaim_diagnostic || fault.is_some() || tiers != "host") {
        return Err("REFUSED: the grow series takes no diagnostic, fault or tier route".into());
    }
    if fault.is_some() {
        // A fault arm is a door on the active roundtrip: anywhere else it is a usage error.
        if case != "active" {
            return Err(format!("--fault requires --case active; {USAGE}"));
        }
        if reclaim_diagnostic {
            return Err("REFUSED: fault arms do not combine with the reclaim diagnostic".into());
        }
        if kv_allocator != KvAllocator::Pooled {
            return Err(
                "REFUSED: fault arms are bound to the pooled allocator; the VMM door is not a fault surface"
                    .into(),
            );
        }
    }
    Ok(Args {
        artifact,
        kv_allocator,
        reclaim_diagnostic,
        reclaim_cycles,
        fault,
        case,
        context,
        tiers,
        out,
    })
}

/// ASCII digits only and N >= 2: one cycle is the existing single roundtrip, not a series.
fn parse_cycles(value: &str) -> Result<usize, String> {
    match value.parse::<usize>() {
        Ok(n) if n >= 2 && value.bytes().all(|b| b.is_ascii_digit()) => Ok(n),
        _ => Err(CYCLES_REFUSAL.into()),
    }
}

/// The collector recognizes an explicit refusal only at the start of the last line.
/// Do not turn unexpected runtime errors into unsupported-route refusals.
pub fn diagnostic(error: &str) -> String {
    if error.starts_with("REFUSED:") {
        error.to_owned()
    } else {
        format!("kv-tier-gate: {error}")
    }
}

#[cfg(test)]
mod tests {
    fn argv(extra: &[&str]) -> Vec<String> {
        let mut v: Vec<String> = [
            "--artifact",
            "m.gguf",
            "--context",
            "32768",
            "--tiers",
            "host",
            "--same-program",
            "--out",
            "o",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        v.extend(extra.iter().map(|s| s.to_string()));
        v
    }

    #[test]
    fn the_grow_case_and_the_on_demand_allocator_come_together() {
        let ok = super::parse(argv(&["--case", "grow", "--kv-allocator", "vmm-ondemand"])).unwrap();
        assert_eq!(ok.case, "grow");
        assert_eq!(ok.kv_allocator, super::KvAllocator::VmmOnDemand);
        for bad in [
            &["--case", "grow"][..],
            &["--case", "grow", "--kv-allocator", "vmm"][..],
            &["--case", "baseline", "--kv-allocator", "vmm-ondemand"][..],
            &["--case", "active", "--kv-allocator", "vmm-ondemand"][..],
        ] {
            assert!(
                super::parse(argv(bad)).unwrap_err().starts_with("REFUSED:"),
                "{bad:?}"
            );
        }
    }

    #[test]
    fn collector_refusal_is_unwrapped_but_failures_are_not_refusals() {
        assert_eq!(
            super::diagnostic("REFUSED: missing binding"),
            "REFUSED: missing binding"
        );
        assert_eq!(
            super::diagnostic("CUDA failed"),
            "kv-tier-gate: CUDA failed"
        );
    }
    use super::*;
    fn base() -> Vec<String> {
        "--artifact fixture.gguf --case baseline --context 8192 --tiers host,nvme --same-program --out receipts"
            .split_whitespace().map(str::to_owned).collect()
    }
    #[test]
    fn exact_contract_and_all_cases() {
        for case in ["baseline", "active", "prefix"] {
            for context in ["8192", "32768"] {
                let mut args = base();
                args[3] = case.into();
                args[5] = context.into();
                let parsed = parse(args).unwrap();
                assert_eq!(parsed.case, case);
                assert_eq!(parsed.context.to_string(), context);
            }
        }
    }
    #[test]
    fn missing_duplicate_and_unknown_refuse() {
        let mut args = base();
        args.retain(|a| a != "--same-program");
        assert!(parse(args).unwrap_err().contains("mandatory"));
        for suffix in [
            "--same-program",
            "--case active",
            "--fallback bf16",
            "--out",
        ] {
            let mut args = base();
            args.extend(suffix.split_whitespace().map(str::to_owned));
            assert!(parse(args).is_err());
        }
        let mut args = base();
        args.drain(0..2);
        assert!(parse(args).unwrap_err().contains("--artifact"));
    }
    #[test]
    fn allocator_defaults_and_refusals() {
        assert_eq!(parse(base()).unwrap().kv_allocator, KvAllocator::Pooled);
        for (value, expected) in [("pooled", KvAllocator::Pooled), ("vmm", KvAllocator::Vmm)] {
            let mut args = base();
            args.extend(["--kv-allocator".into(), value.into()]);
            assert_eq!(parse(args).unwrap().kv_allocator, expected);
        }
        let mut args = base();
        args.extend(["--kv-allocator".into(), "fallback".into()]);
        assert!(parse(args).unwrap_err().starts_with("REFUSED:"));
    }
    #[test]
    fn residual_diagnostic_is_explicit_and_scoped() {
        let mut args = base();
        args[3] = "active".into();
        args[5] = "16384".into();
        args[7] = "host".into();
        args.extend(["--kv-allocator".into(), "vmm".into()]);
        assert!(parse(args.clone()).is_err());
        args.push("--reclaim-diagnostic".into());
        assert!(parse(args.clone()).unwrap().reclaim_diagnostic);
        args[3] = "baseline".into();
        assert!(parse(args).is_err());
    }
    fn diagnostic_vmm() -> Vec<String> {
        let mut args = base();
        args[3] = "active".into();
        args[7] = "host".into();
        args.extend(["--kv-allocator", "vmm", "--reclaim-diagnostic"].map(String::from));
        args
    }
    #[test]
    fn reclaim_cycles_accepts_a_count_of_at_least_two_under_the_diagnostic() {
        assert_eq!(parse(diagnostic_vmm()).unwrap().reclaim_cycles, None);
        for (value, expected) in [("2", 2usize), ("5", 5), ("64", 64)] {
            let mut args = diagnostic_vmm();
            args.extend(["--reclaim-cycles".into(), value.into()]);
            let parsed = parse(args).unwrap();
            assert_eq!(parsed.reclaim_cycles, Some(expected));
            assert!(parsed.reclaim_diagnostic);
            assert_eq!(parsed.kv_allocator, KvAllocator::Vmm);
        }
        // Argument order is not part of the contract.
        let mut args = vec!["--reclaim-cycles".to_string(), "3".to_string()];
        args.extend(diagnostic_vmm());
        assert_eq!(parse(args).unwrap().reclaim_cycles, Some(3));
    }
    #[test]
    fn reclaim_cycles_refusals_are_explicit_and_fail_closed() {
        let refused = |args: Vec<String>| {
            let error = parse(args).unwrap_err();
            assert!(error.starts_with("REFUSED:"), "{error}");
            assert!(!error.contains('\n'), "{error}");
            error
        };
        // N < 2, junk, negative, fractional, empty, another flag, hex and a missing value.
        for value in [
            "0", "1", "five", "-2", "2.5", "", "--out", "+3", "3 ", "0x10",
        ] {
            let mut args = diagnostic_vmm();
            args.extend(["--reclaim-cycles".into(), value.into()]);
            assert!(refused(args).contains(">= 2"), "{value:?}");
        }
        let mut args = diagnostic_vmm();
        args.push("--reclaim-cycles".into());
        assert!(refused(args).contains(">= 2"));
        let mut args = diagnostic_vmm();
        args.extend(["--reclaim-cycles", "3", "--reclaim-cycles", "3"].map(String::from));
        assert!(refused(args).contains("duplicate --reclaim-cycles"));
        let mut args = diagnostic_vmm();
        args.retain(|a| a != "--reclaim-diagnostic");
        args.extend(["--reclaim-cycles".into(), "3".into()]);
        assert!(refused(args).contains("requires --reclaim-diagnostic"));
        // A pooled cache, explicit or by default, has no released chunk to cycle.
        for allocator in [Some("pooled"), None] {
            let mut args = base();
            args[3] = "active".into();
            args[7] = "host".into();
            if let Some(allocator) = allocator {
                args.extend(["--kv-allocator".into(), allocator.into()]);
            }
            args.extend(["--reclaim-diagnostic", "--reclaim-cycles", "3"].map(String::from));
            refused(args);
        }
        // Neither a baseline case nor a second storage route may carry the series.
        for (index, value) in [(3, "baseline"), (7, "host,nvme")] {
            let mut args = diagnostic_vmm();
            args[index] = value.into();
            args.extend(["--reclaim-cycles".into(), "3".into()]);
            refused(args);
        }
    }
    fn active_host() -> Vec<String> {
        let mut args = base();
        args[3] = "active".into();
        args[7] = "host".into();
        args
    }
    #[test]
    fn fault_arms_parse_only_on_the_active_pooled_roundtrip() {
        use super::super::fault_contract::Arm;
        assert_eq!(parse(active_host()).unwrap().fault, None);
        for arm in Arm::ALL {
            let mut args = active_host();
            args.extend(["--fault".into(), arm.name().into()]);
            let parsed = parse(args.clone()).unwrap();
            assert_eq!(parsed.fault, Some(arm));
            assert_eq!(parsed.kv_allocator, KvAllocator::Pooled);
            assert!(!parsed.reclaim_diagnostic);
            // Explicit pooled is the same door; argument order is not part of the contract.
            args.extend(["--kv-allocator".into(), "pooled".into()]);
            assert_eq!(parse(args.clone()).unwrap().fault, Some(arm));
            args.rotate_right(2);
            assert_eq!(parse(args).unwrap().fault, Some(arm));
        }
    }
    #[test]
    fn fault_door_is_a_usage_error_off_the_active_case_and_fails_closed() {
        // Not a refusal: the collector must classify a wrong invocation as a failed cell.
        for case in ["baseline", "prefix"] {
            let mut args = base();
            args[3] = case.into();
            args[7] = "host".into();
            args.extend(["--fault".into(), "cancel-demote".into()]);
            let error = parse(args).unwrap_err();
            assert!(
                error.starts_with("--fault requires --case active"),
                "{error}"
            );
            assert!(!error.starts_with("REFUSED"));
        }
        let mut args = active_host();
        args.retain(|a| a != "--same-program");
        args.extend(["--fault".into(), "cancel-demote".into()]);
        assert!(parse(args).unwrap_err().contains("mandatory"));
        // Duplicate, missing value, a flag in the value slot, and junk.
        let mut args = active_host();
        args.extend(["--fault", "cancel-demote", "--fault", "cancel-demote"].map(String::from));
        assert_eq!(parse(args).unwrap_err(), "duplicate --fault");
        let mut args = active_host();
        args.push("--fault".into());
        assert_eq!(parse(args).unwrap_err(), "missing value for --fault");
        let mut args = active_host();
        args.extend(["--fault".into(), "--out".into()]);
        assert_eq!(parse(args).unwrap_err(), "missing value for --fault");
        for junk in ["cancel", "Cancel-Demote", "cancel-demote ", "vmm", ""] {
            let mut args = active_host();
            args.extend(["--fault".into(), junk.into()]);
            let error = parse(args).unwrap_err();
            if junk.is_empty() {
                assert_eq!(error, "missing value for --fault");
            } else {
                assert!(
                    error.starts_with("REFUSED: unknown fault arm"),
                    "{junk:?}: {error}"
                );
                if !super::super::fault_contract::Arm::NAMES.contains(junk) {
                    assert!(!error.contains(junk), "junk echoed: {error}");
                }
            }
        }
    }
    #[test]
    fn fault_arms_refuse_the_vmm_door_and_the_reclaim_diagnostic() {
        let mut args = active_host();
        args.extend(["--fault", "device-short", "--kv-allocator", "vmm"].map(String::from));
        let error = parse(args).unwrap_err();
        assert!(
            error.starts_with("REFUSED: fault arms are bound to the pooled allocator"),
            "{error}"
        );
        let mut args = diagnostic_vmm();
        args.extend(["--fault".into(), "device-short".into()]);
        let error = parse(args).unwrap_err();
        assert_eq!(
            error,
            "REFUSED: fault arms do not combine with the reclaim diagnostic"
        );
        let mut args = diagnostic_vmm();
        args.extend(["--reclaim-cycles", "5", "--fault", "cancel-demote"].map(String::from));
        assert!(parse(args).unwrap_err().starts_with("REFUSED:"));
        // Without --same-program the mandatory-program error still wins.
        let mut args = active_host();
        args.retain(|a| a != "--same-program");
        args.extend(["--fault", "missing-host", "--kv-allocator", "vmm"].map(String::from));
        assert!(parse(args).unwrap_err().contains("mandatory"));
    }
    #[test]
    fn unsupported_ladder_and_routes_refuse() {
        for (index, value) in [
            (3, "churn"),
            (5, "262144"),
            (5, "0"),
            (7, "nvme"),
            (7, "host,peer"),
        ] {
            let mut args = base();
            args[index] = value.into();
            assert!(parse(args).is_err());
        }
    }
}
