//! Pure argument contract; `rustc --test` can exercise this without CUDA.
use std::path::PathBuf;

#[derive(Debug, PartialEq, Eq)]
pub enum KvAllocator {
    Pooled,
    Vmm,
}

#[derive(Debug, PartialEq, Eq)]
pub struct Args {
    pub artifact: PathBuf,
    pub kv_allocator: KvAllocator,
    pub reclaim_diagnostic: bool,
    /// Day 11: repeat the demote/restore program N >= 2 times in one process (diagnostic only).
    pub reclaim_cycles: Option<usize>,
    pub case: String,
    pub context: usize,
    pub tiers: String,
    pub out: PathBuf,
}

pub const USAGE: &str = "kv-tier-gate --artifact <gguf> --case baseline|active|prefix --context 8192|32768 --tiers host|host,nvme --same-program [--kv-allocator pooled|vmm] [--reclaim-diagnostic [--reclaim-cycles N]] --out <new-directory>";
const CYCLES_REFUSAL: &str = "REFUSED: --reclaim-cycles requires an integer count >= 2";

pub fn parse(args: impl IntoIterator<Item = String>) -> Result<Args, String> {
    let mut iter = args.into_iter();
    let mut fields = std::collections::BTreeMap::new();
    let mut same = false;
    let mut reclaim_diagnostic = false;
    let mut reclaim_cycles = None;
    while let Some(key) = iter.next() {
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
        _ => return Err("REFUSED: unknown KV allocator (expected pooled or vmm)".into()),
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
    if !["baseline", "active", "prefix"].contains(&case.as_str()) {
        return Err("case must be baseline, active or prefix".into());
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
    Ok(Args {
        artifact,
        kv_allocator,
        reclaim_diagnostic,
        reclaim_cycles,
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
