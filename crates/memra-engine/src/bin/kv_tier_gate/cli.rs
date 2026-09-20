//! Pure argument contract; `rustc --test` can exercise this without CUDA.
use std::path::PathBuf;

#[derive(Debug, PartialEq, Eq)]
pub struct Args {
    pub artifact: PathBuf,
    pub case: String,
    pub context: usize,
    pub tiers: String,
    pub out: PathBuf,
}

pub const USAGE: &str = "kv-tier-gate --artifact <gguf> --case baseline|active|prefix --context 8192|32768 --tiers host|host,nvme --same-program --out <new-directory>";

pub fn parse(args: impl IntoIterator<Item = String>) -> Result<Args, String> {
    let mut iter = args.into_iter();
    let mut fields = std::collections::BTreeMap::new();
    let mut same = false;
    while let Some(key) = iter.next() {
        if key == "--same-program" {
            if same {
                return Err("duplicate --same-program".into());
            }
            same = true;
            continue;
        }
        if !["--artifact", "--case", "--context", "--tiers", "--out"].contains(&key.as_str()) {
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
    let mut get = |key: &str| fields.remove(key).ok_or_else(|| format!("missing {key}"));
    let artifact = get("--artifact")?.into();
    let case = get("--case")?;
    if !["baseline", "active", "prefix"].contains(&case.as_str()) {
        return Err("case must be baseline, active or prefix".into());
    }
    let context = match get("--context")?.as_str() {
        "8192" => 8192,
        "32768" => 32768,
        _ => return Err("only fitting development contexts 8192 and 32768 are implemented".into()),
    };
    let tiers = get("--tiers")?;
    if !["host", "host,nvme"].contains(&tiers.as_str()) {
        return Err("tiers must be host or host,nvme (no substitute storage route)".into());
    }
    let out = get("--out")?.into();
    Ok(Args {
        artifact,
        case,
        context,
        tiers,
        out,
    })
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
