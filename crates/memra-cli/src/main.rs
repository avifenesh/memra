use memra_cli::{
    InspectRequest, ScaffoldRequest, VerifyRequest, VerifyStage, inspect_model,
    scaffold_model_pack, verify_model,
};
use std::path::PathBuf;

fn main() {
    if let Err(error) = run() {
        eprintln!("memra: {error}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    match (args.next().as_deref(), args.next().as_deref()) {
        (Some("model"), Some("inspect")) => {
            let source = args.next().ok_or(USAGE)?;
            let mut against = None;
            let mut out = None;
            while let Some(flag) = args.next() {
                match flag.as_str() {
                    "--against" => against = args.next(),
                    "--out" => out = args.next().map(PathBuf::from),
                    _ => return Err(format!("unknown argument {flag}\n{USAGE}").into()),
                }
            }
            let summary = inspect_model(InspectRequest {
                source,
                against: against.ok_or("--against is required")?,
                out_dir: out.ok_or("--out is required")?,
            })?;
            println!(
                "family={} tensors={} artifacts={}",
                summary.family,
                summary.tensor_count,
                summary.out_dir.display()
            );
            Ok(())
        }
        (Some("model"), Some("scaffold")) => {
            let family = args.next().ok_or(USAGE)?;
            let mut out = None;
            while let Some(flag) = args.next() {
                match flag.as_str() {
                    "--out" => out = args.next().map(PathBuf::from),
                    _ => return Err(format!("unknown argument {flag}\n{USAGE}").into()),
                }
            }
            let out_dir = out.ok_or("--out is required")?;
            scaffold_model_pack(ScaffoldRequest {
                family: family.clone(),
                out_dir: out_dir.clone(),
            })?;
            println!("family={family} scaffold={}", out_dir.display());
            Ok(())
        }
        (Some("model"), Some("verify")) => {
            let stage = match args.next().as_deref() {
                Some("config") => VerifyStage::Config,
                Some("tiny") => VerifyStage::Tiny,
                Some("checkpoint") => VerifyStage::Checkpoint,
                Some("rewrite") => VerifyStage::Rewrite,
                Some("serve") => VerifyStage::Serve,
                _ => return Err(USAGE.into()),
            };
            let source = if stage == VerifyStage::Tiny {
                String::new()
            } else {
                args.next().ok_or(USAGE)?
            };
            let mut against = None;
            let mut out = None;
            let mut oracle = None;
            let mut native_runner = None;
            while let Some(flag) = args.next() {
                match flag.as_str() {
                    "--against" => against = args.next(),
                    "--out" => out = args.next().map(PathBuf::from),
                    "--oracle" => oracle = args.next().map(PathBuf::from),
                    "--native-runner" => native_runner = args.next().map(PathBuf::from),
                    _ => return Err(format!("unknown argument {flag}\n{USAGE}").into()),
                }
            }
            let summary = verify_model(VerifyRequest {
                stage,
                source,
                against: against.ok_or("--against is required")?,
                out_dir: out,
                oracle,
                native_runner,
            })?;
            println!("family={} verified={:?}", summary.family, summary.stage);
            Ok(())
        }
        _ => Err(USAGE.into()),
    }
}

const USAGE: &str = "usage:
  memra model inspect <local-path|hf-id@40-char-sha> --against <family> --out <dir>
  memra model scaffold <new-family> --out <dir>
  memra model verify <config|checkpoint|rewrite|serve> <source> --against <family> [--out <dir>] [--oracle <file>] [--native-runner <binary>]
  memra model verify tiny --against <family> --out <dir>";
