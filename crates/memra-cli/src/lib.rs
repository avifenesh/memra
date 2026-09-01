use memra_gguf::config::{HfConfig, ModelConfig};
use memra_gguf::model_packs::{self, Gate, ModelPack, TokenizerSource};
use memra_gguf::safetensors::{StInfo, StModel, parse_header_json, parse_index_weight_map_json};
use memra_gguf::tensor_contract::{
    CheckpointDialect, ContractOptions, FloatType, IntegerType, OutputHead, QuantLayout,
    StorageLayout, TensorCensusEntry,
};
use memra_gguf::{GgmlType, GgufFile};
use memra_reference::{deterministic_fixture, execute, execute_multimodal, execute_vision};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};

const MAX_TEXT_BYTES: usize = 100_000_000;

pub struct InspectRequest {
    pub source: String,
    pub against: String,
    pub out_dir: PathBuf,
}

pub struct InspectSummary {
    pub family: &'static str,
    pub tensor_count: usize,
    pub out_dir: PathBuf,
}

pub struct ScaffoldRequest {
    pub family: String,
    pub out_dir: PathBuf,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VerifyStage {
    Config,
    Tiny,
    Checkpoint,
    Rewrite,
    Serve,
}

pub struct VerifyRequest {
    pub stage: VerifyStage,
    pub source: String,
    pub against: String,
    pub out_dir: Option<PathBuf>,
    pub oracle: Option<PathBuf>,
    pub native_runner: Option<PathBuf>,
}

pub struct VerifySummary {
    pub family: &'static str,
    pub stage: VerifyStage,
}

pub fn verify_model(request: VerifyRequest) -> Result<VerifySummary, Box<dyn std::error::Error>> {
    match request.stage {
        VerifyStage::Config => {
            let pack = model_packs::by_alias(&request.against)
                .ok_or_else(|| format!("unknown model pack {:?}", request.against))?;
            let config = load_config_only(&request.source)?;
            pack.compile_plan(&config)?;
            Ok(VerifySummary {
                family: pack.family,
                stage: VerifyStage::Config,
            })
        }
        VerifyStage::Checkpoint => {
            let pack = model_packs::by_alias(&request.against)
                .ok_or_else(|| format!("unknown model pack {:?}", request.against))?;
            let out_dir = request.out_dir.ok_or("verify checkpoint requires --out")?;
            let summary = inspect_model(InspectRequest {
                source: request.source.clone(),
                against: request.against.clone(),
                out_dir: out_dir.clone(),
            })?;
            write_hf_oracle_bundle(&request.source, &out_dir)?;
            let gate = pack.checkpoint_parity.ok_or_else(|| {
                format!(
                    "model pack {} has no checkpoint parity threshold; capture bundle written to {} and no fallback is allowed",
                    pack.family,
                    out_dir.display()
                )
            })?;
            let oracle_path = request.oracle.ok_or_else(|| {
                format!(
                    "checkpoint tensor contract passed; run {} offline, then repeat with --oracle <hf-oracle.tsv>; no fallback is allowed",
                    out_dir.join("capture-hf-oracle.py").display()
                )
            })?;
            let runner = request
                .native_runner
                .or_else(|| std::env::var_os("MEMRA_NATIVE_CHECKPOINT_RUNNER").map(PathBuf::from))
                .ok_or("checkpoint parity requires --native-runner or MEMRA_NATIVE_CHECKPOINT_RUNNER; no fallback is allowed")?;
            let native_path = out_dir.join("native-oracle.tsv");
            run_native_checkpoint(&runner, &request.source, &native_path)?;
            let runner_hash = hex_sha256(&std::fs::read(&runner)?);
            let expected = parse_checkpoint_oracle(&std::fs::read_to_string(&oracle_path)?)?;
            let actual = parse_checkpoint_oracle(&std::fs::read_to_string(&native_path)?)?;
            let receipt = match compare_checkpoint_oracles(&expected, &actual, gate) {
                Ok(receipt) => receipt,
                Err(error) => {
                    write_atomic(
                        &out_dir.join("checkpoint-parity.tsv"),
                        format!(
                            "status\tfailed\nerror\t{}\n",
                            lock_value(&error.to_string())
                        )
                        .as_bytes(),
                    )?;
                    write_atomic(
                        &out_dir.join("gates.txt"),
                        format_gate_results_with_receipts(
                            pack,
                            &out_dir,
                            &[Gate::Config, Gate::TokenizerTemplate, Gate::TensorCensus],
                            &[Gate::CheckpointParity],
                        )
                        .as_bytes(),
                    )?;
                    return Err(error);
                }
            };
            let artifact_lock = std::fs::read(out_dir.join("artifact.lock"))?;
            let receipt = format!(
                "{receipt}artifact_lock_sha256\t{}\nnative_runner_sha256\t{runner_hash}\n",
                hex_sha256(&artifact_lock)
            );
            write_atomic(&out_dir.join("checkpoint-parity.tsv"), receipt.as_bytes())?;
            write_atomic(
                &out_dir.join("gates.txt"),
                format_gate_results_with_receipts(
                    pack,
                    &out_dir,
                    &[
                        Gate::Config,
                        Gate::TokenizerTemplate,
                        Gate::TensorCensus,
                        Gate::CheckpointParity,
                    ],
                    &[],
                )
                .as_bytes(),
            )?;
            Ok(VerifySummary {
                family: summary.family,
                stage: VerifyStage::Checkpoint,
            })
        }
        VerifyStage::Tiny => {
            let pack = model_packs::by_alias(&request.against)
                .ok_or_else(|| format!("unknown model pack {:?}", request.against))?;
            if pack.support.is_none() {
                return Err(format!(
                    "model pack {} is inspect-only and has no native support state",
                    pack.family
                )
                .into());
            }
            let out_dir = request.out_dir.ok_or("verify tiny requires --out")?;
            let plan = pack.compile_tiny_plan()?;
            let fixture = deterministic_fixture(&plan)?;
            let first = execute(&plan, &fixture.weights, &fixture.token_ids)?;
            let second = execute(&plan, &fixture.weights, &fixture.token_ids)?;
            if first != second {
                return Err("native reference fixture is not bit-deterministic".into());
            }
            let vision = fixture
                .vision
                .as_ref()
                .map(|input| {
                    let first = execute_vision(&plan, &fixture.weights, input)?;
                    let second = execute_vision(&plan, &fixture.weights, input)?;
                    if first != second {
                        return Err(ReferenceVisionError::Nondeterministic);
                    }
                    Ok(first)
                })
                .transpose()
                .map_err(|error| -> Box<dyn std::error::Error> { error.into() })?;
            let multimodal = match (
                fixture.multimodal_token_ids.as_ref(),
                fixture.vision.as_ref(),
            ) {
                (Some(token_ids), Some(input)) => {
                    let first = execute_multimodal(&plan, &fixture.weights, token_ids, input)?;
                    let second = execute_multimodal(&plan, &fixture.weights, token_ids, input)?;
                    if first != second {
                        return Err(
                            "native multimodal reference fixture is not bit-deterministic".into(),
                        );
                    }
                    Some(first)
                }
                (None, None) | (None, Some(_)) if plan.multimodal.is_none() => None,
                _ => {
                    return Err(
                        "multimodal plan is missing its combined tiny fixture inputs".into(),
                    );
                }
            };
            std::fs::create_dir_all(&out_dir)?;
            write_atomic(
                &out_dir.join("tiny-fixture.txt"),
                format_tiny_fixture(&plan, &fixture).as_bytes(),
            )?;
            write_atomic(
                &out_dir.join("reference-oracle.tsv"),
                format_reference_oracle(&first).as_bytes(),
            )?;
            if let Some(vision) = vision.as_ref() {
                write_atomic(
                    &out_dir.join("reference-vision-oracle.tsv"),
                    format_reference_vision_oracle(vision).as_bytes(),
                )?;
            }
            if let Some(multimodal) = multimodal.as_ref() {
                write_atomic(
                    &out_dir.join("reference-multimodal-oracle.tsv"),
                    format_reference_oracle(&multimodal.language).as_bytes(),
                )?;
            }
            write_atomic(
                &out_dir.join("tiny-gate.tsv"),
                format!("status\tpassed\nfamily\t{}\n", pack.family).as_bytes(),
            )?;
            write_atomic(
                &out_dir.join("gates.txt"),
                format_gate_results_with_receipts(
                    pack,
                    &out_dir,
                    &[Gate::Config, Gate::TinyParity],
                    &[],
                )
                .as_bytes(),
            )?;
            Ok(VerifySummary {
                family: pack.family,
                stage: VerifyStage::Tiny,
            })
        }
        VerifyStage::Rewrite => {
            let pack = model_packs::by_alias(&request.against)
                .ok_or_else(|| format!("unknown model pack {:?}", request.against))?;
            let out_dir = request
                .out_dir
                .ok_or("verify rewrite requires --out; no fallback is allowed")?;
            verify_rewrite_receipt(pack, Path::new(&request.source), &out_dir)?;
            Ok(VerifySummary {
                family: pack.family,
                stage: VerifyStage::Rewrite,
            })
        }
        VerifyStage::Serve => {
            let pack = model_packs::by_alias(&request.against)
                .ok_or_else(|| format!("unknown model pack {:?}", request.against))?;
            let out_dir = request
                .out_dir
                .ok_or("verify serve requires --out; no fallback is allowed")?;
            let runner = request
                .native_runner
                .or_else(|| std::env::var_os("MEMRA_NATIVE_SERVE_RUNNER").map(PathBuf::from))
                .ok_or("verify serve requires --native-runner or MEMRA_NATIVE_SERVE_RUNNER; no fallback is allowed")?;
            verify_native_serve(pack, &request.source, &out_dir, &runner)?;
            Ok(VerifySummary {
                family: pack.family,
                stage: VerifyStage::Serve,
            })
        }
    }
}

fn verify_rewrite_receipt(
    pack: &ModelPack,
    receipt_path: &Path,
    out_dir: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let artifact_lock = std::fs::read_to_string(out_dir.join("artifact.lock"))?;
    let artifact_lock_sha256 = hex_sha256(artifact_lock.as_bytes());
    if !artifact_lock
        .lines()
        .any(|line| line == format!("family={}", pack.family))
    {
        return Err("rewrite receipt family does not match artifact.lock".into());
    }
    let manifest = std::fs::read_to_string(out_dir.join("execution-rewrites.tsv"))?;
    let receipt = std::fs::read_to_string(receipt_path)?;
    let mut fields = BTreeMap::new();
    for line in receipt.lines() {
        let Some((key, value)) = line.split_once('\t') else {
            return Err(format!("malformed rewrite receipt line {line:?}").into());
        };
        if fields.insert(key, value).is_some() {
            return Err(format!("duplicate rewrite receipt field {key}").into());
        }
    }
    for (key, expected) in [
        ("format", "memra-rewrite-parity-v1"),
        ("status", "passed"),
        ("first_violation", "none"),
    ] {
        if fields.get(key).copied() != Some(expected) {
            return Err(format!("rewrite receipt requires {key}={expected}").into());
        }
    }
    match fields.get("value_kind").copied() {
        Some("logits-f32") if fields.get("require_argmax").copied() == Some("true") => {}
        Some("token-ids-u32") if fields.get("require_argmax").copied() == Some("false") => {}
        _ => return Err("rewrite receipt has an invalid value_kind/argmax policy".into()),
    }
    let rewrite_id = *fields.get("rewrite").ok_or("rewrite receipt has no id")?;
    let row = manifest
        .lines()
        .skip(1)
        .find(|line| line.split('\t').next() == Some(rewrite_id))
        .ok_or_else(|| format!("rewrite {rewrite_id} is absent from execution manifest"))?;
    let columns: Vec<_> = row.split('\t').collect();
    if columns.len() != 8 || columns[4] != "true" {
        return Err(format!("rewrite {rewrite_id} is not eligible in this artifact").into());
    }
    for (field, expected) in [
        ("surface", columns[1]),
        ("implementation", columns[2]),
        ("plan_sha256", columns[3]),
    ] {
        if fields.get(field).copied() != Some(expected) {
            return Err(format!("rewrite receipt {field} does not match manifest").into());
        }
    }
    if fields.get("artifact_lock_sha256").copied() != Some(artifact_lock_sha256.as_str()) {
        return Err("rewrite receipt does not match artifact.lock".into());
    }
    let reference = fields
        .get("reference_sha256")
        .ok_or("rewrite receipt has no reference hash")?;
    let candidate = fields
        .get("candidate_sha256")
        .ok_or("rewrite receipt has no candidate hash")?;
    let parse_nonnegative = |field: &str| -> Result<f32, Box<dyn std::error::Error>> {
        let value = fields
            .get(field)
            .ok_or_else(|| format!("rewrite receipt has no {field}"))?
            .parse::<f32>()?;
        if !value.is_finite() || value < 0.0 {
            return Err(format!("rewrite receipt {field} is not finite and nonnegative").into());
        }
        Ok(value)
    };
    let atol = parse_nonnegative("atol")?;
    let rtol = parse_nonnegative("rtol")?;
    let max_abs = parse_nonnegative("max_abs")?;
    let _max_rel = parse_nonnegative("max_rel")?;
    if atol == 0.0 && rtol == 0.0 && (max_abs != 0.0 || reference != candidate) {
        return Err("exact rewrite receipt has nonzero error or different stream hashes".into());
    }
    for field in [
        "implementation_sha256",
        "reference_sha256",
        "candidate_sha256",
    ] {
        let value = fields[field];
        if value.len() != 64
            || !value
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        {
            return Err(format!("rewrite receipt {field} is not a lowercase SHA-256").into());
        }
    }
    if fields
        .get("values")
        .and_then(|value| value.parse::<usize>().ok())
        .is_none_or(|values| values == 0)
    {
        return Err("rewrite receipt compared no values".into());
    }
    let receipt_hash = hex_sha256(receipt.as_bytes());
    let receipt_dir = out_dir.join("rewrite-receipts");
    std::fs::create_dir_all(&receipt_dir)?;
    write_atomic(
        &receipt_dir.join(format!("{rewrite_id}.tsv")),
        receipt.as_bytes(),
    )?;
    let index_path = out_dir.join("rewrite-receipts.tsv");
    let mut index = BTreeMap::new();
    if let Ok(existing) = std::fs::read_to_string(&index_path) {
        for line in existing.lines().skip(1) {
            let columns: Vec<_> = line.split('\t').collect();
            if columns.len() == 4 {
                index.insert(
                    columns[0].to_string(),
                    (
                        columns[1].to_string(),
                        columns[2].to_string(),
                        columns[3].to_string(),
                    ),
                );
            }
        }
    }
    index.insert(
        rewrite_id.to_string(),
        (columns[3].to_string(), receipt_hash, "passed".to_string()),
    );
    let mut index_text = String::from("rewrite\tplan_sha256\treceipt_sha256\tstatus\n");
    for (rewrite, (plan, hash, status)) in index {
        writeln!(index_text, "{rewrite}\t{plan}\t{hash}\t{status}").unwrap();
    }
    write_atomic(&index_path, index_text.as_bytes())?;
    write_atomic(
        &out_dir.join("gates.txt"),
        format_gate_results_with_receipts(pack, out_dir, &[], &[]).as_bytes(),
    )?;
    Ok(())
}

fn verify_native_serve(
    pack: &ModelPack,
    source: &str,
    out_dir: &Path,
    runner: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    if !Path::new(source).exists() {
        return Err("verify serve requires a local model artifact; no fallback is allowed".into());
    }
    let checkpoint_receipt = out_dir.join("checkpoint-parity.tsv");
    let artifact_lock_path = out_dir.join("artifact.lock");
    let artifact_lock = std::fs::read_to_string(&artifact_lock_path).map_err(|error| {
        format!(
            "verify serve requires {} from inspect/checkpoint first: {error}; no fallback is allowed",
            artifact_lock_path.display()
        )
    })?;
    if !artifact_lock
        .lines()
        .any(|line| line == format!("source={}", lock_value(source)))
        || !artifact_lock.lines().any(|line| line == "binding=passed")
        || !artifact_lock.lines().any(|line| line == "tokenizer=passed")
    {
        return Err(
            "verify serve artifact.lock does not match this source with binding/tokenizer passed; no fallback is allowed"
                .into(),
        );
    }
    let checkpoint = std::fs::read_to_string(&checkpoint_receipt).map_err(|error| {
        format!(
            "verify serve requires a passed {} first: {error}; no fallback is allowed",
            checkpoint_receipt.display()
        )
    })?;
    if !checkpoint.lines().any(|line| line == "status\tpassed") {
        return Err(
            "verify serve requires status=passed checkpoint parity; no fallback is allowed".into(),
        );
    }
    let lock_hash = hex_sha256(artifact_lock.as_bytes());
    if !checkpoint
        .lines()
        .any(|line| line == format!("artifact_lock_sha256\t{lock_hash}"))
    {
        return Err(
            "verify serve checkpoint receipt does not match artifact.lock; no fallback is allowed"
                .into(),
        );
    }
    std::fs::create_dir_all(out_dir)?;
    let listener = std::net::TcpListener::bind("127.0.0.1:0")?;
    let port = listener.local_addr()?.port();
    drop(listener);
    let address = format!("127.0.0.1:{port}");
    let api_key = "memra-onboarding-verify";
    let log_path = out_dir.join("serve.log");
    let log = std::fs::File::create(&log_path)?;
    let mut child = Command::new(runner)
        .env("MEMRA_MODELS", format!("verify={source}"))
        .env("MEMRA_REWRITE_BUNDLE", out_dir)
        .env("MEMRA_ADDR", &address)
        .env("MEMRA_API_KEY", api_key)
        .stdin(Stdio::null())
        .stdout(Stdio::from(log.try_clone()?))
        .stderr(Stdio::from(log))
        .spawn()?;
    let result = (|| -> Result<String, Box<dyn std::error::Error>> {
        let timeout = std::env::var("MEMRA_SERVE_VERIFY_TIMEOUT_S")
            .ok()
            .and_then(|value| value.parse::<u64>().ok())
            .unwrap_or(180);
        let started = std::time::Instant::now();
        loop {
            if let Some(status) = child.try_wait()? {
                return Err(format!(
                    "native server exited before readiness with {status}; inspect {}",
                    log_path.display()
                )
                .into());
            }
            let ready = Command::new("curl")
                .args([
                    "--fail",
                    "--silent",
                    "--output",
                    "/dev/null",
                    &format!("http://{address}/readyz"),
                ])
                .status();
            if ready.is_ok_and(|status| status.success()) {
                break;
            }
            if started.elapsed() >= std::time::Duration::from_secs(timeout) {
                return Err(format!(
                    "native server did not become ready within {timeout}s; inspect {}",
                    log_path.display()
                )
                .into());
            }
            std::thread::sleep(std::time::Duration::from_millis(250));
        }
        let response = Command::new("curl")
            .args([
                "--fail",
                "--silent",
                "--show-error",
                "--header",
                &format!("Authorization: Bearer {api_key}"),
                "--header",
                "Content-Type: application/json",
                "--data",
                r#"{"model":"verify","prompt":"Hello","max_tokens":1,"temperature":0}"#,
                &format!("http://{address}/v1/completions"),
            ])
            .output()?;
        if !response.status.success() {
            return Err(format!(
                "native completion failed with {}: {}",
                response.status,
                String::from_utf8_lossy(&response.stderr)
            )
            .into());
        }
        let response = String::from_utf8(response.stdout)?;
        if !response.contains("\"choices\"") || response.contains("\"error\"") {
            return Err(format!("native completion response is not successful: {response}").into());
        }
        Ok(response)
    })();
    let _ = child.kill();
    let _ = child.wait();
    let response = result?;
    let runner_hash = hex_sha256(&std::fs::read(runner)?);
    write_atomic(&out_dir.join("serve-response.json"), response.as_bytes())?;
    write_atomic(
        &out_dir.join("serve-gate.tsv"),
        format!(
            "status\tpassed\nfamily\t{}\nmodel\tverify\nendpoint\t/v1/completions\nartifact_lock_sha256\t{lock_hash}\nnative_runner_sha256\t{runner_hash}\n",
            pack.family,
        )
        .as_bytes(),
    )?;
    write_atomic(
        &out_dir.join("gates.txt"),
        format_gate_results_with_receipts(
            pack,
            out_dir,
            &[
                Gate::Config,
                Gate::TokenizerTemplate,
                Gate::TensorCensus,
                Gate::CheckpointParity,
                Gate::Serve,
            ],
            &[],
        )
        .as_bytes(),
    )?;
    Ok(())
}

#[derive(Debug, Clone, PartialEq)]
struct CheckpointOracle {
    engine: String,
    numeric_class: String,
    tokens: Vec<u32>,
    vocab: usize,
    logits: Vec<f32>,
}

fn write_hf_oracle_bundle(source: &str, out_dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let tokens = [1u32, 2, 3, 4];
    let (model, revision) = if Path::new(source).exists() {
        (source.to_string(), None)
    } else {
        let (model, revision) = parse_pinned_hf_source(source)?;
        (model.to_string(), Some(revision.to_string()))
    };
    let request = format!(
        "format\tmemra-checkpoint-request-v1\nsource\t{}\nrevision\t{}\nnumeric_class\tsource-weights-float32-accumulation\ntokens\t{}\n",
        lock_value(&model),
        revision.as_deref().unwrap_or("local"),
        tokens
            .iter()
            .map(u32::to_string)
            .collect::<Vec<_>>()
            .join(",")
    );
    write_atomic(&out_dir.join("oracle-request.tsv"), request.as_bytes())?;
    let model_literal = format!("{model:?}");
    let revision_literal = revision
        .as_ref()
        .map(|revision| format!("{revision:?}"))
        .unwrap_or_else(|| "None".to_string());
    let script = format!(
        r#"#!/usr/bin/env python3
import argparse
import struct
import torch
import transformers
from transformers import AutoModelForCausalLM

MODEL = {model_literal}
REVISION = {revision_literal}
TOKENS = [1, 2, 3, 4]

parser = argparse.ArgumentParser(description="Offline HF correctness oracle for Memra onboarding")
parser.add_argument("--out", default="hf-oracle.tsv")
args = parser.parse_args()

model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    revision=REVISION,
    dtype=torch.float32,
    trust_remote_code=False,
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device).eval()
with torch.no_grad():
    logits = model(input_ids=torch.tensor([TOKENS], device=device)).logits[0, -1].float().cpu()

with open(args.out, "w", encoding="utf-8") as f:
    f.write("format\tmemra-checkpoint-oracle-v1\n")
    f.write("engine\thf-transformers-fp32\n")
    f.write("numeric_class\tsource-weights-float32-accumulation\n")
    f.write(f"transformers_version\t{{transformers.__version__}}\n")
    f.write(f"torch_version\t{{torch.__version__}}\n")
    f.write("tokens\t" + ",".join(map(str, TOKENS)) + "\n")
    f.write(f"vocab\t{{logits.numel()}}\n")
    for index, value in enumerate(logits.tolist()):
        bits = struct.unpack("<I", struct.pack("<f", value))[0]
        f.write(f"logit\t{{index}}\t{{bits:08x}}\n")
"#
    );
    write_atomic(&out_dir.join("capture-hf-oracle.py"), script.as_bytes())?;
    Ok(())
}

fn run_native_checkpoint(
    runner: &Path,
    source: &str,
    output: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    if !Path::new(source).is_dir() {
        return Err(
            "native checkpoint parity requires a local safetensors directory; inspect may use a pinned remote header, execution may not"
                .into(),
        );
    }
    let result = Command::new(runner)
        .arg(source)
        .args(["1", "2", "3", "4"])
        .env("MEMRA_FULL_PREC", "1")
        .env("MEMRA_ORACLE_OUT", output)
        .output()?;
    if !result.status.success() {
        return Err(format!(
            "native checkpoint runner failed ({}): stdout={} stderr={}",
            result.status,
            String::from_utf8_lossy(&result.stdout),
            String::from_utf8_lossy(&result.stderr)
        )
        .into());
    }
    if !output.is_file() {
        return Err(format!(
            "native checkpoint runner did not create {}",
            output.display()
        )
        .into());
    }
    Ok(())
}

fn parse_checkpoint_oracle(text: &str) -> Result<CheckpointOracle, Box<dyn std::error::Error>> {
    let mut format_ok = false;
    let mut engine = None;
    let mut numeric_class = None;
    let mut tokens = None;
    let mut vocab = None;
    let mut logits = BTreeMap::new();
    for line in text.lines() {
        let fields: Vec<_> = line.split('\t').collect();
        match fields.as_slice() {
            ["format", "memra-checkpoint-oracle-v1"] => format_ok = true,
            ["engine", value] => engine = Some((*value).to_string()),
            ["numeric_class", value] => numeric_class = Some((*value).to_string()),
            ["tokens", value] => {
                tokens = Some(
                    value
                        .split(',')
                        .map(str::parse)
                        .collect::<Result<Vec<u32>, _>>()?,
                )
            }
            ["vocab", value] => vocab = Some(value.parse::<usize>()?),
            ["logit", index, bits] => {
                let index = index.parse::<usize>()?;
                let bits = u32::from_str_radix(bits, 16)?;
                if logits.insert(index, f32::from_bits(bits)).is_some() {
                    return Err(format!("duplicate oracle logit index {index}").into());
                }
            }
            _ => {}
        }
    }
    if !format_ok {
        return Err("oracle is missing format=memra-checkpoint-oracle-v1".into());
    }
    let vocab = vocab.ok_or("oracle is missing vocab")?;
    if logits.len() != vocab || (0..vocab).any(|index| !logits.contains_key(&index)) {
        return Err(format!(
            "oracle has {} logits, expected contiguous {vocab}",
            logits.len()
        )
        .into());
    }
    Ok(CheckpointOracle {
        engine: engine.ok_or("oracle is missing engine")?,
        numeric_class: numeric_class.ok_or("oracle is missing numeric_class")?,
        tokens: tokens.ok_or("oracle is missing tokens")?,
        vocab,
        logits: (0..vocab).map(|index| logits[&index]).collect(),
    })
}

fn compare_checkpoint_oracles(
    expected: &CheckpointOracle,
    actual: &CheckpointOracle,
    gate: model_packs::CheckpointParityGate,
) -> Result<String, Box<dyn std::error::Error>> {
    if expected.numeric_class != actual.numeric_class {
        return Err(format!(
            "oracle numeric class mismatch: expected={} native={}",
            expected.numeric_class, actual.numeric_class
        )
        .into());
    }
    if expected.tokens != actual.tokens || expected.vocab != actual.vocab {
        return Err(format!(
            "oracle identity mismatch: expected tokens={:?} vocab={}, native tokens={:?} vocab={}",
            expected.tokens, expected.vocab, actual.tokens, actual.vocab
        )
        .into());
    }
    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    let mut worst = 0usize;
    let mut first_violation = None;
    for (index, (&reference, &native)) in expected.logits.iter().zip(&actual.logits).enumerate() {
        if !reference.is_finite() || !native.is_finite() {
            return Err(format!("non-finite checkpoint logit at token {index}").into());
        }
        let absolute = (reference - native).abs();
        let relative = absolute / reference.abs().max(1e-6);
        if absolute > max_abs {
            max_abs = absolute;
            worst = index;
        }
        max_rel = max_rel.max(relative);
        let allowed = gate.max_abs + gate.max_rel * reference.abs();
        if absolute > allowed && first_violation.is_none() {
            first_violation = Some((index, absolute, allowed));
        }
    }
    let reference_argmax = stable_argmax(&expected.logits);
    let native_argmax = stable_argmax(&actual.logits);
    if let Some((index, absolute, allowed)) = first_violation {
        return Err(format!(
            "checkpoint parity failed at token {index}: abs={absolute} exceeds atol+rtol*abs(reference)={allowed}; observed max_abs={max_abs} at token {worst}, max_rel={max_rel}"
        )
        .into());
    }
    if gate.require_argmax && reference_argmax != native_argmax {
        return Err(format!(
            "checkpoint parity argmax mismatch: reference={reference_argmax} native={native_argmax}"
        )
        .into());
    }
    Ok(format!(
        "status\tpassed\nreference_engine\t{}\nnative_engine\t{}\nnumeric_class\t{}\ntokens\t{}\nvocab\t{}\nmax_abs\t{max_abs}\nmax_rel\t{max_rel}\nreference_argmax\t{reference_argmax}\nnative_argmax\t{native_argmax}\n",
        expected.engine,
        actual.engine,
        expected.numeric_class,
        expected
            .tokens
            .iter()
            .map(u32::to_string)
            .collect::<Vec<_>>()
            .join(","),
        expected.vocab,
    ))
}

fn stable_argmax(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|(left_index, left), (right_index, right)| {
            left.total_cmp(right)
                .then_with(|| right_index.cmp(left_index))
        })
        .map(|(index, _)| index)
        .unwrap_or(0)
}

#[derive(Debug)]
enum ReferenceVisionError {
    Reference(memra_reference::ReferenceError),
    Nondeterministic,
}

impl std::fmt::Display for ReferenceVisionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Reference(error) => error.fmt(f),
            Self::Nondeterministic => write!(
                f,
                "native vision reference fixture is not bit-deterministic"
            ),
        }
    }
}

impl std::error::Error for ReferenceVisionError {}

impl From<memra_reference::ReferenceError> for ReferenceVisionError {
    fn from(value: memra_reference::ReferenceError) -> Self {
        Self::Reference(value)
    }
}

pub fn scaffold_model_pack(request: ScaffoldRequest) -> Result<(), Box<dyn std::error::Error>> {
    validate_family_name(&request.family)?;
    if request.out_dir.exists() && request.out_dir.read_dir()?.next().is_some() {
        return Err(format!(
            "refusing to scaffold into non-empty directory {}",
            request.out_dir.display()
        )
        .into());
    }
    std::fs::create_dir_all(&request.out_dir)?;
    write_atomic(
        &request.out_dir.join("pack.toml"),
        format!(
            "family = {:?}\nconfig_layout = \"pending\"\nsupport = \"pending\"\n\n[checkpoint_parity]\nmax_abs = \"pending\"\nmax_rel = \"pending\"\nrequire_argmax = true\n",
            request.family
        )
        .as_bytes(),
    )?;
    write_atomic(
        &request.out_dir.join("aliases.txt"),
        format!("{}\n", request.family).as_bytes(),
    )?;
    write_atomic(
        &request.out_dir.join("config-normalization.txt"),
        b"# source field\tcanonical field\ttransform\n",
    )?;
    write_atomic(
        &request.out_dir.join("tensor-schema.tsv"),
        b"semantic_id\tcheckpoint_pattern\tshape\townership\ttransform\tquant_layout\n",
    )?;
    write_atomic(
        &request.out_dir.join("tokenizer-template.txt"),
        b"tokenizer_source=pending\ntemplate=artifact-required\n",
    )?;
    write_atomic(
        &request.out_dir.join("gates.txt"),
        format_gates(&[
            Gate::Config,
            Gate::TokenizerTemplate,
            Gate::TensorCensus,
            Gate::TinyParity,
            Gate::CheckpointParity,
            Gate::RewriteParity,
            Gate::Serve,
        ])
        .as_bytes(),
    )?;
    Ok(())
}

struct SourceData {
    label: String,
    revision: String,
    dialect: CheckpointDialect,
    config: ModelConfig,
    config_bytes: Vec<u8>,
    tensors: Vec<CensusRow>,
    shards: Vec<String>,
    tokenizer: Result<TokenizerEvidence, String>,
}

struct TokenizerEvidence {
    source: TokenizerSource,
    tokenizer_sha256: String,
    template_sha256: String,
    template_bytes: usize,
}

#[derive(Clone)]
struct CensusRow {
    physical_name: String,
    entry: TensorCensusEntry,
    dtype: String,
}

pub fn inspect_model(
    request: InspectRequest,
) -> Result<InspectSummary, Box<dyn std::error::Error>> {
    let pack = model_packs::by_alias(&request.against)
        .ok_or_else(|| format!("unknown model pack {:?}", request.against))?;
    let source = load_source(&request.source)?;
    let plan = pack.compile_plan(&source.config)?;
    let output_head = if source
        .tensors
        .iter()
        .any(|row| row.entry.name == "lm_head.weight" || row.entry.name == "output.weight")
    {
        OutputHead::Separate
    } else {
        OutputHead::TiedToEmbedding
    };
    let entries: Vec<_> = source.tensors.iter().map(|row| row.entry.clone()).collect();
    std::fs::create_dir_all(&request.out_dir)?;
    let config_hash = hex_sha256(&source.config_bytes);
    let census = format_census(&source.tensors);
    let census_hash = hex_sha256(census.as_bytes());
    let plan_text = format!("{plan:#?}\n");
    let plan_hash = hex_sha256(plan_text.as_bytes());
    let rewrites = memra_gguf::execution_manifest::execution_rewrites(&plan);
    debug_assert!(
        rewrites
            .iter()
            .all(|rewrite| rewrite.plan_sha256 == plan_hash)
    );
    let rewrite_manifest = format_execution_rewrites(&rewrites);
    let rewrite_hash = hex_sha256(rewrite_manifest.as_bytes());
    write_atomic(
        &request.out_dir.join("tensor-census.tsv"),
        census.as_bytes(),
    )?;
    write_atomic(
        &request.out_dir.join("model-plan.txt"),
        plan_text.as_bytes(),
    )?;
    write_atomic(
        &request.out_dir.join("execution-rewrites.tsv"),
        rewrite_manifest.as_bytes(),
    )?;
    let binding_error = match pack.compile_tensor_contract(
        &source.config,
        &plan,
        source.dialect,
        ContractOptions { output_head },
    ) {
        Ok(contract) => contract.bind(&entries).err().map(|error| error.to_string()),
        Err(error) => Some(error.to_string()),
    };
    let tokenizer_error = match &source.tokenizer {
        Ok(evidence) if pack.tokenizer_sources.contains(&evidence.source) => None,
        Ok(evidence) => Some(format!(
            "model pack {} does not accept tokenizer source {:?}",
            pack.family, evidence.source
        )),
        Err(error) => Some(error.clone()),
    };
    if let Ok(evidence) = &source.tokenizer {
        write_atomic(
            &request.out_dir.join("tokenizer-contract.tsv"),
            format!(
                "status\tpassed\nsource\t{:?}\ntokenizer_sha256\t{}\ntemplate_sha256\t{}\ntemplate_bytes\t{}\n",
                evidence.source,
                evidence.tokenizer_sha256,
                evidence.template_sha256,
                evidence.template_bytes,
            )
            .as_bytes(),
        )?;
    }
    write_atomic(
        &request.out_dir.join("artifact.lock"),
        format_lock(
            pack,
            &source,
            &config_hash,
            &census_hash,
            &plan_hash,
            &rewrite_hash,
            if binding_error.is_some() {
                "failed"
            } else {
                "passed"
            },
        )
        .as_bytes(),
    )?;
    let error_path = request.out_dir.join("contract-error.txt");
    let tokenizer_error_path = request.out_dir.join("tokenizer-error.txt");
    let mut passed = vec![Gate::Config];
    let mut failed = Vec::new();
    if tokenizer_error.is_some() {
        failed.push(Gate::TokenizerTemplate);
    } else {
        passed.push(Gate::TokenizerTemplate);
    }
    if binding_error.is_some() {
        failed.push(Gate::TensorCensus);
    } else {
        passed.push(Gate::TensorCensus);
    }
    write_atomic(
        &request.out_dir.join("gates.txt"),
        format_gate_results_with_receipts(pack, &request.out_dir, &passed, &failed).as_bytes(),
    )?;
    if let Some(error) = binding_error.as_ref() {
        write_atomic(&error_path, format!("{error}\n").as_bytes())?;
    } else {
        match std::fs::remove_file(&error_path) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(error.into()),
        }
    }
    if let Some(error) = tokenizer_error.as_ref() {
        write_atomic(&tokenizer_error_path, format!("{error}\n").as_bytes())?;
    } else {
        match std::fs::remove_file(&tokenizer_error_path) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(error.into()),
        }
    }
    match (binding_error, tokenizer_error) {
        (Some(binding), Some(tokenizer)) => {
            return Err(
                format!("tensor contract: {binding}; tokenizer contract: {tokenizer}").into(),
            );
        }
        (Some(error), None) | (None, Some(error)) => return Err(error.into()),
        (None, None) => {}
    }

    Ok(InspectSummary {
        family: pack.family,
        tensor_count: source.tensors.len(),
        out_dir: request.out_dir,
    })
}

fn load_source(source: &str) -> Result<SourceData, Box<dyn std::error::Error>> {
    let path = Path::new(source);
    if path.exists() {
        return load_local(path);
    }
    let (repo, revision) = parse_pinned_hf_source(source)?;
    load_remote(repo, revision)
}

fn load_config_only(source: &str) -> Result<ModelConfig, Box<dyn std::error::Error>> {
    let path = Path::new(source);
    if path.is_file() {
        return Ok(ModelConfig::from_gguf(&GgufFile::open(path)?));
    }
    if path.is_dir() {
        let bytes = std::fs::read(path.join("config.json"))?;
        return Ok(ModelConfig::from_hf(&HfConfig::parse(std::str::from_utf8(
            &bytes,
        )?)));
    }
    let (repo, revision) = parse_pinned_hf_source(source)?;
    let url = format!("https://huggingface.co/{repo}/resolve/{revision}/config.json");
    let config = http_text(&url)?.ok_or("pinned model has no config.json")?;
    Ok(ModelConfig::from_hf(&HfConfig::parse(&config)))
}

fn load_local(path: &Path) -> Result<SourceData, Box<dyn std::error::Error>> {
    if path.is_file() {
        let gguf = GgufFile::open(path)?;
        let tokenizer = inspect_gguf_tokenizer(&gguf);
        let config = ModelConfig::from_gguf(&gguf);
        let tensors = gguf
            .tensors
            .iter()
            .map(|tensor| CensusRow {
                physical_name: tensor.name.clone(),
                entry: TensorCensusEntry {
                    name: tensor.name.clone(),
                    shape: tensor.ne.clone(),
                    storage: ggml_storage(tensor.ggml_type),
                },
                dtype: format!("{:?}", tensor.ggml_type),
            })
            .collect();
        let config_bytes = format!("{config:#?}").into_bytes();
        return Ok(SourceData {
            label: path.display().to_string(),
            revision: "local".to_string(),
            dialect: CheckpointDialect::Gguf,
            config,
            config_bytes,
            tensors,
            shards: (0..gguf.n_shards())
                .map(|index| gguf.shard_path(index).display().to_string())
                .collect(),
            tokenizer,
        });
    }

    let config_bytes = std::fs::read(path.join("config.json"))?;
    let config_text = std::str::from_utf8(&config_bytes)?;
    let config = ModelConfig::from_hf(&HfConfig::parse(config_text));
    let tokenizer = inspect_hf_tokenizer_dir(path);
    let model = StModel::open(path)?;
    let shards = local_shards(path)?;
    let revision = local_hf_revision(path, &shards).unwrap_or_else(|| "local".to_string());
    let headers = model
        .names()
        .map(|name| {
            let (info, _) = model.raw(name).expect("StModel name must resolve");
            (name.clone(), info.clone())
        })
        .collect();
    Ok(SourceData {
        label: path.display().to_string(),
        revision,
        dialect: CheckpointDialect::HfSafetensors,
        config,
        config_bytes,
        tensors: census_from_headers(headers)?,
        shards,
        tokenizer,
    })
}

fn load_remote(repo: &str, revision: &str) -> Result<SourceData, Box<dyn std::error::Error>> {
    let base = format!("https://huggingface.co/{repo}/resolve/{revision}");
    let config_bytes = http_text(&format!("{base}/config.json"))?
        .ok_or("pinned model has no config.json")?
        .into_bytes();
    let config = ModelConfig::from_hf(&HfConfig::parse(std::str::from_utf8(&config_bytes)?));
    let tokenizer = inspect_remote_hf_tokenizer(&base);
    let index = http_text(&format!("{base}/model.safetensors.index.json"))?;
    let shards: Vec<String> = if let Some(index) = index {
        let mut files: Vec<_> = parse_index_json(&index)?.into_values().collect();
        files.sort();
        files.dedup();
        files
    } else {
        vec!["model.safetensors".to_string()]
    };
    let mut headers = BTreeMap::new();
    for shard in &shards {
        validate_remote_filename(shard)?;
        let url = format!("{base}/{shard}");
        let prefix = http_range(&url, 0, 7)?;
        if prefix.len() != 8 {
            return Err(format!("{shard}: expected 8-byte safetensors prefix").into());
        }
        let header_len = u64::from_le_bytes(prefix.try_into().unwrap()) as usize;
        if header_len == 0 || header_len > MAX_TEXT_BYTES {
            return Err(format!("{shard}: invalid safetensors header length {header_len}").into());
        }
        let bytes = http_range(&url, 8, 7 + header_len)?;
        if bytes.len() != header_len {
            return Err(format!(
                "{shard}: range returned {} header bytes, expected {header_len}",
                bytes.len()
            )
            .into());
        }
        let parsed = parse_header(std::str::from_utf8(&bytes)?)?;
        for (name, info) in parsed {
            if headers.insert(name.clone(), info).is_some() {
                return Err(format!("tensor {name} occurs in multiple safetensors shards").into());
            }
        }
    }
    Ok(SourceData {
        label: repo.to_string(),
        revision: revision.to_string(),
        dialect: CheckpointDialect::HfSafetensors,
        config,
        config_bytes,
        tensors: census_from_headers(headers)?,
        shards,
        tokenizer,
    })
}

fn inspect_gguf_tokenizer(gguf: &GgufFile) -> Result<TokenizerEvidence, String> {
    let model = gguf
        .metadata
        .get("tokenizer.ggml.model")
        .and_then(|value| value.as_str())
        .ok_or_else(|| "GGUF is missing tokenizer.ggml.model".to_string())?;
    let tokens = gguf
        .metadata
        .get("tokenizer.ggml.tokens")
        .and_then(|value| value.as_str_array())
        .ok_or_else(|| "GGUF is missing tokenizer.ggml.tokens".to_string())?;
    if tokens.is_empty() {
        return Err("GGUF tokenizer.ggml.tokens is empty".to_string());
    }
    let template = gguf
        .metadata
        .get("tokenizer.chat_template")
        .and_then(|value| value.as_str())
        .filter(|template| !template.trim().is_empty())
        .ok_or_else(|| "GGUF is missing tokenizer.chat_template".to_string())?;
    let mut hasher = Sha256::new();
    hasher.update(model.as_bytes());
    if let Some(pre) = gguf
        .metadata
        .get("tokenizer.ggml.pre")
        .and_then(|value| value.as_str())
    {
        hasher.update([0]);
        hasher.update(pre.as_bytes());
    }
    for token in tokens {
        hasher.update([0]);
        hasher.update(token.as_bytes());
    }
    Ok(TokenizerEvidence {
        source: TokenizerSource::GgufMetadata,
        tokenizer_sha256: hasher
            .finalize()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect(),
        template_sha256: hex_sha256(template.as_bytes()),
        template_bytes: template.len(),
    })
}

fn inspect_hf_tokenizer_dir(path: &Path) -> Result<TokenizerEvidence, String> {
    let tokenizer_path = path.join("tokenizer.json");
    let tokenizer = std::fs::read(&tokenizer_path)
        .map_err(|error| format!("read {}: {error}", tokenizer_path.display()))?;
    let template = local_hf_template(path)?;
    Ok(TokenizerEvidence {
        source: TokenizerSource::TokenizerJson,
        tokenizer_sha256: hex_sha256(&tokenizer),
        template_sha256: hex_sha256(template.as_bytes()),
        template_bytes: template.len(),
    })
}

fn local_hf_template(path: &Path) -> Result<String, String> {
    let config_path = path.join("tokenizer_config.json");
    if let Ok(config) = std::fs::read_to_string(&config_path)
        && let Some(template) = template_from_tokenizer_config(&config)
    {
        return Ok(template);
    }
    let template_path = path.join("chat_template.jinja");
    std::fs::read_to_string(&template_path)
        .map_err(|error| format!("read {}: {error}", template_path.display()))
        .and_then(nonempty_template)
}

fn inspect_remote_hf_tokenizer(base: &str) -> Result<TokenizerEvidence, String> {
    let tokenizer = http_text(&format!("{base}/tokenizer.json"))
        .map_err(|error| error.to_string())?
        .ok_or_else(|| "pinned HF model has no tokenizer.json".to_string())?;
    let config =
        http_text(&format!("{base}/tokenizer_config.json")).map_err(|error| error.to_string())?;
    let template = config
        .as_deref()
        .and_then(template_from_tokenizer_config)
        .or_else(|| {
            http_text(&format!("{base}/chat_template.jinja"))
                .ok()
                .flatten()
        })
        .ok_or_else(|| {
            "pinned HF model has neither tokenizer_config chat_template nor chat_template.jinja"
                .to_string()
        })
        .and_then(nonempty_template)?;
    Ok(TokenizerEvidence {
        source: TokenizerSource::TokenizerJson,
        tokenizer_sha256: hex_sha256(tokenizer.as_bytes()),
        template_sha256: hex_sha256(template.as_bytes()),
        template_bytes: template.len(),
    })
}

fn template_from_tokenizer_config(config: &str) -> Option<String> {
    let config = memra_gguf::config::JsonObj::parse(config);
    config
        .string("chat_template")
        .filter(|value| !value.trim().is_empty())
}

fn nonempty_template(template: String) -> Result<String, String> {
    if template.trim().is_empty() {
        Err("chat template is empty".to_string())
    } else {
        Ok(template)
    }
}

fn census_from_headers(
    headers: BTreeMap<String, StInfo>,
) -> Result<Vec<CensusRow>, Box<dyn std::error::Error>> {
    let mut auxiliary_names = BTreeSet::new();
    let mut rows = Vec::new();
    for (physical_name, info) in &headers {
        if auxiliary_names.contains(physical_name) || is_quant_auxiliary(physical_name, &headers) {
            continue;
        }
        let stem = physical_name.strip_suffix(".weight");
        let auxiliaries: Vec<String> = stem
            .map(|stem| {
                [
                    format!("{stem}.weight_scale"),
                    format!("{stem}.weight_scale_inv"),
                    format!("{stem}.weight_scale_2"),
                    format!("{stem}.input_scale"),
                    format!("{stem}.scale"),
                ]
                .into_iter()
                .filter(|name| headers.contains_key(name))
                .collect()
            })
            .unwrap_or_default();
        auxiliary_names.extend(auxiliaries.iter().cloned());
        let (shape, storage) = st_storage(info, &auxiliaries)?;
        rows.push(CensusRow {
            physical_name: physical_name.clone(),
            entry: TensorCensusEntry {
                name: canonical_hf_name(physical_name),
                shape,
                storage: match storage {
                    StorageLayout::Quantized(mut layout) => {
                        layout.auxiliaries = auxiliaries
                            .iter()
                            .map(|name| canonical_hf_name(name))
                            .collect();
                        StorageLayout::Quantized(layout)
                    }
                    other => other,
                },
            },
            dtype: info.dtype.clone(),
        });
    }
    rows.sort_by(|left, right| left.entry.name.cmp(&right.entry.name));
    let mut names = BTreeSet::new();
    for row in &rows {
        if !names.insert(&row.entry.name) {
            return Err(
                format!("multiple physical tensors normalize to {}", row.entry.name).into(),
            );
        }
    }
    Ok(rows)
}

fn st_storage(
    info: &StInfo,
    auxiliaries: &[String],
) -> Result<(Vec<u64>, StorageLayout), Box<dyn std::error::Error>> {
    let float = match info.dtype.as_str() {
        "F32" => Some(FloatType::F32),
        "F16" => Some(FloatType::F16),
        "BF16" => Some(FloatType::Bf16),
        "F8_E4M3" if auxiliaries.is_empty() => Some(FloatType::Fp8E4m3),
        _ => None,
    };
    if let Some(float) = float {
        return Ok((info.shape.clone(), StorageLayout::Float(float)));
    }
    if info.dtype == "I64" && auxiliaries.is_empty() {
        return Ok((info.shape.clone(), StorageLayout::Integer(IntegerType::I64)));
    }
    if auxiliaries.is_empty() {
        return Err(format!("unsupported standalone safetensors dtype {}", info.dtype).into());
    }
    let mut shape = info.shape.clone();
    let (format, block_shape) = match info.dtype.as_str() {
        "U8" => {
            let last = shape
                .last_mut()
                .ok_or("packed U8 weight has no dimensions")?;
            *last *= 2;
            ("NVFP4", vec![16])
        }
        "I8" => {
            let last = shape
                .last_mut()
                .ok_or("packed I8 weight has no dimensions")?;
            *last *= 2;
            ("MXFP4", vec![32])
        }
        "F8_E4M3" => ("FP8_E4M3", vec![128, 128]),
        other => return Err(format!("unsupported quantized weight dtype {other}").into()),
    };
    Ok((
        shape,
        StorageLayout::Quantized(QuantLayout {
            format: format.to_string(),
            block_shape,
            auxiliaries: Vec::new(),
        }),
    ))
}

fn is_quant_auxiliary(name: &str, headers: &BTreeMap<String, StInfo>) -> bool {
    for suffix in [
        ".weight_scale",
        ".weight_scale_inv",
        ".weight_scale_2",
        ".input_scale",
        ".scale",
    ] {
        if let Some(stem) = name.strip_suffix(suffix) {
            if headers.contains_key(&format!("{stem}.weight")) {
                return true;
            }
        }
    }
    false
}

fn canonical_hf_name(name: &str) -> String {
    if let Some(suffix) = name.strip_prefix("model.language_model.") {
        return format!("model.{suffix}");
    }
    if let Some(suffix) = name.strip_prefix("language_model.model.") {
        return format!("model.{suffix}");
    }
    if let Some(suffix) = name.strip_prefix("language_model.lm_head.") {
        return format!("lm_head.{suffix}");
    }
    name.to_string()
}

fn ggml_storage(kind: GgmlType) -> StorageLayout {
    match kind {
        GgmlType::F32 => StorageLayout::Float(FloatType::F32),
        GgmlType::F16 => StorageLayout::Float(FloatType::F16),
        GgmlType::BF16 => StorageLayout::Float(FloatType::Bf16),
        GgmlType::I64 => StorageLayout::Integer(IntegerType::I64),
        other => {
            let (block, _) = other.block_and_type_size();
            StorageLayout::Quantized(QuantLayout {
                format: format!("{other:?}"),
                block_shape: vec![block as u32],
                auxiliaries: Vec::new(),
            })
        }
    }
}

fn parse_pinned_hf_source(source: &str) -> Result<(&str, &str), Box<dyn std::error::Error>> {
    let (repo, revision) = source
        .rsplit_once('@')
        .ok_or("remote sources must be pinned as hf-id@40-char-sha")?;
    if repo.split('/').count() != 2
        || repo.split('/').any(|part| part.is_empty())
        || !repo
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'/' | b'-' | b'_' | b'.'))
        || repo.contains("..")
    {
        return Err("HF model id must be namespace/repository".into());
    }
    if revision.len() != 40 || !revision.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err("HF revision must be a full 40-character commit SHA".into());
    }
    Ok((repo, revision))
}

fn validate_family_name(family: &str) -> Result<(), Box<dyn std::error::Error>> {
    if family.is_empty()
        || !family
            .bytes()
            .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'_')
    {
        return Err(
            "family must contain only lowercase ASCII letters, digits, and underscores".into(),
        );
    }
    Ok(())
}

fn validate_remote_filename(name: &str) -> Result<(), Box<dyn std::error::Error>> {
    if name.is_empty()
        || name.starts_with('/')
        || name.split('/').any(|part| part.is_empty() || part == "..")
        || !name
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'/' | b'-' | b'_' | b'.'))
    {
        return Err(format!("unsafe shard filename {name:?}").into());
    }
    Ok(())
}

fn http_text(url: &str) -> Result<Option<String>, Box<dyn std::error::Error>> {
    let mut command = curl_command();
    command.args([
        "--silent",
        "--show-error",
        "--location",
        "--max-filesize",
        &MAX_TEXT_BYTES.to_string(),
        "--write-out",
        "\n%{http_code}",
        url,
    ]);
    let output = curl_output(command)?;
    if !output.status.success() {
        return Err(format!(
            "curl failed for {url}: {}",
            String::from_utf8_lossy(&output.stderr)
        )
        .into());
    }
    let split = output
        .stdout
        .iter()
        .rposition(|byte| *byte == b'\n')
        .ok_or("curl response omitted HTTP status")?;
    let status = std::str::from_utf8(&output.stdout[split + 1..])?.trim();
    match status {
        "200" => Ok(Some(String::from_utf8(output.stdout[..split].to_vec())?)),
        "404" => Ok(None),
        other => Err(format!("HTTP {other} for {url}").into()),
    }
}

fn http_range(url: &str, start: usize, end: usize) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let mut command = curl_command();
    command.args([
        "--fail",
        "--silent",
        "--show-error",
        "--location",
        "--max-filesize",
        &MAX_TEXT_BYTES.to_string(),
        "--range",
        &format!("{start}-{end}"),
        url,
    ]);
    let output = curl_output(command)?;
    if !output.status.success() {
        return Err(format!(
            "range request failed for {url}: {}",
            String::from_utf8_lossy(&output.stderr)
        )
        .into());
    }
    Ok(output.stdout)
}

fn curl_command() -> Command {
    Command::new("curl")
}

fn curl_output(mut command: Command) -> std::io::Result<Output> {
    let token = std::env::var("HF_TOKEN").ok();
    if token.is_none() {
        return command.output();
    }
    command
        .args(["--header", "@-"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut child = command.spawn()?;
    let mut stdin = child.stdin.take().expect("piped curl stdin");
    writeln!(stdin, "Authorization: Bearer {}", token.unwrap())?;
    drop(stdin);
    child.wait_with_output()
}

fn local_shards(path: &Path) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    let index = path.join("model.safetensors.index.json");
    if index.exists() {
        let mut shards: Vec<_> = parse_index_json(&std::fs::read_to_string(index)?)?
            .into_values()
            .collect();
        shards.sort();
        shards.dedup();
        Ok(shards)
    } else {
        Ok(vec!["model.safetensors".to_string()])
    }
}

fn local_hf_revision(path: &Path, shards: &[String]) -> Option<String> {
    let metadata = path.join(".cache/huggingface/download");
    let mut files = Vec::with_capacity(shards.len() + 1);
    files.push("config.json");
    files.extend(shards.iter().map(String::as_str));
    let revisions: Option<Vec<_>> = files
        .into_iter()
        .map(|file| {
            let text = std::fs::read_to_string(metadata.join(format!("{file}.metadata"))).ok()?;
            let revision = text.lines().next()?;
            (revision.len() == 40 && revision.bytes().all(|byte| byte.is_ascii_hexdigit()))
                .then(|| revision.to_ascii_lowercase())
        })
        .collect();
    let revisions = revisions?;
    let first = revisions.first()?;
    revisions
        .iter()
        .all(|revision| revision == first)
        .then(|| first.clone())
}

fn format_census(rows: &[CensusRow]) -> String {
    let mut output = String::from("semantic_name\tphysical_name\tdtype\tshape\tstorage\n");
    for row in rows {
        writeln!(
            output,
            "{}\t{}\t{}\t{:?}\t{:?}",
            row.entry.name, row.physical_name, row.dtype, row.entry.shape, row.entry.storage
        )
        .unwrap();
    }
    output
}

fn format_execution_rewrites(
    rewrites: &[memra_gguf::execution_manifest::ExecutionRewrite],
) -> String {
    let mut output = String::from(
        "rewrite\tsurface\timplementation\tplan_sha256\teligible\tblockers\toperations\treceipt\n",
    );
    for rewrite in rewrites {
        let blockers = rewrite
            .blockers
            .iter()
            .map(|operation| format!("{operation:?}"))
            .collect::<Vec<_>>()
            .join(",");
        let mut unique_operations = Vec::new();
        for operation in &rewrite.canonical_operations {
            if !unique_operations.contains(operation) {
                unique_operations.push(*operation);
            }
        }
        let operations = unique_operations
            .iter()
            .map(|operation| format!("{operation:?}"))
            .collect::<Vec<_>>()
            .join(",");
        writeln!(
            output,
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\tpending",
            rewrite.id,
            rewrite.surface.as_str(),
            rewrite.implementation,
            rewrite.plan_sha256,
            rewrite.eligible(),
            blockers,
            operations,
        )
        .unwrap();
    }
    output
}

fn format_lock(
    pack: &ModelPack,
    source: &SourceData,
    config: &str,
    census: &str,
    plan: &str,
    rewrites: &str,
    binding: &str,
) -> String {
    let mut output = String::from("format_version=2\n");
    writeln!(output, "source={}", lock_value(&source.label)).unwrap();
    writeln!(output, "revision={}", lock_value(&source.revision)).unwrap();
    writeln!(output, "family={}", pack.family).unwrap();
    match pack.support {
        Some(support) => writeln!(output, "support={support:?}").unwrap(),
        None => writeln!(output, "support=unsupported").unwrap(),
    }
    if let Some(gate) = pack.checkpoint_parity {
        writeln!(output, "checkpoint_atol={}", gate.max_abs).unwrap();
        writeln!(output, "checkpoint_rtol={}", gate.max_rel).unwrap();
        writeln!(output, "checkpoint_require_argmax={}", gate.require_argmax).unwrap();
    }
    writeln!(output, "config_sha256={config}").unwrap();
    writeln!(output, "census_sha256={census}").unwrap();
    writeln!(output, "plan_sha256={plan}").unwrap();
    writeln!(output, "rewrite_manifest_sha256={rewrites}").unwrap();
    writeln!(output, "binding={binding}").unwrap();
    match &source.tokenizer {
        Ok(evidence) if pack.tokenizer_sources.contains(&evidence.source) => {
            writeln!(output, "tokenizer=passed").unwrap();
            writeln!(output, "tokenizer_source={:?}", evidence.source).unwrap();
            writeln!(output, "tokenizer_sha256={}", evidence.tokenizer_sha256).unwrap();
            writeln!(output, "template_sha256={}", evidence.template_sha256).unwrap();
        }
        Ok(_) | Err(_) => writeln!(output, "tokenizer=failed").unwrap(),
    }
    writeln!(output, "tensor_count={}", source.tensors.len()).unwrap();
    for shard in &source.shards {
        writeln!(output, "shard={}", lock_value(shard)).unwrap();
    }
    output
}

fn parse_header(
    json: &str,
) -> Result<std::collections::HashMap<String, StInfo>, Box<dyn std::error::Error>> {
    std::panic::catch_unwind(|| parse_header_json(json))
        .map_err(|_| "invalid safetensors header JSON".into())
}

fn parse_index_json(
    json: &str,
) -> Result<std::collections::HashMap<String, String>, Box<dyn std::error::Error>> {
    std::panic::catch_unwind(|| parse_index_weight_map_json(json))
        .map_err(|_| "invalid safetensors index JSON".into())
}

fn lock_value(value: &str) -> String {
    value
        .replace('\\', "\\\\")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
}

fn format_gates(gates: &[Gate]) -> String {
    format_gate_results(gates, &[], &[])
}

fn format_gate_results(gates: &[Gate], passed: &[Gate], failed: &[Gate]) -> String {
    let mut output = String::new();
    for gate in gates {
        let status = if passed.contains(gate) {
            "passed"
        } else if failed.contains(gate) {
            "failed"
        } else {
            "pending"
        };
        writeln!(output, "{gate:?}={status}").unwrap();
    }
    output
}

fn all_eligible_rewrites_have_receipts(out_dir: &Path) -> bool {
    let Ok(manifest) = std::fs::read_to_string(out_dir.join("execution-rewrites.tsv")) else {
        return false;
    };
    let Ok(index_text) = std::fs::read_to_string(out_dir.join("rewrite-receipts.tsv")) else {
        return false;
    };
    let mut index = BTreeMap::new();
    for line in index_text.lines().skip(1) {
        let columns: Vec<_> = line.split('\t').collect();
        if columns.len() != 4 || columns[3] != "passed" {
            return false;
        }
        index.insert(columns[0], (columns[1], columns[2]));
    }
    let mut eligible = 0usize;
    for line in manifest.lines().skip(1) {
        let columns: Vec<_> = line.split('\t').collect();
        if columns.len() != 8 {
            return false;
        }
        if columns[4] != "true" {
            continue;
        }
        eligible += 1;
        let Some(&(plan, receipt_hash)) = index.get(columns[0]) else {
            return false;
        };
        if plan != columns[3] {
            return false;
        }
        let Ok(receipt) = std::fs::read(
            out_dir
                .join("rewrite-receipts")
                .join(format!("{}.tsv", columns[0])),
        ) else {
            return false;
        };
        if hex_sha256(&receipt) != receipt_hash {
            return false;
        }
    }
    eligible > 0
}

fn format_gate_results_with_receipts(
    pack: &ModelPack,
    out_dir: &Path,
    passed: &[Gate],
    failed: &[Gate],
) -> String {
    let mut passed = passed.to_vec();
    let artifact_lock = std::fs::read(out_dir.join("artifact.lock")).ok();
    let lock_hash = artifact_lock.as_ref().map(|bytes| hex_sha256(bytes));
    if let Some(lock) = artifact_lock
        .as_deref()
        .and_then(|bytes| std::str::from_utf8(bytes).ok())
        .filter(|lock| {
            lock.lines().any(|line| line == "format_version=2")
                && lock
                    .lines()
                    .any(|line| line == format!("family={}", pack.family))
        })
    {
        for (gate, evidence) in [
            (Gate::Config, None),
            (Gate::TokenizerTemplate, Some("tokenizer=passed")),
            (Gate::TensorCensus, Some("binding=passed")),
        ] {
            if evidence.is_none_or(|line| lock.lines().any(|candidate| candidate == line))
                && !passed.contains(&gate)
                && !failed.contains(&gate)
            {
                passed.push(gate);
            }
        }
    }
    let receipt_passes = |name: &str, family_bound: bool, lock_bound: bool| {
        let Ok(receipt) = std::fs::read_to_string(out_dir.join(name)) else {
            return false;
        };
        if !receipt.lines().any(|line| line == "status\tpassed") {
            return false;
        }
        if family_bound
            && !receipt
                .lines()
                .any(|line| line == format!("family\t{}", pack.family))
        {
            return false;
        }
        if lock_bound
            && !lock_hash.as_ref().is_some_and(|hash| {
                receipt
                    .lines()
                    .any(|line| line == format!("artifact_lock_sha256\t{hash}"))
            })
        {
            return false;
        }
        true
    };
    for (gate, name, family_bound, lock_bound) in [
        (Gate::TinyParity, "tiny-gate.tsv", true, false),
        (Gate::CheckpointParity, "checkpoint-parity.tsv", false, true),
        (Gate::Serve, "serve-gate.tsv", true, true),
    ] {
        if receipt_passes(name, family_bound, lock_bound)
            && !passed.contains(&gate)
            && !failed.contains(&gate)
        {
            passed.push(gate);
        }
    }
    if all_eligible_rewrites_have_receipts(out_dir)
        && !passed.contains(&Gate::RewriteParity)
        && !failed.contains(&Gate::RewriteParity)
    {
        passed.push(Gate::RewriteParity);
    }
    format_gate_results(pack.gates, &passed, failed)
}

fn format_tiny_fixture(
    plan: &memra_gguf::model_plan::ModelPlan,
    fixture: &memra_reference::ReferenceFixture,
) -> String {
    let mut output = format!("tokens={:?}\nplan={plan:#?}\n", fixture.token_ids);
    for (id, tensor) in &fixture.weights {
        let mut bytes = Vec::with_capacity(tensor.data.len() * 4);
        for value in &tensor.data {
            bytes.extend_from_slice(&value.to_bits().to_le_bytes());
        }
        writeln!(
            output,
            "tensor={id:?}\tshape={:?}\tsha256={}",
            tensor.shape,
            hex_sha256(&bytes)
        )
        .unwrap();
    }
    if let Some(vision) = fixture.vision.as_ref() {
        let mut bytes = Vec::with_capacity(vision.patches.data.len() * 4);
        for value in &vision.patches.data {
            bytes.extend_from_slice(&value.to_bits().to_le_bytes());
        }
        writeln!(
            output,
            "vision_patches={:?}\tsha256={}\tpositions={:?}\toutput_tokens={}",
            vision.patches.shape,
            hex_sha256(&bytes),
            vision.positions,
            vision.output_tokens,
        )
        .unwrap();
    }
    if let Some(token_ids) = fixture.multimodal_token_ids.as_ref() {
        writeln!(output, "multimodal_tokens={token_ids:?}").unwrap();
    }
    output
}

fn format_reference_oracle(output: &memra_reference::ReferenceOutput) -> String {
    let mut text = String::from("stream\tposition\ttoken\tlogit_f32_bits\n");
    append_oracle_rows(
        &mut text,
        "main",
        &output.logits,
        output.tokens,
        output.vocab,
    );
    for mtp in &output.mtp {
        append_oracle_rows(
            &mut text,
            &format!("mtp:{}", mtp.depth),
            &mtp.logits,
            output.tokens,
            output.vocab,
        );
    }
    if let Some(draft) = output.draft.as_ref() {
        append_oracle_rows(
            &mut text,
            "dspark",
            &draft.logits,
            draft.block_size,
            output.vocab,
        );
        for (position, (&token, &confidence)) in draft
            .output_ids
            .iter()
            .skip(1)
            .zip(&draft.confidence)
            .enumerate()
        {
            writeln!(
                text,
                "dspark-confidence\t{position}\t{token}\t{:08x}",
                confidence.to_bits()
            )
            .unwrap();
        }
    }
    text
}

fn append_oracle_rows(
    text: &mut String,
    stream: &str,
    logits: &[f32],
    tokens: usize,
    vocab: usize,
) {
    for position in 0..tokens {
        for token in 0..vocab {
            writeln!(
                text,
                "{stream}\t{position}\t{token}\t{:08x}",
                logits[position * vocab + token].to_bits()
            )
            .unwrap();
        }
    }
}

fn format_reference_vision_oracle(output: &memra_reference::ReferenceVisionOutput) -> String {
    let mut text = String::from("stream\tposition\tchannel\tf32_bits\n");
    for (stream, values, rows, width) in [
        (
            "vision-encoder",
            output.encoder_hidden.as_slice(),
            output.patch_count,
            output.hidden_size,
        ),
        (
            "vision-pooled",
            output.pooled_hidden.as_slice(),
            output.output_tokens,
            output.hidden_size,
        ),
        (
            "vision-projected",
            output.projected_hidden.as_slice(),
            output.output_tokens,
            output.projection_size,
        ),
    ] {
        for position in 0..rows {
            for channel in 0..width {
                writeln!(
                    text,
                    "{stream}\t{position}\t{channel}\t{:08x}",
                    values[position * width + channel].to_bits()
                )
                .unwrap();
            }
        }
    }
    text
}

fn hex_sha256(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn write_atomic(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    let temporary = path.with_extension(format!("tmp-{}", std::process::id()));
    std::fs::write(&temporary, bytes)?;
    std::fs::rename(temporary, path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn local_glm_fixture_generates_deterministic_onboarding_artifacts() {
        let root = std::env::temp_dir().join(format!("memra-cli-inspect-{}", std::process::id()));
        let model = root.join("model.gguf");
        let output = root.join("out");
        std::fs::create_dir_all(&root).unwrap();
        memra_gguf::micro_gguf::write_glm_dsa_micro(&model, 0x434c_4901).unwrap();
        let verified = verify_model(VerifyRequest {
            stage: VerifyStage::Config,
            source: model.display().to_string(),
            against: "glm_dsa".to_string(),
            out_dir: None,
            oracle: None,
            native_runner: None,
        })
        .unwrap();
        assert_eq!(verified.stage, VerifyStage::Config);
        let first = inspect_model(InspectRequest {
            source: model.display().to_string(),
            against: "glm_dsa".to_string(),
            out_dir: output.clone(),
        })
        .unwrap();
        assert_eq!(first.family, "glm_dsa");
        assert!(first.tensor_count > 0);
        let lock = std::fs::read(output.join("artifact.lock")).unwrap();
        inspect_model(InspectRequest {
            source: model.display().to_string(),
            against: "glm_dsa".to_string(),
            out_dir: output.clone(),
        })
        .unwrap();
        assert_eq!(std::fs::read(output.join("artifact.lock")).unwrap(), lock);
        for artifact in [
            "artifact.lock",
            "tensor-census.tsv",
            "model-plan.txt",
            "execution-rewrites.tsv",
            "gates.txt",
        ] {
            assert!(output.join(artifact).is_file(), "missing {artifact}");
        }
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn pinned_source_and_wrapper_normalization_fail_closed() {
        assert!(parse_pinned_hf_source("org/model@main").is_err());
        let sha = "a".repeat(40);
        assert_eq!(
            parse_pinned_hf_source(&format!("org/model@{sha}")).unwrap(),
            ("org/model", sha.as_str())
        );
        assert_eq!(
            canonical_hf_name("model.language_model.layers.1.self_attn.q_proj.weight"),
            "model.layers.1.self_attn.q_proj.weight"
        );
    }

    #[test]
    fn scaffold_is_deterministic_and_refuses_non_empty_targets() {
        let root = std::env::temp_dir().join(format!("memra-cli-scaffold-{}", std::process::id()));
        scaffold_model_pack(ScaffoldRequest {
            family: "new_family".to_string(),
            out_dir: root.clone(),
        })
        .unwrap();
        for artifact in [
            "pack.toml",
            "aliases.txt",
            "config-normalization.txt",
            "tensor-schema.tsv",
            "tokenizer-template.txt",
            "gates.txt",
        ] {
            assert!(root.join(artifact).is_file(), "missing {artifact}");
        }
        assert!(
            scaffold_model_pack(ScaffoldRequest {
                family: "new_family".to_string(),
                out_dir: root.clone(),
            })
            .is_err()
        );
        assert!(validate_family_name("Bad-Family").is_err());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn unimplemented_verify_stages_refuse_without_fallback() {
        for stage in [VerifyStage::Serve] {
            let error = verify_model(VerifyRequest {
                stage,
                source: "unused".to_string(),
                against: "qwen3".to_string(),
                out_dir: None,
                oracle: None,
                native_runner: None,
            })
            .err()
            .unwrap()
            .to_string();
            assert!(error.contains("no fallback is allowed"));
        }
    }

    #[test]
    fn supported_packs_write_deterministic_native_oracles() {
        let root = std::env::temp_dir().join(format!("memra-cli-tiny-{}", std::process::id()));
        let request = || VerifyRequest {
            stage: VerifyStage::Tiny,
            source: "unused".to_string(),
            against: "qwen3".to_string(),
            out_dir: Some(root.clone()),
            oracle: None,
            native_runner: None,
        };
        verify_model(request()).unwrap();
        let fixture = std::fs::read(root.join("tiny-fixture.txt")).unwrap();
        let oracle = std::fs::read(root.join("reference-oracle.tsv")).unwrap();
        verify_model(request()).unwrap();
        assert_eq!(
            std::fs::read(root.join("tiny-fixture.txt")).unwrap(),
            fixture
        );
        assert_eq!(
            std::fs::read(root.join("reference-oracle.tsv")).unwrap(),
            oracle
        );
        for pack in model_packs::PACKS {
            if pack.family == "qwen3" {
                continue;
            }
            let result = verify_model(VerifyRequest {
                stage: VerifyStage::Tiny,
                source: "unused".to_string(),
                against: pack.family.to_string(),
                out_dir: Some(root.join(pack.family)),
                oracle: None,
                native_runner: None,
            });
            if pack.support.is_some() {
                result.unwrap();
                assert!(
                    root.join(pack.family)
                        .join("reference-oracle.tsv")
                        .is_file()
                );
                if pack.family.starts_with("gemma4") {
                    assert!(
                        root.join(pack.family)
                            .join("reference-vision-oracle.tsv")
                            .is_file()
                    );
                    assert!(
                        root.join(pack.family)
                            .join("reference-multimodal-oracle.tsv")
                            .is_file()
                    );
                }
            } else {
                assert!(result.is_err());
            }
        }
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn checkpoint_oracle_bundle_is_pinned_and_parity_is_fail_closed() {
        let root = std::env::temp_dir().join(format!(
            "memra-cli-checkpoint-oracle-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&root).unwrap();
        let sha = "0123456789abcdef0123456789abcdef01234567";
        write_hf_oracle_bundle(&format!("org/model@{sha}"), &root).unwrap();
        let request = std::fs::read_to_string(root.join("oracle-request.tsv")).unwrap();
        let script = std::fs::read_to_string(root.join("capture-hf-oracle.py")).unwrap();
        assert!(request.contains(&format!("revision\t{sha}")));
        assert!(script.contains(&format!("REVISION = \"{sha}\"")));
        assert!(script.contains("trust_remote_code=False"));
        assert!(script.contains("dtype=torch.float32"));
        assert!(script.contains("source-weights-float32-accumulation"));

        let oracle = |engine: &str, values: &[f32]| {
            let mut text = format!(
                "format\tmemra-checkpoint-oracle-v1\nengine\t{engine}\nnumeric_class\tsource-weights-float32-accumulation\ntokens\t1,2,3,4\nvocab\t{}\n",
                values.len()
            );
            for (index, value) in values.iter().enumerate() {
                writeln!(text, "logit\t{index}\t{:08x}", value.to_bits()).unwrap();
            }
            parse_checkpoint_oracle(&text).unwrap()
        };
        let reference = oracle("hf-transformers", &[0.0, 1.0, -1.0]);
        let native = oracle("memra-native", &[0.0, 1.001, -1.001]);
        let gate = model_packs::CheckpointParityGate {
            max_abs: 0.01,
            max_rel: 2.0,
            require_argmax: true,
        };
        assert!(compare_checkpoint_oracles(&reference, &native, gate).is_ok());
        let failing = oracle("memra-native", &[2.0, 1.0, -1.0]);
        assert!(compare_checkpoint_oracles(&reference, &failing, gate).is_err());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn rewrite_verifier_binds_manifest_plan_and_exact_streams() {
        let root =
            std::env::temp_dir().join(format!("memra-cli-rewrite-receipt-{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        let config = ModelConfig::from_hf(&HfConfig::parse(
            r#"{"model_type":"qwen3","num_hidden_layers":2,"hidden_size":64,
            "num_attention_heads":2,"num_key_value_heads":1,"head_dim":32,
            "intermediate_size":128,"vocab_size":16,"max_position_embeddings":128}"#,
        ));
        let plan = memra_gguf::model_plan::ModelPlan::compile(&config).unwrap();
        let rewrites = memra_gguf::execution_manifest::execution_rewrites(&plan);
        let rewrite = rewrites
            .iter()
            .find(|rewrite| {
                rewrite.surface == memra_gguf::execution_manifest::RewriteSurface::DecodeBatch
            })
            .unwrap();
        let artifact_lock = b"format_version=2\nfamily=qwen3\n";
        std::fs::write(root.join("artifact.lock"), artifact_lock).unwrap();
        std::fs::write(
            root.join("execution-rewrites.tsv"),
            format_execution_rewrites(&rewrites),
        )
        .unwrap();
        let receipt = rewrite
            .verify_logits(
                &"00".repeat(32),
                &[0.0, 1.0, -1.0],
                &[0.0, 1.0, -1.0],
                memra_gguf::execution_manifest::RewriteParityPolicy {
                    max_abs: 0.0,
                    max_rel: 0.0,
                    require_argmax: true,
                },
            )
            .unwrap()
            .bind_artifact_lock(artifact_lock)
            .to_tsv();
        let receipt_path = root.join("receipt.tsv");
        std::fs::write(&receipt_path, &receipt).unwrap();
        verify_rewrite_receipt(
            model_packs::by_alias("qwen3").unwrap(),
            &receipt_path,
            &root,
        )
        .unwrap();
        assert!(
            std::fs::read_to_string(root.join("rewrite-receipts.tsv"))
                .unwrap()
                .contains("decode-batch.v1")
        );

        let wrong = receipt.replace(&rewrite.plan_sha256, &"11".repeat(32));
        std::fs::write(&receipt_path, wrong).unwrap();
        assert!(
            verify_rewrite_receipt(
                model_packs::by_alias("qwen3").unwrap(),
                &receipt_path,
                &root,
            )
            .is_err()
        );
        std::fs::remove_dir_all(root).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn native_serve_gate_launches_readiness_and_completion_on_real_http() {
        use std::os::unix::fs::PermissionsExt;

        let root =
            std::env::temp_dir().join(format!("memra-cli-serve-gate-{}", std::process::id()));
        let model = root.join("model");
        std::fs::create_dir_all(&model).unwrap();
        let artifact_lock = format!(
            "source={}\nbinding=passed\ntokenizer=passed\n",
            lock_value(model.to_str().unwrap())
        );
        std::fs::write(root.join("artifact.lock"), &artifact_lock).unwrap();
        std::fs::write(
            root.join("checkpoint-parity.tsv"),
            format!(
                "status\tpassed\nartifact_lock_sha256\t{}\n",
                hex_sha256(artifact_lock.as_bytes())
            ),
        )
        .unwrap();
        let runner = root.join("fake-memra-server.py");
        std::fs::write(
            &runner,
            r#"#!/usr/bin/env python3
import json, os
from http.server import BaseHTTPRequestHandler, HTTPServer
host, port = os.environ["MEMRA_ADDR"].rsplit(":", 1)
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args): pass
    def do_GET(self):
        self.send_response(200 if self.path == "/readyz" else 404)
        self.end_headers()
        self.wfile.write(b"ready")
    def do_POST(self):
        length = int(self.headers.get("content-length", "0"))
        self.rfile.read(length)
        body = json.dumps({"choices":[{"text":"ok"}]}).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
HTTPServer((host, int(port)), Handler).serve_forever()
"#,
        )
        .unwrap();
        let mut permissions = std::fs::metadata(&runner).unwrap().permissions();
        permissions.set_mode(0o755);
        std::fs::set_permissions(&runner, permissions).unwrap();
        verify_native_serve(
            model_packs::by_alias("qwen3").unwrap(),
            model.to_str().unwrap(),
            &root,
            &runner,
        )
        .unwrap();
        assert!(root.join("serve-response.json").is_file());
        assert!(
            std::fs::read_to_string(root.join("gates.txt"))
                .unwrap()
                .contains("Serve=passed")
        );
        std::fs::remove_dir_all(root).unwrap();
    }
}
