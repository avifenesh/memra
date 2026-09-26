//! Pinned Xiaomi MiMo layer 1 router component gate. Requires the complete source
//! checkpoint and a GPU. This does not run an MoE layer or qualify serving.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::io::Write as _;
use std::path::Path;

use memra_engine::Engine;
use memra_gguf::config::{HfConfig, ModelConfig};
use memra_gguf::model_packs::mimo_v2::inspect_pinned_source_headers;
use memra_gguf::model_plan::{MlpPlan, ModelPlan, RouterPlan};
use memra_gguf::safetensors::StModel;
use sha2::{Digest, Sha256};

type Fail = Box<dyn std::error::Error>;

const SOURCE_REVISION: &str = "3b38d063180c3e4aed9691fdc735f3d10b266ee4";
const CONFIG_SHA256: &str = "61bea4a0f7a0dd8969f8cae528761e26b697dd12ff63e98804c3f0945492e621";
const HIDDEN: usize = 4096;
const EXPERTS: usize = 256;
const TOP_K: usize = 8;
const LOGIT_TOL: f32 = 5e-4;
const WEIGHT_TOL: f32 = 5e-5;
const WEIGHT_NAME: &str = "model.layers.1.mlp.gate.weight";
const BIAS_NAME: &str = "model.layers.1.mlp.gate.e_score_correction_bias";

fn read_float_tensor(
    model: &StModel,
    name: &str,
    shape: &[u64],
) -> Result<(Vec<f32>, String), Fail> {
    let (info, bytes) = model.raw(name).ok_or_else(|| format!("{name} missing"))?;
    if info.shape != shape {
        return Err(format!("{name}: shape {:?} != {shape:?}", info.shape).into());
    }
    let count = shape.iter().product::<u64>() as usize;
    let element_bytes = match info.dtype.as_str() {
        "BF16" => 2,
        "F32" => 4,
        other => return Err(format!("{name}: unsupported dtype {other}").into()),
    };
    if bytes.len() != count * element_bytes {
        return Err(format!("{name}: byte extent does not match dtype and shape").into());
    }
    let values = match info.dtype.as_str() {
        "BF16" => bytes
            .chunks_exact(2)
            .map(|pair| f32::from_bits(u32::from(u16::from_le_bytes([pair[0], pair[1]])) << 16))
            .collect::<Vec<_>>(),
        "F32" => bytes
            .chunks_exact(4)
            .map(|quad| f32::from_le_bytes([quad[0], quad[1], quad[2], quad[3]]))
            .collect::<Vec<_>>(),
        _ => unreachable!("dtype was checked above"),
    };
    if values.len() != count || values.iter().any(|value| !value.is_finite()) {
        return Err(format!("{name}: wrong byte extent or non-finite value").into());
    }
    Ok((values, info.dtype.clone()))
}

fn hidden_row() -> Vec<f32> {
    (0..HIDDEN)
        .map(|index| {
            if index.is_multiple_of(23) {
                0.0
            } else {
                ((index * 37 % 251) as i32 - 125) as f32 / 128.0
            }
        })
        .collect()
}

fn cpu_logits(hidden: &[f32], matrix: &[f32]) -> Vec<f32> {
    matrix
        .chunks_exact(hidden.len())
        .map(|row| {
            row.iter()
                .zip(hidden)
                .fold(0.0f32, |sum, (&weight, &value)| weight.mul_add(value, sum))
        })
        .collect()
}

/// Xiaomi's inference path with one group: rank by sigmoid(logit) plus bias,
/// gather the original sigmoid scores, then normalize and scale those scores.
/// Equal keys use the smaller expert id, matching Memra's device API.
fn cpu_noaux_tc(
    logits: &[f32],
    bias: &[f32],
    top_k: usize,
    normalize: bool,
    scale: f32,
) -> Result<(Vec<usize>, Vec<f32>, f32), Fail> {
    if logits.len() != bias.len()
        || top_k == 0
        || top_k >= logits.len()
        || !scale.is_finite()
        || logits.iter().chain(bias).any(|v| !v.is_finite())
    {
        return Err("invalid CPU router operands".into());
    }
    let scores = logits
        .iter()
        .map(|&logit| 1.0f32 / (1.0 + (-logit).exp()))
        .collect::<Vec<_>>();
    let mut ids = (0..logits.len()).collect::<Vec<_>>();
    ids.sort_unstable_by(|&a, &b| {
        (scores[b] + bias[b])
            .total_cmp(&(scores[a] + bias[a]))
            .then_with(|| a.cmp(&b))
    });
    let margin =
        (scores[ids[top_k - 1]] + bias[ids[top_k - 1]]) - (scores[ids[top_k]] + bias[ids[top_k]]);
    ids.truncate(top_k);
    let denominator = if normalize {
        ids.iter().map(|&id| scores[id]).sum::<f32>() + 1e-20
    } else {
        1.0
    };
    let weights = ids
        .iter()
        .map(|&id| scores[id] / denominator * scale)
        .collect();
    Ok((ids, weights, margin))
}

fn main() {
    if let Err(error) = run() {
        eprintln!("mimo_router_gate: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Fail> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if args.len() != 3 {
        return Err("usage: mimo_router_gate <source_dir> <gpu_index> <report.tsv>".into());
    }
    let source_dir = Path::new(&args[0]);
    let gpu_index: usize = args[1].parse()?;
    let report = Path::new(&args[2]);
    if report.exists() {
        return Err(format!("refusing to overwrite {}", report.display()).into());
    }

    // Admit the exact config and full official source header census before GPU
    // allocation. Header inspection does not hash or verify tensor payloads.
    let config_bytes = std::fs::read(source_dir.join("config.json"))?;
    let config_hash = format!("{:x}", Sha256::digest(&config_bytes));
    if config_hash != CONFIG_SHA256 {
        return Err(format!("MiMo source config SHA256 changed: {config_hash}").into());
    }
    let config_text = std::str::from_utf8(&config_bytes)?;
    let config = ModelConfig::from_hf(&HfConfig::try_parse(config_text)?);
    let plan = ModelPlan::compile(&config)?;
    let MlpPlan::Moe(moe) = &plan.layers[1].mlp else {
        return Err("pinned MiMo layer 1 is not MoE".into());
    };
    if plan.hidden_size as usize != HIDDEN
        || moe.expert_count as usize != EXPERTS
        || moe.experts_per_token as usize != TOP_K
        || !matches!(
            &moe.router,
            RouterPlan::Sigmoid {
                normalize_selected: true,
                scaling_factor: 1.0,
                selection_bias: true
            }
        )
        || config.mimo.as_ref().is_none_or(|mimo| {
            mimo.n_group != Some(1)
                || mimo.topk_group != Some(1)
                || mimo.topk_method.as_deref() != Some("noaux_tc")
                || mimo.scoring_func.as_deref() != Some("sigmoid")
        })
    {
        return Err("pinned MiMo layer 1 router contract changed".into());
    }
    let model = StModel::open(source_dir)?;
    let headers = model
        .names()
        .map(|name| (name.clone(), model.info(name).unwrap().clone()))
        .collect::<BTreeMap<_, _>>();
    let bound = inspect_pinned_source_headers(&config, &headers)?;
    let bound_count = bound.tensors.len();
    drop(bound);
    drop(headers);
    let (matrix, matrix_dtype) =
        read_float_tensor(&model, WEIGHT_NAME, &[EXPERTS as u64, HIDDEN as u64])?;
    let (bias, bias_dtype) = read_float_tensor(&model, BIAS_NAME, &[EXPERTS as u64])?;
    let hidden = hidden_row();
    let expected_logits = cpu_logits(&hidden, &matrix);
    let (expected_ids, expected_weights, margin) =
        cpu_noaux_tc(&expected_logits, &bias, TOP_K, true, 1.0)?;
    if expected_logits.iter().any(|value| !value.is_finite()) || margin <= 2.0 * LOGIT_TOL {
        return Err(format!("CPU logits non-finite or top-8 margin {margin} too small").into());
    }

    let engine = Engine::new(gpu_index)?;
    engine.gpu.ctx.bind_to_thread()?;
    let hidden_gpu = engine.htod(&hidden)?;
    let matrix_gpu = engine.htod(&matrix)?;
    let bias_gpu = engine.htod(&bias)?;
    let active_gpu = engine.htod_bytes(&vec![1u8; EXPERTS])?;
    let logits_gpu = engine.linear(&hidden_gpu, &matrix_gpu, 1, HIDDEN, EXPERTS)?;
    let actual_logits = engine.dtoh(&logits_gpu)?;
    if actual_logits.len() != EXPERTS || actual_logits.iter().any(|value| !value.is_finite()) {
        return Err("GPU logits wrong length or non-finite".into());
    }
    let max_logit_abs = expected_logits
        .iter()
        .zip(&actual_logits)
        .map(|(&expected, &actual)| (expected - actual).abs())
        .fold(0.0f32, f32::max);
    let (ids_gpu, weights_gpu) = engine.moe_router_sigmoid_topk(
        &logits_gpu,
        1,
        EXPERTS,
        TOP_K,
        EXPERTS,
        &bias_gpu,
        &active_gpu,
        1.0,
        true,
    )?;
    let actual_ids = engine.dtoh_i32(&ids_gpu)?;
    let actual_weights = engine.dtoh(&weights_gpu)?;
    if actual_ids.len() != TOP_K
        || actual_weights.len() != TOP_K
        || actual_weights.iter().any(|value| !value.is_finite())
    {
        return Err("GPU top-8 wrong length or non-finite".into());
    }
    let id_mismatches = expected_ids
        .iter()
        .zip(&actual_ids)
        .filter(|&(expected, actual)| *expected as i32 != *actual)
        .count();
    let max_weight_abs = expected_weights
        .iter()
        .zip(&actual_weights)
        .map(|(&expected, &actual)| (expected - actual).abs())
        .fold(0.0f32, f32::max);
    let passed = max_logit_abs <= LOGIT_TOL && id_mismatches == 0 && max_weight_abs <= WEIGHT_TOL;

    let mut receipt = String::from("format\tmemra-mimo-router-gate-v1\n");
    writeln!(receipt, "source\tXiaomiMiMo/MiMo-V2.6-Flash-RL")?;
    writeln!(receipt, "declared_source_revision\t{SOURCE_REVISION}")?;
    writeln!(receipt, "config_sha256\t{CONFIG_SHA256}")?;
    writeln!(receipt, "source_header_tensors_bound\t{bound_count}")?;
    writeln!(
        receipt,
        "payload_verification\tnot_verified_by_header_check"
    )?;
    writeln!(receipt, "gpu_index\t{gpu_index}")?;
    writeln!(receipt, "layer\t1")?;
    writeln!(receipt, "matrix_dtype\t{matrix_dtype}")?;
    writeln!(receipt, "bias_dtype\t{bias_dtype}")?;
    writeln!(receipt, "grouping\tnoaux_tc_n_group_1_topk_group_1_top8")?;
    writeln!(receipt, "logit_tolerance\t{LOGIT_TOL:.8e}")?;
    writeln!(receipt, "weight_tolerance\t{WEIGHT_TOL:.8e}")?;
    writeln!(receipt, "cpu_top8_margin\t{margin:.8e}")?;
    writeln!(receipt, "max_logit_abs\t{max_logit_abs:.8e}")?;
    writeln!(receipt, "selected_id_mismatches\t{id_mismatches}")?;
    writeln!(receipt, "max_weight_abs\t{max_weight_abs:.8e}")?;
    writeln!(receipt, "cpu_ids\t{expected_ids:?}")?;
    writeln!(receipt, "gpu_ids\t{actual_ids:?}")?;
    writeln!(receipt, "passed\t{passed}")?;
    let staged_path = report.with_extension("tsv.writing");
    let mut staged = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&staged_path)?;
    staged.write_all(receipt.as_bytes())?;
    staged.sync_all()?;
    drop(staged);
    std::fs::rename(staged_path, report)?;
    if !passed {
        return Err("layer 1 router component comparison failed; see receipt".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn correction_bias_selects_only_and_weights_use_original_sigmoid() {
        let logits = [2.0, 1.0, 0.0, -1.0];
        let bias = [0.0, 0.0, 2.0, 0.0];
        let (ids, weights, margin) = cpu_noaux_tc(&logits, &bias, 2, true, 1.0).unwrap();
        assert_eq!(ids, [2, 0]);
        assert!(margin > 0.0);
        let raw = [0.5f32, 1.0 / (1.0 + (-2.0f32).exp())];
        assert!((weights[0] - raw[0] / (raw[0] + raw[1])).abs() < 1e-7);
        assert!((weights[1] - raw[1] / (raw[0] + raw[1])).abs() < 1e-7);
        assert!((weights.iter().sum::<f32>() - 1.0).abs() < 1e-7);
    }
}
