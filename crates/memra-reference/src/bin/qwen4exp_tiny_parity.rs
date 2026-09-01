//! qwen4_exp tiny cross-oracle parity gate (CPU-only).
//!
//! Opens a tiny HF safetensors checkpoint through the REAL qwen4_exp path
//! config parse -> pack -> plan -> census-gated tensor contract bind: loads the
//! bound tensors into `ReferenceWeights` (applying the contract's declared
//! transforms plus the family (1+w) norm fold), executes `memra_reference::execute`
//! on the golden token ids, and compares per-layer wide hidden states, final
//! logits, and the MTP block against transformers goldens
//! (research/qwen4exp-bringup-20260829/tinyparity/dump-hf-goldens.py).
//!
//! Usage: qwen4exp_tiny_parity <ckpt_dir> <goldens.bin> [<goldens.bin> ...]
//!
//! Goldens container (little-endian): magic "Q48FNTP1", u32 record count, then
//! per record: u32 name_len, name utf-8, u8 dtype (0=f32 | 1=i64), u32 ndim,
//! ndim x u64 dims, raw payload.

use memra_gguf::config::{HfConfig, ModelConfig};
use memra_gguf::safetensors::StModel;
use memra_gguf::tensor_contract::{
    CheckpointDialect, ContractOptions, FloatType, IntegerType, LayerTensor, StorageLayout,
    TensorCensusEntry, TensorId, TensorMatch, TensorOwner, TensorRequirement, TensorTransform,
};
use memra_reference::{ReferenceTensor, ReferenceWeights, execute};
use std::collections::BTreeMap;

// f32 HF run vs f32 reference on identical BF16-rounded weights: the residual is
// pure op-order noise. Measured envelope on the 4 seeded probes (2026-08-29):
// max_abs 2.015e-5, max_rel 1.566e-3: thresholds carry ~2-5x headroom.
const MAX_ABS: f32 = 1e-4;
const MAX_REL: f32 = 3e-3;
const REL_FLOOR: f32 = 1e-2;

fn main() {
    let mut args = std::env::args().skip(1);
    let ckpt = args
        .next()
        .expect("usage: qwen4exp_tiny_parity <ckpt_dir> <goldens.bin>...");
    let goldens: Vec<String> = args.collect();
    assert!(
        !goldens.is_empty(),
        "usage: qwen4exp_tiny_parity <ckpt_dir> <goldens.bin>..."
    );

    let dir = std::path::Path::new(&ckpt);
    let config_json =
        std::fs::read_to_string(dir.join("config.json")).expect("checkpoint config.json");
    let cfg = ModelConfig::from_hf(&HfConfig::parse(&config_json));
    let pack = memra_gguf::model_packs::for_config(&cfg)
        .expect("qwen4_exp pack must match the tiny config");
    assert_eq!(pack.family, "qwen4_exp", "pack resolution");
    let plan = pack.compile_plan(&cfg).expect("plan compiles");
    let contract = pack
        .compile_tensor_contract(
            &cfg,
            &plan,
            CheckpointDialect::HfSafetensors,
            ContractOptions::default(),
        )
        .expect("tensor contract compiles");

    // Census straight from the safetensors header: the same evidence a real
    // artifact presents. bind() fails on missing, extra, mis-shaped, or
    // mis-typed tensors; that IS the census gate.
    let st = StModel::open(dir).expect("open safetensors");
    let census: Vec<TensorCensusEntry> = st
        .names()
        .map(|name| {
            let (info, bytes) = st.raw(name).expect("census name resolves");
            TensorCensusEntry {
                name: name.clone(),
                shape: info.shape.clone(),
                storage: match info.dtype.as_str() {
                    "BF16" => StorageLayout::Float(FloatType::Bf16),
                    "F32" => StorageLayout::Float(FloatType::F32),
                    "I64" => StorageLayout::Integer(IntegerType::I64),
                    other => panic!("unexpected tiny-checkpoint dtype {other}"),
                },
                // Exact checkpoint bytes for this row, straight from the header's
                // data_offsets extent — the same evidence a real artifact presents.
                physical_bytes: bytes.len() as u64,
            }
        })
        .collect();
    let bound = contract.bind(&census).unwrap_or_else(|error| {
        panic!("census-gated contract bind FAILED: {error:?}");
    });
    println!(
        "contract bind: {} requirements over {} checkpoint tensors: OK",
        bound.tensors.len(),
        census.len()
    );

    let weights = load_reference_weights(&st, &cfg, &contract.requirements);

    let mut failures = 0usize;
    for path in &goldens {
        let records = read_goldens(path);
        let token_ids: Vec<u32> = records
            .get("token_ids")
            .expect("goldens carry token_ids")
            .ints
            .as_ref()
            .expect("token_ids are i64")
            .iter()
            .map(|&t| u32::try_from(t).expect("token id fits u32"))
            .collect();
        println!("\n== probe {path} ({} tokens) ==", token_ids.len());
        let output = execute(&plan, &weights, &token_ids).expect("reference executes");

        for (layer, hidden) in output.layer_hidden.iter().enumerate() {
            let name = format!("layer_hidden.{layer}");
            failures += compare(&name, records.get(&name), hidden);
        }
        failures += compare("logits", records.get("logits"), &output.logits);
        if let Some(golden) = records.get("logits") {
            let vocab = output.vocab;
            let last = &output.logits[(output.tokens - 1) * vocab..];
            let golden_last = &golden.data[(output.tokens - 1) * vocab..];
            let ours = argmax(last);
            let theirs = argmax(golden_last);
            if ours != theirs {
                println!("  FAIL logits argmax(last token): ref {ours} vs hf {theirs}");
                failures += 1;
            } else {
                println!("  OK   logits argmax(last token) = {ours}");
            }
        }
        let mtp = output.mtp.first().expect("plan carries one MTP block");
        failures += compare("mtp_hidden", records.get("mtp_hidden"), &mtp.hidden);
        failures += compare("mtp_logits", records.get("mtp_logits"), &mtp.logits);
    }

    if failures > 0 {
        eprintln!("\nTINY PARITY FAILED: {failures} comparison(s) out of tolerance");
        std::process::exit(1);
    }
    println!("\nTINY PARITY PASSED (max_abs<= {MAX_ABS}, max_rel <= {MAX_REL})");
}

fn argmax(values: &[f32]) -> usize {
    let mut best = 0;
    for (index, &value) in values.iter().enumerate() {
        if value > values[best] {
            best = index;
        }
    }
    best
}

/// Compare one golden record against the reference tensor; prints the measured
/// extrema and returns 1 on a tolerance violation.
fn compare(name: &str, golden: Option<&Record>, ours: &[f32]) -> usize {
    let Some(golden) = golden else {
        println!("  SKIP {name}: no golden record");
        return 0;
    };
    if golden.data.len() != ours.len() {
        println!(
            "  FAIL {name}: golden has {} elements, reference {}",
            golden.data.len(),
            ours.len()
        );
        return 1;
    }
    let mut max_abs = 0f32;
    let mut max_rel = 0f32;
    let mut at = 0usize;
    for (index, (&theirs, &mine)) in golden.data.iter().zip(ours).enumerate() {
        let abs = (theirs - mine).abs();
        let rel = abs / theirs.abs().max(mine.abs()).max(REL_FLOOR);
        if abs > max_abs {
            max_abs = abs;
            at = index;
        }
        max_rel = max_rel.max(rel);
    }
    let ok = max_abs <= MAX_ABS && max_rel <= MAX_REL;
    println!(
        "  {} {name}: max_abs={max_abs:.3e} max_rel={max_rel:.3e} (worst at flat index {at}: hf {} vs ref {})",
        if ok { "OK  " } else { "FAIL" },
        golden.data[at],
        ours[at],
    );
    usize::from(!ok)
}

// ---------------------------------------------------------------------------
// checkpoint -> ReferenceWeights
// ---------------------------------------------------------------------------

/// Effective-weight fold: every Qwen4ExpTextRMSNorm / Qwen3_5RMSNorm /
/// GemmaRMSNorm row ships zero-centered and computes `(1+w) * x̂`
/// (modular_qwen4_exp.py L859-861 zero-init receipt; Qwen3_5RMSNorm.forward;
/// SGLang GemmaRMSNorm for the MTP glue). The reference consumes EFFECTIVE
/// weights, so +1 folds here at binding. The ONE carve-out is the GDN gated
/// output norm `linear_attn.norm.weight` (Qwen4ExpTextRMSNormGated: plain
/// `w * x̂`, ones-init: same carve-out as llama.cpp qwen.py:302-303).
fn norm_needs_plus_one(name: &str) -> bool {
    const FOLDED_SUFFIXES: &[&str] = &[
        ".hc_norm.weight",
        ".q_norm.weight",
        ".k_norm.weight",
        ".q_layernorm.weight",
        ".k_layernorm.weight",
        ".norm_key.weight",
        ".norm_query.weight",
        ".norm_conv.weight",
    ];
    if name == "mtp.pre_fc_norm_embedding.weight" || name == "mtp.pre_fc_norm_hidden.weight" {
        return true;
    }
    FOLDED_SUFFIXES.iter().any(|suffix| name.ends_with(suffix))
}

fn load_reference_weights(
    st: &StModel,
    cfg: &ModelConfig,
    requirements: &[TensorRequirement],
) -> ReferenceWeights {
    let ssm = cfg.ssm.as_ref().expect("qwen4_exp carries an ssm block");
    let nk = ssm.group_count as usize;
    let nv = ssm.time_step_rank as usize;
    let hk = ssm.state_size as usize;
    let hv = (ssm.inner_size / ssm.time_step_rank) as usize;

    let mut weights = ReferenceWeights::new();
    for requirement in requirements {
        if matches!(requirement.owner, TensorOwner::Vision(_)) {
            continue; // text-only execution: the tower is censused, not run
        }
        match requirement.match_mode {
            TensorMatch::All => {
                // The ngram shard bank: ONE semantic table, concat on dim 0 in
                // shard order (the pack sorted the names by shard index).
                let mut data = Vec::new();
                let mut rows = 0usize;
                let mut width = 0usize;
                for name in &requirement.names {
                    let (info, bytes) = st
                        .raw(name)
                        .unwrap_or_else(|| panic!("bound tensor {name} vanished"));
                    assert_eq!(info.dtype, "BF16", "{name} dtype");
                    rows += info.shape[0] as usize;
                    width = info.shape[1] as usize;
                    data.extend(bf16_to_f32(bytes));
                }
                weights.insert(
                    requirement.id.clone(),
                    ReferenceTensor::new(vec![rows, width], data).expect("ngram bank shape"),
                );
            }
            TensorMatch::OneOf => {
                let name = &requirement.names[0];
                let (info, bytes) = st
                    .raw(name)
                    .unwrap_or_else(|| panic!("bound tensor {name} vanished"));
                let shape: Vec<usize> = info.shape.iter().map(|&d| d as usize).collect();
                if info.dtype == "I64" {
                    let ints: Vec<i64> = bytes
                        .chunks_exact(8)
                        .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
                        .collect();
                    weights.insert(
                        requirement.id.clone(),
                        ReferenceTensor::new_i64(shape, ints).expect("i64 tensor shape"),
                    );
                    continue;
                }
                assert_eq!(info.dtype, "BF16", "{name} dtype");
                let mut shape = shape;
                // shared_expert_gate ships [1, hidden]; the reference consumes
                // the squeezed [hidden] row (deterministic_fixture layout).
                if matches!(
                    requirement.id,
                    TensorId::Layer {
                        tensor: LayerTensor::SharedMlpInputGate,
                        ..
                    }
                ) && shape.first() == Some(&1)
                {
                    shape.remove(0);
                }
                // The PLE depthwise conv ships [wide, 1, kernel]; the reference
                // consumes the squeezed [wide, kernel] form (contract transform
                // stays Identity for family rows: the squeeze is binding work,
                // like the GDN conv row's Conv1dSqueezeReorder without reorder).
                if name.ends_with("ple.conv1d.weight") && shape.len() == 3 && shape[1] == 1 {
                    shape.remove(1);
                }
                let mut data = bf16_to_f32(bytes);
                if norm_needs_plus_one(name) {
                    for value in &mut data {
                        *value += 1.0;
                    }
                }
                let (shape, data) =
                    apply_transform(requirement.transform, name, shape, data, nk, nv, hk, hv);
                if requirement.transform == TensorTransform::SplitExpertGateUp {
                    // [experts, 2*ff, hidden], gate rows first per expert
                    // (Qwen3NextExperts: linear(x, gate_up[e]).chunk(2)).
                    let TensorId::Layer { index, .. } = requirement.id else {
                        panic!("SplitExpertGateUp outside a layer tensor: {name}");
                    };
                    let (experts, two_ff, hidden) = (shape[0], shape[1], shape[2]);
                    let ff = two_ff / 2;
                    let mut gate = Vec::with_capacity(experts * ff * hidden);
                    let mut up = Vec::with_capacity(experts * ff * hidden);
                    for expert in 0..experts {
                        let base = expert * two_ff * hidden;
                        gate.extend_from_slice(&data[base..base + ff * hidden]);
                        up.extend_from_slice(&data[base + ff * hidden..base + two_ff * hidden]);
                    }
                    let bank_shape = vec![experts, ff, hidden];
                    weights.insert(
                        TensorId::Layer {
                            index,
                            tensor: LayerTensor::MoeExpertGateBank,
                        },
                        ReferenceTensor::new(bank_shape.clone(), gate).expect("gate bank"),
                    );
                    weights.insert(
                        TensorId::Layer {
                            index,
                            tensor: LayerTensor::MoeExpertUpBank,
                        },
                        ReferenceTensor::new(bank_shape, up).expect("up bank"),
                    );
                    continue;
                }
                weights.insert(
                    requirement.id.clone(),
                    ReferenceTensor::new(shape, data)
                        .unwrap_or_else(|e| panic!("{name} shape: {e}")),
                );
            }
        }
    }
    weights
}

/// The qwen4_exp per-layer load transforms, applied to the HF row-major f32
/// buffer. GDN reorders mirror hf_mapping.rs / llama.cpp conversion/qwen.py
/// (dst V-head `j*nk + g` <- src `g*(nv/nk) + j`), which is what makes the
/// reference's `key_head = value_head % key_heads` mapping equal HF's
/// `repeat_interleave` grouping.
#[allow(clippy::too_many_arguments)]
fn apply_transform(
    transform: TensorTransform,
    name: &str,
    shape: Vec<usize>,
    mut data: Vec<f32>,
    nk: usize,
    nv: usize,
    hk: usize,
    hv: usize,
) -> (Vec<usize>, Vec<f32>) {
    match transform {
        TensorTransform::Identity | TensorTransform::SplitExpertGateUp => (shape, data),
        TensorTransform::NormAddOne => {
            for value in &mut data {
                *value += 1.0;
            }
            (shape, data)
        }
        TensorTransform::NegExpReorderHeads => {
            for value in &mut data {
                *value = -value.exp();
            }
            (
                shape.clone(),
                reorder_rows_v(&data, nv, nk, 1, 1, 0, shape[0]),
            )
        }
        TensorTransform::ReorderHeads => (
            shape.clone(),
            reorder_rows_v(&data, nv, nk, 1, 1, 0, shape[0]),
        ),
        TensorTransform::AbReorderRows => {
            let width = shape[1];
            (
                shape.clone(),
                reorder_rows_v(&data, nv, nk, 1, width, 0, nv),
            )
        }
        TensorTransform::ZReorderRows => {
            let width = shape[1];
            let rows = shape[0];
            (
                shape.clone(),
                reorder_rows_v(&data, nv, nk, hv, width, 0, rows),
            )
        }
        TensorTransform::QkvVReorderRows => {
            let width = shape[1];
            let rows = shape[0];
            (
                shape.clone(),
                reorder_rows_v(&data, nv, nk, hv, width, 2 * nk * hk, rows),
            )
        }
        TensorTransform::Conv1dSqueezeReorder => {
            // [C, 1, K] row-major == [C][K]; squeeze, reorder the V channel band.
            assert_eq!(shape.len(), 3, "{name} conv shape");
            let channels = shape[0];
            let kernel = shape[2];
            let squeezed = vec![channels, kernel];
            (
                squeezed,
                reorder_rows_v(&data, nv, nk, hv, kernel, 2 * nk * hk, channels),
            )
        }
        TensorTransform::OutReorderColumns => {
            let rows = shape[0];
            let width = shape[1];
            (
                shape.clone(),
                reorder_cols_v(&data, rows, width, nv, nk, hv),
            )
        }
        other => panic!("unexpected transform {other:?} on {name} for qwen4_exp"),
    }
}

fn reorder_rows_v(
    data: &[f32],
    nv: usize,
    nk: usize,
    head_dim: usize,
    row_width: usize,
    row_lo: usize,
    row_hi: usize,
) -> Vec<f32> {
    assert_eq!(row_hi - row_lo, nv * head_dim, "V band width");
    let per_key = nv / nk;
    let mut out = data.to_vec();
    for j in 0..per_key {
        for g in 0..nk {
            let dst_head = j * nk + g;
            let src_head = g * per_key + j;
            for d in 0..head_dim {
                let dst = (row_lo + dst_head * head_dim + d) * row_width;
                let src = (row_lo + src_head * head_dim + d) * row_width;
                out[dst..dst + row_width].copy_from_slice(&data[src..src + row_width]);
            }
        }
    }
    out
}

fn reorder_cols_v(
    data: &[f32],
    rows: usize,
    width: usize,
    nv: usize,
    nk: usize,
    head_dim: usize,
) -> Vec<f32> {
    assert_eq!(width, nv * head_dim, "column band width");
    let per_key = nv / nk;
    let mut out = data.to_vec();
    for row in 0..rows {
        let base = row * width;
        for j in 0..per_key {
            for g in 0..nk {
                let dst_head = j * nk + g;
                let src_head = g * per_key + j;
                for d in 0..head_dim {
                    out[base + dst_head * head_dim + d] = data[base + src_head * head_dim + d];
                }
            }
        }
    }
    out
}

fn bf16_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
        .collect()
}

// ---------------------------------------------------------------------------
// goldens container
// ---------------------------------------------------------------------------

struct Record {
    data: Vec<f32>,
    ints: Option<Vec<i64>>,
}

fn read_goldens(path: &str) -> BTreeMap<String, Record> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read goldens {path}: {e}"));
    let mut cursor = 0usize;
    let take = |cursor: &mut usize, n: usize| -> &[u8] {
        let slice = &bytes[*cursor..*cursor + n];
        *cursor += n;
        slice
    };
    assert_eq!(take(&mut cursor, 8), b"Q48FNTP1", "goldens magic");
    let count = u32::from_le_bytes(take(&mut cursor, 4).try_into().unwrap()) as usize;
    let mut records = BTreeMap::new();
    for _ in 0..count {
        let name_len = u32::from_le_bytes(take(&mut cursor, 4).try_into().unwrap()) as usize;
        let name = String::from_utf8(take(&mut cursor, name_len).to_vec()).expect("record name");
        let dtype = take(&mut cursor, 1)[0];
        let ndim = u32::from_le_bytes(take(&mut cursor, 4).try_into().unwrap()) as usize;
        let mut elements = 1usize;
        for _ in 0..ndim {
            elements *= u64::from_le_bytes(take(&mut cursor, 8).try_into().unwrap()) as usize;
        }
        let record = match dtype {
            0 => Record {
                data: take(&mut cursor, elements * 4)
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                    .collect(),
                ints: None,
            },
            1 => Record {
                data: Vec::new(),
                ints: Some(
                    take(&mut cursor, elements * 8)
                        .chunks_exact(8)
                        .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
                        .collect(),
                ),
            },
            other => panic!("unknown goldens dtype tag {other}"),
        };
        records.insert(name, record);
    }
    assert_eq!(cursor, bytes.len(), "goldens trailing bytes");
    records
}
