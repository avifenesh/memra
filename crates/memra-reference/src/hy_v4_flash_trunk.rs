//! Hy4-flash (`hy_v4_flash`) whole-trunk reference: token ids to logits.
//!
//! The semantic source is transformers `modeling_hy_v4_flash.py` (PR #50903 at b865850d0532),
//! `HYV4FlashTextModel.forward` and the F32 `lm_head`: the embedding copied to every stream, each
//! decoder layer ([`ops::decoder_layer_cached`], Engram first where the plan puts it), the
//! `hc_head` collapse, `model.norm`, the head. A request runs as one prefill chunk and then one
//! teacher-forced decode step per later token over the per-layer [`ops::MlaCache`], as the vendor's
//! cached forward does; the Engram n-grams of a step reach back into the token history
//! (`prepare_inputs_for_generation`). Everything is f64; the logits round once to f32.
//!
//! The tiny fixture ([`fixture`]) is the pack's tiny config with seeded weights in the
//! artifact's storage classes: BF16, F32, the I32 lookup tables, FP8 128x128 blocks with an F32
//! `weight_scale_inv`, NVFP4 (e2m1 pairs, an e4m3 scale per 16 inputs, an F32 `weight_scale_2`)
//! and the Engram tables' FP8 rows with one F32 `weight_scale` each. The tensor list is the
//! pack's census of the text program. The fixture also writes the checkpoint image
//! (`model.safetensors`, `config.json`, `fixture.json`) the external oracle loads
//! (research/modelplan-onboarding-hy4flash-20261119/tiny/make_tiny_oracle.py).

use std::collections::{BTreeMap, BTreeSet};

use memra_gguf::config::HyV4FlashConfig;
use memra_gguf::dsv4_forward::e2m1_rne;
use memra_gguf::dsv41_forward::{bf16_bits_to_f32, e2m1, e4m3, f32_to_bf16_bits, to_e2m1, to_e4m3};
use memra_gguf::hy_v4_flash_engram as hashing;
use memra_gguf::model_packs::hy_v4_flash as pack;
use memra_gguf::model_plan::{
    AttentionPlan, HyV4AttentionKind, HyV4EngramPlan, MlpPlan, ModelPlan, ResidualTopology,
};
use memra_gguf::tensor_contract::TensorId;

use crate::hy_v4_flash as ops;
use crate::{
    ReferenceDecodeTrace, ReferenceError, ReferenceFixture, ReferenceLayerState, ReferenceOutput,
    ReferenceState, ReferenceTensor, ReferenceWeights, TinyVariant,
};

fn invalid(layer: Option<u32>, reason: &'static str) -> ReferenceError {
    ReferenceError::InvalidPlan { layer, reason }
}

/// The token count of the tiny fixture: past the 512-token window by 32 and past the indexer
/// budget of 256 by 288.
pub const TINY_TOKENS: usize = 544;
/// Where the main fixture's prefill ends: 8 decode steps follow, past both limits.
pub const TINY_PREFILL: usize = 536;

/// How a checkpoint tensor is stored.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Storage {
    Bf16,
    F32,
    I32,
    /// F8_E4M3 `weight` with an F32 `weight_scale_inv` per 128x128 block, multiplied in.
    Fp8Block,
    /// U8 e2m1 pairs (element `2i` in the low nibble), an F8_E4M3 `weight_scale` per 16 inputs
    /// and an F32 `weight_scale_2` per matrix (per expert on a stacked bank).
    Nvfp4,
    /// The Engram table: F8_E4M3 rows with one F32 `weight_scale` per row.
    Fp8Row,
}

/// One tensor the text program reads, with its logical shape (an NVFP4 weight's `in` counts
/// elements, not bytes).
#[derive(Debug, Clone, PartialEq)]
pub struct CheckpointTensor {
    pub name: String,
    pub storage: Storage,
    pub shape: Vec<usize>,
}

/// The name under which the reference weights carry one Engram layer's moduli
/// (`engram_layer_head_vocab_sizes`, order-major). Config data the hash reads, not a checkpoint
/// tensor; the hash program takes it from the config as transformers does.
pub fn engram_moduli_name(layer: u32) -> String {
    format!("model.layers.{layer}.engram_layer.head_vocab_sizes")
}

fn id(name: &str) -> TensorId {
    pack::semantic_tensor_id(name)
}

/// Every tensor of the text program in the pack's census (`expected_census`), with its storage.
/// The tower (`vit.*`) and the DFly2 drafter (`model.dfly_layers.*`) are not read.
pub fn checkpoint_tensors(hy: &HyV4FlashConfig) -> Result<Vec<CheckpointTensor>, ReferenceError> {
    let census = pack::expected_census(hy)
        .map_err(|_| invalid(None, "the hy_v4_flash census derives from the config"))?;
    let text: BTreeMap<&String, _> = census
        .iter()
        .filter(|(name, _)| !name.starts_with("vit.") && !name.starts_with("model.dfly_layers."))
        .collect();
    let mut consumed = BTreeSet::new();
    let mut out = Vec::new();
    for (name, spec) in &text {
        if name.ends_with(".weight_scale")
            || name.ends_with(".weight_scale_inv")
            || name.ends_with(".weight_scale_2")
        {
            continue;
        }
        let stem = name.strip_suffix(".weight").unwrap_or(name);
        let has = |suffix: &str| text.contains_key(&format!("{stem}.{suffix}"));
        let mut shape: Vec<usize> = spec.shape.iter().map(|v| *v as usize).collect();
        let storage = match spec.dtype {
            "BF16" => Storage::Bf16,
            "F32" => Storage::F32,
            "I32" => Storage::I32,
            "F8_E4M3" if has("weight_scale_inv") => {
                consumed.insert(format!("{stem}.weight_scale_inv"));
                Storage::Fp8Block
            }
            "F8_E4M3" if has("weight_scale") && name.contains("engram_layer") => {
                consumed.insert(format!("{stem}.weight_scale"));
                Storage::Fp8Row
            }
            "U8" if has("weight_scale") && has("weight_scale_2") => {
                consumed.insert(format!("{stem}.weight_scale"));
                consumed.insert(format!("{stem}.weight_scale_2"));
                *shape.last_mut().unwrap() *= 2;
                Storage::Nvfp4
            }
            _ => return Err(invalid(None, "a census tensor has no storage class here")),
        };
        consumed.insert((*name).clone());
        out.push(CheckpointTensor {
            name: (*name).clone(),
            storage,
            shape,
        });
    }
    if text.keys().any(|name| !consumed.contains(*name)) {
        return Err(invalid(None, "a census scale tensor has no weight"));
    }
    Ok(out)
}

// ============================ deterministic fixture ============================

fn fnv1a(name: &str) -> u64 {
    name.bytes().fold(0xcbf2_9ce4_8422_2325u64, |hash, byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x0100_0000_01b3)
    })
}

/// Seeded values: norms and gains around 1, hyper-connection functions small with scales
/// around 1, sinks around 0.5, router biases large enough to move the selection, the
/// embedding and the Engram rows at unit scale, the rest uniform with unit variance per input.
fn fixture_values(tensor: &CheckpointTensor) -> Result<Vec<f32>, ReferenceError> {
    let name = tensor.name.as_str();
    let fan_in = *tensor.shape.last().unwrap() as f32;
    let (scale, offset) = if name.ends_with("norm.weight")
        || name.ends_with("layernorm.weight")
        || name.contains("lookup_per_head_norms")
        || name.contains("norm_key.")
        || name.contains("norm_query.")
        || name.ends_with("unified_pre_gamma")
    {
        (0.25, 1.0)
    } else if name.ends_with("hc_fn") || name.ends_with("hc_head_fn") {
        (0.1, 0.0)
    } else if name.ends_with("hc_scale") || name.ends_with("hc_head_scale") {
        (0.5, 1.0)
    } else if name.ends_with("hc_base") || name.ends_with("hc_head_base") {
        (0.5, 0.0)
    } else if name.ends_with("learnable_sink_param") {
        (1.0, 0.5)
    } else if name.ends_with("e_score_correction_bias") {
        (0.2, 0.0)
    } else if name.ends_with("intra_group_ape") || name.contains("unified_per_hc_scales") {
        (0.5, 0.0)
    } else if name.ends_with(".bias") {
        (0.1, 0.0)
    } else if name == "model.embed_tokens.weight" || name.contains("multi_head_embedding") {
        (1.0, 0.0)
    } else {
        ((3.0 / fan_in).sqrt(), 0.0)
    };
    Ok(crate::generated_tensor(&tensor.shape, fnv1a(name), scale)?
        .data
        .into_iter()
        .map(|value| value + offset)
        .collect())
}

/// The smallest power of two at or above `x` (`x > 0`).
fn pow2_ceil(x: f32) -> f32 {
    let mut p = 2f32.powi(x.log2().floor() as i32);
    while p < x {
        p *= 2.0;
    }
    while p / 2.0 >= x {
        p /= 2.0;
    }
    p
}

/// One safetensors entry: name, dtype, shape, bytes.
type Entry = (String, &'static str, Vec<usize>, Vec<u8>);
/// What the reference reads back from one stored tensor, by tensor id.
type ReadBack = Vec<(TensorId, ReferenceTensor)>;

fn le_f32(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// Encode seeded values the way the artifact stores the tensor, and what the reference reads
/// back. The FP8 block scales and the NVFP4 `weight_scale_2` are powers of two here, so the
/// dequantized values are exact in f32; the Engram row scales follow the artifact's producer
/// (`f32(amax) / 448`, step 5) and the reference reads codes and scales apart.
fn encode(
    tensor: &CheckpointTensor,
    values: &[f32],
) -> Result<(Vec<Entry>, ReadBack), ReferenceError> {
    let name = tensor.name.clone();
    let shape = tensor.shape.clone();
    let stem = name.strip_suffix(".weight").unwrap_or(&name).to_string();
    Ok(match tensor.storage {
        Storage::Bf16 => {
            let bits: Vec<u16> = values.iter().map(|v| f32_to_bf16_bits(*v)).collect();
            let back = bits.iter().map(|b| bf16_bits_to_f32(*b)).collect();
            (
                vec![(
                    name.clone(),
                    "BF16",
                    shape.clone(),
                    bits.iter().flat_map(|b| b.to_le_bytes()).collect(),
                )],
                vec![(id(&name), ReferenceTensor::new(shape, back)?)],
            )
        }
        Storage::F32 => (
            vec![(name.clone(), "F32", shape.clone(), le_f32(values))],
            vec![(id(&name), ReferenceTensor::new(shape, values.to_vec())?)],
        ),
        Storage::I32 => {
            let ints: Vec<i32> = values.iter().map(|v| *v as i32).collect();
            (
                vec![(
                    name.clone(),
                    "I32",
                    shape.clone(),
                    ints.iter().flat_map(|v| v.to_le_bytes()).collect(),
                )],
                vec![(
                    id(&name),
                    ReferenceTensor::new_i64(shape, ints.iter().map(|v| i64::from(*v)).collect())?,
                )],
            )
        }
        Storage::Fp8Block => {
            let (rows, cols) = (shape[0], shape[1]);
            let (grid_rows, grid_cols) = (rows.div_ceil(128), cols.div_ceil(128));
            let mut scales = vec![0f32; grid_rows * grid_cols];
            for (block, scale) in scales.iter_mut().enumerate() {
                let (br, bc) = (block / grid_cols, block % grid_cols);
                let mut amax = 0f32;
                for r in br * 128..((br + 1) * 128).min(rows) {
                    for c in bc * 128..((bc + 1) * 128).min(cols) {
                        amax = amax.max(values[r * cols + c].abs());
                    }
                }
                *scale = pow2_ceil(amax.max(1e-4) / 448.0);
            }
            let scale_at = |i: usize| scales[(i / cols / 128) * grid_cols + (i % cols) / 128];
            let codes: Vec<u8> = (0..rows * cols)
                .map(|i| to_e4m3((values[i] / scale_at(i)).clamp(-448.0, 448.0)))
                .collect();
            let back = (0..rows * cols)
                .map(|i| e4m3(codes[i]) * scale_at(i))
                .collect();
            (
                vec![
                    (name.clone(), "F8_E4M3", shape.clone(), codes),
                    (
                        format!("{stem}.weight_scale_inv"),
                        "F32",
                        vec![grid_rows, grid_cols],
                        le_f32(&scales),
                    ),
                ],
                vec![(id(&name), ReferenceTensor::new(shape, back)?)],
            )
        }
        Storage::Nvfp4 => {
            let rank = shape.len();
            let (out, input) = (shape[rank - 2], shape[rank - 1]);
            let lead: Vec<usize> = shape[..rank - 2].to_vec();
            let banks: usize = lead.iter().product();
            let per = out * input;
            let mut packed = vec![0u8; banks * per / 2];
            let mut group_scales = vec![0u8; banks * per / 16];
            let mut second = vec![0f32; banks];
            let mut back = vec![0f32; banks * per];
            for bank in 0..banks {
                let slice = &values[bank * per..(bank + 1) * per];
                let amax = slice.iter().fold(0f32, |a, v| a.max(v.abs()));
                let scale_2 = pow2_ceil(amax.max(1e-6) / (6.0 * 448.0));
                second[bank] = scale_2;
                for row in 0..out {
                    for group in 0..input / 16 {
                        let at = row * input + group * 16;
                        let g = &slice[at..at + 16];
                        let gmax = g.iter().fold(0f32, |a, v| a.max(v.abs()));
                        let code = to_e4m3(gmax / 6.0 / scale_2);
                        group_scales[(bank * per + at) / 16] = code;
                        let step = e4m3(code) * scale_2;
                        for (k, &v) in g.iter().enumerate() {
                            let nibble = if step > 0.0 {
                                to_e2m1(e2m1_rne((v / step).clamp(-6.0, 6.0)))
                            } else {
                                0
                            };
                            let i = bank * per + at + k;
                            packed[i / 2] |= if i % 2 == 0 { nibble } else { nibble << 4 };
                            back[i] = e2m1(nibble) * e4m3(code) * scale_2;
                        }
                    }
                }
            }
            let with =
                |tail: usize| -> Vec<usize> { lead.iter().copied().chain([out, tail]).collect() };
            (
                vec![
                    (name.clone(), "U8", with(input / 2), packed),
                    (
                        format!("{stem}.weight_scale"),
                        "F8_E4M3",
                        with(input / 16),
                        group_scales,
                    ),
                    (
                        format!("{stem}.weight_scale_2"),
                        "F32",
                        lead.clone(),
                        le_f32(&second),
                    ),
                ],
                vec![(id(&name), ReferenceTensor::new(shape, back)?)],
            )
        }
        Storage::Fp8Row => {
            let (rows, dim) = (shape[0], shape[1]);
            let mut codes = vec![0u8; rows * dim];
            let mut scales = vec![0f32; rows];
            for r in 0..rows {
                let row = &values[r * dim..(r + 1) * dim];
                let amax = row.iter().fold(0f32, |a, v| a.max(v.abs())).max(1e-12);
                let scale = amax / 448.0;
                scales[r] = scale;
                for (c, v) in row.iter().enumerate() {
                    codes[r * dim + c] = to_e4m3((v / scale).clamp(-448.0, 448.0));
                }
            }
            (
                vec![
                    (name.clone(), "F8_E4M3", shape.clone(), codes.clone()),
                    (
                        format!("{stem}.weight_scale"),
                        "F32",
                        vec![rows],
                        le_f32(&scales),
                    ),
                ],
                vec![
                    (
                        id(&name),
                        ReferenceTensor::new_i64(
                            shape,
                            codes.iter().map(|c| i64::from(*c)).collect(),
                        )?,
                    ),
                    (
                        id(&format!("{stem}.weight_scale")),
                        ReferenceTensor::new(vec![rows], scales)?,
                    ),
                ],
            )
        }
    })
}

/// The tiny lookup table: a map onto fewer compressed ids than entries, not the identity.
fn lookup_values(size: usize) -> Vec<f32> {
    let compressed = (size * 5 / 7).max(1);
    (0..size)
        .map(|i| ((7 * i + 3) % compressed) as f32)
        .collect()
}

/// The tiny token stream: a seeded walk over the vocabulary with the Engram pad id and the image
/// and video placeholders as real tokens (the mask drops all three), ids past the lookup table,
/// and a repeated span so later n-grams hash to rows the earlier ones read.
fn fixture_tokens(hy: &HyV4FlashConfig, count: usize) -> Vec<u32> {
    let vocab = hy.vocab_size as u64;
    let mut state = 0x2026_1120u64;
    let mut tokens: Vec<u32> = (0..count)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((state >> 33) % vocab) as u32
        })
        .collect();
    for (at, token) in [
        (5, hy.engram_pad_id),
        (6, hy.image_token_id),
        (7, hy.video_token_id),
        (40, hy.engram_lookup_table_size),
        (41, hy.vocab_size - 1),
    ] {
        if at < count {
            tokens[at] = token;
        }
    }
    if count > 320 {
        let span: Vec<u32> = tokens[100..116].to_vec();
        tokens[300..316].copy_from_slice(&span);
    }
    tokens
}

/// A minimal safetensors writer: little-endian header length, a JSON header, the data.
fn safetensors_bytes(entries: &[Entry]) -> Vec<u8> {
    use std::fmt::Write as _;
    let mut header = String::from("{");
    let mut offset = 0usize;
    for (index, (name, dtype, shape, bytes)) in entries.iter().enumerate() {
        if index > 0 {
            header.push(',');
        }
        let shape: Vec<String> = shape.iter().map(usize::to_string).collect();
        write!(
            header,
            "\"{name}\":{{\"dtype\":\"{dtype}\",\"shape\":[{}],\"data_offsets\":[{offset},{}]}}",
            shape.join(","),
            offset + bytes.len()
        )
        .unwrap();
        offset += bytes.len();
    }
    header.push('}');
    while !(8 + header.len()).is_multiple_of(8) {
        header.push(' ');
    }
    let mut out = Vec::with_capacity(8 + header.len() + offset);
    out.extend_from_slice(&(header.len() as u64).to_le_bytes());
    out.extend_from_slice(header.as_bytes());
    for (_, _, _, bytes) in entries {
        out.extend_from_slice(bytes);
    }
    out
}

fn json_list<T: std::fmt::Display>(values: impl IntoIterator<Item = T>) -> String {
    let items: Vec<String> = values.into_iter().map(|value| value.to_string()).collect();
    format!("[{}]", items.join(", "))
}

/// The pack's tiny config, refusing a plan that is not the one it compiles to: the fixture's
/// weights, moduli and checkpoint image belong to that config.
fn tiny_config_for(plan: &ModelPlan) -> Result<HyV4FlashConfig, ReferenceError> {
    let hy =
        pack::tiny_config().map_err(|_| invalid(None, "the hy_v4_flash tiny config compiles"))?;
    let tiny = memra_gguf::model_packs::by_alias("hy_v4_flash")
        .and_then(|p| p.compile_tiny_plan().ok())
        .ok_or(invalid(None, "the hy_v4_flash tiny config compiles"))?;
    if &tiny != plan {
        return Err(invalid(
            None,
            "the hy_v4_flash fixture is the pack's tiny config; this plan is another",
        ));
    }
    Ok(hy)
}

/// The tiny fixture: [`TINY_TOKENS`] tokens, prefill [`TINY_PREFILL`] (past the window and the
/// indexer budget), then teacher-forced decode steps.
pub fn fixture(plan: &ModelPlan) -> Result<ReferenceFixture, ReferenceError> {
    fixture_split(plan, TINY_TOKENS, TINY_PREFILL)
}

/// [`fixture`]'s weights on the first `tokens` tokens of its stream, split at `decode_from`.
pub fn fixture_split(
    plan: &ModelPlan,
    tokens: usize,
    decode_from: usize,
) -> Result<ReferenceFixture, ReferenceError> {
    let hy = tiny_config_for(plan)?;
    let mut weights = ReferenceWeights::new();
    let mut entries = Vec::new();
    for tensor in checkpoint_tensors(&hy)? {
        let values = if tensor.storage == Storage::I32 {
            lookup_values(tensor.shape[0])
        } else {
            fixture_values(&tensor)?
        };
        let (stored, read) = encode(&tensor, &values)?;
        entries.extend(stored);
        weights.extend(read);
    }
    for il in 0..hy.trunk_layers() {
        if let Some(slot) = hy.engram_slot(il) {
            let moduli: Vec<i64> = hy.engram_head_vocab_sizes[slot]
                .iter()
                .map(|m| *m as i64)
                .collect();
            weights.insert(
                id(&engram_moduli_name(il)),
                ReferenceTensor::new_i64(vec![moduli.len()], moduli)?,
            );
        }
    }
    let token_ids = fixture_tokens(&hy, tokens);
    if decode_from == 0 || decode_from >= token_ids.len() {
        return Err(invalid(
            None,
            "the fixture's prefill is a non-empty proper prefix of its tokens",
        ));
    }
    let fixture_json = format!(
        "{{\n  \"token_ids\": {},\n  \"decode_from\": {decode_from}\n}}\n",
        json_list(&token_ids)
    );
    Ok(ReferenceFixture {
        token_ids,
        weights,
        vision: None,
        multimodal_token_ids: None,
        decode_from: Some(decode_from),
        checkpoint_files: vec![
            ("model.safetensors".into(), safetensors_bytes(&entries)),
            ("config.json".into(), pack::TINY_CONFIG.as_bytes().to_vec()),
            ("fixture.json".into(), fixture_json.into_bytes()),
        ],
    })
}

/// The tiny gate's variants beside [`fixture`], each pinned to its own oracle: the request split
/// below one limit, so its decode steps cross it.
///
/// - `split250`: 280 tokens, prefill 250 (dense on the indexed layers), 30 decode steps; the
///   indexer starts at the step whose cache passes the budget of 256.
/// - `split500`: 530 tokens, prefill 500 (past the budget, inside the window), 30 decode steps;
///   the sliding layers drop their oldest key from slot 512 on.
pub fn tiny_variants(plan: &ModelPlan) -> Result<Vec<TinyVariant>, ReferenceError> {
    Ok(vec![
        TinyVariant {
            name: "split250",
            plan: plan.clone(),
            fixture: fixture_split(plan, 280, 250)?,
        },
        TinyVariant {
            name: "split500",
            plan: plan.clone(),
            fixture: fixture_split(plan, 530, 500)?,
        },
    ])
}

// ============================ execution ============================

struct Reader<'a> {
    weights: &'a ReferenceWeights,
}

impl Reader<'_> {
    fn tensor(&self, name: &str) -> Result<&ReferenceTensor, ReferenceError> {
        let key = id(name);
        self.weights
            .get(&key)
            .ok_or(ReferenceError::MissingTensor(key))
    }

    /// An f32 tensor of exactly `shape`, as f64. A tensor with the right element count in
    /// another shape (a transposed matrix) is refused.
    fn f(&self, name: &str, shape: &[usize]) -> Result<Vec<f64>, ReferenceError> {
        let t = self.tensor(name)?;
        let len: usize = shape.iter().product();
        if t.ints.is_some() || t.shape != shape || t.data.len() != len {
            return Err(ReferenceError::TensorShape {
                id: Some(id(name)),
                expected: shape.to_vec(),
                actual_elements: t.data.len(),
            });
        }
        Ok(t.data.iter().map(|v| f64::from(*v)).collect())
    }

    /// An integer tensor of exactly `shape`.
    fn ints(&self, name: &str, shape: &[usize]) -> Result<&[i64], ReferenceError> {
        let t = self.tensor(name)?;
        let ints = t
            .ints
            .as_deref()
            .ok_or_else(|| ReferenceError::IntegerTensorRequired(id(name)))?;
        if t.shape != shape || ints.len() != shape.iter().product::<usize>() {
            return Err(ReferenceError::TensorShape {
                id: Some(id(name)),
                expected: shape.to_vec(),
                actual_elements: ints.len(),
            });
        }
        Ok(ints)
    }
}

struct OwnedHc {
    function: Vec<f64>,
    base: Vec<f64>,
    scale: Vec<f64>,
}

impl OwnedHc {
    fn view(&self) -> ops::HyperConnectionWeights<'_> {
        ops::HyperConnectionWeights {
            function: &self.function,
            base: &self.base,
            scale: &self.scale,
        }
    }
}

struct OwnedIndexer {
    query: Vec<f64>,
    head_weight: Vec<f64>,
    token_key: Vec<f64>,
    token_key_norm: Vec<f64>,
    key_value: Vec<f64>,
    key_gate: Vec<f64>,
    ape: Vec<f64>,
    key_norm: Vec<f64>,
}

enum OwnedMlp {
    Dense {
        gate: Vec<f64>,
        up: Vec<f64>,
        down: Vec<f64>,
        output_norm: Vec<f64>,
        intermediate: usize,
    },
    Latent(Box<[Vec<f64>; 11]>),
}

struct OwnedEngram {
    plan: HyV4EngramPlan,
    codes: Vec<u8>,
    scales: Vec<f64>,
    head_norms: Vec<Vec<f64>>,
    value: Vec<f64>,
    value_bias: Vec<f64>,
    keys: Vec<Vec<f64>>,
    key_biases: Vec<Vec<f64>>,
    norm_key: Vec<Vec<f64>>,
    norm_query: Vec<Vec<f64>>,
    pre_gamma: Vec<f64>,
    stream_scales: Vec<Vec<f64>>,
    lookup: Vec<i32>,
    multipliers: Vec<i64>,
    moduli: Vec<u64>,
    offsets: Vec<u64>,
}

struct OwnedLayer {
    input_norm: Vec<f64>,
    mlp_input_norm: Vec<f64>,
    attention: [Vec<f64>; 10],
    indexer: Option<OwnedIndexer>,
    attention_hc: OwnedHc,
    mlp_hc: OwnedHc,
    mlp: OwnedMlp,
    engram: Option<OwnedEngram>,
}

impl OwnedLayer {
    fn view(&self) -> ops::LayerWeights<'_> {
        let a = &self.attention;
        ops::LayerWeights {
            input_norm: &self.input_norm,
            mlp_input_norm: &self.mlp_input_norm,
            attention: ops::AttentionWeights {
                q_a: &a[0],
                q_a_norm: &a[1],
                q_b: &a[2],
                kv_a: &a[3],
                kv_a_norm: &a[4],
                kv_b: &a[5],
                gate: &a[6],
                o: &a[7],
                post_norm: &a[8],
                sinks: &a[9],
                indexer: self.indexer.as_ref().map(|i| ops::IndexerWeights {
                    query: &i.query,
                    head_weight: &i.head_weight,
                    token_key: &i.token_key,
                    token_key_norm: &i.token_key_norm,
                    key_value: &i.key_value,
                    key_gate: &i.key_gate,
                    ape: &i.ape,
                    key_norm: &i.key_norm,
                }),
            },
            attention_hc: self.attention_hc.view(),
            mlp_hc: self.mlp_hc.view(),
            mlp: match &self.mlp {
                OwnedMlp::Dense {
                    gate,
                    up,
                    down,
                    output_norm,
                    intermediate,
                } => ops::MlpWeights::Dense {
                    gate,
                    up,
                    down,
                    output_norm,
                    intermediate: *intermediate,
                },
                OwnedMlp::Latent(w) => ops::MlpWeights::Latent(ops::LatentMoeWeights {
                    router: &w[0],
                    router_bias: &w[1],
                    latent_down: &w[2],
                    latent_up: &w[3],
                    expert_gate: &w[4],
                    expert_up: &w[5],
                    expert_down: &w[6],
                    shared_gate: &w[7],
                    shared_up: &w[8],
                    shared_down: &w[9],
                    output_norm: &w[10],
                }),
            },
            engram: self.engram.as_ref().map(|e| ops::EngramWeights {
                table: ops::EngramTable {
                    codes: &e.codes,
                    scales: &e.scales,
                },
                head_norms: e.head_norms.iter().map(Vec::as_slice).collect(),
                value: &e.value,
                value_bias: &e.value_bias,
                keys: e.keys.iter().map(Vec::as_slice).collect(),
                key_biases: e.key_biases.iter().map(Vec::as_slice).collect(),
                norm_key: e.norm_key.iter().map(Vec::as_slice).collect(),
                norm_query: e.norm_query.iter().map(Vec::as_slice).collect(),
                pre_gamma: &e.pre_gamma,
                stream_scales: e.stream_scales.iter().map(Vec::as_slice).collect(),
            }),
        }
    }
}

fn load_layer(
    plan: &ModelPlan,
    layer: &memra_gguf::model_plan::LayerPlan,
    r: &Reader,
) -> Result<OwnedLayer, ReferenceError> {
    let il = layer.index;
    let h = plan.hidden_size as usize;
    let p = format!("model.layers.{il}.");
    let (AttentionPlan::HyV4(a), ResidualTopology::HyV4(res)) = (&layer.attention, &layer.residual)
    else {
        return Err(invalid(Some(il), "not a Hy4-flash layer"));
    };
    let n = res.streams as usize;
    let heads = a.query_heads as usize;
    let (nope, rope, vdim) = (
        a.qk_nope_head_dim as usize,
        a.qk_rope_head_dim as usize,
        a.value_head_dim as usize,
    );
    let (q_lora, kv_lora) = (a.q_lora_rank as usize, a.kv_lora_rank as usize);
    let at = |s: &str| format!("{p}self_attn.{s}");
    let attention = [
        r.f(&at("q_a_proj.weight"), &[q_lora, h])?,
        r.f(&at("q_a_layernorm.weight"), &[q_lora])?,
        r.f(&at("q_b_proj.weight"), &[heads * (nope + rope), q_lora])?,
        r.f(&at("kv_a_proj_with_mqa.weight"), &[kv_lora + rope, h])?,
        r.f(&at("kv_a_layernorm.weight"), &[kv_lora])?,
        r.f(&at("kv_b_proj.weight"), &[heads * (nope + vdim), kv_lora])?,
        r.f(&at("linear_gate.weight"), &[heads * vdim, h])?,
        r.f(&at("o_proj.weight"), &[h, heads * vdim])?,
        r.f(&at("post_proj_norm.weight"), &[h])?,
        r.f(&at("learnable_sink_param"), &[heads])?,
    ];
    let indexer = match a.kind {
        HyV4AttentionKind::Indexed(index) => {
            let (ih, id_) = (index.heads as usize, index.head_dim as usize);
            let i = |s: &str| format!("{p}self_attn.compressed_indexer.{s}");
            Some(OwnedIndexer {
                query: r.f(&i("query_projection.weight"), &[ih * id_, q_lora])?,
                head_weight: r.f(&i("head_weight_projection.weight"), &[ih, h])?,
                token_key: r.f(&i("token_key_projection.weight"), &[id_, h])?,
                token_key_norm: r.f(&i("token_key_norm.weight"), &[id_])?,
                key_value: r.f(&i("key_value_projection.weight"), &[id_, id_])?,
                key_gate: r.f(&i("key_gate_projection.weight"), &[id_, id_])?,
                ape: r.f(&i("intra_group_ape"), &[index.pool as usize, id_])?,
                key_norm: r.f(&i("key_norm.weight"), &[id_])?,
            })
        }
        HyV4AttentionKind::Sliding { .. } => None,
    };
    let hc = |sub: &str| -> Result<OwnedHc, ReferenceError> {
        Ok(OwnedHc {
            function: r.f(&format!("{p}{sub}.hc_pre.hc_fn"), &[2 * n, n * h])?,
            base: r.f(&format!("{p}{sub}.hc_pre.hc_base"), &[2 * n])?,
            scale: r.f(&format!("{p}{sub}.hc_pre.hc_scale"), &[2])?,
        })
    };
    let m = |s: &str| format!("{p}mlp.{s}");
    let mlp = match &layer.mlp {
        MlpPlan::Dense(dense) => {
            let inter = dense.intermediate_size as usize;
            OwnedMlp::Dense {
                gate: r.f(&m("gate_proj.weight"), &[inter, h])?,
                up: r.f(&m("up_proj.weight"), &[inter, h])?,
                down: r.f(&m("down_proj.weight"), &[h, inter])?,
                output_norm: r.f(&m("post_fc2_norm.weight"), &[h])?,
                intermediate: inter,
            }
        }
        MlpPlan::LatentMoe(moe) => {
            let e = moe.moe.expert_count as usize;
            let latent = moe.latent_size as usize;
            let inter = moe.moe.expert_intermediate_size as usize;
            let shared = moe
                .moe
                .shared
                .as_ref()
                .ok_or(invalid(Some(il), "hy_v4_flash has a shared expert"))?
                .intermediate_size as usize;
            OwnedMlp::Latent(Box::new([
                r.f(&m("gate.weight"), &[e, h])?,
                r.f(&m("gate.e_score_correction_bias"), &[e])?,
                r.f(&m("routed_expert_down_proj.weight"), &[latent, h])?,
                r.f(&m("routed_expert_up_proj.weight"), &[h, latent])?,
                r.f(&m("experts.gate_proj.weight"), &[e, inter, latent])?,
                r.f(&m("experts.up_proj.weight"), &[e, inter, latent])?,
                r.f(&m("experts.down_proj.weight"), &[e, latent, inter])?,
                r.f(&m("shared_experts.gate_proj.weight"), &[shared, h])?,
                r.f(&m("shared_experts.up_proj.weight"), &[shared, h])?,
                r.f(&m("shared_experts.down_proj.weight"), &[h, shared])?,
                r.f(&m("combine_output_norm.weight"), &[h])?,
            ]))
        }
        _ => {
            return Err(invalid(
                Some(il),
                "a Hy4-flash layer runs a dense or latent MoE MLP",
            ));
        }
    };
    let engram = match &res.engram {
        None => None,
        Some(ep) => {
            let e = |s: &str| format!("{p}engram_layer.{s}");
            let heads = ep.heads as usize;
            let dim = ep.head_dim as usize;
            let width = heads * dim;
            let rows = ep.table_rows as usize;
            let codes = r
                .ints(&e("multi_head_embedding.embedding.weight"), &[rows, dim])?
                .iter()
                .map(|c| u8::try_from(*c))
                .collect::<Result<Vec<u8>, _>>()
                .map_err(|_| invalid(Some(il), "Engram table codes are bytes"))?;
            let lookup = r
                .ints(&e("lookup_table"), &[ep.lookup_table_size as usize])?
                .iter()
                .map(|v| i32::try_from(*v))
                .collect::<Result<Vec<i32>, _>>()
                .map_err(|_| invalid(Some(il), "the lookup table is I32"))?;
            let moduli = r
                .ints(&engram_moduli_name(il), &[heads])?
                .iter()
                .map(|v| u64::try_from(*v))
                .collect::<Result<Vec<u64>, _>>()
                .map_err(|_| invalid(Some(il), "Engram moduli are positive"))?;
            let vocab = hashing::compressed_vocab(ep, &lookup)
                .map_err(|_| invalid(Some(il), "the Engram lookup table fits its plan"))?;
            let multipliers = hashing::hash_multipliers(ep, vocab)
                .map_err(|_| invalid(Some(il), "the Engram multipliers draw"))?;
            let per_stream = |s: &str, shape: &[usize]| -> Result<Vec<Vec<f64>>, ReferenceError> {
                (0..n)
                    .map(|k| r.f(&e(&s.replace('#', &k.to_string())), shape))
                    .collect()
            };
            Some(OwnedEngram {
                plan: *ep,
                codes,
                scales: r.f(&e("multi_head_embedding.embedding.weight_scale"), &[rows])?,
                head_norms: (0..heads)
                    .map(|k| r.f(&e(&format!("lookup_per_head_norms.{k}.weight")), &[dim]))
                    .collect::<Result<_, _>>()?,
                value: r.f(&e("value_proj.weight"), &[h, width])?,
                value_bias: r.f(&e("value_proj.bias"), &[h])?,
                keys: per_stream("key_projs.#.weight", &[h, width])?,
                key_biases: per_stream("key_projs.#.bias", &[h])?,
                norm_key: per_stream("norm_key.#.weight", &[h])?,
                norm_query: per_stream("norm_query.#.weight", &[h])?,
                pre_gamma: r.f(&e("unified_pre_gamma"), &[h])?,
                stream_scales: per_stream("unified_per_hc_scales.#", &[h])?,
                offsets: hashing::head_offsets(&moduli),
                lookup,
                multipliers,
                moduli,
            })
        }
    };
    Ok(OwnedLayer {
        input_norm: r.f(&format!("{p}input_layernorm.weight"), &[h])?,
        mlp_input_norm: r.f(&format!("{p}post_attention_layernorm.weight"), &[h])?,
        attention,
        indexer,
        attention_hc: hc("hc_attn_layer")?,
        mlp_hc: hc("hc_mlp_layer")?,
        mlp,
        engram,
    })
}

/// What one forward call (the prefill chunk, or one decode step) selected: per indexed layer
/// past the budget, each new query's recalled groups; per Engram layer, each new token's table
/// rows (`[tokens, heads]`, head offsets included).
#[derive(Debug, Clone, PartialEq)]
pub struct TraceCall {
    pub start: usize,
    pub tokens: usize,
    pub recalled: Vec<(u32, Vec<Vec<usize>>)>,
    pub engram_rows: Vec<(u32, Vec<u64>)>,
}

/// [`execute`] and every call's selections, for the gate that compares selections before logits.
pub fn execute_traced(
    plan: &ModelPlan,
    weights: &ReferenceWeights,
    token_ids: &[u32],
    decode_from: Option<usize>,
) -> Result<(ReferenceOutput, Vec<TraceCall>), ReferenceError> {
    let numerics = plan
        .hy_v4_flash
        .ok_or(invalid(None, "not a hy_v4_flash plan"))?;
    if !numerics.output_projection_f32 {
        return Err(invalid(None, "the Hy4-flash head is computed in F32"));
    }
    if token_ids.is_empty() {
        return Err(ReferenceError::EmptyInput);
    }
    let vocab = plan.vocab_size as usize;
    let hidden = plan.hidden_size as usize;
    let n = numerics.streams as usize;
    if let Some(&token) = token_ids.iter().find(|t| **t as usize >= vocab) {
        return Err(ReferenceError::TokenOutOfRange { token, vocab });
    }
    let prefill = decode_from.unwrap_or(token_ids.len());
    if prefill == 0 || prefill > token_ids.len() {
        return Err(invalid(
            None,
            "the prefill is a non-empty prefix of the request",
        ));
    }
    for layer in &plan.layers {
        match &layer.residual {
            ResidualTopology::HyV4(res) if res.streams as usize == n => {}
            _ => {
                return Err(invalid(
                    Some(layer.index),
                    "every layer runs the plan's streams",
                ));
            }
        }
    }
    let r = Reader { weights };
    let layers = plan
        .layers
        .iter()
        .map(|layer| load_layer(plan, layer, &r))
        .collect::<Result<Vec<_>, _>>()?;
    let embed = r.f("model.embed_tokens.weight", &[vocab, hidden])?;
    let head = r.f("lm_head.weight", &[vocab, hidden])?;
    let final_norm = r.f("model.norm.weight", &[hidden])?;
    let exit = OwnedHc {
        function: r.f("model.hc_head.hc_head_fn", &[n, n * hidden])?,
        base: r.f("model.hc_head.hc_head_base", &[n])?,
        scale: r.f("model.hc_head.hc_head_scale", &[1])?,
    };

    let mut caches: Vec<ops::MlaCache> = layers.iter().map(|_| ops::MlaCache::default()).collect();
    let mut logits = Vec::with_capacity(token_ids.len() * vocab);
    let mut layer_hidden: Vec<Vec<f32>> = vec![Vec::new(); layers.len()];
    let mut trace = Vec::new();
    let mut calls = vec![(0, prefill)];
    calls.extend((prefill..token_ids.len()).map(|t| (t, t + 1)));
    for (start, end) in calls {
        let positions: Vec<u32> = (start as u32..end as u32).collect();
        let mut streams = Vec::with_capacity((end - start) * n * hidden);
        for &token in &token_ids[start..end] {
            let row = &embed[token as usize * hidden..(token as usize + 1) * hidden];
            for _ in 0..n {
                streams.extend_from_slice(row);
            }
        }
        let mut call = TraceCall {
            start,
            tokens: end - start,
            recalled: Vec::new(),
            engram_rows: Vec::new(),
        };
        for (il, (layer, owned)) in plan.layers.iter().zip(&layers).enumerate() {
            let inputs = match &owned.engram {
                None => None,
                Some(e) => {
                    let history: Vec<u32> = (0..end as u32).collect();
                    let ngrams = hashing::ngram_tokens(&e.plan, &token_ids[..end], &history);
                    let width = e.plan.max_ngram as usize;
                    let hashes = hashing::hash_ids(
                        &e.plan,
                        &e.lookup,
                        &e.multipliers,
                        &e.moduli,
                        &ngrams[start * width..],
                    )
                    .map_err(|_| invalid(Some(layer.index), "the Engram hash runs"))?;
                    let heads = e.plan.heads as usize;
                    let rows: Vec<u64> = hashes
                        .iter()
                        .enumerate()
                        .map(|(i, v)| v + e.offsets[i % heads])
                        .collect();
                    let mask = hashing::engram_mask(&e.plan, &token_ids[start..end]);
                    call.engram_rows.push((layer.index, rows.clone()));
                    Some((rows, mask))
                }
            };
            let engram_inputs = inputs
                .as_ref()
                .map(|(rows, mask)| ops::EngramInputs { rows, mask });
            let (next, recalled) = ops::decoder_layer_cached(
                layer,
                &owned.view(),
                &streams,
                &positions,
                hidden,
                engram_inputs.as_ref(),
                &mut caches[il],
            );
            streams = next;
            if let Some(lists) = recalled {
                call.recalled.push((layer.index, lists));
            }
            layer_hidden[il].extend(streams.iter().map(|v| *v as f32));
        }
        let collapsed =
            ops::hyper_head(&numerics, plan.output_norm, &exit.view(), &streams, hidden);
        let normed = ops::rms_norm(&collapsed, &final_norm, plan.output_norm);
        logits.extend(
            ops::linear(&normed, &head, hidden, vocab)
                .into_iter()
                .map(|v| v as f32),
        );
        trace.push(call);
    }
    let state = ReferenceState {
        layers: caches
            .iter()
            .zip(&plan.layers)
            .map(|(cache, layer)| {
                let AttentionPlan::HyV4(a) = &layer.attention else {
                    unreachable!("checked when the layer loaded")
                };
                let (kv_lora, rope) = (a.kv_lora_rank as usize, a.qk_rope_head_dim as usize);
                let mut rows = Vec::with_capacity(cache.slots() * (kv_lora + rope));
                for s in 0..cache.slots() {
                    rows.extend(
                        cache.latent[s * kv_lora..(s + 1) * kv_lora]
                            .iter()
                            .map(|v| *v as f32),
                    );
                    rows.extend(
                        cache.k_rope[s * rope..(s + 1) * rope]
                            .iter()
                            .map(|v| *v as f32),
                    );
                }
                ReferenceLayerState::LatentKv {
                    rows,
                    tokens: cache.slots(),
                    width: kv_lora + rope,
                }
            })
            .collect(),
    };
    Ok((
        ReferenceOutput {
            logits,
            tokens: token_ids.len(),
            vocab,
            state,
            mtp: Vec::new(),
            draft: None,
            layer_hidden,
            decode: decode_from.map(|prefill_tokens| ReferenceDecodeTrace {
                prefill_tokens,
                drafts: Vec::new(),
            }),
        },
        trace,
    ))
}

/// The Hy4-flash text program on `token_ids`: `token_ids[..decode_from]` as one prefill chunk,
/// then one teacher-forced decode step per later token (`None`: one prefill).
pub fn execute(
    plan: &ModelPlan,
    weights: &ReferenceWeights,
    token_ids: &[u32],
    decode_from: Option<usize>,
) -> Result<ReferenceOutput, ReferenceError> {
    execute_traced(plan, weights, token_ids, decode_from).map(|(output, _)| output)
}
