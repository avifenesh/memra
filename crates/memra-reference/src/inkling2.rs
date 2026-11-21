//! Inkling 2 reference execution: the program `ModelPlan::inkling2` plans compile to.
//!
//! The executor is the semantic baseline of the family. It runs every operation unfused, in f64,
//! from the values the checkpoint stores: BF16 and F32 tensors are exact in f32 and are widened,
//! and the routed expert banks are read from their stored bytes. It has two inputs for a layer's
//! experts, chosen per physical layer by the plan
//! (`Inkling2Numerics::experts[layer].activation`):
//!
//! - `Float`: the exactly dequantized weights on float inputs ([`ops::expert_float`]);
//! - `Declared`: the checkpoint's required activation quantizer, the fp32 program of
//!   [`crate::inkling2_actquant`] (NVFP4 W4A4 or MXFP8 W8A8), on the layer input rounded to bf16.
//!   Everything around the experts stays f64 here: the bf16 rounding points a serving class puts
//!   on the rest of the trunk are that class's, and it is compared with this under its own band.
//!
//! A request is a schedule of chunks: [`execute_schedule`] prefills each chunk as one call that
//! walks the slots in execution order over all of the chunk's rows, then runs each decode token
//! as a chunk of one. A slot's state between calls is its K and V rows (a sliding slot keeps the
//! last `window - 1`) and the last `kernel - 1` inputs of its four short convolutions. The loop's
//! message bank is rebuilt inside every call and carries nothing between rows.
//!
//! The drafter and the vision tower are not executed: text only, no draft.

pub mod ops;

use memra_gguf::model_packs::inkling2 as pack;
use memra_gguf::model_plan::{
    ActivationPlan, AttentionInputs, AttentionPlan, BeyondExtent, BlockAttentionPlan, ClipScale,
    ExpertActivationMode, ExpertWeightFormat, Inkling2Plan, L2ClipPlan, LayerPlan, LogitsTransform,
    LoopMessageTarget, MlpPlan, ModelPlan, MoeMlpPlan, NormKind, NormPlan, OutputGatePlan,
    QuerySinkPlan, RelativeBiasAttentionPlan, ResidualTopology, RouterPlan, ShortConvPlan,
    ShortConvTaps, StatePlan, ValueNorm, WeightTransform,
};
use memra_gguf::quant_groups::{ActivationAmax, ActivationQuant, ActivationScale, ScaleSelection};
use memra_gguf::tensor_contract::TensorId;

use crate::inkling2_actquant as actquant;
use crate::{
    ReferenceDecodeTrace, ReferenceError, ReferenceFixture, ReferenceLayerState, ReferenceOutput,
    ReferenceState, ReferenceTensor, ReferenceWeights, TinyGate, tensor, tensor_i64,
};
use ops::{AttentionShape, ConvHistory, ExpertBank};

/// Rows of one chunk: `[tokens][width]`.
type Rows = Vec<Vec<f64>>;

const LLM: &str = "model.llm.";

fn invalid(layer: Option<u32>, reason: &'static str) -> ReferenceError {
    ReferenceError::InvalidPlan { layer, reason }
}

fn unsupported(layer: Option<u32>, operation: &'static str) -> ReferenceError {
    ReferenceError::UnsupportedOperation { layer, operation }
}

/// The semantic id of a checkpoint tensor (the pack's map; expert planes are family keys).
pub fn tensor_id(name: &str) -> TensorId {
    pack::semantic_tensor_id(name)
}

/// SiTU's factor tables for the caps (built once per thread and caps: 2 x 65,536 f64).
fn situ_tables(gate_cap: f32, up_cap: f32) -> std::rc::Rc<actquant::SituTables> {
    thread_local! {
        static TABLES: std::cell::RefCell<Option<std::rc::Rc<actquant::SituTables>>> =
            const { std::cell::RefCell::new(None) };
    }
    TABLES.with(|cell| {
        let mut slot = cell.borrow_mut();
        match slot.as_ref() {
            Some(t)
                if t.gate_cap.to_bits() == gate_cap.to_bits()
                    && t.up_cap.to_bits() == up_cap.to_bits() =>
            {
                t.clone()
            }
            _ => {
                let t = std::rc::Rc::new(actquant::situ_tables(gate_cap, up_cap));
                *slot = Some(t.clone());
                t
            }
        }
    })
}

fn program(plan: &ModelPlan) -> Result<&Inkling2Plan, ReferenceError> {
    plan.inkling2
        .as_ref()
        .ok_or(invalid(None, "inkling2 execution needs an inkling2 plan"))
}

fn attention_of(layer: &LayerPlan) -> Result<&RelativeBiasAttentionPlan, ReferenceError> {
    match &layer.attention {
        AttentionPlan::RelativeBias(attention) => Ok(attention),
        _ => Err(invalid(
            Some(layer.index),
            "inkling2 layers run relative-bias attention",
        )),
    }
}

fn residual_of(layer: &LayerPlan) -> Result<&BlockAttentionPlan, ReferenceError> {
    match &layer.residual {
        ResidualTopology::BlockAttention(residual) => Ok(residual),
        _ => Err(invalid(
            Some(layer.index),
            "inkling2 layers use the block attention residual",
        )),
    }
}

fn moe_of(layer: &LayerPlan) -> Result<&MoeMlpPlan, ReferenceError> {
    match &layer.mlp {
        MlpPlan::Moe(moe) => Ok(moe),
        _ => Err(invalid(Some(layer.index), "inkling2 layers are MoE")),
    }
}

fn rms(norm: &NormPlan, layer: Option<u32>) -> Result<f64, ReferenceError> {
    if norm.kind != NormKind::Rms || norm.weight_transform != WeightTransform::Identity {
        return Err(unsupported(layer, "a norm other than plain RMS"));
    }
    Ok(ops::decimal(norm.epsilon))
}

/// Refuse every plan shape this executor does not implement, by name, before any math.
fn check_layer(layer: &LayerPlan, program: &Inkling2Plan) -> Result<(), ReferenceError> {
    let at = Some(layer.index);
    let attention = attention_of(layer)?;
    let residual = residual_of(layer)?;
    let moe = moe_of(layer)?;
    rms(&layer.pre_attention_norm, at)?;
    rms(&layer.pre_mlp_norm, at)?;
    if attention.value_norm != ValueNorm::WeightlessRms {
        return Err(unsupported(at, "a value norm other than weightless RMS"));
    }
    let QuerySinkPlan::PerQueryProjection = attention.sink;
    let OutputGatePlan::SeparatePerDimSigmoid = attention.output_gate;
    if attention.key_clip.scale != ClipScale::None {
        return Err(unsupported(at, "a scaled key clip"));
    }
    if !attention.query_heads.is_multiple_of(attention.kv_heads) {
        return Err(invalid(at, "query heads are a multiple of KV heads"));
    }
    if attention.key_conv != attention.value_conv {
        return Err(unsupported(at, "different K and V short convolutions"));
    }
    for conv in [
        &attention.key_conv,
        &attention.output_conv,
        &program.mlp_output_conv,
    ] {
        if conv.kernel < 2 {
            return Err(invalid(at, "a short convolution has at least two taps"));
        }
        if let ShortConvTaps::StaticPlusDynamic { groups } = conv.taps
            && (groups == 0 || !conv.channels.is_multiple_of(groups))
        {
            return Err(invalid(at, "dynamic taps divide the channels into groups"));
        }
    }
    if attention.output_conv.taps != ShortConvTaps::Static
        || program.mlp_output_conv.taps != ShortConvTaps::Static
    {
        return Err(unsupported(at, "dynamic taps on an output convolution"));
    }
    match &layer.state {
        StatePlan::RelativeBias(state) => {
            if state.window != attention.window
                || state.conv_history != attention.key_conv.kernel - 1
            {
                return Err(invalid(at, "the slot state matches its attention"));
            }
        }
        _ => return Err(invalid(at, "inkling2 slots carry relative-bias state")),
    }
    if !residual.normalize_values {
        return Err(unsupported(at, "a depth mix over unnormalized states"));
    }
    match (attention.inputs, residual.attention_queries) {
        (_, 0) | (AttentionInputs::SharedNorm, 1) | (AttentionInputs::SeparateQkvNorms, 3) => {}
        _ => {
            return Err(invalid(
                at,
                "one residual query per attention input of the layer class",
            ));
        }
    }
    if moe.shared.is_some() {
        return Err(unsupported(at, "a shared expert"));
    }
    match moe.router {
        RouterPlan::Sigmoid {
            normalize_selected: true,
            ..
        } => {}
        _ => return Err(unsupported(at, "a router other than normalized sigmoid")),
    }
    let ActivationPlan::SwiGluSoftcapped { .. } = moe.activation else {
        return Err(unsupported(
            at,
            "an activation other than softcapped SwiGLU",
        ));
    };
    if moe.experts_per_token == 0 || moe.experts_per_token > moe.expert_count {
        return Err(invalid(at, "top-k is between 1 and the expert count"));
    }
    Ok(())
}

/// The declared quantizer the activation-quant reference implements for a bank format.
fn check_declared(
    physical: u32,
    weight: ExpertWeightFormat,
    declared: ActivationQuant,
    hidden: usize,
    width: usize,
    program: &Inkling2Plan,
) -> Result<(), ReferenceError> {
    let at = Some(physical);
    let block = match (weight, declared) {
        (
            ExpertWeightFormat::Nvfp4Amax,
            ActivationQuant::Nvfp4E2m1 {
                block_size: 16,
                amax: ActivationAmax::PerToken,
                scale: ActivationScale::E4m3,
                scale_selection: ScaleSelection::FourOverSixMae,
            },
        ) => 16,
        (
            ExpertWeightFormat::Mxfp8Row32,
            ActivationQuant::Mxfp8E4m3 {
                block_size: 32,
                scale: ActivationScale::Ue8m0,
            },
        ) => 32,
        _ => {
            return Err(unsupported(
                at,
                "a declared activation quantizer other than NVFP4 four-over-six on NVFP4 banks or MXFP8 on MXFP8 banks",
            ));
        }
    };
    if !hidden.is_multiple_of(block) || !width.is_multiple_of(block) {
        return Err(invalid(
            at,
            "quantized expert inputs are whole quantizer blocks",
        ));
    }
    if !program.numerics.gate_up_interleaved || !program.numerics.routed_weight_before_down {
        return Err(unsupported(
            at,
            "the declared expert program without interleaved gate/up and the routing weight before down",
        ));
    }
    Ok(())
}

/// One slot's state between calls.
#[derive(Debug, Clone, PartialEq, Default)]
struct SlotState {
    positions: Vec<u64>,
    keys: Vec<f64>,
    values: Vec<f64>,
    /// K, V, attention output, MLP output.
    conv: [ConvHistory; 4],
}

struct Session<'a> {
    plan: &'a ModelPlan,
    program: &'a Inkling2Plan,
    weights: &'a ReferenceWeights,
    /// One (gate/up, down) pair per physical layer.
    banks: Vec<(ExpertBank, ExpertBank)>,
    slots: Vec<SlotState>,
    position: u64,
    softcap: Option<f64>,
    keep: usize,
}

struct ChunkOutput {
    logits: Vec<f32>,
    layer_hidden: Vec<Vec<f32>>,
}

fn bytes(
    weights: &ReferenceWeights,
    name: &str,
    shape: &[usize],
) -> Result<Vec<u8>, ReferenceError> {
    let id = tensor_id(name);
    tensor_i64(weights, &id, shape)?
        .iter()
        .map(|&value| {
            u8::try_from(value).map_err(|_| ReferenceError::IntegerTensorRequired(id.clone()))
        })
        .collect()
}

fn load_bank(
    weights: &ReferenceWeights,
    name: &str,
    format: ExpertWeightFormat,
    experts: usize,
    rows: usize,
    inputs: usize,
    physical: u32,
) -> Result<ExpertBank, ReferenceError> {
    let bank = match format {
        ExpertWeightFormat::Nvfp4Amax => {
            if !inputs.is_multiple_of(16) {
                return Err(invalid(Some(physical), "NVFP4 rows are whole blocks of 16"));
            }
            ExpertBank::Nvfp4 {
                codes: bytes(weights, name, &[experts, rows, inputs / 2])?,
                scales: bytes(
                    weights,
                    &format!("{name}_scale"),
                    &[experts, rows, inputs / 16],
                )?,
                amax: tensor(weights, &tensor_id(&format!("{name}_amax")), &[experts])?.to_vec(),
                denominator: tensor(
                    weights,
                    &tensor_id(&format!("{name}_outer_scale_denominator")),
                    &[1],
                )?[0],
                rows,
                inputs,
            }
        }
        ExpertWeightFormat::Mxfp8Row32 => {
            if !inputs.is_multiple_of(32) {
                return Err(invalid(Some(physical), "MXFP8 rows are whole blocks of 32"));
            }
            ExpertBank::Mxfp8 {
                values: bytes(weights, name, &[experts, rows, inputs])?,
                exponents: bytes(
                    weights,
                    &format!("{name}_scale"),
                    &[experts, rows, inputs / 32],
                )?,
                rows,
                inputs,
            }
        }
    };
    if bank.has_nan_code() {
        return Err(invalid(
            Some(physical),
            "an expert bank holds an e4m3 NaN code",
        ));
    }
    Ok(bank)
}

impl<'a> Session<'a> {
    fn new(plan: &'a ModelPlan, weights: &'a ReferenceWeights) -> Result<Self, ReferenceError> {
        let program = program(plan)?;
        if plan.embedding_scale != 1.0 {
            return Err(unsupported(None, "a scaled embedding"));
        }
        if program.vision_executed {
            return Err(unsupported(None, "the Inkling 2 vision tower"));
        }
        if program.output_queries != 1 {
            return Err(unsupported(None, "an output mix with several queries"));
        }
        rms(&program.embedding_norm, None)?;
        rms(&plan.output_norm, None)?;
        let lp = &program.loop_plan;
        if lp.slots.len() != plan.layers.len() || lp.execution_order.len() != plan.layers.len() {
            return Err(invalid(None, "one loop slot and one visit per plan layer"));
        }
        if lp.targets
            != [
                LoopMessageTarget::Query,
                LoopMessageTarget::Key,
                LoopMessageTarget::Value,
                LoopMessageTarget::Mlp,
            ]
        {
            return Err(unsupported(
                None,
                "a loop message order other than q, k, v, mlp",
            ));
        }
        let hidden = plan.hidden_size as usize;
        let mut softcap = None;
        let mut keep = plan.vocab_size as usize;
        for transform in &plan.logits {
            match transform {
                LogitsTransform::Softcap(cap) => softcap = Some(ops::decimal(*cap)),
                LogitsTransform::Truncate(rows) => keep = keep.min(*rows as usize),
                _ => {
                    return Err(unsupported(
                        None,
                        "a logits transform other than softcap and truncation",
                    ));
                }
            }
        }
        let physical_layers = program.numerics.experts.len();
        let mut banks = Vec::with_capacity(physical_layers);
        let mut seen = vec![false; physical_layers];
        for layer in &plan.layers {
            check_layer(layer, program)?;
            let slot = lp
                .slots
                .get(layer.index as usize)
                .filter(|slot| slot.slot == layer.index)
                .ok_or(invalid(Some(layer.index), "plan layers are in slot order"))?;
            let physical = slot.physical_layer as usize;
            if physical >= physical_layers {
                return Err(invalid(Some(layer.index), "a slot names a physical layer"));
            }
            seen[physical] = true;
        }
        if seen.contains(&false) {
            return Err(invalid(None, "every physical layer has a slot"));
        }
        for numerics in &program.numerics.experts {
            let physical = numerics.physical_layer;
            let layer = plan
                .layers
                .iter()
                .find(|layer| lp.slots[layer.index as usize].physical_layer == physical)
                .ok_or(invalid(Some(physical), "every physical layer has a slot"))?;
            let moe = moe_of(layer)?;
            let (experts, width) = (
                moe.expert_count as usize,
                moe.expert_intermediate_size as usize,
            );
            if numerics.activation == ExpertActivationMode::Declared {
                check_declared(
                    physical,
                    numerics.weight,
                    numerics.declared_activation,
                    hidden,
                    width,
                    program,
                )?;
            }
            let prefix = format!("{LLM}layers.{physical}.mlp.experts.");
            banks.push((
                load_bank(
                    weights,
                    &format!("{prefix}w13_weight"),
                    numerics.weight,
                    experts,
                    2 * width,
                    hidden,
                    physical,
                )?,
                load_bank(
                    weights,
                    &format!("{prefix}w2_weight"),
                    numerics.weight,
                    experts,
                    hidden,
                    width,
                    physical,
                )?,
            ));
        }
        Ok(Self {
            plan,
            program,
            weights,
            banks,
            slots: vec![SlotState::default(); plan.layers.len()],
            position: 0,
            softcap,
            keep,
        })
    }

    fn tensor(&self, name: &str, shape: &[usize]) -> Result<&'a [f32], ReferenceError> {
        tensor(self.weights, &tensor_id(name), shape)
    }

    /// One call: `tokens` at positions `self.position..`, every slot over all rows.
    fn forward(&mut self, tokens: &[u32]) -> Result<ChunkOutput, ReferenceError> {
        if tokens.is_empty() {
            return Err(ReferenceError::EmptyInput);
        }
        let plan = self.plan;
        let program = self.program;
        let hidden_size = plan.hidden_size as usize;
        let vocab = plan.vocab_size as usize;
        let embedding = self.tensor(&format!("{LLM}embed.weight"), &[vocab, hidden_size])?;
        let embedding_norm = self.tensor(&format!("{LLM}embed_norm.weight"), &[hidden_size])?;
        let embedding_epsilon = rms(&program.embedding_norm, None)?;
        let mut hidden: Rows = tokens
            .iter()
            .map(|&token| {
                let token = token as usize;
                if token >= vocab {
                    return Err(ReferenceError::TokenOutOfRange {
                        token: token as u32,
                        vocab,
                    });
                }
                let row: Vec<f64> = embedding[token * hidden_size..(token + 1) * hidden_size]
                    .iter()
                    .map(|value| f64::from(*value))
                    .collect();
                Ok(ops::rms_norm(&row, Some(embedding_norm), embedding_epsilon))
            })
            .collect::<Result<_, _>>()?;

        let lp = &program.loop_plan;
        let mut blocks: Vec<Rows> = Vec::new();
        let mut layer_hidden = vec![Vec::new(); plan.layers.len()];
        // The stream and block list every pass restarts from.
        let mut prelude: Option<(Rows, Vec<Rows>)> = None;
        // The previous pass's blocks after its first, and its final stream.
        let mut bank: Option<(Vec<Rows>, Rows)> = None;
        let mut pass = None;
        for &slot in &lp.execution_order {
            let info = lp.slots[slot as usize];
            if info.pass != pass {
                match info.pass {
                    Some(0) => prelude = Some((hidden.clone(), blocks.clone())),
                    Some(_) => {
                        let (stream, list) = prelude
                            .as_ref()
                            .ok_or(invalid(Some(slot), "a loop pass follows the first pass"))?;
                        let after_first = blocks.get(list.len() + 1..).ok_or(invalid(
                            Some(slot),
                            "the first core layer starts a residual block",
                        ))?;
                        bank = Some((after_first.to_vec(), hidden));
                        hidden = stream.clone();
                        blocks = list.clone();
                    }
                    None => {}
                }
                pass = info.pass;
            }
            let messages = match (&bank, info.message) {
                (Some(bank), true) => Some(self.messages(slot, info.physical_layer, bank)?),
                (None, true) => {
                    return Err(invalid(
                        Some(slot),
                        "a slot reads a message before any pass ran",
                    ));
                }
                (_, false) => None,
            };
            hidden = self.run_slot(slot, info.physical_layer, hidden, &mut blocks, messages)?;
            layer_hidden[slot as usize] =
                hidden.iter().flatten().map(|value| *value as f32).collect();
        }

        let residual = residual_of(plan.layers.last().ok_or(ReferenceError::EmptyInput)?)?;
        let mix_epsilon = ops::decimal(residual.epsilon);
        let output_query = self.tensor(&format!("{LLM}attn_res.output_query"), &[hidden_size])?;
        let output_norm = self.tensor(&format!("{LLM}norm.weight"), &[hidden_size])?;
        let output_epsilon = rms(&plan.output_norm, None)?;
        let unembed = self.tensor(&format!("{LLM}unembed.weight"), &[vocab, hidden_size])?;
        let mut logits = Vec::with_capacity(tokens.len() * self.keep);
        for (row, stream) in hidden.iter().enumerate() {
            let states = states_at(&blocks, stream, row);
            let mixed = ops::depth_mix(output_query, &states, mix_epsilon).remove(0);
            let normed = ops::rms_norm(&mixed, Some(output_norm), output_epsilon);
            // Truncation: the padded rows of the output projection are never computed.
            for value in ops::linear(&unembed[..self.keep * hidden_size], &normed) {
                let value = self.softcap.map_or(value, |cap| ops::softcap(value, cap));
                logits.push(value as f32);
            }
        }
        self.position += tokens.len() as u64;
        Ok(ChunkOutput {
            logits,
            layer_hidden,
        })
    }

    /// The four messages of a core slot, per row, in `[q, k, v, mlp]` order: a depth mix of the
    /// previous pass's bank with the slot's four queries. The checkpoint stores the queries of
    /// core layer `i` as rows `2i` (q) and `2i + 1` (mlp) of one tensor and rows `2i` (k) and
    /// `2i + 1` (v) of the other.
    fn messages(
        &self,
        slot: u32,
        physical: u32,
        bank: &(Vec<Rows>, Rows),
    ) -> Result<Vec<Vec<Vec<f64>>>, ReferenceError> {
        let lp = &self.program.loop_plan;
        let hidden = self.plan.hidden_size as usize;
        let core = (lp.core_end - lp.core_start) as usize;
        let index = physical
            .checked_sub(lp.core_start)
            .filter(|index| (*index as usize) < core)
            .ok_or(invalid(Some(slot), "a message slot is a core layer"))?
            as usize;
        let shape = [2 * core, hidden];
        let query_mlp = self.tensor(&format!("{LLM}middle_cycle_cross_pass_queries"), &shape)?;
        let key_value = self.tensor(&format!("{LLM}middle_cycle_cross_pass_kv_queries"), &shape)?;
        let queries = ops::message_queries(query_mlp, key_value, index, hidden);
        let epsilon = ops::decimal(residual_of(&self.plan.layers[slot as usize])?.epsilon);
        Ok(bank
            .1
            .iter()
            .enumerate()
            .map(|(row, stream)| {
                ops::depth_mix(&queries, &states_at(&bank.0, stream, row), epsilon)
            })
            .collect())
    }

    fn controller(
        &self,
        physical: u32,
        target: &str,
        x: &[f64],
        message: &[f64],
    ) -> Result<Vec<f64>, ReferenceError> {
        let hidden = x.len();
        let rank = self.program.loop_plan.gate_rank as usize;
        let prefix = format!("{LLM}layers.{physical}.controllers.{target}.");
        Ok(ops::controller(
            x,
            message,
            self.tensor(&format!("{prefix}down_weight"), &[rank, 2 * hidden])?,
            self.tensor(&format!("{prefix}forget_weight"), &[hidden, rank])?,
            self.tensor(&format!("{prefix}gain_weight"), &[hidden, rank])?,
            ops::decimal(self.program.loop_plan.controller_epsilon),
        ))
    }

    fn clip(
        &self,
        rows: &mut [f64],
        width: usize,
        clip: &L2ClipPlan,
        scale: Option<&[f32]>,
        tau: f64,
    ) {
        ops::clip_heads(
            rows,
            width,
            ops::decimal(clip.threshold),
            ops::decimal(clip.epsilon),
            scale,
            tau,
        );
    }

    /// K or V of one row: projection, short convolution with the slot's history, then the K
    /// clip or the V norm per head.
    #[allow(clippy::too_many_arguments)] // allow: the slot, the conv and the input of one projection
    fn key_or_value(
        &mut self,
        slot: u32,
        prefix: &str,
        projection: &str,
        conv_name: &str,
        conv_index: usize,
        conv: &ShortConvPlan,
        input: &[f64],
    ) -> Result<Vec<f64>, ReferenceError> {
        let hidden = input.len();
        let (channels, kernel) = (conv.channels as usize, conv.kernel as usize);
        let projected = ops::linear(
            self.tensor(
                &format!("{prefix}attn.{projection}.weight"),
                &[channels, hidden],
            )?,
            input,
        );
        let history = &mut self.slots[slot as usize].conv[conv_index];
        Ok(match conv.taps {
            ShortConvTaps::Static => ops::short_conv_static(
                &projected,
                history,
                tensor(
                    self.weights,
                    &tensor_id(&format!("{prefix}attn.{conv_name}.weight")),
                    &[channels, 1, kernel],
                )?,
                kernel,
            ),
            ShortConvTaps::StaticPlusDynamic { groups } => {
                let groups = groups as usize;
                let dynamic = ops::linear(
                    tensor(
                        self.weights,
                        &tensor_id(&format!("{prefix}attn.{conv_name}.weight_proj")),
                        &[kernel * groups, hidden],
                    )?,
                    input,
                );
                ops::short_conv_dynamic(
                    &projected,
                    history,
                    tensor(
                        self.weights,
                        &tensor_id(&format!("{prefix}attn.{conv_name}.static_w")),
                        &[kernel, channels],
                    )?,
                    &dynamic,
                    kernel,
                    groups,
                )
            }
        })
    }

    /// One slot over the rows of a call. Pushes the incoming stream onto `blocks` when the
    /// layer starts a residual block.
    fn run_slot(
        &mut self,
        slot: u32,
        physical: u32,
        hidden: Rows,
        blocks: &mut Vec<Rows>,
        messages: Option<Vec<Vec<Vec<f64>>>>,
    ) -> Result<Rows, ReferenceError> {
        let plan = self.plan;
        let program = self.program;
        let layer = &plan.layers[slot as usize];
        let attention = attention_of(layer)?;
        let residual = residual_of(layer)?;
        let moe = moe_of(layer)?;
        let at = Some(slot);
        let hidden_size = plan.hidden_size as usize;
        let (heads, kv_heads, dim) = (
            attention.query_heads as usize,
            attention.kv_heads as usize,
            attention.head_dim as usize,
        );
        let (q_width, kv_width) = (heads * dim, kv_heads * dim);
        let prefix = format!("{LLM}layers.{physical}.");
        let mix_epsilon = ops::decimal(residual.epsilon);
        let attention_epsilon = rms(&layer.pre_attention_norm, at)?;
        let first_position = self.position;

        // Attention inputs: mix, controllers, norms.
        let norm_names: [&str; 3] = match attention.inputs {
            AttentionInputs::SharedNorm => ["attn_norm", "attn_norm", "attn_norm"],
            AttentionInputs::SeparateQkvNorms => ["q_input_norm", "k_input_norm", "v_input_norm"],
        };
        let norms = [
            self.tensor(&format!("{prefix}{}.weight", norm_names[0]), &[hidden_size])?,
            self.tensor(&format!("{prefix}{}.weight", norm_names[1]), &[hidden_size])?,
            self.tensor(&format!("{prefix}{}.weight", norm_names[2]), &[hidden_size])?,
        ];
        let attention_queries = match residual.attention_queries {
            0 => None,
            count => {
                // attn_queries.N belongs to layer N + 1.
                let index = physical
                    .checked_sub(1)
                    .ok_or(invalid(at, "the first layer has no attention mix"))?;
                let name = format!("{LLM}attn_res.attn_queries.{index}");
                Some(if count == 1 {
                    self.tensor(&name, &[hidden_size])?
                } else {
                    self.tensor(&name, &[count as usize, hidden_size])?
                })
            }
        };
        let mut inputs: Vec<[Vec<f64>; 3]> = Vec::with_capacity(hidden.len());
        for (row, stream) in hidden.iter().enumerate() {
            let mut three: [Vec<f64>; 3] = match attention_queries {
                None => [stream.clone(), stream.clone(), stream.clone()],
                Some(queries) => {
                    let mut mixes =
                        ops::depth_mix(queries, &states_at(blocks, stream, row), mix_epsilon);
                    if mixes.len() == 3 {
                        let v = mixes.remove(2);
                        let k = mixes.remove(1);
                        [mixes.remove(0), k, v]
                    } else {
                        let one = mixes.remove(0);
                        [one.clone(), one.clone(), one]
                    }
                }
            };
            if let Some(messages) = &messages {
                for (index, target) in ["attn", "k", "v"].into_iter().enumerate() {
                    three[index] =
                        self.controller(physical, target, &three[index], &messages[row][index])?;
                }
            }
            for (input, norm) in three.iter_mut().zip(norms) {
                *input = ops::rms_norm(input, Some(norm), attention_epsilon);
            }
            inputs.push(three);
        }

        // Projections. Every row's K and V enter the slot before any row attends.
        let wq = self.tensor(
            &format!("{prefix}attn.wq_du.weight"),
            &[q_width, hidden_size],
        )?;
        let q_scale = match attention.query_clip.scale {
            ClipScale::PerHeadChannel => {
                Some(self.tensor(&format!("{prefix}attn.q_l2_clip.scale"), &[heads, dim])?)
            }
            ClipScale::None => None,
        };
        let rank = attention.relative.rank as usize;
        let extent = attention.relative.extent as usize;
        let wr = self.tensor(
            &format!("{prefix}attn.wr_du.weight"),
            &[heads * rank, hidden_size],
        )?;
        let r_scale = match attention.relative.clip.scale {
            ClipScale::PerHeadChannel => {
                Some(self.tensor(&format!("{prefix}attn.rel_l2_clip.scale"), &[heads, rank])?)
            }
            ClipScale::None => None,
        };
        let table = self.tensor(
            &format!("{prefix}attn.rel_logits_proj.proj"),
            &[rank, extent],
        )?;
        let sink_weight = self.tensor(
            &format!("{prefix}attn.attn_sink_proj.weight"),
            &[heads, hidden_size],
        )?;
        let gate_weight = self.tensor(
            &format!("{prefix}attn.gate_attn.weight"),
            &[q_width, hidden_size],
        )?;
        let wo = self.tensor(
            &format!("{prefix}attn.wo_ud.weight"),
            &[hidden_size, q_width],
        )?;
        let key_clip = attention.key_clip;
        let value_epsilon = ops::decimal(attention.value_norm_epsilon);
        let mut queries: Rows = Vec::with_capacity(hidden.len());
        let mut biases: Rows = Vec::with_capacity(hidden.len());
        for (row, [xq, xk, xv]) in inputs.iter().enumerate() {
            let position = first_position + row as u64;
            let tau = attention.log_scale.map_or(1.0, |scale| {
                ops::log_position_scale(position, scale.floor, ops::decimal(scale.alpha))
            });
            let mut query = ops::linear(wq, xq);
            self.clip(&mut query, dim, &attention.query_clip, q_scale, tau);
            queries.push(query);

            let mut relative = ops::linear(wr, xq);
            self.clip(&mut relative, rank, &attention.relative.clip, r_scale, tau);
            biases.push(ops::relative_bias(&relative, table, rank, extent));

            let mut key = self.key_or_value(
                slot,
                &prefix,
                "wk_dv",
                "k_sconv",
                0,
                &attention.key_conv,
                xk,
            )?;
            self.clip(&mut key, dim, &key_clip, None, 1.0);
            let value = self.key_or_value(
                slot,
                &prefix,
                "wv_dv",
                "v_sconv",
                1,
                &attention.value_conv,
                xv,
            )?;
            let state = &mut self.slots[slot as usize];
            state.positions.push(position);
            state.keys.extend(key);
            for head in value.chunks_exact(dim) {
                state
                    .values
                    .extend(ops::rms_norm(head, None, value_epsilon));
            }
        }

        let shape = AttentionShape {
            query_heads: heads,
            kv_heads,
            head_dim: dim,
            window: attention.window.map(|window| window as usize),
            extent,
            mask_beyond_extent: attention.relative.beyond_extent == BeyondExtent::Masked,
        };
        let output_kernel = attention.output_conv.kernel as usize;
        let output_taps = self.tensor(
            &format!("{prefix}attn_sconv.weight"),
            &[hidden_size, 1, output_kernel],
        )?;
        let mut stream: Rows = Vec::with_capacity(hidden.len());
        for (row, [xq, _, _]) in inputs.iter().enumerate() {
            let state = &self.slots[slot as usize];
            let mut attended = ops::attend(
                &shape,
                &queries[row],
                &biases[row],
                &ops::linear(sink_weight, xq),
                first_position + row as u64,
                ops::CachedRows {
                    positions: &state.positions,
                    keys: &state.keys,
                    values: &state.values,
                },
            );
            for (value, gate) in attended.iter_mut().zip(ops::linear(gate_weight, xq)) {
                *value *= ops::sigmoid(gate);
            }
            let projected = ops::linear(wo, &attended);
            let convolved = ops::short_conv_static(
                &projected,
                &mut self.slots[slot as usize].conv[2],
                output_taps,
                output_kernel,
            );
            // The first layer of a block starts a new stream; the others add to theirs.
            stream.push(if residual.starts_block {
                convolved
            } else {
                hidden[row]
                    .iter()
                    .zip(&convolved)
                    .map(|(a, b)| a + b)
                    .collect()
            });
        }
        if residual.starts_block {
            blocks.push(hidden);
        }
        // A sliding slot keeps the rows a later query can still see.
        if let Some(window) = attention.window {
            let state = &mut self.slots[slot as usize];
            let keep = (window as usize).saturating_sub(1);
            if state.positions.len() > keep {
                let drop = state.positions.len() - keep;
                state.positions.drain(..drop);
                state.keys.drain(..drop * kv_width);
                state.values.drain(..drop * kv_width);
            }
        }

        // MLP: mix, controller, norm, routed experts, short convolution.
        let mlp_query = self.tensor(
            &format!("{LLM}attn_res.mlp_queries.{physical}"),
            &[hidden_size],
        )?;
        let mlp_norm = self.tensor(&format!("{prefix}mlp_norm.weight"), &[hidden_size])?;
        let mlp_epsilon = rms(&layer.pre_mlp_norm, at)?;
        let mlp_kernel = program.mlp_output_conv.kernel as usize;
        let mlp_taps = self.tensor(
            &format!("{prefix}mlp_sconv.weight"),
            &[hidden_size, 1, mlp_kernel],
        )?;
        let mut out: Rows = Vec::with_capacity(stream.len());
        for (row, current) in stream.iter().enumerate() {
            let mut mixed =
                ops::depth_mix(mlp_query, &states_at(blocks, current, row), mix_epsilon).remove(0);
            if let Some(messages) = &messages {
                mixed = self.controller(physical, "mlp", &mixed, &messages[row][3])?;
            }
            let normed = ops::rms_norm(&mixed, Some(mlp_norm), mlp_epsilon);
            let routed = self.experts(physical, &prefix, moe, &normed)?;
            let convolved = ops::short_conv_static(
                &routed,
                &mut self.slots[slot as usize].conv[3],
                mlp_taps,
                mlp_kernel,
            );
            out.push(current.iter().zip(&convolved).map(|(a, b)| a + b).collect());
        }
        Ok(out)
    }

    /// The routed experts of one row, on the input the plan declares for the layer.
    fn experts(
        &self,
        physical: u32,
        prefix: &str,
        moe: &MoeMlpPlan,
        x: &[f64],
    ) -> Result<Vec<f64>, ReferenceError> {
        let hidden = x.len();
        let experts = moe.expert_count as usize;
        let RouterPlan::Sigmoid {
            scaling_factor,
            selection_bias,
            ..
        } = moe.router
        else {
            return Err(unsupported(
                Some(physical),
                "a router other than normalized sigmoid",
            ));
        };
        let ActivationPlan::SwiGluSoftcapped { gate_cap, up_cap } = moe.activation else {
            return Err(unsupported(
                Some(physical),
                "an activation other than softcapped SwiGLU",
            ));
        };
        let bias = if selection_bias {
            Some(self.tensor(&format!("{prefix}mlp.gate.bias"), &[experts])?)
        } else {
            None
        };
        let (selected, weights) = ops::route(
            x,
            self.tensor(&format!("{prefix}mlp.gate.weight"), &[experts, hidden])?,
            bias,
            moe.experts_per_token as usize,
            ops::decimal(scaling_factor),
        );
        let numerics = &self.program.numerics;
        let (gate_up, down) = &self.banks[physical as usize];
        match numerics.experts[physical as usize].activation {
            ExpertActivationMode::Float => {
                let caps = (ops::decimal(gate_cap), ops::decimal(up_cap));
                let mut out = vec![0.0; hidden];
                for (&expert, &weight) in selected.iter().zip(&weights) {
                    let row = ops::expert_float(
                        x,
                        &gate_up.expert(expert),
                        &down.expert(expert),
                        weight,
                        caps,
                        numerics.gate_up_interleaved,
                        numerics.routed_weight_before_down,
                    );
                    for (o, v) in out.iter_mut().zip(row) {
                        *o += v;
                    }
                }
                Ok(out)
            }
            ExpertActivationMode::Declared => {
                // The declared program reads a bf16 row and fp32 routing weights.
                let row: Vec<f32> = x
                    .iter()
                    .map(|value| actquant::bf16(*value as f32))
                    .collect();
                let gamma: Vec<f32> = weights.iter().map(|weight| *weight as f32).collect();
                let situ = situ_tables(gate_cap, up_cap);
                let out = match (gate_up, down) {
                    (
                        ExpertBank::Nvfp4 {
                            codes,
                            scales,
                            amax,
                            denominator,
                            rows,
                            inputs,
                        },
                        ExpertBank::Nvfp4 {
                            codes: down_codes,
                            scales: down_scales,
                            amax: down_amax,
                            denominator: down_denominator,
                            rows: down_rows,
                            inputs: down_inputs,
                        },
                    ) => actquant::expert_block_w4a4(
                        &row,
                        &selected,
                        &gamma,
                        &actquant::Nvfp4Experts {
                            codes,
                            scales,
                            amax,
                            denominator: *denominator,
                            n: *rows,
                            k: *inputs,
                        },
                        &actquant::Nvfp4Experts {
                            codes: down_codes,
                            scales: down_scales,
                            amax: down_amax,
                            denominator: *down_denominator,
                            n: *down_rows,
                            k: *down_inputs,
                        },
                        &situ,
                        actquant::Accumulation::IndexOrder,
                    ),
                    (
                        ExpertBank::Mxfp8 {
                            values,
                            exponents,
                            rows,
                            inputs,
                        },
                        ExpertBank::Mxfp8 {
                            values: down_values,
                            exponents: down_exponents,
                            rows: down_rows,
                            inputs: down_inputs,
                        },
                    ) => actquant::expert_block_w8a8(
                        &row,
                        &selected,
                        &gamma,
                        &actquant::Mxfp8Experts {
                            values,
                            exponents,
                            n: *rows,
                            k: *inputs,
                        },
                        &actquant::Mxfp8Experts {
                            values: down_values,
                            exponents: down_exponents,
                            n: *down_rows,
                            k: *down_inputs,
                        },
                        &situ,
                        actquant::Accumulation::IndexOrder,
                    ),
                    _ => {
                        return Err(invalid(
                            Some(physical),
                            "both expert banks of a layer are one format",
                        ));
                    }
                };
                Ok(out.out.into_iter().map(f64::from).collect())
            }
        }
    }

    fn state(&self) -> ReferenceState {
        ReferenceState {
            layers: self
                .plan
                .layers
                .iter()
                .zip(&self.slots)
                .map(|(layer, state)| {
                    let narrow = |values: &[f64]| values.iter().map(|v| *v as f32).collect();
                    let attention = attention_of(layer).expect("checked at session start");
                    ReferenceLayerState::RelativeBias {
                        key: narrow(&state.keys),
                        value: narrow(&state.values),
                        positions: state.positions.clone(),
                        kv_heads: attention.kv_heads as usize,
                        head_dim: attention.head_dim as usize,
                        window: attention.window.map(|window| window as usize),
                        conv: state
                            .conv
                            .iter()
                            .map(|history| narrow(&history.flat()))
                            .collect(),
                    }
                })
                .collect(),
        }
    }
}

/// The states one row mixes over: every block output at that row, then the running stream.
fn states_at<'s>(blocks: &'s [Rows], stream: &'s [f64], row: usize) -> Vec<&'s [f64]> {
    blocks
        .iter()
        .map(|block| block[row].as_slice())
        .chain(std::iter::once(stream))
        .collect()
}

/// Run `token_ids` as one request under a schedule: prefill chunks ending at each of
/// `chunk_ends` and at `decode_from` (the end of the tokens when `None`), then every later token as
/// one decode step. Logit rows are `[tokens, unpadded vocabulary]`: truncation drops the padded
/// ids instead of masking them.
pub fn execute_schedule(
    plan: &ModelPlan,
    weights: &ReferenceWeights,
    token_ids: &[u32],
    chunk_ends: &[usize],
    decode_from: Option<usize>,
) -> Result<ReferenceOutput, ReferenceError> {
    if token_ids.is_empty() {
        return Err(ReferenceError::EmptyInput);
    }
    let prefill = decode_from.unwrap_or(token_ids.len());
    if prefill == 0 || prefill > token_ids.len() {
        return Err(invalid(
            None,
            "the prefill is a non-empty prefix of the tokens",
        ));
    }
    let mut ends = chunk_ends.to_vec();
    ends.push(prefill);
    if ends[0] == 0 || ends.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err(invalid(
            None,
            "prefill chunks end at increasing positions inside the prefill",
        ));
    }
    ends.extend(prefill + 1..=token_ids.len());
    let mut session = Session::new(plan, weights)?;
    let mut logits = Vec::new();
    let mut layer_hidden = vec![Vec::new(); plan.layers.len()];
    let mut start = 0;
    for end in ends {
        let chunk = session.forward(&token_ids[start..end])?;
        logits.extend(chunk.logits);
        for (all, rows) in layer_hidden.iter_mut().zip(chunk.layer_hidden) {
            all.extend(rows);
        }
        start = end;
    }
    Ok(ReferenceOutput {
        logits,
        tokens: token_ids.len(),
        vocab: session.keep,
        state: session.state(),
        mtp: Vec::new(),
        draft: None,
        layer_hidden,
        decode: decode_from.map(|prefill_tokens| ReferenceDecodeTrace {
            prefill_tokens,
            drafts: Vec::new(),
        }),
    })
}

/// [`execute_schedule`] with one prefill chunk.
pub fn execute(
    plan: &ModelPlan,
    weights: &ReferenceWeights,
    token_ids: &[u32],
    decode_from: Option<usize>,
) -> Result<ReferenceOutput, ReferenceError> {
    execute_schedule(plan, weights, token_ids, &[], decode_from)
}

// ============================ tiny fixture ============================

/// The tiny checkpoint: the family's real layout (`model.llm.*` names, BF16 / F32 / U8 /
/// F8_E4M3 storage, NVFP4 experts on layers 0-5 and MXFP8 on 6-7, with zero and subnormal block
/// scales, signed zeros and subnormal values on purpose) at hidden 64. Seeded random bytes, written
/// by the oracle lane's generator, not by this crate.
pub const TINY_CHECKPOINT: &[u8] = include_bytes!("fixtures/inkling2-tiny-checkpoint.bin");

/// The tiny request: 40 ids, past the sliding window (8), the relative extent (12) and the log
/// scale floor (24) of the tiny config.
pub const TINY_TOKENS: [u32; 40] = [
    13, 46, 117, 229, 235, 19, 168, 136, 41, 149, 46, 6, 48, 273, 197, 43, 264, 225, 292, 199, 230,
    212, 162, 92, 92, 141, 194, 37, 182, 6, 96, 140, 256, 230, 174, 288, 79, 87, 150, 124,
];

/// Reference weights from a safetensors image in the checkpoint's layout. BF16 and F32 tensors
/// become exact f32 tensors; U8 and F8_E4M3 tensors (expert codes, values and scales) keep their
/// bytes as integers.
pub fn weights_from_safetensors(image: &[u8]) -> Result<ReferenceWeights, ReferenceError> {
    let bad = |reason: &'static str| invalid(None, reason);
    let header_len = image
        .get(..8)
        .map(|bytes| u64::from_le_bytes(bytes.try_into().expect("8 bytes")) as usize)
        .ok_or(bad("a safetensors image starts with its header length"))?;
    let header = image
        .get(8..8 + header_len)
        .and_then(|bytes| std::str::from_utf8(bytes).ok())
        .ok_or(bad("a safetensors header is UTF-8 inside the image"))?;
    let data = &image[8 + header_len..];
    let entries = memra_gguf::safetensors::parse_header_json_checked(header)
        .map_err(|_| bad("a safetensors header parses"))?;
    let mut weights = ReferenceWeights::new();
    for (name, info) in entries {
        let shape: Vec<usize> = info.shape.iter().map(|dim| *dim as usize).collect();
        let raw = data
            .get(info.data_offsets[0]..info.data_offsets[1])
            .ok_or(bad("a safetensors tensor lies inside the image"))?;
        let tensor = match info.dtype.as_str() {
            "BF16" => ReferenceTensor::new(
                shape,
                raw.chunks_exact(2)
                    .map(|pair| {
                        f32::from_bits(u32::from(u16::from_le_bytes([pair[0], pair[1]])) << 16)
                    })
                    .collect(),
            )?,
            "F32" => ReferenceTensor::new(
                shape,
                raw.chunks_exact(4)
                    .map(|quad| f32::from_le_bytes(quad.try_into().expect("4 bytes")))
                    .collect(),
            )?,
            "U8" | "F8_E4M3" => {
                ReferenceTensor::new_i64(shape, raw.iter().map(|byte| i64::from(*byte)).collect())?
            }
            _ => return Err(bad("an inkling2 tensor is BF16, F32, U8 or F8_E4M3")),
        };
        weights.insert(tensor_id(&name), tensor);
    }
    Ok(weights)
}

/// The tiny gate's tolerance for the `declared` variant, and for nothing else. The variant's
/// oracle runs an f32 stream and this executor's trunk is f64; on the tiny fixture the two are
/// 2.3e-6 apart at most. The float-activation runs are gated on bit equality. This is not a
/// checkpoint-parity band: the pack declares none.
pub const TINY_DECLARED_GATE: TinyGate = TinyGate::Band {
    max_abs: 1e-5,
    max_rel: 1e-5,
};

fn tiny_fixture(
    chunk_ends: &[usize],
    decode_from: Option<usize>,
    tiny_gate: TinyGate,
) -> Result<ReferenceFixture, ReferenceError> {
    let schedule = format!(
        "{{\n  \"input_ids\": {:?},\n  \"chunk_ends\": {:?},\n  \"decode_from\": {}\n}}\n",
        TINY_TOKENS,
        chunk_ends,
        decode_from.map_or("null".to_owned(), |at| at.to_string()),
    );
    Ok(ReferenceFixture {
        token_ids: TINY_TOKENS.to_vec(),
        weights: weights_from_safetensors(TINY_CHECKPOINT)?,
        vision: None,
        multimodal_token_ids: None,
        decode_from,
        chunk_ends: chunk_ends.to_vec(),
        tiny_gate,
        checkpoint_files: vec![
            ("model.safetensors".into(), TINY_CHECKPOINT.to_vec()),
            ("config.json".into(), pack::TINY_CONFIG.as_bytes().to_vec()),
            (
                "hf_quant_config.json".into(),
                pack::TINY_QUANT_CONFIG.as_bytes().to_vec(),
            ),
            ("schedule.json".into(), schedule.into_bytes()),
        ],
    })
}

/// The tiny gate's main fixture: [`TINY_CHECKPOINT`] and [`TINY_TOKENS`] as one prefill chunk,
/// with the checkpoint image an external oracle loads to run the source program on the same
/// bytes.
pub fn fixture(plan: &ModelPlan) -> Result<ReferenceFixture, ReferenceError> {
    program(plan)?;
    tiny_fixture(&[], None, TinyGate::Exact)
}

/// The tiny gate's variants: the same plan, weights and tokens under schedules that move slot
/// state across a call boundary at each position rule of the program, each pinned to the source
/// program run under the same schedule.
///
/// - `splits`: prefill chunks ending at 8, 12 and 24: a call boundary exactly at the sliding
///   window, at the relative extent and at the log-scale floor;
/// - `offsplits`: chunks ending at 7, 11 and 23, so each rule is crossed inside a chunk that
///   starts from carried state;
/// - `steps`: a prefill of 5, then one token per call across all three.
///
/// And one variant of another plan: `declared`, the tiny config under the family's default
/// class (every layer on its declared activation quantizer), as one prefill chunk. No
/// float-activation oracle computes that class; it is pinned to a CPU simulation of the
/// activation-quant expert program on an f32 stream, an implementation written apart from
/// this one.
pub fn tiny_variants(plan: &ModelPlan) -> Result<Vec<crate::TinyVariant>, ReferenceError> {
    program(plan)?;
    let variant = |name, chunk_ends: &[usize], decode_from| {
        Ok(crate::TinyVariant {
            name,
            plan: plan.clone(),
            fixture: tiny_fixture(chunk_ends, decode_from, TinyGate::Exact)?,
        })
    };
    Ok(vec![
        variant("splits", &[8, 12, 24], None)?,
        variant("offsplits", &[7, 11, 23], None)?,
        variant("steps", &[], Some(5))?,
        crate::TinyVariant {
            name: "declared",
            plan: pack::tiny_declared_plan()
                .map_err(|_| invalid(None, "the inkling2 tiny config compiles its default plan"))?,
            fixture: tiny_fixture(&[], None, TINY_DECLARED_GATE)?,
        },
    ])
}

#[cfg(test)]
mod tests;
