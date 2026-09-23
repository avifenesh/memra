//! Standalone NextN artifacts have a separate contract from complete text models.
use super::consumer::{FfnActivation, step_mtp_clamp, validate_nvfp4_macro_metadata};
use super::*;
use crate::model_plan::{
    AttentionPlan, AttentionScale, MlpPlan, MtpBlockPlan, MtpFusionPlan, ResidualTopology,
    RopeFactors, ValueNorm, ValueProjection,
};
use crate::tensor_contract::{
    MtpTensor, QuantConstraint, TensorMatch, TensorOwner, TensorRequirement,
};
use std::collections::BTreeSet;

pub mod composite;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DraftGeometry {
    Natural,
    Student {
        hidden_size: u32,
        query_heads: u32,
        kv_heads: u32,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DraftVocabulary {
    Full {
        target_size: u32,
    },
    Trimmed {
        target_size: u32,
        token_ids: Vec<u32>,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct ExternalDraftBlock {
    pub layer: MtpBlockPlan,
    pub geometry: DraftGeometry,
}

/// Compiler-owned standalone GGUF draft. A sparse draft never receives model-root authority.
/// All declared blocks and optional trunk copies are validated; only depth zero is selected,
/// matching the existing external-draft loader. Later heads remain inventoried and identity-bound.
pub struct PreparedExternalDraftSource<'a> {
    bound: BoundTensorSource<'a>,
    blocks: Vec<ExternalDraftBlock>,
    head_name: String,
    norm_name: Option<String>,
    vocabulary: DraftVocabulary,
}

fn map_id() -> TensorId {
    TensorId::Family {
        family: "external_nextn",
        key: "token_map".into(),
    }
}
fn up_id(depth: u32) -> TensorId {
    TensorId::Family {
        family: "external_nextn",
        key: format!("output_up/{depth}"),
    }
}
fn mtp_id(depth: u32, tensor: MtpTensor) -> TensorId {
    TensorId::Mtp { depth, tensor }
}
fn base_id(id: &TensorId) -> &TensorId {
    match id {
        TensorId::QuantAux { tensor, .. } => base_id(tensor),
        _ => id,
    }
}
fn block_tensor(id: &TensorId, block: &MtpBlockPlan) -> bool {
    match base_id(id) {
        TensorId::Layer { index, .. } => *index == block.layer.index,
        TensorId::Mtp { depth, .. } => *depth == block.depth,
        id => *id == up_id(block.depth),
    }
}
fn extra_requirement(
    id: TensorId,
    name: String,
    shape: Vec<u64>,
    quant: QuantConstraint,
) -> TensorRequirement {
    TensorRequirement {
        id,
        names: vec![name],
        match_mode: TensorMatch::OneOf,
        shape,
        owner: TensorOwner::Mtp(0),
        transform: TensorTransform::Identity,
        quant,
        auxiliaries: Some(Vec::new()),
        required: true,
    }
}

impl<'a> PreparedExternalDraftSource<'a> {
    pub fn compile(source: &'a dyn TensorSource, target: &ModelConfig) -> Result<Self, String> {
        Self::compile_with_inventory(source, target, |_| Ok(()))
    }

    fn compile_with_inventory(
        source: &'a dyn TensorSource,
        target: &ModelConfig,
        validate_inventory: impl FnOnce(&TensorContract) -> Result<(), String>,
    ) -> Result<Self, String> {
        if source.bound_program().is_some()
            || source.composite_input().is_some()
            || source.bound_interpretation()? != BoundSourceInterpretation::Gguf
        {
            return Err("external NextN requires an opened standalone GGUF source".into());
        }
        let config = source.try_config()?;
        let target_plan = model_packs::compile_for_load(target).map_err(|e| e.to_string())?;
        let mut plan = model_packs::compile_for_load(&config).map_err(|e| e.to_string())?;
        if plan.mtp_blocks.is_empty() {
            return Err("external draft has no typed NextN blocks".into());
        }
        if config.n_embd != target.n_embd
            || config.head_dim_k != target.head_dim_k
            || config.head_dim_v != target.head_dim_v
        {
            return Err("external draft hidden/head interface differs from target".into());
        }
        if target.n_vocab == 0 || (config.n_vocab != 0 && config.n_vocab != target.n_vocab) {
            return Err("external draft declared vocabulary differs from target".into());
        }
        // Some standalone GGUFs omit both a vocab key and the unused embedding copy. Their
        // output interface is still checked against the explicit target, never inferred by shape.
        plan.vocab_size = target.n_vocab;
        if plan.vision.is_some() || plan.multimodal.is_some() || plan.speech.is_some() {
            return Err("standalone NextN has no multimodal draft contract".into());
        }
        let mut census = source.tensor_census()?;
        if census.dialect != CheckpointDialect::Gguf {
            return Err("external NextN has no contract for this checkpoint dialect".into());
        }
        census
            .tensors
            .sort_by(|a, b| a.entry.name.cmp(&b.entry.name));
        let mut by_name = BTreeMap::new();
        for (i, row) in census.tensors.iter().enumerate() {
            if by_name.insert(row.entry.name.clone(), i).is_some() {
                return Err(format!("duplicate draft tensor {}", row.entry.name));
            }
            if !row.auxiliaries.is_empty() || !row.entry.auxiliaries.is_empty() {
                return Err("GGUF draft auxiliaries must be separate declared rows".into());
            }
        }
        let rows = &by_name;
        let pack = model_packs::for_config(&config).ok_or("draft has no model pack")?;
        // A draft's private/file head is an explicit output, independent of the target's tied head.
        let options = ContractOptions {
            output_head: OutputHead::Separate,
        };
        let mut contract = pack
            .compile_tensor_contract(&config, &plan, census.dialect, options)
            .map_err(|e| e.to_string())?;
        let mut blocks = Vec::new();
        for block in plan.mtp_blocks.clone() {
            if block.input.fusion != MtpFusionPlan::ConcatenateProjection {
                return Err("external NextN requires its own concat-fusion contract".into());
            }
            let n = block.layer.index;
            let up_name = format!("blk.{n}.nextn.out_up.weight");
            let mut selected = block.clone();
            let geometry = if rows.contains_key(&up_name) {
                let fusion_name = format!("blk.{n}.nextn.eh_proj.weight");
                let fusion = &census.tensors
                    [*rows.get(&fusion_name).ok_or("student fusion is missing")?]
                .entry;
                let [input, inner] = fusion.shape.as_slice() else {
                    return Err("student fusion must be a matrix".into());
                };
                let inner = u32::try_from(*inner).map_err(|_| "student width exceeds u32")?;
                if *input != 2 * u64::from(config.n_embd) || inner == 0 || inner >= config.n_embd {
                    return Err(
                        "student fusion must map 2*target_hidden to a narrower positive width"
                            .into(),
                    );
                }
                let mut inner_config = config.clone();
                inner_config.n_embd = inner;
                let inner_plan =
                    model_packs::compile_for_load(&inner_config).map_err(|e| e.to_string())?;
                selected.layer = inner_plan.mtp_blocks[block.depth as usize].layer.clone();
                if !matches!(&selected.layer.attention, AttentionPlan::Full(_))
                    || !matches!(&selected.layer.mlp, MlpPlan::Dense(_))
                    || config.step35.is_some()
                {
                    return Err(
                        "student output-up contract requires a dense full-attention block".into(),
                    );
                }
                let inner_contract = pack
                    .compile_tensor_contract(&inner_config, &inner_plan, census.dialect, options)
                    .map_err(|e| e.to_string())?;
                // Only the inner block changes width. Input norms, output head/norm and the
                // recurrent carrier remain at the target interface width.
                for r in &mut contract.requirements {
                    if matches!(base_id(&r.id),TensorId::Layer { index,.. } if *index==n) {
                        *r = inner_contract
                            .requirements
                            .iter()
                            .find(|x| x.id == r.id)
                            .ok_or("student inner block lost a tensor role")?
                            .clone();
                    } else if r.id == mtp_id(block.depth, MtpTensor::FusionProjection) {
                        r.shape = vec![2 * u64::from(config.n_embd), u64::from(inner)];
                    }
                }
                let mut up = extra_requirement(
                    up_id(block.depth),
                    up_name,
                    vec![u64::from(inner), u64::from(config.n_embd)],
                    QuantConstraint::Weight,
                );
                up.owner = TensorOwner::Mtp(block.depth);
                // Quant scale rows belong to this projection just as they do to every GGUF weight.
                up.auxiliaries = None;
                contract.requirements.push(up);
                for (kind, suffix) in [
                    (QuantAuxTensor::WeightScale, "scale"),
                    (QuantAuxTensor::InputScale, "input_scale"),
                ] {
                    let mut aux = extra_requirement(
                        TensorId::QuantAux {
                            tensor: Box::new(up_id(block.depth)),
                            kind,
                        },
                        format!("blk.{n}.nextn.out_up.{suffix}"),
                        vec![1],
                        QuantConstraint::FloatOnly,
                    );
                    aux.owner = TensorOwner::Mtp(block.depth);
                    aux.required = false;
                    contract.requirements.push(aux);
                }
                DraftGeometry::Student {
                    hidden_size: inner,
                    query_heads: config.n_head,
                    kv_heads: config.n_head_kv,
                }
            } else {
                DraftGeometry::Natural
            };
            validate_target_block(&selected, geometry, &config, target)?;
            blocks.push(ExternalDraftBlock {
                layer: selected,
                geometry,
            });
        }
        plan.mtp_blocks = blocks.iter().map(|b| b.layer.clone()).collect();
        let first = &blocks[0].layer;
        let head_private = mtp_id(first.depth, MtpTensor::OutputProjection);
        let norm_private = mtp_id(first.depth, MtpTensor::OutputNorm);
        let present = |id: &TensorId| {
            contract
                .requirements
                .iter()
                .find(|r| &r.id == id)
                .is_some_and(|r| r.names.iter().any(|name| rows.contains_key(name)))
        };
        let head_id = if present(&head_private) {
            head_private
        } else {
            TensorId::OutputProjection
        };
        let norm_id = if present(&norm_private) {
            Some(norm_private)
        } else if present(&TensorId::OutputNorm) {
            Some(TensorId::OutputNorm)
        } else {
            None
        };
        let trimmed = rows.get("d2t").map(|i| &census.tensors[*i].entry);
        let head_rows = if let Some(map) = trimmed {
            let [count] = map.shape.as_slice() else {
                return Err("d2t must have one dimension".into());
            };
            if *count == 0 || *count > u64::from(target.n_vocab) {
                return Err("d2t row count exceeds target vocabulary or is empty".into());
            }
            *count
        } else {
            u64::from(target.n_vocab)
        };
        let has_map = trimmed.is_some();
        for r in &mut contract.requirements {
            // Trunk copies are optional inventory, never implicit executable draft inputs.
            if !blocks.iter().any(|b| block_tensor(&r.id, &b.layer))
                && r.id != TensorId::RopeFactors
            {
                r.required = false;
            }
            if r.id == head_id {
                r.required = true;
                r.shape = vec![u64::from(config.n_embd), head_rows];
            }
        }
        if has_map {
            contract.requirements.push(extra_requirement(
                map_id(),
                "d2t".into(),
                vec![head_rows],
                QuantConstraint::I32OrI64,
            ));
        }
        let entries: Vec<_> = census.tensors.iter().map(|r| r.entry.clone()).collect();
        contract.declared_expert_members(&plan, &entries)?;
        validate_inventory(&contract)?;
        let binding = contract.bind(&entries).map_err(|e| e.to_string())?;
        for (id, b) in &binding.tensors {
            for name in &b.checkpoint_names {
                source.validate_bound_metadata(&BoundTensorRequest {
                    id,
                    record: &census.tensors[by_name[name]],
                    dialect: census.dialect,
                    transform: b.transform,
                    view: BoundTensorView::Whole,
                })?;
            }
        }
        let head_name = binding.tensors[&head_id].checkpoint_names[0].clone();
        let norm_name = norm_id
            .as_ref()
            .map(|id| binding.tensors[id].checkpoint_names[0].clone());
        let selected = contract
            .requirements
            .iter()
            .filter(|r| {
                let id = base_id(&r.id);
                (block_tensor(id, first)
                    && *id != mtp_id(first.depth, MtpTensor::OutputProjection)
                    && *id != mtp_id(first.depth, MtpTensor::OutputNorm))
                    || *id == head_id
                    || Some(id) == norm_id.as_ref()
                    || *id == map_id()
                    || *id == TensorId::RopeFactors
            })
            .map(|r| r.id.clone())
            .collect();
        let scope = CatalogScope {
            requested: LoadScope::Text,
            selected,
            inventory_only: Vec::new(),
        };
        for id in &scope.selected {
            let TensorId::QuantAux {
                tensor,
                kind: QuantAuxTensor::WeightScale,
            } = id
            else {
                continue;
            };
            let (Some(owner), Some(scale)) = (
                binding.tensors.get(tensor.as_ref()),
                binding.tensors.get(id),
            ) else {
                continue;
            };
            let nvfp4 = owner.checkpoint_names.iter().any(|name| matches!(
                &census.tensors[by_name[name]].entry.storage,
                crate::tensor_contract::StorageLayout::Quantized(layout) if layout.format == "NVFP4"
            ));
            if nvfp4 {
                for name in &scale.checkpoint_names {
                    validate_nvfp4_macro_metadata(&census.tensors[by_name[name]].entry)?;
                }
            }
        }
        // Complete metadata validation precedes the only payload reads at preparation: the small
        // token map and required RoPE factors. We never read a model weight to infer its role.
        let runtime_metadata = source.runtime_metadata()?;
        let mut bound = BoundTensorSource {
            source,
            config,
            plan,
            contract,
            options,
            binding,
            census,
            by_name,
            digest: String::new(),
            interpretation: BoundSourceInterpretation::Gguf,
            runtime_metadata,
            active_experts: BTreeMap::new(),
            scope,
            disk_cache: BoundDiskCache::default(),
        };
        let factors = if bound.contains(&TensorId::RopeFactors) {
            Some(bound.tensor(&TensorId::RopeFactors)?)
        } else {
            None
        };
        let mut config = bound.config.clone();
        model_packs::prepare_bound_rope_factors(
            &mut config,
            &bound.plan,
            &DraftFactors {
                config: &bound.config,
                factors: factors.as_ref(),
            },
        )
        .map_err(|e| e.to_string())?;
        drop(factors);
        bound.config = config;
        let vocabulary = if has_map {
            let view = bound.tensor(&map_id())?;
            DraftVocabulary::Trimmed {
                target_size: target.n_vocab,
                token_ids: decode_token_map(&view, head_rows, target.n_vocab)?,
            }
        } else {
            DraftVocabulary::Full {
                target_size: target.n_vocab,
            }
        };
        let binding_hash = binding_digest(
            &bound.config,
            &bound.plan,
            &bound.contract,
            bound.options,
            &bound.binding,
            &bound.census,
            &bound.interpretation,
            &bound.runtime_metadata,
            &bound.scope,
        );
        let mut hash = Sha256::new();
        hash.update(b"memra-external-nextn-source-v1\0");
        for section in [
            binding_hash,
            format!("{blocks:?}"),
            format!("{vocabulary:?}"),
            format!("{target_plan:?}"),
            format!("{target:?}"),
        ] {
            hash.update((section.len() as u64).to_le_bytes());
            hash.update(section.as_bytes());
        }
        bound.digest = format!("{:x}", hash.finalize());
        Ok(Self {
            bound,
            blocks,
            head_name,
            norm_name,
            vocabulary,
        })
    }
    pub fn config(&self) -> &ModelConfig {
        self.bound.config()
    }
    pub fn declared_plan(&self) -> &ModelPlan {
        self.bound.plan()
    }
    pub fn blocks(&self) -> &[ExternalDraftBlock] {
        &self.blocks
    }
    pub fn head_name(&self) -> &str {
        &self.head_name
    }
    pub fn norm_name(&self) -> Option<&str> {
        self.norm_name.as_deref()
    }
    pub fn vocabulary(&self) -> &DraftVocabulary {
        &self.vocabulary
    }
    pub fn token_map(&self) -> Option<&[u32]> {
        match &self.vocabulary {
            DraftVocabulary::Trimmed { token_ids, .. } => Some(token_ids),
            _ => None,
        }
    }
    pub fn artifact_identity(&self) -> Result<BoundArtifactIdentity, String> {
        self.bound.artifact_identity()
    }
    pub fn with_runtime<T>(&self, load: impl FnOnce(&dyn TensorSource) -> T) -> Result<T, String> {
        Ok(load(&self.bound.external_draft_runtime()?))
    }
}

struct DraftFactors<'a, 'b> {
    config: &'a ModelConfig,
    factors: Option<&'a TensorView<'b>>,
}
impl TensorSource for DraftFactors<'_, '_> {
    fn config(&self) -> ModelConfig {
        self.config.clone()
    }
    fn find(&self, name: &str) -> Option<TensorView<'_>> {
        if name != "rope_freqs.weight" {
            return None;
        }
        self.factors.map(|view| TensorView {
            bytes: std::borrow::Cow::Borrowed(&view.bytes),
            ggml_type: view.ggml_type,
            ne: view.ne.clone(),
        })
    }
}

fn validate_target_block(
    block: &MtpBlockPlan,
    geometry: DraftGeometry,
    source: &ModelConfig,
    target: &ModelConfig,
) -> Result<(), String> {
    if block.layer.residual != ResidualTopology::Serial
        || source.rms_eps.to_bits() != target.rms_eps.to_bits()
    {
        return Err("external draft residual/norm program differs from the draft executor".into());
    }
    let attention = match &block.layer.attention {
        AttentionPlan::Full(a) | AttentionPlan::SlidingWindow { attention: a, .. } => a,
        _ => return Err("external NextN mixer has no supported draft execution contract".into()),
    };
    if attention.value_projection != ValueProjection::Separate
        || attention.value_norm != ValueNorm::None
        || attention.scale != AttentionScale::InverseSqrtKeyDim
    {
        return Err("external draft attention has no exact executor contract".into());
    }
    if source.step35.is_some() != target.step35.is_some() {
        return Err("external draft sliding-gated program differs from target".into());
    }
    if source.step35.is_some() {
        if attention.kv_heads != target.n_head_kv {
            return Err("external draft KV heads differ from target scratch".into());
        }
    } else {
        // The ordinary MTP kernel reads these values from the TARGET config. The student
        // contract overrides only inner width/head counts; it cannot override gate or RoPE.
        let expected = target
            .full_attention_geometry_at(target.n_layer.saturating_sub(target.nextn_predict_layers));
        if !matches!(block.layer.attention, AttentionPlan::Full(_))
            || attention.output_gate != expected.attention_gate
            || attention.key_head_dim != expected.head_dim_k
            || attention.value_head_dim != expected.head_dim_v
            || attention.rope.dimensions != expected.n_rot
            || attention.rope.base.to_bits() != expected.rope_base.to_bits()
            || !matches!(
                attention.rope.factors,
                RopeFactors::None | RopeFactors::PartialRotary { .. }
            )
            || target.gemma4.is_some()
        {
            return Err(
                "external draft attention/RoPE program differs from target executor".into(),
            );
        }
        if geometry == DraftGeometry::Natural
            && (attention.query_heads != expected.n_head
                || attention.kv_heads != expected.n_head_kv)
        {
            return Err("external draft head counts differ from target".into());
        }
    }
    validate_ffn_activation(block, source, target)
}

fn validate_ffn_activation(
    block: &MtpBlockPlan,
    source: &ModelConfig,
    target: &ModelConfig,
) -> Result<(), String> {
    match &block.layer.mlp {
        MlpPlan::Dense(dense) => {
            // spec.rs reads the target config, plus the source block's resolved Step limit.
            let clamp = source
                .step35
                .as_ref()
                .and_then(|_| step_mtp_clamp(&dense.activation))
                .map(crate::config::SwigluClamp::Post);
            FfnActivation::for_target(target, clamp).validate_declared(&dense.activation)
        }
        MlpPlan::Moe(moe) => {
            // MTP's routed/shared path uses the existing external-block cache key u16::MAX.
            let index = u32::from(u16::MAX);
            FfnActivation::for_target(target, target.clamp_exp_at(index))
                .validate_declared(&moe.activation)?;
            if moe.shared.is_some() {
                let declared =
                    FfnActivation::for_target(source, source.clamp_shexp_at(block.layer.index));
                let executed = FfnActivation::for_target(target, target.clamp_shexp_at(index));
                if declared != executed {
                    return Err(format!(
                        "external draft shared FFN declares {declared:?}, but target-driven execution selects {executed:?}"
                    ));
                }
            }
            Ok(())
        }
    }
}

fn decode_token_map(view: &TensorView<'_>, rows: u64, target: u32) -> Result<Vec<u32>, String> {
    let ids = super::ranks::decode_integer_ids(view, rows)?;
    let mut seen = BTreeSet::new();
    for &id in &ids {
        if id >= target {
            return Err("d2t token ID exceeds target vocabulary".into());
        }
        if !seen.insert(id) {
            return Err("d2t token IDs must be unique".into());
        }
    }
    Ok(ids)
}

#[cfg(test)]
mod tests;
