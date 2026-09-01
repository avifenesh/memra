//! Model-specific parallel topology contracts.
//!
//! The rank planner is reusable, but model support is never inferred from a loader or a few
//! scalar dimensions. Each family must register the complete geometry that its TP/EP program
//! shards. Step-3.7-Flash is the first registered contract because its query-head count varies by
//! layer (64 full-attention / 96 sliding-attention), while KV heads stay at 8. Step-3.5 and other
//! siblings do not inherit this contract merely because they share the `step35` architecture tag.

use std::fmt;
use std::ops::Range;

use memra_gguf::config::ModelConfig;
use memra_gguf::source::TensorSource;

/// The execution planner's supported rank envelope. Hardware qualification and tuned defaults
/// remain model x rig evidence, but the placement/runtime contract must not stop at earlier
/// three-card qualification cells.
pub const PRODUCT_MAX_CARDS: usize = 8;
pub const STEP37_TRUNK_LAYERS: usize = 45;
const STEP_FP8_BLOCK: usize = 128;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HardwareTarget {
    Rtx5090,
    RtxPro6000Blackwell,
}

impl HardwareTarget {
    fn max_cards(self) -> usize {
        match self {
            Self::Rtx5090 => 1,
            Self::RtxPro6000Blackwell => PRODUCT_MAX_CARDS,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Rtx5090 => "rtx-5090",
            Self::RtxPro6000Blackwell => "rtx-pro-6000-blackwell",
        }
    }

    fn from_device_name(name: &str) -> Result<Self, TopologyError> {
        if name.contains("RTX PRO 6000") && name.contains("Blackwell") {
            return Ok(Self::RtxPro6000Blackwell);
        }
        if name.contains("RTX 5090") {
            return Ok(Self::Rtx5090);
        }
        Err(TopologyError::new(format!(
            "unqualified CUDA device {name:?}; first-class targets are RTX 5090 and RTX PRO 6000 \
             Blackwell"
        )))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TopologyRequest {
    pub pipeline: usize,
    pub tensor: usize,
    /// Routed experts are partitioned across the TP group. When false, each rank owns every
    /// expert and tensor-shards the expert projections instead.
    pub expert_parallel: bool,
    pub available_devices: usize,
    pub hardware: HardwareTarget,
}

impl TopologyRequest {
    pub fn world_size(self) -> Result<usize, TopologyError> {
        self.pipeline
            .checked_mul(self.tensor)
            .ok_or_else(|| TopologyError::new("PP x TP world size overflow"))
    }
}

/// One pipeline stage's model-specific tensor/expert group.
///
/// Stage groups make odd physical card counts useful without pretending PP is TP. For example,
/// three cards can run a TP1 dense-prefix stage followed by a TP2/EP2 MoE stage. The layer range is
/// explicit because memory-balanced Step placement is not necessarily an equal layer split.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StageGroupRequest {
    pub layers: Range<usize>,
    pub tensor: usize,
    pub expert_parallel: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GroupedTopologyRequest {
    pub stages: Vec<StageGroupRequest>,
    pub available_devices: usize,
    pub hardware: HardwareTarget,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelParallelContract {
    pub family: &'static str,
    pub variant: String,
    pub trunk_layers: usize,
    pub mtp_layers: usize,
    pub hidden_size: usize,
    pub vocab_size: usize,
    pub dense_ffn_size: usize,
    pub dense_prefix_layers: usize,
    pub head_dim: usize,
    pub query_heads: Vec<usize>,
    pub kv_heads: Vec<usize>,
    pub expert_count: usize,
    pub experts_per_token: usize,
    pub expert_ffn_size: usize,
    pub shared_expert_ffn_size: usize,
    pub partition_boundaries: Vec<usize>,
    pub hardware_targets: Vec<HardwareTarget>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum StepTpExpertLayout {
    AttentionOnly,
    TensorParallel,
    ExpertParallel,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct StepTpLayerPlan {
    pub layer: usize,
    pub devices: Vec<usize>,
    pub owner_device: usize,
    pub expert_layout: StepTpExpertLayout,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct StepTpPreflightPlan {
    pub layers: Vec<StepTpLayerPlan>,
    pub runtime_groups: Vec<Vec<usize>>,
    pub full_trunk: bool,
}

impl StepTpPreflightPlan {
    pub fn dense_attention_layers(&self) -> usize {
        self.layers
            .iter()
            .filter(|layer| layer.expert_layout == StepTpExpertLayout::AttentionOnly)
            .count()
    }

    pub fn tensor_parallel_expert_layers(&self) -> usize {
        self.layers
            .iter()
            .filter(|layer| layer.expert_layout == StepTpExpertLayout::TensorParallel)
            .count()
    }

    pub fn expert_parallel_layers(&self) -> usize {
        self.layers
            .iter()
            .filter(|layer| layer.expert_layout == StepTpExpertLayout::ExpertParallel)
            .count()
    }
}

impl ModelParallelContract {
    /// Build the model-specific contract. Unregistered families refuse rather than inheriting a
    /// generic transformer assumption.
    pub fn from_model(cfg: &ModelConfig) -> Result<Self, TopologyError> {
        let plan = memra_gguf::model_plan::ModelPlan::compile(cfg).map_err(|error| {
            TopologyError::new(format!("cannot compile parallel ModelPlan: {error}"))
        })?;
        Self::from_plan(cfg, &plan)
    }

    fn from_plan(
        cfg: &ModelConfig,
        plan: &memra_gguf::model_plan::ModelPlan,
    ) -> Result<Self, TopologyError> {
        use memra_gguf::model_plan::{AttentionPlan, MlpPlan};

        if crate::plan_backend::decode_batch_program(plan)
            != crate::plan_backend::DecodeBatchProgram::SlidingGatedMoe
        {
            return Err(TopologyError::new(format!(
                "no parallel contract registered for plan operations {:?}; loading/running does not establish TP/EP support",
                plan.trunk_operations()
            )));
        }
        // TRUNK scope, like every other text-serving surface (worker.rs precedent): the
        // TP/pipeline contract governs the text trunk; a checkpoint that also carries a
        // vision encoder (step37-flash) must not have its TEXT decode blocked by vision
        // operations the pipeline program never runs. Vision serving gates on its own
        // surface (multimodal_prefill_capabilities), not here.
        let pipeline = crate::plan_backend::PIPELINE
            .trunk_capabilities(plan)
            .pipeline;
        if !pipeline.supported {
            return Err(TopologyError::new(format!(
                "pipeline program {} does not implement plan operations {:?}",
                crate::plan_backend::PIPELINE.name,
                pipeline.blockers
            )));
        }
        let trunk_layers = plan.layers.len();
        let mtp_layers = plan.mtp_blocks.len();
        if trunk_layers == 0 {
            return Err(TopologyError::new("parallel contract has no trunk layers"));
        }
        let layers: Vec<_> = plan
            .layers
            .iter()
            .chain(plan.mtp_blocks.iter().map(|block| &block.layer))
            .collect();
        let attention_geometry = layers
            .iter()
            .map(|layer| match &layer.attention {
                AttentionPlan::Full(attention) | AttentionPlan::SlidingWindow { attention, .. } => {
                    Ok((
                        attention.query_heads as usize,
                        attention.kv_heads as usize,
                        attention.key_head_dim as usize,
                    ))
                }
                _ => Err(TopologyError::new(format!(
                    "parallel contract has unsupported attention at layer {}",
                    layer.index
                ))),
            })
            .collect::<Result<Vec<_>, _>>()?;
        let query_heads: Vec<_> = attention_geometry
            .iter()
            .map(|geometry| geometry.0)
            .collect();
        let kv_heads: Vec<_> = attention_geometry
            .iter()
            .map(|geometry| geometry.1)
            .collect();
        let head_dim = attention_geometry[0].2;
        if attention_geometry
            .iter()
            .any(|geometry| geometry.2 != head_dim)
        {
            return Err(TopologyError::new(
                "parallel contract requires one sharding head dimension",
            ));
        }
        let dense_prefix_layers = plan
            .layers
            .iter()
            .take_while(|layer| matches!(layer.mlp, MlpPlan::Dense(_)))
            .count();
        let dense_ffn_size = plan
            .layers
            .iter()
            .find_map(|layer| match &layer.mlp {
                MlpPlan::Dense(dense) => Some(dense.intermediate_size as usize),
                _ => None,
            })
            .ok_or_else(|| TopologyError::new("parallel contract requires a dense prefix"))?;
        let moe = layers
            .iter()
            .find_map(|layer| match &layer.mlp {
                MlpPlan::Moe(moe) => Some(moe),
                _ => None,
            })
            .ok_or_else(|| TopologyError::new("parallel contract requires routed experts"))?;
        let shared_expert_ffn_size = moe
            .shared
            .as_ref()
            .map_or(0, |shared| shared.intermediate_size as usize);
        let is_step37 = trunk_layers == STEP37_TRUNK_LAYERS
            && mtp_layers == 3
            && plan.hidden_size == 4096
            && dense_ffn_size == 11_264
            && plan.vocab_size == 128_896
            && query_heads
                .iter()
                .enumerate()
                .all(|(il, &heads)| heads == if il % 4 == 0 { 64 } else { 96 })
            && kv_heads.iter().all(|&heads| heads == 8)
            && moe.expert_count == 288
            && moe.experts_per_token == 8
            && moe.expert_intermediate_size == 1280
            && shared_expert_ffn_size == 1280
            && dense_prefix_layers == 3;
        if !is_step37 {
            return Err(TopologyError::new(format!(
                "no qualified parallel contract for variant {:?}: only the exact Step-3.7-Flash geometry is registered; derived trunk={trunk_layers} mtp={mtp_layers} hidden={} vocab={} dense_ff={dense_ffn_size} dense_prefix={dense_prefix_layers} head_dim={head_dim} q_heads={query_heads:?} kv_heads={kv_heads:?} experts={}/{}/{} shared={shared_expert_ffn_size}",
                cfg.name,
                plan.hidden_size,
                plan.vocab_size,
                moe.expert_count,
                moe.experts_per_token,
                moe.expert_intermediate_size,
            )));
        }

        Ok(Self {
            family: "sliding-gated-moe",
            variant: "Step-3.7-Flash-FP8".to_string(),
            trunk_layers,
            mtp_layers,
            hidden_size: cfg.n_embd as usize,
            vocab_size: cfg.n_vocab as usize,
            dense_ffn_size,
            dense_prefix_layers,
            head_dim,
            query_heads,
            kv_heads,
            expert_count: moe.expert_count as usize,
            experts_per_token: moe.experts_per_token as usize,
            expert_ffn_size: moe.expert_intermediate_size as usize,
            shared_expert_ffn_size,
            partition_boundaries: plan.partition_boundaries.clone(),
            hardware_targets: vec![HardwareTarget::RtxPro6000Blackwell],
        })
    }

    pub fn plan(&self, request: TopologyRequest) -> Result<ParallelPlan, TopologyError> {
        let pp = request.pipeline;
        let tp = request.tensor;
        if !(1..=PRODUCT_MAX_CARDS).contains(&pp) {
            return Err(TopologyError::new(format!(
                "PP size {pp} outside product range 1..={PRODUCT_MAX_CARDS}"
            )));
        }
        if !(1..=PRODUCT_MAX_CARDS).contains(&tp) {
            return Err(TopologyError::new(format!(
                "TP size {tp} outside product range 1..={PRODUCT_MAX_CARDS}"
            )));
        }
        let world = request.world_size()?;
        if world > PRODUCT_MAX_CARDS {
            return Err(TopologyError::new(format!(
                "PP={pp} x TP={tp} requires {world} cards; product envelope is \
                 {PRODUCT_MAX_CARDS}"
            )));
        }
        if !self.hardware_targets.contains(&request.hardware) {
            return Err(TopologyError::new(format!(
                "{} has no qualified {} contract",
                self.variant,
                request.hardware.label()
            )));
        }
        if world > request.hardware.max_cards() {
            return Err(TopologyError::new(format!(
                "{} target permits at most {} card(s), requested {world}",
                request.hardware.label(),
                request.hardware.max_cards()
            )));
        }
        if request.available_devices < world {
            return Err(TopologyError::new(format!(
                "PP={pp} x TP={tp} requires {world} cards, only {} available",
                request.available_devices
            )));
        }
        if pp > self.trunk_layers {
            return Err(TopologyError::new(format!(
                "PP={pp} exceeds {} trunk layers",
                self.trunk_layers
            )));
        }
        if request.expert_parallel && tp == 1 {
            return Err(TopologyError::new(
                "expert parallelism requires TP group size greater than one",
            ));
        }

        // Check the family-specific, per-layer attention geometry before generic dimensions so a
        // refused topology names the model program that actually makes it invalid.
        for (il, (&q, &kv)) in self.query_heads.iter().zip(&self.kv_heads).enumerate() {
            require_divisible(&format!("layer {il} query heads"), q, tp)?;
            require_divisible(&format!("layer {il} KV heads"), kv, tp)?;
        }
        require_divisible("hidden size", self.hidden_size, tp)?;
        require_divisible("vocabulary size", self.vocab_size, tp)?;
        require_fp8_block_shard("dense FFN size", self.dense_ffn_size, tp)?;
        if request.expert_parallel {
            require_divisible("routed expert count", self.expert_count, tp)?;
        } else {
            require_fp8_block_shard("routed expert FFN size", self.expert_ffn_size, tp)?;
        }

        let stage_ranges = (0..pp)
            .map(|stage| stage * self.trunk_layers / pp..(stage + 1) * self.trunk_layers / pp)
            .collect();

        Ok(ParallelPlan {
            contract: self.clone(),
            request,
            world_size: world,
            stage_ranges,
            mtp_owner_stage: self.mtp_layers.gt(&0).then_some(pp - 1),
            // Step's 1280-wide shared expert cannot be split four or eight ways without cutting
            // through checkpoint 128-row E4M3 scale blocks. Replication is the exact program for
            // those TP/EP layouts; only routed experts are distributed.
            shared_expert_replicated: tp > 1 && self.shared_expert_ffn_size > 0,
        })
    }

    /// Validate every selected Step TP layer before the loader opens its first weight tensor.
    ///
    /// Physical device availability is checked separately because this pure plan is also the
    /// topology oracle for tests and offline launch preparation.
    pub(crate) fn preflight_step_tp_specs<'a>(
        &self,
        specs: impl IntoIterator<Item = (usize, &'a [usize])>,
        layer_owners: &[usize],
    ) -> Result<StepTpPreflightPlan, TopologyError> {
        if layer_owners.len() != self.trunk_layers {
            return Err(TopologyError::new(format!(
                "Step TP owner map has {} layers, expected {}",
                layer_owners.len(),
                self.trunk_layers
            )));
        }

        let mut seen = vec![false; self.trunk_layers];
        let mut layers = Vec::new();
        let mut runtime_groups: Vec<Vec<usize>> = Vec::new();
        for (layer, devices) in specs {
            if layer >= self.trunk_layers {
                return Err(TopologyError::new(format!(
                    "Step TP layer {layer} is outside trunk layers 0..{}",
                    self.trunk_layers
                )));
            }
            if seen[layer] {
                return Err(TopologyError::new(format!(
                    "Step TP preflight assigns layer {layer} more than once"
                )));
            }
            if !(2..=PRODUCT_MAX_CARDS).contains(&devices.len()) {
                return Err(TopologyError::new(format!(
                    "Step TP layer {layer} requires 2..={PRODUCT_MAX_CARDS} devices, got {}",
                    devices.len()
                )));
            }
            let mut unique = devices.to_vec();
            unique.sort_unstable();
            unique.dedup();
            if unique.len() != devices.len() {
                return Err(TopologyError::new(format!(
                    "Step TP layer {layer} devices must be distinct, got {devices:?}"
                )));
            }
            let owner_device = layer_owners[layer];
            if devices.first().copied() != Some(owner_device) {
                return Err(TopologyError::new(format!(
                    "Step TP layer {layer} owning PP device {owner_device} must be the first rank, \
                     got {devices:?}"
                )));
            }

            let expert_layout = if layer < self.dense_prefix_layers {
                StepTpExpertLayout::AttentionOnly
            } else if devices.len() > 2 {
                StepTpExpertLayout::ExpertParallel
            } else {
                StepTpExpertLayout::TensorParallel
            };
            let plan = self.plan(TopologyRequest {
                pipeline: 1,
                tensor: devices.len(),
                // Dense-prefix layers have no routed expert bank, but the full-model TP4/TP8
                // contract still uses the EP geometry arm so the irrelevant 1280-wide expert
                // projection is not falsely tensor-sharded during topology validation.
                expert_parallel: devices.len() > 2,
                available_devices: devices.len(),
                hardware: HardwareTarget::RtxPro6000Blackwell,
            })?;
            for rank in 0..devices.len() {
                let query = plan.query_head_range(layer, rank).ok_or_else(|| {
                    TopologyError::new(format!(
                        "Step TP layer {layer} has no query-head range for rank {rank}"
                    ))
                })?;
                let kv = plan.kv_head_range(layer, rank).ok_or_else(|| {
                    TopologyError::new(format!(
                        "Step TP layer {layer} has no KV-head range for rank {rank}"
                    ))
                })?;
                if query.is_empty() || kv.is_empty() {
                    return Err(TopologyError::new(format!(
                        "Step TP layer {layer} rank {rank} has an empty attention shard"
                    )));
                }
            }

            if !runtime_groups.iter().any(|group| group == devices) {
                runtime_groups.push(devices.to_vec());
            }
            seen[layer] = true;
            layers.push(StepTpLayerPlan {
                layer,
                devices: devices.to_vec(),
                owner_device,
                expert_layout,
            });
        }
        layers.sort_unstable_by_key(|layer| layer.layer);

        Ok(StepTpPreflightPlan {
            layers,
            runtime_groups,
            full_trunk: seen.into_iter().all(|selected| selected),
        })
    }

    /// Plan an explicit sequence of PP stages whose TP/EP widths may differ.
    ///
    /// This is the placement contract for arbitrary 1-8 card counts. It validates only the layers
    /// assigned to each group, so a TP1 dense-prefix stage can coexist with a TP2/TP4/TP8 MoE
    /// stage. Logical ranks are contiguous per stage; physical-device binding is a separate
    /// runtime concern and must preserve each group's rank order.
    pub fn plan_grouped(
        &self,
        request: GroupedTopologyRequest,
    ) -> Result<GroupedParallelPlan, TopologyError> {
        if request.stages.is_empty() {
            return Err(TopologyError::new(
                "grouped Step topology requires at least one stage",
            ));
        }
        if request.stages.len() > self.trunk_layers {
            return Err(TopologyError::new(format!(
                "{} grouped stages exceed {} trunk layers",
                request.stages.len(),
                self.trunk_layers
            )));
        }
        if !self.hardware_targets.contains(&request.hardware) {
            return Err(TopologyError::new(format!(
                "{} has no qualified {} contract",
                self.variant,
                request.hardware.label()
            )));
        }

        let mut world_size = 0usize;
        let mut expected_layer = 0usize;
        let mut rank_groups = Vec::with_capacity(request.stages.len());
        for (stage, group) in request.stages.iter().enumerate() {
            if group.layers.start != expected_layer
                || group.layers.start >= group.layers.end
                || group.layers.end > self.trunk_layers
            {
                return Err(TopologyError::new(format!(
                    "grouped stage {stage} layers {:?} do not continue the exact 0..{} trunk \
                     partition at layer {expected_layer}",
                    group.layers, self.trunk_layers
                )));
            }
            if !(1..=PRODUCT_MAX_CARDS).contains(&group.tensor) {
                return Err(TopologyError::new(format!(
                    "grouped stage {stage} TP={} outside product range 1..={PRODUCT_MAX_CARDS}",
                    group.tensor
                )));
            }
            if group.expert_parallel && group.tensor == 1 {
                return Err(TopologyError::new(format!(
                    "grouped stage {stage} expert parallelism requires more than one rank"
                )));
            }

            validate_group_geometry(self, stage, group)?;
            let rank_start = world_size;
            world_size = world_size
                .checked_add(group.tensor)
                .ok_or_else(|| TopologyError::new("grouped topology world size overflow"))?;
            rank_groups.push(StageRankGroup {
                stage,
                layers: group.layers.clone(),
                global_ranks: rank_start..world_size,
                tensor: group.tensor,
                expert_parallel: group.expert_parallel,
                shared_expert_replicated: group.tensor > 1 && self.shared_expert_ffn_size > 0,
            });
            expected_layer = group.layers.end;
        }
        if expected_layer != self.trunk_layers {
            return Err(TopologyError::new(format!(
                "grouped Step topology ends at layer {expected_layer}, expected {}",
                self.trunk_layers
            )));
        }
        if world_size > PRODUCT_MAX_CARDS {
            return Err(TopologyError::new(format!(
                "grouped Step topology requires {world_size} cards; product envelope is \
                 {PRODUCT_MAX_CARDS}"
            )));
        }
        if world_size > request.hardware.max_cards() {
            return Err(TopologyError::new(format!(
                "{} target permits at most {} card(s), requested {world_size}",
                request.hardware.label(),
                request.hardware.max_cards()
            )));
        }
        if request.available_devices < world_size {
            return Err(TopologyError::new(format!(
                "grouped Step topology requires {world_size} cards, only {} available",
                request.available_devices
            )));
        }

        let mtp_owner_stage = self.mtp_layers.gt(&0).then_some(expected_layer_stage(
            self.trunk_layers - 1,
            &request.stages,
        )?);
        Ok(GroupedParallelPlan {
            contract: self.clone(),
            request,
            world_size,
            rank_groups,
            mtp_owner_stage,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParallelPlan {
    pub contract: ModelParallelContract,
    pub request: TopologyRequest,
    pub world_size: usize,
    pub stage_ranges: Vec<Range<usize>>,
    /// MTP layers are not pipeline stages of their own; the final PP stage owns them.
    pub mtp_owner_stage: Option<usize>,
    pub shared_expert_replicated: bool,
}

impl ParallelPlan {
    pub fn global_rank(&self, pipeline_rank: usize, tensor_rank: usize) -> Option<usize> {
        if pipeline_rank >= self.request.pipeline || tensor_rank >= self.request.tensor {
            return None;
        }
        Some(pipeline_rank * self.request.tensor + tensor_rank)
    }

    pub fn query_head_range(&self, layer: usize, tensor_rank: usize) -> Option<Range<usize>> {
        split_range(
            *self.contract.query_heads.get(layer)?,
            self.request.tensor,
            tensor_rank,
        )
    }

    pub fn kv_head_range(&self, layer: usize, tensor_rank: usize) -> Option<Range<usize>> {
        split_range(
            *self.contract.kv_heads.get(layer)?,
            self.request.tensor,
            tensor_rank,
        )
    }

    /// Column-parallel Q output range and the matching row-parallel O input range.
    pub fn query_feature_range(&self, layer: usize, tensor_rank: usize) -> Option<Range<usize>> {
        let heads = self.query_head_range(layer, tensor_rank)?;
        Some(heads.start * self.contract.head_dim..heads.end * self.contract.head_dim)
    }

    /// Column-parallel K/V output range. Step-3.7 has eight KV heads, so TP2 and TP4 partition
    /// them exactly; no KV-head replication is part of this registered contract.
    pub fn kv_feature_range(&self, layer: usize, tensor_rank: usize) -> Option<Range<usize>> {
        let heads = self.kv_head_range(layer, tensor_rank)?;
        Some(heads.start * self.contract.head_dim..heads.end * self.contract.head_dim)
    }

    /// Column-parallel dense gate/up output range and matching row-parallel down input range.
    pub fn dense_ffn_range(&self, tensor_rank: usize) -> Option<Range<usize>> {
        split_range(
            self.contract.dense_ffn_size,
            self.request.tensor,
            tensor_rank,
        )
    }

    pub fn routed_expert_range(&self, tensor_rank: usize) -> Option<Range<usize>> {
        self.request
            .expert_parallel
            .then(|| split_range(self.contract.expert_count, self.request.tensor, tensor_rank))?
    }

    pub fn routed_expert_ffn_range(&self, tensor_rank: usize) -> Option<Range<usize>> {
        (!self.request.expert_parallel).then(|| {
            split_range(
                self.contract.expert_ffn_size,
                self.request.tensor,
                tensor_rank,
            )
        })?
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StageRankGroup {
    pub stage: usize,
    pub layers: Range<usize>,
    pub global_ranks: Range<usize>,
    pub tensor: usize,
    pub expert_parallel: bool,
    pub shared_expert_replicated: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GroupedParallelPlan {
    pub contract: ModelParallelContract,
    pub request: GroupedTopologyRequest,
    pub world_size: usize,
    pub rank_groups: Vec<StageRankGroup>,
    pub mtp_owner_stage: Option<usize>,
}

impl GroupedParallelPlan {
    pub fn group_for_layer(&self, layer: usize) -> Option<&StageRankGroup> {
        self.rank_groups
            .iter()
            .find(|group| group.layers.contains(&layer))
    }

    pub fn group_for_global_rank(&self, rank: usize) -> Option<&StageRankGroup> {
        self.rank_groups
            .iter()
            .find(|group| group.global_ranks.contains(&rank))
    }

    pub fn global_rank(&self, stage: usize, tensor_rank: usize) -> Option<usize> {
        let group = self.rank_groups.get(stage)?;
        (tensor_rank < group.tensor).then_some(group.global_ranks.start + tensor_rank)
    }

    pub fn query_head_range(&self, layer: usize, tensor_rank: usize) -> Option<Range<usize>> {
        let group = self.group_for_layer(layer)?;
        split_range(
            *self.contract.query_heads.get(layer)?,
            group.tensor,
            tensor_rank,
        )
    }

    pub fn kv_head_range(&self, layer: usize, tensor_rank: usize) -> Option<Range<usize>> {
        let group = self.group_for_layer(layer)?;
        split_range(
            *self.contract.kv_heads.get(layer)?,
            group.tensor,
            tensor_rank,
        )
    }

    pub fn routed_expert_range(&self, layer: usize, tensor_rank: usize) -> Option<Range<usize>> {
        let group = self.group_for_layer(layer)?;
        group
            .expert_parallel
            .then(|| split_range(self.contract.expert_count, group.tensor, tensor_rank))?
    }
}

/// Validate the live Step PP request before the loader allocates CUDA state. Checkpoint tensor
/// census is deliberately a separate loader gate: topology legality must remain testable without
/// opening model files, while serving requires both gates.
pub fn validate_step_pp_request(cfg: &ModelConfig) -> Result<Option<ParallelPlan>, TopologyError> {
    let pp = match std::env::var("MEMRA_PP_STAGES") {
        Err(_) => return Ok(None),
        Ok(value) if value.is_empty() || value == "0" || value == "1" => return Ok(None),
        Ok(value) => value.parse::<usize>().map_err(|_| {
            TopologyError::new(format!("MEMRA_PP_STAGES={value} is not a positive integer"))
        })?,
    };
    let devices = selected_pp_devices(pp)?;
    let hardware = detect_uniform_hardware(&devices)?;
    let contract = ModelParallelContract::from_model(cfg)?;
    let trunk_layers = contract.trunk_layers;
    let plan = contract.plan(TopologyRequest {
        pipeline: pp,
        tensor: 1,
        expert_parallel: false,
        available_devices: devices.len(),
        hardware,
    })?;
    let fence = crate::pp::pp_cuts(trunk_layers).ok_or_else(|| {
        TopologyError::new(format!(
            "Step PP={pp} has no valid runtime stage fence over {trunk_layers} trunk layers"
        ))
    })?;
    let plan = apply_stage_fence(plan, &fence)?;
    Ok(Some(plan))
}

/// Prove that the official Step checkpoint exposes every routed expert projection as a native
/// stacked block-128 E4M3 bank. Converted and per-tensor artifacts do not inherit this contract.
pub fn validate_step_fp8_checkpoint(
    src: &dyn TensorSource,
    contract: &ModelParallelContract,
) -> Result<usize, TopologyError> {
    if src.st_dir().is_none() {
        return Err(TopologyError::new(
            "Step-3.7-Flash-FP8 topology qualification requires the official safetensors \
             checkpoint source; a converted artifact cannot inherit this contract",
        ));
    }

    let projections = [
        (
            "ffn_gate_exps",
            contract.hidden_size,
            contract.expert_ffn_size,
        ),
        (
            "ffn_up_exps",
            contract.hidden_size,
            contract.expert_ffn_size,
        ),
        (
            "ffn_down_exps",
            contract.expert_ffn_size,
            contract.hidden_size,
        ),
    ];
    let mut qualified = 0usize;
    for layer in contract.dense_prefix_layers..contract.trunk_layers {
        for &(projection, expected_in, expected_out) in &projections {
            let name = format!("blk.{layer}.{projection}.weight");
            let fp8 = src.find_fp8_stacked_native(&name).ok_or_else(|| {
                TopologyError::new(format!(
                    "{name} is not a checkpoint-faithful stacked block-128 E4M3 bank"
                ))
            })?;
            if fp8.n_expert != contract.expert_count {
                return Err(TopologyError::new(format!(
                    "{name} carries {} experts, expected {}",
                    fp8.n_expert, contract.expert_count
                )));
            }
            if fp8.in_f != expected_in || fp8.out_f != expected_out {
                return Err(TopologyError::new(format!(
                    "{name} expert shape {}x{} != expected {expected_out}x{expected_in}",
                    fp8.out_f, fp8.in_f
                )));
            }
            let expected_rows = expected_out.div_ceil(STEP_FP8_BLOCK);
            let expected_cols = expected_in.div_ceil(STEP_FP8_BLOCK);
            let expected_scales = contract.expert_count * expected_rows * expected_cols;
            if fp8.scale_rows != expected_rows
                || fp8.scale_cols != expected_cols
                || fp8.scales.len() != expected_scales
            {
                return Err(TopologyError::new(format!(
                    "{name} block-128 E4M3 grid {}x{} ({} scales) != expected {} experts x \
                     {expected_rows}x{expected_cols} ({expected_scales} scales)",
                    fp8.scale_rows,
                    fp8.scale_cols,
                    fp8.scales.len(),
                    contract.expert_count
                )));
            }
            qualified += fp8.n_expert;
        }
    }

    let expected = (contract.trunk_layers - contract.dense_prefix_layers)
        * contract.expert_count
        * projections.len();
    if qualified != expected {
        return Err(TopologyError::new(format!(
            "Step E4M3 tensor census qualified {qualified}, expected {expected}"
        )));
    }
    Ok(qualified)
}

/// Prove that the official Step NVFP4 checkpoint exposes every routed expert projection as a
/// native stacked modelopt NVFP4 bank: packed e2m1 codes `[E, out, in/2]`, per-16 UE4M3 scales
/// `[E, out, in/16]`, and a finite positive per-expert `weight_scale_2` macro. Converted and
/// per-tensor artifacts do not inherit this contract. The macro census matters: those values run
/// ~1e-5..1e-4 in the official artifact and dropping them silently produces garbage, so a bank
/// whose macros fail the finite-positive check refuses here rather than at first decode.
pub fn validate_step_nvfp4_checkpoint(
    src: &dyn TensorSource,
    contract: &ModelParallelContract,
) -> Result<usize, TopologyError> {
    if src.st_dir().is_none() {
        return Err(TopologyError::new(
            "Step-3.7-Flash-NVFP4 topology qualification requires the official safetensors \
             checkpoint source; a converted artifact cannot inherit this contract",
        ));
    }

    let projections = [
        (
            "ffn_gate_exps",
            contract.hidden_size,
            contract.expert_ffn_size,
        ),
        (
            "ffn_up_exps",
            contract.hidden_size,
            contract.expert_ffn_size,
        ),
        (
            "ffn_down_exps",
            contract.expert_ffn_size,
            contract.hidden_size,
        ),
    ];
    let mut qualified = 0usize;
    for layer in contract.dense_prefix_layers..contract.trunk_layers {
        for &(projection, expected_in, expected_out) in &projections {
            let name = format!("blk.{layer}.{projection}.weight");
            let bank = src.find_nvfp4_stacked_native(&name).ok_or_else(|| {
                TopologyError::new(format!(
                    "{name} is not a checkpoint-faithful stacked modelopt NVFP4 bank \
                     (packed e2m1 codes + per-16 UE4M3 scales + per-expert macro)"
                ))
            })?;
            if bank.n_expert != contract.expert_count {
                return Err(TopologyError::new(format!(
                    "{name} carries {} experts, expected {}",
                    bank.n_expert, contract.expert_count
                )));
            }
            if bank.in_f != expected_in || bank.out_f != expected_out {
                return Err(TopologyError::new(format!(
                    "{name} expert shape {}x{} != expected {expected_out}x{expected_in}",
                    bank.out_f, bank.in_f
                )));
            }
            if bank.in_f % 64 != 0 {
                return Err(TopologyError::new(format!(
                    "{name} in_features {} is not 64-aligned; memra block_nvfp4 kernels \
                     require whole 64-element superblocks",
                    bank.in_f
                )));
            }
            if bank.macros.len() != contract.expert_count {
                return Err(TopologyError::new(format!(
                    "{name} carries {} weight_scale_2 macros, expected {}",
                    bank.macros.len(),
                    contract.expert_count
                )));
            }
            qualified += bank.n_expert;
        }
    }

    let expected = (contract.trunk_layers - contract.dense_prefix_layers)
        * contract.expert_count
        * projections.len();
    if qualified != expected {
        return Err(TopologyError::new(format!(
            "Step NVFP4 tensor census qualified {qualified}, expected {expected}"
        )));
    }
    Ok(qualified)
}

fn apply_stage_fence(
    mut plan: ParallelPlan,
    fence: &[usize],
) -> Result<ParallelPlan, TopologyError> {
    let expected = plan.request.pipeline + 1;
    if fence.len() != expected
        || fence.first() != Some(&0)
        || fence.last() != Some(&plan.contract.trunk_layers)
        || fence.windows(2).any(|window| window[0] >= window[1])
        || fence[1..fence.len() - 1]
            .iter()
            .any(|boundary| !plan.contract.partition_boundaries.contains(boundary))
    {
        return Err(TopologyError::new(format!(
            "invalid PP fence {fence:?} for {} stages over {} trunk layers",
            plan.request.pipeline, plan.contract.trunk_layers
        )));
    }
    plan.stage_ranges = fence
        .windows(2)
        .map(|window| window[0]..window[1])
        .collect();
    Ok(plan)
}

fn selected_pp_devices(pp: usize) -> Result<Vec<usize>, TopologyError> {
    let raw = std::env::var("MEMRA_PP_DEVICES").map_err(|_| {
        TopologyError::new(format!(
            "Step PP={pp} requires explicit MEMRA_PP_DEVICES with one distinct CUDA ordinal per \
             stage; same-device diagnostics do not qualify the multi-card product"
        ))
    })?;
    let devices: Result<Vec<usize>, _> = raw
        .split(',')
        .map(|part| part.trim().parse::<usize>())
        .collect();
    let devices = devices.map_err(|_| {
        TopologyError::new(format!(
            "MEMRA_PP_DEVICES={raw:?} is not a comma-separated CUDA ordinal list"
        ))
    })?;
    if devices.len() != pp {
        return Err(TopologyError::new(format!(
            "MEMRA_PP_DEVICES lists {} devices but MEMRA_PP_STAGES={pp}",
            devices.len()
        )));
    }
    let mut unique = devices.clone();
    unique.sort_unstable();
    unique.dedup();
    if unique.len() != devices.len() {
        return Err(TopologyError::new(format!(
            "Step PP={pp} requires {pp} distinct devices; MEMRA_PP_DEVICES={raw:?} repeats an \
             ordinal"
        )));
    }
    Ok(devices)
}

pub(crate) fn detect_uniform_hardware(devices: &[usize]) -> Result<HardwareTarget, TopologyError> {
    cudarc::driver::result::init().map_err(|error| {
        TopologyError::new(format!("CUDA driver initialization failed: {error}"))
    })?;
    let mut target = None;
    for &ordinal in devices {
        let device = cudarc::driver::result::device::get(ordinal as i32).map_err(|error| {
            TopologyError::new(format!("CUDA device {ordinal} lookup failed: {error}"))
        })?;
        let name = cudarc::driver::result::device::get_name(device).map_err(|error| {
            TopologyError::new(format!("CUDA device {ordinal} name lookup failed: {error}"))
        })?;
        let current = HardwareTarget::from_device_name(&name)?;
        if let Some(expected) = target {
            if current != expected {
                return Err(TopologyError::new(format!(
                    "mixed hardware targets in MEMRA_PP_DEVICES: expected {}, device {ordinal} is \
                     {}",
                    expected.label(),
                    current.label()
                )));
            }
        } else {
            target = Some(current);
        }
    }
    target.ok_or_else(|| TopologyError::new("MEMRA_PP_DEVICES is empty"))
}

fn require_divisible(label: &str, value: usize, parts: usize) -> Result<(), TopologyError> {
    if value == 0 {
        return Err(TopologyError::new(format!("{label} is zero")));
    }
    if value % parts != 0 {
        return Err(TopologyError::new(format!(
            "{label} {value} is not divisible by TP={parts}"
        )));
    }
    Ok(())
}

fn require_fp8_block_shard(label: &str, value: usize, parts: usize) -> Result<(), TopologyError> {
    require_divisible(label, value, parts)?;
    let local = value / parts;
    if local % STEP_FP8_BLOCK != 0 {
        return Err(TopologyError::new(format!(
            "{label} shard {local} for TP={parts} cuts through the Step E4M3 block size \
             {STEP_FP8_BLOCK}"
        )));
    }
    Ok(())
}

fn validate_group_geometry(
    contract: &ModelParallelContract,
    stage: usize,
    group: &StageGroupRequest,
) -> Result<(), TopologyError> {
    let tp = group.tensor;
    for layer in group.layers.clone() {
        require_divisible(
            &format!("stage {stage} layer {layer} query heads"),
            contract.query_heads[layer],
            tp,
        )?;
        require_divisible(
            &format!("stage {stage} layer {layer} KV heads"),
            contract.kv_heads[layer],
            tp,
        )?;
    }
    require_divisible(
        &format!("stage {stage} hidden size"),
        contract.hidden_size,
        tp,
    )?;
    if group.layers.start < contract.dense_prefix_layers {
        require_fp8_block_shard(
            &format!("stage {stage} dense FFN size"),
            contract.dense_ffn_size,
            tp,
        )?;
    }
    if group.layers.end > contract.dense_prefix_layers {
        if group.expert_parallel {
            require_divisible(
                &format!("stage {stage} routed expert count"),
                contract.expert_count,
                tp,
            )?;
        } else {
            require_fp8_block_shard(
                &format!("stage {stage} routed expert FFN size"),
                contract.expert_ffn_size,
                tp,
            )?;
        }
    }
    if group.layers.end == contract.trunk_layers {
        require_divisible(
            &format!("stage {stage} vocabulary size"),
            contract.vocab_size,
            tp,
        )?;
    }
    Ok(())
}

fn expected_layer_stage(
    layer: usize,
    stages: &[StageGroupRequest],
) -> Result<usize, TopologyError> {
    stages
        .iter()
        .position(|stage| stage.layers.contains(&layer))
        .ok_or_else(|| TopologyError::new(format!("no grouped stage owns layer {layer}")))
}

fn split_range(total: usize, parts: usize, rank: usize) -> Option<Range<usize>> {
    if parts == 0 || rank >= parts || total % parts != 0 {
        return None;
    }
    let width = total / parts;
    Some(rank * width..(rank + 1) * width)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TopologyError {
    message: String,
}

impl TopologyError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for TopologyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.message.fmt(f)
    }
}

impl std::error::Error for TopologyError {}

#[cfg(test)]
mod tests {
    use super::*;
    use memra_gguf::config::{Arch, MoeConfig, Step35Config};
    use memra_gguf::source::{Fp8StackedNative, TensorView};
    use std::path::Path;

    fn step37_contract() -> ModelParallelContract {
        let total_layers = 48;
        ModelParallelContract {
            family: "sliding-gated-moe",
            variant: "Step-3.7-Flash-FP8".to_string(),
            trunk_layers: 45,
            mtp_layers: 3,
            hidden_size: 4096,
            vocab_size: 128_896,
            dense_ffn_size: 11_264,
            dense_prefix_layers: 3,
            head_dim: 128,
            query_heads: (0..total_layers)
                .map(|il| if il % 4 == 0 { 64 } else { 96 })
                .collect(),
            kv_heads: vec![8; total_layers],
            expert_count: 288,
            experts_per_token: 8,
            expert_ffn_size: 1280,
            shared_expert_ffn_size: 1280,
            partition_boundaries: (1..45).collect(),
            hardware_targets: vec![HardwareTarget::RtxPro6000Blackwell],
        }
    }

    fn step37_model_config() -> ModelConfig {
        let total_layers = 48;
        let head_count: Vec<u32> = (0..total_layers)
            .map(|il| if il % 4 == 0 { 64 } else { 96 })
            .collect();
        ModelConfig {
            arch: Arch::Step35,
            name: "Step-3.7-Flash-FP8".to_string(),
            n_layer: total_layers,
            n_embd: 4096,
            n_head: 96,
            n_head_kv: 8,
            head_dim_k: 128,
            head_dim_v: 128,
            n_ff: 11_264,
            n_vocab: 128_896,
            context_length: 262_144,
            rms_eps: 1e-6,
            rope_freq_base: 5_000_000.0,
            rope_dim_count: 128,
            rope_sections: Vec::new(),
            full_attention_interval: 0,
            ssm: None,
            moe: Some(MoeConfig {
                expert_count: 288,
                expert_used_count: 8,
                expert_ff_length: 1280,
                expert_shared_ff_length: 1280,
            }),
            m3: None,
            hy3: None,
            gemma4: None,
            vision: None,
            multimodal: None,
            mla: None,
            dsv4: None,
            step35: Some(Step35Config {
                head_count,
                head_count_kv: vec![8; total_layers as usize],
                swa_pattern: (0..total_layers).map(|il| il % 4 != 0).collect(),
                sliding_window: 512,
                rope_base_global: 5_000_000.0,
                rope_base_swa: 10_000.0,
                rope_dims_full: 64,
                rope_dims_swa: 128,
                rope_freq_factors: None,
                swiglu_clamp_exp: vec![0.0; total_layers as usize],
                swiglu_clamp_shexp: vec![0.0; total_layers as usize],
                sigmoid_routing: true,
                routed_scaling_factor: 3.0,
                route_norm: true,
                first_k_dense_replace: 3,
            }),
            geometry: None,
            nextn_predict_layers: 3,
            n_layer_total: total_layers,
        }
    }

    fn request(pp: usize, tp: usize, expert_parallel: bool) -> TopologyRequest {
        TopologyRequest {
            pipeline: pp,
            tensor: tp,
            expert_parallel,
            available_devices: pp * tp,
            hardware: HardwareTarget::RtxPro6000Blackwell,
        }
    }

    struct MockStepFp8Source {
        safetensors: bool,
        block_scales: bool,
    }

    impl TensorSource for MockStepFp8Source {
        fn config(&self) -> ModelConfig {
            step37_model_config()
        }

        fn find(&self, _ggml_name: &str) -> Option<TensorView<'_>> {
            None
        }

        fn st_dir(&self) -> Option<&Path> {
            self.safetensors.then(|| Path::new("/mock-step-fp8"))
        }

        fn find_fp8_stacked_native(&self, name: &str) -> Option<Fp8StackedNative<'_>> {
            let (in_f, out_f): (usize, usize) = if name.contains("ffn_down_exps") {
                (1280, 4096)
            } else if name.contains("ffn_gate_exps") || name.contains("ffn_up_exps") {
                (4096, 1280)
            } else {
                return None;
            };
            let (scale_rows, scale_cols) = if self.block_scales {
                (
                    out_f.div_ceil(STEP_FP8_BLOCK),
                    in_f.div_ceil(STEP_FP8_BLOCK),
                )
            } else {
                (1, 1)
            };
            Some(Fp8StackedNative {
                bytes: &[],
                scales: vec![1.0; 288 * scale_rows * scale_cols],
                n_expert: 288,
                out_f,
                in_f,
                scale_rows,
                scale_cols,
            })
        }
    }

    #[test]
    fn step_fp8_checkpoint_census_covers_every_routed_projection() {
        let source = MockStepFp8Source {
            safetensors: true,
            block_scales: true,
        };
        let qualified =
            validate_step_fp8_checkpoint(&source, &step37_contract()).expect("valid FP8 source");
        assert_eq!(qualified, 42 * 288 * 3);
    }

    #[test]
    fn step_fp8_checkpoint_census_refuses_conversion_and_wrong_scale_class() {
        let converted = MockStepFp8Source {
            safetensors: false,
            block_scales: true,
        };
        assert!(
            validate_step_fp8_checkpoint(&converted, &step37_contract())
                .unwrap_err()
                .to_string()
                .contains("official safetensors")
        );

        let per_tensor = MockStepFp8Source {
            safetensors: true,
            block_scales: false,
        };
        assert!(
            validate_step_fp8_checkpoint(&per_tensor, &step37_contract())
                .unwrap_err()
                .to_string()
                .contains("block-128 E4M3")
        );
    }

    #[test]
    fn step_pp3_maps_fifteen_trunk_layers_per_card() {
        let plan = step37_contract().plan(request(3, 1, false)).unwrap();
        assert_eq!(plan.world_size, 3);
        assert_eq!(plan.stage_ranges, vec![0..15, 15..30, 30..45]);
        assert_eq!(plan.mtp_owner_stage, Some(2));
    }

    #[test]
    fn step_pp_marker_uses_the_runtime_stage_fence() {
        let plan = step37_contract().plan(request(3, 1, false)).unwrap();
        let plan = apply_stage_fence(plan, &[0, 10, 28, 45]).unwrap();
        assert_eq!(plan.stage_ranges, vec![0..10, 10..28, 28..45]);
    }

    #[test]
    fn stage_fence_must_use_model_plan_partition_boundaries() {
        let mut contract = step37_contract();
        contract
            .partition_boundaries
            .retain(|&boundary| boundary != 10);
        let plan = contract.plan(request(3, 1, false)).unwrap();
        let error = apply_stage_fence(plan, &[0, 10, 28, 45]).unwrap_err();
        assert!(error.to_string().contains("invalid PP fence"));
    }

    #[test]
    fn step_contract_is_extracted_from_model_specific_geometry() {
        let contract = ModelParallelContract::from_model(&step37_model_config()).unwrap();
        assert_eq!(contract.family, "sliding-gated-moe");
        assert_eq!(contract.trunk_layers, 45);
        assert_eq!(contract.mtp_layers, 3);
        assert_eq!(contract.query_heads[0], 64);
        assert_eq!(contract.query_heads[1], 96);
        assert_eq!(contract.kv_heads[47], 8);
        assert_eq!(contract.expert_count, 288);
        assert_eq!(contract.experts_per_token, 8);
    }

    #[test]
    fn step_sibling_does_not_inherit_the_step37_contract() {
        let mut sibling = step37_model_config();
        sibling.name = "Step-3.5-Flash".to_string();
        sibling.n_vocab = 128_000;
        let error = ModelParallelContract::from_model(&sibling).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("only the exact Step-3.7-Flash geometry is registered")
        );
    }

    #[test]
    fn step_without_the_official_mtp_geometry_does_not_inherit_the_contract() {
        let mut stripped = step37_model_config();
        stripped.nextn_predict_layers = 0;
        let error = ModelParallelContract::from_model(&stripped).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("only the exact Step-3.7-Flash geometry is registered")
        );
    }

    #[test]
    fn hardware_target_classification_is_exact() {
        assert_eq!(
            HardwareTarget::from_device_name("NVIDIA RTX PRO 6000 Blackwell Server Edition")
                .unwrap(),
            HardwareTarget::RtxPro6000Blackwell
        );
        assert_eq!(
            HardwareTarget::from_device_name("NVIDIA GeForce RTX 5090 Laptop GPU").unwrap(),
            HardwareTarget::Rtx5090
        );
        assert!(HardwareTarget::from_device_name("NVIDIA H100 80GB HBM3").is_err());
    }

    #[test]
    fn step_tp2_tp4_tp8_and_hybrid_plans_are_geometry_valid() {
        let tp2 = step37_contract().plan(request(1, 2, true)).unwrap();
        assert_eq!(tp2.query_head_range(0, 1), Some(32..64));
        assert_eq!(tp2.query_head_range(1, 1), Some(48..96));
        assert_eq!(tp2.kv_head_range(0, 1), Some(4..8));
        assert_eq!(tp2.routed_expert_range(1), Some(144..288));

        let tp4 = step37_contract().plan(request(1, 4, true)).unwrap();
        assert_eq!(tp4.query_head_range(0, 3), Some(48..64));
        assert_eq!(tp4.query_head_range(1, 3), Some(72..96));
        assert_eq!(tp4.kv_head_range(0, 3), Some(6..8));
        assert_eq!(tp4.query_feature_range(0, 3), Some(6144..8192));
        assert_eq!(tp4.query_feature_range(1, 3), Some(9216..12_288));
        assert_eq!(tp4.kv_feature_range(0, 3), Some(768..1024));
        assert_eq!(tp4.dense_ffn_range(3), Some(8448..11_264));
        assert_eq!(tp4.routed_expert_range(3), Some(216..288));
        assert!(tp4.shared_expert_replicated);

        let tp8 = step37_contract().plan(request(1, 8, true)).unwrap();
        assert_eq!(tp8.query_head_range(0, 7), Some(56..64));
        assert_eq!(tp8.query_head_range(1, 7), Some(84..96));
        assert_eq!(tp8.kv_head_range(0, 7), Some(7..8));
        assert_eq!(tp8.dense_ffn_range(7), Some(9856..11_264));
        assert_eq!(tp8.routed_expert_range(7), Some(252..288));
        assert!(tp8.shared_expert_replicated);

        let hybrid = step37_contract().plan(request(2, 4, true)).unwrap();
        assert_eq!(hybrid.world_size, 8);
        assert_eq!(hybrid.stage_ranges, vec![0..22, 22..45]);
        assert_eq!(hybrid.global_rank(1, 3), Some(7));
        assert_eq!(hybrid.global_rank(2, 0), None);
    }

    #[test]
    fn grouped_three_card_plan_is_pp1_then_tp2_ep2() {
        let plan = step37_contract()
            .plan_grouped(GroupedTopologyRequest {
                stages: vec![
                    StageGroupRequest {
                        layers: 0..15,
                        tensor: 1,
                        expert_parallel: false,
                    },
                    StageGroupRequest {
                        layers: 15..45,
                        tensor: 2,
                        expert_parallel: true,
                    },
                ],
                available_devices: 3,
                hardware: HardwareTarget::RtxPro6000Blackwell,
            })
            .unwrap();

        assert_eq!(plan.world_size, 3);
        assert_eq!(plan.rank_groups[0].global_ranks, 0..1);
        assert_eq!(plan.rank_groups[1].global_ranks, 1..3);
        assert_eq!(plan.global_rank(0, 0), Some(0));
        assert_eq!(plan.global_rank(1, 0), Some(1));
        assert_eq!(plan.global_rank(1, 1), Some(2));
        assert_eq!(plan.query_head_range(16, 1), Some(32..64));
        assert_eq!(plan.query_head_range(17, 1), Some(48..96));
        assert_eq!(plan.kv_head_range(16, 1), Some(4..8));
        assert_eq!(plan.routed_expert_range(16, 1), Some(144..288));
        assert_eq!(plan.mtp_owner_stage, Some(1));
    }

    #[test]
    fn grouped_step_layouts_cover_every_card_count_through_eight() {
        let layouts: Vec<Vec<usize>> = vec![
            vec![1],
            vec![2],
            vec![1, 2],
            vec![4],
            vec![1, 4],
            vec![2, 4],
            vec![1, 2, 4],
            vec![8],
        ];
        for (index, widths) in layouts.into_iter().enumerate() {
            let cards = index + 1;
            let cuts: Vec<usize> = match widths.len() {
                1 => vec![0, 45],
                2 => vec![0, 3, 45],
                3 => vec![0, 3, 15, 45],
                _ => unreachable!(),
            };
            let stages = widths
                .iter()
                .enumerate()
                .map(|(stage, &tensor)| StageGroupRequest {
                    layers: cuts[stage]..cuts[stage + 1],
                    tensor,
                    expert_parallel: tensor > 1 && cuts[stage + 1] > 3,
                })
                .collect();
            let plan = step37_contract()
                .plan_grouped(GroupedTopologyRequest {
                    stages,
                    available_devices: cards,
                    hardware: HardwareTarget::RtxPro6000Blackwell,
                })
                .unwrap_or_else(|error| panic!("{cards}-card grouped plan failed: {error}"));
            assert_eq!(plan.world_size, cards);
            assert_eq!(plan.rank_groups.last().unwrap().layers.end, 45);
        }
    }

    #[test]
    fn grouped_step_layout_refuses_gaps_overlap_and_invalid_stage_tp() {
        for stages in [
            vec![
                StageGroupRequest {
                    layers: 0..3,
                    tensor: 1,
                    expert_parallel: false,
                },
                StageGroupRequest {
                    layers: 4..45,
                    tensor: 2,
                    expert_parallel: true,
                },
            ],
            vec![
                StageGroupRequest {
                    layers: 0..16,
                    tensor: 1,
                    expert_parallel: false,
                },
                StageGroupRequest {
                    layers: 15..45,
                    tensor: 2,
                    expert_parallel: true,
                },
            ],
        ] {
            let error = step37_contract()
                .plan_grouped(GroupedTopologyRequest {
                    stages,
                    available_devices: 3,
                    hardware: HardwareTarget::RtxPro6000Blackwell,
                })
                .unwrap_err();
            assert!(error.to_string().contains("do not continue"));
        }

        let tp3 = step37_contract()
            .plan_grouped(GroupedTopologyRequest {
                stages: vec![StageGroupRequest {
                    layers: 0..45,
                    tensor: 3,
                    expert_parallel: true,
                }],
                available_devices: 3,
                hardware: HardwareTarget::RtxPro6000Blackwell,
            })
            .unwrap_err();
        assert!(tp3.to_string().contains("layer 0 query heads 64"));
    }

    #[test]
    fn full_model_step_tp8_preflight_binds_one_runtime_group() {
        let contract = step37_contract();
        let devices = (0..8).collect::<Vec<_>>();
        let owners = vec![0; contract.trunk_layers];
        let plan = contract
            .preflight_step_tp_specs(
                (0..contract.trunk_layers).map(|layer| (layer, devices.as_slice())),
                &owners,
            )
            .unwrap();

        assert!(plan.full_trunk);
        assert_eq!(plan.layers.len(), STEP37_TRUNK_LAYERS);
        assert_eq!(plan.runtime_groups, vec![devices.clone()]);
        assert_eq!(plan.dense_attention_layers(), 3);
        assert_eq!(plan.tensor_parallel_expert_layers(), 0);
        assert_eq!(plan.expert_parallel_layers(), 42);
        assert_eq!(plan.layers.first().unwrap().layer, 0);
        assert_eq!(plan.layers.last().unwrap().layer, 44);
        assert!(
            plan.layers
                .iter()
                .all(|layer| layer.owner_device == 0 && layer.devices == devices)
        );
    }

    #[test]
    fn step_tp_preflight_is_partial_for_tp2_and_fails_closed_on_invalid_specs() {
        let contract = step37_contract();
        let owners = vec![0; contract.trunk_layers];
        let tp2 = vec![0, 1];
        let partial = contract
            .preflight_step_tp_specs([(3, tp2.as_slice()), (44, tp2.as_slice())], &owners)
            .unwrap();
        assert!(!partial.full_trunk);
        assert_eq!(partial.runtime_groups, vec![tp2]);
        assert_eq!(partial.tensor_parallel_expert_layers(), 2);
        assert_eq!(partial.expert_parallel_layers(), 0);

        let wrong_owner = vec![1, 2];
        assert!(
            contract
                .preflight_step_tp_specs([(24, wrong_owner.as_slice())], &owners)
                .unwrap_err()
                .to_string()
                .contains("owning PP device 0 must be the first rank")
        );

        let tp3 = vec![0, 1, 2];
        assert!(
            contract
                .preflight_step_tp_specs([(24, tp3.as_slice())], &owners)
                .unwrap_err()
                .to_string()
                .contains("layer 0 query heads 64")
        );

        assert!(
            contract
                .preflight_step_tp_specs(
                    [(24, [0, 1].as_slice()), (24, [0, 1].as_slice())],
                    &owners,
                )
                .unwrap_err()
                .to_string()
                .contains("assigns layer 24 more than once")
        );
    }

    #[test]
    fn step_tp4_requires_whole_expert_parallelism() {
        let error = step37_contract().plan(request(1, 4, false)).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("routed expert FFN size shard 320")
        );
        let tp2 = step37_contract().plan(request(1, 2, false)).unwrap();
        assert_eq!(tp2.routed_expert_ffn_range(1), Some(640..1280));
        assert!(tp2.shared_expert_replicated);
    }

    #[test]
    fn step_tp3_refuses_the_real_per_layer_head_geometry() {
        let error = step37_contract()
            .plan(TopologyRequest {
                pipeline: 1,
                tensor: 3,
                expert_parallel: true,
                available_devices: 3,
                hardware: HardwareTarget::RtxPro6000Blackwell,
            })
            .unwrap_err();
        assert!(error.to_string().contains("layer 0 query heads 64"));
    }

    #[test]
    fn product_envelope_accepts_eight_and_refuses_more() {
        let pp8 = step37_contract().plan(request(8, 1, false)).unwrap();
        assert_eq!(pp8.world_size, 8);
        assert_eq!(pp8.stage_ranges.len(), 8);
        assert!(pp8.stage_ranges.iter().all(|range| !range.is_empty()));

        let error = step37_contract()
            .plan(TopologyRequest {
                pipeline: 3,
                tensor: 4,
                expert_parallel: true,
                available_devices: 12,
                hardware: HardwareTarget::RtxPro6000Blackwell,
            })
            .unwrap_err();
        assert!(error.to_string().contains("product envelope is 8"));
    }

    #[test]
    fn expert_parallel_requires_a_multi_rank_tp_group() {
        let error = step37_contract().plan(request(3, 1, true)).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("expert parallelism requires TP group size greater than one")
        );
    }

    #[test]
    fn step_does_not_inherit_the_5090_hardware_contract() {
        let error = step37_contract()
            .plan(TopologyRequest {
                pipeline: 1,
                tensor: 1,
                expert_parallel: false,
                available_devices: 1,
                hardware: HardwareTarget::Rtx5090,
            })
            .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("has no qualified rtx-5090 contract")
        );
    }
}
