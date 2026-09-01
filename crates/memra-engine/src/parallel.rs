//! ModelPlan-driven parallel topology and artifact placement contracts.
//!
//! The automatic loader does not select a family-specific TP/EP recipe. It compiles the canonical
//! operation plan, binds the source tensor census, estimates the checkpoint residency of every
//! legal program, and then selects a registered numeric backend. Family packs remain responsible
//! for semantic tensor/config validation; they do not carry per-layer placement lists.

use std::fmt;
use std::ops::Range;

use memra_gguf::config::ModelConfig;
use memra_gguf::model_plan::{MlpPlan, ModelPlan};
use memra_gguf::placement::{LayerPlacementCost, PlacementRequest, plan_contiguous_stages};
use memra_gguf::source::{ExpertActivationPrecision, TensorSource};
use memra_gguf::tensor_contract::{
    ContractOptions, LayerTensor, OutputHead, TensorContract, TensorId, TensorOwner,
};

/// The execution planner's supported rank envelope. Hardware qualification and tuned defaults
/// remain model x rig evidence, but the placement/runtime contract must not stop at earlier
/// three-card qualification cells.
pub const PRODUCT_MAX_CARDS: usize = 8;
pub const AUTO_PARALLEL_MAX_CARDS: usize = 4;
pub const STEP37_TRUNK_LAYERS: usize = 45;
const STEP_FP8_BLOCK: usize = 128;
const AUTO_PARALLEL_RESERVE_MB_DEFAULT: u64 = 6_144;

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
    /// True when every layer exposes a Full/Sliding attention geometry that the generic TP
    /// planner can shard. Expert-only EP does not require this.
    pub tensor_attention_supported: bool,
    pub expert_count: usize,
    pub experts_per_token: usize,
    pub expert_ffn_size: usize,
    pub shared_expert_ffn_size: usize,
    /// Trunk layer indices whose ModelPlan MLP is routed MoE. This is the automatic EP scope.
    pub routed_layers: Vec<usize>,
    pub partition_boundaries: Vec<usize>,
    pub hardware_targets: Vec<HardwareTarget>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AutoParallelBackend {
    Pipeline,
    ExpertParallel,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct AutoParallelPlacement {
    pub backend: AutoParallelBackend,
    pub devices: Vec<usize>,
    pub routed_layers: Vec<usize>,
    pub pipeline_splits: Vec<usize>,
    pub checkpoint_peak_bytes: u64,
    pub expert_root_bytes: u64,
    pub expert_peer_bytes: u64,
    pub reserve_bytes: u64,
    pub device_capacity_bytes: Vec<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct AutoArtifactCosts {
    layers: Vec<LayerPlacementCost>,
    first_fixed_bytes: u64,
    last_fixed_bytes: u64,
    trunk_expert_bytes: u64,
    non_distributed_bytes: u64,
}

fn placement_first_stage_tensor(id: &TensorId) -> bool {
    match id {
        TensorId::TokenEmbedding | TensorId::RopeFactors | TensorId::Vision { .. } => true,
        TensorId::QuantAux { tensor, .. } => placement_first_stage_tensor(tensor),
        _ => false,
    }
}

fn routed_expert_tensor(id: &TensorId) -> bool {
    match id {
        TensorId::Expert { .. } => true,
        TensorId::Layer {
            tensor:
                LayerTensor::MoeExpertGateUpBank
                | LayerTensor::MoeExpertGateBank
                | LayerTensor::MoeExpertUpBank
                | LayerTensor::MoeExpertDownBank
                | LayerTensor::MoeExpertOutputScale,
            ..
        } => true,
        TensorId::QuantAux { tensor, .. } => routed_expert_tensor(tensor),
        _ => false,
    }
}

fn checked_add_bytes(total: &mut u64, bytes: u64, label: &str) -> Result<(), TopologyError> {
    *total = total
        .checked_add(bytes)
        .ok_or_else(|| TopologyError::new(format!("{label} byte total overflows u64")))?;
    Ok(())
}

fn artifact_costs(
    src: &dyn TensorSource,
    cfg: &ModelConfig,
    plan: &ModelPlan,
) -> Result<AutoArtifactCosts, TopologyError> {
    let census = src.tensor_census().map_err(|error| {
        TopologyError::new(format!(
            "automatic parallel placement requires a source tensor census: {error}"
        ))
    })?;
    let output_head = if census
        .tensors
        .iter()
        .any(|row| row.entry.name == "lm_head.weight" || row.entry.name == "output.weight")
    {
        OutputHead::Separate
    } else {
        OutputHead::TiedToEmbedding
    };
    let contract = match memra_gguf::model_packs::for_config(cfg) {
        Some(pack) => {
            pack.compile_tensor_contract(cfg, plan, census.dialect, ContractOptions { output_head })
        }
        None => TensorContract::for_plan(plan, census.dialect, ContractOptions { output_head }),
    }
    .map_err(|error| {
        TopologyError::new(format!(
            "cannot compile automatic parallel tensor contract: {error}"
        ))
    })?;
    let entries = census
        .tensors
        .iter()
        .map(|row| row.entry.clone())
        .collect::<Vec<_>>();
    let binding = contract.bind(&entries).map_err(|error| {
        TopologyError::new(format!(
            "cannot bind automatic parallel tensor census: {error}"
        ))
    })?;

    let mut layers = vec![LayerPlacementCost::default(); plan.layers.len()];
    let mut first_fixed_bytes = 0u64;
    let mut last_fixed_bytes = 0u64;
    let mut trunk_expert_bytes = 0u64;
    let mut total_bytes = 0u64;
    for (id, tensor) in &binding.tensors {
        checked_add_bytes(
            &mut total_bytes,
            tensor.physical_bytes,
            "automatic placement checkpoint",
        )?;
        match tensor.owner {
            TensorOwner::Layer(layer) if (layer as usize) < layers.len() => {
                checked_add_bytes(
                    &mut layers[layer as usize].weight_bytes,
                    tensor.physical_bytes,
                    "automatic placement layer",
                )?;
            }
            // Some legacy contracts retain the physical MTP index rather than rewriting the
            // owner to TensorOwner::Mtp. It executes with the tail/head stage either way.
            TensorOwner::Layer(_) => checked_add_bytes(
                &mut last_fixed_bytes,
                tensor.physical_bytes,
                "automatic placement head stage",
            )?,
            TensorOwner::Vision(_) => checked_add_bytes(
                &mut first_fixed_bytes,
                tensor.physical_bytes,
                "automatic placement first stage",
            )?,
            TensorOwner::Global if placement_first_stage_tensor(id) => checked_add_bytes(
                &mut first_fixed_bytes,
                tensor.physical_bytes,
                "automatic placement first stage",
            )?,
            TensorOwner::Global | TensorOwner::Mtp(_) => checked_add_bytes(
                &mut last_fixed_bytes,
                tensor.physical_bytes,
                "automatic placement head stage",
            )?,
        }

        let trunk_expert = routed_expert_tensor(id)
            && matches!(
                tensor.owner,
                TensorOwner::Layer(layer)
                    if (layer as usize) < plan.layers.len()
                        && matches!(plan.layers[layer as usize].mlp, MlpPlan::Moe(_))
            );
        if trunk_expert {
            checked_add_bytes(
                &mut trunk_expert_bytes,
                tensor.physical_bytes,
                "automatic placement trunk experts",
            )?;
        }
    }
    let non_distributed_bytes = total_bytes
        .checked_sub(trunk_expert_bytes)
        .ok_or_else(|| TopologyError::new("automatic placement expert bytes exceed total bytes"))?;
    Ok(AutoArtifactCosts {
        layers,
        first_fixed_bytes,
        last_fixed_bytes,
        trunk_expert_bytes,
        non_distributed_bytes,
    })
}

fn auto_parallel_reserve_bytes() -> Result<u64, TopologyError> {
    let reserve_mb = match std::env::var("MEMRA_PARALLEL_RESERVE_MB") {
        Ok(raw) => raw.parse::<u64>().map_err(|_| {
            TopologyError::new(format!(
                "MEMRA_PARALLEL_RESERVE_MB={raw:?} is not an unsigned integer"
            ))
        })?,
        Err(std::env::VarError::NotPresent) => AUTO_PARALLEL_RESERVE_MB_DEFAULT,
        Err(error) => {
            return Err(TopologyError::new(format!(
                "cannot read MEMRA_PARALLEL_RESERVE_MB: {error}"
            )));
        }
    };
    reserve_mb
        .checked_mul(1024 * 1024)
        .ok_or_else(|| TopologyError::new("MEMRA_PARALLEL_RESERVE_MB overflows bytes"))
}

fn device_capacity_bytes(devices: &[usize]) -> Result<Vec<u64>, TopologyError> {
    cudarc::driver::result::init().map_err(|error| {
        TopologyError::new(format!("CUDA driver initialization failed: {error}"))
    })?;
    devices
        .iter()
        .map(|&ordinal| {
            let device = cudarc::driver::result::device::get(ordinal as i32).map_err(|error| {
                TopologyError::new(format!("CUDA device {ordinal} lookup failed: {error}"))
            })?;
            // SAFETY: `device` was returned by CUDA for this exact process-local ordinal.
            let bytes =
                unsafe { cudarc::driver::result::device::total_mem(device) }.map_err(|error| {
                    TopologyError::new(format!(
                        "CUDA device {ordinal} memory query failed: {error}"
                    ))
                })?;
            u64::try_from(bytes).map_err(|_| {
                TopologyError::new(format!("CUDA device {ordinal} memory exceeds u64"))
            })
        })
        .collect()
}

fn fits_capacity(bytes: u64, reserve: u64, capacity: u64) -> bool {
    bytes
        .checked_add(reserve)
        .is_some_and(|required| required <= capacity)
}

fn choose_auto_parallel_placement(
    costs: &AutoArtifactCosts,
    contract: &ModelParallelContract,
    activation: ExpertActivationPrecision,
    devices: &[usize],
    capacity_bytes: &[u64],
    reserve_bytes: u64,
) -> Result<AutoParallelPlacement, TopologyError> {
    if devices.len() != capacity_bytes.len() {
        return Err(TopologyError::new(format!(
            "automatic placement has {} devices but {} capacity rows",
            devices.len(),
            capacity_bytes.len()
        )));
    }
    let mut fixed = vec![0u64; devices.len()];
    fixed[0] = costs.first_fixed_bytes;
    fixed[devices.len() - 1] = fixed[devices.len() - 1]
        .checked_add(costs.last_fixed_bytes)
        .ok_or_else(|| TopologyError::new("automatic placement fixed bytes overflow"))?;
    let pipeline = plan_contiguous_stages(PlacementRequest {
        layers: &costs.layers,
        fixed_stage_bytes: &fixed,
        context_tokens: 0,
        devices,
        legal_boundaries: &contract.partition_boundaries,
    })
    .map_err(|error| TopologyError::new(format!("automatic PP placement failed: {error}")))?;
    let pipeline_fits = pipeline
        .stages
        .iter()
        .enumerate()
        .all(|(stage, placement)| {
            fits_capacity(
                placement.cost.total_bytes,
                reserve_bytes,
                capacity_bytes[stage],
            )
        });

    let world = devices.len() as u64;
    let expert_peer_bytes = if contract.expert_count == 0 {
        0
    } else {
        let expert_count = contract.expert_count as u64;
        let bytes_per_expert = costs.trunk_expert_bytes.div_ceil(expert_count);
        bytes_per_expert
            .checked_mul(expert_count.div_ceil(world))
            .ok_or_else(|| TopologyError::new("automatic EP peer byte total overflows"))?
    };
    let expert_root_bytes = costs
        .non_distributed_bytes
        .checked_add(expert_peer_bytes)
        .ok_or_else(|| TopologyError::new("automatic EP root byte total overflows"))?;
    let expert_fits = !contract.routed_layers.is_empty()
        && activation == ExpertActivationPrecision::Bf16
        && capacity_bytes.iter().enumerate().all(|(rank, &capacity)| {
            let bytes = if rank == 0 {
                expert_root_bytes
            } else {
                expert_peer_bytes
            };
            fits_capacity(bytes, reserve_bytes, capacity)
        });

    if expert_fits {
        return Ok(AutoParallelPlacement {
            backend: AutoParallelBackend::ExpertParallel,
            devices: devices.to_vec(),
            routed_layers: contract.routed_layers.clone(),
            pipeline_splits: Vec::new(),
            checkpoint_peak_bytes: expert_root_bytes,
            expert_root_bytes,
            expert_peer_bytes,
            reserve_bytes,
            device_capacity_bytes: capacity_bytes.to_vec(),
        });
    }
    if pipeline_fits {
        let pipeline_splits = pipeline
            .stages
            .iter()
            .take(pipeline.stages.len() - 1)
            .map(|stage| stage.layers.end)
            .collect();
        return Ok(AutoParallelPlacement {
            backend: AutoParallelBackend::Pipeline,
            devices: devices.to_vec(),
            routed_layers: contract.routed_layers.clone(),
            pipeline_splits,
            checkpoint_peak_bytes: pipeline.max_stage_bytes,
            expert_root_bytes,
            expert_peer_bytes,
            reserve_bytes,
            device_capacity_bytes: capacity_bytes.to_vec(),
        });
    }

    Err(TopologyError::new(format!(
        "automatic placement found no capacity-safe program: PP peak={} bytes, EP root={} bytes, \
         reserve={} bytes, device capacities={capacity_bytes:?}",
        pipeline.max_stage_bytes, expert_root_bytes, reserve_bytes,
    )))
}

pub(crate) fn plan_auto_parallel(
    src: &dyn TensorSource,
    cfg: &ModelConfig,
    plan: &ModelPlan,
    devices: &[usize],
) -> Result<AutoParallelPlacement, TopologyError> {
    let hardware = detect_uniform_hardware(devices)?;
    let contract = ModelParallelContract::from_plan(cfg, plan)?;
    if !contract.hardware_targets.contains(&hardware) {
        return Err(TopologyError::new(format!(
            "{} has no qualified {} automatic placement contract",
            contract.variant,
            hardware.label()
        )));
    }
    let costs = artifact_costs(src, cfg, plan)?;
    let capacity_bytes = device_capacity_bytes(devices)?;
    let reserve_bytes = auto_parallel_reserve_bytes()?;
    choose_auto_parallel_placement(
        &costs,
        &contract,
        src.expert_activation_precision(),
        devices,
        &capacity_bytes,
        reserve_bytes,
    )
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
    /// Build the structural contract from the canonical operation plan. Unsupported operations
    /// refuse during ModelPlan compilation; family names do not select placement.
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
                    Some((
                        attention.query_heads as usize,
                        attention.kv_heads as usize,
                        attention.key_head_dim as usize,
                    ))
                }
                _ => None,
            })
            .collect::<Vec<_>>();
        let query_heads: Vec<_> = attention_geometry
            .iter()
            .map(|geometry| geometry.map_or(0, |geometry| geometry.0))
            .collect();
        let kv_heads: Vec<_> = attention_geometry
            .iter()
            .map(|geometry| geometry.map_or(0, |geometry| geometry.1))
            .collect();
        let head_dim = attention_geometry
            .iter()
            .flatten()
            .map(|geometry| geometry.2)
            .next()
            .unwrap_or(cfg.head_dim_k as usize);
        let tensor_attention_supported = attention_geometry
            .iter()
            .all(|geometry| geometry.is_some_and(|geometry| geometry.2 == head_dim));
        let dense_prefix_layers = plan
            .layers
            .iter()
            .take_while(|layer| matches!(layer.mlp, MlpPlan::Dense(_)))
            .count();
        if plan.layers[dense_prefix_layers..]
            .iter()
            .any(|layer| matches!(layer.mlp, MlpPlan::Dense(_)))
        {
            return Err(TopologyError::new(
                "generic parallel loader requires dense layers to form one prefix before routed \
                 MoE layers",
            ));
        }
        let dense_sizes = layers
            .iter()
            .filter_map(|layer| match &layer.mlp {
                MlpPlan::Dense(dense) => Some(dense.intermediate_size as usize),
                _ => None,
            })
            .collect::<std::collections::BTreeSet<_>>();
        if dense_sizes.len() > 1 {
            return Err(TopologyError::new(format!(
                "generic parallel loader requires one dense FFN width, got {dense_sizes:?}"
            )));
        }
        let dense_ffn_size = dense_sizes.iter().next().copied().unwrap_or(0);
        let routed_layers = plan
            .layers
            .iter()
            .enumerate()
            .filter_map(|(layer, plan)| match plan.mlp {
                MlpPlan::Moe(_) => Some(layer),
                MlpPlan::Dense(_) => None,
            })
            .collect::<Vec<_>>();
        let moe_layers = layers
            .iter()
            .filter_map(|layer| match &layer.mlp {
                MlpPlan::Moe(moe) => Some(moe),
                MlpPlan::Dense(_) => None,
            })
            .collect::<Vec<_>>();
        let (expert_count, experts_per_token, expert_ffn_size, shared_expert_ffn_size) =
            if let Some(first) = moe_layers.first() {
                let shared = first
                    .shared
                    .as_ref()
                    .map_or(0, |shared| shared.intermediate_size as usize);
                if moe_layers.iter().any(|moe| {
                    moe.expert_count != first.expert_count
                        || moe.experts_per_token != first.experts_per_token
                        || moe.expert_intermediate_size != first.expert_intermediate_size
                        || moe
                            .shared
                            .as_ref()
                            .map_or(0, |shared| shared.intermediate_size as usize)
                            != shared
                }) {
                    return Err(TopologyError::new(
                        "generic parallel loader requires one routed-expert geometry across the \
                         selected model plan",
                    ));
                }
                (
                    first.expert_count as usize,
                    first.experts_per_token as usize,
                    first.expert_intermediate_size as usize,
                    shared,
                )
            } else {
                (0, 0, 0, 0)
            };
        if routed_layers.is_empty() && dense_ffn_size == 0 {
            return Err(TopologyError::new(
                "generic parallel loader found neither dense nor routed MLP layers",
            ));
        };

        Ok(Self {
            family: if routed_layers.is_empty() {
                "dense-transformer"
            } else {
                "routed-moe"
            },
            variant: cfg.name.clone(),
            trunk_layers,
            mtp_layers,
            hidden_size: cfg.n_embd as usize,
            vocab_size: cfg.n_vocab as usize,
            dense_ffn_size,
            dense_prefix_layers,
            head_dim,
            query_heads,
            kv_heads,
            tensor_attention_supported,
            expert_count,
            experts_per_token,
            expert_ffn_size,
            shared_expert_ffn_size,
            routed_layers,
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
        if request.expert_parallel && self.expert_count == 0 {
            return Err(TopologyError::new(
                "expert parallelism requested for a dense-only ModelPlan",
            ));
        }
        if tp > 1 && !self.tensor_attention_supported {
            return Err(TopologyError::new(format!(
                "{} has attention operations without a generic TP shard contract; expert-only \
                 EP may still be selected independently",
                self.variant
            )));
        }

        // Check the plan-derived per-layer attention geometry before generic dimensions so a
        // refused topology names the operation that actually makes it invalid.
        for (il, (&q, &kv)) in self.query_heads.iter().zip(&self.kv_heads).enumerate() {
            require_divisible(&format!("layer {il} query heads"), q, tp)?;
            require_divisible(&format!("layer {il} KV heads"), kv, tp)?;
        }
        require_divisible("hidden size", self.hidden_size, tp)?;
        require_divisible("vocabulary size", self.vocab_size, tp)?;
        if self.dense_ffn_size > 0 {
            require_divisible("dense FFN size", self.dense_ffn_size, tp)?;
        }
        if self.expert_count > 0 {
            if request.expert_parallel {
                require_divisible("routed expert count", self.expert_count, tp)?;
            } else {
                require_divisible("routed expert FFN size", self.expert_ffn_size, tp)?;
            }
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
            // Shared experts remain replicated until a backend advertises a sharded shared branch.
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

/// Prove that every routed layer in the structural contract exposes native stacked block-128
/// E4M3 expert banks. Converted and per-tensor artifacts do not inherit this backend.
pub fn validate_fp8_expert_checkpoint(
    src: &dyn TensorSource,
    contract: &ModelParallelContract,
) -> Result<usize, TopologyError> {
    if src.st_dir().is_none() {
        return Err(TopologyError::new(
            "native E4M3 expert parallelism requires a safetensors checkpoint source; a \
             converted artifact cannot inherit this backend",
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
            "E4M3 expert tensor census qualified {qualified}, expected {expected}"
        )));
    }
    Ok(qualified)
}

/// Prove that every routed layer in the structural contract exposes native ModelOpt NVFP4
/// experts: either one stacked bank or the Hugging Face per-expert layout. Both carry packed
/// e2m1 codes, per-16 UE4M3 scales, and finite-positive per-expert macros.
pub fn validate_nvfp4_expert_checkpoint(
    src: &dyn TensorSource,
    contract: &ModelParallelContract,
) -> Result<usize, TopologyError> {
    if src.st_dir().is_none() {
        return Err(TopologyError::new(
            "native NVFP4 expert parallelism requires a safetensors checkpoint source; a \
             converted artifact cannot inherit this backend",
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
            if let Some(bank) = src.find_nvfp4_stacked_native(&name) {
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
                continue;
            }

            for expert in 0..contract.expert_count {
                let expert_name = format!("blk.{layer}.{projection}.{expert}.weight");
                let tensor = src.find_nvfp4_native(&expert_name).ok_or_else(|| {
                    TopologyError::new(format!(
                        "{name} is neither a checkpoint-faithful stacked modelopt NVFP4 bank nor \
                         a complete per-expert NVFP4 set; missing {expert_name}"
                    ))
                })?;
                if tensor.in_f != expected_in || tensor.out_f != expected_out {
                    return Err(TopologyError::new(format!(
                        "{expert_name} shape {}x{} != expected {expected_out}x{expected_in}",
                        tensor.out_f, tensor.in_f
                    )));
                }
                if tensor.in_f % 64 != 0 {
                    return Err(TopologyError::new(format!(
                        "{expert_name} in_features {} is not 64-aligned; memra block_nvfp4 \
                         kernels require whole 64-element superblocks",
                        tensor.in_f
                    )));
                }
                qualified += 1;
            }
        }
    }

    let expected = (contract.trunk_layers - contract.dense_prefix_layers)
        * contract.expert_count
        * projections.len();
    if qualified != expected {
        return Err(TopologyError::new(format!(
            "NVFP4 expert tensor census qualified {qualified}, expected {expected}"
        )));
    }
    Ok(qualified)
}

/// Legacy API retained for existing focused gates.
pub fn validate_step_fp8_checkpoint(
    src: &dyn TensorSource,
    contract: &ModelParallelContract,
) -> Result<usize, TopologyError> {
    validate_fp8_expert_checkpoint(src, contract)
}

/// Legacy API retained for existing focused gates.
pub fn validate_step_nvfp4_checkpoint(
    src: &dyn TensorSource,
    contract: &ModelParallelContract,
) -> Result<usize, TopologyError> {
    validate_nvfp4_expert_checkpoint(src, contract)
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

#[allow(clippy::manual_is_multiple_of)] // allow: divisor is runtime-derived; the modulo form keeps a zero divisor loud (a panic), where is_multiple_of would return false silently
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
    if !local.is_multiple_of(STEP_FP8_BLOCK) {
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

#[allow(clippy::manual_is_multiple_of)] // allow: divisor is runtime-derived; the modulo form keeps a zero divisor loud (a panic), where is_multiple_of would return false silently
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
    use memra_gguf::config::{Arch, HfConfig, MoeConfig, Step35Config};
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
            tensor_attention_supported: true,
            expert_count: 288,
            experts_per_token: 8,
            expert_ffn_size: 1280,
            shared_expert_ffn_size: 1280,
            routed_layers: (3..45).collect(),
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
            vision_glm5: None,
            multimodal: None,
            mla: None,
            dsv4: None,
            qwen4exp: None,
            rope_yarn: None,
            glm5: None,
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

    fn hy3_model_config() -> ModelConfig {
        ModelConfig::from_hf(&HfConfig::parse(
            r#"{
                "model_type":"hy_v3",
                "num_hidden_layers":80,
                "num_nextn_predict_layers":1,
                "hidden_size":4096,
                "num_attention_heads":64,
                "num_key_value_heads":8,
                "head_dim":128,
                "intermediate_size":13312,
                "vocab_size":120832,
                "max_position_embeddings":262144,
                "first_k_dense_replace":1,
                "num_experts":192,
                "num_experts_per_tok":8,
                "moe_intermediate_size":1536,
                "num_shared_experts":1,
                "moe_router_use_sigmoid":true,
                "moe_router_enable_expert_bias":true,
                "route_norm":true,
                "router_scaling_factor":2.826,
                "qk_norm":true
            }"#,
        ))
    }

    fn dense_model_config() -> ModelConfig {
        ModelConfig::from_hf(&HfConfig::parse(
            r#"{
                "model_type":"qwen3",
                "num_hidden_layers":4,
                "hidden_size":4096,
                "num_attention_heads":32,
                "num_key_value_heads":8,
                "head_dim":128,
                "intermediate_size":12288,
                "vocab_size":131072,
                "max_position_embeddings":32768
            }"#,
        ))
    }

    fn synthetic_auto_contract(routed: bool) -> ModelParallelContract {
        ModelParallelContract {
            family: if routed {
                "routed-moe"
            } else {
                "dense-transformer"
            },
            variant: "synthetic-auto".to_string(),
            trunk_layers: 4,
            mtp_layers: 0,
            hidden_size: 64,
            vocab_size: 128,
            dense_ffn_size: 128,
            dense_prefix_layers: if routed { 1 } else { 4 },
            head_dim: 32,
            query_heads: vec![2; 4],
            kv_heads: vec![1; 4],
            tensor_attention_supported: true,
            expert_count: if routed { 8 } else { 0 },
            experts_per_token: if routed { 2 } else { 0 },
            expert_ffn_size: if routed { 32 } else { 0 },
            shared_expert_ffn_size: 0,
            routed_layers: if routed { vec![1, 2, 3] } else { Vec::new() },
            partition_boundaries: vec![1, 2, 3],
            hardware_targets: vec![HardwareTarget::RtxPro6000Blackwell],
        }
    }

    fn synthetic_auto_costs(trunk_expert_bytes: u64) -> AutoArtifactCosts {
        AutoArtifactCosts {
            layers: vec![
                LayerPlacementCost {
                    weight_bytes: 25,
                    kv_bytes_per_token: 0,
                };
                4
            ],
            first_fixed_bytes: 0,
            last_fixed_bytes: 0,
            trunk_expert_bytes,
            non_distributed_bytes: 20,
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
                .contains("safetensors checkpoint source")
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
        assert_eq!(contract.family, "routed-moe");
        assert_eq!(contract.trunk_layers, 45);
        assert_eq!(contract.mtp_layers, 3);
        assert_eq!(contract.query_heads[0], 64);
        assert_eq!(contract.query_heads[1], 96);
        assert_eq!(contract.kv_heads[47], 8);
        assert_eq!(contract.expert_count, 288);
        assert_eq!(contract.experts_per_token, 8);
    }

    #[test]
    fn hy3_contract_is_extracted_from_exact_full_sigmoid_moe_geometry() {
        let contract = ModelParallelContract::from_model(&hy3_model_config()).unwrap();
        assert_eq!(contract.family, "routed-moe");
        assert_eq!(contract.trunk_layers, 80);
        assert_eq!(contract.mtp_layers, 1);
        assert_eq!(contract.query_heads, vec![64; 81]);
        assert_eq!(contract.kv_heads, vec![8; 81]);
        assert_eq!(contract.dense_prefix_layers, 1);
        assert_eq!(contract.expert_count, 192);
        assert_eq!(contract.experts_per_token, 8);
        assert_eq!(contract.expert_ffn_size, 1536);
        assert_eq!(contract.routed_layers, (1..80).collect::<Vec<_>>());
    }

    #[test]
    fn hy3_sibling_geometry_is_derived_without_a_family_loader() {
        let mut sibling = hy3_model_config();
        sibling.n_vocab += 1;
        let contract = ModelParallelContract::from_model(&sibling).unwrap();
        assert_eq!(contract.vocab_size, 120_833);
        assert_eq!(contract.routed_layers.len(), 79);
    }

    #[test]
    fn dense_transformer_contract_and_tp_geometry_are_plan_derived() {
        let contract = ModelParallelContract::from_model(&dense_model_config()).unwrap();
        assert_eq!(contract.family, "dense-transformer");
        assert!(contract.routed_layers.is_empty());
        assert_eq!(contract.dense_prefix_layers, 4);
        assert_eq!(contract.dense_ffn_size, 12_288);
        let tp4 = contract.plan(request(1, 4, false)).unwrap();
        assert_eq!(tp4.query_feature_range(0, 3), Some(3072..4096));
        assert_eq!(tp4.dense_ffn_range(3), Some(9216..12_288));
    }

    #[test]
    fn automatic_placement_uses_capacity_not_family_recipes() {
        let routed = synthetic_auto_contract(true);
        let costs = synthetic_auto_costs(80);

        let pp2 = choose_auto_parallel_placement(
            &costs,
            &routed,
            ExpertActivationPrecision::Bf16,
            &[0, 1],
            &[60, 60],
            6,
        )
        .unwrap();
        assert_eq!(pp2.backend, AutoParallelBackend::Pipeline);
        assert_eq!(pp2.pipeline_splits, vec![2]);
        assert_eq!(pp2.checkpoint_peak_bytes, 50);

        let ep3 = choose_auto_parallel_placement(
            &costs,
            &routed,
            ExpertActivationPrecision::Bf16,
            &[0, 1, 2],
            &[60, 60, 60],
            6,
        )
        .unwrap();
        assert_eq!(ep3.backend, AutoParallelBackend::ExpertParallel);
        assert_eq!(ep3.expert_root_bytes, 50);
        assert_eq!(ep3.expert_peer_bytes, 30);

        let ep4 = choose_auto_parallel_placement(
            &costs,
            &routed,
            ExpertActivationPrecision::Bf16,
            &[0, 1, 2, 3],
            &[60; 4],
            6,
        )
        .unwrap();
        assert_eq!(ep4.backend, AutoParallelBackend::ExpertParallel);
        assert_eq!(ep4.expert_root_bytes, 40);
        assert_eq!(ep4.expert_peer_bytes, 20);
    }

    #[test]
    fn automatic_placement_routes_dense_and_non_w4a16_plans_to_pipeline() {
        let costs = synthetic_auto_costs(80);
        let dense = choose_auto_parallel_placement(
            &costs,
            &synthetic_auto_contract(false),
            ExpertActivationPrecision::Bf16,
            &[0, 1, 2, 3],
            &[60; 4],
            6,
        )
        .unwrap();
        assert_eq!(dense.backend, AutoParallelBackend::Pipeline);

        let activation_quantized = choose_auto_parallel_placement(
            &costs,
            &synthetic_auto_contract(true),
            ExpertActivationPrecision::Quantized,
            &[0, 1, 2, 3],
            &[60; 4],
            6,
        )
        .unwrap();
        assert_eq!(activation_quantized.backend, AutoParallelBackend::Pipeline);
    }

    #[test]
    fn automatic_placement_refuses_when_no_program_preserves_reserve() {
        let error = choose_auto_parallel_placement(
            &synthetic_auto_costs(80),
            &synthetic_auto_contract(true),
            ExpertActivationPrecision::Bf16,
            &[0, 1],
            &[55, 55],
            6,
        )
        .unwrap_err();
        assert!(error.to_string().contains("no capacity-safe program"));
    }

    #[test]
    fn step_sibling_geometry_is_derived_without_a_family_loader() {
        let mut sibling = step37_model_config();
        sibling.name = "Step-3.5-Flash".to_string();
        sibling.n_vocab = 128_000;
        let contract = ModelParallelContract::from_model(&sibling).unwrap();
        assert_eq!(contract.variant, "Step-3.5-Flash");
        assert_eq!(contract.vocab_size, 128_000);
    }

    #[test]
    fn step_without_mtp_keeps_the_same_structural_parallel_contract() {
        let mut stripped = step37_model_config();
        stripped.nextn_predict_layers = 0;
        let contract = ModelParallelContract::from_model(&stripped).unwrap();
        assert_eq!(contract.mtp_layers, 0);
        assert_eq!(contract.routed_layers, (3..48).collect::<Vec<_>>());
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
    fn structural_tp4_defers_quant_block_legality_to_the_artifact_backend() {
        let tp4 = step37_contract().plan(request(1, 4, false)).unwrap();
        assert_eq!(tp4.routed_expert_ffn_range(1), Some(320..640));
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
