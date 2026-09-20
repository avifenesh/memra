//! Plan-derived catalog of routed-expert projection banks.
//!
//! An exact-byte consumer of expert banks (the `--experts-via-tier` installer in memra-engine)
//! must not spell checkpoint names itself. This module reads the compiled `ModelPlan` for its
//! MoE blocks, takes each projection's semantic tensor id, accepted names, required shape and
//! quant constraint from the `TensorContract` compiled for the artifact, and binds them against
//! the artifact's tensor census through the contract's own `bind`. A missing, ambiguous,
//! shape- or layout-incompatible entry is an error, and so is a scale plane the artifact
//! carries for a bank: the consumer must declare what it consumes, never skip a plane.
//!
//! The catalog is an identity, not a byte program: it names tensors and their extents in plan
//! order (trunk blocks by position, then MTP blocks by depth; gate, up, down inside a block).

use std::collections::BTreeSet;

use crate::model_plan::{MlpPlan, ModelPlan};
use crate::tensor_contract::{
    CheckpointDialect, ExpertTensor, LayerTensor, StorageLayout, TensorCensusEntry, TensorContract,
    TensorContractError, TensorId, TensorMatch,
};

/// Which block of the plan a bank belongs to. `Trunk.position` indexes `plan.layers`;
/// `Mtp.depth` is the NextN depth. The plan's own layer index travels beside it as
/// `checkpoint_layer`, the only number a checkpoint name may derive from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpertBankBlock {
    Trunk { position: usize },
    Mtp { depth: u32 },
}

/// One routed-expert projection bank bound to exactly one checkpoint tensor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpertBankProjection {
    pub block: ExpertBankBlock,
    pub checkpoint_layer: u32,
    pub projection: ExpertTensor,
    pub id: TensorId,
    pub name: String,
    /// Checkpoint-native dimension order (GGUF: inner to outer, experts last).
    pub shape: Vec<u64>,
    pub storage: StorageLayout,
    pub expert_count: u32,
    pub physical_bytes: u64,
}

/// Every routed-expert bank of the plan, three projections per MoE block, in plan order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpertBankCatalog {
    pub projections: Vec<ExpertBankProjection>,
}

pub const PROJECTIONS_PER_BLOCK: usize = 3;

impl ExpertBankCatalog {
    /// Blocks in plan order; every chunk holds the gate, up and down bank of one block.
    pub fn blocks(&self) -> impl Iterator<Item = &[ExpertBankProjection]> {
        self.projections.chunks(PROJECTIONS_PER_BLOCK)
    }

    /// Stable text identity of the catalog: one line per projection with its block, name,
    /// shape, storage and byte count. Consumers hash this to prove two derivations agree.
    pub fn identity(&self) -> String {
        let mut text = String::new();
        for projection in &self.projections {
            let block = match projection.block {
                ExpertBankBlock::Trunk { position } => format!("trunk:{position}"),
                ExpertBankBlock::Mtp { depth } => format!("mtp:{depth}"),
            };
            let shape: Vec<String> = projection.shape.iter().map(u64::to_string).collect();
            text.push_str(&format!(
                "{block}\t{}\t{:?}\t{}\t{}\t{}\t{}\n",
                projection.checkpoint_layer,
                projection.projection,
                projection.name,
                shape.join("x"),
                storage_text(&projection.storage),
                projection.physical_bytes
            ));
        }
        text
    }
}

fn storage_text(storage: &StorageLayout) -> String {
    match storage {
        StorageLayout::Float(kind) => format!("{kind:?}"),
        StorageLayout::Integer(kind) => format!("{kind:?}"),
        StorageLayout::Quantized(layout) => {
            let blocks: Vec<String> = layout.block_shape.iter().map(u32::to_string).collect();
            format!("{}[{}]", layout.format, blocks.join(","))
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExpertBankCatalogError {
    /// Only GGUF banks are stacked expert tensors the exact-byte consumers read today.
    Dialect(CheckpointDialect),
    /// The compiled plan has no MoE block, so there is no expert projection to bank.
    NoMoeExpertProjections,
    /// The contract has no entry for a projection the plan requires.
    ContractEntryMissing { id: TensorId },
    /// The contract carries more than one entry for one projection id.
    ContractEntryAmbiguous { id: TensorId, count: usize },
    /// The contract's own binding verdict: missing, duplicate, ambiguous, shape or layout.
    Contract(Box<TensorContractError>),
    /// The artifact carries scale planes for these banks; the consumer declares none.
    ScalePlanes { names: Vec<String> },
    /// The bound shape's expert dimension disagrees with the plan's expert count.
    ExpertCountMismatch {
        id: TensorId,
        plan: u32,
        shape: Vec<u64>,
    },
}

impl std::fmt::Display for ExpertBankCatalogError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Dialect(dialect) => {
                write!(f, "checkpoint dialect {dialect:?} is not GGUF")
            }
            Self::NoMoeExpertProjections => {
                f.write_str("the compiled plan has no MoE expert projections")
            }
            Self::ContractEntryMissing { id } => {
                write!(f, "tensor contract has no entry for {id:?}")
            }
            Self::ContractEntryAmbiguous { id, count } => {
                write!(f, "tensor contract has {count} entries for {id:?}")
            }
            Self::Contract(error) => write!(f, "{error}"),
            Self::ScalePlanes { names } => write!(
                f,
                "artifact carries expert scale planes the consumer does not declare: {}",
                names.join(", ")
            ),
            Self::ExpertCountMismatch { id, plan, shape } => write!(
                f,
                "{id:?} binds shape {shape:?} but the plan routes over {plan} experts"
            ),
        }
    }
}

impl std::error::Error for ExpertBankCatalogError {}

/// Derive the expert bank catalog of `plan` for the artifact described by `census`, taking
/// every name, shape and quant constraint from `contract` (compiled for the same plan).
pub fn expert_bank_catalog(
    plan: &ModelPlan,
    contract: &TensorContract,
    census: &[TensorCensusEntry],
) -> Result<ExpertBankCatalog, ExpertBankCatalogError> {
    if contract.dialect != CheckpointDialect::Gguf {
        return Err(ExpertBankCatalogError::Dialect(contract.dialect));
    }
    let mut blocks = Vec::new();
    for (position, layer) in plan.layers.iter().enumerate() {
        if let MlpPlan::Moe(moe) = &layer.mlp {
            blocks.push((
                ExpertBankBlock::Trunk { position },
                layer.index,
                moe.expert_count,
            ));
        }
    }
    for block in &plan.mtp_blocks {
        if let MlpPlan::Moe(moe) = &block.layer.mlp {
            blocks.push((
                ExpertBankBlock::Mtp { depth: block.depth },
                block.layer.index,
                moe.expert_count,
            ));
        }
    }
    if blocks.is_empty() {
        return Err(ExpertBankCatalogError::NoMoeExpertProjections);
    }

    let mut requirements = Vec::new();
    let mut wanted = Vec::new();
    for (block, index, expert_count) in blocks {
        for (projection, tensor) in [
            (ExpertTensor::Gate, LayerTensor::MoeExpertGateBank),
            (ExpertTensor::Up, LayerTensor::MoeExpertUpBank),
            (ExpertTensor::Down, LayerTensor::MoeExpertDownBank),
        ] {
            let id = TensorId::Layer { index, tensor };
            let matched: Vec<_> = contract
                .requirements
                .iter()
                .filter(|requirement| requirement.id == id)
                .collect();
            let requirement = match matched.as_slice() {
                [] => return Err(ExpertBankCatalogError::ContractEntryMissing { id }),
                [requirement] => *requirement,
                _ => {
                    return Err(ExpertBankCatalogError::ContractEntryAmbiguous {
                        id,
                        count: matched.len(),
                    });
                }
            };
            if requirement.match_mode != TensorMatch::OneOf {
                return Err(ExpertBankCatalogError::ContractEntryAmbiguous {
                    id,
                    count: requirement.names.len(),
                });
            }
            requirements.push(requirement.clone());
            // The bank's declared auxiliaries travel with it, so a plane the artifact
            // carries is bound and reported rather than left unclaimed and invisible.
            requirements.extend(
                contract
                    .requirements
                    .iter()
                    .filter(|aux| {
                        matches!(&aux.id, TensorId::QuantAux { tensor, .. } if **tensor == id)
                    })
                    .cloned(),
            );
            wanted.push((block, index, projection, id, expert_count));
        }
    }

    // Every census row carrying one of the accepted names is offered, so a duplicated
    // checkpoint name is the contract's `DuplicateCensusName`, never a silent first match.
    let names: BTreeSet<&str> = requirements
        .iter()
        .flat_map(|requirement| requirement.names.iter().map(String::as_str))
        .collect();
    let rows: Vec<TensorCensusEntry> = census
        .iter()
        .filter(|entry| names.contains(entry.name.as_str()))
        .cloned()
        .collect();
    let bound = TensorContract {
        dialect: contract.dialect,
        requirements,
    }
    .bind(&rows)
    .map_err(|error| ExpertBankCatalogError::Contract(Box::new(error)))?;

    let mut planes: Vec<String> = bound
        .tensors
        .iter()
        .filter(|(id, _)| matches!(id, TensorId::QuantAux { .. }))
        .flat_map(|(_, tensor)| tensor.checkpoint_names.iter().cloned())
        .collect();
    if !planes.is_empty() {
        planes.sort();
        return Err(ExpertBankCatalogError::ScalePlanes { names: planes });
    }

    let mut projections = Vec::with_capacity(wanted.len());
    for (block, checkpoint_layer, projection, id, expert_count) in wanted {
        let tensor = bound
            .tensors
            .get(&id)
            .ok_or_else(|| ExpertBankCatalogError::ContractEntryMissing { id: id.clone() })?;
        let (name, shape, storage) = match (
            tensor.checkpoint_names.as_slice(),
            tensor.shapes.as_slice(),
            tensor.storage.as_slice(),
        ) {
            ([name], [shape], [storage]) => (name.clone(), shape.clone(), storage.clone()),
            _ => {
                return Err(ExpertBankCatalogError::ContractEntryAmbiguous {
                    id,
                    count: tensor.checkpoint_names.len(),
                });
            }
        };
        if shape.last().copied() != Some(u64::from(expert_count)) {
            return Err(ExpertBankCatalogError::ExpertCountMismatch {
                id,
                plan: expert_count,
                shape,
            });
        }
        projections.push(ExpertBankProjection {
            block,
            checkpoint_layer,
            projection,
            id,
            name,
            shape,
            storage,
            expert_count,
            physical_bytes: tensor.physical_bytes,
        });
    }
    Ok(ExpertBankCatalog { projections })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{HfConfig, ModelConfig};
    use crate::tensor_contract::{
        ContractOptions, FloatType, QuantConstraint, QuantLayout, TensorRequirement,
    };

    fn moe_plan(trunk_layers: u32, nextn: u32) -> ModelPlan {
        let cfg = ModelConfig::from_hf(&HfConfig::parse(&format!(
            r#"{{"model_type":"qwen3_5_moe","num_hidden_layers":{trunk_layers},
            "num_nextn_predict_layers":{nextn},"hidden_size":64,
            "num_attention_heads":2,"num_key_value_heads":1,"head_dim":32,
            "intermediate_size":128,"vocab_size":16,"max_position_embeddings":128,
            "full_attention_interval":2,"linear_conv_kernel_dim":4,
            "linear_key_head_dim":32,"linear_value_head_dim":32,
            "linear_num_key_heads":2,"linear_num_value_heads":4,
            "num_experts":4,"num_experts_per_tok":2,"moe_intermediate_size":32,
            "shared_expert_intermediate_size":48}}"#
        )));
        ModelPlan::compile(&cfg).unwrap()
    }

    fn dense_plan() -> ModelPlan {
        let cfg = ModelConfig::from_hf(&HfConfig::parse(
            r#"{"model_type":"qwen3","num_hidden_layers":1,"hidden_size":64,
            "num_attention_heads":2,"num_key_value_heads":1,"head_dim":32,
            "intermediate_size":128,"vocab_size":16,"max_position_embeddings":128}"#,
        ));
        ModelPlan::compile(&cfg).unwrap()
    }

    fn gguf_contract(plan: &ModelPlan) -> TensorContract {
        TensorContract::for_plan(plan, CheckpointDialect::Gguf, ContractOptions::default()).unwrap()
    }

    fn quant(format: &str, block: u32) -> StorageLayout {
        StorageLayout::Quantized(QuantLayout {
            format: format.to_string(),
            block_shape: vec![block],
            auxiliaries: Vec::new(),
        })
    }

    /// A census with exactly the required rows of `contract`, quantized weights as IQ4_XS.
    fn census_for(contract: &TensorContract) -> Vec<TensorCensusEntry> {
        contract
            .requirements
            .iter()
            .filter(|requirement| requirement.required)
            .flat_map(|requirement| {
                let names = match requirement.match_mode {
                    TensorMatch::OneOf => &requirement.names[..1],
                    TensorMatch::All => requirement.names.as_slice(),
                };
                names.iter().map(|name| TensorCensusEntry {
                    name: name.clone(),
                    shape: requirement.shape.clone(),
                    storage: if requirement.quant == QuantConstraint::FloatOnly {
                        StorageLayout::Float(FloatType::F32)
                    } else {
                        quant("IQ4_XS", 256)
                    },
                    physical_bytes: requirement.shape.iter().product::<u64>() / 2,
                })
            })
            .collect()
    }

    /// The day-ten literal spelling the installer used to hard-code: trunk banks by loaded
    /// position, then the first MTP block by its plan layer index.
    fn literal_day10_names(plan: &ModelPlan) -> Vec<String> {
        let mut names = Vec::new();
        for (position, layer) in plan.layers.iter().enumerate() {
            if matches!(layer.mlp, MlpPlan::Moe(_)) {
                for projection in ["gate", "up", "down"] {
                    names.push(format!("blk.{position}.ffn_{projection}_exps.weight"));
                }
            }
        }
        if let Some(block) = plan.mtp_blocks.first()
            && matches!(block.layer.mlp, MlpPlan::Moe(_))
        {
            let index = block.layer.index;
            for projection in ["gate", "up", "down"] {
                names.push(format!("blk.{index}.ffn_{projection}_exps.weight"));
            }
        }
        names
    }

    #[test]
    fn plan_derived_catalog_matches_the_literal_day10_spelling() {
        let plan = moe_plan(2, 1);
        let contract = gguf_contract(&plan);
        let census = census_for(&contract);
        let catalog = expert_bank_catalog(&plan, &contract, &census).unwrap();
        let names: Vec<&str> = catalog
            .projections
            .iter()
            .map(|projection| projection.name.as_str())
            .collect();
        assert_eq!(names, literal_day10_names(&plan));
        assert_eq!(catalog.projections.len(), 9);
        assert_eq!(catalog.blocks().count(), 3);
        for block in catalog.blocks() {
            assert_eq!(block.len(), PROJECTIONS_PER_BLOCK);
            assert!(block.iter().all(|p| p.block == block[0].block));
            assert_eq!(
                block.iter().map(|p| p.projection).collect::<Vec<_>>(),
                [ExpertTensor::Gate, ExpertTensor::Up, ExpertTensor::Down]
            );
        }
        let first = &catalog.projections[0];
        assert_eq!(first.block, ExpertBankBlock::Trunk { position: 0 });
        assert_eq!((first.checkpoint_layer, first.expert_count), (0, 4));
        assert_eq!(first.shape, vec![64, 32, 4]);
        assert_eq!(first.storage, quant("IQ4_XS", 256));
        let down = &catalog.projections[2];
        assert_eq!(down.shape, vec![32, 64, 4]);
        let mtp = &catalog.projections[6];
        assert_eq!(mtp.block, ExpertBankBlock::Mtp { depth: 0 });
        assert_eq!(mtp.checkpoint_layer, 2);
        assert_eq!(mtp.name, "blk.2.ffn_gate_exps.weight");
        // The identity text is one line per projection and names every checkpoint tensor.
        let identity = catalog.identity();
        assert_eq!(identity.lines().count(), 9);
        assert!(
            identity.starts_with(
                "trunk:0\t0\tGate\tblk.0.ffn_gate_exps.weight\t64x32x4\tIQ4_XS[256]\t"
            )
        );
        assert!(identity.contains("\nmtp:0\t2\tGate\tblk.2.ffn_gate_exps.weight\t"));
    }

    #[test]
    fn a_dense_plan_has_no_expert_projections() {
        let plan = dense_plan();
        let contract = gguf_contract(&plan);
        let census = census_for(&contract);
        assert_eq!(
            expert_bank_catalog(&plan, &contract, &census),
            Err(ExpertBankCatalogError::NoMoeExpertProjections)
        );
    }

    #[test]
    fn a_missing_expert_tensor_is_the_contract_missing_verdict() {
        let plan = moe_plan(2, 0);
        let contract = gguf_contract(&plan);
        let census: Vec<_> = census_for(&contract)
            .into_iter()
            .filter(|entry| entry.name != "blk.1.ffn_up_exps.weight")
            .collect();
        let id = TensorId::Layer {
            index: 1,
            tensor: LayerTensor::MoeExpertUpBank,
        };
        assert_eq!(
            expert_bank_catalog(&plan, &contract, &census),
            Err(ExpertBankCatalogError::Contract(Box::new(
                TensorContractError::Missing {
                    id,
                    accepted_names: vec!["blk.1.ffn_up_exps.weight".to_string()],
                }
            )))
        );
    }

    #[test]
    fn a_duplicated_expert_tensor_is_ambiguous() {
        let plan = moe_plan(1, 0);
        let contract = gguf_contract(&plan);
        let mut census = census_for(&contract);
        let duplicate = census
            .iter()
            .find(|entry| entry.name == "blk.0.ffn_down_exps.weight")
            .unwrap()
            .clone();
        census.push(duplicate);
        assert_eq!(
            expert_bank_catalog(&plan, &contract, &census),
            Err(ExpertBankCatalogError::Contract(Box::new(
                TensorContractError::DuplicateCensusName {
                    name: "blk.0.ffn_down_exps.weight".to_string(),
                }
            )))
        );
    }

    #[test]
    fn a_shape_incompatible_expert_tensor_is_refused() {
        let plan = moe_plan(1, 0);
        let contract = gguf_contract(&plan);
        let mut census = census_for(&contract);
        let entry = census
            .iter_mut()
            .find(|entry| entry.name == "blk.0.ffn_gate_exps.weight")
            .unwrap();
        entry.shape = vec![64, 32, 8];
        assert_eq!(
            expert_bank_catalog(&plan, &contract, &census),
            Err(ExpertBankCatalogError::Contract(Box::new(
                TensorContractError::ShapeMismatch {
                    id: TensorId::Layer {
                        index: 0,
                        tensor: LayerTensor::MoeExpertGateBank,
                    },
                    name: "blk.0.ffn_gate_exps.weight".to_string(),
                    expected: vec![64, 32, 4],
                    actual: vec![64, 32, 8],
                }
            )))
        );
    }

    #[test]
    fn a_scale_plane_on_a_bank_is_refused_by_name() {
        let plan = moe_plan(1, 0);
        let contract = gguf_contract(&plan);
        let mut census = census_for(&contract);
        for name in ["blk.0.ffn_up_exps.scale", "blk.0.ffn_gate_exps.input_scale"] {
            census.push(TensorCensusEntry {
                name: name.to_string(),
                shape: vec![1],
                storage: StorageLayout::Float(FloatType::F32),
                physical_bytes: 4,
            });
        }
        assert_eq!(
            expert_bank_catalog(&plan, &contract, &census),
            Err(ExpertBankCatalogError::ScalePlanes {
                names: vec![
                    "blk.0.ffn_gate_exps.input_scale".to_string(),
                    "blk.0.ffn_up_exps.scale".to_string(),
                ],
            })
        );
    }

    #[test]
    fn a_contract_without_the_bank_entry_is_refused() {
        let plan = moe_plan(1, 0);
        let mut contract = gguf_contract(&plan);
        let census = census_for(&contract);
        let id = TensorId::Layer {
            index: 0,
            tensor: LayerTensor::MoeExpertDownBank,
        };
        contract
            .requirements
            .retain(|requirement| requirement.id != id);
        assert_eq!(
            expert_bank_catalog(&plan, &contract, &census),
            Err(ExpertBankCatalogError::ContractEntryMissing { id })
        );
    }

    #[test]
    fn a_contract_with_two_entries_for_one_bank_is_ambiguous() {
        let plan = moe_plan(1, 0);
        let mut contract = gguf_contract(&plan);
        let census = census_for(&contract);
        let id = TensorId::Layer {
            index: 0,
            tensor: LayerTensor::MoeExpertGateBank,
        };
        let twin: TensorRequirement = contract
            .requirements
            .iter()
            .find(|requirement| requirement.id == id)
            .unwrap()
            .clone();
        contract.requirements.push(twin);
        assert_eq!(
            expert_bank_catalog(&plan, &contract, &census),
            Err(ExpertBankCatalogError::ContractEntryAmbiguous { id, count: 2 })
        );
    }

    #[test]
    fn a_non_gguf_contract_is_refused() {
        let plan = moe_plan(1, 0);
        let contract = TensorContract::for_plan(
            &plan,
            CheckpointDialect::HfSafetensors,
            ContractOptions::default(),
        )
        .unwrap();
        assert_eq!(
            expert_bank_catalog(&plan, &contract, &[]),
            Err(ExpertBankCatalogError::Dialect(
                CheckpointDialect::HfSafetensors
            ))
        );
    }

    #[test]
    fn errors_read_as_one_sentence() {
        assert_eq!(
            ExpertBankCatalogError::NoMoeExpertProjections.to_string(),
            "the compiled plan has no MoE expert projections"
        );
        assert_eq!(
            ExpertBankCatalogError::ScalePlanes {
                names: vec!["a".into(), "b".into()]
            }
            .to_string(),
            "artifact carries expert scale planes the consumer does not declare: a, b"
        );
    }
}
