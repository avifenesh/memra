//! CUDA-free source for #544's native mixed KDA/indexed-MLA allocation gate.
//! The source is derived from the canonical plan/contract and Memra reference fixture.
use memra_gguf::GgmlType;
use memra_gguf::config::{HfConfig, ModelConfig};
use memra_gguf::model_plan::ModelPlan;
use memra_gguf::source::{TensorSource, TensorView};
use memra_gguf::tensor_contract::{
    CheckpointDialect, ContractOptions, FloatType, OutputHead, StorageLayout, TensorCensusEntry,
    TensorContract, TensorMatch,
};
use sha2::{Digest, Sha256};
use std::borrow::Cow;
use std::collections::BTreeMap;

pub(crate) const CONFIG: &str = include_str!("model_memory_fixture.json");
pub(crate) const CAPACITY: usize = 8192;
pub(crate) const KDA_LAYERS: [usize; 2] = [0, 2];
pub(crate) const MLA_LAYERS: [usize; 2] = [1, 3];

struct Tensor {
    bytes: Vec<u8>,
    shape: Vec<u64>,
}

pub(crate) struct FixtureSource {
    config: ModelConfig,
    pub(crate) plan: ModelPlan,
    tensors: BTreeMap<String, Tensor>,
}

impl FixtureSource {
    pub(crate) fn new() -> Self {
        let config = ModelConfig::from_hf(&HfConfig::parse(CONFIG));
        let plan = memra_gguf::model_packs::for_config(&config)
            .expect("GLM5-next fixture pack")
            .compile_plan(&config)
            .expect("GLM5-next fixture plan");
        let reference = memra_reference::deterministic_fixture(&plan).expect("reference tensors");
        let contract = TensorContract::for_plan(
            &plan,
            CheckpointDialect::Gguf,
            ContractOptions {
                output_head: OutputHead::Separate,
            },
        )
        .expect("fixture tensor contract");
        let mut tensors = BTreeMap::new();
        for req in contract
            .requirements
            .iter()
            .filter(|req| req.required || reference.weights.contains_key(&req.id))
        {
            // memra#541: glm5_next declares a separate head; the fixture serves the embedding
            // rows under `output.weight` (the reference reads the same numbers either way).
            let tensor = reference
                .weights
                .get(&req.id)
                .or_else(|| {
                    (req.id == memra_gguf::tensor_contract::TensorId::OutputProjection)
                        .then(|| {
                            reference
                                .weights
                                .get(&memra_gguf::tensor_contract::TensorId::TokenEmbedding)
                        })
                        .flatten()
                })
                .expect("required reference tensor");
            assert!(tensor.ints.is_none(), "this fixture uses F32 tensors only");
            let bytes: Vec<u8> = tensor
                .data
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect();
            let names = match req.match_mode {
                TensorMatch::OneOf => &req.names[..1],
                TensorMatch::All => req.names.as_slice(),
            };
            for name in names {
                assert!(
                    tensors
                        .insert(
                            name.clone(),
                            Tensor {
                                bytes: bytes.clone(),
                                shape: req.shape.clone(),
                            }
                        )
                        .is_none(),
                    "duplicate fixture tensor {name}"
                );
            }
        }
        let source = Self {
            config,
            plan,
            tensors,
        };
        contract
            .bind(&source.census())
            .expect("complete, shape-exact fixture census");
        source
    }

    fn census(&self) -> Vec<TensorCensusEntry> {
        self.tensors
            .iter()
            .map(|(name, tensor)| TensorCensusEntry {
                auxiliaries: Vec::new(),
                name: name.clone(),
                shape: tensor.shape.clone(),
                storage: StorageLayout::Float(FloatType::F32),
                physical_bytes: tensor.bytes.len() as u64,
            })
            .collect()
    }

    /// Versioned, delimited identity of config + sorted tensor names, shapes and exact bytes.
    pub(crate) fn identity(&self) -> (usize, usize, String) {
        let mut hash = Sha256::new();
        hash.update(b"memra-544-indexed-kda-fixture-v1\0");
        hash.update((CONFIG.len() as u64).to_le_bytes());
        hash.update(CONFIG.as_bytes());
        let mut bytes = 0;
        for (name, tensor) in &self.tensors {
            hash.update((name.len() as u64).to_le_bytes());
            hash.update(name.as_bytes());
            hash.update((tensor.shape.len() as u64).to_le_bytes());
            for dim in &tensor.shape {
                hash.update(dim.to_le_bytes());
            }
            hash.update((tensor.bytes.len() as u64).to_le_bytes());
            hash.update(&tensor.bytes);
            bytes += tensor.bytes.len();
        }
        (self.tensors.len(), bytes, format!("{:x}", hash.finalize()))
    }
}

impl TensorSource for FixtureSource {
    fn config(&self) -> ModelConfig {
        self.config.clone()
    }
    fn tensor_census(&self) -> Result<memra_gguf::source::TensorCensus, String> {
        Ok(memra_gguf::source::TensorCensus {
            dialect: memra_gguf::tensor_contract::CheckpointDialect::Gguf,
            tensors: self
                .census()
                .into_iter()
                .map(|entry| memra_gguf::source::TensorCensusRecord {
                    auxiliaries: Vec::new(),
                    physical_name: entry.name.clone(),
                    dtype: "F32".to_string(),
                    entry,
                })
                .collect(),
        })
    }
    fn find(&self, name: &str) -> Option<TensorView<'_>> {
        let tensor = self.tensors.get(name)?;
        Some(TensorView {
            bytes: Cow::Borrowed(&tensor.bytes),
            ggml_type: GgmlType::F32,
            ne: tensor.shape.clone(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use memra_gguf::model_plan::{AttentionPlan, MlaAttentionPlan, SparseIndexPlan, StatePlan};

    #[test]
    fn fixture_has_two_tp2_kda_and_two_independent_kpool_layers() {
        let source = FixtureSource::new();
        assert_eq!(source.plan.layers.len(), 4);
        assert!(source.plan.mtp_blocks.is_empty());
        for il in KDA_LAYERS {
            let AttentionPlan::KimiDeltaNet(kda) = source.plan.layers[il].attention else {
                panic!("layer {il} must be KDA")
            };
            assert_eq!((kda.num_heads, kda.head_dim, kda.conv_kernel), (2, 128, 4));
        }
        for il in MLA_LAYERS {
            let AttentionPlan::Mla(MlaAttentionPlan::LatentKv { sparse_index, .. }) =
                &source.plan.layers[il].attention
            else {
                panic!("layer {il} must be MLA")
            };
            assert!(
                matches!(
                    sparse_index,
                    SparseIndexPlan::Own {
                        heads: 2,
                        head_dim: 8,
                        kpool: Some(_),
                        ..
                    }
                ),
                "layer {il} must own a nonzero k-pool indexer: {sparse_index:?}"
            );
            assert!(
                source
                    .find(&format!("blk.{il}.indexer.kpool_ape.weight"))
                    .is_some()
            );
        }
        assert_eq!(
            source
                .plan
                .layers
                .iter()
                .map(|layer| &layer.state)
                .filter(|state| matches!(state, StatePlan::Recurrent { .. }))
                .count(),
            2
        );
        assert_eq!(
            source
                .plan
                .layers
                .iter()
                .map(|layer| &layer.state)
                .filter(|state| matches!(state, StatePlan::LatentKvCache { .. }))
                .count(),
            2
        );
        assert_eq!(CAPACITY, 8192);
    }

    #[test]
    fn fixture_census_rejects_a_missing_indexer_tensor_and_identity_is_repeatable() {
        let source = FixtureSource::new();
        let contract = TensorContract::for_plan(
            &source.plan,
            CheckpointDialect::Gguf,
            ContractOptions {
                output_head: OutputHead::Separate,
            },
        )
        .unwrap();
        let mut census = source.census();
        let removed = census
            .iter()
            .position(|entry| entry.name == "blk.1.indexer.kpool_ape.weight")
            .expect("first indexed layer must have pool APE");
        census.remove(removed);
        assert!(
            contract.bind(&census).is_err(),
            "a non-indexed fixture must fail instead of producing partial_key_bytes=0"
        );
        let id = source.identity();
        eprintln!(
            "[memory-fixture-cpu] tensors={} bytes={} sha256={}",
            id.0, id.1, id.2
        );
        assert!(id.0 > 0 && id.1 > 0);
        assert_eq!(id, FixtureSource::new().identity());
    }
}
