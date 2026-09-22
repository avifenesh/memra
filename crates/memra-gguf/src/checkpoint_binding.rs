//! The loader-side tensor-contract boundary (memra#541).
//!
//! `memra model inspect` has always bound the canonical [`TensorContract`] against a
//! checkpoint's census and refused missing, unexpected, ambiguous, wrong-shape and wrong-quant
//! tensors. The engine's loaders did not: they compiled a plan and then read string-named
//! tensors one by one, so the strict inspector's verdict was never a loading guarantee and the
//! ggml name map could drift from the contract without anything failing. This module is the
//! shared boundary both loaders (and the inspector, the auto-parallel planner and the expert
//! catalog) now go through:
//!
//! - [`bind_source`] takes the metadata-only census of a [`TensorSource`], decides output-head
//!   ownership from the pack's declaration ([`OutputHeadContract`]) plus the HF
//!   `tie_word_embeddings` key, compiles the pack's contract and binds it BEFORE any byte is
//!   uploaded. Every contract error is returned with the pack, dialect and census size in
//!   front of it.
//! - [`CheckpointBinding`] answers "which ggml name does semantic id X resolve to" from the
//!   contract itself (the GGUF-dialect spelling of the same contract), so a loader can address
//!   tensors by [`TensorId`] instead of a second hand-written name table.
//! - [`RecordingSource`] wraps a source and records every tensor the loader actually read;
//!   [`CheckpointBinding::audit_consumption`] then names the bound tensors that were never
//!   consumed, the silent-drop class the name-map drift produced (`docs/ONBOARDING.md`,
//!   "The name map is a SECOND surface"). The pack's [`TensorConsumption`] decides whether that
//!   is a report or a refusal.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, Mutex};

use memmap2::Mmap;

use crate::GgufFile;
use crate::config::ModelConfig;
use crate::hf_mapping::{HfTarget, resolve_ggml};
use crate::model_packs::{ModelPack, OutputHeadContract, TensorConsumption, for_config};
use crate::model_plan::ModelPlan;
use crate::source::{
    DiskExtent, ExpertActivationPrecision, Fp8Native, Fp8StackedNative, Nvfp4Native,
    Nvfp4StackedNative, TensorCensus, TensorSource, TensorView, canonical_hf_name,
};
use crate::tensor_contract::{
    BoundTensor, BoundTensorContract, CheckpointDialect, ContractOptions, OutputHead,
    TensorContract, TensorContractError, TensorId, TensorOwner,
};

/// A checkpoint whose census the compiled contract accepted, plus the facts a loader needs.
#[derive(Clone)]
pub struct CheckpointBinding {
    /// The pack that compiled the contract, or `None` when the architecture has no pack and the
    /// canonical plan contract was used.
    pub pack: Option<&'static ModelPack>,
    pub dialect: CheckpointDialect,
    pub output_head: OutputHead,
    pub contract: TensorContract,
    pub bound: BoundTensorContract,
    /// GGUF-dialect spelling of every semantic id: the engine's request vocabulary. `None` when
    /// the pack cannot spell its contract in the GGUF dialect (safetensors-only families).
    ggml_names: Option<BTreeMap<TensorId, Vec<String>>>,
}

#[derive(Debug)]
pub enum CheckpointBindError {
    NoCensus {
        family: &'static str,
        error: String,
    },
    OutputHead {
        family: &'static str,
        dialect: CheckpointDialect,
        reason: String,
    },
    Contract {
        family: &'static str,
        dialect: CheckpointDialect,
        error: Box<TensorContractError>,
    },
    Bind {
        family: &'static str,
        dialect: CheckpointDialect,
        tensors: usize,
        error: Box<TensorContractError>,
    },
}

impl std::fmt::Display for CheckpointBindError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoCensus { family, error } => write!(
                f,
                "checkpoint refused before upload (pack {family}): the tensor source exposes no \
                 metadata census, so its contract cannot be bound: {error}"
            ),
            Self::OutputHead {
                family,
                dialect,
                reason,
            } => write!(
                f,
                "checkpoint refused before upload (pack {family}, {dialect:?}): output head \
                 ownership: {reason}"
            ),
            Self::Contract {
                family,
                dialect,
                error,
            } => write!(
                f,
                "checkpoint refused before upload (pack {family}, {dialect:?}): the tensor \
                 contract does not compile for this plan: {error}"
            ),
            Self::Bind {
                family,
                dialect,
                tensors,
                error,
            } => write!(
                f,
                "checkpoint refused before upload (pack {family}, {dialect:?}, {tensors} census \
                 tensors): {error}"
            ),
        }
    }
}

impl std::error::Error for CheckpointBindError {}

const HEAD_NAMES: [&str; 2] = ["output.weight", "lm_head.weight"];

fn family_of(pack: Option<&'static ModelPack>) -> &'static str {
    pack.map(|pack| pack.family).unwrap_or("canonical")
}

/// Output-head ownership for a checkpoint, from the pack's declaration and the census.
///
/// A present head tensor is always the head (the bytes exist; a tied HF export that also ships
/// `lm_head.weight` is served from it). An absent head is the token embedding ONLY when the
/// family declares that ownership ([`OutputHeadContract::TiedHeadAllowed`]) and the HF config,
/// when it speaks, does not contradict it; a `SeparateHead` family or a checkpoint whose
/// config.json says `tie_word_embeddings: false` refuses instead of silently reading the
/// embedding as a head.
pub fn output_head_for(
    pack: Option<&'static ModelPack>,
    cfg: &ModelConfig,
    census: &TensorCensus,
) -> Result<OutputHead, CheckpointBindError> {
    let family = family_of(pack);
    let head_present = census
        .tensors
        .iter()
        .any(|row| HEAD_NAMES.contains(&row.entry.name.as_str()));
    if head_present {
        return Ok(OutputHead::Separate);
    }
    let policy = pack
        .map(|pack| pack.output_head)
        .unwrap_or(OutputHeadContract::TiedHeadAllowed);
    let declared = match census.dialect {
        CheckpointDialect::Gguf => None,
        CheckpointDialect::HfSafetensors => cfg.tie_word_embeddings,
    };
    match (policy, declared) {
        (OutputHeadContract::SeparateHead, _) => Err(CheckpointBindError::OutputHead {
            family,
            dialect: census.dialect,
            reason: format!(
                "the checkpoint carries none of {HEAD_NAMES:?} and pack {family} declares a \
                 separate output projection; a tied head is not a shape this family ships, so \
                 the embedding is not substituted"
            ),
        }),
        (OutputHeadContract::TiedHeadAllowed, Some(false)) => {
            Err(CheckpointBindError::OutputHead {
                family,
                dialect: census.dialect,
                reason: format!(
                    "config.json declares tie_word_embeddings: false but the checkpoint carries \
                     none of {HEAD_NAMES:?}; the embedding is not substituted for a head the \
                     config says is separate"
                ),
            })
        }
        (OutputHeadContract::TiedHeadAllowed, _) => Ok(OutputHead::TiedToEmbedding),
    }
}

#[allow(clippy::result_large_err)] // allow: the fat error type is the diagnostic contract here; boxing it would change the error surface
fn compile_contract(
    pack: Option<&'static ModelPack>,
    cfg: &ModelConfig,
    plan: &ModelPlan,
    dialect: CheckpointDialect,
    options: ContractOptions,
) -> Result<TensorContract, TensorContractError> {
    match pack {
        Some(pack) => pack.compile_tensor_contract(cfg, plan, dialect, options),
        None => TensorContract::for_plan(plan, dialect, options),
    }
}

/// Bind the checkpoint behind `src` against the contract its pack compiles for `plan`.
pub fn bind_source(
    src: &dyn TensorSource,
    cfg: &ModelConfig,
    plan: &ModelPlan,
) -> Result<CheckpointBinding, CheckpointBindError> {
    let pack = for_config(cfg);
    let family = family_of(pack);
    let census = src
        .tensor_census()
        .map_err(|error| CheckpointBindError::NoCensus { family, error })?;
    bind_census(pack, cfg, plan, &census)
}

/// [`bind_source`] on an already-read census (the inspector and the tests hold one).
pub fn bind_census(
    pack: Option<&'static ModelPack>,
    cfg: &ModelConfig,
    plan: &ModelPlan,
    census: &TensorCensus,
) -> Result<CheckpointBinding, CheckpointBindError> {
    let family = family_of(pack);
    let dialect = census.dialect;
    let output_head = output_head_for(pack, cfg, census)?;
    let options = ContractOptions { output_head };
    let contract = compile_contract(pack, cfg, plan, dialect, options).map_err(|error| {
        CheckpointBindError::Contract {
            family,
            dialect,
            error: Box::new(error),
        }
    })?;
    let entries: Vec<_> = census.tensors.iter().map(|row| row.entry.clone()).collect();
    let bound = contract
        .bind(&entries)
        .map_err(|error| CheckpointBindError::Bind {
            family,
            dialect,
            tensors: entries.len(),
            error: Box::new(error),
        })?;
    let ggml_names = match dialect {
        CheckpointDialect::Gguf => Some(
            bound
                .tensors
                .iter()
                .map(|(id, tensor)| (id.clone(), tensor.checkpoint_names.clone()))
                .collect(),
        ),
        CheckpointDialect::HfSafetensors => {
            compile_contract(pack, cfg, plan, CheckpointDialect::Gguf, options)
                .ok()
                .map(|gguf| {
                    gguf.requirements
                        .into_iter()
                        .map(|requirement| (requirement.id, requirement.names))
                        .collect()
                })
        }
    };
    Ok(CheckpointBinding {
        pack,
        dialect,
        output_head,
        contract,
        bound,
        ggml_names,
    })
}

impl std::fmt::Debug for CheckpointBinding {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CheckpointBinding")
            .field("pack", &self.family())
            .field("dialect", &self.dialect)
            .field("output_head", &self.output_head)
            .field("bound", &self.bound.tensors.len())
            .finish()
    }
}

/// A bound tensor the loader never read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnconsumedTensor {
    pub id: TensorId,
    pub checkpoint_names: Vec<String>,
}

impl CheckpointBinding {
    pub fn family(&self) -> &'static str {
        family_of(self.pack)
    }

    pub fn consumption_policy(&self) -> TensorConsumption {
        self.pack
            .map(|pack| pack.tensor_consumption)
            .unwrap_or(TensorConsumption::Report)
    }

    /// Whether the contract bound this semantic id at all (optional ids may be absent).
    pub fn has(&self, id: &TensorId) -> bool {
        self.bound.tensors.contains_key(id)
    }

    pub fn tensor(&self, id: &TensorId) -> Option<&BoundTensor> {
        self.bound.tensors.get(id)
    }

    /// The ggml-dialect name the engine requests for a bound semantic id, or `None` when the id is
    /// not bound or the pack has no GGUF spelling.
    pub fn ggml_name(&self, id: &TensorId) -> Option<&str> {
        if !self.has(id) {
            return None;
        }
        self.ggml_names
            .as_ref()?
            .get(id)
            .and_then(|names| names.first())
            .map(String::as_str)
    }

    /// [`Self::ggml_name`] for an id the plan requires; the error names the id and the pack.
    pub fn require_ggml(&self, id: &TensorId) -> Result<String, String> {
        self.ggml_name(id).map(str::to_string).ok_or_else(|| {
            if self.has(id) {
                format!(
                    "pack {} binds {id:?} but has no GGUF-dialect spelling for it; the loader \
                     cannot address it by semantic id",
                    self.family()
                )
            } else {
                format!(
                    "pack {} contract binds no {id:?}; the plan requires it and the loader does \
                     not substitute another tensor",
                    self.family()
                )
            }
        })
    }

    /// The ggml name of the output projection: the bound `OutputProjection`, or the token
    /// embedding when the binding decided the head is tied.
    pub fn output_head_ggml_name(&self) -> Result<String, String> {
        match self.output_head {
            OutputHead::Separate => self.require_ggml(&TensorId::OutputProjection),
            OutputHead::TiedToEmbedding => self.require_ggml(&TensorId::TokenEmbedding),
        }
    }

    /// Bound tensors none of whose checkpoint names were read through the recorded source.
    /// `skip` excludes ids the loader legitimately leaves to another owner (vision towers, an MTP
    /// block the caller asked not to load).
    pub fn audit_consumption(
        &self,
        requested_ggml: &BTreeSet<String>,
        cfg: &ModelConfig,
        skip: impl Fn(&TensorId, &BoundTensor) -> bool,
    ) -> Vec<UnconsumedTensor> {
        let requested_physical: BTreeSet<String> = requested_ggml
            .iter()
            .flat_map(|name| self.physical_names_for_request(name, cfg))
            .collect();
        self.bound
            .tensors
            .iter()
            .filter(|(id, tensor)| !skip(id, tensor))
            .filter(|(_, tensor)| {
                !tensor
                    .checkpoint_names
                    .iter()
                    .any(|name| requested_physical.contains(name))
            })
            .map(|(id, tensor)| UnconsumedTensor {
                id: id.clone(),
                checkpoint_names: tensor.checkpoint_names.clone(),
            })
            .collect()
    }

    /// Census names a ggml request touches. GGUF: the name itself, plus the stacked bank behind a
    /// per-expert slice request. HF: whatever the engine's name map resolves it to, canonicalized
    /// the way the census is; `.scale`-style auxiliary siblings fold into their owner's census
    /// row and are covered by the owner's request.
    fn physical_names_for_request(&self, ggml: &str, cfg: &ModelConfig) -> Vec<String> {
        match self.dialect {
            CheckpointDialect::Gguf => {
                let mut names = vec![ggml.to_string()];
                if let Some(bank) = stacked_bank_of_expert_slice(ggml) {
                    names.push(bank);
                }
                names
            }
            CheckpointDialect::HfSafetensors => match resolve_ggml(ggml, cfg) {
                Some(HfTarget::Plain(hf)) | Some(HfTarget::Transform { hf, .. }) => {
                    vec![canonical_hf_name(&hf)]
                }
                None => Vec::new(),
            },
        }
    }
}

/// `blk.N.ffn_{gate,up,down}_exps.{e}.weight` is a per-expert slice of the stacked GGUF bank
/// `blk.N.ffn_{gate,up,down}_exps.weight`.
fn stacked_bank_of_expert_slice(ggml: &str) -> Option<String> {
    let rest = ggml.strip_prefix("blk.")?;
    let (il, suffix) = rest.split_once('.')?;
    for tag in ["ffn_gate_exps.", "ffn_up_exps.", "ffn_down_exps."] {
        if let Some(e_part) = suffix.strip_prefix(tag)
            && let Some(e) = e_part.strip_suffix(".weight")
            && e.parse::<u32>().is_ok()
        {
            return Some(format!("blk.{il}.{tag}weight"));
        }
    }
    None
}

/// Auxiliary planes the checkpoint carries and no memra program reads. modelopt's static
/// activation `input_scale` planes are declared by the contract (a checkpoint is refused if they
/// are malformed) but the engine's activation programs never consume them: W4A16 quantizes
/// activations dynamically and the calibrated A4 program carries its own `.a4_input_scale`
/// values (`model_packs::qwen35::activation::SCALE_SUFFIX`, "deliberately not `.input_scale`").
/// Receipt: the 9B NVFP4 GGUF reports exactly its 113 `.input_scale` rows and nothing else
/// (`research/loader-census-20260922/`).
pub fn unread_by_design(id: &TensorId) -> bool {
    matches!(
        id,
        TensorId::QuantAux {
            kind: crate::tensor_contract::QuantAuxTensor::InputScale,
            ..
        }
    )
}

/// Whether a bound tensor belongs to an owner the trunk loader does not read itself.
pub fn owned_by_vision(tensor: &BoundTensor) -> bool {
    matches!(tensor.owner, TensorOwner::Vision(_))
}

/// Whether a bound tensor belongs to an MTP block: the glue tensors carry `TensorOwner::Mtp`,
/// but the block's own layer tensors keep their appended layer index (`blk.<n_trunk+d>.*` in a
/// GGUF), so anything owned by a layer at or past the trunk is the draft head's too. Receipt:
/// `decode-batch-gate` loads without MTP and the first Refuse run named exactly the eleven
/// `blk.32.*` tensors of the 9B's appended block.
pub fn owned_by_mtp(tensor: &BoundTensor, n_trunk_layers: u32) -> bool {
    match tensor.owner {
        TensorOwner::Mtp(_) => true,
        TensorOwner::Layer(index) => index >= n_trunk_layers,
        _ => false,
    }
}

/// A [`TensorSource`] that records the ggml name of every tensor read through it. Probes
/// (`has`) are not reads; the metadata census is not a read.
pub struct RecordingSource<'a> {
    inner: &'a dyn TensorSource,
    requested: Mutex<BTreeSet<String>>,
}

impl<'a> RecordingSource<'a> {
    pub fn new(inner: &'a dyn TensorSource) -> Self {
        Self {
            inner,
            requested: Mutex::new(BTreeSet::new()),
        }
    }

    fn record(&self, name: &str) {
        self.requested
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .insert(name.to_string());
    }

    /// Every ggml name read so far.
    pub fn requested(&self) -> BTreeSet<String> {
        self.requested
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }
}

impl TensorSource for RecordingSource<'_> {
    fn config(&self) -> ModelConfig {
        self.inner.config()
    }
    fn try_config(&self) -> Result<ModelConfig, String> {
        self.inner.try_config()
    }
    fn tensor_census(&self) -> Result<TensorCensus, String> {
        self.inner.tensor_census()
    }
    fn expert_activation_precision(&self) -> ExpertActivationPrecision {
        self.inner.expert_activation_precision()
    }
    fn find(&self, ggml_name: &str) -> Option<TensorView<'_>> {
        self.record(ggml_name);
        self.inner.find(ggml_name)
    }
    fn has(&self, ggml_name: &str) -> bool {
        self.inner.has(ggml_name)
    }
    fn gguf(&self) -> Option<&GgufFile> {
        self.inner.gguf()
    }
    fn find_nvfp4_native(&self, ggml_name: &str) -> Option<Nvfp4Native<'_>> {
        self.record(ggml_name);
        self.inner.find_nvfp4_native(ggml_name)
    }
    fn find_fp8_native(&self, ggml_name: &str) -> Option<Fp8Native<'_>> {
        self.record(ggml_name);
        self.inner.find_fp8_native(ggml_name)
    }
    fn find_fp8_stacked_native(&self, ggml_name: &str) -> Option<Fp8StackedNative<'_>> {
        self.record(ggml_name);
        self.inner.find_fp8_stacked_native(ggml_name)
    }
    fn find_nvfp4_stacked_native(&self, ggml_name: &str) -> Option<Nvfp4StackedNative<'_>> {
        self.record(ggml_name);
        self.inner.find_nvfp4_stacked_native(ggml_name)
    }
    fn st_dir(&self) -> Option<&std::path::Path> {
        self.inner.st_dir()
    }
    fn nvfp4_cache_tag(&self) -> &'static str {
        self.inner.nvfp4_cache_tag()
    }
    fn preserve_expert_encodings(&self) -> bool {
        self.inner.preserve_expert_encodings()
    }
    fn active_experts(&self, layer: u32) -> Option<&[bool]> {
        self.inner.active_experts(layer)
    }
    fn find_expert_disk(&self, ggml_name: &str) -> Option<DiskExtent> {
        self.record(ggml_name);
        self.inner.find_expert_disk(ggml_name)
    }
    fn find_expert_mmap(&self, ggml_name: &str) -> Option<(Arc<Mmap>, usize, usize)> {
        self.record(ggml_name);
        self.inner.find_expert_mmap(ggml_name)
    }
}

/// One line for the load log: what was bound and under which head ownership.
pub fn describe(binding: &CheckpointBinding) -> String {
    format!(
        "[tensor-contract] bound {} semantic tensors (pack {}, {:?}, output head {:?}); every \
         census tensor is claimed and every required tensor is present with its declared shape \
         and storage",
        binding.bound.tensors.len(),
        binding.family(),
        binding.dialect,
        binding.output_head,
    )
}

/// Apply the pack's [`TensorConsumption`] policy to an audit result.
pub fn settle_consumption(
    binding: &CheckpointBinding,
    unconsumed: &[UnconsumedTensor],
) -> Result<(), String> {
    settle_with(
        binding.consumption_policy(),
        binding.family(),
        binding.dialect,
        unconsumed,
    )
}

fn settle_with(
    policy: TensorConsumption,
    family: &str,
    dialect: CheckpointDialect,
    unconsumed: &[UnconsumedTensor],
) -> Result<(), String> {
    if unconsumed.is_empty() {
        return Ok(());
    }
    let text = describe_unconsumed(unconsumed);
    match policy {
        TensorConsumption::Refuse => Err(format!(
            "checkpoint refused after load (pack {family}, {dialect:?}): {} bound tensor(s) were \
             never read by the loader, so their semantics would be silently dropped: {text}",
            unconsumed.len()
        )),
        TensorConsumption::Report => {
            eprintln!(
                "[tensor-contract] {} bound tensor(s) were not read by the loader (pack {family} \
                 reports, does not refuse): {text}",
                unconsumed.len()
            );
            Ok(())
        }
    }
}

/// Format an unconsumed-tensor list for a log line or an error.
pub fn describe_unconsumed(unconsumed: &[UnconsumedTensor]) -> String {
    let shown: Vec<String> = unconsumed
        .iter()
        .take(12)
        .map(|tensor| format!("{:?} <- {:?}", tensor.id, tensor.checkpoint_names))
        .collect();
    let more = unconsumed.len().saturating_sub(shown.len());
    if more > 0 {
        format!("{} (+{more} more)", shown.join("; "))
    } else {
        shown.join("; ")
    }
}

#[cfg(test)]
mod tests {
    //! CPU teeth for the loader boundary: the real GGUF census path over the glm-dsa micro
    //! fixture, byte-tampered copies of it, the head-ownership matrix, and the consumption audit.
    use super::*;
    use crate::GgufFile;
    use crate::micro_gguf::write_glm_dsa_micro;
    use crate::model_packs::by_alias;
    use crate::source::{GgufSource, TensorCensusRecord, census_from_gguf};
    use crate::tensor_contract::{FloatType, StorageLayout, TensorCensusEntry};

    fn micro_path(tag: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "memra-541-{tag}-{}-{}.gguf",
            std::process::id(),
            std::thread::current()
                .name()
                .unwrap_or("t")
                .replace("::", "-")
        ))
    }

    /// Same-length rename inside the GGUF header: offsets stay valid, the census changes.
    fn tampered_copy(src: &std::path::Path, from: &str, to: &str, tag: &str) -> std::path::PathBuf {
        assert_eq!(from.len(), to.len());
        let mut bytes = std::fs::read(src).unwrap();
        let needle = from.as_bytes();
        let at = bytes
            .windows(needle.len())
            .position(|w| w == needle)
            .expect("tensor name present in header");
        bytes[at..at + needle.len()].copy_from_slice(to.as_bytes());
        let out = micro_path(tag);
        std::fs::write(&out, bytes).unwrap();
        out
    }

    fn plan_for(cfg: &ModelConfig) -> ModelPlan {
        for_config(cfg)
            .expect("glm_dsa pack")
            .compile_plan(cfg)
            .unwrap()
    }

    #[test]
    fn clean_micro_fixture_binds_with_a_separate_head_and_ggml_spellings() {
        let p = micro_path("clean");
        write_glm_dsa_micro(&p, 0x6_10AD_0802).unwrap();
        let g = GgufFile::open(&p).unwrap();
        let src = GgufSource(&g);
        let cfg = src.config();
        let plan = plan_for(&cfg);
        let binding = bind_source(&src, &cfg, &plan).unwrap();
        std::fs::remove_file(&p).ok();
        assert_eq!(binding.family(), "glm_dsa");
        assert_eq!(binding.dialect, CheckpointDialect::Gguf);
        assert_eq!(binding.output_head, OutputHead::Separate);
        assert_eq!(
            binding.ggml_name(&TensorId::TokenEmbedding),
            Some("token_embd.weight")
        );
        assert_eq!(binding.output_head_ggml_name().unwrap(), "output.weight");
        assert_eq!(
            binding.require_ggml(&TensorId::Layer {
                index: 0,
                tensor: crate::tensor_contract::LayerTensor::PreAttentionNorm,
            }),
            Ok("blk.0.attn_norm.weight".to_string())
        );
        assert!(binding.bound.tensors.len() > 20);
        assert!(describe(&binding).contains("pack glm_dsa"));
    }

    #[test]
    fn renamed_trunk_tensor_is_refused_before_upload_with_the_pack_named() {
        let p = micro_path("rename-src");
        write_glm_dsa_micro(&p, 0x6_10AD_0802).unwrap();
        let t = tampered_copy(
            &p,
            "blk.0.attn_norm.weight",
            "blk.0.attn_nOrm.weight",
            "rename",
        );
        std::fs::remove_file(&p).ok();
        let g = GgufFile::open(&t).unwrap();
        let src = GgufSource(&g);
        let cfg = src.config();
        let plan = plan_for(&cfg);
        let err = bind_source(&src, &cfg, &plan).unwrap_err().to_string();
        std::fs::remove_file(&t).ok();
        assert!(
            err.starts_with("checkpoint refused before upload (pack glm_dsa, Gguf"),
            "{err}"
        );
        assert!(err.contains("missing tensor"), "{err}");
        assert!(err.contains("blk.0.attn_norm.weight"), "{err}");
    }

    #[test]
    fn absent_head_under_a_separate_head_pack_is_refused_not_substituted() {
        let p = micro_path("head-src");
        write_glm_dsa_micro(&p, 0x6_10AD_0802).unwrap();
        let t = tampered_copy(&p, "output.weight", "outpuT.weight", "head");
        std::fs::remove_file(&p).ok();
        let g = GgufFile::open(&t).unwrap();
        let src = GgufSource(&g);
        let cfg = src.config();
        let plan = plan_for(&cfg);
        let err = bind_source(&src, &cfg, &plan).unwrap_err().to_string();
        std::fs::remove_file(&t).ok();
        assert!(err.contains("output head ownership"), "{err}");
        assert!(
            err.contains("declares a separate output projection"),
            "{err}"
        );
        assert!(err.contains("the embedding is not substituted"), "{err}");
    }

    fn census(dialect: CheckpointDialect, names: &[&str]) -> TensorCensus {
        TensorCensus {
            dialect,
            tensors: names
                .iter()
                .map(|name| TensorCensusRecord {
                    physical_name: name.to_string(),
                    dtype: "F32".into(),
                    entry: TensorCensusEntry {
                        name: name.to_string(),
                        shape: vec![1],
                        storage: StorageLayout::Float(FloatType::F32),
                        physical_bytes: 4,
                    },
                })
                .collect(),
        }
    }

    #[test]
    fn head_ownership_matrix() {
        let tied_ok = by_alias("qwen35").unwrap();
        let separate = by_alias("glm_dsa").unwrap();
        let mut cfg = ModelConfig::from_hf(&crate::config::HfConfig::default());
        // a present head is the head for every policy
        for pack in [tied_ok, separate] {
            let c = census(
                CheckpointDialect::Gguf,
                &["token_embd.weight", "output.weight"],
            );
            assert_eq!(
                output_head_for(Some(pack), &cfg, &c).unwrap(),
                OutputHead::Separate
            );
            let c = census(CheckpointDialect::HfSafetensors, &["lm_head.weight"]);
            assert_eq!(
                output_head_for(Some(pack), &cfg, &c).unwrap(),
                OutputHead::Separate
            );
        }
        // GGUF, absent head: the pack decides
        let c = census(CheckpointDialect::Gguf, &["token_embd.weight"]);
        assert_eq!(
            output_head_for(Some(tied_ok), &cfg, &c).unwrap(),
            OutputHead::TiedToEmbedding
        );
        assert!(output_head_for(Some(separate), &cfg, &c).is_err());
        // no pack: the historical convention (tied when absent)
        assert_eq!(
            output_head_for(None, &cfg, &c).unwrap(),
            OutputHead::TiedToEmbedding
        );
        // HF, absent head: the config's declaration is honored
        let c = census(
            CheckpointDialect::HfSafetensors,
            &["model.embed_tokens.weight"],
        );
        cfg.tie_word_embeddings = Some(true);
        assert_eq!(
            output_head_for(Some(tied_ok), &cfg, &c).unwrap(),
            OutputHead::TiedToEmbedding
        );
        cfg.tie_word_embeddings = None;
        assert_eq!(
            output_head_for(Some(tied_ok), &cfg, &c).unwrap(),
            OutputHead::TiedToEmbedding
        );
        cfg.tie_word_embeddings = Some(false);
        let err = output_head_for(Some(tied_ok), &cfg, &c)
            .unwrap_err()
            .to_string();
        assert!(err.contains("tie_word_embeddings: false"), "{err}");
        // a SeparateHead pack refuses regardless of what the config says
        cfg.tie_word_embeddings = Some(true);
        assert!(output_head_for(Some(separate), &cfg, &c).is_err());
    }

    #[test]
    fn recording_source_records_reads_not_probes_and_the_audit_names_the_unread() {
        let p = micro_path("audit");
        write_glm_dsa_micro(&p, 0x6_10AD_0802).unwrap();
        let g = GgufFile::open(&p).unwrap();
        let src = GgufSource(&g);
        let cfg = src.config();
        let plan = plan_for(&cfg);
        let binding = bind_source(&src, &cfg, &plan).unwrap();
        let recording = RecordingSource::new(&src);
        assert!(recording.has("output.weight"));
        assert!(recording.requested().is_empty(), "a probe is not a read");
        // read everything except the output norm
        for row in census_from_gguf(&g).tensors {
            if row.entry.name != "output_norm.weight" {
                assert!(recording.find(&row.entry.name).is_some());
            }
        }
        std::fs::remove_file(&p).ok();
        let unconsumed = binding.audit_consumption(&recording.requested(), &cfg, |_, _| false);
        assert_eq!(unconsumed.len(), 1, "{unconsumed:?}");
        assert_eq!(unconsumed[0].id, TensorId::OutputNorm);
        assert_eq!(
            unconsumed[0].checkpoint_names,
            vec!["output_norm.weight".to_string()]
        );
        // the skip predicate excludes an owner the trunk loader leaves to someone else
        let none = binding.audit_consumption(&recording.requested(), &cfg, |id, _| {
            *id == TensorId::OutputNorm
        });
        assert!(none.is_empty());
        // policies
        assert!(
            settle_with(
                TensorConsumption::Report,
                "glm_dsa",
                binding.dialect,
                &unconsumed
            )
            .is_ok()
        );
        let err = settle_with(
            TensorConsumption::Refuse,
            "glm_dsa",
            binding.dialect,
            &unconsumed,
        )
        .unwrap_err();
        assert!(err.contains("refused after load (pack glm_dsa"), "{err}");
        assert!(err.contains("OutputNorm"), "{err}");
        assert!(settle_with(TensorConsumption::Refuse, "x", binding.dialect, &[]).is_ok());
    }

    #[test]
    fn per_expert_slice_requests_cover_the_stacked_gguf_bank() {
        assert_eq!(
            stacked_bank_of_expert_slice("blk.3.ffn_gate_exps.17.weight"),
            Some("blk.3.ffn_gate_exps.weight".to_string())
        );
        assert_eq!(
            stacked_bank_of_expert_slice("blk.3.ffn_gate_exps.weight"),
            None
        );
        assert_eq!(stacked_bank_of_expert_slice("blk.3.attn_q.weight"), None);
    }

    #[test]
    fn a_source_without_a_census_is_refused_with_the_reason() {
        struct NoCensus(ModelConfig);
        impl TensorSource for NoCensus {
            fn config(&self) -> ModelConfig {
                self.0.clone()
            }
            fn find(&self, _: &str) -> Option<TensorView<'_>> {
                None
            }
        }
        let p = micro_path("nocensus");
        write_glm_dsa_micro(&p, 0x6_10AD_0802).unwrap();
        let g = GgufFile::open(&p).unwrap();
        let cfg = GgufSource(&g).config();
        std::fs::remove_file(&p).ok();
        let plan = plan_for(&cfg);
        let err = bind_source(&NoCensus(cfg.clone()), &cfg, &plan)
            .unwrap_err()
            .to_string();
        assert!(err.contains("exposes no metadata census"), "{err}");
    }
}
