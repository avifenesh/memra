//! Opt-in target/draft source receipts. These do not identify final uploads or admit rewrites.
use super::draft::composite::{CompositeDraftSourceIdentity, CompositeExternalDraftInput};
use super::draft::{DraftVocabulary, ExternalDraftBlock, PreparedExternalDraftSource};
use super::head_trim::{HeadChoice, HeadTrimIdentity, PreparedHeadTrim, TrimPolicy};
use super::ranks::RankArtifact;
use super::{BoundArtifactIdentity, BoundProgramRef, ModelConfig, PreparedModelSource, TensorId};
use crate::GgufFile;
use crate::source::{GgufSource, TensorSource};
use crate::tensor_contract::MtpTensor;
use sha2::{Digest, Sha256};

/// Complete external artifact identity remains separate from its selected program descriptor.
#[derive(Debug, Clone, PartialEq)]
pub struct ExternalDraftIdentity {
    artifact: BoundArtifactIdentity,
    components: Option<CompositeDraftSourceIdentity>,
    head_role: TensorId,
    head_name: String,
    norm_name: Option<String>,
    blocks: Vec<ExternalDraftBlock>,
    vocabulary: DraftVocabulary,
}
impl ExternalDraftIdentity {
    pub fn artifact(&self) -> &BoundArtifactIdentity {
        &self.artifact
    }
    pub fn components(&self) -> Option<&CompositeDraftSourceIdentity> {
        self.components.as_ref()
    }
    pub fn head_role(&self) -> &TensorId {
        &self.head_role
    }
    pub fn head_name(&self) -> &str {
        &self.head_name
    }
    pub fn norm_name(&self) -> Option<&str> {
        self.norm_name.as_deref()
    }
    /// The existing external consumer executes only the first declared NextN block.
    pub fn selected_block(&self) -> &ExternalDraftBlock {
        &self.blocks[0]
    }
    pub fn declared_blocks(&self) -> &[ExternalDraftBlock] {
        &self.blocks
    }
    pub fn vocabulary(&self) -> &DraftVocabulary {
        &self.vocabulary
    }
}

/// The role cannot be inferred from equal shapes, equal bytes or a target-only receipt.
#[derive(Debug, Clone, PartialEq)]
pub enum DraftProposalIdentity {
    External(Box<ExternalDraftIdentity>),
    TargetHeadTrim(Box<HeadTrimIdentity>),
}

/// Sealed source-stage composition. Whole opened artifacts, selected semantic bindings and
/// immutable host materialization/rank receipts are distinct fields. No upload/binary identity.
#[derive(Debug, Clone, PartialEq)]
pub struct DraftSourcePairIdentity {
    target: BoundArtifactIdentity,
    proposal: DraftProposalIdentity,
    source_pair_sha256: String,
}
impl DraftSourcePairIdentity {
    pub fn target(&self) -> &BoundArtifactIdentity {
        &self.target
    }
    pub fn proposal(&self) -> &DraftProposalIdentity {
        &self.proposal
    }
    pub fn source_pair_sha256(&self) -> &str {
        &self.source_pair_sha256
    }
    fn new(target: BoundArtifactIdentity, proposal: DraftProposalIdentity) -> Self {
        let mut hash = Sha256::new();
        hash.update(b"memra-draft-source-pair-v1\0");
        for (tag, value) in [
            ("target", format!("{target:?}")),
            ("proposal", format!("{proposal:?}")),
        ] {
            hash.update((tag.len() as u64).to_le_bytes());
            hash.update(tag.as_bytes());
            hash.update((value.len() as u64).to_le_bytes());
            hash.update(value.as_bytes());
        }
        Self {
            target,
            proposal,
            source_pair_sha256: format!("{:x}", hash.finalize()),
        }
    }
}

/// The receipt and actual immutable payload are produced together; callers cannot attach a
/// trim made from another source merely because its semantic binding happens to match.
pub struct PreparedPairedHeadTrim {
    materialization: PreparedHeadTrim,
    identity: DraftSourcePairIdentity,
}
impl PreparedPairedHeadTrim {
    pub fn materialization(&self) -> &PreparedHeadTrim {
        &self.materialization
    }
    pub fn identity(&self) -> &DraftSourcePairIdentity {
        &self.identity
    }
}

/// Explicit source-proof operation. Full opened-source hashes are checked before and after
/// preparation/consumption, so detected drift refuses. This is not an atomic writer lock.
/// Existing loaders do not pay this cost unless they use the paired entry points.
pub struct PreparedDraftTarget<'a> {
    source: &'a dyn TensorSource,
    config: ModelConfig,
    identity: BoundArtifactIdentity,
}
impl<'a> PreparedDraftTarget<'a> {
    pub fn bind(source: &'a dyn TensorSource) -> Result<Self, String> {
        let identity = target_identity(source)?;
        // Validate the actual sealed text-root scope; never compile a caller's raw target here.
        PreparedModelSource::text(source)?;
        let (config, _) = source.bound_program().unwrap().cloned_pair();
        Ok(Self {
            source,
            config,
            identity,
        })
    }
    pub fn identity(&self) -> &BoundArtifactIdentity {
        &self.identity
    }
    pub fn config(&self) -> &ModelConfig {
        &self.config
    }
    fn validate(&self) -> Result<(), String> {
        if target_identity(self.source)? != self.identity {
            return Err("paired target opened source changed since capture".into());
        }
        Ok(())
    }
    pub fn with_external<T>(
        &self,
        draft: &GgufFile,
        load: impl FnOnce(&PreparedExternalDraftSource<'_>, &dyn TensorSource) -> T,
    ) -> Result<(T, DraftSourcePairIdentity), String> {
        self.validate()?;
        let source = GgufSource(draft);
        let opened = source.artifact_sha256()?;
        let prepared = PreparedExternalDraftSource::compile(&source, &self.config)?;
        prepared.with_runtime(|runtime| {
            self.consume_external(&prepared, runtime, &opened, None, load)
        })?
    }
    pub fn with_composite_external<T>(
        &self,
        draft: &CompositeExternalDraftInput,
        load: impl FnOnce(&PreparedExternalDraftSource<'_>, &dyn TensorSource) -> T,
    ) -> Result<(T, DraftSourcePairIdentity), String> {
        self.validate()?;
        let components = draft.source_identity()?;
        let opened = components.opened_source_sha256().to_owned();
        draft.with_runtime(&self.config, |prepared, runtime| {
            self.consume_external(prepared, runtime, &opened, Some(components), load)
        })?
    }
    fn consume_external<T>(
        &self,
        prepared: &PreparedExternalDraftSource<'_>,
        runtime: &dyn TensorSource,
        opened: &str,
        components: Option<CompositeDraftSourceIdentity>,
        load: impl FnOnce(&PreparedExternalDraftSource<'_>, &dyn TensorSource) -> T,
    ) -> Result<(T, DraftSourcePairIdentity), String> {
        let artifact = prepared.artifact_identity()?;
        if artifact.opened_source_sha256 != opened {
            return Err("paired external draft opened source changed during preparation".into());
        }
        let first = &prepared.blocks()[0].layer;
        let head_name = prepared.head_name();
        let head_role = if head_name == "output.weight" {
            TensorId::OutputProjection
        } else if [
            format!("blk.{}.nextn.shared_head_head.weight", first.layer.index),
            format!("blk.{}.nextn.shared_head.weight", first.layer.index),
        ]
        .iter()
        .any(|name| name == head_name)
        {
            TensorId::Mtp {
                depth: first.depth,
                tensor: MtpTensor::OutputProjection,
            }
        } else {
            return Err("paired external draft has an unknown selected head role".into());
        };
        let external = ExternalDraftIdentity {
            artifact: artifact.clone(),
            components,
            head_role,
            head_name: head_name.to_owned(),
            norm_name: prepared.norm_name().map(str::to_owned),
            blocks: prepared.blocks().to_vec(),
            vocabulary: prepared.vocabulary().clone(),
        };
        let result = load(prepared, runtime);
        if prepared.artifact_identity()? != artifact {
            return Err("paired external draft opened source changed during consumption".into());
        }
        self.validate()?;
        Ok((
            result,
            DraftSourcePairIdentity::new(
                self.identity.clone(),
                DraftProposalIdentity::External(Box::new(external)),
            ),
        ))
    }
    pub fn prepare_head_trim(
        &self,
        ranks: &RankArtifact,
        choice: HeadChoice,
        policy: TrimPolicy,
    ) -> Result<Option<PreparedPairedHeadTrim>, String> {
        self.validate()?;
        let materialization = PreparedHeadTrim::prepare(self.source, ranks, choice, policy)?;
        self.validate()?;
        Ok(materialization.map(|materialization| {
            let identity = DraftSourcePairIdentity::new(
                self.identity.clone(),
                DraftProposalIdentity::TargetHeadTrim(Box::new(materialization.identity().clone())),
            );
            PreparedPairedHeadTrim {
                materialization,
                identity,
            }
        }))
    }
}

fn target_identity(source: &dyn TensorSource) -> Result<BoundArtifactIdentity, String> {
    match source
        .bound_program()
        .ok_or("paired identity requires a compiler-bound target source")?
    {
        BoundProgramRef::Single(bound) => bound.artifact_identity(),
        BoundProgramRef::Composite(bound) => bound.artifact_identity(),
        BoundProgramRef::ExternalDraft(_) => {
            Err("external draft cannot supply paired target authority".into())
        }
    }
}
