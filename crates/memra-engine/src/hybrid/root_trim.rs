//! Strict target-trim loading. Incomplete uploads stay privately owned until the actual
//! constructed model's load barrier; only then are those exact tensors moved into its slots.
use super::{HybridModel, MtpHead};
use crate::model::final_upload::{FinalUploadIdentity, UploadedHeadTrim};
use memra_gguf::{
    GgmlType,
    bound_source::{
        BoundArtifactIdentity,
        draft_pair::{DraftSourcePairIdentity, PreparedDraftTarget, PreparedPairedHeadTrim},
        head_trim::{HeadChoice, TrimPolicy},
        ranks::RankArtifact,
    },
    execution_manifest::{ExecutionRewrite, RewriteSurface, execution_rewrites},
    model_plan::{DraftSourcePlan, MlpPlan, ModelPlan},
    source::TensorSource,
};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, ffi::OsString, sync::Arc};

// Same external sources refused by the ordinary path, except the rank input which is
// consumed by this typed preparation. A source receipt alone never enables the exception.
const OTHER_EXTERNAL: [&str; 5] = [
    "MEMRA_MTP_DRAFT",
    "MEMRA_DRAFT",
    "MEMRA_SPEC_DFLASH",
    "MEMRA_DSPARK_DRAFT",
    "MEMRA_GLM5_DFLASH",
];
const OPTIONS: [&str; 10] = [
    "MEMRA_MTP_DRAFT",
    "MEMRA_DRAFT",
    "MEMRA_SPEC_DFLASH",
    "MEMRA_DSPARK_DRAFT",
    "MEMRA_GLM5_DFLASH",
    "MEMRA_FRSPEC_TRIM",
    "MEMRA_MTP_SKIP",
    "MEMRA_MTP_HEADS",
    "MEMRA_FULL_PREC",
    "MEMRA_FRSPEC_TRIM_NVFP4",
];
fn options() -> BTreeMap<&'static str, Option<OsString>> {
    OPTIONS
        .into_iter()
        .map(|key| (key, std::env::var_os(key)))
        .collect()
}
fn refuse_other_external() -> Result<(), String> {
    for key in OTHER_EXTERNAL {
        if std::env::var_os(key).is_some_and(|v| !v.is_empty()) {
            return Err(format!(
                "paired target trim does not admit external source {key}"
            ));
        }
    }
    Ok(())
}
#[derive(Debug, Clone, PartialEq, Eq)]
struct SelectedPrograms {
    eager: ExecutionRewrite,
    spec: ExecutionRewrite,
}
fn selected_programs(plan: &ModelPlan) -> Result<SelectedPrograms, String> {
    if plan.draft_source != DraftSourcePlan::Embedded || plan.mtp_blocks.is_empty() {
        return Err("paired target trim requires actual embedded MTP blocks".into());
    }
    // This is the shared compiler's canonical selection, including its family-specific
    // Eager implementation. Never reproduce that selector or assume a generic rewrite ID.
    let rewrites = execution_rewrites(plan);
    let select = |surface| -> Result<ExecutionRewrite, String> {
        let rewrite = rewrites
            .iter()
            .find(|r| r.surface == surface)
            .ok_or("paired target manifest is absent")?;
        if !rewrite.eligible() {
            return Err(format!(
                "paired target {:?} is ineligible: {:?}",
                surface, rewrite.blockers
            ));
        }
        Ok(rewrite.clone())
    };
    Ok(SelectedPrograms {
        eager: select(RewriteSurface::DecodeEager)?,
        spec: select(RewriteSurface::MtpSpec)?,
    })
}

fn digest(fields: impl IntoIterator<Item = String>) -> String {
    let mut h = Sha256::new();
    for field in fields {
        h.update((field.len() as u64).to_le_bytes());
        h.update(field.as_bytes());
    }
    format!("{:x}", h.finalize())
}

pub(super) struct StageInfo {
    pub(super) name: String,
    pub(super) dtype: GgmlType,
    pub(super) from_model_output: bool,
    pub(super) sizes: Option<(usize, usize)>,
}
#[derive(Clone)]
pub(super) struct SlotReservation(Arc<()>);
struct PendingSlot {
    reservation: SlotReservation,
    upload: UploadedHeadTrim,
    ids: Vec<u32>,
    from_model_output: bool,
}
#[derive(Debug)]
struct InstalledSlot {
    index: usize,
    block_index: u32,
    source: DraftSourcePairIdentity,
    upload: FinalUploadIdentity,
    submitted_identity: String,
    ids: Vec<u32>,
    from_model_output: bool,
}

impl InstalledSlot {
    fn portable(&self) -> String {
        format!(
            "{:?}",
            (
                self.index,
                self.block_index,
                &self.source,
                &self.upload,
                &self.submitted_identity,
                &self.ids,
                self.from_model_output
            )
        )
    }
}

pub(super) struct RootTrimLoad<'a> {
    source: &'a dyn TensorSource,
    target: PreparedDraftTarget<'a>,
    config_debug: String,
    programs: SelectedPrograms,
    options: BTreeMap<&'static str, Option<OsString>>,
    ranks: RankArtifact,
    rank_sha16: String,
    declared: usize,
    requested: usize,
    selected: usize,
    n_trunk: usize,
    prepared: Vec<Option<PreparedPairedHeadTrim>>,
    slots: Vec<PendingSlot>,
    loaded_count_seen: bool,
    truncation_seen: bool,
}
impl<'a> RootTrimLoad<'a> {
    pub(super) fn begin(
        source: &'a dyn TensorSource,
        load_mtp: bool,
        requested: bool,
        rank_input: &mut crate::trim_ranks::RankInput,
    ) -> Result<Option<Self>, String> {
        if !requested {
            return Ok(None);
        }
        let initial_options = options();
        let Some(spec) = initial_options["MEMRA_FRSPEC_TRIM"]
            .clone()
            .filter(|v| !v.is_empty())
        else {
            return Ok(None);
        };
        for key in OTHER_EXTERNAL {
            if initial_options[key].as_ref().is_some_and(|v| !v.is_empty()) {
                return Err(format!(
                    "paired target trim does not admit external source {key}"
                ));
            }
        }
        let option = |key| initial_options[key].as_ref().and_then(|v| v.to_str());
        if !load_mtp || crate::model::full_prec_enabled() || option("MEMRA_FULL_PREC") == Some("1")
        {
            return Err("paired target trim requires enabled MTP and the actual trim program, not a full-precision/no-MTP fallback".into());
        }
        match option("MEMRA_MTP_SKIP") {
            None | Some("") | Some("0") => {}
            _ => return Err("paired target trim does not admit an MTP skip/stub".into()),
        }
        let spec = spec
            .into_string()
            .map_err(|_| "paired rank spec must be valid UTF-8")?;
        let target = PreparedDraftTarget::bind(source)?;
        let (_, plan) =
            memra_gguf::model_packs::compile_for_source(source).map_err(|e| e.to_string())?;
        let programs = selected_programs(&plan)?;
        let cfg = target.config();
        let declared = plan.mtp_blocks.len();
        let requested = option("MEMRA_MTP_HEADS")
            .and_then(|v| v.parse::<u32>().ok())
            .filter(|&v| v > 0)
            .map_or(declared, |cap| declared.min(cap as usize));
        let n_trunk = usize::try_from(
            cfg.n_layer
                .checked_sub(cfg.nextn_predict_layers)
                .ok_or("paired target trunk underflow")?,
        )
        .map_err(|_| "paired trunk exceeds usize")?;
        if n_trunk == 0 || declared != cfg.nextn_predict_layers as usize {
            return Err("paired target block geometry is inconsistent".into());
        }
        let ranks = rank_input.capture(&spec)?.artifact().clone();
        let rank_sha16 = ranks.identity().source_sha256()[..16].to_owned();
        let policy = if option("MEMRA_FRSPEC_TRIM_NVFP4") == Some("1") {
            TrimPolicy::Nvfp4ForEligibleBf16
        } else {
            TrimPolicy::Preserve
        };
        let mut prepared = Vec::new();
        for slot in 0..requested {
            let choice = if slot == 0 {
                HeadChoice::FirstMtpOrModel
            } else {
                HeadChoice::MtpBlock {
                    index: u32::try_from(n_trunk + slot)
                        .map_err(|_| "paired MTP index overflows")?,
                }
            };
            match target.prepare_head_trim(&ranks, choice, policy)? {
                Some(trim) => prepared.push(Some(trim)),
                None if slot > 0 => break,
                None => return Err("paired target first head is missing".into()),
            }
        }
        let selected = prepared.len();
        // Preserve the existing root's multi-head program restriction, including caps and
        // missing-head truncation. Do not silently widen it while adding identity.
        if selected > 1
            && (selected != declared
                || plan
                    .mtp_blocks
                    .iter()
                    .any(|b| !matches!(b.layer.mlp, MlpPlan::Dense(_))))
        {
            return Err(
                "multi-head MTP requires embedded dense canonical blocks and matching loaded heads"
                    .into(),
            );
        }
        if options() != initial_options {
            return Err("paired target trim options changed during preparation".into());
        }
        Ok(Some(Self {
            config_debug: format!("{cfg:?}"),
            programs,
            options: initial_options,
            source,
            target,
            ranks,
            rank_sha16,
            declared,
            requested,
            selected,
            n_trunk,
            prepared,
            slots: Vec::new(),
            loaded_count_seen: false,
            truncation_seen: false,
        }))
    }
    pub(super) fn note_loaded_heads(&mut self, count: usize) -> Result<(), String> {
        if self.loaded_count_seen || count != self.requested {
            return Err("paired target loaded MTP count differs from typed preparation".into());
        }
        self.loaded_count_seen = true;
        Ok(())
    }
    pub(super) fn stage(
        &mut self,
        e: &crate::Engine,
        slot: usize,
        head: &mut MtpHead,
    ) -> Result<Option<StageInfo>, Box<dyn std::error::Error>> {
        if !self.loaded_count_seen || slot != self.slots.len() || slot >= self.requested {
            return Err("paired target upload slot is missing, duplicated or reordered".into());
        }
        if head.embedded_block_index
            != Some(u32::try_from(self.n_trunk + slot).map_err(|_| "paired block index overflow")?)
            || head.pending_trim_slot.is_some()
        {
            return Err("paired target slot has reordered/substituted block provenance".into());
        }
        if head.external_source_identity.is_some() || head.geom.is_some() {
            return Err("paired target slot is external/student rather than embedded".into());
        }
        if slot == self.selected && self.selected < self.requested {
            if self.truncation_seen {
                return Err("paired target truncation was observed twice".into());
            }
            self.truncation_seen = true;
            return Ok(None);
        }
        let paired = self
            .prepared
            .get_mut(slot)
            .and_then(Option::take)
            .ok_or("paired target slot has no unconsumed materialization")?;
        let trim = paired.materialization();
        let info = StageInfo {
            name: trim.runtime_name().into(),
            dtype: trim.source_dtype(),
            from_model_output: trim.from_model_output(),
            sizes: trim.requant_sizes(),
        };
        let ids = trim.ids().to_vec();
        let upload = UploadedHeadTrim::load(e, &paired)?;
        let reservation = SlotReservation(Arc::new(()));
        head.pending_trim_slot = Some(reservation.clone());
        self.slots.push(PendingSlot {
            reservation,
            upload,
            ids,
            from_model_output: info.from_model_output,
        });
        // Drop the old full head at the same point as the ordinary replacement. The exact
        // uploaded replacement stays privately owned until post-barrier installation.
        head.shared_head_head = None;
        Ok(Some(info))
    }
    fn validate_model(
        &self,
        model: &HybridModel,
        artifact: &str,
        n_trunk: usize,
    ) -> Result<(), String> {
        refuse_other_external()?;
        if options() != self.options {
            return Err("paired target trim options changed during load".into());
        }
        if !self.loaded_count_seen
            || self.slots.len() != self.selected
            || self.prepared.iter().any(Option::is_some)
            || self.truncation_seen != (self.selected < self.requested)
        {
            return Err("paired target load proof is partial or has unaccounted truncation".into());
        }
        if n_trunk != self.n_trunk
            || artifact != self.target.identity().artifact_sha256
            || format!("{:?}", model.cfg) != self.config_debug
            || selected_programs(&model.plan)? != self.programs
        {
            return Err(
                "paired target constructed model/source differs from typed preparation".into(),
            );
        }
        if model.glm5_dflash.is_some()
            || model.dflash_trim.is_some()
            || model.frspec_src_sha16.as_deref() != Some(&self.rank_sha16)
        {
            return Err(
                "paired target model has an external/stub source or mismatched rank sentinel"
                    .into(),
            );
        }
        let heads = model
            .mtp
            .iter()
            .chain(model.mtp_extra.iter())
            .collect::<Vec<_>>();
        if model.mtp.is_none() || heads.len() != self.selected {
            return Err("paired target actual MTP chain has missing or extra slots".into());
        }
        let devices = model.devices();
        for (index, (head, slot)) in heads.iter().zip(&self.slots).enumerate() {
            if head.embedded_block_index != Some((self.n_trunk + index) as u32)
                || head
                    .pending_trim_slot
                    .as_ref()
                    .is_none_or(|r| !Arc::ptr_eq(&r.0, &slot.reservation.0))
                || head.shared_head_head.is_some()
                || head.external_source_identity.is_some()
                || head.geom.is_some()
                || head.d2t.as_deref() != Some(slot.ids.as_slice())
                || head.d2t_from_target_head != slot.from_model_output
                || slot.ids != self.ranks.ids()
                || !devices.contains(&slot.upload.upload().device())
            {
                return Err(
                    "paired target actual slot has a substituted tensor, mapping, origin or device"
                        .into(),
                );
            }
        }
        Ok(())
    }
    pub(super) fn complete(
        self,
        e: &crate::Engine,
        model: &mut HybridModel,
        artifact: &str,
        n_trunk: usize,
    ) -> Result<CompleteTargetTrimProof, Box<dyn std::error::Error>> {
        self.validate_model(model, artifact, n_trunk)?;
        crate::pp::sync_stages_after_load(e, n_trunk)?;
        if PreparedDraftTarget::bind(self.source)?.identity() != self.target.identity() {
            return Err("paired target opened source changed before completed load".into());
        }
        self.validate_model(model, artifact, n_trunk)?;
        let mut installed = Vec::with_capacity(self.selected);
        {
            let program = &mut *model.program;
            for (index, (head, slot)) in program
                .mtp
                .iter_mut()
                .chain(program.mtp_extra.iter_mut())
                .zip(self.slots)
                .enumerate()
            {
                head.pending_trim_slot = None;
                let (source, upload, submitted_identity) =
                    slot.upload.install_into(&mut head.shared_head_head)?;
                upload.validate_assigned(
                    head.shared_head_head
                        .as_ref()
                        .ok_or("paired head installation failed")?,
                )?;
                installed.push(InstalledSlot {
                    index,
                    block_index: (self.n_trunk + index) as u32,
                    source,
                    upload,
                    submitted_identity,
                    ids: slot.ids,
                    from_model_output: slot.from_model_output,
                });
            }
        }
        let target = self.target.identity().clone();
        let artifact_sha256 = digest([
            "memra-complete-target-trim-v1".into(),
            format!("{target:?}"),
            format!("{:?}", self.programs),
            format!(
                "{:?}",
                (
                    self.declared,
                    self.requested,
                    self.selected,
                    self.truncation_seen
                )
            ),
            format!(
                "{:?}",
                installed
                    .iter()
                    .map(InstalledSlot::portable)
                    .collect::<Vec<_>>()
            ),
        ]);
        Ok(CompleteTargetTrimProof {
            target,
            artifact_sha256,
            installed,
            config_debug: self.config_debug,
            programs: self.programs,
            rank_sha16: self.rank_sha16,
            options: self.options,
            generation: Arc::clone(&model.rewrite_generation),
            mutations: model.rewrite_mutations(),
        })
    }
}

/// Private, linear proof created only by post-barrier moves into the actual model. The
/// generation association is process-local authority and never enters portable identity.
pub(crate) struct CompleteTargetTrimProof {
    target: BoundArtifactIdentity,
    artifact_sha256: String,
    installed: Vec<InstalledSlot>,
    config_debug: String,
    programs: SelectedPrograms,
    rank_sha16: String,
    options: BTreeMap<&'static str, Option<OsString>>,
    generation: Arc<crate::plan_backend::ProgramGeneration>,
    mutations: u64,
}
impl CompleteTargetTrimProof {
    pub(crate) fn validate(
        &self,
        model: &HybridModel,
        source: &dyn TensorSource,
        artifact: &str,
    ) -> Result<(), String> {
        if !Arc::ptr_eq(&self.generation, &model.rewrite_generation)
            || model.rewrite_mutations() != self.mutations
        {
            return Err(
                "complete trim proof belongs to a different or mutated loaded model".into(),
            );
        }
        if options() != self.options
            || artifact != self.target.artifact_sha256
            || PreparedDraftTarget::bind(source)?.identity() != &self.target
        {
            return Err("complete trim proof target/source/options mismatch".into());
        }
        if format!("{:?}", model.cfg) != self.config_debug
            || selected_programs(&model.plan)? != self.programs
            || model.frspec_src_sha16.as_deref() != Some(&self.rank_sha16)
            || model.glm5_dflash.is_some()
            || model.dflash_trim.is_some()
        {
            return Err("complete trim proof no longer describes the loaded target".into());
        }
        let heads = model
            .mtp
            .iter()
            .chain(model.mtp_extra.iter())
            .collect::<Vec<_>>();
        if model.mtp.is_none() || heads.len() != self.installed.len() {
            return Err("complete trim proof loaded slot count mismatch".into());
        }
        for (index, (head, slot)) in heads.iter().zip(&self.installed).enumerate() {
            if index != slot.index
                || head.embedded_block_index != Some(slot.block_index)
                || head.pending_trim_slot.is_some()
                || head.external_source_identity.is_some()
                || head.geom.is_some()
                || head.d2t.as_deref() != Some(slot.ids.as_slice())
                || head.d2t_from_target_head != slot.from_model_output
            {
                return Err("complete trim proof loaded slot mapping/origin mismatch".into());
            }
            slot.upload.validate_assigned(
                head.shared_head_head
                    .as_ref()
                    .ok_or("complete trim proof tensor is absent")?,
            )?;
        }
        Ok(())
    }
    pub(crate) fn program_descriptor(&self) -> String {
        format!(
            "eager={}/{} spec={}/{}",
            self.programs.eager.id,
            self.programs.eager.implementation,
            self.programs.spec.id,
            self.programs.spec.implementation
        )
    }
    pub(crate) fn artifact_sha256(&self) -> &str {
        &self.artifact_sha256
    }
}
