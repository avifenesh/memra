//! Target-head selection and immutable host materialization for self-trim.
use super::ranks::{RankArtifact, RankIdentity};
use super::{BoundProgramRef, TensorId, TensorSource, TensorTransform};
use crate::tensor_contract::{CheckpointDialect, MtpTensor, OutputHead};
use crate::{GgmlType, source::TensorCensusRecord};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, sync::Arc};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HeadChoice {
    ModelOutput,
    FirstMtpOrModel,
    /// A declared block without a private projection returns None. An unknown block refuses.
    MtpBlock {
        index: u32,
    },
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrimPolicy {
    Preserve,
    Nvfp4ForEligibleBf16,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrimProgram {
    RowGather,
    Bf16RowsToNvfp4V1,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct HeadOrigin {
    id: TensorId,
    component_path: Vec<u32>,
    dialect: CheckpointDialect,
    record: TensorCensusRecord,
    transform: TensorTransform,
}

/// Evidence for the selected head's runtime view and produced payload. Whole-model opened
/// identity, binary identity and paired target/draft admission remain separate obligations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HeadTrimIdentity {
    source_binding_sha256: String,
    origin: HeadOrigin,
    rank: RankIdentity,
    policy: TrimPolicy,
    program: TrimProgram,
    source_dtype: GgmlType,
    source_shape: [u64; 2],
    source_head_sha256: String,
    macro_present: bool,
    source_macro_bits: u32,
    output_dtype: GgmlType,
    output_shape: [u64; 2],
    output_macro_bits: u32,
    output_sha256: String,
    materialization_sha256: String,
}
impl HeadTrimIdentity {
    pub fn materialization_sha256(&self) -> &str {
        &self.materialization_sha256
    }
    pub fn source_binding_sha256(&self) -> &str {
        &self.source_binding_sha256
    }
    pub fn source_head_sha256(&self) -> &str {
        &self.source_head_sha256
    }
    pub fn output_sha256(&self) -> &str {
        &self.output_sha256
    }
    pub fn rank(&self) -> &RankIdentity {
        &self.rank
    }
    pub fn semantic_head(&self) -> &TensorId {
        &self.origin.id
    }
    pub fn physical_name(&self) -> &str {
        &self.origin.record.physical_name
    }
    pub fn component_path(&self) -> &[u32] {
        &self.origin.component_path
    }
    pub fn source_macro_bits(&self) -> u32 {
        self.source_macro_bits
    }
    pub fn program(&self) -> TrimProgram {
        self.program
    }
}

/// Caller-assembled tensor views, names and scales cannot be substituted for this payload.
pub struct PreparedHeadTrim {
    bytes: Box<[u8]>,
    ids: Arc<[u32]>,
    runtime_name: String,
    identity: HeadTrimIdentity,
    gathered_bytes: usize,
}
impl PreparedHeadTrim {
    pub fn prepare(
        source: &dyn TensorSource,
        ranks: &RankArtifact,
        choice: HeadChoice,
        policy: TrimPolicy,
    ) -> Result<Option<Self>, String> {
        let bound = source
            .bound_program()
            .ok_or("self-trim requires a compiler-bound target source")?;
        if matches!(bound, BoundProgramRef::ExternalDraft(_)) {
            return Err("target self-trim cannot use external-draft source authority".into());
        }
        let (config, plan) = bound.cloned_pair();
        let model_id = if crate::model_packs::for_config(&config)
            .ok_or("bound model pack disappeared")?
            .contract_options(&config)
            .output_head
            == OutputHead::TiedToEmbedding
        {
            TensorId::TokenEmbedding
        } else {
            TensorId::OutputProjection
        };
        let model_name = if model_id == TensorId::TokenEmbedding {
            "token_embd.weight"
        } else {
            "output.weight"
        };
        let candidate = match choice {
            HeadChoice::ModelOutput => None,
            HeadChoice::FirstMtpOrModel => plan.mtp_blocks.first(),
            HeadChoice::MtpBlock { index } => Some(
                plan.mtp_blocks
                    .iter()
                    .find(|b| b.layer.index == index)
                    .ok_or_else(|| {
                        format!("self-trim MTP block {index} is outside the target plan")
                    })?,
            ),
        };
        let mut selected = None;
        if let Some(block) = candidate {
            let id = TensorId::Mtp {
                depth: block.depth,
                tensor: MtpTensor::OutputProjection,
            };
            if let Some(origin) = origin(&bound, &id)? {
                selected = Some((
                    origin,
                    format!("blk.{}.nextn.shared_head_head.weight", block.layer.index),
                ));
            } else if matches!(choice, HeadChoice::MtpBlock { .. }) {
                return Ok(None);
            }
        }
        let (origin, runtime_name) = match selected {
            Some(selected) => selected,
            None => (
                origin(&bound, &model_id)?.ok_or("bound target has no model output head")?,
                model_name.into(),
            ),
        };
        let source_binding_sha256 = match &bound {
            BoundProgramRef::Single(b) => b.binding_sha256(),
            BoundProgramRef::Composite(b) => b.binding_sha256(),
            BoundProgramRef::ExternalDraft(_) => unreachable!(),
        }
        .to_owned();
        let view = source
            .try_find(&runtime_name)?
            .ok_or_else(|| format!("selected bound head {runtime_name} disappeared"))?;
        let [width, rows] = view.ne.as_slice() else {
            return Err("self-trim head must be a matrix".into());
        };
        if *width == 0 || *rows == 0 {
            return Err("self-trim head has an empty dimension".into());
        }
        let rows_usize = usize::try_from(*rows).map_err(|_| "head rows exceed usize")?;
        ranks.validate_for_head(rows_usize, &runtime_name)?;
        let row_bytes = row_bytes(view.ggml_type, *width)?;
        if row_bytes.checked_mul(rows_usize) != Some(view.bytes.len()) {
            return Err("self-trim source byte length does not match complete rows".into());
        }
        let (macro_present, source_macro) = if view.ggml_type == GgmlType::NVFP4 {
            let name = format!("{}.scale", runtime_name.strip_suffix(".weight").unwrap());
            match source.try_find(&name)? {
                Some(scale) => (
                    true,
                    super::consumer::read_nvfp4_macro_scale(&scale)
                        .map_err(|e| format!("{name}: {e}"))?,
                ),
                None => (false, 1.0),
            }
        } else {
            (false, 1.0)
        };
        let output_shape = [*width, ranks.ids().len() as u64];
        let gathered_bytes = row_bytes
            .checked_mul(ranks.ids().len())
            .ok_or("trim payload byte count overflow")?;
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(gathered_bytes)
            .map_err(|e| format!("cannot allocate trimmed head: {e}"))?;
        bytes.resize(gathered_bytes, 0);
        let positions: BTreeMap<usize, usize> = ranks
            .ids()
            .iter()
            .enumerate()
            .map(|(rank, &id)| (id as usize, rank))
            .collect();
        let mut head_hash = Sha256::new();
        let mut captured_row = vec![0u8; row_bytes];
        for (row, source_row) in view.bytes.chunks_exact(row_bytes).enumerate() {
            captured_row.copy_from_slice(source_row);
            head_hash.update(&captured_row);
            if let Some(&rank) = positions.get(&row) {
                bytes[rank * row_bytes..(rank + 1) * row_bytes].copy_from_slice(&captured_row);
            }
        }
        let source_head_sha256 = format!("{:x}", head_hash.finalize());
        let convert = policy == TrimPolicy::Nvfp4ForEligibleBf16
            && view.ggml_type == GgmlType::BF16
            && width.is_multiple_of(64);
        let (program, output_dtype, output_macro) = if convert {
            let values: Vec<f32> = bytes
                .chunks_exact(2)
                .map(|b| f32::from_bits(u32::from(u16::from_le_bytes([b[0], b[1]])) << 16))
                .collect();
            bytes = crate::nvfp4_repack::f32_to_nvfp4(&values);
            (TrimProgram::Bf16RowsToNvfp4V1, GgmlType::NVFP4, 1.0f32)
        } else {
            (TrimProgram::RowGather, view.ggml_type, source_macro)
        };
        let output_sha256 = format!("{:x}", Sha256::digest(&bytes));
        let mut hash = Sha256::new();
        hash.update(b"memra-bound-head-trim-v1\0");
        for encoded in [
            format!("{source_binding_sha256:?}"),
            format!("{origin:?}"),
            format!("{:?}", ranks.identity()),
            format!(
                "{:?}",
                (
                    policy,
                    program,
                    view.ggml_type,
                    [*width, *rows],
                    &source_head_sha256,
                    macro_present,
                    source_macro.to_bits()
                )
            ),
            format!(
                "{:?}",
                (
                    output_dtype,
                    output_shape,
                    output_macro.to_bits(),
                    &output_sha256
                )
            ),
        ] {
            hash.update((encoded.len() as u64).to_le_bytes());
            hash.update(encoded.as_bytes());
        }
        let identity = HeadTrimIdentity {
            source_binding_sha256,
            origin,
            rank: ranks.identity().clone(),
            policy,
            program,
            source_dtype: view.ggml_type,
            source_shape: [*width, *rows],
            source_head_sha256,
            macro_present,
            source_macro_bits: source_macro.to_bits(),
            output_dtype,
            output_shape,
            output_macro_bits: output_macro.to_bits(),
            output_sha256,
            materialization_sha256: format!("{:x}", hash.finalize()),
        };
        Ok(Some(Self {
            bytes: bytes.into_boxed_slice(),
            ids: Arc::from(ranks.ids()),
            runtime_name,
            identity,
            gathered_bytes,
        }))
    }
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }
    pub fn ids(&self) -> &[u32] {
        &self.ids
    }
    pub fn runtime_name(&self) -> &str {
        &self.runtime_name
    }
    pub fn identity(&self) -> &HeadTrimIdentity {
        &self.identity
    }
    pub fn source_dtype(&self) -> GgmlType {
        self.identity.source_dtype
    }
    pub fn source_shape(&self) -> [u64; 2] {
        self.identity.source_shape
    }
    pub fn dtype(&self) -> GgmlType {
        self.identity.output_dtype
    }
    pub fn shape(&self) -> [u64; 2] {
        self.identity.output_shape
    }
    pub fn macro_scale(&self) -> f32 {
        f32::from_bits(self.identity.output_macro_bits)
    }
    pub fn from_model_output(&self) -> bool {
        matches!(
            self.identity.origin.id,
            TensorId::TokenEmbedding | TensorId::OutputProjection
        )
    }
    pub fn requant_sizes(&self) -> Option<(usize, usize)> {
        (self.identity.program == TrimProgram::Bf16RowsToNvfp4V1)
            .then_some((self.bytes.len(), self.gathered_bytes))
    }
}

fn row_bytes(dtype: GgmlType, width: u64) -> Result<usize, String> {
    match dtype {
        GgmlType::BF16
        | GgmlType::F32
        | GgmlType::Q8_0
        | GgmlType::Q4_K
        | GgmlType::Q6_K
        | GgmlType::Q5_K
        | GgmlType::Q3_K
        | GgmlType::IQ4_XS
        | GgmlType::IQ3_S
        | GgmlType::NVFP4
        | GgmlType::Q4_0 => {}
        _ => return Err(format!("self-trim has no resident consumer for {dtype:?}")),
    }
    let (block, size) = dtype.block_and_type_size();
    if !width.is_multiple_of(block) {
        return Err(format!(
            "self-trim row width {width} is not aligned to {dtype:?} block {block}"
        ));
    }
    usize::try_from(
        (width / block)
            .checked_mul(size)
            .ok_or("head row byte count overflow")?,
    )
    .map_err(|_| "head row exceeds usize".into())
}

fn origin(bound: &BoundProgramRef<'_>, id: &TensorId) -> Result<Option<HeadOrigin>, String> {
    match bound {
        BoundProgramRef::Single(b) => {
            let Some(tensor) = b.binding().tensors.get(id) else {
                return Ok(None);
            };
            b.scope().authorize(id)?;
            if tensor.checkpoint_names.len() != 1 {
                return Err("head requires one physical materialization".into());
            }
            let record = b
                .census()
                .tensors
                .iter()
                .find(|r| r.entry.name == tensor.checkpoint_names[0])
                .ok_or("bound head census row missing")?
                .clone();
            Ok(Some(HeadOrigin {
                id: id.clone(),
                component_path: vec![],
                dialect: b.census().dialect,
                record,
                transform: tensor.transform,
            }))
        }
        BoundProgramRef::Composite(b) => {
            let Some(selected) = b.catalog().selected().get(id) else {
                if b.catalog()
                    .components()
                    .iter()
                    .any(|c| c.binding().tensors.contains_key(id))
                {
                    return Err("head is outside selected composite scope".into());
                }
                return Ok(None);
            };
            if selected.member_ids.is_some() || selected.members.len() != 1 {
                return Err("head requires one selected composite materialization".into());
            }
            let member = &selected.members[0];
            Ok(Some(HeadOrigin {
                id: id.clone(),
                component_path: member.component_path.clone(),
                dialect: member.dialect,
                record: member.record.clone(),
                transform: member.transform,
            }))
        }
        BoundProgramRef::ExternalDraft(_) => {
            Err("target head cannot use external draft authority".into())
        }
    }
}
