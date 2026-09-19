//! Native adapter metadata, not a new expert or PLE numerical executor.
use super::*;
use crate::contracts::*;
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug)]
pub struct ExpertMetadata {
    pub offset: u64,
    pub len: u64,
    pub qtype: i32,
    pub row_bytes: u64,
}
/// Thin engine bridge uses HostExps::expert_layout/max_expert_bytes rather than
/// projection-wide fields. Returning Some metadata is never a uniform attestation.
pub trait HostExpsView {
    fn is_uniform_layout(&self) -> bool;
    fn n_expert(&self) -> usize;
    fn max_expert_bytes(&self) -> u64;
    fn expert_layout(&self, original_id: usize) -> Result<ExpertMetadata>;
}
#[derive(Clone, Debug)]
pub struct ExpertSource {
    /// Original semantic checkpoint identity. Split sources have their own tensor
    /// IDs and positioned offset 0, never the slab offset applied a second time.
    pub tensor: TensorId,
    pub split: bool,
    pub scales: Vec<ByteSegment>,
    pub checksums: Vec<Digest>,
}
pub struct HostExpsMapping {
    pub catalog: Catalog,
    pub ids: Vec<BankId>,
    pub max_expert_bytes: u64,
}
pub fn host_exps_catalog<V: HostExpsView>(
    view: &V,
    tensor: TensorId,
    layer: u32,
    projection: Projection,
    active: &[bool],
    sources: &BTreeMap<u32, ExpertSource>,
) -> Result<HostExpsMapping> {
    if active.len() != view.n_expert() || view.n_expert() > u32::MAX as usize {
        return Err(Error::InvalidLayout);
    }
    let class = if view.is_uniform_layout() {
        LayoutClass::Uniform
    } else {
        LayoutClass::PerRecord
    };
    let mut entries = Vec::new();
    let mut ids = Vec::new();
    for (original, &retained) in active.iter().enumerate() {
        let record_id = RecordId::Expert {
            layer,
            original_id: original as u32,
            projection: projection.clone(),
        };
        let (layout_id, record) = if retained {
            let m = view.expert_layout(original)?;
            if m.len == 0
                || m.len > view.max_expert_bytes()
                || m.row_bytes == 0
                || !m.len.is_multiple_of(m.row_bytes)
            {
                return Err(Error::InvalidLayout);
            }
            let source = sources.get(&(original as u32)).ok_or(Error::Incomplete)?;
            if source.tensor.artifact != tensor.artifact {
                return Err(Error::ProgramMismatch);
            }
            let offset = if source.split { 0 } else { m.offset };
            let payload = ByteSegment {
                version: WIRE_VERSION,
                group: 0,
                page: 0,
                owner: 0,
                role: Role::Payload,
                tensor: Some(source.tensor.clone()),
                offset,
                valid_bytes: m.len,
                storage_bytes: m.len,
                alignment: 1,
                encoding: EncodingId {
                    version: WIRE_VERSION,
                    program: digest("host-exps-qtype-v1", &m.qtype.to_le_bytes()),
                    row_bytes: m.row_bytes,
                },
            };
            let mut segments = vec![payload];
            if source
                .scales
                .iter()
                .any(|s| !matches!(s.role, Role::Scale | Role::MacroScale))
            {
                return Err(Error::InvalidLayout);
            }
            segments.extend(source.scales.clone());
            let requirements = segments
                .iter()
                .map(|s| GroupRequirement {
                    version: WIRE_VERSION,
                    group: s.group,
                    owner: s.owner,
                    role: s.role,
                    page_count: 1,
                    pages: PageRequirement::AllPages,
                })
                .collect();
            let layout = RecordLayout {
                version: WIRE_VERSION,
                segments,
                requirements,
            };
            (
                layout.identity()?,
                Some(CatalogRecord {
                    layout,
                    checksums: source.checksums.clone(),
                }),
            )
        } else {
            // Masked records have no backing assignment and keep router positions.
            if sources.contains_key(&(original as u32)) {
                return Err(Error::MaskedId);
            }
            (digest("masked-bank-record-v1", &[]), None)
        };
        let id = BankId {
            version: WIRE_VERSION,
            tensor: tensor.clone(),
            record: record_id,
            layout: layout_id,
        };
        ids.push(id.clone());
        entries.push((id, record));
    }
    if sources.keys().any(|&id| id as usize >= active.len()) {
        return Err(Error::InvalidLayout);
    }
    Ok(HostExpsMapping {
        catalog: Catalog::new(class, entries)?,
        ids,
        max_expert_bytes: view.max_expert_bytes(),
    })
}

/// New uniform-only adapter entry boundary. This does not replace any runtime
/// kernel yet. Callback cannot receive an unchecked ordinary bank batch.
/// ```compile_fail
/// use memra_tier::{bank::with_uniform_experts, contracts::BankLease};
/// fn mixed(lease: &BankLease) { with_uniform_experts(lease, |_| ()); }
/// ```
pub fn with_uniform_experts<T>(lease: &UniformLease, kernel: impl FnOnce(&UniformLease) -> T) -> T {
    kernel(lease)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PleEncoding {
    F32,
    Bf16,
}
impl PleEncoding {
    pub fn element_bytes(self) -> u64 {
        match self {
            Self::F32 => 4,
            Self::Bf16 => 2,
        }
    }
    fn program(self) -> Digest {
        digest(
            "ple-row-encoding-v1",
            match self {
                Self::F32 => b"f32-le",
                Self::Bf16 => b"bf16-le",
            },
        )
    }
}
/// Descriptor is O(1) in table size. Only the requested bounded trace/catalog
/// window is materialized; no whole-table pinning or row allocation.
pub struct PleTable {
    pub tensor: TensorId,
    pub rows: u64,
    pub head_dim: u64,
    pub encoding: PleEncoding,
}
impl PleTable {
    pub fn layout(&self, row: u64) -> Result<RecordLayout> {
        if row >= self.rows {
            return Err(Error::NotFound);
        }
        let width = self
            .head_dim
            .checked_mul(self.encoding.element_bytes())
            .ok_or(Error::Overflow)?;
        if width == 0 {
            return Err(Error::InvalidLayout);
        }
        let segment = ByteSegment {
            version: WIRE_VERSION,
            group: 0,
            page: 0,
            owner: 0,
            role: Role::Payload,
            tensor: Some(self.tensor.clone()),
            offset: row.checked_mul(width).ok_or(Error::Overflow)?,
            valid_bytes: width,
            storage_bytes: width,
            alignment: 1,
            encoding: EncodingId {
                version: WIRE_VERSION,
                program: self.encoding.program(),
                row_bytes: width,
            },
        };
        Ok(RecordLayout {
            version: WIRE_VERSION,
            segments: vec![segment],
            requirements: vec![GroupRequirement {
                version: WIRE_VERSION,
                group: 0,
                owner: 0,
                role: Role::Payload,
                page_count: 1,
                pages: PageRequirement::AllPages,
            }],
        })
    }
    pub fn id(&self, row: u64) -> Result<BankId> {
        Ok(BankId {
            version: WIRE_VERSION,
            tensor: self.tensor.clone(),
            record: RecordId::Row(row),
            layout: self.layout(row)?.identity()?,
        })
    }
    /// Consume the existing model's authoritative full-history/cached IDs. The
    /// storage adapter never hashes tokens, guesses history or repairs stale IDs.
    pub fn last_chunk(
        &self,
        all_ids: &[i64],
        tokens: usize,
        chunk: usize,
        heads: usize,
        epochs: Epochs,
        request: BudgetRequest,
    ) -> Result<RowBatch> {
        if heads == 0
            || chunk == 0
            || chunk > tokens
            || tokens.checked_mul(heads) != Some(all_ids.len())
        {
            return Err(Error::InvalidLayout);
        }
        let ids = all_ids[(tokens - chunk) * heads..]
            .iter()
            .map(|&row| self.id(u64::try_from(row).map_err(|_| Error::NotFound)?))
            .collect::<Result<_>>()?;
        Ok(RowBatch {
            ids,
            epochs,
            request,
        })
    }
    /// Same bit expansion as NgramTable::gather_into, after host publication only.
    /// No FP8 conversion, arithmetic or projection change.
    pub fn expand_row(&self, lease: &BankLease) -> Result<Vec<f32>> {
        let RecordId::Row(row) = lease.id().record else {
            return Err(Error::InvalidLayout);
        };
        if lease.id() != &self.id(row)? {
            return Err(Error::ProgramMismatch);
        }
        let bytes = lease.resource::<Vec<u8>>()?;
        let width = self
            .head_dim
            .checked_mul(self.encoding.element_bytes())
            .ok_or(Error::Overflow)?;
        if bytes.len() as u64 != width {
            return Err(Error::Incomplete);
        }
        Ok(match self.encoding {
            PleEncoding::F32 => bytes
                .chunks_exact(4)
                .map(|b| f32::from_bits(u32::from_le_bytes(b.try_into().expect("four bytes"))))
                .collect(),
            PleEncoding::Bf16 => bytes
                .chunks_exact(2)
                .map(|b| f32::from_bits(u32::from(u16::from_le_bytes([b[0], b[1]])) << 16))
                .collect(),
        })
    }
}
