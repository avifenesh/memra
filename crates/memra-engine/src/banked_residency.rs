//! WP-C exact HostExps metadata bridge. The explicit qualification installer in
//! banked_residency/native.rs uses this mapping before installing the owner-thread
//! proxy; normal loader and dispatch defaults do not install that door.
use crate::model::HostExps;
use memra_tier::{bank::*, contracts::*};
use std::collections::BTreeMap;

// Native qualification binds real HostExps. The CPU bank integration test also
// checks map_host_exps with an API-shaped fixture, not a CUDA loader substitute.
struct HostView<'a>(&'a HostExps);
impl HostExpsView for HostView<'_> {
    fn is_uniform_layout(&self) -> bool {
        self.0.is_uniform_layout()
    }
    fn n_expert(&self) -> usize {
        self.0.n_expert
    }
    fn max_expert_bytes(&self) -> u64 {
        self.0.max_expert_bytes() as u64
    }
    fn expert_layout(&self, original_id: usize) -> Result<ExpertMetadata> {
        if original_id >= self.0.n_expert {
            return Err(Error::NotFound);
        }
        let layout = self.0.expert_layout(original_id);
        Ok(ExpertMetadata {
            offset: layout.offset as u64,
            len: layout.len as u64,
            qtype: layout.qtype,
            row_bytes: layout.row_bytes as u64,
        })
    }
}

/// Sources name immutable tensors and their checksum-bound payload/scale extents.
/// The loader supplies original router masks. Never infer active IDs from a dense
/// repacked vector or omit an existing macro/block-scale plane.
pub fn map_host_exps(
    host: &HostExps,
    tensor: TensorId,
    layer: u32,
    projection: Projection,
    active: &[bool],
    sources: &BTreeMap<u32, ExpertSource>,
) -> Result<HostExpsMapping> {
    if active.len() != host.n_expert
        || host
            .layouts
            .as_ref()
            .is_some_and(|v| v.len() != host.n_expert)
        || host
            .tiers
            .as_ref()
            .is_some_and(|v| v.len() != host.n_expert)
        || host
            .macros
            .as_ref()
            .is_some_and(|v| v.len() != host.n_expert)
    {
        return Err(Error::InvalidLayout);
    }
    for (&id, source) in sources {
        if id as usize >= host.n_expert || source.split != host.tiers.is_some() {
            return Err(Error::InvalidLayout);
        }
        let macros: Vec<_> = source
            .scales
            .iter()
            .filter(|s| s.role == Role::MacroScale)
            .collect();
        if macros.len() != usize::from(host.macros.is_some()) {
            return Err(Error::Incomplete);
        }
        if let Some(values) = &host.macros {
            let s = macros[0];
            let index = source
                .scales
                .iter()
                .position(|x| std::ptr::eq(x, s))
                .ok_or(Error::Incomplete)?
                + 1;
            if s.valid_bytes != 4
                || source.checksums.get(index)
                    != Some(&checksum(&values[id as usize].to_bits().to_le_bytes()))
            {
                return Err(Error::Corrupt);
            }
        }
        let blocks: Vec<_> = source
            .scales
            .iter()
            .filter(|s| s.role == Role::Scale)
            .collect();
        if blocks.len() != usize::from(host.fp8_blk.is_some()) {
            return Err(Error::Incomplete);
        }
        if let Some(scales) = &host.fp8_blk {
            let start = (id as usize)
                .checked_mul(scales.expert_stride)
                .ok_or(Error::Overflow)?;
            let end = start
                .checked_add(scales.expert_stride)
                .ok_or(Error::Overflow)?;
            let values = scales.scales.get(start..end).ok_or(Error::Incomplete)?;
            let raw: Vec<_> = values
                .iter()
                .flat_map(|v| v.to_bits().to_le_bytes())
                .collect();
            let s = blocks[0];
            let index = source
                .scales
                .iter()
                .position(|x| std::ptr::eq(x, s))
                .ok_or(Error::Incomplete)?
                + 1;
            if s.valid_bytes != raw.len() as u64
                || source.checksums.get(index) != Some(&checksum(&raw))
            {
                return Err(Error::Corrupt);
            }
        }
    }
    host_exps_catalog(&HostView(host), tensor, layer, projection, active, sources)
}

/// Qualification-only host budget, independent of native CUDA slot sizing.
/// Refuse an impossible record instead of silently increasing the budget.
pub fn host_bank_slots(bytes: u64, max_record: u64) -> std::result::Result<usize, &'static str> {
    if max_record == 0 || bytes < max_record {
        return Err("experts-via-tier host bank budget cannot hold one expert record");
    }
    if bytes > 256 * 1024 * 1024 {
        return Err("experts-via-tier host bank budget exceeds qualification ceiling");
    }
    Ok((bytes / max_record).min(16) as usize)
}
