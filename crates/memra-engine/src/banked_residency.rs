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

/// Gate-only budgets for the `--experts-via-tier` door. Both come from the gate
/// binary's argv (`--expert-bank-host-bytes=N`, `--expert-bank-gpu-bytes=N`), never
/// from an environment variable. `gpu_bytes == None` leaves native slot sizing
/// (`MEMRA_MOE_SLOTS` / auto) exactly as it is.
///
/// `stage_clock` is not a budget: `--expert-bank-stages` installs the door's log-only stage
/// clock (`research/spill-c-20260919/DAY40.md`), an explanatory diagnostic that reads
/// `Instant` and two timing events per GPU miss and changes no decision. It shares the
/// door's decide-by (`MOE-SLOT-CACHE-DOOR.md`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExpertBankBudget {
    pub host_bytes: u64,
    pub gpu_bytes: Option<u64>,
    pub stage_clock: bool,
}
impl Default for ExpertBankBudget {
    fn default() -> Self {
        Self {
            host_bytes: 256 * 1024 * 1024,
            gpu_bytes: None,
            stage_clock: false,
        }
    }
}

/// The only installer error the gate binaries map to the refusal token contract
/// (final stderr line `REFUSED: <reason>`, exit 2): a budget the bank cannot hold, or an
/// expert catalog the plan and tensor contract cannot bind for the artifact. Every other
/// error stays a failure.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExpertBankRefusal(pub String);
impl std::fmt::Display for ExpertBankRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for ExpertBankRefusal {}

/// The refusal reason if `err` is exactly a typed budget refusal, else `None`.
pub fn refusal_reason<'a>(err: &'a (dyn std::error::Error + 'static)) -> Option<&'a str> {
    err.downcast_ref::<ExpertBankRefusal>()
        .map(|refusal| refusal.0.as_str())
}

/// Parse the door and its budgets from argv. `Ok(None)` when the door is absent;
/// a budget flag without `--experts-via-tier`, a malformed or repeated value, a bare
/// flag, or a key that merely starts with a flag name is a usage error (a failure, not
/// a refusal). Keys match exactly: `--expert-bank-host-bytes-x=1` is not the host flag.
pub fn expert_bank_cli<I: IntoIterator<Item = String>>(
    args: I,
) -> std::result::Result<Option<ExpertBankBudget>, String> {
    const DOOR: &str = "--experts-via-tier";
    const HOST: &str = "--expert-bank-host-bytes";
    const GPU: &str = "--expert-bank-gpu-bytes";
    const STAGES: &str = "--expert-bank-stages";
    const FAMILY: &str = "--expert-bank-";
    let mut door = false;
    let mut stages = false;
    let mut host = None;
    let mut gpu = None;
    for arg in args {
        let (key, value) = match arg.split_once('=') {
            Some((key, value)) => (key, Some(value)),
            None => (arg.as_str(), None),
        };
        let slot = match key {
            DOOR => {
                if value.is_some() {
                    return Err(format!("{DOOR} takes no value"));
                }
                door = true;
                continue;
            }
            STAGES => {
                if value.is_some() {
                    return Err(format!("{STAGES} takes no value"));
                }
                if std::mem::replace(&mut stages, true) {
                    return Err(format!("{STAGES} given more than once"));
                }
                continue;
            }
            HOST => &mut host,
            GPU => &mut gpu,
            _ if key.starts_with(FAMILY) || key.starts_with(DOOR) => {
                return Err(format!(
                    "unknown expert bank flag {key:?}; expected {DOOR}, {HOST}=<bytes>, {GPU}=<bytes> or {STAGES}"
                ));
            }
            _ => continue,
        };
        let value = value.ok_or_else(|| format!("{key} expects {key}=<bytes>"))?;
        let bytes = value
            .parse::<u64>()
            .map_err(|_| format!("{key} expects an unsigned byte count, got {value:?}"))?;
        if slot.replace(bytes).is_some() {
            return Err(format!("{key} given more than once"));
        }
    }
    if !door {
        if host.is_some() || gpu.is_some() {
            return Err(format!("expert bank budgets require {DOOR}"));
        }
        if stages {
            return Err(format!("{STAGES} requires {DOOR}"));
        }
        return Ok(None);
    }
    let mut budget = ExpertBankBudget::default();
    if let Some(bytes) = host {
        budget.host_bytes = bytes;
    }
    budget.gpu_bytes = gpu;
    budget.stage_clock = stages;
    Ok(Some(budget))
}

/// Typed host budget refusal naming the requested and minimum bytes.
pub fn host_bank_budget(
    bytes: u64,
    max_record: u64,
) -> std::result::Result<usize, ExpertBankRefusal> {
    host_bank_slots(bytes, max_record).map_err(|reason| {
        ExpertBankRefusal(format!(
            "{reason} (requested {bytes}, minimum {max_record}, ceiling {})",
            256u64 * 1024 * 1024
        ))
    })
}

/// Tail pad `MoeSlotCache` allocates after every slot: wide expert dots may issue an
/// aligned read past the final block. One definition for the native slot sizing
/// (`moe_cache.rs` imports it) and for the door's budget arithmetic below; it lives here
/// because the tier bank tests compile this file verbatim.
pub const SLOT_TAIL_PAD_BYTES: usize = 8;

/// Bytes one native GPU slot costs for a record of `max_record` bytes: the record
/// plus `SLOT_TAIL_PAD_BYTES`.
pub fn gpu_slot_bytes(max_record: u64) -> Option<u64> {
    max_record.checked_add(SLOT_TAIL_PAD_BYTES as u64)
}

/// Qualification-only GPU slot budget for the door. `hard_bytes` is the machine hard
/// ceiling the native cache would apply (`MEMRA_MOE_HARD_VRAM_FRAC` of free VRAM
/// minus two slots), measured by the installer before any allocation. Refuses below
/// the eight-slot minimum and above the ceiling; never clamps, never saturates.
pub fn gpu_bank_slots(
    bytes: u64,
    max_record: u64,
    hard_bytes: u64,
) -> std::result::Result<usize, &'static str> {
    if max_record == 0 {
        return Err("experts-via-tier GPU bank budget has no expert record");
    }
    let slot =
        gpu_slot_bytes(max_record).ok_or("experts-via-tier GPU bank budget arithmetic overflow")?;
    let minimum = slot
        .checked_mul(8)
        .ok_or("experts-via-tier GPU bank budget arithmetic overflow")?;
    if bytes < minimum {
        return Err("experts-via-tier GPU bank budget cannot hold the eight-slot minimum");
    }
    if bytes > hard_bytes {
        return Err("experts-via-tier GPU bank budget exceeds the hard VRAM ceiling");
    }
    usize::try_from(bytes / slot)
        .map_err(|_| "experts-via-tier GPU bank budget arithmetic overflow")
}

/// Typed GPU budget refusal naming the requested, minimum and ceiling bytes.
pub fn gpu_bank_budget(
    bytes: u64,
    max_record: u64,
    hard_bytes: u64,
) -> std::result::Result<usize, ExpertBankRefusal> {
    gpu_bank_slots(bytes, max_record, hard_bytes).map_err(|reason| {
        let minimum = gpu_slot_bytes(max_record)
            .and_then(|slot| slot.checked_mul(8))
            .map_or_else(|| "overflow".to_owned(), |m| m.to_string());
        ExpertBankRefusal(format!(
            "{reason} (requested {bytes}, minimum {minimum}, ceiling {hard_bytes})"
        ))
    })
}
