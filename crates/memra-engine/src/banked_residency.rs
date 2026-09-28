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

/// The host tier's plan under `--expert-bank-host-bytes` (day 43,
/// `research/spill-c-20260919/DAY43.md`): one SLRU class per exact record size of the catalog,
/// each class's slots proportional to its record count under the budget (the Hamilton
/// remainder pass of `moe_cache.rs::size_class_plan`), capped at the class's record count, with
/// at least one slot of the largest size. Refused, never clamped: a budget that cannot hold one
/// record of the largest size, and a budget above the machine ceiling.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HostBankPlan {
    /// Ascending `(record bytes, slots)`, every count positive: `SlruPolicy::new`'s input.
    pub classes: Vec<(u64, usize)>,
    /// Bytes the planned slots hold at their record sizes.
    pub planned_bytes: u64,
    /// Planned slots, one record each.
    pub records_held: usize,
}

/// Plan `bytes` over `record_sizes` (one entry per retained record). `ceiling` is the machine
/// ceiling the installer measured (`host_bank_ceiling`).
pub fn host_bank_plan(
    bytes: u64,
    record_sizes: &[u64],
    ceiling: u64,
) -> std::result::Result<HostBankPlan, &'static str> {
    let mut counts: BTreeMap<u64, u64> = BTreeMap::new();
    for &size in record_sizes.iter().filter(|&&size| size > 0) {
        *counts.entry(size).or_insert(0) += 1;
    }
    let Some((&largest, _)) = counts.iter().next_back() else {
        return Err("experts-via-tier host bank has no expert record");
    };
    if bytes < largest {
        return Err("experts-via-tier host bank budget cannot hold one expert record");
    }
    if bytes > ceiling {
        return Err("experts-via-tier host bank budget exceeds the machine ceiling");
    }
    let total: u128 = counts
        .iter()
        .map(|(&size, &count)| u128::from(size) * u128::from(count))
        .sum();
    let budget = u128::from(bytes);
    // (size, available, planned, remainder)
    let mut plan: Vec<(u64, u64, u64, u128)> = counts
        .iter()
        .map(|(&size, &count)| {
            let scaled = u128::from(count) * budget;
            let planned = (scaled / total).min(u128::from(count)) as u64;
            (size, count, planned, scaled % total)
        })
        .collect();
    let used = |plan: &[(u64, u64, u64, u128)]| -> u128 {
        plan.iter()
            .map(|&(size, _, planned, _)| u128::from(size) * u128::from(planned))
            .sum()
    };
    let mut order: Vec<usize> = (0..plan.len()).collect();
    order.sort_by(|&a, &b| plan[b].3.cmp(&plan[a].3).then(a.cmp(&b)));
    for index in order {
        let (size, available, planned, _) = plan[index];
        if planned < available && used(&plan) + u128::from(size) <= budget {
            plan[index].2 += 1;
        }
    }
    // One slot of the largest size, taking slots from the smallest classes if the budget
    // is full (the budget holds one largest record, checked above).
    let last = plan.len() - 1;
    if plan[last].2 == 0 {
        plan[last].2 = 1;
        let mut index = 0;
        while used(&plan) > budget && index < last {
            if plan[index].2 > 0 {
                plan[index].2 -= 1;
            } else {
                index += 1;
            }
        }
    }
    let planned_bytes = u64::try_from(used(&plan))
        .map_err(|_| "experts-via-tier host bank plan arithmetic overflow")?;
    let classes: Vec<(u64, usize)> = plan
        .iter()
        .filter(|&&(_, _, planned, _)| planned > 0)
        .map(|&(size, _, planned, _)| (size, planned as usize))
        .collect();
    let records_held = classes.iter().map(|&(_, slots)| slots).sum();
    Ok(HostBankPlan {
        classes,
        planned_bytes,
        records_held,
    })
}

/// Typed host plan refusal naming the requested, minimum and ceiling bytes.
pub fn host_bank_budget(
    bytes: u64,
    record_sizes: &[u64],
    ceiling: u64,
) -> std::result::Result<HostBankPlan, ExpertBankRefusal> {
    host_bank_plan(bytes, record_sizes, ceiling).map_err(|reason| {
        let minimum = record_sizes.iter().copied().max().unwrap_or(0);
        ExpertBankRefusal(format!(
            "{reason} (requested {bytes}, minimum {minimum}, ceiling {ceiling})"
        ))
    })
}

/// The machine ceiling for the host tier: three quarters of `MemAvailable` in a
/// `/proc/meminfo` text, in bytes. `None` when the field is absent or malformed (the installer
/// then refuses: an unknown host memory is never read as unlimited).
pub fn host_bank_ceiling(meminfo: &str) -> Option<u64> {
    let line = meminfo.lines().find(|l| l.starts_with("MemAvailable:"))?;
    let mut fields = line.split_whitespace().skip(1);
    let kib: u64 = fields.next()?.parse().ok()?;
    if fields.next() != Some("kB") {
        return None;
    }
    kib.checked_mul(1024)?.checked_div(4)?.checked_mul(3)
}

/// The door's budgets, from the gate binary's argv (`--expert-bank-host-bytes=N`,
/// `--expert-bank-gpu-bytes=N`), never from an environment variable. `gpu_bytes == None` leaves
/// native slot sizing (`MEMRA_MOE_SLOTS` / auto) exactly as it is.
///
/// `stage_clock` is not a budget: `--expert-bank-stages` installs the door's log-only stage
/// clock (`research/spill-c-20260919/DAY40.md`), an explanatory diagnostic that reads
/// `Instant` and two timing events per GPU miss and changes no decision.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ExpertBankBudget {
    /// Day 88 (`research/spill-c-20260919/DAY88.md`): `None`, the default, holds the whole bank
    /// (every record of the catalog, the shape every card cell measured); `Some(N)` plans N bytes.
    pub host_bytes: Option<u64>,
    pub gpu_bytes: Option<u64>,
    pub stage_clock: bool,
    /// Day 88: `--expert-bank-pool-allocated` makes the host pool with `cuMemHostAlloc`, the pool
    /// before the promotion (the rollback seam, decide-by 2026-10-11 in the door document). The
    /// default pool is private anonymous memory registered with `cuMemHostRegister` (DAY80), which
    /// compaction skips instead of isolating.
    pub pool_allocated: bool,
    /// Day 92 (I25a, `research/spill-c-20260919/DAY92.md`): `--expert-bank-trace` writes the complete host demand
    /// trace (`[expert-host-slru] key=...`, one line per demanded record) that the gates' and cells' integrity checks
    /// read. A diagnostic the gate sets: without it the door traces nothing and changes no decision.
    pub trace: bool,
}

/// Day 88 (`research/spill-c-20260919/DAY88.md`, phase 1 of the door's promotion): how the gate
/// binary asks for the MoE slot cache door.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExpertBankMode {
    /// No `--experts-via-tier`: the door installs by default where it applies (a qualified
    /// artifact whose experts go to the slot cache, the requested bank under the host ceiling);
    /// elsewhere the program is today's and one line says why.
    Default,
    /// `--experts-via-tier`: the gate's assertion; a door that does not install refuses.
    Required,
}

/// Day 88: the gate binary's parsed door request.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExpertBankCli {
    pub mode: ExpertBankMode,
    pub budget: ExpertBankBudget,
    /// Whether a budget, pool or clock flag was given: each needs a door that installs, so with
    /// the door off it is a usage error, never a silent no-op.
    pub door_flags: bool,
}

/// Day 88 (`research/spill-c-20260919/DAY88.md` section 2.1): an opened artifact the door is
/// qualified on, authenticated once before the model loads (`Engine::qualify_expert_door`): its
/// SHA-256, the opened file's device and inode (the installer accepts it only for that same open
/// file), the host ceiling read then, and the whole bank's bytes the installer's catalog must equal.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct QualifiedArtifact {
    pub(crate) digest: [u8; 32],
    pub(crate) dev: u64,
    pub(crate) ino: u64,
    pub(crate) ceiling: u64,
    pub(crate) whole_bank: u64,
    /// Whether the requested host bank (the whole bank unless `--expert-bank-host-bytes`) fits
    /// under that ceiling; the door is off on a qualified artifact that does not fit.
    pub(crate) fits: bool,
}
impl QualifiedArtifact {
    /// The artifact's SHA-256, lowercase hex.
    pub fn sha256_hex(&self) -> String {
        self.digest.iter().map(|b| format!("{b:02x}")).collect()
    }
}

/// Day 88: whether the door applies to an artifact, decided before the model loads, or why not.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Qualification {
    Qualified(QualifiedArtifact),
    Off(String),
}

/// Day 88: `MEMRA_EXPERTS_VIA_TIER` against the parsed request. `Ok(true)` when the rollback is
/// asked for (`=0`: the legacy SLRU slot cache from pinned copies); `=0` together with
/// `--experts-via-tier` or with a door flag is a usage error. Any other value, or none, is the
/// default (`Ok(false)`).
pub fn expert_bank_rollback(
    cli: &ExpertBankCli,
    env: Option<&str>,
) -> std::result::Result<bool, String> {
    if env != Some("0") {
        return Ok(false);
    }
    if cli.mode == ExpertBankMode::Required {
        return Err("--experts-via-tier with MEMRA_EXPERTS_VIA_TIER=0 names two programs".into());
    }
    if cli.door_flags {
        return Err(
            "expert bank flags need the door, and MEMRA_EXPERTS_VIA_TIER=0 turns it off".into(),
        );
    }
    Ok(true)
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

/// Parse the door and its budgets from argv. Day 88: the door is the default, so the budgets need
/// no `--experts-via-tier` (which asks the door to install or refuse); a malformed or repeated
/// value, a bare flag, a retired flag, or a key that merely starts with a flag name is a usage
/// error (a failure, not a refusal). Keys match exactly: `--expert-bank-host-bytes-x=1` is not the
/// host flag.
pub fn expert_bank_cli<I: IntoIterator<Item = String>>(
    args: I,
) -> std::result::Result<ExpertBankCli, String> {
    const DOOR: &str = "--experts-via-tier";
    const HOST: &str = "--expert-bank-host-bytes";
    const GPU: &str = "--expert-bank-gpu-bytes";
    const STAGES: &str = "--expert-bank-stages";
    const ALLOCATED: &str = "--expert-bank-pool-allocated";
    const TRACE: &str = "--expert-bank-trace";
    const FAMILY: &str = "--expert-bank-";
    let mut door = false;
    let mut stages = false;
    let mut allocated = false;
    let mut trace = false;
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
            STAGES | ALLOCATED | TRACE => {
                if value.is_some() {
                    return Err(format!("{key} takes no value"));
                }
                let flag = match key {
                    STAGES => &mut stages,
                    TRACE => &mut trace,
                    _ => &mut allocated,
                };
                if std::mem::replace(flag, true) {
                    return Err(format!("{key} given more than once"));
                }
                continue;
            }
            HOST => &mut host,
            GPU => &mut gpu,
            _ if key.starts_with(FAMILY) || key.starts_with(DOOR) => {
                return Err(format!(
                    "unknown expert bank flag {key:?}; expected {DOOR}, {HOST}=<bytes>, {GPU}=<bytes>, {ALLOCATED}, {STAGES} or {TRACE}"
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
    Ok(ExpertBankCli {
        mode: if door {
            ExpertBankMode::Required
        } else {
            ExpertBankMode::Default
        },
        door_flags: host.is_some() || gpu.is_some() || stages || allocated || trace,
        budget: ExpertBankBudget {
            host_bytes: host,
            gpu_bytes: gpu,
            stage_clock: stages,
            pool_allocated: allocated,
            trace,
        },
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
