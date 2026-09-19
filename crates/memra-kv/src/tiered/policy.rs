//! Deterministic policies only. No defaults are promoted from synthetic measurements.
use crate::record::TierError;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WritePolicy {
    Through,
    Back,
    Selective { reuse_threshold: u64 },
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WriteAction {
    Nothing,
    BackupBeforePublish,
    BackupBeforeEvict,
    RetainDirty,
}
pub fn write_action(
    policy: WritePolicy,
    committed: bool,
    backed: bool,
    reuses: u64,
    evict: bool,
) -> WriteAction {
    if !committed || backed {
        return WriteAction::Nothing;
    }
    match policy {
        WritePolicy::Through => WriteAction::BackupBeforePublish,
        WritePolicy::Selective { reuse_threshold } if reuses >= reuse_threshold => {
            WriteAction::BackupBeforePublish
        }
        _ if evict => WriteAction::BackupBeforeEvict,
        _ => WriteAction::RetainDirty,
    }
}
pub use memra_tier::contracts::PrefetchPolicy;
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrefetchDecision {
    Load,
    Wait,
    SameProgramCold,
    Refuse,
}
pub fn prefetch_decision(
    policy: PrefetchPolicy,
    complete: bool,
    can_run: bool,
    now_ns: u64,
    mandatory: bool,
    same_program_cold: bool,
) -> PrefetchDecision {
    if complete {
        return PrefetchDecision::Load;
    }
    let stop = match policy {
        PrefetchPolicy::BestEffort => can_run,
        PrefetchPolicy::Wait => false,
        PrefetchPolicy::Timeout(deadline) => now_ns >= deadline.0,
    };
    if !stop {
        return PrefetchDecision::Wait;
    }
    if mandatory || !same_program_cold {
        PrefetchDecision::Refuse
    } else {
        PrefetchDecision::SameProgramCold
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RestoreDecision {
    Load,
    Recompute,
    RequireState,
}
/// Decimal GB/s. Must be measured over the full route, including staging/materialization.
/// This simple cost model is only for optional prefixes before admission. It is NOT a
/// substitute for a (prompt,prefix) calibrated suffix-work model when that is available.
pub fn recompute_vs_load(
    prefix_tokens: u64,
    restore_bytes: u64,
    restore_gb_s: f64,
    prefill_tokens_s: f64,
    fixed_restore_seconds: f64,
    mandatory: bool,
    same_program_cold: bool,
) -> Result<RestoreDecision, TierError> {
    if mandatory {
        return Ok(RestoreDecision::RequireState);
    }
    if !same_program_cold {
        return Ok(RestoreDecision::Load);
    }
    if !restore_gb_s.is_finite()
        || restore_gb_s <= 0.0
        || !prefill_tokens_s.is_finite()
        || prefill_tokens_s <= 0.0
        || !fixed_restore_seconds.is_finite()
        || fixed_restore_seconds < 0.0
    {
        return Err(TierError::InvalidLayout);
    }
    let load = restore_bytes as f64 / restore_gb_s / 1e9 + fixed_restore_seconds;
    let cold = prefix_tokens as f64 / prefill_tokens_s;
    // Tie goes cold: no transfer has been admitted, and no tier value is assumed.
    Ok(if load < cold {
        RestoreDecision::Load
    } else {
        RestoreDecision::Recompute
    })
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EvictionPolicy {
    Lru,
    Slru,
}
#[derive(Clone, Copy, Debug)]
pub struct EvictionCandidate {
    pub id: u64,
    pub last_use: u64,
    pub protected: bool,
    pub leased: bool,
    pub inflight: bool,
    pub dirty: bool,
    pub has_backing: bool,
    pub mandatory: bool,
}
/// Optional immutable prefixes can be discarded; mandatory active state needs verified backing.
/// Dirty objects must go through the write policy before being reconsidered here.
pub fn eviction_victim(candidates: &[EvictionCandidate], policy: EvictionPolicy) -> Option<u64> {
    candidates
        .iter()
        .filter(|c| !c.leased && !c.inflight && !c.dirty && (!c.mandatory || c.has_backing))
        .min_by_key(|c| {
            (
                policy == EvictionPolicy::Slru && c.protected,
                c.last_use,
                c.id,
            )
        })
        .map(|c| c.id)
}

/// Measured full-route costs for one (prompt,reused-prefix,tier) cell, in ns.
/// Unlike the linear diagnostic above this includes the exact suffix program and
/// materialization. The caller must bind calibration to artifact/plan/binary/device.
#[derive(Clone, Copy, Debug)]
pub struct PrefixCosts {
    pub prompt_tokens: u64,
    pub reused_tokens: u64,
    pub cold_prefill_ns: u64,
    pub suffix_prefill_ns: u64,
    pub read_ns: u64,
    pub copy_ns: u64,
    pub materialize_ns: u64,
}
pub fn calibrated_recompute_vs_load(
    cost: PrefixCosts,
    mandatory: bool,
    already_admitted: bool,
    same_program_cold: bool,
) -> Result<RestoreDecision, TierError> {
    if mandatory {
        return Ok(RestoreDecision::RequireState);
    }
    if already_admitted {
        return Err(TierError::Busy);
    }
    if !same_program_cold {
        return Ok(RestoreDecision::Load);
    }
    if cost.reused_tokens > cost.prompt_tokens || cost.cold_prefill_ns == 0 {
        return Err(TierError::InvalidLayout);
    }
    let load = [
        cost.suffix_prefill_ns,
        cost.read_ns,
        cost.copy_ns,
        cost.materialize_ns,
    ]
    .into_iter()
    .try_fold(0u64, |a, b| a.checked_add(b).ok_or(TierError::Overflow))?;
    Ok(if load < cost.cold_prefill_ns {
        RestoreDecision::Load
    } else {
        RestoreDecision::Recompute
    })
}
