//! Per-burst phase attribution for speculative rounds (`MEMRA_SPEC_TRACE`, generalized
//! lane/glm5-extract-general from the glm5 loop's `MEMRA_GLM5_SPEC_TRACE` — the alias
//! stays honored). The draft / verify / accept / rollback / source-maintenance split is
//! SPEC-FAMILY-GENERIC: any spec loop owns those five boundaries, and the level-2 verify
//! sub-split buckets are MIXER-CLASS buckets (KDA, MLA — multi-family classes), not one
//! model's. The emit TAGS are the caller's, so a family's banked receipts keep their
//! exact grep shape (`[glm5-phase]` / `[glm5-phase-v]` for the glm5 loop).
//!
//! DEFAULT OFF BY DESIGN (the flag row's law): each phase boundary SYNCHRONIZES the
//! stream so device time lands in the right bucket, which serializes the round — a
//! diagnostic instrument, never a serving mode, and its numbers are phase SHARES, not
//! round walls (the un-traced round overlaps what the trace separates).

use crate::Engine;
use std::sync::atomic::Ordering;

/// `MEMRA_SPEC_TRACE=1` (or the glm5 alias): per-burst phase attribution is on.
pub fn spec_trace_on() -> bool {
    spec_trace_level() >= 1
}

/// Trace LEVEL: `1` = the per-burst phase lines (draft/verify/accept/roll/maint);
/// `2` = additionally the VERIFY sub-split — batched-class vs sequential-class time per
/// burst (vkda with its in-kernel scan share, vmla, vrest = glue+FFN+head). Level 2 adds
/// per-layer stream drains on top of level 1's phase drains: shares, never walls, never
/// a perf row (the standing trace law). Read once per process (the worker chunk-policy
/// pattern). The general name wins when both names are set to DIFFERENT levels — with
/// one loud stderr line naming the override (the alias is never silently dead).
pub fn spec_trace_level() -> u8 {
    use std::sync::OnceLock;
    static L: OnceLock<u8> = OnceLock::new();
    *L.get_or_init(|| {
        spec_trace_level_from(
            std::env::var("MEMRA_SPEC_TRACE").ok().as_deref(),
            std::env::var("MEMRA_GLM5_SPEC_TRACE").ok().as_deref(),
        )
    })
}

fn parse_level(v: Option<&str>) -> Option<u8> {
    match v {
        Some("1") => Some(1),
        Some("2") => Some(2),
        _ => None,
    }
}

/// Pure resolution over the general name and the glm5 alias (unit-tested without env
/// mutation). Either name alone is honored; both set and disagreeing = the general name
/// wins LOUDLY (one stderr line naming both values).
fn spec_trace_level_from(general: Option<&str>, glm5_alias: Option<&str>) -> u8 {
    let g = parse_level(general);
    let a = parse_level(glm5_alias);
    if let (Some(gv), Some(av)) = (g, a)
        && gv != av
    {
        eprintln!(
            "[spec-trace] MEMRA_SPEC_TRACE={gv} overrides MEMRA_GLM5_SPEC_TRACE={av} \
             (the general flag wins; unset one to silence this)"
        );
    }
    g.or(a).unwrap_or(0)
}

/// Trace-level-2 verify sub-phase accumulators (ns), drained by [`SpecPhaseNs::emit`].
/// Module-level atomics so the walk needs no signature plumbing through the ppN twin
/// (the KDA_FUSED6_DISPATCHES precedent); level 2 is a single-session instrument, so
/// cross-session interleaving is out of scope by definition.
pub(crate) static V_KDA_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub(crate) static V_KDA_SCAN_NS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub(crate) static V_MLA_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// FFN-branch share of the verify walk (lane/glm5-vrest): MoE + dense + shexp time inside
/// vrest, so the box window can split the vrest bucket without re-deriving it. Ticks only
/// on the batched arm, like its siblings; vrest's own definition (verify - vkda - vmla)
/// stays unchanged for cross-window comparability — the line prints vffn INSIDE vrest.
pub(crate) static V_FFN_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Per-burst phase counters (ns) — the verify-toll dataset the dspark loop banks under
/// its `stats` clocks (`ns_draft/ns_verify/...`, dflash.rs).
#[derive(Default)]
pub(crate) struct SpecPhaseNs {
    pub(crate) draft: u64,
    pub(crate) verify: u64,
    pub(crate) accept: u64,
    pub(crate) roll: u64,
    pub(crate) maint: u64,
    pub(crate) rounds: u64,
}

impl SpecPhaseNs {
    /// Phase-boundary clock: drain the engines' streams so the elapsed time since the last
    /// clock is attributable to the phase that just ran (the dspark `clock(stats, e)`
    /// contract). `eh` == `e` when the ppN door is shut; under a split the verify walk's own
    /// terminal drain already covers the stage streams transitively, so syncing the primary
    /// and head streams here bounds every phase that runs on them.
    pub(crate) fn clock(e: &Engine, eh: &Engine) -> std::time::Instant {
        let _ = e.stream().synchronize();
        if !std::ptr::eq(e, eh) {
            let _ = eh.stream().synchronize();
        }
        std::time::Instant::now()
    }

    /// One line per burst under `tag`; the level-2 verify sub-split under `tag_v` — both
    /// tags belong to the CALLING family so its banked receipts keep their grep shape.
    pub(crate) fn emit(&self, tag: &str, tag_v: &str, k: usize) {
        if self.rounds == 0 {
            return;
        }
        let ms = |ns: u64| ns as f64 / 1e6;
        let per = |ns: u64| ns as f64 / 1e6 / self.rounds as f64;
        let total = self.draft + self.verify + self.accept + self.roll + self.maint;
        eprintln!(
            "[{tag}] rounds={} k={k} total={:.2}ms | draft={:.2} verify={:.2} \
             accept={:.2} roll={:.2} maint={:.2} | per-round ms: draft={:.3} verify={:.3} \
             accept={:.3} roll={:.3} maint={:.3} total={:.3}",
            self.rounds,
            ms(total),
            ms(self.draft),
            ms(self.verify),
            ms(self.accept),
            ms(self.roll),
            ms(self.maint),
            per(self.draft),
            per(self.verify),
            per(self.accept),
            per(self.roll),
            per(self.maint),
            per(total),
        );
        // Level-2 verify sub-split (lane/glm5-verify-batch): batched-class vs
        // sequential-class shares. vrest = the verify phase minus the mixer buckets
        // (hc glue + FFN/MoE + head); scan = the sequential KDA chain inside the
        // batched call. Drained per burst so consecutive bursts stay comparable.
        if spec_trace_level() >= 2 {
            let vkda = V_KDA_NS.swap(0, Ordering::Relaxed);
            let scan = V_KDA_SCAN_NS.swap(0, Ordering::Relaxed);
            let vmla = V_MLA_NS.swap(0, Ordering::Relaxed);
            let vffn = V_FFN_NS.swap(0, Ordering::Relaxed);
            let vrest = self.verify.saturating_sub(vkda + vmla);
            eprintln!(
                "[{tag_v}] rounds={} k={k} | per-round ms: vkda={:.3} (scan={:.3}) \
                 vmla={:.3} vrest={:.3} (vffn={:.3})",
                self.rounds,
                per(vkda),
                per(scan),
                per(vmla),
                per(vrest),
                per(vffn),
            );
        }
    }
}

/// `MEMRA_SPEC_PROF=1` (lane/b200-spec-ttft-20260902): the ONCE-PER-REQUEST first-token
/// phase profile for a served spec session — every phase between the session's prime and
/// the first streamed token, in ms. Distinct from `MEMRA_SPEC_TRACE` (per-burst round
/// SHARES): this instrument answers "where did the first-token latency go on THIS
/// request", so it buckets the one-time costs the round trace cannot see (cache alloc,
/// target prime, boundary draw, drafter KV alloc, the round-1 drafter prime over the
/// prompt) and the first burst's wall. DEFAULT OFF BY DESIGN: phase boundaries synchronize
/// the stream (the `SpecPhaseNs::clock` contract), so the traced first burst is a little
/// slower than the untraced one; the line attributes, the untraced TTFT claims. Read once
/// per process.
pub fn spec_prof_on() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("MEMRA_SPEC_PROF").as_deref() == Ok("1"))
}

/// The first-token phase buckets (ms) one served spec session carries until the worker
/// prints its `[spec-prof]` line after the first burst. Every field is a wall interval
/// bounded by stream drains on both sides, so the buckets are device-inclusive and
/// additive; a bucket the route never runs stays 0.0.
#[derive(Default, Debug, Clone, PartialEq)]
pub struct SpecFirstTokenProf {
    /// Session creation: the trunk cache allocation (`pp::new_cache_planned`).
    pub cache_alloc_ms: f64,
    /// Session creation: the target prime over the prompt (`prime_cache`), INCLUDING the
    /// host-staged tap DtoHs the DFlash2 sink takes inside the walk and the prime's own
    /// full-vocab logits readback.
    pub prime_ms: f64,
    /// Session creation: the prompt-boundary prefix capture (0 with the prefix door shut).
    pub capture_ms: f64,
    /// Session creation: the boundary token draw (sampled: host logits HtoD + filtered
    /// Gumbel + readback; greedy: host argmax).
    pub anchor_ms: f64,
    /// Session creation: the draft-source state build — DFlash2: `DflashKv::new` at the
    /// session ctx (2 x n_layer x (ctx + block) x n_kv x head_dim x f32, uninit);
    /// native MTP: the batched plane fill over the prompt.
    pub draft_alloc_ms: f64,
    /// Round 1: the DFlash2 drafter's ctx ingest of the WHOLE prompt's feature rows
    /// (HtoD of the host-staged taps + `ctx_features` + per-layer k/v projections). The
    /// only prompt-length-linear cost after the target prime. 0 on the native arm.
    pub draft_prime_ms: f64,
    /// Round 1: draft production after the ingest (block forward + lm_head over the
    /// mask-fill rows + selector walk; native arm: the MTP chain).
    pub first_draft_ms: f64,
    /// Round 1: the t=K+1 verify walk.
    pub first_verify_ms: f64,
    /// Round 1: the accept walk (device argmaxes / rejection sampling + readbacks).
    pub first_accept_ms: f64,
    /// Round 1: the trunk rollback to the accepted prefix.
    pub first_roll_ms: f64,
    /// Round 1: draft-source maintenance (tap drain / plane reset + re-seed).
    pub first_maint_ms: f64,
    /// Round 1: tokens the round committed (j accepted drafts + the bonus).
    pub first_round_tokens: usize,
    /// First burst: wall from burst entry to return (no extra drains; the burst's tokens
    /// are host-visible at return, so this IS the un-instrumented wall of the burst).
    pub first_burst_ms: f64,
    /// First burst: rounds it ran and tokens it returned (anchor included).
    pub first_burst_rounds: usize,
    pub first_burst_tokens: usize,
}

/// Phase clock for [`SpecFirstTokenProf`]: `lap` drains the engines' streams and returns
/// the ms since the previous lap (or `start`). Only ever constructed with the profile on.
pub(crate) struct ProfClock {
    t: std::time::Instant,
}

impl ProfClock {
    pub(crate) fn start(e: &Engine, eh: &Engine) -> Self {
        Self {
            t: SpecPhaseNs::clock(e, eh),
        }
    }
    pub(crate) fn lap(&mut self, e: &Engine, eh: &Engine) -> f64 {
        let now = SpecPhaseNs::clock(e, eh);
        let ms = now.duration_since(self.t).as_secs_f64() * 1e3;
        self.t = now;
        ms
    }
}

#[cfg(test)]
mod tests {
    use super::{parse_level, spec_trace_level_from};

    #[test]
    fn level_resolution_honors_both_names_general_wins() {
        // off by default; junk values are off (the original match-arm law)
        assert_eq!(spec_trace_level_from(None, None), 0);
        assert_eq!(spec_trace_level_from(Some("x"), None), 0);
        // either name alone
        assert_eq!(spec_trace_level_from(Some("1"), None), 1);
        assert_eq!(spec_trace_level_from(Some("2"), None), 2);
        assert_eq!(spec_trace_level_from(None, Some("1")), 1);
        assert_eq!(spec_trace_level_from(None, Some("2")), 2);
        // agreement and (loud) general-wins disagreement
        assert_eq!(spec_trace_level_from(Some("2"), Some("2")), 2);
        assert_eq!(spec_trace_level_from(Some("1"), Some("2")), 1);
        assert_eq!(spec_trace_level_from(Some("2"), Some("1")), 2);
        // a junk general value never masks a valid alias
        assert_eq!(spec_trace_level_from(Some("x"), Some("1")), 1);
        assert_eq!(parse_level(Some("0")), None);
    }
}
