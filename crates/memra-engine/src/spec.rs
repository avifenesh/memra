//! Qwen3.5 MTP (NextN) greedy speculative decode (research/mtp/MTP-PLAN.md §A/§B/§C/§D).
//!
//! Greedy spec decode is MATHEMATICALLY EXACT: the accepted+bonus token stream is token-for-token
//! identical to plain greedy `generate`. This module provides:
//!   - `mtp_head_forward`  (§A, T=1): one NextN draft-token forward.
//!   - `decode_step_t`     (§D.3, T=K+1): batched target verify forward, all-column logits.
//!   - `generate_spec`     (§B): the draft/verify/accept/rollback orchestrator.
//!     Cache snapshot/rollback lives in cache.rs (§D.4). The MTP head uses its OWN scratch KV (§D.6),
//!     PERSISTENT over the committed sequence (see `MtpScratch`).

use crate::Engine;
use crate::cache::{Cache, KvLayer};
use crate::forward::argmax;
use crate::hybrid::{FullAttnLayer, HybridModel, LinearAttnLayer, Mixer, MtpHead};
use cudarc::driver::CudaSlice;
use memra_gguf::config::SwigluClamp;
use std::sync::atomic::{AtomicU64, Ordering};

/// Parse the documented `MEMRA_SPEC_REPLAY=1` rollback seam.
///
/// Keep this shared with serving admission so `=0` cannot select replay in one
/// layer while another layer treats it as disabled.
pub fn spec_replay_env_on(value: Option<&str>) -> bool {
    value == Some("1")
}

pub fn spec_replay_env_enabled() -> bool {
    let value = std::env::var("MEMRA_SPEC_REPLAY").ok();
    spec_replay_env_on(value.as_deref())
}

/// step35 dcw draft-chain door (lane/step37-draft-graph-20260829). ON routes the step35 MTP
/// block's draft attention through the WINDOWED device-counter family
/// (`append_kv_quantized_dcw` + `fa_decode_dcw`, the step TP graph arc's kernels), which
/// derives the SWA view entirely from device state (len_d, base_d, window): exactly the view
/// offset the old capture refusal said `fa_decode_dc` could not express. BOTH draft modes
/// switch together: eager and captured run the ONE launcher at the ONE bucket
/// (min(cap, window)), so graph-vs-eager draft parity holds by construction (the
/// `mtp_full_attn_dc` precedent).
///
/// DEFAULT ON since lane/step37-draft-graph-serving-20260830: the 20260829 lane shipped it
/// OFF because it enabled nothing at the shipping head count (capture was structurally
/// unreachable at heads=3); with the multi-head chain capture and the in-graph filtered
/// sampler landed, this door is the kernel prerequisite for the captured chain on the
/// QUALIFIED serving shape, and the exactness battery (greedy K=1..8 identity, per-K
/// acceptance identity, seeded sampled twins) banks on the ON arm. Rollback seam:
/// MEMRA_STEP35_DRAFT_DCW=0 restores the host-len eager arm (`mtp_step35_attn`) plus the
/// named capture refusal, byte-for-byte the pre-lane serving; no state survives restart.
fn step35_draft_dcw_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("MEMRA_STEP35_DRAFT_DCW").as_deref() != Ok("0"))
}

/// Multi-head MTP draft-chain capture door (lane/step37-draft-graph-serving-20260830,
/// default ON — receipts in the lane RESULTS). ON lets the step-modulo prefix-replay chain
/// (`mtp_extra` non-empty, the step37 3-head shipping shape) capture per-head single-row
/// CUDA graphs and replay them in the exact eager launch order; the chain POLICY (head
/// selection, prefix length, seed history) stays host-side, so graph-vs-eager drafts are
/// bit-identical by construction. A failed capture degrades LOUDLY to the eager chain (the
/// draft-graph WARN contract). OFF (=0) keeps the eager chain as the only multi-head path —
/// the pre-lane serving byte-for-byte. Single-head capture is untouched by this door.
fn mtp_chain_graph_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("MEMRA_MTP_CHAIN_GRAPH").as_deref() != Ok("0"))
}

/// In-graph FILTERED sampled draft door (lane/step37-draft-graph-serving-20260830, default
/// ON — receipts in the lane RESULTS). ON widens the sampled draft-graph capture from the
/// pure-temp regime to every truncation-filtered regime (top_k / top_p / min_p): the capture
/// body runs `filter_stats` + `gumbel_perturb_filtered_ctr` IN-GRAPH, so the draft draws
/// from the SAME filtered distribution the verify's accept test reconstructs (the
/// graph-s-key exactness law, now satisfied inside the graph instead of by refusing it).
/// Penalties stay eager either way (the history varies per round and cannot be baked).
/// The pure-temp capture body is UNTOUCHED by this door (byte-identical to the pre-lane
/// graph). OFF (=0) restores the pure-temp-only capture guard: filtered requests draft
/// eager, byte-for-byte the pre-lane behavior.
fn spec_graph_filtered_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("MEMRA_SPEC_GRAPH_FILTERED").as_deref() != Ok("0"))
}

fn parse_prime_trows_width(value: Option<&str>) -> Result<usize, String> {
    let Some(raw) = value else {
        return Ok(8);
    };
    let width = raw
        .parse::<usize>()
        .map_err(|_| format!("MEMRA_PRIME_TROWS_T must be an integer in 2..=8, got {raw:?}"))?;
    if !(2..=8).contains(&width) {
        return Err(format!("MEMRA_PRIME_TROWS_T must be in 2..=8, got {width}"));
    }
    Ok(width)
}

#[cfg(test)]
mod prime_trows_width_tests {
    #[test]
    fn width_defaults_to_eight_and_refuses_invalid_operator_values() {
        assert_eq!(super::parse_prime_trows_width(None), Ok(8));
        assert_eq!(super::parse_prime_trows_width(Some("2")), Ok(2));
        assert_eq!(super::parse_prime_trows_width(Some("8")), Ok(8));
        for invalid in ["", "1", "9", "32", "wide"] {
            let err = super::parse_prime_trows_width(Some(invalid)).unwrap_err();
            assert!(err.contains("MEMRA_PRIME_TROWS_T"), "{err}");
            assert!(err.contains("2..=8"), "{err}");
        }
    }
}

/// One compact, anchor-bounded DSpark supervision record. `tokens[0]` is the anchor at p and
/// `hidden` is its predecessor carrier h[p-1], matching the live NextN/DSpark pairing. Target
/// rows p..p+gamma-1 score tokens p+1..p+gamma. They are the full-target softmax's top-k
/// entries; `target_tail_probs[j]` is the probability mass outside those rows. All flattened
/// target arrays are `[gamma, top_k]` in row-major order.
pub struct DsparkAnchorRecord {
    pub position: usize,
    pub hidden: Vec<f32>,
    pub tokens: Vec<u32>,
    pub target_top_ids: Vec<u32>,
    pub target_top_logits: Vec<f32>,
    pub target_top_probs: Vec<f32>,
    pub target_tail_probs: Vec<f32>,
}

#[allow(clippy::type_complexity)] // allow: one-shot composite type; naming it would hide the shape that matters at the call site
fn dspark_sparse_softmax_topk(
    logits: &[f32],
    top_k: usize,
    temperature: f32,
) -> Result<(Vec<u32>, Vec<f32>, Vec<f32>, f32), Box<dyn std::error::Error>> {
    if logits.is_empty() || top_k == 0 || top_k > logits.len() || temperature <= 0.0 {
        return Err("invalid DSpark sparse-softmax shape or temperature".into());
    }
    if logits.iter().any(|value| !value.is_finite()) {
        return Err("DSpark target logits contain a non-finite value".into());
    }
    let mut ranked: Vec<(u32, f32)> = logits
        .iter()
        .copied()
        .enumerate()
        .map(|(index, value)| (index as u32, value))
        .collect();
    let compare = |left: &(u32, f32), right: &(u32, f32)| {
        right.1.total_cmp(&left.1).then(left.0.cmp(&right.0))
    };
    ranked.select_nth_unstable_by(top_k - 1, compare);
    ranked[..top_k].sort_unstable_by(compare);

    let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let inv_temperature = 1.0f64 / temperature as f64;
    let denominator: f64 = logits
        .iter()
        .map(|value| (((*value - max_logit) as f64) * inv_temperature).exp())
        .sum();
    let ids: Vec<u32> = ranked[..top_k].iter().map(|(index, _)| *index).collect();
    let top_logits: Vec<f32> = ranked[..top_k].iter().map(|(_, value)| *value).collect();
    let top_probs: Vec<f32> = top_logits
        .iter()
        .map(|value| ((((value - max_logit) as f64) * inv_temperature).exp() / denominator) as f32)
        .collect();
    let top_mass: f64 = top_probs.iter().map(|value| *value as f64).sum();
    let tail = (1.0f64 - top_mass).clamp(0.0, 1.0) as f32;
    Ok((ids, top_logits, top_probs, tail))
}

fn flatten_dspark_rows<T>(
    rows: Vec<Option<Vec<T>>>,
    position: usize,
    label: &str,
) -> Result<Vec<T>, Box<dyn std::error::Error>> {
    let mut flattened = Vec::new();
    for (slot, row) in rows.into_iter().enumerate() {
        flattened.extend(
            row.ok_or_else(|| format!("missing DSpark {label} at {position} slot {slot}"))?,
        );
    }
    Ok(flattened)
}

/// H-SEED CONVENTION (MEMRA_SPEC_HPOST=1): feed the MTP head the POST-norm hidden — trunk rows
/// hand over `output_norm(x)` and the draft chain recurrence hands over `shared_head_norm(h_nextn)`
/// (= final_h) — matching the reference engines: llama.cpp #24025 ("qwen35: use post-norm hidden
/// state for MTP", t_h_nextn is taken AFTER the final norm in both trunk and MTP graphs) and
/// SGLang's qwen3_5_mtp (spec_info.hidden_states = the target model's post-norm output). memra's
/// historical convention (default, MTP-PLAN §A) is PRE-norm x. Draft-quality-only: exactness is
/// the verify's job either way; acceptance arbitrates. OnceLock: read once, hot-loop safe.
/// `MEMRA_SPEC_HEAD_ROWS=1` — batch the verify tail's LM head over its t columns instead of running
/// it at m=1 once per column. See the call site in `decode_step_t_core_stream` for why the batched
/// form is the same per-row arithmetic (the bf16/q8 rows twins, not cuBLASLt) and what it costs
/// today: the head is re-streamed t times per verify pass. Default off until the byte tape says so.
pub(crate) fn head_rows_on() -> bool {
    static ENV: std::sync::OnceLock<Option<bool>> = std::sync::OnceLock::new();
    crate::step37_door(&ENV, "MEMRA_SPEC_HEAD_ROWS")
}

/// The serving walk's own doors, tri-stated the same way (owner flip 2026-08-27): env forces,
/// unset takes the step37 family default. Call sites are the t-row verify walk itself.
pub(crate) fn spec_verify_eager_on() -> bool {
    static ENV: std::sync::OnceLock<Option<bool>> = std::sync::OnceLock::new();
    crate::step37_door(&ENV, "MEMRA_SPEC_VERIFY_EAGER")
}

pub(crate) fn spec_verify_tcol_on() -> bool {
    static ENV: std::sync::OnceLock<Option<bool>> = std::sync::OnceLock::new();
    crate::step37_door(&ENV, "MEMRA_SPEC_VERIFY_TCOL")
}

/// NOT family-armed (2026-08-27): the walk's prime leaves its sub-32 TAIL chunk out of the
/// DISTRIBUTED kv, so the server refuses before decode with "cache lengths diverged
/// local=N distributed=floor(N/32)*32" for every prompt whose token count is not a multiple of
/// 32 — i.e. nearly all real traffic. Isolated on the server route: defaults ERR (local=445
/// distributed=416), MEMRA_PRIME_TROWS=0 OK. It was default-OFF before the 2026-08-27 flip and
/// goes back to opt-in until the tail append is fixed and gated ON THE SERVER ROUTE, not just
/// run-gen (run-gen calls decode_step_t on the whole prompt and never exercises this path — the
/// reason a run-gen-only receipt could not see it). The GEMM prime supersedes it on this route.
pub(crate) fn prime_trows_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("MEMRA_PRIME_TROWS").as_deref() == Ok("1"))
}

pub(crate) fn tcol_ffn_on() -> bool {
    static ENV: std::sync::OnceLock<Option<bool>> = std::sync::OnceLock::new();
    crate::step37_door(&ENV, "MEMRA_TCOL_FFN")
}

pub(crate) fn spec_hpost() -> bool {
    static H: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *H.get_or_init(|| {
        std::env::var("MEMRA_SPEC_HPOST")
            .map(|v| v != "0")
            .unwrap_or(false)
    })
}

/// LEAN VERIFY (default ON since 2026-07-08; MEMRA_SPEC_LEAN=0 reverts — close35 lane): the verify m-scaling
/// probe + nsys diff showed the verify t-path pays ~1.0ms/call at m=1 over eager decode on the
/// 35B, and the kernels are NOT the cause (dev-MoE identical, kernel-time delta only +179us).
/// The overhead is (a) ~250 extra cuMemsetD8Async/call from `e.zeros()` on buffers every kernel
/// fully overwrites (~0.9ms host issue + ~0.35ms GPU) and (b) the t=1 FA rows dispatch (rows_v2 +
/// combine_rows, +50us vs the eager fa_decode pair). This flag switches (a) fully-overwritten
/// verify buffers to `e.uninit` (identical bytes: every element is written before read) and
/// (b) t==1 verify FA to the eager `fa_decode` entry (byte-identical: kernel-check pins the
/// rows-vs-loop identity and the per-row loop at t=1 IS fa_decode on the same q). Gates arbitrate.
pub(crate) fn spec_lean() -> bool {
    static L: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    // DEFAULT ON since 2026-07-08 (MEMRA_SPEC_LEAN=0 reverts): bit-identical (buffers fully
    // overwritten; gates green incl maxdiff-identical run-gen) and measured +2.4% e2e p3 /
    // +1.5% p2 at the daily 35B config. m=1 verify now costs eager-decode parity.
    *L.get_or_init(|| {
        std::env::var("MEMRA_SPEC_LEAN")
            .map(|v| v != "0")
            .unwrap_or(true)
    })
}

/// SMALL-M BATCHED VERIFY (default ON since 2026-07-09; MEMRA_SPEC_M2=0 reverts — lane/spec-m2): extend the
/// batched linear-attn verify arm down to t=2 and batch the MoE dev token loop over a
/// grid.z=token axis at every verify t. The close35 m-scaling probe put the m=2 verify tier at
/// x1.54 of m=1 (llama x1.14); the per-column linear chain (t<3) and the serial MoE dev token
/// loop are the two launch-structure causes. Both changes are LAUNCH-STRUCTURE ONLY:
/// (a) the batched conv's t<pad ring update is pure copies (ssm_conv_ring_rebuild from a cloned
///     ring — the ring stores raw input columns); every arithmetic kernel is the same one the
///     t>=3 arm already runs (matmul_decode_exact bit-identical at m=2-4, gdn_scan's internal
///     t-loop == chained T=1 steps);
/// (b) the MoE dev-rows twins run the serial loop's per-token warp program with tok-offset
///     pointers (same sel/w/aq/ad bytes, same dot order, same slot-ordered FMA chain).
/// Gates arbitrate: run-spec K=1..8 self-consistency (35B+9B), kernel-check, run-gen argmax.
pub(crate) fn spec_m2() -> bool {
    static M: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    // DEFAULT ON since 2026-07-09 (MEMRA_SPEC_M2=0 reverts): launch-structure only — t=2
    // batched linear arm (ring-roll copies, zero new FP order) + MoE dev-rows kernels
    // (grid.z=token, 4 launches/layer at any verify t). Acceptance bit-identical at every K;
    // 35B p2 +3.4% / p3 +3.6%; the profitable-K plateau widens (new optimum K=3 at 223).
    *M.get_or_init(|| {
        std::env::var("MEMRA_SPEC_M2")
            .map(|v| v != "0")
            .unwrap_or(true)
    })
}
pub(crate) fn spec_stream() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("MEMRA_SPEC_STREAM").as_deref() == Ok("1"))
}
pub(crate) fn spec_stream_m() -> usize {
    static M: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *M.get_or_init(|| {
        std::env::var("MEMRA_SPEC_STREAM_M")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(4)
    })
}
pub(crate) fn spec_devacc() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("MEMRA_SPEC_DEVACC").as_deref() == Ok("1"))
}
/// Engine-bundle slice 2 (DSF-ROUNDCOST-20260820 §1.1 host/device round trips + §2 rows 2-3),
/// DEFAULT ON (`MEMRA_DSPARK_DEFER_READBACK=0` reverts): the dspark round's draft-chain DtoH
/// is DEFERRED past verify dispatch and merged with the verify-argmax readback into ONE host
/// sync (2 blocking DtoH/round -> 1). Verify embeds DEVICE tokens (`chain_d`) through the
/// resident embed table — `embed_gather_u32_t`, bit-identical rows to the host gather by its
/// own pinned contract. The host therefore dispatches snap + the whole verify while the DRAFT
/// is still executing, instead of blocking ~1.7 ms on the chain and letting the device drain.
/// Ladder arm only: the confidence policies size vt from a pre-verify head readback (their
/// chain readback merges into that same sync instead). Exactness unchanged BY CONSTRUCTION —
/// same tokens, same kernels, same order; E2E + accept-bank gates arbitrate.
pub(crate) fn dspark_defer_readback_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("MEMRA_DSPARK_DEFER_READBACK")
            .map(|v| v != "0")
            .unwrap_or(true)
    })
}
/// Engine-bundle slice 1 (DSF-ROUNDCOST-20260820 §1.1, lane/dspark-engine-bundle-20260820),
/// DEFAULT ON (`MEMRA_STATE_COPY_BATCH=0` reverts): batch the dspark round's GDN state
/// snapshot and partial-accept restore into single `copy_batch_uniform_f32` launches
/// instead of ~2 memcpy dispatches (+2 alloc_zeros on the snap side) per linear layer per
/// round — measured 0.67 ms/round snap + 0.25 ms/round commit of pure dispatch on the q38
/// route. Launch-structure only: bytes, buffers and stream order are unchanged, so
/// acceptance and streams stay bit-identical (E2E-gated on the B1 packs).
pub(crate) fn state_copy_batch_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("MEMRA_STATE_COPY_BATCH")
            .map(|v| v != "0")
            .unwrap_or(true)
    })
}
/// Engine-bundle slice 3 + fa-execupdate slice 4c (DSF-ROUNDCOST-20260820 §5 rank 1),
/// DEFAULT OFF — `MEMRA_DSPARK_VERIFY_GRAPH=1` opts in: per-(segment, vt) CUDA graphs
/// for the LINEAR-layer runs, plus the full-verify single graph per (vt, rung) when a
/// round's rows all ride one seqs rung — see [`DsparkVerifyGraphs`]. Requires the
/// slice-2 deferred path (device tokens); the eager walk is the byte-identical fallback.
///
/// MEASURED disposition (box6 card0, agentic pack, 2026-08-20, both slices): exactness
/// holds everywhere (ALL EXACT, accept lines byte-match the banks, ckpt-gate oracle
/// green over the graph + slab-commit paths). Slice-3's AUTO_FREE launch-scan limiter
/// (25.6 us x 16 launches ≈ 0.41 ms/round) is FIXED — the captured bodies' alloc nodes
/// are balanced by in-graph frees (census 84/84 per segment, 1776/1776 full) so graphs
/// instantiate USE_NODE_PRIORITY and the scan is gone. What remains at gate scale:
/// segment graphs +0.1 tok/s over the batched-rows default (114.4 vs 114.3 x5
/// interleaved — the linear launch overhead was only ~0.1 ms); the FULL-verify graph is
/// NET NEGATIVE at gate scale (110.6 vs 114.2: ~14-21 (vt, rung) captures/process at
/// 2 full-walk executions + ~2.9k-node instantiate each eat far more than the ~0.2-0.3
/// ms/round of remaining launch overhead). The orchestration ceiling of §1.3 is spent —
/// the fa/append recovery landed DEFAULT-ON as the batched rows arm
/// (`dspark_fa_rows_on`), not as a graph. The serve-lifetime cell (DSF-ROUNDCOST §9,
/// nj-ws-solo) measured the amortization: crossover K≈33 requests, steady −0.246
/// ms/round, −1.25% session wall over 240 requests — and the graphs-serve lane wired
/// the door into the session arm (`dspark_spec_session_burst`) as a model-owned
/// capture pool shared across sessions. Stays opt-in pending the owner's default-ON
/// ratification on the serve-surface battery.
pub(crate) fn dspark_verify_graph_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("MEMRA_DSPARK_VERIFY_GRAPH").as_deref() == Ok("1"))
}
/// MTP-ROUTE verify graphs, DEFAULT ON for the GDN+MoE family since 2026-08-23
/// (`MEMRA_SPEC_VERIFY_GRAPH=0` is the kill switch, `=1` opts other families in).
///
/// The slice-4c capture already lived inside `qwen35_verify_tparallel` and said so in its own
/// comment — "stream rides the qwen35moe burst, graphs ride the dspark route" — with no caller
/// on this route. The MTP spec round is that caller.
///
/// WHY it is worth a default (receipts: `research/orndecode-20260822/VGRAPH.md`). With
/// `MEMRA_SPEC_PHASE=1` this route's round reads verify-ISSUE 44-58% and verify-WAIT **0.0%**:
/// the host is never waiting for the device, it is spending its own time launching the trunk.
/// Replay collapses that into one graph launch and the phase all but disappears (55-62 ms ->
/// 8-10 ms per burst).
///
/// MEASURED, two host generations, forced ON/OFF, balanced 4+4 boots in both orders:
///   * current-generation host (9950X, the serving class): OFF 266.0-266.5, ON 318.8-319.5
///     tok/s — **+19.7%**, no overlap, sub-1% spread per arm; per-round 6.9 -> 5.7 ms.
///   * Zen 3 host: +3-9% (that rig's own clock drift is wider than the effect, so the ratio
///     comes from per-round phase totals, which are internal to each boot).
///     The ON arm lands at ~320 tok/s on BOTH hosts while OFF tracks host speed — the arm moves
///     the round off the host and onto the device, which is the whole point.
///
/// EXACTNESS is structural (same kernels, same order) and gated anyway: a fixed-seed SAMPLED
/// completion hashes identically ON vs OFF **and across both hosts** (`08941d5bb9762b21`),
/// greedy seed-pinned likewise, `run-spec` K=1..8 PASS on both arms with identical acceptance
/// at every K, kernel-check ALL GREEN.
///
/// SCOPE, deliberately narrow: default ON only where it was measured — the GatedDeltaNet +
/// MoE family (`vgraph_family_default`). Qwen3.8-27B is GDN + DENSE mlp and would otherwise
/// inherit this default unmeasured, which is the family-by-family law this repo keeps; it can
/// opt in with `=1` once it has its own interleave. Also never armed together with
/// ROUND-STREAM, and a round wider than the pool declines it for the eager walk.
pub(crate) fn spec_verify_graph_env() -> Option<bool> {
    static ON: std::sync::OnceLock<Option<bool>> = std::sync::OnceLock::new();
    *ON.get_or_init(
        || match std::env::var("MEMRA_SPEC_VERIFY_GRAPH").as_deref() {
            Ok("1") => Some(true),
            Ok("0") => Some(false),
            _ => None,
        },
    )
}
/// SERVE-ROUTE twin of [`dspark_verify_graph_on`], DEFAULT ON — owner-ratified
/// 2026-08-22 on the §10 serve-lifetime battery (DSF-ROUNDCOST-20260820 §10.3:
/// crossover K=36–43, steady −0.357 ms/round, session wall −1.55..−1.65%, byte-exact
/// 240/240 ×3 pairs, pool bounded at 8,852 MiB under `MEMRA_DSPARK_VG_MAX`). The env
/// stays as the kill-switch: `MEMRA_DSPARK_VERIFY_GRAPH=0` restores the eager walk
/// (byte-identical body); `MEMRA_DSPARK_VG_MAX=0` is the finer freeze valve. The BIN
/// arm keeps its own opt-in default (`dspark_verify_graph_on`): at gate scale the
/// capture toll is never repaid (§8 measured disposition — 14–21 captures over a
/// 256-token run vs the serve session's thousands of rounds), and the two
/// instruments must keep their own measured dispositions rather than share one flag.
pub(crate) fn dspark_verify_graph_serve_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("MEMRA_DSPARK_VERIFY_GRAPH").as_deref() != Ok("0"))
}
/// Capture-count ceiling for the dspark verify-graph pool (graphs-serve lane) — the
/// pool's memory policy STATED instead of silently unbounded. The keyspace is
/// intrinsically finite — segment keys (run_start, vt) ≤ 16 runs x 7 windows, full
/// keys (vt, rung, hi) ≤ 7 windows x the split-rung ladder (8 rungs at 32k ctx), ~168
/// on the q38 export — so the default (256) never engages there; the knob is the
/// safety valve for a future export with a wider ladder. At the ceiling the pool
/// FREEZES: existing keys keep replaying, rounds needing a new capture run the eager
/// walk byte-identically (round-atomic — a partial refusal would mix slab- and
/// cols-stashed layers inside one commit). No eviction by design: destroying a live
/// exec graph re-opens the stale-address class the indirect tables exist to close,
/// and the bounded keyspace makes reclaim worthless.
pub(crate) fn dspark_vg_cap() -> usize {
    static CAP: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *CAP.get_or_init(|| {
        std::env::var("MEMRA_DSPARK_VG_MAX")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(256)
    })
}

/// PROJECTED REMAINING GROWTH of the verify-graph pool, in bytes (lane/hermes-perf-fixes,
/// 2026-08-23 — the admission accounting the "pool dwarfs spec admission reserve" finding
/// asks for). The pool was measured at 8,852 MiB at storm-complete on the q38 export while
/// admission's transient floor (`SPEC_SHRINK_RESERVE`) is 1.5 GiB and never charged for it:
/// sessions admitted while the pool is cold overcommit VRAM the pool WILL hold, because the
/// pool grows monotonically (no eviction by design) and is model-owned across sessions.
///
/// SELF-MEASURING, no per-model constant (generic-model law — the 8,852 MiB is a q38 number
/// and proves nothing about another export): the debt is remaining capture slots x the
/// MARGINAL bytes a capture adds to this device's graph mem pool.
///
/// MARGINAL, NOT MEAN — measured correction (box9 on-box receipt, 2026-08-23). The first
/// version of this used the mean (`reserved / captures`) and the live serve log showed why
/// that is wrong: with the pool's reservation flat at ~33.6 MiB across captures 1..3, the
/// mean-based debt printed **8,556 MB, then 4,261, then 2,830** — it extrapolated capture
/// #1's ONE-TIME shared allocation (staging buffers, stash slabs, pointer tables: sized
/// once per pool, shared by every key) across all 256 slots. An 8.5 GB phantom reserve at
/// boot can refuse admissions that would have fit, which is a worse defect than the
/// under-charge this accounting exists to remove. The marginal reading prices what an
/// ADDITIONAL key actually costs: two observations `(captures, reserved)` give
/// `(r1 - r0) / (c1 - c0)`, which is ~0 on an export whose pool does not grow per key and
/// tracks real growth on one that does.
///
/// BOOTSTRAP (only one observation so far, so growth is unmeasurable): reserve one more
/// pool's worth — `min(remaining x mean, reserved)`. "We have measured `reserved` bytes for
/// `captures` keys; until growth is measurable, assume at most a doubling" is fail-safe in
/// the same direction as the old rule without the 255x extrapolation.
///
/// Before the FIRST capture the debt is 0 (a single capture lands well inside the existing
/// 1.5 GiB floor). `cap` is the intrinsic freeze ceiling (`MEMRA_DSPARK_VG_MAX`; =0 freeze
/// valve => the pool cannot grow => debt 0); at or past the cap the pool FREEZES, so the
/// debt is 0 there too.
pub fn dspark_vg_debt_projection(
    captures: usize,
    cap: usize,
    reserved_bytes: usize,
    prev: Option<(usize, usize)>,
) -> usize {
    if captures == 0 || cap == 0 {
        return 0;
    }
    let remaining = cap.saturating_sub(captures);
    if remaining == 0 {
        return 0;
    }
    match prev {
        // marginal growth between two observations of the same pool
        Some((c0, r0)) if captures > c0 => {
            let marginal = reserved_bytes.saturating_sub(r0) / (captures - c0);
            remaining.saturating_mul(marginal)
        }
        // bootstrap: at most one more pool's worth
        _ => remaining
            .saturating_mul(reserved_bytes / captures)
            .min(reserved_bytes),
    }
}
/// PRE-CAPTURE VRAM RESERVE CHECK door (lane/step37-vram-admission-20260830), DEFAULT ON.
/// A draft-graph capture attempt on a tight card used to be try-and-fail: the 2 warmup
/// forwards + instantiate grew the pool to the edge BEFORE the OOM surfaced, and the
/// "eager fallback" then ran on a card the failed attempt had just exhausted (the owner's
/// single-session second-prompt OOM: capture WARN followed by 28 step-OOM engine errors,
/// device at 5 MiB free). With the gate ON, a capture is attempted only when the device's
/// effective free (driver free + async-pool cached) covers the capture's expected appetite
/// PLUS a post-capture safety floor — otherwise the session falls back to eager EARLY,
/// with headroom intact, through the same LOUD once-per-flip WARN. `=0` restores
/// try-and-fail (diagnostics door; the trim-on-OOM recovery below stays active either way).
pub fn spec_capture_gate_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("MEMRA_SPEC_CAPTURE_GATE").as_deref() != Ok("0"))
}

/// Post-capture safety floor the reserve check keeps free ON TOP of the capture's own
/// appetite: the same measured constant class as the admission transient floor
/// (capture arenas + verify activations — the admit-oom control fit). A capture that
/// would leave less than this behind is not worth its eager-coverage risk.
pub(crate) const CAPTURE_HEADROOM_FLOOR: usize = 1536 << 20;

/// Pure verdict half of the pre-capture reserve check (unit-testable): given the device's
/// driver-free and pool-cached bytes and the capture's expected `need`, returns
/// `Some((required, effective))` when the capture must be REFUSED, `None` when it fits.
pub(crate) fn capture_headroom_verdict(
    driver_free: usize,
    pool_cached: usize,
    need: usize,
    floor: usize,
) -> Option<(usize, usize)> {
    let effective = driver_free.saturating_add(pool_cached);
    let required = need.saturating_add(floor);
    (effective < required).then_some((required, effective))
}

/// Expected device appetite of a draft-graph capture attempt when no measurement exists
/// yet (bootstrap only — the model-owned high-water gauge takes over after the first
/// observed capture). Deliberately conservative and shape-derived, never a per-family
/// constant: per (head, mode) capture the two warmups + capture each walk one head
/// forward whose dominant transients are a handful of `n_embd` rows and one `d_vocab`
/// logits row, retained by the keeper; the sampled tail additionally parks
/// `k` q-slots + perturb/q buffers of `d_vocab` each.
pub(crate) fn draft_capture_bootstrap_estimate(
    heads: usize,
    k: usize,
    d_vocab: usize,
    n_embd: usize,
) -> usize {
    let per_capture = 3usize // 2 warmups + capture body, each retaining its transients
        .saturating_mul(d_vocab.saturating_add(8 * n_embd))
        .saturating_mul(4)
        .max(32 << 20); // instantiate + driver-side graph backing per capture, floor
    let captures = heads.max(1).saturating_mul(2); // interior + last per head
    let sampled_slots = (k.saturating_add(2))
        .saturating_mul(d_vocab)
        .saturating_mul(4);
    captures
        .saturating_mul(per_capture)
        .saturating_add(sampled_slots)
        .max(64 << 20)
}

/// OOM predicate for capture-failure recovery (engine-side twin of the worker's
/// `is_cuda_oom` — the same quoted-text contract).
pub(crate) fn capture_err_is_oom(reason: &str) -> bool {
    reason.contains("CUDA_ERROR_OUT_OF_MEMORY") || reason.contains("out of memory")
}

/// Impure half of the pre-capture reserve check: reads the device, trims the async pool
/// when the driver alone is short but cached blocks would cover it (graph instantiate and
/// cuBLAS workspaces allocate from the DRIVER, not from our pool — a pool sitting on freed
/// blocks starves them), and returns the refusal reason line when the capture must not be
/// attempted. `None` = go ahead.
pub(crate) fn capture_headroom_refusal(e: &Engine, need: usize) -> Option<String> {
    let Ok((driver_free, _total)) = e.ctx().mem_get_info() else {
        return None; // unreadable device: keep the historical try-and-fail behavior
    };
    let pool_cached = e.pool_cached_bytes();
    // A capture may take AT MOST HALF the discretionary headroom: required =
    // 2x appetite + two floors (owner's contract: "fall back to eager EARLY with headroom
    // intact"). Measured escalation on the owner-shape cells: one floor of slack let the
    // capture walk the card to the edge and the burst step-OOM'd immediately; two floors
    // still allowed a capture whose session then OOM'd on its own admission-charged work,
    // because the capture had consumed the memory the charge was counting on. Requiring
    // the appetite TWICE means the card retains a whole capture's worth of room after the
    // capture lands - enough for the session's charged classes and its peers' bursts. The
    // capture is an optimization worth ~2-3 ms of TTFT (draft-graph lane receipts); at the
    // margin it is never worth an OOM incident.
    let floor = CAPTURE_HEADROOM_FLOOR.saturating_mul(2);
    let required_need = need.saturating_mul(2);
    let required = required_need.saturating_add(floor);
    match capture_headroom_verdict(driver_free, pool_cached, required_need, floor) {
        Some((required, effective)) => Some(format!(
            "insufficient VRAM headroom for capture: effective free {}MB (driver {}MB + pool-cached \
             {}MB) < required {}MB (2x appetite {}MB + floor {}MB); capture skipped pre-attempt",
            effective / (1 << 20),
            driver_free / (1 << 20),
            pool_cached / (1 << 20),
            required / (1 << 20),
            need / (1 << 20),
            floor / (1 << 20),
        )),
        None => {
            if driver_free < required && pool_cached > 0 {
                let trimmed = e.pool_trim_to_zero();
                if trimmed > 0 {
                    eprintln!(
                        "[spec] pre-capture pool trim: released {}MB cached back to the driver \
                         (driver free {}MB < required {}MB; instantiate allocates from the driver)",
                        trimmed / (1 << 20),
                        driver_free / (1 << 20),
                        required / (1 << 20),
                    );
                }
            }
            None
        }
    }
}

/// GRAPH-LAUNCH HEADROOM FLOOR (lane/step37-vram-admission-20260830, defect 3 root
/// cause): `cuGraphLaunch` SEGFAULTS inside libcuda (offset +0x27c87f, a null internal
/// dereference at address 0x60) when a captured graph is dispatched into a
/// driver-exhausted card — reproduced on this lane's box with core dumps on BOTH the
/// pre-lane and lane binaries (multi-active step-OOM squeeze; the crashing thread sits in
/// `CudaGraph::launch` inside `generate_spec_inner2`). The eager arms fail RECOVERABLY on
/// the same card (a quoted CUDA OOM the park path handles), so below this driver-free
/// floor every graph arm yields to eager for the round. A named constant, not a knob: the
/// winning value is the default and the guard exists to make a driver segfault
/// unreachable, not to tune anything.
pub(crate) const GRAPH_LAUNCH_MIN_FREE: usize = 256 << 20;

/// Per-round guard for the floor above. Read failure keeps serving (never a false
/// refusal from an unreadable device); one `mem_get_info` (~microseconds) per ~25ms round.
pub(crate) fn graph_launch_headroom_ok(e: &Engine) -> bool {
    match e.ctx().mem_get_info() {
        Ok((free, _total)) => free >= GRAPH_LAUNCH_MIN_FREE,
        Err(_) => true,
    }
}

/// One grep-stable suspension line per ROUTE (each call site holds its own
/// process-lifetime `Once`): every captured-graph launch route below the floor names
/// itself in the tag while keeping the same `graph replay suspended:` key the step37
/// admission lane's squeeze cell greps for. The spec-round guard keeps its original
/// per-generation `[spec]` line; the sweep routes (graph-launch-guard-sweep lane,
/// 2026-08-31) note once per process — presence is what the gates assert, and a
/// suspended round is otherwise byte-identical to its eager twin.
pub(crate) fn graph_replay_suspended_note(route: &str) {
    eprintln!(
        "[{route}] graph replay suspended: driver free below the {}MB launch floor \
         (eager arms serve; cuGraphLaunch segfaults into an exhausted card)",
        GRAPH_LAUNCH_MIN_FREE / (1 << 20)
    );
}

/// Engine-bundle slice 4 (fa-execupdate lane, DSF-ROUNDCOST-20260820 §6 close: "the
/// residual gap lives in the FULL-ATTENTION per-row section"), DEFAULT ON —
/// `MEMRA_DSPARK_FA_ROWS=0` reverts to the per-row loop: when every row of a verify
/// round takes the v4-seqs arm on ONE `fa_split_keys` rung (the straddle law, evaluated
/// at the round's first and last t_kv — both eligibility gates are intervals in t_kv),
/// the qwen35 t-parallel verify's per-row KV-append + fa-decode loop collapses into the
/// z-batched serving twins: ONE `append_quantize_kv_q8_0_q5_1_seqs` + ONE
/// `fa_decode_vec_q_seqs_v4` + ONE combine per full-attention layer, replacing
/// T x (4 dtod row copies + append + 3 memsets + main + combine) launches. Bytes are
/// pinned by the batched-tick increment-2 kernel-check (seqs-vs-per-seq-loop bit
/// identity: per-row T_kv derives in-kernel from pos_seq[z]; splits >= ns_eff write the
/// empty partial the combine never reads, so the shared n_splits_max stride changes no
/// bytes) and re-gated e2e by this lane's battery.
pub(crate) fn dspark_fa_rows_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("MEMRA_DSPARK_FA_ROWS")
            .map(|v| v != "0")
            .unwrap_or(true)
    })
}

/// `t_pred0` for the `MEMRA_DEBUG_SPEC` per-round print, sampled-safe.
///
/// `generate_spec_inner2` fills its `preds` vector ONLY on the greedy path (`if !sampled`), and
/// the per-round debug print was the sole consumer in the sampled arm: `t_pred(0)` survives round
/// 0 (`base == 0` returns `last_pred`) and from round 1 (`base == 1`, a pending bonus) indexes an
/// EMPTY vector — `index out of bounds: the len is 0 but the index is 0`, in the GPU worker
/// thread, which then respawns and reloads weights while the request dies. So any sampled spec
/// request longer than one round used to kill the worker whenever `MEMRA_DEBUG_SPEC` was set:
/// the flag crashed precisely the regime it exists to investigate.
///
/// Fixed at the print site, not inside the closure, so the greedy accept walk keeps its strict
/// indexing (an out-of-range pred there is a real bug and must still be loud).
fn debug_t_pred0(sampled: bool, base: usize, last_pred: u32, preds: &[u32]) -> String {
    if base == 0 {
        return last_pred.to_string();
    }
    match preds.get(base - 1) {
        Some(p) => p.to_string(),
        // sampled: the greedy per-column argmax was never run for this round.
        None => {
            debug_assert!(
                sampled,
                "greedy spec: preds[{}] missing at base {base}",
                base - 1
            );
            "n/a".to_string()
        }
    }
}

/// `MEMRA_SKEY_PROBE=1` — sampled-draft-graph key probe (lane/graph-s-key-exactness-20260819).
///
/// Reports, per burst and per round, which draft chain the sampled arm chose and under which
/// filter regime, plus the ONE observable that separates a legal filtered draft from a stale
/// pure-temp graph replayed under filters: an accept test whose gathered `q` is exactly 0.
/// A draft token sampled from the FILTERED softmax can never gather q=0 (it was drawn from the
/// kept set), so `q=0` in the verify means the draft came from a distribution the verify does
/// not believe in — and `u * 0 < p` then accepts it unconditionally.
///
/// Its own env var, deliberately NOT `MEMRA_DEBUG_SPEC`: that flag panicked the GPU worker on
/// any sampled spec request past round 0 until this lane fixed it (§2 of the bank note).
pub(crate) fn skey_probe() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("MEMRA_SKEY_PROBE").as_deref() == Ok("1"))
}

/// GRAMMAR HOOK for constrained spec decode (lane/constrained-full, 2026-08-03). The engine
/// stays llguidance-agnostic: the server adapts its per-session grammar state behind this
/// trait. CONTRACT (the verify-side truncation rule — token-identical to constrained plain
/// greedy decode): the exactness walk runs UNMASKED first; the hook then (a) truncates
/// acceptance at the first grammar-illegal accepted token, and (b) when the truncation fired
/// or the bonus is illegal, the engine recomputes that slot as the MASKED argmax of the
/// target's own verify column (an unmasked argmax that is grammar-legal IS the masked argmax
/// — masking only removes tokens — so the common case pays nothing). `consume` advances the
/// state with each EMITTED token in order; EOS handling is the implementor's job (skip).
pub trait SpecConstraint {
    /// -inf the current state's banned ids on a HOST logits row (prompt-tail / init-feed
    /// masked argmax).
    fn mask_logits(&mut self, logits: &mut [f32]) -> Result<(), String>;
    /// Packed 32-bit bitset words of the CURRENT state's allowed set (device-mask form).
    fn mask_words(&mut self) -> Result<Vec<u32>, String>;
    /// Is `tok` consumable in the CURRENT state?
    fn is_allowed(&mut self, tok: u32) -> Result<bool, String>;
    /// Advance the state with an emitted token.
    fn consume(&mut self, tok: u32) -> Result<(), String>;

    // --- DRAFT-SIDE MASKING (lane/draft-mask, 2026-08-04) ---
    // The drafter proposed grammar-illegal tokens under tight schemas, so verify-side
    // truncation cut nearly every round (measured acceptance 0.467-0.513 tight vs 0.62-0.82
    // loose, research/constrained-full-20260803). These three methods let the engine mask the
    // DRAFT model's own sampling with the grammar's legal set, so proposals are legal by
    // construction. The state they walk is a SPECULATIVE CLONE of the session matcher — the
    // real state is advanced only by `consume` (emitted tokens), so verify-side truncation
    // stays the correctness backstop and the emitted stream is unchanged by construction
    // (an accepted draft is the target's unmasked argmax AND grammar-legal, hence the masked
    // argmax; a cut slot is recomputed as the masked argmax either way).
    // Default impls = feature OFF (pre-lane behaviour: unmasked drafts).

    /// Is draft-side masking available on this hook? Probed ONCE per burst, before the draft
    /// graph is captured (the mask is an in-graph node — its presence is a capture-time shape).
    fn draft_mask_enabled(&self) -> bool {
        false
    }
    /// Start a draft chain: clone the CURRENT (committed) grammar state into the speculative
    /// slot. Called once per spec round, before the first draft position.
    fn draft_begin(&mut self) -> Result<(), String> {
        Ok(())
    }
    /// Packed 32-bit bitset words of the SPECULATIVE state's allowed set (target-vocab ids),
    /// for the draft position about to be sampled. `None` = draft masking off (no-op).
    fn draft_mask_words(&mut self) -> Result<Option<Vec<u32>>, String> {
        Ok(None)
    }
    /// Advance the SPECULATIVE state with a PROPOSED draft token. `false` = the chain cannot
    /// continue (EOS proposed, or an unmasked position proposed something illegal) — the
    /// engine stops drafting; the token already pushed still goes through verify.
    fn draft_advance(&mut self, _tok: u32) -> Result<bool, String> {
        Ok(false)
    }
}

/// DRAFT-MASK UPLOAD (lane/draft-mask): pull the speculative state's allowed set (TARGET-id
/// space) from the hook, project it into the DRAFT head's vocab space, and upload it into the
/// stable device buffer the draft chain reads. Returns false when the chain must stop drafting:
/// the hook handed out no mask, or NO draft-vocab row is grammar-legal at this position (a
/// trimmed FR-Spec head genuinely cannot propose a legal token there — masking it would leave
/// a fully-banned row whose argmax is meaningless, so the round drafts fewer tokens and the
/// verify emits the masked argmax as usual).
fn upload_draft_mask(
    e: &Engine,
    c: &mut dyn SpecConstraint,
    dst: &mut CudaSlice<u32>,
    d2t: Option<&Vec<u32>>,
    d_vocab: usize,
    words: usize,
) -> Result<bool, Box<dyn std::error::Error>> {
    let Some(tw) = c
        .draft_mask_words()
        .map_err(|e2| format!("constraint: {e2}"))?
    else {
        return Ok(false);
    };
    let bit = |t: usize| -> bool {
        let w = t >> 5;
        w < tw.len() && (tw[w] >> (t & 31)) & 1 == 1
    };
    let mut buf = vec![0u32; words];
    match d2t {
        // TRIMMED draft head: row i proposes target id d2t[i] — permute the mask accordingly.
        Some(map) => {
            for (i, &t) in map.iter().enumerate().take(d_vocab) {
                if bit(t as usize) {
                    buf[i >> 5] |= 1u32 << (i & 31);
                }
            }
        }
        // UNTRIMMED: draft ids ARE target ids; the packed words transfer verbatim (a short
        // mask leaves the padded tail zeroed == banned, same rule as constrained::apply_mask).
        None => {
            let n = tw.len().min(words);
            buf[..n].copy_from_slice(&tw[..n]);
        }
    }
    if buf.iter().all(|w| *w == 0) {
        return Ok(false);
    }
    e.htod_u32_into(dst, &buf)?;
    Ok(true)
}

/// Keep the full token-embedding table in host memory and upload only the rows needed by each
/// MTP/verify step. This is an exact memory-capacity seam for very large BF16 vocab tables: host
/// gather expands the same source bits to f32, and only O(T*n_embd) bytes cross PCIe per step.
/// CUDA-graph/round-stream draft paths require device token ids and therefore stay disabled.
pub(crate) fn spec_host_embd() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("MEMRA_SPEC_HOST_EMBD").as_deref() == Ok("1"))
}

/// VERIFY-TIER TRUNK LAUNCH-FUSION (default ON since 2026-07-09; MEMRA_SPEC_FUSED_T=0 reverts — lane/close35b): extend
/// the t=1 fused2/fused3 Q8_0 trunk launches to the batched verify tier (t=2-4, the K=1..3
/// verify shapes). At t>1 the trunk pairs/triples (35B wqkv+wqkv_gate, wq/wk/wv,
/// gate_shexp+up_shexp) each run a separate `matmul_decode_exact` — one q8_1 re-quantize of the
/// SAME activation plus one _b2/_b4 launch per tensor. The fused twins share ONE quantize and
/// ONE launch per group; per (tensor,token,row) the kernel body is q8_0_mmvq_batched verbatim
/// with the identical row mapping -> BIT-IDENTICAL by construction (kernel-check pins it,
/// run-spec K=1..8 + acceptance identity arbitrate e2e).
pub(crate) fn spec_fused_t() -> bool {
    static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    // DEFAULT ON since 2026-07-09 (MEMRA_SPEC_FUSED_T=0 reverts): verify t=2-4 trunk launch-fusion
    // (fused2/fused3 Q8_0 batched twins, bit-identical by construction — m=1 block-offset split on
    // the batched body). m=2 marginal token 2117->1762us; 35B daily: p3 +3.7% (crosses llama), p2 +5%.
    *F.get_or_init(|| {
        std::env::var("MEMRA_SPEC_FUSED_T")
            .map(|v| v != "0")
            .unwrap_or(true)
    })
}

/// zeros/uninit switch for verify-path buffers that are FULLY OVERWRITTEN before any read.
/// Only call this on such buffers — the lean contract is "identical bytes by construction".
/// TOKEN-ID GUARD for every id that reaches an embed gather (#87 family).
///
/// A device argmax seeds its running index with 0x7FFFFFFF and replaces it only through
/// comparisons, all of which are FALSE against NaN. An all-NaN logits row therefore returns
/// the sentinel, and the next thing done with a token id is `embed_row(id)` — table +
/// ~4.6 TB, never mapped, an MMU fault that kills the CUDA context for the whole process
/// (research/pp2spec-crash-20260807). The draft chain and the GREEDY verify walk already
/// trap this; the SAMPLED verify bonus, the boundary sampler and the replay arm's last_pred
/// did not, which is why the recoverable fault on the greedy instrument is a TERMINAL one on
/// the vendor-default sampled shape we actually serve.
pub(crate) fn guard_vocab_token(
    tok: u32,
    n_vocab: usize,
    what: &str,
) -> Result<u32, Box<dyn std::error::Error>> {
    if (tok as usize) >= n_vocab {
        return Err(format!(
            "{what}: token id 0x{tok:08x} >= n_vocab {n_vocab} — an all-NaN logits row left \
             the device argmax's init sentinel in place; refusing to dereference the embed \
             row (#87 trap)"
        )
        .into());
    }
    Ok(tok)
}

/// SPEC NaN-ORIGIN SCAN (`MEMRA_SPEC_NAN_SCAN=1`, DEFAULT OFF, diagnostic only).
///
/// The `#87` trap reports an all-NaN VERIFY logits column, which says the poison reached the
/// head but not where it entered. With the scan armed the verify walk syncs and reads back
/// every layer's output, so the FIRST layer whose residual carries a NaN names itself with the
/// round's row and position. Off by default and never on a serving path: it costs one host
/// sync + one `t*n_embd` D2H per layer, and the syncs change scheduling (so a run that stops
/// reproducing under the scan is itself a datum, not an all-clear).
///
/// Rollback seam: unset `MEMRA_SPEC_NAN_SCAN` (or set it to 0). Every call site is behind
/// `spec_nan_scan()`, so the default path keeps the exact launch sequence it had.
pub(crate) fn spec_nan_scan() -> bool {
    spec_nan_scan_level() > 0
}

/// `MEMRA_SPEC_NAN_SCAN` as a LEVEL, not a boolean. `1` scans each layer's residual, which
/// names the layer. `2` also scans INSIDE the t-column layer body — the per-column attention
/// output, the deferred-column o-proj/fa2 join, the post-attention norm and the routed-MoE
/// output — because "layer 20 poisons row 0" does not say whether the attention or the routed
/// MoE produced it, and those are different bugs with different fixes.
pub(crate) fn spec_nan_scan_level() -> u8 {
    static LVL: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *LVL.get_or_init(|| match std::env::var("MEMRA_SPEC_NAN_SCAN").as_deref() {
        Ok("1") => 1,
        Ok("2") => 2,
        _ => 0,
    })
}

/// Read back `[rows, cols]` and fail with the first NaN's coordinates. `what` names the
/// producer (layer index, walk arm) so the error line is the localization.
/// VERIFY-ARM RECEIPT (rides `MEMRA_SPEC_NAN_SCAN>=1`, bounded to 200 lines).
///
/// Names, per trunk layer, WHICH attention arm the t-column walk actually took. This exists
/// because the level-1 residual scan below sat only on the non-fused tail: the fused
/// rope+append+fa arm ends in `continue`, so every layer that fused was NEVER SCANNED and
/// silently read as "clean". A poisoned residual therefore first reported at the next
/// non-fused layer, which is how "layer 20 creates the poison" could be true of the scan and
/// false of the engine. Also carries the row-table lookup counter, so "the fused path never
/// ran" is distinguishable from "it ran and was innocent".
/// KV-PLANE SCAN (`MEMRA_KV_PLANE_SCAN=1`, DEFAULT OFF, diagnostic only).
///
/// Reads back the STAGED rows of a layer's distributed K/V planes and reports the first row
/// whose quantization scale is not finite. No kernel required: q8_0 blocks are
/// `[half d][32 x i8]` and q5_1 blocks carry `half d` then `half m`, so the fp16 scale at the
/// head of each block is host-checkable straight out of the byte plane.
///
/// It exists because the level-2 bad-row bitmap says EVERY verify row is non-finite at a
/// global-attention layer's join, and row r attends a strict superset of row r-1's keys: that
/// implicates the shared KV history those rows walk, not per-column staging. "The attention
/// output is NaN" and "the KV history it attends is already NaN" are different bugs with
/// different owners, and nothing measured so far separates them. A first-corrupt-row index
/// also dates the corruption against the prime/decode boundary.
///
/// Bounded hard: only layers whose geometry has NO window (the global planes), only the first
/// `MEMRA_KV_PLANE_SCAN_ROUNDS` verify rounds of a process (default 2), and it copies only
/// `[0, staged_len)`, which is ~1.6 MB at the 1480-token repro rather than the 262144-row
/// provision. It still syncs per layer, so it is never a serving or a measured-perf arm.
pub(crate) fn kv_plane_scan_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("MEMRA_KV_PLANE_SCAN").as_deref() == Ok("1"))
}

fn kv_plane_scan_rounds() -> usize {
    static R: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *R.get_or_init(|| {
        std::env::var("MEMRA_KV_PLANE_SCAN_ROUNDS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(2)
    })
}

/// First non-finite fp16 block scale in `bytes`, as (block index, raw u16), scanning one
/// scale every `stride` bytes. Returns None when every block scale is finite.
fn first_bad_scale(bytes: &[u8], stride: usize) -> Option<(usize, u16)> {
    if stride == 0 {
        return None;
    }
    for (i, blk) in bytes.chunks_exact(stride).enumerate() {
        let raw = u16::from_le_bytes([blk[0], blk[1]]);
        if half_is_non_finite(raw) {
            return Some((i, raw));
        }
    }
    None
}

/// IEEE binary16: exponent all ones is Inf or NaN, whatever the mantissa says.
fn half_is_non_finite(raw: u16) -> bool {
    (raw & 0x7C00) == 0x7C00
}

/// Scan one layer's staged K/V planes for a non-finite quantization scale. Returns the
/// receipt line, or None when the layer is out of scope or every scale is finite.
pub(crate) fn scan_kv_plane(
    e: &crate::Engine,
    distributed: &memra_kv::ResidentTpKvCache,
    il: usize,
    pos0: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    // One "round" is one pos0, not one layer: the walk visits 45 layers per verify. The
    // default of 2 rounds is for a fault that shows up immediately; the step37 repro does not
    // fire until rep 3 or later, i.e. round ~60 of the process, so that arm MUST raise
    // MEMRA_KV_PLANE_SCAN_ROUNDS or it will scan only the two rounds that were never going to
    // be poisoned and report a clean history it never looked at.
    static ROUNDS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    static LAST_POS: std::sync::atomic::AtomicUsize =
        std::sync::atomic::AtomicUsize::new(usize::MAX);
    if LAST_POS.swap(pos0, std::sync::atomic::Ordering::Relaxed) != pos0 {
        ROUNDS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    if ROUNDS.load(std::sync::atomic::Ordering::Relaxed) > kv_plane_scan_rounds() {
        return Ok(());
    }
    let staged = distributed.staged_len();
    if staged == 0 {
        return Ok(());
    }
    // ENGAGEMENT RECEIPT. This scan prints only on corruption, so `kvbad=0` in a cell would
    // read the same whether the history was clean or the scan never ran once. Bounded so a
    // 45-layer walk cannot flood the log.
    static SEEN: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let seen = SEEN.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let (ktb, vtb) = (distributed.k_tok_bytes(), distributed.v_tok_bytes());
    if seen < 4 {
        eprintln!(
            "[kv-plane] engaged #{seen} layer {il} pos0={pos0} staged={staged} \
             ktok={ktb} vtok={vtb} (scan armed; a corrupt plane prints its own line)"
        );
    }
    for rank in 0..distributed.ranks().len() {
        let Some(rc) = distributed.rank(rank) else {
            continue;
        };
        // q8_0 K blocks are [half d][32 x i8] = 34B; q5_1 V blocks lead with half d then half m.
        let kbytes = e.dtoh_u8_view(&rc.k().slice(0..staged * ktb))?;
        let vbytes = e.dtoh_u8_view(&rc.v().slice(0..staged * vtb))?;
        let kbad = first_bad_scale(&kbytes, 34);
        let vbad = first_bad_scale(&vbytes, 24);
        if kbad.is_some() || vbad.is_some() {
            let row = |b: Option<(usize, u16)>, tok: usize| {
                b.map(|(i, raw)| format!("blk {i} (row {}) raw={raw:#06x}", i * 34 / tok.max(1)))
                    .unwrap_or_else(|| "clean".into())
            };
            eprintln!(
                "[kv-plane] layer {il} rank {rank} pos0={pos0} staged={staged}                  K={} V={} - the attended KV history is ALREADY non-finite, so a non-finite                  attention output here is a symptom and not the origin",
                row(kbad, ktb),
                row(vbad, vtb)
            );
            return Ok(());
        }
    }
    Ok(())
}

pub(crate) fn verify_arm_receipt(
    arm: &str,
    il: usize,
    pos0: usize,
    t: usize,
    staged: Option<usize>,
) {
    static N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    if N.fetch_add(1, std::sync::atomic::Ordering::Relaxed) >= 200 {
        return;
    }
    eprintln!(
        "[verify-arm] layer {il} arm={arm} pos0={pos0} t={t} staged_len={} rows_tab_lookups={}",
        staged.map(|v| v as i64).unwrap_or(-1),
        crate::tp::ROWS_TAB_ENGAGED.load(std::sync::atomic::Ordering::Relaxed)
    );
}

pub(crate) fn nan_scan_rows(
    e: &Engine,
    buf: &CudaSlice<f32>,
    rows: usize,
    cols: usize,
    what: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    // The readback is also the ATTRIBUTION point for an asynchronous fault: a
    // CUDA_ERROR_ILLEGAL_ADDRESS raised by any launch since the previous scan surfaces on this
    // sync, and the bare DriverError names nothing. Wrapping it with `what` turns "the process
    // died somewhere" into "it died at or before this layer, on this row, at this position".
    let host = e.dtoh(buf).map_err(|err| -> Box<dyn std::error::Error> {
        format!(
            "spec nan-scan: sync at {what} FAILED: {err} — the fault is at or before \
                     this point in the walk"
        )
        .into()
    })?;
    if host.len() < rows * cols {
        return Err(format!(
            "nan-scan {what}: buffer holds {} < {rows}x{cols}",
            host.len()
        )
        .into());
    }
    // SCAN EVERY ROW BEFORE REPORTING. A first-hit return says "row 0 is bad" and leaves the
    // other rows UNEXAMINED, which is exactly the bit that discriminates the two mechanisms: in
    // the t-column verify, row 0 attends keys [0..p+1) and row 1 attends [0..p+2), a strict
    // superset, so poison in the SHARED KV history must appear in BOTH rows, while poison in
    // per-column staging can appear in one. Report the whole map.
    let mut per_row: Vec<usize> = Vec::with_capacity(rows);
    let mut first_bad: Option<(usize, usize)> = None;
    for r in 0..rows {
        let row = &host[r * cols..(r + 1) * cols];
        let bad = row.iter().filter(|v| !v.is_finite()).count();
        per_row.push(bad);
        if bad > 0 && first_bad.is_none() {
            first_bad = Some((r, row.iter().position(|v| !v.is_finite()).unwrap_or(0)));
        }
    }
    if let Some((r0, c0)) = first_bad {
        let map: String = per_row
            .iter()
            .map(|&b| if b == 0 { '.' } else { 'X' })
            .collect();
        return Err(format!(
            "spec nan-scan: {what} produced non-finite values — rows[{rows}] map={map} \
             counts={per_row:?} of {cols} each; first at row {r0} element {c0}. Both rows bad \
             implicates shared state (the KV history this layer reads); one row bad implicates \
             per-column staging."
        )
        .into());
    }
    Ok(())
}

fn vbuf(e: &Engine, n: usize) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
    if spec_lean() { e.uninit(n) } else { e.zeros(n) }
}

/// Scratch KV for the MTP block (one full-attn layer).
///
/// PERSISTENT MODE (default, 2026-07-03 — the acceptance lever): sized cap = max_ctx and kept in
/// sync with the COMMITTED sequence — slot p holds the MTP block's K/V for committed token p
/// (roped p+1, the chain's rope convention), so the draft chain's self-attention sees the FULL
/// committed history instead of only the current round's 1..K+1 chain tokens (the reference
/// engine's "mtp_update" design). Entries come from two sources:
///   - chain appends: accepted positions KEEP their chain-computed entries (embedding exact,
///     hidden chain-approximate — the reference engine accepts the same);
///   - `mtp_kv_fill` batches: prompt positions + the last-draft position on full accept, computed
///     from EXACT trunk hiddens (K/V-only MTP-block pass, no attention/FFN/lm_head).
///     Rejected drafts / p-min extras / pseudo-seed appends are all discarded by the round-start
///     `set_len` truncation (the KvLayer len mechanism — §C rollback for the draft side).
///     Multi-turn spec-decode session (2026-07-05): trunk Cache + persistent MTP draft scratch +
///     the committed token list, alive across generate_spec_session calls. Turn N+1 primes ONLY its
///     suffix (chunked continuation prime over the quantized past) and mtp_kv_fill's its suffix rows,
///     then the round loop runs unchanged. `last_h` carries the pre-output_norm hidden of the last
///     committed row across turns (the predecessor-pairing seed + fill anchor).
///     Per-request sampling config for the sampled-spec serve path.
#[derive(Clone, Copy, Debug)]
pub struct SpecSampling {
    pub temp: f32,
    pub seed: u64,
    pub top_k: i32,            // 0 = off
    pub top_p: f32,            // 1.0 = off
    pub min_p: f32,            // 0.0 = off
    pub penalty_last_n: usize, // 0 = penalties off
    pub penalty_repeat: f32,
    pub penalty_freq: f32,
    pub penalty_present: f32,
}

impl SpecSampling {
    /// Non-identity penalties requested — THE `pen_on` predicate (one definition; the
    /// same group-off rule `SamplerIdentity::of` canonicalizes: a window with neutral
    /// coefficients is penalties-absent). Both spec routes and the dspark accept walk
    /// key their penalty arms off this.
    pub fn pen_on(&self) -> bool {
        self.penalty_last_n > 0
            && (self.penalty_repeat != 1.0
                || self.penalty_freq != 0.0
                || self.penalty_present != 0.0)
    }
}

/// Which draft source a spec session is pinned to. The ENGINE-LEVEL half of
/// `DraftSourcePlan` (memra-gguf `model_plan.rs`, always general): the plan states what the
/// model DECLARES, this states what actually LOADED and therefore what the session runs.
/// Pinned at session creation for the session's lifetime.
///
/// Family-agnostic on purpose (lane/glm5-extract2, the DraftSource seam): glm5 is today's
/// consumer with NativeMtp | Dflash2; the hy3/qwen-next spec lanes select through the same
/// three-way law instead of re-deriving it. What each family still owns is the per-session
/// STATE behind the kind (see `dflash.rs`'s seam note for why that half is not a trait yet).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DraftSourceKind {
    /// The model's own embedded NextN/MTP head.
    NativeMtp,
    /// A separately loaded DFlash2 block-diffusion drafter
    /// ([`crate::dflash::DflashDrafter`]).
    Dflash2,
}

/// The uniform draft-source selection law. Pure — no env, no engine, no family types — so it
/// is CPU-gateable and so every spec family answers "which source" the same way.
///
/// THE LAW, in precedence order:
/// 1. A LOADED DFlash2 drafter IS the source. The operator asked for it by name (a set
///    drafter flag that cannot load is already a loud boot failure, never a silent
///    fallback), and the family's embedded head is deliberately NOT loaded for this source —
///    it is a full trunk layer of VRAM.
/// 2. Otherwise the embedded head, and only when the PLAN declares an embedded source: a
///    loaded head under a plan that does not declare `Embedded` is a load-path bug, not a
///    draft source, and it is refused by name rather than drafted from.
/// 3. Otherwise there is no draft source and speculative decode must refuse before drafting.
pub fn resolve_draft_source_kind(
    plan: memra_gguf::model_plan::DraftSourcePlan,
    embedded_head_loaded: bool,
    dflash_loaded: bool,
) -> Result<DraftSourceKind, String> {
    use memra_gguf::model_plan::DraftSourcePlan as P;
    if dflash_loaded {
        return Ok(DraftSourceKind::Dflash2);
    }
    if embedded_head_loaded {
        if plan != P::Embedded {
            return Err(format!(
                "an embedded draft head is loaded but the ModelPlan declares \
                 draft_source={plan:?} — refused rather than drafting from a head the plan \
                 does not claim"
            ));
        }
        return Ok(DraftSourceKind::NativeMtp);
    }
    Err(format!(
        "no draft source loaded (ModelPlan declares draft_source={plan:?}): speculative \
         decode has nothing to draft from"
    ))
}

#[cfg(test)]
mod draft_source_kind_tests {
    use super::{DraftSourceKind, resolve_draft_source_kind};
    use memra_gguf::model_plan::DraftSourcePlan as P;

    #[test]
    fn a_loaded_drafter_wins_over_a_co_loaded_embedded_head() {
        // The operator asked for the drafter BY NAME (a set drafter flag that cannot load is
        // already a loud boot failure), so it takes precedence under every plan value —
        // including ExternalArtifact, which is what a pack declares when the draft weights
        // are not in the model file.
        for plan in [P::Embedded, P::ExternalArtifact, P::None] {
            assert_eq!(
                resolve_draft_source_kind(plan, true, true).unwrap(),
                DraftSourceKind::Dflash2,
                "plan {plan:?}: a loaded drafter must win"
            );
            assert_eq!(
                resolve_draft_source_kind(plan, false, true).unwrap(),
                DraftSourceKind::Dflash2
            );
        }
    }

    #[test]
    fn the_embedded_head_is_the_source_only_under_a_plan_that_claims_it() {
        assert_eq!(
            resolve_draft_source_kind(P::Embedded, true, false).unwrap(),
            DraftSourceKind::NativeMtp
        );
        // A head loaded under a plan that does not declare Embedded is a LOAD-PATH BUG, not a
        // draft source. Unreachable on glm5 today (its pack hardcodes Embedded and the head
        // only loads under it) — which is exactly why it is pinned here: an unreachable
        // refusal with no arm is an untested refusal, and the next family is the one that
        // makes it reachable.
        for plan in [P::ExternalArtifact, P::None] {
            let err = resolve_draft_source_kind(plan, true, false)
                .expect_err("a head under a non-Embedded plan must refuse");
            assert!(err.contains("does not claim"), "{err}");
            assert!(err.contains(&format!("{plan:?}")), "{err}");
        }
    }

    #[test]
    fn nothing_loaded_refuses_before_drafting_and_names_the_plan() {
        for plan in [P::Embedded, P::ExternalArtifact, P::None] {
            let err =
                resolve_draft_source_kind(plan, false, false).expect_err("no source must refuse");
            assert!(err.contains("no draft source loaded"), "{err}");
            assert!(err.contains(&format!("{plan:?}")), "{err}");
        }
    }
}

/// `MEMRA_SPEC_PMIN` break semantics over per-slot draft confidences (the chain break this
/// module's drafting loops apply inline: `p < p_min && (j > 0 || pmin0)`): keep the longest
/// prefix whose every slot clears `p_min`; slot 0 survives a miss unless PMIN0 arms
/// zero-draft rounds. Prefix truncation is forced by the accept rule anyway (a kept slot
/// after a dropped one could never commit — the dspark confidence-slot argument). Pure so
/// the rule is CPU-gateable; the SHARED K-policy surface every spec family consumes
/// (hoisted from the glm5 loop, lane/glm5-extract-general).
pub fn spec_conf_keep(q: &[f32], p_min: f32, pmin0: bool) -> usize {
    if p_min <= 0.0 {
        return q.len();
    }
    let mut kept = 0usize;
    for (j, &qj) in q.iter().enumerate() {
        if qj < p_min && (j > 0 || pmin0) {
            break;
        }
        kept += 1;
    }
    kept
}

/// Host Philox4x32-10 uniform in (0,1) — mirrors spec_sample.cu's `philox4`/`u01` with the
/// ctr_lo tag 0xFFFF_FFFE, so the host accept-test stream never collides with any device
/// sampling event (device Gumbel uses (i>>2, stream_pos); device residual uses 0xFFFF_FFFD).
/// One value per (seed, ctr) EVENT; callers own the counter discipline. Extracted verbatim
/// from generate_spec_inner2's closure for the dspark sampled-admission walk (the two paths
/// MUST consume the identical stream construction — two ad-hoc Philox copies drifting apart
/// is a distributional bug, not a style problem).
pub(crate) fn host_u01(seed: u64, ctr: u32) -> f32 {
    let (m0, m1) = (0xD2511F53u32, 0xCD9E8D57u32);
    let (mut c0, mut c1, mut c2, mut c3) = (0xFFFF_FFFEu32, ctr, 0u32, 0u32);
    let (mut k0, mut k1) = ((seed & 0xFFFF_FFFF) as u32, (seed >> 32) as u32);
    for _ in 0..10 {
        let (h0, l0) = (((m0 as u64 * c0 as u64) >> 32) as u32, m0.wrapping_mul(c0));
        let (h1, l1) = (((m1 as u64 * c2 as u64) >> 32) as u32, m1.wrapping_mul(c2));
        let (n0, n1, n2, n3) = (h1 ^ c1 ^ k0, l1, h0 ^ c3 ^ k1, l0);
        c0 = n0;
        c1 = n1;
        c2 = n2;
        c3 = n3;
        k0 = k0.wrapping_add(0x9E3779B9);
        k1 = k1.wrapping_add(0xBB67AE85);
    }
    (c0 as f32 + 1.0) * (1.0 / 4294967296.0)
}

/// Tracked draft positions for [`SpecTelemetry`] (serve K defaults to 3; the run-spec gate
/// sweeps K=1..8, and MEMRA_SPEC_CAPMAX defaults to 7 — 8 covers every tuned config).
pub const SPEC_TELEM_POS: usize = 8;

/// Always-on per-draft-position acceptance telemetry (lane/accept-telemetry, 2026-08-05 —
/// the llama.cpp #26389 / vLLM spec-decode counter schema, upstream-sweeps 2026-08-05).
/// Lives on the [`SpecSession`] and accumulates across bursts; the serve worker diffs a
/// stashed copy per burst for its per-model /metrics aggregation and per-request usage.
/// Same normalization as the `[spec-stats]` line: p-min-discarded chain tokens are counted
/// in NEITHER drafted nor accepted.
#[derive(Clone, Copy, Default, Debug)]
pub struct SpecTelemetry {
    /// verify rounds completed (a round-stream burst counts each of its M rounds).
    pub rounds: u64,
    /// tokens drafted / accepted across all rounds.
    pub drafted: u64,
    pub accepted: u64,
    /// how often draft position j (0-based within a round's chain) was offered / accepted.
    /// Positions >= SPEC_TELEM_POS are untracked (totals still count them). The opt-in
    /// round-stream arm (MEMRA_SPEC_STREAM=1) reads back only totals, so under it these
    /// arrays cover the standard-path rounds only and their sums may undercount the totals.
    pub pos_drafted: [u64; SPEC_TELEM_POS],
    pub pos_accepted: [u64; SPEC_TELEM_POS],
}

impl SpecTelemetry {
    /// Fieldwise `self - prev` — the worker's per-burst delta off a copy stashed before the
    /// burst call. Saturating: a caller diffing against the wrong snapshot gets zeros, not
    /// a wrapped counter.
    pub fn delta_since(&self, prev: &SpecTelemetry) -> SpecTelemetry {
        let mut d = SpecTelemetry {
            rounds: self.rounds.saturating_sub(prev.rounds),
            drafted: self.drafted.saturating_sub(prev.drafted),
            accepted: self.accepted.saturating_sub(prev.accepted),
            ..Default::default()
        };
        for j in 0..SPEC_TELEM_POS {
            d.pos_drafted[j] = self.pos_drafted[j].saturating_sub(prev.pos_drafted[j]);
            d.pos_accepted[j] = self.pos_accepted[j].saturating_sub(prev.pos_accepted[j]);
        }
        d
    }
    /// Fieldwise `self += d` — the worker's per-model aggregation.
    pub fn merge(&mut self, d: &SpecTelemetry) {
        self.rounds += d.rounds;
        self.drafted += d.drafted;
        self.accepted += d.accepted;
        for j in 0..SPEC_TELEM_POS {
            self.pos_drafted[j] += d.pos_drafted[j];
            self.pos_accepted[j] += d.pos_accepted[j];
        }
    }

    /// Mean accepted draft-prefix length per verify round (tau).
    pub fn tau(&self) -> f64 {
        if self.rounds > 0 {
            self.accepted as f64 / self.rounds as f64
        } else {
            0.0
        }
    }
}

/// Session-lifetime atomic acceptance counters. The verifier records only after the greedy or
/// rejection-sampling walk has resolved on the host, so these relaxed increments add no GPU
/// launch, synchronization, allocation, or ordering dependency to the numeric path.
struct SpecTelemetryCounters {
    rounds: AtomicU64,
    drafted: AtomicU64,
    accepted: AtomicU64,
    pos_drafted: [AtomicU64; SPEC_TELEM_POS],
    pos_accepted: [AtomicU64; SPEC_TELEM_POS],
}

impl Default for SpecTelemetryCounters {
    fn default() -> Self {
        Self {
            rounds: AtomicU64::new(0),
            drafted: AtomicU64::new(0),
            accepted: AtomicU64::new(0),
            pos_drafted: std::array::from_fn(|_| AtomicU64::new(0)),
            pos_accepted: std::array::from_fn(|_| AtomicU64::new(0)),
        }
    }
}

impl SpecTelemetryCounters {
    fn record_round(&self, drafted: usize, accepted: usize) {
        debug_assert!(accepted <= drafted);
        self.rounds.fetch_add(1, Ordering::Relaxed);
        self.drafted.fetch_add(drafted as u64, Ordering::Relaxed);
        self.accepted.fetch_add(accepted as u64, Ordering::Relaxed);
        for counter in self.pos_drafted.iter().take(drafted) {
            counter.fetch_add(1, Ordering::Relaxed);
        }
        for counter in self.pos_accepted.iter().take(accepted) {
            counter.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Round-stream keeps each round's accept length on device; retain exact scalar totals while
    /// leaving the per-position arrays untouched, matching the pre-existing telemetry contract.
    fn record_totals(&self, rounds: usize, drafted: usize, accepted: usize) {
        self.rounds.fetch_add(rounds as u64, Ordering::Relaxed);
        self.drafted.fetch_add(drafted as u64, Ordering::Relaxed);
        self.accepted.fetch_add(accepted as u64, Ordering::Relaxed);
    }

    fn snapshot(&self) -> SpecTelemetry {
        SpecTelemetry {
            rounds: self.rounds.load(Ordering::Relaxed),
            drafted: self.drafted.load(Ordering::Relaxed),
            accepted: self.accepted.load(Ordering::Relaxed),
            pos_drafted: std::array::from_fn(|j| self.pos_drafted[j].load(Ordering::Relaxed)),
            pos_accepted: std::array::from_fn(|j| self.pos_accepted[j].load(Ordering::Relaxed)),
        }
    }
}

pub struct SpecSession {
    pub(crate) cache: Cache,
    pub(crate) scratch: MtpScratch,
    /// Every token whose state the caches hold, in order (prompt turns + generated), INCLUDING
    /// overshoot: spec commits accepted drafts past max_new; those rows are in the caches, so the
    /// session must count them. Callers render output from this, not from their own echo.
    pub committed: Vec<u32>,
    /// Pre-output_norm hidden of the LAST committed row (device). None before the first turn.
    pub(crate) last_h: Option<CudaSlice<f32>>,
    /// Greedy argmax predicting the token AFTER committed.last() (from the last turn's final
    /// logits). Fuels empty-suffix continuation bursts (serve): the next turn emits this token
    /// first, feeds it, and the round loop resumes without any prime. None before the first turn.
    pub next_pred: Option<u32>,
    /// SAMPLED-SPEC stream continuity across bursts: Philox event counters persist here so a
    /// session's randomness never repeats between generate_spec_session calls. (0,0) at admit.
    pub sctr: u32,
    pub uctr: u32,
    /// PERSISTENT DRAFT-GRAPH CONTEXT (2026-08-01, the serve-burst fixed-cost fix): the captured
    /// draft graph(s) + every device I/O buffer they bake, carried ACROSS generate_spec_session
    /// calls. Before this, every serve burst re-captured the draft graph (2 warmup forwards +
    /// instantiate) — measured ~16ms/burst on H100 q27 (MEMRA_SPEC_BURST sweep,
    /// research/spec-serving-20260801). None before the first turn; error paths drop it
    /// (next burst recaptures — serve retires errored sessions anyway).
    pub(crate) draft_ctx: Option<DraftGraphCtx>,
    /// PENDING-CARRY across bursts (2026-08-01, the serve burst-boundary fix): the bonus token
    /// emitted by the last round but NOT committed to the caches. The old tail committed it with
    /// a solo T=1 trunk pass (+ draft fill), and the next burst's setup fed the stashed next_pred
    /// with ANOTHER solo pass — 2x ~11.5ms/burst measured on H100 q27 ([spec-setup] trace).
    /// Carrying it lets the next empty-suffix greedy burst consume it as round-0 verify col 0,
    /// exactly like a mid-burst full-accept boundary (no solo passes). INVARIANT: when set,
    /// `committed` (== cache rows) EXCLUDES this token although it was already emitted in the
    /// last burst's output, and `last_h` holds the hidden of the last COMMITTED row (its
    /// predecessor — the chain-seed/fill anchor). `next_pred` is None (unknown without the
    /// commit pass). Non-empty-suffix or sampled turns must flush first (spec_flush_pending);
    /// generate_spec_session_sampled does this at entry, and serve parks only flushed sessions.
    pub pending_tok: Option<u32>,
    /// SESSION-AFFINITY TURN CHECKPOINT (lane/session-affinity, 2026-08-05): the state at this
    /// turn's PROMPT-END boundary, retained so a later turn can REWIND here. See
    /// [`SpecCheckpoint`]. Refreshed by every non-empty prime; None until the first one, and on
    /// a rig too tight to hold it (a failed capture is silent — resume just isn't available).
    pub(crate) turn_ckpt: Option<SpecCheckpoint>,
    /// Session-lifetime acceptance telemetry. Relaxed atomics update at the host-side round
    /// accounting the loop already does — no syncs, no allocation. NOTE a
    /// pool-resumed session carries the PREVIOUS requests' counts; per-request consumers
    /// diff with [`SpecTelemetry::delta_since`] around each burst.
    telem: SpecTelemetryCounters,
    /// PREFIX-CACHE publication request (lane/spec-prefix-cache): worker sets this to the
    /// miss-LCP boundary before a cold burst; the prime captures at exactly that split (it must
    /// coincide with the burst's `prime_split` or no capture happens). One-shot: consumed by the
    /// prime, result lands in `boundary_captures`.
    pub capture_at: Option<usize>,
    /// The captures the last prime produced (see [`SpecBoundaryCapture`]). Worker drains them
    /// post-burst to assemble prefix entries. A failed capture is silent, like `turn_ckpt` —
    /// publication just isn't available for that request. Plural since
    /// lane/frspec-multiturn-cache (2026-08-21): a cold burst can capture BOTH the miss-LCP
    /// split (the shared-prefix class) and the stable pre-generation boundary (the
    /// next-turn re-render class) — one entry per stop, exactly the boundary set the plain
    /// prefill tick publishes/checkpoints.
    pub boundary_captures: Vec<SpecBoundaryCapture>,
    /// STABLE-BOUNDARY TURN CHECKPOINT REQUEST (lane/frspec-multiturn-cache, 2026-08-21): the
    /// ABSOLUTE committed-length position the next non-empty prime should capture `turn_ckpt`
    /// at, instead of prompt-end. The worker sets it to the STABLE PRE-GENERATION boundary
    /// (`plain_checkpoint_boundary` — before the live generation header the client rewrites),
    /// porting the 2026-08-09 plain-tier fix: a prompt-end spec checkpoint includes the
    /// template's live assistant-generation header (`<|im_start|>assistant\n<think>\n`), which
    /// the NEXT turn's re-render replaces, so `affinity_match` diverged a couple tokens below
    /// the checkpoint and the spec pool declined 100% of multi-turn agent traffic (measured:
    /// `spec-affinity: declined (history diverged at 6811 of checkpoint 6813)`,
    /// research/multiturn-cache-20260821 B4). One-shot, `capture_at` convention; None = legacy
    /// prompt-end capture.
    pub ckpt_at: Option<usize>,
    /// FAIL-SAFE (lane/step37-vram-admission-20260830, external-review corroboration): set
    /// by the worker on a session serving a step-OOM park REPLAY. The burst entry pre-marks
    /// the draft-graph fallback so the replay never re-enters the capture path — the capture
    /// appetite is part of what drove the card to the OOM, and a replay that recaptures
    /// re-runs the incident. If the eager replay still cannot fit, the bounded retry budget
    /// exhausts into the honest recoverable Overloaded error instead of looping.
    pub capture_disabled: bool,
}
impl SpecSession {
    /// Context capacity of the session's caches (the server's ContextFull guard).
    pub fn cache_max_ctx(&self) -> usize {
        self.cache.max_ctx
    }
    /// Read access to the live trunk cache (lane/spec-prefix-cache): the worker slices
    /// full-attn KV rows `[0..capture.pos)` out of it when publishing a boundary capture —
    /// those rows are append-only for the session's lifetime (rollbacks never truncate below
    /// the prime boundary), so no copy was taken at prime time.
    pub fn cache_ref(&self) -> &Cache {
        &self.cache
    }
    /// Read access to the persistent draft-scratch plane (lane/spec-on-cache-hit): the
    /// worker slices rows `[0..capture.pos)` when publishing a boundary capture, exactly
    /// like the trunk KV — draft rows below the prompt end are append-only for the
    /// session's lifetime (the prime fill wrote them once; rollbacks reset `len_d` to the
    /// committed length, never below the prime boundary, and the true-hidden refresh
    /// rewrites generated positions only). Returns `(k, v, k_tok_bytes, v_tok_bytes)`.
    /// None when the scratch is ring-backed (Step35 SWA — physical rows are not
    /// prefix-addressable; the prefix cache already refuses that class end to end).
    pub fn draft_plane_ref(&self) -> Option<(&CudaSlice<u8>, &CudaSlice<u8>, usize, usize)> {
        if self.scratch.kv.ring.is_some() {
            return None;
        }
        Some((
            &self.scratch.kv.k,
            &self.scratch.kv.v,
            self.scratch.kv.k_tok_bytes,
            self.scratch.kv.v_tok_bytes,
        ))
    }
    /// Snapshot the session's process-local acceptance counters for per-burst diffing.
    pub fn telemetry(&self) -> SpecTelemetry {
        self.telem.snapshot()
    }
    /// Committed position this session can REWIND to (its retained prompt-end boundary), if any.
    /// A request whose prompt matches `committed[..pos]` exactly can resume from here — see
    /// `spec_rewind_to_checkpoint`.
    pub fn rewind_pos(&self) -> Option<usize> {
        self.turn_ckpt.as_ref().map(|c| c.pos)
    }
    /// Whether every ring-backed trunk/draft row needed by the retained checkpoint is resident.
    pub fn rewind_is_resident(&self) -> bool {
        self.turn_ckpt.as_ref().is_some_and(|ckpt| {
            self.cache.can_rollback(&ckpt.snap, 0) && self.scratch.can_rewind_to(ckpt.pos)
        })
    }
    /// Is this session in the DEMOTION-READY shape (see [`SpecSession::into_demoted`])?
    /// `false` means a carried pending must be flushed first (`spec_flush_pending`), or the
    /// session has never run a turn and has no prediction to hand over.
    pub fn demote_ready(&self) -> bool {
        self.pending_tok.is_none() && self.next_pred.is_some()
    }
    /// Does this session hold a carried pending bonus (flush required before a handoff/park)?
    pub fn has_pending(&self) -> bool {
        self.pending_tok.is_some()
    }
    /// Committed row count == cache rows (the session invariant), for the caller's own
    /// `fed`-length cross-check at a handoff boundary.
    pub fn committed_len(&self) -> usize {
        self.committed.len()
    }
    /// DEMOTION HANDOFF (lane/spec-gate, 2026-08-07): consume this session and hand its trunk
    /// cache + next-token prediction to the plain batched-decode path.
    ///
    /// WHY THIS IS EXACT (greedy). The invariant at a burst boundary is `cache.pos ==
    /// committed.len()`: every committed row has trunk KV + recurrent state, exactly as a plain
    /// tokenwise prime of the same `committed` sequence would have left it (that is the
    /// session-tail contract, and the same property `spec_rewind_to_checkpoint` and the reuse
    /// pool already rely on). `next_pred` is the argmax of the verify's logits for the LAST
    /// committed row — and verify-column logits are bit-identical to plain decode's logits at
    /// that position, because `matmul_decode_exact` bit-identity IS the basis of the greedy
    /// accept walk. So handing (cache, next_pred) to the batched path continues the stream from
    /// a state indistinguishable from one the batched path produced itself: the batched tick
    /// emits `next_pred`, feeds it into this same cache, and decodes on.
    ///
    /// `None` when the session is not in the handoff shape — a carried pending (its bonus row is
    /// NOT in the cache, so `spec_flush_pending` must commit it first) or no `next_pred` yet
    /// (never bursted). Callers must not force it: a half-committed cache handed to the batched
    /// path would silently skip a token.
    ///
    /// The MTP draft scratch, the persistent draft-graph context and the turn checkpoint are
    /// DROPPED here (freeing their VRAM): the batched path never drafts, and this handoff is
    /// one-way by design — there is no cheap symmetric re-promotion (rebuilding the draft KV
    /// would mean an `mtp_kv_fill` over the whole committed history).
    pub fn into_demoted(self) -> Option<(Cache, u32)> {
        if self.pending_tok.is_some() || self.cache.tainted {
            return None;
        }
        let np = self.next_pred?;
        debug_assert_eq!(
            self.cache.pos,
            self.committed.len(),
            "demotion handoff: cache rows != committed tokens"
        );
        Some((self.cache, np))
    }
    /// Pool-resume hook (audit Q2): clear the parked draft-graph failure memoization so a
    /// NEW request resuming this session gets one fresh capture chance — a transient-pressure
    /// capture failure must not persist for the pool's whole lifetime (the TRT #16072 class).
    /// Logs once iff a flag was actually set; a no-fallback resume is silent and free.
    pub fn reset_graph_fallback_on_resume(&mut self) {
        if let Some(line) = self
            .draft_ctx
            .as_mut()
            .and_then(|c| c.failed.reset_on_resume())
        {
            eprintln!("{line}");
        }
    }
}

/// A session's PROMPT-END boundary state, the rewind target for session-affinity resume.
///
/// WHY THIS BOUNDARY, AND WHY IT IS THE ONLY ONE WORTH KEEPING. The rewrite class this lane
/// exists for (a client that strips `<think>` blocks out of prior assistant turns) mutates the
/// text the session GENERATED, never the prompt it was given. So turn N's prompt agrees with
/// turn N-1's committed tokens up to almost exactly where turn N-1's generation began — the
/// prompt-end boundary. Keeping a checkpoint there means the next turn re-primes only its own
/// delta (the rewritten answer + the new user turn) instead of the whole conversation.
///
/// WHAT IT MUST HOLD. Full-attn KV is append-only and position-addressed, so rewinding it is a
/// `len` truncation (no data). Linear-attn (GDN) conv/ssm state is mutated IN PLACE with no
/// position index, so it must be a real device COPY — that copy is the entire reason a spec
/// session could not previously rewind. The MTP draft scratch needs no copy either: its rows
/// below the boundary were written by this turn's fill and are never revisited (the per-round
/// true-hidden refresh only rewrites the CURRENT burst's committed positions), so rewinding it
/// is also just a `len` reset. `last_h` is the hidden of the last row below the boundary — the
/// predecessor-pairing anchor the next prime's fill reads for its first row.
///
/// COST: one `Cache::snapshot` per TURN, on a code path that already takes one per ROUND.
pub(crate) struct SpecCheckpoint {
    snap: crate::cache::CacheSnapshot,
    /// Committed length at the boundary (== cache.pos there, the session invariant).
    pos: usize,
    /// Pre-output_norm hidden of row `pos - 1`.
    last_h: CudaSlice<f32>,
}

/// PREFIX-CACHE BOUNDARY CAPTURE (lane/spec-prefix-cache, 2026-08-14): the state a spec session
/// records at its cold-prime split so the WORKER can publish a cross-request prefix entry —
/// the commit-gated-publication port (research/cache-spec-design-20260814/PORT-PLAN.md item 1).
/// Only the pieces that are DESTROYED by continuing the prime need copies here: the in-place
/// GDN conv/ssm states (via `Cache::snapshot`, same mechanism as [`SpecCheckpoint`]) and the
/// boundary logits. Full-attn KV rows `[0..pos)` and draft-scratch rows `[0..pos)` are
/// append-only for the session's lifetime (rollbacks never truncate below the prime boundary),
/// so the worker slices those from the live caches post-burst instead of copying at prime time.
pub struct SpecBoundaryCapture {
    pub snap: crate::cache::CacheSnapshot,
    /// Token boundary (== cache.pos at capture; == the worker's miss-LCP split).
    pub pos: usize,
    /// Full-vocab logits after the prefix prime — the entry's boundary logits.
    pub logits: Vec<f32>,
    /// Pre-output_norm trunk hidden of row `pos - 1` (lane/spec-on-cache-hit): the
    /// predecessor-pairing anchor a RESTORED spec session's first suffix-fill row reads
    /// (the `SpecSession::last_h` convention). Empty = unavailable (capture stays valid;
    /// the fill's zeros row-0 fallback covers it at a bounded acceptance cost).
    pub last_h: Vec<f32>,
    /// Per-layer latent boundary tails (lane/glm5-prefix-latent2, 2026-09-01): the
    /// generation-destroyed slice of each MLA/DSA layer's boundary state, captured eagerly
    /// so the worker's DEFERRED publication can slice the append-only planes from the live
    /// cache (`LatentKvLayer::snapshot_plane_at`). EMPTY on every two-plane model — the
    /// pre-field captures are byte-identical; a latent-bearing cache with an EMPTY vec here
    /// keeps the publisher's loud refusal (the fail-closed door stays shut).
    pub latent_tails: Vec<Option<crate::cache::LatentTailCapture>>,
}

/// D2H one hidden row out of a `[T, n_embd]` prime hidden stack — the boundary anchor a
/// spec boundary capture carries for later restored-session fills. Failure is silent
/// (`turn_ckpt` convention): the capture publishes without an anchor.
pub(crate) fn capture_boundary_hidden(
    e: &Engine,
    h_rows: &CudaSlice<f32>,
    pos: usize,
    n_embd: usize,
) -> Vec<f32> {
    if pos == 0 || h_rows.len() < pos * n_embd {
        return Vec::new();
    }
    let Ok(mut row) = e.uninit(n_embd) else {
        return Vec::new();
    };
    if e.copy_view_into(
        &mut row,
        0,
        &h_rows.slice((pos - 1) * n_embd..pos * n_embd),
        n_embd,
    )
    .is_err()
    {
        return Vec::new();
    }
    e.dtoh(&row).unwrap_or_default()
}

/// ROLLBACK DOOR for sampled BOUNDARY tokens (lane/sampled-spec-quality, 2026-08-19).
/// Default ON: the token a burst emits at its own boundary is drawn from the request's
/// sampler. `MEMRA_SPEC_SAMPLED_BOUNDARY=0` restores the pre-lane posture (an ARGMAX at
/// every boundary) without touching greedy, which is byte-unaffected either way.
pub fn spec_sampled_boundary_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("MEMRA_SPEC_SAMPLED_BOUNDARY").as_deref() != Ok("0"))
}

/// ROLLBACK DOOR for SESSION-SPANNING penalty history (lane/sampled-spec-quality).
/// Default ON: `pen_hist` is seeded from the session's committed tail, so repetition /
/// frequency / presence penalties see the whole stream. `MEMRA_SPEC_PEN_SESSION=0`
/// restores the pre-lane posture (each burst restarts the window from its own prompt
/// slice, i.e. from NOTHING on a continuation burst) — and with the door shut the worker
/// must keep refusing penalized sampled prefix-cache restores, because the restored
/// session's continuation burst is handed no prompt slice at all.
pub fn spec_pen_session_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("MEMRA_SPEC_PEN_SESSION").as_deref() != Ok("0"))
}

/// ROLLBACK DOOR for extended-entry publication from a RESTORED session
/// (lane/sampled-spec-quality, Item 3). Default ON: a converted prefix-cache hit that fed a
/// suffix captures its own prompt-end boundary so the NEXT turn can hit a longer prefix.
/// `MEMRA_SPEC_RESTORE_REPUBLISH=0` restores the pre-lane posture (a namespace learns exactly
/// one boundary and never advances it). Whole-entry semantics only — the boundary is the
/// restored session's own prompt end, so `entry_pos != fed_len` still refuses on the way in.
pub fn spec_restore_republish_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("MEMRA_SPEC_RESTORE_REPUBLISH").as_deref() != Ok("0"))
}

/// Diagnostics: name every boundary token on stderr (`MEMRA_SPEC_BOUNDARY_TRACE=1`), with
/// the argmax the pre-lane code would have emitted from the same row. This is how the
/// lane MEASURES the boundary rate and the deviation rate instead of estimating them.
fn spec_boundary_trace() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("MEMRA_SPEC_BOUNDARY_TRACE").as_deref() == Ok("1"))
}

/// llama-parity floor for the penalty window when the request does not ask for a bigger
/// one (`repeat_last_n` default). The serve API arms `penalty_last_n = PEN_WINDOW_MAX` for any
/// non-identity penalty, so this floor only matters to explicit small windows and to the
/// CLI env path.
const PEN_WINDOW_FLOOR: usize = 64;

/// CEILING on the penalty window, and it is a COST bound, not a semantic preference.
/// `penalize_logits_f32` (cu/spec_sample.cu) dedups on device by having thread `i` scan
/// `hist[0..i]`, so a pass is O(n_hist²) and it runs ~3x per verify round (the q rows, the
/// p column, the bonus column). The serve API uses this same bound for every non-identity
/// penalty so host/plain, sparse-device, and speculative sampling cannot change logits on
/// admission demotion. An uncapped 128k-token history would put ~1.7e10
/// comparisons per pass, tens of ms per round, i.e. penalties would silently destroy decode
/// throughput on exactly the long-context requests that most want them. 8192 keeps a pass
/// at ~7e7 comparisons (tens of microseconds) while still being **128x wider than the
/// pre-lane effective window** (64 prompt-tail tokens + whatever the current burst had
/// generated). A request that genuinely needs a window beyond this wants host-side dedup +
/// counts through a new kernel signature — a follow-up lane, named here rather than hidden.
/// `pub` since lane/dspark-penalized-sampled-20260821: the dspark route's accept walk and
/// the dspark_sample_gate binary trim their uploads with the SAME cap — a second constant
/// is a second thing to drift.
pub const PEN_WINDOW_MAX: usize = 8192;

/// Seed a penalty window over the SESSION, not the burst (lane/sampled-spec-quality,
/// Item 2). The window is the last `max(penalty_last_n, 64)` tokens of
/// `session_committed ++ burst_prompt` — for a cold turn-1 burst (`session_committed`
/// empty, default `penalty_last_n`) that is byte-identically the pre-lane
/// `prompt.iter().rev().take(64).rev()`; for a continuation burst it is the stream the
/// client actually asked us to penalize, where the pre-lane code had NOTHING.
/// `pub` since lane/dspark-penalized-sampled-20260821: the dspark route seeds its session
/// window through the SAME function (one definition of "the window" across both spec
/// routes and the gate binary's trunk-only reference arm).
pub fn pen_window_seed(
    session_committed: &[u32],
    burst_prompt: &[u32],
    penalty_last_n: usize,
) -> Vec<u32> {
    let win = penalty_last_n.clamp(PEN_WINDOW_FLOOR, PEN_WINDOW_MAX);
    let take_prompt = burst_prompt.len().min(win);
    let take_sess = (win - take_prompt).min(session_committed.len());
    let mut hist = Vec::with_capacity(take_sess + take_prompt);
    hist.extend_from_slice(&session_committed[session_committed.len() - take_sess..]);
    hist.extend_from_slice(&burst_prompt[burst_prompt.len() - take_prompt..]);
    hist
}

/// Draw a BOUNDARY token from the target distribution the request asked for
/// (lane/sampled-spec-quality, Item 1) — the fix for "sampled spec emits an ARGMAX token at
/// every burst boundary".
///
/// WHY THIS EXISTS. A spec burst's first emitted token is not produced by the accept walk:
/// it comes off a logits row that already exists (the prime's last row on a cold burst; the
/// row after the last committed token on a continuation burst; the prefix-cache entry's
/// boundary row on a restored one). Pre-lane that token was `argmax` in BOTH sampling
/// regimes, so a sampled stream took a greedy token once per burst — measured, not
/// estimated, in research/spec-cache-20260818/SAMPLED-QUALITY.md. At temperature > 0 the
/// customer asked for a sampled token, so this draws one.
///
/// THE PROGRAM IS THE FULL-ACCEPT BONUS'S PROGRAM, deliberately: penalize the row (over the
/// session's window), take this row's OWN filter stats (the sampfix-20260805 law — stats
/// from a neighbour row mis-scale every `e0` and can wipe the row to token 0), gumbel-perturb
/// with the session's Philox stream at `*sctr`, argmax the perturbed row. Reusing the bonus's
/// composition means `sample_check`'s distributional oracle covers this draw too, and the
/// boundary token is drawn from the same filtered/penalized `p` the accept walk targets.
///
/// THE STREAM IS THE SESSION'S, NOT A FRESH ONE. `sctr` is the caller's live counter and is
/// advanced by exactly one, so a boundary draw consumes the next value in the same Philox
/// stream the accept walk uses — never a second, independently seeded stream (which would be
/// a new distributional bug: two streams from one seed correlate wherever their counters
/// collide). That also makes a restored session's boundary draw at `sctr == 0` bit-identical
/// to the cold session's own first draw from the same logits row, which is what preserves the
/// sampled-hit lane's per-seed hit==cold byte identity.
#[allow(clippy::too_many_arguments)]
pub fn sample_boundary_token_dev(
    e: &Engine,
    logits: &CudaSlice<f32>,
    n_vocab: usize,
    sp: &SpecSampling,
    pen_hist: &[u32],
    sctr: &mut u32,
    site: &str,
) -> Result<u32, Box<dyn std::error::Error>> {
    debug_assert!(
        sp.temp > 0.0,
        "boundary sampling is the sampled regime only"
    );
    // Own copy: penalize_logits mutates in place and the caller's row is live state
    // (prime_logits back the constrained recompute; last_col_logits backs round 0's accept).
    let mut col = e.zeros(n_vocab)?;
    e.copy_into(&mut col, 0, logits, n_vocab)?;
    let pen_on = sp.penalty_last_n > 0
        && (sp.penalty_repeat != 1.0 || sp.penalty_freq != 0.0 || sp.penalty_present != 0.0);
    if pen_on && !pen_hist.is_empty() {
        // window trim mirrors the round loop's own upload (`pen_hist[w0..]`), cap included.
        let w0 = pen_hist
            .len()
            .saturating_sub(sp.penalty_last_n.min(PEN_WINDOW_MAX));
        let hist = &pen_hist[w0..];
        let hd = e.htod_u32_v(hist)?;
        e.penalize_logits(
            &mut col,
            &hd,
            hist.len(),
            sp.penalty_repeat,
            sp.penalty_freq,
            sp.penalty_present,
            n_vocab,
        )?;
    }
    let rows0 = e.htod_i32(&[0])?;
    let (mut th_d, mut z_d, mut mx_d) = (e.zeros(1)?, e.zeros(1)?, e.zeros(1)?);
    e.filter_stats(
        &col, n_vocab, &rows0, &mut th_d, &mut z_d, &mut mx_d, n_vocab, 1, sp.temp, sp.top_k,
        sp.top_p, sp.min_p,
    )?;
    let (th, mx) = (e.dtoh(&th_d)?[0], e.dtoh(&mx_d)?[0]);
    let mut perturb = e.zeros(n_vocab)?;
    e.gumbel_perturb_filtered(&col, &mut perturb, n_vocab, sp.seed, *sctr, sp.temp, mx, th)?;
    *sctr = sctr.wrapping_add(1);
    let td = e.argmax_token_device(&perturb, n_vocab)?;
    let tok = guard_vocab_token(
        e.dtoh_u32_one(&td)?,
        n_vocab,
        &format!("sampled boundary token (site={site})"),
    )?;
    if spec_boundary_trace() {
        // the pre-lane token, from the SAME row, so the deviation rate is measurable.
        let raw = e.argmax_token_device(logits, n_vocab)?;
        let greedy = e.dtoh_u32_one(&raw)?;
        eprintln!(
            "[spec-boundary] site={site} sampled={tok} argmax={greedy} \
             deviates={} temp={} sctr={}",
            (tok != greedy) as u8,
            sp.temp,
            sctr.wrapping_sub(1),
        );
    }
    Ok(tok)
}

/// Host-row twin of [`sample_boundary_token_dev`] (the prime / feed / entry rows arrive as
/// host `Vec<f32>`).
#[allow(clippy::too_many_arguments)]
pub fn sample_boundary_token(
    e: &Engine,
    logits: &[f32],
    sp: &SpecSampling,
    pen_hist: &[u32],
    sctr: &mut u32,
    site: &str,
) -> Result<u32, Box<dyn std::error::Error>> {
    let n_vocab = logits.len();
    let d = e.htod(logits)?;
    sample_boundary_token_dev(e, &d, n_vocab, sp, pen_hist, sctr, site)
}

struct SpecPipeTraceClock {
    pair: usize,
    started: std::time::Instant,
}

#[derive(Clone)]
struct SpecPipeTraceCtx {
    clock: std::sync::Arc<SpecPipeTraceClock>,
    round: usize,
    lane: usize,
}

struct SpecPipeTraceMarker {
    trace: SpecPipeTraceCtx,
    phase: &'static str,
    edge: &'static str,
    slot: Option<usize>,
}

unsafe extern "C" fn spec_pipe_trace_marker(raw: *mut std::ffi::c_void) {
    let marker = unsafe { Box::from_raw(raw.cast::<SpecPipeTraceMarker>()) };
    let lane = if marker.trace.lane == 0 { "A" } else { "B" };
    let slot = marker
        .slot
        .map(|v| v.to_string())
        .unwrap_or_else(|| "-".into());
    let t_ms = marker.trace.clock.started.elapsed().as_secs_f64() * 1e3;
    use std::io::Write as _;
    let stderr = std::io::stderr();
    let mut stderr = stderr.lock();
    let _ = writeln!(
        stderr,
        "[spec-pipe-timeline] pair={} round={} lane={lane} phase={} edge={} \
         slot={slot} t_ms={t_ms:.3}",
        marker.trace.clock.pair, marker.trace.round, marker.phase, marker.edge,
    );
}

fn enqueue_spec_pipe_trace_marker(
    stream: &cudarc::driver::CudaStream,
    trace: Option<&SpecPipeTraceCtx>,
    phase: &'static str,
    edge: &'static str,
    slot: Option<usize>,
) -> Result<(), Box<dyn std::error::Error>> {
    let Some(trace) = trace else {
        return Ok(());
    };
    let marker = Box::new(SpecPipeTraceMarker {
        trace: trace.clone(),
        phase,
        edge,
        slot,
    });
    let raw = Box::into_raw(marker);
    let result = unsafe {
        cudarc::driver::result::stream::launch_host_function(
            stream.cu_stream(),
            spec_pipe_trace_marker,
            raw.cast(),
        )
    };
    if let Err(err) = result {
        unsafe {
            drop(Box::from_raw(raw));
        }
        return Err(err.into());
    }
    Ok(())
}

#[derive(Default)]
struct SpecPipeProgress {
    setup_done: [bool; 2],
    draft_done: [usize; 2],
    stage0_done: [usize; 2],
    verify_done: [usize; 2],
    accept_done: [usize; 2],
    finished: [bool; 2],
    aborted: bool,
}

/// Host-side issue coordinator for the reduced two-session speculative pipeline. Each session
/// keeps its existing call stack and round locals; this object only orders phase entry. The
/// primary mutex spans whole draft/accept/tail issue regions so Engine's single-stream scratch
/// cannot be interleaved by the two host threads.
struct SpecPipeSync {
    progress: std::sync::Mutex<SpecPipeProgress>,
    changed: std::sync::Condvar,
    primary: std::sync::Mutex<()>,
    trace: Option<std::sync::Arc<SpecPipeTraceClock>>,
}

impl SpecPipeSync {
    fn new() -> Self {
        static TRACE_PAIR: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let trace = (std::env::var("MEMRA_SPEC_PIPE_TRACE").as_deref() == Ok("1")).then(|| {
            std::sync::Arc::new(SpecPipeTraceClock {
                pair: TRACE_PAIR.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1,
                started: std::time::Instant::now(),
            })
        });
        Self {
            progress: std::sync::Mutex::new(SpecPipeProgress::default()),
            changed: std::sync::Condvar::new(),
            primary: std::sync::Mutex::new(()),
            trace,
        }
    }
}

#[derive(Clone)]
struct SpecPipeLane {
    sync: std::sync::Arc<SpecPipeSync>,
    lane: usize,
    rt: &'static crate::pp::PpNRt,
    walk_permit: crate::pp::PpWalkPermit,
}

struct SpecPipePrimaryGuard<'a> {
    _primary: std::sync::MutexGuard<'a, ()>,
    _walk: crate::pp::PpWalkBorrowGuard,
}

impl SpecPipeLane {
    fn peer(&self) -> usize {
        1 - self.lane
    }

    fn aborted() -> Box<dyn std::error::Error> {
        "paired speculative peer aborted".into()
    }

    fn trace(&self, round: usize) -> Option<SpecPipeTraceCtx> {
        self.sync.trace.as_ref().map(|clock| SpecPipeTraceCtx {
            clock: clock.clone(),
            round,
            lane: self.lane,
        })
    }

    fn setup_begin(&self) -> Result<crate::pp::PpWalkBorrowGuard, Box<dyn std::error::Error>> {
        let mut p = self.sync.progress.lock().unwrap();
        while !p.aborted && self.lane == 1 && !p.setup_done[0] && !p.finished[0] {
            p = self.sync.changed.wait(p).unwrap();
        }
        if p.aborted {
            Err(Self::aborted())
        } else {
            drop(p);
            self.rt.borrow_walk(&self.walk_permit, "spec_pipe/setup")
        }
    }

    fn setup_end(&self) {
        let mut p = self.sync.progress.lock().unwrap();
        p.setup_done[self.lane] = true;
        self.sync.changed.notify_all();
    }

    fn draft_begin(
        &self,
        round: usize,
    ) -> Result<SpecPipePrimaryGuard<'_>, Box<dyn std::error::Error>> {
        let peer = self.peer();
        let mut p = self.sync.progress.lock().unwrap();
        loop {
            if p.aborted {
                return Err(Self::aborted());
            }
            let setup_ready =
                (p.setup_done[0] || p.finished[0]) && (p.setup_done[1] || p.finished[1]);
            let prior_ready = p.accept_done[self.lane] >= round
                && (p.accept_done[peer] >= round || p.finished[peer]);
            let turn_ready = if self.lane == 0 {
                true
            } else {
                p.draft_done[0] > round || p.finished[0]
            };
            if setup_ready && prior_ready && turn_ready {
                break;
            }
            p = self.sync.changed.wait(p).unwrap();
        }
        drop(p);
        let primary = self.sync.primary.lock().unwrap();
        let walk = self.rt.borrow_walk(&self.walk_permit, "spec_pipe/draft")?;
        Ok(SpecPipePrimaryGuard {
            _primary: primary,
            _walk: walk,
        })
    }

    fn draft_end(&self, round: usize) {
        let mut p = self.sync.progress.lock().unwrap();
        p.draft_done[self.lane] = round + 1;
        self.sync.changed.notify_all();
    }

    /// Admit stage 0 and return whether this lane owns the interval's one reverse fence.
    /// Lane B releases as soon as lane A has issued its boundary TX, not after A's full body.
    fn stage0_begin(&self, round: usize) -> Result<bool, Box<dyn std::error::Error>> {
        let peer = self.peer();
        let mut p = self.sync.progress.lock().unwrap();
        loop {
            if p.aborted {
                return Err(Self::aborted());
            }
            let ready = if self.lane == 0 {
                p.draft_done[0] > round && (p.draft_done[1] > round || p.finished[1])
            } else {
                p.draft_done[1] > round && (p.stage0_done[0] > round || p.finished[0])
            };
            if ready {
                return Ok(self.lane == 0 || p.finished[peer]);
            }
            p = self.sync.changed.wait(p).unwrap();
        }
    }

    fn stage0_end(&self, round: usize) {
        let mut p = self.sync.progress.lock().unwrap();
        p.stage0_done[self.lane] = round + 1;
        self.sync.changed.notify_all();
    }

    /// Stage 1 is single-owner per engine. A proceeds immediately after its own ticket; B waits
    /// for A's full stage1/head issue so only A.S1 and B.S0 can overlap.
    fn stage1_begin(&self, round: usize) -> Result<(), Box<dyn std::error::Error>> {
        let mut p = self.sync.progress.lock().unwrap();
        while !p.aborted
            && !(p.stage0_done[self.lane] > round
                && (self.lane == 0 || p.verify_done[0] > round || p.finished[0]))
        {
            p = self.sync.changed.wait(p).unwrap();
        }
        if p.aborted {
            Err(Self::aborted())
        } else {
            Ok(())
        }
    }

    fn verify_end(&self, round: usize) {
        let mut p = self.sync.progress.lock().unwrap();
        p.verify_done[self.lane] = round + 1;
        self.sync.changed.notify_all();
    }

    fn accept_begin(
        &self,
        round: usize,
    ) -> Result<SpecPipePrimaryGuard<'_>, Box<dyn std::error::Error>> {
        let mut p = self.sync.progress.lock().unwrap();
        loop {
            if p.aborted {
                return Err(Self::aborted());
            }
            let ready = if self.lane == 0 {
                p.verify_done[0] > round && (p.verify_done[1] > round || p.finished[1])
            } else {
                p.verify_done[1] > round && (p.accept_done[0] > round || p.finished[0])
            };
            if ready {
                break;
            }
            p = self.sync.changed.wait(p).unwrap();
        }
        drop(p);
        let primary = self.sync.primary.lock().unwrap();
        let walk = self.rt.borrow_walk(&self.walk_permit, "spec_pipe/accept")?;
        Ok(SpecPipePrimaryGuard {
            _primary: primary,
            _walk: walk,
        })
    }

    fn accept_end(&self, round: usize) {
        let mut p = self.sync.progress.lock().unwrap();
        p.accept_done[self.lane] = round + 1;
        self.sync.changed.notify_all();
    }

    fn primary(&self) -> Result<SpecPipePrimaryGuard<'_>, Box<dyn std::error::Error>> {
        let primary = self.sync.primary.lock().unwrap();
        let walk = self.rt.borrow_walk(&self.walk_permit, "spec_pipe/tail")?;
        Ok(SpecPipePrimaryGuard {
            _primary: primary,
            _walk: walk,
        })
    }

    fn coordinated_walk(&self) -> Result<crate::pp::PpWalkBorrowGuard, Box<dyn std::error::Error>> {
        self.rt
            .borrow_walk(&self.walk_permit, "spec_pipe/coordinated_verify")
    }

    fn finish(&self, failed: bool) {
        let mut p = self.sync.progress.lock().unwrap();
        p.finished[self.lane] = true;
        p.aborted |= failed;
        self.sync.changed.notify_all();
    }
}

struct SpecPipeFinish<'a> {
    lane: &'a SpecPipeLane,
    closed: bool,
}

impl<'a> SpecPipeFinish<'a> {
    fn new(lane: &'a SpecPipeLane) -> Self {
        Self {
            lane,
            closed: false,
        }
    }

    fn close(&mut self, failed: bool) {
        self.lane.finish(failed);
        self.closed = true;
    }
}

impl Drop for SpecPipeFinish<'_> {
    fn drop(&mut self) {
        if !self.closed {
            self.lane.finish(true);
        }
    }
}

/// Scoped transfer of one exclusively-borrowed session to the second host issue thread.
/// `CudaGraph` is not marked Send by cudarc because its raw driver handles carry no automatic
/// trait. CUDA driver graph handles are context-scoped rather than OS-thread-affine; the caller
/// binds that context before touching the session, joins before returning, and never aliases the
/// pointer. Keep this exception local to the experimental pair call instead of marking the public
/// session type Send.
struct SpecPipeSessionPtr(*mut SpecSession);

unsafe impl Send for SpecPipeSessionPtr {}

impl SpecPipeSessionPtr {
    unsafe fn get_mut(&mut self) -> &mut SpecSession {
        unsafe { &mut *self.0 }
    }
}

/// Per-session persistent draft-graph context: the captured CUDA graph(s) plus the device
/// buffers whose POINTERS the capture bakes. Reuse legality: the greedy capture bakes only
/// session-stable pointers (the session's own MtpScratch KV — allocated once, never realloc'd;
/// the model's resident embedding; the process-wide OnceLock p_min) and the g_* buffers held
/// HERE — so one capture serves the session's whole lifetime. The sampled capture additionally
/// bakes (seed, temp) as capture-time constants and needs k q-slots — keyed by `s_key`, dropped
/// and recaptured when a pool-resumed request changes them. `*_failed` memoizes a failed capture
/// so the eager fallback doesn't pay a doomed capture attempt every burst.
/// Capture identity of the parked SAMPLED draft graph (`DraftGraphCtx::graph_s`).
///
/// EXACTNESS, not perf (lane/graph-s-key-exactness-20260819; receipts
/// `research/spec-cache-20260818/GRAPH-S-KEY.md`). Two classes of field live here, both
/// load-bearing:
///
/// - **Baked constants.** `seed` and `temp` are capture-time constants INSIDE the graph and `k`
///   sizes the q slots its replays write. A resumed request changing any of them must recapture.
///   This is all the key used to carry.
/// - **Regime fields.** `top_k`/`top_p`/`min_p`/`pen_on` are not baked, but they decide whether
///   the captured graph is a legal draft chain AT ALL. The in-graph draw is one gumbel-max over
///   the RAW softmax (`gumbel_perturb_ctr`, unfiltered by construction), while the verify builds
///   the accept test's `q` from `filter_stats(q_slots, top_k, top_p, min_p)`. If those disagree
///   the accept test evaluates a distribution the draft was never sampled from: a draft token
///   below the filter threshold gathers `q = 0` (`softmax_gather_filtered_f32`,
///   `cu/spec_sample.cu`) and `u * 0 < p` accepts it UNCONDITIONALLY.
///
/// Omitting the regime fields was reachable — not through the prefix-cache spec restore (that
/// path is greedy-only, `memra-server` `spec_restore_convertible`), but through WHOLE-SESSION
/// spec reuse: a parked `SpecSession` carries this `DraftGraphCtx`, and the pool-resume probe
/// applies no sampler predicate at all. Turn 1 pure-temp parks a graph; turn 2 of the same
/// conversation, same explicit seed and temperature, adds `top_p`/`top_k` and inherits it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct SampledGraphKey {
    seed: u64,
    temp_bits: u32,
    k: usize,
    top_k: i32,
    top_p_bits: u32,
    min_p_bits: u32,
    pen_on: bool,
}

impl SampledGraphKey {
    pub(crate) fn new(
        seed: u64,
        temp: f32,
        k: usize,
        top_k: i32,
        top_p: f32,
        min_p: f32,
        pen_on: bool,
    ) -> Self {
        SampledGraphKey {
            seed,
            temp_bits: temp.to_bits(),
            k,
            top_k,
            top_p_bits: top_p.to_bits(),
            min_p_bits: min_p.to_bits(),
            pen_on,
        }
    }

    /// The one regime the PURE-TEMP in-graph sampled chain may stand in for the eager one:
    /// nothing but temperature shapes `q`. Computed FROM THE KEY so the capture guard, the
    /// launch guard and the key can never drift apart (they were three separate expressions
    /// before this lane, and the launch site simply forgot to ask).
    pub(crate) fn pure_temp(&self) -> bool {
        self.top_k == 0
            && f32::from_bits(self.top_p_bits) >= 1.0
            && f32::from_bits(self.min_p_bits) <= 0.0
            && !self.pen_on
    }

    /// Truncation filters active — the capture body needs the IN-GRAPH filter nodes
    /// (`filter_stats` + `gumbel_perturb_filtered_ctr`) so the draft draws from the same
    /// filtered distribution the accept test reconstructs. Meaningful only when
    /// `graph_capturable`; penalties never reach a capture body.
    pub(crate) fn filtered(&self) -> bool {
        !self.pure_temp()
    }

    /// May the sampled draft graph be CAPTURED (and a parked one LAUNCHED) for this regime?
    /// Pure-temp always; filtered regimes when the filtered-capture door is on
    /// (lane/step37-draft-graph-serving-20260830); penalties never — the per-round history
    /// cannot be baked into a graph, and composing a raw-softmax (or stale-history) draw
    /// with a penalized accept test is the unconditional-accept exactness bug. Computed FROM
    /// THE KEY for the same no-drift reason as `pure_temp`.
    pub(crate) fn graph_capturable(&self) -> bool {
        !self.pen_on && (self.pure_temp() || spec_graph_filtered_on())
    }
}

/// Per-head captured graphs for the MULTI-HEAD MTP draft chain (step-modulo prefix-replay,
/// lane/step37-draft-graph-serving-20260830). The chain POLICY — which head serves step j,
/// how long the replayed prefix is, which stored seed feeds row r — stays HOST-SIDE in the
/// launch loop, exactly `mtp_chain_forward_dev`'s order; the graphs capture ONE head-row
/// forward each, on the head's OWN scratch plane:
/// - `interior[i]`: head i, `with_head=false` — KV append + carrier only. Interior rows'
///   logits are dead in the eager chain too (`mtp_chain_forward_dev` keeps only the last
///   row), so skipping the head matmul changes no consumed byte and removes the eager
///   chain's per-replay-row full-vocab matmul.
/// - `last[i]`: head i, `with_head=true` + the mode's tail (greedy argmax, or the sampled
///   gumbel draw — filtered in-graph when the request carries filters).
///
/// One `DraftChainGraphs` per MODE (greedy vs sampled), owning its keeper: dropping the
/// sampled chain on an s_key change never invalidates the greedy one.
struct DraftChainGraphs {
    interior: Vec<cudarc::driver::CudaGraph>,
    last: Vec<cudarc::driver::CudaGraph>,
    /// Never read: exists to OWN the captured graphs' backing buffers for as long as the
    /// graphs replay (the capture-retain law; same class as `DsparkSegGraph::_keeper`).
    _keeper: Vec<Box<dyn std::any::Any + Send>>,
}

/// Sampled-tail capture pack for `mtp_head_forward_cap`: the persistent buffers and baked
/// constants of the in-graph categorical draw. `filt: None` = the PURE-TEMP body (gumbel
/// over the raw softmax), byte-identical to the pre-lane capture; `Some` adds the in-graph
/// truncation filter (`filter_stats` + `gumbel_perturb_filtered_ctr`) so the draft draws
/// from the same filtered distribution the accept test reconstructs
/// (lane/step37-draft-graph-serving-20260830).
struct SampledCapArgs<'a> {
    ctr: &'a mut CudaSlice<u32>,
    perturb: &'a mut CudaSlice<f32>,
    q_out: &'a mut CudaSlice<f32>,
    seed: u64,
    temp: f32,
    filt: Option<SampledCapFilter<'a>>,
}

/// In-graph truncation-filter nodes: the stat slots `filter_stats` fills and the perturb
/// reads, plus the filter constants baked into the capture (they live in `s_key`, so a
/// request whose filters differ drops the parked graph before this ever goes stale).
struct SampledCapFilter<'a> {
    rows0: &'a CudaSlice<i32>,
    th: &'a mut CudaSlice<f32>,
    z: &'a mut CudaSlice<f32>,
    mx: &'a mut CudaSlice<f32>,
    top_k: i32,
    top_p: f32,
    min_p: f32,
}

pub(crate) struct DraftGraphCtx {
    g_tok: CudaSlice<u32>,
    g_pos: CudaSlice<i32>,
    g_seed: CudaSlice<f32>,
    g_p: CudaSlice<f32>,
    g_ctr: CudaSlice<u32>,
    g_q: CudaSlice<f32>,
    g_perturb: CudaSlice<f32>,
    /// IN-GRAPH filter-stat slots (filtered sampled capture): `filter_stats` writes
    /// (th, z, mx) here inside the graph; `gumbel_perturb_filtered_ctr` reads (mx, th) from
    /// the same slots. Persistent so the baked pointers survive replays. `g_rows0` is the
    /// constant row-index-0 the single-row `filter_stats` launch reads (a captured memcpy
    /// source must not be a host temporary).
    g_rows0: CudaSlice<i32>,
    g_th: CudaSlice<f32>,
    g_z: CudaSlice<f32>,
    g_mx: CudaSlice<f32>,
    q_slots: Vec<CudaSlice<f32>>,
    /// DRAFT-SIDE GRAMMAR MASK (lane/draft-mask): packed allowed-set words over the DRAFT
    /// head's vocab, at a STABLE address so the captured draft graph's mask node reads the
    /// per-position contents the host re-uploads before each replay (the graph-promote
    /// pattern from decode.rs). Empty unless the session drafts under a grammar.
    g_dmask: CudaSlice<u32>,
    /// was `graph` captured WITH the mask node? A parked graph of the wrong shape is dropped.
    /// Covers the multi-head `chain` too (single-head and chain are mutually exclusive for a
    /// given model, so one flag serves whichever is active).
    graph_masked: bool,
    graph: Option<cudarc::driver::CudaGraph>,
    graph_s: Option<cudarc::driver::CudaGraph>,
    /// Multi-head chain graphs (see [`DraftChainGraphs`]): greedy and sampled chains, the
    /// chain twins of `graph` / `graph_s`. `chain_s`'s capture identity is `s_key` (shared
    /// with `graph_s` — a session is either single-head or chain, never both), and it obeys
    /// the same drop rules (key mismatch, penalty regime, mask-shape change).
    chain: Option<DraftChainGraphs>,
    chain_s: Option<DraftChainGraphs>,
    /// Failed-capture memoization for both graphs — LOUD on flip, cleared on pool resume
    /// (audit Q2, the TRT #16072 silent-permanent-coverage-loss class).
    failed: DraftGraphFallback,
    /// Capture identity of `graph_s` — see [`SampledGraphKey`]. `None` iff no sampled graph is
    /// parked; a request whose key differs drops the parked graph (and its q slots/keeper).
    s_key: Option<SampledGraphKey>,
    /// CAPTURE-RETAIN keepers (#68 root cause, 2026-08-04): the warmup-run transients whose
    /// pool addresses the captured graph(s) bake. Without these, the transients return to the
    /// pool at capture-body exit and later work (burst-boundary prime/fill/commit passes, or a
    /// co-served session in the worker) reuses those addresses — the persisted graph's replay
    /// then reads/writes live unrelated buffers (exactness corruption, first seen as the ST
    /// serve-spec 4B graph-arm corruption; one-shot CLI calls never re-shuffled the pool, which
    /// is why run-spec K=1..8 passed on the same checkpoint). Same fix class as
    /// capture_graph_retained's gemma/decode.rs sites — hold as long as the graph replays.
    keeper: Vec<Box<dyn std::any::Any + Send>>,
    keeper_s: Vec<Box<dyn std::any::Any + Send>>,
}

/// Failed-capture memoization for the two draft graphs (audit Q2, 2026-08-05 — the
/// TRT #16072 trap class: pressure-triggered, silent, long-lived coverage loss).
///
/// Three contracts:
/// - LOUD FLIP: `mark_*` returns the warn line exactly on the false→true transition
///   (returned, not printed, so the once-per-flip contract is unit-testable); the caller
///   `eprintln!`s it UNCONDITIONALLY — a dropped draft graph is never silent. Re-marking
///   an already-failed graph returns None (the per-burst memoization that keeps the eager
///   fallback from paying a doomed capture attempt every burst).
/// - RESET ON RESUME: `reset_on_resume` clears both flags — a parked session resumed by a
///   NEW request gets one fresh capture chance instead of carrying a transient-pressure
///   failure for the pool's whole lifetime. Returns the note line only when a flag was
///   actually set (quiet on the common clean-resume path).
/// - Shape-change clears (`clear_*`) stay silent, exactly as before: they precede a fresh
///   capture attempt whose own failure would re-flip loudly.
#[derive(Default)]
pub(crate) struct DraftGraphFallback {
    greedy: bool,
    sampled: bool,
}
impl DraftGraphFallback {
    fn mark_greedy(&mut self, reason: &str) -> Option<String> {
        if self.greedy {
            return None;
        }
        self.greedy = true;
        Some(format!(
            "[spec] WARN: draft-graph capture failed ({reason}); eager fallback until session resume"
        ))
    }
    fn mark_sampled(&mut self, reason: &str) -> Option<String> {
        if self.sampled {
            return None;
        }
        self.sampled = true;
        Some(format!(
            "[spec] WARN: sampled draft-graph capture failed ({reason}); eager fallback until session resume"
        ))
    }
    fn greedy_failed(&self) -> bool {
        self.greedy
    }
    fn sampled_failed(&self) -> bool {
        self.sampled
    }
    fn clear_greedy(&mut self) {
        self.greedy = false;
    }
    fn clear_sampled(&mut self) {
        self.sampled = false;
    }
    /// Pool-resume reset: both graphs get a fresh capture chance. Some(note) iff any flag
    /// was set (so clean resumes stay quiet).
    pub(crate) fn reset_on_resume(&mut self) -> Option<String> {
        if !self.greedy && !self.sampled {
            return None;
        }
        let which = match (self.greedy, self.sampled) {
            (true, true) => "greedy+sampled",
            (true, false) => "greedy",
            _ => "sampled",
        };
        self.greedy = false;
        self.sampled = false;
        Some(format!(
            "[spec] draft-graph fallback reset on session resume ({which}); recapture eligible"
        ))
    }
}

impl DraftGraphCtx {
    fn new(e: &Engine, n_embd: usize, qlen: usize) -> Result<Self, Box<dyn std::error::Error>> {
        Ok(DraftGraphCtx {
            g_tok: e.alloc_u32_zeroed(1)?,
            g_pos: e.htod_i32(&[0])?,
            g_seed: e.zeros(n_embd)?,
            g_p: e.zeros(1)?,
            g_ctr: e.alloc_u32_zeroed(1)?,
            g_q: e.zeros(qlen)?,
            g_perturb: e.zeros(qlen)?,
            g_rows0: e.htod_i32(&[0])?,
            g_th: e.zeros(1)?,
            g_z: e.zeros(1)?,
            g_mx: e.zeros(1)?,
            q_slots: Vec::new(),
            g_dmask: e.alloc_u32_zeroed(1)?,
            graph_masked: false,
            graph: None,
            graph_s: None,
            chain: None,
            chain_s: None,
            failed: DraftGraphFallback::default(),
            s_key: None,
            keeper: Vec::new(),
            keeper_s: Vec::new(),
        })
    }
}

pub(crate) struct MtpScratch {
    kv: KvLayer,
    /// Logical row capacity. On the graph/DC draft path it also doubles as fa_decode_dc's
    /// bucket_max: n_splits is sized from it ONCE, so the graph captured at round 0 stays valid
    /// for every later t_kv. Step35 refuses that path and may back this logical extent with the
    /// smaller host-indexed SWA ring instead.
    cap: usize,
    extra: Vec<MtpScratchPlane>,
}

struct MtpScratchPlane {
    kv: KvLayer,
    cap: usize,
}

fn mtp_scratch_layout(
    cfg: &memra_gguf::config::ModelConfig,
    geom: Option<&crate::hybrid::DraftGeom>,
) -> (usize, usize, usize, usize) {
    // Student draft heads carry fewer KV heads (head_dim unchanged) -> smaller scratch rows.
    let n_head_kv = geom.map(|g| g.n_head_kv).unwrap_or(cfg.n_head_kv as usize);
    let head_dim_k = cfg.head_dim_k as usize;
    let head_dim_v = cfg.head_dim_v as usize;
    assert!(
        head_dim_k.is_multiple_of(32) && head_dim_v.is_multiple_of(32),
        "KVQUANT requires head_dim%32==0 (MTP scratch)"
    );
    let kv_dim_k = head_dim_k * n_head_kv;
    let kv_dim_v = head_dim_v * n_head_kv;
    // The fp8-KV arm deliberately does not reach the draft scratch; keep the exact format
    // policy shared with `MtpScratch::new` so admission scales the same allocation.
    let (kbb, vbb) = crate::kv_blk_bytes();
    let k_tok_bytes = (kv_dim_k / 32) * kbb;
    let v_tok_bytes = (kv_dim_v / 32) * vbb;
    (kv_dim_k, kv_dim_v, k_tok_bytes, v_tok_bytes)
}

fn mtp_chain_head_index(step: usize, head_count: usize) -> usize {
    assert!(head_count > 0, "MTP chain requires at least one head");
    step % head_count
}

impl MtpScratch {
    fn alloc_plane(
        e: &Engine,
        cfg: &memra_gguf::config::ModelConfig,
        plan: &memra_gguf::model_plan::ModelPlan,
        cap: usize,
        geom: Option<&crate::hybrid::DraftGeom>,
    ) -> Result<MtpScratchPlane, Box<dyn std::error::Error>> {
        let (kv_dim_k, kv_dim_v, k_tok_bytes, v_tok_bytes) = mtp_scratch_layout(cfg, geom);
        let ring = if crate::cache::swa_ring_on()
            && crate::plan_backend::decode_batch_program(plan)
                == crate::plan_backend::DecodeBatchProgram::SlidingGatedMoe
        {
            let window = plan
                .layers
                .iter()
                .find_map(|layer| match layer.attention {
                    memra_gguf::model_plan::AttentionPlan::SlidingWindow { window, .. } => {
                        Some(window as usize)
                    }
                    _ => None,
                })
                .ok_or("sliding-gated-MoE draft scratch has no sliding-window layer")?;
            Some(crate::cache::KvRing::new(
                crate::cache::swa_ring_rows(window, cap),
                window,
            ))
        } else {
            None
        };
        let alloc_rows = ring.as_ref().map(crate::cache::KvRing::rows).unwrap_or(cap);
        // Ring-backed planes arm the device base mirror for the dcw draft arm (see
        // KvLayer::base_d): the captured chain derives its physical rows from
        // (len_d, base_d, window) with zero per-token node updates.
        let base_d = match ring.as_ref() {
            Some(_) => Some(e.htod_i32(&[0])?),
            None => None,
        };
        Ok(MtpScratchPlane {
            kv: KvLayer {
                k: e.alloc_u8(alloc_rows * k_tok_bytes)?,
                v: e.alloc_u8(alloc_rows * v_tok_bytes)?,
                kv_dim_k,
                kv_dim_v,
                k_tok_bytes,
                v_tok_bytes,
                len: 0,
                ring,
                len_d: e.htod_i32(&[0])?,
                base_d,
            },
            cap,
        })
    }

    fn new(
        e: &Engine,
        cfg: &memra_gguf::config::ModelConfig,
        plan: &memra_gguf::model_plan::ModelPlan,
        cap: usize,
        geom: Option<&crate::hybrid::DraftGeom>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        // env-selected KV formats (default 34/24). The fp8-KV arm (MEMRA_KV_FP8) deliberately
        // does NOT reach the draft scratch: fp8 drafts drifted acceptance 69-88% -> 46%
        // (2026-07-12 A/B); the scratch is tiny, so it keeps baseline q8_0/q5_1 numerics
        // while the TRUNK cache carries the fp8 depth win. Scratch append/fa pass g=false.
        let primary = Self::alloc_plane(e, cfg, plan, cap, geom)?;
        Ok(MtpScratch {
            kv: primary.kv,
            cap: primary.cap,
            extra: Vec::new(),
        })
    }

    fn push_plane(
        &mut self,
        e: &Engine,
        cfg: &memra_gguf::config::ModelConfig,
        plan: &memra_gguf::model_plan::ModelPlan,
        geom: Option<&crate::hybrid::DraftGeom>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.extra
            .push(Self::alloc_plane(e, cfg, plan, self.cap, geom)?);
        Ok(())
    }

    fn plane_count(&self) -> usize {
        1 + self.extra.len()
    }

    fn plane(&self, index: usize) -> (&KvLayer, usize) {
        if index == 0 {
            (&self.kv, self.cap)
        } else {
            let plane = &self.extra[index - 1];
            (&plane.kv, plane.cap)
        }
    }

    fn plane_mut(&mut self, index: usize) -> (&mut KvLayer, usize) {
        if index == 0 {
            (&mut self.kv, self.cap)
        } else {
            let plane = &mut self.extra[index - 1];
            (&mut plane.kv, plane.cap)
        }
    }

    // #[track_caller]: set_len/set_plane_len have eight call sites (checkpoint restore, spec
    // rollback, session grow, seed replay ...) and the lap failure needs to say WHICH one, not
    // just that a rewind was refused.
    #[track_caller]
    fn set_plane_len(
        &mut self,
        e: &Engine,
        index: usize,
        n: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let caller = std::panic::Location::caller();
        let (kv, cap) = self.plane_mut(index);
        if let Some(ring) = kv.ring.as_ref()
            && !ring.can_rewind_to(n)
        {
            // NAME THE NUMBERS (2026-08-28). This error is a step37 serving blocker on the
            // vendor-default shape and it fires from more than one call path with more than
            // one trigger: a long generation walks the checkpoint out of the ring, but a
            // ~4.5k-token prompt also fails within 5 s of prime, which accumulation cannot
            // explain. A bare message forced two rounds of guessing; the operands make each
            // trigger name itself.
            let raw = n.saturating_sub(ring.window().saturating_sub(1));
            return Err(format!(
                    "SWA ring MTP checkpoint has been lapped; full re-prime required (plane={index} rewind_to={n} window={} base={} rows={} cap={cap} needed_view_start={} < base, called from {caller})",
                    ring.window(),
                    ring.base(),
                    ring.rows(),
                    raw & !31usize,
                )
                .into());
        }
        kv.len = n;
        e.set_i32_one(&mut kv.len_d, n as i32)
    }

    /// Set BOTH length counters: the host mirror AND the device len_d the captured append/fa read
    /// (a 4-byte in-place htod — the counter pointer is baked into the graph, never realloc'd).
    /// This is the ONLY truncation/rollback mechanism the persistent draft KV needs.
    #[track_caller]
    fn set_len(&mut self, e: &Engine, n: usize) -> Result<(), Box<dyn std::error::Error>> {
        let caller = std::panic::Location::caller();
        if !self.can_rewind_to(n) {
            // set_plane_len re-checks and reports the operands; call it so the failure carries
            // which plane refused and why, instead of this bare aggregate.
            for index in 0..self.plane_count() {
                self.set_plane_len(e, index, n)?;
            }
            return Err(format!(
                "SWA ring MTP checkpoint has been lapped; full re-prime required (aggregate rewind_to={n}, no single plane reported, called from {caller})"
            )
            .into());
        }
        for index in 0..self.plane_count() {
            self.set_plane_len(e, index, n)?;
        }
        Ok(())
    }

    fn can_rewind_to(&self, n: usize) -> bool {
        (0..self.plane_count()).all(|index| {
            self.plane(index)
                .0
                .ring
                .as_ref()
                .is_none_or(|ring| ring.can_rewind_to(n))
        })
    }

    /// Pre-arm ring headroom for `rows` upcoming DEVICE-COUNTER appends (the dcw draft arm):
    /// a captured chain cannot rebase mid-replay, so any rebase the coming appends could need
    /// happens HERE, host-side, before the capture warmups or the round's replays (the rebase
    /// arm of `prepare_kv_append` also refreshes the plane's `base_d` device mirror). No-op on
    /// flat planes and when the ring already has room; `len` is untouched either way.
    fn ensure_dcw_headroom(
        &mut self,
        e: &Engine,
        rows: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        for index in 0..self.plane_count() {
            let (kv, _) = self.plane_mut(index);
            let Some(ring) = kv.ring.as_ref() else {
                continue;
            };
            let retain = memra_kv::swa_retain_from(kv.len, ring.window(), ring.base());
            e.prepare_kv_append(kv, retain, rows)?;
        }
        Ok(())
    }
}

/// Retained verify intermediates for the REPLAY-FREE partial accept (2026-07-03, the profiled
/// #1 spec cost at long ctx: the partial-accept replay was a DUPLICATE trunk pass — ~0.54 extra
/// full weight reads per round — recomputing columns the verify had already produced
/// bit-identically). Holds, per linear layer, everything needed to rebuild its recurrent state
/// to "after the first j verify columns" WITHOUT re-running the trunk:
/// - BATCHED-path layers (`gdn`): the exact token-major inputs the round's ONE gdn_scan
///   consumed. A prefix re-run of the SAME kernel (t=j) from the snapshot state is bit-identical
///   to the first j iterations of the verify's scan — the kernel's t-loop carries state in
///   registers and iteration t never depends on T. `qkv_mixed` (the conv input) feeds the
///   pure-copy ring rebuild.
/// - PER-COLUMN-path layers (`cols`): dtod clones of (conv_state, ssm_state) taken after each
///   column 0..t-2 — pure copies of the actual chain states (the last column is never a rebuild
///   target: j <= t-1).
///   Full-attn layers need nothing: their verify KV rows are bit-identical to eager's (the
///   decode-exact contract; verify-probe pins it), so rollback = len truncation.
struct GdnStash {
    qkv_mixed: CudaSlice<f32>, // [t, conv_dim] token-major (conv input)
    q_l2: CudaSlice<f32>,
    k_l2: CudaSlice<f32>,
    v_g: CudaSlice<f32>, // [t, num_v, d_state]
    g_log: CudaSlice<f32>,
    beta: CudaSlice<f32>, // [t, num_v]
}
pub(crate) struct VerifyCkpt {
    gdn: Vec<Option<GdnStash>>, // [n_layer], Some iff batched linear path ran
    #[allow(clippy::type_complexity)]
    // allow: one-shot composite type; naming it would hide the shape that matters at the call site
    cols: Vec<Option<Vec<(CudaSlice<f32>, CudaSlice<f32>)>>>, // [n_layer][col] = (conv, ssm) after col
}
/// Opaque handle for the dspark round (dflash.rs) — VerifyCkpt stays spec-private.
pub(crate) struct DsparkVerifyCkpt(VerifyCkpt);

/// Engine-bundle slice 3 (DSF-ROUNDCOST-20260820 §2 row 4 / §5 rank 1): bucketed CUDA
/// graphs for the dspark verify's LINEAR-layer segments. The measured verify is ~2,800
/// eager launches whose residual cost is DEVICE-side per-launch overhead (slice 2 proved
/// host dispatch is not the binder: fully-deferred dispatch bought ~0 wall). The 48 GDN
/// layers between full-attention layers are shape-static given vt — no positions, no
/// t_kv, state addressed through pointer tables — so runs of them capture per
/// (segment, vt) and replay as ONE graph launch each. Full-attention layers stay eager
/// (their per-row append/fa arm picks are t_kv-driven — the exec-update extension).
///
/// Per round out-of-graph: one pointer-table refresh (gdn ping-pong moves the canonical
/// handles), one input-staging copy per segment, host parity bookkeeping. Captured via
/// `capture_graph_retained` (2 warmups + capture, keeper retains warmup transients so
/// pool addresses stay stable); the warmups EXECUTE, so segment conv/ssm state is saved
/// before and restored after — the graph's first real launch starts from the exact
/// pre-round state. The ckpt column stash rides persistent slabs (written inside the
/// graph as memcpy nodes); commit reads them via `dspark_commit_prefix_slab`.
/// `MEMRA_DSPARK_VERIFY_GRAPH=0` reverts to the eager walk (byte-identical body).
pub(crate) struct DsparkVerifyGraphs {
    /// Linear-attention layer indices ascending; `lin_pos[il]` = index into the vecs.
    lin: Vec<usize>,
    lin_pos: std::collections::HashMap<usize, usize>,
    /// [n_lin x 6] pointer table (conv, s0, s1, conv, s1, s0 per layer), refreshed per
    /// verify from the live handles; layer il's slice starts at lin_pos[il]*6.
    table_all: CudaSlice<u64>,
    host_table: Vec<u64>,
    /// Persistent per-layer ckpt stash slabs: row r of the verify at slab offset
    /// r*words. Shared by every (segment, vt) bucket — one verify runs at a time.
    stash_conv: Vec<CudaSlice<f32>>,
    stash_ssm: Vec<CudaSlice<f32>>,
    conv_words: usize,
    ssm_words: usize,
    /// Per-vt input/output staging (stable addresses the graphs bake).
    stage: std::collections::HashMap<usize, (CudaSlice<f32>, CudaSlice<f32>)>,
    /// Per-vt dflash tap-sink buffers — the captured segments bake the tap dst address,
    /// so the sink buffer must live (and persist) with the graphs, not with the round.
    pub(crate) tap_bufs: std::collections::HashMap<usize, CudaSlice<f32>>,
    graphs: std::collections::HashMap<(usize, usize), DsparkSegGraph>,
    /// Warmup-corruption guard scratch: pre-capture conv/ssm of every linear layer
    /// (sized n_lin — the slice-4c full-verify warmups execute the whole walk).
    save_conv: CudaSlice<f32>,
    save_ssm: CudaSlice<f32>,
    max_run: usize,
    n_embd: usize,
    /// Set by the verify walk: this round's linear ckpt lives in the slabs (the caller
    /// commits through `dspark_commit_prefix_slab` instead of the cols arm).
    pub(crate) round_slab: bool,
    // ---- slice 4c: full-verify single graph per (vt, rung) ----
    /// Full-attention layer indices ascending; `fa_pos[il]` = index into the vec.
    fa: Vec<usize>,
    fa_pos: std::collections::HashMap<usize, usize>,
    /// [n_fa x 2 x t_cap] interleaved (k,v) base-pointer pairs, refreshed per verify;
    /// layer il's slice starts at `fa_pos[il] * 2 * t_cap` (the seqs twins read pairs
    /// [2z], z < t <= t_cap, so one t_cap-sized table serves every vt).
    fa_table: CudaSlice<u64>,
    fa_host_table: Vec<u64>,
    t_cap: usize,
    /// Per-vt position staging for the captured bodies — contents refreshed per round
    /// (rope reads row r; the seqs twins derive append slot and T_kv per z from it).
    pos_stage: std::collections::HashMap<usize, CudaSlice<i32>>,
    /// Full-verify graphs keyed (vt, rung_end, hi).
    full: std::collections::HashMap<(usize, usize, usize), DsparkSegGraph>,
    /// Largest n with every layer in [0, n) linear or full-attention (walk coverage).
    covered: usize,
    /// Every layer in [0, n) is linear or full-attention (no MLA/unknown mixers) — the
    /// full-verify capture walks all of them.
    walk_uniform: bool,
    /// Last `(captures, device graph-mem reserved bytes)` reading taken by
    /// `HybridModel::dspark_vg_admission_debt` — the two-point base of the MARGINAL debt
    /// projection (see `dspark_vg_debt_projection`; a mean-based reading extrapolated the
    /// pool's one-time shared allocation and reserved 8.5 GB of phantom VRAM).
    debt_obs: Option<(usize, usize)>,
}

struct DsparkSegGraph {
    graph: cudarc::driver::CudaGraph,
    _keeper: Vec<Box<dyn std::any::Any + Send>>,
}

/// Per-call arguments of [`HybridModel::qwen35_tparallel_fa_layer`] — one struct so the
/// eager walk and the slice-4c captured full-verify graphs hand the SAME body its two
/// modes without a second copy of the math.
pub(crate) struct FaLayerArgs<'a> {
    /// [T] per-row positions (device): rope reads them row-indexed; the seqs twins read
    /// them per-z (append slot = pos, T_kv = pos + 1).
    pub pos_d: &'a CudaSlice<i32>,
    /// Verify-level lazy per-row 1-element position buffers — only the per-row fallback
    /// arm builds/uses them (graph mode refuses that arm).
    pub pos_rows: &'a mut Option<Vec<CudaSlice<i32>>>,
    pub pos0: usize,
    pub seqs_append: bool,
    pub batch_fa_on: bool,
    /// Some((kv pointer table, offset-in-u64s, rung_end)) = captured-graph mode.
    pub graph_cap: Option<(&'a CudaSlice<u64>, usize, usize)>,
    /// ROUND-STREAM (lane/draftcost-moe, v0.100 train merge): Some((token stream, device
    /// round counter)) routes the FA attend through the dc rows kernels and the Linear
    /// mixer through `linear_attn_verify_t` (the stream arms the old inline body carried).
    /// Never armed together with `graph_cap` (the verify-level merge guard refuses).
    pub stream: Option<(&'a CudaSlice<u32>, &'a CudaSlice<i32>)>,
    /// VerifyCkpt for the stream-Linear arm's GdnStash install; None in graph mode and
    /// for FA layers that never touch it.
    pub ckpt: Option<&'a mut VerifyCkpt>,
}

// SAFETY: `CudaGraph` is not marked Send by cudarc because its raw driver handles carry
// no automatic trait; CUDA driver graph handles are context-scoped rather than
// OS-thread-affine (the SpecPipeSessionPtr precedent above). The ctx lives in
// `HybridModel::dspark_vgraphs` behind a Mutex and every touch happens on the engine's
// single decode-stream thread.
unsafe impl Send for DsparkVerifyGraphs {}

impl DsparkVerifyGraphs {
    /// Live capture count (segment + full graphs) — the denominator of
    /// [`dspark_vg_debt_projection`]'s observed bytes/capture mean.
    pub(crate) fn captures(&self) -> usize {
        self.graphs.len() + self.full.len()
    }

    /// Take the marginal-growth debt reading and record this observation for the next one.
    /// Called under the pool mutex by `HybridModel::dspark_vg_admission_debt`.
    pub(crate) fn admission_debt(&mut self, reserved_bytes: usize) -> usize {
        let captures = self.captures();
        let debt =
            dspark_vg_debt_projection(captures, dspark_vg_cap(), reserved_bytes, self.debt_obs);
        if captures > 0 {
            match self.debt_obs {
                Some((c0, _)) if captures <= c0 => {}
                _ => self.debt_obs = Some((captures, reserved_bytes)),
            }
        }
        debt
    }

    /// Build for this cache's shape. None when there are no linear layers, sizes are
    /// non-uniform, or the trunk keeps a gemma4 config (never on the qwen35 family).
    pub(crate) fn new(
        e: &Engine,
        cache: &Cache,
        t_max: usize,
        n_embd: usize,
    ) -> Result<Option<Self>, Box<dyn std::error::Error>> {
        let lin: Vec<usize> = (0..cache.recur.len())
            .filter(|&il| cache.recur[il].is_some())
            .collect();
        if lin.is_empty() || t_max < 2 {
            return Ok(None);
        }
        let first = cache.recur[lin[0]].as_ref().unwrap();
        let (conv_words, ssm_words) = (first.conv_state.len(), first.ssm_state.len());
        for &il in &lin {
            let rl = cache.recur[il].as_ref().unwrap();
            if rl.conv_state.len() != conv_words || rl.ssm_state.len() != ssm_words {
                return Ok(None);
            }
        }
        let n = lin.len();
        let mut lin_pos = std::collections::HashMap::with_capacity(n);
        for (k, &il) in lin.iter().enumerate() {
            lin_pos.insert(il, k);
        }
        // longest run of consecutive linear layers (save-scratch sizing)
        let mut max_run = 1usize;
        let mut run = 1usize;
        for w in lin.windows(2) {
            if w[1] == w[0] + 1 {
                run += 1;
                max_run = max_run.max(run);
            } else {
                run = 1;
            }
        }
        let rows = t_max - 1;
        let mut stash_conv = Vec::with_capacity(n);
        let mut stash_ssm = Vec::with_capacity(n);
        for _ in 0..n {
            stash_conv.push(e.uninit(rows * conv_words)?);
            stash_ssm.push(e.uninit(rows * ssm_words)?);
        }
        let host_table = vec![0u64; n * 6];
        let table_all = e.htod_u64(&host_table)?;
        // slice 4c: full-attention census for the full-verify graphs.
        let fa: Vec<usize> = (0..cache.kv.len())
            .filter(|&il| cache.kv[il].is_some())
            .collect();
        let mut fa_pos = std::collections::HashMap::with_capacity(fa.len());
        for (k, &il) in fa.iter().enumerate() {
            fa_pos.insert(il, k);
        }
        let n_layers = cache.kv.len().max(cache.recur.len());
        // exactly one of (linear state, kv cache) per layer — no MLA/unknown mixers.
        let walk_uniform = (0..n_layers).all(|il| {
            cache.recur.get(il).is_some_and(|r| r.is_some())
                != cache.kv.get(il).is_some_and(|k| k.is_some())
        });
        // Contiguous covered prefix: the largest n such that every layer in [0, n) is
        // linear or full-attention. The TRUNK walk is [0, layers.len()) and the cache
        // vecs can carry EXTRA state slots past it (the q38 export keeps the MTP head
        // layer's kv at the tail — hi == lin+fa never held, the s4c battery's zero
        // 'full' captures). The full-graph guard is walk coverage, not slot arithmetic.
        let covered = (0..n_layers)
            .take_while(|il| lin_pos.contains_key(il) || fa_pos.contains_key(il))
            .count();
        let t_cap = t_max;
        let fa_host_table = vec![0u64; fa.len() * 2 * t_cap];
        let fa_table = e.htod_u64(&fa_host_table)?;
        Ok(Some(Self {
            lin,
            lin_pos,
            table_all,
            host_table,
            stash_conv,
            stash_ssm,
            conv_words,
            ssm_words,
            stage: std::collections::HashMap::new(),
            tap_bufs: std::collections::HashMap::new(),
            graphs: std::collections::HashMap::new(),
            save_conv: e.uninit(n * conv_words)?,
            save_ssm: e.uninit(n * ssm_words)?,
            max_run,
            n_embd,
            round_slab: false,
            fa,
            fa_pos,
            fa_table,
            fa_host_table,
            t_cap,
            pos_stage: std::collections::HashMap::new(),
            full: std::collections::HashMap::new(),
            covered,
            walk_uniform,
            debt_obs: None,
        }))
    }

    /// Rebuild the pointer tables from the live handles (once per verify — the gdn
    /// ping-pong swaps the canonical/alt handles between rounds; a fresh generation's
    /// cache buffers land at new addresses; a stale table would read the wrong state).
    pub(crate) fn refresh_tables(
        &mut self,
        e: &Engine,
        cache: &Cache,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use cudarc::driver::DevicePtr;
        {
            let s = &e.gpu.stream();
            for (k, &il) in self.lin.iter().enumerate() {
                let rl = cache.recur[il].as_ref().unwrap();
                let (pc, _g0) = rl.conv_state.device_ptr(s);
                let (p0, _g1) = rl.ssm_state.device_ptr(s);
                let (p1, _g2) = rl.ssm_state_alt.device_ptr(s);
                let o = k * 6;
                self.host_table[o] = pc;
                self.host_table[o + 1] = p0;
                self.host_table[o + 2] = p1;
                self.host_table[o + 3] = pc;
                self.host_table[o + 4] = p1;
                self.host_table[o + 5] = p0;
            }
            for (k, &il) in self.fa.iter().enumerate() {
                let kvl = cache.kv[il].as_ref().unwrap();
                let (pk, _g0) = kvl.k.device_ptr(s);
                let (pv, _g1) = kvl.v.device_ptr(s);
                let o = k * 2 * self.t_cap;
                for z in 0..self.t_cap {
                    self.fa_host_table[o + 2 * z] = pk;
                    self.fa_host_table[o + 2 * z + 1] = pv;
                }
            }
        }
        e.htod_u64_into(&self.host_table, &mut self.table_all)?;
        if !self.fa_host_table.is_empty() {
            e.htod_u64_into(&self.fa_host_table, &mut self.fa_table)?;
        }
        Ok(())
    }

    /// Slice 4c eligibility: Some(rung_end) when this round can replay (or capture) a
    /// full-verify graph — the whole walk [lo, hi) is covered, every layer is linear or
    /// full-attention, and ALL of the round's per-row t_kv values take the v4-seqs arm
    /// on ONE `fa_split_keys` ladder step that the rung also sits on (the straddle law;
    /// both gates are t_kv intervals, so ends-inside means all-inside). The rung is the
    /// round's next power of two — grid/partial sizing only (`n_splits_max` is pure
    /// stride; splits >= ns_eff write the empty partial the combine never reads), so one
    /// captured graph is bit-identical for every round the rung covers.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn full_rung(
        &self,
        model: &crate::hybrid::HybridModel,
        cache: &Cache,
        lo: usize,
        hi: usize,
        t: usize,
        seqs_arms_on: bool,
    ) -> Option<usize> {
        if std::env::var("MEMRA_DSPARK_FULLG_DEBUG").as_deref() == Ok("1") {
            static ONCE: std::sync::Once = std::sync::Once::new();
            let len0 = self
                .fa
                .first()
                .and_then(|&il| cache.kv[il].as_ref())
                .map(|k| k.len);
            ONCE.call_once(|| {
                eprintln!(
                    "[fullg-debug] walk_uniform={} covered={} seqs_arms_on={} fa_rows_on={} t={} lo={} hi={} lin={} fa={} t_cap={} len0={:?}",
                    self.walk_uniform, self.covered, seqs_arms_on, dspark_fa_rows_on(), t, lo, hi,
                    self.lin.len(), self.fa.len(), self.t_cap, len0
                );
            });
        }
        if !self.walk_uniform
            || !seqs_arms_on
            || !dspark_fa_rows_on()
            || t < 2
            || lo != 0
            || hi > self.covered
            || t > self.t_cap
            || self.fa.is_empty()
        {
            return None;
        }
        let cfg = &model.cfg;
        let head_dim_global = cfg.head_dim_k as usize;
        let nkv = cfg.n_head_kv as usize;
        let kvl0 = cache.kv[self.fa[0]].as_ref().unwrap();
        // the z-batched twins read stacked rows at the cache's kv dims — must equal the
        // projection stride (the body's guard, hoisted so ineligible models fall back
        // instead of refusing mid-capture).
        let geom = cfg.full_attention_geometry_at(self.fa[0] as u32);
        let kv_dim = geom.n_head_kv as usize * geom.head_dim_k as usize;
        if kvl0.kv_dim_k != kv_dim || kvl0.kv_dim_v != kv_dim {
            return None;
        }
        let len0 = kvl0.len;
        let (t_kv_first, t_kv_last) = (len0 + 1, len0 + t);
        if !crate::fa_seqs_eligible(t_kv_first, head_dim_global)
            || !crate::fa_seqs_eligible(t_kv_last, head_dim_global)
            || crate::fa_split_keys(t_kv_first, nkv) != crate::fa_split_keys(t_kv_last, nkv)
        {
            return None;
        }
        let rung = t_kv_last.next_power_of_two().max(256);
        if crate::fa_split_keys(rung, nkv) != crate::fa_split_keys(t_kv_last, nkv) {
            return None;
        }
        Some(rung)
    }

    /// Run the WHOLE verify walk [lo, hi) as one captured graph at (vt=t, rung): stage
    /// the residual + refresh the per-vt position staging, capture on first encounter
    /// (2 executing warmups bracketed by a full linear-state save/restore; KV warmup
    /// appends write the exact slots the replay writes — idempotent), launch, then apply
    /// the host bookkeeping the captured body skipped (per-linear-layer parity swap for
    /// odd t, per-fa-layer len bump). Returns the fresh residual.
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::map_entry)] // allow: the init body is fallible (`?`); Entry::or_insert_with cannot propagate errors
    pub(crate) fn run_full(
        &mut self,
        model: &crate::hybrid::HybridModel,
        e: &Engine,
        lo: usize,
        hi: usize,
        x: &CudaSlice<f32>,
        t: usize,
        pos0: usize,
        rung: usize,
        cache: &mut Cache,
    ) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        let n_embd = self.n_embd;
        if !self.stage.contains_key(&t) {
            let xin = e.uninit(t * n_embd)?;
            let xout = e.uninit(t * n_embd)?;
            self.stage.insert(t, (xin, xout));
        }
        if !self.pos_stage.contains_key(&t) {
            self.pos_stage.insert(t, e.htod_i32(&vec![0i32; t])?);
        }
        // Per-round refresh: position contents + input staging (both addresses are baked
        // by the captured bodies; only their CONTENTS change round to round).
        {
            let pos_host: Vec<i32> = (0..t).map(|r| (pos0 + r) as i32).collect();
            let pb = self.pos_stage.get_mut(&t).unwrap();
            e.htod_i32_into(pb, &pos_host)?;
            let (xin, _) = self.stage.get_mut(&t).unwrap();
            e.copy_into(xin, 0, x, t * n_embd)?;
        }
        let key = (t, rung, hi);
        if !self.full.contains_key(&key) {
            // The warmups EXECUTE the whole walk on live state — save every linear
            // layer's conv + canonical ssm first, restore after (KV needs no restore:
            // graph mode never bumps host lens and the appends write this round's own
            // slots).
            for (k, &il) in self.lin.iter().enumerate() {
                let rl = cache.recur[il].as_ref().unwrap();
                e.copy_into(
                    &mut self.save_conv,
                    k * self.conv_words,
                    &rl.conv_state,
                    self.conv_words,
                )?;
                e.copy_into(
                    &mut self.save_ssm,
                    k * self.ssm_words,
                    &rl.ssm_state,
                    self.ssm_words,
                )?;
            }
            let (graph, keeper) = {
                let table_all = &self.table_all;
                let lin_pos = &self.lin_pos;
                let fa_pos = &self.fa_pos;
                let fa_table = &self.fa_table;
                let t_cap = self.t_cap;
                let stash_conv = &mut self.stash_conv;
                let stash_ssm = &mut self.stash_ssm;
                let pos_d: &CudaSlice<i32> = &self.pos_stage[&t];
                let (xin, xout) = self
                    .stage
                    .get_mut(&t)
                    .map(|(a, b)| (&*a, b))
                    .expect("stage bucket created above");
                let cache_ref: &mut Cache = cache;
                let iflag = if std::env::var("MEMRA_DSPARK_VG_AUTOFREE").as_deref() == Ok("1") {
                    cudarc::driver::sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH
                } else {
                    cudarc::driver::sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_USE_NODE_PRIORITY
                };
                e.capture_graph_retained_flags(iflag, move |e| {
                    let mut xc: Option<CudaSlice<f32>> = None;
                    for il in lo..hi {
                        let xr: &CudaSlice<f32> = xc.as_ref().unwrap_or(xin);
                        let nx = if let Some(&k) = lin_pos.get(&il) {
                            model.qwen35_tparallel_linear_layer(
                                e,
                                il,
                                xr,
                                t,
                                cache_ref,
                                None,
                                Some((&mut stash_conv[k], &mut stash_ssm[k])),
                                Some((table_all, k * 6)),
                            )?
                        } else if let Some(&kf) = fa_pos.get(&il) {
                            let mut no_rows: Option<Vec<CudaSlice<i32>>> = None;
                            model.qwen35_tparallel_fa_layer(
                                e,
                                il,
                                xr,
                                t,
                                cache_ref,
                                FaLayerArgs {
                                    pos_d,
                                    pos_rows: &mut no_rows,
                                    pos0,
                                    seqs_append: true,
                                    batch_fa_on: true,
                                    graph_cap: Some((fa_table, kf * 2 * t_cap, rung)),
                                    stream: None,
                                    ckpt: None,
                                },
                            )?
                        } else {
                            return Err(format!(
                                "run_full: layer {il} is neither linear nor full-attention"
                            )
                            .into());
                        };
                        xc = Some(nx);
                    }
                    e.copy_into(xout, 0, xc.as_ref().unwrap(), t * n_embd)?;
                    Ok(())
                })?
            };
            // Undo the net host parity motion of the 3 body runs (each run swaps iff t
            // is odd -> 3 runs = net one swap), then restore the device state the
            // warmups consumed (walk scope only — layers past hi never executed). The
            // launch below then behaves exactly like one run.
            if t % 2 == 1 {
                for &il in &self.lin {
                    if il < lo || il >= hi {
                        continue;
                    }
                    let rl = cache.recur[il].as_mut().unwrap();
                    std::mem::swap(&mut rl.ssm_state, &mut rl.ssm_state_alt);
                }
            }
            for (k, &il) in self.lin.iter().enumerate() {
                if il < lo || il >= hi {
                    continue;
                }
                let rl = cache.recur[il].as_mut().unwrap();
                let (cw, sw) = (self.conv_words, self.ssm_words);
                {
                    let sv = e.view(&self.save_conv, self.lin.len() * cw);
                    let win = sv.slice(k * cw..(k + 1) * cw);
                    e.copy_view_into(&mut rl.conv_state, 0, &win, cw)?;
                }
                {
                    let sv = e.view(&self.save_ssm, self.lin.len() * sw);
                    let win = sv.slice(k * sw..(k + 1) * sw);
                    e.copy_view_into(&mut rl.ssm_state, 0, &win, sw)?;
                }
            }
            if std::env::var("MEMRA_GRAPH_CENSUS").as_deref() == Ok("1")
                && let Ok(c) = crate::graph_update::node_census(&graph)
            {
                eprintln!("[dspark-vg-census] full vt={t} rung={rung} {c:?}");
            }
            self.full.insert(
                key,
                DsparkSegGraph {
                    graph,
                    _keeper: keeper,
                },
            );
        }
        self.full[&key].graph.launch()?;
        // Host bookkeeping for the replayed body (captured host code does not re-run):
        // gdn parity swap per linear layer (t odd), kv len bump per fa layer — scoped
        // to the WALK [lo, hi): the cache can carry extra state slots past it (the MTP
        // head layer's kv) that the walk never touches.
        if t % 2 == 1 {
            for &il in &self.lin {
                if il < lo || il >= hi {
                    continue;
                }
                let rl = cache.recur[il].as_mut().unwrap();
                std::mem::swap(&mut rl.ssm_state, &mut rl.ssm_state_alt);
            }
        }
        for &il in &self.fa {
            if il < lo || il >= hi {
                continue;
            }
            cache.kv[il].as_mut().unwrap().len += t;
        }
        let (_, xout) = self.stage.get(&t).unwrap();
        let mut out = e.uninit(t * n_embd)?;
        e.copy_into(&mut out, 0, xout, t * n_embd)?;
        Ok(out)
    }

    /// Run layers [start, end) (all linear) as one captured graph at this vt: stage the
    /// residual into the bucket's x_in, capture on first encounter (2 executing warmups
    /// bracketed by a segment state save/restore), launch, then apply the host parity
    /// bookkeeping the captured body would have done. Returns the fresh residual.
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::map_entry)] // allow: the init body is fallible (`?`); Entry::or_insert_with cannot propagate errors
    fn run_segment(
        &mut self,
        model: &crate::hybrid::HybridModel,
        e: &Engine,
        start: usize,
        end: usize,
        x: &CudaSlice<f32>,
        t: usize,
        cache: &mut Cache,
    ) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        let n_embd = self.n_embd;
        debug_assert!(end - start <= self.max_run);
        if !self.stage.contains_key(&t) {
            let xin = e.uninit(t * n_embd)?;
            let xout = e.uninit(t * n_embd)?;
            self.stage.insert(t, (xin, xout));
        }
        // Stage the residual at the bucket's baked input address.
        {
            let (xin, _) = self.stage.get_mut(&t).unwrap();
            e.copy_into(xin, 0, x, t * n_embd)?;
        }
        let key = (start, t);
        if !self.graphs.contains_key(&key) {
            // The 2 warmups EXECUTE the segment on live state — save conv + the canonical
            // ssm of every segment layer first, restore after, so the graph's first real
            // launch starts from the exact pre-round state (bytes gated e2e).
            for (k, il) in (start..end).enumerate() {
                let rl = cache.recur[il].as_ref().unwrap();
                e.copy_into(
                    &mut self.save_conv,
                    k * self.conv_words,
                    &rl.conv_state,
                    self.conv_words,
                )?;
                e.copy_into(
                    &mut self.save_ssm,
                    k * self.ssm_words,
                    &rl.ssm_state,
                    self.ssm_words,
                )?;
            }
            let (graph, keeper) = {
                let table_all = &self.table_all;
                let lin_pos = &self.lin_pos;
                let stash_conv = &mut self.stash_conv;
                let stash_ssm = &mut self.stash_ssm;
                let (xin, xout) = self
                    .stage
                    .get_mut(&t)
                    .map(|(a, b)| (&*a, b))
                    .expect("stage bucket created above");
                let cache_ref: &mut Cache = cache;
                // Slice 4 (fa-execupdate lane): USE_NODE_PRIORITY instead of
                // AUTO_FREE_ON_LAUNCH. The slice-3 measured limiter was AUTO_FREE's
                // launch-time mem-pool scan — 25.6 us per cuGraphLaunch x 16 segments
                // = ~0.41 ms/round, most of the eager-launch savings. The captured
                // body's cuMemAllocAsync transients are BALANCED by in-graph frees
                // (every transient drops inside the capture region — the generic
                // capture path's census precedent, 1589/1589), so AUTO_FREE has
                // nothing to reclaim and the graph is legal to instantiate without
                // it; PRIORITY is the flag the gemma slotted door ships for exactly
                // this reason (both alternatives drop the scan; UPLOAD via
                // cuGraphInstantiateWithFlags is WithParams-only and refused).
                // MEMRA_DSPARK_VG_AUTOFREE=1 reverts; MEMRA_GRAPH_CENSUS=1 prints
                // the node census at capture (the ALLOC==FREE receipt).
                let iflag = if std::env::var("MEMRA_DSPARK_VG_AUTOFREE").as_deref() == Ok("1") {
                    cudarc::driver::sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH
                } else {
                    cudarc::driver::sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_USE_NODE_PRIORITY
                };
                e.capture_graph_retained_flags(iflag, move |e| {
                    let mut xc: Option<CudaSlice<f32>> = None;
                    for il in start..end {
                        let k = lin_pos[&il];
                        let xr: &CudaSlice<f32> = xc.as_ref().unwrap_or(xin);
                        let nx = model.qwen35_tparallel_linear_layer(
                            e,
                            il,
                            xr,
                            t,
                            cache_ref,
                            None,
                            Some((&mut stash_conv[k], &mut stash_ssm[k])),
                            Some((table_all, k * 6)),
                        )?;
                        xc = Some(nx);
                    }
                    e.copy_into(xout, 0, xc.as_ref().unwrap(), t * n_embd)?;
                    Ok(())
                })?
            };
            // Undo the net host parity motion of the 3 body runs (each run swaps iff t
            // is odd -> 3 runs = net one swap), then restore the device state the
            // warmups consumed. The launch below then behaves exactly like one run.
            if t % 2 == 1 {
                for il in start..end {
                    let rl = cache.recur[il].as_mut().unwrap();
                    std::mem::swap(&mut rl.ssm_state, &mut rl.ssm_state_alt);
                }
            }
            for (k, il) in (start..end).enumerate() {
                let rl = cache.recur[il].as_mut().unwrap();
                let (cw, sw) = (self.conv_words, self.ssm_words);
                {
                    let sv = e.view(&self.save_conv, self.lin.len() * cw);
                    let win = sv.slice(k * cw..(k + 1) * cw);
                    e.copy_view_into(&mut rl.conv_state, 0, &win, cw)?;
                }
                {
                    let sv = e.view(&self.save_ssm, self.lin.len() * sw);
                    let win = sv.slice(k * sw..(k + 1) * sw);
                    e.copy_view_into(&mut rl.ssm_state, 0, &win, sw)?;
                }
            }
            if std::env::var("MEMRA_GRAPH_CENSUS").as_deref() == Ok("1")
                && let Ok(c) = crate::graph_update::node_census(&graph)
            {
                eprintln!("[dspark-vg-census] seg={start}..{end} vt={t} {c:?}");
            }
            self.graphs.insert(
                key,
                DsparkSegGraph {
                    graph,
                    _keeper: keeper,
                },
            );
        }
        self.graphs[&key].graph.launch()?;
        // Host parity bookkeeping for the replayed body (the captured host swaps do not
        // re-run at replay).
        if t % 2 == 1 {
            for il in start..end {
                let rl = cache.recur[il].as_mut().unwrap();
                std::mem::swap(&mut rl.ssm_state, &mut rl.ssm_state_alt);
            }
        }
        let (_, xout) = self.stage.get(&t).unwrap();
        let mut out = e.uninit(t * n_embd)?;
        e.copy_into(&mut out, 0, xout, t * n_embd)?;
        Ok(out)
    }

    /// Pool freeze check (`dspark_vg_cap`): below the ceiling new keys may capture.
    fn can_capture(&self) -> bool {
        self.graphs.len() + self.full.len() < dspark_vg_cap()
    }

    /// Round-atomic segment-door readiness: TRUE when this round's walk can ride the
    /// per-(segment, vt) graphs without a NEW capture past the pool ceiling — every
    /// linear run in [lo, hi) already has its (run_start, t) key, or capture is still
    /// allowed. FALSE sends the WHOLE round down the eager cols-ckpt walk: a partial
    /// refusal would stash some layers in the ctx slabs and others in the round's cols
    /// while one commit reads only one of them.
    pub(crate) fn segments_ready(
        &self,
        model: &crate::hybrid::HybridModel,
        lo: usize,
        hi: usize,
        t: usize,
    ) -> bool {
        if self.can_capture() {
            return true;
        }
        let mut il = lo;
        while il < hi {
            if matches!(model.layers[il].mixer, Mixer::Linear(_)) {
                let start = il;
                while il < hi && matches!(model.layers[il].mixer, Mixer::Linear(_)) {
                    il += 1;
                }
                if !self.graphs.contains_key(&(start, t)) {
                    return false;
                }
            } else {
                il += 1;
            }
        }
        true
    }

    /// Widest verify window this pool was built for. A caller whose round exceeds it must
    /// take the eager walk: the stash slabs hold `t_capacity() - 1` column rows, and slicing
    /// past them is a panic rather than a refusal.
    pub(crate) fn t_capacity(&self) -> usize {
        self.t_cap
    }

    /// Slab row (conv, ssm) device pointers + lengths for the commit restore of column
    /// `row` (0-based) of layer `il`. None for non-linear layers.
    pub(crate) fn slab_row(
        &self,
        e: &Engine,
        il: usize,
        row: usize,
    ) -> Option<(u64, u64, usize, usize)> {
        use cudarc::driver::DevicePtr;
        let k = *self.lin_pos.get(&il)?;
        let s = &e.gpu.stream();
        let (pc, _g0) = self.stash_conv[k].device_ptr(s);
        let (ps, _g1) = self.stash_ssm[k].device_ptr(s);
        Some((
            pc + (row * self.conv_words * 4) as u64,
            ps + (row * self.ssm_words * 4) as u64,
            self.conv_words,
            self.ssm_words,
        ))
    }
}

impl VerifyCkpt {
    fn new(n_layer: usize) -> Self {
        VerifyCkpt {
            gdn: (0..n_layer).map(|_| None).collect(),
            cols: (0..n_layer).map(|_| None).collect(),
        }
    }
}

/// The stage-0/TX half of one PP verify. The boundary slot is the ownership token: stage 1
/// consumes exactly the slot selected by `tx()` / `tx_pipelined()`, never a slot inferred from
/// a logical round number.
struct VerifyBoundaryTicket {
    rt: &'static crate::pp::PpNRt,
    caller_stream: std::sync::Arc<cudarc::driver::CudaStream>,
    slot: usize,
    pos0: usize,
    t: usize,
    payload: usize,
    n_st: usize,
    pipelined: bool,
    pp_anatomy: bool,
    pp_started: std::time::Instant,
    reverse_ms: f64,
    stage0_ms: f64,
    tx_ms: f64,
    trace: Option<SpecPipeTraceCtx>,
    _walk_owner: crate::pp::PpWalkLease,
}

/// Explicit OPTIPIPE diagnostic control. Forced modes are set only by `optipipe-gate`; the
/// increment-2 controller can also be armed by the server's fresh-process research door.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OptiForkGateMode {
    Disabled,
    Hit,
    Miss,
    Alternate,
    Abort,
    Controller,
}

static OPTI_FORK_GATE_MODE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);
static OPTI_CONTROLLER_THRESHOLD: std::sync::atomic::AtomicU32 =
    std::sync::atomic::AtomicU32::new(0);
static OPTI_FORK_ATTEMPTS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static OPTI_FORK_HITS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static OPTI_FORK_MISSES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static OPTI_FORK_ABORT_DRAINS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static OPTI_FORK_REFUSALS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static OPTI_GATE_CHECKS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static OPTI_GATE_ADMITS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static OPTI_GATE_REJECTS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static OPTI_RECONCILES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static OPTI_WASTED_DRAFT_TOKENS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
static OPTI_SHADOW_DRAFT_TOKENS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
static OPTI_BREAKER_TRIPS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

impl OptiForkGateMode {
    fn code(self) -> u8 {
        match self {
            Self::Disabled => 0,
            Self::Hit => 1,
            Self::Miss => 2,
            Self::Alternate => 3,
            Self::Abort => 4,
            Self::Controller => 5,
        }
    }

    fn configured() -> Self {
        match OPTI_FORK_GATE_MODE.load(std::sync::atomic::Ordering::Relaxed) {
            1 => Self::Hit,
            2 => Self::Miss,
            3 => Self::Alternate,
            4 => Self::Abort,
            5 => Self::Controller,
            _ => Self::Disabled,
        }
    }

    fn action(self, generation: u64) -> OptiForkAction {
        match self {
            Self::Hit => OptiForkAction::Hit,
            Self::Miss => OptiForkAction::Miss,
            Self::Alternate if generation & 1 == 0 => OptiForkAction::Hit,
            Self::Alternate => OptiForkAction::Miss,
            Self::Abort => OptiForkAction::Abort,
            Self::Disabled | Self::Controller => {
                unreachable!("non-forced mode cannot choose a forced fork action")
            }
        }
    }

    fn is_forced(self) -> bool {
        matches!(self, Self::Hit | Self::Miss | Self::Alternate | Self::Abort)
    }
}

/// Arm or disarm the forced harness. Serving uses only `set_optipipe_controller_threshold`.
pub fn set_optipipe_gate_mode(mode: OptiForkGateMode) {
    OPTI_FORK_GATE_MODE.store(mode.code(), std::sync::atomic::Ordering::Relaxed);
}

/// Arm the increment-2 diagnostic controller. The threshold applies to the uncalibrated
/// two-token draft-probability product. Serving can call this only through its explicit
/// fresh-process research door; the absent-door default remains byte-for-byte disabled.
pub fn set_optipipe_controller_threshold(threshold: f32) {
    assert!(threshold.is_finite() && (0.0..=1.0).contains(&threshold));
    OPTI_CONTROLLER_THRESHOLD.store(threshold.to_bits(), std::sync::atomic::Ordering::Relaxed);
    set_optipipe_gate_mode(OptiForkGateMode::Controller);
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct OptiForkGateStats {
    pub attempts: u64,
    pub hits: u64,
    pub misses: u64,
    pub abort_drains: u64,
    pub refusals: u64,
    pub gate_checks: u64,
    pub gate_admits: u64,
    pub gate_rejects: u64,
    pub reconciles: u64,
    pub wasted_draft_tokens: u64,
    pub shadow_draft_tokens: u64,
    pub breaker_trips: u64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct OptiForkStateIdentity {
    pub trunk_kv_bytes: usize,
    pub recurrent_bytes: usize,
    pub scratch_kv_bytes: usize,
    pub hidden_bytes: usize,
}

pub fn reset_optipipe_gate_stats() {
    for counter in [
        &OPTI_FORK_ATTEMPTS,
        &OPTI_FORK_HITS,
        &OPTI_FORK_MISSES,
        &OPTI_FORK_ABORT_DRAINS,
        &OPTI_FORK_REFUSALS,
        &OPTI_GATE_CHECKS,
        &OPTI_GATE_ADMITS,
        &OPTI_GATE_REJECTS,
        &OPTI_RECONCILES,
        &OPTI_WASTED_DRAFT_TOKENS,
        &OPTI_SHADOW_DRAFT_TOKENS,
        &OPTI_BREAKER_TRIPS,
    ] {
        counter.store(0, std::sync::atomic::Ordering::Relaxed);
    }
}

pub fn optipipe_gate_stats() -> OptiForkGateStats {
    let load = |v: &std::sync::atomic::AtomicU64| v.load(std::sync::atomic::Ordering::Relaxed);
    OptiForkGateStats {
        attempts: load(&OPTI_FORK_ATTEMPTS),
        hits: load(&OPTI_FORK_HITS),
        misses: load(&OPTI_FORK_MISSES),
        abort_drains: load(&OPTI_FORK_ABORT_DRAINS),
        refusals: load(&OPTI_FORK_REFUSALS),
        gate_checks: load(&OPTI_GATE_CHECKS),
        gate_admits: load(&OPTI_GATE_ADMITS),
        gate_rejects: load(&OPTI_GATE_REJECTS),
        reconciles: load(&OPTI_RECONCILES),
        wasted_draft_tokens: load(&OPTI_WASTED_DRAFT_TOKENS),
        shadow_draft_tokens: load(&OPTI_SHADOW_DRAFT_TOKENS),
        breaker_trips: load(&OPTI_BREAKER_TRIPS),
    }
}

#[derive(Clone, Copy, Debug)]
struct OptiControllerPolicy {
    threshold: f32,
    consecutive_misses: u8,
    breaker_tripped: bool,
}

impl OptiControllerPolicy {
    fn configured() -> Self {
        Self {
            threshold: f32::from_bits(
                OPTI_CONTROLLER_THRESHOLD.load(std::sync::atomic::Ordering::Relaxed),
            ),
            consecutive_misses: 0,
            breaker_tripped: false,
        }
    }

    fn admit(&self, q_proxy: f32) -> bool {
        q_proxy.is_finite()
            && (0.0..=1.0).contains(&q_proxy)
            && (self.threshold == 0.0 || (!self.breaker_tripped && q_proxy >= self.threshold))
    }

    /// Returns true exactly when this resolution newly trips the three-miss breaker.
    fn resolve(&mut self, hit: bool) -> bool {
        // q*=0 is the lane's explicit unconditional measurement arm. Its purpose is to price
        // every optimistic opportunity, so the safety breaker is measured separately and must
        // not silently turn this arm into "three attempts then serial".
        if self.threshold == 0.0 {
            self.consecutive_misses = 0;
            return false;
        }
        if hit {
            self.consecutive_misses = 0;
            return false;
        }
        self.consecutive_misses = self.consecutive_misses.saturating_add(1);
        if !self.breaker_tripped && self.consecutive_misses >= 3 {
            self.breaker_tripped = true;
            return true;
        }
        false
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum OptiForkAction {
    Hit,
    Miss,
    Abort,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct OptiForkGeneration {
    id: u64,
    slot: usize,
}

#[derive(Default)]
struct OptiForkGenerationTracker {
    next: u64,
    live: [Option<u64>; 2],
}

impl OptiForkGenerationTracker {
    fn reserve(&mut self) -> Result<OptiForkGeneration, Box<dyn std::error::Error>> {
        let generation = OptiForkGeneration {
            id: self.next,
            slot: (self.next & 1) as usize,
        };
        if let Some(live) = self.live[generation.slot] {
            return Err(format!(
                "optipipe snapshot slot {} still owns generation {live}; refusing to overwrite it",
                generation.slot,
            )
            .into());
        }
        self.next += 1;
        self.live[generation.slot] = Some(generation.id);
        Ok(generation)
    }

    fn retire(&mut self, generation: OptiForkGeneration) -> Result<(), Box<dyn std::error::Error>> {
        match self.live[generation.slot] {
            Some(id) if id == generation.id => {
                self.live[generation.slot] = None;
                Ok(())
            }
            other => Err(format!(
                "optipipe generation teardown mismatch: ticket={} slot={} live={other:?}",
                generation.id, generation.slot,
            )
            .into()),
        }
    }
}

struct OptiForkSeedGeneration {
    h_seed: CudaSlice<f32>,
    fill_prev: CudaSlice<f32>,
    scratch_len: usize,
}

/// Allocate or refresh one full checkpoint through the engine that owns each PP stage. The
/// generic cache helper accepts one device and therefore cannot copy GDN state split across
/// devices. KV lengths and position stay host metadata; only recurrent buffers need stage-local
/// device ownership.
fn opti_snapshot_stage_owned(
    e: &Engine,
    cache: &Cache,
    rt: &'static crate::pp::PpNRt,
    fence: &[usize],
) -> Result<crate::cache::CacheSnapshot, Box<dyn std::error::Error>> {
    let n = cache.kv.len();
    let mut snapshot = crate::cache::CacheSnapshot {
        kv_len: vec![None; n],
        tp_kv_len: vec![None; n],
        conv: (0..n).map(|_| None).collect(),
        ssm: (0..n).map(|_| None).collect(),
        pos: cache.pos,
    };
    opti_snapshot_stage_owned_into(e, cache, rt, fence, &mut snapshot)?;
    Ok(snapshot)
}

fn opti_snapshot_stage_owned_into(
    e: &Engine,
    cache: &Cache,
    rt: &'static crate::pp::PpNRt,
    fence: &[usize],
    snapshot: &mut crate::cache::CacheSnapshot,
) -> Result<(), Box<dyn std::error::Error>> {
    if fence.len() != rt.n_stages() + 1
        || snapshot.kv_len.len() != cache.kv.len()
        || snapshot.tp_kv_len.len() != cache.tp_kv.len()
    {
        return Err("optipipe stage-owned snapshot shape mismatch".into());
    }
    for stage in 0..rt.n_stages() {
        opti_snapshot_one_stage_owned_into(e, cache, rt, fence, stage, snapshot)?;
    }
    snapshot.pos = cache.pos;
    Ok(())
}

/// Refresh one PP stage of a checkpoint. Increment 2 uses this split form so stage 0's
/// optimistic post-N state is captured before N+1 stage 0 is queued, while stage 1's matching
/// post-N state is captured only after N stage 1 is enqueued. Calling the all-stage helper at
/// either point would capture one side of the fork at the wrong generation.
fn opti_snapshot_one_stage_owned_into(
    e: &Engine,
    cache: &Cache,
    rt: &'static crate::pp::PpNRt,
    fence: &[usize],
    stage: usize,
    snapshot: &mut crate::cache::CacheSnapshot,
) -> Result<(), Box<dyn std::error::Error>> {
    if fence.len() != rt.n_stages() + 1
        || snapshot.kv_len.len() != cache.kv.len()
        || snapshot.tp_kv_len.len() != cache.tp_kv.len()
        || stage >= rt.n_stages()
    {
        return Err("optipipe single-stage snapshot shape mismatch".into());
    }
    let _scope = rt.enter(stage);
    let owner = rt.engine(stage, e);
    for il in fence[stage]..fence[stage + 1] {
        snapshot.kv_len[il] = cache.kv[il].as_ref().map(|kv| kv.len);
        snapshot.tp_kv_len[il] = cache.tp_kv[il]
            .as_ref()
            .map(crate::tp::ResidentTpKvCache::committed_len);
        match &cache.recur[il] {
            Some(recur) => {
                match snapshot.conv[il].as_mut() {
                    Some(dst) => {
                        owner.copy_into(dst, 0, &recur.conv_state, recur.conv_state.len())?
                    }
                    None => snapshot.conv[il] = Some(owner.clone_dtod(&recur.conv_state)?),
                }
                match snapshot.ssm[il].as_mut() {
                    Some(dst) => {
                        owner.copy_into(dst, 0, &recur.ssm_state, recur.ssm_state.len())?
                    }
                    None => snapshot.ssm[il] = Some(owner.clone_dtod(&recur.ssm_state)?),
                }
            }
            None if snapshot.conv[il].is_some() || snapshot.ssm[il].is_some() => {
                return Err(
                    format!("optipipe stage-owned snapshot layer {il} changed shape").into(),
                );
            }
            None => {}
        }
    }
    snapshot.pos = cache.pos;
    Ok(())
}

/// Increment-1 persistent fork state. Exactly two snapshot/seed slots alternate; a live ticket
/// names its generation and keeps teardown fail-closed. Only stage 0 is allowed to mutate before
/// resolve, so the reconcile tables and conditional restores are stage-local.
struct OptiForkState {
    mode: OptiForkGateMode,
    controller: Option<OptiControllerPolicy>,
    generations: OptiForkGenerationTracker,
    active_snapshot_slot: usize,
    alternate_snapshot: crate::cache::CacheSnapshot,
    seeds: [OptiForkSeedGeneration; 2],
    rt: &'static crate::pp::PpNRt,
    fence: [usize; 3],
    split: usize,
    len_ptrs: CudaSlice<u64>,
    saved_lens: CudaSlice<i32>,
    forced_acc: CudaSlice<u32>,
    valid: CudaSlice<u32>,
    stage0_stream: std::sync::Arc<cudarc::driver::CudaStream>,
    logical_payload_bytes: [usize; 2],
}

struct OptiForkTicket {
    generation: OptiForkGeneration,
    boundary: Option<VerifyBoundaryTicket>,
    drain: std::sync::Arc<cudarc::driver::CudaStream>,
    settled: bool,
}

struct OptiControllerTicket {
    generation: OptiForkGeneration,
    boundary: Option<VerifyBoundaryTicket>,
    ckpt: Option<VerifyCkpt>,
    verify_tokens: [u32; 2],
    draft_prob: f32,
    eager_seed: Option<CudaSlice<f32>>,
    q_proxy: f32,
    scratch_len: usize,
    issued_at: std::time::Instant,
    drain: std::sync::Arc<cudarc::driver::CudaStream>,
    settled: bool,
}

struct OptiControllerPrepared {
    verify_tokens: [u32; 2],
    draft_prob: f32,
    eager_seed: Option<CudaSlice<f32>>,
    q_proxy: f32,
    scratch_len: usize,
}

impl OptiControllerTicket {
    fn take_boundary(&mut self) -> VerifyBoundaryTicket {
        self.boundary
            .take()
            .expect("controller boundary ticket already consumed")
    }

    fn take_ckpt(&mut self) -> VerifyCkpt {
        self.ckpt
            .take()
            .expect("controller verify checkpoint already consumed")
    }

    fn take_eager_seed(&mut self) -> Option<CudaSlice<f32>> {
        self.eager_seed.take()
    }

    fn settle(&mut self) {
        self.settled = true;
    }
}

impl Drop for OptiControllerTicket {
    fn drop(&mut self) {
        if !self.settled {
            let _ = self.drain.synchronize();
            OPTI_FORK_ABORT_DRAINS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
    }
}

impl OptiForkTicket {
    fn take_boundary(&mut self) -> VerifyBoundaryTicket {
        self.boundary
            .take()
            .expect("fork ticket boundary already consumed")
    }

    fn settle(&mut self) {
        self.settled = true;
    }
}

impl Drop for OptiForkTicket {
    fn drop(&mut self) {
        if !self.settled {
            let _ = self.drain.synchronize();
            OPTI_FORK_ABORT_DRAINS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
    }
}

impl OptiForkState {
    #[allow(clippy::too_many_arguments)]
    fn new(
        e: &Engine,
        cache: &Cache,
        mode: OptiForkGateMode,
        alternate_snapshot: crate::cache::CacheSnapshot,
        h_seed: &CudaSlice<f32>,
        fill_prev: &CudaSlice<f32>,
        rt: &'static crate::pp::PpNRt,
        split: usize,
        n_layer: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let fence = [0, split, n_layer];
        let mut logical_payload_bytes = [0usize; 2];
        for stage in 0..2 {
            for il in fence[stage]..fence[stage + 1] {
                logical_payload_bytes[stage] += alternate_snapshot.conv[il]
                    .as_ref()
                    .map_or(0, |v| v.len() * std::mem::size_of::<f32>());
                logical_payload_bytes[stage] += alternate_snapshot.ssm[il]
                    .as_ref()
                    .map_or(0, |v| v.len() * std::mem::size_of::<f32>());
            }
        }
        let seeds = [
            OptiForkSeedGeneration {
                h_seed: e.clone_dtod(h_seed)?,
                fill_prev: e.clone_dtod(fill_prev)?,
                scratch_len: 0,
            },
            OptiForkSeedGeneration {
                h_seed: e.clone_dtod(h_seed)?,
                fill_prev: e.clone_dtod(fill_prev)?,
                scratch_len: 0,
            },
        ];
        let (len_ptrs, saved_lens, forced_acc, valid, stage0_stream) = {
            let _stage = rt.enter(0);
            let e0 = rt.engine(0, e);
            (
                crate::round_stream::kv_len_ptr_table_range(e0, cache, 0..split, None)?,
                e0.htod_i32(&vec![0; split])?,
                e0.alloc_u32_zeroed(2)?,
                e0.alloc_u32_zeroed(1)?,
                e0.stream(),
            )
        };
        logical_payload_bytes[0] += seeds
            .iter()
            .map(|seed| (seed.h_seed.len() + seed.fill_prev.len()) * std::mem::size_of::<f32>())
            .sum::<usize>();
        logical_payload_bytes[0] += len_ptrs.len() * std::mem::size_of::<u64>()
            + saved_lens.len() * std::mem::size_of::<i32>()
            + forced_acc.len() * std::mem::size_of::<u32>()
            + valid.len() * std::mem::size_of::<u32>();
        Ok(Self {
            mode,
            controller: (mode == OptiForkGateMode::Controller)
                .then(OptiControllerPolicy::configured),
            generations: OptiForkGenerationTracker::default(),
            active_snapshot_slot: 0,
            alternate_snapshot,
            seeds,
            rt,
            fence,
            split,
            len_ptrs,
            saved_lens,
            forced_acc,
            valid,
            stage0_stream,
            logical_payload_bytes,
        })
    }

    fn reserve(
        &mut self,
        current_snapshot: &mut crate::cache::CacheSnapshot,
    ) -> Result<OptiForkGeneration, Box<dyn std::error::Error>> {
        let generation = self.generations.reserve()?;
        if generation.slot != self.active_snapshot_slot {
            std::mem::swap(current_snapshot, &mut self.alternate_snapshot);
            self.active_snapshot_slot = generation.slot;
        }
        Ok(generation)
    }

    fn capture_seed(
        &mut self,
        e: &Engine,
        generation: OptiForkGeneration,
        h_seed: &CudaSlice<f32>,
        fill_prev: &CudaSlice<f32>,
        scratch_len: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let seed = &mut self.seeds[generation.slot];
        e.copy_into(&mut seed.h_seed, 0, h_seed, h_seed.len())?;
        e.copy_into(&mut seed.fill_prev, 0, fill_prev, fill_prev.len())?;
        seed.scratch_len = scratch_len;
        Ok(())
    }

    fn ticket(
        &self,
        generation: OptiForkGeneration,
        boundary: VerifyBoundaryTicket,
    ) -> OptiForkTicket {
        OptiForkTicket {
            generation,
            boundary: Some(boundary),
            drain: self.stage0_stream.clone(),
            settled: false,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn controller_ticket(
        &self,
        generation: OptiForkGeneration,
        boundary: VerifyBoundaryTicket,
        ckpt: VerifyCkpt,
        verify_tokens: [u32; 2],
        draft_prob: f32,
        eager_seed: Option<CudaSlice<f32>>,
        q_proxy: f32,
        scratch_len: usize,
    ) -> OptiControllerTicket {
        OptiControllerTicket {
            generation,
            boundary: Some(boundary),
            ckpt: Some(ckpt),
            verify_tokens,
            draft_prob,
            eager_seed,
            q_proxy,
            scratch_len,
            issued_at: std::time::Instant::now(),
            drain: self.stage0_stream.clone(),
            settled: false,
        }
    }

    fn reserve_successor(&mut self) -> Result<OptiForkGeneration, Box<dyn std::error::Error>> {
        self.generations.reserve()
    }

    fn successor_snapshot_mut(&mut self) -> &mut crate::cache::CacheSnapshot {
        &mut self.alternate_snapshot
    }

    fn promote_successor_snapshot(
        &mut self,
        current_snapshot: &mut crate::cache::CacheSnapshot,
        generation: OptiForkGeneration,
    ) {
        std::mem::swap(current_snapshot, &mut self.alternate_snapshot);
        self.active_snapshot_slot = generation.slot;
    }

    fn queue_actual_reconcile(
        &mut self,
        e: &Engine,
        snapshot: &crate::cache::CacheSnapshot,
        acc: &CudaSlice<u32>,
        optimistic_pending: u32,
        base: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let saved: Vec<i32> = (0..self.split)
            .map(|il| snapshot.kv_len[il].map(|v| v as i32).unwrap_or(0))
            .collect();
        // Serving keeps the caller/accept walk on the head (stage-1) device. Record the accept
        // decision point there and append a wait to stage 0 after its optimistic successor/TX;
        // the validity/reconcile kernels must never peer-read acc before it is written. The
        // increment-1 harness uses primary stage 0, where stream order already provides this.
        if self.rt.engine(0, e).ctx().ordinal() != e.ctx().ordinal() {
            self.rt.fence_stages_behind(&e.stream())?;
        }
        let _stage = self.rt.enter(0);
        let e0 = self.rt.engine(0, e);
        e0.htod_i32_into(&mut self.saved_lens, &saved)?;
        e0.spec_fork_valid(acc, optimistic_pending, &mut self.valid)?;
        e0.spec_fork_reconcile_kv(
            &self.len_ptrs,
            &self.saved_lens,
            acc,
            &self.valid,
            base,
            self.split,
        )
    }

    fn finish_actual_reconcile(
        &mut self,
        e: &Engine,
        cache: &mut Cache,
        snapshot: &crate::cache::CacheSnapshot,
        n_acc: usize,
        base: usize,
        hit: bool,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if hit {
            return Ok(());
        }
        let len_delta = base + n_acc;
        for il in 0..self.split {
            if let (Some(kv), Some(saved)) = (cache.kv[il].as_mut(), snapshot.kv_len[il]) {
                kv.len = saved + len_delta;
            }
        }
        {
            let _stage = self.rt.enter(1);
            let e1 = self.rt.engine(1, e);
            for il in self.split..self.fence[2] {
                if let (Some(kv), Some(saved)) = (cache.kv[il].as_mut(), snapshot.kv_len[il]) {
                    kv.len = saved + len_delta;
                    e1.set_i32_one(&mut kv.len_d, kv.len as i32)?;
                }
            }
        }
        self.rt.publish_to(0, &e.stream())?;
        Ok(())
    }

    fn cancel_controller_ticket(
        &mut self,
        e: &Engine,
        cache: &mut Cache,
        scratch: &mut MtpScratch,
        snapshot: &crate::cache::CacheSnapshot,
        ticket: &mut OptiControllerTicket,
    ) -> Result<(), Box<dyn std::error::Error>> {
        {
            let _stage = self.rt.enter(0);
            let e0 = self.rt.engine(0, e);
            for il in 0..self.split {
                if let (Some(kv), Some(saved)) = (cache.kv[il].as_mut(), snapshot.kv_len[il]) {
                    kv.len = saved;
                    e0.set_i32_one(&mut kv.len_d, saved as i32)?;
                }
            }
        }
        scratch.set_len(e, snapshot.pos)?;
        ticket.settle();
        self.generations.retire(ticket.generation)?;
        OPTI_FORK_ABORT_DRAINS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        OPTI_WASTED_DRAFT_TOKENS.fetch_add(2, std::sync::atomic::Ordering::Relaxed);
        eprintln!(
            "[opti-controller] tail-drain generation={} slot={}",
            ticket.generation.id, ticket.generation.slot,
        );
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn reconcile(
        &mut self,
        e: &Engine,
        cache: &mut Cache,
        scratch: &mut MtpScratch,
        snapshot: &crate::cache::CacheSnapshot,
        h_seed: &mut CudaSlice<f32>,
        fill_prev: &mut CudaSlice<f32>,
        generation: OptiForkGeneration,
        action: OptiForkAction,
        optimistic_pending: u32,
    ) -> Result<(), Box<dyn std::error::Error>> {
        debug_assert!(action != OptiForkAction::Abort);
        let miss_started = std::time::Instant::now();
        let keep = action == OptiForkAction::Hit;
        let saved: Vec<i32> = (0..self.split)
            .map(|il| snapshot.kv_len[il].map(|v| v as i32).unwrap_or(0))
            .collect();
        let seed = &self.seeds[generation.slot];
        {
            let _stage = self.rt.enter(0);
            let e0 = self.rt.engine(0, e);
            e0.htod_i32_into(&mut self.saved_lens, &saved)?;
            let forced = if keep {
                [1u32, optimistic_pending]
            } else {
                [0u32, optimistic_pending]
            };
            e0.htod_u32_into(&mut self.forced_acc, &forced)?;
            e0.spec_fork_valid(&self.forced_acc, optimistic_pending, &mut self.valid)?;
            e0.spec_fork_reconcile_kv(
                &self.len_ptrs,
                &self.saved_lens,
                &self.forced_acc,
                &self.valid,
                0,
                self.split,
            )?;
            for il in 0..self.split {
                if let Some(recur) = cache.recur[il].as_mut() {
                    let conv = snapshot.conv[il]
                        .as_ref()
                        .ok_or("optipipe stage0 snapshot missing conv state")?;
                    let ssm = snapshot.ssm[il]
                        .as_ref()
                        .ok_or("optipipe stage0 snapshot missing ssm state")?;
                    e0.spec_fork_restore_f32(conv, &mut recur.conv_state, &self.valid)?;
                    e0.spec_fork_restore_f32(ssm, &mut recur.ssm_state, &self.valid)?;
                }
            }
            e0.spec_fork_restore_f32(&seed.h_seed, h_seed, &self.valid)?;
            e0.spec_fork_restore_f32(&seed.fill_prev, fill_prev, &self.valid)?;
        }

        if keep {
            OPTI_FORK_HITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return Ok(());
        }

        for il in 0..self.split {
            if let (Some(kv), Some(saved)) = (cache.kv[il].as_mut(), snapshot.kv_len[il]) {
                kv.len = saved;
            }
        }
        scratch.set_len(e, seed.scratch_len)?;
        // Targeted E_restart: publish only stage 0's reconcile to the caller, then bound the
        // forced diagnostic so the retained number is the actual miss cost, not enqueue time.
        let caller = e.stream();
        self.rt.publish_to(0, &caller)?;
        caller.synchronize()?;
        let miss_ms = miss_started.elapsed().as_secs_f64() * 1e3;
        eprintln!(
            "[opti-fork-reconcile] generation={} slot={} miss_ms={miss_ms:.3}",
            generation.id, generation.slot,
        );
        OPTI_FORK_MISSES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }

    fn retire(&mut self, generation: OptiForkGeneration) -> Result<(), Box<dyn std::error::Error>> {
        self.generations.retire(generation)
    }
}

/// MEMRA_SPEC_ROUND_PROF counters: whole-round wall, so the round can be weighed against the
/// draft-step ([spec-anatomy]) and verify-walk ([tcol-prof]) splits we already print.
static ROUND_PROF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
static ROUND_MS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static ROUND_N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

fn validate_tp_kv_snapshot_shape(
    tp_kv: &[Option<crate::tp::ResidentTpKvCache>],
    saved_lens: &[Option<usize>],
) -> Result<(), Box<dyn std::error::Error>> {
    if tp_kv.len() != saved_lens.len() {
        return Err("spec TP KV snapshot shape mismatch".into());
    }
    for (layer, (cache, saved)) in tp_kv.iter().zip(saved_lens).enumerate() {
        if cache.is_some() != saved.is_some() {
            return Err(
                format!("spec TP KV layer {layer} changed shape since its snapshot").into(),
            );
        }
    }
    Ok(())
}

impl HybridModel {
    /// memra#128: the canonical-to-rank byte copy that `5e0fffb97` added to
    /// `restore_step_tp_kv_verified_prefix`. OFF by default (written decision, docs/FLAGS.md
    /// `MEMRA_STEP_TP_KV_RESTORE`): on step37 NVFP4 TP2 the only shapes that pass the
    /// production acceptance gate are the engine before the copy (arm A) and the copy skipped
    /// on every step-TP layer (arm F, byte-identical answers to A); the copy on any layer
    /// spliced answers or shifted decode (darklanes research/memra128-bisect-20260903).
    /// `1` re-enables the copy for ordinary-commit layers; on-device-written layers
    /// (`rows_external`) are skipped either way, their canonical bytes are stale.
    fn step_tp_kv_restore_copy_on() -> bool {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ON.get_or_init(|| std::env::var("MEMRA_STEP_TP_KV_RESTORE").ok().as_deref() == Some("1"))
    }

    fn restore_step_tp_kv_verified_prefix(
        &self,
        e: &Engine,
        cache: &mut Cache,
        snap: &crate::cache::CacheSnapshot,
        accepted: usize,
        // memra#128: what an externally-written (dcw / fa2) layer needs from this call.
        // PARTIAL accept (commit_verified_prefix): 5e0fffb97 replaced the standalone
        // rewind_tp_kv_verified_prefix with this restore, so the length shrink to
        // saved+accepted must still happen here - without it E ran with distributed=259
        // against local=257. FULL accept: before 5e0fffb97 nothing touched the
        // distributed length there and it was right (arm A passed); rewinding to
        // saved+t_v shrinks it by one and the next verify's SWA ring view falls off the
        // end ("view [5148,5152) is outside resident [0,5151)", arm E2).
        rewind_external: bool,
    ) -> Result<(), Box<dyn std::error::Error>> {
        validate_tp_kv_snapshot_shape(&cache.tp_kv, &snap.tp_kv_len)?;
        e.stream().synchronize()?;
        {
            let (local_layers, distributed_layers) = (&cache.kv, &mut cache.tp_kv);
            let stream = e.gpu.stream();
            let mut runtime: Option<std::sync::Arc<crate::tp::TpE4m3HostBounce>> = None;
            let mut uniform_runtime = true;
            let mut batch = Vec::new();
            for (il, (distributed_slot, local_slot)) in distributed_layers
                .iter_mut()
                .zip(local_layers.iter())
                .enumerate()
            {
                let (Some(distributed), Some(saved)) =
                    (distributed_slot.as_mut(), snap.tp_kv_len[il])
                else {
                    continue;
                };
                // memra#128: on the dcw / fa2 verify path the rank rows for [saved, target)
                // were written on-device and the canonical cache holds only stale bytes for
                // them (the verify bumps `local.len` and writes nothing). Copying those over
                // the correct rank rows is exactly what spliced two requests' answers
                // together on step-3.7-flash. The rewind already kept the right rows; skip.
                let target = saved
                    .checked_add(accepted)
                    .ok_or("spec TP KV batch restore length overflow")?;
                if !Self::step_tp_kv_restore_copy_on() || distributed.rows_external() {
                    // Rank rows already right (written on-device). Length: see the
                    // `rewind_external` note on the signature.
                    if rewind_external {
                        distributed.rewind_to(target)?;
                    }
                    continue;
                }
                let local = local_slot
                    .as_ref()
                    .ok_or_else(|| format!("spec TP KV layer {il} lost its owning cache"))?;
                if local.len < target {
                    return Err(format!(
                        "spec TP KV layer {il} local length {} precedes restore target {target}",
                        local.len
                    )
                    .into());
                }
                let physical = local.physical_rows(saved, target)?;
                if physical.len() != accepted {
                    return Err(format!(
                        "spec TP KV layer {il} restore [{saved},{target}) is not contiguous"
                    )
                    .into());
                }
                let Mixer::Full(fa) = &self.layers[il].mixer else {
                    return Err(format!("spec TP KV layer {il} is not full attention").into());
                };
                let tp = fa
                    .step_tp_qkv
                    .as_ref()
                    .ok_or_else(|| format!("spec TP KV layer {il} lost its TP runtime"))?;
                if let Some(first) = runtime.as_ref() {
                    if !std::sync::Arc::ptr_eq(first, &tp.runtime) {
                        uniform_runtime = false;
                        break;
                    }
                } else {
                    runtime = Some(tp.runtime.clone());
                }
                use cudarc::driver::DevicePtr;
                let (k_base, _k_guard) = local.k.device_ptr(&stream);
                let (v_base, _v_guard) = local.v.device_ptr(&stream);
                batch.push(crate::tp::TpKvVerifiedLayer {
                    cache: distributed,
                    start: saved,
                    logical_len: target,
                    source_k_raw: k_base + (physical.start * local.k_tok_bytes) as u64,
                    source_v_raw: v_base + (physical.start * local.v_tok_bytes) as u64,
                    source_k_tok_bytes: local.k_tok_bytes,
                    source_v_tok_bytes: local.v_tok_bytes,
                });
            }
            if uniform_runtime
                && let Some(runtime) = runtime
                && runtime.restore_tp_kv_layers_from_device(&mut batch)?
            {
                return Ok(());
            }
        }
        for il in 0..self.layers.len() {
            let (Some(distributed), Some(saved)) = (cache.tp_kv[il].as_mut(), snap.tp_kv_len[il])
            else {
                continue;
            };
            let target = saved
                .checked_add(accepted)
                .ok_or("spec TP KV restore length overflow")?;
            if !Self::step_tp_kv_restore_copy_on() || distributed.rows_external() {
                if rewind_external {
                    distributed.rewind_to(target)?;
                }
                continue;
            }
            let local = cache.kv[il]
                .as_ref()
                .ok_or_else(|| format!("spec TP KV layer {il} lost its owning cache"))?;
            if local.len < target {
                return Err(format!(
                    "spec TP KV layer {il} local length {} precedes restore target {target}",
                    local.len
                )
                .into());
            }
            let physical = local.physical_rows(saved, target)?;
            if physical.len() != accepted {
                return Err(format!(
                    "spec TP KV layer {il} restore [{saved},{target}) is not contiguous"
                )
                .into());
            }
            use cudarc::driver::DevicePtr;
            let stream = e.gpu.stream();
            let (k_base, _k_guard) = local.k.device_ptr(&stream);
            let (v_base, _v_guard) = local.v.device_ptr(&stream);
            let k_raw = k_base + (physical.start * local.k_tok_bytes) as u64;
            let v_raw = v_base + (physical.start * local.v_tok_bytes) as u64;
            let Mixer::Full(fa) = &self.layers[il].mixer else {
                return Err(format!("spec TP KV layer {il} is not full attention").into());
            };
            let tp = fa
                .step_tp_qkv
                .as_ref()
                .ok_or_else(|| format!("spec TP KV layer {il} lost its TP runtime"))?;
            tp.runtime.restore_tp_kv_rows_from_device(
                distributed,
                saved,
                target,
                k_raw,
                v_raw,
                local.k_tok_bytes,
                local.v_tok_bytes,
            )?;
        }
        Ok(())
    }

    fn mtp_head_count(&self) -> usize {
        usize::from(self.mtp.is_some()) + self.mtp_extra.len()
    }

    fn mtp_head_at(&self, index: usize) -> &MtpHead {
        if index == 0 {
            self.mtp.as_ref().expect("MTP head 0 is unavailable")
        } else {
            &self.mtp_extra[index - 1]
        }
    }

    fn new_mtp_scratch(
        &self,
        e: &Engine,
        cap: usize,
    ) -> Result<MtpScratch, Box<dyn std::error::Error>> {
        let mut scratch = MtpScratch::new(
            e,
            &self.cfg,
            &self.plan,
            cap,
            self.mtp.as_ref().and_then(|head| head.geom.as_ref()),
        )?;
        for head in &self.mtp_extra {
            scratch.push_plane(e, &self.cfg, &self.plan, head.geom.as_ref())?;
        }
        Ok(scratch)
    }

    fn opti_graph_draft_step(
        &self,
        e: &Engine,
        mtp: &MtpHead,
        dctx: &mut DraftGraphCtx,
        scratch: &mut MtpScratch,
        d_vocab: usize,
    ) -> Result<(u32, f32), Box<dyn std::error::Error>> {
        // dcw door: one replay appends one device-counter row; pre-arm ring headroom
        // host-side before launching (no-op on flat planes).
        if step35_draft_dcw_on() {
            scratch.ensure_dcw_headroom(e, 2)?;
        }
        dctx.graph
            .as_ref()
            .ok_or("optipipe controller requires the greedy draft graph")?
            .launch()?;
        scratch.kv.len += 1;
        let idx = e.dtoh_u32_one(&dctx.g_tok)?;
        if (idx as usize) >= d_vocab {
            return Err(
                format!("optipipe draft argmax sentinel 0x{idx:08x} >= d_vocab {d_vocab}").into(),
            );
        }
        let probability = e.dtoh(&dctx.g_p)?[0];
        if !probability.is_finite() || !(0.0..=1.0).contains(&probability) {
            return Err(format!("optipipe draft probability is invalid: {probability}").into());
        }
        let token = match &mtp.d2t {
            Some(map) => map[idx as usize],
            None => idx,
        };
        if token != idx {
            e.set_u32_one(&mut dctx.g_tok, token)?;
        }
        Ok((token, probability))
    }

    #[allow(clippy::too_many_arguments)]
    fn opti_controller_draft_step(
        &self,
        e: &Engine,
        mtp: &MtpHead,
        dctx: &mut DraftGraphCtx,
        scratch: &mut MtpScratch,
        d_vocab: usize,
        eager_state: &mut Option<(u32, CudaSlice<f32>)>,
        eager_pos: usize,
        embd_dev: Option<(&CudaSlice<u8>, i32, usize)>,
        round_graph_ok: bool,
    ) -> Result<(u32, f32), Box<dyn std::error::Error>> {
        // GRAPH-LAUNCH HEADROOM GUARD (see GRAPH_LAUNCH_MIN_FREE): `round_graph_ok` is
        // the round's headroom snapshot. Below the floor the main draft arm already ran
        // eager (13651-class gate), which seeded `eager_state`, so the controller probe
        // rides its eager twin below instead of replaying the draft graph into an
        // exhausted card. The seed-unavailable Err beneath stays the recoverable
        // fail-closed for the shapes that never seed it.
        if dctx.graph.is_some() && round_graph_ok {
            return self.opti_graph_draft_step(e, mtp, dctx, scratch, d_vocab);
        }
        let (input_token, input_seed) = eager_state
            .take()
            .ok_or("optipipe eager continuation seed is unavailable")?;
        let (logits, next_seed) = self.mtp_head_forward_dev(
            e,
            mtp,
            input_token,
            &input_seed,
            scratch,
            eager_pos,
            embd_dev,
            None,
        )?;
        let token_d = e.argmax_token_device(&logits, d_vocab)?;
        let idx = e.dtoh_u32_one(&token_d)?;
        if (idx as usize) >= d_vocab {
            return Err(format!(
                "optipipe eager draft argmax sentinel 0x{idx:08x} >= d_vocab {d_vocab}"
            )
            .into());
        }
        let probability_d = e.prob_of_token_device(&logits, &token_d, d_vocab)?;
        let probability = e.dtoh(&probability_d)?[0];
        if !probability.is_finite() || !(0.0..=1.0).contains(&probability) {
            return Err(
                format!("optipipe eager draft probability is invalid: {probability}").into(),
            );
        }
        let token = match &mtp.d2t {
            Some(map) => map[idx as usize],
            None => idx,
        };
        *eager_state = Some((token, next_seed));
        Ok((token, probability))
    }

    /// NextN head forward for ONE draft token (§A ops 1-13, T=1).
    /// Inputs: `e_tok` = the token to predict FROM (last committed / previous draft); `h_seed` =
    /// the trunk's pre-output_norm hidden of that token (§A op 2 input). `mtp_pos` = absolute
    /// position of the token being predicted from. Returns (draft_logits[n_vocab] host, h_nextn dev).
    /// `h_nextn` (§A op 10) becomes `h_seed` for the next autoregressive draft step.
    /// Device-resident: returns draft logits ON DEVICE (no [n_vocab] dtoh). The greedy draft
    /// loop only needs argmax — paired with `argmax_token_device` this cuts the ~600KB logits
    /// transfer + host argmax per draft token from the K-token draft chain.
    #[allow(clippy::too_many_arguments)]
    fn mtp_head_forward_dev(
        &self,
        e: &Engine,
        mtp: &MtpHead,
        e_tok: u32,
        h_seed: &CudaSlice<f32>,
        scratch: &mut MtpScratch,
        mtp_pos: usize,
        embd_dev: Option<(&CudaSlice<u8>, i32, usize)>,
        mask: Option<(&CudaSlice<u32>, usize)>,
    ) -> Result<(CudaSlice<f32>, CudaSlice<f32>), Box<dyn std::error::Error>> {
        self.mtp_head_forward_dev_at(e, mtp, e_tok, h_seed, scratch, 0, mtp_pos, embd_dev, mask)
    }

    #[allow(clippy::too_many_arguments)]
    fn mtp_head_forward_dev_at(
        &self,
        e: &Engine,
        mtp: &MtpHead,
        e_tok: u32,
        h_seed: &CudaSlice<f32>,
        scratch: &mut MtpScratch,
        scratch_index: usize,
        mtp_pos: usize,
        embd_dev: Option<(&CudaSlice<u8>, i32, usize)>,
        // DRAFT-SIDE GRAMMAR MASK (lane/draft-mask): (packed draft-vocab allowed set, words).
        // Applied to the head logits BEFORE they are returned, so every consumer (argmax,
        // gumbel draw, p-min prob) sees the grammar-legal row. None = unmasked (pre-lane).
        mask: Option<(&CudaSlice<u32>, usize)>,
    ) -> Result<(CudaSlice<f32>, CudaSlice<f32>), Box<dyn std::error::Error>> {
        // MEMRA_SPEC_ANATOMY=1 — eager-step phase timers (diagnostic only). Phase boundaries
        // sync the stream, so absolute time inflates; the BREAKDOWN is the signal. Cumulative
        // summary on stderr every 128 steps: glue (embed..attn_norm), attn, ffn, head.
        use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
        static ANAT_NS: [AtomicU64; 5] = [
            AtomicU64::new(0),
            AtomicU64::new(0),
            AtomicU64::new(0),
            AtomicU64::new(0),
            AtomicU64::new(0),
        ];
        static ANAT_STEPS: AtomicU64 = AtomicU64::new(0);
        let anat = {
            static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            *ON.get_or_init(|| std::env::var("MEMRA_SPEC_ANATOMY").as_deref() == Ok("1"))
        };
        if anat {
            e.stream().synchronize()?; // drain prior queue so phase 0 starts clean
        }
        let t_all = std::time::Instant::now();
        let mut t_ph = std::time::Instant::now();
        let anat_mark = |i: usize,
                         e: &Engine,
                         t: &mut std::time::Instant|
         -> Result<(), Box<dyn std::error::Error>> {
            if anat {
                e.stream().synchronize()?;
                ANAT_NS[i].fetch_add(t.elapsed().as_nanos() as u64, Relaxed);
                *t = std::time::Instant::now();
            }
            Ok(())
        };
        let cfg = &self.cfg;
        let n_embd = cfg.n_embd as usize;
        // Distilled-student geometry: the block runs at the INNER width `di` (eh_proj out /
        // attn / ffn); the n_embd interface (embed, norms in, carrier out, head in) is unchanged.
        let di = mtp.geom.as_ref().map(|g| g.d_inner).unwrap_or(n_embd);
        let eps = cfg.rms_eps;
        let pos_d = e.htod_i32(&[mtp_pos as i32])?;

        // op A: a resident table transfers one 4B token id. The exact host-row capacity path
        // expands this one row on CPU and transfers n_embd f32 values instead.
        let e_emb = match embd_dev {
            Some((g, qt, rb)) => e.embed_gather_device_t(g, &[e_tok], n_embd, qt, rb)?,
            None => e.htod(&self.embd.try_gather(n_embd, &[e_tok])?)?,
        };

        // op 1/2: e_norm = RMSNorm(e, enorm); h_norm = RMSNorm(h_seed, hnorm)
        let mut e_norm = e.zeros(n_embd)?;
        e.rms_norm(&e_emb, mtp.enorm.float_data(), &mut e_norm, n_embd, 1, eps)?;
        let mut h_norm = e.zeros(n_embd)?;
        e.rms_norm(h_seed, mtp.hnorm.float_data(), &mut h_norm, n_embd, 1, eps)?;

        // op 3: concat = [e_norm ; h_norm] -> [2*n_embd], e_norm in [0,n_embd), h_norm in [n_embd,2n_embd)
        let mut concat = e.zeros(2 * n_embd)?;
        e.copy_into(&mut concat, 0, &e_norm, n_embd)?;
        e.copy_into(&mut concat, n_embd, &h_norm, n_embd)?;

        // op 4: inpSA = eh_proj @ concat  (eh_proj [2*n_embd, n_embd]) -> [n_embd]
        let inp_sa = e.matmul(&mtp.eh_proj, &concat, 1)?;

        // op 5: a_norm = RMSNorm(inpSA, attn_norm)
        let mut a_norm = e.zeros(di)?;
        e.rms_norm(&inp_sa, mtp.attn_norm.float_data(), &mut a_norm, di, 1, eps)?;
        anat_mark(0, e, &mut t_ph)?;

        // op 6: attention on the scratch KV. SAME dc launcher as the graph path (bucket_max =
        // scratch.cap, length from the device len_d) so eager drafts match graph drafts
        // bit-for-bit at any t_kv (the parity gate). Host len mirrored here (the dc append
        // advances only the device counter).
        let attn_out = match (&mtp.mixer, mtp.step35.as_ref()) {
            // step35 MTP block, dcw door armed: the SAME windowed device-counter launcher as
            // the captured chain (draft parity by construction). Per-step ring headroom runs
            // HERE (eager is host-len work, a rebase is legal); host len mirrored like the
            // plain dc arm below.
            (Mixer::Full(fa), Some(g))
                if self.step35_dcw_eligible(g, scratch.plane(scratch_index).1) =>
            {
                {
                    let (kv, _) = scratch.plane_mut(scratch_index);
                    let retain = match kv.ring.as_ref() {
                        Some(ring) => memra_kv::swa_retain_from(kv.len, ring.window(), ring.base()),
                        None => 0,
                    };
                    e.prepare_kv_append(kv, retain, 1)?;
                }
                let out =
                    self.mtp_step35_attn_dcw(e, fa, g, &a_norm, &pos_d, scratch, scratch_index)?;
                scratch.plane_mut(scratch_index).0.len += 1;
                out
            }
            // step35 MTP block, door off (MEMRA_STEP35_DRAFT_DCW=0 rollback) or class-
            // ineligible: PER-LAYER geometry + a separate head-wise gate + an SWA window,
            // none of which the plain dc launcher can express (see `mtp_step35_attn`).
            // Host-len arm. Advances BOTH the
            // host len and the device counter itself (unlike the dc arm, whose host-side
            // mirror the caller does).
            (Mixer::Full(fa), Some(g)) => {
                self.mtp_step35_attn(e, fa, g, &a_norm, &pos_d, scratch, scratch_index)?
            }
            (Mixer::Full(fa), None) => {
                let out = self.mtp_full_attn_dc(
                    e,
                    fa,
                    &a_norm,
                    &pos_d,
                    scratch,
                    scratch_index,
                    mtp.geom.as_ref(),
                )?;
                scratch.plane_mut(scratch_index).0.len += 1;
                out
            }
            (Mixer::Linear(_), _) => {
                panic!("MTP block is full-attn in qwen35; linear MTP not supported")
            }
            (Mixer::Mla(_), _) => crate::hybrid::mla_path_unimplemented("MTP head forward"),
            (Mixer::Kda(_), _) => crate::hybrid::kda_path_unimplemented("MTP head forward"),
        };
        anat_mark(1, e, &mut t_ph)?;

        // op 7: x1 = inpSA + attn_out
        let mut x1 = e.zeros(di)?;
        e.add(&inp_sa, &attn_out, &mut x1, di)?;

        // op 8: z = RMSNorm(x1, post_attn_norm)  (pre-FFN norm)
        let mut z = e.zeros(di)?;
        e.rms_norm(&x1, mtp.post_attn_norm.float_data(), &mut z, di, 1, eps)?;

        // op 9: FFN (Dense or MoE) — same as the trunk decode FFN
        let ffn_out = match &mtp.ffn {
            crate::hybrid::Ffn::Dense {
                ffn_gate,
                ffn_up,
                ffn_down,
            } => {
                let n_ff = ffn_gate.out_features();
                let (gate, up) = if e.uses_q8_1_fast(ffn_gate) && e.uses_q8_1_fast(ffn_up) {
                    let (zq, zd) = e.quantize_q8_1(&z, 1, di)?;
                    (
                        e.matmul_pre(ffn_gate, &zq, &zd, &z, 1)?,
                        e.matmul_pre(ffn_up, &zq, &zd, &z, 1)?,
                    )
                } else {
                    (e.matmul(ffn_gate, &z, 1)?, e.matmul(ffn_up, &z, 1)?)
                };
                let mut act = e.zeros(n_ff)?;
                // step35: a DENSE FFN reads the per-layer SHEXP clamp (upstream's one `build_ffn`
                // serves the dense MLP and the shared expert off `swiglu_clamp_shexp` —
                // llama-graph.cpp:1751), resolved for the MTP block's OWN index. Every other arch
                // passes None, which is `ffn_act`'s dispatch verbatim.
                Self::ffn_act_lim(
                    e,
                    &self.cfg,
                    &gate,
                    &up,
                    1.0,
                    1.0,
                    mtp.step35
                        .as_ref()
                        .and_then(|s| s.clamp_shexp)
                        .map(SwigluClamp::Post),
                    &mut act,
                    n_ff,
                )?;
                e.matmul(ffn_down, &act, 1)?
            }
            // MTP head is a distinct block — key its experts under a separate layer index (u16::MAX)
            // so they never alias trunk layer 0's cache keys.
            crate::hybrid::Ffn::Moe(m) => self.moe_ffn_il(e, m, &z, 1, u16::MAX)?,
        };
        anat_mark(2, e, &mut t_ph)?;

        // op 10: h_nextn = x1 + ffn_out (at di)
        let mut h_inner = e.zeros(di)?;
        e.add(&x1, &ffn_out, &mut h_inner, di)?;

        // op 10.5 (student): up-project the inner hidden back to n_embd — training semantics:
        // the chain carrier AND the head input are out_up(h_inner) (pre-final-norm).
        let h_nextn = match mtp.geom.as_ref() {
            Some(g) => e.matmul(&g.out_up, &h_inner, 1)?,
            None => h_inner,
        };

        // op 11: final = RMSNorm(h_nextn, shared_head_norm OR output_norm)
        let final_norm = mtp.shared_head_norm.as_ref().unwrap_or(&self.output_norm);
        let mut final_h = e.zeros(n_embd)?;
        e.rms_norm(
            &h_nextn,
            final_norm.float_data(),
            &mut final_h,
            n_embd,
            1,
            eps,
        )?;

        // op 12: draft_logits = (shared_head_head OR output) @ final — stays ON DEVICE.
        let head = mtp.shared_head_head.as_ref().unwrap_or(&self.output);
        let mut logits = e.matmul(head, &final_h, 1)?;
        // op 12b (lane/draft-mask): grammar mask over the DRAFT vocab, applied here so the
        // caller's argmax / gumbel draw / p-min prob all read the grammar-legal row.
        if let Some((mask_d, mw)) = mask {
            let d_vocab = head.out_features();
            e.mask_logits_col(&mut logits, mask_d, 0, d_vocab, mw)?;
        }
        anat_mark(3, e, &mut t_ph)?;
        if anat {
            ANAT_NS[4].fetch_add(t_all.elapsed().as_nanos() as u64, Relaxed);
            let n = ANAT_STEPS.fetch_add(1, Relaxed) + 1;
            if n.is_multiple_of(128) {
                let us = |i: usize| ANAT_NS[i].load(Relaxed) / n / 1000;
                eprintln!(
                    "[spec-anatomy] steps={n} avg us/step: glue={} attn={} ffn={} head={} total={}",
                    us(0),
                    us(1),
                    us(2),
                    us(3),
                    us(4)
                );
            }
        }
        // Chain recurrence hand-over: pre-norm h_nextn (default) or post-norm final_h
        // (MEMRA_SPEC_HPOST — llama.cpp #24025's t_h_nextn is taken AFTER the head norm).
        Ok((logits, if spec_hpost() { final_h } else { h_nextn }))
    }

    /// One NextN/MTP draft step for an **MLA-mixer** MTP block (glm5_next class: MLA + own
    /// k-pool indexer + MoE, serial residual — the NextN layer carries no hc_* tensors), on
    /// the model `Cache`'s own MTP latent plane rather than the full-attn `MtpScratch` the
    /// qwen35/step35 chain uses. Gate: `glm5_mtp_head_gpu` (engine vs `memra_reference`
    /// `execute_mtp`, teacher-forced walk, eh_proj-transpose and h_seed-off-by-one red arms).
    ///
    /// The interface, stated precisely for the verify arc:
    /// - `h_seed`: `[n_embd]` f32 device — the trunk's COLLAPSED PRE-output_norm hidden of
    ///   the position whose next token is being drafted (MTP-PLAN §A; exactly what
    ///   `prime_cache`/`decode_step` return for hc models). `MEMRA_SPEC_HPOST` flips both
    ///   this producer and the returned carrier to the post-norm variant, same as the dev path.
    /// - `e_tok`: the token at the seeded position's SUCCESSOR — the token the trunk just
    ///   sampled/accepted (reference oracle pairing: `fused[i] = eh_proj([enorm(embed(ids[i]));
    ///   hnorm(trunk_hidden[i])])`, i.e. this call with `e_tok = ids[i]`, `h_seed = h[i]`,
    ///   `mtp_pos = i` reproduces the reference's row `i`).
    /// - `mtp_pos`: the absolute position this step appends to the MTP block's latent plane;
    ///   must equal that plane's current length (the plane advances by ONE row per call inside
    ///   `mla_attn_cached`; rollback on rejection = the verify arc's latent-plane len reset).
    /// - returns `(draft_logits [n_vocab], carrier [n_embd])` on device. glm5_next ships no
    ///   private MTP head, so the logits ride the trunk `lm_head` (full vocab, no d2t).
    pub fn mtp_head_forward_mla_cached(
        &self,
        e: &Engine,
        depth: usize,
        e_tok: u32,
        h_seed: &CudaSlice<f32>,
        cache: &mut Cache,
        mtp_pos: usize,
    ) -> Result<(CudaSlice<f32>, CudaSlice<f32>), Box<dyn std::error::Error>> {
        if depth >= self.mtp_head_count() {
            return Err(format!(
                "MTP depth {depth} out of range: {} embedded head(s) loaded \
                 (is MEMRA_GLM5_MTP=1 set for a glm5_next model?)",
                self.mtp_head_count()
            )
            .into());
        }
        let mtp = self.mtp_head_at(depth);
        let block = self
            .plan
            .mtp_blocks
            .get(depth)
            .ok_or_else(|| format!("ModelPlan declares no MTP block at depth {depth}"))?;
        let il = block.layer.index as usize;
        let Mixer::Mla(mla) = &mtp.mixer else {
            return Err(
                "mtp_head_forward_mla_cached serves MLA-mixer MTP blocks only; full-attn \
                 blocks take mtp_head_forward_dev's scratch path"
                    .into(),
            );
        };
        if matches!(mtp.ffn, crate::hybrid::Ffn::Dense { .. }) {
            return Err(
                "MLA-mixer MTP block with a Dense FFN has no gated arm yet (glm5_next and \
                 glm-dsa NextN blocks are MoE); refusing rather than running ungated math"
                    .into(),
            );
        }
        let plane_len = cache
            .latent
            .get(il)
            .and_then(|plane| plane.as_ref())
            .map(|plane| plane.len)
            .ok_or_else(|| {
                format!(
                    "MTP block layer {il} has no latent cache plane — the Cache must be \
                     built from a plan whose mtp_blocks declare StatePlan::LatentKvCache"
                )
            })?;
        if mtp_pos != plane_len {
            return Err(format!(
                "MTP draft position {mtp_pos} != the MTP latent plane's length {plane_len} — \
                 the plane advances one row per draft step and rolls back by len reset; a \
                 skipped or repeated position would attend the wrong horizon"
            )
            .into());
        }

        let cfg = &self.cfg;
        let n_embd = cfg.n_embd as usize;
        let eps = cfg.rms_eps;
        let pos_d = e.htod_i32(&[mtp_pos as i32])?;

        // Same op chain as `mtp_head_forward_dev_at` (ops 1-12), same kernels — only the
        // attention arm differs: `mla_attn_cached` on the plan's own MTP plane instead of
        // `mtp_full_attn_dc` on the MtpScratch.
        let e_emb = e.htod(&self.embd.try_gather(n_embd, &[e_tok])?)?;
        let mut e_norm = e.zeros(n_embd)?;
        e.rms_norm(&e_emb, mtp.enorm.float_data(), &mut e_norm, n_embd, 1, eps)?;
        let mut h_norm = e.zeros(n_embd)?;
        e.rms_norm(h_seed, mtp.hnorm.float_data(), &mut h_norm, n_embd, 1, eps)?;

        let mut concat = e.zeros(2 * n_embd)?;
        e.copy_into(&mut concat, 0, &e_norm, n_embd)?;
        e.copy_into(&mut concat, n_embd, &h_norm, n_embd)?;
        let inp_sa = e.matmul(&mtp.eh_proj, &concat, 1)?;

        let mut a_norm = e.zeros(n_embd)?;
        e.rms_norm(
            &inp_sa,
            mtp.attn_norm.float_data(),
            &mut a_norm,
            n_embd,
            1,
            eps,
        )?;
        let attn_out = self.mla_attn_cached(e, mla, &a_norm, &pos_d, 1, il, cache)?;

        let mut x1 = e.zeros(n_embd)?;
        e.add(&inp_sa, &attn_out, &mut x1, n_embd)?;
        let mut z = e.zeros(n_embd)?;
        e.rms_norm(&x1, mtp.post_attn_norm.float_data(), &mut z, n_embd, 1, eps)?;
        let ffn_out = match &mtp.ffn {
            // Distinct block — key its experts off the trunk layers' cache keys (dev-path rule).
            crate::hybrid::Ffn::Moe(m) => self.moe_ffn_il(e, m, &z, 1, u16::MAX)?,
            crate::hybrid::Ffn::Dense { .. } => unreachable!("refused above"),
        };
        let mut h_nextn = e.zeros(n_embd)?;
        e.add(&x1, &ffn_out, &mut h_nextn, n_embd)?;

        let final_norm = mtp.shared_head_norm.as_ref().unwrap_or(&self.output_norm);
        let mut final_h = e.zeros(n_embd)?;
        e.rms_norm(
            &h_nextn,
            final_norm.float_data(),
            &mut final_h,
            n_embd,
            1,
            eps,
        )?;
        let head = mtp.shared_head_head.as_ref().unwrap_or(&self.output);
        let logits = e.matmul(head, &final_h, 1)?;
        Ok((logits, if spec_hpost() { final_h } else { h_nextn }))
    }

    #[allow(clippy::too_many_arguments)]
    fn mtp_chain_forward_dev(
        &self,
        e: &Engine,
        tokens: &[u32],
        seeds: &[CudaSlice<f32>],
        scratch: &mut MtpScratch,
        committed_scratch_len: usize,
        embd_dev: Option<(&CudaSlice<u8>, i32, usize)>,
        mask: Option<(&CudaSlice<u32>, usize)>,
    ) -> Result<(CudaSlice<f32>, CudaSlice<f32>), Box<dyn std::error::Error>> {
        if tokens.is_empty() || tokens.len() != seeds.len() {
            return Err("multi-head MTP prefix tokens/seeds are malformed".into());
        }
        let index = mtp_chain_head_index(tokens.len() - 1, self.mtp_head_count());
        let head = self.mtp_head_at(index);
        scratch.set_plane_len(e, index, committed_scratch_len)?;

        let mut last = None;
        for row in 0..tokens.len() {
            let is_last = row + 1 == tokens.len();
            last = Some(self.mtp_head_forward_dev_at(
                e,
                head,
                tokens[row],
                &seeds[row],
                scratch,
                index,
                committed_scratch_len + row + 1,
                embd_dev,
                if is_last { mask } else { None },
            )?);
        }
        Ok(last.expect("non-empty MTP prefix produced no row"))
    }

    /// step35 MTP-block attention, T=1, on the scratch KV — the EAGER-ONLY twin of
    /// `mtp_full_attn_dc`. Three things force a separate arm rather than a geometry parameter on
    /// the dc path, and all three are properties of this arch's MTP block:
    ///
    /// 1. **The SWA window.** Block 45 is an SWA-type block (`sliding_window_pattern[45]=true`,
    ///    window 512). Windowed decode in memra is a token-aligned VIEW OFFSET into the quantized
    ///    cache (the gemma4 R6 / `step35_decode_attn` pattern: keys carry absolute rope and the
    ///    mask is purely positional, so one query at `len-1` attending the last `win` rows IS the
    ///    windowed result). `fa_decode_dc` takes the key count from a DEVICE counter and always
    ///    starts at row 0 — it cannot express a nonzero offset. The windowed dc arm is
    ///    `mtp_step35_attn_dcw` (`fa_decode_dcw`, doored via MEMRA_STEP35_DRAFT_DCW —
    ///    default ON since lane/step37-draft-graph-serving-20260830); this host-len arm is
    ///    the =0 rollback and the class-ineligibility fallback.
    /// 2. **Per-layer head count.** 96 q heads over 8 KV (GQA 12) at this block, vs the trunk's 64
    ///    on its full-attn layers. The trunk cfg's `n_head` scalar is the MAX over layers, and the
    ///    trunk ARTIFACT's per-layer arrays stop at index 44 — so the count must come from the
    ///    resolved `Step35MtpGeom`, never from `cfg`.
    /// 3. **The separate head-wise gate.** `blk.45.attn_gate.weight [n_embd, 96]` produces one
    ///    sigmoid scalar per head (broadcast over head_dim) — `attn_head_gate`, not the qwen35
    ///    fused-into-wq `q_gate_split` form the dc arm handles.
    ///
    /// DOOR STATE: with MEMRA_STEP35_DRAFT_DCW=0 (or a sub-eligible kernel class),
    /// `mtp_head_forward_cap` refuses step35 heads explicitly (rather than silently capturing
    /// a window-less, wrong-past-`win` graph) and this eager chain IS the served path. With
    /// the door armed (the default), BOTH draft modes run the `mtp_step35_attn_dcw` twin
    /// instead of this arm.
    ///
    /// Unlike the dc arm this advances BOTH the host `kv.len` and the device counter, so the
    /// caller must not mirror.
    #[allow(clippy::too_many_arguments)] // allow: the parameter list mirrors the kernel/FFI/call contract; bundling into a struct is a refactor, not a lint fix
    fn mtp_step35_attn(
        &self,
        e: &Engine,
        fa: &FullAttnLayer,
        g: &crate::hybrid::Step35MtpGeom,
        h: &CudaSlice<f32>,
        pos_d: &CudaSlice<i32>,
        scratch: &mut MtpScratch,
        scratch_index: usize,
    ) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        let (nh, nkv, hd) = (g.n_head, g.n_head_kv, self.cfg.head_dim_k as usize);
        // MTP-GEOM RECEIPT, once per process, on the SERVED draft path. Slot-0 acceptance is
        // 0.725 here against 0.994 for vLLM MTP3 on the same checkpoint family and card class, and
        // the first three explanations for that gap were all wrong: head assignment (step-modulo
        // is index 0 at K=1, correct), MEMRA_SPEC_HPOST (identical 84/116 both arms), and this
        // block's geometry. Geometry was the one that could have failed SILENTLY — a wrong window
        // makes the draft attend the whole context instead of Step-3.7's 512, stays fluent, and
        // shows up only as acceptance — so it gets a standing receipt rather than another reading
        // of the source. Prints the resolved Step35MtpGeom the served path actually runs on;
        // `full_attention_geometry_at`'s missing-row fallback (window: None) does NOT reach here.
        {
            static ONCE: std::sync::OnceLock<()> = std::sync::OnceLock::new();
            ONCE.get_or_init(|| {
                eprintln!(
                    "[mtp-geom] arm=eager block={} swa={} window={} n_head={nh} n_head_kv={nkv} \
                     head_dim_k={hd} n_rot={} rope_base={} clamp_shexp={:?}",
                    g.il, g.swa, g.window, g.n_rot, g.rope_base, g.clamp_shexp,
                );
            });
        }
        let eps = self.cfg.rms_eps;
        let scale = 1.0 / (hd as f32).sqrt(); // step35.cpp:255 kq_scale
        let n_embd = self.cfg.n_embd as usize;
        let gw = fa
            .attn_gate
            .as_ref()
            .ok_or("step35 MTP block is missing attn_gate.weight (head-wise attention gate)")?;

        let (q0, k0, v0, gt) = if e.uses_q8_1_fast(&fa.wq)
            && e.uses_q8_1_fast(&fa.wk)
            && e.uses_q8_1_fast(&fa.wv)
            && e.uses_q8_1_fast(gw)
        {
            let (hq, hdq) = e.quantize_q8_1(h, 1, n_embd)?;
            let (a, b, c) = match e.matmul_q8_fused3(&fa.wq, &fa.wk, &fa.wv, &hq, &hdq)? {
                Some(t3) => t3,
                None => (
                    e.matmul_pre(&fa.wq, &hq, &hdq, h, 1)?,
                    e.matmul_pre(&fa.wk, &hq, &hdq, h, 1)?,
                    e.matmul_pre(&fa.wv, &hq, &hdq, h, 1)?,
                ),
            };
            (a, b, c, e.matmul_pre(gw, &hq, &hdq, h, 1)?)
        } else {
            (
                e.matmul(&fa.wq, h, 1)?,
                e.matmul(&fa.wk, h, 1)?,
                e.matmul(&fa.wv, h, 1)?,
                e.matmul(gw, h, 1)?,
            )
        };

        let mut q = e.uninit(nh * hd)?;
        e.rms_norm(&q0, fa.q_norm.float_data(), &mut q, hd, nh, eps)?;
        let mut k = e.uninit(nkv * hd)?;
        e.rms_norm(&k0, fa.k_norm.float_data(), &mut k, hd, nkv, eps)?;
        // `rope_freqs.weight` (llama3 factors) applies to the FULL-attn layers ONLY; SWA passes
        // null (llama-hparams / step35.cpp). Block 45 is SWA, so `ff` is None there — but read
        // the resolved flag, not the constant, so an all-full sibling stays correct.
        let ff = if g.swa {
            None
        } else {
            self.step35_aux.as_ref().and_then(|a| a.rope_freqs(e))
        };
        #[cfg(debug_assertions)]
        if let Some(ff) = ff {
            crate::debug_assert_tensor_stream_device(ff, &e.stream(), "mtp_step35_attn.rope_freqs");
        }
        e.rope_neox2(
            &mut q,
            &mut k,
            pos_d,
            hd,
            g.n_rot,
            nh,
            nkv,
            1,
            g.rope_base,
            1.0,
            ff,
        )?;

        // Append at the HOST slot, then re-stamp the device counter: the eager chain has the
        // length on the host anyway, and the windowed view below needs it there to compute the
        // offset. The device counter is kept in lockstep so `mtp_kv_fill`'s `set_i32_one` and any
        // dc-family consumer of this scratch still agree.
        let (kv, scratch_cap) = scratch.plane_mut(scratch_index);
        assert!(
            kv.len < scratch_cap,
            "step35 MTP scratch overflow ({} >= {})",
            kv.len,
            scratch_cap
        );
        let next_len = kv.len + 1;
        let (off, t_kv) = if g.swa && next_len > g.window {
            (next_len - g.window, g.window)
        } else {
            (0, next_len)
        };
        // `off`/`t_kv` stay the ATTENTION view; the retain is a separate, lower bound so the
        // rewind that follows this append is still resident. THIS is the only site that rebases
        // this plane (MEMRA_KV_REBASE_TRACE, one run: 1 rebase, all from here), so it is the site
        // that decides `base` for everyone.
        let retain_from = match kv.ring.as_ref() {
            Some(ring) => memra_kv::swa_retain_from(kv.len, ring.window(), ring.base()),
            None => off & !31usize,
        };
        let write_row = e.prepare_kv_append(kv, retain_from, 1)?;
        e.append_kv_quantized(
            &k,
            &v0,
            &mut kv.k,
            &mut kv.v,
            write_row,
            kv.kv_dim_k,
            kv.kv_dim_v,
            kv.k_tok_bytes,
            kv.v_tok_bytes,
            false,
        )?;
        kv.len = next_len;
        e.set_i32_one(&mut kv.len_d, kv.len as i32)?;
        // SWA view offset (see note 1). The draft chain is short (k+2 rows), but the scratch is
        // PERSISTENT across rounds — `mtp_kv_fill` leaves one row per committed token behind, so
        // `kv.len` tracks absolute position and crosses 512 in any real generation. The window is
        // therefore live, not theoretical.
        let physical = kv.physical_rows(off, off + t_kv)?;
        let k_view = e.view_u8_range(
            &kv.k,
            physical.start * kv.k_tok_bytes,
            physical.end * kv.k_tok_bytes,
        );
        let v_view = e.view_u8_range(
            &kv.v,
            physical.start * kv.v_tok_bytes,
            physical.end * kv.v_tok_bytes,
        );
        let mut attn = e.uninit(nh * hd)?;
        e.fa_decode_kvmod(
            &q,
            &k_view,
            &v_view,
            &mut attn,
            hd,
            nh,
            nkv,
            t_kv,
            scale,
            kv.k_tok_bytes,
            kv.v_tok_bytes,
            false,
        )?;

        let mut ag = e.uninit(nh * hd)?;
        e.attn_head_gate(&attn, &gt, &mut ag, None, hd, nh, 1)?;
        e.matmul(&fa.wo, &ag, 1)
    }

    /// The dcw draft arm's kernel-class precondition, mirrored from `fa_decode_dcw`'s own
    /// refusal plus the v3 walk's format contract (`fa_v3_active`), so the DEV dispatch can
    /// never pick an arm the launcher would refuse mid-chain (the eager chain has no graceful
    /// fallback point) and the CAP site refuses with the named reason instead.
    ///
    /// `cap` = the SESSION's scratch-plane row capacity: the launcher's vec gate reads
    /// `bucket_max = min(window, cap)`, so a SMALL session (tiny prompt + tiny max_tokens,
    /// e.g. a max_tokens=8 probe: cap ~62 < the 96 vec floor) is OUTSIDE the dcw domain even
    /// though the WINDOW clears the floor. Mirroring the window alone shipped exactly that
    /// hole when the door default flipped ON (2026-08-30, vision-cell receipt: sampled
    /// capture WARN + `[engine-error] fa_decode_dcw supports the default v3-vec class only`
    /// hard-failing the burst — the eager dcw arm has no graceful fallback point). Sub-floor
    /// sessions now take the host-len kvmod arm, byte-for-byte the door-off serving.
    fn step35_dcw_eligible(&self, g: &crate::hybrid::Step35MtpGeom, cap: usize) -> bool {
        let hd = self.cfg.head_dim_k as usize;
        step35_draft_dcw_on()
            && g.swa
            && g.window.min(cap) >= crate::fa_vec_min_tkv()
            && std::env::var("MEMRA_NO_FA_VEC").is_err()
            && crate::fa_v3_active(hd)
            && hd <= 256
            && hd.is_multiple_of(32)
    }

    /// step35 MTP-block attention, T=1, on the scratch KV: the WINDOWED DEVICE-COUNTER twin
    /// of `mtp_step35_attn`, serving BOTH draft paths when `step35_draft_dcw_on`. Write slot,
    /// key bound and SWA view offset all derive from device state (`len_d`, `base_d` written
    /// only at host-side rebases, and the block's `window`), so ONE captured graph serves the
    /// whole chain and replays see KV growth through the counter: the `mtp_full_attn_dc`
    /// contract plus the view offset the plain `_dc` kernel could not express (the old
    /// capture-refusal root cause). The three step35 properties stay per-geom exactly as in
    /// the eager twin: nh/nkv from `Step35MtpGeom`, the separate head-wise gate
    /// (`attn_head_gate`), per-layer rope width/base with SWA passing null freqs.
    ///
    /// bucket_max = min(cap, window): the windowed view never exceeds `window` rows, so the
    /// capture-time grid stays valid for every replayed len, and the kernel derives ns_eff
    /// from the LIVE T_kv at the fixed split_keys (one-partition law). Both arms call THIS
    /// launcher at THIS bucket, so eager and captured drafts are bit-identical by
    /// construction; vs the retired-by-flag `mtp_step35_attn` the only numeric-class deltas
    /// are the sub-vec-floor region (t_kv < 96: kvmod ran scalar, dcw stays vec) and any
    /// live-len split-ladder rung below the bucket's, both draft-side only (the verify
    /// arbitrates emitted bytes; acceptance is gated by the battery).
    ///
    /// Host len is NOT advanced here (graph contract); callers mirror. The EAGER caller runs
    /// `prepare_kv_append` per step (ring headroom, rebase legal there); the CAPTURED path
    /// pre-arms headroom at capture time and round start (`MtpScratch::ensure_dcw_headroom`)
    /// because a rebase is host work no captured chain may contain.
    #[allow(clippy::too_many_arguments)] // allow: the parameter list mirrors the capture/call contract; bundling into a struct is a refactor, not a lint fix
    fn mtp_step35_attn_dcw(
        &self,
        e: &Engine,
        fa: &FullAttnLayer,
        g: &crate::hybrid::Step35MtpGeom,
        h: &CudaSlice<f32>,
        pos_d: &CudaSlice<i32>,
        scratch: &mut MtpScratch,
        scratch_index: usize,
    ) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        let (nh, nkv, hd) = (g.n_head, g.n_head_kv, self.cfg.head_dim_k as usize);
        // MTP-GEOM RECEIPT (dcw twin of the `mtp_step35_attn` receipt): once per process,
        // naming the arm, so a serving log proves WHICH draft attention program ran (the
        // engagement receipt for the flag door, both directions).
        {
            static ONCE: std::sync::OnceLock<()> = std::sync::OnceLock::new();
            ONCE.get_or_init(|| {
                eprintln!(
                    "[mtp-geom] arm=dcw block={} swa={} window={} n_head={nh} n_head_kv={nkv} \
                     head_dim_k={hd} n_rot={} rope_base={} clamp_shexp={:?}",
                    g.il, g.swa, g.window, g.n_rot, g.rope_base, g.clamp_shexp,
                );
            });
        }
        let eps = self.cfg.rms_eps;
        let scale = 1.0 / (hd as f32).sqrt(); // step35.cpp:255 kq_scale
        let n_embd = self.cfg.n_embd as usize;
        let gw = fa
            .attn_gate
            .as_ref()
            .ok_or("step35 MTP block is missing attn_gate.weight (head-wise attention gate)")?;

        let (q0, k0, v0, gt) = if e.uses_q8_1_fast(&fa.wq)
            && e.uses_q8_1_fast(&fa.wk)
            && e.uses_q8_1_fast(&fa.wv)
            && e.uses_q8_1_fast(gw)
        {
            let (hq, hdq) = e.quantize_q8_1(h, 1, n_embd)?;
            let (a, b, c) = match e.matmul_q8_fused3(&fa.wq, &fa.wk, &fa.wv, &hq, &hdq)? {
                Some(t3) => t3,
                None => (
                    e.matmul_pre(&fa.wq, &hq, &hdq, h, 1)?,
                    e.matmul_pre(&fa.wk, &hq, &hdq, h, 1)?,
                    e.matmul_pre(&fa.wv, &hq, &hdq, h, 1)?,
                ),
            };
            (a, b, c, e.matmul_pre(gw, &hq, &hdq, h, 1)?)
        } else {
            (
                e.matmul(&fa.wq, h, 1)?,
                e.matmul(&fa.wk, h, 1)?,
                e.matmul(&fa.wv, h, 1)?,
                e.matmul(gw, h, 1)?,
            )
        };

        let mut q = e.zeros(nh * hd)?;
        e.rms_norm(&q0, fa.q_norm.float_data(), &mut q, hd, nh, eps)?;
        let mut k = e.zeros(nkv * hd)?;
        e.rms_norm(&k0, fa.k_norm.float_data(), &mut k, hd, nkv, eps)?;
        // rope_freqs (llama3 factors) apply to the FULL-attn layers ONLY; SWA passes null
        // (the eager twin's rule, resolved from the flag, not the constant).
        let ff = if g.swa {
            None
        } else {
            self.step35_aux.as_ref().and_then(|a| a.rope_freqs(e))
        };
        #[cfg(debug_assertions)]
        if let Some(ff) = ff {
            crate::debug_assert_tensor_stream_device(
                ff,
                &e.stream(),
                "mtp_step35_attn_dcw.rope_freqs",
            );
        }
        e.rope_neox2(
            &mut q,
            &mut k,
            pos_d,
            hd,
            g.n_rot,
            nh,
            nkv,
            1,
            g.rope_base,
            1.0,
            ff,
        )?;

        let (kv, cap) = scratch.plane_mut(scratch_index);
        // Append at the DEVICE slot's PHYSICAL row (len_d - base_d), then advance the counter
        // in-graph. Physical room is the callers' headroom contract (see the fn doc).
        e.append_kv_quantized_dcw(
            &k,
            &v0,
            &mut kv.k,
            &mut kv.v,
            &kv.len_d,
            kv.base_d.as_ref(),
            kv.kv_dim_k,
            kv.kv_dim_v,
            kv.k_tok_bytes,
            kv.v_tok_bytes,
        )?;
        e.inc_seqlen(&mut kv.len_d)?;
        // Full-buffer views (any in-round physical row stays in range under the headroom
        // contract); the kernel bounds and offsets the key range from (len_d, base_d, window).
        let k_view = e.view_u8(&kv.k, kv.k.len());
        let v_view = e.view_u8(&kv.v, kv.v.len());
        let bucket = g.window.min(cap);
        let mut attn = e.zeros(nh * hd)?;
        e.fa_decode_dcw(
            &q,
            &k_view,
            &v_view,
            &mut attn,
            hd,
            nh,
            nkv,
            &kv.len_d,
            kv.base_d.as_ref(),
            if g.swa { g.window } else { 0 },
            bucket,
            scale,
            kv.k_tok_bytes,
            kv.v_tok_bytes,
            None,
        )?;

        let mut ag = e.zeros(nh * hd)?;
        e.attn_head_gate(&attn, &gt, &mut ag, None, hd, nh, 1)?;
        e.matmul(&fa.wo, &ag, 1)
    }

    /// MTP-block full attention, T=1, on the scratch KV (BOTH draft paths — eager and graph):
    /// the scratch write slot and the attention bound come from `scratch.kv.len_d` (device i32[1])
    /// so the launch args are FIXED across draft steps — ONE captured graph serves the whole
    /// chain, and replays keep seeing KV growth through the device counter (no recapture).
    /// Geometry contract: n_splits is sized from `scratch.cap` (the persistent capacity); splits
    /// whose key range lies beyond the device t_kv exit empty and the shared combine skips them
    /// (fa_decode_dc bit-correct-for-any-t_kv<=bucket_max contract). The eager path uses the SAME
    /// launcher with the SAME bucket_max -> identical dispatch -> bit-identical draft tokens (the
    /// graph-vs-eager parity gate). Host len is NOT advanced here (graph contract); callers mirror.
    #[allow(clippy::too_many_arguments)] // allow: the parameter list mirrors the kernel/FFI/call contract; bundling into a struct is a refactor, not a lint fix
    fn mtp_full_attn_dc(
        &self,
        e: &Engine,
        fa: &FullAttnLayer,
        h: &CudaSlice<f32>,
        pos_d: &CudaSlice<i32>,
        scratch: &mut MtpScratch,
        scratch_index: usize,
        geom: Option<&crate::hybrid::DraftGeom>,
    ) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        let cfg = &self.cfg;
        let mtp_il = cfg.n_layer.saturating_sub(cfg.nextn_predict_layers);
        let geometry = cfg.full_attention_geometry_at(mtp_il);
        let n_head = geom.map(|g| g.n_head).unwrap_or(geometry.n_head as usize);
        let n_head_kv = geom
            .map(|g| g.n_head_kv)
            .unwrap_or(geometry.n_head_kv as usize);
        let head_dim = geometry.head_dim_k as usize;
        let eps = cfg.rms_eps;
        let scale = geometry.attention_scale();
        let n_embd = geom.map(|g| g.d_inner).unwrap_or(cfg.n_embd as usize);
        let bucket_max = scratch.plane(scratch_index).1;

        let (qf, mut k, v) =
            if e.uses_q8_1_fast(&fa.wq) && e.uses_q8_1_fast(&fa.wk) && e.uses_q8_1_fast(&fa.wv) {
                let (hq, hd) = e.quantize_q8_1(h, 1, n_embd)?;
                (
                    e.matmul_pre(&fa.wq, &hq, &hd, h, 1)?,
                    e.matmul_pre(&fa.wk, &hq, &hd, h, 1)?,
                    e.matmul_pre(&fa.wv, &hq, &hd, h, 1)?,
                )
            } else {
                (
                    e.matmul(&fa.wq, h, 1)?,
                    e.matmul(&fa.wk, h, 1)?,
                    e.matmul(&fa.wv, h, 1)?,
                )
            };
        // M3/Hy3 have no attention output gate — wq out is exactly q; skip the split.
        let gated = geometry.attention_gate == memra_gguf::config::AttentionGateKind::FusedQ;
        let (mut q, gate) = if gated {
            let mut q = e.zeros(n_head * head_dim)?;
            let mut gate = e.zeros(n_head * head_dim)?;
            e.q_gate_split(&qf, &mut q, &mut gate, head_dim, n_head, 1)?;
            (q, Some(gate))
        } else {
            (qf, None)
        };

        let mut qn = e.zeros(n_head * head_dim)?;
        e.rms_norm(&q, fa.q_norm.float_data(), &mut qn, head_dim, n_head, eps)?;
        q = qn;
        let mut kn = e.zeros(n_head_kv * head_dim)?;
        e.rms_norm(
            &k,
            fa.k_norm.float_data(),
            &mut kn,
            head_dim,
            n_head_kv,
            eps,
        )?;
        k = kn;
        let rope_dims = geometry.n_rot as usize;
        e.rope_neox(
            &mut q,
            pos_d,
            head_dim,
            rope_dims,
            n_head,
            1,
            geometry.rope_base,
            1.0,
        )?;
        e.rope_neox(
            &mut k,
            pos_d,
            head_dim,
            rope_dims,
            n_head_kv,
            1,
            geometry.rope_base,
            1.0,
        )?;

        let kv = scratch.plane_mut(scratch_index).0;
        // append at the DEVICE slot (kv.len_d == old len), then advance the counter in-graph.
        e.append_kv_quantized_dc(
            &k,
            &v,
            &mut kv.k,
            &mut kv.v,
            &kv.len_d,
            kv.kv_dim_k,
            kv.kv_dim_v,
            kv.k_tok_bytes,
            kv.v_tok_bytes,
            false,
        )?;
        e.inc_seqlen(&mut kv.len_d)?;
        // full-buffer views (any in-round t_kv stays in range on replay); the kernel bounds the
        // key range from the device counter.
        let k_view = e.view_u8(&kv.k, kv.k.len());
        let v_view = e.view_u8(&kv.v, kv.v.len());
        let (ktb, vtb) = (kv.k_tok_bytes, kv.v_tok_bytes);
        let mut attn = e.zeros(n_head * head_dim)?;
        e.fa_decode_dc(
            &q, &k_view, &v_view, &mut attn, head_dim, n_head, n_head_kv, &kv.len_d, bucket_max,
            scale, ktb, vtb, false,
        )?;

        let attn_g = match &gate {
            Some(gate) => {
                let mut gsig = e.zeros(n_head * head_dim)?;
                e.sigmoid(gate, &mut gsig, n_head * head_dim)?;
                let mut ag = e.zeros(n_head * head_dim)?;
                e.mul(&attn, &gsig, &mut ag, n_head * head_dim)?;
                ag
            }
            None => attn,
        };
        e.matmul(&fa.wo, &attn_g, 1)
    }

    /// PERSISTENT-DRAFT-KV fill (the reference engine's "mtp_update" analogue): compute the MTP
    /// block's K/V for `tokens` (committed tokens at positions pos0..pos0+T) from their EXACT
    /// trunk hiddens `h` ([T, n_embd] token-major, pre-output_norm) and append at slots pos0.. of
    /// the scratch KV. K/V-ONLY — ops A/1-5 plus the K-side of op 6 (wk/wv + k_norm + rope +
    /// quantized append); no wq/attention/FFN/lm_head, so per-token cost ~= eh_proj + wk/wv (a
    /// small fraction of one trunk layer), T-batched. Rope follows the chain convention
    /// rope(token@p) = p+1. Runs at round boundaries OUTSIDE the captured graph in BOTH draft
    /// modes -> draft parity by construction. Caller must have scratch.kv.len == pos0.
    #[allow(clippy::too_many_arguments)]
    fn mtp_kv_fill_at(
        &self,
        e: &Engine,
        mtp: &MtpHead,
        tokens: &[u32],
        h: &CudaSlice<f32>,
        pos0: usize,
        scratch: &mut MtpScratch,
        scratch_index: usize,
        embd_dev: Option<(&CudaSlice<u8>, i32, usize)>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let cfg = &self.cfg;
        let n_embd = cfg.n_embd as usize;
        let eps = cfg.rms_eps;
        let t = tokens.len();
        let (scratch_kv, scratch_cap) = scratch.plane(scratch_index);
        assert_eq!(scratch_kv.len, pos0, "mtp_kv_fill: append slot mismatch");
        assert!(pos0 + t <= scratch_cap, "mtp_kv_fill: scratch overflow");
        let Mixer::Full(fa) = &mtp.mixer else {
            panic!("MTP block is full-attn in qwen35; linear MTP not supported")
        };
        let pos_vec: Vec<i32> = (0..t).map(|i| (pos0 + i + 1) as i32).collect();
        let pos_d = e.htod_i32(&pos_vec)?;

        // ops A/1/2: embed + the two input norms, T-wide.
        let e_emb = match embd_dev {
            Some((g, qt, rb)) => e.embed_gather_device_t(g, tokens, n_embd, qt, rb)?,
            None => e.htod(&self.embd.try_gather(n_embd, tokens)?)?,
        };
        let mut e_norm = e.zeros(t * n_embd)?;
        e.rms_norm(&e_emb, mtp.enorm.float_data(), &mut e_norm, n_embd, t, eps)?;
        let mut h_norm = e.zeros(t * n_embd)?;
        e.rms_norm(h, mtp.hnorm.float_data(), &mut h_norm, n_embd, t, eps)?;

        // op 3: per-row [e_norm ; h_norm] concat, token-major [T, 2*n_embd].
        let mut concat = e.zeros(t * 2 * n_embd)?;
        for i in 0..t {
            e.copy_view_into(
                &mut concat,
                i * 2 * n_embd,
                &e_norm.slice(i * n_embd..(i + 1) * n_embd),
                n_embd,
            )?;
            e.copy_view_into(
                &mut concat,
                i * 2 * n_embd + n_embd,
                &h_norm.slice(i * n_embd..(i + 1) * n_embd),
                n_embd,
            )?;
        }

        // ops 4/5: eh_proj + attn_norm, T-wide (at the student inner width when geom is set).
        let di = mtp.geom.as_ref().map(|g| g.d_inner).unwrap_or(n_embd);
        let inp_sa = e.matmul(&mtp.eh_proj, &concat, t)?;
        let mut a_norm = e.zeros(t * di)?;
        e.rms_norm(&inp_sa, mtp.attn_norm.float_data(), &mut a_norm, di, t, eps)?;

        // op 6 (K/V half): wk/wv + k_norm + rope + per-row quantized append. No wq/attention —
        // the fill only has to leave correct K/V rows behind for later chains to attend over.
        let n_head_kv = mtp
            .geom
            .as_ref()
            .map(|g| g.n_head_kv)
            .or(mtp.step35.as_ref().map(|s| s.n_head_kv))
            .unwrap_or_else(|| {
                let mtp_il = cfg.n_layer.saturating_sub(cfg.nextn_predict_layers);
                cfg.full_attention_geometry_at(mtp_il).n_head_kv as usize
            });
        let mtp_il = cfg.n_layer.saturating_sub(cfg.nextn_predict_layers);
        let geometry = cfg.full_attention_geometry_at(mtp_il);
        let head_dim = geometry.head_dim_k as usize;
        let mut k = e.matmul(&fa.wk, &a_norm, t)?;
        let v = e.matmul(&fa.wv, &a_norm, t)?;
        let mut kn = e.zeros(t * n_head_kv * head_dim)?;
        e.rms_norm(
            &k,
            fa.k_norm.float_data(),
            &mut kn,
            head_dim,
            n_head_kv * t,
            eps,
        )?;
        k = kn;
        // step35: rotary width AND base are per-layer, and the MTP block's values come from the
        // resolved `Step35MtpGeom` — NOT from `cfg.rope_dim_count`/`cfg.rope_freq_base`, which
        // carry the arch defaults (128 / 5e6, i.e. the FULL-attn layers' base). Getting this wrong
        // writes K rows the attention arm then re-derives at a different theta: correct-looking
        // output with dead acceptance, invisible to the exactness gates.
        let (rope_dims, rope_base, ff) = match mtp.step35.as_ref() {
            Some(s) => (
                s.n_rot,
                s.rope_base,
                if s.swa {
                    None
                } else {
                    self.step35_aux.as_ref().and_then(|a| a.rope_freqs(e))
                },
            ),
            None => (geometry.n_rot as usize, geometry.rope_base, None),
        };
        #[cfg(debug_assertions)]
        if let Some(ff) = ff {
            crate::debug_assert_tensor_stream_device(ff, &e.stream(), "mtp_kv_fill.rope_freqs");
        }
        match ff {
            Some(f) => e.rope_neox_ff(
                &mut k, &pos_d, head_dim, rope_dims, n_head_kv, t, rope_base, 1.0, f,
            )?,
            None => e.rope_neox(
                &mut k, &pos_d, head_dim, rope_dims, n_head_kv, t, rope_base, 1.0,
            )?,
        }

        let kv = scratch.plane_mut(scratch_index).0;
        // Match the trunk prime contract: a chunk may need the aligned window immediately before
        // its first row, so preserve that prefix when the physical tail rebases at wrap.
        let retain_from = kv
            .ring
            .as_ref()
            .map(|ring| memra_kv::swa_retain_from(pos0, ring.window(), ring.base()))
            .unwrap_or(0);
        let write_row = e.prepare_kv_append(kv, retain_from, t)?;
        for i in 0..t {
            let k_row = k.slice(i * kv.kv_dim_k..(i + 1) * kv.kv_dim_k);
            let v_row = v.slice(i * kv.kv_dim_v..(i + 1) * kv.kv_dim_v);
            e.append_kv_quantized_view(
                &k_row,
                &v_row,
                &mut kv.k,
                &mut kv.v,
                write_row + i,
                kv.kv_dim_k,
                kv.kv_dim_v,
                kv.k_tok_bytes,
                kv.v_tok_bytes,
                false,
            )?;
        }
        kv.len = pos0 + t;
        e.set_i32_one(&mut kv.len_d, kv.len as i32)?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn mtp_kv_fill_all(
        &self,
        e: &Engine,
        tokens: &[u32],
        h: &CudaSlice<f32>,
        pos0: usize,
        scratch: &mut MtpScratch,
        embd_dev: Option<(&CudaSlice<u8>, i32, usize)>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        debug_assert_eq!(self.mtp_head_count(), scratch.plane_count());
        for index in 0..self.mtp_head_count() {
            self.mtp_kv_fill_at(
                e,
                self.mtp_head_at(index),
                tokens,
                h,
                pos0,
                scratch,
                index,
                embd_dev,
            )?;
        }
        Ok(())
    }

    /// CAPTURE body for the GRAPH DRAFT (stage 2 of graph-grade spec): ONE MTP head forward with
    /// every varying input device-resident —
    ///   - token id from the persistent `tok_d` (the previous replay's in-graph argmax wrote it,
    ///     so the chain feeds itself; the host reads the same 4 bytes for the draft list),
    ///   - h_seed from the persistent `h_seed_d` (h_nextn is copied BACK into it at the end),
    ///   - rope pos from the persistent `pos_d` counter (inc'd in-graph),
    ///   - scratch KV slot/bound from `scratch.kv.len_d` (see mtp_full_attn_dc).
    ///     The p-min confidence lands in the persistent `p_d` iff `with_prob` (env is fixed per run).
    ///     Same kernels, same dispatch as the eager mtp_head_forward_dev chain -> same draft tokens
    ///     (exactness never depends on drafts — the verify arbitrates — but acceptance parity does).
    ///     `with_head=false` captures the HEAD-LESS twin for the pseudo-seed replay (2026-07-03):
    ///     the pseudo pass only needs h_nextn (op 10) + the scratch append — the lm_head read
    ///     (~1.06ms q6_K on the 9B), argmax and prob are dead weight there. h_nextn's inputs are
    ///     untouched, so the seed value is identical; round-start resets overwrite tok_d/p_d anyway.
    ///     `sampled_cap` = Some((ctr_d, perturb_d, q_out_d, seed, temp)) captures the SAMPLED twin
    ///     (step 3 of the sampled-spec arc): head logits are retained in the persistent `q_out_d`
    ///     (host D2Ds them to the round's q slot after each replay), the DEVICE event counter is
    ///     bumped in-graph, and the argmax reads GUMBEL-PERTURBED logits — one categorical draw per
    ///     replay, bit-identical to the eager arm's gumbel_perturb at the same (seed, sctr, temp).
    ///     seed/temp are capture-time constants (fixed per generate call, like p_min).
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::type_complexity)] // allow: one-shot composite type; naming it would hide the shape that matters at the call site
    fn mtp_head_forward_cap(
        &self,
        e: &Engine,
        mtp: &MtpHead,
        tok_d: &mut CudaSlice<u32>,
        pos_d: &mut CudaSlice<i32>,
        h_seed_d: &mut CudaSlice<f32>,
        p_d: &mut CudaSlice<f32>,
        scratch: &mut MtpScratch,
        // Which scratch plane this head appends to / attends over: 0 for the single-head
        // chain (every pre-lane caller), the head's own plane index for the multi-head
        // chain graphs (each head owns one plane — `mtp_chain_forward_dev`'s contract).
        scratch_index: usize,
        with_prob: bool,
        with_head: bool,
        embd_gpu: &CudaSlice<u8>,
        embd_qt: i32,
        embd_rb: usize,
        d_vocab: usize,
        sampled_cap: Option<SampledCapArgs<'_>>,
        stream_pack: Option<(&mut CudaSlice<u32>, usize, Option<&CudaSlice<u32>>)>,
        // DRAFT-SIDE GRAMMAR MASK (lane/draft-mask): (packed draft-vocab allowed-set buffer,
        // word count). Captured as ONE mask_logits_f32 node between the head matmul and the
        // in-graph argmax; the buffer address is baked, its CONTENTS are re-uploaded by the
        // host before every replay (the decode.rs graph-mask pattern). All-ones contents = a
        // no-op ban, so a position the grammar cannot constrain costs one pass over the row.
        mask_cap: Option<(&CudaSlice<u32>, usize)>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let cfg = &self.cfg;
        let n_embd = cfg.n_embd as usize;
        // step35: capturable through the WINDOWED device-counter arm (`mtp_step35_attn_dcw`)
        // once the dcw door is armed and the v3-vec class is live. Without the door this stays
        // the deliberate, named refusal: the plain `_dc` attention's key bound always starts at
        // row 0, cannot express this block's SWA view offset, and a captured chain would
        // silently attend OUTSIDE the window once the persistent scratch passes 512 rows.
        // Returning Err (not a panic) is what the capture sites already handle by degrading to
        // the eager chain (`mtp_head_forward_dev` -> `mtp_step35_attn`).
        // ROUND-STREAM stays refused EITHER WAY: the stream VERIFY has no step35 twin (see the
        // step35_verify refusal), so a stream capture that succeeded here would only move the
        // failure from capture time (graceful stream-off) to serve time (a failed round).
        if let Some(g) = mtp.step35.as_ref() {
            if stream_pack.is_some() {
                return Err(
                    "step35 has no ROUND-STREAM draft arm (the stream verify has no step35 \
                     twin); stream off"
                        .into(),
                );
            }
            if !self.step35_dcw_eligible(g, scratch.plane(scratch_index).1) {
                return Err(format!(
                    "step35 has no captured draft chain (fa_decode_dc cannot express the MTP \
                        block's SWA view offset; the windowed dcw capture needs \
                        MEMRA_STEP35_DRAFT_DCW armed [default ON, =0 disarms] and the v3-vec \
                        class live at bucket=min(window {}, scratch cap {})) - the eager draft \
                        chain serves this shape",
                    g.window,
                    scratch.plane(scratch_index).1,
                )
                .into());
            }
        }
        // student inner width (see mtp_head_forward_dev) — interface dims stay n_embd.
        let di = mtp.geom.as_ref().map(|g| g.d_inner).unwrap_or(n_embd);
        let eps = cfg.rms_eps;
        let e_emb = e.embed_gather_device(embd_gpu, tok_d, n_embd, embd_qt, embd_rb)?;
        let mut e_norm = e.zeros(n_embd)?;
        e.rms_norm(&e_emb, mtp.enorm.float_data(), &mut e_norm, n_embd, 1, eps)?;
        let mut h_norm = e.zeros(n_embd)?;
        e.rms_norm(
            &*h_seed_d,
            mtp.hnorm.float_data(),
            &mut h_norm,
            n_embd,
            1,
            eps,
        )?;
        let mut concat = e.zeros(2 * n_embd)?;
        e.copy_into(&mut concat, 0, &e_norm, n_embd)?;
        e.copy_into(&mut concat, n_embd, &h_norm, n_embd)?;
        let inp_sa = e.matmul(&mtp.eh_proj, &concat, 1)?;
        let mut a_norm = e.zeros(di)?;
        e.rms_norm(&inp_sa, mtp.attn_norm.float_data(), &mut a_norm, di, 1, eps)?;
        let attn_out = match (&mtp.mixer, mtp.step35.as_ref()) {
            // step35 (eligibility already enforced by the refusal above): the windowed dcw
            // arm, the SAME launcher the eager dev arm runs when the door is armed. No host
            // work here (this is the capture body); headroom is the callers' pre-arm.
            (Mixer::Full(fa), Some(g)) => {
                self.mtp_step35_attn_dcw(e, fa, g, &a_norm, pos_d, scratch, scratch_index)?
            }
            (Mixer::Full(fa), None) => self.mtp_full_attn_dc(
                e,
                fa,
                &a_norm,
                pos_d,
                scratch,
                scratch_index,
                mtp.geom.as_ref(),
            )?,
            (Mixer::Linear(_), _) => {
                panic!("MTP block is full-attn in qwen35; linear MTP not supported")
            }
            (Mixer::Mla(_), _) => {
                crate::hybrid::mla_path_unimplemented("captured MTP head forward")
            }
            (Mixer::Kda(_), _) => {
                crate::hybrid::kda_path_unimplemented("captured MTP head forward")
            }
        };
        let mut x1 = e.zeros(di)?;
        e.add(&inp_sa, &attn_out, &mut x1, di)?;
        let mut z = e.zeros(di)?;
        e.rms_norm(&x1, mtp.post_attn_norm.float_data(), &mut z, di, 1, eps)?;
        let ffn_out = match &mtp.ffn {
            crate::hybrid::Ffn::Dense {
                ffn_gate,
                ffn_up,
                ffn_down,
            } => {
                let n_ff = ffn_gate.out_features();
                let (gate, up) = if e.uses_q8_1_fast(ffn_gate) && e.uses_q8_1_fast(ffn_up) {
                    let (zq, zd) = e.quantize_q8_1(&z, 1, di)?;
                    (
                        e.matmul_pre(ffn_gate, &zq, &zd, &z, 1)?,
                        e.matmul_pre(ffn_up, &zq, &zd, &z, 1)?,
                    )
                } else {
                    (e.matmul(ffn_gate, &z, 1)?, e.matmul(ffn_up, &z, 1)?)
                };
                let mut act = e.zeros(n_ff)?;
                // step35: the dense FFN reads the per-layer SHEXP clamp, resolved for the MTP
                // block's own index (the mtp_head_forward_dev rule; None for every other arch,
                // which is `ffn_act`'s dispatch verbatim). The eager and captured chains must
                // run the ONE activation program.
                Self::ffn_act_lim(
                    e,
                    &self.cfg,
                    &gate,
                    &up,
                    1.0,
                    1.0,
                    mtp.step35
                        .as_ref()
                        .and_then(|s| s.clamp_shexp)
                        .map(SwigluClamp::Post),
                    &mut act,
                    n_ff,
                )?;
                e.matmul(ffn_down, &act, 1)?
            }
            // ROUND-STREAM: a softmax-routed resident MoE takes the zero-D2H device router +
            // expert program and is capture-legal. Sigmoid-routed MoE (Hy3/M3/Step) still
            // selects through the host-visible sigmoid router; capturing that stream sync
            // invalidates CUDA capture, so it stays on the eager draft chain even when every
            // expert is resident. Non-resident (SLRU-lock) is likewise rejected.
            crate::hybrid::Ffn::Moe(m)
                if m.dev_exps.is_some() && self.cfg.sigmoid_router().is_none() =>
            {
                self.moe_ffn_il(e, m, &z, 1, u16::MAX)?
            }
            crate::hybrid::Ffn::Moe(_) => {
                return Err(
                    "graph draft requires a Dense or device-routed resident-MoE MTP FFN".into(),
                );
            }
        };
        let mut h_inner = e.zeros(di)?;
        e.add(&x1, &ffn_out, &mut h_inner, di)?;
        // student: up-project back to n_embd (carrier + head input; see mtp_head_forward_dev).
        let h_nextn = match mtp.geom.as_ref() {
            Some(g) => e.matmul(&g.out_up, &h_inner, 1)?,
            None => h_inner,
        };
        // MEMRA_SPEC_HPOST needs final_h even head-less (it IS the next seed under that convention).
        let final_h = if with_head || spec_hpost() {
            let final_norm = mtp.shared_head_norm.as_ref().unwrap_or(&self.output_norm);
            let mut fh = e.zeros(n_embd)?;
            e.rms_norm(&h_nextn, final_norm.float_data(), &mut fh, n_embd, 1, eps)?;
            Some(fh)
        } else {
            None
        };
        if with_head {
            let head = mtp.shared_head_head.as_ref().unwrap_or(&self.output);
            let mut logits = e.matmul(head, final_h.as_ref().unwrap(), 1)?;
            // DRAFT-SIDE GRAMMAR MASK: ban the grammar-illegal draft ids IN the captured chain,
            // before the argmax — proposals become legal by construction. Contents-only
            // per-replay upload keeps the capture valid.
            if let Some((mask_d, mw)) = mask_cap {
                e.mask_logits_col(&mut logits, mask_d, 0, d_vocab, mw)?;
            }
            if let Some(SampledCapArgs {
                ctr: ctr_d,
                perturb: perturb_d,
                q_out: q_out_d,
                seed,
                temp,
                filt,
            }) = sampled_cap
            {
                // SAMPLED chain: retain q (raw head logits -> persistent q_out_d; the matmul's
                // own buffer is pool-recycled after the capture body returns, so it can't be the
                // retention target), bump the device event counter, gumbel-perturb reading it,
                // and argmax the PERTURBED logits into tok_d — the in-graph categorical draw.
                e.copy_into(q_out_d, 0, &logits, d_vocab)?;
                e.sctr_inc(ctr_d)?;
                match filt {
                    // PURE-TEMP: gumbel over the raw softmax — byte-identical to the
                    // pre-lane capture body.
                    None => e.gumbel_perturb_ctr(&logits, perturb_d, d_vocab, seed, ctr_d, temp)?,
                    // FILTERED (lane/step37-draft-graph-serving-20260830): the SAME
                    // filter_stats program the eager arm and the accept path run (the
                    // wrapper's coop/plain choice is deployment-keyed, never per-call), then
                    // the device-stat/device-counter perturb twin — the draft draws from the
                    // exact filtered distribution the verify gathers `q` from. q was
                    // retained ABOVE, pre-perturb, so the accept path's post-replay stats
                    // recompute (same kernel, same bits) reconstructs these th/z exactly.
                    Some(f) => {
                        e.filter_stats(
                            &logits, d_vocab, f.rows0, f.th, f.z, f.mx, d_vocab, 1, temp, f.top_k,
                            f.top_p, f.min_p,
                        )?;
                        e.gumbel_perturb_filtered_ctr(
                            &logits, perturb_d, d_vocab, seed, ctr_d, temp, f.mx, f.th,
                        )?;
                    }
                }
                e.argmax_token_device_into(perturb_d, tok_d, d_vocab)?;
                // p-min prob = the head's RAW softmax confidence in the SAMPLED pick — same
                // semantics as the eager sampled arm's prob_of_token_device(dl_d, tok_d).
                if with_prob {
                    e.prob_of_token_device_into(&logits, tok_d, p_d, d_vocab)?;
                }
            } else {
                // draft token -> persistent tok_d (next replay's embed reads it; host reads the 4 bytes).
                e.argmax_token_device_into(&logits, tok_d, d_vocab)?;
                // p-min under a draft mask reads the MASKED row: confidence relative to the
                // grammar-LEGAL alternatives (illegal ids leave the softmax denominator), which
                // is the right semantics for "does the drafter know what comes next here" and
                // the same row the pick came from. Draft-quality only — verify arbitrates.
                if with_prob {
                    e.prob_of_token_device_into(&logits, tok_d, p_d, d_vocab)?;
                }
            }
        }
        // ROUND-STREAM K-chain: pack (tok, p) into slot j, then remap tok through d2t so the
        // NEXT chained body's embed reads the TARGET id — zero host involvement per step.
        if let Some((out, slot, d2t)) = stream_pack {
            e.pack_tok_p(tok_d, p_d, out, slot)?;
            if let Some(map) = d2t {
                e.tok_map_u32(tok_d, map)?;
            }
        }
        // Next draft step's h_seed: pre-norm h_nextn (default) or post-norm final_h (HPOST).
        if spec_hpost() {
            e.copy_into(h_seed_d, 0, final_h.as_ref().unwrap(), n_embd)?;
        } else {
            e.copy_into(h_seed_d, 0, &h_nextn, n_embd)?;
        }
        // advance the draft rope position in-graph.
        e.inc_seqlen(pos_d)?;
        Ok(())
    }

    /// Batched target verify forward over `tokens` at positions `pos0..pos0+T` (§D.3, T=K+1).
    /// Returns ALL T logit columns (host f32, [T*n_vocab]); appends T cols to every full-attn KV
    /// and advances every linear-attn recur state by T steps (the recur steps are SEQUENTIAL T=1).
    /// Advances `cache.pos` by T.
    pub fn decode_step_t(
        &self,
        e: &Engine,
        tokens: &[u32],
        pos0: usize,
        cache: &mut Cache,
    ) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        if self.is_gemma4_e4b() {
            return Ok(self.gemma4_e4b_decode_step_t_h(e, tokens, pos0, cache)?.0);
        }
        if self.gemma_batch_program() {
            return self.gemma4_decode_step_t(e, tokens, pos0, cache);
        }
        Ok(self.decode_step_t_h(e, tokens, pos0, cache)?.0)
    }

    /// Like `decode_step_t` but ALSO returns the LAST column's pre-output_norm hidden (h_seed for
    /// the next draft round). This lets partial-accept replay run as ONE batched T=(n_acc+1) forward
    /// (single weight read) instead of n_acc+1 separate T=1 decode_steps (n_acc+1 weight reads).
    /// At batch=1 decode is bandwidth-bound, so batching the replay is THE MTP profitability lever.
    pub fn decode_step_t_h(
        &self,
        e: &Engine,
        tokens: &[u32],
        pos0: usize,
        cache: &mut Cache,
    ) -> Result<(Vec<f32>, CudaSlice<f32>), Box<dyn std::error::Error>> {
        self.decode_step_t_h_emb(e, tokens, pos0, cache, None)
    }

    /// Like `decode_step_t_h` with an optional RESIDENT embed table (spec hot loop): device
    /// gather instead of host dequant + [T, n_embd] f32 htod. Bit-identical rows.
    pub fn decode_step_t_h_emb(
        &self,
        e: &Engine,
        tokens: &[u32],
        pos0: usize,
        cache: &mut Cache,
        embd_dev: Option<(&CudaSlice<u8>, i32, usize)>,
    ) -> Result<(Vec<f32>, CudaSlice<f32>), Box<dyn std::error::Error>> {
        let (logits_d, h_seed) = self.decode_step_t_h_emb_dev(e, tokens, pos0, cache, embd_dev)?;
        Ok((e.dtoh(&logits_d)?, h_seed))
    }

    /// DEVICE-LOGITS verify forward (spec device-argmax lever): identical kernel chain to
    /// `decode_step_t_h_emb` but returns the [T, n_vocab] logits ON DEVICE — the accept walk
    /// argmaxes each column on-device and reads back ONE [T] u32 instead of dtoh'ing the full
    /// T x n_vocab f32 block (~1-4 MB + T host argmaxes, every round). Kernel dispatch is
    /// UNCHANGED (same decode-exact kernels); only the post-logits transfer moves.
    pub fn decode_step_t_h_emb_dev(
        &self,
        e: &Engine,
        tokens: &[u32],
        pos0: usize,
        cache: &mut Cache,
        embd_dev: Option<(&CudaSlice<u8>, i32, usize)>,
    ) -> Result<(CudaSlice<f32>, CudaSlice<f32>), Box<dyn std::error::Error>> {
        cache.ensure_usable("decode_step_t")?;
        let n_embd = self.cfg.n_embd as usize;
        let t = tokens.len();
        let (logits, x) = self.decode_step_t_core(e, tokens, pos0, cache, embd_dev, None)?;
        // h_seed for the next round = LAST column's pre-output_norm hidden ([n_embd]).
        let mut hs = vbuf(e, n_embd)?; // fully written by copy_view_into below
        e.copy_view_into(&mut hs, 0, &x.slice((t - 1) * n_embd..t * n_embd), n_embd)?;
        Ok((logits, hs))
    }

    /// CORE verify forward: the `decode_step_t_h_emb_dev` kernel chain, returning the FULL
    /// pre-output_norm hidden stack x ([T, n_embd], any column extractable) and optionally
    /// filling a `VerifyCkpt` (retained per-layer state-rebuild inputs) for the REPLAY-FREE
    /// partial accept. `ckpt: None` => byte-for-byte the old behavior (the ckpt writes are pure
    /// retains/copies — they never change what any kernel computes).
    fn decode_step_t_core(
        &self,
        e: &Engine,
        tokens: &[u32],
        pos0: usize,
        cache: &mut Cache,
        embd_dev: Option<(&CudaSlice<u8>, i32, usize)>,
        mut ckpt: Option<&mut VerifyCkpt>,
    ) -> Result<(CudaSlice<f32>, CudaSlice<f32>), Box<dyn std::error::Error>> {
        self.decode_step_t_core_stream(
            e,
            tokens,
            pos0,
            cache,
            embd_dev,
            ckpt.take(),
            None,
            None,
            None,
            None,
        )
    }

    /// [`Self::decode_step_t_core`] with the MTP route's verify-graph pool armed
    /// (`MEMRA_SPEC_VERIFY_GRAPH`). `graphs: None` reproduces `decode_step_t_core`
    /// argument-for-argument, so the eager walk stays the byte-identical fallback.
    #[allow(clippy::too_many_arguments)] // allow: the parameter list mirrors the kernel/FFI/call contract; bundling into a struct is a refactor, not a lint fix
    fn decode_step_t_core_vg(
        &self,
        e: &Engine,
        tokens: &[u32],
        pos0: usize,
        cache: &mut Cache,
        embd_dev: Option<(&CudaSlice<u8>, i32, usize)>,
        mut ckpt: Option<&mut VerifyCkpt>,
        graphs: Option<&mut DsparkVerifyGraphs>,
    ) -> Result<(CudaSlice<f32>, CudaSlice<f32>), Box<dyn std::error::Error>> {
        self.decode_step_t_core_stream(
            e,
            tokens,
            pos0,
            cache,
            embd_dev,
            ckpt.take(),
            None,
            None,
            None,
            graphs,
        )
    }

    /// Increment-0 two-session PP seam: release the peer after this lane's stage-0 boundary TX.
    /// The two independent sessions keep their own cache/checkpoint state; only issue order moves.
    #[allow(clippy::too_many_arguments)] // allow: the parameter list mirrors the kernel/FFI/call contract; bundling into a struct is a refactor, not a lint fix
    fn decode_step_t_core_pipelined(
        &self,
        e: &Engine,
        tokens: &[u32],
        pos0: usize,
        cache: &mut Cache,
        embd_dev: Option<(&CudaSlice<u8>, i32, usize)>,
        mut ckpt: Option<&mut VerifyCkpt>,
        pipe: &SpecPipeLane,
        round: usize,
    ) -> Result<(CudaSlice<f32>, CudaSlice<f32>), Box<dyn std::error::Error>> {
        let fence = crate::pp::pp_cuts(self.layers.len())
            .ok_or("two-session speculative pipeline requires a PP stage cut")?;
        if crate::pp::pp2_streams_off() || !crate::pp::spec_pp_on() {
            return Err("two-session speculative pipeline requires the PP verify split".into());
        }
        let interval_fence = pipe.stage0_begin(round)?;
        let _walk = pipe.coordinated_walk()?;
        let ticket = self.verify_stage0_issue(
            e,
            tokens,
            pos0,
            cache,
            embd_dev,
            ckpt.as_deref_mut(),
            None,
            &fence,
            Some(interval_fence),
            pipe.trace(round),
        )?;
        pipe.stage0_end(round);
        pipe.stage1_begin(round)?;
        let result = self.verify_stage1_finish(e, ticket, cache, ckpt, None, &fence, true)?;
        pipe.verify_end(round);
        Ok(result)
    }

    /// ROUND-STREAM stage (c) 4: `stream` = (device verify tokens [t], device pos counter) —
    /// when Some, rope positions come from pos_iota over the counter, the embed gathers the
    /// device tokens, and full_attn_verify routes appends/FA through the _dc twins reading the
    /// SAME counter (every layer's kvl.len == cache.pos, one counter drives all three). The
    /// host `tokens`/`pos0` args still size buffers (t is FIXED K+1 in stream mode).
    /// `vtok_dev` (engine-bundle slice 2): device verify tokens for the EMBED only —
    /// unlike `stream` mode it changes nothing else (host pos iota, host-len KV appends).
    /// `tokens` then only sizes buffers (the dummy-slice pattern the round-stream arm uses).
    #[allow(clippy::too_many_arguments)]
    fn decode_step_t_core_stream(
        &self,
        e: &Engine,
        tokens: &[u32],
        pos0: usize,
        cache: &mut Cache,
        embd_dev: Option<(&CudaSlice<u8>, i32, usize)>,
        mut ckpt: Option<&mut VerifyCkpt>,
        stream: Option<(&CudaSlice<u32>, &CudaSlice<i32>)>,
        pp_pipe: Option<bool>,
        vtok_dev: Option<&CudaSlice<u32>>,
        graphs: Option<&mut DsparkVerifyGraphs>,
    ) -> Result<(CudaSlice<f32>, CudaSlice<f32>), Box<dyn std::error::Error>> {
        // PP DOOR (lane/pp2-spec 2026-08-06): the verify trunk now takes its OWN stage split,
        // exactly as the eager and batched steps do. This is the single funnel every verify
        // forward reaches (decode_step_t / _h / _h_emb / _h_emb_dev / _core all land here), so
        // wiring it here wires the whole spec surface — the draft/accept/commit machinery above
        // is untouched.
        //
        // History: pp2-hardening (2026-08-06) made this funnel FAIL CLOSED, because its trunk
        // walk was unsplit on one stream and a sharded cross-device placement peer-read every
        // remote layer's weights on every spec round (measured 13.9-28x on the batched twin).
        // The refusal below survives to cover the residue — MEMRA_SPEC_PP=0, MEMRA_PP_STREAMS=0,
        // or a placement whose PpNRt fails to build — so a config that would still walk the
        // whole trunk on one stream refuses instead of regressing 28x.
        if let Some(fence) = crate::pp::pp_cuts(self.layers.len())
            && !crate::pp::pp2_streams_off()
            && crate::pp::spec_pp_on()
        {
            if vtok_dev.is_some() {
                return Err(
                    "device-token dspark verify (slice-2 deferred readback) has no PP \
                         stage-split arm; set MEMRA_DSPARK_DEFER_READBACK=0 or run the dspark \
                         route on one device"
                        .into(),
                );
            }
            return self.decode_step_t_core_ppn(
                e,
                tokens,
                pos0,
                cache,
                embd_dev,
                ckpt.take(),
                stream,
                &fence,
                pp_pipe,
            );
        }
        crate::pp::refuse_unsplit_if_remote(
            "decode_step_t (spec verify)",
            "drop MEMRA_SPEC_PP=0 / MEMRA_PP_STREAMS=0 so the verify trunk takes its OWN stage \
             split (decode_step_t_core_ppn); or run spec on one device",
        )?;
        let cfg = &self.cfg;
        let n_embd = cfg.n_embd as usize;
        let eps = cfg.rms_eps;
        let t = tokens.len();
        let pos_d = match stream {
            Some((_, ctr)) => {
                let mut p = e.alloc_uninit::<i32>(t)?;
                e.pos_iota(ctr, &mut p, t)?;
                p
            }
            None => {
                let pos_vec: Vec<i32> = (0..t).map(|i| (pos0 + i) as i32).collect();
                e.htod_i32(&pos_vec)?
            }
        };

        // embed T tokens -> [T, n_embd] token-major (device gather on the spec hot loop)
        let x = match (stream, embd_dev) {
            (Some((vtok, _)), Some((g, qt, rb))) => {
                e.embed_gather_device_td(g, vtok, t, n_embd, qt, rb)?
            }
            (None, Some((g, qt, rb))) => match vtok_dev {
                // slice 2: device verify tokens, same embed_gather_u32_t kernel —
                // bit-identical rows to the host-token arm (same per-dtype deq).
                Some(vt_d) => e.embed_gather_device_td(g, vt_d, t, n_embd, qt, rb)?,
                None => e.embed_gather_device_t(g, tokens, n_embd, qt, rb)?,
            },
            _ => {
                assert!(
                    vtok_dev.is_none(),
                    "device-token verify requires the resident embed table (embd_dev)"
                );
                e.htod(&self.embd.try_gather(n_embd, tokens)?)?
            }
        };

        // TRUNK WALK: layers [0, n_layers) through the SAME range-scoped subgraph the PP-N
        // stage split calls per stage (`verify_layers`) — one code path, so the split cannot
        // drift from the unsplit dispatch mirroring. lane/pp2-spec 2026-08-06.
        let x = self.verify_layers(
            e,
            x,
            0,
            self.layers.len(),
            &pos_d,
            pos0,
            t,
            cache,
            ckpt.take(),
            stream,
            graphs,
        )?;
        if spec_nan_scan() {
            nan_scan_rows(e, &x, t, n_embd, &format!("verify trunk exit pos0={pos0}"))?;
        }

        let mut hn = vbuf(e, t * n_embd)?;
        // Stage-A door: with the serving-class row-outer verify walk, the TAIL must be the
        // t=1 decode program per row too (rms_norm t=1 + the single-row bf16 head — the
        // split head's concat is receipted bit-identical to it). The batched cuBLASLt head
        // is a different ULP class and flips near-tie argmaxes off the greedy tape.
        let eager_tail = self.sliding_gated_moe_batch_program() && spec_verify_eager_on();
        if eager_tail {
            let n_vocab = self.cfg.n_vocab as usize;
            // MEMRA_SPEC_HEAD_ROWS=1 — THE VERIFY TAIL'S REDUNDANT HEAD READ.
            //
            // The loop below runs the head at m=1 once PER COLUMN, so the LM head's weights are
            // streamed t times per verify pass. On step37 that head is ~0.49 GiB per card after the
            // rank split, ~1.07 ms of pure re-read at t=2 and worse at every wider t — which is a
            // large part of why the fixed K ladder LOSES (K=1 81.2 > K=2 73.1 > K=3 62.7 tok/s).
            //
            // The loop's justification is the comment above: the batched cuBLASLt head is a
            // different ULP class and flips near-tie argmaxes off the greedy tape. That is true of
            // cuBLASLt and it does NOT apply here, because a FloatBf16 head at 1..=32 rows never
            // reaches cuBLASLt: `matmul` routes it to `matvec_bf16_rows_into` (lib.rs:12248), whose
            // own doc says `matvec_bf16_f32acc_x4_rows` "runs the t=1 decode head program PER ROW
            // (identical dot + reduce), so decode/verify tiers keep the t=1 numeric class". Under
            // the W8 doors both widths route to the q8 mirror instead, and the t-column mirror is
            // documented "bit-identical to t single-row calls". So the batched form is the SAME
            // arithmetic per row on both paths, with one weight read instead of t.
            //
            // rms_norm is row-wise, so norm(t) is per-row identical to t x norm(1) by construction.
            //
            // DEFAULT OFF for exactly one turn of the crank: "bit-identical by two documented
            // claims" is still an argument. The greedy byte tape decides, and the door flips only
            // once the tape is a receipt.
            if head_rows_on() {
                e.rms_norm(&x, self.output_norm.float_data(), &mut hn, n_embd, t, eps)?;
                let logits = e.matmul(&self.output, &hn, t)?;
                if stream.is_none() {
                    cache.pos += t;
                }
                return Ok((logits, if spec_hpost() { hn } else { x }));
            }
            let mut logits = vbuf(e, t * n_vocab)?;
            for r in 0..t {
                let mut row = e.uninit(n_embd)?;
                e.dtod_copy_view(&x.slice(r * n_embd..(r + 1) * n_embd), &mut row)?;
                let mut hr = e.uninit(n_embd)?;
                e.rms_norm(&row, self.output_norm.float_data(), &mut hr, n_embd, 1, eps)?;
                let lr = e.matmul(&self.output, &hr, 1)?;
                e.dtod_copy_into(&lr, &mut logits, r * n_vocab)?;
                e.dtod_copy_into(&hr, &mut hn, r * n_embd)?;
            }
            if stream.is_none() {
                cache.pos += t;
            }
            return Ok((logits, if spec_hpost() { hn } else { x }));
        }
        let serving_head =
            self.sliding_gated_moe_batch_program() || self.batched_serving_numeric_class();
        let logits = if serving_head {
            // Step35 and the qwen35 family (MoE 2026-08-14 AM, dense-hybrid same day PM — the
            // Q3.8 bring-up reproduced the identical near-tie class on dense: eager-class verify
            // vs batched-class live serving, ULP drift amplified through the GDN recurrence)
            // serve one batched numeric class at every live width, including B=1. Keep the
            // verify head in that same class; other generic families retain the decode-exact
            // head that their run-spec contract pins.
            e.rms_norm(&x, self.output_norm.float_data(), &mut hn, n_embd, t, eps)?;
            e.matmul(&self.output, &hn, t)?
        } else {
            e.rms_norm_decode(&x, self.output_norm.float_data(), &mut hn, n_embd, t, eps)?;
            e.matmul_decode_exact(&self.output, &hn, t)?
        };
        // stream: the device pos counter owns position; host mirror reconciles at drain.
        if stream.is_none() {
            cache.pos += t;
        }
        // Hidden stack for seeds/refresh-fills: pre-norm x (default) or post-norm hn (HPOST).
        Ok((logits, if spec_hpost() { hn } else { x }))
    }

    /// THE VERIFY TRUNK OVER PP-N (lane/pp2-spec 2026-08-06): `decode_step_t_core_stream`'s walk
    /// as N stage subgraphs, each on its own engine/stream (and, under `MEMRA_PP_DEVICES`, its own
    /// device), with a `[T, n_embd]` boundary transfer between them. T = K+1 (the verify batch),
    /// so this is the batched-boundary shape the pp2-batch lane's grow-only slots already handle
    /// (`tx(b, x, t*n_embd)`; the slot grows to the high-water T and the transport moves exactly
    /// the payload).
    ///
    /// Structure is `decode_step_batch_ppn`'s, which is `decode_step_h_ppn`'s. FOUR THINGS ARE
    /// PER-STAGE and each for a measured reason (see `decode_step_batch_ppn`'s header for the
    /// receipts):
    ///
    /// 1. THE ENGINE (`rt.engine(s, e)`) — `Engine` owns lazily-grown stable-pointer scratch
    ///    (`fa_part_pool`, `fa_vf16_scratch`, `argmax_partials`) that is single-stream-safe BY
    ///    DESIGN. Two stage streams through one Engine is the 2026-08-02 shared-scratch race
    ///    (35% flake, nondeterministic all-logits divergence). `PpNRt::build` gives every stage
    ///    s>0 its own Engine even on the primary device; honouring it here is what scopes the
    ///    pools. The verify path allocates MORE of that scratch than eager decode does (FA at
    ///    m=T, and the per-layer `GdnStash` retains), so this is load-bearing, not inherited.
    ///
    /// 2. `pos_d` — each stage uploads its OWN copy of the T rope positions on ITS stream, so the
    ///    buffer is allocated, consumed and freed on one stream. In `stream` mode that means each
    ///    stage runs its own `pos_iota` over the SHARED device counter (`pos_ctr`): the counter is
    ///    read-only during the forward (the round's `inc`/`copy_add` happen outside it), so every
    ///    stage derives the identical iota, and each stage's own output buffer is stream-local.
    ///
    /// 3. THE EMBED lives with stage 0 (`self.embd` / `embd_gpu` are host/primary-side; the
    ///    sharded loader leaves the table with stage 0 by construction).
    ///
    /// 4. THE HEAD (`output_norm` + `output`) runs on the LAST stage — the sharded loader uploaded
    ///    both through that stage's engine (`hybrid.rs`: `e_head = layer_engine(e, n_trunk,
    ///    n_trunk-1)`), so reading them anywhere else is a peer read of the biggest tensor in the
    ///    model, every round.
    ///
    /// WHAT STAYS ON THE PRIMARY, deliberately: the returned logits and hidden stack `x`. Both are
    /// last-stage-allocated device buffers, and every consumer (the device argmax walk, the accept
    /// kernels, `spec_seed_gather`, the ckpt rebuild in `commit_verified_prefix`) reads them
    /// through the primary context by UVA — the same read the batched serving epilogue's
    /// `last_logits_dev` park does. Those consumers are per-round O(T x n_vocab) and O(n_embd),
    /// not per-layer, so they are not the 28x class; splitting them is a separate lane.
    ///
    /// The MTP HEAD (draft side) is NOT split: it is one block, it lives wherever the loader put
    /// it (`load_mtp` uses the primary engine), and it is ~1-2 GB against the trunk's tens. Draft
    /// placement is measured, not assumed — see `research/pp2-spec-20260806`.
    ///
    /// EXACTNESS: PP-N adds ZERO deviation. Each stage runs the SAME kernels on the SAME bytes in
    /// the same order via the SAME `verify_layers` the unsplit body calls; the only change is
    /// where the residual is materialized, and the boundary is a straight f32 copy (dtod
    /// same-device / `cudaMemcpyPeerAsync` cross-device, no conversion). So the split MUST be
    /// BIT-IDENTICAL to the unsplit verify at the same T, in both placement orders. Gate:
    /// `decode-batch-gate --mode ppspec`. Acceptance counts are a DERIVED consequence — greedy
    /// accept argmaxes these logits, so bit-identical logits force identical accept walks; the
    /// `run-spec` K=1..8 arm checks that end-to-end rather than trusting the implication.
    #[allow(clippy::too_many_arguments)]
    fn decode_step_t_core_ppn(
        &self,
        e: &Engine,
        tokens: &[u32],
        pos0: usize,
        cache: &mut Cache,
        embd_dev: Option<(&CudaSlice<u8>, i32, usize)>,
        mut ckpt: Option<&mut VerifyCkpt>,
        stream: Option<(&CudaSlice<u32>, &CudaSlice<i32>)>,
        fence: &[usize],
        pp_pipe: Option<bool>,
    ) -> Result<(CudaSlice<f32>, CudaSlice<f32>), Box<dyn std::error::Error>> {
        let ticket = self.verify_stage0_issue(
            e,
            tokens,
            pos0,
            cache,
            embd_dev,
            ckpt.as_deref_mut(),
            stream,
            fence,
            pp_pipe,
            None,
        )?;
        self.verify_stage1_finish(e, ticket, cache, ckpt, stream, fence, true)
    }

    /// Enqueue embed, stage 0, and the first boundary TX, then return the actual boundary slot.
    /// The ordinary PP verify wrapper calls `verify_stage1_finish` immediately after this return.
    #[allow(clippy::too_many_arguments)]
    fn verify_stage0_issue(
        &self,
        e: &Engine,
        tokens: &[u32],
        pos0: usize,
        cache: &mut Cache,
        embd_dev: Option<(&CudaSlice<u8>, i32, usize)>,
        ckpt: Option<&mut VerifyCkpt>,
        stream: Option<(&CudaSlice<u32>, &CudaSlice<i32>)>,
        fence: &[usize],
        pp_pipe: Option<bool>,
        trace: Option<SpecPipeTraceCtx>,
    ) -> Result<VerifyBoundaryTicket, Box<dyn std::error::Error>> {
        assert!(
            !self.is_gemma4_e4b() && !self.gemma_batch_program(),
            "decode_step_t_core_ppn covers the hybrid non-gemma4 verify trunk only \
             (the gemma4 arms have their own decode_step_t twins)"
        );
        if crate::pp::pp_host_bounce_active() && (stream.is_some() || embd_dev.is_some()) {
            return Err(
                "decode_step_t_core_ppn: refused with MEMRA_PP_HOST_BOUNCE=1 — the trunk \
                 boundary itself is host-staged, but device-resident verify still peer-reads \
                 primary-device token/position/embedding buffers from stage 0. Run plain PP \
                 serving on this host class; spec requires local per-stage inputs first."
                    .into(),
            );
        }
        let rt = crate::pp::PpNRt::get(e)?;
        // Pipelined callers do not bypass ownership: their explicit coordinator borrow makes
        // this acquire clone the same active generation. Ordinary callers acquire a fresh lease.
        let walk_owner = rt.acquire_walk("verify_stage0_issue")?;
        let n_st = fence.len() - 1;
        assert_eq!(
            rt.n_stages(),
            n_st,
            "PpNRt stage count {} != fence stages {n_st}",
            rt.n_stages()
        );
        let n_embd = self.cfg.n_embd as usize;
        let t = tokens.len();
        let payload = t * n_embd;
        if pp_pipe.is_some() {
            assert_eq!(n_st, 2, "spec pipeline requires exactly two PP stages");
        }
        // One-shot lane diagnostic: force natural PP-2 boundaries to completion so the server
        // log can price stage 0, the peer hop, the RX copy, and stage 1 + head separately without
        // nsys. The ordinary path keeps every enqueue asynchronous. N>2 is deliberately excluded:
        // the report below names exactly two stages and must never imply it measured middle ones.
        let pp_anatomy = n_st == 2 && std::env::var("MEMRA_SPEC_PP_ANATOMY").as_deref() == Ok("1");
        let pp_started = std::time::Instant::now();
        let (mut reverse_ms, mut stage0_ms, mut tx_ms) = (0.0f64, 0.0f64, 0.0f64);
        // The CALLER's ambient stream, captured BEFORE any `rt.enter()` pushes a stage stream:
        // this body returns DEVICE-RESIDENT buffers (the device-argmax accept walk's contract),
        // so the exit needs the same publication the boundaries get — see `PpNRt::publish_to`.
        // Taken here, not at the end, because inside the last-stage scope `e.stream()` IS the
        // stage stream and the wait would self-order into a no-op.
        let caller_stream = e.stream();
        // #87 ROOT-CAUSE FENCE (lane/pp2spec-crash): the PREVIOUS round's stage-allocated
        // outputs (logits/hidden/ckpt stashes) freed stream-ordered on the STAGE streams while
        // the primary stream still holds queued reads of them — with event tracking elided,
        // nothing stops the pool from reusing those blocks for THIS round's stage allocations,
        // whose writes then race the queued reads (measured: 13/4096-NaN random-bits garbage in
        // the spec round seed; the full anatomy is on `PpNRt::fence_stages_behind`). Order every
        // stage stream behind the caller before enqueueing new stage work.
        let reverse_started = std::time::Instant::now();
        if pp_pipe != Some(false) {
            rt.fence_stages_behind(&caller_stream)?;
        }
        if pp_pipe == Some(true) {
            // Both session verifies must alternate boundary slots even when the ordinary
            // decode overlap experiment is off. Prewarm before A's stage 0 so B cannot grow
            // slot 1 by synchronizing the RX stream while A's stage 1 is in flight.
            rt.prepare_overlap_slots(0, payload)?;
        }
        if pp_anatomy {
            // Drain the reverse-publication dependency before timing stage 0 itself. At c=1 this
            // prices any primary-stream rollback/refresh tail inherited from the prior round.
            for s in 0..n_st {
                let _st = rt.enter(s);
                rt.engine(s, e).stream().synchronize()?;
            }
            reverse_ms = reverse_started.elapsed().as_secs_f64() * 1e3;
        }

        // Per-stage rope positions: in host mode the same [T] iota each stage uploads itself; in
        // stream mode each stage's own `pos_iota` over the shared read-only device counter.
        let stage_pos = |es: &Engine| -> Result<CudaSlice<i32>, Box<dyn std::error::Error>> {
            match stream {
                Some((_, ctr)) => {
                    let mut p = es.alloc_uninit::<i32>(t)?;
                    es.pos_iota(ctr, &mut p, t)?;
                    Ok(p)
                }
                None => {
                    let pos_vec: Vec<i32> = (0..t).map(|i| (pos0 + i) as i32).collect();
                    es.htod_i32(&pos_vec)
                }
            }
        };

        // ---- STAGE 0: embed (the table lives with stage 0) + layers [0, fence[1]) + TX ----
        let slot = {
            let _st0 = rt.enter(0);
            let e0 = rt.engine(0, e);
            enqueue_spec_pipe_trace_marker(&e0.stream(), trace.as_ref(), "S0", "start", None)?;
            let stage0_started = std::time::Instant::now();
            let pos_d = stage_pos(e0)?;
            let x = match (stream, embd_dev) {
                (Some((vtok, _)), Some((g, qt, rb))) => {
                    e0.embed_gather_device_td(g, vtok, t, n_embd, qt, rb)?
                }
                (None, Some((g, qt, rb))) => e0.embed_gather_device_t(g, tokens, n_embd, qt, rb)?,
                _ => e0.htod(&self.embd.try_gather(n_embd, tokens)?)?,
            };
            let x = self.verify_layers(
                e0, x, fence[0], fence[1], &pos_d, pos0, t, cache, ckpt, stream, None,
            )?;
            if pp_anatomy {
                e0.stream().synchronize()?;
                stage0_ms = stage0_started.elapsed().as_secs_f64() * 1e3;
            }
            let tx_started = std::time::Instant::now();
            let slot = if pp_pipe.is_some() {
                rt.tx_pipelined(0, &x, payload)?
            } else {
                rt.tx(0, &x, payload)?
            };
            enqueue_spec_pipe_trace_marker(&e0.stream(), trace.as_ref(), "S0", "end", Some(slot))?;
            if pp_anatomy {
                e0.stream().synchronize()?;
                tx_ms = tx_started.elapsed().as_secs_f64() * 1e3;
            }
            slot
            // x + pos_d drop here: freed stream-ordered on stage-0's stream after use.
        };

        Ok(VerifyBoundaryTicket {
            rt,
            caller_stream,
            slot,
            pos0,
            t,
            payload,
            n_st,
            pipelined: pp_pipe.is_some(),
            pp_anatomy,
            pp_started,
            reverse_ms,
            stage0_ms,
            tx_ms,
            trace,
            _walk_owner: walk_owner,
        })
    }

    /// Consume a stage-0 boundary ticket and enqueue the remaining PP stages plus the head.
    /// On PP-2 this is exactly stage 1; PP-N keeps its pre-existing middle-stage walk here.
    #[allow(clippy::too_many_arguments)]
    fn verify_stage1_finish(
        &self,
        e: &Engine,
        ticket: VerifyBoundaryTicket,
        cache: &mut Cache,
        mut ckpt: Option<&mut VerifyCkpt>,
        stream: Option<(&CudaSlice<u32>, &CudaSlice<i32>)>,
        fence: &[usize],
        publish_to_caller: bool,
    ) -> Result<(CudaSlice<f32>, CudaSlice<f32>), Box<dyn std::error::Error>> {
        let VerifyBoundaryTicket {
            rt,
            caller_stream,
            slot,
            pos0,
            t,
            payload,
            n_st,
            pipelined,
            pp_anatomy,
            pp_started,
            reverse_ms,
            stage0_ms,
            tx_ms,
            trace,
            _walk_owner,
        } = ticket;
        let n_embd = self.cfg.n_embd as usize;
        let eps = self.cfg.rms_eps;
        let mut slot = slot;
        let (mut rx_ms, mut stage1_ms) = (0.0f64, 0.0f64);
        let stage_pos = |es: &Engine| -> Result<CudaSlice<i32>, Box<dyn std::error::Error>> {
            match stream {
                Some((_, ctr)) => {
                    let mut p = es.alloc_uninit::<i32>(t)?;
                    es.pos_iota(ctr, &mut p, t)?;
                    Ok(p)
                }
                None => {
                    let pos_vec: Vec<i32> = (0..t).map(|i| (pos0 + i) as i32).collect();
                    es.htod_i32(&pos_vec)
                }
            }
        };

        // ---- MIDDLE STAGES: RX boundary s-1 -> range -> TX boundary s ----
        for s in 1..n_st - 1 {
            let _st = rt.enter(s);
            let es = rt.engine(s, e);
            let pos_d = stage_pos(es)?;
            let x = rt.rx(s - 1, slot, payload)?;
            let x = self.verify_layers(
                es,
                x,
                fence[s],
                fence[s + 1],
                &pos_d,
                pos0,
                t,
                cache,
                ckpt.as_deref_mut(),
                stream,
                None,
            )?;
            slot = if pipelined {
                rt.tx_pipelined(s, &x, payload)?
            } else {
                rt.tx(s, &x, payload)?
            };
        }

        // ---- LAST STAGE: RX + final range + output_norm + lm head ----
        let _stl = rt.enter(n_st - 1);
        let el = rt.engine(n_st - 1, e);
        let pos_d = stage_pos(el)?;
        let rx_started = std::time::Instant::now();
        let x = rt.rx(n_st - 2, slot, payload)?;
        if pp_anatomy {
            el.stream().synchronize()?;
            rx_ms = rx_started.elapsed().as_secs_f64() * 1e3;
        }
        enqueue_spec_pipe_trace_marker(&el.stream(), trace.as_ref(), "S1", "start", Some(slot))?;
        let stage1_started = std::time::Instant::now();
        let x = self.verify_layers(
            el,
            x,
            fence[n_st - 1],
            fence[n_st],
            &pos_d,
            pos0,
            t,
            cache,
            ckpt,
            stream,
            None,
        )?;

        let mut hn = vbuf(el, payload)?;
        let logits = if self.sliding_gated_moe_batch_program() {
            // The PP Step35 serving path uses rms_norm + matmul for B=1 as well as B>1.
            // Verify must not switch numeric class merely because the same session speculates.
            el.rms_norm(&x, self.output_norm.float_data(), &mut hn, n_embd, t, eps)?;
            el.matmul(&self.output, &hn, t)?
        } else {
            el.rms_norm_decode(&x, self.output_norm.float_data(), &mut hn, n_embd, t, eps)?;
            el.matmul_decode_exact(&self.output, &hn, t)?
        };
        enqueue_spec_pipe_trace_marker(&el.stream(), trace.as_ref(), "S1", "end", Some(slot))?;
        if pp_anatomy {
            el.stream().synchronize()?;
            stage1_ms = stage1_started.elapsed().as_secs_f64() * 1e3;
        }
        // EXIT PUBLICATION: both returned buffers are still being produced on the last stage's
        // stream. Order the caller's stream behind that work before the buffers escape this
        // scope (the 2026-08-06 same-device ppspec find: without it the caller's primary-stream
        // consumer read unwritten logits — nondeterministic, one-device-only, and it poisoned
        // the following arm's KV in the same process).
        if publish_to_caller {
            rt.publish_to(n_st - 1, &caller_stream)?;
        }
        if pp_anatomy {
            if publish_to_caller {
                caller_stream.synchronize()?;
            }
            eprintln!(
                "[spec-pp-anatomy] t={t} reverse={reverse_ms:.3}ms stage0={stage0_ms:.3}ms \
                 tx={tx_ms:.3}ms rx={rx_ms:.3}ms stage1-head={stage1_ms:.3}ms total={:.3}ms",
                pp_started.elapsed().as_secs_f64() * 1e3,
            );
        }
        // stream: the device pos counter owns position; host mirror reconciles at drain.
        if stream.is_none() {
            cache.pos += t;
        }
        Ok((logits, if spec_hpost() { hn } else { x }))
    }

    /// Step3.5/Step3.7 verify trunk in the serving batched numeric class.
    ///
    /// `step35_decode_batch_layers` is now authoritative at every live serving width, including
    /// B=1 (lane/cx-b1fix). The older verify walk deliberately mirrored the eager T=1 class:
    /// it replayed `step35_decode_attn` per row and used the eager/decode-exact FFN dispatch.
    /// Those classes are individually stable, but a near-tie prompt can choose different greedy
    /// bytes when a request moves from batched plain serving into speculative verify. Run the
    /// same authoritative B=1 stage subgraph for each verify row here. Rows still advance
    /// layer-by-layer, so every layer sees the preceding verify rows in its attention cache while
    /// every norm/projection/FFN uses exactly the live serving dispatch.
    #[allow(clippy::too_many_arguments)]
    /// PRIME-BY-T-ROWS (MEMRA_PRIME_TROWS=1): prefill the prompt through the same-session
    /// t-row walk in 32-row chunks — every row runs the t=1 decode program bit-for-bit
    /// (the TOKENWISE-prime ORACLE class), so this door is exact against the exactness
    /// reference while replacing the host-canonical per-token prime. Requires the walk
    /// doors (MEMRA_SPEC_VERIFY_EAGER/TCOL); returns the prime contract trio.
    #[allow(clippy::type_complexity)]
    pub(crate) fn step35_prime_trows(
        &self,
        e: &Engine,
        tokens: &[u32],
        cache: &mut Cache,
    ) -> Result<Option<(Vec<f32>, CudaSlice<f32>, CudaSlice<f32>)>, Box<dyn std::error::Error>>
    {
        let dbg = std::env::var("MEMRA_SPEC_FA2_DEBUG").as_deref() == Ok("1");
        if !prime_trows_on() {
            return Ok(None);
        }
        if !self.uses_sliding_gated_moe_program()
            || cache.pos != 0
            || cache.dflash_taps.is_some()
            || !spec_verify_eager_on()
            || !spec_verify_tcol_on()
        {
            if dbg {
                eprintln!(
                    "[prime-trows] refuse: program={} pos={} taps={} eager={:?} tcol={:?}",
                    self.uses_sliding_gated_moe_program(),
                    cache.pos,
                    cache.dflash_taps.is_some(),
                    std::env::var("MEMRA_SPEC_VERIFY_EAGER").ok(),
                    std::env::var("MEMRA_SPEC_VERIFY_TCOL").ok()
                );
            }
            return Ok(None);
        }
        let n_embd = self.cfg.n_embd as usize;
        let n_layers = self.layers.len();
        let t_total = tokens.len();
        let Some(embd_gpu) = self.embd_gpu_try(e) else {
            if dbg {
                eprintln!("[prime-trows] refuse: no device embed table");
            }
            return Ok(None);
        };
        let embd_qtype = match self.embd.ggml_type {
            memra_gguf::GgmlType::BF16 => crate::QT_BF16,
            memra_gguf::GgmlType::Q8_0 => crate::QT_Q8_0,
            other => {
                if dbg {
                    eprintln!("[prime-trows] refuse: embed dtype {other:?}");
                }
                return Ok(None);
            }
        };
        let embd_row_bytes = self.embd.raw.len() / self.cfg.n_vocab as usize;
        // Chunk plan: 32-row chunks; a 1-token tail folds into the previous chunk
        // (the walk floor is t >= 2).
        let mut bounds = Vec::new();
        let mut start = 0usize;
        while start < t_total {
            let mut end = (start + 32).min(t_total);
            if t_total - end == 1 {
                end -= 1;
            }
            bounds.push((start, end));
            start = end;
        }
        if bounds.iter().any(|(a, b)| b - a < 2) {
            return Ok(None); // degenerate short prompt keeps the ordinary prime
        }
        let mut hiddens = e.uninit(t_total * n_embd)?;
        let mut last: Option<CudaSlice<f32>> = None;
        for &(a, b) in &bounds {
            let tc = b - a;
            let tok_d = e.stream().clone_htod(&tokens[a..b])?;
            let x =
                e.embed_gather_device_td(embd_gpu, &tok_d, tc, n_embd, embd_qtype, embd_row_bytes)?;
            let out = self.step35_verify_batch_layers(e, x, 0, n_layers, a, tc, cache)?;
            e.copy_into(&mut hiddens, a * n_embd, &out, tc * n_embd)?;
            if b == t_total {
                let mut h = e.uninit(n_embd)?;
                e.dtod_copy_view(&out.slice((tc - 1) * n_embd..tc * n_embd), &mut h)?;
                last = Some(h);
            }
        }
        let h_seed = last.expect("last chunk produced the seed row");
        let mut hn = e.uninit(n_embd)?;
        e.rms_norm_decode(
            &h_seed,
            self.output_norm.float_data(),
            &mut hn,
            n_embd,
            1,
            self.cfg.rms_eps,
        )?;
        let logits_d = e.matmul_decode_exact(&self.output, &hn, 1)?;
        let logits = e.dtoh(&logits_d)?;
        cache.pos = t_total;
        Ok(Some((logits, h_seed, hiddens)))
    }

    #[allow(clippy::too_many_arguments)] // allow: the parameter list mirrors the kernel/FFI/call contract; bundling into a struct is a refactor, not a lint fix
    fn step35_verify_batch_layers(
        &self,
        e: &Engine,
        mut x: CudaSlice<f32>,
        lo: usize,
        hi: usize,
        pos0: usize,
        t: usize,
        cache: &mut Cache,
    ) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        let n_embd = self.cfg.n_embd as usize;
        if !self.uses_sliding_gated_moe_program() {
            return Err(
                "serving-class verify requires sliding-gated-MoE canonical operations".into(),
            );
        }
        // SERVING-CLASS VERIFY (MEMRA_SPEC_VERIFY_EAGER=1, step37 MTP bring-up): each verify
        // column rides decode_layers_eager — the EXACT t=1 program live serving runs (all TP2
        // doors) — row-outer, so row r's appends land before row r+1 attends: bit-equal to
        // plain greedy by construction. Only the unsplit full-range walk qualifies; PP splits
        // and the tap path keep the batch-layer class.
        static VE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let eager_verify =
            *VE.get_or_init(spec_verify_eager_on) && lo == 0 && hi == self.layers.len();
        if eager_verify {
            // T-COLUMN LAYER-OUTER WALK (MEMRA_SPEC_VERIFY_TCOL=1): per layer, one t-grid
            // attn norm + ONE weight-amortized QKV(+gate) over all T columns, then each
            // column runs the UNMODIFIED t=1 attention program via the col-select door and
            // the ordinary residual/FFN body. Values per column are bit-equal to the
            // row-outer walk: rms over the materialized residual == the fused add+norm
            // (kernel_check identity), the tcol kernel's per-column FP order == the t=1
            // kernel, and every downstream op IS the t=1 program.
            static TCOL: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            let tcol = *TCOL.get_or_init(spec_verify_tcol_on);
            // T > 32 (prefill-class): run the SAME walk in 32-row chunks — each chunk's
            // rows are the t=1 program bit-for-bit and the rope pass advances the cache,
            // so a chunked call is value-identical to the row-outer loop it replaces.
            static TROWS_PREFILL: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            // MEMRA_STEP_GEMM_PRIME outranks the walk: with the grouped GEMM prime armed, the
            // t-row walk defers so the batch path (GEMM trunk + grouped MoE) takes the prompt —
            // flag precedence between two existing doors, not a new flag. Without this, both
            // doors ON meant the walk still won and the GEMM prime needed PRIME_TROWS=0 by hand.
            let trows_prefill =
                *TROWS_PREFILL.get_or_init(|| prime_trows_on() && !crate::step_gemm_prime_on());
            // MEMRA_PRIME_TROWS_T=<w>: chunk width, default 8 = the REAL cap of this walk.
            // The workspace slabs go to 32 rows, but `matvec_bf16_qkvg_tcol_into` refuses
            // t > 8 (compile-time-T twins exist for 2/4/8 only; the runtime-t kernel spills
            // its accumulators to local memory), so a wider chunk fails the request with
            // "matvec_bf16_qkvg_tcol geometry" — which is exactly how the first server-path
            // TROWS arm died. Measured at 193 tokens: w=8 2.459 s, w=4 2.574 s.
            static TROWS_W: std::sync::OnceLock<Result<usize, String>> = std::sync::OnceLock::new();
            let trows_w = match TROWS_W.get_or_init(|| {
                let value = std::env::var("MEMRA_PRIME_TROWS_T").ok();
                parse_prime_trows_width(value.as_deref())
            }) {
                Ok(width) => *width,
                Err(err) => return Err(err.clone().into()),
            };
            if tcol && trows_prefill && t > trows_w {
                // One-time engagement receipt: without it a prefill gate cannot tell a
                // chunked walk from the row-outer fallback it is supposed to replace
                // (the first PRIME_TROWS gate passed vacuously on exactly that).
                static SEEN: std::sync::atomic::AtomicBool =
                    std::sync::atomic::AtomicBool::new(false);
                if !SEEN.swap(true, std::sync::atomic::Ordering::Relaxed) {
                    eprintln!(
                        "[prime-trows] ENGAGED t={t} width={trows_w} chunks={} layers={}..{}",
                        t.div_ceil(trows_w),
                        lo,
                        hi
                    );
                }
                let mut out = e.uninit(t * n_embd)?;
                let mut start = 0usize;
                while start < t {
                    let mut end = (start + trows_w).min(t);
                    if t - end == 1 {
                        end -= 1;
                    }
                    let tc = end - start;
                    let mut xc = e.uninit(tc * n_embd)?;
                    e.dtod_copy_view(&x.slice(start * n_embd..end * n_embd), &mut xc)?;
                    let oc =
                        self.step35_verify_batch_layers(e, xc, lo, hi, pos0 + start, tc, cache)?;
                    e.copy_into(&mut out, start * n_embd, &oc, tc * n_embd)?;
                    start = end;
                }
                return Ok(out);
            }
            if tcol && (2..=32).contains(&t) {
                // MEMRA_TCOL_PROF=1: synchronized per-segment wall profile of the walk
                // (norm+QKV precompute / per-col attention / per-col residual+FFN). The
                // syncs serialize the stream, so the split is for TARGETING amortization
                // work only — never a perf claim.
                static PROF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
                let prof =
                    *PROF.get_or_init(|| std::env::var("MEMRA_TCOL_PROF").as_deref() == Ok("1"));
                let mut prof_ms = [0f64; 3];
                let eps = self.cfg.rms_eps;
                let mut x_t = x;
                let mut h_t = e.uninit(t * n_embd)?;
                let mut h_row = e.uninit(n_embd)?; // real row: the non-dcw fallback reads it
                // Per-column pos buffers hoisted out of the layer loop (a per-col-per-layer
                // pageable htod was an in-stream engine turnaround x t x 45).
                let mut pos_rows = Vec::with_capacity(t);
                for r in 0..t {
                    pos_rows.push(e.htod_i32(&[(pos0 + r) as i32])?);
                }
                let mut ok = true;
                // MEMRA_TCOL_OPROJ=1: defer each column's o_proj — the finish seam
                // stashes `gated` instead of joining per column; one b4_tcol per rank +
                // one slab join produce every column's `mixed` after the attention pass.
                // Bit-exact per column (t=1 b4 program per column; elementwise join).
                // MEMRA_TCOL_FFN=1: today this only IMPLIES the o_proj defer above. Its
                // named feature, the two-column device-routed FFN sweep, rode the
                // slot-major v2 TP banks and was REMOVED with the MEMRA_NVFP4_BANK_V2 door
                // (2026-08-29, research/step37-bankv2-removal-20260829): the v2 layout
                // changed generated text in serving. The flag itself stays because it is
                // family-armed in the step37 serving defaults and killing it here would
                // silently drop the o_proj defer from the qualified serving shape.
                static FFN2: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
                let ffn_batch = *FFN2.get_or_init(tcol_ffn_on);
                let oproj_batch = crate::tp::tcol_oproj_on() || ffn_batch;
                // MEMRA_SPEC_FA2=1 (T=2 only): eligible layers defer BOTH columns' fa —
                // the per-column pass norms/ropes/appends and stashes q+gate, then one
                // shared-KV fa_decode_dcw2 per rank + the o_proj join produce the
                // [2, o_out] mixed slab. The precheck runs before arming (stashing is
                // unrecoverable); ineligible/boundary layers run the ordinary program.
                let fa2 = crate::tp::spec_fa2_on() && t <= 32;
                let mut mixed_row = e.uninit(n_embd)?;
                let mut pos_staged = false;
                for il in lo..hi {
                    let layer = &self.layers[il];
                    // BEFORE this layer touches its planes: is the history it is about to
                    // attend already poisoned? Global (non-ring) layers only, which are the
                    // ones the level-2 bitmap implicates.
                    if kv_plane_scan_on()
                        && self.step35_geom(il).window.is_none()
                        && let Some(distributed) = cache.tp_kv[il].as_ref()
                    {
                        scan_kv_plane(e, distributed, il, pos0)?;
                    }
                    let fa2_layer = fa2 && self.step35_fa_rows_precheck(cache, il, pos0, t)?;
                    let mut seg = std::time::Instant::now();
                    e.rms_norm(&x_t, layer.attn_norm.float_data(), &mut h_t, n_embd, t, eps)?;
                    if !self.step35_verify_qkv_precompute(e, il, &h_t, t)? {
                        ok = false;
                        break;
                    }
                    // FULL t-row attention pass (rope/append + fa + combine + o_proj in
                    // 3 launches/rank): same-session rows, slot = len-base+r, one len
                    // advance by t. Host cache bookkeeping mirrors the per-column tail.
                    if fa2_layer
                        && let Some(mixed_t) =
                            self.step35_verify_rope_fa_pass(e, il, cache, pos0, t, !pos_staged)?
                    {
                        pos_staged = true;
                        {
                            let tp_kv = cache.tp_kv[il]
                                .as_mut()
                                .expect("precheck verified the distributed cache");
                            let transaction = tp_kv.begin_transaction()?;
                            let crate::hybrid::Mixer::Full(fa) = &layer.mixer else {
                                return Err("verify rope pass expects full attention".into());
                            };
                            let tp = fa
                                .step_tp_qkv
                                .as_ref()
                                .ok_or("verify rope pass lost its TP state")?;
                            let empty: [CudaSlice<f32>; 0] = [];
                            tp.runtime.append_tp_kv_transaction_inner(
                                tp_kv,
                                transaction,
                                &empty,
                                &empty,
                                t,
                                true,
                            )?;
                            tp.runtime
                                .commit_tp_kv_transaction_external(tp_kv, transaction, t)?;
                            if let Some(local) = cache.kv[il].as_mut() {
                                local.len = pos0 + t;
                                if !crate::tp::len_mirror_lazy_on() {
                                    e.set_i32_one(&mut local.len_d, local.len as i32)?;
                                }
                            }
                        }
                        if prof {
                            e.stream().synchronize()?;
                            prof_ms[1] += seg.elapsed().as_secs_f64() * 1e3;
                            seg = std::time::Instant::now();
                        }
                        let o_out = mixed_t.len() / t;
                        let mut next = e.uninit(t * n_embd)?;
                        {
                            for r in 0..t {
                                e.dtod_copy_view(
                                    &mixed_t.slice(r * o_out..(r + 1) * o_out),
                                    &mut mixed_row,
                                )?;
                                let mut x_row = e.uninit(n_embd)?;
                                e.dtod_copy_view(
                                    &x_t.slice(r * n_embd..(r + 1) * n_embd),
                                    &mut x_row,
                                )?;
                                let (x1, ffn_out) = self.residual_norm_ffn(
                                    e, layer, &x_row, &mixed_row, n_embd, il, eps,
                                )?;
                                let mut x2 = e.uninit(n_embd)?;
                                e.add(&x1, &ffn_out, &mut x2, n_embd)?;
                                e.dtod_copy_into(&x2, &mut next, r * n_embd)?;
                            }
                        }
                        if prof {
                            e.stream().synchronize()?;
                            prof_ms[2] += seg.elapsed().as_secs_f64() * 1e3;
                        }
                        x_t = next;
                        if spec_nan_scan() {
                            // The scan MUST sit on this arm too. It used to live only on
                            // the non-fused tail, so a fused layer's poison was first
                            // reported by the next non-fused layer.
                            verify_arm_receipt(
                                "fused",
                                il,
                                pos0,
                                t,
                                cache.tp_kv[il].as_ref().map(|d| d.staged_len()),
                            );
                            nan_scan_rows(
                                e,
                                &x_t,
                                t,
                                n_embd,
                                &format!("tcol layer {il} pos0={pos0} arm=fused"),
                            )?;
                        }
                        continue;
                    }
                    if prof {
                        e.stream().synchronize()?;
                        prof_ms[0] += seg.elapsed().as_secs_f64() * 1e3;
                        seg = std::time::Instant::now();
                    }
                    let mut next = e.uninit(t * n_embd)?;
                    // Columns whose o_proj was deferred (their FFN runs after the join).
                    // A NON-deferred column's FFN must run INSIDE the column loop: the
                    // oproj-tail handoff is a single cell that the same column's
                    // residual_norm_ffn consumes before the next column's finish.
                    let mut deferred: Vec<usize> = Vec::new();
                    let mut fa2_deferred: Vec<usize> = Vec::new();
                    let ffn_col = |r: usize,
                                   mixed: &CudaSlice<f32>,
                                   next: &mut CudaSlice<f32>|
                     -> Result<(), Box<dyn std::error::Error>> {
                        let mut x_row = e.uninit(n_embd)?;
                        e.dtod_copy_view(&x_t.slice(r * n_embd..(r + 1) * n_embd), &mut x_row)?;
                        let (x1, ffn_out) =
                            self.residual_norm_ffn(e, layer, &x_row, mixed, n_embd, il, eps)?;
                        if spec_nan_scan_level() >= 2 {
                            nan_scan_rows(
                                e,
                                &ffn_out,
                                1,
                                n_embd,
                                &format!("tcol layer {il} col {r} per-column FFN out"),
                            )?;
                        }
                        let mut x2 = e.uninit(n_embd)?;
                        e.add(&x1, &ffn_out, &mut x2, n_embd)?;
                        e.dtod_copy_into(&x2, next, r * n_embd)?;
                        Ok(())
                    };
                    #[allow(clippy::needless_range_loop)]
                    // allow: the explicit index loop keeps the offset arithmetic visible and aligned with the device-side indexing
                    for r in 0..t {
                        e.dtod_copy_view(&h_t.slice(r * n_embd..(r + 1) * n_embd), &mut h_row)?;
                        let row_pos = &pos_rows[r];
                        crate::tp::set_verify_tcol(Some(r));
                        if fa2_layer {
                            crate::tp::set_spec_fa2_defer(Some(r));
                        } else if oproj_batch {
                            crate::tp::set_tcol_oproj_defer(Some(r));
                        }
                        let mixed = match &layer.mixer {
                            crate::hybrid::Mixer::Full(fa) => {
                                self.full_attn_decode(e, fa, &h_row, row_pos, pos0 + r, cache, il)
                            }
                            _ => Err("step35 verify expects full attention".into()),
                        };
                        crate::tp::set_verify_tcol(None);
                        crate::tp::set_spec_fa2_defer(None);
                        crate::tp::set_tcol_oproj_defer(None);
                        let mixed = mixed?;
                        if fa2_layer && crate::tp::take_spec_fa2_stashed() {
                            fa2_deferred.push(r);
                        } else if oproj_batch && crate::tp::take_tcol_oproj_stashed() {
                            deferred.push(r);
                        } else {
                            if spec_nan_scan_level() >= 2 {
                                let cols = mixed.len();
                                nan_scan_rows(
                                    e,
                                    &mixed,
                                    1,
                                    cols,
                                    &format!("tcol layer {il} col {r} per-column ATTN out"),
                                )?;
                            }
                            ffn_col(r, &mixed, &mut next)?;
                        }
                    }
                    if !fa2_deferred.is_empty() && fa2_deferred.len() != t {
                        // The precheck guarantees both columns stash or neither; a strict
                        // subset means a column's output was never produced anywhere.
                        return Err("spec fa2 stash engaged for a subset of columns".into());
                    }
                    if prof {
                        e.stream().synchronize()?;
                        prof_ms[1] += seg.elapsed().as_secs_f64() * 1e3;
                        seg = std::time::Instant::now();
                    }
                    if !fa2_deferred.is_empty() {
                        deferred = fa2_deferred;
                    }
                    if !deferred.is_empty() {
                        let mixed_t = if fa2_layer {
                            self.step35_verify_fa_rows_join(e, il, cache, pos0, t)?
                        } else {
                            self.step35_verify_oproj_tcol(e, il, t)?
                        };
                        let o_out = mixed_t.len() / t;
                        if spec_nan_scan_level() >= 2 {
                            nan_scan_rows(
                                e,
                                &mixed_t,
                                t,
                                o_out,
                                &format!("tcol layer {il} JOINED attn over deferred cols"),
                            )?;
                        }
                        // Batched t=2 residual+MoE: one t-grid add_rms_norm (per-row
                        // program == t=1; bit-identical to the oproj-tail join per the
                        // M2 verbatim-program contract) feeding the two-column routed
                        // sweep. Ineligible layers (dense FFN, non-nvfp4) fall through
                        // to the per-column body.
                        {
                            for &r in &deferred {
                                e.dtod_copy_view(
                                    &mixed_t.slice(r * o_out..(r + 1) * o_out),
                                    &mut mixed_row,
                                )?;
                                ffn_col(r, &mixed_row, &mut next)?;
                            }
                        }
                    }
                    if prof {
                        e.stream().synchronize()?;
                        prof_ms[2] += seg.elapsed().as_secs_f64() * 1e3;
                    }
                    x_t = next;
                    if spec_nan_scan() {
                        verify_arm_receipt(
                            if fa2_layer { "join" } else { "percol" },
                            il,
                            pos0,
                            t,
                            cache.tp_kv[il].as_ref().map(|d| d.staged_len()),
                        );
                        nan_scan_rows(
                            e,
                            &x_t,
                            t,
                            n_embd,
                            &format!(
                                "tcol layer {il} pos0={pos0} arm={}",
                                if fa2_layer { "join" } else { "percol" }
                            ),
                        )?;
                    }
                }
                if prof {
                    eprintln!(
                        "[tcol-prof] t={t} norm+qkv={:.3}ms attn={:.3}ms ffn={:.3}ms",
                        prof_ms[0], prof_ms[1], prof_ms[2]
                    );
                }
                if ok {
                    return Ok(x_t);
                }
                // fall through to the row-outer walk on ineligible layers
                x = x_t;
            }
            let mut next = e.uninit(t * n_embd)?;
            let scan = spec_nan_scan();
            for r in 0..t {
                let mut row = e.uninit(n_embd)?;
                e.dtod_copy_view(&x.slice(r * n_embd..(r + 1) * n_embd), &mut row)?;
                let row_pos = e.htod_i32(&[(pos0 + r) as i32])?;
                let out = if scan {
                    // Diagnostic arm: the same range walked one layer at a time so the first
                    // poisoned layer names itself. `decode_layers_eager(lo, hi)` is range-scoped
                    // and executes its trailing residual add, so a per-layer chain is the same
                    // program with the cross-layer add+norm fusion unrolled.
                    nan_scan_rows(
                        e,
                        &row,
                        1,
                        n_embd,
                        &format!("embed row r={r} pos={}", pos0 + r),
                    )?;
                    let mut acc = row;
                    for il in lo..hi {
                        acc = self.decode_layers_eager(
                            e,
                            acc,
                            il,
                            il + 1,
                            &row_pos,
                            pos0 + r,
                            cache,
                        )?;
                        nan_scan_rows(
                            e,
                            &acc,
                            1,
                            n_embd,
                            &format!("row-outer layer {il} r={r} pos={}", pos0 + r),
                        )?;
                    }
                    acc
                } else {
                    self.decode_layers_eager(e, row, lo, hi, &row_pos, pos0 + r, cache)?
                };
                e.dtod_copy_into(&out, &mut next, r * n_embd)?;
            }
            // dflash taps are NOT produced on this arm (they need per-layer hiddens the
            // row-outer walk does not materialize); the door is a step37 MTP bring-up
            // surface where taps are unused.
            return Ok(next);
        }
        let mut ph_last = std::time::Instant::now();
        for il in lo..hi {
            let mut next = e.uninit(t * n_embd)?;
            for r in 0..t {
                let mut row = e.uninit(n_embd)?;
                e.dtod_copy_view(&x.slice(r * n_embd..(r + 1) * n_embd), &mut row)?;
                // The caller owns this verify's position. During controller overlap, cache.pos
                // still describes generation N while this stage-0 walk belongs to N+1.
                let row_pos = e.htod_i32(&[(pos0 + r) as i32])?;
                let mut one = [&mut *cache];
                let out = self.step35_decode_batch_layers(
                    e,
                    row,
                    &mut one,
                    &[(pos0 + r) as i32],
                    &row_pos,
                    il,
                    il + 1,
                    &mut ph_last,
                )?;
                e.dtod_copy_into(&out, &mut next, r * n_embd)?;
            }
            self.dflash_tap(e, cache, il, &next, t)?;
            x = next;
            if spec_nan_scan() {
                nan_scan_rows(e, &x, t, n_embd, &format!("batch-layer {il} pos0={pos0}"))?;
            }
        }
        Ok(x)
    }

    /// DSpark drafter verify (lane/dspark-q38-recover): one t-row forward through the
    /// SERVING-CLASS verify funnel (`decode_step_t_core_stream` — the same numeric class
    /// MTP verify rides, GDN state advanced in place), returning per-row argmax tokens.
    /// Advances `cache.pos += t`; the caller owns snapshot/rollback (block acceptance is
    /// prefix-keep, not all-or-nothing).
    pub(crate) fn dspark_verify_t_am(
        &self,
        e: &Engine,
        tokens: &[u32],
        pos0: usize,
        cache: &mut Cache,
    ) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
        let (logits, _hn) = self.decode_step_t_core_stream(
            e, tokens, pos0, cache, None, None, None, None, None, None,
        )?;
        let t = tokens.len();
        let v = self.output.out_features();
        let mut am_d = e.stream().alloc_zeros::<u32>(t)?;
        for r in 0..t {
            e.argmax_token_device_col(&logits, r, v, &mut am_d, r)?;
        }
        e.dtoh_u32(&am_d)
    }

    /// DSpark verify returning the RAW verify logits [t, n_vocab] (device-resident) instead
    /// of per-row argmaxes — the sampled-admission arm's input (rejection-sampling accept
    /// gathers filtered p from these columns; lane/dspark-sampled-admission-20260820). Same
    /// forward as `dspark_verify_t_am`; the greedy arm keeps its argmax wrapper untouched.
    pub(crate) fn dspark_verify_t_logits(
        &self,
        e: &Engine,
        tokens: &[u32],
        pos0: usize,
        cache: &mut Cache,
    ) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        let (logits, _hn) = self.decode_step_t_core_stream(
            e, tokens, pos0, cache, None, None, None, None, None, None,
        )?;
        Ok(logits)
    }

    /// DSpark verify with the MTP column-stash armed: identical forward to
    /// `dspark_verify_t_am`, but fills a `VerifyCkpt` so a partial accept can restore
    /// column state directly (`dspark_commit_prefix`) instead of snapshot-replay.
    /// The ckpt type is opaque outside spec.rs (newtype) — dflash.rs threads it through.
    pub(crate) fn dspark_verify_t_am_ckpt(
        &self,
        e: &Engine,
        tokens: &[u32],
        pos0: usize,
        cache: &mut Cache,
    ) -> Result<(Vec<u32>, DsparkVerifyCkpt), Box<dyn std::error::Error>> {
        let mut ck = VerifyCkpt::new(self.layers.len());
        let (logits, _hn) = self.decode_step_t_core_stream(
            e,
            tokens,
            pos0,
            cache,
            None,
            Some(&mut ck),
            None,
            None,
            None,
            None,
        )?;
        let t = tokens.len();
        let v = self.output.out_features();
        let mut am_d = e.stream().alloc_zeros::<u32>(t)?;
        for r in 0..t {
            e.argmax_token_device_col(&logits, r, v, &mut am_d, r)?;
        }
        Ok((e.dtoh_u32(&am_d)?, DsparkVerifyCkpt(ck)))
    }

    /// Engine-bundle slice 2: `dspark_verify_t_am_ckpt` with DEVICE tokens and NO readback.
    /// The verify tokens are the round's `chain_d` (cand layout: [anchor, drafts...]); the
    /// embed gathers its first `t` entries on-device (`embed_gather_u32_t` — bit-identical
    /// rows to the host arm), so the host never blocks on the draft chain before dispatching
    /// verify. Returns the device per-row argmax buffer; the caller merges its readback with
    /// the chain's into ONE sync. Forward, ckpt fill and argmax walk are `_ckpt` verbatim.
    #[allow(clippy::too_many_arguments)] // allow: the parameter list mirrors the kernel/FFI/call contract; bundling into a struct is a refactor, not a lint fix
    pub(crate) fn dspark_verify_t_am_ckpt_dev(
        &self,
        e: &Engine,
        vtok: &CudaSlice<u32>,
        t: usize,
        pos0: usize,
        cache: &mut Cache,
        embd_dev: (&CudaSlice<u8>, i32, usize),
        graphs: Option<&mut DsparkVerifyGraphs>,
    ) -> Result<(CudaSlice<u32>, DsparkVerifyCkpt), Box<dyn std::error::Error>> {
        debug_assert!(
            vtok.len() >= t,
            "verify window exceeds the device token buffer"
        );
        // The slab flag is a per-round statement: clear it here so a verify that never
        // reaches the graphs door (rowwise env, a non-tparallel arm) cannot leave a
        // stale `true` steering the commit at slabs the round never wrote.
        let mut graphs = graphs;
        if let Some(g) = graphs.as_deref_mut() {
            g.round_slab = false;
        }
        let mut ck = VerifyCkpt::new(self.layers.len());
        // Dummy host tokens size the funnel; the embed reads `vtok` (the round-stream
        // arm's established pattern — spec.rs stream-mode verify does the same).
        let dummy = vec![0u32; t];
        let (logits, _hn) = self.decode_step_t_core_stream(
            e,
            &dummy,
            pos0,
            cache,
            Some(embd_dev),
            Some(&mut ck),
            None,
            None,
            Some(vtok),
            graphs,
        )?;
        let v = self.output.out_features();
        let mut am_d = e.stream().alloc_zeros::<u32>(t)?;
        for r in 0..t {
            e.argmax_token_device_col(&logits, r, v, &mut am_d, r)?;
        }
        Ok((am_d, DsparkVerifyCkpt(ck)))
    }

    /// Ckpt-armed twin of [`Self::dspark_verify_t_logits`] (sampled-admission arm).
    pub(crate) fn dspark_verify_t_logits_ckpt(
        &self,
        e: &Engine,
        tokens: &[u32],
        pos0: usize,
        cache: &mut Cache,
    ) -> Result<(CudaSlice<f32>, DsparkVerifyCkpt), Box<dyn std::error::Error>> {
        let mut ck = VerifyCkpt::new(self.layers.len());
        let (logits, _hn) = self.decode_step_t_core_stream(
            e,
            tokens,
            pos0,
            cache,
            None,
            Some(&mut ck),
            None,
            None,
            None,
            None,
        )?;
        Ok((logits, DsparkVerifyCkpt(ck)))
    }

    /// Restore the round to `keep` accepted columns from the verify stash: KV lens and
    /// pos from the pre-verify snapshot + keep, GDN conv/ssm from the stashed column
    /// state — no replay forward. The exact `commit_verified_prefix` the MTP path ships.
    pub(crate) fn dspark_commit_prefix(
        &self,
        e: &Engine,
        cache: &mut Cache,
        snap: &crate::cache::CacheSnapshot,
        ckpt: &DsparkVerifyCkpt,
        keep: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.commit_verified_prefix(e, cache, snap, &ckpt.0, keep, false, None)
    }

    /// Slice-3 commit twin: restore to `keep` accepted columns when the round's linear
    /// column stash lives in the graphs ctx's persistent slabs (`DsparkVerifyGraphs`) —
    /// the cols arm's exact semantics (KV lens + pos from the snapshot, GDN conv/ssm
    /// from the stash of column keep-1), slab-addressed and batched into two copy
    /// launches. `MEMRA_STATE_COPY_BATCH=0` falls back to per-layer view copies.
    pub(crate) fn dspark_commit_prefix_slab(
        &self,
        e: &Engine,
        cache: &mut Cache,
        snap: &crate::cache::CacheSnapshot,
        ctx: &DsparkVerifyGraphs,
        keep: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use cudarc::driver::DevicePtr;
        debug_assert!(keep >= 1, "keep==0 rounds take the legacy rollback");
        let mut conv_src: Vec<u64> = Vec::new();
        let mut ssm_src: Vec<u64> = Vec::new();
        let mut conv_dst: Vec<u64> = Vec::new();
        let mut ssm_dst: Vec<u64> = Vec::new();
        for il in 0..self.layers.len() {
            if let (Some(kvl), Some(saved)) = (cache.kv[il].as_mut(), snap.kv_len[il]) {
                kvl.len = saved + keep;
                e.set_i32_one(&mut kvl.len_d, kvl.len as i32)?;
            }
            if let Some(rl) = cache.recur[il].as_ref() {
                let (pc, ps, _cw, _sw) = ctx
                    .slab_row(e, il, keep - 1)
                    .ok_or("slab commit: linear layer missing from the graphs ctx")?;
                conv_src.push(pc);
                ssm_src.push(ps);
                let st = &e.gpu.stream();
                let (dc, _g0) = rl.conv_state.device_ptr(st);
                let (ds, _g1) = rl.ssm_state.device_ptr(st);
                conv_dst.push(dc);
                ssm_dst.push(ds);
            }
        }
        let n = conv_src.len();
        if n > 0 {
            if state_copy_batch_on() {
                let mut tt = vec![0u64; 2 * n];
                tt[..n].copy_from_slice(&conv_src);
                tt[n..].copy_from_slice(&conv_dst);
                let ct = e.htod_u64(&tt)?;
                tt[..n].copy_from_slice(&ssm_src);
                tt[n..].copy_from_slice(&ssm_dst);
                let st = e.htod_u64(&tt)?;
                e.copy_batch_uniform_f32(&ct, n, ctx.conv_words)?;
                e.copy_batch_uniform_f32(&st, n, ctx.ssm_words)?;
            } else {
                let (cw, sw) = (ctx.conv_words, ctx.ssm_words);
                let row = keep - 1;
                for il in 0..self.layers.len() {
                    let Some(rl) = cache.recur[il].as_mut() else {
                        continue;
                    };
                    let k = ctx.lin_pos[&il];
                    {
                        let sv = e.view(&ctx.stash_conv[k], (row + 1) * cw);
                        let win = sv.slice(row * cw..(row + 1) * cw);
                        e.copy_view_into(&mut rl.conv_state, 0, &win, cw)?;
                    }
                    {
                        let sv = e.view(&ctx.stash_ssm[k], (row + 1) * sw);
                        let win = sv.slice(row * sw..(row + 1) * sw);
                        e.copy_view_into(&mut rl.ssm_state, 0, &win, sw)?;
                    }
                }
            }
        }
        cache.pos = snap.pos + keep;
        Ok(())
    }

    /// Qwen35-family verify trunk in the live serving numeric class.
    ///
    /// Serving intentionally keeps this architecture in the generic batched program even at
    /// B=1. The older verify walk used its own mirrored dispatch and can flip near-tie argmaxes.
    ///
    /// Two arms, one numeric class:
    /// - DENSE GDN (`DenseMlp`, t<=16): `qwen35_verify_tparallel` — the weight ops (norms,
    ///   projections, FFN) hoist to m=T through the exact-tier batched kernels whose per-row
    ///   program IS the m=1 program (`matmul_pre == fused2 per (tensor,row); _bN mmvq per-row
    ///   == m=1` — decode_batch.rs v2 note), while the state ops (conv ring, gdn scan, KV
    ///   append, fa decode) stay a per-row loop running the b_n=1 serving kernels with each
    ///   row's own t_kv-driven arm pick (the straddle law: every row executes the exact
    ///   program its isolated serving step would). One weight read per layer per round
    ///   instead of T — this is what makes MTP profitable in the exact class (the per-row
    ///   walk measured verify(K+1) ~= (K+1) plain steps: 69 -> 44 tok/s served, 2026-08-15).
    /// - MoE / t>16 / `MEMRA_SPEC_VERIFY_ROWWISE=1`: the per-row replay of the authoritative
    ///   serving layer body, preserving single-session autoregressive cache order (the
    ///   correctness reference; also the rollback seam for the t-parallel arm).
    ///
    /// Bit-identity of the t-parallel arm vs the rowwise arm is gated by spec-serve-gate
    /// (zero differing logits at T=1..4, K arms) + the 8-prompt ON/OFF canary before ship.
    #[allow(clippy::too_many_arguments)]
    fn qwen35_verify_batch_layers(
        &self,
        e: &Engine,
        x: CudaSlice<f32>,
        lo: usize,
        hi: usize,
        pos0: usize,
        t: usize,
        cache: &mut Cache,
        ckpt: Option<&mut VerifyCkpt>,
        stream: Option<(&CudaSlice<u32>, &CudaSlice<i32>)>,
        graphs: Option<&mut DsparkVerifyGraphs>,
    ) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        // Qwen35Moe admitted 2026-08-20 (lane/draftcost-moe): the t-parallel arm already
        // carries the MoE FFN (`moe_ffn_il_zq8` at m=T) and the GDN per-row state loop; the
        // arch fence was a qualification gate, not a mechanism gap. Measured disease on the
        // 35B-A3B class: rowwise verify ~= 5.6 ms per drafted token (one full trunk step
        // each) — the same (K+1)-plain-steps wall the dense admission fixed on 2026-08-15.
        // Rollback seam unchanged: MEMRA_SPEC_VERIFY_ROWWISE=1.
        let rowwise = std::env::var("MEMRA_SPEC_VERIFY_ROWWISE").as_deref() == Ok("1")
            || !self.batched_serving_numeric_class()
            || t > 16;
        if rowwise {
            if stream.is_some() {
                // rowwise replays per row with host cache.pos — irreconcilable with a
                // device position counter. Burst callers must keep t <= 16 and the
                // ROWWISE env unset; refusing beats silently mispositioned rows.
                return Err("qwen35 rowwise verify has no ROUND-STREAM arm \
                            (t > 16 or MEMRA_SPEC_VERIFY_ROWWISE=1)"
                    .into());
            }
            self.qwen35_verify_rowwise(e, x, lo, hi, pos0, t, cache, ckpt)
        } else {
            self.qwen35_verify_tparallel(e, x, lo, hi, pos0, t, cache, ckpt, stream, graphs)
        }
    }

    /// The per-row correctness reference: replay each verify row through the authoritative
    /// serving layer body (`decode_batch_layers` at b_n=1). T full weight reads per layer.
    #[allow(clippy::too_many_arguments)]
    fn qwen35_verify_rowwise(
        &self,
        e: &Engine,
        mut x: CudaSlice<f32>,
        lo: usize,
        hi: usize,
        pos0: usize,
        t: usize,
        cache: &mut Cache,
        mut ckpt: Option<&mut VerifyCkpt>,
    ) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        let n_embd = self.cfg.n_embd as usize;
        let saved_pos = cache.pos;
        let mut ph_last = std::time::Instant::now();
        for il in lo..hi {
            let mut next = e.uninit(t * n_embd)?;
            let mut col_states: Option<Vec<(CudaSlice<f32>, CudaSlice<f32>)>> =
                if ckpt.is_some() && t >= 2 && matches!(self.layers[il].mixer, Mixer::Linear(_)) {
                    Some(Vec::with_capacity(t - 1))
                } else {
                    None
                };
            for r in 0..t {
                cache.pos = pos0 + r;
                let mut row = e.uninit(n_embd)?;
                e.dtod_copy_view(&x.slice(r * n_embd..(r + 1) * n_embd), &mut row)?;
                let row_pos = e.htod_i32(&[(pos0 + r) as i32])?;
                let mut one = [&mut *cache];
                let ctx = self.batch_layer_ctx(e, &one, il, il + 1)?;
                let out = match self.decode_batch_layers(
                    e,
                    row,
                    &mut one,
                    &ctx,
                    &row_pos,
                    &mut ph_last,
                ) {
                    Ok(out) => out,
                    Err(error) => {
                        cache.pos = saved_pos;
                        return Err(error);
                    }
                };
                e.dtod_copy_into(&out, &mut next, r * n_embd)?;
                if r + 1 < t
                    && let Some(states) = col_states.as_mut()
                {
                    let recur = cache.recur[il]
                        .as_ref()
                        .ok_or("Qwen35-MoE linear verify layer has no recurrent state")?;
                    states.push((
                        e.clone_dtod(&recur.conv_state)?,
                        e.clone_dtod(&recur.ssm_state)?,
                    ));
                }
            }
            if let (Some(checkpoint), Some(states)) = (ckpt.as_deref_mut(), col_states) {
                checkpoint.cols[il] = Some(states);
            }
            x = next;
        }
        cache.pos = saved_pos;
        Ok(x)
    }

    /// T-PARALLEL VERIFY IN THE SERVING NUMERIC CLASS (lane/tparallel-verify, 2026-08-15).
    ///
    /// The weight ops run ONCE per layer at m=T; the state ops run per row through the same
    /// b_n=1 serving kernels the rowwise replay uses. Per-row bit-identity rests on the two
    /// pins the serving batch tier already carries:
    ///   * `matmul_pre` / `_bN` mmvq: per-row program == m=1 program (decode_batch.rs v2 note,
    ///     kernel-check pinned) — so a [T, n_embd] projection row equals the row projected
    ///     alone;
    ///   * row-indexed norms/elementwise (`rms_norm`, `quantize_q8_1`, `add_rms_norm`,
    ///     `gated_rmsnorm[_q8_1]`, `silu_mul`, `rope_neox` with per-row positions): the T-row
    ///     launch is the per-row program (same pin the generic verify's fused norms rely on).
    ///     The sequential dependencies keep their exact serving order: the conv ring / gdn scan
    ///     chain state row -> row through the `_b` kernels at b_n=1 (ping-pong via a 6-entry
    ///     alternating pointer table, host handles swapped per row so VerifyCkpt clones the
    ///     canonical state exactly as the rowwise arm does), and each row's KV append + fa decode
    ///     picks its arm from ITS OWN t_kv (append: format-only; fa: `fa_seqs_eligible` + its own
    ///     `fa_split_keys` rung at b_n=1) — the straddle law per row, so every row executes the
    ///     program its isolated B=1 serving step would.
    ///
    /// Cost: 1 weight read per layer per round + T state micro-launches, vs the rowwise arm's
    /// T weight reads. Gated bit-identical vs the rowwise arm by spec-serve-gate + canary.
    #[allow(clippy::too_many_arguments)]
    fn qwen35_verify_tparallel(
        &self,
        e: &Engine,
        mut x: CudaSlice<f32>,
        lo: usize,
        hi: usize,
        pos0: usize,
        t: usize,
        cache: &mut Cache,
        mut ckpt: Option<&mut VerifyCkpt>,
        stream: Option<(&CudaSlice<u32>, &CudaSlice<i32>)>,
        mut graphs: Option<&mut DsparkVerifyGraphs>,
    ) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        let seqs_append =
            std::env::var("MEMRA_BATCH_APPEND").as_deref() != Ok("0") && !Engine::kv_fp8_on();
        let batch_fa_on = std::env::var("MEMRA_BATCH_FA").as_deref() != Ok("0");

        // Merge guard (v0.98 train, re-affirmed on the v0.100 train over slice 4c): the
        // ROUND-STREAM arm (lane/draftcost-moe, device position counter) and the dspark
        // verify graphs (engine-bundle slice 3 / trunk slice 4c) have no common caller —
        // stream rides the qwen35moe burst, graphs ride the dspark route. If a future
        // caller arms both, refuse loudly instead of silently dropping the graphs ctx
        // (the stream linear arm takes linear_attn_verify_t, not the graphed segment or
        // full-verify bodies).
        if stream.is_some() && graphs.is_some() {
            return Err(
                "qwen35 tparallel verify: ROUND-STREAM and dspark verify graphs \
                        cannot arm together"
                    .into(),
            );
        }
        // Engine-bundle slice 3 + slice 4c: with a graphs ctx armed, pointer tables are
        // refreshed once per verify (the gdn ping-pong moves handles; a fresh generation
        // moves the kv caches). Then:
        //  - slice 4c: when the WHOLE round rides one seqs rung (every row batchable, one
        //    split-ladder step, rung covers the round), the ENTIRE walk replays as ONE
        //    full-verify graph per (vt, rung) — linear layers through the shared
        //    `qwen35_tparallel_linear_layer` body, full-attention layers through the
        //    shared `qwen35_tparallel_fa_layer` body in graph mode.
        //  - fallback (straddle rounds, below the vec floor, partial walks): runs of
        //    consecutive LINEAR layers replay the slice-3 per-(segment, vt) graphs and
        //    the full-attention layers run eager (batched rows when eligible).
        //
        // GRAPH-LAUNCH HEADROOM GUARD (see GRAPH_LAUNCH_MIN_FREE): the dspark verify
        // graphs replay through this walk from THREE callers — the MTP spec round's vg
        // door (already dropped per round by `graph_round_ok` before it gets here), the
        // dspark one-shot, and the dspark SERVE round (default ON since v0.108). Below
        // the driver-free floor the WHOLE round takes the byte-identical eager
        // cols-ckpt walk — the same drop-the-ctx fallback the pool ceiling already
        // takes — instead of feeding cuGraphLaunch a card it segfaults on.
        if let Some(g) = graphs.as_deref_mut()
            && !graph_launch_headroom_ok(e)
        {
            g.round_slab = false;
            graphs = None;
            static NOTED: std::sync::Once = std::sync::Once::new();
            NOTED.call_once(|| graph_replay_suspended_note("dspark-vg"));
        }
        if let Some(g) = graphs.as_deref_mut() {
            g.refresh_tables(e, cache)?;
            g.round_slab = false;
            if let Some(rung) = g.full_rung(self, cache, lo, hi, t, seqs_append && batch_fa_on) {
                // Pool ceiling (dspark_vg_cap): an existing key always replays; a NEW
                // full capture past the ceiling falls through to the segment/eager arms.
                if g.full.contains_key(&(t, rung, hi)) || g.can_capture() {
                    let out = g.run_full(self, e, lo, hi, &x, t, pos0, rung, cache)?;
                    g.round_slab = true;
                    return Ok(out);
                }
            }
            // Round-atomic ceiling check for the segment door: if any linear run in this
            // walk would need a NEW capture past the ceiling, the whole round runs the
            // eager cols-ckpt walk (mixing slab- and cols-stashed layers in one round
            // would corrupt the commit).
            if !g.segments_ready(self, lo, hi, t) {
                graphs = None;
            }
        }
        // STREAM (2b, lane/draftcost-moe): positions come from the device round counter
        // (pos_iota / i32_copy_add) so a burst round needs no host position knowledge.
        let pos_d = match stream {
            Some((_, ctr)) => {
                let mut p = e.alloc_uninit::<i32>(t)?;
                e.pos_iota(ctr, &mut p, t)?;
                p
            }
            None => {
                let pos_host: Vec<i32> = (0..t).map(|r| (pos0 + r) as i32).collect();
                e.htod_i32(&pos_host)?
            }
        };
        // Per-row 1-element position buffers, built ONCE per verify (the append/fa wrappers
        // take owned pos slices; building these inside the layer x row loops cost 16xT H2Ds).
        // LAZY since slice 4: the batched fa/append arm never touches them — they are built
        // on the first per-row fallback layer only (stream-aware there; the stream FA arm
        // rides the dc rows kernels and never reaches the fallback).
        let mut pos_rows: Option<Vec<CudaSlice<i32>>> = None;
        let mut il = lo;
        while il < hi {
            if graphs.is_some() && matches!(self.layers[il].mixer, Mixer::Linear(_)) {
                let mut end = il;
                while end < hi && matches!(self.layers[end].mixer, Mixer::Linear(_)) {
                    end += 1;
                }
                let g = graphs.as_deref_mut().expect("checked above");
                x = g.run_segment(self, e, il, end, &x, t, cache)?;
                g.round_slab = true;
                il = end;
                continue;
            }
            let layer = &self.layers[il];
            if stream.is_none() && matches!(layer.mixer, Mixer::Linear(_)) {
                // Eager linear layer (no graphs ctx): the shared body, legacy cols-ckpt arm.
                // Under ROUND-STREAM the linear layers ride the fa-body match's stream arm
                // below (linear_attn_verify_t — the stream COMMIT needs its GdnStash).
                x = self.qwen35_tparallel_linear_layer(
                    e,
                    il,
                    &x,
                    t,
                    cache,
                    ckpt.as_deref_mut(),
                    None,
                    None,
                )?;
                il += 1;
                continue;
            }
            // Full-attention (or stream-Linear, or MLA-refusing) layer: the extracted
            // shared body — eager arm (fresh per-verify pos/table, exact t_kv sizing,
            // in-body len bump). The slice-4c captured full-verify graphs run the SAME
            // body in graph mode; under ROUND-STREAM the body's dc-rows / GDN stream arms
            // run (lane/draftcost-moe).
            x = self.qwen35_tparallel_fa_layer(
                e,
                il,
                &x,
                t,
                cache,
                FaLayerArgs {
                    pos_d: &pos_d,
                    pos_rows: &mut pos_rows,
                    pos0,
                    seqs_append,
                    batch_fa_on,
                    graph_cap: None,
                    stream,
                    ckpt: ckpt.as_deref_mut(),
                },
            )?;
            il += 1;
        }
        Ok(x)
    }

    /// SHARED dense-FFN body for the qwen35 t-parallel layers (trunk-kernels slice B) —
    /// ONE copy for the fa and linear layer bodies (the verify_layers extraction lesson).
    /// Dual arm (MEMRA_TK_FFN_DUAL, default on): gate+up in ONE dual launch from the
    /// pre-quantized activation with macro-scales DEFERRED into the fused SwiGLU+q8_1
    /// epilogue, then ffn_down from the fused (aq, ad) — the q27 verify chain verbatim.
    /// Every door is the bit-identical proven one: `matmul_decode_exact_dual_pre` (per
    /// (tensor,token,row) == the two singles), `silu_mul_scaled_q8_1` (y*s inline == the
    /// scale_inplace store, value-exact; fused quantize == quantize_q8_1 bytes),
    /// `matmul_decode_exact_pre` (dispatch mirror of the singles' q8_1-fast tail).
    /// Dual-refused (t outside 2..=7, non-NVFP4, layout mismatch) or seam off -> the
    /// original singles chain, byte-for-byte.
    #[allow(clippy::too_many_arguments)]
    fn qwen35_tparallel_dense_ffn(
        &self,
        e: &Engine,
        ffn_gate: &crate::model::GpuTensor,
        ffn_up: &crate::model::GpuTensor,
        ffn_down: &crate::model::GpuTensor,
        zn: &CudaSlice<f32>,
        t: usize,
        n_embd: usize,
    ) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        let n_ff = ffn_gate.out_features();
        let (zq, zd) = e.quantize_q8_1(zn, t, n_embd)?;
        if Engine::tk_ffn_dual_on()
            && let Some(((g, gs), (u, us))) =
                e.matmul_decode_exact_dual_pre(ffn_gate, ffn_up, &zq, &zd, t)?
        {
            if e.uses_q8_1_fast(ffn_down) {
                let (aq, ad) = e.silu_mul_scaled_q8_1(&g, &u, gs, us, t * n_ff)?;
                return e.matmul_decode_exact_pre(ffn_down, &aq, &ad, t);
            }
            let mut act = e.uninit(t * n_ff)?;
            e.silu_mul_scaled(&g, &u, gs, us, &mut act, t * n_ff)?;
            let (aq, ad) = e.quantize_q8_1(&act, t, n_ff)?;
            return e.matmul_pre(ffn_down, &aq, &ad, &act, t);
        }
        // v1 singles chain (seam off or dual-refused) — the pre-slice-B body verbatim.
        let g = e.matmul_pre(ffn_gate, &zq, &zd, zn, t)?;
        let u = e.matmul_pre(ffn_up, &zq, &zd, zn, t)?;
        let mut act = e.uninit(t * n_ff)?;
        e.silu_mul(&g, &u, &mut act, t * n_ff)?;
        let (aq, ad) = e.quantize_q8_1(&act, t, n_ff)?;
        e.matmul_pre(ffn_down, &aq, &ad, &act, t)
    }

    /// ONE t-parallel FULL-ATTENTION layer (attn_norm + fa mixer + post_attn_norm + FFN +
    /// tap) — extracted from the walk exactly like `qwen35_tparallel_linear_layer` so the
    /// eager walk and the slice-4c captured full-verify graphs execute the SAME body (a
    /// second copy is how dispatch mirrors drift — the verify_layers extraction lesson).
    ///
    /// `args.graph_cap = Some((table, off, rung_end))` is the captured-graph mode:
    /// - kv base-pointer pairs come from the ctx-owned persistent table at `off` (a fresh
    ///   generation's cache lands at new addresses that only the per-verify table refresh
    ///   knows — the slice-3 baked-address lesson);
    /// - the seqs twins size partials/grid at `rung_end` and pin `split_keys` to the
    ///   rung's ladder value: `n_splits_max` is pure stride, splits >= ns_eff write the
    ///   EMPTY partial the combine never reads, and every per-row T_kv derives in-kernel
    ///   from `pos_seq[z]` — so one captured launch replays bit-identically for every
    ///   round whose rows all sit inside the rung;
    /// - the host len bump moves to the replay caller (captured host code does not
    ///   re-run at replay).
    ///   Graph mode REFUSES any round the batched arm cannot take: the per-row fallback
    ///   host-branches on t_kv and must never be captured.
    #[allow(clippy::too_many_arguments)]
    fn qwen35_tparallel_fa_layer(
        &self,
        e: &Engine,
        il: usize,
        x: &CudaSlice<f32>,
        t: usize,
        cache: &mut Cache,
        args: FaLayerArgs<'_>,
    ) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        use cudarc::driver::DevicePtr;
        let cfg = &self.cfg;
        let n_embd = cfg.n_embd as usize;
        let eps = cfg.rms_eps;
        let head_dim_global = cfg.head_dim_k as usize;
        let layer = &self.layers[il];
        let FaLayerArgs {
            pos_d,
            pos_rows,
            pos0,
            seqs_append,
            batch_fa_on,
            graph_cap,
            stream,
            ckpt,
        } = args;

        // ---- attn_norm + q8_1 quantize at m=T (row-indexed == per-row) ----
        let anorm = layer.attn_norm.float_data();
        let mut xn = e.uninit(t * n_embd)?;
        e.rms_norm(x, anorm, &mut xn, n_embd, t, eps)?;
        let (hq, hd) = e.quantize_q8_1(&xn, t, n_embd)?;

        let mixed: CudaSlice<f32> = match &layer.mixer {
            Mixer::Mla(_) => crate::hybrid::mla_path_unimplemented("tensor-parallel attention"),
            Mixer::Kda(_) => crate::hybrid::kda_path_unimplemented("T-parallel attention"),
            // STREAM ARM (2b, lane/draftcost-moe): under a device position counter the
            // per-row serving-kernel chain cannot run (host state swaps keyed on host
            // row index are fine, but the stream COMMIT needs the GdnStash for its _dc
            // rebuild — the per-row chain only produces per-column clones). GDN rides
            // `linear_attn_verify_t`: batched q8_1-class projections, stash-producing,
            // and its one-scan recurrence is pinned bit-identical to T chained T=1
            // steps (its header + kernel-check). Position-independent, so no counter
            // plumbing is needed. Guards mirror the generic call site exactly.
            Mixer::Linear(la) if stream.is_some() => {
                if !(t >= 3 || (t == 2 && spec_m2()))
                    || !self.mixer_in_q8_1_fast(e, &layer.mixer)
                    || !e.uses_q8_1_fast(&la.ssm_out)
                {
                    return Err("qwen35 stream verify: GDN batched arm requires t>=3 \
                                (or MEMRA_SPEC_M2 at t=2) and q8_1-fast projections"
                        .into());
                }
                let want = ckpt.is_some();
                let (out, stash) =
                    self.linear_attn_verify_t(e, la, &xn, Some((&hq, &hd)), t, cache, il, want)?;
                if let (Some(ck), Some(st)) = (ckpt, stash) {
                    ck.gdn[il] = Some(st);
                }
                out
            }
            Mixer::Linear(_) => {
                unreachable!("linear layers ride qwen35_tparallel_linear_layer")
            }
            Mixer::Full(fa) => {
                let geometry = cfg.full_attention_geometry_at(il as u32);
                let n_head = geometry.n_head as usize;
                let n_head_kv = geometry.n_head_kv as usize;
                let head_dim = geometry.head_dim_k as usize;
                let rope_dims = geometry.n_rot as usize;
                let rope_base = geometry.rope_base;
                let scale = geometry.attention_scale();
                // Batched projections: one weight read serves all T rows.
                // GROUP-3 twin (trunk-kernels slice D): q/k/v in ONE launch — the group4
                // kernel with n3=0, bit-identical per (tensor, token, row) to the three
                // singles; refused or MEMRA_TK_FA_GROUP=0 -> singles byte-for-byte.
                let (qf, mut k, v) = match e.matmul_decode_exact_group3_pre(
                    [&fa.wq, &fa.wk, &fa.wv],
                    &hq,
                    &hd,
                    t,
                )? {
                    Some(mut g3) => {
                        let v = g3.pop().unwrap();
                        let k = g3.pop().unwrap();
                        let qf = g3.pop().unwrap();
                        (qf, k, v)
                    }
                    None => (
                        e.matmul_pre(&fa.wq, &hq, &hd, &xn, t)?,
                        e.matmul_pre(&fa.wk, &hq, &hd, &xn, t)?,
                        e.matmul_pre(&fa.wv, &hq, &hd, &xn, t)?,
                    ),
                };
                let gated =
                    geometry.attention_gate == memra_gguf::config::AttentionGateKind::FusedQ;
                let (mut q, gate) = if gated {
                    let mut qs = e.uninit(t * n_head * head_dim)?;
                    let mut gs = e.uninit(t * n_head * head_dim)?;
                    e.q_gate_split(&qf, &mut qs, &mut gs, head_dim, n_head, t)?;
                    (qs, Some(gs))
                } else {
                    (qf, None)
                };
                let mut qn = e.uninit(t * n_head * head_dim)?;
                e.rms_norm(
                    &q,
                    fa.q_norm.float_data(),
                    &mut qn,
                    head_dim,
                    t * n_head,
                    eps,
                )?;
                q = qn;
                let mut kn = e.uninit(t * n_head_kv * head_dim)?;
                e.rms_norm(
                    &k,
                    fa.k_norm.float_data(),
                    &mut kn,
                    head_dim,
                    t * n_head_kv,
                    eps,
                )?;
                k = kn;
                e.rope_neox(
                    &mut q, pos_d, head_dim, rope_dims, n_head, t, rope_base, 1.0,
                )?;
                e.rope_neox(
                    &mut k, pos_d, head_dim, rope_dims, n_head_kv, t, rope_base, 1.0,
                )?;

                // Per-row append + attend: row r sees rows 0..r in KV (causal within the
                // draft), each through the b_n=1 serving kernels at its own t_kv.
                let q_dim = n_head * head_dim;
                let kv_dim = n_head_kv * head_dim;
                let mut attn = e.uninit(t * q_dim)?;
                let (kdk, kdv, ktb, vtb, len0, kv_local) = {
                    let kvl = cache.kv[il].as_ref().unwrap();
                    // [2T] interleaved k,v base pointers: entry pair z serves row z of
                    // the batched twins; the per-row fallback reads pair 0 (same cache
                    // for every row of one layer). Graph mode reads the ctx table.
                    let local: Option<CudaSlice<u64>> = match graph_cap {
                        Some(_) => None,
                        None => {
                            let s = &e.gpu.stream();
                            let (pk, _g) = kvl.k.device_ptr(s);
                            let (pv, _g2) = kvl.v.device_ptr(s);
                            let mut tbl = Vec::with_capacity(2 * t);
                            for _ in 0..t {
                                tbl.push(pk);
                                tbl.push(pv);
                            }
                            Some(e.htod_u64(&tbl)?)
                        }
                    };
                    (
                        kvl.kv_dim_k,
                        kvl.kv_dim_v,
                        kvl.k_tok_bytes,
                        kvl.v_tok_bytes,
                        kvl.len,
                        local,
                    )
                };
                let (kv_tbl, kv_off): (&CudaSlice<u64>, usize) = match graph_cap {
                    Some((tb, off, _)) => (tb, off),
                    None => (kv_local.as_ref().expect("built above"), 0),
                };
                // Slice 4 (fa/append rows — see dspark_fa_rows_on): the whole per-row
                // section batches into the z-batched serving twins when every row of
                // this round takes the v4-seqs arm on ONE fa_split_keys rung. Both
                // guards are evaluated at the round's FIRST and LAST t_kv — the
                // eligibility window (vec floor .. v4 max) and each split-ladder rung
                // are intervals in t_kv, so ends-inside means all-inside (the straddle
                // law). Appending all T rows before any attend is read-equivalent to
                // the interleaved order: row r's walk reads keys 0..len0+r only, and
                // rows > r land at slots it never touches; every written cache row is
                // the per-token appender's exact warp program (kernel-check pinned).
                let t_kv_first = len0 + 1;
                let t_kv_last = len0 + t;
                let rows_batched = t >= 2
                    && seqs_append
                    && batch_fa_on
                    && dspark_fa_rows_on()
                    // the z-batched twins read stacked rows at the CACHE's kv dims;
                    // the projection stack is [T, n_head_kv*head_dim] — they must be
                    // the same stride or row z misaligns (true for this family; the
                    // guard keeps any asymmetric-kv model on the per-row loop).
                    && kdk == kv_dim
                    && kdv == kv_dim
                    && crate::fa_seqs_eligible(t_kv_first, head_dim_global)
                    && crate::fa_seqs_eligible(t_kv_last, head_dim_global)
                    && crate::fa_split_keys(t_kv_first, cfg.n_head_kv as usize)
                        == crate::fa_split_keys(t_kv_last, cfg.n_head_kv as usize);
                // Sizing: eager = exact round bound; graph mode = the rung end (stride +
                // grid only — bytes proven equal above). Capture-time invariants refuse
                // loudly rather than bake a divergent body.
                let (size_kv_max, sp) = match graph_cap {
                    Some((_, _, rung)) => {
                        if !rows_batched {
                            return Err(format!(
                                "fa graph capture: layer {il} round is not batchable \
                                 (t_kv {t_kv_first}..{t_kv_last}) — the per-row fallback \
                                 must never be captured"
                            )
                            .into());
                        }
                        let sp_r = crate::fa_split_keys(rung, cfg.n_head_kv as usize);
                        if t_kv_last > rung
                            || sp_r != crate::fa_split_keys(t_kv_last, cfg.n_head_kv as usize)
                        {
                            return Err(format!(
                                "fa graph capture: rung {rung} does not cover round \
                                 t_kv {t_kv_first}..{t_kv_last} on one split ladder step"
                            )
                            .into());
                        }
                        (rung, sp_r)
                    }
                    None => (
                        t_kv_last,
                        crate::fa_split_keys(t_kv_last, cfg.n_head_kv as usize),
                    ),
                };
                if let Some((_, ctr)) = stream {
                    // STREAM ARM (2b): one batched dc append + the multi-row dc attention
                    // — the generic stream arm's exact shape (rows kernels are pinned
                    // byte-identical to the per-row programs by kernel-check). Host len
                    // stays a stale lower bound; the burst drain reconciles it.
                    let kvl = cache.kv[il].as_mut().unwrap();
                    e.append_kv_quantized_rows_dc(
                        &k,
                        &v,
                        &mut kvl.k,
                        &mut kvl.v,
                        ctr,
                        t,
                        kdk,
                        kdv,
                        ktb,
                        vtb,
                        Engine::kv_fp8_on(),
                    )?;
                    let upper = (kvl.len + t + 64).min(cache.max_ctx);
                    let k_view = e.view_u8(&kvl.k, upper * ktb);
                    let v_view = e.view_u8(&kvl.v, upper * vtb);
                    e.fa_decode_rows_dc(
                        &q, &k_view, &v_view, &mut attn, head_dim, n_head, n_head_kv, ctr, upper,
                        t, scale, ktb, vtb, 0, false,
                    )?;
                } else if rows_batched {
                    e.append_kv_quantized_seqs(
                        &k,
                        &v,
                        &kv_tbl.slice(kv_off..kv_off + 2 * t),
                        pos_d,
                        t,
                        kdk,
                        kdv,
                        ktb,
                        vtb,
                    )?;
                    if graph_cap.is_none() {
                        cache.kv[il].as_mut().unwrap().len += t;
                    }
                    e.fa_decode_batch_seqs_v4(
                        &q,
                        &kv_tbl.slice(kv_off..kv_off + 2 * t),
                        pos_d,
                        &mut attn,
                        head_dim,
                        n_head,
                        n_head_kv,
                        t,
                        size_kv_max,
                        scale,
                        sp,
                        ktb,
                        vtb,
                    )?;
                } else {
                    if pos_rows.is_none() {
                        // Stream-aware for symmetry with pos_d (the stream FA arm rides
                        // the dc rows kernels above and never reaches this fallback).
                        *pos_rows = Some(match stream {
                            Some((_, ctr)) => (0..t)
                                .map(|r| {
                                    let mut b = e.alloc_uninit::<i32>(1)?;
                                    e.i32_copy_add(ctr, &mut b, r as i32)?;
                                    Ok(b)
                                })
                                .collect::<Result<_, Box<dyn std::error::Error>>>()?,
                            None => (0..t)
                                .map(|r| e.htod_i32(&[(pos0 + r) as i32]))
                                .collect::<Result<_, _>>()?,
                        });
                    }
                    let pos_rows = pos_rows.as_ref().unwrap();
                    #[allow(clippy::needless_range_loop)]
                    // allow: the explicit index loop keeps the offset arithmetic visible and aligned with the device-side indexing
                    for r in 0..t {
                        // Owned per-row scratch: the b_n=1 kernels take packed batch buffers
                        // whose row 0 is this row (arithmetic-free materialization copies,
                        // same as decode's per-seq fallback arm).
                        let mut k_row = e.uninit(kv_dim)?;
                        e.dtod_copy_view(&k.slice(r * kv_dim..(r + 1) * kv_dim), &mut k_row)?;
                        let mut v_row = e.uninit(kv_dim)?;
                        e.dtod_copy_view(&v.slice(r * kv_dim..(r + 1) * kv_dim), &mut v_row)?;
                        let pos_row = &pos_rows[r];
                        let kvl = cache.kv[il].as_mut().unwrap();
                        if seqs_append {
                            e.append_kv_quantized_seqs(
                                &k_row,
                                &v_row,
                                &kv_tbl.slice(kv_off..kv_off + 2),
                                pos_row,
                                1,
                                kdk,
                                kdv,
                                ktb,
                                vtb,
                            )?;
                            kvl.len += 1;
                        } else {
                            e.append_kv_quantized_view(
                                &k_row.slice(0..kv_dim),
                                &v_row.slice(0..kv_dim),
                                &mut kvl.k,
                                &mut kvl.v,
                                kvl.len,
                                kvl.kv_dim_k,
                                kvl.kv_dim_v,
                                kvl.k_tok_bytes,
                                kvl.v_tok_bytes,
                                Engine::kv_fp8_on(),
                            )?;
                            kvl.len += 1;
                        }
                        let t_kv = kvl.len;
                        let mut q_row = e.uninit(q_dim)?;
                        e.dtod_copy_view(&q.slice(r * q_dim..(r + 1) * q_dim), &mut q_row)?;
                        let mut a_row = e.uninit(q_dim)?;
                        if batch_fa_on && crate::fa_seqs_eligible(t_kv, head_dim_global) {
                            let sp0_r = crate::fa_split_keys(t_kv, cfg.n_head_kv as usize);
                            e.fa_decode_batch_seqs_v4(
                                &q_row,
                                &kv_tbl.slice(kv_off..kv_off + 2),
                                pos_row,
                                &mut a_row,
                                head_dim,
                                n_head,
                                n_head_kv,
                                1,
                                t_kv,
                                scale,
                                sp0_r,
                                ktb,
                                vtb,
                            )?;
                        } else {
                            let k_view = e.view_u8(&kvl.k, t_kv * kvl.k_tok_bytes);
                            let v_view = e.view_u8(&kvl.v, t_kv * kvl.v_tok_bytes);
                            let mut a_view = a_row.slice_mut(0..q_dim);
                            e.fa_decode_kvmod_view(
                                &q_row.slice(0..q_dim),
                                &k_view,
                                &v_view,
                                &mut a_view,
                                head_dim,
                                n_head,
                                n_head_kv,
                                t_kv,
                                scale,
                                kvl.k_tok_bytes,
                                kvl.v_tok_bytes,
                                Engine::kv_fp8_on(),
                            )?;
                        }
                        e.dtod_copy_into(&a_row, &mut attn, r * q_dim)?;
                    }
                }

                // Output gate (element-wise) + o-proj at m=T.
                let attn_g = match &gate {
                    Some(g) => {
                        let n = t * q_dim;
                        let mut gsig = e.uninit(n)?;
                        e.sigmoid(g, &mut gsig, n)?;
                        let mut ag = e.uninit(n)?;
                        e.mul(&attn, &gsig, &mut ag, n)?;
                        ag
                    }
                    None => attn,
                };
                e.matmul(&fa.wo, &attn_g, t)?
            }
        };

        // ---- residual add + post_attn_norm + FFN at m=T (serving dispatch verbatim) ----
        let pnorm = layer.post_attn_norm.float_data();
        let mut x1 = e.uninit(t * n_embd)?;
        let mut zn = e.uninit(t * n_embd)?;
        e.add_rms_norm(x, &mixed, pnorm, &mut x1, &mut zn, n_embd, t, eps)?;
        let ffn_out = match &layer.ffn {
            crate::hybrid::Ffn::Dense {
                ffn_gate,
                ffn_up,
                ffn_down,
            } => {
                assert!(
                    self.cfg.m3.is_none(),
                    "qwen35 t-parallel verify: M3 swigluoai FFN not yet batched"
                );
                self.qwen35_tparallel_dense_ffn(e, ffn_gate, ffn_up, ffn_down, &zn, t, n_embd)?
            }
            crate::hybrid::Ffn::Moe(m) => self.moe_ffn_il_zq8(e, m, &zn, None, t, il as u16)?,
        };
        let mut x2 = e.uninit(t * n_embd)?;
        e.add(&x1, &ffn_out, &mut x2, t * n_embd)?;
        // dspark drafter tap (no-op when no sink armed): post-layer residual verify rows
        self.dflash_tap(e, cache, il, &x2, t)?;
        Ok(x2)
    }

    /// ONE t-parallel LINEAR layer (attn_norm + gdn mixer + post_attn_norm + FFN + tap) —
    /// the exact body the old in-loop Linear arm ran, extracted so the eager walk and the
    /// slice-3 captured segments execute the SAME code (a second copy is how dispatch
    /// mirrors drift — the verify_layers extraction lesson). Two deliberate changes, both
    /// bit-identical by construction:
    /// - the gdn ping-pong host swap moves from per-row to ONE end-of-body swap (t odd):
    ///   the device sequence is driven entirely by the 6-entry pointer table, which
    ///   already encodes both parities; the ckpt stash reads name row r's out buffer
    ///   directly (r even -> alt handle, odd -> canonical) — the same physical bytes the
    ///   legacy post-swap clone read.
    /// - `stash` (slice-3 ctx): persistent per-layer slabs written by copy_into instead of
    ///   per-row clone_dtod allocs — same bytes, capture-legal (no per-round host objects).
    ///   `table_src` = (persistent pointer table, offset) when the ctx owns the tables;
    ///   None builds the per-verify table exactly as before.
    #[allow(clippy::too_many_arguments)]
    fn qwen35_tparallel_linear_layer(
        &self,
        e: &Engine,
        il: usize,
        x: &CudaSlice<f32>,
        t: usize,
        cache: &mut Cache,
        ckpt: Option<&mut VerifyCkpt>,
        stash: Option<(&mut CudaSlice<f32>, &mut CudaSlice<f32>)>,
        table_src: Option<(&CudaSlice<u64>, usize)>,
    ) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        use cudarc::driver::DevicePtr;
        let cfg = &self.cfg;
        let n_embd = cfg.n_embd as usize;
        let eps = cfg.rms_eps;
        let layer = &self.layers[il];
        let Mixer::Linear(la) = &layer.mixer else {
            return Err("qwen35_tparallel_linear_layer on a non-linear layer".into());
        };
        // ---- attn_norm + q8_1 quantize at m=T (row-indexed == per-row) ----
        let anorm = layer.attn_norm.float_data();
        let mut xn = e.uninit(t * n_embd)?;
        e.rms_norm(x, anorm, &mut xn, n_embd, t, eps)?;
        let (hq, hd) = e.quantize_q8_1(&xn, t, n_embd)?;

        let geometry = la.geometry;
        let d_state = geometry.key_head_dim as usize;
        let num_k = geometry.key_heads as usize;
        let num_v = geometry.value_heads as usize;
        let d_conv = geometry.conv_kernel as usize;
        let key_dim = d_state * num_k;
        let value_dim = geometry.value_head_dim as usize * num_v;
        let conv_dim = key_dim * 2 + value_dim;
        let gdn_scale = 1.0 / (d_state as f32).sqrt();

        // ---- batched projections: one weight read for all T rows ----
        // GROUP-4 twin (trunk-kernels slice C): the whole 4-tuple in ONE launch, bit-identical
        // per (tensor, token, row) to the four singles; refused (layout/tier) or
        // MEMRA_TK_GDN_GROUP=0 -> the singles chain byte-for-byte.
        let (qkv_mixed, z, beta_raw, alpha) = match e.matmul_decode_exact_group4_pre(
            [&la.wqkv, &la.wqkv_gate, &la.ssm_beta, &la.ssm_alpha],
            &hq,
            &hd,
            t,
        )? {
            Some(mut g4) => {
                let alpha = g4.pop().unwrap();
                let beta_raw = g4.pop().unwrap();
                let z = g4.pop().unwrap();
                let qkv_mixed = g4.pop().unwrap();
                (qkv_mixed, z, beta_raw, alpha)
            }
            None => (
                e.matmul_pre(&la.wqkv, &hq, &hd, &xn, t)?,
                e.matmul_pre(&la.wqkv_gate, &hq, &hd, &xn, t)?,
                e.matmul_pre(&la.ssm_beta, &hq, &hd, &xn, t)?,
                e.matmul_pre(&la.ssm_alpha, &hq, &hd, &xn, t)?,
            ),
        };
        let beta_w = la.ssm_beta.out_features();
        let alpha_w = la.ssm_alpha.out_features();
        let qkv_w = la.wqkv.out_features();

        // ---- per-row state chain through the b_n=1 serving kernels ----
        // 6-entry alternating pointer table expresses the ping-pong without a rebuild per
        // row: even rows scan s0 -> s1, odd rows s1 -> s0.
        let table_local: Option<CudaSlice<u64>> = match table_src {
            Some(_) => None,
            None => {
                let rl = cache.recur[il].as_ref().unwrap();
                let s = &e.gpu.stream();
                let (pc, _g0) = rl.conv_state.device_ptr(s);
                let (p0, _g1) = rl.ssm_state.device_ptr(s);
                let (p1, _g2) = rl.ssm_state_alt.device_ptr(s);
                Some(e.htod_u64(&[pc, p0, p1, pc, p1, p0])?)
            }
        };
        let (table, toff): (&CudaSlice<u64>, usize) = match table_src {
            Some((tb, off)) => (tb, off),
            None => (table_local.as_ref().unwrap(), 0),
        };
        let mut o_all = e.uninit(t * value_dim)?;
        let mut col_states: Option<Vec<(CudaSlice<f32>, CudaSlice<f32>)>> =
            if ckpt.is_some() && stash.is_none() && t >= 2 {
                Some(Vec::with_capacity(t - 1))
            } else {
                None
            };
        let mut stash = stash;
        // Per-row scratch reused across rows (uninit is cheap but not free at
        // 48 layers x T rows); row inputs/outputs pass as VIEWS into the packed
        // [T, ...] buffers — zero arithmetic-free copies in this loop.
        let mut conv_out = e.uninit(conv_dim)?;
        let mut q_l2 = e.uninit(value_dim)?;
        let mut k_l2 = e.uninit(value_dim)?;
        let mut v_gd = e.uninit(value_dim)?;
        let mut beta_b = e.uninit(num_v)?;
        let mut g_log = e.uninit(num_v)?;
        for r in 0..t {
            let base = toff + if r % 2 == 0 { 0 } else { 3 };
            let conv_view = table.slice(base..base + 1);
            let in_view = table.slice(base + 1..base + 2);
            let out_view = table.slice(base + 2..base + 3);
            e.ssm_conv1d_fused_decode_b_view(
                &qkv_mixed.slice(r * qkv_w..(r + 1) * qkv_w),
                &conv_view,
                la.ssm_conv1d.float_data(),
                &mut conv_out,
                conv_dim,
                d_conv,
                1,
            )?;
            e.gdn_prep_decode_b_view(
                &conv_out,
                &beta_raw.slice(r * beta_w..(r + 1) * beta_w),
                &alpha.slice(r * alpha_w..(r + 1) * alpha_w),
                la.ssm_dt.float_data(),
                la.ssm_a.float_data(),
                &mut q_l2,
                &mut k_l2,
                &mut v_gd,
                &mut beta_b,
                &mut g_log,
                d_state,
                num_v,
                num_k,
                key_dim,
                eps,
                conv_dim,
                1,
            )?;
            let mut o_row = o_all.slice_mut(r * value_dim..(r + 1) * value_dim);
            e.gdn_scan_s128_batched_view(
                &q_l2, &k_l2, &v_gd, &g_log, &beta_b, &in_view, &out_view, &mut o_row, num_v, 1,
                gdn_scale,
            )?;
            if r + 1 < t {
                // Row r's out buffer: even rows write s1 (the alt handle — no swaps ran),
                // odd rows write s0 — the same physical state the legacy post-swap
                // canonical clone read.
                let rl = cache.recur[il]
                    .as_ref()
                    .ok_or("qwen35 linear verify layer has no recurrent state")?;
                let ssm_src = if r % 2 == 0 {
                    &rl.ssm_state_alt
                } else {
                    &rl.ssm_state
                };
                match stash.as_mut() {
                    Some((conv_slab, ssm_slab)) => {
                        // BOTH stash reads go through the pointer table at run time: the
                        // ssm handles ping-pong between rounds, and the ctx (with its
                        // captured graphs) outlives the Cache — a fresh generation's
                        // conv/ssm buffers land at new addresses that only the per-round
                        // table refresh knows. A baked direct copy would read freed
                        // memory (parity was the slice-3 smoke divergence; cache
                        // lifetime is the cross-generation twin).
                        e.copy_indirect_src_f32(
                            &conv_view,
                            conv_slab,
                            r * conv_dim * (d_conv - 1),
                            conv_dim * (d_conv - 1),
                        )?;
                        // The ssm handles PING-PONG between rounds: a captured direct
                        // copy would bake the capture-time physical buffer and read the
                        // wrong parity after any odd-vt round (the slice-3 smoke
                        // divergence). Read the src address from row r's OUT table
                        // entry at run time — the same entry the scan just wrote.
                        e.copy_indirect_src_f32(
                            &out_view,
                            ssm_slab,
                            r * d_state * d_state * num_v,
                            d_state * d_state * num_v,
                        )?;
                    }
                    None => {
                        if let Some(states) = col_states.as_mut() {
                            states.push((e.clone_dtod(&rl.conv_state)?, e.clone_dtod(ssm_src)?));
                        }
                    }
                }
            }
        }
        // ONE end-of-body parity swap (t odd) — the legacy loop swapped per row; the net
        // handle motion is identical and the device sequence never read the handles.
        if t % 2 == 1 {
            let rl = cache.recur[il].as_mut().unwrap();
            std::mem::swap(&mut rl.ssm_state, &mut rl.ssm_state_alt);
        }
        if let (Some(checkpoint), Some(states)) = (ckpt, col_states) {
            checkpoint.cols[il] = Some(states);
        }

        // ---- batched gated norm + out-projection at m=T ----
        let mixed = if e.uses_q8_1_fast(&la.ssm_out) {
            let (gq, gd) = e.gated_rmsnorm_q8_1(
                &o_all,
                la.ssm_norm.float_data(),
                &z,
                d_state,
                t * num_v,
                eps,
            )?;
            let g0 = e.zeros(0)?;
            e.matmul_pre(&la.ssm_out, &gq, &gd, &g0, t)?
        } else {
            let mut gn = e.uninit(t * value_dim)?;
            e.gated_rmsnorm(
                &o_all,
                la.ssm_norm.float_data(),
                &z,
                &mut gn,
                d_state,
                t * num_v,
                eps,
            )?;
            e.matmul(&la.ssm_out, &gn, t)?
        };

        // ---- residual add + post_attn_norm + FFN at m=T (serving dispatch verbatim) ----
        let pnorm = layer.post_attn_norm.float_data();
        let mut x1 = e.uninit(t * n_embd)?;
        let mut zn = e.uninit(t * n_embd)?;
        e.add_rms_norm(x, &mixed, pnorm, &mut x1, &mut zn, n_embd, t, eps)?;
        let ffn_out = match &layer.ffn {
            crate::hybrid::Ffn::Dense {
                ffn_gate,
                ffn_up,
                ffn_down,
            } => {
                assert!(
                    self.cfg.m3.is_none(),
                    "qwen35 t-parallel verify: M3 swigluoai FFN not yet batched"
                );
                self.qwen35_tparallel_dense_ffn(e, ffn_gate, ffn_up, ffn_down, &zn, t, n_embd)?
            }
            crate::hybrid::Ffn::Moe(m) => self.moe_ffn_il_zq8(e, m, &zn, None, t, il as u16)?,
        };
        let mut x2 = e.uninit(t * n_embd)?;
        e.add(&x1, &ffn_out, &mut x2, t * n_embd)?;
        // dspark drafter tap (no-op when no sink armed): post-layer residual verify rows
        self.dflash_tap(e, cache, il, &x2, t)?;
        Ok(x2)
    }

    /// PP-N STAGE SUBGRAPH of the verify trunk: layers `[lo, hi)` of `decode_step_t_core_stream`'s
    /// walk, verbatim. Enters with a MATERIALIZED `[T, n_embd]` residual (no pending fusion pair
    /// carried in from outside the range) and exits with the range's final residual materialized
    /// (the trailing add executed) — exactly the `decode_layers_eager(lo, hi)` contract, T rows
    /// instead of one.
    ///
    /// EXTRACTED (lane/pp2-spec 2026-08-06) rather than duplicated: `decode_step_t_core_stream` IS
    /// the single funnel every verify forward reaches, and its per-layer dispatch MIRRORING (norm
    /// fusion per layer, the t>=3/spec_m2 batched-linear window, the fused-q8 FFN chain, the
    /// decode-exact projections) is what makes verify bit-identical to eager decode. A second copy
    /// for the split arm is how those mirrors drift apart on the next lever. The unsplit body now
    /// calls this with `(0, n_layers)`, so the whole-trunk path and every stage range run the SAME
    /// code — there is no "split version" of the verify math.
    ///
    /// Bit-identity of a cut rests on the same kernel-check-pinned identity the eager arm's cut
    /// does — `add_rms_norm_q8_1 == add then rms_norm_q8_1` at nrows=T — because the ONLY thing a
    /// fence changes is that the cross-layer fusion carry breaks at `hi-1` and is re-materialized
    /// as an explicit `add`. `decode-batch-gate --mode ppspec` verifies end-to-end on real weights.
    #[allow(clippy::too_many_arguments)]
    fn verify_layers(
        &self,
        e: &Engine,
        mut x: CudaSlice<f32>,
        lo: usize,
        hi: usize,
        pos_d: &CudaSlice<i32>,
        pos0: usize,
        t: usize,
        cache: &mut Cache,
        mut ckpt: Option<&mut VerifyCkpt>,
        stream: Option<(&CudaSlice<u32>, &CudaSlice<i32>)>,
        graphs: Option<&mut DsparkVerifyGraphs>,
    ) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        if self.sliding_gated_moe_batch_program() {
            if stream.is_some() {
                return Err(
                    "step35 has no ROUND-STREAM verify arm (the device-counter _dc twins \
                            cannot express the SWA offset KV view)"
                        .into(),
                );
            }
            return self.step35_verify_batch_layers(e, x, lo, hi, pos0, t, cache);
        }
        if self.batched_serving_numeric_class() {
            return self.qwen35_verify_batch_layers(
                e,
                x,
                lo,
                hi,
                pos0,
                t,
                cache,
                ckpt.take(),
                stream,
                graphs,
            );
        }
        let n_embd = self.cfg.n_embd as usize;
        let eps = self.cfg.rms_eps;
        // CROSS-LAYER ADD+NORM FUSION (lane/vt-fixes fix 2, mirroring decode_step_h's
        // launch-arc form): layer il's post-FFN residual add (x2 = x1 + ffn_out) and layer
        // il+1's attn_norm(+quantize) are consecutive row-wise ops — ONE add_rms_norm_q8_1
        // launch at nrows=t does all three (bit-identity pinned by the T-row kernel-check
        // arms). Carry the un-added (x1, ffn_out) pair; the fused launch materializes x2 (the
        // residual the next layer needs) as its `res` output. Falls back to the separate add
        // when the next layer is off the fused-q8 path.
        let mut pending: Option<(CudaSlice<f32>, CudaSlice<f32>)> = None;
        for il in lo..hi {
            let layer = &self.layers[il];
            // DISPATCH-MIRRORED attn-input RMSNorm (FP-order lesson #8): eager decode fuses the
            // 1024-thread rms_norm_q8_1 ONLY when every mixer projection is q8_1-fast; layers with
            // Float projections (ssm_beta/ssm_alpha on layers 1/2/4 of the 9B NVFP4 GGUF) take the
            // UNFUSED 256-thread rms_norm. The verify norm must mirror that PER-LAYER choice —
            // blockDim changes the sum-of-squares reduce order, and the ULP shift amplifies through
            // the GDN recurrence into argmax flips (measured: 9B text prompt, 1 ULP at layer 2 ->
            // 2.3e-1 logit maxdiff at the head -> K=1..8 divergence at a 0.03-margin token).
            let mixer_fast = self.mixer_in_q8_1_fast(e, &layer.mixer);
            let norm_fused = std::env::var("MEMRA_NO_FUSE_NORMQ").is_err() && mixer_fast;
            // BATCHED EPILOGUE RE-FUSE (lane/vt-fixes fix 2, 2026-08-03): when the norm is
            // dispatch-fused AND every consumer of `h` reads only its q8_1 form (Full mixer:
            // projections only; Linear mixer: the batched arm — the per-column fallback needs
            // f32 h), emit the attn-input norm DIRECTLY as q8_1 via `rms_norm_q8_1` at nrows=t
            // (row-indexed kernel — the T-row launch is the per-row m=1 program, kernel-check
            // pins bit-identity vs rms_norm_decode -> quantize_q8_1). Kills the standalone
            // quantize launch(es) + the f32 h HBM round-trip that decode never pays.
            // step35 (Full mixer) is the third case that needs f32 `h`: its verify arm is a
            // per-ROW replay of the eager decode mixer, whose `pre_q` contract is a single row —
            // a T-row q8_1 pair cannot be handed to it, and re-deriving per-row q8_1 from the f32
            // rows is exactly the dispatch being mirrored. Keep step35 on the unfused arm.
            let lin_q8_only = match &layer.mixer {
                Mixer::Linear(la) => {
                    (t >= 3 || (t == 2 && spec_m2())) && e.uses_q8_1_fast(&la.ssm_out)
                }
                Mixer::Full(_) if self.sliding_gated_moe_batch_program() => false,
                _ => true,
            };
            // NOTE decode.rs's take()-first lesson: take the pending pair BEFORE branching so
            // a non-fused layer still performs the residual add.
            let taken = pending.take();
            let (h, h_q8) = if norm_fused && lin_q8_only {
                let pair = match taken {
                    // fused add + attn_norm + q8_1: ONE launch resolves the carried residual
                    // AND emits this layer's mixer input pre-quantized. res -> x2 (= new x).
                    Some((x1p, f1p)) => {
                        let mut x2 = vbuf(e, t * n_embd)?; // fully written (res output)
                        let p = e.add_rms_norm_q8_1(
                            &x1p,
                            &f1p,
                            layer.attn_norm.float_data(),
                            &mut x2,
                            n_embd,
                            t,
                            eps,
                        )?;
                        x = x2;
                        p
                    }
                    None => e.rms_norm_q8_1(&x, layer.attn_norm.float_data(), n_embd, t, eps)?,
                };
                (e.zeros(0)?, Some(pair)) // h unused on this path (q8-only consumers)
            } else {
                if let Some((x1p, f1p)) = taken {
                    let mut x2 = vbuf(e, t * n_embd)?; // fully written by add
                    e.add(&x1p, &f1p, &mut x2, t * n_embd)?;
                    x = x2;
                }
                let mut h = vbuf(e, t * n_embd)?; // fully written by either rms_norm arm
                if norm_fused {
                    e.rms_norm_decode(&x, layer.attn_norm.float_data(), &mut h, n_embd, t, eps)?;
                } else {
                    e.rms_norm(&x, layer.attn_norm.float_data(), &mut h, n_embd, t, eps)?;
                }
                (h, None)
            };
            let h_q8_ref = h_q8.as_ref().map(|(q, d)| (q, d));

            let mixed = match &layer.mixer {
                Mixer::Full(fa) => self.full_attn_verify(
                    e,
                    fa,
                    &h,
                    h_q8_ref,
                    pos_d,
                    t,
                    cache,
                    il,
                    stream.map(|(_, c)| c),
                )?,
                Mixer::Mla(_) => crate::hybrid::mla_path_unimplemented("speculative verify"),
                Mixer::Kda(_) => crate::hybrid::kda_path_unimplemented("speculative verify"),
                Mixer::Linear(la) => {
                    // BATCHED linear verify (2026-07-03, the MTP-profit lever): one T-token pass —
                    // batched projections (weight read ONCE, hits the m=2-4 weight-resident matvec),
                    // carried-state conv (ssm_conv1d_tm_state), GDN prep on the prefill kernels, and
                    // ONE gdn_scan whose internal sequential t-loop is the SAME recurrence as T
                    // chained T=1 steps (bit-identical). Falls back to the sequential per-column
                    // chain when T < d_conv-1 (conv ring update needs T >= pad) — or when ANY
                    // projection is off the q8_1 fast path: matmul_decode_exact would route a Float
                    // tensor to cuBLAS at m=t (different FP accumulation than eager's per-token
                    // GEMV), so mixed-dtype layers stay on the eager-identical per-column chain.
                    // MEMRA_SPEC_M2 (lane/spec-m2): the t==2 batch rides the same arm — the conv
                    // wrapper handles t<pad with a pure-copy ring rebuild; see spec_m2() header.
                    if (t >= 3 || (t == 2 && spec_m2()))
                        && mixer_fast
                        && e.uses_q8_1_fast(&la.ssm_out)
                    {
                        let want = ckpt.is_some();
                        let (out, stash) =
                            self.linear_attn_verify_t(e, la, &h, h_q8_ref, t, cache, il, want)?;
                        if let (Some(ck), Some(st)) = (ckpt.as_deref_mut(), stash) {
                            ck.gdn[il] = Some(st);
                        }
                        out
                    } else {
                        let mut out = vbuf(e, t * n_embd)?; // every col written by copy_into
                        let mut col_states: Option<Vec<(CudaSlice<f32>, CudaSlice<f32>)>> =
                            if ckpt.is_some() && t >= 2 {
                                Some(Vec::with_capacity(t - 1))
                            } else {
                                None
                            };
                        for col in 0..t {
                            let mut h_col = vbuf(e, n_embd)?; // fully written by copy_view_into
                            let src = h.slice(col * n_embd..(col + 1) * n_embd);
                            e.copy_view_into(&mut h_col, 0, &src, n_embd)?;
                            let m_col = self.linear_attn_decode(e, la, &h_col, cache, il)?;
                            e.copy_into(&mut out, col * n_embd, &m_col, n_embd)?;
                            // REPLAY-FREE ckpt: clone the chain's ACTUAL state after this column
                            // (pure dtod — cannot change any computed value). Last column skipped:
                            // rebuild targets are j <= t-1 columns.
                            if let Some(cs) = col_states.as_mut()
                                && col + 1 < t
                            {
                                let rl = cache.recur[il].as_ref().unwrap();
                                cs.push((
                                    e.clone_dtod(&rl.conv_state)?,
                                    e.clone_dtod(&rl.ssm_state)?,
                                ));
                            }
                        }
                        if let (Some(ck), Some(cs)) = (ckpt.as_deref_mut(), col_states) {
                            // ReplaySSM-assessment instrumentation (2026-07-30): the
                            // per-column clones are the only true state snapshots left in
                            // the verify (the batched path stashes INPUTS and replays).
                            if std::env::var("MEMRA_SPEC_STATS").as_deref() == Ok("1") {
                                static ONCE: std::sync::Once = std::sync::Once::new();
                                let bytes: usize =
                                    cs.iter().map(|(c, s)| (c.len() + s.len()) * 4).sum();
                                ONCE.call_once(|| eprintln!(
                                    "[verify-ckpt] per-column layer il={il}: {} clones, {:.2} MB/layer/round",
                                    cs.len(), bytes as f64 / 1e6));
                            }
                            ck.cols[il] = Some(cs);
                        }
                        out
                    }
                }
            };
            if spec_nan_scan_level() >= 2 {
                let mixed_width = mixed.len() / t;
                nan_scan_rows(
                    e,
                    &mixed,
                    t,
                    mixed_width,
                    &format!("verify layer {il} batched ATTN out pos0={pos0}"),
                )?;
            }

            // DISPATCH-MIRRORED post-attn norm: eager residual_norm_ffn fuses add+norm+quant
            // (1024-thread add_rms_norm_q8_1) only for Dense FFNs whose gate+up are q8_1-fast;
            // otherwise (and for MoE) it runs the 256-thread fused add_rms_norm. Mirror per layer.
            let ffn_fuse = match &layer.ffn {
                crate::hybrid::Ffn::Dense {
                    ffn_gate, ffn_up, ..
                } => {
                    std::env::var("MEMRA_NO_FUSE_NORMQ").is_err()
                        && e.uses_q8_1_fast(ffn_gate)
                        && e.uses_q8_1_fast(ffn_up)
                }
                crate::hybrid::Ffn::Moe(_) => false,
            };
            // BATCHED EPILOGUE RE-FUSE (lane/vt-fixes fix 2): on the ffn_fuse path (Dense,
            // gate+up q8_1-fast, non-M3) the FFN input is emitted DIRECTLY as q8_1 by ONE
            // add_rms_norm_q8_1 launch at nrows=t (row-indexed kernel: the T-row launch is the
            // per-row m=1 program; kernel-check pins bit-identity vs the unfused
            // add_f32 -> rms_norm_decode -> quantize_q8_1 chain at T=2/4/5/8) — replacing the
            // add + rms_norm_decode launches AND the dual/singles' internal re-quantize.
            // M3's swigluoai must keep the f32 chain (the fused SwiGLU epilogue encodes plain
            // SiLU), mirroring residual_norm_ffn's m3 guard on the decode path.
            // step35: same guard per LAYER. A dense FFN's clamp is the SHEXP array (upstream's
            // one build_ffn serves dense + shared expert, llama-graph.cpp:1751), and verify MUST
            // mirror decode's dispatch or spec self-consistency fails.
            let dense_lim = self.cfg.clamp_shexp_at(il as u32);
            let fuse_q8 = ffn_fuse && self.cfg.m3.is_none() && dense_lim.is_none();
            let mut x1 = vbuf(e, t * n_embd)?; // fully written by add / add_rms_norm*
            let mut z = e.zeros(0)?; // replaced below on the unfused arms
            let z_q8 = if fuse_q8 {
                Some(e.add_rms_norm_q8_1(
                    &x,
                    &mixed,
                    layer.post_attn_norm.float_data(),
                    &mut x1,
                    n_embd,
                    t,
                    eps,
                )?)
            } else {
                let mut zf = vbuf(e, t * n_embd)?; // fully written by rms_norm_decode / add_rms_norm
                if ffn_fuse {
                    e.add(&x, &mixed, &mut x1, t * n_embd)?;
                    e.rms_norm_decode(
                        &x1,
                        layer.post_attn_norm.float_data(),
                        &mut zf,
                        n_embd,
                        t,
                        eps,
                    )?;
                } else {
                    e.add_rms_norm(
                        &x,
                        &mixed,
                        layer.post_attn_norm.float_data(),
                        &mut x1,
                        &mut zf,
                        n_embd,
                        t,
                        eps,
                    )?;
                }
                z = zf;
                None
            };
            if spec_nan_scan_level() >= 2 && !z.is_empty() {
                nan_scan_rows(
                    e,
                    &z,
                    t,
                    n_embd,
                    &format!("verify layer {il} post-attn norm z pos0={pos0}"),
                )?;
            }
            // DECODE-EXACT FFN projections: force MMVQ for gate/up/down at any T to match the
            // T=1 decode FP accumulation order. At T>=5 the generic matmul/matmul_pre falls to dp4a
            // (128-thread, different FP sum order). At T=2-4 the batched MMVQ is already bit-identical.
            let ffn_out = match &layer.ffn {
                crate::hybrid::Ffn::Dense {
                    ffn_gate,
                    ffn_up,
                    ffn_down,
                } => {
                    let n_ff = ffn_gate.out_features();
                    if let Some((zq, zd)) = z_q8.as_ref() {
                        // FUSED CHAIN (fix 2): pre-quantized z feeds the projections; the SwiGLU
                        // epilogue emits act pre-quantized for ffn_down (silu_mul_scaled_q8_1,
                        // bit-identical to silu_mul + quantize — kernel-check-pinned) with the
                        // NVFP4 macro-scales folded (deferred-scale dual: y*s inline == the
                        // scale_inplace store, value-exact) — the exact m=1 decode epilogue
                        // structure at nrows=t.
                        let pair = e
                            .matmul_decode_exact_dual_pre(ffn_gate, ffn_up, zq, zd, t)?
                            .map(|((g, gs), (u, us))| (g, gs, u, us));
                        let (gate, gs, up, us) = match pair {
                            Some(x4) => x4,
                            None => (
                                e.matmul_decode_exact_pre(ffn_gate, zq, zd, t)?,
                                1.0, // scale already applied inside _pre
                                e.matmul_decode_exact_pre(ffn_up, zq, zd, t)?,
                                1.0,
                            ),
                        };
                        if e.uses_q8_1_fast(ffn_down) {
                            let (aq, ad) = e.silu_mul_scaled_q8_1(&gate, &up, gs, us, t * n_ff)?;
                            e.matmul_decode_exact_pre(ffn_down, &aq, &ad, t)?
                        } else {
                            let mut act = vbuf(e, t * n_ff)?;
                            e.silu_mul_scaled(&gate, &up, gs, us, &mut act, t * n_ff)?;
                            e.matmul_decode_exact(ffn_down, &act, t)?
                        }
                    } else {
                        // UNFUSED (pre-fix) chain — MoE-adjacent/M3/off-fast layers, unchanged.
                        // DUAL gate+up batched twin (lane/verify-economics, 2026-08-02): one launch
                        // for the pair at t=2..8 — bit-identical per (tensor,token,row) to the two
                        // singles (kernel-check pins bitwise; MEMRA_SPEC_DUAL_T=0 reverts). None
                        // (non-NVFP4 / t outside the tier / seam off) -> the two singles, unchanged.
                        let (gate, up) =
                            match e.matmul_decode_exact_dual(ffn_gate, ffn_up, &z, t)? {
                                Some(pair) => pair,
                                None => (
                                    e.matmul_decode_exact(ffn_gate, &z, t)?,
                                    e.matmul_decode_exact(ffn_up, &z, t)?,
                                ),
                            };
                        let mut act = vbuf(e, t * n_ff)?; // fully written by ffn_act_lim
                        Self::ffn_act_lim(
                            e,
                            &self.cfg,
                            &gate,
                            &up,
                            1.0,
                            1.0,
                            dense_lim,
                            &mut act,
                            t * n_ff,
                        )?;
                        e.matmul_decode_exact(ffn_down, &act, t)?
                    }
                }
                crate::hybrid::Ffn::Moe(m) => self.moe_ffn_il(e, m, &z, t, il as u16)?,
            };
            if spec_nan_scan_level() >= 2 {
                nan_scan_rows(
                    e,
                    &ffn_out,
                    t,
                    n_embd,
                    &format!("verify layer {il} batched FFN out pos0={pos0}"),
                )?;
            }
            if spec_nan_scan() {
                let mut residual = vbuf(e, t * n_embd)?;
                e.add(&x1, &ffn_out, &mut residual, t * n_embd)?;
                nan_scan_rows(
                    e,
                    &residual,
                    t,
                    n_embd,
                    &format!("verify layer {il} residual pos0={pos0}"),
                )?;
            }
            // CROSS-LAYER fusion: defer this layer's post-FFN residual add — the next layer's
            // fused-q8 attn norm folds it in (add_rms_norm_q8_1 == add; rms_norm; quantize,
            // kernel-check-pinned at nrows=T). Non-fused next layers add explicitly above.
            pending = Some((x1, ffn_out));
        }
        // RANGE's final add (no next norm INSIDE the range to fuse with; for the
        // whole-trunk call that is the last layer, whose next norm is output_norm — f32-out).
        if let Some((x1p, f1p)) = pending.take() {
            let mut x2 = vbuf(e, t * n_embd)?; // fully written by add
            e.add(&x1p, &f1p, &mut x2, t * n_embd)?;
            x = x2;
        }
        Ok(x)
    }
    /// BATCHED linear-attn verify (T=K+1): the whole layer in ~10 launches instead of T x the
    /// T=1 decode chain (T x ~12 launches + T weight reads of the four projections). The GDN
    /// recurrence itself is inherently sequential — gdn_scan_s128 runs its internal t-loop with
    /// the SAME per-token math as chained T=1 calls (bit-identical state evolution); everything
    /// around it (projections, conv, prep, gated norm, out-proj) batches. Advances conv ring +
    /// ssm state exactly like T sequential decode steps.
    /// `want_stash`: additionally RETAIN the gdn-scan inputs (pure buffer keep-alives, zero extra
    /// kernels) so a partial accept can rebuild the state after any column prefix (REPLAY-FREE).
    #[allow(clippy::too_many_arguments)]
    fn linear_attn_verify_t(
        &self,
        e: &Engine,
        la: &LinearAttnLayer,
        h: &CudaSlice<f32>,
        h_q8: Option<(&CudaSlice<i8>, &CudaSlice<f32>)>,
        t: usize,
        cache: &mut Cache,
        il: usize,
        want_stash: bool,
    ) -> Result<(CudaSlice<f32>, Option<GdnStash>), Box<dyn std::error::Error>> {
        let cfg = &self.cfg;
        let geometry = la.geometry;
        let d_state = geometry.key_head_dim as usize;
        let num_k = geometry.key_heads as usize;
        let num_v = geometry.value_heads as usize;
        let d_conv = geometry.conv_kernel as usize;
        let key_dim = d_state * num_k;
        let conv_dim = key_dim * 2 + geometry.value_head_dim as usize * num_v;
        let eps = cfg.rms_eps;
        let scale = 1.0 / (d_state as f32).sqrt();

        // DECODE-EXACT projections: matmul_decode_exact forces the MMVQ (warp-per-row, 32-thread)
        // accumulation order for EVERY m, matching the T=1 decode path bit-for-bit. The generic
        // `matmul` at m>=5 falls to dp4a (128-thread, two-level reduce) which has a different FP
        // sum order — ULP differences propagate through gdn_scan and flip argmax on the 27B.
        // Q8 TRUNK-FUSION at T=1 (35B: wqkv+wqkv_gate both Q8_0): one fused2 launch, bit-identical
        // per (tensor,row) to the two m=1 MMVQ dispatches below — decode-exact contract holds.
        // VERIFY-TIER TRUNK FUSION (MEMRA_SPEC_FUSED_T, t=2-4): quantize h ONCE for every
        // fused-eligible same-input Q8_0 pair of this layer (35B wqkv+wqkv_gate; 9B
        // ssm_beta+ssm_alpha) — each fused2 batched launch then replaces two decode-exact
        // calls (each of which re-quantizes the same h + runs its own _b2/_b4 launch).
        // Bit-identical per (tensor,token,row) — see spec_fused_t().
        // BATCHED EPILOGUE RE-FUSE (lane/vt-fixes fix 2): `h_q8` = the attn-input norm emitted
        // directly as q8_1 by the caller's fused rms_norm_q8_1 (bit-identical to the unfused
        // chain, kernel-check-pinned). When present it REPLACES the standalone quantize below
        // and feeds every projection; the caller guaranteed all four input projections are
        // q8_1-fast. When absent, the old shared-quantize (fused-t window) stands.
        let h_q8_t = if h_q8.is_none()
            && spec_fused_t()
            && (2..=4).contains(&t)
            && ((e.uses_q8_1_fast(&la.wqkv) && e.uses_q8_1_fast(&la.wqkv_gate))
                || (e.uses_q8_1_fast(&la.ssm_beta) && e.uses_q8_1_fast(&la.ssm_alpha)))
        {
            Some(e.quantize_q8_1(h, t, cfg.n_embd as usize)?)
        } else {
            None
        };
        // one view: the caller's fused-norm q8 or this fn's own shared quantize.
        let hq8_any: Option<(&CudaSlice<i8>, &CudaSlice<f32>)> =
            h_q8.or(h_q8_t.as_ref().map(|(q, d)| (q, d)));
        let (qkv_mixed, z) = {
            let mut fused = None;
            if t == 1 && e.uses_q8_1_fast(&la.wqkv) && e.uses_q8_1_fast(&la.wqkv_gate) {
                let (hq, hd) = e.quantize_q8_1(h, 1, cfg.n_embd as usize)?;
                fused = e.matmul_q8_fused2(&la.wqkv, &la.wqkv_gate, &hq, &hd)?;
            } else if let Some((hq, hd)) = hq8_any
                && spec_fused_t()
                && (2..=4).contains(&t)
            {
                fused = e.matmul_q8_fused2_t(&la.wqkv, &la.wqkv_gate, hq, hd, t)?;
            }
            match (fused, hq8_any) {
                (Some(pair), _) => pair,
                (None, Some((hq, hd))) if h_q8.is_some() => (
                    e.matmul_decode_exact_pre(&la.wqkv, hq, hd, t)?,
                    e.matmul_decode_exact_pre(&la.wqkv_gate, hq, hd, t)?,
                ),
                (None, _) => (
                    e.matmul_decode_exact(&la.wqkv, h, t)?,
                    e.matmul_decode_exact(&la.wqkv_gate, h, t)?,
                ),
            }
        };
        // beta+alpha DUAL at T=1 (75% of p3 rounds run T=1 verify — p-min chain cuts): the dual
        // mr2 kernel is bit-identical per element to the m=1 MMVQ matmul_decode_exact dispatches
        // (same warp-per-row body, blockIdx.y picks the weight), so the decode-exact contract
        // holds; the run-spec battery is the arbiter. T>1 keeps the per-tensor decode-exact path.
        let (beta_raw, alpha) = if t == 1 {
            let (hq, hd) = e.quantize_q8_1(h, 1, cfg.n_embd as usize)?;
            match e.matmul_pre_dual_noscale(&la.ssm_beta, &la.ssm_alpha, &hq, &hd, 1)? {
                Some(((mut b, bs), (mut a, as_))) => {
                    if bs != 1.0 {
                        e.scale_inplace(&mut b, bs, la.ssm_beta.out_features())?;
                    }
                    if as_ != 1.0 {
                        e.scale_inplace(&mut a, as_, la.ssm_alpha.out_features())?;
                    }
                    (b, a)
                }
                // Q8_0 fused2 twin (9B stores beta/alpha as Q8_0): DISPATCH-MIRRORS the eager
                // decode's beta_alpha closure — the fused body is qmatvec_q8_0_mmvq verbatim,
                // bit-identical per row (kernel-check rel=0.00e0 gate), so decode==verify holds.
                None => match e.matmul_q8_fused2(&la.ssm_beta, &la.ssm_alpha, &hq, &hd)? {
                    Some((b, a)) => (b, a),
                    None => (
                        e.matmul_decode_exact(&la.ssm_beta, h, 1)?,
                        e.matmul_decode_exact(&la.ssm_alpha, h, 1)?,
                    ),
                },
            }
        } else {
            // fused-t twin (9B stores beta/alpha as Q8_0): same shared-quantize + one launch
            // contract as the wqkv pair above; 35B beta/alpha are Float -> None -> fallback.
            let mut nvfp4_fused = None;
            let mut q8_fused = None;
            if let Some((hq, hd)) = hq8_any {
                if t == 3 && std::env::var("MEMRA_NVFP4_AUX_DUAL").as_deref() != Ok("0") {
                    nvfp4_fused =
                        e.matmul_decode_exact_dual_pre(&la.ssm_beta, &la.ssm_alpha, hq, hd, t)?;
                    if nvfp4_fused.is_some() && std::env::var("MEMRA_DEBUG").is_ok() {
                        static ONCE: std::sync::Once = std::sync::Once::new();
                        ONCE.call_once(|| {
                            eprintln!("[memra] NVFP4 beta+alpha batched aux dual ENGAGED (t={t})")
                        });
                    }
                }
                if nvfp4_fused.is_none() && spec_fused_t() && (2..=4).contains(&t) {
                    q8_fused = e.matmul_q8_fused2_t(&la.ssm_beta, &la.ssm_alpha, hq, hd, t)?;
                }
            }
            if let Some(((mut b, bs), (mut a, as_))) = nvfp4_fused {
                if bs != 1.0 {
                    e.scale_inplace(&mut b, bs, t * la.ssm_beta.out_features())?;
                }
                if as_ != 1.0 {
                    e.scale_inplace(&mut a, as_, t * la.ssm_alpha.out_features())?;
                }
                (b, a)
            } else if let Some(pair) = q8_fused {
                pair
            } else {
                match hq8_any {
                    Some((hq, hd)) if h_q8.is_some() => (
                        e.matmul_decode_exact_pre(&la.ssm_beta, hq, hd, t)?,
                        e.matmul_decode_exact_pre(&la.ssm_alpha, hq, hd, t)?,
                    ),
                    _ => (
                        e.matmul_decode_exact(&la.ssm_beta, h, t)?,
                        e.matmul_decode_exact(&la.ssm_alpha, h, t)?,
                    ),
                }
            }
        };

        // conv with CARRIED state + ring roll (T >= pad rides the input-column update kernel;
        // T < pad — the MEMRA_SPEC_M2 t=2 arm — rolls via the pure-copy ring rebuild).
        let rl = cache.recur[il].as_mut().unwrap();
        let mut conv_out = e.uninit(conv_dim * t)?;
        e.ssm_conv1d_tm_state(
            &qkv_mixed,
            &mut rl.conv_state,
            la.ssm_conv1d.float_data(),
            &mut conv_out,
            conv_dim,
            t,
            d_conv,
        )?;

        // GDN prep via the prefill kernels (repack + L2 + sigmoid + glog), T-wide.
        let mut q_g = e.uninit(d_state * num_v * t)?;
        let mut k_g = e.uninit(d_state * num_v * t)?;
        let mut v_g = e.uninit(d_state * num_v * t)?;
        e.qkv_to_gdn_repack(
            &conv_out, &mut q_g, &mut k_g, &mut v_g, d_state, num_v, num_k, key_dim, t,
        )?;
        let mut q_l2 = e.uninit(d_state * num_v * t)?;
        e.l2_norm_decode(&q_g, &mut q_l2, d_state, num_v * t, eps)?;
        let mut k_l2 = e.uninit(d_state * num_v * t)?;
        e.l2_norm_decode(&k_g, &mut k_l2, d_state, num_v * t, eps)?;
        let mut beta = e.uninit(t * num_v)?;
        e.sigmoid(&beta_raw, &mut beta, t * num_v)?;
        let mut g_log = e.uninit(t * num_v)?;
        e.gdn_glog(
            &alpha,
            la.ssm_dt.float_data(),
            la.ssm_a.float_data(),
            &mut g_log,
            num_v,
            t,
        )?;

        // ONE gdn_scan over T tokens from the carried state (internal sequential loop ==
        // T chained T=1 steps). Ping-pong the resident buffers like eager decode.
        let mut o = e.uninit(d_state * num_v * t)?;
        {
            let crate::cache::RecurLayer {
                ssm_state,
                ssm_state_alt,
                ..
            } = rl;
            e.gdn_scan_s128(
                &q_l2,
                &k_l2,
                &v_g,
                &g_log,
                &beta,
                ssm_state,
                ssm_state_alt,
                &mut o,
                num_v,
                t,
                scale,
            )?;
        }
        std::mem::swap(&mut rl.ssm_state, &mut rl.ssm_state_alt);

        // gated RMSNorm + out projection, T-wide. FUSED-QUANTIZE ARM (lane/vt-fixes fix 2,
        // mirroring the T=1 decode's launch-arc form): when ssm_out rides the q8_1 fast path,
        // emit q8_1 straight from the gated norm at nrows=num_v*t (row-indexed kernel, the
        // T-wide launch is the per-row program; kernel-check pins bit-identity vs
        // gated_rmsnorm -> quantize_q8_1 at T=1 and T=5) and feed the decode-exact dispatch
        // pre-quantized — one launch replaces norm + quantize. Fallback = the f32 chain.
        let out = if e.uses_q8_1_fast(&la.ssm_out) {
            let (gq, gd) =
                e.gated_rmsnorm_q8_1(&o, la.ssm_norm.float_data(), &z, d_state, num_v * t, eps)?;
            e.matmul_decode_exact_pre(&la.ssm_out, &gq, &gd, t)?
        } else {
            let mut gn = e.uninit(d_state * num_v * t)?;
            e.gated_rmsnorm(
                &o,
                la.ssm_norm.float_data(),
                &z,
                &mut gn,
                d_state,
                num_v * t,
                eps,
            )?;
            // DECODE-EXACT out-projection: same MMVQ path as the T=1 decode (ssm_out at m>=5
            // would fall to dp4a with a different FP reduction order — same class of bug as
            // the input projs).
            e.matmul_decode_exact(&la.ssm_out, &gn, t)?
        };
        let stash = if want_stash {
            Some(GdnStash {
                qkv_mixed,
                q_l2,
                k_l2,
                v_g,
                g_log,
                beta,
            })
        } else {
            None
        };
        Ok((out, stash))
    }

    /// REPLAY-FREE partial-accept commit (2026-07-03): make the cache state == "committed through
    /// the first `j` verify columns" WITHOUT the legacy rollback + duplicate trunk replay.
    /// - Full-attn KV: truncate both the owning-stage shadow and every TP rank to snapshot + j.
    ///   The verify's appended rows for those columns are bit-identical to what an eager T=1
    ///   chain writes (the decode-exact contract the verify-probe gates), so keeping them ==
    ///   replaying them.
    /// - Linear layers, batched path: rebuild the conv ring by PURE COPIES (ring holds raw input
    ///   columns) and the ssm state by a prefix re-run of the SAME gdn_scan kernel (t=j) from the
    ///   snapshot state over the stash's identical inputs — the kernel's t-loop carries state in
    ///   registers and writes it once at the end, so iterations 0..j-1 are independent of T:
    ///   bit-identical to the verify's own state after j tokens == the eager chain state.
    /// - Linear layers, per-column path: restore the cloned actual state after column j-1.
    ///   Caller guarantees 1 <= j <= t-1 (j==0 rounds take the legacy rollback; j==t is full accept).
    #[allow(clippy::too_many_arguments)] // allow: the parameter list mirrors the kernel/FFI/call contract; bundling into a struct is a refactor, not a lint fix
    fn commit_verified_prefix(
        &self,
        e: &Engine,
        cache: &mut Cache,
        snap: &crate::cache::CacheSnapshot,
        ckpt: &VerifyCkpt,
        j: usize,
        kv_lens_done: bool,
        dev_j: Option<(&CudaSlice<u32>, usize, usize)>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // GDN geometry derives lazily inside recurrent-layer arms. Full-attention plans carry no
        // recurrent state and must never be forced through a synthetic SSM geometry.
        // Engine-bundle slice 1 (DSF-ROUNDCOST-20260820 §1.1): the per-column-arm restores
        // are 2 tiny D2D copies per linear layer (~96 dispatches/partial round on the q38
        // route). When every cols-arm layer shares uniform state sizes (single ssm cfg —
        // always true today), batch them into two `copy_batch_uniform_f32` launches. Bytes,
        // buffers and stream order are identical to the per-layer memcpy sequence; the
        // kernel-rebuild (gdn-stash) arm below is untouched. MEMRA_STATE_COPY_BATCH=0 reverts.
        let mut batched_cols = false;
        if state_copy_batch_on() && dev_j.is_none() {
            use cudarc::driver::DevicePtr;
            let s = &e.gpu.stream();
            let mut conv_pairs: Vec<(u64, u64)> = Vec::new();
            let mut ssm_pairs: Vec<(u64, u64)> = Vec::new();
            let (mut conv_words, mut ssm_words) = (0usize, 0usize);
            let mut uniform = true;
            for il in 0..self.layers.len() {
                let Some(rl) = cache.recur[il].as_ref() else {
                    continue;
                };
                if ckpt.gdn[il].is_some() {
                    continue; // kernel-rebuild arm restores below, per layer
                }
                let Some(cols) = &ckpt.cols[il] else {
                    continue; // missing-ckpt error surfaces in the main loop
                };
                let (c, st) = &cols[j - 1];
                if conv_pairs.is_empty() {
                    conv_words = c.len();
                    ssm_words = st.len();
                } else if c.len() != conv_words || st.len() != ssm_words {
                    uniform = false;
                    break;
                }
                let (pc, _g0) = c.device_ptr(s);
                let (dc, _g1) = rl.conv_state.device_ptr(s);
                let (ps, _g2) = st.device_ptr(s);
                let (ds, _g3) = rl.ssm_state.device_ptr(s);
                conv_pairs.push((pc, dc));
                ssm_pairs.push((ps, ds));
            }
            if uniform && !conv_pairs.is_empty() {
                let n = conv_pairs.len();
                let mut t = vec![0u64; 2 * n];
                for (k, &(src, dst)) in conv_pairs.iter().enumerate() {
                    t[k] = src;
                    t[n + k] = dst;
                }
                let conv_t = e.htod_u64(&t)?;
                for (k, &(src, dst)) in ssm_pairs.iter().enumerate() {
                    t[k] = src;
                    t[n + k] = dst;
                }
                let ssm_t = e.htod_u64(&t)?;
                e.copy_batch_uniform_f32(&conv_t, n, conv_words)?;
                e.copy_batch_uniform_f32(&ssm_t, n, ssm_words)?;
                batched_cols = true;
            }
        }
        for il in 0..self.layers.len() {
            if let (Some(kvl), Some(saved)) = (cache.kv[il].as_mut(), snap.kv_len[il]) {
                kvl.len = saved + j;
                // devacc 3a: spec_rollback_kv already wrote len_d on-device (same value).
                if !kv_lens_done {
                    e.set_i32_one(&mut kvl.len_d, kvl.len as i32)?;
                }
            }
            if let Some(rl) = cache.recur[il].as_mut() {
                let Mixer::Linear(linear) = &self.layers[il].mixer else {
                    return Err(format!("recurrent cache layer {il} has no GDN plan").into());
                };
                let geometry = linear.geometry;
                let d_state = geometry.key_head_dim as usize;
                let num_k = geometry.key_heads as usize;
                let num_v = geometry.value_heads as usize;
                let d_conv = geometry.conv_kernel as usize;
                let conv_dim = d_state * num_k * 2 + geometry.value_head_dim as usize * num_v;
                let scale = 1.0 / (d_state as f32).sqrt();
                if let Some(st) = &ckpt.gdn[il] {
                    let ring_old = snap.conv[il].as_ref().expect("snapshot missing conv");
                    let state_in = snap.ssm[il].as_ref().expect("snapshot missing ssm");
                    if let Some((acc, base, t_v)) = dev_j {
                        // 3b: j read on-device (_dc twins, same bodies; full accept early-exits).
                        e.ssm_conv_ring_rebuild_dc(
                            &st.qkv_mixed,
                            ring_old,
                            &mut rl.conv_state,
                            conv_dim,
                            acc,
                            base,
                            t_v,
                            d_conv,
                        )?;
                        let mut o = e.uninit(d_state * num_v * j.max(1))?;
                        e.gdn_scan_s128_dc(
                            &st.q_l2,
                            &st.k_l2,
                            &st.v_g,
                            &st.g_log,
                            &st.beta,
                            state_in,
                            &mut rl.ssm_state,
                            &mut o,
                            num_v,
                            acc,
                            base,
                            t_v,
                            scale,
                        )?;
                    } else {
                        e.ssm_conv_ring_rebuild(
                            &st.qkv_mixed,
                            ring_old,
                            &mut rl.conv_state,
                            conv_dim,
                            j,
                            d_conv,
                        )?;
                        let mut o = e.uninit(d_state * num_v * j)?; // scan output, discarded
                        e.gdn_scan_s128(
                            &st.q_l2,
                            &st.k_l2,
                            &st.v_g,
                            &st.g_log,
                            &st.beta,
                            state_in,
                            &mut rl.ssm_state,
                            &mut o,
                            num_v,
                            j,
                            scale,
                        )?;
                    }
                } else if let Some(cols) = &ckpt.cols[il] {
                    if !batched_cols {
                        let (c, s) = &cols[j - 1];
                        e.copy_into(&mut rl.conv_state, 0, c, c.len())?;
                        e.copy_into(&mut rl.ssm_state, 0, s, s.len())?;
                    }
                } else {
                    return Err(
                        "commit_verified_prefix: verify ckpt missing for linear layer".into(),
                    );
                }
            }
        }
        self.restore_step_tp_kv_verified_prefix(e, cache, snap, j, true)?;
        cache.pos = snap.pos + j;
        Ok(())
    }

    /// ROUND-STREAM: recur restore with device-j (the _dc twins; full accept early-exits
    /// in-kernel). Requires the batched-linear stash on every linear layer (stream gate).
    #[allow(clippy::too_many_arguments)] // allow: the parameter list mirrors the kernel/FFI/call contract; bundling into a struct is a refactor, not a lint fix
    fn commit_verified_prefix_stream(
        &self,
        e: &Engine,
        cache: &mut Cache,
        snap: &crate::cache::CacheSnapshot,
        ckpt: &VerifyCkpt,
        acc: &CudaSlice<u32>,
        base: usize,
        t_v: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        for il in 0..self.layers.len() {
            if let Some(rl) = cache.recur[il].as_mut() {
                let Mixer::Linear(linear) = &self.layers[il].mixer else {
                    return Err(format!("recurrent cache layer {il} has no GDN plan").into());
                };
                let geometry = linear.geometry;
                let d_state = geometry.key_head_dim as usize;
                let num_k = geometry.key_heads as usize;
                let num_v = geometry.value_heads as usize;
                let d_conv = geometry.conv_kernel as usize;
                let conv_dim = d_state * num_k * 2 + geometry.value_head_dim as usize * num_v;
                let scale = 1.0 / (d_state as f32).sqrt();
                let st = ckpt.gdn[il]
                    .as_ref()
                    .ok_or("stream restore: batched-linear stash missing")?;
                let ring_old = snap.conv[il].as_ref().expect("snapshot missing conv");
                let state_in = snap.ssm[il].as_ref().expect("snapshot missing ssm");
                e.ssm_conv_ring_rebuild_dc(
                    &st.qkv_mixed,
                    ring_old,
                    &mut rl.conv_state,
                    conv_dim,
                    acc,
                    base,
                    t_v,
                    d_conv,
                )?;
                let mut o = e.uninit(d_state * num_v * t_v)?;
                e.gdn_scan_s128_dc(
                    &st.q_l2,
                    &st.k_l2,
                    &st.v_g,
                    &st.g_log,
                    &st.beta,
                    state_in,
                    &mut rl.ssm_state,
                    &mut o,
                    num_v,
                    acc,
                    base,
                    t_v,
                    scale,
                )?;
            }
        }
        Ok(())
    }

    /// EAGLE3 aux-capturing verify forward over `tokens` (T) — mirrors `decode_step_t_h` exactly
    /// (same KV append, same causal verify, same recur advance) but ALSO clones the aux residual-
    /// stream hiddens (blocks in `aux_layers`) for TWO columns: the LAST column (always) and the
    /// optional `pred_col` (the EAGLE seed = bonus's predecessor). Returns
    /// (all_T_logits host, last_col_aux, pred_col_aux?). Used by the EAGLE3 orchestrator's commit.
    #[allow(clippy::type_complexity)] // allow: one-shot composite type; naming it would hide the shape that matters at the call site
    pub fn decode_step_t_aux2(
        &self,
        e: &Engine,
        tokens: &[u32],
        pos0: usize,
        cache: &mut Cache,
        aux_layers: &[usize],
        pred_col: Option<usize>,
    ) -> Result<
        (Vec<f32>, Vec<CudaSlice<f32>>, Option<Vec<CudaSlice<f32>>>),
        Box<dyn std::error::Error>,
    > {
        cache.ensure_usable("decode_step_t_aux2")?;
        let cfg = &self.cfg;
        let n_embd = cfg.n_embd as usize;
        let eps = cfg.rms_eps;
        let t = tokens.len();
        let pos_vec: Vec<i32> = (0..t).map(|i| (pos0 + i) as i32).collect();
        let pos_d = e.htod_i32(&pos_vec)?;
        let mut x = e.htod(&self.embd.try_gather(n_embd, tokens)?)?;
        let mut aux_last: Vec<CudaSlice<f32>> = Vec::with_capacity(aux_layers.len());
        let mut aux_pred: Vec<CudaSlice<f32>> = Vec::new();
        let want_pred = pred_col.is_some();

        for (il, layer) in self.layers.iter().enumerate() {
            // DISPATCH-MIRRORED norms (FP-order lesson #8) — see decode_step_t_h_emb.
            let mixer_fast = self.mixer_in_q8_1_fast(e, &layer.mixer);
            let norm_fused = std::env::var("MEMRA_NO_FUSE_NORMQ").is_err() && mixer_fast;
            let mut h = vbuf(e, t * n_embd)?; // fully written by either rms_norm arm
            if norm_fused {
                e.rms_norm_decode(&x, layer.attn_norm.float_data(), &mut h, n_embd, t, eps)?;
            } else {
                e.rms_norm(&x, layer.attn_norm.float_data(), &mut h, n_embd, t, eps)?;
            }
            let mixed = match &layer.mixer {
                Mixer::Full(fa) => {
                    self.full_attn_verify(e, fa, &h, None, &pos_d, t, cache, il, None)?
                }
                Mixer::Mla(_) => {
                    crate::hybrid::mla_path_unimplemented("auxiliary T-parallel decode")
                }
                Mixer::Kda(_) => crate::hybrid::kda_path_unimplemented("aux decode step"),
                Mixer::Linear(la) => {
                    let mut out = e.zeros(t * n_embd)?;
                    for col in 0..t {
                        let mut h_col = e.zeros(n_embd)?;
                        let src = h.slice(col * n_embd..(col + 1) * n_embd);
                        e.copy_view_into(&mut h_col, 0, &src, n_embd)?;
                        let m_col = self.linear_attn_decode(e, la, &h_col, cache, il)?;
                        e.copy_into(&mut out, col * n_embd, &m_col, n_embd)?;
                    }
                    out
                }
            };
            let ffn_fuse = match &layer.ffn {
                crate::hybrid::Ffn::Dense {
                    ffn_gate, ffn_up, ..
                } => {
                    std::env::var("MEMRA_NO_FUSE_NORMQ").is_err()
                        && e.uses_q8_1_fast(ffn_gate)
                        && e.uses_q8_1_fast(ffn_up)
                }
                crate::hybrid::Ffn::Moe(_) => false,
            };
            let mut x1 = vbuf(e, t * n_embd)?; // fully written by add / add_rms_norm
            let mut z = vbuf(e, t * n_embd)?; // fully written by rms_norm_decode / add_rms_norm
            if ffn_fuse {
                e.add(&x, &mixed, &mut x1, t * n_embd)?;
                e.rms_norm_decode(
                    &x1,
                    layer.post_attn_norm.float_data(),
                    &mut z,
                    n_embd,
                    t,
                    eps,
                )?;
            } else {
                e.add_rms_norm(
                    &x,
                    &mixed,
                    layer.post_attn_norm.float_data(),
                    &mut x1,
                    &mut z,
                    n_embd,
                    t,
                    eps,
                )?;
            }
            let ffn_out = match &layer.ffn {
                crate::hybrid::Ffn::Dense {
                    ffn_gate,
                    ffn_up,
                    ffn_down,
                } => {
                    let n_ff = ffn_gate.out_features();
                    let gate = e.matmul_decode_exact(ffn_gate, &z, t)?;
                    let up = e.matmul_decode_exact(ffn_up, &z, t)?;
                    let mut act = vbuf(e, t * n_ff)?; // fully written by ffn_act_lim
                    // dense FFN clamp = the SHEXP array (upstream build_ffn serves both).
                    Self::ffn_act_lim(
                        e,
                        &self.cfg,
                        &gate,
                        &up,
                        1.0,
                        1.0,
                        self.cfg.clamp_shexp_at(il as u32),
                        &mut act,
                        t * n_ff,
                    )?;
                    e.matmul_decode_exact(ffn_down, &act, t)?
                }
                crate::hybrid::Ffn::Moe(m) => self.moe_ffn_il(e, m, &z, t, il as u16)?,
            };
            let mut x2 = vbuf(e, t * n_embd)?; // fully written by add
            e.add(&x1, &ffn_out, &mut x2, t * n_embd)?;
            if aux_layers.contains(&il) {
                let mut a = e.zeros(n_embd)?;
                e.copy_view_into(&mut a, 0, &x2.slice((t - 1) * n_embd..t * n_embd), n_embd)?;
                aux_last.push(a);
                if let Some(pc) = pred_col {
                    let mut ap = e.zeros(n_embd)?;
                    e.copy_view_into(
                        &mut ap,
                        0,
                        &x2.slice(pc * n_embd..(pc + 1) * n_embd),
                        n_embd,
                    )?;
                    aux_pred.push(ap);
                }
            }
            x = x2;
        }
        let mut hn = vbuf(e, t * n_embd)?; // fully written by rms_norm_decode
        e.rms_norm_decode(&x, self.output_norm.float_data(), &mut hn, n_embd, t, eps)?;
        let logits = e.matmul_decode_exact(&self.output, &hn, t)?;
        let host = e.dtoh(&logits)?;
        cache.pos += t;
        Ok((
            host,
            aux_last,
            if want_pred { Some(aux_pred) } else { None },
        ))
    }

    /// step35 SPEC-VERIFY attention over T query tokens — a per-row REPLAY of the eager
    /// `step35_decode_attn`.
    ///
    /// WHY A REPLAY AND NOT A BATCHED TWIN. The verify's whole job is to be bit-identical to what
    /// the eager decode would have computed for the same tokens; that is what makes greedy spec
    /// decode exact (run-spec asserts token identity for K=1..8). Every other verify arm in this
    /// file earns that identity by carefully mirroring dispatch (`matmul_decode_exact` to force
    /// MMVQ at any m, per-layer `ffn_fuse` mirroring, per-row `fa_decode` key bounds). step35
    /// stacks FOUR more per-layer degrees of freedom on top of that — per-layer `n_head`
    /// (64 full / 96 SWA), per-layer rotary width (64 full / 128 SWA), per-layer rope base, and a
    /// SEPARATE `attn_gate` tensor whose projection shares the attn-normed input — and its SWA
    /// layers attend through a token-OFFSET view whose offset is a function of the ABSOLUTE
    /// position of each query row. A batched twin would have to reproduce all of that AND the
    /// per-row offset in one launch; the offset alone rules out the existing rows kernels (they
    /// take one `base_len`, not a per-row offset).
    ///
    /// So this arm calls the eager path itself, once per row, on the same cache. Identity is then
    /// true BY CONSTRUCTION rather than by mirroring: row r runs exactly the kernel sequence that
    /// eager decode step r runs (same projections, same q8_1 fusion decision, same append, same
    /// view arithmetic, same `fa_decode_kvmod`, same gate), because it IS that code. Cost: T x the
    /// eager decode mixer instead of one batched pass — the same trade the generic arm's `else`
    /// per-row loop already accepts when `fa_rows_eligible` says no. Correctness first; a batched
    /// step35 twin is a perf lane's job and must be gated against this arm.
    ///
    /// The `h_q8` pre-quantized pair from the caller's fused norm is NOT forwarded: it is a
    /// T-row buffer and `step35_decode_attn`'s `pre_q` contract is one row. Instead each row's
    /// f32 `h` slice is handed over and the callee re-derives its own q8_1 exactly as eager decode
    /// does (`quantize_q8_1(h, 1, n_embd)`) — which is the dispatch being mirrored. Callers that
    /// took the fused arm therefore MUST still pass a live `h`; `step35_verify` asserts that.
    #[allow(clippy::too_many_arguments)]
    fn step35_verify(
        &self,
        e: &Engine,
        fa: &FullAttnLayer,
        h: &CudaSlice<f32>,
        h_q8: Option<(&CudaSlice<i8>, &CudaSlice<f32>)>,
        t: usize,
        cache: &mut Cache,
        il: usize,
    ) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        let n_embd = self.cfg.n_embd as usize;
        // The fused (h-less) attn-norm arm hands `h` as a zero-length placeholder. This arm needs
        // the f32 rows, so the caller must not take that lever for step35 — enforced at the call
        // site by the sliding-gated-MoE `Mixer::Full(_) => false` arm of
        // `lin_q8_only` in `decode_step_t_core_stream`, and asserted here so a future caller
        // cannot regress it into silently reading an empty buffer.
        assert_eq!(
            h.len(),
            t * n_embd,
            "step35_verify needs the f32 attn-normed rows ([t*n_embd]); the caller took the \
             fused q8-only norm arm (h_q8={}) — step35 must stay on the unfused arm",
            h_q8.is_some()
        );
        // ROW WIDTH IS n_embd, NOT n_head*head_dim: `step35_decode_attn` returns the mixer output
        // AFTER `wo`, so a row is [n_embd] — the same contract the generic arm's
        // `matmul_decode_exact(&fa.wo, &attn_g, t)` return has. Sizing this buffer from the
        // per-layer head geometry (8192 on full-attn, 12288 on SWA) instead overran the row on the
        // FIRST copy and panicked inside `copy_into`'s `CudaView::slice` unwrap
        // (raw/mtp-bt-20260806T212127Z.log frames 12-13).
        let mut out = vbuf(e, t * n_embd)?; // each row fully written by the copy below
        for r in 0..t {
            // Absolute position of this query row. `cache.pos` is the committed length at round
            // start and every row before r has already been appended by this loop, so the r-th
            // verify token sits at cache.pos + r — the same position eager decode would give it.
            let pos_d = e.htod_i32(&[(cache.pos + r) as i32])?;
            let mut h_row = vbuf(e, n_embd)?; // fully written by copy_view_into
            e.copy_view_into(
                &mut h_row,
                0,
                &h.slice(r * n_embd..(r + 1) * n_embd),
                n_embd,
            )?;
            // THE eager decode mixer: appends this row's K/V at kvl.len, advances it, then
            // attends over the (SWA-offset) view. Post-`wo`, same contract as this fn returns.
            let o = self.step35_decode_attn(e, fa, il, &h_row, None, &pos_d, cache)?;
            debug_assert_eq!(
                o.len(),
                n_embd,
                "step35_decode_attn returns post-wo [n_embd]"
            );
            e.copy_into(&mut out, r * n_embd, &o, n_embd)?;
        }
        Ok(out)
    }

    /// Full-attention mixer over T query tokens with a GROWING resident KV (verify path, §D.3).
    /// Appends the T new K/V columns to cache.kv[il] then attends causally over [0..len) via
    /// fa_prefill. Token-major [T, kv_dim] projection layout == cache row layout (single copy).
    #[allow(clippy::too_many_arguments)]
    fn full_attn_verify(
        &self,
        e: &Engine,
        fa: &FullAttnLayer,
        h: &CudaSlice<f32>,
        h_q8: Option<(&CudaSlice<i8>, &CudaSlice<f32>)>,
        pos_d: &CudaSlice<i32>,
        t: usize,
        cache: &mut Cache,
        il: usize,
        stream_ctr: Option<&CudaSlice<i32>>,
    ) -> Result<CudaSlice<f32>, Box<dyn std::error::Error>> {
        // step35: the generic geometry below is wrong for this arch (per-layer n_head, partial
        // per-layer rope, the SWA offset view, and a SEPARATE head-wise gate tensor), so it takes
        // its own arm. A verify that silently computes different attention than decode defeats the
        // whole self-consistency gate, so the arm is a per-row REPLAY of `step35_decode_attn`
        // rather than a batched twin — see `step35_verify` for why that is the exactness-correct
        // shape and not laziness.
        if self.sliding_gated_moe_batch_program() {
            if stream_ctr.is_some() {
                return Err(
                    "step35 has no ROUND-STREAM verify arm (the device-counter _dc twins \
                            cannot express the SWA offset KV view; same root cause as the dc \
                            decode refusal) — run spec without the stream arm"
                        .into(),
                );
            }
            return self.step35_verify(e, fa, h, h_q8, t, cache, il);
        }
        let cfg = &self.cfg;
        let geometry = cfg.full_attention_geometry_at(il as u32);
        let n_head = geometry.n_head as usize;
        let n_head_kv = geometry.n_head_kv as usize;
        let head_dim = geometry.head_dim_k as usize;
        let eps = cfg.rms_eps;
        let scale = geometry.attention_scale();
        let n_embd = cfg.n_embd as usize;

        // DECODE-EXACT Q/K/V projections: matmul_decode_exact forces the MMVQ (warp-per-row) path
        // for every m, matching the T=1 decode's FP accumulation order. matmul_pre at m>=5 would
        // fall to dp4a (128-thread, two-level reduce) with a different FP sum order.
        // Q8 TRUNK-FUSION at T=1: DISPATCH-MIRRORS the eager decode's fused3 (bit-identical body).
        // BATCHED EPILOGUE RE-FUSE (lane/vt-fixes fix 2): `h_q8` = the attn-input norm's q8_1
        // form emitted by the fused rms_norm_q8_1 (bit-identical to rms_norm_decode ->
        // quantize_q8_1, kernel-check-pinned). When present (caller checked mixer q8_1-fast),
        // every projection consumes it — `h` may be a zero-len placeholder and must not be read.
        let (qf, mut k, v) = if let Some(mut qkv) = self.full_attn_tp_qkv(e, fa, h, t)? {
            let v = qkv.pop().ok_or("full-attention TP verify QKV omitted V")?;
            let k = qkv.pop().ok_or("full-attention TP verify QKV omitted K")?;
            let q = qkv.pop().ok_or("full-attention TP verify QKV omitted Q")?;
            if !qkv.is_empty() {
                return Err("full-attention TP verify QKV returned extra projections".into());
            }
            (q, k, v)
        } else {
            let mut fused = None;
            let qkv_fast =
                e.uses_q8_1_fast(&fa.wq) && e.uses_q8_1_fast(&fa.wk) && e.uses_q8_1_fast(&fa.wv);
            if t == 1 && qkv_fast {
                let (hq_o, hd_o);
                let (hq, hd): (&CudaSlice<i8>, &CudaSlice<f32>) = match h_q8 {
                    Some(p) => p,
                    None => {
                        (hq_o, hd_o) = e.quantize_q8_1(h, 1, n_embd)?;
                        (&hq_o, &hd_o)
                    }
                };
                fused = e.matmul_q8_fused3(&fa.wq, &fa.wk, &fa.wv, hq, hd)?;
            } else if spec_fused_t() && (2..=4).contains(&t) && qkv_fast {
                // VERIFY-TIER TRUNK FUSION (MEMRA_SPEC_FUSED_T): one shared quantize + one
                // fused3 batched launch replaces three decode-exact calls (3 re-quantizes of
                // the same h + 3 _b2/_b4 launches). Bit-identical per (tensor,token,row).
                let (hq_o, hd_o);
                let (hq, hd): (&CudaSlice<i8>, &CudaSlice<f32>) = match h_q8 {
                    Some(p) => p,
                    None => {
                        (hq_o, hd_o) = e.quantize_q8_1(h, t, n_embd)?;
                        (&hq_o, &hd_o)
                    }
                };
                fused = e.matmul_q8_fused3_t(&fa.wq, &fa.wk, &fa.wv, hq, hd, t)?;
            }
            match (fused, h_q8) {
                (Some(triple), _) => triple,
                // shared pre-quantized activation (q8_1-fast guaranteed by the caller): the
                // decode-exact dispatch consumes (hq, hd) instead of re-quantizing 3x.
                (None, Some((hq, hd))) if qkv_fast => (
                    e.matmul_decode_exact_pre(&fa.wq, hq, hd, t)?,
                    e.matmul_decode_exact_pre(&fa.wk, hq, hd, t)?,
                    e.matmul_decode_exact_pre(&fa.wv, hq, hd, t)?,
                ),
                (None, _) => (
                    e.matmul_decode_exact(&fa.wq, h, t)?,
                    e.matmul_decode_exact(&fa.wk, h, t)?,
                    e.matmul_decode_exact(&fa.wv, h, t)?,
                ),
            }
        };
        // M3/Hy3 have no attention output gate — wq out is exactly q; skip the split.
        let gated = geometry.attention_gate == memra_gguf::config::AttentionGateKind::FusedQ;
        let (mut q, gate) = if gated {
            let mut q = vbuf(e, t * n_head * head_dim)?; // fully written by q_gate_split
            let mut gate = vbuf(e, t * n_head * head_dim)?; // fully written by q_gate_split
            e.q_gate_split(&qf, &mut q, &mut gate, head_dim, n_head, t)?;
            (q, Some(gate))
        } else {
            (qf, None)
        };

        let mut qn = vbuf(e, t * n_head * head_dim)?; // fully written by rms_norm
        e.rms_norm(
            &q,
            fa.q_norm.float_data(),
            &mut qn,
            head_dim,
            n_head * t,
            eps,
        )?;
        q = qn;
        let mut kn = vbuf(e, t * n_head_kv * head_dim)?; // fully written by rms_norm
        e.rms_norm(
            &k,
            fa.k_norm.float_data(),
            &mut kn,
            head_dim,
            n_head_kv * t,
            eps,
        )?;
        k = kn;
        let rope_dims = geometry.n_rot as usize;
        e.rope_neox(
            &mut q,
            pos_d,
            head_dim,
            rope_dims,
            n_head,
            t,
            geometry.rope_base,
            1.0,
        )?;
        e.rope_neox(
            &mut k,
            pos_d,
            head_dim,
            rope_dims,
            n_head_kv,
            t,
            geometry.rope_base,
            1.0,
        )?;

        // append T new K/V columns to the resident QUANTIZED cache. k/v are token-major [T, kv_dim]
        // f32; append-quantize each of the T token rows into the byte cache (q8_0 K / q5_1 V).
        let kvl = cache.kv[il].as_mut().unwrap();
        let (kv_dim_k, kv_dim_v, ktb, vtb) =
            (kvl.kv_dim_k, kvl.kv_dim_v, kvl.k_tok_bytes, kvl.v_tok_bytes);
        if let Some(ctr) = stream_ctr {
            // stream: ONE batched append at the device counter (rows kernel = the per-view warp
            // math on a (block, token) grid, documented byte-identical); host len is a stale
            // LOWER BOUND under pre-issue (drain reconciles it).
            e.append_kv_quantized_rows_dc(
                &k,
                &v,
                &mut kvl.k,
                &mut kvl.v,
                ctr,
                t,
                kv_dim_k,
                kv_dim_v,
                ktb,
                vtb,
                crate::Engine::kv_fp8_on(),
            )?;
        } else {
            for i in 0..t {
                let k_row = k.slice(i * kv_dim_k..(i + 1) * kv_dim_k);
                let v_row = v.slice(i * kv_dim_v..(i + 1) * kv_dim_v);
                e.append_kv_quantized_view(
                    &k_row,
                    &v_row,
                    &mut kvl.k,
                    &mut kvl.v,
                    kvl.len + i,
                    kv_dim_k,
                    kv_dim_v,
                    ktb,
                    vtb,
                    crate::Engine::kv_fp8_on(),
                )?;
            }
            kvl.len += t;
        }

        // BIT-IDENTICAL VERIFY ATTENTION (spec-exactness fix): the FP accumulation order must be
        // byte-for-byte identical to the eager decode path. fa_prefill uses a different tile size
        // (BLOCK_Q=64, BK=32) and online-softmax structure than fa_decode's split-K + combine,
        // which changes FP summation order and can flip argmax at tight logit margins. Query row r
        // attends to keys [0..base_len+r+1) — each successive row sees one more key (the causal
        // property). This matches eager: decode appends k at len, then fa_decode sees t_kv = len+1
        // keys. The verify appends all T tokens first but bounds the key range per row.
        //
        // MULTI-ROW FUSED PATH (the long-ctx spec fix, 2026-07-03): when every row takes the vec
        // kernel (base_len+1 >= FA_VEC_MIN_TKV), ONE fa_decode_rows launch executes the exact
        // per-row program for all T rows (grid.z = row, per-row n_splits from the same
        // fa_split_keys formula) — replacing T x (2 launches + 2 dtod copies + 5 partial allocs)
        // and multiplying resident CTAs by T on a latency-bound kernel. Bit-identical per row by
        // construction; kernel-check pins rows-vs-loop byte identity, run-spec is the end gate.
        // Short ctx (any row below the vec crossover) and MEMRA_NO_FA_VEC/MEMRA_FA_ROWS_OFF keep the
        // per-row loop (whose fa_decode picks scalar/vec per row exactly like eager decode).
        let mut attn = vbuf(e, t * n_head * head_dim)?; // fully written by every FA arm below
        let base_len = kvl.len - t; // KV len BEFORE this round's T tokens were appended
        // T=1 INCLUDED (2026-07-05): p-min cuts the draft to 1 in ~75% of rounds on hard
        // (agentic) content — the old t>1 gate sent those rounds to the per-row loop (262us/row
        // + q-row copy + per-row allocs vs 93us/row through the fused kernel at grid.z=1, same
        // program). nsys accounting: 1088 of 1456 verify FA launches were T=1 escapees.
        // LEAN T=1 ARM (MEMRA_SPEC_LEAN, close35): at t==1, q IS one row and fa_decode on it is
        // the EXACT eager decode dispatch (vec_q_v2 + combine_f32; the rows pair measured +50us
        // at m=1). Byte-identical: kernel-check pins rows-vs-loop identity, and the per-row loop
        // at t=1 is fa_decode on the same q with zero-offset copies. Gates arbitrate.
        if let Some(ctr) = stream_ctr {
            // STREAM ARM: causal base from the device counter; host kvl.len is a stale lower
            // bound used only for the split-sizing upper bound (+64 slack covers M pre-issued
            // rounds at K<=8). Views span the bound; per-row limits derive in-kernel.
            let upper = kvl.len + t + 64;
            let k_view = e.view_u8(&kvl.k, (upper.min(cache.max_ctx)) * ktb);
            let v_view = e.view_u8(&kvl.v, (upper.min(cache.max_ctx)) * vtb);
            e.fa_decode_rows_dc(
                &q,
                &k_view,
                &v_view,
                &mut attn,
                head_dim,
                n_head,
                n_head_kv,
                ctr,
                upper.min(cache.max_ctx),
                t,
                scale,
                ktb,
                vtb,
                0,
                false,
            )?;
        } else if spec_lean() && t == 1 {
            let t_kv = base_len + 1;
            let k_view = e.view_u8(&kvl.k, t_kv * ktb);
            let v_view = e.view_u8(&kvl.v, t_kv * vtb);
            e.fa_decode_kvmod(
                &q,
                &k_view,
                &v_view,
                &mut attn,
                head_dim,
                n_head,
                n_head_kv,
                t_kv,
                scale,
                ktb,
                vtb,
                crate::Engine::kv_fp8_on(),
            )?;
        } else if e.fa_rows_eligible(base_len, head_dim) {
            let k_view = e.view_u8(&kvl.k, (base_len + t) * ktb);
            let v_view = e.view_u8(&kvl.v, (base_len + t) * vtb);
            e.fa_decode_rows(
                &q,
                &k_view,
                &v_view,
                &mut attn,
                head_dim,
                n_head,
                n_head_kv,
                base_len,
                t,
                scale,
                ktb,
                vtb,
                None,
                false,
                crate::Engine::kv_fp8_on(),
                None,
            )?;
        } else {
            for r in 0..t {
                let t_kv_r = base_len + r + 1; // this row sees keys [0..t_kv_r)
                let k_view_r = e.view_u8(&kvl.k, t_kv_r * ktb);
                let v_view_r = e.view_u8(&kvl.v, t_kv_r * vtb);
                // copy q row into an owned buffer (fa_decode takes &CudaSlice, not CudaView)
                let mut q_row = vbuf(e, n_head * head_dim)?; // fully written by copy_view_into
                let q_src = q.slice(r * n_head * head_dim..(r + 1) * n_head * head_dim);
                e.copy_view_into(&mut q_row, 0, &q_src, n_head * head_dim)?;
                let mut attn_row = vbuf(e, n_head * head_dim)?; // fully written by fa_decode
                e.fa_decode_kvmod(
                    &q_row,
                    &k_view_r,
                    &v_view_r,
                    &mut attn_row,
                    head_dim,
                    n_head,
                    n_head_kv,
                    t_kv_r,
                    scale,
                    ktb,
                    vtb,
                    crate::Engine::kv_fp8_on(),
                )?;
                e.copy_into(
                    &mut attn,
                    r * n_head * head_dim,
                    &attn_row,
                    n_head * head_dim,
                )?;
            }
        }

        let attn_g = match &gate {
            Some(gate) => {
                let mut gsig = vbuf(e, t * n_head * head_dim)?; // fully written by sigmoid
                e.sigmoid(gate, &mut gsig, t * n_head * head_dim)?;
                let mut ag = vbuf(e, t * n_head * head_dim)?; // fully written by mul
                e.mul(&attn, &gsig, &mut ag, t * n_head * head_dim)?;
                ag
            }
            None => attn,
        };
        // DECODE-EXACT wo projection: at m>=5 (K=4+ with pending) the generic matmul would use dp4a
        // (128-thread, different FP sum order than MMVQ). Force MMVQ for bit-identity with decode.
        match self.full_attn_tp_o(e, fa, &attn_g, t)? {
            Some(output) => Ok(output),
            None => Ok(e.matmul_decode_exact(&fa.wo, &attn_g, t)?),
        }
    }

    /// Context-linear bytes for a plain serving session's trunk cache.
    pub fn plain_session_kv_bytes_per_token(&self) -> usize {
        crate::cache::cache_bytes_per_token_for_plan(
            &self.cfg,
            &self.plan,
            0,
            self.plan.layers.len(),
        )
    }

    /// `(logical bytes/token, ring-capped bytes/token, ring row cap)` for exact admission.
    pub fn plain_session_kv_shape(&self) -> (usize, usize, usize) {
        (
            self.plain_session_kv_bytes_per_token(),
            crate::cache::cache_ring_bytes_per_token_for_plan(
                &self.cfg,
                &self.plan,
                0,
                self.plan.layers.len(),
            ),
            crate::cache::cache_ring_row_cap_for_plan(&self.plan),
        )
    }

    /// Context-linear bytes for a speculative serving session: trunk cache plus persistent MTP
    /// scratch. With no MTP head this equals the plain coefficient.
    pub fn spec_session_kv_bytes_per_token(&self) -> usize {
        let scratch = self
            .mtp
            .iter()
            .chain(self.mtp_extra.iter())
            .map(|mtp| {
                let (_, _, k, v) = mtp_scratch_layout(&self.cfg, mtp.geom.as_ref());
                k + v
            })
            .sum::<usize>();
        self.plain_session_kv_bytes_per_token()
            .saturating_add(scratch)
    }

    /// Spec twin of [`HybridModel::plain_session_kv_shape`]; Step35's persistent MTP scratch is
    /// capped by the same SWA ring rows as the trunk.
    pub fn spec_session_kv_shape(&self) -> (usize, usize, usize) {
        let total = self.spec_session_kv_bytes_per_token();
        let (_, mut ring, rows) = self.plain_session_kv_shape();
        if rows > 0 {
            ring = ring.saturating_add(
                self.mtp
                    .iter()
                    .chain(self.mtp_extra.iter())
                    .map(|mtp| {
                        let (_, _, k, v) = mtp_scratch_layout(&self.cfg, mtp.geom.as_ref());
                        k + v
                    })
                    .sum::<usize>(),
            );
        }
        (total, ring, rows)
    }

    /// Greedy MTP speculative decode (§B). Token-identical to `generate(prompt, max_new)` but uses
    /// the NextN head to draft K tokens then verifies them in one batched target forward.
    /// Returns (generated tokens, total_drafted, total_accepted) so the caller can report
    /// acceptance rate. `k` = draft length per round.
    ///
    /// GRAPH DRAFT (stage 2 of graph-grade spec): when the model is all-Dense and the MTP head is
    /// Dense (no MoE host readbacks), the fixed-shape T=1 MTP forward is CUDA-graph-captured ONCE
    /// and replayed per draft step — the ~40 eager launches per drafted token collapse into one
    /// graph dispatch; only the 4-byte token id (and 4-byte p-min confidence) round-trip per step.
    /// Event tracking is disabled for the whole call (generate_graph pattern) so every buffer the
    /// captured graph references is event-free; the spec loop is strictly single-stream.
    /// MEMRA_SPEC_NOGRAPH=1 forces the eager draft chain.
    /// SAMPLED mode (MEMRA_SPEC_TEMP>0) has its OWN capture (gumbel-perturbed in-graph argmax,
    /// device Philox event counter, persistent q retention) — graph-vs-eager sampled streams are
    /// bit-identical for the same (seed, prompt, K, temp); see the sampled-graph setup in
    /// generate_spec_inner2.
    /// Multi-turn session: trunk cache + MTP draft scratch persist across generate calls, so
    /// turn N+1 primes ONLY its new suffix (the 124k-conversation daily pattern — re-priming a
    /// 32k history costs ~54s; a suffix prime costs seconds). APPEND-ONLY by construction: the
    /// hybrid linear-attn states are in-place (no position index), so a session can extend but
    /// never rewind — `committed` is the exact token list whose state the caches hold (includes
    /// any overshoot tokens past max_new; the caller renders from `committed`, not its own echo).
    pub fn new_session(
        &self,
        e: &Engine,
        max_ctx: usize,
    ) -> Result<SpecSession, Box<dyn std::error::Error>> {
        Ok(SpecSession {
            // STAGE-OWNED KV (lane/pp2-spec 2026-08-06): `pp::new_cache`, not `Cache::new`. This
            // is the SERVING spec-session path, and with the ppN door open across two cards a
            // primary-homed cache makes every remote stage peer-read its OWN KV on every verify
            // round — the wrong-card class already fixed on the two batched serving paths
            // (worker.rs 2483 / 2837). With the door shut `new_cache` IS `Cache::new` (same
            // branch, same allocations), so single-device behavior is byte-unchanged.
            cache: crate::pp::new_cache_planned(e, &self.cfg, &self.plan, max_ctx)?,
            scratch: self.new_mtp_scratch(e, max_ctx)?,
            committed: Vec::new(),
            last_h: None,
            next_pred: None,
            sctr: 0,
            uctr: 0,
            draft_ctx: None,
            pending_tok: None,
            turn_ckpt: None,
            telem: SpecTelemetryCounters::default(),
            capture_at: None,
            boundary_captures: Vec::new(),
            ckpt_at: None,
            capture_disabled: false,
        })
    }

    /// SPEC-ON-CACHE-HIT restore (lane/spec-on-cache-hit, 2026-08-18 — PORT-PLAN item 3,
    /// research/cache-spec-design-20260814, scoped to WHOLE-ENTRY restores only): build a
    /// SpecSession around a trunk cache the worker already restored from a prefix-cache
    /// entry, re-installing the entry's published draft plane as the MTP scratch rows
    /// `[0..prefix.len())` and the entry's boundary hidden as `last_h`, then feeding the
    /// prompt SUFFIX here — through EXACTLY the plain path's program selection — so the
    /// worker always receives a fully-warm continuation session (committed = whole
    /// prompt, `next_pred` + `last_h` set; caller sets `next_pred` from the entry's
    /// boundary logits on the empty-suffix shape).
    ///
    /// PROGRAM LAW (the splitiso two-programs class, learned AGAIN in this lane's own
    /// gate): the identity target for a converted hit is the PLAIN hit serving the same
    /// request, and plain feeds a carried suffix via eager `decode_step` below
    /// PRIME_MIN_T and via `prime_cache` at/above it (prefill_tick's arms). The generate
    /// path's tokenwise arm routes qwen35-class through the BATCHED T=1 program
    /// (`spec_target_step_h`) instead — ULP-different suffix rows, and the gate measured
    /// the near-tie flip at generated token ~8 (research/spec-cache-20260818, qwen r3).
    /// So the suffix is fed HERE, mirroring prefill_tick arm-for-arm, not handed to the
    /// burst prime.
    ///
    /// SEED RULE (both sampling regimes; lane/sampled-hit-spec 2026-08-19, sampled draw
    /// added by lane/sampled-spec-quality 2026-08-19). The boundary token is produced by
    /// EXACTLY the rule the cold burst entry applies to its own first token from the same
    /// logits row: `argmax` when greedy, and a `sample_boundary_token` draw at Philox
    /// counter 0 when sampled. Both shapes are covered — the entry's boundary logits on a
    /// full-cover (empty-suffix) hit, this feed's own boundary logits on a suffix hit.
    /// That is what keeps a restored session seed-identical to a cold one PER SEED: the
    /// cold session draws from the identical row at counter 0 and then runs its rounds from
    /// counter 1, so the restored session admits with `sctr = 1` after its own draw.
    /// The WORKER owns the one refusal this constructor cannot see — a constrained request.
    /// (The penalized-sampled refusal was LIFTED once the burst's penalty window learned to
    /// span the session: `committed` here is the WHOLE prompt, so the restored session's
    /// window is the cold session's window. It comes back if `MEMRA_SPEC_PEN_SESSION=0`.)
    ///
    /// NOT the rolled-back partial-restore hazard: the caller restores at exactly the
    /// entry's captured endpoint (`e.pos`) through the shipping whole-entry path;
    /// mid-entry (`at < e.pos`) trunk restores stay behind MEMRA_PREFIX_PARTIAL_RESTORE
    /// and are never routed here.
    ///
    /// Failure contract: `Err((Some(cache), why))` before any trunk mutation — the
    /// worker rebuilds the plain carrier and the hit serves plain, byte-unchanged.
    /// `Err((None, why))` after the suffix feed began — the carrier is part-fed and
    /// UNUSABLE; the worker serves the request cold-plain (correct, slower) and the
    /// entry stays published for the next request.
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::result_large_err)] // allow: the fat error type is the diagnostic contract here; boxing it would change the error surface
    pub fn spec_session_from_restored(
        &self,
        e: &Engine,
        mut cache: Cache,
        prefix: Vec<u32>,
        suffix: &[u32],
        draft_k: &CudaSlice<u8>,
        draft_v: &CudaSlice<u8>,
        draft_k_tok_bytes: usize,
        draft_v_tok_bytes: usize,
        draft_len: usize,
        last_h: &[f32],
        // The ENTRY's boundary logits row (the full-cover shape's seed source). May be empty
        // when a suffix follows — the feed's own logits are the boundary then.
        boundary_logits: &[f32],
        // The request's sampler, or None for greedy. Owned here so the seed rule lives in
        // ONE place instead of being half-applied by the worker.
        sampling: Option<SpecSampling>,
        require_anchor: bool,
        max_ctx: usize,
        // STABLE-BOUNDARY REPUBLICATION (lane/frspec-multiturn-cache, 2026-08-21): ABSOLUTE
        // prompt position to split the suffix feed at and capture the extended-entry
        // publication + this session's `turn_ckpt` — the worker's stable pre-generation
        // boundary (`plain_checkpoint_boundary`). None = legacy prompt-end republication.
        // WHY: the prompt-end capture below includes the template's live generation header
        // (`<|im_start|>assistant\n<think>\n`), which the next turn's re-render replaces, so
        // for a hybrid (whole-entry restores only) every extended entry's last ~2 tokens
        // diverged from every future prompt and the hit boundary FROZE at the first
        // lcp-split entry forever (measured: cached 6811 of 38228 by turn 8, B4).
        republish_at: Option<usize>,
    ) -> Result<SpecSession, (Option<Cache>, String)> {
        let pos = prefix.len();
        let fail = |cache: Cache, msg: String| -> Result<SpecSession, (Option<Cache>, String)> {
            Err((Some(cache), msg))
        };
        if let Err(error) = cache.ensure_usable("spec_session_from_restored") {
            drop(cache);
            return Err((None, error.to_string()));
        }
        if self.mtp.is_none() {
            return fail(cache, "no MTP head attached (nothing to draft with)".into());
        }
        if pos == 0 {
            return fail(cache, "empty committed prefix".into());
        }
        if cache.pos != pos {
            let msg = format!(
                "restored cache pos {} != restored prefix len {pos}",
                cache.pos
            );
            return fail(cache, msg);
        }
        if draft_len != pos {
            return fail(
                cache,
                format!("draft plane len {draft_len} != restored prefix len {pos}"),
            );
        }
        if pos + suffix.len() >= max_ctx {
            return fail(
                cache,
                format!(
                    "prompt {} + suffix would not leave generation room in ctx {max_ctx}",
                    pos + suffix.len(),
                ),
            );
        }
        let mut scratch = match MtpScratch::new(
            e,
            &self.cfg,
            &self.plan,
            max_ctx,
            self.mtp.as_ref().and_then(|m| m.geom.as_ref()),
        ) {
            Ok(s) => s,
            Err(err) => return fail(cache, format!("draft scratch alloc failed: {err}")),
        };
        if scratch.kv.ring.is_some() {
            return fail(
                cache,
                "ring-backed draft scratch (Step35 SWA) cannot take a flat prefix restore".into(),
            );
        }
        if scratch.kv.k_tok_bytes != draft_k_tok_bytes
            || scratch.kv.v_tok_bytes != draft_v_tok_bytes
        {
            return fail(
                cache,
                format!(
                    "draft plane layout {draft_k_tok_bytes}/{draft_v_tok_bytes} != scratch \
                     {}/{} bytes/token (stale entry across a format change)",
                    scratch.kv.k_tok_bytes, scratch.kv.v_tok_bytes,
                ),
            );
        }
        if pos > scratch.cap {
            return fail(
                cache,
                format!(
                    "draft plane rows {pos} exceed scratch capacity {}",
                    scratch.cap
                ),
            );
        }
        let kb = pos * draft_k_tok_bytes;
        let vb = pos * draft_v_tok_bytes;
        if draft_k.len() < kb || draft_v.len() < vb {
            return fail(
                cache,
                format!(
                    "truncated draft plane: K {} < {kb} or V {} < {vb} bytes",
                    draft_k.len(),
                    draft_v.len(),
                ),
            );
        }
        if kb > 0
            && let Err(err) = e.copy_u8_into(&mut scratch.kv.k, 0, draft_k, kb)
        {
            return fail(cache, format!("draft K restore copy failed: {err}"));
        }
        if vb > 0
            && let Err(err) = e.copy_u8_into(&mut scratch.kv.v, 0, draft_v, vb)
        {
            return fail(cache, format!("draft V restore copy failed: {err}"));
        }
        if let Err(err) = scratch.set_len(e, pos) {
            return fail(cache, format!("draft scratch len set failed: {err}"));
        }
        let mut last_h_dev = if last_h.len() == self.cfg.n_embd as usize {
            // anchor upload failure is acceptance-only when a suffix feed follows (fill
            // row-0 falls back to zeros) but FATAL for an empty-suffix continuation (the
            // burst entry asserts committed + last_h + next_pred) — the caller says which.
            e.htod(last_h).ok()
        } else {
            None
        };
        if require_anchor && last_h_dev.is_none() {
            return fail(
                cache,
                "empty-suffix continuation requires the entry's boundary hidden anchor".into(),
            );
        }
        let mut committed = prefix;
        // Set on BOTH shapes below (suffix-fed and full-cover) — never left None, which is
        // what the empty-suffix continuation assert in the burst entry requires.
        let next_pred;
        // Philox: (0,0) at admit exactly like a fresh session; a sampled boundary draw below
        // consumes counter 0 and leaves 1, which is the state a cold session reaches after
        // drawing its own first token from the same row.
        let mut sctr = 0u32;
        let sampled = sampling.is_some_and(|s| s.temp > 0.0) && spec_sampled_boundary_on();
        // Penalty window for the boundary draw: the last `penalty_last_n` tokens of the WHOLE
        // prompt, which is what the cold session's own burst sees (Item 2's window). Built
        // after the suffix joins `committed` below.
        let mut boundary_captures: Vec<SpecBoundaryCapture> = Vec::new();
        let mut restored_turn_ckpt: Option<SpecCheckpoint> = None;
        if !suffix.is_empty() {
            // ---- SUFFIX FEED, mirroring prefill_tick's program selection exactly ----
            // From here on the trunk cache mutates: failures return Err((None, _)) and
            // the worker serves the request cold-plain instead of reusing the carrier.
            let dirty =
                |msg: String| -> Result<SpecSession, (Option<Cache>, String)> { Err((None, msg)) };
            let n_embd = self.cfg.n_embd as usize;
            let t = suffix.len();
            let mut h_rows = match e.uninit(t * n_embd) {
                Ok(b) => b,
                Err(err) => return fail(cache, format!("suffix hidden buffer alloc: {err}")),
            };
            // STABLE-BOUNDARY split (see `republish_at`): feed stops at the boundary so the
            // in-place GDN conv/ssm state can be snapshotted there — the only moment it
            // exists (the cold prime-split law). suffix-relative; None = one-segment legacy.
            let b_rel = republish_at
                .and_then(|abs| abs.checked_sub(pos))
                .filter(|&r| r > 0 && r < t);
            let mut feed_logits = Vec::new();
            let tokenwise_env = std::env::var("MEMRA_PRIME_TOKENWISE").is_ok()
                || e.frozen_cpu_experts_prefer_tokenwise_prime();
            let mut fed = 0usize;
            for seg_end in [b_rel, Some(t)].into_iter().flatten() {
                if seg_end <= fed {
                    continue;
                }
                let seg = &suffix[fed..seg_end];
                let batched = seg.len() >= crate::hybrid_forward::PRIME_MIN_T && !tokenwise_env;
                if batched {
                    // prefill_tick's prime arm: request-level prime_cache call; tokens still
                    // queued after this segment ride `queued_after` so Step35 arm selection
                    // stays keyed to the request's end (tick-seg law).
                    match self.prime_cache(e, seg, &mut cache, t - seg_end) {
                        Ok((l, _h_seed, hiddens)) => {
                            if let Err(err) =
                                e.copy_into(&mut h_rows, fed * n_embd, &hiddens, seg.len() * n_embd)
                            {
                                return dirty(format!("suffix hidden copy: {err}"));
                            }
                            feed_logits = l;
                        }
                        Err(err) => return dirty(format!("suffix prime failed: {err}")),
                    }
                } else {
                    // prefill_tick's tokenwise arm: eager decode_step, one token at a time.
                    for (i, &tok) in seg.iter().enumerate() {
                        match self.decode_step_h(e, tok, &mut cache) {
                            Ok((l, h)) => {
                                if let Err(err) =
                                    e.copy_into(&mut h_rows, (fed + i) * n_embd, &h, n_embd)
                                {
                                    return dirty(format!("suffix hidden copy: {err}"));
                                }
                                feed_logits = l;
                            }
                            Err(err) => return dirty(format!("suffix decode_step failed: {err}")),
                        }
                    }
                }
                fed = seg_end;
                if Some(seg_end) == b_rel {
                    // The stable pre-generation boundary: capture the extended-entry
                    // publication AND this session's own turn checkpoint here instead of at
                    // prompt-end (both would otherwise carry the volatile live-header tail
                    // the next re-render replaces). Failure silent, turn_ckpt convention.
                    debug_assert_eq!(
                        cache.pos,
                        pos + seg_end,
                        "stable-boundary capture off the feed split"
                    );
                    if spec_restore_republish_on()
                        && let Ok(snap) = cache.snapshot(e)
                    {
                        boundary_captures.push(SpecBoundaryCapture {
                            snap,
                            pos: pos + seg_end,
                            logits: feed_logits.clone(),
                            last_h: capture_boundary_hidden(e, &h_rows, seg_end, n_embd),
                            latent_tails: Vec::new(),
                        });
                    }
                    let anchor: Result<CudaSlice<f32>, Box<dyn std::error::Error>> =
                        e.uninit(n_embd).and_then(|mut a| {
                            e.copy_view_into(
                                &mut a,
                                0,
                                &h_rows.slice((seg_end - 1) * n_embd..seg_end * n_embd),
                                n_embd,
                            )?;
                            Ok(a)
                        });
                    if let (Ok(snap), Ok(last_h)) = (cache.snapshot(e), anchor) {
                        restored_turn_ckpt = Some(SpecCheckpoint {
                            snap,
                            pos: pos + seg_end,
                            last_h,
                        });
                    }
                }
            }
            // Draft-scratch fill for the suffix rows, predecessor-paired: row `pos` reads
            // the entry's boundary anchor (zeros fallback — acceptance-only), row `pos+i`
            // reads h_rows[i-1]. Chunked like the generate path's fill (transients scale
            // with T). Fill failures are acceptance-only — truncate to the restored rows
            // and continue; the burst's own set_len keeps the invariant.
            let _mtp = self.mtp.as_ref().expect("mtp checked above"); // invariant check only; the fill below re-reads self.mtp
            let (embd_qt, embd_rb) = self.embd.qt_and_row_bytes(n_embd);
            let embd_gpu = if spec_host_embd() {
                None
            } else {
                Some(
                    self.embd_gpu
                        .get_or_init(|| e.upload_u8(&self.embd.raw).expect("embed table upload")),
                )
            };
            let embd_dev = embd_gpu.map(|g| (g, embd_qt, embd_rb));
            let fill_chunk = 4096usize;
            let mut filled = true;
            let mut start = 0usize;
            'fill: while start < t {
                let end = (start + fill_chunk).min(t);
                let tc = end - start;
                let Ok(mut phs) = e.zeros(tc * n_embd) else {
                    filled = false;
                    break 'fill;
                };
                let (src_lo, dst_off, n_copy) = if start == 0 {
                    (0, n_embd, (tc - 1) * n_embd)
                } else {
                    ((start - 1) * n_embd, 0, tc * n_embd)
                };
                if start == 0
                    && let Some(lh) = last_h_dev.as_ref()
                    && e.copy_into(&mut phs, 0, lh, n_embd).is_err()
                {
                    filled = false;
                    break 'fill;
                }
                if n_copy > 0
                    && e.copy_view_into(
                        &mut phs,
                        dst_off,
                        &h_rows.slice(src_lo..src_lo + n_copy),
                        n_copy,
                    )
                    .is_err()
                {
                    filled = false;
                    break 'fill;
                }
                if self
                    .mtp_kv_fill_all(
                        e,
                        &suffix[start..end],
                        &phs,
                        pos + start,
                        &mut scratch,
                        embd_dev,
                    )
                    .is_err()
                {
                    filled = false;
                    break 'fill;
                }
                start = end;
            }
            if !filled {
                // acceptance-only: drafts over missing suffix rows are cheap and wrong,
                // so keep only the restored rows resident and let verify arbitrate.
                if let Err(err) = scratch.set_len(e, pos) {
                    return dirty(format!("scratch truncation after failed fill: {err}"));
                }
            }
            // EXTENDED-ENTRY PUBLICATION (lane/sampled-spec-quality, Item 3 — the fix for
            // "a restored spec session never publishes an extended entry", SAMPLED-HIT.md
            // finding (d)). Pre-lane, publication was armed only for COLD sessions
            // (`spec_resumed == 0` in the worker) and both engine capture sites require a
            // non-continuation burst — but a converted hit's first burst IS a continuation,
            // so a growing conversation learned exactly ONE boundary and turn 3 could never
            // hit a longer prefix than turn 2 did.
            //
            // WHERE, and why it is safe here: `cache.pos == prefix + suffix` at this exact
            // line — the trunk is primed over the whole prompt, nothing is generated, and the
            // draft plane rows [0..prompt) are filled just above. That is a complete
            // whole-entry boundary (`pos == fed_len`), the same shape the cold seed capture
            // publishes; the worker's existing publication sweep picks it up because it is
            // keyed on non-empty `boundary_captures` and is sampler- and resume-independent.
            // NOT the partial-restore hazard: the boundary is this session's own prompt END,
            // never mid-entry, so `entry_pos != fed_len` still refuses on the way back in.
            // Failure is SILENT by design (the turn_ckpt / boundary-capture convention):
            // publication is an optimization, never a correctness dependency.
            //
            // SUPERSEDED WHEN `republish_at` FIRED (lane/frspec-multiturn-cache): a prompt-end
            // entry's tail is the live generation header the next re-render replaces, so on a
            // hybrid (whole-entry restores) it can never serve the conversation's next turn —
            // the stable-boundary capture above IS this publication, minus the poisoned tail.
            if spec_restore_republish_on() && boundary_captures.is_empty() {
                debug_assert_eq!(
                    cache.pos,
                    pos + t,
                    "extended-entry capture must sit at the restored session's prompt end",
                );
                if let Ok(snap) = cache.snapshot(e) {
                    boundary_captures.push(SpecBoundaryCapture {
                        snap,
                        pos: pos + t,
                        logits: feed_logits.clone(),
                        last_h: capture_boundary_hidden(e, &h_rows, t, n_embd),
                        latent_tails: Vec::new(),
                    });
                }
            }
            // continuation seed: the feed's boundary logits ARE the plain path's boundary
            // logits (same program), so greedy's argmax here is plain's first emitted token,
            // and the sampled draw is the cold sampled session's own first token.
            next_pred = Some(if sampled {
                let sp = sampling.expect("sampled implies a sampler");
                // `committed` is still the restored prefix here; the suffix joins it below —
                // so this is the last-N window over the WHOLE prompt, exactly the cold
                // session's own window at its first token.
                let hist = pen_window_seed(&committed, suffix, sp.penalty_last_n);
                match sample_boundary_token(
                    e,
                    &feed_logits,
                    &sp,
                    &hist,
                    &mut sctr,
                    "restore-suffix-feed",
                ) {
                    Ok(t) => t,
                    // the trunk is already fed: hand nothing back, the worker serves the
                    // request cold-plain. Never fall back to an argmax — that would put a
                    // greedy token in a sampled stream to save a slow path.
                    Err(err) => {
                        return dirty(format!("boundary token draw failed: {err}"));
                    }
                }
            } else {
                argmax(&feed_logits) as u32
            });
            let mut lh = match e.uninit(n_embd) {
                Ok(b) => b,
                Err(err) => return dirty(format!("boundary hidden alloc: {err}")),
            };
            if let Err(err) = e.copy_view_into(
                &mut lh,
                0,
                &h_rows.slice((t - 1) * n_embd..t * n_embd),
                n_embd,
            ) {
                return dirty(format!("boundary hidden copy: {err}"));
            }
            last_h_dev = Some(lh);
            committed.extend_from_slice(suffix);
        } else {
            // FULL-COVER shape (empty suffix — the identical-repeat / agent-loop shape): the
            // ENTRY's boundary logits are the boundary row, and this is the token the cold
            // session emits from that same row. Owned here rather than in the worker so the
            // sampled draw cannot be half-applied on one shape (the worker used to argmax it).
            if boundary_logits.is_empty() {
                return fail(
                    cache,
                    "full-cover restore without the entry's boundary logits".into(),
                );
            }
            next_pred = Some(if sampled {
                let sp = sampling.expect("sampled implies a sampler");
                let hist = pen_window_seed(&committed, &[], sp.penalty_last_n);
                match sample_boundary_token(
                    e,
                    boundary_logits,
                    &sp,
                    &hist,
                    &mut sctr,
                    "restore-full-cover",
                ) {
                    Ok(t) => t,
                    // nothing has been mutated on this shape — hand the carrier back and let
                    // the hit serve PLAIN (the banked pre-lane path).
                    Err(err) => {
                        return fail(cache, format!("boundary token draw failed: {err}"));
                    }
                }
            } else {
                argmax(boundary_logits) as u32
            });
        }
        Ok(SpecSession {
            cache,
            scratch,
            committed,
            last_h: last_h_dev,
            next_pred,
            sctr,
            uctr: 0,
            draft_ctx: None,
            pending_tok: None,
            // Stable-boundary capture from the split feed above (None on the legacy shape):
            // a restored session previously parked WITHOUT a checkpoint, so the next turn's
            // affinity probe declined ("no turn checkpoint retained") and the conversation
            // fell back to the frozen prefix entry forever.
            turn_ckpt: restored_turn_ckpt,
            telem: SpecTelemetryCounters::default(),
            capture_at: None,
            boundary_captures,
            ckpt_at: None,
            capture_disabled: false,
        })
    }

    /// Forced-gate exact state comparison. This intentionally reads the real live prefixes from
    /// their owning PP devices: matching emitted ids alone would miss a stale `len_d`, recurrent
    /// snapshot, or draft-KV row that only corrupts the following round.
    pub fn optipipe_compare_session_state(
        &self,
        e: &Engine,
        reference: &SpecSession,
        candidate: &SpecSession,
    ) -> Result<OptiForkStateIdentity, Box<dyn std::error::Error>> {
        fn fail(what: &str) -> Box<dyn std::error::Error> {
            format!("optipipe state mismatch: {what}").into()
        }
        fn same_f32(a: &[f32], b: &[f32]) -> bool {
            a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
        }
        fn compare_layers(
            es: &Engine,
            range: std::ops::Range<usize>,
            reference: &SpecSession,
            candidate: &SpecSession,
            report: &mut OptiForkStateIdentity,
        ) -> Result<(), Box<dyn std::error::Error>> {
            for il in range {
                match (&reference.cache.kv[il], &candidate.cache.kv[il]) {
                    (Some(a), Some(b)) => {
                        if a.len != b.len {
                            return Err(fail(&format!(
                                "layer {il} host KV len {} != {}",
                                a.len, b.len
                            )));
                        }
                        let ad = es.dtoh_i32(&a.len_d)?;
                        let bd = es.dtoh_i32(&b.len_d)?;
                        if ad != bd || ad.first().copied() != Some(a.len as i32) {
                            return Err(fail(&format!(
                                "layer {il} device KV len {ad:?} != {bd:?} (host={})",
                                a.len,
                            )));
                        }
                        let kb = a.len * a.k_tok_bytes;
                        let vb = a.len * a.v_tok_bytes;
                        if kb > 0 {
                            let ak = es.dtoh_u8_view(&a.k.slice(0..kb))?;
                            let bk = es.dtoh_u8_view(&b.k.slice(0..kb))?;
                            if ak != bk {
                                let at = ak.iter().zip(&bk).position(|(x, y)| x != y).unwrap();
                                return Err(fail(&format!(
                                    "layer {il} K bytes at byte {at} row {} offset {}: {} != {}",
                                    at / a.k_tok_bytes,
                                    at % a.k_tok_bytes,
                                    ak[at],
                                    bk[at],
                                )));
                            }
                        }
                        if vb > 0 {
                            let av = es.dtoh_u8_view(&a.v.slice(0..vb))?;
                            let bv = es.dtoh_u8_view(&b.v.slice(0..vb))?;
                            if av != bv {
                                let at = av.iter().zip(&bv).position(|(x, y)| x != y).unwrap();
                                return Err(fail(&format!(
                                    "layer {il} V bytes at byte {at} row {} offset {}: {} != {}",
                                    at / a.v_tok_bytes,
                                    at % a.v_tok_bytes,
                                    av[at],
                                    bv[at],
                                )));
                            }
                        }
                        report.trunk_kv_bytes += kb + vb;
                    }
                    (None, None) => {}
                    _ => return Err(fail(&format!("layer {il} KV presence"))),
                }
                match (&reference.cache.recur[il], &candidate.cache.recur[il]) {
                    (Some(a), Some(b)) => {
                        let ac = es.dtoh(&a.conv_state)?;
                        let bc = es.dtoh(&b.conv_state)?;
                        if !same_f32(&ac, &bc) {
                            return Err(fail(&format!("layer {il} conv state")));
                        }
                        let as_ = es.dtoh(&a.ssm_state)?;
                        let bs = es.dtoh(&b.ssm_state)?;
                        if !same_f32(&as_, &bs) {
                            return Err(fail(&format!("layer {il} SSM state")));
                        }
                        report.recurrent_bytes += (ac.len() + as_.len()) * 4;
                    }
                    (None, None) => {}
                    _ => return Err(fail(&format!("layer {il} recurrent presence"))),
                }
            }
            Ok(())
        }

        if reference.committed != candidate.committed {
            return Err(fail("committed token ids"));
        }
        if reference.cache.pos != candidate.cache.pos
            || reference.cache.max_ctx != candidate.cache.max_ctx
        {
            return Err(fail("cache pos/capacity"));
        }
        if reference.pending_tok != candidate.pending_tok
            || reference.next_pred != candidate.next_pred
            || reference.sctr != candidate.sctr
            || reference.uctr != candidate.uctr
        {
            return Err(fail("pending/prediction/counter tail"));
        }

        let mut report = OptiForkStateIdentity::default();
        if let Some(fence) = crate::pp::pp_cuts(self.layers.len()) {
            let rt = crate::pp::PpNRt::get(e)?;
            for stage in 0..rt.n_stages() {
                let _scope = rt.enter(stage);
                compare_layers(
                    rt.engine(stage, e),
                    fence[stage]..fence[stage + 1],
                    reference,
                    candidate,
                    &mut report,
                )?;
            }
        } else {
            compare_layers(e, 0..self.layers.len(), reference, candidate, &mut report)?;
        }

        if reference.scratch.plane_count() != candidate.scratch.plane_count() {
            return Err(fail("draft scratch plane count"));
        }
        for index in 0..reference.scratch.plane_count() {
            let (a, _) = reference.scratch.plane(index);
            let (b, _) = candidate.scratch.plane(index);
            if a.len != b.len
                || a.kv_dim_k != b.kv_dim_k
                || a.kv_dim_v != b.kv_dim_v
                || a.k_tok_bytes != b.k_tok_bytes
                || a.v_tok_bytes != b.v_tok_bytes
                || e.dtoh_i32(&a.len_d)? != e.dtoh_i32(&b.len_d)?
            {
                return Err(fail(&format!("draft scratch plane {index} length/layout")));
            }
            let kb = a.len * a.k_tok_bytes;
            let vb = a.len * a.v_tok_bytes;
            if kb > 0 && e.dtoh_u8_view(&a.k.slice(0..kb))? != e.dtoh_u8_view(&b.k.slice(0..kb))? {
                return Err(fail(&format!("draft scratch plane {index} K bytes")));
            }
            if vb > 0 && e.dtoh_u8_view(&a.v.slice(0..vb))? != e.dtoh_u8_view(&b.v.slice(0..vb))? {
                return Err(fail(&format!("draft scratch plane {index} V bytes")));
            }
            report.scratch_kv_bytes += kb + vb;
        }

        match (&reference.last_h, &candidate.last_h) {
            (Some(a), Some(b)) => {
                let ah = e.dtoh(a)?;
                let bh = e.dtoh(b)?;
                if !same_f32(&ah, &bh) {
                    return Err(fail("last hidden/seed bytes"));
                }
                report.hidden_bytes = ah.len() * 4;
            }
            (None, None) => {}
            _ => return Err(fail("last hidden/seed presence")),
        }
        Ok(report)
    }

    /// SESSION-AFFINITY REWIND (lane/session-affinity, 2026-08-05): roll `sess` back to its
    /// retained prompt-end checkpoint, so a request whose prompt matches
    /// `committed[..rewind_pos()]` exactly can resume there and prime only its own delta.
    ///
    /// EXACTNESS. After this returns, the session is byte-for-byte the state it was in AT that
    /// boundary: full-attn KV truncated to it (append-only, position-addressed), GDN conv/ssm
    /// restored from the device copy taken there, draft scratch length reset, `committed`
    /// truncated, `last_h` = the boundary's predecessor anchor. That is precisely the state a
    /// fresh prime of `committed[..pos]` would have produced, so the following suffix prime and
    /// every burst after it are identical to a cold run of the same token stream — the
    /// committed-tokens-authoritative contract.
    ///
    /// `next_pred` and `pending_tok` are CLEARED: both describe generation past the boundary,
    /// which the rewind discards. The caller therefore must supply a non-empty suffix (a
    /// rewound session cannot serve an empty-suffix continuation burst — there is nothing to
    /// continue). The persistent draft graph survives: it bakes only session-stable pointers
    /// (the scratch KV, the resident embedding), none of which the rewind moves.
    ///
    /// The checkpoint is CONSUMED (`turn_ckpt` taken): its snapshot buffers are freed here, and
    /// this turn's own prime installs a fresh one at the new prompt end. Returns the position
    /// rewound to, or `None` when the session holds no checkpoint (caller: full re-prime).
    pub fn spec_rewind_to_checkpoint(
        &self,
        e: &Engine,
        sess: &mut SpecSession,
    ) -> Result<Option<usize>, Box<dyn std::error::Error>> {
        if sess.turn_ckpt.as_ref().is_some_and(|ckpt| {
            !sess.cache.can_rollback(&ckpt.snap, 0) || !sess.scratch.can_rewind_to(ckpt.pos)
        }) {
            return Err(
                "SWA ring rewind checkpoint has been lapped; full re-prime required".into(),
            );
        }
        let Some(ckpt) = sess.turn_ckpt.take() else {
            return Ok(None);
        };
        assert!(
            ckpt.pos <= sess.committed.len(),
            "checkpoint past committed ({} > {})",
            ckpt.pos,
            sess.committed.len()
        );
        // Restore through each layer's owning engine. A single primary-engine rollback is not
        // sufficient when the serving cache is stage-owned under cross-device PP.
        crate::pp::restore_cache_checkpoint(e, self, None, &mut sess.cache, &ckpt.snap)?;
        debug_assert_eq!(
            sess.cache.pos, ckpt.pos,
            "rollback landed off the checkpoint"
        );
        sess.scratch.set_len(e, ckpt.pos)?;
        sess.committed.truncate(ckpt.pos);
        sess.last_h = Some(ckpt.last_h);
        sess.next_pred = None;
        sess.pending_tok = None;
        Ok(Some(ckpt.pos))
    }

    /// Grow a parked speculative session to `target_cap` and rewind it to its retained turn
    /// checkpoint without re-priming the checkpoint prefix.
    ///
    /// The trunk cache is restored exactly like a plain grown cache: append-only full-attention
    /// KV rows come from the parked cache, while recurrent state comes from the checkpoint's
    /// owned snapshot. The MTP scratch is also context-linear and its rows below the checkpoint
    /// remain authoritative, so they are copied into a fresh larger scratch before its length is
    /// truncated. Pointer-baking draft graphs are dropped and recaptured on the next burst.
    ///
    /// All fallible work completes before `sess` is mutated. A failed allocation or copy leaves
    /// the parked session intact, allowing the caller one reclaim-and-retry attempt.
    pub fn spec_grow_and_rewind_to_checkpoint(
        &self,
        e: &Engine,
        sess: &mut SpecSession,
        target_cap: usize,
    ) -> Result<Option<usize>, Box<dyn std::error::Error>> {
        if target_cap <= sess.cache.max_ctx {
            return self.spec_rewind_to_checkpoint(e, sess);
        }
        let Some(ckpt) = sess.turn_ckpt.as_ref() else {
            return Ok(None);
        };
        if ckpt.pos == 0 || ckpt.pos > sess.committed.len() {
            return Err(format!(
                "checkpoint pos {} outside committed length {}",
                ckpt.pos,
                sess.committed.len(),
            )
            .into());
        }
        if ckpt.pos > target_cap {
            return Err(format!(
                "checkpoint pos {} exceeds grown capacity {target_cap}",
                ckpt.pos,
            )
            .into());
        }

        let mut grown_cache = crate::pp::new_cache_planned(e, &self.cfg, &self.plan, target_cap)?;
        let mut grown_scratch = self.new_mtp_scratch(e, target_cap)?;
        crate::pp::restore_cache_checkpoint(
            e,
            self,
            Some(&sess.cache),
            &mut grown_cache,
            &ckpt.snap,
        )?;

        if sess.scratch.plane_count() != grown_scratch.plane_count() {
            return Err("checkpoint draft plane count mismatch".into());
        }
        for index in 0..sess.scratch.plane_count() {
            let (src, _) = sess.scratch.plane(index);
            let (dst, _) = grown_scratch.plane_mut(index);
            if ckpt.pos > src.len
                || src.kv_dim_k != dst.kv_dim_k
                || src.kv_dim_v != dst.kv_dim_v
                || src.k_tok_bytes != dst.k_tok_bytes
                || src.v_tok_bytes != dst.v_tok_bytes
            {
                return Err(format!(
                    "checkpoint draft plane {index} layout mismatch (pos {}, source len {})",
                    ckpt.pos, src.len,
                )
                .into());
            }
            match (&src.ring, dst.ring.as_ref()) {
                (Some(sring), Some(_)) => {
                    // Ring-backed draft plane (step35): `ckpt.pos` is absolute and exceeds the
                    // physical rows once lapped — same class as the trunk-KV restore panic
                    // (2026-08-29 warm-turn-at-40k). Copy the aligned live window, rebase.
                    let (new_base, phys) = sring.restore_plan(ckpt.pos).map_err(|err| {
                        format!("checkpoint draft plane {index} SWA restore refused: {err}")
                    })?;
                    let rows = phys.len();
                    let kb = rows * src.k_tok_bytes;
                    let vb = rows * src.v_tok_bytes;
                    if kb > 0 {
                        e.copy_u8_range_into(
                            &mut dst.k,
                            0,
                            &src.k,
                            phys.start * src.k_tok_bytes,
                            kb,
                        )?;
                    }
                    if vb > 0 {
                        e.copy_u8_range_into(
                            &mut dst.v,
                            0,
                            &src.v,
                            phys.start * src.v_tok_bytes,
                            vb,
                        )?;
                    }
                    dst.ring
                        .as_mut()
                        .expect("ring presence checked above")
                        .apply_rebase(new_base);
                    if let Some(base_d) = dst.base_d.as_mut() {
                        e.set_i32_one(base_d, new_base as i32)?;
                    }
                }
                (None, None) => {
                    let kb = ckpt.pos * src.k_tok_bytes;
                    let vb = ckpt.pos * src.v_tok_bytes;
                    if kb > 0 {
                        e.copy_u8_into(&mut dst.k, 0, &src.k, kb)?;
                    }
                    if vb > 0 {
                        e.copy_u8_into(&mut dst.v, 0, &src.v, vb)?;
                    }
                }
                _ => {
                    return Err(format!("checkpoint draft plane {index} ring/flat mismatch").into());
                }
            }
        }
        grown_scratch.set_len(e, ckpt.pos)?;
        // The old scratch is dropped immediately after publication below. Bound its D2D reads
        // first; growth happens once per rewritten turn, outside the decode hot loop.
        e.stream().synchronize()?;

        let ckpt = sess
            .turn_ckpt
            .take()
            .expect("checkpoint remained present through transactional grow");
        let pos = ckpt.pos;
        sess.cache = grown_cache;
        sess.scratch = grown_scratch;
        sess.committed.truncate(pos);
        sess.last_h = Some(ckpt.last_h);
        sess.next_pred = None;
        sess.pending_tok = None;
        sess.draft_ctx = None;
        debug_assert_eq!(sess.cache.pos, pos, "grown rewind landed off checkpoint");
        debug_assert!(
            (0..sess.scratch.plane_count()).all(|index| sess.scratch.plane(index).0.len == pos),
            "grown draft rewind landed off checkpoint"
        );
        Ok(Some(pos))
    }

    /// Commit a carried pending bonus (see SpecSession::pending_tok): one T=1 trunk pass
    /// (its logits' argmax becomes next_pred) + the draft-KV fill at the carried anchor —
    /// byte-identical to the pre-carry session tail. Required before a non-empty-suffix
    /// prime, a sampled turn, or parking a session for pool reuse. No-op without a pending.
    /// `sampling` is the sampler of the request that will CONSUME the resulting `next_pred`
    /// (lane/sampled-spec-quality): this is a boundary site like any other, so a sampled
    /// consumer must get a DRAWN token, not an argmax. Pass `None` from the park/demote
    /// callers — a pending only ever exists on the GREEDY tail, and the consumer of a
    /// park-time flush is a future request whose sampler is not knowable here (residual
    /// named at the pool-resume probe in worker.rs and in SAMPLED-QUALITY.md).
    pub fn spec_flush_pending(
        &self,
        e: &Engine,
        sess: &mut SpecSession,
        sampling: Option<SpecSampling>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        sess.cache.ensure_usable("spec_flush_pending")?;
        let Some(b) = sess.pending_tok.take() else {
            return Ok(());
        };
        if self.mtp.is_none() {
            return Err("pending carry requires an MTP head".into());
        }
        let n_embd = self.cfg.n_embd as usize;
        let (embd_qt, embd_rb) = self.embd.qt_and_row_bytes(n_embd);
        let embd_gpu = if spec_host_embd() {
            None
        } else {
            Some(
                self.embd_gpu
                    .get_or_init(|| e.upload_u8(&self.embd.raw).expect("embed table upload")),
            )
        };
        let embd_dev = embd_gpu.map(|g| (g, embd_qt, embd_rb));
        let pos_b = sess.cache.pos;
        sess.scratch.set_len(e, pos_b)?;
        let (lg_b, hb) = self.spec_target_step_h(e, b, &mut sess.cache)?;
        sess.next_pred = Some(match sampling {
            Some(sp) if sp.temp > 0.0 && spec_sampled_boundary_on() => {
                // window includes `b` itself: it is committed by this pass, and the pre-lane
                // code never counted a boundary token in the penalty history at all.
                let hist = pen_window_seed(&sess.committed, &[b], sp.penalty_last_n);
                sample_boundary_token(e, &lg_b, &sp, &hist, &mut sess.sctr, "flush-pending")?
            }
            _ => argmax(&lg_b) as u32,
        });
        let anchor = sess
            .last_h
            .as_ref()
            .expect("pending carry requires last_h (the predecessor-row anchor)");
        self.mtp_kv_fill_all(e, &[b], anchor, pos_b, &mut sess.scratch, embd_dev)?;
        sess.last_h = Some(hb);
        sess.committed.push(b);
        Ok(())
    }

    /// Solo target feed used only at speculative round boundaries. Step35 serving made its
    /// staged batched B=1 graph authoritative, so a speculative session must enter and leave
    /// rounds through that same graph. Other model families keep their eager T=1 contract.
    fn spec_target_step_h(
        &self,
        e: &Engine,
        token: u32,
        cache: &mut Cache,
    ) -> Result<(Vec<f32>, CudaSlice<f32>), Box<dyn std::error::Error>> {
        cache.ensure_usable("spec_target_step_h")?;
        if !self.sliding_gated_moe_batch_program() && !self.batched_serving_numeric_class() {
            return self.decode_step_h(e, token, cache);
        }
        let pos0 = cache.pos;
        let (logits, hidden) = self.decode_step_t_core(e, &[token], pos0, cache, None, None)?;
        Ok((e.dtoh(&logits)?, hidden))
    }

    /// The archs whose LIVE B=1 serving runs the generic BATCHED numeric class (decode_step_batch
    /// walk + batched head), so their spec verify must run the SAME class. MoE learned this
    /// 2026-08-14 AM (4b777ccc5); the dense hybrid reproduced the identical near-tie flip class
    /// the same day on Qwen3.8-27B — eager-class verify logits drift from batched-class serving
    /// logits ("1 ULP at layer 2 → 2.3e-1 logit maxdiff at the head"), and the GDN recurrence
    /// carries the drift until a near-tie flips deep in generation. One predicate so the five
    /// dispatch sites cannot drift apart again.
    /// Draft-graph head admissibility (lane/draftcost-moe, 2026-08-20): the capture body
    /// (`mtp_head_forward_cap`) supports Dense heads and SOFTMAX device-routed resident-MoE
    /// heads. Residency alone is insufficient: Hy3/M3/Step sigmoid routing returns selected
    /// experts through a host synchronization, which is capture-illegal. Those heads use the
    /// exact eager draft chain until a device-only sigmoid expert program lands. Trunk FFN class
    /// is irrelevant — the graph body is the HEAD forward only. One predicate for all three
    /// eligibility sites so they cannot drift (the serving numeric-class lesson).
    fn mtp_graph_capturable(&self) -> bool {
        let sigmoid_router = self.cfg.sigmoid_router().is_some();
        for head in self.mtp.iter().chain(self.mtp_extra.iter()) {
            let reason = match &head.ffn {
                crate::hybrid::Ffn::Dense { .. } => None,
                crate::hybrid::Ffn::Moe(mo) if mo.dev_exps.is_none() => {
                    Some("non-resident MoE MTP head")
                }
                crate::hybrid::Ffn::Moe(_) if sigmoid_router => {
                    Some("sigmoid-router MoE MTP head requires host-visible routing")
                }
                crate::hybrid::Ffn::Moe(_) => None,
            };
            if let Some(reason) = reason {
                static NOTICE: std::sync::Once = std::sync::Once::new();
                NOTICE.call_once(|| {
                    eprintln!(
                        "[spec] draft graph unavailable: {reason}; eager draft chain engaged"
                    );
                });
                return false;
            }
        }
        self.mtp.is_some()
    }

    fn batched_serving_numeric_class(&self) -> bool {
        self.plan
            .trunk_operations()
            .contains(&memra_gguf::model_plan::OperationKind::GatedDeltaNet)
    }

    /// The family the MTP verify-graph default was measured on: GatedDeltaNet state layers
    /// (a `recur` mixer) together with a routed-MoE FFN — Ornith-1.5-35B-A3B and its kin. The
    /// server-side twin of this test is `model_forces_spec_replay` (GatedDeltaNet + MoeMlp);
    /// keeping the engine's own version structural rather than name-based means a new
    /// checkpoint of the same shape inherits the default, and a different shape does not.
    /// pub(crate) since lane/graph-launch-guard-sweep-20260831: `dspark_vg_admission_debt`
    /// consults it so the MTP-route pool stops escaping the admission charge.
    pub(crate) fn vgraph_family_default(&self) -> bool {
        let has_linear = self
            .layers
            .iter()
            .any(|l| matches!(l.mixer, Mixer::Linear(_)));
        let has_moe = self
            .layers
            .iter()
            .any(|l| matches!(l.ffn, crate::hybrid::Ffn::Moe(_)));
        has_linear && has_moe
    }

    fn sliding_gated_moe_batch_program(&self) -> bool {
        self.uses_sliding_gated_moe_program()
    }

    fn gemma_batch_program(&self) -> bool {
        self.uses_gemma_program()
    }

    /// Reduced-matrix admission for increment 1. This deliberately does not change the PP-2
    /// serving policy: the worker calls it only after `MEMRA_SPEC_PIPE=1` and an explicit spec
    /// session already exist.
    pub fn spec_pipe_available(&self, e: &Engine) -> bool {
        if std::env::var("MEMRA_SPEC_PIPE").as_deref() != Ok("1")
            || !spec_devacc()
            || spec_replay_env_enabled()
            || spec_stream()
            || std::env::var("MEMRA_SPEC_ADAPT").as_deref() == Ok("1")
            || std::env::var("MEMRA_SPEC_PMIN0").as_deref() == Ok("1")
            || std::env::var("MEMRA_SPEC_PP_ANATOMY").as_deref() == Ok("1")
            || std::env::var("MEMRA_SPEC_PMIN")
                .ok()
                .and_then(|v| v.parse::<f32>().ok())
                .unwrap_or(0.0)
                > 0.0
            || self.is_gemma4_e4b()
            || self.gemma_batch_program()
            || self.mtp.is_none()
            || !self.mtp_extra.is_empty()
            // Both paired lanes would otherwise hold the model-global verify-graph mutex across
            // setup and wait for each other. Independent graph pools are future work; the pair
            // requires the explicit eager-verify arm today.
            || crate::spec::spec_verify_graph_env()
                .unwrap_or_else(|| self.vgraph_family_default())
        {
            return false;
        }
        let Some(cuts) = crate::pp::pp_cuts(self.layers.len()) else {
            return false;
        };
        if cuts.len() != 3 || crate::pp::pp2_streams_off() || !crate::pp::spec_pp_on() {
            return false;
        }
        crate::pp::PpNRt::get(e)
            .map(|rt| rt.n_stages() == 2 && rt.cross_device())
            .unwrap_or(false)
    }

    /// Two warm greedy continuation bursts over one PP-2 interval coordinator. The two existing
    /// `generate_spec_inner2` call stacks own all per-session round locals; only phase issue order
    /// changes. No callback is accepted in increment 1 — the worker publishes each completed burst.
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::type_complexity)] // allow: one-shot composite type; naming it would hide the shape that matters at the call site
    pub fn generate_spec_session_pair(
        &self,
        e: &Engine,
        sess_a: &mut SpecSession,
        max_new_a: usize,
        k_a: usize,
        sess_b: &mut SpecSession,
        max_new_b: usize,
        k_b: usize,
    ) -> Result<((Vec<u32>, usize, usize), (Vec<u32>, usize, usize)), Box<dyn std::error::Error>>
    {
        self.refuse_hyper("generate_spec_session_pair")?;
        if !self.spec_pipe_available(e) {
            return Err("two-session speculative pipeline is outside its reduced matrix".into());
        }
        let rt = crate::pp::PpNRt::get(e)?;
        let pp_walk = rt.acquire_walk("generate_spec_session_pair")?;
        let pp_permit = rt.walk_permit(&pp_walk, "generate_spec_session_pair")?;
        if max_new_a == 0 || max_new_b == 0 || k_a == 0 || k_b == 0 {
            return Err(
                "two-session speculative pipeline requires non-empty positive-K bursts".into(),
            );
        }
        for sess in [&*sess_a, &*sess_b] {
            if sess.committed.is_empty()
                || sess.last_h.is_none()
                || (sess.next_pred.is_none() && sess.pending_tok.is_none())
            {
                return Err("two-session speculative pipeline requires warm continuations".into());
            }
        }

        let graph_ok = std::env::var("MEMRA_SPEC_NOGRAPH").is_err()
            && !spec_host_embd()
            && self.mtp_graph_capturable()
            && self.mtp_extra.is_empty()
            && !crate::model::full_prec_enabled();
        let graph_a = graph_ok && k_a + 2 < 96;
        let graph_b = graph_ok && k_b + 2 < 96;
        let was_tracking = e.ctx().is_event_tracking();
        if (graph_a || graph_b) && was_tracking {
            unsafe {
                e.ctx().disable_event_tracking();
            }
        }

        static LOGGED: std::sync::Once = std::sync::Once::new();
        LOGGED.call_once(|| {
            eprintln!("[spec-pipe] two-session PP-2 continuation pipeline engaged");
        });
        let sync = std::sync::Arc::new(SpecPipeSync::new());
        let lane_a = SpecPipeLane {
            sync: sync.clone(),
            lane: 0,
            rt,
            walk_permit: pp_permit.clone(),
        };
        let lane_b = SpecPipeLane {
            sync,
            lane: 1,
            rt,
            walk_permit: pp_permit,
        };
        let mut sess_b_ptr = SpecPipeSessionPtr(sess_b as *mut SpecSession);
        let (result_a, result_b) = std::thread::scope(|scope| {
            let b = scope.spawn(move || {
                let mut finish = SpecPipeFinish::new(&lane_b);
                let sess_b = unsafe { sess_b_ptr.get_mut() };
                let result = (|| -> Result<_, String> {
                    e.ctx().bind_to_thread().map_err(|err| err.to_string())?;
                    self.generate_spec_inner2(
                        e,
                        &[],
                        max_new_b,
                        k_b,
                        graph_b,
                        Some(sess_b),
                        None,
                        None,
                        None,
                        None,
                        Some(&lane_b),
                    )
                    .map_err(|err| err.to_string())
                })();
                finish.close(result.is_err());
                result
            });
            let mut finish = SpecPipeFinish::new(&lane_a);
            let result_a = self.generate_spec_inner2(
                e,
                &[],
                max_new_a,
                k_a,
                graph_a,
                Some(sess_a),
                None,
                None,
                None,
                None,
                Some(&lane_a),
            );
            finish.close(result_a.is_err());
            let result_b = b
                .join()
                .map_err(|_| "paired speculative session B panicked".to_string())
                .and_then(|r| r);
            (result_a, result_b)
        });

        if (graph_a || graph_b) && was_tracking {
            unsafe {
                e.ctx().enable_event_tracking();
            }
        }
        let result_a = result_a?;
        let result_b = result_b.map_err(|err| -> Box<dyn std::error::Error> { err.into() })?;
        Ok((result_a, result_b))
    }

    /// One spec-decode turn on a live session. `suffix` = the NEW tokens only (turn N+1's user
    /// message rendered through the chat template continuation). Returns (new tokens emitted,
    /// drafted, accepted); session.committed grows by suffix + emitted.
    pub fn generate_spec_session(
        &self,
        e: &Engine,
        sess: &mut SpecSession,
        suffix: &[u32],
        max_new: usize,
        k: usize,
    ) -> Result<(Vec<u32>, usize, usize), Box<dyn std::error::Error>> {
        self.generate_spec_session_sampled(e, sess, suffix, max_new, k, None, None)
    }

    /// Serve-path sampled spec: routes the burst through the rejection-sampling verify with
    /// per-SESSION Philox continuity (sess.sctr/uctr). None = env-driven (CLI) or greedy.
    /// Filters (top-k/p/min-p) apply SYMMETRICALLY to draft q and verify p — distribution-exact
    /// for the filtered target (feat/filtered-spec).
    ///
    /// `on_commit` (sse-cadence, 2026-08-05): called with each newly-emitted slice of the
    /// output — once right after the prime's first token, then once per round commit — so a
    /// streaming caller can flush text at round cadence instead of once per burst. The slices
    /// are disjoint, in order, and concatenate to exactly the returned token vec. Emission-
    /// timing only: token bytes, session state, and exactness are untouched.
    ///
    /// The returned bool is a CONTINUE-VERDICT (admission yield, 2026-08-06): `false` ends
    /// the burst at the current round boundary, exactly as if `max_new` had been reached —
    /// the caller's scheduler regains control without waiting the burst out. Burst size is
    /// content-neutral (spec-levers battery), so an early exit moves WHEN the burst returns,
    /// never what tokens say. The slice may be EMPTY (a poll-only boundary — round-stream
    /// drains and the defensive tail flush can land with nothing new committed).
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::type_complexity)] // allow: one-shot composite type; naming it would hide the shape that matters at the call site
    pub fn generate_spec_session_sampled(
        &self,
        e: &Engine,
        sess: &mut SpecSession,
        suffix: &[u32],
        max_new: usize,
        k: usize,
        sampling: Option<SpecSampling>,
        on_commit: Option<&mut dyn FnMut(&[u32]) -> bool>,
    ) -> Result<(Vec<u32>, usize, usize), Box<dyn std::error::Error>> {
        self.generate_spec_session_sampled_prime_split(
            e, sess, suffix, max_new, k, sampling, None, on_commit,
        )
    }

    /// Serve-only cold-prime segmentation twin. `prime_split` is the same stable boundary the
    /// plain worker would honor before entering its sub-floor tokenwise tail; warm continuations
    /// pass `None` and stay on the existing zero-prime path.
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::type_complexity)] // allow: one-shot composite type; naming it would hide the shape that matters at the call site
    pub fn generate_spec_session_sampled_prime_split(
        &self,
        e: &Engine,
        sess: &mut SpecSession,
        suffix: &[u32],
        max_new: usize,
        k: usize,
        sampling: Option<SpecSampling>,
        prime_split: Option<usize>,
        on_commit: Option<&mut dyn FnMut(&[u32]) -> bool>,
    ) -> Result<(Vec<u32>, usize, usize), Box<dyn std::error::Error>> {
        self.generate_spec_session_constrained_prime_split(
            e,
            sess,
            suffix,
            max_new,
            k,
            sampling,
            None,
            prime_split,
            on_commit,
        )
    }

    /// `generate_spec_session_sampled` + GRAMMAR (constrained decoding, 2026-08-03): the
    /// hook truncates acceptance at the first grammar-illegal token AFTER the exactness
    /// verify (grammar is an extra rejection rule, ordering like the batched-verify twins)
    /// and replaces an illegal bonus with the MASKED argmax of the target's own verify
    /// column — token-identical to constrained plain greedy decode. GREEDY only (the
    /// worker routes sampled constrained to plain decode). Acceptance under tight grammars
    /// may drop (drafter is unconstrained); that is measured, not hidden.
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::type_complexity)] // allow: one-shot composite type; naming it would hide the shape that matters at the call site
    pub fn generate_spec_session_constrained(
        &self,
        e: &Engine,
        sess: &mut SpecSession,
        suffix: &[u32],
        max_new: usize,
        k: usize,
        sampling: Option<SpecSampling>,
        constraint: Option<&mut dyn SpecConstraint>,
        on_commit: Option<&mut dyn FnMut(&[u32]) -> bool>,
    ) -> Result<(Vec<u32>, usize, usize), Box<dyn std::error::Error>> {
        self.generate_spec_session_constrained_prime_split(
            e, sess, suffix, max_new, k, sampling, constraint, None, on_commit,
        )
    }

    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::type_complexity)] // allow: one-shot composite type; naming it would hide the shape that matters at the call site
    pub fn generate_spec_session_constrained_prime_split(
        &self,
        e: &Engine,
        sess: &mut SpecSession,
        suffix: &[u32],
        max_new: usize,
        k: usize,
        sampling: Option<SpecSampling>,
        constraint: Option<&mut dyn SpecConstraint>,
        prime_split: Option<usize>,
        on_commit: Option<&mut dyn FnMut(&[u32]) -> bool>,
    ) -> Result<(Vec<u32>, usize, usize), Box<dyn std::error::Error>> {
        if constraint.is_some() && sampling.is_some_and(|s| s.temp > 0.0) {
            return Err(
                "constrained spec decode is greedy-only (worker routes sampled \
                        constrained to plain decode)"
                    .into(),
            );
        }
        // PENDING-CARRY entry flush: a carried bonus precedes any new suffix in the sequence,
        // so it must commit BEFORE the suffix primes; the sampled path doesn't carry (its
        // round-0 accept needs the commit pass's logits). Empty-suffix greedy bursts — the
        // serve continuation case — consume the carry in-loop with zero solo passes.
        if sess.pending_tok.is_some()
            && (!suffix.is_empty() || sampling.is_some_and(|s| s.temp > 0.0))
        {
            self.spec_flush_pending(e, sess, sampling)?;
        }

        // FULL_PREC forces the EAGER draft: the graph capture would enclose cuBLASLt f32 GEMV
        // (the FloatBf16 else-branches) and a bf16_to_f32 dequant alloc — neither is stream-capture
        // safe. Eager rides matmul/matmul_decode_exact, which dequant FloatBf16 on use. (§item 2.)
        // Multi-head MTP (mtp_extra non-empty) no longer disqualifies: the chain captures
        // per-head graphs (lane/step37-draft-graph-serving-20260830, MEMRA_MTP_CHAIN_GRAPH).
        let graph_draft = std::env::var("MEMRA_SPEC_NOGRAPH").is_err()
            && !spec_host_embd()
            && self.mtp_graph_capturable()
            && k + 2 < 96
            && !crate::model::full_prec_enabled();
        let was_tracking = e.ctx().is_event_tracking();
        if graph_draft && was_tracking {
            unsafe {
                e.ctx().disable_event_tracking();
            }
        }
        let r = self.generate_spec_inner2(
            e,
            suffix,
            max_new,
            k,
            graph_draft,
            Some(sess),
            sampling,
            constraint,
            on_commit,
            prime_split,
            None,
        );
        if graph_draft && was_tracking {
            unsafe {
                e.ctx().enable_event_tracking();
            }
        }
        let (out, d, a) = r?;
        Ok((out, d, a))
    }

    pub fn generate_spec(
        &self,
        e: &Engine,
        prompt: &[u32],
        max_new: usize,
        k: usize,
    ) -> Result<(Vec<u32>, usize, usize), Box<dyn std::error::Error>> {
        // glm5 T-parallel verify door (lane/glm5-tparallel-verify): an hc trunk with a
        // loaded DRAFT SOURCE — the embedded MTP head OR the DFlash2 drafter
        // (lane/glm5-dflash-draft-src) — routes to the glm5 draft->verify->rollback loop —
        // MEMRA_GLM5_SPEC=1 only (default OFF; flag row in FLAGS.md). Unset/0 falls
        // through to the standing named refusal below, byte-identical to the pre-lane
        // binary. Same fail-closed manifest stance as the generic path: an unqualified
        // MtpSpec rewrite refuses before any drafting.
        if self.hyper.is_some()
            && crate::glm_spec::glm5_spec_on()
            && (self.mtp.is_some() || self.glm5_dflash.is_some())
        {
            if !self.rewrite_allowed(memra_gguf::execution_manifest::RewriteSurface::MtpSpec) {
                return Err("speculative rewrite is not qualified for this ModelPlan".into());
            }
            return self.generate_spec_glm5(e, prompt, max_new, k);
        }
        self.refuse_hyper("generate_spec")?;
        if crate::pp::pp_cuts(self.layers.len()).is_some()
            && !self.rewrite_allowed(memra_gguf::execution_manifest::RewriteSurface::Pipeline)
        {
            return Err("pipeline rewrite is not qualified for speculative decode".into());
        }
        if !self.rewrite_allowed(memra_gguf::execution_manifest::RewriteSurface::MtpSpec) {
            return Err("speculative rewrite is not qualified for this ModelPlan".into());
        }
        // FULL_PREC forces eager (see generate_spec_session note): CUDA graph capture cannot
        // enclose cuBLASLt f32 GEMV or the bf16_to_f32 dequant alloc the FloatBf16 path needs.
        // Multi-head MTP no longer disqualifies (chain graphs; see generate_spec_session).
        let graph_draft = std::env::var("MEMRA_SPEC_NOGRAPH").is_err()
            && !spec_host_embd()
            && self.mtp_graph_capturable()
            && k + 2 < 96
            && !crate::model::full_prec_enabled();
        if !graph_draft {
            return self.generate_spec_inner2(
                e, prompt, max_new, k, false, None, None, None, None, None, None,
            );
        }
        let was_tracking = e.ctx().is_event_tracking();
        if was_tracking {
            unsafe {
                e.ctx().disable_event_tracking();
            }
        }
        let r = self.generate_spec_inner2(
            e, prompt, max_new, k, true, None, None, None, None, None, None,
        );
        if was_tracking {
            unsafe {
                e.ctx().enable_event_tracking();
            }
        }
        r
    }

    #[allow(clippy::type_complexity)] // allow: one-shot composite type; naming it would hide the shape that matters at the call site
    #[allow(clippy::too_many_arguments)] // allow: the parameter list mirrors the kernel/FFI/call contract; bundling into a struct is a refactor, not a lint fix
    fn generate_spec_inner2(
        &self,
        e: &Engine,
        prompt: &[u32],
        max_new: usize,
        k: usize,
        graph_draft: bool,
        mut sess: Option<&mut SpecSession>,
        sampling: Option<SpecSampling>,
        mut constraint: Option<&mut dyn SpecConstraint>,
        mut on_commit: Option<&mut dyn FnMut(&[u32]) -> bool>,
        prime_split: Option<usize>,
        pipe: Option<&SpecPipeLane>,
    ) -> Result<(Vec<u32>, usize, usize), Box<dyn std::error::Error>> {
        assert!(k >= 1, "k must be >= 1");
        let pipe_setup_walk = match pipe {
            Some(p) => Some(p.setup_begin()?),
            None => None,
        };
        // sse-cadence flush cursor: everything in out[..flushed] has been handed to on_commit.
        let mut flushed = 0usize;
        // admission yield (2026-08-06): on_commit's continue-verdict; false = end the burst
        // at the next round boundary (same exit as max_new reached — the session tail runs).
        // Initialized by the unconditional post-prime flush below.
        let mut keep_going;
        let mtp = self
            .mtp
            .as_ref()
            .expect("generate_spec requires an MTP head (nextn_predict_layers>0)");
        let n_vocab = self.output.out_features();
        // FR-Spec: the draft head may be TRIMMED (fewer rows than n_vocab); the draft argmax runs
        // over the draft vocab and the winning index maps through d2t to a TARGET token id.
        // Everything downstream (verify/accept/commit) sees target ids only — exactness unchanged.
        let d_vocab = mtp
            .shared_head_head
            .as_ref()
            .unwrap_or(&self.output)
            .out_features();
        if !self.mtp_extra.is_empty() {
            if self.plan.draft_source != memra_gguf::model_plan::DraftSourcePlan::Embedded
                || self.plan.mtp_blocks.len() != self.mtp_head_count()
            {
                return Err(
                    "multi-head MTP requires one embedded canonical block per loaded head".into(),
                );
            }
            // TRIMMED chains (2026-08-27): every head must carry the SAME d2t — the ranking is
            // token-frequency and head-independent, and every downstream remap (per-step argmax,
            // stream pack, sampled d2t_dev) reads head 0's map, so equality is what makes that
            // single map correct for the whole chain. Mixed trimmed/untrimmed is refused.
            for (offset, head) in self.mtp_extra.iter().enumerate() {
                if head.d2t != mtp.d2t
                    || head
                        .shared_head_head
                        .as_ref()
                        .unwrap_or(&self.output)
                        .out_features()
                        != d_vocab
                {
                    return Err(format!(
                        "embedded MTP head {} has incompatible draft vocabulary",
                        offset + 1
                    )
                    .into());
                }
            }
            eprintln!(
                "[mtp-chain] heads={} policy=step-modulo prefix-replay kv=per-head",
                self.mtp_head_count()
            );
        }
        let n_embd = self.cfg.n_embd as usize;
        // SESSION MODE: reuse the live cache/scratch, prime only the suffix. `base` = tokens
        // already committed (their state is in the caches); 0 = fresh single-shot call.
        let session_mode = sess.is_some();
        let max_ctx = match sess.as_ref() {
            Some(s) => s.cache.max_ctx,
            None => prompt.len() + max_new + k + 8,
        };
        let mut own_cache;
        let mut own_scratch;
        // PREFIX-CACHE capture request threaded out of the session (lane/spec-prefix-cache):
        // (requested split, destination list). Single-shot per burst; fresh calls have none.
        let mut sess_capture: Option<(Option<usize>, &mut Vec<SpecBoundaryCapture>)> = None;
        // STABLE-BOUNDARY turn-checkpoint request (lane/frspec-multiturn-cache): ABSOLUTE
        // committed-length position; consumed one-shot like `capture_at`. None = legacy
        // prompt-end capture below.
        let mut ckpt_req: Option<usize> = None;
        // FAIL-SAFE bit threaded out of the session (see `SpecSession::capture_disabled`).
        let mut sess_capture_disabled = false;
        let (
            cache,
            scratch,
            mut sess_tail,
            mut sess_draft_slot,
            mut sess_pending_slot,
            sess_ckpt_slot,
            sess_telem,
        ): (
            &mut Cache,
            &mut MtpScratch,
            Option<(
                &mut Vec<u32>,
                &mut Option<CudaSlice<f32>>,
                &mut Option<u32>,
                &mut u32,
                &mut u32,
            )>,
            Option<&mut Option<DraftGraphCtx>>,
            Option<&mut Option<u32>>,
            Option<&mut Option<SpecCheckpoint>>,
            Option<&SpecTelemetryCounters>,
        ) = match sess.take() {
            Some(sr) => {
                let SpecSession {
                    cache,
                    scratch,
                    committed,
                    last_h,
                    next_pred,
                    sctr: s_sctr,
                    uctr: s_uctr,
                    draft_ctx,
                    pending_tok,
                    turn_ckpt,
                    telem,
                    capture_at,
                    boundary_captures,
                    ckpt_at,
                    capture_disabled,
                } = sr;
                sess_capture_disabled = *capture_disabled;
                sess_capture = Some((capture_at.take(), boundary_captures));
                ckpt_req = ckpt_at.take();
                (
                    cache,
                    scratch,
                    Some((committed, last_h, next_pred, s_sctr, s_uctr)),
                    Some(draft_ctx),
                    Some(pending_tok),
                    Some(turn_ckpt),
                    Some(telem),
                )
            }
            None => {
                // STAGE-OWNED KV (lane/pp2-spec 2026-08-06) — see `new_session`. Door shut =
                // `Cache::new` verbatim.
                own_cache = crate::pp::new_cache_planned(e, &self.cfg, &self.plan, max_ctx)?;
                // Persistent scratch = max_ctx rows (~2KB/token quantized).
                own_scratch = self.new_mtp_scratch(e, max_ctx)?;
                (
                    &mut own_cache,
                    &mut own_scratch,
                    None,
                    None,
                    None,
                    None,
                    None,
                )
            }
        };
        cache.ensure_usable("generate_spec")?;
        if scratch.plane_count() != self.mtp_head_count() {
            return Err(format!(
                "MTP scratch/head count mismatch ({}/{})",
                scratch.plane_count(),
                self.mtp_head_count()
            )
            .into());
        }
        let base = cache.pos;
        // PENDING-CARRY consume (2026-08-01): a carried bonus reaches here only on the
        // empty-suffix GREEDY continuation path (generate_spec_session_sampled flushed every
        // other case). It enters the round loop as round-0's pending — verify col 0 — exactly
        // like a mid-burst full-accept boundary: no init feed, no tail commit pass.
        let carried_pending: Option<u32> = sess_pending_slot.as_mut().and_then(|s| s.take());
        // PERSISTENT DRAFT KV (the only mode since 2026-07-08 — the legacy round-local scratch,
        // MEMRA_SPEC_KVLOCAL, measured -35 acceptance pts on the 27B p3 sweep and was removed;
        // acceptance-only — exactness is verify's job either way).
        // HIDDEN-PAIRING CONVENTION (DEFAULT = predecessor-row, 2026-07-04 — the 27B acceptance
        // unlock, +16pts): the MTP head is TRAINED on rows pairing token x_p with the trunk
        // hidden of its PREDECESSOR h_{p-1} (the reference engine's mtp_update shifts the target
        // hiddens right by one; its draft step 0 feeds (id_last, TRUE hidden of the row id_last
        // was sampled from)). memra's historical convention paired SAME-ROW (x_p, h_p) in the fill
        // and seeded chain step 0 through an extra MTP pass on a duplicated token (the
        // pseudo-seed) — measured 27B p2 K=3 acceptance 0.569 vs 0.731, p3 0.445 vs 0.63+, and
        // the chain steps j>=1 were already predecessor-shaped, so ONLY the fill + step-0 seed
        // move. The fill shifts by one and the chain seeds from the predecessor's true hidden
        // DIRECTLY (vh_seed / vx[j-1]) — the pseudo pass disappears (one MTP-block pass saved
        // per round on top of the acceptance win). Draft-quality-only: exactness stays the
        // verify's job either way. (The legacy same-row pairing seam, MEMRA_SPEC_HSAME, and its
        // pseudo-seed passes were removed 2026-07-08 — predecessor pairing won by +16 acc pts;
        // the legacy round-local scratch, MEMRA_SPEC_KVLOCAL, went with it.)
        // REPLAY-FREE PARTIAL ACCEPT (default, 2026-07-03): partial rounds keep the verify's own
        // bit-identical committed-prefix state (KV truncate + recur rebuild from the VerifyCkpt)
        // and leave the bonus PENDING — no duplicate trunk pass (profiled ~0.54 extra full weight
        // reads/round at long ctx). MEMRA_SPEC_REPLAY=1 restores the legacy rollback+replay (A/B
        // + fallback seam).
        // Qwen35-MoE replay pin LIFTED (lane/draftcost-moe, 2026-08-20). The pin's stated
        // bar — the retained verify-state commit proven equivalent to sequential serving —
        // was waiting on this arch running the serving batched verify class, which the
        // t-parallel admission (this lane, increment 1) provided: the VerifyCkpt the
        // replay-free commit consumes is now produced by the SAME serving-class verify that
        // qualified dense qwen35 on 2026-08-15 (where the per-round duplicate replay
        // measured 69 -> 30 tok/s). Qualification receipts (run-spec K=1..8 both arms,
        // 8-prompt replay-vs-replay-free canary, long-prompt cell):
        // research/draftcost-moe-20260820/RECEIPTS.md. MEMRA_SPEC_REPLAY=1 stays the
        // rollback + A/B seam.
        let spec_replay = spec_replay_env_enabled();
        if constraint.is_some() && spec_replay {
            return Err(
                "constrained spec decode does not support MEMRA_SPEC_REPLAY=1 \
                        (legacy replay commits an unmasked bonus)"
                    .into(),
            );
        }
        // TRUE-HIDDEN REFRESH (default in persistent-draft-KV mode): every round overwrites the
        // committed positions' scratch entries from the verify's exact hiddens (mtp_kv_fill batch)
        // instead of keeping chain-approximate entries. MEMRA_SPEC_NOREFRESH=1 = legacy (A/B seam).
        let refresh = std::env::var("MEMRA_SPEC_NOREFRESH").is_err();
        if !refresh && !self.mtp_extra.is_empty() {
            return Err("multi-head MTP requires exact accepted-prefix refresh".into());
        }

        // prime: BATCHED cache prime (prime_cache — the measured #1 e2e gap: tokenwise primed at
        // ~102/38 tok/s vs the engine's ~2000-5900 tok/s batched prefill). prime_cache returns the
        // full pre-output_norm hidden stack [T, n_embd], which IS prompt_h (the persistent-draft-KV
        // mtp_kv_fill input) — no per-token collection needed. Prompts below PRIME_MIN_T, and
        // MEMRA_PRIME_TOKENWISE=1, and frozen Hy3 CPU/GPU expert splits take the tokenwise
        // decode_step_h loop. The latter avoids transient GPU staging of the spilled expert bank.
        // EMPTY-SUFFIX CONTINUATION (serve bursts): a session turn with NO new tokens resumes
        // generation exactly where the last turn stopped — no prime at all. The stashed
        // `next_pred` plays prime_logits' role: it is the token produced from the logits after
        // committed.last() by the same rule this entry applies to a cold prime's last row —
        // an argmax when greedy, a `sample_boundary_token` draw when sampled (the burst tail,
        // or `spec_session_from_restored` for a converted prefix-cache hit, did the drawing
        // where the sampler and the session's Philox counters were live). `last_h` seeds the
        // predecessor pairing below. Fresh calls and non-empty suffixes take the normal path.
        let continuation = prompt.is_empty();
        if continuation {
            assert!(session_mode, "empty prompt requires a session");
            assert!(
                sess_tail
                    .as_ref()
                    .is_some_and(|(c, lh, np, _, _)| !c.is_empty()
                        && lh.is_some()
                        && (np.is_some() || carried_pending.is_some())),
                "empty-suffix continuation needs a primed session (committed + last_h + next_pred|pending)"
            );
        }
        let mut prime_logits;
        let mut prompt_h: Option<CudaSlice<f32>> = None;
        let t_prime = std::time::Instant::now();
        let batched_prime = !continuation
            && prompt.len() >= crate::hybrid_forward::PRIME_MIN_T
            && std::env::var("MEMRA_PRIME_TOKENWISE").is_err()
            && !e.frozen_cpu_experts_prefer_tokenwise_prime();
        let prime_split = prime_split.filter(|&split| split > 0 && split < prompt.len());
        if prime_split.is_some() && continuation {
            return Err("spec prime split requires a non-empty prime".into());
        }
        // STABLE-BOUNDARY TURN CHECKPOINT stop (lane/frspec-multiturn-cache, 2026-08-21):
        // the worker's `ckpt_at` request, ABSOLUTE -> prompt-relative. On WARM bursts
        // (base != 0, an affinity-rewound or pool-resumed session priming its own delta)
        // this is the only stop; on COLD bursts it usually coincides with `prime_split`
        // (both are the plain tier's stable pre-generation boundary). A boundary the prime
        // cannot honor (outside this prime's range) silently drops the capture — the
        // turn_ckpt convention: the next turn re-primes in full, never a wrong resume.
        let ckpt_rel = if continuation {
            None
        } else {
            ckpt_req
                .and_then(|abs| abs.checked_sub(base))
                .filter(|&r| r > 0 && r < prompt.len())
        };
        // Prime stops, ordered: each is a boundary the prime halts at so the in-place GDN
        // conv/ssm state can be snapshotted there (the only moment it exists). One stop =
        // the legacy single-split program, byte-for-byte.
        let mut stops: Vec<usize> = Vec::new();
        for b in [prime_split, ckpt_rel].into_iter().flatten() {
            if !stops.contains(&b) {
                stops.push(b);
            }
        }
        stops.sort_unstable();
        // Captured at the ckpt stop, installed into the session slot post-prime (replacing
        // the legacy prompt-end capture). Some(None) = capture attempted and failed -> the
        // slot is cleared (a stale checkpoint would rewind to the WRONG boundary).
        let mut ckpt_early: Option<Option<SpecCheckpoint>> = None;
        if continuation {
            prime_logits = Vec::new();
        } else if !stops.is_empty() {
            if let Some(&first) = stops.first()
                && prime_split == Some(first)
                && first < crate::hybrid_forward::PRIME_MIN_T
            {
                return Err(format!(
                    "spec prime split {first} is below PRIME_MIN_T {}",
                    crate::hybrid_forward::PRIME_MIN_T,
                )
                .into());
            }
            // Mirror the plain worker's boundary stops exactly. Each segment is a
            // request-level prime (`queued_after` keeps Step35 arm selection independent of
            // the stops — tick-seg law); a segment below PRIME_MIN_T (and the final tail
            // under MEMRA_PRIME_TOKENWISE) takes the same eager tokenwise continuation as
            // prefill_tick. Retain every hidden row so the draft scratch fill remains one
            // coherent prompt.
            let mut h_all = e.uninit(prompt.len() * n_embd)?;
            prime_logits = Vec::new();
            let mut prev = 0usize;
            for seg_end in stops.iter().copied().chain(std::iter::once(prompt.len())) {
                if seg_end <= prev {
                    continue;
                }
                let seg = &prompt[prev..seg_end];
                let is_final = seg_end == prompt.len();
                let batched_seg = seg.len() >= crate::hybrid_forward::PRIME_MIN_T
                    && (!is_final
                        || (std::env::var("MEMRA_PRIME_TOKENWISE").is_err()
                            && !e.frozen_cpu_experts_prefer_tokenwise_prime()));
                if batched_seg {
                    let (l, _, h_seg) =
                        self.prime_cache(e, seg, &mut *cache, prompt.len() - seg_end)?;
                    e.copy_into(&mut h_all, prev * n_embd, &h_seg, seg.len() * n_embd)?;
                    prime_logits = l;
                } else {
                    for (i, &tok) in seg.iter().enumerate() {
                        let (l, h) = self.decode_step_h(e, tok, &mut *cache)?;
                        e.copy_into(&mut h_all, (prev + i) * n_embd, &h, n_embd)?;
                        prime_logits = l;
                    }
                }
                prev = seg_end;
                if is_final {
                    break;
                }
                debug_assert_eq!(cache.pos, base + seg_end, "prime stop landed off boundary");
                // PREFIX-CACHE BOUNDARY CAPTURE (lane/spec-prefix-cache): the GDN conv/ssm
                // states are about to be advanced in place by the next segment, so this is
                // the ONLY moment the boundary's recurrent state exists. Capture iff the
                // worker requested exactly this stop (cold sessions only — `capture_at` is
                // never armed warm). A failed snapshot is silent (turn_ckpt convention) —
                // publication is an optimization, never a correctness dependency.
                if base == 0
                    && let Some((requested, slot)) = sess_capture.as_mut()
                {
                    // Publish at the requested miss-LCP stop (the shared-prefix class)
                    // AND at the stable-boundary stop (the next-turn re-render class,
                    // lane/frspec-multiturn-cache) — the same boundary set the plain
                    // prefill tick learns. Without the second entry, the turn after a
                    // cold re-park could only hit the OLDER lcp entry (the measured
                    // one-turn transient: t3 restored 607 of 24122 while the plain arm
                    // rewound to 15222). Dedupe is the worker sweep's has_key.
                    if (*requested == Some(seg_end) || ckpt_rel == Some(seg_end))
                        && let Ok(snap) = cache.snapshot(e)
                    {
                        slot.push(SpecBoundaryCapture {
                            snap,
                            pos: seg_end,
                            logits: prime_logits.clone(),
                            // rows [0..seg_end) of h_all are primed — the following
                            // segments append, never overwrite.
                            last_h: capture_boundary_hidden(e, &h_all, seg_end, n_embd),
                            latent_tails: Vec::new(),
                        });
                    }
                }
                // SESSION-AFFINITY TURN CHECKPOINT at the STABLE boundary (see `ckpt_at`):
                // same snapshot mechanics, installed post-prime in place of the prompt-end
                // capture the re-render class always diverged below.
                if ckpt_rel == Some(seg_end) {
                    let anchor: Result<CudaSlice<f32>, Box<dyn std::error::Error>> =
                        e.uninit(n_embd).and_then(|mut a| {
                            e.copy_view_into(
                                &mut a,
                                0,
                                &h_all.slice((seg_end - 1) * n_embd..seg_end * n_embd),
                                n_embd,
                            )?;
                            Ok(a)
                        });
                    ckpt_early = Some(match (cache.snapshot(e), anchor) {
                        (Ok(snap), Ok(last_h)) => Some(SpecCheckpoint {
                            snap,
                            pos: base + seg_end,
                            last_h,
                        }),
                        _ => None,
                    });
                }
            }
            if std::env::var("MEMRA_SPEC_STATS").as_deref() == Ok("1") {
                eprintln!(
                    "[spec-prime] stops={stops:?} tail={}",
                    prompt.len() - stops.last().copied().unwrap_or(0)
                );
            }
            prompt_h = Some(h_all);
        } else if batched_prime {
            let (l, _h_seed, hiddens) = self.prime_cache(e, prompt, &mut *cache, 0)?;
            prime_logits = l;
            prompt_h = Some(hiddens);
        } else {
            prime_logits = Vec::new();
            prompt_h = Some(e.uninit(prompt.len() * n_embd)?);
            for (i, &tok) in prompt.iter().enumerate() {
                let (l, h) = self.spec_target_step_h(e, tok, &mut *cache)?;
                if let Some(ph) = prompt_h.as_mut() {
                    e.copy_into(ph, i * n_embd, &h, n_embd)?;
                }
                prime_logits = l;
            }
        }
        e.stream().synchronize()?;
        // PREFIX-CACHE SEED CAPTURE (lane/spec-prefix-cache): boundary == prompt end (the seed
        // case — no shared-prefix split, publish the whole prompt). The prime just finished, so
        // cache.pos == base + prompt.len() and the recurrent state IS the boundary state;
        // prime_logits are the boundary logits. Cold sessions only (base == 0) — same law as
        // prime_split. The mid-prompt capture above already consumed the request if it matched.
        if !continuation
            && base == 0
            && let Some((requested, slot)) = sess_capture.as_mut()
            && *requested == Some(prompt.len())
            && slot.is_empty()
        {
            debug_assert_eq!(cache.pos, prompt.len(), "seed capture off prompt end");
            if let Ok(snap) = cache.snapshot(e) {
                slot.push(SpecBoundaryCapture {
                    snap,
                    pos: prompt.len(),
                    logits: prime_logits.clone(),
                    last_h: prompt_h
                        .as_ref()
                        .map(|ph| capture_boundary_hidden(e, ph, prompt.len(), n_embd))
                        .unwrap_or_default(),
                    latent_tails: Vec::new(),
                });
            }
        }
        // Harness timing contract (see crate::PRIME_NANOS): gen-only throughput without the
        // prime-subtraction hack.
        crate::PRIME_NANOS.store(
            t_prime.elapsed().as_nanos() as u64,
            std::sync::atomic::Ordering::Relaxed,
        );

        let (embd_qt, embd_rb) = self.embd.qt_and_row_bytes(n_embd);
        // Resident table is fastest when it fits. Large spill deployments can preserve that HBM
        // for expert-cache slots and gather only the exact rows needed by MTP/verify from host.
        let host_embd = spec_host_embd();
        let embd_gpu = if host_embd {
            None
        } else {
            Some(
                self.embd_gpu
                    .get_or_init(|| e.upload_u8(&self.embd.raw).expect("embed table upload")),
            )
        };
        let embd_dev = embd_gpu.map(|g| (g, embd_qt, embd_rb));
        if host_embd {
            eprintln!(
                "[spec] host-row embedding: {} bytes kept off HBM",
                self.embd.raw.len()
            );
        }
        let mut out: Vec<u32> = Vec::with_capacity(max_new);
        let mut total_drafted = 0usize;
        let mut total_accepted = 0usize;

        // --- SAMPLER FIRST (lane/sampled-spec-quality, 2026-08-19) ---
        // The sampler config, the session's Philox counters and the penalty window are parsed
        // HERE, above the boundary-token selection, because the boundary token must be drawn
        // from the sampler the request asked for. Pre-lane this block sat ~50 lines BELOW the
        // selection, which is the whole mechanical reason the boundary token was an argmax:
        // the sampler state was not in scope yet. Nothing here depends on the round loop, so
        // moving it up is a pure reordering for greedy (`sampled == false` ⇒ every branch
        // below takes the argmax path it always took).
        // --- SAMPLED SPEC (MEMRA_SPEC_TEMP>0, research/sampled-spec-impl-map.md): rejection-
        // sampling verify (Leviathan/Chen) — accept draft x at u < p(x)/q(x), resample from
        // norm(max(0,p-q)) on reject, bonus sampled from p on full accept. Counter-based Philox
        // everywhere (seed, event) -> reproducible. temp==0/unset = the greedy path, untouched.
        let sp = sampling.unwrap_or_else(|| SpecSampling {
            temp: std::env::var("MEMRA_SPEC_TEMP")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0.0),
            seed: std::env::var("MEMRA_SEED")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(42),
            top_k: std::env::var("MEMRA_TOP_K")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0),
            top_p: std::env::var("MEMRA_TOP_P")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(1.0),
            min_p: std::env::var("MEMRA_MIN_P")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0.0),
            penalty_last_n: std::env::var("MEMRA_PENALTY_LAST_N")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0),
            penalty_repeat: std::env::var("MEMRA_PENALTY_REPEAT")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(1.0),
            penalty_freq: std::env::var("MEMRA_PENALTY_FREQ")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0.0),
            penalty_present: std::env::var("MEMRA_PENALTY_PRESENT")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0.0),
        });
        let (sp_temp, sp_seed) = (sp.temp, sp.seed);
        let sampled = sp_temp > 0.0;
        // Counters resume from the session (burst continuity: randomness must never repeat
        // across generate_spec_session calls); one-shot callers start at (0,0). Read through
        // sess_tail — `sess` was take()n into it above, so sess.as_ref() here is always None.
        let mut sctr: u32 = sess_tail.as_ref().map(|(_, _, _, s, _)| **s).unwrap_or(0);
        let mut uctr: u32 = sess_tail.as_ref().map(|(_, _, _, _, u)| **u).unwrap_or(0);
        // Penalties (v2.1): applied to COPIES of q rows and p columns symmetrically (exactness
        // for the penalized+filtered target). History = generated tokens, host-tracked window.
        let pen_on = sampled
            && sp.penalty_last_n > 0
            && (sp.penalty_repeat != 1.0 || sp.penalty_freq != 0.0 || sp.penalty_present != 0.0);
        // SESSION-SPANNING PENALTY WINDOW (Item 2). Pre-lane this was
        // `prompt.iter().rev().take(64).rev()` — the BURST's suffix slice — so a continuation
        // burst (the majority of a stream's tokens, and ALL of a converted cache hit's) started
        // with an EMPTY penalty history and the client's repetition/frequency/presence penalties
        // silently reset at every burst boundary. The window now spans `committed ++ prompt`,
        // which is what the API contract says and what the plain sampler's own `history` does.
        // Byte-identical to the pre-lane seed for a cold turn-1 burst at the default window.
        let mut pen_hist: Vec<u32> = if pen_on {
            let sess_hist: &[u32] = if spec_pen_session_on() {
                sess_tail
                    .as_ref()
                    .map(|(c, ..)| c.as_slice())
                    .unwrap_or(&[])
            } else {
                &[] // MEMRA_SPEC_PEN_SESSION=0: pre-lane burst-local window
            };
            pen_window_seed(sess_hist, prompt, sp.penalty_last_n)
        } else {
            Vec::new()
        };
        // First generated token = the BOUNDARY token: greedy takes the argmax of the prompt's
        // last logits (== greedy's first token, byte-contract); SAMPLED draws it from the
        // request's own filtered/penalized target through the session's Philox stream
        // (`sample_boundary_token`, lane/sampled-spec-quality Item 1 — pre-lane this was an
        // argmax in both regimes, so ~1 token per burst of a sampled stream was greedy).
        // Emit it, then FEED it to establish the loop invariant below.
        // PENDING-CARRY: the carried bonus was already emitted by the LAST burst — it becomes
        // last_token WITHOUT re-emission, and round 0 consumes it as pending (no init feed).
        // CONSTRAINED entry rules: the first emitted token is the MASKED argmax of the
        // prompt's last logits (plain constrained-greedy identity); a continuation without
        // a carried pending would emit an UNMASKED stashed next_pred — refused loudly (the
        // worker never resumes constrained sessions from the pool, so this cannot fire).
        if let Some(c) = constraint.as_deref_mut() {
            if continuation && carried_pending.is_none() {
                return Err("constrained spec continuation requires a carried pending \
                            (pool resume is unconstrained-only)"
                    .into());
            }
            if !continuation {
                c.mask_logits(&mut prime_logits)
                    .map_err(|e2| format!("constraint: {e2}"))?;
            }
        }
        let mut last_token = if let Some(b) = carried_pending {
            b
        } else if continuation {
            // A continuation's boundary token was DRAWN by the burst that stashed it (the
            // session tail below), or by `spec_session_from_restored` for a converted
            // prefix-cache hit — in both cases from the correct logits row with this same
            // session's Philox stream, which is why it can be consumed here as-is.
            sess_tail.as_ref().unwrap().2.unwrap()
        } else if sampled && constraint.is_none() && spec_sampled_boundary_on() {
            sample_boundary_token(e, &prime_logits, &sp, &pen_hist, &mut sctr, "cold-prime")?
        } else {
            // greedy (byte contract), the rollback door, or constrained (masked-argmax
            // identity — the worker routes sampled+constrained to the plain path, and this
            // function refuses the combination outright above).
            argmax(&prime_logits) as u32
        };
        if pen_on {
            // The boundary token is a GENERATED token: the plain sampler `accept()`s every
            // emitted token into its penalty history, and pre-lane the burst's first token
            // was invisible to penalties forever (never pushed, and never in `committed`
            // until this burst's tail). Covers the carry/continuation seeds too — neither is
            // in `committed` yet.
            pen_hist.push(last_token);
        }
        if carried_pending.is_none() {
            out.push(last_token);
            // grammar advances with every emitted token (carried pendings were consumed
            // by the burst that emitted them).
            if let Some(c) = constraint.as_deref_mut() {
                c.consume(last_token)
                    .map_err(|e2| format!("constraint: {e2}"))?;
            }
        }
        if continuation {
            // draft-KV invariant: entries [0..base) are the session's exact fills; truncate any
            // overhang so the chain's first append lands at slot base (== committed.len()).
            scratch.set_len(e, base)?;
        }
        // sse-cadence: hand the caller every not-yet-flushed token (disjoint in-order slices
        // concatenating to the full `out`). Called after the prime's first token and after each
        // round commit — emission timing only, token bytes untouched. The slice may be EMPTY
        // (poll-only boundary: zero-round folds commit nothing new); returns the caller's
        // continue-verdict (admission yield, 2026-08-06) — false ends the burst at this round.
        #[allow(clippy::type_complexity)] // allow: one-shot composite type; naming it would hide the shape that matters at the call site
        fn flush_commit(
            cb: &mut Option<&mut dyn FnMut(&[u32]) -> bool>,
            out: &[u32],
            flushed: &mut usize,
        ) -> bool {
            if let Some(f) = cb.as_mut() {
                let keep = f(&out[*flushed..]);
                *flushed = out.len();
                keep
            } else {
                true
            }
        }
        keep_going = flush_commit(&mut on_commit, &out, &mut flushed);
        // INVARIANT at loop top: `last_token` is the most-recently-committed/emitted token, its
        // KV+recur state IS in `cache` (cache.pos = position right AFTER last_token), `last_pred`
        // is the greedy ARGMAX of the logits that predict the token FOLLOWING last_token, and
        // `h_seed` = last_token's pre-output_norm hidden. Establish it by feeding last_token once
        // (mirrors plain greedy). DEVICE-ARGMAX lever: the accept walk only ever consumes the
        // argmax of those logits — never the full vector — so a host u32 replaces the Vec<f32>.
        // Trimmed heads: q lives on the trimmed vocab; accept gathers use the TRIMMED index and
        // the residual scatters q into target-id space (q=-inf off-trim — the head cannot propose
        // those, so their residual mass is p(x), correct by construction).
        let d2t_dev: Option<CudaSlice<u32>> = if sampled || crate::spec::spec_stream() {
            match &mtp.d2t {
                Some(map) => Some(e.htod_u32_v(map)?),
                None => None,
            }
        } else {
            None
        };
        let mut q_full_buf: Option<CudaSlice<f32>> = None;
        // host Philox4x32-10 accept-test uniforms: module fn `host_u01` (shared with the
        // dspark sampled-admission walk); byte-identical to the closure it replaces.
        let mut draft_logits: Vec<CudaSlice<f32>> = Vec::new(); // retained head logits (q), per slot
        let mut draft_stats: Vec<(f32, f32, f32)> = Vec::new(); // (row_max, th_e, z_e) per slot
        let mut perturb_buf: Option<CudaSlice<f32>> = None; // gumbel scratch (max(n_vocab,d_vocab))
        let mut sample_tok = e.alloc_u32_zeroed(1)?; // residual/bonus sample out
        let mut col_buf: Option<CudaSlice<f32>> = None; // materialized verify column
        let mut pen_hist_d: Option<CudaSlice<u32>> = None;
        let mut pcol_buf: Option<CudaSlice<f32>> = None; // penalized p-column scratch
        // MEMRA_SPEC_SETUP_TRACE=1 (diagnostics): per-call wall decomposition of the burst
        // SETUP + TAIL segments (the round loop's internals are MEMRA_SPEC_PHASE's job) —
        // built to pin the serve per-burst fixed cost (research/spec-serving-20260801).
        let setup_trace = std::env::var("MEMRA_SPEC_SETUP_TRACE").as_deref() == Ok("1");
        let t_ent = std::time::Instant::now();

        // SESSION-AFFINITY TURN CHECKPOINT (lane/session-affinity, 2026-08-05): capture the
        // PROMPT-END boundary state so a LATER turn can rewind here and re-prime only its own
        // delta instead of the whole conversation. See `SpecCheckpoint` for why this boundary is
        // the one that matters (a history-rewriting client mutates what the session GENERATED,
        // so the next turn's prompt agrees with this one up to exactly here).
        //
        // WHERE — AND WHY THIS EXACT LINE. Right after the trunk prime, BEFORE the init feed
        // (`decode_step_h(last_token)`) and before round 0: the last instant at which the caches
        // hold exactly `base + prompt.len()` rows and nothing generated.
        //
        // This was WRONG in the first cut of this lane: the capture sat after the draft-KV fill,
        // which is also after the init feed, so `cache.pos` was `base + prompt.len() + 1` — the
        // boundary included the FIRST GENERATED TOKEN. That token is the first thing inside the
        // `<think>` block the client strips, so every later turn's diff diverged exactly one
        // token below the checkpoint and affinity declined 100% of the time. Measured on the
        // owner regime: "history diverged at 12233 of checkpoint 12234". The off-by-one made the
        // whole mechanism inert while looking, from the outside, like a working
        // correctness-declines-safely path — hence the decline log carries the offsets.
        //
        // The full-attn planes are `len`-truncatable so the snapshot copies only the GDN conv/ssm
        // state (the reason a spec session could not rewind before). The draft scratch needs no
        // copy: rows below the boundary are rewritten by the next turn's own fill.
        //
        // WHEN: non-empty prime only. An empty-suffix continuation burst adds no prompt boundary
        // (its "prompt end" IS the previous checkpoint's, already held), so it keeps the existing
        // checkpoint rather than replacing it with a strictly worse one.
        //
        // FAILURE IS SILENT BY DESIGN: on a VRAM-tight rig the snapshot alloc can fail. That
        // costs the NEXT turn its rewind (it re-primes fully, today's behavior) and must never
        // fail the burst that is already running — so the error is swallowed, loud only under
        // MEMRA_DEBUG_SPEC.
        //
        // STABLE-BOUNDARY OVERRIDE (lane/frspec-multiturn-cache, 2026-08-21): the prompt-end
        // posture above was DISPROVED for the think-posture template class — the prompt's own
        // tail is the live generation header (`<|im_start|>assistant\n<think>\n`) that the
        // next turn's re-render replaces, so the diff diverged a couple tokens BELOW the
        // checkpoint and affinity declined 100% of multi-turn agent traffic (the same class
        // the plain tier fixed on 2026-08-09 via `plain_checkpoint_boundary`; the port to the
        // spec tier is this lane). When the worker armed `ckpt_at`, the capture happened at
        // that stop inside the prime above (`ckpt_early`) and is installed here instead;
        // capture-attempted-but-failed clears the slot exactly like the legacy arm.
        if let Some(slot) = sess_ckpt_slot {
            if let Some(early) = ckpt_early {
                if early.is_none() && std::env::var("MEMRA_DEBUG_SPEC").is_ok() {
                    eprintln!(
                        "[spec] stable-boundary turn checkpoint skipped; \
                               next turn re-primes in full"
                    );
                }
                *slot = early;
            } else if !continuation {
                let pos = cache.pos;
                debug_assert_eq!(
                    pos,
                    base + prompt.len(),
                    "turn checkpoint must sit at the prompt end, before the init feed"
                );
                let anchor: Result<CudaSlice<f32>, Box<dyn std::error::Error>> =
                    if let Some(ph) = &prompt_h {
                        // hidden of the LAST primed row = the predecessor anchor at this
                        // boundary (exactly what a fresh prime of committed[..pos] leaves in
                        // last_h, and what the next prime's fill reads for its first row).
                        let np = prompt.len();
                        e.uninit(n_embd).and_then(|mut a| {
                            e.copy_view_into(
                                &mut a,
                                0,
                                &ph.slice((np - 1) * n_embd..np * n_embd),
                                n_embd,
                            )?;
                            Ok(a)
                        })
                    } else {
                        Err("no prompt hiddens".into())
                    };
                match (cache.snapshot(e), anchor) {
                    (Ok(snap), Ok(last_h)) => {
                        *slot = Some(SpecCheckpoint { snap, pos, last_h });
                    }
                    (s, a) => {
                        *slot = None; // a stale checkpoint would rewind to the WRONG boundary
                        if std::env::var("MEMRA_DEBUG_SPEC").is_ok() {
                            let err = s
                                .err()
                                .map(|e| e.to_string())
                                .or_else(|| a.err().map(|e| e.to_string()))
                                .unwrap_or_default();
                            eprintln!(
                                "[spec] turn checkpoint skipped ({err}); \
                                       next turn re-primes in full"
                            );
                        }
                    }
                }
            }
        }
        // INIT FEED — skipped on a pending carry: last_token (the carried bonus) is NOT in the
        // caches and must NOT be fed solo; round 0's batched verify commits it as col 0. Its
        // seed/anchor hidden is the carried last_h (copied below); last_pred is dead in the
        // pending path (t_pred reads verify col 0 — the accept walk overwrites it).
        let mut last_pred = 0u32;
        let mut last_col_logits: Option<CudaSlice<f32>> = None;
        // CONSTRAINED: the init feed's logits back the (n_acc==0, base==0) masked-argmax
        // recompute in the grammar-truncation walk — retained host-side, round 0 only.
        let mut init_logits_host: Option<Vec<f32>> = None;
        let h_seed0: CudaSlice<f32> = if carried_pending.is_none() {
            let (init_logits, h) = self.spec_target_step_h(e, last_token, &mut *cache)?;
            last_pred = argmax(&init_logits) as u32;
            if constraint.is_some() {
                init_logits_host = Some(init_logits.clone());
            }
            // sampled mode: p-distribution after last_token, for the j==0/base==0 accept test.
            if sampled {
                last_col_logits = Some(e.htod(&init_logits)?);
            }
            h
        } else {
            // predecessor-row anchor: hidden of the last COMMITTED row (the carry contract).
            let lh = sess_tail
                .as_ref()
                .unwrap()
                .1
                .as_ref()
                .expect("pending carry requires last_h");
            e.clone_dtod(lh)?
        };
        let t_init = t_ent.elapsed();
        let mut last_col_stats: Option<(f32, f32, f32)> = None;
        // PERSISTENT h_seed buffer (allocated BEFORE any graph capture so no captured scratch can
        // alias it): every path that updates the round seed copies INTO it — no per-round allocs,
        // stable pointer for the graph-draft round-start copy.
        let mut h_seed_buf = e.clone_dtod(&h_seed0)?;
        // Predecessor-pairing trackers: `fill_prev` = trunk hidden AT the last COMMITTED row (the
        // predecessor of the next verify's col 0 — the reference's carried pending-h analogue;
        // also the predecessor-row hidden for the round-0 legacy-replay seed). At round 0 that
        // row is last_token's own (h_seed0). The chain step-0 seed under the pairing default =
        // hidden of the row BEFORE last_token = the prompt's last row at round 0 (h_seed_buf
        // overwritten below).
        let mut fill_prev = e.clone_dtod(&h_seed0)?;
        {
            if let Some(ph) = &prompt_h {
                let np = prompt.len();
                e.copy_view_into(
                    &mut h_seed_buf,
                    0,
                    &ph.slice((np - 1) * n_embd..np * n_embd),
                    n_embd,
                )?;
            } else if continuation
                && let Some((_, lh, _, _, _)) = sess_tail.as_ref()
                && let Some(lh) = lh.as_ref()
            {
                e.copy_into(&mut h_seed_buf, 0, lh, n_embd)?;
            }
        }
        // Persistent device prediction slots for the accept walk (max k+1 verify columns).
        let mut preds_d = e.alloc_u32_zeroed(k + 2)?;

        let debug_spec = std::env::var("MEMRA_DEBUG_SPEC").is_ok();
        let fork_mode = OptiForkGateMode::configured();
        // MEMRA_SPEC_STATS=1: per-slot accept histogram + draft-length histogram, printed once at
        // the end. Metric normalization vs the reference engine: BOTH engines count
        // accepted/drafted where the chain stopped at p-min and the sub-threshold token is
        // discarded uncounted — per-slot decay + chain-length mix are the extra dimensions.
        let spec_stats = std::env::var("MEMRA_SPEC_STATS").is_ok();
        let mut st_drafted = vec![0usize; k];
        let mut st_accepted = vec![0usize; k];
        let mut st_len_hist = vec![0usize; k + 1];
        let mut st_full = 0usize;
        // P-MIN CONFIDENCE GATE (MEMRA_SPEC_PMIN, the serve script's --spec-draft-p-min mechanism):
        // stop the draft chain early when the head's softmax confidence in its own pick drops
        // below p_min. Hoisted above the loop: the graph capture bakes the prob kernels iff on.
        static PMIN: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
        let p_min = *PMIN.get_or_init(|| {
            std::env::var("MEMRA_SPEC_PMIN")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0.0)
        });
        // ZERO-DRAFT ROUNDS (MEMRA_SPEC_PMIN0=1, vendored from llama.cpp's draft gating): let the
        // p-min gate apply at j==0 too, so a low-confidence round drafts NOTHING and the verify
        // batch is just the pending bonus (m=1 = a plain decode step). llama's 35B win rides
        // exactly this — draft acceptance 76% at mean len 2.5 because unpredictable stretches
        // never pay draft+verify overhead. Only legal when a pending bonus exists (an empty
        // verify batch is not); the j==0 exemption stays for pending-less rounds.
        let pmin0 = std::env::var("MEMRA_SPEC_PMIN0")
            .map(|v| v == "1")
            .unwrap_or(false);

        // --- GRAPH DRAFT setup: persistent I/O buffers + ONE capture (2 warmups inside). The
        // warmups mutate scratch len_d / pos / tok / seed — all reset at every round start, so the
        // only restore needed is the scratch counter. Capture failure (e.g. a non-capturable
        // cuBLAS path in an exotic head) falls back to the eager draft chain.
        // PER-SESSION PERSISTENCE (2026-08-01): session calls reuse the DraftGraphCtx parked on
        // the SpecSession — the capture (2 warmup head forwards + instantiate) ran ONCE at the
        // session's first burst, not per burst (measured ~16ms/burst fixed cost on H100 q27,
        // research/spec-serving-20260801). Reuse is pointer-exact: the graph bakes the session's
        // own scratch KV (never realloc'd), the model's resident embedding, the OnceLock p_min,
        // and the g_* buffers carried in the ctx — replay dispatch is identical to a fresh
        // capture, so draft tokens are bit-identical (drafts never decide exactness anyway; the
        // verify arbitrates). Single-shot calls (sess=None) build a fresh ctx and drop it.
        let mut dctx: DraftGraphCtx = match sess_draft_slot.as_mut().and_then(|s| s.take()) {
            Some(c) => c,
            None => DraftGraphCtx::new(e, n_embd, if sampled { d_vocab } else { 1 })?,
        };
        // FAIL-SAFE (step-OOM park replay): pre-mark both fallback flags so no capture arm
        // below can fire — LOUD once per replayed session through the standard WARN line.
        if sess_capture_disabled {
            let reason =
                "session replayed after a step-OOM park; draft capture disabled (fail-safe)";
            let flip = dctx.failed.mark_greedy(reason);
            let flip_s = dctx.failed.mark_sampled(reason);
            if let Some(line) = flip.or(flip_s) {
                eprintln!("{line}");
            }
        }
        // A session that ran greedy bursts first sized g_q/g_perturb at 1; a sampled resume
        // needs d_vocab. Realloc is legal exactly while graph_s is None (nothing baked them).
        if sampled && dctx.g_q.len() < d_vocab {
            dctx.g_q = e.zeros(d_vocab)?;
            dctx.g_perturb = e.zeros(d_vocab)?;
        }
        // DRAFT-SIDE GRAMMAR MASK (lane/draft-mask, 2026-08-04): the drafter samples the
        // grammar's legal set, so proposals are legal BY CONSTRUCTION and the verify-side
        // truncation (the correctness backstop) stops cutting every tight-schema round.
        // The mask is one node inside the captured draft chain — presence is a CAPTURE-TIME
        // shape, so a parked graph of the other shape is dropped and recaptured.
        let dmask_on = constraint
            .as_deref()
            .is_some_and(|c| c.draft_mask_enabled());
        let dmask_words = if dmask_on { d_vocab.div_ceil(32) } else { 0 };
        if dmask_on && dctx.g_dmask.len() < dmask_words {
            dctx.g_dmask = e.alloc_u32_zeroed(dmask_words)?;
            dctx.graph = None; // the old capture baked the old (or no) mask pointer
            dctx.chain = None; // chain last-row graphs bake the same pointer
            dctx.failed.clear_greedy();
            dctx.keeper.clear();
        }
        if (dctx.graph.is_some() || dctx.chain.is_some()) && dctx.graph_masked != dmask_on {
            dctx.graph = None;
            dctx.chain = None;
            dctx.failed.clear_greedy();
            dctx.keeper.clear();
        }
        // MULTI-HEAD CHAIN mode (mtp_extra non-empty — step37's 3-head shipping shape): the
        // step-modulo prefix-replay chain captures PER-HEAD single-row graphs
        // (`DraftChainGraphs`) instead of the one self-feeding graph below; the single-head
        // capture arms are untouched and unreachable in this mode (the launch arms branch the
        // same way). This removes the historical `mtp_extra.is_empty()` capture exclusion —
        // and with it the silent no-attempt hole: a chain capture that FAILS now trips the
        // same LOUD draft-graph WARN as a single-head failure.
        let chain_mode = !self.mtp_extra.is_empty();
        // ---- PRE-CAPTURE VRAM RESERVE CHECK + PER-SESSION DRAFT-STATE MEASUREMENT ----
        // (lane/step37-vram-admission-20260830). `cap_eff0` opens the measurement bracket:
        // when any capture succeeds in THIS call, the effective-free delta across the whole
        // capture section is recorded as the model's per-session draft-state high-water
        // (admission charges it per spec-capable session — this state was charged at ZERO
        // before the lane). The reserve check runs BEFORE any capture arm can allocate: a
        // refused capture trips the same LOUD once-per-flip WARN class as a failed one, but
        // with the card's headroom still intact (the owner's single-session OOM was a capture
        // attempt walking the card to the edge and stranding the eager fallback at 5 MiB free).
        let cap_eff0 = e
            .ctx()
            .mem_get_info()
            .ok()
            .map(|(f, _)| f.saturating_add(e.pool_cached_bytes()));
        // Peak instrument for the same bracket: the CAPTURE-TIME peak (warmup transients +
        // instantiate scratch, alive together) dwarfs the parked delta — measured on the
        // owner shape: a capture whose PARKED state reads ~2.6GB walked a ~7GB-free card to
        // OOM mid-capture. Reset the pool watermark here; read it at bracket end.
        let _ = e.pool_high_water_reset();
        let cap_used0 = e.pool_reserved_used().1;
        let mut captured_now = false;
        let mut capture_oom_entry_eff: Option<usize> = None;
        let capture_need = {
            let observed = self.draft_session_admission_bytes();
            if observed > 0 {
                observed
            } else {
                draft_capture_bootstrap_estimate(
                    if chain_mode { self.mtp_head_count() } else { 1 },
                    k,
                    d_vocab,
                    n_embd,
                )
            }
        };
        if spec_capture_gate_on()
            && graph_draft
            && !sampled
            && !dctx.failed.greedy_failed()
            && ((chain_mode && dctx.chain.is_none() && mtp_chain_graph_on())
                || (!chain_mode && dctx.graph.is_none()))
            && let Some(reason) = capture_headroom_refusal(e, capture_need)
            && let Some(line) = dctx.failed.mark_greedy(&reason)
        {
            eprintln!("{line}");
        }
        if graph_draft
            && !sampled
            && chain_mode
            && dctx.chain.is_none()
            && !dctx.failed.greedy_failed()
        {
            if mtp_chain_graph_on() {
                let heads_n = self.mtp_head_count();
                let DraftGraphCtx {
                    g_tok,
                    g_pos,
                    g_seed,
                    g_p,
                    g_dmask,
                    ..
                } = &mut dctx;
                if dmask_on {
                    e.htod_u32_into(g_dmask, &vec![u32::MAX; dmask_words])?;
                }
                let g_dmask_ro: &CudaSlice<u32> = &*g_dmask;
                let with_prob = p_min > 0.0;
                // CAPTURE-RETAIN (#68 fix): one keeper for the whole chain — every graph's
                // warmup transients stay pinned as long as any of them replays.
                let cap_res = (|| -> Result<DraftChainGraphs, Box<dyn std::error::Error>> {
                    // dcw door: same warmup headroom pre-arm as the single-head capture
                    // below — every plane, because each head's capture warmups append on
                    // its OWN plane. INSIDE the fallible closure (vram-admission lane): an
                    // OOM here used to `?` out of the whole burst as a step error; now it
                    // is a capture failure — LOUD WARN, eager chain serves.
                    if step35_draft_dcw_on() {
                        scratch.ensure_dcw_headroom(e, k + 2)?;
                    }
                    let mut interior = Vec::with_capacity(heads_n);
                    let mut last = Vec::with_capacity(heads_n);
                    let mut keeper: Vec<Box<dyn std::any::Any + Send>> = Vec::new();
                    for hi in 0..heads_n {
                        let head = self.mtp_head_at(hi);
                        // interior row: KV append + carrier only (`with_head=false` — the
                        // eager chain discards interior logits too, so this is the same
                        // consumed-byte program minus the dead full-vocab head matmul).
                        let (g, keep) = e.capture_graph_retained(|e| {
                            self.mtp_head_forward_cap(
                                e,
                                head,
                                g_tok,
                                g_pos,
                                g_seed,
                                g_p,
                                &mut *scratch,
                                hi,
                                false,
                                false,
                                embd_gpu.expect("graph draft requires resident embedding"),
                                embd_qt,
                                embd_rb,
                                d_vocab,
                                None,
                                None,
                                None,
                            )
                        })?;
                        // the warmups appended rows on plane hi; rewind before the next
                        // capture so successive warmups never outrun the pre-armed headroom.
                        scratch.set_plane_len(e, hi, base)?;
                        interior.push(g);
                        keeper.extend(keep);
                        // last row: head matmul + greedy argmax tail (+ p when the policy
                        // reads it, + the grammar-mask node when constrained).
                        let (g2, keep2) = e.capture_graph_retained(|e| {
                            self.mtp_head_forward_cap(
                                e,
                                head,
                                g_tok,
                                g_pos,
                                g_seed,
                                g_p,
                                &mut *scratch,
                                hi,
                                with_prob,
                                true,
                                embd_gpu.expect("graph draft requires resident embedding"),
                                embd_qt,
                                embd_rb,
                                d_vocab,
                                None,
                                None,
                                if dmask_on {
                                    Some((g_dmask_ro, dmask_words))
                                } else {
                                    None
                                },
                            )
                        })?;
                        scratch.set_plane_len(e, hi, base)?;
                        last.push(g2);
                        keeper.extend(keep2);
                    }
                    Ok(DraftChainGraphs {
                        interior,
                        last,
                        _keeper: keeper,
                    })
                })();
                match cap_res {
                    Ok(cg) => {
                        scratch.set_len(e, base)?;
                        // POSITIVE engagement receipt (the 3a lesson: a WARN-free boot is
                        // NOT evidence of capture — the captured state must name itself).
                        eprintln!(
                            "[mtp-chain-graph] captured mode=greedy heads={heads_n} \
                             interior={heads_n} last={heads_n} masked={}",
                            dmask_on as u8
                        );
                        dctx.chain = Some(cg);
                        dctx.graph_masked = dmask_on;
                        captured_now = true;
                    }
                    Err(err) => {
                        scratch.set_len(e, base)?;
                        // LOUD flip (audit Q2): a dropped draft graph is a coverage loss,
                        // never silent — now including the multi-head shipping shape.
                        // OOM RECOVERY (vram-admission lane): a failed attempt's freed
                        // transients sit CACHED in the async pool where the driver cannot
                        // see them; trim them back so the eager fallback (and any driver-
                        // side allocation) actually has the headroom the free suggests.
                        let mut reason = err.to_string();
                        if capture_err_is_oom(&reason) {
                            capture_oom_entry_eff = capture_oom_entry_eff.max(cap_eff0);
                            let trimmed = e.pool_trim_to_zero();
                            if trimmed > 0 {
                                reason.push_str(&format!(
                                    "; pool trimmed {}MB back to the driver",
                                    trimmed / (1 << 20)
                                ));
                            }
                        }
                        if let Some(line) = dctx.failed.mark_greedy(&reason) {
                            eprintln!("{line}");
                        }
                    }
                }
            } else {
                // Disarmed by MEMRA_MTP_CHAIN_GRAPH=0: say so once per process — the OFF arm
                // must be attributable in a boot log, never inferable from silence.
                static NOTE: std::sync::Once = std::sync::Once::new();
                NOTE.call_once(|| {
                    eprintln!(
                        "[spec] multi-head draft-chain capture disarmed \
                         (MEMRA_MTP_CHAIN_GRAPH=0); eager chain serves this shape"
                    );
                });
            }
        }
        if graph_draft
            && !sampled
            && !chain_mode
            && dctx.graph.is_none()
            && !dctx.failed.greedy_failed()
        {
            let DraftGraphCtx {
                g_tok,
                g_pos,
                g_seed,
                g_p,
                g_dmask,
                ..
            } = &mut dctx;
            // capture-time contents: ALL-ONES (ban nothing). A replay only ever runs after the
            // host uploads the position's real words, so the warmups stay grammar-free.
            if dmask_on {
                e.htod_u32_into(g_dmask, &vec![u32::MAX; dmask_words])?;
            }
            let g_dmask_ro: &CudaSlice<u32> = &*g_dmask;
            // CAPTURE-RETAIN (#68 fix): the warmup transients' pool addresses are baked into the
            // captured graph; the keeper pins them for the graph's lifetime. capture_graph (non-
            // retained) freed them at exit — safe for one-shot generate_spec (nothing else touches
            // the pool between replays) but WRONG for sessions: burst-boundary prime/fill/commit
            // passes (and, in serve, other sessions) recycle those addresses and the replay then
            // clobbers live buffers — the ST serve-spec corruption (research/serve-st-20260803).
            let cap_res = (|| {
                // dcw door: the capture warmups append device-counter rows the capture body
                // cannot rebase for; pre-arm ring headroom host-side (no-op on flat planes /
                // room-enough rings, and the door-off path is untouched). INSIDE the fallible
                // closure (vram-admission lane): an OOM here is a capture failure, not a
                // burst-killing step error.
                if step35_draft_dcw_on() {
                    scratch.ensure_dcw_headroom(e, k + 2)?;
                }
                e.capture_graph_retained(|e| {
                    self.mtp_head_forward_cap(
                        e,
                        mtp,
                        g_tok,
                        g_pos,
                        g_seed,
                        g_p,
                        &mut *scratch,
                        0,
                        p_min > 0.0 || fork_mode == OptiForkGateMode::Controller,
                        true,
                        embd_gpu.expect("graph draft requires resident embedding"),
                        embd_qt,
                        embd_rb,
                        d_vocab,
                        None,
                        None,
                        if dmask_on {
                            Some((g_dmask_ro, dmask_words))
                        } else {
                            None
                        },
                    )
                })
            })();
            match cap_res {
                Ok((g, keep)) => {
                    scratch.set_len(e, base)?;
                    dctx.graph = Some(g);
                    dctx.graph_masked = dmask_on;
                    dctx.keeper = keep;
                    captured_now = true;
                }
                Err(err) => {
                    scratch.set_len(e, base)?;
                    // LOUD flip (audit Q2): a dropped draft graph is a coverage loss, never
                    // silent. Once per flip — mark returns None on an already-failed ctx.
                    let mut reason = err.to_string();
                    if capture_err_is_oom(&reason) {
                        capture_oom_entry_eff = capture_oom_entry_eff.max(cap_eff0);
                        let trimmed = e.pool_trim_to_zero();
                        if trimmed > 0 {
                            reason.push_str(&format!(
                                "; pool trimmed {}MB back to the driver",
                                trimmed / (1 << 20)
                            ));
                        }
                    }
                    if let Some(line) = dctx.failed.mark_greedy(&reason) {
                        eprintln!("{line}");
                    }
                }
            }
        }
        // --- SAMPLED GRAPH DRAFT setup (step 3 of the sampled-spec arc): a SECOND capture, own
        // graph object, built only when sampled && graph-eligible — the greedy capture above is
        // untouched (and skipped when sampled: its graph would never be launched). Same head
        // forward, but the in-graph argmax reads GUMBEL-PERTURBED logits; the Philox event
        // counter lives in the persistent device g_ctr (bumped in-graph, host-seeded from sctr
        // once per round); the raw head logits land in the persistent g_q for the host's
        // per-replay async D2D into the round's q slot (q_slots, K x d_vocab, allocated once).
        // seed/temp are capture-time constants — baked into graph_s, so a pool-resumed request
        // with a different (seed, temp, k) drops the parked sampled graph and recaptures.
        // COST OF THE FRESH-SEED SERVE DEFAULT (dogfood F4, 2026-08-04): omitting `seed` on a
        // serve request now draws fresh per-request entropy (it used to default to a pinned 0),
        // so a seed-omitting request that RESUMES a parked spec session finds an s_key baked
        // with the PREVIOUS request's seed and pays one recapture. Bounded, and it does not
        // reopen the ~16ms/burst regression the persistent ctx exists to fix: a session's seed
        // is fixed for its whole lifetime (worker.rs reads s.sampler.seed() per burst), so
        // this compare misses at most ONCE per resumed request — the first burst recaptures
        // and every later burst in that request replays. A client that wants the parked graph
        // AND reproducibility supplies an explicit `seed`, honored exactly, which keeps s_key
        // stable across its whole conversation.
        // COMPOSITION RULE (fspec x gsd merge): the in-graph chain samples from the RAW
        // softmax — it can hold neither per-row filter stats nor the varying penalty history.
        // The sampled graph therefore engages only in the PURE-TEMP regime; filters/penalties
        // force the eager draft (which computes stats/penalties per row).
        // KEY THE WHOLE REGIME, not just the baked constants (lane/graph-s-key-exactness-
        // 20260819). `s_key` used to be `(seed, temp, k)`; the filters and penalties were left
        // out, so a filtered request resuming a session that parked a PURE-TEMP graph kept it —
        // and the launch site never re-asked `pure_temp`. See [`SampledGraphKey`] for what that
        // costs (an unconditional accept of out-of-head draft tokens, i.e. an exactness bug on
        // the request shape the vendor-default flip makes the majority).
        let s_key = SampledGraphKey::new(sp_seed, sp_temp, k, sp.top_k, sp.top_p, sp.min_p, pen_on);
        let pure_temp = s_key.pure_temp();
        // The regime the sampled graph may be captured/launched in: pure-temp always;
        // truncation-filtered when the filtered-capture door is on (the filter runs
        // IN-GRAPH — lane/step37-draft-graph-serving-20260830); penalties never.
        let s_capturable = s_key.graph_capturable();
        if sampled && dctx.s_key.is_some_and(|old| old != s_key) {
            dctx.graph_s = None;
            dctx.chain_s = None;
            dctx.failed.clear_sampled();
            dctx.s_key = None;
            dctx.q_slots.clear();
            dctx.keeper_s.clear();
        }
        // PRE-CAPTURE VRAM RESERVE CHECK, sampled arms (vram-admission lane): same contract
        // as the greedy check above — refuse BEFORE allocating, LOUD once, eager serves.
        if spec_capture_gate_on()
            && graph_draft
            && sampled
            && s_capturable
            && !dctx.failed.sampled_failed()
            && ((chain_mode && dctx.chain_s.is_none() && mtp_chain_graph_on())
                || (!chain_mode && dctx.graph_s.is_none()))
            && let Some(reason) = capture_headroom_refusal(e, capture_need)
            && let Some(line) = dctx.failed.mark_sampled(&reason)
        {
            eprintln!("{line}");
        }
        // FILTERED capture nodes need q slots sized d_vocab AND the stat slots; the pure-temp
        // body leaves g_th/g_z/g_mx untouched (they exist from ctx creation either way).
        if graph_draft
            && sampled
            && s_capturable
            && chain_mode
            && dctx.chain_s.is_none()
            && !dctx.failed.sampled_failed()
        {
            if mtp_chain_graph_on() {
                let heads_n = self.mtp_head_count();
                let filtered = s_key.filtered();
                let DraftGraphCtx {
                    g_tok,
                    g_pos,
                    g_seed,
                    g_p,
                    g_ctr,
                    g_perturb,
                    g_q,
                    g_rows0,
                    g_th,
                    g_z,
                    g_mx,
                    ..
                } = &mut dctx;
                let with_prob = p_min > 0.0;
                let cap_res = (|| -> Result<DraftChainGraphs, Box<dyn std::error::Error>> {
                    // dcw pre-arm INSIDE the fallible closure (vram-admission lane): an OOM
                    // here is a capture failure with the LOUD WARN, never a step error.
                    if step35_draft_dcw_on() {
                        scratch.ensure_dcw_headroom(e, k + 2)?;
                    }
                    let mut interior = Vec::with_capacity(heads_n);
                    let mut last = Vec::with_capacity(heads_n);
                    let mut keeper: Vec<Box<dyn std::any::Any + Send>> = Vec::new();
                    for hi in 0..heads_n {
                        let head = self.mtp_head_at(hi);
                        // interior row: no head, no draw — shared shape with the greedy
                        // chain's interior, captured per mode for keeper-lifetime hygiene.
                        let (g, keep) = e.capture_graph_retained(|e| {
                            self.mtp_head_forward_cap(
                                e,
                                head,
                                g_tok,
                                g_pos,
                                g_seed,
                                g_p,
                                &mut *scratch,
                                hi,
                                false,
                                false,
                                embd_gpu.expect("graph draft requires resident embedding"),
                                embd_qt,
                                embd_rb,
                                d_vocab,
                                None,
                                None,
                                None,
                            )
                        })?;
                        scratch.set_plane_len(e, hi, base)?;
                        interior.push(g);
                        keeper.extend(keep);
                        // last row: head matmul + the in-graph categorical draw (filtered
                        // nodes when the request carries filters).
                        let (g2, keep2) = e.capture_graph_retained(|e| {
                            self.mtp_head_forward_cap(
                                e,
                                head,
                                g_tok,
                                g_pos,
                                g_seed,
                                g_p,
                                &mut *scratch,
                                hi,
                                with_prob,
                                true,
                                embd_gpu.expect("graph draft requires resident embedding"),
                                embd_qt,
                                embd_rb,
                                d_vocab,
                                Some(SampledCapArgs {
                                    ctr: &mut *g_ctr,
                                    perturb: &mut *g_perturb,
                                    q_out: &mut *g_q,
                                    seed: sp_seed,
                                    temp: sp_temp,
                                    filt: if filtered {
                                        Some(SampledCapFilter {
                                            rows0: &*g_rows0,
                                            th: &mut *g_th,
                                            z: &mut *g_z,
                                            mx: &mut *g_mx,
                                            top_k: sp.top_k,
                                            top_p: sp.top_p,
                                            min_p: sp.min_p,
                                        })
                                    } else {
                                        None
                                    },
                                }),
                                None,
                                None, // constrained spec is greedy-only
                            )
                        })?;
                        scratch.set_plane_len(e, hi, base)?;
                        last.push(g2);
                        keeper.extend(keep2);
                    }
                    Ok(DraftChainGraphs {
                        interior,
                        last,
                        _keeper: keeper,
                    })
                })();
                match cap_res {
                    Ok(cg) => {
                        scratch.set_len(e, base)?;
                        // NO STRANDED PARTIAL STATE (vram-admission lane): the q-slot allocs
                        // after a successful capture are themselves fallible on a tight card.
                        // A mid-loop failure used to `?` out as a step error, leaving orphan
                        // slots parked on the ctx (wrong count, stale contents) for the next
                        // capture attempt to stack onto. Allocate all-or-nothing: on failure
                        // drop the fresh graphs AND the partial slots, mark the LOUD fallback.
                        dctx.q_slots.clear();
                        let slots = (0..k)
                            .map(|_| e.zeros(d_vocab))
                            .collect::<Result<Vec<_>, _>>();
                        match slots {
                            Ok(slots) => {
                                dctx.q_slots = slots;
                                eprintln!(
                                    "[mtp-chain-graph] captured mode=sampled heads={heads_n} \
                                     interior={heads_n} last={heads_n} filtered={} key={s_key:?}",
                                    s_key.filtered() as u8
                                );
                                dctx.chain_s = Some(cg);
                                dctx.s_key = Some(s_key);
                                captured_now = true;
                            }
                            Err(err) => {
                                drop(cg);
                                dctx.q_slots.clear();
                                let mut reason = format!("q-slot alloc failed: {err}");
                                if capture_err_is_oom(&reason) {
                                    capture_oom_entry_eff = capture_oom_entry_eff.max(cap_eff0);
                                    let trimmed = e.pool_trim_to_zero();
                                    if trimmed > 0 {
                                        reason.push_str(&format!(
                                            "; pool trimmed {}MB back to the driver",
                                            trimmed / (1 << 20)
                                        ));
                                    }
                                }
                                if let Some(line) = dctx.failed.mark_sampled(&reason) {
                                    eprintln!("{line}");
                                }
                            }
                        }
                    }
                    Err(err) => {
                        scratch.set_len(e, base)?;
                        let mut reason = err.to_string();
                        if capture_err_is_oom(&reason) {
                            capture_oom_entry_eff = capture_oom_entry_eff.max(cap_eff0);
                            let trimmed = e.pool_trim_to_zero();
                            if trimmed > 0 {
                                reason.push_str(&format!(
                                    "; pool trimmed {}MB back to the driver",
                                    trimmed / (1 << 20)
                                ));
                            }
                        }
                        if let Some(line) = dctx.failed.mark_sampled(&reason) {
                            eprintln!("{line}");
                        }
                    }
                }
            } else {
                static NOTE_S: std::sync::Once = std::sync::Once::new();
                NOTE_S.call_once(|| {
                    eprintln!(
                        "[spec] multi-head draft-chain capture disarmed \
                         (MEMRA_MTP_CHAIN_GRAPH=0); eager chain serves this shape"
                    );
                });
            }
        }
        if graph_draft
            && sampled
            && s_capturable
            && !chain_mode
            && dctx.graph_s.is_none()
            && !dctx.failed.sampled_failed()
        {
            let filtered = s_key.filtered();
            let DraftGraphCtx {
                g_tok,
                g_pos,
                g_seed,
                g_p,
                g_ctr,
                g_perturb,
                g_q,
                g_rows0,
                g_th,
                g_z,
                g_mx,
                ..
            } = &mut dctx;
            // CAPTURE-RETAIN (#68 fix): same keeper contract as the greedy capture above.
            let cap_res = (|| {
                // dcw pre-arm INSIDE the fallible closure (vram-admission lane): an OOM
                // here is a capture failure with the LOUD WARN, never a step error.
                if step35_draft_dcw_on() {
                    scratch.ensure_dcw_headroom(e, k + 2)?;
                }
                e.capture_graph_retained(|e| {
                    self.mtp_head_forward_cap(
                        e,
                        mtp,
                        g_tok,
                        g_pos,
                        g_seed,
                        g_p,
                        &mut *scratch,
                        0,
                        p_min > 0.0,
                        true,
                        embd_gpu.expect("graph draft requires resident embedding"),
                        embd_qt,
                        embd_rb,
                        d_vocab,
                        Some(SampledCapArgs {
                            ctr: &mut *g_ctr,
                            perturb: &mut *g_perturb,
                            q_out: &mut *g_q,
                            seed: sp_seed,
                            temp: sp_temp,
                            filt: if filtered {
                                Some(SampledCapFilter {
                                    rows0: &*g_rows0,
                                    th: &mut *g_th,
                                    z: &mut *g_z,
                                    mx: &mut *g_mx,
                                    top_k: sp.top_k,
                                    top_p: sp.top_p,
                                    min_p: sp.min_p,
                                })
                            } else {
                                None
                            },
                        }),
                        None,
                        None, // constrained spec is greedy-only — sampled never carries a hook
                    )
                })
            })();
            match cap_res {
                Ok((g, keep)) => {
                    scratch.set_len(e, base)?;
                    // NO STRANDED PARTIAL STATE: all-or-nothing q slots, same contract as
                    // the chain arm above.
                    dctx.q_slots.clear();
                    let slots = (0..k)
                        .map(|_| e.zeros(d_vocab))
                        .collect::<Result<Vec<_>, _>>();
                    match slots {
                        Ok(slots) => {
                            dctx.q_slots = slots;
                            dctx.graph_s = Some(g);
                            dctx.s_key = Some(s_key);
                            dctx.keeper_s = keep;
                            captured_now = true;
                        }
                        Err(err) => {
                            drop(g);
                            drop(keep);
                            dctx.q_slots.clear();
                            let mut reason = format!("q-slot alloc failed: {err}");
                            if capture_err_is_oom(&reason) {
                                capture_oom_entry_eff = capture_oom_entry_eff.max(cap_eff0);
                                let trimmed = e.pool_trim_to_zero();
                                if trimmed > 0 {
                                    reason.push_str(&format!(
                                        "; pool trimmed {}MB back to the driver",
                                        trimmed / (1 << 20)
                                    ));
                                }
                            }
                            if let Some(line) = dctx.failed.mark_sampled(&reason) {
                                eprintln!("{line}");
                            }
                        }
                    }
                }
                Err(err) => {
                    scratch.set_len(e, base)?;
                    // LOUD flip (audit Q2): same contract as the greedy capture above.
                    let mut reason = err.to_string();
                    if capture_err_is_oom(&reason) {
                        capture_oom_entry_eff = capture_oom_entry_eff.max(cap_eff0);
                        let trimmed = e.pool_trim_to_zero();
                        if trimmed > 0 {
                            reason.push_str(&format!(
                                "; pool trimmed {}MB back to the driver",
                                trimmed / (1 << 20)
                            ));
                        }
                    }
                    if let Some(line) = dctx.failed.mark_sampled(&reason) {
                        eprintln!("{line}");
                    }
                }
            }
        }
        // ---- PER-SESSION DRAFT-STATE MEASUREMENT bracket end (vram-admission lane): when a
        // capture landed in THIS call, the effective-free delta across the capture section is
        // this session's parked draft-graph state (keepers + q slots + instantiated graphs'
        // backing). Recorded as a model-owned high-water; admission charges it per
        // spec-capable session (see `draft_session_admission_bytes`).
        if captured_now
            && let Some(eff0) = cap_eff0
            && let Ok((f1, _)) = e.ctx().mem_get_info()
        {
            let eff1 = f1.saturating_add(e.pool_cached_bytes());
            let parked_delta = eff0.saturating_sub(eff1);
            let (_res_high, used_high) = e.pool_high_water_reset();
            let peak_delta = used_high.saturating_sub(cap_used0);
            let observed = parked_delta.max(peak_delta);
            if observed > 0
                && let Some(hw) = self.record_draft_state_bytes(observed)
            {
                eprintln!(
                    "[spec] draft-session state high-water: {}MB (max of parked delta {}MB \
                     and capture-time pool peak {}MB; charged per spec admission and gating \
                     future captures)",
                    hw / (1 << 20),
                    parked_delta / (1 << 20),
                    peak_delta / (1 << 20),
                );
            }
        }
        // FAILURE IS AN OBSERVATION TOO: a capture that OOM'd at entry-effective E proved
        // the capture-time peak exceeds E. Feed E into the gauge so every future gate
        // refuses at or below the headroom that just failed (self-healing even when the
        // boot probe is disarmed and the bootstrap estimate was blind).
        if let Some(entry_eff) = capture_oom_entry_eff
            && let Some(hw) = self.record_draft_state_bytes(entry_eff)
        {
            eprintln!(
                "[spec] draft-session capture appetite floor raised to {}MB: a capture \
                 attempt OOM'd with that much effective free (failure-observed bound)",
                hw / (1 << 20)
            );
        }
        // ---- EXACTNESS GUARD, the enforceable half (lane/graph-s-key-exactness-20260819,
        // widened by lane/step37-draft-graph-serving-20260830) ----
        // With the filters and penalties in `s_key`, a graph that SURVIVED the drop above was
        // captured under THIS request's exact regime, and capture requires `graph_capturable`
        // (pure-temp, or filtered with the in-graph filter nodes; never penalties) — so a
        // parked graph implies both. That implication is the whole exactness argument for the
        // graph arm, so it is asserted here rather than assumed: a future change that widens
        // the capture condition, narrows the key, or copies a `DraftGraphCtx` across regimes
        // fails LOUDLY at this line instead of silently drafting from a distribution the
        // verify never reconstructs. Release builds refuse the graph (drop it, draft eager)
        // rather than launching it; the launch site re-tests the regime independently.
        if sampled
            && (dctx.graph_s.is_some() || dctx.chain_s.is_some())
            && (!s_capturable || dctx.s_key != Some(s_key))
        {
            debug_assert!(
                false,
                "sampled draft graph parked under {:?} survived into a request outside its \
                 capture regime (top_k={} top_p={} min_p={} pen_on={} capturable={}): the \
                 in-graph draw and the verify's accept test would see different distributions",
                dctx.s_key, sp.top_k, sp.top_p, sp.min_p, pen_on, s_capturable,
            );
            eprintln!(
                "[spec] BUG: dropping a parked sampled draft graph that outlived its capture \
                 regime (s_key={:?}, request top_k={} top_p={} min_p={} pen_on={} \
                 capturable={}); drafting EAGER — the key must carry every field that shapes q",
                dctx.s_key, sp.top_k, sp.top_p, sp.min_p, pen_on, s_capturable,
            );
            dctx.graph_s = None;
            dctx.chain_s = None;
            dctx.s_key = None;
            dctx.q_slots.clear();
            dctx.keeper_s.clear();
        }
        // SKEY PROBE (MEMRA_SKEY_PROBE=1): the burst-entry facts the reachability question turns
        // on — is this request sampled, is it in a regime the sampled graph is legal in, and is
        // a graph PARKED from an earlier request of the same session? The launch arms below
        // print which chain actually ran, so the probe never restates the condition.
        if skey_probe() {
            eprintln!(
                "[skey] burst sampled={} pure_temp={} capturable={} temp={} top_k={} top_p={} \
                 min_p={} pen_on={} k={} graph_draft={} graph_s_parked={} chain_s_parked={} \
                 s_key_parked={:?}",
                sampled as u8,
                pure_temp as u8,
                s_capturable as u8,
                sp_temp,
                sp.top_k,
                sp.top_p,
                sp.min_p,
                pen_on as u8,
                k,
                graph_draft as u8,
                dctx.graph_s.is_some() as u8,
                dctx.chain_s.is_some() as u8,
                dctx.s_key,
            );
        }
        let t_cap = t_ent.elapsed();
        // PERSISTENT DRAFT KV: fill the MTP block's K/V for every prompt position from the exact
        // trunk hiddens collected during prime — ONE batched K/V-only pass (overwrites any
        // capture-warmup garbage; capture left len at 0). last_token (the init feed) needs no
        // fill: the first chain step processes it and appends its entry at slot prompt.len().
        if let Some(ph) = &prompt_h {
            // SESSION: rows [0..base) are the previous turns' exact fills (refresh overwrote them
            // with true verify hiddens) — truncate any draft overhang, fill ONLY the suffix at
            // global positions [base..base+tp). Fresh call: base==0, identical to before.
            scratch.set_len(e, base)?;
            // CHUNKED FILL (long-ctx OOM fix, 2026-07-05): mtp_kv_fill's transients scale with its
            // T (concat = T*2*n_embd*4B — 1.5GB at 40k) and its concat loop is 2*T launches. The
            // fill is a pure sequential append, so chunking is exact: each chunk appends its rows
            // at pos0=base+start with the identical per-row math. Same knob as the trunk prime.
            let tp = prompt.len();
            let fill_chunk: usize = if crate::cache::swa_ring_on() {
                crate::hybrid_forward::prime_chunk_tokens(tp, self.layers.len())
            } else {
                // Preserve the flag-OFF schedule byte-for-byte, including the legacy zero value
                // meaning one monolithic fill.
                std::env::var("MEMRA_PRIME_CHUNK")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(4096)
            };
            let fill_chunk = if fill_chunk == 0 { tp } else { fill_chunk };
            // CUDA launch wall (same class as the trunk prime's PRIME_CHUNK_LAUNCH_CAP):
            // a fill call's matmuls can land on the grid.y=m dp4a family, and grid.y caps
            // at 65,535. This loop has no tail fold, so the raw limit is exact:
            // tp <= 65,535 keeps the legacy schedule (monolithic included) byte-for-byte,
            // and larger fills — unreachable before the trunk prime's own cap fix — chunk.
            let fill_chunk = fill_chunk.min(crate::hybrid_forward::CUDA_GRID_YZ_MAX);
            let mut start = 0usize;
            while start < tp {
                let end = (start + fill_chunk).min(tp);
                let tc = end - start;
                {
                    // PREDECESSOR pairing: row i gets h[i-1]; global row 0 a zeros row (the
                    // reference engine's initial pending-h is zeroed too); a session turn's row 0
                    // gets the PREVIOUS turn's last committed hidden (sess.last_h). Per chunk:
                    // rows start..end read h[start-1..end-1] — one dtod into a chunk buffer.
                    let mut phs = e.zeros(tc * n_embd)?;
                    let (src_lo, dst_off) = if start == 0 {
                        (0, n_embd)
                    } else {
                        ((start - 1) * n_embd, 0)
                    };
                    let n_copy = if start == 0 {
                        (tc - 1) * n_embd
                    } else {
                        tc * n_embd
                    };
                    if start == 0
                        && let Some((_, lh, _, _, _)) = sess_tail.as_ref()
                        && let Some(lh) = lh.as_ref()
                    {
                        e.copy_into(&mut phs, 0, lh, n_embd)?;
                    }
                    if n_copy > 0 {
                        e.copy_view_into(
                            &mut phs,
                            dst_off,
                            &ph.slice(src_lo..src_lo + n_copy),
                            n_copy,
                        )?;
                    }
                    self.mtp_kv_fill_all(
                        e,
                        &prompt[start..end],
                        &phs,
                        base + start,
                        &mut *scratch,
                        embd_dev,
                    )?;
                }
                start = end;
            }
        }
        // MEMRA_PROFILE_SPEC=2: profiler capture starts HERE — after the prime, so an
        // `nsys -c cudaProfilerApi` capture contains ONLY the round loop (draft/verify/commit).
        // (=1 brackets the whole call in run_spec.rs, prime included.)
        if std::env::var("MEMRA_PROFILE_SPEC").as_deref() == Ok("2") {
            unsafe extern "C" {
                fn cudaProfilerStart() -> i32;
            }
            unsafe {
                cudaProfilerStart();
            }
        }
        // ROUND-STREAM stage (c) 4 (MEMRA_SPEC_STREAM=1, experimental): pre-issued M-round
        // bursts with ZERO per-round host readbacks — the accept/seed/rollback/ring kernels
        // consume each other's device outputs; the host drains the ring every M rounds. v1
        // constraints: greedy, !spec_replay, single-shot, batched-linear layers, no refresh
        // fills (acceptance effect A/B-arbitrated), enters from round 1 (pending guaranteed).
        // NOTE: not gated on the caller's graph_draft (its trunk_dense conjunct turns the 35B
        // MoE off) — the stream capture encloses ONLY the dense MTP head; the head-dense /
        // full-prec / k gates are re-derived here and a failed capture degrades to stream-off.
        let stream_on = crate::spec::spec_stream()
            && !sampled
            && !spec_replay
            && self.mtp_extra.is_empty()
            && constraint.is_none()
            && !session_mode
            && embd_gpu.is_some()
            && !crate::model::full_prec_enabled()
            && k + 2 < 96;
        let mut stream_graph: Option<cudarc::driver::CudaGraph> = None;
        let mut g_tokp2k = e.alloc_u32_zeroed(2 * k.max(1))?;
        if stream_on {
            let cap = e.capture_graph(|e| {
                for j in 0..k.max(1) {
                    self.mtp_head_forward_cap(
                        e,
                        mtp,
                        &mut dctx.g_tok,
                        &mut dctx.g_pos,
                        &mut dctx.g_seed,
                        &mut dctx.g_p,
                        &mut *scratch,
                        0,
                        true,
                        true,
                        embd_gpu.expect("round stream requires resident embedding"),
                        embd_qt,
                        embd_rb,
                        d_vocab,
                        None,
                        Some((&mut g_tokp2k, j, d2t_dev.as_ref())),
                        None, // round-stream requires constraint.is_none() (see stream_on)
                    )?;
                }
                Ok(())
            });
            match cap {
                Ok(g) => {
                    scratch.set_len(e, 0)?;
                    stream_graph = Some(g);
                }
                Err(err) => {
                    scratch.set_len(e, 0)?;
                    if debug_spec {
                        eprintln!("[spec] stream-graph capture failed ({err}); stream off");
                    }
                }
            }
        }
        let stream_active = stream_on && stream_graph.is_some();
        if debug_spec {
            eprintln!(
                "[spec] stream_on={stream_on} env={} samp={sampled} dg={} captured={} active={stream_active} session={session_mode} replay={spec_replay}",
                crate::spec::spec_stream(),
                dctx.graph.is_some(),
                stream_graph.is_some()
            );
        }
        let t_v_s = k + 1;
        // ROUND-STREAM buffers + ptr tables now live in the model-generic round_stream
        // module (extracted 2026-07-12; the gemma burst reuses them).
        let sb = crate::round_stream::StreamBufs::new(e, k, crate::spec::spec_stream_m())?;
        let crate::round_stream::StreamBufs {
            mut vtok_d,
            mut brk_d,
            mut pend_d,
            last_pred_d,
            mut pos_ctr,
            mut pos_start_d,
            mut ring_d,
            acc_d: mut stream_acc,
            m_rounds,
            k: _,
        } = sb;
        let stream_ptrs: Option<CudaSlice<u64>> = if stream_active {
            Some(crate::round_stream::kv_len_ptr_table(
                e,
                cache,
                Some(&pos_ctr),
            )?)
        } else {
            None
        };

        let t_fill = t_ent.elapsed();
        let mut round = 0usize;
        // ADAPTIVE DRAFT LENGTH (MEMRA_SPEC_ADAPT=1, opt-in — the gemma_spec accepted-run law,
        // ported 2026-08-01): next round's draft depth = last round's accepted run + 1, clamped
        // to [floor(pos), k_cap] — a miss shrinks the next draft to the miss point + 1,
        // full-accept streaks re-deepen one step per round. NOT the 2026-07-07 acceptance-EMA
        // (that arm measured an HONEST LOSS to static per-class optima — 115.0/85.8/73.4 vs
        // 121.6/92.7/75.6, EMA lag — and was removed 2026-07-08; rig5090.jsonl has the record).
        // The gemma law has no lag class: it reacts within one round, and was worth +7-20% on
        // the gemma cells at unchanged exactness (2026-07-10 flip; floor sweep 2026-07-25;
        // position key 2026-07-26). Signal = n_acc from the round's EXISTING accept readback —
        // zero new syncs; the draft graph is a SINGLE-STEP capture replayed per drafted token,
        // so a per-round depth needs no re-capture (unlike gemma's whole-chain graphs). qwen's
        // in-round p-min cut already shortens chains mid-round, so gemma's one-round-late p-min
        // fold into kc is unnecessary here — the accepted-run law sees the cut via n_acc.
        // Exactness is the verify's job at ANY depth (same contract as p-min variable rounds).
        // DEFAULT OFF on the qwen path until its cells gate a flip (gemma's is default-on).
        // MEASURED 2026-08-01 (H100 GPU-3, interleaved x3, NGEN=256, same-invocation plain
        // denominators; research/qwen-adaptive-k-20260801/): REFUTED on the tuned qwen configs.
        // q27 K=3+HPOST+PMIN=0.3: short +0.8% (noise; law ~idles, len_hist identical), board
        // -1.9%, agentic -0.5%; board PMIN=0 -2.1% (not p-min shadowing — the law itself);
        // floor=1 -2.8% (gemma's floor-collapse, reproduced). q35 K=2 board: -6.4% (52/136
        // rounds shrink to depth 1; no depth to reclaim at K=2). The gemma direction DOES
        // appear at untuned depth-K — q27 K=6 floor=4 +1.5% over fixed K=6 — but stays -3.7%
        // below fixed K=3: same verdict class as the retired EMA arm (honest loss to static
        // per-class optima). Acceptance-rate rises under the law while tokens/round falls —
        // it buys accept-% by adding rounds, and a round's fixed draft+verify cost wins.
        // K=1..8 self-consistency PASS both models with the law ON (exactness held).
        let adapt = std::env::var("MEMRA_SPEC_ADAPT").as_deref() == Ok("1");
        // floor: per-model default keyed on n_embd (gemma's tiering — models with an expensive
        // verify keep deep drafts after a miss); MEMRA_SPEC_ADAPT_FLOOR pins it everywhere.
        let adapt_floor_env: Option<usize> = std::env::var("MEMRA_SPEC_ADAPT_FLOOR")
            .ok()
            .and_then(|v| v.parse().ok());
        let adapt_floor_default: usize = if self.cfg.n_embd as usize >= 3500 {
            4
        } else if self.cfg.n_embd as usize >= 2500 {
            2
        } else {
            1
        };
        let adapt_floor: usize = adapt_floor_env.unwrap_or(adapt_floor_default);
        // position key: past floor_ctx a HIGH floor (>=4) relaxes to 1 — forced-deep drafts
        // turn net-negative at depth (gemma 31B d1736 evidence); MEMRA_SPEC_FLOOR_CTX moves
        // the boundary, an explicit MEMRA_SPEC_ADAPT_FLOOR pins the floor everywhere.
        let floor_ctx: usize = std::env::var("MEMRA_SPEC_FLOOR_CTX")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1024);
        let floor_at = |pos: usize| -> usize {
            if adapt_floor_env.is_some() || pos < floor_ctx {
                adapt_floor
            } else if adapt_floor >= 4 {
                1
            } else {
                adapt_floor
            }
        };
        // cap: MEMRA_SPEC_CAPMAX (gemma semantics, default 7). Binds only under adapt — the
        // fixed-K default path is untouched by this whole block.
        let cap_max: usize = std::env::var("MEMRA_SPEC_CAPMAX")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(7);
        let k_cap = k.min(cap_max).max(1);
        let mut kc = k_cap;
        let mut opti_fork: Option<OptiForkState> = None;
        let mut _opti_walk: Option<crate::pp::PpWalkLease> = None;
        let mut _opti_walk_borrow: Option<crate::pp::PpWalkBorrowGuard> = None;
        let mut fork_snapshot: Option<crate::cache::CacheSnapshot> = None;
        if fork_mode != OptiForkGateMode::Disabled {
            let fence = crate::pp::pp_cuts(self.layers.len());
            let refusal = if !session_mode {
                Some("not-session")
            } else if k != 1 || adapt {
                Some("requires-fixed-k1")
            } else if sampled || constraint.is_some() || spec_replay {
                Some("sampled-constrained-or-replay")
            } else if pipe.is_some() {
                Some("two-session-pipeline")
            } else if !spec_devacc() {
                Some("requires-device-accept")
            } else if stream_active || crate::spec::spec_stream() {
                Some("round-stream")
            } else if !self.mtp_extra.is_empty() {
                Some("multi-head-mtp")
            } else if crate::cache::swa_ring_on() || cache.has_swa_ring() {
                Some("swa-ring")
            } else if crate::pp::pp_host_bounce_active() {
                Some("host-bounce")
            } else if fork_mode == OptiForkGateMode::Controller
                && cache.recur.iter().any(Option::is_some)
            {
                Some("controller-requires-zero-recurrent-state")
            } else if fence.as_ref().is_none_or(|f| f.len() != 3) {
                Some("requires-pp2")
            } else {
                None
            };
            if let Some(reason) = refusal {
                OPTI_FORK_REFUSALS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                eprintln!("[opti-fork] refused reason={reason}");
            } else {
                let fence = fence.expect("validated PP-2 fence");
                let rt = crate::pp::PpNRt::get(e)?;
                let primary_stage0 = rt.engine(0, e).ctx().ordinal() == e.ctx().ordinal();
                let primary_stage1 = rt.engine(1, e).ctx().ordinal() == e.ctx().ordinal();
                let primary_supported =
                    primary_stage0 || (fork_mode == OptiForkGateMode::Controller && primary_stage1);
                if !rt.cross_device() || !primary_supported {
                    OPTI_FORK_REFUSALS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    eprintln!("[opti-fork] refused reason=requires-supported-primary-cross-device");
                } else {
                    // The optimistic controller can keep two boundary tickets in flight. Give
                    // every nested verify an explicit borrow of one whole-walk generation; no
                    // `pp_pipe` boolean is allowed to bypass ownership on its own.
                    let walk = rt.acquire_walk("opti_fork_coordinator")?;
                    let permit = rt.walk_permit(&walk, "opti_fork_coordinator")?;
                    let borrow = rt.borrow_walk(&permit, "opti_fork_coordinator")?;
                    // Both recurrent snapshots and both seed generations are allocated before
                    // the first fork, each through its owning PP stage. Allocation failure
                    // therefore happens before any optimistic state mutation can occur.
                    let current_snapshot = opti_snapshot_stage_owned(e, cache, rt, &fence)?;
                    let alternate_snapshot = opti_snapshot_stage_owned(e, cache, rt, &fence)?;
                    let fork = OptiForkState::new(
                        e,
                        cache,
                        fork_mode,
                        alternate_snapshot,
                        &h_seed_buf,
                        &fill_prev,
                        rt,
                        fence[1],
                        self.layers.len(),
                    )?;
                    eprintln!(
                        "[opti-fork] armed mode={fork_mode:?} snapshots=2 seeds=2 split={} \
                         payload_dev0={} payload_dev1={} q_threshold={:.3}",
                        fence[1],
                        fork.logical_payload_bytes[0],
                        fork.logical_payload_bytes[1],
                        fork.controller.map_or(0.0, |policy| policy.threshold),
                    );
                    fork_snapshot = Some(current_snapshot);
                    opti_fork = Some(fork);
                    _opti_walk = Some(walk);
                    _opti_walk_borrow = Some(borrow);
                }
            }
        }
        // Persistent snapshot buffers are allocated once and refreshed in place. The fork arm
        // uses stage-owned snapshots; refused/disabled arms retain the existing generic helper.
        let mut snap = match fork_snapshot {
            Some(snapshot) => snapshot,
            None => cache.snapshot(e)?,
        };
        let mut carried_opti: Option<OptiControllerTicket> = None;
        // ROUND-STREAM stage (b) 3a: device table of per-layer kvl.len_d pointers (stable — the
        // cache never reallocates len_d; see cache.rs "stable pointer" note). 0 = no KV layer.
        let kv_len_ptrs: Option<CudaSlice<u64>> = if spec_devacc() && !spec_replay {
            Some(crate::round_stream::kv_len_ptr_table(e, cache, None)?)
        } else {
            None
        };
        // BONUS FOLD (2026-07-04): after a FULL accept the bonus token is NOT committed with a
        // separate T=1 trunk pass (a full weight read per round). It stays PENDING and rides as
        // column 0 of the NEXT round's verify batch. Under predecessor pairing the next chain
        // seeds from the bonus's predecessor's TRUE verify hidden (free — no extra
        // pass of any kind). Verify still
        // checks every emitted token against the target -> exactness holds by construction; only
        // DRAFT QUALITY can shift, which the acceptance numbers arbitrate.
        // bonus emitted but not yet committed to cache. A carried pending (see SpecSession::
        // pending_tok) enters round 0 directly — the burst boundary becomes a plain round edge.
        let mut pending: Option<u32> = carried_pending;
        // MEMRA_SPEC_PHASE=1: per-round wall decomposition (draft / verify / accept+commit) —
        // no tracing, no extra syncs (each phase is naturally sync-bounded: draft readbacks,
        // the verify accept readback). Printed once at loop end via spec-stats.
        let anatomy_on = std::env::var("MEMRA_SPEC_PP_ANATOMY").as_deref() == Ok("1");
        let phase_on = anatomy_on || std::env::var("MEMRA_SPEC_PHASE").as_deref() == Ok("1");
        // MEMRA_SPEC_PHASE_SYNC=1 — reads the phase split correctly, and proves it. `ph_mark` is a
        // bare Instant, so `verify-issue` is the host QUEUEING the walk (the GPU is already running
        // under it) and `verify-wait` is only the residual drain at the accept readback: one
        // overlapped interval cut at the first blocking call, NOT "GPU time" beside "host time".
        // Syncing right after the walk is issued moves the whole GPU wall into `verify-issue`. If
        // the walk's GPU total is really issue+wait, then with this on verify-issue jumps to that
        // sum, verify-wait collapses to the readback alone, and the ROUND WALL DOES NOT MOVE —
        // which is what says the queueing time was hidden and is not a target. Diagnostic only.
        let phase_sync = std::env::var("MEMRA_SPEC_PHASE_SYNC").as_deref() == Ok("1");
        // DRAFT-MASK receipt (lane/draft-mask): speculative-clone wall + rounds, printed with
        // spec-stats. The clone is the one cost the design adds per round — measured, not assumed.
        let (mut dm_clone_ns, mut dm_rounds) = (0u128, 0usize);
        // grammar-truncation counters: how many rounds the verify-side cut fired and how many
        // already-verified tokens it threw away. THIS is the quantity draft masking targets.
        let (mut dm_cuts, mut dm_cut_tokens) = (0usize, 0usize);
        let (mut ph_draft, mut ph_verify, mut ph_rest) = (0f64, 0f64, 0f64);
        let mut ph_wait = 0f64;
        let mut ph_commit = 0f64;
        let mut ph_t = std::time::Instant::now();
        let mut ph_mark = |acc: &mut f64, on: bool| {
            if on {
                let now = std::time::Instant::now();
                *acc += (now - ph_t).as_secs_f64();
                ph_t = now;
            }
        };
        // MTP-ROUTE VERIFY GRAPHS (`MEMRA_SPEC_VERIFY_GRAPH`, see the flag doc): the
        // model-owned capture pool, locked for the whole burst exactly as the dspark serve
        // arm holds it — the slab stash is live verify -> commit inside a round, and the
        // worker drives rounds from one scheduler thread. PERSISTENT across generations on
        // the model (rebuilding per call re-captures the pool per prompt, which is the
        // measured way to lose more than the launches cost); the captured bodies are
        // cache-independent, every state read going through per-round refreshed pointer
        // tables. None = the eager walk, byte-identical.
        //
        // Never armed together with ROUND-STREAM: the tparallel verify refuses that pair
        // loudly, and `stream_active` owns the burst arm above, so the door stays shut
        // whenever the stream is live rather than relying on that refusal.
        // The lock is taken ONLY when the door is armed: with the flag off this whole block
        // is inert, so the default path cannot serialize two spec generations behind a mutex
        // it never reads.
        let vg_armed =
            crate::spec::spec_verify_graph_env().unwrap_or_else(|| self.vgraph_family_default());
        let mut vg_guard = if vg_armed && !stream_active {
            let mut g = self.dspark_vgraphs.lock().unwrap();
            if g.is_none() {
                // Size by the WIDEST verify this run can present, which is k+1 and NOT
                // k_cap+1: the sampled arm's own window is `t_v_s = k + 1`, so a pool built
                // from a smaller adaptive cap gets sliced past its stash rows (a `slice_mut`
                // panic in the sampled ON arm, measured before this line said k+1).
                let vt_cap = (k.max(k_cap) + 1).max(2);
                *g = DsparkVerifyGraphs::new(e, cache, vt_cap, n_embd)?;
                if g.is_some() {
                    // Engagement receipt (the dead-arm lesson): prove the door is LIVE rather
                    // than trusting that a flag set means a pool built.
                    eprintln!("[spec-vg] MTP verify-graph pool ENGAGED (vt_cap={vt_cap})");
                } else {
                    eprintln!(
                        "[spec-vg] MTP verify-graph pool declined (no linear layers, \
                         non-uniform state, or vt_cap < 2) — eager walk"
                    );
                }
            }
            Some(g)
        } else {
            None
        };
        // Capacity fail-safe: a round wider than the pool was built for must take the eager
        // walk, not slice the stash past its rows. The sizing above already covers every
        // round this run can present; this keeps a future caller (or a k that grows behind
        // the pool's back) on the byte-identical fallback instead of a panic.
        let vg_t_cap = vg_guard
            .as_ref()
            .and_then(|g| g.as_ref())
            .map(|g| g.t_capacity())
            .unwrap_or(0);
        if let Some(p) = pipe {
            p.setup_end();
        }
        drop(pipe_setup_walk);
        let mut graph_guard_noted = false;
        while keep_going && out.len() < max_new {
            // GRAPH-LAUNCH HEADROOM GUARD (see GRAPH_LAUNCH_MIN_FREE): below the floor,
            // every captured-graph arm in this round yields to its byte-identical eager
            // twin instead of feeding cuGraphLaunch a card it segfaults on.
            let graph_round_ok = graph_launch_headroom_ok(e);
            if !graph_round_ok && !graph_guard_noted {
                graph_guard_noted = true;
                eprintln!(
                    "[spec] graph replay suspended: driver free below the {}MB launch floor \
                     (eager arms serve; cuGraphLaunch segfaults into an exhausted card)",
                    GRAPH_LAUNCH_MIN_FREE / (1 << 20)
                );
            }
            // MEMRA_SPEC_ROUND_PROF=1: wall of the WHOLE round against the pieces we already
            // instrument. Needed because the parts do not add up: the draft step measures 1.27 ms
            // ([spec-anatomy] glue 92 / attn 280 / ffn 222 / head 670 us) and the t=2 verify walk
            // 25.6 ms ([tcol-prof] attn 10.1 + ffn 15.3), yet a K=1 round takes 177 ms on the
            // step37 TP2 stack. This prints where the other ~150 ms lives.
            let round_prof = ROUND_PROF
                .get_or_init(|| std::env::var("MEMRA_SPEC_ROUND_PROF").as_deref() == Ok("1"));
            let round_t0 = round_prof.then(std::time::Instant::now);
            // ROUND-STREAM BURST: from round 1 (pending guaranteed by every non-replay arm),
            // issue M rounds with zero readbacks, then drain the ring + reconcile mirrors.
            if let (true, Some(sg), Some(ptrs)) = (
                stream_active && round >= 1 && pending.is_some() && graph_round_ok,
                &stream_graph,
                &stream_ptrs,
            ) {
                if debug_spec {
                    static ONCE: std::sync::Once = std::sync::Once::new();
                    ONCE.call_once(|| {
                        eprintln!("[memra] ROUND-STREAM burst engaged (M={m_rounds} k={k})")
                    });
                }
                e.set_i32_one(&mut pos_ctr, cache.pos as i32)?;
                e.set_u32_one(&mut pend_d, pending.unwrap())?;
                e.set_u32_one(&mut ring_d, 0)?; // ring count = 0 (writes element 0)
                for _mi in 0..m_rounds {
                    e.i32_copy_add(&pos_ctr, &mut pos_start_d, 0)?;
                    cache.snapshot_into(e, &mut snap)?; // device D2Ds, stream-ordered
                    e.i32_copy_add(&pos_ctr, &mut scratch.kv.len_d, 0)?; // draft-KV rollback
                    e.i32_copy_add(&pos_ctr, &mut dctx.g_pos, 1)?; // rope pos = pos + base
                    e.u32_copy(&pend_d, &mut dctx.g_tok)?;
                    e.copy_into(&mut dctx.g_seed, 0, &h_seed_buf, n_embd)?;
                    sg.launch()?;
                    e.spec_assemble_verify(
                        &g_tokp2k,
                        &pend_d,
                        d2t_dev.as_ref(),
                        &mut vtok_d,
                        &mut brk_d,
                        p_min,
                        k,
                        pmin0,
                    )?;
                    let mut ck = VerifyCkpt::new(self.layers.len());
                    let dummy = vec![0u32; t_v_s];
                    let (tl_d, vx) = self.decode_step_t_core_stream(
                        e,
                        &dummy,
                        0,
                        &mut *cache,
                        embd_dev,
                        Some(&mut ck),
                        Some((&vtok_d, &pos_ctr)),
                        None,
                        None,
                        None,
                    )?;
                    for j in 0..t_v_s {
                        e.argmax_token_device_col(&tl_d, j, n_vocab, &mut preds_d, j)?;
                    }
                    e.spec_accept_greedy_dc(
                        &preds_d,
                        &vtok_d,
                        &last_pred_d,
                        &brk_d,
                        &mut stream_acc,
                    )?;
                    e.spec_seed_gather(&vx, &fill_prev, &stream_acc, &mut h_seed_buf, 1, n_embd)?;
                    e.copy_into(&mut fill_prev, 0, &h_seed_buf, n_embd)?;
                    self.commit_verified_prefix_stream(
                        e,
                        &mut *cache,
                        &snap,
                        &ck,
                        &stream_acc,
                        1,
                        t_v_s,
                    )?;
                    e.spec_rollback_stream(
                        ptrs,
                        &pos_start_d,
                        &stream_acc,
                        1,
                        self.layers.len() + 1,
                    )?;
                    e.spec_ring_commit(&vtok_d, &stream_acc, &brk_d, &mut ring_d, &mut pend_d)?;
                }
                e.stream().synchronize()?;
                let ring_h = e.dtoh_u32(&ring_d)?;
                let cnt = ring_h[0] as usize;
                for i in 0..cnt {
                    if out.len() < max_new {
                        out.push(ring_h[1 + i]);
                    }
                }
                let pos_h = e.dtoh_i32(&pos_ctr)?[0] as usize;
                for il in 0..self.layers.len() {
                    if let Some(kvl) = cache.kv[il].as_mut() {
                        kvl.len = pos_h;
                    }
                }
                cache.pos = pos_h;
                scratch.kv.len = pos_h;
                pending = Some(ring_h[cnt]); // last drained token = the live bonus
                last_token = ring_h[cnt];
                total_drafted += k * m_rounds; // upper bound (p-min breaks uncounted)
                total_accepted += cnt.saturating_sub(m_rounds);
                if let Some(t) = sess_telem {
                    // totals only — the burst's per-round accept counts stayed on device
                    // (that is the point of the round-stream arm). pos_* untouched.
                    t.record_totals(m_rounds, k * m_rounds, cnt.saturating_sub(m_rounds));
                }
                round += m_rounds;
                // sse-cadence: the drained ring is committed — flush it at burst-drain cadence.
                keep_going = flush_commit(&mut on_commit, &out, &mut flushed);
                continue;
            }
            let pipe_draft = match pipe {
                Some(p) => Some(p.draft_begin(round)?),
                None => None,
            };
            let pos = cache.pos; // #tokens committed (EXCLUDES a pending bonus)
            let mut current_opti = carried_opti.take();
            let mut fork_generation = if current_opti.is_none() && pending.is_some() {
                match opti_fork.as_mut() {
                    Some(fork) if fork.mode.is_forced() => Some(fork.reserve(&mut snap)?),
                    None => None,
                    Some(_) => None,
                }
            } else {
                None
            };
            if current_opti.is_none() {
                if let Some(fork) = opti_fork.as_ref() {
                    opti_snapshot_stage_owned_into(e, cache, fork.rt, &fork.fence, &mut snap)?;
                } else {
                    cache.snapshot_into(e, &mut snap)?;
                }
            } else if snap.pos != pos {
                return Err(format!(
                    "optipipe carried snapshot pos {} != current pos {pos}",
                    snap.pos
                )
                .into());
            } // §C: snapshot BEFORE draft+verify (already retained for a carried successor)
            ph_mark(&mut ph_rest, phase_on);

            // --- 1. DRAFT k tokens with the NextN head (autoregressive, T=1 each) ---
            // p-min semantics (both paths): stop the chain early when the head's confidence in
            // its own pick drops below p_min — the just-drafted token is DISCARDED, but its
            // scratch append stands (identical to the eager chain's ordering). j==0 always drafts.
            let base0 = if pending.is_some() { 1usize } else { 0usize };
            // fixed draft length by default; MEMRA_SPEC_ADAPT=1 drafts at last round's
            // accepted run + 1 (the gemma law — see the setup block above the loop).
            let k_this = if adapt { kc } else { k };
            let mut draft: Vec<u32> = Vec::with_capacity(k);
            let mut draft_idx: Vec<u32> = Vec::with_capacity(k); // trimmed-vocab ids (== draft when untrimmed)
            let mut controller_draft_prob: Option<f32> = None;
            let mut controller_eager_state: Option<(u32, CudaSlice<f32>)> = None;
            if let Some(ticket) = current_opti.as_mut() {
                let carried_pending = pending.ok_or("optipipe carried successor lost pending")?;
                if ticket.verify_tokens[0] != carried_pending {
                    return Err(format!(
                        "optipipe carried pending mismatch: ticket={} live={carried_pending}",
                        ticket.verify_tokens[0],
                    )
                    .into());
                }
                draft.push(ticket.verify_tokens[1]);
                controller_draft_prob = Some(ticket.draft_prob);
                controller_eager_state = ticket
                    .take_eager_seed()
                    .map(|seed| (ticket.verify_tokens[1], seed));
            } else {
                // Round-start draft-KV sync (BOTH paths). Persistent: truncate/align to the committed
                // history — slots 0..P hold entries for the tokens before last_token@P (P = pos +
                // base0 - 1); this single set_len IS the draft-side rollback (drops last round's
                // rejected drafts and p-min extras via the len mechanism).
                scratch.set_len(e, pos + base0 - 1)?;
                // dcw door: a captured chain appends k_this device-counter rows (plus the
                // pseudo-seed replay) with no host intervention; any ring rebase those appends
                // could need happens HERE, host-side, before the replays. The eager arm keeps
                // its own per-step prepare, so this is graph-path-only work.
                if step35_draft_dcw_on()
                    && (dctx.graph.is_some()
                        || dctx.graph_s.is_some()
                        || dctx.chain.is_some()
                        || dctx.chain_s.is_some())
                {
                    scratch.ensure_dcw_headroom(e, k_this + 2)?;
                }
                if pen_on {
                    // PEN_WINDOW_MAX also bounds the per-round upload and the O(n_hist^2)
                    // device dedup: the serve window is already PEN_WINDOW_MAX, and this
                    // defensive min also bounds non-server callers.
                    let win = sp.penalty_last_n.min(PEN_WINDOW_MAX);
                    let w0 = pen_hist.len().saturating_sub(win);
                    pen_hist_d = Some(e.htod_u32_v(&pen_hist[w0..])?);
                }
                if sampled {
                    draft_logits.clear();
                    draft_stats.clear();
                }
                // DRAFT-SIDE GRAMMAR MASK: clone the committed grammar state ONCE per round; each
                // position's mask is computed on that clone and advanced by the PROPOSED token. The
                // real state moves only on emission (verify's job), so the emitted stream is
                // unchanged — the mask only removes tokens the verify would have truncated anyway.
                let mut dmask_live = dmask_on;
                if dmask_live {
                    let t_c = std::time::Instant::now();
                    constraint
                        .as_deref_mut()
                        .unwrap()
                        .draft_begin()
                        .map_err(|e2| format!("constraint: {e2}"))?;
                    dm_clone_ns += t_c.elapsed().as_nanos();
                    dm_rounds += 1;
                }
                if let (false, Some(cg)) = (sampled || pen_on || !graph_round_ok, &dctx.chain) {
                    // GREEDY CHAIN GRAPH (lane/step37-draft-graph-serving-20260830): the
                    // eager multi-head chain's EXACT launch order — step j rewinds head
                    // (j % heads)'s plane to the committed length and replays rows 0..=j —
                    // with each row's whole head-forward as ONE graph launch. The chain
                    // POLICY (head choice, prefix length, stored-seed feed) is host-side,
                    // identical to `mtp_chain_forward_dev`, so graph-vs-eager drafts are
                    // bit-identical by construction (same launcher, same bucket — the dcw
                    // parity contract). Interior rows launch the head-less graph: their
                    // logits are dead in the eager chain too, so the consumed bytes match.
                    let heads_n = self.mtp_head_count();
                    let committed = pos + base0 - 1;
                    let mut chain_tokens: Vec<u32> = vec![last_token];
                    let mut chain_seed_bufs: Vec<CudaSlice<f32>> = vec![e.clone_dtod(&h_seed_buf)?];
                    for j in 0..k_this {
                        let index = mtp_chain_head_index(j, heads_n);
                        if debug_spec {
                            eprintln!(
                                "[mtp-chain-step] round={round} j={j} head={index} \
                                 replay_rows={} arm=graph",
                                chain_tokens.len(),
                            );
                        }
                        scratch.set_plane_len(e, index, committed)?;
                        e.set_i32_one(&mut dctx.g_pos, (committed + 1) as i32)?;
                        for row in 0..=j {
                            e.set_u32_one(&mut dctx.g_tok, chain_tokens[row])?;
                            e.copy_into(&mut dctx.g_seed, 0, &chain_seed_bufs[row], n_embd)?;
                            if row < j {
                                cg.interior[index].launch()?;
                            } else {
                                // per-position mask upload before the LAST row only — the
                                // eager chain applies the mask on is_last exactly the same.
                                if dmask_live
                                    && !upload_draft_mask(
                                        e,
                                        constraint.as_deref_mut().unwrap(),
                                        &mut dctx.g_dmask,
                                        mtp.d2t.as_ref(),
                                        d_vocab,
                                        dmask_words,
                                    )?
                                {
                                    e.htod_u32_into(
                                        &mut dctx.g_dmask,
                                        &vec![u32::MAX; dmask_words],
                                    )?;
                                    dmask_live = false;
                                }
                                cg.last[index].launch()?;
                            }
                            // host mirror (len_d advanced in-graph by the dcw append)
                            scratch.plane_mut(index).0.len += 1;
                        }
                        let idx = e.dtoh_u32_one(&dctx.g_tok)?;
                        // #87 SENTINEL TRAP (see the single-head graph arm below).
                        if (idx as usize) >= d_vocab {
                            let seed_h = e.dtoh(&dctx.g_seed)?;
                            let seed_nan = seed_h.iter().filter(|v| v.is_nan()).count();
                            return Err(format!(
                                "draft(chain-graph) argmax sentinel 0x{idx:08x} >= d_vocab \
                             {d_vocab} at round {round} j={j} head={index} pos={pos}: \
                             head-out NaN {seed_nan}/{n_embd} — refusing to dereference \
                             the embed row (#87 trap)"
                            )
                            .into());
                        }
                        // multi-head MTP forbids a trimmed head (validated at entry), so the
                        // draft index IS the target id; keep the map for uniformity.
                        let d = match &mtp.d2t {
                            Some(map) => map[idx as usize],
                            None => idx,
                        };
                        let draft_p = if p_min > 0.0 {
                            Some(e.dtoh(&dctx.g_p)?[0])
                        } else {
                            None
                        };
                        if j == 0 {
                            controller_draft_prob = draft_p;
                        }
                        if let Some(p) = draft_p.filter(|_| p_min > 0.0)
                            && p < p_min
                            && (j > 0 || (pmin0 && base0 == 1))
                        {
                            break;
                        }
                        draft.push(d);
                        chain_tokens.push(d);
                        // step j's h_nextn: the last-row graph self-fed it into g_seed —
                        // snapshot it as the chain history seed for row j+1 (stream-ordered
                        // after the launch, exactly the eager chain's chain_seeds push).
                        chain_seed_bufs.push(e.clone_dtod(&dctx.g_seed)?);
                        // speculative grammar advance (see the single-head graph arm).
                        if dmask_live
                            && !constraint
                                .as_deref_mut()
                                .unwrap()
                                .draft_advance(d)
                                .map_err(|e2| format!("constraint: {e2}"))?
                        {
                            e.htod_u32_into(&mut dctx.g_dmask, &vec![u32::MAX; dmask_words])?;
                            break;
                        }
                    }
                } else if let (true, Some(cg)) = (
                    sampled && s_capturable && dctx.s_key == Some(s_key) && graph_round_ok,
                    &dctx.chain_s,
                ) {
                    if skey_probe() {
                        eprintln!(
                            "[skey] chain=graph_chain_s round={round} capturable={} top_k={} \
                             top_p={} min_p={} s_key_parked={:?}",
                            s_capturable as u8, sp.top_k, sp.top_p, sp.min_p, dctx.s_key,
                        );
                    }
                    // SAMPLED CHAIN GRAPH: the greedy chain arm's launch order with the
                    // sampled last-row graphs — in-graph counter bump + (filtered) gumbel
                    // draw + argmax; q retained per step into q_slots exactly like the
                    // single-head sampled graph arm. Counter continuity: g_ctr host-seeded
                    // to sctr-1 once per ROUND; each step's last-row graph bumps it BEFORE
                    // the perturb, so step j consumes counter sctr+j — the eager Philox
                    // stream (interior rows never draw, never bump).
                    let heads_n = self.mtp_head_count();
                    let committed = pos + base0 - 1;
                    let filtered_stats_in_graph = s_key.filtered();
                    let mut chain_tokens: Vec<u32> = vec![last_token];
                    let mut chain_seed_bufs: Vec<CudaSlice<f32>> = vec![e.clone_dtod(&h_seed_buf)?];
                    e.set_u32_one(&mut dctx.g_ctr, sctr.wrapping_sub(1))?;
                    for j in 0..k_this {
                        let index = mtp_chain_head_index(j, heads_n);
                        if debug_spec {
                            eprintln!(
                                "[mtp-chain-step] round={round} j={j} head={index} \
                                 replay_rows={} arm=graph_s",
                                chain_tokens.len(),
                            );
                        }
                        scratch.set_plane_len(e, index, committed)?;
                        e.set_i32_one(&mut dctx.g_pos, (committed + 1) as i32)?;
                        for row in 0..=j {
                            e.set_u32_one(&mut dctx.g_tok, chain_tokens[row])?;
                            e.copy_into(&mut dctx.g_seed, 0, &chain_seed_bufs[row], n_embd)?;
                            if row < j {
                                cg.interior[index].launch()?;
                            } else {
                                cg.last[index].launch()?;
                            }
                            scratch.plane_mut(index).0.len += 1;
                        }
                        sctr += 1; // mirrors the in-graph g_ctr bump (eager parity:
                        // counts the p-min-discarded token too)
                        // q retention: ONE async D2D of the persistent head-logits buffer
                        // into this round's slot j (stream-ordered after the replay).
                        e.copy_into(&mut dctx.q_slots[j], 0, &dctx.g_q, d_vocab)?;
                        // FILTERED capture: read the in-graph filter_stats scalars back per
                        // replay instead of a second full-vocab filter_stats per slot post-
                        // chain — bit-exact (the values the in-graph perturb consumed) and
                        // measured worth ~5% of vendor-default serving tok/s at K=3. Before
                        // the p-min break so the discarded slot's stats land too.
                        if filtered_stats_in_graph {
                            draft_stats.push((
                                e.dtoh(&dctx.g_mx)?[0],
                                e.dtoh(&dctx.g_th)?[0],
                                e.dtoh(&dctx.g_z)?[0],
                            ));
                        }
                        let idx = e.dtoh_u32_one(&dctx.g_tok)?;
                        // #87 SENTINEL TRAP (see the single-head graph arms).
                        if (idx as usize) >= d_vocab {
                            let seed_h = e.dtoh(&dctx.g_seed)?;
                            let seed_nan = seed_h.iter().filter(|v| v.is_nan()).count();
                            return Err(format!(
                                "draft(chain-graph-sampled) argmax sentinel 0x{idx:08x} >= \
                             d_vocab {d_vocab} at round {round} j={j} head={index} pos={pos}: \
                             head-out NaN {seed_nan}/{n_embd} — refusing to dereference the \
                             embed row (#87 trap)"
                            )
                            .into());
                        }
                        let d = match &mtp.d2t {
                            Some(map) => map[idx as usize],
                            None => idx,
                        };
                        draft_idx.push(idx);
                        if p_min > 0.0 {
                            let p = e.dtoh(&dctx.g_p)?[0];
                            if p < p_min && (j > 0 || (pmin0 && base0 == 1)) {
                                break;
                            }
                        }
                        draft.push(d);
                        chain_tokens.push(d);
                        chain_seed_bufs.push(e.clone_dtod(&dctx.g_seed)?);
                    }
                    // PURE-TEMP accept path: stats per used slot recomputed from the RETAINED
                    // q with the SAME filter_stats program the eager arm runs (deployment-
                    // keyed coop/plain choice, same input bits). The FILTERED graph read its
                    // stats back per replay above.
                    if !filtered_stats_in_graph {
                        for j in 0..draft.len().max(draft_idx.len()) {
                            let rows0 = e.htod_i32(&[0])?;
                            let (mut th_d, mut z_d, mut mx_d) =
                                (e.zeros(1)?, e.zeros(1)?, e.zeros(1)?);
                            e.filter_stats(
                                &dctx.q_slots[j],
                                d_vocab,
                                &rows0,
                                &mut th_d,
                                &mut z_d,
                                &mut mx_d,
                                d_vocab,
                                1,
                                sp_temp,
                                sp.top_k,
                                sp.top_p,
                                sp.min_p,
                            )?;
                            draft_stats.push((
                                e.dtoh(&mx_d)?[0],
                                e.dtoh(&th_d)?[0],
                                e.dtoh(&z_d)?[0],
                            ));
                        }
                    }
                } else if let (false, Some(gr)) =
                    (sampled || pen_on || !graph_round_ok, &dctx.graph)
                {
                    // GRAPH DRAFT: one dispatch per drafted token. The chain feeds itself on-device
                    // (in-graph argmax -> tok_d -> next replay's embed; h_nextn -> h_seed_d; pos_d
                    // inc'd in-graph); the host only reads 4B token (+4B p) and decides the break.
                    e.set_i32_one(&mut dctx.g_pos, (pos + base0) as i32)?;
                    e.set_u32_one(&mut dctx.g_tok, last_token)?;
                    e.copy_into(&mut dctx.g_seed, 0, &h_seed_buf, n_embd)?;
                    for j in 0..k_this {
                        // per-position mask upload (contents only — the graph's baked pointer is
                        // dctx.g_dmask). All-ones once masking goes dead mid-chain, so the captured
                        // mask node degrades to a no-op ban instead of needing a second graph.
                        if dmask_live
                            && !upload_draft_mask(
                                e,
                                constraint.as_deref_mut().unwrap(),
                                &mut dctx.g_dmask,
                                mtp.d2t.as_ref(),
                                d_vocab,
                                dmask_words,
                            )?
                        {
                            // no draft-vocab row is grammar-legal here (a trimmed FR-Spec head can
                            // genuinely miss the legal set): neutralize the captured mask node and
                            // finish the chain UNMASKED — exactly pre-lane behaviour, never worse.
                            e.htod_u32_into(&mut dctx.g_dmask, &vec![u32::MAX; dmask_words])?;
                            dmask_live = false;
                        }
                        gr.launch()?;
                        scratch.kv.len += 1; // host mirror (len_d advanced in-graph)
                        let idx = e.dtoh_u32_one(&dctx.g_tok)?;
                        // #87 SENTINEL TRAP: an all-NaN head-logits row leaves the device argmax's
                        // init sentinel (0x7FFFFFFF) in g_tok — feeding it onward dereferences
                        // embed_row(sentinel) = table + ~4.6TB (never mapped) inside the NEXT graph
                        // replay's embed node, and the MMU fault kills the CUDA context for the
                        // whole process (research/pp2spec-crash-20260807: 3 coredumps, byte-exact
                        // VA arithmetic). Refuse loudly instead; the diagnostics name the first-NaN
                        // buffer (g_seed = the verify-side handoff vs head-side compute).
                        if (idx as usize) >= d_vocab {
                            // g_seed is SELF-FED (the replay writes h_nextn back into it), so it
                            // reads as the head's OUTPUT at j; h_seed_buf is the round's INPUT
                            // seed, untouched since the round-start copy — the pair discriminates
                            // "seed arrived poisoned" from "head forward produced NaN".
                            let seed_h = e.dtoh(&dctx.g_seed)?;
                            let seed_nan = seed_h.iter().filter(|v| v.is_nan()).count();
                            let in_h = e.dtoh(&h_seed_buf)?;
                            let in_nan = in_h.iter().filter(|v| v.is_nan()).count();
                            return Err(format!(
                                "draft(graph) argmax sentinel 0x{idx:08x} >= d_vocab {d_vocab} at \
                             round {round} j={j} pos={pos}: head-out NaN {seed_nan}/{n_embd}, \
                             round-input-seed NaN {in_nan}/{n_embd} — refusing to dereference \
                             the embed row (#87 trap)"
                            )
                            .into());
                        }
                        // trimmed draft vocab -> target token id (identity when no d2t map)
                        let d = match &mtp.d2t {
                            Some(map) => map[idx as usize],
                            None => idx,
                        };
                        let draft_p = if p_min > 0.0
                            || opti_fork
                                .as_ref()
                                .is_some_and(|fork| fork.controller.is_some())
                        {
                            Some(e.dtoh(&dctx.g_p)?[0])
                        } else {
                            None
                        };
                        if j == 0 {
                            controller_draft_prob = draft_p;
                        }
                        if let Some(p) = draft_p.filter(|_| p_min > 0.0)
                            && p < p_min
                            && (j > 0 || (pmin0 && base0 == 1))
                        {
                            break;
                        }
                        draft.push(d);
                        // with a trimmed head the NEXT embed must read the TARGET id, not the draft
                        // index the argmax wrote — patch the persistent token buffer (4B htod).
                        if d != idx {
                            e.set_u32_one(&mut dctx.g_tok, d)?;
                        }
                        // advance the SPECULATIVE state with the proposal; a dead chain drops to
                        // unmasked drafting for the remaining positions (verify still arbitrates).
                        // speculative advance; a chain the grammar can no longer follow (EOS
                        // proposed) ends here. The captured mask node always runs, so a dead chain
                        // leaves the buffer NEUTRAL (all-ones = ban nothing) before it exits.
                        if dmask_live
                            && !constraint
                                .as_deref_mut()
                                .unwrap()
                                .draft_advance(d)
                                .map_err(|e2| format!("constraint: {e2}"))?
                        {
                            e.htod_u32_into(&mut dctx.g_dmask, &vec![u32::MAX; dmask_words])?;
                            break;
                        }
                    }
                // REGIME RE-TEST (lane/graph-s-key-exactness-20260819, widened by
                // lane/step37-draft-graph-serving-20260830): the sampled graph is legal ONLY
                // in the regime it was captured in. The condition used to read
                // `(sampled, &dctx.graph_s)` and trusted `s_key` to have dropped anything
                // else — which it could not, because the key omitted the filters. Both
                // halves are enforced: the key drops a stale graph, and this site refuses to
                // launch one whose key differs or whose regime is uncapturable (penalties).
                } else if let (true, Some(gr)) = (
                    sampled && s_capturable && dctx.s_key == Some(s_key) && graph_round_ok,
                    &dctx.graph_s,
                ) {
                    if skey_probe() {
                        eprintln!(
                            "[skey] chain=graph_s round={round} pure_temp={} capturable={} \
                             top_k={} top_p={} min_p={} s_key_parked={:?}",
                            pure_temp as u8,
                            s_capturable as u8,
                            sp.top_k,
                            sp.top_p,
                            sp.min_p,
                            dctx.s_key,
                        );
                    }
                    // SAMPLED GRAPH DRAFT: one replay per drafted token — head forward + gumbel +
                    // argmax in ONE dispatch; the host reads 4B token (+4B p), D2Ds q into slot j,
                    // and decides the break. Event-counter continuity: g_ctr is host-seeded to
                    // sctr-1 ONCE per round (outside the graph); the in-graph bump runs BEFORE the
                    // perturb, so replay j consumes counter sctr+j — exactly the eager arm's Philox
                    // stream. Host sctr advances in lockstep (computed, no readback needed).
                    e.set_i32_one(&mut dctx.g_pos, (pos + base0) as i32)?;
                    e.set_u32_one(&mut dctx.g_tok, last_token)?;
                    e.copy_into(&mut dctx.g_seed, 0, &h_seed_buf, n_embd)?;
                    e.set_u32_one(&mut dctx.g_ctr, sctr.wrapping_sub(1))?;
                    let filtered_stats_in_graph = s_key.filtered();
                    for j in 0..k_this {
                        gr.launch()?;
                        scratch.kv.len += 1; // host mirror (len_d advanced in-graph)
                        sctr += 1; // mirrors the in-graph g_ctr bump (eager parity:
                        // counts the p-min-discarded token too)
                        // q retention: ONE async D2D of the persistent head-logits buffer into this
                        // round's slot j (stream-ordered after the replay, before the next one).
                        e.copy_into(&mut dctx.q_slots[j], 0, &dctx.g_q, d_vocab)?;
                        // FILTERED capture: the replay's own filter_stats node already computed
                        // (th, z, mx) — read the three scalars back instead of paying a SECOND
                        // full-vocab filter_stats per slot post-chain (measured ~5% of vendor-
                        // default serving tok/s at K=3). Bit-exact by construction: these are
                        // the very values the in-graph perturb consumed. Read BEFORE the p-min
                        // break so the discarded slot's stats land too (accept-path indexing).
                        if filtered_stats_in_graph {
                            draft_stats.push((
                                e.dtoh(&dctx.g_mx)?[0],
                                e.dtoh(&dctx.g_th)?[0],
                                e.dtoh(&dctx.g_z)?[0],
                            ));
                        }
                        let idx = e.dtoh_u32_one(&dctx.g_tok)?;
                        // #87 SENTINEL TRAP (see the greedy graph arm above).
                        if (idx as usize) >= d_vocab {
                            let seed_h = e.dtoh(&dctx.g_seed)?;
                            let seed_nan = seed_h.iter().filter(|v| v.is_nan()).count();
                            return Err(format!(
                                "draft(graph-sampled) argmax sentinel 0x{idx:08x} >= d_vocab \
                             {d_vocab} at round {round} j={j} pos={pos}: round-seed NaN \
                             {seed_nan}/{n_embd} — refusing to dereference the embed row \
                             (#87 trap)"
                            )
                            .into());
                        }
                        let d = match &mtp.d2t {
                            Some(map) => map[idx as usize],
                            None => idx,
                        };
                        draft_idx.push(idx);
                        if p_min > 0.0 {
                            let p = e.dtoh(&dctx.g_p)?[0];
                            if p < p_min && (j > 0 || (pmin0 && base0 == 1)) {
                                break;
                            }
                        }
                        draft.push(d);
                        // trimmed head: the NEXT embed must read the TARGET id (see the greedy arm).
                        if d != idx {
                            e.set_u32_one(&mut dctx.g_tok, d)?;
                        }
                    }
                    // PURE-TEMP accept path: fill draft_stats per used slot post-chain (the
                    // stats degenerate to th=0 / full-Z; one filter_stats launch per slot).
                    // The FILTERED graph read its stats back per replay above.
                    if !filtered_stats_in_graph {
                        for j in 0..draft.len().max(draft_idx.len()) {
                            let rows0 = e.htod_i32(&[0])?;
                            let (mut th_d, mut z_d, mut mx_d) =
                                (e.zeros(1)?, e.zeros(1)?, e.zeros(1)?);
                            e.filter_stats(
                                &dctx.q_slots[j],
                                d_vocab,
                                &rows0,
                                &mut th_d,
                                &mut z_d,
                                &mut mx_d,
                                d_vocab,
                                1,
                                sp_temp,
                                sp.top_k,
                                sp.top_p,
                                sp.min_p,
                            )?;
                            draft_stats.push((
                                e.dtoh(&mx_d)?[0],
                                e.dtoh(&th_d)?[0],
                                e.dtoh(&z_d)?[0],
                            ));
                        }
                    }
                } else {
                    if skey_probe() && sampled {
                        eprintln!(
                            "[skey] chain=eager round={round} pure_temp={} top_k={} \
                             top_p={} min_p={} s_key_parked={:?}",
                            pure_temp as u8, sp.top_k, sp.top_p, sp.min_p, dctx.s_key,
                        );
                    }
                    // EAGER DRAFT (fallback: MoE head/trunk, huge k, MEMRA_SPEC_NOGRAPH, capture fail).
                    let chain_heads = !self.mtp_extra.is_empty();
                    let mut e_tok = last_token;
                    let mut d_seed = e.clone_dtod(&h_seed_buf)?;
                    let mut chain_tokens = if chain_heads {
                        vec![last_token]
                    } else {
                        Vec::new()
                    };
                    let mut chain_seeds = if chain_heads {
                        vec![e.clone_dtod(&h_seed_buf)?]
                    } else {
                        Vec::new()
                    };
                    for j in 0..k_this {
                        // GPU-ARGMAX DRAFT (2026-07-03): device logits + device argmax + 4-byte token
                        // read instead of the ~600KB full-vocab dtoh + host argmax per draft token.
                        let mtp_pos = pos + base0 + j;
                        // draft-side grammar mask (eager twin of the graph arm's in-graph node).
                        // A position with no legal draft-vocab row drops to unmasked drafting for
                        // the rest of the chain (pre-lane behaviour; verify still arbitrates).
                        if dmask_live {
                            dmask_live = upload_draft_mask(
                                e,
                                constraint.as_deref_mut().unwrap(),
                                &mut dctx.g_dmask,
                                mtp.d2t.as_ref(),
                                d_vocab,
                                dmask_words,
                            )?;
                        }
                        let mask = if dmask_live {
                            Some((&dctx.g_dmask, dmask_words))
                        } else {
                            None
                        };
                        let (dl_d, h_nextn) = if chain_heads {
                            if debug_spec {
                                eprintln!(
                                    "[mtp-chain-step] round={round} j={j} head={} replay_rows={}",
                                    mtp_chain_head_index(j, self.mtp_head_count()),
                                    chain_tokens.len(),
                                );
                            }
                            self.mtp_chain_forward_dev(
                                e,
                                &chain_tokens,
                                &chain_seeds,
                                &mut *scratch,
                                pos + base0 - 1,
                                embd_dev,
                                mask,
                            )?
                        } else {
                            self.mtp_head_forward_dev(
                                e,
                                mtp,
                                e_tok,
                                &d_seed,
                                &mut *scratch,
                                mtp_pos,
                                embd_dev,
                                mask,
                            )?
                        };
                        let tok_d = if sampled {
                            // FILTERED Gumbel-max: stats -> masked perturb -> argmax = one draw from
                            // the filtered softmax (filters off => th=0, exact v1 semantics).
                            if perturb_buf.is_none() {
                                perturb_buf = Some(e.zeros(d_vocab.max(n_vocab))?);
                            }
                            let mut q_row = e.clone_dtod(&dl_d)?; // retained q (penalized when on)
                            if pen_on {
                                let h = pen_hist_d.as_ref().unwrap();
                                let nh = h.len();
                                e.penalize_logits(
                                    &mut q_row,
                                    h,
                                    nh,
                                    sp.penalty_repeat,
                                    sp.penalty_freq,
                                    sp.penalty_present,
                                    d_vocab,
                                )?;
                            }
                            let rows0 = e.htod_i32(&[0])?;
                            let (mut th_d, mut z_d, mut mx_d) =
                                (e.zeros(1)?, e.zeros(1)?, e.zeros(1)?);
                            e.filter_stats(
                                &q_row, d_vocab, &rows0, &mut th_d, &mut z_d, &mut mx_d, d_vocab,
                                1, sp_temp, sp.top_k, sp.top_p, sp.min_p,
                            )?;
                            let (th, z, mx) =
                                (e.dtoh(&th_d)?[0], e.dtoh(&z_d)?[0], e.dtoh(&mx_d)?[0]);
                            let pb = perturb_buf.as_mut().unwrap();
                            e.gumbel_perturb_filtered(
                                &q_row, pb, d_vocab, sp_seed, sctr, sp_temp, mx, th,
                            )?;
                            sctr += 1;
                            draft_logits.push(q_row);
                            draft_stats.push((mx, th, z));
                            e.argmax_token_device(pb, d_vocab)?
                        } else {
                            e.argmax_token_device(&dl_d, d_vocab)?
                        };
                        let idx = e.dtoh_u32_one(&tok_d)?;
                        // #87 SENTINEL TRAP (eager twin — see the graph arm). Extra diagnostics
                        // here because the eager chain's operands are all readable: dl_d (the head
                        // logits row) and d_seed (this step's h_seed) name the first-NaN buffer.
                        if (idx as usize) >= d_vocab {
                            let dl_h = e.dtoh(&dl_d)?;
                            let dl_nan = dl_h.iter().filter(|v| v.is_nan()).count();
                            let seed_h = if chain_heads {
                                e.dtoh(chain_seeds.last().unwrap())?
                            } else {
                                e.dtoh(&d_seed)?
                            };
                            let seed_nan = seed_h.iter().filter(|v| v.is_nan()).count();
                            return Err(format!(
                                "draft(eager) argmax sentinel 0x{idx:08x} >= d_vocab {d_vocab} at \
                             round {round} j={j} pos={pos}: head-logits NaN {dl_nan}/{d_vocab}, \
                             step-seed NaN {seed_nan}/{n_embd} — refusing to dereference the \
                             embed row (#87 trap)"
                            )
                            .into());
                        }
                        let d = match &mtp.d2t {
                            Some(map) => map[idx as usize],
                            None => idx,
                        };
                        if sampled {
                            draft_idx.push(idx);
                        }
                        let draft_p = if p_min > 0.0
                            || opti_fork
                                .as_ref()
                                .is_some_and(|fork| fork.controller.is_some())
                        {
                            let p_d = e.prob_of_token_device(&dl_d, &tok_d, d_vocab)?;
                            Some(e.dtoh(&p_d)?[0])
                        } else {
                            None
                        };
                        if j == 0 {
                            controller_draft_prob = draft_p;
                        }
                        if let Some(p) = draft_p.filter(|_| p_min > 0.0)
                            && p < p_min
                            && (j > 0 || (pmin0 && base0 == 1))
                        {
                            break;
                        }
                        draft.push(d);
                        if chain_heads {
                            chain_tokens.push(d);
                            chain_seeds.push(h_nextn);
                        } else {
                            e_tok = d;
                            d_seed = h_nextn;
                        }
                        // speculative advance; a chain the grammar can no longer follow (EOS
                        // proposed) ends here — the prefix already proposed still rides verify.
                        if dmask_live
                            && !constraint
                                .as_deref_mut()
                                .unwrap()
                                .draft_advance(d)
                                .map_err(|e2| format!("constraint: {e2}"))?
                        {
                            break;
                        }
                    }
                    if !chain_heads
                        && opti_fork
                            .as_ref()
                            .is_some_and(|fork| fork.controller.is_some())
                    {
                        controller_eager_state = Some((e_tok, d_seed));
                    }
                }
            }
            let k_round = draft.len();
            if let Some(p) = pipe {
                p.draft_end(round);
            }
            drop(pipe_draft);

            ph_mark(&mut ph_draft, phase_on);
            // --- 2. VERIFY: one batched target forward. With a pending bonus, it rides as col 0
            //         (committing its KV/recur inside the SAME weight read); drafts follow. ---
            let verify_tokens: Vec<u32> = match pending {
                Some(b) => {
                    let mut v = Vec::with_capacity(k_round + 1);
                    v.push(b);
                    v.extend_from_slice(&draft);
                    v
                }
                None => draft.clone(),
            };
            let base = if pending.is_some() { 1 } else { 0 };
            // ckpt (REPLAY-FREE partial accept): retain per-layer state-rebuild inputs alongside
            // the verify. Pure buffer keep-alives + dtod clones — kernel work is unchanged.
            let mut ckpt = if let Some(ticket) = current_opti.as_mut() {
                Some(ticket.take_ckpt())
            } else if spec_replay {
                None
            } else {
                Some(VerifyCkpt::new(self.layers.len()))
            };
            let controller_can_probe = base == 1
                && k_round == 1
                && out.len().saturating_add(2) < max_new
                && controller_draft_prob.is_some()
                && opti_fork
                    .as_ref()
                    .and_then(|fork| fork.controller.as_ref())
                    .is_some_and(|policy| !policy.breaker_tripped);
            let mut successor_attempt: Option<OptiControllerTicket> = None;
            let mut rejected_probe: Option<(f32, u32)> = None;
            let mut controller_prepared: Option<OptiControllerPrepared> = None;
            if controller_can_probe {
                // Prepare d2/q and, on admission, d3 before either current verify half is
                // issued. N stage 0 can then be followed immediately by N+1 stage 0; once N's
                // boundary fires, those dev0 launches overlap N stage 1 on dev1. Preparing on
                // the primary stream after N stage 1 would serialize the supposed pipeline.
                let eager_pos = scratch.kv.len + 1;
                let (optimistic_pending, pending_probability) = self.opti_controller_draft_step(
                    e,
                    mtp,
                    &mut dctx,
                    &mut *scratch,
                    d_vocab,
                    &mut controller_eager_state,
                    eager_pos,
                    embd_dev,
                    graph_round_ok,
                )?;
                let first_probability = controller_draft_prob
                    .ok_or("optipipe controller probe lost first-token probability")?;
                let q_proxy = first_probability * pending_probability;
                OPTI_GATE_CHECKS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                OPTI_SHADOW_DRAFT_TOKENS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let admitted = opti_fork
                    .as_ref()
                    .and_then(|fork| fork.controller.as_ref())
                    .ok_or("optipipe controller policy disappeared")?
                    .admit(q_proxy);
                if admitted {
                    OPTI_GATE_ADMITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    let eager_pos = scratch.kv.len + 1;
                    let (optimistic_draft, optimistic_draft_probability) = self
                        .opti_controller_draft_step(
                            e,
                            mtp,
                            &mut dctx,
                            &mut *scratch,
                            d_vocab,
                            &mut controller_eager_state,
                            eager_pos,
                            embd_dev,
                            graph_round_ok,
                        )?;
                    OPTI_SHADOW_DRAFT_TOKENS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    let eager_seed = controller_eager_state.take().map(|(token, seed)| {
                        debug_assert_eq!(token, optimistic_draft);
                        seed
                    });
                    controller_prepared = Some(OptiControllerPrepared {
                        verify_tokens: [optimistic_pending, optimistic_draft],
                        draft_prob: optimistic_draft_probability,
                        eager_seed,
                        q_proxy,
                        scratch_len: scratch.kv.len,
                    });
                } else {
                    OPTI_GATE_REJECTS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    OPTI_WASTED_DRAFT_TOKENS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    rejected_probe = Some((q_proxy, optimistic_pending));
                    eprintln!(
                        "[opti-controller] reject q={q_proxy:.6} threshold={:.3}",
                        opti_fork
                            .as_ref()
                            .and_then(|fork| fork.controller.as_ref())
                            .expect("controller policy")
                            .threshold,
                    );
                }
            }
            let fork_attempt = match fork_generation.take() {
                Some(generation) if base == 1 && k_round == 1 => Some(generation),
                Some(generation) => {
                    opti_fork
                        .as_mut()
                        .expect("fork generation without fork state")
                        .retire(generation)?;
                    None
                }
                None => None,
            };
            let (tlogits_d, vx) = if let Some(p) = pipe {
                self.decode_step_t_core_pipelined(
                    e,
                    &verify_tokens,
                    pos,
                    &mut *cache,
                    embd_dev,
                    ckpt.as_mut(),
                    p,
                    round,
                )?
            } else if controller_can_probe {
                let fence = opti_fork
                    .as_ref()
                    .ok_or("optipipe controller probe lost fork state")?
                    .fence;
                let boundary = match current_opti.as_mut() {
                    Some(ticket) => ticket.take_boundary(),
                    None => self.verify_stage0_issue(
                        e,
                        &verify_tokens,
                        pos,
                        &mut *cache,
                        embd_dev,
                        ckpt.as_mut(),
                        None,
                        &fence,
                        Some(true),
                        None,
                    )?,
                };
                if let Some(prepared) = controller_prepared.take() {
                    let generation = {
                        let fork = opti_fork
                            .as_mut()
                            .ok_or("optipipe controller admission lost fork state")?;
                        let generation = fork.reserve_successor()?;
                        let rt = fork.rt;
                        let snapshot_fence = fork.fence;
                        opti_snapshot_one_stage_owned_into(
                            e,
                            cache,
                            rt,
                            &snapshot_fence,
                            0,
                            fork.successor_snapshot_mut(),
                        )?;
                        generation
                    };
                    let mut successor_ckpt = VerifyCkpt::new(self.layers.len());
                    let successor_boundary = self.verify_stage0_issue(
                        e,
                        &prepared.verify_tokens,
                        pos + verify_tokens.len(),
                        &mut *cache,
                        embd_dev,
                        Some(&mut successor_ckpt),
                        None,
                        &fence,
                        Some(false),
                        None,
                    )?;
                    OPTI_FORK_ATTEMPTS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    let fork = opti_fork
                        .as_ref()
                        .ok_or("optipipe controller ticket lost fork state")?;
                    successor_attempt = Some(fork.controller_ticket(
                        generation,
                        successor_boundary,
                        successor_ckpt,
                        prepared.verify_tokens,
                        prepared.draft_prob,
                        prepared.eager_seed,
                        prepared.q_proxy,
                        prepared.scratch_len,
                    ));
                    eprintln!(
                        "[opti-controller] issue generation={} q={:.6} threshold={:.3} \
                         verify={:?}",
                        generation.id,
                        prepared.q_proxy,
                        fork.controller.expect("controller policy").threshold,
                        prepared.verify_tokens,
                    );
                }
                let result = self.verify_stage1_finish(
                    e,
                    boundary,
                    &mut *cache,
                    ckpt.as_mut(),
                    None,
                    &fence,
                    successor_attempt.is_none(),
                )?;
                if let Some(ticket) = current_opti.as_mut() {
                    ticket.settle();
                }
                if successor_attempt.is_some() {
                    let fork = opti_fork
                        .as_mut()
                        .ok_or("optipipe successor snapshot lost fork state")?;
                    let rt = fork.rt;
                    let snapshot_fence = fork.fence;
                    opti_snapshot_one_stage_owned_into(
                        e,
                        cache,
                        rt,
                        &snapshot_fence,
                        1,
                        fork.successor_snapshot_mut(),
                    )?;
                    // Publish N only after both independent successor-state queues are complete.
                    fork.rt.publish_to(1, &e.stream())?;
                }
                result
            } else if let Some(ticket) = current_opti.as_mut() {
                let fork = opti_fork
                    .as_mut()
                    .ok_or("optipipe carried controller ticket lost fork state")?;
                let boundary = ticket.take_boundary();
                let result = self.verify_stage1_finish(
                    e,
                    boundary,
                    &mut *cache,
                    ckpt.as_mut(),
                    None,
                    &fork.fence,
                    true,
                )?;
                ticket.settle();
                result
            } else if let Some(generation) = fork_attempt {
                let fork = opti_fork
                    .as_mut()
                    .expect("fork generation without fork state");
                fork.capture_seed(e, generation, &h_seed_buf, &fill_prev, scratch.kv.len)?;
                let action = fork.mode.action(generation.id);
                let boundary = self.verify_stage0_issue(
                    e,
                    &verify_tokens,
                    pos,
                    &mut *cache,
                    embd_dev,
                    ckpt.as_mut(),
                    None,
                    &fork.fence,
                    Some(true),
                    None,
                )?;
                OPTI_FORK_ATTEMPTS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let mut ticket = fork.ticket(generation, boundary);
                if action == OptiForkAction::Abort {
                    return Err(format!(
                        "optipipe forced abort with generation {} stage0 in flight",
                        generation.id,
                    )
                    .into());
                }
                fork.reconcile(
                    e,
                    &mut *cache,
                    &mut *scratch,
                    &snap,
                    &mut h_seed_buf,
                    &mut fill_prev,
                    generation,
                    action,
                    verify_tokens[0],
                )?;
                let result = if action == OptiForkAction::Hit {
                    let boundary = ticket.take_boundary();
                    self.verify_stage1_finish(
                        e,
                        boundary,
                        &mut *cache,
                        ckpt.as_mut(),
                        None,
                        &fork.fence,
                        true,
                    )?
                } else {
                    // The optimistic boundary slot has no reader. Re-run the unchanged serial
                    // verify only after E_restart published the restored stage-0 state.
                    self.decode_step_t_core(
                        e,
                        &verify_tokens,
                        pos,
                        &mut *cache,
                        embd_dev,
                        ckpt.as_mut(),
                    )?
                };
                ticket.settle();
                debug_assert_eq!(ticket.generation, generation);
                fork.retire(generation)?;
                result
            } else {
                // The serial verify every non-fork round takes — the MTP route's
                // verify-graph door. The pool is None unless MEMRA_SPEC_VERIFY_GRAPH armed
                // a pool above, and then the walk replays the captured trunk instead of
                // re-issuing it launch by launch. `graph_round_ok` is the round's
                // headroom snapshot (see GRAPH_LAUNCH_MIN_FREE): below the floor the
                // round declines the pool exactly like an over-cap round and rides the
                // byte-identical eager walk — the `[spec]` suspension line above
                // already named the round.
                let vg_round = if verify_tokens.len() <= vg_t_cap && graph_round_ok {
                    vg_guard.as_mut().and_then(|g| g.as_mut())
                } else {
                    if let Some(g) = vg_guard.as_mut().and_then(|g| g.as_mut()) {
                        // The commit reads this flag to pick its arm; a round that declines
                        // the pool must not inherit a stale `true` from the round before it.
                        g.round_slab = false;
                    }
                    None
                };
                self.decode_step_t_core_vg(
                    e,
                    &verify_tokens,
                    pos,
                    &mut *cache,
                    embd_dev,
                    ckpt.as_mut(),
                    vg_round,
                )?
            };
            let pipe_accept = match pipe {
                Some(p) => Some(p.accept_begin(round)?),
                None => None,
            };

            if phase_sync {
                e.stream().synchronize()?;
            }
            ph_mark(&mut ph_verify, phase_on);
            // --- 3. GREEDY ACCEPT (walk prefix, stop at first mismatch) ---
            // DEVICE-ARGMAX ACCEPT: argmax every verify column ON DEVICE (same 2-pass kernels +
            // smallest-index tie-break as host argmax, argmax_gate-validated) and read back ONE
            // [T] u32 — replaces the T x n_vocab f32 dtoh + T host argmaxes per round.
            // t_pred[j] = target's greedy prediction for the slot after draft[j-1] (j>=1) or after
            // last_token (j==0). With a pending bonus, col 0 IS the prediction after last_token
            // (== the bonus), so every index shifts by `base` and last_pred is unused.
            let t_v = verify_tokens.len();
            let mut preds: Vec<u32> = Vec::new();
            if !sampled {
                for j in 0..t_v {
                    e.argmax_token_device_col(&tlogits_d, j, n_vocab, &mut preds_d, j)?;
                }
                preds = e.dtoh_u32(&preds_d)?; // <- the verify-GPU wait lands here
                // #87 SENTINEL TRAP, verify side: a sentinel pred becomes the round's bonus =
                // next round's last_token = the next chain's embed lookup. Catch it at the
                // source with the column named — an all-NaN VERIFY column implicates the
                // stage-split trunk (decode_step_t_core_ppn), not the draft head.
                if let Some(bad) = preds[..t_v].iter().position(|&p| (p as usize) >= n_vocab) {
                    let col = &tlogits_d.slice(bad * n_vocab..(bad + 1) * n_vocab);
                    let mut probe = e.zeros(n_vocab)?;
                    e.copy_view_into(&mut probe, 0, col, n_vocab)?;
                    let col_h = e.dtoh(&probe)?;
                    let col_nan = col_h.iter().filter(|v| v.is_nan()).count();
                    return Err(format!(
                        "verify argmax sentinel 0x{:08x} >= n_vocab {n_vocab} at round {round} \
                         col {bad}/{t_v} pos={pos}: verify-logits col NaN {col_nan}/{n_vocab} \
                         — the verify TRUNK produced a poisoned column (#87 trap). Run \
                         MEMRA_SPEC_NAN_SCAN=1 to name the layer that creates it (=2 to split \
                         that layer into attention and routed MoE). NOT the draft head, and NOT \
                         the PP stage split this message used to name: pp_cuts() returns None \
                         without MEMRA_PP_STAGES, so decode_step_t_core_ppn never runs unless \
                         that variable is set.",
                        preds[bad]
                    )
                    .into());
                }
            }
            ph_mark(&mut ph_wait, phase_on);
            let t_pred = |j: usize| -> u32 {
                if j == 0 && base == 0 {
                    last_pred
                } else {
                    // GREEDY-ONLY: `preds` is filled under `if !sampled` above. The debug print
                    // used to call this from the sampled arm and panicked the worker; it now goes
                    // through `debug_t_pred0`. Keep the strict index here — in the greedy walk an
                    // out-of-range pred is a real bug, not something to paper over.
                    debug_assert!(
                        !sampled,
                        "t_pred is greedy-only: `preds` is empty in the sampled arm"
                    );
                    preds[base + j - 1]
                }
            };
            let mut devacc_seeded = false;
            let mut devacc_acc: Option<CudaSlice<u32>> = None;
            let (n_acc, bonus) = if !sampled {
                // ROUND-STREAM stage (a) (MEMRA_SPEC_DEVACC=1 opt-in): the walk runs ON DEVICE
                // (spec_accept_greedy, verbatim rule) and the host reads back 8B (n_acc, bonus)
                // instead of the [T] preds. Same sync count — machinery for stages (b)/(c),
                // gated on token identity vs the host walk (the arms below are bit-equal rules).
                if crate::spec::spec_devacc() && k_round > 0 && !spec_replay && constraint.is_none()
                {
                    let draft_d = e.htod_u32_v(&draft)?;
                    let mut acc_out = e.alloc_u32_zeroed(2)?;
                    e.spec_accept_greedy(
                        &preds_d,
                        &draft_d,
                        last_pred,
                        base,
                        k_round,
                        &mut acc_out,
                    )?;
                    devacc_acc = Some(acc_out.clone());
                    // stage (b): next-round seed gathered ON DEVICE from acc_out before the host
                    // ever reads n_acc (j=base+n_acc -> vx col j-1; j==0 -> fill_prev). The three
                    // non-replay commit arms skip their host-offset seed copies (guarded below);
                    // the legacy spec_replay arm keeps its own rx-based seeding (excluded here).
                    // NOTE: fill_prev is NOT updated here — the commit arms' TRUE-HIDDEN
                    // REFRESH reads the OLD fill_prev (predecessor of this round's verify batch);
                    // the update lands after the arms (devacc_seeded guard below).
                    e.spec_seed_gather(&vx, &fill_prev, &acc_out, &mut h_seed_buf, base, n_embd)?;
                    // 3a: KV lens roll back on device (len = saved + base + n_acc, all arms'
                    // unified rule; full accept rewrites the verify-left value). Host mirrors
                    // update after the readback; commit_verified_prefix skips its len_d writes.
                    if let Some(successor) = successor_attempt.as_ref() {
                        opti_fork
                            .as_mut()
                            .ok_or("optipipe successor reconcile lost fork state")?
                            .queue_actual_reconcile(
                                e,
                                &snap,
                                &acc_out,
                                successor.verify_tokens[0],
                                base,
                            )?;
                    } else if let Some(ptrs) = &kv_len_ptrs {
                        let saved: Vec<i32> = (0..self.layers.len())
                            .map(|il| snap.kv_len[il].map(|v| v as i32).unwrap_or(0))
                            .collect();
                        let saved_d = e.htod_i32(&saved)?;
                        e.spec_rollback_kv(ptrs, &saved_d, &acc_out, base, self.layers.len())?;
                    }
                    devacc_seeded = true;
                    let ab = e.dtoh_u32(&acc_out)?;
                    (ab[0] as usize, ab[1])
                } else {
                    let mut n_acc = 0usize;
                    #[allow(clippy::needless_range_loop)]
                    // allow: the explicit index loop keeps the offset arithmetic visible and aligned with the device-side indexing
                    for j in 0..k_round {
                        if t_pred(j) == draft[j] {
                            n_acc += 1;
                        } else {
                            break;
                        }
                    }
                    // bonus = target's own token at the first non-accepted slot. n_acc in 0..=k; t_pred
                    // is defined for j in 0..=k (j==0 -> last_logits, j>=1 -> col j-1, last col = k-1).
                    (n_acc, t_pred(n_acc))
                }
            } else {
                // --- SAMPLED ACCEPT (rejection sampling): u_j < p_j(x_j)/q_j(x_j) walk ---
                if col_buf.is_none() {
                    col_buf = Some(e.zeros(n_vocab)?);
                }
                // FILTERED p_j: per-verify-col stats (one batched filter_stats call), then the
                // filtered gather. j==0&&base==0 reads last_col (its own stats row appended).
                let mut pj = vec![0f32; k_round.max(1)];
                let mut col_stats: Vec<(f32, f32, f32)> = Vec::new(); // (max, th, z) per verify col used
                if k_round > 0 {
                    let mut ids: Vec<u32> = Vec::new();
                    let mut rows: Vec<i32> = Vec::new();
                    #[allow(clippy::needless_range_loop)]
                    // allow: the explicit index loop keeps the offset arithmetic visible and aligned with the device-side indexing
                    for j in 0..k_round {
                        if j > 0 || base == 1 {
                            ids.push(draft[j]);
                            rows.push((base + j) as i32 - 1);
                        }
                    }
                    if !ids.is_empty() {
                        let nr = rows.len();
                        // penalties: materialize the used columns into one contiguous penalized
                        // buffer (rows remapped 0..nr) so stats+gathers see the penalized p.
                        // penalties: materialize used columns contiguously, penalize all rows in
                        // one launch, and point stats+gathers at the penalized buffer (rows 0..nr).
                        let p_rows: Vec<i32> = if pen_on {
                            (0..nr as i32).collect()
                        } else {
                            rows.clone()
                        };
                        if pen_on {
                            if pcol_buf.as_ref().map(|b| b.len()).unwrap_or(0) < nr * n_vocab {
                                pcol_buf = Some(e.zeros(nr * n_vocab)?);
                            }
                            let pc = pcol_buf.as_mut().unwrap();
                            for (i2, &r) in rows.iter().enumerate() {
                                let c = r as usize;
                                e.copy_view_into(
                                    pc,
                                    i2 * n_vocab,
                                    &tlogits_d.slice(c * n_vocab..(c + 1) * n_vocab),
                                    n_vocab,
                                )?;
                            }
                            let h = pen_hist_d.as_ref().unwrap();
                            let nh = h.len();
                            e.penalize_logits_rows(
                                pc,
                                h,
                                nh,
                                sp.penalty_repeat,
                                sp.penalty_freq,
                                sp.penalty_present,
                                n_vocab,
                                nr,
                            )?;
                        }
                        let p_src: &CudaSlice<f32> = if pen_on {
                            pcol_buf.as_ref().unwrap()
                        } else {
                            &tlogits_d
                        };
                        let rowsd = e.htod_i32(&p_rows)?;
                        let (mut th_d, mut z_d, mut mx_d) =
                            (e.zeros(nr)?, e.zeros(nr)?, e.zeros(nr)?);
                        e.filter_stats(
                            p_src, n_vocab, &rowsd, &mut th_d, &mut z_d, &mut mx_d, n_vocab, nr,
                            sp_temp, sp.top_k, sp.top_p, sp.min_p,
                        )?;
                        let idsd = e.htod_u32_v(&ids)?;
                        let mut outd = e.zeros(nr)?;
                        e.softmax_gather_filtered(
                            p_src, n_vocab, &idsd, &rowsd, &th_d, &z_d, &mut outd, n_vocab, nr,
                            sp_temp,
                        )?;
                        let outv = e.dtoh(&outd)?;
                        let (thv, zv, mxv) = (e.dtoh(&th_d)?, e.dtoh(&z_d)?, e.dtoh(&mx_d)?);
                        let mut oi = 0usize;
                        #[allow(clippy::needless_range_loop)]
                        // allow: the explicit index loop keeps the offset arithmetic visible and aligned with the device-side indexing
                        for j in 0..k_round {
                            if j > 0 || base == 1 {
                                pj[j] = outv[oi];
                                oi += 1;
                            }
                        }
                        col_stats = (0..nr).map(|i| (mxv[i], thv[i], zv[i])).collect();
                    }
                    if base == 0 {
                        let lc: &CudaSlice<f32> = if pen_on {
                            if col_buf.is_none() {
                                col_buf = Some(e.zeros(n_vocab)?);
                            }
                            let cb = col_buf.as_mut().unwrap();
                            e.copy_into(
                                cb,
                                0,
                                last_col_logits
                                    .as_ref()
                                    .expect("sampled: last_col_logits unset"),
                                n_vocab,
                            )?;
                            let h = pen_hist_d.as_ref().unwrap();
                            let nh = h.len();
                            e.penalize_logits(
                                cb,
                                h,
                                nh,
                                sp.penalty_repeat,
                                sp.penalty_freq,
                                sp.penalty_present,
                                n_vocab,
                            )?;
                            col_buf.as_ref().unwrap()
                        } else {
                            last_col_logits
                                .as_ref()
                                .expect("sampled: last_col_logits unset")
                        };
                        let rows0 = e.htod_i32(&[0])?;
                        let (mut th_d, mut z_d, mut mx_d) = (e.zeros(1)?, e.zeros(1)?, e.zeros(1)?);
                        e.filter_stats(
                            lc, n_vocab, &rows0, &mut th_d, &mut z_d, &mut mx_d, n_vocab, 1,
                            sp_temp, sp.top_k, sp.top_p, sp.min_p,
                        )?;
                        let idsd = e.htod_u32_v(&[draft[0]])?;
                        let mut outd = e.zeros(1)?;
                        e.softmax_gather_filtered(
                            lc, n_vocab, &idsd, &rows0, &th_d, &z_d, &mut outd, n_vocab, 1, sp_temp,
                        )?;
                        pj[0] = e.dtoh(&outd)?[0];
                        last_col_stats =
                            Some((e.dtoh(&mx_d)?[0], e.dtoh(&th_d)?[0], e.dtoh(&z_d)?[0]));
                    }
                }
                // q source: the graph arms (single-head AND chain) retained the head logits
                // in the persistent q_slots; the eager arm in per-round draft_logits clones.
                // Same raw-logit values either way. FILTERED q_j: stats from draft_stats
                // (eager pushes in-chain; the graph arms compute them post-replay from the
                // retained q with the same filter_stats program — bit-identical to the
                // in-graph stats that shaped the draw, keeping ONE accept path).
                let q_bufs: &[CudaSlice<f32>] = if dctx.graph_s.is_some() || dctx.chain_s.is_some()
                {
                    &dctx.q_slots
                } else {
                    &draft_logits
                };
                let mut n_acc = 0usize;
                for j in 0..k_round {
                    let (qmx, qth, qz) = draft_stats[j];
                    let idsd = e.htod_u32_v(&[draft_idx[j]])?;
                    let rowsd = e.htod_i32(&[0])?;
                    let thd = e.htod(&[qth])?;
                    let zd = e.htod(&[qz])?;
                    let _ = qmx;
                    let mut outd = e.zeros(1)?;
                    e.softmax_gather_filtered(
                        &q_bufs[j], d_vocab, &idsd, &rowsd, &thd, &zd, &mut outd, d_vocab, 1,
                        sp_temp,
                    )?;
                    let qj = e.dtoh(&outd)?[0];
                    let u = host_u01(sp_seed, uctr);
                    uctr += 1;
                    let accept = (u as f64) * (qj as f64) < pj[j] as f64;
                    // SKEY PROBE: q == 0 for the token the draft actually proposed is the
                    // exactness signature (see `skey_probe`). Impossible when the draft was
                    // drawn from the same filtered distribution the verify reconstructs here;
                    // `u * 0 < p` makes it an UNCONDITIONAL accept whenever p > 0.
                    if skey_probe() && qj == 0.0 {
                        eprintln!(
                            "[skey] EXACTNESS q=0 round={round} j={j} draft_tok={} \
                             draft_idx={} p={:e} u={u} accepted={} th_z={:?}",
                            draft[j], draft_idx[j], pj[j], accept as u8, draft_stats[j],
                        );
                    }
                    if accept {
                        n_acc += 1;
                    } else {
                        break;
                    }
                }
                let bonus = if n_acc == k_round {
                    // FULL ACCEPT: bonus ~ FILTERED softmax at the last verify column.
                    let col = base + k_round - 1;
                    let cb = col_buf.as_mut().unwrap();
                    e.copy_view_into(
                        cb,
                        0,
                        &tlogits_d.slice(col * n_vocab..(col + 1) * n_vocab),
                        n_vocab,
                    )?;
                    if pen_on {
                        let h = pen_hist_d.as_ref().unwrap();
                        let nh = h.len();
                        e.penalize_logits(
                            cb,
                            h,
                            nh,
                            sp.penalty_repeat,
                            sp.penalty_freq,
                            sp.penalty_present,
                            n_vocab,
                        )?;
                    }
                    if perturb_buf.is_none() {
                        perturb_buf = Some(e.zeros(d_vocab.max(n_vocab))?);
                    }
                    // STATS MUST COME FROM THIS COLUMN (bug fix 2026-08-05, lane/sampler-
                    // truncation-fix; receipts research/sampfix-20260805/). The old code reused
                    // `col_stats.last()` here, which is ALWAYS the wrong row: the gathered set
                    // covers verify columns 0..=(base+k_round-2) (rows pushed as base+j-1), while
                    // the full-accept bonus samples column base+k_round-1 — exactly ONE PAST the
                    // last gathered column, in both base arms. `th` is a threshold in e-units of
                    // its OWN row's max, so feeding a neighbour's (row_max, th) into
                    // gumbel_perturb_filtered mis-scales every e0 = exp((x-row_max)/T). When the
                    // donor column's peak is higher by more than T*ln(1/th), EVERY id fails
                    // `e0 >= th`, the whole perturbed row becomes -3.4e38, and the 2-pass argmax
                    // falls through to its smallest-index tie-break => token id 0 ("!") spliced
                    // mid-word. Fragility is ordered by how large th is: min_p pins th = min_p
                    // (0.05 => trigger at delta > 2.4 at T=0.8, fires constantly), top_p's
                    // mass-boundary th is smaller, top_k's k-th-largest th smaller still — which
                    // is why the head-to-head matrix saw min_p and top_p corrupt while top_k-only
                    // stayed clean. The pure-temp default regime is immune (th == 0 masks nothing,
                    // and row_max is unused once nothing is masked), so this fix is a byte-level
                    // no-op for the untruncated serve default. One extra one-block filter_stats
                    // per full-accept round is the whole cost.
                    let (mx, th) = {
                        let rows0 = e.htod_i32(&[0])?;
                        let (mut th_d, mut z_d, mut mx_d) = (e.zeros(1)?, e.zeros(1)?, e.zeros(1)?);
                        let cb0 = col_buf.as_ref().unwrap();
                        e.filter_stats(
                            cb0, n_vocab, &rows0, &mut th_d, &mut z_d, &mut mx_d, n_vocab, 1,
                            sp_temp, sp.top_k, sp.top_p, sp.min_p,
                        )?;
                        (e.dtoh(&mx_d)?[0], e.dtoh(&th_d)?[0])
                    };
                    let pb = perturb_buf.as_mut().unwrap();
                    let cb2 = col_buf.as_ref().unwrap();
                    e.gumbel_perturb_filtered(cb2, pb, n_vocab, sp_seed, sctr, sp_temp, mx, th)?;
                    sctr += 1;
                    let td = e.argmax_token_device(pb, n_vocab)?;
                    e.dtoh_u32_one(&td)?
                } else {
                    // REJECT at n_acc: bonus ~ norm(max(0, softmax_T(p) - softmax_T(q))).
                    let cb = col_buf.as_mut().unwrap();
                    if n_acc > 0 || base == 1 {
                        let col = base + n_acc - 1;
                        e.copy_view_into(
                            cb,
                            0,
                            &tlogits_d.slice(col * n_vocab..(col + 1) * n_vocab),
                            n_vocab,
                        )?;
                    } else {
                        let lc = last_col_logits.as_ref().unwrap();
                        e.copy_into(cb, 0, lc, n_vocab)?;
                    }
                    if pen_on {
                        let h = pen_hist_d.as_ref().unwrap();
                        let nh = h.len();
                        e.penalize_logits(
                            cb,
                            h,
                            nh,
                            sp.penalty_repeat,
                            sp.penalty_freq,
                            sp.penalty_present,
                            n_vocab,
                        )?;
                    }
                    let cb2 = col_buf.as_ref().unwrap();
                    let sc = sctr;
                    sctr += 1;
                    // p-stats for the reject column: from col_stats when the col was gathered,
                    // else (j==0&&base==0) from last_col_stats.
                    let p_stats = if n_acc > 0 || base == 1 {
                        // col index within the gathered set == number of gathered cols before n_acc
                        let gi = if base == 1 { n_acc } else { n_acc - 1 };
                        col_stats.get(gi).copied().unwrap_or({
                            (0.0, 0.0, 1.0) // unreachable: gathered cols always cover the reject slot
                        })
                    } else {
                        last_col_stats.expect("sampled: last_col_stats unset at reject")
                    };
                    let q_stats = draft_stats[n_acc];
                    if let Some(map) = &d2t_dev {
                        if q_full_buf.is_none() {
                            q_full_buf = Some(e.zeros(n_vocab)?);
                        }
                        let qf = q_full_buf.as_mut().unwrap();
                        e.scatter_trim_logits(&q_bufs[n_acc], map, qf, d_vocab, n_vocab)?;
                        let qf2 = q_full_buf.as_ref().unwrap();
                        e.residual_sample_filtered(
                            cb2,
                            Some(qf2),
                            n_vocab,
                            sp_temp,
                            sp_seed,
                            sc,
                            p_stats,
                            q_stats,
                            &mut sample_tok,
                        )?;
                    } else {
                        e.residual_sample_filtered(
                            cb2,
                            Some(&q_bufs[n_acc]),
                            n_vocab,
                            sp_temp,
                            sp_seed,
                            sc,
                            p_stats,
                            q_stats,
                            &mut sample_tok,
                        )?;
                    }
                    e.dtoh_u32(&sample_tok)?[0]
                };
                (
                    n_acc,
                    guard_vocab_token(
                        bonus,
                        n_vocab,
                        &format!("sampled verify bonus at round {round} pos={pos} n_acc={n_acc}"),
                    )?,
                )
            };
            // --- 3b. GRAMMAR TRUNCATION (constrained spec, 2026-08-03): the grammar is
            // an extra rejection rule AFTER the exactness verify (the batched-verify-twins
            // ordering). Walk the accepted drafts through the grammar in commit order; the
            // first illegal token truncates acceptance at its slot, and that slot's emission
            // is recomputed as the MASKED argmax of the target's own verify column — token-
            // identical to constrained plain greedy decode (an unmasked argmax that is
            // grammar-legal IS the masked argmax: masking only removes competitors). The
            // column D2H (~1MB) is paid only when a cut fires — the tight-grammar cost,
            // measured in acceptance numbers, never hidden.
            let (n_acc, bonus) = match constraint.as_deref_mut() {
                None => (n_acc, bonus),
                Some(c) => {
                    fn ce(e2: String) -> Box<dyn std::error::Error> {
                        format!("constraint: {e2}").into()
                    }
                    let mut na = n_acc;
                    let mut cut = false;
                    for (j, &d) in draft.iter().enumerate().take(n_acc) {
                        if c.is_allowed(d).map_err(ce)? {
                            c.consume(d).map_err(ce)?;
                        } else {
                            na = j;
                            cut = true;
                            dm_cut_tokens += n_acc - j;
                            break;
                        }
                    }
                    if cut {
                        dm_cuts += 1;
                    }
                    let mut bo = bonus;
                    if cut || !c.is_allowed(bo).map_err(ce)? {
                        let mut row = if na == 0 && base == 0 {
                            init_logits_host
                                .clone()
                                .ok_or("constraint: init logits missing (round-0 cut)")?
                        } else {
                            e.dtoh_view(
                                &tlogits_d.slice((base + na - 1) * n_vocab..(base + na) * n_vocab),
                            )?
                        };
                        c.mask_logits(&mut row).map_err(ce)?;
                        bo = argmax(&row) as u32;
                    }
                    c.consume(bo).map_err(ce)?;
                    (na, bo)
                }
            };
            let mut successor_valid = false;
            if let Some((q_proxy, expected_d2)) = rejected_probe {
                let v_n = n_acc == 1 && bonus == expected_d2;
                eprintln!(
                    "[opti-controller] shadow q={q_proxy:.6} admitted=false v_n={v_n} \
                     expected_d2={expected_d2} n_acc={n_acc} bonus={bonus}",
                );
            }
            if let Some(successor) = successor_attempt.as_ref() {
                successor_valid = n_acc == 1 && bonus == successor.verify_tokens[0];
                let generation = successor.generation;
                let q_proxy = successor.q_proxy;
                let expected_pending = successor.verify_tokens[0];
                let resolution_ms = successor.issued_at.elapsed().as_secs_f64() * 1e3;
                let fork = opti_fork
                    .as_mut()
                    .ok_or("optipipe successor resolution lost fork state")?;
                fork.finish_actual_reconcile(e, &mut *cache, &snap, n_acc, base, successor_valid)?;
                if successor_valid {
                    OPTI_FORK_HITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                } else {
                    OPTI_FORK_MISSES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    OPTI_RECONCILES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    OPTI_WASTED_DRAFT_TOKENS.fetch_add(2, std::sync::atomic::Ordering::Relaxed);
                }
                let breaker_tripped = fork
                    .controller
                    .as_mut()
                    .expect("controller policy")
                    .resolve(successor_valid);
                if breaker_tripped {
                    OPTI_BREAKER_TRIPS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
                eprintln!(
                    "[opti-controller] resolve generation={} hit={} q={q_proxy:.6} \
                     expected_pending={expected_pending} n_acc={n_acc} bonus={bonus} \
                     resolution_ms={resolution_ms:.3} reconcile={} breaker={}",
                    generation.id, successor_valid, !successor_valid, breaker_tripped,
                );
                if !successor_valid {
                    let mut successor = successor_attempt
                        .take()
                        .expect("controller successor disappeared on miss");
                    successor.settle();
                    fork.retire(generation)?;
                }
            }
            total_drafted += k_round;
            total_accepted += n_acc;
            if let Some(t) = sess_telem {
                // Greedy, rejection-sampling, and grammar truncation all converge here after
                // the accept decision is already on host. Fixed-size relaxed atomics only.
                t.record_round(k_round, n_acc);
            }
            if spec_stats {
                st_len_hist[k_round] += 1;
                #[allow(clippy::needless_range_loop)]
                // allow: the explicit index loop keeps the offset arithmetic visible and aligned with the device-side indexing
                for j in 0..k_round {
                    st_drafted[j] += 1;
                }
                #[allow(clippy::needless_range_loop)]
                // allow: the explicit index loop keeps the offset arithmetic visible and aligned with the device-side indexing
                for j in 0..n_acc {
                    st_accepted[j] += 1;
                }
                if n_acc == k_round {
                    st_full += 1;
                }
            }

            if debug_spec {
                eprintln!(
                    "[R{round}] pos={pos} out_len={} last_tok={last_token} draft={draft:?} n_acc={n_acc} bonus={bonus} t_pred0={}",
                    out.len(),
                    // NOT `t_pred(0)`: `preds` is filled only under `if !sampled` above, so on a
                    // sampled request round >= 1 (base == 1) indexed an EMPTY vector and PANICKED
                    // the GPU worker thread — a debug flag that killed the exact regime you would
                    // set it to investigate. See `debug_t_pred0`.
                    debug_t_pred0(sampled, base, last_pred, &preds)
                );
            }

            // --- 4. COMMIT: draft[0..n_acc] then bonus (n_acc + 1 tokens) ---
            let commit_started = std::time::Instant::now();
            // SESSION MODE: every accepted column is already in the CACHE — `out` must carry all
            // of them (overshoot past max_new included) or `committed` under-counts the cache rows
            // and the next turn's continuation seeds one token off (gate-caught 2026-07-05). The
            // single-shot path keeps the cap (its caller truncates + drops the cache anyway).
            #[allow(clippy::needless_range_loop)]
            // allow: the explicit index loop keeps the offset arithmetic visible and aligned with the device-side indexing
            for j in 0..n_acc {
                if !session_mode && out.len() >= max_new {
                    break;
                }
                out.push(draft[j]);
            }
            if pen_on {
                pen_hist.extend_from_slice(&draft[0..n_acc]);
                pen_hist.push(bonus);
            }
            let bonus_emitted = session_mode || out.len() < max_new;
            if bonus_emitted {
                out.push(bonus);
            }
            last_token = bonus;

            // --- 5. ROLLBACK + advance (§C) ---
            if n_acc == k_round && !spec_replay {
                // FULL ACCEPT, BONUS FOLD: all verify columns (pending? + drafts) are committed in
                // cache; the NEW bonus stays PENDING for the next round's verify batch — NO extra
                // T=1 trunk pass. The next draft chain seeds from the MTP block's h_nextn at the
                // bonus position: one MTP-block pass (~1/33 trunk cost) replaces the trunk read.
                // last_pred is dead in the pending path (t_pred reads verify col 0).
                //
                // PERSISTENT DRAFT KV, full-accept fill: the chain covered last_token +
                // draft[0..k_round-2] as INPUTS (slots P..P'-2); draft[k_round-1] (slot P'-1) was
                // only ever an output, so its entry is MISSING. Fill it from vh_seed — its EXACT
                // trunk hidden (the last verify column). set_len first: a p-min break may have
                // left one extra chain append at that slot. Partial accepts need NO fill (the
                // chain already covered every accepted position; round-start set_len truncates).
                self.restore_step_tp_kv_verified_prefix(e, &mut *cache, &snap, t_v, false)?;
                let mut vh_seed = e.zeros(n_embd)?;
                e.copy_view_into(
                    &mut vh_seed,
                    0,
                    &vx.slice((t_v - 1) * n_embd..t_v * n_embd),
                    n_embd,
                )?;
                if refresh {
                    // TRUE-HIDDEN REFRESH (2026-07-03, the HANDOVER-listed acceptance lever):
                    // overwrite ALL committed positions' scratch entries with K/V from their EXACT
                    // verify hiddens — the reference engine's mtp_update fills from true hiddens;
                    // the full stack (vx) is already resident from the verify. Replaces both the
                    // chain-approximate entries AND the old last-token-only fill. Acceptance-only
                    // (draft attention quality); exactness stays the verify's job.
                    scratch.set_len(e, pos)?;
                    // PREDECESSOR pairing: row i gets vx[i-1]; row 0 the carried fill_prev
                    // (hidden of the last committed row before this verify batch).
                    let mut vxs = e.zeros(t_v * n_embd)?;
                    e.copy_into(&mut vxs, 0, &fill_prev, n_embd)?;
                    if t_v > 1 {
                        e.copy_view_into(
                            &mut vxs,
                            n_embd,
                            &vx.slice(0..(t_v - 1) * n_embd),
                            (t_v - 1) * n_embd,
                        )?;
                    }
                    self.mtp_kv_fill_all(e, &verify_tokens, &vxs, pos, &mut *scratch, embd_dev)?;
                } else {
                    scratch.set_len(e, pos + base + k_round - 1)?;
                    // predecessor of the last draft = verify col t_v-2 (or fill_prev at t_v==1)
                    let mut hp = e.zeros(n_embd)?;
                    if t_v >= 2 {
                        e.copy_view_into(
                            &mut hp,
                            0,
                            &vx.slice((t_v - 2) * n_embd..(t_v - 1) * n_embd),
                            n_embd,
                        )?;
                    } else {
                        e.copy_into(&mut hp, 0, &fill_prev, n_embd)?;
                    }
                    self.mtp_kv_fill_all(
                        e,
                        &[draft[k_round - 1]],
                        &hp,
                        pos + base + k_round - 1,
                        &mut *scratch,
                        embd_dev,
                    )?;
                }
                // REFERENCE SEEDING: no pseudo pass — the next chain's step 0 IS the
                // reference's (id_last, h_prev) draft row; it appends the bonus's scratch
                // entry itself. Seed = TRUE hidden of the bonus's predecessor (last verify
                // col). Saves one MTP-block pass per round on top of the pairing fix.
                if !devacc_seeded {
                    e.copy_into(&mut h_seed_buf, 0, &vh_seed, n_embd)?;
                    e.copy_into(&mut fill_prev, 0, &vh_seed, n_embd)?;
                }
                pending = Some(bonus);
                if debug_spec {
                    eprintln!("  -> FULL ACCEPT (bonus pending, prev-h seed)");
                }
            } else if !spec_replay && base + n_acc >= 1 {
                // PARTIAL ACCEPT, REPLAY-FREE (2026-07-03 — the profiled #1 long-ctx spec cost):
                // the verify's first j = base+n_acc columns ARE the committed sequence, computed
                // bit-identically to eager (decode-exact contract) — so KEEP them: KV truncates to
                // pos+j, recurrent state rebuilds from the VerifyCkpt (same-kernel gdn prefix
                // re-run / pure state-clone restore), and the bonus stays PENDING exactly like the
                // full-accept path — the legacy duplicate trunk replay is gone. The next chain
                // seeds from the MTP pseudo-hidden of the bonus, whose seed = the TRUE verify
                // hidden of its predecessor (col j-1) — same one-hop pseudo structure as full
                // accept (never compounds: the next verify recomputes true hiddens for all
                // committed columns).
                let j = base + n_acc;
                // VERIFY-GRAPH SLAB COMMIT: when the captured trunk ran, the linear layers'
                // column stash was written into the graphs ctx's persistent slabs as in-graph
                // memcpy nodes, NOT into the per-column VerifyCkpt the cols arm reads — so the
                // commit must take the slab twin (same semantics, slab-addressed sources). The
                // ctx states which of the two this round produced via `round_slab`; trusting the
                // flag rather than the env keeps a round that fell back to the eager walk (a
                // capture that declined, a t the pool never captured) on the cols arm.
                let slab_commit = vg_guard
                    .as_ref()
                    .and_then(|g| g.as_ref())
                    .map(|g| g.round_slab)
                    .unwrap_or(false);
                if slab_commit {
                    self.dspark_commit_prefix_slab(
                        e,
                        &mut *cache,
                        &snap,
                        vg_guard
                            .as_ref()
                            .and_then(|g| g.as_ref())
                            .expect("slab_commit implies a graphs ctx"),
                        j,
                    )?;
                } else {
                    self.commit_verified_prefix(
                        e,
                        &mut *cache,
                        &snap,
                        ckpt.as_ref().unwrap(),
                        j,
                        devacc_seeded,
                        if devacc_seeded {
                            devacc_acc.as_ref().map(|a| (a, base, t_v))
                        } else {
                            None
                        },
                    )?;
                }
                let mut seed = e.zeros(n_embd)?;
                e.copy_view_into(
                    &mut seed,
                    0,
                    &vx.slice((j - 1) * n_embd..j * n_embd),
                    n_embd,
                )?;
                // Draft scratch: TRUE-HIDDEN REFRESH of the committed prefix (see the full-accept
                // branch); without it the chain entries stand and only the tail truncates. Either
                // way len ends at pos+j so the pseudo append lands at the bonus's slot pos+j
                // (persistent mode), rope pos+j+1 (chain convention).
                if refresh {
                    scratch.set_len(e, pos)?;
                    let mut vxs = e.zeros(j * n_embd)?;
                    e.copy_into(&mut vxs, 0, &fill_prev, n_embd)?;
                    if j > 1 {
                        e.copy_view_into(
                            &mut vxs,
                            n_embd,
                            &vx.slice(0..(j - 1) * n_embd),
                            (j - 1) * n_embd,
                        )?;
                    }
                    self.mtp_kv_fill_all(
                        e,
                        &verify_tokens[0..j],
                        &vxs,
                        pos,
                        &mut *scratch,
                        embd_dev,
                    )?;
                } else {
                    scratch.set_len(e, pos + j)?;
                }
                // REFERENCE SEEDING (see the full-accept branch): seed = TRUE hidden of the
                // bonus's predecessor (verify col j-1); no pseudo pass.
                if !devacc_seeded {
                    e.copy_into(&mut h_seed_buf, 0, &seed, n_embd)?;
                    e.copy_into(&mut fill_prev, 0, &seed, n_embd)?;
                }
                pending = Some(bonus);
                if debug_spec {
                    eprintln!("  -> PARTIAL(replay-free j={j}, bonus pending, prev-h seed)");
                }
            } else if !spec_replay {
                // ZERO ROUND FOLD (2026-07-10, verify-cost target #3): base+n_acc == 0 — a
                // pending-less round where nothing was accepted (PMIN0 zero-draft chains after a
                // replay/commit, or plain 0-accept rounds at round 0). The old path replayed
                // [bonus] through a FULL m=1 trunk+head forward (the 489us full-vocab head pass
                // measured at ~0.75/round on PMIN0 configs). Instead: restore the pre-round
                // snapshot and let the bonus ride the NEXT round's verify as col 0 — the existing
                // base=1 pending machinery, bit-identical by the decode-exact verify contract.
                // Seed: the bonus's predecessor is the last COMMITTED token, whose hidden
                // fill_prev already carries (same seeding as the 1-token-replay case it replaces).
                cache.rollback(e, &snap, 0)?;
                scratch.set_len(e, pos)?;
                e.copy_into(&mut h_seed_buf, 0, &fill_prev, n_embd)?;
                pending = Some(bonus);
                if debug_spec {
                    eprintln!("  -> ZERO-ROUND FOLD (bonus pending, fill_prev seed)");
                }
            } else {
                // PARTIAL ACCEPT, LEGACY REPLAY (seam MEMRA_SPEC_REPLAY=1 — or j==0: nothing of
                // this round survives, only possible before the first pending exists, ~round 0):
                // restore EVERYTHING to the pre-round snapshot (KV truncate to pos + recur
                // restore), then replay the committed prefix pending? ++ draft[0..n_acc] ++
                // [bonus] as ONE batched T forward — single weight read, bit-identical to greedy
                // (the verify-all-columns path is the same math). Commits the bonus with a TRUE
                // trunk hidden.
                cache.rollback(e, &snap, 0)?; // accept_len=0: KV len = pos, recur = snapshot
                let mut replay: Vec<u32> = Vec::with_capacity(base + n_acc + 1);
                if let Some(b) = pending.take() {
                    replay.push(b);
                }
                replay.extend_from_slice(&draft[0..n_acc]);
                replay.push(bonus);
                // Full-stack forward (decode_step_t_core = decode_step_t_h_emb_dev's body):
                // Predecessor pairing seeds from the PREDECESSOR row (col len-2) — the same-row path takes the
                // last col exactly as before (byte-identical to the old _h_emb_dev call).
                let (rl_d, rx) = if self.batched_serving_numeric_class() {
                    let mut logits = Vec::with_capacity(replay.len() * n_vocab);
                    let mut hidden = e.uninit(replay.len() * n_embd)?;
                    for (row, &token) in replay.iter().enumerate() {
                        let (row_logits, row_hidden) =
                            self.spec_target_step_h(e, token, &mut *cache)?;
                        logits.extend_from_slice(&row_logits);
                        e.dtod_copy_into(&row_hidden, &mut hidden, row * n_embd)?;
                    }
                    (e.htod(&logits)?, hidden)
                } else {
                    self.decode_step_t_core(e, &replay, pos, &mut *cache, embd_dev, None)?
                };
                // last_pred = argmax of the LAST column's logits (predicts the token after `bonus`)
                // — device argmax + one 4-byte read instead of the full-vocab column dtoh.
                e.argmax_token_device_col(&rl_d, replay.len() - 1, n_vocab, &mut preds_d, 0)?;
                last_pred = guard_vocab_token(
                    e.dtoh_u32(&preds_d)?[0],
                    n_vocab,
                    &format!("replay last_pred at round {round} pos={pos}"),
                )?;
                if sampled {
                    let lr0 = replay.len();
                    let lc = last_col_logits
                        .as_mut()
                        .expect("sampled: last_col_logits unset");
                    e.copy_view_into(
                        lc,
                        0,
                        &rl_d.slice((lr0 - 1) * n_vocab..lr0 * n_vocab),
                        n_vocab,
                    )?;
                }
                let lr = replay.len();
                if lr >= 2 {
                    e.copy_view_into(
                        &mut h_seed_buf,
                        0,
                        &rx.slice((lr - 2) * n_embd..(lr - 1) * n_embd),
                        n_embd,
                    )?;
                } else {
                    // 1-token replay (round-0 miss): the bonus's predecessor is the OLD
                    // last_token, whose own-row hidden fill_prev still holds.
                    e.copy_into(&mut h_seed_buf, 0, &fill_prev, n_embd)?;
                }
                // the bonus is COMMITTED here — it becomes the last committed row.
                let mut rh_last = e.zeros(n_embd)?;
                e.copy_view_into(
                    &mut rh_last,
                    0,
                    &rx.slice((lr - 1) * n_embd..lr * n_embd),
                    n_embd,
                )?;
                e.copy_into(&mut fill_prev, 0, &rh_last, n_embd)?;
                if debug_spec {
                    eprintln!("  -> PARTIAL(replay={replay:?}), next_pred={last_pred}");
                }
            }
            if devacc_seeded {
                // stage (b) epilogue: fill_prev takes the gathered seed AFTER the refresh fills
                // consumed the old value (both slots carry the same value in every non-replay arm).
                e.copy_into(&mut fill_prev, 0, &h_seed_buf, n_embd)?;
            }
            if successor_valid {
                let optimistic_scratch_len = successor_attempt
                    .as_ref()
                    .expect("valid controller successor disappeared")
                    .scratch_len;
                // The normal current-round commit refreshed/truncated the logical scratch tail.
                // Its optimistic successor row was already written physically, so restoring only
                // the retained logical length makes that row live for the carried round.
                scratch.set_len(e, optimistic_scratch_len)?;
            }
            if let Some(current) = current_opti.take() {
                opti_fork
                    .as_mut()
                    .ok_or("optipipe current retirement lost fork state")?
                    .retire(current.generation)?;
            }
            if successor_valid {
                let successor = successor_attempt
                    .take()
                    .expect("valid controller successor disappeared before promotion");
                let generation = successor.generation;
                opti_fork
                    .as_mut()
                    .ok_or("optipipe successor promotion lost fork state")?
                    .promote_successor_snapshot(&mut snap, generation);
                carried_opti = Some(successor);
            }
            if anatomy_on {
                // Commit/rollback is normally asynchronous on the primary/head stream. Bound it
                // only for this diagnostic so it does not disappear into the following draft's
                // first token readback.
                e.stream().synchronize()?;
                ph_commit += commit_started.elapsed().as_secs_f64();
            }
            // adaptive-K update (host math, zero syncs): next round drafts accepted-run + 1,
            // clamped to [floor(pos), k_cap]. cache.pos is post-rollback here (the round's
            // final position — the floor's position key reads the committed depth). Burst
            // rounds (`continue` above) draft the captured fixed depth and skip this, exactly
            // like gemma's burst arm.
            if adapt {
                let fl_now = floor_at(cache.pos);
                kc = (n_acc + 1).clamp(fl_now.min(k_cap), k_cap);
            }
            ph_mark(&mut ph_rest, phase_on);
            if let Some(p) = pipe {
                p.accept_end(round);
            }
            drop(pipe_accept);
            if let Some(t0) = round_t0 {
                let ms = t0.elapsed().as_secs_f64() * 1e3;
                ROUND_MS.fetch_add((ms * 1e3) as u64, std::sync::atomic::Ordering::Relaxed);
                let n = ROUND_N.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                if n.is_multiple_of(32) {
                    eprintln!(
                        "[spec-round] rounds={n} avg round wall={:.2} ms (emitted={} drafted so far)",
                        ROUND_MS.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1e3 / n as f64,
                        out.len()
                    );
                }
            }
            round += 1;
            // sse-cadence: this round's accepted drafts + bonus are committed (out is
            // append-only past step 4) — flush at round cadence.
            keep_going = flush_commit(&mut on_commit, &out, &mut flushed);
        }
        if let Some(mut ticket) = carried_opti.take() {
            opti_fork
                .as_mut()
                .ok_or("optipipe tail drain lost fork state")?
                .cancel_controller_ticket(e, &mut *cache, &mut *scratch, &snap, &mut ticket)?;
        }
        // sse-cadence: nothing below appends to `out`; flush any remainder (defensive).
        // (verdict ignored — the burst is over either way; the session tail runs unchanged.)
        let _ = flush_commit(&mut on_commit, &out, &mut flushed);

        if spec_stats {
            let per_slot: Vec<String> = (0..k)
                .map(|j| {
                    if st_drafted[j] > 0 {
                        format!(
                            "{}/{}={:.3}",
                            st_accepted[j],
                            st_drafted[j],
                            st_accepted[j] as f64 / st_drafted[j] as f64
                        )
                    } else {
                        "0/0".into()
                    }
                })
                .collect();
            let acc = if total_drafted > 0 {
                total_accepted as f64 / total_drafted as f64
            } else {
                0.0
            };
            eprintln!(
                "[spec-stats] rounds={round} full_accept={st_full} len_hist={st_len_hist:?} \
                       per_slot=[{}] total={total_accepted}/{total_drafted}={acc:.3} \
                       tok_per_round={:.3}",
                per_slot.join(" "),
                (total_accepted + round) as f64 / round.max(1) as f64
            );
        }
        if constraint.is_some() {
            eprintln!(
                "[draft-mask] mask_rounds={dm_rounds} clone_total={:.3}ms \
                 clone_per_round={:.4}ms gram_cuts={dm_cuts}/{round} cut_tokens={dm_cut_tokens}",
                dm_clone_ns as f64 / 1e6,
                dm_clone_ns as f64 / 1e6 / dm_rounds.max(1) as f64
            );
        }
        if phase_on {
            let tot = ph_draft + ph_verify + ph_wait + ph_rest;
            eprintln!(
                "[spec-phase] draft={:.1}ms ({:.1}%) verify-issue={:.1}ms ({:.1}%) verify-wait={:.1}ms ({:.1}%) commit-host={:.1}ms ({:.1}%) rounds={round}",
                ph_draft * 1e3,
                ph_draft / tot * 100.0,
                ph_verify * 1e3,
                ph_verify / tot * 100.0,
                ph_wait * 1e3,
                ph_wait / tot * 100.0,
                ph_rest * 1e3,
                ph_rest / tot * 100.0
            );
        }
        if anatomy_on {
            let rounds_f = round.max(1) as f64;
            let other = (ph_rest - ph_commit).max(0.0);
            eprintln!(
                "[spec-anatomy] per-round draft={:.3}ms pp-verify={:.3}ms \
                 verify-accept={:.3}ms commit-rollback={:.3}ms other={:.3}ms rounds={round}",
                ph_draft * 1e3 / rounds_f,
                ph_verify * 1e3 / rounds_f,
                ph_wait * 1e3 / rounds_f,
                ph_commit * 1e3 / rounds_f,
                other * 1e3 / rounds_f,
            );
        }
        let _pipe_tail = pipe.map(|p| p.primary()).transpose()?;
        // SESSION TAIL: leave the session in the exact invariant the next turn's suffix prime
        // expects — every row in `committed` has trunk KV/recur state AND an exact draft-KV row.
        // Park the draft-graph ctx back on the session (the serve-burst fixed-cost fix): the next
        // burst replays instead of recapturing. Error paths (`?` above) drop it — recaptured then.
        if let Some(slot) = sess_draft_slot.take() {
            *slot = Some(dctx);
        }
        let t_rounds = t_ent.elapsed();
        if let Some((committed, last_h, next_pred_slot, sctr_slot, uctr_slot)) = sess_tail.take() {
            // NEXT BURST'S BOUNDARY TOKEN (lane/sampled-spec-quality, Item 1). Greedy stashes
            // the argmax `last_pred` exactly as before (byte contract). SAMPLED draws the token
            // HERE, where the sampler, the session Philox counters and the penalty window are
            // all live and the boundary logits row still exists — that is the "make the state
            // available" half of the fix; the consuming burst then just emits it. `sctr` is
            // written to the session BELOW the draws so the advance is never lost.
            *next_pred_slot = Some(last_pred);
            let sample_boundary = sampled && constraint.is_none() && spec_sampled_boundary_on();
            let mut stashed_pending = false;
            if let Some(b) = pending.take() {
                if !sampled {
                    // PENDING-CARRY (2026-08-01): stash the bonus on the session instead of
                    // committing it with a solo T=1 pass — the next empty-suffix greedy burst
                    // consumes it as round-0 verify col 0 (a plain round edge; the old tail
                    // commit + next burst's init feed were 11.6+11.5ms solo trunk passes per
                    // burst on H100 q27, [spec-setup] trace). b stays in `out` (emitted) but
                    // OUT of `committed` (cache rows == committed); the consuming call
                    // prepends it once its verify commits the row. next_pred is unknowable
                    // without the commit pass — None; callers gate on pending_tok too.
                    debug_assert_eq!(out.last(), Some(&b), "pending must be the last emitted");
                    if let Some(slot) = sess_pending_slot.take() {
                        *slot = Some(b);
                    }
                    *next_pred_slot = None;
                    // fill_prev = hidden of the last COMMITTED row (b's predecessor) — the
                    // exact chain-seed/fill anchor the consuming burst (or a flush) needs.
                    *last_h = Some(e.clone_dtod(&fill_prev)?);
                    stashed_pending = true;
                } else {
                    // SAMPLED tail (unchanged): commit the bonus (one T=1 pass) + draft fill —
                    // the sampled round-0 accept needs this pass's logits (last_col_logits).
                    let pos_b = cache.pos;
                    scratch.set_len(e, pos_b)?;
                    let (lg_b, hb) = self.spec_target_step_h(e, b, &mut *cache)?;
                    // after a FULL-accept exit `last_pred` is STALE (it predicted the bonus
                    // itself — the prediction AFTER the bonus never materialized; it would have
                    // been the next round's verify col 0). The commit's logits ARE that
                    // prediction — so they are also the row the next burst's boundary token
                    // comes off, and (lane/sampled-spec-quality) it is DRAWN from them here.
                    *next_pred_slot = Some(if sample_boundary {
                        sample_boundary_token(
                            e,
                            &lg_b,
                            &sp,
                            &pen_hist,
                            &mut sctr,
                            "burst-tail-commit",
                        )?
                    } else {
                        argmax(&lg_b) as u32
                    });
                    self.mtp_kv_fill_all(e, &[b], &fill_prev, pos_b, &mut *scratch, embd_dev)?;
                    *last_h = Some(hb);
                }
            } else {
                // fill_prev tracks the hidden of the last COMMITTED row throughout the loop.
                *last_h = Some(e.clone_dtod(&fill_prev)?);
                if sample_boundary {
                    // No pending to commit, so the boundary row is the one `last_pred` was
                    // argmaxed from and the sampled path keeps it on device: the init feed's
                    // logits when the burst ran zero rounds, else the legacy-replay path's
                    // last verify column (both predict the token AFTER the last committed
                    // row). It is retained precisely because round 0's accept test needs it,
                    // so the draw costs no extra D2H of the [n_vocab] row.
                    match last_col_logits.as_ref() {
                        Some(lc) => {
                            *next_pred_slot = Some(sample_boundary_token_dev(
                                e,
                                lc,
                                n_vocab,
                                &sp,
                                &pen_hist,
                                &mut sctr,
                                "burst-tail-nopending",
                            )?);
                        }
                        // NAME THE FALLBACK (house standard): unreachable today — a sampled
                        // burst always feeds or replays, so the row exists — but if it ever
                        // is, the stream takes a greedy token and SAYS so rather than
                        // silently regressing to the pre-lane behaviour.
                        None => eprintln!(
                            "[spec-boundary] sampled tail kept the ARGMAX boundary token \
                             (reason: no retained boundary logits row)"
                        ),
                    }
                }
            }
            *sctr_slot = sctr;
            *uctr_slot = uctr;
            committed.extend_from_slice(prompt);
            if let Some(cb) = carried_pending {
                // the consumed carry's cache row landed in round 0's verify (every pending
                // round commits col 0) — it joins `committed` here, in sequence order.
                committed.push(cb);
            }
            if stashed_pending {
                // ZERO-EMIT BURST (2026-08-06 c=8 serve panic, pre-existing since b4aea184):
                // `out.len() - 1` underflowed on an EMPTY `out` — "range end index
                // 18446744073709551615 out of range for slice of length 0", killing the
                // memra-gpu-worker and failing 31 of 32 concurrent requests with "worker closed
                // stream". Reachable because `pending` starts as `carried_pending` (a bonus
                // stashed by the PREVIOUS burst) while `out` starts empty, and the carry is
                // deliberately NOT pushed to `out` (line ~3239: the burst that emitted it already
                // did). So a burst that stashes a pending without emitting anything of its own —
                // the round loop exits before a push, e.g. the ring drain's `out.len() < max_new`
                // guard skipping every token under a tight budget — arrives here with
                // out.len() == 0 and stashed_pending == true.
                //
                // The invariant is unchanged: `committed` gets every emitted token EXCEPT the
                // stashed bonus. With nothing emitted, that is nothing — and the carry pushed
                // just above is already accounted. Saturating, not a min/assert: an empty `out`
                // here is a legitimate burst shape, not a corrupt state.
                let emitted = out.len().saturating_sub(1);
                committed.extend_from_slice(&out[..emitted]);
            } else {
                committed.extend_from_slice(&out); // FULL out incl. overshoot — all committed
            }
            debug_assert_eq!(
                cache.pos,
                committed.len(),
                "session invariant: cache rows == committed tokens"
            );
            if setup_trace {
                e.stream().synchronize()?; // bound the async tail fill in the trace
                let t_tail = t_ent.elapsed();
                eprintln!(
                    "[spec-setup] init={:.2}ms cap={:.2}ms fill={:.2}ms rounds={:.2}ms tail={:.2}ms total={:.2}ms out={} cont={}",
                    t_init.as_secs_f64() * 1e3,
                    (t_cap - t_init).as_secs_f64() * 1e3,
                    (t_fill - t_cap).as_secs_f64() * 1e3,
                    (t_rounds - t_fill).as_secs_f64() * 1e3,
                    (t_tail - t_rounds).as_secs_f64() * 1e3,
                    t_tail.as_secs_f64() * 1e3,
                    out.len(),
                    continuation
                );
            }
            return Ok((out, total_drafted, total_accepted));
        }
        out.truncate(max_new);
        Ok((out, total_drafted, total_accepted))
    }

    /// Anchor-bounded DSpark target extraction. The trunk sees the exact generated token tape;
    /// only requested hidden rows and target-logit rows cross PCIe. An anchor token at p pairs
    /// with the pre-output-norm h[p-1] carrier, exactly as the existing replay/NextN path does.
    #[allow(clippy::too_many_arguments)] // allow: the parameter list mirrors the kernel/FFI/call contract; bundling into a struct is a refactor, not a lint fix
    pub fn extract_dspark_anchors(
        &self,
        e: &Engine,
        tokens: &[u32],
        anchor_positions: &[usize],
        gamma: usize,
        top_k: usize,
        chunk: usize,
        temperature: f32,
    ) -> Result<Vec<DsparkAnchorRecord>, Box<dyn std::error::Error>> {
        if tokens.len() < gamma + 2 || gamma == 0 || chunk < 2 {
            return Err("DSpark extraction token tape/gamma/chunk is invalid".into());
        }
        if anchor_positions.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err("DSpark anchor positions must be sorted and unique".into());
        }
        for &position in anchor_positions {
            if position == 0 || position + gamma >= tokens.len() {
                return Err(format!(
                    "DSpark anchor {position} has no predecessor or cannot cover gamma={gamma} in {} tokens",
                    tokens.len()
                )
                .into());
            }
        }

        let n_vocab = self.output.out_features();
        let n_embd = self.cfg.n_embd as usize;
        let mut cache =
            crate::pp::new_cache_planned(e, &self.cfg, &self.plan, tokens.len() + gamma + 8)?;
        let (embd_qt, embd_rb) = self.embd.qt_and_row_bytes(n_embd);
        let embd_gpu = if spec_host_embd() {
            None
        } else {
            Some(
                self.embd_gpu
                    .get_or_init(|| e.upload_u8(&self.embd.raw).expect("embed table upload")),
            )
        };
        let embd_dev = embd_gpu.map(|gpu| (gpu, embd_qt, embd_rb));

        struct PendingRecord {
            position: usize,
            hidden: Option<Vec<f32>>,
            tokens: Vec<u32>,
            target_top_ids: Vec<Option<Vec<u32>>>,
            target_top_logits: Vec<Option<Vec<f32>>>,
            target_top_probs: Vec<Option<Vec<f32>>>,
            target_tail_probs: Vec<Option<f32>>,
        }

        let mut pending: Vec<PendingRecord> = anchor_positions
            .iter()
            .map(|&position| PendingRecord {
                position,
                hidden: None,
                tokens: tokens[position..=position + gamma].to_vec(),
                target_top_ids: vec![None; gamma],
                target_top_logits: vec![None; gamma],
                target_top_probs: vec![None; gamma],
                target_tail_probs: vec![None; gamma],
            })
            .collect();

        let mut start = 0usize;
        while start < tokens.len() {
            let end = (start + chunk).min(tokens.len());
            let chunk_tokens = &tokens[start..end];
            let (target_logits, hidden_rows) =
                self.decode_step_t_core(e, chunk_tokens, start, &mut cache, embd_dev, None)?;
            for record in &mut pending {
                let hidden_position = record.position - 1;
                if hidden_position >= start && hidden_position < end {
                    let local = hidden_position - start;
                    record.hidden = Some(
                        e.dtoh_view(&hidden_rows.slice(local * n_embd..(local + 1) * n_embd))?,
                    );
                }
                for slot in 0..gamma {
                    let target_row = record.position + slot;
                    if target_row < start || target_row >= end {
                        continue;
                    }
                    let local = target_row - start;
                    let logits =
                        e.dtoh_view(&target_logits.slice(local * n_vocab..(local + 1) * n_vocab))?;
                    let (ids, top_logits, probs, tail) =
                        dspark_sparse_softmax_topk(&logits, top_k, temperature)?;
                    record.target_top_ids[slot] = Some(ids);
                    record.target_top_logits[slot] = Some(top_logits);
                    record.target_top_probs[slot] = Some(probs);
                    record.target_tail_probs[slot] = Some(tail);
                }
            }
            start = end;
        }

        pending
            .into_iter()
            .map(|record| {
                let hidden = record
                    .hidden
                    .ok_or_else(|| format!("missing DSpark hidden at {}", record.position))?;
                let target_top_ids =
                    flatten_dspark_rows(record.target_top_ids, record.position, "target ids")?;
                let target_top_logits = flatten_dspark_rows(
                    record.target_top_logits,
                    record.position,
                    "target logits",
                )?;
                let target_top_probs =
                    flatten_dspark_rows(record.target_top_probs, record.position, "target probs")?;
                let target_tail_probs = record
                    .target_tail_probs
                    .into_iter()
                    .enumerate()
                    .map(|(slot, value)| {
                        value.ok_or_else(|| {
                            format!("missing DSpark tail at {} slot {slot}", record.position)
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(DsparkAnchorRecord {
                    position: record.position,
                    hidden,
                    tokens: record.tokens,
                    target_top_ids,
                    target_top_logits,
                    target_top_probs,
                    target_tail_probs,
                })
            })
            .collect()
    }

    /// TEACHER-FORCED REPLAY ACCEPTANCE (hqmtp MTP-heal protocol): walk a FIXED token
    /// sequence and, at sampled positions, compare the MTP head's K-token draft chain against
    /// the trunk's own teacher-forced greedy predictions. Nothing is generated — the context is
    /// the corpus text itself, so (a) degenerate self-generated loops cannot inflate acceptance
    /// and (b) two arms (bf16 ceiling vs NVFP4) score on IDENTICAL contexts, isolating the
    /// quant-induced head/hidden-state mismatch from text drift.
    ///
    /// Per eval position p (context = tokens[0..=p], predecessor pairing as in spec decode):
    ///   draft_j  = chain token j from (tokens[p], h_{p-1}), then its own drafts — the exact
    ///              eager spec-decode chain (same mtp_head_forward_dev, same rope positions).
    ///   target_j = teacher-forced greedy pick for position p+1+j (argmax of the trunk logits
    ///              at forced context tokens[0..p+j]). For j==0 this equals live spec
    ///              acceptance; for j>=1 live verify would condition on the drafts, here it
    ///              conditions on the corpus — deterministic and arm-comparable by design.
    ///
    /// Returns (rows, bg): one (p, drafts[k], targets[k]) row per eval position (ascending p),
    /// plus the full teacher-forced greedy track bg (bg[i] = greedy pick for position i, i>=1)
    /// so harnesses can cross-check runs (e.g. different chunk sizes must give identical bg).
    ///
    /// `hdump`: when Some, every position's pre-output_norm trunk hidden (the exact rows the
    /// draft-KV fill pairs from) streams to the file as little-endian f32 [t_total, n_embd] —
    /// the head-distillation extraction (hqmtp): the ENGINE is the source of truth for trunk
    /// hiddens (HF torch reproductions of the hybrid trunk measured only ~0.5 greedy
    /// agreement vs this path — not usable as a training-data source).
    #[allow(clippy::type_complexity)] // allow: one-shot composite type; naming it would hide the shape that matters at the call site
    pub fn replay_acceptance(
        &self,
        e: &Engine,
        tokens: &[u32],
        k: usize,
        stride: usize,
        chunk: usize,
        mut hdump: Option<&mut std::fs::File>,
    ) -> Result<(Vec<(usize, Vec<u32>, Vec<u32>)>, Vec<u32>), Box<dyn std::error::Error>> {
        assert!(k >= 1 && stride >= 1 && chunk >= 2);
        let mtp = self
            .mtp
            .as_ref()
            .expect("replay_acceptance requires an MTP head");
        let n_vocab = self.output.out_features();
        let d_vocab = mtp
            .shared_head_head
            .as_ref()
            .unwrap_or(&self.output)
            .out_features();
        let n_embd = self.cfg.n_embd as usize;
        let t_total = tokens.len();
        assert!(t_total >= 8, "corpus too short ({t_total} tokens)");
        // STAGE-OWNED KV (lane/pp2-spec 2026-08-06) — see `new_session`. Door shut = `Cache::new`.
        let mut cache = crate::pp::new_cache_planned(e, &self.cfg, &self.plan, t_total + k + 8)?;
        let mut scratch = self.new_mtp_scratch(e, t_total + k + 8)?;
        let (embd_qt, embd_rb) = self.embd.qt_and_row_bytes(n_embd);
        let embd_gpu = if spec_host_embd() {
            None
        } else {
            Some(
                self.embd_gpu
                    .get_or_init(|| e.upload_u8(&self.embd.raw).expect("embed table upload")),
            )
        };
        let embd_dev = embd_gpu.map(|g| (g, embd_qt, embd_rb));

        // bg[i] = the trunk's greedy pick for position i under the forced context (i >= 1).
        let mut bg: Vec<u32> = vec![0; t_total + 1];
        let mut rows: Vec<(usize, Vec<u32>, Vec<u32>)> = Vec::new();
        let mut prev_last_h = e.zeros(n_embd)?; // predecessor hidden entering the chunk
        let mut seed_buf = e.zeros(n_embd)?;
        let mut preds_d = e.alloc_u32_zeroed(chunk)?;
        let nll_on = std::env::var("MEMRA_REPLAY_NLL").as_deref() == Ok("1");
        let (mut nll_sum, mut nll_cnt) = (0f64, 0u64);
        let mut s = 0usize;
        while s < t_total {
            let cend = (s + chunk).min(t_total);
            let tc = cend - s;
            let ch = &tokens[s..cend];
            // 1. forced trunk pass — verify path (decode-exact contract): all-column logits +
            //    the chunk's true hiddens.
            let (tl_d, vx) = self.decode_step_t_core(e, ch, s, &mut cache, embd_dev, None)?;
            for j in 0..tc {
                e.argmax_token_device_col(&tl_d, j, n_vocab, &mut preds_d, j)?;
            }
            let preds = e.dtoh_u32(&preds_d)?;
            for j in 0..tc {
                bg[s + j + 1] = preds[j];
            }
            // MEMRA_REPLAY_NLL=1: teacher-forced NLL/perplexity over the same forced pass — the
            // checkpoint-quality metric (position j's logits score the GOLD next token).
            if nll_on {
                let jmax = if cend < t_total { tc } else { tc - 1 }; // last pos has no gold next
                if jmax > 0 {
                    let ids: Vec<u32> = (0..jmax).map(|j| tokens[s + j + 1]).collect();
                    let rows: Vec<i32> = (0..jmax as i32).collect();
                    let idsd = e.htod_u32_v(&ids)?;
                    let rowsd = e.htod_i32(&rows)?;
                    let mut outd = e.zeros(jmax)?;
                    e.softmax_gather(&tl_d, n_vocab, &idsd, &rowsd, &mut outd, n_vocab, jmax, 1.0)?;
                    for pr in e.dtoh(&outd)? {
                        nll_sum += -((pr.max(1e-30)) as f64).ln();
                        nll_cnt += 1;
                    }
                }
            }
            if let Some(f) = hdump.as_deref_mut() {
                use std::io::Write;
                let host: Vec<f32> = e.dtoh(&vx)?;
                // bf16 round-to-nearest-even — f32 doubled the disk bill at bulk
                // extraction scale (20M tokens x 4096 = 320GB f32 vs 160GB bf16).
                let mut bytes = Vec::with_capacity(tc * n_embd * 2);
                for v in &host[..tc * n_embd] {
                    let b = v.to_bits();
                    let r = b.wrapping_add(0x7FFF + ((b >> 16) & 1));
                    bytes.extend_from_slice(&((r >> 16) as u16).to_le_bytes());
                }
                f.write_all(&bytes)?;
            }
            // CHAINLESS extraction (stride > corpus, the bulk-hdump mode): no chunk ever
            // drafts, so the draft-KV fills are pure waste — skip them (2 MTP-block passes
            // per token saved; the forced trunk pass + hdump is all the mode needs).
            let chainless = stride > t_total;
            if chainless {
                e.copy_view_into(
                    &mut prev_last_h,
                    0,
                    &vx.slice((tc - 1) * n_embd..tc * n_embd),
                    n_embd,
                )?;
                s = cend;
                continue;
            }
            // 2. TRUE predecessor-paired draft-KV fill for the chunk (row i carries h_{i-1};
            //    row s reads the previous chunk's last true hidden, zeros at corpus start).
            let mut vxs = e.zeros(tc * n_embd)?;
            e.copy_into(&mut vxs, 0, &prev_last_h, n_embd)?;
            if tc > 1 {
                e.copy_view_into(
                    &mut vxs,
                    n_embd,
                    &vx.slice(0..(tc - 1) * n_embd),
                    (tc - 1) * n_embd,
                )?;
            }
            scratch.set_len(e, s)?;
            self.mtp_kv_fill_all(e, ch, &vxs, s, &mut scratch, embd_dev)?;
            // 3. draft chains at sampled positions, DESCENDING: a chain reads only slots
            //    [0..p) (true fills) and appends at >= p; the next (smaller-p) chain's set_len
            //    truncates those approximate appends before they can ever be read.
            let ps: Vec<usize> = (s..cend)
                .filter(|p| *p >= 1 && *p % stride == 0 && *p + k <= t_total)
                .collect();
            for &p in ps.iter().rev() {
                scratch.set_len(e, p)?;
                if p == s {
                    e.copy_into(&mut seed_buf, 0, &prev_last_h, n_embd)?;
                } else {
                    e.copy_view_into(
                        &mut seed_buf,
                        0,
                        &vx.slice((p - 1 - s) * n_embd..(p - s) * n_embd),
                        n_embd,
                    )?;
                }
                let mut e_tok = tokens[p];
                let mut d_seed = e.clone_dtod(&seed_buf)?;
                let chain_heads = !self.mtp_extra.is_empty();
                let mut chain_tokens = if chain_heads {
                    vec![tokens[p]]
                } else {
                    Vec::new()
                };
                let mut chain_seeds = if chain_heads {
                    vec![e.clone_dtod(&seed_buf)?]
                } else {
                    Vec::new()
                };
                let mut drafts: Vec<u32> = Vec::with_capacity(k);
                for j in 0..k {
                    let (dl_d, h_nextn) = if chain_heads {
                        self.mtp_chain_forward_dev(
                            e,
                            &chain_tokens,
                            &chain_seeds,
                            &mut scratch,
                            p,
                            embd_dev,
                            None,
                        )?
                    } else {
                        self.mtp_head_forward_dev(
                            e,
                            mtp,
                            e_tok,
                            &d_seed,
                            &mut scratch,
                            p + 1 + j,
                            embd_dev,
                            None,
                        )?
                    };
                    let tok_d = e.argmax_token_device(&dl_d, d_vocab)?;
                    let idx = e.dtoh_u32_one(&tok_d)?;
                    let d = match &mtp.d2t {
                        Some(map) => map[idx as usize],
                        None => idx,
                    };
                    drafts.push(d);
                    if chain_heads {
                        chain_tokens.push(d);
                        chain_seeds.push(h_nextn);
                    } else {
                        e_tok = d;
                        d_seed = h_nextn;
                    }
                }
                // targets may live in a LATER chunk's bg — resolved after the walk.
                rows.push((p, drafts, Vec::new()));
            }
            // 4. restore TRUE entries for the whole chunk (the next chunk's chains and fills
            //    expect scratch.len == cend with exact rows).
            scratch.set_len(e, s)?;
            self.mtp_kv_fill_all(e, ch, &vxs, s, &mut scratch, embd_dev)?;
            e.copy_view_into(
                &mut prev_last_h,
                0,
                &vx.slice((tc - 1) * n_embd..tc * n_embd),
                n_embd,
            )?;
            s = cend;
        }
        for (p, drafts, targets) in rows.iter_mut() {
            for j in 0..drafts.len() {
                targets.push(bg[*p + 1 + j]);
            }
        }
        rows.sort_by_key(|r| r.0);
        if nll_cnt > 0 {
            let mean = nll_sum / nll_cnt as f64;
            println!(
                "[replay-nll] tokens={nll_cnt} nll/token={mean:.5} ppl={:.4}",
                mean.exp()
            );
        }
        Ok((rows, bg))
    }
}

#[cfg(test)]
mod vg_debt_tests {
    use super::dspark_vg_debt_projection;

    /// TOOTH for the verify-graph admission accounting: the pool's projected remaining
    /// growth must be charged (pre-fix, admission charged 0 for a pool measured at
    /// 8,852 MiB), the projection must price the MARGINAL cost of one more key rather than
    /// extrapolating the pool's one-time shared allocation, and the doors that make growth
    /// impossible must zero the debt.
    #[test]
    fn vg_debt_projects_remaining_growth_and_respects_the_freeze_valves() {
        const MIB: usize = 1 << 20;
        let d = dspark_vg_debt_projection;
        // cold pool: nothing observed, one capture fits inside SPEC_SHRINK_RESERVE.
        assert_eq!(d(0, 256, 0, None), 0);
        // freeze valve MEMRA_DSPARK_VG_MAX=0: the pool cannot grow.
        assert_eq!(d(10, 0, 500 * MIB, None), 0);
        // saturated pool: at/past the cap the pool FREEZES, nothing left to reserve.
        assert_eq!(d(256, 256, 8852 * MIB, None), 0);
        assert_eq!(d(300, 256, 8852 * MIB, None), 0);

        // BOOTSTRAP (one observation, growth unmeasurable): at most one more pool's worth.
        // The pre-fix mean rule extrapolated 255x here — the measured 8.5 GB phantom.
        assert_eq!(d(1, 256, 33 * MIB, None), 33 * MIB);

        // MARGINAL, flat pool (the box9 receipt: reserved stayed ~33.6 MiB across captures
        // 1..3, so an additional key costs ~nothing and the debt must collapse to ~0 —
        // NOT the 8,556/4,261/2,830 MB the mean rule printed).
        assert_eq!(d(3, 256, 33 * MIB, Some((1, 33 * MIB))), 0);

        // MARGINAL, genuinely growing pool: 40 MiB per new key over 2 keys, 250 slots left.
        let debt = d(6, 256, 273 * MIB, Some((4, 193 * MIB)));
        assert_eq!(debt, 250 * (40 * MIB));
        assert!(
            debt > 3 * (1536 * MIB),
            "real growth must dwarf SPEC_SHRINK_RESERVE"
        );

        // a shrinking/recycled reading never becomes a negative charge.
        assert_eq!(d(6, 256, 10 * MIB, Some((4, 99 * MIB))), 0);
        // a stale observation at the same capture count falls back to bootstrap.
        assert_eq!(d(4, 256, 80 * MIB, Some((4, 80 * MIB))), 80 * MIB);
    }
}

#[cfg(test)]
mod capture_headroom_tests {
    use super::{
        CAPTURE_HEADROOM_FLOOR, capture_err_is_oom, capture_headroom_verdict,
        draft_capture_bootstrap_estimate,
    };

    /// TOOTH for the pre-capture reserve check (lane/step37-vram-admission-20260830): a
    /// capture attempt must be refused BEFORE it allocates when the device cannot cover its
    /// appetite plus the post-capture floor — and pool-cached bytes count as headroom
    /// (driver `free` alone under-counts, the wrong direction for a gate that drops
    /// coverage).
    #[test]
    fn capture_reserve_check_refuses_short_devices_and_counts_pool_cache() {
        const MIB: usize = 1 << 20;
        let need = 900 * MIB;
        // Plenty of room: no refusal.
        assert_eq!(
            capture_headroom_verdict(8_000 * MIB, 0, need, CAPTURE_HEADROOM_FLOOR),
            None
        );
        // The owner's shape: capture appetite would walk the card to the edge — refused,
        // with the arithmetic surfaced for the WARN line.
        let (required, effective) =
            capture_headroom_verdict(1_200 * MIB, 0, need, CAPTURE_HEADROOM_FLOOR)
                .expect("short device must refuse");
        assert_eq!(required, need + CAPTURE_HEADROOM_FLOOR);
        assert_eq!(effective, 1_200 * MIB);
        // Pool-cached bytes are real headroom (the trim path makes them driver-visible).
        assert_eq!(
            capture_headroom_verdict(1_200 * MIB, 7_000 * MIB, need, CAPTURE_HEADROOM_FLOOR),
            None
        );
        // Boundary: exactly enough is enough (>=, never a fencepost refusal).
        assert_eq!(
            capture_headroom_verdict(
                need + CAPTURE_HEADROOM_FLOOR,
                0,
                need,
                CAPTURE_HEADROOM_FLOOR
            ),
            None
        );
        // POLICY at the call site (owner-shape receipts, escalated twice on-box): the
        // refusal fn is handed 2x the appetite plus TWO floors — a capture may take at
        // most half the discretionary headroom, so the card retains a whole capture's
        // worth of room after it lands. One floor of slack above one appetite (the shape
        // that step-OOM'd on the owner cell) must therefore REFUSE under the call-site
        // requirement.
        assert!(
            capture_headroom_verdict(
                need + CAPTURE_HEADROOM_FLOOR + (100 << 20),
                0,
                2 * need,
                CAPTURE_HEADROOM_FLOOR * 2
            )
            .is_some()
        );
    }

    #[test]
    fn bootstrap_estimate_scales_with_heads_and_never_underflows() {
        // 3-head chain on a step37-shaped vocab must expect strictly more than one head.
        let one = draft_capture_bootstrap_estimate(1, 3, 128_896, 4_096);
        let three = draft_capture_bootstrap_estimate(3, 3, 128_896, 4_096);
        assert!(three > one);
        // Degenerate shapes keep a sane minimum (the estimate feeds a refusal gate; a
        // zero-need gate refuses nothing).
        assert!(draft_capture_bootstrap_estimate(0, 0, 0, 0) >= 64 << 20);
    }

    #[test]
    fn capture_oom_predicate_matches_the_quoted_driver_text() {
        assert!(capture_err_is_oom(
            "DriverError(CUDA_ERROR_OUT_OF_MEMORY, \"out of memory\")"
        ));
        assert!(capture_err_is_oom("allocation failed: out of memory"));
        assert!(!capture_err_is_oom("capture produced no graph"));
    }
}

#[cfg(test)]
mod mtp_chain_tests {
    use super::mtp_chain_head_index;

    #[test]
    fn embedded_step_heads_cycle_in_declared_order() {
        let actual: Vec<usize> = (0..8).map(|step| mtp_chain_head_index(step, 3)).collect();
        assert_eq!(actual, [0, 1, 2, 0, 1, 2, 0, 1]);
    }

    #[test]
    fn standalone_draft_remains_single_head() {
        assert!((0..8).all(|step| mtp_chain_head_index(step, 1) == 0));
    }
}

#[cfg(test)]
mod tp_verified_prefix_tests {
    use super::validate_tp_kv_snapshot_shape;
    use crate::tp::ResidentTpKvCache;

    #[test]
    fn snapshot_shape_accepts_matching_tp_presence() {
        let layers = vec![
            Some(ResidentTpKvCache::new(Vec::new(), 1, 1, 1, 1, 8)),
            None,
        ];
        validate_tp_kv_snapshot_shape(&layers, &[Some(2), None]).unwrap();
    }

    #[test]
    fn snapshot_shape_rejects_changed_tp_presence() {
        let layers = vec![Some(ResidentTpKvCache::new(Vec::new(), 1, 1, 1, 1, 8))];
        let error = validate_tp_kv_snapshot_shape(&layers, &[None])
            .unwrap_err()
            .to_string();
        assert!(error.contains("changed shape"), "unexpected error: {error}");
    }
}

#[cfg(test)]
mod dspark_sparse_tests {
    use super::dspark_sparse_softmax_topk;

    #[test]
    fn topk_keeps_full_softmax_mass_and_stable_ties() {
        let logits = [1.0f32, 3.0, 3.0, -2.0];
        let (ids, top_logits, probs, tail) = dspark_sparse_softmax_topk(&logits, 2, 1.0).unwrap();
        assert_eq!(ids, vec![1, 2]);
        assert_eq!(top_logits, vec![3.0, 3.0]);
        let denominator = logits.iter().map(|value| (value - 3.0).exp()).sum::<f32>();
        let expected = 1.0 / denominator;
        assert!((probs[0] - expected).abs() < 1.0e-6);
        assert!((probs[1] - expected).abs() < 1.0e-6);
        assert!((tail - (1.0 - 2.0 * expected)).abs() < 1.0e-6);
        assert!((probs.iter().sum::<f32>() + tail - 1.0).abs() < 1.0e-6);
    }
}

#[cfg(test)]
mod spec_replay_env_tests {
    use super::spec_replay_env_on;

    #[test]
    fn replay_requires_literal_one() {
        assert!(!spec_replay_env_on(None));
        assert!(!spec_replay_env_on(Some("")));
        assert!(!spec_replay_env_on(Some("0")));
        assert!(!spec_replay_env_on(Some("true")));
        assert!(!spec_replay_env_on(Some("2")));
        assert!(spec_replay_env_on(Some("1")));
    }
}

#[cfg(test)]
mod telem_tests {
    use super::{SPEC_TELEM_POS, SpecTelemetry, SpecTelemetryCounters};

    #[test]
    fn synthetic_accept_masks_produce_tau_and_position_histogram() {
        let counters = SpecTelemetryCounters::default();
        for mask in [
            [true, true, true],
            [true, true, false],
            [true, false, false],
            [false, false, false],
        ] {
            let accepted = mask.iter().take_while(|&&value| value).count();
            counters.record_round(mask.len(), accepted);
        }

        let snapshot = counters.snapshot();
        assert_eq!(
            (snapshot.rounds, snapshot.drafted, snapshot.accepted),
            (4, 12, 6)
        );
        assert_eq!(&snapshot.pos_drafted[..3], &[4, 4, 4]);
        assert_eq!(&snapshot.pos_accepted[..3], &[3, 2, 1]);
        assert_eq!(snapshot.tau(), 1.5);
        assert_eq!(snapshot.pos_drafted[3..], [0; SPEC_TELEM_POS - 3]);
        assert_eq!(snapshot.pos_accepted[3..], [0; SPEC_TELEM_POS - 3]);
    }

    /// The worker's per-burst pattern: stash, accumulate, diff — the delta must isolate
    /// exactly the burst's contribution (pool-resumed sessions carry prior requests' counts).
    #[test]
    fn delta_isolates_burst_contribution() {
        let mut t = SpecTelemetry::default();
        // "previous request": 2 rounds of k=3, accepts 3 then 1.
        for (kr, na) in [(3usize, 3usize), (3, 1)] {
            t.rounds += 1;
            t.drafted += kr as u64;
            t.accepted += na as u64;
            for j in 0..kr {
                t.pos_drafted[j] += 1;
            }
            for j in 0..na {
                t.pos_accepted[j] += 1;
            }
        }
        let before = t;
        // "this burst": 1 round k=3, accepts 2.
        t.rounds += 1;
        t.drafted += 3;
        t.accepted += 2;
        for j in 0..3 {
            t.pos_drafted[j] += 1;
        }
        for j in 0..2 {
            t.pos_accepted[j] += 1;
        }
        let d = t.delta_since(&before);
        assert_eq!((d.rounds, d.drafted, d.accepted), (1, 3, 2));
        assert_eq!(&d.pos_drafted[..3], &[1, 1, 1]);
        assert_eq!(&d.pos_accepted[..3], &[1, 1, 0]);
        assert_eq!(d.pos_drafted[3..], [0; SPEC_TELEM_POS - 3]);
    }

    /// merge(delta) then merge(delta2) equals accumulating both — the per-model /metrics
    /// aggregation invariant.
    #[test]
    fn merge_accumulates_fieldwise() {
        let mut agg = SpecTelemetry::default();
        let mut d1 = SpecTelemetry {
            rounds: 2,
            drafted: 6,
            accepted: 4,
            ..Default::default()
        };
        d1.pos_drafted[0] = 2;
        d1.pos_accepted[0] = 2;
        let mut d2 = SpecTelemetry {
            rounds: 1,
            drafted: 3,
            accepted: 1,
            ..Default::default()
        };
        d2.pos_drafted[0] = 1;
        d2.pos_accepted[0] = 1;
        d2.pos_drafted[1] = 1;
        agg.merge(&d1);
        agg.merge(&d2);
        assert_eq!((agg.rounds, agg.drafted, agg.accepted), (3, 9, 5));
        assert_eq!(agg.pos_drafted[0], 3);
        assert_eq!(agg.pos_accepted[0], 3);
        assert_eq!(agg.pos_drafted[1], 1);
        assert_eq!(agg.pos_accepted[1], 0);
    }

    /// Wrong-snapshot diff saturates to zero instead of wrapping — the counters feed a
    /// public metrics surface and must never publish a u64-wrapped garbage value.
    #[test]
    fn delta_saturates_never_wraps() {
        let small = SpecTelemetry {
            rounds: 1,
            drafted: 2,
            accepted: 1,
            ..Default::default()
        };
        let big = SpecTelemetry {
            rounds: 5,
            drafted: 15,
            accepted: 9,
            ..Default::default()
        };
        let d = small.delta_since(&big);
        assert_eq!((d.rounds, d.drafted, d.accepted), (0, 0, 0));
    }
}

#[cfg(test)]
mod opti_fork_tests {
    use super::{
        OptiControllerPolicy, OptiForkAction, OptiForkGateMode, OptiForkGenerationTracker,
    };

    #[test]
    fn controller_threshold_and_three_miss_breaker_are_exact() {
        let mut policy = OptiControllerPolicy {
            threshold: 0.7,
            consecutive_misses: 0,
            breaker_tripped: false,
        };
        assert!(!policy.admit(0.699_999));
        assert!(policy.admit(0.7));
        assert!(!policy.resolve(false));
        assert!(!policy.resolve(false));
        assert!(policy.resolve(false));
        assert!(policy.breaker_tripped);
        assert!(!policy.admit(1.0));
        assert!(
            !policy.resolve(true),
            "a resolved hit cannot re-arm a tripped request"
        );
        assert!(policy.breaker_tripped);
    }

    #[test]
    fn zero_threshold_is_the_true_unconditional_measurement_arm() {
        let mut policy = OptiControllerPolicy {
            threshold: 0.0,
            consecutive_misses: 0,
            breaker_tripped: false,
        };
        for _ in 0..16 {
            assert!(policy.admit(0.0));
            assert!(!policy.resolve(false));
        }
        for invalid in [f32::NAN, f32::INFINITY, -0.01, 1.01] {
            assert!(
                !policy.admit(invalid),
                "invalid q proxy must fail closed: {invalid}"
            );
        }
        assert!(!policy.breaker_tripped);
        assert_eq!(policy.consecutive_misses, 0);
    }

    #[test]
    fn alternating_mode_flips_by_generation_not_round_parity() {
        assert_eq!(OptiForkGateMode::Alternate.action(0), OptiForkAction::Hit);
        assert_eq!(OptiForkGateMode::Alternate.action(1), OptiForkAction::Miss);
        assert_eq!(OptiForkGateMode::Alternate.action(8), OptiForkAction::Hit);
        assert_eq!(OptiForkGateMode::Alternate.action(9), OptiForkAction::Miss);
    }

    #[test]
    fn live_generation_cannot_be_overwritten() {
        let mut tracker = OptiForkGenerationTracker::default();
        let g0 = tracker.reserve().unwrap();
        let g1 = tracker.reserve().unwrap();
        let err = tracker.reserve().unwrap_err().to_string();
        assert!(
            err.contains("still owns generation 0"),
            "unexpected error: {err}"
        );
        tracker.retire(g0).unwrap();
        let g2 = tracker.reserve().unwrap();
        assert_eq!((g2.id, g2.slot), (2, 0));
        tracker.retire(g1).unwrap();
        tracker.retire(g2).unwrap();
    }

    #[test]
    fn teardown_rejects_a_stale_generation_tag() {
        let mut tracker = OptiForkGenerationTracker::default();
        let g0 = tracker.reserve().unwrap();
        tracker.retire(g0).unwrap();
        let err = tracker.retire(g0).unwrap_err().to_string();
        assert!(err.contains("teardown mismatch"), "unexpected error: {err}");
    }
}

#[cfg(test)]
mod draft_graph_fallback_tests {
    use super::DraftGraphFallback;

    /// The Q2 contract, part (a): a fallback flip is LOUD — exactly once per flip.
    #[test]
    fn flip_is_loud_once_and_memoized_after() {
        let mut f = DraftGraphFallback::default();
        let line = f
            .mark_greedy("out of memory")
            .expect("first flip must return the warn line");
        assert!(
            line.contains("WARN"),
            "flip line must be warn-level: {line}"
        );
        assert!(
            line.contains("out of memory"),
            "flip line must carry the reason: {line}"
        );
        assert!(f.greedy_failed());
        // re-marking an already-failed graph is the memoization: quiet, still failed.
        assert!(f.mark_greedy("out of memory").is_none());
        assert!(f.greedy_failed());
        // the two graphs' flags are independent (greedy flip leaves sampled capturable).
        assert!(!f.sampled_failed());
        let line_s = f
            .mark_sampled("capture unsupported")
            .expect("sampled flip is its own flip");
        assert!(
            line_s.contains("sampled"),
            "sampled flip names itself: {line_s}"
        );
        assert!(f.mark_sampled("capture unsupported").is_none());
    }

    /// The Q2 contract, part (b): resume-from-pool RESETS both flags (fresh capture chance),
    /// and says so exactly when there was something to reset.
    #[test]
    fn reset_on_resume_clears_flags_and_logs_once() {
        let mut f = DraftGraphFallback::default();
        // clean session: resume is silent, nothing to reset.
        assert!(f.reset_on_resume().is_none());
        f.mark_greedy("oom").unwrap();
        f.mark_sampled("oom").unwrap();
        let note = f
            .reset_on_resume()
            .expect("a set flag must produce the reset note");
        assert!(
            note.contains("greedy+sampled"),
            "note names what was reset: {note}"
        );
        assert!(
            !f.greedy_failed() && !f.sampled_failed(),
            "both flags cleared"
        );
        // and the NEXT failure after a reset is a fresh flip — loud again.
        assert!(f.mark_greedy("oom again").is_some());
        let note2 = f.reset_on_resume().expect("greedy-only reset");
        assert!(note2.contains("(greedy)"), "single-flag note: {note2}");
    }

    /// Shape-change clears (dmask realloc / mask-shape mismatch / s_key change) stay silent —
    /// they precede a fresh capture attempt whose own failure re-flips loudly.
    #[test]
    fn shape_change_clears_are_silent() {
        let mut f = DraftGraphFallback::default();
        f.mark_greedy("oom").unwrap();
        f.clear_greedy();
        assert!(!f.greedy_failed());
        f.mark_sampled("oom").unwrap();
        f.clear_sampled();
        assert!(!f.sampled_failed());
        // after a silent clear there is nothing left for resume to report.
        assert!(f.reset_on_resume().is_none());
    }
}

/// SAMPLED DRAFT-GRAPH KEY (lane/graph-s-key-exactness-20260819).
///
/// These are the CPU teeth for an exactness bug whose live reproduction needs a GPU, a trunk, a
/// drafter and a two-turn session: the key itself. Every test below fails against the pre-fix key
/// `(seed, temp.to_bits(), k)` — `legacy_key` restates it so the collision is explicit rather
/// than remembered.
#[cfg(test)]
mod sampled_graph_key_tests {
    use super::{SampledGraphKey, debug_t_pred0};

    /// The pre-fix key, verbatim: `let s_key = (sp_seed, sp_temp.to_bits(), k);`
    fn legacy_key(k: &SampledGraphKey) -> (u64, u32, usize) {
        (k.seed, k.temp_bits, k.k)
    }

    fn pure_temp_key() -> SampledGraphKey {
        // temperature 1.0, filters off — today's serve default, the shape that parks a graph.
        SampledGraphKey::new(12345, 1.0, 3, 0, 1.0, 0.0, false)
    }

    /// THE COLLISION. Two requests that differ ONLY in the truncation filters shared one key, so
    /// a parked pure-temp graph survived into a filtered request and the launch site launched it.
    #[test]
    fn vendor_filters_change_the_key() {
        let parked = pure_temp_key();
        // qwen3.8 generation_config.json — what the vendor-default flip makes the default shape.
        let vendor = SampledGraphKey::new(12345, 1.0, 3, 20, 0.95, 0.0, false);
        assert_eq!(
            legacy_key(&parked),
            legacy_key(&vendor),
            "pre-fix key collided: this is the bug, and the reason a test asserts on it",
        );
        assert_ne!(parked, vendor, "post-fix key must separate the two regimes");
        assert!(parked.pure_temp());
        assert!(!vendor.pure_temp());
    }

    /// Each distribution-shaping field alone is enough to drop the parked graph.
    #[test]
    fn every_filter_field_is_keyed() {
        let base = pure_temp_key();
        for (what, other) in [
            (
                "top_k",
                SampledGraphKey::new(12345, 1.0, 3, 20, 1.0, 0.0, false),
            ),
            (
                "top_p",
                SampledGraphKey::new(12345, 1.0, 3, 0, 0.95, 0.0, false),
            ),
            (
                "min_p",
                SampledGraphKey::new(12345, 1.0, 3, 0, 1.0, 0.05, false),
            ),
            (
                "penalties",
                SampledGraphKey::new(12345, 1.0, 3, 0, 1.0, 0.0, true),
            ),
        ] {
            assert_ne!(base, other, "{what} must be part of the key");
            assert!(!other.pure_temp(), "{what} leaves the pure-temp regime");
            assert_eq!(
                legacy_key(&base),
                legacy_key(&other),
                "{what} was invisible to the pre-fix key",
            );
        }
    }

    /// The baked constants stay keyed (this half was always right — regression cover for it).
    #[test]
    fn baked_constants_stay_keyed() {
        let base = pure_temp_key();
        assert_ne!(
            base,
            SampledGraphKey::new(999, 1.0, 3, 0, 1.0, 0.0, false),
            "seed"
        );
        assert_ne!(
            base,
            SampledGraphKey::new(12345, 0.7, 3, 0, 1.0, 0.0, false),
            "temp"
        );
        assert_ne!(
            base,
            SampledGraphKey::new(12345, 1.0, 4, 0, 1.0, 0.0, false),
            "k"
        );
        // bitwise on temperature: 0.7f32 vs the same value re-derived must NOT differ.
        assert_eq!(
            SampledGraphKey::new(1, 0.7, 3, 0, 1.0, 0.0, false),
            SampledGraphKey::new(1, 7.0 / 10.0, 3, 0, 1.0, 0.0, false),
        );
    }

    /// THE LOAD-BEARING HALF OF THE SEED DECISION (lane/session-resume-sampler-predicate-
    /// 20260820). The whole-session resume predicate deliberately does NOT compare `seed`: an
    /// omitted serve `seed` draws fresh per-request entropy, so comparing it would refuse every
    /// seed-omitting sampled conversation. That is only sound because the one piece of parked state
    /// that BAKES the seed — this graph — is re-keyed on it, so a seed change drops and recaptures.
    ///
    /// This test is the other end of that argument, asserted here rather than remembered in a
    /// comment: if a future change dropped `seed` from the key, the resume predicate's exclusion
    /// would silently become the unsound thing it is documented not to be.
    /// (Paired with `seed_alone_does_not_refuse` in `memra-sampling`.)
    #[test]
    fn seed_alone_still_rekeys_the_draft_graph() {
        let parked = pure_temp_key();
        let reseeded = SampledGraphKey::new(999, 1.0, 3, 0, 1.0, 0.0, false);
        assert_ne!(
            parked, reseeded,
            "a seed-only change MUST drop the parked sampled graph — the resume predicate's \
             decision not to compare seed rests on exactly this",
        );
        // Same regime on both sides: the drop is a recapture, not a fall to the eager chain
        // because of a filter difference.
        assert!(parked.pure_temp() && reseeded.pure_temp());
    }

    /// `pure_temp()` is the capture guard's predicate, computed from the key so the two cannot
    /// drift. The equality below is the invariant the launch-site guard asserts: identical keys
    /// agree on the regime, so a graph that survives the drop is legal to launch.
    #[test]
    fn equal_keys_agree_on_the_regime() {
        let a = SampledGraphKey::new(7, 0.8, 3, 20, 0.95, 0.0, false);
        let b = SampledGraphKey::new(7, 0.8, 3, 20, 0.95, 0.0, false);
        assert_eq!(a, b);
        assert_eq!(a.pure_temp(), b.pure_temp());
        // top_p slightly above 1.0 (a client sending 1.0 exactly, or an operator default) is
        // still the unfiltered regime, matching the original `sp.top_p >= 1.0` test.
        assert!(SampledGraphKey::new(7, 0.8, 3, 0, 1.0, 0.0, false).pure_temp());
        assert!(SampledGraphKey::new(7, 0.8, 3, 0, 1.5, -1.0, false).pure_temp());
    }

    /// The WIDENED capture regime (lane/step37-draft-graph-serving-20260830): truncation-
    /// filtered shapes are capturable — the filter runs IN-GRAPH (`filter_stats` +
    /// `gumbel_perturb_filtered_ctr`), so the draft draws from the same filtered
    /// distribution the accept test reconstructs. Penalties never are: the per-round
    /// history cannot be baked. The step37 vendor-default shape (temp 0.5 / top_p 0.9) is
    /// exactly the previously-excluded regime this lane exists to capture.
    #[test]
    fn filtered_regimes_are_capturable_penalties_never() {
        let vendor = SampledGraphKey::new(12345, 0.5, 3, 0, 0.9, 0.0, false);
        assert!(!vendor.pure_temp());
        assert!(vendor.filtered());
        assert!(
            vendor.graph_capturable(),
            "the vendor-default filtered shape must be capturable (default door state)",
        );
        assert!(pure_temp_key().graph_capturable());
        assert!(
            !pure_temp_key().filtered(),
            "pure-temp takes the legacy (filterless) capture body",
        );
        let pen = SampledGraphKey::new(12345, 0.5, 3, 0, 0.9, 0.0, true);
        assert!(
            !pen.graph_capturable(),
            "penalty history varies per round and can never be baked into a graph",
        );
    }

    /// MEMRA_DEBUG_SPEC on a SAMPLED spec request past round 0: the print must render without
    /// indexing the empty greedy `preds` vector (it panicked the GPU worker before this lane).
    #[test]
    fn debug_print_survives_the_sampled_arm() {
        // round >= 1 with a pending bonus == base 1, sampled == `preds` empty.
        assert_eq!(debug_t_pred0(true, 1, 4242, &[]), "n/a");
        assert_eq!(debug_t_pred0(true, 2, 4242, &[]), "n/a");
        // round 0 without a pending bonus still reports last_pred, in both arms.
        assert_eq!(debug_t_pred0(true, 0, 4242, &[]), "4242");
        assert_eq!(debug_t_pred0(false, 0, 4242, &[7, 8]), "4242");
        // greedy keeps the real prediction it always printed.
        assert_eq!(debug_t_pred0(false, 1, 4242, &[7, 8]), "7");
        assert_eq!(debug_t_pred0(false, 2, 4242, &[7, 8]), "8");
    }
}
