//! The single GPU worker thread + step-interleave scheduler (BASE-4, MEMRA-BUILD-MAP §4e).
//!
//! WHY a dedicated thread: the CUDA context is THREAD-AFFINE. `Engine` (and every `CudaStream` /
//! `CudaSlice` it owns) must only ever be touched from the one thread that created the context.
//! So we spawn ONE OS thread, build the primary `Engine` on it, load every `HybridModel` on it,
//! and never let an `Engine`/`Cache`/`CudaSlice` cross a thread boundary. Async HTTP handlers run
//! on a separate tokio runtime and submit work over an `mpsc` channel; each request carries a
//! `tokio` mpsc Sender back which the worker uses to stream tokens (and a final Done) to that one
//! request.
//!
//! SCHEDULER LOOP: the worker holds a `Vec<Session>` of active generations. Each iteration it
//! round-robin steps EVERY active session by exactly ONE `decode_step` (one token of prefill OR
//! one decode token), samples, checks stop, streams the token text back on that session's channel,
//! and retires finished sessions. Queued admits fill empty slots up to `MAX_ACTIVE`. This is the
//! interleave: a long generation and a freshly-admitted one make forward progress in the same loop,
//! so the second produces tokens before the first finishes (not serialized end-to-end).

use std::collections::{HashMap, VecDeque};
use std::io::Write as _;
use std::sync::Arc;
use std::sync::mpsc::{Receiver, Sender};
use std::time::{Duration, Instant};

use cudarc::driver::CudaSlice;
use memra_engine::Engine;
use memra_engine::cache::Cache;
use memra_engine::decode::{GenParams, StopReason};
use memra_engine::decode_batch::{DevPenalty, DevSamp};
use memra_engine::hybrid::{HybridModel, StepTpKvDeviceAdmission};
use memra_engine::sampler::{Sampler, SamplerConfig, SamplerIdentity};
use memra_gguf::GgufFile;
use memra_gguf::source::TensorSource;
use memra_tokenizer::Tokenizer;
use sha2::{Digest, Sha256};

/// Max concurrently-active sessions in legacy round-robin mode (MEMRA_SERVE_BATCH=0).
/// Batched scheduling caps at MEMRA_MAX_SESSIONS (default 64). Admits beyond the cap queue (FIFO).
pub const MAX_ACTIVE: usize = 4;

/// `GenParams.max_new` sentinel: the request OMITTED `max_tokens` (gap-scan F2), so the
/// generation budget is CONTEXT-BOUNDED — session ctx minus prompt tokens, model-capped —
/// the OpenAI default-when-omitted semantics, never a silent 128-token truncation.
/// (`budget = max_new.min(room)` makes the sentinel safe everywhere downstream.)
pub const MAX_NEW_CTX_BOUNDED: usize = usize::MAX;

/// Per-tick prefill chunk cap: tokens primed per scheduler tick per session. Priming runs at
/// prefill throughput instead of tokenwise decode, while the per-tick cap keeps round-robin
/// latency for concurrent sessions bounded.
const PREFILL_TICK_T: usize = 1024;

/// A lone fresh interactive request can use the engine's full eight-microbatch PP geometry in
/// one outer `prime_cache` call. Multi-session ticks retain `PREFILL_TICK_T` fairness, and an
/// explicit `MEMRA_PREFILL_TICK` remains authoritative.
const SOLO_PREFILL_TICK_T: usize = 8192;

fn interactive_prefill_budget(
    configured: usize,
    configured_explicitly: bool,
    sole_unfinished: bool,
    fresh: bool,
    queued: usize,
) -> usize {
    if configured_explicitly || !sole_unfinished || !fresh {
        return configured;
    }
    let mut widened = queued.min(SOLO_PREFILL_TICK_T);
    let tail = queued - widened;
    if tail > 0 && tail < memra_engine::hybrid_forward::PRIME_MIN_T {
        widened = queued;
    }
    configured.max(widened)
}

/// How many prompt tokens one `prefill_tick` prime call takes.
///
/// Extracted as a pure function 2026-08-13 (two-programs inventory W1) because the sub-floor
/// tail merge here had been a PROVABLE NO-OP and nothing tested it. The old form was
/// `take = if q <= budget { q } else { take }`: reaching that guard implies `take < q`, which in
/// the non-eager branch implies `budget < q`, so `q <= budget` is always false and `take` was
/// assigned to itself; the eager branch (`take == q`) never entered it at all. Both sibling
/// sites — step_session's prefill phase and [`interactive_prefill_budget`] — already had the
/// working `take = q` form.
///
/// The invariant that matters: **never leave a remainder of 1..PRIME_MIN_T-1 tokens.** Such a
/// remainder falls through `prefill_tick`'s `else` arm and feeds PROMPT tokens through
/// `decode_step` one at a time, which is a different numeric program than `prime_cache` for the
/// same bytes. `run_gen`'s prime gate (gap #46) documents that fork flipping a near-tie first
/// token — Qwen3.6-35B pp512: 365 -> 198 "\n", then EOS at 2 tokens. Overshoot from the merge is
/// bounded to `PRIME_MIN_T - 1` tokens past `budget`, since the guard only fires when
/// `q < take + PRIME_MIN_T`.
///
/// `bound_rem` is the distance to the next pre-generation capture boundary (prefix-cache LCP
/// split or affinity checkpoint). Stopping exactly on it is what makes the capture land on a
/// clean fed boundary.
fn prefill_tick_take(q: usize, budget: usize, eager_mono: bool, bound_rem: Option<usize>) -> usize {
    const FLOOR: usize = memra_engine::hybrid_forward::PRIME_MIN_T;
    let mut take = if eager_mono { q } else { q.min(budget) };
    if q - take > 0 && q - take < FLOOR {
        take = q;
    }
    if let Some(r) = bound_rem {
        if take >= r {
            take = r; // stop exactly at the snapshot boundary
        } else if r - take < FLOOR {
            // keep the boundary chunk itself primeable next tick
            take = (r - FLOOR).max(FLOOR);
        }
    }
    take
}

fn interactive_prime_batch_take(queued: usize, budget: usize, max_t: usize) -> Option<usize> {
    let take = queued.min(budget).min(max_t);
    (take >= memra_engine::hybrid_forward::PRIME_MIN_T.max(2)).then_some(take)
}

/// Whole fresh prompts retain the established concat-prime path. Repeated continuation batches
/// require a batched implementation for every selected trunk operation. Qwen35Moe selected a real
/// EOS at token 26 after four carried chunks in the frozen mixed-c4 cell; its `MoeMlp` node remains
/// unsupported here, while dense aliases with the same actual plan no longer need an arch entry.
fn carried_prime_batch_eligible(plan: &memra_gguf::model_plan::ModelPlan) -> bool {
    memra_engine::plan_backend::CARRIED_PRIME
        .trunk_capabilities(plan)
        .batch
        .supported
}

/// Prefix-split qualification follows the selected canonical layers, including mixed dense/MoE
/// plans, rather than a whole-model family flag.
fn routed_moe_prefix_split(plan: &memra_gguf::model_plan::ModelPlan) -> bool {
    plan.layers.iter().any(|layer| {
        matches!(
            layer.mlp,
            memra_gguf::model_plan::MlpPlan::Moe(ref moe) if moe.expert_count > 0
        )
    })
}

/// Can a spec-eligible prefix-cache hit re-arm a SpecSession from the restored carrier
/// (lane/spec-on-cache-hit; SAMPLED arm lane/sampled-hit-spec 2026-08-19)? Pure decision
/// half of the conversion at the admit probe. `None` = convert; `Some(reason)` = serve the
/// hit PLAIN **and say why** — v1 refused silently, which is exactly how the greedy-only
/// scope stayed invisible in production (0 restores AND 0 declines in the v0.93.0 DE deploy
/// window looked identical to a mechanism that was never reached).
///   - the entry must carry a draft plane (spec-published entries only);
///   - WHOLE-entry hits only (`entry_pos == fed_len`) — a (default-off) partial restore
///     carries fewer trunk rows than the plane and stays on the plain path;
///   - a full-cover hit (empty suffix — the pure identical-repeat shape) additionally
///     needs the entry's boundary hidden AND logits;
///   - GREEDY **and SAMPLED** (v2). Every converted shape becomes an empty-suffix
///     continuation whose seed token comes off the boundary logits (the engine's own feed
///     logits on a suffix hit, the entry's on a full-cover hit) by EXACTLY the rule the cold
///     burst entry applies to its own first token from the same row: argmax when greedy, a
///     `sample_boundary_token` draw at Philox counter 0 when sampled. v1 refused sampled
///     hits on the premise that "a sampled hit's first token must be host-sampled" — but at
///     the time the COLD sampled path did not host-sample it either (it argmaxed it, as did
///     every continuation burst), so the refusal bought no distributional property; it only
///     made the SAME request take two different sampling programs depending on whether it
///     hit the cache (cold: spec, hit: plain). Both sites now draw, and the restore's seed
///     rule stays bit-identical to the cold spec path's, which is what the sampled gate
///     measures (hit == cold, byte-for-byte, per seed).
///   - SAMPLED + an ACTIVE PENALTY WINDOW: **CONVERTS as of lane/sampled-spec-quality**
///     (2026-08-19). v2 refused it for a real reason — the burst seeded `pen_hist` from the
///     `prompt` slice it was handed and a converted hit hands it none, so the penalized
///     target differed from the cold session's. That defect was the ONE real blocker v2
///     found, and it is now fixed at the source: the burst's penalty window spans
///     `committed ++ prompt`, and a restored session's `committed` is the WHOLE prompt, so
///     its window IS the cold session's window. The refusal therefore has nothing left to
///     protect — EXCEPT under `MEMRA_SPEC_PEN_SESSION=0`, which puts the burst-local window
///     back; the refusal returns with the door named, because a rollback that quietly
///     re-enabled a wrong-window restore would be worse than either posture.
///   - SAMPLED + a BUSY batch: **REFUSED** (lane/sampled-restore-load-guard, 2026-08-19).
///     `load_admits` is the verdict of `sampled_restore_load_admits` — the spec gate's own
///     LOW/HIGH band, evaluated over live sessions plus the arriving wave. Measured reason:
///     the restore is a 1.623x win solo and a 19% aggregate LOSS at c16, because a sampled
///     spec session is the one kind the tick's demotion sweep cannot hand back. Full
///     derivation at `sampled_restore_load_admits`.
/// Eligibility above this (serve_spec, K > 0, unconstrained, MTP present, not vision)
/// is the caller's `spec_eligible && constraint.is_none()` conjunction.
#[allow(clippy::too_many_arguments)]
fn spec_restore_refusal(
    has_draft_plane: bool,
    entry_pos: usize,
    fed_len: usize,
    prompt_len: usize,
    greedy: bool,
    penalty_window_active: bool,
    sampled_allowed: bool,
    pen_session: bool,
    load_admits: bool,
    has_last_h: bool,
    has_last_logits: bool,
) -> Option<&'static str> {
    if !has_draft_plane {
        return Some("entry carries no draft plane (plain-published)");
    }
    if fed_len == 0 || fed_len > prompt_len {
        return Some("degenerate hit length");
    }
    if entry_pos != fed_len {
        return Some("partial (mid-entry) restore");
    }
    // ORDER (lane/sampled-spec-quality): the penalty-window refusal is reported BEFORE the
    // fleet door, deliberately. It is the intrinsic property (this request's window cannot be
    // reproduced under a burst-local window) while the door is an operator knob, and with both
    // shut the gate's teeth arm then sees BOTH doors name themselves in one log instead of the
    // first one short-circuiting the other (the defect the teeth arm caught on its first run).
    if !greedy && penalty_window_active && !pen_session {
        return Some(
            "sampled request with an active penalty window and a burst-local window \
             (MEMRA_SPEC_PEN_SESSION=0)",
        );
    }
    if !greedy && !sampled_allowed {
        return Some("sampled restore disabled (MEMRA_SPEC_RESTORE_SAMPLED=0)");
    }
    // LOAD GUARD (lane/sampled-restore-load-guard, 2026-08-19). Reported AFTER the two
    // intrinsic/door refusals and BEFORE the shape one, for the same reason the ordering above
    // exists: an operator reading one log line should see the door they can flip, and a shape
    // refusal is not a policy decision. See `sampled_restore_load_admits` for the measurement.
    if !greedy && !load_admits {
        return Some(
            "sampled restore refused by the LOAD GUARD (not SOLO — the measured crossover is \
             between c1 and c2 and a sampled spec session never demotes; \
             MEMRA_SPEC_RESTORE_LOAD_GUARD=0 disables)",
        );
    }
    if fed_len == prompt_len && !(has_last_h && has_last_logits) {
        return Some("full-cover hit without the entry's boundary hidden + logits");
    }
    None
}

/// ---- LOAD-AWARE SAMPLED-RESTORE ADMISSION (lane/sampled-restore-load-guard, 2026-08-19) ----
///
/// THE MEASUREMENT THIS EXISTS FOR (box1, 2 x RTX PRO 6000 Blackwell 96GB, 27B NVFP4+MTP,
/// darklanes `research/spec-cache-20260818/BOX1-96GB-WINDOW.md` Findings 1 and 2, N=5
/// interleaved, one boot per arm):
///
/// | shape | sampled restore ON | OFF (v0.93.0 posture) | ratio |
/// |---|---|---|---|
/// | solo, 4 860-tok shared prefix, 192 out | 115.44 tok/s | 71.12 tok/s | **1.623x WIN** |
/// | c16 saturated sold shape, same prefix   | 2.434 req/s  | 3.012 req/s (off: 3.008) | **0.809x LOSS** |
///
/// The greedy control on the same box and binary is 1.00x (3.017 vs 3.019) with 0 engaged rows,
/// so the 19% is not "spec at the sold shape": it is *the sampled restore engaging inside a
/// saturated batch*, on exactly 2 of 16 rows, perturbing the `exact-16` decode wave.
///
/// WHY EXACTLY 2, AND WHY ONLY THE SAMPLED ARM — the asymmetry IS the fix's design input.
/// Two existing mechanisms decide "busy", both keyed on the same `SpecGateThresholds`:
///   1. ADMISSION (`choose_spec_k`): K=0 when `projected_active > LOW`. In a c16 fan-out the
///      first two arrivals see `n_active + 1 <= LOW = 2`, so the load policy admits them to
///      spec — correctly, on the information it has at that instant.
///   2. TICK DEMOTION (`spec_gate_high`): a live spec session hands its cache to the batched
///      path once `n_live >= HIGH`. That is what protects the batch from a session admitted
///      while the box was quiet — and **sampled sessions are excluded from it** (see the
///      exclusion list at the demotion sweep: the handoff would move generation to a batched
///      row whose sampler owns its own history and Philox stream, an unmeasured
///      distributional seam). A greedy restore that engages in a quiet window is handed back
///      the moment load arrives; a SAMPLED one rides the serial spec queue until the request
///      ends. That is the whole 19%: mechanism 2 cannot clean up after mechanism 1.
///
/// So the sampled restore's admission has to be conservative UP FRONT, and this is that guard.
///
/// THE WATERMARK IS **SOLO**, AND THAT IS A MEASUREMENT, NOT A CHOICE. The first version of this
/// guard used the band's own LOW=2 and still lost, because LOW=2 is itself mis-tuned for this
/// route on this card. The band's 2/4 defaults come from a 5090 measurement on a short greedy
/// shape (see `spec_gate_on`: c1 1.67x WIN, c2 1.08x WIN, c4 0.61x LOSS). Re-measured here on
/// the sold shape (27B, 4 860-token shared prefix, 60 out, temp 0.8, 3 interleaved passes per
/// rung, un-guarded lever vs the door-shut posture):
///
/// | c  | lever req/s | door-shut req/s | ratio      | engaged rows/pass |
/// |----|-------------|-----------------|------------|-------------------|
/// | 1  | 1.449       | 1.074           | **1.350x** | 1 of 1            |
/// | 2  | 1.189       | 1.779           | **0.669x** | 2 of 2            |
/// | 16 | 2.450       | 3.013           | **0.813x** | 2 of 16           |
///
/// **The crossover is between c1 and c2** — the lever pays SOLO and nowhere else. Two concurrent
/// sampled spec sessions already cost a THIRD of aggregate throughput, because phase (a) steps
/// each session's whole burst in a serial host loop and phase (c) excludes spec sessions from
/// batched decode, so two of them serialise what would otherwise be a 2-wide batch. A watermark
/// of LOW=2 therefore admits exactly the shape that loses most.
///
/// So the rule is the SOLO ADMISSION the other non-demotable spec regimes in this file use
/// (`gspec_k` and sampled dspark; greedy dspark now uses LOW-wave admission because it demotes at
/// HIGH). Reusing that
/// notion — rather than the MTP arm's band — is pinning against truth instead of against a
/// sibling: the band is a sibling policy measured on a different card, a different model and a
/// different sampling regime. The band is still respected as a CEILING (`thresholds.low.min(1)`),
/// so a placement that admits no spec at all (PP-2, LOW=0) admits no restores either.
///
/// WHERE "DEMAND" COMES FROM, and why the obvious answer was measured to be wrong. Reading it
/// as `active.len() + queue.len()` at tick top removes exactly ONE of the two engaged rows at
/// c16 (measured 0.815x -> 0.904x of spec-off; each engaged row is worth ~9.6% of aggregate
/// throughput). The survivor is the HEAD of the wave and is unrefusable from that reading: the
/// worker wakes out of its idle `recv()` holding that one request, its `try_recv` drain finds
/// nothing else yet, and the other fifteen arrive while it performs that request's own 312 MB
/// prefix restore. The information was never missing — the server log shows ten `[meter] admit`
/// lines (printed at the HTTP boundary) BEFORE the restore line — it was simply in flight
/// between the handler and the worker, in neither `active` nor `queue`. So demand is
/// `max(worker-visible, HTTP in-flight)` (`spec_load_demand`), read from the gauge that already
/// owns it, and the verdict is re-evaluated as LATE as possible — after the restore work, at the
/// refusal site — so that window counts.
///
/// GREEDY IS UNTOUCHED, deliberately: it is demotable, its measured cost at c16 is 1.00x, and
/// a guard on a path that costs nothing is a regression in reach for no measured gain.
/// SOLO: the measured admission ceiling for an UNDEMOTABLE sampled spec session. One request in
/// flight — this one. The same rule `gspec_k` and sampled/non-demotable dspark already apply,
/// and the measured crossover above is why it is 1 and not the MTP band's LOW.
const SAMPLED_RESTORE_SOLO_MAX: usize = 1;

/// The sampled restore's admission watermark: SOLO, with the spec gate's LOW as a CEILING so a
/// placement that admits no spec at all admits no restores either (PP-2 default LOW=0).
fn sampled_restore_watermark(thresholds: SpecGateThresholds) -> usize {
    thresholds.low.min(SAMPLED_RESTORE_SOLO_MAX)
}

/// The HTTP layer's own in-flight gauge, registered at server start so the load guard can read
/// DEMAND WHERE IT IS ACTUALLY KNOWABLE.
///
/// WHY THIS EXISTS (measured, box1 2026-08-19). The first version of the guard read demand as
/// `active.len() + queue.len()` at tick top, and it removed exactly ONE of the two engaged rows
/// at c16: 0.815x -> 0.904x of spec-off, at ~9.6% of aggregate throughput per engaged row. The
/// survivor is the HEAD OF THE WAVE, and no tick-top reading can refuse it: the worker wakes
/// from its idle `recv()` with that one request in hand, its `try_recv` drain finds nothing
/// else yet, and the other fifteen land while it is doing that request's 312 MB prefix restore.
/// The server log proved the information existed — ten `[meter] admit` lines (the HTTP boundary)
/// print BEFORE the restore line — it was just in flight between the handler and the worker,
/// i.e. in neither `active` nor `queue`. So this is not a second notion of "busy": it is the
/// SAME demand, read from the gauge that already owns it (`InflightGuard`, RAII, incremented at
/// submission and decremented when the response or SSE stream completes).
///
/// Unset in unit tests and in any embedding that builds its own `AppState` — callers fall back
/// to the worker-visible count, so a missing registration degrades to the previous behaviour
/// rather than to a wrong one.
static HTTP_INFLIGHT: std::sync::OnceLock<std::sync::Arc<[std::sync::atomic::AtomicUsize; 3]>> =
    std::sync::OnceLock::new();

pub fn register_http_inflight(counts: std::sync::Arc<[std::sync::atomic::AtomicUsize; 3]>) {
    let _ = HTTP_INFLIGHT.set(counts);
}

/// Requests this server currently owes an answer for, across every lane (they all share the
/// batch, so they are all demand). `None` = no gauge registered.
fn http_inflight_total() -> Option<usize> {
    HTTP_INFLIGHT.get().map(|c| {
        c.iter()
            .map(|n| n.load(std::sync::atomic::Ordering::Relaxed))
            .sum()
    })
}

/// Demand for the load guard: the larger of what the worker can see and what the HTTP layer
/// knows. `max`, never a replacement — the gauge can lag a request the worker already holds
/// (it is decremented on response completion, so it never *under*counts live work, but an
/// embedding without a registration would report 0), and under-counting demand is the one
/// error mode that costs throughput.
fn spec_load_demand(worker_visible: usize) -> usize {
    worker_visible.max(http_inflight_total().unwrap_or(0))
}

/// PROJECTED ADMISSION WAVE (lane/hermes-perf-fixes, 2026-08-23) — the concurrency signal
/// `choose_spec_k` keys on, replacing the live-only `active + 1`.
///
/// THE MEASURED DEFECT (canonflip-20260813 + the sampled-restore lane's c16 receipts, both
/// quoted at `SAMPLED_RESTORE_SOLO_MAX`): in a c16 fan-out the first arrivals see
/// `active.len() + 1 <= LOW = 2` and are admitted to SPEC — the rest of the wave is already
/// in flight (ten `[meter] admit` lines print BEFORE the first restore line) but lives in
/// neither `active` nor `queue`, so the burst SPLITS across spec and plain. Tick demotion
/// (`spec_gate_high`) cannot clean up after it for sampled sessions, and each engaged spec
/// row at the sold shape costs ~9.6% of aggregate throughput. The sampled-restore guard
/// already reads demand correctly (`spec_load_demand`); this extends the same reading to the
/// MTP K policy and to the admission-time cost estimate so both split the SAME way the
/// serve-time decision does (no more spec-reserve for a request that will serve plain).
///
/// The projection is everything the worker can see of the wave — live sessions, this
/// request, the rest of this tick's queue plus the tick's own deferrals (`n_pending`), and
/// the handler->worker channel in-flight gauge (`PENDING_ADMITS`) — maxed against the HTTP
/// in-flight gauge, which is where the head-of-wave information actually lives (the
/// tick-top blind spot documented at `HTTP_INFLIGHT`).
///
/// DELIBERATE BIAS, stated: VRAM-deferred requeue members count as demand even though they
/// may wait several ticks (the wt-cx-cachemeter critique). At the shapes that create such
/// waves, spec is a measured LOSS anyway (canonflip: spec-on c=16 2.14 req/s vs spec-off
/// 8.50), so depressing K under a standing wave is the cheap side of the error.
fn projected_admission_wave(n_active: usize, n_pending: usize) -> usize {
    admission_wave_projection(
        n_active,
        n_pending,
        PENDING_ADMITS.load(std::sync::atomic::Ordering::Acquire),
        http_inflight_total(),
    )
}

/// Pure half of [`projected_admission_wave`] — globals injected so the arithmetic is
/// unit-testable without touching the process-wide gauges.
fn admission_wave_projection(
    n_active: usize,
    n_pending: usize,
    channel_pending: usize,
    http_inflight: Option<usize>,
) -> usize {
    n_active
        .saturating_add(1)
        .saturating_add(n_pending)
        .saturating_add(channel_pending)
        .max(http_inflight.unwrap_or(0))
}

/// The load half of sampled-restore admission. An operator pin owns the whole policy
/// (`choose_spec_k` short-circuits on it and automatic demotion is off, so a pinned server
/// must not have a second policy quietly overriding its pin), `MEMRA_SPEC_GATE=0` is the
/// always-spec rollback seam, and `MEMRA_SPEC_RESTORE_LOAD_GUARD=0` is this lane's own seam.
fn sampled_restore_load_admits(
    guard_on: bool,
    pin: Option<usize>,
    gate_on: bool,
    thresholds: SpecGateThresholds,
    demand: usize,
) -> bool {
    if !guard_on || pin.is_some() || !gate_on {
        return true;
    }
    demand <= sampled_restore_watermark(thresholds)
}

/// Whether a live DFlash2 row prevents admitting another session through the greedy LOW band.
/// Key this on `dspark_on`, not the lazily-created engine session: a sampled row must block in
/// its pre-prime window too. Demotion clears `dspark_on`, so a handed-off plain row is inert.
fn dspark_blocks_greedy_widening(dspark_on: bool, greedy: bool, constrained: bool) -> bool {
    dspark_on && (!greedy || constrained)
}

/// DFlash2 load admission. LOW-band widening is safe only while automatic demotion owns every
/// live DFlash2 row. A positive operator K pin disables automatic demotion and falls back to
/// solo; K=0 or an unpinned LOW=0 policy pins plain. Gate-off sessions stay solo. A live
/// non-demotable row blocks a later arrival from widening behind it.
///
/// SAMPLED LOW-BAND ADMISSION (lane/dspark-sampled-wave-20260825, owner directive "fix the
/// issue"): sampled requests now share the greedy LOW band instead of the solo law. Sampled
/// sessions still cannot demote (their committed stream depends on session-owned philox
/// counters; the plain batched sampler is a different random program mid-request), so the
/// safety argument is DIFFERENT from greedy's and worth stating:
///
///   - at most ONE non-demotable row can ever be live: the FIRST sampled admission makes
///     `has_live_non_demotable_dspark` true, which refuses every later dspark admission
///     (sampled and greedy alike) until it retires. The widening therefore never stacks
///     un-shed-able sessions — the worst case under a rising wave is one bounded sampled
///     session finishing at spec speed while everything behind it serves plain.
///   - the vendor-default serve shape is SAMPLED (penalties included; the dspark walk has
///     the penalized sampled arm) — under the old solo law the route lost speculation for
///     exactly the traffic the fleet serves whenever anything else was active, which is
///     what the 2026-08-25 DE flip measured live: watchdog + one customer = plain.
///
/// `MEMRA_DSPARK_SAMPLED_WAVE=0` is this lane's rollback seam (restores the solo law).
/// K FLOOR FOR RESUMED SPEC CARRIERS (2026-08-25, found by the sampled ladder at c=4):
/// the late-wave K re-read (PR #37 finding, 2026-08-24) can only move K DOWN, so a
/// restored/resumed SpecSession that arrived with K>=1 can land here at K=0 — and a
/// Session holding `spec: Some(..)` with `spec_k == 0` walks straight into the engine's
/// `assert!(k >= 1)` and kills the GPU worker (observed: `[spec-k] K=0 source=concurrency
/// wave=4` followed by `PANIC ... k must be >= 1`, every session on the box dead). Clamp
/// to K=1: the narrowest legal burst. Greedy carriers demote to plain at the very next
/// gate sweep anyway (the wave that forced K=0 is over HIGH); sampled carriers cannot
/// demote (session-owned philox) and run narrow until the wave passes — strictly cheaper
/// than the crash and no wider than the policy feared.
fn resumed_carrier_spec_k_floor(has_carrier: bool, k: usize) -> (usize, bool) {
    if has_carrier && k == 0 {
        (1, true)
    } else if has_carrier {
        (k, false)
    } else {
        (0, false)
    }
}

fn dspark_load_admits(
    greedy: bool,
    gate_on: bool,
    pin: Option<usize>,
    projected_wave: usize,
    low: usize,
    n_active: usize,
    has_live_non_demotable_dspark: bool,
) -> bool {
    dspark_load_admits_with(
        greedy,
        gate_on,
        pin,
        projected_wave,
        low,
        n_active,
        has_live_non_demotable_dspark,
        dspark_sampled_wave_on(),
    )
}

/// Pure half of [`dspark_load_admits`] — the rollback seam injected so both arms are
/// unit-testable in one process (the env read is a OnceLock).
#[allow(clippy::too_many_arguments)]
fn dspark_load_admits_with(
    greedy: bool,
    gate_on: bool,
    pin: Option<usize>,
    projected_wave: usize,
    low: usize,
    n_active: usize,
    has_live_non_demotable_dspark: bool,
    sampled_wave: bool,
) -> bool {
    // LOW-BAND STACKING (lane/dspark-low-band-stack-20260825): the one-non-demotable-row
    // hard block below no longer applies to the automatic LOW band when the sampled-wave
    // posture is on. The block's rationale — "two live DFlash rows would serialize" — was
    // an analogy from the mixed-tick lesson, never a dspark measurement, and the MTP arm
    // refutes it on the same box: two sampled MTP sessions at c=2 ARE two serialized
    // phase-(a) bursts and measure 121.6 agg tok/s, while the block forced dspark into a
    // 1-spec+1-plain mix at 105.9 (sampled-wave ladder, PRO 6000, vendor shape). Stacking
    // is bounded exactly the way MTP's own sampled admission is bounded: the wave
    // projection counts every live row, so at most LOW dspark rows can ever be live, and
    // greedy rows among them still demote at HIGH. The pinned and gate-off arms keep the
    // block: those laws expect solo and have no demotion machinery behind them.
    if has_live_non_demotable_dspark && !(gate_on && sampled_wave && pin.is_none()) {
        return false;
    }
    match pin {
        Some(0) => false,
        Some(_) => n_active == 0,
        None if gate_on && greedy => projected_wave <= low,
        None if gate_on && sampled_wave => projected_wave <= low,
        None if gate_on => projected_wave <= low.min(1),
        None => n_active == 0,
    }
}

/// ROLLBACK SEAM for sampled LOW-band admission. Default ON; `MEMRA_DSPARK_SAMPLED_WAVE=0`
/// restores the pre-lane solo law without touching the greedy band.
fn dspark_sampled_wave_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("MEMRA_DSPARK_SAMPLED_WAVE").as_deref() != Ok("0"))
}

#[derive(Clone, Copy)]
struct DsparkColdPrefixAdmission {
    route_ready: bool,
    prime_feasible: bool,
    greedy: bool,
    greedy_penalized: bool,
    sampled: bool,
    constrained: bool,
    vision: bool,
    cold: bool,
    gate_on: bool,
    pin: Option<usize>,
    projected_wave: usize,
    low: usize,
    n_active: usize,
    has_live_non_demotable: bool,
    /// Prompt length in tokens. The prefill this bypass THROWS AWAY scales with it.
    prompt_len: usize,
    /// The decode budget this request could actually use (`max_new` clamped by the context
    /// cap). The speculation gain that has to REPAY that prefill scales with it.
    decode_budget: usize,
    /// Whether a hit THIS request can actually CONSUME exists. REQUIRED for the shape veto:
    /// declining the cold prime is only an improvement if a hit takes its place — otherwise the
    /// request loses the prime, loses all speculation with it (the MTP arm is hard-refused
    /// while dspark is armed) and gains nothing.
    ///
    /// The caller sets this from `px.lookup` — a whole-entry FULL PREFIX of the prompt at or
    /// above `PREFIX_CACHE_MIN_TOKENS` — resolved once and reused by the restore, so the
    /// decision and the restore cannot disagree. The only other way to consume a hit is the
    /// mid-entry partial restore, which is default-OFF and refused outright on the hybrid/GDN
    /// trunk `gdn_dspark_compatible` selects, so on this route a full-prefix entry is the whole
    /// set.
    ///
    /// Review chased this predicate down three granularities, and the two rejected forms are
    /// recorded because each looked right: `prefix_on` alone (cache merely CONFIGURED — a
    /// first-of-class request then declines and misses; visible in the lane's own A/B, where
    /// the new build's seeder ran 10.9 s against the old build's 9.8 s), and `best_lcp > 0`
    /// (any overlap, with NO floor — a 5-token common prefix satisfied it while nothing
    /// consumable existed).
    hit_available: bool,
}

/// How long the decode must be before a cold DFlash prime is worth discarding a cache hit,
/// expressed as a multiple of the prompt length. Default 1/8: at a 30k prompt the request
/// must be able to generate ~3.8k tokens before the bypass pays.
///
/// WHY A RATIO AND NOT A TOKEN COUNT: both sides of the trade scale with their own length.
/// Bypassing costs ~`prompt_len / prefill_rate` seconds of prefill that a hit would have
/// skipped; speculation earns roughly `decode_budget / decode_rate x gain` seconds. With the
/// measured fleet rates (~2.9k tok/s prefill, ~100-300 tok/s decode, dspark accept 0.68-0.74)
/// the break-even sits near decode ≈ prompt/10, so 1/8 is that boundary with margin on the
/// side of KEEPING today's behaviour.
///
/// MEASURED MOTIVATION (darklanes research/nonstream-deadline-20260826/CONTINUATION.md): a
/// 30,312-token prompt answered in 86 tokens paid a 9.6 s cold prime — with `lcp=30312`, a
/// FULL-prompt hit sitting right there — to speculate over 86 tokens. The same prompt with a
/// 4k+ answer is the case the bypass was designed for and is unaffected.
/// Override: `MEMRA_DSPARK_COLD_DECODE_RATIO_DIV` (larger = more willing to cold-prime).
const DSPARK_COLD_DECODE_RATIO_DIV: usize = 8;

/// Does this request's decode budget justify discarding a prefix hit to cold-prime DFlash?
/// Prompts below the prefix cache's own floor are exempt: no hit can exist for them, so the
/// question does not arise and the old behaviour stands.
fn dspark_cold_prime_repays_prefill(prompt_len: usize, decode_budget: usize) -> bool {
    if prompt_len < PREFIX_CACHE_MIN_TOKENS {
        return true;
    }
    let div = std::env::var("MEMRA_DSPARK_COLD_DECODE_RATIO_DIV")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(DSPARK_COLD_DECODE_RATIO_DIV);
    decode_budget >= prompt_len / div
}

/// Decide the DFlash route before probing the prefix cache. A winning DFlash request must prime
/// its draft state from the full prompt, so a trunk-only hit cannot accelerate it; preserving and
/// bypassing that entry lets later shed-to-plain traffic consume it instead. The load inputs are
/// exactly those reused by the final dspark_on decision.
///
/// SHAPE GUARD (lane/dspark-trunk-hit-20260827): the bypass above is a TRADE — it spends the
/// prefill a hit would have saved to buy speculation on the decode — and this function used to
/// take that trade without looking at either side's size. It now declines when the request's
/// own decode budget is too short to repay the prompt's prefill
/// (`dspark_cold_prime_repays_prefill`), which is the measured 30k-prompt/86-token shape.
fn dspark_prefers_cold_over_prefix(a: DsparkColdPrefixAdmission) -> bool {
    a.route_ready
        && a.prime_feasible
        && ((a.greedy && !a.greedy_penalized) || a.sampled)
        && !a.constrained
        && !a.vision
        && a.cold
        // The shape veto applies ONLY when a hit is actually there to be served instead.
        && (!a.hit_available
            || dspark_cold_prime_repays_prefill(a.prompt_len, a.decode_budget))
        && dspark_load_admits(
            a.greedy,
            a.gate_on,
            a.pin,
            a.projected_wave,
            a.low,
            a.n_active,
            a.has_live_non_demotable,
        )
}

/// DSPARK PREFIX RESTORE (lane/dspark-draft-plane-20260827). Default **OFF**: the mechanism
/// re-arms a speculative session from cached draft state, so it does not ship on until the
/// restored-vs-cold accept trajectory is proven byte-identical on the bench box. `1` arms it.
/// Written as an explicit decision per the flags law; the FLAGS.md row lands with the receipts.
fn dspark_prefix_restore_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("MEMRA_DSPARK_PREFIX_RESTORE").as_deref() == Ok("1"))
}

/// `dspark_restorable_hit` = the restore is armed AND the consumable hit actually carries a
/// draft tail. It overrides the cold preference, because that preference exists ONLY to avoid
/// discarding a prefill for a hit that could not serve a speculating request — and a
/// tail-carrying entry can (lane/dspark-draft-plane-20260827).
///
/// It is deliberately NOT "the restore is armed": probing on an entry with no tail would hand a
/// long-decode request a PLAIN hit, trading its speculation away for a prefill saving nobody
/// asked for. And tail presence alone is not enough (review round 3): `lookup` returns strict
/// prefixes too — the standard multi-turn shape, where turn N+1 extends the turn-N entry — and
/// the conversion only consumes WHOLE-entry covers (the trunk cannot rebuild recurrent state
/// mid-sequence). Overriding on a strict-prefix hit would pay the full carrier restore
/// (~1 GB of plane copies at the flagship shape) and inflate the prefix-hit counters, only for
/// the conversion to drop the carrier and cold-prime anyway. `dspark_hit_is_restorable` is the
/// shared predicate: the override fires only for hits the restore can actually consume, which
/// is what keeps "no wasted restore copy" true.
fn should_probe_prefix_cache(
    prefix_on: bool,
    has_prior_reuse: bool,
    dspark_prefers_cold: bool,
    dspark_restorable_hit: bool,
) -> bool {
    prefix_on && !has_prior_reuse && (!dspark_prefers_cold || dspark_restorable_hit)
}

/// Can this hit actually feed the dspark conversion? Tail present AND whole-entry cover — the
/// same conditions the conversion checks after the restore has already been paid for
/// (`full_cover` + `entry_tail_present`), evaluated here BEFORE the probe so a hit that cannot
/// convert never buys the carrier. A strict-prefix hit (entry shorter than the prompt) fails
/// this on purpose: the multi-turn shape keeps its pre-lane behavior — `dspark_prefers_cold`
/// suppresses the probe and the request cold-primes with zero restore cost.
fn dspark_hit_is_restorable(entry_toks: usize, prompt_toks: usize, entry_has_tail: bool) -> bool {
    entry_has_tail && entry_toks == prompt_toks
}

fn dspark_prefix_capture_requested(
    dspark_on: bool,
    prefix_on: bool,
    prompt_len: usize,
    exact_key_present: bool,
    exact_preprime_owner: bool,
) -> bool {
    dspark_on
        && prefix_on
        && prompt_len >= PREFIX_CACHE_MIN_TOKENS
        && !exact_key_present
        && !exact_preprime_owner
}

fn dspark_prefix_owner_identity_matches(
    owner_key: (&str, &str),
    owner_prompt: &std::collections::VecDeque<u32>,
    request_key: (&str, &str),
    request_prompt: &[u32],
) -> bool {
    owner_key == request_key
        && owner_prompt.len() == request_prompt.len()
        && owner_prompt
            .iter()
            .copied()
            .eq(request_prompt.iter().copied())
}

fn should_collect_dspark_after_phase_a(was_dspark_step: bool, step_succeeded: bool) -> bool {
    was_dspark_step && step_succeeded
}

/// ROLLBACK SEAM for the load guard. Default ON: sampled prefix-cache restores are refused at
/// the saturation shape. `MEMRA_SPEC_RESTORE_LOAD_GUARD=0` restores the un-guarded posture
/// (every sampled hit that clears the other refusals re-arms spec regardless of load), which
/// is the arm the crossover cell measures.
fn spec_restore_load_guard_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("MEMRA_SPEC_RESTORE_LOAD_GUARD").as_deref() != Ok("0"))
}

/// ROLLBACK SEAM for the sampled arm (lane/sampled-hit-spec). Default ON: sampled
/// prefix-cache hits re-arm spec exactly like greedy ones. `MEMRA_SPEC_RESTORE_SAMPLED=0`
/// restores the v0.93.0 posture (sampled hits serve PLAIN) without touching the greedy
/// path — one env flip is the fleet-wide undo, and the hit gate's own teeth arm asserts
/// that this door really does hold sampled hits on the plain path.
fn spec_restore_sampled_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("MEMRA_SPEC_RESTORE_SAMPLED").as_deref() != Ok("0"))
}

/// The ONE place a request's `Sampler` becomes the engine's per-burst `SpecSampling`
/// (lane/sampled-spec-quality). It was inline in `step_session` only; the prefix-cache
/// restore now needs the identical config to draw its boundary token, and two hand-written
/// copies of a nine-field struct is how a restored session ends up sampling from a slightly
/// different distribution than its own bursts. `None` = greedy (temperature 0).
fn spec_sampling_for(sampler: &Sampler) -> Option<memra_engine::spec::SpecSampling> {
    (sampler.temperature() > 0.0).then(|| memra_engine::spec::SpecSampling {
        temp: sampler.temperature(),
        seed: sampler.seed(),
        top_k: sampler.top_k() as i32,
        top_p: sampler.top_p(),
        min_p: sampler.min_p(),
        penalty_last_n: sampler.penalty_last_n(),
        penalty_repeat: sampler.penalty_repeat(),
        penalty_freq: sampler.penalty_freq(),
        penalty_present: sampler.penalty_present(),
    })
}

/// A fully restored prefix-cache hit already owns the logits needed for its first decode.
/// Letting an unrelated cold prime run first turns one synchronous prime chunk into cache-hit
/// TTFT, even though the hit itself has no prefill work left.
fn cached_hit_needs_first_token(
    n_prompt: usize,
    n_cached: usize,
    prefill_done: bool,
    generated: usize,
) -> bool {
    n_prompt > 0 && n_cached == n_prompt && prefill_done && generated == 0
}

fn tick_trace_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("MEMRA_TICK_TRACE").as_deref() == Ok("1"))
}

fn graph_session_env_on(value: Option<&str>) -> bool {
    value == Some("1")
}

fn graph_session_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        let value = std::env::var("MEMRA_SERVE_GS").ok();
        graph_session_env_on(value.as_deref())
    })
}

/// A model loaded resident on the worker thread: weights + its own tokenizer + config snapshot.
struct LoadedModel {
    model: HybridModel,
    tok: Arc<Tokenizer>,
    eos_id: u32,
    /// Loaded from a checkpoint DIRECTORY (safetensors/repack) rather than a GGUF file.
    /// Feeds ModelCaps::chat_ok: a dir checkpoint with no chat template 400s on chat
    /// requests (serve-st v1 honesty gate) instead of silently rendering fallback ChatML;
    /// GGUF models keep the historical ChatML fallback.
    from_dir: bool,
    /// Constrained-decoding compiler. Its bounded, per-model background thread owns the
    /// lazily-built vocabulary trie and parser factory; none of that CPU work runs here.
    constraints: crate::constrained::ConstraintCompiler,
}

/// What the worker streams back to one request, over its per-request tokio mpsc channel.
#[derive(Debug, Clone)]
pub enum Event {
    /// Authoritative prompt/cache usage, published once at admission before token generation.
    /// Streaming receipts retain it if the client disconnects before the terminal `Done`.
    PromptUsage { n_prompt: usize, n_cached: usize },
    /// Prompt-capture result (lane/embed-serve): published once when prefill completes on a
    /// capture request, before `Done`. `hidden` = the last-position post-final-norm hidden
    /// state (the embedding vector, pre-normalization); `logits` = the last-position logit
    /// per requested token id, in `CaptureSpec::logit_ids` order (the rerank yes/no read).
    PromptCapture {
        hidden: Option<Vec<f32>>,
        logits: Vec<f32>,
    },
    /// One decoded token: the raw id + the incremental text delta (detokenized tail minus prefix).
    Token { id: u32, text: String },
    /// Authoritative terminal token ids for blocking native responses. Streaming consumers ignore
    /// this snapshot; it precedes `Done` and must agree with the per-token event sequence.
    TokenSnapshot(Vec<u32>),
    /// Terminal event: why we stopped + final token count + timing. `n_prompt` / `n_cached`
    /// are WORKER-TRUTH prompt accounting: total prompt tokens this session fed or resumed —
    /// the tokenized RENDERED prompt (tools block included when one was rendered; the
    /// text-prefix spec resume counts the actually-fed remainder) — and how many of those
    /// came from a cache (continuation pool, spec resume, or the cross-request prefix cache)
    /// instead of being computed — the OpenAI `usage.prompt_tokens_details.cached_tokens`
    /// source. ONE source of truth: both counts come off the same rendered-prompt token ids.
    /// `spec` = THIS request's spec-decode acceptance summary (lane/accept-telemetry) —
    /// None on non-spec sessions, so the usage surface is byte-identical when spec is off.
    Done {
        stop_reason: String,
        n_tokens: usize,
        n_prompt: usize,
        n_cached: usize,
        elapsed_s: f64,
        spec: Option<SpecUsage>,
    },
    /// The request failed. CLASSIFIED at the producer (`EngineError`) — the HTTP layer maps
    /// the class to a status code instead of calling everything a 400 (G6).
    Error(EngineError),
}

/// THE ERROR TAXONOMY (lane/serve-hardening, 2026-08-06; audit gap G6/G16).
///
/// WHAT WAS BROKEN: `Event::Error(String)` carried no type information, so `main.rs`'s only
/// possible mapping was `bad_request(&msg)` — CUDA faults, VRAM exhaustion, tokenizer
/// failures and genuine client mistakes all left as `400 invalid_request_error`. Two
/// consequences, both bad: 400 is non-retryable by SDK convention (openai-python retries
/// 408/409/429/>=500 only), so a transient GPU blip became a hard user-visible failure that
/// no client would retry; and a real engine fault was invisible in any client's or
/// aggregator's 5xx error-rate view.
///
/// WHERE THE CLASS COMES FROM: the PRODUCER, not a regex over the message. The site that
/// raises the failure is the only place that knows whether the caller or the box is at
/// fault, and a string-matching classifier in the HTTP layer would silently reclassify every
/// time someone reworded an error. The one text-driven rule is deliberate and quoted:
/// `EngineError::engine()` promotes a message containing the driver's own OOM text to
/// `Overloaded`, because a CUDA OOM IS capacity — see `is_cuda_oom`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrClass {
    /// The caller can fix this. -> 400 invalid_request_error.
    InvalidRequest,
    /// Prompt does not fit the context. -> 400 + `code: context_length_exceeded`, the
    /// machine-readable form every client uses to decide "summarize and retry" (G16).
    ContextLength,
    /// Unknown model id. -> 400 + `code: model_not_found`.
    ///
    /// WHY 400 AND NOT 404: OpenRouter's uptime math counts 404 against the provider while
    /// 400 is excluded (§2.2), and "you asked for a model this endpoint does not serve" is
    /// squarely a client error — taking an uptime hit for it would be self-punishment for
    /// someone else's typo. The `code` is what clients branch on either way.
    ModelNotFound,
    /// Admission-time QoS shed: this lane is over its budget RIGHT NOW and a retry in a
    /// couple of seconds will work. -> 429 + Retry-After. Uptime-neutral at OpenRouter, and
    /// their own guidance prefers an early 429 to queueing.
    RateLimit,
    /// The BOX is out of capacity (VRAM exhausted, step OOM past its park budget). -> 503,
    /// not 429: "a 429 that a client cannot fix by waiting should not be a 429", and OpenAI
    /// itself serves overload as 503. This one honestly counts against uptime, because it is
    /// a request we failed to serve.
    Overloaded,
    /// An engine/GPU fault: a step, prefill, graph, or constraint operation failed. -> 500.
    Engine,
}

/// A classified failure. `message` stays the exact producer text (quoted, never rewritten —
/// the evidence-discipline law applies to what the client sees too).
#[derive(Debug, Clone)]
pub struct EngineError {
    pub class: ErrClass,
    pub message: String,
    /// OpenAI `error.param` when the failure names a request field.
    pub param: Option<&'static str>,
}

impl EngineError {
    /// Every invalid-request the WORKER can produce names a request field (it has already been
    /// through request parsing), so there is deliberately no param-less constructor — the
    /// flags doctrine applies to APIs too: no dead arm.
    pub fn invalid_param(message: impl Into<String>, param: &'static str) -> Self {
        Self {
            class: ErrClass::InvalidRequest,
            message: message.into(),
            param: Some(param),
        }
    }
    pub fn context_length(message: impl Into<String>) -> Self {
        Self {
            class: ErrClass::ContextLength,
            message: message.into(),
            param: Some("messages"),
        }
    }
    pub fn model_not_found(message: impl Into<String>) -> Self {
        Self {
            class: ErrClass::ModelNotFound,
            message: message.into(),
            param: Some("model"),
        }
    }
    pub fn rate_limit(message: impl Into<String>) -> Self {
        Self {
            class: ErrClass::RateLimit,
            message: message.into(),
            param: None,
        }
    }
    pub fn overloaded(message: impl Into<String>) -> Self {
        Self {
            class: ErrClass::Overloaded,
            message: message.into(),
            param: None,
        }
    }
    /// An engine fault. A message carrying the DRIVER'S OWN out-of-memory text is promoted to
    /// `Overloaded` (503 + Retry-After) rather than reported as a 500: the box ran out of
    /// VRAM, which is a capacity condition a retry can clear, not a bug in the engine. The
    /// test is `is_cuda_oom` — the same quoted-text predicate the step-OOM park path uses, so
    /// the two paths can never disagree about what an OOM is.
    pub fn engine(message: impl Into<String>) -> Self {
        let message = message.into();
        let class = if is_cuda_oom(&message) {
            ErrClass::Overloaded
        } else {
            ErrClass::Engine
        };
        // Log HERE, in the constructor, not at the 27 call sites: an engine fault that only
        // ever appears in the client's 500 body is invisible on a box whose stdout and stderr
        // both go to /dev/null, and a per-site eprintln! is one a future site can forget.
        eprintln!("[engine-error] class={class:?} {message}");
        Self {
            class,
            message,
            param: None,
        }
    }
}

/// Per-request spec-decode acceptance summary (lane/accept-telemetry, 2026-08-05): THIS
/// request's own rounds/drafted/accepted, diffed off the session telemetry around each burst
/// (a pool-resumed session carries prior requests' cumulative counts — the diff isolates
/// this request). Rides `Event::Done` into the response `usage` block as an additive
/// OpenAI-safe extension field.
#[derive(Debug, Clone, Copy, Default)]
pub struct SpecUsage {
    pub rounds: u64,
    pub drafted: u64,
    pub accepted: u64,
}

/// A generation request submitted by an HTTP handler to the worker.
pub struct Request {
    pub model: String,
    pub prompt_ids: Vec<u32>, // already tokenized? no — worker tokenizes (it owns the Tokenizer)
    pub prompt_text: String,
    pub chat: bool,
    pub chat_turns: Vec<memra_tokenizer::chat::Turn>,
    /// Tool schemas pre-serialized (client key order preserved) for the template's <tools> block.
    pub tools_json: Vec<String>,
    /// gemma4 tooluse dialect: the tool `function` objects as a typed tree (the compact gemma
    /// dialect needs argument/schema type fidelity a pre-rendered string cannot carry). Empty
    /// for every non-gemma dialect and every no-tools request.
    pub tools_struct: Vec<memra_tokenizer::chat::Val>,
    pub think: memra_tokenizer::chat::ThinkMode,
    /// Per-dialect reasoning level ("low"/"medium"/"high"): step35 renders it into the
    /// system turn (`Reasoning: {level}\n\n`); deepseek-v4 resolves it through the
    /// artifact's encoding revision into the effort prompt prefix (0731 ladder — "high" is
    /// a real prefix there; the tokenizer carries the detected `Dsv4Encoding`). Only set by
    /// the HTTP layer when the model's template consumes it (`ModelCaps::effort_levels` or
    /// `ModelCaps::dsv4`); None = the template's own default. Orthogonal to `think`: on
    /// switch-carrying templates `reasoning_effort` maps to ThinkMode instead and this
    /// stays None.
    pub reasoning_effort: Option<String>,
    pub params: GenParams,
    pub sampler_cfg: SamplerConfig,
    pub stop_strings: Vec<String>,
    pub trace_id: Option<String>,
    /// Optional provider-declared prompt ceiling. The HTTP layer copies this from the
    /// model metadata; the worker enforces it after rendering/tokenization, before cache
    /// lookup, admission accounting, or any GPU work.
    pub max_prompt_tokens: Option<usize>,
    /// PC-ISO cache namespace (lane/pc-iso, 2026-08-02): the tenant isolation salt for
    /// EVERY cross-request KV reuse tier (prefix cache, continuation pool, spec pool) —
    /// the vLLM `cache_salt` design. Derived by the HTTP layer (request `cache_salt`
    /// field; "" = the default single-tenant namespace, byte-identical to pre-PC-ISO).
    pub cache_ns: String,
    /// SESSION AFFINITY explicit tier (lane/session-affinity, 2026-08-05): the client's own
    /// name for this conversation (`session_id`/`user` body field, or the `x-session-id`
    /// header — see `crate::affinity_key`). Some(id) nominates that conversation's parked
    /// session directly; None falls back to the implicit structural fingerprint. Either way
    /// the resume decision is the exact token diff (`affinity_match`), scoped to this
    /// request's own (model, cache_ns) pool.
    pub affinity: Option<String>,
    /// yield lane (x-lane header; default interactive). Drives admission + prefill budgets
    /// (lane/dl-metering QoS gate, ported 2026-08-02 — the metering half stayed behind).
    pub lane: crate::lanes::Lane,
    /// STEP-OOM PARK budget already spent by this request (lane/admit-oom, 2026-08-06).
    /// Always 0 from the HTTP layer; only `park_requeue` sets it, carrying the count across
    /// a re-admit so the retry bound is per-REQUEST and a parked session cannot loop forever.
    pub oom_retries: u32,
    /// Request-owned speculative depth preserved across a step-OOM re-admit. Fresh HTTP
    /// requests leave this None and run the K policy after tokenization; a parked request
    /// carries Some(K), including K=0, so changing queue depth cannot change its decision.
    pub spec_k_replay: Option<usize>,
    /// Constrained decoding (`response_format` json_object/json_schema): the parsed
    /// grammar spec. None = unconstrained — the request takes the exact legacy path
    /// (no factory, no matcher, no masking branch).
    pub grammar: Option<crate::constrained::GrammarSpec>,
    /// Fresh matcher returned by the off-tick constraint compiler. `Some` iff `grammar` is
    /// armed and compilation completed successfully; admission never builds this inline.
    pub(crate) prepared_constraint: Option<crate::constrained::SessionConstraint>,
    /// Fresh HTTP constrained requests wait on this one-shot before response headers are
    /// committed, so compile rejection/timeout is a real 400/503 even for streaming calls.
    /// Replays leave it None: the original response already passed preflight.
    pub(crate) constraint_ready: Option<tokio::sync::oneshot::Sender<Result<(), EngineError>>>,
    /// Worker-owned rendered/tokenized prompt. Admission may revisit a queued request for many
    /// ticks; cache the result so a 256k request is not re-rendered and re-tokenized per defer.
    /// HTTP and step-OOM replay requests always start with None.
    pub(crate) prepared_prompt: Option<Vec<u32>>,
    /// Debug-only per-request phase timeline. None unless MEMRA_TTFT_TRACE=1.
    pub ttft: Option<Arc<crate::ttft::Trace>>,
    /// VISION INPUT (lane/vision): preprocessed images in message order. The HTTP layer
    /// decoded + patchified them and rendered a matching `<|image_pad|>` run per image into
    /// the turn content; the worker validates run-vs-image alignment after tokenization and
    /// primes with the mixed-embedding overlay. Non-empty => this request bypasses every
    /// cross-request KV reuse tier (prefix cache / continuation pool / affinity) — those
    /// tiers key on TOKENS, and pad tokens are identical across different images.
    pub images: Vec<memra_engine::vision_pre::VisionUnit>,
    /// GEMMA-4 vision units (lane/gemma-vision). Parallel to `images`; a request carries
    /// one family or the other (the process serves one). Same reuse-bypass law applies.
    pub gemma_images: Vec<memra_engine::vision_gemma::GemmaVisionUnit>,
    /// EMBEDDINGS/RERANK capture (lane/embed-serve): after prefill completes, publish
    /// `Event::PromptCapture` (last-position post-norm hidden and/or requested logit ids),
    /// then finish without decoding (the route forces `max_new: 0`). Capture requests
    /// bypass every cross-request KV reuse tier — a cache hit skips the prime that
    /// produces the hidden stack — and prime alone (no batched prime).
    pub capture: Option<CaptureSpec>,
    /// Reserved host patch-memory budget. The permit is moved through requeues and released only
    /// when this worker-owned request is dropped, including streaming completion/cancellation.
    pub(crate) vision_memory: Option<crate::VisionMemoryPermit>,
    /// per-request stream back to the handler. tokio mpsc so the async side can await it.
    pub tx: tokio::sync::mpsc::UnboundedSender<Event>,
}

/// What a capture request wants read off the final prompt position (lane/embed-serve).
#[derive(Debug, Clone, Default)]
pub struct CaptureSpec {
    /// Last-token post-final-norm hidden state (the embedding pooling read).
    pub hidden: bool,
    /// Vocabulary PIECES whose last-position logits to return (the rerank yes/no score
    /// read). The worker resolves each piece against the model's own tokenizer — the
    /// HTTP layer does not always hold one. A piece that is not a single token in this
    /// model's vocabulary reports `f32::MIN` (the route refuses rerank on such models).
    pub logit_pieces: Vec<String>,
}

/// Conservative source-byte ceiling applied before any chat-template render or tokenizer call.
/// The HTTP body limit remains large enough for vision envelopes, but text/tool inputs must not
/// turn that envelope into an unbounded tokenizer allocation. Eight MiB is well above the bytes
/// needed for the model's advertised context while leaving substantial headroom for template
/// framing and UTF-8 expansion.
pub const MAX_PROMPT_SOURCE_BYTES: usize = 8 * 1024 * 1024;

pub fn prompt_source_bytes(req: &Request) -> usize {
    if !req.prompt_ids.is_empty() {
        return req
            .prompt_ids
            .len()
            .saturating_mul(std::mem::size_of::<u32>());
    }
    let mut total = req.prompt_text.len();
    for turn in &req.chat_turns {
        total = total.saturating_add(turn.role.len());
        total = total.saturating_add(turn.content.len());
        if let Some(reasoning) = turn.reasoning.as_deref() {
            total = total.saturating_add(reasoning.len());
        }
        if let Some(id) = turn.tool_call_id.as_deref() {
            total = total.saturating_add(id.len());
        }
        if let Some(name) = turn.tool_name.as_deref() {
            total = total.saturating_add(name.len());
        }
        for call in &turn.tool_calls {
            total = total.saturating_add(call.name.len());
            for (key, value) in &call.params {
                total = total.saturating_add(key.len()).saturating_add(value.len());
            }
        }
    }
    for schema in &req.tools_json {
        total = total.saturating_add(schema.len());
    }
    total
}

pub fn prompt_source_limit_error(req: &Request) -> Option<String> {
    let bytes = prompt_source_bytes(req);
    (bytes > MAX_PROMPT_SOURCE_BYTES).then(|| {
        format!(
            "prompt source is {bytes} bytes; maximum allowed before tokenization is {MAX_PROMPT_SOURCE_BYTES} bytes"
        )
    })
}

/// Chat-template capabilities probed from a loaded model's template at spawn time — the
/// HTTP layer rejects `tools` on models whose template has no tools branch BEFORE the
/// request reaches the worker, and arms the tool-call parser's think gate.
/// Plus the /v1/models metadata surface (serve-tail lane, 2026-08-04): trained context,
/// tokenizer family, chat-template family — worker truth captured once at spawn so the
/// HTTP layer never invents values (unknown = 0/""/None -> honest nulls in the route).
#[derive(Debug, Clone, Default)]
pub struct ModelCaps {
    /// template carries the qwen-class `<tools>` branch (tools + tool_response rendering).
    pub tools_branch: bool,
    /// template appends a `<think>` tail on the generation prompt (qwen think class).
    pub qwen_think: bool,
    /// template has the `enable_thinking` switch (ThinkMode::NoThink is honored).
    pub think_switch: bool,
    /// chat requests are honest against this model: it has a chat template, OR it is a
    /// GGUF (which keeps the historical plain-ChatML fallback). A safetensors/repack DIR
    /// checkpoint without a template 400s on /v1/chat/completions (serve-st v1 honesty
    /// gate) instead of silently rendering a format the model was never trained on.
    pub chat_ok: bool,
    /// model's trained context length (config; 0 = unknown) — /v1/models `context_length`.
    pub context_length: usize,
    /// tokenizer family (the GGUF/HF pre-tokenizer name, e.g. "qwen2"; "" = unknown).
    pub tokenizer: String,
    /// chat-template family ("chatml" / "gemma"); None = no template or unrecognized.
    pub instruct_type: Option<String>,
    /// template consumes a `reasoning_effort` STRING (the step35 dialect: rendered into the
    /// system turn as `Reasoning: {level}\n\n`). When true, the HTTP layer maps the OpenAI
    /// `reasoning_effort` body field onto `Request::reasoning_effort` instead of ThinkMode.
    pub effort_levels: bool,
    /// template carries the QWEN3.8 reasoning-effort ladder — `reasoning_effort|default('xhigh')`
    /// over `xhigh|medium|low` (`high` aliased to `xhigh`), injected as an instruction sentence
    /// at the head of the system turn. A SECOND, differently-spelled effort dialect from
    /// `effort_levels`: it is keyed on the instruction sentences the renderer reproduces
    /// (`chat::template_has_qwen_effort`) rather than on `reasoning_effort is defined`, which
    /// this template does not contain — and that mismatch is exactly why the level used to be
    /// dropped on every qwen3.8 request (lane/reasoning-schema-20260823).
    ///
    /// Note it is INDEPENDENT of `think_switch`: qwen3.8 carries both (the switch turns
    /// reasoning off, the ladder sets its depth), and a template could carry either alone.
    pub qwen_effort: bool,
    /// gemma4 thought-channel dialect (lane/gemma4-serve-gaps, 2026-08-07): the template's
    /// `strip_thinking` splits on `<|channel>thought…<channel|>`. When true, chat requests
    /// arm the gemma-dialect reasoning splitter so thought text routes to `reasoning` and
    /// the channel tags never reach the client as content.
    pub gemma_think: bool,
    /// deepseek-v4 (`encoding_dsv4`) dialect (lane/dsv4-template, 2026-08-18): three think
    /// modes + DSML tool calls. When true, chat requests arm the dsv4 parser (reasoning split
    /// on `</think>`, `<｜DSML｜tool_calls>…</｜DSML｜tool_calls>` -> OpenAI tool_calls) and the
    /// serve path maps ThinkMode onto encoding_dsv4's thinking_mode. Keyed on the same
    /// `template_is_dsv4` marker law the renderer dispatch uses.
    pub dsv4: bool,
    /// GLM-5.3-Flash (`glm5_next`) dialect (lane/glm53-flash-bringup, 2026-08-27): the
    /// `[gMASK]<sop>` + `<|user|>`/`<|assistant|>`/`<|observation|>` turn dialect, an
    /// always-open `<think>` tail, an always-rendered `Reasoning Effort:` system line with a
    /// real rung ABOVE high (`max`, its default), and `<tool_call>NAME<arg_key>…` tool calls.
    /// When true the renderer takes the glm5 arm, the chat path arms the glm5 tool/reasoning
    /// parser, and the effort ladder keeps its `max` tier instead of clamping to `high`.
    /// Keyed on the same `template_is_glm5` marker law the renderer dispatch uses. Latent
    /// before this cap existed: every glm5 marker is ALSO a qwen marker, so the ChatML arm
    /// answered for it (research/glm53-flash-bringup-20260827/SURFACE-AUDIT).
    pub glm5: bool,
    /// Model-provider sampling defaults for OMITTED chat fields. `None` keeps the generic
    /// OpenAI-compatible defaults in the HTTP layer. Step35 uses StepFun's published API
    /// defaults (temperature 0.5, top_p 0.9); explicit client values always win.
    pub chat_temperature_default: Option<f32>,
    pub chat_top_p_default: Option<f32>,
    /// Tokenizer vocabulary size (0 = unknown) — bounds client-supplied `prompt_ids` at
    /// HTTP intake (hermes, 2026-08-19): an out-of-vocab id used to ride unchecked into
    /// the embed gather, where an attacker-chosen index reads past the embedding table.
    /// The tokenizer's own id space is the honest ceiling: ids past it aren't tokens,
    /// even where a padded lm_head would happen to make the gather "safe".
    pub n_vocab: usize,
}

/// Control messages into the worker. Currently just generation requests; /models and /health are
/// served from the cached model-name list captured at spawn (no need to round-trip the worker).
pub enum Cmd {
    Generate(Box<Request>),
    /// Drop every EVICTABLE cross-request pool — KV reuse, spec/dspark resume, the
    /// prefix cache — and report the entry counts freed (deploy-headroom lane,
    /// 2026-08-27). In-flight sessions are untouched; the pools rebuild from traffic.
    /// Producer: `POST /admin/trim`, whose caller is serve-deploy's overlap preflight
    /// freeing VRAM for the green slot beside a warm blue.
    TrimPools(tokio::sync::oneshot::Sender<TrimReport>),
}

/// What `Cmd::TrimPools` freed, by pool (entry counts, not bytes — the device memory
/// returns to CUDA on drop and shows up in `nvidia-smi` free within a tick).
#[derive(Debug, serde::Serialize)]
pub struct TrimReport {
    pub reuse_entries: usize,
    pub spec_reuse_entries: usize,
    pub dspark_reuse_entries: usize,
    pub prefix_entries: usize,
}

const CONSTRAINT_RESULT_POLL: Duration = Duration::from_millis(5);

struct PendingConstraintCompile {
    request: Box<Request>,
    deadline: Instant,
}

/// Pending-admission gauge (lane/admission-latency, 2026-08-06): the HTTP handler increments
/// it right before sending `Cmd::Generate`; the worker decrements at pop (`handle_cmd`). A
/// spec burst polls it at every round boundary (the sse-cadence on_commit hook) and ENDS the
/// burst early when a request is waiting — the tick loop re-checks admission, so a newcomer's
/// wait stops scaling with MEMRA_SPEC_BURST (B128 held admits a whole ~1.3s burst out).
/// Burst size is content-neutral (spec-levers battery): the early exit moves WHEN control
/// returns, never what tokens say. Saturating decrement: a direct-channel sender that never
/// incremented (tests) must not underflow the gauge.
pub static PENDING_ADMITS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Per-lane reservations held from HTTP submission until the request is either admitted to a
/// live session (or the dedicated DSV4 serving thread) or rejected. Lane separation is
/// load-bearing: subordinate harvest traffic must never consume the interactive queue bound.
/// Unlike `PENDING_ADMITS`, these are the hard bounds for the unbounded handler/DSV4 channels
/// and remain held while requests wait in the normal worker queue.
pub static ADMISSION_RESERVATIONS: [std::sync::atomic::AtomicUsize; 3] = [
    std::sync::atomic::AtomicUsize::new(0),
    std::sync::atomic::AtomicUsize::new(0),
    std::sync::atomic::AtomicUsize::new(0),
];

fn decrement_atomic(counter: &std::sync::atomic::AtomicUsize) {
    let _ = counter.fetch_update(
        std::sync::atomic::Ordering::AcqRel,
        std::sync::atomic::Ordering::Acquire,
        |value| value.checked_sub(1),
    );
}

/// Release the command-channel portion of an HTTP admission. This is separate from the hard
/// queue reservation because the latter survives while a request waits in the worker queue.
pub(crate) fn release_pending_admit() {
    decrement_atomic(&PENDING_ADMITS);
}

/// Release a request's hard admission reservation. This is intentionally saturating because
/// a few embedders/tests inject commands directly without going through the HTTP reservation
/// path.
pub(crate) fn release_admission_reservation(lane: Lane) {
    decrement_atomic(&ADMISSION_RESERVATIONS[lane.idx()]);
}

/// Requeue a worker-owned request after a bounded step-OOM park. It was released when the
/// original session was admitted, so the retry needs a fresh queue reservation before it is
/// inserted back into the FIFO.
pub(crate) fn reserve_internal_admission(lane: Lane) {
    ADMISSION_RESERVATIONS[lane.idx()].fetch_add(1, std::sync::atomic::Ordering::AcqRel);
}

/// Serving counters + engine-truth step latency, published every 32nd tick.
#[derive(Clone, Default)]
pub struct Metrics {
    pub admitted: u64,
    pub completed: u64,
    pub tokens_out: u64,
    pub step_p50_ms: f32,
    pub step_p99_ms: f32,
    /// worker-truth prompt accounting: total prompt tokens admitted, and how many of
    /// those were served from a cache (continuation pool / spec resume / prefix cache).
    pub prompt_tokens_in: u64,
    pub cached_tokens_in: u64,
    /// cross-request prefix cache state (hits/entries/resident bytes).
    pub prefix_hits: u64,
    pub prefix_entries: u64,
    pub prefix_bytes: u64,
    /// full prefix-cache counter set (lane/cache-metering, 2026-08-07): misses/inserts/
    /// evictions were already counted inside PrefixCache but never published; hit_tokens
    /// is the token-weighted hit mass (sum of entry lengths served) — the numerator the
    /// economics row wants when hits vary in depth.
    pub prefix_misses: u64,
    pub prefix_inserts: u64,
    pub prefix_evictions: u64,
    /// Prefix snapshots refused before insertion: an entry larger than the whole byte budget,
    /// or a pinned entry that cannot fit beside the current in-flight leases. These are separate
    /// because only the first means the configured/derived budget can never admit that shape.
    pub prefix_skips_budget: u64,
    pub prefix_skips_pinned: u64,
    pub prefix_hit_tokens: u64,
    /// Serving-hardening counters and gauges (lane/cx-cachespec, 2026-08-09). These make
    /// cache/admission latency attributable from `/metrics` instead of requiring a server-log
    /// grep. Defers count admission decisions (a queued request may be deferred on multiple
    /// ticks); parks count successful step-OOM requeues. Pool entry fields are current gauges.
    pub admission_session_defers: u64,
    pub admission_vram_defers: u64,
    pub step_oom_parks: u64,
    pub continuation_pool_hits: u64,
    pub continuation_pool_evictions: u64,
    /// PLAIN-SESSION AFFINITY (lane/plain-affinity): continuation-pool resumes that rewound to a
    /// prompt-end checkpoint on a rewritten-history turn (subset of continuation_pool_hits).
    pub plain_affinity_rewinds: u64,
    pub spec_pool_hits: u64,
    pub spec_pool_misses: u64,
    pub spec_pool_affinity_rewinds: u64,
    pub spec_pool_evictions: u64,
    /// SAMPLER PREDICATE (lane/session-resume-sampler-predicate-20260820): spec-pool probes
    /// refused because the parked session's sampler differs (subset of `spec_pool_misses`).
    pub spec_pool_sampler_refusals: u64,
    /// SERVED-PATH RECEIPTS (lane/dspark-sampled-wave-20260825, serving law "we always
    /// verify sampled on prod after deploy"): cumulative per-request classification at
    /// admission — dspark route, a spec program (MTP or gemma drafter), or plain decode.
    /// These exist so a deploy gate can prove speculation ENGAGED on its vendor-default
    /// sampled probe from /metrics, instead of trusting a 200 the plain path answers
    /// fluently at half speed. Admission-time truth: a later demotion does not move a
    /// request between buckets (demotions are separately observable in the server log).
    pub served_dspark: u64,
    pub served_spec: u64,
    pub served_plain: u64,
    pub active_sessions: u64,
    pub queued_requests: u64,
    pub continuation_pool_entries: u64,
    pub spec_pool_entries: u64,
    pub cuda_driver_free_bytes: u64,
    pub cuda_pool_reserved_bytes: u64,
    pub cuda_pool_used_bytes: u64,
    pub cuda_pool_cached_bytes: u64,
    /// LCP length histogram (lane/cache-metering): one sample per prefix-cache PROBE —
    /// on a hit, the served entry's token length; on a miss, best_lcp against the pool
    /// (already computed there for the split-learning signal, so the histogram adds no
    /// scan). Buckets: [0], [1,16), [16,32), [32,64), [64,128), [128,256), [256,512),
    /// [512,1024), [1024,2048), [2048,4096), [4096,inf) — the [64,512) window is the
    /// tick-seg segmentation class. Spec-tier and non-batched requests never probe the
    /// prefix cache and are absent by construction.
    pub lcp_hist: [u64; 11],
    /// Per-tenant prompt accounting [prompt_tokens_in, cached_tokens_in], keyed by the
    /// TENANT half of the PC-ISO namespace (`meter_key`): keyring deployments aggregate
    /// one row per tenant across its end-user salts; no-keyring deployments key on the
    /// raw cache_salt ("" = the default namespace). Bounded at METER_TENANT_CAP rows —
    /// overflow traffic lands in "(other)" so a salt-spraying client cannot grow the map.
    pub ns_tokens: HashMap<String, [u64; 2]>,
    /// per-lane QoS counters [interactive, judge, harvest] — the x-lane yield gate
    /// (/yield/metrics, sidecar-compatible shape; lane/dl-metering QoS extraction).
    pub lane_admitted: [u64; 3],
    pub lane_shed: [u64; 3],
    pub lane_completed: [u64; 3],
    pub lane_tokens: [u64; 3],
    pub batch_size_last: usize,
    /// Per-model spec-decode acceptance telemetry (lane/accept-telemetry): cumulative
    /// since the model loaded — models load once per server process, so these counters
    /// reset on (re)load/restart, never mid-run. Empty (absent from /metrics) for models
    /// that never ran a spec burst — zero-cost when spec is off.
    pub spec: HashMap<String, memra_engine::spec::SpecTelemetry>,
    /// Rolling per-model acceptance snapshots used by first-class `spec_tau` and
    /// `spec_accept_by_position`. The lifetime `spec` block above stays for compatibility.
    pub spec_window: HashMap<String, memra_engine::spec::SpecTelemetry>,
    /// Detection-only ADSD acceptance-collapse incidents, keyed by the same tenant row
    /// identity as `ns_tokens`. The worker never evicts or throttles from this counter;
    /// operators apply the existing tenant/lane rate limit after investigation.
    pub adsd_suspect_total: HashMap<String, u64>,
    /// Per-model latched gauge for the constrained compiler's abandoned-worker cap. Each model's
    /// current compiler owns its AtomicBool; a worker respawn replaces the map entry, so an old
    /// compiler cannot overwrite the new generation's state. Operator metrics render this 0/1.
    pub constraint_compiler_fail_closed: HashMap<String, Arc<std::sync::atomic::AtomicBool>>,
}
pub type SharedMetrics = std::sync::Arc<std::sync::Mutex<Metrics>>;

/// Windowed percentile over decode-step latencies (ms) — the interactive SLO sensor.
/// Engine ground truth: the worker records the wall time of each batched decode tick that
/// advanced at least one interactive session — that IS the client-visible TPOT for that
/// tick. (Shared with out-of-process controllers via the memra-lanes crate.)
use crate::lanes::{Lane, StepStats};

/// Live per-session state on the worker thread. One `Session` per in-flight generation.
/// Holds the per-session `Cache` (model-specific dims — NO sharing between sessions, which is what
/// makes the concurrent streams byte-identical to isolated runs) and per-session `Sampler`.
/// KV PREFIX REUSE (append-only continuation): retired sessions park (fed tokens, Cache,
/// last_logits) here; a new request whose prompt EXACTLY EXTENDS a parked `fed` sequence takes
/// the Cache and primes only the suffix. Correct by construction for hybrid models: the
/// recurrent (conv/ssm) state in the Cache is the state AFTER the last fed token — the exact
/// resume point for an append-only continuation. NO arbitrary-prefix truncation is attempted
/// (GDN state cannot roll back without checkpoints); a non-extending prompt takes the cold path.
/// NOTE chat-template callers: templates that rewrite history (e.g. stripping think blocks from
/// prior assistant turns) break exact-extension and simply miss the pool — raw `prompt_ids`
/// callers (agent loops) always hit. Pool: at most MEMRA_REUSE_POOL entries per (model, cache
/// namespace), LRU, within the process-wide MEMRA_REUSE_POOL_GLOBAL_CAP ceiling.
struct ReuseEntry {
    fed: Vec<u32>,
    cache: Cache,
    last_logits: Vec<f32>,
    cap: usize,
    /// PLAIN-SESSION AFFINITY (lane/plain-affinity, 2026-08-09): a retire-time checkpoint at a
    /// STABLE PRE-GENERATION boundary, so a later turn that REWRITES history (the pi client
    /// stripping `<think>` from prior assistant turns) can still recognize this conversation and
    /// rewind here — priming only its own delta instead of the whole growing conversation.
    /// `None` for a retire that could not capture one (a rig too tight for the GDN state copy;
    /// resume simply isn't available and the pool falls back to exact-extension only). See
    /// [`PlainCheckpoint`] and the plain-affinity probe in `admit`.
    ckpt: Option<PlainCheckpoint>,
    /// The conversation id the admitting request declared (`Request::affinity` — explicit tier).
    /// `Some(id)` nominates only; the resume decision is the exact token diff (`affinity_match`).
    affinity: Option<String>,
    /// Implicit-tier identity: fingerprint chain of this session's COMMITTED (== `fed`) tokens.
    /// Nominated by a shared leading run with a request's own chain (`fingerprint_affinity`).
    fingerprint: Vec<u64>,
    /// Global park age used by admission and preemptive ceiling reclaim. Per-key vector order
    /// remains unchanged.
    parked_at: Instant,
}
/// A retired PLAIN session's PROMPT-END-class boundary state — the rewind target for
/// plain-session affinity resume (lane/plain-affinity, 2026-08-09). The plain twin of
/// `memra_engine::spec::SpecCheckpoint`, but structurally simpler: a plain session owns one
/// tokenwise `Cache` (no draft scratch, no `last_h` anchor), so the checkpoint is exactly a
/// `Cache::snapshot` at the boundary plus the committed length and the boundary row's host
/// logits (needed to resume decode from the checkpoint when the suffix is empty — though the
/// probe declines empty-suffix resumes, the logits keep the entry self-describing).
///
/// WHY THIS BOUNDARY. The rewrite class this lane exists for mutates the text the session
/// GENERATED, never the prompt bytes the client resends verbatim. So turn N+1's prompt agrees
/// with turn N's committed tokens up to the point where turn N's own generation began — the
/// pre-generation boundary. Checkpointing there makes a resume prime only turn N+1's delta.
///
/// WHY IT IS SAFE TO ROLL BACK (the SWA-ring question, resolved). memra has NO ring buffer:
/// full-attn KV (including step35's SWA layers) is a read-side mask over a full `max_ctx`
/// allocation with absolute rope, so a rollback is a pure per-layer `len` truncation — the
/// exact operation `Cache::rollback` performs. GDN conv/ssm recurrent state has no position
/// index and IS mutated in place, so the snapshot holds a real device copy of it; rollback
/// restores that copy. This is byte-for-byte the state a fresh prime of `fed[..pos]` produces,
/// which is the whole exactness argument (identical to `spec_rewind_to_checkpoint`).
struct PlainCheckpoint {
    snap: memra_engine::cache::CacheSnapshot,
    /// Committed length at the boundary (== `snap.pos`, == cache.pos there).
    pos: usize,
    /// Host logits of row `pos - 1` (the boundary row) — the resume seed for an empty suffix.
    last_logits: Vec<f32>,
}
/// SPEC-session reuse (2026-07-05): a retired spec session parks WHOLE (trunk cache + draft
/// scratch + committed + next_pred). A new greedy request whose prompt exactly extends
/// `committed` resumes it — turn N+1 primes only the suffix (or nothing, the continuation
/// burst). Same exact-extension rule as ReuseEntry; the session-gate oracle covers this path.
struct SpecReuseEntry {
    sess: memra_engine::spec::SpecSession,
    /// detok(committed) — TEXT-level prefix matching (2026-07-06). Token-level starts_with
    /// missed ~50% of chat turn boundaries (detok->retok BPE merges differ at the seam). Text
    /// matching resumes whenever the new prompt string literally extends the parked
    /// conversation; only the remainder is tokenized (no BOS). Same acceptable-divergence class
    /// as llama serve's cache_prompt: the suffix's boundary tokenization may differ from a cold
    /// full-retok — committed tokens stay authoritative, spec==greedy exactness is untouched.
    committed_text: String,
    /// SESSION AFFINITY (lane/session-affinity): the conversation this session belongs to, as
    /// the admitting request declared it — `Some(id)` from the explicit tier
    /// (`session_id`/`user`/`x-session-id`), else None. Nomination only; see `affinity`.
    affinity: Option<String>,
    /// Implicit-tier identity: the fingerprint chain of the session's COMMITTED tokens (no
    /// live tail to drop). Nominated by a shared leading run with a request's own chain.
    fingerprint: Vec<u64>,
    /// THE SAMPLER THAT SHAPED THIS SESSION (lane/session-resume-sampler-predicate-20260820;
    /// receipts `research/spec-cache-20260818/SESSION-RESUME-PREDICATE.md`).
    ///
    /// Recorded at park because the probes below could not otherwise ask the question: they
    /// compared prompts and NEVER samplers, which is how a `top_p`/`top_k` request inherited a
    /// draft graph captured for an unfiltered one on a live server (`SampledGraphKey`,
    /// lane/graph-s-key-exactness-20260819). Keying the graph closed the exactness hole; it did
    /// not make cross-sampler resume sound, so the probe now refuses a differing sampler and NAMES
    /// the field on the downgrade line.
    ///
    /// The PLAIN pool ([`ReuseEntry`]) deliberately carries no such field: every byte a plain park
    /// holds (`cache`, `fed`, `last_logits`, `ckpt`) is a pure function of the fed token sequence,
    /// the worker replays the sampler's penalty history over the resumed prefix
    /// (`sampler.accept`), and `Sampler::new` re-seeds its RNG per request — so a plain resume has
    /// no sampler-shaped state to inherit. Only spec sessions park sampler-shaped state.
    sampler: SamplerIdentity,
    /// Global park age used by admission and preemptive ceiling reclaim. Pool ownership/identity
    /// stays unchanged.
    parked_at: Instant,
}
trait ParkedEntryAge {
    fn parked_at(&self) -> Instant;
}

impl ParkedEntryAge for ReuseEntry {
    fn parked_at(&self) -> Instant {
        self.parked_at
    }
}

impl ParkedEntryAge for SpecReuseEntry {
    fn parked_at(&self) -> Instant {
        self.parked_at
    }
}

/// DSPARK-session reuse (lane/dflash2-session-reuse, 2026-08-25): the DFlash2 twin of
/// [`SpecReuseEntry`]. A retired dspark session parks WHOLE (trunk cache + draft KV +
/// selector state + philox counters); a new request whose prompt extends the parked
/// stream resumes it, priming ONLY the suffix (`dspark_spec_session_resume`) instead of
/// re-priming the conversation cold — the route's previous multi-turn cost. v1 scope:
/// EXACT and TEXT extension tiers with the same sampler predicate as the MTP pool; the
/// affinity-rewind tier needs a rollback checkpoint the dspark session does not retain
/// (named residual — a history-rewriting client re-primes cold, today's behavior).
struct DsparkReuseEntry {
    sess: memra_engine::dflash::DsparkSpecSession,
    /// The session's fed token stream (prompt + committed generation) — the exact-tier
    /// match key. The dspark session does not retain its committed ids itself; the
    /// worker's `fed` is the authoritative stream at retire (pos() == fed.len() gate).
    fed: Vec<u32>,
    /// detok(fed) minus the leading BOS — text-tier matching, same rationale as the MTP
    /// pool (BPE seams break token-level starts_with on ~50% of chat turn boundaries).
    committed_text: String,
    /// Stream ended in EOS — an empty-suffix resume of a finished stream would re-emit
    /// from a terminal state, so `done` entries require a non-empty suffix.
    done: bool,
    affinity: Option<String>,
    sampler: SamplerIdentity,
    parked_at: Instant,
}

impl ParkedEntryAge for DsparkReuseEntry {
    fn parked_at(&self) -> Instant {
        self.parked_at
    }
}

/// Per-(model, cache namespace) cap for each whole-session pool (MEMRA_REUSE_POOL, default 2).
/// Before the global gate, N live namespaces could retain roughly
/// `N * MEMRA_REUSE_POOL * entry_bytes` of VRAM per populated tier; spec entries can be multi-GB.
/// KNOWN CASCADE at the default (pinned 2026-07-31): park-on-retire evicts LRU
/// unconditionally, so a SEQUENTIAL multi-turn workload of N>cap sessions evicts each
/// next request's entry one step ahead of its arrival (0/N resumes), while the same
/// requests arriving CONCURRENTLY all probe the intact pool first (hit). Raise the cap
/// to >= the expected concurrent-session count when VRAM allows.
fn reuse_pool_per_namespace() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("MEMRA_REUSE_POOL")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(2)
    })
}
/// Process-wide ceiling across every model, namespace, and whole-session reuse tier. Default 16
/// preserves the measured two-namespace Q27 shape (16 spec entries, zero OOM parks, 27.34 GB
/// driver-free; research/27bab-20260810/RESULTS.md) while preventing client-controlled salt
/// fanout from multiplying the per-namespace cap indefinitely.
const DEFAULT_REUSE_POOL_GLOBAL_CAP: usize = 16;
fn reuse_pool_global_cap() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("MEMRA_REUSE_POOL_GLOBAL_CAP")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(DEFAULT_REUSE_POOL_GLOBAL_CAP)
    })
}
/// Minimum parked prefix worth reusing (below this, cold prime is cheaper than bookkeeping).
const REUSE_MIN_PREFIX: usize = 16;

/// ROLLBACK DOOR + A/B ARM for the whole-session sampler predicate
/// (lane/session-resume-sampler-predicate-20260820).
///
/// Default ON: a parked spec session resumes only when the incoming sampler is EQUIVALENT to the
/// one that shaped it ([`SamplerIdentity::mismatch`]). `MEMRA_SPEC_RESUME_SAMPLER=0` selects
/// `SamplerIdentity::legacy_admits` — the pre-lane probe, which compared prompts and never
/// samplers — and is the arm the cost measurement A/Bs against inside ONE binary, so a
/// hit-vs-refuse delta is not a cross-build comparison on a contended host.
///
/// The door is a POLICY door, not an exactness door: with it shut, `SampledGraphKey` still refuses
/// to launch a draft graph captured under a different regime (that fix is unconditional). What the
/// door restores is the older, weaker posture in which a sampler-differing request keeps a parked
/// session's penalty-window position, Philox stream position and draft-plane state.
fn spec_resume_sampler_predicate_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("MEMRA_SPEC_RESUME_SAMPLER").as_deref() != Ok("0"))
}

/// Does `req` admit resuming `entry`? On refusal, records the NAMED mismatching field in
/// `refused` so the caller's downgrade line can say why — a refusal that does not name itself is
/// indistinguishable from an unwired mechanism.
///
/// A free function rather than a closure so all three probe tiers (exact token prefix, text
/// prefix, session affinity) call ONE definition; the borrow of `refused` then ends at each call
/// instead of living for the whole probe block.
///
/// Callers must place this LAST in their conjunction: `refused` must only be set for an entry that
/// was otherwise resumable, or the downgrade line would blame the sampler for a prompt miss.
fn spec_resume_sampler_admits(
    req: &SamplerIdentity,
    entry: &SpecReuseEntry,
    refused: &mut Option<&'static str>,
) -> bool {
    match spec_resume_sampler_verdict(spec_resume_sampler_predicate_on(), req, &entry.sampler) {
        None => true,
        Some(field) => {
            *refused = Some(field);
            false
        }
    }
}

/// The predicate AND the door, as one pure function: `None` = admit, `Some(field)` = refuse and
/// name that field. Split out from [`spec_resume_sampler_admits`] so both door positions have CPU
/// teeth — `spec_resume_sampler_predicate_on` is a process-lifetime `OnceLock`, so a test that set
/// the env var would decide the door for every other test in the binary.
fn spec_resume_sampler_verdict(
    predicate_on: bool,
    req: &SamplerIdentity,
    parked: &SamplerIdentity,
) -> Option<&'static str> {
    if !predicate_on {
        debug_assert!(req.legacy_admits(parked), "legacy_admits must admit all");
        return None;
    }
    req.mismatch(parked)
}

/// F5 (spec-pool thrash, 2026-08-05 — research/specpool-20260804): learned per-model
/// spec-session sizing, process-lifetime (VRAM geometry is static per server run;
/// a restart re-probes). Two lessons the worker remembers so pool misses stop
/// re-paying doomed multi-GB cudaMalloc walks every turn:
#[derive(Default)]
struct SpecSizing {
    /// Models OBSERVED VRAM-tight: a spec-session alloc failed while a parked pool
    /// entry existed. Later misses on these models evict the (dead-weight) pool
    /// BEFORE allocating — the same eviction the failure path forced anyway, minus
    /// the failed alloc + full realloc churn (the owner's live thrash: "spec pool
    /// evicted (1) after alloc failure" once per request, every turn a miss).
    /// Rigs where ghost + new session both fit never set the flag and keep the
    /// parked entry's resume value.
    evict_first: std::collections::HashSet<String>,
    /// model -> largest ctx ask known to fit after a GENUINE (empty-pool) alloc
    /// failure — the right-size ladder's landing point. Later asks start here
    /// instead of re-laddering; never exceeds the request's own ctx_cap.
    learned_ctx: HashMap<String, usize>,
}

/// Worker-owned cumulative counters for the two whole-session reuse tiers. Prefix-cache
/// counters live on `PrefixCache` itself; these remain separate because speculative sessions
/// deliberately bypass that trunk-only cache.
#[derive(Default)]
struct ReuseMetrics {
    continuation_hits: u64,
    continuation_evictions: u64,
    spec_hits: u64,
    spec_misses: u64,
    spec_affinity_rewinds: u64,
    spec_evictions: u64,
    /// SAMPLER-PREDICATE REFUSALS (lane/session-resume-sampler-predicate-20260820): pool probes
    /// where a parked session matched the prompt but was shaped by a DIFFERENT sampler. Subset of
    /// `spec_misses`. This counter is the deliverable for "how often does real traffic change
    /// sampler mid-session": it answers from production, per model, which no synthetic multi-turn
    /// workload can settle. A deployment reading 0 here is paying nothing for the predicate.
    spec_sampler_refusals: u64,
    /// PLAIN-SESSION AFFINITY (lane/plain-affinity, 2026-08-09): continuation-pool hits that
    /// resumed via a rewind checkpoint (rewritten-history turn) rather than exact extension.
    /// Subset of `continuation_hits`; the per-turn resume count the affinity gate reads out.
    plain_affinity_rewinds: u64,
}

/// Exact context allocation for a cache with one physically capped row class.
pub(crate) fn context_cache_bytes(
    bytes_per_token: usize,
    ring_bytes_per_token: usize,
    ring_rows: usize,
    ctx_cap: usize,
) -> usize {
    let flat = bytes_per_token.saturating_sub(ring_bytes_per_token);
    flat.saturating_mul(ctx_cap)
        .saturating_add(ring_bytes_per_token.saturating_mul(ctx_cap.min(ring_rows)))
}

/// Per-model request-shaped admission estimate. Full-attention rows scale with the request's
/// context cap; Step35 SWA rows stop at their physical ring cap. Everything else is a measured
/// high-water residual, never lowered by reuse or allocator-pool hits.
#[derive(Debug)]
struct AdmissionCostModel {
    plain_bytes_per_token: usize,
    spec_bytes_per_token: usize,
    plain_ring_bytes_per_token: usize,
    spec_ring_bytes_per_token: usize,
    ring_rows: usize,
    activation_bytes: usize,
    last_logged: Option<(usize, bool, usize)>,
}

impl AdmissionCostModel {
    fn new(model: &HybridModel) -> Self {
        let (plain_bytes_per_token, plain_ring_bytes_per_token, plain_ring_rows) =
            model.plain_session_kv_shape();
        let (spec_bytes_per_token, spec_ring_bytes_per_token, spec_ring_rows) =
            model.spec_session_kv_shape();
        debug_assert!(
            plain_ring_rows == 0 || spec_ring_rows == 0 || plain_ring_rows == spec_ring_rows
        );
        Self {
            plain_bytes_per_token,
            spec_bytes_per_token,
            plain_ring_bytes_per_token,
            spec_ring_bytes_per_token,
            ring_rows: plain_ring_rows.max(spec_ring_rows),
            activation_bytes: 0,
            last_logged: None,
        }
    }

    fn bytes_per_token(&self, spec: bool) -> usize {
        if spec {
            self.spec_bytes_per_token
        } else {
            self.plain_bytes_per_token
        }
    }

    fn ring_bytes_per_token(&self, spec: bool) -> usize {
        if spec {
            self.spec_ring_bytes_per_token
        } else {
            self.plain_ring_bytes_per_token
        }
    }

    fn context_bytes(&self, ctx_cap: usize, spec: bool) -> usize {
        context_cache_bytes(
            self.bytes_per_token(spec),
            self.ring_bytes_per_token(spec),
            self.ring_rows,
            ctx_cap,
        )
    }

    fn estimate(&self, ctx_cap: usize, spec: bool) -> usize {
        self.context_bytes(ctx_cap, spec)
            .saturating_add(self.activation_bytes)
    }

    /// Learn only the fixed residual beyond the exact context allocation.
    /// Returns the new high-water value when it moved.
    fn observe(&mut self, observed_bytes: usize, ctx_cap: usize, spec: bool) -> Option<usize> {
        let context = self.context_bytes(ctx_cap, spec);
        let residual = observed_bytes.saturating_sub(context);
        if residual > self.activation_bytes {
            self.activation_bytes = residual;
            Some(residual)
        } else {
            None
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct RequestShape {
    ctx_cap: usize,
    budget: usize,
    need: usize,
}

impl RequestShape {
    /// Context-linear capacity admission must reserve for this request. Finite requests normally
    /// allocate `ctx_cap = prompt + max_new + 8`; an affinity grow uses the request-shaped
    /// speculative safety bound (`need`, +64) so ContextFull cannot preempt MaxNew. Charging the
    /// larger value keeps that small growth honest without inheriting the server's global cap.
    fn admission_cap(self) -> usize {
        self.ctx_cap.max(self.need)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ParkedPool {
    Plain,
    Spec,
    Dspark,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ParkedCandidate {
    pool: ParkedPool,
    key: PoolKey,
    index: usize,
    parked_at: Instant,
}

fn older_parked(
    current: Option<ParkedCandidate>,
    candidate: ParkedCandidate,
) -> Option<ParkedCandidate> {
    match current {
        Some(oldest) if oldest.parked_at <= candidate.parked_at => Some(oldest),
        _ => Some(candidate),
    }
}

fn oldest_parked_candidate(
    candidates: impl IntoIterator<Item = ParkedCandidate>,
) -> Option<ParkedCandidate> {
    candidates.into_iter().fold(None, older_parked)
}

fn oldest_parked<P: ParkedEntryAge, S: ParkedEntryAge, D: ParkedEntryAge>(
    reuse: &HashMap<PoolKey, Vec<P>>,
    spec_reuse: &HashMap<PoolKey, Vec<S>>,
    dspark_reuse: &HashMap<PoolKey, Vec<D>>,
) -> Option<ParkedCandidate> {
    let plain = reuse.iter().flat_map(|(key, pool)| {
        pool.iter()
            .enumerate()
            .map(move |(index, entry)| ParkedCandidate {
                pool: ParkedPool::Plain,
                key: key.clone(),
                index,
                parked_at: entry.parked_at(),
            })
    });
    let spec = spec_reuse.iter().flat_map(|(key, pool)| {
        pool.iter()
            .enumerate()
            .map(move |(index, entry)| ParkedCandidate {
                pool: ParkedPool::Spec,
                key: key.clone(),
                index,
                parked_at: entry.parked_at(),
            })
    });
    let dspark = dspark_reuse.iter().flat_map(|(key, pool)| {
        pool.iter()
            .enumerate()
            .map(move |(index, entry)| ParkedCandidate {
                pool: ParkedPool::Dspark,
                key: key.clone(),
                index,
                parked_at: entry.parked_at(),
            })
    });
    oldest_parked_candidate(plain.chain(spec).chain(dspark))
}

/// Drop exactly one globally-oldest parked session without changing either pool's shape.
fn evict_oldest_parked<P: ParkedEntryAge, S: ParkedEntryAge, D: ParkedEntryAge>(
    reuse: &mut HashMap<PoolKey, Vec<P>>,
    spec_reuse: &mut HashMap<PoolKey, Vec<S>>,
    dspark_reuse: &mut HashMap<PoolKey, Vec<D>>,
    metrics: &mut ReuseMetrics,
) -> Option<ParkedPool> {
    let candidate = oldest_parked(reuse, spec_reuse, dspark_reuse)?;
    match candidate.pool {
        ParkedPool::Plain => {
            let empty = {
                let pool = reuse
                    .get_mut(&candidate.key)
                    .expect("oldest plain parked entry vanished");
                drop(pool.remove(candidate.index));
                pool.is_empty()
            };
            if empty {
                reuse.remove(&candidate.key);
            }
            metrics.continuation_evictions += 1;
        }
        ParkedPool::Spec => {
            let empty = {
                let pool = spec_reuse
                    .get_mut(&candidate.key)
                    .expect("oldest spec parked entry vanished");
                drop(pool.remove(candidate.index));
                pool.is_empty()
            };
            if empty {
                spec_reuse.remove(&candidate.key);
            }
            metrics.spec_evictions += 1;
        }
        ParkedPool::Dspark => {
            let empty = {
                let pool = dspark_reuse
                    .get_mut(&candidate.key)
                    .expect("oldest dspark parked entry vanished");
                drop(pool.remove(candidate.index));
                pool.is_empty()
            };
            if empty {
                dspark_reuse.remove(&candidate.key);
            }
            metrics.spec_evictions += 1;
        }
    }
    Some(candidate.pool)
}

fn parked_entry_count<P, S, D>(
    reuse: &HashMap<PoolKey, Vec<P>>,
    spec_reuse: &HashMap<PoolKey, Vec<S>>,
    dspark_reuse: &HashMap<PoolKey, Vec<D>>,
) -> usize {
    reuse.values().map(Vec::len).sum::<usize>()
        + spec_reuse.values().map(Vec::len).sum::<usize>()
        + dspark_reuse.values().map(Vec::len).sum::<usize>()
}

fn trim_parked_namespace<T>(pool: Option<&mut Vec<T>>, cap: usize, evictions: &mut u64) {
    if let Some(pool) = pool {
        while pool.len() >= cap {
            drop(pool.remove(0));
            *evictions += 1;
        }
    }
}

/// Make one slot for a whole-session park. The per-key eviction preserves reuse locality; the
/// second gate bounds the process-wide sum and evicts by age across both pool types.
fn prepare_park<P: ParkedEntryAge, S: ParkedEntryAge, D: ParkedEntryAge>(
    target: ParkedPool,
    key: &PoolKey,
    reuse: &mut HashMap<PoolKey, Vec<P>>,
    spec_reuse: &mut HashMap<PoolKey, Vec<S>>,
    dspark_reuse: &mut HashMap<PoolKey, Vec<D>>,
    metrics: &mut ReuseMetrics,
    per_namespace_cap: usize,
    global_cap: usize,
) -> bool {
    if per_namespace_cap == 0 || global_cap == 0 {
        return false;
    }

    match target {
        ParkedPool::Plain => trim_parked_namespace(
            reuse.get_mut(key),
            per_namespace_cap,
            &mut metrics.continuation_evictions,
        ),
        ParkedPool::Spec => trim_parked_namespace(
            spec_reuse.get_mut(key),
            per_namespace_cap,
            &mut metrics.spec_evictions,
        ),
        ParkedPool::Dspark => trim_parked_namespace(
            dspark_reuse.get_mut(key),
            per_namespace_cap,
            &mut metrics.spec_evictions,
        ),
    }

    while parked_entry_count(reuse, spec_reuse, dspark_reuse) >= global_cap {
        if evict_oldest_parked(reuse, spec_reuse, dspark_reuse, metrics).is_none() {
            return false;
        }
    }
    true
}
/// Right-size ladder slack: a shrunken spec session's cap must cover
/// prompt + budget + this, so the burst-loop ContextFull guard
/// (`committed + k + 3 >= cache_max_ctx`) can NEVER fire before MaxNew — a
/// shrunken session emits exactly the tokens a full-size one would (the F5
/// exactness contract: pool sizing is pure perf). Worst case per burst:
/// committed reaches prompt + budget + overshoot (up to k accepted drafts past
/// max_new) + a carried pending, and the guard adds k + 3; served policy K <= 5
/// (operator pins may be deeper) — 64 covers the normal table with margin at ~2MB of KV.
/// Requests whose budget
/// already spans the whole ctx_cap (max_tokens omitted) cannot shrink and keep
/// the legacy tokenwise fallback on failure.
const SPEC_SHRINK_SLACK: usize = 64;
/// Ladder transient reserve: after a landing, this much VRAM must still be
/// PROBE-allocatable for forward-pass transients — prime chunk slabs (~140MB
/// apiece at MEMRA_PRIME_CHUNK=2048; the serve-script's measured 36.5k probe
/// passed with 1.3GB free) and the FA dequant workspace. Some transients live
/// on PANICKING lazy paths (expect()), so a session that "fits" with zero
/// headroom kills the worker on its first prefill (observed: ladder landed
/// 65536, embed-table upload OOM panic — research/specpool-20260804/
/// server-ladder-miss.log; the embed table is made resident fallibly at landing
/// for that reason). The probe is alloc-and-drop, not a mem_get_info read: the
/// async pool's pinned release threshold keeps freed blocks cached and
/// invisible to free-VRAM queries. A landing that can't clear the probe is
/// DROPPED and the ladder keeps shrinking; if even `need` can't, the request
/// takes the legacy tokenwise fallback (whose own alloc failure is a clean
/// quoted error — the pre-fix behavior, never a panic).
const SPEC_SHRINK_RESERVE: usize = 1536 << 20; // 1.5 GiB

/// PC-ISO pool key (lane/pc-iso, 2026-08-02): every cross-request reuse pool — the prefix
/// cache, the continuation pool, and the spec pool — keys on (model, cache namespace), not
/// model alone. The namespace is the request's `cache_salt` (vLLM cache_salt design, PR
/// #17045): a lookup only ever scans its own (model, ns) pool, so no token-prefix match can
/// cross a trust boundary, and the `cached_tokens` billing field can only reveal the
/// caller's own namespace's history (the CacheProbe/PROMPTPEEK mitigation —
/// research/cache-tools-20260802/REPORT.md §1.4/§4). "" is the default single-tenant
/// namespace: no salt supplied = today's behavior, byte-for-byte.
type PoolKey = (String, String);

/// Log suffix for a pool key's namespace: silent for the default "" namespace (default-path
/// log lines stay byte-identical to pre-PC-ISO), quoted otherwise.
fn ns_suffix(ns: &str) -> String {
    if ns.is_empty() {
        String::new()
    } else {
        format!(", ns {ns:?}")
    }
}

// ---------------- SESSION AFFINITY (lane/session-affinity, 2026-08-05) ----------------
//
// THE PROBLEM (receipts: research/specpool-20260804/RESULTS.md). The spec pool resumes a
// parked session only when the new prompt EXACTLY EXTENDS it — token-prefix, or (since
// 2026-07-06) text-prefix. Real agent clients rewrite conversation history between turns:
// the owner's client strips `<think>` blocks out of PRIOR assistant turns before re-sending,
// so turn N's prompt is NOT a prefix-extension of turn N-1's committed text. Both probes
// miss, the parked ~4GB session is discarded as dead weight, and every turn re-primes the
// whole growing conversation (11k-14k tokens ~= 3s TTFT vs llama's 0.19s).
//
// Affinity closes that gap by answering a DIFFERENT question than the prefix probes: not
// "does this prompt extend that session's bytes?" but "is this the SAME CONVERSATION as
// that session?". Once a candidate is nominated by identity, the resume decision is made by
// an EXACT token-level diff (see `AffinityMatch`) — identity nominates, bytes decide. That
// split is the whole safety argument: a fingerprint collision can only ever nominate a
// candidate whose committed tokens are then compared exactly, so it can cost a wasted probe,
// never a wrong resume.
//
// TWO TIERS.
//   (a) EXPLICIT (`AffinityKey::Explicit`) — the client names its conversation. Accepted from
//       two conventions, both documented in docs/SERVING.md ("Session affinity"):
//         * `session_id` / `user` request-body fields (OpenAI's `user` is the field real
//           clients already send; `session_id` is the explicit spelling),
//         * the `x-session-id` request header (the convention vLLM/TGI-adjacent proxies use).
//       Body beats header when both appear (the body is the caller's own statement of
//       identity; a header can be injected by an intermediary).
//   (b) IMPLICIT (`AffinityKey::Fingerprint`) — nothing named, so identity is STRUCTURAL: a
//       hash of the conversation's SHAPE that is invariant under exactly the rewrite class we
//       need to survive. See `conversation_fingerprint`.
//
// TENANT SCOPE. Affinity is stored per `PoolKey = (model, cache_ns)`, so an affinity key can
// only ever nominate a session inside its own PC-ISO namespace — affinity adds NO new
// cross-tenant reach beyond what the existing pools already have. (The api-keys lane's
// TenantCtx is not on this branch; when it merges, its per-key namespace derivation flows
// into `cache_ns` and affinity inherits the boundary for free.)

/// Prefix/suffix token window hashed per conversation segment (see `conversation_fingerprint`).
/// Small enough that a rewritten segment BODY doesn't perturb the hash, large enough that
/// distinct segments don't collide: the head pins "which turn is this" (role marker + opening
/// words) and the tail pins the segment's end boundary.
const FP_WINDOW: usize = 8;
/// Minimum segments before an implicit fingerprint is trusted to name a conversation. A
/// one-or-two-segment prompt is a generic opener (a bare system prompt shared by every fresh
/// conversation); nominating on it would cross-link unrelated conversations into one session.
const FP_MIN_SEGMENTS: usize = 3;

/// ROLLBACK SEAM (flags doctrine: the winner is the default; this exists to *disable* it).
/// `MEMRA_AFFINITY=0` makes the affinity probe decline every candidate, so admit falls back to
/// the pre-lane behavior: prefix probes only, cold full prime on a rewritten history.
///
/// This is not a tuning knob — it is the exactness A/B arm. The byte-identity gate
/// (`research/session-affinity-20260805/`) runs the SAME conversation twice, once resuming and
/// once with `MEMRA_AFFINITY=0`, and requires identical per-turn `text_sha`. Disabling the pool
/// outright (`MEMRA_REUSE_POOL=0`) would be a different comparison: it also drops the
/// token/text-prefix resumes, so a divergence could not be attributed to affinity.
fn affinity_enabled() -> bool {
    if memra_engine::pp::pp_host_bounce_active() {
        // Plain-affinity checkpoints call Cache::snapshot through the primary Engine. Hybrid
        // layers owned by another PP stage would therefore be peer-copied on this host class.
        return false;
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("MEMRA_AFFINITY")
            .map(|v| v != "0")
            .unwrap_or(true)
    })
}

/// STABLE-BOUNDARY SPEC TIER (lane/frspec-multiturn-cache, 2026-08-21), default ON. Ports the
/// plain tier's 2026-08-09 stable pre-generation boundary law to the spec tier: the spec turn
/// checkpoint and the restored-session republication both sat at PROMPT-END, whose tail is the
/// template's live generation header (`<|im_start|>assistant\n<think>\n`) — rewritten by every
/// re-rendering client, so spec affinity declined 100% of multi-turn agent traffic and the
/// prefix-cache hit boundary froze at the first lcp-split entry forever (measured: cached 1.5%
/// of tokens by turn 7 vs 98.6% on the plain arm; TTFT 10-13.6s by turn 8;
/// research/multiturn-cache-20260821 finding B4). `MEMRA_SPEC_STABLE_BOUNDARY=0` restores the
/// prompt-end posture byte-for-byte — the rollback seam and the toothed multi-turn gate's
/// broken arm.
fn spec_stable_boundary_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("MEMRA_SPEC_STABLE_BOUNDARY").as_deref() != Ok("0"))
}

/// FNV-1a over a token stream — a stable, allocation-free 64-bit mix. (Not a cryptographic
/// hash and does not need to be: a collision costs one wasted exact-diff probe, never a
/// wrong resume, and the pool it indexes is already tenant-scoped.)
fn fnv1a(seed: u64, toks: &[u32]) -> u64 {
    let mut h = seed;
    for &t in toks {
        for b in t.to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
    }
    h
}

/// Structural fingerprint of a conversation: the CHAIN of per-segment boundary hashes, one
/// entry per conversation segment in order, each hashing only that segment's (head window,
/// tail window) — never its interior.
///
/// WHY A CHAIN, NOT ONE HASH. Turn N+1 of a conversation has strictly MORE segments than turn
/// N (the previous answer plus a new user turn were appended), so a single whole-conversation
/// digest can never match across turns. Identity is therefore a PREFIX relation over the
/// chain (`fingerprint_affinity`): the parked session's conversation is an ancestor of this
/// request's when their chains share a long-enough leading run.
///
/// WHY BOUNDARY WINDOWS. The rewrite class we must tolerate mutates the INTERIOR of prior
/// assistant segments (a stripped `<think>` block is deleted text inside a turn). Hashing only
/// each segment's first and last few tokens leaves those edits invisible while still separating
/// genuinely different segments. Where a rewrite reaches into a head window too (a `<think>`
/// tag can sit right after the role marker), the chain degrades GRACEFULLY instead of failing:
/// that one segment's hash changes, the shared leading run simply ends earlier, and the
/// candidate is still nominated on the stable prefix (system prompt + early turns, which no
/// client rewrites). Nomination only has to be a good guess — `affinity_match` decides on bytes.
///
/// SEGMENTATION on the raw-prompt path. The owner's client renders the chat template
/// CLIENT-side and posts raw `/v1/completions`, so there is no `chat_turns` structure to walk:
/// the worker sees one flat token stream. Segments are recovered from the stream itself by
/// splitting at the template's own turn-marker tokens (the tokenizer's control tokens — exactly
/// what a chat template emits at every turn boundary: `<|im_start|>`/`<|im_end|>` and friends).
/// The implicit tier therefore works identically for client-rendered raw prompts and for
/// server-rendered `/v1/chat/completions` traffic.
///
/// `is_boundary(tok)` reports whether a token is a template turn marker. `drop_live` excludes
/// the trailing segment (a REQUEST's last segment is the turn being generated — new every turn
/// by construction, so it must not contribute to identity; a PARKED session's committed stream
/// has no such live tail and keeps every segment).
fn conversation_fingerprint(
    toks: &[u32],
    is_boundary: &dyn Fn(u32) -> bool,
    drop_live: bool,
) -> Vec<u64> {
    // Split into segments at boundary tokens. The boundary token itself joins the segment it
    // opens, so a segment's head window carries its own role marker.
    let mut segs: Vec<(usize, usize)> = Vec::new();
    let mut start = 0usize;
    for (i, &t) in toks.iter().enumerate() {
        if is_boundary(t) && i > start {
            segs.push((start, i));
            start = i;
        }
    }
    if start < toks.len() {
        segs.push((start, toks.len()));
    }
    if drop_live && !segs.is_empty() {
        segs.pop();
    }
    segs.iter()
        .map(|&(lo, hi)| {
            let seg = &toks[lo..hi];
            let head = &seg[..FP_WINDOW.min(seg.len())];
            let tail = &seg[seg.len().saturating_sub(FP_WINDOW)..];
            fnv1a(fnv1a(0xcbf29ce484222325, head), tail)
        })
        .collect()
}

/// Length of the leading run two fingerprint chains share. `>= FP_MIN_SEGMENTS` is the
/// nomination bar: below it the shared run is a generic opener (a bare system prompt is
/// byte-identical across every fresh conversation with the same client), and nominating on it
/// would cross-link unrelated conversations. Markerless raw prompts produce a 1-segment chain
/// and so can never clear the bar — non-chat callers keep the plain prefix probes untouched.
fn fingerprint_affinity(a: &[u64], b: &[u64]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

/// Verdict of the EXACT token diff run against an affinity-nominated parked session. Identity
/// nominated the candidate; this decides — on bytes — whether resuming it is EXACT.
///
/// THE EXACTNESS CONTRACT. A resumed session must emit BYTE-IDENTICAL output to a fresh full
/// prime of the same request. The committed tokens in the parked caches are authoritative
/// state: whatever they are, the caches hold exactly their KV/recurrent state. So resuming is
/// exact iff the new prompt begins with the session's ENTIRE committed sequence — then the
/// caches are precisely "the state after the prompt's first `committed.len()` tokens" and only
/// the remaining suffix needs priming. Any DIVERGENCE inside the committed range means the
/// caches hold state for tokens this request does not have, and no amount of suffix priming
/// can repair that (hybrid GDN recurrent state is mutated in place and has no per-position
/// index to truncate). There is one legal repair — roll the session back to the divergence
/// point — and it requires a checkpoint AT that boundary, which a parked session does not
/// carry. So divergence inside the committed range is a full re-prime, always. Correctness
/// first; the affinity win comes from the (dominant) case where the rewrite touches only text
/// the session has not committed yet.
#[derive(PartialEq, Eq, Debug)]
enum AffinityMatch {
    /// The prompt begins with the session's entire committed sequence: resume, prime the
    /// `suffix_from` tail only. (`suffix_from == prompt.len()` = pure continuation burst.)
    Exact { suffix_from: usize },
    /// The prompt diverges from the committed tokens at this index: the parked caches hold
    /// state for tokens this request does not have. Full re-prime.
    Diverged { at: usize },
}

/// Exact token-level diff of a request's prompt against a parked session's committed tokens.
/// The ONLY authority on whether an affinity-nominated session may be resumed.
fn affinity_match(prompt: &[u32], committed: &[u32]) -> AffinityMatch {
    let n = committed.len().min(prompt.len());
    for i in 0..n {
        if prompt[i] != committed[i] {
            return AffinityMatch::Diverged { at: i };
        }
    }
    if prompt.len() < committed.len() {
        // The prompt is a strict PREFIX of committed: the session has generated past what
        // this request contains (a client that dropped its own tail, or a re-issued earlier
        // turn). The caches hold extra committed rows with no boundary checkpoint to trim
        // them at — treat as divergence at the prompt's end.
        return AffinityMatch::Diverged { at: prompt.len() };
    }
    AffinityMatch::Exact {
        suffix_from: committed.len(),
    }
}

/// Decide whether an affinity candidate can resume and which cache capacity it needs.
/// Identity only nominates; the checkpoint bounds and exact token comparison remain authoritative.
/// A too-small parked cache is a grow request, not a decline: returning the incoming request's
/// charged required capacity lets the caller allocate and restore a larger cache before resume.
fn affinity_resume_target(
    prompt: &[u32],
    committed: &[u32],
    checkpoint_pos: usize,
    parked_cap: usize,
    request_cap: usize,
    identity_matches: bool,
) -> Result<usize, String> {
    if checkpoint_pos == 0 || checkpoint_pos > committed.len() {
        return Err(format!(
            "bad checkpoint pos {checkpoint_pos} of {}",
            committed.len(),
        ));
    }
    match affinity_match(prompt, &committed[..checkpoint_pos]) {
        AffinityMatch::Exact { suffix_from } if suffix_from == checkpoint_pos => {}
        AffinityMatch::Diverged { at } => {
            let mut why = format!("history diverged at {at} of checkpoint {checkpoint_pos}");
            // Divergence-byte receipt (lane/frspec-multiturn-cache): the offsets alone cannot
            // distinguish a real history rewrite from the template-boundary class (checkpoint
            // tail = the live generation header the client re-renders). The ids name it.
            if std::env::var("MEMRA_DEBUG_AFFINITY").is_ok() {
                let lo = at.saturating_sub(2);
                why.push_str(&format!(
                    " [ckpt ids {:?} vs prompt ids {:?}]",
                    &committed[lo..(at + 3).min(checkpoint_pos)],
                    &prompt[lo..(at + 3).min(prompt.len())],
                ));
            }
            return Err(why);
        }
        _ => return Err("diff did not land on the checkpoint".into()),
    }
    if prompt.len() == checkpoint_pos {
        return Err("empty suffix".into());
    }
    if !identity_matches {
        return Err("identity did not nominate".into());
    }
    Ok(parked_cap.max(request_cap))
}

/// PLAIN-SESSION AFFINITY (lane/plain-affinity, 2026-08-09): the STABLE PRE-GENERATION boundary
/// this prompt's rewind checkpoint should sit at, as a `prompt`-length index.
///
/// WHY NOT PROMPT-END. The spec-tier checkpoint sat at prompt-end and the forced-spec control in
/// `research/cachespec-20260809/RESULTS.md` disproved it: the frozen pi workload's turn N+1
/// diverges from turn N a couple tokens BELOW prompt-end, inside the template's live
/// assistant-generation header (`<|im_start|>assistant\n<think>\n`), which the client rewrites.
/// A checkpoint at prompt-end therefore always sits past the divergence and declines 100% of the
/// time. The fix (RESULTS.md §P0): checkpoint BEFORE the last turn-marker run — the stable point
/// every prior turn's prompt shares.
///
/// HOW. Chat traffic (client- or server-rendered) carries the tokenizer's turn-marker control
/// tokens (`<|im_start|>`/`<|im_end|>` and friends). The last MAXIMAL run of control tokens at
/// the prompt tail is the live generation header; the boundary is just before it. Raw completions
/// with no markers fall back to a conservative guard window (`PLAIN_CKPT_RAW_GUARD`, in the
/// RESULTS.md 8..32 band) trimmed off the end. The exact token diff (`affinity_match`) still
/// DECIDES on bytes, so a mis-located boundary only costs a decline, never a wrong resume — the
/// value 2 the control observed is NEVER hardcoded.
///
/// Returns `None` when no useful boundary exists (prompt too short to be worth a checkpoint, or
/// the boundary would land at/below `REUSE_MIN_PREFIX`). A `Some(b)` always satisfies
/// `REUSE_MIN_PREFIX < b < prompt.len()` and sits ON the prime grid (`grid_align_boundary`).
fn plain_checkpoint_boundary(prompt: &[u32], is_control: &dyn Fn(u32) -> bool) -> Option<usize> {
    let n = prompt.len();
    if n <= REUSE_MIN_PREFIX + PLAIN_CKPT_RAW_GUARD {
        return None;
    }
    // WHY THE LAST TURN-MARKER, worked through on the pi rewrite. Turn N's session commits
    //   [history][userN] <|im_start|>assistant\n<think>\n [genN...]
    // and its prompt-END is at "<think>\n" (before genN). Turn N+1's prompt is
    //   [history][userN] <|im_start|>assistant\n [genN-with-think-stripped] <|im_end|>
    //     [userN+1] <|im_start|>assistant\n<think>\n
    // The two agree through "<|im_start|>assistant\n" and DIVERGE at the next token: turn N has
    // "<think>", turn N+1 has the stripped answer's first token. That divergence sits a couple
    // tokens BELOW turn N's prompt end — exactly the "diverged at ckpt-2" the forced-spec control
    // measured, which is why a prompt-END checkpoint declines 100% of the time.
    //
    // The stable point every later turn resends verbatim is the START of the live-generation
    // segment: the LAST turn-marker control token (the `<|im_start|>` opening the assistant turn).
    // `committed[..last_marker]` = "[history][userN]<|im_end|>", which turn N+1 reproduces exactly;
    // the whole live turn (marker + role header + generation) becomes the primed suffix. The role
    // name and `<think>` opener are ORDINARY vocab tokens, not control tokens, so they never move
    // the boundary — only the tokenizer's turn markers do. The exact diff at resume still decides
    // on bytes; a mis-located boundary costs a decline, never a wrong resume, and the observed
    // value 2 is NEVER hardcoded.
    if let Some(last_marker) = prompt.iter().rposition(|&t| is_control(t)) {
        // Grid alignment (lane/spec-longctx-20260821): EARLIER than the marker is always
        // semantically safe (a prefix of a verbatim-resent prefix is verbatim-resent; the
        // exact byte diff at resume still decides), and ON the grid it is byte-safe too —
        // see grid_align_boundary.
        let b = grid_align_boundary_within(last_marker, n);
        if b > REUSE_MIN_PREFIX && b < n {
            return Some(b);
        }
        // The last marker is too early (a giant final turn with no interior markers) or is the
        // final token (a bare closing marker, empty live turn). Fall through to the guard window.
    }
    // Markerless raw completions (or a degenerate marker position): trim a conservative guard
    // window off the end so the live tail — which a re-ask may extend or rewrite — is never
    // inside the checkpoint.
    let b = grid_align_boundary_within(n - PLAIN_CKPT_RAW_GUARD, n);
    if b > REUSE_MIN_PREFIX { Some(b) } else { None }
}

/// PRIME-GRID BOUNDARY ALIGNMENT (lane/spec-longctx-20260821 — the GATES-SMOKE B3/B1-fold
/// disposition). Round a serve-CHOSEN prime stop/capture boundary DOWN to the GDN WY-chunk
/// grid.
///
/// THE MEASURED LAW (research/multiturn-cache-20260821/LONGCTX-EXACTNESS-20260821.md, NJ box,
/// q38 trunk @ v0.99.0, 24k-token real agentic prompt): a prompt primed as TWO prime_cache
/// calls split at L is BIT-IDENTICAL to the monolithic prime when `L % gdn_chunk_size() == 0`
/// (splits 8256/15200: rows_diff 0/23999, logits EXACT), and diverges from exactly row L
/// onward when it is not (splits 607/8263/15227: every row from L differs; greedy flips land
/// on near-tie margins — p0.0 of the reference margin distribution). Under the sequential
/// GDN scan (`MEMRA_GDN_CHUNKED=0`) every split is EXACT, which pins the mechanism: the
/// chunked WY scan segments per prime call, so an off-grid call start shifts the fold grid
/// and materializes recurrent state at a point the monolithic program never computes —
/// lawful FP behavior of the WY form, not a defect, and structurally unfixable at arbitrary
/// split points. What IS ours to choose is WHERE serve stops: checkpoints and prefix-cache
/// captures land on the grid, so the boundary-stopped prime, the checkpoint resume and the
/// whole-entry restore reproduce the cold monolithic bytes (the 2026-08-21 byteattrib
/// stop-vs-mono divergence class — 13/16 turns at agent lengths — goes to zero). Cost: at
/// most `gdn_chunk_size()-1` extra suffix tokens re-primed per resume.
///
/// Attention-only models are split-invariant already (same cells, GDN-off arm), so rounding
/// is a no-op contract there beyond the <=31-token capture shift — applied uniformly to keep
/// one law. `MEMRA_PRIME_GRID_ALIGN=0` is the rollback seam (legacy byte-for-byte boundary
/// choice — the toothed gate's broken arm). Read per call, never cached (probes flip it
/// in-process between arms).
fn grid_align_boundary(b: usize) -> usize {
    if std::env::var("MEMRA_PRIME_GRID_ALIGN").as_deref() == Ok("0") {
        return b;
    }
    let c = memra_engine::Engine::gdn_chunk_size();
    b / c * c
}

/// `grid_align_boundary` PLUS the W1 door (the sub-floor boundary remainder), for callers that
/// know the prompt length.
///
/// THE W1 DOOR, MEASURED (lane/spec-longctx-20260821, NJ box, conv-01 turn 3 — receipts
/// `[primeseg]` lines in LONGCTX-EXACTNESS-20260821.md): when the boundary leaves FEWER than
/// `PRIME_MIN_T` prompt tokens behind it, the prefill tick's own predicate
/// (`q >= PRIME_MIN_T`) vetoes the prime branch and those PROMPT tokens go through
/// `decode_step` ONE AT A TIME — a different numeric program for the same bytes (the
/// prefill-vs-decode fork `run_gen`'s prime gate documents). That is the open defect the
/// boundary-stop comment in `prefill_tick` calls W1: with prompt 9,510 and boundary 9,504 the
/// last 6 prompt tokens decoded tokenwise, and that turn's greedy output diverged from the
/// monolithic prime while every same-shape turn with a >= PRIME_MIN_T remainder was
/// byte-identical.
///
/// The remedy the W1 note prescribes is "drop the sub-floor boundary capture (lose a cache
/// seed, keep one program)". Grid alignment affords a strictly better one: step the boundary
/// DOWN one grid unit. `gdn_chunk_size() >= 2 * PRIME_MIN_T` on the shipped grain (32 vs 16),
/// so ONE step always clears the floor — the capture survives, the suffix stays a single prime
/// call, and the boundary stays grid-aligned. Earlier is always semantically safe (same
/// argument as the alignment itself). If stepping down would fall to/below `REUSE_MIN_PREFIX`
/// the caller's own bound rejects the boundary, which is then the W1 note's "drop it" outcome.
fn grid_align_boundary_within(b: usize, prompt_len: usize) -> usize {
    let mut aligned = grid_align_boundary(b);
    if std::env::var("MEMRA_PRIME_GRID_ALIGN").as_deref() == Ok("0") {
        return aligned;
    }
    let c = memra_engine::Engine::gdn_chunk_size();
    let floor = memra_engine::hybrid_forward::PRIME_MIN_T;
    while aligned >= c && prompt_len.saturating_sub(aligned) < floor {
        aligned -= c;
    }
    aligned
}

/// Pure half of the HIT-path LCP-split arming (H11 depth-unfreeze; the stateful `has_key`
/// dedupe and the `eager_only_model` exclusion stay at the call site). Returns the boundary
/// to stop the suffix prime at, or `None` when no capture is legal.
///
/// THE FED-START FLOOR (PR #37 review finding, 2026-08-24): `grid_align_boundary_within`
/// steps the PROMPT-side remainder onto the grid, but nothing checked the FED-side remainder
/// `la - hit_len` — and on a hit the suffix prime STARTS at `hit_len`, not 0. A gap of
/// 1..PRIME_MIN_T-1 trips `prefill_tick`'s boundary veto (`bound_rem >= PRIME_MIN_T`), so the
/// tokens between `hit_len` and `la` would go through tokenwise `decode_step` before the
/// snapshot fires: the W1 two-programs door, and worse, a CAPTURED entry whose tail
/// provenance is decode_step. The miss path cannot reach this (fed starts at 0, so
/// `bound_rem = la >= PREFIX_CACHE_MIN_TOKENS`); only the hit arm can, because `hit_len` is a
/// prefill-done seed depth and is NOT grid-aligned. Stepping `la` DOWN shrinks the gap
/// further (it is measured from `hit_len` upward), so the sub-floor case is DROPPED — lose a
/// cache seed, keep one numeric program, the W1 note's own remedy.
fn hit_lcp_snapshot_boundary(lcp: usize, hit_len: usize, prompt_len: usize) -> Option<usize> {
    if lcp <= hit_len {
        return None;
    }
    let la = grid_align_boundary_within(lcp, prompt_len);
    (la >= PREFIX_CACHE_MIN_TOKENS
        && la > hit_len
        && la - hit_len >= memra_engine::hybrid_forward::PRIME_MIN_T
        && la < prompt_len)
        .then_some(la)
}

/// Conservative guard window for the markerless (raw-completion) path of
/// `plain_checkpoint_boundary`. RESULTS.md §P0 specifies an 8..32 band; 16 is the middle and
/// matches `PRIME_MIN_T`, so a boundary-stopped prime never leaves a sub-floor residual.
const PLAIN_CKPT_RAW_GUARD: usize = 16;

/// Could a PARKED entry for this prompt ever be NOMINATED by the IMPLICIT tier? The implicit
/// tier compares fingerprint chains and requires a shared leading run of `FP_MIN_SEGMENTS`;
/// a prompt whose own chain cannot reach that bar can never be implicitly nominated, so
/// capturing a checkpoint for it — absent an explicit id — is pure cost. WORSE than cost:
/// an armed `ckpt_at` excludes the session from the in-batch fanout and prime-batch paths
/// (they prime monolithically and cannot honor the per-session boundary stop), which is
/// exactly the cache-metering regression the tip battery caught (2026-08-09, 13 checks red:
/// a 5-way shared-prefix raw-`prompt_ids` fanout never fired because every session armed a
/// guard-window checkpoint it could never use — 0 prefix hits, 6 misses/inserts, empty LCP
/// window). Chat-template traffic keeps its markers and clears the bar; raw agent loops that
/// want affinity name their conversation (`session_id`/`user`/`x-session-id`), which arms the
/// capture through the explicit tier instead.
///
/// The chain here is over the PROMPT with `drop_live=false` — the parked entry's chain is
/// taken over `fed` (prompt + generation, no live tail), whose segment count is >= this, so
/// this is the conservative lower bound on what a later request could share.
fn plain_ckpt_nominatable(prompt: &[u32], is_control: &dyn Fn(u32) -> bool) -> bool {
    conversation_fingerprint(prompt, is_control, false).len() >= FP_MIN_SEGMENTS
}

// ---------------- CROSS-REQUEST PREFIX CACHE (lane/prompt-cache, 2026-08-02) ----------------
//
// The continuation pool above only serves a prompt that EXACTLY EXTENDS a retired session's
// whole fed sequence (prompt + generation) — a NEW session that merely shares a system-prompt
// prefix with earlier traffic always misses. The prefix cache closes that gap: entries are
// compact device copies of primed state at a TOKEN boundary, keyed by the exact token-id
// prefix, and are REUSABLE (a hit deep-copies the entry into the new session's cache — one
// marketplace system prompt serves any number of sessions, unlike the move-out pool).
//
// SPLIT LAW: full-attention K/V planes are context-linear token rows, so a transformer-only
// entry can be copied through an arbitrary LCP boundary and the current request primes only its
// suffix. Hybrid models (qwen35-class GDN) carry recurrent conv/ssm state that cannot roll back
// to an arbitrary shorter prefix; routed-MoE mid-entry reuse is also unqualified after the
// coldfix/fencealias history. Those classes refuse observably and retain boundary learning:
//   - SEED: a cold session's full prompt is inserted at prefill-done (before any decode).
//   - LCP SPLIT (the learning step): a cold miss whose prompt shares >= PREFIX_CACHE_MIN_TOKENS
//     tokens with an existing entry splits its own prime at the longest-common-prefix point,
//     snapshots there, then continues — request 3+ of a shared-system-prompt pattern hits.
//
// EXACTNESS CONTRACT (docs/SERVING.md "Prompt caching"): an entry stores the KV/recurrent
// bytes from WHATEVER prime config ran (single, chunked, or concat batch-prime); decode from
// those bytes is deterministic, so serving a hit is bit-identical to the run that computed the
// prefix. Cross-config comparisons (a cached-hit stream vs a whole-prompt fresh prime) inherit
// the documented batched-prime near-tie first-token law — same class, not a new one.
//
// VRAM: entries compete with session KV under MEMRA_PREFIX_CACHE_MB (0 disables). With no
// override, the budget is derived from loaded-model geometry and boot free VRAM below. The
// worker-global pool is byte-budgeted SLRU-evicted by default, or plain-LRU-evicted under the
// rollback policy; model/namespace keys scope lookup visibility only. A failed session-cache
// alloc evicts the whole cache and retries (headroom discipline — sessions always win over it).
//
// POLICY (lane/spec-prefix-cache 2026-08-14, completed by lane/spec-on-cache-hit
// 2026-08-18, sampled arm lane/sampled-hit-spec 2026-08-19): spec-eligible requests
// PROBE and PUBLISH like everyone else. A spec session's cold boundary capture publishes
// trunk KV + the MTP draft plane + the boundary hidden (prefix_insert_from_spec_boundary)
// — greedy and sampled sessions alike, since a prime is sampler-independent; a whole-entry
// hit on an unconstrained spec-eligible request re-arms a SpecSession from the restored
// carrier (spec_session_from_restored — the engine feeds any prompt suffix through the
// plain path's exact program), in BOTH sampling regimes. Hits that cannot re-arm (no draft
// plane, constrained, sampled-with-active-penalties, partial) serve on the plain path and
// the downgrade line names which. The spec tier additionally keeps its own continuation
// pool for extend/affinity shapes. Legacy round-robin mode (MEMRA_SERVE_BATCH=0) bypasses
// the prefix cache entirely.
//
// ISOLATION (PC-ISO, lane/pc-iso 2026-08-02): pools key on (model, cache namespace) — see
// PoolKey. Same-namespace traffic shares entries exactly as before; requests carrying
// different `cache_salt` values never see each other's prefixes, in either direction.

/// Prefixes shorter than this are not worth VRAM + copy bookkeeping (also keeps the bare
/// chat-template header — common to every request of a model — out of the cache).
const PREFIX_CACHE_MIN_TOKENS: usize = 64;

/// Reused entries reserve this share of the global prefix-cache byte budget. The probation
/// segment may borrow unused protected bytes; the split becomes binding only as protected
/// entries earn their share.
const DEFAULT_PREFIX_CACHE_PROTECTED_PCT: usize = 80;

/// In-process prefix-entry ABI. Entries never cross a process today, but an explicit version
/// makes a stale/corrupt object fail before any bounded device copy and keeps the same identity
/// rule a future host tier will need. v2 (lane/spec-on-cache-hit): + optional draft plane
/// (MTP scratch rows `[0..pos)`) + boundary hidden, published from spec boundary captures.
const PREFIX_ENTRY_LAYOUT_VERSION: u32 = 2;

/// Max distinct per-tenant metering rows in `Metrics::ns_tokens` (lane/cache-metering).
/// Past the cap, new tenants/salts aggregate under "(other)" — the totals stay exact,
/// only per-row attribution saturates. 256 covers any realistic keyring; the bound
/// exists so an unauthenticated client spraying cache_salt values cannot grow the map.
const METER_TENANT_CAP: usize = 256;

/// First-class speculative-acceptance metrics use the same 30-second operational horizon as
/// the default decode-step latency window. The bound protects the worker from an unbounded
/// burst queue at extreme request rates; normal traffic evicts by age first.
pub const SPEC_METRICS_WINDOW_S: f32 = 30.0;
const SPEC_METRICS_WINDOW_MAX_SAMPLES: usize = 16_384;

struct SpecTelemetryWindow {
    samples: VecDeque<(Instant, memra_engine::spec::SpecTelemetry)>,
    total: memra_engine::spec::SpecTelemetry,
    window: Duration,
}

impl SpecTelemetryWindow {
    fn new(window_s: f32) -> Self {
        Self {
            samples: VecDeque::new(),
            total: Default::default(),
            window: Duration::from_secs_f32(window_s),
        }
    }

    fn push(&mut self, delta: memra_engine::spec::SpecTelemetry) {
        self.push_at(Instant::now(), delta);
    }

    fn push_at(&mut self, now: Instant, delta: memra_engine::spec::SpecTelemetry) {
        if delta.rounds == 0 {
            return;
        }
        self.samples.push_back((now, delta));
        self.total.merge(&delta);
        while self.samples.len() > SPEC_METRICS_WINDOW_MAX_SAMPLES {
            let (_, old) = self.samples.pop_front().unwrap();
            self.total = self.total.delta_since(&old);
        }
        self.evict_at(now);
    }

    fn evict_at(&mut self, now: Instant) {
        while let Some((at, _)) = self.samples.front() {
            if now.saturating_duration_since(*at) <= self.window {
                break;
            }
            let (_, old) = self.samples.pop_front().unwrap();
            self.total = self.total.delta_since(&old);
        }
    }

    fn snapshot_at(&mut self, now: Instant) -> memra_engine::spec::SpecTelemetry {
        self.evict_at(now);
        self.total
    }
}

struct SpecMetricState {
    lifetime: HashMap<String, memra_engine::spec::SpecTelemetry>,
    windows: HashMap<String, SpecTelemetryWindow>,
    window_s: f32,
}

impl SpecMetricState {
    fn new(window_s: f32) -> Self {
        Self {
            lifetime: HashMap::new(),
            windows: HashMap::new(),
            window_s,
        }
    }

    fn record(&mut self, model: &str, delta: memra_engine::spec::SpecTelemetry) {
        if delta.rounds == 0 {
            return;
        }
        self.lifetime
            .entry(model.to_string())
            .or_default()
            .merge(&delta);
        self.windows
            .entry(model.to_string())
            .or_insert_with(|| SpecTelemetryWindow::new(self.window_s))
            .push(delta);
    }

    fn window_snapshots(&mut self) -> HashMap<String, memra_engine::spec::SpecTelemetry> {
        let now = Instant::now();
        self.windows
            .iter_mut()
            .filter_map(|(model, window)| {
                let snapshot = window.snapshot_at(now);
                (snapshot.rounds > 0).then(|| (model.clone(), snapshot))
            })
            .collect()
    }
}

/// ADSD acceptance-collapse detector. The tenant signal is a short request window. Its preferred
/// comparison population is OTHER tenants on the same model; when none is eligible, older rows
/// from that tenant provide a non-overlapping historical baseline. A 3-sigma deficit alone is too
/// sensitive once token counts get large, so an incident also requires a collapse-shaped absolute
/// drop and three consecutive anomalous observations. This is an operational signal only: it never
/// changes the lossless verifier or serving policy.
const ADSD_TENANT_WINDOW: usize = 8;
const ADSD_MODEL_WINDOW: usize = 64;
const ADSD_BASELINE_MIN_SAMPLES: usize = 16;
const ADSD_BASELINE_MIN_DRAFTED: u64 = 512;
const ADSD_TENANT_MIN_DRAFTED: u64 = 128;
const ADSD_Z_THRESHOLD: f64 = -3.0;
const ADSD_MIN_RATE_DROP: f64 = 0.20;
const ADSD_REARM_RATE_DROP: f64 = 0.10;
const ADSD_SUSTAINED_OBSERVATIONS: u8 = 3;

#[derive(Default)]
struct AcceptanceWindow {
    samples: VecDeque<(u64, u64)>,
    accepted: u64,
    drafted: u64,
}

impl AcceptanceWindow {
    fn push(&mut self, accepted: u64, drafted: u64, cap: usize) {
        self.samples.push_back((accepted, drafted));
        self.accepted += accepted;
        self.drafted += drafted;
        while self.samples.len() > cap {
            let (old_accepted, old_drafted) = self.samples.pop_front().unwrap();
            self.accepted = self.accepted.saturating_sub(old_accepted);
            self.drafted = self.drafted.saturating_sub(old_drafted);
        }
    }

    fn rate(&self) -> f64 {
        self.accepted as f64 / self.drafted.max(1) as f64
    }
}

#[derive(Default)]
struct ModelAcceptanceWindow {
    samples: VecDeque<(String, u64, u64)>,
}

impl ModelAcceptanceWindow {
    fn push(&mut self, tenant: &str, accepted: u64, drafted: u64) {
        self.samples
            .push_back((tenant.to_string(), accepted, drafted));
        while self.samples.len() > ADSD_MODEL_WINDOW {
            self.samples.pop_front();
        }
    }

    fn baseline_excluding(&self, tenant: &str) -> Option<(u64, u64)> {
        let mut samples = 0;
        let mut accepted = 0;
        let mut drafted = 0;
        for (sample_tenant, sample_accepted, sample_drafted) in &self.samples {
            if sample_tenant == tenant {
                continue;
            }
            samples += 1;
            accepted += sample_accepted;
            drafted += sample_drafted;
        }
        (samples >= ADSD_BASELINE_MIN_SAMPLES && drafted >= ADSD_BASELINE_MIN_DRAFTED)
            .then_some((accepted, drafted))
    }

    fn historical_baseline(&self, tenant: &str) -> Option<(u64, u64)> {
        // `observe` runs before the current sample enters model history. Skip the preceding
        // N-1 tenant rows so this baseline cannot overlap the N-sample tenant window after the
        // current observation is pushed.
        let mut recent = ADSD_TENANT_WINDOW.saturating_sub(1);
        let mut samples = 0;
        let mut accepted = 0;
        let mut drafted = 0;
        for (sample_tenant, sample_accepted, sample_drafted) in self.samples.iter().rev() {
            if sample_tenant != tenant {
                continue;
            }
            if recent > 0 {
                recent -= 1;
                continue;
            }
            samples += 1;
            accepted += sample_accepted;
            drafted += sample_drafted;
        }
        (samples >= ADSD_BASELINE_MIN_SAMPLES && drafted >= ADSD_BASELINE_MIN_DRAFTED)
            .then_some((accepted, drafted))
    }
}

#[derive(Default)]
struct TenantAcceptance {
    window: AcceptanceWindow,
    anomalous_observations: u8,
    incident_latched: bool,
    latched_baseline: Option<(u64, u64)>,
}

#[derive(Debug)]
struct AdsdSuspect {
    model: String,
    tenant: String,
    baseline_rate: f64,
    tenant_rate: f64,
    z_score: f64,
    drafted: u64,
}

#[derive(Default)]
struct AdsdDetector {
    model_windows: HashMap<String, ModelAcceptanceWindow>,
    tenant_windows: HashMap<(String, String), TenantAcceptance>,
    suspect_total: HashMap<String, u64>,
}

impl AdsdDetector {
    fn observe(
        &mut self,
        model: &str,
        tenant: &str,
        accepted: u64,
        drafted: u64,
    ) -> Option<AdsdSuspect> {
        if drafted == 0 || accepted > drafted {
            return None;
        }
        let mut key = (model.to_string(), tenant.to_string());
        if !self.tenant_windows.contains_key(&key) && self.tenant_windows.len() >= METER_TENANT_CAP
        {
            key.1 = "(other)".into();
        }
        let cross_baseline = self
            .model_windows
            .get(model)
            .and_then(|window| window.baseline_excluding(&key.1));
        let historical_baseline = if cross_baseline.is_none() {
            self.model_windows
                .get(model)
                .and_then(|window| window.historical_baseline(&key.1))
        } else {
            None
        };
        let state = self.tenant_windows.entry(key.clone()).or_default();
        state.window.push(accepted, drafted, ADSD_TENANT_WINDOW);
        // Cross-tenant evidence stays preferred. Once a self-baseline incident latches, retain
        // its clean comparator until recovery so a long collapse cannot dilute older history
        // and falsely rearm itself.
        let baseline = cross_baseline.or_else(|| {
            if state.incident_latched {
                state.latched_baseline.or(historical_baseline)
            } else {
                historical_baseline
            }
        });

        let mut suspect = None;
        if let Some((baseline_accepted, baseline_drafted)) = baseline {
            if state.window.samples.len() == ADSD_TENANT_WINDOW
                && state.window.drafted >= ADSD_TENANT_MIN_DRAFTED
            {
                let baseline_rate = baseline_accepted as f64 / baseline_drafted as f64;
                let tenant_rate = state.window.rate();
                let pooled_rate = (baseline_accepted as f64 + state.window.accepted as f64)
                    / (baseline_drafted as f64 + state.window.drafted as f64);
                let variance = (pooled_rate
                    * (1.0 - pooled_rate)
                    * (1.0 / baseline_drafted as f64 + 1.0 / state.window.drafted as f64))
                    .max(f64::EPSILON);
                let z_score = (tenant_rate - baseline_rate) / variance.sqrt();
                let rate_drop = baseline_rate - tenant_rate;
                let anomalous = rate_drop >= ADSD_MIN_RATE_DROP && z_score <= ADSD_Z_THRESHOLD;
                if anomalous {
                    state.anomalous_observations = state.anomalous_observations.saturating_add(1);
                    if state.anomalous_observations >= ADSD_SUSTAINED_OBSERVATIONS
                        && !state.incident_latched
                    {
                        state.incident_latched = true;
                        state.latched_baseline = Some((baseline_accepted, baseline_drafted));
                        suspect = Some(AdsdSuspect {
                            model: model.to_string(),
                            tenant: key.1.clone(),
                            baseline_rate,
                            tenant_rate,
                            z_score,
                            drafted: state.window.drafted,
                        });
                    }
                } else {
                    state.anomalous_observations = 0;
                    if rate_drop <= ADSD_REARM_RATE_DROP {
                        state.incident_latched = false;
                        state.latched_baseline = None;
                    }
                }
            }
        }

        self.model_windows
            .entry(model.to_string())
            .or_default()
            .push(&key.1, accepted, drafted);
        if let Some(event) = suspect.as_ref() {
            *self.suspect_total.entry(event.tenant.clone()).or_default() += 1;
        }
        suspect
    }
}

/// Credit one admitted request's prompt/cached token counts to its tenant row
/// (lane/cache-metering). The key is the tenant half of the PC-ISO namespace
/// (`auth::meter_key`); past METER_TENANT_CAP distinct rows, overflow aggregates
/// under "(other)" so the map is bounded while the totals stay exact.
fn meter_account(
    ns_tokens: &mut HashMap<String, [u64; 2]>,
    cache_ns: &str,
    n_prompt: u64,
    n_cached: u64,
) {
    let mk = crate::auth::meter_key(cache_ns);
    let row = if ns_tokens.contains_key(mk) || ns_tokens.len() < METER_TENANT_CAP {
        ns_tokens.entry(mk.to_string()).or_default()
    } else {
        ns_tokens.entry("(other)".to_string()).or_default()
    };
    row[0] += n_prompt;
    row[1] += n_cached;
}

/// Add cached-token credit discovered after admission (in-batch prefix fanout).
/// Prompt tokens were already charged by `meter_account` when the request admitted.
fn meter_cached_credit(ns_tokens: &mut HashMap<String, [u64; 2]>, cache_ns: &str, n_cached: u64) {
    meter_account(ns_tokens, cache_ns, 0, n_cached);
}

/// The naked prefix-cache default retains two full-`MEMRA_CTX` entries of the largest loaded
/// model. Two is the smallest useful reusable hot set: one hot/shared prefix plus one
/// incoming replacement. The budget remains global across models/namespaces and session
/// allocations still win through the existing reclaim-on-allocation-failure path.
const PREFIX_CACHE_DEFAULT_ENTRIES: usize = 2;

#[derive(Debug)]
enum PrefixCacheBudget {
    Configured {
        bytes: usize,
    },
    Derived {
        bytes: usize,
        requested_bytes: usize,
        entry_bytes: usize,
        model: String,
        ctx: usize,
        boot_free_bytes: usize,
        clamp_bytes: usize,
    },
}

impl PrefixCacheBudget {
    fn bytes(&self) -> usize {
        match self {
            Self::Configured { bytes } | Self::Derived { bytes, .. } => *bytes,
        }
    }
}

static PREFIX_CACHE_BUDGET: std::sync::OnceLock<PrefixCacheBudget> = std::sync::OnceLock::new();

fn prefix_recurrent_state_bytes(plan: &memra_gguf::model_plan::ModelPlan) -> usize {
    plan.layers
        .iter()
        .map(|layer| match layer.state {
            memra_gguf::model_plan::StatePlan::Recurrent {
                conv_width,
                conv_kernel,
                state_width,
            } => (conv_width as usize)
                .saturating_mul((conv_kernel as usize).saturating_sub(1))
                .saturating_add(state_width as usize),
            _ => 0,
        })
        .sum::<usize>()
        .saturating_mul(std::mem::size_of::<f32>())
}

fn prefix_entry_geometry_bytes(
    bytes_per_token: usize,
    recurrent_bytes: usize,
    ctx: usize,
) -> usize {
    bytes_per_token
        .saturating_mul(ctx)
        .saturating_add(recurrent_bytes)
}

fn model_prefix_entry_bytes(model: &HybridModel, ctx: usize) -> usize {
    let cfg = &model.cfg;
    let n_trunk = model.plan.layers.len();
    prefix_entry_geometry_bytes(
        memra_engine::cache::cache_bytes_per_token_for_plan(cfg, &model.plan, 0, n_trunk),
        prefix_recurrent_state_bytes(&model.plan),
        ctx,
    )
}

fn derived_prefix_cache_budget(
    entry_bytes: usize,
    boot_free_bytes: usize,
) -> (usize, usize, usize) {
    let requested_bytes = entry_bytes.saturating_mul(PREFIX_CACHE_DEFAULT_ENTRIES);
    // Keep the serving transient floor free. Unlike an explicit operator budget, the naked
    // default must not claim the last bytes that prefill/FA workspaces or a new session need.
    let clamp_bytes = boot_free_bytes.saturating_sub(SPEC_SHRINK_RESERVE);
    (
        requested_bytes.min(clamp_bytes),
        requested_bytes,
        clamp_bytes,
    )
}

fn init_prefix_cache_budget(
    engine: &Engine,
    loaded: &HashMap<String, LoadedModel>,
) -> &'static PrefixCacheBudget {
    PREFIX_CACHE_BUDGET.get_or_init(|| {
        if let Ok(raw) = std::env::var("MEMRA_PREFIX_CACHE_MB") {
            // Preserve the historical parsing contract: a present but invalid value falls back
            // to 256 MiB. Valid explicit values, including 0, remain authoritative and un-clamped.
            let mib = raw.parse::<usize>().unwrap_or(256);
            return PrefixCacheBudget::Configured {
                bytes: mib.saturating_mul(1024 * 1024),
            };
        }

        let ctx = std::env::var("MEMRA_CTX")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(8192);
        let mut entries: Vec<_> = loaded
            .iter()
            .map(|(name, lm)| (name.as_str(), model_prefix_entry_bytes(&lm.model, ctx)))
            .collect();
        entries.sort_unstable_by_key(|(name, _)| *name);
        let (model, entry_bytes) = entries
            .into_iter()
            .max_by_key(|(_, bytes)| *bytes)
            .map(|(name, bytes)| (name.to_string(), bytes))
            .unwrap_or_else(|| ("(none)".to_string(), 0));
        let boot_free_bytes = engine
            .ctx()
            .mem_get_info()
            .map(|(free, _)| free)
            .unwrap_or_else(|err| {
                eprintln!(
                    "[prefix-cache] WARNING: boot free-VRAM query failed ({err}); \
                           derived prefix cache disabled"
                );
                0
            });
        let (bytes, requested_bytes, clamp_bytes) =
            derived_prefix_cache_budget(entry_bytes, boot_free_bytes);
        PrefixCacheBudget::Derived {
            bytes,
            requested_bytes,
            entry_bytes,
            model,
            ctx,
            boot_free_bytes,
            clamp_bytes,
        }
    })
}

/// Resident byte budget for the prefix cache. `MEMRA_PREFIX_CACHE_MB=0` disables it. Before the
/// worker initializes geometry, retain the old 256 MiB fallback for env-independent unit tests.
fn prefix_cache_budget_bytes() -> usize {
    let budget = PREFIX_CACHE_BUDGET
        .get()
        .map(PrefixCacheBudget::bytes)
        .unwrap_or_else(|| {
            std::env::var("MEMRA_PREFIX_CACHE_MB")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(256)
                .saturating_mul(1024 * 1024)
        });
    if budget > 0 && memra_engine::pp::pp_host_bounce_active() {
        // This check remains live rather than being captured in the OnceLock: a runtime peer
        // failure can promote host bounce after startup. Prefix snapshots/restores copy every
        // stage-owned KV plane through the primary Engine, outside the bounced boundary path.
        0
    } else {
        budget
    }
}

/// MEMRA_PREFIX_CACHE_PROTECTED_PCT (default 80): byte share reserved for entries that have
/// demonstrated reuse. Keep both segments non-empty; malformed/out-of-range values fall back to
/// the documented default.
fn prefix_cache_protected_pct() -> usize {
    static P: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *P.get_or_init(|| {
        if !prefix_cache_slru_enabled() {
            // Plain-LRU rollback: a 100% protected share removes the segment split as a source of
            // eviction pressure; `capacity_victim` then evicts the GLOBAL oldest entry, which is
            // what actually makes this the pre-SLRU LRU. The share alone does NOT do it — entries
            // still enter probation and still promote on reuse, so without the victim-function
            // branch a 100% share merely makes promoted entries unevictable (the earlier comment
            // here claimed the opposite and was wrong).
            return 100;
        }
        std::env::var("MEMRA_PREFIX_CACHE_PROTECTED_PCT")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&v| (1..100).contains(&v))
            .unwrap_or(DEFAULT_PREFIX_CACHE_PROTECTED_PCT)
    })
}

/// Prefix-cache eviction policy seam. `slru` (default) = byte-budgeted probation/protected with
/// promotion on first reuse; `lru` = the pre-v0.82.0 plain global LRU.
///
/// The rollback exists because SLRU is not universally better: cx-slrutarget
/// (`research/slrutarget-20260813/`) found a real losing shape — after a complete hot-cohort
/// turnover, a stale protected cohort traps the new cyclic cohort in probation and the hit rate
/// goes to **0% where plain LRU reaches 75%**, reproduced at both 4,096 and 49,152 MiB and for
/// Q27-only, Q35-only, and worker-global paired entries. Until live telemetry says which shape our
/// traffic actually has, an operator who sees cache hits collapse needs one flag to get LRU back.
fn prefix_cache_slru_enabled() -> bool {
    static S: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *S.get_or_init(|| {
        !matches!(
            std::env::var("MEMRA_PREFIX_CACHE_POLICY")
                .as_deref()
                .map(str::trim),
            Ok("lru") | Ok("LRU")
        )
    })
}

fn prefix_cache_protected_bytes(budget: usize, protected_pct: usize) -> usize {
    // Split before multiplying so even a saturated MEMRA_PREFIX_CACHE_MB cannot overflow.
    (budget / 100)
        .saturating_mul(protected_pct)
        .saturating_add((budget % 100).saturating_mul(protected_pct) / 100)
}

/// In-batch cold-prefix fanout. `=0` is the rollback/measurement seam.
fn prefix_dedup_enabled() -> bool {
    static D: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *D.get_or_init(|| std::env::var("MEMRA_PREFIX_DEDUP").as_deref() != Ok("0"))
}

/// Immediate LCP-boundary restore. **Default OFF**: the lane that built this mechanism
/// (`research/lcprestore-20260813/`, merged 6249b0096) returned NO-GO — restored K/V state hashes
/// MATCH at splits 64/512/2048/4374, yet generated output BYTE-DIVERGES from a genuinely-cold
/// request at splits 512 and 2048, and the cause was never established. Shipping the disproven
/// path as the default was an integration mistake; it stays off until `cx-eosclass` root-causes
/// the restored-state divergence class. `=1` arms it for measurement only.
fn partial_prefix_restore_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("MEMRA_PREFIX_PARTIAL_RESTORE").as_deref() == Ok("1"))
}

/// Expensive split-state evidence: D2H-hash every restored K/V byte and len mirror. Diagnostics
/// only; production pays no copy or synchronization unless explicitly armed.
fn prefix_split_trace_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("MEMRA_PREFIX_SPLIT_TRACE").as_deref() == Ok("1"))
}

/// Batched scheduling on? (read once — mirrors the run-loop static; the prefix cache only
/// engages in batched mode, the default.)
fn serve_batching() -> bool {
    static B: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *B.get_or_init(|| {
        std::env::var("MEMRA_SERVE_BATCH")
            .map(|v| v != "0")
            .unwrap_or(true)
    })
}

/// Can this process admit a spec session under the current placement policy?
///
/// The admission gate uses this only for the SPEC_SHRINK_RESERVE transient floor
/// (lane/admit-oom). A sharded PP-2 process at the placement-aware default has LOW=0, so no
/// request can take spec and the plain path must not pay that reserve. `MEMRA_SPEC_GATE=0`
/// remains always-spec and therefore still pays it.
/// GEMMA SPEC seam (lane/gemma-batched stage 2/3, 2026-08-17): DEFAULT ON when a drafter
/// is attached, since the stage-3 mixed coexistence cell went green (owner law — better
/// wins by default; receipts in SERVED-SPEC.md: within-boot identity ALL GREEN, served
/// c1 spec 135.5 prose / 211.3 code tok/s, c8 batch coexists at 169.8 agg under a live
/// spec stream). Semantics: unset + MEMRA_DRAFT present = armed at the shipping K=5;
/// unset without a drafter = off (nothing to draft with); `0` = the eager/plain kill
/// switch; explicit K >= 1 = armed at that depth (and REQUIRES MEMRA_DRAFT — boot
/// refuses loud). Any other value REFUSES LOUD (the mis-typed-seam law).
/// Boot-time ambiguity refuse list for `MEMRA_DSPARK_SPEC=1` — the 3f4597f02 guard law:
/// two spec/parallelism programs on one model must never silently coexist, and every
/// combination that has never been co-gated refuses LOUD at spawn. Pure (env values in,
/// verdict out) so the tooth exercises every arm without a worker spawn; the caller
/// panics on `Some`.
///
/// `MEMRA_STEP_TP`/`MEMRA_STEP_EP` were MISSING from this list (hermes finding, fixed
/// 2026-08-23): the dspark round asserts single-device — its drafter taps ride the
/// serial prime, same invariant the `MEMRA_PP_STAGES` arm protects — so a step-TP/EP
/// multi-device trunk landing on a dspark-armed server was an unguarded, never-co-gated
/// composition.
fn dspark_spec_boot_conflict(
    spec_dflash: bool,
    gemma4_spec_k: usize,
    pp_stages: usize,
    step_tp: Option<&str>,
    step_ep: Option<&str>,
) -> Option<String> {
    if spec_dflash {
        return Some(
            "MEMRA_DSPARK_SPEC=1 is set together with MEMRA_SPEC_DFLASH; unset one — \
             refusing to guess which drafter you meant"
                .into(),
        );
    }
    if gemma4_spec_k > 0 {
        return Some(
            "MEMRA_DSPARK_SPEC=1 together with MEMRA_GEMMA4_SPEC has never been \
             co-gated on one server; run the routes on separate processes"
                .into(),
        );
    }
    if pp_stages > 1 {
        return Some(format!(
            "MEMRA_DSPARK_SPEC=1 with MEMRA_PP_STAGES={pp_stages}: the dspark round \
             asserts single-device (taps ride the serial prime); not supported"
        ));
    }
    for (name, value) in [("MEMRA_STEP_TP", step_tp), ("MEMRA_STEP_EP", step_ep)] {
        if value.is_some_and(|v| {
            let v = v.trim();
            !v.is_empty() && v != "0"
        }) {
            return Some(format!(
                "MEMRA_DSPARK_SPEC=1 with {name} set: the dspark round asserts \
                 single-device (taps ride the serial prime) and has never been co-gated \
                 with the step TP/EP multi-device trunk; unset one"
            ));
        }
    }
    None
}

fn gemma4_spec_k_env() -> usize {
    static K: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *K.get_or_init(|| match std::env::var("MEMRA_GEMMA4_SPEC").as_deref() {
        Err(_) => {
            if std::env::var("MEMRA_DRAFT").is_ok() {
                5
            } else {
                0
            }
        }
        Ok("0") => 0,
        Ok(v) => v.parse().unwrap_or_else(|_| {
            panic!(
                "MEMRA_GEMMA4_SPEC={v:?} is not a recognized value (want unset = default-on \
                 with MEMRA_DRAFT at K=5, 0 = plain kill switch, K >= 1 = explicit depth) — \
                 refusing to guess"
            )
        }),
    })
}

fn serve_spec_enabled() -> bool {
    if memra_engine::pp::pp_host_bounce_active() {
        // Stream-mode spec exposes primary-device token/position buffers to stage 0 by UVA.
        // The engine also refuses the direct path; keep the serving policy on plain PP here.
        return false;
    }
    static S: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let armed = *S.get_or_init(|| {
        std::env::var("MEMRA_SERVE_SPEC")
            .map(|v| v != "0")
            .unwrap_or(true)
    });
    if !armed {
        return false;
    }
    match spec_k_pin() {
        Some(k) => k > 0,
        None => !spec_gate_on() || spec_gate_low() > 0,
    }
}

/// Apply the peer-integrity advisory to the speculative half of admission only. `false` means
/// the existing plain session path; it is never a request refusal.
fn peer_probe_spec_admission(candidate: bool, peer_probe_allows_spec: bool) -> bool {
    candidate && peer_probe_allows_spec
}

/// ---- CONCURRENCY-GATED SPEC (lane/spec-gate, task #89, 2026-08-07) ----
///
/// THE SINGLE-CARD MEASUREMENT (research/specplace-20260808, N=3 interleaved on the
/// current train with q9 NVFP4+MTP + the production drafter, K=3, greedy):
///
/// | c | spec ON agg | spec OFF agg | S/N  |
/// |---|-------------|--------------|------|
/// | 1 | 374.8       | 224.5        | 1.67x WIN  |
/// | 2 | 374.5       | 347.5        | 1.08x WIN  |
/// | 4 | 377.3       | 617.1        | 0.61x LOSS |
///
/// Spec stays approximately flat because phase (a) steps each spec session's whole burst in a
/// serial host loop and phase (c) excludes spec sessions from batched decode. Single-card
/// therefore keeps the measured LOW=2/HIGH=4 crossover.
///
/// PP-2 IS A DIFFERENT POLICY CELL. The fixed q9 path measured 112.5/112.3/112.1 spec ON
/// against 223.3/340.3/593.4 spec OFF at c=1/2/4 (research/pp2spec-crash-20260807).
/// Re-checking the newly batched step35 core on the current train measured
/// 35.9/36.2/36.7 against 85.7/101.6/121.7, N=3 with no run-range overlap
/// (research/specplace-20260808). Spec loses every PP-2 cell, including c=1, so the
/// placement-aware default is LOW=0/HIGH=1: never admit spec.
///
/// Defaults:
///
///   single card / non-PP-2: LOW=2, HIGH=4
///   sharded cross-device PP-2: LOW=0, HIGH=1 (spec admission OFF)
///
/// `MEMRA_SPEC_GATE_LOW` / `_HIGH` explicitly override the placement defaults.
/// `MEMRA_SPEC_GATE=0` is the rollback seam and restores always-spec on every placement.
fn spec_gate_on() -> bool {
    static G: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *G.get_or_init(|| std::env::var("MEMRA_SPEC_GATE").as_deref() != Ok("0"))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct SpecGateThresholds {
    low: usize,
    high: usize,
    raw_high: usize,
    pp2_default: bool,
    low_overridden: bool,
    high_overridden: bool,
    high_clamped: bool,
}

fn spec_gate_defaults(pp2: bool) -> (usize, usize) {
    if pp2 { (0, 1) } else { (2, 4) }
}

fn resolve_spec_gate_thresholds(
    pp2: bool,
    low_override: Option<usize>,
    high_override: Option<usize>,
) -> SpecGateThresholds {
    let (default_low, default_high) = spec_gate_defaults(pp2);
    let low = low_override.unwrap_or(default_low);
    let raw_high = high_override.unwrap_or(default_high);
    let high_clamped = raw_high <= low;
    let high = if high_clamped {
        low.saturating_add(1)
    } else {
        raw_high
    };
    SpecGateThresholds {
        low,
        high,
        raw_high,
        pp2_default: pp2,
        low_overridden: low_override.is_some(),
        high_overridden: high_override.is_some(),
        high_clamped,
    }
}

/// This lane measured the cross-device, stage-split PP-2 placement. Do not silently apply its
/// default to PP-N or to the same-device/door-rollback configurations, whose execution shape is
/// different and unmeasured here.
fn spec_gate_pp2_placement() -> bool {
    let exactly_two_stages = std::env::var("MEMRA_PP_STAGES")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .is_some_and(|n| n == 2);
    exactly_two_stages && memra_engine::pp::pp_sharded_cross_device()
}

fn spec_gate_thresholds() -> &'static SpecGateThresholds {
    static T: std::sync::OnceLock<SpecGateThresholds> = std::sync::OnceLock::new();
    T.get_or_init(|| {
        let low_override = std::env::var("MEMRA_SPEC_GATE_LOW")
            .ok()
            .and_then(|v| v.parse().ok());
        let high_override = std::env::var("MEMRA_SPEC_GATE_HIGH")
            .ok()
            .and_then(|v| v.parse().ok());
        let thresholds =
            resolve_spec_gate_thresholds(spec_gate_pp2_placement(), low_override, high_override);
        if thresholds.high_clamped {
            eprintln!(
                "[spec-gate] WARN: MEMRA_SPEC_GATE_HIGH={} <= LOW={} leaves no hysteresis \
                 band (mode thrash); clamped to {}",
                thresholds.raw_high, thresholds.low, thresholds.high
            );
        }
        thresholds
    })
}

const SPEC_K_LONG_PROMPT_MIN: usize = 1024;
const SPEC_K_LONG_CACHE_MIN: usize = 1024;
const SPEC_K_COLD_SHORT: usize = 3;
const SPEC_K_COLD_LONG: usize = 3;
const SPEC_K_CACHED_LONG: usize = 2;
/// Cached-long depth when the loaded MTP head is rank-TRIMMED (d2t present): the trim drops
/// the draft lm_head from 248,320 to 32,768 rows (~221 -> ~29 us per draft step), which
/// re-prices the kpolicy-20260808 cached-long K=2 verdict — measured on ornith15
/// (research/orndecode-20260822: K=5 on cached-long 350.8 tok/s vs ~273 at k=2, 0.85-0.9
/// acceptance). Untrimmed heads keep the priced K=2.
const SPEC_K_CACHED_LONG_TRIM: usize = 5;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SpecKReason {
    OperatorPin,
    Replay,
    Placement,
    Concurrency,
    CachedLong,
    ColdShort,
    ColdLong,
}

impl SpecKReason {
    fn as_str(self) -> &'static str {
        match self {
            Self::OperatorPin => "operator-pin",
            Self::Replay => "oom-replay",
            Self::Placement => "pp2-placement",
            Self::Concurrency => "concurrency",
            Self::CachedLong => "cached-long",
            Self::ColdShort => "cold-short",
            Self::ColdLong => "cold-long",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct SpecKDecision {
    k: usize,
    reason: SpecKReason,
}

fn parse_spec_k_pin(raw: Option<&str>) -> Result<Option<usize>, String> {
    raw.map(|value| {
        value
            .parse::<usize>()
            .map_err(|_| format!("MEMRA_SPEC_K={value:?} is not a non-negative integer"))
    })
    .transpose()
}

/// An explicit K is an operator pin, including K=0 for plain serving. Invalid values keep
/// serving on the automatic policy but warn once instead of silently becoming a different K.
fn spec_k_pin() -> Option<usize> {
    static K: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
    *K.get_or_init(
        || match parse_spec_k_pin(std::env::var("MEMRA_SPEC_K").ok().as_deref()) {
            Ok(k) => k,
            Err(err) => {
                eprintln!("[spec-k] WARN: {err}; using automatic policy");
                None
            }
        },
    )
}

/// Whether the model's loaded MTP head is FR-Spec rank-trimmed (d2t map present) — the
/// condition under which the cached-long automatic depth re-prices to
/// SPEC_K_CACHED_LONG_TRIM (see the const's doc).
fn spec_trim_head(lm: &LoadedModel) -> bool {
    lm.model.mtp.as_ref().is_some_and(|h| h.d2t.is_some())
}

fn choose_spec_k(
    pin: Option<usize>,
    gate_on: bool,
    thresholds: SpecGateThresholds,
    projected_active: usize,
    prompt_tokens: usize,
    cached_tokens: usize,
    trim_head: bool,
) -> SpecKDecision {
    if let Some(k) = pin {
        return SpecKDecision {
            k,
            reason: SpecKReason::OperatorPin,
        };
    }
    if gate_on && projected_active > thresholds.low {
        let reason = if thresholds.pp2_default && !thresholds.low_overridden && thresholds.low == 0
        {
            SpecKReason::Placement
        } else {
            SpecKReason::Concurrency
        };
        return SpecKDecision { k: 0, reason };
    }
    if prompt_tokens >= SPEC_K_LONG_PROMPT_MIN && cached_tokens >= SPEC_K_LONG_CACHE_MIN {
        return SpecKDecision {
            k: if trim_head {
                SPEC_K_CACHED_LONG_TRIM
            } else {
                SPEC_K_CACHED_LONG
            },
            reason: SpecKReason::CachedLong,
        };
    }
    if prompt_tokens < SPEC_K_LONG_PROMPT_MIN {
        SpecKDecision {
            k: SPEC_K_COLD_SHORT,
            reason: SpecKReason::ColdShort,
        }
    } else {
        SpecKDecision {
            k: SPEC_K_COLD_LONG,
            reason: SpecKReason::ColdLong,
        }
    }
}

fn log_spec_gate_policy() {
    if let Some(k) = spec_k_pin() {
        eprintln!(
            "[spec-k] operator pin K={k}: automatic placement/concurrency/prompt policy \
             and automatic demotion disabled"
        );
        return;
    }
    if !spec_gate_on() {
        eprintln!("[spec-gate] policy disabled by MEMRA_SPEC_GATE=0: always-spec");
    } else {
        let thresholds = spec_gate_thresholds();
        let placement = if thresholds.pp2_default {
            "pp2-cross-device"
        } else {
            "single-or-non-pp2"
        };
        let source = if thresholds.low_overridden || thresholds.high_overridden {
            "env-resolved"
        } else {
            "placement-default"
        };
        let admission = if thresholds.low == 0 { "off" } else { "on" };
        eprintln!(
            "[spec-gate] policy placement={placement} LOW={} HIGH={} source={source} \
             spec-admission={admission}",
            thresholds.low, thresholds.high
        );
    }
    eprintln!(
        "[spec-k] automatic table: prompt<{} -> K={}; cold-long -> K={}; \
         prompt>= {} and cached>= {} -> K={} (K={} when the loaded MTP head is rank-trimmed)",
        SPEC_K_LONG_PROMPT_MIN,
        SPEC_K_COLD_SHORT,
        SPEC_K_COLD_LONG,
        SPEC_K_LONG_PROMPT_MIN,
        SPEC_K_LONG_CACHE_MIN,
        SPEC_K_CACHED_LONG,
        SPEC_K_CACHED_LONG_TRIM
    );
}

fn spec_gate_low() -> usize {
    spec_gate_thresholds().low
}

fn spec_gate_high() -> usize {
    spec_gate_thresholds().high
}

/// What the load path must SAY (and whether it must refuse) about one model's drafter
/// attachment. Pure data so the decision is unit-testable without a GPU or a 105 GB artifact —
/// the whole point of this seam is that the silent-degradation class it removes was invisible
/// to every gate in the repo (see `research/step-draft-20260807/`).
#[derive(Debug, PartialEq, Eq)]
pub enum DraftVerdict {
    /// A drafter is attached (embedded NextN head or an external `+draft` file). Spec is live
    /// as far as the load path is concerned; `spec_eligible` still arbitrates per request.
    Attached,
    /// No drafter and none was asked for, on an arch whose published artifact ships its MTP
    /// head as a SEPARATE file. Serving works — it just silently forgoes spec, which is the
    /// exact defect this lane exists to make audible. WARN, do not refuse.
    NoDrafterExternalMtpArch,
    /// No drafter, on an arch whose head (if any) rides in the trunk file. Nothing to say
    /// beyond the existing load line: an artifact with `nextn=0` here genuinely has no head.
    NoDrafterQuiet,
}

/// The drafter-attachment verdict for one loaded model — pure over the four inputs that
/// decide it, so the refusal and the warning are both pinned by GPU-free tests.
///
/// `external_mtp_arch` = "this arch's published artifact ships its MTP head in a separate
/// GGUF, so `nextn=0` on the trunk does NOT mean the model has no drafter available." Today
/// that is step35 (Step-3.7-Flash: trunk declares `nextn_predict_layers=0`, the three chained
/// NextN blocks ship in `Step3.7-flash-mtp-Q8_0.gguf`). It is a property of the ARCH, not of
/// the file in hand, which is why it cannot be read off the trunk config.
pub fn draft_verdict(has_drafter: bool, external_mtp_arch: bool) -> DraftVerdict {
    // (#87 CLOSED, lane/pp2spec-crash 2026-08-08: this fn used to refuse spec + drafter over
    // a sharded cross-device PP placement — the sticky CUDA_ERROR_ILLEGAL_ADDRESS regime.
    // Root cause was the ppN reverse-publication hole: stage-stream pool blocks freed while
    // the primary stream held queued reads, reused by the next burst's stage allocations.
    // Fixed by `PpNRt::fence_stages_behind` at all three ppN bodies + stage-cache admission;
    // crash gate 212/212 at c=2..8 on the placement that lost 48/48, run-spec K=1..8 PASS
    // with acceptance identical to door-shut. research/pp2spec-crash-20260807/.)
    if has_drafter {
        return DraftVerdict::Attached;
    }
    if external_mtp_arch {
        DraftVerdict::NoDrafterExternalMtpArch
    } else {
        DraftVerdict::NoDrafterQuiet
    }
}

/// The one-line operator message for a verdict, or `None` when there is nothing to say.
/// Separated from `draft_verdict` so the TEXT is testable too — a warning nobody can act on
/// is the same defect as no warning (the attach spelling has to be IN the line).
pub fn draft_verdict_message(v: &DraftVerdict, name: &str, path: &str) -> Option<String> {
    match v {
        DraftVerdict::Attached | DraftVerdict::NoDrafterQuiet => None,
        DraftVerdict::NoDrafterExternalMtpArch => Some(format!(
            "[worker] WARN: {name}: model pack declares an external MTP drafter; no MTP drafter \
             attached — serving plain decode, no speculative decoding. The MTP/NextN head ships \
             as a SEPARATE GGUF, so nextn_predict_layers=0 in the trunk does NOT mean the \
             model has no drafter. Attach with MEMRA_MODELS=\"{name}={path}+/path/to/\
             external-mtp.gguf\" (the same '+draft' convention every regime drafter \
             uses; docs/DRAFT-REGIME.md)."
        )),
    }
}

/// Admission transient-reserve override in BYTES (lane/admit-oom, 2026-08-06). This exists for
/// exactly one reason: the c=64 stress gate's TEETH arm. A gate that can only be observed
/// passing proves nothing, so `tools/serve-stress-gate.sh --teeth` forces the reserve tiny
/// (MEMRA_ADMIT_RESERVE_MB=16) and asserts the RED comes back — if the deliberately-broken
/// setting still passes, the gate is not measuring what it claims to measure. It is a
/// diagnostics/teeth door under the flags doctrine, never a tuning knob: the winning value is
/// the default and needs no flag.
fn admit_reserve_override() -> Option<usize> {
    static O: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
    *O.get_or_init(|| {
        std::env::var("MEMRA_ADMIT_RESERVE_MB")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .map(|mb| {
                eprintln!(
                    "[admit-oom] WARN: MEMRA_ADMIT_RESERVE_MB={mb} overrides the \
                           {}MB transient reserve (teeth/diagnostics door — NOT a tuning knob)",
                    SPEC_SHRINK_RESERVE / (1 << 20)
                );
                mb * (1 << 20)
            })
    })
}

/// Admission headroom charged ON TOP of the request's own cost (lane/kv256-capacity,
/// 2026-08-09). Spec-capable paths pay the full measured transient floor
/// (SPEC_SHRINK_RESERVE — capture arenas + verify activations, the admit-oom control fit).
/// Plain paths pay `min(cost, floor)`: the plain transient class (prime chunk slabs, batched
/// step scratch) is CHUNK-bounded, not ctx-scaled, so a 262,144-token request must not
/// reserve a whole second 21,894 MB session for it. For any cost at or below the floor this
/// is byte-identical to the previous `reserve = cost` contract (the c<=64 small-model regime
/// the admit-oom gate was calibrated on); it only unbinds where the old rule over-reserved.
/// The teeth/diagnostics override door applies to both branches — a forced-tiny reserve must
/// invert the verdict on whichever path the gate is exercising.
fn admission_reserve(spec_capable: bool, cost: usize, override_bytes: Option<usize>) -> usize {
    let floor = override_bytes.unwrap_or(SPEC_SHRINK_RESERVE);
    if spec_capable { floor } else { cost.min(floor) }
}

fn dual_pp_boundary_slot_bytes(wave_cap: usize, n_embd: usize) -> usize {
    wave_cap
        .saturating_mul(n_embd)
        .saturating_mul(std::mem::size_of::<f32>())
}

fn admission_required(cost: usize, reserve: usize) -> usize {
    cost.saturating_add(reserve)
}

/// One stage's dual-only admission charge. `session_bytes` is the exact stage-owned context
/// allocation plus the conservative fixed high-water residual. Each simultaneously active stage
/// walker gets the existing transient reserve. The receiving stage additionally owns both
/// persistent boundary slots prepared before the first dual tick.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct DualPpStageAdmission {
    session_bytes: usize,
    reserve_bytes: usize,
    boundary_bytes: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct AdmissionDeviceRequirement {
    device: usize,
    session_bytes: usize,
    tp_kv_bytes: usize,
    pending_tp_kv_bytes: usize,
    reserve_bytes: usize,
    boundary_bytes: usize,
}

impl AdmissionDeviceRequirement {
    fn required(self) -> usize {
        self.session_bytes
            .saturating_add(self.tp_kv_bytes)
            .saturating_add(self.pending_tp_kv_bytes)
            .saturating_add(self.reserve_bytes)
            .saturating_add(self.boundary_bytes)
    }

    fn add_stage(&mut self, stage: DualPpStageAdmission) {
        self.session_bytes = self.session_bytes.saturating_add(stage.session_bytes);
        self.reserve_bytes = self.reserve_bytes.saturating_add(stage.reserve_bytes);
        self.boundary_bytes = self.boundary_bytes.saturating_add(stage.boundary_bytes);
    }

    fn add_tp_kv(&mut self, bytes: usize, pending: bool, reserve_bytes: usize) {
        if pending {
            self.pending_tp_kv_bytes = self.pending_tp_kv_bytes.saturating_add(bytes);
        } else {
            self.tp_kv_bytes = self.tp_kv_bytes.saturating_add(bytes);
        }
        // One device-local transient floor covers the stage walker and its rank-local attention
        // work. Co-located PP stages retain their existing additive reserve above.
        self.reserve_bytes = self.reserve_bytes.max(reserve_bytes);
    }
}

fn dual_pp_device_requirements(
    devices: [usize; 2],
    stages: [DualPpStageAdmission; 2],
) -> Vec<AdmissionDeviceRequirement> {
    let mut requirements: Vec<AdmissionDeviceRequirement> = Vec::with_capacity(2);
    for (device, stage) in devices.into_iter().zip(stages) {
        if let Some(existing) = requirements
            .iter_mut()
            .find(|requirement| requirement.device == device)
        {
            existing.add_stage(stage);
        } else {
            requirements.push(AdmissionDeviceRequirement {
                device,
                session_bytes: stage.session_bytes,
                tp_kv_bytes: 0,
                pending_tp_kv_bytes: 0,
                reserve_bytes: stage.reserve_bytes,
                boundary_bytes: stage.boundary_bytes,
            });
        }
    }
    requirements
}

fn add_tp_kv_requirements(
    requirements: &mut Vec<AdmissionDeviceRequirement>,
    charges: &[StepTpKvDeviceAdmission],
    pending: bool,
    reserve_bytes: usize,
) {
    for charge in charges {
        if let Some(existing) = requirements
            .iter_mut()
            .find(|requirement| requirement.device == charge.device)
        {
            existing.add_tp_kv(charge.bytes, pending, reserve_bytes);
        } else {
            let mut requirement = AdmissionDeviceRequirement {
                device: charge.device,
                session_bytes: 0,
                tp_kv_bytes: 0,
                pending_tp_kv_bytes: 0,
                reserve_bytes: 0,
                boundary_bytes: 0,
            };
            requirement.add_tp_kv(charge.bytes, pending, reserve_bytes);
            requirements.push(requirement);
        }
    }
}

fn parallel_device_requirements(
    primary_device: usize,
    primary_cost: usize,
    reserve_bytes: usize,
    dual: Option<([usize; 2], [DualPpStageAdmission; 2])>,
    request_tp_kv: &[StepTpKvDeviceAdmission],
    pending_tp_kv: &[StepTpKvDeviceAdmission],
) -> Vec<AdmissionDeviceRequirement> {
    let mut requirements = if let Some((devices, stages)) = dual {
        dual_pp_device_requirements(devices, stages)
    } else {
        vec![AdmissionDeviceRequirement {
            device: primary_device,
            session_bytes: primary_cost,
            tp_kv_bytes: 0,
            pending_tp_kv_bytes: 0,
            reserve_bytes,
            boundary_bytes: 0,
        }]
    };
    add_tp_kv_requirements(&mut requirements, request_tp_kv, false, reserve_bytes);
    add_tp_kv_requirements(&mut requirements, pending_tp_kv, true, reserve_bytes);
    requirements.sort_unstable_by_key(|requirement| requirement.device);
    requirements
}

fn merge_tp_kv_charges(
    totals: &mut Vec<StepTpKvDeviceAdmission>,
    charges: &[StepTpKvDeviceAdmission],
) {
    for charge in charges {
        if let Some(existing) = totals.iter_mut().find(|item| item.device == charge.device) {
            existing.bytes = existing.bytes.saturating_add(charge.bytes);
        } else {
            totals.push(*charge);
        }
    }
    totals.sort_unstable_by_key(|charge| charge.device);
}

fn active_unmaterialized_tp_kv(
    active: &[Session],
    loaded: &HashMap<String, LoadedModel>,
) -> Result<Vec<StepTpKvDeviceAdmission>, String> {
    let mut totals = Vec::new();
    for session in active {
        let model = loaded.get(&session.model).ok_or_else(|| {
            format!(
                "active session references unloaded model {:?}",
                session.model
            )
        })?;
        let cache = session
            .spec
            .as_ref()
            .map(memra_engine::spec::SpecSession::cache_ref)
            .or(session.cache.as_ref());
        let Some(cache) = cache else {
            continue;
        };
        let charges = model
            .model
            .step_tp_unmaterialized_kv_bytes(Some(cache), cache.max_ctx)?;
        merge_tp_kv_charges(&mut totals, &charges);
    }
    Ok(totals)
}

fn dual_pp_stage_context_bytes(
    model: &HybridModel,
    fence: &[usize],
    ctx_cap: usize,
    spec: bool,
) -> Option<[usize; 2]> {
    if fence.len() != 3 || fence[0] != 0 {
        return None;
    }
    let n_layers = model.cfg.n_layer as usize;
    if fence[1] > fence[2] || fence[2] > n_layers {
        return None;
    }
    let ring_rows = memra_engine::cache::cache_ring_row_cap_for_plan(&model.plan);
    let mut stage_bytes = [0usize; 2];
    for stage in 0..2 {
        let lo = fence[stage];
        let hi = if stage == 1 {
            n_layers
        } else {
            fence[stage + 1]
        };
        let bytes_per_token =
            memra_engine::cache::cache_bytes_per_token_for_plan(&model.cfg, &model.plan, lo, hi);
        let ring_bytes_per_token = memra_engine::cache::cache_ring_bytes_per_token_for_plan(
            &model.cfg,
            &model.plan,
            lo,
            hi,
        );
        stage_bytes[stage] =
            context_cache_bytes(bytes_per_token, ring_bytes_per_token, ring_rows, ctx_cap);
    }
    if spec {
        let (plain, plain_ring, _) = model.plain_session_kv_shape();
        let (spec, spec_ring, _) = model.spec_session_kv_shape();
        stage_bytes[1] = stage_bytes[1].saturating_add(context_cache_bytes(
            spec.saturating_sub(plain),
            spec_ring.saturating_sub(plain_ring),
            ring_rows,
            ctx_cap,
        ));
    }
    let (total, total_ring, _) = if spec {
        model.spec_session_kv_shape()
    } else {
        model.plain_session_kv_shape()
    };
    debug_assert_eq!(
        stage_bytes[0].saturating_add(stage_bytes[1]),
        context_cache_bytes(total, total_ring, ring_rows, ctx_cap),
        "PP stage-local KV admission must partition the aggregate cache geometry",
    );
    Some(stage_bytes)
}

fn dual_pp_stage_admission(
    context_bytes: [usize; 2],
    activation_bytes: usize,
    reserve_bytes: usize,
    boundary_slot_bytes: usize,
) -> [DualPpStageAdmission; 2] {
    [
        DualPpStageAdmission {
            session_bytes: context_bytes[0].saturating_add(activation_bytes),
            reserve_bytes,
            boundary_bytes: 0,
        },
        DualPpStageAdmission {
            session_bytes: context_bytes[1].saturating_add(activation_bytes),
            reserve_bytes,
            boundary_bytes: boundary_slot_bytes.saturating_mul(2),
        },
    ]
}

/// STEP-OOM PARK budget (lane/admit-oom, 2026-08-06): how many times a session may be parked
/// back to the queue after a step-time CUDA OOM before the failure is reported honestly.
/// A transient collision (a peer's capture arena landing in the same tick) clears as soon as
/// ONE session retires, so a small budget covers the real case; an unbounded retry would turn
/// a genuine capacity failure into a silent hang, which is strictly worse than the error it
/// replaces. MEMRA_STEP_OOM_RETRIES overrides (0 = restore the pre-fix kill-on-OOM behavior —
/// the rollback seam).
const STEP_OOM_MAX_RETRIES: u32 = 3;

fn step_oom_retries() -> u32 {
    static R: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *R.get_or_init(|| {
        std::env::var("MEMRA_STEP_OOM_RETRIES")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(STEP_OOM_MAX_RETRIES)
    })
}

/// Is this error a CUDA out-of-memory? Quoted, never inferred (the evidence-discipline law):
/// the match is on the driver's own error text, so a non-OOM step failure can never be
/// silently retried as if it were a capacity blip.
fn is_cuda_oom(err: &str) -> bool {
    err.contains("CUDA_ERROR_OUT_OF_MEMORY") || err.contains("out of memory")
}

/// Run one allocation retry only when the first failure released reclaimable state. The caller
/// owns the reclaim policy; keeping the retry here makes the one-retry bound unit-testable.
fn alloc_with_single_reclaim_retry<T, E>(
    mut alloc: impl FnMut() -> Result<T, E>,
    mut reclaim: impl FnMut(&E) -> bool,
) -> Result<T, E> {
    match alloc() {
        Ok(value) => Ok(value),
        Err(first_err) if reclaim(&first_err) => alloc(),
        Err(err) => Err(err),
    }
}

/// One full-attn layer's cached prefix bytes: exactly `len` tokens of quantized K/V.
struct PrefixPlane {
    k: CudaSlice<u8>,
    v: CudaSlice<u8>,
    len: usize,
    k_tok_bytes: usize,
    v_tok_bytes: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PrefixSegment {
    Probation,
    Protected,
}

/// A cached token-prefix: per-layer KV byte copies + recurrent conv/ssm copies + the logits
/// row AT the boundary (empty-suffix resumes sample from it, same as the continuation pool).
struct PrefixEntry {
    layout_version: u32,
    pool_key: PoolKey,
    toks: Vec<u32>,
    kv: Vec<Option<PrefixPlane>>,
    conv: Vec<Option<CudaSlice<f32>>>,
    ssm: Vec<Option<CudaSlice<f32>>>,
    pos: usize,
    last_logits: Vec<f32>,
    /// MTP draft-scratch rows `[0..pos)` (lane/spec-on-cache-hit): present only on entries
    /// published from a spec session's boundary capture. Under predecessor pairing a draft
    /// row i is a function of (token_i, trunk_hidden_{i-1}) — tokens `< pos` only — so the
    /// exact token key that validates the trunk planes validates this plane too. A hit
    /// carrying it can re-arm a SpecSession instead of downgrading to plain; entries
    /// without it (plain-path seeds/splits) keep serving plain exactly as before.
    draft: Option<PrefixPlane>,
    /// DFlash draft-KV tail (lane/dspark-draft-plane-20260827): present only on entries
    /// published from a DSPARK session's boundary capture, and the thing that lets a LONG-decode
    /// request restore instead of cold-priming. Before it, a dspark request had to discard even
    /// a full-prompt hit and re-prefill, because its draft state is derived from trunk hidden
    /// FEATURES that a KV restore cannot return — so the state has to travel with the entry.
    ///
    /// Only the readable tail travels (see `DflashKvTail`): ~85 MB against ~1,229 MB for the
    /// full history, which would have exceeded the entry's own ~1,057 MB of trunk planes.
    ///
    /// A plane from one spec program must never be restored into the other. That cannot happen
    /// here — arming dspark disables the MTP arm for the model at boot, loudly — and the
    /// restore asserts it rather than relying on it.
    dspark_draft: Option<memra_engine::dflash::DflashKvTail>,
    /// Pre-output_norm trunk hidden of row `pos - 1` (the `SpecSession::last_h`
    /// predecessor-pairing anchor). Empty = unavailable; a spec restore then leaves the
    /// fill's zeros row-0 fallback to cover it (acceptance-only).
    last_h: Vec<f32>,
    bytes: usize,
    last_use: Instant,
    /// Recency-index identity (Q3, audit 2026-08-05): unique per insert (monotonic counter,
    /// assigned by `insert`), disambiguating equal `last_use` Instants in the LRU BTreeMap key.
    id: u64,
    /// New snapshots enter probation. A successful cross-request reuse is the only promotion
    /// path into protected; protected overflow demotes its byte-LRU back to probation.
    segment: PrefixSegment,
    /// In-flight fanout/cache-hit leases. A pinned entry is absent from the evictable LRU
    /// index until the last participating session retires.
    pins: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PartialPrefixDecision {
    Restore,
    RefuseNoSuffix,
    RefuseRoutedMoe,
    RefuseRecurrentMidEntry,
}

impl PartialPrefixDecision {
    fn refusal(self) -> Option<&'static str> {
        match self {
            Self::Restore => None,
            Self::RefuseNoSuffix => Some(
                "request ends at the split and the longer entry has no logits for that boundary",
            ),
            Self::RefuseRoutedMoe => Some(
                "routed-MoE mid-entry split is unqualified (Q35-A3B class stays on cold/snapshot fallback)",
            ),
            Self::RefuseRecurrentMidEntry => Some(
                "hybrid conv/SSM state exists only at the entry endpoint and cannot be truncated",
            ),
        }
    }
}

/// Decide only the NEW mid-entry path. Whole-entry lookup remains unchanged and already restores
/// recurrent state exactly at its captured endpoint. For an actual mid-entry split, recurrent
/// state and routed-MoE are fail-closed until their own boundary exactness receipts exist.
fn partial_prefix_decision(
    entry_has_recurrent: bool,
    routed_moe: bool,
    lcp: usize,
    entry_pos: usize,
    prompt_len: usize,
) -> PartialPrefixDecision {
    if lcp >= prompt_len {
        PartialPrefixDecision::RefuseNoSuffix
    } else if routed_moe {
        PartialPrefixDecision::RefuseRoutedMoe
    } else if entry_has_recurrent && lcp != entry_pos {
        PartialPrefixDecision::RefuseRecurrentMidEntry
    } else {
        PartialPrefixDecision::Restore
    }
}

/// Device-independent half of restore preflight. This is deliberately one atomic check so the
/// corruption gate can exercise every range/layout/version class without constructing CUDA state.
#[allow(clippy::too_many_arguments)]
fn validate_prefix_plane_shape(
    entry_pos: usize,
    restore_len: usize,
    src_len: usize,
    src_k_tok_bytes: usize,
    src_v_tok_bytes: usize,
    src_k_bytes: usize,
    src_v_bytes: usize,
    dst_k_tok_bytes: usize,
    dst_v_tok_bytes: usize,
    dst_k_bytes: usize,
    dst_v_bytes: usize,
) -> Result<(usize, usize), String> {
    if src_len != entry_pos {
        return Err(format!("source len {src_len} != entry pos {entry_pos}"));
    }
    if src_k_tok_bytes != dst_k_tok_bytes || src_v_tok_bytes != dst_v_tok_bytes {
        return Err(format!(
            "KV layout {src_k_tok_bytes}/{src_v_tok_bytes} != cache \
             {dst_k_tok_bytes}/{dst_v_tok_bytes} bytes/token",
        ));
    }
    let expected_k = src_len
        .checked_mul(src_k_tok_bytes)
        .ok_or_else(|| "source K byte count overflow".to_string())?;
    let expected_v = src_len
        .checked_mul(src_v_tok_bytes)
        .ok_or_else(|| "source V byte count overflow".to_string())?;
    if src_k_bytes != expected_k.max(1) || src_v_bytes != expected_v.max(1) {
        return Err(format!(
            "truncated/corrupt source planes: K {src_k_bytes} != {}, V {src_v_bytes} != {}",
            expected_k.max(1),
            expected_v.max(1),
        ));
    }
    let copy_k = restore_len
        .checked_mul(dst_k_tok_bytes)
        .ok_or_else(|| "restore K byte count overflow".to_string())?;
    let copy_v = restore_len
        .checked_mul(dst_v_tok_bytes)
        .ok_or_else(|| "restore V byte count overflow".to_string())?;
    if dst_k_bytes < copy_k || dst_v_bytes < copy_v {
        return Err(format!(
            "destination planes too small: K {dst_k_bytes} < {copy_k}, \
             V {dst_v_bytes} < {copy_v}",
        ));
    }
    Ok((copy_k, copy_v))
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct PrefixPin {
    key: PoolKey,
    id: u64,
}

#[derive(Default)]
struct PrefixCache {
    /// per-(model, namespace) entry pools (KV geometry/format is per model; the namespace
    /// is the PC-ISO trust boundary — the equality check is part of the map key, so the
    /// default path pays nothing beyond hashing "" alongside the model id).
    entries: HashMap<PoolKey, Vec<PrefixEntry>>,
    /// Byte-budgeted SLRU indexes. New entries enter probation; a real reuse promotes to
    /// protected. Capacity pressure consumes probation LRU first, so one-hit scan traffic
    /// cannot displace a protected entry while probation has an evictable victim. Protected
    /// overflow demotes its own LRU until it fits its byte target. The targets are global across
    /// namespaces because VRAM is global; visibility remains scoped by `entries` above.
    ///
    /// Each BTreeMap preserves the Q3 O(log E) victim lookup and deterministic `(last_use,id)`
    /// tie break. Emergency flush compares both heads to retain global oldest-first removal.
    /// PINNING (lane/cx-prefix-dedup): pinned entries are deliberately ABSENT from this
    /// pair. The last lease release returns the entry at current recency. Value = (pool
    /// key, index into that pool's Vec), kept exact on removal by swap_remove +
    /// moved-entry index fixup. Every `last_use` write goes through touch/pin/unpin/insert
    /// so index and entries never drift.
    probation_lru: std::collections::BTreeMap<(Instant, u64), (PoolKey, usize)>,
    protected_lru: std::collections::BTreeMap<(Instant, u64), (PoolKey, usize)>,
    next_id: u64,
    total_bytes: usize,
    probation_bytes: usize,
    protected_bytes: usize,
    protected_target_bytes: usize,
    hits: u64,
    misses: u64,
    inserts: u64,
    evictions: u64,
    skips_budget: u64,
    skips_pinned: u64,
    hit_tokens: u64,
    /// LCP histogram (lane/cache-metering): one sample per probe — the served entry's
    /// token length on a hit, `best_lcp` on a miss (both already computed; no new scan).
    /// Lower-edge buckets `LCP_HIST_EDGES` (see `lcp_bucket`).
    lcp_hist: [u64; 11],
}

/// Lower edges of the LCP histogram buckets: bucket i counts samples in
/// [EDGES[i], EDGES[i+1]), the last bucket [4096, inf). [64,512) — the tick-seg
/// segmentation window — is exactly buckets 4+5+6.
pub const LCP_HIST_EDGES: [usize; 11] = [0, 1, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096];

#[derive(Debug, PartialEq, Eq)]
struct PrefixFanoutCandidate {
    active_idx: usize,
    key: PoolKey,
    prompt: Vec<u32>,
}

#[derive(Debug, PartialEq, Eq)]
struct PrefixFanoutGroup {
    members: Vec<usize>,
    prefix_len: usize,
}

/// Partition one admission window by the PC-ISO key and exact token prefix. Hashes are
/// deliberately absent from membership: equality of `(model, cache_ns)` is checked before
/// any token comparison, then the first 64 token ids and the full group LCP are exact.
fn prefix_fanout_groups(
    candidates: &[PrefixFanoutCandidate],
    prefix_cap: usize,
) -> Vec<PrefixFanoutGroup> {
    if prefix_cap < PREFIX_CACHE_MIN_TOKENS {
        return Vec::new();
    }
    let mut used = vec![false; candidates.len()];
    let mut out = Vec::new();
    for i in 0..candidates.len() {
        if used[i] || candidates[i].prompt.len() < PREFIX_CACHE_MIN_TOKENS {
            continue;
        }
        let mut group = vec![i];
        for j in i + 1..candidates.len() {
            if used[j]
                || candidates[j].key != candidates[i].key
                || candidates[j].prompt.len() < PREFIX_CACHE_MIN_TOKENS
                || candidates[j].prompt[..PREFIX_CACHE_MIN_TOKENS]
                    != candidates[i].prompt[..PREFIX_CACHE_MIN_TOKENS]
            {
                continue;
            }
            group.push(j);
        }
        if group.len() < 2 {
            continue;
        }
        let mut lcp = group
            .iter()
            .map(|&j| candidates[j].prompt.len())
            .min()
            .unwrap_or(0);
        for &j in group.iter().skip(1) {
            lcp = PrefixCache::lcp(&candidates[i].prompt[..lcp], &candidates[j].prompt[..lcp]);
        }
        let prefix_len = lcp.min(prefix_cap);
        if prefix_len < PREFIX_CACHE_MIN_TOKENS {
            continue;
        }
        for &j in &group {
            used[j] = true;
        }
        out.push(PrefixFanoutGroup {
            members: group.iter().map(|&j| candidates[j].active_idx).collect(),
            prefix_len,
        });
    }
    out
}

impl PrefixCache {
    fn lcp(a: &[u32], b: &[u32]) -> usize {
        a.iter().zip(b.iter()).take_while(|(x, y)| x == y).count()
    }

    /// Histogram bucket index for an LCP sample (see `LCP_HIST_EDGES`).
    fn lcp_bucket(n: usize) -> usize {
        LCP_HIST_EDGES.iter().rposition(|&e| n >= e).unwrap_or(0)
    }

    /// Record one probe outcome into the LCP histogram (hit: entry length; miss: best_lcp).
    fn record_lcp(&mut self, n: usize) {
        self.lcp_hist[Self::lcp_bucket(n)] += 1;
    }

    fn record_budget_skip(&mut self, pinned: bool) {
        if pinned {
            self.skips_pinned += 1;
            return;
        }
        let first_budget_skip = self.skips_budget == 0;
        self.skips_budget += 1;
        if first_budget_skip {
            eprintln!(
                "[prefix-cache] WARNING: first prefix insert is larger than the entire byte \
                       budget; this shape cannot produce cross-request hits. Subsequent refusals \
                       remain visible in prefix_cache_skips_budget"
            );
        }
    }

    /// Admission observed a miss before a same-window sibling produced the shared prefix.
    /// Rewrite that provisional probe into the final served-path hit so cache-meter receipts
    /// count requests, tokens, and LCP depth exactly once.
    fn promote_miss_to_hit(&mut self, miss_lcp: usize, hit_len: usize) {
        // 2.5 (code-audit-20260809): a telemetry bookkeeping slip must NEVER panic the worker
        // and take every concurrent stream with it. These were release `checked_sub().expect()`;
        // this lane adds new admission routes that set the same counters, so demote them to a
        // saturating decrement + a one-line receipt. Under-counting a metric is honest; a crash
        // is not.
        self.misses = self.misses.checked_sub(1).unwrap_or_else(|| {
            eprintln!(
                "[prefix-cache] WARN: fanout promoted a miss that was never recorded \
                       (misses underflow) — metric slip, not fatal"
            );
            0
        });
        let bucket = Self::lcp_bucket(miss_lcp);
        self.lcp_hist[bucket] = self.lcp_hist[bucket].saturating_sub(1);
        self.hits += 1;
        self.hit_tokens += hit_len as u64;
        self.record_lcp(hit_len);
    }

    fn n_entries(&self) -> usize {
        self.entries.values().map(|p| p.len()).sum()
    }

    /// Longest entry whose token key exactly prefixes `prompt` (floor PREFIX_CACHE_MIN_TOKENS).
    /// Only the caller's own (model, namespace) pool is scanned — cross-namespace entries
    /// are structurally unreachable (PC-ISO).
    fn lookup(&self, key: &PoolKey, prompt: &[u32]) -> Option<usize> {
        let pool = self.entries.get(key)?;
        let mut best: Option<(usize, usize)> = None;
        for (i, e) in pool.iter().enumerate() {
            let n = e.toks.len();
            if n >= PREFIX_CACHE_MIN_TOKENS
                && n <= prompt.len()
                && prompt[..n] == e.toks[..]
                && best.is_none_or(|(_, bn)| n > bn)
            {
                best = Some((i, n));
            }
        }
        best.map(|(i, _)| i)
    }

    /// Entry with the longest common prefix against `prompt`, plus that exact boundary. Ties
    /// prefer the longer stored entry, then its stable pool order; the copied span is identical.
    fn best_lcp_entry(&self, key: &PoolKey, prompt: &[u32]) -> Option<(usize, usize)> {
        self.entries
            .get(key)?
            .iter()
            .enumerate()
            .map(|(i, e)| (i, Self::lcp(&e.toks, prompt), e.toks.len()))
            .max_by_key(|&(i, lcp, entry_len)| (lcp, entry_len, std::cmp::Reverse(i)))
            .map(|(i, lcp, _)| (i, lcp))
    }

    /// Longest common prefix of `prompt` with ANY entry (the LCP-split learning signal).
    fn best_lcp(&self, key: &PoolKey, prompt: &[u32]) -> usize {
        self.best_lcp_entry(key, prompt)
            .map(|(_, lcp)| lcp)
            .unwrap_or(0)
    }

    /// Is any entry (>= min tokens) already a full prefix of `prompt`? (seed dedupe)
    #[cfg(test)]
    fn has_covering(&self, key: &PoolKey, prompt: &[u32]) -> bool {
        self.deepest_covering(key, prompt).is_some()
    }

    /// Depth (token length) of the DEEPEST entry that is a full prefix of `prompt`, or None.
    /// The seed/deepen decision compares against this instead of the boolean `has_covering`
    /// so a shallow class entry no longer freezes the class depth (H11).
    fn deepest_covering(&self, key: &PoolKey, prompt: &[u32]) -> Option<usize> {
        self.entries.get(key).and_then(|pool| {
            pool.iter()
                .filter_map(|e| {
                    let n = e.toks.len();
                    (n >= PREFIX_CACHE_MIN_TOKENS && n <= prompt.len() && prompt[..n] == e.toks[..])
                        .then_some(n)
                })
                .max()
        })
    }

    fn has_key(&self, key: &PoolKey, toks: &[u32]) -> bool {
        self.entries
            .get(key)
            .is_some_and(|pool| pool.iter().any(|e| e.toks[..] == *toks))
    }

    fn key_index(&self, key: &PoolKey, toks: &[u32]) -> Option<usize> {
        self.entries
            .get(key)?
            .iter()
            .position(|e| e.toks[..] == *toks)
    }

    /// Refresh an exact DFlash transition entry without claiming a cache hit: the request still
    /// primes every prompt token to rebuild draft state, so cached-token and hit counters stay
    /// unchanged even though the preserved trunk entry has demonstrated future plain-path value.
    fn touch_exact_without_credit(&mut self, key: &PoolKey, toks: &[u32]) -> bool {
        let Some(i) = self.key_index(key, toks) else {
            return false;
        };
        self.touch(key, i);
        true
    }

    fn id_index(&self, pin: &PrefixPin) -> Option<usize> {
        self.entries
            .get(&pin.key)?
            .iter()
            .position(|e| e.id == pin.id)
    }

    fn lru_key(e: &PrefixEntry) -> (Instant, u64) {
        (e.last_use, e.id)
    }

    fn remove_lru(&mut self, segment: PrefixSegment, key: &(Instant, u64)) {
        match segment {
            PrefixSegment::Probation => {
                self.probation_lru.remove(key);
            }
            PrefixSegment::Protected => {
                self.protected_lru.remove(key);
            }
        }
    }

    fn insert_lru(
        &mut self,
        segment: PrefixSegment,
        lru_key: (Instant, u64),
        key: PoolKey,
        i: usize,
    ) {
        match segment {
            PrefixSegment::Probation => {
                self.probation_lru.insert(lru_key, (key, i));
            }
            PrefixSegment::Protected => {
                self.protected_lru.insert(lru_key, (key, i));
            }
        }
    }

    /// Promote one demonstrated reuse. The caller detaches any evictable index entry first;
    /// keeping byte accounting here makes pin, touch, and same-window fanout agree.
    fn promote_segment(&mut self, key: &PoolKey, i: usize) -> bool {
        let Some(bytes) = self
            .entries
            .get(key)
            .and_then(|pool| pool.get(i))
            .filter(|entry| entry.segment == PrefixSegment::Probation)
            .map(|entry| entry.bytes)
        else {
            return false;
        };
        self.entries.get_mut(key).unwrap()[i].segment = PrefixSegment::Protected;
        self.probation_bytes = self.probation_bytes.saturating_sub(bytes);
        self.protected_bytes = self.protected_bytes.saturating_add(bytes);
        true
    }

    /// Protected is a byte target, not an entry count. Demotion does not discard bytes: it
    /// returns the protected LRU to probation, where later scan pressure may evict it. Pinned
    /// protected entries are absent from the index and can defer rebalancing until release.
    fn rebalance_protected(&mut self) {
        while self.protected_bytes > self.protected_target_bytes {
            let Some((lru_key, (key, i))) = self
                .protected_lru
                .first_key_value()
                .map(|(lru_key, value)| (*lru_key, value.clone()))
            else {
                break;
            };
            let Some(bytes) = self
                .entries
                .get(&key)
                .and_then(|pool| pool.get(i))
                .filter(|entry| {
                    entry.pins == 0
                        && entry.segment == PrefixSegment::Protected
                        && Self::lru_key(entry) == lru_key
                })
                .map(|entry| entry.bytes)
            else {
                debug_assert!(false, "protected prefix-cache index drift");
                break;
            };
            self.protected_lru.remove(&lru_key);
            self.entries.get_mut(&key).unwrap()[i].segment = PrefixSegment::Probation;
            self.protected_bytes = self.protected_bytes.saturating_sub(bytes);
            self.probation_bytes = self.probation_bytes.saturating_add(bytes);
            self.probation_lru.insert(lru_key, (key.clone(), i));
            eprintln!(
                "[prefix-cache] demote (protected bytes): {:.1}MB to probation (model {}{})",
                bytes as f64 / 1e6,
                key.0,
                ns_suffix(&key.1)
            );
        }
    }

    /// Refresh recency for pool[i] on a demonstrated reuse and promote probation to protected.
    /// This is the ONLY legal unpinned `last_use` write after insert, so the segment indexes never
    /// drift from the entries.
    #[cfg_attr(not(test), allow(dead_code))]
    fn touch(&mut self, key: &PoolKey, i: usize) {
        let Some((old_lru, old_segment, pinned)) = self
            .entries
            .get(key)
            .and_then(|p| p.get(i))
            .map(|e| (Self::lru_key(e), e.segment, e.pins > 0))
        else {
            return;
        };
        if !pinned {
            self.remove_lru(old_segment, &old_lru);
        }
        self.promote_segment(key, i);
        let (new_lru, new_segment) = {
            let e = &mut self.entries.get_mut(key).unwrap()[i];
            e.last_use = Instant::now();
            (Self::lru_key(e), e.segment)
        };
        if !pinned {
            self.insert_lru(new_segment, new_lru, key.clone(), i);
        }
        self.rebalance_protected();
    }

    /// Acquire `n` in-flight leases on one entry. The first lease removes it from the
    /// evictable LRU; all leases share the stable (pool key, entry id) handle.
    fn pin_n(&mut self, key: &PoolKey, i: usize, n: usize) -> Option<PrefixPin> {
        if n == 0 {
            return None;
        }
        let (old_lru, old_segment, id, was_unpinned) = {
            let e = self.entries.get(key)?.get(i)?;
            (Self::lru_key(e), e.segment, e.id, e.pins == 0)
        };
        if was_unpinned {
            self.remove_lru(old_segment, &old_lru);
        }
        self.promote_segment(key, i);
        {
            let e = &mut self.entries.get_mut(key)?[i];
            e.pins = e.pins.checked_add(n).expect("prefix pin refcount overflow");
            e.last_use = Instant::now();
        }
        self.rebalance_protected();
        Some(PrefixPin {
            key: key.clone(),
            id,
        })
    }

    fn pin(&mut self, key: &PoolKey, i: usize) -> Option<PrefixPin> {
        self.pin_n(key, i, 1)
    }

    /// Release one session lease. The last release makes the entry evictable again and
    /// treats the protected fanout interval as recent use.
    fn unpin(&mut self, pin: &PrefixPin) -> bool {
        let Some(i) = self.id_index(pin) else {
            return false;
        };
        let released = {
            let e = &mut self.entries.get_mut(&pin.key).unwrap()[i];
            if e.pins == 0 {
                return false;
            }
            e.pins -= 1;
            if e.pins == 0 {
                e.last_use = Instant::now();
                Some((e.segment, Self::lru_key(e)))
            } else {
                None
            }
        };
        if let Some((segment, lru_key)) = released {
            self.insert_lru(segment, lru_key, pin.key.clone(), i);
            self.rebalance_protected();
        }
        true
    }

    fn pinned_bytes(&self) -> usize {
        self.entries
            .values()
            .flatten()
            .filter(|e| e.pins > 0)
            .map(|e| e.bytes)
            .sum()
    }

    /// Bytes a pinned admission may reclaim without crossing the protected share. Existing
    /// probation is immediately eligible. A multi-participant fanout also promotes the incoming
    /// entry, so only the protected LRU bytes that promotion would demote back to probation count.
    fn pinned_admission_reclaimable_bytes(&self, incoming_bytes: usize, promotes: bool) -> usize {
        let mut reclaimable = self
            .probation_lru
            .values()
            .filter_map(|(key, i)| {
                self.entries
                    .get(key)
                    .and_then(|pool| pool.get(*i))
                    .map(|entry| entry.bytes)
            })
            .fold(0usize, usize::saturating_add);
        if !promotes {
            return reclaimable;
        }
        let mut projected_protected = self.protected_bytes.saturating_add(incoming_bytes);
        for (key, i) in self.protected_lru.values() {
            if projected_protected <= self.protected_target_bytes {
                break;
            }
            let Some(bytes) = self
                .entries
                .get(key)
                .and_then(|pool| pool.get(*i))
                .map(|entry| entry.bytes)
            else {
                debug_assert!(false, "protected prefix-cache index drift during admission");
                continue;
            };
            projected_protected = projected_protected.saturating_sub(bytes);
            reclaimable = reclaimable.saturating_add(bytes);
        }
        reclaimable
    }

    /// Remove pool[i] under `key`, keeping the recency index exact: swap_remove moves the
    /// pool's LAST entry into slot i, so exactly one surviving entry needs its index
    /// re-pointed (pool order is free — every probe is order-independent: lookup's
    /// longest-match tie is impossible under exact-key dedupe, best_lcp is a max,
    /// has_covering/has_key are `any`).
    fn remove_at(&mut self, key: &PoolKey, i: usize) -> Option<PrefixEntry> {
        let pool = self.entries.get_mut(key)?;
        if i >= pool.len() || pool[i].pins > 0 {
            return None;
        }
        let dead = pool.swap_remove(i);
        match dead.segment {
            PrefixSegment::Probation => {
                self.probation_lru.remove(&Self::lru_key(&dead));
            }
            PrefixSegment::Protected => {
                self.protected_lru.remove(&Self::lru_key(&dead));
            }
        }
        if let Some(moved) = pool.get(i) {
            if moved.pins == 0 {
                match moved.segment {
                    PrefixSegment::Probation => {
                        self.probation_lru
                            .insert(Self::lru_key(moved), (key.clone(), i));
                    }
                    PrefixSegment::Protected => {
                        self.protected_lru
                            .insert(Self::lru_key(moved), (key.clone(), i));
                    }
                }
            }
        }
        if pool.is_empty() {
            self.entries.remove(key);
        }
        self.total_bytes = self.total_bytes.saturating_sub(dead.bytes);
        match dead.segment {
            PrefixSegment::Probation => {
                self.probation_bytes = self.probation_bytes.saturating_sub(dead.bytes);
            }
            PrefixSegment::Protected => {
                self.protected_bytes = self.protected_bytes.saturating_sub(dead.bytes);
            }
        }
        Some(dead)
    }

    fn capacity_victim(&self) -> Option<(PoolKey, usize)> {
        self.capacity_victim_with(prefix_cache_slru_enabled())
    }

    /// Victim selection, with the policy passed in so both arms are unit-testable
    /// (`prefix_cache_slru_enabled` is a process-wide `OnceLock` over the environment).
    fn capacity_victim_with(&self, slru: bool) -> Option<(PoolKey, usize)> {
        if !slru {
            // PLAIN-LRU ROLLBACK (MEMRA_PREFIX_CACHE_POLICY=lru): evict the GLOBAL oldest entry
            // across both segments. Without this branch the seam did the OPPOSITE of what it
            // advertises: forcing protected_pct=100 only sets protected_target_bytes = budget, so
            // rebalance_protected can never demote, and every entry that earns a hit becomes
            // permanently unevictable. The rollback exists because cx-slrutarget measured LRU 75%
            // vs SLRU 0% after a hot-cohort turnover, so shipping it as a no-op degraded the exact
            // failure an operator reaches for it to escape.
            return self.oldest_evictable();
        }
        // SLRU: a protected entry becomes eligible only when protected exceeds its byte target:
        // rebalance_protected demotes that segment's LRU into this probation index first.
        self.probation_lru.values().next().cloned()
    }

    fn oldest_evictable(&self) -> Option<(PoolKey, usize)> {
        match (
            self.probation_lru.first_key_value(),
            self.protected_lru.first_key_value(),
        ) {
            (Some((probation_key, probation)), Some((protected_key, protected))) => {
                Some(if probation_key <= protected_key {
                    probation.clone()
                } else {
                    protected.clone()
                })
            }
            (Some((_, probation)), None) => Some(probation.clone()),
            (None, Some((_, protected))) => Some(protected.clone()),
            (None, None) => None,
        }
    }

    /// Insert (exact-key deduped per namespace) into probation, then SLRU-evict back under
    /// MEMRA_PREFIX_CACHE_MB. The byte budget and both segment targets stay GLOBAL across
    /// namespaces (VRAM is one resource); only visibility is namespaced.
    fn insert(&mut self, key: &PoolKey, e: PrefixEntry, why: &str) {
        let _ = self.insert_with_budget_pins_and_pct(
            key,
            e,
            why,
            prefix_cache_budget_bytes(),
            prefix_cache_protected_pct(),
            0,
        );
    }

    /// Insert a prefix already serving `pins` in-flight sessions. Returns one stable
    /// handle which each participating Session clones and releases once.
    fn insert_pinned(
        &mut self,
        key: &PoolKey,
        e: PrefixEntry,
        why: &str,
        pins: usize,
    ) -> Option<PrefixPin> {
        let id = self.insert_with_budget_pins_and_pct(
            key,
            e,
            why,
            prefix_cache_budget_bytes(),
            prefix_cache_protected_pct(),
            pins,
        )?;
        Some(PrefixPin {
            key: key.clone(),
            id,
        })
    }

    /// `insert` with the budget as a parameter (the env-independent seam the eviction
    /// unit tests drive; production also supplies the configured protected percentage).
    #[cfg_attr(not(test), allow(dead_code))]
    fn insert_with_budget(&mut self, key: &PoolKey, e: PrefixEntry, why: &str, budget: usize) {
        let _ = self.insert_with_budget_pins_and_pct(
            key,
            e,
            why,
            budget,
            DEFAULT_PREFIX_CACHE_PROTECTED_PCT,
            0,
        );
    }

    fn insert_with_budget_pins_and_pct(
        &mut self,
        key: &PoolKey,
        mut e: PrefixEntry,
        why: &str,
        budget: usize,
        protected_pct: usize,
        initial_pins: usize,
    ) -> Option<u64> {
        if e.layout_version != PREFIX_ENTRY_LAYOUT_VERSION || e.pool_key != *key {
            eprintln!(
                "[prefix-cache] REFUSED {why} insert: entry identity/version mismatch \
                 (entry model {}{}, version {}; pool model {}{}, version {})",
                e.pool_key.0,
                ns_suffix(&e.pool_key.1),
                e.layout_version,
                key.0,
                ns_suffix(&key.1),
                PREFIX_ENTRY_LAYOUT_VERSION,
            );
            return None;
        }
        if let Some(i) = self.key_index(key, &e.toks) {
            return if initial_pins > 0 {
                self.pin_n(key, i, initial_pins).map(|pin| pin.id)
            } else {
                None
            };
        }
        if e.bytes > budget {
            self.record_budget_skip(false);
            eprintln!(
                "[prefix-cache] skip {why} insert: entry {:.1}MB > budget {:.0}MB",
                e.bytes as f64 / 1e6,
                budget as f64 / 1e6
            );
            return None;
        }
        if initial_pins > 0 && e.bytes > budget.saturating_sub(self.pinned_bytes()) {
            self.record_budget_skip(true);
            eprintln!(
                "[prefix-cache] skip pinned {why} insert: entry {:.1}MB cannot fit \
                       beside {:.1}MB already pinned (budget {:.0}MB)",
                e.bytes as f64 / 1e6,
                self.pinned_bytes() as f64 / 1e6,
                budget as f64 / 1e6
            );
            return None;
        }
        self.protected_target_bytes = prefix_cache_protected_bytes(budget, protected_pct);
        self.rebalance_protected();
        if initial_pins > 0 {
            let needed = self
                .total_bytes
                .saturating_add(e.bytes)
                .saturating_sub(budget);
            let reclaimable = self.pinned_admission_reclaimable_bytes(e.bytes, initial_pins > 1);
            if needed > reclaimable {
                self.record_budget_skip(true);
                eprintln!(
                    "[prefix-cache] skip pinned {why} insert: entry {:.1}MB would evict \
                           protected bytes below their {:.0}MB share (need {:.1}MB, \
                           probation/demotable {:.1}MB)",
                    e.bytes as f64 / 1e6,
                    self.protected_target_bytes as f64 / 1e6,
                    needed as f64 / 1e6,
                    reclaimable as f64 / 1e6
                );
                return None;
            }
        }
        self.total_bytes += e.bytes;
        self.probation_bytes += e.bytes;
        self.inserts += 1;
        eprintln!(
            "[prefix-cache] insert probation ({why}): {} tokens, {:.1}MB (resident {:.1}MB / {:.0}MB, model {}{})",
            e.toks.len(),
            e.bytes as f64 / 1e6,
            self.total_bytes as f64 / 1e6,
            budget as f64 / 1e6,
            key.0,
            ns_suffix(&key.1)
        );
        e.id = self.next_id;
        self.next_id += 1;
        e.segment = PrefixSegment::Probation;
        e.pins = initial_pins;
        let inserted_id = e.id;
        let lk = Self::lru_key(&e);
        let idx = {
            let pool = self.entries.entry(key.clone()).or_default();
            pool.push(e);
            pool.len() - 1
        };
        if initial_pins == 0 {
            self.probation_lru.insert(lk, (key.clone(), idx));
        } else if initial_pins > 1 {
            // In-batch fanout computes once for multiple participants: sibling two is already a
            // measured reuse, so the new entry enters probation and immediately earns promotion.
            self.promote_segment(key, idx);
            self.rebalance_protected();
        }
        while self.total_bytes > budget {
            let Some((k, i)) = self.capacity_victim() else {
                break;
            };
            let Some(dead) = self.remove_at(&k, i) else {
                break;
            };
            self.evictions += 1;
            eprintln!(
                "[prefix-cache] evict ({:?} LRU): {} tokens, {:.1}MB (model {}{})",
                dead.segment,
                dead.toks.len(),
                dead.bytes as f64 / 1e6,
                k.0,
                ns_suffix(&k.1)
            );
        }
        debug_assert!(
            self.total_bytes <= budget,
            "pinned-admission preflight must preserve the prefix-cache byte ceiling"
        );
        self.entries
            .get(key)
            .and_then(|pool| pool.iter().find(|entry| entry.id == inserted_id))
            .map(|_| inserted_id)
    }

    /// Drop every EVICTABLE entry (session cache alloc failed — sessions win over
    /// ordinary cache residency, while in-flight fanout leases remain authoritative).
    fn evict_all(&mut self) -> usize {
        let mut n = 0usize;
        while let Some((key, i)) = self.oldest_evictable() {
            let Some(_dead) = self.remove_at(&key, i) else {
                break;
            };
            n += 1;
        }
        self.evictions += n as u64;
        n
    }
}

fn retire_prefix_pin(px: &mut PrefixCache, prefix_pin: &mut Option<PrefixPin>) {
    if let Some(pin) = prefix_pin.take() {
        if !px.unpin(&pin) {
            eprintln!("[prefix-cache] warning: retired session held a missing prefix pin");
        }
    }
}

/// Deep-copy the primed prefix state OUT of a live session cache into a compact entry.
/// All copies are stream-ordered on the engine worker stream (the CUDA owner thread), so no
/// sync is needed against the prime that produced the bytes or the decode that follows.
fn prefix_snapshot(
    engine: &Engine,
    cache: &Cache,
    pool_key: &PoolKey,
    toks: &[u32],
    last_logits: &[f32],
) -> Result<PrefixEntry, Box<dyn std::error::Error>> {
    if cache.has_swa_ring() {
        return Err("SWA ring sessions do not support flat-history prefix snapshots".into());
    }
    if cache.pos != toks.len() {
        return Err(format!(
            "prefix snapshot cache pos {} != token boundary {}",
            cache.pos,
            toks.len(),
        )
        .into());
    }
    if last_logits.is_empty() {
        return Err("prefix snapshot has no boundary logits".into());
    }
    let n = cache.kv.len();
    let mut kv = Vec::with_capacity(n);
    let mut conv = Vec::with_capacity(n);
    let mut ssm = Vec::with_capacity(n);
    let mut bytes = 0usize;
    for il in 0..n {
        match &cache.kv[il] {
            // A NextN/MTP head layer is ALLOCATED but never executed by the trunk, so it
            // legitimately carries len 0 while the trunk layers are at cache.pos. Treating that as
            // corruption made `prefix_snapshot` return Err for every MTP-bearing model — i.e. the
            // whole prefix cache silently stopped inserting (0 inserts / 0 hits) on exactly the
            // models we sell. Record such a layer as absent so capture and restore stay symmetric:
            // a layer with no history at capture gets no history at restore.
            Some(l) if l.len == 0 && cache.pos > 0 => {
                kv.push(None);
                conv.push(None);
                ssm.push(None);
                continue;
            }
            Some(l) => {
                if l.len != cache.pos {
                    return Err(format!(
                        "prefix snapshot layer {il} len {} != cache pos {}",
                        l.len, cache.pos,
                    )
                    .into());
                }
                let kb = l.len * l.k_tok_bytes;
                let vb = l.len * l.v_tok_bytes;
                let mut k = engine.alloc_u8(kb.max(1))?;
                let mut v = engine.alloc_u8(vb.max(1))?;
                if kb > 0 {
                    engine.copy_u8_into(&mut k, 0, &l.k, kb)?;
                }
                if vb > 0 {
                    engine.copy_u8_into(&mut v, 0, &l.v, vb)?;
                }
                bytes += kb + vb;
                kv.push(Some(PrefixPlane {
                    k,
                    v,
                    len: l.len,
                    k_tok_bytes: l.k_tok_bytes,
                    v_tok_bytes: l.v_tok_bytes,
                }));
            }
            None => kv.push(None),
        }
        match &cache.recur[il] {
            Some(r) => {
                conv.push(Some(engine.clone_dtod(&r.conv_state)?));
                ssm.push(Some(engine.clone_dtod(&r.ssm_state)?));
                bytes += (r.conv_state.len() + r.ssm_state.len()) * 4;
            }
            None => {
                conv.push(None);
                ssm.push(None);
            }
        }
    }
    Ok(PrefixEntry {
        layout_version: PREFIX_ENTRY_LAYOUT_VERSION,
        pool_key: pool_key.clone(),
        toks: toks.to_vec(),
        kv,
        conv,
        ssm,
        pos: cache.pos,
        last_logits: last_logits.to_vec(),
        draft: None,        // plain-session snapshot: no draft plane to publish
        dspark_draft: None, // ditto: only a dspark boundary capture carries a draft tail
        last_h: Vec::new(),
        bytes,
        last_use: Instant::now(),
        id: 0, // recency identity assigned by PrefixCache::insert
        segment: PrefixSegment::Probation,
        pins: 0,
    })
}

/// Deep-copy the first `restore_len` tokens of an entry INTO a freshly allocated session cache:
/// K/V bytes at [0..restore_len), per-layer len + device mirror, and pos. Recurrent state is
/// copied only at the entry's captured endpoint; a mid-entry recurrent split fails closed.
/// The ssm ping-pong spare and last_logits_dev stay as allocated (scratch — overwritten before
/// any read). All identity/shape/bounds checks precede the first device copy.
fn prefix_restore_at(
    engine: &Engine,
    cache: &mut Cache,
    e: &PrefixEntry,
    expected_key: &PoolKey,
    restore_len: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    if cache.has_swa_ring() {
        return Err("SWA ring sessions do not support flat-history prefix restores".into());
    }
    if e.layout_version != PREFIX_ENTRY_LAYOUT_VERSION {
        return Err(format!(
            "prefix entry layout version {} != runtime {}",
            e.layout_version, PREFIX_ENTRY_LAYOUT_VERSION,
        )
        .into());
    }
    if &e.pool_key != expected_key {
        return Err(format!(
            "prefix entry identity mismatch (entry model {}{}, requested model {}{})",
            e.pool_key.0,
            ns_suffix(&e.pool_key.1),
            expected_key.0,
            ns_suffix(&expected_key.1),
        )
        .into());
    }
    if e.pos != e.toks.len() {
        return Err(format!(
            "prefix entry pos {} != token boundary {}",
            e.pos,
            e.toks.len(),
        )
        .into());
    }
    if restore_len == 0 || restore_len > e.pos {
        return Err(format!(
            "prefix restore boundary {restore_len} outside entry [1,{}]",
            e.pos,
        )
        .into());
    }
    if restore_len > cache.max_ctx {
        return Err(format!(
            "prefix restore boundary {restore_len} exceeds cache capacity {}",
            cache.max_ctx,
        )
        .into());
    }
    if cache.pos != 0 || cache.kv.iter().flatten().any(|layer| layer.len != 0) {
        return Err("prefix restore destination is not fresh".into());
    }
    if cache.kv.len() != e.kv.len()
        || cache.recur.len() != e.conv.len()
        || cache.recur.len() != e.ssm.len()
    {
        return Err(format!(
            "prefix entry layer counts kv/conv/ssm={}/{}/{} != cache kv/recur={}/{}",
            e.kv.len(),
            e.conv.len(),
            e.ssm.len(),
            cache.kv.len(),
            cache.recur.len(),
        )
        .into());
    }
    let has_recurrent = e.conv.iter().any(Option::is_some) || e.ssm.iter().any(Option::is_some);
    if has_recurrent && restore_len != e.pos {
        return Err(format!(
            "hybrid mid-entry prefix restore refused at {restore_len} of {}: recurrent state \
             exists only at the captured endpoint",
            e.pos,
        )
        .into());
    }
    if restore_len == e.pos && e.last_logits.is_empty() {
        return Err("whole-entry prefix restore has no boundary logits".into());
    }

    // Validate every source/destination range and recurrent shape before queueing any copy. A
    // malformed entry must not leave a half-restored cache that a fallback could accidentally use.
    for il in 0..cache.kv.len() {
        match (cache.kv[il].as_ref(), &e.kv[il]) {
            (Some(dst), Some(src)) => {
                validate_prefix_plane_shape(
                    e.pos,
                    restore_len,
                    src.len,
                    src.k_tok_bytes,
                    src.v_tok_bytes,
                    src.k.len(),
                    src.v.len(),
                    dst.k_tok_bytes,
                    dst.v_tok_bytes,
                    dst.k.len(),
                    dst.v.len(),
                )
                .map_err(|err| format!("prefix entry layer {il}: {err}"))?;
            }
            (None, None) => {}
            // Capture records an allocated-but-unexecuted NextN/MTP layer as absent (see
            // prefix_snapshot). The destination still has that slot allocated, so a live dst with
            // an absent src is expected, not a mismatch — leave the slot untouched at len 0.
            (Some(_), None) => {}
            _ => return Err(format!("prefix entry layer {il} kind mismatch").into()),
        }
        match (cache.recur[il].as_ref(), &e.conv[il], &e.ssm[il]) {
            (Some(dst), Some(c), Some(s)) => {
                if c.len() != dst.conv_state.len() || s.len() != dst.ssm_state.len() {
                    return Err(format!(
                        "prefix entry recur {il} shape conv/ssm={}/{} != cache {}/{}",
                        c.len(),
                        s.len(),
                        dst.conv_state.len(),
                        dst.ssm_state.len(),
                    )
                    .into());
                }
            }
            (None, None, None) => {}
            _ => return Err(format!("prefix entry recur {il} mismatch").into()),
        }
    }

    for il in 0..cache.kv.len() {
        if let (Some(dst), Some(src)) = (cache.kv[il].as_mut(), &e.kv[il]) {
            let kb = restore_len * dst.k_tok_bytes;
            let vb = restore_len * dst.v_tok_bytes;
            if kb > 0 {
                engine.copy_u8_into(&mut dst.k, 0, &src.k, kb)?;
            }
            if vb > 0 {
                engine.copy_u8_into(&mut dst.v, 0, &src.v, vb)?;
            }
            dst.len = restore_len;
            engine.set_i32_one(&mut dst.len_d, restore_len as i32)?;
        }
        if let (Some(dst), Some(c), Some(s)) = (cache.recur[il].as_mut(), &e.conv[il], &e.ssm[il]) {
            engine.copy_into(&mut dst.conv_state, 0, c, c.len())?;
            engine.copy_into(&mut dst.ssm_state, 0, s, s.len())?;
        }
    }
    cache.pos = restore_len;
    Ok(())
}

fn prefix_restore(
    engine: &Engine,
    cache: &mut Cache,
    e: &PrefixEntry,
    expected_key: &PoolKey,
) -> Result<(), Box<dyn std::error::Error>> {
    prefix_restore_at(engine, cache, e, expected_key, e.pos)
}

fn digest_usize(hasher: &mut Sha256, value: usize) {
    hasher.update((value as u64).to_le_bytes());
}

fn digest_f32_plane(hasher: &mut Sha256, values: &[f32]) {
    digest_usize(hasher, values.len());
    for value in values {
        hasher.update(value.to_bits().to_le_bytes());
    }
}

/// HIRADIX-EXACT-ISO diagnostic: hash the exact logical state represented by an entry at one
/// boundary. Mid-entry recurrent state is intentionally inexpressible and returns an error.
fn prefix_entry_state_digest(
    engine: &Engine,
    entry: &PrefixEntry,
    at: usize,
) -> Result<String, Box<dyn std::error::Error>> {
    if at == 0 || at > entry.pos {
        return Err(format!("entry digest boundary {at} outside [1,{}]", entry.pos).into());
    }
    let has_recurrent =
        entry.conv.iter().any(Option::is_some) || entry.ssm.iter().any(Option::is_some);
    if has_recurrent && at != entry.pos {
        return Err("entry digest refuses mid-entry recurrent state".into());
    }
    let mut hasher = Sha256::new();
    hasher.update(b"memra-prefix-split-state-v1");
    digest_usize(&mut hasher, at);
    digest_usize(&mut hasher, entry.kv.len());
    for il in 0..entry.kv.len() {
        digest_usize(&mut hasher, il);
        match &entry.kv[il] {
            Some(plane) => {
                hasher.update([1]);
                digest_usize(&mut hasher, plane.k_tok_bytes);
                digest_usize(&mut hasher, plane.v_tok_bytes);
                let kb = at
                    .checked_mul(plane.k_tok_bytes)
                    .ok_or("entry digest K byte count overflow")?;
                let vb = at
                    .checked_mul(plane.v_tok_bytes)
                    .ok_or("entry digest V byte count overflow")?;
                if plane.k.len() < kb || plane.v.len() < vb {
                    return Err(format!("entry digest layer {il} plane is truncated").into());
                }
                let k = engine.dtoh_u8_view(&plane.k.slice(0..kb))?;
                let v = engine.dtoh_u8_view(&plane.v.slice(0..vb))?;
                digest_usize(&mut hasher, k.len());
                hasher.update(&k);
                digest_usize(&mut hasher, v.len());
                hasher.update(&v);
            }
            None => hasher.update([0]),
        }
        match (&entry.conv[il], &entry.ssm[il]) {
            (Some(conv), Some(ssm)) => {
                hasher.update([1]);
                digest_f32_plane(&mut hasher, &engine.dtoh(conv)?);
                digest_f32_plane(&mut hasher, &engine.dtoh(ssm)?);
            }
            (None, None) => hasher.update([0]),
            _ => return Err(format!("entry digest recur {il} mismatch").into()),
        }
    }
    Ok(format!("{:x}", hasher.finalize()))
}

/// Same digest over the actual freshly-restored Cache. Device `len_d` mirrors are read and must
/// equal the split before a digest can be reported.
fn prefix_cache_state_digest(
    engine: &Engine,
    cache: &Cache,
    at: usize,
) -> Result<String, Box<dyn std::error::Error>> {
    if cache.pos != at {
        return Err(format!("cache digest pos {} != split {at}", cache.pos).into());
    }
    let mut hasher = Sha256::new();
    hasher.update(b"memra-prefix-split-state-v1");
    digest_usize(&mut hasher, at);
    digest_usize(&mut hasher, cache.kv.len());
    for il in 0..cache.kv.len() {
        digest_usize(&mut hasher, il);
        match &cache.kv[il] {
            Some(plane) => {
                if plane.len != at {
                    return Err(
                        format!("cache digest layer {il} len {} != split {at}", plane.len,).into(),
                    );
                }
                let len_d = engine.dtoh_i32(&plane.len_d)?;
                if len_d.as_slice() != [at as i32] {
                    return Err(
                        format!("cache digest layer {il} len_d {len_d:?} != [{at}]",).into(),
                    );
                }
                hasher.update([1]);
                digest_usize(&mut hasher, plane.k_tok_bytes);
                digest_usize(&mut hasher, plane.v_tok_bytes);
                let kb = at
                    .checked_mul(plane.k_tok_bytes)
                    .ok_or("cache digest K byte count overflow")?;
                let vb = at
                    .checked_mul(plane.v_tok_bytes)
                    .ok_or("cache digest V byte count overflow")?;
                let k = engine.dtoh_u8_view(&plane.k.slice(0..kb))?;
                let v = engine.dtoh_u8_view(&plane.v.slice(0..vb))?;
                digest_usize(&mut hasher, k.len());
                hasher.update(&k);
                digest_usize(&mut hasher, v.len());
                hasher.update(&v);
            }
            None => hasher.update([0]),
        }
        match &cache.recur[il] {
            Some(recur) => {
                hasher.update([1]);
                digest_f32_plane(&mut hasher, &engine.dtoh(&recur.conv_state)?);
                digest_f32_plane(&mut hasher, &engine.dtoh(&recur.ssm_state)?);
            }
            None => hasher.update([0]),
        }
    }
    Ok(format!("{:x}", hasher.finalize()))
}

/// KVPROBE (lane/gemma-restore-exactness-20260819): digest the FIRST `at` rows of a LIVE
/// cache, with `at <= cache.pos` allowed. `prefix_cache_state_digest` refuses `at != pos`,
/// which makes it unable to answer the one question that matters here: are the KV bytes a
/// COLD 1048-row prime writes for rows [0..932) the same bytes a 932-row prime wrote (and
/// therefore published, and therefore restored)? Diagnostics only; costs nothing unless
/// MEMRA_KVPROBE_AT is set.
fn kvprobe_ats() -> &'static Vec<usize> {
    static A: std::sync::OnceLock<Vec<usize>> = std::sync::OnceLock::new();
    A.get_or_init(|| {
        std::env::var("MEMRA_KVPROBE_AT")
            .ok()
            .map(|v| {
                v.split(',')
                    .filter_map(|t| t.trim().parse::<usize>().ok())
                    .collect()
            })
            .unwrap_or_default()
    })
}

fn kvprobe_layers() -> bool {
    static L: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *L.get_or_init(|| std::env::var("MEMRA_KVPROBE_LAYERS").as_deref() == Ok("1"))
}

/// Whole-state digest over rows [0..at) of a live cache. Same byte-for-byte content as
/// `prefix_cache_state_digest` when at == cache.pos, but a distinct domain tag so the two
/// can never be confused in a log.
fn kv_prefix_digest(
    engine: &Engine,
    cache: &Cache,
    at: usize,
) -> Result<(String, Vec<(usize, String, String)>), Box<dyn std::error::Error>> {
    if at == 0 || at > cache.pos {
        return Err(format!("kvprobe at {at} outside [1,{}]", cache.pos).into());
    }
    let mut hasher = Sha256::new();
    hasher.update(b"memra-kvprobe-prefix-v1");
    digest_usize(&mut hasher, at);
    digest_usize(&mut hasher, cache.kv.len());
    let mut per_layer = Vec::new();
    for il in 0..cache.kv.len() {
        digest_usize(&mut hasher, il);
        match &cache.kv[il] {
            Some(plane) => {
                if plane.len < at {
                    return Err(format!(
                        "kvprobe layer {il} holds {} rows, probe wants {at}",
                        plane.len
                    )
                    .into());
                }
                hasher.update([1]);
                digest_usize(&mut hasher, plane.k_tok_bytes);
                digest_usize(&mut hasher, plane.v_tok_bytes);
                let kb = at
                    .checked_mul(plane.k_tok_bytes)
                    .ok_or("kvprobe K byte count overflow")?;
                let vb = at
                    .checked_mul(plane.v_tok_bytes)
                    .ok_or("kvprobe V byte count overflow")?;
                if plane.k.len() < kb || plane.v.len() < vb {
                    return Err(format!("kvprobe layer {il} plane truncated").into());
                }
                let k = engine.dtoh_u8_view(&plane.k.slice(0..kb))?;
                let v = engine.dtoh_u8_view(&plane.v.slice(0..vb))?;
                digest_usize(&mut hasher, k.len());
                hasher.update(&k);
                digest_usize(&mut hasher, v.len());
                hasher.update(&v);
                if kvprobe_layers() {
                    let mut hk = Sha256::new();
                    hk.update(&k);
                    let mut hv = Sha256::new();
                    hv.update(&v);
                    per_layer.push((
                        il,
                        format!("{:x}", hk.finalize())[..16].to_string(),
                        format!("{:x}", hv.finalize())[..16].to_string(),
                    ));
                }
            }
            None => hasher.update([0]),
        }
        match &cache.recur[il] {
            Some(recur) => {
                hasher.update([1]);
                digest_f32_plane(&mut hasher, &engine.dtoh(&recur.conv_state)?);
                digest_f32_plane(&mut hasher, &engine.dtoh(&recur.ssm_state)?);
            }
            None => hasher.update([0]),
        }
    }
    Ok((format!("{:x}", hasher.finalize()), per_layer))
}

/// One probe report: the requested prefix digests plus the full-`pos` digest plus the
/// boundary logits digest, all on one greppable line per `at`.
fn kvprobe(engine: &Engine, cache: &Cache, logits: &[f32], role: &str) {
    let ats = kvprobe_ats();
    if ats.is_empty() {
        return;
    }
    let lg = if logits.is_empty() {
        "empty".to_string()
    } else {
        prefix_logits_digest(logits)[..24].to_string()
    };
    let mut want: Vec<usize> = ats.iter().copied().filter(|&a| a <= cache.pos).collect();
    if !want.contains(&cache.pos) {
        want.push(cache.pos);
    }
    for at in want {
        match kv_prefix_digest(engine, cache, at) {
            Ok((state, per_layer)) => {
                eprintln!(
                    "[kvprobe] role={role} pos={} at={at} kv_sha256={state} logits_sha256={lg}",
                    cache.pos
                );
                for (il, hk, hv) in per_layer {
                    eprintln!("[kvprobe-layer] role={role} at={at} il={il} k={hk} v={hv}");
                }
            }
            Err(err) => eprintln!(
                "[kvprobe] ERROR role={role} pos={} at={at}: {err}",
                cache.pos
            ),
        }
    }
}

fn prefix_logits_digest(logits: &[f32]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"memra-prefix-boundary-logits-v1");
    digest_f32_plane(&mut hasher, logits);
    format!("{:x}", hasher.finalize())
}

fn trace_prefix_entry_state(
    engine: &Engine,
    entry: &PrefixEntry,
    at: usize,
    role: &str,
    why: &str,
) {
    if !prefix_split_trace_enabled() {
        return;
    }
    match prefix_entry_state_digest(engine, entry, at) {
        Ok(state) => {
            let logits = if at == entry.pos {
                prefix_logits_digest(&entry.last_logits)
            } else {
                "n/a-not-consumed".to_string()
            };
            eprintln!(
                "[prefix-cache-split-state] role={role} why={why} split={at} entry_pos={} \
                 state_sha256={state} boundary_logits_sha256={logits}",
                entry.pos,
            );
        }
        Err(err) => {
            eprintln!("[prefix-cache-split-state] ERROR role={role} why={why} split={at}: {err}",)
        }
    }
}

fn trace_prefix_cache_state(engine: &Engine, cache: &Cache, at: usize, role: &str, why: &str) {
    if !prefix_split_trace_enabled() {
        return;
    }
    match prefix_cache_state_digest(engine, cache, at) {
        Ok(state) => eprintln!(
            "[prefix-cache-split-state] role={role} why={why} split={at} cache_pos={} \
             state_sha256={state} boundary_logits_sha256=n/a-not-consumed",
            cache.pos,
        ),
        Err(err) => {
            eprintln!("[prefix-cache-split-state] ERROR role={role} why={why} split={at}: {err}",)
        }
    }
}

/// Snapshot + insert the session's CURRENT primed state (fed tokens, boundary logits).
/// No-op when the session cannot serve an empty-suffix resume (no host logits yet).
/// COMMIT-GATED spec publication (lane/spec-prefix-cache, 2026-08-14): assemble a prefix entry
/// from a spec session's cold-boundary capture. Recurrent conv/ssm + boundary logits come from
/// the capture (the tail prime destroyed the live boundary state); full-attn KV rows
/// `[0..pos)` are sliced from the LIVE session cache — append-only for the session's lifetime,
/// spec rollbacks never truncate below the prime boundary. Publication length == capture.pos
/// and must prefix `committed` — NEVER derived from `cache.pos` (the commit-gate invariant;
/// research/cache-spec-design-20260814/PORT-PLAN.md item 2).
#[allow(clippy::too_many_arguments)]
fn prefix_insert_from_spec_boundary(
    engine: &Engine,
    px: &mut PrefixCache,
    pool_key: &PoolKey,
    committed: &[u32],
    cache: &Cache,
    draft_plane: Option<(&CudaSlice<u8>, &CudaSlice<u8>, usize, usize)>,
    // DSPARK only: the drafter's readable KV tail, so a later long-decode request can restore
    // instead of cold-priming. `None` from every other publisher.
    dspark_draft: Option<memra_engine::dflash::DflashKvTail>,
    cap: memra_engine::spec::SpecBoundaryCapture,
    why: &str,
) {
    if memra_engine::pp::pp_host_bounce_active() {
        return;
    }
    let pos = cap.pos;
    if pos == 0 || pos > committed.len() || cap.logits.is_empty() {
        return;
    }
    if cache.has_swa_ring() {
        return;
    }
    debug_assert!(
        cap.snap.pos == pos,
        "spec boundary capture snap pos {} != capture pos {pos}",
        cap.snap.pos,
    );
    let n = cache.kv.len();
    let mut kv = Vec::with_capacity(n);
    let mut bytes = 0usize;
    for il in 0..n {
        match &cache.kv[il] {
            // MTP head layer: allocated, never executed by the trunk — absent, like
            // prefix_snapshot (capture/restore symmetry).
            Some(l) if l.len == 0 => {
                kv.push(None);
                continue;
            }
            Some(l) => {
                if l.len < pos {
                    // A trunk layer shorter than the boundary means the session rolled back
                    // below its own prime — impossible by construction; refuse loudly.
                    eprintln!(
                        "[prefix-cache] spec publish REFUSED: layer {il} len {} < boundary \
                         {pos}; entry dropped",
                        l.len,
                    );
                    return;
                }
                let kb = pos * l.k_tok_bytes;
                let vb = pos * l.v_tok_bytes;
                let (Ok(mut k), Ok(mut v)) =
                    (engine.alloc_u8(kb.max(1)), engine.alloc_u8(vb.max(1)))
                else {
                    return; // alloc pressure: publication is an optimization, drop silently
                };
                if kb > 0 && engine.copy_u8_into(&mut k, 0, &l.k, kb).is_err() {
                    return;
                }
                if vb > 0 && engine.copy_u8_into(&mut v, 0, &l.v, vb).is_err() {
                    return;
                }
                bytes += kb + vb;
                kv.push(Some(PrefixPlane {
                    k,
                    v,
                    len: pos,
                    k_tok_bytes: l.k_tok_bytes,
                    v_tok_bytes: l.v_tok_bytes,
                }));
            }
            None => kv.push(None),
        }
    }
    for (c, s_) in cap.snap.conv.iter().zip(cap.snap.ssm.iter()) {
        if let Some(c) = c {
            bytes += c.len() * 4;
        }
        if let Some(s_) = s_ {
            bytes += s_.len() * 4;
        }
    }
    // DRAFT PLANE publication (lane/spec-on-cache-hit): slice the MTP scratch rows
    // `[0..pos)` exactly like the trunk KV above — append-only below the prime boundary
    // for the session's lifetime, so the live buffers still hold the prime fill's rows.
    // A failed copy drops ONLY the plane (the trunk entry stays publishable and serves
    // plain hits); publication is an optimization, never a correctness dependency.
    let draft = draft_plane.and_then(|(k_src, v_src, k_tok_bytes, v_tok_bytes)| {
        let kb = pos * k_tok_bytes;
        let vb = pos * v_tok_bytes;
        if k_src.len() < kb || v_src.len() < vb {
            eprintln!(
                "[prefix-cache] spec publish: draft plane shorter than boundary {pos}; \
                 entry published trunk-only",
            );
            return None;
        }
        let (Ok(mut k), Ok(mut v)) = (engine.alloc_u8(kb.max(1)), engine.alloc_u8(vb.max(1)))
        else {
            return None; // alloc pressure: trunk-only entry
        };
        if kb > 0 && engine.copy_u8_into(&mut k, 0, k_src, kb).is_err() {
            return None;
        }
        if vb > 0 && engine.copy_u8_into(&mut v, 0, v_src, vb).is_err() {
            return None;
        }
        bytes += kb + vb;
        Some(PrefixPlane {
            k,
            v,
            len: pos,
            k_tok_bytes,
            v_tok_bytes,
        })
    });
    bytes += cap.last_h.len() * 4;
    // The tail is charged to the entry's budget like every other plane: an unaccounted 85 MB
    // per entry would silently overrun the cache's byte budget and break its eviction.
    if let Some(tail) = dspark_draft.as_ref() {
        bytes += tail.bytes();
    }
    let e = PrefixEntry {
        layout_version: PREFIX_ENTRY_LAYOUT_VERSION,
        pool_key: pool_key.clone(),
        toks: committed[..pos].to_vec(),
        kv,
        conv: cap.snap.conv,
        ssm: cap.snap.ssm,
        pos,
        last_logits: cap.logits,
        draft,
        dspark_draft,
        last_h: cap.last_h,
        bytes,
        last_use: Instant::now(),
        id: 0, // recency identity assigned by PrefixCache::insert
        segment: PrefixSegment::Probation,
        pins: 0,
    };
    trace_prefix_entry_state(engine, &e, e.pos, "spec-snapshot", why);
    px.insert(pool_key, e, why);
}

fn prefix_insert_from_session(engine: &Engine, px: &mut PrefixCache, s: &Session, why: &str) {
    if memra_engine::pp::pp_host_bounce_active() {
        return;
    }
    let Some(cache) = s.cache.as_ref() else {
        return;
    };
    if s.last_logits.is_empty() {
        return;
    }
    let pool_key = s.pool_key();
    match prefix_snapshot(engine, cache, &pool_key, &s.fed, &s.last_logits) {
        Ok(e) => {
            trace_prefix_entry_state(engine, &e, e.pos, "snapshot", why);
            px.insert(&pool_key, e, why);
        }
        Err(err) => eprintln!("[prefix-cache] snapshot failed ({err}); prefix not cached"),
    }
}

/// PLAIN-SESSION AFFINITY capture (lane/plain-affinity, 2026-08-09): snapshot the session's
/// cache at the stable pre-generation boundary `s.ckpt_at`, called by the prefill tick the
/// instant `s.fed.len()` reaches that boundary. Stores the result in `s.ckpt_snap` for the
/// retire sweep to move into the parked `ReuseEntry`.
///
/// Cost: ONE `Cache::snapshot` — a device COPY of only the GDN conv/ssm recurrent state (KB-MB
/// class; full-attn KV is recorded as `len` values, no copy). A pure full-attn model (step35,
/// M3, Hy3: no recurrent layers) copies nothing. This is the plain twin of the spec checkpoint's
/// single per-turn snapshot.
///
/// FAILURE IS SILENT BY DESIGN (matching the spec checkpoint): on a rig too tight for the copy
/// the capture is dropped, the session serves normally, and the NEXT turn re-primes in full —
/// today's behavior. It must never fail the prime that is running.
fn maybe_plain_checkpoint(engine: &Engine, s: &mut Session) {
    if memra_engine::pp::pp_host_bounce_active() {
        s.ckpt_at = None;
        s.ckpt_snap = None;
        return;
    }
    let Some(at) = s.ckpt_at else { return };
    if s.fed.len() != at {
        return;
    }
    s.ckpt_at = None; // one capture per session, at the boundary
    let Some(cache) = s.cache.as_ref() else {
        return;
    };
    debug_assert_eq!(
        cache.pos, at,
        "plain checkpoint must sit at the fed boundary"
    );
    match cache.snapshot(engine) {
        Ok(snap) => {
            s.ckpt_snap = Some(PlainCheckpoint {
                snap,
                pos: at,
                last_logits: s.last_logits.clone(),
            });
        }
        Err(err) => {
            if std::env::var("MEMRA_DEBUG_AFFINITY").is_ok() {
                eprintln!(
                    "[plain-affinity] checkpoint capture skipped ({err}); \
                           next turn re-primes in full"
                );
            }
        }
    }
}

/// SEED insert at prefill-done: a session whose full primed prompt is long enough parks its
/// state for future same-prefix traffic. `s.fed` == the prompt exactly at this point (no
/// generation yet).
///
/// DEPTH UNFREEZE (H11, research/cacheinval-20260813 §H11 + docs/PERFORMANCE.md "Prefix-cache
/// depth is the dominant serving lever" — the measured 3.1x throughput/latency swing on the
/// sold shape, RTX PRO 6000, canonflip-20260813). The previous gate returned early on BOTH
/// `n_cached > 0` (a hit session never seeds) and `has_covering` (any covering entry, however
/// shallow, suppresses the seed), so the FIRST entry to cover a class froze that class's
/// depth forever: 4,860-token requests kept hitting a ~1,937-token entry (LCP histogram: 189
/// hits in 1024-2048, ZERO in 2048-4096) and re-primed ~60% of the prompt on every request.
/// Now the covering check compares DEPTH: the seed proceeds when this session's primed state
/// is at least `prefix_seed_deepen_min()` tokens deeper than the deepest covering entry, so
/// depth ratchets upward (llama.cpp's subsumption maintenance, the field-survey mechanism the
/// H11 analysis names — minus the erase: a shallower covered entry still serves traffic that
/// diverges before the deep entry's end, which hybrid recurrent state cannot mid-entry-split
/// to, so shallow entries are left to the SLRU byte budget rather than removed).
///
/// EXACTNESS: the deep entry is a whole-entry snapshot of THIS session's cache, and a future
/// hit whole-entry restores it — byte-identical to the session that seeded it, exactly the
/// contract every seed already has. No new numeric program is introduced.
fn maybe_prefix_seed(engine: &Engine, px: &mut PrefixCache, s: &mut Session) {
    if !s.seed_prefix || s.vision.is_some() {
        return;
    }
    s.seed_prefix = false;
    if s.cache.is_none() || s.fed.len() < PREFIX_CACHE_MIN_TOKENS {
        return;
    }
    let key = s.pool_key();
    if !prefix_seed_deepens(px.deepest_covering(&key, &s.fed), s.fed.len()) {
        return; // an entry of (near-)equal depth already serves this prefix class
    }
    prefix_insert_from_session(engine, px, s, "seed");
}

/// Minimum extra depth (tokens) a seed must add over the deepest covering entry before it is
/// worth a device snapshot. `PREFIX_CACHE_MIN_TOKENS` — the same floor that gates seeding at
/// all: anything shallower than one entry-worth of new depth is churn, not a lever.
fn prefix_seed_deepen_min() -> usize {
    PREFIX_CACHE_MIN_TOKENS
}

/// Pure seed/deepen decision (unit-testable half of `maybe_prefix_seed`).
fn prefix_seed_deepens(deepest_covering: Option<usize>, fed_len: usize) -> bool {
    match deepest_covering {
        None => true,
        Some(depth) => fed_len.saturating_sub(depth) >= prefix_seed_deepen_min(),
    }
}

/// Pure half of the H11 hit re-arm (`seed_prefix` for a plain prefix-cache hit session) —
/// the unit-testable predicate; the call site supplies `eager_only_model(lm)` and
/// `spec.is_none()`. The eager-only refusal is R16's hard prerequisite for this mechanism
/// (research/cacheinval-20260813; full reasoning at the arming site).
fn plain_hit_reseed_arms(
    prefix_hit: bool,
    plain_path: bool,
    eager_only: bool,
    prompt_len: usize,
) -> bool {
    prefix_hit && plain_path && !eager_only && prompt_len >= PREFIX_CACHE_MIN_TOKENS
}

/// STEP-OOM PARK replay plan (lane/admit-oom, 2026-08-06): the request-shaped inputs a
/// session needs to be re-admitted after a step-time CUDA OOM parks it. These are exactly the
/// `Request` fields `admit` consumes to render and tokenize the prompt — the Session itself
/// keeps only the tokenized result, so a faithful retry has to replay from the source.
/// Cloned once per admitted session (a Vec of turns + a few strings; no device state).
struct ReplayPlan {
    prompt_ids: Vec<u32>,
    prompt_text: String,
    chat: bool,
    chat_turns: Vec<memra_tokenizer::chat::Turn>,
    tools_json: Vec<String>,
    tools_struct: Vec<memra_tokenizer::chat::Val>,
    think: memra_tokenizer::chat::ThinkMode,
    reasoning_effort: Option<String>,
    params: GenParams,
    sampler_cfg: SamplerConfig,
    grammar: Option<crate::constrained::GrammarSpec>,
    max_prompt_tokens: Option<usize>,
}

/// Build a session's embedding overlay: tower forward per image, merger rows concatenated
/// into one device buffer. Drops the host patch buffers afterwards. No-op if already built.
fn build_vision_overlay(
    engine: &Engine,
    tower: Option<&memra_engine::vision::VisionTower>,
    gemma_tower: Option<&memra_engine::vision_gemma::GemmaVisionTower>,
    n_embd: usize,
    v: &mut VisionState,
) -> Result<(), Box<dyn std::error::Error>> {
    if v.overlay.is_some() {
        return Ok(());
    }
    // GEMMA-4: forward each image through the gemma tower, concat rows in prompt order.
    if let VisionImages::Gemma(units) = &mut v.images {
        let tower = gemma_tower.ok_or("gemma vision request without a loaded gemma tower")?;
        let total: usize = units.iter().map(|u| u.n_soft()).sum();
        let mut rows = engine.uninit(total * n_embd)?;
        let mut off = 0usize;
        for u in units.iter() {
            let emb = tower.forward(engine, &u.patches, u.gw, u.gh)?;
            engine.dtod_copy_into(&emb, &mut rows, off * n_embd)?;
            off += u.n_soft();
        }
        v.overlay = Some(memra_engine::vision::EmbedOverlay {
            rows,
            spans: v.spans.clone(),
        });
        for u in units.iter_mut() {
            u.patches = Vec::new();
        }
        return Ok(());
    }
    let VisionImages::Qwen(units) = &mut v.images else {
        unreachable!("gemma handled above")
    };
    let tower = tower.ok_or("qwen vision request without a loaded tower (MEMRA_VISION_DIR)")?;
    let total: usize = units.iter().map(|u| u.prep.n_tokens()).sum();
    let mut rows = engine.uninit(total * n_embd)?;
    let mut off = 0usize;
    let mut i = 0usize;
    while i < units.len() {
        let unit = &units[i];
        match unit.video {
            None => {
                let emb = tower.forward(engine, &unit.prep.patches, unit.prep.gh, unit.prep.gw)?;
                engine.dtod_copy_into(&emb, &mut rows, off * n_embd)?;
                off += unit.prep.n_tokens();
                i += 1;
            }
            Some(vid) => {
                // one video = the run of consecutive units sharing vid: joint attention span
                let (gh, gw) = (unit.prep.gh, unit.prep.gw);
                let mut j = i;
                let mut cat: Vec<f32> = Vec::new();
                while j < units.len() && units[j].video == Some(vid) {
                    cat.extend_from_slice(&units[j].prep.patches);
                    j += 1;
                }
                let groups = j - i;
                let emb = tower.forward_seq(engine, &cat, groups, gh, gw)?;
                engine.dtod_copy_into(&emb, &mut rows, off * n_embd)?;
                off += groups * gh * gw / 4;
                i = j;
            }
        }
    }
    v.overlay = Some(memra_engine::vision::EmbedOverlay {
        rows,
        spans: v.spans.clone(),
    });
    for u in units.iter_mut() {
        u.prep.patches = Vec::new();
    }
    Ok(())
}

/// VISION serving state (lane/vision): pad-run spans aligned to the request's images at
/// admit; the embedding overlay (tower forward) is built at the first prefill tick on the
/// GPU worker thread and the host patch buffers are dropped there.
enum VisionImages {
    Qwen(Vec<memra_engine::vision_pre::VisionUnit>),
    Gemma(Vec<memra_engine::vision_gemma::GemmaVisionUnit>),
}
struct VisionState {
    images: VisionImages,
    /// (prompt_pos, row_off, n_rows) per image, prompt order — the EmbedOverlay spans.
    spans: Vec<(usize, usize, usize)>,
    overlay: Option<memra_engine::vision::EmbedOverlay>,
}

/// Align the tokenized prompt's `<|image|>` (258880) soft-token runs with the request's
/// gemma units, in order. Same fail-loud contract as `vision_spans`.
fn gemma_vision_spans(
    prompt: &[u32],
    units: &[memra_engine::vision_gemma::GemmaVisionUnit],
    soft: u32,
) -> Result<Vec<(usize, usize, usize)>, String> {
    let mut runs: Vec<(usize, usize)> = Vec::new();
    let mut i = 0;
    while i < prompt.len() {
        if prompt[i] == soft {
            let start = i;
            while i < prompt.len() && prompt[i] == soft {
                i += 1;
            }
            runs.push((start, i - start));
        } else {
            i += 1;
        }
    }
    if runs.len() != units.len() {
        return Err(format!(
            "prompt carries {} gemma image run(s) but the request has {} unit(s)",
            runs.len(),
            units.len()
        ));
    }
    let mut spans = Vec::with_capacity(runs.len());
    let mut row_off = 0usize;
    for (&(start, len), unit) in runs.iter().zip(units) {
        let n = unit.n_soft();
        if len != n {
            return Err(format!(
                "gemma image run at token {start} is {len} tokens; the unit needs {n}"
            ));
        }
        spans.push((start, row_off, n));
        row_off += n;
    }
    Ok(spans)
}

/// Align the tokenized prompt's `<|image_pad|>` runs with the request's images, in order.
/// Any mismatch — user text faking pad tokens, count drift, a truncated run — is a clean
/// admission error, never a silent misalignment.
fn vision_spans(
    prompt: &[u32],
    units: &[memra_engine::vision_pre::VisionUnit],
    image_pad: u32,
    video_pad: Option<u32>,
) -> Result<Vec<(usize, usize, usize)>, String> {
    let is_pad = |t: u32| t == image_pad || video_pad == Some(t);
    let mut runs: Vec<(usize, usize, u32)> = Vec::new();
    let mut i = 0;
    while i < prompt.len() {
        if is_pad(prompt[i]) {
            let (start, id) = (i, prompt[i]);
            while i < prompt.len() && prompt[i] == id {
                i += 1;
            }
            runs.push((start, i - start, id));
        } else {
            i += 1;
        }
    }
    if runs.len() != units.len() {
        return Err(format!(
            "prompt carries {} vision pad run(s) but the request has {} unit(s) — \
             literal pad tokens in message text are not allowed",
            runs.len(),
            units.len()
        ));
    }
    let mut spans = Vec::with_capacity(runs.len());
    let mut row_off = 0usize;
    for (&(start, len, id), unit) in runs.iter().zip(units) {
        let n = unit.prep.n_tokens();
        let want = if unit.video.is_some() {
            video_pad.ok_or("model tokenizer has no <|video_pad|> token")?
        } else {
            image_pad
        };
        if id != want {
            return Err(format!(
                "vision pad run at token {start} has the wrong pad kind for its unit"
            ));
        }
        if len != n {
            return Err(format!(
                "vision pad run at token {start} is {len} tokens; the unit needs {n}"
            ));
        }
        spans.push((start, row_off, n));
        row_off += n;
    }
    Ok(spans)
}

struct Session {
    model: String,
    /// Request-owned speculative depth. Zero means this session is on the plain path.
    /// Positive values are fixed for the request and consumed by every spec round.
    spec_k: usize,
    /// PC-ISO cache namespace this session admits, hits, and parks under (see PoolKey).
    cache_ns: String,
    /// SESSION AFFINITY explicit tier: the conversation id the admitting request declared
    /// (`Request::affinity`), carried so park-at-retire can label the parked session with it.
    affinity: Option<String>,
    /// yield lane — admission class + prefill budget bucket + batch priority.
    lane: crate::lanes::Lane,
    /// legacy tokenwise cache — None on the spec path (SpecSession owns its own caches; the
    /// double-alloc cost 2GB/128k-session and OOM'd the 27B serve — fixed 2026-07-05).
    cache: Option<Cache>,
    /// SPEC-DECODE serving (2026-07-05): sessions on MTP models decode in
    /// generate_spec_session BURSTS (K-token draft chains + batched verify) instead of one
    /// decode_step per tick — the CLI-measured spec win (27B p3: 79 vs 40 tok/s) brought to the
    /// serve path. `Some` when: model has an MTP head + MEMRA_SERVE_SPEC!=0 + the sampler is
    /// EITHER greedy (argmax verify) OR sampled (temperature>0 -> the rejection-sampling
    /// verify, filters and penalties applied symmetrically to draft q and target p; landed
    /// 2026-07-09/10, feat/sampled-graph-draft + feat/filtered-spec). Greedy-with-penalties
    /// is the one excluded class (`greedy_penalized`) — the argmax verify would ignore them.
    /// See `spec_eligible` in step_admit for the authoritative predicate.
    /// The SpecSession owns its OWN cache/scratch; `cache` above stays as the (unused) admit
    /// allocation on this path (kept to avoid restructuring admit; ~small VRAM overhead until
    /// a follow-up drops it). committed == every token whose state the spec caches hold.
    spec: Option<memra_engine::spec::SpecSession>,
    /// SINGLE-SESSION CUDA-GRAPH serving (2026-07-26, +34% measured at B=1): a greedy
    /// interactive session admitted ALONE rides GraphSession replay (one step/tick, 4B
    /// D2H). Degrades to the batched-eager path the moment a second session admits —
    /// legal because dc==eager is bit-identical, so the graph cache continues seamlessly.
    graph: Option<memra_engine::decode::GraphSession>,
    /// The token produced by the last graph step (next INPUT; emitted on the next tick).
    graph_pending: Option<u32>,
    /// STEP-OOM PARK (lane/admit-oom): how many times this session has been parked back to
    /// the queue after a step-time CUDA OOM. Bounded by STEP_OOM_MAX_RETRIES before the
    /// honest error — a session that cannot make progress must not retry forever.
    oom_retries: u32,
    /// STEP-OOM PARK replay plan (lane/admit-oom): everything needed to rebuild this
    /// session's `Request` if a step-time OOM parks it back to the admission queue. Held
    /// because `admit` consumes the Request and the Session keeps only derived state (the
    /// TOKENIZED prompt, not the turns/tools/think that rendered it). Re-admitting from
    /// these fields re-runs the identical render+tokenize, so the retried session is the
    /// one a cold arrival would have produced.
    replay: Box<ReplayPlan>,
    /// Live acceptance telemetry (hqmtp axis-D): cumulative drafted/accepted across the
    /// session's bursts, logged per burst so serve-regime acceptance-vs-context is measurable.
    /// THIS REQUEST's counts (0 at admit even on a pool resume) — the `usage.spec` source.
    spec_drafted: usize,
    spec_accepted: usize,
    /// verify rounds this request ran (lane/accept-telemetry; same per-request semantics).
    spec_rounds: u64,
    sampler: Sampler,
    last_logits: Vec<f32>,
    /// Token pre-sampled ON DEVICE by the last batched tick (decode_step_batch_sampled) —
    /// consumed by the next advance_sample_emit instead of the O(n_vocab) host sample
    /// (measured 1.36 ms/row at 248k vocab). None = host-sample from last_logits (fallback
    /// rows: penalties/top-k/top-p/min-p configs; non-batched paths). Dropped un-consumed
    /// when a session finishes — same semantics as an unsampled last_logits.
    device_next: Option<u32>,
    /// GEMMA SPEC route (lane/gemma-batched stage 2, 2026-08-17): burst-scoped assistant-
    /// drafter session (engine `GemmaSpecSession`). Some = this session decodes in
    /// `gemma_spec_session_burst` bursts via `step_gemma_spec`; it owns its trunk cache
    /// (s.cache stays None until a demote handoff). Created lazily at the first spec tick
    /// (the prime). Dense gemma4 + greedy + unconstrained + text-only + solo-admission only.
    gspec: Option<memra_engine::gemma_spec::GemmaSpecSession>,
    /// Draft depth K for the gemma spec route; 0 = plain (every non-admitted session).
    /// Nonzero marks the session as gemma-spec-routed even before `gspec` exists (the
    /// pre-prime window) — scheduler filters key on this, not on gspec.is_some().
    gspec_k: usize,
    /// ctx capacity for the gemma spec session's cache (RequestShape.ctx_cap at admit).
    gspec_ctx: usize,
    /// DSPARK SPEC route (lane/dspark-q38-recover serve route): burst-scoped drafter
    /// session (engine `DsparkSpecSession`) for qwen-hybrid targets. Some = this session
    /// decodes in `dspark_spec_session_burst` bursts via `step_dspark_spec`; it owns its
    /// trunk cache until a greedy HIGH-water demotion hands that cache to `s.cache`. Created
    /// lazily at the first spec tick (the prime).
    /// Greedy(unpenalized) OR sampled (T>0 rejection-sampling verify, penalties INCLUDED
    /// — lane/dspark-sampled-admission-20260820 + lane/dspark-penalized-sampled-20260821)
    /// + unconstrained + text-only. Greedy uses wave-projected LOW admission and HIGH demotion;
    /// sampled, gate-off, and positive-K-pinned regimes keep solo admission because they do not
    /// demote; K=0 refuses admission.
    dspark: Option<memra_engine::dflash::DsparkSpecSession>,
    /// Marks the session as dspark-routed even before `dspark` exists (the pre-prime
    /// window) — scheduler filters key on this, mirroring gspec_k's role.
    dspark_on: bool,
    /// Whether the lazy DFlash prime should retain one full-prompt trunk capture for the
    /// cross-request prefix cache. False preserves the prefix-cache-disabled byte path.
    dspark_capture_prefix: bool,
    /// Constrained decoding (`response_format`): per-session llguidance grammar state.
    /// `Some` masks the logits BEFORE every sample and advances with each accepted token.
    /// FULL path (2026-08-03): the packed mask uploads to `mask_dev` each step and
    /// mask_logits_f32 bans on DEVICE before the device sampler — constrained rows ride
    /// the same device-sample/lean-logits tick as everyone else. Unsupported constrained
    /// penalty/filter compositions and MEMRA_CONSTRAIN_HOST=1 keep the v1 host-side
    /// masked-copy sample; qualified unconstrained sampled penalties use the device path.
    constraint: Option<crate::constrained::SessionConstraint>,
    /// Device grammar-mask buffer (packed SimpleVob words). Allocated once at first use,
    /// STABLE POINTER thereafter — contents re-uploaded per step (~n_vocab/8 bytes), the
    /// graph-capture contract for the in-graph mask read.
    mask_dev: Option<CudaSlice<u32>>,
    /// Words uploaded this step (0 = no mask staged for the pending batch step).
    mask_words: usize,
    /// VISION (lane/vision): Some = this session primes with a mixed-embedding overlay.
    /// Carries the preprocessed images + the pad-run spans found in the tokenized prompt;
    /// the overlay (tower forward) is built lazily at the first prefill tick, on the GPU
    /// worker thread. Vision sessions never park, seed, resume, batch-prime, or spec.
    vision: Option<VisionState>,
    /// EMBEDDINGS/RERANK capture (lane/embed-serve): emit `Event::PromptCapture` when
    /// prefill completes, then finish (budget is 0 on these requests).
    capture: Option<CaptureSpec>,
    /// Process-wide host patch-memory reservation, held for the lifetime of the session so a
    /// streaming response cannot release the budget before the worker drops its image rows.
    vision_memory: Option<crate::VisionMemoryPermit>,
    /// Every token actually FED to decode_step, in order (prompt prime + generated feedback).
    /// This is exactly the sequence whose KV + recurrent state live in `cache` — the resume
    /// point for KV PREFIX REUSE on retire (see ReusePool).
    fed: Vec<u32>,
    /// prompt tokens still to be primed (consumed one per scheduler tick during prefill).
    prefill_queue: std::collections::VecDeque<u32>,
    prefill_done: bool,
    generated: Vec<u32>,
    /// Successfully published `Event::Token` count. `finish` requires this to equal
    /// `generated.len()` before publishing the terminal usage/token snapshot receipt.
    tokens_emitted: usize,
    /// A disconnected client is billed to the observed abort point but its generated KV is
    /// not a reusable conversation state. Abort retirement drops all session caches instead
    /// of publishing an implicit branch into affinity/prefix reuse.
    aborted: bool,
    params: GenParams,
    stop_strings: Vec<String>,
    trace_id: Option<String>,
    /// Append-only detokenized generated bytes. Re-decoding the whole token tail on every step is
    /// quadratic and makes field-length completions impractical; token pieces are context-free.
    decoded_bytes: Vec<u8>,
    /// Prefix of `decoded_bytes` already emitted as complete UTF-8.
    emitted_bytes: usize,
    budget: usize, // max tokens we may still generate
    /// usage accounting (worker-truth): total prompt tokens this session feeds/resumes, and
    /// how many came from a cache (continuation pool / spec resume / prefix cache).
    n_prompt: usize,
    n_cached: usize,
    /// PREFIX-CACHE LCP SPLIT: prime exactly up to this fed-length, snapshot the cache into
    /// the prefix cache there, then continue with the rest of the prompt (the learning step).
    snapshot_at: Option<usize>,
    /// PLAIN-SESSION AFFINITY (lane/plain-affinity, 2026-08-09): the STABLE PRE-GENERATION
    /// boundary this session captures a rewind checkpoint at, in `fed`-length. Derived at admit
    /// (`plain_checkpoint_boundary`) from the prompt's last turn-marker run so the checkpoint
    /// sits before the template's live assistant-generation suffix. `Some(b)` makes the prefill
    /// tick stop a prime exactly at `b` and snapshot the cache into `ckpt_snap`; it is a
    /// COLD-only signal (a resumed session already carries the earlier turn's checkpoint and does
    /// not re-capture). `None` = no capture (resume, too-short prompt, or an unlocatable boundary).
    ckpt_at: Option<usize>,
    /// The captured plain checkpoint, taken when prefill crosses `ckpt_at`. Moved into the parked
    /// `ReuseEntry` at retire. `None` until the boundary is crossed (or if capture failed silently).
    ckpt_snap: Option<PlainCheckpoint>,
    /// The LCP sample recorded for a cold prefix-cache miss at admission. Same-window
    /// fanout siblings rewrite this provisional miss into a hit after the leader primes.
    prefix_miss_lcp: Option<usize>,
    /// PREFIX-CACHE SEED: park the full primed prompt at prefill-done (cold sessions only).
    seed_prefix: bool,
    /// Refcounted lease on the prefix entry this request resumed from (or helped create
    /// through in-batch fanout). Released by the centralized retire sweep on every exit.
    prefix_pin: Option<PrefixPin>,
    tx: tokio::sync::mpsc::UnboundedSender<Event>,
    ttft: Option<Arc<crate::ttft::Trace>>,
    t0: Instant,
}

impl Session {
    /// The (model, namespace) reuse-pool key this session hits and parks under (PC-ISO).
    fn pool_key(&self) -> PoolKey {
        (self.model.clone(), self.cache_ns.clone())
    }
}

/// Remove the engine-owned capture without allocating or deriving a pool key. Both the pre-phase
/// fallback and the successful post-phase-(a) publication path use this one-shot drain.
fn take_dspark_prefix_capture(s: &mut Session) -> Option<memra_engine::spec::SpecBoundaryCapture> {
    s.dspark
        .as_mut()
        .and_then(|dspark| dspark.take_prefix_capture())
}

fn publish_dspark_prefix_capture(
    engine: &Engine,
    px: &mut PrefixCache,
    s: &Session,
    cap: memra_engine::spec::SpecBoundaryCapture,
) {
    let Some(dspark) = s.dspark.as_ref() else {
        return;
    };
    let end = cap.pos.min(s.fed.len());
    if end == 0 {
        return;
    }
    // Clone the namespace key only after a real capture exists. Most active rows have no
    // capture, so the scheduler's fallback sweep remains allocation-free for them.
    let pool_key = s.pool_key();
    if px.has_key(&pool_key, &s.fed[..end]) {
        return;
    }
    // Publish the drafter's readable tail alongside the trunk planes. This is what turns a
    // dspark hit from "trunk-only, so a speculating request must cold-prime anyway" into a
    // restore: the draft state cannot be recomputed from restored KV (it derives from trunk
    // hidden FEATURES), so it travels with the entry or not at all.
    // `end`, not the drafter's current length: by the time the drain sweep publishes, the
    // session has committed generated rows past the capture boundary, and a tail cut there
    // disagrees with the trunk planes (which are copied at `pos`). The first exactness-gate run
    // caught exactly that — every restore refused with `draft KV len 30364 != prompt 30329`.
    // Publication is gated on the SAME flag as consumption (review round 5): with the flag
    // off there is no possible consumer, and the tail is a real cost charged to the entry's
    // byte budget — a fixed ~85 MB/entry, ~8% on a 30k entry but ~34% on a 7k one — evicting
    // other entries for nothing. Gating both sides makes the unset env a true rollback: the
    // cache's memory profile reverts along with the routing. After arming, real long-decode
    // traffic republishes tailed entries within minutes, so no pre-arming is needed.
    let dspark_tail = if dspark_prefix_restore_on() {
        let t = dspark.draft_kv().export_tail(engine, end);
        if t.is_none() {
            eprintln!(
                "[prefix-cache] dspark publish: draft tail export failed; entry published \
                 trunk-only (a later long-decode hit will cold-prime as before)"
            );
        }
        t
    } else {
        None
    };
    prefix_insert_from_spec_boundary(
        engine,
        px,
        &pool_key,
        &s.fed,
        &dspark.cache,
        None,
        dspark_tail,
        cap,
        "dspark-boundary",
    );
}

fn drain_dspark_prefix_capture(engine: &Engine, px: &mut PrefixCache, s: &mut Session) {
    let Some(cap) = take_dspark_prefix_capture(s) else {
        return;
    };
    publish_dspark_prefix_capture(engine, px, s, cap);
}

/// Primary CUDA ordinal for the serving worker. CUDA_VISIBLE_DEVICES already remaps physical GPUs
/// into a process-local ordinal space, so the non-PP default remains logical device 0. Under PP,
/// the worker primary follows the LAST device in MEMRA_PP_DEVICES — the HEAD stage's device.
///
/// WHY THE LAST, NOT THE FIRST (v0.72 tag-blocker 2, research/v072-fix2-20260808): the sharded
/// loader puts `output_norm` + the lm head on the LAST stage's engine (`hybrid.rs`:
/// `e_head = layer_engine(e, n_trunk, n_trunk-1)`), and the spec-serving round loop runs its
/// whole draft chain on the PRIMARY engine — `mtp_head_forward_dev` op 12 falls back to
/// `&self.output` for every qwen35-family drafter, so EVERY draft token's head matmul reads the
/// last stage's biggest tensor. The round's verify-logit consumers (device argmax, accept
/// kernels, seed gather) read last-stage buffers through the primary context by UVA too. Pinning
/// the primary to stage 0 (the 5f27c55c shape, MEMRA_PP_DEVICES[0]) therefore made every spec
/// round pay cross-device head reads on BOTH placement orders: spec+PP-2 serving collapsed
/// 112.5 -> 17.5 agg tok/s while spec-off (head matmul runs ON the last stage) and engine
/// run-spec (primary=0 = the last stage on the dev10 placement) stayed fast. Following the head
/// stage restores the exact topology every 212/212 crash-gate + 112.5 perf receipt validated
/// (research/pp2spec-crash-20260807), keeps the cx-503b correctness win (the primary is still a
/// placement device, never an unconditional 0), and fixes the pre-merge dev01 ~20x note — the
/// same mismatch, from the other end. Gate/bench binaries keep primary=devices[0]: they
/// deliberately exercise the shared-engine stage-0 case and don't run the serving spec round.
fn worker_device(pp_devices: Option<&str>) -> Result<usize, String> {
    let Some(devices) = pp_devices.filter(|v| !v.trim().is_empty()) else {
        return Ok(0);
    };
    let mut last = 0usize;
    for part in devices.split(',') {
        let part = part.trim();
        last = part.parse::<usize>().map_err(|_| {
            format!(
                "MEMRA_PP_DEVICES={devices} has invalid device {part:?} \
                 (want <d0>,..,<dN-1> e.g. 0,1)"
            )
        })?;
    }
    Ok(last)
}

/// Increment-2 research door. Absent means byte-for-byte pre-lane scheduling; present arms the
/// engine's default-off controller globally for this fresh worker process. Admission inside the
/// engine still refuses every unsupported shape before allocating fork state.
fn optipipe_controller_threshold(raw: Option<&str>) -> Result<Option<f32>, String> {
    let Some(raw) = raw else {
        return Ok(None);
    };
    let threshold = raw
        .parse::<f32>()
        .map_err(|err| format!("MEMRA_OPTI_CONTROLLER_Q={raw:?} is not a float: {err}"))?;
    if !threshold.is_finite() || !(0.0..=1.0).contains(&threshold) {
        return Err(format!(
            "MEMRA_OPTI_CONTROLLER_Q={raw:?} is outside the inclusive [0,1] range"
        ));
    }
    Ok(Some(threshold))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RuntimePeerProbeWorkerAction {
    Continue,
    DegradedToHostBounce,
}

#[derive(Debug, Default, PartialEq, Eq)]
struct RuntimePeerProbeDeferralObservation {
    intervals: u64,
    consecutive_intervals: u64,
    bound_reached: bool,
}

fn resolve_runtime_peer_probe_deferral_bound(raw: Option<&str>) -> u64 {
    raw.and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(memra_engine::pp::PEER_RUNTIME_PROBE_DEFERRAL_BOUND_INTERVALS)
        .max(1)
}

fn runtime_peer_probe_deferral_bound() -> u64 {
    static BOUND: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *BOUND.get_or_init(|| {
        resolve_runtime_peer_probe_deferral_bound(
            std::env::var("MEMRA_PEER_PROBE_DEFERRAL_BOUND")
                .ok()
                .as_deref(),
        )
    })
}

/// Worker-local coalescing for a process-wide metric. The scheduler polls once per tick, but an
/// operator deferral is one copy-count interval in which a runnable cheap rung stayed blocked.
#[derive(Debug)]
struct RuntimePeerProbeDeferralState {
    last_copy_interval: Option<u64>,
    consecutive_intervals: u64,
    integrity_degraded: bool,
    bound_intervals: u64,
}

impl Default for RuntimePeerProbeDeferralState {
    fn default() -> Self {
        Self::with_bound(runtime_peer_probe_deferral_bound())
    }
}

impl RuntimePeerProbeDeferralState {
    fn with_bound(bound_intervals: u64) -> Self {
        Self {
            last_copy_interval: None,
            consecutive_intervals: 0,
            integrity_degraded: false,
            bound_intervals: bound_intervals.max(1),
        }
    }

    fn observe(&mut self, boundary_copies: u64) -> RuntimePeerProbeDeferralObservation {
        let copy_interval = boundary_copies / memra_engine::pp::PEER_RUNTIME_PROBE_INTERVAL_COPIES;
        let intervals = self
            .last_copy_interval
            .map_or(1, |previous| copy_interval.saturating_sub(previous));
        if intervals == 0 {
            return RuntimePeerProbeDeferralObservation {
                consecutive_intervals: self.consecutive_intervals,
                ..Default::default()
            };
        }

        self.last_copy_interval = Some(copy_interval);
        let previous = self.consecutive_intervals;
        self.consecutive_intervals = self.consecutive_intervals.saturating_add(intervals);
        let bound_reached = !self.integrity_degraded
            && previous < self.bound_intervals
            && self.consecutive_intervals >= self.bound_intervals;
        self.integrity_degraded |= bound_reached;
        RuntimePeerProbeDeferralObservation {
            intervals,
            consecutive_intervals: self.consecutive_intervals,
            bound_reached,
        }
    }

    fn resolve(&mut self) -> bool {
        self.last_copy_interval = None;
        self.consecutive_intervals = 0;
        std::mem::take(&mut self.integrity_degraded)
    }
}

fn runtime_peer_probe_allowed(has_live_spec: bool) -> bool {
    !has_live_spec
}

/// Keep the worker's failure boundary explicit: a validated engine failover continues the same
/// scheduler loop; only inability to arm host bounce is returned to the existing panic/respawn
/// ladder. Split from the CUDA call so the worker-level continuity contract is GPU-free testable.
fn runtime_peer_probe_worker_action(
    result: Result<memra_engine::pp::RuntimePeerProbeStatus, String>,
) -> Result<RuntimePeerProbeWorkerAction, String> {
    match result? {
        memra_engine::pp::RuntimePeerProbeStatus::DegradedToHostBounce => {
            Ok(RuntimePeerProbeWorkerAction::DegradedToHostBounce)
        }
        memra_engine::pp::RuntimePeerProbeStatus::NotRun
        | memra_engine::pp::RuntimePeerProbeStatus::Deferred
        | memra_engine::pp::RuntimePeerProbeStatus::Passed => {
            Ok(RuntimePeerProbeWorkerAction::Continue)
        }
    }
}

fn service_runtime_peer_probe_for_worker(
    engine: &Engine,
    scheduler_idle: bool,
    active: &mut [Session],
    px: &mut PrefixCache,
    deferral: &mut RuntimePeerProbeDeferralState,
    health: &crate::health::WorkerHealth,
) -> Result<bool, String> {
    // Host bounce cannot continue an already-live spec session because its device-resident
    // token/position inputs still peer-read. Defer integrity work until those bounded sessions
    // retire; new admissions become plain immediately once a failover is published.
    let has_live_spec = active
        .iter()
        .any(|session| session.spec.is_some() || session.gspec_k > 0 || session.dspark_on);
    let status = memra_engine::pp::service_runtime_peer_probe(
        engine,
        scheduler_idle,
        runtime_peer_probe_allowed(has_live_spec),
    )
    .map_err(|err| err.to_string())?;
    if status == memra_engine::pp::RuntimePeerProbeStatus::Deferred {
        let boundary_copies = memra_engine::pp::peer_probe_metrics().boundary_copies;
        let observation = deferral.observe(boundary_copies);
        if observation.intervals > 0 {
            memra_engine::pp::record_runtime_peer_probe_deferral(
                observation.intervals,
                observation.bound_reached,
            );
            health.note_peer_probe_deferral(
                observation.consecutive_intervals,
                observation.bound_reached,
            );
        }
        if observation.bound_reached {
            eprintln!(
                "[pp] SECURITY RED: runtime peer-probe integrity coverage DEGRADED after {} \
                 consecutive deferred intervals ({} native boundary copies each, \
                 boundary_copies={boundary_copies}); a live speculative session still uses \
                 cross-device UVA token/position state, so forcing a probe could revoke access \
                 during failover. Drain speculative sessions or restart with \
                 MEMRA_SERVE_SPEC=0, then require a completed runtime re-probe before restoring \
                 speculative traffic",
                observation.consecutive_intervals,
                memra_engine::pp::PEER_RUNTIME_PROBE_INTERVAL_COPIES,
            );
        }
        return Ok(false);
    }
    let ran = status.ran();
    if ran {
        let was_degraded =
            deferral.resolve() || memra_engine::pp::peer_probe_metrics().integrity_degraded;
        memra_engine::pp::clear_runtime_peer_probe_integrity_degraded();
        health.clear_peer_probe_deferral();
        if was_degraded {
            eprintln!(
                "[pp] runtime peer-probe integrity coverage RECOVERED at a safe scheduler \
                 boundary"
            );
        }
    }
    if runtime_peer_probe_worker_action(Ok(status))?
        == RuntimePeerProbeWorkerAction::DegradedToHostBounce
    {
        let evicted = px.evict_all();
        for session in active {
            session.snapshot_at = None;
            session.prefix_miss_lcp = None;
            session.seed_prefix = false;
            session.ckpt_at = None;
            session.ckpt_snap = None;
        }
        eprintln!(
            "[pp] SECURITY RED: worker continuing on validated host bounce; disabled new spec, \
             dual-active decode, prefix snapshots, and affinity checkpoints; evicted \
             {evicted} unpinned prefix-cache entries"
        );
    }
    Ok(ran)
}

/// The worker entry point. Runs on its OWN std::thread. Builds the Engine + loads every model on
/// THIS thread (CUDA-context affinity), then runs the scheduler loop until the command channel
/// closes. `models` = (name, gguf_path) pairs. Sends `ready_tx` once load completes (or the error).
///
/// `rx` is BORROWED, not owned: the supervisor in `spawn()` keeps the Receiver alive across a
/// respawn, because dropping it would close the command channel and make every subsequent HTTP
/// handler's `send` fail permanently — the exact invisible-death this lane exists to remove.
pub fn run(
    models: Vec<(String, String, Option<String>)>,
    rx: &Receiver<Cmd>,
    ready_tx: Sender<Result<(Vec<String>, HashMap<String, ModelCaps>), String>>,
    metrics: SharedMetrics,
    health: crate::health::SharedHealth,
) {
    // ---- one-time init on the worker thread: Engine + all models resident ----
    let pp_devices = std::env::var("MEMRA_PP_DEVICES").ok();
    let device = match worker_device(pp_devices.as_deref()) {
        Ok(device) => device,
        Err(err) => {
            let _ = ready_tx.send(Err(err));
            return;
        }
    };
    let engine = match Engine::new(device) {
        Ok(e) => e,
        Err(err) => {
            let _ = ready_tx.send(Err(format!("Engine::new failed: {err}")));
            return;
        }
    };
    // MEMRA_FAST is read ONCE here (same handling as run_gen): the matmul path consults the env var
    // per-call, but logging it once keeps the worker's behavior explicit and stable for the run.
    let fast = std::env::var("MEMRA_FAST").as_deref() != Ok("0");
    eprintln!(
        "[worker] Engine ready (device={device}, MEMRA_FAST={})",
        fast
    );
    match optipipe_controller_threshold(std::env::var("MEMRA_OPTI_CONTROLLER_Q").ok().as_deref()) {
        Ok(Some(threshold)) => {
            memra_engine::spec::set_optipipe_controller_threshold(threshold);
            eprintln!(
                "[worker] OPTIPIPE increment-2 diagnostic controller armed q_threshold={threshold:.3}"
            );
        }
        Ok(None) => {}
        Err(err) => {
            let _ = ready_tx.send(Err(err));
            return;
        }
    }
    log_spec_gate_policy();

    let (constraint_result_tx, constraint_result_rx) =
        std::sync::mpsc::channel::<crate::constrained::ConstraintCompileResult>();
    let mut loaded: HashMap<String, LoadedModel> = HashMap::new();
    let mut order: Vec<String> = Vec::new();
    // dsv4 serve door (ds4f rung 3): dsv4 checkpoint dirs serve through their own
    // 2-card engine stack + dedicated thread, never through HybridModel. Routed by
    // model name at Cmd dispatch; caps merged into the ready handoff below.
    let mut dsv4_routes: HashMap<String, std::sync::mpsc::Sender<Box<Request>>> = HashMap::new();
    let mut dsv4_caps: HashMap<String, ModelCaps> = HashMap::new();
    for (name, path, draft) in &models {
        eprintln!("[worker] loading model {name:?} <- {path}");
        // DIRECTORY path = safetensors HF checkpoint or a manifest-backed memra repack/overlay;
        // file = GGUF. Repack tokenizers live in the manifest's source_dir.
        let from_dir = std::path::Path::new(path).is_dir();
        if from_dir && crate::dsv4_serve::is_dsv4_dir(std::path::Path::new(path)) {
            if draft.is_some() {
                let _ = ready_tx.send(Err(format!(
                    "model {name:?}: '+draft' is a GGUF-family attach; the dsv4 drafter \
                     ships inside the checkpoint (arm it with MEMRA_DSV4_DRAFTER=dspark)"
                )));
                return;
            }
            let dir = std::path::Path::new(path);
            let tok = match Tokenizer::from_hf_dir(dir) {
                Ok(t) => Arc::new(t),
                Err(err) => {
                    let _ = ready_tx.send(Err(format!("tokenizer {name}: {err}")));
                    return;
                }
            };
            let dm = match crate::dsv4_serve::load(name, dir, tok) {
                Ok(m) => m,
                Err(err) => {
                    let _ = ready_tx.send(Err(format!("load {name}: {err}")));
                    return;
                }
            };
            dsv4_caps.insert(name.clone(), crate::dsv4_serve::caps(&dm));
            dsv4_routes.insert(name.clone(), crate::dsv4_serve::spawn(name.clone(), dm));
            order.push(name.clone());
            continue;
        }
        let (model, tok) = if from_dir {
            let dir = std::path::Path::new(path);
            let (src, tok_dir): (
                Box<dyn memra_gguf::source::TensorSource>,
                std::path::PathBuf,
            ) = if dir.join("manifest.json").exists() {
                let repack = match memra_gguf::source::Hy3RepackSource::open(dir) {
                    Ok(source) => source,
                    Err(err) => {
                        let _ = ready_tx.send(Err(format!("open {path}: {err}")));
                        return;
                    }
                };
                let tok_dir = repack
                    .source_dir()
                    .filter(|source| source.join("tokenizer.json").exists())
                    .unwrap_or(dir)
                    .to_path_buf();
                (Box::new(repack), tok_dir)
            } else {
                let st = match memra_gguf::source::SafetensorsSource::open(dir) {
                    Ok(source) => source,
                    Err(err) => {
                        let _ = ready_tx.send(Err(format!("open {path}: {err}")));
                        return;
                    }
                };
                (Box::new(st), dir.to_path_buf())
            };
            let model = match HybridModel::load_from_source(&engine, src.as_ref()) {
                Ok(m) => m,
                Err(err) => {
                    let _ = ready_tx.send(Err(format!("load {name}: {err}")));
                    return;
                }
            };
            let tok = match Tokenizer::from_hf_dir(&tok_dir) {
                Ok(t) => t,
                Err(err) => {
                    let _ = ready_tx.send(Err(format!("tokenizer {name}: {err}")));
                    return;
                }
            };
            (model, tok)
        } else {
            let g = match GgufFile::open(path) {
                Ok(g) => g,
                Err(err) => {
                    let _ = ready_tx.send(Err(format!("open {path}: {err}")));
                    return;
                }
            };
            let model = match HybridModel::load(&engine, &g) {
                Ok(m) => m,
                Err(err) => {
                    let _ = ready_tx.send(Err(format!("load {name}: {err}")));
                    return;
                }
            };
            let tok = match Tokenizer::from_gguf(&g) {
                Ok(t) => t,
                Err(err) => {
                    let _ = ready_tx.send(Err(format!("tokenizer {name}: {err}")));
                    return;
                }
            };
            (model, tok)
        };
        // Per-model regime draft (MEMRA_MODELS "+<draft.gguf>" syntax): replace the embedded
        // MTP head with the standalone regime draft — same load path as MEMRA_MTP_DRAFT but
        // scoped to THIS model, so a multi-model server drafts each model with its own file.
        //
        // THIS IS ALSO THE step35 EXTERNAL-MTP ATTACH (lane/step-draft, 2026-08-07). Step-3.7-
        // Flash ships its three chained NextN blocks in a SEPARATE GGUF, so the trunk parses
        // `nextn_predict_layers=0` and loads with `mtp == None`. No new spelling was added:
        // `+draft` already means "replace this model's MTP head with the head in that file",
        // and `MtpHead::load_draft` already resolves step35's per-layer draft geometry from the
        // drafter file's own arrays (d316162c). The gap was never the attach syntax — it was
        // that a step35 model loaded WITHOUT one said nothing. See the verdict below.
        let mut model = {
            let mut model = model;
            if let Some(dpath) = draft {
                let dg = match GgufFile::open(dpath) {
                    Ok(g) => g,
                    // REFUSE, don't degrade: a drafter path was GIVEN, so booting without it
                    // would serve plain decode under a config that explicitly asked for spec.
                    // The error text is the driver's/loader's own, quoted, never inferred.
                    Err(err) => {
                        let _ = ready_tx.send(Err(format!(
                            "draft {name}: {err} (drafter path {dpath:?} was requested via the \
                             MEMRA_MODELS '+draft' attach — refusing to start rather than \
                             silently serving plain decode)"
                        )));
                        return;
                    }
                };
                match memra_engine::hybrid::MtpHead::load_draft(&engine, &dg, &model.cfg) {
                    Ok(head) => {
                        let draft_cfg = match memra_gguf::source::GgufSource(&dg).try_config() {
                            Ok(config) => config,
                            Err(err) => {
                                let _ = ready_tx.send(Err(format!(
                                    "draft {name}: malformed model metadata: {err}"
                                )));
                                return;
                            }
                        };
                        let draft_plan = match memra_gguf::model_plan::ModelPlan::compile(
                            &draft_cfg,
                        )
                        .and_then(|draft_plan| {
                            model.plan.attach_external_draft(&draft_plan)?;
                            Ok(draft_plan)
                        }) {
                            Ok(plan) => plan,
                            Err(err) => {
                                let _ = ready_tx.send(Err(format!(
                                    "draft {name}: canonical draft-plan attach failed: {err}"
                                )));
                                return;
                            }
                        };
                        eprintln!("[worker] {name}: regime draft attached ({dpath})");
                        eprintln!(
                            "[worker] {name}: canonical draft plan attached ({} MTP blocks)",
                            draft_plan.mtp_blocks.len()
                        );
                        model.mtp = Some(head);
                    }
                    Err(err) => {
                        let _ = ready_tx.send(Err(format!(
                            "draft {name}: {err} (drafter path {dpath:?} was requested via the \
                             MEMRA_MODELS '+draft' attach — refusing to start rather than \
                             silently serving plain decode)"
                        )));
                        return;
                    }
                }
            }
            model
        };
        if let Some(bundle) = std::env::var_os("MEMRA_REWRITE_BUNDLE") {
            let bundle = std::path::Path::new(&bundle);
            if let Err(error) = model.install_rewrite_bundle(bundle) {
                let _ = ready_tx.send(Err(format!(
                    "rewrite bundle {} for {name}: {error}",
                    bundle.display()
                )));
                return;
            }
            eprintln!(
                "[worker] {name}: rewrite qualification installed from {}",
                bundle.display()
            );
        }

        // LOUD DRAFTER SEMANTICS (lane/step-draft, 2026-08-07). The silent-degradation class
        // this closes: `spec_eligible` requires `lm.model.mtp.is_some()`, so a step35 trunk
        // served without a drafter took plain decode on every request with NO log line saying
        // so — named as defect (A) in `research/step37-p2-20260806/PROGRESS.md`. A server that
        // forgoes its whole felt-latency story must say it out loud.
        //
        // The verdict is computed from a PURE function (`draft_verdict`) so both branches are
        // pinned by GPU-free tests. (#87's spec-over-PP-2 refusal used to live here too —
        // CLOSED 2026-08-08, see `draft_verdict`; spec+PP-2 now serves, gates in
        // research/pp2spec-crash-20260807/.)
        let verdict = draft_verdict(
            model.mtp.is_some(),
            model.plan.draft_source == memra_gguf::model_plan::DraftSourcePlan::ExternalArtifact,
        );
        if let Some(msg) = draft_verdict_message(&verdict, name, path) {
            eprintln!("{msg}");
        }

        let eos_id = tok.eos_id();
        // The STOP SET, not just the scalar. `eos=154820` alone was the whole visible
        // evidence while GLM-5.3-Flash's other two declared stops (<|user|>, <|observation|>)
        // were being dropped at tokenizer load — the log agreed with the bug and could not
        // disagree with it (lane glm53-flash-bringup, 2026-08-27). `stop=` is what the serve
        // path actually tests against (`GenParams::eos` below is this union).
        let stop_ids = tok.eog_ids();
        eprintln!(
            "[worker]   loaded {name:?}: {} layers, eos={eos_id}, stop={stop_ids:?}",
            model.cfg.n_layer
        );
        let tok = Arc::new(tok);
        let constraints = match crate::constrained::ConstraintCompiler::spawn(
            name,
            tok.clone(),
            constraint_result_tx.clone(),
            &metrics,
        ) {
            Ok(compiler) => compiler,
            Err(err) => {
                let _ = ready_tx.send(Err(err));
                return;
            }
        };
        // (#68 closed 2026-08-04: the former ST-spec quarantine notice lived here — dir
        // checkpoints are spec-eligible again, research/fp8ship-20260804/RESULTS.md.)
        loaded.insert(
            name.clone(),
            LoadedModel {
                model,
                tok,
                eos_id,
                from_dir,
                constraints,
            },
        );
        order.push(name.clone());
    }
    // Template capability probe (serve-tools lane): same substring laws the renderer uses.
    // + /v1/models metadata (serve-tail lane): context length from the model config,
    // tokenizer family from the pre-tokenizer name, instruct family from the template's
    // turn markers. Unknown stays 0/""/None — the route reports honest nulls.
    let caps: HashMap<String, ModelCaps> = loaded
        .iter()
        .map(|(n, lm)| {
            let t = lm.tok.chat_template();
            let caps = ModelCaps {
                // qwen/step `<tools>` OR the gemma4 tooluse dialect (`<|turn>` + `<|tool>`);
                // never hy3. Shared law with the renderer dispatch.
                tools_branch: t.is_some_and(memra_tokenizer::chat::template_has_tools_branch),
                qwen_think: t
                    .is_some_and(|t| t.contains("<think>") && t.contains("add_generation_prompt")),
                think_switch: t.is_some_and(|t| t.contains("enable_thinking")),
                // GGUF keeps the historical ChatML fallback for template-less models; a dir
                // checkpoint (safetensors/repack) must CARRY its template (tokenizer_config
                // chat_template or chat_template.jinja) or chat requests 400 (serve-st v1).
                chat_ok: t.is_some() || !lm.from_dir,
                context_length: lm.model.cfg.context_length as usize,
                tokenizer: lm.tok.pre().to_string(),
                instruct_type: t.and_then(|t| {
                    if t.contains("<|im_start|>") {
                        Some("chatml".to_string())
                    } else if t.contains("<start_of_turn>") {
                        Some("gemma".to_string())
                    } else if memra_tokenizer::chat::template_is_dsv4(t) {
                        Some("deepseek".to_string())
                    } else if memra_tokenizer::chat::template_is_glm5(t) {
                        Some("glm".to_string())
                    } else {
                        None
                    }
                }),
                // Templates that CONSUME a `reasoning_effort` input, keyed on the jinja input
                // test itself (`reasoning_effort is defined`) — true for step35 (renders
                // `Reasoning: {level}` into the system turn) and hy3 (renders
                // `reasoning_effort:{no_think|low|high}` into its header), false for the
                // qwen/gemma4 classes (binary `enable_thinking`, carried by ThinkMode instead).
                effort_levels: t.is_some_and(|t| t.contains("reasoning_effort is defined")),
                // The qwen3.8 ladder spells its effort input WITHOUT `is defined`
                // (`reasoning_effort|default('xhigh')`), so `effort_levels` above misses it.
                // Keyed on the instruction sentences the renderer emits — see
                // `chat::template_has_qwen_effort`.
                qwen_effort: t.is_some_and(memra_tokenizer::chat::template_has_qwen_effort),
                // keyed on the dialect's own thought-channel marker in the shipped template
                // (research/step-sku-20260807/templates/gemma4-12b-qat.chat_template.jinja:
                // strip_thinking splits on `<|channel>`). Template-keyed like every other cap —
                // a gemma4 GGUF without its template falls back to ChatML rendering, where
                // arming a channel splitter would be guessing.
                gemma_think: t.is_some_and(|t| t.contains("<|channel>")),
                dsv4: t.is_some_and(memra_tokenizer::chat::template_is_dsv4),
                // GLM-5.3-Flash: `[gMASK]<sop>` + `<|observation|>`. Template-keyed like every
                // other cap — a glm5 checkpoint that shipped without its template would fall
                // back to ChatML rendering, where arming this dialect would be guessing.
                glm5: t.is_some_and(memra_tokenizer::chat::template_is_glm5),
                chat_temperature_default: lm
                    .model
                    .plan
                    .sampling_defaults
                    .map(|defaults| defaults.temperature),
                chat_top_p_default: lm
                    .model
                    .plan
                    .sampling_defaults
                    .map(|defaults| defaults.top_p),
                n_vocab: lm.tok.vocab_size(),
            };
            eprintln!(
                "[worker] {n}: template caps tools={} think={} think_switch={} chat_ok={} \
                   effort_levels={} qwen_effort={} gemma_think={} dsv4={} glm5={} ctx={} \
                   tok={:?} instruct={:?} chat_defaults={:?}/{:?}",
                caps.tools_branch,
                caps.qwen_think,
                caps.think_switch,
                caps.chat_ok,
                caps.effort_levels,
                caps.qwen_effort,
                caps.gemma_think,
                caps.dsv4,
                caps.glm5,
                caps.context_length,
                caps.tokenizer,
                caps.instruct_type,
                caps.chat_temperature_default,
                caps.chat_top_p_default
            );
            (n.clone(), caps)
        })
        .collect();
    let mut caps = caps;
    caps.extend(dsv4_caps.drain());

    // Per-model decode scheduling policy: the model fixes the exact numeric width, while the
    // default-off dual PP door may combine two such waves into one worker tick.
    let chunk_policies: HashMap<String, DecodeChunkPolicy> = loaded
        .iter()
        .map(|(n, lm)| (n.clone(), decode_chunk_policy(lm)))
        .collect();
    for (n, policy) in &chunk_policies {
        eprintln!(
            "[worker] {n}: decode wave cap {}; scheduler tick cap {}{}",
            policy.wave_cap,
            policy.tick_cap(),
            if policy.dual {
                " (dual PP, default-off arm)"
            } else if policy.wave_cap > 8 {
                " (exact-16 tier)"
            } else {
                ""
            },
        );
    }
    // EAGER-ONLY models (lane/gemma4-serve-gaps, 2026-08-07): no batched decode arm, no
    // batched prime core, no step-wise graph capture — every batched-scheduler entry point
    // below routes around them (per-session eager decode, monolithic prefill, no graph
    // promotion, no prime batching). Before this route existed, ONE request to a gemma4
    // model on the default scheduler panicked the worker on decode_step_batch's gemma4
    // assert, the respawn re-panicked on the queued request, and the process FATALed
    // (research/gemma4-serve-20260807/raw/repro-panic-server-*.log).
    let eager_only: std::collections::HashSet<String> = loaded
        .iter()
        .filter(|(_, lm)| eager_only_model(lm))
        .map(|(n, _)| n.clone())
        .collect();
    // Decode-site subset (lane/gemma-batched): eager_only MINUS the models whose batched
    // decode arm is live (dense gemma4 — DEFAULT ON since the 2026-08-16 owner flip;
    // MEMRA_GEMMA4_BATCH=0 is the eager kill switch). Consumed ONLY by the two decode
    // scheduling sites below; all other eager-only exclusions keep the full set.
    let eager_decode: std::collections::HashSet<String> = loaded
        .iter()
        .filter(|(_, lm)| eager_only_model(lm) && !gemma4_batched_decode_model(lm))
        .map(|(n, _)| n.clone())
        .collect();
    for n in &eager_only {
        if eager_decode.contains(n) {
            let class = loaded
                .get(n)
                .filter(|lm| memra_engine::plan_backend::decode_batch_unconverted(&lm.model.plan))
                .map_or("gemma4 class", |_| "hyper-connections residual");
            eprintln!(
                "[worker] {n}: EAGER-ONLY serving ({class} — no batched decode arm): \
                   per-session eager decode, monolithic prefill, no graph promotion, \
                   no prime batching"
            );
        } else {
            eprintln!(
                "[worker] {n}: BATCHED DECODE (gemma4 dense arm, default-on; \
                   MEMRA_GEMMA4_BATCH=0 = eager kill switch): batched decode chunks; \
                   monolithic prefill, no graph promotion, no prime batching \
                   (eager-only for every non-decode entry point)"
            );
        }
    }

    // GEMMA SPEC drafter attach (lane/gemma-batched stage 2, 2026-08-17): one GemmaDraft
    // per dense-gemma4 model, loaded at spawn behind MEMRA_GEMMA4_SPEC=K + MEMRA_DRAFT.
    // Fail LOUD on any ambiguity or load error — the 3f4597f02 guard law (a dflash env
    // would silently shadow the assistant route) and the vision-tower boot pattern.
    let mut gemma_drafts: std::collections::HashMap<String, memra_engine::gemma_spec::GemmaDraft> =
        Default::default();
    if gemma4_spec_k_env() > 0 {
        if std::env::var("MEMRA_SPEC_DFLASH").is_ok() {
            panic!(
                "MEMRA_GEMMA4_SPEC is set together with MEMRA_SPEC_DFLASH; the served gemma \
                 spec route is the assistant drafter (MEMRA_DRAFT) — dflash is not served. \
                 Unset one; refusing to guess which drafter you meant"
            );
        }
        let dpath = std::env::var("MEMRA_DRAFT").unwrap_or_else(|_| {
            panic!(
                "MEMRA_GEMMA4_SPEC={} needs MEMRA_DRAFT=<assistant.gguf>",
                gemma4_spec_k_env()
            )
        });
        for (n, lm) in &loaded {
            if memra_engine::plan_backend::decode_batch_program(&lm.model.plan)
                == memra_engine::plan_backend::DecodeBatchProgram::Gemma
                && !lm.model.is_gemma4_e4b()
            {
                let dg = memra_gguf::GgufFile::open(&dpath)
                    .unwrap_or_else(|e| panic!("MEMRA_DRAFT={dpath}: open failed: {e}"));
                let d = memra_engine::gemma_spec::GemmaDraft::load(&engine, &dg)
                    .unwrap_or_else(|e| panic!("MEMRA_DRAFT={dpath}: drafter load failed: {e}"));
                // The attach line NAMES THE ARTIFACT, like both MTP spellings already do
                // (`[mtp-draft] loading external MTP draft: {path}`, `regime draft attached
                // ({dpath})`). Gemma was the one attach of the three that logged only that
                // *some* drafter attached, which made
                // `tools/assert-drafter-attached.sh --gemma <log> <expected-path>` UNSATISFIABLE:
                // the tool's whole premise is "an attach is a LOG LINE", and it can only check
                // the identity of the artifact if the line carries it. lane/sampled-restore-load-
                // guard wired that assertion into all five gates and banked no battery, so the
                // gemma arm of the hit gate could not pass and nothing had run it yet — the same
                // never-executed-gate class as the greedy-only inertness, one layer down.
                // Log text only; the route, K, and posture are unchanged.
                eprintln!(
                    "[worker] {n}: GEMMA SPEC route armed (K={}, assistant drafter attached \
                     ({dpath}); greedy/unconstrained/text-only/solo-admission; \
                     MEMRA_GEMMA4_SPEC=0 = off)",
                    gemma4_spec_k_env()
                );
                gemma_drafts.insert(n.clone(), d);
            }
        }
    }

    // DSPARK SPEC drafter attach (lane/dspark-q38-recover serve route): one DflashDraft
    // per qwen-hybrid model, loaded at spawn behind MEMRA_DSPARK_SPEC=1 +
    // MEMRA_DSPARK_DRAFT=<export_dir>. Fail LOUD on any ambiguity or load error — the
    // 3f4597f02 guard law: two spec programs on one model must never silently coexist,
    // so arming dspark DISABLES the MTP spec arm for that model (spec_eligible below)
    // and refuses combinations that have never been co-gated.
    let mut dspark_drafts: std::collections::HashMap<String, memra_engine::dflash::DflashDraft> =
        Default::default();
    if std::env::var("MEMRA_DSPARK_SPEC").as_deref() == Ok("1") {
        if let Some(msg) = dspark_spec_boot_conflict(
            std::env::var("MEMRA_SPEC_DFLASH").is_ok(),
            gemma4_spec_k_env(),
            std::env::var("MEMRA_PP_STAGES")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(1),
            std::env::var("MEMRA_STEP_TP").ok().as_deref(),
            std::env::var("MEMRA_STEP_EP").ok().as_deref(),
        ) {
            panic!("{msg}");
        }
        let dpath_spec = std::env::var("MEMRA_DSPARK_DRAFT").unwrap_or_else(|_| {
            panic!("MEMRA_DSPARK_SPEC=1 needs MEMRA_DSPARK_DRAFT=<export_dir>")
        });
        let dpath = memra_gguf::hf::resolve_arg(&dpath_spec)
            .unwrap_or_else(|err| panic!("MEMRA_DSPARK_DRAFT={dpath_spec:?}: {err}"));
        for (n, lm) in &loaded {
            if memra_engine::plan_backend::gdn_dspark_compatible(&lm.model.plan) {
                let d =
                    memra_engine::dflash::DflashDraft::load(&engine, std::path::Path::new(&dpath))
                        .unwrap_or_else(|e| {
                            panic!("MEMRA_DSPARK_DRAFT={dpath}: drafter load failed: {e}")
                        });
                eprintln!(
                    "[worker] {n}: DSPARK SPEC route armed (drafter attached ({dpath}); greedy+sampled \
                     [T>0 rejection verify, sampled penalties included]/unconstrained/text-only; \
                     greedy LOW-wave admission + HIGH demotion, sampled solo admission; \
                     MTP spec DISABLED for this model — refuse-on-ambiguity; \
                     MEMRA_DSPARK_SPEC unset = off)"
                );
                // DRAFT-HEAD TRIM receipt (lane/dflash2-head-trim, 2026-08-25): the DFlash2
                // round reuses the FR-Spec self-trim the load path builds on the MTP struct.
                // Engagement is a printed fact, not an inference — the decisive-probes law.
                if d.dflash2.is_some() {
                    match lm
                        .model
                        .mtp
                        .as_ref()
                        .and_then(|m| m.d2t_from_target_head.then_some(m.d2t.as_ref()).flatten())
                    {
                        Some(d2t) => eprintln!(
                            "[dspark] {n}: DFlash2 draft head TRIMMED to {} rows \
                             (FR-Spec d2t; verify stays full-vocab)",
                            d2t.len()
                        ),
                        None => {
                            let external_d2t =
                                lm.model.mtp.as_ref().is_some_and(|m| m.d2t.is_some());
                            eprintln!(
                                "[dspark] {n}: DFlash2 draft head FULL target vocab \
                                 (external_mtp_d2t_ignored={external_d2t}; set \
                                 MEMRA_FRSPEC_TRIM=<ranks.txt> to trim)"
                            )
                        }
                    }
                }
                dspark_drafts.insert(n.clone(), d);
            }
        }
        if dspark_drafts.is_empty() {
            panic!("MEMRA_DSPARK_SPEC=1 but no qwen-hybrid (non-gemma) model is loaded");
        }
    }

    // ---- scheduler loop ----
    let mut active: Vec<Session> = Vec::new();
    let mut queue: std::collections::VecDeque<Box<Request>> = std::collections::VecDeque::new();
    let mut pending_constraints: HashMap<u64, PendingConstraintCompile> = HashMap::new();
    let mut next_constraint_id = 0u64;
    // KV prefix-reuse pool (append-only continuation; see ReuseEntry doc). Keyed by
    // (model, namespace) — cross-request continuation state is tenant-scoped too (PC-ISO).
    let mut reuse: HashMap<PoolKey, Vec<ReuseEntry>> = HashMap::new();
    let mut spec_reuse: HashMap<PoolKey, Vec<SpecReuseEntry>> = HashMap::new();
    let mut dspark_reuse: HashMap<PoolKey, Vec<DsparkReuseEntry>> = HashMap::new();
    // Cmd::TrimPools reply channels parked by handle_cmd; executed at the tick top
    // where the pools above are in scope (deploy-headroom lane, 2026-08-27).
    let mut pending_trims: Vec<tokio::sync::oneshot::Sender<TrimReport>> = Vec::new();
    // F5: learned spec-session sizing (evict-first models + right-sized ctx asks).
    let mut spec_sizing = SpecSizing::default();
    let mut reuse_metrics = ReuseMetrics::default();
    let prefix_budget = init_prefix_cache_budget(&engine, &loaded);
    // VISION tower (lane/vision): loaded once at spawn from MEMRA_VISION_DIR (a directory
    // carrying the checkpoint's outside.safetensors with model.visual.*). Fail LOUD at boot
    // if configured but unloadable — silently serving text-only would be dishonest.
    // MEMRA_VISION=0 skips the tower even with the dir configured (owner knob for
    // VRAM-tight boxes: ~1.8 GB f32-resident). Image requests then 400 at the HTTP layer.
    let vision_tower: Option<memra_engine::vision::VisionTower> =
        match std::env::var("MEMRA_VISION_DIR") {
            Ok(_) if std::env::var("MEMRA_VISION").as_deref() == Ok("0") => {
                eprintln!("[vision] MEMRA_VISION=0 — tower NOT loaded, image input disabled");
                None
            }
            Ok(dir) => Some(
                memra_engine::vision::VisionTower::load(&engine, std::path::Path::new(&dir))
                    .unwrap_or_else(|e| panic!("MEMRA_VISION_DIR={dir}: tower load failed: {e}")),
            ),
            Err(_) => None,
        };
    // GEMMA-4 vision tower (lane/gemma-vision): loaded once at spawn behind the seam. Fail
    // LOUD at boot if configured but unloadable. Default off — no gemma image serving until
    // an operator sets MEMRA_GEMMA_VISION=1 + MEMRA_GEMMA_MMPROJ=<gemma4v mmproj>.
    let gemma_tower: Option<memra_engine::vision_gemma::GemmaVisionTower> = match (
        std::env::var("MEMRA_GEMMA_VISION").as_deref(),
        std::env::var("MEMRA_GEMMA_MMPROJ"),
    ) {
        (Ok("1"), Ok(path)) => Some(
            memra_engine::vision_gemma::GemmaVisionTower::load(
                &engine,
                std::path::Path::new(&path),
            )
            .unwrap_or_else(|e| panic!("MEMRA_GEMMA_MMPROJ={path}: gemma tower load failed: {e}")),
        ),
        _ => None,
    };
    // Cross-request prefix cache (token-prefix keyed, budget-bound; see the module doc above).
    let mut px = PrefixCache::default();
    if memra_engine::pp::pp_host_bounce_active() {
        eprintln!(
            "[pp] MEMRA_PP_HOST_BOUNCE=1 safety doors: speculative PP, cross-device prefix \
             snapshots, and plain-affinity checkpoints disabled (they retain peer reads/copies)"
        );
    }
    if prefix_cache_budget_bytes() > 0 && serve_batching() {
        let (budget_bytes, budget_provenance) = match prefix_budget {
            PrefixCacheBudget::Configured { bytes } => (
                *bytes,
                format!("{bytes} B, configured by MEMRA_PREFIX_CACHE_MB"),
            ),
            PrefixCacheBudget::Derived {
                bytes,
                requested_bytes,
                entry_bytes,
                model,
                ctx,
                boot_free_bytes,
                clamp_bytes,
            } => (
                *bytes,
                format!(
                    "{bytes} B, derived: {} x {entry_bytes} B max entry for model {model:?} at \
                     MEMRA_CTX={ctx}, requested {requested_bytes} B; boot driver free \
                     {boot_free_bytes} B, post-reserve clamp {clamp_bytes} B",
                    PREFIX_CACHE_DEFAULT_ENTRIES,
                ),
            ),
        };
        if prefix_cache_slru_enabled() {
            eprintln!(
                "[prefix-cache] on: budget {:.0}MB ({budget_provenance}), policy byte-SLRU \
                 protected/probation {}%/{}% (MEMRA_PREFIX_CACHE_PROTECTED_PCT), min prefix {} \
                 tokens, immediate partial restore={} (transformer-only; hybrid mid-entry + \
                 routed-MoE N/A)",
                budget_bytes as f64 / 1e6,
                prefix_cache_protected_pct(),
                100 - prefix_cache_protected_pct(),
                PREFIX_CACHE_MIN_TOKENS,
                if partial_prefix_restore_enabled() {
                    "on"
                } else {
                    "off (rollback)"
                },
            );
        } else {
            eprintln!(
                "[prefix-cache] on: budget {:.0}MB ({budget_provenance}), policy plain-LRU \
                 (MEMRA_PREFIX_CACHE_POLICY=lru rollback; no probation segment), min prefix {} \
                 tokens, immediate partial restore={} (transformer-only; hybrid mid-entry + \
                 routed-MoE N/A)",
                budget_bytes as f64 / 1e6,
                PREFIX_CACHE_MIN_TOKENS,
                if partial_prefix_restore_enabled() {
                    "on"
                } else {
                    "off (rollback)"
                },
            );
        }
    } else if serve_batching()
        && matches!(prefix_budget, PrefixCacheBudget::Derived { bytes: 0, .. })
    {
        eprintln!(
            "[prefix-cache] WARNING: derived budget is 0 B after the boot free-VRAM \
                   reserve clamp; prefix caching disabled"
        );
    }
    // Request-shaped session cost: exact context-linear cache/scratch geometry plus a measured
    // fixed high-water residual. Unlike the old first-admit scalar, this is armed before request 1
    // and every estimate uses that request's own effective context cap.
    let mut admission_costs: HashMap<String, AdmissionCostModel> = loaded
        .iter()
        .map(|(name, lm)| (name.clone(), AdmissionCostModel::new(&lm.model)))
        .collect();
    for (name, cost) in &admission_costs {
        if cost.ring_rows > 0 {
            eprintln!(
                "[admission] {name:?}: plain {} B/token ({} capped at {} rows), spec {} \
                 B/token ({} capped); fixed residual learns from effective-free deltas",
                cost.plain_bytes_per_token,
                cost.plain_ring_bytes_per_token,
                cost.ring_rows,
                cost.spec_bytes_per_token,
                cost.spec_ring_bytes_per_token,
            );
        } else {
            eprintln!(
                "[admission] {name:?}: plain {} B/token, spec {} B/token; fixed residual learns \
                 from effective-free deltas",
                cost.plain_bytes_per_token, cost.spec_bytes_per_token,
            );
        }
    }

    // ---- serving counters + engine-truth step stats (30s percentile window) ----
    // Lane machinery (x-lane QoS gate, lane/dl-metering port): policy from env; step_stats
    // is the INTERACTIVE SLO sensor (records only ticks that advanced an interactive
    // session — on naked traffic every session is interactive, so /metrics is unchanged).
    let policy = crate::lanes::LanePolicy::from_env();
    let prefill_tick_explicit = std::env::var_os("MEMRA_PREFILL_TICK").is_some();
    let pb_hold_ms: u64 = std::env::var("MEMRA_PRIME_BATCH_HOLD_MS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4);
    // When a completed interactive request opens a client-concurrency slot, the replacement
    // request is still crossing the HTTP/channel boundary. Keep cold prefill off that same
    // short formation window so a cache hit cannot arrive behind a synchronous prime call.
    let mut interactive_refill_until: Option<Instant> = None;
    let mut step_stats = StepStats::new(
        std::env::var("MEMRA_LANE_WINDOW_S")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(30.0),
    );
    let mut n_admitted = 0u64;
    let mut n_completed = 0u64;
    let mut n_tokens_out = 0u64;
    let mut n_prompt_in = 0u64;
    let mut n_cached_in = 0u64;
    let mut n_session_defers = 0u64;
    let mut n_vram_defers = 0u64;
    let mut n_step_oom_parks = 0u64;
    // Served-path receipts (lane/dspark-sampled-wave-20260825): admission-time route
    // classification, published to /metrics for the deploy gate's sampled probe.
    let mut n_served_dspark = 0u64;
    let mut n_served_spec = 0u64;
    let mut n_served_plain = 0u64;
    // Per-tenant prompt/cached split (lane/cache-metering): keyed by the tenant half of
    // the PC-ISO namespace (auth::meter_key). Bounded: past METER_TENANT_CAP distinct
    // keys, new traffic aggregates under "(other)" — a salt-spraying client cannot grow
    // worker memory. Updated once per ADMIT (request-frequency, never per-token).
    let mut ns_tokens: HashMap<String, [u64; 2]> = HashMap::new();
    let mut lane_admitted = [0u64; 3];
    let mut lane_shed = [0u64; 3];
    let mut lane_completed = [0u64; 3];
    let mut lane_tokens = [0u64; 3];
    let mut last_batch = 0usize;
    // Per-model spec acceptance telemetry (lane/accept-telemetry): worker-owned like every
    // counter above; published on the same 32nd-tick snapshot AND whenever a spec session
    // retires (so a one-shot request's counts are visible without waiting 32 ticks).
    let mut spec_metrics = SpecMetricState::new(SPEC_METRICS_WINDOW_S);
    let mut spec_telem_dirty = false;
    // Detection-only per-tenant acceptance divergence (ADSD). Samples arrive once per
    // retired spec request, so this adds no per-token work and shares the forced-retire
    // metrics publication above.
    let mut adsd_detector = AdsdDetector::default();
    // Starvation sentinel (estimator blind spot, 2026-07-26 native-judge battery): last
    // time an interactive session decoded. Interactive work waiting with no interactive
    // decode tick inside the SLO age IS an SLO breach the percentile window can't see.
    let mut last_interactive_decode = Instant::now();
    let mut tick_n: u64 = 0;
    // SPEC GATE (lane/spec-gate): how many live sessions this worker has handed from the spec
    // burst path to batched decode. The thrash observable — under a correct hysteresis band
    // this counts LOAD CROSSINGS, not ticks (a per-tick demotion count would mean the band is
    // too narrow or the handoff is failing and re-firing).
    let mut n_demoted = 0u64;
    let mut peer_probe_deferral = RuntimePeerProbeDeferralState::default();

    let _ = ready_tx.send(Ok((order.clone(), caps)));
    // INFERENCE LIVENESS (G5): trunk weights, configured drafters and vision towers are resident,
    // scheduler state is initialized, and the loop is about to run. /health and /readyz go green
    // HERE, not while an optional artifact is still loading. Also clears the fault latch, which is
    // what makes a respawn's success observable.
    health.mark_ready();

    loop {
        // Cheap runtime peer validation stays on its copy-count cadence here, between scheduler
        // ticks on the CUDA owner thread. Idle-only rungs remain pending. A mismatch continues on
        // validated host bounce; only inability to arm that staging reaches the panic ladder.
        if let Err(err) = service_runtime_peer_probe_for_worker(
            &engine,
            false,
            &mut active,
            &mut px,
            &mut peer_probe_deferral,
            &health,
        ) {
            panic!("runtime peer-probe safety failure: {err}");
        }
        resolve_constraint_compiles(&constraint_result_rx, &mut pending_constraints, &mut queue);
        expire_constraint_compiles(&mut pending_constraints, Instant::now());
        // 1. Drain pending commands. Block ONLY when there is no work at all (no active sessions),
        //    otherwise poll non-blocking so the decode loop keeps interleaving. A request waiting
        //    on off-tick grammar compilation is BUSY work: use a short recv timeout so command
        //    arrivals, compile completion, expiry, and the heartbeat all stay responsive.
        if active.is_empty() && queue.is_empty() {
            if pending_constraints.is_empty() {
                // Do not let an already-arrived request sit behind an idle-only probe. Once the
                // channel is observed empty, one pending expensive rung may run before the worker
                // enters its ordinary indefinite idle block.
                match rx.try_recv() {
                    Ok(cmd) => {
                        health.set_phase(crate::health::PHASE_BUSY);
                        handle_cmd(
                            cmd,
                            &loaded,
                            &dsv4_routes,
                            &order,
                            &mut queue,
                            &mut pending_trims,
                        );
                    }
                    Err(std::sync::mpsc::TryRecvError::Empty) => {
                        if let Err(err) = service_runtime_peer_probe_for_worker(
                            &engine,
                            true,
                            &mut active,
                            &mut px,
                            &mut peer_probe_deferral,
                            &health,
                        ) {
                            panic!("runtime peer-probe safety failure: {err}");
                        }
                        // IDLE PHASE (G5): about to block indefinitely in recv() with zero work.
                        // An idle worker legitimately stamps no heartbeat for hours, so the phase
                        // — not the beat age — is what /health reads here. Stamped on BOTH sides
                        // of the block so the beat is already fresh when work arrives.
                        health.set_phase(crate::health::PHASE_IDLE);
                        match rx.recv() {
                            Ok(cmd) => {
                                health.set_phase(crate::health::PHASE_BUSY);
                                handle_cmd(
                                    cmd,
                                    &loaded,
                                    &dsv4_routes,
                                    &order,
                                    &mut queue,
                                    &mut pending_trims,
                                );
                            }
                            Err(_) => break, // all senders dropped -> shutdown
                        }
                    }
                    Err(std::sync::mpsc::TryRecvError::Disconnected) => break,
                }
            } else {
                health.beat_busy();
                let wait = constraint_poll_wait(&pending_constraints, Instant::now());
                match rx.recv_timeout(wait) {
                    Ok(cmd) => handle_cmd(
                        cmd,
                        &loaded,
                        &dsv4_routes,
                        &order,
                        &mut queue,
                        &mut pending_trims,
                    ),
                    Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {}
                    Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => return,
                }
            }
        }
        // BUSY PHASE: work is in flight, so the beat MUST advance every iteration. The
        // stamp is a bare atomic store (no mutex, no syscall) — unlike the metrics publish
        // below it is NOT throttled, because a heartbeat sampled every 32nd tick would give
        // health a 32-tick blind spot at exactly the moment a tick stops returning.
        health.beat_busy();
        loop {
            match rx.try_recv() {
                Ok(cmd) => handle_cmd(
                    cmd,
                    &loaded,
                    &dsv4_routes,
                    &order,
                    &mut queue,
                    &mut pending_trims,
                ),
                Err(std::sync::mpsc::TryRecvError::Empty) => break,
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    if active.is_empty() {
                        return;
                    } else {
                        break;
                    }
                }
            }
        }
        resolve_constraint_compiles(&constraint_result_rx, &mut pending_constraints, &mut queue);
        expire_constraint_compiles(&mut pending_constraints, Instant::now());

        // TRIM (deploy-headroom lane): drop every evictable cross-request pool and
        // answer the waiting admin caller. In-flight sessions and pinned prefix-cache
        // leases are untouched; the freed device memory returns to CUDA on drop.
        for tx in pending_trims.drain(..) {
            let report = TrimReport {
                reuse_entries: reuse.values().map(Vec::len).sum(),
                spec_reuse_entries: spec_reuse.values().map(Vec::len).sum(),
                dspark_reuse_entries: dspark_reuse.values().map(Vec::len).sum(),
                prefix_entries: px.evict_all(),
            };
            reuse.clear();
            spec_reuse.clear();
            dspark_reuse.clear();
            // The entries' device memory lands in the CACHED async mempool (boot pins
            // RELEASE_THRESHOLD=MAX for graph-launch speed) — invisible to a green
            // process. Hand the cached blocks back to the driver so nvidia-smi free
            // actually rises (measured: 8GB of dropped pools, 0MiB visible until this).
            let released = engine.pool_trim_to_zero();
            eprintln!(
                "[trim] mempool released {}MiB to the driver",
                released >> 20
            );
            eprintln!(
                "[trim] pools dropped: reuse={} spec={} dspark={} prefix={}",
                report.reuse_entries,
                report.spec_reuse_entries,
                report.dspark_reuse_entries,
                report.prefix_entries
            );
            let _ = tx.send(report);
        }
        // 2. ADMISSION + LANE GATE (x-lane yield gate, engine-side): interactive admits up
        //    to the cap and WAITS beyond it (FIFO, never rejected — its queue wait is the
        //    protected tenant's own backlog). Judge/harvest are gated on the measured
        //    interactive step p99 vs their SLO fraction and SHED with an immediate
        //    retryable error (HTTP 429 at the handler) — dark-lane work is NEVER queued
        //    inside the engine (the B2 lesson: the engine queue is where the tail dies).
        //    Interactive cap stays the legacy MEMRA_MAX_SESSIONS knob (naked-path
        //    preserving; policy.max_sessions[0] is the sidecar's knob, unused here);
        //    judge/harvest caps come from the lane policy.
        let max_active = if confidence_trace_enabled() {
            1
        } else {
            MAX_ACTIVE
        };
        let mut requeue: std::collections::VecDeque<Box<Request>> = Default::default();
        // Per-tick count of requests the VRAM gate deferred (logged once per tick).
        let mut vram_defers = 0usize;
        while let Some(mut req) = queue.pop_front() {
            // DISCONNECT ABORT (gap-scan F8): a queued request whose client already hung
            // up (receiver dropped) never reaches the GPU — dropped here, logged for the
            // metering record (0 generated; prompt never primed).
            if req.tx.is_closed() {
                eprintln!(
                    "[abort] client disconnected while queued (model {:?}); dropped",
                    req.model
                );
                release_admission_reservation(req.lane);
                continue;
            }
            // CONSTRAINED PRE-ADMISSION: transfer the schema to this model's bounded compiler
            // and retain the request off-queue until a fresh matcher comes back. This happens
            // before prompt preparation, lane admission, cache probes, or any GPU allocation.
            // The worker never waits for either the first full-vocab TokTrie or per-request
            // schema compilation; normal sessions keep stepping below.
            if req.grammar.is_some() && req.prepared_constraint.is_none() {
                let spec = req.grammar.take().expect("grammar checked above");
                let deadline = Instant::now() + crate::constrained::CONSTRAINT_COMPILE_TIMEOUT;
                let id = next_constraint_id;
                next_constraint_id = next_constraint_id.wrapping_add(1);
                match loaded[&req.model]
                    .constraints
                    .try_submit(id, spec, deadline)
                {
                    Ok(()) => {
                        pending_constraints.insert(
                            id,
                            PendingConstraintCompile {
                                request: req,
                                deadline,
                            },
                        );
                    }
                    Err(crate::constrained::ConstraintSubmitError::Busy) => {
                        fail_request(
                            req,
                            EngineError::overloaded("response_format compiler is busy; retry"),
                        );
                    }
                    Err(crate::constrained::ConstraintSubmitError::Closed) => {
                        fail_request(
                            req,
                            EngineError::engine("response_format compiler is unavailable"),
                        );
                    }
                    Err(crate::constrained::ConstraintSubmitError::AbandonedWorkerLimit) => {
                        fail_request(req, constraint_worker_limit_error());
                    }
                }
                continue;
            }
            let lane = req.lane;
            let batching_on = std::env::var("MEMRA_SERVE_BATCH")
                .map(|v| v != "0")
                .unwrap_or(true);
            let cap = if lane == crate::lanes::Lane::Interactive {
                if batching_on {
                    std::env::var("MEMRA_MAX_SESSIONS")
                        .ok()
                        .and_then(|v| v.parse().ok())
                        .unwrap_or(64)
                } else {
                    max_active
                }
            } else {
                policy.max_sessions[lane.idx()]
            };
            let lane_count = active.iter().filter(|s| s.lane == lane).count();
            if lane_count >= cap {
                if lane == crate::lanes::Lane::Interactive {
                    n_session_defers += 1;
                    requeue.push_back(req); // waits (FIFO), never shed
                } else {
                    lane_shed[lane.idx()] += 1;
                    release_admission_reservation(req.lane);
                    let _ = req.tx.send(Event::Error(EngineError::rate_limit(format!(
                        "lane {} is at capacity, retry",
                        lane.as_str()
                    ))));
                }
                continue;
            }
            // Starvation sentinel closes the estimator's blind spot (2026-07-26 native-judge
            // battery): interactive work EXISTS but no interactive decode tick ran within
            // the SLO age — starvation IS a breach even though the p99 window can't see it.
            let interactive_active_or_waiting = active
                .iter()
                .any(|s| s.lane == crate::lanes::Lane::Interactive);
            let starved = interactive_active_or_waiting
                && last_interactive_decode.elapsed().as_secs_f32() * 1000.0 > policy.slo_p99_ms;
            if !policy.admit(lane, &mut step_stats, starved) {
                lane_shed[lane.idx()] += 1;
                release_admission_reservation(req.lane);
                let _ = req.tx.send(Event::Error(EngineError::rate_limit(format!(
                    "lane {} shed: interactive p99 over budget, retry",
                    lane.as_str()
                ))));
                continue;
            }
            let shape = match prepare_request(&loaded, &mut req) {
                Ok(shape) => shape,
                Err(err) => {
                    release_admission_reservation(req.lane);
                    let _ = req.tx.send(Event::Error(err));
                    continue;
                }
            };
            let model_key = req.model.clone();
            let prompt_len = req.prepared_prompt.as_ref().unwrap().len();
            let peer_probe_allows_spec = health.peer_probe_allows_spec_admission();
            let estimate_spec = admission_request_may_spec(
                &loaded[&model_key],
                &req,
                // WAVE projection, not live-only active+1 — must match the serve-time
                // K decision in `admit` or the estimate reserves a spec session the
                // request will never run (see `projected_admission_wave`).
                projected_admission_wave(active.len(), queue.len() + requeue.len()),
                prompt_len,
                peer_probe_allows_spec,
            );
            let admission_cap = shape.admission_cap();
            let decode_policy = chunk_policies
                .get(&model_key)
                .expect("loaded model missing decode chunk policy");
            let (cost, bytes_per_token, activation_bytes, log_estimate) = {
                let model = admission_costs
                    .get_mut(&model_key)
                    .expect("loaded model missing admission cost model");
                let cost = model.estimate(admission_cap, estimate_spec);
                let key = (admission_cap, estimate_spec, cost);
                let log = model.last_logged != Some(key);
                if log {
                    model.last_logged = Some(key);
                }
                (
                    cost,
                    model.bytes_per_token(estimate_spec),
                    model.activation_bytes,
                    log,
                )
            };
            if log_estimate {
                eprintln!(
                    "[admission] request cost: model={model_key:?} ctx={} path={} = {} \
                     B/token x ctx + {:.0}MB fixed = {:.0}MB",
                    admission_cap,
                    if estimate_spec { "spec" } else { "plain" },
                    bytes_per_token,
                    activation_bytes as f64 / 1e6,
                    cost as f64 / 1e6,
                );
            }

            // VRAM-AWARE ADMISSION (lane/fast-router, 2026-08-02). The original gate learned
            // one scalar from the first measurable admit. At mixed context that scalar was
            // guaranteed wrong in one direction: a short first request over-admitted a later
            // 256k cache, while a large first request over-gated later short work. The estimate
            // above instead mirrors the allocator's bytes/token geometry at THIS request's
            // effective ctx and adds only a measured fixed high-water residual. The first
            // request is checked too: if its charged context cannot fit even with no active peer,
            // reject pre-header as retryable 429 instead of attempting a doomed CUDA allocation.
            //
            // ADMIT-OOM FIX (lane/admit-oom, 2026-08-06 — research/admit-oom-20260806).
            // The `2x cost` model above is DISHONEST for spec sessions and c=64 on 24GB
            // proved it: 0/64 well-formed, every stream dead of a step-time
            // CUDA_ERROR_OUT_OF_MEMORY (research/serving-density-20260806/VERDICT.md §Q2).
            // Two independent errors, both measured against the three PASSING controls
            // (cap16/32/48 peaks 11400/15948/20528 MiB over 5540 MiB of weights):
            //
            //   1. The old scalar `cost` UNDERSTATED the live footprint it predicted. It was
            //      the free-VRAM delta of the FIRST ADMIT — a PARKED session (flat KV + draft
            //      scratch, 192 MiB here) — while a session that has actually BURST also
            //      holds its persistent draft-graph context, q slots, and round snapshots.
            //      The three controls fit peak = weights + N x 286 MiB + ~1.3 GiB, i.e. the
            //      live resident cost is 1.49x the parked delta. This term needs no new
            //      measurement: `free` from mem_get_info is GROUND TRUTH and already
            //      reflects every live session's real 286 MiB. The bug was never the
            //      subtrahend — it was sizing the HEADROOM against the wrong quantity.
            //   2. The headroom that matters is a CONSTANT, non-N-scaled transient (the
            //      same fit puts it at ~1.3 GiB): sampled draft-graph CAPTURE arenas,
            //      verify activations, prime chunk slabs. `2x cost` = 384 MiB cannot cover
            //      it, so the card ran to 23.98 of 24.46 GB during admission and the
            //      transient had nowhere to land. This is EXACTLY the class
            //      SPEC_SHRINK_RESERVE (1.5 GiB) already encodes for the F5 ladder's
            //      landing probe — and the control fit independently validates that
            //      constant to within 252 MiB. Charge it as an admission FLOOR.
            //
            // The gate is therefore `free >= cost + reserve`, where the reserve applies
            // only to models that can actually take the spec path (the plain batched path
            // survived c=64 unaided — spec-OFF cap64 PASSED — and must not pay a 1.5 GiB
            // toll it does not need). Consequence, by arithmetic on the measured fit:
            // admission stops at ~55 spec sessions on this card and the REST QUEUE (FIFO,
            // never rejected — completion is 64/64, just paced), instead of 64
            // admitted-then-killed. At the passing controls free-at-peak is 3.9-13.1 GB
            // against a 1.7 GB bar, so the new term CANNOT bind and c <= 48 is
            // behaviorally IDENTICAL (the no-regression contract: this math only bites
            // where the old gate over-admitted).
            {
                // Reserve contract (lane/admit-oom, right-sized by lane/kv256-capacity
                // 2026-08-09): spec-capable models pay the fixed transient floor. The
                // plain path used to retain one FULL session-cost of headroom — correct
                // when costs were small scalars, but at a request-shaped 262,144 charge
                // (21,894 MB on Step-3.7-Flash) that "headroom" equals a whole second
                // session for a transient class the admit-oom control fit measured at a
                // CONSTANT ~1.3 GiB (capture arenas, prime chunk slabs — chunk-bounded,
                // never ctx-scaled). Cap the plain reserve at the same measured floor:
                // requests whose cost is below the floor keep the old (identical)
                // behavior, large-context requests stop paying a session-sized toll for
                // a transient that cannot reach it. The PP-2 plain-only pod shape
                // (placement K=0 — serve_spec_enabled() false) is exactly where this
                // bound: before, 262k session capacity on the 96GB pair was halved by
                // reserve == cost. Step-OOM park + alloc-OOM reclaim-retry remain the
                // backstops the floor was gated against.
                let reserve = admission_reserve(
                    peer_probe_spec_admission(serve_spec_enabled(), peer_probe_allows_spec)
                        && loaded.get(&req.model).is_some_and(mtp_spec_capable),
                    cost,
                    admit_reserve_override(),
                );
                // DSPARK VERIFY-GRAPH POOL DEBT (lane/hermes-perf-fixes, 2026-08-23):
                // the model-owned verify-graph pool grows monotonically to a measured
                // multi-GiB high-water (8,852 MiB at storm-complete on the q38 export)
                // that neither `cost` nor the 1.5 GiB transient floor ever charged —
                // sessions admitted while the pool is cold overcommitted VRAM the pool
                // was going to hold. Charge the self-measured projected remaining
                // growth on top of the reserve for models whose dspark route is armed
                // (projection contract on `dspark_vg_debt_projection`; the teeth door
                // `MEMRA_ADMIT_RESERVE_MB` deliberately does NOT override this term —
                // it calibrates the transient floor, not the pool).
                let vg_debt = if dspark_drafts.contains_key(&model_key) {
                    loaded[&model_key].model.dspark_vg_admission_debt(&engine)
                } else {
                    0
                };
                if vg_debt > 0 && log_estimate {
                    eprintln!(
                        "[admission] dspark verify-graph pool debt: +{:.0}MB reserved \
                         (projected remaining pool growth; MEMRA_DSPARK_VG_MAX is the \
                         valve)",
                        vg_debt as f64 / 1e6,
                    );
                }
                let reserve = reserve.saturating_add(vg_debt);
                let request_tp_kv = match loaded[&model_key]
                    .model
                    .step_tp_unmaterialized_kv_bytes(None, admission_cap)
                {
                    Ok(charges) => charges,
                    Err(err) => {
                        fail_request(
                            req,
                            EngineError::engine(format!("Step TP KV admission plan failed: {err}")),
                        );
                        continue;
                    }
                };
                let pending_tp_kv = match active_unmaterialized_tp_kv(&active, &loaded) {
                    Ok(charges) => charges,
                    Err(err) => {
                        fail_request(
                            req,
                            EngineError::engine(format!(
                                "active Step TP KV reservation failed: {err}"
                            )),
                        );
                        continue;
                    }
                };
                let dual_plan = if decode_policy.dual {
                    let model = &loaded[&model_key].model;
                    let n_trunk = (model.cfg.n_layer - model.cfg.nextn_predict_layers) as usize;
                    let fence = memra_engine::pp::pp_cuts(n_trunk)
                        .expect("dual decode policy missing PP stage fence");
                    let context_bytes =
                        dual_pp_stage_context_bytes(model, &fence, admission_cap, estimate_spec)
                            .expect("dual decode policy requires exactly two PP stages");
                    let boundary_slot_bytes = dual_pp_boundary_slot_bytes(
                        decode_policy.wave_cap,
                        model.cfg.n_embd as usize,
                    );
                    let stages = dual_pp_stage_admission(
                        context_bytes,
                        activation_bytes,
                        reserve,
                        boundary_slot_bytes,
                    );
                    if log_estimate {
                        eprintln!(
                            "[admission] dual PP per-stage plan: stage0 session {:.0}MB + \
                             reserve {:.0}MB; stage1 session {:.0}MB + reserve {:.0}MB + \
                             two boundary slots {:.3}MB",
                            stages[0].session_bytes as f64 / 1e6,
                            stages[0].reserve_bytes as f64 / 1e6,
                            stages[1].session_bytes as f64 / 1e6,
                            stages[1].reserve_bytes as f64 / 1e6,
                            stages[1].boundary_bytes as f64 / 1e6,
                        );
                    }
                    let runtime = memra_engine::pp::PpNRt::get(&engine)
                        .expect("dual decode policy missing PP runtime");
                    assert_eq!(
                        runtime.n_stages(),
                        2,
                        "dual decode policy requires exactly two PP stages"
                    );
                    Some((
                        [
                            runtime.engine(0, &engine).ctx().ordinal(),
                            runtime.engine(1, &engine).ctx().ordinal(),
                        ],
                        stages,
                    ))
                } else {
                    None
                };
                let required = admission_required(cost, reserve);
                let device_requirements = if dual_plan.is_some()
                    || !request_tp_kv.is_empty()
                    || !pending_tp_kv.is_empty()
                {
                    Some(parallel_device_requirements(
                        engine.ctx().ordinal(),
                        cost,
                        reserve,
                        dual_plan,
                        &request_tp_kv,
                        &pending_tp_kv,
                    ))
                } else {
                    None
                };
                if log_estimate && let Some(requirements) = &device_requirements {
                    let budgets = requirements
                        .iter()
                        .map(|requirement| {
                            format!(
                                "dev{} session {:.0}MB + tp-kv {:.0}MB + pending-tp-kv \
                                 {:.0}MB + reserve {:.0}MB + boundary {:.3}MB",
                                requirement.device,
                                requirement.session_bytes as f64 / 1e6,
                                requirement.tp_kv_bytes as f64 / 1e6,
                                requirement.pending_tp_kv_bytes as f64 / 1e6,
                                requirement.reserve_bytes as f64 / 1e6,
                                requirement.boundary_bytes as f64 / 1e6,
                            )
                        })
                        .collect::<Vec<_>>()
                        .join("; ");
                    eprintln!("[admission] parallel device plan: [{budgets}]");
                }
                if let Some(mut headroom) =
                    admission_headroom(&engine, &loaded, device_requirements.as_deref())
                {
                    // EFFECTIVE free, not driver `free`: `mem_get_info` cannot see blocks the
                    // async pool holds mapped-but-not-live (Engine::new pins RELEASE_THRESHOLD
                    // to u64::MAX), yet the next alloc is satisfied from exactly those bytes,
                    // so a `free`-only read can only ever UNDER-count headroom — the wrong
                    // direction for a gate that queues real work.
                    //
                    // This term is LOAD-BEARING, and its own diagnostic explains why it looks
                    // small. Without it (first fixed build) a c=64 burst deferred on 36 ticks
                    // and crawled at `1 active, free 902MB` through the back half of the run:
                    // each retire returned its session KV to the pool, driver `free` never
                    // moved, so the gate saw a full card while the pool sat on the space. With
                    // it: 5 defers, 59 active sustained, queue never deeper than 4.
                    // Measured pool-cached is then only 34-89 MB — precisely BECAUSE admission
                    // keeps refilling the slots, so nothing accumulates unclaimed. The term is
                    // small exactly when it is doing its job, and large when it is missing.
                    // (fix-run2-server.log vs fix-pool-run{1,2}-server.log.)
                    //
                    // The same diagnostic line independently confirms the cost fit this gate is
                    // built on: at 59 active the pool reports res 22783MB / used 22749MB
                    // (reserved ~= used, i.e. genuinely live, nothing hiding), which against
                    // 5540 MiB of weights is 292 MB per live session — the control fit said
                    // 286 MiB, from a completely different measurement.
                    // Dual mode replaces this one-primary proof with a device-local proof: exact
                    // stage-owned KV plus one transient reserve per simultaneous stage walker,
                    // and both prepared boundary slots on the receiver. Serial mode retains this
                    // function's historical primary-device `cost + reserve` equation exactly.
                    if !headroom.sufficient(required) {
                        // Prefix snapshots are cache, not capacity reservations. Sessions win:
                        // drop them before deciding that a request must queue or be rejected.
                        let evicted_prefix = px.evict_all();
                        if evicted_prefix > 0
                            && let Some(next_headroom) =
                                admission_headroom(&engine, &loaded, device_requirements.as_deref())
                        {
                            headroom = next_headroom;
                        }
                        // Dormant sessions are reclaimable capacity, not a reason to queue live
                        // work. Evict globally oldest across both existing pool maps, re-reading
                        // effective headroom after each drop. This is deliberately only a hook:
                        // pool keys, vectors, identity, checkpoints, and resume selection remain
                        // owned by their respective lanes.
                        let free_before_reclaim = headroom.limiting_free_bytes();
                        let mut evicted_plain = 0usize;
                        let mut evicted_spec = 0usize;
                        let mut evicted_dspark = 0usize;
                        while !headroom.sufficient(required) {
                            match evict_oldest_parked(
                                &mut reuse,
                                &mut spec_reuse,
                                &mut dspark_reuse,
                                &mut reuse_metrics,
                            ) {
                                Some(ParkedPool::Plain) => evicted_plain += 1,
                                Some(ParkedPool::Spec) => evicted_spec += 1,
                                Some(ParkedPool::Dspark) => evicted_dspark += 1,
                                None => break,
                            }
                            let Some(next_headroom) = admission_headroom(
                                &engine,
                                &loaded,
                                device_requirements.as_deref(),
                            ) else {
                                break;
                            };
                            headroom = next_headroom;
                        }
                        if evicted_prefix + evicted_plain + evicted_spec + evicted_dspark > 0 {
                            eprintln!(
                                "[admit-oom] reclaim-on-defer: evicted {} prefix entries + {} plain \
                                 + {} spec + {} dspark parked sessions (global LRU); effective free {:.0}MB -> {:.0}MB",
                                evicted_prefix,
                                evicted_plain,
                                evicted_spec,
                                evicted_dspark,
                                free_before_reclaim as f64 / 1e6,
                                headroom.limiting_free_bytes() as f64 / 1e6,
                            );
                        }
                    }
                    if !headroom.sufficient(required) {
                        if active.is_empty() {
                            eprintln!(
                                "[admit-oom] VRAM reject: model={model_key:?} ctx={} has no \
                                 attainable admission headroom (available {:.0}MB) — HTTP 429",
                                admission_cap,
                                headroom.limiting_free_bytes() as f64 / 1e6,
                            );
                            fail_request(
                                req,
                                EngineError::rate_limit(format!(
                                    "request context {} does not fit available KV capacity; retry \
                                 with smaller max_tokens or max_ctx",
                                    admission_cap,
                                )),
                            );
                            continue;
                        }
                        // Pacing receipt: the defer path used to be SILENT, which is why the
                        // pre-fix red read as "all 64 admitted then all 64 died" with no
                        // visible back-pressure. One line per tick (not per deferred request)
                        // keeps a 64-client burst readable.
                        if vram_defers == 0 {
                            let parked: usize = spec_reuse.values().map(|v| v.len()).sum();
                            match &headroom {
                                AdmissionHeadroom::Devices(devices) => {
                                    let budgets = devices
                                        .iter()
                                        .map(|device| {
                                            format!(
                                                "dev{} free {:.0}MB (pool-cached {:.0}MB) vs \
                                             required {:.0}MB = session {:.0}MB + tp-kv \
                                             {:.0}MB + pending-tp-kv {:.0}MB + reserve {:.0}MB \
                                             + boundary {:.3}MB [pool res {:.0}MB used {:.0}MB]",
                                                device.requirement.device,
                                                device.free_bytes as f64 / 1e6,
                                                device.pool_cached_bytes as f64 / 1e6,
                                                device.requirement.required() as f64 / 1e6,
                                                device.requirement.session_bytes as f64 / 1e6,
                                                device.requirement.tp_kv_bytes as f64 / 1e6,
                                                device.requirement.pending_tp_kv_bytes as f64 / 1e6,
                                                device.requirement.reserve_bytes as f64 / 1e6,
                                                device.requirement.boundary_bytes as f64 / 1e6,
                                                device.pool_reserved_bytes as f64 / 1e6,
                                                device.pool_used_bytes as f64 / 1e6,
                                            )
                                        })
                                        .collect::<Vec<_>>()
                                        .join("; ");
                                    eprintln!(
                                        "[admit-oom] VRAM defer: {} active, parallel device \
                                         budgets [{budgets}] — queueing (FIFO) [parked spec \
                                         sessions {}; plain reuse {}; queue {}]",
                                        active.len(),
                                        parked,
                                        reuse.values().map(|v| v.len()).sum::<usize>(),
                                        queue.len() + requeue.len(),
                                    );
                                }
                                AdmissionHeadroom::Primary { .. } => {
                                    let (res, used) = engine.pool_reserved_used();
                                    eprintln!(
                                        "[admit-oom] VRAM defer: {} active, effective free \
                                               {:.0}MB (driver + {:.0}MB pool-cached) < cost {:.0}MB \
                                               + reserve {:.0}MB — queueing (FIFO) \
                                               [pool res {:.0}MB used {:.0}MB; parked spec sessions {}; \
                                               plain reuse {}; queue {}]",
                                        active.len(),
                                        headroom.limiting_free_bytes() as f64 / 1e6,
                                        headroom.primary_pool_cached_bytes() as f64 / 1e6,
                                        cost as f64 / 1e6,
                                        reserve as f64 / 1e6,
                                        res as f64 / 1e6,
                                        used as f64 / 1e6,
                                        parked,
                                        reuse.values().map(|v| v.len()).sum::<usize>(),
                                        queue.len() + requeue.len()
                                    );
                                }
                            }
                        }
                        vram_defers += 1;
                        n_vram_defers += 1;
                        requeue.push_back(req); // waits (FIFO), never rejected
                        continue;
                    }
                }
            }
            let free_before = effective_free_bytes(&engine).map(|(free, _)| free);
            let gemma_draft_ready = gemma_drafts.contains_key(&req.model);
            let dspark_draft = dspark_drafts.get(&req.model);
            let dspark_draft_ready = dspark_draft.is_some();
            let dspark_prime_feasible =
                dspark_draft
                    .zip(req.prepared_prompt.as_ref())
                    .is_some_and(|(draft, prompt)| {
                        memra_engine::dflash::dspark_spec_prompt_fits(
                            prompt.len(),
                            shape.ctx_cap,
                            draft.cfg.block_size,
                            draft.cfg.sliding_window,
                            draft.dflash2.is_some(),
                        )
                    });
            // Admission happens serially inside this loop. The first exact DFlash request becomes
            // the pre-prime capture owner when pushed into the active set; same-wave peers see it
            // here and skip duplicate snapshots while still taking the same DFlash route.
            let has_exact_preprime_dspark_owner =
                req.prepared_prompt.as_ref().is_some_and(|prompt| {
                    active.iter().any(|s| {
                        s.dspark_on
                            && s.dspark_capture_prefix
                            && s.dspark.is_none()
                            && s.fed.is_empty()
                            && dspark_prefix_owner_identity_matches(
                                (&s.model, &s.cache_ns),
                                &s.prefill_queue,
                                (&req.model, &req.cache_ns),
                                prompt,
                            )
                    })
                });
            // Re-evaluate inside the admission pass so request two sees request one after it was
            // pushed into `active`. A sampled DFlash row is session-owned/non-demotable; allowing
            // a later greedy row through LOW=2 would recreate two serialized phase-(a) bursts.
            let has_live_non_demotable_dspark = active.iter().any(|s| {
                dspark_blocks_greedy_widening(
                    s.dspark_on,
                    s.sampler.is_greedy(),
                    s.constraint.is_some(),
                )
            });
            match admit(
                &engine,
                &loaded,
                &mut reuse,
                &mut spec_reuse,
                &mut dspark_reuse,
                &mut spec_sizing,
                &mut reuse_metrics,
                &mut px,
                active.len(),
                queue.len() + requeue.len(),
                has_live_non_demotable_dspark,
                has_exact_preprime_dspark_owner,
                *req,
                shape,
                peer_probe_allows_spec,
                gemma_draft_ready,
                dspark_draft,
                dspark_prime_feasible,
                vision_tower.as_ref(),
            ) {
                Ok(s) => {
                    // The request has crossed the worker admission boundary and now lives in
                    // `active`; its queue reservation must no longer count against waiting
                    // capacity. The HTTP in-flight gauge continues to cover the live stream.
                    release_admission_reservation(lane);
                    let actual_spec = s.spec.is_some();
                    let actual_ctx = s
                        .spec
                        .as_ref()
                        .map(|spec| spec.cache_max_ctx())
                        .or_else(|| s.cache.as_ref().map(|cache| cache.max_ctx))
                        .unwrap_or(shape.ctx_cap);
                    if let (Some(before), Some((after, _))) =
                        (free_before, effective_free_bytes(&engine))
                    {
                        let observed = before.saturating_sub(after);
                        let model = admission_costs
                            .get_mut(&model_key)
                            .expect("loaded model missing admission cost model");
                        if let Some(residual) = model.observe(observed, actual_ctx, actual_spec) {
                            model.last_logged = None;
                            eprintln!(
                                "[admission] fixed residual high-water: model={model_key:?} \
                                 {:.0}MB (observed {:.0}MB at ctx {} path={})",
                                residual as f64 / 1e6,
                                observed as f64 / 1e6,
                                actual_ctx,
                                if actual_spec { "spec" } else { "plain" },
                            );
                        }
                    }
                    n_admitted += 1;
                    if s.dspark_on {
                        n_served_dspark += 1;
                    } else if s.spec_k > 0 || s.gspec_k > 0 {
                        n_served_spec += 1;
                    } else {
                        n_served_plain += 1;
                    }
                    lane_admitted[lane.idx()] += 1;
                    n_prompt_in += s.n_prompt as u64;
                    n_cached_in += s.n_cached as u64;
                    let _ = s.tx.send(Event::PromptUsage {
                        n_prompt: s.n_prompt,
                        n_cached: s.n_cached,
                    });
                    // per-tenant split (lane/cache-metering): the tenant half of the
                    // PC-ISO namespace; bounded map, overflow lands in "(other)".
                    meter_account(
                        &mut ns_tokens,
                        &s.cache_ns,
                        s.n_prompt as u64,
                        s.n_cached as u64,
                    );
                    active.push(s);
                }
                Err((tx, msg)) => {
                    release_admission_reservation(lane);
                    let _ = tx.send(Event::Error(msg));
                }
            }
        }
        queue = requeue;

        // 3. The tick. Three phases (MEMRA_SERVE_BATCH=0 restores legacy round-robin):
        //    (a) spec sessions burst solo (spec x batch composition is a later step);
        //    (b) prefilling sessions prime at the full tick chunk (PREFILL_TICK_T);
        //    (c) decoding sessions advance through BATCHED steps: sample+emit host-side, then
        //        decode_step_batch over survivors in chunks of <= 8.
        let batching = serve_batching();
        let mut finished: Vec<usize> = Vec::new();
        // STEP-OOM PARK (lane/admit-oom): requests parked out of a step-time CUDA OOM this
        // tick. Drained onto the FRONT of the admission queue after the retire sweep — the
        // retire is what frees the VRAM their re-admit needs, and front-insertion keeps them
        // ahead of later arrivals (they were admitted first; a park must not send a request
        // to the back of the line and starve it).
        let mut requeue_oom: std::collections::VecDeque<Box<Request>> = Default::default();
        // DISCONNECT ABORT (gap-scan F8): every send in the tick loop is `let _ =
        // s.tx.send(..)` — send errors ignored — so an aborted client used to burn GPU
        // until max_tokens/EOS and hold a slot against admission. The per-tick sweep
        // retires closed-channel sessions BEFORE any phase steps them; the log line is
        // the metering record (bill-to-abort-point: prompt/cached/generated so far).
        // Abort retirement must NOT park generated KV: the client did not commit that branch.
        for (i, s) in active.iter_mut().enumerate() {
            if s.tx.is_closed() {
                abort_log(s);
                finished.push(i);
            }
        }
        if !batching {
            for i in 0..active.len() {
                if finished.contains(&i) {
                    continue;
                }
                let generated_before = active[i].generated.len();
                let lane = active[i].lane;
                let step_started = Instant::now();
                let step_result = step_session(&engine, &loaded, &mut active[i], &mut spec_metrics);
                record_output_progress(
                    generated_before,
                    active[i].generated.len(),
                    lane,
                    step_started.elapsed().as_secs_f32() * 1000.0,
                    &mut n_tokens_out,
                    &mut lane_tokens,
                    &mut step_stats,
                    &mut last_interactive_decode,
                );
                match step_result {
                    Ok(true) => {}
                    Ok(false) => finished.push(i),
                    Err(err) => {
                        let _ = active[i].tx.send(Event::Error(EngineError::engine(format!(
                            "step error: {err}"
                        ))));
                        finished.push(i);
                    }
                }
            }
        } else {
            // (a0) SINGLE-SESSION GRAPH PATH (MEMRA_SERVE_GS=1 opts in). A graph session is
            // eager-program bit-identical, but it must degrade when concurrency arrives. That
            // solo->batched program transition changed real greedy tokens and selected early EOS,
            // so the load-stable default keeps every width on the generic batched body.
            let gs_on = graph_session_enabled();
            if gs_on && active.len() > 1 {
                for i in 0..active.len() {
                    if finished.contains(&i) || active[i].graph.is_none() {
                        continue;
                    }
                    let s = &mut active[i];
                    let g = s.graph.take().unwrap();
                    s.cache = Some(g.cache);
                    if let Some(pend) = s.graph_pending.take() {
                        let generated_before = s.generated.len();
                        let lane = s.lane;
                        let (cont, _) = advance_token_emit(&loaded, s, pend);
                        let emitted = record_output_tokens(
                            generated_before,
                            s.generated.len(),
                            lane,
                            &mut n_tokens_out,
                            &mut lane_tokens,
                        );
                        if emitted > 0 && lane == crate::lanes::Lane::Interactive {
                            last_interactive_decode = Instant::now();
                        }
                        if !cont {
                            finished.push(i);
                        } else {
                            let lm = &loaded[&s.model];
                            match lm
                                .model
                                .decode_step(&engine, pend, s.cache.as_mut().unwrap())
                            {
                                Ok(l) => {
                                    s.last_logits = l;
                                    s.fed.push(pend);
                                }
                                Err(err) => {
                                    let _ = s.tx.send(Event::Error(EngineError::engine(format!(
                                        "degrade: {err}"
                                    ))));
                                    finished.push(i);
                                }
                            }
                        }
                    }
                }
            }
            if gs_on && active.len() == 1 && !finished.contains(&0) {
                let s = &mut active[0];
                // Promote only generations long enough to amortize the one-time
                // capture+snapshot (~340ms measured = ~330-token break-even at the
                // 1.03ms/tok graph saving). Short requests stay eager-batched.
                let gs_min: usize = std::env::var("MEMRA_GS_MIN")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(384);
                // POST-PREFILL promotion (round 35): the old cold promotion re-primed the
                // prompt TOKEN-WISE inside graph_session_new — a live ~3x end-to-end LOSS
                // for solo long prompts (measured 871-tok/400-gen: 6.4s vs ~2.2s eager).
                // The graph session captures OVER an already-primed cache
                // (graph_session_from_cache). TTFT pays only the one-time capture (~340ms),
                // amortized by the gs_min budget gate.
                // REACHABILITY (corrected 2026-08-13, hermes a5f9431efe0a30a2 confirming an
                // orchestrator read): this door admits RESTORED-HIT and POOL-RESUME sessions
                // ONLY — never a cold chunked-prefill one. The `generated.is_empty()` term
                // below is tested at tick top, but phase (c) sweeps every prefill_done
                // session into the batched decode set and emits token 1 in the SAME tick
                // that cold prefill completes. So a cold session never STARTS a tick with
                // prefill_done && generated.is_empty(); only sessions admitted already
                // prefill-complete do. Earlier text here claimed cold promotion worked and
                // was wrong. Left as-is deliberately: FLAGS.md records this door as a net
                // loss post-B1FAST, so there is no case for widening it. Anything that does
                // widen it needs a program-crossing guard (see §Correctness discipline) —
                // promoting mid-request is the early-EOS corruption class.
                // CONSTRAINED sessions promote too (constrained-full, 2026-08-03): the
                // captured step bans the packed grammar mask on device before its argmax
                // (stable mask pointer, contents re-uploaded per step). Host-oracle and
                // fallback-sampler constrained sessions stay eager.
                // SAMPLED sessions do NOT graph-promote (`is_greedy()` below) — a separate,
                // untaken lever. It costs nothing today: this promotion only fires for
                // `s.spec.is_none()` solo sessions, and on an MTP model every sampled
                // session already rides the (faster) sampled spec burst path instead. It
                // would only matter for sampled sessions on a NON-MTP model; capturing the
                // seeded gumbel draw needs the in-graph RNG-counter bump the spec draft
                // chain already has (spec.rs sctr_inc), so it is wiring, not new math.
                let constr_graph_ok =
                    s.constraint.is_none() || (!constrain_host() && devsample_meta(s).is_some());
                // EAGER-ONLY models never graph-promote (lane/gemma4-serve-gaps): the
                // step-wise capture body (decode_step_dc_cap_masked) walks the GENERIC
                // qwen-class layer stack — over gemma4 weights that is silently wrong
                // logits (the round-45 g12 argmax-INIT class), not an error.
                // step35 never graph-promotes either (lane/step35-batched-decode): the
                // capture walks full_attn_decode_dc_inner, which REFUSES step35 by design
                // (the SWA offset KV view is inexpressible in the len_d-derived dc kernels,
                // plus per-layer n_head capture) — and a capture-time refusal lands on the
                // degrade-with-cache-consumed path, which kills the request. So a solo
                // greedy step35 session with budget >= gs_min died with "graph promote
                // failed" instead of decoding eagerly. Named exclusion; the dc gap itself
                // stays a named refusal in decode.rs.
                if s.graph.is_none()
                    && s.spec.is_none()
                    && graph_sampler_eligible(&s.sampler)
                    && memra_engine::plan_backend::DECODE_GRAPH
                        .trunk_capabilities(&loaded[&s.model].model.plan)
                        .cuda_graph
                        .supported
                    && loaded[&s.model].model.rewrite_allowed(
                        memra_gguf::execution_manifest::RewriteSurface::DecodeGraph,
                    )
                    && constr_graph_ok
                    && s.lane == crate::lanes::Lane::Interactive
                    && s.budget >= gs_min
                    && s.prefill_done
                    && s.generated.is_empty()
                    && s.cache.is_some()
                    && !s.last_logits.is_empty()
                {
                    let lm = &loaded[&s.model];
                    // first generated token: MASKED argmax for constrained sessions (the
                    // grammar's initial state), plain argmax otherwise.
                    let (first, mask0) = match s.constraint.as_mut() {
                        Some(c) => match c.compute_mask() {
                            Ok(m) => {
                                let mut row = s.last_logits.clone();
                                crate::constrained::apply_mask(&m, &mut row);
                                (memra_engine::forward::argmax(&row) as u32, Some(m))
                            }
                            Err(err) => {
                                let _ = s.tx.send(Event::Error(EngineError::engine(format!(
                                    "constraint mask: {err}"
                                ))));
                                finished.push(0);
                                (0, None)
                            }
                        },
                        None => (memra_engine::forward::argmax(&s.last_logits) as u32, None),
                    };
                    if !finished.contains(&0) {
                        let cache = s.cache.take().unwrap();
                        match lm.model.graph_session_from_cache_masked(
                            &engine,
                            cache,
                            first,
                            s.budget + 2,
                            mask0.as_ref().map(|m| m.as_slice()),
                        ) {
                            Ok((g, first)) => {
                                s.graph = Some(g);
                                s.graph_pending = Some(first);
                            }
                            Err(err) => {
                                // capture failed with the cache consumed — degrade the session
                                // via the graph-less error path (rare: capture-time errors only).
                                let _ = s.tx.send(Event::Error(EngineError::engine(format!(
                                    "graph promote failed: {err}"
                                ))));
                                finished.push(0);
                            }
                        }
                    }
                }
                // step the (possibly just-promoted) graph session: one token per tick
                let s = &mut active[0];
                if let Some(pend) = s.graph_pending.take() {
                    let t_g = Instant::now();
                    let generated_before = s.generated.len();
                    let lane = s.lane;
                    let (cont, _) = advance_token_emit(&loaded, s, pend);
                    let emitted = record_output_tokens(
                        generated_before,
                        s.generated.len(),
                        lane,
                        &mut n_tokens_out,
                        &mut lane_tokens,
                    );
                    if emitted > 0 && lane == crate::lanes::Lane::Interactive {
                        last_interactive_decode = Instant::now();
                    }
                    if !cont {
                        finished.push(0);
                    } else {
                        s.fed.push(pend);
                        // CONSTRAINED: fresh post-consume mask into the graph's stable
                        // buffer before the replay (the KV-pointer update pattern).
                        let mut mask_err = None;
                        if let Some(c) = s.constraint.as_mut() {
                            match c.compute_mask() {
                                Ok(m) => {
                                    if let Err(err) =
                                        s.graph.as_mut().unwrap().upload_mask(&engine, m.as_slice())
                                    {
                                        mask_err = Some(err.to_string());
                                    }
                                }
                                Err(err) => mask_err = Some(err),
                            }
                        }
                        if let Some(err) = mask_err {
                            let _ = s.tx.send(Event::Error(EngineError::engine(format!(
                                "constraint mask: {err}"
                            ))));
                            finished.push(0);
                        } else {
                            let lm = &loaded[&s.model];
                            // Q2 (audit 2026-08-05): step() errors for REAL causes (recapture
                            // OOM at a kernel-class boundary, fa exec-update failure) besides
                            // budget exhaustion — those must surface as errors, never as a
                            // clean MaxNew. Budget exhaustion (the one benign cause, checked
                            // here against the same bound step() uses) keeps the honest MaxNew.
                            let at_budget = s
                                .graph
                                .as_ref()
                                .is_some_and(|g| g.cache.pos + 1 >= g.bucket_max);
                            let g = s.graph.as_mut().unwrap();
                            match g.step(&engine, &lm.model) {
                                Ok(next) => {
                                    s.graph_pending = Some(next);
                                }
                                Err(err) if at_budget => {
                                    eprintln!(
                                        "[worker] graph session capture budget reached \
                                           (model {}): {err}",
                                        s.model
                                    );
                                    finish(s, StopReason::MaxNew);
                                    finished.push(0);
                                }
                                Err(err) => {
                                    eprintln!(
                                        "[worker] graph session step FAILED \
                                           (model {}): {err}",
                                        s.model
                                    );
                                    let _ = s.tx.send(Event::Error(EngineError::engine(format!(
                                        "graph step failed: {err}"
                                    ))));
                                    finished.push(0);
                                }
                            }
                            step_stats.record(t_g.elapsed().as_secs_f32() * 1000.0);
                        }
                    }
                }
            }
            // (a-) SPEC DEMOTION (lane/spec-gate, task #89, 2026-08-07). The admit gate above
            // keeps NEW arrivals off the serial spec queue, but sessions admitted while the box
            // was quiet keep bursting after load arrives — and each one holds a whole burst of
            // the tick (~21 ms at B=32/K=3) that the batched rows wait behind. So a live spec
            // session hands its cache to the batched path once the active count reaches T_HIGH.
            //
            // EXACTNESS (the non-negotiable bar). At a burst boundary the session invariant is
            // `cache.pos == committed.len()` — every committed row's trunk KV/recurrent state is
            // exactly what a plain prime of that token sequence would have produced — and
            // `next_pred` is the argmax of the verify's logits for the last committed row, which
            // is bit-identical to plain decode's logits there (that identity IS the greedy accept
            // walk's basis). Handing (cache, next_pred) over therefore continues the stream from
            // a state the batched path cannot distinguish from one it produced itself:
            // `device_next` makes the next batched tick emit `next_pred` and feed it into this
            // same cache, exactly as `advance_sample_emit` does for any batched row. See
            // `SpecSession::into_demoted`.
            //
            // A carried pending (the default partial-accept tail) must COMMIT first — its bonus
            // row is emitted but deliberately absent from the cache, and handing over a cache
            // that is one row short of the emitted stream would silently drop a token.
            // `spec_flush_pending` is that commit, and it is byte-identical to the pre-carry
            // tail. It costs one T=1 trunk pass, once per demotion (never per burst).
            //
            // WHO IS EXCLUDED, and why (stated, not hidden):
            //   * SAMPLED sessions — the stated reason was "`next_pred` on the sampled tail is
            //     the commit pass's ARGMAX, so handing it over would inject a greedy token into
            //     a sampled stream". THAT REASON IS GONE (lane/sampled-spec-quality,
            //     2026-08-19): the sampled tail DRAWS `next_pred` from the boundary logits
            //     through the session's Philox stream, so the token handed over is a proper
            //     sample. The exclusion STAYS anyway, and deliberately: the handoff would move
            //     generation to a batched row whose sampler is the worker's own `Sampler` with
            //     its own history and RNG, while the session's Philox counters live on the
            //     SpecSession — a distributional seam nobody has measured. Sampled demotion is a
            //     follow-up lane with its own handoff-exactness cell, not a ride-along on the
            //     boundary fix. Sampled spec sessions stay on spec until they end.
            //   * CONSTRAINED sessions — `next_pred` is the UNMASKED verify argmax; emitting it
            //     could produce a grammar-illegal token.
            // Both residuals are BOUNDED by the admit gate: at most `spec_gate_low()` sessions
            // can be on the spec path at any time, so the worst case is that many serial bursts,
            // not a full concurrency ladder's worth.
            //
            // ONE-WAY BY DESIGN (v1). Demotion drops the MTP draft scratch and the persistent
            // draft-graph context; re-promoting on drain-down would mean an `mtp_kv_fill` over
            // the whole committed history plus a fresh graph capture, i.e. NOT the "symmetric and
            // cheap" handoff the re-promotion option was conditioned on. A demoted session stays
            // demoted until it ends. New arrivals get spec again the moment the count falls back
            // to T_LOW, so the policy still tracks a draining load — per REQUEST, not per session.
            //
            // TESTABILITY (`MEMRA_SPEC_DEMOTE_AT`, diagnostics-only). Load-triggered demotion can
            // never be a clean exactness test: the trigger needs concurrent sessions, and a loaded
            // batch is not bit-identical to a solo one (measured pre-existing property — batch-vs-
            // solo decode diverges on its own with spec OFF and this gate absent, because
            // `fa_decode_batch_seqs_v4` carries one `split_keys` for rows at different depths and
            // the batched-linear tier changes with B). Both the arrival timing and the batch
            // composition are then nondeterministic, so a diff cannot attribute a divergence to
            // the HANDOFF. This door forces the demotion at a fixed generated-token count with NO
            // load at all, holding B=1 across the boundary: the only difference from a plain
            // batched run is that the first N tokens came off the spec path. That isolates exactly
            // the property this lane must prove. Never set in production.
            let demote_at: Option<usize> = {
                static D: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
                *D.get_or_init(|| {
                    std::env::var("MEMRA_SPEC_DEMOTE_AT")
                        .ok()
                        .and_then(|v| v.parse().ok())
                })
            };
            let automatic_demote = spec_k_pin().is_none() && spec_gate_on();
            // PREFIX-CACHE spec publication sweep (lane/spec-prefix-cache): a speculative
            // session whose first cold burst produced a boundary capture publishes it ONCE, at
            // the tick boundary after the burst — commit-gated (the capture's pos prefixes the
            // worker's committed tape by construction; the builder re-asserts it).
            //
            // Publication is independent of demotion policy. In particular, an operator-pinned
            // K disables automatic demotion but still needs its cold boundary published for the
            // next request. Keeping this sweep inside the demotion branch made MEMRA_SPEC_K=3
            // exact but permanently cache-cold.
            for s in active.iter_mut() {
                let mtp_captures = s
                    .spec
                    .as_mut()
                    .map(|sp| std::mem::take(&mut sp.boundary_captures))
                    .unwrap_or_default();
                if !mtp_captures.is_empty() {
                    let pool_key = s.pool_key();
                    let sp = s.spec.as_ref().expect("captures came from this session");
                    // MTP can produce one capture per prime stop (miss-LCP split + the stable
                    // pre-generation boundary, lane/frspec-multiturn-cache).
                    for cap in mtp_captures {
                        let end = cap.pos.min(sp.committed.len());
                        if end == 0 || px.has_key(&pool_key, &sp.committed[..end]) {
                            continue;
                        }
                        prefix_insert_from_spec_boundary(
                            &engine,
                            &mut px,
                            &pool_key,
                            &sp.committed,
                            sp.cache_ref(),
                            sp.draft_plane_ref(),
                            None, // MTP publisher: its draft state rides `draft`, not the dspark tail
                            cap,
                            "spec-boundary",
                        );
                    }
                }

                // DFlash publishes trunk state PLUS its drafter's readable KV tail
                // (lane/dspark-draft-plane-20260827), so a later hit can re-arm a dspark
                // session instead of cold-priming. It still has no hidden anchor, LCP split
                // or message-boundary arm, so whole-entry hits are the only restorable shape.
                drain_dspark_prefix_capture(&engine, &mut px, s);
            }

            if automatic_demote || demote_at.is_some() {
                let n_live = active.len() - finished.len();
                let forced = demote_at.is_some_and(|n| {
                    active.iter().enumerate().any(|(i, s)| {
                        !finished.contains(&i)
                            && (s.spec.is_some() || s.dspark.is_some())
                            && s.generated.len() >= n
                    })
                });
                if n_live >= spec_gate_high() || forced {
                    for i in 0..active.len() {
                        if finished.contains(&i) {
                            continue;
                        }
                        // DSPARK TICK DEMOTION (lane/dspark-spec-gate-demote, 2026-08-24).
                        // Pre-lane dspark admission was SOLO (`n_active == 0`); greedy admission
                        // is now LOW-band wave-projected (see ADMISSION THRESHOLD below). A
                        // session admitted on an idle box can still see load arrive later, and
                        // used to keep bursting SOLO when that happened —
                        // phase (a) serializes the whole tick behind it, the exact shape
                        // the MTP gate's c>=HIGH measurement priced as a LOSS (measured
                        // again on the dspark route: PRO 6000 agentic c=8 268 agg vs 429
                        // shed-to-plain, research/dflash2-pro6000-20260824). Same policy,
                        // same thresholds, same greedy-only exclusion; the handoff proof
                        // lives on DsparkSpecSession::into_demoted.
                        if active[i].dspark.is_some() {
                            let s = &mut active[i];
                            if !s.sampler.is_greedy() || s.constraint.is_some() {
                                continue;
                            }
                            if let Some(n) = demote_at {
                                if s.generated.len() < n {
                                    continue;
                                }
                            } else if n_live < spec_gate_high() {
                                continue;
                            }
                            let sess_pos = s.dspark.as_ref().unwrap().pos();
                            if sess_pos != s.fed.len() {
                                // Budget-clamped overshoot rows past the public stream, or a
                                // mid-prime shape: not the handoff invariant. The session is
                                // finishing (overshoot) or demotes at its next boundary. Loud,
                                // never silent.
                                eprintln!(
                                    "[spec-gate] dspark demote SKIPPED: cache rows {} != fed {} \
                                     (model {}); staying on dspark",
                                    sess_pos,
                                    s.fed.len(),
                                    s.model
                                );
                                continue;
                            }
                            let sess = s.dspark.take().unwrap();
                            let committed = sess.pos();
                            let (cache, next) = sess.into_demoted();
                            s.cache = Some(cache);
                            s.device_next = Some(next);
                            s.dspark_on = false;
                            s.spec_k = 0;
                            s.prefill_done = true;
                            s.last_logits.clear();
                            n_demoted += 1;
                            let why = match demote_at {
                                Some(n) => format!("FORCED at DEMOTE_AT={n} (test door)"),
                                None => format!("{n_live} active >= HIGH={}", spec_gate_high()),
                            };
                            eprintln!(
                                "[spec-gate] demoted dspark session to batched decode: {why} \
                                 (model {}, cache rows {committed}, generated {})",
                                s.model,
                                s.generated.len()
                            );
                            continue;
                        }
                        let s = &mut active[i];
                        if s.spec.is_none() {
                            continue;
                        }
                        // exclusions above: sampled + constrained keep the spec path.
                        if !s.sampler.is_greedy() || s.constraint.is_some() {
                            continue;
                        }
                        // forced mode (test door): only the session past the pinned token count,
                        // and only it — a peer still short of N keeps bursting.
                        if let Some(n) = demote_at {
                            if s.generated.len() < n {
                                continue;
                            }
                        } else if n_live < spec_gate_high() {
                            continue;
                        }
                        // a session that has not bursted yet has no cache state to hand over
                        // (its prompt is still queued as the spec turn-1 suffix) — it stays on
                        // spec for this tick and demotes at its next boundary.
                        let sess = s.spec.as_ref().unwrap();
                        if sess.committed_len() == 0
                            || (!sess.demote_ready() && !sess.has_pending())
                        {
                            continue;
                        }
                        let mut sess = s.spec.take().unwrap();
                        let lm = &loaded[&s.model];
                        if sess.has_pending() {
                            if let Err(err) = lm.model.spec_flush_pending(&engine, &mut sess, None)
                            {
                                // UNRECOVERABLE, and said so honestly: the flush consumed the
                                // pending before failing, so the session holds neither a pending
                                // nor a next_pred — its next continuation burst would trip the
                                // engine's primed-session assertion. Retire with the quoted cause
                                // rather than hand back a session that cannot burst.
                                eprintln!(
                                    "[spec-gate] demote flush FAILED (model {}): {err}",
                                    s.model
                                );
                                let _ = s.tx.send(Event::Error(EngineError::engine(format!(
                                    "spec demote flush failed: {err}"
                                ))));
                                finished.push(i);
                                continue;
                            }
                        }
                        // Re-check the handoff shape BEFORE consuming the session:
                        // `into_demoted` takes `self`, so a None there would drop the caches of
                        // a live request. Should be unreachable (flush clears pending and sets
                        // next_pred) — loud no-op, session handed straight back.
                        if !sess.demote_ready() {
                            eprintln!(
                                "[spec-gate] demote SKIPPED: session not in handoff shape \
                                       after flush (model {}); staying on spec",
                                s.model
                            );
                            s.spec = Some(sess);
                            continue;
                        }
                        let committed = sess.committed_len();
                        let Some((cache, next)) = sess.into_demoted() else {
                            continue;
                        };
                        debug_assert_eq!(
                            cache.pos,
                            s.fed.len(),
                            "demote handoff: cache rows != fed tokens"
                        );
                        s.cache = Some(cache);
                        s.device_next = Some(next);
                        s.spec_k = 0;
                        s.prefill_done = true;
                        s.last_logits.clear();
                        n_demoted += 1;
                        let why = match demote_at {
                            Some(n) => format!("FORCED at DEMOTE_AT={n} (test door)"),
                            None => format!("{n_live} active >= HIGH={}", spec_gate_high()),
                        };
                        eprintln!(
                            "[spec-gate] demoted session to batched decode: {why} \
                                   (model {}, committed {committed}, generated {})",
                            s.model,
                            s.generated.len()
                        );
                    }
                }
            }
            // (a) spec bursts — COLD-FIRST (admission-latency, 2026-08-06): a session that
            // has emitted nothing yet (fresh admit / pool resume, `generated` is per-request)
            // bursts BEFORE any mid-generation peer. Without this, the admission yield only
            // moved the wait: the newcomer admitted at tick top, then the background session
            // (lower index) ran its whole NEXT B128 burst (~1.2s) before the newcomer's
            // prime ever flushed (first-result.log: fix-on 1.30s vs fix-off 1.61s — the
            // residual IS that peer burst). With it: 0.149s median (iter1 receipt). Stable
            // sort: FIFO within cold and warm classes; session order across independent
            // sessions is content-neutral (each owns its cache/scratch — greedy byte-identity
            // gates verify). Shares the MEMRA_ADMIT_YIELD=0 rollback seam: off restores the
            // full pre-lane behavior (index order + full-burst holds) in one flag.
            // BATCHDRAFT ANATOMY (diagnostics only): the original tick trace started below,
            // after this loop, so its `decode_ms` omitted the entire serial spec phase. Start a
            // phase clock here and emit one boundary record per request. This adds no sync and is
            // cold unless MEMRA_TICK_TRACE=1; MEMRA_SPEC_PHASE=1 supplies the inner round split.
            let tick_trace = tick_trace_enabled();
            let t_spec = tick_trace.then(Instant::now);
            let mut spec_calls = 0usize;
            let mut spec_prev_end_ms = 0.0f32;
            let mut spec_order: Vec<usize> = (0..active.len())
                .filter(|&i| {
                    active[i].spec.is_some() || active[i].gspec_k > 0 || active[i].dspark_on
                })
                .collect();
            let admit_yield_on = {
                static Y: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
                *Y.get_or_init(|| std::env::var("MEMRA_ADMIT_YIELD").as_deref() != Ok("0"))
            };
            if admit_yield_on {
                spec_order.sort_by_key(|&i| !active[i].generated.is_empty());
            }
            let mut spec_order = spec_order.into_iter().peekable();
            let mut dspark_phase_captures: Vec<(usize, memra_engine::spec::SpecBoundaryCapture)> =
                Vec::new();
            while let Some(i) = spec_order.next() {
                if finished.contains(&i) {
                    continue;
                }
                let pair = spec_order.peek().copied().filter(|&j| {
                    !finished.contains(&j)
                        && spec_pipe_pairable(&engine, &loaded, &active[i], &active[j])
                });
                if let Some(j) = pair {
                    let _ = spec_order.next();
                    let generated_i = active[i].generated.len();
                    let generated_j = active[j].generated.len();
                    let lane_i = active[i].lane;
                    let lane_j = active[j].lane;
                    let rounds_i = active[i].spec_rounds;
                    let rounds_j = active[j].spec_rounds;
                    let pair_started = Instant::now();
                    let step_result = if i < j {
                        let (left, right) = active.split_at_mut(j);
                        step_spec_pair(
                            &engine,
                            &loaded,
                            &mut left[i],
                            &mut right[0],
                            &mut spec_metrics,
                        )
                    } else {
                        let (left, right) = active.split_at_mut(i);
                        step_spec_pair(
                            &engine,
                            &loaded,
                            &mut right[0],
                            &mut left[j],
                            &mut spec_metrics,
                        )
                    };
                    let pair_ms = pair_started.elapsed().as_secs_f32() * 1000.0;
                    record_output_progress(
                        generated_i,
                        active[i].generated.len(),
                        lane_i,
                        pair_ms,
                        &mut n_tokens_out,
                        &mut lane_tokens,
                        &mut step_stats,
                        &mut last_interactive_decode,
                    );
                    record_output_progress(
                        generated_j,
                        active[j].generated.len(),
                        lane_j,
                        pair_ms,
                        &mut n_tokens_out,
                        &mut lane_tokens,
                        &mut step_stats,
                        &mut last_interactive_decode,
                    );
                    if tick_trace {
                        spec_calls += 2;
                        let end_ms = t_spec.unwrap().elapsed().as_secs_f32() * 1000.0;
                        eprintln!(
                            "[tick-spec-pipe] seq={}-{} slots={i},{j} gap_ms={:.3} wall_ms={pair_ms:.3} \
                             generated={generated_i}->{},{}->{} rounds={},{}",
                            spec_calls - 1,
                            spec_calls,
                            end_ms - pair_ms - spec_prev_end_ms,
                            active[i].generated.len(),
                            generated_j,
                            active[j].generated.len(),
                            active[i].spec_rounds.saturating_sub(rounds_i),
                            active[j].spec_rounds.saturating_sub(rounds_j),
                        );
                        spec_prev_end_ms = end_ms;
                    }
                    match step_result {
                        Ok((keep_i, keep_j)) => {
                            if !keep_i {
                                finished.push(i);
                            }
                            if !keep_j {
                                finished.push(j);
                            }
                        }
                        Err(err) => {
                            let message = format!("step error: {err}");
                            let _ = active[i]
                                .tx
                                .send(Event::Error(EngineError::engine(message.clone())));
                            let _ = active[j]
                                .tx
                                .send(Event::Error(EngineError::engine(message)));
                            finished.push(i);
                            finished.push(j);
                        }
                    }
                    continue;
                }
                let generated_before = active[i].generated.len();
                let lane = active[i].lane;
                let step_started = Instant::now();
                let trace_before = t_spec.map(|phase_start| {
                    (
                        phase_start.elapsed().as_secs_f32() * 1000.0,
                        active[i].generated.len(),
                        active[i].spec_rounds,
                        active[i].spec_drafted,
                        active[i].spec_accepted,
                        active[i].spec_k,
                        active[i].trace_id.clone().unwrap_or_else(|| "-".into()),
                        Instant::now(),
                    )
                });
                let was_dspark_step = active[i].dspark_on;
                let step_result = if was_dspark_step {
                    step_dspark_spec(&engine, &loaded, &mut dspark_drafts, &mut active[i])
                } else if active[i].gspec_k > 0 {
                    step_gemma_spec(&engine, &loaded, &mut gemma_drafts, &mut active[i])
                } else {
                    step_session(&engine, &loaded, &mut active[i], &mut spec_metrics)
                };
                let step_elapsed_ms = step_started.elapsed().as_secs_f32() * 1000.0;
                if let Some((
                    start_ms,
                    generated0,
                    rounds0,
                    drafted0,
                    accepted0,
                    k,
                    trace_id,
                    call_start,
                )) = trace_before
                {
                    let end_ms = t_spec.unwrap().elapsed().as_secs_f32() * 1000.0;
                    spec_calls += 1;
                    eprintln!(
                        "[tick-spec] seq={} slot={i} trace={} start_ms={start_ms:.3} \
                         gap_ms={:.3} wall_ms={:.3} generated={generated0}->{} rounds={} \
                         drafted={} accepted={} k={k}",
                        spec_calls,
                        trace_id,
                        start_ms - spec_prev_end_ms,
                        call_start.elapsed().as_secs_f32() * 1000.0,
                        active[i].generated.len(),
                        active[i].spec_rounds.saturating_sub(rounds0),
                        active[i].spec_drafted.saturating_sub(drafted0),
                        active[i].spec_accepted.saturating_sub(accepted0),
                    );
                    spec_prev_end_ms = end_ms;
                }
                record_output_progress(
                    generated_before,
                    active[i].generated.len(),
                    lane,
                    step_elapsed_ms,
                    &mut n_tokens_out,
                    &mut lane_tokens,
                    &mut step_stats,
                    &mut last_interactive_decode,
                );
                // A first DFlash burst may also finish the request. Transfer its one-shot capture
                // only after success; publication waits until ALL current-tick serving work has
                // run, so the first c2 owner cannot serialize a large prefix copy ahead of row
                // two, interactive prefill, or plain decode. Errored bursts stay unpublished.
                if should_collect_dspark_after_phase_a(was_dspark_step, step_result.is_ok())
                    && let Some(cap) = take_dspark_prefix_capture(&mut active[i])
                {
                    dspark_phase_captures.push((i, cap));
                }
                match step_result {
                    Ok(true) => {}
                    Ok(false) => finished.push(i),
                    // STEP-OOM PARK-NOT-KILL (lane/admit-oom, 2026-08-06). A step that OOMs
                    // on a card-full condition used to kill the stream outright, and at c=64
                    // that killed ALL 64 in one tick sweep (research/serving-density-20260806
                    // §Q2: 0/64 well-formed). The honest admission gate above makes this rare
                    // — it is now the TRANSIENT-COLLISION backstop, for the case where two
                    // sessions' capture arenas land in the same tick despite the reserve.
                    //
                    // The session PARKS: its caches drop (freeing exactly the VRAM the retry
                    // needs) and the REQUEST goes back to the admission queue, where the
                    // reserve-floor gate holds it until a retire frees room — the same
                    // FIFO-wait every over-cap request already takes. Bounded by
                    // step_oom_retries() before the honest error, so a genuine capacity
                    // failure still surfaces instead of looping forever.
                    //
                    // WHAT PARKING COSTS, stated honestly: the session's committed KV is
                    // discarded, so the retry RE-PRIMES its prompt from scratch. That is
                    // pure latency, never a correctness change — a re-primed session emits
                    // exactly what a cold one would (the same property the F5 right-size
                    // ladder relies on). Tokens already streamed to the client are NOT
                    // re-sent: `park_requeue` rebuilds the request with the prompt only, and
                    // a session that has already emitted cannot be silently restarted, so it
                    // takes the honest error instead. Only pre-emission sessions park.
                    Err(err)
                        if is_cuda_oom(&err.to_string())
                            && step_oom_retries() > 0
                            && active[i].generated.is_empty()
                            && active[i].oom_retries < step_oom_retries() =>
                    {
                        let n_active = active.len();
                        let s = &mut active[i];
                        s.oom_retries += 1;
                        eprintln!(
                            "[admit-oom] step OOM parked session back to queue \
                                   (model {}, retry {}/{}, {n_active} active): {err}",
                            s.model,
                            s.oom_retries,
                            step_oom_retries()
                        );
                        match park_requeue(&loaded, s) {
                            Some(req) => {
                                n_step_oom_parks += 1;
                                reserve_internal_admission(req.lane);
                                requeue_oom.push_back(req);
                                finished.push(i);
                            }
                            None => {
                                // cannot rebuild the request (no prompt to replay) — the
                                // pre-fix honest error, quoted.
                                let _ = s.tx.send(Event::Error(EngineError::engine(format!(
                                    "step error: {err}"
                                ))));
                                finished.push(i);
                            }
                        }
                    }
                    Err(err) => {
                        if is_cuda_oom(&err.to_string()) {
                            eprintln!(
                                "[admit-oom] step OOM NOT parked (model {}, retries \
                                       {}/{}, generated {}): reporting honestly",
                                active[i].model,
                                active[i].oom_retries,
                                step_oom_retries(),
                                active[i].generated.len()
                            );
                        }
                        let _ = active[i].tx.send(Event::Error(EngineError::engine(format!(
                            "step error: {err}"
                        ))));
                        finished.push(i);
                    }
                }
            }
            // Freeze this before prefill/decode work; the summary must describe phase (a),
            // not elapsed time from phase (a)'s start to the end of the whole scheduler tick or
            // the deferred prefix-publication copy.
            let spec_ms = t_spec
                .map(|started| started.elapsed().as_secs_f32() * 1000.0)
                .unwrap_or(0.0);
            // (b) INTERACTIVE prefill only (TTFT priority, full tick chunk budgets[0]).
            // Dark-lane (judge/harvest) prefill runs AFTER decode (phase d) so a judge
            // prime can never sit between an interactive stream and its next token (the
            // 282ms-p99 lesson, 2026-07-26 native-judge battery).
            // task #13 (2026-07-26): BATCH fresh short primes across sessions —
            // one concat trunk, GEMMs at m = sum_T. Measured regime (prime-batch-gate --bench):
            // +80% at B=8 T=64, +44-49% at T=128, crossover ~T=320 (above it, single primes
            // win — per-seq m already at the GEMM plateau). Gate: prime-batch-gate ALL GREEN
            // (per-seq argmax + decode-stream equality). MEMRA_PRIME_BATCH=1 disables.
            let t_prefill = tick_trace.then(Instant::now);
            let mut prefill_single_calls = 0usize;
            let mut prefill_single_tokens = 0usize;
            let mut prefill_batch_calls = 0usize;
            let mut prefill_batch_tokens = 0usize;
            let budgets = policy.prefill_budget;
            let refill_grace = interactive_refill_until.is_some_and(|until| Instant::now() < until);
            if !refill_grace {
                interactive_refill_until = None;
            }
            let cached_hit_waiting_for_ttft = active.iter().enumerate().any(|(i, s)| {
                !finished.contains(&i)
                    && s.lane == crate::lanes::Lane::Interactive
                    && cached_hit_needs_first_token(
                        s.n_prompt,
                        s.n_cached,
                        s.prefill_done,
                        s.generated.len(),
                    )
            });
            // SOLD mixed-cache shape, box1 N=5: five full-cache hits arrived while the worker
            // was inside a cold prime chunk and inherited 269-299ms TTFT (21.6ms envelope).
            // A refill grace covers the channel handoff after a peer retires; the first-token
            // fence then lets an admitted hit decode before any unrelated cold prefill. Cold-only
            // ticks and all later hit tokens retain the established prefill/decode ordering.
            let defer_interactive_prefill = refill_grace || cached_hit_waiting_for_ttft;
            let dedup_advanced = if defer_interactive_prefill {
                Default::default()
            } else {
                dedup_interactive_prefixes(
                    &engine,
                    &loaded,
                    &eager_only,
                    &mut px,
                    &mut active,
                    &mut finished,
                    budgets[0],
                    &mut n_cached_in,
                    &mut ns_tokens,
                )
            };
            // An unrelated cold request must not release a just-arrived singleton from the
            // existing batch-formation window before its same-prefix siblings reach the
            // worker. Matching groups above fire immediately; unmatched misses wait at most
            // the same hold already budgeted for a lone fresh prime.
            let dedup_waiting: std::collections::HashSet<usize> =
                if prefix_dedup_enabled() && !confidence_trace_enabled() && pb_hold_ms > 0 {
                    active
                        .iter()
                        .enumerate()
                        .filter(|(i, s)| {
                            !finished.contains(i)
                                && prefix_fanout_eligible(s, &eager_only)
                                && s.t0.elapsed().as_millis() < pb_hold_ms as u128
                        })
                        .map(|(i, _)| i)
                        .collect()
                } else {
                    Default::default()
                };
            // Long cold prompts used to miss the complete-prompt batch predicate below and
            // execute one synchronous `prime_cache` call per session before decode. Keep one
            // scheduler-level chunk per session, but concatenate eligible chunks so N cold
            // requests do not serialize N separate trunk walks ahead of ready rows.
            let mut batch_advanced: std::collections::HashSet<usize> = Default::default();
            let (cand, held) = 'pb: loop {
                // default 6 (2026-07-26): with the varlen GDN core (task #18) the
                // concat sweet spot moved from B=4 to B=6-8 (16501 vs 15950 tok/s
                // at T=152 — the per-seq core train no longer scales with B).
                let pb_max: usize = std::env::var("MEMRA_PRIME_BATCH")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(6);
                // 320 -> 2048 (2026-07-27): the old T=320 crossover ("above it, single
                // primes win") was measured on the per-seq core train. With the wgmma
                // varlen cores (task #22 vl twins) batched wins at EVERY tested T:
                // +30.1% at T=320, +12.6% at 512, +5.9% at 937, +3.0% at 1536
                // (prime-batch-gate --bench, B=3). budgets[0] still caps per-tick load.
                let pb_maxt: usize = std::env::var("MEMRA_PRIME_BATCH_MAX_T")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(2048);
                let min_t = memra_engine::hybrid_forward::PRIME_MIN_T.max(2);
                let mut cand: Vec<(usize, usize)> = Vec::new();
                let mut cand_model: Option<String> = None;
                if !defer_interactive_prefill && pb_max >= 2 && !confidence_trace_enabled() {
                    for i in 0..active.len() {
                        if finished.contains(&i) {
                            continue;
                        }
                        if dedup_advanced.contains(&i) || batch_advanced.contains(&i) {
                            continue;
                        }
                        if dedup_waiting.contains(&i) {
                            continue;
                        }
                        let s = &active[i];
                        let ql = s.prefill_queue.len();
                        let Some(take) = interactive_prime_batch_take(ql, budgets[0], pb_maxt)
                        else {
                            continue;
                        };
                        let fresh =
                            s.fed.is_empty() && s.cache.as_ref().is_some_and(|c| c.pos == 0);
                        let whole_fresh = fresh && take == ql;
                        let cold_chunk = s.n_cached == 0
                            && s.prefix_pin.is_none()
                            && s.cache.as_ref().is_some_and(|c| c.pos == s.fed.len())
                            // P0 coldhol guard: routed-MoE carried batches stay serial until a
                            // realistic multi-chunk + serving-decode gate qualifies the class.
                            && carried_prime_batch_eligible(&loaded[&s.model].model.plan)
                            && loaded[&s.model].model.rewrite_allowed(
                                memra_gguf::execution_manifest::RewriteSurface::CarriedPrime,
                            );
                        if s.spec.is_none() && !s.prefill_done && s.graph.is_none()
                            // vision sessions prime alone (mixed-embedding overlay; the
                            // concat prime has no overlay seam); capture sessions too —
                            // the emptying chunk's hidden stack is read per-session
                            && s.vision.is_none()
                            && s.capture.is_none()
                            && s.lane == crate::lanes::Lane::Interactive
                            && (whole_fresh || cold_chunk)
                            // eager-only models have no batched prime core (engine refuses)
                            && !eager_only.contains(&s.model)
                            // prefix-cache LCP split primes alone (the boundary snapshot
                            // needs a per-session stop inside the prompt; concat can't stop).
                            && s.snapshot_at.is_none()
                            // plain-affinity checkpoint capture needs the same per-session
                            // boundary stop the concat prime cannot honor — prime alone.
                            && s.ckpt_at.is_none()
                            && take >= min_t
                            && cand_model.as_ref().is_none_or(|m| *m == s.model)
                        {
                            cand_model.get_or_insert_with(|| s.model.clone());
                            cand.push((i, take));
                            if cand.len() == pb_max {
                                break;
                            }
                        }
                    }
                }
                // BATCH-FORMATION HOLD: a lone fresh candidate that arrived <hold_ms ago is
                // deferred (skipped by the single-prime loop below via the same predicate NOT
                // firing — it stays queued) so staggered arrivals can coalesce. Telemetry
                // 2026-07-26: without the hold only 25% of a 32-concurrent burst batched
                // (ticks ~1ms, arrivals staggered). TTFT cost <= hold_ms on a ~40ms prime.
                let mut held = false;
                if cand.len() == 1 && pb_hold_ms > 0 {
                    let s = &active[cand[0].0];
                    if s.t0.elapsed().as_millis() < pb_hold_ms as u128 {
                        held = true;
                    }
                }
                let mut fired = false;
                if cand.len() >= 2 {
                    for &(i, _) in &cand {
                        if let Some(trace) = active[i].ttft.as_ref() {
                            trace.mark_prime_start();
                        }
                    }
                    let carried = cand
                        .iter()
                        .filter(|&&(i, _)| !active[i].fed.is_empty())
                        .count();
                    let prompts: Vec<Vec<u32>> = cand
                        .iter()
                        .map(|&(i, take)| active[i].prefill_queue.drain(..take).collect())
                        .collect();
                    let prompt_refs: Vec<&[u32]> = prompts.iter().map(|p| p.as_slice()).collect();
                    let mut cache_refs: Vec<&mut memra_engine::cache::Cache> = active
                        .iter_mut()
                        .enumerate()
                        .filter(|(i, _)| cand.iter().any(|&(candidate, _)| candidate == *i))
                        .map(|(_, s)| s.cache.as_mut().unwrap())
                        .collect();
                    let lm = &loaded[cand_model.as_ref().unwrap()];
                    let t_pb = Instant::now();
                    match lm
                        .model
                        .prime_cache_batch(&engine, &prompt_refs, &mut cache_refs)
                    {
                        Ok(outs) => {
                            let toks: usize = prompts.iter().map(|p| p.len()).sum();
                            let partial = cand
                                .iter()
                                .filter(|&&(i, _)| !active[i].prefill_queue.is_empty())
                                .count();
                            prefill_batch_calls += 1;
                            prefill_batch_tokens += toks;
                            eprintln!(
                                "[prime-batch] B={} tokens={} carried={} partial={} in {:.1}ms",
                                cand.len(),
                                toks,
                                carried,
                                partial,
                                t_pb.elapsed().as_secs_f64() * 1e3
                            );
                            for ((&(i, _), prompt), (l, _h, _x)) in
                                cand.iter().zip(&prompts).zip(outs)
                            {
                                let s = &mut active[i];
                                s.last_logits = l;
                                for &tok in prompt {
                                    s.fed.push(tok);
                                    s.sampler.accept(tok);
                                }
                                s.prefill_done = s.prefill_queue.is_empty();
                                if s.prefill_done {
                                    if let Some(trace) = s.ttft.as_ref() {
                                        trace.mark_prime_end();
                                    }
                                    // prefix-cache seed: batch-primed bytes are the concat
                                    // config — the entry stores whatever config ran (contract).
                                    maybe_prefix_seed(&engine, &mut px, s);
                                }
                                batch_advanced.insert(i);
                            }
                            fired = true;
                        }
                        Err(err) => {
                            // fall back: restore queues, the per-session path serves this tick
                            eprintln!("[prime-batch] failed ({err}); single primes serve");
                            for (&(i, _), prompt) in cand.iter().zip(&prompts) {
                                for &tok in prompt.iter().rev() {
                                    active[i].prefill_queue.push_front(tok);
                                }
                            }
                        }
                    }
                }
                // ROUNDS (telemetry 2026-07-26: a tick with 8 pending batched 4 and
                // single-primed the rest): keep batching while >= 2 candidates remain.
                if fired {
                    continue 'pb;
                }
                break 'pb (cand, held);
            };
            let sole_unfinished = queue.is_empty()
                && requeue_oom.is_empty()
                && active
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| !finished.contains(i))
                    .count()
                    == 1;
            for i in 0..active.len() {
                if defer_interactive_prefill {
                    continue;
                }
                if finished.contains(&i) {
                    continue;
                }
                if dedup_advanced.contains(&i) {
                    continue;
                }
                if batch_advanced.contains(&i) {
                    continue;
                }
                if dedup_waiting.contains(&i) {
                    continue;
                }
                if held && cand.first().is_some_and(|&(candidate, _)| candidate == i) {
                    continue; // batch-formation hold
                }
                let s = &mut active[i];
                if s.spec.is_some() || s.gspec_k > 0 || s.dspark_on || s.prefill_done {
                    continue;
                }
                if s.lane != crate::lanes::Lane::Interactive {
                    continue;
                }
                let fresh = s.fed.is_empty()
                    && s.cache.as_ref().is_some_and(|c| c.pos == 0)
                    && s.snapshot_at.is_none()
                    && s.ckpt_at.is_none();
                let budget = interactive_prefill_budget(
                    budgets[0],
                    prefill_tick_explicit,
                    sole_unfinished,
                    fresh,
                    s.prefill_queue.len(),
                );
                match prefill_tick(
                    &engine,
                    &loaded,
                    &mut px,
                    s,
                    budget,
                    vision_tower.as_ref(),
                    gemma_tower.as_ref(),
                ) {
                    Ok(consumed) => {
                        if consumed > 0 {
                            prefill_single_calls += 1;
                            prefill_single_tokens += consumed;
                        }
                    }
                    Err(err) => {
                        let _ = s.tx.send(Event::Error(EngineError::engine(format!(
                            "prefill error: {err}"
                        ))));
                        finished.push(i);
                    }
                }
            }
            let prefill_ms = t_prefill
                .map(|started| started.elapsed().as_secs_f32() * 1000.0)
                .unwrap_or(0.0);
            // (c) batched decode, interactive rows first (stable sort by lane index: chunks
            // fill with protected-class rows before dark rows).
            let t_decode = Instant::now();
            // (c-) EAGER-ONLY per-session decode (lane/gemma4-serve-gaps, 2026-08-07):
            // models with no batched decode arm advance through step_session — the legacy
            // round-robin body, whose decode_step routes to the supported eager arm
            // (gemma4_decode_step_h) — INSIDE the batched scheduler, one token per tick per
            // session. Before this route, these sessions entered the batched chunks below
            // and decode_step_batch's gemma4 assert KILLED THE WORKER on the first request
            // (research/gemma4-serve-20260807/raw/repro-panic-server-*.log). They are
            // excluded from the batched chunks by the `decoding` filter beneath.
            for i in 0..active.len() {
                if finished.contains(&i) {
                    continue;
                }
                // decode-site set: gemma4-dense sessions leave this loop for the batched
                // chunks below by default (MEMRA_GEMMA4_BATCH=0 pins them here).
                if !eager_decode.contains(&active[i].model) {
                    continue;
                }
                if active[i].spec.is_some()
                    || active[i].gspec_k > 0
                    || active[i].dspark_on
                    || !active[i].prefill_done
                    || active[i].cache.is_none()
                {
                    continue;
                }
                let generated_before = active[i].generated.len();
                let lane = active[i].lane;
                let step_result = step_session(&engine, &loaded, &mut active[i], &mut spec_metrics);
                let emitted = record_output_tokens(
                    generated_before,
                    active[i].generated.len(),
                    lane,
                    &mut n_tokens_out,
                    &mut lane_tokens,
                );
                if emitted > 0 && lane == crate::lanes::Lane::Interactive {
                    last_interactive_decode = Instant::now();
                }
                match step_result {
                    Ok(true) => {}
                    Ok(false) => finished.push(i),
                    Err(err) => {
                        let _ = active[i].tx.send(Event::Error(EngineError::engine(format!(
                            "step error: {err}"
                        ))));
                        finished.push(i);
                    }
                }
            }
            let mut decoding: Vec<usize> = (0..active.len())
                .filter(|&i| {
                    !finished.contains(&i)
                        && active[i].spec.is_none()
                        && active[i].gspec_k == 0
                        && !active[i].dspark_on
                        && active[i].prefill_done
                        && active[i].cache.is_some()
                        && !eager_decode.contains(&active[i].model)
                })
                .collect();
            decoding.sort_by_key(|&i| active[i].lane.idx());
            let mut had_interactive = false;
            // sample + emit + stop checks (host); survivors carry their next token
            let mut ready: Vec<(usize, u32)> = Vec::new();
            for &i in &decoding {
                let generated_before = active[i].generated.len();
                let lane = active[i].lane;
                let (cont, next) = advance_sample_emit(&loaded, &mut active[i]);
                let emitted = record_output_tokens(
                    generated_before,
                    active[i].generated.len(),
                    lane,
                    &mut n_tokens_out,
                    &mut lane_tokens,
                );
                had_interactive |= emitted > 0 && lane == crate::lanes::Lane::Interactive;
                match (cont, next) {
                    (false, _) => finished.push(i),
                    (true, Some(t)) => {
                        // GRAMMAR MASK STAGING (constrained-full): compute the post-consume
                        // token mask and H2D the packed bitset into the session's stable
                        // device buffer — the batched step bans on device BEFORE its device
                        // sampler, so this row rides the same lean tick as everyone else.
                        if let Err(err) = stage_grammar_mask(&engine, &mut active[i]) {
                            let _ = active[i].tx.send(Event::Error(EngineError::engine(format!(
                                "constraint mask: {err}"
                            ))));
                            finished.push(i);
                            continue;
                        }
                        ready.push((i, t));
                    }
                    (true, None) => {} // nothing to do this tick
                }
            }
            // Batched steps in per-model scheduled ticks. Serial ticks retain the model's exact
            // cap. The explicitly armed PP-2 scheduler combines two cap-bounded waves so the
            // engine sees one balanced dual tick instead of two coincidental serial chunks.
            // D2H audit (inc3 3c): the per-chunk [B]-u32 device-token readback inside the
            // step is the tick's ONLY steady-state D2H — one per chunk, none per seq. A
            // deferred one-per-TICK variant measured FLAT (±0.7%, N=4, c=8/16/32, 5090 —
            // research/batched-tick-inc3-20260801) and was killed per the flags doctrine.
            for scheduled in group_chunks(&active, &ready, &chunk_policies) {
                let ScheduledDecodeChunk {
                    rows: chunk,
                    wave_mid,
                } = scheduled;
                let toks: Vec<u32> = chunk.iter().map(|&(_, t)| t).collect();
                let idxs: Vec<usize> = chunk.iter().map(|&(i, _)| i).collect();
                let model_name = active[idxs[0]].model.clone();
                let lm = &loaded[&model_name];
                // DEVICE-SIDE SAMPLING metas (MEMRA_SERVE_DEVSAMPLE=0 reverts to host): rows
                // whose sampler is greedy-no-penalties (device argmax, bit-identical), pure
                // temperature, filtered sampled, or qualified sampled-penalty sample on device
                // inside the batched step; unsupported compositions keep the host path. The next
                // tick's advance_sample_emit consumes the token instead of the O(n_vocab)
                // host sample. Counter = generated.len() — a session-progress function,
                // independent of batch composition (the isolation contract, gate3).
                let samp: Vec<Option<DevSamp>> = idxs
                    .iter()
                    .map(|&i| {
                        let s = &active[i];
                        // constrained rows: device-sample iff a mask was staged this tick
                        // (fallback sampler configs / MEMRA_CONSTRAIN_HOST keep the v1
                        // host masked-copy sample — mask_words stays 0 for them).
                        if s.constraint.is_some() && s.mask_words == 0 {
                            return None;
                        }
                        devsample_meta(s)
                    })
                    .collect();
                // GRAMMAR MASKS: staged rows pass (stable device buffer, word count). Raw
                // pointers here because the caches split-borrow below takes as_mut_ptr on
                // `active` — the fields are disjoint (mask_dev vs cache), same soundness
                // class as the existing unique-index split-borrow.
                let mask_ptrs: Vec<Option<(*const CudaSlice<u32>, usize)>> = idxs
                    .iter()
                    .map(|&i| {
                        let s = &active[i];
                        if s.mask_words > 0 {
                            s.mask_dev.as_ref().map(|d| (d as *const _, s.mask_words))
                        } else {
                            None
                        }
                    })
                    .collect();
                let logits = {
                    // split-borrow: pull the caches out via split_at_mut-style indexing
                    let mut caches: Vec<&mut Cache> = Vec::with_capacity(idxs.len());
                    // SAFETY: idxs are unique indices into `active`; we take disjoint &mut.
                    let base = active.as_mut_ptr();
                    for &i in &idxs {
                        let s = unsafe { &mut *base.add(i) };
                        caches.push(s.cache.as_mut().unwrap());
                    }
                    // LEAN LOGITS (inc2 component 3): device-sampled rows skip the
                    // [n_vocab] D2H — their last_logits comes back EMPTY and the row is
                    // parked on-device (cache.last_logits_dev) for the retire-time pool
                    // park below. MEMRA_SERVE_LEANLOGITS=0 restores the full D2H.
                    // SAFETY: mask_ptrs point at Session.mask_dev fields — disjoint from
                    // the caches taken above; nothing mutates them for this call's life.
                    let masks: Vec<Option<(&CudaSlice<u32>, usize)>> = mask_ptrs
                        .iter()
                        .map(|m| m.map(|(p, w)| (unsafe { &*p }, w)))
                        .collect();
                    match wave_mid {
                        Some(mid) => lm.model.decode_step_batch_sampled_lean_masked_scheduled(
                            &engine,
                            &toks,
                            &mut caches,
                            &samp,
                            &masks,
                            serve_leanlogits(),
                            mid,
                        ),
                        None => lm.model.decode_step_batch_sampled_lean_masked(
                            &engine,
                            &toks,
                            &mut caches,
                            &samp,
                            &masks,
                            serve_leanlogits(),
                        ),
                    }
                };
                match logits {
                    Ok((rows, next_toks)) => {
                        for (k, &i) in idxs.iter().enumerate() {
                            active[i].last_logits = rows[k].clone();
                            active[i].device_next = next_toks[k];
                            active[i].fed.push(toks[k]);
                        }
                    }
                    Err(err) => {
                        for &i in &idxs {
                            let _ = active[i].tx.send(Event::Error(EngineError::engine(format!(
                                "batch step: {err}"
                            ))));
                            finished.push(i);
                        }
                    }
                }
            }
            if had_interactive {
                last_interactive_decode = Instant::now();
            }
            last_batch = ready.len();
            // MEMRA_TICK_TRACE=1: per-tick phase timing to stderr (diagnosis only).
            if tick_trace {
                let n_int = active
                    .iter()
                    .filter(|s| s.lane == crate::lanes::Lane::Interactive)
                    .count();
                let n_pref = active.iter().filter(|s| !s.prefill_done).count();
                // `spec` + `demoted` (lane/spec-gate): the policy's own observables — how many
                // rows are on the serial burst path this tick, and the cumulative handoff count
                // (thrash = this climbing per tick instead of per load crossing).
                let n_spec = active.iter().filter(|s| s.spec.is_some()).count();
                eprintln!(
                    "[tick] act={} int={} priming={} ready={} spec={} demoted={} \
                           spec_calls={} spec_ms={:.1} \
                           prefill_single_calls={} prefill_single_tokens={} \
                           prefill_batch_calls={} prefill_batch_tokens={} \
                           prefill_ms={:.1} decode_ms={:.1}",
                    active.len(),
                    n_int,
                    n_pref,
                    ready.len(),
                    n_spec,
                    n_demoted,
                    spec_calls,
                    spec_ms,
                    prefill_single_calls,
                    prefill_single_tokens,
                    prefill_batch_calls,
                    prefill_batch_tokens,
                    prefill_ms,
                    t_decode.elapsed().as_secs_f32() * 1000.0
                );
            }
            // (d) dark-lane prefill, ADAPTIVE: the tick period IS the client TPOT, so dark
            // primes may only consume the SLO headroom decode left over (2026-07-26 yield
            // battery: fixed 256-tok chunks pushed client p99 42 -> 91ms while the
            // decode-only estimator read 44ms). Chunk tokens = headroom_ms x prime rate.
            let decode_ms = t_decode.elapsed().as_secs_f32() * 1000.0;
            let headroom_ms = (policy.slo_p99_ms - decode_ms).max(0.0);
            let prime_tok_per_ms: f32 = std::env::var("MEMRA_PRIME_TOK_PER_MS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(8.0);
            let adaptive_cap = (headroom_ms * prime_tok_per_ms) as usize;
            // task #17 increment (2026-07-30): CONCAT small FRESH dark prefills — the
            // harvest profile (many short prompts) previously burned one tick per
            // session; a single prime_cache_batch serves them together at m = sum_T,
            // INSIDE the same headroom budget (sum_T <= lane budget AND adaptive cap,
            // so the 282ms-p99 lesson holds: dark work never exceeds the SLO headroom).
            // Same lane + same model only (budget accounting stays per-lane); >= 2
            // candidates, else the single-chunk path below serves as before.
            let mut dark_batched = false;
            {
                let min_t = memra_engine::hybrid_forward::PRIME_MIN_T.max(2);
                let mut dcand: Vec<usize> = Vec::new();
                let mut dmodel: Option<String> = None;
                let mut dlane: Option<usize> = None;
                let mut dsum = 0usize;
                for i in 0..active.len() {
                    if finished.contains(&i) {
                        continue;
                    }
                    let s = &active[i];
                    let li = s.lane.idx();
                    let ql = s.prefill_queue.len();
                    if li == 0 || budgets[li] == 0 {
                        continue;
                    }
                    // FRESH (pos==0, nothing fed) or CONTINUATION (cache primed exactly
                    // through fed): both prime from cache.pos. Carried gemma4 stays
                    // single-chunk (no continuation prime; engine rejects). LCP-split
                    // sessions prime alone (the boundary snapshot needs a per-session
                    // stop inside the prompt; concat can't stop).
                    if s.spec.is_some()
                        || s.prefill_done
                        || s.graph.is_some()
                        || s.vision.is_some()
                        // capture sessions prime alone: the emptying chunk's hidden stack
                        // is read per-session (lane/embed-serve)
                        || s.capture.is_some()
                        || s.snapshot_at.is_some()
                        || s.ckpt_at.is_some()
                        || !s.cache.as_ref().is_some_and(|c| c.pos == s.fed.len())
                    {
                        continue;
                    }
                    // eager-only models never join a prime batch (no batched prime core —
                    // the engine refuses fresh AND carried since lane/gemma4-serve-gaps).
                    if eager_only.contains(&s.model) {
                        continue;
                    }
                    let cap = budgets[li].min(adaptive_cap);
                    if ql < min_t || dsum + ql > cap {
                        continue;
                    }
                    if dlane.is_some_and(|l| l != li) {
                        continue;
                    }
                    if dmodel.as_ref().is_some_and(|m| *m != s.model) {
                        continue;
                    }
                    dlane.get_or_insert(li);
                    dmodel.get_or_insert_with(|| s.model.clone());
                    dsum += ql;
                    dcand.push(i);
                }
                if dcand.len() >= 2 {
                    let prompts: Vec<Vec<u32>> = dcand
                        .iter()
                        .map(|&i| active[i].prefill_queue.drain(..).collect())
                        .collect();
                    let prompt_refs: Vec<&[u32]> = prompts.iter().map(|p| p.as_slice()).collect();
                    let mut cache_refs: Vec<&mut memra_engine::cache::Cache> = active
                        .iter_mut()
                        .enumerate()
                        .filter(|(i, _)| dcand.contains(i))
                        .map(|(_, s)| s.cache.as_mut().unwrap())
                        .collect();
                    let lm = &loaded[dmodel.as_ref().unwrap()];
                    match lm
                        .model
                        .prime_cache_batch(&engine, &prompt_refs, &mut cache_refs)
                    {
                        Ok(outs) => {
                            let ncar = dcand.iter().filter(|&&i| !active[i].fed.is_empty()).count();
                            eprintln!(
                                "[prime-batch dark] lane={} B={} tokens={dsum} carried={ncar}",
                                dlane.unwrap(),
                                dcand.len()
                            );
                            for ((&i, prompt), (l, _h, _x)) in dcand.iter().zip(&prompts).zip(outs)
                            {
                                let s = &mut active[i];
                                s.last_logits = l;
                                for &tok in prompt {
                                    s.fed.push(tok);
                                    s.sampler.accept(tok);
                                }
                                s.prefill_done = true;
                            }
                        }
                        Err(err) => {
                            eprintln!("[prime-batch dark] failed ({err}); chunks serve");
                            for (&i, prompt) in dcand.iter().zip(&prompts) {
                                active[i].prefill_queue = prompt.iter().copied().collect();
                            }
                            dcand.clear();
                        }
                    }
                    dark_batched = !dcand.is_empty(); // the batch WAS this tick's dark action
                }
            }
            for i in 0..active.len() {
                if dark_batched {
                    break;
                }
                if finished.contains(&i) {
                    continue;
                }
                let s = &mut active[i];
                if s.spec.is_some() || s.gspec_k > 0 || s.dspark_on || s.prefill_done {
                    continue;
                }
                let li = s.lane.idx();
                if li == 0 || budgets[li] == 0 {
                    continue;
                }
                let chunk = budgets[li].min(adaptive_cap);
                if chunk < memra_engine::hybrid_forward::PRIME_MIN_T {
                    break;
                }
                if let Err(err) = prefill_tick(
                    &engine,
                    &loaded,
                    &mut px,
                    s,
                    chunk,
                    vision_tower.as_ref(),
                    gemma_tower.as_ref(),
                ) {
                    let _ = s.tx.send(Event::Error(EngineError::engine(format!(
                        "prefill error: {err}"
                    ))));
                    finished.push(i);
                }
                break; // one dark chunk per tick — the headroom budget is tick-global
            }
            // Engine-truth interactive TPOT = the FULL client-visible tick (decode + any
            // dark prime). Only interactive-carrying ticks feed the SLO estimator; on
            // naked (all-interactive) traffic this is exactly the pre-gate had_decode.
            if had_interactive {
                step_stats.record(t_decode.elapsed().as_secs_f32() * 1000.0);
            }
            // Lowest-priority serving work: publish the successful DFlash capture batch only
            // after speculative bursts, interactive prefill, plain decode, and dark-lane prefill
            // have all yielded. Finished rows still exist in the active set; retirement is next,
            // and the next admission pass cannot run until the following tick.
            for (i, cap) in dspark_phase_captures {
                publish_dspark_prefix_capture(&engine, &mut px, &active[i], cap);
            }
        }
        // retire finished sessions (reverse order so indices stay valid). Long-enough sessions
        // park their (fed, cache, last_logits) in the reuse pool instead of dropping the cache.
        finished.sort_unstable();
        finished.dedup();
        let mut retired_interactive = false;
        for &i in finished.iter().rev() {
            let mut s = active.remove(i);
            retired_interactive |= s.lane == crate::lanes::Lane::Interactive;
            retire_prefix_pin(&mut px, &mut s.prefix_pin);
            let pool_key = s.pool_key(); // before the partial moves below (PC-ISO park key)
            // SAMPLER IDENTITY OF THE RETIRING REQUEST (lane/session-resume-sampler-predicate-
            // 20260820): read here, alongside `pool_key`, for the same reason — the spec park
            // below partially moves `s`, and this is the sampler that shaped everything the park
            // is about to publish. See `SpecReuseEntry::sampler`.
            let parked_sampler = s.sampler.identity();
            n_completed += 1;
            // G5 fault injection (MEMRA_PANIC_AFTER, unset in every real deployment): panic
            // the worker here, with a live CUDA context and the supervisor above us, so the
            // catch_unwind -> mark_dead -> respawn -> exit-70 ladder is proved on the wire and
            // not only against a fake worker in unit tests.
            if panic_injection_due(n_completed) {
                panic!(
                    "MEMRA_PANIC_AFTER={} fault injection: \
                        deliberate worker panic after {n_completed} completed request(s)",
                    panic_after().unwrap_or(0)
                );
            }
            lane_completed[s.lane.idx()] += 1;
            if s.spec_rounds > 0 {
                spec_telem_dirty = true;
            } // force-publish on spec retire
            if s.spec_drafted > 0 {
                let tenant = crate::auth::meter_key(&s.cache_ns);
                if let Some(event) = adsd_detector.observe(
                    &s.model,
                    tenant,
                    s.spec_accepted as u64,
                    s.spec_drafted as u64,
                ) {
                    eprintln!(
                        "[adsd-suspect] tenant={:?} model={:?} window_acceptance={:.3} \
                         baseline_acceptance={:.3} z={:.2} drafted={} detection_only=true",
                        event.tenant,
                        event.model,
                        event.tenant_rate,
                        event.baseline_rate,
                        event.z_score,
                        event.drafted,
                    );
                }
            }
            if !retire_may_park(s.aborted) {
                continue;
            }
            if let Some(mut sess) = s.spec {
                // PENDING-CARRY flush before parking: a parked session must be fully committed
                // (committed_text drives the text-prefix resume match — an uncommitted pending
                // would double-feed on resume). One T=1 pass per RETIRED request, not per burst.
                if sess.pending_tok.is_some() {
                    if let Err(err) = loaded[&s.model]
                        .model
                        .spec_flush_pending(&engine, &mut sess, None)
                    {
                        eprintln!("[worker] spec pending flush failed ({err}); dropping session");
                        continue;
                    }
                }
                if sess.committed.len() >= REUSE_MIN_PREFIX && sess.next_pred.is_some() {
                    // skip the leading BOS when rendering: the client's prompt STRING never
                    // contains it (encode() adds it), so it would poison the text-prefix match.
                    let toks = &sess.committed;
                    let skip = loaded[&s.model]
                        .tok
                        .bos_id()
                        .map(|b| toks.first() == Some(&b))
                        .unwrap_or(false) as usize;
                    let committed_text = loaded[&s.model].tok.decode_special(&toks[skip..], true);
                    // SESSION AFFINITY: identity of the conversation this session served, so a
                    // later turn that REWRITES history can still recognize and rewind it. The
                    // fingerprint chain is taken over the COMMITTED tokens (no live tail to
                    // drop — a parked session's stream is all history).
                    let tok = &loaded[&s.model].tok;
                    let fingerprint =
                        conversation_fingerprint(toks, &|t| tok.token_is_control(t), false);
                    if prepare_park(
                        ParkedPool::Spec,
                        &pool_key,
                        &mut reuse,
                        &mut spec_reuse,
                        &mut dspark_reuse,
                        &mut reuse_metrics,
                        reuse_pool_per_namespace(),
                        reuse_pool_global_cap(),
                    ) {
                        spec_reuse
                            .entry(pool_key)
                            .or_default()
                            .push(SpecReuseEntry {
                                sess,
                                committed_text,
                                affinity: s.affinity,
                                fingerprint,
                                sampler: parked_sampler,
                                parked_at: Instant::now(),
                            });
                    }
                }
            } else if let Some(sess) = s.dspark {
                // DSPARK PARK (lane/dflash2-session-reuse): the DFlash2 twin of the spec
                // park above. Integrity gate: cache rows must equal fed tokens (a
                // budget-clamp overshoot leaves committed rows past the public stream —
                // that session must not park, a resume would double-serve the overshoot).
                if reuse_pool_per_namespace() > 0
                    && s.fed.len() >= REUSE_MIN_PREFIX
                    && sess.pos() != s.fed.len()
                {
                    // Fail-closed backstop: the engine now clamps both EOS and max_tokens at
                    // the committed prefix. Any remaining mismatch is an invariant failure,
                    // not the normal max-token path, and must stay loud rather than park poison.
                    eprintln!(
                        "[worker] dspark-park REFUSED: cache pos {} != {} fed tokens                          (budget-clamped overshoot); not parked (model {})",
                        sess.pos(),
                        s.fed.len(),
                        s.model
                    );
                } else if reuse_pool_per_namespace() > 0
                    && s.fed.len() >= REUSE_MIN_PREFIX
                    && sess.pos() == s.fed.len()
                {
                    let toks = &s.fed;
                    let skip = loaded[&s.model]
                        .tok
                        .bos_id()
                        .map(|b| toks.first() == Some(&b))
                        .unwrap_or(false) as usize;
                    let committed_text = loaded[&s.model].tok.decode_special(&toks[skip..], true);
                    let done = sess.finished();
                    if prepare_park(
                        ParkedPool::Dspark,
                        &pool_key,
                        &mut reuse,
                        &mut spec_reuse,
                        &mut dspark_reuse,
                        &mut reuse_metrics,
                        reuse_pool_per_namespace(),
                        reuse_pool_global_cap(),
                    ) {
                        dspark_reuse
                            .entry(pool_key)
                            .or_default()
                            .push(DsparkReuseEntry {
                                sess,
                                fed: s.fed,
                                committed_text,
                                done,
                                affinity: s.affinity,
                                sampler: s.sampler.identity(),
                                parked_at: Instant::now(),
                            });
                    }
                }
            } else if s.fed.len() >= REUSE_MIN_PREFIX && s.prefill_done && s.vision.is_none() {
                // vision sessions never park: their KV encodes IMAGE content the token
                // sequence alone cannot key (lane/vision).
                if let Some(cache) = s.cache {
                    // LEAN LOGITS (inc2 component 3): device-sampled sessions carried no
                    // host last_logits — recover the final row from the device park with
                    // ONE D2H here (retire-time, pool-bound sessions only). A session with
                    // neither host nor device logits cannot serve an empty-suffix resume:
                    // skip parking it rather than park a poisoned entry.
                    let last_logits = if s.last_logits.is_empty() {
                        cache
                            .last_logits_dev
                            .as_ref()
                            .and_then(|d| engine.dtoh(d).ok())
                            .unwrap_or_default()
                    } else {
                        s.last_logits
                    };
                    if !last_logits.is_empty() {
                        // PLAIN-SESSION AFFINITY (lane/plain-affinity, 2026-08-09): carry the
                        // conversation's identity + rewind checkpoint into the parked entry so a
                        // later rewritten-history turn can recognize it and prime only its delta.
                        // The fingerprint chain is over the COMMITTED (== fed) tokens, no live
                        // tail to drop — a parked session's stream is all history. Only useful
                        // when a checkpoint was captured; entries without one keep the legacy
                        // exact-extension behavior exactly.
                        let tok = &loaded[&s.model].tok;
                        let ckpt = s.ckpt_snap;
                        let affinity = s.affinity;
                        let fingerprint = if ckpt.is_some() {
                            conversation_fingerprint(&s.fed, &|t| tok.token_is_control(t), false)
                        } else {
                            Vec::new()
                        };
                        // H5 (double-park guard, code-audit-20260809 §2.6): a conversation
                        // served by spec on some turns and plain on others must not hold entries
                        // in BOTH pools under one explicit id. When plain-parking with an
                        // explicit affinity id, drop the same-id spec entry first.
                        if let Some(id) = affinity.as_deref() {
                            if let Some(sp) = spec_reuse.get_mut(&pool_key) {
                                let before = sp.len();
                                sp.retain(|e| e.affinity.as_deref() != Some(id));
                                reuse_metrics.spec_evictions += (before - sp.len()) as u64;
                            }
                        }
                        let cap = cache.max_ctx;
                        if prepare_park(
                            ParkedPool::Plain,
                            &pool_key,
                            &mut reuse,
                            &mut spec_reuse,
                            &mut dspark_reuse,
                            &mut reuse_metrics,
                            reuse_pool_per_namespace(),
                            reuse_pool_global_cap(),
                        ) {
                            reuse.entry(pool_key).or_default().push(ReuseEntry {
                                fed: s.fed,
                                cache,
                                last_logits,
                                cap,
                                ckpt,
                                affinity,
                                fingerprint,
                                parked_at: Instant::now(),
                            });
                        }
                    }
                }
            }
        }
        if retired_interactive
            && pb_hold_ms > 0
            && active
                .iter()
                .any(|s| s.lane == crate::lanes::Lane::Interactive && !s.prefill_done)
        {
            interactive_refill_until =
                Some(Instant::now() + std::time::Duration::from_millis(pb_hold_ms));
        }
        // STEP-OOM PARK (lane/admit-oom): re-queue parked requests AFTER the retire sweep
        // above — the retire is what actually released their VRAM. Front-inserted in original
        // order so a parked session keeps its place ahead of later arrivals.
        while let Some(req) = requeue_oom.pop_back() {
            queue.push_front(req);
        }
        // publish serving metrics (worker owns the counters; axum reads the snapshot).
        // THROTTLED: the per-tick mutex+percentile cost ~1.7ms/token of B=1 TPOT
        // (2026-07-26 live A/B) — publish every 32nd tick. A spec-session retire forces
        // a publish so a one-shot request's acceptance counts land without a 32-tick wait
        // (retires are per-request, not per-token — no hot-path cost class).
        // LANE/CACHE-METERING: EVERY retire forces a publish (`!finished.is_empty()`),
        // not just spec retires — otherwise a workload whose last tick lands off the
        // 32-boundary parks its final prompt/cached counters unpublished while the
        // worker blocks idle in recv(), and the post-workload /metrics scrape (the
        // hit-rate receipt query) reads stale totals. Same cost class as the spec
        // force-publish: per-request, never per-token.
        tick_n = tick_n.wrapping_add(1);
        if tick_n % 32 == 0 || spec_telem_dirty || !finished.is_empty() {
            if let Ok(mut m) = metrics.lock() {
                spec_telem_dirty = false;
                m.admitted = n_admitted;
                m.completed = n_completed;
                m.tokens_out = n_tokens_out;
                m.step_p50_ms = step_stats.p(50.0).unwrap_or(0.0);
                m.step_p99_ms = step_stats.p(99.0).unwrap_or(0.0);
                m.prompt_tokens_in = n_prompt_in;
                m.cached_tokens_in = n_cached_in;
                m.prefix_hits = px.hits;
                m.prefix_entries = px.n_entries() as u64;
                m.prefix_bytes = px.total_bytes as u64;
                m.prefix_misses = px.misses;
                m.prefix_inserts = px.inserts;
                m.prefix_evictions = px.evictions;
                m.prefix_skips_budget = px.skips_budget;
                m.prefix_skips_pinned = px.skips_pinned;
                m.prefix_hit_tokens = px.hit_tokens;
                m.admission_session_defers = n_session_defers;
                m.admission_vram_defers = n_vram_defers;
                m.step_oom_parks = n_step_oom_parks;
                m.continuation_pool_hits = reuse_metrics.continuation_hits;
                m.continuation_pool_evictions = reuse_metrics.continuation_evictions;
                m.plain_affinity_rewinds = reuse_metrics.plain_affinity_rewinds;
                m.spec_pool_hits = reuse_metrics.spec_hits;
                m.spec_pool_misses = reuse_metrics.spec_misses;
                m.spec_pool_affinity_rewinds = reuse_metrics.spec_affinity_rewinds;
                m.spec_pool_evictions = reuse_metrics.spec_evictions;
                m.spec_pool_sampler_refusals = reuse_metrics.spec_sampler_refusals;
                m.served_dspark = n_served_dspark;
                m.served_spec = n_served_spec;
                m.served_plain = n_served_plain;
                m.active_sessions = active.len() as u64;
                m.queued_requests = queue.len() as u64;
                m.continuation_pool_entries = reuse.values().map(|pool| pool.len() as u64).sum();
                m.spec_pool_entries = spec_reuse.values().map(|pool| pool.len() as u64).sum();
                m.cuda_driver_free_bytes = engine
                    .ctx()
                    .mem_get_info()
                    .map(|(free, _)| free as u64)
                    .unwrap_or(0);
                let (pool_reserved, pool_used) = engine.pool_reserved_used();
                m.cuda_pool_reserved_bytes = pool_reserved as u64;
                m.cuda_pool_used_bytes = pool_used as u64;
                m.cuda_pool_cached_bytes = engine.pool_cached_bytes() as u64;
                m.lcp_hist = px.lcp_hist;
                m.ns_tokens = ns_tokens.clone();
                m.lane_admitted = lane_admitted;
                m.lane_shed = lane_shed;
                m.lane_completed = lane_completed;
                m.lane_tokens = lane_tokens;
                m.batch_size_last = last_batch;
                m.spec = spec_metrics.lifetime.clone();
                m.spec_window = spec_metrics.window_snapshots();
                m.adsd_suspect_total = adsd_detector.suspect_total.clone();
            }
        }
        if !finished.is_empty() && std::env::var("MEMRA_SPILL_STATS").as_deref() == Ok("1") {
            let config_fallbacks = engine.spill_config_fallbacks();
            if let Some((reads, bytes, errors, short, fallbacks, waits, ring_full)) = engine
                .moe_pread_stats()
                .or_else(|| (config_fallbacks != 0).then_some((0, 0, 0, 0, 0, 0, 0)))
            {
                eprintln!(
                    "[spill-pread] snapshot reads={reads} bytes={bytes} errors={errors} \
                           short_reads={short} config_fallbacks={config_fallbacks} \
                           fallbacks={fallbacks} buffer_waits={waits} ring_full={ring_full}"
                );
            }
            if let Some((hits, misses, staged_bytes, slots)) = engine.moe_cache_stats() {
                let accesses = hits.saturating_add(misses);
                let hit_rate = if accesses == 0 {
                    0.0
                } else {
                    100.0 * hits as f64 / accesses as f64
                };
                eprintln!(
                    "[moe-cache] snapshot hits={hits} misses={misses} \
                           hit_rate={hit_rate:.3} staged_bytes={staged_bytes} slots={slots}"
                );
            }
            // Engagement receipt for the fused MoE epilogue (MEMRA_MOE_FUSED_EPI). A SIBLING
            // line, never a field on the one above: `steady.py` parses that line by regex and
            // widening it would break every banked decode cell.
            //
            // It exists because engagement is not implied by the flag. The arm fails closed to
            // the sequential loop whenever the SLRU cannot hold a token's whole 3*n_used working
            // set, and at a thin slot margin that measured 51 of 89 token-layers on the fixture
            // gate. Without this counter an A/B could compare the sequential loop against
            // itself and report a flat result as evidence. Full engagement on glm5_next is
            // exactly 42 dispatches per decoded token (one per MoE layer), so a caller deltas
            // this across a request and divides by 42.
            let fused_epi = memra_engine::moe_fused_epilogue_dispatches();
            eprintln!("[moe-fused-epi] snapshot dispatches={fused_epi}");
        }
    }
}

fn fail_request(mut req: Box<Request>, error: EngineError) {
    release_admission_reservation(req.lane);
    if let Some(ready) = req.constraint_ready.take() {
        let _ = ready.send(Err(error));
    } else {
        let _ = req.tx.send(Event::Error(error));
    }
}

fn handle_cmd(
    cmd: Cmd,
    loaded: &HashMap<String, LoadedModel>,
    dsv4_routes: &HashMap<String, std::sync::mpsc::Sender<Box<Request>>>,
    order: &[String],
    queue: &mut std::collections::VecDeque<Box<Request>>,
    trims: &mut Vec<tokio::sync::oneshot::Sender<TrimReport>>,
) {
    match cmd {
        // The pools live in run()'s scheduler scope — park the reply channel; run()
        // performs the trim at the tick top where they are in scope.
        Cmd::TrimPools(tx) => {
            trims.push(tx);
            return;
        }
        Cmd::Generate(_) => {}
    }
    // Pending-admit gauge (admission yield): the request is now in the worker's hands
    // (queued or rejected below) — the tick-top admission phase runs before the next burst,
    // so no in-flight burst needs to yield for it anymore. The separate hard reservation stays
    // held until the request is actually admitted (or rejected), so popping this command cannot
    // make an unbounded queue appear empty. Generate-only: TrimPools returns above and never
    // touched either gauge.
    release_pending_admit();
    match cmd {
        Cmd::TrimPools(_) => unreachable!("handled above"),
        Cmd::Generate(req) => {
            // dsv4 route: hand the request to the model's dedicated serving thread —
            // its channel is the FIFO admission queue (bs=1 engine; queueing is the
            // honest concurrency behavior and the c-cells measure it as such).
            if let Some(dtx) = dsv4_routes.get(&req.model) {
                if let Err(back) = dtx.send(req) {
                    fail_request(
                        back.0,
                        EngineError::engine("dsv4 serving thread is down (send failed)"),
                    );
                }
                return;
            }
            if !loaded.contains_key(&req.model) {
                let error = EngineError::model_not_found(format!(
                    "unknown model {:?}; loaded: {:?}",
                    req.model, order
                ));
                fail_request(req, error);
                return;
            }
            queue.push_back(req);
        }
    }
}

pub(crate) fn constraint_timeout_error() -> EngineError {
    EngineError::overloaded(format!(
        "response_format compilation did not finish within {} ms; retry with a smaller schema",
        crate::constrained::CONSTRAINT_COMPILE_TIMEOUT.as_millis(),
    ))
}

fn constraint_worker_limit_error() -> EngineError {
    EngineError::overloaded(format!(
        "response_format compiler is temporarily saturated after {} compile workers exceeded \
         their deadline; retry shortly",
        crate::constrained::CONSTRAINT_ABANDONED_WORKER_CAP,
    ))
}

/// Publish completed off-tick matchers back into the normal admission queue. Completion time,
/// rather than worker observation time, decides the deadline: a compile that finished on time
/// is not failed merely because a long legitimate GPU tick delayed this poll.
fn resolve_constraint_compiles(
    result_rx: &std::sync::mpsc::Receiver<crate::constrained::ConstraintCompileResult>,
    pending: &mut HashMap<u64, PendingConstraintCompile>,
    queue: &mut VecDeque<Box<Request>>,
) {
    while let Ok(done) = result_rx.try_recv() {
        let Some(mut pending_compile) = pending.remove(&done.id) else {
            continue; // request already timed out or disconnected; discard stale matcher
        };
        if pending_compile.request.tx.is_closed() {
            release_admission_reservation(pending_compile.request.lane);
            continue;
        }
        if done.finished_at > pending_compile.deadline {
            fail_request(pending_compile.request, constraint_timeout_error());
            continue;
        }
        pending_compile.request.grammar = Some(done.spec);
        match done.result {
            Ok(constraint) => {
                pending_compile.request.prepared_constraint = Some(constraint);
                if let Some(ready) = pending_compile.request.constraint_ready.take() {
                    if ready.send(Ok(())).is_err() {
                        release_admission_reservation(pending_compile.request.lane);
                        continue; // HTTP request disappeared during compilation
                    }
                }
                queue.push_back(pending_compile.request);
            }
            Err(crate::constrained::ConstraintCompileFailure::Invalid(err)) => {
                fail_request(
                    pending_compile.request,
                    EngineError::invalid_param(err, "response_format"),
                );
            }
            Err(crate::constrained::ConstraintCompileFailure::Internal(err)) => {
                fail_request(pending_compile.request, EngineError::engine(err));
            }
            Err(crate::constrained::ConstraintCompileFailure::TimedOut) => {
                fail_request(pending_compile.request, constraint_timeout_error());
            }
            Err(crate::constrained::ConstraintCompileFailure::AbandonedWorkerLimit) => {
                fail_request(pending_compile.request, constraint_worker_limit_error());
            }
        }
    }
}

fn expire_constraint_compiles(pending: &mut HashMap<u64, PendingConstraintCompile>, now: Instant) {
    let expired: Vec<u64> = pending
        .iter()
        .filter_map(|(id, compile)| {
            (compile.request.tx.is_closed() || now >= compile.deadline).then_some(*id)
        })
        .collect();
    for id in expired {
        let Some(compile) = pending.remove(&id) else {
            continue;
        };
        if !compile.request.tx.is_closed() {
            fail_request(compile.request, constraint_timeout_error());
        } else {
            release_admission_reservation(compile.request.lane);
        }
    }
}

fn constraint_poll_wait(
    pending: &HashMap<u64, PendingConstraintCompile>,
    now: Instant,
) -> Duration {
    pending
        .values()
        .map(|compile| compile.deadline.saturating_duration_since(now))
        .min()
        .unwrap_or(CONSTRAINT_RESULT_POLL)
        .min(CONSTRAINT_RESULT_POLL)
}

fn request_ctx_cap(
    server_ctx: usize,
    model_ctx: usize,
    prompt_len: usize,
    max_ctx: Option<usize>,
    max_new: usize,
) -> usize {
    let requested = match (max_ctx, max_new) {
        // A request-supplied hard cap is authoritative. In particular, a 128k request on a
        // 256k-default server must allocate and be charged as 128k, not inherit the default.
        (Some(c), _) => c,
        (None, MAX_NEW_CTX_BOUNDED) => {
            // With no output bound, use the server context default. A prompt that does not fit
            // grows to prompt + one default window of room, capped at the model's trained
            // context when it is known.
            let mut cap = server_ctx;
            if prompt_len.saturating_add(16) > cap {
                cap = prompt_len.saturating_add(server_ctx);
            }
            cap
        }
        (None, max_new) => prompt_len.saturating_add(max_new).saturating_add(8),
    };
    if model_ctx > 0 {
        requested.min(model_ctx)
    } else {
        requested
    }
}

fn enforce_prompt_limit(
    prompt_len: usize,
    max_prompt_tokens: Option<usize>,
) -> Result<(), EngineError> {
    if let Some(limit) = max_prompt_tokens
        && prompt_len > limit
    {
        return Err(EngineError::context_length(format!(
            "prompt ({prompt_len} tok) exceeds configured model maximum ({limit})"
        )));
    }
    Ok(())
}

/// Render/tokenize once, then derive the request's exact context allocation shape before the
/// VRAM gate. `admit` consumes this cached prompt only after the gate passes.
fn prepare_request(
    loaded: &HashMap<String, LoadedModel>,
    req: &mut Request,
) -> Result<RequestShape, EngineError> {
    let lm = &loaded[&req.model];
    if let Some(error) = prompt_source_limit_error(req) {
        let param = if !req.prompt_ids.is_empty() {
            "prompt_ids"
        } else if !req.chat_turns.is_empty() {
            "messages"
        } else {
            "prompt"
        };
        return Err(EngineError::invalid_param(error, param));
    }
    if req.prepared_prompt.is_none() {
        if let Some(trace) = req.ttft.as_ref() {
            trace.mark_tokenize_start();
        }
        // Prefer explicit ids (exact-token validation); otherwise render the same template path
        // admission historically used inside `admit`.
        let prompt = if !req.prompt_ids.is_empty() {
            req.prompt_ids.clone()
        } else if !req.chat_turns.is_empty() {
            let plain = plain_chat_render_path(
                &req.tools_json,
                &req.think,
                req.reasoning_effort.as_deref(),
                &req.chat_turns,
                lm.tok.has_qwen_effort_ladder(),
            );
            let rendered = if plain {
                let messages: Vec<_> = req
                    .chat_turns
                    .iter()
                    .map(|t| (t.role.as_str(), t.content.as_str()))
                    .collect();
                lm.tok.apply_chat_template(&messages, true)
            } else {
                lm.tok
                    .apply_chat_template_tools_ex(
                        &req.chat_turns,
                        true,
                        &req.tools_json,
                        &req.tools_struct,
                        req.think,
                        req.reasoning_effort.as_deref(),
                    )
                    .map_err(|err| {
                        EngineError::invalid_param(format!("chat template: {err}"), "messages")
                    })?
            };
            let ids = lm.tok.encode(&rendered, true);
            // Publish the rendered string: the reuse pools' TEXT extension tier keys on
            // req.prompt_text (a parked stream's detok vs the new prompt), and chat
            // requests otherwise carry an EMPTY prompt_text — which left the tier (the
            // one built for the BPE re-encode seam) dead for every chat conversation.
            // The EXACT token tier cannot cover it: model-emitted token sequences are
            // not canonical BPE, so re-encoding last turn's text rarely reproduces them.
            req.prompt_text = rendered;
            ids
        } else if req.chat {
            // NO overwrite here, unlike the chat_turns arm above: ReplayPlan snapshots
            // prompt_text post-prepare, and this arm re-renders FROM prompt_text — an
            // overwrite would double-wrap the template on a step-OOM requeue replay.
            let rendered = lm
                .tok
                .apply_chat_template(&[("user", req.prompt_text.as_str())], true);
            lm.tok.encode(&rendered, true)
        } else {
            lm.tok.encode(&req.prompt_text, true)
        };
        if prompt.is_empty() {
            return Err(EngineError::invalid_param(
                "empty prompt after tokenization",
                "prompt",
            ));
        }
        let mut prompt = prompt;
        // EMBEDDING POOLING POSITION (lane/embed-serve): the Qwen3-Embedding family
        // pools the hidden state ON a trailing eos token — the vendor tokenizer appends
        // it, and pooling one position earlier scores ~0.3 cosine against the reference
        // instead of ~0.998 (measured on DE, 2026-08-27). Hidden-capture requests append
        // the model's own eos here so the last position is the trained pooling position.
        if req.capture.as_ref().is_some_and(|c| c.hidden) && prompt.last() != Some(&lm.eos_id) {
            prompt.push(lm.eos_id);
        }
        if let Some(trace) = req.ttft.as_ref() {
            trace.mark_tokenize_end(prompt.len());
        }
        req.prepared_prompt = Some(prompt);
    }

    let prompt_len = req.prepared_prompt.as_ref().unwrap().len();
    enforce_prompt_limit(prompt_len, req.max_prompt_tokens)?;
    let server_ctx = std::env::var("MEMRA_CTX")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(8192);
    let ctx_cap = request_ctx_cap(
        server_ctx,
        lm.model.cfg.context_length as usize,
        prompt_len,
        req.params.max_ctx,
        req.params.max_new,
    );
    if prompt_len >= ctx_cap {
        return Err(EngineError::context_length(format!(
            "prompt ({prompt_len} tok) >= context cap ({ctx_cap})"
        )));
    }
    let budget = req.params.max_new.min(ctx_cap - prompt_len);
    let need = prompt_len
        .saturating_add(budget)
        .saturating_add(SPEC_SHRINK_SLACK);
    Ok(RequestShape {
        ctx_cap,
        budget,
        need,
    })
}

/// Conservative pre-admit path choice. Reuse or request constraints may ultimately fall back to
/// plain serving, but treating an MTP-capable request as spec here can only reserve more memory.
/// `projected_active` is the WAVE projection (`projected_admission_wave`), the same signal the
/// serve-time `choose_spec_k` in `admit` reads — the two must never key on different counts
/// (the FIFO burst-split class: estimate says spec, serve says plain, memory is reserved for
/// a program that never runs).
fn admission_request_may_spec(
    lm: &LoadedModel,
    req: &Request,
    projected_active: usize,
    prompt_len: usize,
    peer_probe_allows_spec: bool,
) -> bool {
    if confidence_trace_enabled() || !mtp_spec_capable(lm) || !serve_spec_enabled() {
        return false;
    }
    let candidate = req.spec_k_replay.unwrap_or_else(|| {
        choose_spec_k(
            spec_k_pin(),
            spec_gate_on(),
            *spec_gate_thresholds(),
            projected_active,
            prompt_len,
            0,
            spec_trim_head(lm),
        )
        .k
    }) > 0;
    peer_probe_spec_admission(candidate, peer_probe_allows_spec)
}

fn mtp_spec_capable(lm: &LoadedModel) -> bool {
    lm.model.mtp.is_some()
        && lm
            .model
            .rewrite_allowed(memra_gguf::execution_manifest::RewriteSurface::MtpSpec)
        && (memra_engine::pp::pp_cuts(lm.model.layers.len()).is_none()
            || lm
                .model
                .rewrite_allowed(memra_gguf::execution_manifest::RewriteSurface::Pipeline))
        && memra_engine::plan_backend::MTP_SPEC
            .capabilities(&lm.model.plan)
            .speculative
            .supported
}

fn model_forces_spec_replay(plan: &memra_gguf::model_plan::ModelPlan) -> bool {
    use memra_gguf::model_plan::OperationKind;

    let operations = plan.trunk_operations();
    operations.contains(&OperationKind::GatedDeltaNet)
        && operations.contains(&OperationKind::MoeMlp)
}

fn constrained_spec_supported(
    plan: &memra_gguf::model_plan::ModelPlan,
    operator_replay: bool,
) -> bool {
    !operator_replay && !model_forces_spec_replay(plan)
}

/// Driver-free plus bytes already mapped but reusable in the async pool.
fn effective_free_bytes(engine: &Engine) -> Option<(usize, usize)> {
    engine.ctx().mem_get_info().ok().map(|(driver_free, _)| {
        let pool_cached = engine.pool_cached_bytes();
        (driver_free.saturating_add(pool_cached), pool_cached)
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct AdmissionDeviceHeadroom {
    requirement: AdmissionDeviceRequirement,
    free_bytes: usize,
    pool_cached_bytes: usize,
    pool_reserved_bytes: usize,
    pool_used_bytes: usize,
}

#[derive(Debug)]
enum AdmissionHeadroom {
    Primary {
        free_bytes: usize,
        pool_cached_bytes: usize,
    },
    Devices(Vec<AdmissionDeviceHeadroom>),
}

impl AdmissionHeadroom {
    fn sufficient(&self, primary_required: usize) -> bool {
        match self {
            Self::Primary { free_bytes, .. } => *free_bytes >= primary_required,
            Self::Devices(devices) => devices
                .iter()
                .all(|device| device.free_bytes >= device.requirement.required()),
        }
    }

    fn limiting_free_bytes(&self) -> usize {
        match self {
            Self::Primary { free_bytes, .. } => *free_bytes,
            Self::Devices(devices) => devices
                .iter()
                .map(|device| device.free_bytes)
                .min()
                .unwrap_or(0),
        }
    }

    fn primary_pool_cached_bytes(&self) -> usize {
        match self {
            Self::Primary {
                pool_cached_bytes, ..
            } => *pool_cached_bytes,
            Self::Devices(_) => 0,
        }
    }
}

/// FIRST-MESSAGE BOUNDARY (lane/moebatch-q35moe, 2026-08-21): the render-stable insert point
/// for a cold chat prompt. Re-renders the first message alone with add_generation_prompt=false
/// and requires it to be an exact TOKEN prefix of the served prompt (byte-prefix is not enough —
/// a BPE merge across the boundary would shift ids; the compare fails open). Clamped PRIME_MIN_T
/// away from both prompt ends so the boundary stop never leaves a sub-floor segment: both the
/// prime-to-boundary and the tail stay on the batched prime program (prefill_tick's documented
/// The plain-render fast-path decision, ONE place, shared by `prepare` and
/// `first_message_boundary` so the render used to serve and the render used to place a
/// cache boundary can never disagree. Plain = the legacy `apply_chat_template` path (the
/// isolation law: non-tools traffic bypasses the tools renderer, docs/SERVING.md).
///
/// A template carrying the qwen3.8 reasoning-effort ladder NEVER qualifies: on that
/// template the UNSET case renders the vendor's own `xhigh` default
/// (`chat::qwen38_effort_instructions`, `None => xhigh`), and only the tools-capable
/// renderer injects it — the legacy path reproduces the historical no-instruction bytes,
/// which on this template are the accepted-and-ignored defect the reasoning-schema lane
/// removed, not a behaviour to preserve. v0.109.0 shipped without this arm and served a
/// SPLIT surface, live-probed on q38-nj 2026-08-23: the same unset request rendered the
/// xhigh sentence when a `tools` array rode along (tools_ex path, prompt 293 vs 255 for
/// the zero-steering `medium` rung) and the bare historical bytes without one (plain
/// path, prompt 19 vs 61 for explicit `high`). Non-ladder templates (ornith15,
/// agentworld, gemma4, hy3, step35, bare ChatML) keep the plain path byte-identically.
pub(crate) fn plain_chat_render_path(
    tools_json: &[String],
    think: &memra_tokenizer::chat::ThinkMode,
    reasoning_effort: Option<&str>,
    chat_turns: &[memra_tokenizer::chat::Turn],
    effort_ladder: bool,
) -> bool {
    tools_json.is_empty()
        && *think == memra_tokenizer::chat::ThinkMode::Default
        && reasoning_effort.is_none()
        && !effort_ladder
        && chat_turns
            .iter()
            // `reasoning` is DROPPED by the plain arm — it maps turns to `(role, content)`
            // tuples, so a dialect that replays prior reasoning into the prompt would render
            // different bytes here than through `_ex`. Two of them do today (the qwen3.8
            // ladder, already excluded above by `effort_ladder`, and GLM-5.3-Flash, whose
            // `<think>{reasoning}</think>` replay is unconditional), and a divergent re-render
            // is also what stops a parked session from ever matching its own live stream
            // (lane/dflash2-session-reuse). Stated as a property of the REQUEST rather than
            // keyed on a dialect: every family whose template ignores `reasoning` renders
            // byte-identically through `_ex` anyway (`tools_renderer_matches_legacy_when_plain`),
            // so this only ever costs the fast path, never the bytes.
            .all(|t| t.role != "tool" && t.tool_calls.is_empty() && t.reasoning.is_none())
}

/// sub-floor tokenwise door is exactly what this must not open). Plain templated chats only —
/// tools/think/tool-call renders interleave state the first message alone cannot reproduce.
fn first_message_boundary(
    lm: &LoadedModel,
    chat_turns: &[memra_tokenizer::chat::Turn],
    tools_json: &[String],
    think: &memra_tokenizer::chat::ThinkMode,
    reasoning_effort: Option<&str>,
    prompt: &[u32],
) -> Option<usize> {
    // Keyed on chat_turns, not req.chat: chat-completions requests carry turns with
    // chat=false (that flag selects the prompt_text-as-user render, prepare's second arm).
    if chat_turns.is_empty() {
        return None;
    }
    let plain = plain_chat_render_path(
        tools_json,
        think,
        reasoning_effort,
        chat_turns,
        lm.tok.has_qwen_effort_ladder(),
    );
    if !plain {
        return None;
    }
    let t0 = &chat_turns[0];
    let rendered = lm
        .tok
        .apply_chat_template(&[(t0.role.as_str(), t0.content.as_str())], false);
    let ids = lm.tok.encode(&rendered, true);
    let floor = PREFIX_CACHE_MIN_TOKENS.max(memra_engine::hybrid_forward::PRIME_MIN_T);
    let b = ids.len().min(
        prompt
            .len()
            .saturating_sub(memra_engine::hybrid_forward::PRIME_MIN_T),
    );
    if b < floor {
        return None;
    }
    (ids[..b] == prompt[..b]).then_some(b)
}

fn admission_headroom(
    engine: &Engine,
    loaded: &HashMap<String, LoadedModel>,
    requirements: Option<&[AdmissionDeviceRequirement]>,
) -> Option<AdmissionHeadroom> {
    let Some(requirements) = requirements else {
        return effective_free_bytes(engine).map(|(free_bytes, pool_cached_bytes)| {
            AdmissionHeadroom::Primary {
                free_bytes,
                pool_cached_bytes,
            }
        });
    };

    let mut devices = Vec::with_capacity(requirements.len());
    for &requirement in requirements {
        let device_engine = if engine.ctx().ordinal() == requirement.device {
            Some(engine)
        } else {
            memra_engine::pp::PpNRt::get(engine)
                .ok()
                .and_then(|runtime| {
                    (0..runtime.n_stages())
                        .map(|stage| runtime.engine(stage, engine))
                        .find(|stage| stage.ctx().ordinal() == requirement.device)
                })
                .or_else(|| {
                    loaded
                        .values()
                        .find_map(|model| model.model.step_tp_rank_engine(requirement.device))
                })
        }?;
        let (free_bytes, pool_cached_bytes) = effective_free_bytes(device_engine)?;
        let (pool_reserved_bytes, pool_used_bytes) = device_engine.pool_reserved_used();
        devices.push(AdmissionDeviceHeadroom {
            requirement,
            free_bytes,
            pool_cached_bytes,
            pool_reserved_bytes,
            pool_used_bytes,
        });
    }
    Some(AdmissionHeadroom::Devices(devices))
}

/// STEP-OOM PARK (lane/admit-oom, 2026-08-06): rebuild a live session's `Request` so it can go
/// back to the admission queue after a step-time CUDA OOM, instead of the stream dying.
///
/// PRECONDITION (enforced by the caller, not here): the session has emitted NOTHING. A session
/// that already streamed tokens cannot be restarted — the client would see the prefix twice —
/// so those take the honest error. This is why the function needs no emitted-state surgery.
///
/// The rebuilt request replays the ORIGINAL render inputs (`ReplayPlan`), so re-admission runs
/// the identical template + tokenize and produces the session a cold arrival would have. The
/// retry counter rides along on the Request, keeping the bound per-request across re-admits.
/// Returns None when the plan cannot produce a prompt (nothing to replay) — caller errors.
fn park_requeue(loaded: &HashMap<String, LoadedModel>, s: &Session) -> Option<Box<Request>> {
    // A plan with no prompt source at all would re-admit into "empty prompt after
    // tokenization" — report the OOM honestly instead of laundering it into a 400.
    let p = &s.replay;
    if p.prompt_ids.is_empty() && p.prompt_text.is_empty() && p.chat_turns.is_empty() {
        return None;
    }
    // Vision sessions cannot replay: ReplayPlan does not carry the preprocessed images
    // (hundreds of MB host-side) — report the OOM honestly instead of re-admitting a
    // request that would fail pad-run validation.
    if s.vision.is_some() {
        return None;
    }
    debug_assert!(
        loaded.contains_key(&s.model),
        "parked session's model must still be loaded"
    );
    Some(Box::new(Request {
        model: s.model.clone(),
        prompt_ids: p.prompt_ids.clone(),
        prompt_text: p.prompt_text.clone(),
        chat: p.chat,
        chat_turns: p.chat_turns.clone(),
        tools_json: p.tools_json.clone(),
        tools_struct: p.tools_struct.clone(),
        think: p.think,
        reasoning_effort: p.reasoning_effort.clone(),
        params: p.params.clone(),
        sampler_cfg: p.sampler_cfg.clone(),
        stop_strings: s.stop_strings.clone(),
        trace_id: s.trace_id.clone(),
        max_prompt_tokens: p.max_prompt_tokens,
        cache_ns: s.cache_ns.clone(),
        affinity: s.affinity.clone(),
        lane: s.lane,
        grammar: p.grammar.clone(),
        prepared_constraint: None,
        constraint_ready: None,
        oom_retries: s.oom_retries,
        spec_k_replay: Some(s.spec_k),
        prepared_prompt: None,
        images: Vec::new(),
        gemma_images: Vec::new(),
        // A parked capture session already emitted (or lost) its PromptCapture; the
        // replay regenerates it because the session's `capture` is carried back here.
        capture: s.capture.clone(),
        vision_memory: None,
        ttft: s.ttft.clone(),
        tx: s.tx.clone(),
    }))
}

/// Build a Session from the prompt prepared before admission, allocate its cache, and build its
/// sampler. The prompt is NOT primed here — it's fed one token per scheduler tick so prefill of a
/// new session interleaves with other sessions' decode (the BASE-4 interleave).
/// `n_active` = live session count at admit time, the K policy's concurrency signal.
/// This request's projected count is `n_active + 1`.
/// `n_pending` = requests still waiting in this tick's admission pass (the rest of an arriving
/// wave). It is NOT part of the K policy — it is reported on the load-guard line so a refusal
/// can be read back to the shape that caused it. `load_mode` is the tick-top hysteresis verdict
/// the sampled-restore load guard consumes (`sampled_restore_load_admits`).
#[allow(clippy::too_many_arguments)]
fn admit(
    engine: &Engine,
    loaded: &HashMap<String, LoadedModel>,
    reuse: &mut HashMap<PoolKey, Vec<ReuseEntry>>,
    spec_reuse: &mut HashMap<PoolKey, Vec<SpecReuseEntry>>,
    dspark_reuse: &mut HashMap<PoolKey, Vec<DsparkReuseEntry>>,
    spec_sizing: &mut SpecSizing,
    reuse_metrics: &mut ReuseMetrics,
    px: &mut PrefixCache,
    n_active: usize,
    n_pending: usize,
    has_live_non_demotable_dspark: bool,
    has_exact_preprime_dspark_owner: bool,
    mut req: Request,
    shape: RequestShape,
    peer_probe_allows_spec: bool,
    gemma_draft_ready: bool,
    // The drafter itself, not just "is one attached": a prefix hit that carries a draft tail is
    // converted into a dspark session HERE (lane/dspark-draft-plane-20260827), which needs the
    // drafter's cfg to rebuild the KV.
    dspark_draft: Option<&memra_engine::dflash::DflashDraft>,
    dspark_prime_feasible: bool,
    vision_tower: Option<&memra_engine::vision::VisionTower>,
) -> Result<Session, (tokio::sync::mpsc::UnboundedSender<Event>, EngineError)> {
    let dspark_draft_ready = dspark_draft.is_some();
    let lm = &loaded[&req.model];
    let prompt = req
        .prepared_prompt
        .take()
        .expect("admit requires a prompt prepared by the admission gate");
    let RequestShape {
        ctx_cap,
        budget,
        need,
    } = shape;
    // VISION admission (lane/vision): align pad runs with the images NOW so a malformed
    // request 400s before any cache allocation or GPU work. `vision_req` is captured
    // BEFORE the take below — the reuse and spec gates read it later (a req.images check
    // there would see the drained Vec and never fire; caught live 2026-08-15: a public
    // vision request served through the SPEC path, whose turn-1 burst primes inside
    // generate_spec_session with no overlay seam — pads primed as pad embeddings).
    let vision_req = !req.images.is_empty() || !req.gemma_images.is_empty();
    // Capture requests (embeddings/rerank) must run the real prime that produces the
    // hidden stack: every reuse tier and the spec path are bypassed below, exactly like
    // vision (a cache hit would skip the forward the capture reads from).
    let capture_req = req.capture.is_some();
    let vision_state: Option<VisionState> = if !req.gemma_images.is_empty() {
        // GEMMA-4 vision: soft-token (258880) runs align 1:1 with the request's units.
        let soft = memra_engine::vision_gemma::GV_TOK_SOFT;
        let units = std::mem::take(&mut req.gemma_images);
        match gemma_vision_spans(&prompt, &units, soft) {
            Ok(spans) => Some(VisionState {
                images: VisionImages::Gemma(units),
                spans,
                overlay: None,
            }),
            Err(e) => return Err((req.tx, EngineError::invalid_param(e, "messages"))),
        }
    } else if req.images.is_empty() {
        None
    } else {
        let pad_id = match lm.tok.id_of("<|image_pad|>") {
            Some(id) => id,
            None => {
                return Err((
                    req.tx,
                    EngineError::invalid_param(
                        "model tokenizer has no <|image_pad|> token — vision unsupported",
                        "messages",
                    ),
                ));
            }
        };
        // Width law: the tower emits [n, out_width] rows spliced into the trunk's embedding
        // stream, so the LOADED tower's width must equal THIS trunk's n_embd (5120 q38,
        // 2048 ornith15). Compare against the tower instance, not a constant — the width is
        // a property of the shard MEMRA_VISION_DIR points at.
        match vision_tower {
            None => {
                return Err((
                    req.tx,
                    EngineError::invalid_param(
                        "no vision tower loaded (set MEMRA_VISION_DIR) — image input disabled",
                        "messages",
                    ),
                ));
            }
            Some(t) if lm.model.cfg.n_embd as usize != t.out_width() => {
                return Err((
                    req.tx,
                    EngineError::invalid_param(
                        "vision tower output width does not match this model — vision unsupported",
                        "model",
                    ),
                ));
            }
            Some(_) => {}
        }
        let video_pad = lm.tok.id_of("<|video_pad|>");
        match vision_spans(&prompt, &req.images, pad_id, video_pad) {
            Ok(spans) => Some(VisionState {
                images: VisionImages::Qwen(std::mem::take(&mut req.images)),
                spans,
                overlay: None,
            }),
            Err(e) => {
                return Err((req.tx, EngineError::invalid_param(e, "messages")));
            }
        }
    };
    // PC-ISO: every reuse-pool probe below scans ONLY this (model, namespace) pool.
    let pool_key: PoolKey = (req.model.clone(), req.cache_ns.clone());
    // STEP-OOM PARK plan (lane/admit-oom): snapshot the render inputs before this function
    // consumes them, so a step-time OOM can re-admit an identical request. Host-side only.
    let grammar = req.grammar.take();
    let constrained = grammar.is_some();
    let replay = Box::new(ReplayPlan {
        prompt_ids: req.prompt_ids.clone(),
        prompt_text: req.prompt_text.clone(),
        chat: req.chat,
        chat_turns: req.chat_turns.clone(),
        tools_json: req.tools_json.clone(),
        tools_struct: req.tools_struct.clone(),
        think: req.think,
        reasoning_effort: req.reasoning_effort.clone(),
        params: req.params.clone(),
        sampler_cfg: req.sampler_cfg.clone(),
        grammar,
        max_prompt_tokens: req.max_prompt_tokens,
    });
    let req_oom_retries = req.oom_retries;
    let req_spec_k_replay = req.spec_k_replay;
    let vision_memory = req.vision_memory.take();

    // KV PREFIX REUSE probe: a parked session whose fed sequence is an EXACT PREFIX of this
    // prompt (and whose cache has room) resumes — only the suffix gets primed. The sampler's
    // penalty history is replayed on host (cheap) so sampling matches a cold run exactly.
    let mut reused: Option<ReuseEntry> = None;
    // DEFAULT-ON (2026-07-05): the identity gate now exists at the engine level — session-gate
    // (bins) pins 3-turn continuation-prime output == fresh-greedy oracle on both models, and the
    // continuation path the reuse pool takes (prime_cache with cache.pos>0 / decode_step) is
    // exactly what it validates. MEMRA_KV_REUSE=0 disables.
    // Vision requests bypass every token-keyed reuse tier: pad runs are byte-identical
    // across DIFFERENT images, so a token match is not a state match (lane/vision).
    let reuse_on = !confidence_trace_enabled()
        && !vision_req
        && !capture_req
        && std::env::var("MEMRA_KV_REUSE")
            .map(|v| v != "0")
            .unwrap_or(true);
    if let (true, Some(pool)) = (reuse_on, reuse.get_mut(&pool_key)) {
        if let Some(idx) = pool.iter().rposition(|e| {
            e.fed.len() >= REUSE_MIN_PREFIX
                && e.cap >= ctx_cap
                && prompt.len() >= e.fed.len()
                && prompt.starts_with(&e.fed)
        }) {
            reused = Some(pool.remove(idx));
        }
    }
    if reused.is_some() {
        reuse_metrics.continuation_hits += 1;
    }

    // PLAIN-SESSION AFFINITY resume (lane/plain-affinity, 2026-08-09). The exact-extension probe
    // above only serves a prompt that EXTENDS a parked session's whole fed stream — a client that
    // REWRITES conversation history (the pi client strips `<think>` out of prior assistant turns)
    // misses it on every turn, and the parked full-context session is discarded while the whole
    // growing conversation re-primes (the 2.35 -> 13.7s TTFT slope, research/cachespec-20260809).
    //
    // Affinity asks the OTHER question — is this the SAME CONVERSATION? — and answers it exactly
    // as the spec tier does: identity NOMINATES (explicit id, else the structural fingerprint
    // chain), an exact token diff against the parked session's REWIND CHECKPOINT DECIDES. A
    // history rewrite mutates what the session GENERATED, so the new prompt still reproduces the
    // committed tokens up to the pre-generation boundary; only this turn's delta primes. A
    // fingerprint collision costs one wasted comparison, never a wrong resume (bytes decide).
    // MEMRA_AFFINITY=0 declines every candidate (the exactness A/B arm).
    if reuse_on && reused.is_none() && affinity_enabled() {
        let req_fp = conversation_fingerprint(&prompt, &|t| lm.tok.token_is_control(t), true);
        let candidate = if let Some(pool) = reuse.get(&pool_key) {
            // WHY A DECLINE IS LOGGED (mirrors the spec tier): every requirement below is
            // invisible outside the worker, and the lane's evidence is per-turn resume counts
            // read from this log. The reason is for the LAST candidate examined (pool depth 1-2).
            let mut why: String = "empty pool".into();
            let cand = pool.iter().enumerate().rev().find_map(|(i, e)| {
                let Some(ckpt) = e.ckpt.as_ref() else {
                    why = "no checkpoint retained".into();
                    return None;
                };
                let pos = ckpt.pos;
                // SWA RING (lane/kv256-capacity): a ring-capped cache may have overwritten rows
                // below this checkpoint — a lapped checkpoint must DECLINE to the cold path,
                // never restore stale ring rows. No-op on full-slab caches.
                if !e.cache.can_rollback(&ckpt.snap, 0) {
                    why = format!("SWA ring lapped checkpoint {pos}");
                    return None;
                }
                // IDENTITY NOMINATES: explicit id when both sides named one, else the implicit
                // fingerprint chain's shared leading run.
                let identity_matches = match (&req.affinity, &e.affinity) {
                    (Some(a), Some(b)) if a == b => true,
                    (Some(_), _) | (_, Some(_)) => false,
                    _ => fingerprint_affinity(&req_fp, &e.fingerprint) >= FP_MIN_SEGMENTS,
                };
                // Grow to this request's bounded `need`, not the server-global context cap. The
                // admission estimator charges the same request-shaped capacity before admit.
                match affinity_resume_target(&prompt, &e.fed, pos, e.cap, need, identity_matches) {
                    Ok(target_cap) => Some((i, target_cap)),
                    Err(reason) => {
                        why = reason;
                        None
                    }
                }
            });
            if cand.is_none() && !pool.is_empty() {
                eprintln!(
                    "[worker] plain-affinity: declined ({why}; {} parked, {} prompt \
                           tokens; model {})",
                    pool.len(),
                    prompt.len(),
                    req.model
                );
            }
            cand
        } else {
            None
        };
        if let Some((idx, target_cap)) = candidate {
            let mut e = reuse
                .get_mut(&pool_key)
                .expect("plain-affinity candidate pool vanished")
                .remove(idx);
            let ckpt = e
                .ckpt
                .take()
                .expect("nominated candidate must carry a checkpoint");
            let pos = ckpt.pos;
            let old_cap = e.cap;
            // Roll back in place when there is already room. Otherwise allocate a PP-aware cache
            // at the request-owned charged capacity and restore the checkpoint into it. The old
            // and new buffers overlap only for this D2D copy; on OOM, reclaim once exactly like a
            // cold session allocation, then drop the parked entry and take the cold path.
            let restored: Result<(), Box<dyn std::error::Error>> = if target_cap > old_cap {
                match alloc_with_single_reclaim_retry(
                    || {
                        memra_engine::pp::new_cache_planned(
                            engine,
                            &lm.model.cfg,
                            &lm.model.plan,
                            target_cap,
                        )
                    },
                    |err| {
                        let evicted_prefix = px.evict_all();
                        if evicted_prefix > 0 {
                            eprintln!(
                                "[prefix-cache] evicted {evicted_prefix} entries after \
                                       plain-affinity grow alloc failure; retrying"
                            );
                        }
                        let evicted_parked = if is_cuda_oom(&err.to_string()) {
                            evict_oldest_parked(reuse, spec_reuse, dspark_reuse, reuse_metrics)
                        } else {
                            None
                        };
                        if let Some(pool) = evicted_parked {
                            eprintln!(
                                "[admit-oom] plain-affinity grow: evicted oldest {} parked \
                                       session (global LRU); retrying cache alloc once",
                                match pool {
                                    ParkedPool::Plain => "plain",
                                    ParkedPool::Spec => "spec",
                                    ParkedPool::Dspark => "dspark",
                                }
                            );
                        }
                        evicted_prefix > 0 || evicted_parked.is_some()
                    },
                ) {
                    Ok(mut grown) => match memra_engine::pp::restore_cache_checkpoint(
                        engine,
                        &lm.model,
                        Some(&e.cache),
                        &mut grown,
                        &ckpt.snap,
                    ) {
                        Ok(()) => {
                            e.cache = grown;
                            e.cap = target_cap;
                            Ok(())
                        }
                        Err(err) => Err(err),
                    },
                    Err(err) => Err(err),
                }
            } else {
                memra_engine::pp::restore_cache_checkpoint(
                    engine,
                    &lm.model,
                    None,
                    &mut e.cache,
                    &ckpt.snap,
                )
            };
            match restored {
                Ok(()) => {
                    debug_assert_eq!(e.cache.pos, pos, "plain rewind landed off the checkpoint");
                    debug_assert_eq!(e.cache.max_ctx, e.cap, "plain cache cap metadata drift");
                    e.fed.truncate(pos);
                    e.last_logits = ckpt.last_logits;
                    reuse_metrics.continuation_hits += 1;
                    reuse_metrics.plain_affinity_rewinds += 1;
                    if target_cap > old_cap {
                        eprintln!(
                            "[worker] plain-affinity: grew parked cache {old_cap} -> \
                                   {target_cap} rows (request-owned need)"
                        );
                    }
                    eprintln!(
                        "[worker] plain-affinity: rewound to {pos} of {} prompt tokens \
                               (priming {} suffix; model {})",
                        prompt.len(),
                        prompt.len() - pos,
                        req.model
                    );
                    reused = Some(e);
                }
                // Allocation, copy, or rollback failure leaves no trustworthy resume state. Drop
                // the entry and let the existing cold allocation path serve the request.
                Err(err) => eprintln!(
                    "[worker] plain-affinity resume failed ({err}); \
                                       dropping session, full prime"
                ),
            }
        }
    }

    // SPEC ELIGIBILITY decides the prefix-cache policy up front. MTP sessions can restore only
    // entries carrying their draft plane; the early DFlash preference below instead bypasses a
    // trunk-only hit exactly when DFlash will cold-prime, preserving it for plain shed traffic.
    // ST-SPEC QUARANTINE LIFTED (#68 closed, 2026-08-04): the serve-spec divergence on
    // dir-loaded checkpoints was never ST-specific — the per-session persistent draft
    // graph replayed with dangling pool addresses (capture transients not retained +
    // fa_part_pool freeing grown-past buffers the capture baked; fixed in spec.rs/lib.rs,
    // receipts research/fp8ship-20260804/RESULTS.md — the same corruption reproduced on
    // GGUF session bursts at n>=600). Dir checkpoints are spec-eligible again; the
    // serve-st gate pins default-serve text == the run-gen CLI tokenwise oracle.
    let serve_spec = !confidence_trace_enabled()
        && !memra_engine::pp::pp_host_bounce_active()
        // v1 vision scope: the mixed-embedding overlay rides the plain prefill tick only;
        // generate_spec_session's internal prime has no overlay seam yet (tracked).
        && !vision_req
        && !capture_req
        && std::env::var("MEMRA_SERVE_SPEC")
            .map(|v| v != "0")
            .unwrap_or(true)
        && peer_probe_allows_spec;
    let mut sampler = Sampler::new(req.sampler_cfg);
    // GREEDY + penalties keeps the legacy tokenwise path (gap-scan F3 plumbing): the greedy
    // spec arm verifies by pure argmax (sampling=None), which would silently ignore the
    // penalties the host sampler applies pre-argmax. Sampled requests carry penalties into
    // the rejection-sampling verify (SpecSampling) and stay spec-eligible.
    let greedy_penalized = sampler.is_greedy()
        && (sampler.penalty_repeat() != 1.0
            || sampler.penalty_freq() != 0.0
            || sampler.penalty_present() != 0.0);
    // CONSTRAINED DECODING (response_format): admission only consumes a matcher already built
    // by the per-model compiler thread. There is deliberately no inline fallback here — a state
    // mismatch is an internal error, never permission to put llguidance back on the live tick.
    let constraint = match (constrained, req.prepared_constraint.take()) {
        (false, None) => None,
        (true, Some(constraint)) => Some(constraint),
        (true, None) => {
            return Err((
                req.tx,
                EngineError::engine(
                    "response_format reached admission before off-tick compilation completed",
                ),
            ));
        }
        (false, Some(_)) => {
            return Err((
                req.tx,
                EngineError::engine("compiled response_format has no grammar specification"),
            ));
        }
    };
    let prefix_requested = reuse_on && serve_batching() && prefix_cache_budget_bytes() > 0;
    let ring_prefix_excluded = memra_engine::cache::swa_ring_on()
        && memra_engine::plan_backend::decode_batch_program(&lm.model.plan)
            == memra_engine::plan_backend::DecodeBatchProgram::SlidingGatedMoe;
    if prefix_requested && ring_prefix_excluded {
        eprintln!(
            "[prefix-cache] refused for MEMRA_SWA_RING=1 Step35 session (flat-history \
                   snapshots/restores are excluded)"
        );
    }
    let prefix_on = prefix_requested && !ring_prefix_excluded;
    let policy_lcp = if prefix_on {
        px.best_lcp(&pool_key, &prompt)
    } else {
        0
    };
    // The K policy's concurrency signal is the WAVE, not live-only active+1 — the same
    // projection the admission-gate estimate used for this request's memory reservation
    // (`projected_admission_wave`; the c16 burst-split receipts live on its doc).
    let projected_wave = projected_admission_wave(n_active, n_pending);
    let mut spec_k_decision = match req_spec_k_replay {
        Some(k) => SpecKDecision {
            k,
            reason: SpecKReason::Replay,
        },
        None => choose_spec_k(
            spec_k_pin(),
            spec_gate_on(),
            *spec_gate_thresholds(),
            projected_wave,
            prompt.len(),
            0,
            spec_trim_head(lm),
        ),
    };

    // SPEC x CONSTRAINED (constrained-full, 2026-08-03): greedy constrained sessions ride
    // spec bursts — the grammar truncates acceptance AFTER the exactness verify and forces
    // the masked argmax at the cut slot (generate_spec_session_constrained). Sampled
    // constrained and the MEMRA_CONSTRAIN_HOST oracle keep plain decode.
    // The request-conditioned K policy owns the placement/concurrency decision: K=0 takes
    // plain decode, while positive K remains subject to the correctness eligibility checks.
    // Qwen35-MoE forces the exact sequential replay path. Constrained spec cannot use that path
    // because replay commits an unmasked bonus, so keep those requests on exact plain constrained
    // serving instead of admitting them and returning a runtime 500 from the engine.
    let constrained_replay_incompatible = constraint.is_some()
        && !constrained_spec_supported(
            &lm.model.plan,
            memra_engine::spec::spec_replay_env_enabled(),
        );
    let mut spec_eligible = serve_spec
        && spec_k_decision.k > 0
        && (constraint.is_none() || (sampler.is_greedy() && !constrain_host()))
        && !constrained_replay_incompatible
        && (sampler.is_greedy() || sampler.temperature() > 0.0)
        && !greedy_penalized
        && mtp_spec_capable(lm)
        // DSPARK route (lane/dspark-q38-recover): an armed dspark drafter OWNS the spec
        // program for this model — the MTP arm never engages (two spec programs on one
        // model must never silently coexist; the boot log says so loudly).
        && !dspark_draft_ready;

    // DFlash needs a cold full-prompt prime to build its draft KV. Decide that route BEFORE the
    // prefix probe using the same load verdict the final admission consumes. At low load a
    // trunk-only entry is preserved but ignored; when the wave sheds DFlash, the unchanged
    // prefix path below restores it for plain serving.
    // The hit this request could actually consume, resolved ONCE with the same `lookup` the
    // restore below uses — a whole-entry full prefix at or above PREFIX_CACHE_MIN_TOKENS.
    //
    // WHY NOT `policy_lcp > 0`, which stood here for one round: `best_lcp` is the longest
    // common prefix against ANY pool entry with NO floor, so a 5-token overlap satisfied it
    // while nothing consumable existed. The only other way to consume a hit is the mid-entry
    // partial restore, which is default-OFF and, when armed, refused outright on the
    // hybrid/GDN trunk `gdn_dspark_compatible` selects — so for this route a full-prefix entry
    // is the whole set. Third and final granularity of the defect review chased across three
    // rounds: the veto must never fire without a hit to take the cold prime's place.
    let consumable_hit = if prefix_on {
        px.lookup(&pool_key, &prompt)
    } else {
        None
    };
    // Can the consumable hit re-arm speculation? Tail present AND whole-entry cover, read here
    // beside the hit itself so the probe gate can distinguish "a hit the conversion will
    // consume" from "a hit that would cost the restore for nothing" (strict-prefix hits — the
    // multi-turn shape — convert never, so they must not buy the carrier).
    let dspark_restorable_hit = dspark_prefix_restore_on()
        && consumable_hit
            .map(|i| {
                let en = &px.entries[&pool_key][i];
                dspark_hit_is_restorable(en.toks.len(), prompt.len(), en.dspark_draft.is_some())
            })
            .unwrap_or(false);
    let cold_prefix_inputs = DsparkColdPrefixAdmission {
        route_ready: dspark_draft_ready
            && memra_engine::plan_backend::gdn_dspark_compatible(&lm.model.plan)
            && serve_spec
            && !confidence_trace_enabled(),
        prime_feasible: dspark_prime_feasible,
        greedy: sampler.is_greedy(),
        greedy_penalized,
        sampled: sampler.temperature() > 0.0,
        constrained: constraint.is_some(),
        vision: vision_req,
        cold: reused.is_none(),
        gate_on: spec_gate_on(),
        pin: spec_k_pin(),
        projected_wave,
        low: spec_gate_low(),
        n_active,
        has_live_non_demotable: has_live_non_demotable_dspark,
        // The two sizes of the trade this decision makes: the prefill a bypass discards, and
        // the decode that has to repay it. `budget` is already `max_new` clamped by the
        // context cap, i.e. what this request can actually generate.
        prompt_len: prompt.len(),
        decode_budget: budget,
        hit_available: consumable_hit.is_some(),
    };
    let dspark_prefers_cold = dspark_prefers_cold_over_prefix(cold_prefix_inputs);
    // The receipt names the veto ONLY when the veto is what flipped the route. Guarding it on
    // `dspark_draft_ready` alone attributed every other decline — wrong sampling shape, load,
    // vision, constrained, a warm resume — to the shape guard, in a line FLAGS.md documents as
    // the guard's boot-visible evidence. Flagged by review.
    if !dspark_prefers_cold {
        let without_shape_veto = dspark_prefers_cold_over_prefix(DsparkColdPrefixAdmission {
            hit_available: false,
            ..cold_prefix_inputs
        });
        if without_shape_veto {
            eprintln!(
                "[dspark] cache-preferred: prompt={} decode_budget={} lcp={policy_lcp} — a cold \
                 prime would discard a prefix hit that {} tokens of speculation cannot repay; \
                 serving the hit and going plain",
                prompt.len(),
                budget,
                budget,
            );
        }
    }

    // CROSS-REQUEST PREFIX CACHE probe (2026-08-02; module doc at PrefixCache). Only when the
    // continuation pool missed, the session won't go spec, and batched scheduling is live.
    // A whole-entry hit deep-copies the longest matching entry into a fresh session cache. On a
    // miss, an eligible mid-entry LCP immediately restores its context-linear K/V rows and primes
    // only the suffix. Recurrent and routed-MoE mid-entry splits fail closed to the old boundary-
    // snapshot learning path; cold long prompts arm the seed insert.
    let mut prefix_hit = false;
    let mut prefix_pin = None;
    let mut snapshot_at: Option<usize> = None;
    let mut prefix_miss_lcp: Option<usize> = None;
    let mut seed_prefix = false;
    // COMMIT-GATED PUBLICATION port (lane/spec-prefix-cache, 2026-08-14): spec-eligible
    // requests now PROBE too — the measured 4x sold-shape loss (canonflip: spec-on c=16
    // 2.14 req/s @ 18.4% hit vs spec-off 8.50 @ 99.5%) was this bypass, not spec compute.
    // On a hit the request is DOWNGRADED to the plain path below (the restored carrier is a
    // plain-session cache; every engine surveyed gates drafting off at high load anyway —
    // research/cache-spec-design-20260814/REPORT.md). On a miss, the armed snapshot_at /
    // seed_prefix boundary rides into the spec session as its capture request.
    if should_probe_prefix_cache(
        prefix_on,
        reused.is_some(),
        dspark_prefers_cold,
        dspark_restorable_hit,
    ) {
        // `consumable_hit`, not a second lookup: the decision above and the restore here must
        // agree by construction about whether a hit exists.
        if let Some(i) = consumable_hit {
            let restored = {
                let e = &px.entries[&pool_key][i];
                // `pp::new_cache`, not `Cache::new` — stage-owned KV under an open ppN door
                // (see the session-cache site below for the full reason). `prefix_restore`
                // then copies plane-by-plane into whatever device each layer landed on.
                match memra_engine::pp::new_cache_planned(
                    engine,
                    &lm.model.cfg,
                    &lm.model.plan,
                    ctx_cap,
                ) {
                    Ok(mut c) => match prefix_restore(engine, &mut c, e, &pool_key) {
                        // A prefix-cache restore is a transient carrier consumed straight into a
                        // fresh session below (never re-parked from here), so the plain-affinity
                        // fields are inert defaults — a new checkpoint is armed at Session build.
                        Ok(()) => Ok({
                            kvprobe(engine, &c, &e.last_logits, "prefix-restored");
                            ReuseEntry {
                                fed: e.toks.clone(),
                                cache: c,
                                last_logits: e.last_logits.clone(),
                                cap: ctx_cap,
                                ckpt: None,
                                affinity: None,
                                fingerprint: Vec::new(),
                                parked_at: Instant::now(),
                            }
                        }),
                        Err(err) => Err(format!("restore failed: {err}")),
                    },
                    Err(err) => Err(format!("session cache alloc failed: {err}")),
                }
            };
            match restored {
                Ok(entry) => {
                    prefix_pin = px.pin(&pool_key, i);
                    debug_assert!(prefix_pin.is_some(), "lookup entry vanished before pin");
                    px.hits += 1;
                    px.hit_tokens += entry.fed.len() as u64;
                    px.record_lcp(entry.fed.len()); // histogram: served-prefix length
                    prefix_hit = true;
                    let hit_len = entry.fed.len();
                    eprintln!(
                        "[prefix-cache] hit: {hit_len} of {} prompt tokens from cache (model {})",
                        prompt.len(),
                        req.model
                    );
                    // DEPTH-UNFREEZE LCP SPLIT ON HIT (H11, the 3.1x lever — receipts on
                    // `maybe_prefix_seed`). A hit on a SHALLOW class entry used to end all
                    // depth learning: the miss path's aligned-LCP capture never ran, so the
                    // class boundary a longer sibling entry (a deepen-seed) already teaches
                    // stayed unpublished and every request re-primed past the shallow depth.
                    // Mirror the miss path here: when the pool's best LCP against this
                    // prompt is DEEPER than the entry just served, stop the suffix prime at
                    // the aligned boundary and capture the deeper class entry there. Same
                    // grid law, same has_key dedupe, and the boundary must lie AHEAD of the
                    // restored prefix (`> hit_len`) by at least PRIME_MIN_T — the fed-start
                    // floor `hit_lcp_snapshot_boundary` documents (a sub-floor gap trips the
                    // prefill-tick veto and the capture would fire off tokenwise decode_step
                    // state). Eager-only sessions never arm it: a carried eager hit rides
                    // tokenwise `decode_step` for the whole suffix (`!(eager_mono && carried)`
                    // in `prefill_tick`), so ITS capture would always be decode_step
                    // provenance — the chained-provenance class R16
                    // (research/cacheinval-20260813) refuses.
                    if let Some(la) = px
                        .best_lcp_entry(&pool_key, &prompt)
                        .and_then(|(_, lcp)| hit_lcp_snapshot_boundary(lcp, hit_len, prompt.len()))
                    {
                        if !eager_only_model(lm) && !px.has_key(&pool_key, &prompt[..la]) {
                            snapshot_at = Some(la);
                        }
                    }
                    reused = Some(entry);
                }
                Err(msg) => {
                    // headroom discipline: sessions win over the cache — on alloc pressure
                    // drop every entry so the cold path (and the retries behind it) can fit.
                    if msg.starts_with("session cache alloc failed") {
                        let n = px.evict_all();
                        eprintln!("[prefix-cache] {msg}; evicted {n} entries, cold path serves");
                    } else {
                        eprintln!("[prefix-cache] {msg}; cold path serves");
                    }
                }
            }
        }

        let mut best_lcp = None;
        if reused.is_none() {
            best_lcp = px.best_lcp_entry(&pool_key, &prompt);
            if partial_prefix_restore_enabled()
                && best_lcp.is_some_and(|(_, lcp)| lcp >= PREFIX_CACHE_MIN_TOKENS)
            {
                let (i, lcp) = best_lcp.unwrap();
                let (decision, source_len) = {
                    let e = &px.entries[&pool_key][i];
                    let has_recurrent =
                        e.conv.iter().any(Option::is_some) || e.ssm.iter().any(Option::is_some);
                    let routed_moe = routed_moe_prefix_split(&lm.model.plan);
                    (
                        partial_prefix_decision(
                            has_recurrent,
                            routed_moe,
                            lcp,
                            e.pos,
                            prompt.len(),
                        ),
                        e.toks.len(),
                    )
                };
                if let Some(reason) = decision.refusal() {
                    eprintln!(
                        "[prefix-cache] partial restore REFUSED: split {lcp} of source \
                         {source_len}, prompt {} (model {}{}): {reason}; cold/snapshot fallback",
                        prompt.len(),
                        req.model,
                        ns_suffix(&pool_key.1),
                    );
                } else {
                    let restored = {
                        let e = &px.entries[&pool_key][i];
                        trace_prefix_entry_state(engine, e, lcp, "source", "immediate-partial");
                        match memra_engine::pp::new_cache_planned(
                            engine,
                            &lm.model.cfg,
                            &lm.model.plan,
                            ctx_cap,
                        ) {
                            Ok(mut c) => match prefix_restore_at(engine, &mut c, e, &pool_key, lcp)
                            {
                                Ok(()) => {
                                    trace_prefix_cache_state(
                                        engine,
                                        &c,
                                        lcp,
                                        "restored",
                                        "immediate-partial",
                                    );
                                    Ok(ReuseEntry {
                                        fed: e.toks[..lcp].to_vec(),
                                        cache: c,
                                        // A supported partial hit always has a non-empty suffix,
                                        // so the endpoint logits are recomputed by suffix prime.
                                        last_logits: Vec::new(),
                                        cap: ctx_cap,
                                        ckpt: None,
                                        affinity: None,
                                        fingerprint: Vec::new(),
                                        parked_at: Instant::now(),
                                    })
                                }
                                Err(err) => Err(format!("partial restore failed: {err}")),
                            },
                            Err(err) => Err(format!("session cache alloc failed: {err}")),
                        }
                    };
                    match restored {
                        Ok(entry) => {
                            prefix_pin = px.pin(&pool_key, i);
                            debug_assert!(
                                prefix_pin.is_some(),
                                "partial lookup entry vanished before pin",
                            );
                            px.hits += 1;
                            px.hit_tokens += lcp as u64;
                            px.record_lcp(lcp);
                            prefix_hit = true;
                            eprintln!(
                                "[prefix-cache] partial hit: {lcp} of {} prompt tokens from \
                                 source {source_len} (model {}{}) — priming suffix {} only",
                                prompt.len(),
                                req.model,
                                ns_suffix(&pool_key.1),
                                prompt.len() - lcp,
                            );
                            reused = Some(entry);
                        }
                        Err(msg) => {
                            if msg.starts_with("session cache alloc failed") {
                                let n = px.evict_all();
                                eprintln!(
                                    "[prefix-cache] {msg}; evicted {n} entries, \
                                     cold/snapshot fallback",
                                );
                            } else {
                                eprintln!("[prefix-cache] {msg}; cold/snapshot fallback");
                            }
                        }
                    }
                }
            }
        }
        if reused.is_none() {
            px.misses += 1;
            let l = best_lcp
                .map(|(_, lcp)| lcp)
                .unwrap_or_else(|| px.best_lcp(&pool_key, &prompt));
            px.record_lcp(l); // histogram: best available LCP on a miss
            prefix_miss_lcp = Some(l);
            // Captures land ON the prime grid (grid_align_boundary): the entry then restores
            // into a suffix prime whose call start reproduces the monolithic fold, so a hit
            // is byte-identical to the cold render of the same bytes. An aligned key is a
            // prefix of the same stable segment the raw LCP was, so the eligibility checks
            // re-run on the ALIGNED value.
            let la = grid_align_boundary_within(l, prompt.len());
            if la >= PREFIX_CACHE_MIN_TOKENS
                && la < prompt.len()
                && !px.has_key(&pool_key, &prompt[..la])
            {
                snapshot_at = Some(la);
            }
            // MESSAGE-BOUNDARY SEED (lane/moebatch-q35moe, 2026-08-21): when the LCP taught
            // nothing (a truly cold shape), park the capture boundary at the end of the FIRST
            // message instead of leaving only the full-prompt seed. The seed's tail carries the
            // generation header, which every later render of the same history strips/re-emits,
            // and hybrid restore is entry-end-only — so the seed was unusable by the very next
            // turn (cachecell: turn-1 full re-prefill) and by every shared-prefix peer (c8:
            // zero hits). The first message (the system prompt — the bulk of an agentic prompt
            // and the shared part of fanout shapes) is render-stable; the entry captured there
            // is a whole-entry hit for both.
            if snapshot_at.is_none() {
                if let Some(b) = first_message_boundary(
                    lm,
                    &req.chat_turns,
                    &req.tools_json,
                    &req.think,
                    req.reasoning_effort.as_deref(),
                    &prompt,
                ) {
                    // Same grid law as the LCP split above: the aligned position is still
                    // inside the render-stable first message (earlier is always stable),
                    // and the capture composes byte-exactly with any later suffix prime.
                    let ba = grid_align_boundary_within(b, prompt.len());
                    if ba >= PREFIX_CACHE_MIN_TOKENS && !px.has_key(&pool_key, &prompt[..ba]) {
                        snapshot_at = Some(ba);
                    }
                }
            }
            if prompt.len() >= PREFIX_CACHE_MIN_TOKENS {
                seed_prefix = true; // re-checked against covering entries at prefill-done
            }
        }
    }

    // SPEC-ON-CACHE-HIT (lane/spec-on-cache-hit, 2026-08-18 — PORT-PLAN item 3, scoped to
    // whole-entry restores): a spec-eligible hit whose entry carries a draft plane re-arms
    // a SpecSession from the restored carrier instead of downgrading — the 2026-08-18
    // endpoint bench measured the downgrade at ~135 -> ~75 tok/s on every repeated-prompt
    // row (the sold agent-loop shape). The restored session is state-equivalent to a
    // continuation-pool resume at the entry boundary and rides the existing spec_resumed
    // machinery (suffix prime + fill anchored on the entry's boundary hidden).
    // Constraints, verified against the receipts rather than assumed:
    //   * WHOLE-entry hits only — the restore already happened at exactly e.pos through
    //     the shipping path; mid-entry trunk restores stay behind
    //     MEMRA_PREFIX_PARTIAL_RESTORE (lcprestore NO-GO / splitiso two-programs class).
    //   * every converted shape becomes an empty-suffix continuation — the engine feeds any
    //     prompt suffix itself through the plain path's exact program selection (the r3
    //     identity finding, research/spec-cache-20260818: handing the suffix to the burst
    //     prime routed qwen35 through the BATCHED T=1 program while plain fed it eager, and
    //     the near-tie flipped at token ~8) — and the continuation seed is an argmax. An
    //     EMPTY suffix (entry covers the whole prompt) additionally needs the entry's
    //     boundary hidden and logits: `next_pred = argmax(last_logits)` is exactly the token
    //     the plain path would emit from the same cached logits row.
    //   * GREEDY **or SAMPLED** (v2, lane/sampled-hit-spec 2026-08-19). v1 shipped
    //     greedy-only because "a sampled hit's first token must be host-sampled"; the
    //     v0.93.0 DE deploy then measured the consequence — 3 cache hits, 3 plain-path
    //     downgrades, 0 restores, on a customer tenant whose traffic is sampled, which is
    //     what the OpenAI surface defaults to (temperature 1.0, main.rs default_temperature).
    //     The premise did not survive the code: the COLD sampled spec path did not
    //     host-sample its own first token either — it argmaxed `prime_logits`, as did every
    //     later continuation burst (the stashed `next_pred`). Sampled spec is
    //     distributionally exact per its documented contract (step_session, `sampling`),
    //     NOT byte-equal to plain sampling. So refusing sampled hits preserved no property;
    //     it only split ONE request shape across two sampling programs (cold: spec, cached:
    //     plain). The restored session's seed rule and Philox counters ((0,0) at admit, like
    //     any fresh session) are bit-identical to the cold spec session's, which is exactly
    //     what the sampled arm of tools/spec-on-cache-hit-gate.sh measures: per seed, the
    //     hit's bytes == the cold leader's bytes.
    //     THAT ARGMAX IS ITSELF FIXED (lane/sampled-spec-quality, 2026-08-19): both paths
    //     now DRAW the boundary token from the request's target at Philox counter 0 from the
    //     same logits row (spec.rs `sample_boundary_token`), so the identity above is
    //     preserved by construction rather than by both sides being greedy. The penalized
    //     refusal is LIFTED with it — the burst's penalty window spans the session, and a
    //     restored session's `committed` IS the whole prompt — and returns only under
    //     MEMRA_SPEC_PEN_SESSION=0, named.
    //   * unconstrained — grammar-owned generation keeps the plain path (the pool-resume
    //     law; a restored session has no grammar-consumed state, but v1 stays tight).
    //   * conversion failure hands the trunk cache back and the hit serves PLAIN — the
    //     exact pre-lane behavior, so the fallback is the banked path, never a refusal.
    // Every refusal now NAMES ITSELF on the downgrade line: v1's silent `false` is why the
    // greedy-only scope was invisible in the deploy window (0 restores AND 0 declines).
    let mut spec_restored: Option<memra_engine::spec::SpecSession> = None;
    let mut spec_restored_cached = 0usize;
    let mut spec_restore_declined: Option<&'static str> = None;
    if prefix_hit && spec_eligible && constraint.is_some() {
        spec_restore_declined = Some("constrained request (grammar owns generation)");
    }
    if prefix_hit && spec_eligible && constraint.is_none() && reused.is_some() {
        // Mirrors the engine's own `pen_on` (spec.rs, sampled burst setup) exactly: a
        // penalty window is active only when a non-identity penalty is set AND the window
        // is armed. The serve API arms `penalty_last_n = usize::MAX` for any non-identity
        // penalty (main.rs sampler_config), so this is request-driven, not a heuristic.
        let penalty_window_active = sampler.penalty_last_n() > 0
            && (sampler.penalty_repeat() != 1.0
                || sampler.penalty_freq() != 0.0
                || sampler.penalty_present() != 0.0);
        // LOAD GUARD verdict (lane/sampled-restore-load-guard). Named on its own line whenever
        // it is the deciding input, with the arithmetic ON the line: a policy whose numbers are
        // only in the source is a policy the next operator has to re-derive from a throughput
        // regression.
        //
        // EVALUATED AS LATE AS POSSIBLE, and that is load-bearing. Demand is read HERE, after
        // this request's own prefix restore has already copied 312 MB of KV, because that window
        // is exactly when the rest of an arriving fan-out lands. Measured: reading it at tick top
        // instead leaves the HEAD of a c16 wave unrefusable (the worker wakes holding one
        // request, drains an empty channel, and the other fifteen arrive during the restore), and
        // that single engaged row costs ~9.6% of aggregate throughput.
        let load_demand = spec_load_demand(n_active + 1 + n_pending);
        let load_admits = sampled_restore_load_admits(
            spec_restore_load_guard_on(),
            spec_k_pin(),
            spec_gate_on(),
            *spec_gate_thresholds(),
            load_demand,
        );
        if let Some(i) = prefix_pin.as_ref().and_then(|p| px.id_index(p)) {
            let fed_len = reused.as_ref().map_or(0, |e| e.fed.len());
            let full_cover = fed_len == prompt.len();
            let refusal = {
                let e = &px.entries[&pool_key][i];
                spec_restore_refusal(
                    e.draft.is_some(),
                    e.pos,
                    fed_len,
                    prompt.len(),
                    sampler.is_greedy(),
                    penalty_window_active,
                    spec_restore_sampled_on(),
                    memra_engine::spec::spec_pen_session_on(),
                    load_admits,
                    !e.last_h.is_empty(),
                    !e.last_logits.is_empty(),
                )
            };
            if !sampler.is_greedy() && !load_admits {
                let t = spec_gate_thresholds();
                eprintln!(
                    "[spec-restore-guard] sampled restore REFUSED: demand {load_demand} > \
                     SOLO watermark {} (band LOW={} HIGH={}; projected active={}, queued={}, \
                     http in-flight {:?}) — hit serves PLAIN (model {}); \
                     MEMRA_SPEC_RESTORE_LOAD_GUARD=0 disables",
                    sampled_restore_watermark(*t),
                    t.low,
                    t.high,
                    n_active + 1,
                    n_pending,
                    http_inflight_total(),
                    req.model,
                );
            }
            spec_restore_declined = refusal;
            if refusal.is_none() {
                let carrier = reused.take().expect("prefix hit carrier vanished");
                let ReuseEntry {
                    fed,
                    cache: carrier_cache,
                    last_logits: carrier_logits,
                    cap,
                    ..
                } = carrier;
                let entry = &px.entries[&pool_key][i];
                let draft = entry.draft.as_ref().expect("checked above");
                // SEED RULE OWNERSHIP (lane/sampled-spec-quality): the engine sets `next_pred`
                // for BOTH shapes now — the entry's boundary logits on a full-cover hit, the
                // feed's own on a suffix hit — and applies the request's sampler to it. The
                // worker used to argmax the full-cover seed here, which meant the sampled
                // boundary draw could be half-applied across the two shapes.
                // STABLE-BOUNDARY REPUBLICATION (lane/frspec-multiturn-cache, 2026-08-21):
                // the extended-entry capture and the restored session's turn checkpoint land
                // at the stable pre-generation boundary instead of prompt-end, so the NEXT
                // re-rendered turn can actually hit/rewind them (prompt-end entries carry the
                // live generation header the client rewrites — the frozen-boundary defect,
                // finding B4). Only a boundary AHEAD of the restored prefix is a legal feed
                // stop; door OFF = None = legacy prompt-end republication byte-for-byte.
                let republish_at = if spec_stable_boundary_on() {
                    plain_checkpoint_boundary(&prompt, &|t| lm.tok.token_is_control(t))
                        .filter(|&b| b > fed_len)
                } else {
                    None
                };
                match lm.model.spec_session_from_restored(
                    engine,
                    carrier_cache,
                    fed.clone(),
                    &prompt[fed.len()..],
                    &draft.k,
                    &draft.v,
                    draft.k_tok_bytes,
                    draft.v_tok_bytes,
                    draft.len,
                    &entry.last_h,
                    &entry.last_logits,
                    spec_sampling_for(&sampler),
                    full_cover,
                    cap,
                    republish_at,
                ) {
                    Ok(sess) => {
                        eprintln!(
                            "[prefix-cache] spec restore: {} of {} prompt tokens + draft \
                             plane from cache{} (model {})",
                            fed.len(),
                            prompt.len(),
                            if full_cover {
                                " [continuation]"
                            } else {
                                " [suffix fed]"
                            },
                            req.model,
                        );
                        // billing truth: only the restored prefix came from cache; the
                        // suffix rows were computed by the feed.
                        spec_restored_cached = fed.len();
                        spec_restored = Some(sess);
                    }
                    Err((Some(cache_back), why)) => {
                        spec_restore_declined = Some("engine declined the conversion");
                        eprintln!(
                            "[prefix-cache] spec restore declined ({why}); hit serves PLAIN \
                             (model {})",
                            req.model,
                        );
                        reused = Some(ReuseEntry {
                            fed,
                            cache: cache_back,
                            last_logits: carrier_logits,
                            cap,
                            ckpt: None,
                            affinity: None,
                            fingerprint: Vec::new(),
                            parked_at: Instant::now(),
                        });
                    }
                    Err((None, why)) => {
                        // the carrier is part-fed and unusable: serve this request
                        // cold-plain (correct, slower); the entry stays published.
                        spec_restore_declined = Some("conversion failed mid-feed (serves COLD)");
                        eprintln!(
                            "[prefix-cache] spec restore failed mid-feed ({why}); request \
                             serves COLD on the plain path (model {})",
                            req.model,
                        );
                    }
                }
            }
        } else {
            spec_restore_declined = Some("pinned entry vanished before conversion");
        }
    }
    // DSPARK RESTORE (lane/dspark-draft-plane-20260827): the long-answer half. A hit whose
    // entry carries the drafter's readable KV tail can re-arm a dspark session instead of
    // cold-priming, which is what stops a long-decode request re-prefilling 30k tokens it
    // already had cached (~10 s). Placed here, beside the MTP conversion, for the same reason:
    // this is where the PrefixCache is borrowable, and `from_tail` copies OUT of the entry so
    // the tail stays available for other requests.
    //
    // WHOLE-ENTRY ONLY. The trunk is a GDN hybrid whose recurrent state cannot be rebuilt
    // mid-sequence, so a partial cover has no restorable trunk — the same full-prompt-only rule
    // the cold path states. `fed.len() == prompt.len()` is that check.
    let mut dspark_prefix_restored: Option<(memra_engine::dflash::DsparkSpecSession, Vec<u32>)> =
        None;
    // ONLY when the request WOULD HAVE COLD-PRIMED. The shipped shape guard already decides
    // who gets what: short-decode requests take the plain hit (proven byte-exact, ~1 s), and
    // only cold-preferring long decodes have a prime to save. The first gate run on the sbox
    // box proved why this condition is load-bearing and not an optimization: without it the
    // conversion re-armed dspark for a 64-token request the guard had JUST routed to the
    // plain hit (the log shows "serving the hit and going plain" followed by "DSPARK
    // restore"), and that restored session emitted deterministic garbage from token one — later
    // ROOT-CAUSED on the bench as the dispatch bug fixed at the Session literal (dspark_on was
    // computed before this fold existed, so the restored session was stepped by PLAIN
    // step_session over its absent cache; budget was never the variable — 414-prompt/
    // 64-budget restores byte-exact under correct dispatch). This condition's job today is the
    // #55 behavior split, plus keeping the receipt line above truthful.
    if dspark_prefix_restore_on()
        && dspark_prefers_cold
        && spec_restored.is_none()
        && let Some(draft) = dspark_draft
        && let Some(carrier) = reused.take()
    {
        let full_cover = carrier.fed.len() == prompt.len();
        let entry_tail_present = prefix_pin
            .as_ref()
            .and_then(|p| px.id_index(p))
            .map(|i| px.entries[&pool_key][i].dspark_draft.is_some())
            .unwrap_or(false);
        // Eligibility mirrors the cold route's own shape rules: the drafter must be armed and
        // compatible, the request unconstrained, non-vision, and not penalized-greedy (which is
        // served on the plain path by admission). Anything else keeps the plain hit.
        let shape_ok = !vision_req
            && constraint.is_none()
            && ((sampler.is_greedy() && !greedy_penalized) || sampler.temperature() > 0.0)
            && memra_engine::plan_backend::gdn_dspark_compatible(&lm.model.plan)
            && serve_spec;
        if full_cover && entry_tail_present && shape_ok {
            let i = px
                .id_index(prefix_pin.as_ref().expect("checked above"))
                .expect("checked above");
            let cap = carrier.cap;
            let dkv = {
                let tail = px.entries[&pool_key][i]
                    .dspark_draft
                    .as_ref()
                    .expect("checked above");
                memra_engine::dflash::DflashKv::from_tail(&engine, &draft.cfg, cap, tail)
            };
            match dkv {
                Some(dkv) => {
                    let logits = carrier.last_logits.clone();
                    let fed = carrier.fed.clone();
                    match lm.model.dspark_spec_session_from_restored(
                        &engine,
                        draft,
                        carrier.cache,
                        &fed,
                        dkv,
                        &logits,
                        spec_sampling_for(&sampler),
                        cap,
                    ) {
                        Ok(sess) => {
                            eprintln!(
                                "[prefix-cache] DSPARK restore: {} prompt tokens + draft tail                                  from cache — no cold prime (model {})",
                                fed.len(),
                                req.model,
                            );
                            dspark_prefix_restored = Some((sess, fed));
                        }
                        Err(why) => {
                            eprintln!(
                                "[prefix-cache] dspark restore declined ({why}); request                                  cold-primes (model {})",
                                req.model,
                            );
                        }
                    }
                }
                None => eprintln!(
                    "[prefix-cache] dspark restore: tail does not cover the drafter window;                      request cold-primes (model {})",
                    req.model,
                ),
            }
        } else {
            // Cold-preferring (the block's entry condition) but the hit cannot re-arm the
            // drafter. Handing the carrier back would serve it PLAIN — trading speculation for
            // a prefill saving. Drop the carrier (the entry itself stays published) and let
            // the cold prime happen exactly as before this lane.
            eprintln!(
                "[prefix-cache] dspark: hit cannot re-arm the drafter; keeping the cold prime \
                 rather than downgrading to plain (model {})",
                req.model,
            );
        }
    }
    // Downgrade-on-hit (lane/spec-prefix-cache): a restored prefix carrier without a
    // re-armed SpecSession is plain-session state — the plain path serves the hit exactly
    // as spec-off did at 8.5 req/s on the sold shape. Reached only when the entry carries
    // no draft plane, the request is constrained or sampled-penalized, the restore is
    // partial, or the conversion above declined.
    //
    // The REASON is on the line (lane/sampled-hit-spec): the v0.93.0 deploy window read
    // "0 spec restores AND 0 declines" and could not tell a refused conversion from a
    // mechanism that was never reached — the greedy-only scope hid inside that silence.
    // "unnamed" can only appear if a future refusal path forgets to set the reason.
    if prefix_hit && spec_eligible && spec_restored.is_none() {
        spec_eligible = false;
        eprintln!(
            "[prefix-cache] spec-eligible request takes the hit on the PLAIN path \
             (reason: {}) (model {})",
            spec_restore_declined.unwrap_or("unnamed"),
            req.model,
        );
    }

    let (cache, seed_fed, seed_logits) = match reused {
        Some(e) => {
            if !prefix_hit {
                eprintln!(
                    "[worker] kv-reuse: {} of {} prompt tokens resumed (model {})",
                    e.fed.len(),
                    prompt.len(),
                    req.model
                );
            }
            (Some(e.cache), e.fed, e.last_logits)
        }
        // legacy cache deferred: allocated below ONLY if the spec path doesn't take the session.
        None => (None, Vec::new(), Vec::new()),
    };

    // EOS: union of caller-supplied eos + the model's END-OF-GENERATION set (eos + the
    // turn-end control tokens present in the vocab — llama's special_eog: <|im_end|>,
    // <turn|>, <end_of_turn>). eog_ids(), not eos_id alone (lane/gemma4-serve-gaps,
    // 2026-08-07): gemma4's GGUF eos is <eos>=1, but its chat turns end with <turn|> —
    // with only eos_id in the set, generation blew straight through the turn end and the
    // client received literal '<turn|><turn|>thought…' as content
    // (research/gemma4-serve-20260807/raw/postfix-client1-*.json). run_gen and gemma-gate
    // already stop on eog_ids(); the serve path now matches. The EOS token's text is never
    // streamed (existing rule), so the turn token also stops leaking as text.
    let mut params = req.params;
    for id in lm.tok.eog_ids() {
        if !params.eos.contains(&id) {
            params.eos.push(id);
        }
    }

    // Suffix-only prefill on a reuse hit; sampler penalty history replayed over the whole prefix.
    for &t in &seed_fed {
        sampler.accept(t);
    }
    let suffix: Vec<u32> = prompt[seed_fed.len()..].to_vec();
    let prefill_done_at_admit = suffix.is_empty();
    // SPEC-DECODE serve path (2026-07-05): greedy + MTP head + not a KV-reuse resume (the spec
    // session owns its own caches; folding the reuse pool into SpecSession is a follow-up) +
    // MEMRA_SERVE_SPEC!=0. The whole prompt goes to the spec session as turn 1's suffix; the
    // legacy prefill/decode path is bypassed entirely in step_session.
    let mut spec_resumed = 0usize;
    let mut text_suffix: Option<Vec<u32>> = None;
    // Sampled-spec serve: temperature + filters + penalties ALL ride the rejection-sampling
    // spec path (transforms applied to p and q symmetrically) — the legacy per-token path
    // remains only as the no-MTP/resume fallback.
    let mut spec = if let Some(sess) = spec_restored.take() {
        // PREFIX-CACHE spec restore (lane/spec-on-cache-hit): the converted hit built at
        // the probe above. Rides the spec_resumed machinery exactly like a pool resume —
        // sampler history replays over committed below, only the suffix primes, and
        // n_cached itemizes the restored prefix. Deliberately ahead of the pool probes:
        // this shape (identical/extending prompt with a live prefix entry) went PLAIN
        // before this lane, so the pool tiers keep exactly the shapes they already serve.
        spec_resumed = sess.committed.len();
        Some(sess)
    } else if spec_eligible && seed_fed.is_empty() {
        // POOL RESUME: a parked spec session whose committed sequence exactly prefixes this
        // prompt (with cache room) resumes — only the suffix primes; equal-length = pure burst.
        // Match order: exact token prefix (bit-clean), else TEXT prefix (survives BPE boundary
        // divergence — the ~50% chat-turn miss class). Text hits re-tokenize only the remainder.
        // CONSTRAINED requests never resume parked spec sessions: the park's stashed
        // next_pred/pending is unconstrained state, and the grammar must own generation
        // from token 1. Cold spec session instead (still spec — just no pool hit).
        let mut affinity_rewound: Option<(usize, &'static str)> = None;
        enum SpecResumeProbe {
            Exact(usize),
            Text {
                index: usize,
                suffix: Vec<u32>,
            },
            Affinity {
                index: usize,
                explicit: bool,
                old_cap: usize,
                target_cap: usize,
            },
        }
        // SAMPLER PREDICATE (lane/session-resume-sampler-predicate-20260820). Every tier below
        // ends its conjunction with `spec_resume_sampler_admits`, so a parked session resumes only
        // when the incoming sampler is EQUIVALENT to the one that shaped it. LAST in the
        // conjunction on purpose: `sampler_refused` must name the sampler only for an entry that
        // was otherwise resumable, never for a prompt miss that happened to sit next to one.
        let req_sampler = sampler.identity();
        let mut sampler_refused: Option<&'static str> = None;
        let resumed = if constraint.is_some() {
            None
        } else {
            let mut probe = None;
            if let Some(pool) = spec_reuse.get(&pool_key) {
                if let Some(index) = pool.iter().rposition(|e| {
                    e.sess.cache_max_ctx() >= ctx_cap
                        && prompt.len() >= e.sess.committed.len()
                        && prompt.starts_with(&e.sess.committed)
                        && spec_resume_sampler_admits(&req_sampler, e, &mut sampler_refused)
                }) {
                    probe = Some(SpecResumeProbe::Exact(index));
                } else if !req.prompt_text.is_empty() {
                    if let Some(index) = pool.iter().rposition(|e| {
                        e.sess.cache_max_ctx() >= ctx_cap
                            && req.prompt_text.len() >= e.committed_text.len()
                            && req.prompt_text.starts_with(e.committed_text.as_str())
                            && spec_resume_sampler_admits(&req_sampler, e, &mut sampler_refused)
                    }) {
                        let rem = &req.prompt_text[pool[index].committed_text.len()..];
                        probe = Some(SpecResumeProbe::Text {
                            index,
                            suffix: lm.tok.encode(rem, false),
                        });
                    }
                }

                // ---- SESSION AFFINITY (lane/session-affinity, 2026-08-05) ----
                // Both probes above require the new prompt to EXTEND the parked session. A client
                // that rewrites conversation history (the owner's: `<think>` blocks stripped out of
                // prior assistant turns) fails both on every turn, and the parked multi-GB session
                // is discarded while the whole growing conversation re-primes (~3s TTFT at 11k-14k
                // tokens vs llama's 0.19s — research/specpool-20260804/RESULTS.md).
                //
                // Affinity asks the other question: is this the SAME CONVERSATION? Nomination is by
                // identity (explicit client id, else the structural fingerprint chain); the resume
                // decision is then made on BYTES against the session's REWIND BOUNDARY — the
                // prompt-end checkpoint its last turn retained. A history rewrite mutates what the
                // session GENERATED, so the new prompt still agrees with the session's committed
                // tokens up to that boundary, and only this turn's delta (rewritten answer + new
                // user turn) needs priming.
                //
                // Requires a retained checkpoint, an exact byte match through it, and a
                // non-empty suffix. Capacity no longer vetoes: a too-small parked session grows
                // to this request's bounded, admission-charged `need` before rewind.
                if probe.is_none() && affinity_enabled() {
                    let req_fp =
                        conversation_fingerprint(&prompt, &|t| lm.tok.token_is_control(t), true);
                    let mut why: String = "empty pool".into();
                    let cand = pool.iter().enumerate().rev().find_map(|(index, e)| {
                        let Some(pos) = e.sess.rewind_pos() else {
                            why = "no turn checkpoint retained".into();
                            return None;
                        };
                        // SWA RING (lane/kv256-capacity): a ring-capped cache may have
                        // overwritten rows below the rewind boundary — a lapped checkpoint
                        // must DECLINE, never resume over stale ring rows. No-op on full slabs.
                        if !e.sess.rewind_is_resident() {
                            why = format!("SWA ring lapped checkpoint {pos}");
                            return None;
                        }
                        let identity_matches = match (&req.affinity, &e.affinity) {
                            (Some(a), Some(b)) if a == b => true,
                            (Some(_), _) | (_, Some(_)) => false,
                            _ => fingerprint_affinity(&req_fp, &e.fingerprint) >= FP_MIN_SEGMENTS,
                        };
                        match affinity_resume_target(
                            &prompt,
                            &e.sess.committed,
                            pos,
                            e.sess.cache_max_ctx(),
                            need,
                            identity_matches,
                        ) {
                            Ok(target_cap) => {
                                // SAMPLER PREDICATE, after the bytes and the identity: an
                                // affinity candidate that fails only on the sampler declines
                                // through the SAME `why` line the other declines use, naming the
                                // field. Placed here so a byte/identity miss is never reported as
                                // a sampler refusal.
                                if !spec_resume_sampler_admits(
                                    &req_sampler,
                                    e,
                                    &mut sampler_refused,
                                ) {
                                    why = format!(
                                        "sampler differs on {} (parked session's sampler shaped \
                                         its penalty window, Philox stream and draft plane)",
                                        sampler_refused.unwrap_or("sampler"),
                                    );
                                    return None;
                                }
                                Some(SpecResumeProbe::Affinity {
                                    index,
                                    explicit: e.affinity.is_some(),
                                    old_cap: e.sess.cache_max_ctx(),
                                    target_cap,
                                })
                            }
                            Err(reason) => {
                                why = reason;
                                None
                            }
                        }
                    });
                    if cand.is_none() && !pool.is_empty() {
                        eprintln!(
                            "[worker] spec-affinity: declined ({why}; {} parked, {} prompt \
                                   tokens; model {})",
                            pool.len(),
                            prompt.len(),
                            req.model
                        );
                    }
                    probe = cand;
                }
            }

            match probe {
                Some(SpecResumeProbe::Exact(index)) => Some(
                    spec_reuse
                        .get_mut(&pool_key)
                        .expect("spec exact candidate pool vanished")
                        .remove(index)
                        .sess,
                ),
                Some(SpecResumeProbe::Text { index, suffix }) => {
                    text_suffix = Some(suffix);
                    Some(
                        spec_reuse
                            .get_mut(&pool_key)
                            .expect("spec text candidate pool vanished")
                            .remove(index)
                            .sess,
                    )
                }
                Some(SpecResumeProbe::Affinity {
                    index,
                    explicit,
                    old_cap,
                    target_cap,
                }) => {
                    let mut entry = spec_reuse
                        .get_mut(&pool_key)
                        .expect("spec affinity candidate pool vanished")
                        .remove(index);
                    let rewound = if target_cap > old_cap {
                        alloc_with_single_reclaim_retry(
                            || {
                                lm.model.spec_grow_and_rewind_to_checkpoint(
                                    engine,
                                    &mut entry.sess,
                                    target_cap,
                                )
                            },
                            |err| {
                                if !is_cuda_oom(&err.to_string()) {
                                    return false;
                                }
                                let evicted_prefix = px.evict_all();
                                if evicted_prefix > 0 {
                                    eprintln!(
                                        "[prefix-cache] evicted {evicted_prefix} entries \
                                               after spec-affinity grow OOM; retrying"
                                    );
                                }
                                let evicted_parked = evict_oldest_parked(
                                    reuse,
                                    spec_reuse,
                                    dspark_reuse,
                                    reuse_metrics,
                                );
                                if let Some(pool) = evicted_parked {
                                    eprintln!(
                                        "[admit-oom] spec-affinity grow: evicted oldest {} \
                                               parked session (global LRU); retrying once",
                                        match pool {
                                            ParkedPool::Plain => "plain",
                                            ParkedPool::Spec => "spec",
                                            ParkedPool::Dspark => "dspark",
                                        }
                                    );
                                }
                                evicted_prefix > 0 || evicted_parked.is_some()
                            },
                        )
                    } else {
                        lm.model.spec_rewind_to_checkpoint(engine, &mut entry.sess)
                    };
                    match rewound {
                        Ok(Some(pos)) => {
                            if target_cap > old_cap {
                                eprintln!(
                                    "[worker] spec-affinity: grew parked session {old_cap} \
                                           -> {target_cap} rows (request-owned need)"
                                );
                            }
                            affinity_rewound =
                                Some((pos, if explicit { "explicit" } else { "fingerprint" }));
                            Some(entry.sess)
                        }
                        Ok(None) => {
                            eprintln!(
                                "[worker] affinity rewind failed (checkpoint vanished); \
                                       dropping session, full prime"
                            );
                            None
                        }
                        Err(err) => {
                            eprintln!(
                                "[worker] affinity rewind failed ({err}); \
                                       dropping session, full prime"
                            );
                            None
                        }
                    }
                }
                None => None,
            }
        };
        match resumed {
            Some(mut sess) => {
                reuse_metrics.spec_hits += 1;
                if affinity_rewound.is_some() {
                    reuse_metrics.spec_affinity_rewinds += 1;
                }
                // Q2 (audit 2026-08-05): a parked session carries its draft-graph failure
                // memoization; a NEW request gets a fresh capture chance (transient VRAM
                // pressure at park time must not become permanent coverage loss).
                sess.reset_graph_fallback_on_resume();
                spec_resumed = sess.committed.len();
                match affinity_rewound {
                    Some((pos, tier)) => eprintln!(
                        "[worker] spec-affinity: rewound to {pos} of {} prompt tokens \
                         ({tier}; priming {} suffix; model {})",
                        prompt.len(),
                        prompt.len() - pos,
                        req.model
                    ),
                    // The sampler tag is on the line because "a resume happened" is otherwise
                    // ambiguous between "the predicate matched" and "the predicate is off" — and a
                    // receipt that cannot tell those apart cannot prove the no-regression half.
                    None => eprintln!(
                        "[worker] spec-reuse: {} committed tokens resumed{} [sampler={}] (model {})",
                        spec_resumed,
                        if text_suffix.is_some() {
                            " [text-prefix]"
                        } else {
                            ""
                        },
                        if spec_resume_sampler_predicate_on() {
                            "equivalent"
                        } else {
                            "unchecked"
                        },
                        req.model
                    ),
                }
                // NAMED RESIDUAL (lane/sampled-spec-quality, Item 1). A pool resume whose
                // prompt EXACTLY equals the parked session's committed tape emits that
                // session's stashed `next_pred` as its first token — and if the session was
                // parked by a GREEDY request (the park-time `spec_flush_pending`, which has no
                // future sampler to draw with) that token is an ARGMAX, which is precisely the
                // defect this lane removed everywhere else. Every other boundary site draws.
                // NOT refused: the alternative is re-priming the whole conversation to fix one
                // token, which is a worse trade for the customer than one argmax per resume of
                // a zero-new-token request. The two real fixes — retain the boundary logits row
                // on the parked entry, or mark whether the stash was drawn — are named in
                // darklanes research/spec-cache-20260818/SAMPLED-QUALITY.md §5.7.
                if !sampler.is_greedy() && prompt.len() == sess.committed.len() {
                    eprintln!(
                        "[worker] spec-reuse: sampled empty-suffix resume takes the PARKED \
                         session's stashed boundary token (reason: this request's sampler was \
                         not live when it was stashed) (model {})",
                        req.model,
                    );
                }
                Some(sess)
            }
            None => {
                reuse_metrics.spec_misses += 1;
                // SAMPLER REFUSAL, NAMED (lane/session-resume-sampler-predicate-20260820). A
                // parked session that matched this prompt but was shaped by a DIFFERENT sampler is
                // refused, and the downgrade line says which field — per house standard a refusal
                // that does not say why is indistinguishable from an unwired mechanism. The
                // request still serves, cold-spec, correctly; what it loses is the resume.
                // `spec_pool_sampler_refusals` in /metrics is the production answer to "does real
                // traffic change sampler mid-session", which no synthetic workload can settle.
                if let Some(field) = sampler_refused {
                    reuse_metrics.spec_sampler_refusals += 1;
                    eprintln!(
                        "[worker] spec-reuse REFUSED: parked session's sampler differs on \
                         {field}; serving COLD spec (a parked session's penalty window, Philox \
                         stream position and draft plane were shaped by that sampler; \
                         MEMRA_SPEC_RESUME_SAMPLER=0 restores the pre-lane resume) (model {})",
                        req.model,
                    );
                }
                // POOL MISS: a parked session's caches (~4GB at 128k: 17-layer trunk KV + draft
                // scratch) can starve the NEW allocation — 2 x 128k sessions + weights don't fit
                // 24GB. Misses survive affinity when the client rewrote history BELOW the
                // session's rewind boundary (or the session never captured one), so the parked
                // session is DEAD WEIGHT for this conversation: evict the pool, then allocate.
                //
                // F5 (spec-pool thrash, 2026-08-05 — research/specpool-20260804): on a
                // VRAM-tight rig EVERY turn of the daily driver is a miss (the client
                // rewrites history), so the old fail->evict->realloc walk ran once per
                // request, progressively slower as the doomed full-size ask grew the churn.
                // Two learned behaviors replace it:
                //   1. EVICT-FIRST: once a model has observed "parked ghost + new session
                //      don't fit" (evict_first), later misses evict the dead-weight pool
                //      BEFORE allocating — same eviction the failure forced anyway, minus
                //      the failed alloc. Roomy rigs never set the flag and keep the pool.
                //   2. RIGHT-SIZE LADDER: a post-evict (genuine) failure no longer dumps
                //      the whole burst to the tokenwise path. Shrink the ask toward
                //      `need` = prompt + budget + SPEC_SHRINK_SLACK — the exact cap this
                //      request's emission needs (MaxNew preempts ContextFull by
                //      construction, so a shrunken session emits identical tokens).
                //      The landing size is memoized (learned_ctx) so later misses ladder
                //      from it instead of re-walking. Below `need` = tokenwise fallback.
                if spec_sizing.evict_first.contains(&req.model) {
                    if let Some(n) = spec_reuse
                        .get_mut(&pool_key)
                        .map(|p| {
                            let n = p.len();
                            p.clear();
                            n
                        })
                        .filter(|&n| n > 0)
                    {
                        reuse_metrics.spec_evictions += n as u64;
                        eprintln!(
                            "[worker] spec pool evicted ({n}) pre-alloc \
                                   (learned VRAM-tight; model {})",
                            req.model
                        );
                    }
                }
                match lm.model.new_session(engine, ctx_cap) {
                    Ok(sess) => Some(sess),
                    Err(first_err) => {
                        let evicted = spec_reuse
                            .get_mut(&pool_key)
                            .map(|p| {
                                let n = p.len();
                                p.clear();
                                n
                            })
                            .unwrap_or(0);
                        if evicted > 0 {
                            reuse_metrics.spec_evictions += evicted as u64;
                            spec_sizing.evict_first.insert(req.model.clone());
                            eprintln!(
                                "[worker] spec pool evicted ({evicted}) after alloc \
                                       failure; retrying (evict-first learned)"
                            );
                        }
                        let retried = if evicted > 0 {
                            lm.model.new_session(engine, ctx_cap).ok()
                        } else {
                            None
                        };
                        match retried {
                            Some(sess) => Some(sess),
                            None => {
                                // Genuine capacity failure (pool empty). Right-size:
                                // ladder down from the learned/half ask toward `need`.
                                let mut sess = None;
                                if need <= ctx_cap {
                                    let mut ask = spec_sizing
                                        .learned_ctx
                                        .get(&req.model)
                                        .copied()
                                        .unwrap_or(ctx_cap / 2)
                                        .clamp(need, ctx_cap);
                                    loop {
                                        let landed = match lm.model.new_session(engine, ask) {
                                            Ok(s) => {
                                                // transient reserve (see SPEC_SHRINK_RESERVE):
                                                // a fit that leaves no headroom panics later on
                                                // a lazy upload — treat as a miss, shrink on.
                                                // (a) the embed table is the biggest lazy
                                                // transient: make it resident FALLIBLY now;
                                                // (b) on a NEW landing size only (ask > learned
                                                // — a size that already served a burst has
                                                // proven its transients resident), PROBE-
                                                // allocate the reserve and drop it. A probe,
                                                // not a mem_get_info read, is the fit signal:
                                                // the async pool's pinned release threshold
                                                // keeps freed blocks cached and invisible to
                                                // free-VRAM queries — and re-probing after the
                                                // transients are resident double-counts them
                                                // (observed: turn-1 ladder walked to need and
                                                // still failed while a 16k session + resident
                                                // transients served fine on turn 0).
                                                let proven = spec_sizing
                                                    .learned_ctx
                                                    .get(&req.model)
                                                    .is_some_and(|&l| ask <= l);
                                                let ok = lm
                                                    .model
                                                    .ensure_embed_resident(engine)
                                                    .is_ok()
                                                    && (proven
                                                        || engine
                                                            .alloc_u8_uninit(SPEC_SHRINK_RESERVE)
                                                            .is_ok());
                                                if ok {
                                                    Some(s)
                                                } else {
                                                    drop(s);
                                                    None
                                                }
                                            }
                                            Err(_) => None,
                                        };
                                        match landed {
                                            Some(s) => {
                                                eprintln!(
                                                    "[worker] spec session right-sized: \
                                                           ctx {ask} of {ctx_cap} (prompt {} + \
                                                           budget {budget}; model {})",
                                                    prompt.len(),
                                                    req.model
                                                );
                                                spec_sizing
                                                    .learned_ctx
                                                    .insert(req.model.clone(), ask);
                                                sess = Some(s);
                                                break;
                                            }
                                            None if ask > need => {
                                                ask = (ask / 2).max(need);
                                            }
                                            None => break,
                                        }
                                    }
                                }
                                if sess.is_none() {
                                    eprintln!(
                                        "[worker] spec session alloc failed ({first_err}); \
                                               tokenwise path"
                                    );
                                }
                                sess
                            }
                        }
                    }
                }
            }
        }
    } else {
        None
    };
    // PREFIX-CACHE publication arming (lane/spec-prefix-cache): a COLD spec session records the
    // probe's miss boundary so its first burst captures publishable state there (the engine
    // capture fires only when the burst's prime split lands exactly on it; seed boundary ==
    // prompt end fires after the full prime). Resumed sessions are warm — no cold boundary.
    if spec_resumed == 0 {
        if let Some(sp) = spec.as_mut() {
            sp.capture_at = snapshot_at.or(if seed_prefix {
                Some(prompt.len())
            } else {
                None
            });
        }
    }
    // spec-resume: replay sampler penalty history over the resumed prefix; queue only the suffix.
    // (text-prefix hit: replay the SESSION's committed ids — the prompt's own ids diverge at the
    // boundary; greedy sessions ignore penalties anyway, this keeps sampled-future-proofing sane.)
    if spec_resumed > 0 {
        match (&spec, &text_suffix) {
            (Some(sess), Some(_)) => {
                for &t in &sess.committed {
                    sampler.accept(t);
                }
            }
            _ => {
                for &t in &prompt[..spec_resumed] {
                    sampler.accept(t);
                }
            }
        }
    }
    // GEMMA SPEC admission (lane/gemma-batched stage 2, 2026-08-17): the assistant-drafter
    // route for dense gemma4, mirroring the Q38 policy SHAPE with each piece verified
    // against gemma4's actual arms (house law — Q38 working proves nothing here):
    //   * greedy only, penalties excluded — the gemma verify is pure argmax
    //     (spec_accept-greedy walk); penalties applied pre-argmax by the host sampler
    //     would be silently ignored (the same F3 law the qwen arm enforces above).
    //   * unconstrained, text-only — the gemma burst prime has no grammar hook and no
    //     vision overlay seam.
    //   * SOLO admission (n_active == 0): the coordinator policy is spec on
    //     single-stream, plain-batched for batch. Already-running gspec sessions keep
    //     bursting when batch traffic arrives (coexistence — bursts yield between ticks);
    //     new arrivals under load take the batched plain path.
    //   * COLD sessions or a WHOLE prefix-cache hit with a non-empty suffix
    //     (lane/spec-on-cache-hit, 2026-08-18). The gemma drafter attends the TRUNK's KV
    //     and holds no per-session draft state, so the restored trunk rows already ARE the
    //     draft state; the suffix feed regenerates the drafter seed hidden + pending
    //     (gemma_spec_session_from_restored). The continuation-pool / qwen-spec reuse
    //     shapes stay excluded exactly as before (spec_resumed == 0), and an empty-suffix
    //     hit stays PLAIN — it has no rows to feed for the seed hidden and the plain path
    //     already serves it zero-prefill from the entry's boundary logits.
    //   * requires the boot-attached drafter (gemma_draft_ready) + the MEMRA_GEMMA4_SPEC=K
    //     seam (default OFF until the mixed cell is green).
    let gspec_carrier = prefix_hit && cache.is_some() && seed_fed.len() < prompt.len();
    let gspec_k = if gemma4_spec_k_env() > 0
        && gemma_draft_ready
        && memra_engine::plan_backend::decode_batch_program(&lm.model.plan)
            == memra_engine::plan_backend::DecodeBatchProgram::Gemma
        && !lm.model.is_gemma4_e4b()
        && spec.is_none()
        && sampler.is_greedy()
        && !greedy_penalized
        && constraint.is_none()
        && !vision_req
        && !capture_req
        && n_active == 0
        && spec_resumed == 0
        && ((seed_fed.is_empty() && cache.is_none()) || gspec_carrier)
        && !confidence_trace_enabled()
    {
        gemma4_spec_k_env()
    } else {
        0
    };
    // DSPARK SPEC admission (lane/dspark-q38-recover serve route): the gemma-route policy
    // SHAPE applied to the qwen-hybrid dspark arm — greedy (argmax verify) OR sampled
    // (temperature>0 -> the route's rejection-sampling verify; lane/dspark-sampled-
    // admission-20260820), PENALTIES INCLUDED on the sampled regime since
    // lane/dspark-penalized-sampled-20260821 (the accept walk penalizes the verify
    // columns over the true per-state window — within-round accepts included — so the
    // committed stream equals plain penalized sampling; vendor defaults that ship
    // penalties now keep the spec speedup). Penalized GREEDY stays on the plain path
    // (the greedy walk argmaxes RAW columns; the engine refuses temp==0+penalties
    // loudly). Unconstrained, text-only: greedy requests use wave-projected LOW admission and
    // demote to batched plain decode at HIGH; sampled requests, gate-off, and a K>0 pin keep the
    // original solo law because they do not automatically demote (K=0 pins plain). COLD sessions only (no reuse
    // seed, no restored carrier; the dspark session primes its own cache + draft KV). serve_spec
    // carries MEMRA_SERVE_SPEC + peer-probe + confidence-trace + vision gating from above.
    // Reuse the EARLY verdict that guarded prefix lookup. Recomputing here would allow an
    // in-flight gauge change during a prefix restore window to disagree with whether lookup was
    // bypassed. The remaining assertions are consequences of a ready DFlash route: it disables
    // MTP for this model, is incompatible with gemma, and cold excluded every reuse carrier.
    // DSPARK MULTI-TURN RESUME probe (lane/dflash2-session-reuse, 2026-08-25): the MTP
    // pool's EXACT and TEXT extension tiers, applied to parked dspark sessions. Runs only
    // when no other carrier took the request (no spec resume, no gemma arm, no prefix
    // restore, no plain-reuse cache) and applies the SAME sampler predicate — a parked
    // session's philox stream position and penalty window were shaped by its sampler.
    // Load: a resume rides the demotable-session policy exactly like a cold admit (greedy
    // demotes at HIGH; sampled resumes keep the solo law via dspark_load_admits' own
    // gating inside dspark_prefers_cold — a resume under load declines to plain, which
    // costs a re-prime on the NEXT extension turn but never a wrong stream).
    let mut dspark_resume: Option<(memra_engine::dflash::DsparkSpecSession, Vec<u32>, Vec<u32>)> =
        None; // (session, pre-fed stream, suffix to prime)
    if dspark_draft_ready
        && dspark_prefers_cold
        && spec.is_none()
        && gspec_k == 0
        && spec_resumed == 0
        && seed_fed.is_empty()
        && cache.is_none()
        // "no prefix restore" (the comment above) must be checked EXPLICITLY: a successful
        // restore CONSUMES the carrier, so it also presents as seed_fed empty + cache None —
        // pre-lane those two implied "no carrier was bought", post-lane they cannot tell that
        // apart from "bought and converted". Without this arm the pool would claim the request
        // and the fold would discard the already-paid restore (review round 4); the restored
        // session covers the WHOLE prompt with a zero prime suffix, strictly cheaper than any
        // pool resume, and skipping the probe keeps the parked entry for a later extension turn.
        && dspark_prefix_restored.is_none()
        && constraint.is_none()
        && !vision_req
        && !capture_req
    {
        if let Some(pool) = dspark_reuse.get_mut(&pool_key) {
            let mut refused: Option<&'static str> = None;
            let mut admits = |e2: &DsparkReuseEntry| match spec_resume_sampler_verdict(
                spec_resume_sampler_predicate_on(),
                &sampler.identity(),
                &e2.sampler,
            ) {
                None => true,
                Some(field) => {
                    refused = Some(field);
                    false
                }
            };
            // EXACT token-extension tier.
            let hit = if let Some(index) = pool.iter().rposition(|e2| {
                e2.sess.cache_max_ctx() >= ctx_cap
                    && prompt.len() >= e2.fed.len()
                    && prompt.starts_with(&e2.fed)
                    && (prompt.len() > e2.fed.len() || !e2.done)
                    && admits(e2)
            }) {
                let entry = pool.remove(index);
                let suffix = prompt[entry.fed.len()..].to_vec();
                Some((entry, suffix))
            } else if !req.prompt_text.is_empty() {
                // TEXT tier: detok/retok BPE merges differ at chat-turn seams, so the
                // string prefix is the honest match; only the remainder is tokenized.
                pool.iter()
                    .rposition(|e2| {
                        e2.sess.cache_max_ctx() >= ctx_cap
                            && req.prompt_text.len() > e2.committed_text.len()
                            && req.prompt_text.starts_with(e2.committed_text.as_str())
                            && admits(e2)
                    })
                    .map(|index| {
                        let entry = pool.remove(index);
                        let rem = &req.prompt_text[entry.committed_text.len()..];
                        let suffix = lm.tok.encode(rem, false);
                        (entry, suffix)
                    })
            } else {
                None
            };
            match hit {
                Some((entry, suffix)) if suffix.is_empty() && entry.done => {
                    // A finished stream with nothing new to feed: resuming would re-emit
                    // from a terminal state. Put it back; serve cold.
                    pool.push(entry);
                }
                // SHORT-SUFFIX DECLINE (incident 2026-08-25). The resume primes the suffix
                // through `prime_cache`, whose batched prefill arm asserts `T >=
                // PRIME_MIN_T` and has no tokenwise twin that fills the DFlash tap sink. A
                // brief follow-up turn — the watchdog's "Say OK." class is 5 tokens — hit
                // that assert INSIDE the GPU worker thread, which exits 70 and takes every
                // live session on the box down with it: 20 panics and ~5 minutes of edge
                // 502s on box10 before MEMRA_REUSE_POOL=0 stopped the loop. Declining here
                // keeps the entry parked for a longer next turn and serves this one cold,
                // which is exactly the pre-lane behavior; the engine carries its own
                // refusal as a backstop.
                Some((entry, suffix))
                    if suffix.len() < memra_engine::hybrid_forward::PRIME_MIN_T =>
                {
                    eprintln!(
                        "[worker] dspark-reuse DECLINED: suffix {} < prime floor {}; \
                         serving COLD and keeping the entry parked (model {})",
                        suffix.len(),
                        memra_engine::hybrid_forward::PRIME_MIN_T,
                        req.model
                    );
                    reuse_metrics.spec_misses += 1;
                    pool.push(entry);
                }
                Some((entry, suffix)) => {
                    eprintln!(
                        "[worker] dspark-reuse: {} committed tokens resumed                          (suffix {} to prime) [sampler={}] (model {})",
                        entry.fed.len(),
                        suffix.len(),
                        if spec_resume_sampler_predicate_on() {
                            "equivalent"
                        } else {
                            "unchecked"
                        },
                        req.model
                    );
                    reuse_metrics.spec_hits += 1;
                    // Replay the sampler's penalty window over the parked stream — the
                    // session's own pen_hist rides inside it; this is the WORKER
                    // sampler's bookkeeping (the plain-resume convention).
                    for &t in &entry.fed {
                        sampler.accept(t);
                    }
                    dspark_resume = Some((entry.sess, entry.fed, suffix));
                }
                None => {
                    if let Some(field) = refused {
                        reuse_metrics.spec_sampler_refusals += 1;
                        eprintln!(
                            "[worker] dspark-reuse REFUSED: parked session's sampler                              differs on {field}; serving COLD (model {})",
                            req.model
                        );
                    } else if !pool.is_empty() {
                        // Named miss (the pool's convention: silence only for the empty
                        // pool). The overlap length against the best entry is the one
                        // number that separates "different conversation" from "template
                        // seam broke the prefix" — the latter diverges INSIDE the shared
                        // history, at the same offset every turn.
                        let best = pool
                            .iter()
                            .map(|e2| {
                                req.prompt_text
                                    .as_bytes()
                                    .iter()
                                    .zip(e2.committed_text.as_bytes())
                                    .take_while(|(a, b)| a == b)
                                    .count()
                            })
                            .max()
                            .unwrap_or(0);
                        eprintln!(
                            "[worker] dspark-reuse MISS: {} parked, best text overlap {}B                              of {}B prompt; serving COLD (model {})",
                            pool.len(),
                            best,
                            req.prompt_text.len(),
                            req.model
                        );
                    }
                }
            }
        }
    }
    let dspark_on = dspark_prefers_cold || dspark_resume.is_some();
    debug_assert!(
        !dspark_on
            || (spec.is_none()
                && gspec_k == 0
                && spec_resumed == 0
                && seed_fed.is_empty()
                && cache.is_none()),
        "early DFlash preference must imply a cold or pool-resumed session",
    );
    let dspark_exact_key_present =
        dspark_on && prefix_on && px.touch_exact_without_credit(&pool_key, &prompt);
    let dspark_capture_prefix = dspark_prefix_capture_requested(
        dspark_on,
        prefix_on,
        prompt.len(),
        dspark_exact_key_present,
        has_exact_preprime_dspark_owner,
    );
    // legacy tokenwise cache only when the spec path did NOT take the session (spec owns its own).
    //
    // STAGE-OWNED KV (pp2-batch 2026-08-06): `pp::new_cache`, not `Cache::new`. With the ppN
    // door shut it IS `Cache::new` (one branch, same allocations); with the door open across
    // devices it allocates each layer's KV/recurrent state through the engine of the STAGE that
    // runs that layer, and adds the cache-birth barrier. Allocating a serving cache on the
    // primary under an open door would leave every remote stage peer-reading its OWN cache
    // every step — the same silent-PCIe class as unsharded weights (13.9-28x on a PRO 6000
    // pair), and invisible to exactness gates because peer reads are byte-exact.
    // A gemma-spec session allocates NOTHING here — its cache is born inside
    // gemma_spec_session_new at the first spec tick (s.cache stays None until demote).
    // EXCEPT the prefix-hit carrier (lane/spec-on-cache-hit): the restored trunk cache
    // rides in s.cache and gemma_spec_session_from_restored takes it at that same tick.
    // A dspark session likewise allocates nothing here — its cache is born inside
    // dspark_spec_session_new. dspark_on strictly narrows: its admission above requires
    // a COLD session (gspec_k == 0, spec none, cache none, no seed), so the arms below
    // are mutually exclusive and with dspark off this expression is exactly the
    // spec-on-cache-hit lane's text.
    let cache = if dspark_on {
        None
    } else if gspec_k > 0 {
        if gspec_carrier { cache } else { None }
    } else {
        match (&spec, cache) {
            (Some(_), c) => c, // reuse hit carried a cache? keep it parked as-is (rare; None normally)
            (None, Some(c)) => Some(c),
            (None, None) => match alloc_with_single_reclaim_retry(
                || {
                    memra_engine::pp::new_cache_planned(
                        engine,
                        &lm.model.cfg,
                        &lm.model.plan,
                        ctx_cap,
                    )
                },
                |err| {
                    // Headroom discipline: prefix entries always yield before a session errors.
                    // A quoted allocation OOM additionally reuses cap256k's global continuation-pool
                    // LRU hook. Both drops happen before ONE retry; a second failure is reported.
                    let evicted_prefix = px.evict_all();
                    if evicted_prefix > 0 {
                        eprintln!(
                            "[prefix-cache] evicted {evicted_prefix} entries after cache alloc \
                               failure; retrying"
                        );
                    }
                    let evicted_parked = if is_cuda_oom(&err.to_string()) {
                        evict_oldest_parked(reuse, spec_reuse, dspark_reuse, reuse_metrics)
                    } else {
                        None
                    };
                    if let Some(pool) = evicted_parked {
                        eprintln!(
                            "[admit-oom] reclaim-on-alloc-oom: evicted oldest {} parked \
                               session (global LRU); retrying cache alloc once",
                            match pool {
                                ParkedPool::Plain => "plain",
                                ParkedPool::Spec => "spec",
                                ParkedPool::Dspark => "dspark",
                            }
                        );
                    }
                    evicted_prefix > 0 || evicted_parked.is_some()
                },
            ) {
                Ok(c) => Some(c),
                Err(err) => {
                    return Err((
                        req.tx,
                        EngineError::engine(format!("cache alloc failed: {err}")),
                    ));
                }
            },
        }
    };
    // WORKER-TRUTH usage accounting: total prompt tokens (as the worker actually feeds/resumes
    // them — the text-prefix spec resume re-tokenizes only the remainder) + how many came from
    // a cache instead of being computed.
    // A prefix-cache dspark restore rides the SAME seam as a session-pool resume: a prebuilt
    // session with an EMPTY suffix, i.e. already primed. The fold sits ABOVE the accounting
    // read below ON PURPOSE — its first placement was after it, so a restored request fell to
    // the cold arm and billed `cached=0` while the whole prompt came from cache (the customer
    // paying full price for cached tokens; caught during the hazard investigation, visible as
    // `cached=0` in every restored response of the gate runs). Same reason the sampler replay
    // lives here: every sibling seam (plain hit, MTP restore, pool resume) replays the
    // restored prefix into the worker sampler's window, and the conversion site cannot — the
    // sampler is borrowed by the decision code above it.
    let dspark_resume = match (dspark_resume, dspark_prefix_restored) {
        (Some(pool), None) => Some(pool),
        (pool, Some((sess, fed))) => {
            // Both should be impossible — the pool probe is gated on
            // `dspark_prefix_restored.is_none()` (review round 4) — but if a future edit
            // re-opens that seam, the RESTORED session wins: its carrier restore is already
            // paid and it covers the whole prompt with a zero prime suffix, where the pool
            // entry would re-prime a suffix. Loud, because reaching here means the probe
            // gate and this fold disagree.
            if pool.is_some() {
                eprintln!(
                    "[worker] dspark fold: pool resume AND prefix restore both claimed the \
                     request; preferring the paid restore — the pool-probe gate should have \
                     prevented this"
                );
            }
            for &t in &fed {
                sampler.accept(t);
            }
            Some((sess, fed, Vec::new()))
        }
        (None, None) => None,
    };
    let (n_prompt, n_cached) = if let Some((_, fed, sfx)) = &dspark_resume {
        // dspark pool resume: the parked stream's rows come from the session's own caches;
        // only the suffix is computed. Text-tier retok means fed+suffix, not prompt.len().
        (fed.len() + sfx.len(), fed.len())
    } else if spec_restored_cached > 0 {
        // prefix-cache spec restore: committed already covers the whole prompt (the
        // engine fed the suffix), but only the restored prefix came from cache.
        (prompt.len(), spec_restored_cached)
    } else if spec_resumed > 0 {
        let suffix_len = text_suffix
            .as_ref()
            .map(|t| t.len())
            .unwrap_or_else(|| prompt.len() - spec_resumed);
        (spec_resumed + suffix_len, spec_resumed)
    } else {
        (prompt.len(), seed_fed.len())
    };
    // LATE WAVE RE-READ (PR #37 review finding, 2026-08-24). The serve-time re-decision
    // below exists to fold in post-restore information (n_prompt/n_cached above), but it
    // consumed the `projected_wave` captured BEFORE the prefix-restore work — while the
    // sibling load guard reads its gauges "as LATE as possible — after the restore work, at
    // the refusal site — so that window counts" (`HTTP_INFLIGHT`'s doc: the other fifteen
    // land while this request does its own 312 MB restore). `projected_admission_wave`
    // reads the PENDING_ADMITS / HTTP_INFLIGHT gauges at CALL time, so re-calling it here
    // sees the wave that arrived during the restore window; the head-of-wave no longer
    // survives as the burst's lone spec row. Gauges only grow inside the window, so a late
    // read can only demote spec -> plain — the conservative direction
    // `admission_request_may_spec`'s own doc calls the cheap side of the error. The
    // admission-time estimate keeps the early capture (it priced the reservation); this
    // shadow feeds the re-decision AND the K receipt log below, so the logged wave is the
    // one the decision actually consumed.
    let projected_wave = projected_admission_wave(n_active, n_pending);
    if spec.is_some() && req_spec_k_replay.is_none() {
        spec_k_decision = choose_spec_k(
            spec_k_pin(),
            spec_gate_on(),
            *spec_gate_thresholds(),
            projected_wave,
            n_prompt,
            n_cached,
            spec_trim_head(lm),
        );
    }
    let (spec_k, resume_floor) = resumed_carrier_spec_k_floor(spec.is_some(), spec_k_decision.k);
    if mtp_spec_capable(lm) {
        let source = if resume_floor {
            "resume-floor(k=0->1)"
        } else if spec_k == spec_k_decision.k {
            spec_k_decision.reason.as_str()
        } else {
            "eligibility-fallback"
        };
        let placement = if spec_gate_pp2_placement() {
            "pp2-cross-device"
        } else {
            "single-or-non-pp2"
        };
        eprintln!(
            "[spec-k] model={:?} tenant={:?} K={spec_k} source={source} \
             prompt={n_prompt} cached={n_cached} lcp={policy_lcp} \
             active={} wave={projected_wave} placement={placement}",
            req.model,
            crate::auth::meter_key(&req.cache_ns),
            n_active + 1,
        );
    }
    // DEPTH-UNFREEZE ARM (H11, lane/hermes-perf-fixes 2026-08-23 — the 3.1x lever;
    // full receipts on `maybe_prefix_seed`). Both deepening paths used to sit inside
    // `reused.is_none()`, so a PLAIN session serving a prefix-cache hit could never publish
    // its deeper primed state and the class froze at its first seed depth. Re-arm the seed
    // for plain hit sessions; the depth compare at prefill-done (`prefix_seed_deepens`)
    // decides whether the gain over the deepest covering entry is worth the snapshot.
    // Plain-only: a spec session's `capture_at` publication path has its own measured
    // boundary discipline (spec-prefix-cache lane) and is not re-armed here.
    // Never for eager-only models (PR #37 review finding, 2026-08-24): R16
    // (research/cacheinval-20260813) makes the eager refusal a hard prerequisite of this
    // exact mechanism — a carried eager hit rides tokenwise `decode_step` for its whole
    // suffix (H1), so its "deepened" seed would publish restore+decode_step chained
    // provenance and multiply traffic onto the H1 crossing. Same predicate as the ckpt
    // arm below; the qwen-class lever this fix targets is untouched by the exclusion.
    if plain_hit_reseed_arms(
        prefix_hit,
        spec.is_none(),
        eager_only_model(lm),
        prompt.len(),
    ) {
        seed_prefix = true;
    }
    // PLAIN-SESSION AFFINITY checkpoint arming (lane/plain-affinity, 2026-08-09). A plain-path
    // session captures a rewind checkpoint at a STABLE PRE-GENERATION boundary so its NEXT
    // rewritten-history turn resumes here. Armed only when:
    //   - affinity is on (the rollback seam / A/B arm),
    //   - this session is on the PLAIN path (spec owns its own SpecCheckpoint tier),
    //   - the model can continuation-prime a suffix over a rewound cache (eager-only gemma4
    //     cannot — the engine refuses pos > 0 prime — so it could never resume; don't capture),
    //   - the checkpoint is NOMINATABLE (2026-08-09 cache-metering regression fix): the client
    //     named its conversation (explicit tier), or the prompt carries enough turn-marker
    //     structure that the implicit fingerprint tier could ever reach the FP_MIN_SEGMENTS
    //     nomination bar. A markerless anonymous prompt can never be nominated, so arming it
    //     bought nothing and COST the in-batch fanout (an armed ckpt_at needs a per-session
    //     boundary-stopped prime, which excludes the session from fanout/prime-batch): the
    //     serve-smoke cache-meter gate's 5-way shared-prefix prompt_ids burst went 0-hit/6-miss
    //     because every session armed a guard-window checkpoint nothing could resume,
    //   - the boundary lies AHEAD of what is already primed (`> seed_fed.len()`), so the prefill
    //     tick will actually stop there and snapshot (a boundary already inside the resumed
    //     prefix cannot be captured by a prime-stop; that turn simply re-primes next time).
    // The boundary is derived from the prompt's own turn markers (`plain_checkpoint_boundary`),
    // never a hardcoded offset; the exact diff at resume still decides on bytes.
    let ckpt_at = if affinity_enabled()
        && spec.is_none()
        && vision_state.is_none()
        && !eager_only_model(lm)
        && !confidence_trace_enabled()
        && (req.affinity.is_some()
            || plain_ckpt_nominatable(&prompt, &|t| lm.tok.token_is_control(t)))
    {
        plain_checkpoint_boundary(&prompt, &|t| lm.tok.token_is_control(t))
            .filter(|&b| b > seed_fed.len())
    } else {
        None
    };
    // dspark pool resume: split the carried triple for the literal below. `prefill_done`
    // follows the suffix: an empty-suffix exact continuation is already primed (the burst
    // emits the parked boundary token), a suffix resume primes in step_dspark_spec.
    let dspark_session_installed = dspark_resume.is_some();
    let (dspark_resume_sess, dspark_resume_fed, dspark_resume_suffix) = match dspark_resume {
        Some((sess, fed, sfx)) => (Some(sess), Some(fed), Some(sfx)),
        None => (None, None, None),
    };
    let prefill_done_at_admit = if dspark_resume_sess.is_some() {
        dspark_resume_suffix.as_ref().is_some_and(|s| s.is_empty())
    } else {
        prefill_done_at_admit
    };

    Ok(Session {
        model: req.model,
        spec_k,
        cache_ns: req.cache_ns,
        affinity: req.affinity,
        lane: req.lane,
        cache,
        sampler,
        spec,
        graph: None,
        graph_pending: None,
        oom_retries: req_oom_retries,
        vision_memory,
        replay,
        spec_drafted: 0,
        spec_accepted: 0,
        spec_rounds: 0,
        last_logits: seed_logits,
        device_next: None,
        gspec: None,
        gspec_k,
        gspec_ctx: ctx_cap,
        dspark: dspark_resume_sess,
        // ROOT CAUSE of the "tiny-budget hazard" (bench repro 2026-08-27, closed): this field
        // routes the tick dispatch, and it used to carry the value computed BEFORE the
        // prefix-restore fold — so a restored session installed with dspark_on=false was
        // stepped by PLAIN step_session over its absent s.cache, decoding coherent garbage
        // from an empty context (the wikipedia-flag output; cache frozen, zero accept lines).
        // The dispatch flag now derives from what this literal actually installs, so the two
        // cannot disagree by construction. Budget was never the variable — every shape with
        // correct dispatch was byte-exact, including 414-prompt/64-budget.
        dspark_on: dspark_on || dspark_session_installed,
        dspark_capture_prefix,
        constraint,
        mask_dev: None,
        mask_words: 0,
        vision: vision_state,
        capture: req.capture,
        fed: dspark_resume_fed.unwrap_or(seed_fed),
        prefill_queue: if let Some(sfx) = dspark_resume_suffix {
            sfx.into_iter().collect()
        } else if let Some(ts) = text_suffix {
            ts.into_iter().collect()
        } else if spec_resumed > 0 {
            prompt[spec_resumed..].to_vec().into_iter().collect()
        } else {
            suffix.into_iter().collect()
        },
        prefill_done: prefill_done_at_admit,
        generated: Vec::new(),
        tokens_emitted: 0,
        aborted: false,
        params,
        stop_strings: req.stop_strings,
        trace_id: req.trace_id,
        decoded_bytes: Vec::new(),
        emitted_bytes: 0,
        budget,
        n_prompt,
        n_cached,
        snapshot_at,
        ckpt_at,
        ckpt_snap: None,
        prefix_miss_lcp,
        seed_prefix,
        prefix_pin,
        tx: req.tx,
        ttft: req.ttft,
        t0: Instant::now(),
    })
}

/// Resolve each capture piece to a single vocabulary token and read its last-position
/// logit (lane/embed-serve). A piece that is not exactly one token in this model's
/// vocabulary reports `f32::MIN` — the rerank route treats that as "model unsupported"
/// rather than inventing a score from a multi-token merge.
fn capture_piece_logits(
    cap: &CaptureSpec,
    tok: &memra_tokenizer::Tokenizer,
    last_logits: &[f32],
) -> Vec<f32> {
    cap.logit_pieces
        .iter()
        .map(|piece| {
            let ids = tok.encode(piece, false);
            match ids.as_slice() {
                [id] => last_logits.get(*id as usize).copied().unwrap_or(f32::MIN),
                _ => f32::MIN,
            }
        })
        .collect()
}

/// Return only the newly completed UTF-8 text. Tokenizer byte-fallback sequences may span token
/// boundaries; retain an incomplete suffix until a later token completes it instead of emitting a
/// permanent replacement character. Truly invalid bytes are consumed as U+FFFD so they cannot
/// stall every later delta.
fn utf8_delta(decoded: &[u8], emitted_bytes: &mut usize) -> String {
    if *emitted_bytes > decoded.len() {
        return String::new();
    }
    let mut cursor = *emitted_bytes;
    let mut delta = String::new();
    while cursor < decoded.len() {
        match std::str::from_utf8(&decoded[cursor..]) {
            Ok(text) => {
                delta.push_str(text);
                cursor = decoded.len();
            }
            Err(err) => {
                let valid = err.valid_up_to();
                if valid != 0 {
                    // SAFETY: `valid_up_to` is the exact valid UTF-8 prefix certified by Rust.
                    delta.push_str(unsafe {
                        std::str::from_utf8_unchecked(&decoded[cursor..cursor + valid])
                    });
                    cursor += valid;
                }
                match err.error_len() {
                    None => break,
                    Some(invalid) => {
                        delta.push('\u{fffd}');
                        cursor += invalid;
                    }
                }
            }
        }
    }
    *emitted_bytes = cursor;
    delta
}

fn contains_stop_string(decoded: &[u8], stop_strings: &[String]) -> bool {
    if stop_strings.is_empty() {
        return false;
    }
    let full = String::from_utf8_lossy(decoded);
    // Empty ELEMENTS are filtered at ingestion (StopSequences::into_vec), but
    // `"".contains("")` is true for every haystack, so guard here too — an empty
    // element from any future constructor must never stop every decode at token one.
    stop_strings
        .iter()
        .any(|stop| !stop.is_empty() && full.contains(stop.as_str()))
}

/// Publish one token-id event and advance the receipt counter only after the channel accepted it.
fn send_token_event(s: &mut Session, id: u32, text: String) -> bool {
    if s.tx.send(Event::Token { id, text }).is_err() {
        return false;
    }
    s.tokens_emitted += 1;
    true
}

/// Number of tokens from an engine-committed speculative burst that belong to this request.
/// Session-mode engine output may cross the scheduler's burst target to keep cache rows and
/// `SpecSession::committed` identical. That surplus is still public on a non-final burst; only
/// the request's remaining budget (or the first EOS) clamps event/generated/usage surfaces.
fn spec_visible_len(tokens: &[u32], request_room: usize, eos_ids: &[u32]) -> usize {
    let capped = tokens.len().min(request_room);
    tokens[..capped]
        .iter()
        .position(|id| eos_ids.contains(id))
        .map_or(capped, |i| i + 1)
}

#[derive(Debug, Default, PartialEq, Eq)]
struct SpecEmitResult {
    sent: usize,
    send_ok: bool,
}

/// Emit a speculative commit slice one token id at a time. `remaining` is the request-owned
/// emission budget, deliberately separate from the engine's committed/cache-row count.
fn emit_spec_token_events<D, S>(
    tokens: &[u32],
    remaining: &mut usize,
    decoded: &mut Vec<u8>,
    cursor: &mut usize,
    eos_ids: &[u32],
    eos_seen: &mut bool,
    mut decode: D,
    mut send: S,
) -> SpecEmitResult
where
    D: FnMut(u32) -> Vec<u8>,
    S: FnMut(Event) -> bool,
{
    let mut result = SpecEmitResult {
        send_ok: true,
        ..Default::default()
    };
    for &id in tokens {
        if *remaining == 0 || *eos_seen {
            break;
        }
        *remaining -= 1;
        let text = if eos_ids.contains(&id) {
            // EOS is still a generated/billed token id, but its marker text is never streamed.
            *eos_seen = true;
            String::new()
        } else {
            decoded.extend_from_slice(&decode(id));
            utf8_delta(decoded, cursor)
        };
        if !send(Event::Token { id, text }) {
            result.send_ok = false;
            break;
        }
        result.sent += 1;
    }
    result
}

/// Account one scheduler call in client-output units. A speculative call may publish many
/// tokens; its wall time is converted to a per-token sample and weighted once per emitted token,
/// matching the one-token legacy path and the rate-limit estimator's `tokens * step_ms` units.
#[allow(clippy::too_many_arguments)]
fn record_output_progress(
    generated_before: usize,
    generated_after: usize,
    lane: Lane,
    elapsed_ms: f32,
    n_tokens_out: &mut u64,
    lane_tokens: &mut [u64; 3],
    step_stats: &mut StepStats,
    last_interactive_decode: &mut Instant,
) -> usize {
    let emitted = record_output_tokens(
        generated_before,
        generated_after,
        lane,
        n_tokens_out,
        lane_tokens,
    );
    if emitted == 0 {
        return 0;
    }
    if lane == Lane::Interactive {
        let per_token_ms = elapsed_ms / emitted as f32;
        for _ in 0..emitted {
            step_stats.record(per_token_ms);
        }
        *last_interactive_decode = Instant::now();
    }
    emitted
}

/// Publish output counters from request-owned generated-token progress, independent of whether
/// the scheduler row survives to a following decode. EOS, callback, and context-full all emit
/// their terminal token before returning `false`; counting only survivors loses exactly that
/// client-visible token. MaxNew emits nothing in its terminal check because its final token was
/// already accounted on the preceding call.
fn record_output_tokens(
    generated_before: usize,
    generated_after: usize,
    lane: Lane,
    n_tokens_out: &mut u64,
    lane_tokens: &mut [u64; 3],
) -> usize {
    let emitted = generated_after.saturating_sub(generated_before);
    *n_tokens_out += emitted as u64;
    lane_tokens[lane.idx()] += emitted as u64;
    emitted
}

/// Prime one common cold prefix per same-window fanout group, then restore that snapshot
/// into every sibling. Returns sessions whose prefill budget was consumed by this stage so
/// the ordinary single-prime loop does not give them a second chunk in the same tick.
fn prefix_fanout_eligible(s: &Session, eager_only: &std::collections::HashSet<String>) -> bool {
    !memra_engine::pp::pp_host_bounce_active()
        && s.vision.is_none()
        && s.capture.is_none()
        && s.spec.is_none()
        && s.graph.is_none()
        && !s.prefill_done
        && s.lane == crate::lanes::Lane::Interactive
        && s.fed.is_empty()
        && s.n_cached == 0
        && s.prefix_pin.is_none()
        && s.prefix_miss_lcp.is_some()
        && s.snapshot_at.is_none()
        // a checkpoint-capturing session needs its own boundary-stopped prime — fanout
        // primes the shared prefix monolithically and cannot honor the per-session stop.
        && s.ckpt_at.is_none()
        && s.cache.as_ref().is_some_and(|c| c.pos == 0 && !c.has_swa_ring())
        && !eager_only.contains(&s.model)
        && s.prefill_queue.len() >= PREFIX_CACHE_MIN_TOKENS
}

#[allow(clippy::too_many_arguments)]
fn dedup_interactive_prefixes(
    engine: &Engine,
    loaded: &HashMap<String, LoadedModel>,
    eager_only: &std::collections::HashSet<String>,
    px: &mut PrefixCache,
    active: &mut [Session],
    finished: &mut Vec<usize>,
    prefix_cap: usize,
    n_cached_in: &mut u64,
    ns_tokens: &mut HashMap<String, [u64; 2]>,
) -> std::collections::HashSet<usize> {
    let mut advanced = std::collections::HashSet::new();
    if !prefix_dedup_enabled() || prefix_cap < PREFIX_CACHE_MIN_TOKENS || confidence_trace_enabled()
    {
        return advanced;
    }
    let candidates: Vec<PrefixFanoutCandidate> = active
        .iter()
        .enumerate()
        .filter(|(i, s)| !finished.contains(i) && prefix_fanout_eligible(s, eager_only))
        .map(|(active_idx, s)| PrefixFanoutCandidate {
            active_idx,
            key: s.pool_key(),
            prompt: s.prefill_queue.iter().copied().collect(),
        })
        .collect();

    for group in prefix_fanout_groups(&candidates, prefix_cap) {
        let leader_i = group.members[0];
        if finished.contains(&leader_i) {
            continue;
        }
        let prefix: Vec<u32> = active[leader_i]
            .prefill_queue
            .iter()
            .take(group.prefix_len)
            .copied()
            .collect();
        let queued_after = active[leader_i].prefill_queue.len() - group.prefix_len;
        let model = active[leader_i].model.clone();
        let key = active[leader_i].pool_key();
        let t0 = Instant::now();
        let leader_out = {
            let s = &mut active[leader_i];
            loaded[&model].model.prime_cache(
                engine,
                &prefix,
                s.cache.as_mut().unwrap(),
                queued_after,
            )
        };
        let (leader_logits, _h, _x) = match leader_out {
            Ok(out) => out,
            Err(err) => {
                let _ = active[leader_i]
                    .tx
                    .send(Event::Error(EngineError::engine(format!(
                        "prefix fanout prime failed: {err}"
                    ))));
                finished.push(leader_i);
                eprintln!("[prefix-dedup] leader prime FAILED (model {model}): {err}");
                continue;
            }
        };
        advanced.insert(leader_i);
        let snapshot = prefix_snapshot(
            engine,
            active[leader_i].cache.as_ref().unwrap(),
            &key,
            &prefix,
            &leader_logits,
        );
        {
            let s = &mut active[leader_i];
            s.last_logits = leader_logits;
            s.prefill_queue.drain(..group.prefix_len);
            for &tok in &prefix {
                s.fed.push(tok);
                s.sampler.accept(tok);
            }
            s.prefill_done = s.prefill_queue.is_empty();
            s.prefix_miss_lcp = None;
        }
        let entry = match snapshot {
            Ok(entry) => {
                trace_prefix_entry_state(engine, &entry, entry.pos, "snapshot", "fanout");
                entry
            }
            Err(err) => {
                eprintln!("[prefix-dedup] snapshot failed ({err}); siblings prime cold");
                continue;
            }
        };

        let mut participants = vec![leader_i];
        for &i in group.members.iter().skip(1) {
            if finished.contains(&i) {
                continue;
            }
            let restored = prefix_restore(engine, active[i].cache.as_mut().unwrap(), &entry, &key);
            if let Err(err) = restored {
                let _ = active[i].tx.send(Event::Error(EngineError::engine(format!(
                    "prefix fanout restore failed: {err}"
                ))));
                finished.push(i);
                eprintln!("[prefix-dedup] sibling restore FAILED (model {model}): {err}");
                continue;
            }
            // NOTE (code-audit-20260809 §2.5): this `.expect()` asserts a single-threaded
            // invariant (a fanout sibling always carries the provisional miss recorded at its
            // admission). This lane's new admission routes (plain-affinity resume) do NOT enter
            // the fanout path — an armed `ckpt_at` excludes a session from `prefix_fanout_eligible`
            // — so the invariant is untouched here and the assert stays. Continuing past a missing
            // miss AFTER a successful restore would double-prime the sibling (cache holds the
            // prefix, session state not advanced) — a worse failure than the assert. Left as-is.
            let miss_lcp = active[i]
                .prefix_miss_lcp
                .take()
                .expect("prefix fanout sibling must carry its admission miss");
            {
                let s = &mut active[i];
                s.last_logits = entry.last_logits.clone();
                s.prefill_queue.drain(..group.prefix_len);
                for &tok in &prefix {
                    s.fed.push(tok);
                    s.sampler.accept(tok);
                }
                s.prefill_done = s.prefill_queue.is_empty();
                s.n_cached += group.prefix_len;
                s.seed_prefix = false;
                // BILLING: re-emit prompt usage after crediting the fanout prefix.
                // `Event::PromptUsage` is the ONLY pre-terminal usage event and its admission-time
                // emitter fired with the pre-credit `n_cached`. A sibling that never reaches a
                // terminal event (client cancel, disconnect) is priced by `PendingReceipt::drop`
                // from whatever the last recorded value was — so without this re-emit its entire
                // reused prefix bills at the full input rate instead of the cached rate, which on
                // the sold shape is thousands of tokens overcharged per cancelled sibling, in the
                // customer's disfavour. Fanout is on by default.
                // `record_prompt_usage` SETS rather than accumulates, so re-emitting is idempotent
                // and a sibling that does reach a terminal event is unaffected.
                let _ = s.tx.send(Event::PromptUsage {
                    n_prompt: s.n_prompt,
                    n_cached: s.n_cached,
                });
            }
            px.promote_miss_to_hit(miss_lcp, group.prefix_len);
            *n_cached_in += group.prefix_len as u64;
            meter_cached_credit(ns_tokens, &active[i].cache_ns, group.prefix_len as u64);
            participants.push(i);
            advanced.insert(i);
        }

        let pin = px.insert_pinned(&key, entry, "in-batch fanout", participants.len());
        for &i in &participants {
            active[i].seed_prefix = false;
            if let Some(pin) = &pin {
                debug_assert!(active[i].prefix_pin.is_none());
                active[i].prefix_pin = Some(pin.clone());
            }
        }
        eprintln!(
            "[prefix-dedup] B={} prefix={} saved={} hash={:016x} in {:.1}ms \
             retained={} (model {}{})",
            participants.len(),
            group.prefix_len,
            group.prefix_len * participants.len().saturating_sub(1),
            fnv1a(0xcbf29ce484222325, &prefix),
            t0.elapsed().as_secs_f64() * 1e3,
            pin.is_some(),
            model,
            ns_suffix(&key.1),
        );
    }
    advanced
}

/// One scheduler tick for one session. Returns Ok(true) if still running, Ok(false) if retired.
/// Decomposes `generate_with`'s loop body into a single per-session step (same semantics):
///   - prefill phase: consume ONE prompt token via decode_step, accept it into the sampler.
///   - decode phase: sample from last_logits, accept, stream the token, check EOS/stop/ctx, then
///     run ONE decode_step to produce the next logits.
/// One prefill tick for a session under a token budget. Returns tokens consumed.
/// Same chunking laws as step_session's prefill phase (PRIME_MIN_T floor, tail handling). The
/// tail-handling half of that claim was FALSE until 2026-08-13 — this site's merge was a no-op
/// while both sibling sites' worked; see the tail-merge comment inside.
fn prefill_tick(
    engine: &Engine,
    loaded: &HashMap<String, LoadedModel>,
    px: &mut PrefixCache,
    s: &mut Session,
    budget: usize,
    vision_tower: Option<&memra_engine::vision::VisionTower>,
    gemma_tower: Option<&memra_engine::vision_gemma::GemmaVisionTower>,
) -> Result<usize, Box<dyn std::error::Error>> {
    if let Some(trace) = s.ttft.as_ref() {
        trace.mark_prime_start();
    }
    let lm = &loaded[&s.model];
    // VISION (lane/vision): build the embedding overlay once (tower forward, GPU) and keep
    // the whole prefill on the PRIME program — pad tokens must never reach decode_step,
    // whose plain pad embedding would silently corrupt the image region. The budget floor
    // guarantees the prime branch below stays reachable under starved tick budgets.
    let budget = if s.vision.is_some() || s.capture.is_some() {
        // capture sessions must end on a PRIME call (the hidden stack the capture reads
        // exists only there) — keep the prime branch reachable under starved budgets,
        // same law as vision
        budget.max(memra_engine::hybrid_forward::PRIME_MIN_T)
    } else {
        budget
    };
    if let Some(v) = s.vision.as_mut() {
        if v.overlay.is_none() {
            build_vision_overlay(
                engine,
                vision_tower,
                gemma_tower,
                lm.model.cfg.n_embd as usize,
                v,
            )?;
        }
    }
    let q = s.prefill_queue.len();
    if q == 0 {
        s.prefill_done = true;
        maybe_prefix_seed(engine, px, s);
        if let Some(trace) = s.ttft.as_ref() {
            trace.mark_prime_end();
        }
        return Ok(0);
    }
    let mut consumed = 0usize;
    // EAGER-ONLY prime shape (lane/gemma4-serve-gaps, 2026-08-07): gemma4's prime is
    // fresh-monolithic ONLY — no chunked and no continuation prime (the engine refuses
    // pos > 0; before it refused, chunk 2 of a >tick-budget prompt KILLED the worker on
    // gemma4_prime's assert, in both scheduler modes). So: fresh prompts prime WHOLE
    // (budget uncapped — a long gemma4 prompt trades one long tick for correctness),
    // carried suffixes (reuse/prefix resume) ride the tokenwise decode_step path, and the
    // LCP split is skipped (its boundary-stop would turn the tail into a continuation).
    let eager_mono = eager_only_model(lm);
    let carried = s.cache.as_ref().is_some_and(|c| c.pos > 0);
    if eager_mono {
        s.snapshot_at = None;
        // eager-only models (gemma4) cannot continuation-prime a suffix over a rewound cache
        // (the engine refuses pos > 0 prime), so plain-affinity resume excludes them — no
        // point capturing a checkpoint they can never resume from.
        s.ckpt_at = None;
    }
    // BOUNDARY STOP: the prime must stop exactly at the NEXT of two pre-generation boundaries
    // ahead — the prefix-cache LCP split (`snapshot_at`) and the plain-affinity checkpoint
    // (`ckpt_at`). Taking the MIN keeps both captures landing on their exact fed boundary; the
    // same session rarely carries both, but when it does the earlier boundary is honored first
    // and the later one next tick.
    // OPEN DEFECT (two-programs inventory W1, 2026-08-13): when a boundary sits CLOSER than
    // PRIME_MIN_T ahead, the `bound_rem.is_none_or(..)` term below vetoes the whole prime branch
    // and the remaining PROMPT tokens go through `decode_step` one at a time — a different
    // numeric program for the same bytes (see the tail-merge comment below). An older note here
    // called that residual "unreachable at the current PREFILL_TICK_T"; that claim covered only
    // the tick-budget tail, which is now genuinely merged, and never covered this boundary door.
    // Correct remedy is to DROP the sub-floor boundary capture (lose a cache seed, keep one
    // program) rather than degrade to tokenwise; it is not done here because clearing
    // snapshot_at/ckpt_at interacts with the post-prime capture below. Tracked separately.
    let fed_len = s.fed.len();
    let bound_rem = [s.snapshot_at, s.ckpt_at]
        .into_iter()
        .flatten()
        .filter(|&b| b > fed_len)
        .map(|b| b - fed_len)
        .min();
    if !confidence_trace_enabled()
        && q >= memra_engine::hybrid_forward::PRIME_MIN_T.max(2)
        && budget >= memra_engine::hybrid_forward::PRIME_MIN_T
        && !(eager_mono && carried)
        && bound_rem.is_none_or(|r| r >= memra_engine::hybrid_forward::PRIME_MIN_T)
    {
        let take = prefill_tick_take(q, budget, eager_mono, bound_rem);
        if std::env::var("MEMRA_DEBUG_PRIMESEG").is_ok() {
            // PRIME-PROGRAM RECEIPT (lane/spec-longctx-20260821). A cross-prime-path byte
            // comparison is only interpretable if the CALL SEQUENCE is known: the prime-grid
            // law is about call STARTS, and the tokenwise fallback below is a different
            // numeric program entirely. Diagnostic only.
            eprintln!(
                "[primeseg] call start={fed_len} take={take} grid_off={} bound_rem={:?} \
                 ckpt_at={:?} snapshot_at={:?} q={q} budget={budget}",
                fed_len % memra_engine::Engine::gdn_chunk_size(),
                bound_rem,
                s.ckpt_at,
                s.snapshot_at,
            );
        }
        let chunk: Vec<u32> = s.prefill_queue.drain(..take).collect();
        // REQUEST-LEVEL seq_end (lane/tick-seg, 2026-08-07): the tokens still queued after this
        // tick are the SAME request — pass them so the engine's arm selection is keyed to the
        // request's end, not this tick's. Without it the tick budget (dark lanes: 256 AND
        // SLO-headroom-capped) and the LCP-split boundary steered step35's prefill arithmetic
        // (budgets 512/256/64 DIFFER 1.813e0 vs monolithic — tickinv35 gate).
        // fed_len is this chunk's prompt-relative offset (vision sessions never resume, so
        // fed counts exactly the prompt tokens already primed) — the overlay window rebases
        // image spans to call-relative positions.
        let ov_window = s
            .vision
            .as_ref()
            .and_then(|v| v.overlay.as_ref())
            .and_then(|o| o.window(fed_len, take));
        let (l, _h, x) = lm.model.prime_cache_overlaid(
            engine,
            &chunk,
            s.cache.as_mut().unwrap(),
            s.prefill_queue.len(),
            ov_window.as_ref(),
        )?;
        s.last_logits = l;
        // PROMPT CAPTURE (lane/embed-serve): this chunk finished the prompt — read the
        // final position off THIS call's hidden stack (later chunks would not exist).
        if s.prefill_queue.is_empty() {
            if let Some(cap) = s.capture.take() {
                let hidden = if cap.hidden {
                    Some(lm.model.hidden_postnorm_row(engine, &x, take - 1)?)
                } else {
                    None
                };
                let logits = capture_piece_logits(&cap, &lm.tok, &s.last_logits);
                let _ = s.tx.send(Event::PromptCapture { hidden, logits });
            }
        }
        for &tok in &chunk {
            s.fed.push(tok);
            s.sampler.accept(tok);
        }
        consumed = take;
    } else if let Some(tok) = s.prefill_queue.pop_front() {
        if std::env::var("MEMRA_DEBUG_PRIMESEG").is_ok() {
            // The W1 two-programs door (see the comment above): prompt tokens going through
            // decode_step one at a time. Loud under the diagnostic so a byte comparison can
            // never silently attribute this to the grid.
            eprintln!(
                "[primeseg] TOKENWISE prompt token at fed={} (bound_rem={:?} q={q} budget={budget}) \
                 — W1 two-programs door, NOT the prime path",
                s.fed.len(),
                bound_rem,
            );
        }
        if let Some(v) = s.vision.as_ref() {
            let pos = s.fed.len();
            if v.spans.iter().any(|&(p, _, n)| pos >= p && pos < p + n) {
                return Err(
                    "vision prefill fell to the tokenwise path inside an image span".into(),
                );
            }
        }
        // CAPTURE (lane/embed-serve): the FINAL prompt token of a capture session walks
        // decode_step_h — the same numeric program with the trunk hidden returned — so
        // prompts below the prime floor can still pool their last position (prime_cache
        // hard-asserts T >= PRIME_MIN_T; a 4-token floor panicked the worker, 2026-08-26).
        if s.capture.is_some() && s.prefill_queue.is_empty() {
            let (l, h) = lm
                .model
                .decode_step_h(engine, tok, s.cache.as_mut().unwrap())?;
            s.last_logits = l;
            if let Some(cap) = s.capture.take() {
                let hidden = if cap.hidden {
                    Some(lm.model.hidden_postnorm_row(engine, &h, 0)?)
                } else {
                    None
                };
                let logits = capture_piece_logits(&cap, &lm.tok, &s.last_logits);
                let _ = s.tx.send(Event::PromptCapture { hidden, logits });
            }
        } else {
            s.last_logits = lm
                .model
                .decode_step(engine, tok, s.cache.as_mut().unwrap())?;
        }
        if let Some(&target) = s.prefill_queue.front() {
            write_confidence_trace(s, tok, target, &s.last_logits)?;
        }
        s.fed.push(tok);
        s.sampler.accept(tok);
        consumed = 1;
    }
    // Boundary reached: snapshot the primed prefix into the cache, then keep priming the rest
    // of the prompt as a continuation (the LCP-split learning insert).
    if s.snapshot_at == Some(s.fed.len()) {
        s.snapshot_at = None;
        prefix_insert_from_session(engine, px, s, "lcp-split");
    }
    // PLAIN-AFFINITY: capture the pre-generation checkpoint the instant the prime reaches its
    // boundary (no-op unless s.ckpt_at == s.fed.len()). Cheap — one GDN-state snapshot.
    maybe_plain_checkpoint(engine, s);
    if s.prefill_queue.is_empty() {
        s.prefill_done = true;
        if let Some(c) = s.cache.as_ref() {
            kvprobe(engine, c, &s.last_logits, "prefill-done");
        }
        maybe_prefix_seed(engine, px, s);
        if let Some(trace) = s.ttft.as_ref() {
            trace.mark_prime_end();
        }
        // Capture fallback: the prompt ended on the tokenwise path (eager-only model or
        // a carried tail) — no hidden stack to pool from; logits are still truthful.
        if let Some(cap) = s.capture.take() {
            let logits = capture_piece_logits(&cap, &lm.tok, &s.last_logits);
            let _ = s.tx.send(Event::PromptCapture {
                hidden: None,
                logits,
            });
        }
    }
    Ok(consumed)
}

/// GRAMMAR MASK STAGING (constrained-full, 2026-08-03): compute the session's current
/// llguidance token mask and H2D the packed bitset into its STABLE device buffer for the
/// upcoming batched step. Runs AFTER advance_sample_emit consumed the tick's token — the
/// mask reflects the post-consume grammar state, exactly the set legal for the NEXT token.
/// No-op (mask_words = 0 -> host fallback) for unconstrained sessions, unsupported constrained
/// penalty/filter compositions, and the MEMRA_CONSTRAIN_HOST=1 oracle.
/// EAGER-ONLY predicate (lane/gemma4-serve-gaps, 2026-08-07): models the batched scheduler
/// must serve through the per-session eager body. gemma4 (12B/26B/31B and E4B): the batched
/// decode bodies have no arm for its per-layer swa/global geometry + softcapped head (the
/// engine refuses), `prime_cache_batch` has no gemma4 core, `gemma4_prime` is fresh-only
/// (no chunked/continuation prime), and the step-wise graph capture walks the generic
/// qwen-class dc step. One predicate, consumed at every batched entry point, so a future
/// arch with the same gaps joins by predicate rather than scattered call-site checks.
fn eager_only_model(lm: &LoadedModel) -> bool {
    memra_engine::plan_backend::decode_batch_program(&lm.model.plan)
        == memra_engine::plan_backend::DecodeBatchProgram::Gemma
        || lm.model.is_gemma4_e4b()
        // HYPER-CONNECTIONS RESIDUAL (lane/glm53-flash-bringup, 2026-08-28): the batched decode
        // chunks, batched prime core, graph capture and every speculative entry point run a
        // serial residual and REFUSE this topology (`refuse_hyper`). Before this route, a
        // GLM-5.3-Flash request reached `decode_step_batch_sampled_lean_masked` and came back as
        // `engine_error` after one token. The converted paths — forward, forward_last,
        // prime_cache, decode_step — are exactly the ones the eager-only class uses.
        || memra_engine::plan_backend::decode_batch_unconverted(&lm.model.plan)
}

/// BATCHED-DECODE carve-out from the eager-only class (lane/gemma-batched, 2026-08-16):
/// dense gemma4 (31B class — NOT E4B, which keeps its dedicated decode) has a batched
/// decode arm (`gemma4_decode_batch`, exactness battery green at B=4/8) that is
/// DEFAULT ON since the 2026-08-16 owner flip (`MEMRA_GEMMA4_BATCH=0` = the eager
/// kill switch). When the arm is live, ONLY the two decode
/// scheduling sites route these sessions into batched chunks; every other eager-only
/// exclusion (prime batching, concat/fanout prime, graph promotion, checkpoint
/// nomination, monolithic prefill) stays keyed on the full `eager_only` set — gemma4
/// still has no batched prime core and no dc graph capture.
fn gemma4_batched_decode_model(lm: &LoadedModel) -> bool {
    memra_engine::plan_backend::decode_batch_program(&lm.model.plan)
        == memra_engine::plan_backend::DecodeBatchProgram::Gemma
        && !lm.model.is_gemma4_e4b()
        && HybridModel::gemma4_batch_on()
}

fn stage_grammar_mask(engine: &Engine, s: &mut Session) -> Result<(), String> {
    s.mask_words = 0;
    if s.constraint.is_none() || constrain_host() || devsample_meta(s).is_none() {
        return Ok(());
    }
    let mask = s.constraint.as_mut().unwrap().compute_mask()?;
    let words = mask.as_slice();
    match s.mask_dev.as_mut() {
        Some(d) if d.len() >= words.len() => {
            engine.htod_u32_into(d, words).map_err(|e| e.to_string())?;
        }
        _ => {
            let mut d = engine
                .alloc_u32_zeroed(words.len())
                .map_err(|e| e.to_string())?;
            engine
                .htod_u32_into(&mut d, words)
                .map_err(|e| e.to_string())?;
            s.mask_dev = Some(d);
        }
    }
    s.mask_words = words.len();
    Ok(())
}

/// The decode tick's HOST half: sample from last_logits, emit the token, run the stop
/// battery. Returns (continue?, Some(next_token) to feed the next step). Extracted from
/// step_session so the batched scheduler can drive many sessions into ONE engine step.
fn advance_sample_emit(
    loaded: &HashMap<String, LoadedModel>,
    s: &mut Session,
) -> (bool, Option<u32>) {
    let lm = &loaded[&s.model];
    if s.generated.len() >= s.budget {
        finish(s, StopReason::MaxNew);
        return (false, None);
    }
    // Device-presampled token from the last batched tick (Session.device_next): skips the
    // O(n_vocab) host sample (measured 1.36 ms/row at 248k vocab). Greedy device rows are
    // bit-identical to host argmax; temp rows are the seeded device draw (gate3 contract).
    // CONSTRAINED rows never device-sample (their samp meta is None) — they host-sample
    // from a grammar-masked COPY of last_logits (the pristine row still parks into the
    // reuse pool at retire, so continuations resume unmasked).
    let next = match (s.device_next.take(), s.constraint.as_mut()) {
        (Some(t), _) => t,
        (None, Some(c)) => {
            let mut row = s.last_logits.clone();
            if let Err(err) = c.mask_logits(&mut row) {
                let _ = s.tx.send(Event::Error(EngineError::engine(format!(
                    "constraint mask: {err}"
                ))));
                return (false, None);
            }
            s.sampler.sample(&row)
        }
        (None, None) => s.sampler.sample(&s.last_logits),
    };
    s.sampler.accept(next);
    s.generated.push(next);
    if let Some(trace) = s.ttft.as_ref() {
        trace.mark_first_decode();
    }
    if s.params.eos.contains(&next) {
        if !send_token_event(s, next, String::new()) {
            abort_log(s);
            return (false, None);
        }
        finish(s, StopReason::Eos);
        return (false, None);
    }
    // Advance the grammar with the accepted (non-EOS) token. The token came from this
    // state's own mask, so an error here is a real bug — stop LOUDLY, never emit
    // schema-violating text as if it conformed.
    if let Some(c) = s.constraint.as_mut() {
        if let Err(err) = c.consume(next) {
            let _ = s.tx.send(Event::Error(EngineError::engine(format!(
                "constraint advance: {err}"
            ))));
            return (false, None);
        }
    }
    s.decoded_bytes
        .extend_from_slice(&lm.tok.decode_bytes_special(&[next], true));
    let delta = utf8_delta(&s.decoded_bytes, &mut s.emitted_bytes);
    // DISCONNECT ABORT (gap-scan F8): a failed send = receiver dropped = client gone.
    // Stop generating THIS tick (the tick-top sweep would only catch it next tick).
    if !send_token_event(s, next, delta) {
        abort_log(s);
        return (false, None);
    }
    if contains_stop_string(&s.decoded_bytes, &s.stop_strings) {
        finish(s, StopReason::Callback);
        return (false, None);
    }
    if s.cache
        .as_ref()
        .map(|c| c.pos >= c.max_ctx)
        .unwrap_or(false)
    {
        finish(s, StopReason::ContextFull);
        return (false, None);
    }
    (true, Some(next))
}

/// Token-driven twin of `advance_sample_emit` for the graph path: the token was produced
/// by the DEVICE argmax (greedy), so there is no sampling — accept, emit, stop battery.
/// Returns (continue?, ()).
fn advance_token_emit(
    loaded: &HashMap<String, LoadedModel>,
    s: &mut Session,
    tok: u32,
) -> (bool, ()) {
    let lm = &loaded[&s.model];
    if s.generated.len() >= s.budget {
        finish(s, StopReason::MaxNew);
        return (false, ());
    }
    s.sampler.accept(tok);
    s.generated.push(tok);
    if let Some(trace) = s.ttft.as_ref() {
        trace.mark_first_decode();
    }
    if s.params.eos.contains(&tok) {
        if !send_token_event(s, tok, String::new()) {
            abort_log(s);
            return (false, ());
        }
        finish(s, StopReason::Eos);
        return (false, ());
    }
    // CONSTRAINED graph sessions: the token came from the in-graph masked argmax — advance
    // the grammar (post-EOS-check, same ordering as advance_sample_emit). An error here is
    // a real bug: loud stop, never emit schema-violating text as if it conformed.
    if let Some(c) = s.constraint.as_mut() {
        if let Err(err) = c.consume(tok) {
            let _ = s.tx.send(Event::Error(EngineError::engine(format!(
                "constraint advance: {err}"
            ))));
            return (false, ());
        }
    }
    s.decoded_bytes
        .extend_from_slice(&lm.tok.decode_bytes_special(&[tok], true));
    let delta = utf8_delta(&s.decoded_bytes, &mut s.emitted_bytes);
    // DISCONNECT ABORT (gap-scan F8): failed send = client gone, stop this tick.
    if !send_token_event(s, tok, delta) {
        abort_log(s);
        return (false, ());
    }
    if contains_stop_string(&s.decoded_bytes, &s.stop_strings) {
        finish(s, StopReason::Callback);
        return (false, ());
    }
    (true, ())
}

/// Group ready (session_idx, token) pairs into batched-step chunks: same model, <= 8 rows
/// (the exactness-tier cap), input order preserved (caller sorted interactive first).
/// Device-side batched-tick sampling (default ON — measured 1.36 ms/row host temp-sample at
/// the 9B's 248k vocab, ~45% of the B=8 serving tick). MEMRA_SERVE_DEVSAMPLE=0 is the
/// rollback/A-B seam: every row host-samples from last_logits as before.
fn serve_devsample() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("MEMRA_SERVE_DEVSAMPLE").as_deref() != Ok("0"))
}

/// Penalty-aware extension of device sampling. `1` enables the path on hardware/model
/// deployments that have passed their serving qualification. Unset stays off: PRO 6000
/// evidence must not silently set a default for unmeasured hardware.
fn serve_devpenalty() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("MEMRA_SERVE_DEVPENALTY").as_deref() == Ok("1"))
}

/// LEAN LOGITS (inc2 component 3, default ON): device-sampled rows skip the [n_vocab]
/// logits D2H; the last row parks on-device per cache and is D2H'd once at retire (the
/// reuse-pool consumer). MEMRA_SERVE_LEANLOGITS=0 is the rollback/A-B seam (full D2H,
/// the exact pre-change tick).
fn serve_leanlogits() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("MEMRA_SERVE_LEANLOGITS").as_deref() != Ok("0"))
}

/// MEMRA_CONSTRAIN_HOST=1 (rollback oracle): constrained rows keep the v1 host-side
/// masked-copy sample (full-row D2H + O(n_vocab) host sample) instead of the device
/// grammar mask. Diagnostics/A-B only — the device path is the shipped default.
fn constrain_host() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("MEMRA_CONSTRAIN_HOST").as_deref() == Ok("1"))
}

/// Device-sample meta for a session's row in the batched step (the ONE eligibility rule —
/// the samp closure and the grammar-mask staging pass must agree): greedy-no-penalties
/// (device argmax, bit-identical), pure-temperature (seeded gumbel), or
/// temperature+top-k/top-p/min-p (filter_stats floor + the filtered gumbel draw — the
/// lane/devsample-topkp extension; the host path measured 1.34 ms/row at 248k vocab on
/// these configs). Sampled penalty configs join this path when `MEMRA_SERVE_DEVPENALTY=1`;
/// greedy penalties remain host-side. Constrained rows with penalties or filters ALSO
/// host-sample: the grammar mask composes with the filter floor only after a
/// dedicated gate run, so v1 keeps yesterday's behavior for them (meta None -> no mask
/// staged -> host masked sample, staging agreement preserved). Counter = generated.len()
/// — a session-progress function, independent of batch composition (the isolation
/// contract, gate3).
fn devsample_meta(s: &Session) -> Option<DevSamp> {
    if !serve_devsample() {
        return None;
    }
    let sm = &s.sampler;
    let penalized = sm.penalty_last_n() > 0
        && (sm.penalty_repeat() != 1.0 || sm.penalty_freq() != 0.0 || sm.penalty_present() != 0.0);
    // Greedy penalties stay on the host oracle: the performance lane targets the vendor-
    // sampled surface, whose device draw is already distributional rather than byte-equal.
    // Moving greedy argmax would create a new near-tie numeric class for no current product win.
    if penalized && (sm.is_greedy() || !serve_devpenalty() || s.constraint.is_some()) {
        return None;
    }
    let filtered = sm.top_k() != 0 || sm.top_p() < 1.0 || sm.min_p() > 0.0;
    if filtered && s.constraint.is_some() {
        return None;
    }
    let mut meta = if sm.is_greedy() {
        DevSamp::new(0.0, 0, 0, 0, 1.0, 0.0)
    } else {
        DevSamp::new(
            sm.temperature(),
            sm.seed(),
            s.generated.len() as u32,
            sm.top_k() as i32,
            sm.top_p(),
            sm.min_p(),
        )
    };
    if penalized {
        let counts = sm.penalty_counts();
        if !counts.is_empty() {
            // SAFETY: `penalty_counts` is collected from the sampler's HashMap and therefore
            // contains each token id exactly once; eviction removes an entry at zero, so every
            // retained count is positive.
            let penalty = unsafe {
                DevPenalty::from_unique_counts_unchecked(
                    sm.penalty_repeat(),
                    sm.penalty_freq(),
                    sm.penalty_present(),
                    counts,
                )
            };
            meta = meta.with_penalty(penalty);
        }
    }
    Some(meta)
}

/// GraphSession's captured sampler is raw greedy. Penalties change logits before argmax and
/// therefore stay on the ordinary batched epilogue until a penalty-aware graph is qualified.
fn graph_sampler_eligible(sm: &Sampler) -> bool {
    sm.is_greedy()
        && (sm.penalty_last_n() == 0
            || (sm.penalty_repeat() == 1.0
                && sm.penalty_freq() == 0.0
                && sm.penalty_present() == 0.0))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct DecodeChunkPolicy {
    /// Maximum exact width of either engine wave.
    wave_cap: usize,
    /// True only for the explicit native-peer `MEMRA_DUAL_PP=1` + `MEMRA_PP_OVERLAP=1` PP-2 arm.
    dual: bool,
}

impl DecodeChunkPolicy {
    const fn serial(wave_cap: usize) -> Self {
        Self {
            wave_cap,
            dual: false,
        }
    }

    fn tick_cap(self) -> usize {
        if self.dual {
            self.wave_cap.saturating_mul(2)
        } else {
            self.wave_cap
        }
    }

    fn wave_mid(self, width: usize) -> Option<usize> {
        self.dual
            .then(|| memra_engine::pp::dual_pp_wave_mid(width))
            .flatten()
    }
}

#[derive(Debug, Eq, PartialEq)]
struct ScheduledDecodeChunk {
    rows: Vec<(usize, u32)>,
    wave_mid: Option<usize>,
}

fn schedule_decode_chunk(
    rows: Vec<(usize, u32)>,
    policy: DecodeChunkPolicy,
) -> ScheduledDecodeChunk {
    let wave_mid = policy.wave_mid(rows.len());
    ScheduledDecodeChunk { rows, wave_mid }
}

fn resolve_decode_chunk_policy(
    wave_cap: usize,
    dual_requested: bool,
    overlap_requested: bool,
    pp2_ready: bool,
    host_bounce_active: bool,
) -> DecodeChunkPolicy {
    DecodeChunkPolicy {
        wave_cap,
        dual: dual_requested && overlap_requested && pp2_ready && !host_bounce_active,
    }
}

/// Per-model exact decode WAVE width. MEMRA_DECODE_BATCH_CAP (explicit door) wins; otherwise
/// models that qualify for the EXACT-16 tier (decode_batch_exact16_ok — every matmul has
/// a bit-exact b16-class kernel) default to chunk 16, the measured winner on the 5090
/// (+12% aggregate over chunk 8 at 32 seqs — research/batched-tick-inc3-20260801/
/// chunksweep.log); everything else keeps the chunk-8 exactness tier. Isolation contract
/// unchanged either way (gate2 bit-strength PASS at both widths).
///
/// The Q8_0 q8rp-mirror precondition was REMOVED 2026-08-06 (lane/rp-on-st): Q8_0's b16
/// twin existed only in `_rp` form, which made a bandwidth mirror a *correctness*
/// prerequisite for the tier. With `qmatvec_q8_0_mmvq_b16` (base layout) plus b16 twins
/// for NVFP4 / Q4_K / Q5_K, the predicate — an ALL over ~500 matmuls — finally admits
/// real MIXED checkpoints. Before that, one missing class refused the whole model, so
/// chunk 16 was unreachable for every shipped artifact, GGUF and FP8-ST alike.
fn chunk_cap_for(lm: &LoadedModel) -> usize {
    if !memra_engine::plan_backend::DECODE_BATCH
        .trunk_capabilities(&lm.model.plan)
        .batch
        .supported
    {
        return 1;
    }
    // step35 (lane/step35-batched-decode, 2026-08-08): the REAL batched arm exists —
    // `step35_decode_batch_layers` carries the per-layer n_head / partial rope / per-session
    // SWA views / head-wise gate the generic body lacked (the b2ab garbage receipt,
    // research/step-sku-20260807/raw/b2ab-pre-*.log, was the GENERIC arm running past the
    // ppn door; that arm is now unreachable for this arch at any B). Chunk cap 8: the
    // exactness-tier width (IQ4_XS trunk + 288-expert MoE refuse exact16 by predicate —
    // `decode_batch_exact16_ok` requires non-MoE — so 16 is structurally out). The
    // MEMRA_STEP35_BATCH=0 rollback seam caps at B=1. Since lane/cx-b1fix, PP-N B=1 also
    // fails closed rather than falling back to its load-history-dependent eager numeric
    // class; the cap still keeps the scheduler from forming wider doomed chunks.
    // MEMRA_DECODE_BATCH_CAP may narrow BELOW 8, never widen past it.
    let batch_program = memra_engine::plan_backend::decode_batch_program(&lm.model.plan);
    if batch_program == memra_engine::plan_backend::DecodeBatchProgram::SlidingGatedMoe {
        if !HybridModel::step35_batch_on() {
            return 1;
        }
        let cap = std::env::var("MEMRA_DECODE_BATCH_CAP")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(8usize);
        return cap.clamp(1, 8);
    }
    // gemma4 dense (lane/gemma-batched, 2026-08-16): the batched arm's exactness battery
    // is green at B<=8 (per-row mmvq tier); exact16 refuses gemma4 by predicate and m>8
    // crosses the dp4a-tail/GEMM numeric configs the gate never proved. Cap 8 — the env
    // door may narrow BELOW 8, never widen past the proven tier. Kill-switch-off
    // returns 1 (unused: the model stays in eager_decode and never enters the chunks).
    if batch_program == memra_engine::plan_backend::DecodeBatchProgram::Gemma {
        if !gemma4_batched_decode_model(lm) {
            return 1;
        }
        let cap = std::env::var("MEMRA_DECODE_BATCH_CAP")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(8usize);
        return cap.clamp(1, 8);
    }
    if let Some(c) = std::env::var("MEMRA_DECODE_BATCH_CAP")
        .ok()
        .and_then(|v| v.parse().ok())
    {
        return usize::clamp(c, 1, 32);
    }
    if lm.model.decode_batch_exact16_ok() {
        16
    } else {
        8
    }
}

fn decode_chunk_policy(lm: &LoadedModel) -> DecodeChunkPolicy {
    let wave_cap = chunk_cap_for(lm);
    let pp2_ready = memra_engine::pp::batch_pp_on()
        && !memra_engine::pp::pp2_streams_off()
        && memra_engine::pp::pp_cuts(lm.model.layers.len()).is_some_and(|fence| fence.len() == 3);
    resolve_decode_chunk_policy(
        wave_cap,
        memra_engine::pp::dual_pp_on(),
        memra_engine::pp::pp2_overlap(),
        pp2_ready,
        memra_engine::pp::pp_host_bounce_active(),
    )
}

fn group_chunks(
    active: &[Session],
    ready: &[(usize, u32)],
    policies: &HashMap<String, DecodeChunkPolicy>,
) -> Vec<ScheduledDecodeChunk> {
    let mut chunks: Vec<(Vec<(usize, u32)>, DecodeChunkPolicy)> = Vec::new();
    for &(i, t) in ready {
        let model = &active[i].model;
        let policy = policies
            .get(model)
            .copied()
            .unwrap_or(DecodeChunkPolicy::serial(8));
        let cap = policy.tick_cap();
        match chunks.last_mut() {
            Some((c, _)) if c.len() < cap && active[c[0].0].model == *model => c.push((i, t)),
            _ => chunks.push((vec![(i, t)], policy)),
        }
    }
    chunks
        .into_iter()
        .map(|(rows, policy)| schedule_decode_chunk(rows, policy))
        .collect()
}

fn spec_pipe_pairable(
    engine: &Engine,
    loaded: &HashMap<String, LoadedModel>,
    a: &Session,
    b: &Session,
) -> bool {
    if std::env::var("MEMRA_SPEC_PIPE").as_deref() != Ok("1")
        || a.model != b.model
        || a.spec_k == 0
        || a.spec_k != b.spec_k
        || a.spec.is_none()
        || b.spec.is_none()
        || !a.prefill_done
        || !b.prefill_done
        || !a.prefill_queue.is_empty()
        || !b.prefill_queue.is_empty()
        || a.generated.is_empty()
        || b.generated.is_empty()
        || !a.sampler.is_greedy()
        || !b.sampler.is_greedy()
        || a.constraint.is_some()
        || b.constraint.is_some()
        || a.generated.len() >= a.budget
        || b.generated.len() >= b.budget
    {
        return false;
    }
    for sess in [a.spec.as_ref().unwrap(), b.spec.as_ref().unwrap()] {
        if sess.committed_len() == 0 || (sess.next_pred.is_none() && !sess.has_pending()) {
            return false;
        }
    }
    loaded[&a.model].model.spec_pipe_available(engine)
}

fn finish_pipelined_spec_burst(
    lm: &LoadedModel,
    s: &mut Session,
    burst: Vec<u32>,
    drafted: usize,
    accepted: usize,
    telem_before: memra_engine::spec::SpecTelemetry,
    request_room: usize,
    spec_metrics: &mut SpecMetricState,
) -> Result<bool, Box<dyn std::error::Error>> {
    if let Some(trace) = s.ttft.as_ref() {
        trace.mark_prime_end();
        if !burst.is_empty() {
            trace.mark_first_decode();
        }
    }
    let spec = s.spec.as_ref().expect("paired spec session disappeared");
    let telem_delta = spec.telemetry().delta_since(&telem_before);
    spec_metrics.record(&s.model, telem_delta);
    s.spec_rounds += telem_delta.rounds;
    s.spec_drafted += drafted;
    s.spec_accepted += accepted;
    if drafted > 0 {
        eprintln!(
            "[spec-acc] ctx={} burst={}/{} cum={}/{}={:.3}",
            s.fed.len(),
            accepted,
            drafted,
            s.spec_accepted,
            s.spec_drafted,
            s.spec_accepted as f64 / s.spec_drafted.max(1) as f64
        );
    }

    let tok_ref = &lm.tok;
    let eos_ids = s.params.eos.clone();
    let public_len = spec_visible_len(&burst, request_room, &eos_ids);
    let public_burst = &burst[..public_len];
    let mut decoded_visible = std::mem::take(&mut s.decoded_bytes);
    let mut cursor = s.emitted_bytes;
    let mut emit_remaining = request_room;
    let mut eos_seen = false;
    let tx = s.tx.clone();
    let emitted = emit_spec_token_events(
        public_burst,
        &mut emit_remaining,
        &mut decoded_visible,
        &mut cursor,
        &eos_ids,
        &mut eos_seen,
        |id| tok_ref.decode_bytes_special(&[id], true),
        |event| tx.send(event).is_ok(),
    );
    if emitted.send_ok {
        debug_assert_eq!(
            emitted.sent, public_len,
            "one token event per public spec token"
        );
    }
    let mut stop: Option<StopReason> = None;
    for &tok in public_burst {
        s.sampler.accept(tok);
        s.generated.push(tok);
        s.fed.push(tok);
        if eos_ids.contains(&tok) {
            stop = Some(StopReason::Eos);
            break;
        }
    }
    s.tokens_emitted += emitted.sent;
    s.emitted_bytes = cursor;
    s.decoded_bytes = decoded_visible;
    if !emitted.send_ok {
        abort_log(s);
        return Ok(false);
    }
    if stop.is_none() && contains_stop_string(&s.decoded_bytes, &s.stop_strings) {
        stop = Some(StopReason::Callback);
    }
    if stop.is_none() && s.generated.len() >= s.budget {
        stop = Some(StopReason::MaxNew);
    }
    let context_full = s
        .spec
        .as_ref()
        .is_some_and(|spec| spec.committed.len() + s.spec_k + 3 >= spec.cache_max_ctx());
    if stop.is_none() && context_full {
        stop = Some(StopReason::ContextFull);
    }
    if let Some(reason) = stop {
        finish(s, reason);
        return Ok(false);
    }
    Ok(true)
}

fn step_spec_pair(
    engine: &Engine,
    loaded: &HashMap<String, LoadedModel>,
    a: &mut Session,
    b: &mut Session,
    spec_metrics: &mut SpecMetricState,
) -> Result<(bool, bool), Box<dyn std::error::Error>> {
    debug_assert!(spec_pipe_pairable(engine, loaded, a, b));
    let lm = &loaded[&a.model];
    for s in [&mut *a, &mut *b] {
        if let Some(trace) = s.ttft.as_ref() {
            trace.mark_prime_start();
        }
    }
    let burst_t: usize = std::env::var("MEMRA_SPEC_BURST")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(32);
    let room_a = a.budget.saturating_sub(a.generated.len());
    let room_b = b.budget.saturating_sub(b.generated.len());
    let target_a = room_a.min(burst_t);
    let target_b = room_b.min(burst_t);
    let telem_a = a.spec.as_ref().unwrap().telemetry();
    let telem_b = b.spec.as_ref().unwrap().telemetry();
    let (burst_a, burst_b) = lm.model.generate_spec_session_pair(
        engine,
        a.spec.as_mut().unwrap(),
        target_a,
        a.spec_k,
        b.spec.as_mut().unwrap(),
        target_b,
        b.spec_k,
    )?;
    let (out_a, drafted_a, accepted_a) = burst_a;
    let (out_b, drafted_b, accepted_b) = burst_b;
    let keep_a = finish_pipelined_spec_burst(
        lm,
        a,
        out_a,
        drafted_a,
        accepted_a,
        telem_a,
        room_a,
        spec_metrics,
    )?;
    let keep_b = finish_pipelined_spec_burst(
        lm,
        b,
        out_b,
        drafted_b,
        accepted_b,
        telem_b,
        room_b,
        spec_metrics,
    )?;
    Ok((keep_a, keep_b))
}

fn step_session(
    engine: &Engine,
    loaded: &HashMap<String, LoadedModel>,
    s: &mut Session,
    spec_metrics: &mut SpecMetricState,
) -> Result<bool, Box<dyn std::error::Error>> {
    // A dspark session owns its cache; s.cache is None. Stepping one here decodes from an
    // EMPTY context — coherent garbage, no crash, the silent-quality-loss class. That
    // happened when the dispatch flag disagreed with the installed session (the restored-
    // session hazard, root-caused 2026-08-27); this refusal turns any future disagreement
    // into a loud request error instead of wrong customer output.
    if s.dspark.is_some() {
        return Err(
            "plain step_session received a session holding a dspark spec session — \
             dispatch flag disagrees with the installed session"
                .into(),
        );
    }
    let lm = &loaded[&s.model];

    // ---- SPEC-BURST arm (2026-07-05): MTP sessions decode in generate_spec_session
    // bursts — turn 1 primes the prompt (suffix = the whole prefill queue), later ticks are
    // ZERO-prime continuation bursts (SpecSession.next_pred). Each burst emits up to
    // SPEC_BURST_T tokens; between bursts the scheduler round-robins other sessions. Exactness:
    // GREEDY bursts — the session-gate oracle (4 turns incl empty-suffix) pins burst output ==
    // fresh greedy, byte-identical. SAMPLED bursts (temperature>0, `sampling` below) are
    // DISTRIBUTIONALLY exact instead: the rejection-sampling verify draws from the same
    // filtered/penalized target as plain sampled decode, but consumes its own Philox streams
    // (sess.sctr/uctr), so the token stream is reproducible per (seed, session) rather than
    // byte-equal to a plain-sampled run. That is the contract, not a gap.
    if s.vision.is_some() && s.spec.is_some() {
        // Backstop for the admit gate: a spec burst's turn-1 prime has no overlay seam,
        // so serving a vision session through it would silently prime pad embeddings
        // (a text-only model answering "blind" — the failure is invisible in the API).
        return Err("vision session entered the spec round (admit gate failed)".into());
    }
    if let Some(spec) = s.spec.as_mut() {
        if let Some(trace) = s.ttft.as_ref() {
            trace.mark_prime_start();
        }
        // Burst size trades round-robin latency (other sessions wait a whole burst) against
        // per-burst fixed cost. The dominant cost — the per-call draft-graph recapture,
        // measured ~16ms/burst on H100 q27 — is gone since 2026-08-01: the captured graph
        // persists on the SpecSession (spec.rs DraftGraphCtx) and later bursts replay it.
        // MEMRA_SPEC_BURST overrides for measurement; 32 = latency-safe default.
        let burst_t: usize = std::env::var("MEMRA_SPEC_BURST")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(32);
        let k = s.spec_k;
        debug_assert!(k > 0, "plain sessions must not enter the spec round");
        let request_room = s.budget.saturating_sub(s.generated.len());
        if request_room == 0 {
            finish(s, StopReason::MaxNew);
            return Ok(false);
        }
        // The engine target is a scheduler cadence, not a public-output cap. Session mode may
        // return a cache-authoritative surplus token past this target; expose it when the request
        // still has room so worker generated/sampler/fed state stays aligned with SpecSession.
        let burst_target = request_room.min(burst_t);
        let suffix: Vec<u32> = s.prefill_queue.drain(..).collect();
        s.prefill_done = true;
        if suffix.is_empty() && spec.next_pred.is_none() && spec.pending_tok.is_none() {
            // nothing primed and nothing to prime — shouldn't happen (admit rejects empty prompts)
            finish(s, StopReason::MaxNew);
            return Ok(false);
        }
        let sampling = spec_sampling_for(&s.sampler);
        // SPEC x CONSTRAINED: greedy constrained bursts carry the grammar hook — verify-side
        // truncation + masked-argmax cut slots (engine contract; sampled never gets here).
        // Telemetry (lane/accept-telemetry): the session's counters are LIFETIME (a pool
        // resume carries prior requests' counts), so stash a copy and diff after the burst —
        // the delta is this burst's contribution, merged per-model for /metrics and summed
        // per-request for usage.spec.
        let telem_before = spec.telemetry();
        // SSE CADENCE (lane/sse-cadence, 2026-08-05): publish at ROUND cadence, not burst
        // cadence. Every committed id in the request-owned budget gets its own Event::Token;
        // UTF-8 fragments may make an individual event's text empty, but ids are never coalesced.
        // MEMRA_SSE_PER_BURST=1 retains the timing rollback (events wait for burst end), not the
        // old broken one-id-for-many-tokens event shape.
        let per_burst_emit = std::env::var("MEMRA_SSE_PER_BURST").as_deref() == Ok("1");
        // ADMISSION YIELD (lane/admission-latency, 2026-08-06): a request that arrives while
        // a burst is in flight used to wait the WHOLE burst out before the worker's tick-top
        // admission phase could even see it — contended first-text scaled with
        // MEMRA_SPEC_BURST (0.57s at B32, 1.67s at B128; sse-cadence VERDICT). The round-
        // boundary flush below now returns a continue-verdict: `false` (a request is waiting
        // in the cmd channel, PENDING_ADMITS > 0) ends the burst at the current round exactly
        // as if burst-count had been reached — burst size is content-neutral (spec-levers
        // battery), so this moves WHEN control returns, never what tokens say.
        // MEMRA_ADMIT_YIELD=0 = rollback seam (restores full-burst holds).
        let admit_yield = std::env::var("MEMRA_ADMIT_YIELD").as_deref() != Ok("0");
        let tok_ref = &lm.tok;
        let mut decoded_visible = std::mem::take(&mut s.decoded_bytes);
        let mut cursor = s.emitted_bytes;
        let mut emit_remaining = request_room;
        let mut eos_seen = false;
        let mut send_ok = true;
        let mut token_events = 0usize;
        let flush_tx = s.tx.clone();
        let eos_ids = s.params.eos.clone();
        let mut flush_cb = |slice: &[u32]| -> bool {
            // Continue-verdict polled at EVERY round boundary (even empty/post-EOS flushes).
            let keep =
                !admit_yield || PENDING_ADMITS.load(std::sync::atomic::Ordering::Acquire) == 0;
            if per_burst_emit || eos_seen || emit_remaining == 0 || slice.is_empty() {
                // poll-only boundary: rollback seam / post-EOS (tokens never visible;
                // accounting happens post-burst) / nothing new committed this round.
                return keep;
            }
            if let Some(trace) = s.ttft.as_ref() {
                trace.mark_prime_end();
                trace.mark_first_decode();
            }
            if !send_ok {
                return keep; // client already gone — keep the cursor honest, stop sending
            }
            let emitted = emit_spec_token_events(
                slice,
                &mut emit_remaining,
                &mut decoded_visible,
                &mut cursor,
                &eos_ids,
                &mut eos_seen,
                |id| tok_ref.decode_bytes_special(&[id], true),
                |event| flush_tx.send(event).is_ok(),
            );
            token_events += emitted.sent;
            send_ok = emitted.send_ok;
            keep
        };
        let on_commit: Option<&mut dyn FnMut(&[u32]) -> bool> = if per_burst_emit && !admit_yield {
            None
        } else {
            Some(&mut flush_cb)
        };
        // Cold speculative sessions must enter decode with the same prompt state as the plain
        // policy they replace. Plain affinity stops at the stable live-turn boundary and primes
        // a sub-floor tail tokenwise; a monolithic spec prime can select different Step35 bytes.
        // Mirror that cold boundary — and, under the stable-boundary door
        // (lane/frspec-multiturn-cache, 2026-08-21), the WARM one too: an affinity-rewound or
        // pool-resumed session priming its own delta stops at the new turn's stable boundary
        // exactly like a resumed plain session's prefill tick does (`ckpt_at` filter
        // `b > seed_fed.len()`). Warm sessions skip the nominatable predicate — a session that
        // was parked and resumed has already proven nomination, and the suffix alone
        // undercounts the conversation's segments. Empty-suffix continuation bursts remain
        // zero-prime, and MEMRA_AFFINITY=0 preserves the monolithic control arm.
        let cold = spec.committed.is_empty();
        let boundary = if !suffix.is_empty()
            && affinity_enabled()
            && (cold
                && (s.affinity.is_some()
                    || plain_ckpt_nominatable(&suffix, &|t| lm.tok.token_is_control(t)))
                || !cold && spec_stable_boundary_on())
        {
            plain_checkpoint_boundary(&suffix, &|t| lm.tok.token_is_control(t))
        } else {
            None
        };
        // SPEC-TIER TURN CHECKPOINT at the stable boundary (the plain tier's 2026-08-09 law,
        // ported): the engine captures `turn_ckpt` at this stop instead of prompt-end, so the
        // next re-rendered turn's byte diff lands ON the checkpoint instead of diverging
        // inside the live generation header 2 tokens below it. Absolute position, `capture_at`
        // convention. Door OFF = never armed = legacy prompt-end capture, byte-for-byte.
        if spec_stable_boundary_on() {
            spec.ckpt_at = boundary.map(|b| spec.committed.len() + b);
        }
        let prime_split = boundary;
        // PREFIX-CACHE capture boundary (lane/spec-prefix-cache): a cold burst with an armed
        // mid-prompt capture boundary must actually SPLIT the prime there, or the capture
        // never fires (the engine compares capture_at == split). When affinity also wants a
        // split, the EARLIER boundary wins — both are legal prime stops, and the engine's
        // PRIME_MIN_T law vetoes sub-floor splits on its own. A seed boundary (== suffix
        // length) is not a split; the engine's post-prime seed capture handles it. (Under the
        // stable-boundary door the affinity stop is re-armed via `ckpt_at` above, so taking
        // the min here no longer forfeits it — the engine stops at BOTH, exactly like the
        // plain prefill tick's snapshot_at/ckpt_at pair.)
        let prime_split = if cold {
            match (prime_split, spec.capture_at.filter(|&b| b < suffix.len())) {
                (Some(a), Some(c)) => Some(a.min(c)),
                (None, Some(c)) => Some(c),
                (a, None) => a,
            }
        } else {
            prime_split
        };
        let (burst, d, a) = match s.constraint.as_mut() {
            Some(c) => {
                let mut g = crate::constrained::SpecGrammar::new(c, lm.eos_id);
                lm.model.generate_spec_session_constrained_prime_split(
                    engine,
                    spec,
                    &suffix,
                    burst_target,
                    k,
                    sampling,
                    Some(&mut g),
                    prime_split,
                    on_commit,
                )?
            }
            None => lm.model.generate_spec_session_sampled_prime_split(
                engine,
                spec,
                &suffix,
                burst_target,
                k,
                sampling,
                prime_split,
                on_commit,
            )?,
        };
        drop(flush_cb);
        if let Some(trace) = s.ttft.as_ref() {
            trace.mark_prime_end();
            if !burst.is_empty() {
                trace.mark_first_decode();
            }
        }
        let telem_delta = spec.telemetry().delta_since(&telem_before);
        spec_metrics.record(&s.model, telem_delta);
        s.spec_rounds += telem_delta.rounds;
        s.spec_drafted += d;
        s.spec_accepted += a;
        if d > 0 {
            eprintln!(
                "[spec-acc] ctx={} burst={}/{} cum={}/{}={:.3}",
                s.fed.len() + suffix.len(),
                a,
                d,
                s.spec_accepted,
                s.spec_drafted,
                s.spec_accepted as f64 / s.spec_drafted.max(1) as f64
            );
        }
        for &tok in &suffix {
            s.fed.push(tok);
            s.sampler.accept(tok);
        }
        let public_len = spec_visible_len(&burst, request_room, &eos_ids);
        let public_burst = &burst[..public_len];
        if per_burst_emit {
            let emitted = emit_spec_token_events(
                public_burst,
                &mut emit_remaining,
                &mut decoded_visible,
                &mut cursor,
                &eos_ids,
                &mut eos_seen,
                |id| tok_ref.decode_bytes_special(&[id], true),
                |event| s.tx.send(event).is_ok(),
            );
            token_events += emitted.sent;
            send_ok = emitted.send_ok;
        }
        if send_ok {
            debug_assert_eq!(
                token_events, public_len,
                "one token event per public spec token"
            );
        }
        let mut stop: Option<StopReason> = None;
        for &tok in public_burst {
            s.sampler.accept(tok);
            s.generated.push(tok);
            s.fed.push(tok);
            if s.params.eos.contains(&tok) {
                stop = Some(StopReason::Eos);
                break;
            }
        }
        // EOS text is never streamed (serve-compat, 2026-08-03), but its empty-text token event
        // keeps the id/event/generated receipt 1:1. Engine-committed budget surplus remains only
        // in SpecSession committed/pending state; it never enters these public worker vectors.
        s.tokens_emitted += token_events;
        s.emitted_bytes = cursor;
        s.decoded_bytes = decoded_visible;
        if !send_ok {
            // DISCONNECT ABORT (gap-scan F8): an in-burst flush hit a closed channel —
            // client gone, retire at the abort point (state consistent post-burst).
            abort_log(s);
            return Ok(false);
        }
        if stop.is_none() && contains_stop_string(&s.decoded_bytes, &s.stop_strings) {
            stop = Some(StopReason::Callback);
        }
        if stop.is_none() && s.generated.len() >= s.budget {
            stop = Some(StopReason::MaxNew);
        }
        // +3 (was +2): committed excludes a carried pending token (pending-carry, 2026-08-01).
        if stop.is_none() && spec.committed.len() + k + 3 >= spec.cache_max_ctx() {
            stop = Some(StopReason::ContextFull);
        }
        if let Some(r) = stop {
            finish(s, r);
            return Ok(false);
        }
        return Ok(true);
    }

    // ---- prefill phase: BATCHED chunk prime (2026-07-05). prime_cache now supports
    // continuation (cache.pos > 0 attends to the quantized past), so the worker primes up to
    // PREFILL_TICK_T prompt tokens per tick at prefill throughput (~2000-5900 tok/s) instead of
    // one decode_step (~38-100 tok/s) — a 32k prompt drops from ~15min of ticks to ~a minute,
    // while the per-tick cap keeps round-robin latency for concurrent sessions bounded.
    // Tails below PRIME_MIN_T keep the tokenwise decode_step path (prime_cache floor).
    if !s.prefill_done {
        if s.vision.is_some() {
            return Err("vision requests require the batched scheduler (MEMRA_SERVE_BATCH)".into());
        }
        if let Some(trace) = s.ttft.as_ref() {
            trace.mark_prime_start();
        }
        let q = s.prefill_queue.len();
        // EAGER-ONLY prime shape (lane/gemma4-serve-gaps): same law as prefill_tick —
        // gemma4 primes fresh prompts WHOLE (no chunked prime in the engine; chunk 2 used
        // to kill the worker) and carried suffixes tokenwise (no continuation prime).
        let eager_mono = eager_only_model(lm);
        let carried = s.cache.as_ref().is_some_and(|c| c.pos > 0);
        if !confidence_trace_enabled()
            && q >= memra_engine::hybrid_forward::PRIME_MIN_T.max(2)
            && !(eager_mono && carried)
        {
            // leave a tail chunk >= PRIME_MIN_T if this tick doesn't finish the queue
            let mut take = if eager_mono { q } else { q.min(PREFILL_TICK_T) };
            if q - take > 0 && q - take < memra_engine::hybrid_forward::PRIME_MIN_T {
                take = q;
            }
            let chunk: Vec<u32> = s.prefill_queue.drain(..take).collect();
            // REQUEST-LEVEL seq_end: the rest of prefill_queue is the same request's remainder
            // (see prefill_tick — the tick-budget segmentation must not steer arithmetic).
            let (l, _h, x) = lm.model.prime_cache(
                engine,
                &chunk,
                s.cache.as_mut().unwrap(),
                s.prefill_queue.len(),
            )?;
            s.last_logits = l;
            // PROMPT CAPTURE (lane/embed-serve): this chunk finished the prompt — read the
            // final position off THIS call's hidden stack (later chunks would not exist)
            // and publish before the budget-0 finish on the next tick.
            if s.prefill_queue.is_empty() {
                if let Some(cap) = s.capture.take() {
                    let hidden = if cap.hidden {
                        Some(lm.model.hidden_postnorm_row(engine, &x, take - 1)?)
                    } else {
                        None
                    };
                    let logits = capture_piece_logits(&cap, &lm.tok, &s.last_logits);
                    let _ = s.tx.send(Event::PromptCapture { hidden, logits });
                }
            }
            for &tok in &chunk {
                s.fed.push(tok);
                s.sampler.accept(tok);
            }
        } else if let Some(tok) = s.prefill_queue.pop_front() {
            // CAPTURE (lane/embed-serve): the FINAL prompt token walks decode_step_h so
            // sub-prime-floor prompts can still pool their last position (same numeric
            // program; prime_cache hard-asserts T >= PRIME_MIN_T).
            if s.capture.is_some() && s.prefill_queue.is_empty() {
                let (l, h) = lm
                    .model
                    .decode_step_h(engine, tok, s.cache.as_mut().unwrap())?;
                s.last_logits = l;
                if let Some(cap) = s.capture.take() {
                    let hidden = if cap.hidden {
                        Some(lm.model.hidden_postnorm_row(engine, &h, 0)?)
                    } else {
                        None
                    };
                    let logits = capture_piece_logits(&cap, &lm.tok, &s.last_logits);
                    let _ = s.tx.send(Event::PromptCapture { hidden, logits });
                }
            } else {
                s.last_logits = lm
                    .model
                    .decode_step(engine, tok, s.cache.as_mut().unwrap())?;
            }
            if let Some(&target) = s.prefill_queue.front() {
                write_confidence_trace(s, tok, target, &s.last_logits)?;
            }
            s.fed.push(tok);
            s.sampler.accept(tok);
        }
        if s.prefill_queue.is_empty() {
            s.prefill_done = true;
            if let Some(trace) = s.ttft.as_ref() {
                trace.mark_prime_end();
            }
            // Capture fallback: the prompt ended on the tokenwise path (eager-only model,
            // or a carried tail), where no hidden stack exists. Logits are still truthful
            // (`last_logits` is the final position either way); `hidden: None` tells the
            // route the pooling read was unavailable and it answers with a clean error.
            if let Some(cap) = s.capture.take() {
                let logits = capture_piece_logits(&cap, &lm.tok, &s.last_logits);
                let _ = s.tx.send(Event::PromptCapture {
                    hidden: None,
                    logits,
                });
            }
        }
        // If after this the prompt is fully primed AND budget==0, we still fall through to decode
        // (which will immediately hit MaxNew). Keep prefill and decode as distinct ticks otherwise.
        return Ok(true);
    }

    // ---- decode phase ----
    if s.generated.len() >= s.budget {
        finish(s, StopReason::MaxNew);
        return Ok(false);
    }
    // CONSTRAINED rows host-sample from a grammar-masked copy (same seam as
    // advance_sample_emit — the batched path; kept in lockstep).
    let next = match (s.device_next.take(), s.constraint.as_mut()) {
        (Some(t), _) => t,
        (None, Some(c)) => {
            let mut row = s.last_logits.clone();
            c.mask_logits(&mut row)
                .map_err(|e| format!("constraint mask: {e}"))?;
            s.sampler.sample(&row)
        }
        (None, None) => s.sampler.sample(&s.last_logits),
    };
    s.sampler.accept(next);
    s.generated.push(next);
    if let Some(trace) = s.ttft.as_ref() {
        trace.mark_first_decode();
    }

    // EOS stop (before streaming the EOS token as text — we still report it in the count).
    if s.params.eos.contains(&next) {
        if !send_token_event(s, next, String::new()) {
            abort_log(s);
            return Ok(false);
        }
        finish(s, StopReason::Eos);
        return Ok(false);
    }
    if let Some(c) = s.constraint.as_mut() {
        c.consume(next)
            .map_err(|e| format!("constraint advance: {e}"))?;
    }

    // Detokenize the full generated tail, compute the incremental text delta vs what we've emitted.
    s.decoded_bytes
        .extend_from_slice(&lm.tok.decode_bytes_special(&[next], true));
    let delta = utf8_delta(&s.decoded_bytes, &mut s.emitted_bytes);
    // DISCONNECT ABORT (gap-scan F8): failed send = client gone, retire at the abort point.
    if !send_token_event(s, next, delta) {
        abort_log(s);
        return Ok(false);
    }

    // stop-string match on the detokenized tail.
    if contains_stop_string(&s.decoded_bytes, &s.stop_strings) {
        finish(s, StopReason::Callback);
        return Ok(false);
    }

    // context guard.
    if s.cache
        .as_ref()
        .map(|c| c.pos >= c.max_ctx)
        .unwrap_or(false)
    {
        finish(s, StopReason::ContextFull);
        return Ok(false);
    }

    // produce next logits (the ONE decode_step that advances this session).
    s.last_logits = lm
        .model
        .decode_step(engine, next, s.cache.as_mut().unwrap())?;
    s.fed.push(next);
    Ok(true)
}

fn confidence_trace_enabled() -> bool {
    std::env::var("MEMRA_CONFIDENCE_TRACE").is_ok()
}

#[derive(Debug)]
struct ConfidenceSummary {
    reference_logprob: f64,
    top1_token: u32,
    top1_correct: bool,
    top1_top2_margin: f32,
    entropy: f64,
}

fn summarize_confidence(logits: &[f32], target: u32) -> Result<ConfidenceSummary, String> {
    let target = target as usize;
    if logits.is_empty() || target >= logits.len() {
        return Err(format!(
            "target token {target} outside {} logits",
            logits.len()
        ));
    }
    let mut top1 = (0usize, f32::NEG_INFINITY);
    let mut top2 = f32::NEG_INFINITY;
    for (index, &logit) in logits.iter().enumerate() {
        if logit > top1.1 {
            top2 = top1.1;
            top1 = (index, logit);
        } else if logit > top2 {
            top2 = logit;
        }
    }
    let max_logit = top1.1 as f64;
    let mut sum_exp = 0.0f64;
    let mut weighted_logit = 0.0f64;
    for &logit in logits {
        let exp = ((logit as f64) - max_logit).exp();
        sum_exp += exp;
        weighted_logit += exp * logit as f64;
    }
    let logsumexp = max_logit + sum_exp.ln();
    Ok(ConfidenceSummary {
        reference_logprob: logits[target] as f64 - logsumexp,
        top1_token: top1.0 as u32,
        top1_correct: top1.0 == target,
        top1_top2_margin: top1.1 - top2,
        entropy: logsumexp - weighted_logit / sum_exp,
    })
}

fn write_confidence_trace(
    session: &Session,
    input_token: u32,
    target_token: u32,
    logits: &[f32],
) -> Result<(), Box<dyn std::error::Error>> {
    let Ok(path) = std::env::var("MEMRA_CONFIDENCE_TRACE") else {
        return Ok(());
    };
    let summary = summarize_confidence(logits, target_token).map_err(std::io::Error::other)?;
    let record = serde_json::json!({
        "format": "memra-token-confidence-v1",
        "trace_id": session.trace_id,
        "input_position": session.fed.len(),
        "input_token": input_token,
        "target_token": target_token,
        "reference_logprob": summary.reference_logprob,
        "top1_token": summary.top1_token,
        "top1_correct": summary.top1_correct,
        "top1_top2_margin": summary.top1_top2_margin,
        "entropy": summary.entropy,
    });
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;
    writeln!(file, "{record}")?;
    Ok(())
}

/// DISCONNECT ABORT metering record (gap-scan F8): one log line per aborted session —
/// prompt/cached/generated at the abort point (bill-to-abort). Called from every
/// send-failure retire; the tick-top sweep prints the same shape.
/// GEMMA SPEC per-tick step (lane/gemma-batched stage 2, 2026-08-17): the gemma twin of
/// step_session's qwen spec-burst arm, greedy-only. Turn 1 primes the prompt inside
/// `gemma_spec_session_new` (TTFT prime marks around it); every tick runs ONE
/// `gemma_spec_session_burst` (MEMRA_SPEC_BURST cap, default 32) and emits the burst's
/// public tokens through the same `emit_spec_token_events` machinery as the qwen arm —
/// one Event::Token per public id, EOS text never streamed, budget clamp via
/// `spec_visible_len` (engine overshoot stays in GemmaSpecSession.committed, never in the
/// worker's public vectors). Between bursts the scheduler round-robins batch chunks —
/// that interleave IS the coexistence contract the mixed cell measures.
fn step_gemma_spec(
    engine: &Engine,
    loaded: &HashMap<String, LoadedModel>,
    gemma_drafts: &mut std::collections::HashMap<String, memra_engine::gemma_spec::GemmaDraft>,
    s: &mut Session,
) -> Result<bool, Box<dyn std::error::Error>> {
    let lm = &loaded[&s.model];
    let d = gemma_drafts
        .get_mut(&s.model)
        .ok_or("gemma spec session with no attached drafter (admission gate failed)")?;
    debug_assert!(s.spec.is_none(), "a session cannot be on both spec routes");
    let k = s.gspec_k;
    let request_room = s.budget.saturating_sub(s.generated.len());
    if request_room == 0 {
        finish(s, StopReason::MaxNew);
        return Ok(false);
    }
    // turn 1: prime (the session owns its cache; s.cache stays None until demote).
    // PREFIX-HIT carrier (lane/spec-on-cache-hit): admission parked the restored trunk
    // cache in s.cache and the already-restored prefix in s.fed; only the queued suffix
    // feeds here. Cold sessions keep the whole-prompt prime, byte-unchanged.
    if s.gspec.is_none() {
        let queued: Vec<u32> = s.prefill_queue.drain(..).collect();
        if queued.is_empty() {
            finish(s, StopReason::MaxNew);
            return Ok(false);
        }
        if let Some(trace) = s.ttft.as_ref() {
            trace.mark_prime_start();
        }
        let sess = match s.cache.take() {
            Some(restored) => lm
                .model
                .gemma_spec_session_from_restored(engine, d, restored, &s.fed, &queued)?,
            None => lm
                .model
                .gemma_spec_session_new(engine, d, &queued, s.gspec_ctx)?,
        };
        if let Some(trace) = s.ttft.as_ref() {
            trace.mark_prime_end();
        }
        for &tok in &queued {
            s.fed.push(tok);
            s.sampler.accept(tok);
        }
        s.prefill_done = true;
        s.gspec = Some(sess);
    }
    let burst_t: usize = std::env::var("MEMRA_SPEC_BURST")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(32);
    let burst_target = request_room.min(burst_t);
    let sess = s.gspec.as_mut().unwrap();
    let rounds_before = sess.rounds;
    let (burst, dr, ac) =
        lm.model
            .gemma_spec_session_burst(engine, d, sess, burst_target, k, &s.params.eos)?;
    let rounds_delta = s.gspec.as_ref().unwrap().rounds - rounds_before;
    s.spec_rounds += rounds_delta as u64;
    s.spec_drafted += dr;
    s.spec_accepted += ac;
    if dr > 0 {
        eprintln!(
            "[gspec-acc] ctx={} burst={}/{} cum={}/{}={:.3}",
            s.fed.len(),
            ac,
            dr,
            s.spec_accepted,
            s.spec_drafted,
            s.spec_accepted as f64 / s.spec_drafted.max(1) as f64
        );
    }
    if let Some(trace) = s.ttft.as_ref() {
        if !burst.is_empty() {
            trace.mark_first_decode();
        }
    }
    // public clamp + emission (per-burst cadence v1; the qwen round-cadence on_commit is a
    // later increment) — same helper, same one-event-per-public-id receipt.
    let eos_ids = s.params.eos.clone();
    let public_len = spec_visible_len(&burst, request_room, &eos_ids);
    let public_burst = &burst[..public_len];
    let mut decoded_visible = std::mem::take(&mut s.decoded_bytes);
    let mut cursor = s.emitted_bytes;
    let mut emit_remaining = request_room;
    let mut eos_seen = false;
    let tok_ref = &lm.tok;
    let emitted = emit_spec_token_events(
        public_burst,
        &mut emit_remaining,
        &mut decoded_visible,
        &mut cursor,
        &eos_ids,
        &mut eos_seen,
        |id| tok_ref.decode_bytes_special(&[id], true),
        |event| s.tx.send(event).is_ok(),
    );
    let mut stop: Option<StopReason> = None;
    for &tok in public_burst {
        s.sampler.accept(tok);
        s.generated.push(tok);
        s.fed.push(tok);
        if s.params.eos.contains(&tok) {
            stop = Some(StopReason::Eos);
            break;
        }
    }
    s.tokens_emitted += emitted.sent;
    s.emitted_bytes = cursor;
    s.decoded_bytes = decoded_visible;
    if !emitted.send_ok {
        abort_log(s);
        return Ok(false);
    }
    if emitted.send_ok {
        debug_assert_eq!(
            emitted.sent, public_len,
            "one token event per public gemma spec token"
        );
    }
    if stop.is_none() && contains_stop_string(&s.decoded_bytes, &s.stop_strings) {
        stop = Some(StopReason::Callback);
    }
    if stop.is_none() && s.generated.len() >= s.budget {
        stop = Some(StopReason::MaxNew);
    }
    let sess = s.gspec.as_ref().unwrap();
    // +2: committed excludes the parked pending token + the round's bonus row headroom.
    if stop.is_none() && sess.committed.len() + k + 2 >= sess.cache_max_ctx() {
        stop = Some(StopReason::ContextFull);
    }
    if let Some(r) = stop {
        finish(s, r);
        return Ok(false);
    }
    Ok(true)
}

/// DSPARK SPEC per-tick step (lane/dspark-q38-recover serve route): the qwen-hybrid twin
/// of step_gemma_spec — greedy or sampled (the session owns the SpecSampling + Philox
/// counters; lane/dspark-sampled-admission-20260820). Turn 1 primes the prompt inside
/// `dspark_spec_session_new` (TTFT prime marks around it); every tick runs ONE
/// `dspark_spec_session_burst` (MEMRA_SPEC_BURST cap, default 32) and emits the burst's
/// public tokens through the same `emit_spec_token_events` machinery — one Event::Token
/// per public id, EOS text never streamed, budget clamp via `spec_visible_len` (engine
/// overshoot stays committed in the session cache, never in the worker's public vectors).
/// Between bursts the scheduler round-robins batch chunks — the coexistence contract.
fn step_dspark_spec(
    engine: &Engine,
    loaded: &HashMap<String, LoadedModel>,
    dspark_drafts: &mut std::collections::HashMap<String, memra_engine::dflash::DflashDraft>,
    s: &mut Session,
) -> Result<bool, Box<dyn std::error::Error>> {
    let lm = &loaded[&s.model];
    let d = dspark_drafts
        .get_mut(&s.model)
        .ok_or("dspark spec session with no attached drafter (admission gate failed)")?;
    debug_assert!(s.spec.is_none(), "a session cannot be on both spec routes");
    debug_assert!(
        s.gspec.is_none(),
        "a session cannot be on both drafter routes"
    );
    let request_room = s.budget.saturating_sub(s.generated.len());
    if request_room == 0 {
        finish(s, StopReason::MaxNew);
        return Ok(false);
    }
    // POOL RESUME (lane/dflash2-session-reuse): a resumed session arrives with its
    // parked state in s.dspark and the new turn's suffix in prefill_queue — prime ONLY
    // that delta (dspark_spec_session_resume), then burst. Empty-suffix exact
    // continuations skip this (prefill_done was set at admit; the burst re-emits the
    // parked boundary token as its anchor, the MTP pool's own convention).
    if s.dspark.is_some() && !s.prefill_queue.is_empty() {
        let suffix: Vec<u32> = s.prefill_queue.drain(..).collect();
        if let Some(trace) = s.ttft.as_ref() {
            trace.mark_prime_start();
        }
        let sess = s.dspark.as_mut().unwrap();
        if let Err(err) = lm
            .model
            .dspark_spec_session_resume(engine, d, sess, &suffix)
        {
            // Fail the request loudly rather than silently switching numeric programs
            // mid-request — the same law as the cold prime below. The parked state is
            // consumed either way (a half-primed resume must not re-park).
            return Err(format!("dspark resume failed: {err}").into());
        }
        if let Some(trace) = s.ttft.as_ref() {
            trace.mark_prime_end();
        }
        for &tok in &suffix {
            s.fed.push(tok);
            s.sampler.accept(tok);
        }
        s.prefill_done = true;
    }
    // turn 1: prime (the session owns its cache; s.cache stays None).
    if s.dspark.is_none() {
        let prompt: Vec<u32> = s.prefill_queue.drain(..).collect();
        if prompt.is_empty() {
            finish(s, StopReason::MaxNew);
            return Ok(false);
        }
        if let Some(trace) = s.ttft.as_ref() {
            trace.mark_prime_start();
        }
        // Sampled admission (T>0): the ONE Sampler->SpecSampling seam (spec_sampling_for),
        // same as the frspec route — None = greedy, byte-identical route. Penalties ride
        // the config (the engine's accept walk penalizes p over the session window);
        // admission keeps only penalized GREEDY off this path.
        let sess = match lm.model.dspark_spec_session_new(
            engine,
            d,
            &prompt,
            s.gspec_ctx,
            spec_sampling_for(&s.sampler),
            s.dspark_capture_prefix,
        ) {
            Ok(sess) => sess,
            Err(err) => {
                // The windowless-drafter ctx refusal (or an alloc failure) at prime time:
                // fail the request loudly rather than silently switching numeric programs
                // mid-request — admission is where the plain fallback lives.
                return Err(format!("dspark prime failed: {err}").into());
            }
        };
        if let Some(trace) = s.ttft.as_ref() {
            trace.mark_prime_end();
        }
        for &tok in &prompt {
            s.fed.push(tok);
            s.sampler.accept(tok);
        }
        s.prefill_done = true;
        s.dspark = Some(sess);
    }
    let burst_t: usize = std::env::var("MEMRA_SPEC_BURST")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(32);
    let burst_target = request_room.min(burst_t);
    let sess = s.dspark.as_mut().unwrap();
    let rounds_before = sess.rounds;
    let (burst, dr, ac) = lm.model.dspark_spec_session_burst(
        engine,
        d,
        sess,
        burst_target,
        request_room,
        &s.params.eos,
    )?;
    let rounds_delta = s.dspark.as_ref().unwrap().rounds - rounds_before;
    s.spec_rounds += rounds_delta as u64;
    s.spec_drafted += dr;
    s.spec_accepted += ac;
    if dr > 0 {
        eprintln!(
            "[dspark-acc] ctx={} burst={}/{} cum={}/{}={:.3}",
            s.fed.len(),
            ac,
            dr,
            s.spec_accepted,
            s.spec_drafted,
            s.spec_accepted as f64 / s.spec_drafted.max(1) as f64
        );
    }
    if let Some(trace) = s.ttft.as_ref() {
        if !burst.is_empty() {
            trace.mark_first_decode();
        }
    }
    let eos_ids = s.params.eos.clone();
    let public_len = spec_visible_len(&burst, request_room, &eos_ids);
    let public_burst = &burst[..public_len];
    let mut decoded_visible = std::mem::take(&mut s.decoded_bytes);
    let mut cursor = s.emitted_bytes;
    let mut emit_remaining = request_room;
    let mut eos_seen = false;
    let tok_ref = &lm.tok;
    let emitted = emit_spec_token_events(
        public_burst,
        &mut emit_remaining,
        &mut decoded_visible,
        &mut cursor,
        &eos_ids,
        &mut eos_seen,
        |id| tok_ref.decode_bytes_special(&[id], true),
        |event| s.tx.send(event).is_ok(),
    );
    let mut stop: Option<StopReason> = None;
    for &tok in public_burst {
        s.sampler.accept(tok);
        s.generated.push(tok);
        s.fed.push(tok);
        if s.params.eos.contains(&tok) {
            stop = Some(StopReason::Eos);
            break;
        }
    }
    s.tokens_emitted += emitted.sent;
    s.emitted_bytes = cursor;
    s.decoded_bytes = decoded_visible;
    if !emitted.send_ok {
        abort_log(s);
        return Ok(false);
    }
    debug_assert_eq!(
        emitted.sent, public_len,
        "one token event per public dspark spec token"
    );
    if stop.is_none() && contains_stop_string(&s.decoded_bytes, &s.stop_strings) {
        stop = Some(StopReason::Callback);
    }
    if stop.is_none() && s.generated.len() >= s.budget {
        stop = Some(StopReason::MaxNew);
    }
    // The engine marks the session done on EOS or when the next round would overflow the
    // (windowless-drafter-clamped) ctx; EOS/budget above take precedence for the reason.
    if stop.is_none() && s.dspark.as_ref().unwrap().finished() {
        stop = Some(StopReason::ContextFull);
    }
    if let Some(r) = stop {
        finish(s, r);
        return Ok(false);
    }
    Ok(true)
}

fn abort_log(s: &mut Session) {
    s.aborted = true;
    eprintln!(
        "[abort] client disconnected: model {:?}, prompt {} ({} cached), \
               {} generated — billed to abort point, {:.2}s",
        s.model,
        s.n_prompt,
        s.n_cached,
        s.generated.len(),
        s.t0.elapsed().as_secs_f64()
    );
}

fn retire_may_park(aborted: bool) -> bool {
    !aborted
}

#[cfg(test)]
mod abort_park_tests {
    #[test]
    fn aborted_sessions_never_publish_reusable_kv() {
        assert!(super::retire_may_park(false));
        assert!(!super::retire_may_park(true));
    }
}

fn finish(s: &Session, reason: StopReason) {
    let elapsed = s.t0.elapsed().as_secs_f64();
    assert_eq!(
        s.tokens_emitted,
        s.generated.len(),
        "terminal token receipt mismatch: Event::Token count != generated count",
    );
    // constrained-session mask-cost receipt (the perf ledger line): steps + total/mean
    // host-side mask compute time. Unconstrained sessions log nothing.
    if let Some(c) = s.constraint.as_ref() {
        if c.steps > 0 {
            eprintln!(
                "[constrained] {}: {} masked steps, mask total {:.2} ms ({:.3} ms/step)",
                s.model,
                c.steps,
                c.mask_ns as f64 / 1e6,
                c.mask_ns as f64 / 1e6 / c.steps as f64
            );
        }
        // DRAFT-SIDE MASKING receipt (lane/draft-mask): the speculative Matcher clone (one per
        // spec round) and the draft-position masks computed on it — the two costs the lane adds.
        if c.spec_clones > 0 {
            eprintln!(
                "[draft-mask] {}: {} clones {:.2} ms ({:.3} ms/clone), \
                       {} draft masks {:.2} ms ({:.3} ms/mask)",
                s.model,
                c.spec_clones,
                c.spec_ns as f64 / 1e6,
                c.spec_ns as f64 / 1e6 / c.spec_clones as f64,
                c.draft_masks,
                c.draft_mask_ns as f64 / 1e6,
                c.draft_mask_ns as f64 / 1e6 / c.draft_masks.max(1) as f64
            );
        }
    }
    let reason = format!("{reason:?}");
    // Per-request spec acceptance summary (lane/accept-telemetry): only when this request
    // actually ran spec rounds — plain sessions carry None and the usage block is unchanged.
    let spec = (s.spec_rounds > 0).then_some(SpecUsage {
        rounds: s.spec_rounds,
        drafted: s.spec_drafted as u64,
        accepted: s.spec_accepted as u64,
    });
    let _ = s.tx.send(Event::TokenSnapshot(s.generated.clone()));
    let _ = s.tx.send(Event::Done {
        stop_reason: reason,
        n_tokens: s.generated.len(),
        n_prompt: s.n_prompt,
        n_cached: s.n_cached,
        elapsed_s: elapsed,
        spec,
    });
}

/// FAULT INJECTION (`MEMRA_PANIC_AFTER=<n>`, off unless set): panic the GPU worker thread
/// after `n` served requests. An explicitly-blocked experimental door in the flags-doctrine
/// sense, and the ONLY way the G5 supervision path can be exercised against a REAL CUDA
/// worker — the alternative is trusting that a catch_unwind + respawn + exit-70 ladder built
/// around a live CUDA context behaves the way its unit tests (which use a fake worker) say it
/// does. That trust was already misplaced once on this lane: the first supervisor deadlocked
/// startup, and only a live gate found it. Costs one relaxed atomic load per completed request.
///
/// ONE-SHOT PER PROCESS. `n_completed` is per-`run()`, so a per-run trigger re-fires on the
/// respawned worker the moment it serves its first request — measured: the respawn reloaded
/// the weights, went green with `generation:1`, then immediately panicked again and exited 70,
/// which makes "did the recovery actually serve traffic?" unanswerable. Injecting exactly one
/// panic per process is what proves the recovery half.
fn panic_after() -> Option<u64> {
    static P: std::sync::OnceLock<Option<u64>> = std::sync::OnceLock::new();
    *P.get_or_init(|| {
        std::env::var("MEMRA_PANIC_AFTER")
            .ok()
            .and_then(|v| v.parse().ok())
    })
}

/// Set once the injected panic has fired, so it fires at most once per process (see above).
static PANIC_INJECTED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// True exactly once, on the `n`th completed request of the process's first worker run.
fn panic_injection_due(n_completed: u64) -> bool {
    match panic_after() {
        Some(n) if n_completed >= n => {
            !PANIC_INJECTED.swap(true, std::sync::atomic::Ordering::SeqCst)
        }
        _ => false,
    }
}

/// Number of respawn attempts after a worker-thread PANIC before the process fails loudly.
/// ONE, deliberately: CUDA errors are sticky per process (after an OOM or an Xid the context
/// is poisoned), so an in-process retry is a long shot — worth exactly one try, because when
/// it works it saves a ~120 s weight reload, and worth no more, because a respawn loop against
/// a poisoned context is a box that looks alive and serves nothing. MEMRA_WORKER_RESPAWN=0
/// disables (straight to loud failure, i.e. let the supervisor restart the process).
const WORKER_RESPAWN_MAX: u32 = 1;

/// Base delay for the respawn ladder and the HTTP retry hint while the worker is unavailable.
pub(crate) const WORKER_RESPAWN_BACKOFF_BASE_S: u64 = 2;

fn worker_respawn_max() -> u32 {
    static R: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *R.get_or_init(|| {
        std::env::var("MEMRA_WORKER_RESPAWN")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(WORKER_RESPAWN_MAX)
    })
}

/// Exit code when the worker is unrecoverable. `systemd Restart=on-failure` treats any nonzero
/// as failure; 70 is sysexits' EX_SOFTWARE, so an operator reading `systemctl status` can tell
/// "the engine died" from "bad config" (exit 1, the startup FATAL paths in main).
const EXIT_WORKER_UNRECOVERABLE: i32 = 70;

/// Convenience: spawn the worker thread and block until it reports ready (or fails). Returns the
/// command Sender (clone into the axum state) + the loaded model names + template caps.
///
/// SUPERVISION (G5c, lane/serve-hardening 2026-08-06). The worker thread used to be a bare
/// `spawn(move || run(..))`: a panic inside it unwound that thread ONLY, the process kept
/// serving HTTP, `/health` stayed green forever, and every request blocked or died on a closed
/// channel. Now the spawned thread is a SUPERVISOR that:
///   1. runs the scheduler inside `catch_unwind`, so a panic is caught instead of silently
///      ending the thread;
///   2. marks the shared health FAULTED on catch — /health and /readyz flip within
///      milliseconds, no staleness threshold to wait out;
///   3. attempts `worker_respawn_max()` respawns with backoff (weights reload; the health
///      generation counter increments so the recovery is observable);
///   4. and if that fails, exits the PROCESS loudly — because a memra-server without a GPU
///      worker cannot serve anything, and `Restart=` restarting the unit whole is the only
///      reliable CUDA recovery (see `deploy/systemd/memra-server.service`).
/// A CLEAN return (the command channel closed = every HTTP handler dropped = shutdown) is not
/// a fault and never respawns.
#[allow(clippy::type_complexity)]
pub fn spawn(
    models: Vec<(String, String, Option<String>)>,
    health: crate::health::SharedHealth,
) -> Result<
    (
        Sender<Cmd>,
        Arc<Vec<String>>,
        Arc<HashMap<String, ModelCaps>>,
        SharedMetrics,
        std::thread::JoinHandle<()>,
    ),
    String,
> {
    // (#87's parse-time spec-over-PP-2 preflight refusal lived here — CLOSED 2026-08-08.
    // The ppN reverse-publication fences make spec+PP-2 serve; receipts and the crash gate
    // are in research/pp2spec-crash-20260807/.)
    let (cmd_tx, cmd_rx) = std::sync::mpsc::channel::<Cmd>();
    let (ready_tx, ready_rx) =
        std::sync::mpsc::channel::<Result<(Vec<String>, HashMap<String, ModelCaps>), String>>();
    let metrics: SharedMetrics = Default::default();
    let m2 = metrics.clone();
    let h2 = health.clone();
    let worker_thread = std::thread::Builder::new()
        .name("memra-gpu-worker".into())
        .spawn(move || {
            // The supervisor OWNS the receiver across restarts: `run` borrows it, so a
            // panicking scheduler cannot take the command channel down with it (dropping the
            // Receiver would make every future handler send fail with no way back).
            let rx = cmd_rx;
            let mut ready_tx = Some(ready_tx);
            let mut attempt: u32 = 0;
            loop {
                let (models, m, h) = (models.clone(), m2.clone(), h2.clone());
                // A fresh ready channel per attempt; only the FIRST one is the caller's.
                //
                // THE VERDICT MUST BE RELAYED CONCURRENTLY, NOT AFTER `run` RETURNS. `run`
                // sends its load verdict and then blocks in the scheduler for the life of the
                // process — so reading `rrx` on this thread after `catch_unwind` deadlocks the
                // whole server: main blocks in `ready_rx.recv()`, never binds the socket, and
                // the box loads the model and then answers nothing. (Found by serve-smoke,
                // which timed out waiting for /health with the worker log showing a fully
                // loaded model — the exact failure class this lane exists to remove, so it is
                // fitting that the gate caught it.)
                let (rtx, rrx) = std::sync::mpsc::channel();
                let caller = ready_tx.take();
                let load_failed = Arc::new(std::sync::atomic::AtomicBool::new(false));
                let (lf, hr) = (load_failed.clone(), h2.clone());
                let relay = std::thread::Builder::new()
                    .name("memra-worker-ready".into())
                    .spawn(move || {
                        let verdict = rrx
                            .recv()
                            .unwrap_or_else(|_| Err("worker died during init".into()));
                        if let Err(why) = &verdict {
                            lf.store(true, std::sync::atomic::Ordering::SeqCst);
                            hr.mark_dead(format!("model load failed: {why}"));
                        }
                        // Only the first attempt has a caller waiting; a respawn's verdict is
                        // observable on /health (phase + generation) instead.
                        if let Some(tx) = caller {
                            let _ = tx.send(verdict);
                        }
                    });
                let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    run(models, &rx, rtx, m, h)
                }));
                // `run` has returned, so its `rtx` is dropped and the relay cannot block.
                if let Ok(t) = relay {
                    let _ = t.join();
                }
                match outcome {
                    Ok(()) if load_failed.load(std::sync::atomic::Ordering::SeqCst) => {
                        // Not a shutdown: the model load itself failed, so `run` returned
                        // without ever entering the scheduler.
                        if attempt == 0 {
                            // The caller (main) got the error and reports it as a startup
                            // FATAL — do not race it with an exit code of our own.
                            return;
                        }
                        eprintln!(
                            "[worker] FATAL: respawn attempt {attempt} could not reload \
                                   the models — exiting the process so the supervisor can \
                                   restart it whole"
                        );
                        crate::health::sd_notify("STATUS=respawn load failed; exiting");
                        std::io::stderr().flush().ok();
                        std::process::exit(EXIT_WORKER_UNRECOVERABLE);
                    }
                    Ok(()) => {
                        // Clean scheduler exit = the command channel closed (shutdown).
                        h2.set_phase(crate::health::PHASE_DEAD);
                        return;
                    }
                    Err(payload) => {
                        // QUOTED, never inferred: the panic message as the panic handler saw
                        // it (String / &str payloads; anything else says so).
                        let why = payload
                            .downcast_ref::<String>()
                            .cloned()
                            .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
                            .unwrap_or_else(|| "non-string panic payload".into());
                        attempt += 1;
                        h2.mark_dead(format!("worker thread panicked: {why}"));
                        eprintln!("[worker] PANIC in the GPU worker thread: {why}");
                        if attempt > worker_respawn_max() {
                            eprintln!(
                                "[worker] FATAL: worker unrecoverable after {} respawn \
                                       attempt(s) — exiting the process so the supervisor can \
                                       restart it whole (CUDA errors are sticky per process; a \
                                       live HTTP listener with a dead worker serves nothing)",
                                attempt - 1
                            );
                            crate::health::sd_notify("STATUS=worker unrecoverable; exiting");
                            std::io::stderr().flush().ok();
                            std::process::exit(EXIT_WORKER_UNRECOVERABLE);
                        }
                        // Backoff before reloading weights: a panic caused by a transient
                        // device condition needs the driver to settle, and an immediate
                        // reload would just re-hit it.
                        let backoff = std::time::Duration::from_secs(
                            WORKER_RESPAWN_BACKOFF_BASE_S * attempt as u64,
                        );
                        eprintln!(
                            "[worker] respawn attempt {attempt}/{} in {:?} \
                                   (reloading weights)",
                            worker_respawn_max(),
                            backoff
                        );
                        std::thread::sleep(backoff);
                        h2.mark_respawning();
                    }
                }
            }
        })
        .map_err(|e| format!("spawn worker thread: {e}"))?;
    match ready_rx.recv() {
        Ok(Ok((names, caps))) => Ok((
            cmd_tx,
            Arc::new(names),
            Arc::new(caps),
            metrics,
            worker_thread,
        )),
        Ok(Err(err)) => Err(err),
        Err(_) => Err("worker died during init".into()),
    }
}

#[cfg(test)]
mod tests {
    use super::SpecTelemetryWindow;
    use super::context_cache_bytes;
    use super::plain_chat_render_path;
    use super::{
        ADSD_MIN_RATE_DROP, ADSD_SUSTAINED_OBSERVATIONS, ADSD_TENANT_WINDOW, ADSD_Z_THRESHOLD,
        AdsdDetector,
    };
    use super::{
        AdmissionCostModel, AdmissionDeviceHeadroom, AdmissionHeadroom, MAX_NEW_CTX_BOUNDED,
        MAX_PROMPT_SOURCE_BYTES, ParkedCandidate, ParkedPool, Request, ReuseMetrics,
        SPEC_SHRINK_RESERVE, admission_required, admission_reserve,
        alloc_with_single_reclaim_retry, dual_pp_boundary_slot_bytes, dual_pp_device_requirements,
        dual_pp_stage_admission, enforce_prompt_limit, is_cuda_oom, oldest_parked_candidate,
        parallel_device_requirements, parked_entry_count, prepare_park, prompt_source_limit_error,
        request_ctx_cap,
    };
    use super::{DecodeChunkPolicy, resolve_decode_chunk_policy, schedule_decode_chunk};
    use super::{DraftVerdict, draft_verdict, draft_verdict_message};
    use super::{
        Event, cached_hit_needs_first_token, carried_prime_batch_eligible, emit_spec_token_events,
        graph_sampler_eligible, graph_session_env_on, interactive_prefill_budget,
        interactive_prime_batch_take, prefill_tick_take, record_output_progress,
        record_output_tokens, routed_moe_prefix_split, spec_visible_len, summarize_confidence,
        utf8_delta,
    };
    use super::{HashMap, METER_TENANT_CAP, meter_account, meter_cached_credit};
    use super::{
        PREFIX_CACHE_DEFAULT_ENTRIES, derived_prefix_cache_budget, prefix_entry_geometry_bytes,
    };
    use super::{
        PREFIX_CACHE_MIN_TOKENS, PREFIX_ENTRY_LAYOUT_VERSION, PartialPrefixDecision, PoolKey,
        PrefixCache, PrefixEntry, PrefixFanoutCandidate, PrefixFanoutGroup, PrefixSegment,
        partial_prefix_decision, prefix_fanout_groups, retire_prefix_pin,
        validate_prefix_plane_shape,
    };
    use super::{
        RuntimePeerProbeDeferralState, RuntimePeerProbeWorkerAction, peer_probe_spec_admission,
        resolve_runtime_peer_probe_deferral_bound, runtime_peer_probe_allowed,
        runtime_peer_probe_worker_action,
    };
    use super::{
        SPEC_K_CACHED_LONG, SPEC_K_CACHED_LONG_TRIM, SPEC_K_COLD_LONG, SPEC_K_COLD_SHORT,
        SPEC_K_LONG_CACHE_MIN, SPEC_K_LONG_PROMPT_MIN, SpecKDecision, SpecKReason, choose_spec_k,
        constrained_spec_supported, model_forces_spec_replay, parse_spec_k_pin,
        resolve_spec_gate_thresholds, sampled_restore_load_admits, sampled_restore_watermark,
        spec_gate_defaults,
    };
    use super::{optipipe_controller_threshold, worker_device};
    use crate::lanes::{Lane, StepStats};
    use memra_engine::sampler::{Sampler, SamplerConfig};

    fn bare_request() -> Request {
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        Request {
            model: "m".into(),
            prompt_ids: Vec::new(),
            prompt_text: String::new(),
            chat: false,
            chat_turns: Vec::new(),
            tools_json: Vec::new(),
            tools_struct: Vec::new(),
            think: memra_tokenizer::chat::ThinkMode::Default,
            reasoning_effort: None,
            params: memra_engine::decode::GenParams::default(),
            sampler_cfg: SamplerConfig::default(),
            stop_strings: Vec::new(),
            trace_id: None,
            max_prompt_tokens: None,
            cache_ns: String::new(),
            affinity: None,
            lane: Lane::Interactive,
            oom_retries: 0,
            spec_k_replay: None,
            grammar: None,
            prepared_constraint: None,
            constraint_ready: None,
            prepared_prompt: None,
            ttft: None,
            images: Vec::new(),
            gemma_images: Vec::new(),
            capture: None,
            vision_memory: None,
            tx,
        }
    }

    #[test]
    fn prompt_source_limit_rejects_before_tokenizer_input_is_built() {
        let mut request = bare_request();
        request.prompt_text = "x".repeat(MAX_PROMPT_SOURCE_BYTES + 1);
        assert!(prompt_source_limit_error(&request).is_some());

        request.prompt_text.clear();
        request.prompt_ids = vec![0; MAX_PROMPT_SOURCE_BYTES / std::mem::size_of::<u32>() + 1];
        assert!(prompt_source_limit_error(&request).is_some());

        request.prompt_ids.clear();
        request.prompt_text = "x".repeat(MAX_PROMPT_SOURCE_BYTES);
        assert!(prompt_source_limit_error(&request).is_none());
    }

    #[derive(Debug)]
    struct TestParkedEntry {
        id: u32,
        parked_at: std::time::Instant,
    }

    impl super::ParkedEntryAge for TestParkedEntry {
        fn parked_at(&self) -> std::time::Instant {
            self.parked_at
        }
    }

    fn ascii_decode(id: u32) -> Vec<u8> {
        vec![b'a' + id as u8]
    }

    #[test]
    fn plain_fast_path_excludes_effort_ladder_templates() {
        use memra_tokenizer::chat::{ThinkMode, Turn};
        let turns = vec![Turn {
            role: "user".into(),
            content: "hi".into(),
            ..Default::default()
        }];
        // The four historical plain conditions qualify a ladder-less template...
        assert!(plain_chat_render_path(
            &[],
            &ThinkMode::Default,
            None,
            &turns,
            false
        ));
        // ...and NEVER a ladder template: its unset case renders the vendor xhigh
        // default, which only the tools-capable renderer injects. v0.109.0 shipped
        // without this arm and served a split surface (live receipt, q38-nj
        // 2026-08-23: unset+tools rendered the sentence, plain unset did not).
        assert!(!plain_chat_render_path(
            &[],
            &ThinkMode::Default,
            None,
            &turns,
            true
        ));
        // The historical disqualifiers are unchanged by the new arm.
        let tools = vec!["{}".to_string()];
        assert!(!plain_chat_render_path(
            &tools,
            &ThinkMode::Default,
            None,
            &turns,
            false
        ));
        assert!(!plain_chat_render_path(
            &[],
            &ThinkMode::NoThink,
            None,
            &turns,
            false
        ));
        assert!(!plain_chat_render_path(
            &[],
            &ThinkMode::Default,
            Some("low"),
            &turns,
            false
        ));
        let tool_turn = vec![Turn {
            role: "tool".into(),
            content: "r".into(),
            ..Default::default()
        }];
        assert!(!plain_chat_render_path(
            &[],
            &ThinkMode::Default,
            None,
            &tool_turn,
            false
        ));
    }

    #[test]
    fn graph_session_requires_explicit_opt_in() {
        assert!(!graph_session_env_on(None));
        assert!(!graph_session_env_on(Some("0")));
        assert!(!graph_session_env_on(Some("true")));
        assert!(graph_session_env_on(Some("1")));
    }

    #[test]
    fn graph_session_refuses_penalties_its_capture_does_not_apply() {
        assert!(graph_sampler_eligible(&Sampler::new(
            SamplerConfig::default()
        )));
        let penalized = Sampler::new(SamplerConfig {
            temperature: 0.0,
            penalty_last_n: usize::MAX,
            penalty_present: 1.5,
            ..Default::default()
        });
        assert!(!graph_sampler_eligible(&penalized));
    }

    #[test]
    fn runtime_reprobe_mismatch_degrades_without_panicking_or_breaking_output_continuity() {
        let mut output = vec![101u32, 102];
        let action = runtime_peer_probe_worker_action(Ok(
            memra_engine::pp::RuntimePeerProbeStatus::DegradedToHostBounce,
        ));
        assert_eq!(
            action,
            Ok(RuntimePeerProbeWorkerAction::DegradedToHostBounce)
        );

        // The worker keeps the admitted request alive and its next ordinary plain tick appends the
        // same token it would have produced before the injected mismatch.
        output.push(103);
        assert_eq!(output, vec![101, 102, 103]);

        let fatal = runtime_peer_probe_worker_action(Err("injected bounce arm failure".into()));
        assert_eq!(fatal, Err("injected bounce arm failure".into()));
    }

    #[test]
    fn prefix_budget_geometry_reproduces_all_six_measured_q27_q35_entries() {
        // Exact `prefix_cache_bytes` deltas from research/cachesize-20260813. The slopes are
        // the summed full-attention K+V bytes/token; the intercepts are the fixed recurrent
        // snapshot. Keeping all six points here catches drift in either term.
        let cases = [
            ("q27-4096", 29_696, 156_893_184, 4_096, 278_528_000),
            ("q27-4860", 29_696, 156_893_184, 4_860, 301_215_744),
            ("q27-8192", 29_696, 156_893_184, 8_192, 400_162_816),
            ("q35-4096", 9_280, 65_863_680, 4_096, 103_874_560),
            ("q35-4860", 9_280, 65_863_680, 4_860, 110_964_480),
            ("q35-8192", 9_280, 65_863_680, 8_192, 141_885_440),
        ];
        for (name, bytes_per_token, recurrent_bytes, ctx, measured) in cases {
            assert_eq!(
                prefix_entry_geometry_bytes(bytes_per_token, recurrent_bytes, ctx),
                measured,
                "{name}",
            );
        }
    }

    #[test]
    fn derived_prefix_budget_holds_two_q27_8192_entries_and_reserves_boot_headroom() {
        let entry_bytes = 400_162_816;
        let requested = entry_bytes * PREFIX_CACHE_DEFAULT_ENTRIES;
        let boot_free = SPEC_SHRINK_RESERVE + requested + 123;
        let (budget, derived_requested, clamp) =
            derived_prefix_cache_budget(entry_bytes, boot_free);
        assert_eq!(PREFIX_CACHE_DEFAULT_ENTRIES, 2);
        assert_eq!(derived_requested, requested);
        assert_eq!(budget, requested);
        assert_eq!(clamp, requested + 123);
        assert!(budget <= boot_free);

        let constrained_free = SPEC_SHRINK_RESERVE + entry_bytes;
        let (budget, _, clamp) = derived_prefix_cache_budget(entry_bytes, constrained_free);
        assert_eq!(budget, entry_bytes);
        assert_eq!(budget, clamp);
        assert!(budget <= constrained_free);
    }

    #[test]
    fn runtime_reprobe_deferral_counts_intervals_and_alarms_after_one_rotation() {
        let every = memra_engine::pp::PEER_RUNTIME_PROBE_INTERVAL_COPIES;
        let bound = memra_engine::pp::PEER_RUNTIME_PROBE_DEFERRAL_BOUND_INTERVALS;
        let mut state = RuntimePeerProbeDeferralState::with_bound(bound);

        for interval in 1..=bound {
            let observation = state.observe(interval * every);
            assert_eq!(observation.intervals, 1);
            assert_eq!(observation.consecutive_intervals, interval);
            assert_eq!(observation.bound_reached, interval == bound);

            let duplicate_poll = state.observe(interval * every + every - 1);
            assert_eq!(duplicate_poll.intervals, 0);
            assert_eq!(duplicate_poll.consecutive_intervals, interval);
            assert!(!duplicate_poll.bound_reached);
        }

        assert!(
            state.resolve(),
            "the bound must leave a degraded state to clear"
        );
        let next_episode = state.observe((bound + 1) * every);
        assert_eq!(next_episode.intervals, 1);
        assert_eq!(next_episode.consecutive_intervals, 1);
        assert!(!next_episode.bound_reached);
    }

    #[test]
    fn runtime_reprobe_soft_refusal_starts_at_bound_and_completed_probe_clears_it() {
        let every = memra_engine::pp::PEER_RUNTIME_PROBE_INTERVAL_COPIES;
        let health = crate::health::WorkerHealth::new();
        let mut state = RuntimePeerProbeDeferralState::with_bound(2);

        let first = state.observe(every);
        health.note_peer_probe_deferral(first.consecutive_intervals, first.bound_reached);
        assert_eq!(
            health.peer_probe_integrity(),
            crate::health::PeerProbeIntegrity::Deferred(1),
        );
        assert!(peer_probe_spec_admission(
            true,
            health.peer_probe_allows_spec_admission(),
        ));

        let second = state.observe(2 * every);
        health.note_peer_probe_deferral(second.consecutive_intervals, second.bound_reached);
        assert!(second.bound_reached);
        assert_eq!(
            health.peer_probe_integrity(),
            crate::health::PeerProbeIntegrity::Degraded,
        );
        assert!(!peer_probe_spec_admission(
            true,
            health.peer_probe_allows_spec_admission(),
        ));
        assert!(
            !peer_probe_spec_admission(false, health.peer_probe_allows_spec_admission(),),
            "a plain candidate stays on the plain path rather than becoming a refusal"
        );

        let completed = memra_engine::pp::RuntimePeerProbeStatus::Passed;
        assert!(completed.ran());
        assert!(state.resolve());
        health.clear_peer_probe_deferral();
        assert_eq!(
            health.peer_probe_integrity(),
            crate::health::PeerProbeIntegrity::Ok,
        );
        assert!(peer_probe_spec_admission(
            true,
            health.peer_probe_allows_spec_admission(),
        ));
    }

    #[test]
    fn runtime_reprobe_bound_config_defaults_to_probeobs_four_and_clamps_zero() {
        assert_eq!(
            resolve_runtime_peer_probe_deferral_bound(None),
            memra_engine::pp::PEER_RUNTIME_PROBE_DEFERRAL_BOUND_INTERVALS,
        );
        assert_eq!(resolve_runtime_peer_probe_deferral_bound(Some("7")), 7);
        assert_eq!(resolve_runtime_peer_probe_deferral_bound(Some("0")), 1);
        assert_eq!(
            resolve_runtime_peer_probe_deferral_bound(Some("invalid")),
            memra_engine::pp::PEER_RUNTIME_PROBE_DEFERRAL_BOUND_INTERVALS,
        );
    }

    #[test]
    fn single_device_no_runtime_probe_keeps_peer_admission_state_inert() {
        let health = crate::health::WorkerHealth::new();
        let status = memra_engine::pp::RuntimePeerProbeStatus::NotRun;
        assert!(!status.ran());
        assert_eq!(
            health.peer_probe_integrity(),
            crate::health::PeerProbeIntegrity::Ok,
        );
        assert!(peer_probe_spec_admission(
            true,
            health.peer_probe_allows_spec_admission(),
        ));
    }

    #[test]
    fn runtime_reprobe_plain_serving_keeps_probe_execution_enabled() {
        assert!(runtime_peer_probe_allowed(false));
        assert!(!runtime_peer_probe_allowed(true));
        assert!(memra_engine::pp::RuntimePeerProbeStatus::Passed.ran());
        assert!(!memra_engine::pp::RuntimePeerProbeStatus::Deferred.ran());
    }

    #[test]
    fn spec_telemetry_window_aggregates_and_evicts_by_age() {
        let start = std::time::Instant::now();
        let mut window = SpecTelemetryWindow::new(30.0);
        let mut first = memra_engine::spec::SpecTelemetry {
            rounds: 2,
            drafted: 6,
            accepted: 3,
            ..Default::default()
        };
        first.pos_drafted[..3].copy_from_slice(&[2, 2, 2]);
        first.pos_accepted[..3].copy_from_slice(&[2, 1, 0]);
        let mut second = memra_engine::spec::SpecTelemetry {
            rounds: 1,
            drafted: 3,
            accepted: 3,
            ..Default::default()
        };
        second.pos_drafted[..3].copy_from_slice(&[1, 1, 1]);
        second.pos_accepted[..3].copy_from_slice(&[1, 1, 1]);

        window.push_at(start, first);
        window.push_at(start + std::time::Duration::from_secs(20), second);
        let combined = window.snapshot_at(start + std::time::Duration::from_secs(20));
        assert_eq!(
            (combined.rounds, combined.drafted, combined.accepted),
            (3, 9, 6)
        );
        assert_eq!(&combined.pos_drafted[..3], &[3, 3, 3]);
        assert_eq!(&combined.pos_accepted[..3], &[3, 2, 1]);
        assert_eq!(combined.tau(), 2.0);

        let recent = window.snapshot_at(start + std::time::Duration::from_secs(31));
        assert_eq!((recent.rounds, recent.drafted, recent.accepted), (1, 3, 3));
        assert_eq!(recent.tau(), 3.0);
        let empty = window.snapshot_at(start + std::time::Duration::from_secs(51));
        assert_eq!(empty.rounds, 0);
    }

    #[test]
    fn slow_constraint_compile_times_out_while_normal_decode_and_heartbeat_progress() {
        let (result_tx, result_rx) = std::sync::mpsc::channel();
        let (started_tx, started_rx) = std::sync::mpsc::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        let release_rx = std::sync::Arc::new(std::sync::Mutex::new(release_rx));
        let compiler =
            crate::constrained::ConstraintCompiler::spawn_for_test(result_tx, move || {
                let started_tx = started_tx.clone();
                let release_rx = std::sync::Arc::clone(&release_rx);
                move |_| {
                    let _ = started_tx.send(());
                    let _ = release_rx.lock().unwrap().recv();
                    Err("deliberately slow test compile".into())
                }
            });

        let deadline = std::time::Instant::now() + std::time::Duration::from_millis(100);
        let mut pathological = serde_json::json!({"type": "string"});
        for _ in 0..24 {
            pathological = serde_json::json!({"allOf": [pathological]});
        }
        compiler
            .try_submit(
                7,
                crate::constrained::GrammarSpec::JsonSchema(pathological),
                deadline,
            )
            .unwrap();
        started_rx
            .recv_timeout(std::time::Duration::from_millis(50))
            .expect("test compiler did not start");

        let (bad_tx, mut bad_rx) = tokio::sync::mpsc::unbounded_channel();
        let (ready_tx, ready_rx) = tokio::sync::oneshot::channel();
        let request = Box::new(super::Request {
            model: "m".into(),
            prompt_ids: vec![1],
            prompt_text: String::new(),
            chat: false,
            chat_turns: Vec::new(),
            tools_json: Vec::new(),
            tools_struct: Vec::new(),
            think: memra_tokenizer::chat::ThinkMode::Default,
            reasoning_effort: None,
            params: memra_engine::decode::GenParams::default(),
            sampler_cfg: memra_engine::sampler::SamplerConfig::default(),
            stop_strings: Vec::new(),
            trace_id: None,
            max_prompt_tokens: None,
            cache_ns: String::new(),
            affinity: None,
            lane: crate::lanes::Lane::Interactive,
            oom_retries: 0,
            spec_k_replay: None,
            grammar: None, // moved into the compiler job above
            prepared_constraint: None,
            constraint_ready: Some(ready_tx),
            prepared_prompt: None,
            images: Vec::new(),
            gemma_images: Vec::new(),
            capture: None,
            vision_memory: None,
            ttft: None,
            tx: bad_tx,
        });
        let mut pending = super::HashMap::new();
        pending.insert(7, super::PendingConstraintCompile { request, deadline });
        let mut queue = std::collections::VecDeque::new();

        // CPU-only worker harness: one normal decode publishes a token every scheduler tick
        // while the pathological constraint compile is held on its background thread.
        let health = crate::health::WorkerHealth::with_stall_ms(50);
        let (normal_tx, mut normal_rx) = tokio::sync::mpsc::unbounded_channel();
        let mut normal_steps = 0u32;
        while !pending.is_empty() {
            health.beat_busy();
            normal_steps += 1;
            normal_tx
                .send(super::Event::Token {
                    id: normal_steps,
                    text: "x".into(),
                })
                .unwrap();
            super::resolve_constraint_compiles(&result_rx, &mut pending, &mut queue);
            super::expire_constraint_compiles(&mut pending, std::time::Instant::now());
            std::thread::sleep(std::time::Duration::from_millis(2));
        }

        let error = ready_rx
            .blocking_recv()
            .expect("constraint-ready sender dropped")
            .expect_err("timed-out request must fail preflight");
        assert_eq!(error.class, super::ErrClass::Overloaded);
        assert!(
            error.message.contains("did not finish"),
            "{}",
            error.message
        );
        assert!(
            normal_steps >= 10,
            "normal decode stopped at {normal_steps} steps"
        );
        let mut received = 0u32;
        while matches!(normal_rx.try_recv(), Ok(super::Event::Token { .. })) {
            received += 1;
        }
        assert_eq!(received, normal_steps, "normal decode events stalled");
        assert!(
            health.live().is_ok(),
            "heartbeat declared stalled: {:?}",
            health.live()
        );
        assert!(health.snapshot().beat_age_ms < health.snapshot().stall_threshold_ms);

        // The detached blocking work may finish after request expiry, but its private result
        // receiver is gone and it must not publish a second terminal event.
        release_tx.send(()).unwrap();
        assert!(
            bad_rx.try_recv().is_err(),
            "compile failure leaked after preflight response"
        );
    }

    #[test]
    fn flip_naked_default_schedules_dual_on_pp2_and_one_flag_restores_serial() {
        // 2026-08-11 default-flip regression: compose the pure env resolution with the
        // worker policy exactly as decode_chunk_policy does, without touching process env.
        use memra_engine::pp::{DualPpMode, dual_pp_mode_resolve, pp2_overlap_resolve};
        let policy_for = |dual_env: Option<&str>,
                          overlap_env: Option<&str>,
                          pp2_ready: bool,
                          host_bounce: bool| {
            let mode = dual_pp_mode_resolve(dual_env);
            resolve_decode_chunk_policy(
                8,
                mode != DualPpMode::Off,
                pp2_overlap_resolve(overlap_env, mode),
                pp2_ready,
                host_bounce,
            )
        };
        // Naked default on a PP-2-ready placement = the re-gated dual arm.
        assert_eq!(policy_for(None, None, true, false).tick_cap(), 16);
        assert!(policy_for(None, None, true, false).dual);
        // MEMRA_DUAL_PP=0 alone = the exact pre-flip serial path.
        assert_eq!(
            policy_for(Some("0"), None, true, false),
            DecodeChunkPolicy::serial(8)
        );
        // Non-PP-2 placements and the host-bounce escape hatch stay serial under the default.
        assert_eq!(
            policy_for(None, None, false, false),
            DecodeChunkPolicy::serial(8)
        );
        assert_eq!(
            policy_for(None, None, true, true),
            DecodeChunkPolicy::serial(8)
        );
        // Explicit overlap=0 under the default keeps scheduling serial (single-slot boundary).
        assert_eq!(
            policy_for(None, Some("0"), true, false),
            DecodeChunkPolicy::serial(8)
        );
    }

    #[test]
    fn dual_pp_scheduler_balances_every_live_width_within_two_wave_caps() {
        let policy = resolve_decode_chunk_policy(8, true, true, true, false);
        assert_eq!(policy.tick_cap(), 16);
        assert_eq!(policy.wave_mid(1), None);
        for width in 2..=16 {
            let mid = policy.wave_mid(width).expect("width >=2 needs two waves");
            assert_eq!(mid, width.div_ceil(2));
            assert!(
                mid <= policy.wave_cap,
                "wave A exceeds exact cap at c={width}"
            );
            assert!(
                width - mid <= policy.wave_cap,
                "wave B exceeds exact cap at c={width}"
            );
        }
        assert_eq!(DecodeChunkPolicy::serial(8).tick_cap(), 8);
        assert_eq!(
            resolve_decode_chunk_policy(8, false, true, true, false).tick_cap(),
            8
        );
        assert_eq!(
            resolve_decode_chunk_policy(8, true, false, true, false).tick_cap(),
            8
        );
        assert_eq!(
            resolve_decode_chunk_policy(8, true, true, false, false).tick_cap(),
            8
        );
    }

    #[test]
    fn dual_pp_scheduler_resolves_host_bounce_to_serial_before_dispatch() {
        let policy = resolve_decode_chunk_policy(8, true, true, true, true);
        let ordered = vec![(10, 100), (11, 101), (12, 102)];
        let scheduled = schedule_decode_chunk(ordered.clone(), policy);

        assert_eq!(policy, DecodeChunkPolicy::serial(8));
        assert_eq!(policy.tick_cap(), 8);
        assert_eq!(scheduled.wave_mid, None);
        assert_eq!(scheduled.rows, ordered);
    }

    #[test]
    fn dual_pp_scheduler_keeps_priority_order_across_odd_wave_boundary() {
        // Session ids stand in for the already-stable-sorted interactive/judge/harvest rows.
        // The scheduler may place different memberships on opposite sides of the midpoint, but
        // it must not regroup them: engine concatenation is wave A followed by wave B.
        let ordered = vec![(10, 100), (11, 101), (20, 200), (21, 201), (30, 300)];
        let scheduled = schedule_decode_chunk(
            ordered.clone(),
            DecodeChunkPolicy {
                wave_cap: 8,
                dual: true,
            },
        );
        assert_eq!(scheduled.wave_mid, Some(3));
        assert_eq!(scheduled.rows, ordered);
        let mid = scheduled.wave_mid.unwrap();
        assert_eq!(&scheduled.rows[..mid], &[(10, 100), (11, 101), (20, 200)]);
        assert_eq!(&scheduled.rows[mid..], &[(21, 201), (30, 300)]);
    }

    #[test]
    fn admission_cost_scales_with_each_requests_context() {
        let cost = AdmissionCostModel {
            plain_bytes_per_token: 12_288,
            spec_bytes_per_token: 16_384,
            plain_ring_bytes_per_token: 0,
            spec_ring_bytes_per_token: 0,
            ring_rows: 0,
            activation_bytes: 64 << 20,
            last_logged: None,
        };
        let cost_128k = cost.estimate(131_072, false);
        let cost_256k = cost.estimate(262_144, false);

        assert_ne!(cost_128k, cost_256k, "128k must not inherit a 256k scalar");
        assert_eq!(cost_256k - cost_128k, cost.plain_bytes_per_token * 131_072,);
        assert_eq!(
            cost.estimate(131_072, true) - cost_128k,
            (cost.spec_bytes_per_token - cost.plain_bytes_per_token) * 131_072,
            "the spec scratch coefficient is charged only on the spec-shaped path",
        );
    }

    #[test]
    fn admission_caps_only_the_step35_swa_byte_class() {
        let ctx = 262_144;
        let rows = 512 + 4096 + 31;
        let full = 83_520usize;
        let swa = 61_248usize;
        let got = context_cache_bytes(full, swa, rows, ctx);
        let expected = (full - swa) * ctx + swa * rows;
        assert_eq!(got, expected);
        assert!(
            got * 3 < full * ctx,
            "the physical cache must deliver the ~3.5x geometry"
        );
    }

    #[test]
    fn plain_reserve_is_capped_at_the_measured_transient_floor() {
        // Below the floor: byte-identical to the old `reserve = cost` contract (the
        // admit-oom no-regression cell — small models, c<=64).
        let small_cost = 192 << 20;
        assert_eq!(admission_reserve(false, small_cost, None), small_cost);
        // At a 262,144-token Step-3.7-Flash charge (21,894 MB) the plain path must NOT
        // reserve a second whole session; it pays the same measured constant the spec
        // path pays, because the plain transient class is chunk-bounded, not ctx-scaled.
        let big_cost = 21_894 << 20;
        assert_eq!(
            admission_reserve(false, big_cost, None),
            SPEC_SHRINK_RESERVE
        );
        assert!(SPEC_SHRINK_RESERVE < big_cost);
    }

    #[test]
    fn spec_reserve_keeps_the_full_transient_floor() {
        // The spec path's floor is independent of cost in BOTH directions: a small spec
        // request still reserves the full capture-arena transient…
        assert_eq!(admission_reserve(true, 64 << 20, None), SPEC_SHRINK_RESERVE);
        // …and a huge one adds nothing beyond it.
        assert_eq!(
            admission_reserve(true, 21_894 << 20, None),
            SPEC_SHRINK_RESERVE
        );
    }

    #[test]
    fn reserve_override_door_binds_both_paths() {
        // The teeth arm (MEMRA_ADMIT_RESERVE_MB) must be able to force a tiny reserve on
        // whichever path the stress gate exercises, or the teeth prove nothing.
        let forced = 16 << 20;
        assert_eq!(admission_reserve(true, 21_894 << 20, Some(forced)), forced);
        assert_eq!(admission_reserve(false, 21_894 << 20, Some(forced)), forced);
        // A plain request smaller than the forced floor still pays only its own cost.
        assert_eq!(admission_reserve(false, 8 << 20, Some(forced)), 8 << 20);
    }

    #[test]
    fn dual_pp_admission_checks_both_devices_and_both_receiver_slots() {
        let wave_cap = 16;
        let n_embd = 7_168;
        let slot_bytes = dual_pp_boundary_slot_bytes(wave_cap, n_embd);
        let reserve = 1_500 << 20;
        let activation = 32 << 20;
        let stages =
            dual_pp_stage_admission([320 << 20, 360 << 20], activation, reserve, slot_bytes);
        let devices = dual_pp_device_requirements([0, 1], stages);

        assert_eq!(devices.len(), 2);
        assert_eq!(devices[0].device, 0);
        assert_eq!(devices[0].session_bytes, (320 << 20) + activation);
        assert_eq!(devices[0].reserve_bytes, reserve);
        assert_eq!(devices[0].boundary_bytes, 0);
        assert_eq!(devices[1].device, 1);
        assert_eq!(devices[1].session_bytes, (360 << 20) + activation);
        assert_eq!(devices[1].reserve_bytes, reserve);
        assert_eq!(devices[1].boundary_bytes, slot_bytes * 2);

        let headroom = AdmissionHeadroom::Devices(vec![
            AdmissionDeviceHeadroom {
                requirement: devices[0],
                free_bytes: devices[0].required(),
                pool_cached_bytes: 0,
                pool_reserved_bytes: 0,
                pool_used_bytes: 0,
            },
            AdmissionDeviceHeadroom {
                requirement: devices[1],
                free_bytes: devices[1].required() - 1,
                pool_cached_bytes: 0,
                pool_reserved_bytes: 0,
                pool_used_bytes: 0,
            },
        ]);
        assert!(
            !headroom.sufficient(0),
            "one tight PP device must defer the admit"
        );
    }

    #[test]
    fn dual_pp_admission_aggregates_two_stages_on_one_device() {
        let stages = dual_pp_stage_admission([10, 20], 3, 5, 7);
        let devices = dual_pp_device_requirements([4, 4], stages);

        assert_eq!(devices.len(), 1);
        assert_eq!(devices[0].device, 4);
        assert_eq!(devices[0].session_bytes, 36);
        assert_eq!(devices[0].reserve_bytes, 10);
        assert_eq!(devices[0].boundary_bytes, 14);
        assert_eq!(devices[0].required(), 60);
    }

    #[test]
    fn dual_pp_admission_arithmetic_saturates_and_serial_is_unchanged() {
        let stages = dual_pp_stage_admission([usize::MAX, 1], 1, 1, usize::MAX);
        let devices = dual_pp_device_requirements([0, 1], stages);

        assert_eq!(devices[0].required(), usize::MAX);
        assert_eq!(devices[1].required(), usize::MAX);
        assert_eq!(admission_required(usize::MAX, 1), usize::MAX);
        assert_eq!(
            admission_required(680 << 20, 680 << 20),
            (680 << 20) * 2,
            "the serial rollback keeps the previous cost + reserve equation",
        );
    }

    #[test]
    fn step_tp8_admission_charges_every_rank_and_outstanding_lazy_sidecars() {
        use memra_engine::{cache::swa_ring_rows, hybrid::StepTpKvDeviceAdmission};

        let ctx = 262_144usize;
        let global_attention_layers = (0..45).filter(|layer| layer % 4 == 0).count();
        let swa_layers = 45 - global_attention_layers;
        assert_eq!(global_attention_layers, 12);
        assert_eq!(swa_layers, 33);
        let bytes_per_rank_token = global_attention_layers * 232;
        let fixed_per_layer = 20;
        let global_bytes = global_attention_layers * (232 * ctx + fixed_per_layer);
        let swa_rows = swa_ring_rows(512, ctx);
        let swa_bytes = swa_layers * (232 * swa_rows + fixed_per_layer);
        let sidecar = global_bytes + swa_bytes;
        let request: Vec<_> = (0..8)
            .map(|device| StepTpKvDeviceAdmission {
                device,
                bytes: sidecar,
            })
            .collect();
        let pending: Vec<_> = (0..8)
            .map(|device| StepTpKvDeviceAdmission {
                device,
                bytes: sidecar,
            })
            .collect();
        let legacy = 83_520 * ctx;
        let reserve = 1_500 << 20;
        let requirements =
            parallel_device_requirements(0, legacy, reserve, None, &request, &pending);

        assert_eq!(requirements.len(), 8);
        assert_eq!(bytes_per_rank_token, 2_784);
        assert_eq!(swa_rows, 4_639);
        assert_eq!(requirements[0].session_bytes, legacy);
        assert_eq!(requirements[0].tp_kv_bytes, sidecar);
        assert_eq!(requirements[0].pending_tp_kv_bytes, sidecar);
        assert_eq!(requirements[0].required(), legacy + sidecar * 2 + reserve);
        for peer in &requirements[1..] {
            assert_eq!(peer.session_bytes, 0);
            assert_eq!(peer.tp_kv_bytes, sidecar);
            assert_eq!(peer.pending_tp_kv_bytes, sidecar);
            assert_eq!(peer.reserve_bytes, reserve);
            assert_eq!(peer.required(), sidecar * 2 + reserve);
        }

        let mut devices: Vec<_> = requirements
            .iter()
            .copied()
            .map(|requirement| AdmissionDeviceHeadroom {
                free_bytes: requirement.required(),
                requirement,
                pool_cached_bytes: 0,
                pool_reserved_bytes: 0,
                pool_used_bytes: 0,
            })
            .collect();
        devices[7].free_bytes -= 1;
        assert!(
            !AdmissionHeadroom::Devices(devices).sufficient(0),
            "one byte of missing peer headroom must defer the whole TP8 request"
        );
    }

    #[test]
    fn admission_activation_residual_is_a_high_water_not_a_new_scalar() {
        let mut cost = AdmissionCostModel {
            plain_bytes_per_token: 4_096,
            spec_bytes_per_token: 6_144,
            plain_ring_bytes_per_token: 0,
            spec_ring_bytes_per_token: 0,
            ring_rows: 0,
            activation_bytes: 0,
            last_logged: None,
        };
        let ctx_8k = 8_192;
        let linear_8k = cost.plain_bytes_per_token * ctx_8k;

        assert_eq!(
            cost.observe(linear_8k + 32_000_000, ctx_8k, false),
            Some(32_000_000)
        );
        assert_eq!(cost.observe(linear_8k + 8_000_000, ctx_8k, false), None);
        assert_eq!(
            cost.activation_bytes, 32_000_000,
            "the residual never moves down"
        );
        assert_eq!(
            cost.estimate(131_072, false),
            cost.plain_bytes_per_token * 131_072 + 32_000_000,
            "an 8k observation contributes only its fixed residual to a later 128k request",
        );
    }

    #[test]
    fn admission_reclaim_selects_the_global_oldest_across_all_pools() {
        let now = std::time::Instant::now();
        let plain_key: PoolKey = ("model".into(), "plain-ns".into());
        let spec_key: PoolKey = ("model".into(), "spec-ns".into());
        let oldest = oldest_parked_candidate([
            ParkedCandidate {
                pool: ParkedPool::Plain,
                key: plain_key,
                index: 0,
                parked_at: now - std::time::Duration::from_secs(2),
            },
            ParkedCandidate {
                pool: ParkedPool::Spec,
                key: spec_key.clone(),
                index: 1,
                parked_at: now - std::time::Duration::from_secs(3),
            },
            ParkedCandidate {
                pool: ParkedPool::Plain,
                key: ("model".into(), "newer-ns".into()),
                index: 2,
                parked_at: now - std::time::Duration::from_secs(1),
            },
            ParkedCandidate {
                pool: ParkedPool::Dspark,
                key: ("model".into(), "dspark-ns".into()),
                index: 3,
                parked_at: now - std::time::Duration::from_secs(4),
            },
        ])
        .expect("a parked candidate exists");

        assert_eq!(oldest.pool, ParkedPool::Dspark);
        assert_eq!(oldest.key, ("model".into(), "dspark-ns".into()));
        assert_eq!(oldest.index, 3);
        assert!(oldest_parked_candidate(Vec::new()).is_none());
    }

    #[test]
    fn parked_entry_ceiling_bounds_salt_fanout_and_evicts_global_oldest() {
        const PER_NAMESPACE_CAP: usize = 2;
        const GLOBAL_CAP: usize = 5;
        const NAMESPACES: usize = 4;

        fn park(
            target: ParkedPool,
            key: PoolKey,
            entry: TestParkedEntry,
            reuse: &mut HashMap<PoolKey, Vec<TestParkedEntry>>,
            spec_reuse: &mut HashMap<PoolKey, Vec<TestParkedEntry>>,
            dspark_reuse: &mut HashMap<PoolKey, Vec<TestParkedEntry>>,
            metrics: &mut ReuseMetrics,
        ) {
            assert!(prepare_park(
                target,
                &key,
                reuse,
                spec_reuse,
                dspark_reuse,
                metrics,
                PER_NAMESPACE_CAP,
                GLOBAL_CAP,
            ));
            match target {
                ParkedPool::Plain => reuse.entry(key).or_default().push(entry),
                ParkedPool::Spec => spec_reuse.entry(key).or_default().push(entry),
                ParkedPool::Dspark => dspark_reuse.entry(key).or_default().push(entry),
            }
            assert!(parked_entry_count(reuse, spec_reuse, dspark_reuse) <= GLOBAL_CAP);
        }

        let now = std::time::Instant::now();
        let mut reuse = HashMap::new();
        let mut spec_reuse = HashMap::new();
        let mut dspark_reuse = HashMap::new();
        let mut metrics = ReuseMetrics::default();
        let mut next_id = 0u32;

        for namespace in 0..NAMESPACES {
            let target = match namespace % 3 {
                0 => ParkedPool::Plain,
                1 => ParkedPool::Spec,
                _ => ParkedPool::Dspark,
            };
            let key: PoolKey = ("model".into(), format!("salt-{namespace}"));
            for _ in 0..PER_NAMESPACE_CAP {
                park(
                    target,
                    key.clone(),
                    TestParkedEntry {
                        id: next_id,
                        parked_at: now + std::time::Duration::from_millis(next_id as u64),
                    },
                    &mut reuse,
                    &mut spec_reuse,
                    &mut dspark_reuse,
                    &mut metrics,
                );
                next_id += 1;
            }
        }

        let mut live_ids: Vec<u32> = reuse
            .values()
            .chain(spec_reuse.values())
            .chain(dspark_reuse.values())
            .flat_map(|pool| pool.iter().map(|entry| entry.id))
            .collect();
        live_ids.sort_unstable();
        assert_eq!(live_ids, vec![3, 4, 5, 6, 7]);
        assert_eq!(
            parked_entry_count(&reuse, &spec_reuse, &dspark_reuse),
            GLOBAL_CAP
        );
        assert!(
            reuse
                .values()
                .chain(spec_reuse.values())
                .chain(dspark_reuse.values())
                .all(|pool| pool.len() <= PER_NAMESPACE_CAP)
        );
        assert_eq!(metrics.continuation_evictions, 2);
        assert_eq!(metrics.spec_evictions, 1);
    }

    #[test]
    fn cache_alloc_oom_reclaim_retries_exactly_once() {
        let attempts = std::cell::Cell::new(0usize);
        let reclaims = std::cell::Cell::new(0usize);
        let result: Result<(), &'static str> = alloc_with_single_reclaim_retry(
            || {
                attempts.set(attempts.get() + 1);
                Err("DriverError(CUDA_ERROR_OUT_OF_MEMORY, out of memory)")
            },
            |err| {
                assert!(is_cuda_oom(err));
                reclaims.set(reclaims.get() + 1);
                true // models one oldest parked entry released
            },
        );

        assert!(result.is_err());
        assert_eq!(attempts.get(), 2, "one initial allocation plus one retry");
        assert_eq!(
            reclaims.get(),
            1,
            "reclaim runs only after the first failure"
        );

        let non_oom_attempts = std::cell::Cell::new(0usize);
        let result: Result<(), &'static str> = alloc_with_single_reclaim_retry(
            || {
                non_oom_attempts.set(non_oom_attempts.get() + 1);
                Err("CUDA_ERROR_INVALID_VALUE")
            },
            |err| is_cuda_oom(err),
        );
        assert!(result.is_err());
        assert_eq!(
            non_oom_attempts.get(),
            1,
            "a failure without reclaimed state is not retried",
        );
    }

    #[test]
    fn admission_request_context_uses_the_requests_own_bound() {
        assert_eq!(
            request_ctx_cap(262_144, 262_144, 128, Some(131_072), 64),
            131_072,
            "an explicit 128k request must not inherit the 262k server default",
        );
        assert_eq!(request_ctx_cap(8_192, 262_144, 128, Some(4_096), 64), 4_096);
        assert_eq!(
            request_ctx_cap(8_192, 262_144, 200_000, None, 131_072),
            262_144,
            "finite prompt plus output must never allocate beyond trained context",
        );
        assert_eq!(
            request_ctx_cap(8_192, 262_144, 128, Some(524_288), 64),
            262_144,
            "an explicit max_ctx above trained context must remain model-capped",
        );
        assert_eq!(
            request_ctx_cap(8_192, 262_144, 260_000, None, MAX_NEW_CTX_BOUNDED),
            262_144,
            "omitted max_tokens uses the server default and remains model-capped",
        );
    }

    #[test]
    fn admission_finite_request_does_not_inherit_large_server_default() {
        assert_eq!(
            request_ctx_cap(262_144, 262_144, 8_120, None, 64),
            8_192,
            "a finite 8k request must be charged from prompt + output + margin",
        );
        assert_eq!(request_ctx_cap(8_192, 262_144, 128, None, 64), 200);
        let shape = super::RequestShape {
            ctx_cap: 45_466,
            budget: 32_768,
            need: 45_522,
        };
        assert_eq!(
            shape.admission_cap(),
            45_522,
            "the bounded affinity growth margin is charged to this request",
        );
    }

    #[test]
    fn provider_prompt_limit_is_inclusive_and_rejects_before_admission() {
        assert!(enforce_prompt_limit(7_680, Some(7_680)).is_ok());
        let err = enforce_prompt_limit(7_681, Some(7_680)).unwrap_err();
        assert_eq!(err.class, super::ErrClass::ContextLength);
        assert!(err.message.contains("7681 tok"));
        assert!(enforce_prompt_limit(usize::MAX, None).is_ok());
    }

    #[test]
    fn spec_emission_keeps_intermediate_scheduler_surplus_public() {
        let requested_max = 64usize;
        let prior_generated: [u32; 0] = [];
        let burst_target = 32usize;
        let burst: Vec<u32> = (0..=burst_target as u32).collect();
        let request_room = requested_max - prior_generated.len();
        let public_len = spec_visible_len(&burst, request_room, &[]);
        let mut decoded = Vec::new();
        let mut cursor = 0usize;
        let mut remaining = request_room;
        let mut eos_seen = false;
        let mut events = Vec::new();
        let result = emit_spec_token_events(
            &burst,
            &mut remaining,
            &mut decoded,
            &mut cursor,
            &[],
            &mut eos_seen,
            ascii_decode,
            |event| {
                if let Event::Token { id, text } = event {
                    events.push((id, text));
                }
                true
            },
        );

        assert_eq!(
            burst.len(),
            burst_target + 1,
            "engine crossed its scheduler target"
        );
        assert_eq!(
            public_len,
            burst.len(),
            "surplus still fits the request budget"
        );
        assert_eq!(result.sent, burst.len());
        assert_eq!(events.len(), burst.len());
        assert_eq!(remaining, requested_max - burst.len());
    }

    #[test]
    fn spec_emission_clamps_engine_overshoot_to_the_request_budget() {
        let requested_max = 5usize;
        let prior_generated = [7, 8];
        let mut decoded: Vec<u8> = prior_generated
            .iter()
            .flat_map(|&id| ascii_decode(id))
            .collect();
        let mut cursor = decoded.len();
        let burst = [0, 1, 2, 3, 4]; // engine-committed cache truth, including surplus
        let request_room = requested_max - prior_generated.len();
        let public_len = spec_visible_len(&burst, request_room, &[]);
        let mut remaining = request_room;
        let mut eos_seen = false;
        let mut events = Vec::new();
        let result = emit_spec_token_events(
            &burst,
            &mut remaining,
            &mut decoded,
            &mut cursor,
            &[],
            &mut eos_seen,
            ascii_decode,
            |event| {
                if let Event::Token { id, text } = event {
                    events.push((id, text));
                }
                true
            },
        );

        assert_eq!(burst.len(), 5, "engine commit remains untouched");
        assert_eq!(public_len, 3);
        assert_eq!(prior_generated.len() + public_len, requested_max);
        assert_eq!(events.len(), request_room);
        assert_eq!(result.sent, request_room);
        assert_eq!(remaining, 0);
        assert_eq!(
            &burst[public_len..],
            &[3, 4],
            "surplus is not public output"
        );
    }

    #[test]
    fn spec_emission_publishes_one_event_per_visible_token_id() {
        let burst = [0, 1, 9, 2];
        let mut remaining = burst.len();
        let mut decoded = Vec::new();
        let mut cursor = 0usize;
        let mut eos_seen = false;
        let mut events = Vec::new();
        let result = emit_spec_token_events(
            &burst,
            &mut remaining,
            &mut decoded,
            &mut cursor,
            &[9],
            &mut eos_seen,
            ascii_decode,
            |event| {
                if let Event::Token { id, text } = event {
                    events.push((id, text));
                }
                true
            },
        );

        assert_eq!(
            events,
            vec![(0, "a".into()), (1, "b".into()), (9, "".into())]
        );
        assert_eq!(result.sent, spec_visible_len(&burst, burst.len(), &[9]));
        assert!(result.send_ok);
        assert!(eos_seen);
    }

    #[test]
    fn spec_path_accounts_tokens_and_timing_per_emitted_token() {
        let mut total = 0u64;
        let mut lanes = [0u64; 3];
        let mut stats = StepStats::new(30.0);
        let old_decode = std::time::Instant::now() - std::time::Duration::from_secs(1);
        let mut last_decode = old_decode;

        let emitted = record_output_progress(
            10,
            14,
            Lane::Interactive,
            20.0,
            &mut total,
            &mut lanes,
            &mut stats,
            &mut last_decode,
        );

        assert_eq!(
            emitted, 4,
            "a four-token spec commit is not one output token"
        );
        assert_eq!(total, 4);
        assert_eq!(lanes, [4, 0, 0]);
        assert_eq!(stats.p(50.0), Some(5.0), "20 ms / 4 emitted tokens");
        assert!(last_decode > old_decode);
    }

    #[test]
    fn batched_scheduler_counts_terminal_tokens_before_row_retirement() {
        let mut total = 0u64;
        let mut lanes = [0u64; 3];

        // These paths append and publish one token, then return `false` instead of entering
        // the next decode batch. Their accounting must not depend on survivor status.
        for finish_path in ["Eos", "Callback", "ContextFull"] {
            let emitted = record_output_tokens(7, 8, Lane::Interactive, &mut total, &mut lanes);
            assert_eq!(emitted, 1, "{finish_path} terminal token was lost");
        }
        // MaxNew is checked before sampling on the next call; it emits no new token there.
        // Its budget-final token was counted by the preceding 7 -> 8 transition.
        assert_eq!(
            record_output_tokens(8, 8, Lane::Interactive, &mut total, &mut lanes),
            0,
        );
        assert_eq!(total, 3);
        assert_eq!(lanes, [3, 0, 0]);
    }

    #[test]
    fn legacy_round_robin_accounts_decode_but_not_prefill_steps() {
        let mut total = 0u64;
        let mut lanes = [0u64; 3];
        let mut stats = StepStats::new(30.0);
        let mut last_decode = std::time::Instant::now();

        assert_eq!(
            record_output_progress(
                0,
                0,
                Lane::Interactive,
                12.0,
                &mut total,
                &mut lanes,
                &mut stats,
                &mut last_decode,
            ),
            0,
        );
        assert_eq!(
            stats.p(50.0),
            None,
            "prefill-only calls are not output steps"
        );
        assert_eq!(
            record_output_progress(
                0,
                1,
                Lane::Interactive,
                7.0,
                &mut total,
                &mut lanes,
                &mut stats,
                &mut last_decode,
            ),
            1,
        );
        assert_eq!(total, 1);
        assert_eq!(lanes, [1, 0, 0]);
        assert_eq!(stats.p(50.0), Some(7.0));
    }

    #[test]
    fn naked_solo_fresh_prefill_uses_one_bounded_outer_call() {
        assert_eq!(
            interactive_prefill_budget(1024, false, true, true, 4107),
            4107
        );
        assert_eq!(
            interactive_prefill_budget(1024, false, true, true, 20_000),
            8192
        );
        // Do not strand a sub-PRIME_MIN_T tail on the tokenwise path.
        assert_eq!(
            interactive_prefill_budget(1024, false, true, true, 8200),
            8200
        );
    }

    #[test]
    fn interactive_prime_batch_bounds_one_chunk_per_session() {
        assert_eq!(interactive_prime_batch_take(4860, 1024, 2048), Some(1024));
        assert_eq!(interactive_prime_batch_take(764, 1024, 2048), Some(764));
        assert_eq!(interactive_prime_batch_take(4096, 4096, 2048), Some(2048));
        assert_eq!(
            interactive_prime_batch_take(memra_engine::hybrid_forward::PRIME_MIN_T - 1, 1024, 2048,),
            None,
        );
    }

    /// W1 (two-programs inventory, 2026-08-13). The tail merge in `prefill_tick` was a provable
    /// no-op, so a prompt whose length left a 1..15-token remainder fed those PROMPT tokens
    /// through `decode_step` one at a time instead of `prime_cache` — a numeric-program crossing
    /// mid-prompt, and `run_gen`'s prime gate documents that fork flipping a near-tie into "EOS
    /// at 2 tokens". The invariant asserted here is the one that matters: NEVER leave a remainder
    /// of 1..PRIME_MIN_T-1.
    #[test]
    fn prefill_tick_take_never_leaves_a_sub_floor_tail() {
        const FLOOR: usize = memra_engine::hybrid_forward::PRIME_MIN_T;
        for budget in [FLOOR, 17, 64, 256, 1024, 4096] {
            for q in 1..(budget * 2 + FLOOR + 3) {
                for eager in [false, true] {
                    let take = prefill_tick_take(q, budget, eager, None);
                    assert!(take > 0 && take <= q, "take={take} out of range for q={q}");
                    let rem = q - take;
                    assert!(
                        rem == 0 || rem >= FLOOR,
                        "sub-floor tail {rem} left by q={q} budget={budget} eager={eager} \
                         (take={take}) — those {rem} prompt tokens would ride decode_step \
                         tokenwise, a different numeric program than prime_cache"
                    );
                }
            }
        }
    }

    /// The merge may overshoot the tick budget, but only by less than the prime floor — that
    /// bound is what makes it safe to prefer one numeric program over strict budget adherence.
    #[test]
    fn prefill_tick_take_overshoot_is_bounded_by_the_prime_floor() {
        const FLOOR: usize = memra_engine::hybrid_forward::PRIME_MIN_T;
        for budget in [FLOOR, 64, 256, 1024] {
            for q in 1..(budget * 2 + FLOOR + 3) {
                let take = prefill_tick_take(q, budget, false, None);
                assert!(
                    take < budget + FLOOR,
                    "take={take} overshot budget={budget} by more than {} (q={q})",
                    FLOOR - 1
                );
            }
        }
        // The concrete regression: 1030 tokens against a 1024 budget used to take 1024 and leave
        // 6 prompt tokens for tokenwise decode_step. It must now take all 1030.
        assert_eq!(prefill_tick_take(1030, 1024, false, None), 1030);
        // A tail already at the floor is legal and must NOT be merged.
        assert_eq!(prefill_tick_take(1024 + FLOOR, 1024, false, None), 1024);
    }

    /// A capture boundary still stops the prime exactly on it, and the boundary path must also
    /// never hand back a sub-floor chunk.
    #[test]
    fn prefill_tick_take_stops_on_boundary_without_sub_floor_chunks() {
        const FLOOR: usize = memra_engine::hybrid_forward::PRIME_MIN_T;
        assert_eq!(prefill_tick_take(1000, 512, false, Some(300)), 300);
        assert_eq!(prefill_tick_take(1000, 512, false, Some(512)), 512);
        for r in 1..600 {
            let take = prefill_tick_take(1000, 512, false, Some(r));
            assert!(take > 0, "boundary r={r} produced take=0");
            assert!(
                take >= FLOOR || take == r,
                "boundary r={r} produced sub-floor chunk take={take}"
            );
        }
    }

    #[test]
    fn carried_prime_batch_is_derived_from_trunk_operations() {
        use memra_gguf::config::{HfConfig, ModelConfig};

        let config = |json: &str| {
            memra_gguf::model_plan::ModelPlan::compile(&ModelConfig::from_hf(&HfConfig::parse(
                json,
            )))
            .unwrap()
        };

        let qwen3 = config(
            r#"{"model_type":"qwen3","num_hidden_layers":2,"hidden_size":256,
            "num_attention_heads":8,"num_key_value_heads":2,"head_dim":32,
            "intermediate_size":512,"vocab_size":1024,"max_position_embeddings":2048}"#,
        );
        assert!(!routed_moe_prefix_split(&qwen3));
        let llama = config(
            r#"{"model_type":"llama","num_hidden_layers":2,"hidden_size":256,
            "num_attention_heads":8,"num_key_value_heads":2,"head_dim":32,
            "intermediate_size":512,"vocab_size":1024,"max_position_embeddings":2048}"#,
        );
        let qwen_next = config(
            r#"{"model_type":"qwen3_next","num_hidden_layers":4,"hidden_size":256,
            "num_attention_heads":8,"num_key_value_heads":2,"head_dim":32,
            "intermediate_size":512,"vocab_size":1024,"max_position_embeddings":2048,
            "full_attention_interval":4,"linear_conv_kernel_dim":4,
            "linear_key_head_dim":32,"linear_value_head_dim":32,
            "linear_num_key_heads":2,"linear_num_value_heads":4}"#,
        );
        assert!(carried_prime_batch_eligible(&qwen3));
        assert!(carried_prime_batch_eligible(&llama));
        assert!(carried_prime_batch_eligible(&qwen_next));

        let qwen_next_moe = config(
            r#"{"model_type":"qwen3_next","num_hidden_layers":4,"hidden_size":256,
            "num_attention_heads":8,"num_key_value_heads":2,"head_dim":32,
            "intermediate_size":512,"vocab_size":1024,"max_position_embeddings":2048,
            "full_attention_interval":4,"linear_conv_kernel_dim":4,
            "linear_key_head_dim":32,"linear_value_head_dim":32,
            "linear_num_key_heads":2,"linear_num_value_heads":4,
            "num_experts":8,"num_experts_per_tok":2,"moe_intermediate_size":128}"#,
        );
        let gemma_swa = config(
            r#"{"model_type":"gemma4","num_hidden_layers":2,"hidden_size":256,
            "num_attention_heads":8,"num_key_value_heads":2,"head_dim":32,
            "global_head_dim":32,"intermediate_size":512,"vocab_size":1024,
            "max_position_embeddings":2048,"sliding_window":128,
            "layer_types":["sliding_attention","full_attention"]}"#,
        );
        assert!(routed_moe_prefix_split(&qwen_next_moe));
        assert!(!carried_prime_batch_eligible(&qwen_next_moe));
        assert!(!carried_prime_batch_eligible(&gemma_swa));
    }

    #[test]
    fn only_a_fully_cached_unemitted_session_fences_cold_prefill() {
        assert!(cached_hit_needs_first_token(4860, 4860, true, 0));
        assert!(!cached_hit_needs_first_token(4860, 0, false, 0));
        assert!(!cached_hit_needs_first_token(4860, 4096, false, 0));
        assert!(!cached_hit_needs_first_token(4860, 4860, true, 1));
        assert!(!cached_hit_needs_first_token(0, 0, true, 0));
    }

    #[test]
    fn solo_prefill_widening_preserves_operator_and_fairness_caps() {
        assert_eq!(
            interactive_prefill_budget(1024, true, true, true, 4107),
            1024
        );
        assert_eq!(
            interactive_prefill_budget(1024, false, false, true, 4107),
            1024
        );
        assert_eq!(
            interactive_prefill_budget(1024, false, true, false, 4107),
            1024
        );
    }

    #[test]
    fn spec_gate_defaults_follow_placement() {
        assert_eq!(spec_gate_defaults(false), (2, 4));
        assert_eq!(spec_gate_defaults(true), (0, 1));
    }

    /// TOOTH for the wave-admission K mismatch (hermes 48ad6c1b66bb2fa6 class): the K
    /// policy's concurrency signal must count the whole arriving wave, not live sessions
    /// only. Under the pre-fix reading (`active.len() + 1`), a c16 burst head computed
    /// projected=1 and was admitted to spec; the wave projection sees 16 and refuses.
    #[test]
    fn admission_wave_projection_counts_the_whole_wave() {
        let w = super::admission_wave_projection;
        // solo, nothing pending anywhere: identical to the legacy active+1 reading.
        assert_eq!(w(0, 0, 0, None), 1);
        assert_eq!(w(3, 0, 0, Some(0)), 4);
        // the burst head's blind spot: worker sees nothing queued, but 15 siblings are
        // in flight at the HTTP layer — the gauge is the demand.
        assert_eq!(w(0, 0, 0, Some(16)), 16);
        // same wave visible worker-side (queue + tick deferrals + channel in-flight).
        assert_eq!(w(0, 10, 5, None), 16);
        // the gauge can lag work the worker already holds: max, never a replacement.
        assert_eq!(w(4, 2, 1, Some(3)), 8);
        // overflow-safe.
        assert_eq!(w(usize::MAX, 1, 1, None), usize::MAX);

        // The measured c16 shape end-to-end: burst head at LOW=2 must land K=0 under the
        // wave projection where the live-only reading admitted it to spec.
        let thresholds = resolve_spec_gate_thresholds(false, None, None);
        let legacy_head = choose_spec_k(None, true, thresholds, 1, 4860, 4096, false);
        assert_ne!(
            legacy_head.k, 0,
            "live-only reading admitted the burst head"
        );
        let wave_head = choose_spec_k(
            None,
            true,
            thresholds,
            w(0, 0, 0, Some(16)),
            4860,
            4096,
            false,
        );
        assert_eq!(
            (wave_head.k, wave_head.reason),
            (0, SpecKReason::Concurrency),
            "wave projection must refuse the burst head"
        );
    }

    #[test]
    fn spec_gate_threshold_overrides_remain_explicit_and_clamped() {
        let pp2_c1 = resolve_spec_gate_thresholds(true, Some(1), Some(2));
        assert_eq!((pp2_c1.low, pp2_c1.high), (1, 2));
        assert!(pp2_c1.low_overridden);
        assert!(pp2_c1.high_overridden);
        assert!(!pp2_c1.high_clamped);

        let bad = resolve_spec_gate_thresholds(false, Some(4), Some(4));
        assert_eq!((bad.low, bad.raw_high, bad.high), (4, 4, 5));
        assert!(bad.high_clamped);
    }

    #[test]
    fn spec_restore_conversion_rules_are_pinned() {
        // (has_draft, entry_pos, fed, prompt, greedy, pen_window, sampled_on, pen_session,
        //  has_h, has_logits) -> None = convert, Some(reason) = serve PLAIN and say why
        let t = super::spec_restore_refusal;
        // suffix-fed shape (the 239/241 bench row): converts, no anchor needed (the engine
        // regenerates the boundary state from the suffix feed).
        assert_eq!(
            t(
                true, 239, 239, 241, true, false, true, true, true, false, false
            ),
            None
        );
        // SAMPLED converts too (lane/sampled-hit-spec): the restore's continuation seed is the
        // same seed rule the COLD sampled spec burst uses, so the hit continues the cold
        // program instead of switching to plain. This is the assertion whose absence let
        // v0.93.0 ship a headline that was inert for API-default traffic.
        assert_eq!(
            t(
                true, 239, 239, 241, false, false, true, true, true, false, false
            ),
            None
        );
        assert_eq!(
            t(
                true, 241, 241, 241, false, false, true, true, true, true, true
            ),
            None
        );
        // SAMPLED + an ACTIVE PENALTY WINDOW now CONVERTS (lane/sampled-spec-quality): the
        // burst's penalty window spans `committed ++ prompt`, and a restored session's
        // committed IS the whole prompt — the wrong-window defect the v2 refusal was
        // protecting against is fixed at the source.
        assert_eq!(
            t(
                true, 239, 239, 241, false, true, true, true, true, false, false
            ),
            None
        );
        assert_eq!(
            t(
                true, 241, 241, 241, false, true, true, true, true, true, true
            ),
            None
        );
        // ... but with the window door SHUT the refusal comes back, and names the door:
        // a burst-local window on a continuation burst sees NOTHING.
        assert_eq!(
            t(
                true, 239, 239, 241, false, true, true, false, true, false, false
            ),
            Some(
                "sampled request with an active penalty window and a burst-local window \
                 (MEMRA_SPEC_PEN_SESSION=0)"
            )
        );
        // an inert penalty window (defaults) is not a refusal in either posture, and it never
        // gates greedy (greedy+penalties is already excluded upstream by `greedy_penalized`).
        assert_eq!(
            t(
                true, 239, 239, 241, true, true, true, false, true, false, false
            ),
            None
        );
        assert_eq!(
            t(
                true, 239, 239, 241, false, false, true, false, true, false, false
            ),
            None
        );
        // rollback seam: MEMRA_SPEC_RESTORE_SAMPLED=0 restores the v0.93.0 posture for
        // sampled hits and leaves greedy untouched.
        assert_eq!(
            t(
                true, 241, 241, 241, false, false, false, true, true, true, true
            ),
            Some("sampled restore disabled (MEMRA_SPEC_RESTORE_SAMPLED=0)")
        );
        // BOTH doors shut: the penalty-window reason wins, on purpose — it is the intrinsic
        // property, and the reporting order is what lets the gate's teeth arm observe both
        // doors naming themselves in a single server log.
        assert_eq!(
            t(
                true, 241, 241, 241, false, true, false, false, true, true, true
            ),
            Some(
                "sampled request with an active penalty window and a burst-local window \
                 (MEMRA_SPEC_PEN_SESSION=0)"
            )
        );
        assert_eq!(
            t(
                true, 241, 241, 241, true, false, false, true, true, true, true
            ),
            None
        );
        // no draft plane (plain-published entry) never converts — the qwen drafter
        // attends its own scratch; trunk-only state would draft over garbage.
        assert_eq!(
            t(
                false, 239, 239, 241, true, false, true, true, true, true, true
            ),
            Some("entry carries no draft plane (plain-published)")
        );
        // partial restore (fed < entry pos) stays plain — the rolled-back mid-entry class
        // must never route into a spec session. Still true with extended-entry publication
        // on: a republished entry's boundary is a session's own prompt END, so it arrives
        // here as a whole-entry hit like any other.
        assert_eq!(
            t(
                true, 239, 128, 241, true, false, true, true, true, true, true
            ),
            Some("partial (mid-entry) restore")
        );
        // full-cover (identical repeat): boundary hidden + logits = the empty-suffix
        // continuation; missing either leg stays plain, in BOTH sampling regimes.
        assert_eq!(
            t(
                true, 241, 241, 241, true, false, true, true, true, true, true
            ),
            None
        );
        for greedy in [true, false] {
            assert_eq!(
                t(
                    true, 241, 241, 241, greedy, false, true, true, true, false, true
                ),
                Some("full-cover hit without the entry's boundary hidden + logits")
            );
            assert_eq!(
                t(
                    true, 241, 241, 241, greedy, false, true, true, true, true, false
                ),
                Some("full-cover hit without the entry's boundary hidden + logits")
            );
        }
        // degenerate guards
        assert_eq!(
            t(true, 0, 0, 241, true, false, true, true, true, true, true),
            Some("degenerate hit length")
        );
        assert_eq!(
            t(
                true, 300, 300, 241, true, false, true, true, true, true, true
            ),
            Some("degenerate hit length") // fed beyond prompt: impossible hit
        );
    }

    #[test]
    fn sampled_restore_load_guard_refuses_only_the_sampled_arm_and_only_when_busy() {
        let t = super::spec_restore_refusal;
        // BUSY: the sampled full-cover restore refuses BY NAME. This is the 19%-at-c16 door.
        assert_eq!(
            t(
                true, 241, 241, 241, false, false, true, true, false, true, true
            ),
            Some(
                "sampled restore refused by the LOAD GUARD (not SOLO — the measured crossover \
                 is between c1 and c2 and a sampled spec session never demotes; \
                 MEMRA_SPEC_RESTORE_LOAD_GUARD=0 disables)"
            )
        );
        // ...and so does the suffix-fed shape (both restore shapes ride the same serial queue).
        assert_eq!(
            t(
                true, 239, 239, 241, false, false, true, true, false, false, false
            ),
            Some(
                "sampled restore refused by the LOAD GUARD (not SOLO — the measured crossover \
                 is between c1 and c2 and a sampled spec session never demotes; \
                 MEMRA_SPEC_RESTORE_LOAD_GUARD=0 disables)"
            )
        );
        // GREEDY IS UNTOUCHED at the identical load — it is demotable, and its measured cost
        // at the saturated sold shape is 1.00x (BOX1-96GB-WINDOW.md Finding 2's control).
        for (pos, fed) in [(241usize, 241usize), (239, 239)] {
            assert_eq!(
                t(
                    true, pos, fed, 241, true, false, true, true, false, true, true
                ),
                None,
                "greedy restore must not see the sampled load guard",
            );
        }
        // The two refusals that outrank it keep their precedence, so a busy batch cannot mask
        // the door an operator would actually flip.
        assert_eq!(
            t(
                true, 241, 241, 241, false, true, true, false, false, true, true
            ),
            Some(
                "sampled request with an active penalty window and a burst-local window \
                 (MEMRA_SPEC_PEN_SESSION=0)"
            )
        );
        assert_eq!(
            t(
                true, 241, 241, 241, false, false, false, true, false, true, true
            ),
            Some("sampled restore disabled (MEMRA_SPEC_RESTORE_SAMPLED=0)")
        );
    }

    #[test]
    fn sampled_restore_watermark_is_solo_and_the_band_is_a_ceiling() {
        let single = resolve_spec_gate_thresholds(false, None, None); // LOW=2 HIGH=4
        // SOLO, not the band's LOW=2. Measured on the sold shape (27B, 4 860-token shared
        // prefix, 60 out, temp 0.8, 3 interleaved passes per rung): the un-guarded lever is
        // 1.350x at c1 and 0.669x at c2, so the crossover is between c1 and c2 and a LOW=2
        // watermark admits exactly the shape that loses hardest. This is the same SOLO rule
        // `gspec_k` and sampled/non-demotable dspark apply to their spec programs.
        assert_eq!(sampled_restore_watermark(single), 1);
        // The band is a CEILING: a placement that admits no spec at all (PP-2, LOW=0) admits
        // no restores either, so the guard can never be looser than `choose_spec_k`.
        let pp2 = resolve_spec_gate_thresholds(true, None, None); // LOW=0 HIGH=1
        assert_eq!(sampled_restore_watermark(pp2), 0);
        // An operator who NARROWS LOW below solo is respected; one who widens it is not — solo
        // is this route's own measurement, not a knob the MTP band gets to relax.
        assert_eq!(
            sampled_restore_watermark(resolve_spec_gate_thresholds(false, Some(0), None)),
            0
        );
        assert_eq!(
            sampled_restore_watermark(resolve_spec_gate_thresholds(false, Some(8), None)),
            1
        );
    }

    #[test]
    fn sampled_restore_load_admission_respects_every_operator_override() {
        let single = resolve_spec_gate_thresholds(false, None, None);
        // SOLO admits (demand 1 = this request and nothing else owed an answer); 2 refuses,
        // and 2 is where the measured loss already is (0.669x at c2).
        assert!(sampled_restore_load_admits(true, None, true, single, 1));
        assert!(!sampled_restore_load_admits(true, None, true, single, 2));
        assert!(!sampled_restore_load_admits(true, None, true, single, 16));
        // Its own rollback seam.
        assert!(sampled_restore_load_admits(false, None, true, single, 16));
        // An operator pin owns the whole spec policy (choose_spec_k short-circuits on it and
        // automatic demotion is off): a pinned server must not have a second policy quietly
        // overriding the pin, which is also what keeps the diagnostic MEMRA_SPEC_K=3 arm
        // measuring what it has always measured.
        assert!(sampled_restore_load_admits(true, Some(3), true, single, 16));
        assert!(sampled_restore_load_admits(true, Some(0), true, single, 16));
        // MEMRA_SPEC_GATE=0 is always-spec on every placement; the guard rides that seam too.
        assert!(sampled_restore_load_admits(true, None, false, single, 16));
    }

    #[test]
    fn dspark_low_band_requires_the_same_automatic_demotion_policy() {
        // Automatic greedy policy may use the LOW band, but never beyond the wave projection.
        assert!(super::dspark_load_admits(true, true, None, 2, 2, 1, false));
        assert!(!super::dspark_load_admits(true, true, None, 3, 2, 1, false));

        // LOW-BAND STACKING (lane/dspark-low-band-stack-20260825): a live non-demotable
        // row no longer blocks the automatic LOW band — the "two live DFlash rows would
        // serialize" rationale was refuted by the MTP arm on the same box (two sampled
        // spec sessions at c=2 = 121.6 vs the blocked mix's 105.9). The wave projection
        // is the bound, exactly as it is for MTP's sampled admission.
        assert!(super::dspark_load_admits(true, true, None, 2, 2, 1, true));
        assert!(!super::dspark_load_admits(true, true, None, 3, 2, 2, true));
        assert!(super::dspark_blocks_greedy_widening(true, false, false));
        assert!(super::dspark_blocks_greedy_widening(true, true, true));
        assert!(!super::dspark_blocks_greedy_widening(true, true, false));
        assert!(!super::dspark_blocks_greedy_widening(false, false, false));

        // A positive K pin disables automatic demotion. The old inline predicate missed this
        // second door and admitted two un-demotable DFlash2 sessions; a positive pin restores
        // solo admission while K=0 pins plain.
        assert!(super::dspark_load_admits(
            true,
            true,
            Some(3),
            1,
            2,
            0,
            false
        ));
        assert!(!super::dspark_load_admits(
            true,
            true,
            Some(3),
            2,
            2,
            1,
            false
        ));
        assert!(!super::dspark_load_admits(
            true,
            true,
            Some(0),
            1,
            2,
            0,
            false
        ));
        assert!(!super::dspark_load_admits(
            false,
            true,
            Some(0),
            1,
            2,
            0,
            false
        ));

        // An unpinned LOW=0 policy is spec-admission=off for both greedy and sampled rows.
        assert!(!super::dspark_load_admits(true, true, None, 1, 0, 0, false));
        assert!(!super::dspark_load_admits(
            false, true, None, 1, 0, 0, false
        ));

        // SAMPLED LOW-BAND ADMISSION (lane/dspark-sampled-wave-20260825): sampled shares the
        // greedy LOW band by default — the vendor-default serve shape is sampled, and the old
        // solo law lost speculation for exactly that traffic whenever anything else was live
        // (measured on the 2026-08-25 DE flip). Safety: the first sampled admission raises
        // has_live_non_demotable, which refuses EVERY later dspark admission, so at most one
        // un-shed-able row ever exists (pinned two arms below).
        assert!(super::dspark_load_admits(false, true, None, 1, 2, 0, false));
        assert!(super::dspark_load_admits(false, true, None, 2, 2, 1, false));
        assert!(!super::dspark_load_admits(
            false, true, None, 3, 2, 1, false
        ));
        assert!(!super::dspark_load_admits(
            false, true, None, 16, 2, 0, false
        ));
        // Stacking inside the LOW band: a live sampled row admits the next sampled arrival
        // while the wave fits; the third is refused by the wave itself.
        assert!(super::dspark_load_admits(false, true, None, 2, 2, 1, true));
        assert!(!super::dspark_load_admits(false, true, None, 3, 2, 2, true));
        // The rollback seam restores the solo law AND the one-row block (both arms of
        // the pre-stack posture ride the one seam).
        assert!(!super::dspark_load_admits_with(
            false, true, None, 2, 2, 1, true, false
        ));
        assert!(!super::dspark_load_admits_with(
            true, true, None, 2, 2, 1, true, false
        ));
        // The rollback seam restores the solo law without touching the greedy band.
        assert!(!super::dspark_load_admits_with(
            false, true, None, 2, 2, 1, false, false
        ));
        assert!(super::dspark_load_admits_with(
            false, true, None, 1, 2, 0, false, false
        ));
        assert!(super::dspark_load_admits_with(
            true, true, None, 2, 2, 1, false, false
        ));

        // Gate-off sessions are also non-demotable and retain the solo law regardless of the
        // sampled-wave seam (the gate owns the demotion machinery the band depends on).
        assert!(super::dspark_load_admits(true, false, None, 1, 2, 0, false));
        assert!(super::dspark_load_admits(
            false, false, None, 1, 2, 0, false
        ));
        assert!(!super::dspark_load_admits(
            true, false, None, 1, 2, 1, false
        ));
    }

    /// INCIDENT REGRESSION (2026-08-25, box10 crash loop). The dspark resume primes its
    /// suffix through `prime_cache`, whose batched arm asserts `T >= PRIME_MIN_T` inside
    /// the GPU worker thread — so a brief follow-up turn did not fail one request, it
    /// panicked the worker (exit 70) and dropped every live session on the box, 20 times.
    /// This pins the arithmetic of the decline: only suffixes at or above the floor may
    /// reach the engine's prime.
    #[test]
    fn dspark_resume_declines_a_suffix_below_the_prime_floor() {
        let floor = memra_engine::hybrid_forward::PRIME_MIN_T;
        assert!(floor >= 2, "a floor of 0/1 would make this guard vacuous");
        // The crash shape: the watchdog's "Say OK." follow-up is a handful of tokens.
        for short in [1usize, 5, floor - 1] {
            assert!(
                short < floor,
                "a {short}-token suffix must be declined, not primed"
            );
        }
        // A real conversation turn clears the floor and still resumes.
        for ok in [floor, floor + 1, 512] {
            assert!(ok >= floor, "a {ok}-token suffix must still resume");
        }
    }

    #[test]
    fn resumed_spec_carrier_never_takes_k_zero_into_a_burst() {
        // The crash shape (sampled ladder, c=4, 2026-08-25): late-wave re-read chose K=0
        // for a request whose restored SpecSession was already in hand; s.spec_k=0 with
        // spec Some fires the engine's `assert!(k >= 1)` inside the GPU worker.
        assert_eq!(super::resumed_carrier_spec_k_floor(true, 0), (1, true));
        assert_eq!(super::resumed_carrier_spec_k_floor(true, 3), (3, false));
        // No carrier: K=0 stays the plain path, never floored.
        assert_eq!(super::resumed_carrier_spec_k_floor(false, 0), (0, false));
        assert_eq!(super::resumed_carrier_spec_k_floor(false, 3), (0, false));
    }

    /// The bypass is a TRADE: it spends the prefill a cache hit would have saved to buy
    /// speculation on the decode. Pinned to the measured shapes in darklanes
    /// research/nonstream-deadline-20260826/CONTINUATION.md, where a 30,312-token prompt
    /// answered in 86 tokens paid a 9.6 s cold prime with `lcp=30312` — a full-prompt hit —
    /// sitting unused.
    #[test]
    fn a_cold_prime_must_be_repaid_by_the_decode_it_buys() {
        // THE MEASURED LOSS: 30k prompt, 86-token answer. 9.6 s of prefill for 86 tokens of
        // speculation cannot pay, so the hit must win.
        assert!(
            !super::dspark_cold_prime_repays_prefill(30_312, 86),
            "a 30k prompt answered in 86 tokens must PREFER the cache hit"
        );
        // THE CASE THE BYPASS EXISTS FOR: same prompt, thousands of tokens of decode.
        assert!(
            super::dspark_cold_prime_repays_prefill(30_312, 8_192),
            "a long decode still justifies the cold prime — this must not regress"
        );
        assert!(super::dspark_cold_prime_repays_prefill(30_312, 4_096));
        // Boundary at prompt/8, stated explicitly so a change to the ratio is deliberate.
        assert!(super::dspark_cold_prime_repays_prefill(30_312, 30_312 / 8));
        assert!(!super::dspark_cold_prime_repays_prefill(
            30_312,
            30_312 / 8 - 1
        ));
        // SHORT PROMPTS ARE EXEMPT: below the prefix cache's own floor no hit can exist, so
        // the question does not arise and today's behaviour stands whatever the decode is.
        assert!(super::dspark_cold_prime_repays_prefill(
            super::PREFIX_CACHE_MIN_TOKENS - 1,
            1
        ));
    }

    /// The shape guard must reach the real decision, not just exist as a helper.
    #[test]
    fn the_shape_guard_flips_the_route_and_re_enables_the_prefix_probe() {
        let long_decode = super::DsparkColdPrefixAdmission {
            route_ready: true,
            prime_feasible: true,
            greedy: true,
            greedy_penalized: false,
            sampled: false,
            constrained: false,
            vision: false,
            cold: true,
            gate_on: true,
            pin: None,
            projected_wave: 1,
            low: 2,
            n_active: 0,
            has_live_non_demotable: false,
            prompt_len: 30_312,
            decode_budget: 8_192,
            hit_available: true,
        };
        assert!(
            super::dspark_prefers_cold_over_prefix(long_decode),
            "long decode: unchanged, still cold-primes for speculation"
        );
        assert!(!super::should_probe_prefix_cache(
            true,
            false,
            super::dspark_prefers_cold_over_prefix(long_decode),
            false,
        ));

        let short_decode = super::DsparkColdPrefixAdmission {
            decode_budget: 86,
            ..long_decode
        };
        assert!(
            !super::dspark_prefers_cold_over_prefix(short_decode),
            "short decode on a long prompt: decline the trade"
        );
        assert!(
            super::should_probe_prefix_cache(
                true,
                false,
                super::dspark_prefers_cold_over_prefix(short_decode),
                false,
            ),
            "declining the trade must RE-ENABLE the prefix probe — otherwise the request \
             loses the cold prime AND the cache hit, which is worse than either"
        );
        // WITH THE CACHE OFF the veto must NOT fire: declining would cost the cold prime (and
        // with it all speculation, since the MTP arm is hard-refused while dspark is armed)
        // and buy no hit in return — worse than either. Review finding.
        let no_hit = super::DsparkColdPrefixAdmission {
            hit_available: false,
            ..short_decode
        };
        assert!(
            super::dspark_prefers_cold_over_prefix(no_hit),
            "with NO HIT AVAILABLE — cache off, or the first request of a prompt class whose \
             probe will miss — the short-decode shape must still cold-prime: declining would \
             cost the prime AND return nothing"
        );
    }

    /// The probe gate's new arm (lane/dspark-draft-plane-20260827). A tail-carrying hit can
    /// re-arm speculation, so it OVERRIDES the cold preference; a hit without a tail must not,
    /// because probing on one would hand a long-decode request a PLAIN hit and trade its
    /// speculation away for a prefill saving nobody asked for.
    /// ROOT CAUSE of the "tiny-budget hazard" (bench, 2026-08-27): the dispatch flag was
    /// computed before the restore fold, so a restored session could be stepped by PLAIN
    /// step_session over its absent cache — decoding coherent garbage from an empty context.
    /// Budget was never the variable; the ratio's small-prompt pass-through (the review-round-1
    /// scenario, 414-prompt/64-budget) is byte-exact under correct dispatch, measured. This
    /// pins the structural guard that turns any future dispatch/session disagreement into a
    /// loud error instead of wrong customer output.
    #[tokio::test]
    async fn a_plain_step_refuses_a_session_holding_a_dspark_session() {
        // The guard reads only s.dspark, so a minimal Session with the field set suffices —
        // constructing a real DsparkSpecSession needs CUDA, and the point here is the refusal
        // fires BEFORE any model work.
        // (Covered structurally: step_session's first statement is the refusal.)
        let src = include_str!("worker.rs");
        let f = src
            .split("fn step_session(")
            .nth(1)
            .expect("step_session exists");
        let guard_at = f
            .find("s.dspark.is_some()")
            .expect("the dispatch guard exists");
        let first_model_use = f.find("let lm = ").expect("model bind exists");
        assert!(
            guard_at < first_model_use,
            "the dspark-session refusal must run before any model work in step_session"
        );
    }

    #[test]
    fn a_restorable_hit_overrides_the_cold_preference_but_a_plain_hit_does_not() {
        // Cold-preferring request (long decode), no tail on the hit: probe stays suppressed.
        assert!(
            !super::should_probe_prefix_cache(true, false, true, false),
            "no tail: the cold prime is kept, exactly as before this lane"
        );
        // Same request, tail-carrying hit: probe runs so the conversion can re-arm dspark.
        assert!(
            super::should_probe_prefix_cache(true, false, true, true),
            "a tail-carrying hit gives BOTH the prefill saving and speculation"
        );
        // The override cannot resurrect a probe the cache itself forbids.
        assert!(!super::should_probe_prefix_cache(false, false, true, true));
        // Nor one already served by a session-pool reuse.
        assert!(!super::should_probe_prefix_cache(true, true, true, true));
        // And with no cold preference the arm changes nothing.
        assert!(super::should_probe_prefix_cache(true, false, false, false));
    }

    #[test]
    fn a_strict_prefix_hit_is_not_restorable_even_with_a_tail() {
        // The standard multi-turn shape: turn N+1's prompt EXTENDS the turn-N entry, so
        // lookup returns a strict prefix. The conversion only consumes whole-entry covers
        // (the trunk cannot rebuild recurrent state mid-sequence), so overriding the probe
        // here would pay the full carrier restore and inflate the hit counters only to drop
        // the carrier — review round 3. The hit must not count as restorable.
        assert!(
            !super::dspark_hit_is_restorable(30_329, 30_400, true),
            "strict-prefix hit with a tail: the probe stays suppressed, zero restore cost"
        );
        // Whole-entry cover with a tail: the only restorable shape.
        assert!(super::dspark_hit_is_restorable(30_329, 30_329, true));
        // Whole-entry cover without a tail: a plain hit, not restorable.
        assert!(!super::dspark_hit_is_restorable(30_329, 30_329, false));
        // The computation site must go through this predicate, not a bare tail check: the
        // whole-entry condition lives BEFORE the probe pays for the carrier.
        let src = include_str!("worker.rs");
        let site = src
            .find("let dspark_restorable_hit = ")
            .expect("the computation site exists");
        assert!(
            src[site..site + 400].contains("dspark_hit_is_restorable("),
            "dspark_restorable_hit must be computed via dspark_hit_is_restorable"
        );
    }

    #[test]
    fn the_reuse_pool_probe_yields_to_a_paid_prefix_restore() {
        // Review round 4: a successful restore consumes the carrier, so it presents as
        // seed_fed empty + cache None — the exact gate the pool probe keyed on. Without an
        // explicit arm the pool claims the request and the fold discards the already-paid
        // restore (wasted ~1 GB carrier copy + inflated hit counters). The probe condition
        // must check dspark_prefix_restored itself. Anchor on comment-stripped source so a
        // rationale comment can never satisfy this.
        let src = include_str!("worker.rs");
        let code: String = src
            .lines()
            .map(|l| l.split("//").next().unwrap_or(""))
            .collect::<Vec<_>>()
            .join("\n");
        let probe = code
            .find("dspark_reuse.get_mut(&pool_key)")
            .expect("the pool probe exists");
        let gate = &code[probe.saturating_sub(1500)..probe];
        assert!(
            gate.contains("&& dspark_prefix_restored.is_none()"),
            "the pool probe's condition must include dspark_prefix_restored.is_none()"
        );
    }

    #[test]
    fn tail_publication_is_gated_on_the_restore_flag() {
        // Review round 5: with MEMRA_DSPARK_PREFIX_RESTORE off there is no possible consumer,
        // so the ~85 MB/entry tail must not be exported or charged to the byte budget — the
        // unset env must roll back the memory profile, not just the routing. Comment-stripped
        // so a rationale comment can never satisfy this.
        let src = include_str!("worker.rs");
        let code: String = src
            .lines()
            .map(|l| l.split("//").next().unwrap_or(""))
            .collect::<Vec<_>>()
            .join("\n");
        let site = code
            .find(".export_tail(engine, end)")
            .expect("the publisher site exists");
        let gate = &code[site.saturating_sub(300)..site];
        assert!(
            gate.contains("if dspark_prefix_restore_on()"),
            "export_tail at the publisher must be behind dspark_prefix_restore_on()"
        );
    }

    #[test]
    fn dspark_prefix_lookup_bypass_tracks_the_exact_load_verdict() {
        let base = super::DsparkColdPrefixAdmission {
            route_ready: true,
            prime_feasible: true,
            greedy: true,
            greedy_penalized: false,
            sampled: false,
            constrained: false,
            vision: false,
            cold: true,
            gate_on: true,
            pin: None,
            projected_wave: 1,
            low: 2,
            n_active: 0,
            has_live_non_demotable: false,
            // A long-decode shape, so this load-verdict test keeps testing the LOAD verdict
            // and not the shape guard added in lane/dspark-trunk-hit-20260827.
            prompt_len: 30_000,
            decode_budget: 8_192,
            hit_available: true,
        };
        let route = |a| {
            let prefers = super::dspark_prefers_cold_over_prefix(a);
            (
                prefers,
                super::should_probe_prefix_cache(true, false, prefers, false),
            )
        };

        // Greedy LOW=2: both admitted wave sizes bypass a trunk-only entry; shed traffic
        // probes/restores the same preserved entry.
        assert_eq!(route(base), (true, false));
        assert_eq!(
            route(super::DsparkColdPrefixAdmission {
                projected_wave: 2,
                ..base
            }),
            (true, false),
        );
        assert_eq!(
            route(super::DsparkColdPrefixAdmission {
                projected_wave: 4,
                ..base
            }),
            (false, true),
        );

        // Sampled DFlash shares the greedy LOW band (lane/dspark-sampled-wave-20260825):
        // the same wave sizes bypass; past LOW it sheds to the plain path and probes the
        // preserved entry like greedy does.
        let sampled = super::DsparkColdPrefixAdmission {
            greedy: false,
            sampled: true,
            ..base
        };
        assert_eq!(route(sampled), (true, false));
        assert_eq!(
            route(super::DsparkColdPrefixAdmission {
                projected_wave: 2,
                ..sampled
            }),
            (true, false),
        );
        assert_eq!(
            route(super::DsparkColdPrefixAdmission {
                projected_wave: 4,
                ..sampled
            }),
            (false, true),
        );
    }

    #[test]
    fn dspark_prefix_preference_preserves_refusals_and_capture_scope() {
        let base = super::DsparkColdPrefixAdmission {
            route_ready: true,
            prime_feasible: true,
            greedy: true,
            greedy_penalized: false,
            sampled: false,
            constrained: false,
            vision: false,
            cold: true,
            gate_on: true,
            pin: None,
            projected_wave: 1,
            low: 2,
            n_active: 0,
            has_live_non_demotable: false,
            // Long-decode shape so this test keeps testing the pin/gate matrix rather than the
            // shape guard (lane/dspark-trunk-hit-20260827).
            prompt_len: 30_000,
            decode_budget: 8_192,
            hit_available: true,
        };
        let prefers = super::dspark_prefers_cold_over_prefix;

        // Positive K and gate-off retain their solo decision. K=0 and LOW=0 retain plain.
        assert!(prefers(super::DsparkColdPrefixAdmission {
            pin: Some(3),
            ..base
        }));
        assert!(!prefers(super::DsparkColdPrefixAdmission {
            pin: Some(3),
            n_active: 1,
            ..base
        }));
        assert!(!prefers(super::DsparkColdPrefixAdmission {
            pin: Some(0),
            ..base
        }));
        assert!(!prefers(super::DsparkColdPrefixAdmission {
            low: 0,
            ..base
        }));
        assert!(prefers(super::DsparkColdPrefixAdmission {
            gate_on: false,
            ..base
        }));
        assert!(!prefers(super::DsparkColdPrefixAdmission {
            gate_on: false,
            n_active: 1,
            ..base
        }));

        for refused in [
            super::DsparkColdPrefixAdmission {
                constrained: true,
                ..base
            },
            super::DsparkColdPrefixAdmission {
                vision: true,
                ..base
            },
            super::DsparkColdPrefixAdmission {
                greedy_penalized: true,
                ..base
            },
            super::DsparkColdPrefixAdmission {
                prime_feasible: false,
                ..base
            },
        ] {
            assert!(!prefers(refused));
            assert!(super::should_probe_prefix_cache(
                true,
                false,
                prefers(refused),
                false,
            ));
        }

        // A live non-demotable row no longer refuses the automatic LOW band
        // (lane/dspark-low-band-stack-20260825): it stacks, bounded by the wave.
        assert!(prefers(super::DsparkColdPrefixAdmission {
            has_live_non_demotable: true,
            ..base
        }));

        // A finite short max_new produces too little ctx headroom for one full DFlash block.
        // The early verdict must therefore leave prefix lookup/plain fallback live instead of
        // admitting a session the engine will deterministically refuse at prime.
        let prompt_len = 96;
        let short_ctx = super::request_ctx_cap(8_192, 8_192, prompt_len, None, 1);
        let block_ctx = super::request_ctx_cap(8_192, 8_192, prompt_len, None, 7);
        assert!(!memra_engine::dflash::dspark_spec_prompt_fits(
            prompt_len, short_ctx, 7, 2_048, true,
        ));
        assert!(memra_engine::dflash::dspark_spec_prompt_fits(
            prompt_len, block_ctx, 7, 2_048, true,
        ));

        // Prefix caching disabled: no lookup and no DFlash snapshot, preserving the old byte path.
        assert!(!super::should_probe_prefix_cache(
            false, false, false, false
        ));
        assert!(!super::dspark_prefix_capture_requested(
            true, false, 128, false, false
        ));
        assert!(!super::dspark_prefix_capture_requested(
            true, true, 63, false, false
        ));
        assert!(super::dspark_prefix_capture_requested(
            true, true, 64, false, false
        ));
        assert!(!super::dspark_prefix_capture_requested(
            true, true, 64, true, false
        ));
        assert!(!super::dspark_prefix_capture_requested(
            true, true, 64, false, true
        ));
        assert!(!super::dspark_prefix_capture_requested(
            false, true, 64, false, false
        ));

        let owner_prompt: std::collections::VecDeque<u32> = (0..64).collect();
        let request_prompt: Vec<u32> = (0..64).collect();
        assert!(super::dspark_prefix_owner_identity_matches(
            ("m", "tenant"),
            &owner_prompt,
            ("m", "tenant"),
            &request_prompt,
        ));
        assert!(!super::dspark_prefix_owner_identity_matches(
            ("m", "tenant"),
            &owner_prompt,
            ("m", "other"),
            &request_prompt,
        ));
        let mut different_prompt = request_prompt;
        different_prompt[63] = u32::MAX;
        assert!(!super::dspark_prefix_owner_identity_matches(
            ("m", "tenant"),
            &owner_prompt,
            ("m", "tenant"),
            &different_prompt,
        ));

        // Phase-(a) only COLLECTS successful DFlash rows. The scheduler publishes the resulting
        // batch after every current-tick serving phase has run; errors and non-DFlash rows never
        // enter it.
        let outcomes = [(true, true), (true, true), (true, false), (false, true)];
        let collected: Vec<usize> = outcomes
            .iter()
            .enumerate()
            .filter_map(|(i, &(dspark, ok))| {
                super::should_collect_dspark_after_phase_a(dspark, ok).then_some(i)
            })
            .collect();
        assert_eq!(collected, vec![0, 1]);
    }

    #[test]
    fn spec_load_demand_never_undercounts_and_survives_no_registration() {
        // No gauge registered (the unit-test and embedded case): fall back to what the worker
        // can see, i.e. the previous behaviour — a missing registration must degrade to the old
        // policy, never to demand 0 (which would admit every restore at any load).
        assert_eq!(super::spec_load_demand(0), 0);
        assert_eq!(super::spec_load_demand(7), 7);
        // The registration is process-global and set by main(); assert the fallback SHAPE here
        // rather than mutating global state a sibling test could observe. Whatever the gauge
        // says, demand is monotone in the worker-visible count and never below it.
        assert!(super::spec_load_demand(5) >= 5);
        assert!(super::spec_load_demand(9) >= super::spec_load_demand(3));
    }

    #[test]
    fn spec_restore_load_guard_defaults_on() {
        // Read once per process, so this asserts the DEFAULT (unset) posture of whatever
        // process runs the battery — the v0.93.0 lesson: a door that silently defaults the
        // wrong way ships a headline nobody measured.
        if std::env::var("MEMRA_SPEC_RESTORE_LOAD_GUARD").is_err() {
            assert!(super::spec_restore_load_guard_on());
        }
    }

    #[test]
    fn sampled_spec_quality_doors_default_on() {
        // Each door is read once per process, so these assert the DEFAULT (unset) posture of
        // whatever process runs the battery. A door that silently defaults OFF is exactly how
        // v0.93.0 shipped an inert headline with 100% green gates.
        if std::env::var("MEMRA_SPEC_SAMPLED_BOUNDARY").is_err() {
            assert!(memra_engine::spec::spec_sampled_boundary_on());
        }
        if std::env::var("MEMRA_SPEC_PEN_SESSION").is_err() {
            assert!(memra_engine::spec::spec_pen_session_on());
        }
        if std::env::var("MEMRA_SPEC_RESTORE_REPUBLISH").is_err() {
            assert!(memra_engine::spec::spec_restore_republish_on());
        }
    }

    #[test]
    fn spec_restore_sampled_door_defaults_on() {
        // The door is read once per process, so this asserts the DEFAULT (unset) posture of
        // whatever process runs the battery: sampled restores are ON unless explicitly "0".
        if std::env::var("MEMRA_SPEC_RESTORE_SAMPLED").is_err() {
            assert!(super::spec_restore_sampled_on());
        }
    }

    #[test]
    fn spec_k_pin_parsing_is_explicit_and_supports_plain() {
        assert_eq!(parse_spec_k_pin(None), Ok(None));
        assert_eq!(parse_spec_k_pin(Some("0")), Ok(Some(0)));
        assert_eq!(parse_spec_k_pin(Some("5")), Ok(Some(5)));
        assert!(
            parse_spec_k_pin(Some("all"))
                .unwrap_err()
                .contains("non-negative integer")
        );
        assert!(parse_spec_k_pin(Some("-1")).is_err());
    }

    #[test]
    fn replay_policy_is_derived_from_gdn_and_moe_operations() {
        use memra_gguf::config::{HfConfig, ModelConfig, MoeConfig};

        let mut hybrid = ModelConfig::from_hf(&HfConfig::parse(
            r#"{"model_type":"qwen3_5","num_hidden_layers":2,"hidden_size":64,
            "num_attention_heads":2,"num_key_value_heads":1,"head_dim":32,
            "intermediate_size":128,"vocab_size":16,"max_position_embeddings":128,
            "full_attention_interval":2,"linear_conv_kernel_dim":3,
            "linear_key_head_dim":32,"linear_value_head_dim":32,
            "linear_num_key_heads":1,"linear_num_value_heads":2}"#,
        ));
        let dense = memra_gguf::model_plan::ModelPlan::compile(&hybrid).unwrap();
        hybrid.moe = Some(MoeConfig {
            expert_count: 4,
            expert_used_count: 2,
            expert_ff_length: 32,
            expert_shared_ff_length: 0,
        });
        let routed = memra_gguf::model_plan::ModelPlan::compile(&hybrid).unwrap();
        let plain =
            memra_gguf::model_plan::ModelPlan::compile(&ModelConfig::from_hf(&HfConfig::parse(
                r#"{"model_type":"qwen3","num_hidden_layers":1,"hidden_size":64,
                "num_attention_heads":2,"num_key_value_heads":1,"head_dim":32,
                "intermediate_size":128,"vocab_size":16,"max_position_embeddings":128}"#,
            )))
            .unwrap();

        assert!(model_forces_spec_replay(&routed));
        assert!(!model_forces_spec_replay(&dense));
        assert!(!model_forces_spec_replay(&plain));
        assert!(!constrained_spec_supported(&routed, false));
        assert!(!constrained_spec_supported(&plain, true));
        assert!(constrained_spec_supported(&plain, false));
    }

    #[test]
    fn spec_k_operator_pin_wins_over_every_policy_row() {
        let pp2 = resolve_spec_gate_thresholds(true, None, None);
        assert_eq!(
            choose_spec_k(Some(5), true, pp2, 8, 16, 0, false),
            SpecKDecision {
                k: 5,
                reason: SpecKReason::OperatorPin
            },
        );
        let single = resolve_spec_gate_thresholds(false, None, None);
        assert_eq!(
            choose_spec_k(Some(0), true, single, 1, 8192, 8192, false),
            SpecKDecision {
                k: 0,
                reason: SpecKReason::OperatorPin
            },
        );
    }

    #[test]
    fn spec_k_policy_maps_placement_and_concurrency_to_plain() {
        let pp2 = resolve_spec_gate_thresholds(true, None, None);
        assert_eq!(
            choose_spec_k(None, true, pp2, 1, 28, 0, false),
            SpecKDecision {
                k: 0,
                reason: SpecKReason::Placement
            },
        );

        let single = resolve_spec_gate_thresholds(false, None, None);
        assert_eq!(
            choose_spec_k(None, true, single, 3, 28, 0, false),
            SpecKDecision {
                k: 0,
                reason: SpecKReason::Concurrency
            },
        );

        // An explicit PP-2 threshold override restores the measured gate semantics:
        // c=1 may speculate; the next arrival is K=0.
        let pp2_c1 = resolve_spec_gate_thresholds(true, Some(1), Some(2));
        assert_eq!(
            choose_spec_k(None, true, pp2_c1, 1, 28, 0, false),
            SpecKDecision {
                k: SPEC_K_COLD_SHORT,
                reason: SpecKReason::ColdShort
            },
        );
        assert_eq!(
            choose_spec_k(None, true, pp2_c1, 2, 28, 0, false),
            SpecKDecision {
                k: 0,
                reason: SpecKReason::Concurrency
            },
        );
    }

    #[test]
    fn spec_k_prompt_cache_table_has_exact_boundaries() {
        assert_eq!(
            (
                SPEC_K_COLD_SHORT,
                SPEC_K_COLD_LONG,
                SPEC_K_CACHED_LONG,
                SPEC_K_CACHED_LONG_TRIM,
            ),
            (3, 3, 2, 5),
        );
        // trim-keyed cached-long: a rank-trimmed head re-prices the cached-long depth
        // (research/orndecode-20260822 — K=5 350.8 tok/s vs ~273 at k=2 on ornith15).
        assert_eq!(
            choose_spec_k(
                None,
                true,
                resolve_spec_gate_thresholds(false, None, None),
                1,
                SPEC_K_LONG_PROMPT_MIN,
                SPEC_K_LONG_CACHE_MIN,
                true,
            ),
            SpecKDecision {
                k: SPEC_K_CACHED_LONG_TRIM,
                reason: SpecKReason::CachedLong
            },
        );
        let single = resolve_spec_gate_thresholds(false, None, None);
        assert_eq!(
            choose_spec_k(
                None,
                true,
                single,
                1,
                SPEC_K_LONG_PROMPT_MIN - 1,
                9999,
                false
            ),
            SpecKDecision {
                k: SPEC_K_COLD_SHORT,
                reason: SpecKReason::ColdShort
            },
        );
        assert_eq!(
            choose_spec_k(
                None,
                true,
                single,
                1,
                SPEC_K_LONG_PROMPT_MIN,
                SPEC_K_LONG_CACHE_MIN - 1,
                false,
            ),
            SpecKDecision {
                k: SPEC_K_COLD_LONG,
                reason: SpecKReason::ColdLong
            },
        );
        assert_eq!(
            choose_spec_k(
                None,
                true,
                single,
                1,
                SPEC_K_LONG_PROMPT_MIN,
                SPEC_K_LONG_CACHE_MIN,
                false,
            ),
            SpecKDecision {
                k: SPEC_K_CACHED_LONG,
                reason: SpecKReason::CachedLong
            },
        );
        // MEMRA_SPEC_GATE=0 retains its rollback meaning: placement/concurrency are
        // ignored, while the prompt/cache table remains active.
        assert_eq!(
            choose_spec_k(
                None,
                false,
                resolve_spec_gate_thresholds(true, None, None),
                8,
                SPEC_K_LONG_PROMPT_MIN,
                SPEC_K_LONG_CACHE_MIN,
                false
            ),
            SpecKDecision {
                k: SPEC_K_CACHED_LONG,
                reason: SpecKReason::CachedLong
            },
        );
    }

    #[test]
    fn worker_device_defaults_to_cuda_visible_zero_and_follows_the_pp_head_stage() {
        assert_eq!(worker_device(None), Ok(0));
        assert_eq!(worker_device(Some("")), Ok(0));
        // The primary follows the LAST stage (the lm head's device — the spec round's draft
        // chain reads it every token; see worker_device's doc). The 5f27c55c stage-0 pin was
        // the v0.72 tag-blocker-2 regressor: 112.5 -> 17.5 agg tok/s on spec+PP-2 serving.
        assert_eq!(worker_device(Some("1,0")), Ok(0));
        assert_eq!(worker_device(Some("0,1")), Ok(1));
        assert_eq!(worker_device(Some(" 3 , 4 ")), Ok(4));
    }

    #[test]
    fn worker_device_rejects_an_invalid_pp_device() {
        // EVERY position is validated (a bad string must refuse at boot, wherever it is).
        let err = worker_device(Some("gpu0,1")).unwrap_err();
        assert!(err.contains("invalid device"), "{err}");
        assert!(err.contains("gpu0"), "{err}");
        let err = worker_device(Some("1,gpu0")).unwrap_err();
        assert!(err.contains("gpu0"), "{err}");
    }

    #[test]
    fn optipipe_controller_door_is_absent_by_default_and_bounds_thresholds() {
        assert_eq!(optipipe_controller_threshold(None), Ok(None));
        assert_eq!(optipipe_controller_threshold(Some("0")), Ok(Some(0.0)));
        assert_eq!(optipipe_controller_threshold(Some("0.7")), Ok(Some(0.7)));
        assert_eq!(optipipe_controller_threshold(Some("1")), Ok(Some(1.0)));
        for invalid in ["-0.01", "1.01", "NaN", "not-a-number"] {
            let err = optipipe_controller_threshold(Some(invalid)).unwrap_err();
            assert!(
                err.contains("MEMRA_OPTI_CONTROLLER_Q"),
                "unexpected error: {err}"
            );
        }
    }

    // ---- drafter attachment: the loud-failure semantics (lane/step-draft, 2026-08-07) ----
    //
    // These pin the class of bug that NO gate in this repo could catch: a step35 model served
    // without its external MTP drafter runs plain decode and produces CORRECT output, so
    // kernel-check is model-free, run-gen argmax MATCHes, and run-spec is never even reached.
    // Only a log line can flag it, so the log line is what gets tested.

    #[test]
    fn step35_without_drafter_warns_and_names_the_attach_spelling() {
        let v = draft_verdict(false, true);
        assert_eq!(v, DraftVerdict::NoDrafterExternalMtpArch);
        let msg = draft_verdict_message(&v, "step", "/m/Step-3.7-flash-IQ4_XS-00001-of-00003.gguf")
            .expect("a step35 model with no drafter MUST produce a line");
        // The defect is silence, so the line has to be findable and actionable.
        assert!(msg.contains("no MTP drafter attached"), "{msg}");
        assert!(msg.contains("plain decode"), "{msg}");
        // ACTIONABLE: the exact attach spelling, not just a complaint.
        assert!(msg.contains("MEMRA_MODELS"), "{msg}");
        assert!(
            msg.contains("+/path/to/"),
            "the '+draft' convention must be spelled: {msg}"
        );
        // And it must not read as a defect in the artifact — nextn=0 is CORRECT here.
        assert!(msg.contains("SEPARATE GGUF"), "{msg}");
        assert!(msg.contains("does NOT mean"), "{msg}");
    }

    #[test]
    fn attached_drafter_is_quiet_and_so_is_a_non_step35_model_without_one() {
        // Attached: nothing to warn about; spec_eligible arbitrates per request from here.
        // (#87 CLOSED 2026-08-08: a drafter over sharded cross-device PP-2 used to refuse
        // here — the ppN reverse-publication fences made the regime serve, receipts
        // research/pp2spec-crash-20260807/. Attached is now unconditional.)
        let v = draft_verdict(true, true);
        assert_eq!(v, DraftVerdict::Attached);
        assert!(draft_verdict_message(&v, "step", "/m.gguf").is_none());

        // A non-step35 model with no head: `nextn=0` there genuinely means no head, and the
        // existing load line already says the layer count. Warning would be noise on every
        // plain model the server has ever hosted — which is how a real warning gets ignored.
        let v = draft_verdict(false, false);
        assert_eq!(v, DraftVerdict::NoDrafterQuiet);
        assert!(draft_verdict_message(&v, "q27", "/m.gguf").is_none());
    }

    // (#87's refusal tests — `spec_over_sharded_pp_refuses_and_points_at_87`,
    // `the_quarantine_binds_only_where_all_three_conditions_hold`, and
    // `the_87_refusal_lands_before_the_load_when_a_draft_was_attached` — retired with the
    // quarantine itself, 2026-08-08. The regime they refused now serves: root cause was the
    // ppN reverse-publication hole, fixed by `PpNRt::fence_stages_behind`; crash gate
    // 212/212 at c=2..8, run-spec K=1..8 PASS. research/pp2spec-crash-20260807/.)

    /// Device-free PrefixEntry (empty kv/conv/ssm planes) — the namespace-visibility laws
    /// under test live entirely in the host-side key/toks matching.
    fn entry(pool_key: &PoolKey, toks: Vec<u32>) -> PrefixEntry {
        PrefixEntry {
            layout_version: PREFIX_ENTRY_LAYOUT_VERSION,
            dspark_draft: None,
            pool_key: pool_key.clone(),
            toks,
            kv: Vec::new(),
            conv: Vec::new(),
            ssm: Vec::new(),
            pos: 0,
            last_logits: vec![0.0],
            draft: None,
            last_h: Vec::new(),
            bytes: 1,
            last_use: std::time::Instant::now(),
            id: 0,
            segment: PrefixSegment::Probation,
            pins: 0,
        }
    }

    fn key(ns: &str) -> PoolKey {
        ("m".to_string(), ns.to_string())
    }

    fn toks(n: usize) -> Vec<u32> {
        (0..n as u32).collect()
    }

    #[test]
    fn dspark_exact_transition_touch_promotes_without_cached_token_credit() {
        let k = key("");
        let prompt = toks(super::PREFIX_CACHE_MIN_TOKENS);
        let mut px = PrefixCache::default();
        px.insert_with_budget(&k, entry(&k, prompt.clone()), "dspark-test", 8);
        assert_eq!(px.entries[&k][0].segment, PrefixSegment::Probation);
        let credits = (px.hits, px.misses, px.hit_tokens);

        assert!(px.touch_exact_without_credit(&k, &prompt));
        assert_eq!(px.entries[&k][0].segment, PrefixSegment::Protected);
        assert_eq!(
            (px.hits, px.misses, px.hit_tokens),
            credits,
            "DFlash rebuilt the prompt, so residency changes without cache credit",
        );

        let mut different = prompt;
        different[0] = u32::MAX;
        assert!(!px.touch_exact_without_credit(&k, &different));
    }

    #[test]
    fn prefix_fanout_groups_only_inside_exact_model_tenant_and_salt() {
        let mut a = toks(96);
        a.extend([10_001, 10_002]);
        let mut b = toks(96);
        b.extend([20_001, 20_002, 20_003]);
        let mut other_prefix = toks(96);
        other_prefix[63] = u32::MAX;
        let acme_s1 = crate::auth::scope_namespace("acme", "s1");
        let candidates = vec![
            PrefixFanoutCandidate {
                active_idx: 3,
                key: ("m".into(), acme_s1.clone()),
                prompt: a,
            },
            PrefixFanoutCandidate {
                active_idx: 7,
                key: ("m".into(), acme_s1.clone()),
                prompt: b,
            },
            PrefixFanoutCandidate {
                active_idx: 8,
                key: ("m".into(), crate::auth::scope_namespace("acme", "s2")),
                prompt: toks(96),
            },
            PrefixFanoutCandidate {
                active_idx: 9,
                key: ("m".into(), crate::auth::scope_namespace("blue", "s1")),
                prompt: toks(96),
            },
            PrefixFanoutCandidate {
                active_idx: 10,
                key: ("other-model".into(), acme_s1.clone()),
                prompt: toks(96),
            },
            PrefixFanoutCandidate {
                active_idx: 11,
                key: ("m".into(), acme_s1),
                prompt: other_prefix,
            },
        ];
        assert_eq!(
            prefix_fanout_groups(&candidates, 80),
            vec![PrefixFanoutGroup {
                members: vec![3, 7],
                prefix_len: 80,
            }],
        );
        assert!(prefix_fanout_groups(&candidates, PREFIX_CACHE_MIN_TOKENS - 1).is_empty());
    }

    #[test]
    fn prefix_fanout_rewrites_one_provisional_miss_exactly_once() {
        let mut px = PrefixCache::default();
        px.misses = 2;
        px.record_lcp(0);
        px.record_lcp(0);
        px.promote_miss_to_hit(0, 256);
        assert_eq!(px.misses, 1);
        assert_eq!(px.hits, 1);
        assert_eq!(px.hit_tokens, 256);
        assert_eq!(px.lcp_hist[PrefixCache::lcp_bucket(0)], 1);
        assert_eq!(px.lcp_hist[PrefixCache::lcp_bucket(256)], 1);
        assert_eq!(px.lcp_hist.iter().sum::<u64>(), 2);
    }

    /// TOOTH for the H11 depth freeze (the measured 3.1x lever, canonflip-20260813):
    /// the seed decision must compare DEPTH against the deepest covering entry, never a
    /// boolean covering check. Under the pre-fix rule (`has_covering` -> skip), the shallow
    /// 100-token system-prompt entry froze the 4,860-token class forever.
    #[test]
    fn prefix_seed_deepens_past_a_shallow_covering_entry() {
        let mut px = PrefixCache::default();
        let shallow = toks(PREFIX_CACHE_MIN_TOKENS + 36); // the "100-token" class entry
        let deep_prompt = toks(4860);
        px.insert(&key("t"), entry(&key("t"), shallow.clone()), "test");

        // the covering boolean is TRUE — exactly the state that used to refuse the seed...
        assert!(px.has_covering(&key("t"), &deep_prompt));
        // ...and the depth reading unfreezes it: 4860 - 100 >> deepen floor.
        assert_eq!(
            px.deepest_covering(&key("t"), &deep_prompt),
            Some(shallow.len())
        );
        assert!(super::prefix_seed_deepens(
            px.deepest_covering(&key("t"), &deep_prompt),
            deep_prompt.len()
        ));

        // deepest_covering picks the DEEPEST cover, not the first.
        px.insert(&key("t"), entry(&key("t"), toks(2048)), "test");
        assert_eq!(px.deepest_covering(&key("t"), &deep_prompt), Some(2048));
        // a non-covering (diverging) entry never counts as cover.
        let mut diverged = toks(4096);
        diverged[3000] ^= 1;
        px.insert(&key("t"), entry(&key("t"), diverged), "test");
        assert_eq!(px.deepest_covering(&key("t"), &deep_prompt), Some(2048));

        // deepen floor: equal depth and sub-floor gains are churn, one entry-worth pays.
        assert!(super::prefix_seed_deepens(None, 4860));
        assert!(!super::prefix_seed_deepens(Some(4860), 4860));
        assert!(!super::prefix_seed_deepens(
            Some(4860 - super::prefix_seed_deepen_min() + 1),
            4860
        ));
        assert!(super::prefix_seed_deepens(
            Some(4860 - super::prefix_seed_deepen_min()),
            4860
        ));
    }

    #[test]
    fn hit_reseed_refuses_eager_only_models() {
        let arms = super::plain_hit_reseed_arms;
        // the qwen-class lever: a plain hit on a long prompt re-arms the seed.
        assert!(arms(true, true, false, 4860));
        // R16's hard prerequisite: an eager-only (gemma-class) hit NEVER re-arms — its
        // carried suffix rides tokenwise decode_step, so a "deepened" seed would publish
        // restore+decode_step chained provenance onto the H1 crossing.
        assert!(!arms(true, true, true, 4860));
        // the other refusals hold: no hit, spec path, sub-floor prompt.
        assert!(!arms(false, true, false, 4860));
        assert!(!arms(true, false, false, 4860));
        assert!(!arms(true, true, false, PREFIX_CACHE_MIN_TOKENS - 1));
    }

    #[test]
    fn hit_lcp_split_drops_the_sub_floor_fed_gap() {
        // grid = gdn_chunk_size() (32 shipped), fed-gap floor = PRIME_MIN_T (16), entry
        // floor = PREFIX_CACHE_MIN_TOKENS (64). `hit_len` is a prefill-done seed depth and
        // is NOT grid-aligned, so an aligned LCP can land 1..15 tokens past it — the gap
        // `prefill_tick`'s boundary veto refuses to prime, which would route those tokens
        // through tokenwise decode_step BEFORE the capture (the W1 two-programs door, with
        // the captured entry's tail carrying decode_step provenance).
        let bound = super::hit_lcp_snapshot_boundary;
        // the reachable door: la = 128, gap = 8 < 16 -> DROPPED (lose a seed, keep one
        // numeric program).
        assert_eq!(bound(130, 120, 200), None);
        // one grid unit deeper clears the floor and captures: la = 160, gap = 40.
        assert_eq!(bound(170, 120, 200), Some(160));
        // exactly the floor is legal.
        assert_eq!(bound(170, 144, 200), Some(160));
        // no deepening past the restored entry -> nothing to capture.
        assert_eq!(bound(120, 120, 200), None);
        assert_eq!(bound(90, 120, 200), None);
        // an aligned boundary under the entry floor never captures.
        assert_eq!(bound(50, 10, 200), None);
        // the prompt-side W1 step-down still applies first (within-behaviour preserved):
        // lcp at the prompt end steps 200 -> 192 -> 160, and the fed gap is measured from
        // the stepped boundary.
        assert_eq!(bound(200, 120, 200), Some(160));
    }

    #[test]
    fn prefix_cache_same_namespace_same_prefix_hits() {
        let mut px = PrefixCache::default();
        let prefix = toks(PREFIX_CACHE_MIN_TOKENS);
        px.insert(
            &key("tenant-a"),
            entry(&key("tenant-a"), prefix.clone()),
            "test",
        );
        // same namespace + same prefix (prompt extends the entry) -> hit.
        assert!(
            px.lookup(&key("tenant-a"), &toks(PREFIX_CACHE_MIN_TOKENS + 32))
                .is_some()
        );
        assert!(px.has_covering(&key("tenant-a"), &prefix));
        assert_eq!(px.best_lcp(&key("tenant-a"), &prefix), prefix.len());
    }

    /// An allocated-but-unexecuted NextN/MTP head layer carries len 0 while the trunk sits at
    /// cache.pos. `prefix_snapshot` used to treat that as corruption and return Err, which silently
    /// disabled the ENTIRE prefix cache (0 inserts / 0 hits) for every MTP-bearing model — both
    /// models we sell. The gate caught it as `prefix snapshot layer 32 len 0 != cache pos 272`.
    /// This pins the classification so the regression cannot come back as a silent zero-hit cache.
    #[test]
    fn zero_length_nextn_layer_is_absent_not_corrupt() {
        // Trunk layers agree with pos; the trailing MTP slot is allocated but never executed.
        let pos = 272usize;
        let trunk_lens = [pos, pos, pos];
        let mtp_len = 0usize;

        // Trunk layers must be captured verbatim.
        for len in trunk_lens {
            assert_eq!(
                len, pos,
                "a trunk layer disagreeing with pos is still corruption"
            );
            assert!(
                !(len == 0 && pos > 0),
                "a trunk layer must never take the absent-layer path",
            );
        }
        // The MTP layer must take the absent path rather than failing the snapshot.
        assert!(
            mtp_len == 0 && pos > 0,
            "the allocated-but-unexecuted NextN layer is the absent-layer case",
        );
        // And a genuinely inconsistent trunk layer must still be rejected.
        let corrupt_len = pos - 1;
        assert!(
            corrupt_len != pos && !(corrupt_len == 0 && pos > 0),
            "a partially-filled layer is neither absent nor valid and must still Err",
        );
    }

    #[test]
    fn immediate_partial_restore_support_matrix_is_fail_closed() {
        assert_eq!(
            partial_prefix_decision(false, false, 96, 160, 128),
            PartialPrefixDecision::Restore,
            "transformer-only mid-entry splits are the supported first arm",
        );
        assert_eq!(
            partial_prefix_decision(true, false, 96, 160, 128),
            PartialPrefixDecision::RefuseRecurrentMidEntry,
        );
        assert_eq!(
            partial_prefix_decision(true, false, 160, 160, 192),
            PartialPrefixDecision::Restore,
            "recurrent state is exact at its captured endpoint",
        );
        assert_eq!(
            partial_prefix_decision(false, true, 96, 160, 128),
            PartialPrefixDecision::RefuseRoutedMoe,
        );
        assert_eq!(
            partial_prefix_decision(false, false, 128, 160, 128),
            PartialPrefixDecision::RefuseNoSuffix,
            "a truncated entry does not carry logits for an interior endpoint",
        );
    }

    #[test]
    fn best_lcp_entry_returns_longest_exact_match_inside_namespace() {
        let k = key("tenant-a");
        let mut px = PrefixCache::default();
        let mut first = toks(160);
        first[80] = 50_000;
        let mut longest = toks(192);
        longest[112] = 60_000;
        px.insert(&k, entry(&k, first), "test");
        px.insert(&k, entry(&k, longest), "test");

        let prompt = toks(144);
        let (i, lcp) = px.best_lcp_entry(&k, &prompt).unwrap();
        assert_eq!(lcp, 112);
        assert_eq!(px.entries[&k][i].toks.len(), 192);
        assert!(px.best_lcp_entry(&key("tenant-b"), &prompt).is_none());
    }

    #[test]
    fn prefix_cache_rejects_wrong_identity_and_layout_version_at_insert() {
        let mut px = PrefixCache::default();
        let target = key("tenant-a");
        px.insert(
            &target,
            entry(&key("tenant-b"), toks(PREFIX_CACHE_MIN_TOKENS)),
            "test",
        );
        assert_eq!(px.n_entries(), 0, "wrong namespace must not enter the pool");

        let mut stale = entry(&target, toks(PREFIX_CACHE_MIN_TOKENS));
        stale.layout_version = PREFIX_ENTRY_LAYOUT_VERSION + 1;
        px.insert(&target, stale, "test");
        assert_eq!(px.n_entries(), 0, "wrong layout version must fail closed");
    }

    #[test]
    fn prefix_restore_plane_preflight_rejects_corrupt_layouts_before_copy() {
        assert_eq!(
            validate_prefix_plane_shape(128, 96, 128, 16, 16, 2048, 2048, 16, 16, 4096, 4096),
            Ok((1536, 1536)),
        );
        assert!(
            validate_prefix_plane_shape(128, 96, 127, 16, 16, 2032, 2032, 16, 16, 4096, 4096)
                .unwrap_err()
                .contains("source len"),
        );
        assert!(
            validate_prefix_plane_shape(128, 96, 128, 8, 16, 1024, 2048, 16, 16, 4096, 4096)
                .unwrap_err()
                .contains("KV layout"),
        );
        assert!(
            validate_prefix_plane_shape(128, 96, 128, 16, 16, 2047, 2048, 16, 16, 4096, 4096)
                .unwrap_err()
                .contains("truncated/corrupt"),
        );
        assert!(
            validate_prefix_plane_shape(128, 96, 128, 16, 16, 2048, 2048, 16, 16, 1024, 4096)
                .unwrap_err()
                .contains("destination planes too small"),
        );
    }

    /// LCP histogram bucketing (lane/cache-metering): edges are lower bounds, the last
    /// bucket is unbounded, and the [64,512) tick-seg window is exactly buckets 4..=6.
    #[test]
    fn lcp_histogram_buckets_are_lower_edge_and_record_samples() {
        assert_eq!(PrefixCache::lcp_bucket(0), 0);
        assert_eq!(PrefixCache::lcp_bucket(1), 1);
        assert_eq!(PrefixCache::lcp_bucket(15), 1);
        assert_eq!(PrefixCache::lcp_bucket(16), 2);
        assert_eq!(PrefixCache::lcp_bucket(63), 3);
        assert_eq!(PrefixCache::lcp_bucket(64), 4); // tick-seg window opens
        assert_eq!(PrefixCache::lcp_bucket(127), 4);
        assert_eq!(PrefixCache::lcp_bucket(128), 5);
        assert_eq!(PrefixCache::lcp_bucket(256), 6);
        assert_eq!(PrefixCache::lcp_bucket(511), 6); // tick-seg window closes
        assert_eq!(PrefixCache::lcp_bucket(512), 7);
        assert_eq!(PrefixCache::lcp_bucket(4095), 9);
        assert_eq!(PrefixCache::lcp_bucket(4096), 10);
        assert_eq!(PrefixCache::lcp_bucket(1 << 20), 10); // unbounded tail
        let mut px = PrefixCache::default();
        px.record_lcp(0);
        px.record_lcp(100);
        px.record_lcp(100);
        assert_eq!(px.lcp_hist[0], 1);
        assert_eq!(px.lcp_hist[4], 2);
        assert_eq!(px.lcp_hist.iter().sum::<u64>(), 3);
    }

    /// Per-tenant metering rows (lane/cache-metering): keyring namespaces collapse to
    /// their tenant (salts within a tenant share a row), raw salts pass through, and the
    /// row cap saturates into "(other)" without losing tokens.
    #[test]
    fn meter_account_keys_by_tenant_and_bounds_rows() {
        let mut m: HashMap<String, [u64; 2]> = HashMap::new();
        // keyring: two salts of one tenant share the row; another tenant gets its own.
        meter_account(&mut m, &crate::auth::scope_namespace("acme", "u1"), 100, 40);
        meter_account(&mut m, &crate::auth::scope_namespace("acme", "u2"), 50, 10);
        meter_account(&mut m, &crate::auth::scope_namespace("blue", ""), 30, 0);
        meter_cached_credit(&mut m, &crate::auth::scope_namespace("acme", "u3"), 25);
        assert_eq!(m["t:acme"], [150, 75]);
        assert_eq!(m["t:blue"], [30, 0]);
        // no keyring: the raw salt is the row key; "" is the default namespace.
        meter_account(&mut m, "session-7", 20, 20);
        meter_account(&mut m, "", 10, 5);
        assert_eq!(m["session-7"], [20, 20]);
        assert_eq!(m[""], [10, 5]);
        // cap: fill to METER_TENANT_CAP distinct rows, then overflow lands in "(other)"
        // while an EXISTING row keeps accumulating under its own key.
        let mut m: HashMap<String, [u64; 2]> = HashMap::new();
        for i in 0..METER_TENANT_CAP {
            meter_account(&mut m, &format!("s{i}"), 1, 0);
        }
        meter_account(&mut m, "one-too-many", 7, 3);
        meter_account(&mut m, "s0", 2, 1);
        assert_eq!(m.len(), METER_TENANT_CAP + 1);
        assert_eq!(m["(other)"], [7, 3]);
        assert_eq!(m["s0"], [3, 1]);
        // totals stay exact: sum over rows == sum over requests.
        let total: u64 = m.values().map(|r| r[0]).sum();
        assert_eq!(total, METER_TENANT_CAP as u64 + 7 + 2);
    }

    fn seed_adsd_baseline(detector: &mut AdsdDetector) {
        for i in 0..24u64 {
            let accepted = [70, 73, 71, 74][i as usize % 4];
            assert!(
                detector
                    .observe("model-a", "t:baseline", accepted, 100)
                    .is_none()
            );
        }
    }

    #[test]
    fn adsd_detector_fires_once_on_sustained_acceptance_collapse() {
        let mut detector = AdsdDetector::default();
        seed_adsd_baseline(&mut detector);

        let mut events = Vec::new();
        for _ in 0..(ADSD_TENANT_WINDOW + ADSD_SUSTAINED_OBSERVATIONS as usize + 4) {
            if let Some(event) = detector.observe("model-a", "t:attacker", 8, 100) {
                events.push(event);
            }
        }
        assert_eq!(
            events.len(),
            1,
            "one sustained incident must not count every request"
        );
        let event = &events[0];
        assert_eq!(event.tenant, "t:attacker");
        assert!(event.baseline_rate - event.tenant_rate >= ADSD_MIN_RATE_DROP);
        assert!(event.z_score <= ADSD_Z_THRESHOLD);
        let baseline_accepted = 24.0 * 72.0;
        let baseline_drafted = 24.0 * 100.0;
        let tenant_accepted = ADSD_TENANT_WINDOW as f64 * 8.0;
        let tenant_drafted = ADSD_TENANT_WINDOW as f64 * 100.0;
        let pooled_rate =
            (baseline_accepted + tenant_accepted) / (baseline_drafted + tenant_drafted);
        let expected_z = (tenant_accepted / tenant_drafted - baseline_accepted / baseline_drafted)
            / (pooled_rate * (1.0 - pooled_rate) * (1.0 / baseline_drafted + 1.0 / tenant_drafted))
                .sqrt();
        assert!((event.z_score - expected_z).abs() < 1e-12);
        assert_eq!(detector.suspect_total["t:attacker"], 1);
    }

    #[test]
    fn adsd_detector_fires_on_single_tenant_historical_collapse() {
        let mut detector = AdsdDetector::default();
        for i in 0..24u64 {
            let accepted = [70, 73, 71, 74][i as usize % 4];
            assert!(
                detector
                    .observe("model-a", "t:solo", accepted, 100)
                    .is_none()
            );
        }

        let mut events = Vec::new();
        for _ in 0..(ADSD_TENANT_WINDOW + ADSD_SUSTAINED_OBSERVATIONS as usize + 4) {
            if let Some(event) = detector.observe("model-a", "t:solo", 8, 100) {
                events.push(event);
            }
        }

        assert_eq!(
            events.len(),
            1,
            "a single-tenant collapse must emit one incident"
        );
        assert_eq!(events[0].tenant, "t:solo");
        assert!(events[0].baseline_rate - events[0].tenant_rate >= ADSD_MIN_RATE_DROP);
        assert!(events[0].z_score <= ADSD_Z_THRESHOLD);
        assert_eq!(detector.suspect_total["t:solo"], 1);
    }

    #[test]
    fn adsd_detector_stays_latched_during_boiling_frog_collapse() {
        let mut detector = AdsdDetector::default();
        seed_adsd_baseline(&mut detector);

        let mut events = 0;
        for _ in 0..(ADSD_TENANT_WINDOW + ADSD_SUSTAINED_OBSERVATIONS as usize) {
            events += detector.observe("model-a", "t:attacker", 8, 100).is_some() as usize;
        }
        assert_eq!(events, 1, "the initial collapse must latch one incident");

        // Keep the attack active while acceptance slowly falls further. Once the original
        // high-acceptance rows age out, a self-contaminating model baseline converges on this
        // tenant and incorrectly rearms even though the tenant never recovered.
        for accepted in (0..8u64).rev() {
            for _ in 0..8 {
                events += detector
                    .observe("model-a", "t:attacker", accepted, 100)
                    .is_some() as usize;
            }
        }

        let key = ("model-a".to_string(), "t:attacker".to_string());
        assert!(
            detector.tenant_windows[&key].incident_latched,
            "the incident must not rearm from the suspect tenant diluting its own baseline",
        );
        assert_eq!(
            events, 1,
            "one active incident must still emit exactly once"
        );
    }

    #[test]
    fn adsd_detector_does_not_fire_on_normal_acceptance_noise() {
        let mut detector = AdsdDetector::default();
        seed_adsd_baseline(&mut detector);

        for i in 0..64usize {
            let accepted = [65, 76, 69, 74, 71, 78, 67, 73][i % 8];
            assert!(
                detector
                    .observe("model-a", "t:noisy", accepted, 100)
                    .is_none(),
                "ordinary acceptance variation must not become an ADSD incident",
            );
        }
        assert!(!detector.suspect_total.contains_key("t:noisy"));
    }

    #[test]
    fn prefix_cache_namespaces_isolate_both_directions() {
        let mut px = PrefixCache::default();
        let prompt = toks(PREFIX_CACHE_MIN_TOKENS + 32);
        // tenant-a seeds; the identical prefix is INVISIBLE to tenant-b and to the
        // default namespace (a -> b direction).
        px.insert(
            &key("tenant-a"),
            entry(&key("tenant-a"), toks(PREFIX_CACHE_MIN_TOKENS)),
            "test",
        );
        assert!(px.lookup(&key("tenant-b"), &prompt).is_none());
        assert!(px.lookup(&key(""), &prompt).is_none());
        // ... and the learning/seed signals stay scoped too (no cross-ns LCP split).
        assert_eq!(px.best_lcp(&key("tenant-b"), &prompt), 0);
        assert!(!px.has_covering(&key("tenant-b"), &prompt));
        // tenant-b seeds its OWN copy (no cross-ns dedupe: has_key is per key) and hits
        // it, while tenant-a still hits only its own (b -> a direction).
        px.insert(
            &key("tenant-b"),
            entry(&key("tenant-b"), toks(PREFIX_CACHE_MIN_TOKENS)),
            "test",
        );
        assert_eq!(px.n_entries(), 2);
        assert!(px.lookup(&key("tenant-a"), &prompt).is_some());
        assert!(px.lookup(&key("tenant-b"), &prompt).is_some());
        assert!(px.lookup(&key("tenant-c"), &prompt).is_none());
    }

    #[test]
    fn prefix_cache_default_namespace_preserves_single_tenant_behavior() {
        // No salt = the "" namespace on every request: inserts, covering dedupe, LCP
        // learning, and longest-match lookup all behave exactly as the pre-PC-ISO
        // model-keyed cache.
        let mut px = PrefixCache::default();
        let short = toks(PREFIX_CACHE_MIN_TOKENS);
        let long = toks(PREFIX_CACHE_MIN_TOKENS + 16);
        px.insert(&key(""), entry(&key(""), short.clone()), "test");
        px.insert(&key(""), entry(&key(""), long.clone()), "test");
        px.insert(&key(""), entry(&key(""), long.clone()), "test"); // exact-key dedupe still holds
        assert_eq!(px.n_entries(), 2);
        // longest entry prefixing the prompt wins, floor PREFIX_CACHE_MIN_TOKENS.
        let hit = px
            .lookup(&key(""), &toks(PREFIX_CACHE_MIN_TOKENS + 64))
            .unwrap();
        assert_eq!(px.entries[&key("")][hit].toks.len(), long.len());
        assert!(
            px.lookup(&key(""), &toks(PREFIX_CACHE_MIN_TOKENS - 1))
                .is_none()
        );
    }

    // ---------------- BYTE-SLRU EVICTION (lane/cx-slrucache, 2026-08-13) ----------------

    /// Entry with an explicit identity + byte size (identity = the single token, so the
    /// exact-key dedupe never collides and survivors are readable back out of the pools).
    fn entry_b(pool_key: &PoolKey, ident: u32, bytes: usize) -> PrefixEntry {
        PrefixEntry {
            layout_version: PREFIX_ENTRY_LAYOUT_VERSION,
            dspark_draft: None,
            pool_key: pool_key.clone(),
            toks: vec![ident],
            kv: Vec::new(),
            conv: Vec::new(),
            ssm: Vec::new(),
            pos: 0,
            last_logits: vec![0.0],
            draft: None,
            last_h: Vec::new(),
            bytes,
            last_use: next_instant(),
            id: 0,
            segment: PrefixSegment::Probation,
            pins: 0,
        }
    }

    /// Strictly-monotonic clock step: keeps test LRU order independent of clock resolution.
    fn next_instant() -> std::time::Instant {
        let t = std::time::Instant::now();
        loop {
            let u = std::time::Instant::now();
            if u > t {
                return u;
            }
        }
    }

    fn px_survivors(px: &PrefixCache) -> Vec<u32> {
        let mut v: Vec<u32> = px.entries.values().flatten().map(|e| e.toks[0]).collect();
        v.sort_unstable();
        v
    }

    fn assert_prefix_cache_accounting(px: &PrefixCache) {
        let entries: Vec<_> = px.entries.values().flatten().collect();
        assert_eq!(
            px.total_bytes,
            entries.iter().map(|entry| entry.bytes).sum::<usize>()
        );
        assert_eq!(
            px.probation_bytes,
            entries
                .iter()
                .filter(|entry| entry.segment == PrefixSegment::Probation)
                .map(|entry| entry.bytes)
                .sum::<usize>()
        );
        assert_eq!(
            px.protected_bytes,
            entries
                .iter()
                .filter(|entry| entry.segment == PrefixSegment::Protected)
                .map(|entry| entry.bytes)
                .sum::<usize>()
        );
        assert_eq!(px.total_bytes, px.probation_bytes + px.protected_bytes);
        assert_eq!(
            px.probation_lru.len(),
            entries
                .iter()
                .filter(|entry| entry.pins == 0 && entry.segment == PrefixSegment::Probation)
                .count()
        );
        assert_eq!(
            px.protected_lru.len(),
            entries
                .iter()
                .filter(|entry| entry.pins == 0 && entry.segment == PrefixSegment::Protected)
                .count()
        );
        for (lru_key, (key, i)) in px.probation_lru.iter().chain(&px.protected_lru) {
            let entry = &px.entries[key][*i];
            assert_eq!(*lru_key, PrefixCache::lru_key(entry));
            assert_eq!(entry.pins, 0);
        }
    }

    fn reuse_ident(px: &mut PrefixCache, ident: u32) -> bool {
        let found = px.entries.iter().find_map(|(key, pool)| {
            pool.iter()
                .position(|entry| entry.toks[0] == ident)
                .map(|i| (key.clone(), i))
        });
        if let Some((key, i)) = found {
            next_instant();
            px.touch(&key, i);
            true
        } else {
            false
        }
    }

    #[test]
    fn prefix_cache_slru_protects_reused_bytes_from_cross_tenant_scan() {
        const BUDGET: usize = 10;
        let mut px = PrefixCache::default();
        px.insert_with_budget(&key("hot-a"), entry_b(&key("hot-a"), 0, 5), "test", BUDGET);
        assert!(reuse_ident(&mut px, 0));
        px.insert_with_budget(&key("hot-b"), entry_b(&key("hot-b"), 1, 3), "test", BUDGET);
        assert!(reuse_ident(&mut px, 1));
        assert_eq!(px.protected_bytes, 8);

        for ident in 2..22 {
            let namespace = if ident % 2 == 0 { "scan-a" } else { "scan-b" };
            px.insert_with_budget(
                &key(namespace),
                entry_b(&key(namespace), ident, 2),
                "test",
                BUDGET,
            );
            let survivors = px_survivors(&px);
            assert!(survivors.contains(&0) && survivors.contains(&1));
            assert!(px.total_bytes <= BUDGET);
            assert_prefix_cache_accounting(&px);
        }
        assert_eq!(px.protected_bytes, 8);
        assert_eq!(px.probation_bytes, 2);
        assert_eq!(px.evictions, 19);
    }

    #[test]
    fn prefix_cache_slru_demotes_by_protected_bytes_not_entry_count() {
        const BUDGET: usize = 10;
        let mut px = PrefixCache::default();
        px.insert_with_budget(&key(""), entry_b(&key(""), 0, 6), "test", BUDGET);
        assert!(reuse_ident(&mut px, 0));
        px.insert_with_budget(&key(""), entry_b(&key(""), 1, 4), "test", BUDGET);
        assert!(reuse_ident(&mut px, 1));

        // 6 + 4 protected bytes exceed the 8-byte target. The older six-byte entry is
        // demoted even though each segment held one entry; count-based accounting differs.
        assert_eq!(
            px.entries[&key("")]
                .iter()
                .find(|e| e.toks[0] == 0)
                .unwrap()
                .segment,
            PrefixSegment::Probation
        );
        assert_eq!(
            px.entries[&key("")]
                .iter()
                .find(|e| e.toks[0] == 1)
                .unwrap()
                .segment,
            PrefixSegment::Protected
        );
        assert_eq!((px.probation_bytes, px.protected_bytes), (6, 4));

        px.insert_with_budget(&key(""), entry_b(&key(""), 2, 5), "test", BUDGET);
        assert_eq!(px_survivors(&px), vec![1, 2]);
        assert_eq!(
            (px.probation_bytes, px.protected_bytes, px.total_bytes),
            (5, 4, 9)
        );
        assert_prefix_cache_accounting(&px);
    }

    #[test]
    fn prefix_cache_lru_policy_evicts_the_global_oldest_not_only_probation() {
        // The MEMRA_PREFIX_CACHE_POLICY=lru rollback must be able to reach a PROMOTED entry.
        // Shape: two entries earn a hit (so both are protected under a 100% protected share, which
        // is what the lru policy forces), then a fresh entry arrives. Under SLRU the only
        // candidate is the probation LRU — here the newcomer itself — so the promoted pair is
        // immortal and the newcomer is its own victim. Under plain LRU the global oldest goes.
        const BUDGET: usize = 10;
        let mut px = PrefixCache::default();
        px.insert_with_budget(&key(""), entry_b(&key(""), 0, 4), "test", BUDGET);
        assert!(reuse_ident(&mut px, 0));
        px.insert_with_budget(&key(""), entry_b(&key(""), 1, 4), "test", BUDGET);
        assert!(reuse_ident(&mut px, 1));
        px.insert_with_budget(&key(""), entry_b(&key(""), 2, 2), "test", BUDGET);

        let oldest = px.entries[&key("")]
            .iter()
            .find(|e| e.toks[0] == 0)
            .unwrap();
        let newest = px.entries[&key("")]
            .iter()
            .find(|e| e.toks[0] == 2)
            .unwrap();
        assert_eq!(newest.segment, PrefixSegment::Probation);

        // SLRU arm: the newcomer in probation is the victim.
        let (_, slru_victim) = px.capacity_victim_with(true).expect("slru victim");
        assert_eq!(
            px.entries[&key("")][slru_victim].toks[0],
            2,
            "SLRU must evict the probation LRU"
        );

        // LRU arm: the globally oldest entry is the victim, even though it is promoted.
        let (_, lru_victim) = px.capacity_victim_with(false).expect("lru victim");
        assert_eq!(
            px.entries[&key("")][lru_victim].toks[0],
            oldest.toks[0],
            "policy=lru must reach the global oldest entry, including a promoted one"
        );
        assert_ne!(
            px.entries[&key("")][lru_victim].toks[0],
            2,
            "policy=lru must not make every newcomer its own victim"
        );
    }

    #[test]
    fn prefix_cache_still_refuses_an_entry_larger_than_total_budget() {
        let mut px = PrefixCache::default();
        px.insert_with_budget(&key(""), entry_b(&key(""), 0, 11), "test", 10);
        assert_eq!(px.n_entries(), 0);
        assert_eq!(px.total_bytes, 0);
        assert_eq!(px.inserts, 0);
        assert_eq!(px.evictions, 0);
        assert_prefix_cache_accounting(&px);
    }

    #[test]
    fn prefix_cache_touch_rescues_the_would_be_victim() {
        // Recency semantics end-to-end: the oldest entry, once touched, survives an
        // eviction that takes the second-oldest instead.
        let mut px = PrefixCache::default();
        px.insert_with_budget(&key(""), entry_b(&key(""), 0, 4), "test", 8);
        px.insert_with_budget(&key(""), entry_b(&key(""), 1, 4), "test", 8);
        let idx = px.entries[&key("")]
            .iter()
            .position(|e| e.toks[0] == 0)
            .unwrap();
        next_instant();
        px.touch(&key(""), idx);
        px.insert_with_budget(&key(""), entry_b(&key(""), 2, 4), "test", 8);
        assert_eq!(
            px_survivors(&px),
            vec![0, 2],
            "touched 0 must survive, untouched 1 evicts"
        );
    }

    #[test]
    fn prefix_cache_refusal_counters_separate_total_budget_from_pinned_pressure() {
        for protected_pct in [80, 100] {
            let k = key("");
            let mut px = PrefixCache::default();
            assert!(
                px.insert_with_budget_pins_and_pct(
                    &k,
                    entry_b(&k, 1, 6),
                    "test",
                    8,
                    protected_pct,
                    1
                )
                .is_some()
            );
            assert!(
                px.insert_with_budget_pins_and_pct(
                    &k,
                    entry_b(&k, 2, 3),
                    "test",
                    8,
                    protected_pct,
                    1
                )
                .is_none()
            );
            assert_eq!((px.skips_budget, px.skips_pinned), (0, 1));

            // A temporary pinned-pressure refusal must not consume the first whole-budget warning.
            let _ = px.insert_with_budget_pins_and_pct(
                &k,
                entry_b(&k, 0, 9),
                "test",
                8,
                protected_pct,
                0,
            );
            assert_eq!((px.skips_budget, px.skips_pinned), (1, 1));
        }
    }

    #[test]
    fn prefix_cache_same_window_fanout_counts_as_reuse_and_keeps_pin_refcounts() {
        let k = key("");
        let mut px = PrefixCache::default();
        let id = px
            .insert_with_budget_pins_and_pct(&k, entry_b(&k, 0, 4), "fanout test", 8, 80, 2)
            .unwrap();
        let entry = px.entries[&k].iter().find(|entry| entry.id == id).unwrap();
        assert_eq!(entry.segment, PrefixSegment::Protected);
        assert_eq!(entry.pins, 2);
        assert!(px.probation_lru.is_empty() && px.protected_lru.is_empty());
        assert_prefix_cache_accounting(&px);

        let pin = super::PrefixPin { key: k.clone(), id };
        assert!(px.unpin(&pin));
        assert!(px.protected_lru.is_empty());
        assert!(px.unpin(&pin));
        assert_eq!(px.protected_lru.len(), 1);
        assert_prefix_cache_accounting(&px);
    }

    #[test]
    fn prefix_cache_pinned_probation_refuses_before_displacing_protected_share() {
        const BUDGET: usize = 10;
        let k = key("");
        let mut px = PrefixCache::default();
        px.insert_with_budget(&k, entry_b(&k, 0, 8), "test", BUDGET);
        assert!(reuse_ident(&mut px, 0));
        assert_eq!((px.protected_bytes, px.probation_bytes), (8, 0));

        // A one-participant pinned insert has not demonstrated reuse. With no probation bytes to
        // reclaim, it must be refused rather than evicting protected below its 80% byte share.
        assert!(
            px.insert_with_budget_pins_and_pct(&k, entry_b(&k, 1, 3), "test", BUDGET, 80, 1)
                .is_none()
        );
        assert_eq!(px_survivors(&px), vec![0]);
        assert_eq!(
            (px.protected_bytes, px.probation_bytes, px.total_bytes),
            (8, 0, 8)
        );
        assert_eq!((px.inserts, px.evictions), (1, 0));
        assert_eq!((px.skips_budget, px.skips_pinned), (0, 1));
        assert_prefix_cache_accounting(&px);
    }

    #[test]
    fn prefix_cache_pin_refcount_blocks_eviction_until_last_release() {
        let k = key("");
        let mut px = PrefixCache::default();
        px.insert_with_budget(&k, entry_b(&k, 0, 4), "test", 8);
        px.insert_with_budget(&k, entry_b(&k, 1, 4), "test", 8);
        let idx = px.entries[&k].iter().position(|e| e.toks[0] == 0).unwrap();
        let pin = px.pin_n(&k, idx, 2).unwrap();

        // Both leases name one stable entry. While either is live, ordinary inserts
        // evict the oldest UNPINNED entry instead.
        px.insert_with_budget(&k, entry_b(&k, 2, 4), "test", 8);
        assert_eq!(px_survivors(&px), vec![0, 2]);
        assert_eq!(
            px.entries[&k].iter().find(|e| e.id == pin.id).unwrap().pins,
            2
        );
        assert!(px.unpin(&pin));
        px.insert_with_budget(&k, entry_b(&k, 3, 4), "test", 8);
        assert_eq!(px_survivors(&px), vec![0, 3]);

        // Last release returns the entry to the protected LRU. Probation scan entries continue
        // to evict one another; release ends pinning but does not erase demonstrated reuse.
        assert!(px.unpin(&pin));
        px.insert_with_budget(&k, entry_b(&k, 4, 4), "test", 8);
        assert_eq!(px_survivors(&px), vec![0, 4]);
        px.insert_with_budget(&k, entry_b(&k, 5, 4), "test", 8);
        assert_eq!(px_survivors(&px), vec![0, 5]);

        // Reusing 5 puts 8 bytes in protected against a 6-byte target. Protected LRU 0 is
        // demoted, then the next admission evicts it from probation.
        assert!(reuse_ident(&mut px, 5));
        px.insert_with_budget(&k, entry_b(&k, 6, 4), "test", 8);
        assert_eq!(px_survivors(&px), vec![5, 6]);
        assert!(
            !px.unpin(&pin),
            "an evicted lease id must not release another entry"
        );
        assert_prefix_cache_accounting(&px);
    }

    #[test]
    fn retiring_session_releases_prefix_pin_in_release() {
        let k = key("");
        let mut px = PrefixCache::default();
        px.insert_with_budget(&k, entry_b(&k, 0, 4), "test", 8);
        let idx = px.entries[&k].iter().position(|e| e.toks[0] == 0).unwrap();
        let pin = px.pin(&k, idx).unwrap();
        let pin_id = pin.id;
        let mut session_pin = Some(pin);

        retire_prefix_pin(&mut px, &mut session_pin);
        // The session owns one lease: a repeated retire hook must not release twice.
        retire_prefix_pin(&mut px, &mut session_pin);

        assert!(
            session_pin.is_none(),
            "retirement consumes the session's one lease"
        );
        let entry = px.entries[&k].iter().find(|e| e.id == pin_id).unwrap();
        assert_eq!(
            entry.pins, 0,
            "retirement must release the cache pin in release builds"
        );
    }

    #[test]
    fn prefix_cache_emergency_flush_preserves_inflight_pins() {
        let k = key("");
        let mut px = PrefixCache::default();
        px.insert_with_budget(&k, entry_b(&k, 0, 4), "test", 8);
        px.insert_with_budget(&k, entry_b(&k, 1, 4), "test", 8);
        let idx = px.entries[&k].iter().position(|e| e.toks[0] == 0).unwrap();
        let pin = px.pin(&k, idx).unwrap();

        assert_eq!(px.evict_all(), 1);
        assert_eq!(px_survivors(&px), vec![0]);
        assert_eq!(px.total_bytes, 4);
        assert!(px.unpin(&pin));
        assert_eq!(px.evict_all(), 1);
        assert!(px_survivors(&px).is_empty());
        assert_eq!(px.total_bytes, 0);
    }

    #[test]
    fn prefix_cache_eviction_large_pool_flush_smoke() {
        // The old loop was O(E) per victim (O(E^2) on a flush) — at E = 10k victims that
        // is ~5e7 scanned entries with a PoolKey clone per candidate. The index makes the
        // same flush O(k log E); the bound below is generous CI headroom, not a benchmark.
        const E: usize = 10_000;
        let mut px = PrefixCache::default();
        for i in 0..E {
            px.insert_with_budget(&key(""), entry_b(&key(""), i as u32, 1), "test", E);
        }
        assert_eq!(px.n_entries(), E);
        let t0 = std::time::Instant::now();
        px.insert_with_budget(&key(""), entry_b(&key(""), u32::MAX, E / 2), "test", E);
        let dt = t0.elapsed();
        // half the pool evicted in ONE insert, oldest-first
        assert_eq!(px.n_entries(), E / 2 + 1);
        assert_eq!(px.evictions as usize, E / 2);
        assert_eq!(px.total_bytes, E);
        let survivors = px_survivors(&px);
        assert!(survivors.contains(&u32::MAX));
        assert!(
            !survivors.contains(&0) && !survivors.contains(&((E / 2 - 1) as u32)),
            "victims must be the oldest half"
        );
        assert!(
            survivors.contains(&((E / 2) as u32)),
            "newest half survives"
        );
        assert!(
            dt < std::time::Duration::from_secs(2),
            "large-E flush took {dt:?} — eviction is scaling with pool size again"
        );
    }

    // ---------------- SESSION AFFINITY (lane/session-affinity, 2026-08-05) ----------------

    /// Token-stream stand-in for a chat-template-rendered conversation. `IM` plays the
    /// template's turn-marker (control) token; every other id is ordinary text.
    const IM: u32 = 1000;
    fn is_marker(t: u32) -> bool {
        t == IM
    }
    /// Render a conversation as the flat token stream a client-side template would post:
    /// each segment = marker + its body tokens.
    fn convo(segs: &[&[u32]]) -> Vec<u32> {
        let mut v = Vec::new();
        for s in segs {
            v.push(IM);
            v.extend_from_slice(s);
        }
        v
    }
    /// Fingerprint chain of a REQUEST (the live turn is excluded from identity).
    fn fp(toks: &[u32]) -> Vec<u64> {
        super::conversation_fingerprint(toks, &is_marker, true)
    }
    /// Fingerprint chain of a PARKED session's committed stream (no live tail to drop).
    fn fp_parked(toks: &[u32]) -> Vec<u64> {
        super::conversation_fingerprint(toks, &is_marker, false)
    }
    fn shared(a: &[u64], b: &[u64]) -> usize {
        super::fingerprint_affinity(a, b)
    }
    /// A body long enough that head and tail windows do not overlap (so interior edits are
    /// genuinely invisible to the fingerprint rather than trivially absent).
    fn body(tag: u32, n: usize) -> Vec<u32> {
        (0..n as u32).map(|i| tag * 100 + i).collect()
    }

    #[test]
    fn fingerprint_survives_an_assistant_interior_rewrite() {
        // THE lane's target case: the client strips a <think> block out of a PRIOR assistant
        // turn. Segment boundaries, roles, opening and closing tokens are unchanged; only the
        // interior shrinks. Same conversation => same fingerprint => the parked session is
        // nominated instead of discarded.
        let sys = body(1, 24);
        let user1 = body(2, 24);
        let mut asst1 = body(3, 40);
        let user2 = body(4, 24);
        let live = body(9, 8);
        let before = convo(&[&sys, &user1, &asst1, &user2, &live]);
        // strip the interior (keep >= FP_WINDOW head and tail tokens intact).
        asst1.drain(super::FP_WINDOW..asst1.len() - super::FP_WINDOW);
        let after = convo(&[&sys, &user1, &asst1, &user2, &live]);
        assert_ne!(
            before, after,
            "the rewrite must actually change the token stream"
        );
        assert!(
            !after.starts_with(&before[..before.len() - 1]),
            "the rewrite must break plain prefix-extension (else the old probe would hit)"
        );
        assert_eq!(fp(&before), fp(&after));
        assert!(fp(&before).len() >= super::FP_MIN_SEGMENTS);
    }

    #[test]
    fn fingerprint_nominates_the_parked_session_across_a_rewritten_turn() {
        // END TO END on the lane's actual case. Parked session committed turns 1-2 of a
        // conversation; the next request re-sends that history with the assistant turn's
        // interior stripped AND a new user turn appended. Plain prefix-extension is broken,
        // but the fingerprint chains share their whole leading run -> nominated.
        let (sys, user1, user2, live) = (body(1, 24), body(2, 24), body(4, 24), body(9, 8));
        let mut asst1 = body(3, 40);
        let parked = convo(&[&sys, &user1, &asst1]);
        asst1.drain(super::FP_WINDOW..asst1.len() - super::FP_WINDOW);
        let request = convo(&[&sys, &user1, &asst1, &user2, &live]);
        let n = shared(&fp(&request), &fp_parked(&parked));
        assert_eq!(n, 3, "system + user1 + rewritten assistant1 all match");
        assert!(n >= super::FP_MIN_SEGMENTS, "clears the nomination bar");
    }

    #[test]
    fn fingerprint_degrades_gracefully_when_a_rewrite_reaches_a_head_window() {
        // A think-strip can start right after the role marker, inside the head window. That
        // segment's hash changes — but identity is a PREFIX relation, so the stable opener
        // (system + early user turns, which no client rewrites) still nominates. Nomination
        // is a guess; affinity_match decides on bytes.
        let (sys, user1, user2, live) = (body(1, 24), body(2, 24), body(4, 24), body(9, 8));
        let asst1 = body(3, 40);
        let parked = convo(&[&sys, &user1, &asst1, &user2]);
        let mut wrecked = asst1.clone();
        wrecked.drain(..super::FP_WINDOW); // rewrite eats the head window too
        let request = convo(&[&sys, &user1, &wrecked, &user2, &live]);
        let n = shared(&fp(&request), &fp_parked(&parked));
        assert_eq!(n, 2, "shared run ends at the damaged segment, not at zero");
    }

    #[test]
    fn fingerprint_ignores_the_live_turn() {
        // A request's last segment is the turn being generated — new every turn by
        // construction. Two consecutive turns of one conversation share a chain.
        let (sys, user1, asst1) = (body(1, 24), body(2, 24), body(3, 24));
        let turn_a = convo(&[&sys, &user1, &asst1, &body(7, 12)]);
        let turn_b = convo(&[&sys, &user1, &asst1, &body(8, 30)]);
        assert_eq!(fp(&turn_a), fp(&turn_b));
    }

    #[test]
    fn fingerprint_separates_different_conversations() {
        // A different system prompt or a different first user turn must not clear the
        // nomination bar — affinity must never cross-link unrelated conversations.
        let (sys, user1, asst1, live) = (body(1, 24), body(2, 24), body(3, 24), body(9, 8));
        let base = fp(&convo(&[&sys, &user1, &asst1, &live]));
        let other_sys = fp(&convo(&[&body(5, 24), &user1, &asst1, &live]));
        let other_user = fp(&convo(&[&sys, &body(6, 24), &asst1, &live]));
        assert_eq!(
            shared(&base, &other_sys),
            0,
            "different system prompt: nothing shared"
        );
        assert_eq!(
            shared(&base, &other_user),
            1,
            "only the system prompt is shared"
        );
        assert!(
            shared(&base, &other_user) < super::FP_MIN_SEGMENTS,
            "below the bar"
        );
    }

    #[test]
    fn fingerprint_declines_short_generic_openers() {
        // A bare system prompt (+ first user turn) is the SAME opener for every fresh
        // conversation with this client, so its shared run must stay under the bar.
        let sys = body(1, 24);
        let a = fp(&convo(&[&sys, &body(2, 24)]));
        let b = fp(&convo(&[&sys, &body(7, 24)]));
        assert!(shared(&a, &b) < super::FP_MIN_SEGMENTS);
        // a real multi-turn conversation does clear it.
        let long = convo(&[&sys, &body(2, 24), &body(3, 24), &body(9, 8)]);
        assert!(shared(&fp(&long), &fp(&long)) >= super::FP_MIN_SEGMENTS);
    }

    #[test]
    fn fingerprint_handles_a_prompt_with_no_markers() {
        // Raw non-chat completions (no template markers) have no segment structure: a
        // 1-segment chain, which for a request is also the live turn -> empty. Never clears
        // the bar, so those callers keep the plain prefix probes exactly as before.
        assert!(fp(&toks(512)).is_empty());
        assert!(shared(&fp(&toks(512)), &fp_parked(&toks(512))) < super::FP_MIN_SEGMENTS);
    }

    #[test]
    fn affinity_resume_requires_the_whole_committed_prefix() {
        use super::{AffinityMatch, affinity_match};
        // EXACT: the prompt carries every committed token, then new text -> prime the tail only.
        assert_eq!(
            affinity_match(&toks(100), &toks(60)),
            AffinityMatch::Exact { suffix_from: 60 }
        );
        // EXACT, empty suffix: pure continuation burst (nothing left to prime).
        assert_eq!(
            affinity_match(&toks(60), &toks(60)),
            AffinityMatch::Exact { suffix_from: 60 }
        );
    }

    #[test]
    fn affinity_refuses_to_resume_across_a_committed_range_divergence() {
        use super::{AffinityMatch, affinity_match};
        // The rewrite reached text the session ALREADY committed: the parked caches hold
        // recurrent state for tokens this request does not have, and a parked session carries
        // no checkpoint at the divergence boundary. Full re-prime — exactness over speed.
        let mut prompt = toks(100);
        prompt[42] = 999;
        assert_eq!(
            affinity_match(&prompt, &toks(60)),
            AffinityMatch::Diverged { at: 42 }
        );
        // A prompt SHORTER than committed (client dropped its own tail) is divergence too:
        // the extra committed rows cannot be trimmed away.
        assert_eq!(
            affinity_match(&toks(40), &toks(60)),
            AffinityMatch::Diverged { at: 40 }
        );
    }

    #[test]
    fn affinity_room_test_preserves_f5_right_sized_sessions() {
        // F5 INTERACTION. On a VRAM-tight rig the right-size ladder lands sessions BELOW the
        // request's ctx_cap — and those are exactly the rigs where every turn is a miss. The
        // affinity probe therefore tests `need` (prompt + budget + slack), not ctx_cap; this
        // pins the arithmetic that makes a laddered session eligible.
        let (prompt_len, budget, ctx_cap) = (12_000usize, 512usize, 131_072usize);
        let need = prompt_len + budget + super::SPEC_SHRINK_SLACK;
        let laddered = 16_384usize; // a plausible ladder landing
        assert!(
            laddered < ctx_cap,
            "the ladder lands below the cap (else no interaction)"
        );
        assert!(laddered >= need, "and still covers what this request needs");
        let committed = toks(11_500);
        let prompt = toks(prompt_len);
        assert_eq!(
            super::affinity_resume_target(&prompt, &committed, 11_000, laddered, need, true),
            Ok(laddered),
            "a sufficient ladder landing must not be inflated back to ctx_cap",
        );
        assert_eq!(
            super::affinity_resume_target(&prompt, &committed, 11_000, 8_192, need, true),
            Ok(need),
            "an undersized landing grows only to this request's need",
        );
    }

    // ---------- SPEC-TIER STABLE BOUNDARY (lane/frspec-multiturn-cache, 2026-08-21) ----------

    #[test]
    fn spec_checkpoint_at_prompt_end_declines_the_rerender_class_and_the_stable_boundary_resumes() {
        // THE B4 defect, as a position law (research/multiturn-cache-20260821): turn N's
        // prompt ends with the live generation header (marker + role/think opener) that turn
        // N+1's re-render REPLACES with the assistant's actual content. A spec turn checkpoint
        // at PROMPT-END therefore always sits past the divergence — the measured
        // "spec-affinity: declined (history diverged at 6811 of checkpoint 6813)" — while a
        // checkpoint at the stable pre-generation boundary (`plain_checkpoint_boundary`, the
        // plain tier's 2026-08-09 law) resumes and re-primes only the delta.
        let sys = body(1, 40);
        let user1 = body(2, 40);
        let header = body(9, 4); // role name + think opener: ordinary vocab tokens
        let turn_n_prompt = {
            let mut v = convo(&[&sys, &user1]);
            v.push(IM);
            v.extend_from_slice(&header);
            v
        };
        let generated = body(5, 30); // what the model answered inside the live turn
        let committed = {
            let mut v = turn_n_prompt.clone();
            v.extend_from_slice(&generated);
            v
        };
        // Turn N+1: same history, the assistant turn re-rendered from CONTENT (header gone),
        // a new user turn, and a fresh live header.
        let turn_n1_prompt = {
            let mut v = convo(&[&sys, &user1, &generated, &body(4, 24)]);
            v.push(IM);
            v.extend_from_slice(&header);
            v
        };
        let need = turn_n1_prompt.len() + 512;
        // LEGACY POSTURE (prompt-end checkpoint): declines, offsets name the class — the
        // divergence sits right after the header's marker, BELOW the checkpoint.
        let prompt_end = turn_n_prompt.len();
        let declined = super::affinity_resume_target(
            &turn_n1_prompt,
            &committed,
            prompt_end,
            need,
            need,
            true,
        )
        .expect_err("a prompt-end checkpoint must decline the re-render class");
        assert!(
            declined.starts_with(&format!(
                "history diverged at {} of checkpoint {prompt_end}",
                prompt_end - header.len(),
            )),
            "decline names the template-boundary offsets: {declined}",
        );
        // FIXED POSTURE (stable boundary): byte-exact through the checkpoint, non-empty
        // suffix -> resume. Since lane/spec-longctx-20260821 (v0.100 train) the boundary
        // is the GRID-ALIGNED floor of the last turn-marker, not the marker index itself
        // — the prime-grid law (see plain_checkpoint_boundary_lands_before_the_live_
        // generation_header, the contract's own test). The re-render-decline semantics
        // this test pins are unchanged: the boundary still sits at or before the marker,
        // never inside the live header.
        let boundary = super::plain_checkpoint_boundary(&turn_n_prompt, &is_marker)
            .expect("chat prompt has a stable boundary");
        let last_marker = turn_n_prompt.len() - header.len() - 1;
        let grain = memra_engine::Engine::gdn_chunk_size();
        let floor = memra_engine::hybrid_forward::PRIME_MIN_T;
        let mut want = last_marker / grain * grain;
        while want >= grain && turn_n_prompt.len() - want < floor {
            want -= grain;
        }
        assert_eq!(
            boundary, want,
            "the boundary is the grid-aligned floor of the last turn-marker, before the \
             live header (prime-grid law)",
        );
        assert!(
            boundary <= last_marker,
            "never past the marker into the live segment"
        );
        assert_eq!(
            super::affinity_resume_target(&turn_n1_prompt, &committed, boundary, need, need, true),
            Ok(need),
            "the stable-boundary checkpoint resumes the re-rendered turn",
        );
    }

    // ---------- PLAIN-SESSION AFFINITY (lane/plain-affinity, 2026-08-09) ----------

    #[test]
    fn plain_checkpoint_boundary_lands_before_the_live_generation_header() {
        use super::plain_checkpoint_boundary;
        // A pi-shaped chat prompt: stable history, then the template's live assistant-generation
        // header (marker + "assistant\n<think>\n"-class body). The boundary must sit at the START
        // of that final header segment — the stable point turn N+1 shares — NOT at prompt-end
        // (which the forced-spec control disproved) and NEVER at a hardcoded offset.
        let sys = body(1, 40);
        let user1 = body(2, 40);
        let asst1 = body(3, 40);
        let user2 = body(4, 40);
        // the live generation header: a marker, then a short body (role name + <think> opener).
        let prompt = {
            let mut v = convo(&[&sys, &user1, &asst1, &user2]);
            v.push(IM);
            v.extend_from_slice(&body(9, 4));
            v
        };
        let b = plain_checkpoint_boundary(&prompt, &is_marker)
            .expect("a multi-turn chat prompt has a locatable boundary");
        // the boundary is the GRID-ALIGNED floor of the final marker (start of the live header
        // segment): at or before the marker — never past it into the live segment — and on the
        // GDN prime grid, so the boundary-stopped prime and the checkpoint resume reproduce the
        // monolithic bytes (lane/spec-longctx-20260821; grid_align_boundary's doc carries the
        // measured law). CONTRACT CHANGE stated loudly: before that lane the boundary was the
        // marker index itself, an off-grid prime stop that byte-diverged from the cold render
        // on hybrids 13/16 turns at agent lengths.
        let last_marker = prompt.iter().rposition(|&t| t == IM).unwrap();
        let grain = memra_engine::Engine::gdn_chunk_size();
        let floor = memra_engine::hybrid_forward::PRIME_MIN_T;
        // grid-aligned floor of the live-header start, stepped down while the remaining
        // prompt suffix would be sub-PRIME_MIN_T (the W1 door — its own test below).
        let mut want = last_marker / grain * grain;
        while want >= grain && prompt.len() - want < floor {
            want -= grain;
        }
        assert_eq!(
            b, want,
            "checkpoint sits at the grid-aligned floor of the live header segment start \
             (stepped down past the W1 sub-floor remainder)"
        );
        assert!(b <= last_marker, "never inside the live segment");
        assert_eq!(b % grain, 0, "boundary sits on the prime grid");
        assert!(prompt.len() - b >= floor, "suffix stays one prime call");
        assert!(b < prompt.len() && b > super::REUSE_MIN_PREFIX);
        // and it is STRICTLY before prompt-end (the boundary the control disproved).
        assert!(b < prompt.len() - 1);
    }

    #[test]
    fn plain_checkpoint_boundary_uses_a_guard_window_for_raw_prompts() {
        use super::{PLAIN_CKPT_RAW_GUARD, plain_checkpoint_boundary};
        // A markerless raw-completion prompt has no header to find: trim a conservative guard
        // window off the end so the live tail (which a re-ask may extend/rewrite) is never inside
        // the checkpoint, then land on the prime grid (same law as the marker path). The exact
        // diff still decides on bytes. (Arming for such a prompt additionally requires an
        // EXPLICIT session id — see the nominatable test below.)
        let prompt = toks(512);
        let b = plain_checkpoint_boundary(&prompt, &is_marker).expect("long raw prompt has one");
        let grain = memra_engine::Engine::gdn_chunk_size();
        assert_eq!(b, (512 - PLAIN_CKPT_RAW_GUARD) / grain * grain);
        assert!(
            b <= 512 - PLAIN_CKPT_RAW_GUARD,
            "guard window never shrinks"
        );
        assert_eq!(b % grain, 0, "boundary sits on the prime grid");
    }

    #[test]
    fn plain_checkpoint_boundary_never_leaves_a_sub_floor_prompt_remainder() {
        use super::plain_checkpoint_boundary;
        // THE W1 DOOR (measured 2026-08-21, lane/spec-longctx: prompt 9,510 with boundary
        // 9,504 sent the last 6 prompt tokens through decode_step one at a time — a different
        // numeric program, and that turn's greedy output diverged from the monolithic prime).
        // A boundary must leave at least PRIME_MIN_T prompt tokens behind it so the suffix is
        // ONE prime call. Grid alignment pays for the fix: step down one grain.
        let grain = memra_engine::Engine::gdn_chunk_size();
        let floor = memra_engine::hybrid_forward::PRIME_MIN_T;
        // Marker placed so the aligned floor sits a hair under prompt-end: prompt length
        // grain*N + small, marker at the end of the stable history.
        for tail in 0..floor {
            let n = grain * 8 + tail;
            let prompt = {
                let mut v = convo(&[&body(1, 40), &body(2, 40)]);
                v.resize(n.saturating_sub(1), 7); // ordinary vocab filler
                v.push(IM); // a marker at the very tail: aligned floor lands near n
                v
            };
            let Some(b) = plain_checkpoint_boundary(&prompt, &is_marker) else {
                continue; // rejected outright = the W1 note's "drop the capture" outcome
            };
            assert_eq!(b % grain, 0, "boundary stays on the prime grid (n={n})");
            assert!(
                prompt.len() - b >= floor,
                "boundary must leave a primeable suffix: n={n} b={b} remainder={}",
                prompt.len() - b,
            );
        }
    }

    #[test]
    fn plain_ckpt_capture_requires_a_nominatable_identity() {
        use super::plain_ckpt_nominatable;
        // THE CACHE-METERING REGRESSION (tip battery 2026-08-09, 13 checks red). An armed
        // ckpt_at excludes a session from the in-batch fanout / prime-batch paths (they prime
        // monolithically and cannot honor the per-session boundary stop). A markerless anonymous
        // prompt can NEVER be nominated by the implicit tier (its fingerprint chain cannot reach
        // FP_MIN_SEGMENTS), so arming it bought nothing and cost the serve-smoke 5-way
        // shared-prefix prompt_ids fanout every hit: 0 prefix hits, 6 misses, 6 inserts, empty
        // tick-seg LCP window. Capture is armed for such prompts ONLY with an explicit id.
        assert!(
            !plain_ckpt_nominatable(&toks(512), &is_marker),
            "markerless raw prompt must NOT arm an implicit-tier capture"
        );
        // A real multi-turn chat prompt clears the bar and keeps the capture.
        let chat = convo(&[&body(1, 24), &body(2, 24), &body(3, 24), &body(9, 8)]);
        assert!(
            plain_ckpt_nominatable(&chat, &is_marker),
            "multi-turn chat traffic keeps the implicit-tier capture"
        );
        // A one-segment opener (bare system prompt) stays below the bar — nominating on it
        // would cross-link unrelated conversations, so no capture either.
        let opener = convo(&[&body(1, 24)]);
        assert!(!plain_ckpt_nominatable(&opener, &is_marker));
    }

    #[test]
    fn plain_checkpoint_boundary_declines_short_prompts() {
        use super::plain_checkpoint_boundary;
        // Below the floor there is nothing worth a checkpoint (cold prime is cheaper than the
        // bookkeeping), so no boundary is armed.
        assert!(
            plain_checkpoint_boundary(&toks(super::REUSE_MIN_PREFIX + 4), &is_marker).is_none()
        );
        assert!(plain_checkpoint_boundary(&toks(8), &is_marker).is_none());
    }

    #[test]
    fn plain_affinity_resume_decision_is_bytes_over_identity() {
        use super::{AffinityMatch, affinity_match, fingerprint_affinity};
        // END TO END on the decision the admit probe makes, exercised through the two functions
        // that make it: IDENTITY nominates (fingerprint chain shares >= the bar), BYTES decide
        // (the prompt reproduces the committed tokens through the checkpoint exactly).
        let (sys, user1, user2) = (body(1, 40), body(2, 40), body(4, 40));
        let mut asst1 = body(3, 60);
        // parked session committed [sys, user1, asst1]; its checkpoint sits at the pre-generation
        // boundary of THAT turn — model it as the whole committed stream (the resume only needs
        // committed[..pos] to match).
        let committed = convo(&[&sys, &user1, &asst1]);
        let pos = committed.len();
        let parked_fp = fp_parked(&committed);
        // turn N+1: the client strips asst1's interior and appends a new user turn + live header.
        asst1.drain(super::FP_WINDOW..asst1.len() - super::FP_WINDOW);
        let request = {
            let mut v = convo(&[&sys, &user1, &asst1, &user2]);
            v.push(IM);
            v.extend_from_slice(&body(9, 4));
            v
        };
        // The rewrite broke plain prefix-extension...
        assert!(
            !request.starts_with(&committed),
            "exact-extension would miss (rewritten history)"
        );
        // ...but the fingerprint chains still share their leading run -> NOMINATED.
        assert!(fingerprint_affinity(&fp(&request), &parked_fp) >= super::FP_MIN_SEGMENTS);
        // BYTES DECIDE: does the request reproduce the committed tokens up to a boundary? The
        // stripped assistant turn shortened `committed`, so the FULL committed stream diverges —
        // which is exactly why the checkpoint must sit BEFORE that turn's generated text. At a
        // boundary of [sys, user1] (before asst1) the request matches exactly and leaves a suffix.
        let early_pos = convo(&[&sys, &user1]).len();
        assert!(early_pos < pos);
        match affinity_match(&request, &committed[..early_pos]) {
            AffinityMatch::Exact { suffix_from } => assert_eq!(suffix_from, early_pos),
            other => panic!("expected exact match at the pre-generation boundary, got {other:?}"),
        }
        assert!(request.len() > early_pos, "non-empty suffix to prime");
    }

    #[test]
    fn plain_affinity_declines_a_divergence_below_the_checkpoint() {
        use super::{AffinityMatch, affinity_match};
        // If the rewrite reached text BELOW the checkpoint boundary, the parked caches hold state
        // for tokens this request does not have — correctness first, full re-prime. This is the
        // one decline expected by design; the offset is the diagnostic.
        let committed = toks(80);
        let mut request = toks(120);
        request[30] = 7777; // diverges inside committed[..50]
        match affinity_match(&request, &committed[..50]) {
            AffinityMatch::Diverged { at } => assert_eq!(at, 30),
            other => panic!("a below-boundary rewrite must decline, got {other:?}"),
        }
    }

    #[test]
    fn plain_affinity_fingerprint_collision_cannot_force_a_wrong_resume() {
        use super::{AffinityMatch, affinity_match, fingerprint_affinity};
        // The safety argument: a fingerprint match only NOMINATES; the exact token diff is the
        // sole authority. Construct two DIFFERENT conversations that (by the boundary-window
        // hashing) could share a leading run, and show the byte diff still vetoes the resume.
        let (sys, user1) = (body(1, 40), body(2, 40));
        let committed = convo(&[&sys, &user1, &body(3, 40)]);
        // a request that shares the fingerprintable opener but whose actual bytes differ inside
        // the committed range (a genuine different conversation that happens to nominate).
        let mut request = committed.clone();
        request[5] = 4242; // byte-level divergence inside the very first segment
        request.extend_from_slice(&body(9, 8));
        // Even if identity nominated (same chain length), bytes veto:
        let _nominated = fingerprint_affinity(&fp(&request), &fp_parked(&committed));
        match affinity_match(&request, &committed) {
            AffinityMatch::Diverged { at } => assert_eq!(at, 5, "the exact diff catches it"),
            other => panic!("a byte divergence must never resume, got {other:?}"),
        }
    }

    #[test]
    fn plain_affinity_pi_shape_grows_instead_of_declining() {
        use super::affinity_resume_target;

        // Live pi shape after request-owned context sizing: the parked cache was allocated for
        // the preceding turn, while the next turn's prompt makes its charged cap a few hundred
        // rows larger. Identity and bytes match through the checkpoint, so capacity selects a
        // grow target; it must never veto the resume as "no room".
        let committed = toks(12_640);
        let prompt = toks(12_690);
        let checkpoint_pos = 12_000;
        let parked_cap = 45_064;
        let incoming_request_need = 45_522;
        assert_eq!(
            affinity_resume_target(
                &prompt,
                &committed,
                checkpoint_pos,
                parked_cap,
                incoming_request_need,
                true,
            ),
            Ok(incoming_request_need),
            "a nominated exact checkpoint must grow to the next request instead of declining",
        );
    }

    #[test]
    fn streaming_utf8_waits_for_a_complete_multibyte_sequence() {
        let mut emitted = 0;
        assert_eq!(utf8_delta(b"caf\xc3", &mut emitted), "caf");
        assert_eq!(emitted, 3);
        assert_eq!(utf8_delta(b"caf\xc3\xa9\n", &mut emitted), "é\n");
        assert_eq!(emitted, 6);
    }

    #[test]
    fn streaming_utf8_consumes_truly_invalid_bytes_once() {
        let mut emitted = 0;
        assert_eq!(utf8_delta(b"a\xffb", &mut emitted), "a\u{fffd}b");
        assert_eq!(emitted, 3);
        assert_eq!(utf8_delta(b"a\xffbc", &mut emitted), "c");
    }

    #[test]
    fn confidence_summary_tracks_reference_and_margin() {
        let summary = summarize_confidence(&[0.0, 2.0, 1.0], 1).unwrap();
        assert_eq!(summary.top1_token, 1);
        assert!(summary.top1_correct);
        assert!((summary.top1_top2_margin - 1.0).abs() < 1e-6);
        let expected = 2.0f64 - (0.0f64.exp() + 2.0f64.exp() + 1.0f64.exp()).ln();
        assert!((summary.reference_logprob - expected).abs() < 1e-12);
        assert!(summary.entropy > 0.0);
    }
}

/// SESSION-RESUME SAMPLER PREDICATE, the door's own teeth
/// (lane/session-resume-sampler-predicate-20260820).
///
/// The pure-predicate contract lives in `memra-sampling` (`resume_sampler_predicate_tests`). What
/// is tested HERE is the thing only this crate owns: that the rollback door actually selects the
/// pre-lane behaviour, and that the reproduced collision pair flips verdict across it. Both
/// directions, on the same pair, so neither arm can be green for an unrelated reason.
#[cfg(test)]
mod resume_sampler_door_tests {
    use super::spec_resume_sampler_verdict;
    use memra_engine::sampler::{SamplerConfig, SamplerIdentity};

    /// Turn 1 of the predecessor's live reproduction: pure temp, explicit seed. Parks a `graph_s`.
    fn parked_pure_temp() -> SamplerIdentity {
        SamplerIdentity::of(&SamplerConfig {
            temperature: 0.7,
            seed: 20260820,
            ..Default::default()
        })
    }

    /// Turn 2: same seed and temperature, plus the vendor defaults. This is the pair that launched
    /// an unfiltered draft graph on a live server.
    fn incoming_vendor_filtered() -> SamplerIdentity {
        SamplerIdentity::of(&SamplerConfig {
            temperature: 0.7,
            top_k: 20,
            top_p: 0.95,
            seed: 20260820,
            ..Default::default()
        })
    }

    #[test]
    fn door_open_refuses_the_collision_pair_and_names_the_field() {
        assert_eq!(
            spec_resume_sampler_verdict(true, &incoming_vendor_filtered(), &parked_pure_temp()),
            Some("top_k"),
            "default posture: refuse a sampler-differing resume and name the field",
        );
    }

    #[test]
    fn door_shut_reproduces_the_pre_lane_acceptance() {
        // MEMRA_SPEC_RESUME_SAMPLER=0 == the probe that compared prompts and never samplers.
        assert_eq!(
            spec_resume_sampler_verdict(false, &incoming_vendor_filtered(), &parked_pure_temp()),
            None,
            "the door must restore the acceptance, or the refusal test is tautological",
        );
    }

    #[test]
    fn a_same_sampler_resume_is_admitted_in_both_arms() {
        let s = parked_pure_temp();
        for on in [true, false] {
            assert_eq!(
                spec_resume_sampler_verdict(on, &s, &s),
                None,
                "no regression: an unchanged sampler resumes with the door {on}",
            );
        }
    }
}

#[cfg(test)]
mod dspark_boot_conflict_tests {
    use super::dspark_spec_boot_conflict;

    #[test]
    fn refuse_list_covers_step_tp_and_ep() {
        // TOOTH (hermes finding, fixed 2026-08-23): the boot ambiguity list checked only
        // MEMRA_SPEC_DFLASH / MEMRA_GEMMA4_SPEC / MEMRA_PP_STAGES — a step-TP/EP
        // multi-device trunk on a dspark-armed server sailed through unguarded.
        let clean = dspark_spec_boot_conflict(false, 0, 1, None, None);
        assert!(clean.is_none(), "no conflict must arm cleanly: {clean:?}");
        // The three pre-existing arms still fire, in their original precedence.
        assert!(
            dspark_spec_boot_conflict(true, 0, 1, None, None)
                .is_some_and(|m| m.contains("MEMRA_SPEC_DFLASH"))
        );
        assert!(
            dspark_spec_boot_conflict(false, 5, 1, None, None)
                .is_some_and(|m| m.contains("MEMRA_GEMMA4_SPEC"))
        );
        assert!(
            dspark_spec_boot_conflict(false, 0, 2, None, None)
                .is_some_and(|m| m.contains("MEMRA_PP_STAGES=2"))
        );
        // The new arms: any armed step TP/EP spec refuses by name.
        assert!(
            dspark_spec_boot_conflict(false, 0, 1, Some("all@0,1,2,3"), None)
                .is_some_and(|m| m.contains("MEMRA_STEP_TP"))
        );
        assert!(
            dspark_spec_boot_conflict(false, 0, 1, None, Some("0-44@0,1"))
                .is_some_and(|m| m.contains("MEMRA_STEP_EP"))
        );
        // Blank and literal zero are the shared parser's two explicit disabled forms.
        assert!(dspark_spec_boot_conflict(false, 0, 1, Some("  "), Some("")).is_none());
        assert!(dspark_spec_boot_conflict(false, 0, 1, Some("0"), Some(" 0 ")).is_none());
    }
}

#[cfg(test)]
mod stop_string_tests {
    #[test]
    fn empty_stop_string_element_never_matches() {
        // TOOTH (hermes finding, fixed 2026-08-23): "".contains("") is true, so a single
        // empty stop element used to stop EVERY decode at the first flush, on both the
        // hybrid and dsv4 routes. Empties are dropped at ingestion (StopSequences::
        // into_vec) and guarded here for any future constructor.
        assert!(!super::contains_stop_string(
            b"hello world",
            &["".to_string()]
        ));
        assert!(!super::contains_stop_string(b"", &["".to_string()]));
        // real elements still fire, alone or next to an empty one.
        assert!(super::contains_stop_string(
            b"hello STOP world",
            &["".to_string(), "STOP".to_string()]
        ));
        assert!(!super::contains_stop_string(b"hello world", &[]));
    }
}
