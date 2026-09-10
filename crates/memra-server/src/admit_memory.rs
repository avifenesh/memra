//! Memory-shaped admission for a high session ceiling (door `MEMRA_ADMIT_BY_MEMORY`).
//!
//! WHY THIS EXISTS. The GLM-5.3-Flash 1M route on the 2x B200 pair serves behind
//! `MEMRA_MAX_SESSIONS=4` (darklanes `deploy/glm5b200/glm5b200_serve_launch.sh`), and the
//! launcher says why in its own words: "a 1M session's latent KV is ~13 GB; four of them plus
//! the resident experts and the prefix cache fit the pair's 2x183 GB with room". That number
//! is a WORST-CASE PROXY: the cap is sized as if every admitted session were a full-envelope
//! one, so a 4k probe occupies the same slot as a 900k conversation. On 2026-09-09, after the
//! 1M cutover, the sweep read `verdict=reject-slot reason=tenant-p95 inflight=4 cap=4` while
//! short probes queued past the client's 10 s pre-header budget behind four long generations,
//! against a catalog that advertises `capacity.concurrency 8`.
//!
//! The engine already admits by MEMORY, exactly and per request: `AdmissionCostModel` charges
//! this request's own `prompt + max_output` context at the model's real bytes/token (latent
//! MLA planes and the DSA indexer pool keys included since the prefix-latent lane, see
//! `memra_kv::latent_kv_bytes_per_token_for_plan`), the VRAM gate compares it against live
//! effective free VRAM per device, and a reclaim ladder frees space before deciding. What the
//! slot cap adds on top of that is a SECOND, cruder gate that binds first. This door removes
//! the reasons the cruder gate was needed:
//!
//!   1. OPEN-OUTPUT CHARGE. A request with no `max_tokens` and no `max_ctx` charges
//!      `MEMRA_CTX` — the whole 1,048,576-token envelope — through `request_ctx_cap`'s
//!      `(None, MAX_NEW_CTX_BOUNDED)` arm. On a deployment whose registry pins
//!      `max_output_length` the HTTP layer bounds `max_new` first, so this arm is the NAKED
//!      case; when it is reached it books ~13 GB for a request that will emit a few hundred
//!      tokens. Armed, the open-output arm charges `prompt + MEMRA_ADMIT_OPEN_OUTPUT_TOKENS`
//!      instead ([`charged_ctx_tokens`]).
//!   2. RECLAIM DESTROYS CACHE WARMTH. The admission flush drops device prefix entries
//!      (`PrefixCache::evict_all`) rather than demoting them; `host_demote_prefix_entry`'s own
//!      doc calls that flush "a named seam for a copy-stream follow-up, not an oversight".
//!      With a high ceiling the ladder fires far more often, and on this route each dropped
//!      entry is up to 43.5 GB of prime the next turn has to redo. Armed, the flush demotes
//!      into the pinned host tier first, bounded by the bytes the admission actually needs
//!      ([`demote_budget_bytes`]).
//!   3. THE DEFER IS UNBOUNDED. A request the VRAM gate cannot fit requeues FIFO forever
//!      unless `active` is empty; nothing bounds the wait against the client's own budget, so
//!      a queued request dies as a pre-header timeout rather than a refusal it can retry.
//!      Armed, a request that has been memory-deferred past `MEMRA_ADMIT_DEFER_BUDGET_MS`
//!      with both tiers exhausted gets a bounded 429 with `Retry-After`
//!      ([`MemoryVerdict::Refuse`]).
//!
//! Everything in this module is pure arithmetic over values the worker owns, so the policy is
//! CPU-testable without a card. The worker owns every side effect (the demotion, the requeue,
//! the refusal) exactly as it owns them today.

use std::time::Instant;

/// Open-output charge when the door is armed and neither `max_tokens` nor `max_ctx` bounds the
/// request. 8192 is the `default_output_length` the fleet's registries already pin for the
/// large models (memra `models.toml` rows), i.e. the bound the HTTP layer applies where a
/// registry exists; the naked path now agrees with it instead of charging the whole envelope.
pub(crate) const DEFAULT_OPEN_OUTPUT_TOKENS: usize = 8192;

/// How long a request may sit in the memory-deferred requeue before the refusal is preferred
/// to the silence. 8 s sits inside the 10 s pre-header budget the darklanes edge gives a
/// request (RESULTS-CUTOVER.md addendum): a refusal the client can read beats a timeout it
/// cannot.
pub(crate) const DEFAULT_DEFER_BUDGET_MS: u64 = 8_000;

/// `prompt + output + this` — the same `+8` slack `request_ctx_cap` adds on the bounded arm.
pub(crate) const CTX_SLACK: usize = 8;

/// Retry-After floor/ceiling for a memory refusal: the shed contract's window, the same one
/// `admit_predict::earliest_completion_retry_s` clamps to.
pub(crate) const RETRY_AFTER_MIN_S: u64 = 1;
pub(crate) const RETRY_AFTER_MAX_S: u64 = 60;

/// Read-once door configuration. Built at worker start; the door cannot arm mid-process.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MemoryAdmitConfig {
    /// `MEMRA_ADMIT_BY_MEMORY=1`. OFF by design: raising the session ceiling is a
    /// customer-visible capacity act, and this door only becomes true when a box has the
    /// requalification receipts for the mixed load it must now carry.
    pub armed: bool,
    /// `MEMRA_ADMIT_OPEN_OUTPUT_TOKENS` (default [`DEFAULT_OPEN_OUTPUT_TOKENS`]).
    pub open_output_tokens: usize,
    /// `MEMRA_ADMIT_DEFER_BUDGET_MS` (default [`DEFAULT_DEFER_BUDGET_MS`]). 0 disables the
    /// bounded refusal and restores today's unbounded FIFO defer while keeping 1 and 2.
    pub defer_budget_ms: u64,
}

impl MemoryAdmitConfig {
    pub(crate) fn from_env() -> Self {
        let armed = std::env::var("MEMRA_ADMIT_BY_MEMORY").is_ok_and(|v| v == "1");
        let open_output_tokens = std::env::var("MEMRA_ADMIT_OPEN_OUTPUT_TOKENS")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .filter(|&n| n > 0)
            .unwrap_or(DEFAULT_OPEN_OUTPUT_TOKENS);
        let defer_budget_ms = std::env::var("MEMRA_ADMIT_DEFER_BUDGET_MS")
            .ok()
            .and_then(|v| v.trim().parse::<u64>().ok())
            .unwrap_or(DEFAULT_DEFER_BUDGET_MS);
        MemoryAdmitConfig {
            armed,
            open_output_tokens,
            defer_budget_ms,
        }
    }

    /// The open-output charge to hand `request_ctx_cap`, or `None` when the door is off (the
    /// caller then takes today's `MEMRA_CTX` arm, byte-identically).
    pub(crate) fn open_output_charge(&self) -> Option<usize> {
        self.armed.then_some(self.open_output_tokens)
    }

    /// Boot receipt. One line, stated with its inputs, so a reader can re-derive the policy
    /// from the log without shell access to the box.
    pub(crate) fn boot_line(&self) -> String {
        format!(
            "[admit-mem] door={} open_output_tokens={} defer_budget_ms={} \
             (admission charges prompt+output, the reclaim flush demotes to the host tier \
             before dropping, and a memory defer past the budget refuses 429 instead of \
             queueing past the client's deadline)",
            if self.armed { "ON" } else { "OFF" },
            self.open_output_tokens,
            self.defer_budget_ms,
        )
    }
}

/// The charged context for a request, in tokens.
///
/// `max_output_bound` is the request's own resolved output bound (`Some` whenever
/// `max_tokens` was given or the registry supplied `default_output_length`/`max_output_length`);
/// `None` is the naked open-output case this door re-charges. `model_ctx` is the model's
/// trained length (0 = unknown, no clamp) and clamps the result exactly as
/// `request_ctx_cap` clamps today.
///
/// The result is never below `prompt + CTX_SLACK`: a charge under the prompt itself would
/// admit a session whose own prompt cannot land.
pub(crate) fn charged_ctx_tokens(
    prompt_tokens: usize,
    max_output_bound: Option<usize>,
    open_output_tokens: usize,
    model_ctx: usize,
) -> usize {
    let output = max_output_bound.unwrap_or(open_output_tokens);
    let charged = prompt_tokens
        .saturating_add(output)
        .saturating_add(CTX_SLACK)
        .max(prompt_tokens.saturating_add(CTX_SLACK));
    if model_ctx > 0 {
        charged.min(model_ctx)
    } else {
        charged
    }
}

/// A per-session KV estimate, decomposed for the receipt line.
///
/// `context_bytes` is the ring-aware context-linear term (the caller passes the model's own
/// `bytes_per_token` / ring geometry, which for a latent-KV family already carries the MLA
/// latent rows AND the DSA indexer pool keys — `memra_kv::latent_kv_bytes_per_token_for_plan`),
/// `fixed_bytes` the measured non-context residual (activations, workspaces, draft plane).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct KvEstimate {
    pub charged_ctx: usize,
    pub context_bytes: u64,
    pub fixed_bytes: u64,
}

impl KvEstimate {
    pub(crate) fn total_bytes(&self) -> u64 {
        self.context_bytes.saturating_add(self.fixed_bytes)
    }
}

/// Ring-aware context-linear bytes plus the fixed residual, at the charged context.
/// Shares `admit_predict::context_cache_bytes`' geometry contract: a ring class is capped at
/// its physical row count, the flat class scales with the whole charge.
pub(crate) fn estimate(
    charged_ctx: usize,
    bytes_per_token: u64,
    ring_bytes_per_token: u64,
    ring_rows: u64,
    fixed_bytes: u64,
) -> KvEstimate {
    let context_bytes = crate::admit_predict::context_cache_bytes(
        bytes_per_token,
        ring_bytes_per_token,
        ring_rows,
        charged_ctx as u64,
    );
    KvEstimate {
        charged_ctx,
        context_bytes,
        fixed_bytes,
    }
}

/// What the two tiers can offer this admission right now. All four are live readings the
/// worker already takes; none is a remembered constant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) struct Tiers {
    /// Device bytes the request may claim after every reserve the real gate charges
    /// (transient floor, verify-graph pool debt, per-device parallel requirements): the
    /// limiting device's `free - required_reserves`, already computed by
    /// `AdmissionHeadroom`.
    pub device_free_bytes: u64,
    /// Device bytes currently held by UNLEASED prefix-cache entries: the bytes a demotion
    /// (or, with the tier off, an eviction) can return to this admission.
    pub demotable_device_bytes: u64,
    /// Free bytes in the pinned host tier (`MEMRA_KV_HOST_MB` budget minus resident). A
    /// demotion needs somewhere to land; when the tier is off or full this is 0 and the
    /// demote arm degrades into today's drop.
    pub host_free_bytes: u64,
}

/// The memory verdict for one arrival.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MemoryVerdict {
    /// Device free already covers the estimate: admit, touching neither tier.
    Admit,
    /// Device free covers it only after the prefix cache yields; the host tier can hold the
    /// yielded bytes, so demote (not drop) and admit. `demote_bytes` is the shortfall the
    /// demotion must cover, and bounds the D2H a single tick will pay.
    DemoteThenAdmit { demote_bytes: u64 },
    /// Not enough on either tier yet, but the request is still inside its defer budget:
    /// requeue FIFO exactly as today.
    Defer { short_by: u64 },
    /// Both tiers are exhausted AND the request has waited out its defer budget: refuse with
    /// a bounded, retryable 429 rather than let it die as a pre-header timeout.
    Refuse { short_by: u64 },
}

impl MemoryVerdict {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            MemoryVerdict::Admit => "admit",
            MemoryVerdict::DemoteThenAdmit { .. } => "demote-then-admit",
            MemoryVerdict::Defer { .. } => "defer",
            MemoryVerdict::Refuse { .. } => "refuse",
        }
    }

    /// Bytes the demote arm must free, 0 for every other arm.
    pub(crate) fn demote_bytes(self) -> u64 {
        match self {
            MemoryVerdict::DemoteThenAdmit { demote_bytes } => demote_bytes,
            _ => 0,
        }
    }
}

/// The decision rule. Pure: the caller supplies the estimate, the live tier readings and how
/// long this request has already been memory-deferred.
///
/// ORDER IS THE POLICY. Device-only first (the cheap arm; no copy, no cache loss), then the
/// demotion arm (a real D2H cost, paid only when it buys the admission AND the host tier can
/// hold the bytes), then time (a request inside its budget waits, exactly as today), and only
/// then the refusal. A `defer_budget_ms` of 0 disables the refusal arm entirely and keeps
/// today's unbounded FIFO defer.
pub(crate) fn decide(
    need_bytes: u64,
    tiers: &Tiers,
    waited_ms: u64,
    defer_budget_ms: u64,
) -> MemoryVerdict {
    if need_bytes <= tiers.device_free_bytes {
        return MemoryVerdict::Admit;
    }
    let short_by = need_bytes.saturating_sub(tiers.device_free_bytes);
    // The demotion can only return bytes the cache actually holds, and only as many as the
    // host tier can accept: bytes with nowhere to land would be dropped, which is the arm
    // this door exists to stop paying blindly.
    let demotable = tiers.demotable_device_bytes.min(tiers.host_free_bytes);
    if short_by <= demotable {
        return MemoryVerdict::DemoteThenAdmit {
            demote_bytes: short_by,
        };
    }
    if defer_budget_ms == 0 || waited_ms < defer_budget_ms {
        return MemoryVerdict::Defer { short_by };
    }
    MemoryVerdict::Refuse { short_by }
}

/// Bytes the flush is allowed to copy to host for one admission: the shortfall, never the
/// whole cache. A 43.5 GB 1M-class entry costs seconds of synchronous D2H, so an unbounded
/// flush would stall the scheduler tick for every other session on the box.
pub(crate) fn demote_budget_bytes(verdict: MemoryVerdict) -> u64 {
    verdict.demote_bytes()
}

/// Clamp a computed hint into the shed contract's window; used for the `Retry-After` a
/// [`MemoryVerdict::Refuse`] sends, so the logged value IS the header value.
pub(crate) fn clamp_retry_after_s(hint: Option<u64>) -> u64 {
    hint.unwrap_or(RETRY_AFTER_MIN_S * 5)
        .clamp(RETRY_AFTER_MIN_S, RETRY_AFTER_MAX_S)
}

/// How long this request has been memory-deferred, from the stamp the worker latches on the
/// FIRST memory defer (a later stamp would reset the budget on every tick and never bound
/// anything).
pub(crate) fn waited_ms(since: Option<Instant>) -> u64 {
    since.map_or(0, |t| t.elapsed().as_millis() as u64)
}

/// The client sentence for a bounded memory refusal. Stable text, no numbers: the numbers are
/// on the `[admit-mem]` line, keyed by request id.
pub(crate) const MEMORY_REFUSE_MESSAGE: &str = "the box has no free KV for this request's context on either the device or the host tier; \
     retry after the Retry-After delay";

/// Everything one `[admit-mem]` receipt line says.
pub(crate) struct MemoryLine<'a> {
    pub request_id: &'a str,
    pub model: &'a str,
    pub verdict: MemoryVerdict,
    pub prompt_tokens: usize,
    /// The resolved output bound, or `None` for the naked open-output case (rendered `-`).
    pub output_bound: Option<usize>,
    pub estimate: KvEstimate,
    pub tiers: Tiers,
    pub inflight: u64,
    pub cap: u64,
    pub waited_ms: u64,
    /// Only on the refuse arm; `-` elsewhere.
    pub retry_after_s: Option<u64>,
}

/// One grep-stable receipt line, `[admit-mem]`-prefixed, all fields `key=value`. The
/// requalification cell greps `verdict=` and the three byte columns; nothing here is
/// reconstructed from prose.
pub(crate) fn memory_line(line: &MemoryLine<'_>) -> String {
    let short_by = match line.verdict {
        MemoryVerdict::Admit => 0,
        MemoryVerdict::DemoteThenAdmit { demote_bytes } => demote_bytes,
        MemoryVerdict::Defer { short_by } | MemoryVerdict::Refuse { short_by } => short_by,
    };
    format!(
        "[admit-mem] id={} model={:?} verdict={} prompt={} output_bound={} charged_ctx={} \
         est_bytes={} est_context={} est_fixed={} device_free={} host_free={} demotable={} \
         short_by={} inflight={} cap={} waited_ms={} retry_after_s={}",
        line.request_id,
        line.model,
        line.verdict.as_str(),
        line.prompt_tokens,
        line.output_bound.map_or("-".to_string(), |v| v.to_string()),
        line.estimate.charged_ctx,
        line.estimate.total_bytes(),
        line.estimate.context_bytes,
        line.estimate.fixed_bytes,
        line.tiers.device_free_bytes,
        line.tiers.host_free_bytes,
        line.tiers.demotable_device_bytes,
        short_by,
        line.inflight,
        line.cap,
        line.waited_ms,
        line.retry_after_s
            .map_or("-".to_string(), |v| v.to_string()),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The GLM-5.3-Flash B200 mint's own geometry, from the launcher and the byte budget in
    /// darklanes `research/glm5-b200-mint-20260904`: 11 MLA/DSA latent layers, no ring class,
    /// and a per-token charge that puts a full 1,048,576-token session at ~13 GB. Solving
    /// 13 GB / 1,048,576 gives ~13,300 B/token; the rounded 13,312 (13 KiB) is used here so
    /// the arithmetic in each assertion is checkable by hand.
    const GLM_BPT: u64 = 13_312;
    /// The launcher's own figure for a 1M session's latent KV.
    const GLM_1M_BYTES: u64 = GLM_BPT * 1_048_576; // 13.96 GB

    /// A B200 card with the resident experts and the 49,152 MB prefix cache already down.
    fn b200_free(gib: u64) -> u64 {
        gib * (1 << 30)
    }

    /// THE CAP'S ARITHMETIC, and why 4 was the wrong shape rather than the wrong number.
    /// A 900k conversation and a 4k probe do not cost the same, and the door's estimate says
    /// so: at the same bytes/token the probe is 0.4% of the long session.
    #[test]
    fn a_short_probe_is_not_a_long_session() {
        let long = estimate(
            charged_ctx_tokens(900_000, Some(32_768), DEFAULT_OPEN_OUTPUT_TOKENS, 1_048_576),
            GLM_BPT,
            0,
            0,
            155_000_000,
        );
        let probe = estimate(
            charged_ctx_tokens(4_000, Some(512), DEFAULT_OPEN_OUTPUT_TOKENS, 1_048_576),
            GLM_BPT,
            0,
            0,
            155_000_000,
        );
        assert_eq!(long.charged_ctx, 932_776);
        assert_eq!(probe.charged_ctx, 4_520);
        assert_eq!(long.context_bytes, 932_776 * GLM_BPT);
        assert_eq!(probe.context_bytes, 4_520 * GLM_BPT);
        // The slot cap charged these two the same slot. The estimate does not.
        assert!(probe.total_bytes() * 50 < long.total_bytes());
    }

    /// GAP 1: the naked open-output arm. Without the door a request with no `max_tokens` and
    /// no `max_ctx` is charged the whole `MEMRA_CTX` envelope; with it, the open-output
    /// charge. On this route that is 13.96 GB versus 0.11 GB for the same request.
    #[test]
    fn open_output_charges_the_output_not_the_envelope() {
        let envelope = estimate(1_048_576, GLM_BPT, 0, 0, 0);
        assert_eq!(envelope.total_bytes(), GLM_1M_BYTES);
        let charged = charged_ctx_tokens(4_000, None, DEFAULT_OPEN_OUTPUT_TOKENS, 1_048_576);
        assert_eq!(charged, 4_000 + 8_192 + 8);
        let armed = estimate(charged, GLM_BPT, 0, 0, 0);
        // 1,048,576 / 12,200 = 85.9x less KV booked for the same request.
        assert_eq!(envelope.charged_ctx / armed.charged_ctx, 85);
        assert!(armed.total_bytes() * 85 < envelope.total_bytes());
        // A given bound always wins over the open-output default, in both directions.
        assert_eq!(
            charged_ctx_tokens(4_000, Some(512), DEFAULT_OPEN_OUTPUT_TOKENS, 1_048_576),
            4_520
        );
        assert_eq!(
            charged_ctx_tokens(4_000, Some(32_768), DEFAULT_OPEN_OUTPUT_TOKENS, 1_048_576),
            36_776
        );
        // The model's trained length still clamps, exactly as request_ctx_cap clamps today.
        assert_eq!(
            charged_ctx_tokens(
                1_040_000,
                Some(32_768),
                DEFAULT_OPEN_OUTPUT_TOKENS,
                1_048_576
            ),
            1_048_576
        );
        // model_ctx 0 = unknown = no clamp.
        assert_eq!(
            charged_ctx_tokens(1_040_000, Some(32_768), 8_192, 0),
            1_072_776
        );
        // A charge can never land under the prompt's own tokens.
        assert_eq!(charged_ctx_tokens(4_000, Some(0), 8_192, 0), 4_008);
    }

    /// A ring class is capped at its physical rows; the flat class scales with the charge.
    /// (Latent families like this one have no ring, so the two paths must agree there.)
    #[test]
    fn ring_geometry_caps_the_ring_class_only() {
        let flat = estimate(100_000, 1_000, 0, 0, 7);
        assert_eq!(flat.context_bytes, 100_000_000);
        assert_eq!(flat.total_bytes(), 100_000_007);
        // 400 of the 1000 B/token live in a 4096-row ring: 600 B/token flat + 400 x 4096.
        let ringed = estimate(100_000, 1_000, 400, 4_096, 0);
        assert_eq!(ringed.context_bytes, 600 * 100_000 + 400 * 4_096);
        // Under the ring rows the ring term is the charge itself, so the two agree.
        assert_eq!(
            estimate(1_000, 1_000, 400, 4_096, 0).context_bytes,
            1_000_000
        );
    }

    /// THE DECISION RULE, arm by arm, on this box's real magnitudes.
    #[test]
    fn decide_walks_device_then_host_then_time() {
        let need = GLM_1M_BYTES; // one 1M-class session
        // Device alone covers it: nothing is demoted, nothing waits.
        let roomy = Tiers {
            device_free_bytes: b200_free(40),
            demotable_device_bytes: 0,
            host_free_bytes: 0,
        };
        assert_eq!(
            decide(need, &roomy, 0, DEFAULT_DEFER_BUDGET_MS),
            MemoryVerdict::Admit
        );
        // Device is 4 GiB short; the cache holds 50 GB and the host tier has room: demote
        // exactly the shortfall, never the whole cache.
        let short = need - b200_free(10);
        let demotable = Tiers {
            device_free_bytes: b200_free(10),
            demotable_device_bytes: 50_000_000_000,
            host_free_bytes: 50_000_000_000,
        };
        assert_eq!(
            decide(need, &demotable, 0, DEFAULT_DEFER_BUDGET_MS),
            MemoryVerdict::DemoteThenAdmit {
                demote_bytes: short
            }
        );
        assert_eq!(
            decide(need, &demotable, 0, DEFAULT_DEFER_BUDGET_MS).demote_bytes(),
            short
        );
        // The cache holds the bytes but the HOST tier is full: they would have to be DROPPED,
        // so this is not a demotion. Inside the budget it waits.
        let host_full = Tiers {
            host_free_bytes: 0,
            ..demotable
        };
        assert_eq!(
            decide(need, &host_full, 0, DEFAULT_DEFER_BUDGET_MS),
            MemoryVerdict::Defer { short_by: short }
        );
        // Same shape, but the request has now waited out its budget: bounded refusal.
        assert_eq!(
            decide(
                need,
                &host_full,
                DEFAULT_DEFER_BUDGET_MS,
                DEFAULT_DEFER_BUDGET_MS
            ),
            MemoryVerdict::Refuse { short_by: short }
        );
        assert_eq!(
            decide(need, &host_full, 60_000, DEFAULT_DEFER_BUDGET_MS),
            MemoryVerdict::Refuse { short_by: short }
        );
        // budget 0 = the refusal arm is disabled and today's unbounded FIFO defer returns.
        assert_eq!(
            decide(need, &host_full, 600_000, 0),
            MemoryVerdict::Defer { short_by: short }
        );
        // Exactly-fits admits (<=, not <): an off-by-one here refuses a legal full-envelope
        // request on an empty box, the shape the router law already had to fix once.
        let exact = Tiers {
            device_free_bytes: need,
            ..Default::default()
        };
        assert_eq!(
            decide(need, &exact, 0, DEFAULT_DEFER_BUDGET_MS),
            MemoryVerdict::Admit
        );
        let one_short = Tiers {
            device_free_bytes: need - 1,
            ..Default::default()
        };
        assert_eq!(
            decide(need, &one_short, 0, DEFAULT_DEFER_BUDGET_MS),
            MemoryVerdict::Defer { short_by: 1 }
        );
    }

    /// THE MIXED LOAD THE CELL RUNS: 1 x 900k + 8 x 4k on one card. The eight short requests
    /// must admit on device alone while the long one holds ~12 GB, which is the whole point of
    /// replacing the slot cap: at cap 4 five of the nine would have queued.
    #[test]
    fn one_long_and_eight_short_all_admit_on_one_card() {
        let fixed = 155_000_000; // the launcher's own per-entry fixed figure
        let long = estimate(
            charged_ctx_tokens(900_000, Some(32_768), DEFAULT_OPEN_OUTPUT_TOKENS, 1_048_576),
            GLM_BPT,
            0,
            0,
            fixed,
        );
        let short = estimate(
            charged_ctx_tokens(4_000, Some(512), DEFAULT_OPEN_OUTPUT_TOKENS, 1_048_576),
            GLM_BPT,
            0,
            0,
            fixed,
        );
        // A B200 with the 124 GB resident experts down: ~55 GiB of the 183 GB card left.
        let mut free = b200_free(55);
        for (i, est) in std::iter::once(long)
            .chain(std::iter::repeat_n(short, 8))
            .enumerate()
        {
            let tiers = Tiers {
                device_free_bytes: free,
                ..Default::default()
            };
            assert_eq!(
                decide(est.total_bytes(), &tiers, 0, DEFAULT_DEFER_BUDGET_MS),
                MemoryVerdict::Admit,
                "session {i} must admit on device alone"
            );
            free -= est.total_bytes();
        }
        // And there is still room left over, so the ceiling is not the binding term here.
        assert!(free > b200_free(35));
        // Nine of these at the OLD open-output charge (the whole envelope each) would not
        // have fitted: that is the charge the door replaces.
        assert!(9 * GLM_1M_BYTES > b200_free(55));
    }

    /// Retry-After is clamped into the shed contract's window in both directions, and the
    /// default is the class fallback rather than 0 (a `Retry-After: 0` is a retry storm).
    #[test]
    fn retry_after_is_clamped_to_the_shed_window() {
        assert_eq!(clamp_retry_after_s(None), 5);
        assert_eq!(clamp_retry_after_s(Some(0)), RETRY_AFTER_MIN_S);
        assert_eq!(clamp_retry_after_s(Some(7)), 7);
        assert_eq!(clamp_retry_after_s(Some(3_600)), RETRY_AFTER_MAX_S);
        // The refusal sentence carries no numbers: they live on the receipt line.
        assert!(!MEMORY_REFUSE_MESSAGE.chars().any(|c| c.is_ascii_digit()));
    }

    #[test]
    fn waited_ms_is_zero_until_the_first_defer_stamps_it() {
        assert_eq!(waited_ms(None), 0);
        assert!(waited_ms(Some(Instant::now())) < 1_000);
    }

    fn sample_line(verdict: MemoryVerdict) -> MemoryLine<'static> {
        MemoryLine {
            request_id: "chatcmpl-abc123",
            model: "zai/glm-5.3-flash",
            verdict,
            prompt_tokens: 900_000,
            output_bound: Some(32_768),
            estimate: KvEstimate {
                charged_ctx: 932_776,
                context_bytes: 12_417_105_920,
                fixed_bytes: 155_000_000,
            },
            tiers: Tiers {
                device_free_bytes: 10_737_418_240,
                demotable_device_bytes: 50_000_000_000,
                host_free_bytes: 200_000_000_000,
            },
            inflight: 9,
            cap: 32,
            waited_ms: 0,
            retry_after_s: None,
        }
    }

    /// Locks the receipt's field NAMES: the cell greps these, it does not parse prose.
    #[test]
    fn memory_line_locks_fields() {
        let s = memory_line(&sample_line(MemoryVerdict::DemoteThenAdmit {
            demote_bytes: 1_834_872_320,
        }));
        assert!(s.starts_with("[admit-mem] "), "grep-stable prefix: {s}");
        for field in [
            "id=chatcmpl-abc123",
            "model=\"zai/glm-5.3-flash\"",
            "verdict=demote-then-admit",
            "prompt=900000",
            "output_bound=32768",
            "charged_ctx=932776",
            "est_bytes=12572105920",
            "est_context=12417105920",
            "est_fixed=155000000",
            "device_free=10737418240",
            "host_free=200000000000",
            "demotable=50000000000",
            "short_by=1834872320",
            "inflight=9",
            "cap=32",
            "waited_ms=0",
            "retry_after_s=-",
        ] {
            assert!(s.contains(field), "line must carry `{field}`: {s}");
        }
        assert_eq!(s.lines().count(), 1);
    }

    #[test]
    fn memory_line_renders_every_arm_and_its_placeholders() {
        let admit = memory_line(&sample_line(MemoryVerdict::Admit));
        assert!(admit.contains("verdict=admit short_by=0") || admit.contains("verdict=admit"));
        assert!(admit.contains(" short_by=0 "), "{admit}");
        let mut refused = sample_line(MemoryVerdict::Refuse {
            short_by: 4_000_000_000,
        });
        refused.output_bound = None;
        refused.waited_ms = 8_112;
        refused.retry_after_s = Some(12);
        let s = memory_line(&refused);
        for field in [
            "verdict=refuse",
            "output_bound=-",
            "short_by=4000000000",
            "waited_ms=8112",
            "retry_after_s=12",
        ] {
            assert!(s.contains(field), "line must carry `{field}`: {s}");
        }
        let deferred = memory_line(&sample_line(MemoryVerdict::Defer { short_by: 42 }));
        assert!(deferred.contains("verdict=defer"), "{deferred}");
        assert!(deferred.contains(" short_by=42 "), "{deferred}");
    }

    /// The boot line states the door AND its two knobs: a receipt reader must be able to
    /// re-derive the policy from the log alone (the qualification-env law).
    #[test]
    fn boot_line_states_the_door_and_its_knobs() {
        let cfg = MemoryAdmitConfig {
            armed: true,
            open_output_tokens: 8_192,
            defer_budget_ms: 8_000,
        };
        let s = cfg.boot_line();
        assert!(s.contains("door=ON"), "{s}");
        assert!(s.contains("open_output_tokens=8192"), "{s}");
        assert!(s.contains("defer_budget_ms=8000"), "{s}");
        assert_eq!(cfg.open_output_charge(), Some(8_192));
        let off = MemoryAdmitConfig {
            armed: false,
            ..cfg
        };
        assert!(off.boot_line().contains("door=OFF"));
        // OFF hands the caller nothing, so request_ctx_cap keeps today's MEMRA_CTX arm.
        assert_eq!(off.open_output_charge(), None);
    }

    /// The demote budget is the shortfall and nothing more: an unbounded flush would stall
    /// the tick behind gigabytes of synchronous D2H for every other session on the box.
    #[test]
    fn demote_budget_is_the_shortfall_only() {
        assert_eq!(
            demote_budget_bytes(MemoryVerdict::DemoteThenAdmit {
                demote_bytes: 4_294_967_296
            }),
            4_294_967_296
        );
        assert_eq!(demote_budget_bytes(MemoryVerdict::Admit), 0);
        assert_eq!(demote_budget_bytes(MemoryVerdict::Defer { short_by: 9 }), 0);
        assert_eq!(
            demote_budget_bytes(MemoryVerdict::Refuse { short_by: 9 }),
            0
        );
    }
}
