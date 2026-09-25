//! dsv4 serve door (ds4f rung 3) — DeepSeek-V4-Flash serving through the worker's
//! `Cmd::Generate(Request) -> Event` contract, on the engine's own 2-card PP stack
//! (`Dsv4Gpu`), NOT through `HybridModel`.
//!
//! Shape: a dsv4 checkpoint directory (config.json `model_type: "deepseek_v4"`) loads
//! at worker boot onto the 2-card placement and serves on a DEDICATED thread with a
//! FIFO channel — one request at a time (the engine is a bs=1 program today; the
//! channel IS the queue, and concurrency cells measure queueing honestly). The HTTP
//! surface (OpenAI routes, template render inputs, DSML tools parse, streaming SSE)
//! is untouched: this module only implements the worker side of the contract.
//!
//! Routes inside a request, decided by `sampler_cfg` + drafter residency:
//!   T == 0, drafter resident  -> `spec_greedy_batched_stream` (identity-gated driver)
//!   T == 0, trunk only        -> plain greedy decode loop
//!   T  > 0, drafter resident  -> `spec_sampled_batched_stream` (rung-2 gated driver)
//!   T  > 0, trunk only        -> plain sampled loop (position-keyed seeded draws)
//! The confidence window stays behind its env seam (`MEMRA_DSV4_VT`, default off —
//! the flip is the owner's ratification call, one env line when it lands).
//!
//! Penalties are IMPLEMENTED (rung-2 slice 2): Keskar penalties over the true per-state
//! window (whole context when no explicit last_n); penalized GREEDY serves on the PLAIN
//! path (the spec greedy driver argmaxes raw columns), sampled rides the pen driver.
//!
//! Deliberate v1 limitations, stated not smuggled:
//!   - `min_p` REFUSES (not implemented on the dsv4 sampler);
//!   - parked-prefix reuse is opt-in through `MEMRA_DSV4_KV_HOST_MB`: live compact state
//!     moves to pinned host RAM and restores on a strict, PC-ISO exact-token prefix;
//!     active-device scheduling is still FIFO in this slice;
//!   - `MEMRA_DSV4_C4_HOST_MB` independently budgets active C4 history. This
//!     experimental matrix-only path allocates host history directly, including
//!     fresh position zero and restored prefixes; it never stages full GPU C4;
//!   - no response_format/grammar (refused by name);
//!   - streaming granularity is the spec ROUND (or every plain token) — the commit
//!     callback seam on the gated drivers, `None` = byte-identical bench behavior.

use crate::dsv4_admit::{self, Admission, AdmissionLine, MemoryProbe, SessionNeed};
use crate::health::RouteHealth;
use crate::route_telemetry::{RouteLoad, RouteRun, ServeStats};
use crate::worker::{EngineError, Event, EventSender, ModelCaps, Request, SpecUsage};
use memra_engine::dsv4_gpu::{
    DSV4_BATCH_WIDTH_MAX, DecodePath, DecodeState, DsparkState, Dsv4Gpu, Dsv4HostDecodeState,
    Dsv4HostDsparkState, Dsv4PenaltyCfg, Dsv4SampleCfg, Dsv4Vt, REPLAY_MIN_CAPACITY, RoundTake,
    StageMemory, dsv4_penalize_row, dsv4_sample_row, resolve_vt,
};
use memra_engine::dsv4_topology::Dsv4Placement;
use memra_gguf::dsv4_forward::ActQuantVariant;
use memra_tokenizer::{Tokenizer, chat};
use std::collections::HashMap;
use std::ffi::OsString;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

/// Cheap boot-time probe; the engine loader re-validates every field strictly.
pub fn is_dsv4_dir(dir: &Path) -> bool {
    match std::fs::read_to_string(dir.join("config.json")) {
        Ok(s) => s.contains("\"model_type\"") && s.contains("\"deepseek_v4\""),
        Err(_) => false,
    }
}

/// What a checkpoint's `config.json` says about its context, as three distinct facts rather than
/// one number. They are different situations for an operator and they get different refusals:
/// a usable declaration is served, an ABSENT one is the ambiguity that needs an explicit
/// `MEMRA_CTX`, and a PRESENT BUT UNUSABLE one (a string, a float, a negative, a zero) is a
/// malformed artifact that no default should paper over.
#[derive(Debug, PartialEq, Eq)]
enum DeclaredContext {
    /// A usable positive count. Kept as `u64` because `config.json` can carry a value wider than
    /// anything the engine can represent, and that case must reach the ceiling check rather than
    /// be truncated on the way there.
    Positions(u64),
    Absent,
    Unusable(String),
}

/// The checkpoint's OWN declared context, read from `config.json`.
///
/// This is the number the serve route must default to. Before this lane the route defaulted to a
/// literal 8192, allocated `max_seq` from it and published it as `context_length` on
/// `/v1/models`, so a launcher that omitted `MEMRA_CTX` served 8192 against a checkpoint
/// declaring `max_position_embeddings: 1048576` and nothing in the response said so.
///
/// Returns `0` for an ABSENT declaration, which `resolve_ctx` refuses as undeclared; refuses here
/// for an UNUSABLE one, naming the value; and saturates a beyond-`usize` declaration so it lands
/// on the engine-ceiling refusal instead of silently wrapping.
fn dsv4_declared_context(dir: &Path) -> Result<usize, String> {
    let path = dir.join("config.json");
    let text = std::fs::read_to_string(&path)
        .map_err(|e| format!("dsv4 config {} unreadable: {e}", path.display()))?;
    let json: serde_json::Value = serde_json::from_str(&text)
        .map_err(|e| format!("dsv4 config {} is not JSON: {e}", path.display()))?;
    match declared_context_from_config(&json) {
        DeclaredContext::Positions(v) => Ok(usize::try_from(v).unwrap_or(usize::MAX)),
        DeclaredContext::Absent => Ok(0),
        DeclaredContext::Unusable(raw) => Err(format!(
            "dsv4 config {}: max_position_embeddings {raw} is not a positive count of positions; \
             refusing to substitute a context window for a malformed declaration",
            path.display()
        )),
    }
}

/// `max_position_embeddings` classified, never substituted.
fn declared_context_from_config(json: &serde_json::Value) -> DeclaredContext {
    let Some(value) = json.get("max_position_embeddings") else {
        return DeclaredContext::Absent;
    };
    if value.is_null() {
        return DeclaredContext::Absent;
    }
    match value.as_u64() {
        Some(0) | None => DeclaredContext::Unusable(value.to_string()),
        Some(v) => DeclaredContext::Positions(v),
    }
}

pub struct Dsv4Model {
    pub gpu: Arc<Dsv4Gpu>,
    pub tok: Arc<Tokenizer>,
    pub max_seq: usize,
    pub spec: bool,
    pub eos: u32,
    pub host_cache_bytes: usize,
    pub c4_host_bytes: usize,
    pub prefill_chunk: usize,
    /// The route's memory book, calibrated at load (memra#503).
    pub memory: Dsv4Memory,
    /// Serving lanes (`MEMRA_DSV4_SESSIONS`, memra #667).
    pub sessions: usize,
    /// B-row step coalescing across the lanes (`MEMRA_DSV4_ROWS`, memra #667 lever 2).
    pub rows: Option<Arc<RowBatcher>>,
    /// The spec route's verify policy, resolved once at load.
    pub spec_policy: Dsv4SpecPolicy,
}

/// The spec route's verify policy: the per-round depth ceiling (`MEMRA_DSV4_SPEC_DEPTH`,
/// unset = the drafter's own `block_size + 1`) and the confidence window (`MEMRA_DSV4_VT`,
/// `MEMRA_DSV4_VT_TAU`, `MEMRA_DSV4_VT_FLOOR`). Resolved at load, so a bad value refuses the
/// boot instead of failing every spec request with an engine error.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Dsv4SpecPolicy {
    pub depth_cap: usize,
    pub vt: Dsv4Vt,
}

fn resolve_spec_policy(
    depth: Option<&str>,
    vt: Option<&str>,
    tau: Option<&str>,
    floor: Option<&str>,
) -> Result<Dsv4SpecPolicy, String> {
    let depth_cap = match depth {
        None => usize::MAX,
        Some(raw) => match raw.trim().parse::<usize>() {
            Ok(d) if d > 0 => d,
            _ => {
                return Err(format!(
                    "MEMRA_DSV4_SPEC_DEPTH {raw:?} is not a positive integer (unset = the \
                     drafter's own depth)"
                ));
            }
        },
    };
    Ok(Dsv4SpecPolicy {
        depth_cap,
        vt: resolve_vt(vt, tau, floor)?,
    })
}

/// What a session costs on each stage beyond its capacity-planned layer caches, and the most
/// each card offers with the route idle. Measured once at load, by allocating the route's own
/// state around two `stage_memory` readings (memra#503): the step workspace, the matrix
/// width-1 transaction and the transaction scratch are sized by the model, not the session, so
/// one small calibration session prices every larger one.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Dsv4Memory {
    /// Owning device of each stage.
    pub devs: Vec<usize>,
    /// Per-stage device bytes a plain session holds beyond its layer caches.
    pub fixed_plain: Vec<u64>,
    /// The same for a speculative session: the drafter state plus the larger of the two
    /// transactions it holds in turn (the chunked prefill with its tap capture, then verify).
    pub fixed_spec: Vec<u64>,
    /// Effective free per stage after calibration, nothing resident.
    pub ceiling: Vec<u64>,
}

fn env_text(name: &str, raw: Option<OsString>) -> Result<Option<String>, String> {
    match raw {
        None => Ok(None),
        Some(value) => value
            .into_string()
            .map(Some)
            .map_err(|_| format!("{name} is not valid Unicode")),
    }
}

/// `MEMRA_DSV4_SESSIONS` (memra #667): serving lanes that share the route's queue and launch
/// turn. `1` is the serial route; `2..=4` pipeline plain steps across sessions. Unset or empty
/// takes `default`, which the load derives from the program it loaded.
fn resolve_sessions(raw: Option<&str>, default: usize) -> Result<usize, String> {
    match raw.map(str::trim) {
        None | Some("") => Ok(default),
        Some(text) => match text.parse::<usize>() {
            Ok(n @ 1..=4) => Ok(n),
            _ => Err(format!("MEMRA_DSV4_SESSIONS {text:?} must be 1..=4")),
        },
    }
}

/// The lanes a load of this program gets when `MEMRA_DSV4_SESSIONS` is unset: two on the
/// plain PP matrix device program over two or more stages, where two requests' steps overlap
/// on the two cards (+48% aggregate at c2 on 2x RTX PRO 6000,
/// `research/dsv4f-bringup-20260923/pp-pipeline/`); one everywhere else. A DSpark route holds
/// the launch turn for a whole request, so a second lane there would only wait, and it was
/// not measured: it keeps one.
fn default_sessions(pipelined_steps: bool, drafter: bool) -> usize {
    if pipelined_steps && !drafter { 2 } else { 1 }
}

/// The lanes a TP/EP load gets when `MEMRA_DSV4_SESSIONS` is unset (memra #710 B-row): two on
/// the plain route, whose requests then share TP/EP B-row steps; one with a drafter, whose
/// rounds hold the launch turn for a whole request.
fn default_tp_ep_sessions(rows_steps: bool, drafter: bool) -> usize {
    if rows_steps && !drafter { 2 } else { 1 }
}

/// `MEMRA_DSV4_ROWS` (memra #667 lever 2): the most plain rows one step runs across the lanes.
/// `0` or `1` keeps each lane's own step; `2..=8` coalesces them. Unset or empty takes
/// `default`: one on PP-2, the lane count on TP/EP (memra #710), where lanes only help by
/// sharing steps.
fn resolve_rows(raw: Option<&str>, default: usize) -> Result<usize, String> {
    match raw.map(str::trim) {
        None | Some("") => Ok(default),
        Some(text) => match text.parse::<usize>() {
            Ok(0) => Ok(1),
            Ok(n @ 1..=8) => Ok(n),
            _ => Err(format!("MEMRA_DSV4_ROWS {text:?} must be 0..=8")),
        },
    }
}

/// The serving lanes for a load: `MEMRA_DSV4_SESSIONS` when set, else [`default_sessions`].
pub fn sessions_from_env(default: usize) -> Result<usize, String> {
    resolve_sessions(
        configured_env_text("MEMRA_DSV4_SESSIONS")?.as_deref(),
        default,
    )
}

/// `MEMRA_DSV4_TOPOLOGY` (memra #710): where a two-card load places the model. Unset, empty or
/// `tp_ep` is the default since 2026-09-25: every layer on both cards, experts split by id and
/// attention split by head (exact attention TP2). `pp` is the PP-2 rollback. Anything else
/// refuses the boot.
fn resolve_topology(raw: Option<&str>) -> Result<Dsv4Placement, String> {
    match raw.map(str::trim) {
        None | Some("") | Some("tp_ep") => Ok(Dsv4Placement::TpEp { attention_tp: true }),
        Some("pp") => Ok(Dsv4Placement::Pp),
        Some(other) => Err(format!(
            "MEMRA_DSV4_TOPOLOGY {other:?} unknown (tp_ep | pp)"
        )),
    }
}

fn configured_env_text(name: &str) -> Result<Option<String>, String> {
    env_text(name, std::env::var_os(name))
}

fn resolve_env_mb(name: &str, raw: Option<OsString>) -> Result<usize, String> {
    let Some(raw) = env_text(name, raw)? else {
        return Ok(0);
    };
    let mb = raw
        .trim()
        .parse::<usize>()
        .map_err(|_| format!("{name} {raw:?} is not a non-negative integer"))?;
    mb.checked_mul(1024 * 1024)
        .ok_or_else(|| format!("{name} byte count overflow"))
}

fn resolve_c4_host_bytes(raw: Option<&str>) -> Result<usize, String> {
    let mb = match raw {
        None => 0,
        Some(raw) => raw
            .trim()
            .parse::<usize>()
            .map_err(|_| format!("MEMRA_DSV4_C4_HOST_MB {raw:?} is not a non-negative integer"))?,
    };
    mb.checked_mul(1024 * 1024)
        .ok_or_else(|| "MEMRA_DSV4_C4_HOST_MB byte count overflow".to_string())
}

/// This dedicated worker runs exactly one active request. The budget covers that
/// request's full-capacity C4 history, not the separate parked-prefix pool or its
/// snapshot workspace. A future concurrent scheduler must reserve the SUM of its
/// active requests before reusing this allocation path.
fn admit_c4_host_bytes(budget: usize, per_stage: &[u64]) -> Result<usize, String> {
    if budget == 0 {
        return Ok(0);
    }
    let needed = per_stage.iter().try_fold(0usize, |sum, &bytes| {
        let bytes = usize::try_from(bytes).map_err(|_| "active C4 byte count overflow")?;
        sum.checked_add(bytes)
            .ok_or("active C4 byte count overflow")
    })?;
    if needed > budget {
        return Err(format!(
            "active C4 history needs {needed} bytes, exceeding MEMRA_DSV4_C4_HOST_MB \
             budget {budget} bytes; no device-history fallback"
        ));
    }
    Ok(needed)
}

impl Dsv4Model {
    /// Transient rows every session state is allocated with: wide enough for the chunked
    /// prefill and for a verify round.
    fn transient_rows(&self) -> usize {
        self.prefill_chunk.max(self.gpu.verify_tmax())
    }

    /// The device charge of a session at `capacity`, per owning card, plus `host_c4` pinned
    /// host bytes (memra#503). Layer caches come from the allocator's own plan; the fixed term
    /// from the load calibration; the active-C4 gather, grown lazily at a workspace's first
    /// round and so invisible to the calibration, is added for each workspace width the
    /// session runs (decode, prefill chunk, and verify when speculative).
    fn session_need(
        &self,
        capacity: usize,
        spec: bool,
        host_c4: u64,
    ) -> Result<SessionNeed, String> {
        let host_c4_on = self.c4_host_bytes > 0;
        let (mut need, _) =
            self.gpu
                .plan_session_cache_bytes(capacity, self.transient_rows(), host_c4_on)?;
        let mem = &self.memory;
        let fixed = if spec {
            &mem.fixed_spec
        } else {
            &mem.fixed_plain
        };
        if fixed.len() != need.len() || mem.ceiling.len() != need.len() {
            return Err(format!(
                "dsv4 memory book covers {} stages, the session plan {}",
                fixed.len(),
                need.len()
            ));
        }
        for (n, f) in need.iter_mut().zip(fixed) {
            *n = n.saturating_add(*f);
        }
        if host_c4_on {
            let mut widths = vec![1, self.prefill_chunk];
            if spec {
                widths.push(self.gpu.verify_tmax());
            }
            for w in widths.into_iter().filter(|&w| w > 0) {
                for (n, g) in need.iter_mut().zip(self.gpu.c4_gather_bytes_for_width(w)) {
                    *n = n.saturating_add(g);
                }
            }
        }
        Ok(SessionNeed {
            devices: dsv4_admit::per_device(&mem.devs, &need, &mem.ceiling),
            host: host_c4,
        })
    }

    /// Whether the session's charge fits every card with the route idle and its active C4
    /// history fits the host budget: what no wait can change.
    fn fits_ceiling(&self, capacity: usize, spec: bool) -> bool {
        let Ok(host) = self.planned_c4_host_bytes(capacity) else {
            return false;
        };
        self.session_need(capacity, spec, host as u64)
            .is_ok_and(|n| n.devices.iter().all(|d| d.need <= d.ceiling))
    }

    fn planned_c4_host_bytes(&self, capacity: usize) -> Result<usize, String> {
        if self.c4_host_bytes == 0 {
            return Ok(0);
        }
        admit_c4_host_bytes(
            self.c4_host_bytes,
            &self.gpu.c4_host_bytes_for_capacity(capacity)?,
        )
    }

    fn request_state(
        &self,
        capacity: usize,
        host: Option<&Dsv4HostDecodeState>,
    ) -> Result<DecodeState, String> {
        let planned = self.planned_c4_host_bytes(capacity)?;
        let transient = self.transient_rows();
        let state = match (self.c4_host_bytes > 0, host) {
            (true, Some(host)) => self
                .gpu
                .restore_decode_state_host_c4(host, capacity, transient)?,
            (true, None) => self.gpu.alloc_decode_state_host_c4(capacity, transient)?,
            (false, Some(host)) => self
                .gpu
                .restore_decode_state_for_transient(host, capacity, transient)?,
            (false, None) => self
                .gpu
                .alloc_decode_state_for_transient(capacity, transient)?,
        };
        if self.c4_host_bytes > 0 {
            let actual = admit_c4_host_bytes(self.c4_host_bytes, &state.host_cache_bytes)?;
            if actual != planned {
                return Err(format!(
                    "active C4 allocation {actual} bytes differs from reserved {planned} bytes"
                ));
            }
            eprintln!(
                "[dsv4-c4-host] capacity={capacity} restored={} bytes={actual} budget={} \
                 per_stage={:?} device_cache={:?} direct=true",
                host.is_some(),
                self.c4_host_bytes,
                state.host_cache_bytes,
                state.cache_bytes,
            );
        }
        Ok(state)
    }
}

/// The served prefill chunk width.
///
/// Two things moved on 2026-09-11 and they moved together (memra #460, #461).
///
/// The CEILING was a second constant `DSV4_SERVING_BATCH_WIDTH_MAX` (64) while the kernel's own is
/// `DSV4_BATCH_WIDTH_MAX` (512); both constants entered together in `8a52da7c7`
/// (#318) with 512 qualified for the bring-up surface and serving left at 64.
/// #460 proposed a door to raise it. There is no door: the width gain survives
/// at the canonical `EP=off` matrix arm, which is where the earlier +18.1% was
/// NOT measured, so under the owner's door rule it is the naked default. The
/// prod-candidate ABBA (darklanes `receipts/epoffwide-r1`, widths 64/128/256/512
/// forward and 512/64 reverse) reads 173.8 / 195.5 / 213.8 / 227.8 tok/s at a
/// 3,686-token prompt, with the two 512 arms IDENTICAL to 0.0% and the two 64
/// arms within 0.1%: +31.1% at a repeat spread the gain is 300x larger than.
///
/// The DEFAULT moved from 0 (monolithic) to the kernel width for the same
/// reason plus a harder one: the matrix expert program is the loaded default now
/// and it REFUSES a zero chunk, so leaving this at 0 would mean the default
/// server configuration refusing to boot. `0` stays selectable and stays the
/// monolithic rollback, and it is still refused in combination with matrix.
fn resolve_prefill_chunk(raw: Option<&str>, max_seq: usize) -> Result<usize, String> {
    let chunk = match raw {
        None => DSV4_BATCH_WIDTH_MAX.min(max_seq),
        Some(raw) => raw.trim().parse::<usize>().map_err(|_| {
            format!("MEMRA_DSV4_PREFILL_CHUNK {raw:?} is not a non-negative integer")
        })?,
    };
    if chunk > max_seq {
        return Err(format!(
            "MEMRA_DSV4_PREFILL_CHUNK {chunk} exceeds MEMRA_CTX {max_seq}"
        ));
    }
    if chunk > DSV4_BATCH_WIDTH_MAX {
        return Err(format!(
            "MEMRA_DSV4_PREFILL_CHUNK {chunk} exceeds kernel width {DSV4_BATCH_WIDTH_MAX}"
        ));
    }
    Ok(chunk)
}

fn use_chunked_prefill(chunk: usize, prompt_tokens: usize) -> bool {
    chunk > 0 && prompt_tokens > chunk
}

struct ParkedEntry {
    toks: Vec<u32>,
    affinity: Option<String>,
    trunk: Dsv4HostDecodeState,
    dspark: Option<Dsv4HostDsparkState>,
    bytes: usize,
    last_use: Instant,
    id: u64,
}

/// DSV4's dedicated parked-session tier. The model already compresses active KV by 4x/128x;
/// this pool solves a different problem: inactive conversations should not reserve their
/// compact state on both GPUs. Entries are exact-token-prefix keyed inside the request's
/// PC-ISO namespace and consumed on restore, so host and device never hold duplicate warm
/// state past the DMA boundary.
struct Dsv4HostCache {
    entries: HashMap<String, Vec<ParkedEntry>>,
    total_bytes: usize,
    budget: usize,
    next_id: u64,
    disabled: bool,
}

const DSV4_HOST_CACHE_MIN_TOKENS: usize = 128;

/// Why a parked-tier lookup did not produce an entry. Carried so the serve log can say it
/// (memra #495): a silent miss and a slow hit read identically in a receipt otherwise.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TakeMiss {
    /// Nothing has ever parked in this request's PC-ISO namespace.
    EmptyNamespace,
    /// Entries exist and none is a usable strict prefix of this prompt.
    NoPrefix {
        candidates: usize,
        /// Token count of the entry that shares the longest prefix with this prompt.
        best_n: usize,
        /// How far that entry agrees with the prompt before diverging.
        best_lcp: usize,
        /// Entries rejected only because the request needs DSpark state they lack.
        dspark_short: usize,
    },
}

impl TakeMiss {
    fn no_prefix(candidates: usize) -> Self {
        TakeMiss::NoPrefix {
            candidates,
            best_n: 0,
            best_lcp: 0,
            dspark_short: 0,
        }
    }

    fn count(&mut self) {
        if let TakeMiss::NoPrefix { candidates, .. } = self {
            *candidates += 1;
        }
    }

    fn observe(&mut self, n: usize, lcp: usize, dspark_missing: bool) {
        if let TakeMiss::NoPrefix {
            best_n,
            best_lcp,
            dspark_short,
            ..
        } = self
        {
            if lcp > *best_lcp {
                *best_lcp = lcp;
                *best_n = n;
            }
            if dspark_missing {
                *dspark_short += 1;
            }
        }
    }
}

/// How far two token sequences agree.
fn common_prefix_len(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b.iter()).take_while(|(x, y)| x == y).count()
}

/// One parked entry as the selector sees it. Pulled out of `ParkedEntry` so the selection law
/// (and its miss report) is testable without a GPU snapshot.
struct Candidate<'a> {
    toks: &'a [u32],
    has_dspark: bool,
    affinity: Option<&'a str>,
    id: u64,
}

/// Pick the index of the longest usable STRICT token prefix, or say why none was usable.
fn select_prefix<'a>(
    pool: impl Iterator<Item = Candidate<'a>>,
    affinity: Option<&str>,
    prompt: &[u32],
    need_dspark: bool,
) -> Result<usize, TakeMiss> {
    let mut best: Option<(usize, usize, bool, u64)> = None;
    let mut miss = TakeMiss::no_prefix(0);
    for (i, entry) in pool.enumerate() {
        miss.count();
        let n = entry.toks.len();
        let lcp = common_prefix_len(prompt, entry.toks);
        let dspark_missing = need_dspark && !entry.has_dspark;
        let usable = n >= DSV4_HOST_CACHE_MIN_TOKENS && n < prompt.len() && lcp == n;
        if !usable || dspark_missing {
            miss.observe(n, lcp, dspark_missing);
            continue;
        }
        let affinity_match = affinity.is_some() && affinity == entry.affinity;
        let candidate = (i, n, affinity_match, entry.id);
        if best.is_none_or(|(_, bn, ba, bid)| {
            n > bn
                || (n == bn && affinity_match && !ba)
                || (n == bn && affinity_match == ba && entry.id > bid)
        }) {
            best = Some(candidate);
        }
    }
    match best {
        Some((i, _, _, _)) => Ok(i),
        None => Err(miss),
    }
}

impl Dsv4HostCache {
    fn new(budget: usize) -> Self {
        Self {
            entries: HashMap::new(),
            total_bytes: 0,
            budget,
            next_id: 0,
            disabled: false,
        }
    }

    fn armed(&self) -> bool {
        self.budget > 0 && !self.disabled
    }

    fn disable(&mut self, why: &str) {
        if !self.disabled {
            eprintln!(
                "[dsv4-host] TIER DISABLED: {why}. No pageable fallback; lower \
                 MEMRA_DSV4_KV_HOST_MB or fix pinned-memory capacity and restart"
            );
        }
        self.disabled = true;
    }

    /// Consume the longest STRICT exact-token prefix. Strictness guarantees the caller has
    /// at least one suffix token to feed, which reconstructs next-token logits without storing
    /// a 129k-f32 row per entry. Affinity is only a tie-breaker; it never bypasses token match.
    ///
    /// A miss answers WHY (memra #495). The parked tier used to return a bare `None`, so a
    /// conversation whose next turn re-rendered to a DIFFERENT prefix than the one parked was
    /// indistinguishable, in every receipt, from a tier that was never armed: same absent `hit:`
    /// line, same `cached_tokens=0`, same cold prefill. The agent shape sat in exactly that
    /// blind spot. `TakeMiss` carries the longest common prefix of the best candidate, which
    /// names the DIVERGENCE POINT: an lcp equal to the previous turn's prompt length means the
    /// assistant turn was re-rendered differently than it was generated.
    fn take(
        &mut self,
        cache_ns: &str,
        affinity: Option<&str>,
        prompt: &[u32],
        need_dspark: bool,
    ) -> Result<ParkedEntry, TakeMiss> {
        let Some(pool) = self.entries.get(cache_ns) else {
            return Err(TakeMiss::EmptyNamespace);
        };
        let i = select_prefix(
            pool.iter().map(|e| Candidate {
                toks: &e.toks,
                has_dspark: e.dspark.is_some(),
                affinity: e.affinity.as_deref(),
                id: e.id,
            }),
            affinity,
            prompt,
            need_dspark,
        )?;
        let pool = self
            .entries
            .get_mut(cache_ns)
            .expect("pool survived lookup");
        let entry = pool.swap_remove(i);
        self.total_bytes = self.total_bytes.saturating_sub(entry.bytes);
        if pool.is_empty() {
            self.entries.remove(cache_ns);
        }
        Ok(entry)
    }

    fn insert(&mut self, cache_ns: String, mut entry: ParkedEntry) {
        if !self.armed() || entry.bytes > self.budget {
            return;
        }
        // Exact state supersedes an older identical boundary in the same namespace.
        if let Some(pool) = self.entries.get_mut(&cache_ns)
            && let Some(i) = pool.iter().position(|e| e.toks == entry.toks)
        {
            let old = pool.swap_remove(i);
            self.total_bytes = self.total_bytes.saturating_sub(old.bytes);
        }
        entry.id = self.next_id;
        self.next_id += 1;
        entry.last_use = Instant::now();
        self.total_bytes += entry.bytes;
        self.entries.entry(cache_ns).or_default().push(entry);
        while self.total_bytes > self.budget {
            if self.evict_lru(None, "LRU").is_none() {
                break;
            }
        }
    }

    /// The entry `take` would restore for this request, by id, without consuming it: the one
    /// parked entry a memory reclaim for this request must spare.
    fn restore_candidate(
        &self,
        cache_ns: &str,
        affinity: Option<&str>,
        prompt: &[u32],
        need_dspark: bool,
    ) -> Option<u64> {
        if !self.armed() {
            return None;
        }
        let pool = self.entries.get(cache_ns)?;
        let i = select_prefix(
            pool.iter().map(|e| Candidate {
                toks: &e.toks,
                has_dspark: e.dspark.is_some(),
                affinity: e.affinity.as_deref(),
                id: e.id,
            }),
            affinity,
            prompt,
            need_dspark,
        )
        .ok()?;
        Some(pool[i].id)
    }

    /// Parked bytes an eviction sparing `spare` can return.
    fn reclaimable(&self, spare: Option<u64>) -> usize {
        self.entries
            .values()
            .flatten()
            .filter(|e| Some(e.id) != spare)
            .map(|e| e.bytes)
            .sum()
    }

    /// Evict least-recently-used entries, sparing `spare`, until at least `bytes` are freed or
    /// nothing else is evictable (memra#503: the host tier's yield to a memory admission).
    fn reclaim(&mut self, bytes: usize, spare: Option<u64>) -> usize {
        let mut got = 0usize;
        while got < bytes {
            match self.evict_lru(spare, "for admission") {
                Some(freed) => got += freed,
                None => break,
            }
        }
        got
    }

    /// Evict the least-recently-used entry other than `spare`, returning its bytes.
    fn evict_lru(&mut self, spare: Option<u64>, why: &str) -> Option<usize> {
        let (ns, i) = lru_victim(
            self.entries.iter().flat_map(|(ns, pool)| {
                pool.iter()
                    .enumerate()
                    .map(move |(i, e)| (ns.as_str(), i, e.last_use, e.id))
            }),
            spare,
        )?;
        let pool = self.entries.get_mut(&ns).expect("victim pool exists");
        let dead = pool.swap_remove(i);
        self.total_bytes = self.total_bytes.saturating_sub(dead.bytes);
        if pool.is_empty() {
            self.entries.remove(&ns);
        }
        eprintln!(
            "[dsv4-host] evict {why}: {} tokens, {:.1} MiB (resident {:.1}/{:.1} MiB)",
            dead.toks.len(),
            dead.bytes as f64 / 1048576.0,
            self.total_bytes as f64 / 1048576.0,
            self.budget as f64 / 1048576.0,
        );
        Some(dead.bytes)
    }
}

/// The least-recently-used `(namespace, index)` among `(namespace, index, last_use, id)`
/// entries, oldest use first and the lower id on a tie, never `spare`.
fn lru_victim<'a>(
    entries: impl Iterator<Item = (&'a str, usize, Instant, u64)>,
    spare: Option<u64>,
) -> Option<(String, usize)> {
    entries
        .filter(|&(_, _, _, id)| Some(id) != spare)
        .min_by_key(|&(_, _, last_use, id)| (last_use, id))
        .map(|(ns, i, _, _)| (ns.to_string(), i))
}

/// The live readings behind the route's memory admission (memra#503).
struct LiveProbe<'a, 'b> {
    gpu: &'a Dsv4Gpu,
    /// The launch turn, which owns the parked-prefix cache. The door holds it for every reading
    /// and gives it up only while it sleeps, so a session already in flight on another lane
    /// can keep stepping (and finish, freeing the memory this request waits for).
    turn: &'a mut Turn<'b>,
    /// The parked entry this request would restore from; never evicted for its admission.
    spare: Option<u64>,
    tx: &'a EventSender,
    t0: Instant,
}

impl MemoryProbe for LiveProbe<'_, '_> {
    fn device_free(&mut self, devs: &[usize]) -> Result<Vec<u64>, String> {
        let stages = self.gpu.stage_memory()?;
        devs.iter()
            .map(|&dev| {
                stages
                    .iter()
                    .find(|s| s.dev == dev)
                    .map(StageMemory::effective_free)
                    .ok_or_else(|| format!("dsv4 memory probe: no stage on device {dev}"))
            })
            .collect()
    }

    fn host(&mut self) -> Option<(u64, u64)> {
        let meminfo = std::fs::read_to_string("/proc/meminfo").ok()?;
        let available = crate::worker::meminfo_available_bytes(&meminfo)?;
        Some((
            available as u64,
            self.turn.cache().reclaimable(self.spare) as u64,
        ))
    }

    fn reclaim_host(&mut self, bytes: u64) -> u64 {
        self.turn
            .cache()
            .reclaim(usize::try_from(bytes).unwrap_or(usize::MAX), self.spare) as u64
    }

    fn cancelled(&self) -> bool {
        self.tx.is_closed()
    }

    fn elapsed_ms(&self) -> u64 {
        self.t0.elapsed().as_millis() as u64
    }

    fn wait(&mut self, ms: u64) {
        sleep_without_turn(self.turn, std::time::Duration::from_millis(ms));
    }
}

/// Load the 2-card dsv4 stack. The serving numeric contract is keyed on the
/// artifact's OWN encoding revision (the config `dspark_*` census the tokenizer
/// already performed): 0731-class ships the reference kernel law => RefFp8Round;
/// the nvidia preview ships clamp-only => ClampOnly; an undetectable revision
/// REFUSES — a numeric contract is never guessed (lane-2/3 law).
pub fn load(name: &str, dir: &Path, tok: Arc<Tokenizer>) -> Result<Dsv4Model, String> {
    let devices: Vec<usize> = match std::env::var("MEMRA_DSV4_DEVICES") {
        Err(_) => vec![0, 1],
        Ok(s) => s
            .split(',')
            .map(|x| {
                x.trim()
                    .parse()
                    .map_err(|_| format!("MEMRA_DSV4_DEVICES '{s}' unparseable"))
            })
            .collect::<Result<_, _>>()?,
    };
    let variant = match tok.dsv4_encoding() {
        Some(chat::Dsv4Encoding::V0731) => ActQuantVariant::RefFp8Round,
        Some(chat::Dsv4Encoding::Preview) => ActQuantVariant::ClampOnly,
        None => {
            return Err(format!(
                "dsv4 model {name:?}: encoding revision undetectable from config.json \
                 dspark_* census — cannot key the act-quant contract (REF vs clamp-only); \
                 refusing to guess"
            ));
        }
    };
    let model_ctx = dsv4_declared_context(dir)?;
    let max_seq = crate::worker::resolve_env_ctx(model_ctx).map_err(|e| {
        format!("dsv4 model {name:?}: {e} (config.json declares {model_ctx} positions)")
    })?;
    let host_cache_bytes = resolve_env_mb(
        "MEMRA_DSV4_KV_HOST_MB",
        std::env::var_os("MEMRA_DSV4_KV_HOST_MB"),
    )?;
    let host_cache_mb = host_cache_bytes / (1024 * 1024);
    let spec_policy = resolve_spec_policy(
        configured_env_text("MEMRA_DSV4_SPEC_DEPTH")?.as_deref(),
        configured_env_text("MEMRA_DSV4_VT")?.as_deref(),
        configured_env_text("MEMRA_DSV4_VT_TAU")?.as_deref(),
        configured_env_text("MEMRA_DSV4_VT_FLOOR")?.as_deref(),
    )
    .map_err(|e| format!("dsv4 model {name:?}: {e}"))?;
    let prefill_chunk_raw = configured_env_text("MEMRA_DSV4_PREFILL_CHUNK")?;
    let prefill_chunk = resolve_prefill_chunk(prefill_chunk_raw.as_deref(), max_seq)?;
    let c4_host_raw = configured_env_text("MEMRA_DSV4_C4_HOST_MB")?;
    let c4_host_bytes = resolve_c4_host_bytes(c4_host_raw.as_deref())?;
    if c4_host_bytes > 0 && prefill_chunk == 0 {
        return Err("MEMRA_DSV4_C4_HOST_MB requires nonzero MEMRA_DSV4_PREFILL_CHUNK".into());
    }
    let topology_raw = configured_env_text("MEMRA_DSV4_TOPOLOGY")?;
    let placement = resolve_topology(topology_raw.as_deref())?;
    let gpu = Dsv4Gpu::load_placed(dir, &devices, variant, max_seq, placement).map_err(|e| {
        if placement.is_tp_ep() {
            format!("{e} (TP/EP is the default placement; MEMRA_DSV4_TOPOLOGY=pp loads PP-2)")
        } else {
            e
        }
    })?;
    eprintln!(
        "[dsv4-serve] {name}: placement {}{}, plain sampler {:?}",
        if placement.is_tp_ep() {
            "TP/EP (expert-ID EP pair, attention TP2)"
        } else {
            "PP-2"
        },
        if topology_raw.is_some() {
            " (MEMRA_DSV4_TOPOLOGY)"
        } else {
            " (default)"
        },
        gpu.plain_sampler(),
    );
    if gpu.matrix_moe_enabled() && prefill_chunk == 0 {
        return Err("experimental matrix program requires nonzero MEMRA_DSV4_PREFILL_CHUNK".into());
    }
    if c4_host_bytes > 0 {
        if !gpu.matrix_moe_enabled() {
            return Err(
                "MEMRA_DSV4_C4_HOST_MB fresh-state path requires the matrix program".into(),
            );
        }
        // Validate the structural host-C4 contract at boot without allocating a
        // max-context state. Per-request capacity is budgeted before consuming a hit.
        let _ = gpu.c4_host_bytes_for_capacity(max_seq)?;
    }
    let spec = gpu.dspark.is_some();
    let eos = tok.eos_id();
    eprintln!(
        "[dsv4-serve] {name}: loaded on devices {devices:?}, contract {variant:?}, \
         max_seq {max_seq}, drafter {}, parked-host-cache {}, chunked-prefill {}, \
         active-C4-host-budget {c4_host_bytes} bytes (0=OFF; separate from parked cache)",
        if spec {
            "RESIDENT (spec route armed)"
        } else {
            "absent (plain only)"
        },
        if host_cache_bytes == 0 {
            "OFF (MEMRA_DSV4_KV_HOST_MB=0)".to_string()
        } else {
            format!("{host_cache_mb} MiB")
        },
        if prefill_chunk == 0 {
            "OFF (MEMRA_DSV4_PREFILL_CHUNK=0)".to_string()
        } else {
            format!("{prefill_chunk} tokens")
        },
    );
    let pipelined = gpu.pipelined_steps_supported();
    let tp_ep_rows = gpu.topology().is_tp_ep() && gpu.rows_steps_supported();
    let sessions = sessions_from_env(if gpu.topology().is_tp_ep() {
        default_tp_ep_sessions(tp_ep_rows, spec)
    } else {
        default_sessions(pipelined, spec)
    })?;
    if sessions > 1 && !pipelined && !tp_ep_rows {
        return Err(format!(
            "MEMRA_DSV4_SESSIONS={sessions} needs the PP matrix device program (pipelined steps) or \
             the TP/EP attention-TP2 program (B-row steps); this load runs another"
        ));
    }
    let rows = resolve_rows(
        configured_env_text("MEMRA_DSV4_ROWS")?.as_deref(),
        if tp_ep_rows { sessions } else { 1 },
    )?;
    if rows > 1 && (!gpu.rows_steps_supported() || rows > sessions) {
        return Err(format!(
            "MEMRA_DSV4_ROWS={rows} coalesces the lanes' plain steps: it needs the PP matrix or \
             TP/EP attention-TP2 device program and at least {rows} serving lanes \
             (MEMRA_DSV4_SESSIONS={sessions})"
        ));
    }
    if sessions > 1 && tp_ep_rows && rows < sessions {
        return Err(format!(
            "MEMRA_DSV4_SESSIONS={sessions} on TP/EP needs MEMRA_DSV4_ROWS >= {sessions}: its lanes \
             only help by sharing B-row steps (MEMRA_DSV4_ROWS={rows})"
        ));
    }
    // The B-row workspace is the route's, allocated before calibration so the memory book's
    // idle readings already exclude it.
    let row_batcher = if rows > 1 {
        // Two groups in flight let a PP pair overlap them; a TP/EP step uses both cards.
        let groups = if 2 * rows <= sessions && pipelined {
            2
        } else {
            1
        };
        let ws = (0..groups)
            .map(|_| gpu.alloc_rows_state(rows))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| format!("MEMRA_DSV4_ROWS={rows} workspace: {e}"))?;
        eprintln!(
            "[dsv4-serve] {name}: B-row steps up to {rows} rows, {groups} group(s) in flight (MEMRA_DSV4_ROWS)"
        );
        let sampler = if gpu.plain_sampler() == memra_engine::dsv4_sampler::Dsv4Sampler::Device {
            Some(
                gpu.device_sampler()
                    .map_err(|e| format!("B-row device sampler: {e}"))?,
            )
        } else {
            None
        };
        Some(Arc::new(RowBatcher::new(rows, ws, sampler)))
    } else {
        None
    };
    eprintln!(
        "[dsv4-serve] {name}: {sessions} serving lane(s){}",
        if std::env::var_os("MEMRA_DSV4_SESSIONS").is_some() {
            " (MEMRA_DSV4_SESSIONS)"
        } else if sessions > 1 && tp_ep_rows {
            " (default on the plain TP/EP program, sharing B-row steps; MEMRA_DSV4_SESSIONS=1 is the serial route)"
        } else if sessions > 1 {
            " (default on the plain PP matrix program; MEMRA_DSV4_SESSIONS=1 is the serial route)"
        } else {
            " (default)"
        }
    );
    let mut m = Dsv4Model {
        gpu: Arc::new(gpu),
        tok,
        max_seq,
        spec,
        eos,
        host_cache_bytes,
        c4_host_bytes,
        prefill_chunk,
        memory: Dsv4Memory::default(),
        sessions,
        rows: row_batcher,
        spec_policy,
    };
    m.memory = calibrate_memory(&m).map_err(|e| format!("dsv4 memory calibration: {e}"))?;
    let per_stage = |v: &[u64]| {
        m.memory
            .devs
            .iter()
            .zip(v)
            .map(|(d, b)| format!("{d}:{b}"))
            .collect::<Vec<_>>()
            .join(",")
    };
    eprintln!(
        "[admit-mem] route=dsv4-thread model={name:?} calibrated fixed_plain={} fixed_spec={} \
         ceiling={} defer_budget_ms={}",
        per_stage(&m.memory.fixed_plain),
        per_stage(&m.memory.fixed_spec),
        per_stage(&m.memory.ceiling),
        route_defer_budget_ms(),
    );
    Ok(m)
}

/// Tokens of the calibration session: small, so it costs the boot nothing it cannot return.
const CALIBRATION_CAPACITY: usize = 1024;

/// Per-stage bytes the readings `now` hold beyond `base`, less the layer caches the plan
/// already prices. A reading is device-wide, so stages sharing a card share one delta: the
/// card's first stage (its `carrier`) holds it, net of every co-located stage's cache, and
/// the others hold 0, so `dsv4_admit::per_device` sums the card back to one delta.
fn fixed_delta(now: &[StageMemory], base: &[StageMemory], cache: &[u64]) -> Vec<u64> {
    let mut out = vec![0; now.len()];
    for (i, (n, b)) in now.iter().zip(base).enumerate() {
        if carrier(now, i) != i {
            continue;
        }
        let card_cache = now
            .iter()
            .zip(cache)
            .filter(|(s, _)| s.dev == n.dev)
            .fold(0u64, |acc, (_, c)| acc.saturating_add(*c));
        out[i] = n
            .occupied()
            .saturating_sub(b.occupied())
            .saturating_sub(card_cache);
    }
    out
}

/// The first stage on stage `i`'s card: the one whose fixed term carries the card's delta.
fn carrier(stages: &[StageMemory], i: usize) -> usize {
    stages
        .iter()
        .position(|s| s.dev == stages[i].dev)
        .unwrap_or(i)
}

/// Measure the route's fixed per-session cost and its idle ceiling (memra#503). Allocates, in
/// the order a request does, a calibration session, the chunked-prefill transaction and (with
/// a drafter) the drafter state and the verify transaction, reading every stage around them.
/// Everything is dropped before the ceiling is read, so the ceiling is what a request can
/// claim with the route idle and co-tenants as they were at boot.
fn calibrate_memory(m: &Dsv4Model) -> Result<Dsv4Memory, String> {
    let gpu = &m.gpu;
    let cap = m.max_seq.min(CALIBRATION_CAPACITY);
    let base = gpu.stage_memory()?;
    let devs: Vec<usize> = base.iter().map(|s| s.dev).collect();
    let (cache, _) = gpu.plan_session_cache_bytes(cap, m.transient_rows(), m.c4_host_bytes > 0)?;
    // Chunked prefill runs batched transactions on the device path, PP or TP/EP; the legacy
    // path has no transaction.
    let transactions = m.prefill_chunk > 0 && matches!(gpu.decode_path, DecodePath::Device { .. });
    let state = m.request_state(cap, None)?;
    let prefill = if transactions {
        Some(gpu.alloc_prefill_state_for(cap, m.prefill_chunk)?)
    } else {
        None
    };
    let fixed_plain = fixed_delta(&gpu.stage_memory()?, &base, &cache);
    let fixed_spec = if m.spec {
        let dstate = gpu.dspark_alloc_state()?;
        let mut with_prefill = fixed_delta(&gpu.stage_memory()?, &base, &cache);
        if transactions && !base.is_empty() {
            // the continuation's tap capture, allocated inside the chunk loop on the last
            // stage's card, booked on that card's carrier with the rest of its delta
            let at = carrier(&base, base.len() - 1);
            with_prefill[at] =
                with_prefill[at].saturating_add(gpu.dspark_prefill_tap_bytes(m.prefill_chunk));
        }
        drop(prefill);
        let verify = gpu.alloc_verify_state_for(cap)?;
        let with_verify = fixed_delta(&gpu.stage_memory()?, &base, &cache);
        drop(verify);
        drop(dstate);
        with_prefill
            .iter()
            .zip(&with_verify)
            .map(|(p, v)| *p.max(v))
            .collect()
    } else {
        drop(prefill);
        fixed_plain.clone()
    };
    drop(state);
    let ceiling = gpu
        .stage_memory()?
        .iter()
        .map(StageMemory::effective_free)
        .collect();
    Ok(Dsv4Memory {
        devs,
        fixed_plain,
        fixed_spec,
        ceiling,
    })
}

/// The defer budget this route spends (memra#503): `MEMRA_ADMIT_DEFER_BUDGET_MS`, clamped
/// under the stall bound.
fn route_defer_budget_ms() -> u64 {
    dsv4_admit::defer_budget_ms(
        crate::admit_memory::MemoryAdmitConfig::from_env().defer_budget_ms,
        crate::health::stall_threshold_ms(),
    )
}

/// The route's policy contract (memra#504): declared in `route_contract.rs`, not here, so the
/// registry's wiring test greps THIS file for the call sites the declarations name without the
/// declarations themselves satisfying it.
///
/// `sessions` is the loaded route's lane count (`Dsv4Model::sessions`). Before the load only
/// the environment is known, and the pre-load registry checks armed policies, not capacity.
pub fn contract(model: &str, sessions: usize) -> crate::route_contract::RouteContract {
    crate::route_contract::RouteContract::dsv4_thread_sessions(model, sessions)
}

/// ModelCaps for the /v1/models surface + the HTTP layer's gates — the same
/// template-string laws the hybrid caps block applies (shared functions, not copies).
pub fn caps(m: &Dsv4Model) -> ModelCaps {
    let t = m.tok.chat_template();
    // The real dsv4 artifacts carry their dialect as encoding CODE, not a template
    // string (rung-3 serve finding) — the detected encoding revision is the caps truth.
    let enc = m.tok.dsv4_encoding().is_some();
    ModelCaps {
        hy3: false,
        tools_branch: enc || t.is_some_and(chat::template_has_tools_branch),
        qwen_think: t.is_some_and(|t| t.contains("<think>") && t.contains("add_generation_prompt")),
        think_switch: t.is_some_and(|t| t.contains("enable_thinking")),
        chat_ok: enc || t.is_some(),
        context_length: m.max_seq,
        tokenizer: m.tok.pre().to_string(),
        instruct_type: (enc || t.is_some_and(chat::template_is_dsv4))
            .then(|| "deepseek".to_string()),
        effort_levels: t.is_some_and(|t| t.contains("reasoning_effort is defined")),
        qwen_effort: t.is_some_and(chat::template_has_qwen_effort),
        gemma_think: false,
        dsv4: enc || t.is_some_and(chat::template_is_dsv4),
        // A dsv4 artifact is never a glm5 one (disjoint dialect markers); stated rather than
        // left to `..Default::default()`, which this literal does not use.
        glm5: false,
        chat_temperature_default: None,
        chat_top_p_default: None,
        n_vocab: m.tok.vocab_size(),
        // dsv4 is NOT a post-think model: its renderer honours NoThink through the
        // encoding's own `chat` thinking mode, so constrained requests keep that path.
        think_close: Vec::new(),
    }
}

/// The route's two books, borrowed by one request's `Emit`: health gets a round stamp, load a
/// round-time sample, at every committed decode step or speculative round.
struct RouteProgress {
    health: Arc<RouteHealth>,
    load: Arc<RouteLoad>,
    last_round: Instant,
}

impl RouteProgress {
    fn round(&mut self) {
        let now = Instant::now();
        self.health.note_round();
        self.load
            .note_round(now.duration_since(self.last_round).as_millis() as u64);
        self.last_round = now;
    }
}

/// Latches the route DEAD when the serving thread ends for any reason: the channel closed, a
/// panic outside the per-request catch, or a respawn dropping the old sender. The record is the
/// one registered for THIS thread, so a late latch cannot mark a successor dead.
struct ExitLatch(Arc<RouteHealth>);

impl Drop for ExitLatch {
    fn drop(&mut self) {
        self.0.mark_dead(if std::thread::panicking() {
            "dsv4 serving thread panicked"
        } else {
            "dsv4 serving thread exited"
        });
    }
}

/// Answer a request with an error. Constraint-carrying requests wait on the constraint_ready
/// channel, not the event stream: failing only via tx leaves the HTTP layer to a 503 compile
/// timeout (rung-3 serve finding: response_format 503 instead of the named 400). Same
/// dispatch as the worker's fail_request.
fn fail_request(req: &mut Request, err: EngineError) {
    if let Some(ready) = req.constraint_ready.take() {
        let _ = ready.send(Err(err));
    } else {
        let _ = req.tx.send(Event::Error(err));
    }
}

/// Spawn the serving thread; the returned Sender is the model's admission queue.
///
/// `health` is this thread's own liveness record (memra#500) and `load` its route book
/// (memra#501): the thread publishes idle before every wait, busy at every dequeue, rows at every
/// completed prime chunk (its progress sink) and a stamp at every decode step or spec round, so
/// the central worker's idle beat can neither mask nor be masked by this route.
pub fn spawn(
    name: String,
    m: Dsv4Model,
    health: Arc<RouteHealth>,
    load: Arc<RouteLoad>,
) -> std::sync::mpsc::Sender<Box<Request>> {
    let (tx, rx) = std::sync::mpsc::channel::<Box<Request>>();
    let lanes = m.sessions.max(1);
    let rx = Arc::new(std::sync::Mutex::new(rx));
    let turn = Arc::new(TurnLock::new(Dsv4HostCache::new(m.host_cache_bytes)));
    let m = Arc::new(m);
    for lane in 0..lanes {
        let (m, rx, turn, health, load) = (
            m.clone(),
            rx.clone(),
            turn.clone(),
            health.clone(),
            load.clone(),
        );
        std::thread::Builder::new()
            .name(if lanes == 1 {
                format!("dsv4-serve-{name}")
            } else {
                format!("dsv4-serve-{name}-{lane}")
            })
            .spawn(move || serve_lane(&m, &rx, &turn, &health, &load, lanes))
            .expect("spawn dsv4 serve thread");
    }
    tx
}

/// One serving lane. With one lane this is the serial route. With several (memra #667) the
/// lanes share the queue and the launch turn: every engine call runs holding the turn, and a
/// pipelined greedy step gives it up only while it waits for its own readbacks, so another
/// lane's step is queued whole on the stage streams behind it.
fn serve_lane(
    m: &Dsv4Model,
    rx: &std::sync::Mutex<std::sync::mpsc::Receiver<Box<Request>>>,
    turn_lock: &TurnLock,
    health: &Arc<RouteHealth>,
    load: &Arc<RouteLoad>,
    lanes: usize,
) {
    let _latch = ExitLatch(health.clone());
    let sink = health.clone();
    let _progress = memra_engine::progress::ProgressSinkScope::install(Box::new(move |rows| {
        sink.note_rows(rows)
    }));
    loop {
        if lanes == 1 {
            health.set_idle();
        } else {
            health.set_idle_if_free();
        }
        let next = match rx.lock() {
            Ok(queue) => queue.recv(),
            Err(poisoned) => poisoned.into_inner().recv(),
        };
        let Ok(mut req) = next else { break };
        // Occupancy rises before the ticket falls (`RouteLoad::begin`), so an arrival
        // never reads a free route between the two.
        let mut run = load.begin();
        // The worker's DSV4 channel is unbounded, so the hard admission reservation
        // remains held until this serving thread actually receives the request. Merely
        // forwarding it from the command channel must not make the queue appear empty.
        crate::worker::release_request_reservation(&mut req);
        if req.tx.is_closed() {
            continue; // client gone while queued
        }
        run.admit();
        health.begin_request();
        let mut progress = RouteProgress {
            health: health.clone(),
            load: load.clone(),
            last_round: Instant::now(),
        };
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut turn = Turn::take(turn_lock);
            serve_one(m, &mut turn, &mut req, Some(&mut progress))
        }));
        match r {
            Ok(served) => settle(run, &mut req, served),
            Err(payload) => {
                health.note_request_fault();
                let why = payload
                    .downcast_ref::<String>()
                    .cloned()
                    .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
                    .unwrap_or_else(|| "non-string panic payload".into());
                let _ = req.tx.send(Event::Error(EngineError::engine(format!(
                    "dsv4 generation panicked: {why}"
                ))));
            }
        }
        health.end_request();
    }
}

/// The route's launch turn (memra #667): a FIFO ticket lock beside the parked-prefix cache it
/// guards. A session holds it for every engine call, so one session's work is queued whole on
/// the stage streams before another's. A pipelined plain step gives it up while it waits for its
/// own readbacks, a chunked plain prefill between chunks, and the memory door while it sleeps.
/// Tickets are served in arrival order, so a session that gives the turn up and asks again at
/// once queues behind every lane already waiting; a plain mutex would hand it straight back
/// and starve the waiter for a whole prompt. With one serving lane nobody else ever asks for it.
struct TurnLock {
    /// (next ticket, ticket now served).
    tickets: std::sync::Mutex<(u64, u64)>,
    cv: std::sync::Condvar,
    cache: std::sync::Mutex<Dsv4HostCache>,
}

impl TurnLock {
    fn new(cache: Dsv4HostCache) -> Self {
        TurnLock {
            tickets: std::sync::Mutex::new((0, 0)),
            cv: std::sync::Condvar::new(),
            cache: std::sync::Mutex::new(cache),
        }
    }

    fn lock(&self) {
        let mut t = self.tickets.lock().unwrap_or_else(|p| p.into_inner());
        let mine = t.0;
        t.0 += 1;
        while t.1 != mine {
            t = self.cv.wait(t).unwrap_or_else(|p| p.into_inner());
        }
    }

    fn unlock(&self) {
        let mut t = self.tickets.lock().unwrap_or_else(|p| p.into_inner());
        t.1 += 1;
        drop(t);
        self.cv.notify_all();
    }
}

/// One session's hold on the [`TurnLock`]. Dropping it (including by unwinding) gives the turn
/// up, so a panicking request cannot wedge the other lanes.
struct Turn<'a> {
    lock: &'a TurnLock,
    held: bool,
    cache: Option<std::sync::MutexGuard<'a, Dsv4HostCache>>,
}

impl<'a> Turn<'a> {
    fn take(lock: &'a TurnLock) -> Self {
        let mut turn = Turn {
            lock,
            held: false,
            cache: None,
        };
        turn.acquire();
        turn
    }

    fn acquire(&mut self) {
        if !self.held {
            self.lock.lock();
            self.held = true;
        }
    }

    fn release(&mut self) {
        self.cache = None;
        if self.held {
            self.held = false;
            self.lock.unlock();
        }
    }

    fn cache(&mut self) -> &mut Dsv4HostCache {
        self.acquire();
        let lock = self.lock;
        self.cache
            .get_or_insert_with(|| lock.cache.lock().unwrap_or_else(|p| p.into_inner()))
    }
}

impl Drop for Turn<'_> {
    fn drop(&mut self) {
        self.release();
    }
}

/// Sleep with the turn given up, then queue for it again (the memory door's defer wait).
fn sleep_without_turn(turn: &mut Turn, d: std::time::Duration) {
    turn.release();
    std::thread::sleep(d);
    turn.acquire();
}

/// Step coalescing core (memra #667 lever 2): lanes deposit one row each; a lane whose deposit
/// completes the batch, or that has waited out [`ROW_BATCH_WAIT`], leads. The leader takes up
/// to `bmax` deposits including its own, runs them, and publishes every ticket's result. A
/// waiting lane holds no launch turn, so the leader can always take it. Generic over the row
/// payload so the coordination is testable without a GPU.
struct Coalescer<S, A = bool> {
    bmax: usize,
    /// Batches that can be in flight at once (the B-row workspaces). A batch targets
    /// `ceil(members / groups)` rows, so the lanes split into that many groups: with two
    /// groups, two concurrent requests run as two pipelined one-row steps (the two cards
    /// overlap) and four run as two pipelined groups of two.
    groups: usize,
    inner: std::sync::Mutex<CoalesceState<S, A>>,
    cv: std::sync::Condvar,
}

struct CoalesceState<S, A> {
    /// Lanes inside a coalesced decode loop; a batch is full when every one of them that is not
    /// already riding a batch in flight has deposited.
    members: usize,
    /// Rows taken by batches that have not published yet.
    in_flight: usize,
    next: u64,
    waiting: Vec<(u64, u32, A, Lent<S>)>,
    done: std::collections::HashMap<u64, Result<RowOut, String>>,
}

/// One row's result: its next token (the device argmax for a greedy row) and, when the row
/// asked for it, its full logits row for the host sampler or a penalty.
#[derive(Clone, Debug, PartialEq)]
struct RowOut {
    tok: u32,
    logits: Option<Vec<f32>>,
}

/// A lane's `&mut S`, lent to the batch leader for one step.
struct Lent<S>(*mut S);

// SAFETY: the lending lane blocks in `Coalescer::step` until the leader publishes its ticket,
// and the leader dereferences the pointer only between taking the deposit and publishing, so
// the two never touch the state at once. Engine calls bind their stage contexts on the calling
// thread, and the launch turn serializes GPU work across lanes.
unsafe impl<S> Send for Lent<S> {}

/// Runs one batch: the rows' tokens, what each asks for (its logits, or a device draw), and
/// their states in deposit order; returns each row's result in the same order.
type BatchRun<'r, S, A> =
    dyn FnMut(&[u32], &[A], &mut [&mut S]) -> Result<Vec<RowOut>, String> + 'r;

/// How long a partial batch waits for the remaining lanes before it runs anyway.
const ROW_BATCH_WAIT: std::time::Duration = std::time::Duration::from_micros(500);

impl<S, A: Copy> Coalescer<S, A> {
    fn new(bmax: usize, groups: usize) -> Self {
        Coalescer {
            bmax,
            groups: groups.max(1),
            inner: std::sync::Mutex::new(CoalesceState {
                members: 0,
                in_flight: 0,
                next: 0,
                waiting: Vec::new(),
                done: std::collections::HashMap::new(),
            }),
            cv: std::sync::Condvar::new(),
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, CoalesceState<S, A>> {
        self.inner.lock().unwrap_or_else(|p| p.into_inner())
    }

    fn join(&self) {
        self.lock().members += 1;
    }

    fn leave(&self) {
        let mut g = self.lock();
        g.members = g.members.saturating_sub(1);
        drop(g);
        // A smaller membership may complete a batch that is already waiting.
        self.cv.notify_all();
    }

    /// Deposit `(tok, state)` and return this row's result; `ask` says what the row needs
    /// besides the argmax (its full row, a device draw). `run` executes one batch and returns
    /// its rows' results in order (or one error for all).
    fn step(
        &self,
        tok: u32,
        ask: A,
        state: &mut S,
        run: &mut BatchRun<'_, S, A>,
    ) -> Result<RowOut, String> {
        let mut g = self.lock();
        let ticket = g.next;
        g.next += 1;
        g.waiting.push((ticket, tok, ask, Lent(state as *mut S)));
        let t0 = std::time::Instant::now();
        loop {
            if let Some(result) = g.done.remove(&ticket) {
                return result;
            }
            let pos = g.waiting.iter().position(|d| d.0 == ticket);
            let free = g.members.saturating_sub(g.in_flight);
            let target = g.members.div_ceil(self.groups).clamp(1, self.bmax);
            let full = g.waiting.len() >= target.min(free).max(1);
            if let Some(mine) = pos
                && (full || t0.elapsed() >= ROW_BATCH_WAIT)
            {
                // Lead: the oldest deposits, with ours among them.
                let mut take: Vec<usize> = (0..g.waiting.len()).collect();
                take.truncate(target);
                if !take.contains(&mine) {
                    *take.last_mut().expect("bmax >= 1") = mine;
                }
                take.sort_unstable();
                let batch: Vec<(u64, u32, A, Lent<S>)> = take
                    .into_iter()
                    .rev()
                    .map(|i| g.waiting.remove(i))
                    .collect::<Vec<_>>()
                    .into_iter()
                    .rev()
                    .collect();
                g.in_flight += batch.len();
                drop(g);
                let toks: Vec<u32> = batch.iter().map(|d| d.1).collect();
                let wants: Vec<A> = batch.iter().map(|d| d.2).collect();
                // SAFETY: see `Lent`; every lender is blocked on its ticket.
                let mut states: Vec<&mut S> =
                    batch.iter().map(|d| unsafe { &mut *d.3.0 }).collect();
                // A panicking step must not wedge the lanes that lent it their rows: every member
                // gets an error, `in_flight` comes down, and the leader then unwinds as before.
                let out = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    run(&toks, &wants, &mut states)
                }));
                drop(states);
                g = self.lock();
                g.in_flight -= batch.len();
                let out = match out {
                    Ok(out) => out,
                    Err(panic) => {
                        for d in &batch {
                            g.done.insert(d.0, Err("B-row step panicked".to_string()));
                        }
                        drop(g);
                        self.cv.notify_all();
                        std::panic::resume_unwind(panic);
                    }
                };
                match out {
                    Ok(next) if next.len() == batch.len() => {
                        for (d, r) in batch.iter().zip(next) {
                            g.done.insert(d.0, Ok(r));
                        }
                    }
                    Ok(next) => {
                        let why = format!(
                            "B-row step returned {} rows for {}",
                            next.len(),
                            batch.len()
                        );
                        for d in &batch {
                            g.done.insert(d.0, Err(why.clone()));
                        }
                    }
                    Err(why) => {
                        for d in &batch {
                            g.done.insert(d.0, Err(why.clone()));
                        }
                    }
                }
                self.cv.notify_all();
                continue;
            }
            g = if pos.is_some() {
                let left = ROW_BATCH_WAIT.saturating_sub(t0.elapsed());
                self.cv
                    .wait_timeout(g, left.max(std::time::Duration::from_micros(20)))
                    .unwrap_or_else(|p| p.into_inner())
                    .0
            } else {
                self.cv.wait(g).unwrap_or_else(|p| p.into_inner())
            };
        }
    }
}

/// What a row asks of a B-row step besides its device argmax.
#[derive(Clone, Copy, Debug)]
enum RowAsk {
    /// The argmax only (greedy).
    Argmax,
    /// The full logits row (the host sampler, a penalty).
    Logits,
    /// A device-sampler draw at the row's position (memra #710: TP/EP sampled rows).
    Draw(Dsv4SampleCfg),
}

impl RowAsk {
    fn logits(self) -> bool {
        matches!(self, RowAsk::Logits)
    }
}

/// The route's B-row batcher (`MEMRA_DSV4_ROWS`): the coalescer plus the B-row workspace the
/// leader uses under the launch turn.
pub struct RowBatcher {
    core: Coalescer<DecodeState, RowAsk>,
    /// The device sampler a leader draws `RowAsk::Draw` rows with: draws are keyed on (seed,
    /// position), so one sampler serves every request's rows. Present when the load's plain
    /// sampler is the device one.
    sampler: std::sync::Mutex<Option<DeviceSampler>>,
    /// One B-row workspace per batch that can be in flight: two when the lanes can fill two
    /// groups (`2 * B <= lanes`), so one group's stage 0 runs while the other's stage 1 does.
    pool: std::sync::Mutex<Vec<RowsWorkspace>>,
    freed: std::sync::Condvar,
}

struct RowsWorkspace(memra_engine::dsv4_gpu::VerifyState);

// SAFETY: only a batch leader touches it, holding both this mutex and the launch turn.
unsafe impl Send for RowsWorkspace {}

struct DeviceSampler(memra_engine::dsv4_sampler::Dsv4DeviceSampler);

// SAFETY: only a batch leader touches it, holding both this mutex and the launch turn.
unsafe impl Send for DeviceSampler {}

impl RowBatcher {
    fn new(
        bmax: usize,
        ws: Vec<memra_engine::dsv4_gpu::VerifyState>,
        sampler: Option<memra_engine::dsv4_sampler::Dsv4DeviceSampler>,
    ) -> Self {
        RowBatcher {
            sampler: std::sync::Mutex::new(sampler.map(DeviceSampler)),
            core: Coalescer::new(bmax, ws.len()),
            pool: std::sync::Mutex::new(ws.into_iter().map(RowsWorkspace).collect()),
            freed: std::sync::Condvar::new(),
        }
    }

    fn take_ws(&self) -> RowsWorkspace {
        let mut pool = self.pool.lock().unwrap_or_else(|p| p.into_inner());
        loop {
            if let Some(ws) = pool.pop() {
                return ws;
            }
            pool = self.freed.wait(pool).unwrap_or_else(|p| p.into_inner());
        }
    }

    fn put_ws(&self, ws: RowsWorkspace) {
        self.pool.lock().unwrap_or_else(|p| p.into_inner()).push(ws);
        self.freed.notify_one();
    }
}

/// A lane's membership in the coalesced decode loop; leaving on drop keeps the batch count
/// right on every exit, including an unwind.
struct RowMember<'a>(&'a RowBatcher);

impl<'a> RowMember<'a> {
    fn join(b: &'a RowBatcher) -> Self {
        b.core.join();
        RowMember(b)
    }
}

impl Drop for RowMember<'_> {
    fn drop(&mut self) {
        self.0.core.leave();
    }
}

/// One plain step through the B-row batcher: the next token, and the full logits row when the
/// row asks for it (the host sampler, a penalty). The lane gives the turn up while its row
/// waits; the leader takes it for the batch. One row runs the one-row program: the full-token
/// replay when the request is armed and covered, else the one-row eager step. Several run one
/// B-row step, full-logits only when some row asked for it. A greedy row's token is always the
/// device argmax, the one-row greedy step's selection; a `Draw` row's is the device sampler's
/// draw at its position; and every row's bits equal its one-row step (`dsv4_rows_gate`).
fn rows_step(
    m: &Dsv4Model,
    batcher: &RowBatcher,
    turn: &mut Turn,
    tok: u32,
    ask: RowAsk,
    state: &mut DecodeState,
) -> Result<RowOut, String> {
    turn.release();
    let lock = turn.lock;
    let result = batcher
        .core
        .step(tok, ask, state, &mut |toks, asks, states| {
            let full = asks.iter().any(|a| a.logits());
            if let [one] = states {
                let mut lead = Turn::take(lock);
                if m.gpu.full_token_replay_covers(one) {
                    // An armed request draws its own token in graph, greedy or sampled.
                    return m
                        .gpu
                        .decode_sample_full_token(toks[0], one)
                        .map(|tok| vec![RowOut { tok, logits: None }]);
                }
                return match asks[0] {
                    RowAsk::Logits => logits_step(m, &mut lead, toks[0], one).map(|row| {
                        vec![RowOut {
                            tok: argmax(&row),
                            logits: Some(row),
                        }]
                    }),
                    RowAsk::Argmax => greedy_step(m, &mut lead, toks[0], one)
                        .map(|tok| vec![RowOut { tok, logits: None }]),
                    RowAsk::Draw(cfg) => {
                        m.gpu.decode_step_device_logits(toks[0], one)?;
                        let mut sampler = batcher.sampler.lock().unwrap_or_else(|p| p.into_inner());
                        let sampler = sampler
                            .as_mut()
                            .ok_or("B-row device draw without a device sampler")?;
                        m.gpu
                            .sample_device_logits(one, &mut sampler.0, &cfg, &[], None)
                            .map(|tok| vec![RowOut { tok, logits: None }])
                    }
                };
            }
            let mut ws = batcher.take_ws();
            let out = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                if m.gpu.topology().is_tp_ep() {
                    // TP/EP: both ranks work on every row, so the whole step runs holding the
                    // turn; there is no stage to overlap another group with.
                    let _lead = Turn::take(lock);
                    let (rows, am) = if full {
                        let (rows, am) = m.gpu.decode_rows_full(toks, states, &mut ws.0)?;
                        (Some(rows), am)
                    } else {
                        (None, m.gpu.decode_rows_greedy(toks, states, &mut ws.0)?)
                    };
                    let mut rows = rows.map(Vec::into_iter);
                    let mut out = Vec::with_capacity(toks.len());
                    for (i, (&tok, ask)) in am.iter().zip(asks).enumerate() {
                        let row = rows.as_mut().and_then(Iterator::next);
                        out.push(match *ask {
                            RowAsk::Argmax => RowOut { tok, logits: None },
                            RowAsk::Logits => RowOut { tok, logits: row },
                            RowAsk::Draw(cfg) => {
                                let mut sampler =
                                    batcher.sampler.lock().unwrap_or_else(|p| p.into_inner());
                                let sampler = sampler
                                    .as_mut()
                                    .ok_or("B-row device draw without a device sampler")?;
                                RowOut {
                                    tok: m.gpu.sample_rows_device(
                                        &ws.0,
                                        i,
                                        states[i].pos,
                                        &mut sampler.0,
                                        &cfg,
                                    )?,
                                    logits: None,
                                }
                            }
                        });
                    }
                    return Ok(out);
                }
                // PP: queue the group holding the turn, wait for its own readbacks without it
                // (so another group can queue behind it and the two cards overlap), then commit.
                let mut lead = Turn::take(lock);
                m.gpu.decode_rows_enqueue(toks, states, &mut ws.0, full)?;
                lead.release();
                let waited = m.gpu.decode_rows_wait(&mut ws.0);
                lead.acquire();
                waited?;
                let (rows, am) = m.gpu.decode_rows_complete(states, &mut ws.0)?;
                Ok(match rows {
                    Some(rows) => rows
                        .into_iter()
                        .zip(am)
                        .zip(asks)
                        .map(|((row, tok), a)| RowOut {
                            tok,
                            logits: a.logits().then_some(row),
                        })
                        .collect(),
                    None => am
                        .into_iter()
                        .map(|tok| RowOut { tok, logits: None })
                        .collect(),
                })
            }));
            // The workspace goes back to the pool on every exit, a panic included.
            batcher.put_ws(ws);
            match out {
                Ok(out) => out,
                Err(panic) => std::panic::resume_unwind(panic),
            }
        });
    turn.acquire();
    result
}

/// A session's cache capacity. TP/EP plain requests serve on the full-token replay graphs,
/// which need a capacity of at least [`REPLAY_MIN_CAPACITY`], so a shorter session is rounded
/// up so it can arm, never past the model context: a boot with a smaller context serves the
/// eager step, because the arm refuses the capacity.
fn session_capacity(replay_route: bool, need: usize, max_seq: usize) -> usize {
    if replay_route {
        need.max(REPLAY_MIN_CAPACITY).min(max_seq).max(need)
    } else {
        need
    }
}

/// The greedy program a TP/EP replay arms: the device argmax, no draw.
const GREEDY_REPLAY: Dsv4SampleCfg = Dsv4SampleCfg {
    temperature: 0.0,
    top_p: 1.0,
    top_k: 0,
    seed: 0,
};

/// Arm a TP/EP plain request on the full-token replay graphs (memra #710). A refused arm
/// leaves the request on the eager step, the same numeric program, and logs why.
fn arm_replay(m: &Dsv4Model, state: &mut DecodeState, cfg: Dsv4SampleCfg) -> bool {
    // SAFETY: the route holds the model behind an `Arc` for the whole request, so it outlives
    // the state at a stable address, and nothing replaces its weights or reconfigures its
    // kernels after load.
    match unsafe { m.gpu.arm_full_token_replay(state, cfg) } {
        Ok(()) => true,
        Err(why) => {
            eprintln!("[dsv4-serve] TP/EP replay not armed: {why}");
            false
        }
    }
}

/// Whether an armed request's next step still replays. Past the replay limit the request
/// drops its graphs and continues on the eager step, the same numeric program
/// (`dsv4_tp_replay_long_gate` crosses that handoff).
fn keep_replay(m: &Dsv4Model, state: &mut DecodeState) -> Result<bool, String> {
    if m.gpu.full_token_replay_covers(state) {
        return Ok(true);
    }
    m.gpu.disarm_full_token_replay(state)?;
    Ok(false)
}

/// One plain greedy step. Serial lanes, and TP/EP lanes (whose steps use both cards), take the
/// one-call step. Pipelined PP lanes queue the step holding the turn, wait for its readbacks
/// without it, then take the turn back to commit.
fn greedy_step(
    m: &Dsv4Model,
    turn: &mut Turn,
    tok: u32,
    state: &mut DecodeState,
) -> Result<u32, String> {
    if m.sessions <= 1 || !m.gpu.pipelined_steps_supported() {
        return m.gpu.decode_step_greedy(tok, state);
    }
    turn.acquire();
    m.gpu.decode_step_greedy_enqueue(tok, state)?;
    turn.release();
    let waited = m.gpu.decode_step_greedy_wait(state);
    turn.acquire();
    waited?;
    m.gpu.decode_step_greedy_complete(state)
}

/// One plain step that returns the logits row (host-sampled and penalized routes), serial or
/// pipelined exactly as [`greedy_step`].
fn logits_step(
    m: &Dsv4Model,
    turn: &mut Turn,
    tok: u32,
    state: &mut DecodeState,
) -> Result<Vec<f32>, String> {
    if m.sessions <= 1 || !m.gpu.pipelined_steps_supported() {
        return m.gpu.decode_step(tok, state);
    }
    turn.acquire();
    m.gpu.decode_step_logits_enqueue(tok, state)?;
    turn.release();
    let waited = m.gpu.decode_step_greedy_wait(state);
    turn.acquire();
    waited?;
    m.gpu.decode_step_logits_complete(state)
}

/// Streaming state shared by every route: incremental detok, EOS, stop strings,
/// budget — one place, so plain and spec paths cannot diverge on stop semantics.
/// Snap a byte index backward to the nearest char boundary. Every byte cut into
/// UTF-8 text in this route goes through this — the DSML markers (fullwidth
/// '\u{ff5c}', 3 bytes) and ordinary typography ('\u{2019}' curly apostrophe, the
/// box10 owner-serve panic) both land inside fixed-offset windows on real output.
pub(crate) fn snap(s: &str, mut i: usize) -> usize {
    i = i.min(s.len());
    while i > 0 && !s.is_char_boundary(i) {
        i -= 1;
    }
    i
}

/// Stop-string scan over the running text (matches may straddle deltas): returns
/// the byte cut in `full` where generation stops, or None. The two PARSER-CLOSING
/// stops stay IN the stream (worker law: "the close stays in the stream so the
/// parser finishes the span" — dsv4 DSML close + gemma `<tool_call|>`); user stops
/// are exclusive. Extracted from Emit::push so the multi-byte window arithmetic has
/// a toothed unit gate (both boundary panics were found LIVE, never by a test).
pub(crate) fn scan_stop_cut(text: &str, full: &str, stops: &[String]) -> Option<usize> {
    let delta = &full[snap(full, text.len())..];
    let scan_from = snap(text, text.len().saturating_sub(64));
    let probe = format!("{}{}", &text[scan_from..], delta);
    let mut cut: Option<usize> = None;
    for ss in stops {
        if let Some(p) = probe.find(ss.as_str()) {
            let abs = scan_from + p;
            let inclusive = ss == "</\u{ff5c}DSML\u{ff5c}tool_calls>" || ss == "<tool_call|>";
            let end = if inclusive { abs + ss.len() } else { abs };
            // never rewind past already-emitted text
            let end = end.max(text.len()).min(full.len());
            cut = Some(cut.map_or(end, |c: usize| c.min(end)));
        }
    }
    cut
}

struct Emit<'a> {
    tok: &'a Tokenizer,
    tx: &'a crate::worker::EventSender,
    eos: Vec<u32>,
    stop_strings: &'a [String],
    stop_token_ids: &'a [u32],
    budget: usize,
    ids: Vec<u32>,
    text: String,
    stop_reason: Option<&'static str>,
    client_gone: bool,
    /// The EOS id this stream CONSUMED but did not deliver. The device state advanced over it,
    /// so it is part of the token boundary the parked-prefix tier has to name; without it an
    /// eos-terminated session cannot park at all (see `processed_prefix_tokens`).
    terminal: Option<u32>,
    /// The serving route's books; `None` off the serving thread (unit tests).
    progress: Option<&'a mut RouteProgress>,
}

impl<'a> Emit<'a> {
    fn new(
        tok: &'a Tokenizer,
        tx: &'a crate::worker::EventSender,
        eos: Vec<u32>,
        stop_strings: &'a [String],
        stop_token_ids: &'a [u32],
        budget: usize,
    ) -> Self {
        Emit {
            tok,
            tx,
            eos,
            stop_strings,
            stop_token_ids,
            budget,
            ids: Vec::new(),
            text: String::new(),
            stop_reason: None,
            client_gone: false,
            terminal: None,
            progress: None,
        }
    }

    /// Feed a speculative round's tokens BEFORE the device commits them, and answer how many
    /// this stream took plus whether generation stops here (memra #495). The driver commits
    /// exactly `taken` rows, which is what keeps the device state at the boundary
    /// `park_prefix` can name. A swallowed EOS COUNTS as taken: the device consumed it and
    /// the next turn's render replays it after the assistant content.
    fn push_round(&mut self, new: &[u32]) -> RoundTake {
        // Every call is one committed decode step or speculative round: forward progress.
        if let Some(p) = self.progress.as_mut() {
            p.round();
        }
        let mut taken = 0usize;
        for &id in new {
            if self.stop_reason.is_some() || self.client_gone {
                break;
            }
            if crate::worker::stop_token_reason(id, self.stop_token_ids).is_some()
                || self.eos.contains(&id)
            {
                self.stop_reason = Some("stop");
                self.terminal = Some(id);
                taken += 1;
                break;
            }
            self.ids.push(id);
            taken += 1;
            // v1 detok: re-decode the whole tail and diff (O(n^2) chars at serve
            // lengths — correctness first; the hybrid worker's append-only decoder
            // is the follow-up if this ever profiles).
            let full = self.tok.decode(&self.ids);
            let mut delta = full[snap(&full, self.text.len())..].to_string();
            let cut = scan_stop_cut(&self.text, &full, self.stop_strings);
            if let Some(abs) = cut {
                delta = full[snap(&full, self.text.len())..snap(&full, abs)].to_string();
                self.text.push_str(&delta);
                if self
                    .tx
                    .send(Event::Token {
                        id,
                        text: delta.clone(),
                    })
                    .is_err()
                {
                    self.client_gone = true;
                }
                self.stop_reason = Some("stop");
                break;
            }
            self.text = full;
            if self.tx.send(Event::Token { id, text: delta }).is_err() {
                self.client_gone = true;
                break;
            }
            if self.ids.len() >= self.budget {
                self.stop_reason = Some("length");
                break;
            }
        }
        RoundTake {
            taken,
            stop: self.stop_reason.is_some() || self.client_gone,
        }
    }

    /// The plain (non-speculative) loops feed one token at a time and only need the stop bit.
    fn push(&mut self, new: &[u32]) -> bool {
        !self.push_round(new).stop
    }

    fn finish(self, n_prompt: usize, n_cached: usize, elapsed_s: f64, spec: Option<SpecUsage>) {
        if self.client_gone {
            return; // receipts stay with the terminal-less stream (HTTP layer owns it)
        }
        let _ = self.tx.send(Event::TokenSnapshot(self.ids.clone()));
        let _ = self.tx.send(Event::Done {
            stop_reason: self.stop_reason.unwrap_or("length").to_string(),
            n_tokens: self.ids.len(),
            n_prompt,
            n_cached,
            elapsed_s,
            spec,
        });
    }
}

fn render_prompt(m: &Dsv4Model, req: &Request) -> Result<Vec<u32>, EngineError> {
    if let Some(error) = crate::worker::prompt_source_limit_error(req) {
        let param = if !req.prompt_ids.is_empty() {
            "prompt_ids"
        } else if !req.chat_turns.is_empty() {
            "messages"
        } else {
            "prompt"
        };
        return Err(EngineError::invalid_param(error, param));
    }
    if let Some(t) = req.ttft.as_ref() {
        t.mark_tokenize_start();
    }
    // The worker's prepare_request rendering law, applied to the dsv4 tokenizer.
    let prompt = if !req.prompt_ids.is_empty() {
        req.prompt_ids.clone()
    } else if !req.chat_turns.is_empty() {
        // ONE predicate for the plain-vs-tools render (memra CLAUDE.md, v0.109.1 lesson): the
        // worker, the HTTP-side accounting and this route must agree, or `/v1/tokenize` and
        // the prepaid reservation count a different prompt than the one served here. The
        // local copy this replaces dropped the `reasoning` and effort-ladder terms.
        let plain = crate::worker::plain_chat_render_path(
            &req.tools_json,
            &req.think,
            req.reasoning_effort.as_deref(),
            &req.chat_turns,
            m.tok.has_qwen_effort_ladder(),
        );
        let rendered = if plain {
            let messages: Vec<_> = req
                .chat_turns
                .iter()
                .map(|t| (t.role.as_str(), t.content.as_str()))
                .collect();
            m.tok.apply_chat_template(&messages, true)
        } else {
            m.tok
                .apply_chat_template_tools_ex(
                    &req.chat_turns,
                    true,
                    &req.tools_json,
                    &req.tools_struct,
                    req.think,
                    req.reasoning_effort.as_deref(),
                )
                .map_err(|e| {
                    EngineError::invalid_param(format!("chat template: {e}"), "messages")
                })?
        };
        m.tok.encode(&rendered, true)
    } else if req.chat {
        let rendered = m
            .tok
            .apply_chat_template(&[("user", req.prompt_text.as_str())], true);
        m.tok.encode(&rendered, true)
    } else {
        m.tok.encode(&req.prompt_text, true)
    };
    if prompt.is_empty() {
        return Err(EngineError::invalid_param(
            "empty prompt after tokenization",
            "prompt",
        ));
    }
    if let Some(t) = req.ttft.as_ref() {
        t.mark_tokenize_end(prompt.len());
    }
    if let Some(limit) = req.max_prompt_tokens
        && prompt.len() > limit
    {
        return Err(EngineError::context_length(format!(
            "prompt ({} tok) exceeds this model's prompt ceiling ({limit})",
            prompt.len()
        )));
    }
    Ok(prompt)
}

struct RestoredPrefix {
    state: DecodeState,
    dstate: Option<DsparkState>,
    logits: Vec<f32>,
    n_cached: usize,
}

/// Feed the strict suffix after a restored plain trunk boundary. Intermediate rows need no
/// host logits, so they use the 4-byte device-argmax path and discard the value; only the
/// final suffix token returns the row that seeds generation.
fn continue_plain_prefix(
    m: &Dsv4Model,
    suffix: &[u32],
    state: &mut DecodeState,
) -> Result<Vec<f32>, String> {
    if suffix.is_empty() {
        return Err("dsv4 restored continuation needs a non-empty suffix".into());
    }
    if m.prefill_chunk > 0 {
        return m
            .gpu
            .continue_prefix_chunked(suffix, state, m.prefill_chunk);
    }
    let mut last = None;
    for (i, &tok) in suffix.iter().enumerate() {
        if i + 1 == suffix.len() {
            last = Some(m.gpu.decode_step(tok, state)?);
        } else {
            let _ = m.gpu.decode_step_greedy(tok, state)?;
        }
        // Each step read its token or row back: one completed prime row (memra#500).
        memra_engine::progress::note_prime_rows(1);
    }
    Ok(last.expect("non-empty suffix has final logits"))
}

/// Say the miss out loud. `reuse MISS` is the line a receipt greps for; without it a prompt that
/// re-rendered differently than it was parked is invisible (memra #495).
fn log_reuse_miss(miss: &TakeMiss, prompt_tokens: usize, need_dspark: bool) {
    match *miss {
        TakeMiss::EmptyNamespace => eprintln!(
            "[dsv4-host] reuse MISS: namespace empty, prompt {prompt_tokens} tokens, \
             dspark={need_dspark}; cold prefill"
        ),
        TakeMiss::NoPrefix {
            candidates,
            best_n,
            best_lcp,
            dspark_short,
        } => eprintln!(
            "[dsv4-host] reuse MISS: {candidates} parked, none a strict prefix of this \
             {prompt_tokens}-token prompt (best entry {best_n} tokens, diverges at {best_lcp}; \
             {dspark_short} lacked DSpark), dspark={need_dspark}; cold prefill"
        ),
    }
}

fn try_restore_prefix(
    m: &Dsv4Model,
    host: &mut Dsv4HostCache,
    req: &Request,
    prompt: &[u32],
    capacity: usize,
    need_dspark: bool,
) -> Option<RestoredPrefix> {
    if !host.armed() {
        return None;
    }
    let entry = match host.take(&req.cache_ns, req.affinity.as_deref(), prompt, need_dspark) {
        Ok(entry) => entry,
        Err(miss) => {
            log_reuse_miss(&miss, prompt.len(), need_dspark);
            return None;
        }
    };
    let n_cached = entry.toks.len();
    let bytes = entry.bytes;
    let t0 = Instant::now();
    let restored = (|| -> Result<RestoredPrefix, String> {
        let mut state = m.request_state(capacity, Some(&entry.trunk))?;
        let suffix = &prompt[n_cached..];
        if need_dspark {
            let host_dstate = entry
                .dspark
                .as_ref()
                .ok_or_else(|| "dsv4 host hit missing required DSpark state".to_string())?;
            let mut dstate = m.gpu.restore_dspark_state(host_dstate)?;
            let logits = if m.prefill_chunk > 0 {
                m.gpu.dspark_continue_prefix_chunked(
                    suffix,
                    &mut state,
                    &mut dstate,
                    m.prefill_chunk,
                )?
            } else {
                m.gpu
                    .dspark_continue_prefix(suffix, &mut state, &mut dstate)?
            };
            Ok(RestoredPrefix {
                state,
                dstate: Some(dstate),
                logits,
                n_cached,
            })
        } else {
            let logits = continue_plain_prefix(m, suffix, &mut state)?;
            Ok(RestoredPrefix {
                state,
                dstate: None,
                logits,
                n_cached,
            })
        }
    })();
    match restored {
        Ok(hit) => {
            eprintln!(
                "[dsv4-host] hit: {} cached + {} suffix tokens, {:.1} MiB, {:.1} ms, dspark={need_dspark}",
                hit.n_cached,
                prompt.len() - hit.n_cached,
                bytes as f64 / 1048576.0,
                t0.elapsed().as_secs_f64() * 1000.0,
            );
            Some(hit)
        }
        Err(err) => {
            eprintln!(
                "[dsv4-host] restore failed after consuming {n_cached}-token entry ({err}); cold prefill"
            );
            None
        }
    }
}

fn park_prefix(
    m: &Dsv4Model,
    host: &mut Dsv4HostCache,
    req: &Request,
    toks: Vec<u32>,
    state: &DecodeState,
    dstate: Option<&DsparkState>,
) {
    if !host.armed() || toks.len() < DSV4_HOST_CACHE_MIN_TOKENS {
        return;
    }
    let t0 = Instant::now();
    let parked = (|| -> Result<ParkedEntry, String> {
        let trunk = m.gpu.snapshot_decode_state(state)?;
        if trunk.pos() != toks.len() {
            return Err(format!(
                "snapshot pos {} != token boundary {}",
                trunk.pos(),
                toks.len()
            ));
        }
        let dspark = dstate.map(|s| m.gpu.snapshot_dspark_state(s)).transpose()?;
        let bytes = trunk.bytes() + dspark.as_ref().map_or(0, Dsv4HostDsparkState::bytes);
        Ok(ParkedEntry {
            toks,
            affinity: req.affinity.clone(),
            trunk,
            dspark,
            bytes,
            last_use: Instant::now(),
            id: 0,
        })
    })();
    match parked {
        Ok(entry) => {
            let n = entry.toks.len();
            let bytes = entry.bytes;
            host.insert(req.cache_ns.clone(), entry);
            eprintln!(
                "[dsv4-host] park: {n} tokens, {:.1} MiB, {:.1} ms (resident {:.1}/{:.1} MiB)",
                bytes as f64 / 1048576.0,
                t0.elapsed().as_secs_f64() * 1000.0,
                host.total_bytes as f64 / 1048576.0,
                host.budget as f64 / 1048576.0,
            );
        }
        Err(err) => host.disable(&format!("snapshot failed: {err}")),
    }
}

/// Token boundary represented by a completed request's device state. The final emitted token
/// is commonly still pending (state one token behind), which is a valid strict prefix for the
/// next turn.
///
/// State AHEAD of the visible stream is the normal end of an eos-terminated session, not an
/// anomaly: `Emit` swallows the EOS id rather than delivering it, while the device consumed it.
/// That one token is known (`terminal`) and it is exactly what the next turn's render replays
/// after the assistant content, so the boundary including it is the one a continuation matches.
/// Measured on the prod candidate 2026-09-11 (darklanes `research/dsv4f-hot-ttft-20260911`):
/// with this token missing, EVERY eos-terminated request refused to park
/// ("device state consumed N generated tokens, stream committed only N-1") and the parked-prefix
/// tier was inert on the only traffic shape that can use it, at any budget.
///
/// State further ahead than that single EOS can still happen when a speculative round commits
/// several tokens before the stop callback. Those ids are past the stop and no continuation will
/// ever replay them, so that shape stays uncacheable rather than being papered over.
fn processed_prefix_tokens(
    prompt: &[u32],
    emitted: &[u32],
    terminal: Option<u32>,
    state_pos: usize,
) -> Result<Vec<u32>, String> {
    let consumed_generated = state_pos.checked_sub(prompt.len()).ok_or_else(|| {
        format!(
            "device state pos {state_pos} precedes prompt boundary {}",
            prompt.len()
        )
    })?;
    let mut toks = Vec::with_capacity(prompt.len() + consumed_generated);
    toks.extend_from_slice(prompt);
    match consumed_generated.checked_sub(emitted.len()) {
        None | Some(0) => toks.extend_from_slice(&emitted[..consumed_generated]),
        Some(1) => {
            let id = terminal.ok_or_else(|| {
                format!(
                    "device state consumed {consumed_generated} generated tokens, stream \
                     committed only {} and swallowed no terminal token",
                    emitted.len()
                )
            })?;
            toks.extend_from_slice(emitted);
            toks.push(id);
        }
        Some(ahead) => {
            return Err(format!(
                "device state consumed {consumed_generated} generated tokens, stream committed \
                 only {} ({ahead} past the stop)",
                emitted.len()
            ));
        }
    }
    Ok(toks)
}

/// How one dequeued request left the route.
#[derive(Debug)]
pub(crate) enum Served {
    Done(ServeStats),
    /// The client left while the request waited for memory.
    Cancelled,
    /// The memory door turned it away (memra#503); the error is the client's answer.
    Refused(EngineError),
}

/// The client's answer for one admission outcome: `None` serves the request, `Some` turns it
/// away, with the Retry-After a refusal carries (for the receipt line). `fits` is the largest
/// session the route can hold; it runs only for a session no wait can fit.
pub(crate) fn admission_answer(
    outcome: &Admission,
    capacity: usize,
    prompt_len: usize,
    fits: impl FnOnce() -> usize,
) -> (Option<Served>, Option<u64>) {
    match *outcome {
        Admission::Admit { .. } => (None, None),
        Admission::Cancelled { .. } => (Some(Served::Cancelled), None),
        Admission::Refuse { .. } => {
            let s = crate::admit_memory::clamp_retry_after_s(None);
            let err = EngineError::rate_limit_after(crate::admit_memory::MEMORY_REFUSE_MESSAGE, s);
            (Some(Served::Refused(err)), Some(s))
        }
        Admission::NeverFits { dev, need, ceiling } => {
            let err = EngineError::context_length(format!(
                "a {capacity}-token session (prompt {prompt_len} + max_tokens {}) needs {need} \
                 bytes on device {dev}, above the {ceiling} bytes this route offers there; the \
                 largest session that fits is {} tokens",
                capacity.saturating_sub(prompt_len),
                fits(),
            ));
            (Some(Served::Refused(err)), None)
        }
    }
}

/// Book how a served request left the route and answer its client when it was turned away.
pub(crate) fn settle(run: RouteRun, req: &mut Request, served: Result<Served, EngineError>) {
    match served {
        Ok(Served::Done(stats)) => run.finish(stats),
        Ok(Served::Cancelled) => run.cancel(),
        Ok(Served::Refused(err)) => {
            run.refuse();
            fail_request(req, err);
        }
        Err(err) => fail_request(req, err),
    }
}

/// The route's memory door (memra#503), run before a parked prefix is consumed or any state
/// is allocated. `None` admits. A refusal answers 429 with `Retry-After`; a session no wait
/// can fit answers a context-length error naming the largest session that fits.
fn admit_request(
    m: &Dsv4Model,
    turn: &mut Turn,
    req: &Request,
    prompt: &[u32],
    capacity: usize,
    spec: bool,
    host_c4: u64,
) -> Result<Option<Served>, EngineError> {
    let need = m
        .session_need(capacity, spec, host_c4)
        .map_err(EngineError::engine)?;
    let spare =
        turn.cache()
            .restore_candidate(&req.cache_ns, req.affinity.as_deref(), prompt, spec);
    let mut probe = LiveProbe {
        gpu: &m.gpu,
        turn,
        spare,
        tx: &req.tx,
        t0: Instant::now(),
    };
    let outcome = dsv4_admit::admit_session(&need, &mut probe, route_defer_budget_ms())
        .map_err(EngineError::engine)?;
    let (answer, retry_after_s) = admission_answer(&outcome, capacity, prompt.len(), || {
        dsv4_admit::largest_fitting_capacity(m.max_seq, |c| m.fits_ceiling(c, spec))
    });
    eprintln!(
        "{}",
        dsv4_admit::admission_line(&AdmissionLine {
            request_id: &req.request_id,
            model: &req.model,
            capacity,
            spec,
            need: &need,
            outcome: &outcome,
            retry_after_s,
        })
    );
    Ok(answer)
}

fn serve_one(
    m: &Dsv4Model,
    turn: &mut Turn,
    req: &mut Request,
    progress: Option<&mut RouteProgress>,
) -> Result<Served, EngineError> {
    let t0 = std::time::Instant::now();
    if req.grammar.is_some() {
        return Err(EngineError::invalid_param(
            "response_format is not available on the dsv4 route yet",
            "response_format",
        ));
    }
    let prompt = render_prompt(m, req)?;
    // The spec drivers' own sizing law (gate bins): prompt + n_new + 96 slack must fit.
    const SLACK: usize = 96;
    if prompt.len() + SLACK >= m.max_seq {
        return Err(EngineError::context_length(format!(
            "prompt ({} tok) >= context cap ({} minus {SLACK} slack)",
            prompt.len(),
            m.max_seq
        )));
    }
    let budget = req
        .params
        .max_new
        .min(m.max_seq - SLACK - prompt.len())
        .max(1);
    let sc = &req.sampler_cfg;
    let greedy = sc.temperature <= 0.0;
    let penalties_set =
        sc.penalty_repeat != 1.0 || sc.penalty_freq != 0.0 || sc.penalty_present != 0.0;
    // rung-2 slice 2: penalties over the true per-state window. The serve API arms
    // the whole context when a penalty is set without an explicit window (the q38
    // arming law). Penalized GREEDY serves on the PLAIN path — the spec greedy
    // driver argmaxes raw columns and would silently drop the penalties (q38 law:
    // "penalized greedy is served on the plain path").
    let pen_cfg = penalties_set.then_some(Dsv4PenaltyCfg {
        last_n: if sc.penalty_last_n > 0 {
            sc.penalty_last_n
        } else {
            usize::MAX
        },
        repeat: sc.penalty_repeat,
        freq: sc.penalty_freq,
        present: sc.penalty_present,
    });
    if !greedy && sc.min_p != 0.0 {
        return Err(EngineError::invalid_param(
            "min_p is not implemented on the dsv4 sampler; use top_p/top_k",
            "min_p",
        ));
    }
    let use_spec = m.spec && !(greedy && penalties_set);
    let session_capacity = session_capacity(
        m.gpu.topology().is_tp_ep() && !use_spec,
        prompt.len() + budget,
        m.max_seq,
    );
    // Memory admission (memra#503), before consuming a parked prefix or starting any state
    // allocation: the active C4 history against its configured budget (a session past it can
    // never fit, so it is the client's error, not a retryable one), then the whole session
    // against every owning card and the host.
    let host_c4 = m
        .planned_c4_host_bytes(session_capacity)
        .map_err(EngineError::context_length)?;
    if let Some(turned_away) = admit_request(
        m,
        turn,
        req,
        &prompt,
        session_capacity,
        use_spec,
        host_c4 as u64,
    )? {
        return Ok(turned_away);
    }
    // In the reference program, a prompt no wider than one configured chunk gets the canonical
    // monolithic prime. It is deliberately not parked: a later, longer cold prompt
    // uses the chunked numeric regime, so retaining this state would make cache use
    // choose a different realization. Once prompts exceed the chunk, cold and restored
    // paths are both chunked and cache-transparent. The experimental matrix program
    // instead uses its one batched executor at every width, including position zero.
    let short_monolithic = !m.gpu.matrix_moe_enabled()
        && m.prefill_chunk > 0
        && !use_chunked_prefill(m.prefill_chunk, prompt.len());
    let mut restored =
        try_restore_prefix(m, turn.cache(), req, &prompt, session_capacity, use_spec);
    let n_cached = restored.as_ref().map_or(0, |hit| hit.n_cached);
    let _ = req.tx.send(Event::PromptUsage {
        n_prompt: prompt.len(),
        n_cached,
    });

    let mut eos_set = req.params.eos.clone();
    if !eos_set.contains(&m.eos) {
        eos_set.push(m.eos);
    }
    let mut emit = Emit::new(
        &m.tok,
        &req.tx,
        eos_set,
        &req.stop_strings,
        &req.stop_token_ids,
        budget,
    );
    emit.progress = progress;
    let mut spec_usage: Option<SpecUsage> = None;
    let state_to_park: DecodeState;
    let mut dstate_to_park: Option<DsparkState> = None;

    if use_spec {
        // the depth/window policy resolved at load — exactly the bench drivers' law
        let Dsv4SpecPolicy { depth_cap, vt } = m.spec_policy;
        let (mut state, mut dstate, initial_logits) = if let Some(hit) = restored.take() {
            (
                hit.state,
                hit.dstate.expect("DSpark restore returned DSpark state"),
                Some(hit.logits),
            )
        } else {
            let mut state = m
                .request_state(session_capacity, None)
                .map_err(EngineError::engine)?;
            let mut dstate = m.gpu.dspark_alloc_state().map_err(EngineError::engine)?;
            let initial_logits = if m.prefill_chunk > 0 && !short_monolithic {
                Some(
                    m.gpu
                        .dspark_prefill_prime_chunked(
                            &prompt,
                            &mut state,
                            &mut dstate,
                            m.prefill_chunk,
                        )
                        .map_err(EngineError::engine)?,
                )
            } else {
                None
            };
            (state, dstate, initial_logits)
        };
        let mut vstate = m
            .gpu
            .alloc_verify_state_for(state.capacity)
            .map_err(EngineError::engine)?;
        // generation budget + 1: the drivers count the head token of the final round
        // inside n_new; Emit owns the exact budget/EOS truncation either way.
        let n_new = budget;
        let mut cb = |new: &[u32]| emit.push_round(new);
        let run = if let Some(initial_logits) = initial_logits.as_deref() {
            if greedy {
                m.gpu.spec_greedy_batched_stream_restored(
                    prompt.len(),
                    initial_logits,
                    n_new,
                    &mut state,
                    &mut dstate,
                    &mut vstate,
                    depth_cap,
                    vt,
                    Some(&mut cb),
                )
            } else {
                let cfg = Dsv4SampleCfg {
                    temperature: sc.temperature,
                    top_p: sc.top_p,
                    top_k: sc.top_k,
                    seed: sc.seed,
                };
                m.gpu.spec_sampled_batched_pen_restored(
                    &prompt,
                    initial_logits,
                    n_new,
                    &mut state,
                    &mut dstate,
                    &mut vstate,
                    depth_cap,
                    vt,
                    &cfg,
                    pen_cfg.as_ref(),
                    Some(&mut cb),
                )
            }
        } else if greedy {
            m.gpu.spec_greedy_batched_stream(
                &prompt,
                n_new,
                &mut state,
                &mut dstate,
                &mut vstate,
                depth_cap,
                vt,
                Some(&mut cb),
            )
        } else {
            let cfg = Dsv4SampleCfg {
                temperature: sc.temperature,
                top_p: sc.top_p,
                top_k: sc.top_k,
                seed: sc.seed,
            };
            m.gpu.spec_sampled_batched_pen(
                &prompt,
                n_new,
                &mut state,
                &mut dstate,
                &mut vstate,
                depth_cap,
                vt,
                &cfg,
                pen_cfg.as_ref(),
                Some(&mut cb),
            )
        }
        .map_err(EngineError::engine)?;
        let usage = SpecUsage {
            rounds: run.rounds.len() as u64,
            drafted: run
                .rounds
                .iter()
                .map(|r| r.t_batch.saturating_sub(1) as u64)
                .sum(),
            accepted: run.rounds.iter().map(|r| r.accepts as u64).sum(),
        };
        eprintln!(
            "[dspark-acc] request_id={:?} model={:?} sampled={} rounds={} drafted={} accepted={}",
            req.request_id, req.model, !greedy, usage.rounds, usage.drafted, usage.accepted
        );
        spec_usage = Some(usage);
        state_to_park = state;
        dstate_to_park = Some(dstate);
    } else {
        // trunk-only plain loops
        let (mut state, pre_logits) = if let Some(hit) = restored.take() {
            (hit.state, hit.logits)
        } else {
            let mut state = m
                .request_state(session_capacity, None)
                .map_err(EngineError::engine)?;
            let logits = if m.prefill_chunk > 0 && !short_monolithic {
                if m.sessions > 1 {
                    // Give the turn up between chunks so another session's steps keep going.
                    m.gpu.prefill_with_cache_chunked_yielding(
                        &prompt,
                        &mut state,
                        m.prefill_chunk,
                        &mut || {
                            turn.release();
                            turn.acquire();
                        },
                    )
                } else {
                    m.gpu
                        .prefill_with_cache_chunked(&prompt, &mut state, m.prefill_chunk)
                }
                .map_err(EngineError::engine)?
            } else {
                m.gpu
                    .prefill_with_cache(&prompt, &mut state)
                    .map_err(EngineError::engine)?
                    .logits
            };
            (state, logits)
        };
        let p0 = prompt.len();
        // running penalty window: prompt ++ every committed token (rung-2 slice 2)
        let mut window: Vec<u32> = if pen_cfg.is_some() {
            prompt.clone()
        } else {
            Vec::new()
        };
        if greedy {
            let mut t = if let Some(pc) = &pen_cfg {
                let mut row = pre_logits.clone();
                dsv4_penalize_row(&mut row, &window, pc);
                argmax(&row)
            } else {
                argmax(&pre_logits)
            };
            let mut step = 0usize;
            // Greedy rows coalesce across the lanes when the route batches them; a penalized
            // row asks for its logits.
            let batcher = m.rows.as_deref();
            let _member = batcher.map(RowMember::join);
            // A TP/EP greedy row without penalties replays its steps (memra #710); through
            // the batcher it replays whenever it steps alone.
            let mut replay = m.gpu.topology().is_tp_ep()
                && pen_cfg.is_none()
                && arm_replay(m, &mut state, GREEDY_REPLAY);
            while emit.push(&[t]) {
                if pen_cfg.is_some() {
                    window.push(t);
                }
                step += 1;
                if step >= budget {
                    break;
                }
                replay = replay && keep_replay(m, &mut state).map_err(EngineError::engine)?;
                t = if replay && batcher.is_none() {
                    m.gpu
                        .decode_sample_full_token(t, &mut state)
                        .map_err(EngineError::engine)?
                } else if let Some(pc) = &pen_cfg {
                    // penalized greedy needs the full row (argmax AFTER penalties)
                    let mut row = match batcher {
                        Some(b) => rows_step(m, b, turn, t, RowAsk::Logits, &mut state)
                            .map_err(EngineError::engine)?
                            .logits
                            .ok_or_else(|| EngineError::engine("B-row logits missing"))?,
                        None => logits_step(m, turn, t, &mut state).map_err(EngineError::engine)?,
                    };
                    dsv4_penalize_row(&mut row, &window, pc);
                    argmax(&row)
                } else if let Some(b) = batcher {
                    rows_step(m, b, turn, t, RowAsk::Argmax, &mut state)
                        .map_err(EngineError::engine)?
                        .tok
                } else {
                    greedy_step(m, turn, t, &mut state).map_err(EngineError::engine)?
                };
            }
        } else {
            let cfg = Dsv4SampleCfg {
                temperature: sc.temperature,
                top_p: sc.top_p,
                top_k: sc.top_k,
                seed: sc.seed,
            };
            let draw = |row: &mut Vec<f32>, pos: usize, window: &[u32]| -> Result<u32, String> {
                if let Some(pc) = &pen_cfg {
                    dsv4_penalize_row(row, window, pc);
                }
                dsv4_sample_row(row, pos, &cfg)
            };
            let mut device_sampler =
                if m.gpu.plain_sampler() == memra_engine::dsv4_sampler::Dsv4Sampler::Device {
                    Some(m.gpu.device_sampler().map_err(EngineError::engine)?)
                } else {
                    None
                };
            let mut row0 = pre_logits;
            let mut t = if let Some(sampler) = &mut device_sampler {
                sampler.sample_host_row(&row0, p0, &cfg, &window, pen_cfg.as_ref())
            } else {
                draw(&mut row0, p0, &window)
            }
            .map_err(EngineError::engine)?;
            let mut step = 0usize;
            // Host-sampled rows coalesce too. On PP the device sampler keeps its own step; on
            // TP/EP its unpenalized rows draw on the device inside the B-row step (memra #710),
            // and a penalized one asks for its row and draws it with the same program.
            let batcher = m
                .rows
                .as_deref()
                .filter(|_| device_sampler.is_none() || m.gpu.topology().is_tp_ep());
            let _member = batcher.map(RowMember::join);
            // A TP/EP device-sampled row at the vendor default (temperature 1, top-p 1, no
            // top-k) without penalties replays its steps: the in-graph draw is the device
            // sampler's position-keyed program (memra #710).
            let mut replay = m.gpu.topology().is_tp_ep()
                && device_sampler.is_some()
                && pen_cfg.is_none()
                && cfg.temperature == 1.0
                && cfg.top_p == 1.0
                && cfg.top_k == 0
                && arm_replay(m, &mut state, cfg);
            while emit.push(&[t]) {
                if pen_cfg.is_some() {
                    window.push(t);
                }
                step += 1;
                if step >= budget {
                    break;
                }
                replay = replay && keep_replay(m, &mut state).map_err(EngineError::engine)?;
                t = if replay && batcher.is_none() {
                    m.gpu.decode_sample_full_token(t, &mut state)
                } else if let (Some(b), Some(sampler)) = (batcher, &mut device_sampler) {
                    if pen_cfg.is_some() {
                        let row = rows_step(m, b, turn, t, RowAsk::Logits, &mut state)
                            .map_err(EngineError::engine)?
                            .logits
                            .ok_or_else(|| EngineError::engine("B-row logits missing"))?;
                        sampler.sample_host_row(&row, state.pos, &cfg, &window, pen_cfg.as_ref())
                    } else {
                        rows_step(m, b, turn, t, RowAsk::Draw(cfg), &mut state).map(|out| out.tok)
                    }
                } else if let Some(sampler) = &mut device_sampler {
                    m.gpu
                        .decode_step_device_logits(t, &mut state)
                        .map_err(EngineError::engine)?;
                    m.gpu
                        .sample_device_logits(&state, sampler, &cfg, &window, pen_cfg.as_ref())
                } else {
                    let mut row = match batcher {
                        Some(b) => rows_step(m, b, turn, t, RowAsk::Logits, &mut state)
                            .map_err(EngineError::engine)?
                            .logits
                            .ok_or_else(|| EngineError::engine("B-row logits missing"))?,
                        None => logits_step(m, turn, t, &mut state).map_err(EngineError::engine)?,
                    };
                    draw(&mut row, p0 + step, &window)
                }
                .map_err(EngineError::engine)?;
            }
        }
        state_to_park = state;
    }

    let park_toks = if short_monolithic {
        eprintln!(
            "[dsv4-host] skip park: short monolithic prime ({} <= chunk {}); numeric-regime isolation",
            prompt.len(),
            m.prefill_chunk
        );
        None
    } else {
        match processed_prefix_tokens(&prompt, &emit.ids, emit.terminal, state_to_park.pos) {
            Ok(toks) => Some(toks),
            Err(err) => {
                // A speculative round commits before its callback. If an EOS/stop lands inside that
                // round, state may be ahead of the user-visible stream; without a semantic rollback
                // that state is not cacheable, and pretending otherwise would poison the next turn.
                eprintln!("[dsv4-host] skip park: {err}");
                None
            }
        }
    };
    let stats = ServeStats {
        tokens_out: emit.ids.len(),
        n_prompt: prompt.len(),
        n_cached,
    };
    emit.finish(
        prompt.len(),
        n_cached,
        t0.elapsed().as_secs_f64(),
        spec_usage,
    );
    if let Some(toks) = park_toks {
        park_prefix(
            m,
            turn.cache(),
            req,
            toks,
            &state_to_park,
            dstate_to_park.as_ref(),
        );
    }
    Ok(Served::Done(stats))
}

fn argmax(v: &[f32]) -> u32 {
    let mut best = 0usize;
    for i in 1..v.len() {
        if v[i] > v[best] {
            best = i;
        }
    }
    best as u32
}

#[cfg(test)]
mod spec_policy_tests {
    use super::{Dsv4SpecPolicy, resolve_spec_policy};
    use memra_engine::dsv4_gpu::Dsv4Vt;

    /// Unset knobs are the drafter's own depth and no window; a valid depth and `slot` pass
    /// through. Anything else refuses at load and names its value, so a typo cannot boot a
    /// server that fails every spec request (`slot@0.5` did exactly that on 2026-09-24).
    #[test]
    fn a_bad_spec_knob_refuses_by_name() {
        assert_eq!(
            resolve_spec_policy(None, None, None, None),
            Ok(Dsv4SpecPolicy {
                depth_cap: usize::MAX,
                vt: Dsv4Vt::Off
            })
        );
        assert_eq!(
            resolve_spec_policy(Some("4"), Some("off"), None, None).map(|p| p.depth_cap),
            Ok(4)
        );
        assert!(matches!(
            resolve_spec_policy(None, Some("slot"), None, None).map(|p| p.vt),
            Ok(Dsv4Vt::Slot { floor: 0, .. })
        ));
        for depth in ["0", "", "four", "-1"] {
            let err = resolve_spec_policy(Some(depth), None, None, None).unwrap_err();
            assert!(err.contains("MEMRA_DSV4_SPEC_DEPTH"), "{depth:?}: {err}");
        }
        let err = resolve_spec_policy(None, Some("slot@0.5"), None, None).unwrap_err();
        assert!(err.contains("slot@0.5"), "{err}");
        let err = resolve_spec_policy(None, None, Some("0.5"), None).unwrap_err();
        assert!(err.contains("MEMRA_DSV4_VT_TAU"), "{err}");
    }
}

#[cfg(test)]
mod calibration_tests {
    use super::{carrier, fixed_delta};
    use crate::dsv4_admit::per_device;
    use memra_engine::dsv4_gpu::StageMemory;

    fn reading(dev: usize, occupied: u64) -> StageMemory {
        StageMemory {
            dev,
            driver_free: 1000 - occupied,
            total: 1000,
            pool_reserved: 0,
            pool_used: 0,
        }
    }

    /// Two stages on card 0 and one on card 1. Each reading is device-wide, so card 0's delta
    /// (300) covers both of its stages' caches (100 + 50) and one fixed term: the card is
    /// charged its delta once, not once per stage.
    #[test]
    fn co_located_stages_share_one_card_delta() {
        let base = [reading(0, 100), reading(0, 100), reading(1, 40)];
        let now = [reading(0, 400), reading(0, 400), reading(1, 120)];
        let cache = [100, 50, 60];
        let fixed = fixed_delta(&now, &base, &cache);
        assert_eq!(fixed, vec![150, 0, 20]);
        assert_eq!((carrier(&now, 1), carrier(&now, 2)), (0, 2));
        let need: Vec<u64> = cache.iter().zip(&fixed).map(|(c, f)| c + f).collect();
        let cards = per_device(&[0, 0, 1], &need, &[900, 900, 880]);
        assert_eq!(
            cards.iter().map(|d| (d.dev, d.need)).collect::<Vec<_>>(),
            vec![(0, 300), (1, 80)],
            "each card's charge is its measured delta"
        );
    }
}

#[cfg(test)]
mod host_reclaim_tests {
    use super::lru_victim;
    use std::time::{Duration, Instant};

    /// The admission reclaim evicts least-recently-used first, breaks a tie by id, and never
    /// picks the entry the waiting request would restore from.
    #[test]
    fn the_victim_is_the_oldest_entry_other_than_the_spared_one() {
        let t0 = Instant::now();
        let old = t0;
        let newer = t0 + Duration::from_millis(5);
        let entries = [
            ("ns-a", 0, newer, 7),
            ("ns-b", 1, old, 9),
            ("ns-b", 2, old, 3),
        ];
        let pick = |spare| lru_victim(entries.iter().copied(), spare);
        assert_eq!(
            pick(None),
            Some(("ns-b".to_string(), 2)),
            "oldest, then lowest id"
        );
        assert_eq!(pick(Some(3)), Some(("ns-b".to_string(), 1)));
        assert_eq!(
            lru_victim(entries[..1].iter().copied(), Some(7)),
            None,
            "the spared entry alone is never evicted"
        );
        assert_eq!(lru_victim(std::iter::empty(), None), None);
    }
}

#[cfg(test)]
mod declared_context_tests {
    use super::{DeclaredContext, declared_context_from_config, dsv4_declared_context};
    use crate::worker::{ENGINE_MAX_CTX, resolve_ctx};

    fn cfg(raw: &str) -> serde_json::Value {
        serde_json::from_str(raw).expect("fixture parses")
    }

    /// CASE 1, a checkpoint that DECLARES a window the engine can carry: the route serves that
    /// window. RED ARM for the shipped defect - with the old literal fallback this resolved 8192
    /// and published it, against a checkpoint declaring a million positions.
    #[test]
    fn the_dsv4_route_serves_the_window_the_checkpoint_declares() {
        let dsv4f = cfg(
            r#"{"model_type":"deepseek_v4","max_position_embeddings":1048576,
                "hidden_size":7168,"num_hidden_layers":61}"#,
        );
        assert_eq!(
            declared_context_from_config(&dsv4f),
            DeclaredContext::Positions(1_048_576)
        );
        assert_eq!(resolve_ctx(None, 1_048_576), Ok(1_048_576));

        // An explicit deployment pin still wins, and a smaller checkpoint is not widened.
        assert_eq!(resolve_ctx(Some("131072"), 1_048_576), Ok(131_072));
        assert_eq!(
            declared_context_from_config(&cfg(r#"{"max_position_embeddings":4096}"#)),
            DeclaredContext::Positions(4_096)
        );
        assert_eq!(resolve_ctx(None, 4_096), Ok(4_096));
    }

    /// CASE 2, a checkpoint that declares NOTHING: absent and null are the ambiguity, and the
    /// route refuses as undeclared rather than filling a number in.
    #[test]
    fn a_checkpoint_that_declares_nothing_refuses_as_undeclared() {
        for raw in [r#"{}"#, r#"{"max_position_embeddings":null}"#] {
            assert_eq!(
                declared_context_from_config(&cfg(raw)),
                DeclaredContext::Absent,
                "{raw}"
            );
        }
        let refusal = resolve_ctx(None, 0).expect_err("undeclared must refuse");
        assert!(
            refusal.contains("undeclared"),
            "the undeclared refusal must say so, not blend into the others: {refusal}",
        );
        // An explicit pin is how an operator answers the ambiguity.
        assert_eq!(resolve_ctx(Some("262144"), 0), Ok(262_144));
    }

    /// CASE 2b, a declaration that is PRESENT but not a count of positions. Distinct from absent:
    /// the artifact is malformed rather than silent, and the message names the value.
    #[test]
    fn a_present_but_unusable_declaration_refuses_and_names_the_value() {
        for raw in [
            r#"{"max_position_embeddings":"1048576"}"#,
            r#"{"max_position_embeddings":-1}"#,
            r#"{"max_position_embeddings":0}"#,
            r#"{"max_position_embeddings":4096.5}"#,
            r#"{"max_position_embeddings":[1048576]}"#,
        ] {
            match declared_context_from_config(&cfg(raw)) {
                DeclaredContext::Unusable(_) => {}
                other => panic!("{raw} must classify as unusable, got {other:?}"),
            }
        }
    }

    /// CASE 3, a checkpoint that declares something THE ENGINE CANNOT HONOUR. The mainline model
    /// config carries `context_length` as a u32, so a wider window cannot survive the engine's own
    /// representation; the dsv4 route reads config.json as u64 where nothing else would catch it.
    /// Refusing is the point: serving a narrower window than the checkpoint declares, without
    /// saying so, is the defect this whole resolver exists to prevent.
    #[test]
    fn a_declaration_beyond_the_engine_ceiling_refuses_instead_of_narrowing() {
        assert_eq!(
            declared_context_from_config(&cfg(r#"{"max_position_embeddings":1099511627776}"#)),
            DeclaredContext::Positions(1_099_511_627_776),
        );
        let refusal = resolve_ctx(None, 1_099_511_627_776)
            .expect_err("a beyond-ceiling declaration must refuse");
        assert!(
            refusal.contains("ceiling") && refusal.contains("1099511627776"),
            "the ceiling refusal must name the ceiling and the declaration: {refusal}",
        );
        assert_ne!(
            resolve_ctx(None, 1_099_511_627_776),
            Ok(ENGINE_MAX_CTX),
            "clamping to the ceiling would be exactly the silent narrowing this forbids",
        );
        // The boundary itself is servable; one position past it is not.
        assert_eq!(resolve_ctx(None, ENGINE_MAX_CTX), Ok(ENGINE_MAX_CTX));
        assert!(resolve_ctx(None, ENGINE_MAX_CTX + 1).is_err());
        // An explicit MEMRA_CTX is subject to the same ceiling.
        assert!(resolve_ctx(Some(&(ENGINE_MAX_CTX + 1).to_string()), 262_144).is_err());
        assert_eq!(
            resolve_ctx(Some(&ENGINE_MAX_CTX.to_string()), 262_144),
            Ok(ENGINE_MAX_CTX)
        );
    }

    /// The three refusals must be TELLABLE APART. A gate that cannot distinguish "declares
    /// nothing" from "declares something impossible" hides which one an operator is looking at.
    #[test]
    fn the_three_cases_produce_three_distinguishable_outcomes() {
        let served = resolve_ctx(None, 1_048_576).expect("declared and carryable");
        let undeclared = resolve_ctx(None, 0).expect_err("undeclared");
        let beyond = resolve_ctx(None, ENGINE_MAX_CTX + 1).expect_err("beyond ceiling");
        let unusable_env = resolve_ctx(Some("abc"), 1_048_576).expect_err("unusable MEMRA_CTX");

        assert_eq!(served, 1_048_576);
        assert!(undeclared.contains("undeclared") && !undeclared.contains("ceiling"));
        assert!(beyond.contains("ceiling") && !beyond.contains("undeclared"));
        assert!(unusable_env.contains("MEMRA_CTX") && !unusable_env.contains("ceiling"));
        assert_ne!(undeclared, beyond);
        assert_ne!(undeclared, unusable_env);
        assert_ne!(beyond, unusable_env);
    }

    /// The reader is the FILE, not a hand-passed number: a real directory resolves, a missing or
    /// malformed `config.json` refuses by name, and a malformed declaration refuses naming the
    /// value instead of substituting one.
    #[test]
    fn the_config_file_itself_is_the_source_and_a_broken_one_refuses() {
        let dir = std::env::temp_dir().join(format!(
            "memra-dsv4-ctx-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("fixture dir");

        assert!(
            dsv4_declared_context(&dir)
                .expect_err("no config.json")
                .contains("unreadable"),
        );

        std::fs::write(dir.join("config.json"), "{not json").expect("write");
        assert!(
            dsv4_declared_context(&dir)
                .expect_err("malformed config")
                .contains("not JSON"),
        );

        std::fs::write(
            dir.join("config.json"),
            r#"{"max_position_embeddings":"1048576"}"#,
        )
        .expect("write");
        let refusal = dsv4_declared_context(&dir).expect_err("unusable declaration");
        assert!(
            refusal.contains("max_position_embeddings") && refusal.contains("1048576"),
            "the refusal must name the field and the value it refused: {refusal}",
        );

        std::fs::write(dir.join("config.json"), r#"{}"#).expect("write");
        assert_eq!(
            dsv4_declared_context(&dir),
            Ok(0),
            "an absent declaration reads as 0 so the resolver refuses it as undeclared",
        );

        std::fs::write(
            dir.join("config.json"),
            r#"{"model_type":"deepseek_v4","max_position_embeddings":1048576}"#,
        )
        .expect("write");
        assert_eq!(dsv4_declared_context(&dir), Ok(1_048_576));

        std::fs::remove_dir_all(&dir).expect("cleanup");
    }
}

#[cfg(test)]
mod c4_host_budget_tests {
    use super::{
        admit_c4_host_bytes, default_sessions, env_text, resolve_c4_host_bytes, resolve_env_mb,
        resolve_sessions, resolve_topology,
    };
    use memra_engine::dsv4_topology::Dsv4Placement;
    use std::ffi::OsString;

    /// The replay round-up never passes the model context (revuto on #727): a boot with a
    /// context below 512 keeps each session at its own need and serves eager.
    #[test]
    fn the_replay_round_up_stays_inside_the_context() {
        use super::session_capacity;
        assert_eq!(session_capacity(true, 300, 1_048_576), 512);
        assert_eq!(session_capacity(true, 900, 1_048_576), 900);
        assert_eq!(session_capacity(true, 300, 400), 400);
        assert_eq!(session_capacity(true, 300, 300), 300);
        assert_eq!(session_capacity(false, 300, 1_048_576), 300);
    }

    /// TP/EP with attention TP2 is the default placement (memra #710); `pp` is the rollback,
    /// and any other value, the retired `tp_ep_attn` measurement name included, refuses.
    #[test]
    fn the_topology_defaults_to_tp_ep_and_pp_is_the_rollback() {
        for raw in [None, Some(""), Some("tp_ep"), Some(" tp_ep ")] {
            assert_eq!(
                resolve_topology(raw),
                Ok(Dsv4Placement::TpEp { attention_tp: true }),
                "{raw:?}"
            );
        }
        assert_eq!(resolve_topology(Some("pp")), Ok(Dsv4Placement::Pp));
        for raw in ["tp_ep_attn", "tp", "PP", "pp2"] {
            let err = resolve_topology(Some(raw)).unwrap_err();
            assert!(err.contains("MEMRA_DSV4_TOPOLOGY"), "{raw}: {err}");
        }
    }

    #[test]
    fn serving_lanes_resolve_literally() {
        for default in [1, 2] {
            for raw in [None, Some("")] {
                assert_eq!(resolve_sessions(raw, default), Ok(default));
            }
            for raw in [Some("1"), Some(" 1 ")] {
                assert_eq!(resolve_sessions(raw, default), Ok(1));
            }
            for n in 2..=4 {
                assert_eq!(resolve_sessions(Some(&n.to_string()), default), Ok(n));
            }
            for raw in ["0", "5", "two", "-1", "2.0"] {
                assert!(resolve_sessions(Some(raw), default).is_err(), "{raw}");
            }
        }
    }

    #[test]
    fn the_turn_is_served_in_arrival_order() {
        use super::{Dsv4HostCache, Turn, TurnLock};
        use std::sync::{Arc, Mutex};
        let lock = Arc::new(TurnLock::new(Dsv4HostCache::new(0)));
        let order = Arc::new(Mutex::new(Vec::new()));
        let mut a = Turn::take(&lock);
        let b = {
            let (lock, order) = (lock.clone(), order.clone());
            std::thread::spawn(move || {
                let _b = Turn::take(&lock);
                order.lock().unwrap().push("b");
            })
        };
        // B holds ticket 1 and waits.
        while lock.tickets.lock().unwrap().0 < 2 {
            std::thread::yield_now();
        }
        // A gives the turn up and asks at once, as the chunked prefill's yield does: it must
        // queue behind B, not take the turn straight back.
        a.release();
        a.acquire();
        order.lock().unwrap().push("a");
        b.join().unwrap();
        assert_eq!(*order.lock().unwrap(), ["b", "a"]);
    }

    #[test]
    fn a_dropped_or_unwound_turn_is_given_up() {
        use super::{Dsv4HostCache, Turn, TurnLock};
        let lock = TurnLock::new(Dsv4HostCache::new(0));
        {
            let mut t = Turn::take(&lock);
            let _ = t.cache();
        }
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _t = Turn::take(&lock);
            panic!("request fault");
        }));
        assert!(r.is_err());
        // Both holds gave the turn up: a third take does not block.
        let _t = Turn::take(&lock);
        assert_eq!(*lock.tickets.lock().unwrap(), (3, 2));
    }

    #[test]
    fn the_memory_door_sleeps_without_the_turn() {
        use super::{Dsv4HostCache, Turn, TurnLock, sleep_without_turn};
        use std::sync::Arc;
        use std::sync::atomic::{AtomicBool, Ordering};
        let lock = Arc::new(TurnLock::new(Dsv4HostCache::new(0)));
        let stepped = Arc::new(AtomicBool::new(false));
        let mut door = Turn::take(&lock);
        let other = {
            let (lock, stepped) = (lock.clone(), stepped.clone());
            std::thread::spawn(move || {
                let _t = Turn::take(&lock);
                stepped.store(true, Ordering::SeqCst);
            })
        };
        while lock.tickets.lock().unwrap().0 < 2 {
            std::thread::yield_now();
        }
        // The in-flight lane gets its step in while the door waits for memory.
        sleep_without_turn(&mut door, std::time::Duration::from_millis(50));
        assert!(stepped.load(Ordering::SeqCst));
        other.join().unwrap();
    }

    #[test]
    fn b_row_width_resolves_literally() {
        use super::resolve_rows;
        // Unset takes the load's default (1 on PP-2, the lane count on TP/EP); an explicit
        // value wins over it.
        for default in [1usize, 2, 4] {
            for raw in [None, Some("")] {
                assert_eq!(resolve_rows(raw, default), Ok(default));
            }
            for raw in [Some("0"), Some("1")] {
                assert_eq!(resolve_rows(raw, default), Ok(1));
            }
            for n in 2..=8 {
                assert_eq!(resolve_rows(Some(&n.to_string()), default), Ok(n));
            }
            for raw in ["9", "two", "-1", "2.5"] {
                assert!(resolve_rows(Some(raw), default).is_err(), "{raw}");
            }
        }
    }

    /// TP/EP lanes only help by sharing B-row steps, so the plain route gets two and a
    /// drafter route one (memra #710).
    #[test]
    fn tp_ep_lanes_default_to_two_on_the_plain_route() {
        use super::default_tp_ep_sessions;
        assert_eq!(default_tp_ep_sessions(true, false), 2);
        assert_eq!(default_tp_ep_sessions(true, true), 1);
        assert_eq!(default_tp_ep_sessions(false, false), 1);
    }

    /// (batch widths in run order, each lane's per-step results, each lane's final counter)
    type CoalesceOutcome = (Vec<usize>, Vec<Vec<Result<u32, String>>>, Vec<u32>);

    /// Runs `lanes` threads that each take `steps` coalesced steps on their own counter.
    /// The batch runner records each batch's width and bumps every row's counter once.
    fn coalesce_run(
        bmax: usize,
        lanes: usize,
        steps: usize,
        fail_at: Option<usize>,
    ) -> CoalesceOutcome {
        use super::{Coalescer, RowOut};
        use std::sync::{Arc, Mutex};
        let core = Arc::new(Coalescer::<u32>::new(bmax, 1));
        let widths = Arc::new(Mutex::new(Vec::new()));
        let barrier = Arc::new(std::sync::Barrier::new(lanes));
        let handles: Vec<_> = (0..lanes)
            .map(|lane| {
                let (core, widths, barrier) = (core.clone(), widths.clone(), barrier.clone());
                std::thread::spawn(move || {
                    core.join();
                    barrier.wait();
                    let mut counter = 0u32;
                    let mut out = Vec::new();
                    for step in 0..steps {
                        let tok = (lane * 1000 + step) as u32;
                        // Odd lanes ask for logits, so batches mix both kinds of row.
                        let want = lane % 2 == 1;
                        let r = core.step(tok, want, &mut counter, &mut |toks, wants, states| {
                            let mut w = widths.lock().unwrap();
                            w.push(toks.len());
                            if fail_at == Some(w.len() - 1) {
                                return Err("injected".into());
                            }
                            for s in states.iter_mut() {
                                **s += 1;
                            }
                            Ok(toks
                                .iter()
                                .zip(wants)
                                .map(|(t, &w)| RowOut {
                                    tok: t + 7,
                                    logits: w.then(|| vec![*t as f32]),
                                })
                                .collect())
                        });
                        // Each row gets its own token, and its own logits exactly when it asked.
                        if let Ok(o) = &r {
                            assert_eq!(o.logits, want.then(|| vec![tok as f32]));
                        }
                        out.push(r.map(|o| o.tok));
                    }
                    core.leave();
                    (out, counter)
                })
            })
            .collect();
        let mut results = Vec::new();
        let mut counters = Vec::new();
        for h in handles {
            let (out, counter) = h.join().unwrap();
            results.push(out);
            counters.push(counter);
        }
        let w = widths.lock().unwrap().clone();
        (w, results, counters)
    }

    #[test]
    fn coalesced_rows_each_get_their_own_token_once_per_step() {
        let (widths, results, counters) = coalesce_run(4, 3, 20, None);
        for (lane, out) in results.iter().enumerate() {
            for (step, r) in out.iter().enumerate() {
                assert_eq!(*r, Ok((lane * 1000 + step) as u32 + 7));
            }
        }
        // Every row of every step ran exactly once.
        assert_eq!(counters, vec![20, 20, 20]);
        assert_eq!(widths.iter().sum::<usize>(), 60);
        assert!(widths.iter().all(|&w| (1..=3).contains(&w)));
        // Three members keep batches full most of the time; a partial batch only waits out
        // the window.
        assert!(
            widths.iter().filter(|&&w| w == 3).count() >= 10,
            "{widths:?}"
        );
    }

    #[test]
    fn two_groups_split_the_lanes_in_half() {
        use super::{Coalescer, RowOut};
        use std::sync::{Arc, Mutex};
        let core = Arc::new(Coalescer::<u32>::new(4, 2));
        let widths = Arc::new(Mutex::new(Vec::new()));
        for lanes in [2usize, 4] {
            let barrier = Arc::new(std::sync::Barrier::new(lanes));
            let handles: Vec<_> = (0..lanes)
                .map(|lane| {
                    let (core, widths, barrier) = (core.clone(), widths.clone(), barrier.clone());
                    std::thread::spawn(move || {
                        core.join();
                        barrier.wait();
                        let mut c = 0u32;
                        for step in 0..8u32 {
                            core.step(step, false, &mut c, &mut |toks, _, _| {
                                widths.lock().unwrap().push((lanes, toks.len()));
                                std::thread::sleep(std::time::Duration::from_millis(1));
                                Ok(toks
                                    .iter()
                                    .map(|&tok| RowOut { tok, logits: None })
                                    .collect())
                            })
                            .unwrap();
                        }
                        barrier.wait();
                        core.leave();
                        let _ = lane;
                    })
                })
                .collect();
            for h in handles {
                h.join().unwrap();
            }
        }
        let w = widths.lock().unwrap();
        // Two lanes never batch together (two pipelined one-row steps); four lanes batch in
        // pairs, never four at once.
        assert!(w.iter().filter(|x| x.0 == 2).all(|x| x.1 == 1), "{w:?}");
        assert!(w.iter().filter(|x| x.0 == 4).all(|x| x.1 <= 2), "{w:?}");
        assert!(w.iter().filter(|x| x.0 == 4).any(|x| x.1 == 2), "{w:?}");
    }

    #[test]
    fn a_batch_never_exceeds_its_width() {
        let (widths, results, counters) = coalesce_run(2, 5, 12, None);
        assert!(widths.iter().all(|&w| w <= 2), "{widths:?}");
        assert_eq!(counters, vec![12; 5]);
        assert!(results.iter().flatten().all(|r| r.is_ok()));
    }

    #[test]
    fn a_failed_batch_fails_every_row_in_it_and_only_them() {
        let (widths, results, _) = coalesce_run(4, 3, 6, Some(2));
        let failed: usize = results
            .iter()
            .flatten()
            .filter(|r| r.as_ref().err().map(String::as_str) == Some("injected"))
            .count();
        assert_eq!(failed, widths[2], "every row of batch 2 and no other");
    }

    /// A step that panics fails every row that lent it a state and lowers `in_flight`, so the
    /// other lanes get an error instead of waiting forever, and the next batch still runs.
    #[test]
    fn a_panicking_batch_fails_its_rows_and_the_next_batch_runs() {
        use super::Coalescer;
        use std::sync::Arc;
        let core = Arc::new(Coalescer::<u32>::new(2, 1));
        core.join();
        core.join();
        let lanes: Vec<_> = (0..2u32)
            .map(|lane| {
                let core = core.clone();
                std::thread::spawn(move || {
                    let mut c = 0u32;
                    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        core.step(lane, false, &mut c, &mut |_, _, _| panic!("injected panic"))
                            .map(|o| o.tok)
                    }))
                })
            })
            .collect();
        let outcomes: Vec<_> = lanes.into_iter().map(|h| h.join().unwrap()).collect();
        let panicked = outcomes.iter().filter(|o| o.is_err()).count();
        let failed = outcomes
            .iter()
            .filter(|o| matches!(o, Ok(Err(e)) if e == "B-row step panicked"))
            .count();
        assert_eq!(
            panicked + failed,
            2,
            "the leader unwinds, every other row gets the error"
        );
        assert!(panicked >= 1, "the leader's own panic is not swallowed");
        let mut c = 0u32;
        core.leave();
        let next = core.step(7, false, &mut c, &mut |toks, _, _| {
            Ok(toks
                .iter()
                .map(|&tok| super::RowOut { tok, logits: None })
                .collect())
        });
        assert_eq!(next.map(|o| o.tok), Ok(7), "in_flight came back down");
    }

    #[test]
    fn a_leaving_lane_completes_the_waiting_batch() {
        use super::Coalescer;
        use std::sync::Arc;
        let core = Arc::new(Coalescer::<u32>::new(4, 1));
        core.join();
        core.join();
        let waiter = {
            let core = core.clone();
            std::thread::spawn(move || {
                let mut c = 0u32;
                let t0 = std::time::Instant::now();
                let r = core
                    .step(5, false, &mut c, &mut |toks, _, _| {
                        Ok(toks
                            .iter()
                            .map(|&tok| super::RowOut { tok, logits: None })
                            .collect())
                    })
                    .map(|o| o.tok);
                (r, t0.elapsed())
            })
        };
        std::thread::sleep(std::time::Duration::from_micros(50));
        core.leave();
        let (r, _) = waiter.join().unwrap();
        assert_eq!(r, Ok(5));
    }

    #[test]
    fn two_lanes_are_the_default_only_on_the_plain_pipelined_program() {
        assert_eq!(default_sessions(true, false), 2);
        assert_eq!(
            default_sessions(true, true),
            1,
            "a DSpark route holds the turn per request"
        );
        assert_eq!(default_sessions(false, false), 1);
        assert_eq!(default_sessions(false, true), 1);
    }

    #[test]
    fn budget_is_opt_in_and_strictly_parsed() {
        assert_eq!(resolve_c4_host_bytes(None), Ok(0));
        assert_eq!(resolve_c4_host_bytes(Some("0")), Ok(0));
        assert_eq!(resolve_c4_host_bytes(Some(" 16 ")), Ok(16 * 1024 * 1024));
        for raw in ["", "-1", "nan", "1.5"] {
            assert!(resolve_c4_host_bytes(Some(raw)).is_err(), "{raw:?}");
        }
        assert!(resolve_c4_host_bytes(Some(&usize::MAX.to_string())).is_err());
    }

    #[test]
    fn budget_sums_both_stages_and_refuses_before_allocation() {
        assert_eq!(admit_c4_host_bytes(100, &[60, 40]), Ok(100));
        assert_eq!(admit_c4_host_bytes(101, &[60, 40]), Ok(100));
        assert_eq!(admit_c4_host_bytes(100, &[0, 0]), Ok(0));
        assert!(
            admit_c4_host_bytes(99, &[60, 40])
                .unwrap_err()
                .contains("no device-history fallback")
        );
        assert!(admit_c4_host_bytes(usize::MAX, &[u64::MAX, 1]).is_err());
        assert_eq!(admit_c4_host_bytes(0, &[u64::MAX, 1]), Ok(0));
    }

    #[test]
    fn boot_environment_helpers_distinguish_absent_malformed_and_non_unicode() {
        assert_eq!(env_text("MEMRA_DSV4_KV_HOST_MB", None), Ok(None));
        assert_eq!(
            env_text("MEMRA_DSV4_KV_HOST_MB", Some(OsString::from(" 16 "))),
            Ok(Some(" 16 ".to_string()))
        );
        assert_eq!(resolve_env_mb("MEMRA_DSV4_KV_HOST_MB", None), Ok(0));
        assert_eq!(
            resolve_env_mb("MEMRA_DSV4_KV_HOST_MB", Some(OsString::from("16"))),
            Ok(16 * 1024 * 1024)
        );
        for raw in ["", "-1", "nan", "1.5"] {
            let err =
                resolve_env_mb("MEMRA_DSV4_KV_HOST_MB", Some(OsString::from(raw))).unwrap_err();
            assert!(err.contains("MEMRA_DSV4_KV_HOST_MB"), "{err}");
        }
        #[cfg(unix)]
        {
            use std::os::unix::ffi::OsStringExt;
            let err = env_text(
                "MEMRA_DSV4_PREFILL_CHUNK",
                Some(OsString::from_vec(vec![0xff, b'1'])),
            )
            .unwrap_err();
            assert!(err.contains("MEMRA_DSV4_PREFILL_CHUNK"), "{err}");
        }
    }
}

#[cfg(test)]
mod prefill_chunk_flag_tests {
    use super::{DSV4_BATCH_WIDTH_MAX, resolve_prefill_chunk, use_chunked_prefill};

    #[test]
    fn the_default_is_the_kernel_width_and_values_are_strictly_bounded() {
        // The measured default, not a door: 512 is the kernel's own ceiling and
        // the top of the EP=off width curve.
        assert_eq!(resolve_prefill_chunk(None, 1_048_576), Ok(512));
        assert_eq!(DSV4_BATCH_WIDTH_MAX, 512);
        // Widths the old 64 ceiling refused are admitted now, and the kernel's
        // own ceiling is still a refusal rather than a clamp.
        assert_eq!(resolve_prefill_chunk(Some("128"), 1_048_576), Ok(128));
        assert_eq!(resolve_prefill_chunk(Some("512"), 1_048_576), Ok(512));
        assert!(resolve_prefill_chunk(Some("513"), 1_048_576).is_err());
        // Red arm for the ceiling itself: a build that let the serving ceiling
        // drift back under the kernel's would fail here rather than silently
        // refusing a width the kernel admits.
        for width in [64usize, 128, 256, 512] {
            assert_eq!(
                resolve_prefill_chunk(Some(&width.to_string()), 1_048_576),
                Ok(width)
            );
        }
        // A short context still bounds the default rather than refusing it.
        assert_eq!(resolve_prefill_chunk(None, 64), Ok(64));
        assert_eq!(resolve_prefill_chunk(Some("0"), 1_048_576), Ok(0));
        assert_eq!(resolve_prefill_chunk(Some("64"), 1_048_576), Ok(64));
        assert!(resolve_prefill_chunk(Some("-1"), 1_048_576).is_err());
        assert!(resolve_prefill_chunk(Some("banana"), 1_048_576).is_err());
        assert!(resolve_prefill_chunk(Some("65"), 64).is_err());
    }

    #[test]
    fn prompts_at_or_below_one_chunk_stay_monolithic() {
        assert!(!use_chunked_prefill(0, 10_000));
        assert!(!use_chunked_prefill(32, 1));
        assert!(!use_chunked_prefill(32, 32));
        assert!(use_chunked_prefill(32, 33));
    }
}

#[cfg(test)]
mod unicode_window_tests {
    use super::{RoundTake, processed_prefix_tokens, scan_stop_cut, snap};
    // Only the boundary red arm needs the driver's commit arithmetic, so it is imported here
    // rather than at module scope where the serving build would carry an unused name.
    use memra_engine::dsv4_gpu::round_commit_rows;

    /// The box10 owner-serve panic class: a long generation whose fixed-offset scan
    /// window (len-64) lands INSIDE a multi-byte char ('’', 3 bytes). Both live
    /// panics (fullwidth '｜' in the tools cell, '’' in owner serve) are this shape.
    #[test]
    fn multibyte_scan_window_never_panics_and_cuts_correctly() {
        // text dominated by curly apostrophes so len-64 is essentially never a boundary
        let base: String = "it\u{2019}s ".repeat(200); // 6 bytes/char-group, '’' is 3 bytes
        for extra in 0..8 {
            let text = format!("{}{}", base, "x".repeat(extra));
            let full = format!("{text}tail STOP more");
            let stops = vec!["STOP".to_string()];
            let cut = scan_stop_cut(&text, &full, &stops).expect("stop found");
            assert_eq!(
                &full[..cut],
                format!("{text}tail "),
                "exclusive stop cuts before STOP"
            );
        }
        // snap itself: any index inside '’' walks back to its start
        let s = "a\u{2019}b";
        assert_eq!(snap(s, 2), 1);
        assert_eq!(snap(s, 3), 1);
        assert_eq!(snap(s, 4), 4);
        assert_eq!(snap(s, 99), s.len());
    }

    /// The inclusive parser-closing stops stay in the stream even when the close's
    /// final byte straddles a delta (the S6 tool_calls-null finding).
    #[test]
    fn dsml_close_is_inclusive_and_straddle_safe() {
        let close = "</\u{ff5c}DSML\u{ff5c}tool_calls>";
        // emitted text ends mid-close (missing final '>'), the next delta completes it
        let text = format!("{}{}", "r".repeat(80), &close[..close.len() - 1]);
        let full = format!("{text}>");
        let cut = scan_stop_cut(&text, &full, &[close.to_string()]).expect("close found");
        assert_eq!(
            cut,
            full.len(),
            "inclusive stop keeps the whole close in-stream"
        );
        // and a user stop straddling stays exclusive but never rewinds emitted text
        let text2 = format!("{}{}", "y\u{2019}".repeat(50), "ST");
        let full2 = format!("{text2}OP after");
        let cut2 = scan_stop_cut(&text2, &full2, &["STOP".to_string()]).expect("stop");
        assert_eq!(
            cut2,
            text2.len(),
            "exclusive straddle clamps at emitted text"
        );
    }

    #[test]
    fn parked_token_boundary_accepts_pending_tail_and_refuses_state_ahead() {
        let prompt = [10, 11, 12];
        let emitted = [20, 21, 22];
        assert_eq!(
            processed_prefix_tokens(&prompt, &emitted, None, 5).unwrap(),
            [10, 11, 12, 20, 21],
            "the final emitted token may be pending and belongs to the next suffix"
        );
        assert!(
            processed_prefix_tokens(&prompt, &emitted, None, 7)
                .unwrap_err()
                .contains("stream committed only 3"),
            "spec state ahead of the visible stream must not enter the host tier"
        );
        assert!(
            processed_prefix_tokens(&prompt, &emitted, None, 2)
                .unwrap_err()
                .contains("precedes prompt boundary"),
            "a corrupt pre-prompt state is refused rather than saturating to zero"
        );
    }

    /// RED ARM for memra #495, the defect that kept the AGENT shape out of the parked tier.
    ///
    /// A tool-call turn stops inside a speculative round: the round produced 5 tokens, the
    /// stream took 3 (a stop-string cut on the DSML close), and the driver now commits 3. The
    /// boundary is therefore `prompt ++ emitted` and the session PARKS. Before the fix the
    /// driver committed all 5 and this exact call returned
    /// "device state consumed 5 generated tokens, stream committed only 3", which is what 10
    /// of 10 agent turn-1s hit on the prod candidate while `reasoning.enabled:false` parked
    /// 10 of 10 (darklanes `research/dsv4f-hot-ttft-20260911`, receipts/hot3-r1).
    #[test]
    fn a_tool_call_turn_that_stops_mid_round_parks() {
        let prompt = [10, 11, 12];
        let emitted = [20, 21, 22];
        let n_round = 5;
        let took = RoundTake {
            taken: emitted.len(),
            stop: true,
        };
        let state_pos = prompt.len() + round_commit_rows(took, n_round);
        assert_eq!(
            processed_prefix_tokens(&prompt, &emitted, None, state_pos).unwrap(),
            [10, 11, 12, 20, 21, 22],
            "a stop-string cut inside a round parks at the boundary the stream reached"
        );
        // The old arithmetic, stated so the regression is named rather than implied: commit
        // the whole round and the same boundary call refuses.
        let over_committed = prompt.len() + n_round;
        assert!(
            processed_prefix_tokens(&prompt, &emitted, None, over_committed)
                .unwrap_err()
                .contains("past the stop"),
            "committing the whole round past the stop is exactly what must stay refused"
        );
    }

    /// RED ARM for the defect this lane closes. Before the terminal token was carried, this is
    /// the shape EVERY eos-terminated dsv4 session ended in: the device consumed the EOS the
    /// stream swallowed, `consumed > emitted` fired, and the session refused to park — so the
    /// parked-prefix tier never held a single entry a real conversation could resume from, at
    /// any `MEMRA_DSV4_KV_HOST_MB`. Measured on the prod candidate 2026-09-11 (darklanes
    /// `research/dsv4f-hot-ttft-20260911`): 4 parks against 4 skips, and every skip was an
    /// eos-terminated turn.
    #[test]
    fn an_eos_terminated_session_parks_at_the_boundary_including_the_swallowed_eos() {
        let prompt = [10, 11, 12];
        let emitted = [20, 21];
        assert_eq!(
            processed_prefix_tokens(&prompt, &emitted, Some(1), 6).unwrap(),
            [10, 11, 12, 20, 21, 1],
            "the swallowed EOS is a real token of the device boundary and the next turn's \
             render replays it after the assistant content"
        );
        // Without a terminal token there is nothing to name the extra position with, so the
        // refusal stays — the fix is carrying the id, not trusting the arithmetic.
        assert!(
            processed_prefix_tokens(&prompt, &emitted, None, 6)
                .unwrap_err()
                .contains("swallowed no terminal token"),
            "one past the stream with no recorded terminal token is still unparkable"
        );
        // Two past the stop is a speculative round that ran beyond EOS. Those ids are not in
        // any continuation, so this shape stays refused rather than being papered over.
        assert!(
            processed_prefix_tokens(&prompt, &emitted, Some(1), 7)
                .unwrap_err()
                .contains("past the stop"),
            "state more than the terminal token ahead must not enter the host tier"
        );
    }
}

#[cfg(test)]
mod parked_tier_lookup_tests {
    use super::{
        Candidate, DSV4_HOST_CACHE_MIN_TOKENS, TakeMiss, common_prefix_len, select_prefix,
    };

    fn entry(toks: &[u32]) -> Candidate<'_> {
        Candidate {
            toks,
            has_dspark: true,
            affinity: None,
            id: 0,
        }
    }

    /// The whole chain in miniature: turn 1's prompt plus what it generated is the boundary that
    /// parks, and turn 2 re-renders that same boundary and asks for it.
    fn turn1_boundary(prompt: usize, generated: usize) -> Vec<u32> {
        (0..(prompt + generated) as u32).collect()
    }

    #[test]
    fn the_next_turn_finds_the_boundary_the_previous_turn_parked() {
        let parked = turn1_boundary(DSV4_HOST_CACHE_MIN_TOKENS + 200, 139);
        let mut turn2 = parked.clone();
        turn2.extend(9000..9019); // the tool result and the next generation prompt
        assert_eq!(
            select_prefix([entry(&parked)].into_iter(), None, &turn2, true),
            Ok(0)
        );
    }

    /// RED ARM for the blind spot this lane closes (memra #495).
    ///
    /// The agent shape's turn 2 diverged from the parked boundary at exactly the point where the
    /// assistant turn began: the turn was re-rendered WITHOUT the tool calls the model had
    /// generated, so the parked tier answered a bare `None` and the receipt read exactly like a
    /// tier that was never armed (`hits=0`, `cached_tokens=0`, no line at all). The miss now
    /// names the divergence point, and this asserts it names the RIGHT one.
    #[test]
    fn a_turn_that_rerenders_differently_reports_where_it_diverged() {
        let prompt_len = DSV4_HOST_CACHE_MIN_TOKENS + 200;
        let parked = turn1_boundary(prompt_len, 139);
        // Turn 2 agrees through the prompt and then renders a DIFFERENT assistant turn.
        let mut turn2: Vec<u32> = (0..prompt_len as u32).collect();
        turn2.extend(7000..7158);
        let miss = select_prefix([entry(&parked)].into_iter(), None, &turn2, true)
            .expect_err("a re-rendered assistant turn is not a prefix of the parked boundary");
        assert_eq!(
            miss,
            TakeMiss::NoPrefix {
                candidates: 1,
                best_n: prompt_len + 139,
                best_lcp: prompt_len,
                dspark_short: 0,
            },
            "the miss must name the divergence point, which is the previous turn's prompt length"
        );
    }

    /// A parked entry with no DSpark state cannot serve a spec-armed request, and the miss says
    /// so rather than looking like an empty pool.
    #[test]
    fn a_dspark_request_reports_entries_that_lack_dspark_state() {
        let parked = turn1_boundary(DSV4_HOST_CACHE_MIN_TOKENS + 200, 10);
        let mut turn2 = parked.clone();
        turn2.push(9999);
        let mut plain = entry(&parked);
        plain.has_dspark = false;
        let miss = select_prefix([plain].into_iter(), None, &turn2, true)
            .expect_err("a plain entry cannot answer a DSpark-armed request");
        let TakeMiss::NoPrefix { dspark_short, .. } = miss else {
            panic!("expected a NoPrefix miss, got {miss:?}");
        };
        assert_eq!(dspark_short, 1);
    }

    /// Strictness: an entry equal to the whole prompt leaves no suffix token to feed, so it is
    /// not usable and the miss reports a full-length agreement rather than a hit.
    #[test]
    fn an_exact_length_entry_is_not_a_strict_prefix() {
        let parked = turn1_boundary(DSV4_HOST_CACHE_MIN_TOKENS + 200, 5);
        let miss = select_prefix([entry(&parked)].into_iter(), None, &parked, true)
            .expect_err("no suffix token left to reconstruct logits from");
        let TakeMiss::NoPrefix { best_lcp, .. } = miss else {
            panic!("expected a NoPrefix miss, got {miss:?}");
        };
        assert_eq!(best_lcp, parked.len());
    }

    #[test]
    fn common_prefix_len_stops_at_the_first_difference() {
        assert_eq!(common_prefix_len(&[1, 2, 3], &[1, 2, 9, 3]), 2);
        assert_eq!(common_prefix_len(&[1, 2], &[1, 2, 3]), 2);
        assert_eq!(common_prefix_len(&[], &[1]), 0);
    }
}
