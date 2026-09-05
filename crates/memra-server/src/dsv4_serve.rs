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
//!   - no response_format/grammar (refused by name);
//!   - streaming granularity is the spec ROUND (or every plain token) — the commit
//!     callback seam on the gated drivers, `None` = byte-identical bench behavior.

use crate::worker::{EngineError, Event, ModelCaps, Request, SpecUsage};
use memra_engine::dsv4_gpu::{
    DSV4_BATCH_WIDTH_MAX, DecodeState, DsparkState, Dsv4Gpu, Dsv4HostDecodeState,
    Dsv4HostDsparkState, Dsv4PenaltyCfg, Dsv4SampleCfg, dsv4_penalize_row, dsv4_sample_row,
    resolve_vt,
};
use memra_gguf::dsv4_forward::ActQuantVariant;
use memra_tokenizer::{Tokenizer, chat};
use std::collections::HashMap;
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

pub struct Dsv4Model {
    pub gpu: Arc<Dsv4Gpu>,
    pub tok: Arc<Tokenizer>,
    pub max_seq: usize,
    pub spec: bool,
    pub eos: u32,
    pub host_cache_bytes: usize,
    pub prefill_chunk: usize,
}

fn resolve_prefill_chunk(raw: Option<&str>, max_seq: usize) -> Result<usize, String> {
    let chunk = match raw {
        None => 0,
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
    fn take(
        &mut self,
        cache_ns: &str,
        affinity: Option<&str>,
        prompt: &[u32],
        need_dspark: bool,
    ) -> Option<ParkedEntry> {
        let pool = self.entries.get(cache_ns)?;
        let mut best: Option<(usize, usize, bool, u64)> = None;
        for (i, entry) in pool.iter().enumerate() {
            let n = entry.toks.len();
            if n < DSV4_HOST_CACHE_MIN_TOKENS
                || n >= prompt.len()
                || prompt[..n] != entry.toks[..]
                || (need_dspark && entry.dspark.is_none())
            {
                continue;
            }
            let affinity_match = affinity.is_some() && affinity == entry.affinity.as_deref();
            let candidate = (i, n, affinity_match, entry.id);
            if best.is_none_or(|(_, bn, ba, bid)| {
                n > bn
                    || (n == bn && affinity_match && !ba)
                    || (n == bn && affinity_match == ba && entry.id > bid)
            }) {
                best = Some(candidate);
            }
        }
        let (i, _, _, _) = best?;
        let pool = self
            .entries
            .get_mut(cache_ns)
            .expect("pool survived lookup");
        let entry = pool.swap_remove(i);
        self.total_bytes = self.total_bytes.saturating_sub(entry.bytes);
        if pool.is_empty() {
            self.entries.remove(cache_ns);
        }
        Some(entry)
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
            let victim = self
                .entries
                .iter()
                .flat_map(|(ns, pool)| {
                    pool.iter()
                        .enumerate()
                        .map(move |(i, e)| (e.last_use, e.id, ns.clone(), i))
                })
                .min_by_key(|(last_use, id, _, _)| (*last_use, *id));
            let Some((_, _, ns, i)) = victim else {
                break;
            };
            let pool = self.entries.get_mut(&ns).expect("victim pool exists");
            let dead = pool.swap_remove(i);
            self.total_bytes = self.total_bytes.saturating_sub(dead.bytes);
            if pool.is_empty() {
                self.entries.remove(&ns);
            }
            eprintln!(
                "[dsv4-host] evict LRU: {} tokens, {:.1} MiB (resident {:.1}/{:.1} MiB)",
                dead.toks.len(),
                dead.bytes as f64 / 1048576.0,
                self.total_bytes as f64 / 1048576.0,
                self.budget as f64 / 1048576.0,
            );
        }
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
    let max_seq: usize = std::env::var("MEMRA_CTX")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(8192);
    let host_cache_mb = match std::env::var("MEMRA_DSV4_KV_HOST_MB") {
        Err(_) => 0usize,
        Ok(raw) => raw
            .trim()
            .parse::<usize>()
            .map_err(|_| format!("MEMRA_DSV4_KV_HOST_MB {raw:?} is not a non-negative integer"))?,
    };
    let host_cache_bytes = host_cache_mb
        .checked_mul(1024 * 1024)
        .ok_or_else(|| "MEMRA_DSV4_KV_HOST_MB byte count overflow".to_string())?;
    let prefill_chunk = resolve_prefill_chunk(
        std::env::var("MEMRA_DSV4_PREFILL_CHUNK").ok().as_deref(),
        max_seq,
    )?;
    let gpu = Dsv4Gpu::load(dir, &devices, variant, max_seq)?;
    let spec = gpu.dspark.is_some();
    let eos = tok.eos_id();
    eprintln!(
        "[dsv4-serve] {name}: loaded on devices {devices:?}, contract {variant:?}, \
         max_seq {max_seq}, drafter {}, parked-host-cache {}, chunked-prefill {}",
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
    Ok(Dsv4Model {
        gpu: Arc::new(gpu),
        tok,
        max_seq,
        spec,
        eos,
        host_cache_bytes,
        prefill_chunk,
    })
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

/// Spawn the serving thread; the returned Sender is the model's admission queue.
pub fn spawn(name: String, m: Dsv4Model) -> std::sync::mpsc::Sender<Box<Request>> {
    let (tx, rx) = std::sync::mpsc::channel::<Box<Request>>();
    std::thread::Builder::new()
        .name(format!("dsv4-serve-{name}"))
        .spawn(move || {
            let mut host_cache = Dsv4HostCache::new(m.host_cache_bytes);
            while let Ok(mut req) = rx.recv() {
                // The worker's DSV4 channel is unbounded, so the hard admission reservation
                // remains held until this serving thread actually receives the request. Merely
                // forwarding it from the command channel must not make the queue appear empty.
                crate::worker::release_admission_reservation(req.lane);
                if req.tx.is_closed() {
                    continue; // client gone while queued
                }
                let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    serve_one(&m, &mut host_cache, &mut req)
                }));
                match r {
                    Ok(Ok(())) => {}
                    Ok(Err(err)) => {
                        // constraint-carrying requests wait on the constraint_ready
                        // channel, not the event stream — failing only via tx leaves
                        // the HTTP layer to a 503 compile timeout (rung-3 serve
                        // finding: response_format 503 instead of the named 400).
                        // Same dispatch as the worker's fail_request.
                        if let Some(ready) = req.constraint_ready.take() {
                            let _ = ready.send(Err(err));
                        } else {
                            let _ = req.tx.send(Event::Error(err));
                        }
                    }
                    Err(payload) => {
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
            }
        })
        .expect("spawn dsv4 serve thread");
    tx
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
    budget: usize,
    ids: Vec<u32>,
    text: String,
    stop_reason: Option<&'static str>,
    client_gone: bool,
}

impl<'a> Emit<'a> {
    fn new(
        tok: &'a Tokenizer,
        tx: &'a crate::worker::EventSender,
        eos: Vec<u32>,
        stop_strings: &'a [String],
        budget: usize,
    ) -> Self {
        Emit {
            tok,
            tx,
            eos,
            stop_strings,
            budget,
            ids: Vec::new(),
            text: String::new(),
            stop_reason: None,
            client_gone: false,
        }
    }

    /// Feed newly committed token ids. Returns false when generation must stop
    /// (EOS / stop string / budget / client disconnect).
    fn push(&mut self, new: &[u32]) -> bool {
        for &id in new {
            if self.stop_reason.is_some() || self.client_gone {
                return false;
            }
            if self.eos.contains(&id) {
                self.stop_reason = Some("stop");
                return false;
            }
            self.ids.push(id);
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
                return false;
            }
            self.text = full;
            if self.tx.send(Event::Token { id, text: delta }).is_err() {
                self.client_gone = true;
                return false;
            }
            if self.ids.len() >= self.budget {
                self.stop_reason = Some("length");
                return false;
            }
        }
        true
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
        let plain = req.tools_json.is_empty()
            && req.think == chat::ThinkMode::Default
            && req.reasoning_effort.is_none()
            && req
                .chat_turns
                .iter()
                .all(|t| t.role != "tool" && t.tool_calls.is_empty());
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
    }
    Ok(last.expect("non-empty suffix has final logits"))
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
    let entry = host.take(&req.cache_ns, req.affinity.as_deref(), prompt, need_dspark)?;
    let n_cached = entry.toks.len();
    let bytes = entry.bytes;
    let t0 = Instant::now();
    let restored = (|| -> Result<RestoredPrefix, String> {
        let transient_rows = m.prefill_chunk.max(m.gpu.verify_tmax());
        let mut state =
            m.gpu
                .restore_decode_state_for_transient(&entry.trunk, capacity, transient_rows)?;
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
/// next turn. State ahead of the visible stream can happen when a speculative round commits
/// before an EOS/stop callback; that shape is not cacheable without a semantic rollback.
fn processed_prefix_tokens(
    prompt: &[u32],
    emitted: &[u32],
    state_pos: usize,
) -> Result<Vec<u32>, String> {
    let consumed_generated = state_pos.checked_sub(prompt.len()).ok_or_else(|| {
        format!(
            "device state pos {state_pos} precedes prompt boundary {}",
            prompt.len()
        )
    })?;
    if consumed_generated > emitted.len() {
        return Err(format!(
            "device state consumed {consumed_generated} generated tokens, stream committed only {}",
            emitted.len()
        ));
    }
    let mut toks = Vec::with_capacity(prompt.len() + consumed_generated);
    toks.extend_from_slice(prompt);
    toks.extend_from_slice(&emitted[..consumed_generated]);
    Ok(toks)
}

fn serve_one(
    m: &Dsv4Model,
    host_cache: &mut Dsv4HostCache,
    req: &mut Request,
) -> Result<(), EngineError> {
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
    let session_capacity = prompt.len() + budget;
    // A prompt no wider than one configured chunk gets the faster canonical
    // monolithic prime. It is deliberately not parked: a later, longer cold prompt
    // uses the chunked numeric regime, so retaining this state would make cache use
    // choose a different realization. Once prompts exceed the chunk, cold and restored
    // paths are both chunked and cache-transparent.
    let short_monolithic =
        m.prefill_chunk > 0 && !use_chunked_prefill(m.prefill_chunk, prompt.len());
    let mut restored = try_restore_prefix(m, host_cache, req, &prompt, session_capacity, use_spec);
    let n_cached = restored.as_ref().map_or(0, |hit| hit.n_cached);
    let _ = req.tx.send(Event::PromptUsage {
        n_prompt: prompt.len(),
        n_cached,
    });

    let mut eos_set = req.params.eos.clone();
    if !eos_set.contains(&m.eos) {
        eos_set.push(m.eos);
    }
    let mut emit = Emit::new(&m.tok, &req.tx, eos_set, &req.stop_strings, budget);
    let mut spec_usage: Option<SpecUsage> = None;
    let state_to_park: DecodeState;
    let mut dstate_to_park: Option<DsparkState> = None;

    if use_spec {
        // the env-seam depth/window policy — exactly the bench drivers' law
        let depth_cap = std::env::var("MEMRA_DSV4_SPEC_DEPTH")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .filter(|t| *t > 0)
            .unwrap_or(usize::MAX)
            .max(1);
        let vt = resolve_vt(
            std::env::var("MEMRA_DSV4_VT").ok().as_deref(),
            std::env::var("MEMRA_DSV4_VT_TAU").ok().as_deref(),
            std::env::var("MEMRA_DSV4_VT_FLOOR").ok().as_deref(),
        )
        .map_err(EngineError::engine)?;
        let (mut state, mut dstate, initial_logits) = if let Some(hit) = restored.take() {
            (
                hit.state,
                hit.dstate.expect("DSpark restore returned DSpark state"),
                Some(hit.logits),
            )
        } else {
            let transient_rows = m.prefill_chunk.max(m.gpu.verify_tmax());
            let mut state = m
                .gpu
                .alloc_decode_state_for_transient(session_capacity, transient_rows)
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
        let mut cb = |new: &[u32]| emit.push(new);
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
        spec_usage = Some(SpecUsage {
            rounds: run.rounds.len() as u64,
            drafted: run
                .rounds
                .iter()
                .map(|r| r.t_batch.saturating_sub(1) as u64)
                .sum(),
            accepted: run.rounds.iter().map(|r| r.accepts as u64).sum(),
        });
        state_to_park = state;
        dstate_to_park = Some(dstate);
    } else {
        // trunk-only plain loops
        let (mut state, pre_logits) = if let Some(hit) = restored.take() {
            (hit.state, hit.logits)
        } else {
            let transient_rows = m.prefill_chunk.max(m.gpu.verify_tmax());
            let mut state = m
                .gpu
                .alloc_decode_state_for_transient(session_capacity, transient_rows)
                .map_err(EngineError::engine)?;
            let logits = if m.prefill_chunk > 0 && !short_monolithic {
                m.gpu
                    .prefill_with_cache_chunked(&prompt, &mut state, m.prefill_chunk)
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
            while emit.push(&[t]) {
                if pen_cfg.is_some() {
                    window.push(t);
                }
                step += 1;
                if step >= budget {
                    break;
                }
                t = if let Some(pc) = &pen_cfg {
                    // penalized greedy needs the full row (argmax AFTER penalties)
                    let mut row = m
                        .gpu
                        .decode_step(t, &mut state)
                        .map_err(EngineError::engine)?;
                    dsv4_penalize_row(&mut row, &window, pc);
                    argmax(&row)
                } else {
                    m.gpu
                        .decode_step_greedy(t, &mut state)
                        .map_err(EngineError::engine)?
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
            let mut row0 = pre_logits;
            let mut t = draw(&mut row0, p0, &window).map_err(EngineError::engine)?;
            let mut step = 0usize;
            while emit.push(&[t]) {
                if pen_cfg.is_some() {
                    window.push(t);
                }
                step += 1;
                if step >= budget {
                    break;
                }
                let mut row = m
                    .gpu
                    .decode_step(t, &mut state)
                    .map_err(EngineError::engine)?;
                t = draw(&mut row, p0 + step, &window).map_err(EngineError::engine)?;
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
        match processed_prefix_tokens(&prompt, &emit.ids, state_to_park.pos) {
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
    emit.finish(
        prompt.len(),
        n_cached,
        t0.elapsed().as_secs_f64(),
        spec_usage,
    );
    if let Some(toks) = park_toks {
        park_prefix(
            m,
            host_cache,
            req,
            toks,
            &state_to_park,
            dstate_to_park.as_ref(),
        );
    }
    Ok(())
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
mod prefill_chunk_flag_tests {
    use super::{resolve_prefill_chunk, use_chunked_prefill};

    #[test]
    fn default_is_off_and_values_are_strictly_bounded() {
        assert_eq!(resolve_prefill_chunk(None, 1_048_576), Ok(0));
        assert_eq!(resolve_prefill_chunk(Some("0"), 1_048_576), Ok(0));
        assert_eq!(resolve_prefill_chunk(Some("64"), 1_048_576), Ok(64));
        assert!(resolve_prefill_chunk(Some("-1"), 1_048_576).is_err());
        assert!(resolve_prefill_chunk(Some("banana"), 1_048_576).is_err());
        assert!(resolve_prefill_chunk(Some("65"), 64).is_err());
        assert!(resolve_prefill_chunk(Some("65"), 1_048_576).is_err());
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
    use super::{processed_prefix_tokens, scan_stop_cut, snap};

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
            processed_prefix_tokens(&prompt, &emitted, 5).unwrap(),
            [10, 11, 12, 20, 21],
            "the final emitted token may be pending and belongs to the next suffix"
        );
        assert!(
            processed_prefix_tokens(&prompt, &emitted, 7)
                .unwrap_err()
                .contains("stream committed only 3"),
            "spec state ahead of the visible stream must not enter the host tier"
        );
        assert!(
            processed_prefix_tokens(&prompt, &emitted, 2)
                .unwrap_err()
                .contains("precedes prompt boundary"),
            "a corrupt pre-prompt state is refused rather than saturating to zero"
        );
    }
}
