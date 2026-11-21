//! hy_v4_flash serve route: Hy4-flash through the worker's `Cmd::Generate(Request) -> Event`
//! contract, on the GPU route of the onboarding's step 8 (`memra_engine::hy4f_route`, numeric
//! class `hy-v4-flash-bf16x2-activations`), never through `HybridModel`. The HTTP surface (the
//! OpenAI routes, the Hy4 chat renderer, the Hy4 tool-call and reasoning parser, streaming SSE) is
//! the shared one; this module is the worker side.
//!
//! The program. A request scores its prompt in prefill calls of at most `MEMRA_HY4F_PREFILL_CHUNK`
//! tokens (default 2,048) against its own carried cache, then one decode call per generated
//! token. The route's kernels compute every row on its own, in an order that depends on neither
//! the call's other rows nor its length, so a token gets the same bits in any prefill chunk as in a
//! decode call (research/modelplan-onboarding-hy4flash-20261119/serve: chunked and whole prefills
//! give equal logits at every position, across the indexer budget, the window and the chunk
//! boundaries). The route picks its own router and indexer selections (the checkpoint gate ran
//! the class with the oracle's picks applied; the free route equals the forced one bit for bit up
//! to its first own near-tie pick).
//!
//! Sessions. Every admitted request owns a `hy4f_route::Session`: its cache rows (576 F32 values a
//! token a layer, a 512-row ring on the sliding layers, the indexer's token and group keys on the
//! indexed layers) and its token history. The weights and the call scratch are shared, and the
//! route runs one call at a time, so a request's numbers do not depend on the others it is
//! served beside. Each step runs at most one prefill chunk (the oldest request still prefilling)
//! and one decode call for every request that is generating.
//!
//! Memory, on unified memory. The weights take about 87 GiB of the box's memory and stay
//! resident; the Engram tables stay on disk, read through the checkpoint's own mapping, so their
//! rows live in the page cache, which the kernel reclaims (MemAvailable counts it). The book's
//! ceiling is MemAvailable after the weights load and the warm-up has grown the call scratch to
//! its chunk-sized maximum, less the route's 4 GiB headroom. A session books its cache at its
//! capacity (prompt plus token budget, `Route::session_bytes`) and the context-sized part of the
//! call scratch (the indexer's group logits, `Route::index_scratch_bytes`) plus a margin. A request
//! that cannot fit with the route idle is refused as a context-length error naming the largest
//! session that fits; one that fits alone waits until running requests finish. Every allocation
//! is checked again against live MemAvailable where it happens (`hy4f_route`), so a co-tenant's
//! memory makes a call fail its request rather than the box run out of memory.
//!
//! Sampling. Greedy is the argmax, the first of equal maxima (the gate's tapes' rule, so a served
//! greedy answer is `hy4f-gpu --gen`'s). Temperature, top-p and top-k draw through the DSv4
//! route's position-keyed host sampler (`dsv4_sample_row`, seeded and reproducible), penalties
//! through `dsv4_penalize_row`; `min_p` is refused. Stops: the end-of-sentence id (unless
//! `ignore_eos`), the request's stop ids, stop strings over the detokenized stream (the Hy4 tool
//! calls' close stays in the stream so the parser closes the block), and the token budget.
//!
//! DFly2 drafts (`MEMRA_HY4F_DRAFT`, default the artifact's `num_speculative_tokens`, 5). Right
//! after a request's call the drafter (`hy4f_route::dfly`) proposes a round from the call's last
//! kept row; the request's next call scores the anchor and the drafts in one call (each row the
//! one-token decode row bit for bit, by the route's row-independence), the tokens are chosen row
//! by row as a decode call chooses them (greedy, or the position-keyed draw with the penalty
//! window grown token by token), and the drafts are kept while they equal those choices; the
//! rest roll back (`Route::rollback_in` restores the sliding rings' rows). The tokens are the
//! undrafted route's; drafts change only the number of calls. A draft round or a verify call's ring
//! backup that is refused (not a device fault) costs that request its drafts, never its answer:
//! it is served on by the undrafted route's calls. Each session books one round's cache
//! positions beside its own (the per-session draft state); the drafter's weights and call scratch
//! are the route's, resident and grown in the warm-up before the book's ceiling is read.
//!
//! Deliberate limits, stated: `response_format`/grammar refused by name; image and video parts
//! refused (no Hy4 vision program); no prefix cache, no CUDA graphs, no batched decode across
//! requests.

use crate::dsv4_admit::Admission;
use crate::dsv4_serve::{admission_answer, fail_request, settle, snap};
use crate::health::RouteHealth;
use crate::route_telemetry::{RouteLoad, RouteRun, ServeStats};
use crate::worker::{EngineError, Event, EventSender, ModelCaps, Request, SpecUsage};
use memra_engine::decode::StopReason;
use memra_engine::dsv4_gpu::{Dsv4PenaltyCfg, Dsv4SampleCfg, dsv4_penalize_row, dsv4_sample_row};
use memra_engine::hy4f_route::{Ckpt, Dev, Route, Session, mem_available};
use memra_tokenizer::Tokenizer;
use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::mpsc::{Receiver, Sender};
use std::time::Instant;

/// Prefill tokens per call by default: the size the route's call scratch is grown for at boot.
const DEFAULT_PREFILL_CHUNK: usize = 2048;

/// Unified memory the book keeps free beside its terms: the route's own headroom, which every
/// allocation's live check keeps (`hy4f_route::HEADROOM`).
const HEADROOM_BYTES: usize = 4 << 30;

/// Bytes a step books beside its itemized terms: allocator granularity, the host copies of a
/// call's logits and Engram rows, and the server's own growth.
const MARGIN_BYTES: usize = 1 << 30;

/// The Hy4 tool calls' close, kept in the stream so the parser closes the block (the HTTP layer
/// adds it as a stop on tool requests).
const HY4_CALLS_CLOSE: &str = "</tool_calls:opensource>";

/// Cheap boot-time probe: the effective `model_type` (text_config over top level) maps to the
/// hy_v4_flash architecture. The loader re-validates strictly.
pub fn is_hy4f_dir(dir: &Path) -> bool {
    let Ok(text) = std::fs::read_to_string(dir.join("config.json")) else {
        return false;
    };
    let Ok(json) = serde_json::from_str::<serde_json::Value>(&text) else {
        return false;
    };
    let mt = json
        .get("text_config")
        .and_then(|t| t.get("model_type"))
        .and_then(|v| v.as_str())
        .or_else(|| json.get("model_type").and_then(|v| v.as_str()))
        .unwrap_or_default();
    memra_gguf::config::Arch::from_hf_model_type(mt).is_hy_v4_flash()
}

/// The hy_v4_flash plan this binary compiles from `config.json` text, with the config it came from.
fn compile(
    config_text: &str,
) -> Result<
    (
        memra_gguf::config::ModelConfig,
        memra_gguf::model_plan::ModelPlan,
    ),
    String,
> {
    use memra_gguf::config::{HfConfig, ModelConfig};
    let config = ModelConfig::from_hf(&HfConfig::try_parse(config_text)?);
    let pack = memra_gguf::model_packs::by_alias("hy_v4_flash")
        .ok_or("the hy_v4_flash pack is not in this build")?;
    let plan = pack
        .compile_plan(&config)
        .map_err(|e| format!("plan: {e:?}"))?;
    if config.hy_v4_flash.is_none() {
        return Err("config.json is not a Hy4-flash config".into());
    }
    Ok((config, plan))
}

fn hex_sha256(b: &[u8]) -> String {
    use sha2::Digest;
    sha2::Sha256::digest(b)
        .iter()
        .map(|x| format!("{x:02x}"))
        .collect()
}

/// Validate `MEMRA_REWRITE_BUNDLE` for this checkpoint: its `artifact.lock` names this family,
/// this `config.json` and the plan this binary compiles from it; a rewrite receipt index, when
/// present, must load against that plan. The route runs one program, the plan's own GPU route, so
/// no receipt widens what it runs; a stale or foreign bundle refuses the boot.
pub(crate) fn install_hy4f_rewrite_bundle(bundle: &Path, dir: &Path) -> Result<String, String> {
    let lock = std::fs::read_to_string(bundle.join("artifact.lock"))
        .map_err(|e| format!("rewrite bundle {}: artifact.lock: {e}", bundle.display()))?;
    let field = |k: &str| {
        lock.lines()
            .find_map(|l| l.strip_prefix(&format!("{k}=")))
            .map(str::to_string)
    };
    if field("family").as_deref() != Some("hy_v4_flash") {
        return Err(format!(
            "rewrite bundle {}: artifact.lock family {:?} is not hy_v4_flash",
            bundle.display(),
            field("family")
        ));
    }
    let config = std::fs::read(dir.join("config.json")).map_err(|e| e.to_string())?;
    if field("config_sha256").as_deref() != Some(hex_sha256(&config).as_str()) {
        return Err(format!(
            "rewrite bundle {}: artifact.lock config_sha256 does not match {}",
            bundle.display(),
            dir.join("config.json").display()
        ));
    }
    let text = String::from_utf8(config).map_err(|e| format!("config.json: {e}"))?;
    let (_, plan) = compile(&text)?;
    let plan_hash = hex_sha256(format!("{plan:#?}\n").as_bytes());
    if field("plan_sha256").as_deref() != Some(plan_hash.as_str()) {
        return Err(format!(
            "rewrite bundle {}: artifact.lock plan_sha256 {:?} is not this binary's plan {plan_hash}",
            bundle.display(),
            field("plan_sha256")
        ));
    }
    if bundle.join("rewrite-receipts.tsv").exists() {
        memra_gguf::execution_manifest::RewriteQualifications::load(bundle, &plan)
            .map_err(|e| format!("rewrite bundle {}: {e}", bundle.display()))?;
    }
    Ok(plan_hash)
}

/// What the route loaded, for the worker's caps and contract.
pub struct Hy4fModel {
    pub tok: Arc<Tokenizer>,
    /// the served context: the declared or `MEMRA_CTX` context, at most the largest session the
    /// book holds with the route idle
    pub max_seq: usize,
    /// requests served at once
    pub sessions: usize,
    start: Sender<Start>,
}

/// The serving thread's start: its admission queue and its books.
struct Start {
    rx: Receiver<Box<Request>>,
    health: Arc<RouteHealth>,
    load: Arc<RouteLoad>,
}

/// What the engine thread reports once the model is up.
struct Ready {
    max_seq: usize,
}

struct EngineCfg {
    name: String,
    dir: PathBuf,
    tok: Arc<Tokenizer>,
    config_text: String,
    /// the requested context (`MEMRA_CTX` or the config's)
    max_seq: usize,
    sessions: usize,
    eos: u32,
    chunk: usize,
    /// DFly2 drafts a round (0: no drafter)
    drafts: usize,
    /// `MEMRA_HY4F_DRAFT_FAULT`: the draft-fault gate's injection
    draft_fault: Option<DraftFault>,
}

/// The draft-fault gate's injection, per request, counted in the request's verified rounds: the
/// draft round after `Draft(n)` rounds, or the ring backup of the verify call after `Backup(n)`,
/// runs and is then refused, as a refusal by the route would be.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DraftFault {
    Draft(u64),
    Backup(u64),
}

const INJECTED: &str = "refused by MEMRA_HY4F_DRAFT_FAULT";

/// `MEMRA_HY4F_DRAFT_FAULT`: `draft:N` or `backup:N` (gates only), unset for none.
fn resolve_draft_fault(raw: Option<&str>) -> Result<Option<DraftFault>, String> {
    let Some(x) = raw.map(str::trim).filter(|x| !x.is_empty()) else {
        return Ok(None);
    };
    let fault = match x.split_once(':') {
        Some(("draft", n)) => n.parse().ok().map(DraftFault::Draft),
        Some(("backup", n)) => n.parse().ok().map(DraftFault::Backup),
        _ => None,
    };
    fault
        .map(Some)
        .ok_or_else(|| format!("MEMRA_HY4F_DRAFT_FAULT {x:?} is not draft:N or backup:N"))
}

/// `MEMRA_HY4F_DRAFT`: the DFly2 drafts a round (1 to 16), `0` to serve without the drafter, or
/// unset for the artifact's `speculative_config.num_speculative_tokens`.
fn resolve_drafts(raw: Option<&str>, declared: Option<u32>) -> Result<usize, String> {
    match raw.map(str::trim) {
        None | Some("") => Ok(declared.map_or(0, |k| k as usize)),
        Some(x) => match x.parse::<usize>() {
            Ok(n) if n <= memra_engine::hy4f_route::MAX_DRAFTS => Ok(n),
            _ => Err(format!(
                "MEMRA_HY4F_DRAFT {x:?} is not a draft count from 0 to {}",
                memra_engine::hy4f_route::MAX_DRAFTS
            )),
        },
    }
}

/// `MEMRA_HY4F_PREFILL_CHUNK`: a positive token count, or unset for the default.
fn resolve_chunk(raw: Option<&str>) -> Result<usize, String> {
    match raw.map(str::trim) {
        None | Some("") => Ok(DEFAULT_PREFILL_CHUNK),
        Some(x) => match x.parse::<usize>() {
            Ok(n) if n > 0 => Ok(n),
            _ => Err(format!(
                "MEMRA_HY4F_PREFILL_CHUNK {x:?} is not a positive token count"
            )),
        },
    }
}

/// Load the route: validate the checkpoint, the template's dialect, the context and the bundle
/// here, then bring the model up on its own serving thread and wait for it.
pub fn load(name: &str, dir: &Path, tok: Arc<Tokenizer>) -> Result<Hy4fModel, String> {
    if !tok
        .chat_template()
        .is_some_and(memra_tokenizer::chat::template_is_hy4)
    {
        return Err(format!(
            "hy_v4_flash model {name:?}: the checkpoint's chat template is not the Hy4 dialect; \
             the route renders only that dialect"
        ));
    }
    let config_text = std::fs::read_to_string(dir.join("config.json"))
        .map_err(|e| format!("hy_v4_flash model {name:?}: config.json: {e}"))?;
    let (mc, _) = compile(&config_text).map_err(|e| format!("hy_v4_flash model {name:?}: {e}"))?;
    if mc.context_length == 0 {
        return Err(format!(
            "hy_v4_flash model {name:?}: config.json declares no max_position_embeddings"
        ));
    }
    let declared = mc.context_length as usize;
    let max_seq = crate::worker::resolve_env_ctx(declared).map_err(|e| {
        format!("hy_v4_flash model {name:?}: {e} (config.json declares {declared} positions)")
    })?;
    if let Some(bundle) = std::env::var_os("MEMRA_REWRITE_BUNDLE") {
        let plan = install_hy4f_rewrite_bundle(Path::new(&bundle), dir)?;
        eprintln!(
            "[hy4f-serve] {name}: rewrite bundle {} installed for plan {plan}; the route runs the \
             plan's own GPU route, no rewrite",
            Path::new(&bundle).display()
        );
    }
    let chunk = resolve_chunk(std::env::var("MEMRA_HY4F_PREFILL_CHUNK").ok().as_deref())?;
    let declared_k = mc
        .hy_v4_flash
        .as_ref()
        .filter(|hy| hy.dfly.num_hidden_layers > 0)
        .and_then(|hy| hy.dfly.num_speculative_tokens);
    let drafts = resolve_drafts(
        std::env::var("MEMRA_HY4F_DRAFT").ok().as_deref(),
        declared_k,
    )
    .map_err(|e| format!("hy_v4_flash model {name:?}: {e}"))?;
    if drafts == 0
        && mc
            .hy_v4_flash
            .as_ref()
            .is_some_and(|hy| hy.dfly.num_hidden_layers > 0)
    {
        eprintln!(
            "[hy4f-serve] {name}: the artifact's DFly2 drafter is not run ({})",
            if declared_k.is_none() {
                "the config declares no speculative_config.num_speculative_tokens; \
                 MEMRA_HY4F_DRAFT=N sets the drafts a round"
            } else {
                "MEMRA_HY4F_DRAFT=0"
            }
        );
    }
    let draft_fault = resolve_draft_fault(std::env::var("MEMRA_HY4F_DRAFT_FAULT").ok().as_deref())
        .map_err(|e| format!("hy_v4_flash model {name:?}: {e}"))?;
    if let Some(f) = draft_fault {
        eprintln!(
            "[hy4f-serve] {name}: MEMRA_HY4F_DRAFT_FAULT: {f:?} is refused in every request (a \
             gate's injection, not a serving setting)"
        );
    }
    let sessions = crate::route_contract::hybrid_interactive_cap();
    if sessions == 0 {
        return Err("MEMRA_MAX_SESSIONS=0 leaves the hy_v4_flash route no session".into());
    }
    let cfg = EngineCfg {
        name: name.to_string(),
        dir: dir.to_path_buf(),
        eos: tok.eos_id(),
        tok: tok.clone(),
        config_text,
        max_seq,
        sessions,
        chunk,
        drafts,
        draft_fault,
    };
    let (ready_tx, ready_rx) = std::sync::mpsc::channel::<Result<Ready, String>>();
    let (start_tx, start_rx) = std::sync::mpsc::channel::<Start>();
    let handle = std::thread::Builder::new()
        .name(format!("hy4f-serve-{name}"))
        .spawn(move || engine_thread(cfg, ready_tx, start_rx))
        .map_err(|e| format!("spawn the hy_v4_flash serving thread: {e}"))?;
    THREADS
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .push(handle);
    let ready = ready_rx
        .recv()
        .map_err(|_| format!("hy_v4_flash model {name:?}: the serving thread died loading"))??;
    Ok(Hy4fModel {
        tok,
        max_seq: ready.max_seq,
        sessions,
        start: start_tx,
    })
}

/// Every serving thread spawned, so a graceful shutdown can wait for them.
static THREADS: std::sync::Mutex<Vec<std::thread::JoinHandle<()>>> =
    std::sync::Mutex::new(Vec::new());

/// Wait at most `deadline` for the serving threads to leave; returns how many are still live.
pub fn join_threads(deadline: std::time::Duration) -> usize {
    let threads = std::mem::take(&mut *THREADS.lock().unwrap_or_else(|p| p.into_inner()));
    let t0 = Instant::now();
    while threads.iter().any(|h| !h.is_finished()) && t0.elapsed() < deadline {
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
    let mut live = 0;
    for h in threads {
        if h.is_finished() {
            let _ = h.join();
        } else {
            live += 1;
        }
    }
    live
}

/// The route's policy contract (`route_contract.rs` declares it; the wiring test greps this file).
pub fn contract(model: &str, sessions: usize) -> crate::route_contract::RouteContract {
    crate::route_contract::RouteContract::hy4f_thread(model, sessions)
}

/// ModelCaps for /v1/models and the HTTP gates: the Hy4 dialect. `hy3` is set with `hy4`: the two
/// Tencent dialects share the suffixed think seam, the call framing and its stop, and the refusal
/// of constrained tool calls; `hy4` selects the Hy4 call body, the think-open default and the
/// `max` rung.
pub fn caps(m: &Hy4fModel) -> ModelCaps {
    ModelCaps {
        max_choices: 1,
        tools_branch: true,
        hy3: true,
        hy4: true,
        qwen_think: false,
        think_switch: false,
        chat_ok: true,
        context_length: m.max_seq,
        tokenizer: m.tok.pre().to_string(),
        instruct_type: Some("hy4".to_string()),
        effort_levels: true,
        qwen_effort: false,
        gemma_think: false,
        gemma_unopened_thought: false,
        dsv4: false,
        dsv41: false,
        glm5: false,
        chat_temperature_default: None,
        chat_top_p_default: None,
        n_vocab: m.tok.vocab_size(),
        think_close: Vec::new(),
    }
}

/// Start serving: the returned sender is the model's admission queue.
pub fn spawn(m: Hy4fModel, health: Arc<RouteHealth>, load: Arc<RouteLoad>) -> Sender<Box<Request>> {
    let (tx, rx) = std::sync::mpsc::channel::<Box<Request>>();
    // the engine thread is parked on this start; a send failure means it already died, and the
    // closed queue then answers every request with the route's error
    let _ = m.start.send(Start { rx, health, load });
    tx
}

/// Latches the route dead when the serving thread ends for any reason: the loop's own reason when
/// it stopped serving on a fault, else the way the thread ended.
struct ExitLatch(Arc<RouteHealth>, Option<String>);

impl Drop for ExitLatch {
    fn drop(&mut self) {
        let why = match self.1.take() {
            Some(why) => why,
            None if std::thread::panicking() => "hy_v4_flash serving thread panicked".into(),
            None => "hy_v4_flash serving thread exited".into(),
        };
        self.0.mark_dead(why);
    }
}

/// The detokenized stream of one request and its stop rules: EOS, stop ids, the budget, and stop
/// strings, with a client's stop sent whole ([`stop_end`]).
struct Stream {
    tok: Arc<Tokenizer>,
    tx: EventSender,
    eos: Vec<u32>,
    stops: Vec<String>,
    stop_ids: Vec<u32>,
    budget: usize,
    ids: Vec<u32>,
    /// the exact bytes of `ids` (`Tokenizer::decode_bytes_special`)
    bytes: Vec<u8>,
    /// bytes of `bytes` already turned into `text`
    emitted: usize,
    text: String,
    stop: Option<StopReason>,
    client_gone: bool,
}

impl Stream {
    /// Take one generated token; false when generation stops here.
    fn push(&mut self, id: u32) -> bool {
        if self.stop.is_some() || self.client_gone {
            return false;
        }
        let stop_id = crate::worker::stop_token_reason(id, &self.stop_ids);
        if stop_id.is_some() || self.eos.contains(&id) {
            self.stop = Some(stop_id.unwrap_or(StopReason::Eos));
            return false;
        }
        self.ids.push(id);
        self.bytes
            .extend_from_slice(&self.tok.decode_bytes_special(&[id], true));
        let last = self.ids.len() >= self.budget;
        let (delta, stopped) = text_step(
            &mut self.text,
            &self.bytes,
            &mut self.emitted,
            last,
            &self.stops,
        );
        if self.tx.send(Event::Token { id, text: delta }).is_err() {
            self.client_gone = true;
            return false;
        }
        if stopped {
            self.stop = Some(StopReason::Callback);
            return false;
        }
        if last {
            self.stop = Some(StopReason::MaxNew);
            return false;
        }
        true
    }

    fn finish(&self, n_prompt: usize, elapsed_s: f64, spec: Option<SpecUsage>) {
        if self.client_gone {
            return;
        }
        let _ = self.tx.send(Event::TokenSnapshot(self.ids.clone()));
        let _ = self.tx.send(Event::Done {
            stop_reason: format!("{:?}", self.stop.unwrap_or(StopReason::MaxNew)),
            n_tokens: self.ids.len(),
            n_prompt,
            n_cached: 0,
            elapsed_s,
            spec,
        });
    }
}

/// The text a token adds to a stream whose bytes so far are `bytes`, of which `emitted` are
/// already text: every completed character (the worker's `utf8_delta` rule). `last`: the answer
/// ends with this token, so an incomplete tail is read lossily, as the detokenized ids read it.
fn text_delta(bytes: &[u8], emitted: &mut usize, last: bool) -> String {
    let mut delta = crate::worker::utf8_delta(bytes, emitted);
    if last && *emitted < bytes.len() {
        delta.push_str(&String::from_utf8_lossy(&bytes[*emitted..]));
        *emitted = bytes.len();
    }
    delta
}

/// One token's step of a stream's text: returns what to send and whether a stop string ended the
/// stream there (the deepseek-v4.1 route's law, `dsv41_serve::text_step`).
fn text_step(
    text: &mut String,
    bytes: &[u8],
    emitted: &mut usize,
    last: bool,
    stops: &[String],
) -> (String, bool) {
    let old = text.len();
    let delta = text_delta(bytes, emitted, last);
    text.push_str(&delta);
    if let Some(abs) = stop_end(&text[..old], text, stops) {
        let end = snap(text, abs).max(old);
        text.truncate(end);
        return (text[old..].to_string(), true);
    }
    (delta, false)
}

/// Where the stream ends when a stop string completes in `full`, whose first `text.len()` bytes
/// are already sent (None: no stop). A client's stop ends it after the whole token (the HTTP
/// layer cuts at the stop); the Hy4 calls' close alone ends the stream at its own end, so the
/// parser sees the closed block.
fn stop_end(text: &str, full: &str, stops: &[String]) -> Option<usize> {
    let starts_at = |stop: &str| -> Option<usize> {
        if stop.is_empty() {
            return None;
        }
        let from = snap(full, text.len().saturating_sub(stop.len() - 1));
        full[from..].find(stop).map(|at| from + at)
    };
    let client = stops
        .iter()
        .filter(|s| s.as_str() != HY4_CALLS_CLOSE)
        .any(|s| starts_at(s).is_some());
    if client {
        return Some(full.len());
    }
    if !stops.iter().any(|s| s.as_str() == HY4_CALLS_CLOSE) {
        return None;
    }
    let at = starts_at(HY4_CALLS_CLOSE)?;
    Some((at + HY4_CALLS_CLOSE.len()).max(text.len()).min(full.len()))
}

/// A request the route is serving.
struct Active {
    req: Box<Request>,
    run: Option<RouteRun>,
    prompt: Vec<u32>,
    /// prompt tokens already scored
    prefilled: usize,
    cap: usize,
    sess: Session,
    stream: Stream,
    /// the token the next decode call feeds; None until the prompt's prefill ran
    next: Option<u32>,
    sample: Option<Dsv4SampleCfg>,
    pen: Option<Dsv4PenaltyCfg>,
    /// the penalty window: the prompt and every generated token (only with penalties)
    window: Vec<u32>,
    t0: Instant,
    steps: usize,
    /// the route failed the request: booked as failed, not as a client that left
    fault: bool,
    /// DFly2: the drafts the next verify call scores after `next` (empty: a plain decode call)
    drafts: Vec<u32>,
    /// DFly2: a draft round or a verify call's ring backup was refused for this request; it is
    /// served without drafts from there on, as the undrafted route serves it
    draft_off: bool,
    /// DFly2 rounds, drafts and accepted drafts
    spec: SpecUsage,
}

/// A request past rendering, waiting for its memory.
struct Prepared {
    req: Box<Request>,
    run: RouteRun,
    /// the memory door deferred it once (logged the first time)
    deferred: bool,
    prompt: Vec<u32>,
    budget: usize,
    sample: Option<Dsv4SampleCfg>,
    pen: Option<Dsv4PenaltyCfg>,
}

/// A rendered request's prompt, token budget and sampler.
struct Rendered {
    prompt: Vec<u32>,
    budget: usize,
    sample: Option<Dsv4SampleCfg>,
    pen: Option<Dsv4PenaltyCfg>,
}

/// Render and validate a request for the route (`Err` answers it).
fn prepare(
    tok: &Arc<Tokenizer>,
    max_seq: usize,
    req: &mut Request,
) -> Result<Rendered, EngineError> {
    if req.grammar.is_some() {
        return Err(EngineError::invalid_param(
            "response_format is not available on the hy_v4_flash route",
            "response_format",
        ));
    }
    if !req.images.is_empty() {
        return Err(EngineError::invalid_param(
            "image and video parts are not available on the hy_v4_flash route (no Hy4 vision program)",
            "messages",
        ));
    }
    let prompt = crate::dsv4_serve::render_prompt_tok(tok, req)?;
    // the request's own context bound (`max_ctx`) tightens the route's
    let ctx = request_ctx(max_seq, req.params.max_ctx);
    let vocab = tok.vocab_size();
    if let Some(&bad) = prompt.iter().find(|&&t| vocab > 0 && t as usize >= vocab) {
        return Err(EngineError::invalid_param(
            format!("prompt token {bad} is past the vocabulary of {vocab}"),
            "prompt",
        ));
    }
    if prompt.len() >= ctx {
        return Err(EngineError::context_length(format!(
            "prompt ({} tok) leaves no room in the {ctx}-position context",
            prompt.len()
        )));
    }
    let budget = req.params.max_new.min(ctx - prompt.len()).max(1);
    let sc = &req.sampler_cfg;
    let greedy = sc.temperature <= 0.0;
    if !greedy && sc.min_p != 0.0 {
        return Err(EngineError::invalid_param(
            "min_p is not implemented on the hy_v4_flash sampler; use top_p/top_k",
            "min_p",
        ));
    }
    let penalties = sc.penalty_repeat != 1.0 || sc.penalty_freq != 0.0 || sc.penalty_present != 0.0;
    let pen = penalties.then_some(Dsv4PenaltyCfg {
        last_n: if sc.penalty_last_n > 0 {
            sc.penalty_last_n
        } else {
            usize::MAX
        },
        repeat: sc.penalty_repeat,
        freq: sc.penalty_freq,
        present: sc.penalty_present,
    });
    let sample = (!greedy).then_some(Dsv4SampleCfg {
        temperature: sc.temperature,
        top_p: sc.top_p,
        top_k: sc.top_k,
        seed: sc.seed,
    });
    Ok(Rendered {
        prompt,
        budget,
        sample,
        pen,
    })
}

/// A request's context: the route's served context, tightened by the request's `max_ctx`.
fn request_ctx(max_seq: usize, max_ctx: Option<usize>) -> usize {
    max_ctx.map_or(max_seq, |c| c.min(max_seq))
}

/// The ids that end a request's generation: the request's own and the model's end-of-sentence
/// id, which `ignore_eos` leaves out.
fn end_ids(request: &[u32], model_eos: u32, ignore_eos: bool) -> Vec<u32> {
    let mut ids = request.to_vec();
    if !ignore_eos && !ids.contains(&model_eos) {
        ids.push(model_eos);
    }
    ids
}

/// The first of equal maxima, as `torch.argmax` and the gate's tapes take it (`hy4f-gpu`).
fn argmax(row: &[f32]) -> usize {
    let mut best = 0;
    for (i, v) in row.iter().enumerate() {
        if v.total_cmp(&row[best]).is_gt() {
            best = i;
        }
    }
    best
}

/// The next token from one logits row: penalties over the window, then the argmax or the
/// position-keyed draw at absolute position `pos`.
fn choose(
    mut row: Vec<f32>,
    pos: usize,
    sample: Option<&Dsv4SampleCfg>,
    pen: Option<&Dsv4PenaltyCfg>,
    window: &[u32],
) -> Result<u32, String> {
    if row.iter().any(|x| !x.is_finite()) {
        return Err("non-finite logits".into());
    }
    if let Some(p) = pen {
        dsv4_penalize_row(&mut row, window, p);
    }
    match sample {
        None => Ok(argmax(&row) as u32),
        Some(cfg) => dsv4_sample_row(&row, pos, cfg),
    }
}

/// The route's memory book: what a session of `cap` positions costs, against the ceiling.
#[derive(Debug, Clone)]
struct Book {
    /// MemAvailable after the weights and the warm-up, less the headroom
    ceiling: usize,
    /// prefill rows a call takes at most
    chunk: usize,
    /// the cache bytes of a session per its capacity, from the route's shapes
    state: SessionShape,
    /// the per-session draft state: the cache positions a verify call scores past the request's
    /// own (the anchor and the drafts of one round), booked with every session of a drafting
    /// route. The drafter's call scratch is the route's, grown in the warm-up and inside the
    /// ceiling, like the call scratch.
    draft_rows: usize,
}

/// The per-capacity cache and scratch arithmetic of the loaded route (`Route::session_bytes`,
/// `Route::index_scratch_bytes`), captured as linear terms so the book needs no route borrow.
#[derive(Debug, Clone, Copy)]
struct SessionShape {
    /// bytes a session holds regardless of its capacity
    fixed: usize,
    /// bytes per position of capacity
    per_pos: usize,
    /// per chunk row and per position of capacity: the indexer's group logits
    index_per_row_pos: usize,
    index_fixed_per_row: usize,
}

impl SessionShape {
    fn of(route: &Route<'_>) -> Self {
        // both are affine in `cap` (one byte count a position, a floor for the `+ 1` rows and the
        // window rings); two samples recover them exactly at the pool's stride
        let (a, b) = (4096usize, 8192usize);
        let (sa, sb) = (route.session_bytes(a), route.session_bytes(b));
        let per_pos = (sb - sa) / (b - a);
        let (ia, ib) = (
            route.index_scratch_bytes(1, a),
            route.index_scratch_bytes(1, b),
        );
        let index_per_row_pos = (ib - ia) / (b - a);
        Self {
            fixed: sa - per_pos * a,
            per_pos,
            index_per_row_pos,
            index_fixed_per_row: ia - index_per_row_pos * a,
        }
    }
}

impl Book {
    /// A session's cache at `cap` positions (rounded up to the pool's stride: never under), its
    /// draft state included.
    fn state(&self, cap: usize) -> usize {
        self.state.fixed + self.state.per_pos * self.session_cap(cap).div_ceil(4) * 4
    }

    /// The cache positions a request of `cap` positions allocates: its own and one round's.
    fn session_cap(&self, cap: usize) -> usize {
        cap + self.draft_rows
    }

    /// The call scratch a session of `cap` positions may grow beyond the warm-up's, plus the
    /// margin.
    fn transient(&self, cap: usize) -> usize {
        self.index_scratch(cap) + MARGIN_BYTES
    }

    /// The indexer's group-logit scratch a prefill chunk of a `cap`-position session holds.
    fn index_scratch(&self, cap: usize) -> usize {
        self.chunk
            * (self.state.index_fixed_per_row + self.state.index_per_row_pos * cap.div_ceil(4) * 4)
    }

    /// Everything the running sessions and a new one book: every cache, and the largest
    /// transient (calls run one at a time and share the scratch).
    fn total(&self, caps: &[usize], new: Option<usize>) -> usize {
        let all = caps.iter().chain(new.as_ref());
        let state: usize = all.clone().map(|&c| self.state(c)).sum();
        let transient = all.map(|&c| self.transient(c)).max().unwrap_or(0);
        state + transient
    }

    /// The largest session the route can hold with nothing else running.
    fn largest_fit(&self) -> usize {
        let fits = |c: usize| self.total(&[], Some(c)) <= self.ceiling;
        let (mut lo, mut hi) = (0usize, 1usize << 24);
        while lo < hi {
            let mid = (lo + hi).div_ceil(2);
            if fits(mid) {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }
        lo
    }
}

/// The memory door for the request at the queue head: `Ok(Some)` admits with its session,
/// `Ok(None)` waits for running requests to finish, `Err` answers it.
fn admit_session(
    route: &Route<'_>,
    book: &Book,
    running: &[usize],
    prompt: usize,
    cap: usize,
) -> Result<Option<Session>, Door> {
    let alone = book.total(&[], Some(cap));
    if alone > book.ceiling {
        return Err(Door::Never(Admission::NeverFits {
            dev: 0,
            need: alone as u64,
            ceiling: book.ceiling as u64,
        }));
    }
    if book.total(running, Some(cap)) > book.ceiling {
        return Ok(None);
    }
    // the live check: a co-tenant may hold memory the book's ceiling assumed free. The scratch
    // an earlier long request grew stays allocated (MemAvailable no longer counts it), so what
    // must be free is the new session's cache and the scratch growth still owed to the longest
    // session (running or new) beyond what the route holds.
    let live = mem_available().unwrap_or(0);
    let growth = live_scratch_growth(book, running, cap, route.index_scratch_held());
    let need = book.state(cap) + growth + MARGIN_BYTES + HEADROOM_BYTES;
    if live < need {
        if running.is_empty() {
            return Err(Door::Engine(format!(
                "hy_v4_flash: a {cap}-token session (prompt {prompt}) needs {need} bytes free; \
                 MemAvailable is {live} with nothing running on the route"
            )));
        }
        return Ok(None);
    }
    match route.new_session(book.session_cap(cap)) {
        Ok(s) => Ok(Some(s)),
        // an allocation the device refused is memory pressure; a device that no longer
        // synchronizes is a fault the route cannot serve past
        Err(e) if route.dev.sync().is_err() => Err(Door::Dead(format!(
            "hy_v4_flash route stopped: the device failed allocating a {cap}-token session: {e}"
        ))),
        Err(e) if running.is_empty() => Err(Door::Engine(format!(
            "hy_v4_flash: the cache of a {cap}-token session: {e}"
        ))),
        Err(e) => {
            eprintln!("[hy4f-serve] a {cap}-token session waits: {e}");
            Ok(None)
        }
    }
}

/// The index scratch growth a step may still need beyond `held`: the largest of the running
/// sessions' and the new one's.
fn live_scratch_growth(book: &Book, running: &[usize], cap: usize, held: usize) -> usize {
    running
        .iter()
        .chain(std::iter::once(&cap))
        .map(|&c| book.index_scratch(c))
        .max()
        .unwrap_or(0)
        .saturating_sub(held)
}

/// Why the memory door answers the request at the queue head instead of admitting it.
enum Door {
    /// no session that size fits with the route idle
    Never(Admission),
    /// the memory is held by something the route does not own: the engine's error answers
    Engine(String),
    /// the device failed: the route stops serving
    Dead(String),
}

/// The loaded route and its books, on the serving thread.
struct Up<'a> {
    route: Route<'a>,
    book: Book,
    steps: usize,
}

fn engine_thread(cfg: EngineCfg, ready: Sender<Result<Ready, String>>, start: Receiver<Start>) {
    let t0 = Instant::now();
    let (config, plan) = match compile(&cfg.config_text) {
        Ok(x) => x,
        Err(e) => {
            let _ = ready.send(Err(format!("hy_v4_flash model {:?}: {e}", cfg.name)));
            return;
        }
    };
    let Some(hy) = config.hy_v4_flash.as_ref() else {
        let _ = ready.send(Err("not a Hy4-flash config".into()));
        return;
    };
    let mut up = match bring_up(&cfg, &plan, hy) {
        Ok(u) => u,
        Err(e) => {
            let _ = ready.send(Err(format!("hy_v4_flash model {:?}: {e}", cfg.name)));
            return;
        }
    };
    let fit = up.book.largest_fit();
    let max_seq = cfg.max_seq.min(fit);
    eprintln!(
        "[hy4f-serve] {}: up in {:.1}s; book ceiling {} bytes, a session {} + {} bytes a position, \
         largest session {fit} positions, serving {max_seq} (requested {}), prefill chunk {}, \
         {} sessions",
        cfg.name,
        t0.elapsed().as_secs_f64(),
        up.book.ceiling,
        up.book.state.fixed,
        up.book.state.per_pos,
        cfg.max_seq,
        cfg.chunk,
        cfg.sessions
    );
    if max_seq < 2 {
        let _ = ready.send(Err(format!(
            "hy_v4_flash model {:?}: the memory left beside the weights holds no session",
            cfg.name
        )));
        return;
    }
    let _ = ready.send(Ok(Ready { max_seq }));
    let Ok(Start { rx, health, load }) = start.recv() else {
        return;
    };
    let mut latch = ExitLatch(health.clone(), None);
    let sink = health.clone();
    let _progress = memra_engine::progress::ProgressSinkScope::install(Box::new(move |rows| {
        sink.note_rows(rows)
    }));
    let (ptx, prx) = std::sync::mpsc::channel::<Prepared>();
    let tok = cfg.tok.clone();
    let (rload, rhealth) = (load.clone(), health.clone());
    let renderer = std::thread::Builder::new()
        .name("hy4f-render".into())
        .spawn(move || render_loop(tok, max_seq, rx, rload, rhealth, ptx));
    if let Err(e) = renderer {
        latch.1 = Some(format!("hy_v4_flash: start the renderer: {e}"));
        return;
    }
    latch.1 = serve_loop(&cfg, &mut up, &prx, &health, &load);
}

/// Upload the weights, warm the route up and size the book.
fn bring_up<'a>(
    cfg: &EngineCfg,
    plan: &'a memra_gguf::model_plan::ModelPlan,
    hy: &'a memra_gguf::config::HyV4FlashConfig,
) -> Result<Up<'a>, String> {
    // the weights stay resident in unified memory: refuse unless they fit with headroom
    let need: usize = 94_000_000_000 + (16usize << 30);
    let avail = mem_available().unwrap_or(0);
    if avail < need {
        return Err(format!(
            "MemAvailable {:.1} GiB: the resident weights and headroom need {:.1} GiB",
            avail as f64 / (1u64 << 30) as f64,
            need as f64 / (1u64 << 30) as f64
        ));
    }
    // the whole checkpoint binds to the pack's contract first (every declared tensor class,
    // the drafter's and the vision tower's included, present with its shape), before any byte
    // is uploaded
    let (mc, _) = compile(&cfg.config_text)?;
    let source = memra_gguf::source::SafetensorsSource::open(&cfg.dir)
        .map_err(|e| format!("open {}: {e}", cfg.dir.display()))?;
    let binding = memra_gguf::checkpoint_binding::bind_source(&source, &mc, plan)
        .map_err(|e| format!("the checkpoint does not bind to the hy_v4_flash contract: {e}"))?;
    if binding.family() != "hy_v4_flash" {
        return Err(format!(
            "the checkpoint binds to the {} pack, not hy_v4_flash",
            binding.family()
        ));
    }
    eprintln!(
        "[hy4f-serve] {}: checkpoint bound to the hy_v4_flash contract ({} tensors)",
        cfg.name,
        binding.bound.tensors.len()
    );
    drop(source);
    let ck = Ckpt::open(&cfg.dir)?;
    let dev = Dev::new(0)?;
    let tl = Instant::now();
    let mut route = Route::load(dev, &ck, plan, hy, 1, plan.layers.len())?;
    if cfg.drafts > 0 {
        route.load_drafter(&ck)?;
        eprintln!(
            "[hy4f-serve] {}: DFly2 drafter resident, {:.2} GiB, {} drafts a round",
            cfg.name,
            route.drafter_mut().map_or(0, |d| d.weight_bytes) as f64 / (1u64 << 30) as f64,
            cfg.drafts
        );
    }
    let weights = route.dev.live_bytes();
    eprintln!(
        "[hy4f-serve] {}: weights resident, {:.1} GiB in {:.1}s",
        cfg.name,
        weights as f64 / (1u64 << 30) as f64,
        tl.elapsed().as_secs_f64()
    );
    drop(ck);
    // the warm-up: two prefill chunks (the second past the indexer budget when the chunk is the
    // default) and a decode call, which grows every call scratch buffer to its chunk-sized
    // maximum and proves the route answers finite logits
    let tw = Instant::now();
    let n = 2 * cfg.chunk;
    let mut warm = route.new_session(n + 1 + cfg.drafts + 1)?;
    let ids: Vec<u32> = (0..n).map(|i| 1000 + (i as u32 * 7919) % 60000).collect();
    let mut last = Vec::new();
    for c in ids.chunks(cfg.chunk) {
        last = route.call_in(&mut warm, c, false, None, None)?;
    }
    let t = argmax(&last) as u32;
    let next = route.call_in(&mut warm, &[t], false, None, None)?;
    if last.iter().chain(&next).any(|x| !x.is_finite()) {
        return Err("the warm-up answered non-finite logits".into());
    }
    if cfg.drafts > 0 {
        // one round past the indexer budget grows the drafter's scratch to its maximum: a
        // draft, the verify call of the anchor and the drafts, the rollback of all of them
        let t2 = argmax(&next) as u32;
        let d = route.draft_in(&mut warm, 0, t2, cfg.drafts, None, false)?;
        let mut verify = vec![t2];
        verify.extend(&d.ids);
        let from = warm.positions();
        let backup = route.ring_backup_in(&warm, from, verify.len())?;
        let rows = route.call_in(&mut warm, &verify, true, None, None)?;
        if rows.iter().any(|x| !x.is_finite()) {
            return Err("the warm-up's verify call answered non-finite logits".into());
        }
        route.rollback_in(&mut warm, from + 1, &backup)?;
        eprintln!(
            "[hy4f-serve] {}: warm-up draft round {:?}; drafter scratch {:.1} MiB",
            cfg.name,
            d.ids,
            route.drafter_scratch_bytes() as f64 / (1u64 << 20) as f64
        );
    }
    drop(warm);
    route.dev.trim_pool()?;
    let held = route.dev.live_bytes();
    let avail = mem_available().unwrap_or(0);
    let book = Book {
        ceiling: avail.saturating_sub(HEADROOM_BYTES),
        chunk: cfg.chunk,
        state: SessionShape::of(&route),
        // a reservation: the serve loop keeps a round's block and verify rows inside the
        // request's own positions (`room`), so these rows are booked and never written
        draft_rows: if cfg.drafts > 0 { cfg.drafts + 1 } else { 0 },
    };
    if book.draft_rows > 0 {
        eprintln!(
            "[admission] route=hy4f-thread per-session draft-state charge: {} positions, {} bytes; \
             drafter scratch {} bytes held inside the ceiling",
            book.draft_rows,
            book.state.per_pos * book.draft_rows,
            route.drafter_scratch_bytes()
        );
    }
    eprintln!(
        "[hy4f-serve] {}: warm-up {n} + 1 tokens in {:.1}s; scratch {:.2} GiB, MemAvailable {:.1} GiB",
        cfg.name,
        tw.elapsed().as_secs_f64(),
        held.saturating_sub(weights) as f64 / (1u64 << 30) as f64,
        avail as f64 / (1u64 << 30) as f64
    );
    Ok(Up {
        route,
        book,
        steps: 0,
    })
}

/// What one call of a step was: a prefill chunk of a request, or its decode call.
enum Work {
    Prefill { i: usize, s0: usize, s1: usize },
    Decode { i: usize, t: u32 },
}

fn serve_loop(
    cfg: &EngineCfg,
    up: &mut Up,
    rx: &Receiver<Prepared>,
    health: &Arc<RouteHealth>,
    load: &Arc<RouteLoad>,
) -> Option<String> {
    let mut pending: VecDeque<Prepared> = VecDeque::new();
    let mut act: Vec<Active> = Vec::new();
    let mut closing = false;
    let mut last_step = Instant::now();
    loop {
        // intake: block only when there is nothing to do
        if act.is_empty() && pending.is_empty() {
            if closing {
                return None;
            }
            health.set_idle_if_free();
            let Ok(pr) = rx.recv() else {
                closing = true;
                continue;
            };
            pending.push_back(pr);
        }
        if drain_into(rx, &mut pending) {
            closing = true;
        }
        // clients that left: their requests end at this step boundary (their sessions free)
        let mut i = 0;
        let mut left = false;
        while i < act.len() {
            if act[i].req.tx.is_closed() {
                let a = act.remove(i);
                if let Some(run) = a.run {
                    run.cancel();
                }
                health.end_request();
                left = true;
            } else {
                i += 1;
            }
        }
        if left && let Err(e) = up.route.dev.trim_pool() {
            let dead = format!("hy_v4_flash route stopped: {e}");
            eprintln!("[hy4f-serve] {dead}");
            fail_all(&mut act, health, &dead);
            drain_into(rx, &mut pending);
            fail_pending(&mut pending, health, &dead);
            return Some(dead);
        }
        let mut k = 0;
        while k < pending.len() {
            if pending[k].req.tx.is_closed() {
                let gone = pending.remove(k).expect("in range");
                gone.run.cancel();
                health.end_request();
            } else {
                k += 1;
            }
        }
        // admission, first come first served: every head that fits now
        while act.len() < cfg.sessions
            && let Some(head) = pending.front()
        {
            let running: Vec<usize> = act.iter().map(|a| a.cap).collect();
            let (p, cap) = (head.prompt.len(), head.prompt.len() + head.budget);
            let door = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                admit_session(&up.route, &up.book, &running, p, cap)
            }))
            .unwrap_or_else(|payload| {
                Err(Door::Engine(format!(
                    "hy_v4_flash admission panicked: {}",
                    panic_text(payload.as_ref())
                )))
            });
            match door {
                Ok(Some(sess)) => {
                    let mut pr = pending.pop_front().expect("the head exists");
                    let eos = end_ids(&pr.req.params.eos, cfg.eos, pr.req.ignore_eos);
                    let _ = pr.req.tx.send(Event::PromptUsage {
                        n_prompt: p,
                        n_cached: 0,
                    });
                    pr.run.admit();
                    eprintln!(
                        "[admit-mem] route=hy4f-thread model={:?} request_id={:?} capacity={cap} \
                         booked={} running={} verdict=admit",
                        pr.req.model,
                        pr.req.request_id,
                        up.book.total(&running, Some(cap)),
                        running.len()
                    );
                    let stream = Stream {
                        tok: cfg.tok.clone(),
                        tx: pr.req.tx.clone(),
                        eos,
                        stops: pr.req.stop_strings.clone(),
                        stop_ids: pr.req.stop_token_ids.clone(),
                        budget: pr.budget,
                        ids: Vec::new(),
                        bytes: Vec::new(),
                        emitted: 0,
                        text: String::new(),
                        stop: None,
                        client_gone: false,
                    };
                    let window = if pr.pen.is_some() {
                        pr.prompt.clone()
                    } else {
                        Vec::new()
                    };
                    act.push(Active {
                        req: pr.req,
                        run: Some(pr.run),
                        prompt: pr.prompt,
                        prefilled: 0,
                        cap,
                        sess,
                        stream,
                        next: None,
                        sample: pr.sample,
                        pen: pr.pen,
                        window,
                        t0: Instant::now(),
                        steps: 0,
                        fault: false,
                        drafts: Vec::new(),
                        draft_off: false,
                        spec: SpecUsage::default(),
                    });
                }
                Ok(None) => {
                    if let Some(head) = pending.front_mut()
                        && !head.deferred
                    {
                        head.deferred = true;
                        eprintln!(
                            "[admit-mem] route=hy4f-thread model={:?} request_id={:?} capacity={cap} \
                             booked={} running={} verdict=wait",
                            head.req.model,
                            head.req.request_id,
                            up.book.total(&running, Some(cap)),
                            running.len()
                        );
                    }
                    break;
                }
                Err(Door::Dead(why)) => {
                    eprintln!("[hy4f-serve] {why}");
                    fail_all(&mut act, health, &why);
                    drain_into(rx, &mut pending);
                    fail_pending(&mut pending, health, &why);
                    return Some(why);
                }
                Err(Door::Engine(why)) => {
                    let mut pr = pending.pop_front().expect("the head exists");
                    eprintln!("[hy4f-serve] request_id={:?}: {why}", pr.req.request_id);
                    fail_request(&mut pr.req, EngineError::engine(why));
                    health.note_request_fault();
                    pr.run.fail();
                    health.end_request();
                }
                Err(Door::Never(outcome)) => {
                    let mut pr = pending.pop_front().expect("the head exists");
                    let (answer, _) = admission_answer(&outcome, cap, p, || up.book.largest_fit());
                    eprintln!(
                        "[admit-mem] route=hy4f-thread model={:?} request_id={:?} capacity={cap} \
                         verdict={}",
                        pr.req.model,
                        pr.req.request_id,
                        outcome.verdict()
                    );
                    if let Some(served) = answer {
                        settle(pr.run, &mut pr.req, Ok(served));
                    }
                    health.end_request();
                }
            }
        }
        if act.is_empty() {
            continue;
        }
        // the step: the oldest prefilling request's next chunk, then every generating request's
        // decode call, each a call of its own on its own session
        let mut work = Vec::with_capacity(act.len());
        if let Some(i) = act.iter().position(|a| a.prefilled < a.prompt.len()) {
            let s0 = act[i].prefilled;
            let s1 = (s0 + cfg.chunk).min(act[i].prompt.len());
            work.push(Work::Prefill { i, s0, s1 });
        }
        for (i, a) in act.iter().enumerate() {
            if let Some(t) = a.next {
                work.push(Work::Decode { i, t });
            }
        }
        let ts = Instant::now();
        up.steps += 1;
        let mut done: Vec<usize> = Vec::new();
        let mut device_fault: Option<String> = None;
        let (mut prefill_rows, mut decode_rows) = (0usize, 0usize);
        for w in work {
            let (i, mut tokens, is_prefill_end) = match w {
                Work::Prefill { i, s0, s1 } => {
                    prefill_rows += s1 - s0;
                    let end = s1 == act[i].prompt.len();
                    (i, act[i].prompt[s0..s1].to_vec(), Some((s1, end)))
                }
                Work::Decode { i, t } => {
                    // a verify call scores the anchor and the round's drafts in one call; every
                    // row computes on its own, so each row is the decode call's row bit for bit
                    let mut tokens = vec![t];
                    tokens.extend(std::mem::take(&mut act[i].drafts));
                    decode_rows += tokens.len();
                    (i, tokens, None)
                }
            };
            let route = &mut up.route;
            let a = &mut act[i];
            let mut verify = is_prefill_end.is_none() && tokens.len() > 1;
            let from = a.sess.positions();
            let called = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let backup = if verify {
                    let taken = route
                        .ring_backup_in(&a.sess, from, tokens.len())
                        .and_then(|b| {
                            if cfg.draft_fault == Some(DraftFault::Backup(a.spec.rounds)) {
                                Err(INJECTED.to_string())
                            } else {
                                Ok(b)
                            }
                        });
                    match taken {
                        Ok(b) => Some(b),
                        Err(e) => {
                            // a refused backup costs the round, never the request: the anchor
                            // alone is the undrafted route's call. No probe here: the call reads
                            // its rows back, so a device fault fails it and is probed below. The
                            // round's dropped drafts are in no usage count.
                            eprintln!(
                                "[hy4f-serve] request_id={:?}: ring backup of {} rows at {from}: \
                                 {e}; serving it without drafts",
                                a.req.request_id,
                                tokens.len(),
                            );
                            a.draft_off = true;
                            tokens.truncate(1);
                            verify = false;
                            None
                        }
                    }
                } else {
                    None
                };
                route
                    .call_in(&mut a.sess, &tokens, verify, None, None)
                    .map(|rows| (rows, backup))
            }));
            let (rows, backup) = match called {
                Ok(Ok(x)) => x,
                Ok(Err(e)) => {
                    eprintln!(
                        "[hy4f-serve] request_id={:?}: call of {} tokens at {from}: {e}; failing it",
                        a.req.request_id,
                        tokens.len(),
                    );
                    let _ = a.req.tx.send(Event::Error(EngineError::engine(format!(
                        "hy_v4_flash route: {e}"
                    ))));
                    a.stream.client_gone = true;
                    a.fault = true;
                    done.push(i);
                    // a call that failed on the device leaves the context unusable: probe it
                    if let Err(probe) = route.dev.sync() {
                        device_fault = Some(format!(
                            "hy_v4_flash route stopped: the device probe after a failed call \
                             failed: {probe} (the call: {e})"
                        ));
                        break;
                    }
                    continue;
                }
                Err(payload) => {
                    device_fault = Some(format!(
                        "hy_v4_flash route stopped on a panic: {}",
                        panic_text(payload.as_ref())
                    ));
                    break;
                }
            };
            if let Some((s1, end)) = is_prefill_end {
                a.prefilled = s1;
                memra_engine::progress::note_prime_rows(tokens.len());
                if !end {
                    continue;
                }
            }
            a.steps += 1;
            // the tokens of the call's rows: each row's choice, kept while the next row's input
            // (the round's draft) is that choice; the first other choice ends the round
            let v = up.route.vocab_size();
            let nrows = if verify { tokens.len() } else { 1 };
            let mut kept = 0usize;
            // drafts the route's own choices equal, counted before a stop ends the round (a
            // matched end token or stop id counts and is not emitted)
            let mut matched = 0usize;
            let mut finished = false;
            let mut failed = None;
            let mut last = None;
            for r in 0..nrows {
                let row = if verify {
                    rows[r * v..(r + 1) * v].to_vec()
                } else {
                    rows.clone()
                };
                let pos = a.prompt.len() + a.stream.ids.len();
                let t = match choose(row, pos, a.sample.as_ref(), a.pen.as_ref(), &a.window) {
                    Ok(t) => t,
                    Err(e) => {
                        failed = Some(format!("hy_v4_flash sampler: {e}"));
                        break;
                    }
                };
                kept = r + 1;
                let matches = verify && r + 1 < tokens.len() && tokens[r + 1] == t;
                matched += usize::from(matches);
                if !a.stream.push(t) {
                    finished = true;
                    break;
                }
                if a.pen.is_some() {
                    a.window.push(t);
                }
                last = Some(t);
                if !matches {
                    break;
                }
            }
            if verify {
                let drafted = tokens.len() - 1;
                a.spec.rounds += 1;
                a.spec.drafted += drafted as u64;
                a.spec.accepted += matched as u64;
                // keep the anchor and the accepted drafts: the positions `from .. from + kept`
                let keep = from + kept.max(1);
                if let Some(b) = &backup
                    && let Err(e) = up.route.rollback_in(&mut a.sess, keep, b)
                {
                    failed = Some(format!("hy_v4_flash route: rollback: {e}"));
                }
            }
            if let Some(e) = failed {
                let _ = a.req.tx.send(Event::Error(EngineError::engine(e)));
                a.stream.client_gone = true;
                a.fault = true;
                done.push(i);
                continue;
            }
            if finished {
                done.push(i);
                continue;
            }
            let Some(t) = last else {
                done.push(i);
                continue;
            };
            // the cache holds `cap` positions: the token after the last one has no row
            if a.sess.positions() + 1 > a.cap {
                a.stream.stop = Some(StopReason::ContextFull);
                done.push(i);
                continue;
            }
            a.next = Some(t);
            // the next round, right after the call that scored its source row
            let room = (a.cap - a.sess.positions())
                .min(a.stream.budget.saturating_sub(a.stream.ids.len()));
            let k = cfg.drafts.min(room.saturating_sub(1));
            if k > 0 && !a.draft_off {
                let src = if verify {
                    kept.max(1) - 1
                } else {
                    tokens.len() - 1
                };
                let route = &mut up.route;
                let drafted = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    route
                        .draft_in(&mut a.sess, src, t, k, None, false)
                        .and_then(|d| {
                            if cfg.draft_fault == Some(DraftFault::Draft(a.spec.rounds)) {
                                Err(INJECTED.to_string())
                            } else {
                                Ok(d)
                            }
                        })
                }));
                match drafted {
                    Ok(Ok(d)) => a.drafts = d.ids,
                    Ok(Err(e)) => {
                        // a drafter fault costs speed, never the answer: the request goes on
                        // without drafts, on the calls the undrafted route makes. The round wrote
                        // only block rows past the session's positions, which the next call
                        // writes again before any query reads them. A device fault stops the route.
                        eprintln!(
                            "[hy4f-serve] request_id={:?}: draft round at {}: {e}; serving it \
                             without drafts",
                            a.req.request_id,
                            a.sess.positions()
                        );
                        a.draft_off = true;
                        if let Err(probe) = up.route.dev.sync() {
                            device_fault = Some(format!(
                                "hy_v4_flash route stopped: the device probe after a failed draft \
                                 round failed: {probe} (the round: {e})"
                            ));
                            break;
                        }
                    }
                    Err(payload) => {
                        device_fault = Some(format!(
                            "hy_v4_flash route stopped on a panic in a draft round: {}",
                            panic_text(payload.as_ref())
                        ));
                        break;
                    }
                }
            }
        }
        health.note_round();
        load.note_round(last_step.elapsed().as_millis() as u64);
        last_step = Instant::now();
        eprintln!(
            "[hy4f-step] step={} prefill_rows={prefill_rows} decode_rows={decode_rows} \
             step_s={:.3} running={} device_live={}",
            up.steps,
            ts.elapsed().as_secs_f64(),
            act.len(),
            up.route.dev.live_bytes()
        );
        if let Some(dead) = device_fault {
            eprintln!("[hy4f-serve] {dead}");
            fail_all(&mut act, health, &dead);
            drain_into(rx, &mut pending);
            fail_pending(&mut pending, health, &dead);
            return Some(dead);
        }
        done.sort_unstable();
        done.dedup();
        let freed = !done.is_empty();
        for &i in done.iter().rev() {
            let a = act.remove(i);
            finish(a, health);
        }
        // the finished sessions' caches go back to the system, so MemAvailable (the live
        // admission check, and every co-tenant) sees them
        if freed && let Err(e) = up.route.dev.trim_pool() {
            let dead = format!("hy_v4_flash route stopped: {e}");
            eprintln!("[hy4f-serve] {dead}");
            fail_all(&mut act, health, &dead);
            drain_into(rx, &mut pending);
            fail_pending(&mut pending, health, &dead);
            return Some(dead);
        }
    }
}

/// The route's renderer, on a thread of its own: every request's prompt is rendered and
/// validated here, so a burst of long prompts never stalls the serving thread's steps.
fn render_loop(
    tok: Arc<Tokenizer>,
    max_seq: usize,
    rx: Receiver<Box<Request>>,
    load: Arc<RouteLoad>,
    health: Arc<RouteHealth>,
    out: Sender<Prepared>,
) {
    while let Ok(req) = rx.recv() {
        let Some(pr) = intake(&tok, max_seq, req, &load, &health) else {
            continue;
        };
        if let Err(std::sync::mpsc::SendError(mut pr)) = out.send(pr) {
            fail_request(
                &mut pr.req,
                EngineError::engine("the hy_v4_flash route stopped serving".to_string()),
            );
            pr.run.fail();
            health.end_request();
            break;
        }
    }
}

/// A request off the route's queue: booked, rendered and validated (`None`: answered here, or its
/// client already left).
fn intake(
    tok: &Arc<Tokenizer>,
    max_seq: usize,
    mut req: Box<Request>,
    load: &Arc<RouteLoad>,
    health: &Arc<RouteHealth>,
) -> Option<Prepared> {
    let run = load.begin();
    if let Some(ticket) = req.route_ticket.as_ref() {
        load.record_wait(ticket.waited());
    }
    crate::worker::release_request_reservation(&mut req);
    if req.tx.is_closed() {
        run.cancel();
        return None;
    }
    health.begin_request();
    match prepare(tok, max_seq, &mut req) {
        Ok(Rendered {
            prompt,
            budget,
            sample,
            pen,
        }) => Some(Prepared {
            req,
            run,
            deferred: false,
            prompt,
            budget,
            sample,
            pen,
        }),
        Err(e) => {
            settle(run, &mut req, Err(e));
            health.end_request();
            None
        }
    }
}

fn finish(mut a: Active, health: &Arc<RouteHealth>) {
    let elapsed = a.t0.elapsed().as_secs_f64();
    let spec = (a.spec.rounds > 0).then_some(a.spec);
    if let Some(u) = spec {
        eprintln!(
            "[hy4f-draft] request_id={:?} rounds={} drafted={} accepted={} tokens={}",
            a.req.request_id,
            u.rounds,
            u.drafted,
            u.accepted,
            a.stream.ids.len()
        );
    }
    a.stream.finish(a.prompt.len(), elapsed, spec);
    let stats = ServeStats {
        tokens_out: a.stream.ids.len(),
        n_prompt: a.prompt.len(),
        n_cached: 0,
        rounds: a.steps,
    };
    if a.fault {
        health.note_request_fault();
    }
    if let Some(run) = a.run.take() {
        if a.fault {
            run.fail();
        } else if a.stream.client_gone {
            run.cancel();
        } else {
            run.finish(stats);
        }
    }
    health.end_request();
}

/// Answer every running request with the route's error: each is a request the route failed.
fn fail_all(act: &mut Vec<Active>, health: &Arc<RouteHealth>, why: &str) {
    eprintln!("[hy4f-serve] {why}; failing {} request(s)", act.len());
    for mut a in act.drain(..) {
        if !a.stream.client_gone {
            fail_request(&mut a.req, EngineError::engine(why.to_string()));
        }
        health.note_request_fault();
        if let Some(run) = a.run.take() {
            run.fail();
        }
        health.end_request();
    }
}

/// Move what `rx` holds now into `pending`, in arrival order, without waiting; true once the
/// sender is gone.
fn drain_into<T>(rx: &Receiver<T>, pending: &mut VecDeque<T>) -> bool {
    loop {
        match rx.try_recv() {
            Ok(x) => pending.push_back(x),
            Err(std::sync::mpsc::TryRecvError::Empty) => return false,
            Err(std::sync::mpsc::TryRecvError::Disconnected) => return true,
        }
    }
}

/// Answer every request still waiting for memory with `why`: the route stops serving.
fn fail_pending(pending: &mut VecDeque<Prepared>, health: &Arc<RouteHealth>, why: &str) {
    for mut pr in pending.drain(..) {
        fail_request(&mut pr.req, EngineError::engine(why.to_string()));
        pr.run.fail();
        health.end_request();
    }
}

/// The text of a caught panic's payload.
fn panic_text(payload: &(dyn std::any::Any + Send)) -> String {
    payload
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
        .unwrap_or_else(|| "non-string panic payload".into())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn book(ceiling: usize) -> Book {
        Book {
            ceiling,
            chunk: 2048,
            // the artifact's shapes: 12 indexed layers of (576 + 128) F32 a position plus 128 F32
            // a group, 30 sliding rings of 512 rows
            state: SessionShape {
                fixed: 30 * 512 * 576 * 4 + 12 * (576 * 4 + 128 * 4 + 128 * 4),
                per_pos: 12 * (576 * 4 + 128 * 4) + 12 * 128,
                index_per_row_pos: 1,
                index_fixed_per_row: 4,
            },
            draft_rows: 0,
        }
    }

    /// A drafting route books every session's round of positions: the anchor and its drafts.
    #[test]
    fn a_drafting_route_books_one_rounds_positions_with_every_session() {
        let plain = book(24 << 30);
        let mut drafting = book(24 << 30);
        drafting.draft_rows = 6;
        assert_eq!(drafting.session_cap(1000), 1006);
        assert_eq!(drafting.state(1000), plain.state(1006));
        assert_eq!(
            drafting.total(&[1000], Some(2000)),
            plain.total(&[1006], Some(2006)) - plain.transient(2006) + drafting.transient(2000)
        );
        assert!(drafting.largest_fit() < plain.largest_fit());
    }

    #[test]
    fn the_draft_count_is_the_artifacts_unless_set() {
        assert_eq!(resolve_drafts(None, Some(5)).unwrap(), 5);
        assert_eq!(resolve_drafts(None, None).unwrap(), 0);
        assert_eq!(resolve_drafts(Some("0"), Some(5)).unwrap(), 0);
        assert_eq!(resolve_drafts(Some("3"), Some(5)).unwrap(), 3);
        assert!(resolve_drafts(Some("17"), Some(5)).is_err());
        assert!(resolve_drafts(Some("x"), Some(5)).is_err());
        assert_eq!(resolve_draft_fault(None).unwrap(), None);
        assert_eq!(
            resolve_draft_fault(Some("draft:3")).unwrap(),
            Some(DraftFault::Draft(3))
        );
        assert_eq!(
            resolve_draft_fault(Some("backup:0")).unwrap(),
            Some(DraftFault::Backup(0))
        );
        assert!(resolve_draft_fault(Some("3")).is_err());
        assert!(resolve_draft_fault(Some("draft:x")).is_err());
    }

    #[test]
    fn the_book_refuses_what_never_fits_and_queues_what_fits_alone() {
        let b = book(24 << 30);
        let fit = b.largest_fit();
        assert!(b.total(&[], Some(fit)) <= b.ceiling);
        assert!(b.total(&[], Some(fit + 4)) > b.ceiling);
        // a 256k session fits the 24 GiB a GB10 leaves beside the weights, twice it does not
        assert!(fit >= 262_144, "{fit}");
        assert!(b.total(&[262_144], Some(262_144)) <= b.ceiling);
        assert!(b.total(&[262_144, 262_144], Some(262_144)) > b.ceiling);
        // the transient is booked once, not per session
        let two = b.total(&[8192], Some(8192));
        assert_eq!(two, 2 * b.state(8192) + b.transient(8192));
    }

    #[test]
    fn a_stop_string_ends_after_its_token_and_the_calls_close_stays_whole() {
        let stops = vec!["STOP".to_string(), HY4_CALLS_CLOSE.to_string()];
        assert_eq!(stop_end("abc", "abcST", &stops), None);
        assert_eq!(stop_end("abcST", "abcSTOPxy", &stops), Some(9));
        let full = format!("x{HY4_CALLS_CLOSE}tail");
        assert_eq!(
            stop_end("x", &full, &stops),
            Some(1 + HY4_CALLS_CLOSE.len())
        );
    }

    #[test]
    fn the_greedy_pick_is_the_first_of_equal_maxima() {
        assert_eq!(argmax(&[0.0, 2.0, 2.0, 1.0]), 1);
        assert!(choose(vec![0.0, f32::NAN], 0, None, None, &[]).is_err());
    }

    /// The live check reserves the scratch a running long session still owes, not only the new
    /// one's: a small request beside a long prefill must leave its growth free.
    #[test]
    fn the_live_check_keeps_a_running_sessions_scratch_growth() {
        let b = book(24 << 30);
        let held = b.index_scratch(4096);
        assert_eq!(live_scratch_growth(&b, &[], 4096, held), 0);
        let long = live_scratch_growth(&b, &[600_000], 4096, held);
        assert_eq!(long, b.index_scratch(600_000) - held);
        assert!(long > 0);
        // a scratch already grown for the long session is not booked twice
        assert_eq!(
            live_scratch_growth(&b, &[600_000], 4096, b.index_scratch(600_000)),
            0
        );
    }

    #[test]
    fn a_requests_max_ctx_tightens_the_served_context() {
        assert_eq!(request_ctx(4096, None), 4096);
        assert_eq!(request_ctx(4096, Some(110)), 110);
        assert_eq!(request_ctx(4096, Some(1 << 20)), 4096);
    }

    #[test]
    fn the_prefill_chunk_is_a_positive_count() {
        assert_eq!(resolve_chunk(None).unwrap(), DEFAULT_PREFILL_CHUNK);
        assert_eq!(resolve_chunk(Some("500")).unwrap(), 500);
        assert!(resolve_chunk(Some("0")).is_err());
        assert!(resolve_chunk(Some("x")).is_err());
    }
}
