//! deepseek-v4.1 serve route: DeepSeek-V4.1-Flash through the worker's `Cmd::Generate(Request) ->
//! Event` contract, on the engine's V4.1 decode loop (`memra_engine::dsv41_gpu::decode`), never
//! through `HybridModel` or the DSv4 stack. The HTTP surface (OpenAI routes, the `encoding_dsv41`
//! renderer inputs, the V4.1 DSML/reasoning parser, streaming SSE) is the shared one; this module
//! is the worker side.
//!
//! The program. Every request runs the checkpoint's own numeric program, the one the decode lane
//! gated against the vendor (`research/modelplan-onboarding-dsv41-20261109/decode`): its prompt as
//! one prefill call at position 0 (the vendor's prefill program), then each generated token as one
//! decode call (the vendor's decode program). A generated token is never prefilled again and a
//! later turn is a new request with its own prefill (no continuation, no prefix reuse), so no
//! request crosses between the vendor's two programs, which below the 128-token window give
//! different bits for the same row.
//!
//! The engine boundary is the runner's rank group (`dsv41_gpu::tp::Group`): the route speaks
//! `Runner::forward` / `Runner::logits` over `Seq`s and never a card, and every residency choice
//! lives in `bring_up`. `MEMRA_DSV41_DEVICES` names the ranks; the session, sampler, stop and
//! encoding layer is the same over either residency.
//!
//! Residency follows the cards (`experts_on_card`). Where every rank's routed experts fit its
//! card (four ranks on four 96 GB cards), the whole trunk and the head load onto the ranks at
//! start and stay (`Runner::load_resident`: each card's budget printed, the boot refused when
//! the weights and one full prefill slice's request do not fit): no host store, no cache, and
//! the memory book admits requests against what the weights leave free with the runner's N-rank
//! arithmetic (`decode::prefill_workspace_bytes`, `decode::seq_state_bytes`). Where they do not
//! (one card), every trunk layer's own weights and the head stay
//! on the card and the routed experts live in pinned host memory
//! (`dsv41_gpu::experts::HostExperts`) behind each rank's device LRU cache, which fills the free
//! memory and gives back slabs, least recently used first, when a prompt's prefill needs the
//! room. The cache moves bytes, never values: a slot holds the checkpoint's MXFP4 expert as
//! stored and the grouped GEMMs read it through the same table.
//!
//! Batching. Up to `MEMRA_MAX_SESSIONS` requests run at once. Each step is one runner forward
//! with at most one new prompt's prefill call and one decode call per running request; the runner
//! runs every call alone through each block, so a request's numbers are the same served alone or
//! beside others (the serve receipts gate this byte for byte).
//!
//! Sampling. Greedy is the argmax, the lowest id among equal maxima (the decode lane's rule, so a
//! served greedy answer is `dsv41-run --gen`'s). Temperature, top-p and top-k draw through the
//! DSv4 route's position-keyed host sampler (`dsv4_sample_row`, seeded and reproducible),
//! penalties through `dsv4_penalize_row` over the prompt and the generated tokens; `min_p` is
//! refused, as on the DSv4 route. Stops: the end-of-sentence id, the request's stop ids, stop
//! strings over the detokenized stream (the V4.1 DSML close stays in the stream so the parser
//! closes the call), and the token budget.
//!
//! Deliberate limits, stated: `response_format`/grammar refused by name; image parts are refused
//! by the HTTP layer (no V4.1 vision program); no DSpark drafts, prefix cache or CUDA graphs.

use crate::dsv4_admit::Admission;
use crate::dsv4_serve::{admission_answer, fail_request, scan_stop_cut_with, settle, snap};
use crate::health::RouteHealth;
use crate::route_telemetry::{RouteLoad, RouteRun, ServeStats};
use crate::worker::{EngineError, Event, EventSender, ModelCaps, Request};
use memra_engine::decode::StopReason;
use memra_engine::dsv4_gpu::{Dsv4PenaltyCfg, Dsv4SampleCfg, dsv4_penalize_row, dsv4_sample_row};
use memra_engine::dsv41_gpu::decode::{Call, Runner, Seq};
use memra_engine::dsv41_gpu::engram::{EngramConsts, token_map_from_file};
use memra_engine::dsv41_gpu::experts::{HostExperts, Layout};
use memra_engine::dsv41_gpu::{self as v41, Geo};
use memra_tokenizer::{Tokenizer, chat};
use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::mpsc::{Receiver, Sender};
use std::time::Instant;

/// Prefill query rows per slice: the decode lane's runner default, the size every gate ran (the
/// slice never changes a value: every sliced step acts per row).
const PREFILL_CHUNK: usize = 8192;

/// Device bytes the memory book keeps beside its itemized terms on every rank: allocator
/// granularity and the per-layer temporaries the shape model does not list (mixes, the MoE plan
/// tables, index rows). Every forward logs the pool's high-water mark against the model.
const MARGIN_BYTES: u64 = 1 << 30;

/// The V4.1 parser-closing stop, kept in the stream like the DSv4 close.
const DSML_V41_CLOSE: &str = "</\u{ff5c}DSML\u{ff5c} calls>";

/// Cheap boot-time probe: the effective `model_type` (text_config over top level, then
/// `architectures[0]`) maps to the deepseek-v41 architecture. The loader re-validates strictly.
pub fn is_dsv41_dir(dir: &Path) -> bool {
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
    memra_gguf::config::Arch::from_hf_model_type(mt).is_dsv41()
}

/// The device memory one request costs on each rank, from the program's shapes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct MemModel {
    pub layers: usize,
    pub window: usize,
    pub head_dim: usize,
    pub index_dim: usize,
    pub hc: usize,
    pub dim: usize,
    pub heads: usize,
    pub topk: usize,
    pub moe_inter: usize,
    pub index_heads: usize,
    pub index_topk: usize,
    pub engram_rows: usize,
    pub nkv: usize,
    /// the compress ratio of each kv-source group
    pub group_ratios: [usize; 4],
    pub n_groups: usize,
    pub chunk: usize,
}

impl MemModel {
    fn of(g: &Geo, layers: usize, chunk: usize) -> Self {
        let mut group_ratios = [0usize; 4];
        let n_groups = g.kv_sources.len().min(4);
        for (i, &k) in g.kv_sources.iter().take(4).enumerate() {
            group_ratios[i] = g.compress_ratios[k].max(1);
        }
        Self {
            layers,
            window: g.window,
            head_dim: g.head_dim,
            index_dim: g.index_dim,
            hc: g.hc,
            dim: g.dim,
            heads: g.heads,
            topk: g.topk,
            moe_inter: g.moe_inter,
            index_heads: g.index_heads,
            index_topk: g.index_topk,
            engram_rows: g.engram_cols() * g.engram_head_dim,
            nkv: g.dim * (g.hc + 1),
            group_ratios,
            n_groups,
            chunk,
        }
    }

    /// A sequence's carried state at `cap` positions: per layer the window ring (bf16) and the
    /// ratio-2 compressor's open pair (fp32 kv and gate), per kv-source group the compressed KV
    /// (bf16 rows of `head_dim`) and the index keys (bf16 rows of `index_dim`).
    pub fn state_bytes(&self, cap: usize) -> u64 {
        let ring = self.layers * self.window * self.head_dim * 2;
        let pairs = self.layers * 2 * 2 * self.head_dim * 4;
        let rows: usize = self.group_ratios[..self.n_groups]
            .iter()
            .map(|&r| (cap / r + 1) * (self.head_dim * 2 + self.index_dim * 2))
            .sum();
        (ring + pairs + rows) as u64
    }

    /// The transient of one prefill call of `s` tokens, every buffer counted as live at once:
    /// the residual streams the forward carries (`[s][hc][dim]` bf16 in, after attention, out,
    /// plus the fp32 pre-mix), the `[s][dim]` sublayer rows, the keys side (latent, window and
    /// compressed rows, the selection, the candidate pool `[s][s/8]`, the compressor's fp32 rows),
    /// and the per-slice terms (query and output heads, the indexer's scores, the MoE pairs, the
    /// Engram rows) at `chunk` rows.
    pub fn prefill_bytes(&self, s: usize) -> u64 {
        let c = s.min(self.chunk).max(1);
        let (d, hd) = (self.dim, self.head_dim);
        let per_tok = 3 * self.hc * d * 2      // streams
            + self.hc * 4 * 4                  // pre-mixes and post/comb mixes
            + 4 * d * 2                        // attention in/out, ffn in/out
            + 1280 * 2                         // query latent
            + 2 * hd * 2                       // window and compressed key rows
            + (self.window + self.index_topk) * 4 // the selection
            + hd * 2 + 2 * hd * 4; //            compressor latent and fp32 rows
        let pairs = c * self.topk;
        let per_slice = c * self.heads * hd * 2 * 2     // query and output heads
            + c * (self.heads / 8).max(1) * 1024 * 2     // wo_a rows
            + c * self.index_heads * self.index_dim * 2  // index queries
            + pairs * (d * 2 + d + d / 32)               // expert rows, quantized inputs
            + pairs * self.moe_inter * (3 * 2 + 1)       // gate/up/hidden, quantized hidden
            + c * (self.moe_inter * 2 * 3 + d * 2)       // shared expert
            + c * (self.engram_rows * 2 + self.nkv * 2); // Engram rows and projections
        let quadratic = (s as u64) * (s as u64) / 8 // candidate pool
            + 2 * (c as u64) * (s as u64); //          index scores
        (per_tok * s + per_slice) as u64 + quadratic
    }

    /// The transient of one decode call at up to `cap` positions: the heads' key rows (window plus
    /// every compressed row), the index scores and candidate row, and one row's slice terms.
    pub fn decode_bytes(&self, cap: usize) -> u64 {
        let keys = (self.window + cap) * self.head_dim * 2;
        let scores = cap * 2 + cap / 8;
        (keys + scores) as u64 + self.prefill_bytes(1)
    }
}

/// Validate `MEMRA_REWRITE_BUNDLE` for this checkpoint: its `artifact.lock` names this family,
/// this `config.json` and the plan this binary compiles from it; a rewrite receipt index, when
/// present, must load against that plan. The route runs one program, the plan's own eager decode
/// loop, so no receipt widens what it runs; a stale or foreign bundle refuses the boot.
pub(crate) fn install_v41_rewrite_bundle(
    bundle: &Path,
    dir: &Path,
    mc: &memra_gguf::config::ModelConfig,
) -> Result<String, String> {
    use sha2::Digest;
    let hex = |b: &[u8]| {
        sha2::Sha256::digest(b)
            .iter()
            .map(|x| format!("{x:02x}"))
            .collect::<String>()
    };
    let lock = std::fs::read_to_string(bundle.join("artifact.lock"))
        .map_err(|e| format!("rewrite bundle {}: artifact.lock: {e}", bundle.display()))?;
    let field = |k: &str| {
        lock.lines()
            .find_map(|l| l.strip_prefix(&format!("{k}=")))
            .map(str::to_string)
    };
    if field("family").as_deref() != Some("deepseek_v41") {
        return Err(format!(
            "rewrite bundle {}: artifact.lock family {:?} is not deepseek_v41",
            bundle.display(),
            field("family")
        ));
    }
    let config = std::fs::read(dir.join("config.json")).map_err(|e| e.to_string())?;
    if field("config_sha256").as_deref() != Some(hex(&config).as_str()) {
        return Err(format!(
            "rewrite bundle {}: artifact.lock config_sha256 does not match {}",
            bundle.display(),
            dir.join("config.json").display()
        ));
    }
    let pack = memra_gguf::model_packs::by_alias("deepseek_v41")
        .ok_or("the deepseek_v41 pack is not in this build")?;
    let plan = pack.compile_plan(mc).map_err(|e| e.to_string())?;
    let plan_hash = hex(format!("{plan:#?}\n").as_bytes());
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
pub struct Dsv41Model {
    pub tok: Arc<Tokenizer>,
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
    devices: Vec<usize>,
    /// `MEMRA_DSV41_TRUNK_LAYERS`: a shortened trunk (a rehearsal of the memory book and the
    /// serving loop on fewer cards; the answers are not the checkpoint's)
    trunk_layers: Option<usize>,
    token_map: PathBuf,
    max_seq: usize,
    sessions: usize,
    eos: u32,
}

/// `MEMRA_DSV41_TRUNK_LAYERS`: a positive layer count, or unset for the checkpoint's trunk.
fn resolve_trunk_layers(raw: Option<&str>) -> Result<Option<usize>, String> {
    match raw.map(str::trim) {
        None | Some("") => Ok(None),
        Some(x) => match x.parse::<usize>() {
            Ok(n) if n > 0 => Ok(Some(n)),
            _ => Err(format!(
                "MEMRA_DSV41_TRUNK_LAYERS {x:?} is not a positive layer count"
            )),
        },
    }
}

fn resolve_devices(raw: Option<&str>) -> Result<Vec<usize>, String> {
    let Some(s) = raw else {
        return Ok(vec![0]);
    };
    let d: Vec<usize> = s
        .split(',')
        .map(|x| {
            x.trim()
                .parse()
                .map_err(|_| format!("MEMRA_DSV41_DEVICES {s:?} is not a list of device ordinals"))
        })
        .collect::<Result<_, _>>()?;
    if d.is_empty() {
        return Err("MEMRA_DSV41_DEVICES names no device".into());
    }
    Ok(d)
}

/// Load the route: validate the checkpoint, the tokenizer's dialect, the context and the bundle
/// here, then bring the model up on its own serving thread and wait for it.
pub fn load(name: &str, dir: &Path, tok: Arc<Tokenizer>) -> Result<Dsv41Model, String> {
    if tok.dsv4_encoding() != Some(chat::Dsv4Encoding::V41) {
        return Err(format!(
            "deepseek-v4.1 model {name:?}: the tokenizer census is not encoding_dsv41 ({:?}); \
             the route renders only the V4.1 dialect",
            tok.dsv4_encoding()
        ));
    }
    let mc = memra_gguf::config::ModelConfig::from_config_json(&dir.join("config.json"))
        .map_err(|e| format!("deepseek-v4.1 model {name:?}: config.json: {e}"))?;
    let g = Geo::from_config(&mc)?;
    if mc.context_length == 0 {
        return Err(format!(
            "deepseek-v4.1 model {name:?}: config.json declares no max_position_embeddings"
        ));
    }
    let declared = mc.context_length as usize;
    let max_seq = crate::worker::resolve_env_ctx(declared).map_err(|e| {
        format!("deepseek-v4.1 model {name:?}: {e} (config.json declares {declared} positions)")
    })?;
    let devices = resolve_devices(std::env::var("MEMRA_DSV41_DEVICES").ok().as_deref())?;
    v41::Part::check(&g, devices.len())?;
    let trunk_layers =
        resolve_trunk_layers(std::env::var("MEMRA_DSV41_TRUNK_LAYERS").ok().as_deref())?;
    let token_map = std::env::var_os("MEMRA_DSV41_TOKEN_MAP")
        .map(PathBuf::from)
        .ok_or_else(|| {
            format!(
                "deepseek-v4.1 model {name:?}: the Engram hash needs the compressed token map; \
                 set MEMRA_DSV41_TOKEN_MAP (memra does not derive it from the tokenizer yet)"
            )
        })?;
    if let Some(bundle) = std::env::var_os("MEMRA_REWRITE_BUNDLE") {
        let plan = install_v41_rewrite_bundle(Path::new(&bundle), dir, &mc)?;
        eprintln!(
            "[dsv41-serve] {name}: rewrite bundle {} installed for plan {plan}; the route runs the \
             plan's own eager decode loop, no rewrite",
            Path::new(&bundle).display()
        );
    }
    let sessions = crate::route_contract::hybrid_interactive_cap();
    if sessions == 0 {
        return Err("MEMRA_MAX_SESSIONS=0 leaves the deepseek-v4.1 route no session".into());
    }
    let eos = tok.eos_id();
    let cfg = EngineCfg {
        name: name.to_string(),
        dir: dir.to_path_buf(),
        tok: tok.clone(),
        devices,
        trunk_layers,
        token_map,
        max_seq,
        sessions,
        eos,
    };
    let (ready_tx, ready_rx) = std::sync::mpsc::channel::<Result<Ready, String>>();
    let (start_tx, start_rx) = std::sync::mpsc::channel::<Start>();
    let handle = std::thread::Builder::new()
        .name(format!("dsv41-serve-{name}"))
        .spawn(move || engine_thread(cfg, ready_tx, start_rx))
        .map_err(|e| format!("spawn the deepseek-v4.1 serving thread: {e}"))?;
    THREADS
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .push(handle);
    let ready = ready_rx
        .recv()
        .map_err(|_| format!("deepseek-v4.1 model {name:?}: the serving thread died loading"))??;
    Ok(Dsv41Model {
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
    crate::route_contract::RouteContract::dsv41_thread(model, sessions)
}

/// ModelCaps for /v1/models and the HTTP gates: the V4.1 dialect, keyed on the tokenizer census
/// (`Dsv4Encoding::V41`, which `load` required), with every law the two DeepSeek dialects share.
pub fn caps(m: &Dsv41Model) -> ModelCaps {
    ModelCaps {
        max_choices: 1,
        hy3: false,
        tools_branch: true,
        qwen_think: false,
        think_switch: false,
        chat_ok: true,
        context_length: m.max_seq,
        tokenizer: m.tok.pre().to_string(),
        instruct_type: Some("deepseek".to_string()),
        effort_levels: false,
        qwen_effort: false,
        gemma_think: false,
        gemma_unopened_thought: false,
        dsv4: true,
        dsv41: true,
        glm5: false,
        chat_temperature_default: None,
        chat_top_p_default: None,
        n_vocab: m.tok.vocab_size(),
        think_close: Vec::new(),
    }
}

/// Start serving: the returned sender is the model's admission queue.
pub fn spawn(
    m: Dsv41Model,
    health: Arc<RouteHealth>,
    load: Arc<RouteLoad>,
) -> Sender<Box<Request>> {
    let (tx, rx) = std::sync::mpsc::channel::<Box<Request>>();
    // the engine thread is parked on this start; a send failure means it already died, and the
    // closed queue then answers every request with the route's error
    let _ = m.start.send(Start { rx, health, load });
    tx
}

/// Latches the route dead when the serving thread ends for any reason.
struct ExitLatch(Arc<RouteHealth>);

impl Drop for ExitLatch {
    fn drop(&mut self) {
        self.0.mark_dead(if std::thread::panicking() {
            "deepseek-v4.1 serving thread panicked"
        } else {
            "deepseek-v4.1 serving thread exited"
        });
    }
}

/// The detokenized stream of one request and its stop rules: EOS, stop ids, the budget, and stop
/// strings through the DSv4 route's scan, with a client's stop sent whole ([`stop_end`]).
struct Stream {
    tok: Arc<Tokenizer>,
    tx: EventSender,
    eos: Vec<u32>,
    stops: Vec<String>,
    stop_ids: Vec<u32>,
    budget: usize,
    ids: Vec<u32>,
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
        let full = self.tok.decode(&self.ids);
        let cut = stop_end(&self.text, &full, &self.stops);
        let start = snap(&full, self.text.len());
        if let Some(abs) = cut {
            let delta = full[start..snap(&full, abs)].to_string();
            self.text.push_str(&delta);
            if self.tx.send(Event::Token { id, text: delta }).is_err() {
                self.client_gone = true;
            }
            self.stop = Some(StopReason::Callback);
            return false;
        }
        let delta = full[start..].to_string();
        self.text = full;
        if self.tx.send(Event::Token { id, text: delta }).is_err() {
            self.client_gone = true;
            return false;
        }
        if self.ids.len() >= self.budget {
            self.stop = Some(StopReason::MaxNew);
            return false;
        }
        true
    }

    fn finish(&self, n_prompt: usize, elapsed_s: f64) {
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
            spec: None,
        });
    }
}

/// Where the stream ends when a stop string completes in `full`, whose first `text.len()` bytes
/// are already sent (None: no stop). A client's stop ends it after the whole token, the worker's
/// law: the HTTP layer finds the complete stop text and cuts there (`truncate_at_stop`, the
/// stream scrubber), so a stop that straddles two tokens leaves no prefix of itself in the
/// answer. The parser-closing DSML stop alone ends the stream at its close.
fn stop_end(text: &str, full: &str, stops: &[String]) -> Option<usize> {
    let cut = scan_stop_cut_with(text, full, stops, &[DSML_V41_CLOSE])?;
    let client = stops
        .iter()
        .filter(|s| s.as_str() != DSML_V41_CLOSE)
        .any(|s| scan_stop_cut_with(text, full, std::slice::from_ref(s), &[]).is_some());
    Some(if client { full.len() } else { cut })
}

/// A request the route is serving.
struct Active {
    req: Box<Request>,
    run: Option<RouteRun>,
    prompt_len: usize,
    cap: usize,
    stream: Stream,
    /// the token the next decode call feeds; None until the prompt's prefill ran
    next: Option<u32>,
    sample: Option<Dsv4SampleCfg>,
    pen: Option<Dsv4PenaltyCfg>,
    /// the penalty window: the prompt and every generated token (only with penalties)
    window: Vec<u32>,
    t0: Instant,
    steps: usize,
}

/// A request past rendering, waiting for its memory.
struct Prepared {
    req: Box<Request>,
    run: RouteRun,
    prompt: Vec<u32>,
    budget: usize,
    sample: Option<Dsv4SampleCfg>,
    pen: Option<Dsv4PenaltyCfg>,
}

/// Render and validate a request for the route (`Err` answers it).
/// A rendered request's prompt, token budget and sampler.
struct Rendered {
    prompt: Vec<u32>,
    budget: usize,
    sample: Option<Dsv4SampleCfg>,
    pen: Option<Dsv4PenaltyCfg>,
}

fn prepare(tok: &Tokenizer, max_seq: usize, req: &mut Request) -> Result<Rendered, EngineError> {
    if req.grammar.is_some() {
        return Err(EngineError::invalid_param(
            "response_format is not available on the deepseek-v4.1 route yet",
            "response_format",
        ));
    }
    if !req.images.is_empty() {
        return Err(EngineError::invalid_param(
            "image parts are not available on the deepseek-v4.1 route (no V4.1 vision program)",
            "messages",
        ));
    }
    let prompt = crate::dsv4_serve::render_prompt_tok(tok, req)?;
    if prompt.len() >= max_seq {
        return Err(EngineError::context_length(format!(
            "prompt ({} tok) leaves no room in the {max_seq}-position context",
            prompt.len()
        )));
    }
    let budget = req.params.max_new.min(max_seq - prompt.len()).max(1);
    let sc = &req.sampler_cfg;
    let greedy = sc.temperature <= 0.0;
    if !greedy && sc.min_p != 0.0 {
        return Err(EngineError::invalid_param(
            "min_p is not implemented on the deepseek-v4.1 sampler; use top_p/top_k",
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

/// The next token from one logits row: penalties over the window, then the argmax (lowest id among
/// equal maxima) or the position-keyed draw at absolute position `pos`.
fn choose(
    mut row: Vec<f32>,
    pos: usize,
    sample: Option<&Dsv4SampleCfg>,
    pen: Option<&Dsv4PenaltyCfg>,
    window: &[u32],
) -> Result<u32, String> {
    if let Some(p) = pen {
        dsv4_penalize_row(&mut row, window, p);
    }
    match sample {
        None => {
            if row.iter().any(|x| x.is_nan()) {
                return Err("non-finite logits".into());
            }
            Ok(memra_engine::dsv41_gpu::decode::argmax(&row) as u32)
        }
        Some(cfg) => dsv4_sample_row(&row, pos, cfg),
    }
}

/// The route's memory book on each rank: the ceiling (effective free with the resident weights
/// loaded and no cache), the cache floor, and what the running requests have booked.
struct Book {
    model: MemModel,
    ceiling: Vec<u64>,
    floor_cache: u64,
    /// the routed experts are on the cards: the N-rank arithmetic of the decode runner
    /// (`decode::prefill_workspace_bytes`, `decode::seq_state_bytes`) books a request
    resident: Option<(Geo, usize)>,
}

impl Book {
    /// Bytes a step must find free beside the cache and the running requests' allocated state:
    /// a new request's state and the step's largest transient (the new prompt's prefill or the
    /// largest decode call), plus the margin.
    fn state(&self, cap: usize) -> u64 {
        match &self.resident {
            Some((g, _)) => v41::decode::seq_state_bytes(g, self.model.layers, cap) as u64,
            None => self.model.state_bytes(cap),
        }
    }

    fn prefill(&self, s: usize) -> u64 {
        match &self.resident {
            Some((g, n)) => v41::decode::prefill_workspace_bytes(g, *n, s, self.model.chunk) as u64,
            None => self.model.prefill_bytes(s),
        }
    }

    fn live_need(&self, caps: &[usize], new: Option<(usize, usize)>) -> u64 {
        let decode = caps
            .iter()
            .chain(new.as_ref().map(|(_, c)| c))
            .map(|&c| self.model.decode_bytes(c))
            .max()
            .unwrap_or(0);
        let (new_state, new_prefill) =
            new.map_or((0, 0), |(p, c)| (self.state(c), self.prefill(p)));
        new_state + decode.max(new_prefill) + MARGIN_BYTES
    }

    /// Everything a step books on a rank: the running requests' state and [`Self::live_need`].
    fn total(&self, caps: &[usize], new: Option<(usize, usize)>) -> u64 {
        let state: u64 = caps.iter().map(|&c| self.state(c)).sum();
        state + self.live_need(caps, new)
    }

    /// How many sessions of `cap` positions with `prompt`-token prompts the route can run at
    /// once (each admitted beside the ones already running).
    fn sessions_fit(&self, prompt: usize, cap: usize, most: usize) -> usize {
        let top = self.ceiling.iter().min().copied().unwrap_or(0);
        let mut running = Vec::new();
        while running.len() < most
            && self.total(&running, Some((prompt, cap))) + self.floor_cache <= top
        {
            running.push(cap);
        }
        running.len()
    }

    /// The largest session a request with a `prompt`-token prompt can have with the route idle.
    fn largest_fit(&self, prompt: usize) -> usize {
        let top = self.ceiling.iter().min().copied().unwrap_or(0);
        let fits = |c: usize| self.total(&[], Some((prompt.min(c), c))) + self.floor_cache <= top;
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

/// The memory door for the request at the queue head: `Ok(true)` admits (the cache has given
/// back what the step needs), `Ok(false)` waits for running requests to finish, `Err` answers it
/// (no session that size fits the card even with the route idle).
fn admit_dsv41_session(
    rn: &mut Runner,
    book: &Book,
    running: &[usize],
    prompt: usize,
    cap: usize,
) -> Result<bool, Admission> {
    // alone on the route: what the session costs against what the card offers beside the
    // resident weights and the cache floor
    let alone = book.total(&[], Some((prompt, cap)));
    for (dev, &ceiling) in book.ceiling.iter().enumerate() {
        if alone + book.floor_cache > ceiling {
            return Err(Admission::NeverFits {
                dev,
                need: alone,
                ceiling: ceiling.saturating_sub(book.floor_cache),
            });
        }
    }
    if book.total(running, Some((prompt, cap))) + book.floor_cache
        > book.ceiling.iter().min().copied().unwrap_or(0)
    {
        return Ok(false);
    }
    let need = book.live_need(running, Some((prompt, cap)));
    match rn.make_room(need) {
        // with nothing running the card holds only the weights and the cache floor: a shortfall
        // there is a co-tenant's, and the step's own error answers the request
        Ok(free) => Ok(free.iter().all(|&f| f >= need) || running.is_empty()),
        Err(e) => {
            eprintln!("[dsv41-serve] make room for {need} bytes: {e}");
            Ok(false)
        }
    }
}

fn engine_thread(cfg: EngineCfg, ready: Sender<Result<Ready, String>>, start: Receiver<Start>) {
    let t0 = Instant::now();
    let stm = match memra_gguf::safetensors::StModel::open(&cfg.dir) {
        Ok(s) => s,
        Err(e) => {
            let _ = ready.send(Err(format!("open {}: {e}", cfg.dir.display())));
            return;
        }
    };
    let mut up = match bring_up(&cfg, &stm) {
        Ok(u) => u,
        Err(e) => {
            let _ = ready.send(Err(format!("deepseek-v4.1 model {:?}: {e}", cfg.name)));
            return;
        }
    };
    eprintln!(
        "[dsv41-serve] {}: up in {:.1}s",
        cfg.name,
        t0.elapsed().as_secs_f64()
    );
    let _ = ready.send(Ok(Ready {
        max_seq: cfg.max_seq,
    }));
    let Ok(Start { rx, health, load }) = start.recv() else {
        return;
    };
    let _latch = ExitLatch(health.clone());
    let sink = health.clone();
    let _progress = memra_engine::progress::ProgressSinkScope::install(Box::new(move |rows| {
        sink.note_rows(rows)
    }));
    serve_loop(&cfg, &mut up, &rx, &health, &load);
}

/// The model on the card, ready to serve.
struct Up<'a> {
    rn: Runner<'a>,
    book: Book,
    max_slots: usize,
}

fn bring_up<'a>(
    cfg: &EngineCfg,
    stm: &'a memra_gguf::safetensors::StModel,
) -> Result<Up<'a>, String> {
    let mc = memra_gguf::config::ModelConfig::from_config_json(&cfg.dir.join("config.json"))
        .map_err(|e| e.to_string())?;
    let g = Geo::from_config(&mc)?;
    let full_trunk = mc
        .dsv41
        .as_ref()
        .ok_or("not a deepseek_v41 config")?
        .trunk_layers() as usize;
    let trunk = cfg.trunk_layers.map_or(full_trunk, |n| n.min(full_trunk));
    if trunk != full_trunk {
        eprintln!(
            "[dsv41-serve] {}: MEMRA_DSV41_TRUNK_LAYERS: {trunk} of the {full_trunk} trunk layers \
             run. The answers are NOT the checkpoint's; this is a rehearsal of the route's \
             memory and serving loop",
            cfg.name
        );
    }
    let ec = EngramConsts::derive(&g, token_map_from_file(&cfg.token_map)?)?;
    let mut rn = Runner::new(
        stm,
        g.clone(),
        Some(ec),
        &cfg.devices,
        cfg.max_seq + 1,
        trunk,
        0,
        PREFILL_CHUNK,
    )?;
    let slot = Layout::of(&g).bytes as u64;
    let per_rank = (g.n_experts / cfg.devices.len()) as u64;
    // a rank's routed experts, all layers
    let routed = trunk as u64 * per_rank * slot;
    let mut free = Vec::new();
    for c in &rn.grp.cx {
        free.push(v41::experts::device_memory(&c.ops)?.0);
    }
    let on_card = experts_on_card(routed, &cfg.devices, &free);
    let n = cfg.devices.len();
    let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
    // a rank's ceiling when the trunk is resident: its card's free memory after the load, shared
    // by the card's ranks
    let mut left = vec![0u64; n];
    let (resident, floor_cache, max_slots) = if on_card {
        // the whole trunk and the head on the ranks, loaded now: one prompt of a full prefill
        // slice must still fit beside them on every rank, or the load refuses before it runs out
        let first = PREFILL_CHUNK.min(cfg.max_seq.saturating_sub(1)).max(1);
        let reserve_rank = v41::decode::prefill_workspace_bytes(&g, n, first, PREFILL_CHUNK)
            + v41::decode::seq_state_bytes(&g, trunk, first + 1)
            + MARGIN_BYTES as usize;
        let most = rn
            .device_ranks()
            .iter()
            .map(|x| x.1.len())
            .max()
            .unwrap_or(1);
        let rep = rn.load_resident(trunk, reserve_rank * most)?;
        let mut resident = vec![0u64; n];
        for b in &rep.devices {
            for &r in &b.ranks {
                left[r] = (b.free_after / b.ranks.len()) as u64;
                resident[r] = (b.resident / b.ranks.len()) as u64;
            }
            eprintln!(
                "[dsv41-serve] {}: card {} ranks {:?}: weights {:.2} GiB resident ({:.2} GiB a \
                 rank) in {:.1}s, {:.2} GiB of {:.2} GiB left for requests ({:.2} GiB a rank)",
                cfg.name,
                b.device,
                b.ranks,
                gib(b.resident as u64),
                gib((b.resident / b.ranks.len()) as u64),
                rep.load_s,
                gib(b.free_after as u64),
                gib(b.total as u64),
                gib((b.free_after / b.ranks.len()) as u64),
            );
        }
        warm_up(&mut rn, cfg)?;
        (resident, 0, 0)
    } else {
        let threads = std::thread::available_parallelism().map_or(8, |n| n.get());
        let ctx = rn.grp.cx[0].ops.st.context().clone();
        let host = Arc::new(HostExperts::fill(stm, &g, trunk, &ctx, threads)?);
        eprintln!(
            "[dsv41-serve] {}: routed experts of {trunk} layers in pinned host memory: {} bytes, \
             filled and locked in {:.1}s on {threads} threads",
            cfg.name, host.bytes, host.fill_s
        );
        let resident = rn.experts_off_card(host)?;
        // one slice can route to every expert of the rank's range: the cache never goes below
        // that; every routed expert of every layer on one rank is the most it can ever use
        let floor = per_rank.div_ceil(v41::experts::SLAB_SLOTS as u64)
            * v41::experts::SLAB_SLOTS as u64
            * slot;
        (resident, floor, trunk * per_rank as usize)
    };
    let mut ceiling = Vec::new();
    for (c, &l) in rn.grp.cx.iter().zip(&left) {
        ceiling.push(if on_card {
            l
        } else {
            v41::experts::device_memory(&c.ops)?.0
        });
    }
    let book = Book {
        model: MemModel::of(&g, trunk, PREFILL_CHUNK),
        ceiling,
        floor_cache,
        resident: on_card.then(|| (g.clone(), n)),
    };
    let slots = if on_card {
        Vec::new()
    } else {
        rn.grow_caches(book.live_need(&[], None), max_slots)?
    };
    eprintln!(
        "[dsv41-serve] {}: {} rank(s) on devices {:?}; routed experts {}; resident weights \
         {resident:?} bytes per rank; ceiling {:?} bytes; expert cache {slots:?} slots of {slot} \
         bytes (floor {} slots, all {max_slots}); max_seq {}; {} session(s); prefill slices of \
         {PREFILL_CHUNK} rows",
        cfg.name,
        rn.grp.n(),
        cfg.devices,
        if on_card {
            "on the cards"
        } else {
            "in pinned host memory"
        },
        book.ceiling,
        if on_card { 0 } else { per_rank },
        cfg.max_seq,
        cfg.sessions,
    );
    if !on_card {
        warm_up(&mut rn, cfg)?;
    }
    // what the memory book admits with the route idle: the longest session, and how many
    // sessions of the benchmark's shapes at once
    eprintln!(
        "[dsv41-serve] {}: capacity: the longest session with the route idle is {} positions \
         (max_seq {}); at once: {} sessions of 1,536 positions (1,024-token prompts), {} of 4,608 \
         (4,096-token prompts), {} of 32,768, {} of 131,072; MEMRA_MAX_SESSIONS {}",
        cfg.name,
        book.largest_fit(cfg.max_seq).min(cfg.max_seq),
        cfg.max_seq,
        book.sessions_fit(1024, 1536, 4096),
        book.sessions_fit(4096, 4608, 4096),
        book.sessions_fit(32256, 32768.min(cfg.max_seq), 4096),
        book.sessions_fit(
            130560.min(cfg.max_seq.saturating_sub(1)),
            131072.min(cfg.max_seq),
            4096
        ),
        cfg.sessions,
    );
    Ok(Up {
        rn,
        book,
        max_slots,
    })
}

/// Whether every rank's routed experts (`routed` bytes a rank, all layers) fit the cards: a card
/// carrying k ranks needs k times that inside its free memory (`free[r]` is rank r's card, before
/// any weight loads). When they fit the trunk is resident; a rank's own weights and the sessions
/// must then fit the rest, and a load that does not says so. When they do not, the experts stay
/// in pinned host memory behind the device cache.
fn experts_on_card(routed: u64, devices: &[usize], free: &[u64]) -> bool {
    devices.iter().zip(free).all(|(d, &f)| {
        let ranks = devices.iter().filter(|x| *x == d).count() as u64;
        routed.saturating_mul(ranks) <= f
    })
}

/// Run the program once, a two-token prompt and one decode call, freed after. With the trunk
/// resident this is also what loads every layer's share and the head.
fn warm_up(rn: &mut Runner, cfg: &EngineCfg) -> Result<(), String> {
    let mut seqs = vec![rn.seq_with_capacity(8)];
    let probe = vec![cfg.eos, cfg.eos];
    let t = Instant::now();
    let outs = rn.forward(
        &mut seqs,
        &[Call {
            seq: 0,
            tokens: probe,
        }],
    )?;
    let row = rn.logits(&outs[0], outs[0].s - 1, 1)?;
    let t1 = memra_engine::dsv41_gpu::decode::argmax(&row) as u32;
    drop(outs);
    let outs = rn.forward(
        &mut seqs,
        &[Call {
            seq: 0,
            tokens: vec![t1],
        }],
    )?;
    rn.logits(&outs[0], 0, 1)?;
    drop(outs);
    drop(seqs);
    eprintln!(
        "[dsv41-serve] {}: warm-up prefill and decode call in {:.1}s",
        cfg.name,
        t.elapsed().as_secs_f64()
    );
    Ok(())
}

fn pool_peak_reset(rn: &Runner) {
    for c in &rn.grp.cx {
        v41::experts::reset_peak(&c.ops);
    }
}

fn pool_peak(rn: &Runner) -> u64 {
    rn.grp
        .cx
        .iter()
        .map(|c| v41::experts::pool_peak(&c.ops))
        .max()
        .unwrap_or(0)
}

fn pool_used(rn: &Runner) -> u64 {
    rn.grp
        .cx
        .iter()
        .map(|c| v41::experts::pool_used(&c.ops))
        .min()
        .unwrap_or(0)
}

fn serve_loop(
    cfg: &EngineCfg,
    up: &mut Up,
    rx: &Receiver<Box<Request>>,
    health: &Arc<RouteHealth>,
    load: &Arc<RouteLoad>,
) {
    let mut pending: VecDeque<Prepared> = VecDeque::new();
    let mut act: Vec<Active> = Vec::new();
    let mut seqs: Vec<Seq> = Vec::new();
    let mut closing = false;
    let mut last_step = Instant::now();
    loop {
        // intake: block only when there is nothing to do
        if act.is_empty() && pending.is_empty() {
            if closing {
                break;
            }
            health.set_idle();
            let Ok(req) = rx.recv() else {
                closing = true;
                continue;
            };
            intake(cfg, req, load, health, &mut pending);
        }
        loop {
            match rx.try_recv() {
                Ok(req) => intake(cfg, req, load, health, &mut pending),
                Err(std::sync::mpsc::TryRecvError::Empty) => break,
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    closing = true;
                    break;
                }
            }
        }
        // clients that left: their requests end at this step boundary
        let mut i = 0;
        while i < act.len() {
            if act[i].req.tx.is_closed() {
                let a = act.swap_remove(i);
                seqs.swap_remove(i);
                if let Some(run) = a.run {
                    run.cancel();
                }
                health.end_request();
            } else {
                i += 1;
            }
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
        // at most one new prompt per step, first come first served
        let mut new_call: Option<(usize, Vec<u32>)> = None;
        if act.len() < cfg.sessions
            && let Some(head) = pending.front()
        {
            let running: Vec<usize> = act.iter().map(|a| a.cap).collect();
            let (p, cap) = (head.prompt.len(), head.prompt.len() + head.budget);
            match admit_dsv41_session(&mut up.rn, &up.book, &running, p, cap) {
                Ok(true) => {
                    let mut pr = pending.pop_front().expect("the head exists");
                    let tok = cfg.tok.clone();
                    let mut eos = pr.req.params.eos.clone();
                    if !eos.contains(&cfg.eos) {
                        eos.push(cfg.eos);
                    }
                    let _ = pr.req.tx.send(Event::PromptUsage {
                        n_prompt: p,
                        n_cached: 0,
                    });
                    pr.run.admit();
                    let stream = Stream {
                        tok,
                        tx: pr.req.tx.clone(),
                        eos,
                        stops: pr.req.stop_strings.clone(),
                        stop_ids: pr.req.stop_token_ids.clone(),
                        budget: pr.budget,
                        ids: Vec::new(),
                        text: String::new(),
                        stop: None,
                        client_gone: false,
                    };
                    let window = if pr.pen.is_some() {
                        pr.prompt.clone()
                    } else {
                        Vec::new()
                    };
                    seqs.push(up.rn.seq_with_capacity(cap));
                    act.push(Active {
                        req: pr.req,
                        run: Some(pr.run),
                        prompt_len: p,
                        cap,
                        stream,
                        next: None,
                        sample: pr.sample,
                        pen: pr.pen,
                        window,
                        t0: Instant::now(),
                        steps: 0,
                    });
                    new_call = Some((act.len() - 1, pr.prompt));
                }
                Ok(false) => {}
                Err(outcome) => {
                    let mut pr = pending.pop_front().expect("the head exists");
                    let (answer, _) = admission_answer(&outcome, cap, p, || up.book.largest_fit(p));
                    eprintln!(
                        "[admit-mem] route=dsv41-thread model={:?} request_id={:?} capacity={cap} \
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
        // the step: the new prompt's prefill and one decode call per running request
        let new_at = new_call.as_ref().map(|(i, _)| *i);
        let mut calls = Vec::with_capacity(act.len());
        let mut who = Vec::with_capacity(act.len());
        if let Some((i, prompt)) = new_call.take() {
            calls.push(Call {
                seq: i,
                tokens: prompt,
            });
            who.push(i);
        }
        for (i, a) in act.iter().enumerate() {
            if let Some(t) = a.next {
                calls.push(Call {
                    seq: i,
                    tokens: vec![t],
                });
                who.push(i);
            }
        }
        let predicted = up.book.live_need(
            &act.iter()
                .enumerate()
                .filter(|&(i, _)| Some(i) != new_at)
                .map(|(_, a)| a.cap)
                .collect::<Vec<_>>(),
            new_at.map(|i| (act[i].prompt_len, act[i].cap)),
        );
        let used_before = pool_used(&up.rn);
        pool_peak_reset(&up.rn);
        let ts = Instant::now();
        let stepped = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let outs = up.rn.forward(&mut seqs, &calls)?;
            let mut rows = Vec::with_capacity(outs.len());
            for o in &outs {
                rows.push(up.rn.logits(o, o.s - 1, 1)?);
            }
            Ok::<_, String>(rows)
        }));
        let rows = match stepped {
            Ok(Ok(rows)) => rows,
            Ok(Err(e)) => {
                fail_all(
                    &mut act,
                    &mut seqs,
                    health,
                    &format!("deepseek-v4.1 step: {e}"),
                );
                continue;
            }
            Err(payload) => {
                health.note_request_fault();
                let why = payload
                    .downcast_ref::<String>()
                    .cloned()
                    .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
                    .unwrap_or_else(|| "non-string panic payload".into());
                fail_all(
                    &mut act,
                    &mut seqs,
                    health,
                    &format!("deepseek-v4.1 step panicked: {why}"),
                );
                continue;
            }
        };
        let step_s = ts.elapsed().as_secs_f64();
        let transient = pool_peak(&up.rn).saturating_sub(used_before);
        health.note_round();
        load.note_round(last_step.elapsed().as_millis() as u64);
        last_step = Instant::now();
        let (cs, slots) = up.rn.cache_stats();
        eprintln!(
            "[dsv41-step] calls={} prefill={} step_s={step_s:.3} transient_peak={transient} \
             predicted_live_need={predicted} cache_slots={slots:?} hits={} misses={} h2d_bytes={}",
            calls.len(),
            new_at.map_or(0, |i| act[i].prompt_len),
            cs.hits,
            cs.misses,
            cs.h2d_bytes
        );
        // sample, emit, finish
        let mut done = Vec::new();
        for (row, &i) in rows.into_iter().zip(&who) {
            let a = &mut act[i];
            let pos = a.prompt_len + a.stream.ids.len();
            let t = match choose(row, pos, a.sample.as_ref(), a.pen.as_ref(), &a.window) {
                Ok(t) => t,
                Err(e) => {
                    let _ = a.req.tx.send(Event::Error(EngineError::engine(format!(
                        "deepseek-v4.1 sampler: {e}"
                    ))));
                    a.stream.client_gone = true;
                    done.push(i);
                    continue;
                }
            };
            a.steps += 1;
            if a.stream.push(t) {
                if a.pen.is_some() {
                    a.window.push(t);
                }
                a.next = Some(t);
            } else {
                done.push(i);
            }
        }
        // finish from the back so swap_remove keeps the other indices valid
        done.sort_unstable();
        for &i in done.iter().rev() {
            let a = act.swap_remove(i);
            seqs.swap_remove(i);
            finish(a, health);
        }
        // the prefill's transient is gone: let the cache take the free memory back
        if pending.is_empty() && up.rn.offloaded() {
            let running: Vec<usize> = act.iter().map(|a| a.cap).collect();
            let reserve = up.book.live_need(&running, None);
            if let Err(e) = up.rn.grow_caches(reserve, up.max_slots) {
                eprintln!("[dsv41-serve] grow the expert cache: {e}");
            }
        }
    }
}

fn intake(
    cfg: &EngineCfg,
    mut req: Box<Request>,
    load: &Arc<RouteLoad>,
    health: &Arc<RouteHealth>,
    pending: &mut VecDeque<Prepared>,
) {
    // Occupancy rises before the ticket falls (`RouteLoad::begin`), so an arrival never reads a
    // free route between the two.
    let run = load.begin();
    if let Some(ticket) = req.route_ticket.as_ref() {
        load.record_wait(ticket.waited());
    }
    crate::worker::release_request_reservation(&mut req);
    if req.tx.is_closed() {
        run.cancel();
        return;
    }
    health.begin_request();
    match prepare(&cfg.tok, cfg.max_seq, &mut req) {
        Ok(Rendered {
            prompt,
            budget,
            sample,
            pen,
        }) => pending.push_back(Prepared {
            req,
            run,
            prompt,
            budget,
            sample,
            pen,
        }),
        Err(e) => {
            settle(run, &mut req, Err(e));
            health.end_request();
        }
    }
}

fn finish(mut a: Active, health: &Arc<RouteHealth>) {
    let elapsed = a.t0.elapsed().as_secs_f64();
    a.stream.finish(a.prompt_len, elapsed);
    let stats = ServeStats {
        tokens_out: a.stream.ids.len(),
        n_prompt: a.prompt_len,
        n_cached: 0,
        rounds: a.steps,
    };
    if let Some(run) = a.run.take() {
        if a.stream.client_gone {
            run.cancel();
        } else {
            run.finish(stats);
        }
    }
    health.end_request();
}

fn fail_all(act: &mut Vec<Active>, seqs: &mut Vec<Seq>, health: &Arc<RouteHealth>, why: &str) {
    eprintln!("[dsv41-serve] {why}; failing {} request(s)", act.len());
    seqs.clear();
    for mut a in act.drain(..) {
        fail_request(&mut a.req, EngineError::engine(why.to_string()));
        if let Some(run) = a.run.take() {
            run.cancel();
        }
        health.end_request();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flash() -> MemModel {
        let mut ratios = vec![0, 0];
        ratios.extend(std::iter::repeat_n(2, 18));
        ratios.extend(std::iter::repeat_n(1, 20));
        let g = Geo {
            dim: 5120,
            heads: 64,
            head_dim: 512,
            window: 128,
            index_heads: 32,
            index_dim: 128,
            index_topk: 512,
            topk: 6,
            moe_inter: 2304,
            hc: 4,
            compress_ratios: ratios,
            kv_sources: vec![2, 8, 14, 20],
            engram_heads: 8,
            engram_ngram: 4,
            engram_head_dim: 256,
            n_experts: 384,
            ..Geo::default()
        };
        MemModel::of(&g, 40, PREFILL_CHUNK)
    }

    #[test]
    fn residency_follows_whether_the_routed_experts_fit_the_cards() {
        // the 1017 checkpoint: 40 layers of 384 routed experts of 18,800,640 bytes
        let all = 40 * 384 * 18_800_640u64;
        assert_eq!(all, 288_777_830_400);
        let card = 95u64 << 30;
        // one card: 289 GB of experts against 95 GiB free
        assert!(!experts_on_card(all, &[0], &[card]));
        // two cards: 144 GB a rank
        assert!(!experts_on_card(all / 2, &[0, 1], &[card, card]));
        // four cards: 72 GB a rank
        assert!(experts_on_card(all / 4, &[0, 1, 2, 3], &[card; 4]));
        // four ranks sharing one card need four shares of it
        assert!(!experts_on_card(all / 4, &[0, 0, 0, 0], &[card; 4]));
        // one busy card among four keeps the experts in host memory
        assert!(!experts_on_card(
            all / 4,
            &[0, 1, 2, 3],
            &[card, card, 60 << 30, card]
        ));
    }

    #[test]
    fn the_state_book_matches_the_decode_lane_arithmetic() {
        let m = flash();
        // per layer a 128 KB ring; per group 1 KB of compressed KV and 256 B of index key a row,
        // n/2 rows in the three ratio-2 groups and n rows in the ratio-1 group
        let base = m.state_bytes(0);
        assert_eq!(base, 40 * 131_072 + 40 * 2 * 2 * 512 * 4 + 4 * 1280);
        let n = 1 << 20;
        assert_eq!(m.state_bytes(n) - base, (3 * (n / 2) + n) as u64 * 1280);
    }

    #[test]
    fn the_prefill_book_grows_linearly_then_with_the_candidate_pool() {
        let m = flash();
        let a = m.prefill_bytes(16_384);
        let b = m.prefill_bytes(32_768);
        let c = m.prefill_bytes(65_536);
        assert!(b > a && c > b);
        // the quadratic pool term dominates the growth past the slice size
        assert!(c - b > b - a);
        // a short prompt is far below a card
        assert!(m.prefill_bytes(16) < 1 << 30);
    }

    #[test]
    fn greedy_takes_the_lowest_id_among_equal_maxima_and_penalties_apply_first() {
        let row = vec![0.0, 3.0, 3.0, 1.0];
        assert_eq!(choose(row.clone(), 0, None, None, &[]).unwrap(), 1);
        let pen = Dsv4PenaltyCfg {
            last_n: usize::MAX,
            repeat: 2.0,
            freq: 0.0,
            present: 0.0,
        };
        assert_eq!(choose(row, 0, None, Some(&pen), &[1]).unwrap(), 2);
    }

    #[test]
    fn seeded_draws_repeat_and_move_with_the_position() {
        let row: Vec<f32> = (0..64).map(|i| (i % 7) as f32 * 0.3).collect();
        let cfg = Dsv4SampleCfg {
            temperature: 1.0,
            top_p: 1.0,
            top_k: 0,
            seed: 1234,
        };
        let a: Vec<u32> = (0..32)
            .map(|p| choose(row.clone(), p, Some(&cfg), None, &[]).unwrap())
            .collect();
        let b: Vec<u32> = (0..32)
            .map(|p| choose(row.clone(), p, Some(&cfg), None, &[]).unwrap())
            .collect();
        assert_eq!(a, b);
        assert!(a.iter().any(|&t| t != a[0]));
    }

    #[test]
    fn the_v41_dsml_close_stays_in_the_stream() {
        let stops = vec![DSML_V41_CLOSE.to_string()];
        let full = format!("x{DSML_V41_CLOSE}tail");
        let cut = stop_end("x", &full, &stops).unwrap();
        assert_eq!(&full[..cut], format!("x{DSML_V41_CLOSE}"));
    }

    #[test]
    fn a_client_stop_across_two_tokens_reaches_the_http_cut_whole() {
        // the served case: stop `:"A` over the tokens `:"` (already sent) and `Au`
        let stops = vec![":\"A".to_string(), DSML_V41_CLOSE.to_string()];
        let sent = "{\"gold\":\"";
        let full = "{\"gold\":\"Au";
        // the shared scan cannot take sent text back: cutting there leaves `:"` in the answer
        // and hides the stop from the HTTP layer
        assert_eq!(
            scan_stop_cut_with(sent, full, &stops, &[DSML_V41_CLOSE]),
            Some(sent.len())
        );
        // the stream ends after the whole token; the HTTP layer's cut is then exact
        let end = stop_end(sent, full, &stops).unwrap();
        assert_eq!(end, full.len());
        let at = full[..end].find(stops[0].as_str()).unwrap();
        assert_eq!(&full[..at], "{\"gold\"");
        // no stop in the text: the stream goes on
        assert_eq!(stop_end(sent, "{\"gold\":\"Fe", &stops), None);
    }
}
