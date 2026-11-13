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
//! Batching. Up to `MEMRA_MAX_SESSIONS` requests run at once. Each step runs at most one new
//! prompt's prefill call and every running request's next token. The one-token steps of the
//! running requests go through the trunk together, up to `MEMRA_DSV41_BATCH_ROWS` rows a call
//! (`decode::Runner::forward_batch`: each row is its request's own one-token call at its own
//! position, against its own state, through shared launches; more requests than that are
//! consecutive calls), and their logits come from one pass over the head (when every step of
//! the call is a plain greedy pick the card returns each row's argmax instead and no logits are
//! downloaded; a lone step is a call of one row through the same path). A row's bytes are those
//! of its one-token call run alone (`dsv41-spec batch-rows`), so a request's numbers are the same
//! served alone or beside others (the serve receipts gate this byte for byte). A prefill and a
//! drafted block's verification run as calls of their own.
//!
//! Drafting (`MEMRA_DSV41_DRAFT=1`; off until the route's served gates and its A/B have run
//! with it, `docs/FLAGS.md`). The checkpoint's DSpark head (`dsv41_gpu::draft`) drafts a block
//! after a step: the request's decode call then carries its next token and drafted ones, each
//! row run as the one-token decode program of its position (`decode::Runner::forward`,
//! byte-identical to feeding the tokens one call at a time: `dsv41-spec verify-rows`). The route
//! chooses each row's token exactly as it chooses a plain step's (the same penalties window and
//! position-keyed draw), keeps the drafts that equal those choices from the front, commits those
//! rows and drops the rest. So a request's tokens are the undrafted route's, greedy or sampled,
//! and a draft only decides how many tokens a step yields. The draft is always the head's argmax
//! (a one-hot proposal).
//!
//! What a step verifies is priced, never fixed: the route keeps what its decode calls cost by
//! their rows (`draft::RowCost`, filled from its own steps, so it is this residency's table) and
//! each step verifies the depth with the most expected tokens per millisecond under the
//! confidence head's survival (`draft::price_depth`); a request whose drafted steps yield less
//! than its one-token steps would stops drafting for a stretch (`draft::DraftGate`).
//! `MEMRA_DSV41_DRAFT_DEPTH` caps the depth.
//!
//! Sampling. Greedy is the argmax, the lowest id among equal maxima (the decode lane's rule, so a
//! served greedy answer is `dsv41-run --gen`'s). Temperature, top-p and top-k draw through the
//! DSv4 route's position-keyed host sampler (`dsv4_sample_row`, seeded and reproducible),
//! penalties through `dsv4_penalize_row` over the prompt and the generated tokens; `min_p` is
//! refused, as on the DSv4 route. Stops: the end-of-sentence id (unless the completions request
//! sets `ignore_eos`, the benchmark field: the id is then an ordinary token), the request's stop
//! ids, stop strings over the detokenized stream (the V4.1 DSML close stays in the stream so the
//! parser closes the call), and the token budget.
//!
//! Deliberate limits, stated: `response_format`/grammar refused by name; image parts are refused
//! by the HTTP layer (no V4.1 vision program); no prefix cache or CUDA graphs.

use crate::dsv4_admit::Admission;
use crate::dsv4_serve::{admission_answer, fail_request, settle, snap};
use crate::health::RouteHealth;
use crate::route_telemetry::{RouteLoad, RouteRun, ServeStats};
use crate::worker::{EngineError, Event, EventSender, ModelCaps, Request, SpecUsage};
use memra_engine::decode::StopReason;
use memra_engine::dsv4_gpu::{Dsv4PenaltyCfg, Dsv4SampleCfg, dsv4_penalize_row, dsv4_sample_row};
use memra_engine::dsv41_gpu::decode::{Call, Out, Runner, Seq};
use memra_engine::dsv41_gpu::draft::{DraftGate, DraftSeq, DraftW, RowCost, price_depth};
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
    /// rows of a decode call: 1, or with the draft head its block plus the token before it
    pub decode_rows: usize,
    /// whether the running requests' one-token steps share a call (their key rows are then live
    /// together)
    pub batched: bool,
    /// draft stages (0: no draft head); each keeps a window ring per sequence
    pub draft_stages: usize,
}

impl MemModel {
    #[cfg(test)]
    fn of(g: &Geo, layers: usize, chunk: usize) -> Self {
        Self::with_draft(g, layers, chunk, 0, 0)
    }

    /// [`Self::of`] with a draft head of `stages` stages drafting `block` tokens.
    fn with_draft(g: &Geo, layers: usize, chunk: usize, stages: usize, block: usize) -> Self {
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
            decode_rows: if stages > 0 { block + 1 } else { 1 },
            batched: false,
            draft_stages: stages,
        }
    }

    /// A sequence's carried state at `cap` positions: per layer the window ring (bf16) and the
    /// ratio-2 compressor's open pair (fp32 kv and gate), per kv-source group the compressed KV
    /// (bf16 rows of `head_dim`) and the index keys (bf16 rows of `index_dim`).
    pub fn state_bytes(&self, cap: usize) -> u64 {
        let ring = (self.layers + self.draft_stages) * self.window * self.head_dim * 2;
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

    /// The transient of one decode call at up to `cap` positions: the heads' key rows the call
    /// makes for itself (the window with a block's own rows after it; the compressed rows are
    /// read where they live, `memra_v41_sparse_attn_keys`, so they are no transient), each row's
    /// index scores and candidate row, and the rows' slice terms. A draft block's own forward is
    /// smaller than its verification.
    pub fn decode_bytes(&self, cap: usize) -> u64 {
        let rows = self.decode_rows.max(1);
        // a one-token call reads the ring in place; a block's rows sit after a copy of it
        let own = if rows > 1 { rows } else { 0 };
        let keys = (self.window + own) * self.head_dim * 2;
        let scores = rows * (cap * 2 + cap / 8);
        (keys + scores) as u64 + self.prefill_bytes(rows)
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
    /// the exact bytes of `ids` (`Tokenizer::decode_bytes_special`, a token's bytes after the
    /// last one's): the stream's text is their lossy UTF-8 reading, `Tokenizer::decode` of every
    /// id so far without decoding them again at every token
    bytes: Vec<u8>,
    /// bytes of `bytes` already turned into `text` ([`text_delta`]): a token that ends inside
    /// a character leaves its tail here until the next token completes it
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
        if stopped {
            if self.tx.send(Event::Token { id, text: delta }).is_err() {
                self.client_gone = true;
            }
            self.stop = Some(StopReason::Callback);
            return false;
        }
        if self.tx.send(Event::Token { id, text: delta }).is_err() {
            self.client_gone = true;
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
/// already text: every completed character (the worker's `utf8_delta` rule: a tokenizer's byte
/// pieces may split a character across tokens, so an incomplete tail waits for the token that
/// completes it, and a byte that can start or continue nothing becomes one U+FFFD). `last`: the
/// answer ends with this token, so a tail still incomplete is read as the detokenized ids read
/// it (lossy), and the stream's whole text equals `Tokenizer::decode` of its ids.
fn text_delta(bytes: &[u8], emitted: &mut usize, last: bool) -> String {
    let mut delta = crate::worker::utf8_delta(bytes, emitted);
    if last && *emitted < bytes.len() {
        delta.push_str(&String::from_utf8_lossy(&bytes[*emitted..]));
        *emitted = bytes.len();
    }
    delta
}

/// One token's step of a stream's text: `text` is what was sent, `bytes` the answer's bytes with
/// the new token's, `emitted` how many of them are text. Returns what to send and whether a stop
/// string ended the stream there. The text grows only by completed characters ([`text_delta`]);
/// the law before this (decode every byte, send what lies past the text already sent) turned a
/// character split across two tokens into replacement characters and lost it. An answer that
/// ends on an end-of-sequence or stop id inside a character drops the open bytes (no event
/// carries text without a token), where the detokenized ids read one U+FFFD.
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
/// are already sent (None: no stop). A client's stop ends it after the whole token, the worker's
/// law: the HTTP layer finds the complete stop text and cuts there (`truncate_at_stop`, the
/// stream scrubber), so a stop that straddles two tokens leaves no prefix of itself in the
/// answer. The parser-closing DSML stop alone ends the stream at its close.
///
/// A stop that completes now ends in the new text, so it starts less than its own length before
/// the sent boundary: the look-back is the stop's length less one byte, whatever that is (the
/// DSv4 route's rule since memra#1193), so a stop of any length spread over tokens is found. An
/// empty stop matches nothing.
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
        .filter(|s| s.as_str() != DSML_V41_CLOSE)
        .any(|s| starts_at(s).is_some());
    if client {
        return Some(full.len());
    }
    if !stops.iter().any(|s| s.as_str() == DSML_V41_CLOSE) {
        return None;
    }
    let at = starts_at(DSML_V41_CLOSE)?;
    Some((at + DSML_V41_CLOSE.len()).max(text.len()).min(full.len()))
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
    /// the draft head's rings for this sequence (None: the route serves without the head)
    ds: Option<DraftSeq>,
    /// the drafted tokens the step's decode call carries after `next`
    drafts: Vec<u32>,
    /// whether this step ran the draft head for the request (its cost counts even at depth 0)
    drafted: bool,
    /// whether the request's drafts pay (it skips the head for a stretch when they do not)
    gate: DraftGate,
    usage: SpecUsage,
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

fn prepare(
    tok: &Arc<Tokenizer>,
    max_seq: usize,
    req: &mut Request,
) -> Result<Rendered, EngineError> {
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

/// The ids that end a request's generation: the request's own and the model's end-of-sentence
/// id. With `ignore_eos` the model's id is left out, so it is an ordinary token (decoded into the
/// stream and fed back) and only the budget or a stop ends the run.
fn end_ids(request: &[u32], model_eos: u32, ignore_eos: bool) -> Vec<u32> {
    let mut ids = request.to_vec();
    if !ignore_eos && !ids.contains(&model_eos) {
        ids.push(model_eos);
    }
    ids
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
            Some((g, _)) => {
                (v41::decode::seq_state_bytes(g, self.model.layers, cap)
                    + self.model.draft_stages * g.window * g.head_dim * 2) as u64
            }
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
        let each = caps
            .iter()
            .chain(new.as_ref().map(|(_, c)| c))
            .map(|&c| self.model.decode_bytes(c));
        // calls run one after another hold one call's transient at a time; a batched call holds
        // every row's
        let decode = if self.model.batched {
            each.sum()
        } else {
            each.max().unwrap_or(0)
        };
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
    /// the DSpark draft head and the drafts verified per step at most (None: serving without it)
    draft: Option<(DraftW, usize)>,
    /// what this residency's decode calls cost by their rows, and a draft, from the route's own
    /// steps
    price: Price,
    /// the most one-token steps one decode call carries (1: every call alone)
    batch_rows: usize,
}

/// What a batched call's rows hand the route: their logits (a sampler or penalties choose from
/// them), or, when every row is a plain greedy pick, the device's argmax of each row
/// (`Runner::argmax_rows`: the host argmax's token, lowest id among equal maxima).
enum Picked {
    Logits(Vec<f32>),
    Ids(Vec<u32>),
}

/// What a chunk of batched one-token steps hands the route. An all-greedy chunk asks the device
/// for each row's argmax (`argmax`); any other chunk downloads its rows' logits (`logits`). The
/// device pick answers for the chunk as a whole, so when it refuses (one row holds a non-finite
/// logit) the chunk's logits come down instead and every request chooses from its own row: the
/// bad row's request fails alone in [`choose`], as on the logits path, and the others go on. A
/// fault of the step itself (a rank's route flag) fails the logits too, and with them the step.
fn pick_rows<R>(
    rn: &mut R,
    greedy: bool,
    argmax: impl FnOnce(&mut R) -> Result<Vec<u32>, String>,
    logits: impl FnOnce(&mut R) -> Result<Vec<f32>, String>,
) -> Result<Picked, String> {
    if greedy {
        match argmax(rn) {
            Ok(ids) => return Ok(Picked::Ids(ids)),
            Err(e) => eprintln!(
                "[dsv41-serve] the device pick of a batched call refused ({e}): its rows choose \
                 from their logits"
            ),
        }
    }
    Ok(Picked::Logits(logits(rn)?))
}

/// The most rows a batched decode call carries: the small-M GEMM bodies' limit
/// (`dsv41_gpu::SMALL_M`).
const BATCH_ROWS_MAX: usize = v41::SMALL_M;

/// One running request's rows of a step: the request, its logits rows (empty when the device
/// chose), the call that holds them, its first row there, and the device's choice of a batched
/// greedy row.
type StepRows<'a> = (usize, Vec<f32>, &'a Out, usize, Option<u32>);

/// `MEMRA_DSV41_BATCH_ROWS`: the one-token steps one decode call carries, 1 (every call alone)
/// to the small-M limit (unset: the limit).
fn resolve_batch_rows(raw: Option<&str>) -> Result<usize, String> {
    match raw.map(str::trim) {
        None | Some("") => Ok(BATCH_ROWS_MAX),
        Some(x) => match x.parse::<usize>() {
            Ok(n) if (1..=BATCH_ROWS_MAX).contains(&n) => Ok(n),
            _ => Err(format!(
                "MEMRA_DSV41_BATCH_ROWS {x:?} is not a row count from 1 to {BATCH_ROWS_MAX}"
            )),
        },
    }
}

/// The route's own costs, in milliseconds: a decode call by its rows and one draft.
struct Price {
    rows: RowCost,
    draft_ms: f64,
}

impl Price {
    fn new(max_rows: usize) -> Self {
        Self {
            rows: RowCost::new(max_rows),
            draft_ms: 0.0,
        }
    }

    fn note_draft(&mut self, ms: f64) {
        self.draft_ms = if self.draft_ms > 0.0 {
            self.draft_ms + 0.25 * (ms - self.draft_ms)
        } else {
            ms
        };
    }
}

/// `MEMRA_DSV41_DRAFT`: `1` drafts with the checkpoint's DSpark head; unset or `0` serves
/// without it.
fn resolve_draft(raw: Option<&str>) -> Result<bool, String> {
    match raw.map(str::trim) {
        None | Some("") | Some("0") => Ok(false),
        Some("1") => Ok(true),
        Some(x) => Err(format!("MEMRA_DSV41_DRAFT {x:?} is not 0 or 1")),
    }
}

/// `MEMRA_DSV41_DRAFT_DEPTH`: the drafts verified per step, 1 to the head's block (unset: the
/// block).
fn resolve_draft_depth(raw: Option<&str>, block: usize) -> Result<usize, String> {
    match raw.map(str::trim) {
        None | Some("") => Ok(block),
        Some(x) => match x.parse::<usize>() {
            Ok(n) if (1..=block).contains(&n) => Ok(n),
            _ => Err(format!(
                "MEMRA_DSV41_DRAFT_DEPTH {x:?} is not a draft count from 1 to the head's block of {block}"
            )),
        },
    }
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
    let want_draft = resolve_draft(std::env::var("MEMRA_DSV41_DRAFT").ok().as_deref())?;
    if want_draft && trunk != full_trunk {
        eprintln!(
            "[dsv41-serve] {}: a shortened trunk has no draft head (its target layers are not all run)",
            cfg.name
        );
    }
    let want_draft = want_draft && trunk == full_trunk;
    let mut draft: Option<(DraftW, usize)> = None;
    let mut warm: Option<Warm> = None;
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
        if want_draft {
            draft = Some(load_draft(&mut rn, &mc, &cfg.name)?);
            // the head took its share of every card
            let mem = rn.device_mem()?;
            for (_, ranks) in rn.device_ranks() {
                for &r in &ranks {
                    left[r] = (mem[ranks[0]].0 / ranks.len()) as u64;
                }
            }
        }
        warm = warm_up(&mut rn, cfg, draft.as_ref())?;
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
        if want_draft {
            // before the ceiling is read and the cache grows: the head's weights stay on the card
            draft = Some(load_draft(&mut rn, &mc, &cfg.name)?);
        }
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
    let (stages, block) = draft
        .as_ref()
        .map_or((0, 0), |(dw, _)| (dw.cfg.stages, dw.cfg.block));
    let batch_rows = resolve_batch_rows(std::env::var("MEMRA_DSV41_BATCH_ROWS").ok().as_deref())?;
    let mut model = MemModel::with_draft(&g, trunk, PREFILL_CHUNK, stages, block);
    model.batched = batch_rows > 1;
    let book = Book {
        model,
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
        warm = warm_up(&mut rn, cfg, draft.as_ref())?;
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
    // the cost table starts from the warm-up's two shapes and a draft
    let mut price = Price::new(block + 1);
    if let Some(w) = warm {
        price.rows.observe(1, w.one_ms);
        price.rows.observe(w.rows, w.rows_ms);
        price.note_draft(w.draft_ms);
    }
    Ok(Up {
        rn,
        book,
        max_slots,
        draft,
        price,
        batch_rows,
    })
}

/// The draft head beside the trunk: its stages on the ranks' cards, the trunk's taps on, and the
/// drafts verified per step.
fn load_draft(
    rn: &mut Runner<'_>,
    mc: &memra_gguf::config::ModelConfig,
    name: &str,
) -> Result<(DraftW, usize), String> {
    let t = Instant::now();
    let dw = DraftW::load(rn, mc)?;
    let depth = resolve_draft_depth(
        std::env::var("MEMRA_DSV41_DRAFT_DEPTH").ok().as_deref(),
        dw.cfg.block,
    )?;
    eprintln!(
        "[dsv41-serve] {name}: DSpark draft head: {} stages of {} experts (top {}), block {}, target layers {:?}, {depth} drafts verified per step; loaded in {:.1}s",
        dw.cfg.stages,
        dw.cfg.n_experts,
        dw.cfg.topk,
        dw.cfg.block,
        dw.cfg.targets,
        t.elapsed().as_secs_f64()
    );
    Ok((dw, depth))
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

/// What the warm-up measured with the draft head: a one-token call, a call of `rows` rows and a
/// draft, in milliseconds (each the second run of its shape).
struct Warm {
    one_ms: f64,
    rows: usize,
    rows_ms: f64,
    draft_ms: f64,
}

/// Run the program once, a two-token prompt and decode calls, freed after. With the trunk
/// resident this is also what loads every layer's share and the head. With the draft head it
/// also runs a draft and a full block's verification twice each and returns their times: the
/// first prices of the route's cost table, which its own steps then correct.
fn warm_up(
    rn: &mut Runner,
    cfg: &EngineCfg,
    draft: Option<&(DraftW, usize)>,
) -> Result<Option<Warm>, String> {
    let argmax = |row: &[f32]| memra_engine::dsv41_gpu::decode::argmax(row) as u32;
    let mut seqs = vec![rn.seq_with_capacity(32)];
    let mut ds = draft.map(|(dw, _)| dw.seq(rn));
    let probe = vec![cfg.eos, cfg.eos];
    let t = Instant::now();
    let outs = rn.forward(
        &mut seqs,
        &[Call {
            seq: 0,
            tokens: probe,
        }],
    )?;
    let mut tok = argmax(&rn.logits(&outs[0], outs[0].s - 1, 1)?);
    if let (Some((dw, _)), Some(ds)) = (draft, ds.as_mut()) {
        let o = &outs[0];
        rn.draft_commit(dw, ds, o, o.tap_row0, o.tap_rows, false)?;
    }
    drop(outs);
    // two one-token calls: the second is timed
    let mut one_ms = 0.0;
    for _ in 0..2 {
        let t1 = Instant::now();
        let outs = rn.forward(
            &mut seqs,
            &[Call {
                seq: 0,
                tokens: vec![tok],
            }],
        )?;
        tok = argmax(&rn.logits(&outs[0], 0, 1)?);
        one_ms = t1.elapsed().as_secs_f64() * 1e3;
        if let (Some((dw, _)), Some(ds)) = (draft, ds.as_mut()) {
            rn.draft_commit(dw, ds, &outs[0], 0, 1, false)?;
        }
    }
    let mut warm = None;
    if let (Some((dw, _)), Some(ds)) = (draft, ds.as_mut()) {
        // a draft and its block's verification, twice each; the block is dropped, then run
        // again and its first row kept
        let mut draft_ms = 0.0;
        let mut ids = Vec::new();
        for _ in 0..2 {
            let td = Instant::now();
            ids = rn.draft(dw, ds, tok, seqs[0].len(), false)?.ids;
            draft_ms = td.elapsed().as_secs_f64() * 1e3;
        }
        let mut tokens = vec![tok];
        tokens.extend_from_slice(&ids);
        let rows = tokens.len();
        let mut rows_ms = 0.0;
        for pass in 0..2 {
            let tv = Instant::now();
            let outs = rn.forward(
                &mut seqs,
                &[Call {
                    seq: 0,
                    tokens: tokens.clone(),
                }],
            )?;
            rn.logits(&outs[0], 0, rows)?;
            rows_ms = tv.elapsed().as_secs_f64() * 1e3;
            if pass == 0 {
                rn.discard(&mut seqs[0])?;
            } else {
                rn.commit(&mut seqs[0], 1)?;
                rn.draft_commit(dw, ds, &outs[0], 0, 1, false)?;
            }
        }
        warm = Some(Warm {
            one_ms,
            rows,
            rows_ms,
            draft_ms,
        });
    }
    drop(seqs);
    eprintln!(
        "[dsv41-serve] {}: warm-up in {:.1}s; one-token call {one_ms:.1} ms{}",
        cfg.name,
        t.elapsed().as_secs_f64(),
        warm.as_ref().map_or(String::new(), |w| format!(
            ", {}-row call {:.1} ms, draft {:.1} ms",
            w.rows, w.rows_ms, w.draft_ms
        ))
    );
    Ok(warm)
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
                    let eos = end_ids(&pr.req.params.eos, cfg.eos, pr.req.ignore_eos);
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
                        ds: up.draft.as_ref().map(|(dw, _)| dw.seq(&up.rn)),
                        drafts: Vec::new(),
                        drafted: false,
                        gate: DraftGate::new(),
                        usage: SpecUsage::default(),
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
        // the step: the new prompt's prefill and one decode call per running request (its next
        // token, and with the draft head the block drafted after it)
        let new_at = new_call.as_ref().map(|(i, _)| *i);
        let mut calls = Vec::with_capacity(act.len());
        let mut who = Vec::with_capacity(act.len());
        // (request, token) of the one-token steps that share a call
        let mut batch: Vec<(usize, u32)> = Vec::with_capacity(act.len());
        if let Some((i, prompt)) = new_call.take() {
            calls.push(Call {
                seq: i,
                tokens: prompt,
            });
            who.push(i);
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
            let mut draft_ms = Vec::new();
            for (i, a) in act.iter_mut().enumerate() {
                a.drafts.clear();
                a.drafted = false;
                let Some(t) = a.next else { continue };
                if let (Some((dw, depth)), Some(ds)) = (up.draft.as_ref(), a.ds.as_mut()) {
                    // a drafted row is worth running only while the budget has room past `t`'s
                    // successor, and the head only while the request's drafts pay
                    let room = a.stream.budget.saturating_sub(a.stream.ids.len() + 1);
                    if (*depth).min(room) > 0 && a.gate.drafts() {
                        let td = Instant::now();
                        let d = up.rn.draft(dw, ds, t, seqs[i].len(), false)?;
                        draft_ms.push(td.elapsed().as_secs_f64() * 1e3);
                        let k = price_depth(
                            &d.conf,
                            &up.price.rows,
                            up.price.draft_ms,
                            (*depth).min(room),
                        );
                        a.drafts.extend_from_slice(&d.ids[..k]);
                        a.drafted = true;
                    }
                }
                if a.drafts.is_empty() && up.batch_rows > 1 {
                    // a one-token step: it shares a call with the other requests' steps
                    batch.push((i, t));
                    continue;
                }
                let mut tokens = Vec::with_capacity(1 + a.drafts.len());
                tokens.push(t);
                tokens.extend_from_slice(&a.drafts);
                calls.push(Call { seq: i, tokens });
                who.push(i);
            }
            // a lone step is a batch of one: the same call, whose bytes are the one-token
            // call's (`dsv41-spec batch-rows` runs calls of width 1 in every case)
            // whether every batched step is a plain greedy pick: its token is then the device's
            // argmax of its row and no logits leave the card
            let greedy = batch
                .iter()
                .all(|&(i, _)| act[i].sample.is_none() && act[i].pen.is_none());
            let tf = Instant::now();
            let outs = if calls.is_empty() {
                Vec::new()
            } else {
                up.rn.forward(&mut seqs, &calls)?
            };
            let mut rows = Vec::with_capacity(outs.len());
            for o in &outs {
                // a prompt's last row; every row of a decode call
                let (r0, n) = if o.sp == 0 { (o.s - 1, 1) } else { (0, o.s) };
                rows.push(up.rn.logits(o, r0, n)?);
            }
            // the one-token steps together, their logits from one pass over the head
            let mut bouts = Vec::new();
            for chunk in batch.chunks(up.batch_rows) {
                let o = up.rn.forward_batch(&mut seqs, chunk)?;
                let picked = pick_rows(
                    &mut up.rn,
                    greedy,
                    |rn| rn.argmax_rows(&o, 0, chunk.len()),
                    |rn| rn.logits(&o, 0, chunk.len()),
                )?;
                bouts.push((o, picked));
            }
            Ok::<_, String>((
                outs,
                rows,
                bouts,
                draft_ms,
                tf.elapsed().as_secs_f64() * 1e3,
            ))
        }));
        let (outs, rows, bouts, draft_ms, forward_ms) = match stepped {
            Ok(Ok(x)) => x,
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
            "[dsv41-step] calls={} prefill={} decode_rows={} step_s={step_s:.3} \
             transient_peak={transient} predicted_live_need={predicted} cache_slots={slots:?} \
             hits={} misses={} h2d_bytes={}",
            calls.len() + bouts.len(),
            new_at.map_or(0, |i| act[i].prompt_len),
            outs.iter().filter(|o| o.sp > 0).map(|o| o.s).sum::<usize>() + batch.len(),
            cs.hits,
            cs.misses,
            cs.h2d_bytes
        );
        // the route's own costs: a draft, and a decode call by its rows when the forward's calls
        // all had the same rows (each call runs alone through each block, so they share it evenly)
        for &ms in &draft_ms {
            up.price.note_draft(ms);
        }
        if let Some(first) = outs.first()
            && bouts.is_empty()
            && first.sp > 0
            && outs.iter().all(|o| o.sp > 0 && o.s == first.s)
        {
            up.price
                .rows
                .observe(first.s, forward_ms / outs.len() as f64);
        }
        // choose, emit, commit, finish
        let vocab = up.rn.g.vocab;
        let mut done = Vec::new();
        // every request's rows with the call that holds them: its own call's from row 0, or its
        // row of a batched call
        let mut items: Vec<StepRows<'_>> = rows
            .into_iter()
            .zip(&outs)
            .zip(&who)
            .map(|((row, o), &i)| (i, row, o, 0, None))
            .collect();
        let mut b0 = 0usize;
        for (o, picked) in &bouts {
            for r in 0..o.s {
                let i = batch[b0 + r].0;
                items.push(match picked {
                    Picked::Logits(lg) => (i, lg[r * vocab..(r + 1) * vocab].to_vec(), o, r, None),
                    Picked::Ids(ids) => (i, Vec::new(), o, r, Some(ids[r])),
                });
            }
            b0 += o.s;
        }
        for (i, row, o, row0, picked) in items {
            let a = &mut act[i];
            let fail = |a: &mut Active, what: &str, e: String| {
                let _ = a.req.tx.send(Event::Error(EngineError::engine(format!(
                    "deepseek-v4.1 {what}: {e}"
                ))));
                a.stream.client_gone = true;
            };
            // a prompt's taps seed the draft rings
            if o.sp == 0
                && let (Some((dw, _)), Some(ds)) = (up.draft.as_ref(), a.ds.as_mut())
                && let Err(e) = up.rn.draft_commit(dw, ds, o, o.tap_row0, o.tap_rows, false)
            {
                fail(a, "draft seed", e);
                done.push(i);
                continue;
            }
            // Row r is the target's distribution after the call's token r. Each row's token is
            // chosen as a plain step's (the penalties window and the position-keyed draw see the
            // tokens emitted so far); a draft stands while it equals that choice, and the first
            // row whose choice differs from the draft after it ends the block.
            let n_rows = if picked.is_some() {
                1
            } else {
                row.len() / vocab
            };
            let drafted = n_rows - 1;
            let mut kept = 0usize;
            let mut live = true;
            a.steps += 1;
            for r in 0..n_rows {
                let pos = a.prompt_len + a.stream.ids.len();
                let chosen = match picked {
                    // the device's argmax of the row: a plain greedy step of a batched call
                    Some(t) => Ok(t),
                    None => choose(
                        row[r * vocab..(r + 1) * vocab].to_vec(),
                        pos,
                        a.sample.as_ref(),
                        a.pen.as_ref(),
                        &a.window,
                    ),
                };
                let t = match chosen {
                    Ok(t) => t,
                    Err(e) => {
                        fail(a, "sampler", e);
                        live = false;
                        break;
                    }
                };
                kept = r + 1;
                if !a.stream.push(t) {
                    live = false;
                    break;
                }
                if a.pen.is_some() {
                    a.window.push(t);
                }
                if r < drafted && t == a.drafts[r] {
                    continue;
                }
                a.next = Some(t);
                break;
            }
            if drafted > 0 {
                a.usage.rounds += 1;
                a.usage.drafted += drafted as u64;
                a.usage.accepted += kept.saturating_sub(1) as u64;
            }
            // what the drafted step yielded against one-token steps, at the route's prices
            if a.drafted
                && let (Some(call), Some(plain)) = (up.price.rows.ms(n_rows), up.price.rows.ms(1))
            {
                a.gate.observe(kept, call + up.price.draft_ms, plain);
            }
            if !live {
                done.push(i);
                continue;
            }
            // the kept rows become the sequence's state; the rejected ones never reach it
            if o.sp > 0 {
                let mut r = Ok(());
                // a call of its own with several rows is a verification: its kept rows commit (a
                // batched call's rows are other requests' steps, written as they ran)
                if o.batch.is_empty() && o.s > 1 {
                    r = up.rn.commit(&mut seqs[i], kept);
                }
                if r.is_ok()
                    && let (Some((dw, _)), Some(ds)) = (up.draft.as_ref(), a.ds.as_mut())
                {
                    r = up.rn.draft_commit(dw, ds, o, row0, kept, false).map(|_| ());
                }
                if let Err(e) = r {
                    fail(a, "commit", e);
                    done.push(i);
                }
            }
        }
        drop(outs);
        drop(bouts);
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
    let spec = a.ds.is_some().then_some(a.usage);
    if spec.is_some() {
        eprintln!(
            "[dsv41-draft] request_id={:?} rounds={} drafted={} accepted={} tokens={}",
            a.req.request_id,
            a.usage.rounds,
            a.usage.drafted,
            a.usage.accepted,
            a.stream.ids.len()
        );
    }
    a.stream.finish(a.prompt_len, elapsed, spec);
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

    /// Feed a stream's text step the tokens' byte pieces; returns the deltas and the final text.
    fn run_stream(pieces: &[&[u8]], stops: &[String]) -> (Vec<String>, String, bool) {
        let (mut bytes, mut emitted, mut text) = (Vec::new(), 0usize, String::new());
        let mut deltas = Vec::new();
        for (i, p) in pieces.iter().enumerate() {
            bytes.extend_from_slice(p);
            let (d, stopped) = text_step(
                &mut text,
                &bytes,
                &mut emitted,
                i + 1 == pieces.len(),
                stops,
            );
            deltas.push(d);
            if stopped {
                return (deltas, text, true);
            }
        }
        (deltas, text, false)
    }

    #[test]
    fn a_character_split_across_tokens_is_sent_whole_once_it_completes() {
        // the pattern a truncated trunk produced in the served gate: a token that ends with a
        // byte that starts nothing and the first byte of a three-byte character, then the token
        // that completes the character. Decoding everything and sending what lies past the text
        // already sent gave two replacement characters and lost the character.
        let pieces: [&[u8]; 4] = [b"ab", &[0xff, 0xe7], &[0x97, 0x9e], b"cd"];
        let (deltas, text, stopped) = run_stream(&pieces, &[]);
        assert!(!stopped);
        assert_eq!(deltas, ["ab", "\u{fffd}", "\u{75de}", "cd"]);
        assert_eq!(text, String::from_utf8_lossy(&pieces.concat()));
        assert_eq!(deltas.concat(), text);
        // an emoji over two tokens, nothing invalid before it
        let emoji: [&[u8]; 3] = [b"ok ", &[0xf0, 0x9f], &[0x99, 0x82]];
        let (deltas, text, _) = run_stream(&emoji, &[]);
        assert_eq!(deltas, ["ok ", "", "\u{1f642}"]);
        assert_eq!(text, "ok \u{1f642}");
        // an answer its budget cuts inside a character reads as its detokenized ids do
        let cut: [&[u8]; 2] = [b"x", &[0xe7, 0x97]];
        let (deltas, text, _) = run_stream(&cut, &[]);
        assert_eq!(text, String::from_utf8_lossy(&cut.concat()));
        assert_eq!(deltas.concat(), text);
    }

    #[test]
    fn a_stop_string_right_after_a_split_character_ends_the_stream_with_the_character_whole() {
        // The character completes in the token that also brings a client's stop string. The
        // stream ends after that whole token (the HTTP layer cuts at the stop text), and what
        // it sent up to there is the answer's text, the character included.
        let pieces: [&[u8]; 4] = [b"ab", &[0xff, 0xe7], b"\x97\x9estop and more", b"never"];
        let (deltas, text, stopped) = run_stream(&pieces, &["stop".to_string()]);
        assert!(stopped);
        assert_eq!(deltas.len(), 3);
        assert_eq!(text, String::from_utf8_lossy(&pieces[..3].concat()));
        assert_eq!(text, "ab\u{fffd}\u{75de}stop and more");
        assert_eq!(deltas.concat(), text);
        // a stop that straddles the token after the split character
        let pieces: [&[u8]; 4] = [b"ab", &[0xe7, 0x97], b"\x9est", b"op tail"];
        let (deltas, text, stopped) = run_stream(&pieces, &["stop".to_string()]);
        assert!(stopped);
        assert_eq!(text, "ab\u{75de}stop tail");
        assert_eq!(deltas.concat(), text);
    }

    #[test]
    fn a_greedy_chunk_whose_device_argmax_refuses_chooses_from_its_logits_row_by_row() {
        // two rows of a 2-token vocabulary; the first holds a NaN, so the device pick refuses
        let rows = vec![f32::NAN, 1.0, 0.0, 2.0];
        let mut asked: Vec<&str> = Vec::new();
        let picked = pick_rows(
            &mut asked,
            true,
            |a| {
                a.push("argmax");
                Err("non-finite logits".into())
            },
            |a| {
                a.push("logits");
                Ok(rows.clone())
            },
        );
        // the step goes on with the chunk's logits: the NaN row's request fails alone in
        // `choose`, the other row's request takes its token
        let Ok(Picked::Logits(lg)) = picked else {
            panic!("a refused device pick must fall back to the logits, not fail the step");
        };
        assert_eq!(asked, ["argmax", "logits"]);
        assert!(choose(lg[..2].to_vec(), 7, None, None, &[]).is_err());
        assert_eq!(choose(lg[2..].to_vec(), 7, None, None, &[]), Ok(1));
        // a finite greedy chunk never downloads its logits; a sampled one never asks the device
        let mut asked: Vec<&str> = Vec::new();
        let picked = pick_rows(
            &mut asked,
            true,
            |a| {
                a.push("argmax");
                Ok(vec![1, 1])
            },
            |a| {
                a.push("logits");
                Ok(rows.clone())
            },
        );
        assert!(matches!(picked, Ok(Picked::Ids(ref ids)) if ids == &[1, 1]));
        assert_eq!(asked, ["argmax"]);
        let mut asked: Vec<&str> = Vec::new();
        let picked = pick_rows(
            &mut asked,
            false,
            |a| {
                a.push("argmax");
                Ok(vec![1, 1])
            },
            |a| {
                a.push("logits");
                Ok(rows.clone())
            },
        );
        assert!(matches!(picked, Ok(Picked::Logits(_))));
        assert_eq!(asked, ["logits"]);
        // the logits' own failure (a route fault) still fails the step
        let picked = pick_rows(
            &mut (),
            true,
            |_| Err("non-finite logits".into()),
            |_| Err("rank 1 route: a layer placed other than rows x topk pairs".into()),
        );
        assert!(picked.is_err());
    }

    #[test]
    fn the_batch_width_defaults_to_the_small_m_limit_and_refuses_what_is_outside_it() {
        assert_eq!(resolve_batch_rows(None), Ok(BATCH_ROWS_MAX));
        assert_eq!(resolve_batch_rows(Some("1")), Ok(1));
        assert_eq!(resolve_batch_rows(Some("8")), Ok(8));
        assert!(resolve_batch_rows(Some("0")).is_err());
        assert!(resolve_batch_rows(Some("9")).is_err());
        // a batched call holds every running request's key rows at once
        let mut m = flash();
        let alone = Book {
            model: m,
            ceiling: vec![1 << 40],
            floor_cache: 0,
            resident: None,
        }
        .live_need(&[4096, 4096, 4096], None);
        m.batched = true;
        let together = Book {
            model: m,
            ceiling: vec![1 << 40],
            floor_cache: 0,
            resident: None,
        }
        .live_need(&[4096, 4096, 4096], None);
        assert_eq!(together - alone, 2 * m.decode_bytes(4096));
    }

    #[test]
    fn the_draft_door_is_off_unless_set_and_refuses_other_values() {
        assert_eq!(resolve_draft(None), Ok(false));
        assert_eq!(resolve_draft(Some("")), Ok(false));
        assert_eq!(resolve_draft(Some("0")), Ok(false));
        assert_eq!(resolve_draft(Some(" 1 ")), Ok(true));
        assert!(resolve_draft(Some("yes")).is_err());
        assert_eq!(resolve_draft_depth(None, 5), Ok(5));
        assert_eq!(resolve_draft_depth(Some("3"), 5), Ok(3));
        assert!(resolve_draft_depth(Some("0"), 5).is_err());
        assert!(resolve_draft_depth(Some("6"), 5).is_err());
        assert!(resolve_draft_depth(Some("x"), 5).is_err());
    }

    #[test]
    fn a_draft_head_adds_its_rings_and_its_rows_to_the_memory_book() {
        let plain = flash();
        let mut with = flash();
        with.decode_rows = 6;
        with.draft_stages = 3;
        // three more rings of 128 rows of 512 bf16
        assert_eq!(
            with.state_bytes(4096) - plain.state_bytes(4096),
            3 * 128 * 512 * 2
        );
        // a block's call books more than a one-token call, and the one-token call is unchanged
        assert!(with.decode_bytes(4096) > plain.decode_bytes(4096));
        assert_eq!(plain.decode_rows, 1);
    }

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
    fn ignore_eos_leaves_the_model_end_id_out() {
        assert_eq!(end_ids(&[], 1, false), vec![1]);
        assert_eq!(end_ids(&[1], 1, false), vec![1]);
        assert_eq!(end_ids(&[7], 1, false), vec![7, 1]);
        assert!(end_ids(&[], 1, true).is_empty());
        // the request's own ids still end it
        assert_eq!(end_ids(&[7], 1, true), vec![7]);
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
    fn a_stop_longer_than_64_bytes_spread_over_tokens_ends_the_stream() {
        // an 80-byte client stop whose first 70 bytes went out before this token: it starts 70
        // bytes before the sent boundary, past a fixed 64-byte look-back
        let stop = "s".repeat(79) + "!";
        let stops = vec![stop.clone(), DSML_V41_CLOSE.to_string()];
        let sent = format!("answer {}", &stop[..70]);
        let full = format!("{sent}{} tail", &stop[70..]);
        assert_eq!(stop_end(&sent, &full, &stops), Some(full.len()));
        let at = full.find(stop.as_str()).unwrap();
        assert_eq!(&full[..at], "answer ");
        // the DSML close spread the same way ends at its close
        let close = DSML_V41_CLOSE;
        let sent = format!("x{}", &close[..close.len() - 2]);
        let full = format!("x{close}tail");
        let end = stop_end(&sent, &full, &[close.to_string()]).unwrap();
        assert_eq!(&full[..end], format!("x{close}"));
    }

    #[test]
    fn an_empty_stop_string_matches_nothing() {
        let stops = vec![String::new(), DSML_V41_CLOSE.to_string()];
        assert_eq!(stop_end("ab", "abc", &stops), None);
        assert_eq!(stop_end("", "a", &[String::new()]), None);
    }

    #[test]
    fn a_client_stop_across_two_tokens_reaches_the_http_cut_whole() {
        // the served case: stop `:"A` over the tokens `:"` (already sent) and `Au`
        let stops = vec![":\"A".to_string(), DSML_V41_CLOSE.to_string()];
        let sent = "{\"gold\":\"";
        let full = "{\"gold\":\"Au";
        // the stream ends after the whole token; the HTTP layer's cut is then exact
        let end = stop_end(sent, full, &stops).unwrap();
        assert_eq!(end, full.len());
        let at = full[..end].find(stops[0].as_str()).unwrap();
        assert_eq!(&full[..at], "{\"gold\"");
        // no stop in the text: the stream goes on
        assert_eq!(stop_end(sent, "{\"gold\":\"Fe", &stops), None);
    }
}
