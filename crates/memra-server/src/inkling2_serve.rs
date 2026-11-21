//! inkling2 serve route: Inkling 2 Mini through the worker's `Cmd::Generate(Request) -> Event`
//! contract, on the native GPU route of the typed plan (`memra_engine::inkling2_gpu`), never
//! through `HybridModel`. The HTTP surface (OpenAI routes, the inkling2 chat dialect's inputs,
//! the token-id response parser, streaming SSE) is the shared one; this module is the worker side.
//!
//! The program. The route runs the plan the pack compiles from the checkpoint's own config: the
//! declared numeric class (the checkpoint's activation quantizers on the routed experts), the
//! family's default. A route runs one plan, so every request on it runs one class and no request
//! crosses between two. Each request owns a sequence (`inkling2_gpu::Seq`: the full-attention
//! caches, the sliding rings and the conv histories of every slot), admitted against what the
//! driver reports free (`cuMemGetInfo`) with the route's own measured cost of a sequence.
//!
//! Steps. Each step runs one prefill chunk (at most the route's chunk rows) of the oldest request
//! still prefilling, then one decode token of every request past its prefill in one batched call
//! (`Route::decode_batch`: each row is its sequence's one-token call at its own position against
//! its own state, through shared launches). A row's bits are those of its one-token call run
//! alone (the `ink2-gpu --batch-check` gate), so a request's numbers are the same served alone
//! or beside others, and a peer's prefill chunks between its steps change none of them.
//!
//! Sampling. Greedy is the argmax, the lowest id among equal maxima (the route's argmax kernel).
//! Temperature, top-p and top-k draw through the DSv4 route's position-keyed host sampler
//! (`dsv4_sample_row`, seeded and reproducible), penalties through `dsv4_penalize_row`;
//! `min_p` is refused. Stops: the checkpoint's end-of-sampling id (200006), the request's stop
//! ids and stop strings, and the token budget.
//!
//! Deliberate limits, stated: `response_format`/grammar refused by name; image parts refused (no
//! vision program yet); no prefix cache, no drafting (the DSpark drafter is its own lane), no
//! CUDA graphs.

use crate::dsv4_admit::Admission;
use crate::dsv4_serve::{admission_answer, fail_request, settle};
use crate::health::RouteHealth;
use crate::route_telemetry::{RouteLoad, RouteRun, ServeStats};
use crate::worker::{EngineError, Event, EventSender, ModelCaps, Request};
use memra_engine::decode::StopReason;
use memra_engine::dsv4_gpu::{Dsv4PenaltyCfg, Dsv4SampleCfg, dsv4_penalize_row, dsv4_sample_row};
use memra_engine::inkling2_gpu::{Ckpt, Dev, Head, Route, Seq};
use memra_tokenizer::Tokenizer;
use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::mpsc::{Receiver, Sender};
use std::time::Instant;

/// Rows of one prefill call and of one batched decode call: the chunk every gate of the route
/// ran (`ink2-gpu`'s default).
const CHUNK: usize = 512;

/// Device bytes the memory door keeps free beside the sequences: the driver's own allocations of
/// a step (the route preallocates every buffer it writes, so this is allocator granularity, not a
/// workspace). Every admission logs driver free against the book.
const MARGIN_BYTES: usize = 256 << 20;

/// Boot-time probe: the pack the config contract selects is the inkling2 pack (never a name).
pub fn is_inkling2_dir(dir: &Path) -> bool {
    load_config(dir).is_ok_and(|c| {
        memra_gguf::model_packs::for_config(&c).is_some_and(|p| p.family == "inkling2")
    })
}

fn load_config(dir: &Path) -> Result<memra_gguf::config::ModelConfig, String> {
    let read =
        |name: &str| std::fs::read_to_string(dir.join(name)).map_err(|e| format!("{name}: {e}"));
    let mut config = memra_gguf::config::ModelConfig::from_hf(
        &memra_gguf::config::HfConfig::try_parse(&read("config.json")?)
            .map_err(|e| format!("config.json: {e}"))?,
    );
    if config.needs_quant_groups() {
        config.attach_quant_groups(&read("hf_quant_config.json")?)?;
    }
    Ok(config)
}

/// `MEMRA_REWRITE_BUNDLE` on this route: the bundle must name this checkpoint (family, config)
/// and this binary's plan, and its rewrite receipts, when it has any, must load against the plan.
/// The route then runs the plan's own native program, no rewrite. Returns the plan's hash.
fn install_inkling2_rewrite_bundle(
    bundle: &Path,
    dir: &Path,
    plan: &memra_gguf::model_plan::ModelPlan,
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
    if field("family").as_deref() != Some("inkling2") {
        return Err(format!(
            "rewrite bundle {}: artifact.lock family {:?} is not inkling2",
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
    let plan_hash = hex(format!("{plan:#?}\n").as_bytes());
    if field("plan_sha256").as_deref() != Some(plan_hash.as_str()) {
        return Err(format!(
            "rewrite bundle {}: artifact.lock plan_sha256 {:?} is not this binary's plan {plan_hash}",
            bundle.display(),
            field("plan_sha256")
        ));
    }
    if bundle.join("rewrite-receipts.tsv").exists() {
        memra_gguf::execution_manifest::RewriteQualifications::load(bundle, plan)
            .map_err(|e| format!("rewrite bundle {}: {e}", bundle.display()))?;
    }
    Ok(plan_hash)
}

/// What the route loaded, for the worker's caps and contract.
pub struct Inkling2Model {
    pub tok: Arc<Tokenizer>,
    pub max_seq: usize,
    /// requests served at once
    pub sessions: usize,
    markers: crate::toolcall::Inkling2Markers,
    start: Sender<Start>,
}

struct Start {
    rx: Receiver<Box<Request>>,
    health: Arc<RouteHealth>,
    load: Arc<RouteLoad>,
}

struct Ready {
    max_seq: usize,
}

struct EngineCfg {
    name: String,
    dir: PathBuf,
    tok: Arc<Tokenizer>,
    max_seq: usize,
    sessions: usize,
    eos: u32,
}

/// Load the route: the tokenizer's dialect, the context, then the model on its own serving thread.
pub fn load(name: &str, dir: &Path, tok: Arc<Tokenizer>) -> Result<Inkling2Model, String> {
    if !tok
        .chat_template()
        .is_some_and(memra_tokenizer::chat::template_is_inkling2)
    {
        return Err(format!(
            "inkling2 model {name:?}: the tokenizer's chat template is not the inkling2 dialect; \
             the route renders only that dialect"
        ));
    }
    let markers = crate::toolcall::Inkling2Markers::resolve(|p| tok.id_of(p))
        .map_err(|e| format!("inkling2 model {name:?}: {e}"))?;
    let config = load_config(dir).map_err(|e| format!("inkling2 model {name:?}: {e}"))?;
    if config.context_length == 0 {
        return Err(format!(
            "inkling2 model {name:?}: config.json declares no max_position_embeddings"
        ));
    }
    let declared = config.context_length as usize;
    let max_seq = crate::worker::resolve_env_ctx(declared).map_err(|e| {
        format!("inkling2 model {name:?}: {e} (config.json declares {declared} positions)")
    })?;
    if let Some(bundle) = std::env::var_os("MEMRA_REWRITE_BUNDLE") {
        let pack = memra_gguf::model_packs::for_config(&config)
            .ok_or_else(|| format!("inkling2 model {name:?}: no pack matches this config"))?;
        let plan = pack
            .compile_plan(&config)
            .map_err(|e| format!("inkling2 model {name:?}: plan: {e:?}"))?;
        let hash = install_inkling2_rewrite_bundle(Path::new(&bundle), dir, &plan)?;
        eprintln!(
            "[inkling2-serve] {name}: rewrite bundle {} installed for plan {hash}; the route runs \
             the plan's own native program, no rewrite",
            Path::new(&bundle).display()
        );
    }
    let sessions = crate::route_contract::hybrid_interactive_cap().min(CHUNK);
    if sessions == 0 {
        return Err("MEMRA_MAX_SESSIONS=0 leaves the inkling2 route no session".into());
    }
    let cfg = EngineCfg {
        name: name.to_string(),
        dir: dir.to_path_buf(),
        tok: tok.clone(),
        max_seq,
        sessions,
        eos: tok.eos_id(),
    };
    let (ready_tx, ready_rx) = std::sync::mpsc::channel::<Result<Ready, String>>();
    let (start_tx, start_rx) = std::sync::mpsc::channel::<Start>();
    let handle = std::thread::Builder::new()
        .name(format!("inkling2-serve-{name}"))
        .spawn(move || engine_thread(cfg, ready_tx, start_rx))
        .map_err(|e| format!("spawn the inkling2 serving thread: {e}"))?;
    THREADS
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .push(handle);
    let ready = ready_rx
        .recv()
        .map_err(|_| format!("inkling2 model {name:?}: the serving thread died loading"))??;
    Ok(Inkling2Model {
        tok,
        max_seq: ready.max_seq,
        sessions,
        markers,
        start: start_tx,
    })
}

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
    crate::route_contract::RouteContract::inkling2_thread(model, sessions)
}

/// ModelCaps for /v1/models and the HTTP gates: the inkling2 dialect (its markers arm the
/// token-id response parser), keyed on the tokenizer's template, which `load` required.
pub fn caps(m: &Inkling2Model) -> ModelCaps {
    ModelCaps {
        max_choices: 1,
        tools_branch: true,
        chat_ok: true,
        context_length: m.max_seq,
        tokenizer: m.tok.pre().to_string(),
        instruct_type: Some(memra_tokenizer::chat::INKLING2_DIALECT.to_string()),
        n_vocab: m.tok.vocab_size(),
        inkling2: Some(m.markers),
        ..ModelCaps::default()
    }
}

/// Start serving: the returned sender is the model's admission queue.
pub fn spawn(
    m: Inkling2Model,
    health: Arc<RouteHealth>,
    load: Arc<RouteLoad>,
) -> Sender<Box<Request>> {
    let (tx, rx) = std::sync::mpsc::channel::<Box<Request>>();
    let _ = m.start.send(Start { rx, health, load });
    tx
}

struct ExitLatch(Arc<RouteHealth>, Option<String>);

impl Drop for ExitLatch {
    fn drop(&mut self) {
        let why = match self.1.take() {
            Some(why) => why,
            None if std::thread::panicking() => "inkling2 serving thread panicked".into(),
            None => "inkling2 serving thread exited".into(),
        };
        self.0.mark_dead(why);
    }
}

/// The detokenized stream of one request and its stop rules.
struct Stream {
    tok: Arc<Tokenizer>,
    tx: EventSender,
    eos: Vec<u32>,
    stops: Vec<String>,
    stop_ids: Vec<u32>,
    budget: usize,
    ids: Vec<u32>,
    bytes: Vec<u8>,
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
        let old = self.text.len();
        let mut delta = crate::worker::utf8_delta(&self.bytes, &mut self.emitted);
        if last && self.emitted < self.bytes.len() {
            delta.push_str(&String::from_utf8_lossy(&self.bytes[self.emitted..]));
            self.emitted = self.bytes.len();
        }
        self.text.push_str(&delta);
        if let Some(end) = stop_end(&self.text[..old], &self.text, &self.stops) {
            let end = crate::dsv4_serve::snap(&self.text, end).max(old);
            self.text.truncate(end);
            let delta = self.text[old..].to_string();
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

/// Where the stream ends when a client's stop string completes in `full`, whose first
/// `text.len()` bytes are already sent: after the whole token (the HTTP layer cuts at the stop).
fn stop_end(text: &str, full: &str, stops: &[String]) -> Option<usize> {
    stops
        .iter()
        .filter(|s| !s.is_empty())
        .any(|stop| {
            let from = crate::dsv4_serve::snap(full, text.len().saturating_sub(stop.len() - 1));
            full[from..].contains(stop.as_str())
        })
        .then_some(full.len())
}

/// A request the route is serving.
struct Active {
    req: Box<Request>,
    run: Option<RouteRun>,
    prompt: Vec<u32>,
    /// prompt rows already prefilled
    prefilled: usize,
    stream: Stream,
    /// the token the next decode call feeds; None until the prompt's prefill ran
    next: Option<u32>,
    sample: Option<Dsv4SampleCfg>,
    pen: Option<Dsv4PenaltyCfg>,
    window: Vec<u32>,
    t0: Instant,
    steps: usize,
    fault: bool,
}

struct Prepared {
    req: Box<Request>,
    run: RouteRun,
    prompt: Vec<u32>,
    budget: usize,
    sample: Option<Dsv4SampleCfg>,
    pen: Option<Dsv4PenaltyCfg>,
}

struct Rendered {
    prompt: Vec<u32>,
    budget: usize,
    sample: Option<Dsv4SampleCfg>,
    pen: Option<Dsv4PenaltyCfg>,
}

/// Render and validate a request for the route (`Err` answers it).
fn render(
    tok: &Arc<Tokenizer>,
    max_seq: usize,
    req: &mut Request,
) -> Result<Rendered, EngineError> {
    if req.grammar.is_some() {
        return Err(EngineError::invalid_param(
            "response_format is not available on the inkling2 route yet",
            "response_format",
        ));
    }
    if !req.images.is_empty() {
        return Err(EngineError::invalid_param(
            "image parts are not available on the inkling2 route (no vision program yet)",
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
            "min_p is not implemented on the inkling2 sampler; use top_p/top_k",
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

/// The next token: penalties over the window, then the device's argmax (greedy) or the
/// position-keyed draw over the row.
fn choose(
    row: Option<Vec<f32>>,
    argmax: u32,
    pos: usize,
    sample: Option<&Dsv4SampleCfg>,
    pen: Option<&Dsv4PenaltyCfg>,
    window: &[u32],
) -> Result<u32, String> {
    match (sample, pen, row) {
        (None, None, _) => Ok(argmax),
        (_, _, None) => Err("the step brought no logits row for a sampled request".into()),
        (s, p, Some(mut row)) => {
            if let Some(p) = p {
                dsv4_penalize_row(&mut row, window, p);
            }
            match s {
                None => {
                    if row.iter().any(|x| x.is_nan()) {
                        return Err("non-finite logits".into());
                    }
                    // the lowest id among equal maxima, the route's argmax rule
                    let mut best = 0usize;
                    for (i, v) in row.iter().enumerate() {
                        if *v > row[best] {
                            best = i;
                        }
                    }
                    Ok(best as u32)
                }
                Some(cfg) => dsv4_sample_row(&row, pos, cfg),
            }
        }
    }
}

/// The ids that end a request's generation: its own and the model's end-of-sampling id.
fn end_ids(request: &[u32], model_eos: u32) -> Vec<u32> {
    let mut ids = request.to_vec();
    if !ids.contains(&model_eos) {
        ids.push(model_eos);
    }
    ids
}

/// The memory book: what one sequence of `cap` positions costs the driver, measured at warm-up
/// (the logical bytes plus the allocator's overhead per sequence).
struct Book {
    overhead: usize,
}

impl Book {
    fn need(&self, route: &Route, cap: usize) -> usize {
        route.seq_bytes(cap) + self.overhead + MARGIN_BYTES
    }

    /// The largest session the card holds now (driver free) for the route idle.
    fn largest_fit(&self, route: &Route, free: usize) -> usize {
        let (mut lo, mut hi) = (0usize, 1usize << 24);
        while lo < hi {
            let mid = (lo + hi).div_ceil(2);
            if self.need(route, mid) <= free {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }
        lo
    }
}

struct Up {
    route: Route,
    book: Book,
}

fn bring_up(cfg: &EngineCfg) -> Result<Up, String> {
    let config = load_config(&cfg.dir)?;
    let pack = memra_gguf::model_packs::for_config(&config).ok_or("no pack matches this config")?;
    let plan = pack
        .compile_plan(&config)
        .map_err(|e| format!("plan: {e:?}"))?;
    let class = memra_engine::inkling2_gpu::numeric_class(&plan)?;
    let ck = Ckpt::open(&cfg.dir, pack, &config, &plan)?;
    let dev = Dev::new(0)?;
    // the route's own sequence is never served: the smallest it can hold, then dropped
    let mut route = Route::load(dev, &ck, &plan, 1, CHUNK)?;
    route.drop_default_seq()?;
    eprintln!(
        "[inkling2-serve] {}: class {class}; {}",
        cfg.name,
        route.account.describe()
    );
    // warm up: every kernel of a prefill chunk and of a batched decode call runs once (lazily
    // loaded modules take their device memory now), and the allocator's overhead of a sequence is
    // measured against the driver
    let (free0, _) = route.dev.driver_free()?;
    let cap = 4096usize.min(cfg.max_seq);
    let mut a = route.new_seq(cap)?;
    let mut b = route.new_seq(cap)?;
    let (free1, _) = route.dev.driver_free()?;
    let logical = a.bytes + b.bytes;
    let overhead = (free0.saturating_sub(free1)).saturating_sub(logical) / 2;
    let warm: Vec<u32> = cfg.tok.encode("Hello", true);
    let warm = if warm.is_empty() { vec![cfg.eos] } else { warm };
    route.forward_seq(&mut a, &warm, Head::Last, None)?;
    route.forward_seq(&mut b, &warm, Head::Last, None)?;
    route.decode_batch(&mut [&mut a, &mut b], &[warm[0], warm[0]], Head::All)?;
    drop((a, b));
    route.dev.sync()?;
    eprintln!(
        "[inkling2-serve] {}: a {cap}-position sequence is {} bytes, {overhead} more at the driver",
        cfg.name,
        route.seq_bytes(cap)
    );
    Ok(Up {
        route,
        book: Book { overhead },
    })
}

fn engine_thread(cfg: EngineCfg, ready: Sender<Result<Ready, String>>, start: Receiver<Start>) {
    let t0 = Instant::now();
    let mut up = match bring_up(&cfg) {
        Ok(u) => u,
        Err(e) => {
            let _ = ready.send(Err(format!("inkling2 model {:?}: {e}", cfg.name)));
            return;
        }
    };
    eprintln!(
        "[inkling2-serve] {}: up in {:.1}s",
        cfg.name,
        t0.elapsed().as_secs_f64()
    );
    let _ = ready.send(Ok(Ready {
        max_seq: cfg.max_seq,
    }));
    let Ok(Start { rx, health, load }) = start.recv() else {
        return;
    };
    let mut latch = ExitLatch(health.clone(), None);
    let sink = health.clone();
    let _progress = memra_engine::progress::ProgressSinkScope::install(Box::new(move |rows| {
        sink.note_rows(rows)
    }));
    let (ptx, prx) = std::sync::mpsc::channel::<Prepared>();
    let (tok, max_seq) = (cfg.tok.clone(), cfg.max_seq);
    let (rload, rhealth) = (load.clone(), health.clone());
    let renderer = std::thread::Builder::new()
        .name("inkling2-render".into())
        .spawn(move || render_loop(tok, max_seq, rx, rload, rhealth, ptx));
    if let Err(e) = renderer {
        latch.1 = Some(format!("inkling2: start the renderer: {e}"));
        return;
    }
    latch.1 = serve_loop(&cfg, &mut up, &prx, &health, &load);
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
    let mut seqs: Vec<Seq> = Vec::new();
    let mut closing = false;
    let mut last_step = Instant::now();
    loop {
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
        // admission: the queue head, first come first served, while sessions and memory allow
        while act.len() < cfg.sessions
            && let Some(head) = pending.front()
        {
            let cap = head.prompt.len() + head.budget;
            let need = up.book.need(&up.route, cap);
            let free = match up.route.dev.driver_free() {
                Ok((f, _)) => f,
                Err(e) => {
                    let why = format!("inkling2: driver free: {e}");
                    fail_all(&mut act, &mut seqs, health, &why);
                    drain_into(rx, &mut pending);
                    fail_pending(&mut pending, health, &why);
                    return Some(why);
                }
            };
            if need > free {
                if act.is_empty() {
                    // nothing running gives memory back: the card cannot hold this session
                    let mut pr = pending.pop_front().expect("the head exists");
                    let outcome = Admission::NeverFits {
                        dev: 0,
                        need: need as u64,
                        ceiling: free as u64,
                    };
                    let (answer, _) = admission_answer(&outcome, cap, pr.prompt.len(), || {
                        up.book.largest_fit(&up.route, free)
                    });
                    eprintln!(
                        "[admit-mem] route=inkling2-thread model={:?} request_id={:?} capacity={cap} \
                         need={need} driver_free={free} verdict={}",
                        pr.req.model,
                        pr.req.request_id,
                        outcome.verdict()
                    );
                    if let Some(served) = answer {
                        settle(pr.run, &mut pr.req, Ok(served));
                    }
                    health.end_request();
                    continue;
                }
                // the running requests' ends give memory back: the head waits
                break;
            }
            let state = match up.route.new_seq(cap) {
                Ok(s) => s,
                Err(e) => {
                    if !act.is_empty() {
                        break;
                    }
                    let mut pr = pending.pop_front().expect("the head exists");
                    fail_request(
                        &mut pr.req,
                        EngineError::engine(format!(
                            "inkling2: the state of a {cap}-token session: {e}"
                        )),
                    );
                    health.note_request_fault();
                    pr.run.fail();
                    health.end_request();
                    continue;
                }
            };
            let mut pr = pending.pop_front().expect("the head exists");
            let _ = pr.req.tx.send(Event::PromptUsage {
                n_prompt: pr.prompt.len(),
                n_cached: 0,
            });
            pr.run.admit();
            let stream = Stream {
                tok: cfg.tok.clone(),
                tx: pr.req.tx.clone(),
                eos: end_ids(&pr.req.params.eos, cfg.eos),
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
            seqs.push(state);
            act.push(Active {
                req: pr.req,
                run: Some(pr.run),
                prompt: pr.prompt,
                prefilled: 0,
                stream,
                next: None,
                sample: pr.sample,
                pen: pr.pen,
                window,
                t0: Instant::now(),
                steps: 0,
                fault: false,
            });
        }
        if act.is_empty() {
            continue;
        }
        // the step: one prefill chunk of the oldest prefilling request, then one batched decode
        // call of every request past its prefill
        let ts = Instant::now();
        let route = &mut up.route;
        let stepped = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut rows: Vec<(usize, u32, Option<Vec<f32>>)> = Vec::new();
            let mut prefill_rows = 0usize;
            if let Some(i) = act.iter().position(|a| a.next.is_none()) {
                let a = &act[i];
                let end = (a.prefilled + CHUNK).min(a.prompt.len());
                let last = end == a.prompt.len();
                let head = if last { Head::Last } else { Head::None };
                let out =
                    route.forward_seq(&mut seqs[i], &a.prompt[a.prefilled..end], head, None)?;
                prefill_rows = end - a.prefilled;
                act[i].prefilled = end;
                if last {
                    let argmax = *out.argmax.last().ok_or("a prefill without its argmax")?;
                    rows.push((i, argmax, Some(out.logits)));
                }
            }
            let ready: Vec<usize> = (0..act.len())
                .filter(|&i| act[i].next.is_some() && act[i].prefilled == act[i].prompt.len())
                .collect();
            let decode_rows = ready.len();
            if !ready.is_empty() {
                let toks: Vec<u32> = ready.iter().map(|&i| act[i].next.expect("ready")).collect();
                let want_logits = ready
                    .iter()
                    .any(|&i| act[i].sample.is_some() || act[i].pen.is_some());
                // a greedy step needs only every row's argmax; a sampled one its logits too
                let head = if want_logits { Head::All } else { Head::Argmax };
                // every ready sequence, in `ready` order, borrowed for the one call
                let mut refs: Vec<&mut Seq> = Vec::with_capacity(ready.len());
                let mut rest: &mut [Seq] = &mut seqs[..];
                let mut base = 0usize;
                for &i in &ready {
                    let (_, tail) = rest.split_at_mut(i - base);
                    let (one, after) = tail.split_at_mut(1);
                    refs.push(&mut one[0]);
                    rest = after;
                    base = i + 1;
                }
                let out = route.decode_batch(&mut refs, &toks, head)?;
                let vocab = route.vocab();
                for (r, &i) in ready.iter().enumerate() {
                    let row = want_logits.then(|| out.logits[r * vocab..(r + 1) * vocab].to_vec());
                    rows.push((i, out.argmax[r], row));
                }
            }
            Ok::<_, String>((rows, prefill_rows, decode_rows))
        }));
        let (rows, prefill_rows, decode_rows) = match stepped {
            Ok(Ok(x)) => x,
            Ok(Err(e)) => {
                // the route's calls commit nothing when they fail: the step's requests fail, the
                // route serves on
                let why = format!("inkling2 step: {e}");
                fail_all(&mut act, &mut seqs, health, &why);
                continue;
            }
            Err(payload) => {
                let why = format!("inkling2 step panicked: {}", panic_text(payload.as_ref()));
                fail_all(&mut act, &mut seqs, health, &why);
                drain_into(rx, &mut pending);
                fail_pending(&mut pending, health, &why);
                return Some(why);
            }
        };
        health.note_round();
        load.note_round(last_step.elapsed().as_millis() as u64);
        last_step = Instant::now();
        eprintln!(
            "[inkling2-step] prefill_rows={prefill_rows} decode_rows={decode_rows} step_s={:.3}",
            ts.elapsed().as_secs_f64()
        );
        // choose, emit, finish
        let mut done = Vec::new();
        for (i, argmax, row) in rows {
            let a = &mut act[i];
            a.steps += 1;
            let pos = a.prompt.len() + a.stream.ids.len();
            let t = match choose(
                row,
                argmax,
                pos,
                a.sample.as_ref(),
                a.pen.as_ref(),
                &a.window,
            ) {
                Ok(t) => t,
                Err(e) => {
                    let _ = a.req.tx.send(Event::Error(EngineError::engine(format!(
                        "inkling2 sampler: {e}"
                    ))));
                    a.stream.client_gone = true;
                    a.fault = true;
                    done.push(i);
                    continue;
                }
            };
            if !a.stream.push(t) {
                done.push(i);
                continue;
            }
            if a.pen.is_some() {
                a.window.push(t);
            }
            if seqs[i].position as usize + 1 > seqs[i].context {
                // the sequence is full: the budget was the session's capacity
                a.stream.stop = Some(StopReason::MaxNew);
                done.push(i);
                continue;
            }
            a.next = Some(t);
        }
        done.sort_unstable();
        done.dedup();
        for &i in done.iter().rev() {
            let a = act.swap_remove(i);
            seqs.swap_remove(i);
            finish(a, health);
        }
    }
}

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
                EngineError::engine("the inkling2 route stopped serving".to_string()),
            );
            pr.run.fail();
            health.end_request();
            break;
        }
    }
}

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
    match render(tok, max_seq, &mut req) {
        Ok(Rendered {
            prompt,
            budget,
            sample,
            pen,
        }) => Some(Prepared {
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
            None
        }
    }
}

fn finish(mut a: Active, health: &Arc<RouteHealth>) {
    let elapsed = a.t0.elapsed().as_secs_f64();
    a.stream.finish(a.prompt.len(), elapsed);
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

fn fail_all(act: &mut Vec<Active>, seqs: &mut Vec<Seq>, health: &Arc<RouteHealth>, why: &str) {
    eprintln!("[inkling2-serve] {why}; failing {} request(s)", act.len());
    seqs.clear();
    for mut a in act.drain(..) {
        fail_request(&mut a.req, EngineError::engine(why.to_string()));
        health.note_request_fault();
        if let Some(run) = a.run.take() {
            run.fail();
        }
        health.end_request();
    }
}

fn drain_into<T>(rx: &Receiver<T>, pending: &mut VecDeque<T>) -> bool {
    loop {
        match rx.try_recv() {
            Ok(x) => pending.push_back(x),
            Err(std::sync::mpsc::TryRecvError::Empty) => return false,
            Err(std::sync::mpsc::TryRecvError::Disconnected) => return true,
        }
    }
}

fn fail_pending(pending: &mut VecDeque<Prepared>, health: &Arc<RouteHealth>, why: &str) {
    for mut pr in pending.drain(..) {
        fail_request(&mut pr.req, EngineError::engine(why.to_string()));
        pr.run.fail();
        health.end_request();
    }
}

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

    #[test]
    fn a_client_stop_ends_the_stream_after_its_token() {
        let stops = vec!["END".to_string()];
        assert_eq!(stop_end("abc E", "abc END x", &stops), Some(9));
        assert_eq!(stop_end("abc", "abc d", &stops), None);
        assert_eq!(stop_end("", "x", &[String::new()]), None);
    }

    #[test]
    fn greedy_takes_the_device_argmax_and_ties_go_low() {
        assert_eq!(choose(None, 7, 0, None, None, &[]).unwrap(), 7);
        let pen = Dsv4PenaltyCfg {
            last_n: usize::MAX,
            repeat: 1.0,
            freq: 0.0,
            present: 0.0,
        };
        assert_eq!(
            choose(Some(vec![1.0, 3.0, 3.0]), 9, 0, None, Some(&pen), &[]).unwrap(),
            1
        );
        assert!(choose(None, 0, 0, None, Some(&pen), &[]).is_err());
    }

    #[test]
    fn the_end_ids_hold_the_model_eos_once() {
        assert_eq!(end_ids(&[5], 200006), vec![5, 200006]);
        assert_eq!(end_ids(&[200006], 200006), vec![200006]);
    }
}
