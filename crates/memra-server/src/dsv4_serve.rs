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
//!   - no prefix/continuation caches: `n_cached` is honestly 0 on every request;
//!   - no response_format/grammar (refused by name);
//!   - streaming granularity is the spec ROUND (or every plain token) — the commit
//!     callback seam on the gated drivers, `None` = byte-identical bench behavior.

use crate::worker::{EngineError, Event, ModelCaps, Request, SpecUsage};
use memra_engine::dsv4_gpu::{
    Dsv4Gpu, Dsv4PenaltyCfg, Dsv4SampleCfg, dsv4_penalize_row, dsv4_sample_row, resolve_vt,
};
use memra_gguf::dsv4_forward::ActQuantVariant;
use memra_tokenizer::{Tokenizer, chat};
use std::path::Path;
use std::sync::Arc;

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
    let gpu = Dsv4Gpu::load(dir, &devices, variant, max_seq)?;
    let spec = gpu.dspark.is_some();
    let eos = tok.eos_id();
    eprintln!(
        "[dsv4-serve] {name}: loaded on devices {devices:?}, contract {variant:?}, \
         max_seq {max_seq}, drafter {}",
        if spec {
            "RESIDENT (spec route armed)"
        } else {
            "absent (plain only)"
        }
    );
    Ok(Dsv4Model {
        gpu: Arc::new(gpu),
        tok,
        max_seq,
        spec,
        eos,
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
    }
}

/// Spawn the serving thread; the returned Sender is the model's admission queue.
pub fn spawn(name: String, m: Dsv4Model) -> std::sync::mpsc::Sender<Box<Request>> {
    let (tx, rx) = std::sync::mpsc::channel::<Box<Request>>();
    std::thread::Builder::new()
        .name(format!("dsv4-serve-{name}"))
        .spawn(move || {
            while let Ok(mut req) = rx.recv() {
                // The worker's DSV4 channel is unbounded, so the hard admission reservation
                // remains held until this serving thread actually receives the request. Merely
                // forwarding it from the command channel must not make the queue appear empty.
                crate::worker::release_admission_reservation(req.lane);
                if req.tx.is_closed() {
                    continue; // client gone while queued
                }
                let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    serve_one(&m, &mut req)
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
    tx: &'a tokio::sync::mpsc::UnboundedSender<Event>,
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
        tx: &'a tokio::sync::mpsc::UnboundedSender<Event>,
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

    fn finish(self, n_prompt: usize, elapsed_s: f64, spec: Option<SpecUsage>) {
        if self.client_gone {
            return; // receipts stay with the terminal-less stream (HTTP layer owns it)
        }
        let _ = self.tx.send(Event::TokenSnapshot(self.ids.clone()));
        let _ = self.tx.send(Event::Done {
            stop_reason: self.stop_reason.unwrap_or("length").to_string(),
            n_tokens: self.ids.len(),
            n_prompt,
            n_cached: 0,
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

fn serve_one(m: &Dsv4Model, req: &mut Request) -> Result<(), EngineError> {
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
    let _ = req.tx.send(Event::PromptUsage {
        n_prompt: prompt.len(),
        n_cached: 0,
    });

    let mut eos_set = req.params.eos.clone();
    if !eos_set.contains(&m.eos) {
        eos_set.push(m.eos);
    }
    let mut emit = Emit::new(&m.tok, &req.tx, eos_set, &req.stop_strings, budget);
    let mut spec_usage: Option<SpecUsage> = None;

    if m.spec && !(greedy && penalties_set) {
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
        let mut state = m.gpu.alloc_decode_state().map_err(EngineError::engine)?;
        let mut dstate = m.gpu.dspark_alloc_state().map_err(EngineError::engine)?;
        let mut vstate = m.gpu.alloc_verify_state().map_err(EngineError::engine)?;
        // generation budget + 1: the drivers count the head token of the final round
        // inside n_new; Emit owns the exact budget/EOS truncation either way.
        let n_new = budget;
        let mut cb = |new: &[u32]| emit.push(new);
        let run = if greedy {
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
    } else {
        // trunk-only plain loops
        let mut state = m.gpu.alloc_decode_state().map_err(EngineError::engine)?;
        let pre = m
            .gpu
            .prefill_with_cache(&prompt, &mut state)
            .map_err(EngineError::engine)?;
        let p0 = prompt.len();
        // running penalty window: prompt ++ every committed token (rung-2 slice 2)
        let mut window: Vec<u32> = if pen_cfg.is_some() {
            prompt.clone()
        } else {
            Vec::new()
        };
        if greedy {
            let mut t = if let Some(pc) = &pen_cfg {
                let mut row = pre.logits.clone();
                dsv4_penalize_row(&mut row, &window, pc);
                argmax(&row)
            } else {
                argmax(&pre.logits)
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
            let mut row0 = pre.logits.clone();
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
    }

    emit.finish(prompt.len(), t0.elapsed().as_secs_f64(), spec_usage);
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
mod unicode_window_tests {
    use super::{scan_stop_cut, snap};

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
}
