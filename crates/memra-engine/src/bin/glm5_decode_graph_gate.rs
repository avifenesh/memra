//! glm5_decode-graph gate: the T=1 decode walk replayed as per-stage CUDA graphs
//! (`MEMRA_GLM5_DECODE_GRAPH=1`) must produce a BYTE-IDENTICAL token stream to the eager walk,
//! and its device MoE arm must select the same experts with the same routing weights, per layer
//! per token.
//!
//! WHY BIT-IDENTITY IS THE BAR AND NOT A TOLERANCE. The door does not reschedule the walk; it
//! records it. Every captured kernel is the kernel the eager walk launches, over the same
//! operands, in the same order. There is exactly ONE arithmetic-adjacent change under the door,
//! and it is the price of capture: the routed MoE stops reading its selection back to the host
//! and builds the pointer/scale tables on device instead (`vrows_t1_dev` in `hybrid_forward.rs`,
//! kernel `moe_vrows_tables_from_sel`), which routes the layer through the verify-rows kernel
//! pair at `n_pairs = n_used`. That pair is the per-row form of the fused epilogue, which is the
//! per-token form of the sequential `qmatvec_expert_q8` + `ffn_act_lim` + `axpy_into` chain, and
//! the table values are term-for-term the host loop's. So the two arms are ONE numeric program,
//! and this gate is the proof rather than the assertion (CLAUDE.md, "one numeric program per
//! request": graph vs eager is a named pair to keep honest).
//!
//! ARMS, from ONE prompt and ONE artifact:
//!   A. EAGER — door off, `MEMRA_GLM5_GRAPH_SEL_LEDGER=1` so the host-oracle arm records the
//!      selection it reads back per layer per token.
//!   B. GRAPH — door on, same ledger, so the device arm records the selection it keeps on the
//!      device. Both arms decode `--steps` greedy tokens from a fresh cache.
//! Then: token ids compared byte for byte, and the per-(token, layer) selection rows compared
//! index for index and weight bit for bit.
//!
//! SCOPE OF THE SELECTION HALF, under a pp split: a device ledger slot can only be read back
//! through an Engine on its OWN device and this binary holds the head engine, so the selection
//! comparison covers the HEAD stage's captured layers (both arms filtered to that device, so
//! the comparison is like for like). The TOKEN comparison is unaffected and covers every stage
//! end to end.
//!
//! THE RE-CAPTURE PATH IS EXERCISED ON PURPOSE. Halfway through the graph arm the gate RE-SEATS
//! one captured layer's recurrent state — a fresh buffer holding the same bytes — which is the
//! shape of a snapshot restore or a reuse-pool rehydrate and the only thing the pool's pointer
//! signature exists to catch. The run then has to stay byte-identical ACROSS the re-capture, and
//! the gate FAILS if no re-capture actually happened. This arm exists because the first box run
//! reached `CUDA_ERROR_INVALID_VALUE` inside that path in production, on a gate that could not
//! have caught it.
//!
//! NON-VACUITY IS ENFORCED. The gate refuses to pass unless (1) the door actually engaged
//! (`GLM5_DECODE_GRAPH_REPLAYS > 0` and captured layers > 0) — an eager fall-through would make
//! arm B a copy of arm A and the comparison meaningless; and (2) the ledger actually recorded
//! rows on both arms. A green run that captured nothing is a FAIL here.
//!
//! TIMING. `--reps N` (default 5) interleaved A/B/A/B... per-token milliseconds for both arms,
//! reported as per-rep values and medians. These are a lane instrument, not a published number:
//! quote them only from the box that ran them, interleaved, per the measurement laws.
//!
//! INVOCATION (the session runs this on the box):
//!   GLM5_ARTIFACT=/data/glm5.3-flash-nvfp4 \
//!   MEMRA_PP_DEVICES=0,1 MEMRA_PP_STAGES=2 \
//!   MEMRA_HTOD_DIET=1 \
//!   cargo run --release -p memra-engine --bin glm5-decode-graph-gate -- --steps 64 --reps 5
//!
//! `MEMRA_HTOD_DIET=1` is REQUIRED, not decorative: without it the shared expert uploads a
//! pageable constant per MoE layer and the door refuses the range (it says so once, on stderr).
//! The gate reads `MEMRA_PP_DEVICES` only to echo it — the pipeline door reads it itself.

use memra_engine::Engine;
use memra_engine::forward::argmax;
use memra_engine::glm5_sel_ledger::{self, SelRow};
use memra_engine::hybrid::HybridModel;
use memra_gguf::source::SafetensorsSource;
use memra_kv::Cache;
use std::sync::atomic::Ordering;
use std::time::Instant;

fn arg_val(rest: &[String], key: &str) -> Option<String> {
    rest.iter()
        .position(|a| a == key)
        .and_then(|i| rest.get(i + 1))
        .cloned()
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.total_cmp(b));
    if v.is_empty() {
        return f64::NAN;
    }
    v[v.len() / 2]
}

/// One arm: prime the prompt, then decode `steps` greedy tokens, returning the token tape, the
/// per-(step, layer) selection rows, and the per-token wall milliseconds.
///
/// The ledger is drained PER STEP: its device slots are overwritten every token, so a drain at
/// the end would only ever see the last one.
///
/// [`ArmOut`] names the arm's shape once: the token tape, the per-(step, layer) selection rows,
/// and the per-token wall milliseconds.
type ArmOut = Result<(Vec<u32>, Vec<Vec<SelRow>>, f64, bool), Box<dyn std::error::Error>>;

/// RE-SEAT one captured layer's recurrent state: allocate a fresh buffer, copy the CURRENT bytes
/// into it, and put it in the cache's slot. The walk is arithmetically untouched (same bytes,
/// same kernels) but the device POINTER the captured graphs baked is now stale — which is the
/// exact shape of a snapshot restore or a reuse-pool rehydrate, and the only thing the pool's
/// pointer signature exists to catch.
///
/// The first layer carrying a recurrent slot is a KDA layer (MLA/DSA layers carry `latent`, not
/// `recur`), so it sits inside a captured run; and being the first, it is on the HEAD stage,
/// which is the device this binary's engine owns — so the fresh allocation and the copy are local.
fn reseat_first_recurrent_layer(
    e: &Engine,
    cache: &mut Cache,
) -> Result<Option<usize>, Box<dyn std::error::Error>> {
    let Some((il, rl)) = cache
        .recur
        .iter_mut()
        .enumerate()
        .find_map(|(i, r)| r.as_mut().map(|r| (i, r)))
    else {
        return Ok(None);
    };
    let n = rl.ssm_state.len();
    let mut fresh = e.uninit(n)?;
    e.copy_into(&mut fresh, 0, &rl.ssm_state, n)?;
    rl.ssm_state = fresh;
    Ok(Some(il))
}

#[allow(clippy::too_many_arguments)] // allow: one arm knob per axis the gate drives; a struct would hide them at the call site
fn run_arm(
    e: &Engine,
    m: &HybridModel,
    prompt: &[u32],
    steps: usize,
    graph_door: bool,
    ledger: bool,
    reseat_at: Option<usize>,
    trace: bool,
) -> ArmOut {
    // SAFETY: single-threaded gate binary; the doors are read per call by the engine, and no
    // other thread exists to observe the environment mid-flight.
    unsafe {
        if graph_door {
            std::env::set_var("MEMRA_GLM5_DECODE_GRAPH", "1");
        } else {
            std::env::remove_var("MEMRA_GLM5_DECODE_GRAPH");
        }
        if ledger {
            std::env::set_var("MEMRA_GLM5_GRAPH_SEL_LEDGER", "1");
        } else {
            std::env::remove_var("MEMRA_GLM5_GRAPH_SEL_LEDGER");
        }
    }
    // SAFETY: single-threaded gate binary (same reasoning as the door vars above).
    unsafe {
        if trace {
            std::env::set_var("MEMRA_GLM5_GRAPH_TRACE", "1");
        } else {
            std::env::remove_var("MEMRA_GLM5_GRAPH_TRACE");
        }
    }
    glm5_sel_ledger::reset_host();
    // RESET THE TRACE BUDGET AT THE ARM SWITCH. Take 10 printed four identical `arm=host il=3`
    // lines and not one `arm=device` line: the budget was process-global, the eager arm runs
    // first, and it spent the whole thing. A two-arm comparison's budget belongs to the arm.
    memra_engine::glm5_trace_reset();

    let mut cache = Cache::new(e, &m.cfg, prompt.len() + steps + 8)?;
    let (logits, _h_seed, _hiddens) = m.prime_cache(e, prompt, &mut cache, 0)?;
    let mut tok = argmax(&logits) as u32;
    let mut tape = vec![tok];
    let mut rows: Vec<Vec<SelRow>> = Vec::with_capacity(steps);
    if ledger {
        glm5_sel_ledger::reset_host();
    }

    e.stream().synchronize()?;
    let t0 = Instant::now();
    let mut recaptured = false;
    for step in 1..steps {
        // `--trace`: one line per token with the door's counters, so a run that dies mid-walk says
        // which step it reached and whether the door was still replaying at that point.
        if trace {
            eprintln!(
                "[gate] step {step} door={graph_door} replays={} captures={}",
                memra_engine::GLM5_DECODE_GRAPH_REPLAYS.load(Ordering::Relaxed),
                memra_engine::GLM5_DECODE_GRAPH_CAPTURES.load(Ordering::Relaxed),
            );
        }
        // FORCED RE-SEAT ARM: make the pool's invalidation path RUN, rather than hope a real
        // session trips it. Without this the gate passes while re-capture is broken — which is
        // exactly how the first box run reached `CUDA_ERROR_INVALID_VALUE` in production
        // instead of here.
        if reseat_at == Some(step) && graph_door {
            let before = memra_engine::GLM5_DECODE_GRAPH_CAPTURES.load(Ordering::Relaxed);
            let il = reseat_first_recurrent_layer(e, &mut cache)?;
            let l = m.decode_step(e, tok, &mut cache)?;
            let after = memra_engine::GLM5_DECODE_GRAPH_CAPTURES.load(Ordering::Relaxed);
            recaptured = after > before;
            eprintln!(
                "[gate] forced re-seat at step {step} (layer {il:?}): captures {before} -> \
                 {after}, recaptured={recaptured}"
            );
            tok = argmax(&l) as u32;
            tape.push(tok);
            if ledger {
                let dev = e.ctx().ordinal();
                let mut step_rows = glm5_sel_ledger::drain_device(e)?;
                let host = glm5_sel_ledger::take_host();
                if !host.is_empty() {
                    step_rows = host;
                }
                step_rows.retain(|r| r.dev == dev);
                step_rows.sort_by_key(|r| (r.dev, r.layer));
                rows.push(step_rows);
            }
            continue;
        }
        let l = m.decode_step(e, tok, &mut cache)?;
        // Step 1's top-1 VALUE, on both arms: an all-zero logit vector argmaxes to 0 and is
        // indistinguishable from a wrong-but-valid token in the tape alone. Box run 4 read
        // `graph=0` at every step and this line is what separates the two readings.
        if step == 1 {
            let (idx, val) =
                l.iter()
                    .enumerate()
                    .fold((0usize, f32::NEG_INFINITY), |acc, (i, &v)| {
                        if v > acc.1 { (i, v) } else { acc }
                    });
            let nz = l.iter().filter(|v| **v != 0.0).count();
            eprintln!(
                "[gate] step 1 door={graph_door} logits: top1 idx={idx} val={val:.6e} \
                 nonzero={nz}/{}",
                l.len()
            );
        }
        tok = argmax(&l) as u32;
        tape.push(tok);
        if ledger {
            // Device rows first (they are this token's, in the persistent slots), then the host
            // rows the eager arm pushed. Exactly one of the two is non-empty per arm.
            //
            // SCOPE, stated rather than assumed: the device slots can only be read back through
            // an Engine on their OWN device, and this binary holds the head engine only. Under a
            // pp split that means the SELECTION comparison covers the head stage's captured
            // layers; both arms are filtered to that same device so the comparison is like for
            // like rather than silently ragged. TOKEN identity is unaffected and covers every
            // stage end to end, and a device/host selection divergence is a per-layer property
            // that would show on the head stage too.
            let dev = e.ctx().ordinal();
            let mut step_rows = glm5_sel_ledger::drain_device(e)?;
            let host = glm5_sel_ledger::take_host();
            if !host.is_empty() {
                step_rows = host;
            }
            step_rows.retain(|r| r.dev == dev);
            step_rows.sort_by_key(|r| (r.dev, r.layer));
            rows.push(step_rows);
        }
    }
    e.stream().synchronize()?;
    let ms_per_token =
        t0.elapsed().as_secs_f64() * 1000.0 / (steps.saturating_sub(1)).max(1) as f64;
    Ok((tape, rows, ms_per_token, recaptured))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let rest: Vec<String> = std::env::args().skip(1).collect();
    let steps: usize = arg_val(&rest, "--steps")
        .and_then(|v| v.parse().ok())
        .unwrap_or(64);
    let reps: usize = arg_val(&rest, "--reps")
        .and_then(|v| v.parse().ok())
        .unwrap_or(5);
    let prompt_len: usize = arg_val(&rest, "--prompt-len")
        .and_then(|v| v.parse().ok())
        .unwrap_or(64);
    let trace = rest.iter().any(|a| a == "--trace");
    let artifact = std::env::var("GLM5_ARTIFACT").map_err(
        |_| "GLM5_ARTIFACT must name the glm5_next artifact directory (safetensors checkpoint)",
    )?;

    println!(
        "[glm5-decode-graph-gate] artifact={artifact} steps={steps} reps={reps} \
         prompt_len={prompt_len} trace={trace} MEMRA_PP_DEVICES={:?} MEMRA_PP_STAGES={:?} \
         MEMRA_HTOD_DIET={:?} MEMRA_HC_DECODE_WS={:?}",
        std::env::var("MEMRA_PP_DEVICES").ok(),
        std::env::var("MEMRA_PP_STAGES").ok(),
        std::env::var("MEMRA_HTOD_DIET").ok(),
        std::env::var("MEMRA_HC_DECODE_WS").ok(),
    );

    let e = Engine::new(0)?;
    println!("[glm5-decode-graph-gate] GPU0: {}", e.ctx().name()?);
    let src = SafetensorsSource::open(std::path::Path::new(&artifact))?;
    let m = HybridModel::load_from_source(&e, &src)?;
    println!(
        "[glm5-decode-graph-gate] loaded: n_layer={} n_embd={} hyper={}",
        m.cfg.n_layer,
        m.cfg.n_embd,
        m.hyper.is_some(),
    );
    if m.hyper.is_none() {
        return Err(
            "this artifact carries no HyperConnections trunk; the door has nothing to capture"
                .into(),
        );
    }

    let prompt: Vec<u32> = (0..prompt_len)
        .map(|i| (101 + (i * 7) % 900) as u32)
        .collect();

    // ---- identity arms ----
    // The graph arm forces a RE-SEAT halfway through, so identity is proven ACROSS a
    // re-capture rather than only on a pool that was captured once and never invalidated. The
    // eager arm has no pool, so the re-seat would prove nothing there and is not run.
    let reseat_at = Some((steps / 2).max(2));
    let (eager_tape, eager_rows, _, _) = run_arm(&e, &m, &prompt, steps, false, true, None, trace)?;
    let replays_before = memra_engine::GLM5_DECODE_GRAPH_REPLAYS.load(Ordering::Relaxed);
    let (graph_tape, graph_rows, _, recaptured) =
        run_arm(&e, &m, &prompt, steps, true, true, reseat_at, trace)?;
    let replays = memra_engine::GLM5_DECODE_GRAPH_REPLAYS.load(Ordering::Relaxed) - replays_before;
    let captured_layers = memra_engine::GLM5_DECODE_GRAPH_LAYERS.load(Ordering::Relaxed);
    let captures = memra_engine::GLM5_DECODE_GRAPH_CAPTURES.load(Ordering::Relaxed);

    // REPORTED FIRST, on purpose: a run that dies later still says how far the door got.
    println!(
        "door: replays={replays} captures={captures} captured_layers={captured_layers} \
         forced_recapture={recaptured}"
    );
    use std::io::Write;
    let _ = std::io::stdout().flush();

    let mut fail = Vec::new();

    // NON-VACUITY: an eager fall-through would make the arms trivially equal.
    if replays == 0 || captured_layers == 0 {
        fail.push(format!(
            "VACUOUS: the door never replayed (replays={replays}, captured_layers={captured_layers}, \
             captures={captures}). The refusal reason is on stderr as [glm5-decode-graph] eager: ..."
        ));
    }
    if !recaptured {
        fail.push(format!(
            "VACUOUS RE-CAPTURE ARM: the forced re-seat at step {reseat_at:?} did not make the \
             pool re-capture, so this run says nothing about the invalidation path. Either the \
             re-seated layer is not inside a captured run, or the pointer signature no longer \
             sees a re-seat."
        ));
    }
    if eager_rows.iter().all(|r| r.is_empty()) || graph_rows.iter().all(|r| r.is_empty()) {
        fail.push("VACUOUS: the selection ledger recorded no rows on one of the arms".to_string());
    }

    // TOKEN IDENTITY.
    if eager_tape.len() != graph_tape.len() {
        fail.push(format!(
            "token tape length {} (eager) != {} (graph)",
            eager_tape.len(),
            graph_tape.len()
        ));
    }
    let mut tok_mismatch = 0usize;
    for (i, (a, b)) in eager_tape.iter().zip(graph_tape.iter()).enumerate() {
        if a != b {
            if tok_mismatch < 5 {
                println!("  TOKEN MISMATCH step {i}: eager={a} graph={b}");
            }
            tok_mismatch += 1;
        }
    }
    if tok_mismatch > 0 {
        fail.push(format!(
            "{tok_mismatch}/{} token ids differ",
            eager_tape.len()
        ));
    }

    // SELECTION IDENTITY, per token per layer. Weights compared on their BITS: a routing weight
    // that differs in the last ulp is a different program, and `total_cmp` on f32 would hide a
    // -0.0/0.0 split that the expert accumulation does not.
    let mut sel_mismatch = 0usize;
    let mut compared = 0usize;
    for (step, (ea, ga)) in eager_rows.iter().zip(graph_rows.iter()).enumerate() {
        if ea.len() != ga.len() {
            fail.push(format!(
                "step {step}: ledger row count {} (eager) != {} (graph)",
                ea.len(),
                ga.len()
            ));
            break;
        }
        for (er, gr) in ea.iter().zip(ga.iter()) {
            compared += 1;
            let idx_same = er.layer == gr.layer && er.sel == gr.sel;
            let w_same = er.w.len() == gr.w.len()
                && er
                    .w
                    .iter()
                    .zip(gr.w.iter())
                    .all(|(a, b)| a.to_bits() == b.to_bits());
            if !(idx_same && w_same) {
                if sel_mismatch < 5 {
                    println!(
                        "  SELECTION MISMATCH step {step} layer {}/{}: eager sel={:?} w={:?} | \
                         graph sel={:?} w={:?}",
                        er.layer, gr.layer, er.sel, er.w, gr.sel, gr.w
                    );
                }
                sel_mismatch += 1;
            }
        }
    }
    if sel_mismatch > 0 {
        fail.push(format!(
            "{sel_mismatch}/{compared} (token, layer) expert selections differ"
        ));
    }

    println!(
        "identity: tokens {}/{} match; selections {}/{} match (head device only under a pp \
         split)",
        eager_tape.len() - tok_mismatch,
        eager_tape.len(),
        compared - sel_mismatch,
        compared,
    );

    // ---- timing, interleaved, ledger OFF (it is an instrument, not a measured configuration) ----
    let mut eager_ms = Vec::with_capacity(reps);
    let mut graph_ms = Vec::with_capacity(reps);
    for r in 0..reps {
        // Timing arms take the un-perturbed walk: a forced re-capture is a correctness arm,
        // never a measured configuration.
        let (_, _, a, _) = run_arm(&e, &m, &prompt, steps, false, false, None, false)?;
        let (_, _, b, _) = run_arm(&e, &m, &prompt, steps, true, false, None, false)?;
        println!("  rep {r}: eager {a:.3} ms/token   graph {b:.3} ms/token");
        eager_ms.push(a);
        graph_ms.push(b);
    }
    let me = median(&mut eager_ms.clone());
    let mg = median(&mut graph_ms.clone());
    println!(
        "per-token ms (N={reps}, interleaved A/B): eager median {me:.3}  graph median {mg:.3}  \
         delta {:+.2}%",
        100.0 * (me - mg) / me
    );

    if fail.is_empty() {
        println!("ALL GREEN: glm5-decode-graph gate ({steps} steps, {compared} selection rows)");
        Ok(())
    } else {
        for f in &fail {
            println!("FAIL: {f}");
        }
        Err("glm5-decode-graph-gate FAILED".into())
    }
}
