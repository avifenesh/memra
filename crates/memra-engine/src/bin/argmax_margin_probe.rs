//! argmax-margin-probe (lane/q8-argmax, 2026-08-06; verdict rewritten for memra#1199): measures
//! the run-gen prefill-vs-decode argmax gate at a window of prompt positions, and judges a
//! measured table.
//!
//! Why this exists: the gate (`run_gen.rs:880`) prints `logit maxdiff`, and a reader naturally
//! treats a big maxdiff as "big error". That is wrong, and it cost this lane's predecessors real
//! time. maxdiff is the max over a 248k-wide vocab of |prefill - decode|, dominated by
//! whatever logit is noisiest anywhere in the vocab, mostly deep in the tail, and it is
//! routinely 0.3-2.4 on runs the same gate calls MATCH. What decides a flip is the two
//! programs' top-2 margins: a flip needs the programs to disagree about the two contenders by
//! at least the sum of the two margins (`memra_engine::argmax_margin`, the gap shift). A near
//! tie is a flip whose gap shift is no larger than the largest one the same two programs show
//! at rows where they agree, measured over the whole prompt (the floor).
//!
//! It reports, for the batched-prefill config and a decode config:
//!   * per-position top-1 ids, top-2 margins and the spread at the contending ids over the last
//!     WINDOW positions of the prompt (teacher-forced on the prompt itself: bit-identical
//!     inputs, no stream separation),
//!   * one `detail` line per position: both programs' top-2 ids with their text and logits,
//!     each program's logit at the other's top-1, the spread and the gap shift,
//!   * the largest gap shift at an agreeing row (what a whole-prompt window calibrates) and
//!     the gap shift of every flip.
//!
//! It prints no verdict of its own on a measurement. `--judge <log>` is the verdict: it reads
//! the table of a log, looks up the floor row and calls `argmax_margin::judge`.
//! `tools/argmax-margin-gate.sh` runs it on the measured log and, for its canary, on a copy
//! with an injected row, so one function judges both.
//!
//! It also runs the BATCHED SERVING PRIME (`prime_cache`, the program that seeds generation in
//! `generate` and serving) once over the whole prompt and judges its last-position logits
//! against the tokenwise reference with the calibrated `forward::prime_gate_verdict`, printing
//! one `PRIME-GATE:` line. run-gen carries the same cell (its gap #46 arm), but the release
//! battery runs this probe and not run-gen, and run-spec primes both its reference and its
//! candidate through `prime_cache`, so without this line a prime-only regression passed the
//! whole battery.
//!
//! `--decode` picks the program on the decode side of the table:
//!   * `eager` (default): `decode_step`, the eager tokenwise program (`decode_step_h`). This arm
//!     also runs the batched-prime line above.
//!   * `batch`: the SERVED batched decode program at batch width 1. The server's batched
//!     scheduler decodes every non-speculative row of a batch-served model through
//!     `decode_step_batch_sampled_lean_masked` at every width, B=1 included (the B=1 fast path
//!     is an opt-in seam), and that body carries a decode-config FP-composition gap against
//!     `decode_step_h` (decode_batch.rs, `b1_fast_on`). So the eager arm can stay green while a
//!     change to the batched body or its epilogue moves served tokens. Speculative rows (the
//!     default `MEMRA_SERVE_SPEC` route for spec-eligible requests on a model with a draft head)
//!     decode through the spec path, which `run-spec` gates, not through this body. This arm
//!     makes the server's own call for one unconstrained greedy row (see `served_batch_decode`)
//!     and judges it against the same prefill reference with the same margin rules. It prints
//!     one `SERVED-ARGMAX:` line instead of the prime line: how many window steps'
//!     device-sampled token differs from the host argmax of the same served row. A model the
//!     scheduler decodes through the per-session eager body
//!     (`HybridModel::served_decode_is_eager`, the predicate the worker's decode sites read)
//!     has no served batched program: the arm then steps the program that does serve it,
//!     `decode_step`, and prints `SERVED-ARGMAX: n/a reason=eager-routed`.
//!
//! usage: argmax-margin-probe <model.gguf|hf_dir> <prompt-file> [window=24|all] [--decode eager|batch]
//!        argmax-margin-probe --judge <log> --max-flips N [--floors <tsv> --model M --prompt P
//!            --prompt-sha256 S --decode eager|batch --device D] [--calibrate --receipt PATH]
//! `window=all` measures every row the prefill reference can give (T - 3): the calibration
//! window. `--judge` runs no engine: exit 0 passes, 1 fails, 2 is a malformed table or call.
//! env: engine knobs (MEMRA_FA_SPLIT, MEMRA_Q80_G2, MEMRA_FAST, ...) steer the arm under test.

use memra_engine::Engine;
use memra_engine::argmax_margin::{self, FloorKey, Row};
use memra_engine::decode_batch::DevSamp;
use memra_engine::forward::argmax;
use memra_engine::hybrid::HybridModel;
use memra_engine::pp::new_cache as new_model_cache;
use memra_gguf::GgufFile;
use memra_tokenizer::Tokenizer;

const USAGE: &str = "usage: argmax-margin-probe <model.gguf|hf_dir> <prompt-file> [window=24|all] [--decode eager|batch]";
const JUDGE_USAGE: &str = "usage: argmax-margin-probe --judge <log> --max-flips N [--floors <tsv> --model M --prompt P --prompt-sha256 S --decode eager|batch --device D] [--calibrate --receipt PATH]";

/// The program on the decode side of the table (see the module doc).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DecodeArm {
    Eager,
    Batch,
}

/// How many of the prompt's last positions to measure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Window {
    Last(usize),
    /// Every row the prefill reference can give: the calibration window.
    All,
}

impl Window {
    /// The row count on a prompt of `t` tokens. The probe needs `window < t - 2`.
    fn rows(self, t: usize) -> usize {
        match self {
            Window::Last(n) => n,
            Window::All => t.saturating_sub(3),
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
struct ProbeArgs {
    model: String,
    prompt: String,
    window: Window,
    decode: DecodeArm,
}

fn parse_args(args: &[String]) -> Result<ProbeArgs, String> {
    let mut positional: Vec<&String> = Vec::new();
    let mut decode = DecodeArm::Eager;
    let mut it = args.iter();
    while let Some(arg) = it.next() {
        if arg == "--decode" {
            decode = match it.next().map(String::as_str) {
                Some("eager") => DecodeArm::Eager,
                Some("batch") => DecodeArm::Batch,
                other => {
                    return Err(format!(
                        "--decode takes eager or batch, got {other:?}; {USAGE}"
                    ));
                }
            };
        } else if arg.starts_with("--") {
            return Err(format!("unknown option {arg}; {USAGE}"));
        } else {
            positional.push(arg);
        }
    }
    let (model, prompt, window) = match positional.as_slice() {
        [model, prompt] => (model, prompt, Window::Last(24)),
        [model, prompt, window] if window.as_str() == "all" => (model, prompt, Window::All),
        [model, prompt, window] => (
            model,
            prompt,
            Window::Last(
                window
                    .parse::<usize>()
                    .map_err(|err| format!("window {window:?}: {err}"))?,
            ),
        ),
        _ => return Err(USAGE.into()),
    };
    if window == Window::Last(0) {
        return Err("window must be a positive integer or all".into());
    }
    Ok(ProbeArgs {
        model: model.to_string(),
        prompt: prompt.to_string(),
        window,
        decode,
    })
}

/// The SERVED batched decode program at batch width 1, teacher-forced over the whole prompt.
///
/// Each step is the call the server's batched scheduler makes for one unconstrained greedy row
/// (`crates/memra-server/src/worker.rs`, the unscheduled `decode_step_batch_sampled_lean_masked`
/// arm): one token, a cache from `pp::new_cache_planned`, the greedy device sample
/// `devsample_meta` builds, no grammar mask, lean logits. The engine routes it as it routes a
/// served tick of a server booted with no rewrite bundle: program arm, the B=1 fast-path seam
/// and the pipeline door. The probe installs no `MEMRA_REWRITE_BUNDLE` (the worker does at boot
/// when it is set), so `rewrite_allowed` is open for every surface and the engine's
/// unqualified-decode-batch fallback to eager rows cannot fire here: this cell gates the default
/// bundle-less served program, not a bundle-governed one. Lean rows come back empty, so each
/// window row is read from the cache's device park (`last_logits_dev`), the buffer the server's
/// retire-time pool park reads. The park is cleared before each window step, so a step that
/// parks nothing fails here instead of handing back the previous position's row.
///
/// Returns the window rows and the number of window steps whose device-sampled token differs
/// from the host argmax of the same parked row. The greedy device argmax is bit-identical to the
/// host argmax by contract (decode-batch-gate gate3 holds it at B>1).
fn served_batch_decode(
    e: &Engine,
    model: &HybridModel,
    prompt: &[u32],
    window: usize,
) -> Result<ServedBatchRun, Box<dyn std::error::Error>> {
    let t = prompt.len();
    let mut cache = memra_engine::pp::new_cache_planned(e, &model.cfg, &model.plan, t + 8)?;
    // devsample_meta's greedy row: temp 0 takes the device argmax.
    let greedy = [Some(DevSamp::new(0.0, 0, 0, 0, 1.0, 0.0))];
    let mut rows = Vec::with_capacity(window);
    let mut device_mismatch = 0usize;
    for (i, &tk) in prompt.iter().enumerate() {
        let in_window = i + window >= t;
        if in_window {
            cache.last_logits_dev = None;
        }
        let (_, next) = model.decode_step_batch_sampled_lean_masked(
            e,
            &[tk],
            &mut [&mut cache],
            &greedy,
            &[None],
            true,
        )?;
        if in_window {
            let parked = cache
                .last_logits_dev
                .as_ref()
                .ok_or("served batch step parked no logits row")?;
            let row = e.dtoh(parked)?;
            let token = next
                .first()
                .copied()
                .flatten()
                .ok_or("served batch step returned no device-sampled token")?;
            if token as usize != argmax(&row) {
                device_mismatch += 1;
            }
            rows.push(row);
        }
    }
    Ok(ServedBatchRun {
        rows,
        device_mismatch,
    })
}

/// What the served batch arm measured over the window.
struct ServedBatchRun {
    /// Host copies of the parked logits rows, one per window position.
    rows: Vec<Vec<f32>>,
    /// Window steps whose device-sampled token differs from the host argmax of its row.
    device_mismatch: usize,
}

/// The one `SERVED-ARGMAX:` line the gate script parses in `--decode batch` mode.
fn served_argmax_row(rows: usize, device_mismatch: usize) -> String {
    format!("SERVED-ARGMAX: rows={rows} device_mismatch={device_mismatch}")
}

/// The `SERVED-ARGMAX:` line for a model the batched scheduler decodes eagerly
/// (`HybridModel::served_decode_is_eager`). The gate reports it as `served=EAGER-ROUTED`; the
/// table above it then measures `decode_step`, the program that serves the model.
const SERVED_EAGER_ROUTED: &str = "SERVED-ARGMAX: n/a reason=eager-routed";

/// What the eager tokenwise program measured over the window.
struct EagerRun {
    /// Logits rows, one per window position.
    rows: Vec<Vec<f32>>,
    /// The decode cache, which the eager arm drops before the prime runs.
    cache: memra_engine::cache::Cache,
}

/// The eager tokenwise program (`decode_step`) over the whole prompt.
fn eager_decode(
    e: &Engine,
    model: &HybridModel,
    prompt: &[u32],
    window: usize,
) -> Result<EagerRun, Box<dyn std::error::Error>> {
    let t = prompt.len();
    let mut cache = new_model_cache(e, &model.cfg, t + 8)?;
    let mut rows = Vec::with_capacity(window);
    for (i, &tk) in prompt.iter().enumerate() {
        let l = model.decode_step(e, tk, &mut cache)?;
        if i + window >= t {
            rows.push(l);
        }
    }
    Ok(EagerRun { rows, cache })
}

/// Why the batched-prime arm does not run, if it does not: the same conditions run-gen's gap #46
/// cell uses. `tokenwise-served` is the only skip under which the served first token still
/// comes from a program this probe gates (the tokenwise loop of the table above).
fn prime_gate_skip_reason(
    t: usize,
    tokenwise_forced: bool,
    gate_off: bool,
    frozen_cpu_experts: bool,
) -> Option<&'static str> {
    if t < memra_engine::hybrid_forward::PRIME_MIN_T {
        Some("short-prompt")
    } else if tokenwise_forced {
        Some("tokenwise-forced")
    } else if gate_off {
        Some("gate-off")
    } else if frozen_cpu_experts {
        Some("tokenwise-served")
    } else {
        None
    }
}

/// The one `PRIME-GATE:` line the gate script parses. Ten significant digits, like the table.
fn prime_gate_row(v: &memra_engine::forward::PrimeGateVerdict) -> String {
    use memra_engine::forward::PrimeGateClass;
    let class = match v.class {
        PrimeGateClass::Match => "MATCH",
        PrimeGateClass::NearTieFlip => "FLIP-NEARTIE",
        PrimeGateClass::Structured => "MISMATCH-STRUCTURED",
    };
    format!(
        "PRIME-GATE: class={class} bp_argmax={} tw_argmax={} tw_margin={:.9e} bp_margin={:.9e} \
         maxdiff={:.9e}",
        v.bp_argmax, v.tw_argmax, v.tw_margin, v.bp_margin, v.maxdiff
    )
}

fn validate_placement(
    sharded_cross_device: bool,
    has_hyper_prefill: bool,
) -> Result<(), &'static str> {
    if sharded_cross_device && !has_hyper_prefill {
        return Err(
            "argmax-margin-probe: sharded cross-device prefill is unsupported for this trunk; no measurement was performed",
        );
    }
    Ok(())
}

/// The arguments of `--judge`.
#[derive(Debug, Default, PartialEq, Eq)]
struct JudgeArgs {
    log: String,
    max_flips: usize,
    floors: Option<String>,
    model: String,
    prompt: String,
    prompt_sha256: String,
    decode: String,
    device: String,
    calibrate: bool,
    receipt: String,
}

fn parse_judge_args(args: &[String]) -> Result<JudgeArgs, String> {
    let mut out = JudgeArgs::default();
    let mut max_flips = None;
    let mut it = args.iter();
    out.log = it
        .next()
        .filter(|a| !a.starts_with("--"))
        .ok_or(JUDGE_USAGE)?
        .clone();
    while let Some(arg) = it.next() {
        if arg == "--calibrate" {
            out.calibrate = true;
            continue;
        }
        let value = it
            .next()
            .ok_or_else(|| format!("{arg} takes a value; {JUDGE_USAGE}"))?
            .clone();
        match arg.as_str() {
            "--max-flips" => {
                max_flips = Some(
                    value
                        .parse::<usize>()
                        .map_err(|err| format!("--max-flips {value:?}: {err}"))?,
                )
            }
            "--floors" => out.floors = Some(value),
            "--model" => out.model = value,
            "--prompt" => out.prompt = value,
            "--prompt-sha256" => out.prompt_sha256 = value,
            "--decode" => out.decode = value,
            "--device" => out.device = value,
            "--receipt" => out.receipt = value,
            other => return Err(format!("unknown option {other}; {JUDGE_USAGE}")),
        }
    }
    let keyed = [
        &out.model,
        &out.prompt,
        &out.prompt_sha256,
        &out.decode,
        &out.device,
    ];
    if (out.floors.is_some() || out.calibrate) && keyed.iter().any(|v| v.is_empty()) {
        return Err(format!(
            "--floors and --calibrate need --model, --prompt, --prompt-sha256, --decode and --device; {JUDGE_USAGE}"
        ));
    }
    if out.calibrate {
        if out.receipt.is_empty() {
            return Err(format!("--calibrate needs --receipt; {JUDGE_USAGE}"));
        }
    } else {
        out.max_flips =
            max_flips.ok_or_else(|| format!("--max-flips is required; {JUDGE_USAGE}"))?;
    }
    Ok(out)
}

/// `--judge`: the verdict on the table of a log, as text and an exit code (0 pass, 1 fail).
/// `floors` is the text of the floors table when one was named. A malformed table or floors
/// table is an `Err` (exit 2).
fn judge_log(args: &JudgeArgs, log: &str, floors: Option<&str>) -> Result<(String, i32), String> {
    let rows = argmax_margin::parse_table(log)?;
    let key = FloorKey {
        model: &args.model,
        prompt: &args.prompt,
        prompt_sha256: &args.prompt_sha256,
        decode: &args.decode,
        device: &args.device,
    };
    if args.calibrate {
        let cal = argmax_margin::calibrate(&rows).ok_or("no agreeing row to calibrate from")?;
        return Ok((
            format!(
                "CALIBRATION: rows={} agreeing={} flips={} floor={:.9e} set_by_row={} p50={:.9e} \
                 p99={:.9e} p99.9={:.9e} flips_above_floor={} max_flips_in_12={}\nFLOOR-ROW: {}\n",
                rows.len(),
                cal.agreeing,
                cal.flips,
                cal.floor,
                cal.set_by_row,
                cal.p50,
                cal.p99,
                cal.p999,
                cal.known.len(),
                argmax_margin::max_flips_in_window(&rows, 12),
                argmax_margin::floor_line(&key, &cal, &args.receipt),
            ),
            0,
        ));
    }
    let table = match floors {
        Some(text) => argmax_margin::parse_floors(text)?,
        None => Vec::new(),
    };
    let row = argmax_margin::find_floor(&table, &key);
    let mut out = match row {
        Some(r) => format!("FLOOR: {}\n", r.describe()),
        None => format!(
            "FLOOR: none (UNCALIBRATED: no floor row for model={} prompt={} sha256={} decode={} \
             device={:?}; a flip fails, agreeing rows are not judged)\n",
            args.model, args.prompt, args.prompt_sha256, args.decode, args.device
        ),
    };
    let verdict = argmax_margin::judge(
        &rows,
        row.map(|r| r.floor),
        row.map_or(&[], |r| r.known.as_slice()),
        args.max_flips,
    );
    let detail = |pos: usize| {
        let head = format!("detail pos={pos} ");
        log.lines()
            .find(|l| l.starts_with(&head))
            .map(str::to_string)
    };
    out.push_str(&verdict.report(&detail));
    Ok((out, if verdict.passes() { 0 } else { 1 }))
}

fn judge_main(args: &[String]) -> i32 {
    let run = || -> Result<(String, i32), String> {
        let args = parse_judge_args(args)?;
        let log = std::fs::read_to_string(&args.log).map_err(|e| format!("{}: {e}", args.log))?;
        let floors = match &args.floors {
            Some(path) => Some(std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?),
            None => None,
        };
        judge_log(&args, &log, floors.as_deref())
    };
    match run() {
        Ok((text, code)) => {
            print!("{text}");
            code
        }
        Err(err) => {
            println!("MALFORMED: {err}");
            2
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.first().map(String::as_str) == Some("--judge") {
        std::process::exit(judge_main(&args[1..]));
    }
    let probe = parse_args(&args)?;
    let (model_path, prompt_file) = (&probe.model, &probe.prompt);

    let e = Engine::new(0)?;
    let path = std::path::Path::new(model_path);
    let (model, tok) = if path.is_dir() {
        let source = memra_gguf::source::SafetensorsSource::open(path)?;
        (
            HybridModel::load_from_source_without_mtp(&e, &source)?,
            Tokenizer::from_hf_dir(path).map_err(|err| format!("tokenizer: {err}"))?,
        )
    } else {
        let g = GgufFile::open(model_path)?;
        (
            HybridModel::load_without_mtp(&e, &g)?,
            Tokenizer::from_gguf(&g).map_err(|err| format!("tokenizer: {err}"))?,
        )
    };
    // forward_last has a PP prefill dispatch for the hyper-connection trunk. The
    // generic trunk still executes prefill through one engine; reject remote
    // weight placement rather than silently substitute different prime arithmetic.
    validate_placement(
        memra_engine::pp::pp_sharded_cross_device(),
        model.hyper.is_some(),
    )?;
    let text = std::fs::read_to_string(prompt_file)?;
    let prompt = tok.encode(&text, true);
    let t = prompt.len();
    let window = probe.window.rows(t);
    assert!(
        window >= 1 && window < t.saturating_sub(2),
        "prompt shorter than the window"
    );

    println!(
        "argmax-margin-probe: T={t} window={window} sm_count={} model={}",
        e.sm_count(),
        model_path
    );
    println!(
        "env: MEMRA_FA_SPLIT={} MEMRA_Q80_G2={} MEMRA_FAST={} MEMRA_PRIME_CHUNK={}",
        std::env::var("MEMRA_FA_SPLIT").unwrap_or_else(|_| "<unset>".into()),
        std::env::var("MEMRA_Q80_G2").unwrap_or_else(|_| "<unset>".into()),
        std::env::var("MEMRA_FAST").unwrap_or_else(|_| "<unset>".into()),
        std::env::var("MEMRA_PRIME_CHUNK").unwrap_or_else(|_| "<unset>".into()),
    );
    // The server's own routing read: a model its batched scheduler decodes eagerly has no
    // served batched program, so the batch arm steps the program that serves it.
    let eager_routed = probe.decode == DecodeArm::Batch && model.served_decode_is_eager();
    if eager_routed {
        println!(
            "decode program: served per-session eager body (decode_step); the batched scheduler \
             routes this model eager (HybridModel::served_decode_is_eager) batch_program={:?}",
            model.decode_batch_program(),
        );
    } else if probe.decode == DecodeArm::Batch {
        println!(
            "decode program: served batch B=1 (decode_step_batch_sampled_lean_masked, greedy \
             device sample, lean) batch_program={:?} MEMRA_SERVE_B1FAST={}",
            model.decode_batch_program(),
            std::env::var("MEMRA_SERVE_B1FAST").unwrap_or_else(|_| "<unset>".into()),
        );
    }

    // --- config A: the decode side, capturing logits at the window positions ---
    // This is the gate's "decode" side. Stepping the whole prompt gives us, for free, the
    // per-position top-2 margin distribution under one fixed arithmetic config: the eager
    // tokenwise path (`decode_step`) or the served batched program at B=1.
    let (dec_at, decode_cache, served_line) = match probe.decode {
        DecodeArm::Eager => {
            let run = eager_decode(&e, &model, &prompt, window)?;
            (run.rows, Some(run.cache), None)
        }
        DecodeArm::Batch if eager_routed => {
            let run = eager_decode(&e, &model, &prompt, window)?;
            (
                run.rows,
                Some(run.cache),
                Some(SERVED_EAGER_ROUTED.to_string()),
            )
        }
        DecodeArm::Batch => {
            let run = served_batch_decode(&e, &model, &prompt, window)?;
            (
                run.rows,
                None,
                Some(served_argmax_row(window, run.device_mismatch)),
            )
        }
    };

    // --- config B: the batched prefill path (the gate's "prefill" side), one forward per
    //     truncation length so we get the SAME positions under the other config. This is the
    //     expensive half (window forwards over a ~2k prompt) — the honest way, since
    //     forward_last only returns the last row.
    let mut pre_at: Vec<Vec<f32>> = Vec::with_capacity(window);
    for i in (t - window)..t {
        pre_at.push(model.forward_last(&e, &prompt[..=i])?);
    }

    // --- config C (eager arm): the batched serving prime (`prime_cache`) over the whole prompt,
    //     judged against the tokenwise logits at the last position (see the module doc). The
    //     decode cache is dropped first so the prime's cache does not double the KV footprint.
    //     The batch arm reports its served-token check instead; the eager cell owns the prime.
    drop(decode_cache);
    let report_line = match served_line {
        Some(line) => line,
        None => match prime_gate_skip_reason(
            t,
            std::env::var("MEMRA_PRIME_TOKENWISE").is_ok(),
            std::env::var("MEMRA_PRIME_GATE").as_deref() == Ok("0"),
            e.frozen_cpu_experts_prefer_tokenwise_prime(),
        ) {
            Some(reason) => format!("PRIME-GATE: SKIP reason={reason}"),
            None => {
                let mut pc = new_model_cache(&e, &model.cfg, t + 8)?;
                let (l_bp, _, _) = model.prime_cache(&e, &prompt, &mut pc, 0)?;
                let tokenwise = dec_at.last().expect("window >= 1 rows of decode logits");
                prime_gate_row(&memra_engine::forward::prime_gate_verdict(tokenwise, &l_bp))
            }
        },
    };

    // --- per-position comparison ---
    println!(
        "\npos      prefill_top1  margin_p     decode_top1  margin_d     delta@ids      agree"
    );
    let rows: Vec<Row> = (0..window)
        .map(|k| Row::from_logits(t - window + k, &pre_at[k], &dec_at[k]))
        .collect();
    for row in &rows {
        println!("{}", row.table_line());
    }

    // --- one examination line per position: enough to read any row from the log alone ---
    println!(
        "\nrows in detail (top-2 ids, text, logits; each program's logit at the other's top-1):"
    );
    let text = |id: usize| tok.decode(&[id as u32]);
    for (k, row) in rows.iter().enumerate() {
        println!(
            "{}",
            argmax_margin::detail_line(row.pos, &pre_at[k], &dec_at[k], &text)
        );
    }

    let mut sorted: Vec<f32> = rows.iter().map(|r| r.margin_d).collect();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let pct = |q: f64| sorted[((sorted.len() - 1) as f64 * q).round() as usize];
    println!(
        "\nmargin distribution (decode config, {} positions): min {:.4}  p10 {:.4}  p50 {:.4}  p90 {:.4}  max {:.4}",
        sorted.len(),
        sorted[0],
        pct(0.10),
        pct(0.50),
        pct(0.90),
        sorted[sorted.len() - 1]
    );
    let mut sp: Vec<f32> = rows.iter().map(|r| r.margin_p).collect();
    sp.sort_by(|a, b| a.total_cmp(b));
    println!(
        "margin distribution (prefill config): min {:.4}  p50 {:.4}  max {:.4}",
        sp[0],
        sp[sp.len() / 2],
        sp[sp.len() - 1]
    );
    let last = rows.last().expect("window >= 1");
    println!(
        "DECISION POSITION (the gate's): prefill_top1={} decode_top1={} margin_p={:.4} \
         margin_d={:.4} config_delta_at_ids={:.4} gap_shift={:.4}",
        last.prefill_top1,
        last.decode_top1,
        last.margin_p,
        last.margin_d,
        last.spread,
        last.gap_shift()
    );
    let flips = rows.iter().filter(|r| r.is_flip()).count();
    println!(
        "agreement across sampled positions: {}/{} ({} flip(s))",
        window - flips,
        window,
        flips
    );
    // What the measurement holds for the verdict. The verdict itself is `--judge`, which reads
    // this log's table against the floor row of this model, prompt, decode program and device.
    for r in rows.iter().filter(|r| r.is_flip()) {
        println!(
            "flip@{}: margins {:.4} + {:.4} = gap shift {:.4} (spread {:.4})",
            r.pos,
            r.margin_p,
            r.margin_d,
            r.gap_shift(),
            r.spread
        );
    }
    match argmax_margin::calibrate(&rows) {
        Some(cal) => println!(
            "largest gap shift at an agreeing row: {:.9e} at row {} ({} agreeing rows; this is \
             the floor when the window is the whole prompt)",
            cal.floor, cal.set_by_row, cal.agreeing
        ),
        None => println!("largest gap shift at an agreeing row: none (no row agrees)"),
    }
    println!("{report_line}");
    // REPORTER ONLY: the gate (tools/argmax-margin-gate.sh) runs `--judge` on this log and
    // reads the PRIME-GATE or SERVED-ARGMAX line, and owns the exit code: a measurement-side
    // exit would double-fail ahead of the judge and break the canary's inject-then-judge flow.
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        DecodeArm, JudgeArgs, ProbeArgs, SERVED_EAGER_ROUTED, Window, judge_log, parse_args,
        parse_judge_args, prime_gate_row, prime_gate_skip_reason, served_argmax_row,
        validate_placement,
    };
    use memra_engine::argmax_margin::Row;
    use memra_engine::forward::{PrimeGateClass, PrimeGateVerdict};

    fn args(list: &[&str]) -> Vec<String> {
        list.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn decode_arm_defaults_to_eager_and_takes_the_batch_cell_anywhere() {
        let expect = |window, decode| ProbeArgs {
            model: "m.gguf".into(),
            prompt: "p.txt".into(),
            window,
            decode,
        };
        assert_eq!(
            parse_args(&args(&["m.gguf", "p.txt"])),
            Ok(expect(Window::Last(24), DecodeArm::Eager))
        );
        assert_eq!(
            parse_args(&args(&["m.gguf", "p.txt", "12"])),
            Ok(expect(Window::Last(12), DecodeArm::Eager))
        );
        assert_eq!(
            parse_args(&args(&["m.gguf", "p.txt", "12", "--decode", "batch"])),
            Ok(expect(Window::Last(12), DecodeArm::Batch))
        );
        assert_eq!(
            parse_args(&args(&["--decode", "batch", "m.gguf", "p.txt"])),
            Ok(expect(Window::Last(24), DecodeArm::Batch))
        );
        assert_eq!(
            parse_args(&args(&["m.gguf", "p.txt", "--decode", "eager"])),
            Ok(expect(Window::Last(24), DecodeArm::Eager))
        );
        // The calibration window: every row the prefill reference can give.
        assert_eq!(
            parse_args(&args(&["m.gguf", "p.txt", "all", "--decode", "batch"])),
            Ok(expect(Window::All, DecodeArm::Batch))
        );
        assert_eq!(Window::All.rows(2048), 2045);
        assert_eq!(Window::Last(12).rows(2048), 12);
    }

    #[test]
    fn bad_probe_arguments_are_refused() {
        for bad in [
            &["m.gguf"][..],
            &["m.gguf", "p.txt", "0"],
            &["m.gguf", "p.txt", "x"],
            &["m.gguf", "p.txt", "12", "13"],
            &["m.gguf", "p.txt", "--decode"],
            &["m.gguf", "p.txt", "--decode", "dc"],
            &["m.gguf", "p.txt", "--window", "12"],
        ] {
            assert!(parse_args(&args(bad)).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn served_row_names_the_fields_the_gate_parses() {
        assert_eq!(
            served_argmax_row(12, 0),
            "SERVED-ARGMAX: rows=12 device_mismatch=0"
        );
        assert_eq!(
            served_argmax_row(12, 3),
            "SERVED-ARGMAX: rows=12 device_mismatch=3"
        );
        // tools/argmax-margin-gate.sh matches this line verbatim (served=EAGER-ROUTED).
        assert_eq!(
            SERVED_EAGER_ROUTED,
            "SERVED-ARGMAX: n/a reason=eager-routed"
        );
    }

    #[test]
    fn prime_arm_runs_on_the_battery_shape_and_names_every_skip() {
        assert_eq!(prime_gate_skip_reason(2048, false, false, false), None);
        assert_eq!(
            prime_gate_skip_reason(8, false, false, false),
            Some("short-prompt")
        );
        assert_eq!(
            prime_gate_skip_reason(2048, true, false, false),
            Some("tokenwise-forced")
        );
        assert_eq!(
            prime_gate_skip_reason(2048, false, true, false),
            Some("gate-off")
        );
        assert_eq!(
            prime_gate_skip_reason(2048, false, false, true),
            Some("tokenwise-served")
        );
    }

    #[test]
    fn prime_row_names_the_class_the_gate_parses() {
        let mut v = PrimeGateVerdict {
            tw_argmax: 7,
            bp_argmax: 7,
            tw_margin: 1.5,
            bp_margin: 1.25,
            maxdiff: 0.03,
            class: PrimeGateClass::Match,
        };
        let row = prime_gate_row(&v);
        assert!(
            row.starts_with("PRIME-GATE: class=MATCH bp_argmax=7 tw_argmax=7 "),
            "{row}"
        );
        v.class = PrimeGateClass::NearTieFlip;
        assert!(prime_gate_row(&v).starts_with("PRIME-GATE: class=FLIP-NEARTIE "));
        v.class = PrimeGateClass::Structured;
        assert!(prime_gate_row(&v).starts_with("PRIME-GATE: class=MISMATCH-STRUCTURED "));
    }

    #[test]
    fn unsupported_sharded_prefill_is_rejected() {
        assert!(validate_placement(true, false).is_err());
        assert!(validate_placement(true, true).is_ok());
        assert!(validate_placement(false, false).is_ok());
        assert!(validate_placement(false, true).is_ok());
    }

    fn table(rows: &[Row]) -> String {
        rows.iter().map(|r| r.table_line() + "\n").collect()
    }

    fn row(pos: usize, p1: usize, mp: f32, d1: usize, md: f32) -> Row {
        Row {
            pos,
            prefill_top1: p1,
            margin_p: mp,
            decode_top1: d1,
            margin_d: md,
            spread: mp.max(md),
        }
    }

    const FLOORS: &str = "# model\tprompt\tsha\tdecode\tdevice\tfloor\trows\tset_by_row\tp50\tp99\tp999\tknown_rows\treceipt\n\
        m\tboard-2048\tabc\teager\tdevice A\t4.0e-1\t2040\t2047\t1.0e-1\t2.0e-1\t3.0e-1\t-\tresearch/x/gate.log\n";
    /// The same cell with one known row: main's calibration flipped row 41 at margins 1.5 and 2.5.
    const FLOORS_KNOWN: &str = "m\tboard-2048\tabc\teager\tdevice A\t4.0e-1\t2040\t2047\t1.0e-1\t2.0e-1\t3.0e-1\t41:18:1.5e0:15:2.5e0\tresearch/x/gate.log\n";

    fn judge_args(extra: &[&str]) -> JudgeArgs {
        let mut list = vec![
            "gate.log",
            "--max-flips",
            "1",
            "--floors",
            "floors.tsv",
            "--model",
            "m",
            "--prompt",
            "board-2048",
            "--prompt-sha256",
            "abc",
            "--decode",
            "eager",
            "--device",
            "device A",
        ];
        list.extend_from_slice(extra);
        parse_judge_args(&args(&list)).unwrap()
    }

    #[test]
    fn judge_arguments_are_checked() {
        let a = judge_args(&[]);
        assert_eq!(
            (a.log.as_str(), a.max_flips, a.calibrate),
            ("gate.log", 1, false)
        );
        assert_eq!(a.floors.as_deref(), Some("floors.tsv"));
        // No floors table: the judge still runs and every table is uncalibrated.
        assert!(parse_judge_args(&args(&["gate.log", "--max-flips", "1"])).is_ok());
        for bad in [
            &[][..],
            &["--max-flips", "1"],
            &["gate.log"],
            &["gate.log", "--max-flips", "x"],
            &["gate.log", "--max-flips"],
            &["gate.log", "--max-flips", "1", "--floors", "f.tsv"],
            &["gate.log", "--max-flips", "1", "--margin-floor", "0.5"],
            &["gate.log", "--calibrate"],
        ] {
            assert!(parse_judge_args(&args(bad)).is_err(), "{bad:?}");
        }
        assert!(judge_args(&["--calibrate", "--receipt", "research/x/gate.log"]).calibrate);
    }

    #[test]
    fn the_judge_fails_a_wide_flip_and_passes_a_near_tie() {
        // The red arm on the gate's own parse path: one table, three injected rows.
        let clean = [row(10, 1, 2.0, 1, 2.1), row(11, 2, 5.0, 2, 5.3)];
        let (text, code) = judge_log(&judge_args(&[]), &table(&clean), Some(FLOORS)).unwrap();
        assert_eq!(code, 0, "{text}");
        assert!(text.starts_with("FLOOR: 0.4000 (largest gap shift over 2040 agreeing rows, set by row 2047; p50 0.1000, p99 0.2000, p99.9 0.3000; no calibration flip above it; research/x/gate.log)\n"), "{text}");
        assert!(text.contains("SUMMARY flips=0 bad=0 "), "{text}");

        let mut wide = clean.to_vec();
        wide.push(row(12, 3, 14.4, 4, 2.5));
        let log = format!("{}detail pos=12 the examination line\n", table(&wide));
        let (text, code) = judge_log(&judge_args(&[]), &log, Some(FLOORS)).unwrap();
        assert_eq!(code, 1, "{text}");
        assert!(
            text.contains(
                "WIDE-FLIP pos=12 prefill_top1=3 margin_p=14.4000 decode_top1=4 margin_d=2.5000 "
            ),
            "{text}"
        );
        assert!(
            text.contains(
                "gap_shift=16.9000 > floor=0.4000\n  detail pos=12 the examination line\n"
            ),
            "{text}"
        );
        assert!(
            text.contains("SUMMARY flips=1 bad=1 near_ties=0 wide_flips=1 "),
            "{text}"
        );

        let mut tie = clean.to_vec();
        tie.push(row(12, 3, 0.02, 4, 0.03));
        let (text, code) = judge_log(&judge_args(&[]), &table(&tie), Some(FLOORS)).unwrap();
        assert_eq!(code, 0, "{text}");
        assert!(text.contains("near-tie pos=12 "), "{text}");
        assert!(
            text.contains("SUMMARY flips=1 bad=0 near_ties=1 "),
            "{text}"
        );

        // A second near tie is over the budget of 1.
        tie.push(row(13, 5, 0.01, 6, 0.01));
        let (text, code) = judge_log(&judge_args(&[]), &table(&tie), Some(FLOORS)).unwrap();
        assert_eq!(code, 1, "{text}");
        assert!(text.contains("TOO-MANY-NEAR-TIES 2 > budget 1"), "{text}");

        // An agreeing row whose margins differ by more than the floor.
        let wide_row = [row(10, 1, 14.4, 1, 7.2)];
        let (text, code) = judge_log(&judge_args(&[]), &table(&wide_row), Some(FLOORS)).unwrap();
        assert_eq!(code, 1, "{text}");
        assert!(text.contains("WIDE-ROW pos=10 "), "{text}");
    }

    #[test]
    fn without_a_floor_row_a_flip_fails_and_a_clean_table_says_uncalibrated() {
        let other_device = judge_args(&["--device", "device B"]);
        let clean = [row(10, 1, 14.4, 1, 7.2)];
        let (text, code) = judge_log(&other_device, &table(&clean), Some(FLOORS)).unwrap();
        assert_eq!(code, 0, "{text}");
        assert!(
            text.starts_with("FLOOR: none (UNCALIBRATED: no floor row for model=m "),
            "{text}"
        );
        assert!(text.contains("SUMMARY flips=0 bad=0 "), "{text}");
        assert!(text.contains(" floor=none "), "{text}");

        let tie = [row(12, 3, 0.02, 4, 0.03)];
        let (text, code) = judge_log(&other_device, &table(&tie), Some(FLOORS)).unwrap();
        assert_eq!(code, 1, "{text}");
        assert!(text.contains("UNCALIBRATED-FLIP pos=12 "), "{text}");
        // The same without any floors table.
        let bare = parse_judge_args(&args(&["gate.log", "--max-flips", "1"])).unwrap();
        assert_eq!(judge_log(&bare, &table(&tie), None).unwrap().1, 1);
    }

    #[test]
    fn calibration_prints_the_floors_row_of_its_table() {
        let rows = [
            row(10, 1, 2.0, 1, 2.25),
            row(11, 2, 5.0, 2, 5.75),
            row(12, 3, 0.02, 4, 0.03),
        ];
        let a = judge_args(&["--calibrate", "--receipt", "research/x/gate.log"]);
        let (text, code) = judge_log(&a, &table(&rows), None).unwrap();
        assert_eq!(code, 0);
        assert_eq!(
            text,
            "CALIBRATION: rows=3 agreeing=2 flips=1 floor=7.500000000e-1 set_by_row=11 p50=7.500000000e-1 p99=7.500000000e-1 p99.9=7.500000000e-1 flips_above_floor=0 max_flips_in_12=1\n\
             FLOOR-ROW: m\tboard-2048\tabc\teager\tdevice A\t7.500000000e-1\t2\t11\t7.500000000e-1\t7.500000000e-1\t7.500000000e-1\t-\tresearch/x/gate.log\n"
        );
        // The printed row parses, and judges its own table green.
        let line = text
            .lines()
            .nth(1)
            .unwrap()
            .strip_prefix("FLOOR-ROW: ")
            .unwrap();
        let (_, code) = judge_log(&judge_args(&[]), &table(&rows), Some(line)).unwrap();
        assert_eq!(code, 0);
    }

    #[test]
    fn a_known_row_in_the_window_is_named_and_passes_unless_it_is_wider() {
        // Main's calibration recorded row 41 as a flip at margins 1.5 and 2.5 (shift 4.0),
        // above the floor of 0.4. The same row in a judged window is named and passes.
        let same = [row(40, 1, 2.0, 1, 2.1), row(41, 18, 1.5, 15, 2.5)];
        let log = format!("{}detail pos=41 the examination line\n", table(&same));
        let (text, code) = judge_log(&judge_args(&[]), &log, Some(FLOORS_KNOWN)).unwrap();
        assert_eq!(code, 0, "{text}");
        assert!(text.starts_with("FLOOR: 0.4000 (largest gap shift over 2040 agreeing rows, set by row 2047; p50 0.1000, p99 0.2000, p99.9 0.3000; known rows (calibration flips above it, memra#1203): 41 (shift 4.0000); research/x/gate.log)\n"), "{text}");
        assert!(
            text.contains(
                "KNOWN-ROW pos=41 prefill_top1=18 margin_p=1.5000 decode_top1=15 margin_d=2.5000 "
            ),
            "{text}"
        );
        assert!(text.contains("gap_shift=4.0000 <= recorded=4.0000 (calibration: prefill_top1=18 margin_p=1.5000 decode_top1=15 margin_d=2.5000; memra#1203) floor=0.4000\n  detail pos=41 the examination line\n"), "{text}");
        assert!(
            text.contains("SUMMARY flips=1 bad=0 near_ties=0 "),
            "{text}"
        );
        assert!(text.contains(" known_rows=1 known_rows_wider=0 "), "{text}");

        // A candidate whose gap shift at that row exceeds main's recorded shift fails.
        let wider = [row(40, 1, 2.0, 1, 2.1), row(41, 18, 1.5, 15, 2.75)];
        let (text, code) = judge_log(&judge_args(&[]), &table(&wider), Some(FLOORS_KNOWN)).unwrap();
        assert_eq!(code, 1, "{text}");
        assert!(text.contains("KNOWN-ROW-WIDER pos=41 "), "{text}");
        assert!(
            text.contains("gap_shift=4.2500 > recorded=4.0000 "),
            "{text}"
        );
        assert!(text.contains(" known_rows=0 known_rows_wider=1 "), "{text}");

        // Without the known row in the floor row, main's own flip there is a wide flip.
        let (text, code) = judge_log(&judge_args(&[]), &table(&same), Some(FLOORS)).unwrap();
        assert_eq!(code, 1, "{text}");
        assert!(text.contains("WIDE-FLIP pos=41 "), "{text}");
    }

    #[test]
    fn calibration_lists_its_own_flips_above_the_floor_as_known_rows() {
        let rows = [
            row(10, 1, 2.0, 1, 2.25),
            row(11, 2, 5.0, 2, 5.75),
            row(41, 18, 1.5, 15, 2.5),
        ];
        let a = judge_args(&["--calibrate", "--receipt", "research/x/gate.log"]);
        let (text, _) = judge_log(&a, &table(&rows), None).unwrap();
        assert!(text.contains(" flips_above_floor=1 "), "{text}");
        let line = text
            .lines()
            .nth(1)
            .unwrap()
            .strip_prefix("FLOOR-ROW: ")
            .unwrap();
        assert!(
            line.contains("\t41:18:1.500000000e0:15:2.500000000e0\t"),
            "{line}"
        );
        // The printed row judges its own table green, the known row named.
        let (text, code) = judge_log(&judge_args(&[]), &table(&rows), Some(line)).unwrap();
        assert_eq!(code, 0, "{text}");
        assert!(text.contains("KNOWN-ROW pos=41 "), "{text}");
    }

    #[test]
    fn a_malformed_table_or_floors_table_is_an_error() {
        assert!(judge_log(&judge_args(&[]), "no table", Some(FLOORS)).is_err());
        let clean = [row(10, 1, 2.0, 1, 2.1)];
        assert!(judge_log(&judge_args(&[]), &table(&clean), Some("one\ttwo\n")).is_err());
    }
}
