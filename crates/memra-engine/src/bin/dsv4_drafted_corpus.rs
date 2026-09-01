//! dsv4-drafted-corpus: the DRAFTED-vs-PLAIN cell on OWNER-BLESSED corpora (iteration 3,
//! rung 4, task 2's honesty half).
//!
//! Why this exists next to `dsv4-decode-bench`: the lane's perf fixture is a 32-token
//! prompt continued free-running for thousands of tokens, and a long greedy continuation
//! DEGENERATES into repetition — which a drafter predicts almost perfectly. Acceptance
//! measured there is the friendliest possible observable, and the banked dspark-q38
//! correction is exactly about trusting a friendly acceptance observable (2.88
//! accept-length delivered 1.00x wall). So the drafted claim gets a second cell on real
//! agentic prompts from the owner-blessed SXC pools, with a BOUNDED generation per prompt
//! so no stream has time to degenerate.
//!
//! Instrument design:
//!   - BOTH arms in ONE process against ONE loaded model, alternating which arm goes first
//!     per (rep, prompt) so neither arm systematically gets the warmer hardware. That is a
//!     finer-grained interleave than the cross-process A/B protocol and removes load-time
//!     and clock-drift confounds entirely (interleaved-A/B protocol law, satisfied more
//!     strongly rather than less).
//!   - Timing covers the DECODE loop only (prefill excluded, reported separately).
//!   - The drafter is resident in both arms because the model loads once; residency is not
//!     time, and the plain arm's numbers here reproduce the drafter-free plain baseline —
//!     stated so the reader can check it.
//!   - Per prompt, the drafted token stream MUST equal the plain one: a free greedy
//!     spec==plain identity gate on real prompts, on top of the fixture gate.
//!
//! Greedy only. This lane's drafted path is the greedy propose-then-verify loop whose
//! verdict instrument IS greedy identity; a sampling-verify arm (rejection sampling over
//! the batched verify's per-position distributions) is a separate rung and is NOT claimed
//! here.
//!
//! Prompt file: {"prompts": [{"pool": "hermes", "ids": [..]}, ...]}
//!
//! Usage: dsv4-drafted-corpus <model-dir> <prompts.json> <out-dir> [n_new] [reps] [dev0,dev1]

use memra_engine::dsv4_gpu::{Dsv4Gpu, Dsv4Vt};
use memra_gguf::dsv4_forward::ActQuantVariant;
use std::io::Write;
use std::path::Path;

fn argmax(v: &[f32]) -> u32 {
    let mut best = 0usize;
    for i in 1..v.len() {
        if v[i] > v[best] {
            best = i;
        }
    }
    best as u32
}

fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    h.update(bytes);
    h.finalize().iter().map(|b| format!("{b:02x}")).collect()
}

/// Minimal reader for the prompt file (no serde dependency in this crate's bin set).
fn load_prompts(path: &Path) -> Vec<(String, Vec<u32>)> {
    let s = std::fs::read_to_string(path).expect("read prompts");
    let mut out = Vec::new();
    let mut rest = s.as_str();
    while let Some(i) = rest.find("\"pool\"") {
        rest = &rest[i + 6..];
        let q0 = rest.find('"').expect("pool open quote");
        let after = &rest[q0 + 1..];
        let q1 = after.find('"').expect("pool close quote");
        let pool = after[..q1].to_string();
        rest = &after[q1 + 1..];
        let j = rest.find("\"ids\"").expect("ids key");
        rest = &rest[j + 5..];
        let b0 = rest.find('[').expect("ids open");
        let b1 = rest.find(']').expect("ids close");
        let ids: Vec<u32> = rest[b0 + 1..b1]
            .split(',')
            .filter_map(|t| t.trim().parse::<u32>().ok())
            .collect();
        rest = &rest[b1 + 1..];
        assert!(!ids.is_empty(), "empty prompt ids for pool {pool}");
        out.push((pool, ids));
    }
    assert!(!out.is_empty(), "no prompts parsed");
    out
}

struct ArmAgg {
    tokens: usize,
    decode_us: u64,
    prefill_us: u64,
}

impl ArmAgg {
    fn new() -> Self {
        ArmAgg {
            tokens: 0,
            decode_us: 0,
            prefill_us: 0,
        }
    }
    fn ms_per_token(&self) -> f64 {
        self.decode_us as f64 / 1e3 / self.tokens as f64
    }
    fn tok_s(&self) -> f64 {
        self.tokens as f64 * 1e6 / self.decode_us as f64
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!(
            "usage: dsv4-drafted-corpus <model-dir> <prompts.json> <out-dir> [n_new] [reps] \
             [dev0,dev1]"
        );
        std::process::exit(2);
    }
    let t0 = std::time::Instant::now();
    let dir = Path::new(&args[1]);
    let prompts = load_prompts(Path::new(&args[2]));
    let out_dir = Path::new(&args[3]);
    std::fs::create_dir_all(out_dir).expect("mkdir out");
    let n_new: usize = args
        .get(4)
        .map(|x| x.parse().expect("n_new"))
        .unwrap_or(128);
    let reps: usize = args.get(5).map(|x| x.parse().expect("reps")).unwrap_or(3);
    let devices: Vec<usize> = args
        .get(6)
        .map(|s| s.split(',').map(|x| x.parse().expect("device")).collect())
        .unwrap_or_else(|| vec![0, 1]);
    if std::env::var("MEMRA_DSV4_DRAFTER").as_deref() != Ok("dspark") {
        eprintln!("REFUSE: this cell requires MEMRA_DSV4_DRAFTER=dspark");
        std::process::exit(2);
    }
    if std::env::var("MEMRA_DSV4_DECODE_PATH").as_deref() != Ok("device") {
        eprintln!("REFUSE: this cell requires MEMRA_DSV4_DECODE_PATH=device");
        std::process::exit(2);
    }
    // the perf fixtures' artifact variant — the numeric class every banked perf cell in
    // this lane ran on (the REF variant is the correctness gates' contract).
    let variant = ActQuantVariant::ClampOnly;
    let max_p = prompts.iter().map(|(_, i)| i.len()).max().unwrap();
    let max_seq = (max_p + n_new + 96).max(256);
    println!(
        "dsv4-drafted-corpus | model {} | prompts {} (max {max_p} ids) | n_new {n_new} | reps \
         {reps} | devices {devices:?} | GREEDY ONLY (this lane's drafted path; sampling \
         verify is a separate rung)",
        dir.display(),
        prompts.len()
    );
    let gpu = Dsv4Gpu::load(dir, &devices, variant, max_seq).expect("load");
    println!(
        "loaded: split at layer {}, verify tmax {}, t={:.0}s",
        gpu.split_at,
        gpu.verify_tmax(),
        t0.elapsed().as_secs_f64()
    );

    // MEMRA_DSV4_SPEC_DEPTH_SWEEP=2,3,4,5,6 -> depth-sweep cell instead of the single-arm
    // cell. Unset => the banked single-arm path below, byte-for-byte as before.
    if let Ok(spec) = std::env::var("MEMRA_DSV4_SPEC_DEPTH_SWEEP") {
        if std::env::var("MEMRA_DSV4_VT_SWEEP").is_ok() || std::env::var("MEMRA_DSV4_VT").is_ok() {
            eprintln!(
                "REFUSE: MEMRA_DSV4_SPEC_DEPTH_SWEEP with MEMRA_DSV4_VT_SWEEP or MEMRA_DSV4_VT \
                 set (one sweep per invocation; arms own the policy)"
            );
            std::process::exit(2);
        }
        let depths: Vec<usize> = spec
            .split(',')
            .filter_map(|t| t.trim().parse::<usize>().ok())
            .filter(|d| *d >= 1)
            .collect();
        assert!(
            !depths.is_empty(),
            "MEMRA_DSV4_SPEC_DEPTH_SWEEP parsed empty"
        );
        run_depth_sweep(
            &gpu,
            &prompts,
            n_new,
            reps,
            out_dir,
            depths.into_iter().map(DepthArm::new).collect(),
            t0,
        );
        return;
    }

    // MEMRA_DSV4_VT_SWEEP="off@4,slot@0.5,slot@0.6" -> the ds4f rung-1 confidence-window
    // A/B: fixed-depth control arms + slot-policy arms in ONE process (one load, one
    // thermal window, plain baseline shared, arm order rotated) — the same interleave
    // as the depth sweep. Mutually exclusive with the depth sweep and with the env
    // policy seam (the arms OWN the policy here).
    if let Ok(spec) = std::env::var("MEMRA_DSV4_VT_SWEEP") {
        if std::env::var("MEMRA_DSV4_VT").is_ok() {
            eprintln!("REFUSE: MEMRA_DSV4_VT_SWEEP with MEMRA_DSV4_VT set (arms own the policy)");
            std::process::exit(2);
        }
        let arms = parse_vt_arms(&spec);
        run_depth_sweep(&gpu, &prompts, n_new, reps, out_dir, arms, t0);
        return;
    }

    let mut plain = ArmAgg::new();
    let mut draft = ArmAgg::new();
    let mut rounds_total = 0usize;
    let mut accepted_total = 0usize;
    let mut t_batch_sum = 0usize;
    let mut t_batch_n = 0usize;
    let mut identity_fails: Vec<String> = Vec::new();
    let mut per_pool: std::collections::BTreeMap<String, (usize, u64, u64, usize, usize)> =
        Default::default();
    let mut rows: Vec<String> = Vec::new();
    // (t_batch, t_cap, accepts, emitted, round_us, confidence) for every round of every
    // prompt/rep -- the SPS(B) profile and the offline Algorithm-1 scan both read this.
    let mut all_rounds: Vec<(usize, usize, usize, usize, u64, Vec<f32>)> = Vec::new();

    for rep in 0..reps {
        for (pi, (pool, ids)) in prompts.iter().enumerate() {
            let plain_first = (rep + pi) % 2 == 0;
            let run_plain = || {
                let mut state = gpu.alloc_decode_state().expect("alloc state");
                let pt0 = std::time::Instant::now();
                let pre = gpu.prefill_with_cache(ids, &mut state).expect("prefill");
                let pre_us = pt0.elapsed().as_micros() as u64;
                let mut t = argmax(&pre.logits);
                let mut toks = Vec::with_capacity(n_new);
                let dt0 = std::time::Instant::now();
                for step in 0..n_new {
                    toks.push(t);
                    if step + 1 == n_new {
                        break;
                    }
                    t = gpu.decode_step_greedy(t, &mut state).expect("plain step");
                }
                (toks, dt0.elapsed().as_micros() as u64, pre_us)
            };
            let run_draft = || {
                let mut state = gpu.alloc_decode_state().expect("alloc state");
                let mut dstate = gpu.dspark_alloc_state().expect("alloc dspark");
                let mut vstate = gpu.alloc_verify_state().expect("alloc verify");
                let dt0 = std::time::Instant::now();
                let out = gpu
                    .spec_greedy_batched_with(ids, n_new, &mut state, &mut dstate, &mut vstate)
                    .expect("drafted run");
                let total_us = dt0.elapsed().as_micros() as u64;
                // prefill+prime is inside spec_greedy_batched_with; subtract the rounds'
                // measured wall from the total to attribute it
                let rounds_us: u64 = out.rounds.iter().map(|r| r.round_us).sum();
                let pre_us = total_us.saturating_sub(rounds_us);
                (out, rounds_us, pre_us)
            };

            // alternate which arm goes first so neither systematically gets the warmer card
            let ((plain_toks, p_us, ppre), (draft_out, d_us, dpre)) = if plain_first {
                let a = run_plain();
                let b = run_draft();
                (a, b)
            } else {
                let b = run_draft();
                let a = run_plain();
                (a, b)
            };
            plain.prefill_us += ppre;
            draft.prefill_us += dpre;
            let d_rounds = draft_out.rounds.len();
            let d_acc: usize = draft_out.rounds.iter().map(|r| r.accepts).sum();
            for r in &draft_out.rounds {
                if r.t_batch > 0 {
                    t_batch_sum += r.t_batch;
                    t_batch_n += 1;
                }
                all_rounds.push((
                    r.t_batch,
                    r.t_cap,
                    r.accepts,
                    r.emitted,
                    r.round_us,
                    r.confidence.clone(),
                ));
            }
            let draft_toks = draft_out.tokens;

            if draft_toks != plain_toks {
                let first = draft_toks
                    .iter()
                    .zip(&plain_toks)
                    .position(|(a, b)| a != b)
                    .unwrap_or(plain_toks.len().min(draft_toks.len()));
                identity_fails.push(format!(
                    "IDENTITY FAIL rep{rep} prompt{pi} ({pool}): drafted != plain at generated \
                     index {first} (drafted {:?} vs plain {:?})",
                    draft_toks.get(first),
                    plain_toks.get(first)
                ));
            }
            plain.tokens += plain_toks.len();
            plain.decode_us += p_us;
            draft.tokens += draft_toks.len();
            draft.decode_us += d_us;
            rounds_total += d_rounds;
            accepted_total += d_acc;
            let e = per_pool.entry(pool.clone()).or_insert((0, 0, 0, 0, 0));
            e.0 += plain_toks.len();
            e.1 += p_us;
            e.2 += d_us;
            e.3 += d_rounds;
            e.4 += d_acc;
            let row = format!(
                "rep{rep} p{pi:02} [{pool}] ids {} | plain {:.2} ms/tok | drafted {:.2} ms/tok \
                 | {:.3}x | rounds {d_rounds} accepted {d_acc} ({:.3}/round, {:.3} tok/round)",
                ids.len(),
                p_us as f64 / 1e3 / plain_toks.len() as f64,
                d_us as f64 / 1e3 / draft_toks.len() as f64,
                p_us as f64 / d_us as f64,
                d_acc as f64 / d_rounds.max(1) as f64,
                draft_toks.len() as f64 / d_rounds.max(1) as f64
            );
            println!("{row}");
            rows.push(row);
        }
        println!(
            "--- rep{rep} cumulative: plain {:.2} ms/tok ({:.1} tok/s) | drafted {:.2} ms/tok \
             ({:.1} tok/s) | {:.3}x | t={:.0}s",
            plain.ms_per_token(),
            plain.tok_s(),
            draft.ms_per_token(),
            draft.tok_s(),
            plain.ms_per_token() / draft.ms_per_token(),
            t0.elapsed().as_secs_f64()
        );
    }

    println!("\n=== OWNER-CORPORA DRAFTED CELL (measured, bench not serving) ===");
    println!(
        "prompts {} x reps {reps} x n_new {n_new} = {} tokens per arm",
        prompts.len(),
        plain.tokens
    );
    println!(
        "plain   : {:.2} ms/token = {:.1} tok/s bs=1 (prefill excluded: {:.1} ms total)",
        plain.ms_per_token(),
        plain.tok_s(),
        plain.prefill_us as f64 / 1e3
    );
    println!(
        "drafted : {:.2} ms/token = {:.1} tok/s bs=1 (prefill+prime excluded: {:.1} ms total)",
        draft.ms_per_token(),
        draft.tok_s(),
        draft.prefill_us as f64 / 1e3
    );
    println!(
        "SPEEDUP : {:.3}x  ({:.1} -> {:.1} tok/s)",
        plain.ms_per_token() / draft.ms_per_token(),
        plain.tok_s(),
        draft.tok_s()
    );
    // iteration-5: F itemisation. Prints only when MEMRA_DSV4_ROUND_PROFILE=1 or
    // MEMRA_DSV4_NVTX=1 armed the phase brackets; the denominator is THIS run's own measured
    // plain step, so every row is quoted in the unit F is defined in.
    memra_engine::dsv4_gpu::dsv4_phase_report(
        "drafted round",
        rounds_total as u64,
        plain.ms_per_token() * 1000.0,
    );
    println!(
        "acceptance (CORRECTNESS observable, never a speed claim — dspark-q38 law): rounds \
         {rounds_total} | accepted {accepted_total} | {:.4}/round | {:.4} tokens/round | mean T \
         forwarded {:.4}",
        accepted_total as f64 / rounds_total.max(1) as f64,
        draft.tokens as f64 / rounds_total.max(1) as f64,
        if t_batch_n > 0 {
            t_batch_sum as f64 / t_batch_n as f64
        } else {
            0.0
        }
    );
    // ---- SPS(B) profile + per-position conditional acceptance, on owner corpora --------
    {
        #[allow(clippy::type_complexity)]
        // allow: one-shot composite type; naming it would hide the shape that matters at the call site
        let steady: Vec<&(usize, usize, usize, usize, u64, Vec<f32>)> = all_rounds
            .iter()
            .filter(|r| r.0 > 0 && r.0 == r.1)
            .collect();
        if !steady.is_empty() {
            let n = steady.len() as f64;
            let mean_us = steady.iter().map(|r| r.4 as f64).sum::<f64>() / n;
            let mean_t = steady.iter().map(|r| r.0 as f64).sum::<f64>() / n;
            let mean_acc = steady.iter().map(|r| r.2 as f64).sum::<f64>() / n;
            let tau = steady.iter().map(|r| r.3 as f64).sum::<f64>() / n;
            println!(
                "[sps] depth-pinned rounds {} | mean T forwarded {:.4} | mean round {:.3} ms \
                 => SPS(B) {:.2} rounds/s | accepts {:.4}/round | tau* {:.4} tok/round | \
                 Theta = tau*.SPS = {:.2} tok/s",
                steady.len(),
                mean_t,
                mean_us / 1e3,
                1e6 / mean_us,
                mean_acc,
                tau,
                tau * 1e6 / mean_us
            );
            let kmax = steady.iter().map(|r| r.0 - 1).max().unwrap_or(0);
            let mut pos: Vec<String> = Vec::new();
            for j in 0..kmax {
                let reached = steady.iter().filter(|r| r.2 >= j).count();
                let took = steady.iter().filter(|r| r.2 > j).count();
                if reached > 0 {
                    pos.push(format!("p{}={:.4}", j + 1, took as f64 / reached as f64));
                }
            }
            println!(
                "[sps] per-position CONDITIONAL acceptance (this model, this box): {}",
                pos.join(" ")
            );
        }
        let side = out_dir.join("spec_rounds.json");
        let mut f = std::fs::File::create(&side).expect("rounds sidecar");
        writeln!(f, "{{\"schema\": \"dsv4-spec-rounds-v1\", \"rounds\": [").expect("w");
        for (i, r) in all_rounds.iter().enumerate() {
            let conf =
                r.5.iter()
                    .map(|c| format!("{c:.6}"))
                    .collect::<Vec<_>>()
                    .join(",");
            writeln!(
                f,
                "  {{\"t_batch\": {}, \"t_cap\": {}, \"accepts\": {}, \"emitted\": {}, \
                 \"round_us\": {}, \"conf\": [{conf}]}}{}",
                r.0,
                r.1,
                r.2,
                r.3,
                r.4,
                if i + 1 == all_rounds.len() { "" } else { "," }
            )
            .expect("w");
        }
        writeln!(f, "]}}").expect("w");
        println!(
            "[sps] {} rounds banked {}",
            all_rounds.len(),
            side.display()
        );
    }

    println!("per pool:");
    for (pool, (tk, pu, du, rd, ac)) in &per_pool {
        println!(
            "  {pool:<8} tokens {tk:<6} plain {:.2} ms/tok  drafted {:.2} ms/tok  {:.3}x  \
             accepted/round {:.3}",
            *pu as f64 / 1e3 / *tk as f64,
            *du as f64 / 1e3 / *tk as f64,
            *pu as f64 / *du as f64,
            *ac as f64 / (*rd).max(1) as f64
        );
    }
    println!(
        "greedy spec==plain identity on real prompts: {} / {} prompt-runs",
        prompts.len() * reps - identity_fails.len(),
        prompts.len() * reps
    );

    let mut f = std::fs::File::create(out_dir.join("drafted_corpus.json")).expect("json");
    write!(
        f,
        "{{\n  \"prompts\": {},\n  \"reps\": {reps},\n  \"n_new\": {n_new},\n  \
         \"tokens_per_arm\": {},\n  \"plain_ms_per_token\": {:.4},\n  \
         \"plain_tok_s\": {:.3},\n  \"drafted_ms_per_token\": {:.4},\n  \
         \"drafted_tok_s\": {:.3},\n  \"speedup\": {:.4},\n  \"rounds\": {rounds_total},\n  \
         \"accepted\": {accepted_total},\n  \"accepted_per_round\": {:.4},\n  \
         \"tokens_per_round\": {:.4},\n  \"identity_fails\": {},\n  \"rows_sha256\": \"{}\"\n}}\n",
        prompts.len(),
        plain.tokens,
        plain.ms_per_token(),
        plain.tok_s(),
        draft.ms_per_token(),
        draft.tok_s(),
        plain.ms_per_token() / draft.ms_per_token(),
        accepted_total as f64 / rounds_total.max(1) as f64,
        draft.tokens as f64 / rounds_total.max(1) as f64,
        identity_fails.len(),
        sha256_hex(rows.join("\n").as_bytes())
    )
    .expect("write json");

    if identity_fails.is_empty() {
        println!(
            "\nCORPORA CELL: identity [PASS] | banked {} | total elapsed {:.0}s",
            out_dir.join("drafted_corpus.json").display(),
            t0.elapsed().as_secs_f64()
        );
    } else {
        println!(
            "\nCORPORA CELL: identity [FAIL] {} finding(s)",
            identity_fails.len()
        );
        for l in &identity_fails {
            println!("  {l}");
        }
        std::process::exit(1);
    }
}

/// Per-arm accumulation for the depth / vt sweeps. A depth arm is (label "T{d}",
/// vt Off, depth cap d); a vt arm carries its policy explicitly (ds4f rung 1 —
/// one load, one thermal window, exactly the depth sweep's interleave).
struct DepthArm {
    label: String,
    depth: usize,
    vt: Dsv4Vt,
    tokens: usize,
    decode_us: u64,
    rounds: usize,
    accepts: usize,
    t_batch_sum: usize,
    /// (t_batch, t_cap, accepts, emitted, round_us, confidence)
    recs: Vec<(usize, usize, usize, usize, u64, Vec<f32>)>,
}

impl DepthArm {
    fn new(depth: usize) -> Self {
        Self::new_vt(format!("T{depth}"), depth, Dsv4Vt::Off)
    }
    fn new_vt(label: String, depth: usize, vt: Dsv4Vt) -> Self {
        DepthArm {
            label,
            depth,
            vt,
            tokens: 0,
            decode_us: 0,
            rounds: 0,
            accepts: 0,
            t_batch_sum: 0,
            recs: Vec::new(),
        }
    }
}

/// Parse the vt-sweep arms spec (`MEMRA_DSV4_VT_SWEEP`): comma-separated arms,
/// `off@<depth>` (fixed-depth control, vt off) or `slot@<tau>[:<floor>]` (confidence
/// window, no depth cap — the window IS the per-round depth). Refuses by name on
/// anything else; refuses an empty list.
fn parse_vt_arms(spec: &str) -> Vec<DepthArm> {
    let mut arms = Vec::new();
    for raw in spec.split(',') {
        let a = raw.trim();
        if a.is_empty() {
            continue;
        }
        if let Some(d) = a.strip_prefix("off@") {
            let depth: usize = d
                .parse()
                .unwrap_or_else(|_| panic!("MEMRA_DSV4_VT_SWEEP arm '{a}': bad depth"));
            assert!(
                depth >= 1,
                "MEMRA_DSV4_VT_SWEEP arm '{a}': depth must be >= 1"
            );
            arms.push(DepthArm::new_vt(format!("T{depth}"), depth, Dsv4Vt::Off));
        } else if let Some(rest) = a.strip_prefix("slot@") {
            let (tau_s, floor_s) = match rest.split_once(':') {
                Some((t, f)) => (t, Some(f)),
                None => (rest, None),
            };
            let vt = memra_engine::dsv4_gpu::resolve_vt(Some("slot"), Some(tau_s), floor_s)
                .unwrap_or_else(|e| panic!("MEMRA_DSV4_VT_SWEEP arm '{a}': {e}"));
            let label = match floor_s {
                Some(f) => format!("slot{tau_s}f{f}"),
                None => format!("slot{tau_s}"),
            };
            arms.push(DepthArm::new_vt(label, usize::MAX, vt));
        } else {
            panic!("MEMRA_DSV4_VT_SWEEP arm '{a}' unknown (off@<depth> | slot@<tau>[:<floor>])");
        }
    }
    assert!(!arms.is_empty(), "MEMRA_DSV4_VT_SWEEP parsed empty");
    arms
}

/// Draft-depth sweep on owner corpora: plain plus one drafted arm per depth, all in one
/// thermal window, arm order rotated per (rep, prompt).
#[allow(clippy::too_many_arguments)]
fn run_depth_sweep(
    gpu: &Dsv4Gpu,
    prompts: &[(String, Vec<u32>)],
    n_new: usize,
    reps: usize,
    out_dir: &Path,
    mut arms: Vec<DepthArm>,
    t0: std::time::Instant,
) {
    let labels: Vec<&str> = arms.iter().map(|a| a.label.as_str()).collect();
    println!(
        "\n=== DRAFTED-ARM SWEEP | arms {labels:?} | plain baseline shared | {} prompts x {reps} \
         reps x {n_new} tok | ONE load, ONE thermal window, arm order rotated ===",
        prompts.len()
    );
    let mut plain = ArmAgg::new();
    let mut identity_fails: Vec<String> = Vec::new();

    for rep in 0..reps {
        for (pi, (pool, ids)) in prompts.iter().enumerate() {
            // arm 0 = plain, arms 1.. = the drafted arms; rotate the visiting order
            let n_arms = arms.len() + 1;
            let rot = (rep + pi) % n_arms;
            let mut plain_toks: Option<Vec<u32>> = None;
            for step in 0..n_arms {
                let which = (rot + step) % n_arms;
                if which == 0 {
                    let mut state = gpu.alloc_decode_state().expect("alloc state");
                    let pre = gpu.prefill_with_cache(ids, &mut state).expect("prefill");
                    let mut t = argmax(&pre.logits);
                    let mut toks = Vec::with_capacity(n_new);
                    let dt0 = std::time::Instant::now();
                    for st in 0..n_new {
                        toks.push(t);
                        if st + 1 == n_new {
                            break;
                        }
                        t = gpu.decode_step_greedy(t, &mut state).expect("plain step");
                    }
                    plain.decode_us += dt0.elapsed().as_micros() as u64;
                    plain.tokens += toks.len();
                    plain_toks = Some(toks);
                } else {
                    let ai = which - 1;
                    let depth = arms[ai].depth;
                    let arm_vt = arms[ai].vt;
                    let mut state = gpu.alloc_decode_state().expect("alloc state");
                    let mut dstate = gpu.dspark_alloc_state().expect("alloc dspark");
                    let mut vstate = gpu.alloc_verify_state().expect("alloc verify");
                    let dt0 = std::time::Instant::now();
                    let out = gpu
                        .spec_greedy_batched_policy(
                            ids,
                            n_new,
                            &mut state,
                            &mut dstate,
                            &mut vstate,
                            depth,
                            arm_vt,
                        )
                        .expect("drafted run");
                    let total_us = dt0.elapsed().as_micros() as u64;
                    let rounds_us: u64 = out.rounds.iter().map(|r| r.round_us).sum();
                    let _prefill_us = total_us.saturating_sub(rounds_us);
                    let a = &mut arms[ai];
                    a.decode_us += rounds_us;
                    a.tokens += out.tokens.len();
                    a.rounds += out.rounds.len();
                    for r in &out.rounds {
                        a.accepts += r.accepts;
                        a.t_batch_sum += r.t_batch;
                        a.recs.push((
                            r.t_batch,
                            r.t_cap,
                            r.accepts,
                            r.emitted,
                            r.round_us,
                            r.confidence.clone(),
                        ));
                    }
                    // identity is the verdict instrument: every depth must reproduce plain
                    if let Some(pt) = &plain_toks
                        && &out.tokens != pt
                    {
                        let first = out
                            .tokens
                            .iter()
                            .zip(pt)
                            .position(|(a, b)| a != b)
                            .unwrap_or(pt.len().min(out.tokens.len()));
                        identity_fails.push(format!(
                            "IDENTITY FAIL rep{rep} prompt{pi} ({pool}) arm={}: drafted \
                                 != plain at generated index {first}",
                            arms[ai].label
                        ));
                    }
                }
            }
        }
        println!(
            "  rep {} / {reps} done, t={:.0}s",
            rep + 1,
            t0.elapsed().as_secs_f64()
        );
    }

    let p_ms = plain.ms_per_token();
    println!("\n=== DEPTH-SWEEP RESULT (owner corpora, measured, bench not serving) ===");
    println!(
        "PLAIN  : {:.3} ms/tok = {:.2} tok/s  ({} tokens)",
        p_ms,
        plain.tok_s(),
        plain.tokens
    );
    println!(
        "\n{:<10} {:>10} {:>9} {:>8} {:>9} {:>9} {:>9} {:>11} {:>9}",
        "arm", "ms/tok", "tok/s", "vs plain", "meanT", "acc/rnd", "tau*", "round ms", "SPS/s"
    );
    let mut table_json: Vec<String> = Vec::new();
    for a in &arms {
        // depth-pinned rounds only: a round the n_new budget truncated is not a depth-T round
        #[allow(clippy::type_complexity)]
        // allow: one-shot composite type; naming it would hide the shape that matters at the call site
        let steady: Vec<&(usize, usize, usize, usize, u64, Vec<f32>)> =
            a.recs.iter().filter(|r| r.0 > 0 && r.0 == r.1).collect();
        let (mean_round_us, tau, sps) = if steady.is_empty() {
            (0.0, 0.0, 0.0)
        } else {
            let n = steady.len() as f64;
            let m = steady.iter().map(|r| r.4 as f64).sum::<f64>() / n;
            let e = steady.iter().map(|r| r.3 as f64).sum::<f64>() / n;
            (m, e, 1e6 / m)
        };
        let ms = a.decode_us as f64 / 1e3 / a.tokens as f64;
        println!(
            "{:<10} {:>10.3} {:>9.2} {:>8.3}x {:>9.4} {:>9.4} {:>9.4} {:>11.3} {:>9.2}",
            a.label,
            ms,
            a.tokens as f64 * 1e6 / a.decode_us as f64,
            p_ms / ms,
            a.t_batch_sum as f64 / a.rounds.max(1) as f64,
            a.accepts as f64 / a.rounds.max(1) as f64,
            tau,
            mean_round_us / 1e3,
            sps
        );
        // per-position conditional acceptance, measured on THIS model and THIS box
        let kmax = steady.iter().map(|r| r.0 - 1).max().unwrap_or(0);
        let mut pos: Vec<String> = Vec::new();
        for j in 0..kmax {
            let reached = steady.iter().filter(|r| r.2 >= j).count();
            let took = steady.iter().filter(|r| r.2 > j).count();
            if reached > 0 {
                pos.push(format!("p{}={:.4}", j + 1, took as f64 / reached as f64));
            }
        }
        println!(
            "      per-position conditional acceptance: {}",
            pos.join(" ")
        );
        table_json.push(format!(
            "{{\"arm\": \"{}\", \"T\": {}, \"ms_per_tok\": {ms:.4}, \"tok_s\": {:.4}, \
             \"speedup\": {:.4}, \
             \"mean_T\": {:.4}, \"accepts_per_round\": {:.4}, \"tau_star\": {tau:.4}, \
             \"mean_round_us\": {mean_round_us:.1}, \"sps\": {sps:.3}, \"rounds\": {}, \
             \"steady_rounds\": {}}}",
            a.label,
            a.depth.min(9999),
            a.tokens as f64 * 1e6 / a.decode_us as f64,
            p_ms / ms,
            a.t_batch_sum as f64 / a.rounds.max(1) as f64,
            a.accepts as f64 / a.rounds.max(1) as f64,
            a.rounds,
            steady.len()
        ));
        // per-arm round sidecar for the offline Algorithm-1 scan
        let side = out_dir.join(format!("spec_rounds_{}.json", a.label));
        let mut f = std::fs::File::create(&side).expect("sidecar");
        writeln!(
            f,
            "{{\"schema\": \"dsv4-spec-rounds-v1\", \"arm\": \"{}\", \"T\": {}, \"rounds\": [",
            a.label,
            a.depth.min(9999)
        )
        .expect("w");
        for (i, r) in a.recs.iter().enumerate() {
            let conf =
                r.5.iter()
                    .map(|c| format!("{c:.6}"))
                    .collect::<Vec<_>>()
                    .join(",");
            writeln!(
                f,
                "  {{\"t_batch\": {}, \"t_cap\": {}, \"accepts\": {}, \"emitted\": {}, \
                 \"round_us\": {}, \"conf\": [{conf}]}}{}",
                r.0,
                r.1,
                r.2,
                r.3,
                r.4,
                if i + 1 == a.recs.len() { "" } else { "," }
            )
            .expect("w");
        }
        writeln!(f, "]}}").expect("w");
    }

    println!(
        "\nidentity across ALL depths: {}",
        if identity_fails.is_empty() {
            "PASS (every drafted arm reproduced plain token-for-token)".to_string()
        } else {
            format!("FAIL ({} cases)", identity_fails.len())
        }
    );
    for l in identity_fails.iter().take(10) {
        println!("  {l}");
    }
    println!(
        "*** acceptance is a CORRECTNESS observable, never a speed claim (dspark-q38 law); \
         the speed statement is the measured ms/tok column ***"
    );
    let jf = out_dir.join("depth_sweep.json");
    let mut f = std::fs::File::create(&jf).expect("json");
    write!(
        f,
        "{{\n  \"cell\": \"dsv4-depth-sweep\",\n  \"corpora\": \"owner-sxc\",\n  \
         \"n_new\": {n_new},\n  \"reps\": {reps},\n  \"prompts\": {},\n  \
         \"plain_ms_per_tok\": {p_ms:.4},\n  \"plain_tok_s\": {:.4},\n  \
         \"identity_fails\": {},\n  \"arms\": [{}]\n}}\n",
        prompts.len(),
        plain.tok_s(),
        identity_fails.len(),
        table_json.join(", ")
    )
    .expect("w");
    println!(
        "banked {} | total elapsed {:.0}s",
        jf.display(),
        t0.elapsed().as_secs_f64()
    );
}
