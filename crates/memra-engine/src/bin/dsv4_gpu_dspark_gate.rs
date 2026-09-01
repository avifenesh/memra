//! dsv4-gpu-dspark-gate: the GPU DSpark drafter gate (iteration 3, rungs 3 and 4).
//!
//! Runs every arm in ONE process against the same loaded model:
//!   arm P  (plain)      — free-running device greedy, drafter resident but never called.
//!   arm DS (drafted)    — propose-then-verify with SEQUENTIAL verification (rung 3).
//!   arm DB (drafted)    — propose-then-verify with the BATCHED T=k+1 device verify and
//!                         §3.1 device commit/rollback (rung 4 — the rung that buys the
//!                         wall time; no wall number is emitted here).
//!
//! Verdicts:
//!
//!  (b) GREEDY SPEC==PLAIN IDENTITY LAW — arm DS and arm DB token streams must be
//!      byte-exact equal to arm P's. Hard pass/fail, no threshold doctrine.
//!
//!  (c) BATCHED == SEQUENTIAL, BIT FOR BIT — the decisive port instrument, and the
//!      reason (b) means anything on the device at all. A batched verify pass is only a
//!      legitimate reading of "what the plain path would have computed" if batching
//!      changed nothing numerically; the banked warning is that a cuBLASLt m-order shift
//!      moves logits 0.18-3.08, which would break identity at near-ties while looking
//!      like a port bug. This gate therefore compares, from the SAME warmed cache state:
//!        - the batched round's per-position logits vs T sequential single-position
//!          decode steps' logits, as RAW BITS (not a tolerance);
//!        - every live trunk cache class after `commit(n)` vs plain sequential decode of
//!          exactly the n committed tokens, as RAW BITS (the §3.1 invariant, device twin
//!          of the CPU-oracle gate that ratified the mechanism).
//!      Device==device is legitimate HERE precisely because both sides are the same
//!      realization and the question is machinery, not numerics.
//!      Swept over every compressor phase (pos0 mod ratio) and over partial/full accept.
//!
//!  (d) ACCEPTED-POSITION RING WRITES — arm DB's drafter main_kv rings must end
//!      bit-identical to a plain greedy run that wrote a ring row at EVERY decoded
//!      position. That is the banked trap: the reference writes one row per step only
//!      because its smoke drafts every step; a verifying engine owes one row per
//!      ACCEPTED position and none for a rejected draft.
//!
//!  (e) determinism x runs on the batched arm (tokens + per-round accept digest).
//!
//! Agreement with the banked torch REF trajectory is REPORTED, not gated: that comparison
//! belongs to the teacher-forcing gate, and a realization flip vs torch is a known
//! in-band class (lane-6 policy; the banked 0731 decode gate records an in-band argmax
//! flip at s=54 in all three dots arms).
//!
//! Acceptance statistics print under a no-speed-claim banner: an accept-length is a
//! CORRECTNESS observable only (dspark-q38 lesson — 2.88 accept-length measured 1.00x
//! wall). No wall-clock number is emitted here at all.
//!
//! Requires MEMRA_DSV4_DRAFTER=dspark and MEMRA_DSV4_DECODE_PATH=device.
//!
//! Usage: dsv4-gpu-dspark-gate <model-dir> <fixtures.json> <out-dir> [runs] [dev0,dev1]

use memra_engine::dsv4_gpu::Dsv4Gpu;
use memra_gguf::dsv4_dspark::DsparkFixtureSpec;
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

fn sha_f32(v: &[f32]) -> String {
    let mut bytes = Vec::with_capacity(v.len() * 4);
    for x in v {
        bytes.extend_from_slice(&x.to_bits().to_le_bytes());
    }
    sha256_hex(&bytes)
}

fn vram_line(gpu: &Dsv4Gpu, tag: &str) {
    if let Ok(rows) = gpu.vram_report() {
        let line = rows
            .iter()
            .map(|(dev, free, total, resident)| {
                format!(
                    "dev{dev} used {:.2}/{:.2} GiB (resident {:.2})",
                    (*total - *free) as f64 / 2f64.powi(30),
                    *total as f64 / 2f64.powi(30),
                    *resident as f64 / 2f64.powi(30)
                )
            })
            .collect::<Vec<_>>()
            .join(" | ");
        println!("[vram {tag}] {line}");
    }
}

/// Arm P: plain device greedy from `prompt`, `n_new` tokens.
fn run_plain(gpu: &Dsv4Gpu, prompt: &[u32], n_new: usize) -> Vec<u32> {
    let mut state = gpu.alloc_decode_state().expect("alloc decode state");
    let pre = gpu
        .prefill_with_cache(prompt, &mut state)
        .expect("plain prefill");
    let mut t = argmax(&pre.logits);
    let mut tokens = Vec::with_capacity(n_new);
    for step in 0..n_new {
        tokens.push(t);
        if step + 1 == n_new {
            break;
        }
        t = gpu.decode_step_greedy(t, &mut state).expect("plain decode");
    }
    tokens
}

/// Arm P + a drafter ring write at EVERY decoded position — the reference the batched
/// arm's accepted-position rule must reproduce bit for bit (verdict (d)).
fn run_plain_with_rings(gpu: &Dsv4Gpu, prompt: &[u32], n_new: usize) -> Vec<(String, Vec<f32>)> {
    let p0 = prompt.len();
    let mut state = gpu.alloc_decode_state().expect("alloc decode state");
    let mut dstate = gpu.dspark_alloc_state().expect("alloc dspark state");
    let pre = gpu
        .dspark_prefill_prime(prompt, &mut state, &mut dstate)
        .expect("prefill + prime");
    let mut t = argmax(&pre.logits);
    for step in 0..n_new {
        if step + 1 == n_new {
            break;
        }
        let m = p0 + step;
        t = gpu
            .decode_step_greedy_tap(t, &mut state, &mut dstate, 0)
            .expect("plain tap decode");
        gpu.dspark_write_rings(&mut dstate, 0, m)
            .expect("plain ring write");
    }
    gpu.dspark_ring_classes(&dstate).expect("ring classes")
}

struct DraftedOut {
    tokens: Vec<u32>,
    rounds: usize,
    accepted: usize,
    verified: usize,
    /// per-round accepted counts, digested for cross-run determinism
    accept_sha: String,
    /// batched arm only: mean forwarded depth per round (the perf-model observable)
    mean_t_batch: f64,
    rings: Option<Vec<(String, Vec<f32>)>>,
}

/// Arm DS: the device propose-then-verify greedy loop with SEQUENTIAL verification.
/// Mirrors `spec_oracle::run_spec_greedy` step for step.
fn run_drafted_seq(gpu: &Dsv4Gpu, prompt: &[u32], n_new: usize) -> DraftedOut {
    let p0 = prompt.len();
    let mut state = gpu.alloc_decode_state().expect("alloc decode state");
    let mut dstate = gpu.dspark_alloc_state().expect("alloc dspark state");
    let pre = gpu
        .dspark_prefill_prime(prompt, &mut state, &mut dstate)
        .expect("drafted prefill + prime");
    let mut t = argmax(&pre.logits);
    let mut tokens: Vec<u32> = Vec::with_capacity(n_new);
    let mut pending: std::collections::VecDeque<u32> = Default::default();
    let mut rounds = 0usize;
    let mut accepted = 0usize;
    let mut verified = 0usize;
    let mut accept_bytes: Vec<u8> = Vec::new();
    let mut cur_accepts = 0u32;
    let mut open_round = false;
    for step in 0..n_new {
        let m = p0 + step; // t sits at index m
        if pending.is_empty() {
            if open_round {
                accept_bytes.extend_from_slice(&cur_accepts.to_le_bytes());
            }
            let prop = gpu
                .dspark_forward_spec(&mut dstate, t, 0, m - 1, false)
                .expect("dspark propose");
            pending = prop.out_ids[1..].iter().cloned().collect();
            rounds += 1;
            cur_accepts = 0;
            open_round = true;
        }
        tokens.push(t);
        if step + 1 == n_new {
            break;
        }
        let t_next = gpu
            .decode_step_greedy_tap(t, &mut state, &mut dstate, 0)
            .expect("drafted verify step");
        gpu.dspark_write_rings(&mut dstate, 0, m)
            .expect("dspark ring write");
        let d = pending.pop_front().expect("pending nonempty");
        verified += 1;
        if d == t_next {
            accepted += 1;
            cur_accepts += 1;
        } else {
            pending.clear();
        }
        t = t_next;
    }
    if open_round {
        accept_bytes.extend_from_slice(&cur_accepts.to_le_bytes());
    }
    DraftedOut {
        tokens,
        rounds,
        accepted,
        verified,
        accept_sha: sha256_hex(&accept_bytes),
        mean_t_batch: 1.0,
        rings: None,
    }
}

/// Arm DB: the batched T=k+1 device verify loop (`spec_greedy_batched_with`).
fn run_drafted_batched(gpu: &Dsv4Gpu, prompt: &[u32], n_new: usize) -> DraftedOut {
    let mut state = gpu.alloc_decode_state().expect("alloc decode state");
    let mut dstate = gpu.dspark_alloc_state().expect("alloc dspark state");
    let mut vstate = gpu.alloc_verify_state().expect("alloc verify state");
    let out = gpu
        .spec_greedy_batched_with(prompt, n_new, &mut state, &mut dstate, &mut vstate)
        .expect("batched drafted run");
    let mut accept_bytes: Vec<u8> = Vec::new();
    let mut accepted = 0usize;
    let mut verified = 0usize;
    let mut t_sum = 0usize;
    let mut t_n = 0usize;
    for r in &out.rounds {
        accept_bytes.extend_from_slice(&(r.accepts as u32).to_le_bytes());
        accepted += r.accepts;
        verified += r.verified;
        if r.t_batch > 0 {
            t_sum += r.t_batch;
            t_n += 1;
        }
    }
    let rings = gpu.dspark_ring_classes(&dstate).expect("ring classes");
    DraftedOut {
        tokens: out.tokens,
        rounds: out.rounds.len(),
        accepted,
        verified,
        accept_sha: sha256_hex(&accept_bytes),
        mean_t_batch: if t_n > 0 {
            t_sum as f64 / t_n as f64
        } else {
            0.0
        },
        rings: Some(rings),
    }
}

/// Verdict (c): batched round == T sequential steps, BIT for BIT (logits AND every live
/// trunk cache class after the commit). One cell = one (warm, t_batch, n_commit) triple.
fn gate_bit_equal(
    gpu: &Dsv4Gpu,
    prompt: &[u32],
    drafts: &[u32],
    warm: usize,
    t_batch: usize,
    n_commit: usize,
) -> (Vec<String>, usize, usize) {
    assert!(n_commit >= 1 && n_commit <= t_batch);
    let tag = format!("warm={warm} T={t_batch} commit={n_commit}");
    let mut fails = Vec::new();

    // --- warm a cache state, deterministically, and learn the round's real head token
    let warm_up = |gpu: &Dsv4Gpu| -> (memra_engine::dsv4_gpu::DecodeState, u32) {
        let mut state = gpu.alloc_decode_state().expect("alloc decode state");
        let pre = gpu
            .prefill_with_cache(prompt, &mut state)
            .expect("bitgate prefill");
        let mut t = argmax(&pre.logits);
        for _ in 0..warm {
            t = gpu
                .decode_step_greedy(t, &mut state)
                .expect("bitgate warm step");
        }
        (state, t)
    };

    // round ids: the real next token, then arbitrary-but-fixed drafts
    let (mut state_a, t0) = warm_up(gpu);
    let mut ids = vec![t0];
    ids.extend_from_slice(&drafts[..t_batch - 1]);

    let mut vstate = gpu.alloc_verify_state().expect("alloc verify state");
    let (logits_b, _am) = gpu
        .verify_batch_dev(&ids, &mut state_a, &mut vstate, None, true)
        .expect("batched verify");
    let logits_b = logits_b.expect("want_logits");
    gpu.commit_verify_dev(&mut state_a, &mut vstate, n_commit)
        .expect("commit");
    let classes_batch = gpu.cache_classes(&state_a).expect("classes batch");
    drop(state_a);

    // --- sequential twin
    let (mut state_b, t0b) = warm_up(gpu);
    if t0b != t0 {
        fails.push(format!(
            "[{tag}] WARM NONDETERMINISM: head token {t0b} != {t0} — the gate's premise \
             (two identically warmed states) is broken"
        ));
    }
    let mut classes_seq: Option<Vec<(String, Vec<f32>)>> = None;
    let mut logit_pairs = 0usize;
    for i in 0..t_batch {
        let row = gpu
            .decode_step(ids[i], &mut state_b)
            .expect("sequential step");
        let vocab = row.len();
        let brow = &logits_b[i * vocab..(i + 1) * vocab];
        let mut diffs = 0usize;
        let mut first: Option<(usize, f32, f32)> = None;
        let mut worst = 0f32;
        for j in 0..vocab {
            if brow[j].to_bits() != row[j].to_bits() {
                diffs += 1;
                worst = worst.max((brow[j] - row[j]).abs());
                if first.is_none() {
                    first = Some((j, brow[j], row[j]));
                }
            }
        }
        logit_pairs += 1;
        if diffs != 0 {
            let (j, b, s) = first.unwrap();
            fails.push(format!(
                "[{tag}] LOGIT BIT FAIL row {i} (pos {}): {diffs}/{vocab} elements differ, \
                 first at vocab id {j} (batched {b:.9e} vs sequential {s:.9e}), worst |Δ| \
                 {worst:.3e}",
                i
            ));
        }
        if i + 1 == n_commit {
            classes_seq = Some(gpu.cache_classes(&state_b).expect("classes seq"));
        }
    }
    let classes_seq = classes_seq.expect("commit point inside the round");
    drop(state_b);

    // --- §3.1 state invariant, per class, bit for bit
    if classes_batch.len() != classes_seq.len() {
        fails.push(format!(
            "[{tag}] CLASS COUNT MISMATCH: batched {} vs sequential {}",
            classes_batch.len(),
            classes_seq.len()
        ));
    }
    let mut class_checks = 0usize;
    for ((nb, vb), (ns, vs)) in classes_batch.iter().zip(&classes_seq) {
        class_checks += 1;
        if nb != ns {
            fails.push(format!("[{tag}] CLASS NAME MISMATCH: {nb} vs {ns}"));
            continue;
        }
        if vb.len() != vs.len() {
            fails.push(format!(
                "[{tag}] CLASS LEN FAIL {nb}: batched {} vs sequential {}",
                vb.len(),
                vs.len()
            ));
            continue;
        }
        let mut diffs = 0usize;
        let mut worst = 0f32;
        for (a, b) in vb.iter().zip(vs.iter()) {
            if a.to_bits() != b.to_bits() {
                diffs += 1;
                worst = worst.max((a - b).abs());
            }
        }
        if diffs != 0 {
            fails.push(format!(
                "[{tag}] STATE BIT FAIL class {nb}: {diffs}/{} elements differ, worst |Δ| \
                 {worst:.3e}",
                vb.len()
            ));
        }
    }
    (fails, logit_pairs, class_checks)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!(
            "usage: dsv4-gpu-dspark-gate <model-dir> <fixtures.json> <out-dir> [runs] [dev0,dev1]"
        );
        std::process::exit(2);
    }
    let t0 = std::time::Instant::now();
    let dir = Path::new(&args[1]);
    let spec = DsparkFixtureSpec::load(Path::new(&args[2]));
    let out_dir = Path::new(&args[3]);
    std::fs::create_dir_all(out_dir).expect("mkdir out");
    let runs: usize = args.get(4).map(|x| x.parse().expect("runs")).unwrap_or(2);
    let devices: Vec<usize> = args
        .get(5)
        .map(|s| s.split(',').map(|x| x.parse().expect("device")).collect())
        .unwrap_or_else(|| vec![0, 1]);

    if std::env::var("MEMRA_DSV4_DRAFTER").as_deref() != Ok("dspark") {
        eprintln!("REFUSE: this gate requires MEMRA_DSV4_DRAFTER=dspark");
        std::process::exit(2);
    }
    if std::env::var("MEMRA_DSV4_DECODE_PATH").as_deref() != Ok("device") {
        eprintln!("REFUSE: this gate requires MEMRA_DSV4_DECODE_PATH=device");
        std::process::exit(2);
    }
    // rung-4 sub-gates can be skipped only for machinery feedback, never for a verdict
    let bitgate_only = std::env::var("MEMRA_DSV4_GATE_BITONLY").is_ok();

    let prompt = spec.prompt.clone();
    let p0 = prompt.len();
    let n_new = spec.tokens.len();
    let banked: Vec<u32> = spec.tokens.clone();
    println!(
        "dsv4-gpu-dspark-gate | model {} | variant {} | p0 {p0} | n_new {n_new} | runs {runs} \
         | devices {devices:?} | torch accept_mean {:.4}{}",
        dir.display(),
        spec.variant_tag,
        spec.accept_mean,
        if bitgate_only {
            " | *** MEMRA_DSV4_GATE_BITONLY: machinery feedback, NOT a verdict ***"
        } else {
            ""
        }
    );

    let variant = match spec.variant_tag.as_str() {
        "ref" => ActQuantVariant::RefFp8Round,
        other => panic!("the spec==plain identity gate runs the REF contract, got {other:?}"),
    };
    let max_seq = (p0 + n_new + 32).max(256);
    let gpu = Dsv4Gpu::load(dir, &devices, variant, max_seq).expect("load");
    println!(
        "loaded: split at layer {}, verify tmax {}, t={:.0}s",
        gpu.split_at,
        gpu.verify_tmax(),
        t0.elapsed().as_secs_f64()
    );
    vram_line(&gpu, "post-load");

    let mut fails: Vec<String> = Vec::new();

    // ---- verdict (c): batched == sequential, bit for bit. Run FIRST: it is the cheapest
    // decisive instrument, and if batching is not bit-exact then (b) cannot be read at all.
    let tmax = gpu.verify_tmax();
    let mut bit_cells = 0usize;
    let mut bit_logit_rows = 0usize;
    let mut bit_class_checks = 0usize;
    {
        let drafts: Vec<u32> = banked.iter().cloned().take(tmax).collect();
        // sweep every compressor phase (fine ratio 4) and partial/full accept
        let mut cells: Vec<(usize, usize, usize)> = Vec::new();
        for warm in 1..=4usize {
            for &nc in &[1usize, tmax - 1, tmax] {
                cells.push((warm, tmax, nc));
            }
        }
        cells.push((1, 2, 1)); // minimum-width round (T=2), partial accept
        cells.push((3, 3, 2));
        for (warm, tb, nc) in cells {
            let nc = nc.min(tb);
            let (f, lp, cc) = gate_bit_equal(&gpu, &prompt, &drafts, warm, tb, nc);
            bit_cells += 1;
            bit_logit_rows += lp;
            bit_class_checks += cc;
            let ok = f.is_empty();
            println!(
                "bit-gate cell warm={warm} T={tb} commit={nc}: {} ({lp} logit rows, {cc} \
                 cache classes) t={:.0}s",
                if ok { "BIT-IDENTICAL" } else { "FAIL" },
                t0.elapsed().as_secs_f64()
            );
            fails.extend(f);
        }
    }
    println!(
        "verdict (c) batched==sequential: {bit_cells} cells, {bit_logit_rows} logit rows, \
         {bit_class_checks} cache-class comparisons — {}",
        if fails.is_empty() {
            "ALL BIT-IDENTICAL"
        } else {
            "FAILURES PRESENT"
        }
    );
    vram_line(&gpu, "post-bitgate");

    if bitgate_only {
        println!(
            "\n*** MEMRA_DSV4_GATE_BITONLY set: verdicts (b)/(d)/(e) NOT run — machinery \
             feedback only, never a verdict ***"
        );
        if fails.is_empty() {
            println!("bit-gate: [PASS] (partial run, no verdict)");
        } else {
            println!("bit-gate: [FAIL] {} finding(s)", fails.len());
            for l in &fails {
                println!("  {l}");
            }
            std::process::exit(1);
        }
        return;
    }

    // ---- arm P
    let plain = run_plain(&gpu, &prompt, n_new);
    println!(
        "arm P (plain device greedy): {} tokens t={:.0}s",
        plain.len(),
        t0.elapsed().as_secs_f64()
    );

    // ---- arm P + per-position ring writes (verdict (d) reference)
    let plain_rings = run_plain_with_rings(&gpu, &prompt, n_new);
    println!(
        "arm P' (plain + a ring write at every position): {} ring classes t={:.0}s",
        plain_rings.len(),
        t0.elapsed().as_secs_f64()
    );

    // ---- arm DS (sequential verify) x1: rung 3 stays gated in the same process
    let ds = run_drafted_seq(&gpu, &prompt, n_new);
    println!(
        "arm DS (drafted, sequential verify): {} tokens | rounds {} | accepted {} (mean \
         {:.4}/round) | verified {} | accept sha {} | t={:.0}s",
        ds.tokens.len(),
        ds.rounds,
        ds.accepted,
        ds.accepted as f64 / ds.rounds as f64,
        ds.verified,
        ds.accept_sha,
        t0.elapsed().as_secs_f64()
    );

    // ---- arm DB (batched verify) x runs
    let mut db_runs: Vec<DraftedOut> = Vec::new();
    for r in 0..runs {
        let d = run_drafted_batched(&gpu, &prompt, n_new);
        println!(
            "arm DB run{r} (drafted, BATCHED T=k+1 verify): {} tokens | rounds {} | accepted \
             {} (mean {:.4}/round) | verified {} | mean T forwarded {:.4} | accept sha {} | \
             t={:.0}s",
            d.tokens.len(),
            d.rounds,
            d.accepted,
            d.accepted as f64 / d.rounds as f64,
            d.verified,
            d.mean_t_batch,
            d.accept_sha,
            t0.elapsed().as_secs_f64()
        );
        db_runs.push(d);
    }
    vram_line(&gpu, "post-drafted");

    // ---- verdict (b): greedy spec == plain, byte-exact, both drafted arms
    let mut check_identity = |name: &str, toks: &[u32]| {
        if toks != plain.as_slice() {
            let first = toks
                .iter()
                .zip(&plain)
                .position(|(a, b)| a != b)
                .unwrap_or(plain.len().min(toks.len()));
            fails.push(format!(
                "IDENTITY FAIL ({name}): drafted != plain, first divergence at generated \
                 index {first} (drafted {:?} vs plain {:?})",
                toks.get(first),
                plain.get(first)
            ));
        }
    };
    check_identity("arm DS", &ds.tokens);
    for (r, d) in db_runs.iter().enumerate() {
        check_identity(&format!("arm DB run{r}"), &d.tokens);
    }

    // ---- verdict (d): accepted-position ring writes
    {
        let db_rings = db_runs[0].rings.as_ref().expect("DB rings");
        if db_rings.len() != plain_rings.len() {
            fails.push(format!(
                "RING CLASS COUNT MISMATCH: DB {} vs plain' {}",
                db_rings.len(),
                plain_rings.len()
            ));
        }
        let mut ring_fails = 0usize;
        for ((nb, vb), (ns, vs)) in db_rings.iter().zip(&plain_rings) {
            if nb != ns || vb.len() != vs.len() {
                fails.push(format!("RING SHAPE MISMATCH: {nb} vs {ns}"));
                ring_fails += 1;
                continue;
            }
            let diffs = vb
                .iter()
                .zip(vs.iter())
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            if diffs != 0 {
                let worst = vb
                    .iter()
                    .zip(vs.iter())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0f32, f32::max);
                fails.push(format!(
                    "RING BIT FAIL {nb}: {diffs}/{} elements differ, worst |Δ| {worst:.3e} — \
                     the accepted-position ring-write rule is broken",
                    vb.len()
                ));
                ring_fails += 1;
            }
        }
        println!(
            "verdict (d) accepted-position ring writes: {} ring classes compared, {} \
             mismatching",
            db_rings.len(),
            ring_fails
        );
    }

    // ---- verdict (e): determinism across batched runs
    if runs > 1 {
        let same_tokens = db_runs.windows(2).all(|w| w[0].tokens == w[1].tokens);
        let same_accept = db_runs
            .windows(2)
            .all(|w| w[0].accept_sha == w[1].accept_sha);
        let same_rings = db_runs.windows(2).all(|w| {
            let a = w[0].rings.as_ref().expect("rings");
            let b = w[1].rings.as_ref().expect("rings");
            a.len() == b.len()
                && a.iter().zip(b).all(|((_, x), (_, y))| {
                    x.len() == y.len()
                        && x.iter()
                            .zip(y.iter())
                            .all(|(p, q)| p.to_bits() == q.to_bits())
                })
        });
        println!(
            "determinism across {runs} batched runs: tokens {} | per-round accepts {} | \
             drafter rings {}",
            if same_tokens {
                "IDENTICAL"
            } else {
                "DIVERGENT"
            },
            if same_accept {
                "BYTE-IDENTICAL"
            } else {
                "DIVERGENT"
            },
            if same_rings {
                "BIT-IDENTICAL"
            } else {
                "DIVERGENT"
            }
        );
        if !same_tokens {
            fails.push("DETERMINISM FAIL: batched token streams differ across runs".into());
        }
        if !same_accept {
            fails.push("DETERMINISM FAIL: per-round accept counts differ across runs".into());
        }
        if !same_rings {
            fails.push("DETERMINISM FAIL: drafter rings differ across runs".into());
        }
    }

    // ---- REPORTED (not gated)
    let agree = plain
        .iter()
        .zip(&banked)
        .take_while(|(a, b)| a == b)
        .count();
    let total_agree = plain.iter().zip(&banked).filter(|(a, b)| a == b).count();
    println!(
        "[REPORTED, not gated] arm P vs banked torch REF trajectory: {total_agree}/{n_new} \
         positions equal, common prefix {agree} (torch-vs-GPU realization flips are the \
         teacher-forcing gate's business, lane-6 in-band policy; the banked 0731 decode gate \
         records an in-band argmax flip at s=54 in all three dots arms)"
    );

    let d0 = &db_runs[0];
    println!(
        "\n*** ACCEPTANCE IS A CORRECTNESS OBSERVABLE ONLY — NOT A SPEED CLAIM ***\n\
         batched arm: rounds {} | accepted {} | mean accepted/round {:.4} | mean T forwarded \
         {:.4} | tokens per round {:.4} | torch teacher-forced accept_mean {:.4} (different \
         loop: teacher-forced vs free-running, so these are not required to match)",
        d0.rounds,
        d0.accepted,
        d0.accepted as f64 / d0.rounds as f64,
        d0.mean_t_batch,
        d0.tokens.len() as f64 / d0.rounds as f64,
        spec.accept_mean
    );

    let plain_sha = sha256_hex(
        &plain
            .iter()
            .flat_map(|t| t.to_le_bytes())
            .collect::<Vec<u8>>(),
    );
    let ring_sha = sha_f32(
        &d0.rings
            .as_ref()
            .expect("rings")
            .iter()
            .flat_map(|(_, v)| v.iter().cloned())
            .collect::<Vec<f32>>(),
    );
    let mut f = std::fs::File::create(out_dir.join("gpu_dspark_gate.json")).expect("json");
    write!(
        f,
        "{{\n  \"variant\": \"{}\",\n  \"p0\": {p0},\n  \"n_new\": {n_new},\n  \
         \"plain_tokens\": [{}],\n  \"drafted_batched_tokens\": [{}],\n  \
         \"plain_sha256\": \"{}\",\n  \"rounds\": {},\n  \"accepted\": {},\n  \
         \"verified\": {},\n  \"mean_t_batch\": {:.6},\n  \"accept_sha256\": \"{}\",\n  \
         \"drafter_ring_sha256\": \"{}\",\n  \"seq_arm_rounds\": {},\n  \
         \"seq_arm_accepted\": {},\n  \"bitgate_cells\": {bit_cells},\n  \
         \"bitgate_logit_rows\": {bit_logit_rows},\n  \
         \"bitgate_class_checks\": {bit_class_checks},\n  \"banked_agree\": {total_agree},\n  \
         \"identity_pass\": {},\n  \"fails\": {}\n}}\n",
        spec.variant_tag,
        plain
            .iter()
            .map(|t| t.to_string())
            .collect::<Vec<_>>()
            .join(","),
        d0.tokens
            .iter()
            .map(|t| t.to_string())
            .collect::<Vec<_>>()
            .join(","),
        plain_sha,
        d0.rounds,
        d0.accepted,
        d0.verified,
        d0.mean_t_batch,
        d0.accept_sha,
        ring_sha,
        ds.rounds,
        ds.accepted,
        fails.is_empty(),
        fails.len()
    )
    .expect("write json");

    if fails.is_empty() {
        println!(
            "\nGPU DSPARK GATE [PASS]\n  (b) greedy spec==plain: {n_new}/{n_new} tokens LITERAL \
             identity, sequential AND batched arms\n  (c) batched==sequential: {bit_cells} \
             cells / {bit_logit_rows} logit rows / {bit_class_checks} cache classes, BIT for \
             BIT\n  (d) accepted-position drafter ring writes: BIT-identical to \
             per-position plain\n  (e) determinism x{runs}\n  banked {}",
            out_dir.join("gpu_dspark_gate.json").display()
        );
    } else {
        println!("\nGPU DSPARK GATE: [FAIL] {} finding(s)", fails.len());
        for l in &fails {
            println!("  {l}");
        }
        std::process::exit(1);
    }
    println!("total elapsed {:.0}s", t0.elapsed().as_secs_f64());
}
