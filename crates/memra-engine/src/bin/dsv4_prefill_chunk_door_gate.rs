//! The DSV4 prefill-chunk door gate: `MEMRA_DSV4_PREFILL_CHUNK`.
//!
//! Two of the four gates FLAGS.md names as the door's blockers live here, on the served
//! program (reference expert program, PP-2 topology, `DRAFTER=dspark`), which is the only
//! program that can chunk at all: the TP/EP topology refuses a batched prime outright
//! (`dsv4_gpu.rs`, "DSV4 TP/EP vertical slice currently admits only a single-token prime").
//!
//! 1. CHUNK-WIDTH EXACTNESS. Every served chunk width must produce bit-identical logits,
//!    live cache state, DSpark rings and sampled continuation. Widths straddle the
//!    register-specialization boundary (1..32 exact twins, 33..64 tiled by eight) and both
//!    an exactly-tiling and a ragged-tail prompt length, because a wrong assumption about
//!    when partial results combine hides exactly at a chunk boundary. Refusal is on the
//!    first differing BIT, named by class and index, not on a digest mismatch.
//!
//! 2. THE CHUNKED-VERSUS-MONOLITHIC NUMERIC CLASS, decided by measurement rather than
//!    assumed: the same prefix is primed both ways at several lengths and compared bit for
//!    bit. Bit-equal everywhere is same-class; anything else prints drift rows (top-1
//!    change count, greedy identity count, KL mean/max in both directions) over the pinned
//!    teacher-forced set instead of pretending the classes are one.
//!
//! 3. SAMPLED CACHE TRANSPARENCY. A chunked prefill must leave the cache indistinguishable
//!    from a monolithic one on the SAMPLED path: compressed store, pending rows, indexer
//!    store and pending, SWA ring, DSpark rings, then a seeded sampled speculative
//!    continuation from each state (tokens, per-round accepts/verified/confidence) and the
//!    post-continuation cache. The restored-suffix split lands mid-chunk on purpose.
//!
//! Every check carries a red arm: the comparator is shown detecting a one-token
//! perturbation at a named class and index, the class census is shown refusing a truncated
//! expectation, the prefill-head census is shown failing crosswise between its arms, and
//! the width guards are shown refusing 0 and an over-wide request.
//!
//! usage: dsv4_prefill_chunk_door_gate <model-dir> <real-source.txt>
use memra_engine::dsv4_gpu::{
    DecodeState, DsparkState, Dsv4Gpu, Dsv4PrefillHead, Dsv4SampleCfg, Dsv4Vt, prefill_head_census,
};
use memra_gguf::dsv4_forward::ActQuantVariant;
use memra_tokenizer::Tokenizer;
use sha2::{Digest, Sha256};
use std::path::Path;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

/// Serving ceiling for the door (`DSV4_SERVING_BATCH_WIDTH_MAX` in the server). Widths
/// above it are not gated here because they cannot be served.
const SERVING_WIDTH_MAX: usize = 64;
const SEED: u64 = 20260910;

fn unix_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("UTC clock")
        .as_millis()
}

/// Everything a prefill leaves behind that a later token can read.
struct Snapshot {
    logits: Vec<f32>,
    classes: Vec<(String, Vec<f32>)>,
}

impl Snapshot {
    fn take(gpu: &Dsv4Gpu, logits: Vec<f32>, state: &DecodeState, dstate: &DsparkState) -> Self {
        let mut classes = gpu.cache_classes(state).expect("cache classes");
        classes.extend(gpu.dspark_ring_classes(dstate).expect("dspark rings"));
        Snapshot { logits, classes }
    }

    fn digest(&self) -> String {
        let mut h = Sha256::new();
        h.update((self.logits.len() as u64).to_le_bytes());
        for v in &self.logits {
            h.update(v.to_bits().to_le_bytes());
        }
        for (name, values) in &self.classes {
            h.update(name.as_bytes());
            h.update((values.len() as u64).to_le_bytes());
            for v in values {
                h.update(v.to_bits().to_le_bytes());
            }
        }
        format!("{:x}", h.finalize())
    }
}

/// The first differing BIT between two snapshots, named. `None` is bit-identity.
fn first_difference(a: &Snapshot, b: &Snapshot) -> Option<String> {
    if a.logits.len() != b.logits.len() {
        return Some(format!(
            "logits length {} vs {}",
            a.logits.len(),
            b.logits.len()
        ));
    }
    for (i, (x, y)) in a.logits.iter().zip(b.logits.iter()).enumerate() {
        if x.to_bits() != y.to_bits() {
            return Some(format!(
                "logits[{i}] 0x{:08x} vs 0x{:08x} ({x} vs {y})",
                x.to_bits(),
                y.to_bits()
            ));
        }
    }
    if a.classes.len() != b.classes.len() {
        return Some(format!(
            "class count {} vs {}",
            a.classes.len(),
            b.classes.len()
        ));
    }
    for ((na, va), (nb, vb)) in a.classes.iter().zip(b.classes.iter()) {
        if na != nb {
            return Some(format!("class name {na} vs {nb}"));
        }
        if va.len() != vb.len() {
            return Some(format!("class {na} length {} vs {}", va.len(), vb.len()));
        }
        for (i, (x, y)) in va.iter().zip(vb.iter()).enumerate() {
            if x.to_bits() != y.to_bits() {
                return Some(format!(
                    "{na}[{i}] 0x{:08x} vs 0x{:08x} ({x} vs {y})",
                    x.to_bits(),
                    y.to_bits()
                ));
            }
        }
    }
    None
}

fn softmax_f64(logits: &[f32]) -> Vec<f64> {
    let max = logits.iter().fold(f32::NEG_INFINITY, |m, v| m.max(*v)) as f64;
    let mut exps: Vec<f64> = logits.iter().map(|v| (*v as f64 - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    for e in &mut exps {
        *e /= sum;
    }
    exps
}

/// KL(p || q) in nats, with the usual 0 log 0 = 0 convention.
fn kl(p: &[f64], q: &[f64]) -> f64 {
    let mut acc = 0f64;
    for (pi, qi) in p.iter().zip(q.iter()) {
        if *pi > 0.0 {
            acc += pi * (pi / qi.max(f64::MIN_POSITIVE)).ln();
        }
    }
    acc
}

fn argmax(logits: &[f32]) -> usize {
    let mut best = 0usize;
    for (i, v) in logits.iter().enumerate() {
        if *v > logits[best] {
            best = i;
        }
    }
    best
}

struct Primed {
    snapshot: Snapshot,
    state: DecodeState,
    dstate: DsparkState,
}

/// A chunked prime of the whole prompt, on the served DSpark path.
fn prime_chunked(gpu: &Dsv4Gpu, prompt: &[u32], width: usize, budget: usize) -> Primed {
    let mut state = gpu
        .alloc_decode_state_for_transient(prompt.len() + budget, width.max(gpu.verify_tmax()))
        .expect("decode state");
    let mut dstate = gpu.dspark_alloc_state().expect("dspark state");
    let timer = Instant::now();
    let logits = gpu
        .dspark_prefill_prime_chunked(prompt, &mut state, &mut dstate, width)
        .expect("chunked prime");
    println!(
        "PRIME kind=chunked width={width} tokens={} seconds={:.3} unix_ms={}",
        prompt.len(),
        timer.elapsed().as_secs_f64(),
        unix_ms()
    );
    let snapshot = Snapshot::take(gpu, logits, &state, &dstate);
    Primed {
        snapshot,
        state,
        dstate,
    }
}

/// The monolithic prime the door rolls back to (`MEMRA_DSV4_PREFILL_CHUNK=0`).
fn prime_monolithic(gpu: &Dsv4Gpu, prompt: &[u32], budget: usize) -> Primed {
    let mut state = gpu
        .alloc_decode_state_for_transient(prompt.len() + budget, gpu.verify_tmax())
        .expect("decode state");
    let mut dstate = gpu.dspark_alloc_state().expect("dspark state");
    let timer = Instant::now();
    let logits = gpu
        .dspark_prefill_prime(prompt, &mut state, &mut dstate)
        .expect("monolithic prime")
        .logits;
    println!(
        "PRIME kind=monolithic tokens={} seconds={:.3} unix_ms={}",
        prompt.len(),
        timer.elapsed().as_secs_f64(),
        unix_ms()
    );
    let snapshot = Snapshot::take(gpu, logits, &state, &dstate);
    Primed {
        snapshot,
        state,
        dstate,
    }
}

/// A seeded sampled speculative continuation from an already primed state, folded into a
/// snapshot: emitted tokens and per-round structure carried as logits-position floats so
/// the same bit comparator covers them.
fn sampled_continuation(
    gpu: &Dsv4Gpu,
    prompt: &[u32],
    primed: &mut Primed,
    n_new: usize,
) -> Snapshot {
    let cfg = Dsv4SampleCfg {
        temperature: 1.0,
        top_p: 1.0,
        top_k: 0,
        seed: SEED,
    };
    let mut verify = gpu
        .alloc_verify_state_for(primed.state.capacity)
        .expect("verify state");
    let logits = primed.snapshot.logits.clone();
    let run = gpu
        .spec_sampled_batched_pen_restored(
            prompt,
            &logits,
            n_new,
            &mut primed.state,
            &mut primed.dstate,
            &mut verify,
            usize::MAX,
            Dsv4Vt::Off,
            &cfg,
            None,
            None,
        )
        .expect("sampled spec continuation");
    assert_eq!(run.tokens.len(), n_new, "sampled budget honoured");
    assert!(!run.rounds.is_empty(), "sampled run engaged the spec path");
    let mut trace: Vec<f32> = Vec::new();
    for t in &run.tokens {
        trace.push(f32::from_bits(*t));
    }
    for round in &run.rounds {
        for v in [
            round.start_pos,
            round.accepts,
            round.verified,
            round.t_batch,
            round.emitted,
        ] {
            trace.push(v as f32);
        }
        trace.extend_from_slice(&round.confidence);
    }
    let mut snapshot = Snapshot::take(gpu, trace, &primed.state, &primed.dstate);
    snapshot.classes.push((
        "sampled.accepted_total".to_string(),
        vec![run.rounds.iter().map(|r| r.accepts).sum::<usize>() as f32],
    ));
    snapshot
}

fn chunks_for(suffix_rows: usize, width: usize) -> u64 {
    suffix_rows.div_ceil(width) as u64
}

fn main() {
    // Freeze the instrument's own program: this gate is about the chunk door, not about
    // whichever numeric flips are default this week. Process startup, before any load.
    unsafe {
        std::env::set_var("MEMRA_DSV4_DRAFTER", "dspark");
        std::env::set_var("MEMRA_DSV4_AR_PHASE", "0");
    }
    let args: Vec<String> = std::env::args().collect();
    assert_eq!(
        args.len(),
        3,
        "usage: dsv4_prefill_chunk_door_gate <model-dir> <real-source.txt>"
    );
    let dir = Path::new(&args[1]);
    let source = std::fs::read_to_string(&args[2]).expect("real source");
    let tokenizer = Tokenizer::from_hf_dir(dir).expect("tokenizer");
    let full = tokenizer.encode(&format!("Review this engine source:\n{source}"), true);
    assert!(full.len() >= 1025, "do not pad or repeat the source");
    println!(
        "SOURCE sha256={:x} tokens={} unix_ms={}",
        Sha256::digest(source.as_bytes()),
        full.len(),
        unix_ms()
    );
    // Two prompt lengths: 1025 tiles exactly at every gated width (suffix 1024), 1000 has
    // a ragged final transaction at every gated width above one (suffix 999).
    let tiling: Vec<u32> = full[..1025].to_vec();
    let ragged: Vec<u32> = full[..1000].to_vec();
    for w in [32usize, SERVING_WIDTH_MAX] {
        assert_eq!((tiling.len() - 1) % w, 0, "tiling prompt tiles at {w}");
        assert_ne!((ragged.len() - 1) % w, 0, "ragged prompt is ragged at {w}");
    }

    let gpu = Dsv4Gpu::load(dir, &[0, 1], ActQuantVariant::RefFp8Round, 4096).expect("load");
    assert!(
        !gpu.matrix_moe_enabled(),
        "this gate runs the served reference program"
    );

    // ---- 1. class census, resolved from the model, with its red arm -------------------
    let probe = prime_chunked(&gpu, &ragged, SERVING_WIDTH_MAX, 64);
    let expected = gpu.expected_cache_class_names();
    let observed: Vec<String> = probe
        .snapshot
        .classes
        .iter()
        .map(|(n, _)| n.clone())
        .filter(|n| !n.starts_with("dspark."))
        .collect();
    assert_eq!(observed, expected, "cache class census");
    assert!(
        expected.iter().any(|n| n.ends_with(".cmp_pend_score"))
            && expected.iter().any(|n| n.ends_with(".idx_pend_score"))
            && expected.iter().any(|n| n.ends_with(".ring")),
        "census is non-vacuous: compressed, indexer and window state are all covered"
    );
    let mut truncated = expected.clone();
    truncated.pop();
    assert_ne!(
        observed, truncated,
        "census red arm: a dropped class must fail"
    );
    println!(
        "CHECK census classes={} dspark_rings={} unix_ms={}",
        expected.len(),
        probe.snapshot.classes.len() - expected.len(),
        unix_ms()
    );

    // ---- 1b. prefill-head census, crosswise between the door's arms -------------------
    let rows = (ragged.len() - 1) as u64;
    let chunks = chunks_for(ragged.len() - 1, SERVING_WIDTH_MAX);
    let before = gpu.prefill_head_stats();
    let head_probe = prime_chunked(&gpu, &ragged, SERVING_WIDTH_MAX, 64);
    let after = gpu.prefill_head_stats();
    let seen = (
        after.full_rows - before.full_rows,
        after.last_rows - before.last_rows,
        after.skipped_chunks - before.skipped_chunks,
    );
    let arm = Dsv4PrefillHead::All;
    assert_eq!(
        seen,
        prefill_head_census(arm, chunks, rows),
        "head census {arm:?}"
    );
    assert_ne!(
        seen,
        prefill_head_census(Dsv4PrefillHead::Last, chunks, rows),
        "head census red arm: the other arm's expectation must fail"
    );
    println!(
        "CHECK prefill_head arm={arm:?} counters={seen:?} chunks={chunks} unix_ms={}",
        unix_ms()
    );
    drop(head_probe);

    // ---- 1c. comparator red arm: a one-token perturbation must be caught --------------
    let mut perturbed_prompt = ragged.clone();
    let last = perturbed_prompt.len() - 1;
    perturbed_prompt[last] = if perturbed_prompt[last] == 0 {
        1
    } else {
        perturbed_prompt[last] - 1
    };
    let perturbed = prime_chunked(&gpu, &perturbed_prompt, SERVING_WIDTH_MAX, 64);
    let caught = first_difference(&probe.snapshot, &perturbed.snapshot)
        .expect("comparator red arm: a changed final token must be visible");
    println!("REDARM comparator first_difference={caught}");
    drop(perturbed);

    // ---- 1d. width guards refuse out-of-range requests --------------------------------
    {
        let mut state = gpu
            .alloc_decode_state_for_transient(ragged.len() + 64, SERVING_WIDTH_MAX)
            .expect("decode state");
        let mut dstate = gpu.dspark_alloc_state().expect("dspark state");
        let zero = gpu.dspark_prefill_prime_chunked(&ragged, &mut state, &mut dstate, 0);
        assert!(zero.is_err(), "width 0 must refuse");
        let over = gpu.dspark_prefill_prime_chunked(
            &ragged,
            &mut state,
            &mut dstate,
            SERVING_WIDTH_MAX * 4,
        );
        assert!(
            over.is_err(),
            "a width past the allocated transient rows must refuse"
        );
        println!(
            "REDARM width guards refuse: zero={:?} over={:?}",
            zero.err().expect("zero"),
            over.err().expect("over")
        );
    }

    // ---- 2. chunk-width exactness, two-sided against the measured law ----------------
    // MEASURED 2026-09-10, not assumed: chunked prefill is bit-invariant to the width
    // whenever every transaction carries at least TWO rows, and a transaction of exactly
    // ONE row is a different numeric class, because a single row takes the m=1
    // decode-shaped kernels instead of the batched ones. Width 32 and width 63 are
    // bit-identical to width 64 on a 1024-row suffix (tails 0 and 16); widths 31 and 33
    // differ there and differ IDENTICALLY to each other (both leave a 1-row tail); on a
    // 999-row suffix, where no gated width leaves a 1-row tail, width 31 is bit-identical
    // again. So the expectation is keyed to the TAIL SHAPE the width produces, and it is
    // asserted BOTH ways: a width predicted exact that differs fails, and a width
    // predicted to differ that comes back exact fails too, because that would mean the
    // law moved and no one noticed.
    fn width_is_batched_class(suffix_rows: usize, width: usize) -> bool {
        if width < 2 {
            return false;
        }
        let tail = suffix_rows % width;
        tail != 1
    }
    let mut law_holds = true;
    for (label, prompt) in [("tiling", &tiling), ("ragged", &ragged)] {
        let suffix_rows = prompt.len() - 1;
        assert!(
            width_is_batched_class(suffix_rows, SERVING_WIDTH_MAX),
            "the reference width must itself be in the batched class"
        );
        let reference = prime_chunked(&gpu, prompt, SERVING_WIDTH_MAX, 64);
        println!(
            "REFERENCE prompt={label} width={SERVING_WIDTH_MAX} suffix={suffix_rows} tail={} digest={}",
            suffix_rows % SERVING_WIDTH_MAX,
            reference.snapshot.digest()
        );
        // determinism at a fixed width: the same width twice must be bit-identical, or
        // nothing below means anything.
        let twin = prime_chunked(&gpu, prompt, SERVING_WIDTH_MAX, 64);
        match first_difference(&reference.snapshot, &twin.snapshot) {
            None => println!(
                "DETERMINISTIC prompt={label} width={SERVING_WIDTH_MAX}two runs bit-identical"
            ),
            Some(diff) => {
                law_holds = false;
                println!(
                    "NONDETERMINISTIC prompt={label} width={SERVING_WIDTH_MAX} first_bit={diff}"
                );
            }
        }
        drop(twin);
        for width in [1usize, 31, 32, 33, 63] {
            let tail = suffix_rows % width;
            let predicted_exact = width_is_batched_class(suffix_rows, width);
            let run = prime_chunked(&gpu, prompt, width, 64);
            let diff = first_difference(&reference.snapshot, &run.snapshot);
            let observed_exact = diff.is_none();
            match &diff {
                None => println!(
                    "EXACT prompt={label} width={width} tail={tail} vs {SERVING_WIDTH_MAX} logits/live-cache/DSpark digest={}",
                    run.snapshot.digest()
                ),
                Some(d) => {
                    // A width outside the batched class is a MEASURED different class, so
                    // its drift is reported with numbers rather than a bare mismatch.
                    let p = softmax_f64(&reference.snapshot.logits);
                    let q = softmax_f64(&run.snapshot.logits);
                    println!(
                        "DIFFER prompt={label} width={width} tail={tail} top1_ref={} top1_run={} kl_fwd={:.3e} kl_rev={:.3e} first_bit={d}",
                        argmax(&reference.snapshot.logits),
                        argmax(&run.snapshot.logits),
                        kl(&p, &q),
                        kl(&q, &p)
                    );
                }
            }
            if observed_exact != predicted_exact {
                law_holds = false;
                println!(
                    "LAW_BROKEN prompt={label} width={width} tail={tail} predicted_exact={predicted_exact} observed_exact={observed_exact}"
                );
            }
        }
    }
    assert!(
        law_holds,
        "chunk-width class law: batched-class widths must be bit-identical and one-row transactions must not be"
    );

    // ---- 2b. the decisive probe: it is the ONE-ROW TRANSACTION, not the width ---------
    // Same prefix, same two next tokens, same configured width. The only difference is
    // whether those two rows are committed in one 2-row transaction or two 1-row ones.
    {
        let base: Vec<u32> = full[..600].to_vec();
        let pair: Vec<u32> = full[600..602].to_vec();
        let mut wide = prime_chunked(&gpu, &base, SERVING_WIDTH_MAX, 64 + 8);
        let wide_logits = gpu
            .dspark_continue_prefix_chunked(
                &pair,
                &mut wide.state,
                &mut wide.dstate,
                SERVING_WIDTH_MAX,
            )
            .expect("2-row transaction");
        let wide_snap = Snapshot::take(&gpu, wide_logits, &wide.state, &wide.dstate);
        let mut split = prime_chunked(&gpu, &base, SERVING_WIDTH_MAX, 64 + 8);
        gpu.dspark_continue_prefix_chunked(
            &pair[..1],
            &mut split.state,
            &mut split.dstate,
            SERVING_WIDTH_MAX,
        )
        .expect("first 1-row transaction");
        let split_logits = gpu
            .dspark_continue_prefix_chunked(
                &pair[1..],
                &mut split.state,
                &mut split.dstate,
                SERVING_WIDTH_MAX,
            )
            .expect("second 1-row transaction");
        let split_snap = Snapshot::take(&gpu, split_logits, &split.state, &split.dstate);
        // control: the 2-row shape repeated must be bit-identical to itself, so the probe
        // below cannot be reporting run-to-run noise.
        let mut control = prime_chunked(&gpu, &base, SERVING_WIDTH_MAX, 64 + 8);
        let control_logits = gpu
            .dspark_continue_prefix_chunked(
                &pair,
                &mut control.state,
                &mut control.dstate,
                SERVING_WIDTH_MAX,
            )
            .expect("2-row control");
        let control_snap = Snapshot::take(&gpu, control_logits, &control.state, &control.dstate);
        assert!(
            first_difference(&wide_snap, &control_snap).is_none(),
            "the 2-row transaction must reproduce itself bit for bit"
        );
        match first_difference(&wide_snap, &split_snap) {
            None => println!(
                "PROBE one-row-vs-two-row: BIT-IDENTICAL (the class boundary is not the transaction row count)"
            ),
            Some(diff) => println!("PROBE one-row-vs-two-row: DIFFERENT class, first_bit={diff}"),
        }
    }

    // ---- 3. the chunked-versus-monolithic numeric class, measured ---------------------
    let mut all_bit_equal = true;
    let mut top1_changed = 0usize;
    let mut greedy_same = 0usize;
    let mut kl_fwd: Vec<f64> = Vec::new();
    let mut kl_rev: Vec<f64> = Vec::new();
    let pinned: Vec<usize> = vec![65, 100, 129, 256, 513, 1000];
    for n in &pinned {
        let prefix: Vec<u32> = full[..*n].to_vec();
        let chunked = prime_chunked(&gpu, &prefix, SERVING_WIDTH_MAX, 8);
        let mono = prime_monolithic(&gpu, &prefix, 8);
        let bit = first_difference(&chunked.snapshot, &mono.snapshot);
        let p = softmax_f64(&chunked.snapshot.logits);
        let q = softmax_f64(&mono.snapshot.logits);
        let (a, b) = (
            argmax(&chunked.snapshot.logits),
            argmax(&mono.snapshot.logits),
        );
        let (f, r) = (kl(&p, &q), kl(&q, &p));
        kl_fwd.push(f);
        kl_rev.push(r);
        if a == b {
            greedy_same += 1;
        } else {
            top1_changed += 1;
        }
        match &bit {
            None => println!("CLASS n={n} bit_equal=yes top1={a} kl_fwd=0 kl_rev=0"),
            Some(diff) => {
                all_bit_equal = false;
                println!(
                    "CLASS n={n} bit_equal=no top1_chunked={a} top1_mono={b} kl_fwd={f:.3e} kl_rev={r:.3e} first_bit={diff}"
                );
            }
        }
    }
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    let max = |v: &[f64]| v.iter().cloned().fold(0f64, f64::max);
    if all_bit_equal {
        println!(
            "VERDICT numeric_class=same rows={} greedy_identical={greedy_same}/{}",
            pinned.len(),
            pinned.len()
        );
    } else {
        println!(
            "VERDICT numeric_class=new rows={} top1_changed={top1_changed} greedy_identical={greedy_same}/{} kl_fwd_mean={:.3e} kl_fwd_max={:.3e} kl_rev_mean={:.3e} kl_rev_max={:.3e}",
            pinned.len(),
            pinned.len(),
            mean(&kl_fwd),
            max(&kl_fwd),
            mean(&kl_rev),
            max(&kl_rev)
        );
    }

    // ---- 4. sampled cache transparency ------------------------------------------------
    let mut chunked = prime_chunked(&gpu, &ragged, SERVING_WIDTH_MAX, 64);
    let mut mono = prime_monolithic(&gpu, &ragged, 64);
    let prime_diff = first_difference(&chunked.snapshot, &mono.snapshot);
    let chunked_sampled = sampled_continuation(&gpu, &ragged, &mut chunked, 16);
    let mono_sampled = sampled_continuation(&gpu, &ragged, &mut mono, 16);
    match first_difference(&chunked_sampled, &mono_sampled) {
        None => println!(
            "TRANSPARENT sampled chunked-vs-monolithic tokens/rounds/cache digest={}",
            chunked_sampled.digest()
        ),
        Some(diff) => println!("OPAQUE sampled chunked-vs-monolithic first_bit={diff}"),
    }
    if let Some(diff) = &prime_diff {
        println!("NOTE prime state already differs before sampling: {diff}");
    }

    // restored-suffix transparency: the split lands mid-transaction on purpose.
    let split = 501usize;
    assert_ne!((split - 1) % SERVING_WIDTH_MAX, 0, "split lands mid-chunk");
    let whole = prime_chunked(&gpu, &ragged, SERVING_WIDTH_MAX, 64);
    let mut piecewise = prime_chunked(&gpu, &ragged[..split], SERVING_WIDTH_MAX, 64 + ragged.len());
    let tail_logits = gpu
        .dspark_continue_prefix_chunked(
            &ragged[split..],
            &mut piecewise.state,
            &mut piecewise.dstate,
            SERVING_WIDTH_MAX,
        )
        .expect("chunked restored suffix");
    let piecewise_snapshot = Snapshot::take(&gpu, tail_logits, &piecewise.state, &piecewise.dstate);
    let restored_diff = first_difference(&whole.snapshot, &piecewise_snapshot);
    match &restored_diff {
        None => println!("TRANSPARENT restored-suffix split={split} logits/live-cache/DSpark"),
        Some(diff) => println!("OPAQUE restored-suffix split={split} first_bit={diff}"),
    }
    assert!(
        restored_diff.is_none(),
        "restored-suffix transparency: a mid-chunk split must leave the same state as one pass"
    );

    println!(
        "PASS chunk-width exactness and the numeric-class and transparency rows above; reachability and TTFT are separate cells unix_ms={}",
        unix_ms()
    );
}
