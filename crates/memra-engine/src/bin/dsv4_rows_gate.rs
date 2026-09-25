//! B-row decode identity gate (memra #667 lever 2).
//!
//! Four sessions with distinct prompts of distinct lengths decode `steps` greedy tokens
//! alone, one `decode_step` each (the served one-row matrix program). Each step's full
//! logits row is hashed bit for bit. Then fresh copies of the same sessions decode again
//! through `decode_rows_logits`: session s joins at step `JOIN[s]` and leaves after its
//! `steps` tokens, so the batch width runs 1, 2, 3, 4 and back down. The row order rotates
//! every step, so a session sits at every row index.
//!
//! Correctness: every session's logits bits and tokens at every step equal its solo run.
//! A request must not change numeric program when a peer joins or leaves, or when its row
//! moves.
//!
//! A second identity arm pipelines two groups of two sessions (`decode_rows_enqueue`, `_wait`,
//! `_complete`): group 0's step is queued, then group 1's behind it, and each is completed and
//! queued again in turn, so the two cards run different groups' stages. Every row's logits bits
//! must again equal its solo step.
//!
//! Timing (after the identity arms, decode wall only): the one-row step against B-row steps
//! of 1, 2 and 4 rows, and two pipelined groups of 2, reported as ms per step and aggregate
//! tokens per second.
//!
//! Profile mode (`DSV4_ROWS_GATE_PROFILE=B`): prime B sessions, warm up, then run `timing-steps`
//! B-row steps between `cuProfilerStart` and `cuProfilerStop` and exit, for
//! `nsys profile --capture-range=cudaProfilerApi`. Comparing B=1 with B=2 attributes the cost an
//! added row brings, kernel by kernel.
//!
//! TP/EP mode (`DSV4_ROWS_GATE_TOPOLOGY=tp_ep`, memra #710 B-row): the same sessions on the
//! served TP/EP program (expert-ID EP, attention TP2). The solo reference is the eager one-row
//! TP/EP step and the batched arm is the TP/EP B-row step. The pipelined arm does not apply (a
//! TP/EP step uses both cards). Instead a replay arm arms session 0 for greedy full-token
//! replay and alternates: replayed steps alone, B-row steps beside session 1 while still
//! armed, then replayed steps again. Every step's logits bits must equal the solo trace.
//!
//! Also in TP/EP mode, a graph arm runs the batched schedule through `decode_rows_draw`, whose
//! steps of two or more rows run a captured graph per batch (memra #710 B-row graphs): every
//! row's logits bits must again equal the solo trace, and the arm must have captured and
//! replayed. A sampled graph arm draws every row at the vendor default and must give the
//! eager B-row step's draws. Timing then adds the graph step at B=2 and 4.
//!
//! Usage: `dsv4_rows_gate <model-dir> <source.txt> [steps] [timing-steps]`.
//! Rig law: under the box GPU lock, served defaults (no MEMRA_DSV4_* overrides).
use memra_engine::dsv4_gpu::{DecodeState, Dsv4Gpu, Dsv4RowDraw, Dsv4SampleCfg};
use memra_engine::dsv4_source_tape::SourceTape;
use memra_gguf::dsv4_forward::ActQuantVariant;
use memra_tokenizer::Tokenizer;
use std::{path::Path, time::Instant};

const CHUNK: usize = 256;
const LENS: [usize; 4] = [256, 197, 311, 150];
const JOIN: [usize; 4] = [0, 3, 7, 12];
const FRAMES: [&str; 4] = [
    "Review this inference engine source:\n\n",
    "Explain what this inference engine source does:\n\n",
    "List the invariants this source keeps:\n\n",
    "Summarize this file for a new contributor:\n\n",
];

fn drain(gpu: &Dsv4Gpu) {
    for stage in &gpu.stages {
        stage.gpu.stream().synchronize().expect("gate drain");
    }
}

fn argmax(row: &[f32]) -> u32 {
    let mut best = 0usize;
    for i in 1..row.len() {
        if row[i] > row[best] {
            best = i;
        }
    }
    best as u32
}

/// FNV-1a over the logits' bit patterns: two rows hash equal only if every bit matches
/// (up to a 64-bit collision).
fn bits_hash(row: &[f32]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for v in row {
        for byte in v.to_bits().to_le_bytes() {
            h ^= u64::from(byte);
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    h
}

struct Session {
    state: DecodeState,
    next: u32,
}

fn prime(gpu: &Dsv4Gpu, prompt: &[u32], capacity: usize) -> Session {
    let mut state = gpu
        .alloc_decode_state_for_transient(capacity, CHUNK.max(gpu.verify_tmax()))
        .expect("decode state");
    let logits = gpu
        .prefill_with_cache_chunked(prompt, &mut state, CHUNK)
        .expect("prefill");
    Session {
        state,
        next: argmax(&logits),
    }
}

/// (token, logits bit hash) per step, per session.
type Trace = Vec<Vec<(u32, u64)>>;

fn solo(gpu: &Dsv4Gpu, prompts: &[Vec<u32>], capacity: usize, steps: usize) -> Trace {
    prompts
        .iter()
        .map(|p| {
            let mut s = prime(gpu, p, capacity);
            (0..steps)
                .map(|_| {
                    let logits = gpu.decode_step(s.next, &mut s.state).expect("solo step");
                    s.next = argmax(&logits);
                    (s.next, bits_hash(&logits))
                })
                .collect()
        })
        .collect()
}

fn batched(
    gpu: &Dsv4Gpu,
    prompts: &[Vec<u32>],
    capacity: usize,
    steps: usize,
) -> (Trace, Vec<usize>) {
    let mut sessions: Vec<Session> = prompts.iter().map(|p| prime(gpu, p, capacity)).collect();
    let mut rows = gpu
        .alloc_rows_state(sessions.len())
        .expect("B-row workspace");
    let mut trace: Trace = vec![Vec::new(); sessions.len()];
    let mut widths = Vec::new();
    let horizon = JOIN.iter().max().unwrap() + steps;
    for k in 0..horizon {
        let mut active: Vec<usize> = (0..sessions.len())
            .filter(|&s| k >= JOIN[s] && trace[s].len() < steps)
            .collect();
        if active.is_empty() {
            continue;
        }
        let n = active.len();
        active.rotate_left(k % n);
        widths.push(n);
        let toks: Vec<u32> = active.iter().map(|&s| sessions[s].next).collect();
        let mut picked: Vec<Option<&mut Session>> = sessions.iter_mut().map(Some).collect();
        let mut states: Vec<&mut DecodeState> = active
            .iter()
            .map(|&s| &mut picked[s].take().expect("each session once").state)
            .collect();
        let logits = gpu
            .decode_rows_logits(&toks, &mut states, &mut rows)
            .expect("B-row step");
        drop(states);
        for (&s, row) in active.iter().zip(&logits) {
            let tok = argmax(row);
            sessions[s].next = tok;
            trace[s].push((tok, bits_hash(row)));
        }
    }
    (trace, widths)
}

/// Two groups of two sessions, pipelined: every step of every session through
/// `decode_rows_enqueue` / `_wait` / `_complete`, full logits kept for the bit comparison.
fn pipelined_groups(gpu: &Dsv4Gpu, prompts: &[Vec<u32>], capacity: usize, steps: usize) -> Trace {
    let mut sessions: Vec<Session> = prompts.iter().map(|p| prime(gpu, p, capacity)).collect();
    let mut rows = [
        gpu.alloc_rows_state(2).expect("group 0 workspace"),
        gpu.alloc_rows_state(2).expect("group 1 workspace"),
    ];
    let mut trace: Trace = vec![Vec::new(); sessions.len()];
    let (a, b) = sessions.split_at_mut(2);
    let mut groups = [a, b];
    for (g, group) in groups.iter_mut().enumerate() {
        let toks: Vec<u32> = group.iter().map(|s| s.next).collect();
        let mut states: Vec<&mut DecodeState> = group.iter_mut().map(|s| &mut s.state).collect();
        gpu.decode_rows_enqueue(&toks, &mut states, &mut rows[g], true)
            .expect("first enqueue");
    }
    for step in 0..steps {
        for (g, group) in groups.iter_mut().enumerate() {
            gpu.decode_rows_wait(&mut rows[g]).expect("group wait");
            let mut states: Vec<&mut DecodeState> =
                group.iter_mut().map(|s| &mut s.state).collect();
            let (logits, _) = gpu
                .decode_rows_complete(&mut states, &mut rows[g])
                .expect("group complete");
            drop(states);
            for (i, (s, row)) in group.iter_mut().zip(logits.expect("full rows")).enumerate() {
                s.next = argmax(&row);
                trace[2 * g + i].push((s.next, bits_hash(&row)));
            }
            if step + 1 < steps {
                let toks: Vec<u32> = group.iter().map(|s| s.next).collect();
                let mut states: Vec<&mut DecodeState> =
                    group.iter_mut().map(|s| &mut s.state).collect();
                gpu.decode_rows_enqueue(&toks, &mut states, &mut rows[g], true)
                    .expect("group enqueue");
            }
        }
    }
    trace
}

/// The batched schedule through `decode_rows_draw`: graph steps whenever two or more rows run.
/// Returns each session's (token, logits hash) per step, read from the head workspace.
fn graphed(gpu: &Dsv4Gpu, prompts: &[Vec<u32>], capacity: usize, steps: usize) -> Trace {
    let mut sessions: Vec<Session> = prompts.iter().map(|p| prime(gpu, p, capacity)).collect();
    let mut rows = gpu
        .alloc_rows_state(sessions.len())
        .expect("B-row workspace");
    let mut trace: Trace = vec![Vec::new(); sessions.len()];
    let horizon = JOIN.iter().max().unwrap() + steps;
    for k in 0..horizon {
        let active: Vec<usize> = (0..sessions.len())
            .filter(|&s| k >= JOIN[s] && trace[s].len() < steps)
            .collect();
        if active.is_empty() {
            continue;
        }
        let n = active.len();
        // No rotation here: the graph step orders its batch by request serial (the sessions'
        // priming order), and the logits read back are in that order. The eager arm above
        // covers row moves.
        let toks: Vec<u32> = active.iter().map(|&s| sessions[s].next).collect();
        let draws = vec![Dsv4RowDraw::Argmax; n];
        let mut picked: Vec<Option<&mut Session>> = sessions.iter_mut().map(Some).collect();
        let mut states: Vec<&mut DecodeState> = active
            .iter()
            .map(|&s| &mut picked[s].take().expect("each session once").state)
            .collect();
        let next = gpu
            .decode_rows_draw(&toks, &mut states, &mut rows, &draws)
            .expect("graph B-row step");
        drop(states);
        let logits = gpu.rows_logits_for_gate(&rows, n).expect("rows logits");
        for ((&s, row), tok) in active.iter().zip(&logits).zip(next) {
            assert_eq!(tok, argmax(row), "graph argmax is the logits argmax");
            sessions[s].next = tok;
            trace[s].push((tok, bits_hash(row)));
        }
    }
    trace
}

/// Four sessions stepping together for `steps` sampled steps at the vendor default, through
/// the eager B-row step (graphs off) and through the captured one; the draws must match.
fn sampled_graph_vs_eager(
    gpu: &Dsv4Gpu,
    prompts: &[Vec<u32>],
    capacity: usize,
    steps: usize,
) -> (Vec<Vec<u32>>, Vec<Vec<u32>>) {
    let run = |graph: bool| -> Vec<Vec<u32>> {
        let prev = memra_engine::dsv4_gpu::set_rows_graph_for_gate(graph);
        let mut sessions: Vec<Session> = prompts.iter().map(|p| prime(gpu, p, capacity)).collect();
        let mut rows = gpu
            .alloc_rows_state(sessions.len())
            .expect("B-row workspace");
        let draws: Vec<Dsv4RowDraw> = (0..sessions.len())
            .map(|s| {
                Dsv4RowDraw::Sample(Dsv4SampleCfg {
                    temperature: 1.0,
                    top_p: 1.0,
                    top_k: 0,
                    seed: 20260926 + s as u64,
                })
            })
            .collect();
        let mut out = vec![Vec::new(); sessions.len()];
        for _ in 0..steps {
            let toks: Vec<u32> = sessions.iter().map(|s| s.next).collect();
            let mut states: Vec<&mut DecodeState> =
                sessions.iter_mut().map(|s| &mut s.state).collect();
            let next = gpu
                .decode_rows_draw(&toks, &mut states, &mut rows, &draws)
                .expect("sampled B-row step");
            drop(states);
            for ((s, t), o) in sessions.iter_mut().zip(next).zip(out.iter_mut()) {
                s.next = t;
                o.push(t);
            }
        }
        memra_engine::dsv4_gpu::set_rows_graph_for_gate(prev);
        out
    };
    let eager = run(false);
    let graph = run(true);
    (eager, graph)
}

/// TP/EP replay arm: session 0 armed for greedy replay steps alone for `steps / 3` steps, then
/// rides B-row steps beside session 1 (still armed) for the next third, then replays alone
/// again. Returns session 0's trace, and session 1's for the steps it shared.
fn replay_reentry(
    gpu: &Dsv4Gpu,
    prompts: &[Vec<u32>],
    capacity: usize,
    steps: usize,
) -> (Vec<(u32, u64)>, Vec<(u32, u64)>) {
    let mut a = prime(gpu, &prompts[0], capacity);
    let mut b = prime(gpu, &prompts[1], capacity);
    let greedy = Dsv4SampleCfg {
        temperature: 0.0,
        top_p: 1.0,
        top_k: 0,
        seed: 0,
    };
    unsafe { gpu.arm_full_token_replay(&mut a.state, greedy) }.expect("arm replay");
    let mut rows = gpu.alloc_rows_state(2).expect("B-row workspace");
    let (mut trace_a, mut trace_b) = (Vec::new(), Vec::new());
    let third = steps / 3;
    for k in 0..steps {
        if (third..2 * third).contains(&k) {
            let toks = [a.next, b.next];
            let mut states = [&mut a.state, &mut b.state];
            let logits = gpu
                .decode_rows_logits(&toks, &mut states, &mut rows)
                .expect("B-row step beside an armed request");
            a.next = argmax(&logits[0]);
            b.next = argmax(&logits[1]);
            trace_a.push((a.next, bits_hash(&logits[0])));
            trace_b.push((b.next, bits_hash(&logits[1])));
        } else {
            let tok = gpu
                .decode_sample_full_token(a.next, &mut a.state)
                .expect("replayed step");
            let logits = gpu
                .read_decode_logits_for_gate(&a.state)
                .expect("replay logits");
            assert_eq!(tok, argmax(&logits), "replayed argmax is the logits argmax");
            a.next = tok;
            trace_a.push((tok, bits_hash(&logits)));
        }
    }
    (trace_a, trace_b)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    assert!(
        (3..=5).contains(&args.len()),
        "usage: dsv4_rows_gate <model-dir> <source.txt> [steps] [timing-steps]"
    );
    let steps: usize = args.get(3).map_or(24, |v| v.parse().expect("steps"));
    let timing_steps: usize = args.get(4).map_or(64, |v| v.parse().expect("timing steps"));
    assert!(steps >= 2);
    let dir = Path::new(&args[1]);
    let tape = SourceTape::read(&args[2]).expect("source tape");
    let tokenizer = Tokenizer::from_hf_dir(dir).expect("tokenizer");
    let prompts: Vec<Vec<u32>> = FRAMES
        .iter()
        .zip(LENS)
        .map(|(frame, len)| tape.prompt(&tokenizer, frame, len)[..len].to_vec())
        .collect();
    // Replay admits capacities of at least 512.
    let capacity = (LENS.iter().max().unwrap() + steps.max(timing_steps) + 128).max(512);
    let tp_ep = match std::env::var("DSV4_ROWS_GATE_TOPOLOGY").as_deref() {
        Err(_) | Ok("pp") => false,
        Ok("tp_ep") => true,
        Ok(other) => panic!("DSV4_ROWS_GATE_TOPOLOGY {other:?} must be pp or tp_ep"),
    };
    if tp_ep {
        Dsv4Gpu::set_tp_ep_topology_for_gate(true);
        Dsv4Gpu::set_attention_tp_for_gate(true);
    }
    let gpu = Dsv4Gpu::load(dir, &[0, 1], ActQuantVariant::RefFp8Round, capacity).expect("load");
    assert_eq!(gpu.topology().is_tp_ep(), tp_ep);
    println!("TOPOLOGY {}", if tp_ep { "tp_ep" } else { "pp" });
    println!(
        "PROTOCOL {{\"sessions\":4,\"prompt_tokens\":{LENS:?},\"join_steps\":{JOIN:?},\"steps\":{steps},\"timing_steps\":{timing_steps},\"greedy\":true,\"compare\":\"full logits bits per step\",\"source_sha256\":\"{}\"}}",
        tape.sha256
    );

    if let Ok(b) = std::env::var("DSV4_ROWS_GATE_PROFILE") {
        let b: usize = b.parse().expect("DSV4_ROWS_GATE_PROFILE rows");
        assert!((1..=prompts.len()).contains(&b));
        let mut sessions: Vec<Session> = prompts[..b]
            .iter()
            .map(|p| prime(&gpu, p, capacity))
            .collect();
        let mut rows = gpu.alloc_rows_state(b).expect("B-row workspace");
        let mut run = |n: usize, sessions: &mut Vec<Session>| {
            for _ in 0..n {
                let toks: Vec<u32> = sessions.iter().map(|s| s.next).collect();
                let mut states: Vec<&mut DecodeState> =
                    sessions.iter_mut().map(|s| &mut s.state).collect();
                let next = gpu
                    .decode_rows_greedy(&toks, &mut states, &mut rows)
                    .expect("B-row");
                drop(states);
                for (s, t) in sessions.iter_mut().zip(next) {
                    s.next = t;
                }
            }
        };
        run(4, &mut sessions);
        drain(&gpu);
        cudarc::driver::profiler_start().expect("cuProfilerStart");
        let t0 = Instant::now();
        run(timing_steps, &mut sessions);
        drain(&gpu);
        let secs = t0.elapsed().as_secs_f64();
        cudarc::driver::profiler_stop().expect("cuProfilerStop");
        println!(
            "PROFILE B={b} steps={timing_steps} ms_per_step={:.3}",
            1e3 * secs / timing_steps as f64
        );
        return;
    }
    let reference = solo(&gpu, &prompts, capacity, steps);
    let (rows, widths) = batched(&gpu, &prompts, capacity, steps);
    println!("WIDTHS {widths:?}");
    let mut bad = 0usize;
    for s in 0..prompts.len() {
        let first = (0..steps).find(|&i| reference[s][i] != rows[s][i]);
        match first {
            None => println!(
                "SESSION {s} len={} join={} steps={steps} IDENTICAL head={:?}",
                LENS[s],
                JOIN[s],
                reference[s].iter().take(6).map(|x| x.0).collect::<Vec<_>>()
            ),
            Some(i) => {
                bad += 1;
                println!(
                    "SESSION {s} len={} join={} FIRST DIVERGENCE step={i} solo=(tok {}, bits {:016x}) rows=(tok {}, bits {:016x})",
                    LENS[s],
                    JOIN[s],
                    reference[s][i].0,
                    reference[s][i].1,
                    rows[s][i].0,
                    rows[s][i].1
                );
            }
        }
    }
    if bad > 0 {
        println!("FAILED: {bad} of {} sessions diverged", prompts.len());
        std::process::exit(1);
    }
    println!("PASS: every row bit-identical to its solo step across join, leave and row moves");
    if tp_ep {
        let (a, b) = replay_reentry(&gpu, &prompts, capacity, steps);
        let third = steps / 3;
        let bad_a = (0..steps).find(|&i| reference[0][i] != a[i]);
        let bad_b = (0..b.len()).find(|&i| reference[1][i] != b[i]);
        if let Some(i) = bad_a {
            println!(
                "REPLAY SESSION 0 FIRST DIVERGENCE step={i} solo=(tok {}, bits {:016x}) arm=(tok {}, bits {:016x})",
                reference[0][i].0, reference[0][i].1, a[i].0, a[i].1
            );
        }
        if let Some(i) = bad_b {
            println!(
                "REPLAY PEER FIRST DIVERGENCE step={i} solo=(tok {}, bits {:016x}) arm=(tok {}, bits {:016x})",
                reference[1][i].0, reference[1][i].1, b[i].0, b[i].1
            );
        }
        if bad_a.is_some() || bad_b.is_some() {
            println!("FAILED: the replay arm diverged");
            std::process::exit(1);
        }
        println!(
            "PASS: replayed, then {third} B-row steps beside a peer while armed, then replayed again: bit-identical to solo steps"
        );
        let (captures0, steps0) = Dsv4Gpu::rows_graph_counts_for_gate();
        let graph = graphed(&gpu, &prompts, capacity, steps);
        let (captures1, steps1) = Dsv4Gpu::rows_graph_counts_for_gate();
        for s in 0..prompts.len() {
            if let Some(i) = (0..steps).find(|&i| reference[s][i] != graph[s][i]) {
                println!(
                    "GRAPH SESSION {s} FIRST DIVERGENCE step={i} solo=(tok {}, bits {:016x}) graph=(tok {}, bits {:016x})",
                    reference[s][i].0, reference[s][i].1, graph[s][i].0, graph[s][i].1
                );
                println!("FAILED: the B-row graph diverged");
                std::process::exit(1);
            }
        }
        let (captures, graph_steps) = (captures1 - captures0, steps1 - steps0);
        if captures == 0 || graph_steps <= captures {
            println!("FAILED: the graph arm captured {captures} and replayed {graph_steps} steps");
            std::process::exit(1);
        }
        println!(
            "PASS: B-row graph steps bit-identical to solo steps across join, leave and row moves ({captures} captures, {graph_steps} graph steps)"
        );
        let (eager, graphs) = sampled_graph_vs_eager(&gpu, &prompts, capacity, steps);
        if eager != graphs {
            println!("FAILED: sampled graph draws differ from the eager B-row draws");
            std::process::exit(1);
        }
        println!(
            "PASS: sampled B-row graph draws equal the eager B-row draws ({steps} steps x 4 rows)"
        );
    } else {
        let piped = pipelined_groups(&gpu, &prompts, capacity, steps);
        for s in 0..prompts.len() {
            if let Some(i) = (0..steps).find(|&i| reference[s][i] != piped[s][i]) {
                println!(
                    "PIPELINED SESSION {s} FIRST DIVERGENCE step={i} solo=(tok {}, bits {:016x}) rows=(tok {}, bits {:016x})",
                    reference[s][i].0, reference[s][i].1, piped[s][i].0, piped[s][i].1
                );
                println!("FAILED: pipelined groups diverged");
                std::process::exit(1);
            }
        }
        println!("PASS: pipelined groups bit-identical to solo steps");
    }

    // Timing: one-row steps vs B-row steps, fresh sessions each arm, decode wall only.
    for rep in 0..2 {
        for b in [1usize, 2, 4] {
            let mut sessions: Vec<Session> = prompts[..b]
                .iter()
                .map(|p| prime(&gpu, p, capacity))
                .collect();
            drain(&gpu);
            let t0 = Instant::now();
            for _ in 0..timing_steps {
                for s in &mut sessions {
                    s.next = gpu
                        .decode_step_greedy(s.next, &mut s.state)
                        .expect("one-row");
                }
            }
            let solo_s = t0.elapsed().as_secs_f64();
            let mut sessions: Vec<Session> = prompts[..b]
                .iter()
                .map(|p| prime(&gpu, p, capacity))
                .collect();
            let mut rows = gpu.alloc_rows_state(b).expect("B-row workspace");
            drain(&gpu);
            let t0 = Instant::now();
            for _ in 0..timing_steps {
                let toks: Vec<u32> = sessions.iter().map(|s| s.next).collect();
                let mut states: Vec<&mut DecodeState> =
                    sessions.iter_mut().map(|s| &mut s.state).collect();
                let next = gpu
                    .decode_rows_greedy(&toks, &mut states, &mut rows)
                    .expect("B-row");
                drop(states);
                for (s, t) in sessions.iter_mut().zip(next) {
                    s.next = t;
                }
            }
            let rows_s = t0.elapsed().as_secs_f64();
            if tp_ep && b >= 2 {
                let mut sessions: Vec<Session> = prompts[..b]
                    .iter()
                    .map(|p| prime(&gpu, p, capacity))
                    .collect();
                let mut rows = gpu.alloc_rows_state(b).expect("B-row workspace");
                let draws = vec![Dsv4RowDraw::Argmax; b];
                drain(&gpu);
                let t0 = Instant::now();
                for _ in 0..timing_steps {
                    let toks: Vec<u32> = sessions.iter().map(|s| s.next).collect();
                    let mut states: Vec<&mut DecodeState> =
                        sessions.iter_mut().map(|s| &mut s.state).collect();
                    let next = gpu
                        .decode_rows_draw(&toks, &mut states, &mut rows, &draws)
                        .expect("graph B-row");
                    drop(states);
                    for (s, t) in sessions.iter_mut().zip(next) {
                        s.next = t;
                    }
                }
                let graph_s = t0.elapsed().as_secs_f64();
                println!(
                    "TIME rep={rep} GRAPH B={b} graph_ms_per_step={:.3} graph_tok_s={:.2} vs_eager_rows={:.3} (first step captures)",
                    1e3 * graph_s / timing_steps as f64,
                    (b * timing_steps) as f64 / graph_s,
                    rows_s / graph_s
                );
            }
            if b == 4 && !tp_ep {
                let mut sessions: Vec<Session> =
                    prompts.iter().map(|p| prime(&gpu, p, capacity)).collect();
                let mut ws = [
                    gpu.alloc_rows_state(2).expect("group 0 workspace"),
                    gpu.alloc_rows_state(2).expect("group 1 workspace"),
                ];
                let (a, c) = sessions.split_at_mut(2);
                let mut groups = [a, c];
                drain(&gpu);
                let t0 = Instant::now();
                for (g, group) in groups.iter_mut().enumerate() {
                    let toks: Vec<u32> = group.iter().map(|s| s.next).collect();
                    let mut states: Vec<&mut DecodeState> =
                        group.iter_mut().map(|s| &mut s.state).collect();
                    gpu.decode_rows_enqueue(&toks, &mut states, &mut ws[g], false)
                        .expect("enqueue");
                }
                for step in 0..timing_steps {
                    for (g, group) in groups.iter_mut().enumerate() {
                        gpu.decode_rows_wait(&mut ws[g]).expect("wait");
                        let mut states: Vec<&mut DecodeState> =
                            group.iter_mut().map(|s| &mut s.state).collect();
                        let (_, next) = gpu
                            .decode_rows_complete(&mut states, &mut ws[g])
                            .expect("complete");
                        drop(states);
                        for (s, t) in group.iter_mut().zip(next) {
                            s.next = t;
                        }
                        if step + 1 < timing_steps {
                            let toks: Vec<u32> = group.iter().map(|s| s.next).collect();
                            let mut states: Vec<&mut DecodeState> =
                                group.iter_mut().map(|s| &mut s.state).collect();
                            gpu.decode_rows_enqueue(&toks, &mut states, &mut ws[g], false)
                                .expect("enqueue");
                        }
                    }
                }
                let piped_s = t0.elapsed().as_secs_f64();
                let tokens = (4 * timing_steps) as f64;
                println!(
                    "TIME rep={rep} PIPELINED 2x2 ms_per_group_step={:.3} tok_s={:.2} vs_rows4={:.3}",
                    1e3 * piped_s / (2 * timing_steps) as f64,
                    tokens / piped_s,
                    rows_s / piped_s
                );
            }
            let tokens = (b * timing_steps) as f64;
            println!(
                "TIME rep={rep} B={b} one_row_ms_per_token={:.3} rows_ms_per_step={:.3} one_row_tok_s={:.2} rows_tok_s={:.2} speedup={:.3}",
                1e3 * solo_s / tokens,
                1e3 * rows_s / timing_steps as f64,
                tokens / solo_s,
                tokens / rows_s,
                solo_s / rows_s
            );
        }
    }
}
