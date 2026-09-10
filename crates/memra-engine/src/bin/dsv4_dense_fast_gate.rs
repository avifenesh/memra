//! Dense-fast same-class qualification on the default split-K/cadence program.
//! OFF eager, OFF replay and ON replay must match at every step.
use memra_engine::dsv4_gpu::{DecodeState, Dsv4Gpu, Dsv4SampleCfg, dsv4_pos_uniform, dsv4_prof_on};
use memra_engine::dsv4_sampler::{Dsv4DeviceSampler, Dsv4Sampler, dsv4_sampler};
use memra_gguf::dsv4_forward::ActQuantVariant;
use memra_tokenizer::Tokenizer;
use sha2::{Digest, Sha256};
use std::{
    ffi::c_int,
    path::{Path, PathBuf},
    time::Instant,
};

const PRIME: usize = 256;
const OUTPUT: usize = 256;
const CAPACITY: usize = PRIME + OUTPUT + 8;
const SOURCE_SHA: &str = "f6e175a6f2588953568746fec0cd43fcd046405f74b5c71ce071fe7f37238ded";
const FP8_NODES: &str = "dsv4_dense_fast_fp8_kernel";
const DOT_NODES: &str = "dsv4_dense_fast_dots_kernel";
const CONTROL_FP8: &str = "dsv4_dense_exact_tail_fp8_kernel";
const CONTROL_DOTS: &str = "dsv4_dense_exact_tail_dots_kernel";

// Existing gate-only C ABI. Selection runs only on this host thread before
// capture; captured functions never read it. No runtime/serving interface added.
unsafe extern "C" {
    fn memra_dsv4_dense_fast_set_for_gate(enabled: c_int) -> c_int;
    fn memra_dsv4_dense_fast_counts_for_gate(fp8: *mut u64, dots: *mut u64) -> c_int;
}
fn enqueues() -> [u64; 2] {
    let mut c = [0u64; 2];
    let rc = unsafe { memra_dsv4_dense_fast_counts_for_gate(&mut c[0], &mut c[1]) };
    assert_eq!(rc, 0, "dense enqueue read");
    c
}
fn drain(gpu: &Dsv4Gpu) {
    for stage in &gpu.stages {
        stage.gpu.stream().synchronize().expect("both-rank drain");
    }
}
fn set_dense(on: bool) {
    assert_eq!(
        unsafe { memra_dsv4_dense_fast_set_for_gate(i32::from(on)) },
        0
    );
}
fn select(gpu: &Dsv4Gpu, on: bool) {
    drain(gpu);
    set_dense(on);
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Program {
    cadence: bool,
    dense: bool,
}
// The two complete programs are fixed before model creation. Only the enqueue
// selector is chosen before a new state's first capture; no retained graph reads it.
const PROGRAMS: [Program; 2] = [
    Program {
        cadence: true,
        dense: false,
    },
    Program {
        cadence: true,
        dense: true,
    },
];
fn block_arms(reverse: bool) -> [usize; 4] {
    if reverse { [0, 1, 1, 0] } else { [1, 0, 0, 1] }
}
fn sha_f32(row: &[f32]) -> String {
    assert!(row.iter().all(|v| v.is_finite()), "finite final logits");
    let mut h = Sha256::new();
    for v in row {
        h.update(v.to_bits().to_le_bytes());
    }
    format!("{:x}", h.finalize())
}
fn sha_tokens(tokens: &[u32]) -> String {
    let mut h = Sha256::new();
    for v in tokens {
        h.update(v.to_le_bytes());
    }
    format!("{:x}", h.finalize())
}
fn looped(tokens: &[u32]) -> bool {
    (1usize..=32).any(|width| {
        let length = width * 4usize.max(32usize.div_ceil(width));
        tokens
            .windows(length)
            .any(|span| span.chunks_exact(width).all(|c| c == &span[..width]))
    })
}
type Identity = (String, [u64; 2], [u64; 2]);
fn identity(gpu: &Dsv4Gpu, state: &DecodeState) -> Identity {
    let logits = gpu
        .read_decode_logits_for_gate(state)
        .expect("final logits");
    let cache = gpu
        .tp_ep_cache_digest_for_gate(state)
        .expect("cache digest");
    let hidden = gpu
        .tp_ep_hidden_digest_for_gate(state)
        .expect("hidden digest");
    assert_eq!(cache[0], cache[1], "cache rank symmetry");
    assert_eq!(hidden[0], hidden[1], "hidden rank symmetry");
    (sha_f32(&logits), cache, hidden)
}
fn state(gpu: &Dsv4Gpu) -> DecodeState {
    gpu.alloc_decode_state_for_transient(CAPACITY, 1)
        .expect("independent state")
}
fn epochs(gpu: &Dsv4Gpu, before: &[Vec<u32>; 2], steps: u32) {
    let after = gpu
        .full_token_ar_epochs_for_gate()
        .expect("device AR epochs");
    let attention = memra_engine::tp_ar::ar_blocks_for(4096) as usize;
    let expert = memra_engine::tp_ar::ar_blocks_for(6 * 4096) as usize;
    for rank in 0..2 {
        assert_eq!(before[rank].len(), 72);
        assert_eq!(after[rank].len(), 72);
        for block in 0..72 {
            let per_step = 43 * (u32::from(block < attention) + u32::from(block < expert));
            assert_eq!(
                after[rank][block].wrapping_sub(before[rank][block]),
                per_step * steps,
                "AR epoch rank {rank} block {block}"
            );
        }
    }
}
type VariantCounts = [[u64; 4]; 2];
type GraphHashes = [[String; 4]; 2];

fn expected_variants(_on: bool, start: usize, end: usize, commit: bool) -> [u64; 4] {
    let mut expected = [0; 4];
    for position in start..end {
        let slot = if !(position + 1).is_multiple_of(4) {
            0
        } else if (position + 1).is_multiple_of(128) {
            3
        } else {
            2
        };
        expected[slot] += 1;
        expected[1] += u64::from(commit);
    }
    expected
}
fn variant_delta(
    gpu: &Dsv4Gpu,
    state: &DecodeState,
    before: VariantCounts,
    expected: [u64; 4],
) -> VariantCounts {
    let after = gpu
        .full_token_replay_variant_counts_for_gate(state)
        .unwrap();
    for rank in 0..2 {
        for (slot, expected_count) in expected.iter().enumerate() {
            assert_eq!(
                after[rank][slot] - before[rank][slot],
                *expected_count,
                "variant progression rank={rank} slot={slot}"
            );
        }
    }
    after
}
fn captures_once(gpu: &Dsv4Gpu, state: &DecodeState, _on: bool) {
    assert_eq!(
        gpu.full_token_replay_captures_for_gate(state).unwrap(),
        [3, 1]
    );
    for rank in gpu
        .full_token_replay_variant_census_for_gate(state)
        .unwrap()
    {
        for slot in [0, 2, 3] {
            assert_eq!(
                [rank[slot][2], rank[slot][3], rank[slot][4], rank[slot][6]],
                [86, 1, 86, 0],
                "AR/embedding/HC/unsupported census"
            );
        }
        assert_eq!(rank[1][6], 0);
    }
}
fn replay_delta(
    gpu: &Dsv4Gpu,
    state: &DecodeState,
    before: [[u64; 2]; 2],
    steps: u64,
) -> [[u64; 2]; 2] {
    let after = gpu.full_token_replay_counts_for_gate(state).unwrap();
    for r in 0..2 {
        for segment in 0..2 {
            assert_eq!(after[r][segment] - before[r][segment], steps);
        }
    }
    after
}
fn count_kernel(dot: &str, needle: &str) -> usize {
    // CUDA verbose DOT has one ID record per kernel, with symbol and geometry.
    // Counting this record avoids graph labels/edges or repeated pointer text.
    dot.lines()
        .filter(|line| line.trim_start().starts_with("| {ID |") && line.contains(needle))
        .count()
}
fn dense_census(gpu: &Dsv4Gpu, state: &DecodeState, on: bool, dir: &Path) -> GraphHashes {
    captures_once(gpu, state, on);
    std::fs::create_dir_all(dir).expect("graph directory");
    gpu.dump_full_token_replay_for_gate(state, dir)
        .expect("public graph dump");
    let mut hashes: GraphHashes = Default::default();
    for (rank, row) in hashes.iter_mut().enumerate() {
        for (segment, hash) in row.iter_mut().enumerate() {
            let path = dir.join(format!("full-token-rank{rank}-segment{segment}.dot"));
            let dot = std::fs::read_to_string(path).expect("DOT");
            let expected = if segment != 1 {
                [494, 253]
            } else if rank == 1 {
                [0, 2]
            } else {
                [0, 0]
            };
            let split = if segment != 1 { 86 } else { 0 };
            assert_eq!(
                count_kernel(&dot, "moe_m1_graph_splitk_partial_kernel"),
                split
            );
            assert_eq!(
                count_kernel(&dot, "moe_m1_graph_splitk_reduce_kernel"),
                split
            );
            let candidate = [count_kernel(&dot, FP8_NODES), count_kernel(&dot, DOT_NODES)];
            let control = [
                count_kernel(&dot, CONTROL_FP8),
                count_kernel(&dot, CONTROL_DOTS),
            ];
            assert_eq!(
                candidate,
                if on { expected } else { [0, 0] },
                "candidate rank={rank} segment={segment}"
            );
            assert_eq!(
                control,
                if on { [0, 0] } else { expected },
                "control rank={rank} segment={segment}"
            );
            *hash = format!("{:x}", Sha256::digest(dot.as_bytes()));
            println!(
                "DENSE_CENSUS on={on} rank={rank} segment={segment} candidate={candidate:?} control={control:?} dot_sha256={hash}"
            );
        }
    }
    hashes
}
struct Arm {
    state: DecodeState,
    program: Program,
    graph_hashes: Option<GraphHashes>,
}
impl Arm {
    fn new(gpu: &Dsv4Gpu, prefix: &DecodeState, cfg: Dsv4SampleCfg, program: Program) -> Self {
        assert!(program.cadence);
        let mut state = state(gpu);
        gpu.restore_full_token_prefix_for_gate(&mut state, prefix)
            .expect("initial restore");
        // gpu is boxed at a stable address and outlives every Arm; its weights,
        // numeric controls and runtime configuration stay fixed throughout.
        unsafe { gpu.arm_full_token_replay_mode_for_gate(&mut state, cfg, true) }
            .expect("arm immutable composed program");
        assert_eq!(
            gpu.full_token_replay_captures_for_gate(&state).unwrap(),
            [0, 0]
        );
        Self {
            state,
            program,
            graph_hashes: None,
        }
    }
    fn prepare(&self, gpu: &Dsv4Gpu) -> bool {
        let captures = gpu
            .full_token_replay_captures_for_gate(&self.state)
            .unwrap();
        let first = captures == [0, 0];
        if first {
            select(gpu, self.program.dense);
        } else {
            assert_eq!(captures, [if self.program.cadence { 3 } else { 1 }, 1]);
        }
        first
    }
    fn check_enqueues(&self, before: [u64; 2], first: bool) {
        let after = enqueues();
        let expected = if first && self.program.dense {
            [2964, 1520]
        } else {
            [0, 0]
        };
        assert_eq!(
            [after[0] - before[0], after[1] - before[1]],
            expected,
            "host enqueues describe capture, never graph replay"
        );
    }
    fn census(&mut self, gpu: &Dsv4Gpu, dir: &Path) {
        let hashes = dense_census(gpu, &self.state, self.program.dense, dir);
        if let Some(prior) = &self.graph_hashes {
            assert_eq!(prior, &hashes, "stable retained graphs after reset");
        } else {
            self.graph_hashes = Some(hashes);
        }
    }
}
fn eager_step(
    gpu: &Dsv4Gpu,
    s: &mut DecodeState,
    sampler: &mut Dsv4DeviceSampler,
    cfg: &Dsv4SampleCfg,
    token: u32,
) -> u32 {
    select(gpu, false);
    let before = enqueues();
    gpu.decode_step_device_logits(token, s)
        .expect("eager control forward");
    let next = gpu
        .sample_device_logits(s, sampler, cfg, &[], None)
        .expect("eager control sample");
    assert_eq!(before, enqueues());
    next
}
fn refusal_cells(
    gpu: &Dsv4Gpu,
    prefix: &DecodeState,
    cfg: Dsv4SampleCfg,
    inputs: &[u32],
    program: Program,
    output: &Path,
) {
    let on = program.dense;
    for rank in 0..2 {
        for (position, layer) in [(258usize, 0usize), (259, 0), (383, 21), (511, 42)] {
            let mut failed = Arm::new(gpu, prefix, cfg, program);
            let first = failed.prepare(gpu);
            let before_host = enqueues();
            for &token in &inputs[..position - PRIME] {
                gpu.decode_sample_full_token_for_gate(token, &mut failed.state)
                    .expect("fault setup");
            }
            failed.check_enqueues(before_host, first);
            assert_eq!(
                gpu.full_token_replay_variant_counts_for_gate(&failed.state)
                    .unwrap(),
                [expected_variants(on, PRIME, position, true); 2],
                "refusal setup variants"
            );
            failed.census(
                gpu,
                &output.join(format!("fault-{}-{rank}-{position}", usize::from(on))),
            );
            let cache = gpu.tp_ep_cache_digest_for_gate(&failed.state).unwrap();
            let counts = gpu
                .full_token_replay_counts_for_gate(&failed.state)
                .unwrap();
            let variants = gpu
                .full_token_replay_variant_counts_for_gate(&failed.state)
                .unwrap();
            let host = enqueues();
            let code = 40043 + rank as i32;
            gpu.arm_attention_tp_join_refusal_for_gate(layer, rank, code)
                .unwrap();
            let error = gpu
                .decode_sample_full_token_for_gate(inputs[position - PRIME], &mut failed.state)
                .unwrap_err();
            assert!(error.contains("one-shot reduction refused"), "{error}");
            let mut words = [0, 0];
            words[rank] = code;
            assert_eq!(gpu.tp_ep_ar_refusal_words().unwrap(), words);
            assert_eq!(failed.state.pos, position);
            assert_eq!(
                gpu.tp_ep_cache_digest_for_gate(&failed.state).unwrap(),
                cache
            );
            let after = gpu
                .full_token_replay_counts_for_gate(&failed.state)
                .unwrap();
            for r in 0..2 {
                assert_eq!(after[r], [counts[r][0] + 1, counts[r][1]], "refusal commit");
            }
            let after_variants = variant_delta(
                gpu,
                &failed.state,
                variants,
                expected_variants(on, position, position + 1, false),
            );
            assert!(
                gpu.decode_sample_full_token_for_gate(inputs[position - PRIME], &mut failed.state)
                    .unwrap_err()
                    .contains("unfinished transaction")
            );
            assert!(
                gpu.restore_full_token_prefix_for_gate(&mut failed.state, prefix)
                    .is_err(),
                "reset bypassed quarantine"
            );
            assert_eq!(
                gpu.full_token_replay_counts_for_gate(&failed.state)
                    .unwrap(),
                after
            );
            assert_eq!(
                gpu.full_token_replay_variant_counts_for_gate(&failed.state)
                    .unwrap(),
                after_variants,
                "retry ran a variant"
            );
            assert_eq!(host, enqueues());
            captures_once(gpu, &failed.state, on);
            println!(
                "REFUSAL on={on} rank={rank} layer={layer} position={position} cache_unchanged=true position_unchanged=true no_commit=true retry_quarantined=true"
            );
            gpu.set_tp_ep_ar_refusal_words_for_gate([0, 0]).unwrap();
        }
    }
}
#[allow(clippy::too_many_arguments)]
fn run(
    gpu: &Dsv4Gpu,
    prompt: &[u32],
    tokenizer: &Tokenizer,
    output: &Path,
    programs: [Program; 2],
    cfg: Dsv4SampleCfg,
    reverse: bool,
    qualify_only: bool,
) {
    select(gpu, false);
    let mut prefix = state(gpu);
    gpu.prefill_with_cache_chunked(&prompt[..1], &mut prefix, 1)
        .expect("prime first");
    for &token in &prompt[1..PRIME] {
        gpu.decode_step_device_logits(token, &mut prefix)
            .expect("prime");
    }
    let mut sampler = gpu.device_sampler().unwrap();
    let first = gpu
        .sample_device_logits(&prefix, &mut sampler, &cfg, &[], None)
        .expect("initial carry draw");
    assert_eq!(prefix.pos, PRIME);
    assert_eq!(gpu.tp_ep_ar_refusal_words().unwrap(), [0, 0]);
    let prefix_identity = identity(gpu, &prefix);
    let mut eager = state(gpu);
    gpu.restore_full_token_prefix_for_gate(&mut eager, &prefix)
        .unwrap();
    let mut control = Arm::new(gpu, &prefix, cfg, programs[0]);
    let mut candidate = Arm::new(gpu, &prefix, cfg, programs[1]);
    let mut inputs = Vec::with_capacity(OUTPUT);
    let mut carry = first;
    for step in 0..OUTPUT {
        assert_ne!(carry, tokenizer.eos_id(), "correctness early EOS");
        inputs.push(carry);
        let before_epochs = gpu.full_token_ar_epochs_for_gate().unwrap();
        let next = eager_step(gpu, &mut eager, &mut sampler, &cfg, carry);
        epochs(gpu, &before_epochs, 1);
        assert_eq!(eager.pos, PRIME + step + 1);
        for arm in [&mut control, &mut candidate] {
            let capturing = arm.prepare(gpu);
            let host = enqueues();
            let arm_epochs = gpu.full_token_ar_epochs_for_gate().unwrap();
            let actual = gpu
                .decode_sample_full_token_for_gate(carry, &mut arm.state)
                .expect("correctness full replay");
            assert_eq!(
                actual, next,
                "first/changing sample step={step} on={}",
                arm.program.dense
            );
            arm.check_enqueues(host, capturing);
            epochs(gpu, &arm_epochs, 1);
            assert_eq!(arm.state.pos, PRIME + step + 1);
            assert_eq!(
                identity(gpu, &arm.state),
                identity(gpu, &eager),
                "logits/cache/hidden step={step}"
            );
            assert_eq!(gpu.tp_ep_ar_refusal_words().unwrap(), [0, 0]);
            assert_eq!(
                gpu.full_token_replay_counts_for_gate(&arm.state).unwrap(),
                [[step as u64 + 1; 2]; 2]
            );
            captures_once(gpu, &arm.state, arm.program.dense);
            assert_eq!(
                gpu.full_token_replay_variant_counts_for_gate(&arm.state)
                    .unwrap(),
                [expected_variants(arm.program.dense, PRIME, PRIME + step + 1, true); 2]
            );
            if step == 0 {
                arm.census(
                    gpu,
                    &output.join(if arm.program.dense {
                        "qual-on"
                    } else {
                        "qual-off"
                    }),
                );
            }
        }
        carry = next;
        if step == 0 || (PRIME + step + 1).is_multiple_of(4) {
            println!(
                "EXACT position={} next_token={carry} eager_off_graph_off_graph_on_identical=true",
                PRIME + step + 1
            );
        }
    }
    assert!(!looped(&inputs));
    let expected_tokens = sha_tokens(&inputs);
    let expected_identity = identity(gpu, &eager);
    let expected_next = carry;
    // Restore into the SAME captured allocations, then replay the changing tape
    // once more to prove stable reset independently of the timed rows.
    for arm in [&mut control, &mut candidate] {
        gpu.restore_full_token_prefix_for_gate(&mut arm.state, &prefix)
            .unwrap();
        assert_eq!(identity(gpu, &arm.state), prefix_identity);
        let counts = gpu.full_token_replay_counts_for_gate(&arm.state).unwrap();
        let variants = gpu
            .full_token_replay_variant_counts_for_gate(&arm.state)
            .unwrap();
        let before_epochs = gpu.full_token_ar_epochs_for_gate().unwrap();
        let host = enqueues();
        let mut token = first;
        for &expected in &inputs {
            assert_eq!(token, expected);
            token = gpu
                .decode_sample_full_token_for_gate(token, &mut arm.state)
                .unwrap();
        }
        assert_eq!(token, expected_next);
        assert_eq!(identity(gpu, &arm.state), expected_identity);
        assert_eq!(enqueues(), host);
        replay_delta(gpu, &arm.state, counts, OUTPUT as u64);
        variant_delta(
            gpu,
            &arm.state,
            variants,
            expected_variants(arm.program.dense, PRIME, PRIME + OUTPUT, true),
        );
        epochs(gpu, &before_epochs, OUTPUT as u32);
        arm.census(
            gpu,
            &output.join(if arm.program.dense {
                "qual-reset-on"
            } else {
                "qual-reset-off"
            }),
        );
    }
    drop(control);
    drop(candidate);
    drop(eager);
    refusal_cells(gpu, &prefix, cfg, &inputs, programs[0], output);
    refusal_cells(gpu, &prefix, cfg, &inputs, programs[1], output);
    println!("PASS dense-fast model correctness, two retained resets, and 16 live refusal cells");
    if qualify_only {
        select(gpu, false);
        println!("QUALIFY_PASS identity_steps=256 arms=2 refusal_cells=16");
        return;
    }

    // BOTH scored arms are new and uncaptured, independent of qualification.
    // First ON row and first OFF row each include their actual graph-capture cost.
    let mut arms = [
        Arm::new(gpu, &prefix, cfg, programs[0]),
        Arm::new(gpu, &prefix, cfg, programs[1]),
    ];
    let mut walls = [0u128; 2];
    let mut rates = [Vec::new(), Vec::new()];
    let mut first_capture_rows = [0usize; 2];
    let blocks = block_arms(reverse);
    let schedule = if reverse {
        "OFF ON ON OFF"
    } else {
        "ON OFF OFF ON"
    };
    println!(
        "PROTOCOL blocks={schedule:?} rows_per_block=5 rows=20 prime=256 output=256 both_full_replay=true device_sampler=true diet=true splitk=graph cadence_A=true cadence_B=true dense_fast_A=false dense_fast_B=true gu_n32=false first_capture_each_arm_inside_timing=true initial_carry_outside_timing=true final_next_draw_inside_timing=true timing_scope=sample_plus_forward_envelope control_hash_provenance=host_reconstructed_intended_sequence"
    );
    for row in 0..20 {
        let index = blocks[row / 5];
        let on = programs[index].dense;
        let arm_name = if on { "B" } else { "A" };
        let active = &mut arms[index];
        gpu.restore_full_token_prefix_for_gate(&mut active.state, &prefix)
            .expect("stable row restore");
        assert_eq!(identity(gpu, &active.state), prefix_identity);
        let first_capture = active.prepare(gpu);
        let arm_row = rates[index].len();
        assert_eq!(
            first_capture,
            arm_row == 0,
            "exactly the first arm row captures"
        );
        let host = enqueues();
        let before_counts = gpu
            .full_token_replay_counts_for_gate(&active.state)
            .unwrap();
        let before_variants = gpu
            .full_token_replay_variant_counts_for_gate(&active.state)
            .unwrap();
        let before_epochs = gpu.full_token_ar_epochs_for_gate().unwrap();
        let mut carry = first;
        let mut tokens = Vec::with_capacity(OUTPUT);
        let start = Instant::now();
        for _ in 0..OUTPUT {
            assert_ne!(carry, tokenizer.eos_id(), "early EOS row");
            tokens.push(carry);
            carry = gpu
                .decode_sample_full_token_for_gate(carry, &mut active.state)
                .expect("scored replay");
        }
        drain(gpu);
        let wall = start.elapsed().as_nanos();
        assert!(!looped(&tokens));
        assert_eq!(sha_tokens(&tokens), expected_tokens);
        assert_eq!(carry, expected_next);
        assert_eq!(active.state.pos, 512);
        assert_eq!(identity(gpu, &active.state), expected_identity);
        assert_eq!(gpu.tp_ep_ar_refusal_words().unwrap(), [0, 0]);
        active.check_enqueues(host, first_capture);
        let counts = replay_delta(gpu, &active.state, before_counts, OUTPUT as u64);
        epochs(gpu, &before_epochs, OUTPUT as u32);
        captures_once(gpu, &active.state, on);
        let variants = variant_delta(
            gpu,
            &active.state,
            before_variants,
            expected_variants(on, PRIME, PRIME + OUTPUT, true),
        );
        let capture_counts = gpu
            .full_token_replay_captures_for_gate(&active.state)
            .unwrap();
        // Dump once after first capture and again after the arm's final reset.
        // All rows still assert captures/replays/epochs; no repeated DOT I/O is timed.
        if first_capture || rates[index].len() == 9 {
            active.census(
                gpu,
                &output.join(format!("row-{row:02}-{}", if on { "on" } else { "off" })),
            );
        }
        first_capture_rows[index] += usize::from(first_capture);
        walls[index] += wall;
        let rate = OUTPUT as f64 * 1e9 / wall as f64;
        rates[index].push(rate);
        let mut h = Sha256::new();
        for (i, &token) in tokens.iter().enumerate() {
            h.update((u64::from(token) | (((PRIME + i) as u64) << 32)).to_le_bytes());
            h.update(
                dsv4_pos_uniform(cfg.seed, PRIME + i + 1)
                    .to_bits()
                    .to_le_bytes(),
            );
            h.update(0u64.to_le_bytes());
        }
        println!(
            r#"MEASURE {{"row":{row},"arm_row":{arm_row},"arm":"{arm_name}","cadence_on":true,"dense_fast_on":{on},"sampler":"device","generated_tokens":256,"decode_wall_ns":{wall},"decode_tok_s":{rate},"first_capture":{first_capture},"timing_scope":"sample_plus_forward_envelope","eligible":true,"full_replay":true,"splitk":true,"generated_sha256":"{expected_tokens}","final_logits_sha256":"{}","final_cache_digest":{:?},"final_hidden_digest":{:?},"device_replays":{counts:?},"captures":{capture_counts:?},"variant_device_counts":{variants:?},"control_sha256":"{:x}"}}"#,
            expected_identity.0,
            expected_identity.1,
            expected_identity.2,
            h.finalize()
        );
    }
    assert_eq!(first_capture_rows, [1, 1]);
    assert_eq!([rates[0].len(), rates[1].len()], [10, 10]);
    let pooled = walls.map(|w| 10.0 * OUTPUT as f64 * 1e9 / w as f64);
    let means = rates.map(|r| r.iter().sum::<f64>() / r.len() as f64);
    println!(
        r#"SUMMARY {{"rows":20,"off_pooled_tok_s":{},"on_pooled_tok_s":{},"off_mean_tok_s":{},"on_mean_tok_s":{},"pooled_delta_pct":{},"mean_delta_pct":{},"first_capture_rows_per_arm":[1,1],"identity":true,"decision":"return to root; no automatic merge or promotion"}}"#,
        pooled[0],
        pooled[1],
        means[0],
        means[1],
        100.0 * (pooled[1] / pooled[0] - 1.0),
        100.0 * (means[1] / means[0] - 1.0)
    );
    select(gpu, false);
}
fn main() {
    // Freeze this historical instrument independently of the newer defaults.
    // This is process startup, before any model or worker threads exist.
    unsafe {
        std::env::set_var("MEMRA_DSV4_HC_DOT_SPLIT", "0");
        std::env::set_var("MEMRA_DSV4_DENSE_FAST", "0");
        std::env::set_var("MEMRA_DSV4_NORM_FUSE", "0");
        std::env::set_var("MEMRA_DSV4_NORM_FUSE2", "0");
        std::env::set_var("MEMRA_DSV4_NORM2_WIDE", "0");
        // Gate-only AR phase instrument: pinned off here so no other bin can inherit
        // an exported instrument or null collective from the environment.
        std::env::set_var("MEMRA_DSV4_AR_PHASE", "0");
    }

    let args: Vec<_> = std::env::args().collect();
    assert!(
        args.len() == 4
            || (args.len() == 5
                && matches!(args[4].as_str(), "--reverse" | "--capture" | "--qualify")),
        "usage: dsv4_dense_fast_gate <model-dir> <source.txt> <new-output-dir> [--reverse|--capture|--qualify]"
    );
    assert!(!dsv4_prof_on(), "unprofiled sampled envelope only");
    for (name, value) in [
        ("MEMRA_DSV4_DECODE_PATH", "device"),
        ("MEMRA_DSV4_EXPERT_ARM", "native"),
        ("MEMRA_DSV4_DENSE_ARM", "fp8"),
        ("MEMRA_DSV4_DOTS_ARM", "f32x"),
        ("MEMRA_DSV4_EP", "pair"),
        ("MEMRA_DSV4_MOE_PROGRAM", "matrix"),
        ("MEMRA_DSV4_GROUPED_ROUTE", "device"),
        ("MEMRA_DSV4_VERIFY_TOPK", "device"),
        ("MEMRA_DSV4_PREFILL_MOE", "reference"),
        ("MEMRA_DSV4_DRAFTER", "off"),
        ("MEMRA_DSV4_SMALL_KERNEL_DIET", "1"),
        ("MEMRA_MOE_F16G", "2"),
        ("MEMRA_F16G_SK", "32"),
    ] {
        assert_eq!(
            std::env::var(name).as_deref(),
            Ok(value),
            "requires {name}={value}"
        );
    }
    assert_eq!(dsv4_sampler().unwrap(), Dsv4Sampler::Device);
    memra_engine::set_moe_m1_splitk_for_gate(false);
    memra_engine::set_moe_m1_graph_splitk_for_gate(true);
    assert!(memra_engine::moe_m1_graph_splitk_on());
    // Default-ON paired-fetch entries would change this control's captured
    // class, so this historical gate pins the base graph split-K partial.
    memra_engine::set_moe_m1_splitk_fast_for_gate(false);
    assert!(!memra_engine::moe_m1_splitk_fast_on());
    memra_engine::dsv4_gpu::set_dense_exact_tail_for_gate(true).unwrap();
    let programs = PROGRAMS;
    let cfg = Dsv4SampleCfg {
        temperature: 1.0,
        top_p: 1.0,
        top_k: 0,
        seed: 20260907,
    };
    set_dense(false);
    println!(
        "COMPOSE_PROGRAMS before_model_creation=true arms={programs:?} sampler=device sampling_fixed=true dense_selector=capture_only"
    );
    // GU N32 is absent from the pinned source; the controller must bind it.
    let source = std::fs::read_to_string(&args[2]).expect("source tape");
    assert_eq!(
        format!("{:x}", Sha256::digest(source.as_bytes())),
        SOURCE_SHA
    );
    let tokenizer = Tokenizer::from_hf_dir(Path::new(&args[1])).expect("tokenizer");
    let prompt = tokenizer.encode(
        &format!("Review this inference engine source:\n\n{source}"),
        true,
    );
    assert!(prompt.len() >= PRIME);
    let output = PathBuf::from(&args[3]);
    std::fs::create_dir(&output).expect("new output directory");
    Dsv4Gpu::set_tp_ep_topology_for_gate(true);
    Dsv4Gpu::set_attention_tp_for_gate(true);
    let gpu = Box::new(
        Dsv4Gpu::load(
            Path::new(&args[1]),
            &[0, 1],
            ActQuantVariant::RefFp8Round,
            PRIME + OUTPUT + 32,
        )
        .expect("pinned TP2 model"),
    );
    assert!(gpu.topology().is_tp_ep());
    assert_eq!(gpu.topology().layers, 43);
    assert!(gpu.attention_tp_geometry().is_some() && gpu.small_kernel_diet_enabled());
    gpu.set_grouped_route_validation_for_gate(false);
    gpu.set_grouped_mirror_validation_for_gate(false);
    gpu.set_grouped_gu_fuse_for_gate(true);
    gpu.set_grouped_m1_tc_for_gate(true);
    memra_engine::set_moe_f16g_gu_m1_tc_for_gate(true);
    memra_engine::set_moe_f16g_gu_half2_for_gate(true);
    memra_engine::set_moe_f16g_down_m1_half2_for_gate(true);
    gpu.set_dense_wo_a_grouped_for_gate(false);
    gpu.set_index_topk_radix_for_gate(true);
    if args.get(4).is_some_and(|v| v == "--capture") {
        capture_operands(&gpu, &prompt, &output, cfg);
        return;
    }
    run(
        &gpu,
        &prompt[..PRIME],
        &tokenizer,
        &output,
        programs,
        cfg,
        args.get(4).is_some_and(|arg| arg == "--reverse"),
        args.get(4).is_some_and(|arg| arg == "--qualify"),
    );
}

// This callback is installed only for --capture, outside any CUDA graph.
// Files contain the actual runtime operands, with their rank/shape/dtype header.
use std::{collections::BTreeSet, ffi::c_void, io::Write, sync::Mutex};
struct Capture {
    dir: PathBuf,
    seen: BTreeSet<(i32, i32, i32, i32)>,
    stream_ranks: [(usize, i32); 2],
}
static CAPTURE: Mutex<Option<Capture>> = Mutex::new(None);
type Observer = unsafe extern "C" fn(
    i32,
    *const c_void,
    *const f32,
    i32,
    *const c_void,
    i32,
    i32,
    *mut c_void,
) -> i32;
unsafe extern "C" {
    fn memra_dsv4_dense_fast_observe_for_gate(observer: Option<Observer>);
    fn cudaGetDevice(device: *mut i32) -> i32;
    fn cudaStreamGetDevice(stream: *mut c_void, device: *mut i32) -> i32;
    fn cudaStreamSynchronize(stream: *mut c_void) -> i32;
    fn cudaMemcpy(dst: *mut c_void, src: *const c_void, count: usize, kind: i32) -> i32;
}
fn capture_stream_rank(stream_ranks: &[(usize, i32); 2], stream: usize) -> i32 {
    stream_ranks
        .iter()
        .find(|(handle, _)| *handle == stream)
        .map(|(_, rank)| *rank)
        .expect("capture stream belongs to this model")
}
unsafe extern "C" fn observe(
    kind: i32,
    w: *const c_void,
    sc: *const f32,
    sc_cols: i32,
    x: *const c_void,
    n: i32,
    k: i32,
    stream: *mut c_void,
) -> i32 {
    let result = std::panic::catch_unwind(|| {
        let mut guard = CAPTURE.lock().unwrap();
        let cap = guard.as_mut().expect("capture active");
        // Rank belongs to the supplied stream, not the host thread's ambient
        // device after a paired collective. Check the registered model owner.
        let rank = capture_stream_rank(&cap.stream_ranks, stream as usize);
        let mut stream_device = -1;
        assert_eq!(
            unsafe { cudaStreamGetDevice(stream, &mut stream_device) },
            0
        );
        assert_eq!(stream_device, rank, "stream device versus model rank");
        let key = (rank, kind, n, k);
        if cap.seen.contains(&key) {
            return;
        }
        // SHAPES.md has these 13 shape/type pairs. Head-only shapes are rank 1.
        let wanted = match kind {
            0 => [
                (512, 4096),
                (1024, 4096),
                (2048, 4096),
                (4096, 2048),
                (4096, 4096),
                (8192, 1024),
                (16384, 1024),
            ]
            .contains(&(n, k)),
            1 => [
                (4, 16384),
                (24, 16384),
                (256, 4096),
                (512, 4096),
                (1024, 4096),
            ]
            .contains(&(n, k)),
            2 => (n, k) == (129280, 4096),
            _ => false,
        };
        if !wanted {
            return;
        }
        let mut ambient_device = -1;
        assert_eq!(unsafe { cudaGetDevice(&mut ambient_device) }, 0);
        for ptr in [w, x].into_iter().chain((kind == 0).then_some(sc.cast())) {
            let mut owner: i32 = -1;
            assert_eq!(
                unsafe {
                    cudarc::driver::sys::cuPointerGetAttribute(
                        (&mut owner as *mut i32).cast(),
                        cudarc::driver::sys::CUpointer_attribute::CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL,
                        ptr as cudarc::driver::sys::CUdeviceptr,
                    )
                },
                cudarc::driver::sys::CUresult::CUDA_SUCCESS,
                "capture operand allocation owner"
            );
            assert_eq!(owner, rank, "operand allocation versus stream rank");
        }
        assert_eq!(unsafe { cudaStreamSynchronize(stream) }, 0);
        let path = cap.dir.join(format!("rank{rank}-kind{kind}-n{n}-k{k}.bin"));
        let mut file = std::fs::File::create_new(path).unwrap();
        file.write_all(b"DENSEF01").unwrap();
        for value in [rank, kind, n, k, sc_cols] {
            file.write_all(&value.to_le_bytes()).unwrap();
        }
        let weight_bytes = n as usize
            * k as usize
            * if kind == 0 {
                1
            } else if kind == 1 {
                4
            } else {
                2
            };
        let input_bytes = k as usize * if kind == 0 { 2 } else { 4 };
        let scale_bytes = if kind == 0 {
            ((n + 127) / 128) as usize * sc_cols as usize * 4
        } else {
            0
        };
        for (ptr, len) in [
            (w, weight_bytes),
            (x, input_bytes),
            (sc.cast(), scale_bytes),
        ] {
            if len == 0 {
                continue;
            }
            let mut bytes = vec![0u8; len];
            assert_eq!(
                unsafe { cudaMemcpy(bytes.as_mut_ptr().cast(), ptr, len, 2) },
                0
            );
            file.write_all(&bytes).unwrap();
        }
        file.sync_all().unwrap();
        cap.seen.insert(key);
        println!(
            "CAPTURE rank={rank} kind={kind} n={n} k={k} stream_device={stream_device} ambient_device={ambient_device} operand_owner_verified=true count={}",
            cap.seen.len()
        );
    });
    if result.is_ok() { 0 } else { 40075 }
}
fn capture_operands(gpu: &Dsv4Gpu, prompt: &[u32], output: &Path, cfg: Dsv4SampleCfg) {
    select(gpu, false);
    let stream_ranks = std::array::from_fn(|rank| {
        (
            gpu.stages[rank].gpu.stream().cu_stream() as usize,
            rank as i32,
        )
    });
    assert_ne!(
        stream_ranks[0].0, stream_ranks[1].0,
        "distinct rank streams"
    );
    *CAPTURE.lock().unwrap() = Some(Capture {
        dir: output.to_path_buf(),
        seen: BTreeSet::new(),
        stream_ranks,
    });
    unsafe {
        memra_dsv4_dense_fast_observe_for_gate(Some(observe));
    }
    let mut work = state(gpu);
    gpu.prefill_with_cache_chunked(&prompt[..1], &mut work, 1)
        .unwrap();
    for &token in &prompt[1..8] {
        gpu.decode_step_device_logits(token, &mut work).unwrap();
    }
    let mut sampler = gpu.device_sampler().unwrap();
    gpu.sample_device_logits(&work, &mut sampler, &cfg, &[], None)
        .unwrap();
    unsafe {
        memra_dsv4_dense_fast_observe_for_gate(None);
    }
    let cap = CAPTURE.lock().unwrap().take().unwrap();
    let mut expected = BTreeSet::new();
    for rank in 0..2 {
        for (n, k) in [
            (512, 4096),
            (1024, 4096),
            (2048, 4096),
            (4096, 2048),
            (4096, 4096),
            (8192, 1024),
            (16384, 1024),
        ] {
            expected.insert((rank, 0, n, k));
        }
        for (n, k) in [(24, 16384), (256, 4096), (512, 4096), (1024, 4096)] {
            expected.insert((rank, 1, n, k));
        }
    }
    expected.insert((1, 1, 4, 16384));
    expected.insert((1, 2, 129280, 4096));
    assert_eq!(
        cap.seen, expected,
        "every replay-map shape on its actual rank"
    );
    println!(
        "CAPTURE_PASS real_cases={} dense_fast=false tokens=8",
        cap.seen.len()
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn paired_tail_captures_distinct_stream_owners_under_one_ambient_device() {
        let streams = [(0x1000, 0), (0x2000, 1)];
        let mut keys = BTreeSet::new();
        // Both calls may follow the rank-1 side of a paired collective.
        let ambient_devices = [1, 1];
        for ((stream, _), _ambient) in streams.into_iter().zip(ambient_devices) {
            keys.insert((capture_stream_rank(&streams, stream), 0, 2048, 4096));
        }
        assert_eq!(
            keys,
            BTreeSet::from([(0, 0, 2048, 4096), (1, 0, 2048, 4096)])
        );
    }
    #[test]
    #[should_panic(expected = "capture stream belongs to this model")]
    fn capture_refuses_an_unregistered_stream() {
        capture_stream_rank(&[(0x1000, 0), (0x2000, 1)], 0x3000);
    }
    #[test]
    fn arm_a_forces_dense_fast_off() {
        assert!(!PROGRAMS[0].dense && PROGRAMS[0].cadence);
        assert!(PROGRAMS[1].dense && PROGRAMS[1].cadence);
        unsafe extern "C" {
            fn memra_dsv4_dense_fast_enabled_for_gate() -> i32;
        }
        std::thread::spawn(|| {
            set_dense(true);
            set_dense(PROGRAMS[0].dense);
            assert_eq!(unsafe { memra_dsv4_dense_fast_enabled_for_gate() }, 0);
            set_dense(PROGRAMS[1].dense);
            assert_eq!(unsafe { memra_dsv4_dense_fast_enabled_for_gate() }, 1);
        })
        .join()
        .unwrap();
    }
    #[test]
    fn both_orders_have_ten_rows_each() {
        assert_eq!(block_arms(false), [1, 0, 0, 1]);
        assert_eq!(block_arms(true), [0, 1, 1, 0]);
    }
    #[test]
    fn both_arms_keep_cadence_and_refusal_boundaries() {
        for on in [false, true] {
            assert_eq!(expected_variants(on, 256, 512, true), [192, 256, 62, 2]);
            assert_eq!(expected_variants(on, 258, 259, false), [1, 0, 0, 0]);
            assert_eq!(expected_variants(on, 259, 260, false), [0, 0, 1, 0]);
            assert_eq!(expected_variants(on, 383, 384, false), [0, 0, 0, 1]);
        }
    }
}
