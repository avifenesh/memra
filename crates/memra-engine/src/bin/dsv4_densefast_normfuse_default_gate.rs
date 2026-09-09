//! Actual unset/zero policy engagement for dense-fast plus norm-fuse defaults.
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
const FUSED: &str = "dsv4_norm_rope_f32_fixed_order_kernel";
const REFUSALS: [(usize, usize); 4] = [(258, 0), (259, 0), (383, 21), (511, 42)];
const CONTROL_DOTS: &str = "dsv4_dense_exact_tail_dots_kernel";

// Existing gate-only C ABI. Selection runs only on this host thread before
// capture; captured functions never read it. No runtime/serving interface added.
unsafe extern "C" {
    fn memra_dsv4_hc_dot_split_slices_for_gate() -> c_int;
    fn memra_dsv4_dense_fast_set_for_gate(enabled: c_int) -> c_int;
    fn memra_dsv4_dense_fast_enabled_for_gate() -> c_int;
    fn memra_dsv4_dense_fast_restore_default_for_gate() -> c_int;
    fn memra_dsv4_dense_fast_counts_for_gate(fp8: *mut u64, dots: *mut u64) -> c_int;
}
fn hc_on() -> bool {
    unsafe { memra_dsv4_hc_dot_split_slices_for_gate() != 0 }
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
    gpu.set_norm_fuse_for_gate(on).unwrap();
    assert_eq!(gpu.norm_fuse_enabled_for_gate(), on);
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Program {
    cadence: bool,
    dense: bool,
    norm: bool,
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
fn captures_once(gpu: &Dsv4Gpu, state: &DecodeState, on: bool) {
    assert_eq!(
        gpu.full_token_replay_captures_for_gate(state).unwrap(),
        [3, 1]
    );
    for rank in gpu
        .full_token_replay_variant_census_for_gate(state)
        .unwrap()
    {
        for (slot, baseline) in [(0, 2827), (2, 3226), (3, 3326)] {
            assert_eq!(
                rank[slot][1],
                baseline - if on { 43 } else { 0 } + if hc_on() { 86 } else { 0 },
                "composed kernel census slot={slot}"
            );
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
fn compose_census(gpu: &Dsv4Gpu, state: &DecodeState, on: bool, dir: &Path) -> GraphHashes {
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
                [494, if hc_on() { 167 } else { 253 }]
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
            for name in [
                "dsv4_hc_dot_split_partial_kernel",
                "dsv4_hc_dot_split_reduce_kernel",
            ] {
                assert_eq!(
                    count_kernel(&dot, name),
                    if hc_on() && segment != 1 { 86 } else { 0 }
                );
            }
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
            let fused = count_kernel(&dot, FUSED);
            let norms = count_kernel(&dot, "dsv4_rmsnorm_f32acc_kernel");
            let ropes = count_kernel(&dot, "dsv4_rope_kernel");
            let expected_norm = match segment {
                0 => 129,
                2 => 171,
                3 => 191,
                _ => usize::from(rank == 1),
            };
            let expected_rope = if segment == 1 { 0 } else { 150 };
            let removed = if on && segment != 1 { 43 } else { 0 };
            assert_eq!(fused, removed, "fused rank={rank} segment={segment}");
            assert_eq!(norms, expected_norm - removed, "unfused norms");
            assert_eq!(ropes, expected_rope - removed, "unfused ropes");
            *hash = format!("{:x}", Sha256::digest(dot.as_bytes()));
            println!(
                "COMPOSE_CENSUS on={on} fused={fused} norms={norms} ropes={ropes} rank={rank} segment={segment} candidate={candidate:?} control={control:?} dot_sha256={hash}"
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
        assert_eq!(program.dense, program.norm, "both doors move together");
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
            drain(gpu);
            assert_eq!(
                unsafe { memra_dsv4_dense_fast_restore_default_for_gate() } != 0,
                self.program.dense,
                "actual dense-fast environment policy"
            );
            assert_eq!(
                gpu.restore_norm_fuse_default_for_gate().unwrap(),
                self.program.norm,
                "actual norm-fuse environment policy"
            );
        } else {
            assert_eq!(captures, [if self.program.cadence { 3 } else { 1 }, 1]);
        }
        first
    }
    fn check_enqueues(&self, before: [u64; 2], first: bool) {
        let after = enqueues();
        let expected = if first && self.program.dense {
            [2964, if hc_on() { 1004 } else { 1520 }]
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
        let hashes = compose_census(gpu, &self.state, self.program.dense, dir);
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
        for (position, layer) in REFUSALS {
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
fn run(
    gpu: &Dsv4Gpu,
    prompt: &[u32],
    tokenizer: &Tokenizer,
    output: &Path,
    program: Program,
    cfg: Dsv4SampleCfg,
) {
    select(gpu, false);
    let mut prefix = state(gpu);
    gpu.prefill_with_cache_chunked(&prompt[..1], &mut prefix, 1)
        .expect("prime first");
    for &token in &prompt[1..PRIME] {
        gpu.decode_step_device_logits(token, &mut prefix)
            .expect("prime control");
    }
    let mut sampler = gpu.device_sampler().unwrap();
    let first = gpu
        .sample_device_logits(&prefix, &mut sampler, &cfg, &[], None)
        .expect("initial carry");
    assert_eq!(prefix.pos, PRIME);
    assert_eq!(gpu.tp_ep_ar_refusal_words().unwrap(), [0, 0]);
    let prefix_identity = identity(gpu, &prefix);
    let mut eager = state(gpu);
    gpu.restore_full_token_prefix_for_gate(&mut eager, &prefix)
        .unwrap();
    let mut selected = Arm::new(gpu, &prefix, cfg, program);
    let mut inputs = Vec::with_capacity(OUTPUT);
    let mut carry = first;
    for step in 0..OUTPUT {
        assert_ne!(carry, tokenizer.eos_id(), "default correctness early EOS");
        inputs.push(carry);
        let before = gpu.full_token_ar_epochs_for_gate().unwrap();
        let next = eager_step(gpu, &mut eager, &mut sampler, &cfg, carry);
        epochs(gpu, &before, 1);
        let capture = selected.prepare(gpu);
        let host = enqueues();
        let before = gpu.full_token_ar_epochs_for_gate().unwrap();
        let actual = gpu
            .decode_sample_full_token_for_gate(carry, &mut selected.state)
            .expect("environment replay");
        assert_eq!(actual, next, "default/eager sampled identity step={step}");
        selected.check_enqueues(host, capture);
        epochs(gpu, &before, 1);
        assert_eq!(selected.state.pos, PRIME + step + 1);
        assert_eq!(selected.state.pos, eager.pos);
        assert_eq!(
            identity(gpu, &selected.state),
            identity(gpu, &eager),
            "default/eager state step={step}"
        );
        assert_eq!(gpu.tp_ep_ar_refusal_words().unwrap(), [0, 0]);
        assert_eq!(
            gpu.full_token_replay_counts_for_gate(&selected.state)
                .unwrap(),
            [[step as u64 + 1; 2]; 2]
        );
        assert_eq!(
            gpu.full_token_replay_variant_counts_for_gate(&selected.state)
                .unwrap(),
            [expected_variants(program.dense, PRIME, PRIME + step + 1, true); 2]
        );
        captures_once(gpu, &selected.state, program.norm);
        if step == 0 {
            selected.census(gpu, &output.join("default-qual"));
        }
        carry = next;
        println!(
            r#"DEFAULT_EXACT {{"step":{step},"position":{},"eager_identity":true,"token":{carry},"dense_fast_on":{},"norm_fuse_on":{}}}"#,
            selected.state.pos, program.dense, program.norm
        );
    }
    assert!(!looped(&inputs));
    let expected_tokens = sha_tokens(&inputs);
    let expected_identity = identity(gpu, &eager);
    let expected_next = carry;
    drop(selected);
    drop(eager);
    refusal_cells(gpu, &prefix, cfg, &inputs, program, output);
    println!(
        "PASS environment-selected identity steps=256 refusal_cells=8; five sanity rows follow"
    );

    // Fresh environment-armed state; first forward/commit captures stay timed.
    let mut active = Arm::new(gpu, &prefix, cfg, program);
    let mut total_wall = 0u128;
    for row in 0..5 {
        gpu.restore_full_token_prefix_for_gate(&mut active.state, &prefix)
            .unwrap();
        assert_eq!(identity(gpu, &active.state), prefix_identity);
        let first_capture = active.prepare(gpu);
        assert_eq!(first_capture, row == 0);
        let host = enqueues();
        let counts = gpu
            .full_token_replay_counts_for_gate(&active.state)
            .unwrap();
        let variants = gpu
            .full_token_replay_variant_counts_for_gate(&active.state)
            .unwrap();
        let before = gpu.full_token_ar_epochs_for_gate().unwrap();
        let mut carry = first;
        let mut tokens = Vec::with_capacity(OUTPUT);
        let start = Instant::now();
        for _ in 0..OUTPUT {
            assert_ne!(carry, tokenizer.eos_id(), "sanity early EOS");
            tokens.push(carry);
            carry = gpu
                .decode_sample_full_token_for_gate(carry, &mut active.state)
                .expect("default sanity replay");
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
        let counts = replay_delta(gpu, &active.state, counts, OUTPUT as u64);
        let variants = variant_delta(
            gpu,
            &active.state,
            variants,
            expected_variants(program.dense, PRIME, PRIME + OUTPUT, true),
        );
        epochs(gpu, &before, OUTPUT as u32);
        captures_once(gpu, &active.state, program.norm);
        if row == 0 || row == 4 {
            active.census(gpu, &output.join(format!("default-row-{row}")));
        }
        let mut control_hash = Sha256::new();
        for (i, &token) in tokens.iter().enumerate() {
            control_hash.update((u64::from(token) | (((PRIME + i) as u64) << 32)).to_le_bytes());
            control_hash.update(
                dsv4_pos_uniform(cfg.seed, PRIME + i + 1)
                    .to_bits()
                    .to_le_bytes(),
            );
            control_hash.update(0u64.to_le_bytes());
        }
        total_wall += wall;
        println!(
            r#"DEFAULT_SANITY {{"row":{row},"rows":5,"dense_fast_on":{},"norm_fuse_on":{},"generated_tokens":256,"decode_wall_ns":{wall},"decode_tok_s":{},"first_capture":{first_capture},"timing_scope":"sample_plus_forward_envelope","sanity_only":true,"eligible":true,"identity":true,"generated_sha256":"{expected_tokens}","final_logits_sha256":"{}","final_cache_digest":{:?},"final_hidden_digest":{:?},"device_replays":{counts:?},"variant_device_counts":{variants:?},"control_sha256":"{:x}","control_hash_provenance":"host_reconstructed_intended_sequence"}}"#,
            program.dense,
            program.norm,
            256e9 / wall as f64,
            expected_identity.0,
            expected_identity.1,
            expected_identity.2,
            control_hash.finalize()
        );
    }
    println!(
        r#"SUMMARY {{"rows":5,"dense_fast_on":{},"norm_fuse_on":{},"pooled_tok_s":{},"identity":true,"first_capture_rows":1,"sanity_only":true,"performance_claim":false,"environment_arming":true}}"#,
        program.dense,
        program.norm,
        1280e9 / total_wall as f64
    );
}

fn main() {
    let args: Vec<_> = std::env::args().collect();
    assert_eq!(
        args.len(),
        4,
        "usage: dsv4_densefast_normfuse_default_gate <model-dir> <source.txt> <new-output-dir>"
    );
    assert!(!dsv4_prof_on(), "unprofiled sampled envelope only");
    assert_ne!(
        std::env::var("MEMRA_DSV4_BENCH_PROFILE").as_deref(),
        Ok("1"),
        "profiling rejected"
    );
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
    let dense_env = std::env::var("MEMRA_DSV4_DENSE_FAST").ok();
    let norm_env = std::env::var("MEMRA_DSV4_NORM_FUSE").ok();
    assert!(
        (dense_env.is_none() && norm_env.is_none())
            || (dense_env.as_deref() == Some("0") && norm_env.as_deref() == Some("0")),
        "engagement requires both unset or both explicit zero"
    );
    // This bin observes the dense-fast and norm-fuse policy on purpose. The
    // separate default-OFF norm2 door is not part of that observation and must
    // not be inherited from the caller.
    assert!(
        matches!(
            std::env::var("MEMRA_DSV4_NORM_FUSE2").as_deref(),
            Err(_) | Ok("0")
        ),
        "default OFF required: MEMRA_DSV4_NORM_FUSE2"
    );
    let dense_initial = unsafe { memra_dsv4_dense_fast_enabled_for_gate() } != 0;
    assert_eq!(
        dense_initial,
        dense_env.is_none(),
        "initial policy before any gate override"
    );
    let cfg = Dsv4SampleCfg {
        temperature: 1.0,
        top_p: 1.0,
        top_k: 0,
        seed: 20260907,
    };
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
    let program = Program {
        cadence: true,
        dense: dense_initial,
        norm: gpu.norm_fuse_enabled_for_gate(),
    };
    assert_eq!(
        program.norm,
        norm_env.is_none(),
        "loaded norm policy before override"
    );
    println!(
        "DEFAULT_POLICY before_override=true dense_fast_on={} norm_fuse_on={} rows=5",
        program.dense, program.norm
    );
    run(&gpu, &prompt[..PRIME], &tokenizer, &output, program, cfg);
}

#[cfg(test)]
mod tests {
    use super::*;
    // Invoked in fresh processes by the parent test below, so environment and
    // C++ thread-local initialization cannot leak between OFF and unset.
    #[test]
    fn dense_policy_child() {
        let expected = std::env::var("MEMRA_DSV4_DENSE_FAST").map_or(true, |v| v == "1");
        assert_eq!(
            unsafe { memra_dsv4_dense_fast_enabled_for_gate() } != 0,
            expected
        );
        set_dense(!expected);
        assert_eq!(
            unsafe { memra_dsv4_dense_fast_restore_default_for_gate() } != 0,
            expected
        );
    }
    #[test]
    fn dense_unset_and_zero_use_actual_process_policy() {
        for value in [None, Some("0")] {
            let mut cmd = std::process::Command::new(std::env::current_exe().unwrap());
            cmd.args(["--exact", "tests::dense_policy_child", "--nocapture"]);
            if let Some(value) = value {
                cmd.env("MEMRA_DSV4_DENSE_FAST", value);
            } else {
                cmd.env_remove("MEMRA_DSV4_DENSE_FAST");
            }
            let output = cmd.output().unwrap();
            assert!(
                output.status.success(),
                "{}",
                String::from_utf8_lossy(&output.stderr)
            );
            assert!(String::from_utf8_lossy(&output.stdout).contains("1 passed"));
        }
    }
}
