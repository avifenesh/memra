//! FP8 K-split numeric-class qualification, drift and sampled ABBA.
//! Derived from the qualified graph split-K protocol; other defaults stay ON.
use memra_engine::dsv4_gpu::{DecodeState, Dsv4Gpu, Dsv4SampleCfg, dsv4_prof_on};
use memra_engine::dsv4_sampler::{Dsv4Sampler, dsv4_sampler};
use memra_gguf::dsv4_forward::ActQuantVariant;
use memra_tokenizer::Tokenizer;
use sha2::{Digest, Sha256};
use std::{
    path::{Path, PathBuf},
    time::Instant,
};

fn slices() -> i32 {
    let value =
        std::env::var("MEMRA_DSV4_FP8_KSPLIT").expect("select component-qualified S: 2, 4 or 8");
    let s: i32 = value.parse().expect("integer S");
    assert!([2, 4, 8].contains(&s), "select S2/S4/S8");
    s
}
unsafe extern "C" {
    fn memra_dsv4_fp8_ksplit_set_for_gate(slices: i32) -> i32;
    fn memra_dsv4_fp8_ksplit_slices_for_gate() -> i32;
}
fn fp8_on() -> bool {
    unsafe { memra_dsv4_fp8_ksplit_slices_for_gate() != 0 }
}
fn default_program() {
    unsafe extern "C" {
        fn memra_dsv4_hc_dot_split_slices_for_gate() -> i32;
    }
    assert_eq!(
        unsafe { memra_dsv4_hc_dot_split_slices_for_gate() },
        16,
        "current HC default required"
    );
    for name in [
        "MEMRA_DSV4_DENSE_FAST",
        "MEMRA_DSV4_NORM_FUSE",
        "MEMRA_DSV4_DENSE_EXACT_TAIL",
        "MEMRA_DSV4_REPLAY_CADENCE",
    ] {
        assert!(
            std::env::var(name).is_err() || std::env::var(name).as_deref() == Ok("1"),
            "default ON required: {name}"
        );
    }
    assert!(
        memra_engine::moe_m1_graph_splitk_on(),
        "default graph split-K required"
    );
}
const PRIME: usize = 256;
const OUTPUT: usize = 256;
const CAPACITY: usize = PRIME + OUTPUT + 8;
const SOURCE_SHA: &str = "f6e175a6f2588953568746fec0cd43fcd046405f74b5c71ce071fe7f37238ded";
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
fn expected_variants(on: bool, start: usize, end: usize, commit: bool) -> [u64; 4] {
    let mut expected = [0; 4];
    for position in start..end {
        let slot = if !on || !(position + 1).is_multiple_of(4) {
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
fn count_kernel(dot: &str, needle: &str) -> usize {
    // CUDA verbose DOT has one ID record per kernel, with symbol and geometry.
    // Counting this record avoids graph labels/edges or repeated pointer text.
    dot.lines()
        .filter(|line| line.trim_start().starts_with("| {ID |") && line.contains(needle))
        .count()
}

fn census(gpu: &Dsv4Gpu, state: &DecodeState, dir: &Path) -> [[String; 4]; 2] {
    let mut hashes: [[String; 4]; 2] = Default::default();
    let on = fp8_on();
    std::fs::create_dir_all(dir).unwrap();
    gpu.dump_full_token_replay_for_gate(state, dir).unwrap();
    assert_eq!(
        gpu.full_token_replay_captures_for_gate(state).unwrap(),
        [3, 1]
    );
    for (rank, rank_hashes) in hashes.iter_mut().enumerate() {
        for (segment, hash) in rank_hashes.iter_mut().enumerate() {
            let dot = std::fs::read_to_string(
                dir.join(format!("full-token-rank{rank}-segment{segment}.dot")),
            )
            .unwrap();
            let partial = count_kernel(&dot, "dsv4_fp8_ksplit_partial_kernel");
            let reduce = count_kernel(&dot, "dsv4_fp8_ksplit_reduce_kernel");
            let expected = if on && segment != 1 { 258 } else { 0 };
            assert_eq!(
                [partial, reduce],
                [expected; 2],
                "FP8 census rank={rank} variant={segment}"
            );
            let forward = segment != 1;
            for name in [
                "dsv4_hc_dot_split_partial_kernel",
                "dsv4_hc_dot_split_reduce_kernel",
            ] {
                assert_eq!(count_kernel(&dot, name), if forward { 86 } else { 0 });
            }
            let expert = if forward { 86 } else { 0 };
            assert_eq!(
                count_kernel(&dot, "moe_m1_graph_splitk_partial_kernel"),
                expert
            );
            assert_eq!(
                count_kernel(&dot, "moe_m1_graph_splitk_reduce_kernel"),
                expert
            );
            assert_eq!(
                count_kernel(&dot, "dsv4_dense_fast_fp8_kernel"),
                if forward {
                    if on { 236 } else { 494 }
                } else {
                    0
                }
            );
            assert_eq!(
                count_kernel(&dot, "dsv4_dense_fast_dots_kernel"),
                if forward {
                    167
                } else if rank == 1 {
                    2
                } else {
                    0
                }
            );
            assert_eq!(
                count_kernel(&dot, "dsv4_norm_rope_f32_fixed_order_kernel"),
                if forward { 43 } else { 0 }
            );
            let norms = match segment {
                0 => 86,
                2 => 128,
                3 => 148,
                _ => usize::from(rank == 1),
            };
            assert_eq!(count_kernel(&dot, "dsv4_rmsnorm_f32acc_kernel"), norms);
            assert_eq!(
                count_kernel(&dot, "dsv4_rope_kernel"),
                if forward { 107 } else { 0 }
            );
            assert_eq!(count_kernel(&dot, "dsv4_dense_exact_tail_fp8_kernel"), 0);
            assert_eq!(count_kernel(&dot, "dsv4_dense_exact_tail_dots_kernel"), 0);
            *hash = format!("{:x}", Sha256::digest(dot.as_bytes()));
            println!(
                "GRAPH_CENSUS on={on} rank={rank} segment={segment} partial={partial} reduce={reduce} sha256={:x}",
                Sha256::digest(dot.as_bytes())
            );
        }
    }
    hashes
}
fn graph_state(gpu: &Dsv4Gpu, prefix: &DecodeState, cfg: Dsv4SampleCfg) -> DecodeState {
    let mut result = state(gpu);
    gpu.restore_full_token_prefix_for_gate(&mut result, prefix)
        .unwrap();
    unsafe {
        gpu.arm_full_token_replay_for_gate(&mut result, cfg)
            .unwrap();
    }
    result
}
fn refusals(gpu: &Dsv4Gpu, prefix: &DecodeState, cfg: Dsv4SampleCfg, inputs: &[u32]) {
    for rank in 0..2 {
        for (position, layer) in [(258usize, 0usize), (259, 0), (383, 21), (511, 42)] {
            let mut failed = graph_state(gpu, prefix, cfg);
            for &token in &inputs[..position - PRIME] {
                gpu.decode_sample_full_token_for_gate(token, &mut failed)
                    .unwrap();
            }
            let cache = gpu.tp_ep_cache_digest_for_gate(&failed).unwrap();
            let counts = gpu.full_token_replay_counts_for_gate(&failed).unwrap();
            let code = 40043 + rank as i32;
            gpu.arm_attention_tp_join_refusal_for_gate(layer, rank, code)
                .unwrap();
            let error = gpu
                .decode_sample_full_token_for_gate(inputs[position - PRIME], &mut failed)
                .unwrap_err();
            assert!(error.contains("one-shot reduction refused"), "{error}");
            let mut words = [0, 0];
            words[rank] = code;
            assert_eq!(gpu.tp_ep_ar_refusal_words().unwrap(), words);
            assert_eq!(failed.pos, position);
            assert_eq!(gpu.tp_ep_cache_digest_for_gate(&failed).unwrap(), cache);
            let after = gpu.full_token_replay_counts_for_gate(&failed).unwrap();
            for r in 0..2 {
                assert_eq!(after[r], [counts[r][0] + 1, counts[r][1]]);
            }
            assert!(
                gpu.decode_sample_full_token_for_gate(inputs[position - PRIME], &mut failed)
                    .unwrap_err()
                    .contains("unfinished transaction")
            );
            assert!(
                gpu.restore_full_token_prefix_for_gate(&mut failed, prefix)
                    .is_err()
            );
            assert_eq!(
                gpu.full_token_replay_counts_for_gate(&failed).unwrap(),
                after
            );
            println!(
                "REFUSAL rank={rank} layer={layer} position={position} cache_unchanged=true no_commit=true quarantined=true"
            );
            gpu.set_tp_ep_ar_refusal_words_for_gate([0, 0]).unwrap();
        }
    }
}
fn qualify_arm(
    gpu: &Dsv4Gpu,
    prompt: &[u32],
    output: &Path,
    cfg: Dsv4SampleCfg,
) -> (String, Identity, u32) {
    let mut prefix = state(gpu);
    gpu.prefill_with_cache_chunked(&prompt[..1], &mut prefix, 1)
        .unwrap();
    for &token in &prompt[1..PRIME] {
        gpu.decode_step_device_logits(token, &mut prefix).unwrap();
    }
    let mut sampler = gpu.device_sampler().unwrap();
    let first = gpu
        .sample_device_logits(&prefix, &mut sampler, &cfg, &[], None)
        .unwrap();
    {
        let mut eager = state(gpu);
        gpu.restore_full_token_prefix_for_gate(&mut eager, &prefix)
            .unwrap();
        let mut graph = graph_state(gpu, &prefix, cfg);
        let mut carry = first;
        let mut inputs = Vec::new();
        for step in 0..OUTPUT {
            inputs.push(carry);
            let before = gpu.full_token_ar_epochs_for_gate().unwrap();
            gpu.decode_step_device_logits(carry, &mut eager).unwrap();
            let next = gpu
                .sample_device_logits(&eager, &mut sampler, &cfg, &[], None)
                .unwrap();
            epochs(gpu, &before, 1);
            let before = gpu.full_token_ar_epochs_for_gate().unwrap();
            let actual = gpu
                .decode_sample_full_token_for_gate(carry, &mut graph)
                .unwrap();
            epochs(gpu, &before, 1);
            assert_eq!(actual, next, "same-class sample step={step}");
            assert_eq!(
                identity(gpu, &graph),
                identity(gpu, &eager),
                "same-class state step={step}"
            );
            assert_eq!(gpu.tp_ep_ar_refusal_words().unwrap(), [0, 0]);
            assert_eq!(
                gpu.full_token_replay_variant_counts_for_gate(&graph)
                    .unwrap(),
                [expected_variants(true, PRIME, PRIME + step + 1, true); 2]
            );
            carry = next;
        }
        census(gpu, &graph, &output.join("qualification-graphs"));
        refusals(gpu, &prefix, cfg, &inputs);
        let ident = identity(gpu, &graph);
        println!(
            "QUALIFIED {{\"fp8_ksplit\":{},\"generated_sha256\":\"{}\",\"final_logits_sha256\":\"{}\",\"final_cache_digest\":{:?},\"final_hidden_digest\":{:?},\"positions\":{OUTPUT},\"within_class\":true}}",
            fp8_on(),
            sha_tokens(&inputs),
            ident.0,
            ident.1,
            ident.2
        );
        (sha_tokens(&inputs), ident, carry)
    }
}

fn select_arm(gpu: &Dsv4Gpu, on: bool) {
    for stage in &gpu.stages {
        stage.gpu.stream().synchronize().unwrap();
    }
    assert_eq!(
        unsafe { memra_dsv4_fp8_ksplit_set_for_gate(if on { slices() } else { 0 }) },
        0
    );
}
fn block_order(reverse: bool) -> [bool; 4] {
    if reverse {
        [false, true, true, false]
    } else {
        [true, false, false, true]
    }
}
struct ScoredArm {
    on: bool,
    prefix: DecodeState,
    graph: DecodeState,
    first: u32,
    rows: u64,
    reference: Option<(String, Identity, u32)>,
    graphs: Option<[[String; 4]; 2]>,
}
impl ScoredArm {
    fn new(gpu: &Dsv4Gpu, prompt: &[u32], cfg: Dsv4SampleCfg, on: bool) -> Self {
        select_arm(gpu, on);
        Self::new_current(gpu, prompt, cfg)
    }
    fn new_current(gpu: &Dsv4Gpu, prompt: &[u32], cfg: Dsv4SampleCfg) -> Self {
        let on = fp8_on();
        let mut prefix = state(gpu);
        gpu.prefill_with_cache_chunked(&prompt[..1], &mut prefix, 1)
            .unwrap();
        for &token in &prompt[1..PRIME] {
            gpu.decode_step_device_logits(token, &mut prefix).unwrap();
        }
        let mut sampler = gpu.device_sampler().unwrap();
        let first = gpu
            .sample_device_logits(&prefix, &mut sampler, &cfg, &[], None)
            .unwrap();
        let graph = graph_state(gpu, &prefix, cfg);
        Self {
            on,
            prefix,
            graph,
            first,
            rows: 0,
            reference: None,
            graphs: None,
        }
    }
}
fn run_abba(
    gpu: &Dsv4Gpu,
    prompt: &[u32],
    tokenizer: &Tokenizer,
    output: &Path,
    cfg: Dsv4SampleCfg,
    reverse: bool,
) {
    // Each numeric class keeps its own prefix lineage and one scored state.
    // These are independent of qualification states, and initially uncaptured.
    let mut arms = [
        ScoredArm::new(gpu, prompt, cfg, false),
        ScoredArm::new(gpu, prompt, cfg, true),
    ];
    println!(
        "ABBA_PROTOCOL reverse={reverse} rows=20 rows_per_arm=10 first_capture_inside_arm_row_0_only=true retained_graphs=true"
    );
    let mut row = 0;
    for on in block_order(reverse) {
        let arm = &mut arms[usize::from(on)];
        assert_eq!(arm.on, on);
        select_arm(gpu, on);
        for _ in 0..5 {
            scored_row(gpu, arm, tokenizer, output, row, reverse);
            row += 1;
        }
    }
    assert_eq!([arms[0].rows, arms[1].rows], [10, 10]);
}

fn scored_row(
    gpu: &Dsv4Gpu,
    arm: &mut ScoredArm,
    tokenizer: &Tokenizer,
    output: &Path,
    row: usize,
    reverse: bool,
) {
    let on = arm.on;
    assert_eq!(fp8_on(), on);
    if arm.rows > 0 {
        gpu.restore_full_token_prefix_for_gate(&mut arm.graph, &arm.prefix)
            .unwrap();
    }
    let before = gpu.full_token_replay_counts_for_gate(&arm.graph).unwrap();
    assert_eq!(before, [[arm.rows * OUTPUT as u64; 2]; 2]);
    let captures_before = gpu.full_token_replay_captures_for_gate(&arm.graph).unwrap();
    assert_eq!(captures_before, if arm.rows == 0 { [0, 0] } else { [3, 1] });
    let before_epochs = gpu.full_token_ar_epochs_for_gate().unwrap();
    let mut carry = arm.first;
    let mut tokens = Vec::with_capacity(OUTPUT);
    let start = Instant::now();
    for _ in 0..OUTPUT {
        tokens.push(carry);
        carry = gpu
            .decode_sample_full_token_for_gate(carry, &mut arm.graph)
            .unwrap();
    }
    let ns = start.elapsed().as_nanos();
    epochs(gpu, &before_epochs, OUTPUT as u32);
    let after = gpu.full_token_replay_counts_for_gate(&arm.graph).unwrap();
    assert_eq!(
        after,
        [[(arm.rows + 1) * OUTPUT as u64; 2]; 2],
        "retained replay progression"
    );
    let captures = gpu.full_token_replay_captures_for_gate(&arm.graph).unwrap();
    assert_eq!(captures, [3, 1], "one capture per arm only");
    let expected = expected_variants(true, PRIME, PRIME + OUTPUT, true).map(|n| n * (arm.rows + 1));
    assert_eq!(
        gpu.full_token_replay_variant_counts_for_gate(&arm.graph)
            .unwrap(),
        [expected; 2]
    );
    let hash = sha_tokens(&tokens);
    let ident = identity(gpu, &arm.graph);
    let current = (hash.clone(), ident.clone(), carry);
    if let Some(reference) = &arm.reference {
        assert_eq!(&current, reference, "within-arm repeat");
    } else {
        arm.reference = Some(current);
    }
    assert!(!tokens.contains(&tokenizer.eos_id()), "early EOS");
    let looped = looped(&tokens);
    let graphs = census(gpu, &arm.graph, &output.join(format!("row-{row}-graphs")));
    if let Some(reference) = &arm.graphs {
        assert_eq!(&graphs, reference, "retained graph identity");
    } else {
        arm.graphs = Some(graphs);
    }
    println!(
        "MEASURE {{\"row\":{row},\"arm_row\":{},\"reverse\":{reverse},\"fp8_ksplit\":{on},\"generated_tokens\":{OUTPUT},\"decode_wall_ns\":{ns},\"decode_tok_s\":{},\"eligible\":{},\"looped\":{looped},\"generated_sha256\":\"{hash}\",\"final_logits_sha256\":\"{}\",\"final_cache_digest\":{:?},\"final_hidden_digest\":{:?},\"first_capture_inside_timing\":{},\"captures_before\":{captures_before:?},\"captures\":{captures:?},\"device_replays\":{after:?}}}",
        arm.rows,
        OUTPUT as f64 * 1e9 / ns as f64,
        !looped,
        ident.0,
        ident.1,
        ident.2,
        arm.rows == 0
    );
    arm.rows += 1;
}
fn reject_profiling() {
    assert!(!dsv4_prof_on(), "unprofiled sampled envelope only");
    for name in [
        "MEMRA_DSV4_NVTX",
        "MEMRA_DSV4_ROUND_PROFILE",
        "MEMRA_DSV4_PROFILE",
        "NSYS_PROFILING_SESSION_ID",
        "NV_COMPUTE_PROFILER_LOCAL_CONNECTION_OVERRIDE",
    ] {
        assert!(
            std::env::var_os(name).is_none(),
            "profiling forbidden: {name}"
        );
    }
}
fn main() {
    let args: Vec<_> = std::env::args().collect();
    if args.get(4).is_some_and(|v| v == "--drift") {
        evidence::run();
        return;
    }
    let selected = slices();
    assert_eq!(unsafe { memra_dsv4_fp8_ksplit_slices_for_gate() }, selected);
    assert_eq!(unsafe { memra_dsv4_fp8_ksplit_set_for_gate(0) }, 0);
    default_program();
    assert!(
        args.len() == 4
            || (args.len() == 5
                && matches!(
                    args[4].as_str(),
                    "--qualify" | "--reverse" | "--drift" | "--capture"
                )),
        "usage: dsv4_fp8_ksplit_gate <model-dir> <source.txt> <new-output-dir> [--qualify|--reverse|--drift|--capture]"
    );
    reject_profiling();
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
    // Host-adaptive MoE OFF; retained graph split-K remains at its default.
    memra_engine::set_moe_m1_splitk_for_gate(false);
    println!(
        "FP8_KSPLIT_POLICY on={} selected_slices={selected}",
        fp8_on()
    );
    let cfg = Dsv4SampleCfg {
        temperature: 1.0,
        top_p: 1.0,
        top_k: 0,
        seed: 20260907,
    };
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
        capture_operands(&gpu, &prompt[..PRIME], &output, cfg);
    } else if args.get(4).is_some_and(|v| v == "--qualify") {
        for on in [false, true] {
            select_arm(&gpu, on);
            let dir = output.join(if on { "on" } else { "off" });
            std::fs::create_dir(&dir).unwrap();
            qualify_arm(&gpu, &prompt[..PRIME], &dir, cfg);
        }
    } else {
        run_abba(
            &gpu,
            &prompt[..PRIME],
            &tokenizer,
            &output,
            cfg,
            args.get(4).is_some_and(|v| v == "--reverse"),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fp8_policy_child() {
        let value = std::env::var("MEMRA_DSV4_FP8_KSPLIT").ok();
        let expected = match value.as_deref() {
            Some("2") => 2,
            Some("4") => 4,
            Some("8") => 8,
            _ => 0,
        };
        assert_eq!(unsafe { memra_dsv4_fp8_ksplit_slices_for_gate() }, expected);
        assert_eq!(unsafe { memra_dsv4_fp8_ksplit_set_for_gate(0) }, 0);
        assert_eq!(unsafe { memra_dsv4_fp8_ksplit_slices_for_gate() }, 0);
        assert_eq!(unsafe { memra_dsv4_fp8_ksplit_set_for_gate(7) }, 40075);
        assert_eq!(unsafe { memra_dsv4_fp8_ksplit_slices_for_gate() }, 0);
        assert_eq!(
            std::thread::spawn(|| unsafe { memra_dsv4_fp8_ksplit_slices_for_gate() })
                .join()
                .unwrap(),
            expected
        );
    }
    #[test]
    fn fp8_unset_zero_and_explicit_slices_use_process_policy() {
        for value in [
            None,
            Some("0"),
            Some("1"),
            Some("2"),
            Some("4"),
            Some("8"),
            Some("32"),
            Some("7"),
            Some(""),
            Some("true"),
        ] {
            let mut child = std::process::Command::new(std::env::current_exe().unwrap());
            child.args(["--exact", "tests::fp8_policy_child", "--nocapture"]);
            if let Some(value) = value {
                child.env("MEMRA_DSV4_FP8_KSPLIT", value);
            } else {
                child.env_remove("MEMRA_DSV4_FP8_KSPLIT");
            }
            let output = child.output().unwrap();
            assert!(
                output.status.success(),
                "policy {value:?}: {}",
                String::from_utf8_lossy(&output.stderr)
            );
            assert!(String::from_utf8_lossy(&output.stdout).contains("1 passed"));
        }
    }
    #[test]
    fn abba_reverse_keeps_ten_rows_per_arm() {
        assert_eq!(block_order(false), [true, false, false, true]);
        assert_eq!(block_order(true), [false, true, true, false]);
        for reverse in [false, true] {
            let mut rows = [0; 2];
            for arm in block_order(reverse) {
                rows[usize::from(arm)] += 5;
            }
            assert_eq!(rows, [10, 10]);
        }
    }
    #[test]
    fn cadence_covers_all_forward_variants() {
        assert_eq!(expected_variants(true, 256, 512, true), [192, 256, 62, 2]);
    }
    #[test]
    fn census_counts_nodes_only() {
        let dot = "graph dsv4_fp8_ksplit_partial_kernel\n| {ID | 3 dsv4_fp8_ksplit_partial_kernel<2> }\nedge dsv4_fp8_ksplit_partial_kernel";
        assert_eq!(count_kernel(dot, "dsv4_fp8_ksplit_partial_kernel"), 1);
    }
    #[test]
    fn short_period_repetition_is_excluded() {
        assert!(looped(&[1, 2].repeat(32)));
        assert!(!looped(&(0..256).collect::<Vec<u32>>()));
    }
}

mod evidence {
    //! Owner evidence only: eight 256-position TF tapes and 64 bounded greedy twins.
    use memra_engine::dsv4_gpu::{DecodeState, Dsv4Gpu, Dsv4SampleCfg};
    use memra_engine::dsv4_sampler::{Dsv4Sampler, dsv4_sampler};
    use memra_gguf::dsv4_forward::ActQuantVariant;
    use memra_tokenizer::Tokenizer;
    use sha2::{Digest, Sha256};
    use std::{
        collections::HashSet,
        io::Write,
        path::{Path, PathBuf},
    };
    const PRIME: usize = 256;
    const TF: usize = 256;
    const GREEDY: usize = 64;
    const SOURCE_SHA: &str = "f6e175a6f2588953568746fec0cd43fcd046405f74b5c71ce071fe7f37238ded";
    fn token_hash(tokens: &[u32]) -> String {
        let mut h = Sha256::new();
        for t in tokens {
            h.update(t.to_le_bytes());
        }
        format!("{:x}", h.finalize())
    }
    fn logit_hash(row: &[f32]) -> String {
        assert!(row.iter().all(|x| x.is_finite()));
        let mut h = Sha256::new();
        for x in row {
            h.update(x.to_bits().to_le_bytes());
        }
        format!("{:x}", h.finalize())
    }
    fn select(gpu: &Dsv4Gpu, on: bool) {
        for rank in &gpu.stages {
            rank.gpu.stream().synchronize().unwrap();
        }
        super::select_arm(gpu, on);
        assert_eq!(super::fp8_on(), on);
    }
    fn prime(gpu: &Dsv4Gpu, tokens: &[u32]) -> DecodeState {
        let mut state = gpu.alloc_decode_state_for_transient(520, 1).unwrap();
        gpu.prefill_with_cache_chunked(&tokens[..1], &mut state, 1)
            .unwrap();
        for &token in &tokens[1..PRIME] {
            gpu.decode_step_device_logits(token, &mut state).unwrap();
        }
        assert_eq!(state.pos, PRIME);
        assert_eq!(gpu.tp_ep_ar_refusal_words().unwrap(), [0, 0]);
        state
    }
    fn tf_arm(gpu: &Dsv4Gpu, tokens: &[u32], id: usize, on: bool, dir: &Path, cfg: Dsv4SampleCfg) {
        select(gpu, on);
        let mut state = prime(gpu, tokens);
        unsafe {
            gpu.arm_full_token_replay_for_gate(&mut state, cfg).unwrap();
        }
        let label = if on { "graph" } else { "control" };
        let mut file = std::io::BufWriter::new(
            std::fs::File::create(dir.join(format!("tf-{id}-{label}.f32le"))).unwrap(),
        );
        for (offset, &token) in tokens[PRIME..PRIME + TF].iter().enumerate() {
            gpu.decode_sample_full_token_for_gate(token, &mut state)
                .unwrap();
            let logits = gpu.read_decode_logits_for_gate(&state).unwrap();
            let hash = logit_hash(&logits);
            for x in &logits {
                file.write_all(&x.to_bits().to_le_bytes()).unwrap();
            }
            assert_eq!(state.pos, PRIME + offset + 1);
            assert_eq!(gpu.tp_ep_ar_refusal_words().unwrap(), [0, 0]);
            println!(
                "TF_ROW {{\"prompt\":{id},\"graph\":{on},\"offset\":{offset},\"position\":{},\"input\":{token},\"vocab\":{},\"logits_sha256\":\"{hash}\"}}",
                state.pos,
                logits.len()
            );
        }
        file.flush().unwrap();
        assert_eq!(
            gpu.full_token_replay_counts_for_gate(&state).unwrap(),
            [[TF as u64; 2]; 2]
        );
        assert_eq!(
            gpu.full_token_replay_captures_for_gate(&state).unwrap(),
            [3, 1]
        );
        println!("TF_ARM_PASS prompt={id} graph={on} positions={TF}");
    }
    fn argmax(row: &[f32]) -> u32 {
        assert!(!row.is_empty() && row.iter().all(|x| x.is_finite()));
        let mut best = 0;
        for i in 1..row.len() {
            if row[i] > row[best] {
                best = i;
            }
        }
        best as u32
    }
    fn looped(tokens: &[u32]) -> bool {
        (1usize..=32).any(|width| {
            let length = width * 4usize.max(32usize.div_ceil(width));
            tokens.windows(length).any(|span| {
                span.chunks_exact(width)
                    .all(|chunk| chunk == &span[..width])
            })
        })
    }
    fn greedy_arm(gpu: &Dsv4Gpu, tokens: &[u32], on: bool) -> Vec<u32> {
        select(gpu, on);
        let mut state = prime(gpu, tokens);
        let mut carry = argmax(&gpu.read_decode_logits_for_gate(&state).unwrap());
        let mut generated = Vec::with_capacity(GREEDY);
        // Greedy is an explicit native eager instrument. Full-token replay admits
        // vendor sampled cfg only; no sampler fallback or replay admission change.
        for _ in 0..GREEDY {
            generated.push(carry);
            carry = gpu.decode_step_greedy(carry, &mut state).unwrap();
            assert_eq!(gpu.tp_ep_ar_refusal_words().unwrap(), [0, 0]);
        }
        assert_eq!(state.pos, PRIME + GREEDY);
        generated
    }
    pub fn run() {
        super::default_program();
        let args: Vec<_> = std::env::args().collect();
        assert_eq!(
            args.len(),
            5,
            "usage: dsv4_graph_splitk_drift_r3 <model-dir> <source.txt> <new-output-dir>"
        );
        super::reject_profiling();
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
            assert_eq!(std::env::var(name).as_deref(), Ok(value), "{name}");
        }
        assert_eq!(dsv4_sampler().unwrap(), Dsv4Sampler::Device);
        memra_engine::set_moe_m1_splitk_for_gate(false);
        let source = std::fs::read_to_string(&args[2]).unwrap();
        assert_eq!(
            format!("{:x}", Sha256::digest(source.as_bytes())),
            SOURCE_SHA
        );
        let tokenizer = Tokenizer::from_hf_dir(Path::new(&args[1])).unwrap();
        let out = PathBuf::from(&args[3]);
        std::fs::create_dir(&out).unwrap();
        assert!(source.len() > 8192 * 64);
        let mut prompts = Vec::new();
        let mut unique = HashSet::new();
        for id in 0..64 {
            let mut start = id * (source.len() - 8192) / 63;
            while !source.is_char_boundary(start) {
                start += 1;
            }
            let mut end = (start + 8192).min(source.len());
            while !source.is_char_boundary(end) {
                end -= 1;
            }
            let text = format!(
                "Review this inference engine source:\n\n{}",
                &source[start..end]
            );
            let tokens = tokenizer.encode(&text, true);
            assert!(tokens.len() >= PRIME + TF);
            let hash = token_hash(&tokens[..PRIME]);
            assert!(unique.insert(hash.clone()), "duplicate prompt");
            std::fs::write(out.join(format!("prompt-{id}.txt")), text).unwrap();
            let mut tape = std::fs::File::create(out.join(format!("prompt-{id}.u32le"))).unwrap();
            for token in &tokens[..PRIME + TF] {
                tape.write_all(&token.to_le_bytes()).unwrap();
            }
            println!(
                "PROMPT {{\"id\":{id},\"source_start\":{start},\"source_end\":{end},\"prefix_tokens\":{PRIME},\"prefix_sha256\":\"{hash}\",\"tf_selected\":{}}}",
                id % 8 == 0
            );
            prompts.push(tokens[..PRIME + TF].to_vec());
        }
        Dsv4Gpu::set_tp_ep_topology_for_gate(true);
        Dsv4Gpu::set_attention_tp_for_gate(true);
        let gpu = Box::new(
            Dsv4Gpu::load(
                Path::new(&args[1]),
                &[0, 1],
                ActQuantVariant::RefFp8Round,
                544,
            )
            .unwrap(),
        );
        assert!(
            gpu.topology().is_tp_ep()
                && gpu.topology().layers == 43
                && gpu.small_kernel_diet_enabled()
        );
        gpu.set_grouped_route_validation_for_gate(false);
        gpu.set_grouped_mirror_validation_for_gate(false);
        gpu.set_grouped_gu_fuse_for_gate(true);
        gpu.set_grouped_m1_tc_for_gate(true);
        memra_engine::set_moe_f16g_gu_m1_tc_for_gate(true);
        memra_engine::set_moe_f16g_gu_half2_for_gate(true);
        memra_engine::set_moe_f16g_down_m1_half2_for_gate(true);
        gpu.set_dense_wo_a_grouped_for_gate(false);
        gpu.set_index_topk_radix_for_gate(true);
        let cfg = Dsv4SampleCfg {
            temperature: 1.0,
            top_p: 1.0,
            top_k: 0,
            seed: 20260907,
        };
        for id in (0..64).step_by(8) {
            for on in [false, true] {
                tf_arm(&gpu, &prompts[id], id, on, &out, cfg);
            }
        }
        let mut exact = 0;
        let mut matches = 0;
        for (id, prompt) in prompts.iter().enumerate() {
            let control = greedy_arm(&gpu, prompt, false);
            let graph = greedy_arm(&gpu, prompt, true);
            let matched = control.iter().zip(&graph).filter(|(a, b)| a == b).count();
            exact += usize::from(control == graph);
            matches += matched;
            let eos_c = control
                .iter()
                .position(|&t| t == tokenizer.eos_id())
                .map_or("null".to_string(), |p| p.to_string());
            let eos_g = graph
                .iter()
                .position(|&t| t == tokenizer.eos_id())
                .map_or("null".to_string(), |p| p.to_string());
            println!(
                "GREEDY_ROW {{\"prompt\":{id},\"tokens_per_arm\":{GREEDY},\"matching_tokens\":{matched},\"exact\":{},\"control_eos_position\":{eos_c},\"graph_eos_position\":{eos_g},\"control_looped\":{},\"graph_looped\":{},\"control\":{control:?},\"graph\":{graph:?}}}",
                control == graph,
                looped(&control),
                looped(&graph)
            );
        }
        println!(
            "QUALITY_COMPLETE tf_prompts=8 tf_positions=2048 greedy_prompts=64 tokens_per_arm=64 exact_prompts={exact} matching_tokens={matches} greedy_denominator=4096 performance_claim=false"
        );
    }
    #[cfg(test)]
    mod tests {
        use super::*;
        #[test]
        fn greedy_ties_choose_the_lowest_id() {
            assert_eq!(argmax(&[1.0, 2.0, 2.0]), 1);
        }
        #[test]
        fn greedy_loop_is_reported_as_an_instrument_property() {
            assert!(looped(&[7; 64]));
            assert!(!looped(&(0..64).collect::<Vec<_>>()));
        }
    }
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
        let wanted = kind == 0 && k == 4096 && (n == 512 || n == 1024);
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
    select_arm(gpu, false);
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
    let expected: BTreeSet<_> = [0, 1]
        .into_iter()
        .flat_map(|rank| [512, 1024].into_iter().map(move |n| (rank, 0, n, 4096)))
        .collect();
    assert_eq!(cap.seen, expected, "four real rank/shape operands");
    println!("PASS capture four real FP8 keys");
}
