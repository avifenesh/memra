//! Graph split-K qualification and retained-graph sampled ABBA.
use memra_engine::dsv4_gpu::{DecodeState, Dsv4Gpu, Dsv4SampleCfg, dsv4_prof_on};
use memra_engine::dsv4_sampler::{Dsv4Sampler, dsv4_sampler};
use memra_gguf::dsv4_forward::ActQuantVariant;
use memra_tokenizer::Tokenizer;
use sha2::{Digest, Sha256};
use std::{
    path::{Path, PathBuf},
    time::Instant,
};
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
    let on = memra_engine::moe_m1_graph_splitk_on();
    std::fs::create_dir_all(dir).unwrap();
    gpu.dump_full_token_replay_for_gate(state, dir).unwrap();
    assert_eq!(
        gpu.full_token_replay_captures_for_gate(state).unwrap(),
        [3, 1]
    );
    for rank in 0..2 {
        for segment in 0..4 {
            let dot = std::fs::read_to_string(
                dir.join(format!("full-token-rank{rank}-segment{segment}.dot")),
            )
            .unwrap();
            let partial = count_kernel(&dot, "moe_m1_graph_splitk_partial_kernel");
            let reduce = count_kernel(&dot, "moe_m1_graph_splitk_reduce_kernel");
            let old = count_kernel(&dot, "moe_m1_splitk_partial_kernel");
            assert_eq!(old, 0, "host-adaptive class captured");
            let expected = if on && segment != 1 { 86 } else { 0 };
            assert_eq!(
                [partial, reduce],
                [expected; 2],
                "graph split-K census rank={rank} segment={segment}"
            );
            if segment != 1 {
                assert_eq!(
                    count_kernel(&dot, "moe_kq_sktail_gu_kernel"),
                    if on { 0 } else { 43 }
                );
                assert_eq!(
                    count_kernel(&dot, "moe_kq_sktail_kernel"),
                    if on { 0 } else { 43 }
                );
            }
            hashes[rank][segment] = format!("{:x}", Sha256::digest(dot.as_bytes()));
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
fn qualify_arm(gpu: &Dsv4Gpu, prompt: &[u32], output: &Path, cfg: Dsv4SampleCfg) {
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
            "QUALIFIED {{\"graph_splitk\":{},\"generated_sha256\":\"{}\",\"final_logits_sha256\":\"{}\",\"final_cache_digest\":{:?},\"final_hidden_digest\":{:?},\"positions\":{OUTPUT},\"within_class\":true}}",
            memra_engine::moe_m1_graph_splitk_on(),
            sha_tokens(&inputs),
            ident.0,
            ident.1,
            ident.2
        );
    }
}

fn select_arm(gpu: &Dsv4Gpu, on: bool) {
    for stage in &gpu.stages {
        stage.gpu.stream().synchronize().unwrap();
    }
    memra_engine::set_moe_m1_graph_splitk_for_gate(on);
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
            let expected =
                expected_variants(true, PRIME, PRIME + OUTPUT, true).map(|n| n * (arm.rows + 1));
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
                "MEASURE {{\"row\":{row},\"arm_row\":{},\"reverse\":{reverse},\"graph_splitk\":{on},\"generated_tokens\":{OUTPUT},\"decode_wall_ns\":{ns},\"decode_tok_s\":{},\"eligible\":{},\"looped\":{looped},\"generated_sha256\":\"{hash}\",\"final_logits_sha256\":\"{}\",\"final_cache_digest\":{:?},\"final_hidden_digest\":{:?},\"first_capture_inside_timing\":{},\"captures_before\":{captures_before:?},\"captures\":{captures:?},\"device_replays\":{after:?}}}",
                arm.rows,
                OUTPUT as f64 * 1e9 / ns as f64,
                !looped,
                ident.0,
                ident.1,
                ident.2,
                arm.rows == 0
            );
            arm.rows += 1;
            row += 1;
        }
    }
    assert_eq!([arms[0].rows, arms[1].rows], [10, 10]);
}

fn teacher_forcing(gpu: &Dsv4Gpu, prompt: &[u32], output: &Path, cfg: Dsv4SampleCfg) {
    use std::io::Write;
    assert!(prompt.len() >= PRIME + 160);
    let mut prefix = state(gpu);
    gpu.prefill_with_cache_chunked(&prompt[..1], &mut prefix, 1)
        .unwrap();
    for &token in &prompt[1..PRIME] {
        gpu.decode_step_device_logits(token, &mut prefix).unwrap();
    }
    let mut graph = graph_state(gpu, &prefix, cfg);
    let mut raw =
        std::io::BufWriter::new(std::fs::File::create(output.join("logits.f32le")).unwrap());
    for i in 0..160 {
        let before = gpu.full_token_ar_epochs_for_gate().unwrap();
        // Fixed source-tape token; ignore the sampled next token in both arms.
        gpu.decode_sample_full_token_for_gate(prompt[PRIME + i], &mut graph)
            .unwrap();
        epochs(gpu, &before, 1);
        let logits = gpu.read_decode_logits_for_gate(&graph).unwrap();
        let hash = sha_f32(&logits);
        for &value in &logits {
            raw.write_all(&value.to_bits().to_le_bytes()).unwrap();
        }
        println!(
            "TF_POSITION offset={i} position={} input={} vocab={} logits_sha256={hash}",
            graph.pos,
            prompt[PRIME + i],
            logits.len()
        );
        assert_eq!(gpu.tp_ep_ar_refusal_words().unwrap(), [0, 0]);
    }
    raw.flush().unwrap();
    census(gpu, &graph, &output.join("tf-graphs"));
    println!(
        "TF_COMPLETE positions=160 report_only=true quality_admission=false identity={:?}",
        identity(gpu, &graph)
    );
}

fn main() {
    let args: Vec<_> = std::env::args().collect();
    assert!(
        args.len() == 4
            || (args.len() == 5
                && matches!(
                    args[4].as_str(),
                    "--qualify" | "--component" | "--tf" | "--reverse"
                )),
        "usage: dsv4_graph_splitk_gate <model-dir> <source.txt> <new-output-dir> [--qualify|--component|--tf|--reverse]"
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
    println!(
        "GRAPH_SPLITK_POLICY on={}",
        memra_engine::moe_m1_graph_splitk_on()
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
    if args.get(4).is_some_and(|v| v == "--tf") {
        teacher_forcing(&gpu, &prompt, &output, cfg);
        return;
    }
    if args.get(4).is_some_and(|v| v == "--component") {
        assert!(memra_engine::moe_m1_graph_splitk_on());
        unsafe extern "C" {
            fn memra_moe_m1_graph_splitk_component_mask() -> u32;
        }
        let mut work = state(&gpu);
        gpu.prefill_with_cache_chunked(&prompt[..1], &mut work, 1)
            .unwrap();
        memra_engine::set_moe_m1_splitk_component_for_gate(true);
        for (i, &token) in prompt[1..PRIME].iter().enumerate() {
            memra_engine::set_moe_m1_splitk_component_token_for_gate(i);
            gpu.decode_step_device_logits(token, &mut work).unwrap();
            let mask = unsafe { memra_moe_m1_graph_splitk_component_mask() };
            println!("GRAPH_COMPONENT_COVERAGE token={i} mask={mask}");
            if mask == 15 {
                break;
            }
        }
        assert_eq!(
            unsafe { memra_moe_m1_graph_splitk_component_mask() },
            15,
            "need real six-live operands on both projections and ranks"
        );
        memra_engine::set_moe_m1_splitk_component_for_gate(false);
        return;
    }
    if args.get(4).is_some_and(|v| v == "--qualify") {
        qualify_arm(&gpu, &prompt[..PRIME], &output, cfg);
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
        let dot = "graph moe_m1_graph_splitk_partial_kernel\n| {ID | 3 moe_m1_graph_splitk_partial_kernel<2> }\nedge moe_m1_graph_splitk_partial_kernel";
        assert_eq!(count_kernel(dot, "moe_m1_graph_splitk_partial_kernel"), 1);
    }
    #[test]
    fn short_period_repetition_is_excluded() {
        assert!(looped(&[1, 2].repeat(32)));
        assert!(!looped(&(0..256).collect::<Vec<u32>>()));
    }
}
