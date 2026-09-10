//! Composition receipt for the two DSV4F doors that flipped default ON on
//! 2026-09-10: `MEMRA_DSV4_SPLITK_FAST` (memra #425) and
//! `MEMRA_DSV4_NORM_FUSE2` (memra #426).
//!
//! Each door was scored alone against the same baseline arm on its own binary.
//! Nobody had run them together on one binary, so the composed number was an
//! estimate. This bin measures it: arm B is the shipped default program with
//! both doors at their new defaults, arm A is the same binary with both doors
//! explicitly `0`. Both arms are the same numeric class and the rows are
//! schedule only.
//!
//! Unlike every other DSV4 gate bin, this one does not pin either door. It
//! reads both from the environment on purpose, refuses a mixed pin, and asserts
//! the loaded policy matches the environment before any gate override runs.
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
const SPLITK_FAST_ENV: &str = "MEMRA_DSV4_SPLITK_FAST";
const NORM_FUSE2_ENV: &str = "MEMRA_DSV4_NORM_FUSE2";
const NORM2_WIDE_ENV: &str = "MEMRA_DSV4_NORM2_WIDE";
/// Forward kernel nodes the norm2 door removes per rank and forward variant.
/// The wide door moves no launch count at all: it swaps one pack symbol for the
/// other, so only the norm2 door appears in the node total.
const NORM_FUSE2_REMOVED: u64 = 215;

fn fast_on() -> bool {
    memra_engine::moe_m1_splitk_fast_on()
}
fn norm2_on(gpu: &Dsv4Gpu) -> bool {
    gpu.norm_fuse2_enabled_for_gate()
}
fn wide_on(gpu: &Dsv4Gpu) -> bool {
    gpu.norm2_wide_enabled_for_gate()
}
/// All three doors move together in this instrument. A process where any of them
/// disagrees is not a composition arm and must never produce a scored row. The
/// wide door additionally cannot engage without the norm2 door, so an arm that
/// resolved them apart would be measuring a program nobody ships.
fn composed_on(gpu: &Dsv4Gpu) -> bool {
    let fast = fast_on();
    let norm2 = norm2_on(gpu);
    let wide = wide_on(gpu);
    assert_eq!(
        [fast, norm2, wide],
        [fast; 3],
        "composed arm requires all three doors on one side, got splitk_fast={fast} norm_fuse2={norm2} norm2_wide={wide}"
    );
    fast
}
/// The only two shapes this instrument accepts: every door unset, or every door
/// explicitly zero. Shared by `main` and its red arm so the test exercises the
/// predicate the binary actually runs, not a copy of it.
fn composition_env_accepted(envs: [Option<&str>; 3]) -> bool {
    envs.iter().all(|v| v.is_none()) || envs.iter().all(|v| *v == Some("0"))
}
fn door_env(name: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| "unset".into())
}
/// Every default this composition inherits and does not measure.
fn default_program() {
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
    assert!(memra_engine::moe_m1_graph_splitk_on());
    unsafe extern "C" {
        fn memra_dsv4_hc_dot_split_slices_for_gate() -> i32;
    }
    assert_eq!(
        unsafe { memra_dsv4_hc_dot_split_slices_for_gate() },
        16,
        "default HC S16 required"
    );
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
fn expected_variants(start: usize, end: usize, commit: bool) -> [u64; 4] {
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
fn count_kernel(dot: &str, needle: &str) -> usize {
    // CUDA verbose DOT has one ID record per kernel, with symbol and geometry.
    // Counting this record avoids graph labels/edges or repeated pointer text.
    dot.lines()
        .filter(|line| line.trim_start().starts_with("| {ID |") && line.contains(needle))
        .count()
}
/// Base forward node count per segment with every inherited default on and both
/// measured doors off. Taken from the norm2 door gate on the same program.
fn base_forward_nodes(segment: usize) -> u64 {
    match segment {
        0 => 2870,
        2 => 3269,
        3 => 3369,
        _ => unreachable!("segment 1 is the commit variant"),
    }
}
/// One census that names both doors. The split-K swap is symbol for symbol, so
/// only the norm2 door moves the total node count; asserting both here is what
/// makes an arm that captured half the composition fail instead of pass.
fn census(gpu: &Dsv4Gpu, state: &DecodeState, dir: &Path) -> [[String; 4]; 2] {
    let mut hashes: [[String; 4]; 2] = Default::default();
    let on = composed_on(gpu);
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
            let forward = segment != 1;
            let expert = if forward { 86 } else { 0 };
            // Door 1: paired-fetch split-K entries replace the base entries.
            let fast_partial = count_kernel(&dot, "moe_m1_splitk_fast_partial_kernel");
            let fast_reduce = count_kernel(&dot, "moe_m1_splitk_fast_reduce_kernel");
            let base_partial = count_kernel(&dot, "moe_m1_graph_splitk_partial_kernel");
            let base_reduce = count_kernel(&dot, "moe_m1_graph_splitk_reduce_kernel");
            assert_eq!(count_kernel(&dot, "moe_m1_splitk_partial_kernel"), 0);
            assert_eq!(
                [fast_partial, fast_reduce],
                [if on { expert } else { 0 }; 2],
                "split-K fast entries rank={rank} segment={segment}"
            );
            assert_eq!(
                [base_partial, base_reduce],
                [if on { 0 } else { expert }; 2],
                "split-K base entries rank={rank} segment={segment}"
            );
            // Door 2: the exact norm2 activation packing.
            let pack = count_kernel(&dot, "dsv4_norm2_pack_f32_fixed_order_kernel");
            let wide = count_kernel(&dot, "dsv4_norm2_pack_f32_fixed_order_wide_kernel");
            let swiglu = count_kernel(&dot, "dsv4_norm2_swiglu_pack_kernel");
            let quant = count_kernel(&dot, "dsv4_norm2_quant_half_kernel");
            // Door 3: the wide pack replaces the single-CTA pack symbol for symbol.
            // The two names do not overlap as substrings, since the wide symbol is
            // `..._fixed_order_wide_kernel` and the narrow needle ends at
            // `..._fixed_order_kernel`; `pack_symbols_do_not_overlap` is the red arm.
            // Totals are asserted against the resolved door state, never against a
            // hard-coded symbol family, so a future default flip cannot leave a
            // stale expectation behind.
            let wide_arm = wide_on(gpu);
            assert_eq!(
                [pack + wide, swiglu, quant],
                if on && forward { [86, 43, 43] } else { [0; 3] },
                "norm2 census total is wide-door independent rank={rank} segment={segment}"
            );
            assert_eq!(
                [pack, wide],
                match (on && forward, wide_arm) {
                    (false, _) => [0, 0],
                    (true, false) => [86, 0],
                    (true, true) => [0, 86],
                },
                "exactly one pack symbol per arm rank={rank} segment={segment}"
            );
            let norms = match segment {
                0 => 86,
                2 => 128,
                3 => 148,
                _ => usize::from(rank == 1),
            };
            assert_eq!(
                count_kernel(&dot, "dsv4_rmsnorm_f32acc_kernel"),
                norms - if on && forward { 86 } else { 0 },
                "unfused norms rank={rank} segment={segment}"
            );
            if forward {
                assert_eq!(
                    count_kernel(&dot, "dsv4_fp8_gather_half_kernel"),
                    if on { 43 } else { 86 }
                );
                assert_eq!(
                    count_kernel(&dot, "dsv4_act_quant_fp8_kernel"),
                    if on { 43 } else { 86 }
                );
            }
            // Inherited defaults are identical in both arms.
            for symbol in [
                "dsv4_hc_dot_split_partial_kernel",
                "dsv4_hc_dot_split_reduce_kernel",
            ] {
                assert_eq!(count_kernel(&dot, symbol), expert);
            }
            assert_eq!(
                count_kernel(&dot, "dsv4_dense_fast_fp8_kernel"),
                if forward { 494 } else { 0 }
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
            assert_eq!(
                count_kernel(&dot, "dsv4_rope_kernel"),
                if forward { 107 } else { 0 }
            );
            assert_eq!(count_kernel(&dot, "dsv4_dense_exact_tail_fp8_kernel"), 0);
            assert_eq!(count_kernel(&dot, "dsv4_dense_exact_tail_dots_kernel"), 0);
            if forward {
                // The runtime census is authoritative for total kernel nodes.
                let runtime = gpu
                    .full_token_replay_variant_census_for_gate(state)
                    .unwrap();
                assert_eq!(
                    runtime[rank][segment][1],
                    base_forward_nodes(segment) - if on { NORM_FUSE2_REMOVED } else { 0 },
                    "composed node census rank={rank} segment={segment}"
                );
            }
            *hash = format!("{:x}", Sha256::digest(dot.as_bytes()));
            println!(
                "GRAPH_CENSUS on={on} rank={rank} segment={segment} splitk_fast={fast_partial}/{fast_reduce} splitk_base={base_partial}/{base_reduce} pack_narrow={pack} pack_wide={wide} swiglu={swiglu} quant={quant} sha256={hash}"
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
                "REFUSAL composed_on={} rank={rank} layer={layer} position={position} cache_unchanged=true no_commit=true quarantined=true",
                composed_on(gpu)
            );
            gpu.set_tp_ep_ar_refusal_words_for_gate([0, 0]).unwrap();
        }
    }
}
/// All three doors move as one arm. Split-K fast has no per-load restore hook, so
/// the caller reads its policy once before the first override and passes it back.
/// The wide door is set after the norm2 door because it cannot engage without it.
fn select_arm(gpu: &Dsv4Gpu, on: bool) {
    memra_engine::set_moe_m1_splitk_fast_for_gate(on);
    gpu.set_norm_fuse2_for_gate(on).unwrap();
    gpu.set_norm2_wide_for_gate(on).unwrap();
    assert_eq!(composed_on(gpu), on);
}
fn qualify_arm(
    gpu: &Dsv4Gpu,
    prompt: &[u32],
    output: &Path,
    cfg: Dsv4SampleCfg,
) -> (String, Identity, u32) {
    let on = composed_on(gpu);
    select_arm(gpu, false);
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
    let mut eager = state(gpu);
    gpu.restore_full_token_prefix_for_gate(&mut eager, &prefix)
        .unwrap();
    select_arm(gpu, on);
    let mut graph = graph_state(gpu, &prefix, cfg);
    let mut carry = first;
    let mut inputs = Vec::new();
    for step in 0..OUTPUT {
        inputs.push(carry);
        let before = gpu.full_token_ar_epochs_for_gate().unwrap();
        select_arm(gpu, false);
        gpu.decode_step_device_logits(carry, &mut eager).unwrap();
        let next = gpu
            .sample_device_logits(&eager, &mut sampler, &cfg, &[], None)
            .unwrap();
        epochs(gpu, &before, 1);
        let before = gpu.full_token_ar_epochs_for_gate().unwrap();
        select_arm(gpu, on);
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
            [expected_variants(PRIME, PRIME + step + 1, true); 2]
        );
        carry = next;
    }
    census(gpu, &graph, &output.join("qualification-graphs"));
    refusals(gpu, &prefix, cfg, &inputs);
    let ident = identity(gpu, &graph);
    println!(
        "QUALIFIED {{\"splitk_fast\":{},\"norm_fuse2\":{},\"norm2_wide\":{},\"generated_sha256\":\"{}\",\"final_logits_sha256\":\"{}\",\"final_cache_digest\":{:?},\"final_hidden_digest\":{:?},\"positions\":{OUTPUT},\"within_class\":true}}",
        fast_on(),
        norm2_on(gpu),
        wide_on(gpu),
        sha_tokens(&inputs),
        ident.0,
        ident.1,
        ident.2
    );
    (sha_tokens(&inputs), ident, carry)
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
    // Both exact arms use fresh prefixes and uncaptured scored states.
    let mut arms = [
        ScoredArm::new(gpu, prompt, cfg, false),
        ScoredArm::new(gpu, prompt, cfg, true),
    ];
    println!(
        "ABBA_PROTOCOL reverse={reverse} rows=20 rows_per_arm=10 first_capture_inside_arm_row_0_only=true retained_graphs=true doors=splitk_fast+norm_fuse2"
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
    assert_eq!(
        arms[0].reference, arms[1].reference,
        "cross-arm raw state/token identity"
    );
    println!("SUMMARY rows=20 identity=true first_capture_rows_per_arm=1,1");
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
    assert_eq!(composed_on(gpu), on);
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
    let expected = expected_variants(PRIME, PRIME + OUTPUT, true).map(|n| n * (arm.rows + 1));
    assert_eq!(
        gpu.full_token_replay_variant_counts_for_gate(&arm.graph)
            .unwrap(),
        [expected; 2]
    );
    assert_eq!(gpu.tp_ep_ar_refusal_words().unwrap(), [0, 0]);
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
    if arm.rows == 0 || arm.rows == 9 {
        let graphs = census(gpu, &arm.graph, &output.join(format!("row-{row}-graphs")));
        if let Some(reference) = &arm.graphs {
            assert_eq!(&graphs, reference, "retained graph identity");
        } else {
            arm.graphs = Some(graphs);
        }
    }
    println!(
        "MEASURE {{\"row\":{row},\"arm_row\":{},\"reverse\":{reverse},\"composed_on\":{on},\"splitk_fast\":{on},\"norm_fuse2\":{on},\"norm2_wide\":{on},\"generated_tokens\":{OUTPUT},\"decode_wall_ns\":{ns},\"decode_tok_s\":{},\"eligible\":{},\"looped\":{looped},\"generated_sha256\":\"{hash}\",\"final_logits_sha256\":\"{}\",\"final_cache_digest\":{:?},\"final_hidden_digest\":{:?},\"first_capture_inside_timing\":{},\"captures_before\":{captures_before:?},\"captures\":{captures:?},\"device_replays\":{after:?}}}",
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
fn main() {
    // The AR phase instrument is gate-only and this is not its gate, so the door is
    // pinned off here: an exported instrument must never ride along with this arm. Process
    // startup, before any model or worker thread exists.
    unsafe {
        std::env::set_var("MEMRA_DSV4_AR_PHASE", "0");
    }
    let args: Vec<_> = std::env::args().collect();
    assert!(
        args.len() == 4
            || (args.len() == 5
                && matches!(args[4].as_str(), "--qualify" | "--reverse" | "--defaults")),
        "usage: dsv4_compose_three_flips_gate <model-dir> <source.txt> <new-output-dir> [--qualify|--reverse|--defaults]"
    );
    assert!(
        !dsv4_prof_on(),
        "profiling rejected in timed/qualification rows"
    );
    for key in [
        "NSYS_PROFILING_SESSION_ID",
        "CUDA_INJECTION64_PATH",
        "NV_INJECTION64_PATH",
    ] {
        assert!(
            std::env::var_os(key).is_none(),
            "profiler injection rejected: {key}"
        );
    }
    default_program();
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
    // This bin observes all three flipped defaults. A mixed pin is not a
    // composition arm, so refuse it before a model exists rather than scoring a
    // fraction of one.
    let fast_env = std::env::var(SPLITK_FAST_ENV).ok();
    let norm2_env = std::env::var(NORM_FUSE2_ENV).ok();
    let wide_env = std::env::var(NORM2_WIDE_ENV).ok();
    let envs = [
        fast_env.as_deref(),
        norm2_env.as_deref(),
        wide_env.as_deref(),
    ];
    assert!(
        composition_env_accepted(envs),
        "composition requires all three doors unset or all three explicit zero, got {envs:?}"
    );
    // Host-adaptive split-K stays off; graph split-K stays on. The measured
    // door only chooses which graph split-K entry pair the capture takes.
    memra_engine::set_moe_m1_splitk_for_gate(false);
    let fast_initial = fast_on();
    assert_eq!(
        fast_initial,
        fast_env.is_none(),
        "split-K fast policy before any gate override"
    );
    println!(
        "COMPOSE_POLICY {SPLITK_FAST_ENV}={} {NORM_FUSE2_ENV}={} {NORM2_WIDE_ENV}={} splitk_fast={fast_initial} graph_splitk={}",
        door_env(SPLITK_FAST_ENV),
        door_env(NORM_FUSE2_ENV),
        door_env(NORM2_WIDE_ENV),
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
    // Loaded norm2 and wide policies, before any gate override, must agree with
    // the environment and with the split-K door read above. The wide door only
    // engages under an admitted norm2 door, which is why unset must resolve it
    // ON here and an all-zero pin must resolve it OFF.
    let norm2_initial = norm2_on(&gpu);
    let wide_initial = wide_on(&gpu);
    assert_eq!(
        norm2_initial,
        norm2_env.is_none(),
        "norm2 policy before any gate override"
    );
    assert_eq!(
        wide_initial,
        wide_env.is_none(),
        "norm2-wide policy before any gate override"
    );
    assert_eq!(
        [fast_initial, norm2_initial, wide_initial],
        [fast_initial; 3],
        "all three doors must load on the same side"
    );
    println!(
        "COMPOSE_LOADED composed_on={norm2_initial} splitk_fast={fast_initial} norm_fuse2={norm2_initial} norm2_wide={wide_initial} before_override=true"
    );

    if args.get(4).is_some_and(|s| s == "--defaults") {
        // The arm comes from the environment policy at load, not from a gate
        // setter, so this is the only cell that proves what an ordinary process
        // gets: unset must engage both doors, explicit 0 must restore both.
        qualify_arm(&gpu, &prompt[..PRIME], &output, cfg);
        select_arm(&gpu, norm2_initial);
        assert_eq!(composed_on(&gpu), norm2_initial, "arm restored");
        println!(
            "DEFAULT_ENGAGEMENT composed_on={norm2_initial} splitk_fast_env={} norm_fuse2_env={} norm2_wide_env={} scored_rows=0",
            door_env(SPLITK_FAST_ENV),
            door_env(NORM_FUSE2_ENV),
            door_env(NORM2_WIDE_ENV)
        );
    } else if args.get(4).is_some_and(|s| s == "--qualify") {
        qualify_arm(&gpu, &prompt[..PRIME], &output, cfg);
        select_arm(&gpu, norm2_initial);
        println!("PASS qualification_only=true composed_on={norm2_initial} scored_rows=0");
    } else {
        run_abba(
            &gpu,
            &prompt[..PRIME],
            &tokenizer,
            &output,
            cfg,
            args.get(4).is_some_and(|s| s == "--reverse"),
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
        assert_eq!(expected_variants(256, 512, true), [192, 256, 62, 2]);
    }
    #[test]
    fn census_counts_nodes_only() {
        let dot = "graph moe_m1_splitk_fast_partial_kernel\n| {ID | 3 moe_m1_splitk_fast_partial_kernel<2> }\nedge moe_m1_splitk_fast_partial_kernel";
        assert_eq!(count_kernel(dot, "moe_m1_splitk_fast_partial_kernel"), 1);
    }
    #[test]
    fn short_period_repetition_is_excluded() {
        assert!(looped(&[1, 2].repeat(32)));
        assert!(!looped(&(0..256).collect::<Vec<u32>>()));
    }

    /// Red arm for the composed node census. The ON expectation must refuse the
    /// OFF total and the OFF expectation must refuse the ON total, in the exact
    /// arithmetic `census` uses, so an arm that captured the wrong class fails.
    #[test]
    fn composed_node_census_refuses_the_other_arm() {
        for segment in [0usize, 2, 3] {
            let base = base_forward_nodes(segment);
            let on_total = base - NORM_FUSE2_REMOVED;
            let off_total = base;
            assert_ne!(on_total, off_total, "segment {segment} arms must differ");
            for (arm_on, observed) in [(true, off_total), (false, on_total)] {
                let expected = base - if arm_on { NORM_FUSE2_REMOVED } else { 0 };
                assert!(
                    std::panic::catch_unwind(move || assert_eq!(observed, expected)).is_err(),
                    "segment {segment} arm_on={arm_on} accepted the other arm's census"
                );
            }
        }
    }

    /// Red arm for the paired-door contract: a mixed pin must never reach a row.
    #[test]
    fn mixed_door_pins_are_refused() {
        // Exercises the predicate main runs, not a copy of it.
        assert!(composition_env_accepted([None, None, None]));
        assert!(composition_env_accepted([Some("0"), Some("0"), Some("0")]));
        // Every mixed shape over three doors, plus the explicit-one shapes.
        let mut refused = 0;
        for a in [None, Some("0"), Some("1")] {
            for b in [None, Some("0"), Some("1")] {
                for c in [None, Some("0"), Some("1")] {
                    let envs = [a, b, c];
                    let accepted = composition_env_accepted(envs);
                    let should = envs == [None; 3] || envs == [Some("0"); 3];
                    assert_eq!(accepted, should, "wrong verdict for {envs:?}");
                    refused += u32::from(!accepted);
                }
            }
        }
        // 27 shapes, exactly two accepted, so the refusal set is not vacuous.
        assert_eq!(refused, 25);
    }

    /// Red arm for the pack-symbol census: the wide needle must not be counted by
    /// the narrow needle, or an ON arm would read as a full narrow arm and pass.
    #[test]
    fn pack_symbols_do_not_overlap() {
        const NARROW: &str = "dsv4_norm2_pack_f32_fixed_order_kernel";
        const WIDE: &str = "dsv4_norm2_pack_f32_fixed_order_wide_kernel";
        assert!(
            !WIDE.contains(NARROW),
            "wide symbol contains the narrow needle"
        );
        let wide_dot = format!("| {{ID | 1 {WIDE} }}\n");
        assert_eq!(count_kernel(&wide_dot, NARROW), 0);
        assert_eq!(count_kernel(&wide_dot, WIDE), 1);
        let narrow_dot = format!("| {{ID | 1 {NARROW} }}\n");
        assert_eq!(count_kernel(&narrow_dot, NARROW), 1);
        assert_eq!(count_kernel(&narrow_dot, WIDE), 0);
    }

    /// Red arm for the wide-door census arm selection, in the shape `census` uses.
    /// Each arm's expectation must refuse the other arm's observation.
    #[test]
    fn pack_census_refuses_the_other_wide_arm() {
        fn expected(on: bool, wide_arm: bool) -> [usize; 2] {
            match (on, wide_arm) {
                (false, _) => [0, 0],
                (true, false) => [86, 0],
                (true, true) => [0, 86],
            }
        }
        assert_eq!(expected(true, true), [0, 86]);
        assert_eq!(expected(true, false), [86, 0]);
        for (arm, other) in [(true, false), (false, true)] {
            let mine = expected(true, arm);
            let theirs = expected(true, other);
            assert!(
                std::panic::catch_unwind(move || assert_eq!(theirs, mine)).is_err(),
                "wide arm {arm} accepted the other arm's pack census"
            );
        }
        // The total is door independent, which is why the split assert is needed.
        for arm in [true, false] {
            assert_eq!(expected(true, arm).iter().sum::<usize>(), 86);
        }
    }

    /// Invoked in fresh processes by the parent test below, so the once-per-process
    /// environment read cannot leak between the unset and explicit-zero arms.
    #[test]
    fn splitk_fast_policy_child() {
        let expected = std::env::var(SPLITK_FAST_ENV).is_err();
        assert_eq!(fast_on(), expected, "policy before any gate override");
        memra_engine::set_moe_m1_splitk_fast_for_gate(!expected);
        assert_eq!(fast_on(), !expected, "gate override took");
    }
    #[test]
    fn splitk_fast_unset_and_zero_use_actual_process_policy() {
        for value in [None, Some("0")] {
            let mut cmd = std::process::Command::new(std::env::current_exe().unwrap());
            cmd.args(["--exact", "tests::splitk_fast_policy_child", "--nocapture"]);
            match value {
                Some(value) => cmd.env(SPLITK_FAST_ENV, value),
                None => cmd.env_remove(SPLITK_FAST_ENV),
            };
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
