//! One-load full-token replay falsification. Included only by the sampled gate.
use super::{Dsv4Gpu, Dsv4SampleCfg, Tokenizer, looped, sha256_f32, sha256_tokens};
use memra_engine::dsv4_gpu::{DecodeState, dsv4_pos_uniform};
use memra_engine::dsv4_sampler::Dsv4DeviceSampler;
use sha2::{Digest, Sha256};
use std::{path::Path, time::Instant};

const PRIME: usize = 256;
const OUTPUT: usize = 256;
const CAPACITY: usize = PRIME + OUTPUT + 8;

fn state(gpu: &Dsv4Gpu) -> DecodeState {
    gpu.alloc_decode_state_for_transient(CAPACITY, 1)
        .expect("replay state")
}
fn identity(gpu: &Dsv4Gpu, state: &DecodeState) -> (String, [u64; 2], [u64; 2]) {
    (
        sha256_f32(
            &gpu.read_decode_logits_for_gate(state)
                .expect("identity logits"),
        ),
        gpu.tp_ep_cache_digest_for_gate(state)
            .expect("identity cache"),
        gpu.tp_ep_hidden_digest_for_gate(state)
            .expect("identity hidden"),
    )
}
fn eager_step(
    gpu: &Dsv4Gpu,
    state: &mut DecodeState,
    sampler: &mut Dsv4DeviceSampler,
    cfg: &Dsv4SampleCfg,
    token: u32,
) -> u32 {
    gpu.decode_step_device_logits(token, state)
        .expect("eager forward");
    gpu.sample_device_logits(state, sampler, cfg, &[], None)
        .expect("eager sample")
}
fn epochs(gpu: &Dsv4Gpu, before: &[Vec<u32>; 2], steps: u32) {
    let after = gpu.full_token_ar_epochs_for_gate().expect("AR epochs");
    let attention = memra_engine::tp_ar::ar_blocks_for(4096) as usize;
    let expert = memra_engine::tp_ar::ar_blocks_for(6 * 4096) as usize;
    for rank in 0..2 {
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
fn forward_kernel_census(norm_fuse: bool, norm_fuse2: bool) -> [(usize, u64); 3] {
    // The graph split-K term (+86 forward nodes per rank, two passes replacing
    // one GU and one down node in each of 43 layers) left with the door on
    // 2026-09-11: no entry family it captured exists any more.
    // norm_fuse removes the 43 separate norm/RoPE chains. norm_fuse2 packs the
    // remaining entry-boundary activation chains into 172 fused nodes while
    // removing 387, a net 215 fewer forward launches per rank and forward
    // variant, exactly as docs/FLAGS.md and docs/KERNELS.md record. Without this
    // term the census refuses the door's own arm: memra #428 observed left 2655
    // against right 2870, which is this 215 with the HC term already added.
    let removed = (if norm_fuse { 43 } else { 0 }) + (if norm_fuse2 { 215 } else { 0 });
    [
        (0, 2741 - removed),
        (2, 3140 - removed),
        (3, 3240 - removed),
    ]
}

/// With the split-K doors deleted (memra #461 flip, 2026-09-11) the forward
/// graph has exactly one expert entry family left, and the census says so in
/// both directions: 43 sktail entries on a forward variant, and ZERO nodes from
/// any deleted family on any variant. The negative half is the half that would
/// notice a split-K entry coming back through a stale captured graph.
fn check_expert_nodes(dot: &str, forward: bool) {
    let count = |name: &str| {
        dot.lines()
            .filter(|line| line.contains("| {ID |") && line.contains(name))
            .count()
    };
    let sktail_nodes = if forward { 43 } else { 0 };
    assert_eq!(count("moe_kq_sktail_gu_kernel"), sktail_nodes);
    assert_eq!(count("moe_kq_sktail_kernel"), sktail_nodes);
    for deleted in [
        "moe_m1_graph_splitk_partial_kernel",
        "moe_m1_graph_splitk_reduce_kernel",
        "moe_m1_splitk_fast_partial_kernel",
        "moe_m1_splitk_fast_reduce_kernel",
        "moe_m1_splitk_partial_kernel",
    ] {
        assert_eq!(count(deleted), 0, "{deleted} is a deleted entry family");
    }
}

fn hc_slices() -> i32 {
    unsafe extern "C" {
        fn memra_dsv4_hc_dot_split_slices_for_gate() -> i32;
    }
    unsafe { memra_dsv4_hc_dot_split_slices_for_gate() }
}
fn capture_once(gpu: &Dsv4Gpu, state: &DecodeState, cadence: bool) {
    assert_eq!(
        gpu.full_token_replay_captures_for_gate(state).unwrap(),
        [if cadence { 3 } else { 1 }, 1],
        "recapture"
    );
    let census = gpu.full_token_replay_census_for_gate(state).unwrap();
    for rank in census {
        assert_eq!(
            rank[0][2], 86,
            "all 43 layers' paired collectives must be captured"
        );
        assert_eq!(rank[0][3], 1, "embedding capture");
        assert_eq!(rank[0][4], 86, "both HC posts in every layer");
        assert_eq!(rank[0][6], 0, "unsupported forward node");
        assert_eq!(rank[1][6], 0, "unsupported commit node");
    }
    if cadence {
        for rank in gpu
            .full_token_replay_variant_census_for_gate(state)
            .unwrap()
        {
            for (slot, kernels) in forward_kernel_census(
                gpu.norm_fuse_enabled_for_gate(),
                gpu.norm_fuse2_enabled_for_gate(),
            ) {
                assert_eq!(
                    rank[slot][1],
                    kernels + if hc_slices() != 0 { 86 } else { 0 },
                    "cadence kernel census slot {slot}"
                );
                assert_eq!(
                    [rank[slot][2], rank[slot][3], rank[slot][4], rank[slot][6]],
                    [86, 1, 86, 0]
                );
            }
        }
    }
}

pub(super) fn run(gpu: &Dsv4Gpu, prompt: &[u32], tokenizer: &Tokenizer, reverse: bool) {
    run_impl(gpu, prompt, tokenizer, reverse, false);
}

pub(super) fn cadence(gpu: &Dsv4Gpu, prompt: &[u32], tokenizer: &Tokenizer, reverse: bool) {
    run_impl(gpu, prompt, tokenizer, reverse, true);
}

fn arm(gpu: &Dsv4Gpu, state: &mut DecodeState, cfg: Dsv4SampleCfg, cadence: bool) {
    // All armed states drop before the immutable model/weights/configuration.
    unsafe {
        if cadence {
            gpu.arm_full_token_replay_cadence_for_gate(state, cfg)
        } else {
            gpu.arm_full_token_replay_mode_for_gate(state, cfg, false)
        }
    }
    .expect("arm replay");
}

fn variant_counts(gpu: &Dsv4Gpu, state: &DecodeState, cadence: bool, start: usize, end: usize) {
    let mut expected = [0u64; 4];
    for pos in start..end {
        let slot = if !cadence || !(pos + 1).is_multiple_of(4) {
            0
        } else if (pos + 1).is_multiple_of(128) {
            3
        } else {
            2
        };
        expected[slot] += 1;
        expected[1] += 1;
    }
    assert_eq!(
        gpu.full_token_replay_variant_counts_for_gate(state)
            .unwrap(),
        [expected; 2]
    );
}

fn run_impl(gpu: &Dsv4Gpu, prompt: &[u32], tokenizer: &Tokenizer, reverse: bool, cadence: bool) {
    if cadence {
        println!(
            r#"CADENCE_PROTOCOL {{"comparison":"full_replay_vs_cadence","variant_slots":["ordinary","commit","c4","c4_c128"],"all_first_captures_inside_first_arm_row":true,"positions_below":512,"minimum_gain_threshold":null}}"#
        );
    }
    let schedule = if reverse {
        "BBBBB AAAAA AAAAA BBBBB"
    } else {
        "AAAAA BBBBB BBBBB AAAAA"
    };
    let cfg = Dsv4SampleCfg {
        temperature: 1.0,
        top_p: 1.0,
        top_k: 0,
        seed: 20260907,
    };
    let mut prefix = state(gpu);
    let prime_start = Instant::now();
    gpu.prefill_with_cache_chunked(&prompt[..1], &mut prefix, 1)
        .expect("prime first");
    for &token in &prompt[1..PRIME] {
        gpu.decode_step_device_logits(token, &mut prefix)
            .expect("prime token");
    }
    let mut seed_sampler = gpu.device_sampler().expect("seed sampler");
    let first = gpu
        .sample_device_logits(&prefix, &mut seed_sampler, &cfg, &[], None)
        .expect("initial token");
    let prime_wall = prime_start.elapsed();
    assert_eq!(prefix.pos, PRIME);
    assert_eq!(gpu.tp_ep_ar_refusal_words().unwrap(), [0, 0]);
    println!(
        r#"REPLAY_PROTOCOL {{"prime_tokens":{PRIME},"sampled_output_tokens":{OUTPUT},"prime_wall_ns":{},"rows":20,"blocks":"{schedule}","timing_scope":"sample_plus_forward_envelope","step_order":"forward_refusal_commit_head_sample_readback","initial_carry_draw_outside_timing":true,"final_next_draw_inside_timing":true,"both_arms_same_draw_positions":true,"first_scored_graph_capture_inside_timing":true,"control_hash_provenance":"host_reconstructed_intended_sequence","split_k":false,"device_sampler":true,"diet":true,"host_c4":false,"speculative":false}}"#,
        prime_wall.as_nanos()
    );

    // Safety for all arming calls below: gpu is borrowed for the complete gate;
    // its immutable weight allocations/configuration are never changed here.
    let mut control = state(gpu);
    let mut graph = state(gpu);
    gpu.restore_full_token_prefix_for_gate(&mut control, &prefix)
        .expect("control prefix");
    gpu.restore_full_token_prefix_for_gate(&mut graph, &prefix)
        .expect("graph prefix");
    arm(gpu, &mut graph, cfg, cadence);
    if cadence {
        arm(gpu, &mut control, cfg, false);
    }
    let mut control_sampler = gpu.device_sampler().unwrap();
    let mut inputs = Vec::with_capacity(OUTPUT);
    let mut carry = first;
    for step in 0..OUTPUT {
        assert_ne!(
            carry,
            tokenizer.eos_id(),
            "correctness stream ended before 256 outputs"
        );
        inputs.push(carry);
        let before = gpu.full_token_ar_epochs_for_gate().unwrap();
        let expected = if cadence {
            gpu.decode_sample_full_token_for_gate(carry, &mut control)
                .expect("full replay oracle")
        } else {
            eager_step(gpu, &mut control, &mut control_sampler, &cfg, carry)
        };
        let actual = gpu
            .decode_sample_full_token_for_gate(carry, &mut graph)
            .expect("full replay");
        assert_eq!(actual, expected, "sample at position {}", PRIME + step + 1);
        assert_eq!(graph.pos, control.pos);
        assert_eq!(
            identity(gpu, &graph),
            identity(gpu, &control),
            "full state at step {step}"
        );
        epochs(gpu, &before, 2);
        assert_eq!(
            gpu.full_token_replay_counts_for_gate(&graph).unwrap(),
            [[step as u64 + 1; 2]; 2]
        );
        capture_once(gpu, &graph, cadence);
        variant_counts(gpu, &graph, cadence, PRIME, PRIME + step + 1);
        if cadence {
            capture_once(gpu, &control, false);
            variant_counts(gpu, &control, false, PRIME, PRIME + step + 1);
        }
        if graph.pos.is_multiple_of(4) || step == 0 {
            println!(
                r#"REPLAY_EXACT {{"position":{},"both_rank_cache_hidden_logits":true,"token":{actual},"device_replays":{:?},"captures":{:?}}}"#,
                graph.pos,
                gpu.full_token_replay_counts_for_gate(&graph).unwrap(),
                gpu.full_token_replay_captures_for_gate(&graph).unwrap()
            );
        }
        carry = expected;
    }
    assert!(
        !looped(&inputs),
        "looped correctness stream is not performance eligible"
    );
    let expected_identity = identity(gpu, &control);
    let expected_tokens = sha256_tokens(&inputs);
    std::fs::create_dir_all("qual-graphs").unwrap();
    gpu.dump_full_token_replay_for_gate(&graph, Path::new("qual-graphs"))
        .unwrap();
    println!(
        r#"REPLAY_CENSUS {{"scope":"correctness","rank_segments":{:?}}}"#,
        gpu.full_token_replay_census_for_gate(&graph).unwrap()
    );
    drop(graph);

    // Each fault is armed AFTER a successful graph capture/replay, proving its
    // rank/layer control is live. C4/C128 and ring-wrap boundaries are included.
    let fault_sites: &[(usize, usize)] = if cadence {
        &[(258, 0), (259, 0), (383, 21), (511, 42)]
    } else {
        &[(259, 0), (383, 21), (511, 42)]
    };
    for rank in 0..2 {
        for &(position, layer) in fault_sites {
            let mut failed = state(gpu);
            gpu.restore_full_token_prefix_for_gate(&mut failed, &prefix)
                .unwrap();
            arm(gpu, &mut failed, cfg, cadence);
            for &token in &inputs[..position - PRIME] {
                gpu.decode_sample_full_token_for_gate(token, &mut failed)
                    .expect("fault setup");
            }
            let before = gpu.tp_ep_cache_digest_for_gate(&failed).unwrap();
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
            assert_eq!(failed.pos, position, "refused position advanced");
            assert_eq!(
                gpu.tp_ep_cache_digest_for_gate(&failed).unwrap(),
                before,
                "refused cache changed"
            );
            let after = gpu.full_token_replay_counts_for_gate(&failed).unwrap();
            for r in 0..2 {
                assert_eq!(
                    after[r],
                    [counts[r][0] + 1, counts[r][1]],
                    "refusal committed rank {r}"
                );
            }
            assert!(
                gpu.decode_sample_full_token_for_gate(inputs[position - PRIME], &mut failed)
                    .unwrap_err()
                    .contains("unfinished transaction")
            );
            assert!(
                gpu.restore_full_token_prefix_for_gate(&mut failed, &prefix)
                    .is_err(),
                "reset bypassed quarantine"
            );
            assert_eq!(
                gpu.full_token_replay_counts_for_gate(&failed).unwrap(),
                after,
                "retry executed"
            );
            capture_once(gpu, &failed, cadence);
            println!(
                r#"REPLAY_REFUSAL {{"rank":{rank},"layer":{layer},"position":{position},"words":{words:?},"cache_unchanged":true,"commit_delta":[0,0],"retry_quarantined":true}}"#
            );
            gpu.set_tp_ep_ar_refusal_words_for_gate([0, 0]).unwrap();
        }
    }
    println!(
        "PASS replay correctness and {} live refusal cells; beginning bounded 20-row comparison cadence={cadence}",
        2 * fault_sites.len()
    );

    // The scored graph state is new: first B captures within the measured wall.
    // Subsequent rows restore buffers in place and must retain exactly four graphs.
    let mut candidate = state(gpu);
    gpu.restore_full_token_prefix_for_gate(&mut candidate, &prefix)
        .unwrap();
    arm(gpu, &mut candidate, cfg, cadence);
    // Both measured graph arms pay their complete first capture in their first row.
    let mut control = if cadence {
        drop(control);
        let mut fresh = state(gpu);
        gpu.restore_full_token_prefix_for_gate(&mut fresh, &prefix)
            .unwrap();
        arm(gpu, &mut fresh, cfg, false);
        fresh
    } else {
        control
    };
    let mut walls = [0u128; 2];
    for row in 0..20 {
        let graph_arm = (row / 5 == 1 || row / 5 == 2) != reverse;
        let active = if graph_arm {
            &mut candidate
        } else {
            &mut control
        };
        gpu.restore_full_token_prefix_for_gate(active, &prefix)
            .expect("row restore");
        let before_epochs = gpu.full_token_ar_epochs_for_gate().unwrap();
        let before_counts =
            (graph_arm || cadence).then(|| gpu.full_token_replay_counts_for_gate(active).unwrap());
        let before_variants = (graph_arm || cadence).then(|| {
            gpu.full_token_replay_variant_counts_for_gate(active)
                .unwrap()
        });
        let mut carry = first;
        let mut tokens = Vec::with_capacity(OUTPUT);
        let start = Instant::now();
        for _ in 0..OUTPUT {
            assert_ne!(carry, tokenizer.eos_id(), "early EOS row is not eligible");
            tokens.push(carry);
            carry = if graph_arm || cadence {
                gpu.decode_sample_full_token_for_gate(carry, active)
                    .expect("scored replay")
            } else {
                eager_step(gpu, active, &mut control_sampler, &cfg, carry)
            };
        }
        super::drain(gpu);
        let elapsed = start.elapsed().as_nanos();
        assert!(!looped(&tokens), "looped row is not eligible");
        assert_eq!(
            sha256_tokens(&tokens),
            expected_tokens,
            "scored token stream"
        );
        assert_eq!(
            identity(gpu, active),
            expected_identity,
            "scored final state"
        );
        assert_eq!(gpu.tp_ep_ar_refusal_words().unwrap(), [0, 0]);
        epochs(gpu, &before_epochs, OUTPUT as u32);
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
        let counts = if graph_arm || cadence {
            let after = gpu.full_token_replay_counts_for_gate(active).unwrap();
            for (rank_after, rank_before) in after.iter().zip(before_counts.unwrap()) {
                for (&count_after, count_before) in rank_after.iter().zip(rank_before) {
                    assert_eq!(count_after - count_before, OUTPUT as u64);
                }
            }
            capture_once(gpu, active, graph_arm && cadence);
            let variants = gpu
                .full_token_replay_variant_counts_for_gate(active)
                .unwrap();
            let expected = if graph_arm && cadence {
                [192, 256, 62, 2]
            } else {
                [256, 256, 0, 0]
            };
            for (after, before) in variants.iter().zip(before_variants.unwrap()) {
                for slot in 0..4 {
                    assert_eq!(after[slot] - before[slot], expected[slot]);
                }
            }
            after
        } else {
            [[0; 2]; 2]
        };
        walls[usize::from(graph_arm)] += elapsed;
        println!(
            r#"MEASURE {{"row":{row},"arm":"{}","generated_tokens":{OUTPUT},"decode_wall_ns":{elapsed},"decode_tok_s":{},"timing_scope":"sample_plus_forward_envelope","eligible":true,"looped":false,"generated_sha256":"{}","final_logits_sha256":"{}","final_cache_digest":{:?},"final_hidden_digest":{:?},"device_replays":{counts:?},"captures":{:?},"control_sha256":"{:x}","speculative":false,"split_k":false}}"#,
            if cadence {
                if graph_arm { "cadence" } else { "full_graph" }
            } else if graph_arm {
                "graph"
            } else {
                "eager"
            },
            OUTPUT as f64 * 1e9 / elapsed as f64,
            expected_tokens,
            expected_identity.0,
            expected_identity.1,
            expected_identity.2,
            if graph_arm && cadence {
                [3, 1]
            } else if graph_arm || cadence {
                [1, 1]
            } else {
                [0, 0]
            },
            control_hash.finalize()
        );
        if cadence {
            println!(
                r#"CADENCE_ROW {{"row":{row},"variant_device_counts":{:?},"variant_census":{:?},"first_capture_included_both_arms":true}}"#,
                gpu.full_token_replay_variant_counts_for_gate(active)
                    .unwrap(),
                gpu.full_token_replay_variant_census_for_gate(active)
                    .unwrap()
            );
        }
        if row == if reverse { 0 } else { 5 } {
            std::fs::create_dir_all("perf-graphs").unwrap();
            gpu.dump_full_token_replay_for_gate(active, Path::new("perf-graphs"))
                .unwrap();
            println!(
                r#"REPLAY_CENSUS {{"scope":"scored","rank_segments":{:?}}}"#,
                gpu.full_token_replay_census_for_gate(active).unwrap()
            );
        }
    }
    let eager = 10.0 * OUTPUT as f64 * 1e9 / walls[0] as f64;
    let graph = 10.0 * OUTPUT as f64 * 1e9 / walls[1] as f64;
    if cadence {
        println!(
            r#"CADENCE_SUMMARY {{"rows":20,"rows_per_arm":10,"blocks":"{schedule}","full_replay_tok_s":{eager},"cadence_tok_s":{graph},"delta_pct":{},"same_load":true,"recapture":false,"first_capture_both_arms":true}}"#,
            100.0 * (graph / eager - 1.0)
        );
        return;
    }
    println!(
        r#"REPLAY_SUMMARY {{"rows":20,"rows_per_arm":10,"blocks":"{schedule}","eager_tok_s":{eager},"graph_tok_s":{graph},"delta_pct":{},"speculative":false,"same_load":true,"recapture":false,"decision":"return to root; no automatic qualification or merge"}}"#,
        100.0 * (graph / eager - 1.0)
    );
}

/// Separate profile-only load. This path emits no MEASURE/rate rows. The scored
/// process must run without profiler injection; the controller starts this only
/// after its independent unprofiled BAAB has completed successfully.
pub(super) fn profile(gpu: &Dsv4Gpu, prompt: &[u32], tokenizer: &Tokenizer) {
    use memra_engine::dsv4_gpu::Dsv4Phase;
    let cadence = memra_engine::dsv4_gpu::dsv4_replay_cadence_default();
    let dense = memra_engine::dsv4_gpu::dense_exact_tail_enabled_for_gate();
    let norm_fuse = gpu.norm_fuse_enabled_for_gate();
    unsafe extern "C" {
        fn memra_dsv4_dense_fast_enabled_for_gate() -> i32;
    }
    let dense_fast = unsafe { memra_dsv4_dense_fast_enabled_for_gate() } != 0;
    assert_eq!(std::env::var("MEMRA_DSV4_NVTX").as_deref(), Ok("1"));
    assert_ne!(
        std::env::var("MEMRA_DSV4_ROUND_PROFILE").as_deref(),
        Ok("1")
    );
    let cfg = Dsv4SampleCfg {
        temperature: 1.0,
        top_p: 1.0,
        top_k: 0,
        seed: 20260907,
    };
    let mut prefix = state(gpu);
    gpu.prefill_with_cache_chunked(&prompt[..1], &mut prefix, 1)
        .expect("profile prime first");
    for &token in &prompt[1..PRIME] {
        gpu.decode_step_device_logits(token, &mut prefix)
            .expect("profile prime");
    }
    let mut sampler = gpu.device_sampler().unwrap();
    let mut first = gpu
        .sample_device_logits(&prefix, &mut sampler, &cfg, &[], None)
        .unwrap();
    let mut prefix_tape = prompt[..PRIME].to_vec();
    while prefix.pos < 368 {
        assert_ne!(first, tokenizer.eos_id(), "profile prefix early EOS");
        prefix_tape.push(first);
        first = eager_step(gpu, &mut prefix, &mut sampler, &cfg, first);
    }
    assert_eq!(prefix.pos, 368);
    println!(
        r#"PROFILE_PROTOCOL {{"start_position":368,"end_position":400,"crosses_c128":true,"cadence_on":{cadence},"dense_on":{dense},"dense_fast_on":{dense_fast},"norm_fuse_on":{norm_fuse},"arming":"environment_default","prefix_sha256":"{}","profile_only":true,"scored":false}}"#,
        sha256_tokens(&prefix_tape)
    );
    let mut control = state(gpu);
    let mut candidate = state(gpu);
    gpu.restore_full_token_prefix_for_gate(&mut control, &prefix)
        .unwrap();
    gpu.restore_full_token_prefix_for_gate(&mut candidate, &prefix)
        .unwrap();
    // Safety: gpu/weights/config remain borrowed and unchanged until both states drop.
    unsafe { gpu.arm_full_token_replay_for_gate(&mut candidate, cfg) }.unwrap();
    let expected = eager_step(gpu, &mut control, &mut sampler, &cfg, first);
    let actual = gpu
        .decode_sample_full_token_for_gate(first, &mut candidate)
        .unwrap();
    assert_eq!(actual, expected);
    assert_eq!(identity(gpu, &candidate), identity(gpu, &control));
    capture_once(gpu, &candidate, cadence);
    std::fs::create_dir_all("profile-graphs").unwrap();
    gpu.dump_full_token_replay_for_gate(&candidate, Path::new("profile-graphs"))
        .unwrap();
    for rank in 0..2 {
        for segment in 0..if cadence { 4 } else { 2 } {
            let dot = std::fs::read_to_string(format!(
                "profile-graphs/full-token-rank{rank}-segment{segment}.dot"
            ))
            .unwrap();
            check_expert_nodes(&dot, segment != 1);
            let count = |name: &str| {
                dot.lines()
                    .filter(|line| line.trim_start().starts_with("| {ID |") && line.contains(name))
                    .count()
            };
            let forward = segment != 1;
            assert_eq!(
                count("dsv4_norm_rope_f32_fixed_order_kernel"),
                if norm_fuse && forward { 43 } else { 0 }
            );
            let hc = hc_slices() != 0;
            for name in [
                "dsv4_hc_dot_split_partial_kernel",
                "dsv4_hc_dot_split_reduce_kernel",
            ] {
                assert_eq!(count(name), if hc && forward { 86 } else { 0 });
            }
            let fast = dense && dense_fast;
            assert_eq!(
                count("dsv4_dense_fast_fp8_kernel"),
                if fast && forward { 494 } else { 0 }
            );
            assert_eq!(
                count("dsv4_dense_fast_dots_kernel"),
                if !fast {
                    0
                } else if forward {
                    if hc { 167 } else { 253 }
                } else if rank == 1 {
                    2
                } else {
                    0
                }
            );
            println!("PROFILE_CENSUS rank={rank} segment={segment} passed=true");
        }
    }
    let mut oracle = None;
    for graph_arm in [false, true] {
        let active = if graph_arm {
            &mut candidate
        } else {
            &mut control
        };
        gpu.restore_full_token_prefix_for_gate(active, &prefix)
            .unwrap();
        let before = gpu.full_token_ar_epochs_for_gate().unwrap();
        let before_variants = graph_arm.then(|| {
            gpu.full_token_replay_variant_counts_for_gate(active)
                .unwrap()
        });
        let mut carry = first;
        let mut tokens = Vec::with_capacity(32);
        super::drain(gpu);
        {
            let _capture = Dsv4Phase::new("FULL_TOKEN_PROFILE\0", None);
            let _arm = Dsv4Phase::new(
                if graph_arm {
                    "FULL_TOKEN_REPLAY_WINDOW\0"
                } else {
                    "FULL_TOKEN_EAGER_WINDOW\0"
                },
                None,
            );
            for _ in 0..32 {
                let _step = Dsv4Phase::new("FULL_TOKEN_STEP\0", None);
                assert_ne!(carry, tokenizer.eos_id(), "profile early EOS");
                tokens.push(carry);
                carry = if graph_arm {
                    gpu.decode_sample_full_token_for_gate(carry, active)
                        .unwrap()
                } else {
                    eager_step(gpu, active, &mut sampler, &cfg, carry)
                };
            }
            super::drain(gpu);
        }
        assert_eq!(active.pos, 400);
        assert!(!looped(&tokens));
        epochs(gpu, &before, 32);
        assert_eq!(gpu.tp_ep_ar_refusal_words().unwrap(), [0, 0]);
        let result = (tokens.clone(), carry, identity(gpu, active));
        if graph_arm {
            assert_eq!(Some(result), oracle, "profile arms differ");
            capture_once(gpu, active, cadence);
            let after_variants = gpu
                .full_token_replay_variant_counts_for_gate(active)
                .unwrap();
            let expected = if cadence {
                [24, 32, 7, 1]
            } else {
                [32, 32, 0, 0]
            };
            for (rank, after) in after_variants.iter().enumerate() {
                for (slot, count) in expected.iter().enumerate() {
                    assert_eq!(after[slot] - before_variants.unwrap()[rank][slot], *count);
                }
            }
            assert_eq!(
                gpu.full_token_replay_counts_for_gate(active).unwrap(),
                [[33, 33], [33, 33]]
            );
        } else {
            oracle = Some(result);
        }
        println!(
            r#"PROFILE_ONLY {{"arm":"{}","steps":32,"start_position":368,"end_position":400,"cadence_on":{cadence},"dense_on":{dense},"dense_fast_on":{dense_fast},"norm_fuse_on":{norm_fuse},"tokens_sha256":"{}","identical":true,"scored":false,"capture_outside_window":true}}"#,
            if graph_arm { "graph" } else { "eager" },
            sha256_tokens(&tokens)
        );
        for (step, token) in tokens.iter().enumerate() {
            println!(
                "PROFILE_STEP arm={} position={} token={token}",
                if graph_arm { "graph" } else { "eager" },
                368 + step
            );
        }
    }
    println!("PASS separate profile-only eager/replay windows; no scored rate");
}

#[cfg(test)]
mod profile_census_tests {
    use super::{check_expert_nodes, forward_kernel_census};

    /// Base 2741/3140/3240 forward launches per rank and variant, minus 43 for
    /// norm-fuse and 215 for the norm2 activation pack. The graph split-K term
    /// (+86) left with the door on 2026-09-11, so this function has two inputs
    /// instead of three.
    #[test]
    fn forward_census_is_the_base_minus_each_doors_own_term() {
        assert_eq!(
            forward_kernel_census(false, false),
            [(0, 2741), (2, 3140), (3, 3240)]
        );
        assert_eq!(
            forward_kernel_census(true, false),
            [(0, 2698), (2, 3097), (3, 3197)]
        );
    }

    #[test]
    fn profile_census_tracks_the_norm2_activation_pack_door() {
        // Assert in the shape capture_once uses, so the red arms below exercise
        // the same comparison the gate makes rather than a weaker inequality.
        fn expect(policy: (bool, bool), want: [(usize, u64); 3]) {
            assert_eq!(
                forward_kernel_census(policy.0, policy.1),
                want,
                "forward census for {policy:?}"
            );
        }
        const ON: [(usize, u64); 3] = [(0, 2483), (2, 2882), (3, 2982)];
        const OFF: [(usize, u64); 3] = [(0, 2698), (2, 3097), (3, 3197)];
        expect((true, true), ON);
        expect((true, false), OFF);
        // Crosswise red arms: each arm's expectation must fail against the other
        // arm's policy. Before memra #428 the ON policy was scored against OFF
        // and the gate panicked with left 2655, right 2870.
        assert!(std::panic::catch_unwind(|| expect((true, false), ON)).is_err());
        assert!(std::panic::catch_unwind(|| expect((true, true), OFF)).is_err());
        // The door's own term is the documented net 215, in every variant and
        // independently of the norm-fuse term.
        for norm_fuse in [false, true] {
            let on = forward_kernel_census(norm_fuse, true);
            let off = forward_kernel_census(norm_fuse, false);
            for (slot, off_kernels) in off {
                let on_kernels = on.into_iter().find(|(s, _)| *s == slot).unwrap().1;
                assert_eq!(
                    off_kernels - on_kernels,
                    215,
                    "net norm2 forward launch reduction, slot {slot}"
                );
            }
        }
    }

    #[test]
    fn expert_census_requires_active_policy_and_ignores_non_nodes() {
        let sktail =
            "| {ID | 1 moe_kq_sktail_gu_kernel }\n| {ID | 2 moe_kq_sktail_kernel }\n".repeat(43);
        check_expert_nodes(&sktail, true);
        // Red arm 1: a forward variant that is missing its sktail entries fails.
        let missing = sktail.replacen("| {ID | 2", "edge 2", 1);
        assert!(std::panic::catch_unwind(|| check_expert_nodes(&missing, true)).is_err());
        // Red arm 2: a DELETED entry family reappearing in a captured graph
        // fails, which is the half of this check the flip added. Without it the
        // census would pass on a stale graph that still dispatches split-K.
        for deleted in [
            "moe_m1_graph_splitk_partial_kernel",
            "moe_m1_graph_splitk_reduce_kernel",
            "moe_m1_splitk_fast_partial_kernel",
            "moe_m1_splitk_fast_reduce_kernel",
            "moe_m1_splitk_partial_kernel",
        ] {
            let resurrected = format!("{sktail}| {{ID | 3 {deleted} }}\n");
            assert!(
                std::panic::catch_unwind(move || check_expert_nodes(&resurrected, true)).is_err(),
                "{deleted} must fail the census"
            );
        }
        // A non-forward segment carries no expert entries, and a line that is
        // not a node line is not counted.
        check_expert_nodes("graph moe_kq_sktail_kernel\n", false);
        assert!(std::panic::catch_unwind(|| check_expert_nodes(&sktail, false)).is_err());
    }
}
