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
fn capture_once(gpu: &Dsv4Gpu, state: &DecodeState) {
    assert_eq!(
        gpu.full_token_replay_captures_for_gate(state).unwrap(),
        [1, 1],
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
}

pub(super) fn run(gpu: &Dsv4Gpu, prompt: &[u32], tokenizer: &Tokenizer, reverse: bool) {
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
    unsafe { gpu.arm_full_token_replay_for_gate(&mut graph, cfg) }.expect("arm replay");
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
        let expected = eager_step(gpu, &mut control, &mut control_sampler, &cfg, carry);
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
        capture_once(gpu, &graph);
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
    for rank in 0..2 {
        for (position, layer) in [(259usize, 0usize), (383, 21), (511, 42)] {
            let mut failed = state(gpu);
            gpu.restore_full_token_prefix_for_gate(&mut failed, &prefix)
                .unwrap();
            unsafe { gpu.arm_full_token_replay_for_gate(&mut failed, cfg) }.unwrap();
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
            capture_once(gpu, &failed);
            println!(
                r#"REPLAY_REFUSAL {{"rank":{rank},"layer":{layer},"position":{position},"words":{words:?},"cache_unchanged":true,"commit_delta":[0,0],"retry_quarantined":true}}"#
            );
            gpu.set_tp_ep_ar_refusal_words_for_gate([0, 0]).unwrap();
        }
    }
    println!("PASS replay correctness and six live refusal cells; beginning bounded 20-row ABBA");

    // The scored graph state is new: first B captures within the measured wall.
    // Subsequent rows restore buffers in place and must retain exactly four graphs.
    let mut candidate = state(gpu);
    gpu.restore_full_token_prefix_for_gate(&mut candidate, &prefix)
        .unwrap();
    unsafe { gpu.arm_full_token_replay_for_gate(&mut candidate, cfg) }.unwrap();
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
            graph_arm.then(|| gpu.full_token_replay_counts_for_gate(active).unwrap());
        let mut carry = first;
        let mut tokens = Vec::with_capacity(OUTPUT);
        let start = Instant::now();
        for _ in 0..OUTPUT {
            assert_ne!(carry, tokenizer.eos_id(), "early EOS row is not eligible");
            tokens.push(carry);
            carry = if graph_arm {
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
        let counts = if graph_arm {
            let after = gpu.full_token_replay_counts_for_gate(active).unwrap();
            for (rank_after, rank_before) in after.iter().zip(before_counts.unwrap()) {
                for (&count_after, count_before) in rank_after.iter().zip(rank_before) {
                    assert_eq!(count_after - count_before, OUTPUT as u64);
                }
            }
            capture_once(gpu, active);
            after
        } else {
            [[0; 2]; 2]
        };
        walls[usize::from(graph_arm)] += elapsed;
        println!(
            r#"MEASURE {{"row":{row},"arm":"{}","generated_tokens":{OUTPUT},"decode_wall_ns":{elapsed},"decode_tok_s":{},"timing_scope":"sample_plus_forward_envelope","eligible":true,"looped":false,"generated_sha256":"{}","final_logits_sha256":"{}","final_cache_digest":{:?},"final_hidden_digest":{:?},"device_replays":{counts:?},"captures":{:?},"control_sha256":"{:x}","speculative":false,"split_k":false}}"#,
            if graph_arm { "graph" } else { "eager" },
            OUTPUT as f64 * 1e9 / elapsed as f64,
            expected_tokens,
            expected_identity.0,
            expected_identity.1,
            expected_identity.2,
            if graph_arm { [1, 1] } else { [0, 0] },
            control_hash.finalize()
        );
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
        r#"PROFILE_PROTOCOL {{"start_position":368,"end_position":400,"crosses_c128":true,"prefix_sha256":"{}","profile_only":true,"scored":false}}"#,
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
    capture_once(gpu, &candidate);
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
            capture_once(gpu, active);
            assert_eq!(
                gpu.full_token_replay_counts_for_gate(active).unwrap(),
                [[33, 33], [33, 33]]
            );
        } else {
            oracle = Some(result);
        }
        println!(
            r#"PROFILE_ONLY {{"arm":"{}","steps":32,"start_position":368,"end_position":400,"tokens_sha256":"{}","identical":true,"scored":false,"capture_outside_window":true}}"#,
            if graph_arm { "graph" } else { "eager" },
            sha256_tokens(&tokens)
        );
    }
    println!("PASS separate profile-only eager/replay windows; no scored rate");
}
