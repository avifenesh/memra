//! Explicit sampled plain-decode capture after warmup; not a throughput cell.
use memra_engine::dsv4_gpu::{
    Dsv4Gpu, Dsv4SampleCfg, Dsv4SamplerOrder, dsv4_sample_row, dsv4_sampler_order,
};
use memra_gguf::dsv4_forward::ActQuantVariant;
use memra_tokenizer::Tokenizer;
use sha2::{Digest, Sha256};
use std::{path::Path, time::Instant};

fn drain(gpu: &Dsv4Gpu) {
    for stage in &gpu.stages {
        stage
            .gpu
            .stream()
            .synchronize()
            .expect("profile boundary drain");
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ProfileArm {
    /// Preserve the profile's pre-existing environment-controlled behavior when no fourth
    /// argument is supplied. This is intentionally not one of the fixed attribution arms.
    Current,
    Baseline,
    Half2,
    ExpertGraph,
}

impl ProfileArm {
    fn parse(args: &[String]) -> Self {
        match args.get(3).map(String::as_str) {
            None => Self::Current,
            Some("baseline") => Self::Baseline,
            Some("half2") => Self::Half2,
            Some("expert-graph") => Self::ExpertGraph,
            Some(other) => {
                panic!("unknown profile arm '{other}' (expected baseline | half2 | expert-graph)")
            }
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Current => "current",
            Self::Baseline => "baseline",
            Self::Half2 => "half2",
            Self::ExpertGraph => "expert-graph",
        }
    }

    fn controlled(self) -> bool {
        !matches!(self, Self::Current)
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct Receipt {
    ep_calls: u64,
    gu_m1: u64,
    gu_half2: u64,
    down_half2: u64,
    captures: u64,
    replays: u64,
    fallbacks: u64,
    eager_prepares: u64,
}

fn receipt(gpu: &Dsv4Gpu) -> Receipt {
    let (captures, replays, fallbacks) = gpu.grouped_graph_counts_for_gate();
    Receipt {
        ep_calls: gpu.ep_calls(),
        gu_m1: memra_engine::moe_f16g_gu_m1_tc_dispatches(),
        gu_half2: memra_engine::moe_f16g_gu_half2_dispatches(),
        down_half2: memra_engine::moe_f16g_down_m1_half2_dispatches(),
        captures,
        replays,
        fallbacks,
        eager_prepares: gpu.grouped_graph_eager_prepare_count_for_gate(),
    }
}

fn receipt_delta(after: Receipt, before: Receipt) -> Receipt {
    Receipt {
        ep_calls: after.ep_calls - before.ep_calls,
        gu_m1: after.gu_m1 - before.gu_m1,
        gu_half2: after.gu_half2 - before.gu_half2,
        down_half2: after.down_half2 - before.down_half2,
        captures: after.captures - before.captures,
        replays: after.replays - before.replays,
        fallbacks: after.fallbacks - before.fallbacks,
        eager_prepares: after.eager_prepares - before.eager_prepares,
    }
}

fn main() {
    let args: Vec<_> = std::env::args().collect();
    assert!(
        (3..=4).contains(&args.len()),
        "usage: dsv4_decode_profile <model-dir> <source.txt> [baseline|half2|expert-graph]"
    );
    let arm = ProfileArm::parse(&args);
    let controlled_arm = arm.controlled();
    assert_eq!(
        dsv4_sampler_order().expect("sampler order"),
        Dsv4SamplerOrder::Radix
    );
    let dir = Path::new(&args[1]);
    let source = std::fs::read_to_string(&args[2]).expect("real source");
    let tokenizer = Tokenizer::from_hf_dir(dir).expect("native tokenizer");
    let mut prompt = tokenizer.encode(
        &format!("Review this inference engine source:\n\n{source}"),
        true,
    );
    assert!(prompt.len() >= 8192);
    prompt.truncate(8192);
    let mut hash = Sha256::new();
    for token in &prompt {
        hash.update(token.to_le_bytes());
    }
    assert_eq!(
        format!("{:x}", hash.finalize()),
        "3f371a0e7d56a4846b28869a780756ceea7dc3a5b1a6426b700e01e4de34593f"
    );
    println!(
        "INPUT count=8192 source_sha256={:x} chunk=512 active_C4=true sampler=radix capture_steps=32..64 arm={}",
        Sha256::digest(source.as_bytes()),
        arm.name()
    );
    let gpu = Dsv4Gpu::load(dir, &[0, 1], ActQuantVariant::RefFp8Round, 8544).expect("model");
    assert!(gpu.matrix_moe_enabled());
    // Preserve the old environment-controlled profile when no fourth argument is supplied.
    // Fixed attribution arms deliberately defer every gate below until after restore, so the
    // 8192-token prefill is outside the candidate posture.
    if !controlled_arm && std::env::var("MEMRA_DSV4_PROFILE_COMPOSE").as_deref() == Ok("1") {
        gpu.set_grouped_route_validation_for_gate(false);
        gpu.set_grouped_mirror_validation_for_gate(false);
        gpu.set_grouped_gu_fuse_for_gate(true);
        println!(
            "PROFILE_COMPOSITION arm=current gu_fuse=true route_validate=false mirror_validate=false"
        );
    }
    let c4_elide =
        !controlled_arm && std::env::var("MEMRA_DSV4_PROFILE_C4_ELIDE").as_deref() == Ok("1");
    // 8544 / ratio-4 is the complete compressed history for this bounded
    // profile. A full recent sidecar makes host publication elision safe for
    // the one-way plain decode capture; older rows remain in canonical host
    // storage and newly emitted rows are served by absolute recent tags.
    let c4_recent_rows = 8544 / 4;
    let mut state = if c4_elide {
        gpu.alloc_decode_state_host_c4_recent(8544, 512, c4_recent_rows)
            .expect("host C4 recent")
    } else {
        gpu.alloc_decode_state_host_c4(8544, 512).expect("host C4")
    };
    let mut draft = gpu.dspark_alloc_state().expect("draft");
    let mut row = gpu
        .dspark_prefill_prime_chunked(&prompt, &mut state, &mut draft, 512)
        .expect("prefill");
    let host = gpu.snapshot_decode_state(&state).expect("snapshot");
    let _host_draft = gpu.snapshot_dspark_state(&draft).expect("draft snapshot");
    drop(state);
    drop(draft);
    let mut state = if c4_elide {
        gpu.restore_decode_state_host_c4_recent(&host, 8544, 512, c4_recent_rows)
            .expect("restore recent C4")
    } else {
        gpu.restore_decode_state_host_c4(&host, 8544, 512)
            .expect("restore")
    };
    if controlled_arm {
        // Fixed attribution arms are decode-only: all gates are armed after the prompt has been
        // prefetched, snapshotted, and restored. This keeps the 8192-token prefill out of the
        // candidate posture and leaves the no-fourth-argument behavior unchanged.
        assert!(
            memra_engine::moe_f16g_mode() >= 2
                && memra_engine::moe_f16g_tail_on()
                && memra_engine::moe_f16g_direct_on(memra_engine::QT_NVFP4_MODELOPT),
            "fixed profile arms require the ModelOpt direct deep-tail visitor"
        );
        gpu.set_grouped_route_validation_for_gate(false);
        gpu.set_grouped_mirror_validation_for_gate(false);
        gpu.set_grouped_gu_fuse_for_gate(true);
        gpu.set_grouped_m1_tc_for_gate(true);
        gpu.set_c4_host_copy_elision_for_gate(false);
        memra_engine::set_moe_f16g_gu_m1_tc_for_gate(false);
        let half2 = matches!(arm, ProfileArm::Half2);
        memra_engine::set_moe_f16g_gu_half2_for_gate(half2);
        memra_engine::set_moe_f16g_down_m1_half2_for_gate(half2);
        gpu.set_grouped_graph_for_gate(matches!(arm, ProfileArm::ExpertGraph))
            .expect("select fixed profile arm");
        println!(
            "PROFILE_ARM arm={} gu_fuse=true m1_down=true gu_m1=false half2={} expert_graph={} route_validate=false mirror_validate=false c4_host_copy_elide=false",
            arm.name(),
            half2,
            matches!(arm, ProfileArm::ExpertGraph),
        );
    }
    if c4_elide {
        gpu.set_c4_host_copy_elision_for_gate(true);
        println!(
            "PROFILE_C4_HOST_COPY_ELISION enabled=true recent_rows={c4_recent_rows} full_capacity=true"
        );
    }
    let cfg = Dsv4SampleCfg {
        temperature: 1.0,
        top_p: 1.0,
        top_k: 0,
        seed: 20260906,
    };
    let trunk_layers = (gpu.model.mc.n_layer - gpu.model.mc.nextn_predict_layers) as u64;
    let initial_receipt = receipt(&gpu);
    let mut tokens = Vec::new();
    let mut capture_timer = None;
    let mut profile_before = None;
    for step in 0..96 {
        if step == 32 {
            drain(&gpu);
            profile_before = Some(receipt(&gpu));
            gpu.stages[0]
                .gpu
                .ctx
                .bind_to_thread()
                .expect("profile start context");
            cudarc::driver::safe::profiler_start().expect("profile start");
            capture_timer = Some(Instant::now());
        }
        let token = dsv4_sample_row(&row, prompt.len() + step, &cfg).expect("sample");
        assert_ne!(
            token,
            tokenizer.eos_id(),
            "early EOS: profile protocol refused"
        );
        tokens.push(token);
        row = gpu.decode_step(token, &mut state).expect("plain step");
        if step == 63 {
            drain(&gpu);
            let seconds = capture_timer.take().unwrap().elapsed().as_secs_f64();
            gpu.stages[0]
                .gpu
                .ctx
                .bind_to_thread()
                .expect("profile stop context");
            cudarc::driver::safe::profiler_stop().expect("profile stop");
            let profile_after = receipt(&gpu);
            let profile_delta = receipt_delta(profile_after, profile_before.unwrap());
            println!(
                "PROFILE_COMPLETE steps=32..64 absolute_positions=8224..8256 seconds={seconds:.6} both_stages_drained=true ep_calls={} gu_half2={} down_half2={} graph_captures={} graph_replays={} graph_fallbacks={} graph_eager_prepares={}",
                profile_delta.ep_calls,
                profile_delta.gu_half2,
                profile_delta.down_half2,
                profile_delta.captures,
                profile_delta.replays,
                profile_delta.fallbacks,
                profile_delta.eager_prepares,
            );
        }
    }
    if c4_elide {
        gpu.set_c4_host_copy_elision_for_gate(false);
        println!("PROFILE_C4_HOST_COPY_ELISION enabled=false");
    }
    drain(&gpu);
    let total_receipt = receipt(&gpu);
    let total_delta = receipt_delta(total_receipt, initial_receipt);
    println!(
        "PROFILE_RECEIPT arm={} ep_calls={} trunk_layers={} gu_m1={} gu_half2={} down_half2={} graph_captures={} graph_replays={} graph_fallbacks={} graph_eager_prepares={}",
        arm.name(),
        total_delta.ep_calls,
        trunk_layers,
        total_delta.gu_m1,
        total_delta.gu_half2,
        total_delta.down_half2,
        total_delta.captures,
        total_delta.replays,
        total_delta.fallbacks,
        total_delta.eager_prepares,
    );
    if controlled_arm {
        assert_eq!(
            total_delta.ep_calls,
            96 * trunk_layers,
            "whole 96-step trunk EP engagement"
        );
        match arm {
            ProfileArm::Baseline => {
                assert_eq!(total_delta.gu_m1, 0, "baseline GU-M1 must be off");
                assert_eq!(total_delta.gu_half2, 0, "baseline GU half2 must be off");
                assert_eq!(total_delta.down_half2, 0, "baseline down half2 must be off");
                assert_eq!(total_delta.captures, 0, "baseline graph captures");
                assert_eq!(total_delta.replays, 0, "baseline graph replays");
                assert_eq!(total_delta.fallbacks, 0, "baseline graph fallbacks");
                assert_eq!(
                    total_delta.eager_prepares,
                    2 * total_delta.ep_calls,
                    "baseline eager rank preparations"
                );
            }
            ProfileArm::Half2 => {
                assert_eq!(total_delta.gu_m1, 0, "half2 GU-M1 must be off");
                assert_eq!(
                    total_delta.gu_half2,
                    2 * total_delta.ep_calls,
                    "half2 GU launches on both EP ranks"
                );
                assert_eq!(
                    total_delta.down_half2,
                    2 * total_delta.ep_calls,
                    "half2 down launches on both EP ranks"
                );
                assert_eq!(total_delta.captures, 0, "half2 graph captures");
                assert_eq!(total_delta.replays, 0, "half2 graph replays");
                assert_eq!(total_delta.fallbacks, 0, "half2 graph fallbacks");
                assert_eq!(
                    total_delta.eager_prepares,
                    2 * total_delta.ep_calls,
                    "half2 eager rank preparations"
                );
            }
            ProfileArm::ExpertGraph => {
                assert_eq!(total_delta.gu_m1, 0, "expert graph GU-M1 must be off");
                assert_eq!(total_delta.gu_half2, 0, "expert graph GU half2 must be off");
                assert_eq!(
                    total_delta.down_half2, 0,
                    "expert graph down half2 must be off"
                );
                assert_eq!(
                    total_delta.captures,
                    2 * trunk_layers,
                    "two retained graph captures per trunk layer"
                );
                assert_eq!(
                    total_delta.replays,
                    2 * trunk_layers * 95,
                    "two-rank graph replays after first capture step"
                );
                assert_eq!(total_delta.fallbacks, 0, "expert graph fallbacks");
                assert_eq!(total_delta.eager_prepares, 0, "expert graph eager prepares");
            }
            ProfileArm::Current => unreachable!("current arm is not controlled"),
        }

        let (graph_workspaces, graph_retained) = gpu
            .grouped_graph_retained_for_gate(&state)
            .expect("profile graph retention");
        let graph_entries = gpu
            .grouped_graph_nodes_for_gate(&state)
            .expect("profile graph nodes");
        let expected_retained = if matches!(arm, ProfileArm::ExpertGraph) {
            2 * trunk_layers as usize
        } else {
            0
        };
        assert_eq!(graph_retained, expected_retained, "retained graph entries");
        assert_eq!(graph_entries.len(), expected_retained, "graph entry census");
        assert_eq!(
            graph_workspaces.iter().sum::<usize>(),
            expected_retained,
            "retained graph workspace census"
        );
        if matches!(arm, ProfileArm::ExpertGraph) {
            let mut per_layer = vec![0usize; trunk_layers as usize];
            for &(_workspace, layer, nodes, kernels) in &graph_entries {
                assert!(layer < per_layer.len(), "graph layer outside trunk");
                assert!(nodes >= kernels && kernels >= 8, "non-empty expert graph");
                per_layer[layer] += 1;
            }
            assert!(
                per_layer.iter().all(|&count| count == 2),
                "two retained ranks per layer"
            );
        }
    }
    let mut hash = Sha256::new();
    for token in &tokens {
        hash.update(token.to_le_bytes());
    }
    assert_eq!(
        format!("{:x}", hash.finalize()),
        "ef41ad6a5caeebf4b5fb54b93d7054b628ace34892b59309b954e20d0d2fb81d",
        "frozen sampled stream"
    );
    assert!(gpu.ep_calls() > 0 && gpu.sink_tiled_calls() > 0);
    println!(
        "PASS radix sampled plain profile with frozen 96-token stream; not a throughput measurement"
    );
}
