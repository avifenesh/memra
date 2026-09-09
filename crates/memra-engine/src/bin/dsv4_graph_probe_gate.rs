//! Explicit, bounded one-layer CUDA-graph probe for the DSV4 matrix decoder.
//!
//! This is a diagnostic gate, not a serving switch.  It intentionally requires
//! `MEMRA_DSV4_GRAPH_PROBE=1` and the literal `graph-probe` mode argument so a
//! release build cannot start a GPU capture by accident.  The probe restores the
//! same prefixed state into eager and graph arms, captures one ratio-0/window-only
//! layer for one token, commits it, then replays that retained graph for the next
//! token.  It reports graph nodes and exact output/route/KV identities.

use memra_engine::dsv4_gpu::{Dsv4Gpu, Dsv4GraphProbeCensus};
use memra_gguf::dsv4_forward::ActQuantVariant;
use memra_tokenizer::Tokenizer;
use sha2::{Digest, Sha256};
use std::{path::Path, time::Instant};

fn digest_bytes(mut feed: impl FnMut(&mut Sha256)) -> String {
    let mut h = Sha256::new();
    feed(&mut h);
    format!("{:x}", h.finalize())
}

fn f32_digest(values: &[f32]) -> String {
    digest_bytes(|h| {
        for value in values {
            h.update(value.to_bits().to_le_bytes());
        }
    })
}

fn i32_digest(values: &[i32]) -> String {
    digest_bytes(|h| {
        for value in values {
            h.update(value.to_le_bytes());
        }
    })
}

fn cache_digest(classes: &[(String, Vec<f32>)]) -> String {
    digest_bytes(|h| {
        for (name, values) in classes {
            h.update((name.len() as u64).to_le_bytes());
            h.update(name.as_bytes());
            for value in values {
                h.update(value.to_bits().to_le_bytes());
            }
        }
    })
}

#[allow(clippy::type_complexity)]
fn route_digest(routes: &[(usize, Vec<i32>, Vec<i32>, Vec<f32>)]) -> String {
    digest_bytes(|h| {
        for (stage, pairs, tokens, weights) in routes {
            h.update((*stage as u64).to_le_bytes());
            h.update((pairs.len() as u64).to_le_bytes());
            for value in pairs.iter().chain(tokens) {
                h.update(value.to_le_bytes());
            }
            for value in weights {
                h.update(value.to_bits().to_le_bytes());
            }
        }
    })
}

fn argmax(row: &[f32]) -> u32 {
    row.iter()
        .enumerate()
        .skip(1)
        .fold((0usize, row[0]), |best, (index, &value)| {
            if value > best.1 { (index, value) } else { best }
        })
        .0 as u32
}

fn exact_f32(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b)
            .all(|(left, right)| left.to_bits() == right.to_bits())
}

fn exact_classes(a: &[(String, Vec<f32>)], b: &[(String, Vec<f32>)]) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b)
            .all(|((an, av), (bn, bv))| an == bn && av.len() == bv.len() && exact_f32(av, bv))
}

#[allow(clippy::type_complexity)]
fn exact_routes(
    a: &[(usize, Vec<i32>, Vec<i32>, Vec<f32>)],
    b: &[(usize, Vec<i32>, Vec<i32>, Vec<f32>)],
) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b)
            .all(|((as_, ap, at, aw), (bs, bp, bt, bw))| {
                as_ == bs && ap == bp && at == bt && exact_f32(aw, bw)
            })
}

fn print_census(census: &[Dsv4GraphProbeCensus]) {
    for entry in census {
        println!(
            "GRAPH_CENSUS layer={} nodes={:?} kernel_count={} kernels={:?}",
            entry.layer,
            entry.nodes,
            entry.kernels.len(),
            entry.kernels
        );
    }
}

fn main() {
    // Freeze this historical instrument independently of the newer defaults.
    // This is process startup, before any model or worker threads exist.
    unsafe {
        std::env::set_var("MEMRA_DSV4_DENSE_FAST", "0");
        std::env::set_var("MEMRA_DSV4_NORM_FUSE", "0");
    }

    let args: Vec<_> = std::env::args().collect();
    assert_eq!(
        args.len(),
        4,
        "usage: dsv4_graph_probe_gate <model-dir> <source.txt> graph-probe"
    );
    assert!(
        args[3] == "graph-probe"
            || args[3] == "graph-multi-probe"
            || args[3] == "graph-indexer-scalar-probe"
            || args[3] == "graph-head-probe"
            || args[3] == "graph-stage0-probe",
        "the explicit graph-probe, graph-multi-probe, graph-indexer-scalar-probe, graph-head-probe, or graph-stage0-probe mode is required"
    );
    assert_eq!(
        std::env::var("MEMRA_DSV4_GRAPH_PROBE").as_deref(),
        Ok("1"),
        "refusing GPU graph capture without MEMRA_DSV4_GRAPH_PROBE=1"
    );
    for (name, expected) in [
        ("MEMRA_DSV4_DECODE_PATH", "device"),
        ("MEMRA_DSV4_EXPERT_ARM", "native"),
        ("MEMRA_DSV4_DENSE_ARM", "fp8"),
        ("MEMRA_DSV4_MOE_PROGRAM", "matrix"),
        ("MEMRA_DSV4_GROUPED_ROUTE", "device"),
    ] {
        assert_eq!(
            std::env::var(name).as_deref(),
            Ok(expected),
            "graph probe requires {name}={expected}"
        );
    }
    assert!(
        std::env::var("MEMRA_DSV4_EP").is_err()
            || matches!(
                std::env::var("MEMRA_DSV4_EP").as_deref(),
                Ok("") | Ok("off")
            ),
        "graph probe requires MEMRA_DSV4_EP=off"
    );

    let dir = Path::new(&args[1]);
    let source = std::fs::read_to_string(&args[2]).expect("source");
    let tokenizer = Tokenizer::from_hf_dir(dir).expect("tokenizer");
    let tokens = tokenizer.encode(
        &format!("Review this inference engine source:\n\n{source}"),
        true,
    );
    let prompt_len = 256usize;
    assert!(
        tokens.len() >= prompt_len + 2,
        "source must provide a 256-token prompt and two continuations"
    );
    println!(
        "PROTOCOL graph_probe=true mode={} prompt={} ep=off validation=route+mirror-off capture_tokens=1 replay_tokens=1 source_sha256={:x}",
        args[3],
        prompt_len,
        Sha256::digest(source.as_bytes())
    );

    let gpu =
        Dsv4Gpu::load(dir, &[0, 1], ActQuantVariant::RefFp8Round, prompt_len + 2).expect("model");
    assert!(gpu.matrix_moe_enabled(), "graph probe requires matrix MoE");

    let mut candidate_layers: Vec<usize> = gpu
        .stages
        .iter()
        .flat_map(|stage| stage.layers.iter())
        .filter(|layer| layer.ratio == 0 && layer.idx.is_none())
        .map(|layer| layer.il as usize)
        .collect();
    assert!(
        !candidate_layers.is_empty(),
        "model has no ratio-0/window-only trunk layer"
    );
    candidate_layers.sort_unstable();
    let multi = args[3] == "graph-multi-probe";
    let indexer_scalar = args[3] == "graph-indexer-scalar-probe";
    let head_graph = args[3] == "graph-head-probe";
    let stage0_graph = args[3] == "graph-stage0-probe";
    if indexer_scalar {
        let indexer_layers: Vec<(usize, usize)> = gpu
            .stages
            .iter()
            .flat_map(|stage| stage.layers.iter())
            .filter(|layer| layer.ratio != 0 && layer.idx.is_some())
            .map(|layer| (layer.il as usize, layer.ratio))
            .collect();
        let (layer, ratio) = *indexer_layers
            .first()
            .expect("model has no fine indexer layer");
        let first_pos = ratio
            .checked_mul(64)
            .and_then(|p| p.checked_sub(2))
            .expect("indexer scalar first position overflow");
        let replay_pos = first_pos + 1;
        let probe = gpu
            .probe_indexer_redirect_scalar_for_gate(layer, first_pos, replay_pos)
            .expect("indexer redirect scalar probe");
        let capture_identity = probe.eager_capture == probe.graph_capture;
        let replay_identity = probe.eager_replay == probe.graph_replay;
        println!(
            "INDEXER_SCALAR layer={} ratio={} window={} cap={} first_pos={} replay_pos={} capture_identity={} replay_identity={} capture_sha256={} replay_sha256={}",
            probe.layer,
            probe.ratio,
            probe.window,
            probe.cap,
            probe.first_pos,
            probe.replay_pos,
            capture_identity,
            replay_identity,
            i32_digest(&probe.graph_capture),
            i32_digest(&probe.graph_replay),
        );
        print_census(std::slice::from_ref(&Dsv4GraphProbeCensus {
            layer: probe.census.layer,
            nodes: probe.census.nodes.clone(),
            kernels: probe.census.kernels.clone(),
        }));
        assert!(capture_identity, "indexer scalar capture identity failed");
        assert!(replay_identity, "indexer scalar replay identity failed");
        assert_ne!(
            probe.graph_capture, probe.graph_replay,
            "indexer scalar replay did not observe the updated position/block scalars"
        );
        println!("PASS gate-only indexer redirect scalar capture/replay exactness");
        return;
    }
    let selected_layers = if head_graph || stage0_graph {
        Vec::new()
    } else if multi {
        candidate_layers
            .windows(2)
            .find(|pair| {
                pair[1] == pair[0] + 1 && gpu.layer_stage[pair[0]] == gpu.layer_stage[pair[1]]
            })
            .map(|pair| pair.to_vec())
            .expect("model has no same-stage contiguous window-only layer pair")
    } else {
        vec![*candidate_layers.iter().max().unwrap()]
    };
    println!(
        "GRAPH_ARM candidate_layers={candidate_layers:?} selected_layers={selected_layers:?} stages={:?}",
        selected_layers
            .iter()
            .map(|&layer| gpu.layer_stage[layer])
            .collect::<Vec<_>>()
    );

    let old_route = gpu.set_grouped_route_validation_for_gate(false);
    let old_mirror = gpu.set_grouped_mirror_validation_for_gate(false);
    assert!(
        old_route && old_mirror,
        "graph gate expected both validation arms initially enabled"
    );

    let mut primed = gpu
        .alloc_decode_state_for_transient(prompt_len + 2, 32)
        .expect("decode state");
    let prime = Instant::now();
    let prefill = gpu
        .prefill_with_cache_chunked(&tokens[..prompt_len], &mut primed, 32)
        .expect("prefill");
    println!(
        "PREFILL prompt={} seconds={:.6}",
        prompt_len,
        prime.elapsed().as_secs_f64()
    );
    let snapshot = gpu.snapshot_decode_state(&primed).expect("snapshot");
    drop(primed);

    let mut eager = gpu
        .restore_decode_state_for_transient(&snapshot, prompt_len + 2, 1)
        .expect("eager restore");
    let mut graph = gpu
        .restore_decode_state_for_transient(&snapshot, prompt_len + 2, 1)
        .expect("graph restore");
    let token1 = argmax(&prefill);
    let eager_row1 = gpu.decode_step(token1, &mut eager).expect("eager token1");
    let eager_routes1 = gpu
        .grouped_route_identity_for_state(&eager)
        .expect("eager routes1");
    let eager_kv1 = gpu.cache_classes(&eager).expect("eager kv1");
    let token2 = argmax(&eager_row1);

    if head_graph {
        gpu.arm_head_graph_probe_for_state(&mut graph)
            .expect("arm head graph probe");
    } else if stage0_graph {
        gpu.arm_stage0_embed_graph_probe_for_state(&mut graph)
            .expect("arm stage-0 embed graph probe");
    } else if multi {
        gpu.arm_multi_layer_graph_probe_for_state(&mut graph, &selected_layers)
            .expect("arm multi-layer graph probe");
    } else {
        gpu.arm_one_layer_graph_probe_for_state(&mut graph, selected_layers[0])
            .expect("arm one-layer graph probe");
    }
    let graph_row1 = gpu
        .decode_step(token1, &mut graph)
        .expect("graph capture token1");
    let census = if head_graph {
        vec![
            gpu.head_graph_probe_census(&graph)
                .expect("head graph census"),
        ]
    } else if stage0_graph {
        vec![
            gpu.stage0_embed_graph_probe_census(&graph)
                .expect("stage-0 embed graph census"),
        ]
    } else {
        gpu.one_layer_graph_probe_census(&graph)
            .expect("graph census")
    };
    assert_eq!(
        census.len(),
        if head_graph || stage0_graph {
            1
        } else {
            selected_layers.len()
        },
        "graph probe must retain exactly the selected layer run"
    );
    print_census(&census);
    let graph_routes1 = gpu
        .grouped_route_identity_for_state(&graph)
        .expect("graph routes1");
    let graph_kv1 = gpu.cache_classes(&graph).expect("graph kv1");
    println!(
        "IDENTITY capture output={} route={} kv={} output_sha256={} route_sha256={} kv_sha256={}",
        exact_f32(&eager_row1, &graph_row1),
        exact_routes(&eager_routes1, &graph_routes1),
        exact_classes(&eager_kv1, &graph_kv1),
        f32_digest(&graph_row1),
        route_digest(&graph_routes1),
        cache_digest(&graph_kv1),
    );
    println!(
        "LIVE_SCALARS capture token={} pos={} commit_state_pos={} selected_layers={:?}",
        token1, prompt_len, graph.pos, selected_layers
    );

    if head_graph {
        gpu.replay_head_graph_probe_for_state(&mut graph)
            .expect("arm head graph replay");
    } else if stage0_graph {
        gpu.replay_stage0_embed_graph_probe_for_state(&mut graph)
            .expect("arm stage-0 embed graph replay");
    } else if multi {
        gpu.replay_multi_layer_graph_probe_for_state(&mut graph)
            .expect("arm multi-layer graph replay");
    } else {
        gpu.replay_one_layer_graph_probe_for_state(&mut graph)
            .expect("arm one-layer graph replay");
    }
    let eager_token2_t0 = Instant::now();
    let eager_row2 = gpu.decode_step(token2, &mut eager).expect("eager token2");
    let eager_token2_s = eager_token2_t0.elapsed().as_secs_f64();
    let eager_routes2 = gpu
        .grouped_route_identity_for_state(&eager)
        .expect("eager routes2");
    let eager_kv2 = gpu.cache_classes(&eager).expect("eager kv2");
    let graph_token2_t0 = Instant::now();
    let graph_row2 = gpu
        .decode_step(token2, &mut graph)
        .expect("graph replay token2");
    let graph_token2_s = graph_token2_t0.elapsed().as_secs_f64();
    let graph_routes2 = gpu
        .grouped_route_identity_for_state(&graph)
        .expect("graph routes2");
    let graph_kv2 = gpu.cache_classes(&graph).expect("graph kv2");
    println!(
        "IDENTITY replay output={} route={} kv={} output_sha256={} route_sha256={} kv_sha256={}",
        exact_f32(&eager_row2, &graph_row2),
        exact_routes(&eager_routes2, &graph_routes2),
        exact_classes(&eager_kv2, &graph_kv2),
        f32_digest(&graph_row2),
        route_digest(&graph_routes2),
        cache_digest(&graph_kv2),
    );
    println!(
        "TIMING replay_token2 mode={} eager_s={:.6} graph_s={:.6} eager_tok_s={:.3} graph_tok_s={:.3} graph_over_eager={:.4}",
        args[3],
        eager_token2_s,
        graph_token2_s,
        1.0 / eager_token2_s,
        1.0 / graph_token2_s,
        graph_token2_s / eager_token2_s,
    );
    println!(
        "LIVE_SCALARS replay token={} pos={} commit_state_pos={} selected_layers={:?}",
        token2,
        prompt_len + 1,
        graph.pos,
        selected_layers
    );

    assert!(
        exact_f32(&eager_row1, &graph_row1),
        "capture output identity failed"
    );
    assert!(
        exact_routes(&eager_routes1, &graph_routes1),
        "capture route identity failed"
    );
    assert!(
        exact_classes(&eager_kv1, &graph_kv1),
        "capture KV identity failed"
    );
    assert!(
        exact_f32(&eager_row2, &graph_row2),
        "replay output identity failed"
    );
    assert!(
        exact_routes(&eager_routes2, &graph_routes2),
        "replay route identity failed"
    );
    assert!(
        exact_classes(&eager_kv2, &graph_kv2),
        "replay KV identity failed"
    );

    gpu.set_grouped_route_validation_for_gate(old_route);
    gpu.set_grouped_mirror_validation_for_gate(old_mirror);
    println!(
        "PASS bounded {} graph probe capture/commit/replay exactness",
        if multi { "multi-layer" } else { "one-layer" }
    );
}
