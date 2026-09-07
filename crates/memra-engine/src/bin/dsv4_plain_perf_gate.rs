//! Plain-only, sampled ABBA over a frozen prompt snapshot. No speculative timing rows.
use memra_engine::dsv4_gpu::{
    Dsv4Gpu, Dsv4GraphBStats, Dsv4GraphBVariantStats, Dsv4HostDecodeState, Dsv4SampleCfg,
    dsv4_sample_row,
};
use memra_gguf::dsv4_forward::ActQuantVariant;
use memra_tokenizer::Tokenizer;
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fmt::Write as _,
    path::Path,
    time::{Instant, SystemTime, UNIX_EPOCH},
};

const OUTPUT: usize = 256;
const REPEATS: usize = 3;

#[derive(Clone, Copy, Debug)]
enum Change {
    GuM1,
    Half2,
    WoA,
    IndexTopk,
    GraphB,
}

impl Change {
    fn name(self) -> &'static str {
        match self {
            Self::GuM1 => "gu-m1",
            Self::Half2 => "half2",
            Self::WoA => "wo-a",
            Self::IndexTopk => "index-topk",
            Self::GraphB => "graph-b",
        }
    }
    fn gu_m1(self, tuned: bool) -> bool {
        tuned && matches!(self, Self::GuM1)
    }
    fn half2(self, tuned: bool) -> bool {
        matches!(self, Self::WoA | Self::IndexTopk | Self::GraphB)
            || (tuned && matches!(self, Self::Half2))
    }
    fn wo_a(self, tuned: bool) -> bool {
        matches!(self, Self::IndexTopk | Self::GraphB) || (tuned && matches!(self, Self::WoA))
    }
    fn index_topk(self, tuned: bool) -> bool {
        matches!(self, Self::GraphB) || (tuned && matches!(self, Self::IndexTopk))
    }
    fn graph_b(self, tuned: bool, prompt: usize) -> bool {
        tuned && prompt == 8192 && matches!(self, Self::GraphB)
    }
}

fn drain(gpu: &Dsv4Gpu) {
    for stage in &gpu.stages {
        stage.gpu.stream().synchronize().expect("gate drain");
    }
}

fn token_hash(tokens: &[u32]) -> String {
    let mut hash = Sha256::new();
    for token in tokens {
        hash.update(token.to_le_bytes());
    }
    format!("{:x}", hash.finalize())
}

fn f32_hash(values: &[f32]) -> String {
    let mut hash = Sha256::new();
    for value in values {
        hash.update(value.to_bits().to_le_bytes());
    }
    format!("{:x}", hash.finalize())
}

fn looped(tokens: &[u32]) -> bool {
    (1usize..=32).any(|width| {
        let length = width * 4usize.max(32usize.div_ceil(width));
        tokens
            .windows(length)
            .any(|span| span.chunks_exact(width).all(|b| b == &span[..width]))
    })
}

fn json_quote(value: &str) -> String {
    let mut out = String::with_capacity(value.len() + 2);
    out.push('"');
    for ch in value.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c.is_control() => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

fn graph_b_nodes_json(nodes: &BTreeMap<String, usize>) -> String {
    nodes
        .iter()
        .map(|(name, count)| format!("{}:{}", json_quote(name), count))
        .collect::<Vec<_>>()
        .join(",")
}

fn graph_b_kernels_json(kernels: &[String]) -> String {
    kernels
        .iter()
        .map(|name| json_quote(name))
        .collect::<Vec<_>>()
        .join(",")
}

fn graph_b_variants_json(stats: &Dsv4GraphBStats) -> String {
    stats
        .variants
        .iter()
        .map(|variant| {
            format!(
                "{{\"layer\":{},\"stage\":{},\"attention_topk\":{},\"slots\":{},\"arm\":{},\"nodes\":{{{}}},\"kernels\":[{}]}}",
                variant.layer,
                variant.stage,
                variant.attention_topk,
                variant.slots,
                variant.arm,
                graph_b_nodes_json(&variant.nodes),
                graph_b_kernels_json(&variant.kernels),
            )
        })
        .collect::<Vec<_>>()
        .join(",")
}

#[derive(Default)]
struct GraphBCoverage {
    layers: BTreeSet<usize>,
    variants_per_layer: BTreeMap<usize, usize>,
    attention_variants: usize,
    output_variants: usize,
    hc_variants: usize,
}

fn graph_b_kernel_coverage(variant: &Dsv4GraphBVariantStats) -> (bool, bool, bool) {
    // CUDA's C++ template encodings are case-sensitive (ILi1ELb1E). Preserve
    // the actual driver names rather than lowercasing their mangled suffixes.
    let names: Vec<&str> = variant.kernels.iter().map(String::as_str).collect();
    let has = |needle: &str| names.iter().any(|name| name.contains(needle));
    // These are the actual Graph-B device symbols, not wrapper/body labels:
    // sink score/soft/out + rope; grouped wo_a + wo_b GEMM + bf16 conversion;
    // hc_post followed by the batch HC pre-chain's real kernels.
    let attention = has("sink_scores_tiled_f32acc")
        && has("sink_soft_mq_f32acc")
        && has("sink_out_mq_f32acc")
        && has("rope");
    let grouped_wo_a = names.iter().any(|name| {
        name.contains("dsv4_gemv_fp8_m_kernel")
            && (name.contains("ILi1ELb1E")
                || name.contains("<1, true>")
                || name.contains("<1,true>"))
    });
    let regular_wo_b = names.iter().any(|name| {
        name.contains("dsv4_gemv_fp8_m_kernel")
            && (name.contains("ILi1ELb0E")
                || name.contains("<1, false>")
                || name.contains("<1,false>"))
    });
    let output = grouped_wo_a && regular_wo_b && has("cvt_bf16");
    let hc = has("hc_post")
        && has("hc_sinkhorn_m")
        && has("hc_collapse")
        && has("rowsq_scale")
        && (has("dots_f32_mrow") || has("dots_f32acc_mrow"))
        && has("rmsnorm");
    (attention, output, hc)
}

fn graph_b_coverage(stats: &Dsv4GraphBStats) -> GraphBCoverage {
    let mut coverage = GraphBCoverage::default();
    for variant in &stats.variants {
        coverage.layers.insert(variant.layer);
        *coverage
            .variants_per_layer
            .entry(variant.layer)
            .or_insert(0) += 1;
        let (attention, output, hc) = graph_b_kernel_coverage(variant);
        coverage.attention_variants += attention as usize;
        coverage.output_variants += output as usize;
        coverage.hc_variants += hc as usize;
    }
    coverage
}

struct Ready<'a> {
    gpu: &'a Dsv4Gpu,
    tokenizer: &'a Tokenizer,
    prompt_len: usize,
    logits: &'a [f32],
    snapshot: &'a Dsv4HostDecodeState,
}

#[derive(Debug, PartialEq, Eq)]
struct Identity {
    tokens: Vec<u32>,
    logits: String,
    cache: String,
    pos: usize,
}

impl Ready<'_> {
    fn run(&self, change: Change, tuned: bool, warmup: bool, ordinal: usize) -> Identity {
        drain(self.gpu);
        let gu_m1 = change.gu_m1(tuned);
        let half2 = change.half2(tuned);
        let wo_a_grouped = change.wo_a(tuned);
        let index_topk_radix = change.index_topk(tuned);
        let graph_b_enabled = change.graph_b(tuned, self.prompt_len);
        self.gpu.clear_dense_wo_a_grouped_for_gate();
        self.gpu.clear_index_topk_radix_for_gate();
        memra_engine::set_moe_f16g_gu_m1_tc_for_gate(gu_m1);
        memra_engine::set_moe_f16g_gu_half2_for_gate(half2);
        memra_engine::set_moe_f16g_down_m1_half2_for_gate(half2);
        let mut state = self
            .gpu
            .restore_decode_state_host_c4(self.snapshot, self.prompt_len + OUTPUT + 96, 512)
            .expect("same host C4 snapshot restore");
        self.gpu.set_dense_wo_a_grouped_for_gate(wo_a_grouped);
        self.gpu.set_index_topk_radix_for_gate(index_topk_radix);
        if graph_b_enabled {
            self.gpu
                .set_graph_b_for_state(&mut state, true)
                .expect("arm Graph-B state");
        }
        let mut row = self.logits.to_vec();
        let mut tokens = Vec::with_capacity(OUTPUT);
        let mut commits = Vec::with_capacity(OUTPUT);
        let mut eos = false;
        let cfg = Dsv4SampleCfg {
            temperature: 1.0,
            top_p: 1.0,
            top_k: 0,
            seed: 20260906,
        };
        drain(self.gpu);
        let calls_before = memra_engine::moe_f16g_gu_m1_tc_dispatches();
        let ep_before = self.gpu.ep_calls();
        let gu_half2_before = memra_engine::moe_f16g_gu_half2_dispatches();
        let down_half2_before = memra_engine::moe_f16g_down_m1_half2_dispatches();
        let wo_a_before = self.gpu.dense_wo_a_grouped_dispatches();
        let index_topk_before = self.gpu.index_topk_radix_dispatches();
        let start_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis();
        println!(
            "START mode={} prompt={} tuned={tuned} warmup={warmup} ordinal={ordinal}",
            change.name(),
            self.prompt_len
        );
        let timer = Instant::now();
        let mut first_step_ns = 0;
        for i in 0..OUTPUT {
            let token = dsv4_sample_row(&row, self.prompt_len + i, &cfg).expect("sample");
            if token == self.tokenizer.eos_id() {
                eos = true;
                break;
            }
            tokens.push(token);
            commits.push(timer.elapsed().as_nanos());
            if i + 1 == OUTPUT {
                break;
            }
            row = self
                .gpu
                .decode_step(token, &mut state)
                .expect("plain decode");
            if i == 0 {
                first_step_ns = timer.elapsed().as_nanos();
            }
        }
        drain(self.gpu);
        let wall_ns = timer.elapsed().as_nanos();
        let gu_m1_calls = memra_engine::moe_f16g_gu_m1_tc_dispatches() - calls_before;
        let ep_calls = self.gpu.ep_calls() - ep_before;
        let trunk_layers =
            (self.gpu.model.mc.n_layer - self.gpu.model.mc.nextn_predict_layers) as u64;
        let steps = tokens.len().saturating_sub(1) as u64;
        let graph_b_stats = self
            .gpu
            .graph_b_stats_for_state(&state)
            .expect("Graph-B stats");
        let graph_b_coverage = graph_b_coverage(&graph_b_stats);
        if graph_b_enabled {
            // Keep the real census even if a later coverage/identity assertion fails.
            // This runs after the wall timer, never inside the performance interval.
            println!(
                "GRAPH_B_CENSUS {{\"prompt\":{},\"ordinal\":{},\"captures\":{},\"replays\":{},\"variants\":[{}]}}",
                self.prompt_len,
                ordinal,
                graph_b_stats.captures,
                graph_b_stats.replays,
                graph_b_variants_json(&graph_b_stats)
            );
        }
        let graph_b_variant_bound = trunk_layers as usize * 8;
        let graph_b_max_variants_per_layer = graph_b_coverage
            .variants_per_layer
            .values()
            .copied()
            .max()
            .unwrap_or(0);
        assert_eq!(ep_calls, steps * trunk_layers, "whole trunk EP engagement");
        let wo_a_calls = self.gpu.dense_wo_a_grouped_dispatches() - wo_a_before;
        let index_topk_calls = self.gpu.index_topk_radix_dispatches() - index_topk_before;
        let indexed_layers = (0..trunk_layers as u32)
            .filter(|&layer| self.gpu.model.cfg().compress_ratio(layer) == 4)
            .count() as u64;
        assert_eq!(
            index_topk_calls,
            if index_topk_radix && self.prompt_len == 8192 {
                indexed_layers * steps
            } else {
                0
            },
            "actual top-k selector launches; the short-context control is inert by design"
        );
        if graph_b_enabled {
            assert_eq!(
                graph_b_stats.captures + graph_b_stats.replays,
                ep_calls,
                "Graph-B captures plus replays cover every trunk layer step"
            );
            assert_eq!(
                wo_a_calls + graph_b_stats.wo_a_replay_nodes,
                ep_calls,
                "legacy wo_a FFI plus actual graph replay nodes cover every trunk layer step"
            );
            assert_eq!(
                graph_b_stats.wo_a_replay_nodes,
                ep_calls - wo_a_calls,
                "Graph-B replay wo_a nodes are not a fabricated FFI counter"
            );
            assert_eq!(
                wo_a_calls, graph_b_stats.captures,
                "the legacy wo_a FFI count is capture-only; replay does not resubmit the body"
            );
            assert_eq!(
                graph_b_coverage.layers.len(),
                trunk_layers as usize,
                "Graph-B retained all trunk layers"
            );
            assert_eq!(
                graph_b_coverage.attention_variants,
                graph_b_stats.variants.len(),
                "every retained Graph-B variant contains attention and rope kernels"
            );
            assert_eq!(
                graph_b_coverage.output_variants,
                graph_b_stats.variants.len(),
                "every retained Graph-B variant contains output projection kernels"
            );
            assert_eq!(
                graph_b_coverage.hc_variants,
                graph_b_stats.variants.len(),
                "every retained Graph-B variant contains attention/FFN HC and norm kernels"
            );
            assert!(
                graph_b_stats.variants.len() <= graph_b_variant_bound
                    && graph_b_max_variants_per_layer <= 8,
                "Graph-B variant census exceeded the per-layer bound"
            );
        } else {
            assert_eq!(
                graph_b_stats.captures, 0,
                "Graph-B captures must be zero when inert"
            );
            assert_eq!(
                graph_b_stats.replays, 0,
                "Graph-B replays must be zero when inert"
            );
            assert_eq!(
                graph_b_stats.wo_a_replay_nodes, 0,
                "Graph-B replay nodes must be zero when inert"
            );
            assert!(
                graph_b_stats.variants.is_empty(),
                "inert Graph-B has no variants"
            );
            assert_eq!(
                wo_a_calls,
                if wo_a_grouped { ep_calls } else { 0 },
                "one grouped wo_a submission per trunk layer"
            );
        }
        assert_eq!(
            gu_m1_calls,
            if gu_m1 { ep_calls * 2 } else { 0 },
            "actual GU-M1 launches on both ranks"
        );
        let gu_half2_calls = memra_engine::moe_f16g_gu_half2_dispatches() - gu_half2_before;
        let down_half2_calls =
            memra_engine::moe_f16g_down_m1_half2_dispatches() - down_half2_before;
        assert_eq!(
            gu_half2_calls,
            if half2 { 2 * ep_calls } else { 0 },
            "actual packed-half2 GU enqueues on both ranks"
        );
        assert_eq!(
            down_half2_calls,
            if half2 { 2 * ep_calls } else { 0 },
            "actual packed-half2 down enqueues on both ranks"
        );
        // Full-state hashing and detokenization run outside the measurement.
        let mut cache_hash = Sha256::new();
        for (name, values) in self
            .gpu
            .cache_classes(&state)
            .expect("committed cache classes")
        {
            cache_hash.update((name.len() as u64).to_le_bytes());
            cache_hash.update(name.as_bytes());
            cache_hash.update((values.len() as u64).to_le_bytes());
            for value in values {
                cache_hash.update(value.to_bits().to_le_bytes());
            }
        }
        let cache_hash = format!("{:x}", cache_hash.finalize());
        let logits_hash = f32_hash(&row);
        let text = self.tokenizer.decode(&tokens);
        let excluded_loop = looped(&tokens);
        let mode = change.name();
        let prompt = self.prompt_len;
        let output_tokens = tokens.len();
        let eligible = !warmup && !excluded_loop && output_tokens >= 32;
        let token_sha256 = token_hash(&tokens);
        let text_sha256 = format!("{:x}", Sha256::digest(text.as_bytes()));
        let state_pos = state.pos;
        let graph_b_variants = graph_b_variants_json(&graph_b_stats);
        let graph_b_captures = graph_b_stats.captures;
        let graph_b_replays = graph_b_stats.replays;
        let graph_b_invalidations = graph_b_stats.invalidations;
        let graph_b_nodes = graph_b_stats.nodes;
        let graph_b_kernels = graph_b_stats.kernels;
        let graph_b_wo_a_replay_nodes = graph_b_stats.wo_a_replay_nodes;
        let graph_b_variant_count = graph_b_stats.variants.len();
        let graph_b_covered_layers = graph_b_coverage.layers.len();
        let graph_b_attention_coverage = graph_b_coverage.attention_variants;
        let graph_b_output_coverage = graph_b_coverage.output_variants;
        let graph_b_hc_coverage = graph_b_coverage.hc_variants;
        let graph_b_no_cpu_eager_body_under_replay =
            graph_b_enabled && wo_a_calls == graph_b_stats.captures;
        println!(
            "MEASURE {{\"mode\":\"{mode}\",\"arm\":\"plain\",\"prompt\":{prompt},\"tuned\":{tuned},\"warmup\":{warmup},\"ordinal\":{ordinal},\"eligible\":{eligible},\"looped\":{excluded_loop},\"eos\":{eos},\"output_tokens\":{output_tokens},\"decode_wall_ns\":{wall_ns},\"first_step_ns\":{first_step_ns},\"capture_included_in_wall\":{graph_b_enabled},\"start_unix_ms\":{start_ms},\"ep_calls\":{ep_calls},\"gu_fuse\":true,\"down_m1_tc\":true,\"route_validate\":false,\"mirror_validate\":false,\"gu_m1\":{gu_m1},\"gu_m1_calls\":{gu_m1_calls},\"wo_a_grouped\":{wo_a_grouped},\"wo_a_calls\":{wo_a_calls},\"index_topk_radix\":{index_topk_radix},\"index_topk_calls\":{index_topk_calls},\"half2\":{half2},\"gu_half2_calls\":{gu_half2_calls},\"down_half2_calls\":{down_half2_calls},\"prefix_graph\":false,\"expert_graph\":false,\"graph_captures\":{graph_b_captures},\"graph_replays\":{graph_b_replays},\"graph_kernel_nodes\":{graph_b_kernels},\"graph_fallbacks\":0,\"c4_recent_rows\":0,\"c4_host_copy_elide\":false,\"graph_b_enabled\":{graph_b_enabled},\"graph_b_captures\":{graph_b_captures},\"graph_b_replays\":{graph_b_replays},\"graph_b_invalidations\":{graph_b_invalidations},\"graph_b_nodes\":{graph_b_nodes},\"graph_b_kernels\":{graph_b_kernels},\"graph_b_wo_a_replay_nodes\":{graph_b_wo_a_replay_nodes},\"graph_b_no_cpu_eager_body_under_replay\":{graph_b_no_cpu_eager_body_under_replay},\"graph_b_variant_count\":{graph_b_variant_count},\"graph_b_covered_layers\":{graph_b_covered_layers},\"graph_b_attention_coverage\":{graph_b_attention_coverage},\"graph_b_output_coverage\":{graph_b_output_coverage},\"graph_b_hc_coverage\":{graph_b_hc_coverage},\"graph_b_hc_variant_count\":{graph_b_hc_coverage},\"graph_b_max_variants_per_layer\":{graph_b_max_variants_per_layer},\"graph_b_variant_bound\":{graph_b_variant_bound},\"graph_b_variants\":[{graph_b_variants}],\"token_sha256\":\"{token_sha256}\",\"logits_sha256\":\"{logits_hash}\",\"cache_sha256\":\"{cache_hash}\",\"text_sha256\":\"{text_sha256}\",\"commit_ns\":{commits:?},\"tokens\":{tokens:?},\"state_pos\":{state_pos}}}"
        );
        self.gpu
            .clear_graph_b_for_state(&mut state)
            .expect("clear Graph-B state");
        Identity {
            tokens,
            logits: logits_hash,
            cache: cache_hash,
            pos: state.pos,
        }
    }
}

fn main() {
    let args: Vec<_> = std::env::args().collect();
    assert_eq!(
        args.len(),
        4,
        "usage: dsv4_plain_perf_gate <model-dir> <source.txt> gu-m1|half2|wo-a|index-topk|graph-b|all"
    );
    let modes = match args[3].as_str() {
        "gu-m1" => vec![Change::GuM1],
        "half2" => vec![Change::Half2],
        "wo-a" => vec![Change::WoA],
        "index-topk" => vec![Change::IndexTopk],
        "graph-b" => vec![Change::GraphB],
        "all" => vec![Change::IndexTopk],
        _ => panic!("unknown plain gate mode"),
    };
    for (name, expected) in [
        ("MEMRA_DSV4_DECODE_PATH", "device"),
        ("MEMRA_DSV4_EXPERT_ARM", "native"),
        ("MEMRA_DSV4_DRAFTER", "dspark"),
        ("MEMRA_DSV4_DENSE_ARM", "fp8"),
        ("MEMRA_DSV4_EP", "pair"),
        ("MEMRA_DSV4_MOE_PROGRAM", "matrix"),
        ("MEMRA_DSV4_GROUPED_ROUTE", "device"),
        ("MEMRA_DSV4_VERIFY_TOPK", "device"),
        ("MEMRA_DSV4_SAMPLE_SORT", "radix"),
        ("MEMRA_DSV4_INDEXER_SCORE", "tiled"),
        ("MEMRA_DSV4_SINK_SCORE", "tiled"),
        ("MEMRA_DSV4_PREFILL_MOE", "reference"),
    ] {
        assert_eq!(
            std::env::var(name).as_deref(),
            Ok(expected),
            "requires {name}={expected}"
        );
    }
    let dir = Path::new(&args[1]);
    let source = std::fs::read_to_string(&args[2]).expect("source");
    assert_eq!(
        format!("{:x}", Sha256::digest(source.as_bytes())),
        "f6e175a6f2588953568746fec0cd43fcd046405f74b5c71ce071fe7f37238ded",
        "pinned source"
    );
    let tokenizer = Tokenizer::from_hf_dir(dir).expect("tokenizer");
    let input = tokenizer.encode(
        &format!("Review this inference engine source:\n\n{source}"),
        true,
    );
    assert!(input.len() >= 8192);
    let rows_per_arm = 2 * REPEATS;
    let mode_names: Vec<_> = modes.iter().map(|mode| mode.name()).collect();
    println!(
        "PROTOCOL {{\"plain_only\":true,\"http\":false,\"new_tokens\":{OUTPUT},\"abba_repeats\":{REPEATS},\"rows_per_arm\":{rows_per_arm},\"prompts\":[256,8192],\"modes\":{mode_names:?},\"temperature\":1.0,\"top_p\":1.0,\"top_k\":0,\"seed\":20260906,\"capture_cost_included\":true,\"restore_cost_included\":false}}"
    );
    let gpu = Dsv4Gpu::load(
        dir,
        &[0, 1],
        ActQuantVariant::RefFp8Round,
        8192 + OUTPUT + 96,
    )
    .expect("model");
    assert!(gpu.matrix_moe_enabled());
    gpu.set_grouped_mirror_validation_for_gate(false);
    gpu.set_grouped_route_validation_for_gate(false);
    gpu.set_grouped_gu_fuse_for_gate(true);
    gpu.set_grouped_m1_tc_for_gate(true);
    gpu.set_c4_host_copy_elision_for_gate(false);
    for count in [256, 8192] {
        let capacity = count + OUTPUT + 96;
        let mut state = gpu
            .alloc_decode_state_host_c4(capacity, 512)
            .expect("host state");
        let mut draft = gpu.dspark_alloc_state().expect("draft prime state");
        let logits = gpu
            .dspark_prefill_prime_chunked(&input[..count], &mut state, &mut draft, 512)
            .expect("prime");
        let snapshot = gpu.snapshot_decode_state(&state).expect("snapshot");
        drop(state);
        drop(draft);
        let ready = Ready {
            gpu: &gpu,
            tokenizer: &tokenizer,
            prompt_len: count,
            logits: &logits,
            snapshot: &snapshot,
        };
        for &mode in &modes {
            let expected = ready.run(mode, false, true, 0);
            let frozen = if count == 256 {
                "6a7d95b9ebe8714ab3ca75b6cbd8feced699937f08d605fd3f822a37bbdd115b"
            } else {
                "7af072c1cdbde293dfa88b1dff10326d33ae6e7531eecb7b43585e3ba9146f33"
            };
            assert_eq!(
                token_hash(&expected.tokens),
                frozen,
                "baseline frozen stream"
            );
            assert_eq!(
                expected,
                ready.run(mode, true, true, 1),
                "warmup full output/logits/cache identity"
            );
            for (i, tuned) in [false, true, true, false]
                .repeat(REPEATS)
                .into_iter()
                .enumerate()
            {
                assert_eq!(
                    expected,
                    ready.run(mode, tuned, false, i + 2),
                    "timed output/logits/cache identity"
                );
            }
            drain(&gpu);
            memra_engine::clear_moe_f16g_gu_m1_tc_for_gate();
            memra_engine::clear_moe_f16g_gu_half2_for_gate();
            memra_engine::clear_moe_f16g_down_m1_half2_for_gate();
            gpu.clear_dense_wo_a_grouped_for_gate();
            gpu.clear_index_topk_radix_for_gate();
            println!(
                "PASS mode={} prompt={count} both_arms=engaged outputs=exact logits=exact cache=exact",
                mode.name()
            );
        }
    }
    println!("PASS plain-only sampled performance gate");
}

#[cfg(test)]
mod coverage_tests {
    use super::*;

    fn real_symbol_variant() -> Dsv4GraphBVariantStats {
        Dsv4GraphBVariantStats {
            layer: 2,
            stage: 0,
            attention_topk: 512,
            slots: 1024,
            arm: 0,
            nodes: BTreeMap::new(),
            kernels: [
                "dsv4_sink_scores_tiled_f32acc_kernel",
                "dsv4_sink_soft_mq_f32acc_kernel",
                "dsv4_sink_out_mq_f32acc_kernel",
                "dsv4_rope_kernel",
                "_Z22dsv4_gemv_fp8_m_kernelILi1ELb1EEv",
                "_Z22dsv4_gemv_fp8_m_kernelILi1ELb0EEv",
                "dsv4_cvt_bf16_kernel",
                "dsv4_hc_post_kernel",
                "dsv4_hc_sinkhorn_m_kernel",
                "dsv4_hc_collapse_kernel",
                "dsv4_rowsq_scale_f32acc_kernel",
                "_Z29dsv4_dots_f32acc_mrow_kernelILi1EEv",
                "dsv4_rmsnorm_f32acc_kernel",
            ]
            .into_iter()
            .map(str::to_owned)
            .collect(),
        }
    }

    #[test]
    fn raw_cuda_template_case_is_preserved_by_coverage() {
        assert_eq!(
            graph_b_kernel_coverage(&real_symbol_variant()),
            (true, true, true)
        );
    }

    #[test]
    fn wrapper_name_cannot_replace_grouped_device_kernel() {
        let mut variant = real_symbol_variant();
        variant.kernels[4] = "memra_dsv4_gemv_fp8_grouped_m1".into();
        assert!(!graph_b_kernel_coverage(&variant).1);
    }

    #[test]
    fn wrong_batch_template_cannot_prove_plain_output_coverage() {
        let mut variant = real_symbol_variant();
        variant.kernels[4] = "_Z22dsv4_gemv_fp8_m_kernelILi2ELb1EEv".into();
        assert!(!graph_b_kernel_coverage(&variant).1);
    }
}
