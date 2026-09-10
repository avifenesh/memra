//! Plain-only, sampled ABBA over a frozen prompt snapshot. No speculative timing rows.
use memra_engine::dsv4_gpu::{Dsv4Gpu, Dsv4HostDecodeState, Dsv4SampleCfg, dsv4_sample_row};
use memra_gguf::dsv4_forward::ActQuantVariant;
use memra_tokenizer::Tokenizer;
use sha2::{Digest, Sha256};
use std::{
    path::Path,
    time::{Instant, SystemTime, UNIX_EPOCH},
};

const OUTPUT: usize = 256;
const REPEATS: usize = 3;

#[derive(Clone, Copy, Debug)]
enum Change {
    GuM1,
    GuM1Half2,
    Half2,
    WoA,
    IndexTopk,
}
impl Change {
    fn name(self) -> &'static str {
        match self {
            Self::GuM1 => "gu-m1",
            Self::GuM1Half2 => "gu-m1-half2",
            Self::Half2 => "half2",
            Self::WoA => "wo-a",
            Self::IndexTopk => "index-topk",
        }
    }
    fn gu_m1(self, tuned: bool) -> bool {
        tuned && matches!(self, Self::GuM1 | Self::GuM1Half2)
    }
    fn half2(self, tuned: bool) -> bool {
        matches!(self, Self::WoA | Self::IndexTopk | Self::GuM1Half2)
            || (tuned && matches!(self, Self::Half2))
    }
    fn wo_a(self, tuned: bool) -> bool {
        matches!(self, Self::IndexTopk | Self::GuM1Half2) || (tuned && matches!(self, Self::WoA))
    }
    fn index_topk(self, tuned: bool) -> bool {
        matches!(self, Self::GuM1Half2) || (tuned && matches!(self, Self::IndexTopk))
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
        let gu_m1_half2_composed = tuned && matches!(change, Change::GuM1Half2);
        let half2 = change.half2(tuned);
        let wo_a_grouped = change.wo_a(tuned);
        let index_topk_radix = change.index_topk(tuned);
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
        assert_eq!(
            wo_a_calls,
            if wo_a_grouped { ep_calls } else { 0 },
            "one grouped wo_a submission per trunk layer"
        );
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
        println!(
            "MEASURE {{\"mode\":\"{mode}\",\"arm\":\"plain\",\"prompt\":{prompt},\"tuned\":{tuned},\"warmup\":{warmup},\"ordinal\":{ordinal},\"eligible\":{eligible},\"looped\":{excluded_loop},\"eos\":{eos},\"output_tokens\":{output_tokens},\"decode_wall_ns\":{wall_ns},\"first_step_ns\":{first_step_ns},\"capture_included_in_wall\":false,\"start_unix_ms\":{start_ms},\"ep_calls\":{ep_calls},\"gu_fuse\":true,\"down_m1_tc\":true,\"route_validate\":false,\"mirror_validate\":false,\"gu_m1\":{gu_m1},\"gu_m1_half2_composed\":{gu_m1_half2_composed},\"gu_m1_calls\":{gu_m1_calls},\"wo_a_grouped\":{wo_a_grouped},\"wo_a_calls\":{wo_a_calls},\"index_topk_radix\":{index_topk_radix},\"index_topk_calls\":{index_topk_calls},\"half2\":{half2},\"gu_half2_calls\":{gu_half2_calls},\"down_half2_calls\":{down_half2_calls},\"prefix_graph\":false,\"expert_graph\":false,\"graph_captures\":0,\"graph_replays\":0,\"graph_kernel_nodes\":0,\"graph_fallbacks\":0,\"c4_recent_rows\":0,\"c4_host_copy_elide\":false,\"token_sha256\":\"{token_sha256}\",\"logits_sha256\":\"{logits_hash}\",\"cache_sha256\":\"{cache_hash}\",\"text_sha256\":\"{text_sha256}\",\"commit_ns\":{commits:?},\"tokens\":{tokens:?},\"state_pos\":{state_pos}}}"
        );
        Identity {
            tokens,
            logits: logits_hash,
            cache: cache_hash,
            pos: state.pos,
        }
    }
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
    }

    let args: Vec<_> = std::env::args().collect();
    assert_eq!(
        args.len(),
        4,
        "usage: dsv4_plain_perf_gate <model-dir> <source.txt> gu-m1|gu-m1-half2|half2|wo-a|index-topk|all"
    );
    let modes = match args[3].as_str() {
        "gu-m1" => vec![Change::GuM1],
        "gu-m1-half2" => vec![Change::GuM1Half2],
        "half2" => vec![Change::Half2],
        "wo-a" => vec![Change::WoA],
        "index-topk" => vec![Change::IndexTopk],
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
