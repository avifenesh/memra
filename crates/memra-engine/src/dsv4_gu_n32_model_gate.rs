//! Fresh-process GU selector qualification and five-row replay timing blocks.
//! The process-fixed production selector is never changed after model creation.
use memra_engine::dsv4_gpu::{DecodeState, Dsv4Gpu, Dsv4Phase, Dsv4SampleCfg};
use memra_engine::dsv4_sampler::{Dsv4Sampler, dsv4_sampler};
use memra_gguf::dsv4_forward::ActQuantVariant;
use memra_tokenizer::Tokenizer;
use sha2::{Digest, Sha256};
use std::{fs, io::Write, path::Path, time::Instant};

const PRIME: usize = 256;
const OUTPUT: usize = 256;
const CAPACITY: usize = PRIME + OUTPUT + 8;
const INPUT_SHA: &str = "f6e175a6f2588953568746fec0cd43fcd046405f74b5c71ce071fe7f37238ded";
const COMPONENT_SHA: &str = "eada0c1659c2a1fdfb854b76ba6081fa1eea3457";

fn runtime_sha() -> String {
    let files: &[(&str, &[u8])] = &[
        ("gu_n32", include_bytes!("../cu/dsv4_gu_n32.cuh")),
        ("grouped_cuda", include_bytes!("../cu/moe_f16_grouped.cu")),
        ("grouped_rust", include_bytes!("dsv4_grouped.rs")),
        ("ffi", include_bytes!("mmq_ffi.rs")),
        ("dsv4_cuda", include_bytes!("../cu/dsv4_gpu.cu")),
        ("dsv4_runtime", include_bytes!("dsv4_gpu.rs")),
        ("replay", include_bytes!("dsv4_graph.rs")),
        ("sampler", include_bytes!("dsv4_sampler.rs")),
    ];
    let mut hash = Sha256::new();
    for (name, bytes) in files {
        hash.update(name.as_bytes());
        hash.update((bytes.len() as u64).to_le_bytes());
        hash.update(bytes);
    }
    format!("{:x}", hash.finalize())
}
fn sha_tokens(tokens: &[u32]) -> String {
    let mut h = Sha256::new();
    for x in tokens {
        h.update(x.to_le_bytes());
    }
    format!("{:x}", h.finalize())
}
#[derive(Clone, Debug, PartialEq, Eq)]
struct Identity {
    logits: String,
    cache: [u64; 2],
    hidden: [u64; 2],
}
impl Identity {
    fn read(gpu: &Dsv4Gpu, state: &DecodeState) -> Self {
        let mut h = Sha256::new();
        for v in gpu
            .read_decode_logits_for_gate(state)
            .expect("logits readback")
        {
            assert!(v.is_finite(), "nonfinite model logit");
            h.update(v.to_bits().to_le_bytes());
        }
        Self {
            logits: format!("{:x}", h.finalize()),
            cache: gpu.tp_ep_cache_digest_for_gate(state).unwrap(),
            hidden: gpu.tp_ep_hidden_digest_for_gate(state).unwrap(),
        }
    }
    fn fields(&self) -> String {
        format!(
            "{}\t{}\t{}\t{}\t{}",
            self.logits, self.cache[0], self.cache[1], self.hidden[0], self.hidden[1]
        )
    }
    fn parse(fields: &[&str]) -> Self {
        assert_eq!(fields.len(), 5);
        assert_eq!(fields[0].len(), 64);
        assert!(fields[0].bytes().all(|x| x.is_ascii_hexdigit()));
        Self {
            logits: fields[0].into(),
            cache: [fields[1].parse().unwrap(), fields[2].parse().unwrap()],
            hidden: [fields[3].parse().unwrap(), fields[4].parse().unwrap()],
        }
    }
}
struct Step {
    position: usize,
    input: u32,
    next: u32,
    state: Identity,
}
struct Reference {
    first: u32,
    prefix: Identity,
    steps: Vec<Step>,
}
impl Reference {
    fn load(path: &Path) -> Self {
        let text = fs::read_to_string(path).expect("OFF reference file");
        let mut lines = text.lines();
        assert_eq!(
            lines.next().unwrap(),
            format!(
                "GU_N32_REFERENCE_V1\t{}\t{INPUT_SHA}\t{COMPONENT_SHA}",
                runtime_sha()
            )
        );
        let first: Vec<_> = lines.next().unwrap().split('\t').collect();
        assert_eq!(first.len(), 8);
        assert_eq!(first[0], "PREFIX");
        assert_eq!(first[1], "256");
        let mut result = Self {
            first: first[2].parse().unwrap(),
            prefix: Identity::parse(&first[3..]),
            steps: Vec::new(),
        };
        for line in lines {
            let f: Vec<_> = line.split('\t').collect();
            assert_eq!(f.len(), 9);
            assert_eq!(f[0], "STEP");
            let step = Step {
                position: f[1].parse().unwrap(),
                input: f[2].parse().unwrap(),
                next: f[3].parse().unwrap(),
                state: Identity::parse(&f[4..]),
            };
            assert_eq!(step.position, PRIME + result.steps.len() + 1);
            assert_eq!(
                step.input,
                result.steps.last().map_or(result.first, |s: &Step| s.next)
            );
            result.steps.push(step);
        }
        assert_eq!(result.steps.len(), OUTPUT);
        result
    }
    fn write(&self, path: &Path) {
        let mut f = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)
            .expect("new OFF reference");
        writeln!(
            f,
            "GU_N32_REFERENCE_V1\t{}\t{INPUT_SHA}\t{COMPONENT_SHA}",
            runtime_sha()
        )
        .unwrap();
        writeln!(
            f,
            "PREFIX\t{PRIME}\t{}\t{}",
            self.first,
            self.prefix.fields()
        )
        .unwrap();
        for s in &self.steps {
            writeln!(
                f,
                "STEP\t{}\t{}\t{}\t{}",
                s.position,
                s.input,
                s.next,
                s.state.fields()
            )
            .unwrap();
        }
        f.sync_all().unwrap();
    }
}
fn state(gpu: &Dsv4Gpu) -> DecodeState {
    gpu.alloc_decode_state_for_transient(CAPACITY, 1).unwrap()
}
fn drain(gpu: &Dsv4Gpu) {
    for stage in &gpu.stages {
        stage.gpu.stream().synchronize().unwrap();
    }
}
fn check_epochs(gpu: &Dsv4Gpu, before: &[Vec<u32>; 2], steps: u32) {
    let after = gpu.full_token_ar_epochs_for_gate().unwrap();
    let a = memra_engine::tp_ar::ar_blocks_for(4096) as usize;
    let e = memra_engine::tp_ar::ar_blocks_for(6 * 4096) as usize;
    for rank in 0..2 {
        assert_eq!(after[rank].len(), 72);
        for b in 0..72 {
            assert_eq!(
                after[rank][b].wrapping_sub(before[rank][b]),
                43 * (u32::from(b < a) + u32::from(b < e)) * steps,
                "epoch rank{rank} block{b}"
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
    for rank in gpu.full_token_replay_census_for_gate(state).unwrap() {
        assert_eq!(rank[0][2], 86);
        assert_eq!(rank[0][3], 1);
        assert_eq!(rank[0][4], 86);
        assert_eq!(rank[0][6], 0);
        assert_eq!(rank[1][6], 0);
    }
}
fn node_census(gpu: &Dsv4Gpu, state: &DecodeState, dir: &Path, on: bool) {
    fs::create_dir(dir).expect("new graph dump directory");
    gpu.dump_full_token_replay_for_gate(state, dir).unwrap();
    for rank in 0..2 {
        for segment in 0..2 {
            let text =
                fs::read_to_string(dir.join(format!("full-token-rank{rank}-segment{segment}.dot")))
                    .unwrap();
            let candidate = text.matches("dsv4_gu_n32_kernel").count();
            let control = text
                .matches("moe_kq_sktail_gu_kernelILi108ELb1ELb1E")
                .count();
            let all_gu = text.matches("moe_kq_sktail_gu_kernel").count();
            let expected = if segment == 0 { 43 } else { 0 };
            assert_eq!(
                candidate,
                if on { expected } else { 0 },
                "retained N32 nodes"
            );
            assert_eq!(
                control,
                if on { 0 } else { expected },
                "retained control nodes"
            );
            assert_eq!(all_gu, control, "unexpected GU specialization");
            println!(
                "GU_GRAPH_CENSUS rank={rank} segment={segment} n32={candidate} current={control} source=retained_cuda_graph_dot execution_proof=separate_qualification_trace"
            );
        }
    }
}
fn arm(gpu: &Dsv4Gpu, prefix: &DecodeState, cfg: Dsv4SampleCfg) -> DecodeState {
    let mut s = state(gpu);
    gpu.restore_full_token_prefix_for_gate(&mut s, prefix)
        .unwrap();
    // The model remains borrowed, and its configuration/allocations are never
    // changed during any of this process's separate request graph lifetimes.
    unsafe { gpu.arm_full_token_replay_for_gate(&mut s, cfg) }.unwrap();
    s
}
fn refusal_cells(gpu: &Dsv4Gpu, prefix: &DecodeState, cfg: Dsv4SampleCfg, reference: &Reference) {
    for rank in 0..2 {
        for (position, layer) in [(259usize, 0usize), (383, 21), (511, 42)] {
            let mut failed = arm(gpu, prefix, cfg);
            for s in &reference.steps[..position - PRIME] {
                gpu.decode_sample_full_token_for_gate(s.input, &mut failed)
                    .unwrap();
            }
            let before = gpu.tp_ep_cache_digest_for_gate(&failed).unwrap();
            let counts = gpu.full_token_replay_counts_for_gate(&failed).unwrap();
            let code = 40043 + rank as i32;
            gpu.arm_attention_tp_join_refusal_for_gate(layer, rank, code)
                .unwrap();
            let err = gpu
                .decode_sample_full_token_for_gate(
                    reference.steps[position - PRIME].input,
                    &mut failed,
                )
                .unwrap_err();
            assert!(err.contains("one-shot reduction refused"), "{err}");
            let mut words = [0, 0];
            words[rank] = code;
            assert_eq!(gpu.tp_ep_ar_refusal_words().unwrap(), words);
            assert_eq!(failed.pos, position);
            assert_eq!(gpu.tp_ep_cache_digest_for_gate(&failed).unwrap(), before);
            let after = gpu.full_token_replay_counts_for_gate(&failed).unwrap();
            for r in 0..2 {
                assert_eq!(after[r], [counts[r][0] + 1, counts[r][1]]);
            }
            assert!(
                gpu.decode_sample_full_token_for_gate(
                    reference.steps[position - PRIME].input,
                    &mut failed
                )
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
            capture_once(gpu, &failed);
            println!(
                "GU_REFUSAL rank={rank} layer={layer} position={position} cache_unchanged=true commit_delta=0 retry_quarantined=true"
            );
            gpu.set_tp_ep_ar_refusal_words_for_gate([0, 0]).unwrap();
        }
    }
}
fn looped(tokens: &[u32]) -> bool {
    (1usize..=32).any(|w| {
        let n = w * 4usize.max(32usize.div_ceil(w));
        tokens
            .windows(n)
            .any(|s| s.chunks_exact(w).all(|c| c == &s[..w]))
    })
}

pub fn run() {
    let args: Vec<String> = std::env::args().collect();
    assert_eq!(
        args.len(),
        6,
        "usage: gu-model-gate model source reference.tsv output-directory reference|qualify|block"
    );
    let mode = args[5].as_str();
    assert!(matches!(mode, "reference" | "qualify" | "block"));
    let timing = mode == "block";
    let on = match std::env::var("MEMRA_DSV4_GU_N32").as_deref() {
        Ok("0") => false,
        Ok("1") => true,
        _ => panic!("explicit GU selector 0/1 required"),
    };
    assert!(mode != "reference" || !on, "only OFF creates reference");
    assert!(
        mode != "qualify" || on,
        "ON qualification consumes OFF reference"
    );
    for (name, value) in [
        ("MEMRA_DSV4_DECODE_PATH", "device"),
        ("MEMRA_DSV4_EXPERT_ARM", "native"),
        ("MEMRA_DSV4_DENSE_ARM", "fp8"),
        ("MEMRA_DSV4_EP", "pair"),
        ("MEMRA_DSV4_MOE_PROGRAM", "matrix"),
        ("MEMRA_DSV4_GROUPED_ROUTE", "device"),
        ("MEMRA_DSV4_VERIFY_TOPK", "device"),
        ("MEMRA_DSV4_PREFILL_MOE", "reference"),
        ("MEMRA_DSV4_SMALL_KERNEL_DIET", "1"),
        ("MEMRA_DSV4_SAMPLER", "device"),
        ("MEMRA_DSV4_DRAFTER", "off"),
        ("MEMRA_DSV4_DOTS_ARM", "f32x"),
        ("MEMRA_MOE_F16G", "2"),
        ("MEMRA_F16G_SK", "32"),
    ] {
        assert_eq!(
            std::env::var(name).as_deref(),
            Ok(value),
            "requires {name}={value}"
        );
    }
    assert!(
        std::env::var_os("MEMRA_DSV4_GU_N32_CAPTURE").is_none(),
        "operand capture must be absent in model gate"
    );
    for (name, value) in std::env::vars() {
        if name.starts_with("MEMRA_")
            && (name.contains("CADENCE")
                || name.contains("DENSE_EXACT")
                || name.contains("DENSE_TAIL")
                || name.contains("SPLITK")
                || name.contains("SPLIT_K")
                || name == "MEMRA_DSV4_GU_N32_CAPTURE")
        {
            assert!(
                value.is_empty() || value == "0",
                "unrelated arm enabled: {name}"
            );
        }
    }
    assert!(!memra_engine::moe_m1_splitk_on());
    memra_engine::set_moe_m1_splitk_for_gate(false);
    assert_eq!(dsv4_sampler().unwrap(), Dsv4Sampler::Device);
    assert_ne!(
        std::env::var("MEMRA_DSV4_ROUND_PROFILE").as_deref(),
        Ok("1")
    );
    if timing {
        assert!(
            !memra_engine::dsv4_gpu::dsv4_prof_on(),
            "scored process must be unprofiled"
        );
        assert!(std::env::var("NSYS_PROFILING_SESSION_ID").is_err());
    } else {
        assert_eq!(
            std::env::var("MEMRA_DSV4_NVTX").as_deref(),
            Ok("1"),
            "qualification requires its execution-evidence NVTX range"
        );
    }
    let out = Path::new(&args[4]);
    fs::create_dir(out).expect("new output directory");
    let source = fs::read_to_string(&args[2]).unwrap();
    assert_eq!(
        format!("{:x}", Sha256::digest(source.as_bytes())),
        INPUT_SHA
    );
    let tokenizer = Tokenizer::from_hf_dir(Path::new(&args[1])).unwrap();
    let prompt = tokenizer.encode(
        &format!("Review this inference engine source:\n\n{source}"),
        true,
    );
    assert!(prompt.len() >= PRIME);
    Dsv4Gpu::set_tp_ep_topology_for_gate(true);
    Dsv4Gpu::set_attention_tp_for_gate(true);
    let mut gpu = Dsv4Gpu::load(
        Path::new(&args[1]),
        &[0, 1],
        ActQuantVariant::RefFp8Round,
        PRIME + OUTPUT + 32,
    )
    .unwrap();
    gpu.set_grouped_route_validation_for_gate(false);
    gpu.set_grouped_mirror_validation_for_gate(false);
    gpu.set_grouped_gu_fuse_for_gate(true);
    gpu.set_grouped_m1_tc_for_gate(true);
    memra_engine::set_moe_f16g_gu_m1_tc_for_gate(true);
    memra_engine::set_moe_f16g_gu_half2_for_gate(true);
    memra_engine::set_moe_f16g_down_m1_half2_for_gate(true);
    gpu.set_dense_wo_a_grouped_for_gate(false);
    gpu.set_index_topk_radix_for_gate(true);
    gpu.set_small_kernel_diet_for_gate(true).unwrap();
    let cfg = Dsv4SampleCfg {
        temperature: 1.0,
        top_p: 1.0,
        top_k: 0,
        seed: 20260907,
    };
    let mut prefix = state(&gpu);
    gpu.prefill_with_cache_chunked(&prompt[..1], &mut prefix, 1)
        .unwrap();
    for &t in &prompt[1..PRIME] {
        gpu.decode_step_device_logits(t, &mut prefix).unwrap();
    }
    let mut sampler = gpu.device_sampler().unwrap();
    let first = gpu
        .sample_device_logits(&prefix, &mut sampler, &cfg, &[], None)
        .unwrap();
    let prefix_id = Identity::read(&gpu, &prefix);
    assert_eq!(prefix.pos, PRIME);
    println!(
        "GU_MODEL_PROTOCOL mode={mode} selector={} runtime_sha256={} component_source={COMPONENT_SHA} prime=256 output=256 device_sampler=true diet=true splitk=false cadence=false dense_tail=false separate_process_graph_states=true first_capture_inside_scored_row=true",
        usize::from(on),
        runtime_sha()
    );
    let reference_path = Path::new(&args[3]);
    if !timing {
        let gold = if on {
            Some(Reference::load(reference_path))
        } else {
            None
        };
        if let Some(g) = &gold {
            assert_eq!(g.first, first);
            assert_eq!(g.prefix, prefix_id);
        }
        let mut result = Reference {
            first,
            prefix: prefix_id,
            steps: Vec::new(),
        };
        let mut graph = arm(&gpu, &prefix, cfg);
        let mut eager = state(&gpu);
        gpu.restore_full_token_prefix_for_gate(&mut eager, &prefix)
            .unwrap();
        let mut carry = first;
        let mut nvtx = None;
        for step in 0..OUTPUT {
            assert_ne!(carry, tokenizer.eos_id());
            let input = carry;
            let eager_expected = if !on {
                let before = gpu.full_token_ar_epochs_for_gate().unwrap();
                gpu.decode_step_device_logits(input, &mut eager).unwrap();
                let next = gpu
                    .sample_device_logits(&eager, &mut sampler, &cfg, &[], None)
                    .unwrap();
                check_epochs(&gpu, &before, 1);
                Some((next, Identity::read(&gpu, &eager)))
            } else {
                None
            };
            if step == 112 {
                drain(&gpu);
                nvtx = Dsv4Phase::new("GU_N32_EXEC\0", None);
            }
            let epochs = gpu.full_token_ar_epochs_for_gate().unwrap();
            carry = gpu
                .decode_sample_full_token_for_gate(input, &mut graph)
                .unwrap();
            if step == 143 {
                drain(&gpu);
                drop(nvtx.take());
            }
            check_epochs(&gpu, &epochs, 1);
            assert_eq!(gpu.tp_ep_ar_refusal_words().unwrap(), [0, 0]);
            assert_eq!(graph.pos, PRIME + step + 1);
            let id = Identity::read(&gpu, &graph);
            if let Some((next, expected)) = eager_expected {
                assert_eq!(carry, next);
                assert_eq!(id, expected);
            }
            if let Some(g) = &gold {
                let expected = &g.steps[step];
                assert_eq!(input, expected.input);
                assert_eq!(carry, expected.next);
                assert_eq!(id, expected.state);
            }
            capture_once(&gpu, &graph);
            assert_eq!(
                gpu.full_token_replay_counts_for_gate(&graph).unwrap(),
                [[step as u64 + 1; 2]; 2]
            );
            println!(
                "GU_STEP position={} input={input} next={carry} logits={} cache={:?} hidden={:?} epochs_exact=true refusals_zero=true",
                graph.pos, id.logits, id.cache, id.hidden
            );
            result.steps.push(Step {
                position: graph.pos,
                input,
                next: carry,
                state: id,
            });
        }
        assert!(!looped(
            &result.steps.iter().map(|s| s.input).collect::<Vec<_>>()
        ));
        node_census(&gpu, &graph, &out.join("qualified-graphs"), on);
        drop(graph);
        refusal_cells(&gpu, &prefix, cfg, &result);
        if !on {
            result.write(reference_path);
        }
        println!(
            "PASS GU model qualification selector={} steps=256 all_state_bits=true six_refusal_cells=true execution_trace_required=true no_scored_rows=true",
            usize::from(on)
        );
        return;
    }
    let gold = Reference::load(reference_path);
    assert_eq!(first, gold.first);
    assert_eq!(prefix_id, gold.prefix);
    let expected_tokens: Vec<_> = gold.steps.iter().map(|s| s.input).collect();
    let token_hash = sha_tokens(&expected_tokens);
    let final_id = &gold.steps.last().unwrap().state;
    let mut graph = arm(&gpu, &prefix, cfg);
    for row in 0..5 {
        gpu.restore_full_token_prefix_for_gate(&mut graph, &prefix)
            .unwrap();
        assert_eq!(graph.pos, PRIME);
        assert_eq!(
            Identity::read(&gpu, &graph),
            gold.prefix,
            "row reset identity"
        );
        let epochs = gpu.full_token_ar_epochs_for_gate().unwrap();
        let before = gpu.full_token_replay_counts_for_gate(&graph).unwrap();
        let mut carry = first;
        let mut tokens = Vec::with_capacity(OUTPUT);
        let start = Instant::now();
        for _ in 0..OUTPUT {
            assert_ne!(carry, tokenizer.eos_id());
            tokens.push(carry);
            carry = gpu
                .decode_sample_full_token_for_gate(carry, &mut graph)
                .unwrap();
        }
        drain(&gpu);
        let elapsed = start.elapsed().as_nanos();
        assert!(!looped(&tokens));
        assert_eq!(tokens, expected_tokens);
        assert_eq!(carry, gold.steps.last().unwrap().next);
        assert_eq!(&Identity::read(&gpu, &graph), final_id);
        assert_eq!(gpu.tp_ep_ar_refusal_words().unwrap(), [0, 0]);
        check_epochs(&gpu, &epochs, OUTPUT as u32);
        capture_once(&gpu, &graph);
        let after = gpu.full_token_replay_counts_for_gate(&graph).unwrap();
        for r in 0..2 {
            for s in 0..2 {
                assert_eq!(after[r][s] - before[r][s], OUTPUT as u64);
            }
        }
        println!(
            r#"MEASURE {{"row":{row},"arm":"{}","generated_tokens":256,"decode_wall_ns":{elapsed},"decode_tok_s":{},"timing_scope":"sample_plus_forward_envelope","eligible":true,"looped":false,"generated_sha256":"{token_hash}","final_logits_sha256":"{}","final_cache_digest":{:?},"final_hidden_digest":{:?},"final_next_token":{carry},"device_replays":{after:?},"captures":[1,1],"first_capture_in_row":{},"device_sampler":true,"diet":true,"split_k":false,"cadence":false,"dense_tail":false}}"#,
            if on { "n32" } else { "current" },
            OUTPUT as f64 * 1e9 / elapsed as f64,
            final_id.logits,
            final_id.cache,
            final_id.hidden,
            row == 0
        );
        if row == 0 {
            node_census(&gpu, &graph, &out.join("scored-graphs"), on);
        }
    }
    println!(
        "PASS GU five-row block selector={} rows=5 all_digests=true no_profiler=true",
        usize::from(on)
    );
}
