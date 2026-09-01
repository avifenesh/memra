//! GPU-vs-reference gate for the mHC residual topology (`ResidualTopology::HyperConnections`).
//!
//! Truth is `memra_reference::execute` — the portable trunk executor whose `execute_hyper_layer`
//! is the arithmetic contract `crate::hyper` transcribes. The candidate is the WHOLE loaded
//! `HybridModel`: the fixture serves the real contract tensor names through a `TensorSource`, so
//! the loader (the six per-layer `hc_*` tensors, their shapes, the absent-tensor refusal) is under
//! gate alongside the kernels. Nothing here reaches into `crate::hyper`'s helpers directly; a
//! wrong wiring in `forward`/`prime_cache`/`decode_step` fails these tests.
//!
//! TRUNK-SCOPED BY DESIGN, MIXER-NEUTRAL. The layers are KDA + dense MLP because that mixer
//! already has its own GPU-vs-reference gate (`kda_fixture_gpu.rs`) — so a failure here is the
//! residual program's, not a mixer's. The stream count is glm5_next's 4 and the collapse is its
//! `Mean`.
//!
//! `swiglu_limit` is set effectively infinite, so the clamp never binds and this gate keeps
//! measuring the residual topology rather than the activation. It is not neutralizing a gap any
//! more: `cfg.clamp_exp_at`/`clamp_shexp_at` now return `SwigluClamp::Pre` for glm5_next and the
//! FFN runs `swiglu_preclamped_mul_scaled`, which the limit here makes a no-op. The activation
//! form has its own gate — `swiglu_preclamp_gpu.rs`.
//!
//! GPU-gated (`#[ignore]`); rig law = exactness only, never timing. Run under
//! `flock /tmp/memra-5090.lock` with `NVIDIA_TF32_OVERRIDE=0` and `-- --ignored`.

use memra_engine::Engine;
use memra_engine::hybrid::HybridModel;
use memra_gguf::GgmlType;
use memra_gguf::config::{HfConfig, ModelConfig};
use memra_gguf::model_plan::ModelPlan;
use memra_gguf::source::{TensorSource, TensorView};
use memra_gguf::tensor_contract::{
    CheckpointDialect, ContractOptions, OutputHead, TensorContract, TensorId, TensorMatch,
};
use memra_reference::{ReferenceTensor, deterministic_fixture};
use std::borrow::Cow;
use std::collections::BTreeMap;

const HIDDEN: usize = 128;
const STREAMS: usize = 4;
const VOCAB: u32 = 32;

/// Scale-relative bound, same shape as `kda_fixture_gpu`'s and `mla_gpu_forward`'s. The reference
/// sums on the host in declaration order; the GPU runs cuBLASLt GEMMs, warp-tree reductions, and
/// a Sinkhorn whose device realization is not byte-identical to `hc_split_sinkhorn` (the dsv4
/// lane calls that arm a "realization fork"), so bit-identity is not the bar and never was.
///
/// CALIBRATED, not guessed (5090, TF32 off, 2026-08-28): the worst of the 10 comparisons these
/// gates make is 8.2e-7 relative, so 2e-5 carries ~24x headroom for reduction order while staying
/// ~35x BELOW the ~7e-4 that TF32-on costs — a rig that lost `NVIDIA_TF32_OVERRIDE=0` fails here
/// instead of passing under a widened bar (the dflash2 parity lesson). The mutation check below
/// lands at 1.7e-2, three orders above this bar. Calibrate downward, never upward.
const TOL: f32 = 2e-5;

/// GPU tests serialize on one device: the model, its cache and the reference stack are all live
/// at once, and cargo runs test fns in parallel by default.
fn gpu_guard() -> std::sync::MutexGuard<'static, ()> {
    static GPU: std::sync::Mutex<()> = std::sync::Mutex::new(());
    GPU.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// cuBLASLt f32 compute rides TF32 on Blackwell by default — right for serving, wrong for a
/// parity gate. The driver reads this at CUDA init, so it must be set before the first
/// `Engine::new` in the process; `call_once` serializes every test thread behind the write.
fn force_true_f32() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        if std::env::var("NVIDIA_TF32_OVERRIDE").as_deref() != Ok("0") {
            // SAFETY: no CUDA call has been made and no Engine handed out in this process yet,
            // and call_once serializes every test thread behind this write.
            unsafe { std::env::set_var("NVIDIA_TF32_OVERRIDE", "0") };
        }
    });
}

/// Two KDA + dense-MLP trunk layers under glm5_next's hyper-connections, expressed the only way
/// the engine will accept: a real `config.json`, parsed by the real `HfConfig`/`ModelConfig`
/// path, compiled by the real glm5_next model pack. `HybridModel::load_from_source` compiles the
/// plan from `src.config()`, so a hand-built `ModelPlan` could not reach it.
///
/// `head_dim` is 128 because that is the only width `memra_kda_scan_s128` is instantiated for.
/// The MLA/DSA and MoE fields are required by the glm5_next config parser and are inert: no
/// layer in `layer_types`/`mlp_layer_types` selects them.
fn mini_config_json() -> String {
    r#"{
      "model_type": "glm5_next_text",
      "num_hidden_layers": 2,
      "num_nextn_predict_layers": 0,
      "hidden_size": 128,
      "intermediate_size": 64,
      "vocab_size": 32,
      "max_position_embeddings": 512,
      "rms_norm_eps": 1e-05,
      "hidden_act": "silu",
      "swiglu_limit": 1e30,
      "tie_word_embeddings": true,
      "hc_mult": 4,
      "hc_eps": 1e-06,
      "hc_sinkhorn_iters": 20,
      "mhc": true,
      "layer_types": ["linear_attention", "linear_attention"],
      "mlp_layer_types": ["dense", "dense"],
      "first_k_dense_replace": 2,
      "indexer_types": ["full", "full"],
      "linear_attn_config": {
        "num_heads": 1,
        "head_dim": 128,
        "short_conv_kernel_size": 4,
        "gate_lower_bound": -5.0,
        "kda_layers": [0, 1],
        "full_attn_layers": []
      },
      "num_attention_heads": 1,
      "num_key_value_heads": 1,
      "q_lora_rank": 16,
      "kv_lora_rank": 16,
      "qk_head_dim": 16,
      "qk_nope_head_dim": 16,
      "qk_rope_head_dim": 0,
      "v_head_dim": 16,
      "mla_use_nope": true,
      "index_n_heads": 1,
      "index_head_dim": 8,
      "index_topk": 8,
      "index_kpool": 4,
      "index_kpool_always_select_tail": true,
      "index_kpool_compress": true,
      "indexer_rope_interleave": true,
      "index_share_for_mtp_iteration": true,
      "n_routed_experts": 4,
      "num_experts_per_tok": 2,
      "moe_intermediate_size": 32,
      "n_shared_experts": 1,
      "scoring_func": "sigmoid",
      "topk_method": "noaux_tc",
      "routed_scaling_factor": 2.5,
      "norm_topk_prob": true,
      "n_group": 1,
      "topk_group": 1,
      "head_dim": 0,
      "attention_bias": false,
      "moe_router_dtype": "float32",
      "dtype": "bfloat16"
    }"#
    .to_string()
}

fn mini_config() -> ModelConfig {
    ModelConfig::from_hf(&HfConfig::parse(&mini_config_json()))
}

fn mini_plan(config: &ModelConfig) -> ModelPlan {
    memra_gguf::model_packs::for_config(config)
        .expect("glm5_next model pack matches the mini config")
        .compile_plan(config)
        .expect("mini glm5_next plan compiles")
}

struct OwnedTensor {
    bytes: Vec<u8>,
    ne: Vec<u64>,
}

/// Serves the reference fixture's own numbers under the contract's ggml names, so the reference
/// and the GPU read ONE set of weights. Unlike the KDA fixture's source this MUST answer
/// `config()`: `HybridModel::load_from_source` compiles the plan from it.
struct FixtureSource {
    config: ModelConfig,
    tensors: BTreeMap<String, OwnedTensor>,
}

impl TensorSource for FixtureSource {
    fn config(&self) -> ModelConfig {
        self.config.clone()
    }
    fn find(&self, name: &str) -> Option<TensorView<'_>> {
        let t = self.tensors.get(name)?;
        Some(TensorView {
            bytes: Cow::Borrowed(&t.bytes),
            ggml_type: GgmlType::F32,
            ne: t.ne.clone(),
        })
    }
}

fn fixture_source(
    config: &ModelConfig,
    plan: &ModelPlan,
    weights: &BTreeMap<TensorId, ReferenceTensor>,
) -> FixtureSource {
    let contract = TensorContract::for_plan(
        plan,
        CheckpointDialect::Gguf,
        ContractOptions {
            output_head: OutputHead::TiedToEmbedding,
        },
    )
    .expect("contract for the mini hyper-connections plan");
    let mut tensors = BTreeMap::new();
    for req in contract
        .requirements
        .iter()
        .filter(|r| r.required || weights.contains_key(&r.id))
    {
        let tensor = weights
            .get(&req.id)
            .unwrap_or_else(|| panic!("reference fixture is missing {:?}", req.id));
        let elements: usize = req.shape.iter().map(|&d| d as usize).product();
        assert_eq!(
            elements,
            tensor.data.len(),
            "fixture {:?} has {} elements, contract requires {elements}",
            req.id,
            tensor.data.len()
        );
        let bytes: Vec<u8> = tensor.data.iter().flat_map(|v| v.to_le_bytes()).collect();
        let names = match req.match_mode {
            TensorMatch::OneOf => &req.names[..1],
            TensorMatch::All => req.names.as_slice(),
        };
        for name in names {
            tensors.insert(
                name.clone(),
                OwnedTensor {
                    bytes: bytes.clone(),
                    ne: req.shape.clone(),
                },
            );
        }
    }
    FixtureSource {
        config: config.clone(),
        tensors,
    }
}

fn maxdiff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "compared slices differ in length");
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn scale_of(v: &[f32]) -> f32 {
    v.iter().fold(0.0f32, |m, x| m.max(x.abs())).max(1e-6)
}

fn relative(got: &[f32], want: &[f32]) -> f32 {
    maxdiff(got, want) / scale_of(want)
}

fn check(name: &str, got: &[f32], want: &[f32]) {
    assert!(
        got.iter().all(|v| v.is_finite()),
        "{name}: GPU output has non-finite values"
    );
    let rel = relative(got, want);
    assert!(
        rel <= TOL,
        "{name}: GPU vs reference relative maxdiff {rel:.3e} (tol {TOL:.1e})"
    );
}

struct Harness {
    engine: Engine,
    model: HybridModel,
    plan: ModelPlan,
    weights: BTreeMap<TensorId, ReferenceTensor>,
}

impl Harness {
    fn new() -> Self {
        force_true_f32();
        let config = mini_config();
        let plan = mini_plan(&config);
        let fixture = deterministic_fixture(&plan).expect("deterministic hc fixture");
        let source = fixture_source(&config, &plan, &fixture.weights);
        let engine = Engine::new(0).expect("CUDA engine on device 0");
        let model = HybridModel::load_from_source_without_mtp(&engine, &source)
            .expect("mini hyper-connections model loads from the contract");
        Self {
            engine,
            model,
            plan,
            weights: fixture.weights,
        }
    }

    fn reference_logits(&self, tokens: &[u32]) -> Vec<f32> {
        memra_reference::execute(&self.plan, &self.weights, tokens)
            .expect("reference execute")
            .logits
    }
}

fn tokens(n: usize, seed: u64) -> Vec<u32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((s >> 33) as u32) % VOCAB
        })
        .collect()
}

/// The plan the whole gate rests on really is the hyper-connections one, with glm5_next's
/// constants. No CUDA needed — if this fails, every GPU assertion below is measuring something
/// other than what it claims.
#[test]
fn the_mini_plan_declares_glm5_next_hyper_connections() {
    use memra_gguf::model_plan::{HcCollapse, ResidualTopology};
    let plan = mini_plan(&mini_config());
    assert_eq!(plan.layers.len(), 2);
    assert_eq!(plan.hidden_size as usize, HIDDEN);
    for layer in &plan.layers {
        assert_eq!(
            layer.residual,
            ResidualTopology::HyperConnections {
                streams: STREAMS as u32,
                epsilon: 1e-6,
                sinkhorn_iterations: 20,
                collapse: HcCollapse::Mean,
            },
            "layer {} residual",
            layer.index
        );
    }
}

/// GATE 1 — stateless prefill. `HybridModel::forward` over the whole prompt against
/// `memra_reference::execute`'s logits, at lengths that cross the KDA scan's chunk size.
#[test]
#[ignore = "needs a CUDA device — run under flock /tmp/memra-5090.lock"]
fn hyper_prefill_logits_match_the_reference() {
    let _gpu = gpu_guard();
    let h = Harness::new();
    for &n in &[1usize, 3, 8, 65] {
        let ids = tokens(n, 0x11C ^ n as u64);
        let want = h.reference_logits(&ids);
        let got = h.model.forward(&h.engine, &ids).expect("GPU hc prefill");
        check(&format!("prefill T={n}"), &got, &want);
    }
}

/// GATE 2 — `forward_last` returns exactly the last row `forward` returns. The two share the hc
/// layer stack and differ only in the head projection; this pins that they cannot drift.
#[test]
#[ignore = "needs a CUDA device — run under flock /tmp/memra-5090.lock"]
fn hyper_forward_last_matches_the_reference_last_row() {
    let _gpu = gpu_guard();
    let h = Harness::new();
    let ids = tokens(11, 0x1A57_5EED);
    let want = h.reference_logits(&ids);
    let vocab = VOCAB as usize;
    let got = h
        .model
        .forward_last(&h.engine, &ids)
        .expect("GPU hc forward_last");
    check(
        "forward_last",
        &got,
        &want[(ids.len() - 1) * vocab..ids.len() * vocab],
    );
}

/// GATE 3 — prime then decode. The stream state is intra-pass, but the MIXER state is not: this
/// is where a prime that left the KDA conv ring or recurrent state wrong shows up, and it is the
/// shape real generation runs.
#[test]
#[ignore = "needs a CUDA device — run under flock /tmp/memra-5090.lock"]
fn hyper_prime_then_decode_matches_a_full_recompute() {
    let _gpu = gpu_guard();
    let h = Harness::new();
    let prompt = 6usize;
    let steps = 4usize;
    let ids = tokens(prompt + steps, 0xDEC0DE);
    let vocab = VOCAB as usize;
    let want = h.reference_logits(&ids);

    // `new_planned`, not `new`: the KDA layers' recurrent state and conv ring are allocated
    // from the ModelPlan's StatePlan, not from the config alone.
    let mut cache = memra_engine::cache::Cache::new_planned(&h.engine, &h.model.cfg, &h.plan, 64)
        .expect("cache for the mini hc model");
    let (primed, _seed, _hiddens) = h
        .model
        .prime_cache(&h.engine, &ids[..prompt], &mut cache, 0)
        .expect("GPU hc prime");
    check(
        "prime last row",
        &primed,
        &want[(prompt - 1) * vocab..prompt * vocab],
    );
    for step in 0..steps {
        let row = prompt + step;
        let got = h
            .model
            .decode_step(&h.engine, ids[row], &mut cache)
            .expect("GPU hc decode step");
        check(
            &format!("decode step {step}"),
            &got,
            &want[row * vocab..(row + 1) * vocab],
        );
    }
}

/// GATE 4 — session continuation: a prompt primed in TWO calls onto one live cache, then
/// decoded, must equal the single-shot recompute. This is the multi-turn serving shape, and it
/// is where a prime that keyed its positions or its mixer state to the CALL instead of the
/// session shows up — gate 3 always primes from `cache.pos == 0`.
#[test]
#[ignore = "needs a CUDA device — run under flock /tmp/memra-5090.lock"]
fn hyper_two_chunk_prime_then_decode_matches_a_full_recompute() {
    let _gpu = gpu_guard();
    let h = Harness::new();
    let vocab = VOCAB as usize;
    let steps = 2usize;
    for &(first, second) in &[(1usize, 5usize), (3, 3), (5, 1)] {
        let prompt = first + second;
        let ids = tokens(prompt + steps, 0xC0_FFEE ^ first as u64);
        let want = h.reference_logits(&ids);
        let mut cache =
            memra_engine::cache::Cache::new_planned(&h.engine, &h.model.cfg, &h.plan, 64)
                .expect("cache for the mini hc model");
        let mut start = 0usize;
        for len in [first, second] {
            let (logits, _seed, _hiddens) = h
                .model
                .prime_cache(&h.engine, &ids[start..start + len], &mut cache, 0)
                .expect("GPU hc chunked prime");
            start += len;
            check(
                &format!("prime chunk ending at {start} (split {first}/{second})"),
                &logits,
                &want[(start - 1) * vocab..start * vocab],
            );
        }
        for step in 0..steps {
            let row = prompt + step;
            let got = h
                .model
                .decode_step(&h.engine, ids[row], &mut cache)
                .expect("GPU hc decode after chunked prime");
            check(
                &format!("decode {step} after split {first}/{second}"),
                &got,
                &want[row * vocab..(row + 1) * vocab],
            );
        }
    }
}

/// MUTATION CHECK — the gate binds to the hc weights.
///
/// Zero one site's `hc_attn_scale` on the GPU side only and re-run gate 1's comparison: it must
/// FAIL. Without this a gate that loaded no hc tensors at all, or dropped the `comb` term, could
/// pass on a lucky topology. Zeroing `scale` leaves every shape valid and every value finite —
/// only the learned gate/combination arithmetic changes.
#[test]
#[ignore = "needs a CUDA device — run under flock /tmp/memra-5090.lock"]
fn zeroing_an_hc_scale_breaks_the_comparison() {
    let _gpu = gpu_guard();
    force_true_f32();
    let config = mini_config();
    let plan = mini_plan(&config);
    let fixture = deterministic_fixture(&plan).expect("deterministic hc fixture");
    let mut source = fixture_source(&config, &plan, &fixture.weights);
    let name = "blk.0.hc_attn_scale".to_string();
    let mutated = source
        .tensors
        .get_mut(&name)
        .unwrap_or_else(|| panic!("{name} must be in the contract-named fixture"));
    mutated.bytes.fill(0);

    let engine = Engine::new(0).expect("CUDA engine on device 0");
    let model = HybridModel::load_from_source_without_mtp(&engine, &source)
        .expect("the mutated fixture still loads");
    let ids = tokens(8, 0x3EED);
    let want = memra_reference::execute(&plan, &fixture.weights, &ids)
        .expect("reference execute")
        .logits;
    let got = model.forward(&engine, &ids).expect("GPU hc prefill");
    let rel = relative(&got, &want);
    assert!(
        rel > TOL,
        "zeroing blk.0.hc_attn_scale left the logits within tolerance (rel {rel:.3e}); the gate \
         is not reading the hc weights it claims to gate"
    );
}

/// The absent-tensor refusal fires, names the tensor, and does NOT fall back to a serial residual.
#[test]
#[ignore = "needs a CUDA device — run under flock /tmp/memra-5090.lock"]
fn a_missing_hc_tensor_is_refused_by_name() {
    let _gpu = gpu_guard();
    force_true_f32();
    let config = mini_config();
    let plan = mini_plan(&config);
    let fixture = deterministic_fixture(&plan).expect("deterministic hc fixture");
    let engine = Engine::new(0).expect("CUDA engine on device 0");
    for name in [
        "blk.0.hc_attn_fn",
        "blk.1.hc_ffn_base",
        "blk.1.hc_ffn_scale",
    ] {
        let mut source = fixture_source(&config, &plan, &fixture.weights);
        assert!(
            source.tensors.remove(name).is_some(),
            "{name} must be in the contract-named fixture"
        );
        let error = HybridModel::load_from_source_without_mtp(&engine, &source)
            .err()
            .unwrap_or_else(|| {
                panic!("loading without {name} must fail, not fall back to a serial residual")
            })
            .to_string();
        assert!(
            error.contains(name),
            "the refusal must name the absent tensor; got: {error}"
        );
        assert!(
            error.contains("HyperConnections"),
            "the refusal must say which plan declaration it is enforcing; got: {error}"
        );
    }
}
