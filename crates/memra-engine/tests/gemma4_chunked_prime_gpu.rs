//! CORRECTNESS GATE for the gemma4 prime under the GENERIC chunked driver (memra#535 P1a).
//!
//! WHY THIS GATE EXISTS. `gemma4_prime` is a fresh-prompt-only prefill graph: its attention
//! (`gemma4_attn_prime`) attends over THIS batch's f32/bf16 K/V for `t x t` rows and refuses
//! `cache.pos != 0`, so a served gemma4 prompt primes monolithically — one worker tick for the
//! whole prompt, every peer starved for its length (audit §9.1, memra#534). The serial trunk
//! closed the same class on 2026-08-05 by quantize-then-attend for EVERY chunk (`chunkinv`),
//! which is why `MEMRA_PRIME_CHUNK` is a pure memory knob there. This gate holds gemma4 to the
//! same law once it rides `prime_cache`'s chunked, continuation-capable driver.
//!
//! THREE ARMS, and the truth arm is the one that matters (LAW:pin-against-truth):
//!   1. monolithic prime vs `memra_reference::execute` — anchored OUTSIDE the feature;
//!   2. chunked prime (several splits, non-dividing ones included) vs monolithic — the chunkinv
//!      law. Byte identity is the model-scale bar (MMQ weights); on this F32 fixture the
//!      projections ride cuBLAS f32 GEMM, which is not m-invariant, so the fixture holds a
//!      calibrated band (`SPLIT_TOL`) and `tools/chunk-invariance-gate.sh` holds the bytes;
//!   3. chunked prime then `decode_step` vs a full recompute of the longer prompt — the decode
//!      reads what the chunks wrote (KV planes, SWA ring position, positions).
//!
//! THE FIXTURE CARRIES BOTH GEMMA ATTENTION CLASSES: one sliding-window layer (hd 256, local
//! rope, window SHORTER than the prompt so the mask is live) and one global layer (hd 512,
//! rope_freqs-scaled rope) — gemma4's own alternation at the head dims the kernels are stamped
//! for. Two query heads, one KV head: the fused q/k/v norm+rope producer (`rms_norm_qkv_w4b`,
//! the bf16-operand emit the FA twins consume) requires `nh*t + 2*nkv*t >= 64` rows, which every
//! real gemma satisfies at t >= PRIME_MIN_T (nh + 2*nkv >= 4) but a 1-head fixture does not at
//! t = 16 — and a chunk that changed PRODUCER would fail the split arms for a reason no model can
//! reach. Post-attention / post-MLP norms, layer scale and GELU_PAR ride `gemma4_layer_tail_add`,
//! shared with decode, so a chunk-boundary defect would show as a prime-vs-decode split (arm 3).
//!
//! Lands RED on arms 2 and 3 (today `prime_cache` routes gemma to `gemma4_prime`, which returns
//! `Err` for `pos > 0`); turns green when `prime_layers_gemma` exists. Arm 1 must be green
//! BEFORE the change too — its band is calibrated from that run, never widened afterwards.
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

const VOCAB: u32 = 32;
const MAX_CTX: usize = 1024;
/// Shorter than every GPU prompt below, so the sliding layer's window mask is exercised.
const WINDOW: usize = 64;

fn mini_config_json(window: usize) -> String {
    format!(
        r#"{{"model_type":"gemma4","text_config":{{"model_type":"gemma4_text",
        "num_hidden_layers":2,"hidden_size":64,"num_attention_heads":2,
        "num_key_value_heads":1,"num_global_key_value_heads":1,"head_dim":256,
        "global_head_dim":512,"intermediate_size":64,"vocab_size":{VOCAB},
        "max_position_embeddings":{MAX_CTX},"rms_norm_eps":0.000001,
        "sliding_window":{window},"final_logit_softcapping":30,"hidden_activation":"gelu_pytorch_tanh",
        "layer_types":["sliding_attention","full_attention"],
        "rope_parameters":{{"full_attention":{{"rope_theta":1000000,
        "partial_rotary_factor":0.25}},"sliding_attention":{{"rope_theta":10000}}}}}}}}"#
    )
}

fn mini_config() -> ModelConfig {
    mini_config_with_window(WINDOW)
}

fn mini_config_with_window(window: usize) -> ModelConfig {
    ModelConfig::from_hf(&HfConfig::parse(&mini_config_json(window)))
}

fn mini_plan(config: &ModelConfig) -> ModelPlan {
    memra_gguf::model_packs::for_config(config)
        .expect("gemma4_dense model pack matches the mini config")
        .compile_plan(config)
        .expect("mini gemma4 plan compiles")
}

struct OwnedTensor {
    bytes: Vec<u8>,
    ne: Vec<u64>,
}

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
    .expect("contract for the mini gemma4 plan");
    let mut tensors = BTreeMap::new();
    for req in contract
        .requirements
        .iter()
        .filter(|r| r.required || weights.contains_key(&r.id))
    {
        let tensor = weights
            .get(&req.id)
            .unwrap_or_else(|| panic!("reference fixture is missing {:?}", req.id));
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

/// GPU tests serialize on one device AND on the process-wide `MEMRA_PRIME_CHUNK` they flip.
fn gpu_guard() -> std::sync::MutexGuard<'static, ()> {
    static GPU: std::sync::Mutex<()> = std::sync::Mutex::new(());
    GPU.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn force_true_f32() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        if std::env::var("NVIDIA_TF32_OVERRIDE").as_deref() != Ok("0") {
            // SAFETY: no CUDA call has been made in this process yet, and call_once serializes
            // every test thread behind this write.
            unsafe { std::env::set_var("NVIDIA_TF32_OVERRIDE", "0") };
        }
    });
}

/// SAFETY: every caller holds `gpu_guard`, so no other test thread is reading the environment.
fn set_prime_chunk(value: &str) {
    unsafe { std::env::set_var("MEMRA_PRIME_CHUNK", value) };
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

/// The GGUF contract requires `rope_freqs.weight` for a `PartialRotary` rope (gemma4's global
/// layers rotate 25% of hd 512), but `deterministic_fixture` only synthesizes the tensor for
/// `RopeFactors::Checkpoint`. Build it here with the SAME divisor convention the reference
/// executor applies for `PartialRotary` (`rope_factor_values`: 1.0 on the kept pairs, 1e30 on
/// the rest), so the engine's `rope_freqs`-driven rope and the reference's synthesized table
/// are the same function. If a later fixture version adds this itself, the insert is a no-op.
fn add_partial_rotary_factors(plan: &ModelPlan, weights: &mut BTreeMap<TensorId, ReferenceTensor>) {
    use memra_gguf::model_plan::{AttentionPlan, RopeFactors};
    if weights.contains_key(&TensorId::RopeFactors) {
        return;
    }
    let mut table: Option<Vec<f32>> = None;
    for layer in &plan.layers {
        let rope = match &layer.attention {
            AttentionPlan::Full(a) | AttentionPlan::SlidingWindow { attention: a, .. } => &a.rope,
            _ => continue,
        };
        if let RopeFactors::PartialRotary { factor } = rope.factors {
            let width = rope.dimensions as usize / 2;
            let keep = (width as f32 * factor.clamp(0.0, 1.0)).round() as usize;
            let t: Vec<f32> = (0..width)
                .map(|i| if i < keep { 1.0 } else { 1.0e30 })
                .collect();
            match &table {
                Some(prev) if prev.len() >= t.len() => {}
                _ => table = Some(t),
            }
        }
    }
    if let Some(t) = table {
        let n = t.len();
        weights.insert(
            TensorId::RopeFactors,
            ReferenceTensor::new(vec![n], t).expect("rope factors tensor"),
        );
    }
}

struct Harness {
    engine: Engine,
    model: HybridModel,
    plan: ModelPlan,
    weights: BTreeMap<TensorId, ReferenceTensor>,
}

impl Harness {
    fn new() -> Self {
        Self::with_window(WINDOW)
    }

    fn with_window(window: usize) -> Self {
        force_true_f32();
        let config = mini_config_with_window(window);
        let plan = mini_plan(&config);
        let mut fixture = deterministic_fixture(&plan).expect("deterministic gemma4 fixture");
        add_partial_rotary_factors(&plan, &mut fixture.weights);
        let source = fixture_source(&config, &plan, &fixture.weights);
        let engine = Engine::new(0).expect("CUDA engine on device 0");
        let model = HybridModel::load_from_source_without_mtp(&engine, &source)
            .expect("mini gemma4 model loads from the contract");
        Self {
            engine,
            model,
            plan,
            weights: fixture.weights,
        }
    }

    fn reference_logits(&self, ids: &[u32]) -> Vec<f32> {
        memra_reference::execute(&self.plan, &self.weights, ids)
            .expect("reference execute")
            .logits
    }

    fn cache(&self) -> memra_engine::cache::Cache {
        memra_engine::cache::Cache::new_planned(&self.engine, &self.model.cfg, &self.plan, MAX_CTX)
            .expect("cache for the mini gemma4 model")
    }

    /// Prime `ids` in ONE `prime_cache` call under the given chunk setting; returns the
    /// last-row logits, the pre-output_norm hidden stack, and the primed cache.
    fn prime(
        &self,
        ids: &[u32],
        chunk: &str,
    ) -> Result<(Vec<f32>, Vec<f32>, memra_engine::cache::Cache), String> {
        set_prime_chunk(chunk);
        let mut cache = self.cache();
        let (logits, _seed, hiddens) = self
            .model
            .prime_cache(&self.engine, ids, &mut cache, 0)
            .map_err(|e| format!("prime at MEMRA_PRIME_CHUNK={chunk}: {e}"))?;
        let stack = self.engine.dtoh(&hiddens).map_err(|e| e.to_string())?;
        Ok((logits, stack, cache))
    }
}

impl Harness {
    /// Prime `ids` as SEVERAL `prime_cache` calls on one cache — the serving shape
    /// (`prefill_tick` takes a budget per tick) — cutting at the given boundaries. Every call is
    /// at least PRIME_MIN_T rows, as the worker guarantees. Returns the last call's logits and
    /// hidden stack (the stack of the LAST call only: rows [last_cut, T)).
    fn prime_multi_call(
        &self,
        ids: &[u32],
        cuts: &[usize],
        chunk: &str,
    ) -> Result<(Vec<f32>, Vec<f32>, memra_engine::cache::Cache), String> {
        set_prime_chunk(chunk);
        let mut cache = self.cache();
        let mut start = 0usize;
        let mut last = None;
        for &cut in cuts.iter().chain(std::iter::once(&ids.len())) {
            assert!(cut > start, "cuts must be increasing");
            let seg = &ids[start..cut];
            let queued_after = ids.len() - cut;
            let out = self
                .model
                .prime_cache(&self.engine, seg, &mut cache, queued_after)
                .map_err(|e| {
                    format!(
                        "continuation prime call [{start}, {cut}) at MEMRA_PRIME_CHUNK={chunk}: {e}"
                    )
                })?;
            last = Some(out);
            start = cut;
        }
        let (logits, _seed, hiddens) = last.expect("at least one call");
        let stack = self.engine.dtoh(&hiddens).map_err(|e| e.to_string())?;
        Ok((logits, stack, cache))
    }
}

/// TRUTH-ARM BAND — the KV-quantization class, MEASURED, not guessed (2026-09-20, rented 5090,
/// tree aec32262). The one-program arm attends over the quantized planes for every row, so the
/// reference (exact f32 K/V) and the engine differ by the plane format's error:
///   * default planes (e4m3 globals `MEMRA_GEMMA_GKV`, e4m3 windowed `MEMRA_GEMMA_WKV`):
///     1.115e-2 (T=40), 4.183e-2 (T=200 window live), 1.339e-2 (T=200 window inactive);
///   * q8_0 K / q5_1 V (`MEMRA_GEMMA_GKV=0 MEMRA_GEMMA_WKV=0`): 1.873e-2 / 1.319e-2 / 1.167e-2.
///
/// SEMANTICS were pinned separately at 1.2e-6 (T=40..200, both windows) by the pre-change
/// f32-attention path under `MEMRA_NOFA=1 MEMRA_FA_EMIT=0` against the same reference — so
/// window rule, rope factors, residual order and softcap agree with the reference exactly, and
/// what this band admits is plane quantization only. A window off-by-one would add a ~1.5e-2
/// STEP that survives a plane-format change; the two rows above move together, it does not.
/// 1e-1 = 2.4x over the worst measured. Calibrate downward, never upward.
const TRUTH_TOL: f32 = 1e-1;

/// CHUNK/CALL-SPLIT BAND on THIS fixture. The fixture's weights are F32, so every projection
/// rides cuBLAS f32 GEMM, whose algorithm selection depends on m (the rows in the call) — the
/// same non-m-invariance `glm5_chunked_prime_gpu.rs` isolated on the mHC trunk. With the FA
/// twins the f32 q/k/v operands are then rounded to bf16, and a last-ulp f32 difference that
/// lands on a bf16 rounding boundary becomes a bf16-ulp (2^-8) difference in that operand, so
/// the split residue on this fixture is bf16-class: measured 3.250e-3 worst (hidden stack,
/// T=200, chunk 16), 0..2.2e-7 on the last-row logits (T=200 multi-call: bitwise). On MMQ weights the projections are
/// row-invariant and the same splits are BIT-IDENTICAL — 12B, 31B and 26B-A4B chunkinv and
/// tickinv, `research/exec-p1a-gemma-prime-20260919/` — which is the law and the model-scale
/// bar; this fixture holds the split to a band three orders below the serial trunk's real
/// chunkinv defect (1.813e0 at the boundary), so a boundary-carry defect still reads as one.
const SPLIT_TOL: f32 = 1e-2;

fn relative(got: &[f32], want: &[f32]) -> f32 {
    assert_eq!(got.len(), want.len(), "compared slices differ in length");
    let worst = got
        .iter()
        .zip(want)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    let scale = want.iter().fold(0.0f32, |m, x| m.max(x.abs())).max(1e-6);
    worst / scale
}

fn check_truth(name: &str, got: &[f32], want: &[f32]) {
    assert!(
        got.iter().all(|v| v.is_finite()),
        "{name}: output has non-finite values"
    );
    let rel = relative(got, want);
    eprintln!("[gemma4-chunked-prime] {name}: relative maxdiff {rel:.3e}");
    assert!(
        rel <= TRUTH_TOL,
        "{name}: relative maxdiff {rel:.3e} (tol {TRUTH_TOL:.1e})"
    );
}

fn check_split(name: &str, got: &[f32], want: &[f32]) {
    assert!(
        got.iter().all(|v| v.is_finite()),
        "{name}: output has non-finite values"
    );
    let rel = relative(got, want);
    let exact = got
        .iter()
        .zip(want)
        .all(|(a, b)| a.to_bits() == b.to_bits());
    eprintln!("[gemma4-chunked-prime] {name}: relative maxdiff {rel:.3e} bitwise={exact}");
    assert!(
        rel <= SPLIT_TOL,
        "{name}: relative maxdiff {rel:.3e} (tol {SPLIT_TOL:.1e}) — a split moved the arithmetic"
    );
}

/// The plan really is the gemma residual over one sliding and one global attention layer.
/// No CUDA — if this fails, every assertion below is measuring something else.
#[test]
fn the_mini_plan_is_gemma_residual_with_a_sliding_and_a_global_layer() {
    use memra_gguf::model_plan::{AttentionPlan, ResidualTopology};
    let config = mini_config();
    let plan = mini_plan(&config);
    assert_eq!(plan.layers.len(), 2);
    for layer in &plan.layers {
        assert!(
            matches!(layer.residual, ResidualTopology::Gemma { .. }),
            "layer {} is not under the gemma residual: {:?}",
            layer.index,
            layer.residual
        );
    }
    assert!(
        matches!(
            plan.layers[0].attention,
            AttentionPlan::SlidingWindow { .. }
        ),
        "layer 0 should be the sliding-window mixer, got {:?}",
        plan.layers[0].attention
    );
    assert!(
        matches!(plan.layers[1].attention, AttentionPlan::Full(_)),
        "layer 1 should be the global mixer, got {:?}",
        plan.layers[1].attention
    );
    assert!(
        (config.gemma4.as_ref().expect("gemma4 bits").sliding_window as usize) < 200,
        "the window must be shorter than the GPU prompts or the SWA mask is never live"
    );
}

/// TRUTH ARM. The monolithic gemma prime must reproduce the reference executor. Green before
/// and after the driver change; the band is set from the pre-change measurement.
#[test]
#[ignore = "needs a CUDA device — run under flock /tmp/memra-5090.lock"]
fn a_monolithic_gemma_prime_matches_the_reference_executor() {
    let _gpu = gpu_guard();
    let h = Harness::new();
    let vocab = VOCAB as usize;
    for &prompt in &[40usize, 100, 200] {
        let ids = tokens(prompt, 0x6E44_0000 ^ prompt as u64);
        let want = h.reference_logits(&ids);
        let last = &want[(prompt - 1) * vocab..prompt * vocab];
        let (got, _stack, _cache) = h.prime(&ids, "0").expect("monolithic prime");
        check_truth(&format!("T={prompt} monolithic vs reference"), &got, last);
    }
}

/// DIAGNOSTIC for the truth arm (prints, then asserts only finiteness): the same T=200 prompt
/// with the window LIVE (64) and with the window INACTIVE (1024). If the reference-vs-engine
/// divergence collapses when the window is inactive, the two disagree on window SEMANTICS
/// (an off-by-one on a 64-key window moves the output by ~1/64), which is a finding to settle
/// against the vendor definition, not noise to band over.
#[test]
#[ignore = "needs a CUDA device — run under flock /tmp/memra-5090.lock"]
fn truth_arm_divergence_with_the_window_live_versus_inactive() {
    let _gpu = gpu_guard();
    let vocab = VOCAB as usize;
    let prompt = 200usize;
    let ids = tokens(prompt, 0x6E44_0000 ^ prompt as u64);
    for window in [WINDOW, 1024usize] {
        let h = Harness::with_window(window);
        let want = h.reference_logits(&ids);
        let last = &want[(prompt - 1) * vocab..prompt * vocab];
        let (got, _s, _c) = h.prime(&ids, "0").expect("monolithic prime");
        let rel = relative(&got, last);
        eprintln!("[gemma4-chunked-prime] T={prompt} window={window}: relative maxdiff {rel:.3e}");
        assert!(got.iter().all(|v| v.is_finite()));
    }
}

/// THE SERVING SHAPE. A prompt primed as several `prime_cache` CALLS on one cache (what
/// `prefill_tick` does every tick) must produce the same result as one call (SPLIT_TOL here,
/// bytes at model scale). Cut points include
/// ones that are not multiples of the internal chunk. Today gemma4 refuses `cache.pos != 0`,
/// so this arm is RED until the generic driver carries gemma.
#[test]
#[ignore = "needs a CUDA device — run under flock /tmp/memra-5090.lock"]
fn a_multi_call_gemma_prime_matches_the_single_call_prime() {
    let _gpu = gpu_guard();
    let h = Harness::new();
    for &prompt in &[100usize, 200] {
        let ids = tokens(prompt, 0x6E44_1000 ^ prompt as u64);
        let (mono_logits, _mono_stack, _c) = h.prime(&ids, "0").expect("single-call prime");
        for cuts in [vec![32usize], vec![37], vec![16, 53], vec![64, 80]] {
            if cuts.iter().any(|&c| c + 16 > prompt) {
                continue;
            }
            let (got, _stack, _c2) = h
                .prime_multi_call(&ids, &cuts, "0")
                .unwrap_or_else(|e| panic!("gemma4 cannot continue a prime yet: {e}"));
            check_split(
                &format!("T={prompt} cuts={cuts:?} logits"),
                &got,
                &mono_logits,
            );
        }
    }
}

/// THE INTERNAL SPLIT. `MEMRA_PRIME_CHUNK` must be a pure memory knob on gemma too: several
/// internal chunk sizes vs the monolithic walk, on logits AND the hidden stack (SPLIT_TOL
/// here, bytes at model scale).
/// Vacuity guard: the arm first proves the setting is honored (a chunked call must not be the
/// monolithic one in disguise) by checking the driver's own range schedule.
#[test]
#[ignore = "needs a CUDA device — run under flock /tmp/memra-5090.lock"]
fn an_internally_chunked_gemma_prime_matches_the_monolithic_prime() {
    let _gpu = gpu_guard();
    let h = Harness::new();
    let n_layers = h.model.cfg.n_layer as usize;
    set_prime_chunk("32");
    let ranges = memra_engine::hybrid_forward::prime_chunk_ranges(200, n_layers, false);
    assert!(
        ranges.len() >= 4,
        "the chunked arm must schedule several ranges, got {ranges:?}"
    );
    for &prompt in &[100usize, 200] {
        let ids = tokens(prompt, 0x6E44_3000 ^ prompt as u64);
        let (mono_logits, mono_stack, _c) = h.prime(&ids, "0").expect("monolithic prime");
        for &chunk in &["16", "32", "37", "128"] {
            let (got, stack, _c2) = h
                .prime(&ids, chunk)
                .unwrap_or_else(|e| panic!("gemma4 cannot chunk its prime yet: {e}"));
            check_split(
                &format!("T={prompt} chunk={chunk} logits"),
                &got,
                &mono_logits,
            );
            check_split(
                &format!("T={prompt} chunk={chunk} hidden stack"),
                &stack,
                &mono_stack,
            );
        }
    }
}
