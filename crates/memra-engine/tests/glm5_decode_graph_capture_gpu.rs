//! Gate for `MEMRA_HC_DECODE_WS` — the persistent hc-glue decode workspace
//! (lane/glm5-decode-diet lever 2, 2026-08-31).
//!
//! THE ONE CHANGE UNDER TEST: the T=1 hc decode walk lands its glue transients (mixes,
//! Sinkhorn gates, comb, collapse y, both norm scratches, the per-site post output) in one
//! per-engine `HyperDecodeWs` instead of fresh allocations every layer. The kernels, their
//! order and their operand bytes are unchanged, so the claim is BYTE identity of the decode
//! logits ON vs OFF — plus a counted receipt that the allocator-call class the launch-diet
//! census measured (2,358 `cuMemAllocAsync+Free`/token) actually shrinks (`SCRATCH_ALLOC_CALLS`
//! delta per 24 steps, printed both arms — the launch-econ instrument's host twin).
//!
//! COMPOSITION: lever 1 (`MEMRA_HC_FUSED_PRE`) shares `pre_finish_into` with this walk; the
//! compose arm runs both doors ON against the both-OFF baseline, byte-compared, with both
//! engagement counters advancing (wiring-assertions lesson: engagement is asserted, never
//! inferred from a green diff).
//!
//! Rig law: correctness-only, run under `flock /tmp/memra-5090.lock`, TF32 forced off,
//! `-- --ignored --test-threads=1`.

use memra_engine::{Engine, SCRATCH_ALLOC_CALLS};
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
use std::sync::atomic::Ordering;

const VOCAB: u32 = 32;

fn gpu_guard() -> std::sync::MutexGuard<'static, ()> {
    static GPU: std::sync::Mutex<()> = std::sync::Mutex::new(());
    GPU.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

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

fn mini_config_json() -> String {
    r#"{
      "model_type": "glm5_next_text",
      "num_hidden_layers": 8,
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
      "layer_types": ["linear_attention", "linear_attention", "linear_attention", "deepseek_sparse_attention", "linear_attention", "linear_attention", "linear_attention", "deepseek_sparse_attention"],
      "mlp_layer_types": ["dense", "sparse", "sparse", "sparse", "sparse", "sparse", "sparse", "sparse"],
      "first_k_dense_replace": 1,
      "indexer_types": ["full", "full", "full", "full", "full", "full", "full", "full"],
      "linear_attn_config": {
        "num_heads": 1,
        "head_dim": 128,
        "short_conv_kernel_size": 4,
        "gate_lower_bound": -5.0,
        "kda_layers": [0, 1, 2, 4, 5, 6],
        "full_attn_layers": [3, 7]
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
      "n_routed_experts": 64,
      "num_experts_per_tok": 8,
      "moe_intermediate_size": 64,
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
    ggml_type: GgmlType,
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
            ggml_type: t.ggml_type,
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
        let names = match req.match_mode {
            TensorMatch::OneOf => &req.names[..1],
            TensorMatch::All => req.names.as_slice(),
        };
        // ROUTED-EXPERT PLANES GO OUT AS NVFP4, and that is the whole reason this fixture exists
        // as a separate file. The decode-graph door's T=1 MoE arm requires `moe_q8`, i.e.
        // `q8_expert_supported` on all three expert planes, and the shared hc fixture emits f32 —
        // so the door refuses it by name and a capture gate over it would be vacuous. Every other
        // conjunct the door needs this fixture already satisfies (glm5_next config with a sigmoid
        // router, `swiglu_limit` live past `pre_if_live`'s 1e-6 bar, `n_used = 8 <= 8`, uniform
        // layout by construction, and a resident `dev_exps` slab since a few-KB bank fits any
        // residency budget). One dtype is the difference between "unreachable" and "reachable".
        let is_expert = names.iter().any(|n| {
            n.contains("ffn_gate_exps") || n.contains("ffn_up_exps") || n.contains("ffn_down_exps")
        });
        let (bytes, ggml_type) = if is_expert {
            // `f32_to_nvfp4` blocks by 64, so the row length must divide by 64 — which is why the
            // config carries `hidden_size` 128 and `moe_intermediate_size` 64. `req.shape` is
            // ggml order, so ne[0] is the row length (the reduction axis).
            let row = req.shape[0] as usize;
            assert!(
                row.is_multiple_of(64),
                "expert row {row} is not a multiple of 64; NVFP4 blocks by 64"
            );
            let mut out = Vec::new();
            for chunk in tensor.data.chunks(row) {
                out.extend_from_slice(&memra_gguf::nvfp4_repack::f32_to_nvfp4(chunk));
            }
            (out, GgmlType::NVFP4)
        } else {
            (
                tensor.data.iter().flat_map(|v| v.to_le_bytes()).collect(),
                GgmlType::F32,
            )
        };
        for name in names {
            tensors.insert(
                name.clone(),
                OwnedTensor {
                    bytes: bytes.clone(),
                    ne: req.shape.clone(),
                    ggml_type,
                },
            );
        }
    }
    FixtureSource {
        config: config.clone(),
        tensors,
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

struct Harness {
    engine: Engine,
    model: memra_engine::hybrid::HybridModel,
    plan: ModelPlan,
}

impl Harness {
    fn new() -> Self {
        force_true_f32();
        let config = mini_config();
        let plan = mini_plan(&config);
        let fixture = deterministic_fixture(&plan).expect("deterministic hc fixture");
        let source = fixture_source(&config, &plan, &fixture.weights);
        let engine = Engine::new(0).expect("CUDA engine on device 0");
        let model =
            memra_engine::hybrid::HybridModel::load_from_source_without_mtp(&engine, &source)
                .expect("mini hyper-connections model loads");
        Self {
            engine,
            model,
            plan,
        }
    }

    /// Prime + `steps` decode steps; returns per-step logits bits and the SCRATCH_ALLOC_CALLS
    /// delta across ONLY the decode loop (the workspace's claim is a decode claim).
    /// Replace one captured layer's `ssm_state` with a FRESH allocation holding the same bytes.
    /// The graphs baked the old address, so this is the re-seat the pool's state signature exists
    /// to catch: `pos` continuity cannot see it, and a stage that replayed through it would be
    /// reading freed memory.
    fn reseat_first_recurrent_layer(
        &self,
        cache: &mut memra_engine::cache::Cache,
    ) -> Option<usize> {
        let (il, rl) = cache
            .recur
            .iter_mut()
            .enumerate()
            .find_map(|(i, r)| r.as_mut().map(|r| (i, r)))?;
        use cudarc::driver::DevicePtr;
        let n = rl.ssm_state.len();
        let mut fresh = self.engine.uninit(n).expect("fresh ssm_state");
        self.engine
            .copy_into(&mut fresh, 0, &rl.ssm_state, n)
            .expect("copy the state into the fresh buffer");
        {
            let st = self.engine.stream();
            let (old_p, _g0) = rl.ssm_state.device_ptr(&st);
            let (alt_p, _g1) = rl.ssm_state_alt.device_ptr(&st);
            let (new_p, _g2) = fresh.device_ptr(&st);
            eprintln!("[gate] reseat il={il} ssm 0x{old_p:x} -> 0x{new_p:x} (alt 0x{alt_p:x})");
        }
        rl.ssm_state = fresh;
        Some(il)
    }

    fn decode_bits(&self, ids: &[u32], prompt: usize, steps: usize) -> (Vec<Vec<u32>>, u64) {
        self.decode_bits_reseat(ids, prompt, steps, None)
    }

    /// `reseat_at`: force the pool's invalidation path at that step by re-seating a captured
    /// layer's recurrent state. With `MEMRA_GLM5_GRAPH_RECAPTURE=1` the stage rebuilds; without
    /// it the stage latches to the eager walk. Both must stay byte-identical to the eager arm,
    /// which is the point of running it here rather than only on the box.
    fn decode_bits_reseat(
        &self,
        ids: &[u32],
        prompt: usize,
        steps: usize,
        reseat_at: Option<usize>,
    ) -> (Vec<Vec<u32>>, u64) {
        let mut cache =
            memra_engine::cache::Cache::new_planned(&self.engine, &self.model.cfg, &self.plan, 64)
                .expect("cache for the mini hc model");
        let (_primed, _seed, _hiddens) = self
            .model
            .prime_cache(&self.engine, &ids[..prompt], &mut cache, 0)
            .expect("GPU hc prime");
        let alloc0 = SCRATCH_ALLOC_CALLS.load(Ordering::Relaxed);
        let mut out = Vec::with_capacity(steps);
        for step in 0..steps {
            if reseat_at == Some(step) {
                let il = self.reseat_first_recurrent_layer(&mut cache);
                eprintln!("[gate] forced re-seat at step {step} (layer {il:?})");
            }
            let logits = self
                .model
                .decode_step(&self.engine, ids[prompt + step], &mut cache)
                .expect("GPU hc decode step");
            assert!(
                logits.iter().all(|v| v.is_finite()),
                "step {step}: non-finite logits"
            );
            out.push(logits.iter().map(|v| v.to_bits()).collect());
        }
        let allocs = SCRATCH_ALLOC_CALLS.load(Ordering::Relaxed) - alloc0;
        (out, allocs)
    }
}

/// GATE 1 — the standing decode-identity gate for the workspace door: 24 decode steps, flag
/// OFF then ON, per-step logits compared `to_bits`; the ON arm must engage (counter) and its
/// decode-loop allocator-call count must be STRICTLY LOWER than the OFF arm's (the census
/// class this lever exists to shrink). Both counts print — the lane receipt.
/// THE CAPTURE-HALF GATE. Everything else this lane built tests the door's OTHER enabler — the
/// T=1 device-table MoE arm — and three rig gates now clear it, including at serving scale with
/// the box's own routing. What has never been exercised on the rig is the capture and replay
/// itself, because the shared hc fixture's f32 expert planes make the door refuse by name.
///
/// This fixture removes that one obstacle (NVFP4 expert planes, see `fixture_source`) and asks
/// the only question left: does a walk that CAPTURES and REPLAYS produce the same tokens as the
/// eager walk on the same weights? Box runs 4 through 11 say it does not on the real artifact —
/// token 0 from step 1, with the corruption appearing at the first routed layer. If that
/// reproduces here it is debuggable locally and no further box slot is needed.
///
/// NON-VACUITY IS THE WHOLE RISK with this gate, and it is asserted rather than hoped: the door
/// prints its refusal reason by name, and this test FAILS if the door never captured or never
/// replayed. A green run that captured nothing would be the third time this lane shipped an
/// instrument that observed nothing, and the assert is there so it cannot be the fourth.
#[test]
#[ignore = "needs a CUDA device — run under flock /tmp/memra-5090.lock"]
fn graph_door_decode_matches_eager_bitwise() {
    let _gpu = gpu_guard();
    let h = Harness::new();
    let ids = tokens(40, 0x5EED);
    let (prompt, steps) = (8usize, 16usize);

    // SAFETY: single-threaded test; the doors are read per call by the engine.
    unsafe {
        std::env::set_var("MEMRA_HTOD_DIET", "1");
<<<<<<< HEAD
        // Default ON since 2026-09-04: the eager arm SETS `0` (unsetting would arm the door
        // here too and the identity below would compare the graph against itself).
||||||| parent of 7d897bc50 (gates: lint and pin door-OFF arms to prevent vacuous passes on default flips (#136))
        std::env::remove_var("MEMRA_GLM5_DECODE_GRAPH");
=======
>>>>>>> 7d897bc50 (gates: lint and pin door-OFF arms to prevent vacuous passes on default flips (#136))
        std::env::set_var("MEMRA_GLM5_DECODE_GRAPH", "0");
    }
    let cap_before_eager = memra_engine::GLM5_DECODE_GRAPH_CAPTURES.load(Ordering::Relaxed);
    let (eager, _) = h.decode_bits(&ids, prompt, steps);
    // The `=0` seam is the OFF arm's whole claim: had it captured anything, the identity below
    // would be the graph against itself. Counted, not assumed (the door is default ON).
    assert_eq!(
        memra_engine::GLM5_DECODE_GRAPH_CAPTURES.load(Ordering::Relaxed),
        cap_before_eager,
        "MEMRA_GLM5_DECODE_GRAPH=0 did not disarm the door: the eager arm captured a graph"
    );

    let cap0 = memra_engine::GLM5_DECODE_GRAPH_CAPTURES.load(Ordering::Relaxed);
    let rep0 = memra_engine::GLM5_DECODE_GRAPH_REPLAYS.load(Ordering::Relaxed);
    unsafe {
        std::env::set_var("MEMRA_GLM5_DECODE_GRAPH", "1");
    }
    let (graphed, _) = h.decode_bits(&ids, prompt, steps);
    unsafe {
        std::env::set_var("MEMRA_GLM5_DECODE_GRAPH", "0");
    }
    let captures = memra_engine::GLM5_DECODE_GRAPH_CAPTURES.load(Ordering::Relaxed) - cap0;
    let replays = memra_engine::GLM5_DECODE_GRAPH_REPLAYS.load(Ordering::Relaxed) - rep0;
    let layers = memra_engine::GLM5_DECODE_GRAPH_LAYERS.load(Ordering::Relaxed);
    println!("door: captures={captures} replays={replays} captured_layers={layers}");
    assert!(
        captures > 0 && replays > 0,
        "VACUOUS: the door never captured ({captures}) or never replayed ({replays}); its \
         refusal reason is on stderr as `[glm5-decode-graph] eager: ...`. This gate says nothing \
         about capture unless capture happened"
    );

    let mut bad = 0usize;
    for (step, (a, b)) in eager.iter().zip(&graphed).enumerate() {
        let d = a.iter().zip(b).filter(|(x, y)| x != y).count();
        if d > 0 {
            if bad < 4 {
                let i = a.iter().zip(b).position(|(x, y)| x != y).unwrap();
                println!(
                    "  step {step}: {d}/{} logits differ, first at {i}: eager {} graph {}",
                    a.len(),
                    f32::from_bits(a[i]),
                    f32::from_bits(b[i])
                );
            }
            bad += 1;
        }
    }
    assert_eq!(
        bad, 0,
        "{bad}/{steps} decode steps diverge between the eager walk and the captured/replayed one"
    );
    println!("graph door: {steps} steps bit-identical to eager ({replays} replays)");

    // ARM 2 — THE INVALIDATION PATH, exercised rather than hoped for. A captured layer's
    // `ssm_state` is replaced mid-run with a fresh allocation holding the same bytes: the graphs
    // baked the old address, `pos` continuity cannot see the move, and a stage that replayed
    // through it would read freed memory. `MEMRA_GLM5_GRAPH_RECAPTURE=1` makes the stage REBUILD
    // (drain, drop the execs, capture again); with the knob off it latches to the eager walk.
    //
    // Both outcomes must stay byte-identical to the eager tape, and that is the assert. Box run 3
    // died in exactly this teardown with `CUDA_ERROR_INVALID_VALUE` (it destroyed execs with a
    // replay still outstanding), and box take 13 could not retest it because the engine no longer
    // takes the path: the gate reported `VACUOUS RE-CAPTURE ARM` on a run whose tokens were all
    // correct. This arm is where that receipt comes from now, on the rig, with no box slot.
    let recaptures_before = memra_engine::GLM5_DECODE_GRAPH_RECAPTURES.load(Ordering::Relaxed);
    // SAFETY: single-threaded test; the doors are read per call by the engine. The door itself
    // has to go back ON here: the graphed arm above sets it to `0` when it finishes, and running
    // this arm without it is exactly the vacuity the assert below exists to catch (it caught it).
    unsafe {
        std::env::set_var("MEMRA_GLM5_DECODE_GRAPH", "1");
        std::env::set_var("MEMRA_GLM5_GRAPH_RECAPTURE", "1");
    }
    let (reseated, _) = h.decode_bits_reseat(&ids, prompt, steps, Some(steps / 2));
    let recaptures =
        memra_engine::GLM5_DECODE_GRAPH_RECAPTURES.load(Ordering::Relaxed) - recaptures_before;
    // SAFETY: same as above.
    unsafe {
        std::env::set_var("MEMRA_GLM5_GRAPH_RECAPTURE", "0");
        std::env::set_var("MEMRA_GLM5_DECODE_GRAPH", "0");
    }
    assert!(
        recaptures > 0,
        "VACUOUS RE-CAPTURE ARM: the forced re-seat at step {} did not rebuild any stage \
         (recaptures={recaptures}). Either the re-seated layer is not inside a captured run, or \
         the pointer signature no longer sees a re-seat.",
        steps / 2
    );
    let mut bad_rs = 0usize;
    for (step, (a, b)) in eager.iter().zip(&reseated).enumerate() {
        if a != b {
            if bad_rs < 4 {
                let i = a.iter().zip(b).position(|(x, y)| x != y).unwrap_or(0);
                println!(
                    "  RE-SEAT step {step}: first differing logit {i}: eager {} graph {}",
                    f32::from_bits(a[i]),
                    f32::from_bits(b[i])
                );
            }
            bad_rs += 1;
        }
    }
    assert_eq!(
        bad_rs, 0,
        "{bad_rs}/{steps} steps diverge across a forced re-capture ({recaptures} rebuilds)"
    );
    println!("re-capture arm: {steps} steps bit-identical across {recaptures} forced rebuilds");
}
