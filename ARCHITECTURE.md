# How memra works

This page explains the engine in one read: the crates, the path a request takes, the path a model
takes from checkpoint to GPU kernels, and the gates that keep every fast path exact. It is for
anyone about to read or change the code. Each section names the source file that owns the
behavior; the linked docs hold the detail.

## 1. What memra is

- One Rust process. `memra-server` runs an axum HTTP server on tokio and one GPU worker thread
  that owns the CUDA context and the loaded models (a CUDA context is thread-affine). A
  DeepSeek-V4-Flash checkpoint is the exception: it runs on its own serve threads and engine
  (`dsv4_serve.rs`).
- Hand-written CUDA kernels are compiled ahead of time by nvcc in `crates/memra-engine/build.rs`
  and embedded in the binary as fatbins. cudarc 0.19 loads the driver, cuBLAS, cuBLASLt and NVRTC
  at run time. No Python runs between the HTTP request and the GPU.
- Every model the GPU worker thread loads compiles to a typed `ModelPlan` through its family's
  pack, and the checkpoint binds to the plan's tensor contract before any byte uploads. The
  DeepSeek-V4-Flash route loads without a plan.
- The fast CUDA programs are hand-written. Each declares which plan operations it implements,
  and the device gates hold it exact (section 11). `memra-reference`, an unfused CPU executor of
  the plan, is the semantic baseline for support states; a fast program's parity with it is a
  per-artifact receipt (`RewriteParity`), not a property of every path.
- Support is stated per pack and per record, with gate receipts in the tree (section 10).
  Memra is in active development ([project status](README.md#project-status)).

## 2. Build targets

A build targets one CUDA architecture. `MEMRA_CUDA_ARCH` picks it; when unset, `build.rs` reads
`nvidia-smi --query-gpu=compute_cap` and falls back to `120a` when no GPU or capability matches.

| `MEMRA_CUDA_ARCH` | Compute capability | Devices | Policy family | Prebuilt release |
|---|---|---|---|---|
| `120a` (default) | 12.0 | RTX 50-series, RTX PRO 6000 Blackwell | `120a` | yes |
| `121a` | 12.1 | GB10 (DGX Spark) | `120a` | no, build from source |
| `100a` | 10.0 | B200 | `100a` | no, build from source |
| `90a` | 9.0 | H100 | `90a` | no, build from source |
| `89` | 8.9 | Ada (portable evaluation) | `89` | no, build from source |

- Every build has two names (`src/cuda_arch.rs`). The target (`MEMRA_BUILT_CUDA_TARGET`) is the
  exact `-gencode` value. The family (`MEMRA_BUILT_CUDA_ARCH`) is the policy key: every kernel
  choice and arch-keyed default reads the family, never the target string. `121a` compiles the
  `120a` sources; only its SASS differs.
- The fatbins carry SASS and no PTX, so a binary loads only on its own compute capability. A
  `120a` binary refuses a 12.1 device at startup, and the reverse.
- Arch-specific code compiles behind cfgs set by `build.rs`: `memra_hopper_mma` (wgmma paths on
  `90a`), `memra_sm100_tcgen05` (tcgen05/TMEM paths on `100a`), `memra_portable_cuda` (`90a` and
  `89`) and `memra_cutlass` (opt-in CUTLASS archives, `120a` only).

Hardware facts that shaped the kernels:

- The `120a` family has warp-level `mma.sync` with the FP8/FP6/FP4 kind (`kind::f8f6f4`) and the
  block-scaled kinds (`kind::mxf8f6f4.block_scale`, `kind::mxf4nvf4.block_scale`), and no wgmma
  or tcgen05. On sm_120a the shared-memory opt-in per block is about 99 KB, so tiles are sized
  to fit it.
- The block-scaled kinds need the arch-specific target. `build.rs` compiles every unit with
  `-gencode arch=compute_<target>,code=sm_<target>`.
- GB10 shares system memory with the CPU. The engine detects integrated devices from
  `CU_DEVICE_ATTRIBUTE_INTEGRATED`, never from the device name (`src/unified_memory.rs`). On such
  a device the sizing reads (prefix-cache budget, admission, MoE residency) take
  `/proc/meminfo` `MemAvailable`, because `cuMemGetInfo` leaves out reclaimable page cache.

Build notes per target: [docs/INSTALLATION.md](docs/INSTALLATION.md); per-card state:
[docs/rigs/](docs/rigs/).

## 3. Crate map

Fourteen workspace members. Thirteen share one version, the release tag, and publish to
crates.io; `memra-probe` (version 0.0.0) is unpublished.

| Crate | Role | Workspace deps |
|---|---|---|
| `memra-server` | HTTP surfaces, auth, admission, the GPU worker thread and its scheduler, metering seams; bins `memra-server` and `kv-handoff-gate` | engine, gguf, lanes, net-guard, tokenizer |
| `memra-engine` | CUDA kernels (`cu/`), FFI, forward passes, batched decode, CUDA graphs, speculative decoding, MoE and expert spill, PP/TP/EP runtimes; gate and bench bins (`kernel-check`, `run-gen`, `run-spec`, ...) | gguf, kv, reference, runtime, sampling, tier, tokenizer, validate |
| `memra-gguf` | GGUF and HF safetensors readers, quant decode, tensor census, `hf:` resolution, and the model compiler: `ModelPlan`, tensor contract, model packs, op registry, execution manifests | none |
| `memra-reference` | Portable unfused `ModelPlan` executor with no CUDA and no external fallback; the correctness oracle. Unsupported operations return a named error | gguf, tokenizer |
| `memra-cli` | The `memra model inspect / scaffold / verify / qualify` onboarding commands | gguf, reference, tokenizer |
| `memra-tokenizer` | GPT-2 BPE and Gemma SPM tokenizers built from GGUF metadata or an HF `tokenizer.json`, plus chat-template rendering. Unknown pre-tokenizers refuse at load | gguf |
| `memra-sampling` | Host sampler chain with llama.cpp CPU semantics; greedy is argmax | none |
| `memra-kv` | KV and recurrent-state cache structure, sizing, the KV format policy, on-demand KV planes and the host prefix tier; re-exported as `memra_engine::cache` | gguf, tier |
| `memra-tier` | Frozen exact-byte tiering contracts (budgets, leases, pools, placement, peers) | none |
| `memra-lanes` | Lane types (`interactive`, `judge`, `harvest`), SLO admission, engine-truth step stats, the audio stream scheduler | none |
| `memra-runtime` | CUDA runtime scaffolding and the `gemm-check` bin | gguf |
| `memra-validate` | Gate protocol: test vectors, error measures, tolerance bands, N-rep medians, the ALL GREEN tally | none |
| `memra-net-guard` | SSRF address classification for server-side `image_url` fetches; pure, no I/O | none |
| `memra-probe` | Unpublished development probe | none |

## 4. Request path

1. **HTTP.** `router()` in `crates/memra-server/src/lib.rs` registers every route. Four surfaces
   generate text: `/v1/chat/completions`, `/v1/completions`, `/v1/messages` (Anthropic) and
   `/v1/responses` (OpenAI Responses). Embeddings, rerank, tokenize, audio, models, health and
   metrics routes sit beside them. The route table and schemas are in
   [docs/API-SURFACES.md](docs/API-SURFACES.md).
2. **Auth before the body.** A middleware checks the API key from headers alone, before any body
   is read (`authenticate_inference_before_body`). The body ceiling is 192 MiB.
3. **Translation and validation.** `/v1/messages` and `/v1/responses` rewrite their bodies into
   the internal chat request and enter the same admission sequence (`surfaces.rs`). A field the
   engine cannot honor is a named 400, never a silent downgrade.
4. **Admission.** The handler resolves the tenant, the lane (`x-lane`, `memra-lanes`), the
   `timeout_ms` deadline and a per-tenant concurrency slot. Interactive requests past the
   session cap wait in a bounded queue; one past the queue bound, or whose estimated wait
   exceeds its deadline, is shed with a 429 before it reaches the worker. The judge and harvest
   lanes shed at their cap or SLO threshold when the worker admits them. A shed is an immediate
   retryable error, never a wait inside the engine.
5. **The worker.** The request crosses an mpsc channel as a `Cmd` to the GPU worker thread
   (`worker.rs`). Each serve route registers a `RouteContract`, checked at boot, that declares
   every policy surface (liveness, metrics, memory admission, rewrite bundle) as implemented or
   refused by name (`route_contract.rs`). DeepSeek-V4-Flash runs on its own route, serve threads
   and engine (`dsv4_serve.rs`, `memra_engine::dsv4_gpu`).
6. **The scheduler tick.** Each tick admits queued requests up to the lane caps, looks up the
   prefix cache, primes prefilling sessions in chunks under a per-lane token budget
   (`MEMRA_PREFILL_TICK`: 1024 tokens for the interactive lane, 256 for judge and harvest, 8192
   for a lone fresh interactive request), steps every decoding session together in batched
   decode, runs speculative verify where a drafter is attached, samples, checks stops and sends
   tokens back on the request's channel.
7. **The engine.** Forward passes live in `memra-engine` (`hybrid.rs` and `hybrid_forward.rs`
   for most families, `decode_batch.rs` for the batched trunk, `graph_update.rs` for CUDA-graph
   replay). Kernels live in `crates/memra-engine/cu/`; [docs/KERNELS.md](docs/KERNELS.md) maps
   each entry symbol to what it computes, its types, arch guard, selector, caller and gate.
8. **The response.** The handler renders the token stream in its own dialect, as SSE or one
   JSON body.

## 5. Model path

```
checkpoint -> reader -> ModelConfig -> model pack -> ModelPlan -> tensor contract
  -> execution manifests -> hand-written CUDA programs
     (RewriteParity receipts against memra-reference where recorded)
```

1. **Checkpoint.** A GGUF file (split shards included), an HF safetensors directory
   (`config.json` plus `model.safetensors[.index.json]`), a memra repack directory with a
   `manifest.json`, or an `hf:owner/repo[:file-substring]` spec that `memra-gguf/src/hf.rs`
   downloads into `$MEMRA_MODELS_DIR/hf/` (default `~/.cache/memra/models/hf/`). The GGUF,
   safetensors and repack readers implement one `TensorSource` trait (`source.rs`) and normalize
   metadata into one `ModelConfig`.
2. **Model pack.** `model_packs/<family>/` owns a family's aliases, config normalization, tensor
   schema, plan construction, tokenizer and template requirements and its gate list. Load-time
   selection walks `PACKS` in order, with `llama_dense` last. Onboarding profiles
   (`ONBOARDING_PROFILES`) are used only by `memra model inspect`. A pack also declares where its
   chat dialect comes from (`TemplateContract`): the artifact's own template, or a named renderer
   in the engine for families whose vendor ships the dialect as code (DeepSeek-V4 and V4.1). A
   renderer family refuses an artifact that carries a template of its own
   ([record](docs/history/MODELS.md#a-pack-declares-where-its-chat-dialect-comes-from-lanedsv4-template-contract-20260910)).
3. **ModelPlan.** `model_plan.rs` compiles the config into typed operations: norms, RoPE,
   attention and state mixers, gates, dense, MoE and shared MLPs, residual topology, logits
   transforms, MTP blocks and multimodal operations. It names no kernel.
4. **Tensor contract.** `tensor_contract.rs` maps checkpoint names to semantic tensor ids with
   shapes, ownership, transforms and quant layouts. `checkpoint_binding.rs` binds it against the
   tensor census before any byte uploads. A missing, extra, ambiguous or wrong-shape tensor
   refuses the load with the pack named.
5. **Execution manifests.** `op_registry.rs` declares, per plan operation, which optimized
   surfaces implement it. The manifests in `execution_manifest.rs` (`NATIVE_EAGER`,
   `DECODE_BATCH`, `DECODE_GRAPH`, `CARRIED_PRIME`, `CHUNKED_PRIME`, `MTP_SPEC`, `GLM5_SPEC`,
   `PIPELINE`) derive from that table. A plan can use a program only when every operation is
   covered; a missing operation fails closed and shows as a capability blocker. The rule for
   runtime policy: read the plan or its manifest, not an architecture-name list
   ([AGENTS.md](AGENTS.md)). The generated table is
   [docs/EXECUTION-SURFACES.md](docs/EXECUTION-SURFACES.md).
6. **Optimized programs.** The eager, batched, graph, carried-prime, speculative and pipeline
   programs are hand-written CUDA paths that run a plan when its manifest covers every operation.
   `memra model verify rewrite` checks one of them against `memra-reference` on a pinned artifact
   and records a `RewriteParity` receipt. A rewrite bundle (`MEMRA_REWRITE_BUNDLE`, built from
   `memra model inspect` and `verify`) installs those receipts, bound to the artifact lock and the
   compiled-plan hash. With a bundle, a surface without a receipt falls back to the
   receipt-backed native eager program, a required speculative or pipeline surface without one
   fails closed, and an invalid bundle fails the load. With no bundle, the native programs run
   under the legacy policy, held exact by the device gates (section 11). A few per-family arms
   (the HyperConnections batched-decode walk, the GLM-5.3-Flash TP walk, the DeepSeek-V4 route)
   still bypass the canonical programs; `execution_manifest::decode_batch_unconverted` names
   that gap.
7. **Onboarding.** `memra model inspect <source> --against <family> --out <dir>` writes the
   artifact lock, normalized config, tensor census, compiled plan, rewrite manifest and tokenizer
   contract. `memra model verify <stage>` runs one stage; the order is `config`, `tiny`,
   `checkpoint`, `rewrite`, `serve`. The procedure is [docs/ONBOARDING.md](docs/ONBOARDING.md).

### One numeric program per request

A request must not produce tokens under two numerical programs unless the transition is
forbidden or proven bit-identical. Unit gates do not catch this; serving-shape gates do. Pairs to
watch: prime and decode, monolithic and tokenwise, graph and eager, batched and solo,
speculative verify and plain decode. A new fast path either makes the crossing impossible or
proves identity across it ([AGENTS.md](AGENTS.md)).

## 6. KV cache, prefix cache and tiering

- **KV format.** The trunk KV cache is q8_0 K and q5_1 V (`memra-kv`, `kv_blk_bytes`). Gemma 4
  global layers keep e4m3 KV by default (`MEMRA_GEMMA_GKV`), and its windowed layers do too in plain
  serving (`MEMRA_GEMMA_WKV`; q8_0/q5_1 when `MEMRA_DRAFT` names an assistant drafter).
  Linear-attention layers (Gated DeltaNet and similar mixers) keep their recurrent state in the same
  dual cache.
- **KV planes.** On the full-power RTX PRO 6000 Blackwell Workstation Edition the server maps
  the full-attention K/V planes of a single-device session on demand through CUDA VMM; other
  cards, and that card when VMM is unavailable, use pooled planes
  ([decision](docs/decisions/KV-VMM-SERVING-DEFAULT.md)).
- **Prefix cache.** A cross-request prefix cache reuses prompt KV between requests in the same
  namespace (`cache_salt`, scoped per tenant when a keyring is configured). Eviction is plain LRU
  ([decision](docs/decisions/PREFIX-CACHE-POLICY.md)). Its size is derived at boot from free
  VRAM, the served context and the session peak unless `MEMRA_PREFIX_CACHE_MB` sets it. A
  continuation pool and the speculative pools hold the state of recent sessions.
- **Host tier.** `MEMRA_KV_HOST_MB` (off by default) adds a pinned-host prefix tier, governed by
  `memra-kv`'s `tiered::hostprefix` over the `memra-tier` contracts.
- **Expert spill.** MoE experts that do not fit in VRAM move between a GPU slot cache with SLRU
  eviction (`moe_cache.rs`) and pinned host memory, plus a memory-mapped disk tier when
  `MEMRA_SPILL_DISK` is set (`spill.rs`). A cache hit and a miss feed the kernel the same bytes,
  so spill changes placement, never numerics. The Hy3 profile for a 24 GB card is
  [docs/HY3-SPILL.md](docs/HY3-SPILL.md).

Operator knobs for all of these are in [docs/SERVING.md](docs/SERVING.md) and
[docs/FLAGS.md](docs/FLAGS.md).

## 7. Sampling, structured output and tokenization

- **Sampling.** `memra-sampling` is the host chain: penalties, temperature, top-k, top-p, min-p,
  then the draw. Greedy is argmax, the bit-exact reference. Batched decode samples on the device
  (Gumbel-max with Philox counters, so a draw depends on seed and position, not launch shape;
  `cu/spec_sample.cu`), and DeepSeek-V4 has its own device sampler (`cu/dsv4_sampler.cu`).
- **Structured output.** `response_format` JSON mode and JSON schema compile through llguidance
  into a token mask applied before the sampler (`constrained.rs`). `tool_choice` `required` or a
  named function arms a tool-call grammar (`tool_constraint.rs`). The draft chain of speculative
  decoding walks a clone of the grammar state, so a drafter never proposes a token the grammar
  forbids.
- **Tokenization.** `memra-tokenizer` builds the vocabulary from the model's own GGUF metadata or
  `tokenizer.json`, so token ids match that model exactly. Chat templates are rendered in Rust per
  dialect; there is no jinja engine. Tool definitions render through the template's tools branch,
  and emitted tool calls parse into OpenAI `tool_calls`. Input normalization rules are in
  [docs/TOKENIZER-PROGRAM.md](docs/TOKENIZER-PROGRAM.md).

## 8. Speculative decoding

Greedy speculative decoding is exact: accepted plus bonus tokens equal plain greedy decoding
token for token. Sampled speculative decoding uses rejection sampling on the verify side.

Drafter sources:

| Drafter | How it attaches | Draft program |
|---|---|---|
| MTP / NextN head embedded in the GGUF | automatic | `spec.rs` |
| A model's own MTP draft GGUF | `name=/model.gguf+/draft.gguf` in `MEMRA_MODELS` | `spec.rs` |
| A process-wide standalone MTP draft | `MEMRA_MTP_DRAFT=<draft.gguf>` | `spec.rs` |
| GLM-5.3-Flash MTP block | `MEMRA_GLM5_MTP=1` + `MEMRA_GLM5_SPEC=1` (off by default) | `glm_spec.rs` (`GLM5_SPEC`) |
| Gemma 4 assistant drafter | `MEMRA_DRAFT=<assistant.gguf>` | `gemma_spec.rs` |
| DFlash-style block drafter (DSpark) | `MEMRA_DSPARK_SPEC=1` + `MEMRA_DSPARK_DRAFT` | `dflash.rs` |

- **Vocab-masked draft heads.** An FR-Spec trim (`MEMRA_FRSPEC_TRIM`) restricts a draft head to
  a ranked subset of the vocabulary and maps draft ids back through a `d2t` table, so each draft
  step reads a fraction of the head bytes. Rank files come from the serving model's own
  generations ([docs/DRAFT-REGIME.md](docs/DRAFT-REGIME.md)).
- **The block drafter** drafts a block of tokens in one non-causal forward, conditioned on the
  target's hidden states.
- **Admission.** A concurrency gate (`MEMRA_SPEC_GATE`) admits drafting at low concurrency and
  keeps it off on a cross-device PP-2 placement, where plain batched decode measured faster.
  Draft depth K comes from a four-rung table keyed on cold or cached prompt, prompt length and a
  trimmed draft head, with a device-keyed cold-long rung. `MEMRA_SPEC_K_*` re-prices one rung;
  `MEMRA_SPEC_K` pins every request. `MEMRA_SERVE_SPEC=0` turns serving spec off.
- EAGLE3.1 exists only as the `run-eagle` research driver, not in the server.

The draft build recipe is [docs/DRAFT-REGIME.md](docs/DRAFT-REGIME.md).

## 9. Multi-GPU

| Mode | How | Notes |
|---|---|---|
| Pipeline parallel (PP) | `MEMRA_PP_STAGES=N`, cuts by `MEMRA_PP_SPLITS`, placement by `MEMRA_PP_DEVICES` | Weights shard per stage; transport is device-to-device, peer copy, or pinned host bounce (`pp.rs`) |
| Automatic placement | `MEMRA_PARALLEL=auto` + `MEMRA_PARALLEL_DEVICES` (2 to 4 distinct cards) | Plans from the `ModelPlan` and the byte census: contiguous PP, or whole-expert EP for a routed MoE with W4A16 experts when every card keeps its reserve; refuses before CUDA allocation when nothing fits; off by default ([decision](docs/decisions/PRO6000-MULTICARD.md)) |
| Tensor and expert parallel (TP, EP) | `MEMRA_STEP_TP`, `MEMRA_STEP_EP` (Step-3.7), `MEMRA_GLM5_TP` (GLM-5.3-Flash, TP-2 or TP-4), `MEMRA_DSV4_DEVICES` and `MEMRA_DSV4_TOPOLOGY` (DeepSeek-V4) | Per-family doors (`tp.rs`, `glm5_tp.rs`, `dsv4_topology.rs`); `tp.rs` is a correctness runtime, not throughput evidence |
| Replicas | one `memra-server` process per card or card group | An operator choice outside the engine |

Never call PP or replicas TP, and never force a TP degree that violates attention-head, KV-head or
expert partitioning. The planner's envelope is up to 8 cards; automatic placement covers 2 to 4.
Workload guidance: [docs/workloads/multi-gpu.md](docs/workloads/multi-gpu.md).

## 10. Support states

A pack's support state is a claim about its `ModelPlan` path. There are exactly three positive
states:

- **NativeReference**: the plan compiles and runs in the reference executor. It needs the `Config` and `TinyParity` gates. <!-- support: none; not NativeReference -->
- **NativeQualified**: adds `TokenizerTemplate`, `TensorCensus`, `CheckpointParity` and `Serve` on a pinned artifact, commit, topology and route. <!-- support: none; not NativeQualified -->
- **NativeTuned**: qualified, plus `RewriteParity` receipts for the selected optimized rewrites. <!-- support: none; not NativeTuned -->

A pack with no state (`support: None`) can be inspected but has no native plan execution claim.
Every record lives in [docs/support-records.toml](docs/support-records.toml), and
`tools/check-support-states.py` (CI gates job) checks that each pack declares exactly what its
records prove and that every state named in the docs carries a support marker (an HTML comment
naming its record, or denying the state on a line that only defines it).
Today every record holds the first state, and no pack has reached the second or third. The
speech packs (Whisper, Nemotron RNNT, Qwen3-TTS) sit outside `PACKS` and hold no state: the
reference tiny fixture refuses speech plans, and no speech operation has a CUDA kernel, so the
audio routes fail closed ([docs/SPEECH.md](docs/SPEECH.md)).

A pack state is separate from a serving route. A route is a measured configuration (artifact,
card, topology) with the gates it passed and the version it was last gated on. Loading through a
handwritten path, sharing an architecture name or running in another engine is not a support
state. The per-model index is [docs/MODELS.md](docs/MODELS.md).

## 11. Gates and evidence

| Gate | What must hold |
|---|---|
| `kernel-check` | Each kernel cell matches its CPU reference within its tolerance; prints `ALL GREEN (<n> cells, <m> skipped)` |
| `run-gen` argmax gate | The last-position argmax of a batched prefill matches tokenwise decode over the same prompt (`verify-prefill`), and so does the batched prime; `argmax-margin-probe` and `tools/argmax-margin-gate.sh` classify near-tie flips by margin |
| `run-spec` K=1..8 | Speculative output is token-identical to the plain target for every K, and acceptance is above zero |
| Serving gates | `tools/serve-smoke.sh` and served route pairs (speculative versus plain, restored versus cold) under each gate's identity rule |

- **Push evidence.** `tools/change_impact.py` maps a diff to (build target, scope) cells, and
  `tools/cell_evidence.py` records the device evidence each push-gated target needs.
- **Daily regression.** `tools/daily-regression.py` runs main on each device's full cell set.
- **Tag battery.** `tools/release-battery.sh` runs `kernel-check`, the argmax gate and `run-spec`
  on the two `tools/release-roster.tsv` checkpoints (Ornith-1.5-35B-A3B and Qwen3.8-27B). It
  carries no evidence for any other checkpoint and starts no server.

GitHub runners have no GPU; these gates run on the target hardware. The gate catalog is
[docs/TESTING.md](docs/TESTING.md); push and tag rules are
[docs/RELEASE-QUALIFICATION.md](docs/RELEASE-QUALIFICATION.md) and
[docs/RELEASING.md](docs/RELEASING.md).

## 12. Per-device defaults and flags

Measured winners become the defaults, so a command with no flags runs the best known path on the
card it finds; where the best path differs by hardware, the default is keyed to the device by
build family, unified-memory detection, VRAM or card class
(`memra_engine::parallel::HardwareTarget`).
An environment variable exists only as a runtime parameter, machine-specific setting, rollback
seam, diagnostic or blocked experiment, and the boot audit refuses retired names and unknown names
in an owned `MEMRA_<FAMILY>_` prefix ([docs/FLAGS.md](docs/FLAGS.md)).

## 13. Where to read next

| Question | Doc |
|---|---|
| Install or build | [docs/INSTALLATION.md](docs/INSTALLATION.md) |
| Run a model on a card | [docs/COOKBOOK.md](docs/COOKBOOK.md), [docs/MODELS.md](docs/MODELS.md), [docs/models/](docs/models/), [docs/rigs/](docs/rigs/) |
| Operate the server | [docs/SERVING.md](docs/SERVING.md) |
| HTTP schemas | [docs/API-SURFACES.md](docs/API-SURFACES.md) |
| Flags | [docs/FLAGS.md](docs/FLAGS.md) |
| Add a model | [docs/ONBOARDING.md](docs/ONBOARDING.md) |
| Kernels | [docs/KERNELS.md](docs/KERNELS.md), [docs/EXECUTION-SURFACES.md](docs/EXECUTION-SURFACES.md) |
| Speech | [docs/SPEECH.md](docs/SPEECH.md) |
| Test or release | [docs/TESTING.md](docs/TESTING.md), [docs/RELEASE-QUALIFICATION.md](docs/RELEASE-QUALIFICATION.md) |
| Numbers | [docs/PERFORMANCE.md](docs/PERFORMANCE.md) |
| Why a default is what it is | [docs/decisions/](docs/decisions/), [research/INDEX.md](research/INDEX.md) |
| The Hopper lane ledger | [ARCHITECTURE-H100.md](ARCHITECTURE-H100.md) |
| The June 2026 plan this page replaced | [docs/archive/ARCHITECTURE-2026-06.md](docs/archive/ARCHITECTURE-2026-06.md) |
| Every doc, by audience | [docs/README.md](docs/README.md) |
| Where any question lives (agents) | [docs/ROUTER.md](docs/ROUTER.md) |
