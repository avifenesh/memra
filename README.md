# memra

[![ci](https://github.com/avifenesh/memra/actions/workflows/ci.yml/badge.svg)](https://github.com/avifenesh/memra/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![release](https://img.shields.io/github/v/release/avifenesh/memra)](https://github.com/avifenesh/memra/releases/latest)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-tiyuvta-0A66C2?logo=linkedin)](https://www.linkedin.com/company/tiyuvta-ai)

Rust + CUDA inference engine tuned separately for RTX PRO 6000 Blackwell and RTX 5090, with
OpenAI-compatible serving and model-specific correctness gates.

[Install](docs/INSTALLATION.md) · [Models](docs/MODELS.md) ·
[Serving](docs/SERVING.md) · [Performance](docs/PERFORMANCE.md) ·
[Hosted API](https://inference.tiyuvta.ai/model?c=github-memra-readme)

> **Want to try Memra without operating a GPU?**
> [Open the hosted instance →](https://inference.tiyuvta.ai/model?c=github-memra-readme).
> Its live model catalog, access path, prices, and terms are documented there.

## Quick start

Prebuilt binaries require Linux x86_64, an NVIDIA driver with CUDA 13 runtime support, and the
CUDA runtime libraries. The installer selects the GPU build, verifies its checksum, and does not
require `nvcc`.

```bash
curl -fsSL https://raw.githubusercontent.com/avifenesh/memra/main/tools/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
kernel-check                  # expect: ALL GREEN
```

Start a server with a supported public checkpoint. The first run downloads and caches the model:

```bash
MEMRA_MODELS="q38=hf:Avifenesh/Qwen3.8-27B-NVFP4-MTP-GGUF:Q5K-mtp" \
memra-server
```

The server listens on `127.0.0.1:8080`. Send a streaming chat completion from another terminal:

```bash
curl -sS -N http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "q38",
    "messages": [{"role": "user", "content": "Explain KV caching in one sentence."}],
    "max_tokens": 128,
    "stream": true
  }'
```

Use the [cookbook](docs/COOKBOOK.md) for qualified model-and-card configurations. See
[installation](docs/INSTALLATION.md) for source builds and platform requirements.

## Project spirit

Memra is narrow on purpose. The best execution path depends on the model, checkpoint, hardware,
and workload, so Memra measures those combinations separately instead of promising that one loader
or one default supports everything. A fast path is promoted only when its own correctness and
performance evidence is current.

The project favors explicit support, reproducible receipts, and useful failures over silent
fallbacks. Issues, model requests, hardware reports, and evidence-backed pull requests are welcome.
Look elsewhere if you need broad hardware coverage, a large architecture catalog, or a Python
library embedded in your application.

## Choose a path

**By model:** [Qwen3.8 27B](docs/models/qwen38-27b.md) ·
[Qwen3.5 9B](docs/models/qwen35-9b.md) · [Qwen3.6 27B](docs/models/qwen36-27b.md) ·
[Qwen3.6 35B-A3B](docs/models/qwen36-35b-a3b.md) ·
[Qwen-AgentWorld 35B-A3B](docs/models/qwen-agentworld-35b-a3b.md) ·
[Ornith 1.5 35B-A3B](docs/models/ornith15-35b-a3b.md) ·
[Ornith 1.0 9B](docs/models/ornith10-9b.md) ·
[Ornith 1.0 35B](docs/models/ornith10-35b.md) ·
[Gemma 4 12B](docs/models/gemma4-12b.md) ·
[Gemma 4 26B-A4B](docs/models/gemma4-26b-a4b.md) ·
[Gemma 4 31B](docs/models/gemma4-31b.md) ·
[Gemma 4 E4B](docs/models/gemma4-e4b.md) ·
[Step 3.7 Flash](docs/models/step37-flash.md) ·
[DeepSeek V4 Flash](docs/models/deepseek-v4-flash.md) ·
[Hy3](docs/models/hy3.md) (NativeReference)

**By hardware:** [RTX PRO 6000 Blackwell](docs/rigs/rtx-pro-6000-blackwell.md) ·
[RTX 5090 / 50-series](docs/rigs/rtx-5090.md) · [H100](docs/rigs/h100.md) ·
[Ada](docs/rigs/ada.md) · [B200](docs/rigs/b200.md)

**By workload:** [interactive agents](docs/workloads/interactive-agents.md) ·
[long-context and prefix reuse](docs/workloads/long-context.md) ·
[batch throughput](docs/workloads/batch-throughput.md) ·
[multimodal](docs/workloads/multimodal.md) ·
[large models on multiple GPUs](docs/workloads/multi-gpu.md)

## What v0.123.0 ships

The glm5_next bring-up consolidation, the step37 NVFP4 program restore with its first
default flips, and two new-architecture bring-ups. Flag defaults are per-row decisions with
receipts in [docs/FLAGS.md](docs/FLAGS.md); numbers below carry their lane receipt paths and
are lane measurements, not serving claims.

- **glm5_next (GLM-5.3-Flash) full serving stack**, bring-up state: the 45-layer hybrid
  KDA+MLA/DSA architecture with sigmoid 288-expert MoE, Sinkhorn mHC, NoPE MLA and the DSA
  indexer, under a 3-card pipeline recipe (resident PP3, SPLITS 15,30). Ships the batched
  spec-verify walk (`MEMRA_GLM5_VERIFY_BATCH`, default ON, bit-gated per row), the DFlash2
  drafter spec session (`MEMRA_GLM5_SPEC` + auto-K + confidence floor), MLA tensor-core
  attention default ON (TTFD −62 to −69% on two boxes), hyper-batch default ON, vision
  default ON, the weight-read-once matvec door family default ON (T/X/K/W; mv-battery
  2026-08-31), verify-rows MoE kernels at 90% DRAM peak, and dedup-schedule/EP-diet doors
  default OFF pending box pricing. Greedy instrument figure on the ship recipe: 71.49 tok/s
  (3 cards); receipts under `research/glm53-flash-bringup-20260827/`. NOT serving-exposed:
  the serving bar (100 tok/s at ctx 262144) is an open lane, and no product claim ships here.
- **glm5 TP widened to rank 4, and spec decode composes with TP** (`GLM5_TP_ALLOWED_RANKS
  = [2, 4]`, TP-3 refuses by name): per-rank transport with an all-ordered-pairs
  byte-integrity ladder, peer-shard KDA/MLA sidecars, rig-gated bit identity; the blanket
  spec-session co-refusal on a TP-armed model becomes a gated admission behind
  `MEMRA_GLM5_SPEC_TP` (default OFF by design) with sharded verify/rollback through the
  batched walk only. The peer-pull transport door and movement census landed with the
  fail-closed TP matrix (`research/glm53-flash-bringup-20260827/composition-20260901/`).
- **`apply_penalties_dense`**: the host sampler's O(n_vocab) per-token hash-and-sort on
  penalized sampled rows replaced by a dense pass, bit-identical by a 24-case `to_bits`
  gate (old scan form kept as the oracle). Found by the host-audit lane tracing a live
  prod shape; `MEMRA_WORKER_AFFINITY` ships alongside as a default-OFF diagnostic seam whose
  box battery measured null on every arm (`research/glm53-flash-bringup-20260827/host-audit-20260901/`).
- **step37 NVFP4 bank-v3: the 2026-08-29 corruption root-caused and the programs restored.**
  The defect was a defaulted `in_f = 0` scale-fetch argument in the prefill grouped GEMM
  (right codes, wrong scale, every k-block but the first) — the slot-major layout was
  innocent. The default is deleted so the compiler enforces all call sites; the three
  removed programs return under three separate strict doors gated by the device-side
  `nvfp4-bank-oracle` with a behavioural teeth arm. First default flips ride the deploy-grade
  12-boot battery: `MEMRA_NVFP4_BANK_SM` + `MEMRA_NVFP4_SEL_DOWN8` default ON as one coupled
  decision (+5.44% decode / +5.92% wall on the vendor-default sampled shape, per-boot ranges
  separated 4/4, 16/16-turn cache twin holds; engages on the device-routed TP path — see
  the eligibility conditions in docs/FLAGS.md), `MEMRA_NVFP4_SEL_GU` stays OFF
  (`research/step37-bankv3-20260901/`).
- **hy3 native tune**: automatic expert-parallel device router with batch-cap admission,
  masked MTP, an internal W4A8/mixed-Q8 activation scope for whole-expert EP, generic TP
  attention composed with expert EP, and the shared-expert overlap door
  (`MEMRA_SHEXP_OVERLAP`, default OFF).
- **qwen4_exp (Qwen3.8-Flash-Next) bring-up**, NativeReference + GPU-eager with exactness
  gates: hybrid GDN 3:1 QSA, 512-expert softmax top-k router with gated shared expert,
  4-branch gated residual, PLE n-gram embedding, YaRN with refuse-at-parse for unimplemented
  keys. Loader/reference lane only; no serving exposure and no product claims
  (`research/qwen4exp-bringup-20260829/`).
- **Public-boundary checker hardened for symlinks**: a tracked symlink publishes only its
  target string and is scanned as such, never dereferenced (a box-absolute link crashed the
  checker on CI runners); the tree itself now carries zero box-absolute links.

## What v0.122.0 ships

KV host tier and serving-guard release. Every new flag defaults OFF with an audited row in
[docs/FLAGS.md](docs/FLAGS.md); numbers below are from the 2026-08-31 qualification pod battery
(2x RTX PRO 6000 Blackwell 96 GB).

- **Graph-launch guard on every serving-reachable captured-graph route.** When driver-free
  memory drops below the 256 MB launch floor, captured-graph replay suspends with a
  route-tagged `graph replay suspended:` line and the request serves on the eager arms
  (fail-closed to eager, never a segfault into an exhausted card). Fired 5/5 squeeze runs on
  each of the q38 verify-graph, ornith MTP verify-graph, and step37 TP-2 routes; zero
  suspended lines at healthy headroom.
- **Prefix-cache host spill tier**, `MEMRA_KV_HOST_MB` (default OFF): device-cache evictions
  demote verbatim into pinned host RAM and promote back through the existing restore path,
  byte-lossless by construction. Gates: restored-vs-cold byte identity (ON == OFF bytes,
  verify digests ok, teeth arm inverts as required); the 8-turn larger-prompt cache twin
  holds TTFT flat (0.61 to 0.77 s p50) through turn 8 while the no-tier arm grows to
  3.88 s p50, a 5.6x p50 TTFT gap at turn 8 in that shape.
- **Tenant lifecycle purge and per-tenant share cap.** `PurgeHandle::purge_tenant` clears a
  tenant's resident host-tier and unpinned device entries on key revocation or deletion
  (`/admin/tenants/{tenant}/purge`); `MEMRA_KV_HOST_TENANT_PCT` (default 50) caps one
  tenant's share of the host pool.
- **Plain-pool park compaction**, `MEMRA_KV_PARK_COMPACT` (default OFF): a retiring
  continuation-pool session parks at exactly its committed length instead of its ladder cap;
  resume restores the parked rows, byte identity after replay 4/4 under the step-OOM
  adjacency battery.
- **Agent-pause KV demotion**, `MEMRA_KV_PAUSE_DEMOTE` (default OFF): a turn that ends in a
  completed tool call arms a pause candidate and demotes its boundary state to the host tier
  after `MEMRA_KV_PAUSE_DEMOTE_MS` (default 5000 ms, set from the A3 gap census). Natural
  `tool_calls` arm 6/6 on both boots; 16 verify-ok round trips, 0 failed; co-run decode tax
  -1.80% median.
- **Predictive-admission shadow receipts**, `MEMRA_ADMIT_PREDICT_SHADOW` (default OFF):
  log-only per-request admit/reject verdicts with the full KV book; nothing is rejected.
- **Boot calibration probes the served route**: the admission floor probe rides the route the
  model actually serves (q38 dspark boot charges zero MTP draft-state; the ornith MTP route
  charges its real measured draft state).
- **Verify-graph pool debt charged by struct**: the MTP verify-graph pool no longer escapes
  admission (by-struct reserved debt plus a per-session measured capture charge).
- **Offline expert-placement map builder**, `tools/build_expert_placement_map.py` (frozen
  format `memra-ep-map-v1`; strategies coactivation, frequency, even; selftest 10/10 with
  proven teeth).

## Documentation

| Document | Use it for |
|---|---|
| [Installation](docs/INSTALLATION.md) | Prebuilt requirements, source builds, architecture selection |
| [Model cards](docs/models/) | Best starting path for each supported model |
| [Hardware cards](docs/rigs/) | Recommended use by GPU target |
| [Workload cards](docs/workloads/) | Recommended use by request shape |
| [Cookbook](docs/COOKBOOK.md) | Copy-paste model and card configurations |
| [Models](docs/MODELS.md) | Supported checkpoints, formats, drafters, and hardware |
| [Serving](docs/SERVING.md) | HTTP contract, caching, auth, admission, multi-GPU, operations |
| [API surfaces](docs/API-SURFACES.md) | Anthropic Messages and OpenAI Responses compatibility |
| Embeddings & rerank | `/v1/embeddings` (OpenAI schema) and `/v1/rerank` (Cohere shape) — prefill-only capture surfaces, [Serving](docs/SERVING.md) |
| [Performance](docs/PERFORMANCE.md) | Measurements, methodology, rigs, and receipts |
| [Flags](docs/FLAGS.md) | Audited environment-variable reference |
| [Testing](docs/TESTING.md) | Correctness gates and evidence requirements |
| [Architecture](ARCHITECTURE.md) | Runtime structure and Blackwell implementation ledger |
| [Decisions](docs/decisions/) | Adopted and rejected design choices with evidence |
| [Releases](https://github.com/avifenesh/memra/releases) | Changelog and release artifacts |

## Issues and requests

Issues and requests are welcome:

- [Report a bug](https://github.com/avifenesh/memra/issues/new?template=bug-report.md)
- [Request a model or feature](https://github.com/avifenesh/memra/issues/new)
- [Submit a hardware validation](.github/ISSUE_TEMPLATE/hardware-validation.md)
- [Report a vulnerability privately](SECURITY.md)

For a model request, include the exact checkpoint, quantization, target GPU, and why the model is
useful. For a performance report, include the command, model artifact, hardware, and raw output.

## Contributing

Issues are welcome even without a proposed fix. Pull requests are welcome when they include the
proof required for the affected path; GPU changes cannot be validated by GitHub's compile-only
runners. Start with [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT — see [LICENSE](LICENSE). Built by [Avi Fenesh](https://github.com/avifenesh) at
[tiyuvta](https://tiyuvta.ai).
