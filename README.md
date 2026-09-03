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
[Qwen3.8 Flash Next](docs/models/qwen38-flash-next.md) (bring-up only) ·
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
[GLM-5.3 Flash](docs/models/glm53-flash.md) (NativeReference) ·
[Hy3](docs/models/hy3.md) (NativeReference · NVFP4 NativeQualified)

**By hardware:** [RTX PRO 6000 Blackwell](docs/rigs/rtx-pro-6000-blackwell.md) ·
[RTX 5090 / 50-series](docs/rigs/rtx-5090.md) · [H100](docs/rigs/h100.md) ·
[Ada](docs/rigs/ada.md) · [B200](docs/rigs/b200.md)

**By workload:** [interactive agents](docs/workloads/interactive-agents.md) ·
[long-context and prefix reuse](docs/workloads/long-context.md) ·
[batch throughput](docs/workloads/batch-throughput.md) ·
[multimodal](docs/workloads/multimodal.md) ·
[large models on multiple GPUs](docs/workloads/multi-gpu.md)

## Overview

memra is a from-scratch LLM inference engine: a Rust host runtime driving hand-written CUDA
kernels, compiled ahead of time into fatbins embedded in the binary, with no Python and no
framework in the serving path. It targets one card class at a time and is tuned separately for
RTX PRO 6000 Blackwell and RTX 5090 (sm_120a), with B200 (sm_100a) as a runtime-qualified
source lane and Hopper/Ada supported via source builds.
The design constraints and the decisions they forced are in [ARCHITECTURE.md](ARCHITECTURE.md).

### Crates

| Crate | Owns |
|---|---|
| `memra-engine` | The CUDA engine: kernels, model programs, KV cache, speculative decoding, multi-GPU placement |
| `memra-server` | OpenAI-compatible HTTP serving (chat, completions, Messages, Responses, embeddings, rerank), admission, prefix cache |
| `memra-gguf` | GGUF and safetensors loading, quant-format decode, tensor inventory, model packs and the support-state enum |
| `memra-tokenizer` | GGUF-native BPE and SPM tokenizer with chat templates |
| `memra-sampling` | Host-side sampler chain (temperature, top-k, top-p, penalties) |
| `memra-kv` | KV-cache format policy (q8_0, q5_1, q4_0, fp8 block layouts) |
| `memra-lanes` | Serving-lane types, SLO admission policy, engine-truth step stats |
| `memra-reference` | Portable unfused reference executor for correctness gates |
| `memra-runtime` | CUDA runtime scaffolding (context, GEMM checks) |
| `memra-validate` | Numeric validation helpers (reference comparisons, tolerance gates) |
| `memra-cli` | Model onboarding compiler: `memra model inspect`, `model pack`, `model verify` |
| `memra-probe` | Unpublished dev spike (`publish = false`) |

### Support states

Support is specific to a model, quantization, and drafter combination, never to a format. There
are exactly three positive states, the enum `NativeSupport` in
`crates/memra-gguf/src/model_packs/mod.rs`:

- **NativeReference**: the plan compiles and runs in memra's reference executor. Bring-up
  evidence only.
- **NativeQualified**: the required checkpoint and serving gates pass.
- **NativeTuned**: qualified, plus current receipts for the optimized rewrites the deployment
  selects.

"Loads", "shares an architecture name", and "works through another engine" are not support
states. [docs/MODELS.md](docs/MODELS.md) is the support matrix; each entry in
[docs/models/](docs/models/) is the shortest recommended path for one model.

### Correctness gates

Every fast path is promoted only against its own oracle. Kernel-level bit identity
(`kernel-check`), argmax parity of generation against the reference path (`run-gen`),
speculative-decode self-consistency across draft lengths (`run-spec`), and served-path
byte identity (spec versus plain, restored versus cold) are the standing battery; GitHub CI is
compile-only, so the battery runs on a GPU before a merge or a tag. Greedy decoding is the
instrument, not the product: it is what makes byte-level gates possible. The gate catalog and
what each one proves is [docs/TESTING.md](docs/TESTING.md).

### Releases

Tagged releases carry prebuilt binaries (the installer above reads them) and publish the
workspace to crates.io. The repository history was rebuilt from a content snapshot on
2026-09-01 (a zero-history swap), so the GitHub Releases list here starts again with the first
tag cut on this history; crates.io is continuous, with `memra-engine 0.123.0` published
2026-09-01T04:59Z from old-history commit `bc0952fe5`, which is tag `v0.123.0`. The notes for
v0.122.0 and v0.123.0 are kept in
[docs/archive/RELEASE-NOTES-v0.122.0-v0.123.0.md](docs/archive/RELEASE-NOTES-v0.122.0-v0.123.0.md).
Release mechanics are in [docs/RELEASING.md](docs/RELEASING.md).

## Documentation

| Document | Use it for |
|---|---|
| [Installation](docs/INSTALLATION.md) | Prebuilt requirements, source builds, architecture selection |
| [Model cards](docs/models/) | Best starting path for each supported model |
| [Hardware cards](docs/rigs/) | Recommended use by GPU target |
| [Workload cards](docs/workloads/) | Recommended use by request shape |
| [Cookbook](docs/COOKBOOK.md) | Copy-paste model and card configurations |
| [Models](docs/MODELS.md) | Supported checkpoints, formats, drafters, and hardware |
| [Serving](docs/SERVING.md) | HTTP contract, caching, auth, admission, multi-GPU, operations. Since v0.125.0 a DFlash2 draft proposal carrying the top-k selector's exhausted-slot sentinel refuses that one request by name instead of panicking the GPU worker, and a BUSY worker is judged on forward progress rather than heartbeat silence |
| [API surfaces](docs/API-SURFACES.md) | Anthropic Messages and OpenAI Responses compatibility |
| Embeddings and rerank | `/v1/embeddings` (OpenAI schema) and `/v1/rerank` (Cohere shape): prefill-only capture surfaces; every item of a multi-item request is metered under its own ledger id `<x-request-id>.<index>` (v0.124.1), [Serving](docs/SERVING.md) |
| [Performance](docs/PERFORMANCE.md) | Measurements, methodology, rigs, and receipts |
| [Flags](docs/FLAGS.md) | Audited environment-variable reference |
| [Testing](docs/TESTING.md) | Correctness gates and evidence requirements |
| [Architecture](ARCHITECTURE.md) | Runtime structure and Blackwell implementation ledger |
| [Decisions](docs/decisions/) | Adopted and rejected design choices with evidence |
| [Releases](https://github.com/avifenesh/memra/releases) | Release artifacts; notes for the pre-swap tags in [docs/archive/](docs/archive/RELEASE-NOTES-v0.122.0-v0.123.0.md) |

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

MIT, see [LICENSE](LICENSE). Built by [Avi Fenesh](https://github.com/avifenesh) at
[tiyuvta](https://tiyuvta.ai).
