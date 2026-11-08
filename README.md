# memra

[![ci](https://github.com/avifenesh/memra/actions/workflows/ci.yml/badge.svg)](https://github.com/avifenesh/memra/actions/workflows/ci.yml)
[![License: FSL-1.1-ALv2](https://img.shields.io/badge/license-FSL--1.1--ALv2-blue.svg)](LICENSE)
[![release](https://img.shields.io/github/v/release/avifenesh/memra)](https://github.com/avifenesh/memra/releases/latest)

**Memra is a Rust + CUDA LLM inference engine with hand-written kernels, NVFP4 and
mixed-precision execution, and correctness gates scoped to each model and GPU.**

The primary tuned targets are RTX PRO 6000 Blackwell and RTX 5090. One server provides
OpenAI chat, completions and Responses APIs, plus Anthropic Messages, without Python in
the serving path.

Built by [Avi Fenesh](https://github.com/avifenesh) at [tiyuvta](https://tiyuvta.ai).

[Install](docs/INSTALLATION.md) · [Models](docs/MODELS.md) ·
[Serving](docs/SERVING.md) · [Cookbook](docs/COOKBOOK.md) ·
[Performance](docs/PERFORMANCE.md)

## Project status

Memra is under active development. Current work focuses on model-specific correctness,
per-device defaults, native model support and measured serving improvements. Support and
performance claims land with their own gates and raw receipts, not from a successful model
load or a shared architecture name.

The x86_64 installer uses tagged releases. `main` can contain newer work that has not yet been
released or qualified on every target.

## Quick start

This quick start uses the x86_64 prebuilt installer. **DGX Spark GB10 is ARM64 (`aarch64`):
do not use this installer.** [Build from source on the GB10 host](docs/INSTALLATION.md#build-from-source);
no GB10 prebuilt is published. The [GB10 card](docs/rigs/dgx-spark.md) has the target details.

The x86_64 prebuilt binaries require Linux, glibc 2.35 or newer, an NVIDIA driver 580 or newer,
and the CUDA 13 runtime libraries (`cudart`, `cublas` and `cublasLt`). The installer selects
the GPU build and checks its SHA-256 digest. You do not need the CUDA toolkit or `nvcc`.

### 1. Install and check the CUDA path

```bash
curl -fsSL https://raw.githubusercontent.com/avifenesh/memra/main/tools/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
kernel-check
```

A successful run ends with `ALL GREEN`. Optional model-backed cells can report `SKIP` when
their artifacts are not installed.

### 2. Start a local server

This example uses Qwen3.8-27B NVFP4 with its MTP head. The first run downloads and caches
the model through the `hf` CLI:

```bash
pip install -U 'huggingface_hub[cli]'
MEMRA_MODELS="q38=hf:Avifenesh/Qwen3.8-27B-NVFP4-MTP-GGUF:Q5K-mtp" \
memra-server
```

The server listens on `127.0.0.1:8080`.

### 3. Send a request

From another terminal:

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
[installation](docs/INSTALLATION.md) for source builds, other GPU targets and troubleshooting.

## Why Memra

- **Support is specific.** A model, quantization, drafter and GPU combination earns its own
  support state. A checkpoint loading is not enough.
- **Correctness comes before speed.** Kernel, generation, speculative-decoding and serving
  gates compare each optimized path with its designated reference before promotion.
- **Defaults fit the device.** RTX PRO 6000 and RTX 5090 keep separate measured defaults
  where their best execution paths differ.
- **The serving path is native.** A Rust runtime drives ahead-of-time compiled CUDA kernels,
  with no Python framework between the HTTP request and the model.

Speculative decoding, prefix caching, tool calls, structured output, embeddings and rerank
are part of the same serving surface. See [Serving](docs/SERVING.md) and
[API surfaces](docs/API-SURFACES.md) for the contracts and their limits.

### When Memra is not the right fit

Look elsewhere if you need broad hardware coverage, a large architecture catalog, or a
Python library embedded in your application. Memra is deliberately narrow: it favors a
smaller set of measured model-and-hardware paths over universal claims.

## Choose a path

**Start with a model:** [Qwen3.8 27B](docs/models/qwen38-27b.md) ·
[Qwen3.5 9B](docs/models/qwen35-9b.md) · [Gemma 4 31B](docs/models/gemma4-31b.md) ·
[Step 3.7 Flash](docs/models/step37-flash.md) ·
[DeepSeek V4 Flash](docs/models/deepseek-v4-flash.md) (experimental) ·
[all model cards](docs/models/)

**Browse by hardware:** [RTX PRO 6000 Blackwell](docs/rigs/rtx-pro-6000-blackwell.md) ·
[RTX 5090 / 50-series](docs/rigs/rtx-5090.md) ·
[DGX Spark GB10](docs/rigs/dgx-spark.md) (measured bring-up) ·
[H100](docs/rigs/h100.md) · [Ada](docs/rigs/ada.md) ·
[B200](docs/rigs/b200.md) (runtime-qualified source backend; source builds only)

**Browse by workload:** [interactive agents](docs/workloads/interactive-agents.md) ·
[long context and prefix reuse](docs/workloads/long-context.md) ·
[batch throughput](docs/workloads/batch-throughput.md) ·
[multimodal](docs/workloads/multimodal.md) ·
[large models on multiple GPUs](docs/workloads/multi-gpu.md)

## How it works

Memra compiles a model into a typed execution plan, binds the checkpoint tensors to that
plan, and validates optimized rewrites against a portable reference executor. The Rust host
runtime drives hand-written CUDA kernels embedded in the binary. Model packs, support
records and gate receipts keep each claim tied to the artifact and hardware that produced it.
See [Architecture](ARCHITECTURE.md) for the design and its decision ledger.

### Support states

There are three positive support states:

- **NativeReference**: the plan compiles and runs in Memra's reference executor. This is <!-- support: none; not NativeReference -->
  bring-up evidence, not serving qualification.
- **NativeQualified**: the required checkpoint and serving gates pass. <!-- support: none; not NativeQualified -->
- **NativeTuned**: the path is qualified and has current receipts for the optimized <!-- support: none; not NativeTuned -->
  rewrites it selects.

Each pack's state is backed by [`docs/support-records.toml`](docs/support-records.toml), and
CI rejects claims that go beyond those records. [Models and hardware](docs/MODELS.md) is the
full support matrix.

### Correctness gates

`kernel-check` compares kernel outputs with each cell's reference implementation and declared
tolerance. `run-gen` compares greedy tokens with the native reference path and requires the
same tokens. `run-spec` compares greedy outputs across draft lengths K=1..8 and requires
identical results. Serving gates compare named route pairs, such as speculative versus plain
decoding and restored versus cold sessions, under each gate's stated identity rule.

GitHub CI cannot run these NVIDIA qualification gates, so they run on the target hardware.
A merge runs the cells a change reaches, or the full local battery including served-path
identity gates. A tag uses the release battery: `kernel-check`, calibrated argmax and
`run-spec` on two qwen35-class checkpoints, without a serving gate. The gate catalog and
evidence requirements are in [Testing](docs/TESTING.md).

## Documentation

| Document | Use it for |
|---|---|
| [Installation](docs/INSTALLATION.md) | Requirements, prebuilt installs, source builds and troubleshooting |
| [Cookbook](docs/COOKBOOK.md) | Copy-paste model-and-card configurations |
| [Models](docs/MODELS.md) and [model cards](docs/models/) | Supported checkpoints, quantization, drafters and recommended paths |
| [Hardware cards](docs/rigs/) | Recommended use and qualification scope for each GPU target |
| [Workload cards](docs/workloads/) | Guidance by request shape and serving workload |
| [Serving](docs/SERVING.md) and [API surfaces](docs/API-SURFACES.md) | HTTP contracts, caching, admission, authentication and multi-GPU operation |
| [Performance](docs/PERFORMANCE.md) | Measurements, conditions, methods and receipts |
| [Testing](docs/TESTING.md) | Correctness gates and evidence requirements |
| [Flags](docs/FLAGS.md) | Audited runtime configuration reference |
| [Architecture](ARCHITECTURE.md) and [decisions](docs/decisions/) | Runtime design and the evidence behind adopted or rejected choices |
| [Releases](https://github.com/avifenesh/memra/releases) | Tagged binaries and release notes; `main` can run ahead of the latest tag |

## Contributing

Bug reports, hardware results and evidence-backed pull requests are welcome.

- [Report a bug](https://github.com/avifenesh/memra/issues/new?template=bug-report.md)
- [Request a model or feature](https://github.com/avifenesh/memra/issues/new)
- [Submit a hardware validation](.github/ISSUE_TEMPLATE/hardware-validation.md)
- [Report a vulnerability privately](SECURITY.md)

For a model request, include the full checkpoint ID, quantization, target GPU and use case. A
request does not create a support commitment; native support still requires implementation
and qualification. For a performance report, include the command, model artifact, hardware
and raw output.

Start with [CONTRIBUTING.md](CONTRIBUTING.md). Pull requests need the proof required for the
path they change; GitHub's runners cannot validate NVIDIA execution.

## License

[FSL-1.1-ALv2](LICENSE). Each version becomes available under Apache-2.0 two years after it
is first made available. Earlier revisions and package releases remain under the license
that accompanied them. Third-party components retain their own terms; see
[THIRD-PARTY.md](crates/memra-engine/THIRD-PARTY.md).
