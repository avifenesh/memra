# memra documentation

The map of memra's documentation: reading paths by audience, then every document with the
question it answers. Memra is a Rust and CUDA LLM inference engine with hand-written kernels,
NVFP4 and mixed-precision execution, and correctness gates scoped to each model on each GPU.
`memra-server` serves the OpenAI and Anthropic APIs from one binary, with measured defaults per
device. The [front page](../README.md) has the quick start and the
[project status](../README.md#project-status).

## Reading paths

- **Run it:** [Installation](INSTALLATION.md), then the [Cookbook](COOKBOOK.md) for your model and
  card. Pick with [model cards](models/README.md), [hardware cards](rigs/README.md) and
  [workload cards](workloads/README.md). Operate with [Serving](SERVING.md),
  [API surfaces](API-SURFACES.md) and [Flags](FLAGS.md).
- **Understand it:** [ARCHITECTURE.md](../ARCHITECTURE.md) first, then
  [Execution surfaces](EXECUTION-SURFACES.md), [Kernels](KERNELS.md) and
  [Draft regime](DRAFT-REGIME.md).
- **Contribute:** [CONTRIBUTING.md](../CONTRIBUTING.md), then [Testing](TESTING.md) and
  [Release qualification](RELEASE-QUALIFICATION.md). A new model starts at
  [Onboarding](ONBOARDING.md).
- **Evidence and history:** [Performance](PERFORMANCE.md) for measurements,
  [decisions](decisions/README.md) for why each choice was made,
  [research/INDEX.md](../research/INDEX.md) for what each research lane concluded, and the
  [history files](#history-files) for text that left the current docs.

## Run it

| Document | Answers |
|---|---|
| [INSTALLATION.md](INSTALLATION.md) | How do I install a prebuilt or build from source for my GPU, fetch a model, and fix a failed first run? |
| [COOKBOOK.md](COOKBOOK.md) | What exact command serves this model on this card, and which boot line proves the right artifact loaded? |
| [MODELS.md](MODELS.md) | Which models run, and for each one its pack state and its measured serving routes? (one row per model) |
| [models/](models/README.md) | What is the shortest recommended path for one model? (one card per model) |
| [rigs/](rigs/README.md) | What should I run on my GPU, and what was measured on it? (one card per GPU) |
| [workloads/](workloads/README.md) | Which model, card and settings fit my request shape: agents, long context, batch, images, multi-GPU? |
| [SERVING.md](SERVING.md) | How do I operate `memra-server`: models, auth, admission, caching, multi-GPU, health, shutdown? |
| [API-SURFACES.md](API-SURFACES.md) | Which HTTP routes and request fields exist, and how do Messages and Responses map onto chat? |
| [FLAGS.md](FLAGS.md) | What does each `MEMRA_*` variable do, what is its default, and which names are retired? |
| [deploy/README.md](../deploy/README.md) | What do the example systemd units show about supervising `memra-server`? |

## Understand it

| Document | Answers |
|---|---|
| [ARCHITECTURE.md](../ARCHITECTURE.md) | How does memra work end to end: crates, request path, model path, KV, sampling, speculative decoding, support states, gates? |
| [EXECUTION-SURFACES.md](EXECUTION-SURFACES.md) | Which execution surface (eager, batched, graph, speculative, prime, pipeline) implements each plan operation? (generated) |
| [KERNELS.md](KERNELS.md) | Which CUDA kernel does what, for which quant type and architecture, behind which dispatch and FFI binding? |
| [THIRD-PARTY.md](../crates/memra-engine/THIRD-PARTY.md) | Which kernels are ported from other projects, and under which license? |
| [DRAFT-REGIME.md](DRAFT-REGIME.md) | How is a model's speculative draft built: ranks, extraction, trim, quantization? |
| [TOKENIZER-PROGRAM.md](TOKENIZER-PROGRAM.md) | How does the tokenizer apply Hugging Face normalizers and added tokens? |
| [HY3-SPILL.md](HY3-SPILL.md) | How does Hy3 run when its expert bank exceeds VRAM and host RAM? |
| [SPEECH.md](SPEECH.md) | What do the speech (ASR and TTS) packs cover, and what would a served speech path need? |
| [ASR-MODALITY-PLAN.md](../ASR-MODALITY-PLAN.md) | What is the native ASR plan, and what did its bring-up measure for the first speech checkpoints? |

## Contribute

| Document | Answers |
|---|---|
| [CONTRIBUTING.md](../CONTRIBUTING.md) | What proof does my change need, and how do I produce it? |
| [ONBOARDING.md](ONBOARDING.md) | How do I bring up a new model family as a native plan, gate by gate? |
| [TESTING.md](TESTING.md) | Which gates exist, what does each catch, and which do I run for my change? |
| [VALIDATION-SELECTION.md](VALIDATION-SELECTION.md) | How are CPU checks and native evidence selected from the inputs a change touches? |
| [CI.md](CI.md) | What do the GitHub CI jobs run, and what do they not prove? |
| [RELEASE-QUALIFICATION.md](RELEASE-QUALIFICATION.md) | What native evidence does a push or a tag need, per build target and cell? |
| [RELEASING.md](RELEASING.md) | How do I claim a version number and cut a release? |
| [tools/README.md](../tools/README.md) | Which script or small binary does a build, gate, measurement or release step? |
| [tools/fast-gate/README.md](../tools/fast-gate/README.md) | Which quick checks does a diff need in the edit-compile loop, between full-battery runs? |
| [tools/fast-gate/BUILD-REUSE.md](../tools/fast-gate/BUILD-REUSE.md) | How do I restore an exact controlled build instead of rebuilding it? |
| [tools/sparse-input-preflight.md](../tools/sparse-input-preflight.md) | Are the inputs of the CPU checks materialized in my sparse checkout? |
| [ROUTER.md](ROUTER.md) | Where does a given kind of question live? (a short index for agents) |
| [AGENTS.md](../AGENTS.md) | What are the standing rules for work in this repo? |
| [SECURITY.md](../SECURITY.md) | How do I report a vulnerability, and what is in scope? |

## Evidence and records

| Document | Answers |
|---|---|
| [PERFORMANCE.md](PERFORMANCE.md) | What was measured, on which card, under which conditions, with which receipts? |
| [COMPETITOR-SETUP.md](COMPETITOR-SETUP.md) | How were other engines configured for comparison runs? |
| [support-records.toml](support-records.toml) | Which onboarding gates has each model pack passed, and on which artifact? |
| [decisions/](decisions/README.md) | What was chosen or rejected, and which measurement settled it? |
| [FLAGS-HISTORY.md](FLAGS-HISTORY.md) | What did a flag row record before its history moved out of FLAGS.md? |
| [ARCHITECTURE-H100.md](../ARCHITECTURE-H100.md) | What was tried and measured on the Hopper (sm_90a) lane? (append-only ledger) |
| [research/INDEX.md](../research/INDEX.md) | What did each research lane conclude, and where are its raw receipts? |
| [ASR lane status](../research/asr-modality-20260909/STATUS.md) | Where does the ASR lane stand? (moved from the root `STATUS.md`) |
| [SFT trace generation](../research/cx-sft-20260808/README.md) | How does `tools/sft-gen.py` generate fine-tuning traces? (moved from `docs/SFT-TRACE-GENERATION.md`) |

## History files

Each file holds text moved out of a current doc on 2026-11-12, verbatim except for private
deployment detail. The current doc states what is true now; these keep the measurements,
decisions and gate histories it no longer carries.

| File | Holds the earlier text of |
|---|---|
| [history/API-SURFACES.md](history/API-SURFACES.md) | API-SURFACES.md: the Messages section, accounting parity across dialects, bounded choices |
| [history/CI.md](history/CI.md) | CI.md: CPU checks and native qualification, requesting a GPU run, the input manifest, receipt custody |
| [history/COOKBOOK.md](history/COOKBOOK.md) | COOKBOOK.md: the per-model launch blocks and their measurements as of commit `d1ed663256` |
| [history/HY3-SPILL.md](history/HY3-SPILL.md) | HY3-SPILL.md: the page as of commit `d1ed663256`, with its run profile and tuning state |
| [history/KERNELS.md](history/KERNELS.md) | KERNELS.md: the full inventory before the rewrite, with its dated lane sections |
| [history/MODELS.md](history/MODELS.md) | MODELS.md and the model and hardware cards: support notes, format direction, family status, lane notes |
| [history/ONBOARDING.md](history/ONBOARDING.md) | ONBOARDING.md: the pre-ModelPlan phase flow, the gate spec example and the Qwen 3.8 27B FP8-ST runbook |
| [history/PERFORMANCE.md](history/PERFORMANCE.md) | PERFORMANCE.md and COMPETITOR-SETUP.md: rig notes, historical boards, the serving exactness arc |
| [history/README-and-CONTRIBUTING.md](history/README-and-CONTRIBUTING.md) | README.md and CONTRIBUTING.md: the earlier proof policy, the perf battery rule and the scope statement |
| [history/RELEASE-QUALIFICATION.md](history/RELEASE-QUALIFICATION.md) | RELEASE-QUALIFICATION.md: the earlier record rules, argmax cells, source identity and build inputs |
| [history/RELEASING.md](history/RELEASING.md) | RELEASING.md: the version scheme, the pre-tag gate, the 2026-09-01 history swap, publishing setup |
| [history/TESTING.md](history/TESTING.md) | TESTING.md, VALIDATION-SELECTION.md and the `tools/` READMEs: the gate structure, admissions, receipts |

## Archive

Superseded plans and release records, each with a banner naming what replaced it.

| File | What it is |
|---|---|
| [archive/ARCHITECTURE-2026-06.md](archive/ARCHITECTURE-2026-06.md) | The June 2026 tech-stack plan and sm_120 feasibility ledger (the old root ARCHITECTURE.md) |
| [archive/ROADMAP-2026-06.md](archive/ROADMAP-2026-06.md) | The June 2026 build roadmap that went with that plan |
| [archive/RELEASE-NOTES-v0.122.0-v0.123.0.md](archive/RELEASE-NOTES-v0.122.0-v0.123.0.md) | Release notes for v0.122.0 and v0.123.0, from before the 2026-09-01 history rebuild |
| [archive/CHANGELOG-preswap-v0.123.0-to-469c8898e.md](archive/CHANGELOG-preswap-v0.123.0-to-469c8898e.md) | The changelog from v0.123.0 to the last commit of the old history |
