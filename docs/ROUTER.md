# Doc router (agents)

One line per question: question -> path (what it holds). Paths are repo-root relative. This index
holds addresses, not content; read the target before acting. Humans start at README.md (front
page) and docs/README.md (doc map), and the one-read engine overview is ARCHITECTURE.md.

## Engine

- How memra works end to end: crates, request path, model path, support states -> ARCHITECTURE.md
- Which kernels exist, symbols, build and FFI binding -> docs/KERNELS.md (update rows with the .cu change)
- What flag X does, its default, its rollback seam -> docs/FLAGS.md; a row's receipt history -> docs/FLAGS-HISTORY.md
- Which tuned program implements plan operation X -> docs/EXECUTION-SURFACES.md (generated from crates/memra-gguf/src/op_registry.rs)
- Hopper (sm_90a) lane evidence -> ARCHITECTURE-H100.md
- Hy3 expert spill on a 24 GB card -> docs/HY3-SPILL.md
- Speech packs and audio routes -> docs/SPEECH.md; ASR plan -> ASR-MODALITY-PLAN.md; ASR lane status -> research/asr-modality-20260909/STATUS.md

## Models

- Which models run, their pack state and measured configurations -> docs/MODELS.md plus docs/models/ (one card per model)
- Support states and their rules -> ARCHITECTURE.md, AGENTS.md ("Model onboarding"); per-pack records -> docs/support-records.toml
- Bringing up a new model, pinned artifact to support record -> docs/ONBOARDING.md (memra model CLI ladder, gates, close)
- Tokenizer normalization, added-token order, HF-to-GGUF tokenizer programs -> docs/TOKENIZER-PROGRAM.md
- How each model's draft is built (ranks, trim, quantization) -> docs/DRAFT-REGIME.md
- SFT trace generation tooling (training data, not engine docs) -> research/cx-sft-20260808/README.md

## Serving

- Install paths -> docs/INSTALLATION.md; copy-paste run commands per model and card -> docs/COOKBOOK.md
- Operating memra-server (models, admission, caching, drafters, supervision) -> docs/SERVING.md
- Request and response schemas per route -> docs/API-SURFACES.md
- Card-specific truth -> docs/rigs/; workload guides -> docs/workloads/

## Gates, CI and releases

- Which gate to run, fast-gate versus full battery, the gate catalog -> docs/TESTING.md
- Generic spill and tiered-KV gates -> docs/TESTING.md (Generic spill / tiered KV); docs/decisions/GENERIC-SPILL-INTERFACE-V1.md; research/spill-lead-20260919/
- Which CPU suites a change selects, and why others are omitted -> docs/VALIDATION-SELECTION.md
- Hosted CI routing, exact-commit GPU dispatch, build transfer -> docs/CI.md
- Per-cell push evidence, the release qualification record and what a tag record covers (tools/release-roster.tsv) -> docs/RELEASE-QUALIFICATION.md
- How to cut a release: version claim, release commit, records, tag -> docs/RELEASING.md

## Performance and evidence

- Published boards and measurement doctrine -> docs/PERFORMANCE.md (generated from research/tune-data/current-board.json; edit the board, never the tables)
- Measurement protocol -> research/benchmarks.md; frozen competitor baselines -> docs/COMPETITOR-SETUP.md
- What a past research lane concluded -> research/INDEX.md (one verdict line per lane)

## Decisions, rules and records

- Why X was chosen or rejected, with the settling measurement -> docs/decisions/README.md (index with verdicts)
- Project rules: branches, flags doctrine, evidence, public boundary, releases -> AGENTS.md (it also names the GPU lesson corpus)
- External contributions -> CONTRIBUTING.md; vulnerability reports -> SECURITY.md
- Records moved out of a current doc -> docs/history/<DOC>.md; superseded plans and old release notes -> docs/archive/
