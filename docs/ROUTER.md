# Doc router (T1)

One line per destination: question shape -> file (what it holds). This index carries no
content, only addresses; read the target before acting. Paths are relative to the repo root.

## Engine

- Which kernels exist, symbol tables, build/binding model, dead files -> docs/KERNELS.md (audited per-file kernel inventory, pinned to a commit; regenerate rows with the .cu change)
- What does flag X do, defaults, rollback seams -> docs/FLAGS.md (audited MEMRA_* catalog; law: a new MEMRA_* read needs its row in the SAME commit, pre-push enforces, no grandfather list; winners are defaults)
- Component map / execution architecture -> ARCHITECTURE.md; H100 lane evidence ledger -> ARCHITECTURE-H100.md

## Models

- Is model X supported, which path/quant/drafter -> docs/MODELS.md (support matrix; support = model+quant+drafter, never a format) plus docs/models/ (one card per model)
- Support states -> CLAUDE.md, "Model onboarding" section: NativeReference (plan runs in the reference executor, bring-up evidence only), NativeQualified (required checkpoint and serving gates pass, minimum production-admission state), NativeTuned (qualified plus current binary-bound rewrite receipts)
- Bringing up a new model, artifact to gates green -> docs/ONBOARDING.md (ordered phase checklist, fail-closed contracts)

## Speculative decode

- How every model gets its draft, rank/trim/quant laws -> docs/DRAFT-REGIME.md (one draft file per model, zero flags; own-gen ranks per model, byte-verbatim extraction, verdict metric = end-to-end tok/s)

## Gates, releases, performance

- Which gate to run, fast-gate vs full battery, hardware-gate receipts -> docs/TESTING.md
- How to cut a release -> docs/RELEASING.md (version scheme; tools/release-battery.sh over tools/release-roster.tsv; own model REQUIRED, a SKIP renders as refusal)
- Tracked perf boards, measurement doctrine, refutation history -> docs/PERFORMANCE.md (generated from research/tune-data/current-board.json by tools/update-perf-board.py; edit the board, never the tables)
- Measurement protocol -> research/benchmarks.md; competitor baselines are frozen reference points -> docs/COMPETITOR-SETUP.md
- Copy-paste serving configs per model per card -> docs/COOKBOOK.md; serve surface, fleet shape, tools API -> docs/SERVING.md; endpoint schemas -> docs/API-SURFACES.md
- Card-specific truth -> docs/rigs/; workload guides -> docs/workloads/; install paths -> docs/INSTALLATION.md

## Decisions and laws

- Why was X chosen or rejected, with the settling measurement -> docs/decisions/ (index docs/decisions/README.md: SAFETENSORS-DECISION, FORMAT-DECISION, QUANT-GEMM-DECISION, RIG-NATIVE-DECODE, PHASE1-HYBRID, BEST-OF-ALL-WORLDS, VISION-LANE, PUBLIC-BOUNDARY-DETECTION, ORNITH-PAIR-OWNER)
- Project laws (branch isolation, public boundary, flags doctrine, evidence discipline, release rules) -> CLAUDE.md

## Cross-cutting lessons (curated corpus, sibling repo, read-only)

- Tuning knees, measurement laws, model quirk cards, serving lessons, gate craft -> ../darklanes/agent-knowledge/gpu/ (index: ../darklanes/agent-knowledge/gpu/README.md; grep-first ID lines: `rg '^LAW:|^TRAP:|^GATE:|^VERDICT:|^KNEE:|^QUIRK:'`)
- Lane-to-verdict map of past research lanes -> ../darklanes/research/INDEX.md
