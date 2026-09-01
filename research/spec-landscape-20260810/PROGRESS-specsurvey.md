# cx-specsurvey progress

Status: complete — ready for orchestrator review

- Confirmed dedicated branch `lane/cx-specsurvey` starts from `1592253f` with a clean worktree.
- Read the lane brief, `/home/avifenesh/projects/bw24/CLAUDE.md`, and the full existing
  `research/spec-landscape-20260810/SURVEY.md` in the mandated order.
- Fetched and deep-read current arXiv revisions 2605.30852v3 (SPD), 2605.02960v2
  (MoE-Prefill / AsyncEP), and 2607.12696v1 (EcoSpec), including mechanism, evaluation,
  limitations, training requirements, and venue status.
- Cross-checked the live PP-2 prime and two-session speculative pipeline, worker-mode positioned
  reads, selected-expert prefetch/promotion, projection-granular MoE residency, mixed expert
  layouts, and pruned-expert guards in `hybrid_forward.rs`, `worker.rs`, `spill_pread.rs`,
  `spec.rs`, `gemma_spec.rs`, `moe_cache.rs`, and `model.rs`.
- Folded SPD into the pipeline-scheduling family, AsyncEP beside spill prefetch, and EcoSpec
  beside DraftExpert/tree selection. The verdicts distinguish SPD's true c=1 mechanism from
  memra's prompt-chunk and two-session pipelines, reject full-bank AsyncEP as a direct spill
  graft, and correct the external EcoSpec note: its verifier is lossless, but its expert
  predictor is trained rather than training-free.
- Constraints: documentation research only; no GPU, build, merge, tag, push, formatting, code,
  or performance-board changes.
