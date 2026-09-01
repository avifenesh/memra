# README rewrite notes — 2026-08-09

## Goal

Make the first reader path answer what memra is, how to run it, which configurations are
supported, and where the full evidence lives. The README now presents memra as a serving engine
with a specialized Rust + CUDA core, rather than as a tuning diary.

## What was cut

- Long serving mechanism narration and historical lane chronology. These details change quickly
  and already have a receipt-backed home in `docs/SERVING.md`.
- The full performance-card embed, deployment-threshold commentary, and repeated board history.
  The README contract is the generated sample table only; the full boards stay in
  `docs/PERFORMANCE.md`.
- Bring-up status for every experimental model and the long known-gaps ledger. First-time readers
  need the generated supported table and the Step PP-2 qualification boundary; the rest belongs
  in `docs/PERFORMANCE.md`.
- Repeated kernel inventory and hardware requirements. The rewrite keeps the deployment-relevant
  targets and links the implementation detail to the architecture ledgers.
- Placeholder `hf:owner/repo` commands presented as runnable examples. The primary path now uses
  an explicit local model path; the verified `hf:` resolver syntax remains documented as an
  optional source form.

## What stayed in the README

- Install, direct generation, server startup, and a streaming chat request.
- Blackwell/H100 posture, GGUF-first and single-model specialization, PP-2, serving capabilities,
  recent contract fixes, design limits, and the high-value documentation map.
- Both generated `PERF-SAMPLES` and `PERF-MODELS` regions unchanged.
