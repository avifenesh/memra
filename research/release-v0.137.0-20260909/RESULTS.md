# v0.137.0 release preparation

PREPARED CANDIDATE ONLY. No tag or deployment is authorized by this record.

Base main: `6285210078609c7e11aa23ae070ad581c184c153`. Required member #325: `dcfeab7c7`. Conditional #378 remains excluded pending the orchestrator decision and reviewed merge. The source already includes #377; it is inherited, although it is not a plain TP-2 cutover dependency.

## Battery scope

The release protocol requires the complete `tools/release-roster.tsv`: Ornith (own, never skippable), Qwen (vendor, required when present; absence needs an explicit waiver), kernel-check, calibrated argmax margins and run-spec K=1..8. The GLM PP-1 cells are additional checks, not a replacement roster. See [RELEASING.md](../../docs/RELEASING.md) and [roster](../../tools/release-roster.tsv).

[v0.132.0 battery receipt](https://github.com/avifenesh/memra/pull/331#issuecomment-5574891784) ran the full roster at `dcc82af95` on one visible B200: kernel-check 95 cells, 22 skips; Ornith flips=1 bad=0, Qwen flips=0 bad=0; both K=1..8 identical. It was not a GLM-only waiver. The privately deployed artifact was derived from the tag separately.

A single B200 can run that roster if its pinned GGUFs are staged, GLM PP-1 plain and DFlash2 shape checks, CPU suites, applicable single-device kernel gates, fmt and clippy. This assigned development box has GLM target and DFlash2 weights but neither roster GGUF. Required missing weights remain a blocker. No weights are copied from production.

TP-2 served shape, host-image two-rank digest/recycle gates, released-composition host retention and cap1 qualification need two devices. They are pending the orchestrator's pair window and are not run here. PP-1 output is not TP-2 qualification. The sm_120a qwen/ornith/glm53tx deployment stacks are unchanged by a future sm_100a-only pin move. The protocol does not require rerunning every fleet stack on every release, but does require the full roster and architecture-appropriate qualification for any stack that deploys. B200 roster evidence cannot qualify new sm_120a deployment behavior. Hosted CI still builds and checks both supported compile architectures.

## Current validation

Single-box build and gates in progress. Exact source, binary hashes, cell times and log tails will be recorded before handoff. All Cargo and GPU work runs on the authorized non-serving B200, with GPU cells serialized by `/tmp/memra-gpu.lock`. No rig gates and no serving-pair access.

## Release contents and doors

- #325 preserves model-owned host images, repairs demotion/LRU behavior, reserves a startup portable pinned arena, and rejects uninitialized fresh/recycled lease reads. `MEMRA_GLM5_TP_KV_HOST` remains OFF, decide-by 2026-09-21. `MEMRA_KV_HOST_MB` remains 0; `MEMRA_KV_HOST_TENANT_PCT` remains 50, with 100 selecting the existing single-tenant global-LRU contract. The generic budget row now distinguishes the opt-in arena from the old clamp path.
- #377 extends retained-prefix admission to validated external DFlash restores; failed carrier restoration refuses instead of silently doing a cold prime.
- Inherited DSV4 work after the preceding tag includes #374: `MEMRA_DSV4_REPLAY_CADENCE` and `MEMRA_DSV4_DENSE_EXACT_TAIL` default ON within their documented admitted paths; exact `0` selects rollback, decide-by 2026-09-22 for seam review. #376/#382 add profiling/gate work. It supplies no GLM performance or support evidence. Full commit and flag census is retained with the receipts.
- #378 is not included. Its prepared alternate is [OPTIONAL-378.md](OPTIONAL-378.md). Do not arm its device sampler from draft evidence.

No published performance number changes. Publicity: skipped, maintenance release.

## Stop boundary

Draft release PR only. The orchestrator decides #378, then the release branch must be rebased and its final composition qualified before merge/tag. Update the README and this ledger from candidate status to the final measured truth before tagging. Preserve the version claim.
