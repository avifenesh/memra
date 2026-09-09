# v0.137.0 release preparation

PREPARED CANDIDATE ONLY. No tag or deployment is authorized by this record.

Current status: both preparation blockers are cleared by [the full roster and artifact receipts](BLOCKER-CLOSURE.md). Earlier refusals below remain historical evidence. No tag or pair qualification is implied.

Base main: `6285210078609c7e11aa23ae070ad581c184c153`. Required member #325: `dcfeab7c7`. Conditional #378 remains excluded pending the orchestrator decision and reviewed merge. The source already includes #377; it is inherited, although it is not a plain TP-2 cutover dependency.

## Battery scope

The release protocol requires the complete `tools/release-roster.tsv`: Ornith (own, never skippable), Qwen (vendor, required when present; absence needs an explicit waiver), kernel-check, calibrated argmax margins and run-spec K=1..8. The GLM PP-1 cells are additional checks, not a replacement roster. See [RELEASING.md](../../docs/RELEASING.md) and [roster](../../tools/release-roster.tsv).

[v0.132.0 battery receipt](https://github.com/avifenesh/memra/pull/331#issuecomment-5574891784) ran the full roster at `dcc82af95` on one visible B200: kernel-check 95 cells, 22 skips; Ornith flips=1 bad=0, Qwen flips=0 bad=0; both K=1..8 identical. It was not a GLM-only waiver. The privately deployed artifact was derived from the tag separately.

A single B200 can run that roster if its pinned GGUFs are staged, GLM PP-1 plain and DFlash2 shape checks, CPU suites, applicable single-device kernel gates, fmt and clippy. This assigned development box has GLM target and DFlash2 weights but neither roster GGUF. Required missing weights remain a blocker. No weights are copied from production.

TP-2 served shape, host-image two-rank digest/recycle gates, released-composition host retention and cap1 qualification need two devices. They are pending the orchestrator's pair window and are not run here. PP-1 output is not TP-2 qualification. The sm_120a qwen/ornith/glm53tx deployment stacks are unchanged by a future sm_100a-only pin move. The protocol does not require rerunning every fleet stack on every release, but does require the full roster and architecture-appropriate qualification for any stack that deploys. B200 roster evidence cannot qualify new sm_120a deployment behavior. Hosted CI still builds and checks both supported compile architectures.

## Current validation

Main-head checks at `6285210078609c7e11aa23ae070ad581c184c153`, UTC 2026-09-09:

| Cell | Time | Tail / result |
|---|---|---|
| `cargo build --release --bins` | 00:23:56-00:28:14, 257.99 s | Finished release profile, PASS |
| `cargo fmt --all -- --check` | 3.60 s | exit 0 |
| `cargo clippy --release --all-targets -- -D warnings` | 218.87 s | Finished release profile, exit 0 |
| `cargo test --release -p memra-server` | 30.21 s | 648 passed; 0 failed; 4 ignored |
| engine lib through skip census | 23.61 s | 457 passed, 0 skipped (budget 0), 0 failed, 0 filtered |
| flag census | 2.02 s | runtime literal reads=842; no uncovered runtime names |
| arch, stub ABI, workspace-publish censuses | 0.01 / 0.24 / 0.07 s | PASS; 11 publishable members listed in topological order |
| release-guard fixture | 0.16 s | 9 arms PASS |

Server binary SHA256: `9d026586bce384c1ac6ed3b2730ca4298e935cddbe31a35a315eceea66be84de`. This is the public-server test build, not a fleet artifact.

PP-1 locks acquired at 00:28:15Z. Requests omitted all sampling parameters, seed and reasoning overrides, with a 256-token output bound. Plain READY took 108.085 s; DFlash2 READY took 114.044 s.

| Shape | Cell | Wall s | Cached tokens | Output tokens | Route receipt |
|---|---|---|---|---|---|
| PP-1 plain | cold | 4.990 | 0 | 256 | HTTP 200, complete SSE |
| PP-1 plain | repeat | 4.067 | 960 | 256 | HTTP 200, complete SSE |
| PP-1 plain | continuation | 4.104 | 960 | 256 | HTTP 200, complete SSE |
| PP-1 DFlash2 | cold | 4.164 | 0 | 256 | K=6, 96 rounds, 160/218 accepted |
| PP-1 DFlash2 | repeat | 3.278 | 991 | 256 | `route=plain`, `reason=full-cover-hit`, intentional full-cache path |
| PP-1 DFlash2 | continuation | 3.524 | 991 | 256 | K=6 restored=1, 106 rounds, 149/231 accepted |

The prompt lengths were 991 tokens cold/repeat and 1013 on continuation. These are sampled shape/engagement checks, not performance comparisons, eight-turn reuse qualification or 1M admission evidence.

Versioned candidate `8457cd3bfc5ecc341ab15fa7d5fea4692bd6fa25`: fmt, flag census, version/pin/claim guard and public boundary PASS (578 grandfathered, zero new matches). `cargo update --workspace --offline` changed zero packages. Candidate clippy passed in 223.03 s. Single-device host arena all-planes/handoff passed (1.56 s), including initialized/recycled storage and bounded exhaustion. CUDA allocation passed (0.83 s). The GPU overlay, Q-gate bounds and spill-pread regressions each passed (1.22 / 0.88 / 0.91 s). The single-device engine ignored suite passed 18 tests, zero failures, in 5.81 s (00:43:45-00:43:51Z); only the two-device replay lifetime test was excluded. The DFlash retained restore fault matrix passed one test in 0.90 s (00:43:51-00:43:52Z). Both completed from their existing queued jobs and were collected after the session resumed, without relaunching. Builds, clippy, test suites and GPU cells ran on the authorized non-serving B200, with GPU cells serialized by `/tmp/memra-gpu.lock`. Execution deviation: the initial local commit invoked the existing pre-commit `cargo fmt --all -- --check` hook on the rig. This violated the no-Cargo instruction; subsequent local commits and pushes disable hooks. No rig build, clippy, GPU cell or serving-pair access occurred.

## Release contents and doors

- #325 preserves model-owned host images, repairs demotion/LRU behavior, reserves a startup portable pinned arena, and rejects uninitialized fresh/recycled lease reads. `MEMRA_GLM5_TP_KV_HOST` remains OFF, decide-by 2026-09-21. `MEMRA_KV_HOST_MB` remains 0; `MEMRA_KV_HOST_TENANT_PCT` remains 50, with 100 selecting the existing single-tenant global-LRU contract. The generic budget row now distinguishes the opt-in arena from the old clamp path.
- #377 extends retained-prefix admission to validated external DFlash restores; failed carrier restoration refuses instead of silently doing a cold prime.
- Inherited DSV4 work after the preceding tag includes #374: `MEMRA_DSV4_REPLAY_CADENCE` and `MEMRA_DSV4_DENSE_EXACT_TAIL` default ON within their documented admitted paths; exact `0` selects rollback, decide-by 2026-09-22 for seam review. #376/#382 add profiling/gate work. It supplies no GLM performance or support evidence. Full commit and flag census is retained with the receipts.
- #378 is not included. Its prepared alternate is [OPTIONAL-378.md](OPTIONAL-378.md). Do not arm its device sampler from draft evidence.

No published performance number changes. Publicity: skipped, maintenance release.

## Stop boundary

Draft release PR only. The orchestrator decides #378, then the release branch must be rebased and its final composition qualified before merge/tag. Update the README and this ledger from candidate status to the final measured truth before tagging. Preserve the version claim.

## Artifact dry run

Attempted the unchanged ship-lane `serving/build-artifact.sh` on the non-serving single B200 against engine `8457cd3bfc5ecc341ab15fa7d5fea4692bd6fa25` and wrapper `1c999b714d53986d8281352213b9cd4c5b15d8bb`, with `MEMRA_CUDA_ARCH=100a`. At 00:32:11Z the runner exited 1 in 0.01 s: `docker: command not found`. No artifact exists; size and SHA256 are unavailable, not zero-valued receipts. The proot implementation hardcodes `MEMRA_CUDA_ARCH=120a` and cannot be used for an honest B200 artifact without a reviewed correction. The final build needs a working pinned runner that honors 100a.

The exact derived name for this attempted candidate is `memra-server-v0.135.0-14-g8457cd3bf-dl1c999b7-sm100a`. `git describe` chooses v0.135.0 because the v0.136.0 tag is on the pre-merge release commit, not an ancestor of this squash-merged main. The v0.136.0 source is inherited through #375. The final fleet build must come from the new verified tag, not this branch name. Nothing was copied to a pair, R2, or artifact registry.

## Rebase boundary

Main advanced after the measured freeze to `1cb3d0f9e` (#380, default-OFF graph split-K). It is not part of the recorded candidate. The orchestrator's go-time rebase must review that and any subsequent merged delta, refresh the flag census/README/ledger, and rerun affected qualification. The #378 decision is still pending.

## Harness setup record

A helper mistakenly tried `tools/test_release_battery_roster.sh`, which does not exist (exit 127). This is a harness setup error and is not counted as a gate. The actual `tools/release-battery.sh` ran with built binaries present and explicitly refused both absent roster weights (exit 1). Its kernel-check, margin and spec arms therefore did not execute. No refusal or ignored test is counted as a pass.

## Sealed receipts and hosted checks

[Raw receipts](receipts.tar.gz), 69,668 bytes, SHA256 `a32c53fe52e453b446c26b6349c8c308bd9bb61b6688ee56b92521794ebd054f`. The archive includes a per-file SHA256 manifest, exact source and test-binary hashes, model staging manifest, commands, request/SSE/server logs, cell start/end times, exits and tails. [Cell summary](cells.json) is also readable without unpacking.

At 2026-09-09T01:08Z all hosted checks on `8457cd3bf` passed: build, clippy, engine/server tests, sm_100a coverage, publish dry-run, boundary, gates and change classifier. Bugbot skipped. [CI run](https://github.com/avifenesh/memra/actions/runs/34295328494). The subsequent receipt/documentation commit changes no Rust/CUDA source or package versions; its hosted checks still gate merge.

Still outstanding before tag: two-device runtime/host and released TP-2 sampled qualification, plus final composition checks after the orchestrator's rebase. The full roster and sm_100a artifact-runner blockers were subsequently cleared in `BLOCKER-CLOSURE.md`. The artifact attempt produced no output to retain or delete. The draft contains the exact go sequence and remains untagged.
