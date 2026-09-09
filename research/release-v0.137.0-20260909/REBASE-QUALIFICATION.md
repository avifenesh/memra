# Rebased v0.137.0 qualification

Base main: `16071ff480fdea92c38f2c825c564562fff25710`. No tag is authorized by this record.

#379 adds the owned speculative-prime fairness walker, `MEMRA_PRIME_YIELD` OFF (unset/`0`), `1` enables bounded peer service; decide-by 2026-09-22. Receipts: `research/prefill-fairness-20260908/{RESULTS,COMPOSITION,CHUNKS,SEAM}.md`. GLM5 adapter qualification remains separate.

#392 changes `MEMRA_DSV4_MOE_M1_SPLITK` to ON (unset = `graph`); explicit `0` restores sktail in a fresh process with fresh graphs, seam review 2026-09-22. Owner-accepted numeric drift and performance receipts: [Darklanes #520](https://github.com/avifenesh/darklanes/pull/520); actual unset/0 engagement: [Memra #392](https://github.com/avifenesh/memra/pull/392), namespace `splitk-default-on-b589df9-gate-r1`, binary `ae91d10f4404759b6bd0b0c102eaa074aec27aa5b86a8b5c0a772c96de5201ee`. This supplies no GLM performance claim.

## Fleet naming trap and tag gate

`git describe --tags origin/main` before release gives `v0.135.0-18-g16071ff48`. The v0.136.0 tag resolves to `55d83a7cf3948d6d173d28e3e1fd18008fdeb9bf`, while the release was squash-merged to main as `72aa777c3763a21281fb9c1c55b2499f2c14d1ee`. The tagged branch commit is not on main's first-parent history and is not an ancestor of current main. Consequently git describe cannot select that tag from main. Never compensate with a hand-written fleet artifact version.

After the orchestrator go and final gates, merge #391 first, fetch main, obtain #391's `mergeCommit.oid` from GitHub, and verify that it is the actual current `origin/main` commit and appears in its first-parent history. Tag v0.137.0 on that merged commit only, never on the release branch. Push the tag, fetch tags/main, then require `git describe --tags origin/main` to equal exactly `v0.137.0` before building. If main has advanced, stop the build and resolve the release boundary with the orchestrator; do not move a published tag or label an untagged build as the release.

## Full battery PASS

Measured head: `72d424979c0e1a6c499a4017a2a36f64e6215730`, rebased onto main `16071ff48` and including #379/#392. The receipt-only follow-up changes no Rust, CUDA or Cargo source. The existing nohup job survived the Relay restart; it was inspected and collected, not relaunched.

Pinned GGUFs were rehashed on the box before the build. `cargo build --release --bins` passed in 253.13 s. `cargo fmt --all -- --check`, diff check and flags census passed; the census found 844 runtime literal reads and zero uncovered names.

The full unchanged roster ran under `/tmp/memra-gpu.lock` from 2026-09-09T02:02:33Z to 02:04:09Z, 96.37 s, exit 0:

```text
kernel-check PASS: ALL GREEN (95 cells, 22 skipped)
Ornith argmax-margin PASS: SUMMARY flips=1 bad=0
Ornith run-spec PASS: K=1..8 self-consistency, identical to plain target
Qwen argmax-margin PASS: SUMMARY flips=0 bad=0
Qwen run-spec PASS: K=1..8 self-consistency, identical to plain target
```

Server SHA256: `4e722f2e92655f3926573088b352b68d9166f5b608402d317b54845c7bb79b60`.
[Raw receipt archive](rebase-receipts.tar.gz): 4,649 bytes, SHA256 `9f224e0ac8294124324a51a3ae2c0c63461d430fd56cb5a589f02247eaa34c33`. It contains exact source, model and binary hashes, commands, build/gate logs, timestamps, exit codes and a per-file checksum manifest.

## Builder merge and handoff

Darklanes #526 was self-reviewed against current main with all exact-head checks green and no open threads, then squash-merged as `2f72fff36898bf61c661239475eb3273d75bdcae` at 2026-09-09T01:57:47Z. Its branch/worktree were removed. Both local root checkouts remain on main and were synced, including submodules, while preserving unrelated dirty work.

No tag or serving-pair access occurred. The staged GGUFs and installed PRoot remain for the final tag gate. Doc commits use --no-verify; local hooks stay disabled under the no-rig-gates instruction. Hosted checks must be green on the receipt head before handoff; the live exact-head status is recorded in PR #391. The pending #378 decision and final-composition/pair gates remain with the orchestrator.

