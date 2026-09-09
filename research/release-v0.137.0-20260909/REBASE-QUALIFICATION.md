# Rebased v0.137.0 qualification

Base main: `16071ff480fdea92c38f2c825c564562fff25710`. No tag is authorized by this record.

#379 adds the owned speculative-prime fairness walker, `MEMRA_PRIME_YIELD` OFF (unset/`0`), `1` enables bounded peer service; decide-by 2026-09-22. Receipts: `research/prefill-fairness-20260908/{RESULTS,COMPOSITION,CHUNKS,SEAM}.md`. GLM5 adapter qualification remains separate.

#392 changes `MEMRA_DSV4_MOE_M1_SPLITK` to ON (unset = `graph`); explicit `0` restores sktail in a fresh process with fresh graphs, seam review 2026-09-22. Owner-accepted numeric drift and performance receipts: [Darklanes #520](https://github.com/avifenesh/darklanes/pull/520); actual unset/0 engagement: [Memra #392](https://github.com/avifenesh/memra/pull/392), namespace `splitk-default-on-b589df9-gate-r1`, binary `ae91d10f4404759b6bd0b0c102eaa074aec27aa5b86a8b5c0a772c96de5201ee`. This supplies no GLM performance claim.

## Fleet naming trap and tag gate

`git describe --tags origin/main` before release gives `v0.135.0-18-g16071ff48`. The v0.136.0 tag resolves to `55d83a7cf3948d6d173d28e3e1fd18008fdeb9bf`, while the release was squash-merged to main as `72aa777c3763a21281fb9c1c55b2499f2c14d1ee`. The tagged branch commit is not on main's first-parent history and is not an ancestor of current main. Consequently git describe cannot select that tag from main. Never compensate with a hand-written fleet artifact version.

After the orchestrator go and final gates, merge #391 first, fetch main, obtain #391's `mergeCommit.oid` from GitHub, and verify that it is the actual current `origin/main` commit and appears in its first-parent history. Tag v0.137.0 on that merged commit only, never on the release branch. Push the tag, fetch tags/main, then require `git describe --tags origin/main` to equal exactly `v0.137.0` before building. If main has advanced, stop the build and resolve the release boundary with the orchestrator; do not move a published tag or label an untagged build as the release.

The full battery on this rebased composition is pending in this preparation record until its attached receipt is committed. Existing staged GGUF hashes remain the pinned inputs. All Cargo and GPU cells run on the non-serving single B200, under nohup and the shared GPU lock. Doc commits use --no-verify and local hooks remain disabled. No tag is created in this preparation task.
