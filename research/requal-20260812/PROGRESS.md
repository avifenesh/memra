# cx-requal progress — 2026-08-12

## Scope

Re-run the frozen `research/sellgate-20260812` qualification battery on the
eu-west RTX PRO 6000 pair from merged `main` commit `ac6ef049b`, with Qwen3.6
35B-A3B MoE and Qwen3.6 27B active simultaneously. Re-confirm every original exactness,
latency, cached-token, knee, and headroom cell and add the mixed c=2 x5 q35bug
regression cell for both models.

## Gates

- [x] Read and hash the frozen sellgate workload, reducer, bars, and original results.
- [x] Check the orchestration inbox and coordinate with the cx-gateway soak.
- [x] Acquire `/tmp/memra-gpu.lock` before touching either GPU and retain one lock hold.
- [x] Verify a clean remote checkout at `ac6ef049b` and make a fresh release build.
- [x] Run both models simultaneously through standard exactness.
- [x] Run both models simultaneously through serial-cache exactness.
- [x] Run required base cells, c=4 sold-cap latency, cached-token reconciliation,
      knee/headroom, and mixed c=2 x5 regression cells for both models.
- [x] Retrieve raw logs and receipts frequently and verify their manifests locally.
- [x] Reduce with the frozen sellgate reducer and compare Q27 values with the original.
- [x] Publish per-model SELLABLE/NOT verdicts and both one-pagers if both pass.
- [x] Commit the complete lane; do not merge, tag, push, edit boards, or format.

## Log

- 2026-08-12 — Started from a clean dedicated worktree on `lane/cx-requal` at
  `ac6ef049b8661008c0da91f4747f68f4dabdaa04`.
- 2026-08-12 — Created this progress ledger as the lane's first artifact before
  remote inspection or measurement work.
- 2026-08-12T06:43:34Z — Read `~/.lanectl/inbox/cx-requal.md`; it matches the
  owner brief and carries no additional steering. The live pair was reachable,
  `/tmp/memra-gpu.lock` was free, both cards were P8 at 0 MiB / 0% / 26 C, and
  no compute application was present. `cx-gateway` has not started its soak and
  remains ordered behind this lane.
- 2026-08-12T06:48:08Z — Re-fetched `origin/main` and confirmed it is exactly
  `ac6ef049b8661008c0da91f4747f68f4dabdaa04`. Frozen the original sellgate
  workload, replay, reducer, prefix exactness, pilot, and metadata-hash bytes.
  The requalification runner changes only the required source commit and adds
  an explicit inherited-lock seam around the unchanged gate/campaign bodies.
  A lane wrapper holds one flock continuously across both phases so the queued
  gateway soak cannot enter between them; every workload and bar is unchanged.
- 2026-08-12T06:48:08Z — Live-state drift: the prior `cx-sellgate` scratch tree
  was cleaned. The gateway lane has already restaged the exact pinned Q27 target
  on the same NVMe; Q35 and both drafters will be restored byte-for-byte from
  the original local sources and independently re-hashed before GPU work.
- 2026-08-12T06:50:07Z — Bundled the complete local `main` history at the fixed
  commit (bundle SHA-256 `fda84f58...b12250b`), cloned it into an isolated remote
  checkout, and verified a clean detached HEAD at `ac6ef049b`. The staged
  workload/replay/reducer hashes match the frozen local ledger.
- 2026-08-12T06:51:33Z — Started the release build detached from an absent
  `target/`; it completed at 06:55:48Z with CUDA 13.2 / auto-detected sm_120a and
  a clean checkout. Fresh binary hashes are `3c3b9dcb...a20a` (`kernel-check`),
  `10c0840f...3b96` (`run-gen`), `2dd70158...b76` (`run-spec`), and
  `53b31fc0...761` (`memra-server`). The complete build log is bundled locally.
- 2026-08-12T06:51:33Z — Re-hashed the gateway's Q27 target to the pinned
  `d8d71c7e...2d517` and hard-linked it into the isolated model directory; the
  source and lane path have the same device/inode and exact 15,705,920,064-byte
  length. The other three artifacts are transferring resumably while the GPUs
  remain idle.
- 2026-08-12T07:04:24Z — All four staged artifacts independently matched their
  frozen SHA-256 values and byte lengths: Q27 15,705,920,064 B, Q35
  18,209,036,576 B, Q27 drafter 1,242,867,296 B, and Q35 drafter 944,118,560 B.
  The detached all-phases driver acquired `/tmp/memra-gpu.lock` immediately as
  PID 965642. It keeps that one hold across gates and campaign; `cx-gateway`
  therefore queues until requalification cleanup completes.
- 2026-08-12T07:07:33Z — Physical GPU 0 completed the fresh full kernel battery:
  `ALL GREEN (95 cells, 13 skipped)`, with no FAIL or MISMATCH marker. Its stable
  log plus source/artifact/binary provenance was copied home and checkpointed
  while the identical battery continued on GPU 1.
- 2026-08-12T07:10:39Z — The complete standard exactness battery sealed PASS and
  its 14-file manifest verifies locally. Both physical GPUs report `ALL GREEN
  (95 cells, 13 skipped)`; Q27 and Q35 each pass prefill/decode plus batched-
  prime/tokenwise argmax MATCH; both `run-spec` logs contain exactly eight K=1..8
  self-consistency PASS rows and the overall PASS sentinel.
- 2026-08-12T07:11:03Z — The unchanged dual-server campaign started under the
  still-held outer flock. Standard exactness cleanup left both cards idle before
  the campaign's fresh server boot; serial cache exactness runs before scoring.
- 2026-08-12T07:11:54Z — Both simultaneously resident models passed the frozen
  serial partial/full prefix-cache gate. Each model reconciled exactly 27,702
  cached tokens in client usage, `cached_tokens_in`, and
  `prefix_cache_hit_tokens`, with six hits, six misses, and no failure. Stable
  exactness receipts were copied home and committed while scoring continued.
- 2026-08-12T07:24:45Z — The detached replay is 56/56 scored cells clean with
  zero invalid cells. The explicit mixed c=2 regression matrix is 4/5 clean for
  each model; every completed repetition has exactly 1,200 response completion
  tokens and 1,200 engine `tokens_out`. Growing campaign receipts have been
  mirrored home twice but remain uncommitted until their remote manifest seals.
- 2026-08-12T07:29:33Z — The explicit mixed c=2 x5 regression matrix is complete
  and clean for both Q27 and Q35. All ten model/repetition cells have exactly
  1,200 response completion tokens and 1,200 engine `tokens_out`, with no short
  completion, counter drift, or invalid cell. The full replay is 70/70 clean and
  continues through the remaining base and knee/headroom cells under the same
  uninterrupted flock; another growing raw receipt was copied home.
- 2026-08-12T07:33:12Z — The fixed c=1/2/4/8 base battery completed 80/80 clean
  cells across both models, cold and mixed90 cache modes, and five repetitions.
  The unchanged capacity extension has started at c=12 and is 4/4 clean so far;
  a third growing receipt copy now preserves the complete base sweep locally.
- 2026-08-12T07:36:27Z — The c=12 knee/headroom tier completed 20/20 clean,
  bringing the replay to 100/100 clean cells. The only remaining scored tier is
  c=16; the active raw bundle was mirrored home again before that final tier.
- 2026-08-12T07:42:06Z — c=16 completed clean, but the frozen adaptive stop rule
  differs from the original evidence: Q35 mixed90 median output throughput still
  rises from c=12 to c=16, while Q27 does not. The unchanged harness therefore
  emitted `candidate_width=24, run_candidate=true` and opened c=24. The campaign
  is 126/126 clean; stopping at the original 120 cells would violate the frozen
  knee/headroom rule, so the dynamic extension continues.
- 2026-08-12T07:46:58Z — c=24 completed 20/20 clean. Both models' clean mixed90
  medians at c=24 exceed c=16, so the frozen stop rule emitted
  `candidate_width=32, run_candidate=true`; its first pair passed, for 142/142
  clean cells overall. The active bundle was mirrored home before continuing.
- 2026-08-12T07:54:03Z — c=32 completed 20/20 clean. Both models' clean mixed90
  medians still rise over c=24, opening c=48 under the frozen rule. The campaign
  is 160/160 clean, and the complete c=32 evidence is now mirrored locally.
- 2026-08-12T08:02:35Z — The frozen adaptive rule stopped after c=48: both
  mixed90 medians no longer rise over c=32, so c=64 was not run. The sealed
  campaign is 180/180 clean cells, 90/90 per model, with 2,400 requests per
  model; replay exited zero and the uninterrupted outer flock released only
  after `campaign.ok` and `REQUAL_PIPELINE_PASS` were written.
- 2026-08-12T08:09:01Z — The frozen reducer returns GO: Q27 SELLABLE at c=4
  with a c=12 knee (200% headroom), and Q35 SELLABLE at c=4 with a c=32 knee
  (700% headroom). Both explicit mixed c=2 x5 matrices pass. The per-metric Q27
  comparison found three regressions above 2%, all mixed90 TTFT p50: c=1
  +5.302%, c=8 +2.047%, and c=12 +3.589%; `RESULTS.md` flags each individually.
- 2026-08-12T08:09:01Z — Campaign, correctness, and post-score template-hash
  manifests verify locally. The pinned templates match the original; all JSON
  and JSONL parse; frozen hashes, Python syntax, shell syntax, shellcheck, exact
  gate sentinels, zero drift/defers/OOM parks, and byte-identical report
  regeneration pass. The inbox is unchanged, the remote flock is free, both
  GPUs are idle at 0 MiB / 0% / 27 C, and the lane diff is confined to this
  directory with no runtime, board, README, merge, tag, push, or format change.
