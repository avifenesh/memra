# v0.73 changelog draft

Frozen target: `cbe25b75e95f9aed8863771b625e12c35b016286`

Release range: `v0.72.0..cbe25b75`

## What lands in v0.73

| lane | shipped truth | default / rollback | merge or evidence |
|---|---|---|---|
| PP-2 prefill pipelining | Concurrent stage-owned host walkers overlap stage 0 of chunk N+1 with stage 1 of chunk N. The final ungrouped pp4096 rate is 417.6 tok/s versus the prior 266.1 tok/s Lever-B baseline (N=5); the 200-prime soak recorded zero divergence or faults. | Naked PP-2 auto geometry pipelines; `MEMRA_PRIME_PIPE=0` keeps the same boundaries and serializes the stage walks. | `d240fc52`; `research/pipeprime-20260808/` |
| Lever C grouped prefill | Step35 host-sigmoid routes are grouped without entering illegal softmax/uniform paths. Box2 gains are +53.3% / +58.9% / +63.4%, but the local KAT transfer cell loses 75.3%. | **Opt-in only** through `MEMRA_MOE_GROUPED=1`; the 5090 default-flip gate keeps naked commands on the established path. | `7c49e22d`, opt-in fix `5961e392`; `research/leverC-20260808/` |
| Step35 primebatch | Complete concurrent fresh prompts share Step35 weight streams across PP stages while attention, KV state, and absolute request positions remain isolated. B=2/B=4 T=520 improves 2.5%/2.3% (N=5). | On for eligible fresh prompts; `MEMRA_STEP35_PRIME_BATCH=0` restores the named refusal and serialized fallback. Carried caches remain fallback-only. | `3067b6ad`; `research/primebatch-20260808/` |
| Spec placement policy | Sharded cross-device PP-2 selects plain batched decode because spec loses every measured q9 and Step35 c=1/2/4 cell; single-card retains the low-concurrency spec gate. | PP-2 LOW=0/HIGH=1; single-card LOW=2/HIGH=4. `MEMRA_SPEC_GATE=0` remains the always-spec rollback and crash-gate arm. | `7df5f4d3`; `research/specplace-20260808/` |
| Prefix fanout dedup and pinning | Same-window cold requests in one exact model/tenant/salt pool compute a common prefix once, deep-copy the snapshot, receive exact cached-token credit, and lease the entry until retirement. N=8 p50 TTFT moves 22.263 s to 3.852 s. | On; `MEMRA_PREFIX_DEDUP=0` restores independent cold primes. Cross-tenant and cross-salt grouping is forbidden and gated. | `01a8cc59`; `research/prefixdedup-20260808/` |
| Dynamic PP-2 microchunks | Naked auto geometry retains the fixed chunk count, shortens fill, and shrinks the drain tail. The measured gain is +1.4% / +0.3% / flat at pp512/2048/4096 (N=5); no long-prompt material-win claim. | `dynamic`; `MEMRA_PRIME_CHUNK_SCHED=fixed` restores equal-token ranges. An explicit `MEMRA_PRIME_CHUNK` remains fixed and authoritative. | `ed438fc2`; `research/microchunk-20260808/` |
| TTFT fixes | Per-request phase tracing now measures completion routes, ignores SSE keepalives, and shows only 6–10 ms outside prime. A sole fresh request widens its outer prefill call, moving 4k TTFT 7.118 s to 5.992 s while short TTFT stays 0.589 s. | Widening is automatic only for a sole fresh request; an explicit `MEMRA_PREFILL_TICK` is authoritative. `MEMRA_TTFT_TRACE=1` is diagnostic only. | `ed1550f8`; `research/ttft-20260808/` |
| Architecture onboarding kit | Per-layer geometry is centralized for migrated Qwen35/Step35 paths; `tools/generate-arch-gates.py` renders chunk/tick/B>1 gate scaffolds and registry rows; `docs/ONBOARDING.md` is the canonical artifact-to-green runbook. | Generator never edits canonical registries and refuses unsafe/shadowed contracts. Architecture semantics still require explicit implementation and target-rig gates. | `ebf2ea90`; `research/archkit-20260808/` |
| Serve-ready receipt | The explicit Step trial config passes every bar: 0.595 s short TTFT, 6.052 s 4k TTFT, 12.2 ms 4k cache-hit TTFT, 36.5 tok/s per stream at c=4, 124/124 ten-minute replay requests, zero 5xx/sheds. | Evidence-only declaration at `MEMRA_MOE_GROUPED=1` with the cache budget sized; it does not promote the grouped default. `MEMRA_PREFIX_CACHE_MB=256` cannot hold a 343 MB 4k entry. | `d43f9e27`; `research/serve-ready-20260808/` |
| Request-conditioned K table | Each request owns its K decision: sharded PP-2 or losing concurrency cells choose K=0, cached-long chooses K=2, and other eligible cold requests choose K=3. Mixed-workload throughput is neutral (-0.22% aggregate). | Unset uses the measured table; any non-negative `MEMRA_SPEC_K`, including 0, is an operator pin. | `86ae193c`; `research/kpolicy-20260808/` |
| Concurrent-prefill verdict | Four different-prefix 4k primes saturate the pair at 580.5 tok/s versus the 674 tok/s one-call solo class. No concurrent-prime scheduler or serving-policy change lands; queueing is refuted as a path to 3K tok/s. | No new production path. `MEMRA_TICK_TRACE=1` is debug-only timing/accounting. Scale with another pair or a new compute mechanism. | `cbe25b75`; `research/concprefill-20260808/` |
| RunPod privacy operations rule | Serving boxes must never set `MEMRA_CONFIDENCE_TRACE` or `MEMRA_DEBUG_SPEC`: they expose decodable prompt/completion token IDs through unrotated files or stderr. | Operational prohibition; the code does not enforce it. | `d8f24db4`; `deploy/runpod/API-USAGE.md` |

## `tools/changelog.sh` floor

Command:

```bash
bash tools/changelog.sh v0.72.0 cbe25b75
```

The range contains 130 commits: 14 merges and 116 non-merges. The script includes 67 non-merge
subjects (9 performance, 13 features, 9 fixes, 0 configuration, 9 documentation, 27 other) and
drops 49 `data:` / `chore:` / `wip:` / `probe:` subjects. SHA-256 of the exact stdout:
`53c5797763e6fde44b50a65c70bd00d0aa48ed0e29216bac30214c8e6629dc54`.

The generator's entries, in exact order and wording, are reproduced below under nested headings:

### Performance

- widen solo fresh prefill
- record grouped prefill win
- batch prime gate and output projections
- default sharded PP-2 to plain decode
- make measured microbatch geometry default
- add microchunk geometry sweep
- add 4k serve TTFT driver
- add three-shape box2 driver
- add interleaved pipeline probe

### Features

- apply measured request table
- choose speculative depth per request
- dedup same-window fanout
- add dynamic microchunk schedule
- pin entries for inflight requests
- generate architecture gate scaffolds
- trace per-request TTFT phases
- batch fresh primes across PP stages
- group step35 prefill experts
- drive PP stages on concurrent host walkers
- overlap adjacent prime chunks on PP-2
- prewarm and alternate boundary slots
- add pipeline seam and explicit chunk base

### Fixes

- require idle perf handoff
- keep grouped prefill opt-in after 5090 gate
- stop battery on first red
- exclude SSE keepalives from TTFT
- trace completion routes only
- restore the Drop impl's closing brace lost in the pp.rs counter union
- scope historical perf comparisons
- preserve clamped grouped arithmetic
- preserve step35 grouped arithmetic

### Documentation

- serving boxes never set MEMRA_CONFIDENCE_TRACE / MEMRA_DEBUG_SPEC — the body-leak doors the privacy audit found
- define request-conditioned K policy
- report fanout dedup results
- register dynamic schedule contract
- consolidate architecture onboarding runbook
- record fanout dedup design
- audit step35 onboarding surface
- clean progress formatting
- register pipeline ordering contract

### Other

- test(kpolicy): run repository acceptance gate
- test(kpolicy): size cache for q27 receipt
- test(kpolicy): add mixed workload comparison
- test(kpolicy): add box1 policy battery
- test(concprefill): add target-rig battery
- test(kpolicy): use OpenAI usage surface
- test(kpolicy): add box1 K matrix harness
- test(serve): force affinity smoke through spec gate
- test(prefix-cache): compose box1 gate battery
- test(prefix-cache): compose receipt under held lock
- test(prefix-cache): add fanout TTFT receipt
- test(pipeprime): gate dynamic microchunk geometry
- test(prefix-cache): register simultaneous fanout gate
- refactor(archkit): centralize migrated geometry
- test(serve): parameterize TTFT control arms
- test(serve): add TTFT anatomy harness
- test(step35): emit paired prime benchmark runs
- test(moe): emit grouped oracle verdicts
- test(leverc): add box2 gate and perf drivers
- test(step35): register batched-prime gate red
- test(pipeprime): register auto geometry in fast gate
- test(pipeprime): gate naked auto microchunks
- refactor(pipeprime): shard prime slab locks by device
- test(pipeprime): add 200-prime soak driver
- test(pipeprime): add exactness soak mode
- test(pipeprime): add target acceptance battery
- test(pipeprime): gate serial and pipelined prime schedules

Boards + reproduction artifacts: https://huggingface.co/Avifenesh/memra-bench · full experiment
log in `research/tune-data/`.

## Interpretation notes

- The generator is a floor, not the full release narrative. It deliberately drops merge bodies
  and `data:` receipts, so the serve-ready declaration and concurrent-prefill refutation must be
  added manually to release notes without pretending they are new production mechanisms.
- Lever C must say **opt-in**. Quoting the Box2 win without the local 5090 rejection would invert
  the merged default.
- Dynamic microchunks must not claim a material pp4096 win; the long-prompt N=5 result is flat.
- Step serve-ready numbers are explicit-trial-config receipts, not a tracked competitor-board row
  and not authorization to add Step to the generated supported-model table.
