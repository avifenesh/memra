# Served decode runs one numeric program at every batch width (2026-08-13)

**Status:** adopted 2026-08-13 for every architecture. A lone session's tick runs the same generic
batched body as a session sharing a batch, so a request that gains or loses peers never changes
floating-point program mid-stream. The eager B=1 fusion trunk stays as an explicit fixed-solo
measurement door (`MEMRA_SERVE_B1FAST=1`); the GraphSession serve door (`MEMRA_SERVE_GS`) was
removed in the 2026-09-05 door sweep ([docs/FLAGS.md](../FLAGS.md), "Removed doors"). Code: `b1_fast_on`,
`b1_fast_plan_eligible` and the `b_n == 1` dispatch in `crates/memra-engine/src/decode_batch.rs`.
Receipts: [`research/iso-gap-20260807/`](../../research/iso-gap-20260807/PROGRESS.md),
[`research/q35bug-20260812/`](../../research/q35bug-20260812/RESULTS.md),
[`research/eosclass-20260813/`](../../research/eosclass-20260813/RESULTS.md),
[`research/servepath-p2-20260805/`](../../research/servepath-p2-20260805/RESULTS.md),
[`research/coldfix-20260812/`](../../research/coldfix-20260812/RESULTS.md).

The rationale below moved out of the serving doc on 2026-11-12. The old text is kept in
[docs/archive/SERVING-2026-11-12.md](../archive/SERVING-2026-11-12.md#the-isolation-contract)
("The isolation contract"). The repo rule it instantiates is in `AGENTS.md`, "One numeric program
per request".

## Question

Greedy serving promises isolated-identical output: a request's tokens are the same whether it
arrives alone or inside a full batch. In August 2026 a lone session could run one of two faster
programs: the m=1 eager fusion trunk (`MEMRA_SERVE_B1FAST`, `decode_layers_eager`, the trunk
`decode_step_h` uses) or GraphSession replay (`MEMRA_SERVE_GS`). When a second session arrived
mid-stream, the first session's ticks moved to the generic batched body, which composes the same
math in a different floating-point order. Is that transition a legal default?

## Measurement

**Attribution, 2026-08-07 (lane/iso-gap, task #91).** The trigger was a spec-gate receipt: on the
RTX 5090 Laptop GPU with Qwen3.5-9B NVFP4+MTP, a 768-token greedy request diverged from its own
solo reference at byte 1347 (about token 331) when it shared batched decode with sessions at
staggered depths, and at byte 2379 on a second run (`research/spec-gate-20260806/logs/exact/`,
arm `REF_LOAD`). The serve gate missed it: it runs 16 prompts at 96 tokens, all arriving together.
The two suspected mechanisms were cleared:

- Depth staggering moves nothing. At the engine tick, with the program held fixed, a co-resident
  session at any other depth, across an `fa_split_keys` ladder-rung boundary, at B=2..8, three
  rungs and 300-step horizons, changes zero bits of a session's logits (`iso-gap-probe`, 8 arms
  and a canary). The fast-gate `isogap` arm pins this.
- The carrier was the solo-to-batched program flip at the co-residence boundary. With
  `B1FAST=0 GS=0` solo and loaded streams were byte-identical; the moving divergence byte was
  arrival-tick jitter.

**Qwen3.6-35B-A3B, 2026-08-12.** The same transition selected the real EOS id at completion
tokens 15, 17 and 25 in the mixed c=2 cell. Keeping `Qwen35Moe` on the batched trunk at B=1 made
the cell one hash, with 100/100 full-length requests on each wire surface
([`research/q35bug-20260812/RESULTS.md`](../../research/q35bug-20260812/RESULTS.md)).

**Dense 27B Qwen, 2026-08-13 (the class-level verdict).** A frozen harness seeded four cache
namespaces with the same 4,860-token prompt, then started one restored target before three
restored peers, varying only the peer-arrival delay over 0-600 ms in 25 ms steps (25 target
cells, 105/105 requests completed, no eviction, defer, OOM park or protocol failure):

| arm | target hashes | early EOS |
|---|---:|---:|
| pre-fix default | 5 | 2 (the historical 11-token EOS at 50 and 225 ms) |
| one-class default | 1 | 0 (every target equals the 60-token solo control) |

A same-binary N=5 control that changed only `MEMRA_SERVE_B1FAST=0` gave two restored-target hashes
on the old default and one with the eager trunk off
([`research/eosclass-20260813/RESULTS.md`](../../research/eosclass-20260813/RESULTS.md)).
Step-3.7-Flash's PP-2 load-history divergence was the third case of the same class
([`research/p0iso-20260810/`](../../research/p0iso-20260810/RESULTS.md),
[`research/b1fix-20260810/`](../../research/b1fix-20260810/RESULTS.md)).

## Decision

- B=1 and B>=2 run the generic batched body on every architecture. GraphSession serve replay is
  off, and its worker path was later deleted.
- `MEMRA_SERVE_B1FAST=1` opts an eligible fixed-solo tick into the eager trunk for measurement
  only. Use it only where admission guarantees the request cannot gain a peer. Qwen35-MoE was
  ineligible even under `=1`. The eligibility now lives on the plan: every plan whose trunk has a
  `GatedDeltaNet` operation is ineligible (`b1_fast_plan_eligible`), because spec verify on that
  recurrence runs the batched numeric class, so the door no longer reaches any Qwen3.5-class
  hybrid. The eager dispatch also skips gemma4 E4B, non-generic decode programs, SwiGLU-OAI
  trunks and verify-exact, and the unsplit dispatch skips penalised samplers; under a pipeline
  split a stage-scoped eager path replaces it.
- `decode-batch-gate` config mode pins B=1 against B=N bit identity on the default program. Strict
  mode and PP arm 4 turn the eager trunk on to keep its `decode_step_h` bit identity covered. The
  release battery's `argmax-margin-batch` cell unsets the door so it gates the served program.

## Rejected

- **The eager B=1 trunk as the serving default.** It was the measured c=1 win: on the RTX 5090
  Laptop GPU, order-paired N=5, decode-only `step_p50` +8.33% on Qwen3.5-9B (123.7 to 134.1 tok/s,
  5/5 wins) and +5.19% on a 27B (43.6 to 45.8, 5/5), c=8 flat. It is unsafe for any session that
  can gain a peer, which is every served session
  ([`research/servepath-p2-20260805/RESULTS.md`](../../research/servepath-p2-20260805/RESULTS.md)).
- **GraphSession replay for a lone session.** With the eager trunk in place it was a net loss at
  every measured length out to 1,024 tokens, and it carries the same transition.
- **Another per-model fence.** The eosclass verdict rejected fencing model by model: the repair is
  a global serving policy. The plan-level eager exclusion (Qwen35-MoE then, every
  `GatedDeltaNet` plan now) stays as defense in depth.

## What it costs

The inverse of the historical solo win: about 8.33% (Qwen3.5-9B) and 5.19% (27B) decode-only at
c=1 on the RTX 5090 Laptop GPU, N=5. Concurrency above 1 is unaffected except while a batch drains
to one. Step models recover most of it inside the generic body: a B=1 specialization that drops 90
arithmetic-free device copies per token moved the Step-3.7-Flash (IQ4_XS) c=1 median from 81.472 to
85.041 tok/s (+4.4%, N=5 interleaved) on 2x RTX PRO 6000 Blackwell Server Edition at PP-2,
byte-identical
([`research/eagerpar-20260810/RESULTS.md`](../../research/eagerpar-20260810/RESULTS.md)).

## The same law on the prime side

- **Carried continuation primes.** The v0.81.0 default concat-primed bounded continuation chunks
  of long cold prompts across requests. In the frozen Qwen3.6-35B-A3B mixed c=4 cell that path
  selected EOS at token 26 for both cold misses; serial priming and one complete fresh B=2 prime
  batch both reached 60/60
  ([`research/coldfix-20260812/RESULTS.md`](../../research/coldfix-20260812/RESULTS.md)). A plan
  whose trunk has a routed `MoeMlp` node primes continuation chunks alone
  (`carried_prime_batch_eligible`, `crates/memra-server/src/worker.rs`); whole fresh prompts still
  batch, and dense plans keep continuation batching. The door opens when the prime-batch gate
  covers the real multi-chunk sequence followed by the serving batched-decode trunk.
- **The prefill router.** The batched cuBLASLt prefill router and shared-expert-gate GEMMs were
  m-dependent, so a MoE request's expert selection changed with its co-arrivals (the serve gate
  caught Ornith-35B at 6/16 and KAT at 7/16; both 16/16 after the fix). Prefill now routes through
  decode's m-invariant router and gate kernels, with a bit-identical batched twin
  (`MEMRA_ROUTER_BATCH`) recovering most of the prefill cost
  ([`research/concat-prime-exact-20260802/`](../../research/concat-prime-exact-20260802/),
  [`research/fast-router-20260802/`](../../research/fast-router-20260802/)).

## What would change it

A faster B=1 program that a serving-shape gate proves bit-identical to the batched body across the
co-residence transition may become the default. A fixed-solo measurement alone never does.
