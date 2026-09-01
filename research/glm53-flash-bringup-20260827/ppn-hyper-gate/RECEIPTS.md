# mHC trunk under the ppN door — gate receipts (lane/glm53-pp, 2026-08-28)

Lane goal: unblock full expert residency across BOTH cards for GLM-5.3-Flash, which the
decode-attribution lane sized as the keystone step from 20.3 tok/s toward the owner's ~90
tok/s target (base decode, no drafter). 171.2 GB of routed experts against 2x96 GB means
card 0 alone holds at most ~57-66 GB of them, so single-card residency is arithmetically
impossible and the second card is the only route. The pp door is how weights get there, and
all three mHC trunk walks refused it.

## What was refused, and what lifted it

`hybrid_forward.rs`, three sites (`forward_hyper`, `prime_cache_hyper`, `decode_step_hyper`):

> "MEMRA_PP_SHARD is set, but the hyper-connection trunk walk is single-engine: the sharded
> stage handoff is unwired for this residual topology"

Lifted by three ppN twins built on shared layer-range helpers and shared trunk exits, so the
split walk and the unsplit walk run the SAME code and differ only in where the stages run.
Exactly one thing differs from the generic ppN arm: the payload on the wire is the mHC stream
state, `[streams, n_embd]` for decode and `[t, streams, n_embd]` for prime, rather than the
serial trunk's `[n_embd]` / `[t, n_embd]`. `pp.rs` BoundarySlot buffers are lazily sized from
the caller's `n`, so nothing about slot sizing changed.

The deferred-readback (pipelined) arm is still refused: `decode_step_h_ppn_deferred` calls
`refuse_hyper`, and this lane did not change that. The gate prints a NOTE rather than
pretending to cover it.

## The truth chain, stated rather than implied

Split-vs-unsplit is ARM-EQUALITY, not truth (GATE:pin-against-truth): both arms run the same
kernels over the same weights, so an error in the hc arithmetic itself would cancel. The chain
closes only by COMPOSITION:

| half | what anchors it | status on this tree |
|---|---|---|
| the UNSPLIT hc walk vs an independent host executor | `tests/hyper_connections_gpu.rs` vs `memra_reference` | 6/6 PASS |
| the SPLIT walk vs the unsplit walk, bit for bit | `glm5-hyper-ppn-gate` | 10/10 arms PASS |

Neither is sufficient alone. Cite both.

Adjacent glm5_next gates re-run green on this tree (the refactor touched shared trunk code):
`swiglu_preclamp_gpu` 7/7, `glm5_routed_router_gpu` 3/3, `glm5_kpool_indexer_gpu` 11/11,
`glm5_moe_residency_gpu` 2/2.

## GREEN — the knob matrix

Rig 5090 under `flock /tmp/memra-5090.lock`, `NVIDIA_TF32_OVERRIDE=0`. Rig law: exactness
only; no timing number is read out of any of these runs. Driver: `run-ppn-hyper-gate.sh`.
Every arm is BIT-IDENTICAL on all THREE comparison arms (decode-serial, prime-twin,
prefill-twin), greedy tapes match. 30 PASS lines across 10 invocations.

| log | knobs | fence | state placement (Recurrent / LatentKvCache stages) |
|---|---|---|---|
| `10-n2-even.log` | N=2 | [0, 2, 4] | [0, 1] / [0, 1] |
| `11-n2-split1.log` | N=2 `SPLITS=1` | [0, 1, 4] | [0, 1] / [1, 1] |
| `12-n2-split3.log` | N=2 `SPLITS=3` | [0, 3, 4] | [0, 0] / [0, 1] |
| `13-n2-streams0.log` | N=2 `STREAMS=0` | [0, 2, 4] | [0, 1] / [0, 1] |
| `14-n2-overlap0.log` | N=2 `OVERLAP=0` | [0, 2, 4] | [0, 1] / [0, 1] |
| `15-n2-shard0.log` | N=2 `SHARD=0` | [0, 2, 4] | [0, 1] / [0, 1] |
| `16-n3-asym.log` | N=3 `SPLITS=1,3` | [0, 1, 3, 4] | [0, 1] / [1, 2] |
| `17-n4-even.log` | N=4 | [0, 1, 2, 3, 4] | [0, 2] / [1, 3] |
| `18-n4-streams0.log` | N=4 `STREAMS=0` | [0, 1, 2, 3, 4] | [0, 2] / [1, 3] |
| `19-n2-longer.log` | N=2, P=16 N=24 | [0, 2, 4] | [0, 1] / [0, 1] |

`00-refusal-standing-n2.log` is the same gate run BEFORE the walk landed: it stops on the
refusal, which is what "the gate arrived before the code it holds down" looks like.

## RED — the mutation check

A gate that has never failed is not a gate. Three sabotages, each a temporary uncommitted edit
whose diff is banked at the head of its own log, each reverted immediately after.

Weight corruption is deliberately NOT in this set: it breaks both arms equally and the
comparison stays green, so it would prove nothing about a stage handoff. Nor is a different
VALID fence a mutation — any correct partition must be bit-identical, so that is another green
arm. The sabotage has to break the seam itself.

| log | sabotage | result |
|---|---|---|
| `93-RED-m4-prefill-layer-gap.log` | the same off-by-one, in the STATELESS PREFILL walk ONLY | **decode-serial PASS, prime-twin PASS, prefill-twin FAIL** (448/448 logits on the all-rows row, 32/32 on `forward_last`) — the arm added for `forward_hyper_ppn` binds and is not vacuous |
| `90-RED-m1-dropped-tx.log` | DROPPED TX: stage 0 runs its layer range, then publishes the PRE-range stream state, so the boundary carries nothing stage 0 computed | **FAIL both arms.** decode-serial 15/15 comparisons mismatched from step 0, 32/32 logits differ; both greedy tapes diverged at index 0 |
| `91-RED-m2-layer-gap.log` | OFF-BY-ONE LAYER RANGE: the last stage starts at `fence[n_st-1] + 1`, so the first layer of that stage is never run (a GAP, not a repartition) | **FAIL both arms.** decode-serial mismatched from step 0; the greedy tape only diverged at index 3 |
| `92-RED-m3-prime-layer-gap.log` | the same off-by-one, in the PRIME walk ONLY; `decode_step_hyper_ppn` untouched | **decode-serial PASS, prime-twin FAIL** — the two arms are independent and the prime twin is not riding the decode comparison |

Two findings the mutation runs produced, which is the argument for running them at all:

0. **The prefill walk was nearly shipped uncovered.** The first cut of this gate drove
   `decode_step` and `prime_cache` only; `forward`/`forward_last` route through
   `forward_hyper_ppn`, which has its own trunk entry and its own `last_only` head branch, and
   nothing compared it split-vs-unsplit. The prefill-twin arm and mutation M4 close that.

1. **The full-logit compare is the load-bearing bar, not the greedy tape.** Under M3 the
   prime-twin's greedy tape MATCHED while 32/32 logits differed at the prime's own last row;
   under M2 the tape survived three tokens past the first bad logit row. A tape-only gate
   would have called both of those green.
2. **The gate's own verdict arithmetic was wrong** and the red arms are what exposed it: a
   tape divergence was being counted into `bad_steps`, so the verdict printed "15/14
   comparisons mismatched". Fixed (`tape_bad` tracked separately) before the green matrix was
   re-run on the clean tree.

## Per-stage state audit (roadmap item 3), and why it is a tested property

glm5_next carries two per-layer state classes across a cut, and `pp::new_cache` already owns
the contract for both — `new_ppn`/`new_ppn_planned` pick the owning stage's `KvDev` per layer
for `Recurrent` (KDA conv ring + delta-rule state), `LatentKvCache` (MLA rows + the kpool
indexer plane) and `KvCache` alike. The kpool `index_pool_keys` plane, which `new_inner` leaves
`None` for the engine to allocate lazily, is allocated through the engine the mixer is called
with (`mla_attn_cached`'s `e`), which under these walks is the stage's engine. So no new
contract was needed.

That is asserted rather than argued: the gate refuses to compare unless the fence actually
SEPARATES those classes across stages (`spread.len() >= 2`), and it prints the placement on
every run. The table above is that print.

## Scope — what this does NOT establish

- **Every arm in the tables above is SAME-DEVICE.** The cross-device arms ran separately on a
  two-card box and are written up in `XDEV-FINDINGS.md` (18/18 bit-identical, plus a
  cross-device mutation arm that turns the gate red). The one arm that cannot run under this
  gate's design is `MEMRA_PP_HOST_BOUNCE=1`, for a reason that is not a defect — see that file.
- F32 fixture weights at tiny widths. The quantized expert class has its own gate
  (`glm5_moe_residency_gpu`); this one says nothing about it.
- No throughput claim of any kind. The rig is correctness-only by law, and the roadmap's
  step-3 projection (26.7 ms/token, 37.4 tok/s) remains a projection until a two-card cell
  measures it on the bench box.
- Nothing about the 190.7 GB artifact itself, which has never been loaded on this machine.
