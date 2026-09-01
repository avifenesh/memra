# ON arm: three predictions confirmed, one falsified, and the falsification is not this lane's

## Confirmed, and prediction 1 was beaten

| quantity | off arm (walk suffix) | on arm (batched suffix) | predicted |
|---|---|---|---|
| suffix cost line | `0.2490 s + 5.8067 ms/tok`, R^2 0.9999 | `0.3060 s + 0.7286 ms/tok`, R^2 0.9987 | ~0.99 ms/tok |
| LEG G pairwise ratio (n=5) | median 1.088x, min 0.973, max 1.332 | median **3.388x**, min 3.058, max 4.171 | ~3.5x |
| ARM VALIDITY | VALID (eng_suffix=0, walk_suffix=41) | VALID (eng_suffix>0, walk_suffix=0) | required |

The slope collapsed by **7.97x** (5.8067 -> 0.7286 ms/suffix-token), beating the predicted
0.99. Per-row agreement with the prediction is tight: s0250 has a 289-token suffix, predicted
`0.253 + 0.99e-3 * 289 = 0.539 s`, measured **0.557 s**. Warm TTFTs across the sweep:
0.311 / 0.557 / 0.703 / 0.866 / 1.155 / 3.547 s at suffixes 80 / 289 / 489 / 740 / 1240 / 4440,
against off-arm 0.768 / 1.971 / 3.135 / 4.491 / 7.324 / 26.066 s. The 1.012x-1.088x wash
becomes a 3.388x median win. Predictions 1, 2 and 3 hold.

## Falsified: prediction 4, byte identity. It fails 0/6 in BOTH arms

I predicted ON-arm identity would be 6/6. It is 0/6. It is also 0/6 in the OFF arm, which is
the incumbent. So this is not a regression introduced by the hoist — but my prediction was
wrong and the reason I gave for expecting it was wrong too.

**The divergence does not start at the rewind boundary.** The receipts show where it starts:

```
s-est  (turn 1, both arms):  ENGAGED t=1440 base=0    seq_end=1487
s-cold (s0250, on arm):      ENGAGED t=1696 base=0    seq_end=1729
s-warm (s0250, on arm):      ENGAGED t=256  base=1440 seq_end=1729
                             ENGAGED t=33   base=1696 seq_end=1729   <- identical in both
```

The warm session's KV rows 0..1439 were produced by turn 1's prime, a batched chunk of
**m = 1440**. The cold twin's rows 0..1439 were produced inside a batched chunk of
**m = 1696**. If the batched prime is m-dependent, those rows already differ before a single
suffix token is primed, and every later row inherits it.

Three facts from the battery say that is what is happening, and rule out the alternatives:

1. **It is not the suffix implementation.** In the OFF arm the suffix ran the walk, which the
   chunkfix lane made chunk-invariant, and identity still failed 6/6.
2. **It is not the SWA arm selection, i.e. not the `seq_end` threading.** In the ON arm
   `seq_end` is identical on both sides of every pair (1520, 1729, 1929, 2180, 2680, 5880) and
   exceeds win=512 everywhere, so both sides take the windowed arm at every chunk.
3. **It is not a prompt mismatch or a hash artefact.** `seq_end` agrees pair-by-pair, the
   trailing chunks are literally identical (`ENGAGED t=33 base=1696 seq_end=1729` in both), and
   the sweep produced 6 distinct completions over 6 prompts.

The one factor common to both failing arms is that the reused prefix was primed by the batched
entry at a different `m` than the cold twin's prime. That is a property of the batched prime,
which `docs/FLAGS.md` already declines to claim byte identity for: "NUMERIC CLASS: the
f16-mirror grouped-prefill class other families serve; admission = prefill-KV acceptance gate +
ship-shape tape + interleaved wall, **never byte identity**." The grouped NVFP4 MoE builds its
CSR and per-expert grouping from the tokens in the chunk, so the per-expert GEMM shapes move
with the chunk contents.

## The consequence for the task's bar

"Greedy byte-identity between a cold full prime of P and a rewound-plus-suffix prime reaching
the same P" is not a bar the incumbent meets, and it cannot be met by ANY suffix arm while the
batched prime is m-dependent, because session reuse by construction replays a prefix that was
primed under a different chunk shape. The bar as written is unreachable, not merely unmet.

**This also means production has a live consistency property nobody has stated: today a rewound
session answers the same bytes differently from a cold session** (off arm, 6/6, suffixes 80 to
4440 tokens). That is worth the owner's attention independent of this lane.

## The decisive diagnostic, queued

One more arm, `MEMRA_STEP_GEMM_PRIME=0`, everything on the chunk-invariant walk including the
turn-1 prefix and the cold twin. Trimmed to s0250 and s0700 so it holds the shared lock for as
little as possible. **If it comes back 2/2 MATCH, the identity failure is the batched prime's
m-dependence and nothing to do with the hoist.** If it fails too, the cause is upstream of the
prime entirely and this lane's reading is wrong.

## Flip decision

Pre-registered gate condition (a) -- ON-arm LEG S MATCH on every valid row -- FAILS.
**The default stays OFF.** The gate was fixed before the data and it is not renegotiated now
that the perf numbers are good. What the data supports instead is a recommendation, for the
owner rather than for me to take unilaterally: admit the suffix arm under the SAME standard the
batched prime itself is admitted under (acceptance gate, ship-shape tape, interleaved wall),
because byte identity against a cold prime is not a property that path has anywhere.
