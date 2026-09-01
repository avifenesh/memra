# OFF arm read (post hoc where marked)

## Predicted and confirmed

* **Slope reproduces in-battery.** `ttft = 0.2490 s + 5.8067 ms/suffix-token, R^2 = 0.9999 (n=6)`
  against the coordinator's `0.2529 s + 5.5978 ms/token`: intercept within 1.5%, slope within
  3.7%. Flip-gate condition (c) is SATISFIED, and the cost frame this lane is built on holds.
  `slope vs cold prime = 5.87x`.
* **The wash reproduces.** LEG G pairwise ratio median 1.088x (n=5, min 0.973, max 1.332)
  against the affinity lane's 1.012x, at suffixes 234-256.
* **ARM VALIDITY VALID**: eng_fresh=19, eng_suffix=0, walk_fresh=0, walk_suffix=41. Every
  suffix took the walk and none rode the batched prime, which is what `SUFFIX=0` must mean.
* **The short-suffix exclusion rule did not need to fire.** s0030's measured suffix was 80
  tokens, above `PRIME_MIN_T`, so all six rows entered `prime_cache`.

## NOT predicted: LEG S is 0/6 in the OFF arm, and the reason matters

I predicted ON-arm identity and said nothing about the OFF arm. It came back pass=0 fail=6,
with 6 distinct completions over 6 swept prompts (so this is not a hash collision artefact).

**Post-hoc explanation, from the receipts rather than from theory.** The engagement lines show
the two sides of each comparison are primed by DIFFERENT IMPLEMENTATIONS in this arm:

```
cold t4: [gemm-prime] ENGAGED t=2112 base=0    seq_end=2129   <- batched GEMM prime
         [gemm-prime] WALK    t=17   base=2112 seq_end=2129
warm t4: [gemm-prime] WALK    t=192  base=1888 seq_end=2122   <- walk, both segments
         [gemm-prime] WALK    t=42   base=2080 seq_end=2122
```

The cold twin runs almost entirely through the batched entry; the rewound twin runs entirely
through the walk. Those are documented as DIFFERENT NUMERIC CLASSES — `docs/FLAGS.md` on
`MEMRA_STEP_GEMM_PRIME` says so in as many words: "NUMERIC CLASS: the f16-mirror
grouped-prefill class other families serve; admission = prefill-KV acceptance gate + ship-shape
tape + interleaved wall, never byte identity."

So the oracle "cold full prime == rewound suffix prime" was NEVER satisfiable while the gate
stood. It is not a bar the incumbent meets and this lane fails to clear; it is a bar the gate
made unreachable, and closing the gate is what makes both sides one implementation.

**This is a second, user-visible defect the lift closes**: today a rewound session answers a
prompt differently from a cold session given the identical bytes, because the two take
different prime implementations. That is a session-affinity consistency bug, independent of
speed, and it is measured here at 6/6 across suffixes from 80 to 4440 tokens.

## Why this does not contradict the affinity lane's 5/5 MATCH

Their identity leg sends the SAME prompt twice on the SAME session: the first send primes the
cache and the second rewinds into KV rows the first send itself produced, so both sides share
one prime. This lane's cold twin is a SEPARATE session priming the same bytes from scratch,
which is what exposes the cross-implementation seam. Both results are correct; they measure
different things, and the distinction should be stated wherever either is cited.

## What the ON arm must now show

Unchanged from PREDICTION.md, and now with a sharper meaning: if the suffix rides the batched
entry, BOTH sides of every LEG S pair are the same implementation at the same `seq_end`, so
identity should go 6/6. A partial pass would say the seq_end threading is right but something
else in the continuation path is not.

## Instrument validation: the two sides really did prime the same bytes

`seq_end` on the engagement lines IS the request's prompt length, so it is a direct check that
the warm and cold twins are the same prompt. Every LEG S pair agrees exactly:

```
        warm                                cold                                seq_end
s0030   WALK t=64 base=1440 + t=16 b=1504   ENGAGED t=1504 b=0 + WALK t=16 b=1504   1520
s0250   WALK t=256 base=1440 + t=33 b=1696  ENGAGED t=1696 b=0 + WALK t=33 b=1696   1729
s0450   WALK t=448 base=1440 + t=41 b=1888  ENGAGED t=1888 b=0 + WALK t=41 b=1888   1929
s0700   WALK t=704 base=1440 + t=36 b=2144  ENGAGED t=2144 b=0 + WALK t=36 b=2144   2180
s1200   WALK t=1216 base=1440 + t=24 b=2656 ENGAGED t=2656 b=0 + WALK t=24 b=2656   2680
s4400   WALK t=4416 base=1440 + t=24 b=5856 ENGAGED t=5856 b=0 + WALK t=24 b=5856   5880
```

Two things this pins down beyond argument:

1. **Same prompt.** Identical `seq_end` on both sides of all six pairs, and identical trailing
   chunks (`t=33 base=1696` appears verbatim in both arms of s0250, and so on). The prompts and
   even the final chunk are shared; only the bulk differs.
2. **The bulk is where the implementations part.** Cold computes rows `[0, N)` through the
   batched GEMM prime; warm computes rows `[1440, N)` through the walk. That overlap region,
   rows 1440..N, is the entire delta, and it is computed by two different implementations of
   different documented numeric classes.

So the ON arm is a clean test of one specific property: whether the BATCHED path is
chunk-invariant, i.e. whether rows 1440..N come out the same computed inside a single
N-token batched chunk (cold) as computed in a standalone batched chunk based at 1440 (warm).
That property has never been exercised below 4096 tokens, because the batched entry only ever
produced one chunk there. s4400's cold side (5856 tokens = 4096 + 1760) exercises it directly.
