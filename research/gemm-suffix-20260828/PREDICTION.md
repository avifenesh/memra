# Pre-registered prediction, written BEFORE the battery ran

Recorded 2026-08-28 while the battery was queued on the GPU lock, so the perf bar is a
falsifiable number rather than a bar drawn around whatever came out.

## Inputs (measured by the coordinator's five-point sweep, not fitted by this lane)

```
suffix= 17  ttft=0.245        suffix=393  ttft=2.501
suffix=133  ttft=1.058        suffix=693  ttft=4.082
suffix=213  ttft=1.490
LEAST SQUARES: ttft = 0.2529 s + 5.5978 ms/suffix-token,  R^2 = 0.9976
cold chunked prime, same cell: 0.99 ms/token
```

That supersedes the two-point fit in this lane's brief (7.18 ms/token + 140 ms). The shape
held; the slope was 28% high and the intercept was 81% low. Per-token the walk suffix costs
**5.65x** the batched prime (5.5978 / 0.99), not 7.2x, and break-even for a rewind at a
1488-token prompt sits at about **217 suffix tokens**.

## What the intercept is, and why it does not move

0.2529 s is not prime work: the affinity lane's LEG 2 reused a 1912-token prompt with a
24-token suffix in 0.314 s, and 0.2529 + 0.99e-3 x 24 = 0.277 s accounts for essentially all
of it. So the intercept is session restore, and this lane's change cannot touch it. Only the
5.5978 ms/token slope is in scope.

## Predictions

1. **Slope collapses to the cold rate.** A suffix on the batched entry runs the same GEMM
   trunk and grouped MoE as a fresh chunk, so the suffix cost becomes ~0.99 ms/token:
   `ttft ~= 0.253 + 0.99e-3 * L` instead of `0.253 + 5.598e-3 * L`.

2. **The growing-conversation pairwise ratio goes from 1.012x to ~3.5x.** At that cell's
   geometry (prompt ~1694, rewind to 1440, suffix 254): warm predicted
   `0.253 + 0.99e-3 * 254 = 0.504 s` against a measured cold twin of 1.755-2.367 s, i.e.
   **3.5x to 4.7x**. The lane PASSES on materially better than 1.012x; it CONFIRMS the
   mechanism only if the warm TTFT lands near 0.5 s rather than merely below the cold one.

3. **Break-even disappears.** With the suffix at the cold per-token rate, a rewind saves
   `reused_prefix_tokens * 0.99 ms` minus the ~0.25 s restore, so it wins at every suffix
   length instead of only below ~217 tokens. At 1440 reused tokens that is ~1.18 s.

4. **Byte identity holds at every swept suffix length** (LEG S), because the hoisted
   `seq_end` makes a cold full prime of P and a rewound suffix prime reaching P compute the
   same request-absolute value and therefore select the same SWA arm.

5. **The canary breaks identity only where the window boundary is straddled.**
   `MEMRA_STEP35_PRIME_BATCH_TSEND=1` passes the chunk-local length, which differs from the
   request-absolute value in ARM CHOICE only when the two straddle win=512. So identity
   should FAIL for the sub-512 suffixes (s0030, s0250, s0450) and for s4400 (its short
   second chunk), and PASS for s0700 and s1200 where both values exceed 512. A partial-fail
   pattern at exactly the 512 boundary is the canary working, not a flaky gate.

If (2) lands well below ~3.5x while (4) passes, the slope did not collapse and the suffix is
paying something this lane has not found; that is a finding, not a pass.

## Addendum, still before the battery ran: how to read the canary, and the flip gate

**The canary's effect scales with suffix length, so a short-suffix MATCH is not a failure.**
Under `MEMRA_STEP35_PRIME_BATCH_TSEND=1` the unwindowed arm hands each query roughly `L`
forbidden keys, where `L` is the suffix length. At s0450 that is ~450 forbidden keys per query
and at s0250 ~254 — perturbations that should flip a greedy token well inside 256 outputs. At
s0030 it is ~30 forbidden keys inside a ~542-row view, which may not move any token. So:

* the canary is DEMONSTRATED by any sub-512 FAIL, with **s0250 and s0450 the load-bearing
  rows**; an s0030 MATCH under canary is expected noise, not the seam failing;
* s0700 and s1200 should MATCH under canary, because chunk-local and request-absolute both
  exceed win=512 and select the same arm;
* if **s0450 MATCHES under canary**, the run cannot distinguish "seam not read" from "defect
  not byte-visible" — there is no engagement receipt for the TSEND read, which is precisely
  the wiring-assertions-match-prose trap. That, and only that, is a rebuild question.

**Flip gate for `MEMRA_STEP_GEMM_PRIME_SUFFIX` default OFF -> ON**, fixed here so the results
cannot renegotiate it. All three required:

1. ON-arm LEG S is MATCH on every valid row.
2. ON-arm ARM VALIDITY = VALID: `eng_suffix > 0` and `walk_suffix == 0`, i.e. the suffix
   provably rode the batched entry and none fell through.
3. OFF-arm slope fit lands near 5.6 ms/suffix-token — the within-battery replication of the
   coordinator's sweep. If the off arm does not reproduce it, the cost frame this lane is
   built on is suspect and nothing gets flipped on it.

A LEG G ratio below the predicted ~3.5x with identity green is a finding to bank and explain,
not a flip blocker: the task's bar is materially better than 1.012x. The flip itself stays the
plain env read (`== Ok("1")` becomes `!= Ok("0")`); joining `arm_step37_serving_defaults` is an
owner-flip-scoped list and is not this lane's to extend.

## Second addendum, also before results: the row-exclusion rule for very short suffixes

`prime_cache` has a floor at `PRIME_MIN_T = 16`, and the worker only calls it when the queue
clears that floor — below it the suffix is fed through `decode_step` one token at a time. That
is a DIFFERENT numeric program for the same bytes (the recorded gap #46 fork), so such a row
tests neither arm of this lane and must not be scored either way.

The s0030 sweep point targets ~30 suffix tokens off a words-to-tokens calibration, so it could
undershoot the floor. The instrument already prints what is needed to detect it: a warm row
that rewound but shows `eng[suffix=0]` AND `walk[suffix=0]` never entered `prime_cache` at all.

**Rule, fixed here: any LEG S warm row with `eng_suffix == 0 && walk_suffix == 0` is EXCLUDED
from the identity verdict and reported separately with its measured suffix length.** It is not
a pass and not a fail. The driver's own `pass/fail/invalid` tally does not implement this, so
the exclusion is applied when reading the rows; it is stated now rather than after seeing which
way the row went. This deliberately does not touch the driver, because the `off` arm was already
running and all three arms must share one instrument.
