# DFlash2 root-cause diagnosis, 2026-09-09

## Current disposition, 2026-09-09

Evidence only. The unserved causal-PMIN door, helper, dispatch and two CPU
regression tests were removed from runtime on 2026-09-09. Re-derive from
`f197eb413f6c23ab1e05eb41316c8d5f8e61dc5c`; `causal-pmin.patch` retains
the fix and tests. The existing distribution bug remains open in
[#412](https://github.com/avifenesh/memra/issues/412), including the composed
1.87 -> 1.56 accepted/round receipt and its comparison limits.
The measurements below describe the archived experiment. rev: 2026-09-23.


**On the requested PP1/p32k shape, vendor-default DFlash2 is slower because it
accepts too little work to amortize an expensive target verifier.** Best measured
vendor K=6 is **71.850085 tok/s**, versus same-boot plain **80.635647 tok/s**, a
10.895382% deficit. It accepts **1.850000 drafts/round** and needs **2.192253** at
its measured **39.588602 ms/round**. This is a software optimization target,
not a demonstrated hardware limit.

## Diagnosis in three sentences

Vendor-default K=6 pays 32.066372 ms in verify out of 39.210672 ms of engine
round wall, about 81.8%, while only 1.850 accepted drafts amortize each round.
The sampler uses rejection/residual sampling, and this p32k cell does not
reproduce a greedy advantage: greedy K=6 is 65.280566 tok/s with 1.492683
accepted drafts/round, while only 9 of 84 rejections in the separate sampled
K6 capture rejected an argmax-matching proposal.
Separately, the selected-token PMIN cutoff biases the target distribution;
PR #393 archives a removed correction with two historically passing CPU regression tests,
without changing serving defaults or claiming an ON-arm model qualification.

## Identity, completion and scope

- Measured engine base `6285210078609c7e11aa23ae070ad581c184c153` plus
  `instrumentation.patch`; binary SHA256
  `18e42b7e083456a7cfe790b1dda27a0f4ac7ab3898377bc7bdf72587ed35d59e`.
  The PMIN correction was added to the source after this executable was built;
  it is absent from the scored executable. The archived correction was later removed.
- One B200, sm_100a, CUDA 13.1.115, one successful `memra-server` boot. All GPU
  phases held the shared lock. Build used an isolated target, nice 19, -j16.
  No cargo, GPU gate, benchmark or server ran on the local rig; the authorized
  pre-push command ran its metadata censuses and recorded MEMRA_SKIP_PERF_CI=1.
- Native mint and DFlash2 artifact inventory: `staged-model-hashes.txt` and
  `final-artifact-hashes.txt`. Drafter weights SHA256
  `b33c03475ba7322cf398828f2d8d1be376df30dc05c6b40c28c8ea8da23e410b`.
  Boot says DFlash2, full target vocabulary, and no native MTP head. Every
  measured spec request has `trim_rounds=0`; FR-Spec masking is not armed.
- The supplied p32k text renders to **29,781 prompt tokens**. All 83 scored and
  agreement requests use a 512-token cap; 80 reach it. Three natural stops are
  retained: timing-r0-p1-K2=382, timing-r2-t06-K4=507,
  timing-r2-vendor-K2=468. The two 64-token warmups are excluded from medians.
- **63 timing requests, N=3 for each of 21 arms; 20 separate agreement requests.**
  All 85 requests complete HTTP 200 with terminal SSE. Timing logs reconcile
  12,473 rounds with usage.spec. K pins are exactly 2/4/6; auto selects 3.
  All scored requests have full 29,781-token cache hits. No loop flags.
  All 16 greedy concatenated-text captures share SHA256
  `40adb5ec1871003eb4d0719256c2f318d2a1ee62bb38bb091de5b02703223270`;
  this is text identity, not a token-ID or plain-path oracle.
- The boot was warmed by spec and plain requests. 250 ms telemetry records
  36..51 C over the whole boot, including load and cold warmup. These are
  synchronized diagnostic timings, not an uninstrumented serving qualification.
  Shadow draws never enter timing medians or advance either serving RNG counter.
- The final diagnostic launcher uses `MEMRA_PRIME_CHUNK=256` to fit retained
  workspace, and the existing `MEMRA_GLM5_SPEC_FULLCOVER=1` to re-arm repeated
  identical prompts. Admission and the 1.5 GiB transient reserve remain enabled.
  `posture.json` records the whole recipe. This prime lineage differs from the
  older tally; do not treat them as one unchanged numerical tuple.

## Acceptance decomposition

Each cell is the median of three per-request measurements, so separately rounded
median columns need not reconstruct the median rate exactly. HTTP wall includes
TTFT; engine wall excludes request admission/restore. Acceptance is total
accepted / total drafted, not a per-position conditional probability.

`t06` is temperature=.6/top_p=.95; `vendor` omits all sampling parameters and
resolves to temperature=1/top_p=.95; `p1` is temperature=1/top_p=1; `k40` is
temperature=1/top_p=1/top_k=40, explicitly disabling top-p to isolate top-k.
One prompt and N=3 are a diagnostic scope, not a broad workload ranking. Every
one of the 12 vendor requests was slower than every plain request: vendor max
74.787807 tok/s versus plain min 80.329599 tok/s.

| Sampling | K | N | Drafted/round | Accepted/round | Acceptance | HTTP ms/round | Engine ms/round | Verify ms/round | HTTP tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| greedy | 2 | 3 | 1.688 | 1.184 | 0.701 | 33.922 | 33.610 | 27.084 | 64.502 |
| greedy | 4 | 3 | 2.292 | 1.410 | 0.615 | 36.957 | 36.647 | 30.136 | 65.349 |
| greedy | 6 | 3 | 2.532 | 1.493 | 0.590 | 38.259 | 37.909 | 31.288 | 65.281 |
| greedy | auto | 3 | 2.078 | 1.344 | 0.647 | 35.414 | 35.074 | 28.481 | 66.320 |
| t06 | 2 | 3 | 1.781 | 1.193 | 0.683 | 34.718 | 34.434 | 27.366 | 63.295 |
| t06 | 4 | 3 | 2.884 | 1.823 | 0.628 | 40.728 | 40.320 | 33.116 | 69.454 |
| t06 | 6 | 3 | 3.553 | 1.882 | 0.529 | 44.044 | 43.661 | 36.462 | 64.857 |
| t06 | auto | 3 | 2.409 | 1.586 | 0.671 | 37.147 | 36.811 | 29.664 | 69.704 |
| vendor | 2 | 3 | 1.631 | 1.256 | 0.753 | 34.210 | 33.866 | 26.812 | 65.707 |
| vendor | 4 | 3 | 2.244 | 1.445 | 0.644 | 37.556 | 37.219 | 30.013 | 65.230 |
| vendor | 6 | 3 | 2.678 | 1.850 | 0.691 | 39.589 | 39.211 | 32.066 | 71.850 |
| vendor | auto | 3 | 2.083 | 1.493 | 0.717 | 35.894 | 35.550 | 28.443 | 69.581 |
| p1 | 2 | 3 | 1.641 | 1.212 | 0.739 | 33.848 | 33.498 | 26.823 | 65.770 |
| p1 | 4 | 3 | 2.323 | 1.586 | 0.679 | 37.481 | 37.149 | 30.427 | 68.991 |
| p1 | 6 | 3 | 2.387 | 1.634 | 0.654 | 37.731 | 37.367 | 30.541 | 69.947 |
| p1 | auto | 3 | 2.068 | 1.490 | 0.716 | 35.428 | 35.083 | 28.382 | 70.155 |
| k40 | 2 | 3 | 1.619 | 1.261 | 0.770 | 34.135 | 33.826 | 26.777 | 66.369 |
| k40 | 4 | 3 | 2.186 | 1.490 | 0.671 | 37.263 | 36.883 | 29.716 | 66.100 |
| k40 | 6 | 3 | 2.616 | 1.700 | 0.648 | 39.320 | 38.964 | 31.769 | 68.046 |
| k40 | auto | 3 | 1.919 | 1.302 | 0.678 | 35.219 | 34.915 | 27.834 | 65.485 |
| plain | 0 | 3 | - | - | - | - | - | - | 80.636 |

## Agreement by position

These are separate 512-token captures, one per sampling/K arm. A shadow sample
is drawn from the target's filtered distribution using a separate Philox seed
namespace; it is not the actual acceptance test. At temperature zero it is the
argmax by definition. Full results for all 20 arms are in [AGREEMENT.md](AGREEMENT.md).

All-position statistics include hypothetical draft prefixes beyond the first
rejection. Reached-position statistics condition on the previous slots having
been accepted, and therefore describe the actual acceptance walk. Tail sample
counts are small and shown explicitly.

| Arm | Position | N verified | Argmax match | Shadow sample match | N reached | Argmax match/reached | Sample match/reached | Accepted/reached | Argmax rejected at reached slot |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| greedy-K6 | 1 | 205 | 72.20% | 72.20% | 205 | 72.20% | 72.20% | 72.20% | 0 |
| greedy-K6 | 2 | 136 | 78.68% | 78.68% | 104 | 83.65% | 83.65% | 83.65% | 0 |
| greedy-K6 | 3 | 79 | 59.49% | 59.49% | 49 | 71.43% | 71.43% | 71.43% | 0 |
| greedy-K6 | 4 | 48 | 66.67% | 66.67% | 22 | 77.27% | 77.27% | 77.27% | 0 |
| greedy-K6 | 5 | 32 | 59.38% | 59.38% | 14 | 78.57% | 78.57% | 78.57% | 0 |
| greedy-K6 | 6 | 19 | 68.42% | 68.42% | 10 | 80.00% | 80.00% | 80.00% | 0 |
| vendor-K6 | 1 | 200 | 75.00% | 72.00% | 200 | 75.00% | 72.00% | 77.50% | 4 |
| vendor-K6 | 2 | 124 | 70.16% | 70.16% | 98 | 75.51% | 76.53% | 76.53% | 3 |
| vendor-K6 | 3 | 73 | 79.45% | 76.71% | 47 | 85.11% | 82.98% | 89.36% | 0 |
| vendor-K6 | 4 | 43 | 72.09% | 65.12% | 28 | 78.57% | 67.86% | 75.00% | 1 |
| vendor-K6 | 5 | 26 | 73.08% | 69.23% | 15 | 80.00% | 80.00% | 80.00% | 1 |
| vendor-K6 | 6 | 14 | 64.29% | 71.43% | 7 | 71.43% | 85.71% | 85.71% | 0 |

In the vendor K6 capture, 480 proposals were verified across 200 rounds. Of 395
reached positions, 303 proposals matched target argmax, 295 matched an independent
target draw, and 311 were accepted. Nine argmax-matching proposals were rejected,
and 17 non-argmax proposals were accepted. Thus the rejection sampler accepts
more reached proposals than an argmax-match rule would on these fixed rows;
this is not a counterfactual whole-generation throughput prediction. Only 9/84
actual rejections are the "sampling rejects a correct-argmax draft" case here.
The recorded p/q/uniforms replay every captured sampled acceptance prefix.

## Confirmed acceptance rule and PMIN defect

At measured base `6285210078609c7e11aa23ae070ad581c184c153`,
`crates/memra-engine/src/glm_spec.rs:4174-4192` dispatches sampled DFlash2 to
`dspark_accept_sampled`. `crates/memra-engine/src/dflash.rs:988-998` accepts a
prefix while `u[j] * q[j] < p[j]`; `:1232-1240` filters the target distribution,
`:1252` reads the recorded selector proposal probability, `:1262` applies the
prefix test, and `:1289-1344` draws the rejected position from the positive
residual, with sparse selector q. The full-accept branch draws a filtered target
bonus. `glm_spec.rs:4126-4154` uses argmax matching only when sampling is absent.
This is the standard [rejection/residual sampling rule](https://arxiv.org/abs/2211.17192),
not sampled-token equality or sampled argmax matching.

The cutoff preceding that walk has a separate bug. `glm_spec.rs:4474-4493`
samples the candidate and retains its original q; `:4503-4505` then stops before
a low-q selected token. At a later slot, conditional on an accepted prefix,
p=(1/2,1/2), q=(9/10,1/10), PMIN=7/10 yields (11/20,9/20), not p. The low-q draw
is censored and replaced by a target bonus, while kept proposals still use the
uncensored q. `pmin-counterexample.py` enumerates this exactly using rational
arithmetic; `pmin-counterexample.json` is its on-box receipt. This demonstrates a
correctness defect; it does not measure its size on GLM or establish it as the
cause of the throughput gap.

The candidate correction, `MEMRA_GLM5_SPEC_CAUSAL_PMIN=1`, tests the maximum q of
the current slot's distribution. Given the preceding draft prefix, this stop
decision does not depend on the current selected token. The unchanged rejection
walk and residual therefore retain their target-distribution contract. It is
OFF by default, scoped to sampled DFlash2 with PMIN>0. Greedy, native MTP and
PMIN=0 are unchanged. Native MTP's selected-token cutoff needs its own audit;
this patch does not claim to fix every speculative source.

Two new CPU unit tests passed on the B200 development host (`causal-tests.log`):
exact enumeration reproduces the legacy (.55,.45) and corrected (.5,.5), and
prefix/slot-zero cutoff contracts remain intact. No model-scale ON-arm serving
or distribution qualification is claimed. The default-off door expires for a
receipt-backed decision on 2026-09-23.


## Verify cost and break-even

Plain baseline: 80.635647 HTTP tok/s; 81.450887 decode-interval tok/s; 12.277337 ms per plain token interval (client wall, includes sampler/transport, not isolated GPU).

| Vendor K | Observed accepted/round | HTTP ms/round | Break-even accepted/round | Engine ms/round | Decode break-even accepted/round |
|---|---:|---:|---:|---:|---:|
| 2 | 1.256039 | 34.209755 | 1.758526 | 33.865826 | 1.758402 |
| 4 | 1.444976 | 37.555889 | 2.028343 | 37.218905 | 2.031513 |
| 6 | 1.850000 | 39.588602 | 2.192253 | 39.210672 | 2.193744 |
| auto | 1.492683 | 35.894493 | 1.894376 | 35.549697 | 1.895554 |

Break-even uses a+1 tokens/round and ignores first-anchor/final-cap boundary corrections. The HTTP calculation compares HTTP with HTTP; the decode screening calculation uses engine round wall and the plain client interval. Neither is a hardware ceiling.

| Verify width | Current synchronized verify median ms | N current rounds | Prior tally GPU span ms | Prior N |
|---|---:|---:|---:|---:|
| 2 | 24.339360 | 4170 | 25.605873 | 10 |
| 3 | 28.227006 | 4405 | 29.379530 | 3 |
| 4 | 31.874907 | 2219 | 32.953999 | 3 |
| 5 | 39.189727 | 1112 | 40.268890 | 1 |
| 6 | 43.341594 | 178 | 44.319680 | 2 |
| 7 | 49.555427 | 389 | 50.458073 | 2 |

At the current K6 draft mix, either raising accepted drafts/round from 1.850000
to 2.192253, or reducing round wall from 39.588602 to 35.344170 ms, crosses the
plain baseline in the steady-state arithmetic. That is **4.244432 ms** of round
saving at fixed accepted work. The fixed-cost all-proposals-accepted ceiling is
above plain, so these measurements do not prove DFlash2 cannot win on this card;
they show that no tested vendor-default K wins on this recipe now.

The target-only upper bound at unchanged accepted work is 88.878 tok/s if all
non-verify work vanished. This is a bound, not an achievable drafter estimate.
Increasing K also increases verify width: t3 -> t5 -> t7 rises roughly
28.2 -> 39.2 -> 49.6 ms before drafting, acceptance and rollback. A policy cannot
price six full drafts with the current mixed-width 39.6 ms round cost.

## Historical receipts reconciled

The old short greedy ladder reports **125.78 tok/s at K=6**. Its companion usage
census reports **4.77 accepted/round**, about **5.25..5.30 drafted/round**, and
**45.5..45.7 ms/round**. This new p32k greedy K6 row has **1.492683 accepted/round**,
**2.531707 drafted/round**, and **38.258875 ms/round**: the old accepted-work
amortization is absent even with temperature zero. The short greedy result does
not predict this long-prompt sampled workload.

The later short PP2 vendor receipt is **72.329898 tok/s**, acceptance **0.622222**,
**2.414634 drafted/round**, **1.502439 accepted/round**, and **34.530133 ms/round**.
PMIN=0 gives **6.000000 drafted/round**, **1.912015 accepted/round**,
**52.921801 ms/round**, and **54.917607 tok/s**. Extra accepted work grows only
27.26%, while round cost grows 53.26%; removing the cutoff therefore loses.
Against that receipt's **87.30 tok/s plain TP2**, the break-even accepted counts
are **2.014481** and **3.620073**, respectively. These are historical short PP2/TP2
rows, not the new PP1 baseline.

The older hybrid-mint `headline4` row is recorded as vendor-default sampled
**105.12 / 104.36 / 105.72**, median **105.12**, versus composed plain **84.27**.
Its request bodies and full source tuple were not recovered locally, and this
lane did not access the serving pair. It remains a distinct historical positive
receipt, not a matched control for today's 72.33 or p32k rows; attributing the
105 -> 72 difference to sampling alone would exceed the available evidence.

The banked tally uses base `dcfeab7c` plus its own instrumentation, retained in
`tally-reference/` from commit `188ab8bf13151c99d87d700c71088e7ed1798829`.
At t4 it records **2933 kernels/round**, **65.066667 per layer**, gathered MLA
attention **7218.395 us**, and six E4M3 input projections **3811.757 us** for
**3467.117 logical MB**. These are logical weight bytes, not measured HBM traffic.
Its source attribution identifies serial tiled softmax/barriers/L2 traffic in
`crates/memra-engine/cu/mla_attn.cu:2652`, while the faster t1 program already
has mechanisms that the verified multi-row path does not use. This is a concrete
software cost discrepancy, not a power-limit attribution.

## Ranked fixes and the gates they need

The following prices are arithmetic scenarios at the K6 median, using
`1000*(accepted+1)/round_ms`. They ignore small cap-boundary corrections and
are not measured optimized-server rates.

1. **Correct PMIN's sampling contract.** The default-OFF correction in this PR
   is a correctness prerequisite. OFF retains the measured 71.850 tok/s path;
   ON has no honest throughput forecast yet because max(q) can retain additional
   verify rows and changes the previously biased continuation distribution.
   Gate: the passing exact-enumeration tests, then full served ON/OFF sampling
   distribution checks, prefix restore, penalties and sampled continuation.
2. **Reduce verify cost, starting with the already banked six-projection fusion
   and the gathered MLA bottleneck.** `tally-reference/FUSION-RESULTS.md` records
   byte-exact real-input checks and **1.789758071 ms/round** weighted component
   saving. If that saving transfers to this width mix, arithmetic gives **75.399
   tok/s**, still below plain. The MLA lane subsequently banked **3.711449136 ms/round** at p32k
   (**3.700307708 ms/round** across p32k/p128k), with 18,304 real latent-row
   argmax/band checks and 66 exact gathered-input checks. See
   `tally-reference/MLA-RESULTS.md` and Memra PR #394. Transferring both component
   savings gives **83.609 tok/s** in the arithmetic; their combined served
   saving is not measured. At **t=4**, the component savings are
   **1.743582405 ms/round KDA** and **3.091277122 ms/round MLA at p32k**,
   a sum of **4.834859527 ms/round**. Applying that fixed t4 saving to the
   K6 median as a screening scenario gives **82.006 tok/s**; actual K6
   spans several verify widths, so this is not a composed serving receipt.
   Gate: preserve the accepted numerical class at t2..7, then compose and run
   sampled served requests plus cache continuation. The earlier faster MLA arm's
   t4 argmax failure is not waived by its speed; the existing MLA lane owns
   this work. No duplicate kernel implementation was started here.
3. **Choose plain at admission when predicted accepted work cannot pay for
   verification.** K=0 already measures **80.635647 tok/s** here. Among speculative
   options K=6 has the highest vendor median, but K=2/4/auto all lose as well.
   A lower K alone does not solve this case. Gate: validate the selection policy
   across prompts and cache states; an in-session sampled handoff additionally
   needs sampler/RNG/state and continuation qualification.
4. **Calibrate the proposal while preserving target temperature=1/top_p=.95.**
   There is no temperature-wiring bug: measured `glm_spec.rs:4475` passes
   `sp.temp` into the DFlash2 selector. The .6 arm changes both distributions and
   does not establish a proposal-only optimum. A gain of **0.350 accepted/round**
   at unchanged cost would price at **80.831 tok/s**. Gate: proposal-only
   calibration with the actual q retained, causal cutoff semantics, and sampled
   distribution/quality/continuation receipts on the pinned artifact.
5. **Keep FR-Spec changes on the proposal side.** The current boot uses a full
   head, so a target-vocabulary mask is not causing the measured loss. The
   **88.878 tok/s** bound removes *all* non-verify work; a draft-head trim removes
   only a fraction, so that number is not its forecast. Gate: correct D2T mapping,
   recorded trimmed proposal q, full-vocabulary target p and residual support,
   plus acceptance and sampled serving checks. Measure the head share first.
6. **Typical/relaxed acceptance changes the exactness contract.** Raising accepted
   work by 0.350 at fixed cost also prices at **80.831 tok/s**, but accepting mass
   beyond p/q generally changes the target distribution. It cannot be presented
   as exact vendor-default sampling. Gate: an explicitly different approximation
   contract with measured bias and task-quality effects, in addition to serving
   correctness. The 9 argmax-matching rejects in this trace do not justify making
   this the leading remedy.

## Evidence and apparatus failures

The complete `rootcause-raw.tar.gz` is banked in private Darklanes at
`research/glm5-1m-b200-ship-20260906/receipts/dflash2-20260908/rootcause-20260909/`.
It contains the complete successful boot, every HTTP body and
SSE stream, per-round logs, shadow proposal/argmax/sample/p/q/u rows, telemetry,
build/test logs, and the excluded apparatus attempts. SHA256:
`020bc33305391e075f2b38b3ad44f872925b1869cae17b3213c2552016e53610`.
The public-boundary hook rejected a provider-name pattern in the compressed
archive, so the intact archive was moved to private custody, without an allowlist
exception. `raw-manifest.json` hashes every member; `summary.json` retains per-request values,
all agreement counts and medians; `validation.json` records reconciliation.
`summarize.py`, `tables.py` and `seal.py` reproduce the tables and checks.

The preflights exposed a too-large retained prefill workspace, plain-created
prefixes lacking drafter tails, the full-cover restore gate, and local queue
clock limits. `PLAN.md` records each intervention and the preserved failure
namespace. They are excluded rather than converted into speculative results.
The shared tune box is retained for the other lanes; this lane releases its GPU
lock and removes only its own processes, worktrees, target and scratch after bank.
