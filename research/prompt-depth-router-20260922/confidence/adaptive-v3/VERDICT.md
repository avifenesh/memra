# Qwen code K=3: live learned confidence

**Verdict: keep K=3/C=0 as the research control.** This versioned
experiment measured a genuinely live, data-derived confidence cutoff.
It did not improve the pinned Qwen code workload. The older
C=0/0.15/0.30 experiment tested preset cutoffs and used a sampled
token-discard path; its positive-C rates remain separate diagnostics.

The pinned model is
`tiyuvta/Qwen3.8-27B-NVFP4-MTP-GGUF@0f82b27dbb264b731e7d20f576582c871ef1969c`,
`Qwen3.8-27B-NVFP4-Q5K-mtp.gguf` SHA-256
`1facf36c2db359dcf9c2475cf8f85fe84a528d10aaaaff20f7c0db3d561e024a`,
with the full 248,320-row embedded MTP head. The corrected research
source archive SHA-256 is
`49015ce03f02b3fee00223b5914377b04b13ab47a88a4d7c36fe96444006223a`;
the measured RTX 5090 binary SHA-256 is
`9f879f700d5626a598c216dc9529d1efdae3ca3af6540ae5b5c519c2613b5ada`.
Six disjoint, eight-turn heldout code conversations used native KV
continuation, sampled temperature 0.7/top-k 20/top-p 0.95, default
thinking, `max_new=8192` and `ctx=65536`. All five arms ran on one
non-production RTX 5090.

| Arm | Complete native tok/s | Versus K=3/C=0 |
|---|---:|---:|
| K=3/C=0 | 139.74 | control |
| Live learned C, K=3 | 120.96 | −13.44% [−14.58%, −11.83%] |
| Same-budget C=0 monitor | 121.75 | −12.87% |
| Calibration-selected fixed C, K=3 | 137.72 | −1.44% [−4.05%, +0.48%] |
| K=2/C=0 | 132.87 | −4.91% |

Intervals resample whole conversations, with only six synthetic
clusters. The live learner was −0.65% versus its monitor control
([−1.88%, +1.01%]), so this run does not show a benefit from the
learner's stopping decisions after paying the observation/controller
cost. The learner applied a nonzero C on 27/48 turns and made 20
full-offer probes, yet offered **2.979 drafts per round** against
K=3/C=0's 3.000. Its trace and controller occupied **37.49 s** of
297.12 s complete native request time. The calibrated fixed cutoff
actually shortened to 2.373 drafts per round but also lost tok/s;
its output was 4.0% shorter and aggregate wall time 2.6% lower.
Neither shorter drafts nor higher acceptance alone was the score.

All arms returned fenced, parseable final code on 48/48 heldout turns.
One valid-domain function check per request passed 47/48 for live C,
46/48 for K=3/C=0, 48/48 for calibrated fixed C, 46/48 for the
monitor, and 47/48 for K=2/C=0. That narrow probe does not establish
general code quality. There were **zero matched exact-loop
exclusions**. [RESULTS.md](RESULTS.md) retains output/wait ratios,
per-length rates and calibration numbers; [RESULTS.json](RESULTS.json)
retains every C decision, controller time, native offered-round
histogram and cached-prefix digest. [QUALITY.json](QUALITY.json)
contains the bounded function checks.

The calibration cost table measured K=1/2/3 eligible rounds at
14.067/15.717/17.486 ms. A hindsight same-tape stopping oracle was
+9.12% versus K=3/C=0, but it assumes future sampled trajectories
do not change when a round stops early. The selected fixed cutoff
C=(0.434978, 0.850344) had a +1.99% *in-sample modeled* gain and
lost 1.44% on heldout native requests. The oracle is an optimistic
bound on that replay, not a demonstrated online opportunity. A
separately flagged phase diagnostic on warm turns 2–8 used 699
rounds at K=3/C=0, 906 with fixed C and 881 with live C. Outputs
also differed, so these phase totals diagnose mechanism rather than
provide another speed score.

The corrected research source offers a sampled low-confidence token
before stopping the next draft slot. A native CUDA two-token fixture
gave target-like frequencies, and a seeded positive-C twin matched
eight token tapes. [EXACTNESS.md](EXACTNESS.md) details the rule and
limits. These tests do not qualify every full-model sampled path or
a production override. Memra #673 remains the sampled-correctness
and serving-configuration gate. No active Memra engine default,
fleet pin or customer route changes in this research.

The [sealed scientific receipts](receipts-v3/manifest.json) contain
3,658 native members, the exact source and a byte-level manifest.
Their compressed and expanded public-boundary review is held in the
private custody lane. Hosted replay must independently regenerate
the cost table, oracle, code probes and this result before the PR
can be reviewed at its final head.
