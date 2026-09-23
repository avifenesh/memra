# Sampled confidence stopping is not distribution preserving

The measured positive-C runtime samples a draft token, computes the
raw MTP probability of that sampled pick, and **discards the pick
before target verification** when its probability is below `p_min`.
The sampled chain graph, sampled single-head graph, and eager draft
paths all break before appending that token to `draft`. Target
verification and the ordinary full-accept bonus then see only the
shorter chain. The rejection test uses the original filtered proposal
probability `q` for retained tokens; it does not correct for the event
that retained tokens passed a token-dependent cutoff.

A two-token counterexample establishes the error without relying on
model statistics. Consider a proposal and target with the same
distribution, `q = p = {A: 0.8, B: 0.2}`, and a cutoff that retains A
but discards B. An offered A is always accepted because `p = q`.
When B is sampled, the runtime discards it and draws the target bonus
from `p`. It therefore emits

```
P_C(A) = 0.8 + 0.2 × 0.8 = 0.96
P_C(B) =       0.2 × 0.2 = 0.04
```

instead of the target's `{A: 0.8, B: 0.2}`. In general, if `S` is
the proposal mass of picks discarded at a position with `p = q`,
the current output law is `q(y) × retain(y) + S × p(y)`.
This is a proof about the stopping algorithm, **not** a measured
Qwen quality or drift estimate. The study's temperature 0.7 and
top-k/top-p proposal need their own matched sampled-distribution
gate after the algorithm is repaired. The zero-draft `PMIN0` branch
has the same token-dependent censoring risk.

Greedy target identity verifies argmax output only. Same-seed sampled
reruns verify reproducibility of each arm, not equality to the target
sampling distribution. Parseable final code likewise does not test
sampling parity or function correctness. The positive-C throughput
rows in `RESULTS.md` and `HELDOUT-RESULTS.md` remain faithful
measurements of the executed runtime, but are **diagnostic**; they
cannot establish an equal-distribution speedup over C=0.

The measured archive is `receipts-v1/runtime-source-confidence.tar.gz`
(SHA-256 `98a0a0118155663aa9abae29acdef845838b9367c6f6d17e87da7fe2fb1957a9`).
Its single research patch changes only admission of C with fixed
depth. The token-discard behavior is also present in Memra main at
the time of this review (`711be12c308c56bfc07abca4e9e4d521b14c8342`).
Memra issue #412 separately records the analogous chosen-token PMIN
bias in GLM DFlash2 and an archived pre-draw max-probability candidate;
that issue does not repair the MTP path. Memra issue #673 owns the
sampled MTP correction and served-configuration audit. Memra's
registered default C=0 path does not take the cutoff branch. The
`qwen4exp_gpu` gate has a different sampler: draft argmax is
deterministic and target rows are sampled directly, so this
rejection-sampling counterexample does not assess that gate. **Do not
promote this MTP positive sampled C or an adaptive C learner until a
distribution-preserving stopping rule and a sampled correctness gate
pass on the exact model and request shape.** A minimal candidate is
to verify the sampled
low-confidence pick as the final offered token, then stop drafting;
preserving zero-draft behavior needs a decision made before a pick
is sampled or an explicitly corrected proposal law. Both arms need
their own correctness and cost measurements.
