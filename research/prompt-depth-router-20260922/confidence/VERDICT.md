# Qwen code K=3 with confidence stopping

This verdict covers the fixed-cutoff baseline. It does **not** answer
the owner's non-hardcoded C question; that live native test is tracked
in [adaptive-v3/PROTOCOL.md](adaptive-v3/PROTOCOL.md).

**Decision: keep K=3/C=0 as the research control; do not promote the current
confidence cutoff or an online C learner.** The [sampled exactness
audit](EXACTNESS.md) found that positive C discards sampled low-confidence
picks before target verification. This token-dependent censoring can
change the target output distribution. Positive-C rates below are
diagnostic measurements of that executed path, **not an exact-sampling
speedup**. On the pinned Qwen3.8-27B
NVFP4+Q5_K artifact and one non-production desktop RTX 5090, development
selected C=0.15 at +6.81% pooled code output tok/s versus C=0. The
preregistered code-only v2 held-out set measured only **+1.03%** pooled
(127.17 versus 125.88 tok/s, 12/24 paired wins). The 4K code cell was
**−4.98%** with a pointwise whole-scenario interval of [−6.85%, −1.76%];
the 16K cell was +23.57% with a wide [−5.13%, +54.31%] interval.
The 256- and 1,024-token cells were −0.68% and −1.98%, with intervals
crossing zero. These six synthetic task families do not show a stable C
gain across code prompt lengths. [Development](RESULTS.md) and
[held-out controls](HELDOUT-RESULTS.md) show every cell.

Fixed K=2/C=0 measured 120.91 tok/s pooled on the held-out code
requests, below K=3/C=0's 125.88 tok/s. The selected C=0.15 arm measured
+5.17% versus K=2 but did not establish a useful gain over K=3/C=0.
All 24 held-out code pairs reached the requested final code format in
every arm; there were zero matched loop exclusions. Syntax and format
coverage do not prove the generated functions solve their tasks.

The selected arm returned 31,333 tokens in 246.387 complete-request
seconds versus K=3/C=0's 24,541 tokens in 194.958 seconds: **27.7% more
output and 26.4% more aggregate wall time** for +1.03% pooled tok/s.
The compared sampled streams can take different output lengths, so that
small rate advantage is not evidence of lower customer wait. The primary
metric still uses all returned tokens divided by complete native request
seconds, including tokenization, prefill, drafting, verification and
detokenization.

An exact speculative confidence policy should control proposal work
without changing the target's sampling distribution. The tested cutoff
does not meet that requirement. Any output difference could reflect
sampling bias; the format checks confirm only that code was returned,
not whether those functions pass task-specific assertions.

## What the confidence probes establish

The [frozen offline policy replay](OFFLINE-RESULTS.json) stitched only its
chosen measured native requests across 16 development evaluation requests.
It measured +8.92% versus C=0 but **−0.17% versus its calibrated fixed
C=0.15**, with two probes and zero C adoptions. It excludes policy CPU
time and full-information calibration cost, and it does not switch C
inside a live model session. It provides no learning-specific speed proof.
Because it chooses among positive-C arms with the sampled exactness issue,
it is also not an equal-distribution policy comparison.

The [post-result phase diagnostic](PHASE-RESULTS.json) compared C=0 with an
almost-zero cutoff that computes confidence but did not shorten any draft.
Their sampled token tapes and K=3 draft lengths matched in both orders,
with no loop exclusions. The confidence calculation added 0.85% to the
reported draft-phase time on eight 512-token code requests, while complete
request tok/s was essentially flat (101.84 versus 101.80). C=0.15
shortened 152/1,486 diagnostic rounds and measured 102.61 tok/s, about
+0.76% versus C=0 at that fixed-output shape. These phase-synchronized
runs diagnose cost; they do not enter the free-generation verdict. They
do not support attributing the held-out variability to the probability
read alone.

Memra's current C signal is the raw MTP probability of the sampled draft
token at temperature 1, while this study samples at temperature 0.7
with top-k 20 and top-p 0.95. [LEARNER-DESIGN.md](LEARNER-DESIGN.md)
records the next falsifiable step: calibrate conditional prefix survival
from verified offers and measure a cheaper score that may reuse filtered
sampler statistics. A result from this synchronous cutoff recipe does not
settle all confidence-aware drafting.

## Qualification and evidence boundary

- The sealed prefix-study runtime initially refused positive C in its
  depth-observation hook. That [unscored attempt](PROTOCOL.md) is preserved.
  The measured binary comes from the same source archive with a single
  reviewed guard change: positive C is admitted only with an explicitly
  fixed K ceiling; contextual depth learning remains refused.
- Target-only greedy identity and same-seed sampled reproducibility passed
  for every fixed-C arm at the study shape. Full 248,320-row Qwen target
  and embedded MTP heads engaged. Source, model, binary and GPU identities
  and 250 ms telemetry are retained with every scored run. Those gates
  do not establish sampled distribution parity across C settings.
- The first all-format held-out qualifier stopped at **7/8** coverage: 4K
  prose used the full 8,192-token cap in reasoning with no final answer.
  [Its failure](FAILED-QUALIFICATION.json) remains unscored. A new,
  [versioned code-only protocol](CODE-ONLY-V2-PROTOCOL.md) reused the six
  untouched scenario files and passed 4/4 disjoint code qualification
  cells before selecting C.
- The study uses independent native requests with fresh caches, one
  GPU, synthetic four-helper code tasks, and sampled 0.7/20/0.95
  decoding. It does not qualify vendor-default HTTP serving,
  concurrency, warm-session KV reuse, or a code-quality improvement.
  No served flag or default changes.

The complete raw scientific projection and exact source-patch proof are in
`receipts-v1/`. The [archive boundary](ARCHIVE-BOUNDARY.md) records a
passing private expanded scan at run `35808836373`; the independent Memra
hosted replay is a separate PR gate. The research pod was deleted after
its 4,076-file operator archive and manifest matched local copies; the
provider's direct list showed no remaining task pod.
