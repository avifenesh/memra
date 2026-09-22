# Calibrated fixed depth versus ongoing MTP learning

Continuation of PR #569, frozen before scoring. Its original eight-set matrices
and results remain unchanged. This implements the first experiment in
`../mtp-learned-depth-20260920/NEXT-EXPERIMENTS.md`; periodic-probe ablation is a
separate later experiment.

Reconstruct the published prototype from base
7326f0e176326bb9b445720068cc502ea132ffad and its hash-pinned patch. Verify all eight
runtime blobs against the publication manifest. The only follow-up runtime edit
adds a `fixed:K` argument to the two research binaries; the learner, forwards,
sampler, timing instrumentation and cold priming remain unchanged. Preserve the
portable source patch, source/patch/archive hashes and actual binary hashes.

## Selection and evaluation

One regime: technical code review and implementation planning. Each file contains
eight continuing user turns with native chat templates and actual generated
history. Use two calibration source pools and two disjoint held-out source pools,
locked by source/excerpt/workload SHA-256 in workloads.lock.json. None contains
the previous learner implementation or the previous study's round-stream source.

- Qwen: all fixed K=1..7 on each calibration pool, ascending order on the first,
  descending order on the second. Gemma: the same procedure for K=1..5.
- Select one K per model by pooled output tokens / request-E2E seconds over both
  calibration conversations. Exact ties choose smaller K. Write the immutable
  selection and input-receipt hashes before executing any held-out request.
- Evaluation has five arms: original fixed K=3/Qwen or K=5/Gemma; native adaptive;
  instrumented native adaptive; calibrated fixed K; unchanged learned policy.
- Ten paired sets use the five-treatment Williams schedule and its reversals.
  Every arm occurs twice at every position; each ordered predecessor pair occurs
  twice. Alternate held-out pools A/B, giving five sets per pool. Every arm in a
  set shares the same sampling seed. Keep all completed uncontaminated sets.
- Qwen runs one eight-turn conversation; Gemma runs two, resetting history and
  learner between them, exactly as the previous study. Primary rate pools output
  tokens / summed request E2E time. Warmup and inter-request file writes stay out;
  tokenization, cold priming, generation and on-path learning remain in the timer.

Keep pinned NVFP4+Q5_K Qwen embedded-MTP and QAT Q4_0 Gemma plus QAT Q8_0
assistant artifacts. Full target/draft heads; H off, C=0. Temperature .7, top-k20,
top-p.95, output cap1024, context reservation49152. At least10s warmup and60s
measured per scored run. A shorter run is unscored and stops the campaign;
any protocol revision requires fresh comparable controls.

## Qualification and audit

Before each family's calibration: short and full-context greedy gates compare
native, measured native, learned, original fixed and every candidate fixed K.
Require exact prompt/output tapes and full-head engagement. Qwen completes before
Gemma begins. Use one dedicated RTX5090 32GB, the existing memra-5090 GPU lock,
250ms telemetry and interference detection. Do not pool old and new timings.
Record driver, configured power, actual clocks and artifacts for the fresh rig.

Audit all selected sets, frozen calibration choice, exact native/measured sampled
identity, counts, E2E sums, cold-priming state, depth bounds, fixed-depth engagement
and learner-schedule replay. Report every pair, pooled throughput, median paired
gain, wins and losses against original fixed, calibrated fixed and native. Round
cost/yield remains diagnostic; no counterfactual probe-free gain is inferred.

The conclusion answers whether Gemma's prior gain survives a calibrated static
control, and whether Qwen exposes unused static-depth headroom. It does not change
runtime defaults or claim HTTP serving performance. Preserve failures, archive and
hash receipts, and release the owned GPU at completion.
