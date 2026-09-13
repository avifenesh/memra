# Adaptive draft head, confidence and horizon — execution plan

Opened 2026-09-13. Owner: Avi / this research lane. Branch `lane/adaptive-draft-joint-20260913`; starting engine commit `3bb21381848067d922dec1320261846f99ceb29a`.

## Question and scope

Does head-aware confidence calibration and horizon control improve completed tokens per second beyond adaptive vocabulary selection with strong independent controllers? The target model is frozen. Stage one learns row selection and a small acceptance predictor; this is not target-model SFT. Gradient training of the draft weights is a later, separately identified arm, not silently mixed into the first comparison.

Models: **Gemma 4 12B IT, official QAT Q4_0 GGUF + matching 12B assistant**, and **Qwen3.5-9B NVFP4 GGUF with its own MTP block**. E4B is explicitly excluded by the owner. These are two different supported draft mechanisms, not two aliases of the same model. Exact artifacts are locked before scoring; alternate conversions require new qualification and are not substituted silently.

Hardware: one dedicated RTX 5090, 32 GB. Provider and rental details remain in the private operations record. Real CUDA allocation must pass before staging. No production host or local development GPU is used for research execution. A single campaign holds `/tmp/memra-5090.lock`. Other GPU processes invalidate timing windows.

## Ordered work

1. Provision, CUDA allocation, hardware/driver/toolchain receipt. Pin source and build hashes. Download exact artifacts and SHA256 every byte. Verify each draft's architecture, vocabulary map and checkpoint match.
2. Native correctness baseline: plain versus spec K=1..8 where admitted, real templated prompts, exact token lists and first divergence, nonzero acceptance. A refusing depth is retained as a refusal, not scored. External reference implementations are offline correctness/training tools only.
3. Recorder: collect every draft position's actual proposal confidence, margin, head version, position, accepted-prefix length, first rejection, normalization/support metadata and component timing. Later speculative suffix positions are censored for actual-continuation training. Instrumented runs are discovery; all timed arms receive identical instrumentation, and final throughput is also repeated without tracing.
4. Train-only own-generation rank corpus, minimum 4x head capacity generated tokens. Separate training, calibration, and held-out prompts by scenario group before collection. Initial pilot prompts are explicitly smoke tests and never become the final held-out benchmark. Keep checkpoints and lineage at each training boundary.
5. Fit a small conditional-acceptance predictor on training blocks and calibrate on the separate calibration split. Include head change, active support, confidence/margin, position and prefix survival. Compare log loss, Brier score and reliability curves. Persist weights, feature schema, seed, data hashes and training cost.
6. Factorial: static/adaptive head x fixed/calibrated threshold x fixed/adaptive depth (8 cells). The fully adaptive factorial arm uses independent controllers. Add head-aware joint control as a ninth arm. Freeze head capacity and target numerical program. Head versions change only at block boundaries.
7. Bounded extra-depth exploration collects evidence below the old confidence cutoff. Record action probabilities and cost. Any sampled gate keeps an already drawn token and gates only the next draw. Never weaken target verification or use greedy acceptance as proof of sampled correctness.
8. Same-window balanced AB/BA, at least five independent paired repetitions, randomized prompt order and shared pre-registered seeds. Include cold, warmed, and domain-switch profiles. Repeat at serving concurrency only after single-stream correctness. Select final policy on calibration data, then evaluate once on held-out data.

## Evidence contract

- `manifest.json`: experiment id, source/binary SHA256, toolchains, exact model source revisions and byte hashes, hardware shape, prompts hash, policy/config hash, seed, UTC time.
- `raw/`: stdout AND stderr before parsing; exact subprocess argv and allowlisted MEMRA configuration; exit code, start/end times, token outputs and mismatch positions. No secrets or customer prompts.
- `telemetry.csv`: GPU utilization, VRAM, power, clocks, temperature at 250 ms. Capture compute-process list at start and at failure. A failed process without an explicit cause remains cause unknown.
- `runs.jsonl`: one row per attempt, including failed and refused attempts; measured wall time separate from engine decode time. Acceptance is explanatory, never the primary objective.
- `trace.jsonl`: block/position proposal metadata and censoring; no claiming unobserved positions as rejection labels. Trace schema and actual observed coverage are versioned.
- `checkpoints/`: head row IDs and immutable base mapping, calibrator parameters, policy parameters, train/calibration data hashes, seed, training time. Qwen and Gemma never share learned ranks.
- `analysis/`: paired effects and confidence intervals at the independent-run level; no token-level pseudoreplication. Report N, thermal regime, bad/looped outputs separately, and every loss.
- Serving results require TTFT/E2E/TPOT/ITL p50/p95/p99, request/token throughput and sampled vendor-default behavior. A greedy CLI pilot is a mechanism/correctness result only.

Success means a reproducible cost-inclusive improvement over the independently adapted baseline with correctness preserved. A fixed-K comparison alone does not establish joint-control novelty. Negative results and the prior flat marginal-rate controller are retained and explained.

## Tracking and banking

`STATUS.md` is the human progress record. Raw files are append-only; retries use new IDs. SHA256 manifests seal each completed rung and receipts are copied off the GPU host after every rung. Keep model artifacts reproducible from immutable public sources or local byte manifests. Destroy this lane's rental when work finishes; confirm absence in the provider inventory. Never alter another lane's instance.

Initial scope is provisioning, reproducible baselines and recorder/training bring-up. No production deployment, public performance claim, runtime default change or release is implied by a pilot.
