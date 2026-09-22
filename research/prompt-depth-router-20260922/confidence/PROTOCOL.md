# Qwen code K=3 with an adaptive confidence cutoff

This continuation keeps the completed prompt-prefix study immutable. Its
Qwen code comparison measured fixed K=3 against fixed K=4 with the confidence
gate off on independent requests; it did not measure a confidence policy.
The archived runtime source confirms the explicit fixed-depth policy overrides
the ambient `MEMRA_SPEC_ADAPT=1` setting in that experiment.

## Question

On the pinned Qwen3.8-27B NVFP4+Q5_K checkpoint with its full-vocabulary
embedded MTP head, does learning a **draft stopping** confidence cutoff while
holding the code draft ceiling at K=3 improve complete native request
throughput? The cutoff may stop a round after 0, 1, or 2 proposals instead of
offering all three. The target still verifies every emitted token. A cutoff
must not be used to change the target's acceptance rule or sampling
distribution.

The first controls are genuinely fixed K=3 with `MEMRA_SPEC_ADAPT=0` and
`MEMRA_SPEC_PMIN` at 0, 0.15, and 0.30; test the 0.30 cutoff both with and
without `MEMRA_SPEC_PMIN0`. The `pmin=0` arm does not pay the confidence
calculation. The positive-cutoff arms do. No arm changes H, the draft head,
sampling, or the output budget. Select any additional cutoff or policy only
from disjoint calibration requests before opening held-out results.

The sealed prefix-study engine refuses confidence cuts whenever its depth
observation hook is present. The first unscored qualification attempt hit
that guard and stopped. `patch_source.py` changes this guard only for the
`LearnedDepth::fixed` control, whose depth remains a K ceiling when confidence
shortens a round. Contextual depth learning still refuses confidence cuts
because its observations would be censored. The patched binary, patched
source archive and the rejected attempt receive distinct hashes and receipt
directories; no result from the rejected attempt enters the scored grid.

## Data and comparison

Use the six frozen synthetic helper scenarios and their exact user prompts,
seeds and 256/1,024/4,096/16,384-token length cells from the completed
prefix study. Score requested code separately at every length; retain the
interleaved prose requests as diagnostics. The six scenarios are paired
across arms. Rotate and reverse arm order before generation, and register
the order and source/model hashes before the first scored request. Use the
previously qualified 8,192-token cap, temperature 0.7, top-k 20, top-p
0.95, a fresh native cache per request and the same Qwen binary in every arm.
Each process loads and warms the model; the clock excludes loading and
warmup but includes tokenization, prefill, draft, verify, detokenization and
the cutoff's device and host work. Keep the local rig free of gates and
benchmarks.

For each arm retain commands, all `MEMRA_*` settings, binary and model
digests, exact prompt and output token tapes, requested-format checks,
round counts, total drafted and accepted tokens, whole-request seconds, GPU
telemetry and any failures. Prove K=3 in every fixed-control round and actual
confidence shortening in a positive-cutoff arm from the draft-length
histogram. An untimed diagnostic may add per-round attempted and kept
lengths. Greedy target identity at C=0 and positive C, the unchanged sampled
rejection-sampling path, full-head path engagement and real CUDA allocation
on a non-production GPU are qualification, not speed results.

Primary metric is pooled generated tokens / complete request seconds on
the code requests, with paired scenario gains and whole-scenario bootstrap
intervals. Report output length and latency beside throughput. Do not
count greedy loops as performance rows; preserve and name complete paired
exclusions. A cutoff that merely raises accepted/drafted while adding
verification rounds is not a win. The existing Qwen K=3 PMIN result on a
24 GB RTX 5090 Laptop and the full-head SGLang C-only loss on a RTX PRO 6000
are prior cautions, not same-hardware controls for this 32 GB RTX 5090 run.

## Adaptive continuation

Freeze an online C controller before held-out requests. It must carry state
across requests, update only after committed output, probe lower and higher
cutoffs, include probe and confidence-read cost, and report every C move.
Compare it with C=0 and the best calibrated **fixed** cutoff at K=3 on the
same held-out code cells. Add fixed K=2 with C=0 and Memra's native depth
adjustment as simpler depth controls; a confidence policy that only mimics a
cheaper fixed depth has not demonstrated a learning gain. Any change to graph capture or to the
rejection-sampling path needs its own exactness gate. A positive fixed C
alone does not establish a gain from adapting C; a negative fixed-C grid
does not prove every adaptive rule fails. No served default is changed by
this research.
