# Qwen code K=3: live learned confidence

This continuation answers the owner's correction to the fixed-C study.
`C=0/0.15/0.30` and the offline replay with preset acceptance bands
were **controls**, not a measurement of a learned confidence policy.
Keep their immutable receipts in `../receipts-v1/`. This experiment
holds the full embedded MTP head and the code draft ceiling at K=3,
then learns when to stop offering further draft tokens from verified
feedback and measured cycle cost. The learned C values must be
computed from observed evidence; no list of candidate C values,
acceptance bands, step size or periodic probe cadence is supplied
to the learner.

## Pinned scope

- Model: `tiyuvta/Qwen3.8-27B-NVFP4-MTP-GGUF` revision
  `0f82b27dbb264b731e7d20f576582c871ef1969c`, file
  `Qwen3.8-27B-NVFP4-Q5K-mtp.gguf`, SHA-256
  `1facf36c2db359dcf9c2475cf8f85fe84a528d10aaaaff20f7c0db3d561e024a`.
  Do not train or alter target/draft weights or trim the 248,320-row head.
- Measured source starts from `../receipts-v1/runtime-source-confidence.tar.gz`
  SHA-256 `98a0a0118155663aa9abae29acdef845838b9367c6f6d17e87da7fe2fb1957a9`.
  Seal a distinct patch, source archive, binary and receipt namespace.
  No active serving default changes.
- One explicitly non-production GPU, real CUDA allocation before staging,
  sampled temperature 0.7/top-k 20/top-p 0.95 and default thinking.
  No local-rig gate, bench or smoke server.
- `workloads-v3/manifest.json` freezes one disjoint eight-turn code
  qualifier, three calibration conversations and six heldout
  conversations, with distinct seeds and exact user-prompt hashes.
  Use `max_new=8192` and `ctx=65536` for all native arms. The C=0
  qualifier must reach a fenced parseable final Python function on
  8/8 turns, with no exact output loop, before opening any heldout
  throughput result. A context/OOM or format failure stops this
  version; a changed budget needs a new protocol and corpus freeze.

## Correctness before learning

The sampled path currently draws a proposal, then discards a low-C pick
before verification; `../EXACTNESS.md` proves this can bias the target
distribution. In the new research source, keep the sampled
low-confidence pick as the **last offered proposal** and stop before
drafting the next slot. Its original filtered proposal `q` then
reaches the ordinary acceptance/rejection correction. Do not claim
zero-draft sampled rounds from `PMIN0` under this rule; refuse that
combination until a separate pre-draw policy is qualified. C=0 and
target-only paths remain unchanged.

Hosted CPU tests must enumerate the two-token cutoff, acceptance,
residual and bonus law, including an accepted-prefix case; verify the
same rule is reached in sampled chain-graph, single-head graph and
eager paths. On the research GPU require greedy target identity,
sampled reproducibility, full-head engagement, and a matched
sampled-distribution check on a controlled small-vocabulary fixture
before scoring Qwen. The fixture gate checks the **output law**, not
only that the same seed repeats.

## First establish an oracle and a cost control

Run uncensored full K=3 offers on disjoint calibration code
conversations. Retain the chosen proposal confidence at each of the
three positions, the target's first rejection position, committed
tokens, full native round/request wall and draft/verify phase timing.
A label is valid only when every earlier proposal in that round was
accepted. A target logit after the first rejected draft is never an
acceptance label. Match calibration prompts, seeds, source, model and
GPU with fixed K=1/2/3 cost controls, keeping their sampled output
length and time separate.

Calculate an optimistic cost-inclusive stopping oracle over the
eligible full-offer prefixes using the measured K-dependent cost
table. It may use hindsight acceptance to bound headroom, but it
must not be called a live policy result. Record its gap above K=3/C=0
and the best calibration-selected fixed C on the corrected source.
Candidate fixed C cutpoints come from the observed calibration
confidence values, not a preset grid. If no positive headroom
survives the recorded cost and uncertainty bound, publish that
negative oracle without selecting a heldout winner.

## Live learner and heldout comparison

Freeze the learner source and the disjoint heldout workload hashes
before the first heldout request. The learner starts from C=0 so its
first offers are uncensored. It updates from **verified, eligible
prefixes and committed output only**; estimates the conditional
chance the next offered slot pays; combines that estimate with
measured marginal draft/verify cost; and chooses the cutoff that
maximizes predicted committed tokens per complete round second.
Its candidate cutpoints are observed confidence values, including
C=0. Probe a full K=3 offer when the expected value of reducing
uncertainty exceeds its measured probe cost, and include that probe
time in the primary score. Every movement, unchanged decision,
sample count, estimated reward, controller CPU time and uncensored
probe is retained. State persists across the eight turns of one
continuing conversation and resets only between independent
conversations. The exact estimator, uncertainty rule and cost
formula are frozen in a versioned source commit before GPU scoring.

Compare live learned C against K=3/C=0, the best calibration-selected
fixed C on the **same corrected source**, and K=2/C=0. Include a
same-budget no-learning probe control if the learner explores.
Balance arm order and match prompt, seed, model and GPU. Primary
metric is output tokens / complete native request seconds over
requested-code turns; the clock includes controller and probe work,
tokenization, new-input prefill, draft, verify and detokenization.
Report pooled rates, paired whole-conversation uncertainty,
per-length rates, output-token and wait-time ratios, actual C
trajectories and format/functional code coverage. Exclude an exact
loop from every matched arm and name the exclusion.

Every turn after the first must prove native checkpoint/KV reuse
with `resumed`, `cached_tokens`, `new_input_tokens` and a stable
prompt-prefix digest; persistent learner state alone is cold
prefill. These native receipts do not qualify vendor-default HTTP,
concurrency or a served C. No positive-C serving policy moves
without the separate exact-stack gates in Memra #673.
