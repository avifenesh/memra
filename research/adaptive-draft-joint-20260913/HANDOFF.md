# Adaptive draft learning: theory, evidence and rig handoff

Updated 2026-09-14. Branch: `lane/adaptive-draft-joint-20260913`.
Starting engine: `3bb21381848067d922dec1320261846f99ceb29a`.

## Current position

The existing head, confidence and depth mechanisms each improved request time
when measured separately. The proposed learned replacement policy has not yet
beaten a strong simple baseline. No combined controller or gradient-trained draft
head has been measured. The immediate next experiment is denser training for the
admission learner, followed by a second untouched test.

The owner requires every research target to be measured alone before combining
them. Continue through those gates without stopping merely to report a completed
stage. Preserve losses and failed attempts. Do not turn a negative admission
result into a joint-policy measurement to hide it.

The rental retry was stopped at the owner's request to push and continue from
the rig. Destruction was confirmed. It produced no new model measurements:
SSH rejected the host's `authorized_keys` ownership/modes, including after a
startup-script update. Earlier replacement attempts encountered DNS/SSH failures
and unavailable offers. Provider identifiers and raw operational logs are private;
scientific logs and checkpoints are committed in this branch.

## Research question and terminology

Can learning which draft vocabulary rows to retain, calibrating confidence after
those changes, and choosing how far to draft improve **completed tokens per unit
time**, beyond the strongest independent simple controllers?

There are distinct meanings of learning here:

1. Existing append-only learning inserts original vocabulary rows observed during
   verification into spare capacity. Neural weights do not change.
2. The current research fits utility tables or a small contextual regression to
   choose an admission/replacement action. Original Q8 row bytes remain unchanged.
3. Gradient training of a draft projection, adapter or MTP block is a future,
   separately registered experiment. It has not been performed in this lane.

The target model and its verification program stay fixed. Confidence controls
whether to spend more draft compute; it never relaxes target verification.

## Working theory

For a fixed state and draft horizon K, let s_i be the probability that the first
i draft proposals are all accepted. Away from EOS and output-budget truncation,
a conventional speculative block emits an expected `L(K) = 1 + sum(s_i, i=1..K)`
tokens, including its correction/bonus. Let C(K) include draft work, verification,
rollback, row maintenance, policy inference and amortized learning cost. The
long-run objective is `rho = E[L] / E[C]`, under an appropriately stationary
workload; it is not the average per-block ratio or accepted/drafted fraction.

A proposed extra draft step is worthwhile when its expected additional committed
tokens exceed `rho * additional_cost`. Increasing acceptance can therefore make
longer drafts worthwhile, but does not guarantee that they are: verification
cost, memory traffic, batch shape and controller overhead also change.

The owner's confidence proposal is plausible for the same reason. After useful
head learning, some proposals below the former cutoff may become worth trying.
However, softmax confidence on a trimmed head is conditional on that head's
support. Adding rows changes its denominator. A lower raw confidence can coexist
with better target agreement. Lowering the threshold mechanically whenever
acceptance rises would confuse these effects.

The proposed acceptance model should instead estimate conditional prefix survival
from confidence/margin, draft position, active support and head version/change
features. Train and calibrate it on separate data. Freeze a head within each
evaluation block when testing confidence alone. A head-aware predictor must beat
a shared predictor and a calibration-selected static threshold before its extra
complexity is credited.

For row selection, frequency is not the same as value. A frequent row that never
changes an argmax may do nothing. A replacement can also remove a useful winner.
The present one-step label is
`delta = I(new_proposal == target) - I(old_proposal == target)`
after replacing one fixed slot, using the same preserved draft hidden state.
This measures immediate correctness change. It does not price an update or predict
the changed hidden states later in a real speculative block. A deployment policy
eventually needs complete-action value, including eviction and runtime cost.

Bounded exploration is needed to learn below an old confidence cutoff or beyond
an old horizon. Record action probabilities and actual costs. Label only reached
positions through the first rejection. Later verifier suffixes can supply causal
candidate hints to a subsequent block; they are not actual-continuation labels.
For sampled decoding, keep an already drawn proposal and decide whether to draw
the next one; qualify the exact sampling/verification algorithm independently.

The intended contribution is a cost-aware learned policy that responds to changes
in the draft head and earns a measurable advantage over matched simple policies.
Append-only learning, recency/frequency selection, confidence stopping and
accepted-length depth adaptation are baselines, not new inventions. Novelty has
not been established, and no literature-exhaustiveness claim is made here.

## What has been implemented and measured

Exact artifacts are in `artifacts.lock.json`: Gemma 4 12B IT QAT Q4_0 plus its
matching Q8_0 assistant, and Qwen3.5-9B NVFP4 with embedded MTP. **Gemma E4B is
excluded.** Quantization or artifact substitutions require fresh qualification.

| Stage | Completed evidence | Interpretation |
|---|---|---|
| Native correctness | Fixed a Gemma 96-token request returning 97 tokens; strengthened full-length comparison; 21/21 rerun checks passed | Real correctness repair, not just a benchmark-harness change |
| Own-generation ranks | Gemma 22,528 and Qwen 22,525 training tokens; locked rank artifacts | Passed the initial corpus floor for 4,096 rows |
| First acceptance predictor | 6,539 labels; held-out Brier 0.1796074 head-aware vs 0.1793217 shared | Head-aware predictor lost narrowly; joint control was not tested |
| Existing head mechanism alone | Request time -4.41%, descriptive interval -6.15% to -2.58% | Append-only learning helped this workload |
| Existing confidence mechanism alone | Request time -4.59%, interval -6.25% to -2.92% | Fixed cutoff 0.7 helped versus disabled stopping |
| Existing depth mechanism alone | Request time -4.14%, interval -5.77% to -2.14% | Accepted-prefix+1 helped versus fixed K=4 |
| Frozen full-row oracle | 391 reached states; 41 of 77 missing targets had potential insertion repairs; zero overlapping score error | There is replacement opportunity, with full-probe cost +7.83% |
| Physical single-slot replay | 7,041 interventions; zero nonambiguous mismatches or score error; 41 positive, 9 negative, 6,991 neutral | The frozen-state single-swap calculation matches native replay |
| Bounded candidate discovery | 32 rows; discovery 9/41; observer cost ratio 1.001956 | A cheap candidate pool misses much of the oracle opportunity |
| Candidate-source ablation | Earlier verifier-suffix hints raised discovery to 12/41; cost ratio 0.999964 vs committed-history hints | Modest recall improvement, no clear incremental cost difference |
| First fresh admission test | 24 prompts / six families, 179 states: highest-score 19 repairs, frequency 2, learned tables 0 | No learned admission model qualified for an online trial |

The three component timing studies used eight independent held-out prompt groups
and six balanced AB/BA repetitions each: 48 pairs per component, 300 runs total
including diagnostics/warmups, all with exact output identity. Same-capacity heads
had 4,096 core rows plus 512 spare rows. Their gains must not be added or multiplied
to predict a combined result. These are greedy single-request measurements on
one RTX 5090, not sampled serving throughput or concurrency results.

Physical replay used 391 deliberately ambiguous duplicate-winner fixtures in its
7,041 interventions. Matching a frozen hidden state does not qualify continuing
the rest of an altered block from an old hidden-state suffix.

Bounded discovery used up to 24 recent unique outside-head IDs from a 256-ID
history, then deterministic exploration to fill 32 candidates. Its paired cost
study and the source ablation each used 48 balanced pairs with 250 ms telemetry.
The suffix source won the **old calibration** discovery comparison, 7 versus 5,
and was fixed before the first fresh test. It supplies hints only from completed
earlier verifier blocks.

## The negative admission result and the next registered test

The first admission checkpoint was frozen before fresh prompt generation:
`7fde7a366117ed993c0d7f97462366dece2711190204e1da7079d75f88526472`.
All 48 full/bounded requests matched their complete 128-token plain output.
An independent re-audit reconstructed causal candidate pools and compared 136
native scores with zero error.

Highest-score made 24 actions: 19 positive, zero negative. Frequency made four:
two positive, zero negative. Both learned per-ID tables made zero actions.
The contextual ridge model was **not fitted**: training had only six nontrivial
interventions, below the registered minimum of eight. This is insufficient data
for that model, not a trained-context-model loss.

`DENSE-ADMISSION.md` changes only collection density: sample every reached block
on the original 16 training and 16 calibration prompts, rather than every 16th
block. Hold the 4,608-row live head frozen, candidate source fixed, K=4 and both
confidence cuts at zero. Full and bounded probes remain separate requests.
Dense diagnostic cost is not the previously measured normal observer cost.

Refit the existing utility tables and eight-feature ridge model with the same
weights, penalties and threshold grid. Do not change the model/grid after looking
at the new test. Freeze `admission-dense-model.json` before generating the second
test: 24 prompts across chronology, conversion, grammar, taxonomy, comparison and
reference families. Treat four cases within a family as correlated. The first
fresh test remains historical test evidence and never becomes training data.

Promote to an online trial only if a learned policy beats **no-op, frequency and
highest-score** on the second test. Abstention is not improvement. This gate
selects what to test next; even a pass does not establish serving benefit.

The dense cadence flag, wrappers and evaluator are implemented but **not compiled
or GPU-qualified yet**. No dense dataset, dense fitted checkpoint or second-test
model output exists. First task on the rig: build and qualify this exact source.

## Plans after the dense admission gate

1. If admission qualifies, test eviction alone with a fixed admitted-candidate
   stream: FIFO, LRU, frequency and complete-swap utility at identical capacity,
   update counts and residency constraints. Include removed-row value.
2. Test actual online head-only behavior on continuing streams and domain switches.
   Restart each arm from identical saved state; execute changed blocks from their
   real initial state. Keep confidence off and K fixed. Include copying, scoring,
   learning, persistence and rollback costs. Compare the strongest simple policy.
3. Test learned confidence alone on pre-banked head states, with maximum depth
   fixed and accepted-length adaptation disabled. Preserve the first predictor's
   negative result. Collect the exploration needed to evaluate lower cutoffs.
4. Test learned depth alone with frozen rows and confidence disabled. Compare a
   calibration-selected fixed K and accepted-prefix+1; price the marginal draft
   and verification work. Record exploration probabilities.
5. Only after separate verdicts, measure pairwise interactions, then the full
   factorial head/confidence/depth study and proposed joint policy. Use strong
   independent-controller baselines and untouched evaluation data.
6. Qualify Qwen's own row-update path separately; current Qwen evidence covers
   artifacts, correctness and ranks, not the Gemma adaptive-row experiments.
   Gradient-trained draft weights are another separate future arm, with frozen
   target, explicit optimizer/data/checkpoint lineage and the same isolated gates.
7. Serving qualification follows: sampled correctness, session persistence,
   concurrency, TTFT/E2E/TPOT/ITL tails, request/token throughput and memory/cost.

If the denser learner loses, report that result and do not install an online
learned arm. A new candidate source, capacity, feature model or neural adapter
requires a new registered experiment and fresh evaluation, not retuning on this
test. Negative runtime doors are retired according to `CLAUDE.md`; explanatory
diagnostics remain subject to their documented decision dates.

## Continuing on the rig

Use an isolated checkout of this branch and a designated non-serving RTX 5090.
The existing Python harnesses use the fixed root `/workspace/adaptive-draft`.
Place the checkout there as `memra`, or make that path a symlink to the isolated
worktree. Do not replace an existing workspace, model directory or experiment.
The canonical single-5090 lock is `/tmp/memra-5090.lock`.

Prerequisites: working CUDA 13.1+ development toolkit/driver, Rust with rustfmt,
Python 3, git, curl, ripgrep, and the repository build dependencies. Run
`cargo fmt --all`, inspect and commit any formatting changes, then invoke:

```bash
bash research/adaptive-draft-joint-20260913/resume-dense-on-rig.sh
```

The script refuses a dirty checkout or pre-existing dense/test outputs. It
verifies the CUDA device, stages the three immutable artifacts (existing correct
files are reused), builds the native gate/tokenizer, checks formatting and flags,
binds source/binary hashes, copies the frozen first checkpoint from committed
receipts, and runs dense collection -> audit -> fit -> freeze -> fresh2 -> audit.
It does not start confidence/depth combinations or alter serving defaults.

If the script fails after collection, preserve the failure and resume the failed
stage manually from its log; do not rerun it by deleting outputs. Checkpoint and
prompt writers deliberately refuse overwrite. Before any further policy fitting,
bank the raw files, checkpoint, prompts and hashes into this research directory.
Cost comparisons require at least five balanced paired repetitions, uncertainty
at prompt/stream level, failure logs, and 250 ms telemetry. Single collection
passes do not substitute for performance experiments.

## Evidence map

- `RESULTS.md`: first correctness repair, rank corpus and predictor loss.
- `ISOLATED.md`, `ISOLATED-RESULTS.md`, `isolated-receipts/`: component protocols,
  timing results and raw comparisons.
- `ROW-ORACLE-RESULTS.md`, `row-oracle-receipts/`: full oracle and original splits.
- `CANDIDATE-RESULTS.md`, `candidate-gates-receipts/`: physical replay, bounded
  discovery, source ablation and first fresh admission; 1,220 manifest-bound files.
- `ADMISSION-PLAN.md`, `DENSE-ADMISSION.md`: fixed fitting and promotion rules.
- `ROW-POLICY-GATES.md`, `NEXT-ISOLATED.md`: detailed remaining isolated gates.
- `STATUS.md`: chronology. `DENSE-SETUP-FAILURE.md`: infrastructure failures.
- `docs/FLAGS.md`: diagnostic controls and decide-by dates (2026-09-27).

No branch merge, release, runtime default change or production deployment is part
of this handoff. Push this research branch, then continue and bank evidence on it.
