# Row-swap headroom, pre-registered 2026-09-13

The original DESIGN.md calls for an optimistic one-swap oracle before implementing
the learned replacement controller. Existing occurrence/recency replacement is
prior art; merely implementing it would not test the proposed contribution.
This stage measures whether recoverable head-selection mistakes exist and how
expensive a deliberately complete probe is. It is not the learned policy itself.

Keep the same locked Gemma 12B/assistant, 4096 core + 512 spare rows, K=4, both
confidence cuts zero, head updates frozen, one-shot eager greedy. Fill all 512
spares beforehand using the first distinct out-of-core token IDs found by the
native tokenizer in the 88 training prompts, in sealed file order. This imitates
an exhausted append-only head built from a code workload. No calibration or
held-out token is used to populate it. Refuse an insufficient seed rather than
silently inserting synthetic rows or shrinking capacity.

At rounds 1,17,33,..., score the same draft hidden vector with the actual active
head and a full native Q8_0 head. Record only reached draft positions (accepted
prefix and first rejection), excluding proposals beyond the requested output
budget. Later draft suffix states have no actual-continuation label.

Measure: missing targets that an insertion can repair; missing targets it cannot
repair; errors repaired by evicting a unique mutable wrong winner; complete
insert/evict rescues; high-scoring outside distractors that would hurt a correct
proposal. Core rows are immutable. A low-scoring duplicate core row is a legal
greedy filler for removal at constant capacity. Tied/near-tied cases are excluded
by twice the maximum overlapping score difference plus 1e-5. Check mapped argmax
agreement explicitly; exclude numerical disagreement from oracle summaries.
This is a conservative *estimate* from a full-width probe until exact replacement
replay is qualified. A full-head scoring kernel may use a different reduction.

The verifier reveals the missing target at the measured state. This is privileged
oracle information, unavailable to an online policy before paying verification.
It therefore gives one-step headroom on sampled states, not an achievable speedup,
an unbiased all-state estimate, or a complete counterfactual draft trajectory.
Top-16 outside candidates are selected by score and have no population interpretation.

Data: eight code prompts from each sealed split; heldout indices [14:22] are fresh
after earlier pilots. Add eight independently authored prose tasks per split to
probe a domain switch; these are synthetic research prompts, no customer data.
Train/calibration/test results remain separate. No tuning from held-out results.

Before recording: native whole-output probe ON/OFF check and a red admission test
that must refuse an unfrozen head. Then collect one traced request per prompt,
with its trace-free counterpart. Six balanced ON/OFF pairs per calibration prompt
measure complete diagnostic overhead on eight code groups. Include full-head
allocation/upload, projection, readback and CPU/file work in request time. The
internal overhead counter is explanatory; paired request time is authoritative.

Bank every raw output, sparse oracle JSONL, initial tail, prompt bytes/hashes,
commands, binary/source hashes, failures and 250ms telemetry. All comparisons
retain exact full output equality, not matching prefixes alone. End by recording
what is actually supported and which estimator/physical-swap gate comes next;
no confidence/depth learner or joint-policy experiment is admitted by this stage.

## First offline estimator, specified before collection

For each state, use only the scored top-16 outside candidates (chosen without
looking at the verifier label). Consider replacing the first mutable slot, index
4096, with each candidate. Its one-step label is changed greedy correctness minus
existing correctness, in {-1,0,1}; discard ambiguous numerical/tie cases. The
fixed victim deliberately isolates candidate admission from learning eviction.

Fit per-candidate mean utility from training exposures with four zero-utility
pseudo-observations. Unseen candidates have utility zero. Pick the highest-utility
available candidate, with token-ID tie break, only above a threshold. Select that
threshold on calibration prompt-mean utility from {0,0.01,0.025,0.05,0.1,1.0}; ties
prefer the higher threshold. Freeze before held-out evaluation. Compare with no
change, highest-scoring outside insertion, and a training-only target-frequency
selector (no action when no available candidate was a training target). Evaluate
all policies on the same states, excluding any state with an ambiguous candidate
comparison. Save counts, scores, threshold and
all split metrics. This is a first table estimator, not a novel algorithm or a
competitive replacement-policy verdict. It excludes candidate-probe cost and
does not replay changed draft suffixes, so it cannot establish inference speedup.

Prose inputs contain related task-format variants across splits; report them as
synthetic within-family diagnostics. They do not establish generalization across
unseen task families. Fresh family-grouped data is required for a final policy test.
