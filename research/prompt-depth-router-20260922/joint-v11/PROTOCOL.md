# Fresh non-code data for learned C/K/D

## Sampled-C exactness prerequisite

The archived v8 source tied to the pinned Qwen research binary appends
the sampled pick before fixed or learned C stops on all sampled paths
and rejects positive sampled PMIN0. Memra #673's mainline
chosen-token-discard counterexample does not apply to that binary.
This source fact does not itself qualify the full sampled target
distribution or a serving endpoint. The v10 run stopped prematurely
with only a partial diagnostic archive. A clean code-trained transfer
rerun is using this split's **training** prompts as a fresh 16-conversation
evaluation per topic, under workload SHA-256
`dd9fc45646931f66fee8a3b328b404d7da227765c58ce78f57536b3605fcbeb8`.
After that independent transfer test, its measurements may become
v11 training data. A byte-verified projection selects the required
native observations from the complete replayed v10 archive and
links to its custody. The v11 training host receives this projection
without the full benchmark source datasets that contain reserved
tasks. The v11 validation and final prompts remain unused.

This split was frozen while the first code-trained transfer test was
running, before it had final topic scores. Its purpose is to let the
clean transfer measurements become **training-only** observations
for a later controller, while keeping new prompts for selection and
final evaluation.

`workloads.py` pins the same official Google Research IFEval and OpenAI
GSM8K source bytes as v10 and excludes every v10 qualification and held-out
task ID. It retains v10's code-term filter and exact prompt wording.
GSM8K questions retain the same `#### <number>` instruction. Each file
contains eight distinct turns for native KV reuse.
The source commits are Google Research
`d36068b845da4c2b24927fee2cea1e6ef98dadda` and OpenAI GSM8K
`3101c7d5072418e28b9008a6636bde82a006892c`.
These custom continuing-conversation splits are internal task-quality
checks, not official standalone benchmark scores.

| Domain | Qualification | New randomized training | Validation | Final |
|---|---:|---:|---:|---:|
| Instruction following | 8 | 128 | 64 | 128 |
| Grade-school math | 8 | 128 | 64 | 128 |

The source manifest is SHA-256
`71c538295aeb970856f5feb5d11c888aa2dea4d220f41ab844d6e0587108a3df`.
The v10 exclusion manifest is SHA-256
`2142c62394fbb3f90c684a942e09a227f24f8a32d0011f550dd7b6841ed32dc1`.
The expanded prompt files and answer metadata are held privately in
Darklanes `joint-v11-private-receipts/workloads/`.
That full split remains local during training. A deterministic
training projection, SHA-256
`655223c8e4ca61fab179f7a4708b152ec35cf351209ba5210c10956080075a58`,
contains only qualification and training prompts and is the sole
workload input in the training seal. Validation projection SHA-256
`e8e60712c7053d9f64eba42b3f9a7b04ad946971fe095364cabb163b6c73eacd`
is released only after the policy weights and arms are frozen. The
full final manifest and final prompt files are released only if
validation selects a learned arm. Each projection commits to the
original full split and to the unreleased final group hash; native
replay checks the phase receipts. The nonproduction rental has a
ten-minute fail-closed deadline for each phase release.

On the accepted research GPU, `pilot.py` exercises one qualification
conversation with randomized draft K and one with randomized D/C
before the long training battery. It checks staged source syntax,
gold-answer parsing, full-head execution and later-turn native KV.
These two diagnostic sessions are sealed and replayed but do not enter
training, validation or final throughput rows.

The v11 training lane must label v10 fixed-arm K and D outcomes as observed
fixed actions. The v10 evaluation files do not contain per-offer confidence
traces, so their aggregate acceptance counts cannot supply C labels.
Collect balanced fixed-K arms, a randomized per-turn draft-K schedule,
and randomized D assignments with per-offer C traces on the new
training split before fitting
action-dependent costs and confidence.
Selected policy actions do not stand in for counterfactual randomization.
Seal and replay all 224 training sessions before deriving C/K/D rows;
the extractor reads that immutable archive and the custody-linked v10
training projection, not loose mutable result directories.
`training_supervise.py` freezes the rental, source, model, binary and
workload metadata before its first native training output, runs the
balanced collection, then seals and replays the raw sessions. On the
same rented GPU it extracts training rows, fits validation candidates,
runs qualification and validation, freezes the selected final arms,
and runs the final split only if validation selects a learned arm.
It then seals and independently replays the evaluation source,
native requests, task-quality checks, selection and score arithmetic.
The private closeout replays the archives again after copying their
bytes through the rig and back to the rented verification host,
then re-extracts V10/V11 training rows and refits the policy weights
using source and code-training bytes extracted from the copied
evaluation archive. It regenerates each validation label-to-model
arm mapping before destroying the rental.
The training replay checks the exact staged source and rental
metadata as well as native K, D, C and later-turn KV receipts.
Validation may select controller features and fixed thresholds. Final
IFEval and GSM8K tasks stay untouched until those choices are sealed.
Before choosing a prompt-level K router, check whether the first 16 and
first 32 tokenizer tokens expose enough task information.
`preflight_visibility.py` reads the sealed and replayed randomized
training archive, holds out whole conversations within its training
split, and must run before the validation arms are frozen.
`visibility.py` later reports separate accuracy on the reserved
validation fixed-K=20/D=3/C=0 arms. Both read only native user
prefixes; topic accuracy is not evidence of a throughput gain.
Compare a router against the best fixed K, including K=3, rather than
a weak no-op alone.
The training-only C calibration audit separately compares
proposal-probability-only, generated-history, first-16/first-32
user-token and combined prompt/history features by topic, K and
offer position. Its prompt-feature result is a signal check; the
pinned native C policy still uses generated history and offered q,
so no prompt-conditioned C throughput claim follows from it.

`fit_mixed.py` prepares non-code-only, mixed, and historically augmented
linear candidates. The mixed candidates add v9 code training observations;
the augmented candidates additionally add older code K utility and
D/C acceptance labels.
Only the new v11 rental's randomized D rows determine D/C marginal
time costs, and only its fixed K=20 rows determine the runtime
reference rate. Earlier K utilities use source-specific
complete-request rates. All native candidate choices still require
fresh validation against fixed controls.
`arms.py` requires the training-only prefix preflight and bounds native
validation to 19 frozen arms: seven fixed
K/D/C controls, three C/D candidate sources and their model-running
no-ops, plus mixed joint and mixed/non-code K-only controllers with
their no-ops.
The three fixed C cutoffs are training-only proposal-probability
quartiles. None may be selected using final task output.
`select.py` retains all seven fixed controls on final prompts. It
chooses at most one quality-eligible C/D source per topic from
validation and carries its no-op; the mixed joint and K-only arms
carry their no-ops if eligible on either topic. Each carried learned
arm must have a positive paired validation point estimate against both
its own no-op and every quality-eligible fixed control in that topic.
If none does,
`selected.json` records a validation no-go and no final arm manifest
is written. The validation score,
selection receipt, and final arm manifest stay together so final
native requests can verify the exact selection lineage.
For each topic, validation also names at most one primary learned arm.
The final score's confirmatory decision applies only to that frozen
arm; other carried options remain descriptive. A primary arm needs a
positive lower 95% paired interval bound against its no-op and each
quality-eligible fixed control on the fresh final conversations.
At most two topic-specific primary claims are tested. Each lower bound
is the 2.5th percentile of the paired bootstrap; treating that as a
one-sided 2.5% test gives an approximate 5% familywise bound across
the two topics. The descriptive arms carry no confirmatory claim.

Evaluation uses complete native request tok/s, paired by whole
conversation, with the same target decode, model-running no-op twins,
fixed K/C/D controls, task-quality and cap guards, loop exclusions, and
native KV receipts as v10. Report each topic separately. Acceptance is a
training observation and diagnostic, never the evaluation score. A new
serving default still requires the model-family and endpoint gates.
The same seed pairs prompts, but active C/K/D decisions can consume
different sampled draws and produce different output lengths.
Byte identity is required of model-running no-op twins, not of
active policies; report output/time ratios and task quality beside
tok/s so that sequence variation is visible.
The GSM8K quality check requires a complete numeric `#### <number>` on
the last answer line.

The earlier v9 learned-versus-best-fixed margin was about 0.3%, with a
paired interval crossing zero. Sixteen conversations per topic can
identify a larger transfer effect or a clear regression, but may leave
a margin that small unresolved. If so, report the uncertainty and use
additional independently frozen corpora before claiming either a
learned-policy win or a universal null result.
Open-ended chat and tool-call JSON are outside this split and require
their own quality oracles before a general-topic claim.
