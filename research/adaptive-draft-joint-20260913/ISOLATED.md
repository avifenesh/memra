# Isolation-first component baseline, 2026-09-13

Owner requires each research target to be measured independently before combinations.
This campaign measures the existing append-only learning and stopping mechanisms.
It does not implement or claim the proposed learned row replacement or joint controller.
No pairwise or joint arm is scheduled by this runner.

## Controls

All three experiments use the exact Gemma 12B target/assistant in artifacts.lock.json,
greedy one-shot eager native Memra, 128 emitted tokens, K maximum 4, the banked
train-only 4096 ranks plus 512 identical duplicate filler slots, no initial sidecar.
Each request gets its own rank copy. No state crosses requests or repetitions.
The physical head has 4608 rows in every arm. Target and draft weights stay frozen.
Model load is outside request time; prompt learning, allocation, prime, decode and
sidecar persistence inside generate_spec_gemma are included in request time.

| Experiment | A | B | Held fixed |
|---|---|---|---|
| Head only | updates frozen | existing append-only updates enabled | K=4; both confidence cuts=0; depth adaptation=0; same initial rows/capacity |
| Confidence only | in-round threshold=0 | in-round threshold=0.7 | head frozen; depth adaptation=0; maximum K=4; next-round threshold=0 |
| Depth only | fixed K=4 | existing accepted-prefix+1 adaptation, floor=1, cap=4 | head frozen; both confidence cuts=0 |

Confidence stopping necessarily changes realized drafted length; fixed maximum K
is the control. Threshold 0.7 is a pre-existing engine setting, not selected on this
test set and not a learned controller. Head updates retain the existing prompt plus
all-verifier-argmax source, including counterfactual suffixes. Row replacement and
actual-continuation-only credit assignment remain separate future experiments.

Qwen retains its separate model/correctness qualification. These Gemma-specific
controls are not assumed to affect Qwen. Cross-family replication remains pending.

## Schedule and acceptance criteria

Use heldout source-file groups sorted [6:14] from the already sealed prompt manifest:
eight groups untouched by the earlier six-group pilot. Six repetitions per group,
balanced AB/BA order, paired adjacent executions. Deterministic random group order
seed 20260913. Each mechanism finishes and writes its own summary before the next.
One unscored warmup pair precedes each mechanism. Primary metric: paired request
time ratio B/A, including all learning cost within the request. Decode time and
acceptance are secondary. Startup wall time is retained separately.

Trace-free timing runs retain exact plain/spec token arrays and require complete
identity and exactly 128 tokens. Instrumented smoke pairs separately establish the
freeze control actually suppresses updates, both arms have 4608 rows, and controls
change drafted length. These trace times are excluded. Every failure is banked and
halts progression. Plain-vs-spec is a correctness reference, not the timed A/B effect.

N=8 independent prompt groups, six paired repeats per group (48 pairs/mechanism).
Average log ratios within group; bootstrap groups 10,000 times for a descriptive
95% interval. Report all per-pair values and per-group signs. Repetitions
measure repeatability but do not multiply independent prompt N. Compare token hashes across
arms and repetitions, not only against each run's plain reference.

Use a dedicated 5090, serialized lock, 250ms telemetry, archive raw at each mechanism
boundary. Inspect thermal/clock regime and repetition spread before interpreting a
small effect. A narrow, greedy code-review experiment cannot support serving speed,
sampled correctness, general workload or novelty claims. Untested policies and
pairwise/joint measurements remain pending even if these component baselines finish.
