# Qwen sampled C/K/D: corrected notation

Owner correction, 2026-09-23: **K is sampler top-k**, shared by
target and MTP proposal sampling. **D is the actual number of draft
tokens selected before a round**. **C is the after-offer confidence
stop decision**. The allocation ceiling is called `cap`, never K.
The earlier `joint-v4/` receipts use `k3` for cap=3 at sampler top-k=20
and temperature 0.7. They are development diagnostics only.

The primary outcome is pooled returned output tokens divided by
complete native request seconds, with total elapsed time, returned
output count, format and bounded functional coverage beside it.
Acceptance trains the C/D learners and explains behavior; it never
decides a speed winner. Changing sampler top-k changes the target
output distribution, so quality and output length must be compared
before a top-k setting can be selected. No serving decode changes
follow from this research protocol.

## Frozen cells

Use the exact full-head Qwen checkpoint and nonproduction RTX 5090
from `joint-v4/PROTOCOL.md`. For the corrected experiment hold
temperature=1.0, top-p=0.95, max_new=8192, ctx=65536, and default
thinking constant. Compare sampler top-k **3, 10, 20** with fixed
D=3/C=0 in complete eight-turn code conversations. Top-k=20 is the
Qwen thinking-mode recommendation; top-k=3 is the owner's proposed
code candidate; top-k=10 measures an intermediate setting. Rotate
arm order, hold sampled RNG seeds, and retain the exact output tape.
Include a separate top-k=20, temperature=0.7 bridge to the v4
diagnostics; never pool its timings or acceptance labels with the
temperature=1.0 cells.

Keep the six old v3 development topics as training and the remaining
three old v3 topics as model selection. The six `joint-v4/workloads-v4`
heldout conversations have not been used for model choice; reserve
them for the final scored battery. Its qualifier is separate from
those six.
Freeze randomized per-turn K schedules independently of model
sampling seeds before GPU output. Record actual sampler K on every
turn, the first 32 tokenizer IDs of the current user turn and native
cached/new tokens. A changed top-k invalidates the parked sampled
draft graph by its full sampling key; its recapture cost remains in
the request clock.

## Learned decisions

First measure whether top-k=3 passes the code-format, functional,
loop and native-continuation gates against top-k=20. If it does,
randomize D=1..4 under top-k=3 on the training conversations and fit
the in-process accepted-prefix and round-cost models there. Train C
from the same uncensored after-offer probabilities and measured
marginal draft cost. The v4 top-k=20, temperature=0.7 model is a
separate development reference; no weights or fitted costs are
transferred across decoding shapes without a matched check.

A request-level K selector may use only a bounded first user-token
prefix and completed prior-turn observations. It cannot inspect a
future target logit, future output token or later acceptance. Select
the tiny K/C/D policy and every hyperparameter on the three
development-selection conversations by complete native throughput,
subject to the output-quality gate. Keep the best fixed K/C/D vector
and fixed top-k=3 and top-k=20 controls.

Freeze the selector binary, weights, action mapping, decoding
parameters, prompt hashes and GPU source before opening any of the
six scored heldout conversations. Run the frozen policy, the best
fixed setting and matched feature-cost no-op on one nonproduction
GPU, with balanced order. Prove native checkpoint reuse and positive
cached/new tokens on turns 2–8. Report paired whole-conversation
uncertainty, output-token ratio, elapsed-time ratio, code probes,
loops separately, and actual K/D/C action distributions. A
nonpassing K=3 quality gate makes that action ineligible; the
learner must not hide a quality loss inside a tok/s gain.
