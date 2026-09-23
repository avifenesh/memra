# Qwen draft-only C/K/D study

Owner clarification, 2026-09-23: **K is MTP draft sampler
top-k only**. The target sampler stays at top-k=20 on every turn.
**D** is chosen MTP draft length (1–4) and **C** is the after-offer
confidence stop decision. `cap` names the allocated maximum draft
length. The completed `joint-v4/` cells used top-k=20 and
temperature 0.7 with an old draft-cap notation. The partial
`joint-v5/` cells changed target and draft top-k together. Neither
is the draft-only verdict. Target-20/draft-20 controls at
temperature 1.0 from v5 remain matched development baselines.

Use the pinned full 248,320-row Qwen MTP checkpoint and one
nonproduction RTX 5090. Keep target top-k=20, temperature=1.0,
top-p=0.95, default thinking, max_new=8192, ctx=65536, prompt
texts, and sampling seeds constant. Compare fixed draft K=3, 10
and 20. Verify that the native source passes draft K to draft graph
capture and draft-q filtering while target-p verification, boundary
and bonus sampling retain top-k=20. A changed draft K must rekey the
parked draft graph; a later turn must still prove native KV reuse
and positive cached/new tokens.

Freeze independent per-turn draft-K assignments before output. On
the six old development conversations, measure fixed K/D/C
controls, randomized D for each quality-eligible draft K, and
per-turn randomized draft K. Fit tiny K, C and D predictors using
only the first bounded user-token prefix, committed output tokens,
offered draft probabilities and completed prior-round outcomes
available at each decision. Never label an offered position beyond
the first rejection. K-specific acceptance and cost models cannot
reuse weights from target-changing or temperature-changing cells
without matched evidence.

Use the three remaining old development conversations to choose
one predictor and all hyperparameters by complete native request
throughput, subject to code-format, bounded functional and loop
gates. Freeze source, binary, model files, prompts and action
mapping before opening the six fresh heldout code conversations.
Compare the learned K-only and joint K/C/D policies with the
strongest executed fixed K/D/C vector and feature-cost no-op
twins, rotating order on the same GPU.

The primary score is pooled returned output tokens divided by
complete native request seconds. Report total elapsed time,
output-token ratio, paired whole-conversation uncertainty, actual
K/D/C actions, model time, cached/new-token receipts and code
coverage. Acceptance and per-round rate are training or mechanism
diagnostics. Exclude looped outputs from performance aggregates.
Changing draft K should leave the target distribution invariant
in theory, but a research load or fluent output is not a sampled
exactness gate. No served default moves without the separate
sampled, vendor-default endpoint and deployment gates.
