# Full-head MTP depth research

Date: 2026-09-20. Scope: one frozen code-analysis workload, one RTX 5090,
the pinned Qwen3.8-27B and Gemma 4 12B artifacts, with full heads in every arm.

Keep the existing runtime policies. Do not promote this new online cost learner.
Retire its experimental engine overrides while preserving the exact source and
all measurements. This is a decision about the tested implementation, not a
rejection of learned-depth research generally.

Qwen learned depth lost 2.45% against fixed K=3 (one win in eight pairs), and 3.42%
against native adaptation (zero wins). Gemma gained 7.74% against fixed K=5 (eight
wins), but only 0.43% over native adaptation (six wins; median paired change 0.36%).
The Gemma fixed-depth improvement is a valid result; the small incremental gain
over the existing native rule does not justify replacing it on this evidence.

Every arm keeps the same full vocabulary head. H learning and confidence cuts are
off. Cold priming is held constant across requests. Eight balanced paired sets per
model, 64 selected runs and 768 turns passed the audit. Interrupted sets were
excluded and retried in full under the original seed/order.

The next comparison should calibrate fixed-depth choices on separate data and
evaluate controller variants on held-out workload regimes, still holding the full
head constant. No fixed-depth optimum or cross-workload generalization is claimed.

See [results and exact receipts](../../research/mtp-learned-depth-20260920/RESULTS.md)
and [round-cost explanation](../../research/mtp-learned-depth-20260920/EXPLANATION.md).
The measured runtime is reconstructed by the archived prototype patch and blob manifest
from public base `7326f0e176326bb9b445720068cc502ea132ffad`; see the study source guide.
This publication adds research evidence and analysis, with no applied runtime change. The earlier DFlash experiment is separate.
