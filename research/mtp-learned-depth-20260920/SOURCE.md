# Reconstructing the measured MTP prototype

The original measurement commit `a37e30a6dd9637ece727c040f12a115dd6d96517` was
local to the experiment checkout. The published source does not depend on fetching
that unpublished commit: `prototype.patch` reconstructs every altered runtime file
from public base `7326f0e176326bb9b445720068cc502ea132ffad`.

`prototype-manifest.json` binds the patch SHA-256 and the exact Git blob/SHA-256
for all eight changed runtime files, including the tokenizer documentation fix.
The patch was applied to an isolated index of the public base and every resulting
runtime blob was checked against the measurement commit.

Use an isolated checkout of that base, apply this patch, and build `mtp-depth-study`
and `gemma-depth-study` on CUDA 13.1 / sm_120a. Use the runner and frozen workload
from this published research directory. Record the actual source commit and binary
hash of a fresh reproduction; do not relabel new results with the original local SHA.
The prototype is an archival artifact, not an applied runtime change in this PR.

Qwen's measured binary SHA-256 is
`4e20676ee9a6ef00f642728b93446cb948d261b077e59b2e503e24a3f7b3b6aa`.
Gemma's is `cf80a950a38a7cf528231b203704eedeb3f8bb3c378a0343521407d9b032237a`.
Both artifact locks are provided. Qwen uses embedded MTP and omits `--draft`;
Gemma requires its matching QAT assistant.

The exact runner is also preserved within each receipt archive and bound by its
identity hash. The default runner schedule executes all eight paired sets. Cycle
ranges preserve the original ordering and seed offsets when execution is partitioned.
`receipts/metadata/controller-v4.py` is the actual interrupted-run continuation;
use the generic runner for a fresh experiment, not that historical continuation.
