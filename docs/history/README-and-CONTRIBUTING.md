# README and CONTRIBUTING history

The history file for the root [README.md](../../README.md) and [CONTRIBUTING.md](../../CONTRIBUTING.md).
Records moved out of either file on 2026-11-12 are kept here verbatim, except for the removal of
private deployment detail, each under a heading that names its source file. The current contract is
in the main docs; this file is not. A note in italics under a heading is added on 2026-11-12 and
says what in the record is stale.

## CONTRIBUTING.md: opening (scope and proof policy)

*Stale since: memra is in active development, CI runs the full CPU test inventory for external
PRs, and the pre-push hook requires native evidence for the cells a push reaches on each
push-gated target. The current policy is the top of CONTRIBUTING.md.*

Contributions follow the [current project scope](../../README.md#project-status): budgeted
research questions and maintenance for existing commitments.

Issues welcome anytime. PRs welcome **only when they carry proof**, per the rules below: CI is compile-only (no GPU runners on either target arch), so a human reviewer is the only gate between claim and merged code. Unproven PRs (no gates run, no numbers, "should be faster", AI-generated diffs with no on-device verification) will be closed, not debated. This is not gatekeeping: every accepted kernel becomes load-bearing in a correctness contract (see [Testing](../TESTING.md)), and reverting a bad merge costs far more than rejecting an unproven one.

## CONTRIBUTING.md: Required proof, in order (gate output and the Hopper lane)

*Stale since: `tools/validate-h100.sh` is retired; the current Hopper gate list is in
CONTRIBUTING.md "The three gates" and the [H100 card](../rigs/h100.md).*

Paste actual pass/fail output (or relevant tail), not "gates pass." A kernel that reduces in different floating-point order can flip an argmax at tight logit margins, which has silently broken "faster" kernels before (`research/tune-data/`), so a green run *right now, on your branch* is required, not an assumption.

Changes touching the Hopper (sm_90a) lane build from source (`MEMRA_CUDA_ARCH=90a`) and run its
gates directly on an H100 — kernel-check config pins (incl. the
KQRP, f16-mirror, f16g-sk, and batched-seqs pins), decode-batch (config + strict, gates 1–3 incl.
gate3c lean-logits), decode-dc, graph-decode, and graph-session. (These lived in the
`tools/validate-h100.sh` one-command battery until it was retired with the Hopper CI lane on
2026-09-02.) `ALL GATES GREEN` output
pasted, same rule as above. Gate1's config-mode verdict is the multi-seed fraction rule
(#47): FAIL iff >= 4 of 6 `MEMRA_GATE_SEED` draws diverge before step 3 (near-tie seeds
are legal FP dice, plumbing fails every draw); `MEMRA_GATE_CANARY=1` is the teeth check
when recalibrating.

## CONTRIBUTING.md: 1b. Perf regression battery (local CI)

*Stale since: the pre-push hook no longer reads a perf-battery row, and `MEMRA_SKIP_PERF_CI=1`
now refuses the push. Push evidence is per cell (`tools/cell_evidence.py`,
[RELEASE-QUALIFICATION.md](../RELEASE-QUALIFICATION.md)).*

```bash
tools/local-ci.sh --perf               # full cell battery (~15 min); --perf-quick = 31B subset
```

This is the drift detector correctness gates cannot be: it re-measures every published
model cell (plain AND spec, short AND depth) and, critically, records **speculative
acceptance and tokens/round** per spec cell, verdicting each against the rolling median of
its last 5 rows (`research/tune-data/perf-ci.jsonl`). FAIL = >3% tok/s drop or >0.05
acceptance drop. Acceptance drift is invisible to every exactness gate (decode and verify
shift *together*, bit-consistently) and silently cost this repo half its 31B short-spec
margin across ~40 green-gated commits in July 2026, hence the battery. Cells whose model
files are absent on your machine skip cleanly; set `MEMRA_MODELS_DIR` to your model root.
The pre-push hook requires a battery row newer than your newest engine-touching commit
(warn-only on machines without models; `MEMRA_SKIP_PERF_CI=1` overrides: say why in the PR).

## CONTRIBUTING.md: Scope

*Stale since: memra builds five targets (`120a`, `121a`, `100a`, `90a`, `89`); the current table
is in CONTRIBUTING.md "Build".*

This is a from-scratch engine hard-tuned for Blackwell sm_120a across two hardware classes:
RTX PRO 6000 Blackwell pairs for verification and final tuning (deployment detail removed),
and RTX 50-series for development and single-card tuning. H100 SXM remains the separately
compile-gated sm_90a lane. Existing 5090 boards remain 5090 receipts; they are not relabeled
as target-pair measurements. See [`docs/PERFORMANCE.md` §Rigs](../PERFORMANCE.md#rigs--what-was-measured-on-what)
and
[Installation](../INSTALLATION.md) and [Models and hardware](../MODELS.md) before proposing portability work: other GPUs compile
via the portable arch (`MEMRA_CUDA_ARCH=89` builds the Ada correctness-first eval lane) but
are untuned, and tuning choices throughout the codebase assume these two Blackwell classes
or the separately gated Hopper lane.
