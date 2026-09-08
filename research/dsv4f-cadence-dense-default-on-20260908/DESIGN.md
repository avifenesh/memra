# Default-ON cadence and dense exact-tree transport

Integration starts from composition PR #371 source
`e591f451c21c05f0a80445cdd2d7cd7a301a8f9c`, containing cadence #368 and dense #366,
rebased onto `001c09e5d451798ef8d570f6087dc11527cbcc19`. The kernel arithmetic,
cadence variant bodies, paired capture/drain/fail-stop, refusal-before-commit,
live control layout and supported shapes remain unchanged.

This draft prepares both defaults ON. Root reports composition confirmation:
R1 +1.87%, reverse R2 +2.11%, identity true in both. No new numerical claim is
attached to this default-policy source. Merge additionally requires its narrow
source review and the environment-selected engagement gate described below.

## Selection and rollback

`MEMRA_DSV4_REPLAY_CADENCE`: unset (or anything other than exact `0`) selects
cadence when `arm_full_token_replay_for_gate` explicitly arms an admitted request.
Exact `0` selects the original full-forward graph. This is read at arming; no
retained graph changes in place. It does not automatically arm arbitrary eager
or serving requests. All existing admission remains, including on-device caches,
pos<512, plain device-sampler/diet TP2+EP and split-K OFF.

`MEMRA_DSV4_DENSE_EXACT_TAIL`: unset (or anything other than exact `0`) enables
exact-tree M=1 twins on each host thread before its first enqueue. Exact `0`
retains the original dense kernels. Existing shape/alignment/grouped/M>1 guards
remain. A thread-local gate override takes precedence. Set rollback environment
before worker creation and build fresh control graphs; a captured function is
immutable regardless of subsequent host selection.
This affects admitted eager DSV4 calls too, including admitted serving calls;
dense exposure is not restricted to explicit replay arming. Cadence alone remains
restricted to explicitly armed admitted replay. Neither fact is serving admission.

Both rollback seams carry decide-by 2026-09-22 for seam-removal review. Neither
is a provider/model setting. No fleet or serving-admission change is included.

## Fixed measurement programs remain fixed

The original sampled/full-replay instrument explicitly selects dense OFF before
model creation and selects its exact requested cadence mode independent of the
environment. Its original full-replay and profile helpers explicitly request
cadence OFF. The dense-only helper likewise explicitly arms cadence OFF.

Composition `PROGRAMS[0]` stays A=(cadence OFF,dense OFF) and `PROGRAMS[1]` stays
B=(cadence ON,dense ON). Its request arming uses an explicit boolean mode, and
its existing `set_dense` calls precede first capture. No arm silently inherits
the new defaults and no extra synchronization is inserted in scored paths.

CPU tests launch isolated subprocesses for all nine unset/0/1 environment pairs,
check the actual Rust arming-default policy and C++ host-thread default/override,
and prove that a fresh host thread gets the environment default rather than the
other thread's gate override. The legacy gate initialization is tested against
an explicit prior ON override. Composition's CPU test verifies its fixed A/B
programs and actual dense override. None allocates CUDA state or loads a model.

Hosted sampled/drift/composition CPU tests run with both defaults explicitly ON;
the subprocess test also covers unset and rollback. Remote formatting only; no
local rig build/test/lint/gate. Hosted CI must pass before this draft is ready.

## Environment-selected engagement gate

The composition binary adds `--defaults`, separate from its unchanged explicit
ABBA/BAAB modes. It reads the actual cadence environment policy and initial C++
dense thread-local value before any override, admitting only both variables
unset or both exactly `0`. It loads the vendor-default sampled shape, compares
all 256 steps against eager OFF, checks selected cadence variants/dense nodes,
AR epochs/cache/hidden/logits/tokens, then runs eight refusal cells. Fresh scored
state emits five sanity rows, first capture included. Two separate invocations,
unset then zero, provide ten rows total; no new speed claim.

The selected state calls the environment-aware arming API. After the eager oracle
temporarily selects dense OFF, the first selected capture restores the actual
C++ environment policy, not a hardcoded boolean. Subsequent graph replays need
no host selector update. Both modes retain exact source/binary/graph/control
hashes and full stdout/stderr/controller/process/lock readbacks. Root assigns
the shared development pair; no model starts before the source checkpoint/CI.

## Receipt pointers and scientific limits

- Cadence #508 merged `03fac532bad64bc7c647fabdb89244170fdaa0a7`: private
  `research/dsv4f-replay-cadence-20260908/COMBINED-RESULT.md`; exact c16385b binary
  ABBA +1.007107%, BAAB +1.074545%, all 20 rows/load and first captures retained.
- Dense #507: private `research/dsv4f-dense-exact-tail-20260908/MODEL-RESULT.md`;
  namespaces `dense-tail-model-5b66fe9-r2` and `dense-tail-model-5b66fe9-r3`,
  +0.459578% and +0.397451%, all identity/retained-graph/refusal gates pass.
- Composition #371, direct receipts in private Darklanes #509 (+1.87%/+2.11%):
  private composition receipt family
  `/root/dsv4-dev/receipts/compose-cadence-dense-*`; the exact reversed twin is
  `compose-cadence-dense-e591f45-r2`. Root owns the combined result write-up and
  final R2 decision. +1.87%/+2.11% are root-reported here, not new measurements.

Root authorizes default-policy qualification, with merge conditional on review
and the engagement gate. This draft references #366/#368/#371 for later
consolidation; it does not yet close or merge those PRs. This PR's current
file-by-file provenance is `PROVENANCE.json` in this directory and the revised
composition `COMPOSE-PIN.md`: ten inherited files remain byte-identical, six
inherited files changed for host policy, helper controls or documentation; the
new engagement helper and other additions are listed separately. Historical
composition hashes are explicitly scoped to their original source snapshot and
are not claimed to describe the changed files at this head.
