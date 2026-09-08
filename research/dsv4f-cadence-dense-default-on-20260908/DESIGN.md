# Default-ON cadence and dense exact-tree transport

Integration starts from composition PR #371 source
`e591f451c21c05f0a80445cdd2d7cd7a301a8f9c`, containing cadence #368 and dense #366,
rebased onto `001c09e5d451798ef8d570f6087dc11527cbcc19`. The kernel arithmetic,
cadence variant bodies, paired capture/drain/fail-stop, refusal-before-commit,
live control layout and supported shapes remain unchanged.

This draft prepares both defaults ON. It is not merge authorization: root gates
the default decision on the composition reverse-order receipt. Composition R1
reported +1.87% with identity; its reverse twin is a separate qualification cell.
No GPU execution or new numerical claim is attached to this default-only source.

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

## Receipt pointers and scientific limits

- Cadence #508 merged `03fac532bad64bc7c647fabdb89244170fdaa0a7`: private
  `research/dsv4f-replay-cadence-20260908/COMBINED-RESULT.md`; exact c16385b binary
  ABBA +1.007107%, BAAB +1.074545%, all 20 rows/load and first captures retained.
- Dense #507: private `research/dsv4f-dense-exact-tail-20260908/MODEL-RESULT.md`;
  namespaces `dense-tail-model-5b66fe9-r2` and `dense-tail-model-5b66fe9-r3`,
  +0.459578% and +0.397451%, all identity/retained-graph/refusal gates pass.
- Composition #371: private composition receipt family
  `/root/dsv4-dev/receipts/compose-cadence-dense-*`; the exact reversed twin is
  `compose-cadence-dense-e591f45-r2`. Root owns the combined result write-up and
  final R2 decision. R1 +1.87% is root-reported here, not a new measurement.

No root decision is inferred from an in-progress composition cell. This draft
references #366/#368/#371 for later consolidation; it does not close or merge
those PRs. Old provenance documents describe their measured source snapshots;
the new environment semantics are the narrow changes specified here and in FLAGS.
