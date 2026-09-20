# Bind rewrite admission to loaded bytes and the running program

Decision: 2026-09-20, issue #542.

Strict rewrite admission uses v2 receipts. The loader supplies the opened artifact digest,
the running executable digest, and the loaded numerical-program digest independently of the
bundle. The validator compares all three plus the serialized plan. Receipt and lock hashes
still protect bundle integrity, but cannot provide these trusted runtime identities.

The old policy accepted a same-plan bundle with an unrelated artifact label and arbitrary
implementation hash. The fail-before reproduction and CPU validation are recorded in
`research/modelplan-onboarding-rewrite-identity-20260920/`.

Rejected: treating a label, geometry, artifact.lock hash, or source revision as the running
implementation identity. Also rejected: promoting an unbound synthetic fixture receipt or
transferring a gate executable's receipt to a different serving executable. Unsupported
composite sources must refuse strict qualification until every contributing artifact is bound.

Unbundled execution remains explicitly legacy and unqualified. Configuring a bundle requests
strict admission, and an installation failure cannot retain legacy or previously granted
permissions. No hardware default or model support state is promoted by this change. Native
GPU qualification remains required before integration.

Native follow-up: the first Qwen3-0.6B probe compared fresh-F32-KV `forward_last` with
quantized-cache eager decode and correctly failed (max_abs3.1681318). The existing standing
`run-gen` quantized-cache verify-prefill control matched decode at1.907e-6. These are distinct
programs, so `forward-fresh-kv` now has a distinct manifest/receipt and cannot borrow eager
admission. The native gate uses independent verify-prefill and tokenwise executions of the
same cached-KV class; tolerances are unchanged. Fresh-KV outputs remain diagnostic evidence
and require their own parity receipt before strict admission. No shared attention math changed.

Phase-boundary native diagnostics then showed that cached verify/eager identity stayed
valid; the separate fresh-KV diagnostic alone loaded another compiler library at the long
prompt. That diagnostic now has its own process and output directory. Driver-link and
scratch warmup experiments were not necessary for the cached program and were removed
from production; their measured controls remain in research. No library drift check was
relaxed and no fresh-program dependency was added to cached-eager admission.

Review follow-up (SEC-557-1): native stacked and per-expert NVFP4 disk caches are
derived data, not trusted checkpoint bytes. With either identity flag configured,
the loader regenerates the canonical layout from the opened source, verifies and
repairs the named cache, and consumes a separate private, unlinked, read-only inode.
Both the mmap and positioned-read backing retain that inode. Changing or truncating
the named cache therefore cannot change live weights. This pays repacking I/O and
private disk capacity at load while keeping RAM bounded to one expert. Legacy cache
reuse is unchanged. Hashing only the named cache, or verifying it and then retaining
its writable inode, was rejected because neither protects the loaded byte lifetime.

Review follow-up (PERF-557-1): strict requests use a validated execution snapshot.
The model and plan digests are captured once. The model owns its program through
`TrackedProgram`; every mutable borrow revokes snapshots and invalidates the loaded
identity, even for a same-shape change. A reload is required before that model can
be qualified again. Both successful and failed bundle reinstalls revoke older
snapshots. Lazy embedding mirrors and mutable graph/workspace caches are private to the engine.

`rewrite_execution_snapshot` checks loaded-library and numerical-environment state
at each new or resumed request. `enter_rewrite_execution` borrows the model immutably
and activates that snapshot for a scheduler tick or synchronous operation. Nested
eager, speculative, graph and pipeline admissions check only the generation and a
fixed surface mask. The server preserves a snapshot per request, including restored
KV requests, and checks every participant in a batch. Retained decode/prime graphs, native MTP and GLM sessions, and suspended primes carry
their originating snapshot; supplying a different qualified model cannot authorize them. Standalone direct calls still validate a boundary;
callers doing a token loop can hold `protect_rewrite_execution` across the loop.

External process libraries and numerical environment must remain fixed while a
request executes. An explicit identity/boundary check that detects drift revokes
in-flight snapshots too; restoring the environment does not revive those snapshots.
This replaces per-token process-map scans with an explicit protected boundary,
rather than caching permissions over freely mutable model fields. CPU call-count,
mutation, reinstall, model-binding and scope-lifetime regressions cover the control
logic. The previous native receipts remain historical; the changed executable needs
fresh native admission, affected exactness and controlled performance validation.
