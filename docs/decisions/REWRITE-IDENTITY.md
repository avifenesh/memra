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

`rewrite_execution_snapshot` retains the originating identity. Activating it through
`enter_rewrite_execution` always validates loaded libraries and numerical environment
at an outermost entry, before installing its permission mask. The validator is a
required argument to the private activation primitive. Only a current enclosing scope
for the same model/generation permits nested calls to reuse validation. The original
epoch is checked again after validation and is never refreshed to make a retained
object pass. This covers standalone graph step/profile methods, prime replay and
worker tick/resume. Cached-token emitters hold the same guard before sending output.
The server checks every batch participant's origin; a peer cannot lend its newer
qualification to a revoked request.

SEC-557-2 / PERF-557-2 exposed the previous ordering error: entering an old snapshot
activated its mask before checking changed external state. The exact reviewer
`MEMRA_FAST` reproduction fails at frozen `cbafa2eb`. The common environment and
library-inventory regressions also fail against that frozen activation body and pass
against the corrected primitive. The library CPU case injects real file metadata into
the production comparator; it is not a Linux executable-mapping or CUDA test. Logs,
source hashes and the signature-only adapter used for the frozen comparison are in
`research/modelplan-onboarding-rewrite-identity-20260920/later-drift-20260920/`.

Nested eager, speculative, graph and pipeline admissions check only the generation
and fixed surface mask, without rescanning libraries or serializing model/plan data.
The 10,000-call test holds one continuous outer scope and observes one inventory
validation; after that scope ends, re-entry requires another. Outermost standalone
calls pay external validation. The worker pays one external validation per model
per worker tick: at the start of the tick it holds a boundary for every model with
an active session, through the retire sweep. Per-session sample/emit, token emit,
batched, graph, speculative, prime and DFlash entries of that model then nest. The
boundary grants no surface, so each entry still checks its own snapshot generation
and a revoked request refuses without borrowing a peer's validation. A failed
boundary holds nothing and revokes the model, and its sessions refuse at their own
entries; the cause is logged once for the tick. External state must stay fixed
within a tick; a change is detected at the next tick. This is not a claim that a
whole serving token step has constant cost or measured throughput improvement. Drift
at a boundary revokes every older snapshot; restoring external state does not revive it.

The native runner includes a separate `library-drift` probe using a real read/execute
mapping (never executed) and a populated eager cache. It must refuse retained re-entry
before token work and preserve cache hashes. Qualified graph/prime/worker probes and real environment-drift control are now
implemented but unrun. The latter suspends the owned Linux process's complete thread
group before a verified one-byte debugger write, with explicit CUDA context drains;
it uses no concurrent in-process environment setter. `NATIVE-REFUSAL-PLAN.md` names
the conditions and mandatory CPU/Linux/native gates. Previous
native receipts remain historical; the changed executable needs fresh native admission,
affected exactness and controlled performance validation.


TC557-CALLERS-01: caller/battery result publication is a final commit. The runner
first records an incomplete result, then completes the cases, terminates telemetry
(with bounded kill/reap after failure), closes its log, writes the evidence manifest,
and revalidates source/binary/lease invariants. Only then does atomic result replacement
publish passed. Cleanup or finalization failures publish failed; a failed atomic
replacement leaves incomplete. The manifest excludes the mutable root result and the
result binds the manifest hash. Raw child/case evidence survives failed finalization.
Both phases have cleanup timeout/error, manifest, final-invariant and publication
fault controls. The original1aee runner fails them; the corrected runner passes.


The same TC557-CALLERS-01 finalization rule now covers `qualify-native.py` too.
Both runners use `native_finalization.py`; baseline variant/binary cleanup is part
of that shared failure accounting. No runner publishes passed before telemetry,
artifact cleanup, log close, evidence hashing and final identity checks complete.
Seven baseline CPU fault controls preserve the twelve-case schedule and expected
refusal semantics, including variant cleanup, and complement the twelve caller
controls. Existing source/build trees remain frozen; this is a gate-code change,
not permission to relabel their executables or qualification records.


Late cancellation is part of the same TC557-CALLERS-01 rule. Cancellation handling
now remains active through final verification and atomic publication in both runners.
Each final verifier checks cancellation after the identity work returns. Preparing or
replacing a passed result temporarily uses an immediate cancellation handler, preserving
the outer signal state rather than clearing it. A cancellation at publication leaves
failed/incomplete, including rollback of an interrupted replacement. The outer signal
scope is restored only after the final status decision and publication. Real SIGTERM
controls cover final verification, pending-result writing and the replacement boundary
for baseline, callers and battery in isolated CPU processes; all nine scenarios refuse.
