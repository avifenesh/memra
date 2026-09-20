# Retained native refusal plan: SEC-557-2 / PERF-557-2

Status: the `rewrite_identity_gate library-drift` case is implemented but has not
been executed natively. This plan records additional coverage, not a qualification
result. Historical source621 evidence and existing result records are unchanged. No historical receipt transfers to the changed executable.

## Implemented, awaiting a native run

Run the separate mode with a real single-device `HybridModel` and an eager-only
bundle freshly qualified for the exact rebuilt executable, artifact lock, plan and
numerical environment. The ordinary loader must install the bundle. This mode
neither creates a receipt nor reinstalls, repairs or broadens qualification.

The coordinator must use the ordinary owned native build/provenance and per-card
lease checks, including source/binary verification before and after the case.
Compile-only `DOCS_RS` artifacts are not usable native executables. Use a new
external evidence namespace; retain stdout/stderr, exit status, source/binary/build
record hashes and the prerequisite bundle. `qualify-native.py` schedules this separate-process case after the positive fresh-process
eager replay, with the same owned-build and lease validation as other native cases.

After fresh capture/check with that same executable, the additional invocation is:

```sh
MEMRA_ARTIFACT_LOCK="$FRESH_BUNDLE/artifact.lock" \
MEMRA_REWRITE_BUNDLE="$FRESH_BUNDLE" \
"$NATIVE_GATE" library-drift "$MODEL_SOURCE" "$FRESH_BUNDLE"
```

Preserve all other numerical environment values from the fresh capture. Linux
and an executable-mapping-capable bundle filesystem are required; inability to
establish the mapping is a failure, never a passed or skipped refusal case.

| Gate / output marker | Required observation |
| --- | --- |
| `RETAINED_NO_DRIFT_REENTRY_PASS` | Decode the fixed short prompt in a real eager scope, admit a nested scope, end both, then successfully re-enter the **same** saved origin with unchanged external state. |
| `RETAINED_LIBRARY_DRIFT_REFUSAL_PASS` | After the scopes end, map a new private named file with `PROT_READ \| PROT_EXEC`, never execute it, and confirm its read/execute mapping in `/proc/self/maps`. The first subsequent identity/admission operation is `model.enter_rewrite_execution(&saved)`. It must report changed executable mappings naming this file, with zero calls to the token continuation. No explicit validation, new snapshot, reinstall or model mutation may trigger refusal first. |
| `RETAINED_LIBRARY_DRIFT_CACHE_PASS` | Hash the populated real cache's supported KV, recurrent, latent, counter and last-logit state before and after refusal; require equal hashes. Unsupported opaque/distributed cache state fails closed through the existing cache probe. Require the original snapshot's generation to be revoked. |
| `RETAINED_LIBRARY_DRIFT_UNMAP_PASS` | Drop the mapping and confirm the file is absent from `/proc/self/maps` before unlinking it. The private file contains 4096 zero bytes; its name, length, SHA-256 and maps row are logged. Cleanup also runs on ordinary early-error returns. |
| `RETAINED_LIBRARY_DRIFT_REVOCATION_PASS` | With the mapping removed, direct re-entry of the old snapshot must still refuse as revoked. Verify the model's external identity again matches baseline without replacing the snapshot or reinstalling a bundle, and confirm both revocation and the unchanged cache hash. |

Only success through all these assertions emits `NATIVE_REFUSAL_GATE_PASS`, scoped
to `retained-eager-snapshot`, with `receipt_emitted=false` and no support promotion.
This exercises the common native retained-snapshot boundary. It does not execute a
qualified graph, prime graph or worker request and supplies no throughput claim or
native inventory call-count measurement.

## Still pending: qualified native methods

Each method needs its own authentic surface qualification on the rebuilt binary,
and a fresh case so revocation in an earlier method cannot make a later one pass.
An eager receipt must never be copied into a graph/prime receipt or used to build
synthetic qualification. Preserve the originating snapshot; never refresh it to
make the retained object pass.

| Native case | Required future gate |
| --- | --- |
| `GraphSession::step` | Create a genuinely `DecodeGraph`-qualified session, finish its original scope, add executable-mapping drift, then call `step` standalone. Refuse before graph update/replay or any cache, resident token or position change. |
| `GraphSession::prof_apply` | Independently create/qualify a retained session and finish its scope; inject drift, then call `prof_apply` first. Refuse before graph parameter updates. Cache hashes alone do not observe graph parameter mutation; add an actual update witness. |
| `GraphSession::prof_launch` | Independently retain a qualified session across ended scopes; inject drift, then call `prof_launch` first. Refuse before launch and host/device counter or cache changes. |
| `GraphSession::prof_read` | Independently retain a qualified session across ended scopes; inject drift, then call `prof_read` first. Refuse before reading or returning its resident token. |
| `HybridModel::prime_graph_run` | Create a genuinely `CarriedPrime`-qualified `PrimeGraph`, end its creation/use scope, retain it, and provide a fresh cache. Inject drift and call `prime_graph_run` first. Refuse before embedding, input/counter uploads, replay, scratch/output changes or copying into the fresh cache. |
| Native worker tick/resume | Coordinator-owned: retain admitted request state across completed tick scopes, inject drift and resume. Refuse before cache/token work or successful output emission. The main lane owns worker implementation and CPU emitter regressions; this native case is not implemented here. |

For each future native method, include an unchanged-state positive control and
post-unmap refusal of the original object. Observe method-specific mutable state
as well as cache bytes and output. Any necessary observation hooks are additional
work, not implied by the common-boundary probe above.

## Still pending: native later-environment drift

No sound controlled same-process environment mutation mechanism is available in
this harness after CUDA initialization. Late `std::env::set_var`/`remove_var` can
race with live driver or library threads, and a mutex around the test does not
protect those readers. This native case is deliberately unimplemented. Fresh
processes with different environment values test load-time receipt mismatch, not
re-entry of an already retained model/snapshot, and cannot close this gap.

Main owns the isolated CPU environment-drift and nested-boundary call-count
regressions. Native environment coverage requires a separately reviewed sound
control mechanism before adding cases for the retained boundary and each
qualified graph/prime/worker method. Nothing in this plan marks those cases passed.
