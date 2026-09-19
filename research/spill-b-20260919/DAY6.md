# WP-B day 6 — gate-only capture contract

Repository: avifenesh/memra; lane/spill-b-20260919. No runtime patch is applied
in this lane; HOSTPREFIX-PATCH.diff remains scratch-only. No serving, active-tier,
performance, deployment or support qualification is claimed.

## Capture fix

The native baseline now binds every allocated cache slot to the ModelPlan. Trunk
KV must be full-history native q8_0 K/q5_1 V and length == committed position.
Plan-declared MTP slots, deliberately unexecuted by load_without_mtp, may have no
KV or zero-length KV; those slots are recorded as absent-unexecuted-mtp. Nonempty
MTP, missing/wrong trunk state, unknown slots, ring/base, bad format geometry,
overflow and out-of-allocation extents refuse with layer, position, geometry and
exact failed predicate. Topology is checked before KV allocation and at capture.

CPU test module: crates/memra-engine/src/bin/kv_tier_gate/capture_contract.rs.
Built with rustc --test, linked to the locally built memra-gguf library. **2 tests
passed**, synthetic four-layer hybrid trunk + one NextN and fail-closed mutations.
Raw logs: day6-capture-checks/{build,test}.log.gz. These are CPU contract evidence,
not native binary or GPU evidence. Native execution and exact-tip checks follow below.

The inherited day-5 receipts/verifier/legacy lock proposal were recovered and
pushed first at 7707b91d. The earlier capture refusal remains immutable.

## Native result — baseline captured, active refused

Native source: **ac67071e4aca198ac3e0e46cb835749e92edbaba**, plus the unchanged
HOSTPREFIX-PATCH.diff applied only to a remote scratch branch. Exact binary/patch
hashes are in `rented-5090-20260919/day6-build/`. No lane runtime was patched.

| Cell | Actual result | Receipt directory under rented-5090-20260919/ | Published commit |
|---|---|---|---|
| Release gate build, `cargo build --release -p memra-engine --bin kv_tier_gate -j 16` | exit 0, 1.57 s incremental build | day6-build/ | fcb10942 |
| Baseline, `--context 8192 --tiers host --same-program` | exit 0, 8064 prompt + 128 generated, committed=8192 | day6-baseline-8192/ | ce447c98 |
| Baseline, `--context 32768 --tiers host --same-program` | exit 0, 32640 prompt + 128 generated, committed=32768 | day6-baseline-32768/ | f45b822f |
| Active, `--context 8192 --tiers host --same-program` | exit 2, explicit missing-binding refusal before CUDA/model initialization | day6-active-8192/ | 0f719b16 |

Both successful cells have **BASELINE.txt**, artifact/prompt/plan/binary identities,
per-layer prefix/final state manifests, all 129 decision/final logit-row hashes,
128 generated token ids and full final logits. The planned inactive NextN slot 64
is explicitly absent-unexecuted-mtp. Both say **active_engaged=false** and
**prefix_engaged=false**. This is a native trunk-only tokenwise baseline, not a
prefill fast path, MTP, tier round-trip, serving or support qualification.

Every cell ran through `tools/tier-battery.py`, canonical `/tmp/memra-5090.lock`,
with idle compute-app checks and 250 ms telemetry. Power cap was **400 W**, maximum
**600 W**, recorded before each cell. The first process listing contains its own
parent shell; it is a self-match, not a competing CUDA process. Later listings use
process names to avoid that ambiguity. The collector's qualification=false remains
unchanged. Single correctness runs (N=1), no thermal balancing or performance claim.
Filesystem remains **overlay, development, not spill speed**.

32k admission used measured idle free memory **32110 MiB**, measured 8k peak
**15008 MiB**, estimated additional KV **739.5 MiB**, and a **2048 MiB** workspace
margin. The actual sampled 32k peak was **18327 MiB**, exceeding that estimate
(**17795.5 MiB**) by **531.5 MiB**, but below measured available memory. The margin
was an admission estimate, not a proven capacity bound; do not reuse it as one.
The original admission JSON and telemetry are retained unchanged.

### Active verdict / exact missing seam

```text
REFUSED: active requires a native CUDA materializer + scheduler binding with nonzero demote/reload engagement; CPU fixtures and HostPrefix patch do not provide that binding
```

No prefix state was demoted/restored by this active invocation and no continuation
identity was demonstrated. The gate refuses before reading the artifact or creating
CUDA state. The collector's generic `died, cause unknown — repro needed` fallback is
not the cause; the raw refusal above is authoritative. Native materialization,
allocator/owner accounting, publication/retirement fences and scheduler restoration
of this request's exact state remain the next implementation boundary. No simulated
engagement or alternate numerical program was introduced to pass it.

## Verification and hygiene

- `day6-checks/` recovered at **5fabde1e**: 14 checks at source **7cbd246b**, all exit 0.
- `day6-relaunch-checks/` published at **ac67071e**: fresh 14-check rerun at
  **5fabde1e**, all exit 0: fmt; Mac/Linux-target all-target checks; memra-kv and
  memra-tier offline tests; scoped clippy `-D warnings`; diff/flags; patch applicability;
  runner tests; runtime-unapplied/shared-unchanged; three standalone CLI tests.
  Existing memra-gguf Darwin unused-import dependency warning remains out of scope.
- `python3 research/spill-b-20260919/verify-day6-receipts.py` checks 69 original/archive
  file digests, three collector descriptor trees, power/idle/source/binary identity,
  both complete baseline bundles and the active refusal. This is offline integrity
  verification, not an additional GPU run.
- `verify-rented.py` also passes: historical raw hashes and original failed baseline
  remain intact; descriptor verification now includes the three new collector cells.
- Every completed native cell was synced, committed and pushed before the next cell.
  No hooks skipped. One GitHub push timeout succeeded on a bounded retry.
- Initial new SSH connections intermittently timed out. The lead-provided persistent
  ControlMaster restored access; connection details stay private. Initial explicit
  lane fetch was needed because the remote's default fetch refspec omitted this lane.
- After execution, reversed only the known patch, detached the pre-existing remote
  B worktree at ac67071e, deleted this task's scratch branch and checked clean status
  plus zero compute apps. Existing remote worktree and prior evidence are preserved;
  the local B lane remains open for lead integration, not merged or deployed.

Stop-at-verdict: baseline milestone delivered; active slice **blocked by the named
native seam**. No NVMe, peer transport, full-context or PRO-pair qualification ran.
This relaunch used approximately **0.8 agent-hours**, including connectivity/build/
verification work; WP-B budget remains **10 agent-days**. Prior sessions' total time
is not reconstructed. No runtime default or published performance number changed.
