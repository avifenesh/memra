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
not native binary or GPU evidence. Full exact-tip checks and native cells follow.

The inherited day-5 receipts/verifier/legacy lock proposal were recovered and
pushed first at 7707b91d. The earlier capture refusal remains immutable.
