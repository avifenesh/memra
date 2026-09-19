# Generic spill shared interface v1

Date: 2026-09-19. Scope: native, exact-byte spill control-plane contracts; no runtime
default, format substitution, new model support or hardware qualification.

## Decision

Use `crates/memra-tier::contracts` as the only shared schema for ObjectStore,
TransferEngine, TierStore, BankedResidency, RowService, PeerBackend/PeerCapacity,
KvMaterializer, and the single BudgetGovernor.

Preserve full program identity and complete state; distinguish state epoch from both
allocation generations. Enumerate every submitted item and segment. Keep publication,
producer readiness, consumer waits and physical retirement distinct. Retain quota and
actual allocation ownership through unknown completion. Require explicit borrowed
release, immutable prefix sealing, original expert/row IDs, checked uniform proofs,
and contiguous-only peer transfers. Do not label host bounce P2P.

Persist explicitly versioned canonical metadata and domain-separated SHA-256 identities.
The lead freeze selects canonical JSON rather than the proposals' new binary metadata
serializer, and groups heterogeneous epoch triples into separate tickets. Both choices
are called out for lead integration review, not hidden compatibility assumptions.

## Rationale, alternatives and evidence

The complete A/B/C/D ambiguity ledger, rejected alternatives, per-WP migration table,
wire byte rules, independent fixture hashes and remaining native integration gates live
in [the interface freeze](../../research/spill-lead-20260919/FREEZE.md).

CPU fake-backend fixtures test shared state-machine invariants. They do not prove OS
persistence, CUDA transfer/graph lifetime, native attention operands, measured fairness,
model correctness or four-card capacity. Those gates remain with the respective WPs.
No implementation or execution of paused/new model work is included.

## Freeze revision v1.1 — 2026-09-19

This is an **additive CPU conformance revision**, not a new runtime or wire contract.
`tests/contracts/conformance.rs` re-exports the source-shared schedules in
`tests/contracts/revision_v11.rs`; lane tests path-import that same source.

Added generic pinned-pool lifetime/quarantine, common governor accounting and actual
queue-dispatch priority, peer capacity/directed-route, explicit producer-complete then
cancel-before-publication, namespace/epoch, indexed acceptance, byte-row/scale-sibling,
and retirement schedules. Test-only factory/observation hooks drive backend state;
assertions are shared rather than duplicated into each lane. Native event truth is
not inferred from these hooks. B's dispatcher binding actually calls its enqueue and
dispatch methods; sorting the Priority enum is not called a fairness proof.

**Unchanged:** every runtime trait and persisted struct, canonical JSON byte encoding,
SHA-256 domains/fixtures, wire version **1**, homogeneous epoch triples, the frozen
priority order, mandatory headroom, explicit borrowed release, and cancellation's
publication boundary. No implementation fix, dependency, flag, GPU path, or new-model
code is included. The reference governor fixture now gives its loader dimension the
same configured test capacity as its other dimensions so it can exercise that ceiling.

The stricter schedules expose **two unresolved lane defects**: A releases quarantined
pool backing on last-pool drop; D's capacity fake cannot distinguish directed grants.
They are kept as failing tests, not weakened, ignored, or converted to expected panics.
Consequently this revision is **not a green integration/release gate**. See
[FREEZE-V1.1.md](../../research/spill-lead-20260919/FREEZE-V1.1.md) for exact failures,
D's nine-item disposition, executed commands, before/after counts, and unrun native
surfaces. `StateBundleAdapter` has no cancellation method; adding one is deliberately
not smuggled into a schedule-only revision.
