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
