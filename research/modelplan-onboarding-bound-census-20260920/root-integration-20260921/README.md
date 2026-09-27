# Engine root preparation integration

Dense `Model::load_dense_from_source` and hybrid `load_from_source_impl` now prepare one
compiler-owned text source before invoking their existing load bodies. `PreparedModelSource`
selects an ordinary or composite bound bundle; it never retries a rejected artifact through
legacy name lookup. A sealed, text-compatible runtime is reused without double binding.
Existing full multimodal bound runtimes must be prepared with Text scope for these text roots.
The source, config, plan, ownership, transforms and identity stay in one validated bundle.

A shared semantic ABI resolver serves ordinary and composite adapters. The composite runtime
retains original expert IDs, separate scale ownership, native planes, private canonical outputs
and opaque disk views. Private composite-input and preflight types prevent external adapters
from substituting another source or caller-assembled program. Loading callbacks cannot leak
borrowed tensor views after the runtime adapter is dropped.

The text scope validates the complete artifact inventory while withholding vision execution.
The original source config remains available to existing native vision loading. Sparse GGUF
text overlays use a text storage schema; the underlying HF component still validates all vision
rows. This does not authorize unknown vision tensors in the overlay. The Gemma loader avoids
probing a per-layer embedding when the config declares zero width; unsupported PLE programs
remain unsupported by their model packs. Known optional router/shared-expert/frequency probes
resolve semantic absence; misspelled or out-of-plan requests still return errors.

## Placement accounting

Prepared sources expose compiler-bound charges containing semantic ID, owner, physical bytes
and an execution-selected marker. Attached scale/auxiliary planes remain included. Composite
charges follow selected original expert members; lower shadowed/pruned banks remain in physical
inventory and opened identity but are not selected charges. No global physical-alias deduplication
was added. Existing tied-head, first/last-stage, non-distributed and peer/device-selection formulas
are preserved. The device-selection function is byte-identical to the reviewed base.

The pure production artifact-cost function is extracted to `parallel/costs.rs` so actual
placement accounting runs in a CPU harness without CUDA. A real composite Gemma text case
records 5,676 selected execution bytes plus 6,656 inventory-only vision reservation bytes,
with exactly the same total/first/last costs as legacy full-artifact accounting. Ordinary
safetensors (tied and separate heads), attached FP8 scales, and GGUF controls match legacy costs.
A retained composite keeps original IDs and charges half of the four-expert bank while retaining
the underlying physical inventory.

## CPU evidence

Five root tests cover metadata refusal before the loading callback, prohibition of legacy name
reads, GGUF byte/identity preservation, existing sealed-source reuse, composite original IDs,
native/canonical scales, detached disk ownership and text-scope vision preservation. Four
production placement controls pass. The real-file composite reference test now loads every
weight through PreparedModelSource and the engine ABI, then matches all weights and logit bits
against the deterministic Memra fixture. This test is also wired into the CPU CI step.

All eight final commands pass: GGUF/CLI (425 GGUF reported passes, two declared ignores),
reference-root, production placement, production repack consumer (10), source Clippy, Linux
DOCS_RS engine lib/test Clippy, formatting and whitespace. Inherited artifact-dependent early
returns retain their limits. Exact source hashes, raw logs, initial compiler/lint fixes and
preserved macOS CPU executables are recorded in evidence.json. Existing targets were reused
with incremental caching disabled; no fresh broad target copy was created.

## Scope and remaining acceptance

This implements root adapter adoption for source loading; it is not native/model/support or
merge promotion. #542 plan_backend/fallback/snapshot code, worker policy, config decoder,
ModelPlan operations and rewrite manifest code remain byte-identical. Existing standalone
artifact-identity implementations are unchanged. Root-bound identities include their semantic
text scope and require fresh receipts for the resulting runtime binary.

External/student/trimmed draft contracts and composite draft identity remain separate unfinished
work; their existing loading/admission policy was not relaxed. Actual GPU numeric, serving,
pinned-copy/worker/H2D and I/O qualification remain pending under the parent's reviewed host
protocol and per-card locks. No rental, topup, remote launch or main integration occurred.
