# Source-owned canonical output

The reviewed source bundle at `5a805c39dd10d6c8858658ba27f6a0bdc8e40774` still needed a
private output boundary before root-loader adoption: the engine's older repack path opens cache
files through a source pathname. This prerequisite adds source-owned canonical output APIs and
leaves engine consumers and raw repack behavior unchanged.

Safetensors source opening retains the source directory and an already-existing cache directory
as capabilities. Capture errors remain latent for read-only loads and become explicit errors if
canonical output is requested; no later pathname reopen can redirect that request. A missing
cache directory is created relative to the retained source directory. Symlinks/FIFOs refuse.
The output-root hook is compiler-private; an external adapter cannot forge or forward it.

`BoundTensorSource` and `BoundCompositeSource` provide `canonical_nvfp4(id, member)` and
`canonical_nvfp4_bank(id)`. They consume only bound native operands and the existing
`repack_modelopt_to_gguf` codec, streaming one expert at a time in original-ID order. The result
is an opaque BoundDiskView. Macro/input scales stay separate under the existing auxiliary API.
Unknown, absent, pruned, mixed-native or shape-incompatible requests refuse.

Every output is produced in a new loader-owned 0700 subdirectory. A separate read-only descriptor
is checked against the writer inode and unlinked before production. Output length is bounded;
short/long/failed producers cannot publish a prefix. The writer closes before the readonly
mapping is returned. No named cache is read or published by this path, and concurrent output
views own independent backing. Engine cache-policy and adapter adoption are later integration
work; no default or native promotion is claimed here.

Nine unique CPU controls cover exact native/canonical codec bytes, stacked banks, retained
composite members, separate macro scales, poisoned cache contents, source/cache directory
replacement, read-only/unlinked file state, bounds, failure cleanup and concurrency. A new
compile-fail doctest protects the output capability. Full GGUF/CLI tests, the loaded composite
reference comparison, identity-host controls, strict Clippy and Linux DOCS_RS engine/server
type checks pass. Existing standalone artifact-identity method bodies are byte-identical; a
source identity also remains unchanged after output production and pathname replacement.

The first full CLI run was blocked by sandbox denial of its loopback HTTP listener. That attempt
is preserved separately; the unchanged suite passed after loopback permission was granted.
After every test completed, evidence banking hit local ENOSPC. Only this checkout's disposable
CPU incremental caches were inventoried and removed; source, linked binaries and all receipts
were retained. That banking failure is not a failed qualification run.
Linux DOCS_RS is type/lint evidence, not native execution. Existing artifact early returns and
#548 reference limits remain; no full-reference or checkpoint qualification is asserted.

This slice has no engine/server/worker changes and does not intake unfrozen #542 or #537 work.
The accepted main LRU behavior and removed segmented flags remain for parent-owned integration.
Next: independent review, engine consumption of canonical views before path-based caches, root
adapter adoption, draft contracts, then coordinated numeric/serving/worker/H2D/I/O gates.
