# Verified scoped mirror readers — execution pending

The candidate retains the existing native mirror-map parser and generation policy. A metadata-only
query returns the declared alternate path/generation without exporting a descriptor. Bound access
requires source/alternate generations, equal file size, aligned window and different filesystems,
then compares every selected byte using bounded buffered reads before publishing the direct reader.
Source and alternate generations are checked again after verification. Cached windows retain those
proof inputs; a shared direct-file cache avoids one persistent descriptor per expert. Files must
remain immutable while loaded, as with the original source and mirror contract.

New optional mirrored token/rows/prefetch entrypoints retain both reader handles. Existing ABI2 and
single-reader entrypoint signatures remain intact. The original numerical routines, cache key and
aligned half-split remain shared. An explicit alternate selector is carried by I/O jobs rather than
using a fake descriptor. Both readers drain before release; detached prefetch owns both contexts.

The planned tiny CPU controls compare scoped mirrored output bits to the actual legacy mirrored
path. Observer counters must show six reads from each half (196,608 primary / 221,184 alternate
bytes), then twelve retained contexts during blocked prefetch and no live contexts after drain,
with original annex bytes verified. Additional controls reject different bytes despite matching
metadata, wrong source/target generations, same-filesystem maps and cached generation changes.

The existing authorized host has one filesystem for owner/receipt directories and a separate
`/dev/shm` filesystem. The positive fixture will use a unique, owned subdirectory there for a small
byte-equivalent copy and remove it on completion. This tests the distinct-filesystem requirement,
not NVMe topology or throughput. Source/targets/receipts remain isolated; CPU threads/jobs stay
bounded at four, Cargo is offline, and CUDA visibility is empty. No mounting or shared configuration
change is required.

Local checks pass: GGUF/CLI tests and Clippy, actual host harness Linux cross-check, DOCS_RS
engine/server lib/bin/test Clippy, formatting and flag census. The mirrored execution cases have
not yet run at this freeze. Earlier single-reader receipts and explicit-refusal results remain
unchanged. This is not a root/model/serving/GPU/performance or support-state promotion.
