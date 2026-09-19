# Typed bank source installation — day 4 CPU contract

`crates/memra-tier/src/bank/source.rs`: `BankSource<S: ObjectStore>::install`
consumes the compiled Catalog, authoritative `BankSourceSpec` list and supplied
ObjectStore bindings together, returning the catalog plus its validated byte
reader via `into_parts`. This is a load-time CPU construction seam, **not an
installed native HostExps loader**. Frozen ObjectStore API used: A's day-4
per-extent catalog was not available in the merged source when implemented.
No speculative API duplication, source amendment, dependency or shared-file edit.

## Source authority and refusal

Each expectation comes from the locked artifact byte manifest/compiler, not from
whatever tensor happened to load. It binds full TensorId to an exact ObjectKey
(version/artifact/semantic identity/**source encoding layout**/generation) and
valid byte length. The object's layout digest need not equal every contained
row's RecordLayout digest: source objects can contain multiple exact records.
Catalog independently binds each record layout, every payload/scale tensor,
original router ID, and expected per-segment checksum.

Install requires precisely the retained catalog's source-tensor set:

- Missing expected, supplied or store source → NotFound; no fallback tensor.
- Duplicate expected or supplied TensorId → Conflict, even identical duplicates.
- Unexpected sources → refusal; masked originals have no backing assignment.
- Wrong artifact/semantic identity/layout/generation or source length → refusal.
- Every segment offset + storage length must fit its expected source; overflow
  fails before I/O. Required scale planes cannot be omitted to make a source fit.
- ObjectManifest wire invariants and returned key must match the requested key;
  changed lengths on the reader's second advisory lookup also refuse.

ObjectReader retains store ownership and revalidates each lease/read using A's
existing transfer machinery; source CRC/checksum corruption still fails the
whole logical batch. The bank's own checksum is expected from the source manifest,
not echoed from a read. CPU fake tests cover ten construction outcomes, and the
existing **real ExtentStore + CpuTransfers** expert/PLE tests now construct through
BankSource: forced misses, ordered duplicates, scale bytes, corrupt siblings,
mandatory-vs-optional governor pressure and retirement remain exercised.

## Conservative patch v2 shared pure check

`validate_bank_source_extent(offset, len, available, split)` validates a positive
exact extent without reading it. Split storage must have exactly len bytes and
uses source offset zero regardless of logical layout offset. Unsplit source uses
the checked layout offset. Tests include short source, oversized split, zero,
and overflow. The unapplied native guard now calls this same CPU-tested function,
after checking layout/tiers/macros vector cardinality and uniform multiplication.
It cannot install a BankSource, authorize a uniform lease, or produce GPU readiness.

## Native installation contract still to implement on rig

At `model.rs:1900–1925,3158–3223`, the loader must retain source files/mmaps and
original mask; install every projection and scale plane before exposing MoeWeights.
Split expert sources keep offset zero, while unsplit sources keep each authoritative
layout offset. Native loader needs a **charged** immutable source/extent catalog
(including maximum manifest/index memory and validation I/O). Today the caller
owns catalog/spec/manifest metadata; installation does not claim bounded model-scale
loader memory. Objects are framed copies in A's current store, not an in-place
view of arbitrary original safetensors offsets. Byte-identical import and bounded
per-extent indexing need the A handoff; this CPU API must not silently re-encode
model weights or pretend a whole-object lookup is one 264-byte physical read.

This closes CPU contract/testing for prerequisite (a); native source installation,
large-source index bounds and actual allocation accounting remain blocking.
