# Opened rank intake for self-trim

This slice follows reviewed `f8707ab2d89ed31dc28bb439106b3312c172e17f`. It supplies the
opened rank component needed before target/draft identity composition: the existing trim
loader parsed rank IDs and separately reopened the pathname to hash it, and another trim
arm could open it again. Those operations could describe different input bytes.

`RankArtifact` now owns immutable IDs and separate raw-source, normalized-order and combined
component digests. Text is parsed and hashed from one captured byte buffer. GGUF input uses
one capture pass over the already-opened shard handles: each buffer contributes to the full
source hash and to the retained header/d2t bytes. Only headers and the rank payload are retained,
not complete model-weight copies. The retained header is checked against the opened tensor
descriptors with the existing GGUF binary value decoder before interpreting the captured ranks.
There is no pathname reopen for stamping, or second payload read to derive the IDs.

The source digest includes unused tensor data, metadata and padding. Single-file stamps keep
the existing raw SHA256 definition. Split GGUF containers use framed hashes of all opened
shards in their validated order. The normalized order digest contains the ordered u32 IDs;
equivalent text/I32/I64 rank lists share that digest, while raw source and encoding remain
distinct. The combined component digest is not a rewrite-admission capability.

## Existing consumer adoption

One `RankInput` belongs to each model-load invocation. The normal MTP self-trim, early
MTP-skip preparation and GLM DFlash2 trim intake use that same object. They derive both IDs
and the existing diagnostic sha16 from its captured artifact; a different input specification
within the load refuses. A fresh model-load invocation captures the new file normally.
The component's full identity remains available during load for later coordinated composition.

The public text parser/rank validator and byte-gather interfaces remain compatible. The signed
I32/I64 decoder is shared with the reviewed standalone draft contract, so oversized integers
cannot truncate into valid token IDs. Empty, duplicate, malformed, missing/ambiguous, wrong-dtype
and wrong-rank inputs refuse. Actual head row bounds are still checked at each existing trim
consumer. Head selection, macro ownership, native gather/requantization and tensor allocation
are not redesigned in this slice; the existing gather body is unchanged apart from local names.

## Evidence

Eight real-file rank tests pass: text/I32/I64 encoding and order; raw-file hash equality;
path replacement; in-place payload edits after capture; tensor-header edits and truncation
before capture; full-source hashing across unused data; split-shard identity; and invalid IDs,
dtypes, shapes and head bounds. The GGUF fixtures include unused data spanning capture buffers.

Three CPU consumer tests compile the actual `trim_ranks.rs` module and exercise its cache,
identity stamp and byte gather across pathname replacement and in-place edits. Cached inputs
keep the captured IDs/stamp; fresh loads capture the changed file. Whitespace changes preserve
normalized rank order while changing raw identity. Different source specifications refuse.
The fourteen existing external-draft controls pass after decoder sharing. Total25 targeted
tests pass; no broad root suite or GPU run was performed.

Strict GGUF, consumer-harness and Linux DOCS_RS engine lib/test Clippy, formatting and whitespace
checks pass. Existing targets and production dependency versions are reused. Exact source/harness
hashes, raw logs, two preserved CPU executables and a preservation receipt are in evidence.json.
The receipt verifies922 other tracked crate files and the unchanged standalone draft loaders,
native trim materializer, existing hashing helpers and byte-gather program.

## Coordination and remaining acceptance

The rank proposal was relayed by the parent to existing542 owner
`01a0bb71-ea9b-7150-b433-856d5dd74a1d` through the working multi-agent channel. Shared
RewriteIdentity/admission, plan_backend/fallback, worker/LRU and support/default policy are
unchanged. Parent authorized this owned reader/consumer slice while the future identity seam
is coordinated. Independent review of this exact slice is pending.

Target-head ownership/materialization receipts, composite external drafts, paired target/draft
rewrite identity and native qualification remain separate acceptance work. Eager/DecodeGraph
proof does not authorize external Spec. No remote541 allocation, native/model/support promotion,
main integration/merge, rental/topup or owned cache cleanup occurred.
