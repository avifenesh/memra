# Physical headers and index entries are distinct

Independent text-schema review of `ef131c82fd53d178c5a2859b5da95e6aae9e25da` fetched
metadata for the exact locked Step FP8 revision and checked all 24 text-shard headers. The
reviewer's files copied here were verified against its original SHA-256 manifest. No weight
payloads are included.

| Surface | Evidence |
|---|---|
| Pinned index | 1,471 entries: 804 text primary names and 667 vision/projector names |
| Actual text headers | 930 physical tensors, including 126 `weight_scale_inv` tensors absent from the index |
| Folded text census | 804 rows/roles, all shapes/storage checked |
| Scale/transform controls | 126 FP8 grids; 202 AddOne and 602 Identity transforms; private MTP owners and 64/96 gate geometry checked |
| Vision/projector | 667 indexed names remain unrepresented; this record is not a physical vision census or execution qualification |

The safetensors reader already merges authoritative shard-header names and checks index ownership.
Binding must retain this behavior. Index-to-header validation is not a one-to-one equality test:
valid unindexed auxiliaries must be assigned to their owning weight with shape, dtype, byte range
and identity preserved. Duplicates and undeclared leftovers still refuse. A focused bound-source
regression checks that an omitted FP8 scale survives index opening, appears in the bound auxiliary
metadata/physical byte total, and reaches the native view.

This corrects any reading of the original 1,471 index-entry count as a complete physical count.
It neither changes the artifact nor grants full-artifact, vision, runtime-activation or native
approval. The supported/unused-surface architecture contract remains under review; see the
[scope proposal](../SCOPED-CENSUS-PROPOSAL.md).

Independent raw files are gzip-compressed byte-for-byte, preserving safetensors header padding.
`independent-review/uncompressed-sha256.json` retains the reviewer's original hashes, while
`independent-review/sha256.json` hashes the stored compressed files.
