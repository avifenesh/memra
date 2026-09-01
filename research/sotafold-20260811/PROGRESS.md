# sotafold — progress log

Lane: `lane/cx-sotafold` in `/home/avifenesh/projects/wt-cx-sotafold`.
Deliverable: fold four fingerprinted Hermes web-research findings into
`research/spec-landscape-20260810/SURVEY.md`, plus the replay-semantics cross-reference.
Scope: documentation/reference prose only; no code, GPU, generated perf surfaces, or push.

## Fingerprint → survey entry mapping

| Fingerprint | Finding | Survey entry | Memra surface | Exactness/posture |
| --- | --- | --- | --- | --- |
| `9de228dce2ae2faf` | ASD | ASD — regret-budgeted approximate verify | `crates/memra-engine/src/spec.rs` verify; `MEMRA_*` regret-budget family; after `dspark2` | Approximate for nonzero budget; default-OFF blocked serving door; strict `run-spec` remains default and gate |
| `1ecd30b9cb632563` | DraftExpert | DraftExpert — fixed-footprint resident draft expert | `crates/memra-engine/src/moe_cache.rs` residency metadata; `MEMRA_SPILL_IO=worker`; after `dspark2` | Exact target verification; default-OFF spill+spec research arm |
| `5401fbbbeb696bce` | OasisKV | OasisKV — spec-draft lookahead as a KV-prefetch oracle | `crates/memra-engine/src/spec.rs` draft-token keying; next-step KV prefetch/stage queue over spill plumbing | Lossless only for prefetch-only adaptation; no distribution change |
| `45248f7c4c694a98` | WiSP / MV-WSA | WiSP / MV-WSA — PCIe ceiling and marginal-value residency | `crates/memra-engine/src/moe_cache.rs` residency plus KV budget | Byte-identical/lossless; CAUTION for Hy3 spill lane |

Cross-reference added: replay semantics from MoE-cache eval (`9280ff62`) requires
pinning fused-event replay and matched-pair probe diversity before ranking spill policies.

## Work log

- [x] Read `~/.lanectl/inbox/cx-sotafold.md` first and read `/home/avifenesh/projects/bw24/CLAUDE.md`.
- [x] Confirmed clean `lane/cx-sotafold` worktree and docs-only scope.
- [x] Read the full report bodies from `~/.hermes/repo-review/memra-shared-report.md` by fingerprint.
- [x] Validated the four cited arXiv abstracts and kept OasisKV's sparse-attention approximation
  distinct from memra's proposed lossless prefetch-only adaptation.
- [x] Added the four survey entries and replay-semantics cross-reference to `SURVEY.md`.
- [x] Re-read the edited survey and verify only intended docs changed.
