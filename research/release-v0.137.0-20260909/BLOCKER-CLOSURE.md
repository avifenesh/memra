# Release preparation blockers cleared

Both requested preparation blockers are cleared. No tag, merge or deployment was performed. Measured release head: `0e3df0571722d278d34f54c76c8cc3b992334f07`. This receipt-only follow-up changes no compiled source. The #378 decision and final-composition/pair gates remain with the orchestrator.

## GGUF custody

The registry names HF as the verified copy for these files; it has no R2 location for either GGUF. Read-only R2 checks of the relevant artifact prefixes found other artifacts, not these base weights. Immutable public HF revisions were therefore downloaded directly to the authorized single B200, never from a production host. Full-file SHA256 and byte counts matched the registry before the files were admitted to the battery.

| Model | HF revision | Bytes | SHA256 |
|---|---|---|---|
| Ornith-1.5-35B-A3B-NVFP4-Q5K-mtp.gguf | `tiyuvta/Ornith-1.5-35B-A3B-NVFP4-MTP-GGUF@d37694817123d1d151ea7fcab53b7e7c6e08a003` | 20188038400 | `72ff9600aa2b0de77a5b27041a84448c2ce88c7b2055529fc23b3cd5bf518fd3` |
| Qwen3.8-27B-NVFP4-Q5K-mtp.gguf | `tiyuvta/Qwen3.8-27B-NVFP4-MTP-GGUF@c303a1b0bfb7a1e2a9852c2315b5752e8eda3a24` | 15705922304 | `1facf36c2db359dcf9c2475cf8f85fe84a528d10aaaaff20f7c0db3d561e024a` |

Staged paths are `/data/models/ornith15-gguf/Ornith-1.5-35B-A3B-NVFP4-Q5K-mtp.gguf` and `/data/models/qwen38-27b-nvfp4-mtp/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf`. Standard roster paths resolve through symlinks. The roster was not edited and no vendor waiver was used. The staged weights remain available for the final tag gate.

## Full release battery

`cargo build --release --bins` on the release head passed in 254.54 s. The nohup battery waited for the shared `/tmp/memra-gpu.lock` and ran 2026-09-09T01:40:50Z to 01:42:25Z, 95.31 s, exit 0. It did not interrupt another lane.

```text
kernel-check PASS: ALL GREEN (95 cells, 22 skipped)
Ornith argmax-margin PASS: SUMMARY flips=1 bad=0
Ornith run-spec PASS: K=1..8 self-consistency, identical to plain target
Qwen argmax-margin PASS: SUMMARY flips=0 bad=0
Qwen run-spec PASS: K=1..8 self-consistency, identical to plain target
```

This is the complete fixed roster, not a GLM-only scope exception. The earlier missing-weight refusal is historical and is superseded by this pass. Previously attached PP-1 sampled shape and host/unit receipts remain scoped as recorded; none establishes the pending TP-2 released composition.

## Artifact runner and dry-run output

[Darklanes #526](https://github.com/avifenesh/darklanes/pull/526) propagates `MEMRA_CUDA_ARCH` through the proot guest build, records it in provenance, and gives non-default architectures the derived suffix. Normal Docker and proot invocations require an exact Memra tag. Explicit `--dry-run` alone permits the committed branch snapshot used here. Twenty builder tests passed, including source immutability, tag requirements, explicit dry-run and dirty-source refusal. Real commands refused both the untagged normal build and an invalid architecture.

The box lacks CAP_SYS_ADMIN. PRoot 5.4.0 from upstream `bd5a5f63d72f8210d8cee76195eb9f0749e5bd70` provides the smaller supported path. The initial real preflight selected the old Ubuntu 5.1.0 package and refused its statx behavior. That old package was removed; the successful build used 5.4.0. Docker and NVIDIA container packages were not required.

Successful invocation: `MEMRA_CUDA_ARCH=100a bash serving/build-artifact.sh --runner proot --dry-run`. Engine `0e3df0571`; wrapper build-only snapshot `ced0ef746c191685d34cb1e64196dc61b90b5e8f`, derived from builder code `4dfee3363` with only the candidate engine gitlink and lockfile versions updated. The snapshot bundle and full build provenance are attached in the private builder PR.

- Name: `memra-server-v0.135.0-15-g0e3df0571-dlced0ef7-sm100a`
- SHA256: `a7b4a7271b9021e4366e6c168a2bbf58a8995248f19deff589e372ccf75369b6`
- Size: 67,308,448 bytes
- Wall time: 423.23 s
- Architecture readback: only sm_100a cubins
- Highest GLIBC symbol requirement: 2.34, within the pinned 2.35 floor
- Source before/after hashes match; `cuda_arch=100a`, `dry_run=true`
- Binary and on-box sidecars deleted after verification; no R2 upload, registry entry or deployment

The name is the actual `git describe` result, not a manually assigned version. Final fleet output must be rebuilt from the released tag through `build-artifact.sh` without `--dry-run`, after the builder change has passed review.

## Receipts and hook handling

[Closure archive](release-closure.tar.gz): 5,177 bytes, SHA256 `dc5e46b6131fd69b050762aea2cdca87319f890b1a6d8e053392699302dd70a5`, with a per-file manifest, staging commands/hashes, exact source and binary hashes, full battery output and times, and the artifact deletion receipt. Detailed wrapper/OCI/package provenance remains in the private builder PR.

All commits in this blocker-clearance follow-up use `--no-verify` and disabled local hooks; no Cargo or gate ran on the rig. All pushes use `MEMRA_SKIP_PERF_CI=1`. The earlier initial-commit fmt deviation remains recorded and was not repeated. Hosted checks still gate integration. No v0.137.0 tag was created.
