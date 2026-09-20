# Native device ownership and memory qualification — 2026-09-20

3/3 synthetic native ownership and memory seam stages passed; full-checkpoint and performance qualification remain pending.

All three ran at source `d413747d3b8a72c5af472300d1c65307133c1a37` on RTX PRO 6000
Blackwell Server Edition hardware (600 W, sm_120, 97,887 MiB reported per card), with
CUDA 13.1.115, driver 595.58.03 and Rust 1.97.1. The paired cards share a PIX path.
The build used a fresh Cargo target and empty `CUDA_VISIBLE_DEVICES`; GPU phases used
one physical card for the same-device stage and exactly two for each paired stage.
Every stage ran in a fresh foreground process inside the coordinator's per-card wrapper,
supervised by tmux, with the whole required set acquired atomically in stable UUID order.

This is correctness/ownership evidence. The backing filesystem is XFS on Ceph RBD;
there is no physical-NVMe, inference-throughput, latency, or full-power comparison claim.

## Results

| Stage | Native evidence | Result |
|---|---|---|
| Same-device ownership | Two independent Engine owners on one physical ordinal remain in the model-owned inventory and both are fenced. | One exact test passed; no skips. |
| GLM peer state and reclaim | The old Step-only accounting returns zero entries; the GLM peer requires 6,152 bytes of cold latent state. Charges disappear after allocation. Primary-only trim cannot reclaim the peer pool; peer trim releases 67,108,864 bytes while live bytes stay 33,653,144; refill succeeds. | One exact test passed; no skips. |
| Worker admission and pinned source | A 32 MiB peer allocation makes the per-device admission check reject while primary-only admission accepts. Three reclaim/refill cycles each release 33,554,432 peer bytes, preserving a real 1,048,576-byte pinned source and primary witness. Unpin restores eviction, and cleanup restores baseline live allocation counts. | One exact test passed; no skips. |

The worker uses the production headroom, all-owner fence, trim and reporting helpers with
bounded physical allocations and synthetic future requirements. It does not stand in for
a complete HTTP serving run or a model-scale admission-capacity measurement.

## Exact identities and raw records

- Engine test binary SHA-256: `fd252922b71faf61768aa488846d93bda0a6f7364df720ce10c34a010b3b0987`.
- Server test binary SHA-256: `6e4c91d1256ae5ed081f0c789c067b414b096aa0bd3361c79510593cc765213f`.
- Wrapper SHA-256: `073518dc842cfa0ec5d1cfe5f3439f378e3e0b0cec29835fae21b28dd0764e55`.
- Generated GLM-DSA F32 micro GGUF: seed `544`, **627,136 bytes**, SHA-256
  `06fcb3c32d503a0d22125606bb4cf91871d53301a22437ec20e8ccb556f75c53`.
  F32 loader warnings are retained in the raw logs; no replacement quantization was used.

The [raw bank](native-pro6000-20260920/summary.json) contains the complete build receipt
and compiler logs, exact per-test commands, selected GPU UUID mappings, per-card leases,
stdout/stderr, 250 ms telemetry, process inventories and topology. Each stage's
`receipt.json` hashes its raw files. The bank-wide
[manifest](native-pro6000-20260920/files.sha256.json) hashes every canonical payload.
The three `test.log` files are stored losslessly as `test.log.gz`, preserving libtest's
trailing blank lines; decompress them before checking their canonical hashes. Their
decompressed bytes were checked with the repository boundary-policy evaluator. All other files are
stored verbatim under their canonical names.
Both native executables were also copied off-box and their hashes verified; they are kept
privately, not embedded in this public repository.

## Lease release and scope

All three wrappers finished with exit code 0 and child exit code 0, no timeout, no
interruption and no lingering compute. An independent final `/proc` check found no
surviving task wrapper PID and no FLOCK row owned by any of them. See the
[release record](native-pro6000-20260920/lock-release.json). All completed receipts and
both binaries were copied off-box and verified; all task GPU leases are released.

The GLM-DSA microfixture yielded **`partial_key_bytes=0`**: it did not instantiate a lazy
index-key plane. GPU coverage of KDA rank allocations and lazy index-key allocations
remains pending. Their arithmetic has CPU/source coverage only, as does the conservative
peer-prefill workspace estimate. The small fixture is not the official GLM-5.3-Flash
checkpoint and does not promote model support or change rewrite-certificate semantics.

Actual CUDA OOM recovery, full-checkpoint serving pressure, token/numeric-program identity
through source-lease replay, the affected release exactness batteries, and performance qualification remain
pending. Issue #544 stays open and the PR stays draft until the required remaining gates
are resolved. Earlier interrupted attempts produced no GPU results; inaccessible prior
build metadata was not reused or counted as qualification.
