# cx-shmown results

Date: 2026-08-13
Branch: `lane/cx-shmown`
Base: `6aba8b2e59bf417e1af689cb8ad11ad2f87d7f24`

## Verdict

PASS for the CPU-only lane. The reported defect was confirmed and repaired: the
CPU expert cache no longer adopts an unsafe POSIX shared-memory object, persisted
rows cannot construct pointers outside the mapped data region, and sampled payload
corruption invalidates the entry before it is exposed as a warm hit.

## Repair

- `ShmArena` first attempts atomic self-creation with
  `O_RDWR | O_CREAT | O_EXCL`, then forces the new object to mode `0600` (including
  under a restrictive umask). If the name already exists, the opened descriptor is
  accepted only when `fstat` reports `st_uid == geteuid()` and permission bits
  exactly `0600`. Any ownership/mode/stat/open refusal is named and the arena stays
  disabled, so `RawBlockPool` uses private allocations.
- Persisted rows are rejected before pointer arithmetic when lengths are invalid,
  the range begins before the arena data region, `shm_offset + pool_bytes`
  overflows, the range ends past `segment_bytes`, or its storage geometry is not
  4 KiB aligned. A rejected row is absent from `WeightCache`, so the ordinary miss
  path re-reads the source.
- The persisted index is version 2 and adds one 64-bit FNV-1a sample checksum per
  entry. Entries up to 32 KiB are hashed completely. Larger entries hash eight
  evenly spaced 4 KiB windows, including the first and last windows: at most
  32 KiB per entry, or about 0.2 GiB for the historical ~6,700-entry warm set,
  instead of scanning a 16 GiB arena. Reload recomputes the sample after bounds
  validation and before constructing the cache-owned pointer.
- Version 1 arenas intentionally reopen cold once; a clean version 2 shutdown is
  warm on the following process.

The checksum is a bounded corruption detector, not a MAC. Cross-tenant protection
comes from refusing foreign/permissive ownership; a process running under the same
uid and able to rewrite both payload and index is outside that ownership boundary.

## Refusal receipts

The focused suite created real POSIX shm objects. Its foreign object was owned by
subordinate uid/gid `100000:100000`, not mocked:

```text
precreated owner=100000:100000 mode=666
[memra-cpu] shm cache REFUSED existing /memra-shmown-foreign-...: uid=100000 mode=0666; require uid=1000 mode=0600; using private cache
PRIVATE_FALLBACK_OK
```

The two unsafe offset classes were refused before `pointer_at` could return a
pointer:

```text
[memra-cpu] shm cache REFUSED persisted entry 0: range exceeds segment_bytes (...); treating as miss
[memra-cpu] shm cache REFUSED persisted entry 0: shm_offset + pool_bytes overflows (...); treating as miss
MISS_REREAD_OK
```

Payload mutation in the first sampled window was likewise rejected:

```text
[memra-cpu] shm cache REFUSED persisted entry 0: sampled checksum mismatch; treating as miss
MISS_REREAD_OK
```

## CPU verification

| Gate | Result | Receipt |
|---|---|---|
| Production companion build (`tools/build_cpu_expert_companion.sh`) | PASS, SHA-256 `03a5eede43aa35efec3efaf6d468b02f8ce64001f5644836bc65d4c80e8626e3` | `raw/build-companion.log` |
| Focused real-shm suite (`TMPDIR=/home/avifenesh/tmp-lanes tools/test_cpu_expert_shm.sh ...`) | **ALL GREEN**: same-uid `0644` refusal/private fallback; foreign-uid `0666` refusal/private fallback; restrictive-umask self-create at `0600`; clean warm hit; past-end, overflow, and checksum rejection plus source re-read | `raw/*.log` |
| Focused suite under AddressSanitizer + UndefinedBehaviorSanitizer | **ALL GREEN**, leak detection disabled only for the companion's intentional process-lifetime singletons | `raw/sanitized/*.log` |
| Existing native companion check (`cargo run -p memra-engine --bin cpu_native_check`) | **ALL GREEN**: ABI/quant checks, cold+warm file-backed identity, detached prefetch identity, and multi-row identity | `raw/cpu-native-check.log` |
| Existing native companion check, two processes with shm enabled and a stable source fixture | **ALL GREEN**: cold process persisted 15 entries; second process reopened 15 warm entries, recorded 24 hits, and performed zero demand projection reads | `raw/cpu-native-shm-cold.log`, `raw/cpu-native-shm-warm.log` |
| Rust CPU-expert unit tests (`cargo test -p memra-engine --lib cpu_experts::tests`) | PASS, 4 passed / 0 failed | `raw/cargo-test-cpu-experts.log` |

The Rust build used `CUDA_VISIBLE_DEVICES=` and `MEMRA_CUDA_ARCH=120a`; nvcc compiled
the existing translation units, but no GPU was visible or executed.

## Explicitly not run

`kernel-check`, `run-gen`, and `run-spec` were **not run**. They are GPU gates and
are outside this CPU-only lane. Neither box1 nor the local RTX 5090 was contacted,
no GPU lock was acquired, and this result does not claim the exactness battery.

No performance board was edited. No merge, tag, or push was performed.
