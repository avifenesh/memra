# OWED 9: storage-bench stage timing (2026-09-25)

**Landed CPU-verified.** Registration: `CPU-PREREG.md` OWED 9. The frozen `StorageSample` wire
struct is unchanged; its `io_ns` (previously always `null`) is now the summed wall time of
`ExtentStore::read` calls. Put, commit, lease and verification times go to one stderr line:

```text
[storage-bench] stages put_ns=<n> commit_ns=<n> lease_ns=<n> read_ns=<n> verify_ns=<n> total_ns=<n>
```

`read_ns` equals the sample's `io_ns`. Verification (fixture regeneration, byte comparison and
SHA-256) is outside `io_ns` and inside `total_ns`, so B1 scores read time without verification.

## CPU gate (`owed9/verify.py`, raw per-invocation logs in `owed9/run/`, summary `owed9/cpu/verify.log`)

```text
OWED9 VERIFY PASS: 24 invocations (2 phases x 3 modes x 4 sizes), io_ns == stage read_ns in every sample, collector join_storage accepted all 24
```

Sizes 264, 4,097, 1,048,576 and 116,654,080 bytes; modes `buffered`, `uncached` (O_DIRECT reads)
and `direct` (O_DIRECT reads and writes); `roundtrip` then `restore` on each object; every
sample `byte-exact` with `fallbacks = 0`; restore rows carry `put_ns = commit_ns = 0`. The
samples pass the collector's own `join_storage` validation. Run on this rig's ext4 root
(proven `nvme-local-direct` in `M1-PROOF-CONTROLS.md`) with a debug build: these timings are
plumbing, not B1 results. `cargo clippy -p memra-engine --bin storage-bench -- -D warnings`
exit 0; `cargo fmt --all -- --check` exit 0; `git diff --check` clean.
