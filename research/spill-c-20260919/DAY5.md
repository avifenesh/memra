# Day 5 — bounded native PLE host-row gate (development milestone banked)

Repository: avifenesh/memra. Feature branch: lane/spill-c-20260919.

## Source slice and boundaries

`ple_rows_tier.rs` borrows the already-loaded F32/BF16 host table; only the
requested window (at most 4096 rows / 8 MiB expanded output) is indexed. The
native authoritative n-gram IDs are passed unchanged. `BankService` /
`BoundedRowService` use exact byte layouts, a zero-cache forced-miss read pump,
publication, copied host expansion, retirement, release and acknowledgement.
The injected governor is reused for all gate arms; output reservation survives
through the existing H2D consumer. No CUDA pointer or host lease is mislabeled
as device-ready. No source table clone occurs in the gather adapter.

Checksums bind the read to the immutable borrowed host source, **not an artifact
manifest**. The temporary namespace and zero cache cannot authorize persisted
reuse. Loader-wide accounting, persistent row caching, async I/O, GPU readiness,
model-scale source registration and HostExps dispatch remain pending. This is a
native qualification slice, not production tiered PLE serving or an SSD benchmark.

The typed per-model `rows_tier` field defaults to `None`. Only the explicit gate
method can arm it, and it resets OFF on return/error. No environment read was
added. Door decision: **decide-by: 2026-10-03**; remove unless the next native
source/cache milestone justifies retaining the qualification seam.

`--rows-via-tier` adds an arm without altering the existing gate arms. The arm
compares actual GPU PLE output planes **and convolution state** as float bits:
F32/BF16 representation fixtures, prefill, decode, verify chunk, shorter diverging
history, exact/non-exact projection classes, 16 pairs total. Both sides of each
pair use identical bytes and the same numeric class. BF16 is a separate synthetic
fixture, never a substitute for a requested checkpoint program. Each shape starts
fresh convolution state; this is not a serving rollback/state-restoration gate.

## Lead-owned module fragments

Apply to `crates/memra-engine/src/lib.rs` for integration / scratch native build:

```rust
mod ple_rows_tier;
mod banked_residency;
```

`ple_rows_tier` is required by the new hook. `banked_residency` is the separate
unexported HostExps bridge compile probe; no HostExps dispatch is changed. The lane
does not edit lead-owned lib.rs. Until the fragment is integrated, this is a lane
slice rather than a standalone engine-buildable tip. No Cargo dependency changes.

Registry fragment (lead): TESTING entry for the existing `qwen4exp_gpu_gate`
command plus `--rows-via-tier`, frozen bank goldens beside receipt, under the
canonical collector/lock. FLAGS diagnostic-door entry above if cataloging typed
gate-only doors; no new MEMRA name. No kernel/FFI or board numbers changed.

## Evidence

CPU development checks: 3 new native-source bridge tests pass (raw bits including
negative zero / non-finite values, duplicates, F32/BF16, zero-cache repeated reads,
invalid IDs/shapes, caps, partial admission and drained charge), strict tier clippy
passes. Final reproducible CPU and rented-development native receipts follow.
No GPU result was claimed at that initial source milestone.

The follow-up native development cells now pass: existing GPU gate `failures=0`,
row-tier PLE outputs and convolution state bit-identical (16 cases, 6144 values,
144 forced read chunks, drained budget), native Linux bank suite 46/46, and strict
native engine/gate + tier clippy. Power cap was 400 W on a rented RTX 5090.
Raw logs, frozen goldens, collector lock/capture/telemetry, exact source fragments
and scope are in `rented-5090-20260919/day5/RESULTS.md`.
Final Mac CPU checks are in `raw/day5-final-cpu/`: fmt, native and Linux-target
checks, 174 tier tests including doctests, strict clippy, diff and flags all pass.
Lead integration still must apply the two module declarations above. No production
or model-scale serving support state is advanced.
