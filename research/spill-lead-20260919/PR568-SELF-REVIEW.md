# PR #568 self-review (lead, 2026-09-20, after the replay onto main `f79b3e57`)

Scope read in full: the 25-file code diff vs `origin/main` (12 memra-engine files, `memra-kv/plane.rs` + `lib.rs`,
memra-tier bank tests, collector Python tests, `tools/tier-battery.py`, `tier-envelope.py`, `tier-rig-bootstrap.sh`,
`docs/decisions/KV-PHYSICAL-RECLAIM.md`). Research receipts were spot-checked against the verifiers, not re-read.

## What the change is
- `memra_kv::KvPlane` owns a KV operand: pooled (`From<CudaSlice<u8>>`, the only constructor the serving path uses)
  or VMM-backed (`KvPlane::vmm`, reserve + per-granule `cuMemCreate/Map/SetAccess`, operand via
  `upgrade_device_ptr`). `KvLayer.k/v` become `KvPlane`. `KvWrite` is the write seam: every engine KV writer takes
  `&mut impl KvWrite` and immediately views it (`kv_view_mut`) so the kernel launches are unchanged; reads go
  through `Deref<Target = CudaSlice<u8>>`. Same pointer, same kernels, no numeric change.
- `CudaTransfers` stores `Rc<RefCell<KvPlane>>`; `take_device` refuses VMM (`Unsupported`, ownership retained),
  `take_plane` moves the typed owner; `release_device_observed` and `device_registry_len` are diagnostics.
- `kv-tier-gate`: `--kv-allocator pooled|vmm` (default pooled) swaps an EMPTY cache's planes to VMM before any
  decode; `--reclaim-diagnostic` (VMM + active + host only) adds a never-mapped spare VA probe; `reclaim_contract`
  is the pure G1 rule (exact vs bounded-no-leak, residual); the active receipt now carries the allocator-aware
  fields. The 8192-only refusal is lifted to `tiers=host` (32k allowed).
- Experts-via-tier qualification: `host_bank_slots` (fail-closed budget → slots, 16-slot / 256 MiB ceiling),
  `TracedDispatch` host-SLRU trace, `bank_pressure` eviction counter, `[expert-gpu-slru]` report on gate drop.
- `h2d-probe --copies 1..100000` with per-operation event intervals summed (n stays 1).
- Collector: `pro-single` rig (one RTX PRO 6000, `/tmp/memra-gpu.lock`), timeout tree / child containment tests.

## Findings
1. **VMM is unreachable from serving.** `KvPlane::vmm` has exactly one caller, `kv_tier_gate.rs:230`; the
   `Deref`/`kv_view_mut` panic fence ("suspended VMM operand cannot be published") can only fire on a
   `demote_prefix`'d plane, which only the gate creates. Production planes are pooled with `suspended == None`
   for their whole life. Door decide-by 2026-10-04 stands in `KV-PHYSICAL-RECLAIM.md`.
2. **Drop order is right.** `KvPlane::drop` leaks the upgraded operand only when a mapping exists, then the
   `Mapping` field drops: bind, synchronize (quarantine on failure), unmap/release per chunk by flag, `cuMemAddressFree`.
   A partially failed `demote_prefix` leaves `suspended = Some`, so the plane can never be published again and
   cleanup still walks every chunk by its own `mapped`/`handle` flags.
3. **Both revuto findings are fixed at the tip** (`a799cf5d`, `eb010c87`): `vmm_fixed_va_restored` is observed
   (counts planes with `Some` before and `Some`-equal after, `not-applicable-pooled` otherwise);
   `g1_reclaim_qualified` is `not-applicable-pooled` for pooled runs and otherwise requires `bounded_no_leak`
   AND a classified residual. `residual_bytes` is also `not-applicable-pooled` for pooled runs.
4. **Nit, not blocking:** `install_expert_bank_gate` reads `--expert-bank-host-bytes=` by scanning
   `std::env::args()` inside the engine library. Its callers are the `run-gen` and `run-spec` gate binaries only
   (`run_gen.rs:1023`, `run_spec.rs:151`; memra-server never calls it) and it is documented as "gate-only CLI budget", but a library function should take the budget as a parameter. Follow-up
   for lane C's next engine-touching commit; no behavior change today.
5. **Nit:** the VMM arm of `demote()` records `source_owners_before/after_release = (1, 1)` as constants rather
   than an observation (the plane is retained by `take_plane`, so the value is true, but it is not measured).
   `reclaim-diagnosis.txt` readers should treat those two fields as pooled-arm evidence only. Follow-up for lane B.
6. **Nit:** `TracedDispatch::demand` unwraps `slru_policy()` twice; the bank is constructed with SLRU three lines
   above so the invariant holds, but `ok_or(Error::Incomplete)` would keep the gate fail-closed instead of panicking.
7. `reclaim_contract::observe` tests cover exact, one-granule-short bounded, leak (`restored != before`), negative
   gain, zero release and zero granularity. `bounded_no_leak` allows `gain > released` (over-release), matching the
   decision record's criterion (a)–(d), which bounds the shortfall, not the surplus.

## Verification on this rig
CPU battery on the replayed tree `4f36a0e7`: `integration-day10/cpu-battery/` (fmt, 261 tier/kv tests, clippy
`-D warnings` on engine/server/tier/kv/gguf, flags census 864 names, publish census 12/12, docs-registry census,
78 collector tests, perf board, `git diff --check`), all green. Local RTX 5090 Laptop `tools/local-ci.sh`
correctness stage on `5b3d0b08`: `integration-day10/local-ci-5090/` (verdict recorded there verbatim).

## Tool halves missing
`revuto` reached its 2-round cap on this PR ("Revuto did not run a review on this pull request"); Bugbot is capped
for the day. Per the owner ruling of 2026-09-16 this self-review is the review; merged with `gh pr merge 568 --merge`.
