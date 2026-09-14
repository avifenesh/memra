# Reusable matrix workspace

2026-09-05 UTC. Continues the whole-request matrix candidate without changing
its quantization or CUDA arithmetic. Matrix and grouped-prefill arms remain
experimental/default OFF. This is not a serving or performance qualification.

## Change

`dsv4_grouped::GroupedWork` owns persistent routing metadata, two checked
FP8-to-half mirrors (each with data, scales, device status and reusable host
status storage), and the CSR contribution plane. `VerifyWs` allocates this
once at its maximum slot capacity. Calls fill and inspect only their live
prefix, including after larger preceding calls.

This removes seven explicit GPU scratch allocations per routed MoE call:
three for each mirror and one contribution plane. No speedup is inferred from
the allocation count. Remaining route/half validation still synchronizes on
the host; full-layer capture remains refused for the matrix program.

The total GPU byte charge is:

```text
route_bytes = 4 * (3 * experts + 2 + 6 * slot_capacity)
extra_bytes = slot_capacity * (6 * hidden + 2 * intermediate + 16)
workspace_bytes = route_bytes + extra_bytes
```

For 256 experts, 6 slots, hidden 4096 and intermediate 2048 this is 175352
bytes. The GPU test checks the reported value against actual allocation
lengths, not only the same arithmetic formula. Shape and byte-overflow checks
run before allocating the planes.

Current CUDA graph documentation reinforces the stable-storage direction and
the remaining constraint: synchronizing a captured stream is invalid.
https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html

## Local validation

- `matrix-workspace-5090.log`: real CUDA test passes fresh/reused byte equality,
  row counts 8/1/5/8, stable device pointers, NaN-code refusal, stale-error
  isolation on a smaller live prefix, recovery and byte accounting.
- `matrix-workspace-memcheck.log`: the same Rust integration test passes with
  `ERROR SUMMARY: 0 errors`.
- `matrix-workspace-cpu.log`: 402 passed, 0 failed, 8 ignored. The new CUDA
  integration test is included among the ignored tests and was run separately.
- `matrix-workspace-server.log`: 602 passed, 0 failed.
- Strict engine/server/gate clippy and optimized build pass; formatting,
  whitespace and flag census pass.

## Target comparison

The updated `dsv4_matrix_program_gate` retains the whole-request phase checks
and adds persistent/fresh/persistent storage comparison at 160 real tokens and
widths 1/32/64. The fresh arm deliberately allocates zeroed storage per call;
its exclusive gate setter and allocation counter prove engagement. It compares
full logits, live cache classes and DSpark rings. This is a test-only control,
not another serving environment flag.

Candidate SHA256:
`9074ed34a20287906e5e80024fdbeda83f163762c143aa298eb061aadc957233`.

The earlier phase candidate `c5ff8f022b34ccf74a88518ea1269f898822c63c5c6c2cebf158361cd4638767`
remains frozen on the target. Those candidate queue descriptions are historical.
The combined repaired candidate
`7654612ebf1ebde481099096519678f4c6143c4ffa233f019443b7f6b7c3954a`
passed the full phase gate and persistent/fresh/persistent comparison on the
PRO pair at 2026-09-05 23:06:43 UTC. Fresh allocation calls were 6880/258/172 at
widths 1/32/64; the reused controls recorded zero. Full logits, live cache
classes and DSpark rings agreed. See `matrix-request-program.md` for the two
integration repairs preceding the pass. No throughput or quality admission is
inferred from this result.
Original 512K/1M, host-C4, concurrency, topology and peak-performance goals are
unchanged.
