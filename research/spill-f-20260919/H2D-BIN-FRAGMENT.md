# Lead-owned manifest fragment

Add to `crates/memra-engine/Cargo.toml` on integration (lane F does not edit the
shared manifest):

```toml
[[bin]]
name = "h2d-probe"
path = "src/bin/h2d_probe.rs"
```

Without the fragment, Cargo auto-discovery exposes `--bin h2d_probe` (underscore).
The F-only remote build uses that auto-discovered target; neither a manifest
mutation nor an engine runtime default is needed. The final desired public
spelling is `h2d-probe`.

On Mac the package build is blocked by nvcc/CUDA, **not attempted as a CUDA
qualification**. Compile this exact standalone source with `rustc --edition=2024`
and execute `--dry-run` (no dependencies/CUDA on Mac); the output exercises
all 40 sample shapes and the RESULT record with null GPU measurements.
`rustc --test` exercises CLI refusal and complete-byte comparator red controls.
Linux builds use existing engine `PinnedHostBuf` and cudarc/sha2 dependencies.

`--bytes` selects one registered size; omit it for all ten sizes. `--direction`
is `h2d|d2h|both`; `--order ab|ba` sets allocation-arm order. `--repeats` and
`--copies` accept only `1`: this binary deliberately cannot be mistaken for the
full calibrated, balanced N>=5 G2 experiment. Each arm/direction/size gets two
complete untimed correctness controls (seeds 3 and 71), then one copy sample
(seed 113). The comparator corrupt-byte red control runs before any GPU work.

The pinned arm uses cacheable flags=0, **not** write-combined memory. Host data
is pre-touched; sources seeded and destinations poisoned/reset each visit.
Copies and error paths fence on one owner stream, complete bytes and SHA256
are checked after completion, and all allocations remain alive until fences.
CUDA event time is stream elapsed including host submission gaps (the copy
API/fence completes before the end event is queued), **not DMA-only time**.
Wall time includes event recording and completion. No throughput or medians are
computed. Power snapshots use CUDA UUID solely to select the correct NVML
index; only device index and raw limit strings appear in JSONL.

No new environment reads, kernels, flags, model paths or serving claims.
