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

A native Mac package CUDA build is unavailable and is not CUDA qualification.
The Linux-target `DOCS_RS=1` engine/bin typecheck passed with the explicit
compile-only archive-hash sentinel; native Linux release build and the
N=1 collector receipt are recorded in `H2D-RESULTS.md`. Compile this exact standalone source with `rustc --edition=2024`
and execute `--dry-run` (no dependencies/CUDA on Mac); the output exercises
all 40 sample shapes and the RESULT record with null GPU measurements.
`rustc --test` exercises CLI refusal and complete-byte comparator red controls.
Linux builds use existing engine `PinnedHostBuf` and cudarc/sha2 dependencies.

`--bytes` selects one registered size; omit it for all ten sizes. `--direction`
is `h2d|d2h|both`; `--order ab|ba` sets allocation-arm order. `--repeats` accepts only `1`; `--copies` accepts `1..100000` (default `1`).
Copies are operations inside one arm visit, never independent observations;
calibration and balanced N>=5 execution belong to D's `tools/tier-envelope.py`. Each arm/direction/size gets two
complete untimed correctness controls (seeds 3 and 71), then one visit containing the requested number of copies
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

Each visit emits bare JSON (`record=sample`), with `order=ab|ba`, total
`completed_bytes=bytes*copies`, and `event_ms` summed from independently
synchronized per-operation owner-stream intervals. This intentionally matches
D at `7f7bf547`: `tier-envelope.py` parses bare JSON, while `tier-battery.py`
rejects multiple line-start `RESULT ` tokens. The final probe summary is also
bare JSON (`record=RESULT`); only the outer collector worker emits one
`RESULT `-prefixed aggregate. Do not prefix individual visits.
