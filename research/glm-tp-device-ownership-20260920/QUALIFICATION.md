# Issue #544 native qualification runner

The coordinator owns rental and the canonical per-card lock wrapper, `memra-gpu-run`.
This runner does neither. Each invocation runs exactly one test in one fresh process, with exactly its
physical GPU set locked. Multi-card locks must be acquired in stable order by that wrapper.
The owner's current per-card instruction supersedes the older whole-rig lock convention
for these runs. No host or wrapper was available when this protocol was prepared.

## Resource envelope

- Native seam stages: one card for `same-device`; two distinct cards for `pair` and `worker`.
- Primary target: RTX PRO 6000 Blackwell, 96 GiB per card. B200 is also a native Memra
  target; receipts on it qualify that device seam, not PRO performance or release gates.
  A conservative allocation floor for these tiny fixtures is 24 GiB per card; this is a
  provisioning allowance, not a measured minimum or peak VRAM claim.
- Linux x86_64, Rust 1.97.1, CUDA 13.1 or newer (13.2 preferred); 16 vCPU, 64 GiB CPU RAM,
  150 GiB local build/scratch space. Build parallelism defaults to 8 jobs.
- No model download for the seam stages. The engine test generates a deterministic
  approximately 100 KB GLM-DSA micro GGUF (seed 544) and records its SHA-256. This proves
  loaded ownership and allocator behavior, not full GLM-5.3 checkpoint qualification.
- A later full-checkpoint stage uses the official FP8 source
  `zai-org/GLM-5.3-Flash@04c4e9e95c5da8862dced7e5056455116f83a7e0`:
  the checked corpus census names 328,326,771,576 bytes (328.3 GB). Two 96 GiB cards cannot
  hold those source weights fully resident. Capacity/placement must be preflighted separately;
  the historical NVFP4 mint is not a substitute. Do not download that artifact for the tiny
  seam run or describe the seam run as checkpoint-faithful serving qualification.

## Build before taking GPU locks

Use this branch in a clean dedicated checkout. Set `COMMIT` to the full reviewed SHA,
`OUT` to a new path under that checkout's ignored `target/` directory, and `ARCH` to the
provided hardware's architecture (`120a` PRO/5090, `100a` B200; `90a`/`89` are compatibility
builds only). The build stage forces empty `CUDA_VISIBLE_DEVICES`, clears inherited
`MEMRA_*`/`DOCS_RS` settings, pins the arch and runs `--no-run`:

```sh
python3 tools/qualify-model-device-memory.py build \
  --expected-sha "$COMMIT" --arch "$ARCH" --jobs 8 --out "$OUT"
```

Add `--nvcc /absolute/path/to/nvcc` if the host's chosen toolkit needs an explicit path.
The runner retains compiler output and an immutable build receipt with test-executable
paths, hashes, compiler identity and commands. Each receipt uses its own fresh
`OUT/cargo-target`, ignoring inherited or default Cargo targets so cached documentation
stubs cannot enter a native build. An existing output directory is refused.

## Run only after the coordinator provides the host and wrapper

The installed interface is:
`memra-gpu-run --gpus <UUIDs-or-host-indices> --receipt <directory> -- <foreground-command>`.
It holds the full set of exclusive per-card `flock` locks in sorted UUID order, preserves
requested UUID rank order in `CUDA_VISIBLE_DEVICES`, and exports `MEMRA_GPU_LEASE_FILE`.
The canonical paths are `/tmp/memra-gpu-locks/<GPU-UUID>.lock`.

Use full physical UUIDs supplied by the host inventory for `GPU_A` and `GPU_B`:

```sh
memra-gpu-run --gpus "$GPU_A" --receipt "$OUT/lease-same-device" -- \
  python3 tools/qualify-model-device-memory.py run \
  --expected-sha "$COMMIT" --out "$OUT" --stage same-device --gpu-uuid "$GPU_A"

memra-gpu-run --gpus "$GPU_A,$GPU_B" --receipt "$OUT/lease-pair" -- \
  python3 tools/qualify-model-device-memory.py run \
  --expected-sha "$COMMIT" --out "$OUT" --stage pair \
  --gpu-uuid "$GPU_A" --gpu-uuid "$GPU_B"

memra-gpu-run --gpus "$GPU_A,$GPU_B" --receipt "$OUT/lease-worker" -- \
  python3 tools/qualify-model-device-memory.py run \
  --expected-sha "$COMMIT" --out "$OUT" --stage worker \
  --gpu-uuid "$GPU_A" --gpu-uuid "$GPU_B"
```

Each line is a separate foreground session; never execute them concurrently on overlapping
sets. The wrapper releases the exact set after the child completes. Keep its receipt,
including other-card activity if collected, alongside the test receipt.

The runner verifies lease UUID order, stable `lock_order`, canonical paths, wrapper/child
ancestry and actual Linux `/proc/locks` FLOCK ownership. It also verifies distinct physical
lock inodes, source/binary hashes, matching compute capability, an empty selected
compute-process set, and the exact ignored test's existence before execution. It records
stdout/stderr, selected-card 250 ms telemetry, process inventories and a hashed receipt.
It will not execute an absent test, accept a skipped test, or overwrite a prior stage.
Source, binary and held locks are rechecked after the native test. No locks are acquired
by the runner itself. `--lock-file GPU-UUID=/canonical/path` is an optional extra cross-check.

## Gates and remaining scope

| Stage | GPUs | Actual surface |
|---|---:|---|
| `same-device` | 1 | Two independent Engine owners on one ordinal are both enumerated and fenced. |
| `pair` | 2 | Loaded GLM TP peer enumeration; old Step-only omission control; lazy MLA allocation accounting; peer-only trim and refill with live state preserved. |
| `worker` | 2 | Production worker headroom/reclaim helpers with asymmetric physical memory pressure and pinned source ownership. |

CPU refusal controls: `python3 tools/test_qualify_model_device_memory.py` (wired into CI).
Local macOS result: 8 passed, 1 explicitly skipped Linux-only live-flock test; no GPU run.
The runner's same-device stage is distinct from a two-card transport proof. The pair
stages are synthetic native ownership/allocator tests. Full checkpoint serving pressure,
source-lease replay token identity and affected kernel/run-gen/run-spec exactness remain
separate required rows in [VALIDATION.md](VALIDATION.md), never inferred from these passes.
