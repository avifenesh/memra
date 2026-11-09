# c8-workspace-20261108: the prefill workspace a prime holds, against what admission charged

Branch `c8r2/prefill-held` (PR #1114, on #1109), started on the fix stack's tip `8633c4bf5d` and measured last on main `0c25263389`. The follow-up
the c8 lane named (`research/c8-serving-20261107/`, `docs/decisions/C8-SERVING-24GB.md`). Decision
records: [`docs/decisions/PREFILL-WORKSPACE-HELD.md`](../../docs/decisions/PREFILL-WORKSPACE-HELD.md)
(item 1) and the 2026-11-08 update in
[`docs/decisions/BATCH-OOM-RECOVER-DEFAULT.md`](../../docs/decisions/BATCH-OOM-RECOVER-DEFAULT.md)
(item 2). Results: [`RESULTS.md`](RESULTS.md).

Card: one RTX 5090 Laptop GPU (24 GB, 82 SMs), balanced platform profile. Model in the c8 cells:
Qwen3.8-27B NVFP4 + Q5_K GGUF with its MTP head, `MEMRA_CTX=32768`, naked defaults unless stated.
The batch-OOM cells use the Qwen3.5 9B NVFP4 MTP GGUF, as DAY49 registered. Local absolute paths in
the receipts are replaced by placeholders, and server logs and telemetry are gzipped (`harness/export_receipts.py`).

## Harness (`harness/`)

- `ws_probe_patch.py` (and `ws-probe.patch`, what it applied to main `2288c4bf82`): the measurement
  build. Around every outermost prime call (`prime_cache`, the walker's `prime_chunk`,
  `prime_cache_batch`) it resets the default pool's `USED_MEM_HIGH`, and after a stream sync logs
  the peak growth during the call, the retained growth and the prime slab before and after
  (`[ws-probe]`). Never landed. `ws_probe_read.py` summarizes the lines; `probe_run.sh` is the
  three-boot run.
- `arm2.sh`, `agentic2.py`, `mixed.py`, `ts.py`, `analyze.py`, `score.py`, `exact_cmp.py`,
  `tables.py`: the c8 lane's harness (`research/c8-serving-20261107/README.md`), unchanged except
  paths and the port. `phase.sh` runs one interleaved phase with one gpulock lease per arm;
  `campaign.sh` chains the phases; `serving_table.py` pools an arm's runs (TTFT, E2E and TPOT
  percentiles, failures, throughput, the free floor, counters from the server log).
- `oom_boots.sh`: DAY49 addendum D's serving shape (`research/spill-b-20260919/day49d-run.sh`) with
  the arm named by the binary, or by the env door on a tree that still has it (`DOOR=1`).
- `census_check.py`, `spark_census.sh`, `spark_census_run.sh`: the git 2.43 census check on a GB10
  unit (#1109).
- `build.sh`, `build3.sh`, `unit.sh`, `checks.sh`: builds under the rig lock and the local checks.
- `group2.sh` runs several phases' arms in one gpulock lease, each arm a fresh boot in the given order. `campaign*.sh` chain the groups (campaign10 to 12 are the scored ones). `grow_check.sh` runs the grow-rewind replay arms base, fix, fix, base in one lease.
- `serving_table.py` pools an arm's runs. `prime_tail.py` gives the prime pass's first-token tail per run. `grow_table.py` and `grow_floor.py` give the grow replay's per-cell table and its memory floor per arm (250 ms telemetry, and the lowest driver-free reading the server logged).
- `grow_ab.sh`, `agent_turns.py`, `summarize.py` and `trace_requests.py` are the grow-rewind lane's harness (`research/pool-grow-dspark-20261108/`), unchanged.

Binaries (`memra-server`, `MEMRA_CUDA_ARCH=120a`, release profile; sha256 prefix):

| name | tree | sha256 |
|---|---|---|
| `probe-2288c4bf82` | main + `ws-probe.patch` | `f18ac3a2594b3aff` |
| `main-2288c4bf82` | main (the env door `MEMRA_BATCH_OOM_RECOVER` still exists) | `217dc3587952b863` |
| `S` | the fix stack's tip `8633c4bf5d` | `9694c523a838600a` |
| `C1` | the first version, `4559347975` (reviewed out; `runs/c8-v1/` only) | `3d5c99deeb6ecb60` |
| `C1b` | the reviewed version, `b868464d9b` (the reserve yield only before the decode step; `runs/c8-v2/` only) | `15b0fd093081d6dd` |
| `C1c` | `47e9d340a8`: the whole reserve yielded before every prime call and batch round (`runs/*-c1c/` only) | `e0ee6417229af73b` |
| `C1d` | `bb7f308cef`: the yield before a plain prime call at the batch formula, no phase (a) yield unless an MTP session primes (reviewed out; `runs/c8-c1d/` only) | `9f1fcc6558db6ca0` |
| `C1e` | `363e888249`: the reserve yield before phase (a), each prime call's own need right before it; the old deadline estimate | `f46c9fcce5ddc3d1` |
| `C1g` | `47f678337f`: C1e plus the deadline estimate that keeps the tick's takes (before its review fixes) | `cbd1a9fd420ac716` |
| `M2` | main `0c25263389` (grow-rewind landed): A of the scored phases | `9413dac84559dd72` |
| `L` | the lane rebased on `0c25263389` (`a99a7a13ef`): B of the scored phases | `65e371c2c9acf48f` |

## Runs (`runs/`)

- `probe/`: the measurement build, three fresh boots in one lease: `c8` (agentic c8, prime pass
  then the measured pass, T=1, 768 new tokens), `c1-prime` (the agentic prime pass at c1),
  `mixed` (the spec-cache-mixed shape, `MEMRA_MAX_SESSIONS=12`). `*/ws-probe.json` holds every
  probe line parsed.
- `oom-main/`: item 2, the eight boots `O1-off O1-on O2-on O2-off P1-off P1-on P2-on P2-off` on
  main (O = spec-default route, P = `MEMRA_SERVE_SPEC=0`), `read-serve.log` the DAY49D reader.
- `c8/`, `g8/`, `mixed/`, `c4/`, `c1/`: the scored A/B, A = `M2`, B = `L`.
- `grow/`: the grow-rewind lane's agentic replay on the DFlash route (`grow_ab.sh`: c1, c2, c4 and c8 in each boot), A = `M2`, B = `L`, arms base fix fix base.
- `prime-c1e-c1g/`: the agentic c8 prime pass alone, A = `C1e`, B = `C1g`.
- `c8-c1g/`: agentic c8 with A = `S` and B = `C1g`, four runs (the rest stopped for the review fixes).
- `c8-c1e/`, `g8-c1e/`, `mixed-c1e/`: the same phases with A = `S` and B = `C1e`.
- `c8-c1c/`, `g8-c1c/`, `mixed-c1c/`: the same phases on `C1c`, an earlier yield (the whole reserve before every prime call and batch round).
- `c8-c1d/`: two arms (`01-A`, `02-B`) on `C1d`, stopped when review found its gaps.
- `c8-v2/`: three pairs on `C1b`, the run with the prefill out-of-memory failures.
- `c8-v1/`: one pair on the first version, a diagnostic, not scored.
