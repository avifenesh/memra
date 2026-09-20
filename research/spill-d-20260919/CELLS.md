# D qualification queue — day 2, all GPU rows PENDING

No GPU/model execution. Day-2 approved development-rig SSH inventory attempt exited 255
with `Connection closed by UNKNOWN port 65535`; no remote command execution established.
One attempt, no retry, no PRO address approved. This expected connectivity failure is not a
day-2 CPU milestone blocker. Check-in: 2026-09-20 for pair access/artifact
locks; target check-in 2026-09-26 and run at arrival+1 day. These are bookings, not claims.
No serving instance, DSv4 0731 cell, V4.1 adapter or format substitution.

## Invocation and lock contract

Build outside the GPU lock. Use the lead-approved non-serving rig and a new receipt namespace.
Two lock names only: RTX 5090 `/tmp/memra-5090.lock`; PRO pair/four `/tmp/memra-gpu.lock`.
One scored campaign per **whole box**, not one per card. The lock holder spans the entire
ON/OFF correctness and AB/BA window. `flock --close` prevents build/cache daemon descendants
inheriting the GPU lock; no nested self-locking shell gate inside another lock acquisition.

For an individual binary that does not acquire its own lock, Linux invocation template:

```sh
# BIN, ARGS, OUT, approved artifact and binary hashes must be filled from the campaign manifest.
# Example is NOT run by day-1 CPU scaffold.
set -o pipefail
flock --exclusive --nonblock --conflict-exit-code 75 --close /tmp/memra-gpu.lock \
  timeout --signal=TERM --kill-after=15s 300s "$BIN" "${ARGS[@]}" 2>&1 | tee "$OUT/raw.log"
rc=${PIPESTATUS[0]}
# Save rc BEFORE parsing raw.log; rc=75 (lock busy) or timeout is NOT a correctness failure/pass.
```

Shell gates owning their own canonical lock run standalone, serially scheduled, with no
wrapper lock. Multi-arm campaign driver must hold one canonical lock and invoke **non-locking
binary** entry points beneath it. Native integrated GPU runner and real instrumented counter bindings are pending; the day-2
collector protocol, raw-log/failure lifecycle and synthetic sampler are CPU-tested; `tier-battery.py --plan` does not acquire a GPU or run commands.

Before/after each campaign and on failures capture verbatim topology, concurrent compute PIDs,
link gen/width, NUMA and local-NVMe mount/device inventory. Read-only topology commands are in
BASELINE.md. Full `box-health.sh` includes GPU work and belongs under the scheduled campaign,
not read-only SSH inventory. SSH retries <=2 per run, bounded timeout/backoff.

## Existing executable gate commands

Shell variables below are **required manifest inputs**, not automatic artifact discovery:
`STEP_FP8` = pinned official Step-3.7-FP8 safetensors directory; `QWEN` = admitted pinned
Qwen3.8 artifact with its required qualified MTP attachment; `DEVICES` = ordered local ordinals;
`OUT` = new local raw receipt directory. Models are staged byte-identically from `/data` to
local NVMe `/scratch`, with byte manifest retained. No model path is invented here.

| Cell | Command (inside the appropriate exclusive campaign) | Initial wall budget | Rig / lock | Required observation |
|---|---|---:|---|---|
| D1-local-bytes | `target/release/pp-transport-smoke` | 5 min | 5090; `/tmp/memra-5090.lock` | Same-device exact primitive only; explicitly NOT P2P. |
| D1-peer-bytes | `env MEMRA_PP_DEVICES=0,1 target/release/pp-transport-smoke --runtime-probe-cycle`; repeat `1,0` | 10 min/direction | PRO pair; `/tmp/memra-gpu.lock` | Nonzero native-peer copies, patterned bytes, both directions, grant/event logs, zero hidden bounce. |
| D2-Step-PP-decode | `env MEMRA_PP_DEVICES=0,1 target/release/decode-batch-gate "$STEP_FP8" --mode pp --stages 2 --batch 1,4,8 --reps 5`; reverse devices | 45 min/placement | PRO pair; `/tmp/memra-gpu.lock` | Same-program logits, row order, position/high-water; all accepted source tensor classes unchanged. |
| D2-Step-PP-spec | `env MEMRA_PP_DEVICES=0,1 target/release/decode-batch-gate "$STEP_FP8" --mode ppspec --stages 2 --ts 2,5,9 --reps 5` | 45 min | PRO pair; `/tmp/memra-gpu.lock` | K=1/4/8 columns + hidden seed; manifest refusal is not pass. Extend every applicable K at integrated battery. |
| D2-Step-PP4 | `env MEMRA_PP_DEVICES=0,1,2,3 MEMRA_PP_WAVE=1 MEMRA_PP_OVERLAP=1 target/release/decode-batch-gate "$STEP_FP8" --mode pp --stages 4 --batch 1,2,4,8,16,24 --reps 5` | 60 min | Four PRO cards; `/tmp/memra-gpu.lock` | Non-vacuous wave counters and native P2P, serial-wave identity. Pair prerequisite first. |
| D2-Qwen-standard | `target/release/run-gen "$QWEN"`; `env -u MEMRA_SPEC_K -u MEMRA_PROMPT_DIR -u MEMRA_GEN_ONLY target/release/run-spec "$QWEN"` | 60 min initial fitting cell | 5090 fitting then PRO pair; respective canonical lock | Existing argmax/self-consistency with same numeric program. B must pin the qualified MTP attachment before launch. `run-spec` refuses if no MTP/NextN head; optional token-ID arguments are NOT a drafter path. No guessed fallback head. |
| D3-kernels | `target/release/kernel-check` | 15 min | Non-serving PRO pair; `/tmp/memra-gpu.lock` | Full gate, explicit declared skip/refusal accounting; no scoped-mode green substituted. |
| D2-step-pro-source-seal | `python3 tools/check_hardware_gate.py --receipt "$STEP_PRO_RECEIPT" --repo-root . --base "$INTEGRATION_BASE"` | 2 min CPU validation, after hardware rows | CPU; no GPU lock | Existing mandatory kernel/topology/official-model receipts + changed engine-file hashes. No skip override. Does not itself execute GPU gates. |

**Source check prevents a format trap:** `ppn-gate` at baseline opens `GgufFile` directly
(`ppn_gate.rs:138`). It cannot consume official FP8 safetensors. `decode-batch-gate` uses
`SafetensorsSource` for directories (`decode_batch_gate.rs:225–234`) and provides PP/PPspec
controls. A separately pinned existing GGUF eager `ppn-gate` is supplementary generic
regression coverage only, never the official FP8 deliverable. Official eager-PP adapter
coverage remains a D2 integration task if the standing official-model bundle does not cover it.

## New adapter cells queued, command binding intentionally unresolved

These commands are **plan output now**, not pretend GPU executables:
`python3 tools/tier-battery.py --plan` enumerates each below. After A/B/C interface freeze,
the lead binds the named case to an engine gate binary / serving harness and adds its exact
CLI/binary hash to the campaign manifest. No zero-test `cargo test <missing-name>` green.

| Case names | Gate / budget | Rig / lock | Required red/green observation |
|---|---|---|---|
| `grant-failure`, `link-downgrade` | D1, 10 min each | PRO pair; `/tmp/memra-gpu.lock` | Fake/owner-controlled refusal injection only, never changing shared driver/PCI settings. Directed context/pool denial blocks; unknown/active downgrade not classified healthy. |
| `timeout`, `late-completion`, `destination-reuse`, `source-free` | D1, 10 min each | PRO pair; `/tmp/memra-gpu.lock` | Retain both allocation pins across timeout/cancel; late completion cannot publish into new epoch; no reuse before DMA+consumer completion; quarantine unknown. |
| `graph-address-stability` | D1, 15 min | PRO pair; `/tmp/memra-gpu.lock` | Stable local materialization target address over capture/replay/churn; graph pins prevent recycling until graph destroyed. Unsupported graph crossing explicitly refused. |
| `qwen-peer-blocks`, `boundary`, `churn`, `spec-rollback`, `cancel` | D2, 30 min each initial short cells | PRO pair; `/tmp/memra-gpu.lock` | Qwen native bytes, all mandatory auxiliary pools, forced real migrations while active; no QSA remote scatter. 8192 QSA cap unchanged; Qwen3.8 full 262144 gate B-owned. |
| `four-tier-pressure`, `tenant-purge`, `corrupt-active`, `pool-smaller-than-object` | D3, 60 min combined | PRO pair; `/tmp/memra-gpu.lock` | Qwen/Hy3/PLE demand through GPU0/peerGPU1/host/local NVMe; real tier counters >0; governor fairness; no deadlock, wrong tenant or optional fallback for sole active backing. |
| `all-directed-routes` | D3, 10 min per directed edge, 120 min budget | Four PRO cards; `/tmp/memra-gpu.lock` | All 12 routes measured separately; ordered source/destination grants/bytes, direct-route evidence and no full-cache hidden replicas. |
| `shared-fabric-30min` | D3/G2, 30 min steady state + 15 min warmup/drain | Four PRO cards; `/tmp/memra-gpu.lock` | One campaign drives simultaneous pressure, not independent tenants. Demand <=70% **measured** route/SSD rate; bounded dirty/queue backlog. Admit lower envelope only by owner decision. |
| `placement-predicted-vs-peak`, `owner-replica-accounting`, `capacity-refusal` | D4, 30 min | Pair first, four PRO last; `/tmp/memra-gpu.lock` | Match per-device allocation/peak, loader/scratch/staging/replica bytes. Refuse before oversubscription; label estimates and unresolved workspace. |

## Four-tier evidence manifest / JSONL

`runs.schema.json` is the **positive exactness row** schema. `tools/tier-battery.py --validate
<bundle/runs.jsonl>` checks each raw file's length/hash, nonempty state/f32-logit/u32-token
outputs, ON movement>0/OFF movement=0, exact arm pairs, same runtime/artifact/plan/layout/
numeric/prompt identity, canonical rig lock labels, and no fake/host-bounce P2P claim.
These checks compare submitted evidence only; they do not authenticate hardware provenance,
prove an actual held lock, establish numeric truth, count a complete cell roster, or verify
model support. GPU rows now require the version-1 telemetry fields. Native GPU collector
bindings and full campaign acceptance remain pending.

- Faults/refusals retain attempted command, exit status, exact stderr quote and concurrent
  GPU state in raw logs, not disguised as positive output rows. Sole active corruption fails
  loudly; optional prefix corruption may return same-program cold behavior only before admission.
- Final campaign manifest must name required cell roster, source/binary and artifact locks,
  storage mount class, layout and auxiliary bundle manifests, cache/thermal regime, raw hashes,
  expected transitions, actual route bytes and explicit unsupported-shape refusals.
- After forced correctness, performance window is >=5 AB pairs **plus** >=5 BA pairs (N>=10
  observations/arm), 250ms GPU clocks/power/temp/VRAM/PCIe + host/SSD/queue telemetry.
  Store per-request TTFT/E2E/TPOT/ITL and p50/p95/p99 summaries, request/token throughput,
  useful/physical/migrated bytes, failures and warm-up boundaries. p99 from tiny N descriptive.
- No performance verdict/default, G0–G7 GO or support promotion from day-1 fixtures. End-to-end
  active reload/model consumption is mandatory; prefix park/copy/allocator-only success is not it.

## Day-2 CPU collector and placement/probe cells (executed; not GPU gates)

| Cell | Command | Evidence / outcome boundary |
|---|---|---|
| D0 frozen peer conformance | `cargo test -p memra-tier --offline` | 36 frozen contract + 10 D peer + 6 D placement tests; 3 compile-fail doctests. See CONFORMANCE-REVIEW.md: reusable PeerCapacity schedule absent, proposed to lead. |
| D0 collector adversarial tests | `python3 -B -m unittest discover -s crates/memra-tier/tests/battery -p 'test_*.py'` | 16 CPU tests; exact bytes, raw failures/timeout/descendant drain, canonical-lock contention, 250ms callback cadence, order/N/thermal/hash/schema red controls, placement and topology. |
| D0 dry campaign | `python3 -B tools/tier-battery.py --dry-run --out research/spill-d-20260919/day2-dry-run` | Real canonical **local Mac** flock held across fake commands; no remote/GPU lock claim. One forced control pair first, then interleaved five AB + five BA pairs: N=10/arm. Three **virtual** 250ms samples/run; one deliberately failing subprocess before correctness, excluded from medians. |
| D0 retained campaign check | `python3 -B tools/tier-battery.py --validate-campaign research/spill-d-20260919/day2-dry-run` | Exact order/cardinality, hashes, state/logits/tokens, telemetry window, thermal/N/median and quoted failure checked. `CPU CAMPAIGN MATCH: 22 runs`; not GPU qualification. |
| D4 arithmetic only | `python3 -B tools/tier-placement.py`; `python3 -B tools/tier-placement.py --record-bytes 264 --json` | PLACEMENT-TABLES.md reproduces 07 §3 class/grid/frontier tables for both 10/10/10/10 and 9/11/9/11. Explicit owner streams/floors and weight census, not layer-count-only capacity. Override is opaque sizing, no format promotion. |
| D1 probe dry-run | `python3 -B tools/tier-topology.py --dry-run crates/memra-tier/tests/battery/topology.fixture.json --out <new-json-path>` | Six read-only captures: topo matrix, nvidia-smi -q, P2P read/write capability, CPU/NUMA inventory. Fixture text retained verbatim; route_qualification=false. |

`telemetry.schema.json` v1 contains cumulative per-device route bytes in/out (local, PCIe-P2P,
host-bounce, host, NVMe), pinned/pageable host occupancy, NVMe queue depth/read/write bytes,
nullable physical bytes, device clock/power/temperature/VRAM, and p50/p95/p99 io/h2d/d2h/p2p/
queue waits. Missing/negative/nonmonotonic data refuses; native gaps above 500ms refuse.
Sampler requests 250ms ticks; actual timestamps are retained rather than made evenly spaced.
Synthetic sampler clocks/power/temperature/VRAM are **null**, not invented GPU observations.
Counters in this dry run are synthetic, physical SSD bytes unknown, thermal regime is
`synthetic-no-thermal-measurement`; every median names N and this regime. Tiny-N percentiles
are descriptive only. No synthetic timing is published to the engine performance board.

Raw merged stdout/stderr is teed/flushed before parsing and hashed afterward. Failure records
quote captured errors, retain exit code/timeout and provenance; CPU failures explicitly have
no concurrent GPU process query. The native failure adapter must capture real compute-apps
at failure, and native serving cells still need TTFT/E2E/TPOT/ITL distributions and request/token
throughput. The fake cannot fulfill those real instrumentation requirements.

Topology inventory remains diagnostic, never a grant: live context/pool grants, byte-path proof,
and measured link/route/fabric pressure still belong to D1/G5. Scoped topology observations bind
direction, both owner contexts, binary and topology digest; any change refuses old observations.
Keep QSA remote-scatter refusal unchanged. PP is never TP, host bounce never P2P.

The official Step ladder above is unchanged: `decode-batch-gate --mode pp/ppspec` consumes
pinned official FP8 safetensors. `ppn-gate` remains **GGUF-only / supplementary**. No PRO pair
address, artifact permission, CUDA toolchain or GPU was available here. G0–G7 remain pending.

## Day-4 offline validation and storage join

- `python3 tools/tier-battery.py --validate <runs.jsonl>` auto-detects byte receipts;
  `--validate <telemetry.jsonl>` auto-detects standalone telemetry. Optional explicit
  `--schema runs|telemetry` refuses wrong input. Both checked-in JSON schemas are enforced
  offline, followed by semantic checks (pair hashes, routes, monotonicity/cadence).
- A's canonical `StorageSample` does not have a run id. Wrap each retained row as
  `{"run_id":"<D-run-id>","sample":<unchanged StorageSample object>}`. Do not infer the
  mapping by line order or fixture label. Then run
  `python3 tools/tier-battery.py --validate <runs.jsonl> --storage-samples <wrapped.jsonl> --out <new-joined.jsonl>`.
  Unknown run ids, missing run coverage, duplicate run ids, wrong counters/checksum/version
  refuse. Multiple samples per run are retained, including failures, fallbacks and null
  physical bytes. A successful join is not a performance or hardware receipt. An A runner
  without run ids needs an explicit raw-log-to-cell mapping, not a canonical schema change.
- `--first-hour` emits the first 5090 hour's correctness-only plan; future performance
  scheduling still requires >=5 pairs in **each** order. CPU timing never enters boards.
- Native `--execute` appends durable CELL start/end rows while raw logs are flushed;
  `--resume --out <prior-cell>` reads the last CELL and reruns identical argv into a new
  attempt, never treating an interrupted cell as pass. See RIG-DAY1.md for idempotence and
  private provider/cost metadata rules. A/B/C self-locking runners run standalone, not here.

## Day-5 first-rental capture integrity (not a positive gate)

`--validate <CELL.jsonl|command.capture.json|receipt-directory>` also checks archived
subprocess evidence. This mode reports **integrity**, not byte exactness: failed commands
and empty diagnostic telemetry are retained as such, never promoted into `runs.schema.json`.
Explicit `--schema runs|telemetry` still rejects capture journals. See
[DAY5-VERIFICATION.md](DAY5-VERIFICATION.md) for the nine real first-hour capture results.

New capture/end rows include `started_utc`, `ended_utc` and `elapsed_seconds` for the
collector window (snapshots/sampler included); device and StorageSample timings are
separate. For storage use `--storage-root <actual-path>`; unproven storage requires explicit
`--allow-unproven-storage` and retains the overlay/unproven label in every receipt. This
opt-in is development exactness only, not a waived local-NVMe or spill-speed gate.

The current launch and recovery sequence is [RIG-DAY1.md](RIG-DAY1.md): mandatory C
pre-streaming goldens copy/hash, pidfile rather than process-name matching, separate
current/max PCIe interpretation, and off-box receipt sync after **every** cell. Interrupted
cells stay incomplete, artifacts require a pinned locator/hash, and a replacement rig
requires fresh bootstrap/source/binary acceptance.
