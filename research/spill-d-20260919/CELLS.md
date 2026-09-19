# D qualification queue — day 1, all GPU rows PENDING

No GPU/model/SSH execution by this session. Check-in: 2026-09-20 for pair access/artifact
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
binary** entry points beneath it. The actual integrated GPU runner, telemetry collection and
failure lifecycle are pending; `tier-battery.py --plan` does not acquire a GPU or run commands.

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
model support. Full GPU collector/campaign acceptance is pending.

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
