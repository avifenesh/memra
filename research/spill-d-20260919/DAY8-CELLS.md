# Day-8/9 collector spellings — restricted-power development only

These are individual, inspected cells, not a batch that treats refusals as passes.
All GPU work is under `tools/tier-battery.py`; legacy scripts require the reviewed
external-lock fragment. An accepted capture is archive integrity, not qualification.
Run from the exact lane checkout used to build each named binary, record source and
binary hashes, and retain raw logs. The available development power envelope is
400 W current / 600 W maximum, re-read in each capture. No N=1 performance median.

## A — actual native transfer adapter

The day-9 integration registers `tier-transfer-gate` and imports A's native
adapter. Build the native target before invoking it. No model argument.

```sh
python3 tools/tier-battery.py --rig rtx5090 --timeout 300 \
  --out "$EV/a-transfer-conformance" --execute \
  target/release/tier-transfer-gate conformance
python3 tools/tier-battery.py --rig rtx5090 --timeout 300 \
  --out "$EV/a-transfer-roundtrip" --execute \
  target/release/tier-transfer-gate roundtrip
python3 tools/tier-battery.py --validate "$EV/a-transfer-conformance/CELL.jsonl"
python3 tools/tier-battery.py --validate "$EV/a-transfer-roundtrip/CELL.jsonl"
```

## B — active gate, with explicit refusal retained

B's day-8 native source `c9569169` implements the host-only active surface.
The 8k receipt is **copy/restore bit-identical, no reclaim — not G1 PASS**;
32k stays unrun until B's physical-reclaim gate passes. `--tiers host,nvme`
is not the executed active program. Use the exact artifact and byte manifest
from B's `DAY8.md`, and the native binary built from its pinned source.

```sh
python3 tools/tier-battery.py --rig rtx5090 --timeout 1800 \
  --out "$EV/b-active-8192" --execute \
  target/release/kv-tier-gate --artifact "$QWEN" --case active \
  --context 8192 --tiers host --same-program --out "$EV/b-active-state"
python3 tools/tier-battery.py --validate "$EV/b-active-8192/CELL.jsonl"
```

The collector command returns nonzero for refused/failed children. Inspect that
result before validating retained evidence; do not use a `set -e` chain that hides
the refusal receipt. After an actual 8k G1 PASS, the corresponding larger spelling
is `--case active --context 32768 --tiers host --same-program` with new cell/state
directories. It is an admission instruction, not evidence that 32k ran.

## C — row tier and device publication

The existing `--rows-via-tier` cell is:

```sh
mkdir -p "$EV/c-rows-state"
cp research/qwen4exp-bringup-20260829/gpu-eager/bank-bytes-goldens.tsv \
  "$EV/c-rows-state/bank-bytes-goldens.tsv"
python3 tools/tier-battery.py --rig rtx5090 --timeout 1800 \
  --out "$EV/c-rows-cell" --execute \
  target/release/qwen4exp_gpu_gate "$EV/c-rows-state/rows.tsv" --rows-via-tier
python3 tools/tier-battery.py --validate "$EV/c-rows-cell/CELL.jsonl"
```

C's day-seven device-publication implementation and receipt are now in the
integration. Use a fresh state directory and copy the same immutable goldens as
above before the combined spelling. Keep the underscore binary name exactly.

```sh
python3 tools/tier-battery.py --rig rtx5090 --timeout 1800 \
  --out "$EV/c-device-cell" --execute \
  target/release/qwen4exp_gpu_gate "$EV/c-device-state/rows.tsv" \
  --rows-via-tier --device-publish
python3 tools/tier-battery.py --validate "$EV/c-device-cell/CELL.jsonl"
```

## Legacy HostPrefix — external lock, same binary/model

Build `memra-server` from B's applied HostPrefix-v2 lane in D's own clone. The
`LEGACY-EXTERNAL-LOCK.diff` fragment is already applied in D's current tree;
copy these applied scripts and keep `port-guard.sh` and `tier-lock-proof.py`
beside them. For older checkouts, apply the fragment exactly once and verify it
with `git apply --reverse --check`. The copy, patch, source,
and binary hashes are part of the receipt. Do not wrap an unpatched self-locking
script; do not nest flock. The collector owns the lock across both server arms and
teardown. `@COLLECTOR_LOCK_FD@` must be one literal standalone argument.

```sh
python3 tools/tier-battery.py --rig rtx5090 --timeout 1800 --external-lock \
  --out "$EV/legacy-identity/cell" --execute \
  env MEMRA_HOSTGATE_CACHE_MB=256 MEMRA_KV_HOST_TENANT_PCT=50 \
  bash "$GATE_TOOLS/kv-host-spill-identity-gate.sh" \
  --external-lock @COLLECTOR_LOCK_FD@ "$QWEN" "$SERVER" "$EV/legacy-identity/state"
python3 tools/tier-battery.py --rig rtx5090 --timeout 1800 --external-lock \
  --out "$EV/legacy-teeth/cell" --execute \
  env MEMRA_HOSTGATE_CACHE_MB=256 MEMRA_KV_HOST_TENANT_PCT=100 MEMRA_HOSTGATE_TEETH=1 \
  bash "$GATE_TOOLS/kv-host-spill-identity-gate.sh" \
  --external-lock @COLLECTOR_LOCK_FD@ "$QWEN" "$SERVER" "$EV/legacy-teeth/state"
python3 tools/tier-battery.py --rig rtx5090 --timeout 2400 --external-lock \
  --out "$EV/legacy-failures/cell" --execute \
  env MEMRA_HOSTGATE_CACHE_MB=256 MEMRA_KV_HOST_TENANT_PCT=100 \
  bash "$GATE_TOOLS/kv-host-spill-failure-gate.sh" \
  --external-lock @COLLECTOR_LOCK_FD@ "$QWEN" "$SERVER" "$EV/legacy-failures/state"
python3 tools/tier-battery.py --validate "$EV/legacy-identity/cell/CELL.jsonl"
python3 tools/tier-battery.py --validate "$EV/legacy-teeth/cell/CELL.jsonl"
python3 tools/tier-battery.py --validate "$EV/legacy-failures/cell/CELL.jsonl"
```

The 256 MiB diagnostic budget is **MEMRA_HOSTGATE_CACHE_MB**, the device prefix
cache, not the host tier budget. B's passing bare cells were identity-r3 (tenant
50%), teeth-r2 (100%), and failures (100%). Teeth overrides the host tier to 1 MiB;
identity/failure retain the gate's 8192 MiB host-tier default. Compare those exact
conditions rather than changing the diagnostic subject. These legacy whole-prefix
surfaces do not prove generic active/prefix TierManager integration.

## C — native expert pressure and host-record refusal

Run with C's source `44f87f181bbbcc75a14d8d9362609f60186f293d` binaries and
its pinned Qwen3.6 expert artifact (`$EXPERT_ARTIFACT`), not B's dense Qwen
artifact. `$C_BIN` and `$C_TOOLS` are operator-owned immutable copies of that
source's binaries and `research/spill-c-20260919/pressure-refusal.py`.
The requested 8/4 GiB budgets apply to GPU SLRU, **not the host bank**;
9986/4993 slots include eight-byte tail padding and round down. The host bank
remains 256 MiB / at most 16 records. All eight cells use prompt ids 55,88,13,
32 generated tokens, existing numerical programs, and resident experts disabled.

```sh
for budget in 8g 4g; do
  if [ "$budget" = 8g ]; then slots=9986; else slots=4993; fi
  for arm in on off; do
    extra=()
    if [ "$arm" = on ]; then extra=(--experts-via-tier); fi
    for mode in gen spec; do
      python3 tools/tier-battery.py --rig rtx5090 --timeout 1800 \
        --out "$EV/c-$budget-$mode-$arm" --execute \
        env MEMRA_MOE_RESIDENT=0 MEMRA_MOE_SLOTS="$slots" MEMRA_NGEN=32 \
        "$C_BIN/run-$mode" "$EXPERT_ARTIFACT" 55 88 13 "${extra[@]}"
      python3 tools/tier-battery.py --validate "$EV/c-$budget-$mode-$arm/CELL.jsonl"
    done
  done
done
python3 tools/tier-battery.py --rig rtx5090 --timeout 1800 \
  --out "$EV/c-host-refusal" --execute \
  python3 "$C_TOOLS/pressure-refusal.py" \
  env MEMRA_MOE_RESIDENT=0 MEMRA_MOE_SLOTS=4993 MEMRA_NGEN=32 \
  "$C_BIN/run-gen" "$EXPERT_ARTIFACT" 55 88 13 \
  --experts-via-tier --expert-bank-host-bytes=1
# Expected collector exit 2; inspect it, then validate separately:
python3 tools/tier-battery.py --validate "$EV/c-host-refusal/CELL.jsonl"
```

The only expected terminal refusal is:
`REFUSED: experts-via-tier host bank budget cannot hold one expert record`.
The wrapper retains native output and native exit status; unrelated failures,
build errors, timeouts and signals are not refusals. This is a host-record
minimum refusal, not a GPU-minimum-budget gate (`MEMRA_MOE_SLOTS=0..7` clamps
to eight slots in that source). See C's `DAY8.md` for all eight ON/OFF verdicts
and separately counted GPU/host evictions, physical reads and re-reads.

## G2 — calibrated N=1 rehearsal versus scored envelope

D imports F's final `7644c41f` probe source unchanged. The integrated Cargo
manifest explicitly names the binary **h2d-probe** (F's isolated auto-discovered
build used **h2d_probe**); both use `h2d_probe.rs`. CLI:
`[--dry-run] [--bytes BYTES] [--direction h2d|d2h|both] [--order ab|ba]`
`[--repeats 1] [--copies 1..100000]`. Probe visits emit bare JSON; only the
outer worker emits one line-start `RESULT ` token.

```sh
for bytes in 4096 16777216; do
  for direction in h2d d2h; do
    python3 tools/tier-envelope.py --rig rtx5090 --probe target/release/h2d-probe \
      --bytes "$bytes" --direction "$direction" --rounds 1 --correctness-only \
      --out "$EV/g2-$bytes-$direction"
  done
done
```

The runner owns the collector invocation; do not nest it inside another collector.
It calibrates the faster arm toward 350 ms (both measured visits must be at least
250 ms), then fixes one count for AB followed by BA. Calibration is discarded,
copies do not increase N, and the four cells run only once as
**executed-not-qualified** plumbing. No medians. Nested probes preserve the
collector worker group and inherited lock FD. Both outer timeout and inner timeout
kill that group; commands must not detach/daemonize. This is cooperative job
containment, not a sandbox for hostile programs.

A fully scored N>=5 same-window run remains a later lead call. The full 20-cell
4 KiB–1 GiB envelope, sustained-duration, thermal and serving gates are not
satisfied by this rehearsal. Peer day-8 archive integrity replay is reproducible:

```sh
python3 research/spill-d-20260919/validate-day8-archives.py --out "$EV/day8-replay"
```

It extracts immutable A `day7/`, B `day8-active-8192/`, C `day8/`, and F
`h2d-copies/` archives to disposable scratch, runs the actual collector
`--validate` over all CELL journals, retains raw output and source hashes,
and removes the extraction. It does not modify peer receipts or re-execute GPU
work. Expected summary:

```text
CAPTURE ARCHIVES MATCH: 14 cells; 12 executed-not-qualified; 1 failed command; 1 refused command; qualification=false
```
