# Day-8 collector spellings — restricted-power development only

These are individual, inspected cells, not a batch that treats refusals as passes.
All GPU work is under `tools/tier-battery.py`; legacy scripts require the reviewed
external-lock fragment. An accepted capture is archive integrity, not qualification.
Run from the exact lane checkout used to build each named binary, record source and
binary hashes, and retain raw logs. The available development power envelope is
400 W current / 600 W maximum, re-read in each capture. No N=1 performance median.

## A — actual native transfer adapter

Requires A's lane containing `tier-transfer-gate`; it is absent from D's current
integration baseline. Build the native target before invoking it. No model argument.

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

B's current CLI admits this spelling, but the active/native tier binding is still
unimplemented: exit 2 plus final `REFUSED: <reason>` (or the older
`kv-tier-gate: REFUSED: <reason>`) is a **refused** capture, not a successful active
spill gate. Generic `Error:` plus exit 2 remains **failed**.

```sh
python3 tools/tier-battery.py --rig rtx5090 --timeout 1800 \
  --out "$EV/b-active-8192" --execute \
  target/release/kv-tier-gate --artifact "$QWEN" --case active \
  --context 8192 --tiers host,nvme --same-program --out "$EV/b-active-state"
python3 tools/tier-battery.py --validate "$EV/b-active-8192/CELL.jsonl"
```

The collector command returns nonzero for refused/failed children. Inspect that
result before validating retained evidence; do not use a `set -e` chain that hides
the refusal receipt. Never call the preceding refusal native-active qualification.

## C — row tier and prospective device publication

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

**Blocked, do not execute as a positive cell yet:** C's inspected lane does not
implement `--device-publish`; its current argument scanner can silently ignore it.
After C implements and tests strict admission, the exact combined collector spelling
is below. A green old binary with an ignored argument is not device-publish evidence.
Use a fresh state directory and the same immutable goldens precondition as above.

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

## G2 — current N=1 probe versus full envelope

```sh
python3 tools/tier-envelope.py --rig rtx5090 --probe target/release/h2d-probe \
  --bytes 4096 --direction h2d --rounds 1 --correctness-only --out "$EV/g2-n1"
```

The runner owns the collector invocation; do not nest it inside another collector.
The current probe accepts `--repeats 1 --copies 1` only. N=1 AB/BA is plumbing and
byte-identity evidence. **Full G2 is blocked on probe `--copies >1` support**, then
same-window N>=5 in both orders with complete 250 ms telemetry. Calibration must
select one fixed count and leave calibration samples out of scoring. Do not publish
N=1 medians or claim the full 4 KiB–1 GiB bidirectional envelope from a tiny cell.
