# WP-B queued GPU cells — NONE EXECUTED

2026-09-19. Day-1 Mac is CPU-only. Local 5090 availability is not established; the lead's
launch probes failed. Required non-serving PRO pair/artifact permission remains lead-owned.
No serving instance, paused model path, external runtime or format substitution is a cell.

## Common acceptance and safety

- Pin Qwen3.8-27B artifact byte manifest, serialized plan, tokenizer/template, prompt token
  arrays, binary hash, numeric/stream class and actual native q8_0 K / q5_1 V. Opaque packed
  fixtures never stand in for a model gate. Stage byte-identical artifacts on local NVMe.
- Same binary, same numerical program for no-tier/host/SSD. State bytes (valid, not padding),
  full logits and nonempty token-id streams must hash identically. Record both reasoning
  and content text; an empty content-only hash is not an output proof.
- At least one forced active demote/load DURING generation, with nonzero bytes/objects and
  attention consuming the restored operands, not merely park/resume. Prefix restore also
  requires nonzero demote/promote bytes. Missing engagement fails the cell.
- Record topology, peer grants, free memory, concurrent GPU processes and source/binary
  identity before/after. Capture 250ms telemetry, raw stdout/stderr first, JSONL second.
- One campaign per box. Build outside GPU lock; GPU commands use the canonical lock only.
  Pre-start sccache before any lock if needed so a daemon cannot inherit the GPU lock.
- Performance only after correctness: >=5 AB pairs +5 BA pairs, warmup/cold/warm/cache
  regime explicit. TTFT/E2E/TPOT/ITL p50/p95/p99, throughput, actual useful/physical bytes,
  queue/inflight/dirty backlog and failures. No default from a single arm or synthetic rows.

## Commands and budgets

`ARTIFACT`, `SERVER_BIN`, `GATE_BIN`, `EV` are caller-provided non-secret local paths.
`GATE_BIN` below is a **PROPOSED future native kv-tier-gate executable**, NOT present in this
slice. These commands specify the required runner contract and cannot run until materializer
and server adapters exist. No claim of executable GPU gate readiness. The existing host
prefix gate command below IS present; it covers a narrower legacy whole-prefix surface only.

| Cell | Command specification | Wall budget | Rig / lock |
|---|---|---|---|
| B2-active-8k | `flock /tmp/memra-5090.lock "$GATE_BIN" --artifact "$ARTIFACT" --case active --context 8192 --tiers host,nvme --same-program --out "$EV/active-8k"` | 20 min correctness | fitting 5090 development; then PRO pair using `/tmp/memra-gpu.lock` |
| B2-active-32k | same command, `--context 32768 --out "$EV/active-32k"` | 30 min | fitting 5090; PRO pair blocking |
| B2-active-128k | `flock /tmp/memra-gpu.lock "$GATE_BIN" --artifact "$ARTIFACT" --case active --context 131072 --tiers host,nvme --same-program --out "$EV/active-128k"` | 60 min | non-serving 2x RTX PRO 6000 Blackwell |
| B2-active-262144 | same command, `--context 262144 --prompt-tokens 262016 --generate 128 --out "$EV/active-262144"` | 120 min | non-serving PRO pair, `/tmp/memra-gpu.lock`; actually reach committed 262144, not allocation-only |
| B2-prefix-{8k,32k,128k,262144} | active command with `--case prefix-restore`, same respective context and tier args; leave an admitted suffix/output within total context | 20/30/60/120 min | 5090 fitting subset; full ladder PRO pair canonical lock |
| B3-churn | `flock /tmp/memra-gpu.lock "$GATE_BIN" --artifact "$ARTIFACT" --case churn --context 32768 --tiers host,nvme --same-program --out "$EV/churn"` | 45 min | PRO pair, lock as shown |
| B3-spec-cancel | same command `--case rollback-cancel --spec-k 1,2,3,4,5,6,7,8` | 60 min | PRO pair, `/tmp/memra-gpu.lock`; unsupported shapes must explicitly refuse |
| B3-errors | same command `--case corrupt-short-partial-stale-epoch-purge --pool-smaller-than-object` | 45 min | PRO pair, `/tmp/memra-gpu.lock` |
| B4-chunk-frontier | same command `--case frontier --chunk-tokens 64,128,256,512 --orders AB,BA --pairs-per-order 5` | 6 h reserved campaign, stop at first correctness failure | 5090 fitting + separately booked PRO pair |

Legacy prefix baseline (not B2 active proof), known script owns its lock internally:

```sh
MEMRA_GPU_LOCK=/tmp/memra-gpu.lock bash tools/kv-host-spill-identity-gate.sh \
  "$ARTIFACT" "$SERVER_BIN" "$EV/legacy-prefix"
MEMRA_GPU_LOCK=/tmp/memra-gpu.lock MEMRA_HOSTGATE_TEETH=1 \
  bash tools/kv-host-spill-identity-gate.sh "$ARTIFACT" "$SERVER_BIN" "$EV/legacy-teeth"
```

Budget: 30 minutes for both legacy arms, non-serving PRO pair. No outer flock around this
script (nested locking would deadlock). Its current prompt/byte comparisons do not replace
full-state/logits/token gates and its short prompts do not prove the long-context ladder.

## Fault expectations

Optional corrupted prefix: refuse that record, then cold only with same-program proof before
admission. Sole active backing corruption: fail request loudly; never turn it into a miss.
Short/rejected item: whole promised load incomplete. Cancel/purge during I/O: no publication,
lease and memory remain until disk/DMA/consumer fences; late completion stays rejected.
Grow/shrink/rollback: immutable old generation retained, current epoch checked; ring wrap and
required tails complete. Eager/batch/graph crossings require exact same-program receipt or
explicit refusal. Graph operands/counters retain stable addresses.

## Pending bookings

M2 chunk/frontier review 2026-09-23 remains a booking, not a measurement. Lead must supply
rig window and immutable artifact lock before any GPU launch. B/C/D shared-pressure campaign
is booked by lead only; these per-cell budgets are not authorization to run in parallel.
