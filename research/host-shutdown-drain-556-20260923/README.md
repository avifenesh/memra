# Host-tier identity and graceful shutdown on the #556 tree (PR #556)

Verdict: `KV-HOST-SPILL IDENTITY GATE: ALL GREEN (teeth=0)` and `ALL GREEN (teeth=1)` on the tested source `8a3adc6e7`, one RTX PRO 6000 Blackwell Server Edition, confidential computing on. Every one of the 10 server boots across the three attempts ended with `[server] GPU worker shutdown complete` after SIGTERM.

This is the affected-case check beside the release battery (`research/release-qualification/candidate-pro-ubuntu-24-04/`, bank `ed413a5f8`). #556 routes the worker loop's three channel-disconnect exits through `break 'worker` into the shutdown drains (capture, restore, then demote), so a graceful stop must reach them. The gate exercises the host tier's demote, promote and restore round trip and stops each server with SIGTERM.

## Attempts

| attempt | cell | default arm | teeth arm | clean shutdown |
|---|---|---|---|---|
| 1 | gate defaults (`MEMRA_HOSTGATE_CACHE_MB=1024`) | rc=1, 6 failures | rc=1, 4 failures | 2/2, 2/2 |
| 2 | `MEMRA_HOSTGATE_CACHE_MB=80` | rc=0, ALL GREEN | rc=1, 1 failure | 2/2, 2/2 |
| 3 | `MEMRA_HOSTGATE_CACHE_MB=80 MEMRA_KV_HOST_TENANT_PCT=100` | not run | rc=0, ALL GREEN | 2/2 |

- Attempt 1 was mis-tuned for this model: `FAIL: no device eviction fired: MEMRA_HOSTGATE_CACHE_MB=1024 holds both seed entries`. One 9B seed entry is 53.8 MB (`[prefix-cache] insert (spec-boundary): 64 tokens, 53.8MB (resident 107.6MB / 1074MB, model gate)`), so the cell needs a device budget between 54 and 107 MB. Attempt 2 used 80.
- Attempt 2 default arm: `E_A` demoted (`[prefix-host] demote: 64 tokens, 54.8MB in 52.7ms`), promoted back (`[prefix-host] promote: 64 tokens, 53.8MB in 100.7ms`), verify digest matched, r3 and r1 byte-identical to the tier-off boot, then `[server] SIGTERM: draining (0 in flight, deadline 30s)`, `[server] drain complete in 0.0s; exiting`, `[server] GPU worker shutdown complete`.
- Attempt 2 teeth arm failed one assertion, `FAIL: teeth: demote refused by name (entry > 1 MiB host budget)`. Everything else held: 0 demotions, 0 promotions, r3 cold, bytes identical. The engine refused the demotion by a different name: at the default 50% tenant share the pre-copy share check (memra#384) runs first, `[prefix-host] demote evaporated at the tenant share cap before the D2H copy: 64 tokens, 54.8MB (50% of 1MB, MEMRA_KV_HOST_TENANT_PCT; model gate); reclaim refused: the image alone exceeds the share (54.8MB > 1MB); nothing evicted`. The gate's teeth regex predates that check. The reviewed recipe (`research/spill-d-20260919/DAY8-CELLS.md`) runs this cell with the share check disarmed, which is attempt 3. The gate itself now accepts either named refusal, keyed on `MEMRA_KV_HOST_TENANT_PCT` the way the server reads it (PR #654).

## Conditions

- Tested source `8a3adc6e712a3c23337e28aeb755ab50955ad832`; gate `tools/kv-host-spill-identity-gate.sh` from the same tree; server `memra-server` built from it; model Qwen3.5-9B NVFP4 MTP GGUF. Hashes: `raw/attempt*/inputs.sha256`.
- One RTX PRO 6000 Blackwell Server Edition, driver 595.91.07, CC ON; each arm ran under `memra-gpu-run` on the card's UUID with `MEMRA_GPU_LOCK=/tmp/memra-gpu.lock` (lease receipts under `raw/attempt*/lease/`).

## Raw

- `raw/attempt1-cache1024-untuned/`, `raw/attempt2-cache80/`, `raw/attempt3-cache80-teeth-pct100/`: per arm the gate log, every request's JSON, metrics, both server logs and the lock proof; `conditions.txt` and `inputs.sha256` per attempt.
- `raw/host-556-attempt1.log`, `raw/host-556b.log`, `raw/host-556c.log`: the runner logs with per-arm rc and shutdown counts.
- `raw/vm-host-556.sh`, `raw/vm-host-556b.sh`, `raw/vm-host-556c.sh`: the runners as executed.
