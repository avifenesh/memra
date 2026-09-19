# WP-B cells — day-2 CPU pass; GPU cells NONE EXECUTED

## Day-2 executed CPU cells (2026-09-19)

Code tip `7f1a61a09a968ad1386c8c0b5899d8b650c7e27c`; raw output in
`day2-verification.log`, repeat with `python3 research/spill-b-20260919/verify-day2.py`.
40 memra-kv tests pass (19 B tier tests plus 21 existing tests); 36 unchanged shared
contract tests and 3 compile-fail doctests pass. No ignored tests.

| Cell | Actual CPU coverage | Result / limitation |
|---|---|---|
| B1-frozen | Imports/re-exports only; canonical JSON/payload pins unchanged; per-group counts, padding, immutable seal | PASS; no persistent production object migration |
| B1-tier | Frozen `conformance::tier_cancel` run against `Hierarchy` | PASS; fake transfer backend, not GPU/NVMe |
| B1-budget | Lead governor headroom/foreign/double-release/pin schedule replayed against shared B Governor; dimensions, bounded queue, deadline, tenant fairness, dirty backlog, simultaneous mutex reservations | PASS; concrete lead test was not generic/path-callable, no frozen files modified; serving-load fairness pending |
| B1-lifetime | Short/missing/rejected/corrupt/duplicate/failed entries, unknown quarantine, three epochs, zero/partial submit cleanup, cancellation/publication and retirement | PASS; real disk/DMA/consumer/graph observers pending |
| B1-materializer | Frozen trait implementation; one-record and multi-page odd-tail q8_0 K/q5_1 V exact bytes; full identity/order/format checks; allocation retention | PASS CPU binding; no CUDA attention call or complete recurrent continuation |
| B1-hostprefix | Unbound legacy sidecar, full identity binding, model-generation mismatch, rebind/drop invalidation, old handoff untrusted | PASS primitive tests; six-line worker constructor/field wiring not compiled as full server here |
| B4-policy | Linear diagnostic plus calibrated suffix/read/copy/materialization fixture table; mandatory/admitted-state refusal | PASS pure policy; synthetic costs, no selected runtime default |

One read-only development-rig SSH attempt returned exit 255:
`Connection closed by UNKNOWN port 65535`. No second attempt, no inventory returned,
no GPU commands executed remotely. Full server/engine, true transfers, long-context,
serving pressure and performance remain pending exactly as below. See
`HOSTPREFIX-EXTENSION.md` for the native patch map and admission sequence.

## GPU queue (unchanged acceptance bar)

2026-09-19. The Mac remains CPU-only. Local 5090 availability is not established; the lead's
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

## Day-3 delta (supersedes CPU status above; GPU rows still NONE EXECUTED)

Implementation `fa763c97` plus the day-3 fairness follow-up: 47 memra-kv tests pass, including seven new tests for
bounded scheduler interleaving/cancel/rollback/quota/timeout and cross-tenant fairness, direct local/peer
consumer fences, owned immutable sealing, resident-charge host twin retention and
all 48 recompute fixture rows. Shared memra-tier suites run unchanged: bank 27,
contracts 36, peer 10, placement 6, storage 26, doctests 4. CPU checks are in
`day3-checks/`; Linux-target `cargo check --all-targets` ran, **not Linux tests/GPU**.

Direct local/peer tests now bypass HostReady and use one direct descriptor ticket,
with no Ready publication until consumer fencing. The B fake uses retained CPU
DeviceOwners; **it does not instantiate D's private PeerCapacity fake**. D's ten
peer tests ran in the common suite. Export/adapter handoff is still required for
one combined B→D PeerCapacity schedule; that task is not claimed completed.

### Executable rig preparation / refusal runner

`rig-cells-b.sh` delegates to the adjacent Python driver. Its 20-command dry run
uses an actual shell stub; two unit teeth test raw hashes/order/locks/no-GPU labels
and a child exit-7 failure. Fresh output directories are immutable (reuse refuses),
and each live run owns a uniquely named scratch worktree/branch, removed on exit.

```sh
bash research/spill-b-20260919/rig-cells-b.sh --dry-run
python3 research/spill-b-20260919/test-rig-cells-b.py
bash -n research/spill-b-20260919/rig-cells-b.sh

# Only on an approved dedicated Linux development 5090, never a serving box:
bash research/spill-b-20260919/rig-cells-b.sh --exclusive-non-serving \
  --artifact "$ARTIFACT" --artifact-sha256 "$ARTIFACT_SHA256" \
  --tokens-8k "$TOKENS_8K" --tokens-32k "$TOKENS_32K" --fit-plan "$FIT_PLAN"
```

Inputs: local-NVMe staged byte-identical GGUF with its SHA256; prompt JSON arrays
of exactly `context-128` u32 tokens (no model-format substitution); a prequalified
run-gen memory envelope JSON with `artifact_sha256`, full `source_commit`, and
`required_free_bytes: {"8192": <bytes>, "32768": <bytes>}`. Include run-gen's multiple
cache/gate allocations and headroom, not just a one-cache payload formula. Missing
fit envelope or insufficient free VRAM **refuses**, never runs a non-fitting cell.
Pin the tokenizer/template/plan/numeric manifest alongside these inputs before GPU use.

Sequence, before then after patch in scratch: build server/run-gen → full server
lib tests → Qwen 8k and 32k raw-token **baseline** probes → existing short host-prefix
identity/teeth/failure gates. Native active/prefix full-state gates are recorded as
PENDING; without the future native gate the live runner exits nonzero after baseline
collection. A supplied `--active-gate` is a future interface, NOT a current binary;
receipt acceptance/bootstrap is still missing and the live runner still refuses final
qualification. A green command exit is labeled PASS_COMMAND, not model exactness.

All run-gen/future native probes use `/tmp/memra-5090.lock`; existing legacy scripts
own that same lock internally, avoiding nested flock deadlock. GPU work records 250 ms
raw nvidia-smi CSV plus command/exit/raw SHA256 JSONL and failure compute-app snapshots.
Raw logs are written before JSONL. Output is `raw/rtx5090-<utc>-<pid>/` (hardware-shaped
host label deliberately avoids deployment machine identity). Existing legacy gate
subreceipts and server logs remain in that directory. Builds are outside GPU lock;
no simultaneous scored campaign is permitted.

**128k and 262144 are not runner options:** they remain non-serving PRO-pair cells
with `/tmp/memra-gpu.lock`. No performance/default decision is made from this runner's
single correctness attempts. Generic active attention consumption, state/logit/token
hash equality, measured engagement and all original B2/B3/B4 requirements are unchanged.
