# GB10 agentic receipts: Qwen3.8-27B, owner GGUF

Date: 2026-11-03, 00:23-04:44 Jerusalem.

- **Device:** one DGX Spark (GB10, sm_121a), whole box, under the memra lease.
- **Model:** memra's NVFP4 + Q5_K GGUF trunk with the FR-Spec-masked (sxc32768) GGUF MTP draft. This is the `gguf-mtpmask` serving arm: derived unified-memory defaults, MEMRA_CTX=262144, 32 sessions.
- **Cell:** `ab_agent.py` from the shared harness, via `bench_memra.sh` with QUALITY=0.
  - Workload: 48 turns cut from 24 real coding-agent sessions, tool definitions included, histories up to 200k characters.
  - Sampling: thinking on, T=1, seed 1234, max 768 new tokens. top_p and top_k are not sent, so the model defaults apply.
  - Each cell primes all 48 prompts first. One boot per arm, with a warm-up of turnbench at 8K and 32K, then c = 1, 2, 4, 8, 16, 32.
  - The client is on the rig and reaches the server through an ssh tunnel.
- **Binaries:**
  - batch-spec a6a480639f, memra-server sha256 `e81be05ff1021484…`
  - base spark-sm121 b014549cb9, built from a clean `git archive`, sha256 `684befd1cb63f307…`
- **Raw data:** `gb10-agentic/<arm>/` (cells.jsonl, run.log, server lines) and `gb10-agentic/receipts.log`.

## Aggregate generated tok/s

| c | base (spark-sm121) | band (batch-spec, `MEMRA_SPEC_GATE_MODEL=0`) | model (batch-spec defaults) | graph (`MEMRA_POOLED_GRAPH=1`) | vLLM |
|---:|---:|---:|---:|---:|---:|
| 1 | 21.4 | 21.5 | 21.5 | 21.5 | 24.2 |
| 2 | 23.5 | 23.5 | **28.8** | 28.3 | — |
| 4 | 39.9 | 39.9 | 38.4 | 39.5 | 68.7 |
| 8 | 58.2 | 60.4 | 60.4 | 60.3 | 96.7 |
| 16 | 66.7 | 67.0 | 67.1 | 66.7 | 121.4 |
| 32 | 67.7 | 67.4 | 67.8 | 67.7 | 135.5 |

Per-user median gen tok/s is in `cells.jsonl`. For the model arm it was 23.5 / 15.0 / 10.5 / 8.2 / 4.5 / 2.4. The vLLM numbers are the bench lane's reference runs of the same cell on the same box, same 27B weights (vLLM has no c=2 reference).

## What the gate did (model arm, `gb10-agentic/model/server-lines.txt`)

- **Boot table:**
  - verify_ms over 1-16 rows: 75.6, 74.7, 74.9, 76.0, 77.1, 78.2, 79.8, 81.4, then 88.1 … 136.5.
  - plain_ms: 74.8, 78.6, …
  - round overhead: 13.31 ms alone, 3.65 ms per session in a 4-session cohort.
- **Realized steady rates, spec vs plain:**

  | n | spec | plain |
  |---:|---|---:|
  | 2 | 45.8 tok/s (K=3, pooled) | 21.6 |
  | 4 | 41.3 (K=2, pooled) | 40.1 |
  | 8 | not tried (plain, 69.6) | 69.6 |
  | 16 | not tried (plain, 74.4) | 74.4 |
  | 32 | not tried (plain, 76.4) | 76.4 |

- **Verdicts:** spec at n = 1-3, a near-tie at n = 4 (realized spec 41.3 vs plain 40.1), plain at n >= 8.
- **Cohorts:** pooled cohorts of 2-4 and batched-draft cohorts of 2-4 engaged.
- **Riders:** the unified step engaged once (1 spec + 1 plain). The gate keeps load all-spec or all-plain, so riders are rare. Their cap from this table is 8 rows, the flat region.

## Verdicts

1. **Pooled spec plus the cost gate pays where speculation pays on this workload: c=2 at 1.23x** over base (28.8 vs 23.5). The steady regime at n=2 runs 2.1x plain (45.8 vs 21.6).
2. **c=4 is a tie.** Pooled K=2 realizes 41.3 vs plain 40.1, and the wave measured 38.4 vs 39.9. One run, inside noise; a validation run is below.
3. **c >= 8 is set by the plain batched step.** All arms plateau at 67-68 tok/s from c=16, while vLLM reaches 121-135. The gap at high concurrency is the plain batched decode path on GB10 at agentic depths (histories to 200k characters). It is not speculation: the gate correctly serves plain there, and the realized plain rate at n=32 is 76.4 tok/s. This is the next lever: the plain step's attention and per-row cost at long context on sm_121a.
4. **Why pooled verify gains less here than at depth 2048.** At these depths a verify row's attention cost grows with context. GB10-CR.md measured 4x4 pooled verify at 208 ms at depth 16k, against 146 ms at 2k. The flat 1-8-row region of the depth-2048 table does not hold at agentic depths, and the realized-rate pricing is what keeps the gate honest about that.
5. **`MEMRA_POOLED_GRAPH`: deleted.** Graph replay is flat on GB10, as on the 5090: 28.3 vs 28.8, 39.5 vs 38.4, 60.3 vs 60.4, 66.7 vs 67.1. The forward is device-bound on both cards, so launch overhead is not the cost. The door and the graph code are removed. UNIFIED-STEP.md keeps the v2b receipts as the record of the negative result.
6. **Rider caps** stay as priced: GB10 caps at the 8-row flat region, the 5090 near 8, and no riders without a table. On this workload the gate rarely creates mixed ticks, so riders are not a GB10 throughput lever.

## Validation run, and the fix it forced (05:14-05:54)

The model arm ran again on a fresh boot, with the c=1 cell left out to stay inside the box's slot: warm-up, then c=2 and c=4. The base arm's rerun was skipped to keep the slot under an hour.

| c | model, first run (after c=1) | model, validation (no c=1) | base | band |
|---:|---:|---:|---:|---:|
| 2 | 28.8 | **23.4** | 23.5 | 23.5 |
| 4 | 38.4 | 39.9 | 39.9 | 39.9 |

The validation boot had no acceptance history when c=2 started (`validation-model/server-lines.txt`). It had no verdict, so the band decided, and the band served the c=2 cell plain. That is the band arm's own 23.5 tok/s. The sessions never speculated, so the history never filled, and the gate never took over.

In the first run, the c=1 cell built the history, and the gate then ran c=2 as pooled spec (28.8 tok/s).

The 1.23x at c=2 is real once the gate has history. A fresh boot facing concurrency first never got there. The fix (`spec_cost_verdict`): a calibrated model with no acceptance history now speculates at its rung whenever every session's verify rows fit one pooled forward (n * (rung + 1) <= 16). The history then fills within a few rounds, and the realized rates take over.

## Fresh-boot c=2 with the exploration fix (08:34-09:04, 2026-11-03)

One boot of the model arm at 1e30f67c69, warm-up and then the c=2 cell only, no c=1 before it (`gb10-agentic/validation-explore/`).

| c | model, fresh boot with exploration | model, after c=1 | base | band |
|---:|---:|---:|---:|---:|
| 2 | **24.7** (20161 tokens, 816.7 s) | 28.8 | 23.5 | 23.5 |

The gate now engages without history. Each first verdict at n=1 and n=2 reads `exploring: no acceptance history yet` and speculates at its rung. After that the realized rates price every verdict (16 priced verdicts at n=2, from 22.4 vs 22.4 up to 66.7 vs 25.4). Most verdicts land close to break-even, so the gate switches between K=0 and K=2..5 inside the cell.

The fresh boot gains 1.05x over base, against 1.23x when the c=1 cell has filled the history first. The gap is the cost of building history on this cell's own turns.
