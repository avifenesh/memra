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

## Plain-decode levers, served (2026-11-03, 12:56-15:06)

The same agentic cell at c = 8, 16 and 32 only, one boot per build: model arm, warm-up then the three cells. The boot log reads `decode wave cap 32; scheduler tick cap 32 (exact-32 tier)`. The spec cost table now covers plain steps 1..32 (`plain_ms[16]` = 148.3 ms, `plain_ms[32]` = 268.4 ms at depth 2048). The gate served plain at every n >= 8.

| c | before the levers (00:23 run, b16 rp) | + b16 rpsc + 32-row tier (`lever2/`) | + batched-tick FA deep walk (`lever3/`) | vLLM |
|---:|---:|---:|---:|---:|
| 8 | 60.4 | 62.5 | 61.8 | 96.7 |
| 16 | 67.1 | **76.0** | 75.8 | 121.4 |
| 32 | 67.8 | **79.1** | 79.0 | 135.5 |

Aggregate generated tok/s, one run per build. Binaries: lever 2 at bfbf46e486, lever 3 at 58398a2a18.

- c=16 gains 13%, from the b16 rpsc schedule.
- c=32 gains 17%, from rpsc plus the one-wave 32-row tick.
- c=8 rides the b8 tier and is unchanged within run noise.
- The FA deep walk is flat here. It pays at long depth (-9.6% at depth 32768, 8 sessions); this cell's median turn is about 11k tokens, where the bench measured -1.7%.

The remaining gap to vLLM at c >= 16 is the dp4a decode matmul, which is latency-bound at every width, and attention at about 59% of bandwidth. GB10-PLAIN-DECODE.md has the per-kernel numbers.

## Exact tensor-core b32 tier, served (2026-11-03, 20:54-21:29)

A fresh boot of the server built at 35897adf14 (the same code tree as 0e914ccf4c). The boot wave probe measured two 16-row waves at 275.6 ms, one 32-row wave at 265.3 ms on dp4a and 232.4 ms on the tensor-core tier, so it chose wave cap 32 with the tensor-core b32 tier on (`gb10-agentic/tc-b32/`).

| c | before the levers | lever 3 (dp4a b32) | fp32 base, c1..c32 sequence | **tensor-core b32** | vLLM |
|---:|---:|---:|---:|---:|---:|
| 8 | 60.4 | 61.8 | 62.3 | 62.4 | 96.7 |
| 16 | 67.1 | 75.8 | 76.0 | 75.8 | 121.4 |
| 32 | 67.8 | 79.0 | 78.6 | **83.6** | 135.5 |

c=32 gains 5.8% over the dp4a b32 tier. c=8 and c=16 do not move, as designed: the tensor-core tier serves only 17..32-row waves, and it is bit-identical anyway.

## Spec gate at c=2: stale spec prices, three variants (2026-11-04)

With bf16 GDN state the c=2 cell served 23.8-23.9 tok/s in two runs against 29.0 on fp32. The gate log showed why. A K=3 price was extrapolated from a K=5 slot measured in a low-acceptance stretch. Plain won the tie, no spec tick ran at n=2 again, and nothing corrected the price. Three fixes were served on fresh boots, each with the memset removal and the tensor-core b32 tier (receipts `gb10-gate/`):

| variant | sha | c1 | c2 | c4 | c8 | c16 | c32 | verdict |
|---|---|---:|---:|---:|---:|---:|---:|---|
| v1: spec slots older than 64 s stop pricing; probe within the plain EMA's noise | 2ac067a1a9 | 22.58 | 28.25 | 38.70 | **49.21** | 71.35 | 83.35 | rejected: with no stale ratio anchors the raw table priced c8 spec at 98.7 against a realized 49.3 |
| v2: prices keep every slot; probe within 2 SE of the plain EMA | 781ecc64ef | 22.45 | **23.95** | 39.80 | 62.25 | stopped | | rejected: c2 still plain (K=3 at 25.8 vs plain 26.1; the plain EMA's SE alone is too small to call that a tie) |
| **v3: spec slots priced at the live acceptance; probe within 2 SE of the difference** | **062ecac658** | **22.60** | **28.94** | **38.10** | **62.10** | **75.77** | **83.11** | **adopted** (default build, fp32) |

v3, two changes:
- Each spec slot records the acceptance window's expected tokens per round at its K when sampled, and is priced at expected-now over expected-then. A slot measured in a low-acceptance stretch recovers as soon as any spec round, at any concurrency, moves the window.
- The tie band is two standard errors of the difference: the plain EMA's SE and the largest relative SE among this n's spec slots, in quadrature.

The bf16 arm on v3 (`caed240c24` = combo + v3, `MEMRA_GDN_STATE_BF16=1`, fresh boot, c2 then c8):

| c | bf16 + TC, old gate (19b3495bab) | bf16 + TC, v3 gate |
|---:|---:|---:|
| 2 | 23.89 | **27.98** |
| 8 | 60.80 | 59.42 |

c2 recovers on both arms (fp32 28.94, bf16 27.98, against the 29.0 fp32 reference).

c8 holds on fp32 (62.10 against 62.4-62.6). On bf16 it is 59.42, inside the 59.4-62.6 band the c8 cell has shown across today's runs:
- that arm ran c8 straight after c2, with no c4 step;
- the n=7/8 verdicts first speculated at the table price (98-103 tok/s predicted) until n=8 K=1 was realized at 53.6;
- after that it served plain.

One run per arm; c2 has varied 23.8-29.0 between runs of one build before, so this is a direction, not a tight number.

Default-build rerun (memset removal + v3 gate + tensor-core b32 chosen at boot), against the earlier default:

| c | previous default, tensor-core b32 (0e914ccf4c tree) | default now (062ecac658) | vLLM |
|---:|---:|---:|---:|
| 1 | 22.34 (fp32 dp4a run) | 22.60 | 24.2 |
| 2 | 29.02 (fp32 dp4a run) | 28.94 | |
| 4 | 37.64 (fp32 dp4a run) | 38.10 | 68.7 |
| 8 | 62.41 | 62.10 | 96.7 |
| 16 | 75.84 | 75.77 | 121.4 |
| 32 | 83.61 | 83.11 | 135.5 |

Served, the memset removal is flat within single-run noise: the cell's decode ticks are mostly shallower than 12288, and the tick gain there is 2.6-3.9%.

## bf16 GDN state arm, rebased onto every exact lever (2026-11-04)

`research/combo-tc-gdnbf16` was rebased (cherry-picked, no merges) onto batch-spec 0d0ac2d8e4. Head 62492dac54; the pre-rebase head caed240c24 is kept on `research/combo-tc-gdnbf16-premerge`.

The arm now carries every exact lever plus `MEMRA_GDN_STATE_BF16=1`:
- the tensor-core 32-row tier;
- the memset removal;
- the v3 spec gate;
- the exact-width 16-row twin.

Served on a fresh boot, c1..c32 (`gb10-combo2/`). The boot probe picked the tensor-core 32-row tier (207.7 ms against 238.5 for two 16-row waves).

| c | fp32 default now (062ecac658 run, c8/c16 from 88805bc1f1) | bf16 arm, rebased | bf16 + TC, before the gate fix (19b3495bab) | vLLM |
|---:|---:|---:|---:|---:|
| 1 | 22.60 | 23.50 | 23.17 | 24.2 |
| 2 | 28.94 | 28.52 | 23.89 | |
| 4 | 38.10 | 38.80 | 39.51 | 68.7 |
| 8 | 62.35 | **50.88** | 60.80 | 96.7 |
| 16 | 77.27 | 78.78 | 77.49 | 121.4 |
| 32 | 83.11 | **86.06** | 85.46 | 135.5 |

The bf16 arm leads at c16 (+2.0%) and c32 (+3.5%), and c2 is fixed. c8 regresses. Both bf16 runs on the v3 gate speculated at n=7/8 for a stretch: 4 and 5 K=1 verdicts, priced at 60-63 tok/s from the lower-concurrency anchors while the plain EMA had dipped to 57-60. They realized 53.6 and 56.6 against plain 72-73 before switching back. This run also hit one acceptance-history "explore" verdict at n=8. The fp32 runs never speculated at n=8.

So the gate's spec price at n>=7 is too optimistic when it is extrapolated from n<=4. That is the next gate fix. It does not touch the decode path. One run per arm.
