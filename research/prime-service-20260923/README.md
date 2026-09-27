# Saved-prime service interval (memra#521, PR #654): the PRO 6000 A/B

Verdict: PASS on every clause. `tools/prime-fairness-gate.py --shape service`, main `5f1b0eda4` (base) against PR #654 `e5167f577` (cand), six boots per arm with the order alternating each rep, on one RTX PRO 6000 Blackwell Server Edition:

```
PRIME-SERVICE: base=base cand=cand reps=6 bytes=yes rate=8.73/52.15 ev/s long_ttft=28.09/46.92s second_ttft=14.12/27.20s itl_p99=190/384ms engaged=yes -> PASS
```

While a saved prime runs, the streaming decoders' event rate rises from 8.73 to 52.15 ev/s (p50 over 12 decoder windows per arm). The price is paid by the primes: the 131k prime's TTFT goes from 28.09 to 46.92 s (+67%) and the 32k prime's from 14.12 to 27.20 s (+93%), inside the declared cost bar of 2.0x base + 5 s (61.17 s and 33.24 s).

Decision record: `docs/decisions/PRIME-FAIRNESS-DEFAULT.md` (the service-interval addendum). Contract: the `MEMRA_PRIME_YIELD` row in `docs/FLAGS.md` and `docs/PREFILL-FAIRNESS.md`.

## The cell

Qwen3.5-9B NVFP4 MTP GGUF, greedy natural-text streams on `/v1/completions`, `MEMRA_SLO_P99_MS` at its default 50 ms (S), `MEMRA_PRIME_CHUNK` at its default 4096, `MEMRA_PRIME_YIELD` unset on both arms. Two decoders (2,165 prompt tokens, 4,096 output tokens each) start at t=0. A 135,494-token cold prime arrives at +3 s, a 32,815-token cold prime at +13 s and a cold 2,165-token peer at +15 s. The gate records every event time, each decoder's rate and inter-token gaps inside each prime's window, each prime's TTFT and `/health` `tick_max_ms`, with 250 ms GPU telemetry per boot.

Declared before the scored run (`receipt.json` `declared`): service `rate_cand >= 2.0 x rate_base`, cost `ttft_cand <= 2.0 x ttft_base + 5 s` per prime, at least 16 generated tokens per request, bytes identical across all 12 boots, every request finished, the interval engaged on the candidate.

## Receipts (6 reps per arm, p50 unless named)

| metric | base (main) | cand (#654) |
|---|---|---|
| decoder rate inside the 131k window (ev/s, p50 / p95) | 8.73 / 9.60 | 52.15 / 54.76 |
| decoder rate inside the 32k window (ev/s, p50 / p95) | 3.98 / 5.05 | 53.13 / 55.85 |
| decoder ITL inside the 131k window, p95 / p99 (ms) | 662 / 1134 | 60 / 480 |
| decoder ITL inside the 32k window, p95 / p99 (ms) | 1127 / 1258 | 46 / 572 |
| decoder ITL over the whole run, p95 / p99 (ms, n=49,140 each) | 31 / 190 | 46 / 384 |
| decoder TPOT (ms) | 14.43 | 14.67 |
| decoder E2E (s, decoder-0 / decoder-1) | 59.48 / 59.75 | 60.81 / 60.57 |
| 131k prime TTFT (s, p50 / p95) | 28.09 / 28.10 | 46.92 / 47.09 |
| 32k prime TTFT (s, p50 / p95) | 14.12 / 14.15 | 27.20 / 27.27 |
| cold peer TTFT (s, p50 / p95) | 0.85 / 0.86 | 0.28 / 0.34 |
| cold peer E2E (s) | 14.26 | 2.99 |
| `tick_max_ms` (p50 / max) | 1280 / 1330 | 855 / 933 |

The whole-run ITL p99 doubling (190 to 384 ms) is a composition effect, not a slower step. On main the decoders stall through each prime and emit most of their 4,096 tokens after both primes finish, at full rate with short gaps. On the candidate they emit through the prime windows, where each saved-prime chunk still costs one gap of about one chunk's wall time. Inside the windows the p99 gap falls (1134 to 480 ms, 1258 to 572 ms) and the rate rises six to thirteen times. TPOT and decoder E2E move by at most 2.2%.

Bytes: each of the five requests produced one sha across all 12 boots, and the five shas are distinct. Both arms engaged the walker on every boot (`yields=170` each).

## Conditions

- One RTX PRO 6000 Blackwell Server Edition (97,887 MiB), driver 595.91.07, P0, power limit 600 W, AMD EPYC Turin guest (30 vCPU, 87 GiB). Confidential Computing ON: every H2D and D2H copy is encrypted, which is a measurement condition of these absolute numbers, not of the A/B.
- Hashes (`raw/pro6000-service-ab/inputs.sha256`): base binary `22b71b82...`, candidate `bfab2028...`, model `52c9cceb...`. Lease receipt `raw/pro6000-service-ab/lease/lease.json`: exit 0, not timed out, no lingering compute.
- Arms interleaved by boot: base/cand, cand/base, three times over.

## Raw

- `raw/pro6000-service-ab/`: the scored run. `out/receipt.json` (verdicts, summaries, declared bars), `out/rep*-{base,cand}-cell.json` (every request's events), `out/rep*-*-server.log`, `out/rep*-*-telemetry.csv`, `gate.log`, `conditions.txt`, `inputs.sha256`, lease record.
- `raw/pro6000-attempt1-cc-not-ready/`: the first launch. The confidential-computing GPU was not yet set ready (`nvidia-smi conf-compute -srs 1`), so the first base boot exited 1 during load (`[server] FATAL: worker init failed: Engine::new failed: DriverError(CUDA_ERROR_SYSTEM_NOT_READY, "system not yet initialized")`) and the gate refused. Kept as the failed attempt; no timing came from it.
- `raw/queue2.log`: the queue that ran the A/B after the #556 battery released the card.
