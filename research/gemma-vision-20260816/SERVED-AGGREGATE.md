# Gemma-4-31B SERVED aggregate — batched arm through the real HTTP surface (lane/gemma-batched, 2026-08-16)

The serve-route arc: worker wiring + serve-stream identity + served aggregate cells on
the Japan box NVFP4mix artifact at the 450W cap. Seam stays DEFAULT-OFF; no models.toml/
catalog/pricing/site contact (serving conversation is owner-gated).

## 1. Wiring (worker.rs, commit 1af8bfb59)

Routing predicate only. A decode-site subset `eager_decode` (= eager_only minus
`gemma4_batched_decode_model`: dense gemma4 + `MEMRA_GEMMA4_BATCH=1`) is consumed by
exactly the TWO decode scheduling sites; prime batching, concat/fanout prime, graph
promotion, checkpoint nomination, and monolithic prefill keep the full eager_only
exclusion (gemma4 still has no batched prime core / dc graph). `chunk_cap_for` pins
gemma4 chunks at 8 — the proven exactness tier; the env door may narrow, never widen —
plus a per-request B>8 Err backstop in the engine arm (never a panic; the 2026-08-07
worker-FATAL law). E4B and every other arch untouched.

## 2. Serve-stream identity gate — ALL GREEN (5090, Q4_0, default env)

`tools/serve-gemma4-batch-gate.sh`: 8 greedy prompts through `/v1/chat/completions`,
byte-compared per prompt across three served configs:

| comparison | verdict |
|---|---|
| seam-off c1 (eager reference) vs seam-on c1 (B=1 chunks through the batched arm) | byte-identical, 8/8 |
| seam-off c1 vs seam-on c4 (B=4 chunks — served batchmate isolation) | byte-identical, 8/8 |

Refuse-on-ambiguity discipline: the gate FAILS unless the seam-on boot prints the
"BATCHED DECODE (MEMRA_GEMMA4_BATCH=1…)" route notice AND the engine's
"[gemma4-batch] first B>1" marker appears during the concurrent phase — both confirmed,
so the batched path was demonstrably the one exercised. Run under the DEFAULT serving
env (eager side on its rows/rows_w fast arms, batched side on kvmod): completions agree
at the byte level, so the served identity holds in the shipping configuration.

## 3. Served aggregate cells — Japan box, NVFP4mix, 450W, GPU0

`tools/gemma4-served-agg-cell.sh`: 5 reps, INTERLEAVED seam-off/seam-on with alternating
boot order per rep (the A/B law — no all-A-then-all-B clock drift), c1/c4/c8/c16, 128
max_tokens, temp 0.7 divergent streams (the realistic serving mix), aggregate tok/s from
the server's own usage blocks. Artifact: `gemma-4-31B-it-NVFP4mix.gguf` (official-weights
convert). Receipts: `served-agg-points.jsonl` (this dir) +
`/data/memra/evidence/gemma-batched-20260816/` on the box (server logs per rep).

| c | seam OFF median (min–max, n) | seam ON median (min–max, n) | ratio | p50 latency ON / OFF |
|---|---|---|---|---|
| 1 | 55.2 (55.2–55.2, n=5) | 58.2 (58.2–58.3, n=5) | 1.06× | 1.15s / 1.23s |
| 4 | 55.3 (55.3–55.4, n=5) | **172.8** (172.8–173.0, n=5) | **3.12×** | 1.47s / 4.93s |
| 8 | 55.3 (55.3–55.3, n=5) | **245.6** (245.5–246.0, n=5) | **4.44×** | 2.00s / 9.79s |
| 16 | 55.3 (55.3–55.3, n=5) | **257.3** (257.3–257.8, n=4) | **4.65×** | 4.04s / 19.29s |

- **The seam-off side REPLICATES the flat-55 line exactly** (55.2–55.4 at every c, dead
  flat) — the baseline the perfection lane reported is confirmed live, not assumed.
- **Seam-on breaks it: 245.6 at c8 / 257.3 at c16 — ABOVE the Q38 board's 228–233
  reference band on the same card class.** The aggregate gate the lane sized as its top
  blocker is closed.
- Per-stream p50 latency also improves ~2–5× under load (round-robin serialization gone).
- One flagged point (`g4-on-c16-r2`): all 48 requests 503'd with "server draining" — a
  harness race at a boot boundary (stray SIGTERM), ZERO engine-side errors in any rep's
  server log; the surviving n=4 spread at c16-on is <0.5 tok/s. Not counted, disclosed.
- Prompt-side note: load-serve uses one fixed ~200-token prompt, so prefill rides the
  prefix cache and the cells isolate DECODE aggregate — the number the batched arm owns.
  Mixed-prompt prefill pressure is a separate serving dimension.

## 4. Spec battery after the changes — no regression

The dflash spec acceptance gate re-run on THIS lane's HEAD binary (Japan GPU0, Q4_0
trunk + dspark head + 447k own-gen ranks, fixed 19-token prompt, n=128):
acceptance **0.549 (78/142)**, plain 74.25 tok/s, spec **132.57 tok/s (1.79×)**,
**stream agreement 128/128** — byte-for-byte the pre-change acceptance receipts.
Spec and batched remain separate paths (spec sessions are excluded from decode chunks
by predicate); no composition attempted per the coordinator's scope.

## Verdict (pre-flip)

Identity: eager == batched at the served level, B=1 and B=4, byte-exact, ambiguity-
refused. Aggregate: flat 55 → 257 tok/s c16 (4.65×) on the product artifact at 450W.
Spec: untouched and green. Later perf increments (not correctness): per-session
rows/rows_w fast attention in the batched arm, exact16-tier width past 8.

## 5. DEFAULT FLIP (owner ruling, 2026-08-16, commit 194782617)

Owner, on the receipts above: "if the performance are so strong in favor so obviously
on by default... we serve the correctness and best performance." The arm is now
**DEFAULT ON**. `MEMRA_GEMMA4_BATCH` becomes an opt-OUT kill switch: unset/`1` =
batched (the shipping default), `0` = eager rollback, any other value REFUSES LOUD at
first use ("not a recognized value... refusing to guess a serving path" — panic
receipt verified live with `MEMRA_GEMMA4_BATCH=yes`). Chunk-cap-8 + B>8 backstop
unchanged; E4B and all other arches stay eager, untouched.

**Identity under the NEW default env (5090, Q4_0):** the gate re-run with boot A =
kill switch (eager reference) and boot B = env UNSET (shipping default) — 8 greedy
prompts byte-identical at c1 and c4, default-on boot notice + first-B>1 marker both
confirmed on the default-env side. ALL GREEN.

**Served confirmation cell (Japan GPU0, NVFP4mix, 450W, 3 interleaved reps,
alternating boot order, kill-switch vs env-unset):**

| c | kill switch (eager) | DEFAULT env (batched) | p50 latency default/eager |
|---|---|---|---|
| 8 | 55.3 (55.2–55.3, n=3) | **245.7** (245.6–245.8, n=3) | 2.00s / 9.80s |
| 16 | 55.3 (55.3–55.3, n=3) | **257.4** (257.4–257.9, n=3) | 4.04s / 19.28s |

Zero errors, zero shed, zero ambiguous boots (all 3 default-env reps show the notice
+ B>1 marker). The flip carries the ×5 opt-in numbers to within 0.1 tok/s
(`flip-confirm-points.jsonl`; box evidence `.../gemma-batched-20260816/flip/`).

**Spec battery on flip HEAD (default env):** acceptance 0.549 (78/142), plain 74.24,
spec 132.32 tok/s (1.78×), stream agreement **128/128** — the spec path is
undisturbed by the default flip (spec sessions remain excluded from decode chunks by
predicate).

Lane is merge-ready with default-on. Still no version bump, no models.toml/catalog/
pricing/site contact — the release train is owner-gated.
