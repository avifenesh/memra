# A sampled resume takes its cold twin's spec K (memra#1093 follow-up, 2026-11-09)

Verdict: K changes the output of a sampled request and not of a greedy one, and a sampled resume of
a long conversation ran another K than its cold twin, so it left the cold text. The fix ships.

- **The rung.** The spec K table sends a request with at least 1,024 prompt tokens and at least
  1,024 of them restored from a cache to the cached-long rung (K=2 on an untrimmed MTP head). Its
  cold twin takes the cold-long rung (K=3). Every resume of a long conversation crosses it: the
  prefix cache's spec restore and the spec pool's rewinds alike.
- **The output.** On the RTX 5090 at the device default, 4 of 4 sampled pair cells left the cold
  text on the base, and 0 of 4 do on the fix. Greedy is byte-equal on both.
- **The fix.** The cached-long rung now serves greedy requests only. A sampled request takes the
  rung its cold twin takes, whatever it restored.
- **The cost.** On the RTX 5090, a sampled resume of a long conversation decodes about 3% slower
  at K=3 than at K=2 (59.6 against 61.5 tok/s pooled over 24 turns per arm). Its TTFT is unchanged,
  and greedy is unchanged.

Model and rig: Qwen3.8-27B NVFP4 + Q5_K GGUF with its untrimmed MTP head, no DFlash drafter, RTX
5090 Laptop, context 8,192, two sessions, prefix cache at its derived size, every switch at the
device default. Binaries:
- base: the code of #1137's head 0ccc338948 (sha256 2c453c9995b3e501);
- fix: this branch (17a12eb72048c534). It was built before three changes that move no measured
  path: the unit test's import fix, the boot policy line's "greedy only" wording, and the restore
  load guard calling the table's own rule instead of a copy of it.

## Does K change the output? (`k-invariance/run2/`)

`k_invariance.py` serves the same four cold prompts at three settings, each on its own boot:
- the shipped cold-long K=3;
- K=2, through the `MEMRA_SPEC_K_COLD_LONG=2` measurement door;
- K=3 again, a determinism control.

Each prompt is about 1,400 tokens and asks for a 300-word report (400 tokens, every row hit the
cap). Every request's `[spec-k]` line records the K it ran.

| sampling | K=3 vs repeat K=3 | K=3 vs K=2 |
|---|---|---|
| greedy | 4 of 4 byte-equal | 4 of 4 byte-equal |
| T=1 (seed 1234) | 4 of 4 byte-equal | 0 of 4 byte-equal (2,316 vs 2,351, 2,409 vs 2,335, 2,324 vs 2,348, 2,355 vs 2,288 characters) |

A greedy verify accepts the target's argmax whatever the draft depth, so it writes the same tokens
at any K. That is the property the battery's `run-spec` K=1..8 gate pins, and it holds here on 400
tokens of four long prompts. A sampled verify draws its Philox events per round, so another K draws
another stream: the T=1 text is reproducible per K (the repeat matches) but not across K.

`k-invariance/run1-t1-restored/` is a first run whose T=1 prompts repeated the greedy ones. The
prefix cache restored them, so every arm ran the cached-long K=2 and its T=1 rows test nothing.
Its greedy rows agree with run 2's.

## Pair cells (`pairs-base/`, `pairs-fix/`)

`k_pair_cells.py` through `k_pairs.sh`, the spec-pool lane's method (one warm boot of three turns,
one cold boot per turn 2 and turn 3, every warm answer byte-equal to its cold twin). The first turn
is about 1,400 tokens (45 notes and a question), then two short follow-ups, so turn 2 restores
1,408 tokens and turn 3 restores 1,440 to 1,472. `raw-long2-nopc-t1` turns the prefix cache off and
sends a `user` id, so the spec pool's affinity tier rewinds instead.

| case | sampling | warm path | base bytes (t2/t3) | base K warm vs cold | fix bytes (t2/t3) | fix K warm vs cold |
|---|---|---|---|---|---|---|
| chat-long2-open | greedy | prefix restore 1408/1440 | PASS | 2 vs 3 / 2 vs 3 | PASS | 2 vs 3 / 2 vs 3 |
| chat-long2-open-t1 | T=1 | prefix restore 1408/1440 | FAIL (10 vs 12 / =) | 2 vs 3 / 2 vs 3 | PASS | 3 vs 3 / 3 vs 3 |
| chat-long2-open-t1-pen | T=1 pen | prefix restore 1408/1472 | FAIL (76 vs 184 / 93 vs 426) | 2 vs 3 / 2 vs 3 | PASS | 3 vs 3 / 3 vs 3 |
| raw-long2-open | greedy | prefix restore 1408/1440 | PASS | 2 vs 3 / 2 vs 3 | PASS | 2 vs 3 / 2 vs 3 |
| raw-long2-open-t1 | T=1 | prefix restore 1408/1440 | FAIL (10 vs 12 / =) | 2 vs 3 / 2 vs 3 | PASS | 3 vs 3 / 3 vs 3 |
| raw-long2-nopc-t1 | T=1 | affinity 1408/1440 | FAIL (10 vs 12 / =) | 2 vs 3 / 2 vs 3 | PASS | 3 vs 3 / 3 vs 3 |

(`N vs M` = warm and cold character counts of a turn that differs; `=` = byte-equal.) The greedy
cases keep the cached-long K=2 on both binaries by design and match their K=3 cold twins: the
invariance in a serving shape. The two greedy rows in `pairs-fix/` carry a `ksame` expectation they
were registered with before that design was settled, which is why their `path_ok` reads false.
`k_pairs.sh` now expects `kdiff` for them on both binaries.

## The change

`spec_k_cached_tokens` (worker.rs): the admission's K re-decision, the one site that reads the
restored length, passes it through only for a greedy request. A sampled request reads none, so the
table answers it as it answers the cold twin. The sampled-restore load guard
(`spec_cost_restore_verdict`) priced the cached-long rung; it now prices the rung the restored
request runs (cold-short or cold-long by prompt length, through `choose_spec_k_with`), and the boot
`[spec-k] automatic table` line says the cached-long rung is greedy only. Greedy requests, the
shape every recorded
K measurement used (`research/kpolicy-20260808/`, `research/orndecode-20260822/`), are unchanged.

## Cost (`cost/`)

`k_cost.py`: base against fix, boots in the order base fix fix base, twice. Every conversation
has a first turn of about 1,400 tokens, then two follow-ups, so turns 2 and 3 restore the prefix
cache's seed (1,216 to 1,632 tokens) and cross the rung. Each request is streamed: TTFT is the
first content chunk, and the decode rate is completion tokens over the time after it. A greedy
conversation runs on every boot as a control. Its path is the same code on both binaries, so the
gap between its arms is the boot-to-boot noise.

**Run 2** (`cost-run2/`, the decode-focused run; `K_COST_LONG=1`): three T=1 conversations and one
greedy per boot, answers capped at 400 tokens (most turns reach the cap).

| sampling | turn | K base / fix | n | TTFT base / fix (s) | decode base / fix (tok/s, median) | acceptance base / fix |
|---|---|---|---:|---|---|---|
| T=1 | 1 (cold) | 3 / 3 | 12 | 1.177 / 1.225 | 65.9 / 62.0 | 0.483 / 0.476 |
| T=1 | 2 | 2 / 3 | 12 | 0.524 / 0.542 | 60.4 / 58.3 | 0.558 / 0.428 |
| T=1 | 3 | 2 / 3 | 12 | 0.533 / 0.551 | 61.7 / 61.8 | 0.587 / 0.494 |
| greedy | 2 | 2 / 2 | 4 | 0.556 / 0.571 | 61.8 / 61.2 | 0.549 / 0.521 |
| greedy | 3 | 2 / 2 | 4 | 0.586 / 0.554 | 65.4 / 66.3 | 0.613 / 0.616 |

Pooled over the resumed turns (completion tokens over decode time):
- T=1: 61.54 tok/s on base against 59.62 on the fix (-3.1%, 9,560 and 9,331 tokens).
- The greedy control: 63.14 against 63.23 (+0.1%).

TTFT moves by +0.02 s on the resumed T=1 turns, as much as on the greedy control (+0.015 s on turn
2), so it is noise.

**Run 1** (`cost/`): two T=1 conversations and one greedy per boot, answers capped at 256 tokens,
many shorter. T=1 resumed pooled: 59.06 against 56.10 tok/s (-5.0%). Its greedy control spread
±10% between boots on identical code (turn 1 59.0 / 53.2, turn 3 55.3 / 60.6), so that run
resolves nothing finer.

So the price of a sampled resume that writes its cold twin's text is about 3% of decode rate on
these turns. The kpolicy matrix that set the rung (greedy, another rig) had K=2 ahead of K=3 by 1.5%
and 7.1% on cached-long; this card's sampled resumes agree in direction and size. Greedy, where K
does not change the output, keeps K=2.

## Left open

- A sampled prefix hit that the sampled-restore load guard refuses serves plain (K=0) at a demand
  where its cold twin may run spec. This is the guard's deliberate throughput trade, older than
  this change, and depends on load; the independent review found it. Single-stream cells never
  reach it.
- Whether K=2 would also serve cold sampled long prompts better (every sampled long request at
  K=2) is not measured. That would move a cold default, so it needs its own A/B, on the 5090 and
  a PRO 6000.
