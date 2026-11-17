# ring-cap-20261114: no pipeline prime call wider than the chunk a ring plan holds, on Step-3.7-Flash PP-2 (memra#1299)

Verdict: PASS on 2x RTX PRO 6000 Blackwell Max-Q (PP-2, the IQ4_XS GGUF route). A 32,768-token
continuation on the prime walk fails on main with `SWA ring append does not fit` at its second
call (4,809 rows) and primes to its last row with the change, in 14 calls of at most 4,026 rows.

## What changes

PP-2's dynamic prime schedule makes its first call half a chunk and spreads the rest at equal
modeled work, so the calls after it outgrow the chunk: 4,097 rows from a 27,916-token prime, and
4,809 at 32,768. A ring plan (step35) holds the reader's window plus one prime call of at most
`PRIME_CHUNK_MAX_TOKENS` rows (`swa_ring_rows`: 5,151 rows at a 512-row window), and a wider call
fails its append. On PP-2 step35 takes this schedule whenever the batched GEMM entry declines, so
a continuation of 27,916 tokens or more with `MEMRA_STEP_GEMM_PRIME_SUFFIX=0` could not prime.

With a ring plan, `prime_chunk_ranges_under` splits every pipeline range wider than the cap into
parts within it. Without a ring the schedule is unchanged.

memra#1275 (landed before this) changed the other step35 schedule, the batched GEMM entry's
(`step_gemm_prime_ranges`); it does not reach the walk. The base arm below is the main that has it.

## The check

`tools/ring-check.sh`, with the PASS rule written in its header before the run and applied by
`tools/ring-compare.py`. Per arm, one process on the pair
(`MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1 MEMRA_PARALLEL=off MEMRA_STEP_GEMM_PRIME_SUFFIX=0`):

`concat-prime-probe <gguf> tickinv --prompt-a @ring32k.txt --budgets 0 --splits 64 --steps 0`

- `ref`: one prime of all 32,832 tokens at position 0 (the batched GEMM entry);
- `sp64`: a 64-token prime at position 0, then the 32,768-token continuation at position 64. With
  the suffix door off the continuation takes the walk on the PP-2 dynamic schedule.

The prompt is a prefix of `research/e2e/prompts/p6-64k.txt` that tokenizes to 32,832 tokens with
BOS (`tools/mkring.py`). No decode steps: the cell is about the prime's calls.

The base arm runs first, alone, under its own limit. It must reach the continuation and fail there
with the ring error; if it ends any other way the cell stops and the head arm is not run.

PASS:

- base: the probe exits non-zero, its log has `SWA ring append does not fit`, and the
  continuation's `[prime-walk]` line lists a call wider than 4,096 rows;
- head: the probe exits 0, the continuation's calls are each within 4,096 rows and at least 16,
  they sum to the continuation's length, and the `sp64` row is printed;
- the continuation is at least 27,916 tokens.

Reported, not judged: `sp64` against `ref` on the head. They are two prime programs (the batched
GEMM prime and the walk).

## Result (`pro6000-pp2/ring-run-002050`, 2026-11-18)

Binaries: base = main 5a6ae2438e, head = 3a766d2dff (this change rebased onto it). Both carry the
probe fix of this change (below) and two print-only patches, so that the receipt shows every
call's width (`tools/print-patches-*.diff`: the GEMM prime's `ENGAGED` line and the prime walk
list their calls). Their sha256 are in `guard.log`.

| arm | exit | wall | continuation's calls (rows) | widest | ends |
|---|---|---|---|---|---|
| base | 1 | 122 s | 2,048, 4,809, 4,646, 4,499, 4,366, 4,243, 4,131, 4,026 | 4,809 | at its second call: `SWA ring append does not fit (keep 512 + append 4809 > 5151) [append len=2112 retain_from=1600 append_rows=4809 ...]` |
| head | 0 | 167 s | 2,048, 2,404, 2,405, 2,323, 2,323, 2,249, 2,250, 2,183, 2,183, 2,121, 2,122, 2,065, 2,066, 4,026 | 4,026 | the `sp64` row |

- Both arms prime `ref` first through the GEMM entry in calls of 8 x 4,096 + 64 rows.
- Reported: on the head, `sp64` reads `EXACT` against `ref` (maxdiff 0, `identical`), and the
  probe's own verdict line is `TICK-INVARIANT`. One prompt; no claim is made from it.
- `verdict.txt` ends `VERDICT PASS`.
- Single run per arm. Another job took a third card of the box while the head arm ran; no
  timing is read from these logs.

## Earlier run (`pro6000-pp2/ring-run-085555`, 2026-11-14): void

On the pair of that day (base 63c351a2f4, head 1ea13cce9e). The probe's tick arm built its cache
with `Cache::new`, which puts every layer's KV on the first device, and the second stage refused
it in both arms before the continuation: `PP cross-device tensor access at
append_kv_quantized_rows.k_cache: tensor on dev0, stream on dev1`. Nothing about this change was
measured. The change's third commit fixes the probe (the arm takes its cache from `pp::new_cache`,
as `run-gen` and `run-spec` do, and reads the hidden rows on the stream that owns them), and the
cell got its base-first order from this run. Kept as it ran; its `verdict.txt` reads `FAIL`.

## Not measured

- A continuation between 27,916 and 32,768 tokens, or past it: one length ran. The schedule's
  bounds at every length are a CPU test (`no_ring_prime_call_exceeds_the_chunk_cap`).
- The serving path: the cell is the engine probe. With the suffix door at its default a
  continuation takes the batched GEMM entry and never this schedule.
- A ring plan with a GDN grid: none exists (step35 has no GDN layers).
- Off the pair: the plans that fit one card have a GatedDeltaNet mixer, which pipeline placement
  refuses at load, so the probe's two-device arm cannot run anywhere else.

## Files

- `pro6000-pp2/ring-run-*/`: `guard.log` (binaries, prompt, cards), `base.log`, `head.log`,
  `base-verdict.txt`, `verdict.txt`. The box's name, account, data mount and card UUIDs read as
  neutral names.
- `tools/`: `ring-check.sh`, `ring-compare.py`, `mkring.py`, the two print-only patch scripts and
  their diffs for this pair.
