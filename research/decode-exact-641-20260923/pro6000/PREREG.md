# memra#641 on the target card: pre-registration (lane E part 2)

Written before any run on the box. Nothing below moves after a result.

## Rig and trees
- One RTX PRO 6000 Blackwell (the lane-B spill box named in the gitignored `LANE-LOCAL.md`), every
  GPU cell under the canonical `/tmp/memra-gpu.lock`, taken through `tools/tier-battery.py
  --external-lock` (the collector) only after lane B confirms its chain is held, never inside one
  of B's orders. Idle card before each window: `nvidia-smi --query-compute-apps` empty.
- Model: the box's 27B, `Qwen3.8-27B-NVFP4-Q5K-mtp.gguf` (the file lane B's day-29 gates use);
  its sha256 is recorded in the run log before the first cell.
- Fix tree: the lane tip after the merge with main `d544c6b82` (the SHA is the first line of
  `run.log`). Base tree for the cost A/B: main `d544c6b82` itself, the same code with the fresh
  varlen FA arm live. Both built on the box in their own checkouts; binary sha256s recorded.

## Correctness cells (fix tree)
1. `tools/prime-batch-exact-gate.sh <27B>` (b3-p24, b4-p1100, carried b3-p600, all `--exact`)
   and `--canary`. PASS and CANARY OK required.
2. `tools/prime-tick-exact-gate.sh <27B>` and `--canary`, if the 27B loads through
   `HybridModel` (the qwen hybrid family `concat-prime-probe` targets). If it does not apply, the
   run log says so with the loader's own error line, and cell 1 is the nearest exact cell.
3. `tools/spec-ctx-edge-gate.sh <27B> <fix memra-server>` with `SCE_CTX=384` (the prompt is about
   31 tokens on the qwen template, so the door-OFF runaway has about 350 rows of room before the cap
   and the count-to-2000 prompt cannot stop short on EOS). If the 27B's prompt is longer than 128
   tokens, `SCE_CTX` becomes `prompt + 384`, recorded before the run. Acceptance: `SPEC-CTX-EDGE
   GATE: ALL GREEN`.
4. For the base tree, cell 1's non-canary rows once, as the red/green pair on this card (expected
   red, as on the 5090; a green base on this card is reported as such).

## Cost A/B: batched prime wall, base vs fix (`executed-not-qualified`)
- `prime-batch-gate <27B> --batch 3 --bench 1024`: three fresh 1024-token prompts, the #641
  tick shape; each process prints `bench B=3 T=1024 N=5 alternating: ... batch_wall_ms=X`, its
  own median of 5 alternating serial/batch reps.
- 6 pairs, one process per run, orders alternating AB, BA, AB, BA, AB, BA (A = base), so N=6 per
  arm, both orders three times each. Same env for both arms (`MEMRA_REWRITE_BUNDLE` unset).
- Telemetry: `nvidia-smi` at 250 ms for the whole window (SM clock, temperature, power, util),
  kept as CSV beside the raw logs. Each run's start line records temperature and SM clock.
- Reported: per-arm median, min and max of `batch_wall_ms` (N=6), the per-pair fix minus base
  and its median, and each arm's batched-vs-serial median.
- Reading, fixed now: the direction is a cost if the paired median is above +1% and at least 5
  of 6 pairs are positive; a gain by the mirror rule; flat otherwise. The fix stays in every case:
  one numeric program per request. A cost is the size a varlen twin over the dequantized view
  would have to recover, and that twin needs its own bit-identity receipt against the solo prime.

## Signal
When the box part is done, the line `LANE-E-PRO-DONE` is appended to `pro6000/run.log` (lane B's
signal), and the box scratch this lane made is removed.
