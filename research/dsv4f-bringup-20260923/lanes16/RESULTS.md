# DSv4 TP/EP serving lanes to 16, and a coalescer that keeps them in one batch (memra #667)

Scope: one model, one hardware shape.
- Model: `tiyuvta/DeepSeek-V4-Flash-0731-NVFP4@bafd09f8cab4f4f4f25e1cdafbcdefc05b90ee38`.
- Hardware: the first 2x RTX PRO 6000 Blackwell Server Edition pair, 2026-09-27.
- Program: the TP/EP served default.
- Lane: `lane/dsv4-lanes8-20260927`.
- Raw: `raw/` (queue scripts, summaries, every gate log and served row).

## What changed

1. **Wider range.** `MEMRA_DSV4_SESSIONS` and `MEMRA_DSV4_ROWS` take up to 16. They took 4 and 8.
2. **Hoist buffer.** A B-row workspace wider than 8 rows kept a one-element hoisted compressor
   buffer, but its 2- to 8-row batches still took the hoisted path. The first 16-lane boots
   failed every c4 and c8 request on `hoisted compressor rows: CUDA_ERROR_INVALID_VALUE`
   (`raw/lanes16-v6n/`). The buffer now covers eight rows at any workspace width.
   - The rows gate's wide phase missed it because it only ran full-width batches. It now runs
     batches of N, N/2 and 3 rows in the N-row workspace.
3. **One-workspace coalescer.** Under the old coalescer, a row that deposited while a batch ran
   led the next partial batch as soon as enough rows arrived. The lanes then split into phases
   that each run a partial batch, and they never merged again.
   - The unit cell `one_workspace_keeps_jittered_lanes_in_full_batches` shows it: 16 lanes, a
     20 ms step and up to 1.5 ms of host work per lane per step. The old policy ran 43 batches,
     1 of them full. The new one runs 12, all full.
   - With one workspace (TP/EP), a row now waits for the batch in flight to publish.
   - The window, 500 us or a tenth of the last step whichever is longer, runs from that
     publication.
4. **Default.** The TP/EP plain route takes 16 lanes and 16 rows by default. The drafter route
   keeps one lane.

## Correctness

- `dsv4_rows_gate` on TP/EP passes with `DSV4_ROWS_GATE_WIDE=8` and `=16` (`raw/lanes16-v6n/`,
  `raw/lanes16-fix-v6s/`). Every row's logits bits equal its solo steps, eager and captured, at
  widths 16, 8 and 3 in the 16-row workspace. The four-session phases (join, leave, row moves,
  replay, sampled graph) pass too.
- Timing from the same gate: the captured B-row step is 36.7 ms at 8 rows (217.9 tok/s) and
  70.4 ms at 16 (227.3 tok/s).
- Served: every request that both arms served has the identical text, sampled cells included.

## Served (`raw/coalesce-v6t/`)

Arms, one boot per row, order O N W W N O, cells `raw/cells-c24.txt`, 256 generated tokens:
- **O:** the build before the coalescer fix, 4 lanes.
- **N:** the fix, 4 lanes.
- **W:** the fix, 16 lanes and rows.

Aggregate tok/s, and served/total where requests were refused:

| cell | O | N | W |
|---|---|---|---|
| greedy c4 | 151.94 / 152.03 | 154.26 / 152.53 | 153.36 / 153.97 |
| greedy c8 | 154.53 / 157.02 | 154.70 / 159.91 | **190.23 / 192.01** |
| greedy c16 | 154.37 / **102.83** | 152.76 / 155.36 | **196.03 / 198.36** |
| greedy c24 | **102.77**, 20/24 / 153.37, 20/24 | 154.12, 20/24 / 155.78, 20/24 | **192.79 / 194.72**, 24/24 |
| sampled c8 | 153.50 / 153.50 | 151.85 / 153.73 | 183.51 / 186.69 |

TTFT p50 at c16:
- O and N: 10.3 .. 15.5 s;
- W: 1.49 .. 1.56 s.

At c24, W serves every request with TTFT p50 2.0 .. 2.2 s. Four lanes refuse four requests (429).

**The bimodal cells.** O's two slow cells (c16 in r6, c24 in r1) are the pattern the earlier
rounds called box noise: TPOT 37 ms against 24. Examples: this round's first 16-lane boot at
c16 (125 and 61 tok/s), the shared-expert lane's one slow c4 boot, and main's slow sampled c8.
N and W show none of them in eight boots. They were lane-phase splits in the coalescer.

**Earlier rounds.** `raw/lanes16-v6n/` (before the fixes) and `raw/lanes16-fix-v6s/` (hoist fix,
old coalescer) show the same c8 gain at 8 and 16 lanes, and slow cells scattered across arms.

## The naked default, on main with the shared-expert owner (`raw/lanes16-clear-v6v/`)

The lane's head, rebased on main `286c0c54c`, boots `16 serving lane(s) (default on the plain
TP/EP program ...)` and `B-row steps up to 16 rows` with no environment set.
- One more change: the fused step clears only its own rows of the contribution plane. A 16-row
  workspace cleared 16 rows' planes per layer before it.
- The long gate hash is `fbce1a0492d69635`.
- The TP/EP rows gate with the 16-row wide phase passes.
- Served, cells-pdl, one boot per row, order M L L M:

| cell | main (4 lanes) | lane (16 lanes) |
|---|---|---|
| greedy c1 | 95.11 / 95.18 | 94.90 / 94.97 |
| sampled c1 | 96.05 / 96.07 | 95.81 / 95.80 |
| greedy c2 | 128.15 / 128.47 | 127.57 / 127.63 |
| greedy c4 | 162.22 / 162.65 | 162.93 / 163.70 |

c1 and c2 read 0.2% and 0.5% lower in both lane rows. The earlier build without the sized clear
read c2 1.0% lower (`raw/lanes16-default-v6u/`). One naked cells-c24 row on the lane:

| cell | agg tok/s | served | TTFT p50 |
|---|---|---|---|
| greedy c4 | 158.57 | 8/8 | 335 ms |
| greedy c8 | 200.16 | 16/16 | 620 ms |
| greedy c16 | 201.37 | 16/16 | 1.39 s |
| greedy c24 | 200.71 | 24/24 | 1.75 s |
| sampled c8 | 193.37 | 16/16 | 598 ms |

Verdict: 16 lanes with the one-workspace coalescer are the TP/EP default. c8 +22%, c16 +27%, c24
every request at +25% or more, and c4 and c1 unchanged.
