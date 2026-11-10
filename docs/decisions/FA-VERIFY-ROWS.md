# A verify's rows read the KV once, routed by depth (2026-11-10)

**Status:** decided. The rows walk is the verify's attention launch from a measured depth floor per GQA shape, on every device, with no flag. Receipts: [`research/gb10-verify-rows-20261109/`](../../research/gb10-verify-rows-20261109/README.md).

## Question

A speculative verify attends T = K+1 rows against one session's cache. It used the batched tick's seqs kernel with one grid z per row, so a round read the whole KV T times. At 131k context on GB10 that was 99.65 ms of a 195.8 ms round for a read that the bandwidth prices at 17 ms (`research/gb10-decode-20261109/`). How should a verify's rows attend?

## Decision

- **`fa_verify_rows_v4_deep`:** one block per (KV head, split, row group) stages each K/V tile once, and every (row, head) warp runs the seqs walk over it with its own bound and writes its row's slot of the seqs partial layout. Per row it is the seqs program on the same values, and kernel-check `FA-VERIFY-ROWS` pins bit identity on the partials and the combined rows.
- **Routed by depth, from the launch's own shape (`fa_rows_floor`):** a row joins a group only if its key bound is at least 1,024 for GQA up to 4, 2,048 for GQA 5 and 6 (the 27B), 16,384 for GQA 7 and 8. Below the floor the row is a group of one, and a launch of one-row groups is the seqs launch itself. Because the two launches are bit-identical, the route chooses speed only.
- **Scope:** the eager qwen35 verify and the pooled verify. A captured verify graph keeps the seqs launch.

## What was measured

- Per attention call (T = 4), rows walk over seqs for the 27B's 24/4 shape: GB10 0.58 at 131k and 0.57 at 260k; RTX 5090 0.77 and 0.76. RTX PRO 6000 Max-Q 0.68 and 0.66.
- Served on GB10: 131k 192.3 to 153.5 ms a round (-20.2%), 260k 294.9 to 221.0 ms (-25.1%). RTX 5090, 28k: 43.3 to 40.2 ms (-7.1%). Every text identical, TTFT unchanged.
- Below the floors the walk is flat or slower on at least one of the two devices: a pooled cohort at depth 300 costs 1.16 to 1.35 times the seqs launch for GQA 6 and 8.
- With the route, a pooled cohort of short chats is at parity: GB10, the 27B, waves of 2, 4 and 8 requests, T/B 0.996, 0.999, 1.001 with overlapping ranges, every stream identical.

The full tables are in the research README.

## Rejected

- **The rows walk at every depth.** Slower per call on the shortest split rung (8 or 16 keys a split: a 768-thread block has almost nothing to share there), and the RTX 5090 pooled-cohort cell read 2.5 to 9% lower aggregate rates with it.
- **One floor for every shape (the 128-key rung).** Exact and never slower, but it gives up the GQA 4 shape's 16 to 60% between 1,024 and 16,384 keys and the 27B's 7 to 33% between 2,048 and 16,384.
- **A device-keyed floor.** GB10, the RTX 5090 and the RTX PRO 6000 Max-Q agree on which side of each floor wins, so one table serves all three.
- **2-row blocks.** Read the KV twice; slower than the full group from 32k on both devices (GB10 131k: 4.4 against 3.7 ms a call).
- **A launch bound on the kernel** (to pin its register budget for a 1,024-thread block): the compiler spilled and the call lost 5% on GB10. The host cuts a group to the build's own thread limit instead.
- **A device-resident group table for captured graphs.** A round's groups depend on its position; the captured verify keeps the seqs launch until a replay can rewrite the table.
- **An environment switch.** The two launches are bit-identical; a rollback seam would have nothing to roll back.
