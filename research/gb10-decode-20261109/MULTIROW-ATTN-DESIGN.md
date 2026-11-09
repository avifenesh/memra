# Lever 2 design: multi-row verify attention (design, not built)

## The gap

A speculative verify runs T = K+1 query rows against one session's cache. Today the rows go through `fa_decode_vec_q_seqs_v4_deep`, the batched-tick kernel, with grid z = row: the verify's fa-rows arm (`dspark_fa_rows_on`, spec.rs) builds a pointer table whose T entries all name the same K/V cache and a `pos_seq` of `len0 .. len0 + T - 1`. Every z walks the whole cache, so a round reads the KV T times. At 131k with K=3 that is 15.6 GB per round for 3.89 GB of unique KV: 90.7 ms of kernel time plus 8.9 ms of combine, 99.65 ms of a 195.8 ms round (`scout/nsys131k-unit2/kernels.txt`). The kernel streams its own bytes at 172 GB/s, about 76% of the 226 GB/s ceiling. The waste is the re-reads, not the streaming.

## The kernel

`fa_verify_rows_v4_deep`: one block per (kv_head, split, row group) runs every row of the group.

- **Block.** R rows x gqa warps. Warp (r, wy) owns query row r and head kv_head x gqa + wy, with its own `m_i`, `l_i` and `acc` registers. A block holds at most 32 warps, so R <= floor(32 / gqa): 5 rows at gqa 6 (Qwen3.8-27B, 24 heads over 4 KV heads). K=3 runs R=4 in a 768-thread block. A verify with more rows than that (K >= 5) splits into row groups, each its own grid z, and reads the KV once per group.
- **Q staging.** `q_ints` / `q_d` grow from [gqa] to [R x gqa] rows. Each warp stages its own row and head with `fa_v4_deep_stage_q`, verbatim, into slot r x gqa + wy.
- **The walk.** `fa_v4_deep_walk`'s tile loop over [t_lo, t_hi_max) in FA_DEC_TILE steps, where t_hi_max is the group's largest bound.
  - Staging: every thread of the block stages V and K for the tile once, with the walk's own staging code and nt = min(TILE, t_hi_max - t0). The staging loops spread their work by thread index (`bt`, `bsz`); with `bt = (r x gqa + wy) x 32 + lane` and `bsz = 32 x R x gqa` each slot is still written once, by the same expression on the same bytes. Which thread stores a slot has no numeric effect.
  - Score, B2, B3: each warp runs the three phases with its own nt_r = min(TILE, t_hi_r - t0), reading its own Q slot. A warp whose t_hi_r <= t0 skips the three phases (the condition is warp-uniform) and still meets both block barriers.
- **Partials.** Written at the row's own (z = row, head, split) slot of today's [T, n_head, n_splits_max, head_dim] layout, including the empty partial for a split at or past the row's ns_eff. `fa_decode_combine_seqs` runs unchanged on the same `pos_seq`.
- **L2 prefetch.** `fa_deep_prefetch_tile` against t_hi_max. It has no numeric effect.
- **Shared memory.** 12,160 B today (Q 2,304 B, K tile 9,856 B) plus the 16,384 B bf16 V tile. The Q part grows to 32 warps x 288 B = 9,216 B, so a block takes about 35.5 KB.

## The host side

The host knows len0, T and the split ladder, so it forms the groups before the launch:

1. Row r has T_kv = len0 + r + 1, ns_eff = ceil(T_kv / split_keys) and per = ceil(T_kv / ns_eff).
2. Consecutive rows with equal (ns_eff, per) form a partition class. A class longer than floor(32 / gqa) is cut into row groups of that size.
3. The launch gets one small device table: per group, its first row and row count. Grid = (n_head_kv, n_splits_max, groups).

The fa-rows arm already requires one `fa_split_keys` rung across the round (the straddle law), so split_keys is one value per launch. The append of all T rows before the attend stays as it is. The seqs kernel stays for the batched tick, where z is a different session's cache.

## Why each row stays bit-identical

The one-partition law sets each row's split from its own T_kv: t_lo = split x per, t_hi = min(T_kv, t_lo + per), empty when split >= ns_eff. Rows of a verify round have consecutive T_kv, so they usually share ns_eff and per, and with them every t_lo. Inside a class they differ only in the last split's t_hi, by at most T - 1 keys. A round that straddles a change in per or ns_eff (per changes when T_kv crosses a multiple of ns_eff, ns_eff when it crosses a multiple of split_keys) has two classes. At 131k on the 128-key rung per changes once in about 1,024 positions, so about 1 round in 340 has two classes at K=3 (three row steps in 1,024 positions). Each class is its own group, so a row never walks another row's partition.

Within a group, row r's partial (acc, m_i, l_i) for split s comes from the same instruction sequence on the same values as block (z = r, x = kv_head, y = s) of the seqs kernel:

- the same tiles in the same order, because t_lo is shared and tiles step by FA_DEC_TILE from it;
- the same staged values: V is dequantized and K repacked by the same code from the same bytes. Keys past the row's bound that another row needs are staged but never read by this row, because every read is guarded by lane < nt_r or j < nt_r, as today's walk guards by nt;
- the same score dp4a chain, the same shuffle max, the same alpha and p, and the same B3 accumulation order, all inside one warp, which the block's other warps cannot reach;
- a tile the row has no key in is skipped whole, as today's loop never enters it;
- an empty split writes the same empty partial (acc 0, m -inf, l 0).

The combine then reads the same partials in the same ascending-split order. Each row's attention output is therefore bit-identical to today's, and so is everything downstream: verify logits, acceptance and tokens.

## Expected gain

Bytes per round at 131k: KV 3.89 GB once instead of 15.6 GB, plus today's partial traffic (1,024 splits x 24 heads x 256 fp32 per row: 1.6 GB written, 1.6 GB read by the combine). The floor is about 31 ms against 99.65 today.

Compute: the staging (the q5_1 dequant of the V tile and the K repack) runs once per tile instead of once per row. The per-row work (score, B2, B3) is unchanged, now four rows per block. That work is about 5.5 G warp-instructions per round, 12-24 ms at 50-100% issue on 48 SMs, under the bandwidth floor.

Expected: -55 to -70 ms per round at 131k (195.8 to about 130-140 ms, 13.5 to 18.5-20 tok/s in bursts), and -110 to -135 ms at 260k (294 to about 160-185, 9.4 to 15-17). Nothing at 1k (0.82 ms of attention there).

Risks, each answered by the probe:

- Occupancy. A 768-thread block with 35.5 KB of shared memory leaves room for two blocks per SM only if the register count per thread allows it. Latency hiding then comes from 24 warps that share one staged tile, instead of three independent 6-warp blocks.
- The barrier couples the rows: every warp waits for the slowest stage. A 2-row block (384 threads, the KV read twice) is the fallback shape, measured in the same probe.
- The RTX 5090 has four times the bandwidth, so the same kernel may gain little there or lose. The arm is then keyed on the device family, as the repo's per-hardware rule says. Detection decides, not a flag.

## Gates

- kernel-check `FA-VERIFY-ROWS`: the new kernel against today's seqs launch on the same Q, K/V bytes and `pos_seq`. Bit identity (bitdiff == 0) on the whole partO / partM / partL buffers and on the combined output, for T = 1..9 (K up to 8), on the default KV module, at T_kv:
  - inside one tile (T_kv < 32);
  - with t_hi landing exactly on a tile boundary for some rows of the class, and one key past it;
  - at a change in per (len0 + T crossing a multiple of ns_eff), so the round has two classes;
  - at a change in ns_eff (crossing a multiple of split_keys, on the 64 and 128 rungs);
  - with a row group cut by the 32-warp limit (T = 6..9 at gqa 6);
  - at 4k, 64k and 131k.
  Both launches start from partial buffers filled with 0x5a, then again with 0xa5, so an unwritten slot shows as a difference between the two fills.
- The same cell with two sessions' caches in one launch (the pooled verify): rows of one session form groups, and sessions never share a block.
- The probe's timing cell: per-call time of both kernels at 131k and 260k on GB10 and at 32k on the 5090, 4-row and 2-row blocks.
- `run-spec` K=1..8 self-consistency and `spec-serve-gate` on GB10 and the 5090.
- Served B/T (B T T B B T) with identical text sha at 1k and 131k on both units, signal ms per round inside bursts. The 5090 confirms the generic path.

## Order of work

1. The probe kernel and its kernel-check cells: bit identity first, then the per-call time at 131k against the seqs kernel. A probe that is not faster ends the lever with a ledger row and no dispatch change.
2. The dispatch in the verify's fa-rows arm (single session, then the pooled verify), the `docs/KERNELS.md` row, and the graph-capture rule. A captured verify cannot bake the groups, because a round's classes depend on its position: either the group table is device-resident and rewritten per round, as `pos_seq` is (the grid baked at its largest group count, a group with no rows returning at entry), or the captured verify keeps the seqs launch. The probe's numbers decide which.
3. The served cells and the device records. The seqs launch leaves the verify path in the same change; no door is kept.
