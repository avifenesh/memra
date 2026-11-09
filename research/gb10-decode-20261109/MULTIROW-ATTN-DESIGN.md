# Lever 2 design: multi-row verify attention (design, not built)

## The gap

A speculative verify runs T = K+1 query rows against one session's cache. Today the rows go through `fa_decode_vec_q_seqs_v4_deep`, the batched-tick kernel, with grid z = row. Every z walks the whole cache, so a round reads the KV T times. At 131k with K=3 that is 15.6 GB per round for 3.89 GB of unique KV: 90.7 ms of kernel time plus 8.9 ms of combine, 99.65 ms of a 195.8 ms round (`scout/nsys131k-unit2/kernels.txt`). The kernel streams its own bytes at 172 GB/s, about 76% of the 226 GB/s ceiling. The waste is the re-reads, not the streaming.

## The kernel

`fa_verify_rows_v4_deep`: one block per (kv_head, split, partition class) runs every row of the class.

- Block = R rows x gqa warps (R <= 4 at K=3; the block size is R x 6 x 32 = 768 threads at R=4). Warp (r, wy) owns query row r, head kv_head x gqa + wy, and its own m_i, l_i, acc registers.
- Q staging: `q_ints` / `q_d` grow from [gqa] to [R x gqa] (12,160 B of struct becomes about 19 KB at R=4). Each warp stages its own row and head with `fa_v4_deep_stage_q`, verbatim.
- The walk: `fa_v4_deep_walk`'s tile loop over [t_lo, t_hi_max) in FA_DEC_TILE steps. All threads stage V and K for the tile once (the same staging code; nt = min(TILE, t_hi_max - t0)), then each warp runs the score, B2 and B3 phases with its own nt_r = min(TILE, t_hi_r - t0). A warp whose t_hi_r <= t0 skips the three phases (warp-uniform) but keeps the block barriers.
- Partials: written at the row's own (z = row, head, split) slot of today's [T, n_head, n_splits_max, head_dim] layout, so `fa_decode_combine_seqs` runs unchanged.
- L2 prefetch: `fa_deep_prefetch_tile` against t_hi_max (no numeric effect).

## Why each row stays bit-identical

The one-partition law sets each row's split from its own T_kv: ns_eff = ceil(T_kv / split_keys), per = ceil(T_kv / ns_eff), t_lo = split x per, t_hi = min(T_kv, t_lo + per). Rows of a verify round have T_kv = pos0 + 1 .. pos0 + T, so they usually share ns_eff and per, and with them every t_lo. They differ only in the last split's t_hi, by at most T - 1 keys. Rows that share (ns_eff, per) form one partition class. A round that straddles a change in per or ns_eff (per changes when T_kv crosses a multiple of ns_eff, ns_eff when it crosses a multiple of split_keys) splits its rows into two classes. Each class gets its own grid z, so a row never walks another row's partition.

Within a class, row r's partial (acc, m_i, l_i) for split s comes from the same instruction sequence on the same values as block (z = r, x = kv_head, y = s) of the seqs kernel:

- the same tiles in the same order, because t_lo is shared and tiles step by FA_DEC_TILE from it;
- the same staged values: V is dequantized and K repacked by the same code, and keys past the row's bound that another row needs are staged but never read by this row (every read is guarded by lane < nt_r or j < nt_r, as today's walk guards by nt);
- the same score dp4a chain, the same shuffle max, the same alpha and p, and the same B3 accumulation order;
- an empty split (s >= ns_eff, or a split with t_hi = t_lo) writes the same empty partial (acc 0, m -inf, l 0).

The combine then reads the same partials in the same ascending-split order. Each row's attention output is therefore bit-identical to today's, and so is everything downstream: verify logits, acceptance and tokens.

## Expected gain

Bytes per round at 131k: KV 3.89 GB once, instead of 15.6 GB, plus today's partial traffic (1,024 splits x 24 heads x 256 fp32 per row: 1.6 GB written, 1.6 GB read by the combine). The floor is about 31 ms against 99.65 today. The compute does not shrink: the same per-row work runs, now four rows per block. An estimate of that work is about 5.5 G warp-instructions per round, 12-24 ms at 50-100% issue on 48 SMs, under the bandwidth floor. Expected: -55 to -70 ms per round at 131k (195.8 to about 130-140 ms, 13.5 to 18.5-20 tok/s in bursts), and -110 to -135 ms at 260k (294 to about 160-185, 9.4 to 15-17). Nothing at 1k (0.82 ms of attention there).

Risks: one block per SM at 768 threads, so latency hiding comes from 24 warps instead of three 6-warp blocks. A 2-row block (384 threads, two blocks per SM, the KV read twice) is the fallback shape, measured in the same probe.

## Gates

- kernel-check `FA-VERIFY-ROWS`: rows-vs-seqs bit identity on the partials and the combined output, for T = 1..9 (K up to 8), on the default KV module, at T_kv:
  - inside one tile (T_kv < 32);
  - with t_hi landing exactly on a tile boundary for some rows of the class;
  - at a change in per (pos0 + T crossing a multiple of ns_eff);
  - at a change in ns_eff (crossing a multiple of split_keys, on the 64 and 128 rungs);
  - at 4k, 64k and 131k.
  Fills of 0x5a and 0xa5 in the partial buffers so an unwritten slot shows.
- `run-spec` K=1..8 self-consistency and `spec-serve-gate` on GB10 and the 5090.
- Served B/T (B T T B B T) with identical text sha at 1k and 131k on both units, signal ms per round inside bursts. The 5090 confirms the generic path.
- Pooled verify (spec pool, several sessions): rows of one session form classes, and sessions stay separate z. Covered by the same kernel-check cells with two sessions.

## Order of work

Probe kernel and kernel-check cells first (bit identity and the per-call time at 131k against the seqs kernel), then the dispatch in the qwen35 verify's fa-rows arm (`dspark_fa_rows_on`), then the served cells.
