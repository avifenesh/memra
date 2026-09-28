# DSv4-Flash lanes banked at the development stop (2026-09-28)

Owner order, 2026-09-28: memra development stops; the open work is banked in this one branch and
the servers are taken down. Nothing here is merged. Main at the stop is `c1a40c577` and carries
#906 (B-row diet), #909 (dense-fast to 16 rows), #881 (attention fusions) and #912 (the ceiling
record, `../ceiling/CEILING.md`).

## Code on this branch, in commit order

1. **Owned-row expert join** (`lane/dsv4-owned-join-20260928`, three commits). The TP/EP expert join
   pushes only each rank's own rows, and the receiver takes the clear's +0.0 for the rest. The
   replay census and the long gate's floor clone count the new kernel among the joins.
   Measured on the second SE pair against the B-row lane (`raw/se2-s2zm/`):
   - the long gate hash is `fbce1a0492d69635`, and the rows gate, wide 16 and the KV split gate
     pass;
   - the long gate is flat, 9.85 .. 10.02 against 9.89 .. 10.03 ms per token;
   - the 16-row captured step is 48.10 .. 48.24 ms against 48.5 .. 48.8, about -0.8%;
   - served c24 (B O O B): c8 292.2 / 292.7 against 288.2 / 290.3 (+1.1%), c24 276.8 / 274.2
     against 269.8 / 267.9 (+2.0%), c16 flat, c4 flat, all 80 texts identical;
   - the cells-pdl rows were not run.

   Small and consistent at c8 and c24, and exact by construction. A merge would still want
   cells-pdl and a rebase on main.
2. **One-row dense-fast with four loads in flight** (`lane/dsv4-dense-mlp4-20260928`). The M = 1
   dense-fast FP8 body issues four iterations' weight, scale and activation loads per leaf before
   its first add, in the same add order. It compiles to 40 registers with no spills. The campaign
   was stopped while building (`raw/se-v7e-v7f/q-v7f.summary`): **not measured, not gated**.
3. **One-row replay compressor through the multi-row launches** (`lane/dsv4-cmp-onerow-20260928`,
   WIP). The replayed one-row compressor would run #906's append, pool and finish launches with one
   row, removing about 200 launches per step at c1. `cargo check` passes only: **never built for
   GPU, never gated**.

## Refuted, kept as patches only (`refuted-patches/`, not applied)

- `dense16-rows4.patch`: four output rows per dense-fast block past 8 token rows. The 16-row
  captured step took 40.62 .. 40.72 ms against 37.43 .. 37.57, +8.4% (R M R, `raw/se-v7e-v7f/`).
  One 512-thread block per SM at 74 registers loses more occupancy than the halved activation
  reads save.
- `indexer-slots.patch`: eight replayed indexer scores per CTA. +3.8% on the long gate, c1 -3.6%
  served (`../levers-20260927/RESULTS.md`).
- `argmax-1024.patch`: a 1024-thread greedy argmax. 18.4 to 10.3 us per row, flat on the long gate
  (`../levers-20260927/RESULTS.md`).

## Not on this branch

`lane/dsv4-vt-default-20260924` (#709, the per-slot DSpark window default) waits on the owner as
before and is left as its own branch.
