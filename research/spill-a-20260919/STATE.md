# WP-A resumable state (2026-09-26, stopped at an integrable milestone: DAY66's fix for integ65)

- Lane `lane/spill-a-20260919`; worktree `wt-spill-a`. integ65 takes `071e1126a` (F2b included), then DAY66.
- Items 21 to 25 closed (DAY53 to DAY58); item 23's addendum F2b (`b4d6f95c2`) accepted.
- Item 10 (the fanout publisher, DAY59). The target twin selected **DESIGN B1** (calls 0.83 / 0.82 of 1.34 ms, DAY59
  section 6). Revuto's split defect on #731 did not move it: all 100 split lines carry the clean counts (section 8).
  The 5090 re-run was NOT RUN, because a foreign app held the card (section 5).
  - B1 is pre-registered (DAY59 section 7). **Its code is in progress, uncommitted in this worktree**: the kernel
    `copy_batch_items_u8`, `Engine::copy_batch_items_u8`, `prefix_snapshot_batch` / `prefix_restore_batch`, the census
    rewrite, cells (a1) and (a2), and the KERNELS.md row. The last build stopped on test-code errors in (a2): the
    `Cache` and fn imports via `super::`, and `recur_fill`'s element type (`Vec<f32>`, not a slice).
  - Next: CPU-green, then the sitting `pro-single-b1/` (a1, a2, gates OFF and ON, the 40-boot paired cell), then NEED
    TARGET CARD.
- DAY66 (revuto on #731): the fanout's split is scoped to its snapshot (`prefix_copy_scoped`), with its census, cell
  and red arm (`day66/`), server lib 935 passed. No card.
- Item 11 (DAY60) is closed with its reading: CLAUSE NOT MET as stated before it ran, and y_minus_x is +86.3 (demote)
  and +76.0 (promote) in both orders. The class-isolating candidate cell is registered as text for the owner (section
  4).
- Pre-registered and waiting in order: item 12 (DAY61, design W), item 13 (DAY62, the retire seam: lines, a price, R1
  or R2), items 14 and 19 (DAY63, design L; item 17 re-read on top), item 18 (DAY64), item 20 (DAY65, design T-H).
- The owed 5090 cells (S4's half, V's half, item 16, and B1's (a1) and (a2) as a compatibility reading) queue on the
  5090.
