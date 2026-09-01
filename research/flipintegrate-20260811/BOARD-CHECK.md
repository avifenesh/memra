# Perf-board check — dual-PP default and qmatvec arm 1

Date: 2026-08-12

## Recommendation

**No tracked cell moves.** Do not edit
`research/tune-data/current-board.json`, and do not regenerate README.md or
`docs/PERFORMANCE.md` from it for these two results.

## Reasoning

The numeric surfaces in `current-board.json` are the RTX 5090 Laptop
`plain_decode`, `plain_decode_depth`, and `speculative` cells, the hand-selected
5090 `extra_card_rows`, and the rented-H100 `h100_board`. Its
`supported_models` section is intentionally numbers-free. There is no tracked
RTX PRO 6000 pair serving board or Step semantic-mix microbenchmark cell.

- The dual-PP flip result is an hyperscaler 2x RTX PRO 6000 Server Edition **serving**
  receipt: Step-3.7 c=8 aggregate throughput moved from 133.553 tok/s on the
  serial rollback to 158.065 tok/s on the naked dual-active default
  (+18.354%, N=5/arm interleaved). Putting that value into a tracked 5090 or
  H100 cell would mix hardware and protocol. It belongs in the dedicated PRO
  prose and the lane result, not `current-board.json`.
- The qmatvec result is a fixed 315-launch Step semantic-mix **microbenchmark**
  on one PRO GPU: 2.863324 to 2.673256 ms (-6.638%, N=8/arm ABBA). It is not an
  end-to-end decode measurement. The tracked 5090 single-stream decode cells
  are unchanged, so no `plain_decode`, `plain_decode_depth`, `speculative`, or
  `extra_card_rows` value moves; the H100 board is also untouched.

The generated board date and every generated marker block should therefore
remain byte-for-byte unchanged in this integration lane.
