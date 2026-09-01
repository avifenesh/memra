# Excluded attempt 2 — cycle-boundary same-window duplicate

This attempt is excluded from scoring. It completed six Q27 budget boots and reached Q35
repetition 1 / 1,024 MiB before the harness failed closed at c=32.

The c=32 cell crossed the 96-key working-set cycle boundary. The independently reshuffled next
cycle placed two just-used keys back into that same concurrent cell. Runtime prefix dedup then
legitimately credited those two requests with 1,024 cached tokens each, while two other requests
were full 4,860-token hits. The exact `/metrics` delta was therefore 4 hits / 36 misses and
11,768 cached tokens, whereas the old harness classifier counted only the two full hits and
expected 2 hits / 38 misses. The two partial requests are preserved in `sweep.jsonl` with ids
`cmpl-30dcf17f905a06f44d85f8734c64068a` and
`cmpl-6daec457d8d501988460f148089d2b17`.

This is a workload-generator defect, not a server failure. The corrected scheduler keeps working
keys distinct within each concurrent cell by swapping a duplicate next-cycle key later within
that same permutation; complete cycles remain permutations of all 96 keys. The full N=5 campaign
restarts from zero. No row from this attempt is used in the scored reduction.
