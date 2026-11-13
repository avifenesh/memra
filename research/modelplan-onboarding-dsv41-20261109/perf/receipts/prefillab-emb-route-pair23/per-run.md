# emb against route 7450e9f065: first-token time per run

Arms (`arms.txt`): A = route `7450e9f065` (`land-route`), B = emb `21bb0500b3` (`land-emb`). Two ranks on cards 2 and 3,
20 layers resident, `MEMRA_FULL_PREC=1`, `MEMRA_DSV41_TP_ISSUE=parallel`, default transport; A B B A in three blocks,
6 runs an arm, 3 rounds a run (4,096, 32,768, 131,072 tokens and back). Computed from `summary.md` (first token s).

| tokens | A, each run's median over its 3 rounds (n 6): median [min, max] ms | B, same | per-run ranges | every round (n 18): A -> B median [min, max] | best B/A | median B/A | rounds 1-2 (n 12) median B/A |
|---|---|---|---|---|---|---|---|
| 4,096 | 490.2 [488.0, 492.1] | 483.8 [480.6, 484.4] | apart | 489.1 [458.6, 493.5] -> 483.5 [452.7, 486.3] | 0.987 | 0.989 | 0.987 |
| 32,768 | 3,317.3 [3,304.8, 3,327.5] | 3,243.1 [3,226.2, 3,246.6] | apart | 3,313.5 [3,207.9, 3,340.6] -> 3,240.8 [3,174.1, 3,266.9] | 0.989 | 0.978 | 0.977 |
| 131,072 | 13,753.8 [13,545.5, 13,783.4] | 13,397.4 [13,313.6, 13,415.3] | apart | 13,753.8 [13,442.9, 13,887.5] -> 13,397.4 [13,214.8, 13,523.8] | 0.983 | 0.974 | 0.975 |
