# Supplementary tune-box observations

Pairs 4 and 5 completed and were mirrored before the tune box was destroyed.
The owner subsequently froze the current PR receipt at pairs 1 through 3 and
reserved pairs 4 and 5 for a later owner-scheduled pair-box cell. These additional
tune-box observations remain archived and are excluded from the current receipt
summary. The owner-scheduled completion cell for pairs 4 and 5 remains pending.

| Pair | OFF wall s | OFF tok/s | OFF server ms/token | ON wall s | ON tok/s | ON server ms/token |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 6.235 | 82.12 | 11.859 | 5.611 | 91.24 | 10.640 |
| 5 | 6.229 | 82.19 | 11.841 | 5.604 | 91.36 | 10.625 |

All four rows generated 512 tokens. Their raw data remains under
`raw/served/pair4-arm*/` and `raw/served/pair5-arm*/`. The original
`raw/served/rows.json` and `summary.json` preserve all ten collected tune-box
rows; `current-receipt-rows.json` and `current-receipt-summary.json` identify
the owner-selected current receipt. The source/binary and measurement protocol
are documented in [RESULTS.md](RESULTS.md).

The current receipt now includes the later owner-scheduled pairs 4/5 under `raw/pair-box/`. These earlier tune observations remain supplementary and excluded from that five-pair selection.
