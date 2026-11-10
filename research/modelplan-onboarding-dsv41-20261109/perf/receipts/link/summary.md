## arms
A $LANE/perf-bins/pu11 device-push fea950a9a0a27fcc 88733b53572979dc 
B $LANE/perf-bins/fin1 device-push 0c15ca18671a8e2f 13dcd3122ec11d97 
C $LANE/perf-bins/pu11 peer-pull fea950a9a0a27fcc 88733b53572979dc prefill-only
D $LANE/perf-bins/fin1 peer-pull 0c15ca18671a8e2f 13dcd3122ec11d97 prefill-only

## steps
pre-run1-C start 10:16:07
pre-run1-C rc 0 end 10:18:13
pre-run2-D start 10:18:13
pre-run2-D rc 0 end 10:18:31
pre-trace start 10:18:31
pre-trace rc 0 end 10:18:53
pre-trace arm D

## slowdown counters
slowdown: 3 runs, 0 flagged (cards 01)

## one-token decode, 20 layers, a rank a card
| run | arm | step ms, median of each round | worst round p95 ms | us/layer | generated ids sha256 |
|---|---|---|---|---|---|

| arm | run medians, ms/step | arm median | us/layer | against A |
|---|---|---|---|---|

## prefill
| run | arm | tokens | round | first token s | ms/layer | tok/s | issue s | gathers | copies | GB copied | peak GiB a rank |
|---|---|---|---|---|---|---|---|---|---|---|---|
| run1-C | C | 4096 | 0 | 14.8544 | 742.72 | 276 | 14.7292 | 143 | 286 | 4.54 | 71.00 |
| run1-C | C | 32768 | 0 | 65.6129 | 3280.65 | 499 | 64.6976 | 263 | 526 | 36.31 | 75.65 |
| run2-D | D | 4096 | 0 | 0.7730 | 38.65 | 5299 | 0.6351 | 143 | 286 | 4.54 | 71.06 |
| run2-D | D | 32768 | 0 | 5.2171 | 260.86 | 6281 | 4.3020 | 263 | 526 | 36.31 | 76.14 |

| arm | tokens | runs | best ms | median ms | median ms/layer | median tok/s |
|---|---|---|---|---|---|---|
| C | 4096 | 1 | 14854.39 | 14854.39 | 742.719 | 276 |
| D | 4096 | 1 | 772.96 | 772.96 | 38.648 | 5299 |
| C | 32768 | 1 | 65612.94 | 65612.94 | 3280.647 | 499 |
| D | 32768 | 1 | 5217.14 | 5217.14 | 260.857 | 6281 |


## a block of 1, 2, 4, 8 rows
| run | arm | tokens | round | first token s | ms/layer | tok/s | issue s | gathers | copies | GB copied | peak GiB a rank |
|---|---|---|---|---|---|---|---|---|---|---|---|

| arm | tokens | runs | best ms | median ms | median ms/layer | median tok/s |
|---|---|---|---|---|---|---|

