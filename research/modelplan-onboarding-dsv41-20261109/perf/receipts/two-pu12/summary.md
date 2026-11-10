## arms
A $LANE/perf-bins/pu11 device-push fea950a9a0a27fcc 88733b53572979dc 
B $LANE/perf-bins/pu12 device-push bd05679b3ce6268d 78f2838b22aede29 
C $LANE/perf-bins/pu12 peer-pull bd05679b3ce6268d 78f2838b22aede29 prefill-only
D $LANE/perf-bins/pu12 peer-pull+serial bd05679b3ce6268d 78f2838b22aede29 prefill-only
E $LANE/perf-bins/pu12 device-push+serial bd05679b3ce6268d 78f2838b22aede29 prefill-only
F $LANE/perf-bins/pu12 device-push bd05679b3ce6268d 78f2838b22aede29 prefill-only

## steps
gate-B-device-push start 09:41:24
gate-B-device-push rc 0 end 09:41:39
gate-B-device-push: 360 checks, 0 failed 0 FAIL lines
PASS stress/push_gather: 20000 gathers on devices [0, 1]: 0 words differ, 0 ranks flagged [every consumer reads what every producer pushed]
PASS stress/push_gather_control_word_first: 2000 gathers on devices [0, 1]: 898226 words differ, 0 ranks flagged [the word before the data is caught (words differ)]
gate-B-peer-pull start 09:41:39
gate-B-peer-pull rc 0 end 09:41:53
gate-B-peer-pull: 358 checks, 0 failed 0 FAIL lines
gate-D-peer-pull-serial start 09:41:53
gate-D-peer-pull-serial rc 0 end 09:42:07
gate-D-peer-pull-serial: 358 checks, 0 failed 0 FAIL lines
gate-E-device-push-serial start 09:42:07
gate-E-device-push-serial rc 0 end 09:42:21
gate-E-device-push-serial: 360 checks, 0 failed 0 FAIL lines
PASS stress/push_gather: 20000 gathers on devices [0, 1]: 0 words differ, 0 ranks flagged [every consumer reads what every producer pushed]
PASS stress/push_gather_control_word_first: 2000 gathers on devices [0, 1]: 857067 words differ, 0 ranks flagged [the word before the data is caught (words differ)]
pre-run1-C start 09:42:21
pre-run1-C rc 0 end 09:42:38
pre-run2-D start 09:42:38
pre-run2-D rc 0 end 09:42:55
pre-run3-E start 09:42:55
pre-run3-E rc 0 end 09:43:12
pre-run4-F start 09:43:12
pre-run4-F rc 0 end 09:43:29
pre-trace start 09:43:29
pre-trace rc 0 end 09:43:51
pre-trace arm F

## slowdown counters
slowdown: 9 runs, 0 flagged (cards 01)

## one-token decode, 20 layers, a rank a card
| run | arm | step ms, median of each round | worst round p95 ms | us/layer | generated ids sha256 |
|---|---|---|---|---|---|

| arm | run medians, ms/step | arm median | us/layer | against A |
|---|---|---|---|---|

## prefill
| run | arm | tokens | round | first token s | ms/layer | tok/s | issue s | gathers | copies | GB copied | peak GiB a rank |
|---|---|---|---|---|---|---|---|---|---|---|---|
| run1-C | C | 4096 | 0 | 0.7268 | 36.34 | 5636 | 0.5898 | 143 | 286 | 4.54 | 71.06 |
| run1-C | C | 32768 | 0 | 4.9805 | 249.03 | 6579 | 4.0683 | 263 | 526 | 36.31 | 76.14 |
| run2-D | D | 4096 | 0 | 0.7303 | 36.51 | 5609 | 0.5881 | 143 | 286 | 4.54 | 71.06 |
| run2-D | D | 32768 | 0 | 5.2291 | 261.46 | 6266 | 4.2521 | 263 | 526 | 36.31 | 76.14 |
| run3-E | E | 4096 | 0 | 0.7271 | 36.35 | 5634 | 0.5858 | 143 | 286 | 4.54 | 71.06 |
| run3-E | E | 32768 | 0 | 5.2027 | 260.13 | 6298 | 4.2234 | 263 | 526 | 36.31 | 76.14 |
| run4-F | F | 4096 | 0 | 0.7018 | 35.09 | 5836 | 0.5653 | 143 | 286 | 4.54 | 71.06 |
| run4-F | F | 32768 | 0 | 4.9764 | 248.82 | 6585 | 4.0627 | 263 | 526 | 36.31 | 76.14 |

| arm | tokens | runs | best ms | median ms | median ms/layer | median tok/s |
|---|---|---|---|---|---|---|
| C | 4096 | 1 | 726.75 | 726.75 | 36.338 | 5636 |
| D | 4096 | 1 | 730.26 | 730.26 | 36.513 | 5609 |
| E | 4096 | 1 | 727.06 | 727.06 | 36.353 | 5634 |
| F | 4096 | 1 | 701.85 | 701.85 | 35.092 | 5836 |
| C | 32768 | 1 | 4980.51 | 4980.51 | 249.025 | 6579 |
| D | 32768 | 1 | 5229.15 | 5229.15 | 261.457 | 6266 |
| E | 32768 | 1 | 5202.67 | 5202.67 | 260.133 | 6298 |
| F | 32768 | 1 | 4976.41 | 4976.41 | 248.820 | 6585 |


## a block of 1, 2, 4, 8 rows
| run | arm | tokens | round | first token s | ms/layer | tok/s | issue s | gathers | copies | GB copied | peak GiB a rank |
|---|---|---|---|---|---|---|---|---|---|---|---|

| arm | tokens | runs | best ms | median ms | median ms/layer | median tok/s |
|---|---|---|---|---|---|---|

