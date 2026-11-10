## arms
A $LANE/perf-bins/a1 peer-pull 2df1d4fdd5b11c20 35d749bddec8ba40 
B $LANE/perf-bins/pu2 device-push e3c0978799ad664c 31f31e438bc8510e 
C $LANE/perf-bins/pu11 device-push fea950a9a0a27fcc 88733b53572979dc 
D $LANE/perf-bins/pu2 device-push e3c0978799ad664c 31f31e438bc8510e prefill-only
E $LANE/perf-bins/pu11 peer-pull fea950a9a0a27fcc 88733b53572979dc prefill-only
F $LANE/perf-bins/pu11 peer-pull+serial fea950a9a0a27fcc 88733b53572979dc prefill-only
G $LANE/perf-bins/pu11 device-push+serial fea950a9a0a27fcc 88733b53572979dc prefill-only
H $LANE/perf-bins/pu11 device-push fea950a9a0a27fcc 88733b53572979dc prefill-only

## steps
gate-B-device-push start 09:09:28
gate-B-device-push rc 0 end 09:10:30
gate-B-device-push: 360 checks, 0 failed 0 FAIL lines
PASS stress/push_gather: 20000 gathers on devices [0, 1]: 0 words differ, 0 ranks flagged [every consumer reads what every producer pushed]
PASS stress/push_gather_control_word_first: 2000 gathers on devices [0, 1]: 839983 words differ, 0 ranks flagged [the word before the data is caught (words differ)]
gate-B-peer-pull start 09:10:30
gate-B-peer-pull rc 0 end 09:10:44
gate-B-peer-pull: 358 checks, 0 failed 0 FAIL lines
gate-C-device-push start 09:10:44
gate-C-device-push rc 0 end 09:10:59
gate-C-device-push: 360 checks, 0 failed 0 FAIL lines
PASS stress/push_gather: 20000 gathers on devices [0, 1]: 0 words differ, 0 ranks flagged [every consumer reads what every producer pushed]
PASS stress/push_gather_control_word_first: 2000 gathers on devices [0, 1]: 863661 words differ, 0 ranks flagged [the word before the data is caught (words differ)]
gate-C-peer-pull start 09:10:59
gate-C-peer-pull rc 0 end 09:11:13
gate-C-peer-pull: 358 checks, 0 failed 0 FAIL lines
gate-F-peer-pull-serial start 09:11:13
gate-F-peer-pull-serial rc 0 end 09:11:27
gate-F-peer-pull-serial: 358 checks, 0 failed 0 FAIL lines
gate-G-device-push-serial start 09:11:27
gate-G-device-push-serial rc 0 end 09:11:42
gate-G-device-push-serial: 360 checks, 0 failed 0 FAIL lines
PASS stress/push_gather: 20000 gathers on devices [0, 1]: 0 words differ, 0 ranks flagged [every consumer reads what every producer pushed]
PASS stress/push_gather_control_word_first: 2000 gathers on devices [0, 1]: 881014 words differ, 0 ranks flagged [the word before the data is caught (words differ)]
dec-run1-A start 09:11:42
dec-run1-A rc 0 end 09:12:52
dec-run2-B start 09:12:52
dec-run2-B rc 0 end 09:13:03
dec-run3-C start 09:13:03
dec-run3-C rc 0 end 09:13:15
dec-run4-C start 09:13:15
dec-run4-C rc 0 end 09:13:26
dec-run5-B start 09:13:26
dec-run5-B rc 0 end 09:13:37
dec-run6-A start 09:13:37
dec-run6-A rc 0 end 09:13:49
dec-run7-A start 09:13:49
dec-run7-A rc 0 end 09:14:00
dec-run8-B start 09:14:00
dec-run8-B rc 0 end 09:14:11
dec-run9-C start 09:14:11
dec-run9-C rc 0 end 09:14:22
pre-run1-D start 09:14:22
pre-run1-D rc 143 end 09:16:39
pre-run2-E start 09:16:39
pre-run2-E rc 0 end 09:17:13
pre-run3-F start 09:17:13
pre-run3-F rc 0 end 09:17:30
pre-run4-G start 09:17:30
pre-run4-G rc 0 end 09:17:52
pre-run5-H start 09:17:52
pre-run5-H rc 0 end 09:18:11
pre-trace start 09:18:11
pre-trace rc 0 end 09:18:34
pre-trace arm H

## slowdown counters
slowdown: 21 runs, 0 flagged (cards 01)

## one-token decode, 20 layers, a rank a card
| run | arm | step ms, median of each round | worst round p95 ms | us/layer | generated ids sha256 |
|---|---|---|---|---|---|
| run1-A | A | 14.196, 9.304, 9.295 | 15.302 | 465.2 | 89ea4569f744 |
| run2-B | B | 7.641, 7.641, 7.637 | 7.671 | 382.0 | 89ea4569f744 |
| run3-C | C | 7.339, 7.315, 7.340 | 7.397 | 367.0 | 89ea4569f744 |
| run4-C | C | 7.330, 7.319, 7.337 | 7.400 | 366.5 | 89ea4569f744 |
| run5-B | B | 7.637, 7.617, 7.620 | 7.670 | 381.0 | 89ea4569f744 |
| run6-A | A | 9.311, 9.313, 9.317 | 9.422 | 465.7 | 89ea4569f744 |
| run7-A | A | 9.294, 9.289, 9.317 | 9.428 | 464.7 | 89ea4569f744 |
| run8-B | B | 7.631, 7.611, 7.621 | 7.667 | 381.1 | 89ea4569f744 |
| run9-C | C | 7.334, 7.349, 7.365 | 7.442 | 367.5 | 89ea4569f744 |

| arm | run medians, ms/step | arm median | us/layer | against A |
|---|---|---|---|---|
| A | 9.304, 9.313, 9.294 | 9.304 | 465.2 | 1.000 |
| B | 7.641, 7.620, 7.621 | 7.621 | 381.1 | 0.819 |
| C | 7.339, 7.330, 7.349 | 7.339 | 367.0 | 0.789 |

generated ids identical across every run of every arm: True

## prefill
| run | arm | tokens | round | first token s | ms/layer | tok/s | issue s | gathers | copies | GB copied | peak GiB a rank |
|---|---|---|---|---|---|---|---|---|---|---|---|
| run1-D | D | 4096 | 0 | 27.5292 | 1376.46 | 149 | 27.3936 | 83 | 166 | 5.69 | 70.99 |
| run2-E | E | 4096 | 0 | 0.7188 | 35.94 | 5698 | 0.5785 | 143 | 286 | 4.54 | 71.00 |
| run2-E | E | 32768 | 0 | 22.5755 | 1128.77 | 1451 | 21.6465 | 263 | 526 | 36.31 | 75.65 |
| run3-F | F | 4096 | 0 | 0.7508 | 37.54 | 5455 | 0.6041 | 143 | 286 | 4.54 | 71.00 |
| run3-F | F | 32768 | 0 | 5.2936 | 264.68 | 6190 | 4.2918 | 263 | 526 | 36.31 | 75.65 |
| run4-G | G | 4096 | 0 | 1.2256 | 61.28 | 3342 | 0.9327 | 143 | 286 | 4.54 | 71.03 |
| run4-G | G | 32768 | 0 | 9.3345 | 466.73 | 3510 | 7.1053 | 263 | 526 | 36.31 | 75.89 |
| run5-H | H | 4096 | 0 | 0.9580 | 47.90 | 4275 | 0.7528 | 143 | 286 | 4.54 | 71.03 |
| run5-H | H | 32768 | 0 | 7.0209 | 351.05 | 4667 | 5.5026 | 263 | 526 | 36.31 | 75.89 |

| arm | tokens | runs | best ms | median ms | median ms/layer | median tok/s |
|---|---|---|---|---|---|---|
| D | 4096 | 1 | 27529.22 | 27529.22 | 1376.461 | 149 |
| E | 4096 | 1 | 718.82 | 718.82 | 35.941 | 5698 |
| F | 4096 | 1 | 750.82 | 750.82 | 37.541 | 5455 |
| G | 4096 | 1 | 1225.58 | 1225.58 | 61.279 | 3342 |
| H | 4096 | 1 | 958.04 | 958.04 | 47.902 | 4275 |
| E | 32768 | 1 | 22575.47 | 22575.47 | 1128.774 | 1451 |
| F | 32768 | 1 | 5293.60 | 5293.60 | 264.680 | 6190 |
| G | 32768 | 1 | 9334.53 | 9334.53 | 466.727 | 3510 |
| H | 32768 | 1 | 7020.91 | 7020.91 | 351.046 | 4667 |


## a block of 1, 2, 4, 8 rows
| run | arm | tokens | round | first token s | ms/layer | tok/s | issue s | gathers | copies | GB copied | peak GiB a rank |
|---|---|---|---|---|---|---|---|---|---|---|---|

| arm | tokens | runs | best ms | median ms | median ms/layer | median tok/s |
|---|---|---|---|---|---|---|

