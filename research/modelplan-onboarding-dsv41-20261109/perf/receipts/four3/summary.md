## arms
A $LANE/perf-bins/pu2 device-push e3c0978799ad664c 31f31e438bc8510e 
B $LANE/perf-bins/pu10 device-push 190de1c78cc8ba9a fac2f881dbede9c8 
C $LANE/perf-bins/pu10 peer-pull 190de1c78cc8ba9a fac2f881dbede9c8 prefill-only
D $LANE/perf-bins/pu10 device-push 190de1c78cc8ba9a fac2f881dbede9c8 prefill-only

## steps
gate-B-device-push start 08:47:36
gate-B-device-push rc 0 end 08:47:56
gate-B-device-push: 706 checks, 0 failed 0 FAIL lines
PASS stress/push_gather: 20000 gathers on devices [0, 1, 2, 3]: 0 words differ, 0 ranks flagged [every consumer reads what every producer pushed]
PASS stress/push_gather_control_word_first: 2000 gathers on devices [0, 1, 2, 3]: 30463342 words differ, 0 ranks flagged [the word before the data is caught (words differ)]
gate-C-peer-pull start 08:47:56
gate-C-peer-pull rc 0 end 08:48:14
gate-C-peer-pull: 704 checks, 0 failed 0 FAIL lines
gate-C-default start 08:48:14
gate-C-default rc 0 end 08:48:32
gate-C-default: 704 checks, 0 failed 0 FAIL lines
gate-D-device-push start 08:48:32
gate-D-device-push rc 0 end 08:48:51
gate-D-device-push: 706 checks, 0 failed 0 FAIL lines
PASS stress/push_gather: 20000 gathers on devices [0, 1, 2, 3]: 0 words differ, 0 ranks flagged [every consumer reads what every producer pushed]
PASS stress/push_gather_control_word_first: 2000 gathers on devices [0, 1, 2, 3]: 30660034 words differ, 0 ranks flagged [the word before the data is caught (words differ)]
pre-run1-C start 08:48:51
pre-run1-C rc 0 end 08:49:45
pre-run2-D start 08:49:45
pre-run2-D rc 0 end 08:50:25
pre-trace start 08:50:25
pre-trace rc 0 end 08:51:26
pre-trace arm D
dec-run1-A start 08:51:26
dec-run1-A rc 0 end 08:51:45
dec-run2-B start 08:51:45
dec-run2-B rc 0 end 08:52:03
dec-run3-B start 08:52:03
dec-run3-B rc 0 end 08:52:21
dec-run4-A start 08:52:21
dec-run4-A rc 0 end 08:52:40
dec-run5-A start 08:52:40
dec-run5-A rc 0 end 08:52:58
dec-run6-B start 08:52:58
dec-run6-B rc 0 end 08:53:16

## slowdown counters
slowdown: 13 runs, 0 flagged (cards 0123)

## one-token decode, 40 layers, four ranks a card each
| run | arm | step ms, median of each round | worst round p95 ms | us/layer | generated ids sha256 |
|---|---|---|---|---|---|
| run1-A | A | 12.380, 12.357, 12.353 | 12.417 | 308.9 | f1f9c8552dd3 |
| run2-B | B | 11.642, 11.624, 11.616 | 11.704 | 290.6 | f1f9c8552dd3 |
| run3-B | B | 11.637, 11.620, 11.617 | 11.710 | 290.5 | f1f9c8552dd3 |
| run4-A | A | 12.390, 12.360, 12.347 | 12.435 | 309.0 | f1f9c8552dd3 |
| run5-A | A | 12.369, 12.366, 12.341 | 12.422 | 309.2 | f1f9c8552dd3 |
| run6-B | B | 11.643, 11.641, 11.635 | 11.707 | 291.0 | f1f9c8552dd3 |

| arm | run medians, ms/step | arm median | us/layer | against A |
|---|---|---|---|---|
| A | 12.357, 12.360, 12.366 | 12.360 | 309.0 | 1.000 |
| B | 11.624, 11.620, 11.641 | 11.624 | 290.6 | 0.940 |

generated ids identical across every run of every arm: True

## prefill
| run | arm | tokens | round | first token s | ms/layer | tok/s | issue s | gathers | copies | GB copied | peak GiB a rank |
|---|---|---|---|---|---|---|---|---|---|---|---|
| run1-C | C | 4096 | 0 | 4.2503 | 106.26 | 964 | 3.4233 | 288 | 3456 | 27.28 | 71.00 |
| run1-C | C | 32768 | 0 | 33.1199 | 828.00 | 989 | 29.1901 | 528 | 6336 | 218.26 | 75.34 |
| run2-D | D | 4096 | 0 | 2.5490 | 63.73 | 1607 | 2.1093 | 288 | 3456 | 27.28 | 71.04 |
| run2-D | D | 32768 | 0 | 20.0105 | 500.26 | 1638 | 17.8770 | 528 | 6336 | 218.26 | 75.71 |

| arm | tokens | runs | best ms | median ms | median ms/layer | median tok/s |
|---|---|---|---|---|---|---|
| C | 4096 | 1 | 4250.35 | 4250.35 | 106.259 | 964 |
| D | 4096 | 1 | 2549.00 | 2549.00 | 63.725 | 1607 |
| C | 32768 | 1 | 33119.87 | 33119.87 | 827.997 | 989 |
| D | 32768 | 1 | 20010.46 | 20010.46 | 500.262 | 1638 |


## a block of 1, 2, 4, 8 rows
| run | arm | tokens | round | first token s | ms/layer | tok/s | issue s | gathers | copies | GB copied | peak GiB a rank |
|---|---|---|---|---|---|---|---|---|---|---|---|

| arm | tokens | runs | best ms | median ms | median ms/layer | median tok/s |
|---|---|---|---|---|---|---|

