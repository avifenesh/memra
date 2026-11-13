## arms
A $LANE/perf-bins/fin1 device-push 0c15ca18671a8e2f 13dcd3122ec11d97 
B $LANE/perf-bins/en2 device-push 89b87aea611282e1 6fc7fda6765e1179 

## steps
dec-run1-A start 11:12:19
dec-run1-A rc 0 end 11:13:01
dec-run2-B start 11:13:01
dec-run2-B rc 0 end 11:13:15
dec-run3-B start 11:13:15
dec-run3-B rc 0 end 11:13:30
dec-run4-A start 11:13:30
dec-run4-A rc 0 end 11:13:42
dec-run5-A start 11:13:42
dec-run5-A rc 0 end 11:13:53
dec-run6-B start 11:13:53
dec-run6-B rc 0 end 11:14:08
pre-run1-A start 11:14:08
pre-run1-A rc 0 end 11:14:25
pre-run2-B start 11:14:25
pre-run2-B rc 0 end 11:14:43
pre-run3-B start 11:14:43
pre-run3-B rc 0 end 11:15:00
pre-run4-A start 11:15:00
pre-run4-A rc 0 end 11:15:17
pre-trace start 11:15:17
pre-trace rc 0 end 11:15:38
pre-trace arm B
trace-A start 11:15:38
trace-A rc 0 end 11:15:53
trace-B start 11:15:53
trace-B rc 0 end 11:16:11

## slowdown counters
slowdown: 13 runs, 0 flagged (cards 01)

## one-token decode, 20 layers, a rank a card
| run | arm | step ms, median of each round | worst round p95 ms | us/layer | generated ids sha256 |
|---|---|---|---|---|---|
| run1-A | A | 7.284, 7.268, 7.257 | 7.333 | 363.4 | 89ea4569f744 |
| run2-B | B | 7.305, 7.281, 7.268 | 7.339 | 364.1 | 89ea4569f744 |
| run3-B | B | 7.281, 7.290, 7.290 | 7.345 | 364.5 | 89ea4569f744 |
| run4-A | A | 7.287, 7.278, 7.267 | 7.341 | 363.9 | 89ea4569f744 |
| run5-A | A | 7.285, 7.281, 7.267 | 7.342 | 364.1 | 89ea4569f744 |
| run6-B | B | 7.314, 7.287, 7.275 | 7.353 | 364.3 | 89ea4569f744 |

| arm | run medians, ms/step | arm median | us/layer | against A |
|---|---|---|---|---|
| A | 7.268, 7.278, 7.281 | 7.278 | 363.9 | 1.000 |
| B | 7.281, 7.290, 7.287 | 7.287 | 364.3 | 1.001 |

generated ids identical across every run of every arm: True

## prefill
| run | arm | tokens | round | first token s | ms/layer | tok/s | issue s | gathers | copies | GB copied | peak GiB a rank |
|---|---|---|---|---|---|---|---|---|---|---|---|
| run1-A | A | 4096 | 0 | 0.7281 | 36.40 | 5626 | 0.5904 | 143 | 286 | 4.54 | 71.06 |
| run1-A | A | 32768 | 0 | 4.7484 | 237.42 | 6901 | 3.8293 | 263 | 526 | 36.31 | 76.14 |
| run2-B | B | 4096 | 0 | 0.4812 | 24.06 | 8512 | 0.3430 | 143 | 286 | 4.54 | 71.06 |
| run2-B | B | 32768 | 0 | 3.3629 | 168.14 | 9744 | 2.4328 | 263 | 526 | 36.31 | 76.14 |
| run3-B | B | 4096 | 0 | 0.4772 | 23.86 | 8584 | 0.3385 | 143 | 286 | 4.54 | 71.06 |
| run3-B | B | 32768 | 0 | 3.3639 | 168.19 | 9741 | 2.4274 | 263 | 526 | 36.31 | 76.14 |
| run4-A | A | 4096 | 0 | 0.7098 | 35.49 | 5771 | 0.5731 | 143 | 286 | 4.54 | 71.06 |
| run4-A | A | 32768 | 0 | 4.6604 | 233.02 | 7031 | 3.7384 | 263 | 526 | 36.31 | 76.14 |

| arm | tokens | runs | best ms | median ms | median ms/layer | median tok/s |
|---|---|---|---|---|---|---|
| A | 4096 | 2 | 709.79 | 718.92 | 35.946 | 5697 |
| B | 4096 | 2 | 477.16 | 479.19 | 23.959 | 8548 |
| A | 32768 | 2 | 4660.43 | 4704.40 | 235.220 | 6965 |
| B | 32768 | 2 | 3362.85 | 3363.36 | 168.168 | 9743 |

4096 tokens: B/A best 0.672 (A 709.79 ms, B 477.16 ms), median 0.667
32768 tokens: B/A best 0.722 (A 4660.43 ms, B 3362.85 ms), median 0.715

## a block of 1, 2, 4, 8 rows
| run | arm | tokens | round | first token s | ms/layer | tok/s | issue s | gathers | copies | GB copied | peak GiB a rank |
|---|---|---|---|---|---|---|---|---|---|---|---|

| arm | tokens | runs | best ms | median ms | median ms/layer | median tok/s |
|---|---|---|---|---|---|---|

