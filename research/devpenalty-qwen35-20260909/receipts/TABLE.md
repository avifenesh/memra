## Page-task, vendor non-thinking shape (client sends no sampling field)

| c | per-stream decode tok/s OFF | ON | delta | aggregate decode tok/s OFF | ON | delta | TTFT t1 p50 s OFF | ON | e2e s OFF | ON |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 233.5 | 242.2 | +3.7% | 233.5 | 242.2 | +3.7% | 0.261 | 0.263 | 1.32 | 1.34 |
| 4 | 27.3 | 138.2 | +405.4% | 109.6 | 527.2 | +381.0% | 0.962 | 0.946 | 10.32 | 2.96 |
| 8 | 14.5 | 99.0 | +580.7% | 115.5 | 758.0 | +556.3% | 1.880 | 1.876 | 19.01 | 4.83 |

## presence_penalty 0.0 control (penalty-free rows never enter the door)

| c | per-stream decode tok/s OFF | ON | delta | aggregate decode tok/s OFF | ON | delta |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 259.3 | 248.8 | -4.1% | 259.3 | 248.8 | -4.1% |
| 4 | 136.0 | 133.8 | -1.6% | 524.6 | 514.1 | -2.0% |
| 8 | 94.8 | 96.3 | +1.5% | 742.8 | 739.4 | -0.5% |

## How much of the collapse the door removes

| c | vendor OFF agg | vendor ON agg | pp0 ceiling agg | ON as % of ceiling | remaining gap tok/s |
|---:|---:|---:|---:|---:|---:|
| 1 | 233.5 | 242.2 | 259.3 | 93.4% | 17.1 |
| 4 | 109.6 | 527.2 | 524.6 | 100.5% | -2.6 |
| 8 | 115.5 | 758.0 | 742.8 | 102.0% | -15.1 |

## pp512/tg128 twin, vendor shape, c=1

| arm | prompt tok | pp tok/s | tg tok/s | completion tok | n |
|---|---:|---:|---:|---:|---:|
| off | 501 | 5,813 | 210.2 | 128 | 15 |
| on | 501 | 5,753 | 215.1 | 128 | 15 |

pp delta -1.0%, tg delta +2.4%

## Spec engagement (turn 1)

| arm | cell | c | requests | usage.spec present | K>0 | median rounds | acceptance |
|---|---|---:|---:|---:|---:|---:|---:|
| off | vendor | 1 | 15 | 15 | 15 | 61 | 0.646 |
| off | vendor | 4 | 60 | 0 | 0 | - | - |
| off | vendor | 8 | 120 | 0 | 0 | - | - |
| off | pp0 | 1 | 9 | 9 | 9 | 62 | 0.672 |
| off | pp0 | 4 | 36 | 0 | 0 | - | - |
| off | pp0 | 8 | 72 | 0 | 0 | - | - |
| on | vendor | 1 | 15 | 15 | 15 | 60 | 0.688 |
| on | vendor | 4 | 60 | 0 | 0 | - | - |
| on | vendor | 8 | 120 | 0 | 0 | - | - |
| on | pp0 | 1 | 9 | 9 | 9 | 63 | 0.677 |
| on | pp0 | 4 | 36 | 0 | 0 | - | - |
| on | pp0 | 8 | 72 | 0 | 0 | - | - |

## Greedy exactness instrument

- `greedy-pp15`: 6 boots, BYTE-IDENTICAL (bb40202d3e37eceac745334394be2a1f1c4e9ef0d850d1c96fc8bce896fa98fc), 185 tokens
- `greedy-pp00`: 6 boots, BYTE-IDENTICAL (4139d34e4470ff074ae44e39ad2de02aef38f4ff903e0158734f46b0369ffae0), 200 tokens
