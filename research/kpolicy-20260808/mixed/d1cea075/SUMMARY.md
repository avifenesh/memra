# Mixed-workload before/after

N=3 independent server boots per arm, rep-major alternating order.
Each rep includes cold-short, cold-long, cached-long setup+continuation, then a staggered c=4 wave with two short, one cold-long, and one cached-long request.

| arm | aggregate tok/s median (range) | c=4 wave tok/s median (range) | workload wall median | cached tok median |
|---|---:|---:|---:|---:|
| before | 294.06 (293.90-295.51) | 390.51 (388.80-391.15) | 5.033s | 11098 |
| after | 293.41 (293.38-294.00) | 390.36 (389.90-391.16) | 5.037s | 11096 |

- Aggregate delta: -0.22%.
- c=4 wave delta: -0.04%.
- Workload wall delta: +0.09%.

## Per-request modes

- before: cached-setup:spec=3, seq-cached:spec=3, seq-long:spec=3, seq-short:spec=3, wave-cached:plain=3, wave-long:plain=3, wave-short-a:plain=3, wave-short-b:plain=3
- after: cached-setup:spec=3, seq-cached:spec=3, seq-long:spec=3, seq-short:spec=3, wave-cached:plain=3, wave-long:plain=3, wave-short-a:plain=3, wave-short-b:plain=3
