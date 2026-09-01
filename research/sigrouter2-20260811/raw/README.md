# Raw evidence notes

All text, JSON, CSV, thermal, request, server, build, correctness, and golden receipts are
committed in this directory. `box1-perf-attempt1/driver.log` preserves the pre-measurement harness
failure; no result from that attempt was admitted.

The primary Nsight reports and derived SQLite databases are intentionally not committed. Nsight
reports can contain the complete process environment, the repository ignores them, and the
pre-push hook rejects profile blobs. They remain on box1 under
`/home/ubuntu/memra-cx-sigrouter2/research/sigrouter2-20260811/raw/box1-perf/`:

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `nsys-default.nsys-rep` | 38,481,148 | `854c487b2a78702475c4fe50f12e808046ae3c652dd427ec0f574cbaa0440f34` |
| `nsys-default.sqlite` | 97,402,880 | `c26bac3d0ffa25e6c823208d30a3ad66484edbbc7b4628275e5078a24cc651ac` |
| `nsys-inc1.nsys-rep` | 40,985,645 | `fa7265664c58973543723c16beb7ac17d81f8cf0231b25e0b8c219ee68b42aee` |
| `nsys-inc1.sqlite` | 103,669,760 | `8df3d625f86259f6f7e67b6e999917c95c82f917706f7701bc6a2a853d3f6fdc` |

The same hashes are machine-produced in `box1-perf/nsys-{default,inc1}-SHA256SUMS`. The committed
`nsys-*-memcpy.json` files are exact grouped queries over each SQLite CUPTI memcpy table;
`nsys-*-cuda-api.csv` and the memory-size/time CSVs are direct `nsys stats` exports.

`local-26b-ab/` is the same-window N=5 settle for the standing `26b-spec-d1736` rolling-baseline
alert. It compares an exact binary built from lane base `30418923` with the candidate under one
local GPU lock. `summary.json` is the machine reduction; `points.jsonl`, per-run logs, binary and
artifact hashes, and before/after thermal snapshots retain the underlying evidence.

`local-ci-perf-quick-final.log` is the complete hook-freshness battery from rebuilt HEAD release
binaries. It includes the standing correctness, serving, stress, acceptance, and 31B quick-perf
cells; the latter appended four machine rows to `research/tune-data/perf-ci.jsonl`.
