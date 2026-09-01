# Profiler artifact note

`nsys-default.nsys-rep` and `nsys-rollback.nsys-rep` are the primary raw Nsight Systems traces.
Their SHA-256 values match the first entry in the corresponding `nsys-*-SHA256SUMS` files. The
repository intentionally ignores `*.nsys-rep`, so they are retained on box1 rather than force-added.

The harness also exported two derived SQLite databases on box1 for `extract-nsys.py`:

- `nsys-default.sqlite` (103,665,664 bytes)
- `nsys-rollback.sqlite` (100,073,472 bytes)

Those derived files exceed the practical Git object limit and are not duplicated here. All four
large artifacts remain in
`/home/ubuntu/memra-cx-sigrouter/research/sigrouter-20260811/raw/box1-perf-arm-expf/` on box1;
their hashes are retained in `nsys-*-SHA256SUMS`. The extracted `nsys-*-memcpy.json` files and CUDA
API CSV reports committed here were produced from the original exports.
