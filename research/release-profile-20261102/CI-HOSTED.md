# Hosted CI job durations, first run of this change

`ci.yml` full mode on GitHub `ubuntu-24.04` runners, run 41514463732 (head 6b21a1fead, 2026-11-02),
against the latest successful scheduled runs on `main` (2026-10-29 to 2026-11-02). Minutes, from
each job's start and completion timestamps. The `gates` failure is the Python public-CI witness
hitting its 660 s deadline (every witness command ran about 1.5 times its time on `main`); it builds
no Rust.

| job | main scheduled runs (min) | this change (min) |
|---|---|---|
| arch-coverage (100a) | 13.9, 17.8, 11.6, 19.5 | 45.6 (success) |
| boundary | 17.3, 11.3, 17.6, 17.5 | 15.5 (success) |
| build | 12.8, 17.8, 18.0, 12.0 | 59.6 (success; timeout is 60) |
| changes | 1.3, 1.3, 1.3, 1.6 | 1.6 (success) |
| clippy | 9.9, 9.1, 6.4, 10.1 | 8.7 (success) |
| engine-tests | 8.8, 11.1, 9.3, 9.5 | 13.0 (success) |
| gates | 14.6, 17.7, 20.7, 20.7 | 13.7 (failure: witness deadline) |
| portable-suites | 5.8, 5.8, 5.8, 5.3 | 5.2 (success) |
| publish-dryrun | 18.0, 18.0, 18.2, 15.6 | 18.3 (success) |
| server-tests | 11.9, 12.2, 13.1, 15.1 | 20.0 (success) |
