# Continuing-session MTP depth study

[Results and scope](RESULTS.md): calibrated fixed depth remains ahead of the cost
learner on both tested artifacts with native prompt-prefix reuse.

The publication contains data, audits and a frozen source archive. The
experimental runtime is not applied to the engine.

## Reproduce the recorded reports

Use Python 3.12 or newer. Choose a new destination:

```sh
python3 research/mtp-continuing-session-20260921/unpack_receipts.py \
  --destination /path/to/new-records

python3 research/mtp-continuing-session-20260921/analyze.py \
  --family qwen \
  --ledger /path/to/new-records/experiment/qwen-selected-sets.json \
  --receipts /path/to/new-records/experiment \
  --output /path/to/qwen-audit.json

python3 research/mtp-continuing-session-20260921/analyze.py \
  --family gemma \
  --ledger /path/to/new-records/experiment/gemma-selected-sets.json \
  --receipts /path/to/new-records/experiment \
  --output /path/to/gemma-audit.json
```

The extractor verifies archive size/hash, every member hash, regular-file-only
contents, path confinement, family boundaries and destination freshness before
writing. [REPRODUCTION.json](receipts/REPRODUCTION.json) records exact matches for
both reports from these archives.

## Rebuild the measurement runtime

`receipts/runtime-source.tar.gz` is the complete buildable source for the pinned
measurement commit. Its expected SHA-256 is in
[the manifest](receipts/manifest.json). Extract it into a separate new source
directory, install the pinned Rust toolchain and CUDA 13.1, and build the
`mtp-depth-study` and `gemma-depth-study` binaries there. The model lock files
name the exact weights and hashes; model access must be obtained from their
hosts.

Use the archived `prepare_workloads.py`, `qualify.py` and `controller.py` from
that source tree to repeat the experiment. The commands, effective flags,
binary hashes, token tapes and telemetry for every recorded run are included
in the record archives. The current engine checkout is not the measurement
source.
