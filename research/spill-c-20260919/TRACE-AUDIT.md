# PLE trace audit and portable request policy — day 3

Scope: local `research/` in `avifenesh/memra`, integrated source `b1df0e73`.
No web, model, GPU, external trace, or paused-model execution.

## Search actually run

- Enumerated filenames matching `*ngram*`, `*ple*trace*`, `*trace*ple*` with `find`.
  The only n-gram ID trace filename found was this lane's already-labeled synthetic fixture.
- Searched `.json/.jsonl/.tsv/.csv/.txt/.log/.gz` files (including ignored raw logs)
  for case-insensitive `ngram_ids`, `ngram trace`, `ngram_trace`, `ple row_ids`, and
  `"row_ids"`. 42,164 candidate files examined outside this lane. Thirteen files
  exceeding the 16 MiB read bound were subsequently streamed in 1 MiB chunks,
  including the 23,598,855-byte Qwen4Exp NVFP4 census. No oversized file matched.
- `fixtures/trace-search.json` records matching paths, SHA256s and excerpts.
  Eleven files matched. Ten are PLE **gate summaries**, not token/history/row-ID
  captures. For example `research/qwen4exp-bringup-20260829/round2-box-receipts/kvq2/tinyC.log:35`
  reports 69,635 cumulative sequence comparisons over six case families, but does
  not contain the row IDs or source tokens needed to reconstruct a real trace.
  The other match is a model-plan test log, not a trace.

**No replayable real PLE/ngram trace was found in this search scope.** This is not
an assertion that no such trace exists on a remote rig or under an unrelated name.
Do not relabel the existing fixture as real because a real gate summary mentions it.
`fixtures/ple-ngram-synthetic.json` remains synthetic; its native host-ID oracle
and source-drift pin still run in the bank Rust test target.

## New trace-dependent arithmetic replay

`python3 research/spill-c-20260919/trace-policy.py --check` regenerates and compares
`fixtures/ple-trace-policy.json` against the SHA256-bound source fixture. All six
history transitions are used, but only each step's **last chunk** is gathered, as
in native PLE. Duplicate logical IDs remain in output; physical extents deduplicate.
264-byte rows are a byte-shape fixture (66 F32 or 132 BF16 elements), not a claim
about the actual checkpoint's PLE head dimension.

| Shape, six synthetic chunks | Granularity | Unique useful B | Aligned requested B | Requested/useful | Straddles |
|---|---:|---:|---:|---:|---:|
| packed | 512 | 5,016 | 13,824 | 2.755981 | 12 |
| packed | 4,096 | 5,016 | 49,152 | 9.799043 | 0 |
| packed | 16,384 | 5,016 | 98,304 | 19.598086 | 0 |
| sparse, 32,768-byte stride | 512 | 5,016 | 9,728 | 1.939394 | 0 |
| sparse, 32,768-byte stride | 4,096 | 5,016 | 77,824 | 15.515152 | 0 |
| sparse, 32,768-byte stride | 16,384 | 5,016 | 311,296 | 62.060606 | 0 |

Keep the portable 512-byte requested-granularity /4 KiB slot policy: it requests
least overfetch in both trace shapes. This is **arithmetic, not SSD IOPS, physical
traffic, latency, or a hardware/default promotion**. A's framed object backend
has 4 KiB headers/padding and may read/verify other chunks during lookup/lease;
`ObjectReader.io_bytes` counts submitted framed chunks, still not all validation
traffic. Therefore do not compare `RowReadPlan.io_bytes` with device physical
traffic or use this table to claim 512-byte O_DIRECT success. Native source
alignment and complete storage-to-compute gates still determine real policy.
No environment flag or format/numeric change was introduced.
