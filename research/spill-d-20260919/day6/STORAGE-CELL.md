# A's storage-cell dispatcher — task 1(d) follow-up

The final re-fetch found A had advanced to
`0e7b4456b33935917136455ffaa9214d95db6329`, after the initial no-fragment check.
This supersedes the carry-forward disposition in COLLECTOR.md.

Imported `research/spill-a-20260919/storage_capture.py` **byte-for-byte** from that
revision; applied the logic of `day5/battery-dispatch.patch` to D's newer CLI.
The original fragment is retained at `day6/a-storage-dispatch.original.patch`.
One strengthening: the dispatcher calls current `validate_cell` first so A's
older helper cannot bypass newer UTC/storage/power/inherited-lock integrity.

Explicit invocation only:

```sh
python3 tools/tier-battery.py --validate /path/to/CELL.jsonl \
  --schema storage-cell --out /new/path/joined.jsonl
# Optional --storage-samples envelopes.jsonl must match exact captured run id/sample.
```

It checks the exact storage-bench command, successful capture, raw hashes,
monotonic window, canonical lock, raw sample versus requested bytes/backend, and
optional run-id envelope. Empty diagnostic GPU CSVs remain empty/unknown;
physical counters, error/fallback statuses and qualification=false survive.
No automatic schema promotion, forced-pair gate or performance/serving claim.

Three dedicated regression tests passed (raw `storage-cell/python.log`), covering
archived sample/nulls, immutable output, foreign/duplicate/missing/altered
samples, command/capture/raw-hash/duration/UTC/path refusals, start-only journal
and strict runs-schema rejection. A real **archived** 264-byte storage capture
joined successfully; output/log are retained under `storage-cell/`. This is not
a new storage/GPU run. A's implementation files outside the one supplied helper
were not imported or edited.
