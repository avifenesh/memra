# Immutable raw receipts

`archives/` holds all 22 original compressed bundles, with exact SHA-256 and byte
counts in `manifest.json`. These contain the individual commands, logs, GPU
telemetry, prompt/output token tapes, decoded text, turn tables and round tables.
Compression preserves all bytes and keeps the GitHub review navigable.

All 4,542 expanded regular-file contents were checked against every current secret
pattern: zero matches. The policy/scanner are unchanged at the publication base.
The result is recorded in `metadata/expanded-boundary-scan.json`. Archived bytes
are verified again before extraction.

From the parent research directory, use Python 3.12+:

```sh
python3 unpack_receipts.py --destination receipts-expanded
python3 analyze.py --family qwen --ledger qwen-selected-sets.json --receipts receipts-expanded --output qwen-audit.json
python3 analyze.py --family gemma --ledger gemma-selected-sets.json --receipts receipts-expanded --output gemma-audit.json
```

The destination must be new. The unpacker rejects hash mismatches, path traversal,
links and special files before writing. `EXCLUSIONS.md` explains rejected data;
the two selection ledgers, not a glob of all runs, determine the scored rows.
`metadata/` contains the captured build, test, audit and controller records.

Request-position diagnostics (after extraction):

```sh
python3 diagnose_requests.py --family qwen --ledger qwen-selected-sets.json --receipts receipts-expanded --output qwen-request-diagnostics.json
python3 diagnose_requests.py --family gemma --ledger gemma-selected-sets.json --receipts receipts-expanded --output gemma-request-diagnostics.json
```

Five archives have accidental `provider_name_aws` matches in their compressed bytes,
with zero matches in any expanded member. The public-boundary exceptions pin only
those exact archive hashes and that single rule; changed bytes and other rules
remain checked. Reproduce the evidence with `python3 verify_archive_exceptions.py`.
The original archives were not recompressed or rewritten to avoid the check.

The pinned-file coverage fix makes both the content check and drift check run the
full matcher on explicitly pinned paths, even if the fast byte-level prefilter
misses a match after UTF-8 normalization. It changes no policy pattern and grants
no additional exemption; ungranted rules still fail. Two regression tests cover
that consistency and rejection behavior.
