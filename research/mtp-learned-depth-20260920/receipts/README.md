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

The measurement-time text matcher produced five accidental
`provider_name_aws` matches after decoding compressed bytes with invalid
UTF-8 removed. No expanded member matched. The historical exception file
and `verify_archive_exceptions.py` reproduce that finding without changing
the archive bytes.

Current main applies the raw-byte prefilter consistently in checkout,
commit, and ref scans. Two of the five compressed archives still match
`provider_name_aws` in raw bytes. They retain exact-hash, single-rule
public-boundary entries backed by the zero-match expanded-member review.
The other three decoded-only findings need no current allowlist entry.
This publication keeps the historical review record and uses current
main's scanner for admission.
