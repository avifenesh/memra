# Archive boundary review

The repository scanner matched `provider_name_aws` in five compressed archives.
The archive bytes, token tapes and outputs remain unchanged.

`verify_archives.py` checks every expanded record against every current secret
rule. It also checks the complete runtime source archive at its original source
paths, applying the repository's current path policy and exact-hash, rule-scoped
source entries.

The record archives contain one reviewed textual match: a generated answer in
the cold Gemma calibration suggests comparing two hypothetical machine types.
It contains no deployment or account identity. The member, its exact SHA-256
and its single allowed rule are pinned in `RECORD_RULE_PINS`. Any changed bytes
or another matching rule fail the review.

The runtime source archive carries twelve source blobs covered by current
repository entries and three policy-bypass files. Their paths, hashes and
covered rules are listed in `receipts/boundary-verification.json`; an expired or
insufficient source entry is rejected. Generated-key fixtures retain the same
reviewed bytes as the source tree.

The five outer archive entries in `tools/public-boundary-allowlist.jsonl` cover
only `provider_name_aws` and only the recorded SHA-256. They grant no exception
for any other rule or a future archive version. Re-run:

```sh
python3 research/mtp-continuing-session-20260921/verify_archives.py --check
python3 tools/check-public-boundary.py check
```

The first command verifies the six required archive paths, archive hashes and
sizes, exact member lists and hashes, and the recorded file counts. It reproduces
and compares the committed expanded-file evidence without rewriting it. The
second applies the
normal repository publication policy, including the outer archive hashes.
