# Archive boundary review

The repository scanner matched `provider_name_aws` in five compressed archives.
The archive bytes, token tapes and outputs remain unchanged.

`verify_archives.py` reproduces the archive review using the scanner and policy
already inside the hash-pinned measurement source archive. Both file hashes and
the twelve reviewed source-rule decisions are frozen in
`boundary-source-rules.json`. The archive hash is checked against the fixed
measurement hash before any archived scanner code is loaded. This avoids
coupling an immutable source snapshot to later edits of live source files,
policy or allowlist entries.

The record archives contain one reviewed textual match: a generated answer in
the cold Gemma calibration suggests comparing two hypothetical machine types.
It contains no deployment or account identity. The member, its exact SHA-256
and its single allowed rule are pinned in `RECORD_RULE_PINS`. Any changed bytes
or another matching rule fail the review.

The runtime source archive carries twelve source blobs whose existing
repository decisions were reviewed at publication, and three policy-bypass
files in the archived policy. Their paths, hashes and covered rules are listed
in the frozen source-rule file and `receipts/boundary-verification.json`.
Changed bytes or a rule outside those decisions are rejected. Generated-key
fixtures retain the same reviewed bytes as the original source tree.

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
second applies the live repository publication policy, including the outer
archive hashes. The historical expanded-source review and the live publication
check are separate checks.
