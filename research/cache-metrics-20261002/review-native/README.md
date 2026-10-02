# Review fixes and affected native checks

The final production fixes use the same `gdn_dspark_compatible` predicate in the
synchronous prime, cooperative prime schedule and existing oracle. Non-GDN cold
priming keeps its prior target program. HTTP observations bind the successful
`lane_for_tenant` result, including batch credentials defaulting to harvest.

Build `83f9aceee09c` passes nine selected prime-schedule tests, the real HTTP
batch-lane regression, the combined background accounting regression, Clippy and
the build. The exact binary hashes are in `../review-fixes/binaries.sha256`.

Native cell `46467bbee82f` passes all 24 short-prime HTTP comparisons, K=1..8 on
the exact failing prompt, the plain lifecycle arm and a real authenticated batch
request. Both HTTP outcomes and worker histograms report harvest, with no
interactive success. The cell retained one failure: an added check compared two
separate live metrics JSON snapshots byte-for-byte. Such observations can change
between requests and do not form a fixed-input serialization oracle.

The JSON builder itself is unchanged byte-for-byte from C904; its source hashes
are in `json-compatibility.json`. The collector now preserves both raw observations
and checks the actual negotiated JSON contract without requiring their metric
values to be frozen. All original counter, histogram, auth and bounded-label
assertions remain. The MTP-only follow-up `6786d05d9968` passes completely. The
original failed attempt is retained. No native result is replaced by a CPU claim.

B independently reviewed both fixes and closed the two P2 findings after inspecting
the CPU results and frozen source hashes. Final published-source identity and CI
remain separate requirements.
