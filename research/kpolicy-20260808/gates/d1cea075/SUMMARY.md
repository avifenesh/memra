# K-policy gate summary

- Host: <private-host-redacted>
- Commit: d1cea0757c171fb9af1bfd0f0440b59ca316152c
- Script-detected failures: 0

## Policy decisions
policy-c4-server.log:[spec-k] model="q9" tenant="" K=3 source=cold-short prompt=226 cached=0 lcp=0 active=1 placement=single-or-non-pp2
policy-c4-server.log:[spec-k] model="q9" tenant="" K=3 source=cold-short prompt=226 cached=0 lcp=0 active=2 placement=single-or-non-pp2
policy-c4-server.log:[spec-k] model="q9" tenant="" K=0 source=concurrency prompt=226 cached=0 lcp=0 active=3 placement=single-or-non-pp2
policy-c4-server.log:[spec-k] model="q9" tenant="" K=0 source=concurrency prompt=226 cached=0 lcp=0 active=4 placement=single-or-non-pp2
policy-pin-pp2-server.log:[spec-k] model="q9" tenant="kpolicy-q9-cold-short-k3-r1" K=3 source=operator-pin prompt=28 cached=0 lcp=0 active=1 placement=pp2-cross-device
policy-pp2-server.log:[spec-k] model="q9" tenant="kpolicy-q9-cold-short-k0-r1" K=0 source=pp2-placement prompt=28 cached=0 lcp=0 active=1 placement=pp2-cross-device
policy-table-server.log:[spec-k] model="q9" tenant="kpolicy-q9-cold-short-k3-r1" K=3 source=cold-short prompt=28 cached=0 lcp=0 active=1 placement=single-or-non-pp2
policy-table-server.log:[spec-k] model="q9" tenant="kpolicy-q9-cold-long-k3-r1" K=3 source=cold-long prompt=5411 cached=0 lcp=0 active=1 placement=single-or-non-pp2
policy-table-server.log:[spec-k] model="q9" tenant="kpolicy-q9-cached-long-k2-r1" K=3 source=cold-long prompt=5411 cached=0 lcp=0 active=1 placement=single-or-non-pp2
policy-table-server.log:[spec-k] model="q9" tenant="kpolicy-q9-cached-long-k2-r1" K=2 source=cached-long prompt=5497 cached=5478 lcp=0 active=1 placement=single-or-non-pp2

## Gate verdicts
PASS: run-spec K=1..8
PASS: policy-short response selected K=3
PASS: policy-cold-long response selected K=3
PASS: policy-cached-long response selected K=2
PASS: policy log K=3 source=cold-short
PASS: policy log K=3 source=cold-long
PASS: policy log K=2 source=cached-long
PASS: policy-pp2-short response selected K=0
PASS: PP-2 resolves to K=0
PASS: PP-2 automatic row stayed plain
PASS: policy-pin-pp2-short response selected K=3
PASS: MEMRA_SPEC_K pin overrides PP-2 automatic K=0
PASS: policy-c4 load completed
PASS: c=4 first wave admitted positive-K requests
PASS: c=4 overflow arrivals resolved to K=0
PASS: c=4 live-session demotion preserved
accept-gate: 1 pass, 0 fail, 0 unpinned, 0 skip
PASS: accept-gate
serve-smoke: 0 failed
PASS: tools/serve-smoke.sh
