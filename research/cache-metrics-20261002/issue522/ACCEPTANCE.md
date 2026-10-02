# #522 acceptance and rig-only scope

The 2026-10-02 work is restricted to the local rig. PR #916 completes the cache
regression, the #918 short-prime fix, and the local/shared metrics scope below.
#522 remains open for dedicated DSv4 integration and native acceptance. No hardware
question, rental, production access or off-rig work is required for today's PR.

| Property | Today's scope | Deferred #522 scope |
| --- | --- | --- |
| Queue, TTFT, E2E and emitted intervals | Hybrid model/lane histograms, cumulative and unsampled | Actual DSv4 emitter/route clock integration and native timing receipt |
| Token distributions and cache credit | Hybrid prompt/completion histograms and late in-flight cache credit | DSv4 route usage/cache parity |
| HTTP outcomes | Shared real HTTP/SSE contracts for four APIs; status/error/refusal/deadline/cancel/truncation counters, including HTTP 200 errors | No DSv4 native qualification inferred from shared HTTP tests |
| Standard exposition | Auth, bounded labels, actual promtool and JSON compatibility on local native traffic | Dedicated-route scrape and lifecycle evidence |
| KV/device gauges | Hybrid live/parked canonical state planes, reservation-aware capacity, active/queued state, cached device readings with age and worker-failure cleanup | DSv4 allocation leases and native release observations |
| Prefix inventory | Existing hybrid counters, explicit backend/tier exposition and closed-form cache gate | DSv4 host lookup/insert/eviction/byte hooks and native restore cell |
| Greedy identity | #918 before/after evidence: 24 HTTP comparisons, K=1..8, longer-prime/cache controls on the two local Qwen artifacts | No broad support promotion |
| API integration | Preserve C's background/image changes and prove acknowledgement versus eventual generation accounting after rebase | No repeated long-deadline campaign when its behavior is unchanged |

The prepared DSv4-specific source and full-scope exploratory evidence were banked
on the local `deferred/dsv4-metrics-20261002` branch. They are excluded from today's
merge. Their future native cell remains named in that branch's DEFERRED.md; its
requirements have not been weakened or relabelled as CPU evidence.

The previous scoped raw receipt, including its original plain/MTP failure, remains
in this directory. The numerical cause and passing controls are in
`research/qwen9b-greedy-identity-20261002/`. The reconciled rig receipts are under `../combined-native/`, `../combined-background/`
and `../review-native/`, with original source/binary identities retained. They include
actual background acknowledgement/terminal accounting and repeated-read checks.
#522 remains partial because dedicated-route native integration is deferred.

The two lifecycle source-patch fields have an explicit metadata correction in
`../review-native/PROVENANCE-AMENDMENT.md`. Original captures and driver hashes are preserved; no new native execution is implied.
