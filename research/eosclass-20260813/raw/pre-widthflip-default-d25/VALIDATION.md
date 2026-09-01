# Validation — controlled default-B1 width transition

- Runner exit: `client.exit=0`; 105 admitted and completed requests, zero admission defers,
  zero OOM parks, zero evictions, and no protocol failures. Every transition-cell request
  was a full 4,860-token prefix-cache hit.
- The sole `failure-signature-scan.log` match is the startup configuration line listing the
  GPU watcher's fatal Xid set. It is not an observed Xid or runtime failure.
- The trace-free sweep used the same pre-fix binary as both restore-mix arms, SHA-256
  `51eb0636b2e15810a203d07c0d5a835c736777d06b715facdb8d7f3cf089c31b`, with B1FAST at
  its default. Four entries were seeded serially from identical prompt ids. The 60-token
  solo restored-hit control produced `200ec271...`.
- One target restored hit started first; three already-restored peers joined after each
  delay from 0 through 600 ms in 25-ms increments. The target produced five distinct hashes.
  At 50 ms and 225 ms it reproduced the exact historical failure: HTTP 200,
  `finish_reason=stop`, 11 completion tokens, and
  `ddbbcf35ae93821874be64e15f63feecb2471bbe6ea0c23b63fa00b1e98a9b73`.
- Because target and peers all restore serially produced state and only peer-arrival delay
  changes, this removes prime bytes, cache eviction, cold-prefill geometry, and restored-vs-cold
  state as discriminants. The failure is selected by the live solo-to-batched decode-program
  transition. The paired B1-off restore-mix control keeps restored output one-hash across N=5.
