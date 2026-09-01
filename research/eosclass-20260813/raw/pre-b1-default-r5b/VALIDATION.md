# Validation — trace-free default-B1 restore mix

- Runner exit: `client.exit=0`; 21 admitted and completed requests, zero admission defers,
  zero OOM parks, and no protocol failures.
- The sole `failure-signature-scan.log` match is the startup configuration line listing the
  GPU watcher's fatal Xid set. It is not an observed Xid or runtime failure.
- The serial seed and all 15 cold peers produced the baseline hash
  `200ec271e8c0eb57fb6b7d42d3ed53e4590c5e72f0303b5ef3c74d363eab88e7` at 60/60 tokens.
- All five full 4,860-token restored hits reached 60/60, so this N=5 arm did not reproduce
  early EOS. Hits 1--4 produced the baseline hash; hit 5 produced
  `988d1e40077d6ec1c5ca95d079c3bc6f83312a1db71ef61ff40651ed6aacd2a6` at the same full
  length. The trace-free arm therefore establishes load-history-dependent Q27 output drift,
  not an EOS rate estimate or width attribution by itself.
- Diagnostics were disabled (`eosclass_trace=0`). The exact server binary SHA-256 was
  `51eb0636b2e15810a203d07c0d5a835c736777d06b715facdb8d7f3cf089c31b`.
