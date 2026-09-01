# Validation — trace-free B1-off restore mix

- Runner exit: `client.exit=0`; 21 admitted and completed requests, zero admission defers,
  zero OOM parks, and no protocol failures.
- The sole `failure-signature-scan.log` match is the startup configuration line listing the
  GPU watcher's fatal Xid set. It is not an observed Xid or runtime failure.
- The one-variable control used the same server binary as the default arm, SHA-256
  `51eb0636b2e15810a203d07c0d5a835c736777d06b715facdb8d7f3cf089c31b`, with
  `MEMRA_SERVE_B1FAST=0`; diagnostics remained off.
- The serial seed and all five full 4,860-token restored hits produced exactly one 60-token
  hash, `5790654979cb98bfacf6d3593b6a5d3def7a5f4bd2a1b8b65e4a6fabe1a72f66`.
  The paired default-B1 arm produced two restored-hit hashes under the same N=5 shape.
- Cold peers produced 14 instances of `200ec271...` and one instance of `57906549...`.
  That is separate prime-composition drift and is not attributed to B1FAST. The controlled
  result is that a state restored from the serial seed remains on its source's one output
  trajectory when live decode width changes.
