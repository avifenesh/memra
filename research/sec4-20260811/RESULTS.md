# cx-sec4 results

## Final verdict

**PASS.** Both sec2 serving-safety regressions are fixed, their focused regressions pass, and
the required `memra-server` CPU test gate is green.

## ADSD baseline hardening

- The rolling 64-request model history now retains tenant identity and derives a comparator only
  from other-tenant rows. The suspect tenant cannot lower the baseline it is measured against.
- Baseline eligibility (16 samples and 512 drafted tokens) is checked after excluding the suspect
  tenant.
- The detector now uses the pooled two-proportion standard error, accounting for sampling error in
  both the other-tenant baseline and the eight-request tenant window.
- Detection remains observational only: verification, scheduling, caches, and rate limits are
  unchanged.

Regression evidence:

- Before the fix, `adsd_detector_stays_latched_during_boiling_frog_collapse` failed because the
  incident rearmed while the attack was still active.
- After the fix, all three ADSD tests pass: synthetic collapse emits exactly once, ordinary noise
  never emits, and the slow sustained collapse stays latched.

## Metrics side-channel hardening

- `MetricsScope::All` is now the operator view: a dedicated `MEMRA_METRICS_TOKEN`, or the existing
  unauthenticated no-key loopback development mode.
- Static and keyring completion credentials receive safe cumulative counters plus their permitted
  tenant token/ADSD rows, but not current cache/pool occupancy, active or queued work, CUDA memory,
  idle/background state, aggregate per-model speculation data, or last batch size.
- The legacy static completion key still sees all raw `cache_salt` rows in its single completion
  domain; it no longer implicitly becomes an operator scrape principal.
- `docs/SERVING.md` now documents this visibility boundary and the corrected ADSD comparator and
  statistic.

Regression evidence:

- Before the fix, the tenant-scope regression failed on exposed `active_sessions`.
- After the fix, six focused metrics/auth tests pass, including tenant omission, operator-token
  visibility, static completion-key restriction, and unchanged full no-key loopback visibility.

## Gate evidence

- `cargo test -p memra-server adsd_detector -- --nocapture`: 3 passed, 0 failed.
- `cargo test -p memra-server metrics_ -- --nocapture`: 6 passed, 0 failed.
- `cargo test -p memra-server`: 174 passed, 0 failed, 0 ignored.
- `git diff --check ae558f55188d5aa93787481a0f7ea3182ce3b49f..HEAD`: clean.
- No GPU runtime or benchmark was used. `cargo fmt` was not run.

No merge, tag, release, serving-default change, or performance-board change was performed in this
lane.

## Hermes follow-up status (review 2026-08-11, kimi 10:05 UTC)

The review findings were non-blocking. sec5 closure state and the remaining documented limitation
are recorded inline.

- **Closed in sec5: single-tenant ADSD fallback** (`ad08837f`). An eligible other-tenant baseline
  remains preferred. When it is absent, the detector compares the eight-request signal window with
  non-overlapping older rows from that tenant, under the same 16-sample/512-drafted-token floor. A
  latched historical comparator stays frozen until recovery, so sustained collapse traffic cannot
  rearm by diluting its own history. The single-tenant synthetic-collapse regression now emits one
  latched incident.

- **[adsd-suspect] is not attack-confirmed — acceptance is content-shaped** (`b13061d9`, arXiv
  2605.30580 "Speculative Decoding and the Curse of Multilinguality"). Speculative acceptance varies
  strongly by output language with the identical draft/verify pair; a tenant whose traffic shifts
  toward low-resource languages can legitimately produce the `>=0.20` deficit at `z<=-3.0`
  signature. Detection-only posture caps the blast radius (no auto scheduling/rate action), but the
  operator playbook must read [adsd-suspect] as "investigate the traffic mix", not "attack
  confirmed". sec5 offer: cite the paper in the ADSD code comment + SERVING.md, and consider
  stratifying the baseline by a cheap content/language signal.

- **Closed in sec5: global LCP/prefix aggregate visibility** (`38bca65c`). `lcp_histogram`, the
  global `prefix_cache_hits/misses/inserts/evictions/hit_tokens`, and the global
  `cache_hit_token_ratio` now require operator scope. Completion credentials retain safe base
  counters and only their permitted `tenants` rows, including that tenant's own
  `cache_hit_token_ratio`. The metrics-scope regression proves tenant omission, operator
  visibility, and preservation of the tenant receipt together.
