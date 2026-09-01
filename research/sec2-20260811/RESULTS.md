# cx-sec2 serving hardening — results

Verdict: **PASS**. Both confirmed serving-hardening cells are fixed on `lane/cx-sec2`.
The required CPU-only package gate passes. No GPU run, model load, performance claim, merge,
tag, push, release, automatic tenant throttling, or session eviction was performed.

## Findings closed

| cell | result |
|---|---|
| Metrics tenant isolation | `MEMRA_METRICS_TOKEN`, when configured, is exclusive for `/metrics` and `/yield/metrics`; a valid completion key receives 403. The scrape token sees all tenant rows. With completion keys only, global counters remain visible while a keyring bearer sees only its own `tenants` row and `adsd_suspect_total` entry. No-key loopback development and the legacy single-key tenancy domain remain unchanged. |
| ADSD acceptance collapse | Each retired speculative request feeds a bounded eight-request `(model, tenant)` window and a rolling 64-request model baseline. After baseline warmup, a tenant deficit of at least 0.20 with one-sided z-score at most -3.0 for three consecutive observations logs `[adsd-suspect]` and increments the tenant's latched `adsd_suspect_total`. Recovery within 0.10 of baseline rearms the detector. Verification decisions, scheduling, cache, routing, and rate-limit behavior are unchanged. |

The metrics counter follows the same tenant visibility policy as token rows. Operators can use a
dedicated scrape principal for fleet-wide visibility or completion credentials for tenant-scoped
visibility. ADSD response remains manual: inspect the request `usage.spec` and log evidence, then
apply the existing tenant/lane rate limit if warranted.

## Verification

| gate | result |
|---|---|
| `cargo test -p memra-server` | **PASS**, 172 passed, 0 failed |
| Focused metrics regressions | **PASS**, 5 passed: multi-key tenant scope, all-tenant scrape-token view, completion-key 403 under exclusive token, protected public override, and unchanged no-key loopback development |
| Synthetic ADSD streams | **PASS**, sustained collapse emitted exactly one latched incident; ordinary acceptance noise emitted none |
| `python3 tools/update-perf-board.py --check` | **PASS**, generated perf surfaces current |
| `bash tools/check-flags.sh` | **PASS**, no new drift beyond the frozen known set |
| `git diff --check` | **PASS** |

Implementation commits: `3cdb4553` (metrics isolation), `cc53ee8c` (ADSD detector), and
`5654c86f` (operator policy documentation). The lane ledger was created first in `d8202955`.
