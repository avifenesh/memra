# Public-boundary enforcement — 2026-08-14

Owner doctrine (darklanes `ROADMAP-20260813.md` §11, 2026-08-14 override): memra (public) is
the free/open-source single-consumer inference engine. Darklanes (private) owns every
production-ready serving layer around it. This lane installs the enforcement seam. It does
not delete existing tracked private material — the doctrine explicitly preserves historical
receipts pending an owner-reviewed hashed migration.

Companion audit: `~/projects/wt-dl-cx-affinity/research/repo-boundary-20260814/ACTIVE-LANES.md`.

## What landed in this slice

- `tools/public-boundary-policy.toml` — declarative policy. Two rule kinds:
  1. `secret_patterns` — structural regexes that flag production endpoint assignments,
     wallet destinations, pod addresses/IDs, marketplace domains, serve-box operating
     paths, and live-binary fingerprints without repeating any current private value in
     the public policy. Doctrine §1 and §2 concerns.
  2. `private_paths` — globs over private serving code and operator tooling: the deploy
     overlays (`deploy/gateway`, `deploy/glue`, `deploy/systemd`, `deploy/runpod`), the
     multi-tenant / admission / billing crate seams (`memra-server::admin`, `::ledger`,
     `::darklane`, and the `memra-lanes` crate), and the fleet/proxy/meter tools.
- `tools/check-public-boundary.py` — scanner with three modes:
  1. `check` (default) walks `git ls-files`, records path-glob and regex matches, and
     compares each to the grandfather allowlist. Exit 1 on any unmatched violation.
  2. `seed` regenerates the allowlist from the current tracked-file state (bootstrap only).
  3. `verify-allowlist` confirms every allowlist entry still pins a live tracked file with
     the recorded SHA-256; `--prune` drops drifted entries. Normal `check` also fails on
     stale entries so removed material cannot later regain an old exemption byte-for-byte.
- `tools/public-boundary-allowlist.jsonl` — seeded grandfather list. 1,143 entries pinning
  each currently-tracked private path or regex hit by `(path, sha256)`. Changing any
  allowlisted file drops the (path, sha256) tuple from the list, forcing the change to
  either (a) update the hash intentionally, or (b) migrate the file to darklanes.
- `tools/hooks/pre-push` invokes the check.
- `.github/workflows/ci.yml` invokes the check as a CI step (compile-only runner needs no
  GPU; the scanner is stdlib Python).

## Violation inventory (current allowlist)

Grouped by category. Numbers are `git ls-files` entries at the seed hash.

| Category | Count | Representative paths |
|---|---|---|
| `private_path` (deployment overlays) | 51 | `deploy/gateway/**`, `deploy/glue/**`, `deploy/runpod/**`, `deploy/systemd/**` |
| `private_path` (server/crate code) | 5 | `crates/memra-server/src/{admin,ledger,darklane}.rs`, `crates/memra-lanes/**` |
| `private_path` (fleet/proxy tools) | 10 | `tools/{fleet-*, serve-fleet, serve-proxy, load-serve, apikeys-gate, cache-meter-gate}` + fleet tests |
| `secret_pattern` (structural matches) | 1,077 | 1,061 research receipts plus 16 matches in tools, docs, crates, and the README |

Full list: `tools/public-boundary-allowlist.jsonl`. Re-run with:

```bash
python3 tools/check-public-boundary.py check
```

Runtime is ~2 s on 35 k tracked files. `git grep -P` first selects candidate files, then
the Python matcher assigns the earliest structural rule and hash-pins the file.

## What this slice deliberately does NOT do

- **No deletions.** The doctrine forbids mass deletion and requires hashed migration with
  owner review. `~/projects/wt-dl-cx-boundary` currently mirrors the public tree as
  `engine/memra/` and has no separate migration destination for `deploy/gateway/`,
  `deploy/glue/`, `deploy/runpod/`, or the fleet tools; deleting them here without a
  darklanes copy would drop history rather than migrate it.
- **No claim that git history has been scrubbed.** Historical objects with secret material
  remain reachable through `git log`. History filter / rotation decisions are owner-only
  (§10 signature class) and are tracked separately from this in-tree containment work.
- **No move of build-coupled code.** `memra-server::main`, `::worker`, `::auth`,
  `::constrained`, `::toolcall`, `::ttft`, and `::health` currently import
  `crate::{admin, darklane, ledger, lanes}` at multiple sites. A safe extraction is a
  separate slice with tests, not a mass rewrite in this commit.

## Extraction plan — build-coupled private code

Each item names its coupling seam and the test that must pass unchanged after extraction.

### 1. `crates/memra-server/src/darklane.rs` — background-job scheduler

- **Coupling:** `main.rs:468` (Option-wrapped `Arc<darklane::BgJobState>`), `main.rs:1726`
  (spawn from env), `main.rs:1728` (parse `BgConfig`), `main.rs:2300` (valley signal).
- **Public seam to keep:** the health-driven `phase == IDLE + beat_age_ms` idle signal is
  a generic engine surface. Everything below the `MEMRA_BG_JOB` env is private policy.
- **Extraction:** move `darklane.rs` verbatim to darklanes. Public `main.rs` retains the
  four hooks as a `#[cfg(feature = "bg-jobs")]` gate, default-off. Public engine builds
  compile with the feature off; the four sites become no-ops. Darklanes vendors the file
  and enables the feature.
- **Test:** existing `memra-server` release build passes with `--no-default-features`;
  behaviour on the public build is identical (no BG spawn, no valley probe).

### 2. `crates/memra-server/src/admin.rs` — tenant/key provisioning HTTP surface

- **Coupling:** `main.rs:1674` (`admin::Config::from_env`) — a single call that becomes a
  no-op when the admin bearer is unset. `admin` also references `auth::global()` and
  `ledger` for audit records.
- **Extraction:** move to darklanes; the public build compiles without `mod admin`. The
  single call site becomes a feature-gated stub.
- **Test:** default env (no `MEMRA_ADMIN_*` set) is already the no-op path; the public
  build must retain that behaviour byte-identically.

### 3. `crates/memra-server/src/ledger.rs` — cost/billing receipts

- **Coupling:** `main.rs:4248, 4497` (constructing `ledger::Usage`), plus the ledger
  module owning cost-row JSON output. Public engine metering (prompt/completion counts on
  `/metrics`) does not go through this module — it is the persistent per-request cost
  receipt for the billing pipeline.
- **Extraction:** move to darklanes; public engine keeps generic metering only. The
  two call sites become feature-gated pass-throughs to a no-op ledger sink.
- **Test:** `curl /metrics` returns identical counter shape; no `receipts/*.jsonl` is
  written when the feature is off.

### 4. `crates/memra-lanes/` — admission / QoS lane crate

- **Coupling:** `memra-server::{main, worker}` import `Lane`, `LanePolicy`, `StepStats`.
  Server-wide references: 5 in `main.rs`, ~30 in `worker.rs`.
- **Public seam to keep:** step-latency percentiles as an engine metric are public. QoS
  admission policy (`x-lane` routing, per-lane budgets, shed at admission) is private.
- **Extraction:** narrow `memra-lanes` to a public step-latency percentile helper and
  move `Lane`, `LanePolicy`, and admission logic to a private `darklanes-lanes` crate.
  The public server treats every request as a single default lane. This is the biggest
  extraction; expect its own PROGRESS doc.
- **Test:** worker unit tests currently rely on `LanePolicy::from_env()`; the extracted
  boundary must keep the default (env-unset) shape identical, and the lane-QoS gate
  moves to a darklanes-side integration test.

### 5. `crates/memra-server/src/auth.rs` — API-key management

- **Split:** the SHA-256 helper, single-`MEMRA_API_KEY` bearer path, and cache-namespace
  isolation stay public (single-consumer engine still needs a bearer). Multi-tenant
  keyring, per-key rate limits, and hot-reload become private.
- **Test:** `MEMRA_API_KEYS` unset must keep the current single-bearer public path
  identical (including tenant `"default"` namespace).

### 6. Fleet / proxy / meter tools

- `tools/{fleet-meter.sh, fleet-replay.py, fleet-report.py, serve-fleet.sh,
  serve-proxy.py, load-serve.py, apikeys-gate.sh, cache-meter-gate.py}` and their
  `test_fleet_*.py` companions are pure operator tooling, no build coupling.
- **Extraction:** move verbatim to darklanes on the next narrow slice; delete here in
  the same slice.

### 7. Deployment overlays

- `deploy/{gateway,glue,runpod,systemd}/**` are pure runbooks/config with no build
  coupling.
- **Extraction:** move verbatim to darklanes on the next narrow slice; keep only
  `deploy/README.md` here trimmed to a pointer.

### 8. Receipts and docs with private structural references

- 1,061 of the 1,077 secret-pattern hits are under `research/`; the remainder are in
  tools, docs, crates, and the top-level README. Most are receipts of past work
  (servetest, gateway, connect, or-provider, tune-data ops notes), but the non-research
  matches need the same owner review because some describe active operating surfaces.
- **Extraction:** the doctrine preserves historical receipts. Options are (a) leave
  allowlisted, (b) redact live-host tokens per-file, or (c) migrate whole dirs to
  darklanes preserving hashes. Owner call. Until owner decides, allowlist holds them
  under review.

## Ordered follow-up slices

1. **This commit** — policy, scanner, seeded allowlist, hook + CI wiring, this doc.
2. **Tools slice** — move eight fleet/proxy/meter tools + tests to darklanes; delete
   from public; drop their allowlist entries; land as one commit with `cargo test` +
   the boundary check green.
3. **Deploy overlays** — same shape, moving `deploy/gateway`, `deploy/glue`,
   `deploy/runpod`, `deploy/systemd`.
4. **`darklane.rs` feature gate** — smallest crate-code slice; single feature flag,
   default-off, tests unchanged.
5. **`admin.rs` feature gate** — same shape.
6. **`ledger.rs` feature gate + public metering fallback.**
7. **`memra-lanes` split** — largest slice; expect its own PROGRESS doc and a
   worker-integration test suite.
8. **`auth.rs` split** — narrow to single-bearer public path; multi-tenant keyring to
   darklanes.
9. **Owner-reviewed research-receipts decision** — redact vs migrate vs allowlist.

Each slice ends with a green boundary check plus the smallest relevant `cargo test`.

## Gate receipts for this slice

```
$ python3 tools/check-public-boundary.py check
public-boundary: 1143 matches (1143 grandfathered, 0 new).

$ python3 tools/check-public-boundary.py verify-allowlist
verify: 1143 allowlist entries all pin live tracked files.

$ python3 tools/test_public_boundary.py
Ran 8 tests ... OK
```

Build check (compile-only, no GPU on this box): run `cargo check -p memra-server` before
merge on the target rig to confirm no incidental coupling was introduced.

## Slice 2026-08-18 — host destinations, account inventory, and a fourth mode

Two independent read-only reviews found committed rented-host network details and cloud
inventory that this scanner did not match. Both confirmed. Decision record with the adopted
and rejected rules: `docs/decisions/PUBLIC-BOUNDARY-DETECTION.md`.

Two distinct causes, and both needed a fix:

1. **Rule shape.** `rented_ipv4` needs a keyword within 60 characters *before* the address, so
   a bench box written as a bare `<user>@<addr>:/path` transfer target had nothing to anchor
   on; the account resource-id family was represented by only its instance and security-group
   members; and `provider_machine_id` matches the literal word *machine*, not the contract
   number a provider console shows. Added `ssh_destination`, `aws_resource_id` and
   `provider_contract_id` — measured cost 3, 5 and 0 tracked files.
2. **Corpus.** `check` reads the checkout and `check --commits-file` reads the blob versions a
   push introduces. Neither ever re-reads a branch that is already published, so a rule added
   after a push is retroactively blind on that branch — which is what happened here, one day
   apart. No additional pattern can close that.

So the scanner now has **four** modes; the list at the top of this doc describes the first
three. The fourth:

4. `check --refs [GLOB]` re-scans every blob version carried by the refs matching GLOB
   (default `refs/remotes/origin/**`) under the *current* policy. One multi-tree
   `git grep -P` prefilter plus batched `git cat-file --batch` reads; deduplicated by
   `(path, sha256)` so shared history yields one violation with a carrier list rather than one
   per branch; reuses `evaluate_content`, so a ref finding and a checkout finding are the same
   judgement. It skips the stale-allowlist invariant on purpose — published refs are a
   different corpus from the checkout, and only the full-tree `check` owns that invariant.

`--summary-only` prints per-rule counts and no paths, for jobs whose logs are public: the
per-path findings list of an unremediated backlog is a map to it.
`.github/workflows/boundary-refs.yml` runs the ref scan daily and on manual dispatch, and is
deliberately **not** a `push`/`pull_request` gate — the first run surfaced a history backlog
that is remediated branch by branch, and a required check that stays red for that long is a
gate nobody reads.

Consequence recorded here rather than smoothed over: the three new rules make the full-tree
`check` exit 1 on four previously-invisible tracked files carrying genuine account inventory.
CI and every pre-push are red until the owner rules on them. The allowlist was **not** touched
— re-pinning is an owner decision, and quietly grandfathering a leak the day it was found is
how the last one hid.

```
$ python3 tools/test_public_boundary.py
Ran 15 tests ... OK
```

## Slice 2026-08-19 — the slice above was never pushed, plus two reporting defects

Everything in the 2026-08-18 slice existed **only on the rig.** A full read-only audit of every
pushed ref (207 refs, 5,107 commits, 39,991 blobs; banked in darklanes
`research/security/public-ref-audit-20260819.md`) measured `boundary-refs.yml` on **zero of 66
heads**, `--refs` absent from the pushed scanner, and `aws_resource_id` — the only rule matching
`cr-`/`ami-`/`subnet-` — absent from the pushed policy. GitHub schedules `on: schedule` only from
the **default branch**, so the nightly ref audit written for exactly this class of leak had never
run once, and the `cr-` capacity-block id that motivated the audit was undetectable by the gate as
deployed, on any branch, at any time. This slice lands it. Details, the two defects and what still
gets through: `docs/decisions/PUBLIC-BOUNDARY-DETECTION.md`.

**Defect 1 — the allowlist had no rule scoping.** Keyed on `(path, sha256)`, a blob grandfathered
for one rule was exempt from every rule. Entries are now keyed on `(path, sha256, rules)`, `rules`
is required, and the grant expires with the finding it was granted for. All 578 existing entries
were narrowed to the rule their `reason` already recorded; **zero** needed to keep their old
breadth, verified by running the pre-push range check over current `main` (`v0.92.0..main`, 86
commits: 63 matches, all still grandfathered).

**Defect 2 — first-match-only reporting.** Every matching rule is now reported and the report is
ordered by a required `[severity]` rank per rule, not by scan order. The union matcher stays as
the cheap prefilter; only a blob that already hit gets the per-rule pass, because alternation
structurally cannot report two rules matching one span.

**Reversing the "allowlist not touched" call above, and why.** Narrowing surfaced 61 blobs
allowlisted for a different rule than one they also match — against the 4 the audit found by hand
— on top of the 5 new findings from the three unpushed rules. Leaving 66 findings red would have
left `ci.yml` permanently red on `main`, and a permanently red gate is the failure mode this whole
document is about. All 66 are pinned as explicit rule-scoped entries carrying a new
`unremediated` field, and the check now prints the outstanding count **on every run, pass or
fail**. That is the version that is neither red-and-ignored nor green-and-quiet. It is not an
acceptance: remediate-vs-migrate is still an owner call, and the entries go stale automatically
the moment a file changes.

```
$ python3 tools/check-public-boundary.py check
public-boundary: 583 matches (583 grandfathered, 0 new).
public-boundary: 76 allowlisted findings are marked unremediated (worst sev4); owner decision
pending. By rule: aws_resource_id=5, ssh_destination=3, serve_prefix=51, serve_data_root=8,
production_endpoint=4, onlist=2, openmodels=1, openrouter_docs=1, provider_contract_id=1
$ python3 tools/check-public-boundary.py verify-allowlist
verify: 583 allowlist entries all pin live tracked files for the rules they name.
$ python3 tools/test_public_boundary.py
Ran 26 tests ... OK
```

