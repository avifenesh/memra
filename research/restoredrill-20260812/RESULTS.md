# cx-restoredrill results

Status: **PASS — durable-restore checklist row closed for the specified same-box engine drill.**

On 2026-08-12, an isolated restore reconstructed the native Q27 + Q35 serve shape on Vast
instance `47529373` under `/scratch/restore-drill`, started a second `memra-server` only on
`127.0.0.1:8004`, passed the 21-check protocol/accounting battery for each model, and stopped that
process. The public `api.tiyuvta.ai` pair remained on PID `15511` and returned 72/72 successful
monitor requests across the before/during/after samples.

This is an explicit **PASS** against `deploy/gateway/APPLICATION-CHECKLIST.md`'s
“Restart/restore drill and off-host manifest restore are captured” row. The claim is bounded to
the requested same-box, isolated engine restore. It does not certify replacement-host ingress,
Cloudflare/relay secret recovery, hardened systemd installation, durable request-ledger recovery,
public cutover, or a replacement-host soak.

## Restored boundary

The restore used only content-free off-host inputs plus the explicitly allowed, hash-verified
same-box caches:

- public GitHub clone at runtime revision
  `b227063f896d261a27922c4aa814b89802080c1f`;
- servetest harness extracted from separately pinned revision
  `d2fba620031920032b253b700443af5ef1ec7866`;
- installed 53,814,712-byte server re-verified as
  `68f72cb3bb284270ab9a84770478574976901905430dbbe0b264c4840e2071cf`;
- Q27, 15,705,920,064 bytes,
  `d8d71c7e8a01a1c964fd904a7b496eaef19bdd66827e0949e66c723da742d517`;
- Q35, 18,209,036,576 bytes,
  `df27a780435b7b45c2597536112ea3cb091f8544c3d0c3318d9f4258b31f7adf`;
- complete two-alias metadata template, SHA-256
  `9a0db030a4283c97179df71ef67fa2885c86c81c70cca45587c03d1572a2de7a`; and
- freshly generated `servetest` key capped at four admissions plus a fresh metrics token.

The model stage used `cp --reflink=never`, then independently verified source and destination
sizes and full SHA-256 values. The generated keyring, API key, and metrics token were mode 0600.
Only their fingerprints were retained; the files were destroyed after the isolated process
stopped and before the scratch root was removed. Neither the live keyring nor its plaintext API
or metrics material was copied into the drill.

The restored runtime shape was the current pair shape: both exact model IDs, plain decode,
context 8,192, 4,096 MiB prefix cache, prefix dedup on, reuse pool and affinity off, session
ceiling 96, one hashed tenant with cap four, and a two-entry metadata TOML. The metadata contains
no invented Q35 price, capacity, readiness, or product claim.

## Timed restore

Durations are wall-clock single-run measurements on the live Vast box. The 131.632-second
end-to-end duration includes the before and during public safety samples and orchestration gaps;
it is not a throughput benchmark.

| Stage | Duration | Result |
| --- | ---: | --- |
| Off-host source fetch + runtime/harness verification | 7.052 s | PASS |
| Exact installed-binary re-verification and isolated copy | 0.111 s | PASS |
| Two-model source verification, non-reflinked copy, destination verification | 67.848 s | PASS |
| Metadata/config render + fresh keyring/metrics material | 0.068 s | PASS |
| VRAM gate, two-model load, and loopback readiness | 14.182 s | PASS |
| Q27 21-check gate + Q35 21-check gate | 17.270 s | PASS |
| Source fetch start through both completed gates | 131.632 s | PASS |
| Isolated TERM drain and port closure | 2.028 s | PASS |

Receipt: [`raw/pass-20260812T140424Z/stages.tsv`](raw/pass-20260812T140424Z/stages.tsv).

Before boot, GPU 0 had 59,377 MiB free against the manifest's 45,000 MiB abort threshold. With
both processes resident, total use was 69,471 MiB and free memory was 27,779 MiB: the live
process held 37,836 MiB and isolated PID `17799` held 31,562 MiB. After TERM, only live PID
`15511` remained and use returned to 37,874 MiB. These are single snapshots in an uncontrolled
live-serving thermal regime (32 C/P8 before, 36 C/P1 during, 47 C/P1 after), not performance
medians.

## Protocol and accounting gates

Both models passed the same pinned `public_gate.py` machinery against loopback port 8004:

| Model | Checks | Duration | Client/engine accounting |
| --- | ---: | ---: | --- |
| `qwen/qwen3.6-27b` | 21/21, zero failed | 12.782 s | exact |
| `qwen/qwen3.6-35b-a3b` | 21/21, zero failed | 4.319 s | exact |

For each model the gate observed 13 admitted/completed requests, 34,716 prompt tokens, 10,013
cached prompt tokens, and 528 output tokens on both the client and engine sides. Each battery
covered catalog/auth, streaming and non-streaming plain chat, tools, strict structured output,
three exact cache requests, an eight-way overload split of four HTTP 200 plus four exact HTTP
429 responses, tenant metering, and final usage reconciliation.

Receipts:
[`Q27 summary`](raw/pass-20260812T140424Z/gate-q27/summary.json),
[`Q35 summary`](raw/pass-20260812T140424Z/gate-q35/summary.json), and
[`server log`](raw/pass-20260812T140424Z/server.log).

## Live endpoint before, during, and after

Each phase ran N=12 sequential streaming requests per model through `https://api.tiyuvta.ai`
(N=24 per phase, N=72 total). This was the uncontrolled live-serving regime described above;
latencies are availability observations, not controlled performance claims.

| Phase | Model | N / errors | TTFT p50 / p95 | Latency p50 / p95 |
| --- | --- | ---: | ---: | ---: |
| Before | Q27 | 12 / 0 | 71.691 / 76.280 ms | 737.229 / 745.896 ms |
| Before | Q35 | 12 / 0 | 69.286 / 77.329 ms | 258.223 / 275.239 ms |
| During | Q27 | 12 / 0 | 70.537 / 156.644 ms | 734.185 / 861.092 ms |
| During | Q35 | 12 / 0 | 72.167 / 208.170 ms | 259.595 / 437.491 ms |
| After | Q27 | 12 / 0 | 70.173 / 75.699 ms | 736.294 / 744.361 ms |
| After | Q35 | 12 / 0 | 73.643 / 111.717 ms | 260.530 / 339.085 ms |

All three readiness, health, metrics, GPU-probe, model-catalog, and process checks passed. The
live server PID stayed `15511`. The tmux pane identities were byte-identical before/during/after:
`cf-tunnel` `16631`, `cx-servetest-relay` `12512`, and `cx-servetest-server` `12983`. The drill
runner sent TERM only after resolving PID `17799` to
`/scratch/restore-drill/bin/memra-server`; port 8004 then closed, while port 8002 remained owned
by live PID `15511`.

Receipts:
[`before`](raw/pass-20260812T140424Z/live-before/monitor/20260812T140431921462Z/summary.json),
[`during`](raw/pass-20260812T140424Z/live-during/monitor/20260812T140606507700Z/summary.json),
[`after`](raw/pass-20260812T140424Z/live-after/monitor/20260812T140638514490Z/summary.json), and
[`process identity`](raw/pass-20260812T140424Z/final-live-identity.txt).

## Off-host inventory and gaps

| Existing record | What it supplied | Gap found |
| --- | --- | --- |
| `deploy/gateway/q27-artifact.manifest` | Frozen Q27 bytes/hash and off-box durable source | Q27 only |
| `deploy/gateway/q27-models.toml` | Q27 technical-preflight catalog metadata | No Q35 entry; Q35 commercial fields are not approved and must not be invented |
| `deploy/gateway/capture-manifest.sh` plus gateway preflight captures | Content-free binary/model/config/unit hashes | Captures were Q27 replicas on another box; unit, environment, and cloudflared records were honestly MISSING there |
| `research/servetest-20260812/raw/setup/` | Original source bundle, binary hashes, Q27 artifacts, initial secret fingerprints | Q27-only environment; setup keyring fingerprint no longer equals the current live keyring fingerprint |
| Q35 artifact and pair-cutover receipts | Frozen Q35 bytes/hash and the complete pair `MEMRA_MODELS`/cache/session shape | Separate receipts, not one restore assignment |

The post-drill content-free inventory is
[`raw/inventory-20260812T141507Z/`](raw/inventory-20260812T141507Z/). It confirms that the
original setup keyring fingerprint
`298814a1d63265a30774d381208c36d241127998713ea5996d4c408f734fe48e` had changed to live
fingerprint `a5e580aac60c712ea721d7640c041e377db0eb2b1b310e8e83c2ad2fcffcc3fc`, while the plaintext
API key's fingerprint remained the recorded
`8f65ed54bcc935b0139620082b8320c9ec468b6192828a1a1071f4329e6d5482`. This confirms why a
fingerprint is an identity receipt, not secret-restoration material.

The bounded fix is the committed `servetest-pair-restore.manifest`, two-alias metadata template,
and fail-closed drill runner. Two attempted restores proved why the additions matter:

1. Attempt 1 stopped during source verification because the runtime commit predates the deployed
   servetest scripts. No binary/model/key/server stage ran. The manifest now pins runtime and
   harness revisions separately.
2. Attempt 2 stopped at the same pre-mutation gate because the consolidated monitor hash was
   stale. The blob at the harness revision and the deployed blob agreed on the corrected hash.
3. Attempt 3 passed end to end.

Failed-attempt receipts are retained under
[`raw/attempts/`](raw/attempts/). Neither failed attempt created a drill process, model copy, or
secret. Production readiness and the original PID/listener were re-proved after each.

The only same-box inputs consumed by the passing drill were caches with off-host reconstruction
authority: the installed binary was accepted only after matching the off-host hash (otherwise a
source build is required), and `/scratch/models` artifacts were accepted only after matching the
off-box byte/hash records. The live plaintext API and metrics credentials remain intentionally
outside the repository. Public ingress secrets, relay recovery, historical ledger recovery, and
replacement-host system services were not consumed or proved; they remain operator-secret-store
or replacement-host work, not hidden dependencies of this engine PASS.

## Runbook delta and receipt integrity

`deploy/gateway/RUNBOOK.md` now requires every served model assignment, separate runtime/harness
identity, fresh restore secrets, an isolated loopback ownership check, a pre-boot VRAM abort gate,
both per-model protocol batteries, and N=12 public samples before/during/after. It also makes the
unproved ingress/new-host boundary explicit.

The passing raw receipt manifest has SHA-256
`3d46fb87ad54534ffe54bfe3c7fbaafae94ac720563a0fe549c586131a7725b9` and verifies in full. The
single transfer archive was 26,323 bytes with SHA-256
`70aad0ac5140cfc087bab0e7216fee04448f0b612c312240481593c096da4b4e`; the local mirror and every
nested gate/monitor manifest verified before remote cleanup. A credential-pattern scan found zero
plaintext key hits. The isolated root, its two model copies, generated secrets, upload bundle, and
transfer archive were then removed; the live endpoint remained ready and `/scratch` returned to
73 GiB free.

Top-level receipts:
[`bundle hash`](raw/pass-20260812T140424Z/BUNDLE.sha256),
[`raw manifest`](raw/pass-20260812T140424Z/MANIFEST.sha256), and
[`transfer verification`](raw/pass-20260812T140424Z/TRANSFER.txt).
