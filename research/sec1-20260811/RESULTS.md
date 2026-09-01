# cx-sec1 serving-auth hardening — results

Verdict: **PASS**. All four confirmed external-review findings are fixed on
`lane/cx-sec1` in the implementation series `94f033ab` through `d27aeefa`.
This was a CPU-only lane: no model load, GPU execution, performance claim, merge, tag,
push, or release was performed.

## Findings closed

| finding | result |
|---|---|
| Public metrics | `/metrics` and `/yield/metrics` now require `Authorization: Bearer ...` whenever an API-key source or `MEMRA_METRICS_TOKEN` is configured, or the bind is non-loopback. Normal API keys and the dedicated scrape token both work. Only no-key loopback development remains open. |
| Open public bind | Bind security is validated before model/GPU initialization. A non-loopback bind without `MEMRA_API_KEY` or `MEMRA_API_KEYS` is a startup FATAL; `MEMRA_ALLOW_OPEN_BIND=1` is the explicit development override, and metrics remain locked in that shape. Empty `MEMRA_API_KEY` is rejected. The systemd template now defaults to `127.0.0.1:8000` and names `keys.toml`. |
| Secret comparison | Static, metrics-token, and keyring authentication compare fixed 32-byte SHA-256 digests with a full-length constant-time equality loop. Keyring lookup scans every stored digest instead of relying on short-circuit string/HashMap equality. |
| Keyring persistence | New keyrings are created mode `0640`. Revocation writes mode-`0640` `keys.toml.tmp`, writes all bytes, calls `sync_all`, and atomically renames it over the live ring. The concurrent reload test repeatedly reparses and authenticates while 32 rewrites occur. |

The repository-owned fleet meter, RunPod provisioner, trial health matrix/runbook, and live
economics scraper were updated to send the dedicated bearer. The RunPod path generates an
unprinted token when one is not supplied, stores it in the mode-`0640` environment file, and
feeds curl headers through stdin rather than process arguments.

## Verification

| gate | result |
|---|---|
| `cargo test -p memra-server` | **PASS**, 167 passed, 0 failed |
| Focused auth suite | **PASS**, 12 passed, including mode checks and concurrent atomic-rewrite/hot-reload stress |
| Metrics + bind regressions | **PASS**, both metrics routes reject missing keyed auth, accept API/scrape bearers, preserve no-key loopback development, and refuse an exposed open bind |
| `cargo build -p memra-server` | **PASS** |
| Real binary open-bind smoke | **PASS**: no-key `MEMRA_ADDR=0.0.0.0:18080` exited `1` with `FATAL: refusing unauthenticated non-loopback bind` before model initialization |
| Shell surfaces | **PASS**: `bash -n` and `shellcheck` on fleet-meter, RunPod provisioner, and trial-up scripts; fleet-meter help and RunPod dry run passed |
| Python/docs/generated surfaces | **PASS**: `cache_economics.py --help`, `tools/update-perf-board.py --check`, and lane-only `git diff --check` |

`systemd-analyze verify` parsed the three source units and reported only the expected missing
installed executables (`/usr/local/bin/memra-server` and `/opt/memra/tools/fleet-meter.sh`) in
this worktree; live installation verification belongs to the deployment host.
