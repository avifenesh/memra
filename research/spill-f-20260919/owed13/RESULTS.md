# OWED 13: B2 handoff driver (2026-09-25)

**Designed, registered, and landed CPU-verified against a stub server; no GPU boot has run.**
Registration: `M1-PREREG.md` "B2 amendment" (written before the code, amended the same day with
the measured prompt lengths).

## Finding that shaped it

The in-tree `memra-server` has no route that triggers the host-tier handoff export: only a
deployment binary's `ServerWiring::on_ready` hook receives `HostHandoffHandle`. The boot-time
import needs no trigger (a present file arms it; the idle loop drips it at a 1 ms wait).

## What landed

- `crates/memra-server/src/bin/kv_handoff_gate.rs` (`[[bin]] kv-handoff-gate`): stock wiring plus
  an `on_ready` task; SIGUSR1 calls `export(force = false)` and prints `[handoff-gate] export ok
  <json>` or `export refused: <reason>`; drops every handle on the drain shutdown signal. No env
  read, no route; refuses any argument before model load (`owed13/cpu/gate-refuses-args.log`,
  exit 2).
- `crates/memra-server/src/worker.rs` `host_handoff_export`: the export line gains
  `write_ms=` (serialize and buffered write) and `fsync_ms=`.
- `m1-handoff-driver.py`, stub `m1-stub-kv-server.py`, tests `test-m1-handoff-driver.py`.
- `m1-b2-prompts.py` plus `m1-prereg/b2-prompts.manifest.json`: 128 deterministic prompts of
  6,500 plain words (SHA-256 `8e397009...e4e8`, 5,743,190 bytes, generated on the box, not
  committed); the pinned artifact's tokenizer (`tok-check`) measures 6,526 to 6,535 tokens with
  the probe suffix, inside the 8,144-token budget. Two generations are byte-identical.

## CPU gates (raw logs in `owed13/`)

| Gate | Result |
|---|---|
| `cargo clippy -p memra-server --lib --bin kv-handoff-gate -- -D warnings` | exit 0 |
| `cargo build -p memra-server --bin kv-handoff-gate` | exit 0 |
| `cargo test -p memra-server --lib handoff` | `test result: ok. 7 passed; 0 failed; 1 ignored` |
| `kv-handoff-gate --bogus` | `[handoff-gate] FATAL: takes no arguments`, exit 2, no model load |
| `test-m1-handoff-driver.py` | **9 of 9 pass**: two green cycles (fill to the size, export, file hashed and made cold, import consumed with 0 skips, four probes restored and identical, both sampler windows valid); red: refused export (import phase skipped), import with skips, a probe missing the cache, a probe whose text drifts, prompts exhausted, a scratch directory on tmpfs refused before any boot, the stub-only lock bypass; the exact real log formats parse |
| `cargo fmt --all -- --check`, `git diff --check`, `tools/check-flags.sh` | clean |

## Still owed on the box

The driver's real run (1 GiB and 8 GiB, N = 5 each) after the M1 proof PASS, with the gate and
the stock server built from this commit. The per-entry KV size of this artifact is not known
offline; if 128 prompts cannot reach 8 GiB the cycle fails as "prompts exhausted", which is a
recorded result, not a reason to change the size mid-run.
