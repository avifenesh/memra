# Deterministic MMQ form admission

- Opened: 2026-08-14
- Branch: `lane/mmq-deterministic-5090`
- Base: Memra `v0.82.2` release tree
- Rig: local RTX 5090 Laptop
- Lock: `/tmp/memra-5090.lock`

## Blocker

The Q4_0 MMQ entry previously timed TILE and SK on the first call for each
shape. Those forms have different floating-point fold orders, so timing made
the numerical program depend on boot-time CUDA scheduling.

## Candidate

- Remove the event-timed selector and shape cache.
- Keep `MEMRA_MMQ_SK_FORM=tile|sk` as the explicit measurement and rollback
  seam.
- Use deterministic TILE when no hardware-specific form has completed its own
  exact interleaved gate.
- Do not infer a form from SM count. In particular, Hopper has a sealed SK
  correctness failure and no current PRO 6000 form is qualified.

## Gates

1. Release build of `gemma-gate`.
2. Forced TILE versus forced SK, N=5, alternating order under one lock.
3. Independent naked debug and naked clean boots must select/reproduce TILE.
4. All arms must return RC=0, each speculative arm must match its own plain
   128-token program, and each forced form must reproduce one stable program
   across five independent boots. Record acceptance, throughput, hashes, and
   250 ms telemetry.
5. Run proportional engine, boundary, docs, and performance-board checks before
   integration.

No PRO-specific default or cross-rig performance claim is allowed from this
lane.
## 2026-08-14 RTX 5090 admission

The deterministic selector candidate was measured on the local RTX 5090 under
one uninterrupted `/tmp/memra-5090.lock`. The immutable local run is:

`raw/20260814T161432Z/`

The campaign ran five TILE/SK pairs in alternating order plus independent
debug and clean unforced boots. Every arm matched its own plain target for all
128 generated tokens.

| form | N | accepted / drafted | acceptance | median spec tok/s |
|---|---:|---:|---:|---:|
| TILE | 5 | 106 / 117 | 0.906 | 354.71 |
| SK | 5 | 94 / 118 | 0.797 | 298.04 |

TILE and SK each reproduced one stable token program across all five runs, but
the two forms did not reproduce the same program. This is the decisive failure
of the former first-call timing choice: it could select a different numerical
program on each process boot. Both unforced candidate boots reproduced TILE's
token hash, acceptance, and selector behavior.

Result: **PASS** for deterministic TILE as the unqualified-device fallback.
`MEMRA_MMQ_SK_FORM=sk` remains an explicit measurement/rollback override.
This 5090 result does not authorize a PRO-specific SK default.

The public neutral arm table and machine-readable verdict are `RESULTS.tsv`
and `RESULTS.json`. Host-specific raw logs, device UUIDs, absolute model paths,
and 250 ms telemetry remain in the local immutable run rather than the public
repository.

Proportional validation on the exact candidate:

- `cargo test -p memra-engine --lib`: 84 passed, 1 CUDA-only test ignored.
- `MEMRA_KC_FAST=1 kernel-check`: 82 cells green, 22 capability-gated skips.
- Public boundary: 1,143 grandfathered matches, 0 new; 8 fixtures passed.
- Flag coverage: no new drift beyond the existing baseline.
- Performance board and diff hygiene: current/clean.
