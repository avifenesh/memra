# Boot-time `MEMRA_*` environment audit (#483)

Verdict: memra-server now refuses to boot on a retired door or on an unknown name inside an owned
family, naming the FLAGS.md ledger that retired it, and warns once about names outside every
family. The registry is generated at build time from `docs/FLAGS.md` by `crates/memra-engine/
build.rs` (`memra_engine::env_audit`), so a deleted door gets a real ending: the refusal, not a
value that parses into nothing.

## Rules (mirroring the flags census, then tightening where the census is loose)

- Legal: every `MEMRA_...` token mentioned in FLAGS.md outside the `## Removed ...` ledgers; a
  trailing `*` documents a prefix family. This is `tools/check-flags.sh`'s own acceptance rule,
  so the audit never refuses a name the census passed (all 866 runtime reads on this tree are
  legal under it).
- Retired: every non-prefix `MEMRA_...` token of each definition block (a `- ` bullet, a `| ` row
  or a backtick-led line, plus its indented continuation lines) under a heading at ANY level whose
  text starts with `Removed`, and every live row whose default cell opens with `REMOVED`, minus
  names that still have a live row DEFINITION (first table cell); prose mentions in live rows do
  not un-retire a door. 145 retired doors on this tree after review round 1 (the first cut took
  only the first token of `## ` ledgers and missed `MEMRA_DSV4_NORM_FUSE2`,
  `MEMRA_PREFIX_CACHE_PROTECTED_PCT` on a continuation line, `MEMRA_PRIME_QW8` under a `###`
  ledger and `MEMRA_SEL_DOWN8` on a REMOVED row).
- Order: retired first, then legal, then family. A prose wildcard (`MEMRA_DSV4_*`, `MEMRA_MOE_*`
  and eight more appear in FLAGS.md prose) documents a whole family, so without this order the
  incident's own name, `MEMRA_DSV4_MOE_PROGRAM`, would have been "legal".
- Owned family: a `MEMRA_<FAMILY>_` prefix under which the engine or server CODE reads three or
  more distinct names (`OWNED_FAMILIES`, derived from the sources at build time; 40 families).
  Unknown inside it refuses; unknown outside is one summarized warning line. The catalog also
  documents shell-side knobs (`MEMRA_CI_*`, `MEMRA_GATE_*`, `MEMRA_HITGATE_*`, 128 names that no
  Rust reads), and the first battery on this lane proved why the family test must come from the
  code: `MEMRA_CI_LOCK_HELD`, which `tools/local-ci.sh` exports around every server boot, was an
  "unknown `MEMRA_CI_*`" refusal under a catalog-derived family and serve-smoke died
  (`raw/local-ci-perf-attempt1-audit-refused-ci-lock-held.log`).
- `MEMRA_ENV_AUDIT=warn` downgrades refusals to warnings for one boot; `=0` turns the audit off,
  announced. The audit runs after the no-engine CLI exits (`--version`, key management) and
  before any door is read.

## Receipts (`raw/unit-tests-and-server-smoke-final.log`; first cut in `raw/unit-tests-and-server-smoke.log`)

Unit tests (CPU, in `env_audit.rs`): registry populated, sorted, retired disjoint from legal
names; every legal name exported at once refuses and warns about nothing (non-vacuity); a retired
door refuses naming its ledger (red arm, plus the incident's name and the four round-1 escapees by
name); unknown inside an owned family refuses, outside is one line; `MEMRA_CI_` is not owned; the
tree reads no retired name except the worker's two deliberate refusal reads (source-derived, the
allowlist asserts they are still retired); the audit's own switch and non-`MEMRA_` names ignored.

Server smoke, real `memra-server` binary on the local rig, one env var at a time:

| env | boot |
| --- | --- |
| `MEMRA_DSV4_MOE_PROGRAM=reference` | REFUSED, "retired. See docs/FLAGS.md 'Removed doors, 2026-09-11 (the matrix expert program is the default...)'" |
| `MEMRA_PREFIX_CACHE_POLICY=lru` (the 2026-09-21 launcher fix, now retired) | REFUSED, ledger 2026-09-21 |
| `MEMRA_NVFP4_BANK_V2=1` | REFUSED, ledger 2026-09-05; the worker's own richer refusal for this name (successors listed) still guards the CLI bins |
| `MEMRA_PRIME_NOPE=1`, `MEMRA_PREFIX_NOPE=1` (unknown, owned families) | REFUSED, "unknown MEMRA_PRIME_* name" |
| `MEMRA_PRIME_QW8=1` (`###` ledger), `MEMRA_SEL_DOWN8=1` (REMOVED row), `MEMRA_PREFIX_CACHE_PROTECTED_PCT=1` (continuation line), `MEMRA_DSV4_NORM_FUSE2=1` (second door of a bullet) | REFUSED, retired, ledger named (round-1 escapees) |
| `MEMRA_CI_LOCK_HELD=1` (battery knob, no Rust read) | one warning line, boot continues |
| `MEMRA_DSV4_NOPE_XYZ=1` | boots: FLAGS.md prose carries `MEMRA_DSV4_*`, so the census and the audit both accept any DSV4 name (see limit below) |
| `MEMRA_ZZ_LAUNCHER_ONLY=1` | one warning, boot continues |
| `MEMRA_ENV_AUDIT=warn` + retired name | warning, boot continues |
| `--version` with a retired name set | answers; the audit runs after the CLI exits |

## Limit worth its own follow-up

Ten prefix wildcards live in FLAGS.md prose (`MEMRA_DSV4_*`, `MEMRA_F16G_*`, `MEMRA_MOE_*`,
`MEMRA_PP_*`, `MEMRA_PENALTY_*`, `MEMRA_SPEC_GATE_*`, `MEMRA_LANE_MAX_*`, plus the three
first-cell families `MEMRA_GSG_*`, `MEMRA_SELSHAPE_*`, `MEMRA_TP2_*`). They make a typo inside
those families "documented" for the census and therefore legal here. Retired names in those
families still refuse (the order above); typos do not. Tightening means documenting the names
instead of the wildcards, which is a FLAGS.md change, not an audit change.

The tree also still reads `MEMRA_NVFP4_BANK_V2` on purpose: `worker.rs` refuses the name with the
three successors spelled out. The audit reaches the server boot first with the generic ledger
message; the worker's message keeps guarding `run-gen` and the other bins, which do not run the
audit yet.
