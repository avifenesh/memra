# Integration — days 8–9 (`lane/spill-integ5-20260920`)

Lead: @agent-c07799. Base: `origin/main` `fdb78136` (replayed after each upstream move). Lane tips merged:
B `9b0fbe81`, C `0b52d688`, D `196dad03`, F `7644c41f` (A `06fda471`, E `4d89a243` unchanged since #563).

## Headline: first gate passed — `ACTIVE-8K G1 PASS`
B's day 9 (`DAY9.md`, `RECLAIM-DESIGN.md`, decision record `docs/decisions/KV-PHYSICAL-RECLAIM.md`):
diagnosis `RECLAIM-DIAG: freed but not observable` (cudaFreeAsync pool retention; trim returns nothing) →
VMM-backed KV planes behind the gate-only `--kv-allocator vmm` door → 8k demote releases and reacquires
exactly 201,326,592 B of driver-visible VRAM with bit-identical continuation. 32k: bit-identical, one-granule
residual **unclassified** → not G1 PASS (label verbatim in DAY9.md). Decide-by 2026-10-04.

## Also landed (all rented-5090, collector-locked, `executed-not-qualified`)
- **C** `DAY8.md`: Qwen3.6-35B-A3B experts-via-tier under pressure — bank budgets 8 GiB and 4 GiB, ON and OFF
  controls, every cell `MATCH` / `SELF-CONSISTENCY PASS` with eviction engaged (4 GiB spec: 178,523 GPU
  evictions, 165,768 re-reads); fail-closed cell `REFUSED: experts-via-tier host bank budget cannot hold one
  expert record` (collector `refused`, exit 2); CPU replay of 311,558 recorded decisions + the 2,013 frozen ones.
  Seam note: `MEMRA_MOE_SLOTS` clamps to 8, so it cannot express a budget refusal.
- **D** `DAY9-VERIFICATION.md`: collector children no longer escape the worker-group kill (F's finding; red
  grandchild test → green on macOS + Linux; inherited lock FD preserved); G2 rehearsal on the final probe CLI
  (16 visits, N=1, every visit ≥250 ms, no medians); `--validate` over the peer archives:
  `CAPTURE ARCHIVES MATCH: 14 cells; 12 executed-not-qualified; 1 failed command; 1 refused command; qualification=false`.
- **F** `H2D-RESULTS.md`: `h2d-probe --copies 1..100000` with per-op event timing, bare-JSON visits (D's
  parser contract), one native N=1 cell (32 visits); `NVME-DECISION.md`: VM spend **NO-GO** until in-guest
  block ancestry is provable.

## Lead-owned edits
`crates/memra-engine/src/bin/kv_tier_gate.rs`: merge resolution only — B's `reclaim_contract` module plus
the `use memra_engine::tier_transfer;` from #563 (B's lane still carried the pre-#563 `#[path]` twin).
`docs/decisions/KV-PHYSICAL-RECLAIM.md` (new arm → decision record). `research/INDEX.md` rows.

## Battery (this tree, Mac, offline)
`cargo test -p memra-tier -p memra-kv`: 261 passed / 0 failed · clippy `-D warnings` (engine, server, tier, kv,
gguf; all-targets on `x86_64-unknown-linux-gnu`, `DOCS_RS=1`) clean · fmt · `check-flags.sh` · publish census
12/12 · collector Python suite 78 passed · perf board up to date · `git diff --check` clean.

## Rig note
The spot instance was host-stopped three times today; the power cap read 400/600 W on some restarts and
600/600 W on others — every cell records its own envelope, and no timing is compared across them.

## Gates
G1: 8k PASS; 32k pending residual classification. G0/G2–G7 unchanged. Next: classify the one-granule residual
(driver mapping metadata vs. anything else) with a targeted probe; allocator injection into `Cache` construction;
C's proxy boundary into `MoeSlotCache` behind its door; D's scored N≥5 G2 when a clean window is authorized.
