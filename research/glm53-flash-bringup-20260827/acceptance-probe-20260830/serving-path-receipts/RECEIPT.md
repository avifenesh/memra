# Serving path does NOT route glm5 spec — fail-closed receipt (card3 lane, 2026-08-30)

Boot: the cc718b988 `memra-server` build (md5 `bcf010c09e8aee027e6d25a4c964fcbc`, run
as `memra-server-card3`), 1-card SLRU probe posture, `MEMRA_GLM5_MTP=1
MEMRA_GLM5_SPEC=1`, vision OFF, MOE_SLOTS=12000, TF32 off, MEMRA_CTX=8192.

- `serve-mtpspec-boot.log` carries `[mtp-glm5] MEMRA_GLM5_MTP=1: loading the
  glm5_next NextN block` — the head LOADS at serve time (73,636 MiB free on the card
  after boot; the head costs ~1.8 GiB vs the vision boot's tower).
- Fresh-boot output-sample gate on THIS boot: PASS (00/01 JSONs — greedy +
  vendor-default sampled, fluent, on-topic).
- Both gate requests served `path=plain` (3 `path=plain` admission lines, zero spec
  lines, zero `[glm5-spec]` engagement lines in the log — grep receipts in the log
  copy). With the spec flags SET, the worker still routes plain: `mtp_spec_capable`
  (worker.rs) requires `plan_backend::MTP_SPEC.capabilities(plan).speculative
  .supported`, which the tparallel-verify lane deliberately reports false for the
  glm5_next plan (fail-closed manifest stance). This is the anticipated outcome, now
  measured live rather than assumed.
- Consequence for this probe: acceptance is measured at ENGINE level
  (`glm5-card3-probe` -> `HybridModel::generate_spec` -> `generate_spec_glm5`), the
  same `MEMRA_GLM5_SPEC` door the fixture gates pinned. Stated per the lane spec.
- `30-neg-flag-off-image.json`: the vision flag-off negative ran on this boot —
  HTTP 400 "image input is not enabled on this deployment" (closes the cell-1
  refusal battery).

Loader observation (banked, not patched): this boot (like every boot of this
artifact family at cc718b988) prints `[loader-law] WARNING: <name> loads as 2D Float
... (src BF16) — a Float matmul weight rides cuBLAS f32 GEMV and poisons
all-or-nothing q8-fast predicates` for `output.weight`, the `kda_*` projections, and
the NextN glue (`blk.*.nextn.eh_proj.weight` [8192, 4096]). The eh_proj row is new
with `MEMRA_GLM5_MTP=1` (the head did not load before this lane); the audit decision
(Q8_0-encode vs float_2d_audited entry) belongs to the engine lane.
