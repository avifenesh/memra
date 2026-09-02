# Release notes: v0.122.0 and v0.123.0

Public change record for the two tags. The repository history was rebuilt from a content
snapshot on 2026-09-01 (zero-history swap), and the old repository's GitHub Releases page did not
survive it, so these notes, which used to live in the README, are kept here byte for byte.
The old history is preserved in the archive bundle `memra-FINAL-preswap-all-refs.bundle`
(rig `~/repo-backups-20260901/`, and R2 bucket `tiyuvta-capture` under `repo-bundles/20260901/`);
it carries `refs/tags/v0.123.0` and the old main tip `469c8898e`. crates.io is continuous:
`memra-engine 0.123.0` was published 2026-09-01T04:59Z from old-history commit `bc0952fe5`,
which is tag `v0.123.0`.

## What v0.123.0 ships

The glm5_next bring-up consolidation, the step37 NVFP4 program restore with its first
default flips, and two new-architecture bring-ups. Flag defaults are per-row decisions with
receipts in [docs/FLAGS.md](docs/FLAGS.md); numbers below carry their lane receipt paths and
are lane measurements, not serving claims.

- **glm5_next (GLM-5.3-Flash) full serving stack**, bring-up state: the 45-layer hybrid
  KDA+MLA/DSA architecture with sigmoid 288-expert MoE, Sinkhorn mHC, NoPE MLA and the DSA
  indexer, under a 3-card pipeline recipe (resident PP3, SPLITS 15,30). Ships the batched
  spec-verify walk (`MEMRA_GLM5_VERIFY_BATCH`, default ON, bit-gated per row), the DFlash2
  drafter spec session (`MEMRA_GLM5_SPEC` + auto-K + confidence floor), MLA tensor-core
  attention default ON (TTFD −62 to −69% on two boxes), hyper-batch default ON, vision
  default ON, the weight-read-once matvec door family default ON (T/X/K/W; mv-battery
  2026-08-31), verify-rows MoE kernels at 90% DRAM peak, and dedup-schedule/EP-diet doors
  default OFF pending box pricing. Greedy instrument figure on the ship recipe: 71.49 tok/s
  (3 cards); receipts under `research/glm53-flash-bringup-20260827/`. NOT serving-exposed:
  the serving bar (100 tok/s at ctx 262144) is an open lane, and no product claim ships here.
- **glm5 TP widened to rank 4, and spec decode composes with TP** (`GLM5_TP_ALLOWED_RANKS
  = [2, 4]`, TP-3 refuses by name): per-rank transport with an all-ordered-pairs
  byte-integrity ladder, peer-shard KDA/MLA sidecars, rig-gated bit identity; the blanket
  spec-session co-refusal on a TP-armed model becomes a gated admission behind
  `MEMRA_GLM5_SPEC_TP` (default OFF by design) with sharded verify/rollback through the
  batched walk only. The peer-pull transport door and movement census landed with the
  fail-closed TP matrix (`research/glm53-flash-bringup-20260827/composition-20260901/`).
- **`apply_penalties_dense`**: the host sampler's O(n_vocab) per-token hash-and-sort on
  penalized sampled rows replaced by a dense pass, bit-identical by a 24-case `to_bits`
  gate (old scan form kept as the oracle). Found by the host-audit lane tracing a live
  prod shape; `MEMRA_WORKER_AFFINITY` ships alongside as a default-OFF diagnostic seam whose
  box battery measured null on every arm (`research/glm53-flash-bringup-20260827/host-audit-20260901/`).
- **step37 NVFP4 bank-v3: the 2026-08-29 corruption root-caused and the programs restored.**
  The defect was a defaulted `in_f = 0` scale-fetch argument in the prefill grouped GEMM
  (right codes, wrong scale, every k-block but the first) — the slot-major layout was
  innocent. The default is deleted so the compiler enforces all call sites; the three
  removed programs return under three separate strict doors gated by the device-side
  `nvfp4-bank-oracle` with a behavioural teeth arm. First default flips ride the deploy-grade
  12-boot battery: `MEMRA_NVFP4_BANK_SM` + `MEMRA_NVFP4_SEL_DOWN8` default ON as one coupled
  decision (+5.44% decode / +5.92% wall on the vendor-default sampled shape, per-boot ranges
  separated 4/4, 16/16-turn cache twin holds; engages on the device-routed TP path — see
  the eligibility conditions in docs/FLAGS.md), `MEMRA_NVFP4_SEL_GU` stays OFF
  (`research/step37-bankv3-20260901/`).
- **hy3 native tune**: automatic expert-parallel device router with batch-cap admission,
  masked MTP, an internal W4A8/mixed-Q8 activation scope for whole-expert EP, generic TP
  attention composed with expert EP, and the shared-expert overlap door
  (`MEMRA_SHEXP_OVERLAP`, default OFF).
- **qwen4_exp (Qwen3.8-Flash-Next) bring-up**, NativeReference + GPU-eager with exactness
  gates: hybrid GDN 3:1 QSA, 512-expert softmax top-k router with gated shared expert,
  4-branch gated residual, PLE n-gram embedding, YaRN with refuse-at-parse for unimplemented
  keys. Loader/reference lane only; no serving exposure and no product claims
  (`research/qwen4exp-bringup-20260829/`).
- **Public-boundary checker hardened for symlinks**: a tracked symlink publishes only its
  target string and is scanned as such, never dereferenced (a box-absolute link crashed the
  checker on CI runners); the tree itself now carries zero box-absolute links.

## What v0.122.0 ships

KV host tier and serving-guard release. Every new flag defaults OFF with an audited row in
[docs/FLAGS.md](docs/FLAGS.md); numbers below are from the 2026-08-31 qualification pod battery
(2x RTX PRO 6000 Blackwell 96 GB).

- **Graph-launch guard on every serving-reachable captured-graph route.** When driver-free
  memory drops below the 256 MB launch floor, captured-graph replay suspends with a
  route-tagged `graph replay suspended:` line and the request serves on the eager arms
  (fail-closed to eager, never a segfault into an exhausted card). Fired 5/5 squeeze runs on
  each of the q38 verify-graph, ornith MTP verify-graph, and step37 TP-2 routes; zero
  suspended lines at healthy headroom.
- **Prefix-cache host spill tier**, `MEMRA_KV_HOST_MB` (default OFF): device-cache evictions
  demote verbatim into pinned host RAM and promote back through the existing restore path,
  byte-lossless by construction. Gates: restored-vs-cold byte identity (ON == OFF bytes,
  verify digests ok, teeth arm inverts as required); the 8-turn larger-prompt cache twin
  holds TTFT flat (0.61 to 0.77 s p50) through turn 8 while the no-tier arm grows to
  3.88 s p50, a 5.6x p50 TTFT gap at turn 8 in that shape.
- **Tenant lifecycle purge and per-tenant share cap.** `PurgeHandle::purge_tenant` clears a
  tenant's resident host-tier and unpinned device entries on key revocation or deletion
  (`/admin/tenants/{tenant}/purge`); `MEMRA_KV_HOST_TENANT_PCT` (default 50) caps one
  tenant's share of the host pool.
- **Plain-pool park compaction**, `MEMRA_KV_PARK_COMPACT` (default OFF): a retiring
  continuation-pool session parks at exactly its committed length instead of its ladder cap;
  resume restores the parked rows, byte identity after replay 4/4 under the step-OOM
  adjacency battery.
- **Agent-pause KV demotion**, `MEMRA_KV_PAUSE_DEMOTE` (default OFF): a turn that ends in a
  completed tool call arms a pause candidate and demotes its boundary state to the host tier
  after `MEMRA_KV_PAUSE_DEMOTE_MS` (default 5000 ms, set from the A3 gap census). Natural
  `tool_calls` arm 6/6 on both boots; 16 verify-ok round trips, 0 failed; co-run decode tax
  -1.80% median.
- **Predictive-admission shadow receipts**, `MEMRA_ADMIT_PREDICT_SHADOW` (default OFF):
  log-only per-request admit/reject verdicts with the full KV book; nothing is rejected.
- **Boot calibration probes the served route**: the admission floor probe rides the route the
  model actually serves (q38 dspark boot charges zero MTP draft-state; the ornith MTP route
  charges its real measured draft state).
- **Verify-graph pool debt charged by struct**: the MTP verify-graph pool no longer escapes
  admission (by-struct reserved debt plus a per-session measured capture charge).
- **Offline expert-placement map builder**, `tools/build_expert_placement_map.py` (frozen
  format `memra-ep-map-v1`; strategies coactivation, frequency, even; selftest 10/10 with
  proven teeth).
