# WP-C day 56 (2026-09-24): the DFlash tail slice of the contracts door, design registered before code (OWED C5)

`OWED.md` C5. `DAY19.md` Task 3 pre-registered the question, the rule and the shape; this day adds the design from
today's source and the cells, before any code. Tree at start: `ef7906504`. The rule stays DAY19's, verbatim:

> The identity gate, default environment with `MEMRA_DSPARK_SPEC=1` and the drafter, OFF against ON: `ALL GREEN`
> both arms, the same verdict lines, equal demote bytes, and under ON one receipt per tail plane per draft layer
> (`Role::Tail`), with a promote that restores the tail and re-arms the drafter (the restore's own verdict line
> unchanged between arms). A refusal line under ON is a FAIL of the slice, not of the door.

## 0. Census, today's tree (`crates/memra-server/src/worker.rs`, `crates/memra-engine/src/dflash.rs`)

- Refused by name in five places, all through `host_tier_entry_class(glm, mtp_draft, dflash_tail)`: the door's demote
  charge, the image build's contract route, the insert, the promote probe and `bind_tier_image`
  (`entry carries a DFlash draft tail outside the contract-routed surface (no drafter artifact identity in this
  slice)`). The D2D capture and restore routes (Move 2) keep the tick program for tail entries by name (unchanged
  here).
- The tail's bytes already cross the tier in both arms by the same program: at demote `HostF32::down` per layer
  (the door's pageable tier: a heap `Vec<f32>` read on the owner stream), at promote `engine.htod` per layer on the
  owner stream. The contract route moves KV planes (u8 `KvPlane` leases with D2H receipts) and, off the tick, the
  recurrent f32 planes as spans; the tail rides neither, so under ON it is copied exactly as under OFF.
- The drafter (`DflashDraft::load`, `MEMRA_DSPARK_SPEC=1`, `MEMRA_DSPARK_DRAFT=<export dir>`) reads exactly
  `config.json` and `model.safetensors` from the export directory and quantizes by `MEMRA_DFLASH_PREC` (q4 default).
  The tail is `rows` rows of `row_bytes` per layer, K and V each `rows * row_bytes` bytes, `base + rows == len`
  (`export_tail`).
- A tail restore re-arms the drafter only under `MEMRA_DSPARK_PREFIX_RESTORE=1` (default off), with the line
  `[prefix-cache] DSPARK restore: ...`.

## 1. Pre-registration

**The design.**

- (a) **A third entry class**, `HostTierEntryClass::DflashTail`. `host_tier_entry_class` keeps GLM refused by name;
  an entry with both an MTP draft plane and a tail is refused by name (two spec programs never coexist on one model;
  the boot guard refuses the combination, so the class does not guess); a tail alone is `DflashTail`.
- (b) **The tail program**, one per model with a drafter attached, built in `host_tier_context` beside the plain
  and draft programs: the plain base with `artifact` = the trunk digest framed with the drafter's byte manifest
  (`config.json=<sha256>;model.safetensors=<sha256>`, the two files the loader reads, streamed at boot),
  `serialized_plan` = the plan text framed with the drafter's `DflashCfg` (its `Debug` form) and the drafter's
  numeric knobs (every `MEMRA_DFLASH_*` and `MEMRA_DSPARK_*` variable set at boot except `MEMRA_DSPARK_DRAFT`, whose
  bytes the manifest names, sorted as `NAME=value`; over-inclusive on purpose, fail closed), `numeric` = the plain
  class plus `+dflash-tail-f32`. Every other field is the plain field. A boot line names it: `[prefix-host]
  contracts door: model <m> DFlash tail program drafter_manifest=config.json=<sha256>;model.safetensors=<sha256>
  dflash_cfg_sha256=... knobs=[...] numeric=...` (the line's field name corrected here to the code's before any
  cell ran: the manifest is printed whole, not hashed again). A tail-bearing entry on a model without a drafter is refused by `program()` with a typed line.
- (c) **The tail in the bound image**: per draft layer a K and a V segment under `Role::Tail` (encodings
  `dflash-tail-k-f32` and `dflash-tail-v-f32`, `row_bytes` as the row), each checksummed by the bind like the
  recurrent planes (hashed on the owner thread: the tail is not handed to the hash helper in this slice; its bind
  cost is printed, below). A geometry check before binding: every layer's K and V hold `rows * row_bytes` bytes and
  `base + rows == len`, else the typed `tier DFlash tail geometry mismatch`. The shape blob of a tail image is the
  v2 blob with the tail framed after it (presence, `base`, `rows`, `len`, `row_bytes`, `floor`, the layer count, per
  layer K and V byte counts) under its own encoding `host-prefix-shape-v2+dflash-tail-v1`; plain and MTP-draft images
  keep `host-prefix-shape-v2` byte for byte.
- (d) **The receipt line**, once per bound tail image: `[prefix-host] contracts door tail bound: <L> draft layers,
  <2L> Role::Tail segments (<B> B, hashed in <t> ms on the owner thread), tail_checksums_sha256=<hex>`. This is the
  rule's "one receipt per tail plane per draft layer": `2L` segments, each with its checksum inside the bundle the
  identity lease requires at promote.
- (e) **Promote.** The candidate's class names the tail program; the identity lease is required against it before
  publication, as for the other classes. The tail's H2D stays the owner stream's `engine.htod` per layer (both arms).
- (f) **The identity gate's drafter arm.** `tools/kv-host-spill-identity-gate.sh` gains checks that run when
  `MEMRA_DSPARK_SPEC=1` is set: the ON boot prints `[prefix-cache] DSPARK restore:` on the promoted request (r3);
  and with `MEMRA_KV_HOST_CONTRACTS=1` also a `contracts door tail bound:` line and no `(contracts door)` refusal
  line. Every existing check stays.
- (g) Docs in the same change: `docs/FLAGS.md` (the door's row: the tail class), `docs/TESTING.md`,
  `HOSTPREFIX-DOOR.md` section D item 2.

**Acceptance.**

- CPU (under the 1200% cap): the entry-class test rewritten to the new surface (GLM and draft-plus-tail refused by
  name, tail alone `DflashTail`); the tail program a pure function of its sources (each input moves it, and it
  differs from the plain and draft programs); the shape blob unchanged for plain and draft images and framed for
  tail images; a census that every `HostTierEntryClass` match has the new arm; server lib, clippy `-D warnings`, fmt,
  the flags census.
- RTX 5090 cells (`day56-cell.sh`, the day-24 driver shape, the 27B `Qwen3.8-27B-NVFP4-Q5K-mtp.gguf` with the
  DFlash2 drafter export, `MEMRA_DSPARK_SPEC=1 MEMRA_DSPARK_DRAFT=<export dir> MEMRA_DSPARK_PREFIX_RESTORE=1`): the
  identity gate door OFF and door ON, DAY19's rule read from the two gate logs by `day56-reading.py` (both `ALL
  GREEN`, the same verdict lines, equal `[prefix-host] demote:` bytes, the ON arm's tail receipt with `2L` segments,
  a `DSPARK restore:` line in both arms with the same text, no `(contracts door)` refusal line under ON). The device
  prefix budget is set so one seed entry fits and two do not, from a dry boot's `insert` line, recorded before the
  cells.
- The target card: the same two cells on the 27B with the drafter, added to the DAY52 sitting before it runs (the
  drafter export staged beside the artifacts, its manifest checked).

**What each card decides.** Each card's cells pass or fail on that card.

## 1a. Addendum, with the code and before any cell ran

- **The device prefix budget** is 256 MB, the target card's day-23 identity-gate shape for this 27B
  (`MEMRA_HOSTGATE_CACHE_MB=256`, a 64-token entry about 160 MB), with the tail's bytes computed rather than read
  from a dry boot: a DFlash2 tail of about 70 rows of 4,096 B per layer (`n_kv 8 x head_dim 128 x 4`), K and V, five
  layers, about 2.9 MB. One entry fits and two do not; the gate itself fails with its tuning line if no demote fires.
- **The drafter** is the local export `config.json` `873e3556...e980` and `model.safetensors` `67fc76d6...b65c`
  (`DAY20.md`'s manifest), q4 by default, no FR-Spec trim (the trim moves proposals and acceptance, never the draft
  KV or the emitted tokens); the cell driver records both files' SHA-256.
- **The reader** (`day56-reading.py`) compares the OFF arm's check lines with the ON arm's minus its door-only lines
  (the two `door ON` checks of the drafter arm), so "the same verdict lines" is read line for line.
- Scripts: `day56-cell.sh` (cells `identity-dspark-off`, `identity-dspark-on`).
