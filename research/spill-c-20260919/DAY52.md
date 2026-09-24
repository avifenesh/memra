# WP-C day 52 (2026-09-24): the MoE slot cache door's target-card sitting, pre-registered

`OWED.md` C1 on the target card (one RTX PRO 6000 Blackwell Workstation Edition). `DAY40.md` section 5 owes the
attribution there as rung 0 of "the target-card sitting that measures the improvement ladder"; days 43 to 50 each say
the target card reads their clauses "in the ladder"; `DAY51.md` registers the deciding cell per card. Written before
any rung cell ran on either card (the eight RTX 5090 rung cells are queued behind another lane's hold at the time of
writing), so nothing here is shaped by a rung result. Tree at start: `38483ca0d`.

## 1. Pre-registration

**One sitting, five cells, in this order**, each its own collector lock hold (`tools/tier-battery.py --rig
pro-single`, `/tmp/memra-gpu.lock`, 250 ms telemetry) through the day-40 runner, the runner under the same CPU cap
as the 5090 cells (a 1200% scope, or 12 pinned cores where the box has no systemd; the driver records which):

1. `attrib`, rung 0: `day40-cell.sh attrib` unchanged (day 40's binary, arms off, on, ons, N=10 per arm), reader
   `day40-attrib.py`: the integrity checks and readings R1 to R6 of `DAY40.md` section 3 on this card. This is the
   attribution of day 18's `door_cost_ms_per_decode_token=60.47`.
2. `ladder`: `day52-cell.sh ladder`, every rung in one hold (below).
3. to 5. `hashlock`, `spec`, `decide`: `day51-cell.sh`, on the final binary, exactly as `DAY51.md` sections 1 and
   1a register them for the target card (G1 on the Qwen3.8-27B NVFP4 MTP GGUF with day 18's
   `MEMRA_MOE_RESIDENT=0`; G2's three shapes; the timing cell and its rule), reader `day51-decide.py`.

**The ladder.** Day 18's overlap environment (`MEMRA_MOE_RESIDENT=0 MEMRA_NGEN=32 MEMRA_MOE_SLOTS=9986`, prompt
`55 88 13`, the approved artifact). Twelve arms, every door arm with `--expert-bank-stages`:

| arm | binary | host budget |
|---|---|---|
| `off` | `run-gen-final`, no door | |
| `base` | `run-gen` (day 40's tree) | default |
| `i6d` | `run-gen-i6` | default |
| `i6g` | `run-gen-i6` | 17179869184 |
| `i9g`, `fill`, `i1`, `i2`, `i8`, `i5`, `i7`, `i4` | `run-gen-<rung>` | 17179869184 |

Order 1 is the table top to bottom (oldest rung first) x 5, order 2 bottom to top x 5: N=5 per arm per order, 120
runs. Every adjacent pair a day compares is interleaved in both orders, as in its own cell.

**Reading each rung's clauses on this card.** `day52-views.py` builds, for each day 43 to 50, a view holding only
that day's arms under that day's arm names (symlinks to the ladder's logs and exit files, the ladder's marks renamed
the same way) and runs that day's reader unchanged with `--rig pro-single`; nothing in a reader, a clause or a bound
changes. The views:

| day | reader | day's arm <- ladder arm |
|---|---|---|
| 43 | `day43-resid.py` | off, base, i6d, i6g |
| 44 | `day44-mapped.py` | off, i6 <- i6g, i9 <- i9g |
| 45 | `day45-fill.py` | off, i9g, fill |
| 46 | `day46-nodrain.py` | off, fill, i1 |
| 47 | `day47-pinned.py` | off, i1, i2 |
| 48 | `day48-small.py` | off, i2, i8, i5 |
| 49 | `day49-install.py` | off, i5, i7 |
| 50 | `day50-prefetch.py` | off, i7, i4 |

Three differences from the 5090 cells, registered here and stated with every target reading:

- **One `off` arm** (the final binary with no door) serves every view. No rung changes the legacy program: I9's
  load option is set only when the door is parsed, I4's call site adds the door's own flag (false without the
  door), I1 returns `admit_native` to the legacy program; the ladder's integrity (one tape and one STEADY-STATE line
  across all 120 runs, checked by every view) holds this to the bytes.
- **I6's budget arm at 16 GiB**, the whole bank (`DAY43.md`: "its budget arm is registered there with that host's
  `MemAvailable`"); the box is required to hold it (section 3).
- **Day 44's pair at 16 GiB** (I6G against I9G, not the default budget): I9 changes where the loaded expert banks
  live, not the tier, and 16 GiB is the budget every later rung runs at.

**What the target card decides.** Each view prints that day's clause lines with `rig=pro-single`: they are this
card's readings of the rungs, as days 43 to 50 register them. A rung the 5090 kept whose clause fails here is
reported by name as a target-card reading; it is not reverted from the tree the deciding cell runs (that tree is
fixed in `DAY51.md` section 2 from the 5090 verdicts), and it becomes a per-card question in the promotion work
(`OWED.md` C2) if the door wins here. The door's verdict on this card is `decide`'s, by `DAY51.md`'s rule; a
failed G1, G2 or G3 voids it.

**Expected duration**, from day 18's target-card marks (a door run 87 s with the serial install, a legacy run 5.4
s) and day 18's box build (3 min 12 s incremental): builds about 45 minutes (eleven labels, the first cold); `attrib`
about 35 minutes; `ladder` about 2.5 hours (nine arms with the serial install at about 90 s, three at about 20 s);
`hashlock` about 2 minutes; `spec` about 20 minutes (day 11's eight-slot run took 298 s); `decide` about 10 minutes.
About 4.5 hours in all; the driver's cell timeouts are 2 h, 4 h, 30 min, 1 h and 1 h.

## 2. The binaries

Every rung is built on the box from the commit that built it on the RTX 5090 host (the 5090 builds, for reference;
a box build of the same commit gets its own SHA-256, recorded in `builds.log`):

| label | commit | 5090 `run-gen` SHA-256 |
|---|---|---|
| `base` (day 40) | `08210a291` | `b73bb4d36da77d5ca802f9307db1672d00d62db529b46e374b51fc7ca0392186` |
| `i6` (with I6 (e)) | `51467de12` | `771ba66bfd01765a9b645903209641f538cee6effbd92b30179b2d9286dd1783` |
| `i9` | `c2d78fb9c` | `878aa1ff8ec08f4c52835a47ae78f171d1fa0aaa28cd33988a4f25335e61cccc` |
| `fill` | `ef7db702e` | `19bcfd591b18176bf56a9e51e9c6c362e7138f40b02cb89e34c8a58a3c2a20ca` |
| `i1` | `a6258a8f0` | `0d90e124de8aa4c569a2da40c5edb3c963b5064ca529d327b479d9b235b17e55` |
| `i2` | `10a1c30df` | `974a7e4bbe676ed25e02164f1d79d1345792500d07c2eccfbe2d5a24a15d17a7` |
| `i8` | `cd49c8bcb` | `308dad3aca5b008ac61ebf8178631f0fb72c9d24482937c47c314a5e2a041747` |
| `i5` | `ae5237e6c` | `2b5a738edcc7c33b9b0c5b721d014108269bf12cdcc121f4112015ac0153335b` |
| `i7` | `14b2b9970` | `3eab35aa74dcabe5ffdd9a0faf290517550d963752ac84d01c88e303257ed561` |
| `i4` | `6745fd062` | `f14df194718fd79ae1066ef2c886c5ffa89baf53c13e4ce43e92454bce1bf7b6` |
| `final` | named in `DAY51.md` section 2 | (`run-spec` on the 5090 at the same crates: `6ad396299ca2fb44827e9917cb7dc15c77512e558088380b45f19365d9ad5157`) |

Each 5090 binary was built by `cargo build --release -p memra-engine --bin run-gen` right after its commit, with
the crates committed (the build commands and their printed SHA-256 are in this lane's session record; the cells bank
each binary's SHA-256 in `ev/binary.sha256`). `D52_BUILDS` for the driver is the label=commit list above plus
`final=<commit>`. The build script's logic was dry-checked with a stand-in `cargo` (`day52-cpu/dry-check-build.log`:
detached checkouts of `08210a291`, `51467de12`, `6745fd062`, the clean-tree check, the copies and names).

## 3. What the box needs

- **Card.** One RTX PRO 6000 Blackwell Workstation Edition, no other compute app during the cells (the runner waits
  for none, bounded, and never signals one). The lead's box acceptance (the clock check for a latched power brake)
  before staging.
- **Host.** At least 96 GB RAM (the runner waits for 48 GiB `MemAvailable` before each cell: a 16 GiB pinned tier
  or the legacy 15 GB of pinned slabs beside the 18 GB page-cached artifact), at least 16 cores (the fill and the
  record pass use `min(8, cores/2)` threads under the 12-core cap), local NVMe with at least 120 GB free (both
  artifacts, two worktrees with a release target each, receipts).
- **Software.** CUDA 13.x with `nvcc` (day 18's box: 13.2, `MEMRA_CUDA_ARCH` auto-detects 120a), the repo's Rust
  toolchain, `python3`, `git`.
- **Artifacts**, staged to `/root/artifacts/` on local NVMe and checked by the driver before anything builds:
  - `Qwen3.6-35B-A3B-UD-IQ4_XS.gguf`, `unsloth/Qwen3.6-35B-A3B-MTP-GGUF@5bc3e238d916f48a861bac2f8a1990a0e9b7e98d`,
    18,209,036,576 B, SHA-256 `df27a780435b7b45c2597536112ea3cb091f8544c3d0c3318d9f4258b31f7adf` (the approved
    artifact; this lane's `download.sh` pattern: resumable fetch of that revision, `sha256sum -c` before the move).
  - `Qwen3.8-27B-NVFP4-Q5K-mtp.gguf`, `tiyuvta/Qwen3.8-27B-NVFP4-MTP-GGUF` file `Q5K-mtp`, 15,705,922,304 B, SHA-256
    `1facf36c2db359dcf9c2475cf8f85fe84a528d10aaaaff20f7c0db3d561e024a` (G1's non-approved artifact, day 18's).
- **Worktrees.** `/root/wt-c`: a clone of the repo at `lane/spill-c-20260919`, at the tip named in `DAY51.md`
  section 2 or later (the sitting tree: scripts, collector, readers); `/root/wt-c-build`: `git -C /root/wt-c
  worktree add --detach /root/wt-c-build` (the build script checks out each commit there).
- **Run.** `D52_BUILDS="base=08210a291 i6=51467de12 i9=c2d78fb9c fill=ef7db702e i1=a6258a8f0 i2=10a1c30df
  i8=cd49c8bcb i5=ae5237e6c i7=14b2b9970 i4=6745fd062 final=<commit>" bash
  /root/wt-c/research/spill-c-20260919/day52-box.sh` (restartable: a finished build list and each finished cell are
  skipped). Receipts land in `/root/spill-receipts/c-day52/` and come back to
  `research/spill-c-20260919/pro-single-day52/` with the driver, build, provenance, validate and reading logs.

## 4. Added before the sitting ran: the verify digest v3 gates (`OWED.md` C6, `DAY53.md`)

`DAY53.md` section 1 registers the target card's verify digest v3 cells as "the same failure and identity cells on
the 27B, added to the DAY52 sitting as its own section before that sitting runs". After `decide`, the driver runs
`day53-cell.sh` (the day-24 driver shape) with the non-approved 27B artifact (it is a serving model for these gates;
the approved-artifact lock is the MoE door's alone), the device prefix budget 256 MB (the target card's day-23 gate
shape: one 64-token entry of about 160 MB fits, two do not) and `MEMRA_GPU_LOCK=/tmp/memra-gpu.lock`, in order:
`unit-server` (the GPU cell `verify_digest_v3_covers_every_round_tripped_plane_and_v2_stays_trunk_only`),
`failure-default-off`, `failure-plain-off`, `failure-default-on`, `identity-default-off`, `identity-default-on`.
Acceptance is `DAY53.md` sections 1 and 1a on this card. The box builds one more label, `server=<commit>` (the
`memra-server` of the verify digest v3 tree, `256c3c640` or the later commit named when the 5090 cells are read),
and the sitting grows by about 30 minutes (six boots of the 27B per failure run, two per identity run, one test
build), about 5 hours in all.

## 5. Added before the sitting ran: the double-park slice cell (`OWED.md` C4, `DAY54.md`)

After section 4, the driver runs `DAY54.md`'s cell `slices` in its own collector hold (timeout 3 h, about 85
minutes expected, the 27B): eight `memra-server` binaries built as labels `srv-e0=0713c1a79 srv-s1=ff64e7f5d
srv-s2=da1f59bf6 srv-s3=226abab0e srv-s4=5df11152f srv-s5=58b814abe srv-s6=f661406e4 srv-s7=269ef2cec` (named
`memra-server-e0` to `-s7`; the build list grows by about 40 minutes, `e0` the farthest back). Reader
`day54-slice-reading.py`. The sitting is then about 7 hours in all.

## 6. Added before the sitting ran: the DFlash tail class cells (`OWED.md` C5, `DAY56.md`)

After section 4, the driver runs `DAY56.md`'s two cells (`day56-cell.sh identity-dspark-off` and `-on`, the 27B,
256 MB device prefix budget, `MEMRA_GPU_LOCK=/tmp/memra-gpu.lock`) on the server built as label `server-c5=<commit>`
(the tail class tree, `1b130f1ef` or the later commit named when the 5090 cells are read), then `day56-reading.py`.
The box stages the DFlash2 drafter export at `/root/artifacts/q38-dflash2/` (`config.json` SHA-256
`873e3556509b0da06e29654ba00d4944888d4b5e8a33afde25f7eb27d321e980`, `model.safetensors`
`67fc76d68dc5a9415511a4f394ef744d67510cd20e93b37cc2cc7d28e4bab65c`, 3.8 GB, `DAY20.md`'s manifest; the driver
records both before the cells). About 10 more minutes (four boots of the 27B with the drafter), one more build.

## 7. Added before the sitting ran: improvement I10 in the ladder (`DAY57.md`)

The ladder grows a thirteenth arm, `i10` (`run-gen-i10`, the I10 tree `70d6633f5`, 16 GiB, the stage clock), last in
order 1 and first in order 2: 130 runs, about 12 minutes more. `day52-views.py` grows a day-57 view (`off, i4,
i10`, reader `day57-fillwait.py`). The build list grows `i10=70d6633f5`, and `final` in `DAY51.md` section 2 is the
tree after I10's 5090 verdict.

## 8. Added before the sitting ran: the order of the cells, and the ladder's `off` arm

- **The ladder's `off` arm** is `run-gen-i10` with no door (the newest rung binary; the legacy program, which no rung
  changes, section 1), not `run-gen-final`, so the ladder does not wait on `DAY51.md` section 2.
- **The order**: `attrib`, `ladder`, section 4 (C6), section 6 (C5), section 5 (C4), then `DAY51.md`'s `hashlock`,
  `spec` and `decide`, which run only when `final` is in the build list (the driver says so otherwise and stops
  there; a rerun with `final` added resumes at those cells, every finished cell skipped). So the box can start before
  the RTX 5090's rung verdicts are in, and the final tree is named before its three cells run.
- **The build list**, all sections: `D52_BUILDS="base=08210a291 i6=51467de12 i9=c2d78fb9c fill=ef7db702e i1=a6258a8f0
  i2=10a1c30df i8=cd49c8bcb i5=ae5237e6c i7=14b2b9970 i4=6745fd062 i10=70d6633f5 server=256c3c640
  server-c5=1b130f1ef srv-e0=0713c1a79 srv-s1=ff64e7f5d srv-s2=da1f59bf6 srv-s3=226abab0e srv-s4=5df11152f
  srv-s5=58b814abe srv-s6=f661406e4 srv-s7=269ef2cec"`, plus `final=<DAY51 section 2 commit>` when named (21
  builds, about 90 minutes, the first cold). The sitting is then about 7.5 hours with everything.
