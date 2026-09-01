# step37 NVFP4 expert-bank re-derive — milestones 1-5 CLOSED

Lane `lane/step37-bankv3-20260901`, cut from a freshly fetched `origin/main` at `d1e9eab57`.
Mandate: re-derive the slot-major coalesced NVFP4 expert-bank layout and its reader as one unit,
gated on the byte-identity oracle the original door asserted in a comment and never had, then
rebuild the down8 selector on top of it. The prize is the removed doors' measured **-21.5% wall /
-23.7% decode** (`research/perf-chain-20260831/` cell 1: 157.60 vs 120.25 decode tok/s, x5
interleaved, acceptance 0.970 vs 0.935).

## Headline

**The layout was never the defect, and it does not need re-deriving. One caller did.**

`DIAGNOSIS.md` has the archaeology. The host permutation is a genuine pure permutation and every
kernel written to read it reads it correctly. The corruption was a **defaulted argument**:
`kq_fetch(wrow, k0v, s_cb, int in_f = 0)` in the prefill grouped GEMM, with exactly two of ten
call sites — the `kb+1` software-prefetch pair inside `moe_kq_sktail_kernel` — silently taking the
default. `in_f` is read by **only** the `QT_NVFP4_V2` branch, to locate the slot-major row's UE4M3
scale tail at `n_slots*16`; at `in_f = 0` the scale byte comes from inside the packed-codes region
while the codes stay correct. Right weights, wrong per-16-element scale, on every k-block but
`kb = 0`.

So the mandate's ordering inverts. There is no layout to re-derive; there is a **gate to build and
a restore to gate**. Milestones 1 and 2 are closed and the defect is fixed. Milestones 3-5 are
scoped below with the restore inventory and the revised risk, which is materially lower than the
brief assumed.

## Milestone 1 — archaeology and corruption diagnosis. CLOSED.

`DIAGNOSIS.md` (commit `381d492e3`). Summary of what it settles:

- The byte map is verified by hand, both codes and scale bytes, against the v1 reader.
- Seven readers audited; six correct, one correct-in-body-and-fed-wrong.
- The mechanism reproduces the whole 2026-08-29 fingerprint, including the two facts the incident
  could not explain: divergence "already in the prime's logits" (it is a prefill kernel) and
  "MARGIN-dependent, not length-dependent" (a wrong scale on correct codes is a bounded
  perturbation, so the corruption was equally present at 613 tokens and merely invisible).
- Three instruments were blind, each a gate-craft lesson: the layout test is host-side; the
  "host-canonical oracle" branched on the flag under test and compared v2 against v2; no v2 gate
  ever ran the prefill GEMM.
- The bisect could not isolate the mechanism: `sel_down8` hard-refuses without the v2 banks and
  the `_sel_v2_gu` fusion auto-arms on the same predicate with no door of its own, so one env var
  moved three program elements.
- The defect **class** — a parallel qtype tag whose every predicate, size macro and argument list
  must be updated in lockstep — has now failed twice (the `KQ_CB_WORDS` `s_cb[1]` shared overrun
  of `068cbc425` being the first, and it was already in the incident tree, so it is not the
  incident's mechanism).
- Four still-unconverted v1-reader sites and one `in_f % 64` hole in the permute are listed so the
  restore does not walk into them.
- Six concrete rules for what "layout and reader as ONE unit" has to mean.

### Live-on-main correctness finding

`fd0a175ab` deliberately kept `QT_NVFP4_V2` alive for the always-slot-major **EP2** banks and the
`moe-tp2-repro` harness. Both defaulted call sites were unchanged at `d1e9eab57`, so **EP2 prefill
has been reading wrong scales on `main`**, independent of the removed door. Fixed in `1b18a61e8`.

## Milestone 2 — oracle infra. CLOSED for gate (a); (b) scoped.

New bin **`nvfp4-bank-oracle`** (`crates/memra-engine/src/bin/nvfp4_bank_oracle.rs`), the
device-side bank oracle `75bf4ce76` named as its follow-up lane and never got. It is a
differential exactness gate, not a tolerance band:

1. one pseudo-random block_nvfp4 bank is the **only** source of values;
2. v2 is produced by the **shipping** `tp::nvfp4_matrix_v2_permute` (now `pub` and documented as
   the single source of truth for the byte map), not a reimplementation;
3. the same activations and the same CSR go through the **same** entry (`Engine::moe_f16_grouped`)
   as `(v1, QT_NVFP4)` and `(v2, QT_NVFP4_V2)`;
4. the f32 outputs must be **bit-identical** — a permutation read correctly cannot move a bit.

Built against the three blindnesses that let the corruption ship: the oracle arm is pinned to v1
bytes and the v1 qtype so it is never a function of the layout under test; it runs the **prefill**
GEMM; and it refuses to report green on a vacuous comparison — an identity permutation, an
all-zero v1 output (a sum of zeros is order-independent regardless of accumulation order), or a
non-finite arm each fail loudly. Scale bytes are drawn from a band disjoint from the code bytes on
purpose, so reading a code **as** a scale cannot pass by luck. Both step37 layer geometries run,
because gate/up and down differ in row alignment and neither stands in for the other, and both
give `nkb > 1` so the prefetch that shipped broken is actually exercised.

Tile-form coverage is driven from the caller, one arm per process, because the form policy is a
`OnceLock`.

### Both arms, on the 2x PRO 6000 bench box

Two binaries built from one checkout, four minutes each, distinguished by md5. The pre-fix arm is
an exact textual revert of the two call sites plus the restored `= 0` default, applied, built,
then reverted (`git status` clean after).

| arm | tile form | binary md5 | gate_up 4096x640 | down 640x4096 | exit |
|---|---|---|---|---|---|
| **FIXED** hybrid (`cross=64`) | deep tail (all groups, `max_m=41 < 64`) | `073b2d5f…` | **DIFF=0** over 327,680 | **DIFF=0** over 2,097,152 | 0 |
| **FIXED** `MEMRA_F16G_SK=128` | 128x64x64 3-stage | `073b2d5f…` | DIFF=0 | DIFF=0 | 0 |
| **FIXED** `MEMRA_F16G_SK=32` | 32x64 tail | `073b2d5f…` | DIFF=0 | DIFF=0 | 0 |
| **FIXED** `MEMRA_F16G_TAIL=0` | 2-stage tail rollback | `073b2d5f…` | DIFF=0 | DIFF=0 | 0 |
| **PRE-FIX** hybrid | deep tail | `832fa62a…` | **DIFF=327,680 (100%)** maxrel 5.588e1 | **DIFF=2,097,152 (100%)** maxrel 6.718e1 | 1 |
| **PRE-FIX** `SK=128` | 128 form | `832fa62a…` | DIFF=0 | DIFF=0 | 0 |
| **PRE-FIX** `SK=32` | 32x64 tail | `832fa62a…` | **DIFF=327,680** maxrel 5.588e1 | **DIFF=2,097,152** maxrel 6.718e1 | 1 |
| **PRE-FIX** `TAIL=0` | 2-stage tail | `832fa62a…` | DIFF=0 | DIFF=0 | 0 |

Every arm reports `nonzero_v1 = elems` and `finite = true`, so no row is a comparison of zeros.
Raw: `raw/oracle-fixed-4arms.log`, `raw/oracle-prefix-4arms.log`,
`raw/build-and-identity.log`. Both cards verified 0 MiB before and after every run.

**The pre-fix arm pair localizes the defect instead of asserting it.** `SK=128` and `TAIL=0` pass
on the *broken* binary and the deep-tail arms fail on it, which is exactly and only
`moe_kq_sktail_kernel` — the two other in-file tile forms (`moe_kq_sk32v_kernel` at lines
1197/1211 and the sk128 form at 1311-1356) pass `in_f` and are clean. It also records, for free,
that **`MEMRA_F16G_TAIL=0` was a latent rollback seam that would have dodged the entire
incident** — nobody knew, because nothing tied the door to the arm.

Toolchain and identity: `rustc 1.98.0`, CUDA 13.2 (`nvcc V13.2.51`), 2x RTX PRO 6000 Blackwell
Server Edition 96 GB at 600 W, lane sha `1b18a61e8`.

### Why the model stayed fluent while every routed-expert FFN in prefill was wrong

The oracle shows 100% of output elements deviating by up to ~67x relative on random data, which
looks irreconcilable with the incident's *fluent* wrong answers. It is not, and the reason is a
step37 property, not a general one: `quantization_config.ignore` keeps `lm_head`, layers 0-2, and
**every** layer's `self_attn*`, `moe.gate` and `share_expert*` out of NVFP4
(`QUIRK:step37:quantization-ignore-list`). Attention, the router and the shared expert were
therefore never touched by this bug. The corruption was confined to the routed-expert FFN
contribution, on top of an intact residual stream, an intact router and an intact shared expert —
which is precisely the shape that produces confident, well-formed, **wrong** text rather than
gibberish. Do not generalize this to a model whose attention is quantized.

### Gate (b), end-to-end, is scoped but not run

Greedy byte identity v3-ON vs OFF on real prompts, plus the short-prompt output-content oracle
from the degen re-run lane. It is deliberately **not** run yet, because at this point there is no
v3 arm to compare: the door is deleted and the fix restores the documented contract rather than
adding an arm. It becomes the milestone-3 admission gate, and per the diagnosis it must use
short, margin-sensitive prompts — the 613-token gates passed on a corrupt binary.

## What this changes about milestones 3-5

The brief assumed the decode prize sat on an unproven layout. It does not.

- **The -23.7% decode prize is attributable to the decode sweep family** — `qmatvec_nvfp4_dp4a_v2`,
  `_sel_v2`, `_sel_v2_gu`, `_sel_v2_down8` and `_down8_rows` — every one of which passed the
  static audit in `DIAGNOSIS.md` (byte map correct, same dp4a and scale order, same reduce tree,
  launcher-enforced `nsb <= 32` and `n_sel <= 8`). The defect that shipped is in a **prefill**
  kernel and cannot move a decode tok/s number. The restore is therefore much lower risk than the
  incident implies.
- **Do not assume the layout is the whole win.** Cell 1 toggled `MEMRA_NVFP4_BANK_V2` and
  `MEMRA_SEL_DOWN8` together, and the `_sel_v2_gu` gate+up fusion auto-arms with the bank with no
  door of its own. At least three program changes are inside that 30 tok/s: coalesced 512B row
  reads, one launch instead of two for gate+up, and the fused down+combine that deletes an
  `n_sel x out_f` partial round trip. Milestone 4 must price them **separately**, each behind its
  own flag, or it will re-bank an unattributable number.
- **Each restored arm gets its own door.** The single-door-three-programs coupling is what made
  the 2026-08-29 bisect unable to name a mechanism. Layout, `gu` fusion and `down8` are three
  flags.

### Restore inventory for milestone 3

Everything needed is recoverable from `fd0a175ab^`; nothing has to be rewritten from scratch.

| piece | source | notes |
|---|---|---|
| TP-bank slot-major permute arm | `tp.rs` (`nvfp4_repack_bank_matrix`) | already `pub` and `in_f % 64`-asserting on this branch; only the TP call sites need the flag |
| `sel_v2` / `sel_v2s` decode kernels | `cu/qmatvec.cu` @ `fd0a175ab^` | audited correct |
| `sel_v2_gu` family (+`_wpr`/`_rpw`/`_r2`/`_r4`) | same | give it its own flag; `_wpr` is a numeric-class arm and must stay separately gated |
| `down8` / `down8_rows` | same | milestone 5, own flag |
| launchers | `lib.rs` (`qmatvec_nvfp4_sel_*_into`) | drop the `nvfp4_bank_v2_on()` refusal in favour of the arm's own flag |
| the two-column/t-row verify program | `hybrid_forward.rs`, `spec.rs` | `MEMRA_TCOL_FFN` was **not** in the 140-era serving env; out of scope for the prize |
| `removed_bank_v2_doors_refusal` | `crates/memra-server/src/worker.rs` | must be retired in the same PR as the new flags, or the server refuses to boot with them |

New-flag law: every restored arm ships default **OFF** with its `docs/FLAGS.md` row, both arms'
receipts and the rollback seam in the same PR. No flag was added by this lane's two commits, and
no default changed — the fix restores the layout's documented contract, and the oracle is a bin.

### Milestone 4 protocol, pinned now so it is not negotiated later

Per `measurement-laws.md`: interleaved fresh boots x3, escalated to x5 when either amendment rule
fires (within-arm spread of the decision median > 0.5%, or verdict within 2x pooled spread), every
arm reporting its spread and every escalation naming the rule. Arm identity binds on **binary
md5** plus a marker test, plus a `BOOT_NONCE` read back from `/proc/<pid>/environ` and
`readlink /proc/<pid>/exe`; `system_fingerprint` is trustworthy on this branch (it is cut from
`d1e9eab57`, the commit that invalidates the baked sha when HEAD moves) but md5 stays the binding
identity. Real prompts, capped `max_tokens`, greedy for the byte gates only, vendor-default
sampled twins for any serving-shaped row, loops flagged and excluded from aggregates with the
exclusion stated. Every door the cell is about proven set or proven absent **from `/proc`**, not
merely intended.

## The perf-CI push gate, and why it is overridden with a receipt rather than a shrug

`tools/hooks` pre-push refuses engine-file changes without a `tools/local-ci.sh --perf` battery.
That battery is a rig-local drift tripwire against a cross-day tok/s median, over gemma Q4_0 and
Qwen NVFP4 **GGUF** cells (`research/tune-data/perf-cells.json`) — models this box does not hold,
and a rig run would produce timing numbers on a thermally throttled laptop 5090, which the
rig-GPU-exactness-only law forbids as a measurement.

So the claim the gate exists to protect — "this diff does not move perf" — is proven
**structurally**, which is stronger than a tripwire anyway. Both trees' `cu/moe_f16_grouped.cu`
were compiled to PTX with the build's own flags
(`-gencode arch=compute_120a,code=sm_120a -O3 -std=c++17 --expt-relaxed-constexpr`, nvcc 13.2):

- 39 kernel entries in each, identical name sets, identical 38,264-line PTX length;
- **38 of 39 entries byte-identical**;
- the single differing entry is `_Z20moe_kq_sktail_kernelILi107EE…` — the `QT_NVFP4_V2` (=107)
  instantiation, and nothing else. Its diff is register renumbering and the scale-address
  computation, at the same instruction count.

`QT_NVFP4` (=7) is what every perf cell exercises, and its instantiations did not move a byte, so
no cell in the manifest can observe this change. Receipt: `raw/ptx-neutrality.log`.

The push therefore carries `MEMRA_SKIP_PERF_CI=1` **knowingly**, with that receipt as the reason.
Milestone 4 is where this lane earns real perf rows, under the interleaved protocol pinned above —
not on a laptop card and not against a cross-day median.

`cargo test --release -p memra-engine --lib` on the bench box: **290 passed, 0 failed, 2 ignored**,
including the pre-existing `tp::bank_v2_layout_tests::the_v2_bank_row_is_the_documented_slot_major_permutation`,
which still passes with the permute made `pub` and the new `in_features % 64` assert.

## Housekeeping

- `tools/check-public-boundary.py`: **685 matches, 685 grandfathered, 0 new.** No allowlist entry
  was added by this lane.
- `cargo fmt --all -- --check` clean; the pre-commit hook ran and was satisfied on both commits.
- Box state: both cards drained to 0 MiB. Everything this lane created lives under
  `/home/ubuntu/bankv3/`; nothing outside it was written, and the shared model directory was
  never touched (this lane needs no model — the oracle is self-contained, which is part of why it
  is cheap enough to run on every change).
- No stale repack cache was involved: the oracle builds its bank in-process from a seeded RNG and
  reads no `.memra-repack` directory. `find / -maxdepth 6 -name '.memra-repack*'` returned nothing
  on this box, so the door-era 100 GB cache is already gone with its ephemeral NVMe.
  **CORRECTION, 2026-09-01 (milestone 4):** that last sentence was wrong, and the search depth is
  why. The cache is very much present, at
  `/opt/dl-image/nvme/data/models/step37-flash-nvfp4/.memra-repack` (100 GB, mtime 2026-08-28 20:10),
  which is 7 levels down and behind the `/data` symlink `-maxdepth 6` never followed. The 126 GB
  step37 checkpoint sits beside it. Nothing in milestone 2 read either, so no milestone-1/2 receipt
  is affected — but the conclusion "already gone" was a false negative from a too-shallow find, and
  milestone 4 needs both. Its provenance is verified in the milestone-4 section below.

## Milestone 3 — the bundle comes back as three doors. CLOSED (commit `09491a5df`).

### What was restored, and what deliberately was not

Everything came from `fd0a175ab^`, rebased onto `d3ac87f80` (which carries the `in_f` fix). Eight
CUDA kernels were lifted with their own comment blocks and then **verified byte-identical to their
pre-removal source, per kernel**, by extracting each `extern "C" __global__` body from both trees
and comparing: `_sel_v2`, `_sel_v2s`, `_sel_v2_gu`, `_gu_r2`, `_gu_r4`, `_gu_wpr`, `_down8`, plus
the `nvfp4_sel_v2_gu_rpw_body<RPW>` template they instantiate. A restore that silently retypes a
reduce tree is a different program wearing the same name, so this is a check and not a claim.

Two pieces were NOT restored, with reasons rather than omissions:

| not restored | why |
|---|---|
| `qmatvec_nvfp4_dp4a_sel_v2_down8_rows` + `qmatvec_nvfp4_sel_down8_rows_into` | its only caller was `run_tensor_parallel_routes_nvfp4_device_routed_tn`, the two-column/t-row verify program. That program is out of this lane's scope and `MEMRA_TCOL_FFN` was **not** in the 140-era serving env, so it cannot be part of the prize. A restored kernel with no live launcher is dead code, not coverage. |
| `qmatvec_nvfp4_dp4a_sel_v2_gu_tcol` | same program, same reason. |

### The three doors, and why three

| door | program | default | armed by |
|---|---|---|---|
| `MEMRA_NVFP4_BANK_SM` | slot-major TP expert banks + the `_sel_v2` readers | **OFF** | its own strict `0`/`1` |
| `MEMRA_NVFP4_SEL_GU` | fused gate+up sweep, one launch instead of two | **OFF** | its own strict `0`/`1` — **its first door ever** |
| `MEMRA_NVFP4_SEL_DOWN8` | fused down+combine, one warp per slot | **OFF** | its own strict `0`/`1` |

Plus three sub-doors, all default OFF and all explicitly **UNPRICED**:
`MEMRA_NVFP4_SEL_SM_STREAM` (the 8-row streaming twin), `MEMRA_NVFP4_SEL_GU_RPW` (2|4 multirow,
bit-identical) and `MEMRA_NVFP4_SEL_GU_WPR` (warp-per-row, **NUMERIC-CLASS**: the per-row reduction
order changes, so a bit tape cannot apply and acceptance is the argmax gate plus the boot battery).
`WPR` is kept out of this lane's priced arm set on purpose — every priced arm here is bit-gateable,
and admitting a numeric-class arm into the same battery would mix acceptance classes.

The reason the count is three and not one is the whole mandate. `fd0a175ab`'s own message lists what
one env var armed; `DIAGNOSIS.md` shows the consequence: `sel_down8` **hard-refused** without the v2
banks and the `_sel_v2_gu` fusion **auto-armed on `nvfp4_bank_v2_on()` with no door of its own**, so
the 2026-08-29 bisect could name a door and never a mechanism, and the perf chain's −21.5% wall /
−23.7% decode was a bundle price. Three doors make it three numbers.

### Layout stopped being an environment question

This is the milestone-1 lesson turned into structure, and it is the part of milestone 3 that is not
a restore. `DIAGNOSIS.md` rule 1 asked for layout geometry that travels with the pointer and cannot
be defaulted. Concretely, now:

- `bank_slot_major_on()` is read **once, at bank build**, and the answer is recorded as
  `slot_major` on `ResidentNvfp4ColumnBankRank` / `ResidentNvfp4RowBankRank`. One decision per
  bank, stored beside the bytes it describes.
- `qmatvec_nvfp4_sel_into` takes `slot_major` as a **required argument with no default**. The
  `_gu` and `_down8` launchers take it too and **refuse** rather than reading v1 bytes as
  slot-major; the removed implementation refused on `!nvfp4_bank_v2_on()`, i.e. on the environment,
  which can disagree with what is resident.
- the grouped prime derives `bank_qt` from the banks and **refuses outright** if gate/up/down
  disagree on the layout: one grouped GEMM cannot serve two byte maps.
- all four host-canonical oracles go through a single `host_canonical_expert` per bank type. One
  place maps layout → reader, so a new producer cannot leave a reader behind.

**Two live bugs fell out of that last point.** `run_column_bank_expert_nvfp4` and
`run_row_bank_expert_nvfp4` hardcoded the v1 reader (`qmatvec_nvfp4_fast`). Under
`MEMRA_NVFP4_BANK_SM=1` they would have read slot-major bytes with the v1 kernel — the exact
v1-reader-on-v2-bytes shape the grouped-prime receipt comment already warns about, in the
*oracles* this lane's byte gates would have been compared against. They were latent, not
shipping (nothing built slot-major TP banks after `fd0a175ab`), and they are fixed.

### The boot refusal is RESCOPED, not retired

The mandate said retire the 75bf4ce76 refusal in the same change, on the reading that it refuses
the env var being restored. It does not: the restore uses three new names. So it stays, and it
keeps refusing `MEMRA_NVFP4_BANK_V2` / `MEMRA_SEL_DOWN8`, for a reason worth stating plainly —
**a 140-era recipe asks for a three-program bundle by a name no code reads.** Booting it as "just
the layout" would serve one third of what the recipe requested while every counter stayed green,
which is the incident's own failure shape. The message now names all three successors so the recipe
is corrected in one pass, and `the_removed_bank_v2_doors_cannot_boot_a_server` asserts that it
does. A second test pins that the new names are not themselves caught.

If the owner prefers the literal instruction, deleting the guard is a two-line change — but it
should be a decision, not a side effect of restoring the programs.

### Milestone-3 gates

- `cargo test --release -p memra-engine --lib`: **290 passed, 0 failed, 2 ignored**, including
  `bank_v2_layout_tests::the_v2_bank_row_is_the_documented_slot_major_permutation`.
- `cargo test --release -p memra-server --lib bank`: **2 passed** (the rescoped refusal and the
  new not-caught-by-the-legacy-guard test).
- `cargo fmt --all -- --check` clean; `cargo clippy -p memra-engine --lib` adds no new warning
  (the tree's 777 are unchanged).
- `tools/check-public-boundary.py`: **685 matches, 685 grandfathered, 0 new.** No allowlist entry
  added. The milestone-4 prompt corpora are deliberately NOT in this repo — see below.
- `docs/FLAGS.md`: six rows, one per new flag, each stating its default as a decision with both
  arms, the rollback seam and the gate battery. The `MEMRA_NVFP4_BANK_V2` removed-door row is
  **corrected**: it recorded the layout's bit-identity claim as FALSIFIED, and that was wrong.

## Milestone 4 — gates then pricing, per program. Gate (a) CLOSED, gates (b)/(c) in progress.

### Where the model and the repack cache actually are, and why provenance was checked first

The bench box holds the 126 GB step37 NVFP4 checkpoint at
`/data/models/step37-flash-nvfp4` (a symlink into ephemeral NVMe) with a **100 GB
`.memra-repack` cache built 2026-08-28 20:10** — squarely the door era. The lane brief's standing
warning is that a stale repack cache must never serve a gate, so it was settled before any boot,
and by construction rather than by trust:

- the cache holds `repack_modelopt_to_gguf` output: **pure block_nvfp4 v1**, 36-byte superblocks.
  It has no dependence on any bank door — the slot-major permutation happens later, at TP bank
  build (`nvfp4_repack_bank_matrix`), entirely in host memory, and never touches this directory.
- `git log --since=2026-08-27 -- crates/memra-gguf/src/nvfp4_repack.rs` is **empty**; the routine
  last changed `33fde5599`, **2026-08-15**. The cache was written 13 days later, so its bytes are
  exactly what today's binary would produce.

Recorded as a latent trap rather than a problem here: `repack_cache_is_fresh` validates only the
**file length**. Any future change to the repack routine that preserves geometry would silently
reuse a stale cache. It did not happen this time; nothing prevents it next time.

### Gate (a) — the device-side bank oracle, extended to cover all three programs

`75bf4ce76` asked for a device-side bank oracle; milestone 2 built it for the **prefill** grouped
GEMM. That is necessary and not sufficient for this lane: the **−23.7% decode** prize lives in the
selected-experts sweep family, which the grouped GEMM never touches. So the bin gained a decode
section with **one arm per restored program**, every oracle arm pinned to **v1 bytes and the v1
kernel** — never to the layout or the fusion under test, which is precisely the mistake the original
"host-canonical oracle" made when it branched on the flag and compared v2 against v2.

Two anti-vacuity details worth stating, because a green light that compares nothing is what shipped
last time: P2 uses **two independent random banks** for gate and up (with one bank, a kernel that
read the gate rows for the up half would pass, and the comparator asserts the two oracle sweeps
actually differ), and every cell refuses on an identity permutation, an all-zero oracle output, or a
non-finite arm.

Binary `46a621157aa16d8e1f1031a9a197e1f7`, HEAD `09491a5df`, both cards 0 MiB before and after.
Raw: `raw/m4-oracle-4arms.log`.

| program / cell | gate_up 4096x640 | down 640x4096 |
|---|---|---|
| prefill GEMM, v1(`QT_NVFP4`) vs v2(`QT_NVFP4_V2`) | **DIFF=0** / 327,680 | **DIFF=0** / 2,097,152 |
| **P1** `_sel_v2` vs `_sel` | **DIFF=0** / 5,120 | **DIFF=0** / 32,768 |
| **P2** `_sel_v2_gu` gate half vs `_sel` | **DIFF=0** / 5,120 | **DIFF=0** / 32,768 |
| **P2** `_sel_v2_gu` up half vs `_sel` | **DIFF=0** / 5,120 | **DIFF=0** / 32,768 |
| **P3** `_down8` vs `_sel` + `axpy_rows_seq_md` | n/a (nsb=128 > 32, out of class) | **DIFF=0** / 4,096 |
| guard: `_gu` launcher refuses non-slot-major bytes | asserted | asserted |

Repeated across **all four prefill tile forms** — the shipping hybrid (`cross=64`),
`MEMRA_F16G_SK=128`, `MEMRA_F16G_SK=32`, and the `MEMRA_F16G_TAIL=0` 2-stage rollback — for
**16 green cells and 4 clean exits**. Every cell reports `nonzero_oracle = elems` and
`finite = true`, so no row is a comparison of zeros. The P3 skip is stated by the instrument itself
rather than silently omitted: `nsb <= 32` is the launcher-enforced fit-block class the reduce
identity is argued at, which the gate/up geometry (`in_f=4096`, nsb=128) is outside by construction.

### The milestone-4 harness, and what it inherits

`harness/` is descended from `research/perf-chain-20260831/harness/`, deliberately: `ERA_BASE`,
`POLICY` and `CURRENT_EXTRA` are **copied verbatim** from that lane, which took them from
toolchain-ab, which took them from the 140-era serving `agentic8.sh` `ENVV` line. A re-derived env
list is a different measurement wearing the same name, and the whole point of pricing against that
env is that it is the same env.

What it adds is the four **cumulative** arms — `v3-off` → `v3-sm` → `v3-sm-gu` → `v3-sm-gu-d8` —
and an expectation table asserted against the **live `/proc/<pid>/environ`**, so every arm proves
from the kernel which doors are set and which are absent. Cumulative rather than one-at-a-time
because programs 2 and 3 read slot-major rows and are **inert** without program 1: a
one-at-a-time sweep would price two no-ops. Each program's contribution is the delta to the arm
below it. The retired names are asserted ABSENT in **every** mode.

Arm identity binds on three things, not one: binary **md5**, a **marker test** (the binary must
contain all three new door strings, so a pre-milestone-3 binary cannot be an arm of this cell no
matter what the environment says), and a **boot nonce** read back from `/proc/<pid>/environ`
together with `readlink /proc/<pid>/exe`.

**Prompt corpora live in darklanes, not here** (`research/step37-bankv3-20260901/prompts/`, staged
to the box). They are real agentic turns and they carry home-directory paths and business content;
the private-content rule sends those to the private repo. Shapes, so this receipt stands alone:
`agentic8` = 8 real turns, 88–355 prompt tokens; `prefill-ladder` = one real conversation replayed
at 5 growing depths, **248 / 613 / 1480 / 3041 / 4555** prompt tokens.

### The perf-CI push gate, overridden again — and this time the neutrality is checkable

`tools/hooks` pre-push refuses engine-file changes without `tools/local-ci.sh --perf`. Milestone 2
overrode it with a PTX-neutrality receipt. The same override is taken here, with a stronger reason
and a mechanical check rather than an argument.

Why the battery cannot answer the question: `research/tune-data/perf-cells.json` is gemma-4-31B
**GGUF** (Q4_0, plain and spec) plus Qwen NVFP4 **GGUF**. Neither model reaches the step37 TP
expert-bank path at all — that path needs `MEMRA_STEP_TP` and a safetensors NVFP4 checkpoint with
stacked expert tensors — and running it on the rig would produce timing numbers on a thermally
throttled laptop 5090, which the rig-GPU-exactness-only law forbids as a measurement.

What is checked instead, mechanically, against `d3ac87f80`:

- **CUDA: all 304 kernels `main` already had are byte-identical.** Extracting every
  `extern "C" __global__` body from both trees and comparing gives **added 7, removed 0,
  changed 0**. The seven are the restored `_sel_v2` family. No pre-existing kernel moved a byte, so
  no cell in the manifest can observe a different instruction stream.
- **Rust: the doors-OFF launch is the same launch.** In `qmatvec_nvfp4_sel_into` with
  `slot_major = false`: `mode` is untouched (the override is `if slot_major { 3 }`), `sm_stream` is
  `false` because it requires `mode == 3`, the function `match (mode, false)` selects exactly the
  three functions `main`'s `match mode` selected, `fit_block`'s `(mode == 0 || mode == 3) &&
  !sm_stream && nsb <= 32` reduces to `main`'s `mode == 0 && nsb <= 32` for `mode ∈ {0,1,2}`, and
  `grid_dim` takes the same `match`. Every other restored path is behind `slot_major`, which is
  `ep2 || bank_slot_major_on()` — identical to `main`'s `ep2` with the door unset.

So `MEMRA_SKIP_PERF_CI=1` is carried KNOWINGLY, with that receipt as the reason.

**One gap in that argument is named rather than papered over, and it is closed below.** The byte
gate proves the four ARMS agree with each other; it does not by itself prove the OFF arm agrees
with pre-milestone-3 `main`. The FLAGS rows claim "OFF = byte-for-byte the current serving path",
and an unverified claim in a FLAGS row is the exact defect this lane exists to end. So the OFF
arm's tape is compared against a `d3ac87f80` binary's tape on the same corpus — see
"Gate (b0)" below.

### Gates (b) and (c) — the greedy tape and the short-prompt content oracle

Greedy is the instrument and never the product: these arms run spec **off** (with spec on the tape
depends on draft/verify scheduling, so the gate would be measuring the scheduler), and the priced
rows in the next section are vendor-default sampled.

Three shaping decisions, each traceable to a blindness the incident exposed:

1. **Prefill-heavy shapes are in the corpus, and they are the reason it exists.** Both defects the
   layout was blamed for lived in a prefill kernel and no v2 gate ever ran one. The ladder reaches
   4,555 prompt tokens.
2. **Short prompts are in the corpus and they are the margin instrument.** The corruption was
   equally present at 613 tokens and merely invisible: a wrong per-16 scale on correct codes is a
   bounded perturbation, so the argmax flips only where the top-1 margin is narrow. Every prompt in
   the original qualification was ≥ 613 tokens. The 25-token `17*23` probe that did catch it is
   `short0` and it stays there.
3. **Content is asserted, not just equality.** step37 is a thinking model whose bytes arrive in
   `reasoning_content`; a content-only reader sees two empty strings and reports PASS. Every cell
   must be non-empty, the arithmetic probes must contain the answer, and `compare.py` **refuses**
   on an empty cell, a differing cell set, a shared binary md5 or a shared boot nonce rather than
   calling any of them a pass.

**RESULT: all three programs PASS gates (b) and (c). Four fresh boots, one binary, 18 cells
each, every cell byte-identical.**

| pair | cells identical | differing | verdict |
|---|---|---|---|
| `GOFF` → `GSM` (**P1** layout + `_sel_v2`) | 18 | 0 | **PASS** |
| `GSM` → `GGU` (**P2** + fused gate+up) | 18 | 0 | **PASS** |
| `GGU` → `GD8` (**P3** + fused down+combine) | 18 | 0 | **PASS** |
| `GOFF` → `GD8` (all three at once) | 18 | 0 | **PASS** |

One tape hash for all four arms:
`d6fe97d0a1928e1b6c9d43985557189d41a3f75533e3da91faaff473d7624e68`. Binary
`b851b14a3647202acef67d796b59b115` in every arm — correct, and deliberate: the ENV is the axis
here, so one binary is what makes the comparison a statement about the doors. Four DISTINCT boot
nonces (`bv3-GOFF-1788197405-28209`, `bv3-GSM-1788197729-4939`, `bv3-GGU-1788197939-31943`,
`bv3-GD8-1788198175-7864`) prove four separate boots rather than one server re-probed, and
`compare.py` refuses a shared nonce for exactly that reason.

**Exactly one axis moved.** Diffing the four banked `/proc/<pid>/environ` censuses against each
other with the door names and the nonce/mode excluded returns **empty in all three pairs** — so the
41 other MEMRA_ vars, the model path and the registry are provably identical and the only variable
is the door under test. That is the receipt the 2026-08-29 bisect wanted and had to argue for.

Raw: `raw/m4-byte-gate-compare.log`, `raw/cell-GOFF.log`, `raw/cell-gates-on.log`,
`raw/boot-G*.receipt`, `raw/environ-G*.txt`.

Every banked receipt in this lane has its server `system_fingerprint` **SCRUBBED**, not
allowlisted: `memra-<12hex>` matches the `live_fingerprint` rule in
`tools/public-boundary-policy.toml`, which exists because a build fingerprint in a public receipt
correlates it to a live deployment. The pre-push boundary scan catches this where the standalone
`check-public-boundary.py` run did not, and the repo rule is scrub rather than grandfather. Nothing
is lost by it: arm identity in this lane binds on **`bin_md5` plus the marker test**, and
`built_from` carries the full commit sha in every receipt.

**Arm `GOFF` (`gate-off`, the OFF baseline) — 18/18 cells green.**
`ENV_VERIFIED mode=gate-off (MEMRA_NVFP4_BANK_SM! MEMRA_NVFP4_SEL_GU! MEMRA_NVFP4_SEL_DOWN8!
MEMRA_SERVE_SPEC=0 MEMRA_NVFP4_BANK_V2! MEMRA_SEL_DOWN8!)`, binary md5
`b851b14a3647202acef67d796b59b115`, tape
`d6fe97d0a1928e1b6c9d43985557189d41a3f75533e3da91faaff473d7624e68`.

Worth recording for its own sake: `short0` answers `Got it, let's calculate 17 times…` → `391`.
That is the **correct arm's** fingerprint from the 2026-08-29 bisect, character for character — the
arm that read `Ass` under the old door. The instrument reproduces the known-good side before it is
trusted to judge the new one.

### A harness bug that orphaned a loaded server, and the guard that now prevents it

First boot attempt used a **relative** binary path. `boot.sh`'s pgrep pattern is anchored on the
absolute path — deliberately, since an unanchored pattern self-matches the driving shell — while
`launch.sh` `exec`s whatever it is handed, so `argv[0]` was relative and pgrep could not see it.
`boot.sh` reported `BOOT_FAIL: no server pid` and exited; the `nohup`'d server did not, and a fully
loaded step37 sat on **130 GB of VRAM with no supervisor and no receipt**.

This is the gate-stop pkill class exactly: a pattern that stops matching orphans a VRAM-holding
process and corrupts every arm after it. The orphan was killed, both cards confirmed back to 0 MiB,
and `boot.sh` now **refuses a relative path, a binary outside `$D/bin`, a non-executable binary and
a held flock** before launching anything. The refusal was executed and observed to fire, not merely
added — a diagnostic that has never run is not a control.

### Gate (b0) — is the OFF arm really `main`? Measured, not argued.

The FLAGS rows for all three doors say "OFF = byte-for-byte the current serving path". That is a
claim about `main`, and the four-arm byte gate does not make it: four arms agreeing with each other
would agree just as well if all four were wrong together. So a **fifth arm** runs the same 18-cell
corpus on a binary built from `d3ac87f80` — pre-restore `main`, which contains none of the three
doors' code.

Its identity is checked in the opposite direction from every other arm. `launch.sh`'s marker test
normally requires all three door strings to be present; for `gate-main` the same test is
**INVERTED** rather than skipped, because skipping a marker test for one arm is how an arm set
quietly stops being what it claims. Both directions were executed and observed to abort:

```
ABORT: gate-main needs a PRE-milestone-3 binary but this one carries the MEMRA_NVFP4_BANK_SM door
ABORT: binary lacks the MEMRA_NVFP4_BANK_SM door — not a milestone-3 binary
```

**All five arms, two binaries, five fresh boots, one tape.** Gates re-run on the PRICED binary
(`0ab9551f60989821f15b9c69a34762de`, the announce build) so the gated program and the priced program
are the same program:

| pair | verdict |
|---|---|
| `M0` (`d3ac87f80`, md5 `64392d961fb27955edb254d2b71de851`) → `GOFF2` | **PASS** 18/18 |
| `GOFF2` → `GSM2` (**P1**) | **PASS** 18/18 |
| `GSM2` → `GGU2` (**P2**) | **PASS** 18/18 |
| `GGU2` → `GD82` (**P3**) | **PASS** 18/18 |
| `M0` → `GD82` (pre-restore main vs all three doors) | **PASS** 18/18 |

Tape `d6fe97d0a1928e1b6c9d43985557189d41a3f75533e3da91faaff473d7624e68` in every arm, and the same
value the first gate round produced on the earlier binary. So the FLAGS claim is a receipt.

### Engagement is a receipt now, and it was not before

Caught by diffing the four byte-gate arms' server logs against each other: with the door names and
per-request UUIDs excluded, the 757-line logs were **identical**. The env census proved
`MEMRA_NVFP4_BANK_SM=1` was SET; nothing anywhere proved the slot-major bank was BUILT or the
`_sel_v2` kernel LAUNCHED.

That is not cosmetic. The first pricing rotation was already producing rows — boot 1 at OFF 106.20,
SM 107.08, SM+GU 106.37 decode tok/s, a ~0.8% spread where the removed bundle was priced at
**+23.7%** — and with no engagement receipt those numbers cannot distinguish "the restored programs
are worth ~1%" from "the doors never reached the code". The `MEMRA_BF16_MMV` lane banked exactly this
defect: its first engagement grep returned 0 in **both** arms, which was a missing-line receipt
defect and not a no-engagement result. So the rotation was **stopped mid-flight** rather than
finished, its rows banked as `rows-price-NOENGAGEMENT.jsonl`, and the card time was cut instead of
doubled.

Four announces, all unconditional. The removed implementation had this behind `MEMRA_SWEEP_TRACE`
and its own comment said why it existed: *"a silently-dead fusion reads as roofline physics without
it."*

Measured on the real model, both extremes, one binary:

| arm | `[nvfp4-bank]` | `[nvfp4-sel]` | `[nvfp4-sweep]` |
|---|---|---|---|
| `v3-off` | `layout=block-nvfp4-v1 source=default` | `kernel=qmatvec_nvfp4_dp4a_sel slot_major=false` (both geometries) | `gu_fused=false door=false`, `down8=false door=false` |
| `v3-sm-gu-d8` | `layout=slot-major source=MEMRA_NVFP4_BANK_SM` | `kernel=qmatvec_nvfp4_dp4a_sel_v2 slot_major=true` | `gu_fused=true door=true`, `down8=true door=true device_routed=true` |

Two cross-checks fall out of that table for free, and both are stronger than the lines themselves:

- **`geometry_match=true` in the OFF arm** proves a false `gu_fused` is the DOOR and not a geometry
  mismatch. Without that field a disarmed fusion and an ineligible one look identical.
- **The gate/up `[nvfp4-sel]` line disappears in the ON arm.** It is present for
  `in_f=4096 out_f=640` with the doors off and absent with them on, because the fused kernel
  replaced those two sweeps entirely. The receipt set proves the fusion took over the launch path,
  not merely that a flag was read.

### Teeth arms, and a build-flow trap that cost two wrong md5 readings

Per the coordinator's requirement, milestone 4's decode oracle claim needed a corruption control it
did not have. One was built: `*1.5f` injected on the per-slot scale multiply **inside the restored
`_sel_v2` kernel only** — the narrowest possible break, in the exact kernel P1's claim rests on.

The build flow is deliberate, and the reason is a trap this lane walked into first:

> **`cargo build -p memra-engine --bin nvfp4-bank-oracle` can complete as a NO-OP in under a second
> and leave a stale binary in `target/release/`.** Two of this lane's early md5 comparisons were
> between no-op builds and were therefore meaningless — one "baseline" (`917f6d3e…`, "Finished in
> 0.65s") was a binary from an earlier tree state entirely, and a later comparison appeared to show
> a revert failing to reach the binary when in fact the build had been cut off by an ssh timeout.
> **Do not compare oracle binaries across builds you did not force.**

So every build `touch`es `crates/memra-engine/build.rs` (which owns the `nvcc --fatbin` step and
declares `rerun-if-changed=cu/qmatvec.cu`) and prints its md5 immediately after its own build:

| label | build time | md5 |
|---|---|---|
| `clean1` (tree as committed) | 4m 54s | `e2bc8c801837d3951d50c45e9377ce4a` |
| `teethA` (`*1.5f` in `_sel_v2`) | 4m 04s | `38ee37b9ed124bad28425b52ef458136` |
| `clean2` (reverted, rebuilt) | — | `802618775ee220a73db70467133362d4` |

**`clean1 != teethA`: the kernel edit reaches the binary on this box.** The reported cross-rig
failure — three `.cu` states producing one byte-identical oracle binary — does not reproduce here
under forced builds.

**`clean1 != clean2`, and that is the more useful finding: the build is NOT byte-reproducible from
identical source.** Two forced builds of the same tree differ. So **md5 inequality is necessary and
not sufficient** — it proves two binaries are different builds, never that they contain different
programs. Which is precisely why the teeth arm has to be *behavioural*: the decisive evidence is the
corrupted binary FAILING the gate, not the two md5s disagreeing. Anyone using md5 alone to argue two
oracle arms ran different programs on this toolchain is arguing from a value that changes for free.

Also worth recording for milestone 2's benefit: its prefill claim already had teeth by construction.
The predecessor's pre-fix twin was a distinct binary (`832fa62a…` vs the fixed `073b2d5f…`) and it
failed loudly, 100% of elements differing and localized to `moe_kq_sktail_kernel`. The `d3ac87f80`
verification is not void.

### Gate (d) — attribution pricing. THE ~24% PRIZE IS NOT THERE, AND ONLY ONE PROGRAM SEPARATES.

Protocol as pinned before any card time: interleaved fresh boots, escalated x3 → **x5** because
both amendment rules fired, every arm reporting its spread and every escalation naming the rule.
Vendor-default sampled (NO sampling params — StepFun's temperature 0.5 / top_p 0.9 from
`models.toml` governs, the shape real traffic sends), spec on at the serving policy (K=3, MTP 3),
real prompts (8 agentic turns, one measured rep each), `max_tokens` capped at 512, loops flagged
and excluded. **20 fresh boots, one binary** `0ab9551f60989821f15b9c69a34762de`, 20 distinct boot
nonces, **zero loop exclusions in any arm**.

| arm | doors | decode tok/s | spread | wall tok/s | spread | TTFT | accept |
|---|---|---|---|---|---|---|---|
| `V3OFF` | — | **106.78** | 1.85% | **100.80** | 2.46% | 0.243 | 0.7926 |
| `V3SM` | +P1 layout | **105.35** | 3.12% | **100.73** | 2.62% | 0.246 | 0.7817 |
| `V3SMGU` | +P2 gate+up | **107.91** | 5.51% | **102.53** | 5.15% | 0.246 | 0.7831 |
| `V3SMGUD8` | +P3 down8 | **113.83** | 1.97% | **107.75** | 2.33% | 0.246 | 0.7747 |

Cumulative deltas: P1 **−1.34%**, P2 **+2.43%**, P3 **+5.48%** decode. Total OFF → all three
**+6.60% decode / +6.89% wall**.

**Every delta is within 2× its pooled spread, so RULE B fires on all three even after x5.** The
median table cannot separate the programs, and saying so is the result. But the per-boot ranges
can, and the non-overlap test is the stronger instrument:

| arm | decode range over 5 boots | vs OFF `[105.11, 107.09]` | verdict |
|---|---|---|---|
| `V3SM` | `[104.66, 107.95]` | overlaps | **P1 NOT separated — no measurable win** |
| `V3SMGU` | `[105.48, 111.44]` | overlaps | **P2 NOT separated** |
| `V3SMGUD8` | `[112.59, 114.82]` | **NO OVERLAP** (min 112.59 > max 107.09) | **P3 separated** |

`V3SMGUD8` also clears `V3SMGU`'s own range (112.59 > 111.44), so P3 separates from the arm
directly beneath it and not merely from the baseline. **PROGRAM 3, the fused down+combine, is the
only one of the three that this box can measure as a win.** P1 and P2 are, on this evidence,
free-or-noise: they are bit-identical (gate a and gate b), they are not slower in any defensible
sense, and they are not faster either.

**The win is not a loop or acceptance artifact, and that is checked rather than assumed.** Zero
reps were excluded as loops in any of the 20 boots, and acceptance *falls* slightly as the doors
arm (0.7926 → 0.7817 → 0.7831 → 0.7747). A degenerate or repetitive arm shows the opposite —
higher acceptance on cheap repeated tokens — so the +5.5% is decode work, not decode theatre.
TTFT is flat across all four arms (0.243–0.246 s), which is what a decode-sweep fusion should do
and is a further check that nothing moved in prefill.

### Why the removal priced the same programs at ~24%

`research/perf-chain-20260831` cell 1 priced the bundle at **−21.5% wall / −23.7% decode**
(157.60 vs 120.25 decode tok/s). This lane measures the same three programs, on one binary, at
**+6.6% decode** — about **3.6× smaller** — with the individual layout and fusion contributions
indistinguishable from noise.

The two numbers are not comparable as measurements of the same thing, and the reason is the whole
point of this lane: **cell 1's door-ON arm ran the corrupted program.** The `in_f` defect was live
in every doors-on boot of that era, so its ON arm was serving fluent wrong text while its OFF arm
served correct text. Cell 1's own receipt records the tell: **acceptance 0.970 with the doors on
against 0.935 with them off.** On a correct binary this lane sees acceptance *decrease* slightly
when the doors arm. An ON arm whose acceptance jumps is the signature the greedy-loop law names —
degenerate output repeats cheap high-accept tokens and inflates both acceptance and tok/s — and it
is exactly what a corrupted routed-expert FFN on an intact residual stream, router and shared
expert produces.

Stated with the right epistemic weight: this is an **inference about a prior lane's numbers from
its own banked acceptance receipt**, not a re-measurement of them, and the era binary is not
rebuilt here. What is measured, and what a merge decision should rest on, is the row above: on the
fixed binary, at the serving env and the serving request shape, the three restored programs
together are worth **+6.6% decode / +6.9% wall**, of which only the down8 fusion separates.

**The mandate's framing therefore does not survive contact with the fixed binary.** There was no
~24% prize to re-earn; there is a ~5.5% decode win in one program, cleanly attributed, plus two
bit-identical programs that cost nothing and earn nothing measurable on this box. Attribution was
the deliverable, and the attribution is that the bundle's headline number was mostly an artifact of
the defect it was measured on.

### Teeth arms — the oracle FAILS when it should, and it localizes

`clean2` (`802618775ee220a73db70467133362d4`) → **PASS**, exit 0, every cell DIFF=0.
`teethA` (`38ee37b9ed124bad28425b52ef458136`, `*1.5f` on the per-slot scale inside `_sel_v2` only)
→ **FAIL**, exit 1. And it does not merely fail, it **localizes**:

| cell | teethA |
|---|---|
| prefill `gate_up` / `down` (`QT_NVFP4_V2` GEMM) | DIFF=0 — untouched kernel |
| **P1** `_sel_v2` vs `_sel`, gate_up | **DIFF=5,120 / 5,120 (100%)** |
| **P1** `_sel_v2` vs `_sel`, down | **DIFF=32,768 / 32,768 (100%)** |
| **P2** `_sel_v2_gu` both halves | DIFF=0 — does not call `_sel_v2` |
| **P3** `_down8` | DIFF=0 — does not call `_sel_v2` |

The first deviation reads `oracle=2.606930542e1 → arm=3.910391235e1` and
`oracle=-1.533289642e2 → arm=-2.299934387e2`: both exactly **×1.5000**, the injected factor. So
the gate detects the break, reports the right magnitude, and names the right kernel while leaving
the other two programs green — the oracle has resolution **per program**, not just per binary.
Raw: `raw/m4-teeth-build.log`, `raw/m4-teeth-run.log`.

Receipts for this section: `raw/m4-attribution-x5.log` (the table and every rule that fired),
`raw/rows-price2.jsonl` (every rep), `raw/rows-price2-summary.json` (per-boot medians),
`raw/progress-price2.txt` (20 boots, preflight and identity lines),
`raw/engagement-probe.log`, `raw/m4-run.log`.

## Milestone 5 — down8 rides last. CLOSED, and it is the only program that earned its door.

The mandate: *"down8 rides last and only on green 3+4 for the layers beneath it."* Both conditions
were met by construction rather than by assertion, because the cumulative arm design enforces them:

- **Green 3 beneath it.** `MEMRA_NVFP4_SEL_DOWN8` refuses at the launcher unless the down shard
  reports `slot_major`, so it cannot arm without PROGRAM 1's layout. Milestone 3 restored both and
  the layout's readers all route through one mapping.
- **Green 4 beneath it.** The `layout` and `layout+gu` arms passed gate (a) (oracle DIFF=0), gate
  (b) (18-cell greedy tape, prefill ladder to 4,555 tokens), gate (b0) (identical to pre-restore
  `main`) and gate (c) (short-prompt content oracle) BEFORE the down8 arm was priced. The pricing
  rotation runs the arms in cumulative order, so `v3-sm-gu-d8` is literally the last arm of every
  boot.
- **Priced last, and separately.** Its contribution is the delta from `layout+gu` to
  `layout+gu+down8`, over five interleaved fresh boots.

**Result: PROGRAM 3 is the only one of the three that separates from noise.** +5.48% decode
(113.83 vs 107.91), +5.09% wall (107.75 vs 102.53), within-arm spread 1.97%/2.33%, per-boot range
`[112.59, 114.82]` with **no overlap** against either the OFF arm or the arm directly beneath it.
Bit-identical throughout: the oracle compares it against `_sel` + `axpy_rows_seq_md` — the exact
two-launch program it replaces — at DIFF=0, and the end-to-end greedy tape is unchanged.

That is the whole case for the mandate's ordering. Had down8 ridden first, its +5.5% would have
been the bundle's number again and the layout would have inherited credit for a win it does not
produce. Riding last, behind two gated and separately priced programs, is what makes "+5.5% is the
fused down+combine" a sentence anyone can check.

## What a merge decision actually turns on

Not this lane's call. Stated so the owner does not have to re-derive it:

- **All three doors are default OFF and every gate is green.** Nothing here changes a served byte
  unless a door is set, and that is a receipt (gate b0, against a pre-restore binary).
- **`MEMRA_NVFP4_SEL_DOWN8` is the only door with a measured win**, and it needs
  `MEMRA_NVFP4_BANK_SM=1` to arm. So the deployable unit is those two together: +5.48% decode /
  +5.09% wall on the serving shape, bit-identical, with a one-line rollback (unset).
- **`MEMRA_NVFP4_SEL_GU` earns nothing measurable** and costs nothing. Its value is structural —
  without its own door the 2026-08-29 bisect could not name a mechanism, and it should keep the
  door for that reason alone.
- **A default flip is a separate decision from a merge**, and per the flag-default law it is the
  owner's with receipts attached. The receipts for a flip would be these rows plus the 12-boot
  battery form the `MEMRA_STEP_TP_GRAPH` row documents; that battery has NOT been run here.
- **Nothing in this lane is a product claim.** No published number, context window or price moves,
  so the product-facts workflow is not involved.

**Publicity (per the every-release-ships-with-its-publicity law): there IS a post-worthy angle, and
it is not the perf win.** It is the measurement story — a perf claim of −23.7% decode that survived
review, drove a door removal, and turns out to be ~3.6× too large because the arm it was measured
on was silently corrupt, caught only by re-deriving it behind three separate flags with an
engagement receipt. That is a gate-craft post, not a benchmark post, and the honest headline is
"we deleted the right code for the wrong reason, and the number was never real". Drafting it is
deferred to the merge, because a publicity draft for an unmerged lane would front-run the owner's
decision on whether the arms ship at all. Flagged here so it is not silently skipped.

## The merge PR's required disclosures (owner's call to merge; these must ride with it)

### EP2 guard disposition — the refusal never covered the posture the defect lived on

`removed_bank_v2_doors_refusal` guards exactly two names, `MEMRA_NVFP4_BANK_V2` and
`MEMRA_SEL_DOWN8`. It has **never** guarded `MEMRA_STEP_NVFP4_EP2`, and EP2 is precisely the
posture the `in_f` defect lived on: `fd0a175ab` deliberately kept `QT_NVFP4_V2` alive for the
always-slot-major EP2 whole-expert banks, so from 2026-08-29 until `1b18a61e8` **EP2 prefill on
`main` was reading wrong per-16 scales with no guard, no warning and no flag to unset.** That is
`DIAGNOSIS.md`'s live-on-main finding, and it means the guard's coverage was never the shape its
name suggests.

Disposition for the PR, stated plainly:

- **Post-fix the posture is correct**, so nothing needs guarding going forward: `1b18a61e8`
  removed the defaultable `in_f` and the compiler now enforces it at every `kq_fetch` call site.
- **Pre-`d3ac87f80` binaries still boot the defective EP2 posture silently.** No refusal catches
  it, because the flag that arms it was never in the refusal's list. Any receipt that armed
  `MEMRA_STEP_NVFP4_EP2` on an older binary is invalid, not merely old.
- The governing corpus row is **`TRAP:ep2-prefix-binaries-unguarded`** (darklanes
  `agent-knowledge/gpu/gate-craft.md`, `e6f3d3908`). It is the authority for judging old EP2 rows
  and it names the verification: confirm the binary is at or past `d3ac87f80` before trusting one.
- **Repo-wide receipt sweep is done: zero pre-fix EP2-armed rows are banked** in either the memra
  or the darklanes research trees. So the trap is a forward-looking guard, not a cleanup backlog.

One correction to the framing this lane received: the refusal is **rescoped, not retired** (see the
milestone-3 section). The EP2 gap above is orthogonal to that choice — the refusal did not cover
EP2 before and does not now, under either disposition.

### Every oracle PASS in this lane carries a teeth arm

Coordinator requirement, 2026-09-01, from the glm session's triage: on at least one rig the
`nvfp4-bank-oracle` **bin target did not track kernel source edits** — three different
`moe_f16_grouped.cu` states (fixed, pre-fix, and a deliberate `*1.5f` scale corruption) produced a
**byte-identical** oracle binary that reported PASS. An oracle PASS from a stale binary is worse
than no gate: it is a green light that proves the build system works, not the kernel.

So every oracle claim here is paired with a corruption-control arm built in the same flow, both
md5s recorded and asserted different:

- **Milestone 2's prefill claim already had its teeth.** The predecessor ran the pre-fix twin as a
  distinct binary (`832fa62a…` against the fixed `073b2d5f…`) and it FAILED loudly — 100% of
  elements differing, localized to `moe_kq_sktail_kernel`. That arm ran by construction, so the
  `d3ac87f80` verification is not void.
- **Milestone 4's decode claim did NOT, and that gap is closed here** — see the teeth section
  below. It was a real omission: the three restored decode programs were gated against a v1 oracle
  with no proof that a broken kernel would have been caught.

A build-flow finding worth its own line, because it cost two wrong md5 readings before it was
understood: **`cargo build -p memra-engine --bin nvfp4-bank-oracle` can complete as a NO-OP in
under a second and leave a stale binary in `target/release/`.** Two of this lane's early md5
comparisons were between no-op builds and therefore meaningless (one "baseline" was a binary from
an earlier tree state entirely). The teeth flow below `touch`es `crates/memra-engine/build.rs`
before every build — build.rs owns the `nvcc --fatbin` step and declares
`rerun-if-changed=cu/qmatvec.cu` — and prints each md5 immediately after its own build, so a no-op
cannot be mistaken for a rebuild. **Do not compare oracle binaries across builds you did not force.**

## Lane state, box state, and what a successor would pick up

**Nothing is open.** Milestones 1-5 are closed, every gate is green, the attribution is banked, and
the merge is the owner's decision. If it is declined, the three doors are default OFF and the tree
is byte-identical to `main` on the served path (gate b0), so declining costs nothing.

Named as NOT done rather than left implied:

- **The 12-boot battery** that a DEFAULT flip would need (the form the `MEMRA_STEP_TP_GRAPH` row
  documents) has not been run. This lane's deliverable was gated arms plus attribution, not a
  deploy, and no default moved.
- **The three sub-doors are UNPRICED by design**: `MEMRA_NVFP4_SEL_SM_STREAM`,
  `MEMRA_NVFP4_SEL_GU_RPW`, and `MEMRA_NVFP4_SEL_GU_WPR` — the last is numeric-class and needs a
  run-gen argmax cell plus a boot battery, not a byte gate, before it may be armed anywhere.
- **The 30k-token prefill pair** is out of the byte-gate corpus. The ladder stops at 4,555 prompt
  tokens; the incident's blind spot was "no prefill gate at all", not "not deep enough".
- **`repack_cache_is_fresh` validates only file length.** Not this lane's to fix, and a real trap
  for any future geometry-preserving change to the repack routine.
- **The two-column / t-row verify program** (`MEMRA_TCOL_FFN`'s named feature, `_down8_rows`,
  `_gu_tcol`) stays deleted. It was not in the 140-era serving env, so it cannot be part of the
  prize, and restoring a kernel with no live launcher is dead code.

**Box state** (the authorized non-prod 2x RTX PRO 6000 bench box): both cards drained to **0 MiB**,
no server process, no `run-ab` or `boot.sh` alive. `/home/ubuntu/bankv3/src` is on the lane commit
with a **clean** `git status` — the teeth injection was reverted and verified. Kept, deliberately,
because they are expensive to rebuild and a successor or a re-price would want them:
`/home/ubuntu/bankv3/` (build cache, `lane/harness`, `lane/bin` with all three binaries,
`lane/receipts`, `bin-teeth`). Nothing outside `/home/ubuntu/bankv3` was written; `guard-bins`,
`guard-lane` and the other lanes' directories were never touched. The **shared** 126 GB step37
checkpoint and its 100 GB repack cache were read only.

One orphan was created and cleaned during this lane (the relative-path boot), and its cleanup was
verified by reading both cards back at 0 MiB, not assumed.

## Commits

- `381d492e3` — milestone 1, `DIAGNOSIS.md`.
- `1b18a61e8` — the `in_f` fix (both call sites, and the default removed so the compiler enforces
  it), `nvfp4_matrix_v2_permute` made `pub` with the `in_f % 64` assert, and the
  `nvfp4-bank-oracle` bin.
- `f30fb4083` — the PTX-neutrality receipt for that fix.
- `d3ac87f80` — merge of milestones 1+2 to `main`.
- `09491a5df` — milestone 3: the three-door restore, the layout-travels-with-the-bank refactor, the
  two host-canonical oracle fixes, the oracle bin's decode coverage, the rescoped boot refusal, and
  the six `docs/FLAGS.md` rows.
- `53cd3e864` — milestone 4 gates (a)/(b)/(c) and the mechanical perf-CI neutrality receipt.
- `01ed43c1a` — the four engagement announces, after the byte-gate arms' server logs proved
  indistinguishable.
- `f21554b8c` — gate (b0) against a pre-restore binary, the engagement table, and the teeth-flow
  build trap.
- `1f2b4920c` — the merge PR's required disclosures (EP2 guard disposition, teeth-arm requirement).
- `ccc61f760` — the aggregator arm-key fix (it had silently dropped the down8 arm).
- `b9e812931` — milestone 4 CLOSED: the x5 attribution table, the overlap verdicts, the teeth-arm
  result, and the corrected `docs/FLAGS.md` rows.
- `be56fe7f6` — `research/INDEX.md`: this lane's row, and the perf-chain row marked SUPERSEDED.
- darklanes `45e355466` (branch `lane/step37-bankv3-corpus-20260901`) — ten gpu-corpus rows and the
  lane's private prompt corpora.
