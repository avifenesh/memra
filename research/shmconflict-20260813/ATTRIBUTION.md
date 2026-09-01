# `fa_prefill_qw_db` shared-memory replay attribution

Date: 2026-08-13
Source state: unmodified `v0.81.3` plus the progress-only checkpoint
Attribution verdict: **Q, K, P, and V are all conflict-bearing; recurring K/V loads contribute
97.03--97.22% of excess wavefronts.**

All observations are relative-only measurements from the local RTX 5090 Laptop GPU under the
owner-imposed 210--1200 MHz cap. Both profiler launches held `/tmp/memra-5090.lock`, and each
preflight found no compute application. No clock setting was changed. The NCU reports remain in
`/tmp`; only their exported source/SASS counter tables and console logs are in the repository.

## Frozen request and source boundary

This capture reuses the predecessor request verbatim:

- 4,860 exact prompt token ids, canonical hash
  `eba9dff66d4ebf3f40d6db80298ab9c884fa1bb2e1a6d4ca2c7dadd4f21513fb`;
- 60 completion tokens, temperature 0, seed 3407, context limit 4,928;
- `cached_tokens=0`; and
- the observed `fa_dequant_kv_ws_bf16` -> `fa_prefill_qw_db` serving route.

The imported [`profile_request.py`](profile_request.py) and the predecessor's
`research/fa3softmax-20260813/profile_request.py` both hash to
`8dc7093dc3a8b245842a4e5eca25c8182d66fb7a7dcbdb3ebd37fa282e652b4d`.
The isolated baseline server is unmodified `v0.81.3` and hashes to
`4f0205c2cc2b31cdde89a395b41211f2663caa76b333b55ba7a4a4a085e48521`.
The predecessor profile and verdict are the required
`main@055adf47d:research/fa3softmax-20260813/{PROFILE,RESULTS}.md`: they establish the frozen
shapes, overall replay, occupancy limit, and the scheduling NO-GO that this lane does not reopen.

## Per-instruction attribution

The relative PCs below come from the baseline `fa_prefill_qw_db` SASS. The execution-count split
is unambiguous: Q's 16 `M88` PCs execute once per CTA; K's 32 `M88` PCs, P's eight `STS` plus two
`M88` PCs, and V's 32 `MT88` PCs execute once per retained KV tile. The same PC groups and ratios
occur in all four captures.

| Array / phase | Relative SASS PCs | Instructions | Current physical layout | Conflict degree | Actual / ideal | Share of all excess |
|---|---|---|---|---:|---:|---:|
| Q transient tile | `0x000e10..0x001430` (16 PCs) | `LDSM.16.M88.4` | `16x256` bf16 row-major, 512-byte stride; no padding/swizzle | 32 | 8.00x | 0.17--0.37% |
| K double buffers | `0x0031d0..0x0042b0` (32 PCs) | `LDSM.16.M88.4` | `32x256` bf16 row-major, 512-byte stride; no padding/swizzle | 32 | 8.00x | 48.51--48.61% |
| P restage stores | `0x0056f0..0x0058c0` (8 PCs) | `STS` | `64x32` bf16 row-major, 64-byte stride; no padding/swizzle | 4 | 4.00x | 1.30% |
| P restage reloads | `0x0058e0`, `0x005960` | `LDSM.16.M88.4` | same unswizzled P rows | 16 | 4.00x | 1.30% |
| V double buffers | `0x005920..0x006e90` (32 PCs) | `LDSM.16.MT88.4` | `32x256` bf16 row-major, 512-byte stride; no padding/swizzle | 32 | 8.00x | 48.51--48.61% |
| Other shared traffic | remaining shared PCs | mixed | mixed | 1 or 4 | 1.00x | 0% |

The 512-byte Q/K/V stride maps the participating `ldmatrix` lanes back onto the same bank
columns. Their 32-way source-counter classification expands every ideal four-wavefront
`ldmatrix.x4` request to 32 wavefronts: exactly 8.00x. P's shorter 64-byte stride has a different
collision pattern but still expands its stores and reloads to 4.00x. This is direct per-PC
evidence, not an inference from the aggregate ratio.

The raw review surfaces are:

- [`raw/baseline-shared-attribution.csv`](raw/baseline-shared-attribution.csv): per-array sums;
- [`raw/baseline-shared-pcs.csv`](raw/baseline-shared-pcs.csv): every classified PC and counter;
- `raw/profile-baseline-{q27,q35}/ncu-source-sass.csv`: complete exported per-PC counter tables;
- `raw/profile-baseline-{q27,q35}/fa_prefill_qw_db.sass`: extracted cubin SASS; and
- [`extract_ncu_shared.py`](extract_ncu_shared.py): deterministic classification and summation.

The extractor's totals reproduce the predecessor aggregate exactly:

| Model / shape | Actual wavefronts | Ideal | Excess | Actual / ideal |
|---|---:|---:|---:|---:|
| Q27 / 4,096 | 873,541,632 | 135,966,720 | 737,574,912 | 6.424672x |
| Q27 / 764 | 354,302,976 | 54,912,000 | 299,390,976 | 6.452196x |
| Q35 / 4,096 | 582,361,088 | 90,644,480 | 491,716,608 | 6.424672x |
| Q35 / 764 | 236,201,984 | 36,608,000 | 199,593,984 | 6.452196x |

## Source-to-SASS map

- Q is written and loaded row-major by `load_q_frags` at
  `crates/memra-engine/cu/flash_attn.cu:680-700`, called by the target at `:4755-4757`.
- K and V are copied to row-major double buffers by `stage_kv_tile_async` at `:4695-4715` and
  issued by the target at `:4786-4801`.
- K uses unswizzled `ld_A` at `:4816-4818`.
- P is stored row-major at `:4873-4880` and reloaded with unswizzled `ld_A` at `:4894-4896`.
- V uses unswizzled `ld_A_trans` at `:4894-4897`.
- The unswizzled address maps are at `:103-112` and `:138-145`. XOR-swizzled Q/K/V-capable
  helpers already exist at `:114-135`, but this target does not call them.
- Dispatch selects `fa_prefill_qw_db` and requests its dynamic shared memory at
  `crates/memra-engine/src/lib.rs:9643-9666`. The source formula requests 69,888 bytes; NCU reports
  70.91 Kbyte/block after device allocation accounting, one CTA/SM, and 8.33% theoretical
  occupancy.

## Fix selected from the attribution

The first candidate is layout-only XOR swizzling, with no padding and no tile-geometry or
arithmetic change:

- Q/K/V: XOR each 16-byte chunk column with `row & 7` and use the existing inverse address map in
  `ld_A_sw` / `ld_A_trans_sw`;
- P: XOR its four 16-byte chunk columns with `row & 3` on both store and `ldmatrix` load.

This targets every attributed conflict while preserving every byte and every MMA/accumulation
order. It does not shrink the shared allocation, so it cannot create a second CTA/SM; occupancy
must be checked for regression, but there is no separate capacity win to claim from this layout.

