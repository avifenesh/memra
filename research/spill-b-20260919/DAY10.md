# WP-B day 10 — target-class active KV (in progress)

Repository: **avifenesh/memra**, branch `lane/spill-b-20260919`.
Original native gate source: **8ef562b12f16fd3378f866939be347df3a6d3d96**
(integ5 merge, including fixed-VA observation and pooled G1-label review fixes).
Allocator-injection and mapped-VA diagnostic implementation is checkpointed in
**418abc431** and later commits; **native compilation/qualification pending**.
This is an open lane, not merged, released, deployed or serving-qualified.

## Target baseline and pending cells

Hardware: **one RTX PRO 6000 Blackwell, 96 GB**, **600/600 W**.
Every GPU cell is a single-run development correctness probe under
`tools/tier-battery.py --rig pro-single`, `/tmp/memra-gpu.lock`, 250 ms telemetry.
This is the target card class, not a two-card serving qualification.
No timing comparisons across cards, throughput claims or default promotion.
Checkpoint: Qwen3.8-27B native NVFP4/Q5K GGUF; native q8_0/q5_1 KV unchanged.
Program: tokenwise `decode_step_h`, trunk-only, no MTP or alternate prefill.

- 8k baseline: captured and pushed; successful collector attempt 1 (attempt 0
  refused lock contention without executing). Native build exit 0.
- 32k baseline: captured; native exit 0, all baseline hashes frozen separately.
- Original VMM 8k: **ACTIVE-8K G1 PASS**. All seven frozen surfaces and restored prefix match; exact physical release/reacquisition, no residual.
- Original VMM 32k: **ACTIVE-32K physical reclaim/restore bit-identical, residual 2097152 B, class unclassified — not G1 PASS**.
- Original pooled 8k: **ACTIVE-8K pooled control: not-applicable-pooled**; no fixed-VA or G1 claim.
- Mapped-VA residual diagnostic 32k and directly injected VMM 8k: **pending**.

`BOX3-BASELINES.json` freezes the seven decoded continuation-surface hashes and
artifact/binary/plan/prompt/source identity for each completed baseline. These are
new target-card bundles, never interchangeable with the earlier RTX 5090 bundles.
`day10-raw-manifest.json` seals raw archive membership and bytes. Final logits are
losslessly gzip archived; replay hashes decoded bytes. `verify-day10.py` checks
collector journals, source/build/binary identity, all continuation surfaces,
physical-chunk arithmetic, deltas, power, and classification. It **does not run
CUDA**. `--require-complete` fails while any requested cell is missing.

## Residual classification boundary

The diagnostic acts on **actual demoted planes**: retained edge/capacity chunks are
unmapped **without releasing their physical handles**, then their VA reservation
is freed, re-reserved at the original address and the same retained handles remapped.
It reads driver-free bytes before unmap, after unmap, after VA free and after remap,
per plane. This separates physical capacity from mapping/reservation metadata.
Failure leaves the gate suspended; no token executes against that state. Cleanup
tracks whether the reservation remains owned. No new unsafe scope outside the
existing `KvPlane` owner is introduced.

A `va-reservation-page-table` class requires the measured residual to return
**only** on VA free (not unmap), with exact remap accounting. The prior RTX 5090
probe freed a **never-mapped spare** reservation, so it is not the corresponding
mapped-range evidence. Until both cards have the required evidence, a nonzero
residual remains **not G1 PASS**, even if this card classifies it. No class or G1
verdict is inferred from the implementation or CPU tests.

## Direct allocation refinement

`Cache::new_with_allocator(..., KvAllocator::Vmm)` selects native K/V allocation
at construction, through `KvDev::alloc_kv_plane` / `Engine::alloc_vmm_u8`.
There is no pooled empty-plane bootstrap or swap. A backend without VMM support
refuses explicitly instead of substituting pooled storage. All old constructors,
including planned and PP constructors, retain pooled allocation. Recurrent and
latent state allocation, formats, kernels and numerical execution are unchanged.
The existing gate-only `--kv-allocator vmm` door remains default-OFF;
**decide-by: 2026-10-04**. No new environment flag or serving registration.

## Checks actually run

- Mac `cargo fmt --all -- --check`: PASS.
- Mac and Linux-target `cargo check -p memra-kv -p memra-tier --all-targets --offline`: PASS.
- Mac `cargo test -p memra-kv -p memra-tier --offline --no-fail-fast`: **262 passed**.
- Scoped all-target clippy with `-D warnings`: PASS.
- Four Python verdict tests including 12 arithmetic/engagement mutation arms: PASS.
- Archived 8k target baseline replay: PASS (integrity only, not G1).
- `git diff --check` and `bash tools/check-flags.sh`: PASS.
- Additional Mac **engine** check: BLOCKED, exit 101, `spawn nvcc: ... No such file or directory`;
  preserved in `day10-checks/engine-mac-check.log`. This does not replace a native build.
- Native original gate build: PASS, raw build and binary/source identity archived.
- Native refined implementation build and GPU cells: NOT RUN yet.

CPU logs and commands are in `day10-checks/`; native receipts in `pro-single-day10/`.
Migration-safe running-cell details and exact next action live in `STATE.md`.
Time accounting will be updated at close; the lane has a 10-agent-day budget,
with this session bounded to approximately 3.5 agent-hours.
