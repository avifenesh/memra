# OWED 17: cold read-once bypass (`MEMRA_MOE_COLD_BYPASS`), results so far

Registration: `../M1-PREREG.md` section F, its precedent note, and the D and F fallback amendment.
Code: `crates/memra-engine/src/moe_cache.rs` (door, doorkeeper, bypass branch, `dispatch_source_once`,
`consumed`, `payload`), `spill_pread.rs` (mapped views, ownership, `mapped_serves`),
`hybrid_forward.rs` (the two batch-1 decode expert GEMMs). Frozen run-gen: `build/`.

## Correctness (all green)

- CPU unit cells: `cold_ghost_first_miss_bypasses_second_admits`,
  `cold_ghost_remembers_only_the_last_cap_first_misses`, `cold_bypass_parse_refuses_unknown_values`.
  Full suites: engine lib 576 passed, engine tests and bins, server 940 plus integration
  (`suites/`), fmt and diff checks clean.
- GPU ownership cell (`5090/mapped-gpu-cell/`): `mapped_serve_reads_in_place_and_owns_the_buffer_until_its_event ... ok`;
  the device alias read the landed bytes, `mapped_serves=1`, `h2d_submits=0`, the buffer was not
  handed out before its consumer event and returned to the pool after it.
- Forced-ON smoke on the 5090 (`5090/f17-smoke/`, gate `5090/f17-smoke-gate.queue.log`): all three
  arms generated 128 tokens equal to the byte oracle; `refused=[]`. Bypass lines:
  `[moe-bypass] mode=staged bypassed=664940 ghost_admits=113104` and
  `[moe-bypass] mode=mapped bypassed=662095 ghost_admits=112831` with `mapped_serves=662095`
  (equal to bypassed) and `h2d_submits=112831` (equal to the repeat-miss admits).
- Door off is the same worker16 program as the B3 build (`5090/doorless-diag/`, unscored, three
  alternating pairs): tokens equal the oracle in all six visits; tok/s 15.65, 9.41, 8.81 (B3 build)
  and 9.11, 9.66, 9.84 (door-off build); fallbacks 895, 36, 0 and 7, 27, 355. The fallback count
  follows run conditions on both binaries (OWED 26), not the build.

## Timing so far (smoke, one visit per arm, never a verdict)

worker16 16.51 tok/s, bypass-staged 14.41, bypass-mapped 10.14. The scored ten-round cell is
queued. Two conditions on this rig bound what it can show: the fallback amendment makes any visit
with an mmap fallback unclean, and the smoke had fallbacks on all three arms (2,372, 3,646,
8,287); and during the diagnostic the card sat at 84 C and 168 of 175 W with SM clock 1,957 of
3,090 MHz (power-cap throttle) under a host load average of 8 to 12 from other lanes' builds, which
dropped both builds from about 16 to about 9 tok/s within minutes. The PRO 6000 half needs a
target card.
