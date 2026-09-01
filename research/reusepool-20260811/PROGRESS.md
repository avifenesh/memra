# cx-reusepool progress

Status: complete

- Confirmed branch `lane/cx-reusepool` starts from `59df5ebb` with a clean worktree.
- Loaded the confirmed salt-spray VRAM finding and delivery contract from `~/.lanectl/inbox/cx-reusepool.md`.
- Fix commit `893836d6` retains the per-`(model, cache namespace)` cap and adds the default-16 process-wide parked-entry ceiling with oldest-global eviction across plain and speculative pools.
- The four-namespace regression fills `M × REUSE_POOL` entries across both pool types, proves the live total never exceeds the global cap, and proves the three globally oldest entries are reclaimed first.
- CPU-capped release verification is green: focused regression 1/1; full `memra-server` package 183/183.
- Documentation now states the multiplicative pre-ceiling VRAM exposure, the new global flag/default/sizing rule, and the 27bab two-namespace receipt; `tools/check-flags.sh` reports no new drift.
- Constraints: CPU only; builds/tests use `nice -n 15 taskset -c 0-7`; no GPU, merge, tag, push, formatting, or perf-board changes.
