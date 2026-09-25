# WP-A resumable state (2026-09-25, stopped at NEED TARGET CARD with the 5090 down)

- Lane `lane/spill-a-20260919`; worktree `wt-spill-a`; merged with `origin/main` `5d653e851` (integ59, #723) at resume.
- Item 4: design S2 pre-registered (DAY42 sections 1 and 1a), built (`7ce3f3243`), CPU cells green (engine lib 552,
  server lib 912, tier crate, clippy, fmt, check-flags, diff --check). Not integrable yet: its 5090 cells and its target
  sitting have not run.
- The 5090 reads `GPU requires reset` since 01:25Z (Xid 119, GSP RPC timeouts from `nvidia-smi` and `nvidia-powerd`,
  then Xid 154); a reset or reboot is the owner's. Its S2 cells (`rtx5090-day42/`) and item 16's cell (`rtx5090-day45/`)
  wait for it.
- Target sittings prepared, in the order to run on one RTX PRO 6000 Blackwell box (any host class for the first two):
  `pro-single-s2/` (S2: build.sh <tip> b4816eda8, then driver.sh), `pro-single-i15/` (item 15: build.sh <tip> b4816eda8
  after the S2 build, then driver.sh), and on a 9950X-class host `pro-single-t9950/` (item 3's owed reading: build.sh
  <tip> b4816eda8 checks the host class first, then driver.sh).
- Pre-registered and waiting: item 15 (DAY43), item 3's 9950X reading (DAY44), item 16 (DAY45). Next after them in the
  ledger: items 6 to 14.
- Local scratch: none (the eza waits of earlier sessions in this worktree were ended by pid; `/tmp/wt-a*` empty).
