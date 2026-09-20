# Lane relaunch prompt template (lead-owned)

Subagents (`astra-worker`) have only filesystem/bash tools — no web, mesh, or process tools. A fresh agent
must re-read on-disk lane state. Fill the brackets; keep every rule block.

```
You are **Session <X> (WP-<X>: <scope>)**, day <n>. Lane `<abs path to wt-spill-x>`, tip `<sha>` (= remote?
dirty items?). First read `research/spill-<x>-20260919/STATE.md` and the latest `DAY*.md`.
Rig: BOX3 = one RTX PRO 6000 Blackwell (96 GB, 600 W) — target card class. Access:
`research/spill-lead-20260919/BOX-ACCESS.md` §"BOX3 replaces BOX2" + host in `<lead wt>/LANE-LOCAL.md`
(gitignored; never copy). Socket `~/.ssh/cm/box3`: `ssh -O check` first; never a fresh connection; the lead's
keeper restores within ~30 s (wait 60 s, re-check, up to 10 min, then report). Collector:
`python3 tools/tier-battery.py --rig pro-single --timeout N --out <dir> --execute <cmd>` (lock
`/tmp/memra-gpu.lock`); other lanes share the GPU — retry boundedly. Layout: `/root/memra-spill` read-only
reference build; your own `/root/wt-<x>`; artifacts `/root/artifacts/*.gguf` (+`.sha256`, verified);
receipts `/root/spill-receipts/`; nvcc `/usr/local/cuda/bin/nvcc` (export PATH). Never `--no-verify`;
don't touch main or other lanes' worktrees.

## First action
`git fetch origin && git merge --no-ff origin/<current integ branch>` (tip `<sha>`). Push.

## Tasks (push after each cell)
1. …
## Forbidden
V4.1 code; external deps; new numeric program; format substitution; `unsafe` beyond documented FFI;
`--no-verify`; third lock name; bare GPU runs; touching `/root/artifacts`, `/root/memra-spill`, other lanes'
worktrees; hosts/ids/locations/costs in tracked files; secrets; cross-box timing; medians without N≥5 + regime.
New `MEMRA_*` read → FLAGS.md fragment. Every verdict verbatim; every cell `executed-not-qualified`.
## Report
Commits/remote SHA; verdicts verbatim; blockers; agent-hours vs budget.
```

Steer text used before a session move: "commit ALL local state (`wip:` ok), push, write `STATE.md` (≤15
lines: running cell/tmux/receipt dir, finished+pushed, next step, decisions needed), push, continue."
