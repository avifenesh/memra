# Quiet-window scheduler

`tools/impact_map/scheduler.py` runs the delta set on each device when the device is free, banks the evidence, and never joins other work. Every wait is bounded (`--max-wait`, default 6 h). A device that never opens is reported, not retried forever.

```
python3 tools/impact_map/scheduler.py plan --base origin/main --head HEAD --out q.json
python3 tools/impact_map/scheduler.py run --device rtx5090 --queue q.json --out ~/.cache/memra-cells/run-1
```

`plan` writes the queue: per device, the models the delta reaches, with their routes and tests. `run` waits for the device's window, builds, runs `tools/cell_evidence.py run` with the device battery and the kinds the queue needs, then banks the record (`cell_evidence.py bank`). Models in the plan that are not in the battery roster go to `advisory.json` next to the evidence. They are the reached families' own gates, and the lane owner names them in the PR body.

## Windows per device class

| device class | the window is open when | how the run holds it |
|---|---|---|
| RTX 5090 (a desktop machine people use) | `gpulock status` says free, the shared build lock is free, and the 1-minute load average is below `--max-load` (default: a quarter of the CPUs, so a busy desktop is never joined) | build under the shared build lock at `nice 15` with 2 jobs (`cell_evidence.py` sets both); GPU phase under `gpulock run`, which also takes `/tmp/memra-5090.lock` |
| GB10 | `sparklock status` says the unit is free; it counts an unleased GPU process or container as busy, so a benchmark another session runs is never joined | `sparklock run --unit U --wait` around an ssh to the unit (`--ssh-host` from local config) |
| RTX PRO 6000 on a shared workstation | a grant covers the queue: the cards belong to other work, so a window is granted, not detected | `request` writes what is needed and for how long; whoever relays the agreement records it with `grant --gpus 1,3 --until <hard stop>`; `run` pins `CUDA_VISIBLE_DEVICES` to the granted cards, holds `/tmp/memra-gpu.lock`, and stops 5 minutes before the hard stop |
| H100, B200 | never (rented boxes) | their cells go to the pre-tag battery |

## How it fits the daily regression

`tools/daily-regression.py` (#1030) runs main's head on each device's full cell set once a day. The scheduler is for the per-change delta between those runs. A record either produces for a push is reused by later pushes as long as the delta since its commit does not reach its target (`cell_evidence.py` "still proves"). A quiet night therefore covers every push that lands before the next change to that target.
