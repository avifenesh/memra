# Step 8 receipts

- `verify/final`: `memra model verify checkpoint <artifact dir at Avifenesh/Hy4-flash-NVFP4@f9eac06cf022> --against hy_v4_flash --oracle <step 7 f64 tape set> --native-runner hy4f-gpu` on one GB10 (sm_121a), 2026-11-20 17:09 to 17:15: `verified=Checkpoint`.
- `verify/red-*`: the same command through `checkpoint/red-runner.sh` with one wrong program each; every run fails (`run.log`). `verify/pr7-class`: step 7's `hy4f-run` through this step's gate, passes.
- Binaries (`bins.sha256`): built for sm_121a from this commit's `crates/`; `hy4f-gpu` is the binary the checkpoint receipt names (`native_runner_sha256`).
- `spreads.jsonl`: every tape of this step against its f64 oracle tape (`gpu/score_spread.py`), labeled: the route (`route-bf16x2-*`: the adopted route; `route-bf16-*`: the rejected single-BF16 arm), the pinned program's band and diagnostic backends (`backend-*`), and red arms run outside the gate (`red-*`). `free-vs-forced.jsonl`: `gpu/free_vs_forced.py`. `position-756.txt`: per-layer router-score distance at three positions (`gpu/pos_spread.py`).
- `tapes/*.tape.json.gz` and `arrays.sha256`: each tape's header and the sha256 of its arrays; the arrays (full logits, up to 1 GB a tape) stay off the repository.
- `bench-*.jsonl`, `logs/pr8-bench-*.log`: `hy4f-gpu --bench 3` (warm-up row first).
- The oracle and band tapes ran on CPU on the same machine (torch 2.15.0+cpu, transformers PR #50903 at b865850d0532), one at a time.
