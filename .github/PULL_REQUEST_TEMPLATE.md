<!--
Read CONTRIBUTING.md first. Fill the sections your change reaches and delete the rest.
A section marked "GPU" is required only when change_impact.py reports a GPU cell.
-->

## What this changes and why

<!-- One or two sentences. Name the mechanism, not "makes it faster". -->

## Change impact

```
$ python3 tools/change_impact.py --base origin/main --head HEAD
<paste>

$ python3 tools/cell_evidence.py need --base origin/main --head HEAD
<paste>
```

## CPU checks

```
<the tests `python3 tools/validation_plan.py local --base origin/main` selects, and their results>
```

## Correctness gates (GPU; paste output, not "passed")

```
$ ./target/release/kernel-check
<tail: ALL GREEN (<n> cells, <m> skipped)>

$ ./target/release/run-gen ...
<tail: the verify-prefill argmax line ending in MATCH; on a prompt of 16+ tokens, the batched-prime line too>

$ ./target/release/run-spec ...
<tail: === SELF-CONSISTENCY PASS ===>
```

## Performance: prefill and decode (GPU, when a perf cell is reached)

Protocol: [the method](https://github.com/avifenesh/memra/blob/main/docs/PERFORMANCE.md#method).
Base and branch interleaved in the same session, both orders, N >= 3 medians, GPU power and clock
state recorded. Runs from different sessions drift by up to about 10% and are not evidence.

| Metric | Base (main) | This branch | Ratio |
|---|---|---|---|
| pp512 (prefill, tok/s) | | | |
| pp2048 (prefill, tok/s) | | | |
| tg128 @ 512-ctx (decode, tok/s) | | | |

## Main runners exercised (GPU)

- [ ] `run-gen` on a real model, full output pasted, argmax line included
- [ ] `run-spec` (required if this touches attention, GEMM, MoE dispatch or the KV cache)
- [ ] `memra-server` request and response (required if this touches request handling or batching)

## Prior work

- [ ] I checked the [research index](https://github.com/avifenesh/memra/blob/main/research/INDEX.md)
      and `research/tune-data/` for prior attempts at this change, and found none or include new
      evidence that overturns the earlier result (link it and say what differs).

## Hardware used

<!-- GPU, VRAM, driver, CUDA version and MEMRA_CUDA_ARCH. With no GPU of any target this change
     reaches, open an issue instead of a PR. Without hardware for one affected target, say so
     here: the change still needs that cell's record before it lands. -->
