# Large models on multiple GPUs

For readers whose model does not fit one card. Each multi-card route is per model; pipeline stages,
tensor parallelism, expert parallelism and replicas are different programs, so name the one that
runs and never call PP or replicas TP.

## Request shape

Any. What changes is placement: the route decides which spec programs, caches and admission paths
are available.

## Recommended model and card

| Model | Topology | Selected by | Run it |
|---|---|---|---|
| [Step-3.7-Flash 196B-A11B](../models/step37-flash.md) | PP-2 on 2x RTX PRO 6000 | `MEMRA_PP_STAGES=2`, `MEMRA_PP_DEVICES=0,1` | [cookbook](../COOKBOOK.md#step-37-flash-196b-a11b) |
| [DeepSeek-V4-Flash](../models/deepseek-v4-flash.md) | TP/EP on 2 cards (default); PP-2 with `MEMRA_DSV4_TOPOLOGY=pp` | its own serve door | [model card](../models/deepseek-v4-flash.md) |
| [GLM-5.3-Flash](../models/glm53-flash.md) | TP-2 or TP-4, plain decode only | `MEMRA_GLM5_TP` with `MEMRA_SERVE_SPEC=0` | [model card](../models/glm53-flash.md) |
| [Hy3](../models/hy3.md) | automatic whole-expert EP-4 on 4 cards | `MEMRA_PARALLEL=auto`, `MEMRA_PARALLEL_DEVICES=0,1,2,3` | [model card](../models/hy3.md) |

`MEMRA_STEP_TP` and `MEMRA_STEP_EP` are Step correctness seams, not serving routes. The Hy3 PP-4
wavefront stays off.

## Settings that matter

- `MEMRA_PARALLEL=auto` with 2 to 4 distinct `MEMRA_PARALLEL_DEVICES`: plan-driven placement that
  picks a pipeline or expert-parallel backend before any CUDA allocation.
- `MEMRA_PP_STAGES`, `MEMRA_PP_SPLITS` and `MEMRA_PP_DEVICES`: explicit pipeline cuts and placement.
- `MEMRA_PP_HOST_BOUNCE=1`: pinned host transport; it turns the prefix cache off.
- On a sharded PP-2 placement the spec gate defaults to LOW=0 / HIGH=1, so spec admission is off.

Flag detail: [FLAGS.md](../FLAGS.md). Placement policy for 2 to 4 RTX PRO 6000 cards:
[PRO6000-MULTICARD](../decisions/PRO6000-MULTICARD.md).

## How to validate

Check the placement boot line (for PP, `[pp] cross-device transport: stage0=dev0 stage1=dev1`),
then run the exactness gates on the exact topology. Pair-only surfaces need a pair; a one-card
result says nothing about them ([TESTING.md](../TESTING.md)).

## Read next

[Serving: PP-2](../SERVING.md#pipeline-parallel-pp-2-serving), the
[RTX PRO 6000 card](../rigs/rtx-pro-6000-blackwell.md).
