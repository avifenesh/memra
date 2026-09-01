# cx-sigrouter: device-side sigmoid routing for Step-3.7 (2026-08-11)

Lane `lane/cx-sigrouter`. Measured runtime source: `62b0d629` (later lane commits add only
receipts, progress text, this verdict, and an evidence-bounding source comment). Target rig:
2x NVIDIA RTX PRO 6000 Blackwell Server Edition, CUDA 13.2. Every target GPU command ran under
one of the lane's exclusive `/tmp/memra-gpu.lock` holds.

## Verdict: ADOPT increment 1

Keep the device sigmoid router as the default for Step-3.7/M3/Hy3 and retain
`MEMRA_SIG_ROUTER=0` as the full-logit host-oracle rollback seam.

The exactness battery is fully green, and the fixed interleaved N=5 matrix is positive at both
requested loads: +1.4251% at c1 and +4.2524% at c8 by the medians of the two arms. The lane brief
promotes any positive result once exactness is green, so this clears the adoption bar.

This increment does not claim zero DtoH or universal host/device `expf` bit identity. It moves
sigmoid, correction bias, mask-before-top-k, stable selection, normalization, and scaling onto the
GPU, then reads the tiny `[sel,w]` pair through one pinned stage and one synchronization. The
device API already returns device-resident `sel` and `w` buffers so increment 2 can feed grouped or
staged dispatch without this final readback.

## Contract implemented

For every token row:

1. Compute the un-biased score as `1 / (1 + expf(-logit))` in host-oracle expression order.
2. Remove inactive original expert ids before selection.
3. Rank on `score + correction_bias`, descending; exact keys choose the smaller original id.
4. Return the selected experts' un-biased scores.
5. If `route_norm`, sum and divide in selected-slot order, then apply the routing scale.

The CUDA scalar exponential is adapted from Arm Optimized Routines' MIT-licensed
[scalar `expf`](https://github.com/ARM-software/optimized-routines/blob/v21.02/math/expf.c) and
[table](https://github.com/ARM-software/optimized-routines/blob/v21.02/math/exp2f_data.c), with the
full notice embedded in `moe_router.cu`. Its evaluation order aligns the measured x86_64
[glibc `expf` path](https://sourceware.org/git/?p=glibc.git;a=blob_plain;f=sysdeps/ieee754/flt-32/e_expf.c;hb=glibc-2.39).
The guarantee is selection and weight-bit exactness on the frozen contract corpus plus the
production golden, not a theorem about every possible libm input.

## Exactness evidence

| Gate | Result | Receipt |
|---|---|---|
| Local RTX 5090 release build | PASS | `raw/local-build-kernel-check-arm-expf-adversarial.log` |
| Local fast synthetic battery | `ALL GREEN`; 68/68 router cases, zero id/mask/tie/weight mismatches, max weight gap 0 ULP | `raw/local-kernel-check-fast-arm-expf.log` |
| Box1 full kernel battery | `ALL GREEN`; same 68/68 exact router result | `raw/box1-correctness-arm-expf/kernel-check.log` |
| Step-3.7 `run-gen` PP-2 | prefill/decode argmax MATCH; batched-prime/tokenwise argmax MATCH | `raw/box1-correctness-arm-expf/run-gen.log` |
| Step-3.7 `run-spec` PP-2 | self-consistency PASS at K=1..8 | `raw/box1-correctness-arm-expf/run-spec.log` |
| Fresh server boots | 10/10 exact `21b8293f2298978c74fb89f32d9b14e3ea921f39924cfce88c73b01f445bb6de`, zero divergences | `raw/box1-correctness-arm-expf/golden-boot-*/qos-summary.json` |

The 68-case corpus includes every Step batch width `t=1..64`, 288 experts/top-8, bias on/off,
normalization on/off, active masks, stable exact ties, 32/256-expert controls, and two additional
Step-width adversarial inputs:

- adjacent representable logits around zero that collapse to one sigmoid-score plateau;
- unequal logits corrected onto exact and one-ULP-adjacent `score + correction_bias` keys, with an
  otherwise-winning masked expert.

The failed variants were retained rather than hidden. CUDA `expf` selected the right ids but left
79 weight mismatches (maximum 2 ULP) locally. Casting `exp(double)` back to float passed the first
synthetic corpus but failed the production golden (`91c89c65...`); the same binary with
`MEMRA_SIG_ROUTER=0` reproduced `21b8293f...`. Route traces located the first difference at one
weight ULP in layer 12, before later selection drift. The adopted evaluation is the first candidate
to pass both adversarial synthetic coverage and all production gates.

## Performance evidence

Metric: `decode_window_tok_s`, 512 generated tokens per request. N=5 per arm and concurrency,
under one exclusive lock hold; a fresh server and one warmup burst per arm; default/rollback order
alternated by repetition; c1/c8 order reversed between arms; continuous 250 ms NVML sampling.
Observed GPU temperatures across the measured points were 32-46 C. The Nsight observations were
run afterward and are excluded from these medians.

| Load | Default samples (tok/s) | Rollback samples (tok/s) | Arm medians | Delta | Paired wins |
|---|---|---|---|---:|---:|
| c1 | 84.4304, 84.4209, 84.3871, 84.5200, 84.4998 | 83.2441, 83.4964, 83.1564, 83.3386, 82.6875 | 84.4304 vs 83.2441 | **+1.4251%** | 5/5 |
| c8 | 162.9695, 162.5537, 163.0338, 162.9924, 163.1364 | 156.4527, 156.3440, 156.4139, 156.2890, 155.5708 | 162.9924 vs 156.3440 | **+4.2524%** | 5/5 |

Paired-delta medians are +1.4251% at c1 (range +1.1072% to +2.1918%) and +4.2323% at c8
(range +3.9718% to +4.8632%). All 90 measured requests completed at the required length with zero
errors. The authoritative reduction is `raw/box1-perf-arm-expf/summary.json`; per-request and
summary rows are retained in `points.jsonl`.

## Transfer and synchronization receipt

Both Nsight runs use the same prompt, `run-gen`, PP-2 placement, and one generated token. Exact
router-sized DtoH groups from the exported CUPTI tables are:

| Router readback | Default | Rollback | Interpretation |
|---|---:|---:|---|
| Decode-sized | 32 B x 18,144 | 1,152 B x 9,072 | two top-8 outputs replace one 288-float logit row |
| Prefill-sized | 2,912 B x 420 | 104,832 B x 210 | two `[91,8]` outputs replace one `[91,288]` logit matrix |
| Router payload total | 1,803,648 B | 32,465,664 B | **94.44% fewer DtoH bytes** |
| `cuMemcpyDtoHAsync_v2` API calls, all sizes | 18,785 | 9,503 | expected doubling from separate `sel` and `w` copies |
| `cuStreamSynchronize` calls | 9,511 | 9,511 | unchanged: one sync per routed readback remains |

The unrelated 515,584-byte DtoH group is identical at 221 calls in both traces. Thus the measured
increment eliminates every full-router-logit transfer and the CPU sigmoid/sort work, but it does
not eliminate the routing synchronization. Device-resident dispatch is the explicitly deferred
increment 2 and is not required for this verdict.

Primary profiler reports `nsys-{default,rollback}.nsys-rep` have hashes `2baa94be...` and
`0f14e942...`; extracted memcpy JSON, CUDA API summaries, and both report/export hash manifests are
committed beside the performance logs. The repository intentionally ignores `*.nsys-rep`, so the
primary reports and derived SQLite exports (103,665,664 and 100,073,472 bytes) remain on box1 at
`/home/ubuntu/memra-cx-sigrouter/research/sigrouter-20260811/raw/box1-perf-arm-expf/` rather than
being force-added. The committed manifests pin their bytes, and the extracted JSON/CSV pins the
receipt used for this verdict.

## Changed runtime surface

- `crates/memra-engine/cu/moe_router.cu`: sigmoid/bias/mask/stable-top-k/weight kernel.
- `crates/memra-engine/src/lib.rs`: device-returning API plus one-sync pinned host twin.
- `crates/memra-engine/src/hybrid.rs`: resident correction-bias and active-mask rows.
- `crates/memra-engine/src/hybrid_forward.rs`: default Step/M3/Hy3 routing integration and
  rollback seam; grouped/staged expert dispatch remains unchanged.
- `docs/FLAGS.md`: documents `MEMRA_SIG_ROUTER=0`.

No published perf board was changed: these are controlled lane measurements, not a board-moving
merge. No merge, tag, release, or push was performed from this isolated lane.
