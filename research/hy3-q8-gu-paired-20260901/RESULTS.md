# HY3 paired Q8 gate/up schedule

Status: **QUALIFIED FIX** on 2026-09-01. This restores the admitted Q8 gate/up
numeric class; it is not a 100 tok/s claim and does not enable the parent Q8 arm.

## Bound tuple

- Tested oracle engine tree: `7a7974c374e625cb858de30690b52151fb7a8e24`.
- Tested sampled-server engine tree: `8f17a824d4fb856bc024312264db2bd4a0601ca7`;
  server tree: `7b8c6f73b291701739725c358c232aaf75506f3c`.
- Receipt-parent runtime engine tree: `32a4c03ad65986ba5c0ad96783deecd6b5799ab2`;
  server tree: `f35927b78081cb221dc6ccd2aca8db50ba5f6afc`.
- Last compiled rebased runtime before scope cleanup used the same engine tree and server tree
  `527c597a8d142f9620eae07bb372d176e79d2877`; binary SHA-256
  `d7c3b3fe674d1111b67188e4967af4c209a82fbec6d1d763b5f2476dd4776f9f`.
  The only receipt-parent delta from that compiled server tree restores origin/main's
  `loop` spelling in `host_handoff_export` instead of carrying an unrelated clippy-only
  `while let` rewrite. The HY3 engine tree and the critical server library blob are unchanged;
  exact-head CI binds the cleaned server tree.
- Artifact: `Tiyuvta/Hy3-NVFP4` revision
  `4e8bbadbdb97b5402cb5a3f997d941946b97c5b5`.
- Artifact index SHA-256:
  `0f22f6fc51ac7e39b7510a77c77098c4fd7c722e9e6cfdb9782247c37f1b6afd`.
- Hardware: 4x NVIDIA RTX PRO 6000 Blackwell Server Edition, 96 GB each.
- Serving shape: automatic EP4, device router, gate/up-only internal Q8,
  shared-expert overlap, `MEMRA_SERVE_SPEC=0`.
- Performance requests supplied no sampling fields; the artifact defaults were used.

The receipt commit that contains this file is documentation-only after the rebased runtime
code. Review must prove `git diff --quiet HEAD^..HEAD -- crates` before accepting that
binding. The tested and rebased HY3 served paths are bound below; rerun the gate if an
executable delta enters those paths.

## Finding

The previously admitted schedule used one CTA per output row for both gate and
up. Integration later changed the Q8 gate/up schedule to separate
gate and up CTAs. The two schedules are not interchangeable for this numeric
class on the served artifact.

The separate schedule produced a first-token flip on prompt 2 and degenerate
sampled output. Restoring the paired schedule fixes both. Exact W4A16 remained
fluent, localizing the regression to the Q8 gate/up schedule rather than the
artifact, router, down projection, or attention path.

## Same-binary causal A/B

The causal A/B used one binary; the response hashes are sealed in the summary receipt.

| arm | first tokens | sampled c1 | sampled c4 aggregate | verdict |
|---|---|---:|---:|---|
| paired CTA (`=1`) | `Below, Below, Expert, Below` | 35.44 tok/s | 39.17 tok/s | fluent, green |
| separate CTA (`=0`) | `Below, !, Expert, Below` | 36.62 tok/s | 41.34 tok/s | degenerate, reject |

The separate schedule's higher token rate is invalid because the generated text
is corrupted. It is retained only as the rollback/teeth arm.

## Exact-head sampled serving

The default-unset oracle-equivalent control passed at 34.78 tok/s c1 and
35.13 tok/s c4 aggregate. The pre-upstream-rebase control passed at
34.90 tok/s c1 and 35.23 tok/s c4 aggregate.

The exact tested binary, with
`MEMRA_PARALLEL_EP_Q8_GU_PAIRED` unset, selected
`gate_up_schedule=paired-cta` and passed three consecutive vendor-default
sampled probes:

| repeat | sampled c1 | sampled c4 aggregate | first tokens | probe SHA-256 |
|---:|---:|---:|---|---|
| 1 | 34.8773 tok/s | 35.0763 tok/s | `Below, Below, Expert, Below` | `e7d84145177cf9bf43cd498b9bfcb616b63187d6573b2ea90fc4d547043fd7ea` |
| 2 | 34.8791 tok/s | 35.0801 tok/s | `Below, Below, Expert, Below` | `7552f4687b4655ad9157aede7394eccbb2d54f19a23efcd3406e5db439592491` |
| 3 | 34.9277 tok/s | 35.2542 tok/s | `Below, Below, Expert, Below` | `9cdad5ec595c8a9b49d6207b8542b8b755a075ae1e9cce15dbb16d72f0ae9467` |

The medians are **34.8791 tok/s c1** and **35.0801 tok/s c4 aggregate**.
Every probe records an empty sampling-field list, zero cached prompt tokens,
fluent output, and the same four-token signature.

The server log has no CUDA, panic, non-finite, OOM, or fatal-Xid event; the only
word `fatal` is the gpu-watch startup line listing which future Xids would be
treated as fatal. Its SHA-256 is
`6b23d8d607abd29a64822ff3068cb8955e3f57cd27d1e355ae1595c258b95b5a`.

The correctness fix does not claim a throughput gain. It removes an invalid
36.62 tok/s row whose output was degenerate.

Exact tested binary SHA-256:
`ca77d2edef1ab509263e2da62582941a4e13788b7cb7f5bba5d02ac598b6bcde`.
The build log SHA-256 is
`744cd83970d4db9918f63bfd882837fb81192fc7f8a0e9993a125a6b9eb5120f`;
the tested-head transfer delta bundle SHA-256 is
`d137f442b30ff9502bbeb6a56bbde9d464e39d01f69d1d2894a37735760c6078`.

## Prefix-cache and CPU-scheduling checks

The performance boot used `MEMRA_SERVE_BATCH=0`, which bypasses the prefix cache
by design. An identical 224-token prompt consequently reported 0 cached tokens on
both requests. That 0/0 row is a bypass receipt, not a cache qualification result.

The same exact binary was then booted with `MEMRA_SERVE_BATCH=1` and
`MEMRA_SERVE_SPEC=0`. Startup admitted a 2,432,696,320-byte cache budget with a
64-token minimum. The cold request completed 128 tokens in 12.0539 seconds with
0/222 cached tokens. The identical hit completed in 3.2329 seconds with 192/222
cached tokens. The log records both the 192-token probation insert and the
`hit: 192 of 222` engagement marker. Hashes:

- request: `5049dc327e4434d2e80239086fa76be31c68ce2a34a746381465fb788edb36ba`;
- cold response: `b37a2407e6d561726e14d22174797deaf1e42a26a49b3d7190cf6e209944b575`;
- hit response: `09280e9b61ebf178880eddbfd397b6486a20b4ac5a92b61e3c348725a1da20a8`;
- cache-enabled server log: `b91ae43e7409e5771a09464fa8d783f62bad3bdf0f68a6404bd31104f9dd79d2`.

All four GPUs report NUMA affinity 1 while the unpinned server may run on both
nodes. A reversible A-B-B-A c1 check with 38-token prompts, below the cache
minimum, measured 35.4249 tok/s unrestricted versus 35.4298 tok/s pinned to
NUMA node 1. The 0.014% delta rejects CPU affinity as the present throughput wall.

## Full-vocabulary gate and rebase binding

`run-safetensors` captured 120,832 logits for exact W4A16 and paired gate/up Q8
from the sealed oracle engine tree using tokens `[1, 2, 3, 4]`.
The paired-Q8 CUDA kernel, TP schedule, hybrid forward path, and server library
blobs are byte-identical at the exact sampled server and rebased runtime heads.
The later main deltas are outside the tested HY3 path: a Qwen-only module/probe,
an sm_100a build gate plus research-only accumulator instrument, an `MEMRA_FP4`
guard that is inert while the flag is unset, and a default-off hyper-trunk suffix
path while HY3 is not a hyper trunk. The rebased head compiled at sm_120a and its
full server battery passed 527/527. The machine-readable proof is
`receipts/rebase-path-equivalence.json`.

| metric | paired vs exact | gate |
|---|---:|---:|
| finite logits | all | all |
| argmax | 5 == 5 | equal |
| top-20 overlap | 19/20 | >=18/20 |
| cosine | 0.9995610804 | >=0.999 |
| RMSE | 0.0565440 | <=0.25 |
| mean absolute error | 0.0447950 | <=0.10 |
| maximum absolute error | 0.3080940 | <=1.0 |

Oracle hashes:

- exact TSV: `e7eab93f4db98fdce168b50ff6648567cc767443f96a0b183843302d048b9bc6`;
- paired TSV: `db702718ff324964e4acf5ad84ba35a4f22b3f84d3c7478230e43ef6a89761cb`;
- raw comparison JSON: `4fcaf6fa7f85d054d1b7d109f3cf686e508be07ce5f98f673b2e94cdbbc81249`;
- tracked path-normalized comparison JSON:
  `722af8aebd3b4433391454ead91cf9afd62f76e90ddd2ec4000db32786e8bbb7`.

## Decision

`MEMRA_PARALLEL_EP_Q8_GU_PAIRED` defaults ON **inside**
`MEMRA_PARALLEL_EP_Q8_ACT`; `=0` is the strict rollback. The parent Q8 arm
remains globally OFF and retains its existing model-and-hardware admission gates.
