# HY3 RTX4 mint and qualification handoff

Refuse any checkout whose `git rev-parse HEAD` differs from the immutable Memra SHA supplied with
this handoff. The deliverable is our NVIDIA ModelOpt artifact, not a third-party checkpoint.

## Immutable inputs and expected output

```text
source          tencent/Hy3@a960ebc3da325ba167f069f76c41eb62c9280d22
source config   0c9daab42bff9cce1b6f058b10d7b730f76d583e583e28ad56e92b36373246f0
source index    9594f1a9419e62ca7afca51bb644f38ef19039374f7812449381ccf42f0ef79b
ModelOpt        0.46.0
ModelOpt git    43fd41a58d52c4e6e5dec1d1ff5989ecc737ae1a
vLLM oracle    0.28.0 (modelopt_fp4, raw_logits)
output payload  180826481152 bytes
stored tensors  139298
logical tensors 47138
NVFP4 weights   46080 (45504 trunk + 576 MTP)
BF16 MTP expert weights 0
output config   3cb16aa29d0046ffddd2f8a4866e4c7511e4018c6fced8dd913d1a788d787af9
output quant    38e5689cd6847427cc28c26c3cd3ca30568822bf311f479f11d21cf8ab632d2e
output index    0f22f6fc51ac7e39b7510a77c77098c4fd7c722e9e6cfdb9782247c37f1b6afd
output census   566db2975edac5cd1a86061ec6943988ef695cc8ae8c6cda050ad0d354ae2600
```

The generated shard hashes are not knowable before quantization. Compute and sync the full manifest
immediately after mint; that manifest becomes the immutable artifact byte identity.

The deployment declaration is `quant_algo=W4A16_NVFP4`. `NVFP4` alone means W4A4 to NVIDIA's
deployment consumers and is invalid for this weight-only artifact. The compressed-tensors view
therefore omits `input_activations` entirely; it does not encode them as `null`.

## Capacity

- Mint disk floor: 850,000,000,000 bytes free. The source is 597,572,342,272 bytes and output is
  180,826,481,152 bytes. The supplied 900 GB persistent workspace with 966,367,637,504 bytes free
  passes.
- Qualification disk with the default safetensors stream-repack cache: at least 400 GB free before
  copying the artifact, or at least 180 GB free after the 180,866,448,377-byte artifact is present.
  The generated `.memra-repack` expert cache is about 163 GB and is not part of the publishable
  artifact. A smaller host must set `MEMRA_ST_REPACK_DISK=0` and use the supported in-RAM gather
  path; 300 GB container storage is not enough for artifact plus disk cache and build tools.
- Host RAM: 128 GiB minimum, 256 GiB preferred. The supplied 1 TiB passes.
- PP-4 fences: `[0,20,40,60,81]`; exact weight payloads are
  39.884 / 41.514 / 41.514 / 44.574 GiB. Initial load budget is 65 GiB/card including runtime,
  8K KV, and transients. Record actual peak.
- Four full-power RTX PRO 6000 cards, directed P2P, and the peer-integrity runtime probe are
  mandatory. A P2P soak outlier blocks performance claims, not the deterministic mint.

## Exact mint command

```bash
export HY3_MEMRA_SHA='<SHA from handoff message>'
export HY3_MEMRA_REPO=/workspace/memra
export HY3_RUN_ROOT=/workspace/hy3-modelopt
export HY3_ARTIFACT=$HY3_RUN_ROOT/output-experts
export HY3_RECEIPTS=$HY3_RUN_ROOT/receipts-experts

test "$(git -C "$HY3_MEMRA_REPO" rev-parse HEAD)" = "$HY3_MEMRA_SHA"
test -z "$(git -C "$HY3_MEMRA_REPO" status --porcelain)"
test "$(df -PB1 "$HY3_RUN_ROOT" | awk 'NR==2 {print $4}')" -ge 850000000000

cd "$HY3_MEMRA_REPO"
HY3_ACCEL_NONPROD=1 \
HY3_RUN_ROOT="$HY3_RUN_ROOT" \
HY3_MINT_DEVICES=0,1,2,3 \
HY3_MINT_SPOT_EVERY=500 \
  research/modelplan-onboarding-hy3-20260830/run-modelopt.sh experts
```

The wrapper holds `/tmp/memra-gpu.lock`, pins the source and ModelOpt checkout, uses all four GPUs
on disjoint source-shard sets, checks ModelOpt-vs-Memra dequantization every 500 quantized tensors,
requires bit-equal shared `weight_scale_2` for all 15,360 fused gate/up expert pairs, keeps all
non-expert tensors byte-identical, validates the exact expected metadata hashes, runs
`memra model inspect --against hy3_nvfp4`, and writes a complete artifact manifest.

Stop immediately on any source hash, ModelOpt version, tensor count, payload, metadata hash,
dequantization, tokenizer/template, or contract mismatch.

## Receipt preamble after mint

```bash
mkdir -p "$HY3_RECEIPTS/rtx4"
git -C "$HY3_MEMRA_REPO" show -s --format=fuller HEAD > "$HY3_RECEIPTS/rtx4/memra-commit.txt"
nvidia-smi -L > "$HY3_RECEIPTS/rtx4/gpus.txt"
nvidia-smi --query-gpu=index,uuid,name,memory.total,power.limit,power.max_limit,driver_version \
  --format=csv,noheader > "$HY3_RECEIPTS/rtx4/gpu-inventory.csv"
nvidia-smi topo -m > "$HY3_RECEIPTS/rtx4/topology.txt"
nvidia-smi topo -p2p r > "$HY3_RECEIPTS/rtx4/p2p-read.txt"
nvidia-smi topo -p2p w > "$HY3_RECEIPTS/rtx4/p2p-write.txt"
nvcc --version > "$HY3_RECEIPTS/rtx4/nvcc.txt"
free -b > "$HY3_RECEIPTS/rtx4/ram.txt"
df -PB1 "$HY3_RUN_ROOT" > "$HY3_RECEIPTS/rtx4/disk-after-mint.txt"

sha256sum "$HY3_ARTIFACT/config.json" \
  "$HY3_ARTIFACT/hf_quant_config.json" \
  "$HY3_ARTIFACT/model.safetensors.index.json" \
  > "$HY3_RECEIPTS/rtx4/metadata.sha256"
```

Copy `receipts-experts/artifact.sha256`, the inspect bundle, mint log, run lock, pip freeze, GPU
inventory, and RTX4 preamble off-box before loading.

## Build

```bash
cargo build --manifest-path "$HY3_MEMRA_REPO/Cargo.toml" --release \
  -p memra-engine --bin kernel-check --bin run-safetensors --bin run-gen --bin run-spec
cargo build --manifest-path "$HY3_MEMRA_REPO/Cargo.toml" --release \
  -p memra-server --bin memra-server

sha256sum "$HY3_MEMRA_REPO/target/release/kernel-check" \
  "$HY3_MEMRA_REPO/target/release/run-safetensors" \
  "$HY3_MEMRA_REPO/target/release/run-gen" \
  "$HY3_MEMRA_REPO/target/release/run-spec" \
  "$HY3_MEMRA_REPO/target/release/memra-server" \
  > "$HY3_RECEIPTS/rtx4/binaries.sha256"
```

## Gate order

Never overlap model processes. Hold the same `/tmp/memra-gpu.lock` across each cell.

1. Kernel gate on one card.
2. PP-4 load and bounded native forward.
3. Same-artifact external logit oracle (external engine is oracle-only, never a serving fallback).
4. Plain-vs-MTP greedy identity at K=1..8 on real prompts; require nonzero acceptance.
5. Plain Memra server with vendor-default sampling.
6. MTP Memra server with vendor-default sampling and explicit engagement.
7. Concurrency, context/admission, tools, reasoning, cache-on eight-turn, uncapped-successor stress,
   and rollback.
8. External-control quality comparison.
9. Upload only after all required gates pass.

### PP-4 forward

```bash
exec 9>/tmp/memra-gpu.lock
flock -n 9

env CUDA_VISIBLE_DEVICES=0 "$HY3_MEMRA_REPO/target/release/kernel-check" \
  > "$HY3_RECEIPTS/rtx4/kernel-check.log" 2>&1

env CUDA_VISIBLE_DEVICES=0,1,2,3 \
  MEMRA_PP_STAGES=4 MEMRA_PP_DEVICES=0,1,2,3 MEMRA_PEER_PROBE=1 \
  MEMRA_ORACLE_OUT="$HY3_RECEIPTS/rtx4/native-forward.tsv" \
  timeout 3600 "$HY3_MEMRA_REPO/target/release/run-safetensors" \
  "$HY3_ARTIFACT" 1 2 3 4 \
  > "$HY3_RECEIPTS/rtx4/native-forward.log" 2>&1
```

Require all four stage owners, successful maximum-payload peer probes, finite logits, NVFP4
engagement, and source-native BF16 residency. Any Q8 substitution is a failure.

### Same-artifact ModelOpt oracle

Plain `transformers.AutoModel` is not a deployment loader for NVIDIA unified ModelOpt exports and
must not be used for this artifact. Install pinned vLLM 0.28.0 in Python 3.12 (the Hy3/FlashInfer
worker is not Python-3.10-compatible), whose `modelopt_fp4` backend supports W4A16 NVFP4 and
`HYV3ForCausalLM`, then capture all-vocabulary raw logits:

The runner pins `VLLM_USE_FLASHINFER_SAMPLER=0`: the external sampler is irrelevant to raw logits,
and this is vLLM's documented native-sampler fallback for the FlashInfer SM120/toolkit-JIT check.
It captures the final-stage tensor through a tensor-preserving logits processor that returns the
tensor unchanged. Do not request `logprobs`: vLLM V1's PP logprobs return path can hang after model
initialization at zero GPU utilization, while the passive processor bypasses that transport path.

```bash
env CUDA_VISIBLE_DEVICES=0,1,2,3 \
  timeout 7200 python \
  "$HY3_MEMRA_REPO/research/modelplan-onboarding-hy3-20260830/capture-vllm-oracle.py" \
  "$HY3_ARTIFACT" \
  --out "$HY3_RECEIPTS/rtx4/vllm-modelopt-oracle.tsv" \
  --devices 0,1,2,3 \
  --parallel-mode pipeline \
  --moe-backend auto \
  --tokens 1,2,3,4 \
  --numeric-class ModelOpt-NVFP4-W4A16 \
  > "$HY3_RECEIPTS/rtx4/vllm-modelopt-oracle.log" 2>&1
```

Compare it to `native-forward.tsv` with `compare-oracles.py`. The lane must predeclare and enforce
finite logits, argmax, top-k overlap, cosine, RMSE, mean-absolute, and maximum-absolute bounds.
Pipeline mode is the gate because it matches Memra's PP-4 whole-layer placement and avoids a
tensor-parallel reduction-order difference at every projection. A `--parallel-mode tensor` capture
may be retained as a topology-sensitivity diagnostic, but it cannot replace the matched-PP gate.

### MTP lossless gate

```bash
env CUDA_VISIBLE_DEVICES=0,1,2,3 \
  MEMRA_PP_STAGES=4 MEMRA_PP_DEVICES=0,1,2,3 MEMRA_PEER_PROBE=1 \
  MEMRA_NGEN=64 MEMRA_CHAT=1 \
  MEMRA_PROMPT='Write a Rust function that parses a decimal u64 without allocating, then explain its overflow check.' \
  timeout 7200 "$HY3_MEMRA_REPO/target/release/run-spec" "$HY3_ARTIFACT" \
  > "$HY3_RECEIPTS/rtx4/mtp-k1-k8.log" 2>&1
```

Every K must emit target-identical tokens with nonzero acceptance. Greedy loops are excluded from
performance and are not quality findings.

### Plain serving arm

```bash
env CUDA_VISIBLE_DEVICES=0,1,2,3 \
  MEMRA_PP_STAGES=4 MEMRA_PP_DEVICES=0,1,2,3 MEMRA_PEER_PROBE=1 \
  MEMRA_MODELS="hy3=$HY3_ARTIFACT" \
  MEMRA_MODEL_METADATA="$HY3_MEMRA_REPO/research/modelplan-onboarding-hy3-20260830/serve-models.toml" \
  MEMRA_ADDR=127.0.0.1:18082 MEMRA_CTX=8192 MEMRA_MAX_SESSIONS=4 \
  MEMRA_SERVE_SPEC=0 MEMRA_SERVE_BATCH=0 \
  "$HY3_MEMRA_REPO/target/release/memra-server" \
  > "$HY3_RECEIPTS/rtx4/server-plain.log" 2>&1
```

Probe readiness, pinned identity, and a chat completion with **no sampling fields**. Metadata
defaults are temperature 0.9/top-p 1.0. A 200 without PP/NVFP4 log engagement is not a pass.

### MTP serving arm

Stop the plain server cleanly, then use the identical environment with:

```text
MEMRA_SERVE_SPEC=1 MEMRA_SPEC_GATE=0 MEMRA_SPEC_K=1
```

Require MTP and PP-verify engagement in server logs plus sampled vendor-default output. Only after
the full serving battery passes may the artifact be uploaded. The upload/model card is owned by
Darklanes and must name both Memra (engine) and Tiyuvta (hosted inference).
