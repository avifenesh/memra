# Native qualification runner for #542

Prepared on 2026-09-20. No GPU execution is claimed by this runbook or the local type checks.

## Resource request

Initial session: exactly **one RTX PRO 6000 Blackwell, 96 GB**, Linux x86_64, CUDA 13.1 or newer,
16 vCPUs, 64 GB host RAM, and 150 GB free fast scratch storage. RTX 5090 32 GB can carry development
rows, but does not replace the PRO qualification target. Record the actual storage backing; do not call overlay storage physical NVMe.
The first host exposed overlay-backed scratch storage. The replacement exposes XFS on
Ceph RBD-backed storage; neither proves physical local NVMe. These admission
checks are not a spill-throughput qualification. The single-card admission test does not require a second GPU. A 96 GB sm_120 Max-Q
variant is sufficient for these correctness/memory rows; record its exact hardware and
do not transfer timings to a full-power card.

The later pipeline/paired release stage requests **exactly two PRO 6000 cards as one set**.
The current owner instruction requires per-physical-card exclusive locks and supersedes the
older blanket per-rig rule for this work. Other cards may have independent owners; record their
activity rather than claiming they were idle.

## Fixed initial artifact

- `Qwen/Qwen3-0.6B@c1899de289a04d12100db370d81485cdf75e47ca`
- Native safetensors source, with config/tokenizer files from the same revision.
- `model.safetensors`: 1,503,300,328 bytes.
- SHA-256: `f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b`.

This artifact is a real loaded-checkpoint witness for identity admission. It does not replace
q9 embedded-MTP or multi-card model qualification. The existing q9 roster file is
`Qwen3.5-9B-NVFP4-MTP-GGUF.gguf`; its exact owner-supplied artifact and byte manifest are still
required. A different quantization is not a substitute.

## Prepare and build before acquiring GPUs

Run from this issue's isolated checkout on the centrally provided host. Python 3.10 or
newer is sufficient; hashing uses bounded streaming reads:

```sh
python3 research/modelplan-onboarding-rewrite-identity-20260920/qualify-native.py \
  --prepare /scratch/memra-542/qwen3-0.6b

MEMRA_CUDA_ARCH=120a cargo build --release \
  -p memra-cli --bin memra \
  -p memra-engine --bin rewrite_identity_gate --bin run-gen \
  --bin decode-batch-gate --bin run-spec --bin kernel-check
```

`DOCS_RS` must be absent. Documentation-only placeholder builds cannot qualify a CUDA program.
Do not start build/cache daemons inside the GPU lock wrapper.

## Execute with the supplied wrapper

Wait for the coordinator's host and self-tested wrapper. Substitute the full physical UUID
assigned to this session, not a guessed device index:

```sh
memra-gpu-run --gpus GPU-<assigned-full-uuid> \
  --receipt /scratch/memra-542/lease-admission \
  -- python3 research/modelplan-onboarding-rewrite-identity-20260920/qualify-native.py \
  --model /scratch/memra-542/qwen3-0.6b \
  --out /scratch/memra-542/admission-001
```

The runner verifies the wrapper and child PID are ancestors, CUDA_VISIBLE_DEVICES exactly
matches the requested UUID order, and `/proc/locks` proves the wrapper holds exclusive
whole-file FLOCKs on `/tmp/memra-gpu-locks/<GPU-UUID>.lock`. It refuses extra cards, a partial
set, another process's lock, an absent lease, and pre-existing compute activity on its card.
It does not acquire its own alternate lock. All GPU commands stay under the same wrapper.

The runner captures 250 ms all-card telemetry, hashes the actual binaries and checkpoint,
and retains stdout/stderr before inspecting results. A negative case passes only when the
process refuses for the named identity mismatch; OOM, startup failure, and unrelated errors
cannot satisfy the negative controls.

## Cases and limits

The same native gate executable measures eager logits on real checkpoint weights, writes its
scoped v2 receipt, and installs it on the loaded model. It exercises eager-only graph refusal,
failed reinstall revocation, and actual eager output after correct admission. Fresh-process
cases cover a matching bundle, missing bundle, changed weight payload with unchanged shape,
changed executable bytes, and changed numerical settings. Bundle file/index hashes stay
internally consistent in the negative controls.

`run-gen` and `decode-batch-gate --mode config` provide additional legacy regression rows on
the same pinned source. These independent executables do not consume the gate executable's
qualification receipt. They are not silently described as qualified serving binaries.

Optional `--mtp-model /absolute/q9.gguf --mtp-sha256 <exact-owner-sha>` runs the standing
`run-spec` K=1..8 test on that exact artifact. Without it, MTP remains pending. Kernel-check,
paired pipeline checks, full serving-shape checks, and balanced N>=5 strict-vs-legacy serving
performance are separate required rows before integration. Per-case wall times in `cases.json`
are diagnostics, not serving throughput or a performance/default verdict. No model support
state is advanced by this runner.

## CPU verification of the runner

```sh
python3 research/modelplan-onboarding-rewrite-identity-20260920/test_qualify_native.py
python3 research/modelplan-onboarding-rewrite-identity-20260920/run-host-tests.py
```

The first suite tests lease/lock rejection without executing a GPU command. The second tests
the actual runtime identity module. Neither is GPU evidence.
