# Kernel-check gate integrity results

Date: 2026-08-11

Branch: `lane/cx-kcheck`

Verdict: **PASS**

## Outcome

`kernel-check` no longer reports an unqualified green result when optional model-backed cells
disappear. Every unavailable modeled cell emits `SKIP <cell-name> (<reason>)`; successful runs
end with `ALL GREEN (<n> cells, <m> skipped)`. Repeatable `--require-cell NAME` and
`--require-manifest FILE` requirements turn skipped or absent required cells into
`MISSING REQUIRED CELL <name>` and a nonzero exit before any green summary.

Model paths supplied through `MEMRA_KC_MODELS_DIR`, and matching explicit model arguments, are
authoritative. A typo cannot fall through to stale bytes elsewhere on the host. Resolved artifacts
with missing or unsupported tensors also receive explicit verdicts. In particular:

- `d2-cache-bit-identity` and `fast-router-batch` now report `SKIP` instead of panicking when their
  pinned tensors are missing or incompatible.
- `nvfp4-gemm` accounts separately for its Q5_K, NVFP4, and target-gated native/static coverage;
  partial tensor coverage cannot credit the whole section.
- The 27B gate manifest requires `DUAL-BATCHED-AUX`; the Step35 manifest requires
  `fa_prefill_view_ws_w_hd128`, `attn_head_gate`, and `swiglu_clamped`.
- `tools/local-ci.sh` and all current mechanism-specific kernel-check callers use the new summary
  contract; the 27B and Step35 callers pass their respective manifests.

## Verification

| Gate | Result | Evidence |
| --- | --- | --- |
| `cargo build --release -p memra-engine --bin kernel-check` | PASS, sm_120a release build | `raw/cargo-build-release.log` |
| Focused unit tests | PASS, 4 passed / 0 failed | `raw/cargo-test-kernel-check.log` |
| Changed shell scripts, `bash -n` | PASS, 17 scripts | `raw/bash-n.log` |
| Full local RTX 5090 kernel-check under `/tmp/memra-gpu.lock` | PASS, `ALL GREEN (98 cells, 0 skipped)` | `raw/kernel-check-full.log` |
| Bogus model root plus required `DUAL-BATCHED-AUX` | PASS negative proof: explicit model skips, required-cell error, exit 1, no green summary | `raw/kernel-check-negative.log`, `raw/kernel-check-negative-status.log` |

The positive command required both manifests and cleared all kernel-check scoping/model-root
environment variables:

```sh
flock -w 1800 /tmp/memra-gpu.lock timeout 3600 env \
  -u MEMRA_KC_FAST -u MEMRA_KC_ONLY -u MEMRA_KC_MODELS_DIR \
  target/release/kernel-check \
  --require-manifest tools/kernel-check-27b.cells \
  --require-manifest tools/kernel-check-step35.cells
```

The negative run set `MEMRA_KC_ONLY=nvfp4-batched`, pointed `MEMRA_KC_MODELS_DIR` at a fresh empty
temporary directory, and passed `--require-cell DUAL-BATCHED-AUX`. Its dispositive lines are:

```text
SKIP nvfp4-batched (missing model Qwen3.5-9B-NVFP4-MTP-GGUF.gguf under MEMRA_KC_MODELS_DIR=...)
SKIP DUAL-BATCHED-AUX (missing model Qwen3.5-9B-NVFP4-MTP-GGUF.gguf under MEMRA_KC_MODELS_DIR=...)
MISSING REQUIRED CELL DUAL-BATCHED-AUX
Error: "1 required cell(s) missing"
```

All raw logs are content-addressed in `raw/SHA256SUMS`. No kernel math, performance board value,
or generated performance surface changed. `cargo fmt` was not run.
