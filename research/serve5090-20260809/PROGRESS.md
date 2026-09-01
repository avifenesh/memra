# lane/cx-5090-queue — single-card serve floor sweep (write-first)

Date: 2026-08-09. Branch: `lane/cx-5090-queue`. Frozen base and starting tip:
`96a09705895af120a0f706558a8c8c0d6fd8520a` (clean worktree before this ledger).

## Question and stop condition

The merged-this-week serving stack has absorbed default-on kpolicy, prefix dedup/pinning,
microchunks, spec placement, batched decode, and developer-role normalization, but its current
tip has no single-card serve-surface floor receipt. This lane establishes that receipt on the
local proof rig. It does not tune a new winner and it does not move the generated board.

Done means one release `memra-server` built from the frozen tip, all measurements made through
its OpenAI-compatible HTTP/SSE surface, raw logs retained, and `RESULTS.md` carrying:

- cold TTFT for the short and ~4k-token prompt classes;
- cached-repeat TTFT with the server-reported `cached_tokens` retained;
- decode throughput at c=1/2/4;
- default on-policy speculation versus `MEMRA_SERVE_SPEC=0`;
- N=3 interleaving inside one `/tmp/gpu5090.lock` hold; and
- explicit comparisons with `research/serve-ready-20260808/RESULTS.md` (pair receipt, not a
  same-rig denominator) and the frozen v0.72 q27 rows (47.6 tok/s plain; 116.4 / 101.2 /
  86.0 tok/s spec by prompt class). Any like-for-like floor miss is called a regression, not
  hidden by a cross-rig qualification.

## Frozen rig and artifact contract

- Rig: local NVIDIA GeForce RTX 5090 Laptop GPU, 24,463 MiB reported VRAM, one physical card.
- Start state: driver 595.84; CUDA compiler 13.1.115; Rust 1.97.1. No `rustup`.
- [NVIDIA's CUDA 13.1 release notes](https://docs.nvidia.com/cuda/archive/13.1.0/cuda-toolkit-release-notes/index.html)
  require Linux driver >=590.44.01, so the installed 595.84 driver clears the build/runtime
  compatibility preflight.
- Co-resident context at write-first: Hermes gateway PID 7600, 394 MiB. It is owner-approved
  and remains in place. Therefore every performance row is `window_clean=false`; correctness
  gates are unaffected.
- Model:
  `/data/ai-ml/hf-models/qwen36-27b-nvfp4-mtp/Qwen3.6-27B-NVFP4-Q4_K_M-mtp.gguf`
  (15,705,920,064 bytes).
- Daily regime draft:
  `/data/ai-ml/hf-models/qwen36-27b-nvfp4-mtp/draft-daily-owntrim-nvfp4head-q4blk.gguf`
  (1,242,867,296 bytes).
- Device selection is single-card only. No PP split and no bench binary may supply a scored
  row. The only explicit behavior control in the comparison is `MEMRA_SERVE_SPEC=0`; server
  capacity/context/cache knobs will be recorded as machine configuration, not tuned arms.

CPU-heavy work (release build, hashing, parsing) runs with positive nice priority. GPU runs are
allowed and serialized by one lock held across the complete interleaved sweep. Raw logs capture
the server boot/default-policy lines, client rows, GPU/thermal samples, compute-app state,
artifact and binary hashes, commands/config, and errors before any summary is written.

## Planned interleave

One warmup is excluded and labeled. Then for rounds r1..r3, alternate arm order to avoid a
fixed thermal-order advantage:

1. odd rounds: default on-policy, spec-off;
2. even rounds: spec-off, default on-policy.

Each arm records cold short TTFT, cold ~4k TTFT, the exact cached repeat, and c=1/2/4 streamed
decode. Prompt bytes, request budgets, cache salts, server lifetime/restart policy, and the
lock hold stay fixed across arms. Per-cell medians state N=3 and the observed temperature,
clock, and power regime. A failure without captured stderr is reported as cause unknown.

## Initial state

- [x] Branch, base, worktree, rig, toolchain, model and draft paths verified.
- [x] Prior pair serve-ready receipt and frozen v0.72 q27 anchors located.
- [x] Hourly inbox checked at start; `~/.lanectl/inbox/cx-5090-queue.md` did not yet exist.
- [x] Commit this write-first ledger before building (`9746d52a`).
- [x] Build and hash release `memra-server`.
- [x] Run the full serve smoke plus the three named q27 correctness gates.
- [x] Execute and parse the single-lock N=3 serve sweep.
- [x] Write `RESULTS.md` and verify raw-log completeness.
- [x] Commit the completed receipt (`ce5848ac`).

## Execution closeout

- Release server build completed at 22:11 UTC. Binary sha256:
  `c44c93db5ca5a95994f592390976956ee2d1d361d9019aad8099f5631f54699e`.
- The scored sweep ran from 22:34:19 through 22:42:06 UTC under one lock. All 18 decode
  points completed with zero errors and zero sheds; every server log passed the fatal-signature
  scan. The three spec-off arms each passed exact four-request fanout accounting (one cold
  leader, three full-prefix followers) and emitted `retained=true`.
- `tools/serve-smoke.sh` completed 44 checks with zero failures on the q27 trunk+draft.
  Freshly built tip binaries also returned `kernel-check` ALL GREEN, both q27 `run-gen`
  argmax comparisons MATCH, and q27 `run-spec` K=1..8 self-consistency PASS.
- Three pre-score shakeouts remain in `raw/` and are explicitly excluded. The first captured
  real CUDA OOMs after the original harness retained eight salt-keyed spec sessions; bounded
  namespaces plus the documented 24.5GB machine profile (`MEMRA_REUSE_POOL=1`) removed that
  artifact. The second exposed a synthetic-token client assumption. The third was clean but
  its final harness assertion incorrectly expected `B=4`; the server correctly reports the
  three deduplicated followers as `B=3`. The scored run started fresh after all three harness
  fixes.
- The final verdict is in `RESULTS.md`: the receipt is complete, but exact-repeat reuse under
  default speculation and the c=4 policy crossover are red default-surface findings.
