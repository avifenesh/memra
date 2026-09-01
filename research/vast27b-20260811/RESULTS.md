# Vast 27B fused-aux pre-release battery — final verdict

- Date: 2026-08-11
- Lane: `lane/cx-vast27b`
- Verification rig: Vast 2x NVIDIA RTX PRO 6000 Blackwell Workstation Edition
- Tested source: `c58ebd6257334c7b2628ec7367efd4713e8126c1` (detached, exact)
- Merged default under test: `MEMRA_NVFP4_AUX_DUAL` enabled unless explicitly set to `0`

## Verdict

**PASS — keep the fused-aux default enabled. Do not revert `c58ebd62`.**

The required PRO 6000 exactness battery is green: `kernel-check` ran the explicit
`DUAL-BATCHED-AUX` cell with zero bad bits and ended `ALL GREEN`; both 27B `run-gen` argmax
comparisons matched; and `run-spec` passed target self-consistency at every K from 1 through 8.

The one-lock interleaved N=5 performance block was flat, not a reproduced speedup and not a
regression: the rollback arm (`MEMRA_NVFP4_AUX_DUAL=0`) had a 163.88 tok/s median, while the naked
default-on arm had a 163.84 tok/s median, a **-0.0244%** median difference. Paired signs were mixed
and all ten outputs were identical. Under the lane's predeclared rule, a flat result with green
exactness does not trigger a revert. The local 5090's +1.15% gain therefore does not transfer as a
positive performance claim to this rig, but the remote evidence also does not establish a default-on
regression.

This lane changes no published performance-board number and makes no tag or release.

## Frozen provenance

The isolated checkout lived at `/workspace/cx-vast27b/memra`; the production checkout and server
binary were not replaced. The release build used Rust 1.97.1 and CUDA 13.1.115, auto-detecting
`sm_120a`, and completed in 2m33s.

| Input or binary | SHA-256 |
|---|---|
| Qwen3.6 27B target GGUF | `d8d71c7e8a01a1c964fd904a7b496eaef19bdd66827e0949e66c723da742d517` |
| External MTP draft GGUF | `b445fbb139e72f9869df06f2f0f91bcaf57527ec34a24bec74d3febd719f3581` |
| Pinned p1 code-short prompt | `6e00d76296069277dc7717115f977aedcab502b610c95a042c63c30eefdb86b2` |
| `kernel-check` | `9fe97e15755da8e04e3b629783f6959a15776ccb989ba22acfaf52e8a6ac6433` |
| `run-gen` | `0cd19a45601798edc695887f9f551dc62943407d1e5a44e8f6a729bf3fe9b2bc` |
| `run-spec` | `963f942158ed298a95930a2357299769ce88b9e498ce3029647157d76901dcc5` |

The complete on-box target, draft, and prompt hashes were checked before the gates, before the A/B,
and again at final handoff.

## Exactness battery

The final battery ran under one uninterrupted GPU lock on the naked default-on path.

| Gate | Result |
|---|---|
| `kernel-check` | `DUAL-BATCHED-AUX [NVFP4 rp] out=48 m=3: bit-bad=0/0 OK`; `ALL GREEN` |
| 27B `run-gen` | prefill/decode argmax `8160 == 8160` MATCH; batched-prime/tokenwise argmax `8160 == 8160` MATCH |
| `run-spec` K=1..8 | 8/8 target self-consistency PASS |

The first protocol attempt is retained but is not the scored battery. It exited zero from
`kernel-check` with `ALL GREEN`, yet the required dual cell was absent because at `c58ebd62` that
cell is nested under a historical 9B filename resolver. The lane wrapper correctly stopped before
`run-gen` or `run-spec`. For the complete rerun, an explicit symlink named
`Qwen3.5-9B-NVFP4-MTP-GGUF.gguf` resolved to the already-hashed 27B target. The wrapper confirmed
the resolved path and the full target SHA-256 before running. This made the required real NVFP4
gate/up tensors visible to the cell without changing the tested source, binary, or model bytes.

## Controlled A/B

Protocol: K=3, NGEN=64, chat template enabled; one lock for all ten runs; frozen order
`A,B,B,A,A,B,B,A,A,B`; N=5 per arm. A was the explicit rollback
`MEMRA_NVFP4_AUX_DUAL=0`; B was the naked default-on environment. The production server remained
resident and idle; only the soak process was stopped for the bounded measurement block.

| Repetition | A: dual disabled (tok/s) | B: default-on (tok/s) | Paired B/A delta |
|---:|---:|---:|---:|
| 1 | 164.03 | 161.84 | -1.3351% |
| 2 | 163.88 | 163.84 | -0.0244% |
| 3 | 163.90 | 163.95 | +0.0305% |
| 4 | 163.81 | 164.06 | +0.1526% |
| 5 | 161.84 | 162.48 | +0.3955% |
| **Median** | **163.88** | **163.84** | **-0.0244%** |

Every run accepted 42/63 positions (66.7%) in 21 rounds, reported 3.000 tokens per round, and
passed target self-consistency. All ten target-token arrays had the same SHA-256:
`899099d0e6017b8715f9b015715994deba0f216304ebd34794a7f2c33fe78479`.

Thermal regime: one continuous 500 ms trace covered the scored block from 03:01:36Z through
03:02:41Z, with 130 samples per GPU. Test GPU 0 warmed from 45 C and reached 57 C, drawing
88.57--438.33 W. GPU 1, which remained production-only, stayed at 39--44 C and 0%
reported utilization. This was a warming interleaved block, not a claimed steady-state plateau;
the fixed order and mixed paired signs are therefore reported rather than normalized away.

## Production handoff

The production `memra-server` (PID 13667) stayed resident throughout the build, correctness gates,
and A/B. The A/B stopped only `/root/soak.py`, then its exit trap restarted it as PID 16815 and
verified `/health`, `/readyz`, `/v1/models`, and a completed streamed request.

An independent final receipt at `2026-08-11T03:06:38Z` rechecked the full artifact hashes and
recorded:

- Step `:8002`: `status=ok`, `status=ready`, worker `phase=idle`, `xid_warnings=0`;
- served model: `stepfun/step-3.7-flash`;
- streamed completion: done, 32 completion tokens, 53 total tokens;
- soak: PID 16815 still live and a fresh log row completed with an empty `err` field.

## Raw evidence

- Build and toolchain: `raw/build-20260811T023608Z/`
- Initial non-scored resolver attempt: `raw/gates-20260811T025338Z/`
- Complete exactness battery: `raw/gates-20260811T025740Z/`
- Interleaved A/B and continuous GPU trace: `raw/ab-20260811T030108Z/`
- Independent production handoff: `raw/final-20260811T030550Z/`
- Artifact and source staging receipts: `raw/artifact-manifest.log`,
  `raw/remote-source-stage.log`, and the transfer logs in `raw/`
- File-level evidence manifest: `raw/SHA256SUMS`
