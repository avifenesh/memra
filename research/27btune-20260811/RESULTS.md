# Qwen3.6 27B NVFP4 decode tuning — final verdict

- Date: 2026-08-11
- Lane: `lane/cx-27btune`
- Development rig: local RTX 5090 Laptop GPU (`sm_120a`)
- Base commit: `429ef3d5d5ca2ecaa96026386f182969439646a1`
- Promoted commit tested: `febb2b98c8f77e1fdbfa2f574047d3a1691f6d94`

## Verdict

**Promote for local development verification:** fuse each linear-attention layer's tiny NVFP4
`ssm_beta` and `ssm_alpha` decode projections into one dual launch. The naked K=3 path improved the
one-lock, interleaved N=5 median from 95.98 to 97.08 spec tok/s, **+1.15%**, clearing the lane's 1%
decision floor. Every paired repetition favored the candidate, and all ten runs preserved the same
target tokens, acceptance, round count, and self-consistency result.

The mechanism subsequently merged default-on in `c58ebd62`. The required Vast 2x RTX PRO 6000
pre-release battery has now passed on that exact commit: the remote exactness gates are green and
the one-lock N=5 default-on/rollback comparison was flat at -0.0244% (163.84 versus 163.88 spec
tok/s), not the local rig's +1.15% and not a measured regression. Per the predeclared decision rule,
the default remains enabled. The performance board is unchanged because neither lane moved a
published number. The complete remote verdict is in `../vast27b-20260811/RESULTS.md`; scored raw
receipts are under `../vast27b-20260811/raw/gates-20260811T025740Z/` and
`../vast27b-20260811/raw/ab-20260811T030108Z/`.

## Promoted mechanism

The affected shape is the 48 linear-attention layers' two 48-row auxiliaries: 96 separate grid-12
launches per speculative round. `qmatvec_nvfp4_mmvq_dual_b4_rp` keeps the existing one-row-per-warp
split-plane template body and reduction order, but combines each sequential beta/alpha pair into one
`grid=(12,2,1)` launch. The full-attention K/V projections and the large `b4_rpr2` carrier are
untouched.

The release fatbin reports 44 registers/thread, zero stack/local memory, and 1,024 bytes shared
memory for the new kernel. The default engages only the exact tiny `t=3`, WROWS=1 pair; setting
`MEMRA_NVFP4_AUX_DUAL=0` restores the two original single launches for rollback and A/B work.

## Decision evidence

Protocol: one GPU lock for all ten runs; fixed order `A,B,B,A,A,B,B,A,A,B`; N=5 per arm; same
binary, model, draft, prompt, and settings; empty compute-app census; 54 C/P8 at entry and 71 C/P0
at exit, with per-run starts through 76 C.

| Repetition | Baseline (tok/s) | Candidate (tok/s) | Paired gain |
|---:|---:|---:|---:|
| 1 | 96.89 | 97.63 | +0.76% |
| 2 | 96.22 | 97.44 | +1.27% |
| 3 | 95.98 | 96.98 | +1.04% |
| 4 | 95.81 | 97.08 | +1.33% |
| 5 | 95.85 | 96.94 | +1.14% |
| **Median** | **95.98** | **97.08** | **+1.15%** |

Every run accepted 42/63 positions (66.7%) in 21 rounds and passed target self-consistency. Raw
model-level runs and the driver receipt are under `raw/rp-aux-ab-*.log` and
`raw/rp-aux-ab-driver.log`.

## Rejected arms

| Arm | Controlled result | Verdict |
|---|---|---|
| Eight-resident `rpr2w8` two-stage async refill | Exact cold-weight microbench: +17.56% latency at m=3 and +22.74% at m=4 | Rejected; source and force seam deleted |
| Four-way short-context scalar FA partition | N=5 median 95.45 to 92.26 tok/s (-3.34%); target fold order and acceptance changed | Rejected; source and force seam deleted |

The NCU-visible `qmatvec_nvfp4_mmvq_dual_b4_rpr2` carrier was already near the measured DRAM
ceiling. It was deliberately left untouched because this lane found no multi-token-reuse or layout
mechanism that could justify another prefetch-only arm. The negative-arm records remain in
`arms.jsonl`; their raw logs are retained under `raw/`.

## Final correctness battery

The promoted naked default ran under one uninterrupted lock:

| Gate | Result |
|---|---|
| `kernel-check` | `DUAL-BATCHED-AUX` 48-row bit identity OK; `ALL GREEN` |
| 27B `run-gen` | prefill/decode argmax MATCH; batched-prime/tokenwise argmax MATCH |
| `run-spec` K=1..8 | 8/8 target self-consistency PASS |

The final driver ended `result=PASS`. Evidence is in `raw/final-gates-driver.log`,
`raw/final-kernel-check.log`, `raw/final-run-gen.log`, and `raw/final-run-spec.log`.

## Provenance

- Target model: `/data/ai-ml/hf-models/qwen36-27b-nvfp4-mtp/Qwen3.6-27B-NVFP4-Q4_K_M-mtp.gguf`
  (`sha256:d8d71c7e8a01a1c964fd904a7b496eaef19bdd66827e0949e66c723da742d517`)
- Draft: `/data/ai-ml/hf-models/qwen36-27b-nvfp4-mtp/draft-daily-owntrim-nvfp4head-q4blk.gguf`
  (`sha256:b445fbb139e72f9869df06f2f0f91bcaf57527ec34a24bec74d3febd719f3581`)
- Prompt: pinned code-short p1 (`sha256:6e00d76296069277dc7717115f977aedcab502b610c95a042c63c30eefdb86b2`)
- `kernel-check`: `sha256:420cbb566a5cfb9eb50d0621aa4ea8a6164118fa6a8a4a0867aa2f914bb1962d`
- `run-gen`: `sha256:6686409ae5b28b30a1bf58456a36f5006706d82cd1fbf32af12f1e81a87e41c2`
- `run-spec`: `sha256:227c57adf1b562d15f46dd20b460998df1c4ecfbb8128fde39ec4139a44f96c7`

## Vast pre-release receipt

The designated remote gate is complete on exact source
`c58ebd6257334c7b2628ec7367efd4713e8126c1`. `kernel-check` included the required
`DUAL-BATCHED-AUX` bit-identity cell and ended `ALL GREEN`; both 27B `run-gen` argmax comparisons
matched; and `run-spec` passed self-consistency at K=1..8. All ten A/B runs retained identical target
tokens and acceptance. The final handoff receipt at `2026-08-11T03:06:38Z` records the production
Step service healthy/ready with zero Xids, a completed streamed request, and a live soak producing
fresh successful rows. See `../vast27b-20260811/RESULTS.md` and
`../vast27b-20260811/raw/final-20260811T030550Z/`.
