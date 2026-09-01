# RTX PRO 6000 2–4 card serving lane

Base: `origin/main` at `18954872117d6fda05ecf834e6bf4186a496dc4c`.

## Hypothesis

On a PCIe-only 2–4 card RTX PRO 6000 host, stage-local pipeline parallelism is the safest
shared-model baseline for both dense and MoE models. A request/prompt wavefront should recover
multi-card utilization at concurrency without paying TP/EP communication inside every layer.

## Implemented before the hardware window

- Header-only exact checkpoint byte census for GGUF, safetensors, and repack manifests.
- Deterministic 2–4 stage contiguous cost planner over legal ModelPlan boundaries.
- PP3/PP4 decode and prompt-microchunk wave scheduling behind `MEMRA_PP_WAVE=1`.
- One generation owner lease for shared PP boundary slots, retained by deferred results and
  borrowed only inside explicit coordinator phases; unrelated re-entry fails closed.
- Numeric device-ordinal duplicate detection (`0,00,1` is a duplicate placement).
- N-stage per-device admission, including Step TP KV, per-stage learned fixed residency, and only
  missing capacity in the process-global boundary slots (including enabled concat-prime width).
- ModelPlan pipeline capability and selected-cut legality refusal before sharded weight upload.
- Transactional fail-stop cache tainting: a partial wave aborts every affected session and cannot
  retry, demote, restore, or enter a reuse pool.
- Cross-device generic dense/non-Step concat prime is excluded until it owns a real PP split;
  individual prompt wavefronts remain the safe path.
- Operator-only wave engagement metrics and PP3/PP4 gate harness plumbing.

## Local gates

The following are development checks, not RTX PRO 6000 qualification:

```text
cargo check --workspace                                      PASS
cargo test -p memra-engine --lib pp_wave                   10 PASS
cargo test -p memra-engine --lib ppn_prime                  3 PASS
cargo test -p memra-server --lib pp_                        18 PASS
cargo test -p memra-gguf --lib placement::tests             6 PASS
cargo test -p memra-gguf --lib census                        3 PASS
cargo check -p memra-engine --bin decode-batch-gate \
  --bin kernel-check                                         PASS
```

The complete engine library was independently reviewed at the anti-diagonal checkpoint: 259
tests passed and 2 were ignored. That review found the missing whole-walk owner lease, numeric
device-alias bypass, exact-scope restoration defect, and per-diagonal thread/barrier tax. All four
were fixed before hardware work; the replacement uses one scoped worker per stage and exact
two-slot acknowledgements. The current post-review libraries pass: engine 270 / 2 GPU-only
ignored, server 416, GGUF 171, KV 17, and CLI 9.

## Hardware order

1. Topology, negotiated links, directed P2P capability, byte integrity, and simultaneous copy
   bandwidth.
2. PP-2 regression on the exact branch and artifact.
3. PP-3 then PP-4 engine exactness: eager, batched wave, prime wave, spec verify, rollback.
4. Sampled serving: vendor defaults, unique row seeds/counters, penalties and grammar masks.
5. Context/admission/prefix/rollback/disconnect gates and a multi-turn cache twin.
6. Interleaved N>=5 serial-vs-wave curves with TTFT/E2E/TPOT/ITL tails, request/output-token
   throughput, failures, power, thermals, and clocks.
7. MoE candidates only after the PP baseline: projection-sharded MoE-TP, whole-expert EP, and
   legal ETP hybrids. A candidate replaces PP only on measured exact-host wins.

## Current verdict

Infrastructure is compile- and unit-gated. PP-3/PP-4 remains **UNQUALIFIED / DEFAULT OFF** until
the hardware order above is complete. No performance number is claimed from local tests.
