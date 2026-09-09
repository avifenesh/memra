# DSV4F remaining activation materializations

Base: fefb019bec216aafaa9c7ad0ccfe41661c450981. Issue #4 sublane.

Draft implementation and gates are in progress. No GPU launch is authorized until
independent narrow review and an orchestrator slot relay. No performance verdict.

| Producer and consumer | Scope decision |
| --- | --- |
| Attention-entry RMSNorm to Q_a BF16 pack, then repeated KV pack | Fuse norm and pack while retaining f32 for compressor fan-out; reuse the pack for KV after proving buffer lifetime. |
| FFN-entry RMSNorm to shared-expert BF16 pack | Fuse and retain f32 for router and FP8 quantization. |
| Shared SwiGLU to BF16 pack | Fuse elementwise expressions with explicit f32 then BF16 RNE; sole consumer is unchanged down projection. |
| Expert intermediate FP8 quantization to half gather | Trace exact per-128 scale/code rounding and row scale/status; GEMVs stay unchanged. |
| Input FP8 quantization to routed half gather | Routing and cross-rank ownership intervene. No direct adjacent norm/gather chain. |
| Residual add to norm | HC post/pre and coefficient calculation intervene. HC kernels are owned by another lane. |
| Q norm to pack | Already fused by small-kernel diet. |
| KV norm to RoPE and compressor norm to rotary emission | Existing rotary fusion and rotary kernels are outside this lane. |
| Attention output and grouped projection output BF16 conversion to GEMV | Moving conversion into the GEMV would modify another lane's dense implementation. |
| Final head norm to dots | Consumer projection is outside scope. |

Door: MEMRA_DSV4_NORM_FUSE2, default OFF, decide-by 2026-09-23.
Exact f32 reduction order and all BF16/FP8/FP16 rounding points must survive.
Component raw-bit identity at every fused site on both ranks, memcheck and
synccheck precede separate qualification and unprofiled scoring. Qualification
uses two fresh processes per arm. Scoring uses fresh uncaptured states, first
capture inside timing, 20 ABBA rows and the reverse order. Retained census must
match each forward variant on both ranks; OFF has no new symbols. Timing events
belong on the execution stream outside capture. Engine PR remains draft.

No local rig builds, gates or CI. Push uses MEMRA_SKIP_PERF_CI=1.
