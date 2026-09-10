# DSV4F remaining activation materializations

Base: fefb019bec216aafaa9c7ad0ccfe41661c450981. Issue #4 sublane.

Candidate implementation and gates are prepared; target validation is pending. No GPU launch is authorized until
independent narrow review and an orchestrator slot relay. No performance verdict.

| Producer and consumer | Scope decision |
| --- | --- |
| Attention-entry RMSNorm to Q_a BF16 pack, then repeated KV pack | Fuse norm and pack while retaining f32 for compressor fan-out; Q_b, head norm and rotary only touch qr_b/q, so the pack remains live through KV. |
| FFN-entry RMSNorm to shared-expert BF16 pack | Fuse and retain f32 for router and FP8 quantization. |
| Shared SwiGLU to BF16 pack | Fuse elementwise expressions with explicit f32 then BF16 RNE; sole consumer is unchanged down projection. |
| Expert intermediate FP8 quantization to half gather | Fuse the original 64-thread per-128 quantizer trees with the original 256-thread half gather; intermediate codes/scales stay in shared memory. |
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

Expected retained graph delta per rank: attention 43 x 2, FFN 43 x 1,
shared SwiGLU 43 x 1, intermediate quant/gather 43 x 1 = 215 launches removed.
New symbols: 86 norm/pack + 43 SwiGLU/pack + 43 quant/half = 172 nodes in
ordinary, C4 and C4+C128 forwards; zero in commit, zero everywhere OFF.
Base total kernel nodes are 2870/3269/3369; ON is 2655/3054/3154.
The shared SwiGLU f32 and intermediate quantizer codes/scales are no longer
materialized globally. Norm f32 output remains required by fan-out consumers.
Attention component timing includes both eliminated pack launches back-to-back;
it is an isolated chain cost, not the model's intervening Q/KV schedule.
