# Native F32 decoder and mapped-host C4 graph gates

R9 corrects the earlier decoder contract mistake: the actual `dsv4_e4m3`
preserves negative zero and maps only NaNs to +0. The CUDA 13.3 native packed
conversion is compared against a CUDA 13.1 control kernel calling that exact
included helper, plus an independent CPU oracle. All 65,536 byte pairs pass
on both RTX PRO 6000 cards. The earlier R6 raw characterization remains
valid; its separately normalized arm was not the engine contract.

The F32 GEMV keeps the original per-thread sequence and 128-leaf reduction.
In the unroll-by-two branch all i0 products precede all i1 products. The
wide-K basis covers 128/256/2048/4096/8192, with explicit cancellation controls
distinguishing original, pair-interleaved and element-interleaved sums.
Both cards pass complete output-bit identity, guards and memcheck.

Three-ABBA cold-cache q_b-shape means (32768x1024), warmed arms and twice-L2
flush before every scored launch: GPU0 current43.797/native41.376 us;
GPU1 current43.445/native41.653 us. This is 4.3-5.9% on the component only.
No runtime integration, model-rate claim or default change is made.
Artifact and source hashes are in `dense-tc-gate-r9-native-compile-20260907.md`.
Raw controller namespaces `native-f32-{basis,memcheck,rate}-20260907-r9`.

The standalone base C4 graph gate also passes both-card memcheck and
synccheck. It covers slots 1/128/640, value/index guards, pointer/shape
refusals, and new mapped-host rows published by an eager async D2H producer
on the owner stream before replay. Selected host outputs are checked against
the known second generation and must differ from the first; equality between
two stale readers cannot pass that tooth. No recent-sidecar or model-rate
claim. Binary `454ee95f5408494db18221e207dfd1a93d50e2b208d310157284f086b17581ec`;
source `a70ad67e3b2a497f19557bd9476f8aacdca2a533c3795d73a874dafeceb4ed1b`.
Raw `c4-graph-{memcheck,synccheck}-20260907-r2`.
