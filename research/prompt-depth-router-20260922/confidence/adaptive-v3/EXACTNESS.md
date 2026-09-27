# Sampled confidence stopping in the v3 research source

The archived fixed-C experiment could discard a sampled low-confidence
proposal *before* target verification. With `p=q=(0.8, 0.2)` and a
cutoff retaining only the first token, that rule emits `(0.96, 0.04)`
rather than the target `(0.8, 0.2)`. The proof and affected paths are
in `../EXACTNESS.md`.

The v3 research patch offers that sampled proposal to the target
regardless of its confidence. A low-confidence sampled pick stops
the chain **after** the pick, before another draft slot. The same
filtered proposal probability `q(x)` reaches the unchanged
accept/reject test. At one position, offering `x` and accepting it
with probability `min(1, p(x)/q(x))`, then sampling rejection mass
from normalized `(p-q)_+`, has target marginal `p`. When `x` is
accepted, the choice to draft a later position is conditioned only
on the committed prefix; verification still determines that
position's output. This reasoning applies at each position and
does not require a hard-coded confidence threshold.

The corrected sampled chain graph, single-head graph and eager
paths all append the sampled token to the proposal list before
testing the stopping cutoff. Positive `MEMRA_SPEC_PMIN0` in sampled
mode is refused: this patch does not justify a zero-proposal
decision made after seeing a token. Greedy C=0 and positive-C
identity was checked with the native oracle. An eight-turn sampled
C=0 twin produced identical token tapes, and the GPU fixture
reproduced the old bias and the corrected two-token law with both
`p=q` and `p!=q`.

After the timed matrix, a second eight-turn twin at the
calibration-selected positive C matched all output-token tapes. A
native CUDA fixture drew 50,000 samples per controlled case using
Memra's filtered proposal and residual-sampling kernels. At
`p=q=(0.8,0.2)`, the corrected first-token frequency was 0.7994
against the target 0.8; the old discard simulation gave 0.9592.
With `p=(0.5,0.5)` and `q=(0.9,0.1)`, the corrected frequency was
0.5009 against target 0.5; the old discard simulation gave 0.5518.
The fixture source, build log, output counts and hashes are members
of the sealed `receipts-v3/native-data.tar.gz` archive.

These checks establish a distribution-preserving **stopping rule**
under the tested proposal/target filtering assumptions and verify
the research source's path order. A controlled native CUDA sampler
fixture is separately retained with the run. They do not establish
full-model statistical parity of every Qwen output path or qualify
a served positive-C override. Memra #673 remains the active sampled
MTP and serving-configuration gate.
