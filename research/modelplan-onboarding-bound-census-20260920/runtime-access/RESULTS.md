# Fallible runtime access preparation

This increment follows independently reviewed foundation repair
`60c07ea2a18eb01892b90fab5b0f94b40ce5b5ba`. It prepares the adapter and error propagation;
**the dense and hybrid model entrypoints have not yet activated complete census binding**.
No native or whole-issue qualification is claimed.

`BoundRuntimeSource` maps the existing ggml executor ABI to semantic IDs from a compiler-produced
GGUF naming schema. Physical targets and transforms still come exclusively from the immutable
bound checkpoint. Known optional probes can report absence; unknown spellings and invalid
members refuse. Grouped HF expert members do not masquerade as physical stacked banks. MLA
key/value views explicitly derive from the bound fused tensor. Backing and source policy
accessors retain the original opened source.

GPU tensor/expert, hyper-connection and parallel admission loading now use fallible source
accessors. Repack worker errors are joined and propagated. Ordered optional reads stop on an
error instead of selecting another tensor. Existing raw sources retain their Option-based APIs;
using those infallible APIs on the bound adapter refuses, so an unconverted caller is visible.
Raw GGUF helper target resolution is available but full spill/MTP conversion is still pending.

Validation:

- Bound-source CPU tests: 26 passed, including role resolution, explicit tied heads, native error
  propagation, expert members, nonuniform MLA splitting and error-preserving ordered reads.
- GGUF/CLI full suite and Clippy: see raw logs; artifact skips remain unqualified.
- Engine library/bins/tests and server Linux-target check and Clippy passed with `DOCS_RS=1`
  in a separate directory. These checks use CUDA documentation stubs, not native CUDA builds.
- Formatting and diff whitespace checks passed.

Next work is exact family/derived-view coverage before activation: Step HF stacked expert banks
and private MTP head schema, standalone trimmed drafts, manifest/pruned overlays, remaining raw
GGUF consumers, and #542 identity composition. Step MTP dense topology is already in the inherited
#537 ModelPlan; the current Step HF expert-bank schema explicitly refuses and must be implemented,
not bypassed. No runtime or support state is promoted by this preparation.
