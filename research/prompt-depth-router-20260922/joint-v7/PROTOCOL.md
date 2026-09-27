# Corrected sampled single-head C/K/D continuation

K remains **MTP draft sampler top-k**; target top-k stays
at 20. D is offered draft length. C is an after-offer
decision about drafting the next position.

The v6 development collection, randomized K/D training and
fixed C/K/D grids are retained as controls. Its attempted joint
selection is **not a learned-C measurement**: `turns.tsv`
reported zero native confidence decisions. The sampled
single-head graph applied fixed C cutoffs but omitted the
learned after-offer hook. A fail-closed engagement guard
stopped v6 before fresh heldout evaluation.

V7 changes only that single-head hook, mirroring the
chain-graph and eager branches. Rebuild the exact v7 source
over the pinned v6 source archive. The frozen Qwen checkpoint,
target sampler, prompts, seeds, GPU, temperature, top-p,
context, and maximum output stay fixed.

Use the frozen K model weights, K selection and K=20/D=3
fixed-C development choice from v6 as training-parent
inputs. Those cells used no learned C hook. Copy their
records and exact model hashes into the v7 lineage receipt;
do not relabel them as v7 native performance.

Before selecting a joint model, run a focused sampled
eight-turn v7 qualifier with a live K/C/D policy. Require
native C decisions above zero, the requested parseable
functions, bounded function cases, no exact loops, and
positive cached/new input tokens on later turns. A policy
making zero C stops may still be measured, but it must
execute the C model.

Rerun the three old joint-selection topics under the new
binary: fixed K=20/D=3/C=0, the development-selected fixed
cutoff, K-only, and token/history/prior joint variants.
Select by pooled complete native request tok/s subject to
code and loop gates, then run the matched no-op twin.
Freeze all weights and action mappings.

Before fresh heldout, require the selected joint policy
to make native C decisions. The sampled no-op K/C/D
qualifier must be byte-identical to the fixed K=20/D=3/C=0
control and pay C model work. Run a live learned-C
qualifier and require C decisions plus code/function
coverage. Then evaluate the six untouched fresh
eight-turn code conversations in balanced arm order,
including fixed K=3/10/20, the selected fixed vector,
K-only and joint learned/no-op twins, and fixed-K
C/D policies where fitted.

Score pooled returned output tokens divided by complete
native request seconds. Report paired whole-conversation
uncertainty, output ratios, actual K/D/C actions, model
seconds, code quality, loops, and native KV reuse.
Acceptance is a learned input and mechanism diagnostic,
not the score. This is research evidence; full sampled
endpoint exactness and any serving default remain separately
gated by Memra #673.
