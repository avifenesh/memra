# Prime-time sampled C/K/D qualification

K is MTP draft sampler top-k, with target top-k fixed at 20.
D is offered draft length. C is the after-offer decision to
draft another position.

V6 fitted small C/K/D models and measured fixed controls, but
the sampled single-head graph never called learned C. A native
engagement guard stopped that attempt before fresh heldout.
V7 added the missing call. It then made 612/612 C stops on its
eight-turn qualifier despite passing all eight code/function
cases. A pinned diagnostic build logged `q=0` at the live C
hook. Replaying the fitted model on 10,030 K=20 and 10,413
K=3 development first-offer examples predicted stop rates
below 0.1%. The mismatch was a missing probability output:
prime-time graph capture requested chosen-token probability
for fixed C or tracing, but omitted the attached learned-C
policy. V7 re-selection is diagnostic and cannot choose the
final joint policy.

V8 passes the learned-C requirement to prime-time draft graph
capture. A zero or invalid offered probability fails closed
before it can become a spurious C decision. The Rust source
patch is pinned against the exact v7 source archive. All
model weights, frozen workloads, target sampling settings and
the same nonproduction RTX 5090 remain unchanged.

Before selection, run an eight-turn sampled native
engagement qualifier. Require nonzero C decisions and a
mix of stops and continuations, 8/8 parseable requested
functions, 8/8 bounded function cases, zero exact loops,
and positive cached and new input tokens on later turns.
This checks live policy engagement, not full sampled
distribution parity.

Reuse only the v6 C=0/randomized training exposures, fitted
model weights, fixed C/K/D development choice and K-router
choice. Their source and binary are named as a training
parent, never as v8 timing. Rerun all three old joint
selection topics under the v8 binary: fixed K=20/D=3/C=0,
the chosen fixed C=(0, 0.529), K-only, and three joint
C/D variants. Select by pooled complete native request
tok/s subject to code/function/loop gates. Run the
selected variant's no-op twin and require native C
decisions in selection.

Before opening six untouched fresh eight-turn code
conversations, require the sampled no-op qualifier to
match the fixed baseline byte for byte and pay C model
work. Require the live joint qualifier to make a mix of
C stop/continue decisions and pass the code gates.
Then compare fixed draft K=3/10/20, the best fixed
C/K/D vector, learned K-only and joint C/K/D, their
no-op twins, and fixed-K C/D policies where fitted.

The primary score is pooled returned output tokens divided
by complete native request seconds. Report paired
whole-conversation uncertainty, output and elapsed ratios,
actual K/D/C actions, model seconds, code quality, loops,
and native KV reuse. Acceptance remains a training
input and diagnostic. Full-model sampled parity and a
vendor-default endpoint serving decision remain separate
under Memra #673.
