# Ordinary reference for short prime inputs

The f53336f5 native run passes the fixed eager comparisons at 4/8/16/15/17 tokens
(zero error at 16/15/17) and all eight graph comparisons. It then panics in the
fixture's ordinary reference: `prime_cache` requires at least `PRIME_MIN_T=16`,
but the first retained-prime prompt has four tokens. No complete capture receipt
was issued, and the 24 later caller cases remain unrun on that build.

The fixture now follows the production short-prime branch: below `PRIME_MIN_T`,
call `decode_step_h` tokenwise and keep the last logits and pre-output-norm seed.
At the threshold, keep the existing `prime_cache` call. This is the same decision
used by production priming; the seed contracts match.

The 4/8/16 prompt bytes, graph bucket, independent teacher-forced quantized-cache
reference, three separate caches, seed comparison, logits and greedy continuation
checks are unchanged. The graph still executes the actual `prime_graph_run`,
including padding coverage for short inputs. Neither production prime code nor
the fixed numerical policy is changed. A later mathematical mismatch remains a
failure and cannot be explained away by this fixture correction.

Validation: focused Linux-target `rewrite_identity_gate` clippy/type checking and
format/diff checks pass. Fresh native capture/caller validation remains pending.
