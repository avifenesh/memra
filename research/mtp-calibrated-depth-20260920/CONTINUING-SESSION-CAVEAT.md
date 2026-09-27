# Cold-prefill scope and continuing-session correction

Owner correction on 2026-09-21: this experiment preserves learned D and the actual
conversation, but rebuilding KV each turn does not represent the intended continuing
session. Its results describe cold-prefill requests with persistent learning. They must
not be presented as the benefit or loss for a conversation with native KV reuse.

The pinned source predates the merged Gemma continuation-prime work:

- `746bba2b4` (#561): continuation-capable text prime with one attention representation
  across chunks.
- `797ae5734` (#564): restores fast attention performance for that representation.

Current main was checked at `34d6bce356b15fd9b3f9a81a611b7e3483c5edad`.
The separate Qwen correction `1bc716343` fixes short carried suffixes in the server
prime walker. Its local receipt qualifies Qwen3.5-9B; it is not evidence that our
Qwen3.8-27B direct-session runner is qualified for reuse.

The ongoing frozen cold study is retained unchanged. A separate continuing-session
candidate is being prepared on current main. It must prove cache reuse on every later
turn, preserve the entire learner state, count cached and newly processed input tokens,
and pass target-specific numerical gates before any performance score is accepted.
Cold and warm studies must be reported separately, with their exact runtime identities.
