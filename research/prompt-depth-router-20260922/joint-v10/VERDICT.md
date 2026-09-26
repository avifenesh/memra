# Qwen C/K/D v10: interrupted transfer, archived C path retains sampled pick

**Verdict: no non-code transfer result was established.** The
frozen v10 battery was stopped before final scoring because the
mainline Memra #673 sampled-C discard bug was incorrectly attributed
to this study's binary. The archived v8 source tied to binary SHA-256
`84b04b6ccccc3b64992377cef0677926cb932c7a98f063aff84d1f1f65224d7f`
is source archive SHA-256
`7765982aacad20867b406029b945cdec9f60e5694e0e9489731e3ffc8ce24d96`.
Its `crates/memra-engine/src/spec.rs` appends each sampled pick before
fixed or learned C stops on its
sampled chain graph, single-head graph, and eager paths. It also
refuses positive sampled PMIN0. The chosen-token discard
counterexample in #673 does **not** apply to that archived binary.
This source inspection does not replace a complete sampled
distribution qualification on the pinned model and request shape.

The intended transfer test held the Qwen3.8-27B GGUF, full-vocabulary
embedded MTP head, native binary, target top-k=20, temperature 1.0,
top-p 0.95, and code-trained C/K/D weights fixed. Its non-code strata
were 16 eight-turn IFEval instruction conversations and 16 eight-turn
GSM8K math conversations, with disjoint eight-turn qualifiers. K was
MTP draft top-k only. All arms ran on one nonproduction Nebius RTX PRO
6000 Blackwell GPU. Later turns required native KV reuse.

The qualifier completed 20 arms across both topics. It verified
full-head engagement, target top-k, prompt identity, positive cached
and new tokens on later turns, live learned C stops and continuations,
joint K-controller execution, and sampled output-ID identity for both
model-running no-op twins. This proves the paths ran; it does not by
itself qualify sampled distribution or task quality.

The scheduler then completed **78 of 320 intended held-out arm
conversations**, all on IFEval prompts. One arm already in flight when
the scheduler stopped finished without a parent result receipt and
was excluded. There was no completed held-out math stratum and no
final IFEval or GSM8K quality/throughput score. Partial per-session
tok/s values must not be combined into a topic-level or general
speed claim.

The private diagnostic archive is SHA-256
`3c849c5cb3e2da1ce4b150545f10cd645e6c12f8e1890106e897ce647162da94`
(52,521,622 bytes). Independent replay of its completed receipts
passed for **20 qualifier and 78 held-out arms**: prompt hashes,
target/draft K, full-head logs, native KV reuse, output token counts,
and command identity. The incomplete arm was named and excluded.
Replay did not produce a final task-quality or paired throughput
verdict. The research VM and its managed disk were deleted and
confirmed absent in fresh provider inventories. No serving
configuration changed.

The full frozen non-code comparison needs a clean rerun on one
nonproduction GPU, followed by its original task-quality, paired
throughput, no-op and archive replay gates. Sampled distribution
qualification on the exact Qwen artifact and binary remains separate
before any equal-distribution or serving claim. The v11 prompt split
is still frozen and unused for final scoring; its prepared extractor
expects a complete v10 archive and cannot consume this partial run
without revision. Memra #673 remains an open mainline engine issue,
separate from the archived research binary.
