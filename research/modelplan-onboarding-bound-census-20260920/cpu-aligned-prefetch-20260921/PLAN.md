# Aligned Rust I/O and production prefetch submission — execution pending

The earlier `a055f272` direct-labelled controls used unaligned GGUF windows and therefore tested
buffered fallback. Their logs/labels remain intact, with this limitation recorded. This candidate
adds complete internal repack fixtures with 128/512-wide Q8_0 banks, 4096-aligned tensor offsets and
69,632-byte per-expert windows. The gate requires every selected window and reader mode to match,
prints its metadata, and checks six cold cache misses and their exact inserted byte count alongside
same-program token/row output parity. Direct descriptor creation also verifies O_DIRECT via F_GETFL.
A separate aligned active-mirror request must refuse, rather than silently dropping the mirror.

The predictor's existing descriptor handoff is extracted into one shared production submission
helper. The new CPU control invokes that helper using controlled selected expert IDs, not a copied
FFI call. A test-only companion observer blocks real scoped I/O before the actual Rust reader runs.
After handoff returns, all Rust argument/model/source owners are dropped and the pathname is
replaced. Read-only weak lifetime diagnostics must show all six contexts alive while blocked;
submitted/inflight counters must be six. Unblock must drain all contexts/counters and publish the
original bytes in the native annex. The observer macro and exported observation functions are
absent from ordinary production companion builds; numerical routines and ABI2 are unchanged.

This deliberately does not exercise prediction/routing computation. The host harness still has an
unused router constructor that panics if called, and an unused CUDA-pinned type. Actual host storage
accessors, descriptor creation, submission, retained readers and the native CPU routines are used.
No full model/serving/GPU qualification, performance statement or root activation follows.

Local pre-execution checks pass: affected GGUF/CLI suites and Clippy, actual C reader interop,
Linux-target host harness and DOCS_RS engine/server lib/bin/test Clippy, formatting and flag census.
The new aligned/mirror/prefetch execution gates remain pending a fresh frozen Linux CPU snapshot.
The existing authorized host/isolated source/target/receipt layout and four-thread/job limits apply;
no installations or global configuration changes are needed.
