# Qwen3.5-9B scoped serving record, issue #543

Frozen before the first GPU cell. Source: 2873dd4ca37faa15cd4261e7b398926cced30106.
Artifact: Qwen3.5-9B-NVFP4-MTP-GGUF.gguf, 5,657,607,424 bytes. The build job
records its SHA256 before execution. The artifact stays read-only.

Scope: one RTX 5090 Laptop, 24 GiB, context envelope 8192, offered concurrency
1 and 4, prefix cache 2048 MiB, host cache disabled. Native runtime and vendor
sampling defaults stay unchanged. Deterministic raw-completion cells explicitly
request temperature 0 and seed 7, with at most 64 generated tokens. The disconnect
cell requests 1024 and closes after eight content-bearing SSE frames.

The five required cells are:

- Streaming: full SSE framing, stable request/model identity, finish, DONE and
  usage, with decoded-text and token-count identity against blocking completion.
- Cache: cold miss and repeated warm hit with the existing GDN capture-grid count
  and identical decoded text and token counts.
- Concurrency: four synchronized client requests overlap and reproduce four
  serial controls. This tests offered concurrency, not a throughput win.
- Cancellation: close a live decode stream, require the worker abort event within
  30 seconds, then recover the prior control through a valid warm hit.
- Long context: tokenize a fixed repeated archive, require 6000 to 8128 prompt
  tokens, and compare cold/warm 64-token budgets inside the 8192 context envelope.

Red arms: malformed stub completion, corrupt usage, missing SSE DONE, and a real
second boot with prefix cache disabled. The cache-disabled cold/warm pair must be
rejected by the same cache assertions. A separate strict serve verification must
refuse the missing checkpoint-parity receipt before starting an endpoint. No threshold changes after seeing output.
Any fixture or runtime failure is retained, named and explained before a retry.

One GPU job, declared timeout 1200 seconds and minimum free VRAM 20000 MiB.
The paired cache-enabled/disabled arms stay in that same job. No compilation in
this cell. Owned subprocesses are terminated and waited before returning the lease.
Raw requests, responses, process logs, hashes and job outcomes are retained.

The TSV remains NativeReference regardless of these cells. A five-cell serving
record does not prove checkpoint parity, tensor census, tokenizer/template
fidelity, strict runtime admission or binary-bound rewrite prerequisites. The
current runtime uses HybridModel::rewrite_allowed with no bundle installed; the strict verifier
requires checkpoint parity first. The raw serving cells do not bypass that verifier
or fabricate a bundle.
No support state, serving default, numerical implementation or tolerance changes.
