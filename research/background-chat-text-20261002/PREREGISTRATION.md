# Background chat/text delivery, bounded local acceptance

Scope: Memra #914 on source base 42e9ed7447a19aa27d6be88b3341296b9ba16661.
No native math, kernel, model artifact, sampling default, JobStore policy value or
support declaration changes. The background switch remains OFF. #550 remains open.

Cached artifact: qwen3.5-9b-judge-q8_0.gguf, expected SHA256
0825505bda37933f5856fd0751273b3bdf7224961d81dad9c4fcc1d47d49210c.
All heavy work uses the shared broker, private target, compiler jobs two, four CPU,
32 GiB host RAM and no swap. GPU cells use the inherited canonical FD9 lease on one
RTX 5090 Laptop GPU. Build precedes GPU; no compilation inside GPU cells.

CPU contracts exercise real route/admission code with controlled worker events:
OFF/stream/type refusals; omitted decoder defaults; success after a declared deadline;
full chat/native token and usage fields; tenant/key-rotation/forged-key refusals;
zero/partial cancellation and worker closure; terminal conflicts/repeated reads;
worker errors/EOF; queue-cap refusal and oversized terminal fallback. A synchronous
partial-deadline control must remain an explicit error. Existing Responses and
JobStore tests remain regression gates. CPU success is not native execution evidence.

Native cells, frozen source and binary across each cell:
1. Fresh OFF/ON boots, short synchronous/background identity, bare requests with no
   decoder fields on chat and text, cancellation after a token-progress callback,
   missing/foreign/rotated credentials, repeat polls and terminal conflicts.
2. Paired streaming/background long chat workload, constrained 600-string result,
   24,000-token output budget and 32,768 context. Completion callback must occur after
   the 90-second synchronous limit, with complete preserved output and usage.
3. Paired streaming/background long text workload on the same cached artifact. Use a
   long legitimate instruction/output budget; retain the streaming reference and
   compare the full original text and usage. Background completion must exceed 90s.

Each GPU cell is bounded to 1,200 seconds. Pairs stay in one cell; separate dialects
may yield between cells. All accepted requests must have exactly one terminal
callback with worker-truth prompt/cached/output counts. Cancellation must follow
observed native progress, return its actual retained prefix and stop the owned worker.
No synthetic delay is counted toward native deadline acceptance. Collect actual
source, binary, model, requests, environments, hardware, callbacks and failed attempts.

Collector negative controls reject missing/double callbacks, incorrect usage,
wrong envelopes, premature deadline success and cancellation without native progress.
This is functional local delivery evidence, not performance tuning or model promotion.
