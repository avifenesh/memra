# Local hybrid metrics qualification

The requested metrics cells pass. The complete cell exits **1** because plain
and MTP greedy text differ on the short control prompt, identically on the
original main binary and the candidate. The failed assertions are retained in
`native-results.json`; neither the old nor new binary is granted a cross-program
identity claim by this run.

## Exact inputs

- Candidate binary source: `925b102a6e92cd1896df23a7533bb1e56f0ca05c`.
- Collector checkout: `436156acf91554d06965fc416b56597fcbf8e79c`. Only collector/receipt changes
  separate it from the binary source; the Rust/Cargo tree is unchanged.
- Candidate binary SHA-256: `9341bd41b6f31137859bee49a6ff23a8396ae937dac7a3e7a3fdc729e76091db`.
- Original baseline source: `2873dd4ca37faa15cd4261e7b398926cced30106`.
- Baseline binary SHA-256: `eaa0afa71439143ae6c05455d85dd42097053b91dc5fb4e520e639b91d143529`.
- Qwen3.5-9B NVFP4 MTP local artifact SHA-256: `52c9cceb190055e0591a9a30c21f7200572eaf3ff1c59f6e9a1eda838a8f39de`.
- Promtool 3.15.0, executable SHA-256: `c736d55d3ccd959fe48329965fb5cb671d45585a9483954766448035100ff75c`.
- One RTX 5090 Laptop GPU. Six separate fresh-process boots under one exclusive
  lease: cache workload; old/new plain; old/new MTP; forced queue. Context ceiling
  8192, prefix cache 512 MiB, reuse/affinity off, GDN grid 32. The cache and
  identity/lifecycle boots allow eight sessions; only the explicit queue test
  sets one serving slot. Sampling and request bodies are in the raw JSON.

## Passed properties

The fresh cache workload passes all **50** named assertions: one cold leader
and four shared-prefix hits, cross-namespace isolation, prompt/cache usage,
JSON aggregates and LCP/tenant splits, Prometheus parity, real promtool,
histogram counts/buckets, bounded labels and JSON compatibility. Prompt tokens
are 1632, cached tokens 1024 and computed tokens 608. The six completions emit
42 token IDs in total, so the emitted-gap count is 36, not a hardcoded function
of the requested output budget.

The current plain lifecycle arm passes **35** named checks, MTP **36**, and the
forced-queue arm **12**. These cover auth 401/403, terminal success, queue/TTFT/E2E
counts, N-1 emitted gaps, rejected-model label bounds, transport cancellation,
and successful recovery without counting cancellation as successful E2E.
The forced queue has an actual positive `admission_session_defers` counter and
positive queue-wait duration. All **11** successful text scrapes pass real
`promtool check metrics` without warnings.

**32 distinct selected Rust CPU tests** and **7 Python parser tests** pass.
Final server Clippy and the focused fixture check also pass. A review follow-up
adds three native/OpenAI usage-normalization CPU tests and fresh live gates for
both completion JSON shapes; each passes all 50 assertions and real promtool.
See `formats/README.md`.
The premature post-200 stream fixture exercises the real SSE response builder:
it requires a typed error, no successful finish reason, and no terminal usage.
That is CPU fixture evidence, not native worker-fault qualification.

## Retained negative control

The same 12-token raw prompt, temperature zero and 16-token output budget is
sent to four fresh processes. Original and candidate plain text both hash to
`605ada875848be082a9f6c0da4e29e2c70887e290b1cec6d51eec8f865a78a48`.
Original and candidate MTP text both hash to
`5e770ec2c0253be0e76c02eda92f95234c2a981f3946e816008eb5689682d1e0`.
Both MTP arms report five rounds, 15 drafts and 12 acceptances. Thus the
telemetry change preserves each mode on this control, while the pre-existing
plain/MTP difference remains a failed gate. Its cause is not diagnosed here. The reproduction is owned and tracked in
[memra#918](https://github.com/avifenesh/memra/issues/918).

`attempt1/` preserves the earlier failed run. It caught the reserved `quantile`
gauge label and a bad pre-/post-traffic JSON key-set comparison. The current
renderer uses separate p50/p99 gauge families; the current JSON check compares
negotiated views in the same quiescent state. The first identity comparison had
unequal warmup; the fresh-process baseline controls above remove that ambiguity.

## Scope and receipts

`cache/`, `plain/`, `mtp/`, `queued/`, `baseline-plain/` and `baseline-mtp/`
retain responses, text scrapes, validator output, server logs, and 250 ms GPU
observations. Cleanup receipts show no remaining inference process in any arm.
These are single-run observations, not a speed comparison or default decision.

`ACCEPTANCE.md` records the full #522 acceptance, including the original asks
that remain outside this local completion: dedicated DSv4 qualification and
actual emission gaps, broader HTTP status/incomplete-stream aggregates, token
volume histograms, and remaining gauge inventory. The issue stays open. No
support state, serving default, kernel, or numerical program changes.
