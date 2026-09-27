# Lifecycle log parser CPU evidence

`tools/serving_trace.py` validates complete mixed server logs and reconstructs queued, active-prime, and decode cancellation facts. The outer caller supplies the expected process, model, routes, worker generation, and client correlation. All recognized records remain in the census, including unrelated traces. Unknown prime work does not become completed work; retirement requires the observed receiver close and explicit worker retirement site.

Independent review found **P2 AD526-TRACE-1** at `de777117`: displaced markers and escaped bare schema values could hide records, including a duplicate after `trace_end`. The unchanged six-case finder control had four missing refusals. Commit `7b130073` rejects reserved markers outside exact framing and recognizes escaped bare schema keys/values. The original finder closed the finding with no new findings. Ordinary text and unrelated JSON remain accepted as ordinary log lines.

The 53 parser/wire tests, six unchanged finder controls, and five retained limit/producer controls pass under normal and optimized Python. `evidence.json` binds the actual sources and byte-exact raw outputs. The retained producer capture exercises real Trace API calls with CPU fixtures; it does not run inference. Additional reviewer compatibility controls and scope are recorded in the closure report.

Run the portable suite and unchanged finder controls from the repository root:

```sh
PYTHONPATH=tools python3 -m unittest test_serving_trace test_serving_release
PYTHONPATH=tools python3 -O -m unittest test_serving_trace test_serving_release
PYTHONPATH=tools python3 research/serving-release-20260920/lifecycle-parser-20260921/framing_repro.py
PYTHONPATH=tools python3 -O research/serving-release-20260920/lifecycle-parser-20260921/framing_repro.py
```

This is source and CPU evidence for a facts parser. Full peer/recovery accounting, wire cancellation, authenticated process/source/binary/controller bindings, GPU leases, and native C4 qualification remain pending. The parser returns no native or release qualification.
