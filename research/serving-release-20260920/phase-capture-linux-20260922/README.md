# Phase cancellation capture and replay: Linux CPU evidence

The collector cancels its own target after observing that request's queued, active-prime or decode phase in the owned server log. The final trace must confirm the cancellation phase, receiver close and explicit retirement site. It preserves all requests, responses, probes and log bytes; a successful reconnect alone is insufficient.

The reader binds an immutable capture to an externally supplied program, process launch and log identity. It reconstructs append bytes and read watermarks, checks request/health/process/listener chronology and body limits, then recalculates phase and wire facts. It returns no native or release qualification.

Independent review closed the three collector findings at `38b8f0e`: the Event.set timing race, a watcher surviving return, and observed file truncation. Reader findings closed at `96ec779a`: trusted body limits, a phase observation preceding its HTTP attempt, and untyped missing-path failure. Their original failures and repaired behavior remain preserved in the review evidence.

On Linux/Python 3.12.3, **123 tests passed in normal mode and 123 in optimized mode, with no failures or skips**. Six additional loopback captures exercised queued, prime and decode cancellation in both modes using actual Linux process/listener evidence and a synthetic lifecycle producer. External expectations were captured from the real invocation and launch configuration before collection; the reader replayed the resulting immutable captures.

The archive retains 64 captures: 40 expected negative-control failures, eight legacy unqualified diagnostics and 16 captures of phase facts. All 64 owned fixture processes closed with completed cleanup. The source hashes were unchanged before and after. The off-host audit verified 2,644 archived files and 2,134 capture payload references. These counts describe CPU tests and synthetic fixtures, not inference requests or successful native cells.

`receipt.tar.gz` contains raw outputs, complete captured payloads, fixture process receipts, external expectations and a transfer manifest. `evidence.json` binds the archive, source manifests, harnesses and independent closure reports. Run the source tests from the repository root with `PYTHONPATH=tools python3 -m unittest test_serving_cancel_phase test_serving_cancel test_serving_cancel_evidence test_serving_trace test_serving_release`; repeat with `python3 -O`.

No model or GPU ran in this validation. Whole native C4 still needs source/build/model/controller/physical-lease binding, the required scenario coverage and real serving execution. The branch remains unqualified development.
