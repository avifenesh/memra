# Health and readiness fault acceptance

Readiness waits for a private native prime and two decode calls, followed by a stream fence. Admission calibration controls cannot skip this warmup.

Serial OOM recovery retries only before output and within the retry budget. Every serial OOM retirement, including exhausted retries and failures after output, fences device state before it is released. Diagnostic targets cover serial and scheduler-tick prime failures, and serial failures after output. Runtime defaults and native math are unchanged.

Validation is in progress. The receipt summary will bind the source, binary, input artifacts, declared request shapes and the observed fault results. Hardware and private wiring receipts remain in the private evidence bank. No support state or fleet pin is promoted by this work.
