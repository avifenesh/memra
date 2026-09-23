# Actual prime observation: strict Linux CPU validation

Tested and independently reviewed source: `aa25e9356b49f4f1f7ee11def8247f4745118c89`.

Production/all-target Clippy with warnings denied passed. The full server package passed 799 tests, with zero failures and eight ignored tests. Focused HTTP (6), trace (28), actual observer (9), and queue (1) controls also passed and overlap the full suite. Real CUDA 13.1 compilation supplied the native link symbols; runtime GPU visibility was empty. No model/GPU qualification was performed.

The reviewed actual-operation wiring preserves the existing scheduling policy and numerical chunks. Disabled tracing avoids the metadata getter. Actual nonzero operations own prime observations; mandatory same-turn finalization remains inside the last quantum. Explicit queue/active retirement sites and the 8192-record bound are retained. Unsupported normal/failure retirement outcomes are absent from the production API.

All source blobs/modes match before and after execution. Raw logs, compiler/environment records and actual linked CPU test executables were archived and verified off-host. Earlier failed compile/lint runs remain failures in their original records.

These results do not qualify the complete native cancellation/serving matrix. The C4 consumer, full controller/process/artifact bindings, ignored manual/CUDA cases and real model validation remain outstanding.
