# Request lifecycle producer and HTTP lifetime CPU evidence

The existing TTFT opt-in now provides request-correlated lifecycle records. The HTTP lifetime wrapper records cancellation while admission is still awaiting response headers, distinguishes an unpolled or dropped SSE body from normal EOF, and leaves non-SSE response transport unchanged. Worker hooks distinguish queue overflow from receiver drop and record aborted request retirement after admission/pin/request state release.

The frozen `ttft.rs` producer passed 18 standalone controls. This linked component harness passed 24 tests: those 18 plus six tests of the actual extracted HTTP middleware/body code through Axum with a fake backend. It uses the dependency versions and features from this repository and denies warnings. The exact harness, lock file, raw log and source hashes are retained here. The source at test time comprised committed `1a04ffc7` plus the parent HTTP changes identified by their content hashes.

The full server/worker build and tests, independent integration review, real serving cancellation, and native qualification remain pending. These controls prove neither model phases nor GPU cleanup. Quantum callbacks are available for the separate scheduler lane; they have not been wired to its production advances. Non-instrumented routes and incomplete/invalid/suppressed histories cannot qualify.
