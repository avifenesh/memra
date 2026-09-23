# Linux CPU validation of request lifecycle diagnostics

Tested source: `7a2d7a40e685ca59341ac8c974e8e93c87707acf`.

The full server package passed 785 tests with zero failures and eight explicitly ignored tests. Targeted HTTP middleware (6), trace (23), and event-queue (1) controls also passed; these overlap the full suite and are not additional unique tests. Compilation used the real CUDA 13.1 toolkit, Rust 1.97.1, release mode and four Cargo jobs, with GPU visibility empty. No models or GPU qualification cells ran.

All 59,183 tracked source blobs/modes match before and after execution. The actual linked CPU test executables, complete source manifests, compiler/environment records and raw logs were preserved and independently rehashed off-host. This directory contains the raw test logs and their source/binary/archive identities.

The original independent reviewer closed the cached-MTP false-prime finding. Legacy timing calls no longer create actual-prime evidence. The earlier failed full-crate compile and its zero executed tests remain preserved separately; these results do not replace that failure.

The ignored manual-proxy and CUDA tests remain unrun in this CPU campaign. Native serving qualification still requires actual caller coverage, queue-versus-active retirement evidence, controller and payload bindings, and the required model/serving runs. Missing quantum records are not proof of cache reuse or absence of priming.
