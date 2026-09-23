# Worker-owned bounded tensor readers

`BoundDiskView` can prepare a worker-owned positioned reader limited to the same authorized
range. The reader exposes no file descriptor, file, mmap, absolute offset, or widening operation.
Both view and reader retain opened backing rather than reopening the checkpoint pathname.

Buffered reads, Linux random-access advice, and Linux direct reads are represented explicitly.
Direct mode reopens the retained `/proc/self/fd` handle and verifies the same inode. Every direct
read checks offset, length and destination alignment. A load-local weak cache shares backing,
random-access advice and the direct descriptor across tensor windows. Subranges preserve the
backing and authority; adjacent joins still require matching authority and opened backing.

Validation on the host: three disk tests passed, including a worker read after unlink/replacement
and dropping all views, range/overflow rejection, adjacent joins, shared backing, and direct-buffer
alignment rejection. All 39 bound-source tests passed. Host and Linux-target GGUF all-targets
clippy passed with warnings denied, as did formatting and whitespace checks. Exact commands:

```
cargo test --locked -p memra-gguf bound_disk::tests -- --nocapture
cargo test --locked -p memra-gguf --lib bound_source::tests
cargo clippy --locked -p memra-gguf --all-targets -- -D warnings
cargo clippy --locked --target x86_64-unknown-linux-gnu --target-dir target/541-linux-typecheck -p memra-gguf --all-targets -- -D warnings
cargo fmt --all -- --check
git diff --check
```

Linux syscalls are typechecked, not executed by these macOS receipts. Actual O_DIRECT I/O and
engine worker/H2D behavior require native checks before activation. No engine root loader or
performance default changes in this commit; engine consumers and the existing raw-handle CPU
expert ABI remain integration work. No GPU activity or native checkpoint qualification is claimed.
