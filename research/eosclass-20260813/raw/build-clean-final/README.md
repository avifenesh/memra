# Clean final build receipt

Authoritative result: `build-verify.log` / `build-verify.exit` (`0`) and
`binary-sha256-final.log`.

The first full invocation (`build.log`) reached Cargo's successful `Finished` line, but
its tool wrapper was reaped before its post-build exit/hash epilogue ran. The immediate
cached confirmation (`build-confirm.log`) also reached `Finished`; its one-byte
`build-confirm.exit` is blank because that command used Bash's `PIPESTATUS` spelling
under zsh. Neither wrapper receipt is used as the verdict.

The authoritative retry ran the same release targets under explicit Bash, rebuilt the
CUDA/Rust artifacts, exited zero, and then hashed the exact binaries that the final GPU
battery will execute. No source changed during that retry.
