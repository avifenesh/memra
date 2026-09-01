# Default-policy host unit tests

Source runtime fix: `7cd4561a6bb6221785f62168432fcbda2ce80e28`.

- `DOCS_RS=1 CARGO_BUILD_JOBS=1 cargo test -p memra-engine --lib b1_eager_program_requires_explicit_opt_in`
  — 1 passed, 0 failed.
- `DOCS_RS=1 CARGO_BUILD_JOBS=1 cargo test -p memra-server graph_session_requires_explicit_opt_in`
  — 1 passed, 0 failed.

These are pure environment-policy tests. They do not substitute for the native release build or
GPU exactness gates.
