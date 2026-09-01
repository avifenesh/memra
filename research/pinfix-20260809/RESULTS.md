# cx pinfix — serving hardening results

Lane: `lane/cx-pinfix`

Base: `06f89163`

Verdict: **PASS** for both requested code changes and both required gates.

Implementation commits:

- `6721a4db` — `fix: release prefix pins on session retire`
- `7b793c26` — `fix: validate client cache salts`

## Fix 1 — release builds now release prefix pins

### Before

The only production release site put the side effect inside `debug_assert!`:

```rust
if let Some(pin) = s.prefix_pin.take() {
    debug_assert!(px.unpin(&pin), "retired session held a missing prefix pin");
}
```

Release-profile RED receipt, using the real prefix-cache pin count:

```text
$ cargo test -p memra-server --release \
    worker::tests::retiring_session_releases_prefix_pin_in_release \
    -- --exact --nocapture
assertion `left == right` failed: retirement must release the cache pin in release builds
  left: 1
 right: 0
test result: FAILED. 0 passed; 1 failed; 132 filtered out
```

### After

Retirement consumes the session's single lease with `Option::take()`, executes
`unpin` unconditionally, and emits a non-panicking warning if the invariant is
already broken:

```rust
fn retire_prefix_pin(px: &mut PrefixCache, prefix_pin: &mut Option<PrefixPin>) {
    if let Some(pin) = prefix_pin.take() {
        if !px.unpin(&pin) {
            eprintln!("[prefix-cache] warning: retired session held a missing prefix pin");
        }
    }
}
```

The production retire loop calls that helper once. The regression calls it twice
against one session lease and proves the second call is a no-op, while the entry's
pin count returns to zero.

Release-profile GREEN receipt:

```text
$ cargo test -p memra-server --release \
    worker::tests::retiring_session_releases_prefix_pin_in_release \
    -- --exact
test result: ok. 1 passed; 0 failed; 132 filtered out
```

Closing source guard: no `debug_assert!` containing `unpin` remains in
`worker.rs`.

## Fix 2 — cache salts are rejected at the HTTP boundary

### Before

`cache_namespace` cloned every client string verbatim. A no-op validation seam
made the requested tests reproduce the base behavior:

```text
$ cargo test -p memra-server --release cache_salt_validation_ -- --nocapture
cache_salt_validation_accepts_normal_value ... ok
cache_salt_validation_rejects_oversized_value ... FAILED
cache_salt_validation_rejects_reserved_open_namespace ... FAILED
test result: FAILED. 1 passed; 2 failed; 133 filtered out
```

### After

Both `/v1/completions` and `/v1/chat/completions` now resolve a validated namespace
immediately after authentication and before rate-limit acquisition, worker queueing,
or cache state. Invalid values return HTTP 400 with `param = "cache_salt"`.

The boundary contract is:

- omitted or empty: the existing default namespace;
- maximum length: 64 bytes (64 accepted, 65 rejected);
- alphabet: ASCII alphanumeric plus `-`, `_`, `.`, `+`, `/`, and `=` (standard and
  URL-safe base64-compatible shapes pass);
- without a configured keyring, a raw `t:` prefix is explicitly rejected before
  the general character check;
- unsupported characters, including whitespace and namespace separators, are rejected.

GREEN receipt:

```text
$ cargo test -p memra-server --release cache_salt_validation_
running 4 tests
test tests::cache_salt_validation_accepts_normal_value ... ok
test tests::cache_salt_validation_rejects_oversized_value ... ok
test tests::cache_salt_validation_rejects_reserved_open_namespace ... ok
test tests::cache_salt_validation_rejects_unsupported_characters ... ok
test result: ok. 4 passed; 0 failed; 133 filtered out
```

## Required gates

```text
$ cargo test -p memra-server
running 137 tests
test result: ok. 137 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out

$ cargo build --release
Finished `release` profile [optimized] target(s) in 19.11s
```

The handoff baseline was 132 server tests; this lane adds five regressions (one
release-pin test and four salt-validation tests). No model or runtime GPU gate was
needed or run.

`git diff --check 06f89163..HEAD` passed before this results file was written.
The closing steering check still found no
`/home/avifenesh/.lanectl/inbox/cx-pinfix.md`.

## Scope boundary

The cache-salt change bounds and validates each client namespace and closes the raw
`t:<tenant>\x1f...` metering-row spoof in no-keyring deployments. It does **not**
bound how many distinct valid salts a client can submit: `a1` through `a200` remain
valid distinct keys. Therefore this result must not be cited as the audit's separate
global byte-accounted reuse-pool LRU / distinct-namespace-cap fix; that remains a
separate serving-memory hardening item.

No origin push, tag, release, `rustup`, model run, profiler run, or mutation outside
this worktree was performed. A workspace-wide `cargo fmt --all -- --check` remains
red on pre-existing formatting drift across unrelated crates; it was check-only and
changed no files.
