# cx pinfix — serving hardening progress

Lane: `lane/cx-pinfix`

Base: `06f89163`

Scope: two pure `memra-server` fixes from
`research/code-audit-20260809/PAPER.md` sections 5.7 and 1.7/2.3. No model, GPU,
perf-board, release, tag, push, or other-worktree mutation is in scope.

Steering check: `/home/avifenesh/.lanectl/inbox/cx-pinfix.md` was absent at lane
start on 2026-08-09, so the handoff contains the complete steering available for
this work block.

## Base-revision defects (exact code)

### 1. Prefix-pin release is compiled out of release builds

`crates/memra-server/src/worker.rs:3366-3368` at the lane base:

```rust
if let Some(pin) = s.prefix_pin.take() {
    debug_assert!(px.unpin(&pin), "retired session held a missing prefix pin");
}
```

Rust release builds do not execute `debug_assert!` by default, and this workspace
has no release-profile `debug-assertions` override. The production side effect is
therefore absent: the session lease is taken from `s`, but `PrefixCache::unpin`
never runs.

Planned regression: route retirement through one small helper, exercise it against
a real pinned `PrefixCache` entry, and assert the entry's pin count returns to zero.
Run that test under the release test profile to prove it covers the configuration
that lost the call. A false `unpin` result will produce a warning receipt, never a
serving panic. `prefix_pin` remains one lease per session and is consumed once with
`Option::take()`.

### 2. Raw client cache salts cross the HTTP boundary without validation

`crates/memra-server/src/main.rs:1010-1012` at the lane base:

```rust
fn cache_namespace(cache_salt: &Option<String>) -> String {
    cache_salt.clone().unwrap_or_default()
}
```

`tenant_namespace` then returns that raw string when no keyring is configured:

```rust
let raw = cache_namespace(cache_salt);
if auth::global().is_some() {
    auth::scope_namespace(&tenant.tenant, &raw)
} else {
    raw
}
```

Planned boundary contract: omitted/empty remains the default namespace; a supplied
salt is at most 64 bytes and uses a small ASCII token alphabet compatible with the
documented base64/base64url salt shape; malformed values return HTTP 400 with
`param = "cache_salt"`. Raw `t:`-prefixed values are explicitly rejected in
no-keyring mode, before any worker/cache state is touched. Unit tests cover an
oversized value, the reserved open-mode prefix, and a normal salt.

Scope receipt: length/charset validation bounds each namespace string and closes
the raw tenant-form spoof. It does not by itself bound the count of distinct valid
salts. This lane will not claim the audit's separate global byte-accounted LRU /
distinct-namespace cap is implemented.

## Verification gates

- Targeted regression tests, including the prefix retirement test in release mode.
- `cargo test -p memra-server` (handoff baseline: 132 tests).
- `cargo build --release`.
- Final diff/status audit, followed by `RESULTS.md` with before/after command receipts.
