# Security policy

How to report a vulnerability in memra, and what counts as one.

memra memory-maps GGUF and safetensors files and parses their headers and tensor metadata
directly. A malicious or corrupt model file is untrusted input to memory-unsafe code (CUDA
kernels, raw byte repacking, mmap'd tensor reads), so a parser bug reachable from a model file is a
security issue, not only a correctness bug.

## Report privately

Do not open a public issue. Use
[GitHub private vulnerability reporting](https://github.com/avifenesh/memra/security/advisories/new),
or email the maintainer (contact on the GitHub profile). Include:

- the component and code path, for example the GGUF parser (`crates/memra-gguf`), the safetensors
  loader (`crates/memra-gguf/src/safetensors.rs`, `source.rs`) or the HTTP server
  (`crates/memra-server`);
- a minimal reproducing input: a crafted file, or the malformed field and its value;
- what you observed (crash, out-of-bounds read or write, panic, wrong output) and how you
  triggered it;
- the version (`memra-server --version`) or the commit.

## In scope

- Crafted model files: GGUF header fields, tensor metadata, the safetensors JSON header, NVFP4 and
  other quantized block layouts, and the tokenizer and chat-template metadata the server renders.
- `memra-server` request handling on every route, including image inputs and the `/v1/audio/*`
  routes.
- Authentication and isolation: `MEMRA_API_KEY`, the `MEMRA_API_KEYS` keyring and its tenant cache
  namespaces, rate limits, metering, metrics access (`MEMRA_METRICS_TOKEN`), and the refusal of
  an open non-loopback bind (`MEMRA_ALLOW_OPEN_BIND`). A fail-open configuration bug counts.
- Server-initiated fetches: `image_url` downloads when `MEMRA_FETCH_URLS=1`, and the SSRF
  classifier in `crates/memra-net-guard`.
- Rewrite bundles and their receipts, which `memra-server` loads at boot (`MEMRA_REWRITE_BUNDLE`).

## Out of scope

The operator is trusted: environment variables, local file paths and the host are under the
operator's control, and running with `MEMRA_ALLOW_OPEN_BIND=1` and no key is a choice the server
warns about at boot. This does not cover the model-file, request, authentication and fetch
surfaces above.

## Supported versions

The latest release and `main`. Fixes land on `main` and ship in the next tag; older tags are not
patched.

## Response

memra has no dedicated security team. Expect an acknowledgment, not an SLA. A fix lands as a
normal commit once triaged; there is no separate security-release channel.
