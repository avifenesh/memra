# Security Policy

memra loads GGUF and safetensors model files by memory-mapping them and parsing headers/tensor
metadata directly: a malicious or corrupted model file is untrusted input to a memory-unsafe
surface (CUDA kernels, raw byte repacking, mmap'd tensor reads). Treat any parser bug reachable
from a model file as a security issue, not just a correctness bug.

## Reporting a vulnerability

Do not open a public issue for a suspected security vulnerability. Instead:

- Use [GitHub's private vulnerability reporting](https://github.com/avifenesh/memra/security/advisories/new) for this repo, or
- Email the maintainer directly (see the GitHub profile for contact) with a clear repro (a minimal
  crafted GGUF/safetensors file or a description of the malformed field, and the crash/UB
  observed).

Include:
- The specific file/loader path involved (`memra-gguf`, the safetensors loader in `memra-engine`, etc.)
- A minimal reproducing input if possible
- What you observed (crash, OOB read/write, panic, incorrect output) and how you triggered it

## Scope

In scope: memory-safety and parsing issues triggerable by a crafted model file (GGUF header
fields, tensor metadata, safetensors JSON header, NVFP4/quant block layouts). Also in scope:
`memra-server`'s HTTP-facing request handling, including its bearer authentication, multi-key
tenant isolation, rate limiting, metering, and metrics access-control surfaces. Configuration
and fail-open bugs involving the auth seams `MEMRA_API_KEY`, `MEMRA_API_KEYS`,
`MEMRA_METRICS_TOKEN`, and `MEMRA_ALLOW_OPEN_BIND` are in scope as well.

The single-user, single-machine research-engine caveat is limited to engine-internal deployment
assumptions that are not exposed through a crafted model file or the server request/auth surface.
It does not put a `memra-server` authentication, tenant-isolation, rate-limit, metering, or
metrics-access bug out of scope.

## Response

This is a small research project without a dedicated security team. Expect an acknowledgment
within a reasonable window, not an SLA. Fixes land as normal commits once triaged; there is no
separate security-release channel at this project's current size.
