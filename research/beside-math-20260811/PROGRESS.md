# Beside VRAM arithmetic — progress

Status: complete on 2026-08-11.

This lane is limited to a source-cited, docs-only co-residency VRAM fit pre-check. It will not run
GPU workloads, inspect model bytes, change runtime code, alter generated performance boards, or
make a business or serving-topology decision.

Delivered:

- `VRAM.md` inventories the protocol, co-residency scope, rig, tracked model, and runtime-memory
  sources;
- it separates recoverable proxy values from `UNKNOWN — needs-measurement` target inputs;
- it records the bounded arithmetic and GO/NO-GO-for-the-A/B fit verdict without making a business
  or serving-topology decision.

The initial progress marker landed in its own commit before this research deliverable.
