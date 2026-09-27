# Scoped catalog and physical inventory prototype

Status: CPU-tested library stage, not universal engine activation. The existing native Step vision
implementation and its historical qualification are preserved. The gap addressed here is canonical
catalog/binding coverage, not the existence of vision support.

All 26 pinned shard headers are accounted for: **1,597 physical tensors** become **1,471 folded
roles**, comprising **804 text roles and 667 vision inventory roles**, with no unclaimed names.
The 126 additional physical text tensors are FP8 scale grids omitted from the index. Index ownership
and duplicate header names are checked; no artifact is filtered, re-exported, or changed.

The two vision headers contain 667 BF16 tensors. Pinned vendor configuration and source files were
read only as offline schema evidence and verified against the existing lock; no external runtime
implementation is used. Their exact hashes/URLs and header receipts are in `raw/retrieval.json`.
The catalog uses the declared geometry and pinned defaults, including the 8,960-wide MLP and two
3x3 downsamplers. It does not infer a Gemma tower or report inventory validation as vision execution
qualification. The raw header inventory and full-catalog metadata probe are retained in this folder.

## Implemented boundary

- `BoundTensorSource::compile_for_scope` separates complete metadata binding from selected access.
  Text scope binds the vision inventory as well but refuses its semantic materialization. Full
  scope refuses components not yet represented by the canonical executable plan. This does not
  disable the existing separate Step vision route.
- Pack-owned catalogs enumerate exact names, shapes, unquantized storage and owners. Missing,
  unknown, malformed or wrong-quant unselected tensors fail; no namespace wildcard is used.
- Captured config is reused from the opened source, matching #542's raw-config capture. Changes
  to a pathname after opening do not change the catalog. Scope and catalog enter digest domain v3;
  this still must be composed with #542's opened-byte identity before trusted runtime use.
- All typed materializers authorize the selected semantic ID. Metadata remains observable for
  unselected components. Header/custody-hash I/O is distinct from semantic payload materialization.
- `BoundDiskView` has private backing handles. Its public API exposes only the authorized bytes,
  bounded positioned reads/subranges and same-authority adjacent joins. It cannot return a file or
  whole mmap. Reads beyond the range, overflow and cross-scope joins refuse.
- Vision float views retain their original encoding rather than inheriting text-weight re-encoding.

## Remaining engine work

The model entrypoints have **not** activated the bound adapter. Its raw GGUF accessor deliberately
refuses, and its old engine disk adapter returns a contextual error if a real disk extent is needed.
This prevents a silent fallback or raw-handle escape while the bounded engine consumer is unfinished.
Before activation: migrate those consumers, compose #542 identity/revocation, finish standalone
trimmed/student MTP and repack/overlay scope handling, preserve existing native vision through its
appropriate binding, obtain integration review, then run the affected native gates. No GPU work,
model-support promotion or issue closure is claimed by this prototype.

## Validation

- 39 bound/census/disk CPU tests pass, including selected-read authorization, malformed unselected
  inventory, captured config, scope identity, preserved vision encoding and bounded disk ranges.
- GGUF/CLI full suite and warnings-denied Clippy pass; the existing artifact-dependent skips are
  not artifact or native qualification.
- Linux-target engine library/bins/tests and server type check passes using `DOCS_RS=1` in its
  separate target directory. These are documentation CUDA stubs, not a native binary.
- Full pinned header/catalog probe: 1,597 physical headers, 1,471 roles, zero leftovers. Metadata
  evidence only; no checkpoint payload or vision semantic/execution validation.
- Formatting and diff whitespace checks pass. Raw logs, source and evidence hashes are retained.
