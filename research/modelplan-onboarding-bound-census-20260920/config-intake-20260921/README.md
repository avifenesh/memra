# Canonical configuration intake

Intakes reviewed #537 `bb637184682216ea38cbb9eb79dab4841368eca1` onto reviewed composite
metadata source `36154128fc9f6d69717a31928d7b3266f4bb2a98`. The shared strict_json and
config_json files match the reviewed dependency byte-for-byte; raw_value and arbitrary_precision
travel with them. No alternate decoder was introduced.

The overlap resolution preserves #541 head policy using the typed boolean accessor, retained raw
config bytes, iterative source opening, physical inventory, strict manifest consumers and all
existing CI gates. Source constructors propagate parse/read errors while retaining the same
opened config used for identity and activation precision. No unrelated dependency ancestors or
main integration are included. Initial failed conflict-resolution attempts are preserved and
were repaired before the successful combined tests; they are not qualification passes.

Combined compiler/CLI tests passed (399 GGUF plus two declared ignores before two added intake
controls). The added controls pass for escaped tied-head declarations, wrong-type rejection and
strict manifest integer lexemes/private-marker objects. Canonical config and decoder tests pass
with preserve_order. Exact source hashes and logs are in evidence.json.

Both input source reviews are preserved. They do not approve this combined source, native
execution or root activation. Composite materialization work follows in a separate commit.
