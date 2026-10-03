# Local planning input reads

LocalTree text and byte reads, Cargo command manifests and the tool-owned include
registry could open a FIFO before checking its type. The shared reader now anchors
directory descriptors below the trusted root, rejects nonregular leaves before
opening, uses nofollow and nonblocking flags, and checks the opened file with fstat.

Four actual original-consumer FIFO observations prevent content I/O. Six original
alias reads are observed at the four leaf paths and two descendant ancestors. Four current
consumers exercise leaf aliases and FIFO/symlink stat-to-open replacements. The two
descendant readers also exercise ancestor aliases and replacement before directory
open. A replacement after directory open retains the original file bytes through
the descriptor. The root and the registry's module directory are trusted bases, not descendant
components. Root manifest failure retains full planning expansion. Direct unsafe
reads retain refusal. Git snapshots, CRLF raw bytes, text newline normalization,
default Cargo command arrays and unknown-reader full fallback have controls.

The exact original planner, support helper and registry are from
`9efd457949a7580593f133492b5592c79ddfb72a`. Seven guard-removal controls must fail
named assertions with zero fixture errors. The first prestat-only attempt did not
fail because fstat still rejected the FIFO. It is not an admitted negative control.
The strengthened test observes the attempted special-file open itself.

Run the public proof with an owned output directory:

```sh
python3 research/local-planning-io-972-20261003/tools/prove.py "$PWD" /path/to/owned-output
```

The producer refuses optimized Python before work. Actual production CLI behavior
is exercised both normally and with `python -O`. The mandatory census is explicit.
Coverage admission binds source/helper bytes and rejects a missing assertion edge
or skipped mandatory result. Raw output, source pins and the actual final composed
tuple are retained in `receipts/`.

This is CPU tooling evidence. No native math, emitted program, compiler/build
default, model artifact/default, qualification tolerance or required native gate
coverage changed. No native or serving qualification is claimed.

Publicity: skipped: maintenance release.

Actual A968 composition at `487494f8ec` passes 186 framework tests, 21 census
tests and all 14 registered packages. The 15 methods and original consumer/CLI/red
groups admit 44 edges. Two admission mutations and four optimized-producer startups
are refused. `PROOF.json` binds that actual source and base. B970 composition remains
separate; the A-only receipt is not final-base evidence.

Actual B970 composition at `4fbf0d4505` on merged `996a64af78` passes 194
framework tests, 21 census tests and all 14 registry packages. Its independently
bound 44-edge proof and all refusal controls pass. `COMPOSED.json` and
`receipts/composed-B-v5/` retain that new tuple. B970 helper/registry/transport-test
bytes and every other A/B planner function match the merged base. The original
A-only receipts remain unchanged. The first nonred prestat attempt is retained
under `receipts/non-admitted/`.

Hosted draft `0264a4515c` exposed an order-dependent test observer: a safe Cargo
manifest read before the FIFO was rejected by the blanket fdopen mock. The failed
186-test run is non-admitted history under `receipts/failed-hosted-A/`. The repaired
test allows regular reads and refuses unsafe content, and explicitly executes both
FIFO-first and FIFO-last manifest orders. Production enumeration is unchanged.

After the observer repair, actual `173c031f61` on `996a64af78` passes 194
framework tests, 21 census tests, all 14 registered packages, the 44-edge proof and
all admission/optimized-producer refusal controls. Both manifest orders are
asserted by the executed method. `COMPOSED.json` and `receipts/composed-B-v6/` bind
the repaired tuple. The earlier local-only composition remains separately in
`COMPOSED-before-order-fix.json`. Current hosted CI and review still gate merge.

The snapshot-bound producer is a historical research proof. Reproduce it at the
`tested_source` named by the receipt. It compares pinned historical command arrays
and applies source-text mutations, so it is not an always-on CI gate for future
intentional planner changes. CI keeps all 15 behavioral methods through the
194-test framework floor. Revuto identified the added permanent proof step as too
strict; that unmerged step was removed without changing existing gate commands.
