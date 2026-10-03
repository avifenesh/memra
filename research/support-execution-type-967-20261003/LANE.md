# Support execution input refusal

A FIFO at a support census input reached the reader before file-type validation.
Execution admission now checks all three pinned census inputs before reading content.
Each read anchors directory components with nofollow descriptors, checks the leaf type,
uses a nonblocking leaf open and verifies the opened object with fstat.

The reader hashes still bind raw bytes. Metadata validation, evidence path checks,
typed unknown-reader expansion and selected-contract missing-evidence refusals remain.
The checkout root is the trusted base. This is CPU tooling evidence only.

## Validation

The 11 controls use real FIFOs, sockets, directories and symlinks. They also assert
FIFO and symlink replacement between stat and open, directory replacement after the
parent descriptor is opened, descriptor cleanup, raw CRLF bytes, typed fallback safety,
contracts CLI refusal before any CPU command, and absent/partial census fixtures.

The baseline and eight individual guard-removal controls must fail their named
assertions. No special-file content is read by the early-refusal observer controls.
A separate source-bound coverage admission rejects a missing assertion or skipped
mandatory test. Framework, support census, cache, fast-gate and registry results
passed at the 4b source: 151 framework, 21 census, 55 dense-cache, 16 cache
qualification, 48 local-cache, 39 fast-gate and 14 registered workspace packages.
The public helper/carrier run at `cf0871a8c2` repeats those counts, admits 20
source/helper-bound edges and rejects missing-assertion and skipped-mandatory results.
Its raw output and complete contracts are in `receipts/` and `PROOF.json`.
Composition onto #964 remains separate.

## Limits

No native source, model artifacts, compiler defaults, serving defaults, qualification
tolerances or native gate coverage changed. No GPU execution or support promotion.
Publicity: skipped: maintenance release.


Reproduce the CPU controls from a clean checkout:

```sh
out=$(mktemp -d)
python3 research/support-execution-type-967-20261003/tools/controls.py "$PWD" "$out"
# Preserve the output before removing the owned scratch directory.
```

The baseline fixture is the byte-exact original helper. The control producer refuses
optimized Python before any work. Production FIFO refusal is also exercised under
both normal Python and `python -O` by the contracts CLI test.
