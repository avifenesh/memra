# CUTLASS, vendored

NVIDIA CUTLASS v4.8.1, tag commit `2c85f497547eb441f0fe80bfb0fb2e6f9add1719`
(https://github.com/NVIDIA/cutlass), BSD-3-Clause (`LICENSE.txt`).

Copied subtrees, byte for byte and nothing else:

- `include/` (CUTLASS and CuTe headers)
- `tools/util/include/` (`cutlass/util/*`: packed strides, host helpers)
- `LICENSE.txt`

`build.rs` compiles every CUTLASS unit against this tree and no other. To move to another release,
replace the three paths from that release's tag, update the tag and commit above and the row in
`../../THIRD-PARTY.md`, and rerun the gates of every unit that includes it.

```
git clone --depth 1 --branch vX.Y.Z https://github.com/NVIDIA/cutlass
rsync -a --delete cutlass/include/ include/
rsync -a --delete cutlass/tools/util/include/ tools/util/include/
cp cutlass/LICENSE.txt .
```
