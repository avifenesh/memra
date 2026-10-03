# Exact dense CPU-control executable reuse

Question: reuse the six native CUDA harness executables when their compiler inputs
are unchanged, while executing all 18 unset/ON/OFF CPU controls after both paths.
Budget: lane #942, $0, one broker build at a time, 4 CPU, 32 GiB, no swap, nvcc
threads 2. No CUDA context, model, rental or serving work.

## Result

Actual cold build and warm restore passed all 18 controls each. All six restored
whole-file hashes match the cold payloads. The executed helper and wrapper hashes
remain unchanged. The original shell baseline also passed all 18. The 52 CPU
regressions cover dependency mutations, complete phase permutations, strict JSON
types, damaged payloads/manifests, symlink targets, ambient override presence and
uncached compiler failure/unsupported-executable behavior.

Single sequential runs, same bounded build cgroup: original baseline 298.48 s,
cold cached build 271.39 s, restore 33.30 s. These are build/control wall times,
not model or GPU performance measurements.

## Exact inputs and remaining gates

The unchanged nvcc recipe compiles the same six sources with the original flags
and effective owned-stub loader environment. The census derives all three actual
host/device preprocessing phases from pinned nvcc's diagnostic output, retaining
each complete command, expanded stream hash and complete source/header dependency
identity. It sorts complete phase records because nvcc threads 2 can report the
independent device phases in either order. It never drops a phase.

The identity includes the complete selected CUDA toolkit, GNU compiler/tools and
dynamic libraries, GCC and system-header forests, direct linker search leaves
and directory membership, followed tool/library symlink targets, stock nvcc
profile, compiler environment and exact binary byte/mode identities. Unrelated
Rust, docs and receipt edits do not enter the native source closure.

Unknown layouts/phases/dependencies, any configured ambient override (including
empty values), damaged/incomplete/type-changed manifests or payloads miss. Only
the proved exact owned libcuda stub prefix is admitted; extra loader inputs miss.
Uncached compilation keeps the original compiler argv and canonical toolkit link
paths, runs once and propagates its exact failure. Cache-only ELF admission never
vetoes an otherwise valid uncached executable. Every restore still runs all 18
actual controls. This reuse grants no model/runtime/serving qualification.

## Native integrity and failed cells

Fresh executable hashes differ solely because of one local STT_FILE generated
`tmpxft` filename in `.strtab` per executable. All other ELF section payloads and
attributes, including host instructions/data, complete device fatbins, dynamic
ABI and numeric symbol tables, are byte-identical across the six paired builds.
The pinned compiler documents random symbol/variable names. The observed
difference is its generated temporary source filename. No random-seed flag or
compiler default was changed. Restore preserves the complete cold bytes.

The retained attempts show two conservative uncached fallbacks (inherited
compiler environment, unreadable broad host census), the phase-order miss and
its redundant compile, and the rejected fresh whole-ELF equality assumption.
The before/after census isolated the phase permutation; canonical complete-record
sorting passed the repeated actual census and native payload roundtrip. Real
mutated JSON manifests reproduced boolean/integer and integer/float admission,
then strict canonical comparison and duplicate/non-finite checks rejected them.

Native compilation emits shared-variable name-linkage warnings in dsv4_gpu.cu.
They are retained verbatim and assigned to A's follow-up Memra #944. No diagnostic
suppression or CUDA source edit belongs to this cache lane.

PROOF.json seals the receipts. The complete producer manifest records the exact
source/header/toolchain/recipe/payload tuple. Native binaries remain in private
custody; their whole hashes, complete section identities and symbol comparisons
are retained here.

Compiler diagnostic phases are version-sensitive. NVIDIA CUDA 13.1 documents
--dryrun as listing commands without execution and warns that internal steps are
unstable. They are used only for this defensive input census; original compilation
still calls nvcc with its unchanged recipe. Unknown output rebuilds.
Source: https://docs.nvidia.com/cuda/archive/13.1.0/cuda-compiler-driver-nvcc/index.html
