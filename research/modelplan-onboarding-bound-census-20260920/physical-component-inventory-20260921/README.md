# Physical component inventory for composite sources

`physical_tensor_inventory` retains each opened component's own tensor census and dialect.
The root component is [], its fallback is [0], and deeper fallbacks append another 0. These are
provenance identifiers, not filesystem paths. A tensor name can appear in several components
without one physical record replacing another. Quantization auxiliaries remain attached to the
original physical weight record even when the effective overlay selects a different weight.
The inventory is owned metadata with no file/mmap access; it stays available after paths vanish.

The existing effective `tensor_census` view retains its previous override behavior. A shared
own-component census builder prevents its metadata and the physical inventory from drifting.
Fallback overlay binding and identity still refuse: this inventory does not yet validate a full
composite semantic selection or authorize materialization. Compiler-owned source/role selection,
per-member transforms and complete composite identity remain required before activation.

Source chains now open iteratively, constructing model objects from leaf to root. Cycles compare
manifest identity together with directory context; hardlinked manifests in different directories
can have different relative-source meaning and are not falsely conflated. Source chains are bounded
to64 manifest components, including the leaf. The maximum valid chain is tested, and a deeper chain
fails contextually. The initial recursive guard attempt overflowed the test stack before the limit;
its actual SIGABRT/stack-overflow logs are retained as development failures. The final walk has no
recursive model-sized open frame. Eleven fixture directories belonging to the two confirmed exited
aborted test processes were inventoried and cleaned; cleanup records are retained.

Manifest files must be regular files. Unix opening uses O_NONBLOCK so a FIFO refuses before waiting
for a writer. Directory identity uses metadata without requiring directory-listing permission.
Controls include two-level shadowing over a complete repack, an FP8 safetensors weight/scale pair
shadowed by a GGUF-layout overlay, self/two-node/symlink cycles, distinct-directory hardlinks,
64-component success/depth overflow, search-only directories and FIFO rejection.

Validation: six new controls pass. GGUF384PASS/2ignored; CLI13, Step1, inspector7, external3,
doctest2; actual identity/snapshot host18; all-target GGUF/CLI Clippy; Linux engine/server lib/bin/test
Clippy with DOCS_RS=1; formatting and whitespace checks pass. Final source/binary hashes and lossless
logs are in evidence.json/raw; initial development failures remain separately labeled.

Carver approved exact `c3fba6b25acfeb640342c5aebf0e1d25fc3ce2cc` with no findings.
Six independent controls and 933 Git blobs were verified; the report and evidence are in `review/`. This is physical inventory groundwork, not composite binding,
model support, optimized/native/GPU qualification or root-loader activation. #537 canonical config
repair, existing #548 reference68PASS/1FAIL, remaining composite/draft/private-cache boundaries and
the coordinator's final #542 intake remain separate.
