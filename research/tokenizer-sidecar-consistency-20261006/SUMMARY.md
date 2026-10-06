# Tokenizer sidecar mismatch warning

The code change is Memra commit 320d1e3a897cfb054fbe9b878aca7d61748b26a2. The tokenizer library source SHA256 is e87aefe0d458a7638d6baab9a5ef833bafddc0db464fc34e4c301728a38068ca.

The CPU parity run used the public Qwen/Qwen3.8-Flash-Next sidecars pinned to revision de4b8e4d43b917e7706784d8bb445c9af86a3540. Vendor tokenizer.json SHA256 is 0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3. Vendor tokenizer_config.json SHA256 is b11349aafa7cdc6a320767cf7ceb29ed82f7eda5d65e8e0819e76f0ce947bf27. The vendor reference was generated with tokenizers 0.23.1.

The raw TSV files contain nine synthetic Unicode inputs as UTF-8 hex and the reference IDs with and without special tokens. The four candidate stdout files report nine of nine matching cases in both modes. Matching vendor and flags-only metadata emitted no warning. Regex-only and combined drift emitted one warning naming the selected config descriptor. The stderr files and machine-emitted warning result are exact copies.

The controlled regex variants removed the mark class \p{M} from the vendor Split. The flags variants changed pre-tokenizer ByteLevel trim_offsets and decoder ByteLevel add_prefix_space, trim_offsets and use_regex to true. Config regex authority stayed pinned.

The stdout copies replace only the local model-directory token with a fixture label. MANIFEST.json records original and publication-copy hashes. Original outputs were not changed or rerun. The four candidate checks reused the baseline vendor reference; the baseline was not replayed.

This is CPU diagnostic and ID-preservation evidence, not a model, native runtime, serving or performance qualification. No model weights, GPU or owner corpus was used. The separate whole-corpus four-ID pointer from issue #71 is outside this change.

The synthetic input copy displays spaces between UTF8 hex bytes. Removing spaces from each input field recovers its exact original raw TSV. This avoids a bare provider-ID-shaped hex token in the public boundary scanner. Original and publication hashes remain in the manifest. The data and checks were not rerun.
