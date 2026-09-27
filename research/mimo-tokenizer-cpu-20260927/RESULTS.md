# MiMo V2.6 Flash RL text tokenizer CPU parity

Verdict: `Tokenizer::from_hf_dir` accepted the pinned BPE/NFC/Split+ByteLevel
program. Three self-authored raw text prompts matched the pinned Hugging Face
`tokenizers==0.22.1` reference in both special-token modes. No tokenizer code
change was needed.

## Source and scope

- Memra candidate head: `6be6674162a66a8a76dc74b57b30e40e27ce398c`.
- Hub model: `XiaomiMiMo/MiMo-V2.6-Flash-RL`.
- Hub revision: `3b38d063180c3e4aed9691fdc735f3d10b266ee4`.
- `tokenizer.json` SHA-256:
  `ff15eb925890d6b71b5160de4b846fbd13178438ab463b38ecc953e8cd1dcb3e`.
- `tokenizer_config.json` SHA-256:
  `413a7845f52943ccf4de0e5c838414507d16c44dbf573da9e20bc8902b384d06`.
- Hub download metadata named the same revision for each file. Its ETag for
  `tokenizer.json` equaled that file's SHA-256. Its ETag for
  `tokenizer_config.json` was Git blob ID
  `39a944cb3f85e797da58bd4ebd0091c60f7bd6d0`, confirmed by
  `git hash-object`.
- Only those two files were downloaded from the model.

The pinned `tokenizer.json` declares BPE, NFC normalization, one isolated
Split regex matching Memra's `qwen2` splitter, then ByteLevel without a prefix
space. `tokenizer_config.json` sets `eos_token` to `<|im_end|>` and has no BOS.
It contains a chat template, but this receipt tests raw text token IDs only.
It does not test template rendering.

The three prompts are:

1. Natural: `The brass clock stopped at 7:04, so I took the long way home.`
2. Code: `def fold(xs):\n    return sum(x*x for x in xs[:3])  # v2\n`
3. Multilingual: `Cafe\u0301, déjà vu; 你好世界。 שלום עולם! رقم ٣.`

`corpus.tsv` stores their exact UTF-8 bytes as hex. `ref-ids.tsv` stores both
reference ID arrays for each prompt. The decomposed accent in the multilingual
prompt exercises NFC.

## Replay from the worktree root

Run the commands below in a shell with `hf`, `uv`, `taskset`, Python 3 and Rust
installed. Delete the scratch directory after inspecting the result.

```sh
scratch=$(mktemp -d /tmp/mimo-tokenizer-cpu.XXXXXX)
hf download XiaomiMiMo/MiMo-V2.6-Flash-RL tokenizer.json tokenizer_config.json \
  --revision 3b38d063180c3e4aed9691fdc735f3d10b266ee4 \
  --local-dir "$scratch/hf" --max-workers 2
sha256sum "$scratch/hf/tokenizer.json" "$scratch/hf/tokenizer_config.json"
uv venv "$scratch/venv"
uv pip install --python "$scratch/venv/bin/python" 'tokenizers==0.22.1'
"$scratch/venv/bin/python" \
  research/mimo-tokenizer-cpu-20260927/generate_reference.py "$scratch/hf"
git diff --exit-code -- research/mimo-tokenizer-cpu-20260927/ref-ids.tsv
cpus=$(python3 -c 'import os; print(",".join(map(str, sorted(os.sched_getaffinity(0))[:3])))')
taskset -c "$cpus" env CARGO_TARGET_DIR="$scratch/target" CARGO_BUILD_JOBS=3 \
  cargo run --locked -j 3 -p memra-tokenizer --bin tok-parity -- \
  "$scratch/hf" research/mimo-tokenizer-cpu-20260927/corpus.tsv \
  research/mimo-tokenizer-cpu-20260927/ref-ids.tsv
rm -rf "$scratch"
```

The local CPU run printed:

```text
tok-parity: model=/tmp/mimo-tokenizer-sidecar-20260927 pre=qwen2 vocab=151675 bos=None add_bos matches encode(add_special)
OK natural
OK code
OK multilingual
tok-parity: 3/3 cases identical in BOTH modes
tok-parity: PASS
```

This receipt is limited to raw text tokenization of those three prompts. It
does not establish chat rendering, model loading, or serving readiness.
