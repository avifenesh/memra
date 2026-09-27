"""Check the pinned V2.6 publisher audio path against Memra's source contract.

Run after downloading source files only:

    hf download XiaomiMiMo/MiMo-V2.6-Flash-RL \
      --revision 3b38d063180c3e4aed9691fdc735f3d10b266ee4 \
      --include 'modeling_mimo_v2.py' --include 'config.json' \
      --include 'audio_tokenizer/config.json' --local-dir /tmp/mimo-v26-source
    python crates/memra-reference/src/fixtures/mimo-v26-audio-source-alignment.py \
      /tmp/mimo-v26-source

Uses only the Python standard library. It executes the pinned source's small
attention-schedule and output-length expressions; it does not load weights or
produce checkpoint code IDs.
"""

import ast
import copy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace


REVISION = "3b38d063180c3e4aed9691fdc735f3d10b266ee4"
SOURCE_SHA256 = "a8c3cb3aae473bcc15f023010547c919f15eba6546e6ed7efb61a8937b12f3ad"
CODEC_CONFIG_SHA256 = "e0702adae37947e0c980c38bae58ffa0d48bd492d523afe814aa4d73f008c7d1"
MODEL_CONFIG_SHA256 = "61bea4a0f7a0dd8969f8cae528761e26b697dd12ff63e98804c3f0945492e621"
HEADER_SHA256 = "45e94c498c6ae5214525ad1060da99dd2ef9be428c1eaf226987739984b33e53"
ROOT = Path(__file__).resolve().parents[4]


def pinned_bytes(path: Path, expected_sha256: str) -> bytes:
    content = path.read_bytes()
    found = hashlib.sha256(content).hexdigest()
    assert found == expected_sha256, f"{path}: SHA256 {found} != {expected_sha256}"
    return content


def named_node(body, name, kind):
    return next(node for node in body if isinstance(node, kind) and node.name == name)


def run_pinned_statements(statements, filename: Path, **scope):
    module = ast.fix_missing_locations(ast.Module(body=copy.deepcopy(statements), type_ignores=[]))
    exec(compile(module, str(filename), "exec"), scope)
    return scope


def main(source_root: Path):
    source_path = source_root / "modeling_mimo_v2.py"
    source = pinned_bytes(source_path, SOURCE_SHA256).decode()
    codec_bytes = pinned_bytes(source_root / "audio_tokenizer/config.json", CODEC_CONFIG_SHA256)
    model_bytes = pinned_bytes(source_root / "config.json", MODEL_CONFIG_SHA256)
    codec = json.loads(codec_bytes)
    model = json.loads(model_bytes)
    fixture = ROOT / "crates/memra-gguf/src/model_packs/mimo_v2/fixtures/audio-tokenizer-config.json"
    assert codec_bytes == fixture.read_bytes(), "Memra codec fixture differs from pinned publisher bytes"
    header_path = ROOT / "crates/memra-gguf/src/model_packs/mimo_v2/fixtures/audio-tokenizer-header.json"
    header = json.loads(pinned_bytes(header_path, HEADER_SHA256))

    tree = ast.parse(source, filename=str(source_path))
    encoder = named_node(tree.body, "AudioTokenizerEncoder", ast.ClassDef)
    constructor = named_node(encoder.body, "__init__", ast.FunctionDef)
    first = next(
        index for index, node in enumerate(constructor.body)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "attn_window_sizes" for target in node.targets)
    )
    schedule = run_pinned_statements(
        constructor.body[first : first + 2],
        source_path,
        config=SimpleNamespace(**codec),
    )["attn_window_sizes"]
    assert schedule == [(128, 0), (-1, -1)] * 12, schedule
    assert codec["hybrid_attention"] and codec["swa_per_block"] == 2
    # The published block-size config is present but this encoder loop does
    # not read it. A block-of-eight interpretation would change code IDs.
    assert "hybrid_block_size" not in ast.unparse(constructor.body[first + 1])

    get_length = named_node(encoder.body, "get_output_length", ast.FunctionDef)
    get_length_fn = run_pinned_statements([get_length], source_path)["get_output_length"]
    mel_lengths = (1, 4, 5, 6000, 6001)
    code_lengths = []
    for length in mel_lengths:
        conv_length = get_length_fn(SimpleNamespace(config=SimpleNamespace(**codec)), length)
        pooled_length = (conv_length + codec["avg_pooler"] - 1) // codec["avg_pooler"]
        assert conv_length == (length + 1) // 2
        assert pooled_length == (length + 3) // 4
        code_lengths.append(pooled_length)

    quantizer = named_node(tree.body, "ResidualVectorQuantization", ast.ClassDef)
    quantize = named_node(
        named_node(tree.body, "EuclideanCodebook", ast.ClassDef).body, "quantize", ast.FunctionDef
    )
    rvq_encode = named_node(quantizer.body, "encode", ast.FunctionDef)
    encoder_encode = named_node(encoder.body, "encode", ast.FunctionDef)
    batch = named_node(tree.body, "tokenize_audio_batch", ast.FunctionDef)
    grouping = named_node(tree.body, "_pad_and_group_audio_codes", ast.FunctionDef)
    attention = named_node(tree.body, "AudioTokenizerAttention", ast.ClassDef)
    attention_mask = named_node(attention.body, "_build_attn_mask", ast.FunctionDef)
    features = named_node(encoder.body, "get_features", ast.FunctionDef)
    for node, required in (
        (quantize, ("x.pow(2).sum", "2 * x @ embed", "dist.max(dim=-1).indices")),
        (rvq_encode, ("residual = residual - quantized", "torch.stack(all_indices)")),
        (encoder_encode, ("self.quantizer.float()", "hidden_states.float()", "return (codes, output_length)")),
        (batch, ("torch.cat(encoded_parts, dim=-1).transpose(0, 1)",)),
        (grouping, ("audio_codes[:, :audio_channels]", "audio_codes[-1:].expand", "audio_codes.reshape")),
        (attention_mask, ("self.causal", "torch.triu", "self.window_size[0]")),
        (features, ("skip_connect_hidden_states", "self.down_sample_layer", "self.down_sample_norm")),
    ):
        body = ast.unparse(node)
        for expression in required:
            assert expression in body, f"{node.name}: missing {expression}"

    bins = codec["codebook_size"]
    assert bins == [1024, 1024, 256] + [128] * 17
    encoder_keys = [name for name in header if name.startswith("encoder.")]
    decoder_keys = [name for name in header if name.startswith("decoder.")]
    assert len(encoder_keys) == 449 and len(decoder_keys) == 379
    for depth, size in enumerate(bins):
        codebook = f"encoder.quantizer.vq.layers.{depth}._codebook."
        assert header[codebook + "embed"]["dtype"] == "F32"
        assert header[codebook + "embed"]["shape"] == [size, codec["d_model"]]
    tokenizer = named_node(tree.body, "MiMoAudioTokenizer", ast.ClassDef)
    assert all(
        not (isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Attribute) and target.attr == "decoder" for target in node.targets
        ))
        for node in ast.walk(tokenizer)
    )
    full_model = named_node(tree.body, "MiMoV2ForCausalLM", ast.ClassDef)
    loader = named_node(full_model.body, "load_audio_tokenizer", ast.FunctionDef)
    assert "load_state_dict(state_dict, strict=False)" in ast.unparse(loader)
    audio = model["audio_config"]
    assert int(audio["audio_channels"]) == 20
    assert audio["group_size"] == 4
    assert int(audio["speech_vocab_size"]) == 1280
    assert audio["input_local_layers"] == 6
    print(json.dumps({
        "revision": REVISION,
        "source_sha256": SOURCE_SHA256,
        "encoder_attention": ["swa128" if window[0] == 128 else "global" for window in schedule],
        "mel_frames": mel_lengths,
        "code_rows": code_lengths,
        "rvq_bins": bins,
        "checkpoint_encoder_tensors": len(encoder_keys),
        "checkpoint_decoder_tensors_ignored_by_loader": len(decoder_keys),
        "source_code_layout": "depth,token",
        "patch_code_layout": "group,4,20",
        "checkpoint_code_ids": "unqualified",
    }, sort_keys=True))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} PINNED_SOURCE_ROOT")
    main(Path(sys.argv[1]))
