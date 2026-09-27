"""Bounded CPU publisher oracle for MiMo-V2.6-Flash-RL audio codes.

Inputs are local files from revision 3b38d063180c3e4aed9691fdc735f3d10b266ee4:
  SOURCE_ROOT/modeling_mimo_v2.py
  SOURCE_ROOT/config.json
  SOURCE_ROOT/audio_tokenizer/config.json
  SOURCE_ROOT/audio_tokenizer/model.safetensors

The mel input is little-endian F32 [frames, 128] without a file header. The
checked-in mimo-pcm-mel-voiced-2048.logmel.f32 is one deterministic example.

  python crates/memra-reference/src/fixtures/mimo-v26-audio-publisher-oracle.py \
    SOURCE_ROOT --mel-f32 crates/memra-reference/src/fixtures/mimo-pcm-mel-voiced-2048.logmel.f32 \
    --out /tmp/mimo-v26-publisher-codes.json

Use --check-source to validate pinned source, configuration, AST extraction,
and mel geometry without weights or third-party packages. The full run needs
torch and safetensors. It loads no LLM or decoder. --device cuda:0 or cuda:1
uses only the named card on a dedicated two-card box.
"""

import argparse
import ast
from contextlib import redirect_stdout
import copy
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import platform
import struct
import sys
from types import SimpleNamespace


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
REVISION = "3b38d063180c3e4aed9691fdc735f3d10b266ee4"
WEIGHTS_SHA256 = "077033345d80eef3a315e8d394e0589667e80e4cdaba9bc5a7488410c6657265"
# x-linked-etag and x-linked-size for the pinned audio_tokenizer/model.safetensors.
WEIGHTS_BYTES = 1_872_618_384
HEADER_PATH = ROOT / "crates/memra-gguf/src/model_packs/mimo_v2/fixtures/audio-tokenizer-header.json"
MAX_MEL_FRAMES = 64
MEL_BINS = 128
CHANNELS = 20
GROUP_SIZE = 4
SOURCE_NAMES = {
    "EuclideanCodebook",
    "VectorQuantization",
    "ResidualVectorQuantization",
    "ResidualVectorQuantizer",
    "AudioTokenizerRotaryEmbedding",
    "_at_get_position_ids",
    "_at_get_sequence_mask",
    "_at_unpack_hidden_states",
    "_at_rotate_half",
    "_at_apply_rotary_pos_emb",
    "AudioTokenizerAttention",
    "AudioTokenizerTransformerLayer",
    "AudioTokenizerEncoder",
    "_pad_and_group_audio_codes",
    "_AT_LAYER_NORM",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def alignment_module():
    path = HERE / "mimo-v26-audio-source-alignment.py"
    spec = importlib.util.spec_from_file_location("mimo_v26_audio_alignment", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_source(root: Path):
    checker = alignment_module()
    for path, expected_sha256 in (
        (root / "modeling_mimo_v2.py", checker.SOURCE_SHA256),
        (root / "config.json", checker.MODEL_CONFIG_SHA256),
        (root / "audio_tokenizer/config.json", checker.CODEC_CONFIG_SHA256),
    ):
        found = sha256_file(path)
        if found != expected_sha256:
            raise ValueError(f"{path}: SHA256 {found} != pinned {expected_sha256}")
    capture = io.StringIO()
    with redirect_stdout(capture):
        checker.main(root)
    alignment = json.loads(capture.getvalue())
    if alignment["revision"] != REVISION or alignment["rvq_bins"] != [1024, 1024, 256] + [128] * 17:
        raise ValueError("pinned source alignment changed")
    codec_path = root / "audio_tokenizer/config.json"
    codec = json.loads(checker.pinned_bytes(codec_path, checker.CODEC_CONFIG_SHA256))
    if (
        codec["d_model"] != 1024
        or codec["n_mels"] != MEL_BINS
        or codec["num_quantizers"] != CHANNELS
        or codec["activation_function"] != "gelu"
        or codec["ln_type"] != "LayerNorm"
    ):
        raise ValueError("pinned audio encoder configuration changed")
    source_path = root / "modeling_mimo_v2.py"
    source = checker.pinned_bytes(source_path, checker.SOURCE_SHA256).decode("utf-8")
    tree = ast.parse(source, filename=str(source_path))
    selected = []
    found = set()
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in SOURCE_NAMES:
            selected.append(copy.deepcopy(node))
            found.add(node.name)
        elif isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "_AT_LAYER_NORM"
            for target in node.targets
        ):
            selected.append(copy.deepcopy(node))
            found.add("_AT_LAYER_NORM")
    if found != SOURCE_NAMES:
        raise ValueError(f"audio source node census changed: missing={sorted(SOURCE_NAMES-found)}")
    # The extracted code is compiled only after the exact publisher source hash
    # is checked. Its module imports, causal LM, vision path, and decoder are
    # never executed.
    compile_audio_nodes(selected, source_path)
    return alignment, codec, selected, source_path


def compile_audio_nodes(selected, source_path):
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(
        ast.Module(body=[future] + copy.deepcopy(selected), type_ignores=[])
    )
    return compile(module, str(source_path), "exec")


def read_mel(path: Path):
    raw = path.read_bytes()
    row_bytes = MEL_BINS * 4
    if not raw or len(raw) % row_bytes:
        raise ValueError("mel must be nonempty little-endian F32 [frames,128]")
    frames = len(raw) // row_bytes
    if frames > MAX_MEL_FRAMES:
        raise ValueError(f"mel has {frames} frames; bounded maximum is {MAX_MEL_FRAMES}")
    values = [value for (value,) in struct.iter_unpack("<f", raw)]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("mel contains a non-finite F32 value")
    return values, frames, hashlib.sha256(raw).hexdigest()


def check_header(path: Path, checker):
    if path.stat().st_size != WEIGHTS_BYTES:
        raise ValueError(f"checkpoint byte count differs from pinned {WEIGHTS_BYTES}")
    with path.open("rb") as stream:
        header_bytes = stream.read(8)
        if len(header_bytes) != 8:
            raise ValueError("truncated safetensors header length")
        (header_size,) = struct.unpack("<Q", header_bytes)
        if not 0 < header_size <= 2_000_000:
            raise ValueError("safetensors header exceeds bound")
        header_raw = stream.read(header_size)
        if len(header_raw) != header_size:
            raise ValueError("truncated safetensors header")
    actual = json.loads(header_raw)
    expected = json.loads(checker.pinned_bytes(HEADER_PATH, checker.HEADER_SHA256))
    actual_encoder = {key: value for key, value in actual.items() if key.startswith("encoder.")}
    expected_encoder = {key: value for key, value in expected.items() if key.startswith("encoder.")}
    if actual_encoder.keys() != expected_encoder.keys():
        missing = sorted(expected_encoder.keys() - actual_encoder.keys())
        unexpected = sorted(actual_encoder.keys() - expected_encoder.keys())
        raise ValueError(f"encoder tensor keys changed: missing={missing}, unexpected={unexpected}")
    if len(actual_encoder) != 449 or actual != expected:
        raise ValueError("safetensors tensor schema, dtype, shape, or offsets differ from pinned header")
    if max(entry["data_offsets"][1] for entry in actual.values()) + 8 + header_size != WEIGHTS_BYTES:
        raise ValueError("safetensors offsets do not reach pinned file size")
    return actual_encoder, hashlib.sha256(header_raw).hexdigest()


def load_audio_nodes(selected, source_path, torch):
    from torch import nn
    from torch.nn import functional as F

    # The pinned config has only "gelu". Transformers ACT2FN["gelu"] uses
    # torch.nn.functional.gelu; binding it here avoids importing the causal LM
    # and the rest of Transformers.
    scope = {
        "torch": torch,
        "nn": nn,
        "F": F,
        "math": math,
        "ACT2FN": {"gelu": F.gelu},
    }
    exec(compile_audio_nodes(selected, source_path), scope)
    return scope


def build_encoder(source, codec, torch, expected, device):
    config = SimpleNamespace(**codec)
    original_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        with torch.device("meta"):
            encoder = source["AudioTokenizerEncoder"](config)
            # The publisher changes RVQ to F32 before encoding. Do this on
            # meta so to_empty allocates the final dtype only once.
            encoder.quantizer.float()
    finally:
        torch.set_default_dtype(original_dtype)

    state = encoder.state_dict()
    model_keys = {"encoder." + name for name in state}
    if model_keys != expected.keys():
        missing = sorted(model_keys - expected.keys())
        unexpected = sorted(expected.keys() - model_keys)
        raise ValueError(f"encoder module/checkpoint keys differ: missing={missing}, unexpected={unexpected}")
    dtypes = {"BF16": torch.bfloat16, "F32": torch.float32}
    for name, tensor in state.items():
        info = expected["encoder." + name]
        if list(tensor.shape) != info["shape"] or tensor.dtype != dtypes[info["dtype"]]:
            raise ValueError(f"encoder tensor geometry or dtype changed: {name}")

    # to_empty loses the nonpersistent RoPE buffer. Recreate it with the
    # pinned publisher class after allocation; all checkpoint state is copied
    # separately below.
    encoder.to_empty(device=device)
    encoder.position_embedding = source["AudioTokenizerRotaryEmbedding"](
        codec["rope_theta"],
        codec["d_model"] // codec["encoder_attention_heads"],
        encoder.max_source_positions,
        codec["rope_type"],
        device=device,
    )
    nonpersistent = {
        name for name, _ in encoder.named_buffers() if name not in encoder.state_dict()
    }
    if nonpersistent != {"position_embedding.inv_freq"}:
        raise ValueError(f"unexpected nonpersistent encoder buffers: {sorted(nonpersistent)}")
    encoder.eval()
    return encoder


def copy_encoder_weights(encoder, path, expected, torch):
    from safetensors import safe_open

    state = encoder.state_dict()
    with safe_open(str(path), framework="pt", device="cpu") as reader:
        if {key for key in reader.keys() if key.startswith("encoder.")} != expected.keys():
            raise ValueError("safetensors reader encoder key census differs from pinned header")
        with torch.no_grad():
            for name, target in state.items():
                tensor = reader.get_tensor("encoder." + name)
                if tensor.shape != target.shape or tensor.dtype != target.dtype:
                    raise ValueError(f"safetensors reader tensor differs from module: {name}")
                target.copy_(tensor)


def run_publisher(source, encoder, mel, frames, codec, torch, device):
    features = torch.tensor(mel, dtype=torch.float32, device=device).reshape(frames, MEL_BINS)
    lengths = torch.tensor([frames], dtype=torch.long, device=device)
    with torch.inference_mode():
        depth_tensor, output_lengths = encoder.encode(
            features, input_lens=lengths, return_codes_only=True
        )
        token_tensor = depth_tensor.transpose(0, 1).contiguous()
        grouped_tensor = source["_pad_and_group_audio_codes"](
            token_tensor, audio_channels=CHANNELS, group_size=GROUP_SIZE
        )
        _, pre_rvq, _, no_quant_codes = encoder.encode(
            features, input_lens=lengths, use_quantizer=False
        )
    tokens = (frames + 3) // 4
    groups = (tokens + GROUP_SIZE - 1) // GROUP_SIZE
    if (
        list(depth_tensor.shape) != [CHANNELS, tokens]
        or list(token_tensor.shape) != [tokens, CHANNELS]
        or list(grouped_tensor.shape) != [groups, GROUP_SIZE, CHANNELS]
        or output_lengths.tolist() != [tokens]
        or list(pre_rvq.shape) != [tokens, codec["d_model"]]
        or pre_rvq.dtype != torch.bfloat16
        or no_quant_codes is not None
    ):
        raise ValueError("publisher code or grouping extent differs from pinned geometry")
    depth_major = depth_tensor.tolist()
    token_major = token_tensor.tolist()
    grouped = grouped_tensor.tolist()
    if any(
        not 0 <= code < codec["codebook_size"][depth]
        for depth, row in enumerate(depth_major)
        for code in row
    ):
        raise ValueError("publisher produced an RVQ ID outside its pinned bin count")
    if any(grouped[g][r] != token_major[min(g * GROUP_SIZE + r, tokens - 1)]
           for g in range(groups) for r in range(GROUP_SIZE)):
        raise ValueError("publisher grouped rows differ from expected final-row padding")
    residual = pre_rvq.float()
    margins = []
    second_codes = []
    for depth, layer in enumerate(encoder.quantizer.vq.layers):
        x = layer.project_in(residual)
        embed = layer._codebook.embed.t()
        distance = -(x.pow(2).sum(1, keepdim=True) - 2 * x @ embed + embed.pow(2).sum(0, keepdim=True))
        best = distance.max(dim=-1).indices
        if best.tolist() != depth_major[depth]:
            raise ValueError(f"publisher RVQ diagnostic replay changed depth {depth} IDs")
        best_score = distance.gather(1, best[:, None])[:, 0]
        remaining = distance.clone()
        remaining.scatter_(1, best[:, None], float("-inf"))
        second_score, second = remaining.max(dim=-1)
        margins.append((best_score - second_score).tolist())
        second_codes.append(second.tolist())
        residual = residual - layer.decode(best)
    bits = [int(value) & 0xffff for value in pre_rvq.contiguous().view(torch.int16).reshape(-1).tolist()]
    pre_rvq_bytes = struct.pack(f"<{len(bits)}H", *bits)
    return depth_major, token_major, grouped, pre_rvq_bytes, margins, second_codes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_root", type=Path)
    parser.add_argument("--mel-f32", type=Path, required=True, help="little-endian F32 [frames,128]")
    parser.add_argument("--weights", type=Path, help="default: SOURCE_ROOT/audio_tokenizer/model.safetensors")
    parser.add_argument("--check-source", action="store_true", help="no weights or third-party packages")
    parser.add_argument("--out", type=Path, help="write JSON receipt after a successful full run")
    parser.add_argument("--features-out", type=Path, help="write pre-RVQ BF16 feature rows after a full run")
    parser.add_argument("--device", choices=("cpu", "cuda:0", "cuda:1"), default="cpu")
    args = parser.parse_args()
    if args.check_source and (args.out or args.features_out or args.device != "cpu"):
        parser.error("--out and --features-out require a full checkpoint run")

    alignment, codec, selected, source_path = check_source(args.source_root)
    mel, frames, mel_sha256 = read_mel(args.mel_f32)
    receipt = {
        "publisher_repo": "XiaomiMiMo/MiMo-V2.6-Flash-RL",
        "publisher_revision": REVISION,
        "source_sha256": alignment["source_sha256"],
        "audio_config_sha256": alignment_module().CODEC_CONFIG_SHA256,
        "model_config_sha256": alignment_module().MODEL_CONFIG_SHA256,
        "mel_path": str(args.mel_f32.resolve()),
        "mel_sha256": mel_sha256,
        "mel_shape": [frames, MEL_BINS],
        "mel_dtype": "little_endian_f32",
        "expected_code_rows": (frames + 3) // 4,
        "source_nodes": sorted(SOURCE_NAMES),
    }
    if args.check_source:
        receipt["status"] = "source_and_mel_only"
        receipt["checkpoint_code_ids"] = "unqualified"
        print(json.dumps(receipt, sort_keys=True))
        return

    weights_path = args.weights or args.source_root / "audio_tokenizer/model.safetensors"
    checker = alignment_module()
    expected, header_sha256 = check_header(weights_path, checker)
    weights_sha256 = sha256_file(weights_path)
    if weights_sha256 != WEIGHTS_SHA256:
        raise ValueError(f"checkpoint SHA256 {weights_sha256} != pinned {WEIGHTS_SHA256}")

    # Set thread and device bounds before importing torch.
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "1"
    try:
        import torch
        import safetensors
    except ImportError as error:
        raise RuntimeError("full oracle needs existing CPU torch and safetensors installations") from error
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if args.device != "cpu" and (
        not torch.cuda.is_available()
        or torch.cuda.device_count() != 2
        or torch.cuda.get_device_name(args.device) != "NVIDIA RTX PRO 6000 Blackwell Server Edition"
    ):
        raise ValueError("publisher GPU request changed from the dedicated two-card box")
    source = load_audio_nodes(selected, source_path, torch)
    encoder = build_encoder(source, codec, torch, expected, args.device)
    copy_encoder_weights(encoder, weights_path, expected, torch)
    depth_major, token_major, grouped, pre_rvq_bytes, margins, second_codes = run_publisher(
        source, encoder, mel, frames, codec, torch, args.device
    )
    if args.features_out:
        args.features_out.write_bytes(pre_rvq_bytes)
    receipt.update({
        "status": "publisher_cpu_codes_generated",
        "memra_code_id_parity": "unchecked",
        "weights_path": str(weights_path.resolve()),
        "weights_sha256": weights_sha256,
        "weights_bytes": WEIGHTS_BYTES,
        "safetensors_header_sha256": header_sha256,
        "encoder_tensor_count": len(expected),
        "encoder_dtype": "bfloat16",
        "rvq_dtype": "float32",
        "code_layouts": {
            "depth_major": "[20,T]",
            "token_major": "[T,20]",
            "grouped_patch_input": "[G,4,20]",
        },
        "depth_major": depth_major,
        "token_major": token_major,
        "grouped_patch_input": grouped,
        "pre_rvq_features_shape": [(frames + 3) // 4, codec["d_model"]],
        "pre_rvq_features_dtype": "BF16",
        "pre_rvq_features_sha256": hashlib.sha256(pre_rvq_bytes).hexdigest(),
        "rvq_top_two_score_margin_depth_major": margins,
        "rvq_second_code_depth_major": second_codes,
        "code_ids_sha256": hashlib.sha256(
            json.dumps(depth_major, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "environment": {
            "device": args.device,
            "gpu_name": torch.cuda.get_device_name(args.device) if args.device != "cpu" else None,
            "gpu_capability": torch.cuda.get_device_capability(args.device) if args.device != "cpu" else None,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "safetensors": safetensors.__version__,
            "cpu_capability": torch.backends.cpu.get_cpu_capability(),
            "mkldnn_enabled": torch.backends.mkldnn.enabled,
            "torch_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "omp_num_threads": os.environ["OMP_NUM_THREADS"],
            "mkl_num_threads": os.environ["MKL_NUM_THREADS"],
        },
    })
    serialized = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.write_text(serialized)
    else:
        print(serialized, end="")


if __name__ == "__main__":
    try:
        main()
    except (AssertionError, ImportError, OSError, ValueError, RuntimeError, KeyError) as error:
        raise SystemExit(f"publisher oracle failed: {error}") from error
