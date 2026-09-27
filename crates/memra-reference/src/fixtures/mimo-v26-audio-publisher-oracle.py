"""Bounded CPU publisher oracle for MiMo-V2.6-Flash-RL audio codes.

Inputs are local files from revision 3b38d063180c3e4aed9691fdc735f3d10b266ee4:
  SOURCE_ROOT/modeling_mimo_v2.py
  SOURCE_ROOT/config.json
  SOURCE_ROOT/audio_tokenizer/config.json
  SOURCE_ROOT/audio_tokenizer/model.safetensors

The mel input is little-endian F32 [frames, 128] without a file header. The
checked-in mimo-pcm-mel-voiced-2048.logmel.f32 is one deterministic example.
--stages-dir includes conv1_pre_gelu.bf16 as frame-major BF16 [frames, 1024]
([9, 1024] for the checked-in mel) alongside the existing post-GELU conv1
and later source stages. It also records layer0.bf16 after the first hybrid
attention transformer layer.
--conv-linear-control reruns that same pinned encoder and checkpoint with
conv1 and conv2 computed by BF16 im2col plus F.linear. The receipt records
both paths' code IDs and stage hashes. With --stages-dir, control tensors go
in its conv_linear_control subdirectory. This is a numeric diagnostic, not
a source-quality or production path.

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
from contextlib import contextmanager, redirect_stdout
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


def conv1d_bf16_linear(input_tensor, conv, torch):
    from torch.nn import functional as F

    if (
        input_tensor.ndim != 3
        or input_tensor.shape[0] != 1
        or not 0 < input_tensor.shape[2] <= MAX_MEL_FRAMES
        or input_tensor.shape[1] != conv.in_channels
        or input_tensor.dtype != torch.bfloat16
        or conv.weight.dtype != torch.bfloat16
        or conv.bias is None
        or conv.bias.dtype != torch.bfloat16
        or input_tensor.device != conv.weight.device
        or input_tensor.device != conv.bias.device
        or conv.groups != 1
        or conv.dilation != (1,)
        or conv.padding_mode != "zeros"
    ):
        raise ValueError("linear control requires bounded BF16 Conv1d geometry")
    kernel = conv.kernel_size[0]
    stride = conv.stride[0]
    padding = conv.padding[0]
    padded = F.pad(input_tensor, (padding, padding))
    windows = padded.unfold(2, kernel, stride)
    rows = windows.permute(0, 2, 1, 3).reshape(-1, conv.in_channels * kernel)
    output = F.linear(rows, conv.weight.reshape(conv.out_channels, -1), conv.bias)
    return (
        output.reshape(1, windows.shape[2], conv.out_channels)
        .transpose(1, 2)
        .contiguous()
    )


@contextmanager
def use_conv_linear_control(encoder, codec, torch):
    conv1 = encoder.conv1
    conv2 = encoder.conv2
    expected = (
        (conv1, MEL_BINS, codec["d_model"], 1),
        (conv2, codec["d_model"], codec["d_model"], codec["stride_size"]),
    )
    for conv, in_channels, out_channels, stride in expected:
        if (
            conv.in_channels != in_channels
            or conv.out_channels != out_channels
            or conv.kernel_size != (codec["kernel_size"],)
            or conv.stride != (stride,)
            or conv.padding != (1,)
        ):
            raise ValueError("publisher conv geometry changed from pinned linear control")
    original_conv1 = conv1.__dict__.get("forward")
    original_conv2 = conv2.__dict__.get("forward")
    try:
        conv1.forward = lambda tensor: conv1d_bf16_linear(tensor, conv1, torch)
        conv2.forward = lambda tensor: conv1d_bf16_linear(tensor, conv2, torch)
        yield
    finally:
        if original_conv1 is None:
            del conv1.forward
        else:
            conv1.forward = original_conv1
        if original_conv2 is None:
            del conv2.forward
        else:
            conv2.forward = original_conv2


def stage_receipt(stage_bytes, codec):
    return {
        name: {
            "sha256": hashlib.sha256(values).hexdigest(),
            "shape": [len(values) // (2 * codec["d_model"]), codec["d_model"]],
            "dtype": "BF16",
        }
        for name, values in stage_bytes.items()
    }


def compare_bf16_stages(publisher_bytes, control_bytes):
    if len(publisher_bytes) != len(control_bytes) or len(publisher_bytes) % 2:
        raise ValueError("publisher and linear control stage extents differ")
    equal = 0
    squared_error = 0.0
    squared_publisher = 0.0
    for (publisher_bits,), (control_bits,) in zip(
        struct.iter_unpack("<H", publisher_bytes),
        struct.iter_unpack("<H", control_bytes),
    ):
        equal += publisher_bits == control_bits
        publisher_value = struct.unpack("<f", struct.pack("<I", publisher_bits << 16))[0]
        control_value = struct.unpack("<f", struct.pack("<I", control_bits << 16))[0]
        if not math.isfinite(publisher_value) or not math.isfinite(control_value):
            raise ValueError("non-finite BF16 source stage in linear control comparison")
        squared_error += (publisher_value - control_value) ** 2
        squared_publisher += publisher_value ** 2
    return {
        "matching_bf16_values": equal,
        "total_bf16_values": len(publisher_bytes) // 2,
        "relative_l2": math.sqrt(squared_error / squared_publisher) if squared_publisher else None,
    }


def run_publisher(source, encoder, mel, frames, codec, torch, device, capture_stages=False):
    features = torch.tensor(mel, dtype=torch.float32, device=device).reshape(frames, MEL_BINS)
    lengths = torch.tensor([frames], dtype=torch.long, device=device)
    stage_snapshots = {}
    handles = []
    if capture_stages:
        def after_conv1(_module, _args, output):
            if list(output.shape) != [1, codec["d_model"], frames] or output.dtype != torch.bfloat16:
                raise ValueError("publisher conv1 pre-GELU output shape or dtype changed")
            stage_snapshots["conv1_pre_gelu"] = output.detach().transpose(1, 2).reshape(
                frames, codec["d_model"]
            ).clone()

        def before_conv2(_module, args):
            stage_snapshots["conv1"] = args[0].detach().transpose(1, 2).reshape(
                frames, codec["d_model"]
            ).clone()

        def after_conv2(_module, _args, output):
            out_frames = (frames + 1) // 2
            if list(output.shape) != [1, codec["d_model"], out_frames] or output.dtype != torch.bfloat16:
                raise ValueError("publisher conv2 pre-GELU output shape or dtype changed")
            stage_snapshots["conv2_pre_gelu"] = output.detach().transpose(1, 2).reshape(
                out_frames, codec["d_model"]
            ).clone()

        def before_frontend_layer(_module, args):
            stage_snapshots["frontend"] = args[0].detach().clone()

        def after_layer0(_module, _args, output):
            stage_snapshots["layer0"] = output.detach().clone()

        def after_stack_norm(_module, _args, output):
            stage_snapshots["stack"] = output.detach().clone()

        handles = [
            encoder.conv1.register_forward_hook(after_conv1),
            encoder.conv2.register_forward_pre_hook(before_conv2),
            encoder.conv2.register_forward_hook(after_conv2),
            encoder.layers[0].register_forward_pre_hook(before_frontend_layer),
            encoder.layers[0].register_forward_hook(after_layer0),
            encoder.layer_norm.register_forward_hook(after_stack_norm),
        ]
    with torch.inference_mode():
        depth_tensor, output_lengths = encoder.encode(
            features, input_lens=lengths, return_codes_only=True
        )
        token_tensor = depth_tensor.transpose(0, 1).contiguous()
        grouped_tensor = source["_pad_and_group_audio_codes"](
            token_tensor, audio_channels=CHANNELS, group_size=GROUP_SIZE
        )
        try:
            _, pre_rvq, _, no_quant_codes = encoder.encode(
                features, input_lens=lengths, use_quantizer=False
            )
        finally:
            for handle in handles:
                handle.remove()
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
    stage_bytes = {}
    if capture_stages:
        stage_snapshots["pre_rvq"] = pre_rvq.detach().clone()
        expected = {
            "conv1_pre_gelu": [frames, codec["d_model"]],
            "conv1": [frames, codec["d_model"]],
            "conv2_pre_gelu": [(frames + 1) // 2, codec["d_model"]],
            "frontend": [(frames + 1) // 2, codec["d_model"]],
            "layer0": [(frames + 1) // 2, codec["d_model"]],
            "stack": [(frames + 1) // 2, codec["d_model"]],
            "pre_rvq": [tokens, codec["d_model"]],
        }
        if stage_snapshots.keys() != expected.keys():
            raise ValueError("publisher stage hooks omitted a source stage")
        for name, tensor in stage_snapshots.items():
            if list(tensor.shape) != expected[name] or tensor.dtype != torch.bfloat16:
                raise ValueError(f"publisher {name} stage shape or dtype changed")
            signed = tensor.to("cpu").contiguous().view(torch.int16).reshape(-1).tolist()
            values = [int(value) & 0xffff for value in signed]
            stage_bytes[name] = struct.pack(f"<{len(values)}H", *values)
    return depth_major, token_major, grouped, pre_rvq_bytes, margins, second_codes, stage_bytes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_root", type=Path)
    parser.add_argument("--mel-f32", type=Path, required=True, help="little-endian F32 [frames,128]")
    parser.add_argument("--weights", type=Path, help="default: SOURCE_ROOT/audio_tokenizer/model.safetensors")
    parser.add_argument("--check-source", action="store_true", help="no weights or third-party packages")
    parser.add_argument("--out", type=Path, help="write JSON receipt after a successful full run")
    parser.add_argument("--features-out", type=Path, help="write pre-RVQ BF16 feature rows after a full run")
    parser.add_argument("--device", choices=("cpu", "cuda:0", "cuda:1"), default="cpu")
    parser.add_argument("--stages-dir", type=Path, help="write bounded publisher BF16 stage tensors")
    parser.add_argument(
        "--conv-linear-control",
        action="store_true",
        help="rerun conv1 and conv2 with BF16 im2col plus F.linear and compare source outputs",
    )
    args = parser.parse_args()
    if args.check_source and (
        args.out or args.features_out or args.stages_dir or args.conv_linear_control
        or args.device != "cpu"
    ):
        parser.error("--check-source cannot be combined with full-run options")

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
    depth_major, token_major, grouped, pre_rvq_bytes, margins, second_codes, stage_bytes = run_publisher(
        source, encoder, mel, frames, codec, torch, args.device,
        args.stages_dir is not None or args.conv_linear_control,
    )
    control = None
    if args.conv_linear_control:
        with use_conv_linear_control(encoder, codec, torch):
            control = run_publisher(source, encoder, mel, frames, codec, torch, args.device, True)
    if args.features_out:
        args.features_out.write_bytes(pre_rvq_bytes)
    if args.stages_dir:
        args.stages_dir.mkdir(parents=True, exist_ok=True)
        for name, values in stage_bytes.items():
            (args.stages_dir / f"{name}.bf16").write_bytes(values)
        if control is not None:
            control_dir = args.stages_dir / "conv_linear_control"
            control_dir.mkdir(parents=True, exist_ok=True)
            for name, values in control[6].items():
                (control_dir / f"{name}.bf16").write_bytes(values)
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
        "stages": stage_receipt(stage_bytes, codec),
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
    if control is not None:
        (
            control_depth, control_tokens, control_grouped, control_features,
            control_margins, control_second, control_stages,
        ) = control
        baseline_stages = receipt["stages"]
        linear_stages = stage_receipt(control_stages, codec)
        if linear_stages.keys() != baseline_stages.keys():
            raise ValueError("linear control stage census differs from publisher")
        mismatches = [
            {
                "depth": depth,
                "token": token,
                "publisher": code,
                "linear_control": control_depth[depth][token],
            }
            for depth, row in enumerate(depth_major)
            for token, code in enumerate(row)
            if code != control_depth[depth][token]
        ]
        receipt["conv_linear_control"] = {
            "method": "BF16 im2col + torch.nn.functional.linear for conv1 and conv2",
            "conv1_geometry": {
                "in_channels": encoder.conv1.in_channels,
                "out_channels": encoder.conv1.out_channels,
                "kernel_size": list(encoder.conv1.kernel_size),
                "stride": list(encoder.conv1.stride),
                "padding": list(encoder.conv1.padding),
            },
            "conv2_geometry": {
                "in_channels": encoder.conv2.in_channels,
                "out_channels": encoder.conv2.out_channels,
                "kernel_size": list(encoder.conv2.kernel_size),
                "stride": list(encoder.conv2.stride),
                "padding": list(encoder.conv2.padding),
            },
            "depth_major": control_depth,
            "token_major": control_tokens,
            "grouped_patch_input": control_grouped,
            "code_ids_sha256": hashlib.sha256(
                json.dumps(control_depth, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
            "rvq_top_two_score_margin_depth_major": control_margins,
            "rvq_second_code_depth_major": control_second,
            "source_output_code_ids_changed": bool(mismatches),
            "code_id_mismatches": mismatches,
            "pre_rvq_features_sha256": hashlib.sha256(control_features).hexdigest(),
            "pre_rvq_features_changed": control_features != pre_rvq_bytes,
            "stages": linear_stages,
            "stage_changed": {
                name: baseline_stages[name]["sha256"] != linear_stages[name]["sha256"]
                for name in baseline_stages
            },
            "stage_comparison": {
                name: compare_bf16_stages(stage_bytes[name], control_stages[name])
                for name in baseline_stages
            },
            "numeric_environment": {
                **receipt["environment"],
                "cuda_runtime": torch.version.cuda,
                "cudnn_version": torch.backends.cudnn.version(),
                "cudnn_enabled": torch.backends.cudnn.enabled,
                "cudnn_benchmark": torch.backends.cudnn.benchmark,
                "cudnn_deterministic": torch.backends.cudnn.deterministic,
                "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                "float32_matmul_precision": torch.get_float32_matmul_precision(),
                "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            },
        }
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
