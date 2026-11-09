"""Tiny-gate oracle for the deepseek_v41 pack, produced by the vendor DeepSeek-V4.1 reference program itself.

The vendor `inference/model.py` and `engram.py` (deepseek-ai/DeepSeek-V4.1-Flash-1017 at revision
1bae0e1088993b9b82de08f59de5be5a34e42170, sha256-pinned below) run unchanged on CPU, with the torch stand-ins in
../oracle/kernel.py shadowing the tilelang kernel module. The weights are the checkpoint image memra writes for its
tiny fixture (`memra model verify tiny --against deepseek_v41 --out OUT` -> OUT/tiny-checkpoint/: model.safetensors
in checkpoint storage, model-args.json with the vendor ModelArgs the plan corresponds to, fixture.json with the token
stream, the prefill/decode split and the compressed token map).

Protocol (the same as memra's `execute_tokens`): prefill token_ids[:decode_from] at position 0 and seed the DSpark
windows (forward_spec at position 0), then one decode step per later token, each followed (except the last) by a
DSpark draft from the next token. Logits for every position come from the trunk's final norm output times the head,
the same product the vendor head computes for the last position. Writes the TSV memra's tiny gate pins
(crates/memra-cli/tests/fixtures/tiny-oracle/deepseek_v41.tsv).

Around the vendor code: a forward hook on the final norm (observation only); ../oracle/exact.py, which evaluates every
GEMM, reduction and transcendental op to correct rounding (the free choices of the vendor program, pinned the way
memra's reference pins them; see its docstring); Tensor.topk replaced by the CUDA selection rule; the Engram table gather through a uint8 view, since CPU F.embedding has no fp8 kernel (same
arithmetic as the vendor ParallelEngramEmbedding.forward); a stand-in tokenizer whose decoded strings normalize onto
the fixture's token map through the vendor build_compressed_token_map.

Why topk: the index scores are bf16, so equal scores at the top-k boundary are common (index top-k, candidate blocks).
Which tied entry wins is the implementation's choice. torch's CUDA topk, the vendor program's runtime, writes the
entries above the k-th value and then fills the remaining slots with entries equal to it in index order, so the lowest
index wins; torch's CPU topk decides by libstdc++ nth_element's partitioning. The stand-in takes the k largest by a
stable descending sort (ties to the lowest index), the CUDA rule and memra's, and the run reports how many boundary
ties it resolved.

usage: uv run --with torch,numpy,sympy,tokenizers,pillow python3 -I make_tiny_oracle.py \\
           OUT/tiny-checkpoint VENDOR_INFERENCE_DIR OUT_TSV [--hidden OUT_PT]
"""
import argparse
import hashlib
import json
import os
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PINNED = {
    "model.py": "e5decd76fe70fd56ef2f10124d245c6f64fb6a0003cc9068a1d3182a168b8d2d",
    "engram.py": "11f35ecbead8150c35aa002b3d180ef290b05a25afe883a11884f94d476d3897",
}

ap = argparse.ArgumentParser()
ap.add_argument("fixture")
ap.add_argument("vendor")
ap.add_argument("out")
ap.add_argument("--hidden", default="", help="also save the per-layer residual stream of every position (.pt)")
a = ap.parse_args()
for name, digest in PINNED.items():
    actual = hashlib.sha256(open(os.path.join(a.vendor, name), "rb").read()).hexdigest()
    assert actual == digest, f"{name} is not the pinned vendor revision ({actual})"
sys.path[:0] = [os.path.join(HERE, "..", "oracle"), a.vendor]

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

torch.set_num_threads(4)
torch.set_default_dtype(torch.bfloat16)
import exact  # noqa: E402
import kernel  # noqa: E402  (the stand-ins; must shadow the vendor kernel module)

exact.install()

assert os.path.dirname(os.path.abspath(kernel.__file__)) == os.path.abspath(os.path.join(HERE, "..", "oracle"))
import model as M  # noqa: E402
from engram import build_compressed_token_map  # noqa: E402

DT = {"F8_E4M3": torch.float8_e4m3fn, "F8_E8M0": torch.float8_e8m0fnu, "BF16": torch.bfloat16, "F32": torch.float32,
      "I8": torch.int8}


def load_safetensors(path):
    data = open(path, "rb").read()
    n = struct.unpack("<Q", data[:8])[0]
    header = json.loads(data[8:8 + n])
    base = 8 + n
    out = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        s, e = info["data_offsets"]
        buf = bytearray(data[base + s:base + e])
        out[name] = torch.frombuffer(buf, dtype=torch.uint8).view(DT[info["dtype"]]).view(info["shape"]).clone()
    return out


def engram_embed_forward(self, indices):
    """ParallelEngramEmbedding.forward without F.embedding on fp8: gather the raw bytes, then the vendor dequant."""
    mask = (indices < self.vocab_start_idx) | (indices >= self.vocab_end_idx)
    li = (indices - self.vocab_start_idx).masked_fill(mask, 0)
    flat = li.reshape(-1)
    values = self.weight.view(torch.uint8).index_select(0, flat).view(torch.float8_e4m3fn).view(*li.shape, -1)
    scales = self.scale.view(torch.uint8).index_select(0, flat).view(torch.float8_e8m0fnu).view(*li.shape, -1)
    values = values.float().unflatten(-1, (-1, self.block_size)) * scales.float().unsqueeze(-1)
    values = values.flatten(-2).to(torch.bfloat16)
    return values.masked_fill(mask.unsqueeze(-1), 0)


M.ParallelEngramEmbedding.forward = engram_embed_forward

ties = {"rows": 0, "tied_rows": 0}


def cuda_rule_topk(self, k, dim=-1, largest=True, sorted=True):
    """torch.topk with the CUDA tie rule: the k largest, equal values taken in index order."""
    assert dim in (-1, self.dim() - 1) and largest
    order = torch.sort(self, dim=-1, descending=True, stable=True)
    if 0 < k < self.size(-1):
        kth, nxt = order.values[..., k - 1], order.values[..., k]
        tied = (kth == nxt) & torch.isfinite(kth)
        ties["rows"] += kth.numel()
        ties["tied_rows"] += int(tied.sum())
    indices = order.indices[..., :k]
    if not sorted:
        indices = indices.sort(dim=-1).values
    return torch.return_types.topk((self.gather(-1, indices), indices))


torch.Tensor.topk = cuda_rule_topk


class Backend:
    def __init__(self, texts):
        self.texts = texts

    def decode(self, ids, skip_special_tokens=False):
        return self.texts[ids[0]]

    def id_to_token(self, token_id):
        return self.texts[token_id]


class Tokenizer:
    """Decoded strings that collapse onto `token_map` under the vendor normalizer (case, whitespace)."""

    def __init__(self, token_map):
        texts, seen = [], set()
        for token, cid in enumerate(token_map):
            if cid not in seen:
                assert cid == len(seen), "compressed ids are numbered in order of first appearance"
                seen.add(cid)
                texts.append(f"w{cid}")
            else:
                texts.append([f"W{cid}", f" w{cid}", f"w{cid}  ", f"\tW{cid}"][token % 4])
        self.backend_tokenizer = Backend(texts)
        self.n = len(texts)

    def __len__(self):
        return self.n


fixture = json.load(open(os.path.join(a.fixture, "fixture.json")))
cfg = json.load(open(os.path.join(a.fixture, "model-args.json")))
tokens, prefill, token_map = fixture["token_ids"], fixture["decode_from"], fixture["engram_token_map"]
tok = Tokenizer(token_map)
derived, size = build_compressed_token_map(tok)
assert derived == token_map and size == cfg["engram_compressed_vocab_size"], "stand-in tokenizer does not realize the map"
# A restricted indexer must always find index_topk candidate entries, else torch picks among -inf entries that are
# reachable but outside the pool, an implementation-defined choice.
assert (cfg["candidate_topk_blocks"] - 1) * cfg["candidate_block_size"] + 1 >= cfg["index_topk"]

cfg.update(max_batch_size=1, max_seq_len=len(tokens) + cfg["dspark_block_size"] + 8, temperature=0.0)
args = M.ModelArgs(**cfg)
with torch.device("cpu"):
    model = M.Transformer(args, tok)

ck = load_safetensors(os.path.join(a.fixture, "model.safetensors"))
used = set()
for name, p in dict(model.named_parameters(remove_duplicate=False)).items():
    if name.startswith("mtp.") and name.split(".", 2)[-1] in ("embed.weight", "head.weight"):
        continue  # tied to the backbone (the same Parameter object)
    assert name in ck, f"checkpoint image misses {name}"
    t = ck[name]
    used.add(name)
    if name.endswith("attn.wo_a.weight") and p.dtype == torch.bfloat16 and t.dtype == torch.float8_e4m3fn:
        s = ck[name.replace("weight", "scale")]
        used.add(name.replace("weight", "scale"))
        bo, bi = t.size(0) // s.size(0), t.size(1) // s.size(1)
        t = (t.float().unflatten(0, (-1, bo)).unflatten(-1, (-1, bi)) * s.float()[:, None, :, None]).flatten(2, 3).flatten(0, 1)
    if p.dtype == torch.float4_e2m1fn_x2 and t.dtype == torch.int8:
        t = t.view(torch.float4_e2m1fn_x2)
    elif p.dtype != t.dtype:
        t = t.to(p.dtype)
    assert tuple(t.shape) == tuple(p.shape), (name, tuple(t.shape), tuple(p.shape))
    p.data = t
unused = sorted(set(ck) - used)
assert not unused, f"checkpoint image tensors the vendor model does not read: {unused[:8]}"
model.eval()

normed = []
model.norm.register_forward_hook(lambda m, i, o: normed.append(o.detach().clone()))
hidden = []
for li, layer in enumerate(model.layers):
    layer.register_forward_hook(lambda m, i, o, li=li: hidden.append((li, o[0].detach().float().clone())))


def last_logits():
    return F.linear(normed[-1][0].float(), model.head.weight)  # [s, vocab]


rows, drafts = [], []
with torch.inference_mode():
    x = torch.tensor([tokens])
    _, _, main_hidden = model(x[:, :prefill], 0)
    rows.append(last_logits())
    model.forward_spec(torch.tensor([tokens[prefill]]), main_hidden, 0)
    for pos in range(prefill, len(tokens)):
        _, _, main_hidden = model(x[:, pos:pos + 1], pos)
        rows.append(last_logits())
        if pos + 1 < len(tokens):
            output_ids, logits, confidence = model.forward_spec(torch.tensor([tokens[pos + 1]]), main_hidden, pos)
            drafts.append((pos, output_ids[0].tolist(), logits[0].float(), confidence[0].float()))
logits = torch.cat(rows)
assert logits.shape == (len(tokens), cfg["vocab_size"]) and bool(torch.isfinite(logits).all())


def bits(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


with open(a.out, "w") as out:
    out.write("stream\tposition\ttoken\tlogit_f32_bits\n")
    for pos, row in enumerate(logits.tolist()):
        for token, value in enumerate(row):
            out.write(f"main\t{pos}\t{token}\t{bits(value):08x}\n")
    for pos, output_ids, block_logits, confidence in drafts:
        for i, row in enumerate(block_logits.tolist()):
            for token, value in enumerate(row):
                out.write(f"dspark@{pos}\t{i}\t{token}\t{bits(value):08x}\n")
        for i, value in enumerate(confidence.tolist()):
            out.write(f"dspark-confidence@{pos}\t{i}\t{output_ids[i + 1]}\t{bits(value):08x}\n")
if a.hidden:
    torch.save(hidden, a.hidden)
print(json.dumps({"tokens": len(tokens), "prefill": prefill, "drafts": len(drafts), "torch": torch.__version__,
                  "topk_rows": ties["rows"], "topk_rows_with_boundary_tie": ties["tied_rows"],
                  "out": a.out, "sha256": hashlib.sha256(open(a.out, "rb").read()).hexdigest()}))
