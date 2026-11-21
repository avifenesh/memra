#!/usr/bin/env python3
"""Inkling 2 Mini DSpark drafter: a float64 torch port of the vLLM image's drafter, and the
tiny fixture's generator and pinned oracle.

Sources (vLLM image vllm/vllm-openai:inkling-2-mini, index digest sha256:c2dc759f...; paths
under vllm/models/inkling2/ = I1, vllm/v1/worker/gpu/spec_decode/ = SD):

PORTED LINE BY LINE (same structure, same order of operations, float64 tensors):
  - I1/nvidia/dspark.py Inkling2DraftLayer.forward (pre-norm residual order), Inkling2DraftMLP,
    Inkling2DSparkDecoder.embed_block / forward / mix, Inkling2MarkovHead.context / step / fold
    (the fold is kept in float64 and never stored in bf16), Inkling2DSparkModel.
    refresh_derived_weights (fuse_alpha = softmax(fuse_weight)), precompute_and_store_context_kv,
    compute_f_rnn, compute_base_logits, markov_logits.
  - I1/speculator.py Inkling2DSparkSpeculator.prepare_context_anchor (f_rnn from the last
    committed row) and _sample_sequential (state zeros, prev = anchor, 7 sequential steps).
  - I1/nvidia/dspark.py Inkling2DraftSelfAttention.forward / store_context_kv (projection,
    per-head norm, RoPE, split order).
REPLACED BY THEIR MATH (the kernels are triton/gluon; the port states what they compute):
  - ops/dspark.py mix_rmsnorm, qk_norm_rope_, rope_, silu_and_mul, add_softcap, markov_gate:
    same formulas in float64 with every bf16 rounding point removed (the oracle class is the
    exact program; the bf16 points are vLLM's serving class).
  - ops/attention.py block_decode_attention (_rel_verify_kernel with DRAFT): a dense masked
    softmax over [context keys | block keys]; local layers keep the keys at
    pos0 - (WINDOW - 1) ..= pos0 with pos0 = P + block - 1 and WINDOW = sliding_window + block.
  - The paged KV cache: one dense [positions, kv_heads, head_dim] array per layer.
  - get_rope's cos/sin cache (fp32 in vLLM): computed in float64 from the same formula
    inv_freq = theta ** (-arange(0, d, 2) / d), angle = position * inv_freq.
NOT PORTED: tensor parallelism and all-reduce fusions (one rank), CUDA graphs, Gumbel sampling
(the oracle is greedy plus a teacher-forced Markov chain).

usage: dspark_oracle.py generate OUT_DIR    tiny weights + taps (bf16-exact values), then the oracle
"""
import hashlib
import json
import struct
import sys
from pathlib import Path

import torch

torch.set_default_dtype(torch.float64)

TINY = dict(
    hidden_size=64, num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
    head_dim=16, intermediate_size=96, rms_norm_eps=1e-6, rope_theta=1e6,
    local_layer_ids=[0, 2], sliding_window_size=5, block_size=8, markov_rank=8,
    taps=4, padded_vocab=48, vocab=40, softcap=30.0, embed_eps=1e-6,
)
SEED = 20261121
SCENARIOS = [1, 4, 13]  # committed positions before the anchor


# ---- safetensors (no dependency) ---------------------------------------------------------------

def bf16_round(t):
    return t.to(torch.float32).to(torch.bfloat16).to(torch.float64)


def write_st(path, tensors):
    header, blobs, off = {}, [], 0
    for name in sorted(tensors):
        t = tensors[name]
        if t.dtype == torch.bfloat16:
            raw = t.contiguous().view(torch.int16).numpy().tobytes()
            dt = "BF16"
        elif t.dtype == torch.float64:
            raw = t.contiguous().numpy().tobytes()
            dt = "F64"
        elif t.dtype == torch.int64:
            raw = t.contiguous().numpy().tobytes()
            dt = "I64"
        else:
            raise SystemExit(f"{name}: dtype {t.dtype}")
        header[name] = {"dtype": dt, "shape": list(t.shape), "data_offsets": [off, off + len(raw)]}
        blobs.append(raw)
        off += len(raw)
    h = json.dumps(header, separators=(",", ":")).encode()
    h += b" " * ((8 - len(h) % 8) % 8)
    data = struct.pack("<Q", len(h)) + h + b"".join(blobs)
    Path(path).write_bytes(data)
    return hashlib.sha256(data).hexdigest()


# ---- the program (float64) ---------------------------------------------------------------------

def rms(x, w, eps):
    # flashinfer rmsnorm / the mix kernel: x * rsqrt(mean(x^2) + eps) * w
    return x * torch.rsqrt((x * x).mean(-1, keepdim=True) + eps) * w


def rope(x, pos, theta):
    # neox halves; x [T, H, D], pos [T]
    d = x.shape[-1]
    inv = theta ** (-torch.arange(0, d, 2, dtype=torch.float64) / d)
    ang = pos.to(torch.float64)[:, None] * inv[None, :]
    cos, sin = ang.cos()[:, None, :], ang.sin()[:, None, :]
    x1, x2 = x[..., : d // 2], x[..., d // 2 :]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], -1)


class Drafter:
    def __init__(self, w, cfg):
        self.w, self.c = w, cfg
        # refresh_derived_weights
        self.fuse_alpha = torch.softmax(w["fuse_weight"], dim=-1)
        self.ctx_k = [[] for _ in range(cfg["num_hidden_layers"])]
        self.ctx_v = [[] for _ in range(cfg["num_hidden_layers"])]

    def L(self, i, name):
        return self.w[f"layers.{i}.{name}"]

    # Inkling2DSparkDecoder.mix
    def mix(self, features, row, norm_w):
        h = self.c["hidden_size"]
        targets = features.view(features.shape[0], -1, h)
        mixed = (targets * self.fuse_alpha[row][None, :, None]).sum(1)
        return rms(mixed, norm_w, self.c["rms_norm_eps"])

    # precompute_and_store_context_kv + store_context_kv
    def store_context(self, features, positions):
        c = self.c
        kvw = c["num_key_value_heads"] * c["head_dim"]
        for i in range(c["num_hidden_layers"]):
            context = self.mix(features, i, self.L(i, "dflare_norm.weight"))
            k = context @ self.L(i, "self_attn.k_proj_ctx.weight").T
            v = context @ self.L(i, "self_attn.v_proj_ctx.weight").T
            assert k.shape[-1] == kvw
            k = rms(k.reshape(-1, c["head_dim"]), self.L(i, "self_attn.k_norm_ctx.weight"), c["rms_norm_eps"])
            k = k.view(-1, c["num_key_value_heads"], c["head_dim"])
            k = rope(k, positions, c["rope_theta"])
            self.ctx_k[i].extend(list(k))
            self.ctx_v[i].extend(list(v.view(-1, c["num_key_value_heads"], c["head_dim"])))

    # Inkling2DraftSelfAttention.forward over one block
    def attn(self, i, positions, x):
        c = self.c
        nh, nkv, d = c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]
        q = (x @ self.L(i, "self_attn.q_proj.weight").T).view(-1, nh, d)
        k = (x @ self.L(i, "self_attn.k_proj.weight").T).view(-1, nkv, d)
        v = (x @ self.L(i, "self_attn.v_proj.weight").T).view(-1, nkv, d)
        q = rope(rms(q, self.L(i, "self_attn.q_norm.weight"), c["rms_norm_eps"]), positions, c["rope_theta"])
        k = rope(rms(k, self.L(i, "self_attn.k_norm.weight"), c["rms_norm_eps"]), positions, c["rope_theta"])
        P = int(positions[0])
        block = c["block_size"]
        ck = torch.stack(self.ctx_k[i][:P]) if P else k[:0]
        cv = torch.stack(self.ctx_v[i][:P]) if P else v[:0]
        keys = torch.cat([ck, k], 0)  # positions 0..P+block-1
        vals = torch.cat([cv, v], 0)
        kpos = torch.arange(P + block)
        pos0 = P + block - 1
        if i in c["local_layer_ids"]:
            window = c["sliding_window_size"] + block
            keep = kpos >= pos0 - (window - 1)
        else:
            keep = torch.ones_like(kpos, dtype=torch.bool)
        keys, vals = keys[keep], vals[keep]
        g = nh // nkv
        out = torch.empty(block, nh, d)
        for h in range(nh):
            s = (q[:, h, :] @ keys[:, h // g, :].T) * d ** -0.5
            p = torch.softmax(s, -1)
            out[:, h, :] = p @ vals[:, h // g, :]
        return out.reshape(block, nh * d) @ self.L(i, "self_attn.o_proj.weight").T

    # Inkling2DSparkDecoder.embed_block + forward (layers: Inkling2DraftLayer.forward)
    def block_forward(self, anchor, P, shared):
        c = self.c
        block = c["block_size"]
        x = self.w["mask_embed"].expand(block, -1).clone()
        x[0] = rms(shared["embed"][anchor], shared["embed_norm"], c["embed_eps"])
        positions = torch.arange(P, P + block)
        residual = None
        mlp_out = None
        for i in range(c["num_hidden_layers"]):
            if residual is None:
                residual = x
                h = rms(x, self.L(i, "input_layernorm.weight"), c["rms_norm_eps"])
            else:
                residual = residual + mlp_out
                h = rms(residual, self.L(i, "input_layernorm.weight"), c["rms_norm_eps"])
            o = self.attn(i, positions, h)
            residual = residual + o
            h = rms(residual, self.L(i, "post_attention_layernorm.weight"), c["rms_norm_eps"])
            gate = h @ self.L(i, "mlp.gate_proj.weight").T
            up = h @ self.L(i, "mlp.up_proj.weight").T
            act = gate / (1 + torch.exp(-gate)) * up
            mlp_out = act @ self.L(i, "mlp.down_proj.weight").T
        residual = residual + mlp_out
        return rms(residual, self.w["norm.weight"], c["rms_norm_eps"])

    # Inkling2MarkovHead: context, step (markov_gate without rounding); fold kept in f64
    def propose(self, anchor, P, last_features, shared, forced=None):
        c = self.c
        r, hdim, keep = c["markov_rank"], c["hidden_size"], c["vocab"]
        head_hidden = self.block_forward(anchor, P, shared)
        hidden = head_hidden[1:]  # sample rows 1..block-1
        f_rnn = self.mix(last_features[None], -1, self.w["markov_head.dflare_norm.weight"])[0]
        head = shared["head"][:keep]
        base = hidden @ head.T
        W = self.w["markov_head.rnn_proj.weight"]
        b = self.w["markov_head.rnn_proj.bias"]
        inputs = torch.cat([hidden, f_rnn.expand_as(hidden)], -1)
        context = b + inputs @ W[:, 2 * r :].T
        fold_w = head @ self.w["markov_head.hidden_out.weight"]
        fold_b = head @ self.w["markov_head.hidden_out.bias"]
        state = torch.zeros(r)
        prev = anchor
        tokens, logits_rows = [], []
        for i in range(c["block_size"] - 1):
            raw = context[i] + torch.cat([state, self.w["markov_head.markov_w1.weight"][prev]]) @ W[:, : 2 * r].T
            gate, cand, out = raw[:r], raw[r : 2 * r], raw[2 * r :]
            g = 1.0 / (1.0 + torch.exp(-gate))
            state = g * state + (1.0 - g) * torch.tanh(cand)
            latent = torch.tanh(out)
            bias = fold_b + latent @ fold_w.T
            logits = c["softcap"] * torch.tanh((base[i] + bias) / c["softcap"])
            logits_rows.append(logits)
            tok = int(torch.argmax(logits))  # first of equal maxima
            tokens.append(tok)
            prev = forced[i] if forced is not None else tok
        return dict(hidden=head_hidden, f_rnn=f_rnn, logits=torch.stack(logits_rows), tokens=tokens)


# ---- tiny fixture ------------------------------------------------------------------------------

def tiny_weights(cfg, gen):
    h, nh, nkv, d, f, r = (cfg[k] for k in ["hidden_size", "num_attention_heads", "num_key_value_heads", "head_dim", "intermediate_size", "markov_rank"])
    V = cfg["padded_vocab"]

    def rnd(*shape, scale):
        return bf16_round(torch.randn(*shape, generator=gen) * scale)

    def normw(n):
        return bf16_round(1.0 + 0.25 * torch.randn(n, generator=gen))

    w = {}
    for i in range(cfg["num_hidden_layers"]):
        p = f"layers.{i}."
        for n in ["dflare_norm", "input_layernorm", "post_attention_layernorm"]:
            w[p + n + ".weight"] = normw(h)
        w[p + "self_attn.q_proj.weight"] = rnd(nh * d, h, scale=h ** -0.5)
        w[p + "self_attn.k_proj.weight"] = rnd(nkv * d, h, scale=h ** -0.5)
        w[p + "self_attn.v_proj.weight"] = rnd(nkv * d, h, scale=h ** -0.5)
        w[p + "self_attn.k_proj_ctx.weight"] = rnd(nkv * d, h, scale=h ** -0.5)
        w[p + "self_attn.v_proj_ctx.weight"] = rnd(nkv * d, h, scale=h ** -0.5)
        w[p + "self_attn.o_proj.weight"] = rnd(h, nh * d, scale=(nh * d) ** -0.5)
        for n in ["q_norm", "k_norm", "k_norm_ctx"]:
            w[p + f"self_attn.{n}.weight"] = normw(d)
        w[p + "mlp.gate_proj.weight"] = rnd(f, h, scale=h ** -0.5)
        w[p + "mlp.up_proj.weight"] = rnd(f, h, scale=h ** -0.5)
        w[p + "mlp.down_proj.weight"] = rnd(h, f, scale=f ** -0.5)
    w["norm.weight"] = normw(h)
    w["mask_embed"] = rnd(h, scale=1.0)
    w["fuse_weight"] = rnd(cfg["num_hidden_layers"] + 1, cfg["taps"], scale=1.0)
    m = "markov_head."
    w[m + "dflare_norm.weight"] = normw(h)
    w[m + "markov_w1.weight"] = rnd(V, r, scale=1.0)
    w[m + "rnn_proj.weight"] = rnd(3 * r, 2 * r + 2 * h, scale=(2 * r + 2 * h) ** -0.5)
    w[m + "rnn_proj.bias"] = rnd(3 * r, scale=0.5)
    w[m + "hidden_out.weight"] = rnd(h, r, scale=r ** -0.5)
    w[m + "hidden_out.bias"] = rnd(h, scale=0.1)
    shared = {
        "embed": rnd(V, h, scale=1.0),
        "embed_norm": normw(h),
        # the trunk's output projection, kept rows only (the padded rows are never computed)
        "head": rnd(cfg["vocab"], h, scale=1.0),
    }
    return w, shared


def generate(out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg = TINY
    gen = torch.Generator().manual_seed(SEED)
    w, shared = tiny_weights(cfg, gen)
    total = max(SCENARIOS)
    taps = torch.randn(total, cfg["taps"] * cfg["hidden_size"], generator=gen) * 2.0
    taps = bf16_round(taps)  # bf16-exact so the Rust side reads the same values from BF16
    anchors = torch.randint(0, cfg["vocab"], (len(SCENARIOS),), generator=gen)
    forced = torch.randint(0, cfg["vocab"], (len(SCENARIOS), cfg["block_size"] - 1), generator=gen)
    tensors = {k: v.to(torch.bfloat16) for k, v in w.items()}
    tensors.update({f"trunk.{k}": v.to(torch.bfloat16) for k, v in shared.items()})
    tensors["taps"] = taps.to(torch.bfloat16)
    tensors["anchors"] = anchors.to(torch.int64)
    tensors["forced"] = forced.to(torch.int64)
    fixture_sha = write_st(out / "inkling2-drafter-tiny.safetensors", tensors)

    oracle = {}
    for s, P in enumerate(SCENARIOS):
        d = Drafter(w, cfg)
        d.store_context(taps[:P], torch.arange(P))
        anchor = int(anchors[s])
        g = d.propose(anchor, P, taps[P - 1], shared)
        f = d.propose(anchor, P, taps[P - 1], shared, forced=[int(t) for t in forced[s]])
        oracle[f"s{s}.hidden"] = g["hidden"]
        oracle[f"s{s}.f_rnn"] = g["f_rnn"]
        oracle[f"s{s}.logits"] = g["logits"]
        oracle[f"s{s}.tokens"] = torch.tensor(g["tokens"], dtype=torch.int64)
        oracle[f"s{s}.forced_logits"] = f["logits"]
        for i in range(cfg["num_hidden_layers"]):
            oracle[f"s{s}.ctx_k.{i}"] = torch.stack(d.ctx_k[i])
            oracle[f"s{s}.ctx_v.{i}"] = torch.stack(d.ctx_v[i])
    oracle_sha = write_st(out / "inkling2-drafter-tiny-oracle.safetensors", oracle)
    run = dict(config=cfg, seed=SEED, scenarios=SCENARIOS, torch=torch.__version__,
               fixture_sha256=fixture_sha, oracle_sha256=oracle_sha,
               greedy=[oracle[f"s{s}.tokens"].tolist() for s in range(len(SCENARIOS))])
    (out / "drafter-tiny-run.json").write_text(json.dumps(run, indent=1) + "\n")
    print(json.dumps(run, indent=1))


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "generate":
        generate(sys.argv[2])
    else:
        raise SystemExit(__doc__)
