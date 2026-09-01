#!/usr/bin/env python3
"""Continued training of the Ornith-1.5-35B-A3B MTP head on own-gen pairs.

The vendor 1-layer MTP head lags the RL'd trunk (measured 53.7% embedded /
48.8% masked acceptance at K=2; owner diagnosis 2026-08-20). This trains
`mtp.*` ONLY — trunk, embeddings and lm_head stay frozen checkpoint bytes —
on teacher-forced (h_{j-1}, t_j) -> t_{j+1} pairs captured from the trunk's
own generations at serving sampling (capture_hiddens.py shards).

Semantics pinned to memra's serve program (crates/memra-engine/src/hybrid.rs +
spec.rs mtp_kv_fill / mtp_head_forward_dev): every draft row p consumes
concat(enorm(emb(t_p)), hnorm(seed))) through fc (= eh_proj), one full-attention
qwen3_5_moe decoder layer, mtp.norm, shared lm_head; rope position p+1; row p
predicts t_{p+1}. Depth changes only the SEED: depth 1 = trunk hidden h_{p-1}
(fill rows + chain step 1), depth d>=2 = the head's OWN pre-mtp.norm output
from row p-1 at depth d-1 (op-10 h_nextn carrier, MEMRA_SPEC_HPOST off).
Training unrolls D depths (hqmtp chain-rollout precedent: teacher-forced
tokens, self-recursive carrier WITH gradient, chain K/V appended): pass d
queries attend depth-1 K/V rows q <= p-d+1 plus the chain band (p-d+k, k) —
the exact serve attention. Loss = mean over depths of response-region CE.

Export re-splits fused experts back to the checkpoint's per-expert `mtp.*`
names so the artifact patches the pinned BF16 checkpoint in place.
"""
import argparse
import copy
import json
import math
import pathlib
import random
import time

import torch
import torch.nn as nn
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeDecoderLayer,
    Qwen3_5MoeRMSNorm,
    Qwen3_5MoeTextRotaryEmbedding,
)


class MiniCache:
    """past-KV shim: this modeling file's attention calls only .update().

    Returns concat(past, new) and records the new (post-rope) K/V so the
    caller can build later depths' contexts. torch.cat keeps gradients — the
    serve chain backprops through carrier AND chain KV rows (hqmtp precedent).
    """

    def __init__(self, k=None, v=None):
        self.k, self.v = k, v
        self.new_k = self.new_v = None

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        self.new_k, self.new_v = key_states, value_states
        if self.k is None:
            return key_states, value_states
        return (torch.cat([self.k, key_states], dim=2),
                torch.cat([self.v, value_states], dim=2))


class MtpHead(nn.Module):
    def __init__(self, text_config):
        super().__init__()
        cfg = copy.deepcopy(text_config)
        cfg.num_hidden_layers = 1
        cfg.layer_types = ["full_attention"]
        cfg._attn_implementation = "sdpa"
        self.cfg = cfg
        hidden, eps = cfg.hidden_size, cfg.rms_norm_eps
        self.pre_fc_norm_embedding = Qwen3_5MoeRMSNorm(hidden, eps)
        self.pre_fc_norm_hidden = Qwen3_5MoeRMSNorm(hidden, eps)
        self.fc = nn.Linear(2 * hidden, hidden, bias=False)
        self.layers = nn.ModuleList([Qwen3_5MoeDecoderLayer(cfg, 0)])
        self.norm = Qwen3_5MoeRMSNorm(hidden, eps)
        self.rotary = Qwen3_5MoeTextRotaryEmbedding(cfg)

    def forward(self, emb, h_prev, position_ids, attn_mask, past=None):
        """One depth pass. Returns (pre-mtp.norm carrier x, this pass's K/V)."""
        x = torch.cat(
            [self.pre_fc_norm_embedding(emb), self.pre_fc_norm_hidden(h_prev)], dim=-1
        )
        x = self.fc(x)
        pos_emb = self.rotary(x, position_ids)
        cache = MiniCache(*(past if past is not None else (None, None)))
        x = self.layers[0](x, position_embeddings=pos_emb, attention_mask=attn_mask,
                           past_key_values=cache)
        return x, (cache.new_k, cache.new_v)


def load_checkpoint_tensors(bf16_dir: pathlib.Path):
    """Returns (mtp state_dict with fused experts, embed bf16, lm_head bf16)."""
    idx = json.loads((bf16_dir / "model.safetensors.index.json").read_text())
    wmap = idx["weight_map"]
    by_shard = {}
    for name, shard in wmap.items():
        if name.startswith("mtp.") or name in (
            "model.language_model.embed_tokens.weight",
            "lm_head.weight",
        ):
            by_shard.setdefault(shard, []).append(name)
    raw = {}
    for shard, names in sorted(by_shard.items()):
        with safe_open(bf16_dir / shard, framework="pt") as f:
            for name in names:
                raw[name] = f.get_tensor(name)
    embed = raw.pop("model.language_model.embed_tokens.weight")
    lm_head = raw.pop("lm_head.weight")

    sd = {}
    gate, up, down = {}, {}, {}
    for name, t in raw.items():
        key = name[len("mtp."):]
        if ".mlp.experts." in key:
            eid = int(key.split(".mlp.experts.")[1].split(".")[0])
            if key.endswith("gate_proj.weight"):
                gate[eid] = t
            elif key.endswith("up_proj.weight"):
                up[eid] = t
            elif key.endswith("down_proj.weight"):
                down[eid] = t
            else:
                raise KeyError(f"unexpected expert tensor {name}")
        else:
            sd[key] = t
    n_exp = len(down)
    assert len(gate) == len(up) == n_exp and n_exp > 0, (len(gate), len(up), n_exp)
    sd["layers.0.mlp.experts.gate_up_proj"] = torch.stack(
        [torch.cat([gate[e], up[e]], dim=0) for e in range(n_exp)]
    )
    sd["layers.0.mlp.experts.down_proj"] = torch.stack([down[e] for e in range(n_exp)])
    return sd, embed, lm_head


def export_mtp(module: MtpHead, out_path: pathlib.Path):
    """Re-split fused experts back to the checkpoint's per-expert mtp.* names."""
    sd = module.state_dict()
    out = {}
    for key, t in sd.items():
        if key.startswith("rotary."):
            continue
        if key == "layers.0.mlp.experts.gate_up_proj":
            inter = t.shape[1] // 2
            for e in range(t.shape[0]):
                out[f"mtp.layers.0.mlp.experts.{e}.gate_proj.weight"] = (
                    t[e, :inter].contiguous().to(torch.bfloat16)
                )
                out[f"mtp.layers.0.mlp.experts.{e}.up_proj.weight"] = (
                    t[e, inter:].contiguous().to(torch.bfloat16)
                )
        elif key == "layers.0.mlp.experts.down_proj":
            for e in range(t.shape[0]):
                out[f"mtp.layers.0.mlp.experts.{e}.down_proj.weight"] = (
                    t[e].contiguous().to(torch.bfloat16)
                )
        else:
            out["mtp." + key] = t.contiguous().to(torch.bfloat16)
    save_file(out, str(out_path))
    print(f"exported {len(out)} tensors -> {out_path}", flush=True)


def load_rows(hiddens_dir: pathlib.Path, corpus: pathlib.Path):
    meta = {}
    for line in open(corpus, encoding="utf-8"):
        try:
            r = json.loads(line)
            meta[str(r["id"])] = (r["split"], r.get("mode", "?"))
        except (json.JSONDecodeError, KeyError):
            continue
    train, heldout = [], []
    for shard in sorted(hiddens_dir.glob("shard-*.pt")):
        for rec in torch.load(shard, map_location="cpu", weights_only=False):
            split, mode = meta.get(str(rec["id"]), ("train", "?"))
            rec["mode"] = mode
            (heldout if split == "heldout" else train).append(rec)
    return train, heldout


def make_batches(rows, batch_tokens, shuffle, seed=0):
    order = list(range(len(rows)))
    order.sort(key=lambda i: len(rows[i]["ids"]))
    batches, cur, cur_max = [], [], 0
    for i in order:
        L = len(rows[i]["ids"])
        m = max(cur_max, L)
        if cur and m * (len(cur) + 1) > batch_tokens:
            batches.append(cur)
            cur, cur_max = [], 0
            m = L
        cur.append(i)
        cur_max = m
    if cur:
        batches.append(cur)
    if shuffle:
        random.Random(seed).shuffle(batches)
    return batches


def ce_over(module, lm_head, x, labels, ce_chunk=2048):
    """CE + top1 of norm(x) @ lm_head against labels (-100 ignored), chunked."""
    flat = module.norm(x).reshape(-1, x.shape[-1])
    flat_labels = labels.reshape(-1)
    keep = flat_labels != -100
    flat, flat_labels = flat[keep], flat_labels[keep]
    loss_sum = x.new_zeros((), dtype=torch.float32)
    top1 = 0
    for c0 in range(0, flat.shape[0], ce_chunk):
        piece = flat[c0 : c0 + ce_chunk]
        lab = flat_labels[c0 : c0 + ce_chunk]
        logits = piece.to(torch.bfloat16) @ lm_head.T
        loss_sum = loss_sum + nn.functional.cross_entropy(
            logits.float(), lab, reduction="sum"
        )
        with torch.no_grad():
            top1 += (logits.argmax(-1) == lab).sum().item()
    return loss_sum, int(flat_labels.shape[0]), top1


def forward_batch(module, embed, lm_head, rows, idxs, device, depths=3):
    """Multi-depth unrolled forward. Returns (weighted loss_sum, per-depth stats).

    Row index r (0-based) = serve row p = r+1: token t_p = ids[r+1], label
    t_{p+1} = ids[r+2], rope p+1 = r+2. Depth-1 seed = trunk h[r]; depth-d
    seed = previous depth's carrier at row r-1. Depth-d rows attend depth-1
    K/V at q <= p-d+1 and the chain band (p-d+k, k) k=2..d — serve-exact.
    """
    seqs = [rows[i] for i in idxs]
    R = max(len(s["ids"]) - 2 for s in seqs)
    B = len(seqs)
    tok_in = torch.zeros(B, R, dtype=torch.long)
    labels = torch.full((B, R), -100, dtype=torch.long)
    h_trunk = torch.zeros(B, R, embed.shape[1], dtype=torch.bfloat16)
    lens = []
    for b, s in enumerate(seqs):
        ids = s["ids"].long()
        L = len(ids)
        n = L - 2
        tok_in[b, :n] = ids[1 : L - 1]
        lab = ids[2:].clone()
        first = max(0, s["prompt_len"] - 2)  # label t_{p+1} in response <=> r >= P-2
        lab[:first] = -100
        labels[b, :n] = lab
        h_trunk[b, :n] = s["h"][: n]
        lens.append(n)
    tok_in, labels, h_trunk = tok_in.to(device), labels.to(device), h_trunk.to(device)
    lens_t = torch.tensor(lens, device=device)

    q = torch.arange(R, device=device)
    key_real = q[None, None, None, :] < lens_t[:, None, None, None]
    NEG = torch.finfo(torch.bfloat16).min

    def to_additive(allowed):
        m = torch.zeros(allowed.shape, device=device, dtype=torch.bfloat16)
        m.masked_fill_(~allowed, NEG)
        return m

    position_ids = (q[None, :] + 2).expand(B, R)
    emb = embed[tok_in].to(torch.bfloat16)
    diag = (q[None, None, :, None] == q[None, None, None, :])

    stats = []
    total = None
    seed = h_trunk
    kv_depth1 = None
    kv_prev = None
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for d in range(1, depths + 1):
            if d == 1:
                allowed = (q[None, None, :, None] >= q[None, None, None, :]) & key_real
                x, kv = module(emb, seed, position_ids, to_additive(allowed))
                kv_depth1 = kv
            else:
                # past = depth-1 K/V (+ previous-depth K/V for d>=3), self = diag only
                # depth-1 part: q <= p-d+1  <=> key_r <= query_r - (d-1)
                a1 = (q[None, None, None, :] <= q[None, None, :, None] - (d - 1)) & key_real
                parts = [a1]
                past_k = [kv_depth1[0]]
                past_v = [kv_depth1[1]]
                if d >= 3:
                    # chain band from the previous depth: key row == query row - 1
                    a_prev = (q[None, None, None, :] == q[None, None, :, None] - 1) & key_real
                    parts.append(a_prev)
                    past_k.append(kv_prev[0])
                    past_v.append(kv_prev[1])
                parts.append(diag & key_real)  # own row
                allowed = torch.cat([p.expand(B, 1, R, R) for p in parts], dim=-1)
                past = (torch.cat(past_k, dim=2), torch.cat(past_v, dim=2))
                # seed = previous depth's carrier shifted right one row
                seed = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)
                x, kv = module(emb, seed, position_ids, to_additive(allowed), past=past)
            kv_prev = kv
            lab_d = labels.clone()
            if d > 1:
                lab_d[:, : d - 1] = -100  # row p needs a round started at p-d >= 0
            loss_sum, n, top1 = ce_over(module, lm_head, x, lab_d)
            stats.append({"depth": d, "loss_sum": loss_sum, "n": n, "top1": top1})
            total = loss_sum / max(n, 1) if total is None else total + loss_sum / max(n, 1)
    return total / depths, stats


def evaluate(module, embed, lm_head, rows, batch_tokens, device, depths=3):
    module.eval()
    agg = {}  # (mode, depth) -> [loss_sum, n, top1]
    with torch.no_grad():
        for mode in sorted({r["mode"] for r in rows}):
            sub = [r for r in rows if r["mode"] == mode]
            for idxs in make_batches(sub, batch_tokens, shuffle=False):
                _, stats = forward_batch(module, embed, lm_head, sub, idxs, device, depths)
                for st in stats:
                    a = agg.setdefault((mode, st["depth"]), [0.0, 0, 0])
                    a[0] += st["loss_sum"].item(); a[1] += st["n"]; a[2] += st["top1"]
    module.train()
    by_depth = {}
    for d in range(1, depths + 1):
        tot = [sum(a[i] for (m, dd), a in agg.items() if dd == d) for i in range(3)]
        by_depth[d] = {"loss": round(tot[0] / max(tot[1], 1), 4),
                       "top1": round(tot[2] / max(tot[1], 1), 4), "n": tot[1]}
    by_mode_d1 = {m: {"loss": round(a[0] / max(a[1], 1), 4),
                      "top1": round(a[2] / max(a[1], 1), 4), "n": a[1]}
                  for (m, dd), a in agg.items() if dd == 1}
    return by_depth, by_mode_d1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bf16-dir", required=True, type=pathlib.Path)
    ap.add_argument("--hiddens-dir", required=True, type=pathlib.Path)
    ap.add_argument("--corpus", required=True, type=pathlib.Path)
    ap.add_argument("--out-dir", required=True, type=pathlib.Path)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--depths", type=int, default=3)
    ap.add_argument("--batch-tokens", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--min-lr", type=float, default=1e-5)
    ap.add_argument("--warmup-frac", type=float, default=0.05)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=20260820)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    metrics = open(args.out_dir / "metrics.jsonl", "a", encoding="utf-8")

    cfg = AutoConfig.from_pretrained(args.bf16_dir)
    text_cfg = cfg.text_config if hasattr(cfg, "text_config") else cfg
    print("loading checkpoint tensors...", flush=True)
    sd, embed, lm_head = load_checkpoint_tensors(args.bf16_dir)
    module = MtpHead(text_cfg)
    missing, unexpected = module.load_state_dict(sd, strict=False)
    missing = [m for m in missing if not m.startswith("rotary.")]
    assert not missing and not unexpected, (missing, unexpected)
    module = module.to(device=device, dtype=torch.float32)
    module.train()
    embed = embed.to(device=device, dtype=torch.bfloat16)
    lm_head = lm_head.to(device=device, dtype=torch.bfloat16)
    embed.requires_grad_(False)
    lm_head.requires_grad_(False)
    n_params = sum(p.numel() for p in module.parameters())
    print(f"MTP head params: {n_params/1e6:.1f}M (all trainable)", flush=True)

    print("loading capture shards...", flush=True)
    train_rows, heldout_rows = load_rows(args.hiddens_dir, args.corpus)
    print(f"train seqs={len(train_rows)} heldout seqs={len(heldout_rows)}", flush=True)

    opt = torch.optim.AdamW(module.parameters(), lr=args.lr, weight_decay=0.0)
    n_steps = args.epochs * len(make_batches(train_rows, args.batch_tokens, False))
    warmup = max(1, int(n_steps * args.warmup_frac))

    def lr_at(step):
        if step < warmup:
            return args.lr * step / warmup
        p = (step - warmup) / max(1, n_steps - warmup)
        return args.min_lr + 0.5 * (args.lr - args.min_lr) * (1 + math.cos(math.pi * p))

    by_depth, by_mode = evaluate(module, embed, lm_head, heldout_rows, args.batch_tokens, device, args.depths)
    print(f"[baseline vendor head] heldout by_depth={by_depth} d1_by_mode={by_mode}", flush=True)
    metrics.write(json.dumps({"event": "eval", "step": 0, "by_depth": by_depth, "d1_by_mode": by_mode}) + "\n")
    metrics.flush()

    step, t0 = 0, time.time()
    for epoch in range(args.epochs):
        batches = make_batches(train_rows, args.batch_tokens, shuffle=True, seed=args.seed + epoch)
        for idxs in batches:
            step += 1
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            loss, stats = forward_batch(module, embed, lm_head, train_rows, idxs, device, args.depths)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(module.parameters(), args.clip)
            opt.step()
            if step % 10 == 0:
                rec = {"event": "train", "step": step, "epoch": epoch,
                       "loss": loss.item(), "lr": lr_at(step), "gnorm": float(gnorm),
                       "elapsed_s": round(time.time() - t0, 1)}
                for st in stats:
                    rec[f"d{st['depth']}_top1"] = round(st["top1"] / max(st["n"], 1), 4)
                    rec[f"d{st['depth']}_loss"] = round(st["loss_sum"].item() / max(st["n"], 1), 4)
                metrics.write(json.dumps(rec) + "\n")
                metrics.flush()
                tops = " ".join(f"d{st['depth']}={st['top1']/max(st['n'],1):.4f}" for st in stats)
                print(f"step {step}/{n_steps} loss={loss.item():.4f} top1 {tops} lr={lr_at(step):.2e}", flush=True)
        by_depth, by_mode = evaluate(module, embed, lm_head, heldout_rows, args.batch_tokens, device, args.depths)
        print(f"[epoch {epoch}] heldout by_depth={by_depth}", flush=True)
        metrics.write(json.dumps({"event": "eval", "step": step, "epoch": epoch,
                                  "by_depth": by_depth, "d1_by_mode": by_mode}) + "\n")
        metrics.flush()
        export_mtp(module, args.out_dir / f"mtp-trained-epoch{epoch}.safetensors")
    print("TRAIN DONE", flush=True)


if __name__ == "__main__":
    main()
