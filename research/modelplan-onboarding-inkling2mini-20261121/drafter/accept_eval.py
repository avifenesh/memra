#!/usr/bin/env python3
"""Lane S: teacher-forced drafter acceptance on CPU, two numeric arms of the drafter.

Arms (the drafter only; the target sequence and taps come from elsewhere):
  f64      the exact program over the stored BF16 weights (the torch port of dspark_oracle.py,
           with the fold head @ hidden_out cached once instead of recomputed per call)
  bf16pts  the vLLM image's serving numerics: fp32 math with a bf16 round (RNE) at every point
           the image rounds (taps stored bf16; every linear output; each norm output; q/k norm
           then RoPE; silu*up; the residual stream; base logits; the folded bias and fold weight
           in bf16; softcap of bf16(base + bias)); rnn context and raw in fp32 (addmm out fp32);
           the gate rounds each op as ops/dspark.py _markov_gate_kernel.

usage:
  accept_eval.py selfcheck FIXTURE ORACLE           f64 arm == dspark_oracle.py on the tiny fixture
  accept_eval.py run CKPT_DIR PP_DIR OUT.jsonl [--taps gpu|oracle] [--arms f64,bf16pts]
        PP_DIR holds <id>.json (seq, argmax, proposals) and <id>.taps.f32 from ink2-spec --per-position,
        and, for --taps oracle, <id>.oracle.taps.f32 / <id>.oracle.json from oracle_taps.py.
"""
import json
import struct
import sys
import time
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(int(__import__("os").environ.get("THREADS", "16")))


def bf(t):
    """round to bf16 (RNE), kept in the tensor's float dtype"""
    return t.to(torch.bfloat16).to(t.dtype)


class St:
    def __init__(self, path):
        self.path = path
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            self.h = json.loads(f.read(n))
        self.base = 8 + n
        self.mm = np.memmap(path, dtype=np.uint8, mode="r")

    def t(self, name):
        e = self.h[name]
        a, b = e["data_offsets"]
        raw = np.asarray(self.mm[self.base + a:self.base + b])
        if e["dtype"] == "BF16":
            u = raw.view(np.uint16).astype(np.uint32) << 16
            return torch.from_numpy(u.view(np.float32).reshape(e["shape"]).copy())
        if e["dtype"] == "F32":
            return torch.from_numpy(raw.view(np.float32).reshape(e["shape"]).copy())
        if e["dtype"] == "F64":
            return torch.from_numpy(raw.view(np.float64).reshape(e["shape"]).copy())
        if e["dtype"] == "I64":
            return torch.from_numpy(raw.view(np.int64).reshape(e["shape"]).copy())
        raise SystemExit(f"{name}: {e['dtype']}")


def rope(x, pos, theta, dt):
    d = x.shape[-1]
    inv = theta ** (-torch.arange(0, d, 2, dtype=torch.float64) / d)
    ang = pos.to(torch.float64)[:, None] * inv[None, :]
    if dt == torch.float32:
        # the image's cos/sin cache is fp32 from an fp32 inv_freq and fp32 positions
        inv32 = 1.0 / (theta ** (torch.arange(0, d, 2, dtype=torch.float32) / d))
        ang = (pos.to(torch.float32)[:, None] * inv32[None, :])
    cos, sin = ang.cos().to(dt)[:, None, :], ang.sin().to(dt)[:, None, :]
    x1, x2 = x[..., : d // 2], x[..., d // 2:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], -1)


class Eval:
    def __init__(self, w, shared, cfg, arm):
        self.c, self.arm = cfg, arm
        self.dt = torch.float64 if arm == "f64" else torch.float32
        dt = self.dt
        self.w = {k: v.to(dt) for k, v in w.items()}
        self.embed = shared["embed"]  # f32 (bf16 values), indexed per anchor
        self.embed_norm = shared["embed_norm"].to(dt)
        self.head = shared["head"].to(dt)
        self.alpha = torch.softmax(w["fuse_weight"].to(torch.float64), -1).to(dt)
        how, hob = w["markov_head.hidden_out.weight"].to(torch.float64), w["markov_head.hidden_out.bias"].to(torch.float64)
        fw, fb = [], []
        for s in range(0, shared["head"].shape[0], 8192):
            hd = shared["head"][s:s + 8192].to(torch.float64)
            fw.append(hd @ how)
            fb.append(hd @ hob)
        self.fold_w, self.fold_b = torch.cat(fw).to(dt), torch.cat(fb).to(dt)
        if arm == "bf16pts":
            self.fold_w, self.fold_b = bf(self.fold_w), bf(self.fold_b)
        self.ctx_k = [[] for _ in range(cfg["layers"])]
        self.ctx_v = [[] for _ in range(cfg["layers"])]

    def r(self, x):
        return bf(x) if self.arm == "bf16pts" else x

    def L(self, i, n):
        return self.w[f"layers.{i}.{n}"]

    def rms(self, x, w):
        x2 = x.to(torch.float64) if self.arm == "f64" else x
        return x2 * torch.rsqrt((x2 * x2).mean(-1, keepdim=True) + self.c["eps"]) * w

    def mix(self, taps, row, norm_w):
        # taps [n, T, H]
        m = (taps * self.alpha[row][None, :, None]).sum(1)
        return self.r(self.rms(m, norm_w))

    def commit(self, taps):
        """taps [n, T, H] for the next n positions"""
        c = self.c
        taps = self.r(taps.to(self.dt))
        p0 = len(self.ctx_k[0])
        pos = torch.arange(p0, p0 + taps.shape[0])
        for i in range(c["layers"]):
            ctx = self.mix(taps, i, self.L(i, "dflare_norm.weight"))
            k = self.r(ctx @ self.L(i, "self_attn.k_proj_ctx.weight").T)
            v = self.r(ctx @ self.L(i, "self_attn.v_proj_ctx.weight").T)
            k = self.r(self.rms(k.reshape(-1, c["kvh"], c["d"]), self.L(i, "self_attn.k_norm_ctx.weight")))
            k = self.r(rope(k, pos, c["theta"], self.dt))
            self.ctx_k[i].extend(list(k))
            self.ctx_v[i].extend(list(v.view(-1, c["kvh"], c["d"])))

    def attn(self, i, P, x):
        c = self.c
        nh, nkv, d, B = c["heads"], c["kvh"], c["d"], c["rows"]
        pos = torch.arange(P, P + B)
        q = self.r(x @ self.L(i, "self_attn.q_proj.weight").T).view(B, nh, d)
        k = self.r(x @ self.L(i, "self_attn.k_proj.weight").T).view(B, nkv, d)
        v = self.r(x @ self.L(i, "self_attn.v_proj.weight").T).view(B, nkv, d)
        q = self.r(rope(self.r(self.rms(q, self.L(i, "self_attn.q_norm.weight"))), pos, c["theta"], self.dt))
        k = self.r(rope(self.r(self.rms(k, self.L(i, "self_attn.k_norm.weight"))), pos, c["theta"], self.dt))
        first = max(0, P - c["window"]) if i in c["local"] else 0
        ck = torch.stack(self.ctx_k[i][first:P]) if P > first else k[:0]
        cv = torch.stack(self.ctx_v[i][first:P]) if P > first else v[:0]
        keys, vals = torch.cat([ck, k]), torch.cat([cv, v])
        g = nh // nkv
        kk = keys.repeat_interleave(g, dim=1)  # [n, nh, d]
        vv = vals.repeat_interleave(g, dim=1)
        s = torch.einsum("bhd,nhd->hbn", q, kk) * d ** -0.5
        p = torch.softmax(s, -1)
        o = self.r(torch.einsum("hbn,nhd->bhd", p, vv).reshape(B, nh * d))
        return self.r(o @ self.L(i, "self_attn.o_proj.weight").T)

    def propose(self, anchor, feature_taps, k=7, forced=None, keep=False):
        """Greedy proposals (the next one conditions on the last), or on `forced`; with `keep`
        also every step's softcapped logits [k, V]."""
        c = self.c
        B, r = c["rows"], c["rank"]
        P = len(self.ctx_k[0])
        x = self.w["mask_embed"].expand(B, -1).clone()
        x[0] = self.r(self.rms(self.embed[anchor].to(self.dt), self.embed_norm))
        res = x
        h = self.r(self.rms(res, self.L(0, "input_layernorm.weight")))
        for i in range(c["layers"]):
            if i > 0:
                s = res + mlp
                res = self.r(s)
                h = self.r(self.rms(s, self.L(i, "input_layernorm.weight")))
            o = self.attn(i, P, h)
            s = res + o
            res = self.r(s)
            h = self.r(self.rms(s, self.L(i, "post_attention_layernorm.weight")))
            gate = self.r(h @ self.L(i, "mlp.gate_proj.weight").T)
            up = self.r(h @ self.L(i, "mlp.up_proj.weight").T)
            act = self.r(gate / (1 + torch.exp(-gate)) * up)
            mlp = self.r(act @ self.L(i, "mlp.down_proj.weight").T)
        hidden = self.r(self.rms(res + mlp, self.w["norm.weight"]))[1:]
        f = self.mix(self.r(feature_taps.to(self.dt))[None], c["layers"], self.w["markov_head.dflare_norm.weight"])[0]
        base = self.r(hidden @ self.head.T)
        W, b = self.w["markov_head.rnn_proj.weight"], self.w["markov_head.rnn_proj.bias"]
        ctx = b + torch.cat([hidden, f.expand_as(hidden)], -1) @ W[:, 2 * r:].T
        state = torch.zeros(r, dtype=self.dt)
        prev = anchor
        out = []
        kept = []
        for i in range(k):
            raw = ctx[i] + torch.cat([state, self.w["markov_head.markov_w1.weight"][prev]]) @ W[:, :2 * r].T
            if self.arm == "bf16pts":
                raw = bf(raw)
                g_ = bf(1.0 / (1.0 + torch.exp(-raw[:r])))
                cand = bf(torch.tanh(raw[r:2 * r]))
                state = bf(bf(g_ * state) + bf(bf(1.0 - g_) * cand))
                latent = bf(torch.tanh(raw[2 * r:]))
                bias = bf(self.fold_b + latent @ self.fold_w.T)
                logits = bf(c["cap"] * torch.tanh(bf(base[i] + bias) / c["cap"]))
            else:
                g_ = 1.0 / (1.0 + torch.exp(-raw[:r]))
                state = g_ * state + (1.0 - g_) * torch.tanh(raw[r:2 * r])
                latent = torch.tanh(raw[2 * r:])
                bias = self.fold_b + latent @ self.fold_w.T
                logits = c["cap"] * torch.tanh((base[i] + bias) / c["cap"])
            tok = int(torch.argmax(logits))
            out.append(tok)
            if keep:
                kept.append(logits)
            prev = forced[i] if forced is not None else tok
        return out, (torch.stack(kept) if keep else logits)


def load_drafter(path):
    st = St(path)
    return {k: st.t(k) for k in st.h if k != "__metadata__"}


REAL = dict(layers=3, heads=24, kvh=8, d=128, rows=8, rank=256, local=[0, 2], window=1024, theta=1e6,
            eps=1e-6, cap=30.0)


def selfcheck(fixture, oracle):
    sys.path.insert(0, str(Path(__file__).parent))
    st, orc = St(fixture), St(oracle)
    w = {k: st.t(k) for k in st.h if k != "__metadata__" and not k.startswith("trunk.") and k not in ("taps", "anchors", "forced")}
    shared = {"embed": st.t("trunk.embed"), "embed_norm": st.t("trunk.embed_norm"), "head": st.t("trunk.head")}
    cfg = dict(layers=3, heads=4, kvh=2, d=16, rows=8, rank=8, local=[0, 2], window=5, theta=1e6, eps=1e-6, cap=30.0)
    taps = st.t("taps").to(torch.float64).view(13, 4, 64)
    for s, P in enumerate([1, 4, 13]):
        e = Eval(w, shared, cfg, "f64")
        e.commit(taps[:P])
        toks, _ = e.propose(int(st.t("anchors")[s]), taps[P - 1])
        want = orc.t(f"s{s}.tokens").tolist()
        assert toks == want, (s, toks, want)
        b = Eval(w, shared, cfg, "bf16pts")
        b.commit(taps[:P])
        tb, _ = b.propose(int(st.t("anchors")[s]), taps[P - 1])
        print(f"scenario {s}: f64 {toks} == oracle; bf16pts {tb}")
    print("SELFCHECK_OK")


def run(ckpt, pp_dir, out, taps_src, arms):
    pp = Path(pp_dir)
    w = load_drafter(Path(ckpt) / "mtp.safetensors")
    main = St(str(Path(ckpt) / "model-00001-of-00001.safetensors"))
    vocab = 200058
    shared = {"embed": main.t("model.llm.embed.weight"), "embed_norm": main.t("model.llm.embed_norm.weight"),
              "head": main.t("model.llm.unembed.weight")[:vocab]}
    H = 2304
    with open(out, "a") as fo:
        for jf in sorted(pp.glob("*.json")):
            if jf.name.endswith(".oracle.json"):
                continue
            meta = json.load(open(jf))
            pid, seq, T0 = meta["id"], meta["seq"], meta["first_anchor"]
            tap_file = pp / (f"{pid}.taps.f32" if taps_src == "gpu" else f"{pid}.oracle.taps.f32")
            taps = torch.from_numpy(np.fromfile(tap_file, dtype=np.float32).reshape(len(seq), 4, H).copy())
            for arm in arms:
                t0 = time.time()
                e = Eval(w, shared, REAL, arm)
                e.commit(taps[:T0])
                props = []
                for P in range(T0, len(seq)):
                    toks, _ = e.propose(seq[P], taps[P - 1])
                    props.append(toks)
                    e.commit(taps[P:P + 1])
                fo.write(json.dumps({"id": pid, "kind": meta["kind"], "taps": taps_src, "arm": arm,
                                     "first_anchor": T0, "proposals": props, "seconds": time.time() - t0}) + "\n")
                fo.flush()
                print(pid, taps_src, arm, f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "selfcheck":
        selfcheck(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == "run":
        a = sys.argv
        taps_src = a[a.index("--taps") + 1] if "--taps" in a else "gpu"
        arms = a[a.index("--arms") + 1].split(",") if "--arms" in a else ["f64", "bf16pts"]
        run(a[2], a[3], a[4], taps_src, arms)
