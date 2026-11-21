#!/usr/bin/env python3
"""KL of memra's GPU route against lane F's reference, with lane F's metric (lane M, nvfp4-rig gate).

memra side: `ink2-gpu CKPT --class C --corpus klsens/corpus/SET.json --hidden-out H.safetensors` writes the
final-norm hidden rows of every position. Reference side: lane F's ref hiddens (klsens/results/ref/SET/NAME.npz,
the checkpoint's most precise program: shipped weights dequantized exactly, f32 activations, f32 KV).
Head: lane F's (klh.Harness.logits): cap * tanh(h @ W.T / cap) over the full padded vocabulary, log-softmax in f64;
the reference uses the BF16 unembed, the arm the unembed its class reads (BF16 for shipped; for nvfp4-rig the
overlay's minted NVFP4 unembed, dequantized fl32(code * fl32(e4m3(scale) * scale_2)), the values lane F's arm
used). Per position: KL(ref || arm), top-1 agreement, the NLL of the next token under both. Statistics: lane F's
analyze.py stats() (lanef_stats.py): real-text positions (next token not sampled), sequence bootstrap.

usage: venv/bin/python score_kl.py SET HIDDEN.safetensors CLASS OVERLAY_DIR OUT.json
"""
import json
import os
import sys

import numpy as np
import torch
from safetensors import safe_open

K = os.path.expanduser("~/.cache/inkling2mini/klsens")
CKPT = os.path.expanduser("~/models/inkling2-mini")
SET, HID, CLASS, OVL, OUT = sys.argv[1:6]
DEV = torch.device("cuda")
CAP = 30.0
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lanef_stats import stats  # noqa: E402


E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])


def e4m3_pos():
    v = np.zeros(256, dtype=np.float32)
    for b in range(127):
        e, m = b >> 3, b & 7
        v[b] = m * 2.0 ** -9 if e == 0 else (1 + m / 8) * 2.0 ** (e - 7)
    return torch.from_numpy(v)


def unembed_bf16():
    with safe_open(os.path.join(CKPT, "model-00001-of-00001.safetensors"), framework="pt") as f:
        return f.get_tensor("model.llm.unembed.weight").to(DEV).float()


def unembed_minted():
    with safe_open(os.path.join(OVL, "overlay.safetensors"), framework="pt") as f:
        n = "model.llm.unembed.weight"
        codes = f.get_tensor(n).to(DEV)
        sc = f.get_tensor(n + "_scale").view(torch.uint8).to(DEV)
        t = f.get_tensor(n + "_scale_2").float().reshape(()).to(DEV)
    rows = codes.shape[0]
    vals = torch.stack([E2M1.to(DEV)[(codes & 15).long()], E2M1.to(DEV)[(codes >> 4).long()]], -1).reshape(rows, -1)
    st = (e4m3_pos().to(DEV)[sc.long()] * t)  # fl32(s * t)
    return (vals.view(rows, -1, 16) * st[..., None]).reshape(rows, -1)  # fl32(code * fl32(s * t))


corpus = json.load(open(os.path.join(K, "corpus", SET + ".json")))
Wr = unembed_bf16()
Wa = Wr if CLASS == "shipped" else unembed_minted()
per = {k: [] for k in ("kl", "top1", "nll_arm", "nll_ref", "has_label", "sampled", "seq")}
with safe_open(HID, framework="pt") as hf:
    for i, c in enumerate(corpus):
        ref = np.load(os.path.join(K, "results", "ref", SET, c["name"] + ".npz"))
        assert list(ref["ids"]) == list(c["ids"]), c["name"]
        hr = torch.from_numpy(ref["hidden"]).to(DEV)
        ha = hf.get_tensor(c["name"]).to(DEV)
        assert ha.shape == hr.shape, (c["name"], ha.shape, hr.shape)
        T = hr.shape[0]
        lab = torch.tensor(c["ids"][1:] + [0], device=DEV)
        for s in range(0, T, 192):
            la = (CAP * torch.tanh((ha[s:s + 192] @ Wa.T) / CAP)).double().log_softmax(-1)
            lr = (CAP * torch.tanh((hr[s:s + 192] @ Wr.T) / CAP)).double().log_softmax(-1)
            per["kl"].append((lr.exp() * (lr - la)).sum(-1).float().cpu().numpy())
            per["top1"].append((la.argmax(-1) == lr.argmax(-1)).cpu().numpy())
            g = lab[s:s + 192, None]
            per["nll_arm"].append((-la.gather(1, g)[:, 0]).float().cpu().numpy())
            per["nll_ref"].append((-lr.gather(1, g)[:, 0]).float().cpu().numpy())
        hl = np.ones(T, dtype=bool); hl[-1] = False
        per["has_label"].append(hl)
        per["sampled"].append(np.array(c["sampled"][1:] + [0], dtype=bool))
        per["seq"].append(np.full(T, i, dtype=np.int16))
arr = {k: np.concatenate(v) for k, v in per.items()}
np.savez_compressed(OUT[:-5] + ".npz", **arr)
res = {"set": SET, "class": CLASS, "hidden": HID, "sequences": len(corpus), "stats": stats(arr)}
json.dump(res, open(OUT, "w"), indent=1)
s = res["stats"]
print(f"{SET} {CLASS}: KLall {s['kl_all']['mean']:.4f} KLreal {s['kl_real']['mean']:.4f} {s['kl_real']['ci95']} p99 {s['kl_real']['p99']:.3f} "
      f"top1 {s['top1_real']:.3f} dNLLreal {s['dnll_real']['mean']:+.4f} {s['dnll_real']['ci95']}")
