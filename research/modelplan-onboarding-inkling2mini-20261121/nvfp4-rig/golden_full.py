#!/usr/bin/env python3
"""Full-tensor proof of the nvfp4-rig overlay: lane F's quantizer (klsens/klh.py nvfp4_weight_quant, unmodified)
on every tensor of the class, on KLH_DEV (cuda or cpu), hashed the way the manifest hashes the overlay planes
(codes, e4m3 scale bytes, then a dense tensor's f32 scale_2 = t, or a bank's f32 amax per expert and f32
denominator 2688), compared with manifest.json. For a bank, fl32(amax / 2688) is lane F's t (checked here too).

usage: KLH_DEV=cuda venv/bin/python golden_full.py OVERLAY_DIR OUT.json
"""
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.expanduser("~/.cache/inkling2mini/klsens"))
import klh  # noqa: E402

overlay_dir, out = sys.argv[1], sys.argv[2]
man = json.load(open(os.path.join(overlay_dir, "manifest.json")))
O = klh.O
st = O.STFile(O.weights_file(klh.CKPT))


def h(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def bf16(name):
    a = np.ascontiguousarray(st.raw(name)).view(np.uint16).reshape(st.info(name)[1])
    return torch.from_numpy((a.astype(np.uint32) << 16).view(np.float32).copy())


rows = []
bad = 0
for t in man["tensors"]:
    t0 = time.time()
    name = t["name"]
    if t["kind"] == "dense":
        W = bf16(name).to(klh.DEV)
        _, codes, sb, ts = klh.nvfp4_weight_quant(W, want_codes=True)
        amax = W.abs().max().reshape(1).float().cpu().numpy()
        del W
    else:
        layer = int(name.split(".")[3])
        bank = name.split(".")[-1].replace("_weight", "")
        store = klh.GpuStore(st, layer)
        cs, ss, ams, tss = [], [], [], []
        for e0 in range(0, store.num_experts, 8):
            W = store.deq(bank, torch.arange(e0, min(store.num_experts, e0 + 8), device=klh.DEV))
            _, c, s, tt = klh.nvfp4_weight_quant(W, per_expert_dim0=True, want_codes=True)
            cs.append(c.cpu()); ss.append(s.cpu()); tss.append(tt.reshape(-1).cpu())
            ams.append(W.abs().amax(dim=(-1, -2)).float().cpu())
            del W
        codes, sb, ts, amax = torch.cat(cs), torch.cat(ss), torch.cat(tss), torch.cat(ams).numpy()
        del store
    t32 = (torch.from_numpy(amax) / torch.tensor(2688.0)).clamp(min=1e-30)
    t_ok = bool((t32.view(torch.int32) == ts.float().cpu().view(torch.int32)).all())
    got = {name: h(codes.cpu().numpy()), name + "_scale": h(sb.cpu().numpy())}
    if t["kind"] == "dense":
        got[name + "_scale_2"] = h(ts.reshape(-1).float().cpu().numpy().astype("<f4"))
    else:
        got[name + "_amax"] = h(amax.astype("<f4"))
        got[name + "_outer_scale_denominator"] = h(np.array([2688.0], dtype="<f4"))
    diff = [k for k in got if got[k] != t["outputs"][k]]
    bad += len(diff) + (0 if t_ok else 1)
    rows.append({"name": name, "planes_differ": diff, "t_equals_amax_over_2688": t_ok, "s": round(time.time() - t0, 1)})
    print(name, "OK" if not diff and t_ok else f"DIFF {diff} t_ok={t_ok}", flush=True)
    if klh.DEV.type == "cuda":
        torch.cuda.empty_cache()
json.dump({"device": str(klh.DEV), "torch": torch.__version__, "overlay_sha256": man["overlay"]["sha256"],
           "tensors": len(rows), "planes_or_scales_differ": bad, "rows": rows}, open(out, "w"), indent=1)
print("tensors", len(rows), "differences", bad)
sys.exit(1 if bad else 0)
