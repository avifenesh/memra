"""Round an f64 oracle tape to f32 (once) for the f32 runner; integer tensors and meta.json copy as is."""
import json, shutil, sys, os
import numpy as np
from safetensors.numpy import load_file, save_file
src, dst = sys.argv[1], sys.argv[2]
os.makedirs(dst, exist_ok=True)
for f in sorted(os.listdir(src)):
    p = os.path.join(src, f)
    if f.endswith(".safetensors"):
        t = load_file(p)
        save_file({k: (v.astype(np.float32) if v.dtype == np.float64 else v) for k, v in t.items()}, os.path.join(dst, f))
    else:
        shutil.copy(p, os.path.join(dst, f))
m = json.load(open(os.path.join(dst, "meta.json")))
m["converted"] = "f64 tape rounded once to f32 by tape_f64_to_f32.py"
json.dump(m, open(os.path.join(dst, "meta.json"), "w"), indent=1)
print("ok", dst)
