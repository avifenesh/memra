#!/usr/bin/env python3
"""The tiny checkpoint's hf_quant_config.json: the real file's groups over the tiny config's tensor names.

Every tensor of the tiny census (one name per line, expert planes excluded) goes to the group that holds its name
class in the real file (layer and block indices normalized). Expert banks go to the NVFP4 or the MXFP8 group by the
layer list given.

  python3 tiny_quant_config.py <real hf_quant_config.json> <tiny census names> <mxfp8 layers, e.g. 6,7> > tiny-hf_quant_config.json
"""
import json
import re
import sys


def norm(name):
    return re.sub(r"\.\d+(\.|$)", r".N\1", name)


def main():
    real = json.load(open(sys.argv[1]))
    names = [line.split("\t")[0] for line in open(sys.argv[2]).read().splitlines() if line]
    mxfp8 = {int(x) for x in sys.argv[3].split(",")}
    planes = ("_scale", "_amax", "_outer_scale_denominator")
    out = {"quant_method": real["quant_method"]}
    for part, prefix in (("model", ""), ("mtp", "mtp.")):
        groups = [dict(g, tensors=[]) for g in real[part]["parameter_groups"]]
        classes = [{norm(t) for t in g["tensors"]} for g in real[part]["parameter_groups"]]
        for name in names:
            if name.startswith("mtp.") != (part == "mtp") or name.endswith(planes):
                continue
            tensor = name[len(prefix):]
            hits = [i for i, c in enumerate(classes) if norm(tensor) in c]
            if ".mlp.experts." in tensor:
                want = "mxfp8" if int(tensor.split(".")[3]) in mxfp8 else "nvfp4"
                hits = [i for i in hits if groups[i]["compute_dtype"] == want]
            assert len(hits) == 1, (name, hits)
            groups[hits[0]]["tensors"].append(tensor)
        for g in groups:
            g["tensors"].sort()
        out[part] = {"parameter_groups": groups}
    json.dump(out, sys.stdout, indent=1)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
