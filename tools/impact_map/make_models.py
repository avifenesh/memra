#!/usr/bin/env python3
"""docs/impact-map/models.json from the packs scout table, the release roster and harvested receipts.

A model counts as available on a device class when the family table measured it there, or when a
receipt with an explicit commit names the model and the device (gb10 receipts live on the open
GB10 branches). Usage: make_models.py SCOUTS.json RAW_LEDGER.jsonl ROSTER.tsv > models.json
"""
import json
import sys

scouts = json.load(open(sys.argv[1]))
packs = scouts.get("scout:packs") or scouts.get("packs")
roster = {}
for line in open(sys.argv[3]):
    if line.strip() and not line.startswith("#"):
        f = line.split("\t")
        roster[f[1]] = f[0]
ALIAS = {"qwen3.8-27b": "qwen/qwen3.8-27b", "ornith-1.5-35b-a3b": "ornith-ai/ornith-1.5-35b-a3b",
         "qwen3.5-9b": "Qwen3.5-9B", "qwen3.6-35b-a3b": "Qwen3.6-35B-A3B", "glm-5": "zai-org/GLM-5.3-Flash@04c4e9e9",
         "deepseek-v4": "Avifenesh/DeepSeek-V4-Flash-0731-NVFP4@bafd09f8", "step-3.x": "Step-3.7-Flash 196B-A11B",
         "gemma-4": "Gemma-4 31B"}
extra_dev = {}
for line in open(sys.argv[2]):
    r = json.loads(line)
    if r["commits"] and r["models"] and r["devices"] and r["tests"]:
        for m in r["models"]:
            mid = ALIAS.get(m)
            if mid:
                extra_dev.setdefault(mid, set()).update(d for d in r["devices"] if d == "gb10")
out = {"schema": "memra-impact-models-v1",
       "note": "device classes are where the model has been measured; per-box artifact paths are local config, not tracked",
       "families": {}}
for fam in packs["families"]:
    rows = []
    for m in fam["models"]:
        devs = set(m.get("devices") or []) | extra_dev.get(m["id"], set())
        if m["id"] in roster:
            devs.add("rtx5090")  # the rtx5090 device row runs the release roster (tools/change-impact.toml)
        devs = sorted(devs)
        if not devs:
            continue
        cls = roster.get(m["id"], "own" if m["id"].lower().startswith(("avifenesh/", "ornith")) else "vendor")
        rows.append({"id": m["id"], "formats": m.get("formats", []), "routes": m.get("routes", []),
                     "devices": devs, "class": cls, "source": m.get("source", "")})
    if rows:
        out["families"][fam["family"]] = rows
json.dump(out, sys.stdout, indent=1)
