import json, sys
d = json.load(open(sys.argv[1]))["tape_cross"]
for t, v in d.items():
    print(t, v["positions"], "positions")
    for k, s in v["pairs"].items():
        print("  %-28s mean %.4f p50 %.2e p90 %.3f p99 %.3f max %.3f top1 %.3f" % (k, s["mean"], s["p50"], s["p90"], s["p99"], s["max"], s["top1"]))
