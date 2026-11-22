import json, sys
d = json.load(open(sys.argv[1]))
for name, s in list(d.get("sets", {}).items()) + [("POOLED", d.get("pooled", {}))]:
    print(name)
    for k in ("kl_route", "kl_self", "kl_route_rev", "kl_peer", "kl_peer_route"):
        if k in s:
            v = s[k]; print("  %-14s n %6d mean %.4f p50 %.2e p90 %.4f p99 %.3f max %.2f" % (k, v["n"], v["mean"], v["p50"], v["p90"], v["p99"], v["max"]))
    for k in ("top1_route", "top1_self", "top1_peer", "mean_route_minus_self_ci95_by_doc", "mean_route_minus_peer_ci95_by_doc"):
        if k in s: print("  ", k, s[k])
for t, v in d.get("tape_cross", {}).items():
    print("tape_cross", t, v["positions"])
    for k, s in v["pairs"].items():
        print("  %-28s mean %.4f p99 %.3f top1 %.3f" % (k, s["mean"], s["p99"], s["top1"]))
print("probe", d.get("probe"))
