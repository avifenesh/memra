#!/usr/bin/env python3
"""Read-only search; publish allowlisted offer fields, not ids/prices/locations.

API output is held in memory only. Cost classes are relative terciles of the
returned sample's dph_total, not quotes or a purchase/rental recommendation.
"""
import datetime
import hashlib
import json
from pathlib import Path
import subprocess

QUERIES = ["num_gpus=1 gpu_name=RTX_5090", "num_gpus=1 gpu_name=RTX_5090 vms_enabled=true",
           "vms_enabled=true"]
FIELDS = ["gpu_name", "num_gpus", "disk_name", "disk_bw", "disk_space",
          "vms_enabled", "is_vm", "bare_metal", "volume_enabled", "volume_support"]


def project(rows):
    prices = sorted(float(r["dph_total"]) for r in rows if isinstance(r.get("dph_total"), (float, int)))
    result = []
    for i, row in enumerate(rows):
        price = row.get("dph_total")
        cost = "unknown"
        if prices and isinstance(price, (float, int)):
            cost = "cheap" if price <= prices[(len(prices)-1)//3] else "medium" if price <= prices[2*(len(prices)-1)//3] else "dear"
        result.append({"sample_row": i + 1, "cost_class_within_sample": cost,
                       **{k: row[k] for k in FIELDS if k in row}})
    return result


def main():
    output = []
    for query in QUERIES:
        argv = [str(Path.home()/".local/bin/vastai"), "search", "offers", "--raw", "--limit", "50", query]
        try:
            p = subprocess.run(argv, capture_output=True, text=True, timeout=60)
        except subprocess.TimeoutExpired:
            output.append({"query": query, "status": "timeout", "timeout_seconds": 60})
            continue
        # Do not print opaque API stderr (may contain deployment/account metadata).
        if p.returncode != 0:
            output.append({"query": query, "exit_code": p.returncode, "status": "search-failed"})
            continue
        rows = json.loads(p.stdout)
        if isinstance(rows, dict):
            rows = rows.get("offers", [])
        keys = sorted({k for row in rows for k in row})
        output.append({"query": query, "limit": 50, "exit_code": 0, "rows_returned": len(rows),
                       "response_sha256": hashlib.sha256(p.stdout.encode()).hexdigest(),
                       "json_keys": keys,
                       "storage_virtualization_keys": [k for k in keys if any(w in k.lower() for w in ("disk", "vm", "bare", "volume", "container"))],
                       "offers_allowlisted": project(rows)})
    print(json.dumps({"utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                      "method": "read-only CLI JSON projection; default rentable/verified filters; on-demand search",
                      "queries": output}, indent=2))


if __name__ == "__main__":
    main()
