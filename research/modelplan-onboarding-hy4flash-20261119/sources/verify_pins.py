"""Check every pin in ../sources.lock.json against its source (HF API, GitHub API) and print one
JSON line per pin. Exit 1 when a pin does not resolve to the recorded revision.

    python3 -I verify_pins.py ../sources.lock.json > verify-pins.jsonl
"""

import json
import sys
import urllib.request


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "pin-check", "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


lock = json.load(open(sys.argv[1]))
ok = True
for pin in lock["pins"]:
    if pin["kind"] == "hf_model":
        d = get(f"https://huggingface.co/api/models/{pin['repo']}/revision/{pin['revision']}")
        row = {"id": pin["id"], "resolved": d.get("sha"), "last_modified": d.get("lastModified"),
               "siblings": len(d.get("siblings", [])), "private": d.get("private")}
        good = d.get("sha") == pin["revision"]
    else:
        d = get(f"https://api.github.com/repos/{pin['repo']}/pulls/{pin['number']}")
        row = {"id": pin["id"], "resolved": d["head"]["sha"], "head_repo": d["head"]["repo"]["full_name"],
               "state": d["state"], "merged": d.get("merged"), "merge_commit": d.get("merge_commit_sha"),
               "title": d.get("title")}
        good = d["head"]["sha"] == pin["revision"]
        for f in pin.get("files", []):
            c = get(f"https://api.github.com/repos/{d['head']['repo']['full_name']}/contents/{f['path']}?ref={pin['revision']}")
            row.setdefault("files", []).append({"path": f["path"], "blob": c["sha"], "match": c["sha"] == f["blob"]})
            good = good and c["sha"] == f["blob"]
    row["match"] = good
    ok = ok and good
    print(json.dumps(row))
sys.exit(0 if ok else 1)
