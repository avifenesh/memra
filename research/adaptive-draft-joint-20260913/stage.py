"""Download only immutable artifacts and verify all bytes before admitting them."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

root = Path(sys.argv[1]).resolve()
root.mkdir(parents=True, exist_ok=True)
lock = json.loads(Path(__file__).with_name("artifacts.lock.json").read_text())
for item in lock["files"]:
    path = root / item["name"]
    url = f'https://huggingface.co/{item["repo"]}/resolve/{item["revision"]}/{item["file"]}'
    if not path.exists():
        temporary = path.with_suffix(".part")
        subprocess.run(["curl", "--fail", "--location", "--retry", "3", "--continue-at", "-", "--output", str(temporary), url], check=True)
        temporary.rename(path)
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    digest = h.hexdigest()
    if path.stat().st_size != item["size"] or digest != item["sha256"]:
        raise RuntimeError(f"Artifact mismatch: {path.name}; size={path.stat().st_size}, sha256={digest}")
    print(json.dumps({"artifact": path.name, "sha256": digest, "status": "verified"}), flush=True)
(root / "VERIFIED").write_text("All files match artifacts.lock.json\n")
