#!/usr/bin/env python3
"""Port the DAY59 cell scripts (sitting tree f0e77b8be) to <box>. Reads the pristine copies in orig/, writes kit/tree/.
Every replacement must match exactly once; re-running starts from orig/ again, so it is idempotent."""
from pathlib import Path

BASE = Path("/home/avifenesh/.cache/memra-maxq/moe-prefetch")
ORIG, TREE = BASE / "orig", BASE / "kit" / "tree"

EDITS = {
    "research/spill-c-20260919/day40-run-cell.sh": [
        (': "${D40_RIG:?}" "${D40_R:?}" "${D40_TREE:?}" "${D40_CELL_SCRIPT:?}"\n',
         ': "${D40_RIG:?}" "${D40_R:?}" "${D40_TREE:?}" "${D40_CELL_SCRIPT:?}"\n'
         '# Max-Q kit (<box>): every card check names the one card under test (D40_CARD, nvidia-smi index, PCI bus order).\n'
         ': "${D40_CARD:?set D40_CARD to the nvidia-smi index of the card under test}"\n'),
        ('    apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .)\n',
         '    apps=$(nvidia-smi -i "$D40_CARD" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .)\n'),
        ('        nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>&1; } >> "$D40_R/waits.log"\n',
         '        nvidia-smi -i "$D40_CARD" --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>&1; } >> "$D40_R/waits.log"\n'),
    ],
    "research/spill-c-20260919/day59-cell.sh": [
        (': "${D40_R:?}" "${D40_BINS:?}" "${D40_TREE:?}" "${D40_LOCK:?}" "${D40_ART:?}"\n',
         ': "${D40_R:?}" "${D40_BINS:?}" "${D40_TREE:?}" "${D40_LOCK:?}" "${D40_ART:?}"\n'
         '# Max-Q kit (<box>): the card is pinned by the runner (CUDA_DEVICE_ORDER=PCI_BUS_ID, CUDA_VISIBLE_DEVICES=<card>);\n'
         '# D40_CARD is the same card for nvidia-smi.\n'
         ': "${D40_CARD:?}" "${CUDA_VISIBLE_DEVICES:?}" "${CUDA_DEVICE_ORDER:?}"\n'),
        ('git rev-parse HEAD | tee "$EV/tree.sha"\n',
         '# Max-Q kit: no git checkout on <box>; the kit carries the source SHA of these scripts and of the binaries.\n'
         'cat "$D40_TREE/SOURCE_SHA" | tee "$EV/tree.sha"\n'
         'cp "$D40_BINS/BUILD_SHA" "$EV/build.sha"\n'),
        ('    { date -u +%FT%T.%3NZ; nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>&1\n',
         '    { date -u +%FT%T.%3NZ; nvidia-smi -i "$D40_CARD" --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>&1\n'),
        ('        env CUDA_VISIBLE_DEVICES=0 MEMRA_COMPAT=openai',
         '        env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" MEMRA_COMPAT=openai'),
    ],
    "tools/tier-battery.py": [
        ('            path = root / f"{stem}.{label}.log"\n'
         '            code, expired = tee_run([smi, "--query-compute-apps=pid,process_name,used_memory",\n'
         '                                    "--format=csv"], path, timeout=20, echo=False)\n',
         '            path = root / f"{stem}.{label}.log"\n'
         '            code, expired = tee_run([smi, *card_args, "--query-compute-apps=pid,process_name,used_memory",\n'
         '                                    "--format=csv"], path, timeout=20, echo=False)\n'),
        ('        smi = shutil.which(self.smi)\n',
         '        smi = shutil.which(self.smi)\n'
         '        # Max-Q kit (<box>): telemetry and snapshots name the one card under test (D40_CARD, nvidia-smi index).\n'
         '        card = os.environ.get("D40_CARD")\n'
         '        card_args = ["-i", card] if card else []\n'),
        ('                argv = [smi, "--query-gpu=timestamp,',
         '                argv = [smi, *card_args, "--query-gpu=timestamp,'),
    ],
}

for rel in ["tools/tier-battery.py", "tools/tier-lock-proof.py", "tools/port-guard.sh",
            "research/spill-c-20260919/day40-run-cell.sh", "research/spill-c-20260919/day59-cell.sh",
            "research/spill-c-20260919/day59-serve-client.py", "research/spill-c-20260919/day59-pf.py",
            "research/spill-c-20260919/day40-attrib.py", "research/spill-c-20260919/c-regime.py"]:
    text = (ORIG / rel).read_text()
    for old, new in EDITS.get(rel, []):
        n = text.count(old)
        assert n == 1, f"{rel}: expected 1 match, found {n}: {old[:60]!r}"
        text = text.replace(old, new)
    out = TREE / rel
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    out.chmod((ORIG / rel).stat().st_mode & 0o777)
    print(f"{rel}: {len(EDITS.get(rel, []))} edits")
