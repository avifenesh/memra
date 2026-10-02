#!/usr/bin/env python3
"""Materialize captured f32 paths for the standalone comparison tools."""
import argparse, hashlib, json, pathlib, tarfile
ap=argparse.ArgumentParser();ap.add_argument('destination');a=ap.parse_args()
r=pathlib.Path(__file__).resolve().parent;out=pathlib.Path(a.destination).resolve()
index=json.loads((r/'f32-index.json').read_text())
with tarfile.open(r/'f32-receipts.tar.gz','r:gz') as archive:
    for name,item in index.items():
        path=(out/name).resolve()
        assert path.is_relative_to(out) and not path.exists(),path
        data=archive.extractfile(item['sha256']+'.f32').read()
        assert len(data)==item['bytes'] and hashlib.sha256(data).hexdigest()==item['sha256']
        path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
print(f'Materialized {len(index)} f32 files in {out}')
