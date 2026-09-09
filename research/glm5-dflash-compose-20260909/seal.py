#!/usr/bin/env python3
import hashlib,json,tarfile
from pathlib import Path
R=Path(__file__).resolve().parent
archive=R/'compose-raw.tar.gz'
excluded={'compose-raw.tar.gz','compose-raw.tar.gz.sha256','raw-manifest.json','boundary.log','boundary.exit'}
files=[p for p in sorted(R.rglob('*')) if p.is_file() and p.relative_to(R).parts[0] not in excluded]
manifest=[{'path':str(p.relative_to(R)),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in files]
(R/'raw-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
with tarfile.open(archive,'w:gz') as tar:
 for p in files+[R/'raw-manifest.json']:tar.add(p,arcname=str(p.relative_to(R)))
hash=hashlib.sha256(archive.read_bytes()).hexdigest()
(R/'compose-raw.tar.gz.sha256').write_text(hash+'  compose-raw.tar.gz\n')
print(hash, len(files), archive.stat().st_size)
