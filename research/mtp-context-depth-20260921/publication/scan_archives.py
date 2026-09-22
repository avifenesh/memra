"""Inventory expanded publication-boundary matches without treating them as approvals."""
import argparse
import hashlib
import importlib.util
import json
import sys
import tarfile
import tempfile
from pathlib import Path

RUNTIME_SHA256 = '7f5a5d5446e3af717085d0258a6993743e892a32badc79af94366523a6144328'
REVIEWED_RUNTIME_HASHES = {
    RUNTIME_SHA256,
    'd313e757edbb3b469890526e75522fd7f9893a74d990e9c5c5345bb50b9cd7f0',
    '943d165b80f268ffacc63e78191c669ed4bda562b3151d45758876321e3f6dc8',
}


def sha(data):return hashlib.sha256(data).hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--receipts',type=Path,default=Path('/opt/adaptive/receipts/public'))
    p.add_argument('--prefix',default='research/mtp-context-depth-20260921/receipts')
    p.add_argument('--output',type=Path)
    a=p.parse_args()
    manifest=json.loads((a.receipts/'manifest.json').read_text())
    source=a.receipts/'runtime-source.tar.gz'
    source_hash = sha(source.read_bytes())
    if source_hash not in REVIEWED_RUNTIME_HASHES:
        raise ValueError('Unrecognized source; archived scanner was not loaded')
    if manifest['runtime_source']['runtime_archive_sha256'] != source_hash:
        raise ValueError('Source declaration does not match its archive')
    with tarfile.open(source) as archive:
        scanner=archive.extractfile('tools/check-public-boundary.py').read()
        policy_data=archive.extractfile('tools/public-boundary-policy.toml').read()
        allowlist=archive.extractfile('tools/public-boundary-allowlist.jsonl').read().decode()
    with tempfile.TemporaryDirectory(prefix='context-boundary-') as d:
        tools=Path(d)/'tools';tools.mkdir()
        (tools/'check-public-boundary.py').write_bytes(scanner)
        (tools/'public-boundary-policy.toml').write_bytes(policy_data)
        spec=importlib.util.spec_from_file_location('context_archived_boundary',tools/'check-public-boundary.py')
        boundary=importlib.util.module_from_spec(spec);sys.modules[spec.name]=boundary;spec.loader.exec_module(boundary)
        policy=boundary.load_policy(tools/'public-boundary-policy.toml')
    approvals=[json.loads(s) for s in allowlist.splitlines() if s.strip() and not s.startswith('#')]
    reports=[]
    entries=[*manifest['archives'],{'file':'runtime-source.tar.gz','sha256':source_hash,
                                  'bytes':manifest['runtime_source']['archive_bytes']}]
    for entry in entries:
        path=a.receipts/entry['file'];data=path.read_bytes()
        if len(data)!=entry['bytes'] or sha(data)!=entry['sha256']:raise ValueError('Archive identity changed')
        outer=boundary.evaluate_content(policy,a.prefix+'/'+path.name,data)
        matches=[];count=0;bypassed=0
        with tarfile.open(path) as archive:
            for member in archive:
                if path.name=='runtime-source.tar.gz' and member.isdir():continue
                if not member.isfile():raise ValueError('Unexpected non-regular archive member')
                content=archive.extractfile(member).read();count+=1
                if path.name=='runtime-source.tar.gz':
                    if boundary.is_bypass(member.name,policy.bypass_paths):bypassed+=1
                    violation=boundary.evaluate_content(policy,member.name,content)
                    rules=list(boundary.violation_rules(violation)) if violation else []
                else:
                    rules=[x[0] for x in boundary.scan_secret_bytes(content,policy.secret_union,policy.secret_groups,policy.secret_patterns)]
                if rules:
                    digest=sha(content)
                    known=[r for r in approvals if r.get('sha256')==digest and r.get('file',r.get('path'))==member.name]
                    matches.append({'file':member.name,'sha256':digest,'rules':rules,'archived_source_approvals':known})
        reports.append({'file':path.name,'sha256':entry['sha256'],'files':count,'bypassed_source_files':bypassed,
                        'compressed_rules':list(boundary.violation_rules(outer)) if outer else [],'expanded_matches':matches})
    result={'status':'review_required' if any(r['expanded_matches'] or r['compressed_rules'] for r in reports) else 'no_matches',
            'runtime_archive_sha256':source_hash,'scanner_sha256':sha(scanner),'policy_sha256':sha(policy_data),
            'archives':reports}
    (a.output or a.receipts/'boundary-scan.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
