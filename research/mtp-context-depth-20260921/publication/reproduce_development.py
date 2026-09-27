"""Reproduce the first iteration's development reports from its exact audited source."""
import argparse
import hashlib
import importlib
import io
import json
import math
import sys
import tarfile
import tempfile
from pathlib import Path
from unpack_receipts import unpack

# Exact measured archive, not the later paired-evidence revision.
SOURCE_SHA='d313e757edbb3b469890526e75522fd7f9893a74d990e9c5c5345bb50b9cd7f0'
PREFIX='research/mtp-context-depth-20260921/'


def main():
    p=argparse.ArgumentParser();p.add_argument('--receipts',type=Path,required=True)
    p.add_argument('--manifest-sha256',required=True);a=p.parse_args()
    if not __debug__:raise RuntimeError('Verification requires normal Python assertions')
    data=(a.receipts/'runtime-source.tar.gz').read_bytes()
    assert hashlib.sha256(data).hexdigest()==SOURCE_SHA
    reports={}
    with tempfile.TemporaryDirectory(prefix='context-prior-reproduction-') as d:
        root=Path(d);manifest=unpack(a.receipts,root/'data',a.manifest_sha256)
        assert not manifest['unused_heldout_outputs_included']
        assert manifest['runtime_source']['runtime_archive_sha256']==SOURCE_SHA
        code=root/'code';code.mkdir()
        with tarfile.open(fileobj=io.BytesIO(data)) as archive:
            for member in archive:
                if member.name.startswith(PREFIX) and member.isfile():
                    name=member.name[len(PREFIX):]
                    if '/' not in name and (name.endswith('.py') or name.endswith('-artifacts.lock.json')):
                        (code/name).write_bytes(archive.extractfile(member).read())
        sys.path.insert(0,str(code))
        audit=importlib.import_module('audit_context');reuse=importlib.import_module('audit_reuse')
        controller=importlib.import_module('controller');runner=importlib.import_module('run_study')
        assert json.loads((root/'data/source.json').read_text())==manifest['runtime_source']
        binaries={Path(path).name:sha for sha,path in
                  (line.split(maxsplit=1) for line in (root/'data/binaries.sha256').read_text().splitlines())}
        for family,cap in [('qwen',7),('gemma',5)]:
            study=root/'data/experiment';priors=study/f'{family}-priors.tsv'
            groups=json.loads((study/f'{family}-development-sets.json').read_text())
            assert len(groups)==2 and [g['cycle'] for g in groups]==[0,1]
            for g in groups:
                parent=study/g['receipt_dir']
                assert json.loads((parent/'runs.json').read_text())==g['records']
                identity=json.loads((parent/'identity.json').read_text())
                assert identity['source_commit']=='e1cd38a47c8c36aeef75860f4cc623d5203188fc'
                binary='mtp-depth-study' if family=='qwen' else 'gemma-depth-study'
                assert identity['binary_sha256']==binaries[binary]
                assert identity['audit_context_sha256']==hashlib.sha256(Path(audit.__file__).read_bytes()).hexdigest()
                assert identity['audit_reuse_sha256']==hashlib.sha256(Path(reuse.__file__).read_bytes()).hexdigest()
                assert identity['runner_sha256']==hashlib.sha256((code/'run_study.py').read_bytes()).hexdigest()
                assert identity['artifacts']==json.loads((code/f'{family}-artifacts.lock.json').read_text())
                assert identity['context_priors_sha256']==hashlib.sha256(priors.read_bytes()).hexdigest()
                runner.audit_control_ids(parent,g['seed'],['measured'])
                for r in g['records']:
                    run=parent/f'{g["seed"]}-{r["arm"]}'
                    assert audit.audit_run(run,r['arm'],cap,priors)==r['context_audit']
                    assert reuse.audit_reuse(run)==r['reuse']
                    turns=audit.rows(run/'turns.tsv')
                    tokens=sum(len((run/f'turn-{i}.output.ids').read_text().split()) for i in range(1,9))
                    seconds=sum(float(t['elapsed_s']) for t in turns)
                    assert tokens==r['tokens'] and math.isclose(seconds,r['elapsed_s'],abs_tol=1e-6)
            report=controller.summarize(groups,2)
            assert report==json.loads((study/f'{family}-development-report.json').read_text())
            reports[family]=report
    result={'status':'passed','development_reports_reproduced_exactly':True,'turns':160,'reports':reports}
    if json.loads((a.receipts/'REPRODUCTION.json').read_text()) != result:
        raise ValueError('The sealed development reproduction differs from this replay')
    print(json.dumps({'status':'passed','development_turns_reproduced':160}))


if __name__=='__main__':main()
