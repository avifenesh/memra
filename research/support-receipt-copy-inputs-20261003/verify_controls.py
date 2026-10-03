"""Admit complete, current transport controls; reject stale or broken proof."""
from pathlib import Path
import hashlib, json, re, subprocess, sys, tomllib
if not __debug__:raise RuntimeError('assertions must be enabled before proof I/O')
ROOT=Path.cwd(); PROOF=Path(__file__).resolve().parent

def verify(data):
 records=tomllib.loads((ROOT/'docs/support-records.toml').read_text())['record']
 roots={str(Path(p).parent) for r in records for ps in r.get('evidence',{}).values() for p in ps if not p.startswith('ci:')}
 roots.add('research/modelplan-onboarding-hy3-20260830/tiny')
 assert len(roots)==7 and set(data['roots'])==roots
 assert re.fullmatch(r'[0-9a-f]{40}',data['source'])
 assert set(data['source_sha256'])=={'tools/support_record_inputs.py','tools/validation_plan.py','tools/test_validation_support_receipt_copies.py','tools/skip-census.py','docs/support-records.toml'}
 for name,digest in data['source_sha256'].items():
  assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest
  assert hashlib.sha256(subprocess.check_output(['git','show',data['source']+':'+name],stderr=subprocess.DEVNULL)).hexdigest()==digest
 assert set(data['readers'])=={'tools/check-support-states.py','tools/test_check_support_states.py'}
 for name,digest in data['readers'].items():
  assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest
  assert hashlib.sha256(subprocess.check_output(['git','show',data['source']+':'+name],stderr=subprocess.DEVNULL)).hexdigest()==digest
 helper=str((PROOF/'consumer_controls.py').relative_to(ROOT))
 assert hashlib.sha256((ROOT/helper).read_bytes()).hexdigest()==data['helper_sha256']
 assert hashlib.sha256(subprocess.check_output(['git','show',data['source']+':'+helper],stderr=subprocess.DEVNULL)).hexdigest()==data['helper_sha256']
 rows=data['outcomes'];assert len(rows)==16 and data['qualification'] is False
 baseline=[r for r in rows if r['case'] in ('baseline','baseline-direct')]
 assert {r['case'] for r in baseline}=={'baseline','baseline-direct'} and all(r['consumer']['exit']==0 for r in baseline)
 copies=[r for r in rows if 'broken_link_consumer' in r]; fifos=[r for r in rows if 'fifo_refusal' in r]
 expected={p+'/CPU_TRANSPORT_CONTROL.txt' for p in roots}
 assert len(copies)==7 and len(fifos)==7 and {r['case'] for r in copies}==expected and {r['case'] for r in fifos}==expected
 for r in copies:
  assert r['direct_checker']['exit']==0 and r['broken_link_consumer']['exit']==1 and 'CPU_TRANSPORT_CONTROL' in r['broken_link_consumer']['stderr'] and r['regular_consumer']['exit']==0
  assert 'receipt copy input type' in r['transport_refusal']
  assert r['regular_plan']=={'mode':'scoped','cpu_contracts':[],'packages':[],'native_qualification':False}
 assert all('receipt copy input type' in r['fifo_refusal'] for r in fifos)
 return {'actual_consumer_outcomes':23,'fifo_refusals':7,'copy_roots':7,'source':data['source'],'qualification':False}

if __name__=='__main__':
 import copy
 data=json.loads((PROOF/'current-consumers.json').read_text()); result=verify(data)
 mutations=[]
 for label,mutate in [('missing_case',lambda d:d['outcomes'].pop()),('stale_source',lambda d:d['source_sha256'].update({'tools/validation_plan.py':'0'*64})),('false_readable_failure',lambda d:next(r for r in d['outcomes'] if 'regular_consumer' in r)['regular_consumer'].update({'exit':1})),('false_broken_success',lambda d:next(r for r in d['outcomes'] if 'broken_link_consumer' in r)['broken_link_consumer'].update({'exit':0})),('weakened_omission',lambda d:next(r for r in d['outcomes'] if 'regular_plan' in r)['regular_plan'].update({'native_qualification':True})),('missing_reader',lambda d:d['readers'].pop('tools/check-support-states.py')),('stale_source_identity',lambda d:d.update({'source':'0'*40})),('stale_helper',lambda d:d.update({'helper_sha256':'0'*64}))]:
  changed=copy.deepcopy(data);mutate(changed)
  try:verify(changed)
  except (AssertionError,KeyError,subprocess.CalledProcessError):mutations.append(label)
  else:raise RuntimeError('invalid proof admitted: '+label)
 result['rejected_mutations']=mutations
 (PROOF/'admission.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
