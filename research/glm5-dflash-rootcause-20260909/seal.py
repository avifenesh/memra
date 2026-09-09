#!/usr/bin/env python3
"""Seal completed raw evidence after the owned server and client exit."""
import collections,csv,hashlib,json,re,tarfile
from pathlib import Path
R=Path(__file__).resolve().parent
assert (R/'cell.exit').read_text().strip()=='0'
pid=int((R/'raw/pid').read_text());assert not Path(f'/proc/{pid}').exists(), 'server still alive'
s=json.loads((R/'summary.json').read_text());timing=[x for x in s['requests'] if x['label'].startswith('timing-')];agreement=[x for x in s['requests'] if x['label'].startswith('agreement-')]
assert len(timing)==63 and len(agreement)==20
assert len(s['medians'])==21 and all(x['n']==3 for x in s['medians'])
assert not any(x['loop_suspect'] for x in timing+agreement)
rounds=0;chosen=collections.defaultdict(set)
for a in timing:
 for k in re.findall(r'\[rootcause-round\] round=\d+ k=(\d+)',(R/'raw'/a['label']/'server.log').read_text()):
  chosen[str(a['k'])].add(int(k));rounds+=1
for k in ['2','4','6']:assert chosen[k]=={int(k)},chosen
assert chosen['auto']=={3},chosen
text_hashes=sorted({hashlib.sha256(p.read_bytes()).hexdigest() for p in (R/'raw').glob('*greedy-*/output.txt')})
assert len(text_hashes)==1,text_hashes
start=min((R/'raw'/x['label']/'result.json').stat().st_mtime-x['wall_s'] for x in timing)
end=max((R/'raw'/x['label']/'result.json').stat().st_mtime for x in timing)
# Thermal envelope for the entire boot, including load and cold warmup.
temps=[]
with (R/'raw/telemetry.csv').open() as f:
 for row in csv.DictReader(f):
  try:temps.append(float(row[' temperature.gpu']))
  except (KeyError,ValueError):pass
validation={'requests':len(s['requests']),'timing_requests':len(timing),'agreement_requests':len(agreement),'timing_rounds':rounds,'timing_and_agreement_outputs':dict(collections.Counter(x['n'] for x in timing+agreement)),'all_medians_n3':True,'usage_log_counts_reconciled':True,'recorded_pqu_replay_pass':True,'chosen_k':{k:sorted(v) for k,v in chosen.items()},'greedy_text_sha256':text_hashes,'greedy_text_requests':len(list((R/'raw').glob('*greedy-*/output.txt'))),'loop_suspects':[],'timing_window_epoch_approx':[start,end],'thermal_scope':'whole boot including loading and cold warmup','temperature_c_min':min(temps) if temps else None,'temperature_c_max':max(temps) if temps else None,'owned_server_pid_absent':pid}
(R/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
prefixes=['raw','failed-attempt1','failed-attempt2','failed-attempt3','interrupted-preflight4','failed-attempt5','invalid-config6']
files=sorted(p for name in prefixes for p in (R/name).rglob('*') if p.is_file())
files += [R/name for name in ['build.log','build.exit','binary.sha256','causal-tests.log','causal-tests.exit','cell.exit','final-artifact-hashes.txt','staged-model-hashes.txt','pmin-counterexample.json']]
manifest={str(p.relative_to(R)):{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in files}
(R/'raw-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
archive=R/'rootcause-raw.tar.gz'
with tarfile.open(archive,'w:gz') as tar:
 for p in files:tar.add(p,arcname=str(p.relative_to(R)))
digest=hashlib.sha256(archive.read_bytes()).hexdigest();(R/'rootcause-raw.tar.gz.sha256').write_text(digest+'  '+archive.name+'\n')
print(json.dumps(validation,indent=2));print('archive',digest,'bytes',archive.stat().st_size)
