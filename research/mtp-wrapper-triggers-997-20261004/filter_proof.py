from pathlib import Path
import json,re,fnmatch,hashlib,subprocess,sys
R=Path(__file__).resolve().parent;W=Path('/home/avifenesh/projects/wt-memra-mtp-wrapper-triggers-997')
paths=['.github/workflows/mtp-context-receipts.yml','.github/workflows/mtp-depth-receipts.yml'];wrappers=['tools/unittest-floor.sh','tools/unittest_floor.py']
def filters(text):
 m=re.search(r'^  pull_request:\n    paths:\n((?:      - .+\n)+)',text,re.M);assert m
 return [x.strip().removeprefix('- ') for x in m[1].splitlines()]
def selected(patterns,path):return any(fnmatch.fnmatchcase(path,p) for p in patterns)
def oracle(old,current,rel):
 before=filters(old);after=filters(current);expected=before[:]
 for w in reversed(wrappers):expected.insert(0,w)
 for w in wrappers:assert selected(after,w),('required wrapper change not selected',rel,w)
 assert after==expected,('not exact two additions',rel,after)
 for p in [rel,*[x.replace('/**','/reader.py') for x in before if x.endswith('/**')]]:assert selected(before,p) and selected(after,p),(rel,p)
 for p in ['tools/unrelated.py','tools/unittest_floor.py.backup','docs/unrelated.md','research/unrelated/reader.py']:
  assert not selected(before,p) and not selected(after,p),(rel,p)
 # Removing only the two path lines must recover every original job/command/pin byte.
 assert current.replace('      - tools/unittest-floor.sh\n','').replace('      - tools/unittest_floor.py\n','')==old,rel
 return {'workflow':rel,'before_paths':before,'after_paths':after,'old_wrapper_selected':[selected(before,p) for p in wrappers],'new_wrapper_selected':[selected(after,p) for p in wrappers],'all_other_workflow_bytes_equal':True}
if len(sys.argv)>1:
 d=json.loads(Path(sys.argv[1]).read_text());print(json.dumps(oracle(d['before'],d['candidate'],d['workflow'])));sys.exit(0)
rows=[];controls=[]
for rel in paths:
 old=subprocess.check_output(['git','show','469e97575ac7a8d734c4fb9602ca85fa16d4ee49:'+rel],cwd=W,text=True);candidate=(W/rel).read_text();row=oracle(old,candidate,rel);rows.append(row)
 for w in wrappers:
  wrong=candidate.replace('      - '+w+'\n','      - '+w+'.wrong-target\n',1);p=R/(Path(rel).stem+'-'+Path(w).name+'.mutant.json');p.write_text(json.dumps({'workflow':rel,'before':old,'candidate':wrong},indent=2)+'\n');command=[sys.executable,str(Path(__file__).resolve()),str(p)];result=subprocess.run(command,capture_output=True,text=True,timeout=30);log=p.with_suffix('.output.log');log.write_text(result.stdout+result.stderr);assert result.returncode!=0 and 'required wrapper change not selected' in result.stderr,(rel,w,result.stdout,result.stderr);controls.append({'workflow':rel,'wrong_target':w+'.wrong-target','command':command,'actual_exit':result.returncode,'raw_sha256':hashlib.sha256(log.read_bytes()).hexdigest(),'actual_mutant':p.name})
(R/'FILTER-RESULTS.json').write_text(json.dumps({'rows':rows,'coherent_wrong_target_controls':controls,'actual_original_wrapper_omissions':4,'candidate_selections':4,'no_job_command_action_pin_floor_default_or_tolerance_change':True,'qualification':False},indent=2)+'\n');print('Both wrapper paths selected in both workflows; exact4 YAML lines only; four coherent wrong-target controls refused.')
