from pathlib import Path
import hashlib, importlib.util, json, os, shutil, subprocess, sys, tempfile
if not __debug__:raise RuntimeError('assertions must be enabled before source/fixture I/O')
repo=Path.cwd(); out=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('current_planner',repo/'tools/validation_plan.py');vp=importlib.util.module_from_spec(spec);spec.loader.exec_module(vp)
tree=vp.LocalTree(repo);vp.support_record_source_inputs(tree); roles=vp.support_record_receipt_copy_inputs(tree); roots=roles['roots']; assert len(roots)==7
results=[]
with tempfile.TemporaryDirectory(prefix='current-consumers-',dir=out) as scratch:
 root=Path(scratch)/'repo';root.mkdir()
 for name in ('crates/memra-gguf/src/model_packs','docs',*roots):shutil.copytree(repo/name,root/name,dirs_exist_ok=True)
 for name in ('crates/memra-cli/src/lib.rs','README.md','STATUS.md','AGENTS.md',*vp._SUPPORT_DATA.READERS):
  target=root/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(repo/name,target)
 subprocess.run(['git','init','-q'],cwd=root,check=True);subprocess.run(['git','add','.'],cwd=root,check=True)
 consumer=[sys.executable,'-m','unittest','discover','-s','tools','-p','test_check_support_states.py','-k','test_clean_copy_passes']
 def run(command=consumer):
  r=subprocess.run(command,cwd=root,capture_output=True,text=True,timeout=30)
  return {'exit':r.returncode,'stdout':r.stdout.replace(scratch,'<fixture>'),'stderr':r.stderr.replace(scratch,'<fixture>')}
 check=[sys.executable,str(root/'tools/check-support-states.py'),'--root',str(root)]
 baseline_direct=run(check);assert baseline_direct['exit']==0;results.append({'case':'baseline-direct','consumer':baseline_direct})
 baseline=run();assert baseline['exit']==0;results.append({'case':'baseline','consumer':baseline})
 for parent in roots:
  name=parent+'/CPU_TRANSPORT_CONTROL.txt';path=root/name;path.symlink_to('missing-owned-target')
  direct=run(check);assert direct['exit']==0
  red=run();assert red['exit']==1 and 'CPU_TRANSPORT_CONTROL' in red['stderr']
  try:vp.support_record_receipt_copy_inputs(vp.LocalTree(root));raise AssertionError('unsafe copy accepted')
  except vp.Refused as error: refusal=str(error)
  path.unlink();path.write_text('Readable excluded content NativeQualified\n')
  green=run();assert green['exit']==0
  assert vp.support_record_receipt_copy_inputs(vp.LocalTree(root))['active']
  plan=vp.make_plan([name],vp.Tree(repo,'HEAD'),vp.Tree(repo,'HEAD'))
  assert not plan['cpu_contracts'] and not plan['packages'] and not plan['native']['qualification']
  results.append({'case':name,'direct_checker':direct,'broken_link_consumer':red,'transport_refusal':refusal,'regular_consumer':green,'regular_plan':{'mode':plan['mode'],'cpu_contracts':plan['cpu_contracts'],'packages':plan['packages'],'native_qualification':plan['native']['qualification']}})
  path.unlink();os.mkfifo(path)
  try:vp.support_record_receipt_copy_inputs(vp.LocalTree(root));raise AssertionError('FIFO accepted')
  except vp.Refused as error:results.append({'case':name,'fifo_refusal':str(error)})
  finally:path.unlink()
result={'source':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'source_sha256':{name:hashlib.sha256((repo/name).read_bytes()).hexdigest() for name in ('tools/support_record_inputs.py','tools/validation_plan.py','tools/test_validation_support_receipt_copies.py','tools/skip-census.py','docs/support-records.toml')},'roots':roots,'readers':{name:hashlib.sha256((repo/name).read_bytes()).hexdigest() for name in vp._SUPPORT_DATA.READERS},'outcomes':results,'qualification':False,'helper_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
out.joinpath('current-consumers.json').write_text(json.dumps(result,indent=2)+'\n');print('PASS: 7 cited/legacy copy roots; 23 actual consumer outcomes: 2 baseline + 7 direct checker successes + 7 broken-link failures + 7 regular-copy successes; 7 FIFO preflight refusals; regular content omissions preserved')
