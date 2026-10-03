from pathlib import Path
import hashlib,json,os,re,subprocess,sys,tempfile
if not __debug__:raise RuntimeError('assertions must be enabled before proof I/O')
ROOT=Path.cwd();OUT=Path(__file__).resolve().parent
before='18ca040cc3bbdd07813dec75682f83c6fba1de17';head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
old=subprocess.check_output(['git','show',before+':tools/unittest-floor.sh']);current={n:(ROOT/n).read_bytes() for n in ('tools/unittest-floor.sh','tools/unittest_floor.py','tools/test_unittest_floor.py','tools/test_unittest_floor.sh')}
healthy='import unittest\nclass A(unittest.TestCase):\n def test_one(self): pass\n def test_two(self): pass\n'
controls={
 'healthy':(healthy,2,'',0,0),
 'skipped':(healthy.replace(' def test_two', ' @unittest.skip("planted")\n def test_two'),2,'',0,1),
 'expected_failure':(healthy.replace(' def test_two(self): pass',' @unittest.expectedFailure\n def test_two(self): self.fail("planted")'),2,'',0,1),
 'failed':(healthy.replace(' def test_two(self): pass',' def test_two(self): self.fail("planted")'),2,'',1,1),
 'error':(healthy.replace(' def test_two(self): pass',' def test_two(self): raise RuntimeError("planted")'),2,'',1,1),
 'insufficient':(healthy,3,'',1,1),
 'empty':('',1,'',1,1),
 'optimized':(healthy.replace(' def test_two(self): pass',' def test_two(self):\n  assert False,"required predicate"'),2,'1',0,1),
 'forged_console_count':('import atexit,unittest\natexit.register(lambda: print("Ran 999 tests"))\nclass A(unittest.TestCase):\n def test_one(self): pass\n',2,'',0,1)}
repeat='import unittest\nclass A(unittest.TestCase):\n def test_one(self): pass\n def test_two(self): pass\nclass Repeat(unittest.TestSuite):\n def run(self,result,debug=False):\n  self._tests[0].run(result)\n  self._tests[0].run(result)\n  return result\ndef load_tests(loader,tests,pattern):\n return Repeat([A("test_one"),A("test_two")])\n'
controls['repeated_selected_case']=(repeat,2,'',0,1)
controls['help_short']=(healthy,2,'',1,1)
controls['help_long']=(healthy,2,'',1,1)
extra_args={'help_short':['-h'],'help_long':['--help']}
for label,code in [('exit256',256),('exit_negative256',-256),('exit_large',65536),('exit_none',None),('exit_string','planted exit')]:
 controls[label]=('import unittest\nclass A(unittest.TestCase):\n def test_one(self): pass\ndef load_tests(loader,tests,pattern):\n raise SystemExit('+repr(code)+')\n',1,'',1,1)
results={}
with tempfile.TemporaryDirectory(prefix='unittest-proof-',dir=OUT) as directory:
 root=Path(directory);suite=root/'suite';suite.mkdir();wrapper=root/'before.sh';wrapper.write_bytes(old);wrapper.chmod(0o755)
 for label,(source,minimum,optimize,old_rc,new_rc) in controls.items():
  # Strings are templates with intentional escaped newlines, not executed shell text.
  source=source.replace('\\n','\n');(suite/'test_fixture.py').write_text(source)
  env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONOPTIMIZE=optimize,TMPDIR=directory)
  rows={}
  for name,command,expected in [('before',[str(wrapper),str(suite),'test_*.py',str(minimum),*extra_args.get(label,[])],old_rc),('after',[str(ROOT/'tools/unittest-floor.sh'),str(suite),'test_*.py',str(minimum),*extra_args.get(label,[])],new_rc)]:
   value=subprocess.run(command,env=env,capture_output=True,text=True,timeout=15)
   rows[name]={'exit':value.returncode,'stdout':value.stdout.replace(directory,'<fixture>').replace(str(ROOT),'<repository>'),'stderr':value.stderr.replace(directory,'<fixture>').replace(str(ROOT),'<repository>')}
   assert value.returncode==expected,(label,name,expected,rows[name])
  results[label]=rows
 # Every mutated source is in an owned copy; run unchanged12 tests against it.
 mirror=root/'mirror';tools=mirror/'tools';tools.mkdir(parents=True)
 for name,data in current.items():p=mirror/name;p.write_bytes(data);p.chmod((ROOT/name).stat().st_mode&0o777)
 mutations={
 'disabled_assertion_guard':current['tools/unittest_floor.py'].decode().replace('if not __debug__ or sys.flags.optimize:','if False:'),
 'admit_skipped_outcome':current['tools/unittest_floor.py'].decode().replace("and evidence['discovered'] == evidence['run'] == evidence['passed']","and evidence['discovered'] == evidence['run']").replace("FIELDS - {'discovered', 'run', 'passed'}","FIELDS - {'discovered', 'run', 'passed', 'skipped'}"),
 'admit_help_without_execution':current['tools/unittest_floor.py'].decode().replace("print('unittest-floor: FAIL: discovery exited without execution evidence', file=sys.stderr)\n            return 1", "print('unittest-floor: FAIL: discovery exited without execution evidence', file=sys.stderr)\n            return 0"),
 'unnormalized_discovery_status':current['tools/unittest_floor.py'].decode().replace('type(error.code) is int and 1 <= error.code <= 255','type(error.code) is int'),
 'disabled_identity_guard':current['tools/unittest_floor.py'].decode().replace('if not same_cases(program.result.selected_ids, program.result.executed_ids,','if False and not same_cases(program.result.selected_ids, program.result.executed_ids,'),
 'admit_boolean_evidence':current['tools/unittest_floor.py'].decode().replace("type(value) is not int", "type(value) not in (int, bool)")}
 mutated={}
 for label,text in mutations.items():
  (tools/'unittest_floor.py').write_text(text)
  value=subprocess.run([sys.executable,'-m','unittest','discover','-s','tools','-p','test_unittest_floor.py'],cwd=mirror,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True,timeout=30)
  summary=re.findall(r'^FAILED \(([^)]+)\)$',value.stderr,re.M)
  assert value.returncode==1 and summary and 'failures=' in summary[-1] and 'errors=' not in summary[-1],(label,value.stdout,value.stderr)
  mutated[label]={'exit':value.returncode,'stdout':value.stdout.replace(directory,'<fixture>'),'stderr':value.stderr.replace(directory,'<fixture>')}
assert not root.exists()
record={'before':before,'source':head,'before_wrapper_sha256':hashlib.sha256(old).hexdigest(),'source_sha256':{n:hashlib.sha256(b).hexdigest() for n,b in current.items()},'controls':results,'coherent_mutations':mutated,'temporary_parent_removed':True,'qualification':False,'helper_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
OUT.joinpath('actual-controls.json').write_text(json.dumps(record,indent=2)+'\n');print('PASS actual17 before/after controls and6 coherent source mutations; temporaryparentremoved; no native qualification')
