"""Bind actual CPU assertions and red controls to current source/helper bytes."""
import contextlib, copy, hashlib, io, json, pathlib, subprocess, sys, unittest
if not __debug__:
 raise RuntimeError('support execution admission assertions must be enabled')
root=pathlib.Path(sys.argv[1]).resolve(); out=pathlib.Path(sys.argv[2]).resolve()
sys.path.insert(0,str(root/'tools'))
import test_validation_support_execution_inputs as suite
import validation_coverage as coverage
names=unittest.defaultTestLoader.getTestCaseNames(suite.SupportExecutionInputs)
inputs=json.loads((out/'source-pins.json').read_text())['inputs']
for helper in ('controls.py','admit.py','original_support_record_inputs.py'):
 path=pathlib.Path(__file__).with_name(helper)
 inputs[str(path.relative_to(root))]=hashlib.sha256(path.read_bytes()).hexdigest()
context={'lane':'support-execution-type-967','program':'CPU admission only','source':subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD']).decode().strip()}
controls=json.loads((out/'mutation-results.json').read_text())
tests=[]; results={}; edges=[]
for name in names:
 edge='execution/'+name.removeprefix('test_');edges.append(edge)
 tests.append({'id':name,'cost':1,'covers':[edge],'inputs':inputs,'scope':context,'mandatory':True,'controls':['coherent-red-controls']})
 buffer=io.StringIO()
 with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
  result=unittest.TextTestRunner(stream=buffer,verbosity=2).run(unittest.TestSuite([suite.SupportExecutionInputs(name)]))
 (out/(name+'.log')).write_text(buffer.getvalue())
 assert result.wasSuccessful() and result.testsRun==1 and not result.skipped, name
 results[name]={'status':'passed','context':context,'executed':result.testsRun,'skipped':0,'edges':{edge:'passed'}}
rededges=['red/'+control['control'] for control in controls]
for control in controls:
 raw=(out/(control['control']+'.log')).read_bytes()
 assert hashlib.sha256(raw).hexdigest()==control['log_sha256'] and control['exit']==1
 assert b'FAIL:' in raw and b'ERROR:' not in raw
edges.extend(rededges)
tests.append({'id':'coherent-red-controls','cost':1,'covers':rededges,'inputs':inputs,'scope':context,'mandatory':True})
results['coherent-red-controls']={'status':'passed','context':context,'executed':len(controls),'skipped':0,'edges':{e:'passed' for e in rededges}}
manifest={'required_edges':edges,'tests':tests,'context':context,'cost_unit':'one mandatory control group, not wall time'}
plan=coverage.select(edges,tests,context,root)
for result in results.values():result['contract_id']=plan['contract_id']
verdict=coverage.validate_results(plan,results,context,root)
negative=[]
for kind in ['missing-edge','skipped-mandatory']:
 bad=copy.deepcopy(results)
 if kind=='missing-edge': bad[names[0]]['edges']={}
 else: bad[names[0]]['skipped']=1
 try:coverage.validate_results(plan,bad,context,root)
 except ValueError as error:negative.append({'control':kind,'refused':str(error)})
 else:raise AssertionError(kind)
for name,data in [('coverage-manifest.json',manifest),('coverage-plan.json',plan),('coverage-results.json',results),('coverage-verdict.json',verdict),('coverage-negative.json',negative)]:
 (out/name).write_text(json.dumps(data,indent=2)+'\n')
print(json.dumps({'verdict':verdict,'negative_controls':negative}))
