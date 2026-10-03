"""Compiling source mutants; every negative must assert-fail without fixture errors."""
import ast,hashlib,json,os,subprocess,sys,tempfile
from pathlib import Path
if not __debug__:raise RuntimeError('summary controls require assertions')
root=Path(sys.argv[1]).resolve();out=Path(sys.argv[2]).resolve();out.mkdir(parents=True,exist_ok=True)
source=(root/'tools/summarize_hy3_plan_agreement.py').read_text();test=(root/'tools/test_agreement_summary_contract.py').read_text();runner=(root/'tools/run_agreement_summary_contract.py').read_text();workflow=(root/'.github/workflows/ci.yml').read_text()
tree=ast.parse(source);node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='self_test');node.body=[ast.Pass()];noop=ast.unparse(ast.fix_missing_locations(tree))
tree=ast.parse(source);node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='self_test')
class Remove(ast.NodeTransformer):
    removed=False
    def visit_Assert(self,n):
        if not self.removed:self.removed=True;return ast.copy_location(ast.Pass(),n)
        return n
Remove().visit(node);missing=ast.unparse(ast.fix_missing_locations(tree))
loop='for plan in result["traffic_overlay"]["plans"].values():'
mutants=[('noop','summary',noop,'test_original_eight_predicates_ten_entries_and_real_calls'),('deleted-predicate','summary',missing,'test_original_eight_predicates_ten_entries_and_real_calls'),('missing-c','summary',source.replace(loop,'for plan in list(result["traffic_overlay"]["plans"].values())[:2]:'),'test_original_eight_predicates_ten_entries_and_real_calls'),('duplicate-a-missing-b','summary',source.replace(loop,'for plan in [result["traffic_overlay"]["plans"][n] for n in ("a", "a", "c")]:'),'test_original_eight_predicates_ten_entries_and_real_calls'),('disabled-refusal','summary',source.replace('    if not __debug__:','    if False:',1),'test_optimization_refuses_before_fixture'),('ignore-observation','test',test.replace('    if (counts != Counter(', '    if (False and counts != Counter('),'test_missing_duplicate_and_reordered_traffic_iterations_refuse')]
# The observation guard has several OR arms: disabling all of it must still be a compiling coherent edit.
start=test.index('    if (counts != Counter(');end=test.index('    return {',start)
mutants[-1]=('ignore-observation','test',test[:start]+test[end:],'test_missing_duplicate_and_reordered_traffic_iterations_refuse')
mutants += [('ignore-row-identity','test',test.replace("or plans != [['a'], ['b'], ['c']]",'or False'),'test_missing_duplicate_and_reordered_traffic_iterations_refuse'),('removed-caller','workflow',workflow.replace('python3 tools/run_agreement_summary_contract.py','true'),'test_mandatory_ci_caller'),('masked-caller','workflow',workflow.replace('      - name: Agreement-summary self-test assertion admission (CPU-only)\n','      - name: Agreement-summary self-test assertion admission (CPU-only)\n        if: false\n'),'test_mandatory_ci_caller'),('ignore-discovery','runner',runner.replace('        discovered = discovered_ids(suite)','        discovered = sorted(REQUIRED_IDS)'),'test_missing_replaced_unrelated_duplicate_discovery_refuses'),('ignore-execution','runner',runner.replace('        return 1\n    fields =','        pass\n    fields =').replace('        return 1\n    original =','        pass\n    original ='),'test_missing_duplicate_or_no_success_execution_refuses'),('ignore-outcome','runner',runner.replace('        return 1\n    original =','        pass\n    original ='),'test_skipped_failed_expected_failure_and_unexpected_success_refuse')]
env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1');env.pop('PYTHONOPTIMIZE',None);rows=[]
code='''import pathlib,sys,importlib.util,unittest
sys.path.insert(0,sys.argv[1]);import test_agreement_summary_contract as fixtures
kind,path,method=sys.argv[2:]
if kind in ('test','runner'):
 name='test_agreement_summary_contract' if kind=='test' else 'run_agreement_summary_contract'
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m)
 if kind=='test':
  fixtures=m;fixtures.ROOT=pathlib.Path(sys.argv[1]).parent;fixtures.SUMMARY=fixtures.ROOT/'tools/summarize_hy3_plan_agreement.py'
 else:fixtures.runner=m
elif kind=='summary':fixtures.SUMMARY=pathlib.Path(path)
else:
 class Root:
  def __truediv__(self,child):
   if child=='.github/workflows/ci.yml':return pathlib.Path(path)
   raise ValueError('unexpected fixture path')
 fixtures.ROOT=Root()
r=unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite([fixtures.AgreementSummaryTests(method)]));sys.exit(0 if r.wasSuccessful() else 1)
'''
with tempfile.TemporaryDirectory(prefix='agreement-red-') as owned:
 for name,kind,changed,method in mutants:
  original={'summary':source,'test':test,'runner':runner,'workflow':workflow}[kind];assert changed!=original,name
  path=Path(owned)/(name+('.yml' if kind=='workflow' else '.py'));path.write_text(changed)
  if kind!='workflow':compile(changed,str(path),'exec')
  r=subprocess.run([sys.executable,'-c',code,str(root/'tools'),kind,str(path),method],env=env,capture_output=True,timeout=20)
  raw=r.stdout+r.stderr;(out/(name+'.log')).write_bytes(raw)
  assert r.returncode==1 and b'FAIL:' in raw and b'ERROR:' not in raw,(name,raw.decode())
  rows.append({'control':name,'method':method,'exit':r.returncode,'source_sha256':hashlib.sha256(changed.encode()).hexdigest(),'raw_sha256':hashlib.sha256(raw).hexdigest()})
(out/'coherent-reds.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps({'coherent_reds':len(rows),'fixture_errors':0,'owned_roots_removed':True}))
