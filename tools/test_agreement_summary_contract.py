"""Exercise every original summary predicate, traffic iteration, and CPU caller."""
import ast
from collections import Counter
from contextlib import ExitStack
import dis
import hashlib
import io
import json
import os
from pathlib import Path
import re
import runpy
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import run_agreement_summary_contract as runner
import validation_plan as vp

ROOT = Path(__file__).resolve().parent.parent
SUMMARY = ROOT / 'tools/summarize_hy3_plan_agreement.py'
REFUSAL = 'agreement summary self-test requires enabled assertions'
PREDICATES = (
    'result["model"]["expert_count"] == 3',
    'result["consensus"]["stable_pruned_experts"] == 0',
    'result["consensus"]["variable_prune_experts"] == 2',
    'result["layers"]["2"]["all_state_agreement_fraction"] == 8 / 9',
    'result["traffic_overlay"]["stable_pruned_router_weight_mass_fraction"] == 0',
    'abs(sum(plan["state_router_weight_mass_fraction"].values()) - 1) < 1e-12',
    'len(result["pairwise"]) == 3',
    'csv_path.read_text().splitlines()[0].startswith("layer,all_state_agreement_fraction")',
)
COMMANDS = ('python3 tools/summarize_hy3_plan_agreement.py --self-test',
            'python3 tools/run_agreement_summary_contract.py')
STEP = '      - name: Agreement-summary self-test assertion admission (CPU-only)\n'


def observe(ns, source):
    if not __debug__:
        raise RuntimeError('agreement observation requires enabled assertions')
    function = ns['self_test']
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'self_test')
    sites = sorted((n for n in ast.walk(node) if isinstance(n, ast.Assert)), key=lambda n: n.lineno)
    if [ast.dump(n.test) for n in sites] != [ast.dump(ast.parse(p, mode='eval').body) for p in PREDICATES]:
        raise ValueError('all eight original assertion predicates are required')
    instructions = list(dis.get_instructions(function))
    entries = {}
    for index, site in enumerate(sites):
        candidates = [n.offset for n in instructions if n.positions.lineno == site.lineno
                      and n.opname not in ('NOP', 'CACHE', 'EXTENDED_ARG')]
        if not candidates:
            raise ValueError('missing original predicate entry')
        entries[min(candidates)] = index
    if len(entries) != 8:
        raise ValueError('eight distinct original predicate entries are required')
    counts, order, plans, builds, expands, csv_calls = Counter(), [], [], [], [], []
    saved = {name: ns[name] for name in ('build_summary', 'expand_plan', 'write_layer_csv')}
    result_holder = []
    def expand(path):
        result = saved['expand_plan'](path)
        expands.append((path.name, result['name'], set(key[2] for key in result['states'])))
        return result
    def build(paths, scores=None):
        result = saved['build_summary'](paths, scores)
        builds.append(([p.name for p in paths], scores.name if scores else None))
        result_holder.append(result)
        return result
    def csv_write(result, path):
        csv_calls.append((path.name, bool(result_holder) and result is result_holder[0]))
        return saved['write_layer_csv'](result, path)
    def trace(frame, event, arg):
        if frame.f_code is function.__code__:
            if event in ('call', 'line'):
                frame.f_trace_opcodes = True
            elif event == 'opcode' and frame.f_lasti in entries:
                index = entries[frame.f_lasti]
                counts[index] += 1
                order.append(index)
                if index == 5:
                    row = frame.f_locals.get('plan')
                    matches = [name for name, value in result_holder[0]['traffic_overlay']['plans'].items()
                               if value is row] if result_holder else []
                    plans.append(matches)
        return trace
    previous = sys.gettrace()
    caller_frame = sys._getframe()
    previous_opcodes = caller_frame.f_trace_opcodes
    previous_local_trace = caller_frame.f_trace
    previous_lines = caller_frame.f_trace_lines
    caller_frame.f_trace_opcodes = True
    ns.update(build_summary=build, expand_plan=expand, write_layer_csv=csv_write)
    try:
        sys.settrace(trace)
        function()
    finally:
        sys.settrace(previous)
        caller_frame.f_trace_opcodes = previous_opcodes
        caller_frame.f_trace = previous_local_trace
        caller_frame.f_trace_lines = previous_lines
        ns.update(saved)
    if (counts != Counter({0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 3, 6: 1, 7: 1})
            or order != [0, 1, 2, 3, 4, 5, 5, 5, 6, 7]
            or plans != [['a'], ['b'], ['c']]
            or builds != [(['a.json', 'b.json', 'c.json'], 'scores.json')]
            or expands != [(name + '.json', name, {'gate', 'up', 'down'}) for name in ('a', 'b', 'c')]
            or csv_calls != [('layers.csv', True)]):
        raise ValueError('all eight predicates, ten entries, a/b/c iterations and real summary/CSV calls must execute: ' + repr((dict(counts), order, plans, builds, expands, csv_calls)))
    return {'sites': 8, 'entries': sum(counts.values()), 'plans': ['a', 'b', 'c'],
            'counts': dict(counts), 'order': order, 'builds': builds, 'csv': csv_calls}


def check_caller(text):
    text = '\n'.join(line for line in text.splitlines() if not line.lstrip().startswith('#'))
    job = re.search(r'(?ms)^  gates:\n(.*?)(?=^  [a-z][a-z_-]*:|\Z)', text)
    header = job[1].split('    steps:', 1)[0] if job else ''
    if '    if: ${{ !cancelled() }}' not in header or re.search(r'(?m)^\s+continue-on-error:', header):
        raise ValueError('mandatory gates job is masked or missing')
    if text.count(STEP) != 1:
        raise ValueError('mandatory summary caller is missing or duplicated')
    step = re.search(re.escape(STEP) + r'(.*?)(?=^      - |\Z)', job[1], re.S | re.M)
    if not step or [line.strip() for line in step[1].splitlines() if line.strip()] != ['run: |', *COMMANDS]:
        raise ValueError('mandatory summary caller changed or failure swallowed')


def namespace():
    return runpy.run_path(str(SUMMARY))['self_test'].__globals__


def cases():
    cls = sys.modules[runner.MODULE].AgreementSummaryTests
    return [cls(name) for name in runner.REQUIRED_METHODS]


class AgreementSummaryTests(unittest.TestCase):
    def command(self, args, *, optimize=None):
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
        env.pop('PYTHONOPTIMIZE', None)
        if optimize is not None: env['PYTHONOPTIMIZE'] = optimize
        return subprocess.run([sys.executable, *args], cwd=ROOT, env=env,
                              text=True, capture_output=True, timeout=20)

    def test_ordinary_selftest_cli(self):
        r = self.command([str(SUMMARY), '--self-test'])
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(r.stdout.strip(), 'Hy3 smart plan agreement self-test: PASS')

    def test_original_eight_predicates_ten_entries_and_real_calls(self):
        before = (sys.gettrace(), sys._getframe().f_trace, sys._getframe().f_trace_opcodes, sys._getframe().f_trace_lines)
        try: result = observe(namespace(), SUMMARY.read_text())
        except ValueError as error: self.fail(str(error))
        after = (sys.gettrace(), sys._getframe().f_trace, sys._getframe().f_trace_opcodes, sys._getframe().f_trace_lines)
        self.assertEqual(after, before)
        self.assertEqual(result['sites'], 8)
        self.assertEqual(result['entries'], 10)
        self.assertEqual(result['plans'], ['a', 'b', 'c'])

    def test_each_wrong_predicate_fails_and_restores(self):
        ns = namespace(); real = ns['build_summary']; csv_real = ns['write_layer_csv']
        targets = [('model',), ('stable',), ('variable',), ('layer',), ('mass',),
                   ('plan', 'a'), ('plan', 'b'), ('plan', 'c'), ('pairs',), ('csv',)]
        for target in targets:
            with self.subTest(target=target):
                def wrong(paths, scores=None):
                    result = real(paths, scores)
                    if target[0] == 'model': result['model']['expert_count'] = 0
                    elif target[0] == 'stable': result['consensus']['stable_pruned_experts'] = 1
                    elif target[0] == 'variable': result['consensus']['variable_prune_experts'] = 0
                    elif target[0] == 'layer': result['layers']['2']['all_state_agreement_fraction'] = 0
                    elif target[0] == 'mass': result['traffic_overlay']['stable_pruned_router_weight_mass_fraction'] = 1
                    elif target[0] == 'plan': result['traffic_overlay']['plans'][target[1]]['state_router_weight_mass_fraction'] = {}
                    elif target[0] == 'pairs': result['pairwise'] = []
                    return result
                def wrong_csv(result, path):
                    csv_real(result, path); path.write_text('wrong-header\n')
                with mock.patch.dict(ns, build_summary=wrong, write_layer_csv=wrong_csv if target[0] == 'csv' else csv_real):
                    with self.assertRaises(AssertionError): ns['self_test']()
                self.assertEqual(observe(ns, SUMMARY.read_text())['entries'], 10)

    def test_missing_duplicate_and_reordered_traffic_iterations_refuse(self):
        source = SUMMARY.read_text()
        original = 'for plan in result["traffic_overlay"]["plans"].values():'
        for changed in ('for plan in list(result["traffic_overlay"]["plans"].values())[:2]:',
                        'for plan in [result["traffic_overlay"]["plans"][name] for name in ("a", "a", "c")]:',
                        'for plan in [result["traffic_overlay"]["plans"][name] for name in ("a", "c", "b")]:'):
            altered = source.replace(original, changed); self.assertNotEqual(altered, source)
            ns = {'__name__': 'iteration_control'}; exec(compile(altered, str(SUMMARY), 'exec'), ns)
            with self.assertRaisesRegex(ValueError, 'ten entries'): observe(ns, altered)

    def test_projection_fixture_and_original_assertions_preserved(self):
        tree = ast.parse(SUMMARY.read_text())
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'self_test')
        fixture = next(n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == 'make_plan')
        loop = next(n for n in ast.walk(fixture) if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == 'projection')
        self.assertEqual(ast.dump(loop.iter), ast.dump(ast.Name(id='PROJECTIONS', ctx=ast.Load())))
        loop.iter = ast.Subscript(value=ast.Name(id='PROJECTIONS', ctx=ast.Load()), slice=ast.Slice(upper=ast.Constant(2)), ctx=ast.Load())
        source = ast.unparse(ast.fix_missing_locations(tree))
        ns = {'__name__': 'projection_control'}
        exec(compile(source, str(SUMMARY), 'exec'), ns)
        with self.assertRaisesRegex(ValueError, 'incomplete state coverage'):
            ns['self_test']()
        self.assertEqual(observe(namespace(), SUMMARY.read_text())['entries'], 10)

    def test_noop_selftest_or_csv_refuses(self):
        ns = namespace()
        with mock.patch.dict(ns, self_test=lambda: None):
            with self.assertRaises(ValueError): observe(ns, SUMMARY.read_text())
        with mock.patch.dict(ns, write_layer_csv=lambda *args: None):
            with self.assertRaises(FileNotFoundError): ns['self_test']()

    def test_missing_or_replaced_predicate_refuses(self):
        for index in range(8):
            for replacement in (False, True):
                tree = ast.parse(SUMMARY.read_text()); node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'self_test')
                class Mutate(ast.NodeTransformer):
                    seen = 0
                    def visit_Assert(self, item):
                        current = self.seen; self.seen += 1
                        if current == index:
                            if replacement: item.test = ast.Constant(True); return item
                            return ast.copy_location(ast.Pass(), item)
                        return item
                Mutate().visit(node); source = ast.unparse(ast.fix_missing_locations(tree))
                ns = {'__name__': 'predicate_control'}; exec(compile(source, str(SUMMARY), 'exec'), ns)
                with self.assertRaisesRegex(ValueError, 'eight original'): observe(ns, source)

    def test_optimization_refuses_before_fixture(self):
        for script, method, args, refusal in ((SUMMARY, 'self_test', ['--self-test'], REFUSAL),
                (ROOT/'tools/run_agreement_summary_contract.py', 'main', [], 'agreement-summary admission controls require enabled assertions')):
            code = (f'import runpy,tempfile,unittest;m=runpy.run_path({str(script)!r});'
                    'tempfile.TemporaryDirectory=lambda **kw:(_ for _ in ()).throw(RuntimeError("FIXTURE_ENTERED"));'
                    'unittest.defaultTestLoader.discover=lambda *a,**kw:(_ for _ in ()).throw(RuntimeError("DISCOVERY_ENTERED"));'
                    f'm[{method!r}]()')
            for flag in ('-O', '-OO'):
                for invocation in ([flag,str(script),*args], [flag,'-c',code]):
                    r=self.command(invocation); self.assertNotEqual(r.returncode,0)
                    self.assertEqual(r.stderr.splitlines()[-1],'RuntimeError: '+refusal)
                    self.assertNotIn('PASS',r.stdout)
            for value in ('1','2'):
                r=self.command([str(script),*args],optimize=value);self.assertNotEqual(r.returncode,0);self.assertIn(refusal,r.stderr)

    def test_mandatory_ci_caller(self):
        try: check_caller((ROOT/'.github/workflows/ci.yml').read_text())
        except ValueError as error: self.fail(str(error))

    def test_removed_masked_swallowed_caller_refuses(self):
        text=(ROOT/'.github/workflows/ci.yml').read_text()
        variants=[text.replace(command,replacement) for command in COMMANDS for replacement in ('true',command+' || true',command.replace('python3','python3 -O'))]
        variants += [text.replace(STEP,STEP+line) for line in ('        if: false\n','        continue-on-error: true\n')]
        variants += [text.replace('    if: ${{ !cancelled() }}','    if: false',1),text.replace('  gates:\n','  gates:\n    continue-on-error: true\n',1),text.replace(STEP,'# removed\n')]
        for changed in variants:
            self.assertNotEqual(changed,text)
            with self.assertRaises(ValueError):check_caller(changed)

    def test_ordinary_summary_and_csv_bytes_unchanged_under_optimization(self):
        # Capture only the self-test-owned files, then exercise the ordinary CLI.
        ns=namespace(); real=ns['write_layer_csv']; captured=[]
        with tempfile.TemporaryDirectory(prefix='agreement-cli-') as folder:
            root=Path(folder)
            def capture(result,path):
                real(result,path)
                for p in path.parent.glob('*.json'): (root/p.name).write_bytes(p.read_bytes())
                captured.append(result)
            with mock.patch.dict(ns,write_layer_csv=capture): ns['self_test']()
            self.assertEqual(len(captured),1)
            output=root/'out.json';csv_path=root/'out.csv'
            args=[str(SUMMARY),*[str(root/(n+'.json')) for n in ('a','b','c')],'--retention-scores',str(root/'scores.json'),'--out',str(output),'--layer-csv',str(csv_path)]
            baseline=None
            for flags in ([],['-O'],['-OO']):
                r=self.command([*flags,*args]);self.assertEqual(r.returncode,0,r.stderr)
                value=(output.read_bytes(),csv_path.read_bytes())
                if baseline is None: baseline=value
                self.assertEqual(value,baseline)
                data=json.loads(value[0]);self.assertEqual([p['name'] for p in data['inputs']],['a','b','c'])
                for p in data['inputs']:self.assertEqual(p['sha256'],hashlib.sha256((root/(p['name']+'.json')).read_bytes()).hexdigest())

    def test_summary_change_retains_full_native_selection(self):
        tree=vp.Tree(ROOT,'HEAD');p=vp.make_plan(['tools/summarize_hy3_plan_agreement.py'],tree,tree)
        self.assertEqual(p['mode'],'full');self.assertTrue(all(p['jobs'].values()));self.assertEqual(p['native']['scope'],'full');self.assertFalse(p['native']['qualification'])

    def test_exact_control_inventory(self):
        names = unittest.defaultTestLoader.getTestCaseNames(type(self))
        self.assertEqual(set(names), set(runner.REQUIRED_METHODS))
        self.assertEqual(len(names), 16)

    def test_missing_replaced_unrelated_duplicate_discovery_refuses(self):
        class Other(unittest.TestCase):
            def runTest(self):
                raise AssertionError('unrelated must not run')
        probes = [[], cases()[:-1] + [Other()], cases()[:-1] + [cases()[0]],
                  cases() + [cases()[0]], [Other() for _ in range(16)]]
        probes += [[case for case in cases() if case.id() != missing.id()] for missing in cases()]
        with ExitStack() as stack:
            for name in runner.REQUIRED_METHODS:
                stack.enter_context(mock.patch.object(type(self), name, lambda case: None))
            for probe in probes:
                output = io.StringIO()
                self.assertEqual(runner.run(unittest.TestSuite(probe), stream=output), 1)
                self.assertIn('FAIL: discovery:', output.getvalue())
                self.assertNotIn('Ran ', output.getvalue())

    def test_missing_duplicate_or_no_success_execution_refuses(self):
        class Empty(unittest.TestSuite):
            def run(self, result, debug=False):
                return result
        class Duplicate(unittest.TestSuite):
            def run(self, result, debug=False):
                self._tests[0](result)
                self._tests[0](result)
                for case in self._tests[2:]:
                    case(result)
                return result
        class NoOutcome(unittest.TestSuite):
            def run(self, result, debug=False):
                for case in self:
                    result.startTest(case)
                    result.stopTest(case)
                return result
        class FloatCount(unittest.TestSuite):
            def run(self, result, debug=False):
                super().run(result, debug)
                result.testsRun = float(result.testsRun)
                return result
        class Malformed(unittest.TestSuite):
            def run(self, result, debug=False):
                super().run(result, debug)
                result.skipped = None
                return result
        with ExitStack() as stack:
            for name in runner.REQUIRED_METHODS:
                stack.enter_context(mock.patch.object(type(self), name, lambda case: None))
            for cls in (Empty, Duplicate, NoOutcome, FloatCount, Malformed):
                with self.subTest(suite=cls.__name__):
                    self.assertEqual(runner.run(cls(cases()), stream=io.StringIO()), 1)

    def test_skipped_failed_expected_failure_and_unexpected_success_refuse(self):
        for mode, expected in (('healthy', 0), ('skip', 1), ('failure', 1), ('error', 1),
                               ('expected-failure', 1), ('unexpected-success', 1)):
            with self.subTest(mode=mode), ExitStack() as stack:
                for name in runner.REQUIRED_METHODS:
                    stack.enter_context(mock.patch.object(type(self), name, lambda case: None))
                def control(case):
                    if mode == 'skip':
                        case.skipTest('planted skip')
                    if mode in ('failure', 'expected-failure'):
                        case.fail('planted failure')
                    if mode == 'error':
                        raise RuntimeError('planted error')
                if mode in ('expected-failure', 'unexpected-success'):
                    control = unittest.expectedFailure(control)
                stack.enter_context(mock.patch.object(type(self), runner.REQUIRED_METHODS[0], control))
                self.assertEqual(runner.run(unittest.TestSuite(cases()), stream=io.StringIO()), expected)


if __name__ == '__main__':
    sys.exit(runner.main())
