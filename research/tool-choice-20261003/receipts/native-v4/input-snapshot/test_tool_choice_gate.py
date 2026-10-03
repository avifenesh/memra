import copy
import importlib.util
import json
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('tool_choice_gate', Path(__file__).with_name('tool-choice-gate.py'))
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


def frames(args=None, name='weather', content=''):
    delta = {'content': content} if args is None else {'tool_calls': [
        {'index': 0, 'id': 'call_fixture', 'function': {'name': name, 'arguments': json.dumps(args)}}]}
    return [{'choices': [{'index': 0, 'delta': delta}]},
        {'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop' if args is None else 'tool_calls'}],
         'usage': {'prompt_tokens': 123, 'completion_tokens': 45}}]


def report():
    r = {'context': {'candidate_sha256': 'candidate', 'baseline_sha256': 'baseline',
        'source': 'source', 'helper_sha256': 'helper', 'gpu_uuid': 'GPU-1a3cbffc-29df-926c-df5c-29b4c210ef5d'}, 'models': {}}
    r['context']['artifacts'] = {name: {'sha256': name+'-artifact','bytes':1234} for name in ['qwen','gemma']}
    args = {'city': 'Paris', 'days': 1, 'note': '</tool_call><tool_call|>'}
    for name in ['qwen', 'gemma']:
        cases = {}
        for regression in ['auto', 'none']:
            case = {'status': 200, 'done': 1, 'request': {'tool_choice': regression},
                'frames': frames({'city': 'Paris', 'days': 1}) if regression == 'auto' else frames(content='A plain answer.')}
            for label in ['baseline', 'candidate']:
                cases[label + '.' + regression] = copy.deepcopy(case)
        for arm, extra in [('required', {'tool_choice': 'required'}),
            ('named', {'tool_choice': {'type': 'function', 'function': {'name': 'weather'}}}),
            ('single-auto', {'tool_choice': 'auto'})]:
            body = gate.request_body(parallel_tool_calls=False, **extra)
            cases[arm] = {'status': 200, 'done': 1, 'request': body, 'frames': frames(args), 'masked_steps': 45}
        for arm, param in [('unknown-name', 'tool_choice'), ('invalid-schema', 'tool_choice'), ('parallel-true', 'parallel_tool_calls')]:
            cases[arm] = {'status': 400, 'body': {'error': {'param': param}}}
        r['models'][name] = {'sha256': name + '-artifact', 'bytes': 1234,
            'capabilities': {'forced_tool_calls': True, 'parallel_tool_calls': name == 'qwen'}, 'cases': cases}
    return r


class ReceiptControls(unittest.TestCase):
    def setUp(self):
        self.report = report()
        self.context = copy.deepcopy(self.report['context'])

    def fails(self, edge):
        with self.assertRaisesRegex(ValueError, edge.replace('.', r'\.')):
            gate.validate(self.report, self.context)

    def test_positive_composite_covers_all_edges(self):
        self.assertEqual(len(gate.validate(self.report, self.context)), 15)

    def test_wall_clock_timing_is_retained_but_not_an_identity_oracle(self):
        self.report['models']['qwen']['cases']['baseline.auto']['frames'][-1]['usage']['elapsed_s'] = 1.2
        self.report['models']['qwen']['cases']['candidate.auto']['frames'][-1]['usage']['elapsed_s'] = 2.3
        gate.validate(self.report, self.context)
        self.assertEqual(self.report['models']['qwen']['cases']['candidate.auto']['frames'][-1]['usage']['elapsed_s'],2.3)

    def test_token_and_cache_counts_remain_mandatory_identity_controls(self):
        for key in ['prompt_tokens','completion_tokens','total_tokens','prompt_tokens_details']:
            with self.subTest(key=key):
                self.report = report()
                value = {'cached_tokens':1} if key=='prompt_tokens_details' else 999
                self.report['models']['qwen']['cases']['candidate.auto']['frames'][-1]['usage'][key] = value
                self.fails('qwen.auto.identity')

    def test_committed_native_receipt_replay_and_mandatory_controls(self):
        import hashlib
        root=Path(__file__).resolve().parents[1]/'research/tool-choice-20261003/receipts/native-v3'
        actual=json.loads((root/'report.json').read_text());context=json.loads((root/'expected-context.json').read_text())
        self.assertEqual(len(gate.validate(actual,context)),15)
        self.assertEqual(len(gate.red_controls(actual,context)),8)
        hashes=json.loads((root/'file-sha256.json').read_text())
        for name,expected in hashes.items():
            self.assertEqual(hashlib.sha256((root/name).read_bytes()).hexdigest(),expected,name)
        spec=importlib.util.spec_from_file_location('native_coverage',Path(__file__).with_name('validation_coverage.py'))
        coverage=importlib.util.module_from_spec(spec);spec.loader.exec_module(coverage)
        selected=json.loads((root/'coverage-plan.json').read_text());results=json.loads((root/'coverage-results.json').read_text())
        admitted=coverage.validate_results(selected,results,context,root/'input-snapshot')
        self.assertEqual(admitted['edges'],23)
        self.assertFalse(admitted['qualification'])

    def test_source_context_change_refuses(self):
        self.report['context']['source'] = 'different'
        self.fails('context.binding')

    def test_missing_mask_witness_refuses(self):
        self.report['models']['qwen']['cases']['required']['masked_steps'] = 0
        self.fails('qwen.required.mask')

    def test_wrong_name_refuses(self):
        self.report['models']['qwen']['cases']['required']['frames'][0]['choices'][0]['delta']['tool_calls'][0]['function']['name'] = 'undeclared'
        self.fails('call.declared_name')

    def test_valid_other_function_cannot_satisfy_named_choice(self):
        call = self.report['models']['qwen']['cases']['named']['frames'][0]['choices'][0]['delta']['tool_calls'][0]
        call['function'] = {'name': 'clock', 'arguments': '{"hour":3}'}
        self.fails('qwen.named.selection')

    def test_wrong_schema_and_bool_integer_refuse(self):
        for args, edge in [({'city': 'Berlin', 'days': 1, 'note': '</tool_call><tool_call|>'}, 'schema.city'),
            ({'city': 'Paris', 'days': True, 'note': '</tool_call><tool_call|>'}, 'schema.days'),
            ({'city': 'Paris', 'days': 1, 'note': 'wrong'}, 'schema.quoted_delimiter'),
            ({'city': 'Paris', 'days': 1}, 'schema.keys')]:
            with self.subTest(edge=edge):
                self.report = report()
                self.report['models']['qwen']['cases']['required']['frames'] = frames(args)
                self.fails(edge)

    def test_integral_json_numbers_remain_valid(self):
        self.report['models']['qwen']['cases']['required']['frames'] = frames(
            {'city': 'Paris', 'days': 1.0, 'note': '</tool_call><tool_call|>'})
        gate.validate(self.report, self.context)

    def test_duplicate_calls_refuse(self):
        calls = self.report['models']['gemma']['cases']['single-auto']['frames'][0]['choices'][0]['delta']['tool_calls']
        other = copy.deepcopy(calls[0]); other['index'] = 1; other['id'] = 'second'; calls.append(other)
        self.fails('gemma.single-auto.single')

    def test_auto_and_none_identity_cannot_be_omitted(self):
        for regression in ['auto', 'none']:
            with self.subTest(regression=regression):
                self.report = report()
                self.report['models']['qwen']['cases']['candidate.' + regression]['frames'][0]['choices'][0]['delta']['content'] = 'changed'
                self.fails('qwen.' + regression + '.identity')

    def test_wrong_request_policy_does_not_pass_on_plausible_output(self):
        self.report['models']['qwen']['cases']['required']['request']['tool_choice'] = 'auto'
        self.fails('qwen.required.request_policy')

    def test_missing_done_wrong_terminal_and_duplicate_terminal_refuse(self):
        for mutation, edge in [('done', 'qwen.named.stream'), ('finish', 'qwen.named.finish'), ('duplicate', 'wire.double_terminal')]:
            with self.subTest(mutation=mutation):
                self.report = report(); case = self.report['models']['qwen']['cases']['named']
                if mutation == 'done': case['done'] = 0
                elif mutation == 'finish': case['frames'][1]['choices'][0]['finish_reason'] = 'stop'
                else: case['frames'].append(copy.deepcopy(case['frames'][1]))
                self.fails(edge)

    def test_unknown_template_capability_and_refusal_param_must_match(self):
        self.report['models']['gemma']['capabilities']['parallel_tool_calls'] = True
        self.fails('gemma.capability.parallel')
        self.report = report(); self.report['models']['gemma']['cases']['parallel-true']['body']['error']['param'] = 'response_format'
        self.fails('gemma.parallel_true.refusal')

    def test_wire_argument_fragments_assemble_without_frame_token_equivalence(self):
        raw = frames({'city': 'Paris', 'days': 1, 'note': '</tool_call><tool_call|>'})
        call = raw[0]['choices'][0]['delta']['tool_calls'][0]
        args = call['function'].pop('arguments')
        raw.insert(1, {'choices': [{'index': 0, 'delta': {'tool_calls': [{'index': 0, 'function': {'arguments': args[:9]}}]}}]})
        raw.insert(2, {'choices': [{'index': 0, 'delta': {'tool_calls': [{'index': 0, 'function': {'arguments': args[9:]}}]}}]})
        gate.check_call(gate.assemble(raw)['calls'][0])
        self.assertEqual(gate.assemble(raw)['usage']['completion_tokens'], 45)


if __name__ == '__main__':
    unittest.main()
