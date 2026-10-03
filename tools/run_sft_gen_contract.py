#!/usr/bin/env python3
"""Admit the original SFT generator CPU controls and their caller-policy controls."""

from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest


ORIGINAL_METHODS = (
    'test_battery_has_six_shapes_and_four_task_kinds',
    'test_all_fixture_sources_compile_and_scan_clean',
    'test_secret_scanner_detects_assigned_and_prefixed_keys',
    'test_content_hash_ignores_ids_times_and_usage',
    'test_event_parser_requires_json_objects_and_one_session',
    'test_prepare_workspace_accepts_precreated_empty_directory',
    'test_file_stdout_capture_preserves_large_single_write',
    'test_opencode_config_requires_exact_provider_pin',
    'test_tool_audit_rejects_rust_and_accelerator_commands',
)
POLICY_METHODS = (
    'test_original_control_inventory',
    'test_empty_missing_replaced_unrelated_duplicate_discovery',
    'test_missing_and_equal_count_duplicate_execution',
    'test_non_successful_results_refuse',
    'test_optimization_refuses_before_fixture_and_import',
    'test_mandatory_caller_and_masking_refusals',
    'test_original_control_call_scope_and_generation_guard',
    'test_private_fixture_boundary_and_cleanup',
    'test_generator_selection_remains_full',
)
GROUPS = {
    ('test_sft_gen', 'SftGeneratorTests'): ORIGINAL_METHODS,
    ('test_sft_gen_contract', 'SftGenContractTests'): POLICY_METHODS,
}
REQUIRED_IDS = frozenset(f'{module}.{name}.{method}' for (module, name), methods
                         in GROUPS.items() for method in methods)
ORIGINAL_IDS = frozenset(f'test_sft_gen.SftGeneratorTests.{method}' for method in ORIGINAL_METHODS)


def control_id(case):
    key = (type(case).__module__, type(case).__name__)
    if key not in GROUPS or type(case) is not getattr(sys.modules.get(key[0]), key[1], None):
        raise ValueError('unrelated control type')
    identity = f'{key[0]}.{key[1]}.{case._testMethodName}'
    if case.id() != identity:
        raise ValueError('control identity does not match actual method')
    return identity


def discovered_ids(suite):
    identities, active = [], set()
    def visit(item):
        if isinstance(item, unittest.TestSuite):
            if id(item) in active:
                raise ValueError('cyclic control suite')
            active.add(id(item))
            for child in item:
                visit(child)
            active.remove(id(item))
        else:
            identities.append(control_id(item))
    visit(suite)
    if len(identities) != len(set(identities)):
        raise ValueError('duplicate discovered control identity')
    if set(identities) != REQUIRED_IDS:
        raise ValueError('missing or replaced required control identity')
    return identities


def run(suite, *, stream=sys.stderr):
    try:
        discovered = discovered_ids(suite)
    except ValueError as error:
        print('SFT CPU contract: FAIL: discovery: ' + str(error), file=stream)
        return 1
    executed = []
    class Result(unittest.TextTestResult):
        def startTest(self, test):
            executed.append(control_id(test))
            super().startTest(test)
    result = unittest.TextTestRunner(stream=stream, verbosity=1, resultclass=Result).run(suite)
    if (len(executed) != len(set(executed)) or set(executed) != set(discovered)
            or result.testsRun != len(discovered)):
        print('SFT CPU contract: FAIL: executed identities do not match discovery', file=stream)
        return 1
    if not result.wasSuccessful() or result.skipped or result.expectedFailures:
        print(f'SFT CPU contract: FAIL: executed={result.testsRun} skipped={len(result.skipped)} '
              f'expected_failures={len(result.expectedFailures)}', file=stream)
        return 1
    original = [identity for identity in executed if identity in ORIGINAL_IDS]
    print(f'SFT CPU contract: PASS: original={len(original)} executed={result.testsRun} '
          f'discovered={len(discovered)} unique={len(set(executed))} skipped=0 expected_failures=0', file=stream)
    print('SFT CPU contract identities: ' + json.dumps({'discovered': discovered,
          'executed': executed, 'original': original}, sort_keys=True), file=stream)
    return 0


@contextmanager
def owned_fixture():
    base = os.environ.get('RUNNER_TEMP') or tempfile.gettempdir()
    with tempfile.TemporaryDirectory(prefix='memra-sft-cpu-', dir=base) as directory:
        root = Path(directory)
        hooks = root / 'hooks'
        hooks.mkdir()
        config = root / 'gitconfig'
        config.write_text('')
        changes = {'TMPDIR': str(root), 'GIT_CONFIG_NOSYSTEM': '1',
                   'GIT_CONFIG_GLOBAL': str(config), 'GIT_CONFIG_COUNT': '1',
                   'GIT_CONFIG_KEY_0': 'core.hooksPath', 'GIT_CONFIG_VALUE_0': str(hooks)}
        previous = {key: os.environ.get(key) for key in (*changes, 'GIT_CONFIG_PARAMETERS')}
        old_tempdir = tempfile.tempdir
        try:
            os.environ.update(changes)
            os.environ.pop('GIT_CONFIG_PARAMETERS', None)
            tempfile.tempdir = str(root)
            yield root
        finally:
            tempfile.tempdir = old_tempdir
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


def block_generation(module):
    def refused(*args, **kwargs):
        raise RuntimeError('SFT generation/provider invocation is outside CPU control scope')
    for name in ('generate_one', 'opencode_version', 'main'):
        setattr(module, name, refused)


def discover():
    tools = Path(__file__).resolve().parent
    suites = [unittest.defaultTestLoader.discover(str(tools), pattern=pattern)
              for pattern in ('test_sft_gen.py', 'test_sft_gen_contract.py')]
    original = sys.modules['test_sft_gen'].SftGeneratorTests
    setup = original.setUpClass
    def guarded_setup():
        setup()
        block_generation(original.tool)
    original.setUpClass = guarded_setup
    return unittest.TestSuite(suites)


def main():
    if not __debug__:
        raise RuntimeError('SFT generator admission requires enabled assertions')
    with owned_fixture():
        return run(discover())


if __name__ == '__main__':
    sys.exit(main())
