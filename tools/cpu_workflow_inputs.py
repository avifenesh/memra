"""Admit only base-known literal CPU steps, without interpreting arbitrary YAML."""

import json
from pathlib import PurePosixPath
import re
import subprocess


POLICY_PATH = 'tools/cpu_workflow_contracts.json'
WORKFLOW_PATH = '.github/workflows/ci.yml'
HELPER_PATH = 'tools/cpu_workflow_inputs.py'
CONTRACT_IDS = ('expert-tier-caller', 'sft-generator-caller', 'score-shard-caller')
COMMANDS = {
    'expert-tier-caller': ('python3 tools/build_expert_tier_plan.py --self-test',
                         'python3 tools/run_expert_tier_contract.py'),
    'sft-generator-caller': ('python3 tools/run_sft_gen_contract.py',),
    'score-shard-caller': ('python3 tools/merge_expert_score_shards.py --self-test',
                         'python3 tools/run_score_shard_contract.py'),
}


def _path(name):
    if (type(name) is not str or not name or name == '.'
            or any(c in name for c in '\r\n\t\0\\')
            or PurePosixPath(name).is_absolute()
            or '..' in PurePosixPath(name).parts
            or PurePosixPath(name).as_posix() != name):
        raise ValueError('noncanonical workflow dependency path')
    return name


def _regular(tree, name):
    name = _path(name)
    # GitTree does not follow symlink ancestors, and LocalTree's shared byte
    # reader opens every ancestor with O_NOFOLLOW. Query the exact leaf: a Git
    # directory pathspec enumerates its children rather than returning itself.
    modes = tree.input_modes(name, recursive=False)
    if modes.get(name) not in ('100644', '100755'):
        raise ValueError('missing or nonregular workflow input: ' + name)
    raw = tree.read_bytes(name)
    if type(raw) is not bytes:
        raise ValueError('workflow input reader did not return bytes: ' + name)
    return raw, modes[name]


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate workflow policy key: ' + key)
        result[key] = value
    return result


def _policy(raw):
    data = json.loads(raw.decode('utf-8'), object_pairs_hook=_unique_object)
    if (type(data) is not dict or set(data) != {'schema', 'contracts'}
            or type(data['schema']) is not int or data['schema'] != 1
            or type(data['contracts']) is not list):
        raise ValueError('unsupported CPU workflow policy')
    rows = data['contracts']
    if [row.get('id') if type(row) is dict else None for row in rows] != list(CONTRACT_IDS):
        raise ValueError('unknown, duplicate or reordered workflow contract')
    names, blocks = set(), set()
    for row in rows:
        if set(row) != {'id', 'block', 'anchor', 'inputs'}:
            raise ValueError('unsupported workflow contract fields')
        block, anchor, inputs = row['block'], row['anchor'], row['inputs']
        if type(block) is not str or '\r' in block or '\t' in block:
            raise ValueError('nonliteral CPU step')
        match = re.fullmatch(
            r'      - name: ([A-Za-z0-9 ():/_-]+)\n'
            r'(?:        run: python3 tools/[a-z0-9_-]+\.py\n'
            r'|        run: \|\n(?:          python3 tools/[a-z0-9_-]+\.py(?: --self-test)?\n)+)\n',
            block)
        if not match or match[1] in names or block in blocks:
            raise ValueError('ambiguous, duplicate or masked CPU step')
        names.add(match[1]); blocks.add(block)
        calls = tuple(line.strip().removeprefix('run: ') for line in block.splitlines()
                      if line.strip().startswith(('python3 ', 'run: python3 ')))
        if calls != COMMANDS[row['id']]:
            raise ValueError('workflow step does not execute its registered CPU commands')
        if (type(anchor) is not str or not re.fullmatch(
                r'      - name: [A-Za-z0-9 ():/_-]+\n', anchor)
                or anchor == block.splitlines(keepends=True)[0]):
            raise ValueError('nonliteral CPU step anchor')
        if type(inputs) is not list or not inputs or len(inputs) != len(set(inputs)):
            raise ValueError('missing or duplicate workflow dependency')
        for name in inputs:
            _path(name)
        commands = re.findall(r'python3 (tools/[a-z0-9_-]+\.py)', block)
        if HELPER_PATH not in inputs or any(name not in inputs for name in commands):
            raise ValueError('CPU step dependency closure omits a caller')
    return rows


def _gates(text):
    if '\r' in text or '\t' in text or '\0' in text:
        raise ValueError('unsupported workflow bytes')
    if text.count('\njobs:\n') != 1:
        raise ValueError('missing or ambiguous jobs mapping')
    jobs_at = text.index('\njobs:\n') + 1
    jobs = [m for m in re.finditer(r'^  ([A-Za-z][A-Za-z0-9_-]*):\n', text, re.M)
            if m.start() > jobs_at]
    if len({m[1] for m in jobs}) != len(jobs):
        raise ValueError('duplicate workflow job')
    positions = [i for i, m in enumerate(jobs) if m[1] == 'gates']
    if len(positions) != 1:
        raise ValueError('missing mandatory gates job')
    i = positions[0]
    start, end = jobs[i].start(), jobs[i + 1].start() if i + 1 < len(jobs) else len(text)
    body = text[start:end]
    if body.count('    steps:\n') != 1:
        raise ValueError('missing or ambiguous gates steps')
    header = body[:body.index('    steps:\n')]
    if (header.count('    needs: changes\n') != 1
            or header.count('    if: ${{ !cancelled() }}\n') != 1):
        raise ValueError('CPU steps are not in mandatory gates')
    # Only the existing block-style shape is admitted. Do not infer semantics
    # from a step-looking line inside a multiline quoted or folded scalar.
    scalar = False
    for line in body.splitlines():
        indent = len(line) - len(line.lstrip(' '))
        if scalar and (not line.strip() or indent > 8):
            continue
        scalar = False
        if indent == 8 and line.startswith('        run: '):
            value = line[13:]
            if value == '|':
                scalar = True
            elif value.startswith(('>', '|', "'", '"', '&', '*', '!')):
                raise ValueError('unsupported gates run scalar')
        if re.match(r'^\s*(?:<<:|[?!&*]|---|%YAML)', line):
            raise ValueError('unsupported gates YAML construct')
        if re.match(r'^ {4,12}(?:- )?[A-Za-z0-9_-]+: [>&*!]', line):
            raise ValueError('ambiguous gates mapping value')
        mapping = re.match(r'^ {4,12}(?:- )?[A-Za-z0-9_-]+: (.*)$', line)
        if mapping and mapping[1].startswith(('{', '[')):
            raise ValueError('flow-style gates mapping value')
        if mapping and mapping[1].startswith(("'", '"')):
            if (len(mapping[1]) < 2 or not mapping[1].endswith(mapping[1][0])
                    or mapping[1].endswith('\\"')):
                raise ValueError('multiline quoted gates scalar')
    step_starts = {start + m.start() for m in re.finditer(r'^      - (?:name|uses|id): ', body, re.M)}
    names = re.findall(r'^      - name: (.*)$', body, re.M)
    if len(names) != len(set(names)):
        raise ValueError('duplicate named gates step')
    return start, end, step_starts


def eligible_additions(base_tree, head_tree):
    """Return added contract IDs, or ValueError so the caller expands validation.

    This answers only the workflow delta. The planner must still inspect every
    other changed input and its compiled consumers/native obligations.
    """
    try:
        base_policy = _regular(base_tree, POLICY_PATH)
        if base_policy != _regular(head_tree, POLICY_PATH):
            raise ValueError('workflow policy changed or bootstrapped in this diff')
        rows = _policy(base_policy[0])
        base_raw, base_mode = _regular(base_tree, WORKFLOW_PATH)
        head_raw, head_mode = _regular(head_tree, WORKFLOW_PATH)
        if base_mode != head_mode:
            raise ValueError('workflow mode changed')
        base, head = base_raw.decode('utf-8'), head_raw.decode('utf-8')
        _gates(base)
        start, end, step_starts = _gates(head)
        additions, spans, locations = [], [], []
        for row in rows:
            block = row['block']
            old_count, new_count = base.count(block), head.count(block)
            if old_count > 1 or new_count > 1 or old_count > new_count:
                raise ValueError('removed or duplicated CPU step: ' + row['id'])
            if not new_count:
                continue
            at = head.index(block)
            if at not in step_starts or not start <= at < at + len(block) <= end:
                raise ValueError('CPU step is outside literal gates steps')
            locations.append(at)
            if old_count:
                continue
            anchor = row['anchor']
            if head.count(anchor) != 1 or base.count(anchor) != 1:
                raise ValueError('missing or ambiguous insertion anchor')
            anchor_at = head.index(anchor)
            if anchor_at not in step_starts or not at < anchor_at < end:
                raise ValueError('CPU step is outside its approved insertion slot')
            between = head[at:anchor_at]
            for approved in rows:
                between = between.replace(approved['block'], '', 1)
            if between:
                raise ValueError('unapproved bytes inside CPU insertion slot')
            for name in row['inputs']:
                if _regular(base_tree, name) != _regular(head_tree, name):
                    raise ValueError('workflow CPU dependency changed: ' + name)
            additions.append(row['id']); spans.append((at, at + len(block)))
        if locations != sorted(locations):
            raise ValueError('approved CPU steps reordered')
        if not additions:
            raise ValueError('no recognized CPU step additions')
        residual = head
        for left, right in sorted(spans, reverse=True):
            residual = residual[:left] + residual[right:]
        if residual.encode('utf-8') != base_raw:
            raise ValueError('workflow changed outside approved CPU additions')
        return additions
    except (OSError, UnicodeError, KeyError, TypeError, subprocess.SubprocessError) as error:
        raise ValueError('workflow inputs unavailable: ' + str(error)) from error
