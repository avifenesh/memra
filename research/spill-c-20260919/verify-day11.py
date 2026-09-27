#!/usr/bin/env python3
"""Replay the day-eleven target-card receipts: plan-derived installer parity and three N=1 cells.

Parity: the catalog digest the installer printed must equal the digest of the day-ten
literal catalog (`blk.N.ffn_{gate,up,down}_exps.weight`, trunk layers then the MTP layer)
recomputed here from memra's own `tensor-census.tsv` (the `model inspect` receipt), and the
bank identity line must be byte-identical to the day-ten exact cell. Cells: gen MATCH with
the day-nine tape, spec K=1..8 PASS on the exact eight-slot bank with the day-nine spec
control, and the seven-slot refusal as the final stderr line with exit 2.
"""
import argparse
import ast
import datetime
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess

LANE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location('day9', LANE / 'verify-day9.py')
DAY9 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DAY9)
DAY7 = DAY9.DAY7
require = DAY7.require
RAW = LANE / 'pro-single-day11'
DAY10 = LANE / 'pro-single-day10-budget'
ROOT = LANE.parents[1]
RECORD = 860160
SLOT = RECORD + 8
MINIMUM = 8 * SLOT
ARTIFACT_SHA = 'df27a780435b7b45c2597536112ea3cb091f8544c3d0c3318d9f4258b31f7adf'
ARTIFACT = '/root/artifacts/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf'
BIN = '/root/wt-c/target-day11/release/'
CATALOG = re.compile(r'^\[experts-via-tier\] catalog blocks=(\d+) banked=(\d+) projections=(\d+) '
                     r'catalog_sha256=([0-9a-f]{64}) records=(\d+) records_sha256=([0-9a-f]{64})$', re.M)
INSTALLED = re.compile(r'^\[experts-via-tier\] installed artifact_sha256=([0-9a-f]{64}) host_slots=(\d+) max_expert_bytes=(\d+)$', re.M)
BUDGET = re.compile(r'^\[experts-via-tier\] gpu_bank_budget bytes=(\d+) slots=(\d+) hard_ceiling=(\d+)$', re.M)
REFUSED = re.compile(r'^REFUSED: experts-via-tier GPU bank budget cannot hold the eight-slot minimum '
                     r'\(requested (\d+), minimum (\d+), ceiling (\d+)\)$')
GPU = re.compile(r'\[expert-gpu-slru\] slots=(\d+) allocated_bytes=(\d+) evictions=(\d+)')
HOST_ROW = re.compile(r'^\[expert-host-slru\] key=(\d+:\d+:\d+) bytes=(\d+) slot=(\d+) hit=(true|false) victim=(-|\d+:\d+:\d+)$', re.M)
STORAGE = re.compile(r'^Quantized\(QuantLayout \{ format: "([A-Za-z0-9_]+)", block_shape: \[([0-9, ]*)\], auxiliaries: \[\] \}\)$')
FLOAT = re.compile(r'^Float\((F32|F16|Bf16|Fp8E4m3)\)$')
GEN_VERDICT = 'prefill argmax=198  decode argmax=198  logit maxdiff=6.482e-1  MATCH'


def raw_bytes(root, rel):
    """A receipt file, or its gzip twin when the raw file was too large to keep uncompressed
    (the 52 MB exact-eight spec log): the bytes hashed are always the decompressed original."""
    path = root / rel
    if path.exists():
        return path.read_bytes()
    packed = path.with_name(path.name + '.gz')
    require(packed.exists(), 'missing receipt ' + str(rel))
    return gzip.decompress(packed.read_bytes())


def read_log(path):
    return raw_bytes(path, 'command.log').decode()


def hashes(root, value):
    """DAY7.hashes with the gzip fallback for `raw_log` entries."""
    if isinstance(value, dict):
        if set(value) == {'path', 'bytes', 'sha256'}:
            rel = Path(value['path'])
            require(not rel.is_absolute() and '..' not in rel.parts, 'escaping receipt')
            raw = raw_bytes(root, rel)
            require(len(raw) == value['bytes'], 'receipt length mismatch')
            require(hashlib.sha256(raw).hexdigest() == value['sha256'], 'receipt hash mismatch')
        for nested in value.values():
            hashes(root, nested)
    elif isinstance(value, list):
        for nested in value:
            hashes(root, nested)


def argv(gate, *flags):
    return ['env', 'MEMRA_MOE_RESIDENT=0', 'MEMRA_NGEN=32', BIN + gate, ARTIFACT, '55', '88', '13',
            '--experts-via-tier', *flags]


def census_rows(path):
    """memra's own GGUF census (model inspect): name -> (shape, storage text, physical bytes)."""
    rows = {}
    lines = path.read_text().splitlines()
    require(lines and lines[0] == 'semantic_name\tphysical_name\tdtype\tshape\tstorage\tphysical_bytes', 'census header')
    for line in lines[1:]:
        name, physical, dtype, shape, storage, physical_bytes = line.split('\t')
        require(name == physical and name not in rows, 'census identity: ' + name)
        quant = STORAGE.match(storage)
        if quant:
            text = quant[1] + '[' + ','.join(part.strip() for part in quant[2].split(',') if part.strip()) + ']'
            require(quant[1] == dtype, 'census dtype disagrees with storage: ' + name)
        else:
            flt = FLOAT.match(storage)
            require(flt, 'unparsed storage: ' + storage)
            text = flt[1]
        rows[name] = (ast.literal_eval(shape), text, int(physical_bytes))
    return rows


def literal_identity(rows, trunk_layers):
    """The day-ten literal catalog in `ExpertBankCatalog::identity()` form: every layer that
    carries the three stacked expert tensors, trunk layers by position, then the MTP layer."""
    layers = sorted({int(m[1]) for name in rows for m in [re.match(r'^blk\.(\d+)\.ffn_gate_exps\.weight$', name)] if m})
    require(layers, 'census has no stacked expert tensors')
    text = ''
    for layer in layers:
        if layer < trunk_layers:
            block = f'trunk:{layer}'
        else:
            require(layer == trunk_layers and f'blk.{layer}.nextn.eh_proj.weight' in rows, 'a non-trunk expert layer is not the MTP block')
            block = 'mtp:0'
        for projection, suffix in [('Gate', 'gate'), ('Up', 'up'), ('Down', 'down')]:
            name = f'blk.{layer}.ffn_{suffix}_exps.weight'
            shape, storage, physical_bytes = rows[name]
            require(shape[-1] == 256 and len(shape) == 3, 'expert tensor shape: ' + name)
            text += f'{block}\t{layer}\t{projection}\t{name}\t{"x".join(str(v) for v in shape)}\t{storage}\t{physical_bytes}\n'
    return text, layers


def envelope(path, capture):
    hashes(path, capture)
    require(capture['qualification'] is False, 'collector is not qualification')
    require(not capture['timed_out'] and capture['parse_error'] is None, 'timed out or unparsable')
    require(capture['gpu_telemetry']['interval_ms'] == 250, 'telemetry cadence')
    require(capture['gpu_power_limits'] == [{'device': '0', 'power.limit': '600.00 W',
            'power.max_limit': '600.00 W'}], 'power envelope changed')
    require(json.loads((path / 'lock.json').read_text()) ==
            {'rig': 'pro-single', 'lock': '/tmp/memra-gpu.lock', 'acquired': True}, 'lock')


def catalog_line(log, expected_blocks, expected_banked):
    lines = CATALOG.findall(log)
    require(len(lines) == 1, 'catalog line missing or repeated')
    blocks, banked, projections, catalog_sha, records, records_sha = lines[0]
    require((int(blocks), int(banked), int(projections)) == (expected_blocks, expected_banked, 3 * expected_blocks), 'catalog counts')
    require(int(records) == 3 * 256 * expected_banked, 'record count')
    return {'blocks': int(blocks), 'banked': int(banked), 'projections': int(projections),
            'catalog_sha256': catalog_sha, 'records': int(records), 'records_sha256': records_sha}


def host_trace(log):
    rows = HOST_ROW.findall(log)
    require(rows and len(rows) == log.count('[expert-host-slru]'), 'host trace')
    seen = set()
    misses = rereads = evictions = 0
    for key, size, slot, hit, victim in rows:
        require(int(size) <= RECORD and int(slot) < 16, 'host extent')
        misses += hit == 'false'
        rereads += hit == 'false' and key in seen
        evictions += victim != '-'
        seen.add(key)
    physical = re.search(r'physical_reads=(\d+) owner_close=Ok\(\(\)\)', log)
    require(physical and int(physical[1]) == misses, 'reads/drain')
    return misses, rereads, evictions


def replay_gen(path, blocks):
    capture = json.loads((path / 'command.capture.json').read_text())
    envelope(path, capture)
    require(capture['exit_code'] == 0 and capture['status'] == 'executed-not-qualified', 'failed cell')
    require(capture['command'] == argv('run-gen'), 'command identity changed')
    log = read_log(path)
    loaded = re.search(r'^loaded qwen35moe \((\d+) trunk layers; optional MTP skipped\)$', log, re.M)
    require(loaded, 'load line')
    trunk = int(loaded[1])
    catalog = catalog_line(log, blocks, trunk)
    require(BUDGET.search(log) is None, 'default budgets must not print a GPU budget line')
    installed = INSTALLED.findall(log)
    require(len(installed) == 1 and installed[0] == (ARTIFACT_SHA, '16', str(RECORD)), 'install identity')
    installed_line = re.search(r'^\[experts-via-tier\] installed .*$', log, re.M)[0]
    require(log.index(catalog_line_text(log)) < log.index(installed_line), 'catalog line must precede the installed line')
    day10 = (DAY10 / 'budget-exact-8-attempt1' / 'command.log').read_text()
    require(re.search(r'^\[experts-via-tier\] installed .*$', day10, re.M)[0] == installed_line, 'installed line differs from day ten')
    require(GEN_VERDICT in log, 'verdict absent')
    tape = DAY7.tokens(log)
    misses, rereads, evictions = host_trace(log)
    gpu = GPU.search(log)
    require(gpu and int(gpu[2]) == int(gpu[1]) * SLOT, 'GPU extents')
    return {'case': path.name, 'n': 1, 'verdict': GEN_VERDICT, 'trunk_layers': trunk, 'catalog': catalog,
            'installed_line': installed_line, 'gpu_slots': int(gpu[1]), 'gpu_evictions': int(gpu[3]),
            'host_evictions': evictions, 'physical_reads': misses, 'rereads': rereads,
            'ended_utc': capture['ended_utc']}, tape


def catalog_line_text(log):
    return re.search(r'^\[experts-via-tier\] catalog .*$', log, re.M)[0]


def replay_spec_exact(path, blocks):
    capture = json.loads((path / 'command.capture.json').read_text())
    envelope(path, capture)
    require(capture['exit_code'] == 0 and capture['status'] == 'executed-not-qualified', 'failed cell')
    require(capture['command'] == argv('run-spec', f'--expert-bank-gpu-bytes={MINIMUM}'), 'command identity changed')
    log = read_log(path)
    catalog = catalog_line(log, blocks, blocks)
    budget = BUDGET.findall(log)
    require(len(budget) == 1 and (int(budget[0][0]), int(budget[0][1])) == (MINIMUM, 8) and int(budget[0][2]) >= MINIMUM, 'budget arithmetic')
    installed = INSTALLED.findall(log)
    require(len(installed) == 1 and installed[0] == (ARTIFACT_SHA, '16', str(RECORD)), 'install identity')
    require('=== SELF-CONSISTENCY PASS ===' in log, 'verdict absent')
    require(re.findall(r'\[generate_spec K=(\d)\]', log) == list('12345678'), 'K ladder')
    acceptance = re.findall(r'acceptance: ([^\n]+)', log)
    require(len(acceptance) == 8 and all('self-consistency: PASS' in row for row in acceptance), 'K row')
    tape = DAY7.tokens(log)
    misses, rereads, evictions = host_trace(log)
    gpu = GPU.search(log)
    require(gpu and (int(gpu[1]), int(gpu[2])) == (8, MINIMUM), 'GPU extents')
    require(int(gpu[3]) > 0 and rereads > 0, 'pressure absent')
    gpu_line = re.search(r'^\[expert-gpu-slru\] slots=8 .*$', log, re.M)[0]
    return {'case': path.name, 'n': 1, 'verdict': '=== SELF-CONSISTENCY PASS ===', 'catalog': catalog,
            'gpu_line': gpu_line, 'gpu_slots': 8, 'gpu_evictions': int(gpu[3]), 'host_evictions': evictions,
            'physical_reads': misses, 'rereads': rereads, 'hard_ceiling_bytes': int(budget[0][2]),
            'ended_utc': capture['ended_utc']}, tape, acceptance


def replay_spec_refusal(path, blocks):
    capture = json.loads((path / 'command.capture.json').read_text())
    envelope(path, capture)
    require(capture['exit_code'] == 2 and capture['status'] == 'refused', 'not a collector refusal')
    require(capture['command'] == argv('run-spec', f'--expert-bank-gpu-bytes={7 * SLOT}'), 'command identity changed')
    log = read_log(path)
    lines = log.splitlines()
    require(lines and lines[-1] == capture['failure_quote'], 'refusal is not the final line')
    match = REFUSED.match(lines[-1])
    require(match, 'refusal text changed')
    requested, minimum, ceiling = (int(v) for v in match.groups())
    require(requested == 7 * SLOT and minimum == MINIMUM and ceiling >= minimum, 'refusal arithmetic')
    catalog = catalog_line(log, blocks, blocks)
    for marker in ['[expert-host-slru]', '[expert-gpu-slru]', '[experts-via-tier] installed',
                   '[experts-via-tier] gpu_bank_budget', '\nloaded ', 'prompt tokens:', 'tokens: [',
                   'SELF-CONSISTENCY', 'Error:']:
        require(marker not in log, 'work after refusal: ' + marker.strip())
    return {'case': path.name, 'n': 1, 'verdict': lines[-1], 'catalog': catalog, 'requested_bytes': requested,
            'minimum_bytes': minimum, 'hard_ceiling_bytes': ceiling, 'ended_utc': capture['ended_utc']}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=RAW)
    parser.add_argument('--cell', help='replay one receipt dir under --root only (self-test); parity is skipped')
    args = parser.parse_args()
    if args.cell:
        path = args.root / args.cell
        blocks = int(re.search(r'blocks=(\d+)', read_log(path))[1])
        if args.cell.startswith('gen-'):
            result = replay_gen(path, blocks)[0]
        elif args.cell.startswith('spec-refuse'):
            result = replay_spec_refusal(path, blocks)
        else:
            result = replay_spec_exact(path, blocks)[0]
        print(json.dumps(result, indent=2))
        return
    status = json.loads((args.root / 'cells-status.json').read_text())
    require(status['state'] == 'complete', 'cells incomplete')
    cells = {row['case']: row for row in status['completed']}
    by_directory = {row['directory']: row for row in status['completed']}
    require(set(cells) == {'gen-default', 'spec-refuse-7', 'spec-exact-8'}, 'cell set')
    for row in cells.values():
        require(Path(row['directory']).name == row['directory'], 'escaping receipt')
    build = json.loads((args.root / 'build.json').read_text())
    require(build['exit'] == 0 and build['cargo_target_dir'] == '/root/wt-c/target-day11', 'build receipt')
    DAY7.hashes(args.root, {'path': 'build.log', 'bytes': (args.root / 'build.log').stat().st_size,
                            'sha256': build['build_log_sha256']})
    manifest = json.loads((args.root / 'binary-manifest.json').read_text())
    post = json.loads((args.root / 'binary-postcheck.json').read_text())
    day9 = {'run-gen': 'b1c4090cb32bfba5f65bddd64ccafe407516a51dd8142b77e73184e2ad3f6d22',
            'run-spec': 'ded88cc84bfd54319b8a6bd6ee08733fedfff4b75be769c47a0b9cb29bb6ba0d'}
    day10 = {'run-gen': 'ec513d94dfed9c94ec419134406b0fe8be190242a6970ed8f5972a9795c2ea98',
             'run-spec': '0685dfa17d4835c0d98143ac75840b598db37ba84cb8e910ed29e4bd22aed1ac'}
    for record in (manifest, post):
        require(record['binary_sha256'] == build['binary_sha256'], 'day-eleven binary identity')
        require(record['frozen'] == {'day9': {'runtime_source': '148e7f0e9994a1c35dd3e0891dae559c377561e0', 'binary_sha256': day9},
                                     'day10': {'runtime_source': '79353d53d7f69713177caf2e9830b48f9e600fa3', 'binary_sha256': day10}},
                'frozen binaries changed')
        diff = subprocess.run(['git', 'diff', '--quiet', build['source'], record['worktree_head'], '--', 'crates/'], cwd=ROOT)
        require(diff.returncode == 0, 'worktree engine sources differ from the built source')
    require(build['binary_sha256'] not in (day9, day10), 'day-eleven build is a frozen build')
    inspect = json.loads((args.root / 'inspect.json').read_text())
    require(inspect['exit'] == 0 and inspect['source'] == build['source'], 'inspect receipt')
    for name, value in inspect['files'].items():
        DAY7.hashes(args.root / 'inspect', {'path': name, 'bytes': value['bytes'], 'sha256': value['sha256']})
    # The lock names the family, the binding verdict and the census it hashed; the artifact
    # identity itself is the installer's `installed artifact_sha256=` line, checked per cell.
    lock = dict(line.split('=', 1) for line in (args.root / 'inspect' / 'artifact.lock').read_text().splitlines() if '=' in line)
    require(lock['shard'] == ARTIFACT and lock['family'] == 'qwen35_moe' and lock['binding'] == 'passed'
            and lock['tensor_count'] == '753', 'inspect artifact lock')
    census_bytes = (args.root / 'inspect' / 'tensor-census.tsv').read_bytes()
    require(hashlib.sha256(census_bytes).hexdigest() == lock['census_sha256'], 'census does not match the lock')
    rows = census_rows(args.root / 'inspect' / 'tensor-census.tsv')
    # The gen cell names the trunk count; the literal catalog is derived from the census alone.
    gen_log = read_log(args.root / cells['gen-default']['directory'])
    trunk = int(re.search(r'^loaded qwen35moe \((\d+) trunk layers', gen_log, re.M)[1])
    identity, layers = literal_identity(rows, trunk)
    literal_sha = hashlib.sha256(identity.encode()).hexdigest()
    blocks = len(layers)
    gen, gen_tape = replay_gen(args.root / cells['gen-default']['directory'], blocks)
    refusal = replay_spec_refusal(args.root / cells['spec-refuse-7']['directory'], blocks)
    exact, spec_tape, acceptance = replay_spec_exact(args.root / cells['spec-exact-8']['directory'], blocks)
    for row in (gen, refusal, exact):
        require(row['catalog']['catalog_sha256'] == literal_sha, 'plan-derived catalog differs from the literal day-ten catalog: ' + row['case'])
    require(refusal['catalog']['records_sha256'] == exact['catalog']['records_sha256'], 'spec record digests differ')
    require(gen['catalog']['records_sha256'] != exact['catalog']['records_sha256'], 'gen (no MTP head) and spec record digests must differ')
    gen_control = DAY9.replay('default-gen-off')
    require(gen_tape == gen_control[1], 'gen tape differs from the day-nine native control')
    day10_exact = (DAY10 / 'budget-exact-8-attempt1' / 'command.log').read_text()
    require(gen_tape == DAY7.tokens(day10_exact), 'gen tape differs from the day-ten exact cell')
    spec_control = DAY9.replay('default-spec-off')
    require(spec_tape == spec_control[1] and acceptance == spec_control[2], 'spec tape/acceptance differ from the day-nine native control')
    checked = datetime.datetime.fromisoformat(post['checked_utc'])
    for row in (gen, refusal, exact):
        require(datetime.datetime.fromisoformat(row['ended_utc']) <= checked, 'postcheck preceded cell')
        row['attempts'] = by_directory[row['case']]['attempts']
    print(json.dumps({'runtime_source': build['source'], 'literal_catalog_sha256': literal_sha,
                      'catalog_layers': layers, 'trunk_layers': trunk, 'parity': 'plan-derived catalog == literal day-ten catalog',
                      'gen': gen, 'spec_refused': refusal, 'spec_exact': exact,
                      'gen_tape_matches_day9_default_gen_off_and_day10_exact': True,
                      'spec_tape_and_acceptance_match_day9_default_spec_off': True}, indent=2))


if __name__ == '__main__':
    main()
