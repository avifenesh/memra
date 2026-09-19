#!/usr/bin/env python3
"""Independent v1 wire/hash oracle. Run --check; regeneration requires lead review."""
import hashlib
import json
import pathlib
import sys

def digest(domain, data):
    domain = domain.encode()
    return hashlib.sha256(b'memra-tier\0v1\0' + len(domain).to_bytes(8, 'little') + domain
                          + len(data).to_bytes(8, 'little') + data).digest()

def encode(value):
    return json.dumps(value, separators=(',', ':'), ensure_ascii=False).encode()

def payload(n):
    return bytes((17*i + 3) % 251 for i in range(n))

program = dict(version=1, **{name: [i]*32 for i, name in enumerate([
    'artifact', 'serialized_plan', 'numeric', 'stream', 'tokenizer', 'template',
    'adapter', 'modality', 'position', 'tenant_salt'], 1)})
key = dict(version=1, artifact=[1]*32, semantic_id=[2]*32, layout=[3]*32, generation=7)
tokens = (3).to_bytes(8, 'little') + b''.join(t.to_bytes(4, 'little') for t in [1, 2, 3])
block = dict(version=1, namespace=list(digest('program', encode(program))), parent=[0]*32,
             tokens=list(digest('tokens', tokens)), start=0, end=3, group=0, owner=0, epoch=7)
encoding = dict(version=1, program=[42]*32, row_bytes=3)
segment = dict(version=1, group=0, page=0, owner=0, role='Payload', tensor=None, offset=0,
               valid_bytes=3, storage_bytes=4, alignment=4, encoding=encoding)
requirement = dict(version=1, group=0, owner=0, role='Payload', page_count=1, pages='AllPages')
layout = dict(version=1, segments=[segment], requirements=[requirement])
bundle = dict(version=1, id=block, program=program, layout=layout, kind='Active',
              committed_high_water=3, owner_aliases=[], checksums=[list(digest('valid-bytes', payload(3)))])
fixture = dict(version=1, name='tier-contract-v1',
               payloads={str(n): digest('valid-bytes', payload(n)).hex()
                         for n in [0, 1, 264, 288, 4095, 4096, 4097, 1048576]},
               wire={name: digest('fixture-wire', encode(value)).hex()
                     for name, value in [('program', program), ('key', key), ('bundle', bundle)]})
path = pathlib.Path(__file__).with_name('fixtures.json')
if '--check' in sys.argv:
    assert json.loads(path.read_text()) == fixture, 'fixture drift'
    print('tier-contract-v1: 8 payload + 3 canonical wire SHA256 pins match')
else:
    path.write_text(json.dumps(fixture, indent=2) + '\n')
