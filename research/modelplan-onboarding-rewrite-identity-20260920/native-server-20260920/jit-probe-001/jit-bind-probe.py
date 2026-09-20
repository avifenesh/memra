#!/usr/bin/env python3
"""Driver JIT initialization diagnostic. Creates a context, links code, never launches a kernel."""
import ctypes as c
import json
import importlib.util
from pathlib import Path
import sys

helper_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).resolve().parents[1] / 'qualify-native.py'
spec = importlib.util.spec_from_file_location('qualified_runner', helper_path)
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
helper.verify_lease()

lib = c.CDLL('libcuda.so.1')
void = c.c_void_p
uint = c.c_uint

def call(name, args, types):
    function = getattr(lib, name)
    function.argtypes = types
    function.restype = c.c_int
    code = function(*args)
    if code:
        raise RuntimeError(f'{name} CUDA error {code}')

def libraries():
    return sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
                   if 'libnvidia' in line or 'libcuda' in line})

variant = sys.argv[1]
call('cuInit', [0], [uint])
device = c.c_int()
call('cuDeviceGet', [c.byref(device), 0], [c.POINTER(c.c_int), c.c_int])
context = void()
call('cuCtxCreate_v2', [c.byref(context), 0, device], [c.POINTER(void), uint, c.c_int])
state = void()
try:
    before = libraries()
    call('cuLinkCreate_v2', [0, None, None, c.byref(state)], [uint, void, void, c.POINTER(void)])
    if variant == 'legacy-ret':
        version, target, body = '7.0', 'sm_80', '.visible .entry init() { ret; }'
    elif variant == 'current-ret':
        version, target, body = '8.8', 'sm_120a', '.visible .entry init() { ret; }'
    elif variant == 'current-store':
        version, target = '8.8', 'sm_120a'
        body = '.visible .entry init(.param .u64 output) { .reg .u64 p; .reg .f32 x; ld.param.u64 p, [output]; ld.global.f32 x, [p]; fma.rn.f32 x, x, 0f3f800001, 0f3f800000; st.global.f32 [p], x; ret; }'
    else:
        raise ValueError('unknown diagnostic variant')
    ptx = f'.version {version}\n.target {target}\n.address_size 64\n{body}\n'.encode() + b'\0'
    buffer = c.create_string_buffer(ptx)
    call('cuLinkAddData_v2', [state, 1, c.cast(buffer, void), len(ptx), b'init', 0, None, None],
         [void, c.c_int, void, c.c_size_t, c.c_char_p, uint, void, void])
    image, size = void(), c.c_size_t()
    call('cuLinkComplete', [state, c.byref(image), c.byref(size)], [void, c.POINTER(void), c.POINTER(c.c_size_t)])
    after = libraries()
    print(json.dumps({'variant': variant, 'kernel_launches': 0, 'image_bytes': size.value,
                      'before': before, 'after': after, 'added': sorted(set(after) - set(before))}))
finally:
    if state.value:
        call('cuLinkDestroy', [state], [void])
    call('cuCtxDestroy_v2', [context], [void])
