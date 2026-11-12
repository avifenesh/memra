"""Print the GGUF metadata keys that size a prompt-end tail (memra#1233): layer counts, the
full-attention interval, KV heads/dims and the SSM (GDN) state dims. Reads the header only."""
import struct
import sys

SCAL = {0: 'B', 1: 'b', 2: 'H', 3: 'h', 4: 'I', 5: 'i', 6: 'f', 7: '?', 10: 'Q', 11: 'q', 12: 'd'}


def rd(f, fmt):
    n = struct.calcsize('<' + fmt)
    return struct.unpack('<' + fmt, f.read(n))


def rstr(f):
    (n,) = rd(f, 'Q')
    return f.read(n).decode('utf-8', 'replace')


def rval(f, t):
    if t in SCAL:
        return rd(f, SCAL[t])[0]
    if t == 8:
        return rstr(f)
    if t == 9:
        (et,) = rd(f, 'I')
        (n,) = rd(f, 'Q')
        if et == 8:
            return [rstr(f) for _ in range(n)]
        if n > 4096:
            f.seek(n * struct.calcsize('<' + SCAL[et]), 1)
            return f'<array {n}>'
        return [rval(f, et) for _ in range(n)]
    raise ValueError(t)


with open(sys.argv[1], 'rb') as f:
    assert f.read(4) == b'GGUF'
    rd(f, 'I')
    _, nkv = rd(f, 'QQ')
    for _ in range(nkv):
        k = rstr(f)
        (t,) = rd(f, 'I')
        v = rval(f, t)
        if any(s in k for s in ('block_count', 'head_count', 'key_length', 'value_length', 'ssm',
                                'full_attention', 'linear', 'architecture', 'nextn', 'embedding_length')):
            print(k, '=', v if not isinstance(v, list) or len(v) < 80 else f'list[{len(v)}]')
