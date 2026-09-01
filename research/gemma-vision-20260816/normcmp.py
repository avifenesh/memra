import json, struct, sys, glob

# --- read one tensor from a safetensors shard (bf16 -> float) ---
def st_read(path_glob, name):
    for p in glob.glob(path_glob):
        with open(p, 'rb') as f:
            hlen = struct.unpack('<Q', f.read(8))[0]
            hdr = json.loads(f.read(hlen))
            if name not in hdr: continue
            info = hdr[name]
            dt, (a, b) = info['dtype'], (info['data_offsets'])
            f.seek(8 + hlen + a)
            raw = f.read(b - a)
            if dt == 'BF16':
                import array
                u = array.array('H'); u.frombytes(raw)
                return [struct.unpack('<f', struct.pack('<I', x << 16))[0] for x in u[:6]]
            elif dt == 'F32':
                import array
                x = array.array('f'); x.frombytes(raw); return list(x[:6])
    return None

# --- read one F32 tensor from GGUF ---
def gguf_read(path, name):
    with open(path, 'rb') as f:
        data = f.read()
    # minimal GGUF v3 parse
    import io
    b = io.BytesIO(data); assert b.read(4) == b'GGUF'
    ver = struct.unpack('<I', b.read(4))[0]
    n_tensors = struct.unpack('<Q', b.read(8))[0]
    n_kv = struct.unpack('<Q', b.read(8))[0]
    def rstr():
        n = struct.unpack('<Q', b.read(8))[0]; return b.read(n).decode('utf-8', 'replace')
    def rval(t):
        if t==0: return struct.unpack('<b', b.read(1))[0]
        if t==1: return struct.unpack('<B', b.read(1))[0]
        if t==2: return struct.unpack('<h', b.read(2))[0]
        if t==3: return struct.unpack('<H', b.read(2))[0]
        if t==4: return struct.unpack('<i', b.read(4))[0]
        if t==5: return struct.unpack('<I', b.read(4))[0]
        if t==6: return struct.unpack('<f', b.read(4))[0]
        if t==7: return struct.unpack('<?', b.read(1))[0]
        if t==8: return rstr()
        if t==9:
            et = struct.unpack('<I', b.read(4))[0]; n = struct.unpack('<Q', b.read(8))[0]
            return [rval(et) for _ in range(n)]
        if t==10: return struct.unpack('<Q', b.read(8))[0]
        if t==11: return struct.unpack('<q', b.read(8))[0]
        if t==12: return struct.unpack('<d', b.read(8))[0]
        raise ValueError(t)
    align = 32
    for _ in range(n_kv):
        k = rstr(); t = struct.unpack('<I', b.read(4))[0]; v = rval(t)
        if k == 'general.alignment': align = v
    tinfo = {}
    for _ in range(n_tensors):
        nm = rstr(); nd = struct.unpack('<I', b.read(4))[0]
        dims = [struct.unpack('<Q', b.read(8))[0] for _ in range(nd)]
        ty = struct.unpack('<I', b.read(4))[0]; off = struct.unpack('<Q', b.read(8))[0]
        tinfo[nm] = (ty, off, dims)
    data_start = b.tell()
    if data_start % align: data_start += align - (data_start % align)
    ty, off, dims = tinfo[name]
    assert ty == 0, f'{name} not F32 (ty {ty})'
    base = data_start + off
    return list(struct.unpack('<6f', data[base:base+24]))

st = '/data/memra/models/gemma4-31b/hf-native/model-00001-of-00002.safetensors'
gg = '/data/memra/models/gemma4-31b/gemma-4-31B_q4_0-it.gguf'
pairs = [
  ('blk.0.attn_norm.weight', 'model.language_model.layers.0.input_layernorm.weight'),
  ('blk.0.attn_q_norm.weight', 'model.language_model.layers.0.self_attn.q_norm.weight'),
  ('blk.0.ffn_norm.weight', 'model.language_model.layers.0.pre_feedforward_layernorm.weight'),
]
for gn, hn in pairs:
    hf = st_read(st, hn)
    ggv = gguf_read(gg, gn)
    if hf is None: print(f'{gn}: HF {hn} not in shard1'); continue
    diff = [round(g-h,4) for g,h in zip(ggv, hf)]
    print(f'{gn}:\n  GGUF {[round(x,4) for x in ggv]}\n  HF   {[round(x,4) for x in hf]}\n  GGUF-HF {diff}')
