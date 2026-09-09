"""GPU oracle for live FA2 depth under graph replay; run under the canonical flock."""
import ctypes as C
import json
import pathlib
import subprocess
import sys

assert not subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip()
import torch
torch.manual_seed(5183)
anchor = torch.empty(1, device="cuda")
d = C.CDLL("libcuda.so.1")
def ck(code):
    if code: raise RuntimeError(f"CUDA status {code}")
module = C.c_void_p()
ck(d.cuModuleLoad(C.byref(module), sys.argv[1].encode()))
funcs = []
for name in ["fa_prefill_qw_fa2", "fa_prefill_qw_fa2_prime_table"]:
    f = C.c_void_p()
    ck(d.cuModuleGetFunction(C.byref(f), module, name.encode()))
    ck(d.cuFuncSetAttribute(f, 8, 49152))
    funcs.append(f)
results = []
for rows, depth in [(16, 33), (33, 65), (128, 257), (1034, 2058)]:
    q = torch.randn(rows, 24, 256, device="cuda")
    k = torch.randn(depth + 32, 4, 256, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    eager = torch.empty_like(q)
    replay = torch.empty_like(q)
    table = torch.zeros(12, device="cuda", dtype=torch.int64)
    table[7] = depth
    def launch(which, out, n):
        args = [C.c_void_p(t.data_ptr()) for t in [q, k, v, out]] + [C.c_int(rows)]
        args += [C.c_void_p(table.data_ptr()) if which else C.c_int(n)]
        params = (C.c_void_p * len(args))(*[C.cast(C.byref(a), C.c_void_p) for a in args])
        ck(d.cuLaunchKernel(funcs[which], (rows*6+63)//64, 4, 1, 32, 4, 1, 49152,
                           C.c_void_p(torch.cuda.current_stream().cuda_stream), params, None))
    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        launch(1, replay, depth)
    first = None
    for current in [depth, depth + 1, depth + 31]:
        table[7] = current
        graph.replay()
        launch(0, eager, current)
        torch.cuda.synchronize()
        assert bool(replay.isfinite().all())
        differences = int((replay.view(torch.int32) != eager.view(torch.int32)).sum())
        assert differences == 0, (rows, current, differences)
        if first is None: first = replay.clone()
        elif current == depth + 31: assert not torch.equal(first, replay), "red arm: depth change had no effect"
        row = dict(rows=rows, depth=current, byte_differences=differences)
        results.append(row)
        print(json.dumps(row), flush=True)
    del graph, q, k, v, table, eager, replay, first
    torch.cuda.empty_cache()
pathlib.Path(sys.argv[2]).write_text(json.dumps(results, indent=2) + "\n")
