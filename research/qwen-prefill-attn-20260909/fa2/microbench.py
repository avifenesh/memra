"""Standalone native CUDA class comparison; run under an outer GPU flock."""
import argparse
import ctypes as C
import hashlib
import json
import pathlib
import statistics
import subprocess

p = argparse.ArgumentParser()
p.add_argument("fatbin")
p.add_argument("--output", required=True)
p.add_argument("--shape", type=int)
p.add_argument("--kernels", default="fa_prefill_qw_db,fa2_gqa16,fa2_gqa16_log2")
p.add_argument("--real", help="torch file with real q/k/v tensors")
args = p.parse_args()
assert not subprocess.check_output(
    ["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"], text=True
).strip(), "compute list must be empty before this process initializes CUDA"
import torch

torch.manual_seed(5183)
torch.cuda.init()
context_anchor = torch.empty(1, device="cuda")
d = C.CDLL("libcuda.so.1")


def ck(status):
    if status:
        raise RuntimeError(f"CUDA driver status {status}")


module = C.c_void_p()
ck(d.cuModuleLoad(C.byref(module), args.fatbin.encode()))
names = args.kernels.split(",")
functions = {}
resources = {}
for name in names:
    f = C.c_void_p()
    ck(d.cuModuleGetFunction(C.byref(f), module, name.encode()))
    smem = 69888 if name == "fa_prefill_qw_db" else (49152 if "32_rotate" in name else 32768)
    ck(d.cuFuncSetAttribute(f, 8, smem))
    regs, local, resident = C.c_int(), C.c_int(), C.c_int()
    ck(d.cuFuncGetAttribute(C.byref(regs), 4, f))
    ck(d.cuFuncGetAttribute(C.byref(local), 3, f))
    ck(d.cuOccupancyMaxActiveBlocksPerMultiprocessor(C.byref(resident), f, 128, C.c_size_t(smem)))
    resources[name] = dict(registers=regs.value, local_bytes=local.value, shared_bytes=smem, ctas_per_sm=resident.value)
    functions[name] = f


def launch(name, q, k, v, out):
    rows, depth = q.shape[0], k.shape[0]
    values = [C.c_void_p(t.data_ptr()) for t in (q, k, v, out)]
    if name == "fa_prefill_qw_db":
        values += [C.c_int(x) for x in (256, 24, 4, rows, depth)]
        values += [C.c_float(1 / 16)] + [C.c_int(x) for x in (1, 1024, 1024)]
        grid = ((rows + 63) // 64, 24)
    else:
        values += [C.c_int(rows), C.c_int(depth)]
        grid = (((rows * 6 + 63) // 64, 4) if "gqa" in name else ((rows + 63) // 64, 24))
    params = (C.c_void_p * len(values))(*[C.cast(C.byref(x), C.c_void_p) for x in values])
    ck(d.cuLaunchKernel(functions[name], *grid, 1, 32, 4, 1,
                       resources[name]["shared_bytes"], C.c_void_p(torch.cuda.current_stream().cuda_stream), params, None))


result = dict(fatbin_sha256=hashlib.sha256(pathlib.Path(args.fatbin).read_bytes()).hexdigest(),
              resources=resources, seed=5183, rows=[])
shapes = [(129, 161), (1024, 8192), (1024, 32768), (1024, 131070)]
if args.shape:
    shapes = [(1024, args.shape)]
real = torch.load(args.real, weights_only=True) if args.real else None
if real:
    shapes = [(real["q"].shape[0], real["k"].shape[0])]
for rows, depth in shapes:
    q = real["q"].cuda().float().contiguous() if real else torch.randn(rows, 24, 256, device="cuda", dtype=torch.float32)
    k = real["k"].cuda().bfloat16().contiguous() if real else torch.randn(depth, 4, 256, device="cuda", dtype=torch.bfloat16)
    v = real["v"].cuda().bfloat16().contiguous() if real else torch.randn_like(k)
    outputs = {name: torch.empty_like(q) for name in names}
    timings = {name: [] for name in names}
    for rep in range(12):
        for name in (names if rep % 2 == 0 else list(reversed(names))):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            launch(name, q, k, v, outputs[name])
            end.record()
            end.synchronize()
            if rep >= 3:
                timings[name].append(start.elapsed_time(end))
    # Useful causal FLOPs, excluding masked entries and padded logical query rows.
    flops = 4 * 24 * 256 * rows * (depth - (rows - 1) / 2)
    baseline = outputs[names[0]]
    for name in names:
        out = outputs[name]
        diff = out - baseline
        ms = statistics.median(timings[name])
        row = dict(rows=rows, depth=depth, kernel=name, ms=ms, samples_ms=timings[name],
                   tflops=flops / ms / 1e9, finite=bool(out.isfinite().all()),
                   max_attention_output_deviation=float(diff.abs().max()),
                   relative_l2=float(diff.norm() / baseline.norm()))
        if rows == 129:
            # Independent small-shape FP32 oracle with explicit offset causal mask.
            qh = q.bfloat16().float().transpose(0, 1)
            kh = k.float().repeat_interleave(6, dim=1).transpose(0, 1)
            vh = v.float().repeat_interleave(6, dim=1).transpose(0, 1)
            scores = qh @ kh.transpose(1, 2) / 16
            mask = torch.arange(depth, device="cuda")[None, :] > (depth - rows + torch.arange(rows, device="cuda"))[:, None]
            ref = (scores.masked_fill(mask, -float("inf")).softmax(-1) @ vh).transpose(0, 1)
            row["max_vs_fp32_reference"] = float((out - ref).abs().max())
        result["rows"].append(row)
        print(json.dumps(row), flush=True)
    pathlib.Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    del q, k, v, outputs, baseline, diff, out
    torch.cuda.empty_cache()
