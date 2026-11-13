"""Which summation order does the vendor's fp32 F.linear (torch -> cuBLAS, TF32 off, as run_oracle.py runs it) use
at the router's and the compressor's shapes? Each output of a bf16 x bf16 product sum over K = 5120 is compared, bit
for bit, with (a) one chain from zero in K order (memra's one-chain program, kpc) and (b) the K-segment order (8
chains of K / 8, then ((s0 + s1) + (s2 + s3)) + ((s4 + s5) + (s6 + s7)): r1). bf16 x bf16 products are exact in fp32,
so an fma chain equals a sequential fp32 sum of the products. Rows: decode (1 row a step) and prefill shapes; for the
big M only a sample of rows is checked on the host.
usage: python -I cublas_order.py OUT.txt"""
import sys

import numpy as np
import torch

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
dev = torch.device("cuda")
out = open(sys.argv[1], "w")


def say(s):
    print(s)
    out.write(s + "\n")
    out.flush()


say(f"torch {torch.__version__} cuda {torch.version.cuda} device {torch.cuda.get_device_name(0)} "
    f"SMs {torch.cuda.get_device_properties(0).multi_processor_count}")
g = torch.Generator().manual_seed(1234)
K = 5120


def orders(x, w):
    """x [r][K], w [n][K] float32 numpy (bf16 values): one chain and the K-segment order, [r][n] each."""
    p = x[:, None, :] * w[None, :, :]  # exact
    chain = np.add.accumulate(p, axis=2, dtype=np.float32)[:, :, -1]
    sl = K // 8
    seg = [np.add.accumulate(p[:, :, i * sl:(i + 1) * sl], axis=2, dtype=np.float32)[:, :, -1] for i in range(8)]
    f = np.float32
    tree = ((seg[0] + seg[1]) + (seg[2] + seg[3])) + ((seg[4] + seg[5]) + (seg[6] + seg[7]))
    return chain.astype(f), tree.astype(f)


for name, n in (("router", 384), ("compressor", 512)):
    w = (torch.randn(n, K, generator=g) * 0.02).to(torch.bfloat16)
    wd = w.to(dev).float()
    for m in (1, 2, 4, 8, 45, 512, 4096):
        x = torch.randn(m, K, generator=g).to(torch.bfloat16)
        y = torch.nn.functional.linear(x.to(dev).float(), wd).cpu().numpy()
        rows = np.arange(m) if m <= 64 else np.random.default_rng(7).choice(m, 16, replace=False)
        chain, tree = orders(x.float().numpy()[rows], w.float().numpy())
        yy = y[rows]
        eq_c = (yy.view(np.uint32) == chain.view(np.uint32)).mean()
        eq_t = (yy.view(np.uint32) == tree.view(np.uint32)).mean()
        rel_c = np.abs(yy - chain).max() / np.abs(yy).max()
        rel_t = np.abs(yy - tree).max() / np.abs(yy).max()
        say(f"{name} n {n} m {m} (rows checked {len(rows)}): cuBLAS == one chain {eq_c * 100:.2f}%, "
            f"== K-segment order {eq_t * 100:.2f}%; max |diff|/max|y| chain {rel_c:.2e}, segments {rel_t:.2e}")
say("CUBLAS-ORDER-DONE")
