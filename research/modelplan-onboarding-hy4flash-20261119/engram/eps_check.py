"""`torch.nn.RMSNorm(eps=None)`: which epsilon does it add? Compares the module's output with the
formula at `finfo(input dtype).eps` and at `finfo(float32).eps`, per input dtype.

    venv/bin/python -I eps_check.py
"""
import torch

torch.manual_seed(0)
for dtype in (torch.bfloat16, torch.float16, torch.float32, torch.float64):
    norm = torch.nn.RMSNorm(96).to(dtype)
    with torch.no_grad():
        norm.weight.copy_(1 + 0.1 * torch.randn(96))
        x = (0.05 * torch.randn(64, 96)).to(dtype)
        got = norm(x).double()

        def formula(eps):
            xd = x.double()
            y = xd * torch.rsqrt(xd.square().mean(-1, keepdim=True) + eps) * norm.weight.double()
            return y.to(dtype).double()

        print(
            f"{str(dtype):15} eps={norm.eps} "
            f"max|out - formula(finfo(input).eps={torch.finfo(dtype).eps:.3g})|={float((got - formula(torch.finfo(dtype).eps)).abs().max()):.3g} "
            f"max|out - formula(finfo(float32).eps)|={float((got - formula(torch.finfo(torch.float32).eps)).abs().max()):.3g}"
        )
