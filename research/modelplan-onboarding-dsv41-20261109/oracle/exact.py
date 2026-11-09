"""Correctly rounded torch primitives for the DeepSeek-V4.1 oracle.

The vendor program fixes the operation sequence, the dtype every tensor is stored in and every quantization point; it
does not fix how a GEMM or a reduction orders its fp32 accumulation, or how exp, sigmoid, cos and the like approximate
their result. torch CPU decides both by its kernels (blocked MKL/oneDNN accumulation, SLEEF approximations), the GPU
kernels decide them differently again. Any such difference can flip a bf16 or fp8 rounding, and the bf16 index scores
tie often enough that a flipped score changes a top-k selection, so two correct implementations drift apart by O(1)
in later layers.

`install()` pins those free choices to the canonical one: every GEMM, einsum, sum and mean accumulates in float64 and
rounds once to float32 (then to the op's own output dtype, as the fp32-accumulating original does), and every
transcendental elementwise op is evaluated in float64 and rounded once. memra's reference executor evaluates its
primitives the same way (memra_gguf::dsv41_forward), so the two agree to the last bit except where a float64 result
lands within one float64 ulp of a float32 rounding boundary. Nothing else changes: the patches only cover ops whose
float32 result is implementation-defined.
"""
import torch
import torch.nn.functional as F

_orig = {}


def _floating(t):
    return isinstance(t, torch.Tensor) and t.is_floating_point() and t.dtype != torch.float64


def _round(y64, dtype):
    """float64 result -> float32 -> the original op's output dtype."""
    return y64.float().to(dtype)


def install():
    if _orig:
        return
    _orig.update(linear=F.linear, einsum=torch.einsum, sum=torch.Tensor.sum, tsum=torch.sum, mean=torch.Tensor.mean,
                 softmax=torch.Tensor.softmax, tsoftmax=torch.softmax, sigmoid=torch.sigmoid,
                 tsigmoid=torch.Tensor.sigmoid, silu=F.silu, softplus=F.softplus, polar=torch.polar,
                 rpow=torch.Tensor.__rpow__, rsqrt=torch.rsqrt, exp=torch.exp, texp=torch.Tensor.exp)

    def linear(x, w, bias=None):
        dtype = torch.promote_types(x.dtype, w.dtype)
        y = _orig["linear"](x.double(), w.double(), None if bias is None else bias.double())
        return _round(y, dtype)

    def einsum(equation, *operands):
        if len(operands) == 1 and isinstance(operands[0], (list, tuple)):
            operands = tuple(operands[0])
        dtype = operands[0].dtype
        for op in operands[1:]:
            dtype = torch.promote_types(dtype, op.dtype)
        y = _orig["einsum"](equation, *[op.double() for op in operands])
        return _round(y, dtype)

    def tensor_sum(self, *args, **kwargs):
        if not _floating(self) or "dtype" in kwargs:
            return _orig["sum"](self, *args, **kwargs)
        return _round(_orig["sum"](self.double(), *args, **kwargs), self.dtype)

    def torch_sum(input, *args, **kwargs):
        return tensor_sum(input, *args, **kwargs)

    def tensor_mean(self, *args, **kwargs):
        if not _floating(self) or "dtype" in kwargs:
            return _orig["mean"](self, *args, **kwargs)
        return _round(_orig["mean"](self.double(), *args, **kwargs), self.dtype)

    def tensor_softmax(self, dim, *args, **kwargs):
        if not _floating(self):
            return _orig["softmax"](self, dim, *args, **kwargs)
        return _round(_orig["softmax"](self.double(), dim), self.dtype)

    def torch_softmax(input, dim, *args, **kwargs):
        return tensor_softmax(input, dim, *args, **kwargs)

    def unary(name):
        def op(x, *args, **kwargs):
            if not _floating(x):
                return _orig[name](x, *args, **kwargs)
            return _round(_orig[name](x.double(), *args, **kwargs), x.dtype)
        return op

    def polar(abs_, angle):
        assert abs_.dtype == torch.float32 and angle.dtype == torch.float32
        return torch.complex(abs_ * torch.cos(angle.double()).float(), abs_ * torch.sin(angle.double()).float())

    def rpow(self, base):
        if not _floating(self):
            return _orig["rpow"](self, base)
        return _round(_orig["rpow"](self.double(), base), self.dtype)

    def rsqrt(x):
        # 1 / sqrt, each correctly rounded in the input's precision (torch's vectorized CPU form).
        return torch.reciprocal(torch.sqrt(x))

    F.linear = linear
    torch.einsum = einsum
    torch.Tensor.sum = tensor_sum
    torch.sum = torch_sum
    torch.Tensor.mean = tensor_mean
    torch.Tensor.softmax = tensor_softmax
    torch.softmax = torch_softmax
    torch.sigmoid = unary("sigmoid")
    torch.Tensor.sigmoid = unary("tsigmoid")
    F.silu = unary("silu")
    F.softplus = unary("softplus")
    torch.exp = unary("exp")
    torch.Tensor.exp = unary("texp")
    torch.polar = polar
    torch.Tensor.__rpow__ = rpow
    torch.rsqrt = rsqrt
