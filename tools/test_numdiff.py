#!/usr/bin/env python3
"""Tests for tools/numdiff.py. Run: python3 -m unittest tools.test_numdiff

The compiled cases build tiny kernels for sm_120a (no GPU needed) and are skipped without nvcc;
the capture cases build a one-file crate with cargo and are skipped without it.
"""
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numdiff as nd  # noqa: E402

NVCC = os.environ.get("NUMDIFF_REAL_NVCC", "/usr/local/cuda/bin/nvcc")
HAVE_NVCC = Path(NVCC).exists() and shutil.which("cuobjdump")


def sass(funcs):
    out = ["\tcode for sm_120a"]
    for name, body in funcs.items():
        out.append(f"\t\tFunction : {name}")
        for i, ins in enumerate(body):
            out.append(f"        /*{i * 16:04x}*/                   {ins} ;      /* 0x0000000000000000 */")
            out.append("                                                         /* 0x000fe20000000000 */")
    return "\n".join(out)


def fp(insns_text, attrs="a0"):
    funcs = nd.parse_sass(sass({"k": insns_text}))
    lines, external = nd.canonical(funcs["k"])
    stream = hashlib.sha256("\n".join(lines).encode()).hexdigest()[:20]
    return {"hash": stream, "attrs": attrs, "fp": stream + attrs,
            "profile": dict(nd.fp_profile(funcs["k"])), "seq": nd.fp_sequence(funcs["k"]),
            "external": external, "n": len(funcs["k"]), "lines": lines,
            "unfused_pairs": nd.unfused_pairs(lines)}


def prof(**counts):
    """A fingerprint with only a float profile (for the classification rules)."""
    p = {k.replace("_", "."): v for k, v in counts.items()}
    return {"hash": str(sorted(p.items())), "profile": p}


class Canonical(unittest.TestCase):
    def test_reuse_hints_do_not_matter_registers_do(self):
        a = fp(["LDG.E R2, desc[UR4][R4.64]", "FFMA R3, R2.reuse, R5, R3", "EXIT"])
        b = fp(["LDG.E R2, desc[UR4][R4.64]", "FFMA R3, R2, R5, R3", "EXIT"])
        self.assertEqual(a["hash"], b["hash"])
        # a tuple register names its neighbours: R8 in a .128 store is R8..R11, so a consistent
        # renaming of R10/R11 swaps two stored lanes
        c = fp(["FMUL R10, R1, R2", "FADD R11, R1, R2", "STG.E.128 desc[UR4][R4.64], R8", "EXIT"])
        d = fp(["FMUL R11, R1, R2", "FADD R10, R1, R2", "STG.E.128 desc[UR4][R4.64], R8", "EXIT"])
        self.assertNotEqual(c["hash"], d["hash"])

    def test_byte_selectors_stay_literal(self):
        a = fp(["I2F.S8 R9, R2.B1", "I2F.S8 R11, R2.B2", "FADD R3, R9, R11", "EXIT"])
        b = fp(["I2F.S8 R9, R2.B2", "I2F.S8 R11, R2.B1", "FADD R3, R9, R11", "EXIT"])
        self.assertNotEqual(a["hash"], b["hash"])

    def test_predicate_vector_bits_stay_literal(self):
        a = fp(["R2P PR, R19, 0x24", "@P2 FADD R1, R1, R2", "@P5 FMUL R1, R1, R3", "EXIT"])
        b = fp(["R2P PR, R19, 0x24", "@P5 FADD R1, R1, R2", "@P2 FMUL R1, R1, R3", "EXIT"])
        self.assertNotEqual(a["hash"], b["hash"])

    def test_branch_targets_are_instruction_indices(self):
        a = fp(["ISETP.GE.AND P0, PT, R0, 0x4, PT", "@P0 BRA 0x20", "FADD R1, R1, R2", "EXIT"])
        self.assertIn("@P0 BRA L2", a["lines"][1])

    def test_a_call_outside_the_listing_is_external(self):
        self.assertTrue(fp(["CALL.ABS.NOINC 0x9990", "EXIT"])["external"])
        self.assertFalse(fp(["CALL.REL.NOINC 0x10", "RET.REL.NODEC R20 0x0", "EXIT"])["external"])

    def test_constant_moves_are_not_half_math(self):
        self.assertEqual(fp(["HFMA2 R3, -RZ, RZ, 0, 0", "EXIT"])["profile"], {})
        self.assertEqual(fp(["HFMA2 R3, R1, R2, R3", "EXIT"])["profile"], {"HFMA2": 1})


class Classify(unittest.TestCase):
    def test_contraction(self):
        a = fp(["FMUL R3, R1, R2", "FADD R4, R3, R5", "EXIT"])
        b = fp(["FFMA R4, R1, R2, R5", "EXIT"])
        self.assertEqual(nd.classify(a, b)[0], ["fma-contraction"])

    def test_contraction_with_other_adds(self):
        # one FMUL folded into an FFMA while two FADDs appeared elsewhere
        self.assertEqual(nd.classify(prof(FMUL=2, FADD=1), prof(FMUL=1, FFMA=1, FADD=3))[0], ["fma-contraction"])

    def test_uniform_datapath_float_ops_count(self):
        a = fp(["UFMUL UR6, UR4, UR5", "UFADD UR7, UR6, UR8", "EXIT"])
        b = fp(["UFFMA UR7, UR4, UR5, UR8", "EXIT"])
        self.assertEqual(nd.classify(a, b)[0], ["fma-contraction"])

    def test_packed_ops_count_as_two_scalar_ops(self):
        self.assertEqual(nd.classify(prof(FFMA2=1), prof(FMUL2=1, FADD2=1))[0], ["fma-contraction"])
        # FFMA2 for two FFMA is a repack, not a contraction
        self.assertNotIn("fma-contraction", nd.classify(prof(FFMA=2), prof(FFMA2=1))[0])

    def test_integer_redux_is_not_a_float_reduction(self):
        self.assertEqual(fp(["REDUX.MAX.S32 UR4, R2", "REDUX.OR UR5, R3", "EXIT"])["profile"], {})
        self.assertEqual(fp(["REDUX.SUM.F32 UR4, R2", "EXIT"])["profile"], {"REDUX.SUM.F32": 1})

    def test_approximation_and_precision(self):
        a = fp(["MUFU.RCP R1, R2", "EXIT"])
        b = fp(["MUFU.EX2 R1, R2", "F2F.F16.F32 R3, R1", "EXIT"])
        self.assertEqual(nd.classify(a, b)[0], ["approximation", "precision"])

    def test_reordered_sum_is_same_fp_ops(self):
        a = fp(["FADD R4, R1, R2", "FADD R5, R4, R3", "EXIT"])
        b = fp(["FADD R4, R2, R3", "FADD R5, R1, R4", "EXIT"])
        self.assertEqual(nd.classify(a, b)[0], ["same-fp-ops"])

    def test_identical_stream_is_identical(self):
        k = fp(["FADD R4, R1, R2", "EXIT"])
        rows = nd.compare_objects(({"k": k}, "d"), ({"k": k}, "d"))
        self.assertEqual(rows[0]["verdict"], "IDENTICAL")

    def test_moved_attributes_block_identical(self):
        a = fp(["FADD R4, R1, R2", "EXIT"], attrs="maxthreads256")
        b = fp(["FADD R4, R1, R2", "EXIT"], attrs="maxthreads64")
        rows = nd.compare_objects(({"k": a}, "d"), ({"k": b}, "d"))
        self.assertEqual((rows[0]["verdict"], rows[0]["classes"]), ("CHANGED", ["attributes"]))

    def test_changed_data_section_blocks_identical(self):
        k = fp(["LDC R1, c[0x3][0x0]", "FADD R4, R1, R2", "EXIT"])
        rows = nd.compare_objects(({"k": k}, "d1"), ({"k": k}, "d2"))
        self.assertEqual((rows[0]["verdict"], rows[0]["classes"]), ("CHANGED", ["data"]))

    def test_added_and_removed(self):
        rows = {r["kernel"]: r["verdict"] for r in nd.compare_objects(({"a": fp(["EXIT"])}, "d"),
                                                                      ({"b": fp(["FADD R1, R1, R2", "EXIT"])}, "d"))}
        self.assertEqual(rows, {"a": "REMOVED", "b": "ADDED"})

    def test_same_code_under_a_new_symbol_is_a_rename(self):
        k = fp(["FADD R1, R1, R2", "EXIT"])
        old, new = "_Z15mul_mat_f8f4v3tILi128EEvv", "_Z14mul_mat_f8f4v3ILi128EEvv"
        rows = nd.compare_objects(({old: k}, "d"), ({new: k}, "d"))
        self.assertEqual([(r["kernel"], r["verdict"], r.get("to")) for r in rows], [(old, "RENAMED", new)])
        # not when the object's data moved too, nor when the attributes differ
        self.assertEqual({r["verdict"] for r in nd.compare_objects(({old: k}, "d"), ({new: k}, "e"))},
                         {"ADDED", "REMOVED"})
        k2 = fp(["FADD R1, R1, R2", "EXIT"], attrs="other")
        self.assertEqual({r["verdict"] for r in nd.compare_objects(({old: k}, "d"), ({new: k2}, "d"))},
                         {"ADDED", "REMOVED"})

    def test_kernel_names(self):
        ks = {"rms_norm_zq8_f32": 1, "rms_norm_zq8_f32_v2": 1, "_Z8score_tcILi8EEvv": 1, "_Z8score_tcILi16EEvv": 1}
        self.assertEqual(nd.find_kernels("rms_norm_zq8_f32", ks), ["rms_norm_zq8_f32"])
        self.assertEqual(nd.find_kernels("score_tc", ks), ["_Z8score_tcILi16EEvv", "_Z8score_tcILi8EEvv"])
        self.assertEqual(nd.find_kernels("_v2", ks), ["rms_norm_zq8_f32_v2"])
        self.assertEqual(nd.find_kernels("rms_norm", ks), [])

    def test_unfused_pairs(self):
        self.assertEqual(fp(["FMUL R3, R1, R2", "FADD R4, R3, R5", "EXIT"])["unfused_pairs"], 1)
        # a product read twice cannot contract into one FFMA
        self.assertEqual(fp(["FMUL R3, R1, R2", "FADD R4, R3, R5", "FADD R6, R3, R4", "EXIT"])["unfused_pairs"], 0)
        # a redefinition ends the product's life, also through a tuple def that covers it
        self.assertEqual(fp(["FMUL R3, R1, R2", "MOV R3, R9", "FADD R4, R3, R5", "EXIT"])["unfused_pairs"], 0)
        self.assertEqual(fp(["FMUL R5, R1, R2", "LDG.E.64 R4, desc[UR4][R8.64]", "FADD R6, R5, R7", "EXIT"])
                         ["unfused_pairs"], 0)

    def test_numeric_flags_keep_ptxas_options_and_defines(self):
        flags = ["-O3", "--ptxas-options=-O1", "-Xptxas=-dlcm=ca", "-DMEMRA_KV_KFMT=1", "-D", "X=2", "-Ifoo", "--fatbin"]
        self.assertEqual(nd.numeric_flags(flags),
                         ["-O3", "--ptxas-options=-O1", "-Xptxas=-dlcm=ca", "-DMEMRA_KV_KFMT=1", "-D X=2"])

    def test_change_verdict(self):
        cases = [({"IDENTICAL": 3, "ADDED": 1}, [], [], [], "CHANGED"),
                 ({"IDENTICAL": 3, "REMOVED": 1}, [], [], [], "CHANGED"),
                 ({"IDENTICAL": 3, "RENAMED": 1}, [], [], [], "IDENTICAL"),
                 ({"IDENTICAL": 3}, ["cu/x.cu [x_v.fatbin]"], [], [], "CHANGED"),
                 ({"IDENTICAL": 3}, [], ["cu/x_stub.cu"], [], "UNCOVERED"),
                 ({"IDENTICAL": 3}, [], [], ["crates/memra-engine/build.rs"], "KERNELS-IDENTICAL")]
        for summary, one_sided, uncovered, host, want in cases:
            self.assertEqual(nd.change_verdict(summary, one_sided, uncovered, host), want, summary)
        # kernel-capable inputs changed and nothing was compared: no identity claim
        self.assertEqual(nd.change_verdict({}, [], [], ["crates/memra-engine/cu/tbl.hpp"], kernel_inputs=True), "UNCOVERED")
        self.assertEqual(nd.change_verdict({}, [], [], [], kernel_inputs=True), "UNCOVERED")
        self.assertEqual(nd.change_verdict({}, [], [], ["crates/x/src/a.rs"]), "KERNELS-IDENTICAL")


class Paths(unittest.TestCase):
    def test_object_key_names_unit_and_toolkit(self):
        e = {"src": "cu/flash_attn.cu", "out": "/o/flash_attn.fatbin", "argv": ["-O3", "-o", "/o/flash_attn.fatbin", "cu/flash_attn.cu"], "tree": "/t"}
        v = dict(e, out="/o/flash_attn_kf8vf8.fatbin")
        real = nd.nvcc_identity
        try:
            nd.nvcc_identity = lambda: {"id": "a"}
            pa, pv = nd.object_path(e, Path("/w")), nd.object_path(v, Path("/w"))
            nd.nvcc_identity = lambda: {"id": "b"}
            pb = nd.object_path(e, Path("/w"))
        finally:
            nd.nvcc_identity = real
        self.assertNotEqual(pa.parent, pv.parent)
        self.assertNotEqual(pa, pb)

    def with_diff(self, changed, fn, *args):
        real = nd.git
        try:
            nd.git = lambda repo, *a, **k: "\n".join(n for n in changed if any(
                n == p or n.startswith(p.rstrip("/") + "/") for p in a[a.index("--") + 1:])) if "--" in a else ""
            return fn(*args)
        finally:
            nd.git = real

    def test_headers_and_build_inputs_recompile_every_unit(self):
        for changed in (["crates/memra-engine/cu/tbl.hpp"], ["crates/memra-engine/cu/x.inc"],
                        ["crates/memra-gguf/include/q.h"], [".cargo/config.toml"], ["Cargo.toml"],
                        ["crates/memra-engine/Cargo.toml"]):
            cu, every, kernel_inputs = self.with_diff(changed, nd.touched_cu, "/r", "a", "b")
            self.assertEqual((cu, every, kernel_inputs), (set(), True, True), changed)
        # a header is a kernel input, not host code
        self.assertEqual(self.with_diff(["crates/memra-engine/cu/tbl.hpp", "crates/memra-gguf/include/q.h"],
                                        nd.host_changes, "/r", "a", "b"), [])
        cu, every, _ = self.with_diff(["crates/memra-engine/cu/k.cu"], nd.touched_cu, "/r", "a", "b")
        self.assertEqual((cu, every), ({"cu/k.cu"}, False))

    def test_build_inputs_key_the_recorded_commands(self):
        keys = []
        saved = (nd.git, nd.archive, nd.capture)
        blobs = {"base": "1111", "head": "2222"}
        nd.git = lambda repo, *a, **k: (blobs[a[1].split(":")[0]] if a[1].endswith(".cargo/config.toml") else "0000") + "\n"
        nd.archive = lambda repo, rev, into: into
        nd.capture = lambda tree, arch, key: keys.append(key) or []
        try:
            nd.build_rev("/r", "base", "120a")
            nd.build_rev("/r", "head", "120a")
        finally:
            nd.git, nd.archive, nd.capture = saved
        self.assertNotEqual(keys[0], keys[1])

    def test_host_changes_count_build_inputs_and_outside_cuda(self):
        changed = ["crates/memra-engine/build.rs", "crates/memra-engine/src/cuda_arch.rs",
                   "crates/memra-probe/src/kernels.cu", "crates/memra-engine/cu/kernels.cu",
                   "crates/memra-engine/tests/x_gpu.rs", "crates/memra-engine/cu/common.cuh"]
        real = nd.git
        try:
            nd.git = lambda repo, *a, **k: "\n".join(changed)
            got = nd.host_changes("/r", "a", "b")
        finally:
            nd.git = real
        self.assertEqual(got, ["crates/memra-engine/build.rs", "crates/memra-engine/src/cuda_arch.rs",
                               "crates/memra-probe/src/kernels.cu"])


KERNELS = r'''
extern "C" __global__ void dot(const float* a, const float* b, const float* c, float* o, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) o[i] = BODY;
}
'''


@unittest.skipUnless(HAVE_NVCC, "nvcc/cuobjdump absent")
class Compiled(unittest.TestCase):
    """The incident and review classes, reproduced with real ptxas output."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="numdiff-test-"))
        os.environ["NUMDIFF_WORK"] = str(self.tmp)
        nd.WORK = self.tmp
        # a private lock: toy compiles need not queue behind real memra builds
        self.lock, nd.BUILD_LOCK = nd.BUILD_LOCK, self.tmp / "build.lock"

    def tearDown(self):
        nd.BUILD_LOCK = self.lock
        shutil.rmtree(self.tmp, ignore_errors=True)

    def compile(self, name, text, *flags, kind="--fatbin"):
        """name may be DIR/FILE: the same file name built in two directories (two trees)."""
        src = self.tmp / f"{name}.cu"
        src.parent.mkdir(parents=True, exist_ok=True)
        src.write_text(text)
        out = self.tmp / (f"{name}.o" if kind == "-c" else f"{name}.fatbin")
        subprocess.run([NVCC, "-gencode", "arch=compute_120a,code=sm_120a", "-O3", kind, *flags,
                        "-o", str(out), str(src)], check=True, capture_output=True)
        return out

    def build(self, name, body, *flags):
        return nd.fingerprint(self.compile(name, KERNELS.replace("BODY", body), *flags))

    def verdict(self, a, b):
        return nd.compare_objects(a, b)[0]

    def test_same_source_same_flags_is_identical(self):
        a = self.build("a", "a[i] * b[i] + c[i]")
        b = self.build("b", "a[i] * b[i] + c[i]")
        self.assertEqual(self.verdict(a, b)["verdict"], "IDENTICAL")

    def test_fmad_flag_is_a_contraction_change(self):
        # the -fmad=false-per-file trap: character-identical source, different rounding
        a = self.build("a", "a[i] * b[i] + c[i]", "-fmad=true")
        b = self.build("b", "a[i] * b[i] + c[i]", "-fmad=false")
        row = self.verdict(a, b)
        self.assertEqual(row["verdict"], "CHANGED")
        self.assertIn("fma-contraction", row["classes"])
        self.assertEqual(row["unfused_pairs"], [0, 1])

    def test_cross_file_twin_names_the_flag(self):
        a = self.build("a", "a[i] * b[i] + c[i]")
        b = self.build("b", "a[i] * b[i] + c[i]", "-fmad=false")
        row = nd.twin_compare(a[0]["dot"], b[0]["dot"], ["-O3"], ["-O3", "-fmad=false"], a[1] == b[1])
        self.assertEqual(row["verdict"], "CHANGED")
        self.assertEqual(row["flags"], {"a": ["-O3"], "b": ["-O3", "-fmad=false"]})
        self.assertEqual(row["census"]["unfused_pairs"], [0, 1])
        # the pinned copy in the -fmad=false file is the same code as the original
        c = self.build("c", "__fmaf_rn(a[i], b[i], c[i])", "-fmad=false")
        self.assertEqual(nd.twin_compare(a[0]["dot"], c[0]["dot"], ["-O3"], ["-O3", "-fmad=false"], a[1] == c[1])
                         ["verdict"], "IDENTICAL-STREAM")

    def test_pinned_intrinsics_match_the_fused_original(self):
        a = self.build("a", "a[i] * b[i] + c[i]", "-fmad=true")
        b = self.build("b", "__fmaf_rn(a[i], b[i], c[i])", "-fmad=true")
        self.assertEqual(self.verdict(a, b)["verdict"], "IDENTICAL")

    def test_unfused_pin_is_a_contraction_change(self):
        a = self.build("a", "a[i] * b[i] + c[i]", "-fmad=true")
        b = self.build("b", "__fadd_rn(__fmul_rn(a[i], b[i]), c[i])", "-fmad=true")
        self.assertIn("fma-contraction", self.verdict(a, b)["classes"])

    def test_reassociated_sum_is_flagged(self):
        a = self.build("a", "(a[i] + b[i]) + c[i]")
        b = self.build("b", "a[i] + (b[i] + c[i])")
        row = self.verdict(a, b)
        self.assertEqual(row["verdict"], "CHANGED")
        self.assertEqual(row["classes"], ["same-fp-ops"])

    def test_fast_math_is_an_approximation_change(self):
        a = self.build("a", "a[i] / b[i] + __expf(c[i]) * 0.f + expf(c[i])")
        b = self.build("b", "a[i] / b[i] + __expf(c[i]) * 0.f + expf(c[i])", "--use_fast_math")
        row = self.verdict(a, b)
        self.assertEqual(row["verdict"], "CHANGED")
        self.assertTrue({"approximation", "fp-ops", "fma-contraction"} & set(row["classes"]), row)

    PACK = r'''extern "C" __global__ void pack4(const float* x, const float* y, float4* o, float s) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  float a = x[i], b = y[i];
  float p0 = a + s, p1 = a * s, p2 = b * s, p3 = b + s;
  o[i] = make_float4(LANES);
}'''

    def test_swapped_vector_lanes_are_changed(self):
        a = nd.fingerprint(self.compile("a", self.PACK.replace("LANES", "p0, p1, p2, p3")))
        b = nd.fingerprint(self.compile("b", self.PACK.replace("LANES", "p0, p1, p3, p2")))
        self.assertEqual(self.verdict(a, b)["verdict"], "CHANGED")

    UNPACK = r'''extern "C" __global__ void unpack(const unsigned* q, float* o, float s) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned v = q[i];
  signed char b0 = (signed char)(v & 0xff), b1 = (signed char)((v >> 8) & 0xff);
  signed char b2 = (signed char)((v >> 16) & 0xff), b3 = (signed char)(v >> 24);
  o[i] = ((float)b0 * s + (float)X) * s + (float)Y + (float)b3;
}'''

    def test_swapped_bytes_are_changed(self):
        a = nd.fingerprint(self.compile("a", self.UNPACK.replace("X", "b1").replace("Y", "b2")))
        b = nd.fingerprint(self.compile("b", self.UNPACK.replace("X", "b2").replace("Y", "b1")))
        self.assertEqual(self.verdict(a, b)["verdict"], "CHANGED")

    def test_launch_bounds_are_attributes(self):
        k = 'extern "C" __global__ void __launch_bounds__(N) sc(const float* x, float* o, float s) { int i = blockIdx.x * blockDim.x + threadIdx.x; o[i] = x[i] * s; }'
        a = nd.fingerprint(self.compile("a", k.replace("N", "256")))
        b = nd.fingerprint(self.compile("b", k.replace("N", "64")))
        row = self.verdict(a, b)
        self.assertEqual(row["verdict"], "CHANGED")
        self.assertIn("attributes", row["classes"])

    def test_cluster_shape_is_an_attribute(self):
        k = r'''#include <cooperative_groups.h>
namespace cg = cooperative_groups;
extern "C" __global__ void __cluster_dims__(1, CY, 1) cl_sum(const float* x, float* o) {
    __shared__ float part;
    cg::cluster_group cl = cg::this_cluster();
    if (threadIdx.x == 0) part = x[blockIdx.y];
    cl.sync();
    if (threadIdx.x == 0) {
        float s = 0.f;
        for (unsigned r = 0; r < cl.num_blocks(); ++r) s += *cl.map_shared_rank(&part, r);
        o[blockIdx.y] = s;
    }
    cl.sync();
}'''
        a = nd.fingerprint(self.compile("a", k.replace("CY", "2")))
        b = nd.fingerprint(self.compile("b", k.replace("CY", "4")))
        self.assertIn("attributes", self.verdict(a, b)["classes"])

    def test_switched_device_global_is_data(self):
        k = '__device__ float A[8];\n__device__ float B[8];\nextern "C" __global__ void rd(const int* idx, float* o) { int i = threadIdx.x; o[i] = G[idx[i] & 7]; }'
        a = nd.fingerprint(self.compile("a", k.replace("G[", "A[")))
        b = nd.fingerprint(self.compile("b", k.replace("float A[8];\n__device__ float B[8];", "float B[8];\n__device__ float A[8];").replace("G[", "B[")))
        row = self.verdict(a, b)
        self.assertEqual(row["verdict"], "CHANGED")
        self.assertIn("data", row["classes"])

    def test_twin_with_a_different_constant_table_is_changed(self):
        k = '__constant__ signed char kv[16] = {-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,LAST};\nextern "C" __global__ void deq(const unsigned char* q, float* o, float d) { int i = blockIdx.x * blockDim.x + threadIdx.x; o[i] = d * kv[q[i] & 15]; }'
        a = nd.fingerprint(self.compile("a", k.replace("LAST", "113"), kind="-c"))
        b = nd.fingerprint(self.compile("b", k.replace("LAST", "112"), kind="-c"))
        row = nd.twin_compare(a[0]["deq"], b[0]["deq"], same_data=a[1] == b[1])
        self.assertEqual(row["verdict"], "CHANGED")
        self.assertIn("data", row["classes"])

    def test_adding_a_kernel_leaves_the_others_identical(self):
        one = KERNELS.replace("BODY", "a[i] * b[i] + c[i]")
        two = 'extern "C" __global__ void first(float* o) { o[threadIdx.x] = 1.f; }\n' + one
        rows = {r["kernel"]: r["verdict"] for r in nd.compare_objects(nd.fingerprint(self.compile("a", one)),
                                                                      nd.fingerprint(self.compile("b", two)))}
        self.assertEqual(rows, {"dot": "IDENTICAL", "first": "ADDED"})

    LAUNCH = r'''__global__ void colsum(const float* x, float* o, int n, int t) {
  int c = threadIdx.x, r0 = blockIdx.x * t;
  float s = 0.f;
  for (int r = r0; r < r0 + t && r < n; ++r) s += x[r * 32 + c];
  atomicAdd(&o[c], s);
}
extern "C" int launch_colsum(const float* x, float* o, int n, void* stream) {
  const int t = SPLIT;
  colsum<<<(n + t - 1) / t, 32, 0, (cudaStream_t)stream>>>(x, o, n, t);
  return (int)cudaGetLastError();
}'''

    def test_host_launcher_change_moves_the_host_digest(self):
        a = self.compile("a", self.LAUNCH.replace("SPLIT", "128"), kind="-c")
        b = self.compile("b", self.LAUNCH.replace("SPLIT", "64"), kind="-c")
        c = self.compile("c", self.LAUNCH.replace("SPLIT", "128"), kind="-c")
        self.assertEqual(self.verdict(nd.fingerprint(a), nd.fingerprint(b))["verdict"], "IDENTICAL")
        self.assertNotEqual(nd.host_digest(a), nd.host_digest(b))
        self.assertEqual(nd.host_digest(a), nd.host_digest(c))

    def test_anonymous_namespace_build_ids_do_not_count(self):
        # cudafe names an anonymous namespace with a per-compilation id: two builds of the same
        # file differ in kernel names and in the host registration strings, nothing else
        k = r'''namespace { __global__ void sc(const float* x, float* o, float s) { o[threadIdx.x] = x[threadIdx.x] * s; } }
extern "C" int launch_sc(const float* x, float* o, float s, void* st) { sc<<<1, 32, 0, (cudaStream_t)st>>>(x, o, s); return 0; }'''
        a = self.compile("ta/unit", k, kind="-c")
        b = self.compile("tb/unit", k, kind="-c")
        rows = nd.compare_objects(nd.fingerprint(a), nd.fingerprint(b))
        self.assertEqual([r["verdict"] for r in rows], ["IDENTICAL"])
        self.assertEqual(nd.host_digest(a), nd.host_digest(b))

    LAUNCHERS = r'''__global__ void kscale(float* o, float s) { o[threadIdx.x] *= s; }
__global__ void kshift(float* o, float s) { o[threadIdx.x] += s; }
FIRST
SECOND'''
    SCALE = 'extern "C" int NAME(float* o, float s, void* st) { kscale<<<1, 32, 0, (cudaStream_t)st>>>(o, s); return 0; }'
    SHIFT = 'extern "C" int NAME(float* o, float s, void* st) { kshift<<<1, 32, 0, (cudaStream_t)st>>>(o, s); return 0; }'

    def launchers(self, name, first, second):
        return self.compile(name, self.LAUNCHERS.replace("FIRST", first).replace("SECOND", second), kind="-c")

    def test_exported_launcher_names_trading_bodies_is_a_host_change(self):
        a = self.launchers("ta/u", self.SCALE.replace("NAME", "launch_scale"), self.SHIFT.replace("NAME", "launch_shift"))
        b = self.launchers("tb/u", self.SCALE.replace("NAME", "launch_shift"), self.SHIFT.replace("NAME", "launch_scale"))
        self.assertEqual({r["verdict"] for r in nd.compare_objects(nd.fingerprint(a), nd.fingerprint(b))}, {"IDENTICAL"})
        self.assertNotEqual(nd.host_digest(a), nd.host_digest(b))

    # ---- rev, with git, the build database and the compiles replaced by fixtures

    def rev(self, base_units, head_units, changed, files=None, included=()):
        """base_units/head_units: {(src, unit): (object path, is -c unit)}."""
        def git(repo, *args, text=True):
            if args[0] == "rev-parse":
                return args[1] + "\n"
            if args[:2] == ("diff", "--name-only"):
                paths = args[args.index("--") + 1:]
                return "\n".join(n for n in changed if any(n == p or n.startswith(p) for p in paths)) + "\n"
            if args[:2] == ("diff", "-U0"):
                return ""
            raise AssertionError(args)

        def entry(k, v):
            out = "/x/" + k[1]
            return {"src": k[0], "out": out, "argv": ["-c" if v[1] else "--fatbin", "-o", out, k[0]],
                    "tree": "/t", "obj": v[0]}

        def build_rev(repo, rev, arch):
            u = base_units if rev == "base" else head_units
            return Path("/t"), {k: entry(k, v) for k, v in u.items()}

        saved = (nd.git, nd.build_rev, nd.compile_one, nd.included_by)
        nd.git, nd.build_rev = git, build_rev
        nd.compile_one = lambda e, tree, outdir: e["obj"]
        nd.included_by = lambda tree, src: src in included
        try:
            return nd.rev_command(types.SimpleNamespace(repo="/r", base="base", head="head", arch="120a", files=files))
        finally:
            nd.git, nd.build_rev, nd.compile_one, nd.included_by = saved

    def objs(self):
        one = KERNELS.replace("BODY", "a[i] * b[i] + c[i]")
        return {"one": self.compile("one", one), "one2": self.compile("one2", one),
                "fmad": self.compile("fmad", one, "-fmad=false"),
                "two": self.compile("two", 'extern "C" __global__ void first(float* o) { o[threadIdx.x] = 1.f; }\n' + one),
                "l128": self.compile("l128", self.LAUNCH.replace("SPLIT", "128"), kind="-c"),
                "l64": self.compile("l64", self.LAUNCH.replace("SPLIT", "64"), kind="-c")}

    def test_rev_compares_every_variant_of_a_source(self):
        o = self.objs()
        k = "crates/memra-engine/cu/f.cu"
        base = {("cu/f.cu", "f.fatbin"): (o["one"], False), ("cu/f.cu", "f_kf8.fatbin"): (o["one"], False)}
        head = {("cu/f.cu", "f.fatbin"): (o["fmad"], False), ("cu/f.cu", "f_kf8.fatbin"): (o["one2"], False)}
        rep = self.rev(base, head, [k])
        self.assertEqual(rep["verdict"], "CHANGED")
        self.assertEqual(set(rep["files"]), {"cu/f.cu [f.fatbin]", "cu/f.cu [f_kf8.fatbin]"})

    def test_rev_added_kernel_is_changed(self):
        o = self.objs()
        rep = self.rev({("cu/f.cu", "f.fatbin"): (o["one"], False)}, {("cu/f.cu", "f.fatbin"): (o["two"], False)},
                       ["crates/memra-engine/cu/f.cu"])
        self.assertEqual((rep["verdict"], rep["summary"]), ("CHANGED", {"IDENTICAL": 1, "ADDED": 1}))

    def test_rev_one_sided_and_unbuilt_units(self):
        o = self.objs()
        rep = self.rev({}, {("cu/f.cu", "f.fatbin"): (o["one"], False)}, ["crates/memra-engine/cu/f.cu"])
        self.assertEqual(rep["verdict"], "CHANGED")
        rep = self.rev({("cu/f.cu", "f.fatbin"): (o["one"], False)}, {("cu/f.cu", "f.fatbin"): (o["one2"], False)},
                       ["crates/memra-engine/cu/f.cu", "crates/memra-engine/cu/g_stub.cu"])
        self.assertEqual(rep["verdict"], "UNCOVERED")
        # an included .cu acts as a header: every unit is compared
        rep = self.rev({("cu/f.cu", "f.fatbin"): (o["one"], False)}, {("cu/f.cu", "f.fatbin"): (o["one2"], False)},
                       ["crates/memra-engine/cu/common_impl.cu"], included={"cu/common_impl.cu"})
        self.assertEqual(rep["verdict"], "IDENTICAL")
        self.assertIn("cu/f.cu", rep["files"])

    def test_rev_a_unit_another_unit_includes_recompares_its_includers(self):
        o = self.objs()
        base = {("cu/k.cu", "k.fatbin"): (o["one"], False), ("cu/m.cu", "m.o"): (o["l128"], True)}
        head = {("cu/k.cu", "k.fatbin"): (o["one2"], False), ("cu/m.cu", "m.o"): (o["l64"], True)}
        rep = self.rev(base, head, ["crates/memra-engine/cu/k.cu"], included={"cu/k.cu"})
        self.assertEqual(set(rep["files"]), {"cu/k.cu", "cu/m.cu"})
        self.assertEqual(rep["verdict"], "KERNELS-IDENTICAL")

    def test_rev_a_header_change_with_nothing_compared_is_uncovered(self):
        rep = self.rev({}, {}, ["crates/memra-engine/cu/tbl.hpp"])
        self.assertEqual(rep["verdict"], "UNCOVERED")

    def test_recorded_nvcc_environment_is_replayed(self):
        tree = self.tmp / "tree"
        (tree / "cu").mkdir(parents=True)
        (tree / "cu/k.cu").write_text('extern "C" __global__ void k(float* o) {\n#ifdef ALT\n o[0] = 2.f;\n#else\n o[0] = 1.f;\n#endif\n}\n')
        def entry(env):
            out = str(tree / "out/k.fatbin")
            return {"src": "cu/k.cu", "cwd_rel": ".", "tree": str(tree), "out": out, "env": env,
                    "argv": ["-gencode", "arch=compute_120a,code=sm_120a", "-O3", "--fatbin", "-o", out, "cu/k.cu"]}
        plain = nd.compile_one(entry({}), tree, self.tmp / "objs")
        alt = nd.compile_one(entry({"NVCC_APPEND_FLAGS": "-DALT=1"}), tree, self.tmp / "objs")
        self.assertNotEqual(plain, alt)
        self.assertEqual(nd.compare_objects(nd.fingerprint(plain), nd.fingerprint(alt))[0]["verdict"], "CHANGED")
        self.assertEqual(nd.numeric_flags(nd.flags_of(entry({"NVCC_APPEND_FLAGS": "-DALT=1"}))),
                         ["-O3", "-DALT=1"])

    def test_rev_launcher_change_is_kernels_identical(self):
        o = self.objs()
        rep = self.rev({("cu/m.cu", "m.o"): (o["l128"], True)}, {("cu/m.cu", "m.o"): (o["l64"], True)},
                       ["crates/memra-engine/cu/m.cu"])
        self.assertEqual(rep["verdict"], "KERNELS-IDENTICAL")
        self.assertEqual(rep["host_changed"], ["cu/m.cu (host code of m.o)"])

    def test_rev_build_inputs_are_host_and_recompile_everything(self):
        o = self.objs()
        units = {("cu/f.cu", "f.fatbin"): (o["one"], False), ("cu/g.cu", "g.fatbin"): (o["one"], False)}
        rep = self.rev(units, units, ["crates/memra-engine/src/cuda_arch.rs"])
        self.assertEqual(rep["verdict"], "KERNELS-IDENTICAL")
        self.assertEqual(set(rep["files"]), {"cu/f.cu", "cu/g.cu"})

    def test_rev_narrowed_files_are_partial(self):
        o = self.objs()
        units = {("cu/f.cu", "f.fatbin"): (o["one"], False), ("cu/g.cu", "g.fatbin"): (o["one"], False)}
        rep = self.rev(units, units, ["crates/memra-engine/cu/f.cu", "crates/memra-engine/cu/g.cu"], files=["cu/f.cu"])
        self.assertTrue(rep["partial"])
        rep = self.rev(units, units, ["crates/memra-engine/cu/f.cu"], files=["cu/f.cu"])
        self.assertFalse(rep["partial"])


@unittest.skipUnless(shutil.which("cargo") and Path(NVCC).exists(), "cargo/nvcc absent")
class Capture(unittest.TestCase):
    """capture() against a one-file crate named memra-engine whose build.rs calls MEMRA_NVCC."""

    BUILD_RS = r'''use std::process::Command;
fn main() {
    let leaked: Vec<String> = std::env::vars().map(|(k, _)| k)
        .filter(|k| (k.starts_with("MEMRA_") && k != "MEMRA_NVCC" && k != "MEMRA_CUDA_ARCH") || k.starts_with("NVCC_"))
        .collect();
    std::fs::write(std::env::var("NUMDIFF_TEST_MARK").unwrap(), leaked.join(",")).unwrap();
    let nvcc = std::env::var("MEMRA_NVCC").unwrap();
    let s = Command::new(nvcc).args(["--version", "-o", "k.fatbin", "cu/k.cu"])
        .env("NVCC_APPEND_FLAGS", "-DFROM_BUILD_RS=1").status().unwrap();
    assert!(s.success());
    if std::env::var("NUMDIFF_TEST_FAIL").is_ok() { panic!("a later nvcc call failed"); }
    if std::env::var("NUMDIFF_TEST_LIBFAIL").is_ok() { println!("cargo:rustc-cfg=libfail"); }
}
'''

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="numdiff-capture-"))
        nd.WORK = self.tmp / "work"
        nd.WORK.mkdir()
        # a private lock: the toy crate need not wait for real memra builds
        self.lock, nd.BUILD_LOCK = nd.BUILD_LOCK, self.tmp / "build.lock"
        self.tree = self.tmp / "tree"
        (self.tree / "src").mkdir(parents=True)
        (self.tree / "Cargo.toml").write_text('[package]\nname = "memra-engine"\nversion = "0.0.0"\nedition = "2021"\n')
        (self.tree / "src/lib.rs").write_text('#[cfg(libfail)]\ncompile_error!("the crate breaks after build.rs");\n')
        (self.tree / "build.rs").write_text(self.BUILD_RS)
        (self.tree / ".cargo").mkdir()
        (self.tree / ".cargo/config.toml").write_text('[env]\nCPATH = "/opt/toy/include"\n')
        self.mark = self.tmp / "leaked.txt"
        self.saved = dict(os.environ)
        os.environ.update(NUMDIFF_TEST_MARK=str(self.mark), MEMRA_FA_PP_MINBLOCKS="2", NVCC_APPEND_FLAGS="-G")

    def tearDown(self):
        nd.BUILD_LOCK = self.lock
        os.environ.clear()
        os.environ.update(self.saved)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_records_commands_with_the_build_seams_cleared(self):
        db = nd.capture(self.tree, "120a", "k1")
        self.assertEqual([e["src"] for e in db], ["cu/k.cu"])
        self.assertEqual(self.mark.read_text(), "")
        # what the build itself gives nvcc is recorded: build.rs's Command::env and cargo [env]
        self.assertEqual(db[0]["env"], {"NVCC_APPEND_FLAGS": "-DFROM_BUILD_RS=1", "CPATH": "/opt/toy/include"})
        # only the recorded commands are kept: the build's target dir is gone
        self.assertFalse((nd.WORK / "target" / "k1-120a").exists())

    def test_a_crate_error_after_build_rs_keeps_the_complete_log(self):
        # a historical commit whose lib does not compile: build.rs ran every nvcc call first
        os.environ["NUMDIFF_TEST_LIBFAIL"] = "1"
        db = nd.capture(self.tree, "120a", "k3")
        self.assertEqual([e["src"] for e in db], ["cu/k.cu"])
        self.assertIn("build.rs completed", db[0]["capture_note"])

    def test_a_failed_build_raises_and_caches_nothing(self):
        os.environ["NUMDIFF_TEST_FAIL"] = "1"
        with self.assertRaises(RuntimeError):
            nd.capture(self.tree, "120a", "k2")
        self.assertFalse((nd.WORK / "db" / "k2-120a.json").exists())


if __name__ == "__main__":
    unittest.main()
