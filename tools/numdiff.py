#!/usr/bin/env python3
"""numdiff: which compiled CUDA kernels a change can make compute different bits, and how.

For two revisions of memra (or two kernels of one build) it compares what ptxas emitted, without
a GPU. Per kernel:

  IDENTICAL       the same SASS as ptxas wrote it (physical registers, predicates and selectors;
                  only `.reuse` hints dropped and in-listing branch targets mapped to instruction
                  indices), the same kernel attributes (.nv.info records with symbol indices
                  resolved to names: launch bounds, cluster shape, register count, parameter
                  bank; static shared and local sizes; per-kernel constant banks; relocations),
                  no call outside its own listing, and the same object-wide user data (constant
                  banks, initialized __device__ data and the relocations that bind them). Same
                  inputs and launch parameters give the same bits, for the nvcc in use.
  CHANGED         anything else, with the classes that moved:
                    fma-contraction  FFMA traded against FMUL/FADD (ptxas fused differently)
                    approximation    MUFU (approximate transcendental / reciprocal) set moved
                    precision        half/double ops or conversions (H*2, D*, F2F, F2FP, I2F, F2I)
                    tensor-core      MMA instructions moved (shape, type, count)
                    reduction        warp shuffles, float REDUX or floating-point atomics moved
                    fp-ops           other float ops added or removed (an algorithm or math change)
                    same-fp-ops      the float op multiset is unchanged: order, registers,
                                     addressing or scheduling moved (equal bits not shown)
                    attributes       the stream is the same but the kernel attributes moved
                    data             the object's constant or __device__ data changed
                    build-flag       the file's nvcc flags changed between the revisions
                    external-call    the kernel calls code outside its own listing
  RENAMED         a removed kernel whose identical code reappears under a new symbol
  ADDED / REMOVED a kernel present on one side only

IDENTICAL is a proof about the compiled code for the nvcc in use. CHANGED says the code moved and
which float classes moved; it does not prove that bits differ (a register reallocation changes
the stream and nothing else).

`rev` gives one verdict for the whole change, the first that holds:
  CHANGED            a kernel CHANGED, ADDED or REMOVED, or a compiled unit exists on one side only
  UNCOVERED          a touched source is compiled on neither side for this arch
  UNCOVERED          also: kernel-capable inputs changed and nothing was compared
  KERNELS-IDENTICAL  every kernel IDENTICAL or RENAMED, but host code changed: Rust sources,
                     manifests, build inputs, the host launchers inside a `-c` unit (their symbols
                     and bodies: launch geometry, split counts, cuBLAS/cuBLASLt setup, which
                     exported name runs which body) or CUDA sources outside
                     crates/memra-engine/cu. Which kernel runs and how it is launched is not covered
  IDENTICAL          everything compared is identical and no host code changed
Headers (any non-.cu file under cu/, C/C++/CUDA headers elsewhere in crates/), a .cu another
unit #includes, and the build inputs (build.rs, cuda_arch.rs, the manifests, .cargo/config.toml)
recompile every unit; the build inputs also key the recorded commands.
A report whose compared set is narrower than the change (--files) is marked PARTIAL.

Not covered (a verdict says nothing about them): units build.rs compiles only under a MEMRA_* seam
(the CUTLASS units behind MEMRA_CUTLASS / MEMRA_DSV4_CUTLASS: the capture clears MEMRA_*), nvcc
calls the capture cannot read as one .cu with a literal `-o` (dropped), code compiled at run time
(PTX JIT, NVRTC) and the algorithm cuBLAS/cuBLASLt picks at run time. A header change that reaches
only those units can therefore come out IDENTICAL.

Usage:
  numdiff.py rev BASE HEAD [--repo ~/projects/memra] [--arch 120a] [--out FILE] [--files cu/x.cu ...]
  numdiff.py twins REV FILE:KERNEL FILE:KERNEL [--repo] [--arch]
                                               a bitwise-twin claim between two kernels of one revision,
                                               across files (flags differ) or in one file
  numdiff.py objs BASE_OBJ HEAD_OBJ            compare two compiled objects (fatbin, .o or cubin)
  numdiff.py twin OBJ KERNEL_A KERNEL_B        compare two kernels of one object
  numdiff.py sass OBJ                          print per-kernel fingerprints and float profiles

`rev` archives both revisions into scratch, records build.rs's nvcc commands once per distinct
(build inputs, nvcc) with MEMRA_NVCC pointed at a pass-through wrapper and the caller's MEMRA_*
and NVCC_* environment cleared (`cargo check -p memra-engine`; the wrapper also records the NVCC_*
and include-path variables the build itself gives nvcc, which are replayed), recompiles every recorded variant
of the .cu files the diff touches (every unit when a header or build.rs changed) with those exact
commands, and compares every kernel in them. Builds take ~/.cache/memra-rig-build.lock.

A twin of different size (a fused kernel against the standalone one it copies) cannot be IDENTICAL,
and its classes come from the extra work as much as from the copied part: numdiff reports a
contraction census per side (FFMA count, FMUL->FADD pairs left unfused) and the numeric flags of
both files, and leaves the decision to an equivalence check of the copied region.
"""
import argparse
import collections
import difflib
import hashlib
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORK = Path(os.environ.get("NUMDIFF_WORK", Path.home() / ".cache/numdiff"))
BUILD_LOCK = Path.home() / ".cache/memra-rig-build.lock"
REAL_NVCC = os.environ.get("NUMDIFF_REAL_NVCC", "/usr/local/cuda/bin/nvcc")

# ---------------------------------------------------------------------------------- SASS parse

INSN = re.compile(r"^\s*/\*([0-9a-f]{4,})\*/\s+(.*?)\s*;\s*/\*\s*0x[0-9a-f]+\s*\*/\s*$")
FUNC = re.compile(r"^\s*Function\s*:\s*(\S+)\s*$")
HEX = re.compile(r"\b0x([0-9a-f]+)\b")
BRANCHY = ("BRA", "BRX", "JMP", "JMX", "CALL", "BSSY", "BREAK", "SSY", "PBK", "PCNT", "BRK", "CONT",
           "WARPSYNC", "BMOV", "RET", "JCAL", "PRET")

# an optional U prefix: the uniform datapath (UFFMA, UFMUL, UI2F, UF2I, ...)
FP_FAMILIES = [
    ("fma-contraction", re.compile(r"^U?(FFMA|FMUL|FADD|FFMA2|FMUL2|FADD2)(\.|$)")),
    ("approximation", re.compile(r"^U?(MUFU|FCHK)(\.|$)")),
    ("precision", re.compile(r"^U?(HFMA2|HADD2|HMUL2|HMNMX2|HSETP2|HSET2|DFMA|DADD|DMUL|DSETP|F2F|F2FP|I2F|I2FP|F2I|F2IP|FRND)(\.|$)")),
    ("tensor-core", re.compile(r"^(HMMA|IMMA|QMMA|OMMA|BMMA|DMMA|HGMMA|IGMMA|QGMMA|UTC\w*MMA\w*)(\.|$)")),
    ("reduction", re.compile(r"^(SHFL|REDUX|UREDUX|REDG|RED|ATOMG|ATOM|ATOMS)(\.|$)")),
    ("fp-ops", re.compile(r"^U?(FMNMX|FSETP|FSET|FSEL|FSWZADD|FMNMX3|FADD3)(\.|$)")),
]
FLOAT_TYPE = re.compile(r"\.(F32|F16|BF16|F64|F16x2|BF16x2|F32x2|F32x4|FADD|ADD\.F)(\.|$)")


def fp_family(opcode):
    for name, rx in FP_FAMILIES:
        if rx.match(opcode):
            if name == "reduction" and opcode.split(".")[0] != "SHFL":
                # only floating-point atomics and float REDUX reorder a float sum
                if not FLOAT_TYPE.search(opcode):
                    return None
            return name
    return None


def scalar_base(opcode):
    """(base opcode without the uniform prefix, scalar ops it stands for): FFMA2 is two FFMA."""
    base = opcode.split(".")[0]
    if base.startswith("U") and fp_family(base[1:]):
        base = base[1:]
    if base in ("FFMA2", "FMUL2", "FADD2"):
        return base[:-1], 2
    return base, 1


def parse_sass(text):
    """Return {function_name: [(addr, raw_instruction_text), ...]} from cuobjdump -sass output."""
    funcs, cur = {}, None
    for line in text.splitlines():
        m = FUNC.match(line)
        if m:
            cur = m.group(1)
            funcs[cur] = []
            continue
        if cur is None:
            continue
        m = INSN.match(line)
        if m:
            body = m.group(2).replace("{", "").replace("}", "").strip()
            funcs[cur].append((int(m.group(1), 16), body))
    return funcs


def split_insn(body):
    pred = ""
    if body.startswith("@"):
        pred, _, body = body.partition(" ")
    opcode, _, operands = body.strip().partition(" ")
    return pred, opcode, operands.strip()


def canonical(insns):
    """The stream as ptxas wrote it, with `.reuse` hints dropped and in-listing branch targets
    mapped to instruction indices (addresses shift when code before them moves). Registers,
    predicates and byte selectors stay literal: register tuples (.64/.128 operands, R2P/P2R
    predicate vectors) make any renaming lossy. Returns (lines, external call flag)."""
    index = {addr: i for i, (addr, _) in enumerate(insns)}
    out, external = [], False
    for addr, body in insns:
        pred, opcode, operands = split_insn(body)
        operands = operands.replace(".reuse", "")
        base = opcode.split(".")[0]
        if base in BRANCHY:
            def target(m):
                nonlocal external
                a = int(m.group(1), 16)
                if a in index:
                    return f"L{index[a]}"
                if base in ("CALL", "JCAL"):
                    external = True
                return f"0x{m.group(1)}"
            operands = HEX.sub(target, operands)
        out.append(f"{pred} {opcode} {operands}".strip())
    return out, external


CONST_MOVE = re.compile(r"^[A-Z]*R\d+, -RZ, RZ, ")


def float_op(body):
    """The opcode when the instruction is floating-point arithmetic, else None. ptxas materializes
    constants with `HFMA2 Rd, -RZ, RZ, imm, imm` (a move, not half math): not counted."""
    _, opcode, operands = split_insn(body)
    if not fp_family(opcode):
        return None
    if opcode.split(".")[0] in ("HFMA2", "UHFMA2") and CONST_MOVE.match(operands):
        return None
    return opcode


def fp_profile(insns):
    c = collections.Counter()
    for _, body in insns:
        op = float_op(body)
        if op:
            c[op] += 1
    return c


def fp_sequence(insns):
    return [op for op in (float_op(body) for _, body in insns) if op]


GPR = re.compile(r"(?<![A-Za-z0-9_])R(\d+)(?![0-9A-Za-z_])")


def def_width(opcode):
    base = opcode.split(".")[0]
    if ".128" in opcode or base in ("HMMA", "IMMA", "QMMA", "OMMA", "BMMA", "DMMA"):
        return 4
    if ".64" in opcode or ".WIDE" in opcode or base.startswith("D"):
        return 2
    return 1


def unfused_pairs(lines):
    """Census: FMULs whose result is read once, by an FADD, before it is redefined, the spots
    where an -fmad=true build would contract. Def-use runs on physical register numbers inside
    straight-line runs (a branch ends a run), with tuple widths from the opcode. A census, not a
    bound: tuple defs the opcode does not name (LDSM, some MMA forms) are approximated."""
    last_def, uses, fmuls = {}, collections.defaultdict(list), []
    for i, line in enumerate(lines):
        body = line.split(" ", 1)[1] if line.startswith("@") else line
        opcode, _, operands = body.partition(" ")
        base = opcode.split(".")[0]
        if base in BRANCHY:
            last_def.clear()
            continue
        first, _, rest = operands.partition(",")
        dst = GPR.fullmatch(first.strip())
        srcs = rest if dst else operands
        for m in GPR.finditer(srcs):
            r = int(m.group(1))
            if r in last_def:
                uses[last_def[r]].append(base)
        if dst:
            for k in range(def_width(opcode)):
                last_def[int(dst.group(1)) + k] = i
            if base == "FMUL":
                fmuls.append(i)
    return sum(1 for i in fmuls if uses[i] == ["FADD"])


# ------------------------------------------------------------------------------- objects, ELF

SHT_SYMTAB, SHT_RELA, SHT_NOBITS, SHT_REL = 2, 4, 8, 9
SHF_ALLOC = 0x2


class Elf:
    """A 64-bit little-endian ELF: sections (with type, flags, link, info), symbols and resolved
    relocations. Enough for cubins and host objects."""

    def __init__(self, path):
        b = Path(path).read_bytes()
        if b[:4] != b"\x7fELF" or b[4] != 2:
            raise ValueError(f"{path}: not a 64-bit ELF")
        self.b = b
        e_shoff, = struct.unpack_from("<Q", b, 0x28)
        e_shentsize, e_shnum, e_shstrndx = struct.unpack_from("<HHH", b, 0x3A)
        self.sections = []
        for i in range(e_shnum):
            name, typ, flags, _addr, off, size, link, info, _align, entsize = struct.unpack_from(
                "<IIQQQQIIQQ", b, e_shoff + i * e_shentsize)
            self.sections.append({"name_off": name, "type": typ, "flags": flags, "off": off,
                                  "size": size, "link": link, "info": info, "entsize": entsize})
        strtab = self.sections[e_shstrndx]["off"] if e_shnum else 0
        for s in self.sections:
            s["name"] = self._str(strtab, s["name_off"])
            s["data"] = b"" if s["type"] == SHT_NOBITS else b[s["off"]:s["off"] + s["size"]]
        self.by_name = {s["name"]: s for s in self.sections}
        self.symbols, self.symdefs, self.syminfo, self.symother = [], [], [], []
        for s in self.sections:
            if s["type"] == SHT_SYMTAB:
                names = self.sections[s["link"]]["off"]
                for i in range(s["size"] // 24):
                    st_name, st_info, other, shndx, value, size = struct.unpack_from(
                        "<IBBHQQ", b, s["off"] + i * 24)
                    n = self._str(names, st_name)
                    if not n and (st_info & 0xF) == 3 and shndx < len(self.sections):  # STT_SECTION
                        n = self.sections[shndx]["name"]
                    self.symbols.append(n)
                    self.symdefs.append((shndx, value, size))
                    self.syminfo.append(st_info)
                    self.symother.append(other)
                break

    def _str(self, base, off):
        end = self.b.index(b"\0", base + off)
        return self.b[base + off:end].decode(errors="replace")

    def is_reloc(self, s):
        return (s["type"] in (SHT_RELA, SHT_REL) or ".rela" in s["name"] or ".rel." in s["name"]) \
            and s["entsize"] in (16, 24) and s["type"] != SHT_NOBITS

    def symbol_target(self, sym):
        """What a relocation binds to: the symbol's name with compiler numbering normalized
        (the host binds user __device__ globals by name), plus, for a symbol defined here, its
        storage (section, offset, size)."""
        if sym >= len(self.symbols):
            return f"#{sym}"
        name = normalize_symbol(self.symbols[sym])
        shndx, value, size = self.symdefs[sym]
        if 0 < shndx < len(self.sections) and shndx < 0xFF00:
            return f"{name}@{normalize_symbol(self.sections[shndx]['name'])}+{value:#x}:{size}"
        return name

    def relocs(self, s):
        """[(offset, type, target, addend)] of a relocation section."""
        out = []
        for i in range(s["size"] // s["entsize"]):
            if s["entsize"] == 24:
                off, info, addend = struct.unpack_from("<QQq", self.b, s["off"] + i * 24)
            else:
                (off, info), addend = struct.unpack_from("<QQ", self.b, s["off"] + i * 16), None
            out.append((off, info & 0xFFFFFFFF, self.symbol_target(info >> 32), addend))
        return out

    def target(self, s):
        return self.sections[s["info"]]["name"] if s["info"] < len(self.sections) else f"#{s['info']}"


# per-compilation ids cudafe puts in anonymous-namespace names (they differ between two builds of
# the same file) and the numbering ptxas gives string literals, unnamed objects and helpers
BUILD_IDS = [(r"_INTERNAL_[0-9a-f]{8}_", "_INTERNAL_"), (r"_GLOBAL__N__[0-9a-f]{8}_", "_GLOBAL__N__"),
             (r"tmpxft_[0-9a-f]{8}_[0-9a-f]{8}-\d+_", "tmpxft_")]
COMPILER_NUMBERING = [(re.compile(r"\$str\$\d+"), "$str$"), (re.compile(r"__unnamed_\d+"), "__unnamed_"),
                      (re.compile(r"\$__internal_\d+_\$"), "$__internal_$")] + \
    [(re.compile(rx), rep) for rx, rep in BUILD_IDS]
BUILD_IDS_BYTES = [(re.compile(rx.encode()), rep.encode()) for rx, rep in BUILD_IDS]


def normalize_symbol(name):
    """Drop the per-build ids of anonymous namespaces and the numbering ptxas gives string
    literals, unnamed objects and internal helpers: both move with unrelated edits or with the
    build directory. The storage and the data digest still pin what a symbol is."""
    for rx, rep in COMPILER_NUMBERING:
        name = rx.sub(rep, name)
    return name


def normalize_bytes(data):
    for rx, rep in BUILD_IDS_BYTES:
        data = rx.sub(rep, data)
    return data


def elf_sections(path):
    """{name: bytes} (NOBITS sections are empty)."""
    return {s["name"]: s["data"] for s in Elf(path).sections}


# .nv.info attributes whose payload starts with a symbol index (a function or a section symbol)
NVINFO_SYMBOL_ATTRS = {0x0A: "PARAM_CBANK", 0x11: "FRAME_SIZE", 0x12: "MIN_STACK_SIZE",
                       0x23: "MAX_STACK_SIZE", 0x2F: "REGCOUNT"}


def nvinfo_records(elf, data, self_name=None):
    """.nv.info records as text: format, attribute, payload; the symbol index in the attributes
    that carry one becomes the symbol's name (indices shift when other kernels come and go).
    Records are 4-byte headers (format, attribute, u16 value or payload size) plus the payload."""
    out, i = [], 0
    while i + 4 <= len(data):
        fmt, attr, v = data[i], data[i + 1], struct.unpack_from("<H", data, i + 2)[0]
        if fmt == 4:
            payload = data[i + 4:i + 4 + v]
            i += 4 + v
            if attr in NVINFO_SYMBOL_ATTRS and len(payload) >= 4:
                sym = struct.unpack_from("<I", payload)[0]
                name = elf.symbols[sym] if sym < len(elf.symbols) else f"#{sym}"
                if self_name:
                    name = name.replace(self_name, "@self")
                out.append((attr, normalize_symbol(name), payload[4:].hex()))
            else:
                out.append((attr, payload.hex()))
        else:
            out.append((attr, fmt, v))
            i += 4
    return out


def nvinfo_symbol(elf, rec):
    return rec[1] if len(rec) == 3 and rec[0] in NVINFO_SYMBOL_ATTRS else None


def kernel_attributes(elf, kernel):
    """Everything the cubin says about one kernel besides its SASS: its .nv.info and merc info
    records, the sizes of its NOBITS sections (static shared, local) and of its parameter bank,
    the contents of its other per-kernel sections (constant banks, capmerc text) and its
    relocations, plus the object-wide .nv.info records naming it. The kernel's own name becomes
    @self, so a renamed kernel keeps its attributes."""
    recs = []
    suffix = "." + kernel
    nk = normalize_symbol(kernel)
    text_index = next((i for i, s in enumerate(elf.sections) if s["name"] == ".text" + suffix), None)
    for s in elf.sections:
        n = s["name"]
        if not n.endswith(suffix):
            continue
        prefix = n[:-len(suffix)]
        if prefix == ".text":
            continue
        if elf.is_reloc(s):
            recs.append((prefix, [(o, t, sym.replace(nk, "@self"), a) for o, t, sym, a in elf.relocs(s)]))
        elif prefix in (".nv.info", ".nv.merc.nv.info"):
            recs.append((prefix, nvinfo_records(elf, s["data"], kernel)))
        elif s["type"] == SHT_NOBITS or prefix == ".nv.constant0":
            recs.append((prefix, "size", s["size"]))
        elif prefix == ".nv.capmerc.text" and len(s["data"]) >= 4 and \
                struct.unpack_from("<I", s["data"])[0] == text_index:
            # the capmerc header opens with the section index of the kernel's .text, which
            # shifts when other kernels come and go
            recs.append((prefix, "@text", hashlib.sha256(s["data"][4:]).hexdigest()))
        else:
            recs.append((prefix, hashlib.sha256(s["data"]).hexdigest()))
    # relocation sections that target one of the kernel's sections under another name
    for s in elf.sections:
        if elf.is_reloc(s) and not s["name"].endswith(suffix) and elf.target(s).endswith(suffix):
            recs.append((normalize_symbol(s["name"]).replace(nk, "@self"),
                         [(o, t, sym.replace(nk, "@self"), a) for o, t, sym, a in elf.relocs(s)]))
    for name in (".nv.info", ".nv.merc.nv.info"):
        s = elf.by_name.get(name)
        if s:
            for r in nvinfo_records(elf, s["data"]):
                if nvinfo_symbol(elf, r) == normalize_symbol(kernel):
                    recs.append((name, (r[0], "@self", r[2])))
    return json.dumps(sorted(recs, key=lambda r: json.dumps(r)), default=str)


OBJECT_DATA = re.compile(r"^\.nv\.(constant\d+|global\.init|global)$")


def data_digest(cubins):
    """Object-wide user data: constant banks not owned by one kernel, initialized and zeroed
    __device__ data (contents, or size for NOBITS), and the relocations that bind them (which
    __device__ global or syscall a constant slot points at)."""
    h = hashlib.sha256()
    for c in cubins:
        elf = Elf(c)
        for s in sorted(elf.sections, key=lambda s: s["name"]):
            if OBJECT_DATA.match(s["name"]):
                h.update(s["name"].encode() + b"\0" + (s["data"] if s["type"] != SHT_NOBITS else str(s["size"]).encode()))
            elif elf.is_reloc(s) and OBJECT_DATA.match(elf.target(s)):
                h.update(json.dumps([s["name"], elf.relocs(s)]).encode())
    return h.hexdigest()


STT_SECTION, STT_FILE = 3, 4
SHF_STRINGS = 0x20


def host_reloc_target(elf, sym, addend):
    """A host relocation's target by name: a named symbol is its (normalized) name; a section
    symbol (local data and static functions the assembler addresses by section) keeps the
    section and the addend, so moving that data moves the digest (conservative)."""
    if sym >= len(elf.symbols):
        return f"#{sym}"
    shndx, _value, _size = elf.symdefs[sym]
    name = normalize_symbol(elf.symbols[sym])
    if (elf.syminfo[sym] & 0xF) == STT_SECTION:
        return f"section:{name}{addend:+d}" if addend is not None else f"section:{name}"
    return name if addend is None else f"{name}{addend:+d}"


def host_digest(obj):
    """The host part of a `-c` object (the launchers), without the embedded device fatbin.
    Every defined symbol counts by name, binding, type, visibility, section and size plus a digest
    of its own body (its byte range, relocations inside it resolved to names), so moving a name to
    another body is a change and a layout shift alone is not. Bytes no symbol covers (string
    literals, init arrays, unwind tables) count with their relocations. None for a device-only
    object."""
    try:
        elf = Elf(obj)
    except ValueError:
        return None
    if ".nv_fatbin" not in elf.by_name:
        return None
    relocs_in = collections.defaultdict(list)  # target section index -> [(off, type, target)]
    for s in elf.sections:
        if elf.is_reloc(s) and s["info"] < len(elf.sections):
            for i in range(s["size"] // s["entsize"]):
                if s["entsize"] == 24:
                    off, info, addend = struct.unpack_from("<QQq", elf.b, s["off"] + i * 24)
                else:
                    (off, info), addend = struct.unpack_from("<QQ", elf.b, s["off"] + i * 16), None
                relocs_in[s["info"]].append((off, info & 0xFFFFFFFF, host_reloc_target(elf, info >> 32, addend)))
    alloc = {i for i, s in enumerate(elf.sections)
             if s["flags"] & SHF_ALLOC and s["name"] != ".nv_fatbin" and not elf.is_reloc(s)}
    covered = collections.defaultdict(list)
    records = []
    for k, name in enumerate(elf.symbols):
        shndx, value, size = elf.symdefs[k]
        typ, bind, vis = elf.syminfo[k] & 0xF, elf.syminfo[k] >> 4, elf.symother[k] & 3
        if shndx not in alloc or typ in (STT_SECTION, STT_FILE):
            continue
        sec = elf.sections[shndx]
        if sec["type"] == SHT_NOBITS:
            body = ""
        else:
            body = hashlib.sha256(normalize_bytes(sec["data"][value:value + size])).hexdigest()
        inside = sorted((off - value, t, tgt) for off, t, tgt in relocs_in[shndx] if value <= off < value + size)
        records.append((normalize_symbol(name), bind, typ, vis, normalize_symbol(sec["name"]), size, body, inside))
        if size:
            covered[shndx].append((value, value + size))
    residual = []
    for i in sorted(alloc):
        sec = elf.sections[i]
        spans, pos, pieces = sorted(covered[i]), 0, []
        for a, b in spans + [(sec["size"], sec["size"])]:
            if a > pos:
                piece = sec["data"][pos:a] if sec["type"] != SHT_NOBITS else b""
                rel = sorted((off - pos, t, tgt) for off, t, tgt in relocs_in[i] if pos <= off < a)
                pieces.append((a - pos, hashlib.sha256(normalize_bytes(piece)).hexdigest(), rel))
            pos = max(pos, b)
        residual.append((normalize_symbol(sec["name"]), pieces))
    h = hashlib.sha256(json.dumps([sorted(records), residual], default=str).encode())
    return h.hexdigest()


def extract_cubins(obj, into):
    """cuobjdump -xelf all OBJ into a directory; a plain cubin is copied."""
    into.mkdir(parents=True, exist_ok=True)
    data = Path(obj).read_bytes()[:4]
    if data == b"\x7fELF" and str(obj).endswith(".cubin"):
        shutil.copy(obj, into / Path(obj).name)
    else:
        subprocess.run(["cuobjdump", "-xelf", "all", str(Path(obj).resolve())], cwd=into,
                       check=True, capture_output=True)
    return sorted(into.glob("*.cubin"))


def sass_of(cubin):
    return subprocess.run(["cuobjdump", "-sass", str(cubin)], check=True, capture_output=True,
                          text=True, errors="replace").stdout


def fingerprint(obj):
    """{kernel: {hash, attrs, fp, profile, seq, external, n, unfused_pairs, lines}}, plus the
    object's data digest. `hash` is the stream, `attrs` the kernel attributes, `fp` both."""
    tmp = Path(tempfile.mkdtemp(prefix="numdiff-", dir=WORK if WORK.exists() else None))
    try:
        cubins = extract_cubins(obj, tmp)
        if not cubins:
            raise ValueError(f"{obj}: no cubin inside")
        kernels = {}
        for c in cubins:
            elf = Elf(c)
            for name, insns in parse_sass(sass_of(c)).items():
                lines, external = canonical(insns)
                lines = [normalize_symbol(l) for l in lines]
                stream = hashlib.sha256("\n".join(lines).encode()).hexdigest()[:20]
                attrs = hashlib.sha256(kernel_attributes(elf, name).encode()).hexdigest()[:20]
                key = normalize_symbol(name)
                if key in kernels:
                    raise ValueError(f"{obj}: two kernels normalize to {key}")
                kernels[key] = {
                    "symbol": name,
                    "hash": stream,
                    "attrs": attrs,
                    "fp": hashlib.sha256((stream + attrs).encode()).hexdigest()[:20],
                    "profile": dict(fp_profile(insns)),
                    "seq": fp_sequence(insns),
                    "external": external,
                    "n": len(insns),
                    "unfused_pairs": unfused_pairs(lines),
                    "lines": lines,
                }
        return kernels, data_digest(cubins)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ------------------------------------------------------------------------------ classification

def base_name(symbol):
    """The function name inside an Itanium-mangled symbol (`_Z15mul_mat_f8f4v3tILi...` ->
    `mul_mat_f8f4v3t`); an extern "C" name is returned as is."""
    m = re.match(r"_Z(\d+)", symbol)
    if m:
        return symbol[m.end():m.end() + int(m.group(1))]
    return symbol


def find_kernels(name, kernels):
    """Kernels a name means: the exact symbol, else every instantiation of that function name, else
    the one symbol containing it. Returns [] when absent or ambiguous."""
    if name in kernels:
        return [name]
    family = sorted(k for k in kernels if base_name(k) == name)
    if family:
        return family
    hits = sorted(k for k in kernels if name in k)
    return hits if len(hits) == 1 else []


def classify(a, b):
    """Float classes that moved between two kernel fingerprints."""
    classes = set()
    delta = collections.Counter(b["profile"])
    delta.subtract(a["profile"])
    fam = collections.defaultdict(int)
    for op, d in delta.items():
        if d:
            fam[fp_family(op)] += abs(d)
    base_d = collections.Counter()
    for op, d in delta.items():
        base, k = scalar_base(op)
        base_d[base] += d * k
    f, m, ad = base_d["FFMA"], base_d["FMUL"], base_d["FADD"]
    # contraction turns FMUL (+FADD) into FFMA: FFMA moves against FMUL, or against FADD when
    # FMUL is flat
    if f and ((m and (f > 0) != (m > 0)) or (not m and ad and (f > 0) != (ad > 0))):
        classes.add("fma-contraction")
    elif fam.get("fma-contraction"):
        classes.add("fp-ops")
    for fm in ("approximation", "precision", "tensor-core", "reduction", "fp-ops"):
        if fam.get(fm):
            classes.add(fm)
    if not classes and a["hash"] != b["hash"]:
        classes.add("same-fp-ops")
    return sorted(classes), {k: v for k, v in delta.items() if v}


def first_fp_difference(a, b, context=2):
    sm = difflib.SequenceMatcher(a=a["seq"], b=b["seq"], autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag != "equal":
            return {"fp_index": i1, "base": a["seq"][max(0, i1 - context):i2 + context],
                    "head": b["seq"][max(0, j1 - context):j2 + context]}
    return None


def compare_objects(base, head, flag_change=False):
    (ka, da), (kb, db) = base, head
    rows = []
    # a symbol rename (template parameters, linkage, a merged template) with the same code
    renamed = {}
    if da == db:
        added = collections.defaultdict(list)
        for name in sorted(set(kb) - set(ka)):
            if not kb[name]["external"]:
                added[kb[name]["fp"]].append(name)
        for name in sorted(set(ka) - set(kb)):
            if not ka[name]["external"] and added.get(ka[name]["fp"]):
                renamed[name] = added[ka[name]["fp"]].pop(0)
    taken = set(renamed.values())
    for name in sorted(set(ka) | set(kb)):
        if name in renamed:
            rows.append({"kernel": name, "verdict": "RENAMED", "to": renamed[name], "instructions": ka[name]["n"]})
            continue
        if name in taken:
            continue
        if name not in ka:
            rows.append({"kernel": name, "verdict": "ADDED"})
            continue
        if name not in kb:
            rows.append({"kernel": name, "verdict": "REMOVED"})
            continue
        a, b = ka[name], kb[name]
        classes, delta = classify(a, b)
        # layout-bound attributes (exit offsets, capmerc text, register count) follow any stream
        # change; the class says something only when the stream is the same
        if a["attrs"] != b["attrs"] and a["hash"] == b["hash"]:
            classes.append("attributes")
        if da != db:
            classes.append("data")
        if a["external"] or b["external"]:
            classes.append("external-call")
        if flag_change:
            classes.append("build-flag")
        same_stream = a["hash"] == b["hash"]
        if a["fp"] == b["fp"] and not ({"data", "external-call"} & set(classes)):
            rows.append({"kernel": name, "verdict": "IDENTICAL", "instructions": b["n"]})
        else:
            if same_stream:
                classes = [c for c in classes if c != "same-fp-ops"]
            row = {"kernel": name, "verdict": "CHANGED", "classes": sorted(set(classes)),
                   "fp_delta": delta, "first_fp_difference": first_fp_difference(a, b),
                   "instructions": [a["n"], b["n"]]}
            if a["unfused_pairs"] != b["unfused_pairs"]:
                row["unfused_pairs"] = [a["unfused_pairs"], b["unfused_pairs"]]
            rows.append(row)
    return rows


NUMERIC_FLAG = re.compile(r"^-{1,2}(fmad|ftz|prec-div|prec-sqrt|use_fast_math|O\d|Xptxas|ptxas-options|"
                          r"extra-device-vectorization|G$|dopt|D)")


def numeric_flags(flags):
    """The nvcc flags that change float code (contraction, denormals, division, approximation,
    optimization, ptxas options) and the -D defines that switch numeric paths, with the value
    that follows a bare flag."""
    expanded = []
    for f in flags:
        # nvcc's own environment flags are flags too
        if f.startswith(("env:NVCC_APPEND_FLAGS=", "env:NVCC_PREPEND_FLAGS=")):
            expanded += f.split("=", 1)[1].split()
        else:
            expanded.append(f)
    flags = expanded
    out, i = [], 0
    while i < len(flags):
        f = flags[i]
        if NUMERIC_FLAG.match(f):
            if f in ("-Xptxas", "--ptxas-options", "-D") and i + 1 < len(flags):
                out.append(f"{f} {flags[i + 1]}")
                i += 1
            else:
                out.append(f)
        i += 1
    return out


def twin_compare(ka, kb, flags_a=None, flags_b=None, same_data=True):
    """A bitwise-twin claim between two kernels (one build, possibly two files).
    IDENTICAL-STREAM: the same stream and kernel attributes and the same object-wide data (one
    object, or two whose data digests match): equal bits for equal inputs and launch parameters.
    Otherwise CHANGED with the classes, the numeric flag difference and a contraction census per
    side (FFMA count and FMUL->FADD pairs ptxas left unfused). The census explains; it does not
    decide a twin of different size."""
    classes, delta = classify(ka, kb)
    if ka["attrs"] != kb["attrs"] and ka["hash"] == kb["hash"]:
        classes = sorted(set(classes) | {"attributes"})
    if not same_data:
        classes = sorted(set(classes) | {"data"})
    ffma = [sum(v * scalar_base(o)[1] for o, v in k["profile"].items() if scalar_base(o)[0] == "FFMA")
            for k in (ka, kb)]
    row = {"verdict": "CHANGED", "classes": classes, "fp_delta": delta,
           "first_fp_difference": first_fp_difference(ka, kb), "instructions": [ka["n"], kb["n"]],
           "census": {"ffma": ffma, "unfused_pairs": [ka["unfused_pairs"], kb["unfused_pairs"]]}}
    if flags_a is not None and flags_b is not None:
        na, nb = numeric_flags(flags_a), numeric_flags(flags_b)
        if na != nb:
            row["classes"] = sorted(set(row["classes"]) | {"build-flag"})
            row["flags"] = {"a": na, "b": nb}
    # the SASS is what runs: equal code is equal whatever flags produced it
    if ka["fp"] == kb["fp"] and same_data and not (ka["external"] or kb["external"]):
        row = {"verdict": "IDENTICAL-STREAM", "instructions": ka["n"]}
    return row


# ---------------------------------------------------------------------------- revisions, builds

def git(repo, *args, text=True):
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True,
                          text=text).stdout


_NVCC_ID = None


def nvcc_identity():
    """The toolkit the objects come from: REAL_NVCC's real path and `nvcc --version`."""
    global _NVCC_ID
    if _NVCC_ID is None:
        path = os.path.realpath(REAL_NVCC)
        version = subprocess.run([path, "--version"], check=True, capture_output=True, text=True).stdout
        tail = [l for l in version.splitlines() if l.strip()][-2:]
        _NVCC_ID = {"path": path, "version": " | ".join(tail),
                    "id": hashlib.sha256((path + "\n" + version).encode()).hexdigest()[:10]}
    return _NVCC_ID


def clean_env(**extra):
    """The environment without MEMRA_* and NVCC_* (build seams and nvcc's own append/prepend
    flags), plus what numdiff sets."""
    env = {k: v for k, v in os.environ.items() if not k.startswith(("MEMRA_", "NVCC_"))}
    env.update(extra)
    return env


def archive(repo, rev, into):
    """Export a revision's build inputs without git metadata."""
    if into.exists():
        return into
    tmp = into.with_suffix(".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    paths = [p for p in ("crates", "Cargo.toml", "Cargo.lock", "rust-toolchain.toml", "rustfmt.toml", ".cargo")
             if git(repo, "ls-tree", "--name-only", rev, p).strip()]
    tar = subprocess.run(["git", "-C", str(repo), "archive", rev, *paths], check=True, capture_output=True).stdout
    subprocess.run(["tar", "-x", "-C", str(tmp)], input=tar, check=True)
    tmp.rename(into)
    return into


def capture(tree, arch, key):
    """Record build.rs's nvcc commands for a tree (cached per build inputs, arch and nvcc).
    A failed build raises: a partial command list would hide the units after the failure."""
    db_path = WORK / "db" / f"{key}-{arch}.json"
    if db_path.exists():
        return json.loads(db_path.read_text())
    db_path.parent.mkdir(parents=True, exist_ok=True)
    log = WORK / "db" / f"{key}-{arch}.jsonl"
    log.unlink(missing_ok=True)
    wrapper = WORK / "nvcc-capture"
    wrapper.write_text("#!/usr/bin/env bash\nset -euo pipefail\n"
                       "python3 -I -c 'import json,os,sys; print(json.dumps({\"cwd\": os.getcwd(), \"argv\": sys.argv[1:], "
                       "\"env\": {k: v for k, v in os.environ.items() if k.startswith(\"NVCC_\") or k in "
                       "(\"CPATH\", \"C_INCLUDE_PATH\", \"CPLUS_INCLUDE_PATH\")}}))' \"$@\" >> \"$NUMDIFF_NVCC_LOG\"\n"
                       "exec \"$NUMDIFF_REAL_NVCC\" \"$@\"\n")
    wrapper.chmod(0o755)
    # the build only has to run build.rs: its target dir is scratch, dropped once the log is read
    target = WORK / "target" / f"{key}-{arch}"
    env = clean_env(MEMRA_NVCC=str(wrapper), NUMDIFF_NVCC_LOG=str(log), NUMDIFF_REAL_NVCC=REAL_NVCC,
                    MEMRA_CUDA_ARCH=arch, CARGO_TARGET_DIR=str(target), CARGO_PROFILE_DEV_DEBUG="0")
    note = None
    try:
        run = subprocess.run(["flock", str(BUILD_LOCK), "nice", "-n", "15", "cargo", "check", "-q", "-p", "memra-engine"],
                             cwd=tree, env=env, capture_output=True, text=True)
        # cargo writes a build script's `output` only when the script succeeded: then every nvcc
        # call build.rs makes is in the log, even if a crate fails to compile afterwards
        script_done = any((target / "debug" / "build").glob("memra-engine-*/output")) and \
            "failed to run custom build command for `memra-engine" not in run.stderr
        if not log.exists() or not script_done:
            raise RuntimeError(f"capture build failed in {tree} (exit {run.returncode}):\n{run.stderr[-3000:]}")
        if run.returncode != 0:
            note = f"build.rs completed; cargo check then failed (exit {run.returncode}): " + \
                " / ".join(l for l in run.stderr.splitlines() if l.startswith("error"))[:500]
        rows = [json.loads(l) for l in log.read_text().splitlines() if l.strip()]
    finally:
        shutil.rmtree(target, ignore_errors=True)
    db = []
    for r in rows:
        argv = r["argv"]
        srcs = [a for a in argv if a.endswith(".cu")]
        if len(srcs) != 1 or "-o" not in argv:
            continue
        # the environment the build gave nvcc ([env] in .cargo/config.toml, build.rs's Command::env)
        db.append({"src": srcs[0], "cwd_rel": os.path.relpath(r["cwd"], tree), "argv": argv,
                   "out": argv[argv.index("-o") + 1], "tree": str(tree), "env": r.get("env", {})})
        if note:
            db[-1]["capture_note"] = note
    db_path.write_text(json.dumps(db, indent=1))
    return db


def flags_of(entry):
    """The flags of a compile entry, paths and the output name removed (for a flags diff), plus
    the nvcc environment it ran with as `env:NAME=value` (tree paths made relative)."""
    argv = list(entry["argv"])
    i = argv.index("-o")
    del argv[i:i + 2]
    flags = [a for a in argv if not a.endswith(".cu") and not a.startswith(entry["tree"]) and "/out/" not in a]
    return flags + [f"env:{k}={v.replace(entry['tree'], '<tree>')}" for k, v in sorted(entry.get("env", {}).items())]


def unit_name(entry):
    return Path(entry["out"]).name


def is_host_unit(entry):
    """A `-c` unit: a static-lib object whose host code (the launchers) ships next to its kernels."""
    return "-c" in entry["argv"]


def object_path(entry, outdir):
    """Where a compile entry's object is cached: keyed by source, output name, flags and nvcc."""
    key = hashlib.sha256(json.dumps([entry["src"], unit_name(entry), flags_of(entry),
                                     nvcc_identity()["id"]]).encode()).hexdigest()[:10]
    return outdir / key / unit_name(entry)


def compile_one(entry, tree, outdir):
    """Run one recorded nvcc command against another tree; returns the output object. The object
    is cached under outdir by source, flags and nvcc (a revision's tree and these fix its
    content)."""
    old_tree = entry["tree"]
    argv = [a.replace(old_tree, str(tree)) for a in entry["argv"]]
    i = argv.index("-o")
    out = object_path(entry, outdir)
    if out.exists():
        return out
    # build into a scratch directory and move the finished object into place
    out.parent.parent.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix="compile-", dir=out.parent.parent))
    argv[i + 1] = str(scratch / out.name)
    # keep build.rs's OUT_DIR includes (generated headers) reachable from the recorded build
    cwd = Path(tree) / entry["cwd_rel"]
    try:
        env = clean_env(**{k: v.replace(old_tree, str(tree)) for k, v in entry.get("env", {}).items()})
        run = subprocess.run(["flock", str(BUILD_LOCK), "nice", "-n", "15", REAL_NVCC, *argv], cwd=cwd,
                             capture_output=True, text=True, env=env)
        if run.returncode != 0:
            raise RuntimeError(f"nvcc failed for {entry['src']} in {tree}:\n{run.stderr[-2000:]}")
        out.parent.mkdir(parents=True, exist_ok=True)
        (scratch / out.name).rename(out)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    return out


# what decides the nvcc commands besides the sources: build.rs and the arch table it includes,
# the manifests (features, build-dependencies) and cargo's [env] seams
BUILD_INPUTS = ("crates/memra-engine/build.rs", "crates/memra-engine/src/cuda_arch.rs",
                "crates/memra-engine/Cargo.toml", "Cargo.toml", ".cargo/config.toml", ".cargo/config")


def build_rev(repo, rev, arch):
    """A revision's tree and its recorded nvcc commands by (source, output name): a source can
    be built more than once (flash_attn.cu: the default and the kf8vf8 variant)."""
    tree = archive(repo, rev, WORK / "trees" / rev[:12])
    blobs = []
    for p in BUILD_INPUTS:
        try:
            blobs.append(git(repo, "rev-parse", f"{rev}:{p}").strip()[:12])
        except subprocess.CalledProcessError:
            blobs.append("none")
    key = hashlib.sha256("|".join(blobs + [nvcc_identity()["id"]]).encode()).hexdigest()[:12]
    units = {}
    for e in capture(tree, arch, key):
        units[(e["src"], unit_name(e))] = e
    return tree, units


HOST_SKIP = ("/tests/", "/benches/", "/examples/", "/docs/")
KERNEL_DIR = "crates/memra-engine/cu/"


def host_changes(repo, base, head):
    """Changed files outside the compared kernels that can decide which kernel runs and how it is
    launched: Rust sources of every crate, manifests, the lock file, build.rs and cuda_arch.rs
    (cfgs, rustc-env, links), and CUDA outside crates/memra-engine/cu (built elsewhere or not at
    all; numdiff does not compare it)."""
    names = git(repo, "diff", "--name-only", base, head, "--", "crates", "Cargo.toml", "Cargo.lock",
                "rust-toolchain.toml", ".cargo").split()
    out = []
    for n in names:
        if any(s in n for s in HOST_SKIP) or n.endswith(".md"):
            continue
        if n.startswith(KERNEL_DIR) or is_kernel_header(n):
            continue  # kernel sources and headers: compared as kernels (headers recompile every unit)
        out.append(n)
    return out


def launch_mentions(repo, base, head, files, kernels):
    """Compiled kernel names that appear on changed lines of the host files (a launch-site hint;
    a launch change made through a shared helper names no kernel)."""
    names = {base_name(k) for k in kernels}
    hits = collections.defaultdict(set)
    for f in files:
        if not f.endswith((".rs", ".cu")):
            continue
        diff = git(repo, "diff", "-U0", base, head, "--", f)
        for line in diff.splitlines():
            if line[:1] in "+-" and not line.startswith(("+++", "---")):
                for word in re.findall(r"[A-Za-z_][A-Za-z0-9_]{5,}", line):
                    if word in names:
                        hits[f].add(word)
    return {f: sorted(v) for f, v in hits.items()}


def included_by(tree, src):
    """True when a source or header in the tree's kernel dir #includes `src` (cu/x.cu)."""
    name = Path(src).name
    rx = re.compile(r'^\s*#\s*include\s*[<"]([^>"]*)[>"]', re.M)
    for f in (Path(tree) / KERNEL_DIR).rglob("*"):
        if f.suffix in (".cu", ".cuh", ".h") and f.is_file():
            for inc in rx.findall(f.read_text(errors="replace")):
                if Path(inc).name == name:
                    return True
    return False


HEADER_EXT = (".cuh", ".h", ".hpp", ".hh", ".hxx", ".inc", ".inl", ".cuinc", ".ipp", ".tcc", ".def")


def is_kernel_header(name):
    """A file that can reach a kernel through #include: any non-.cu file under cu/, and a
    C/C++/CUDA header anywhere else in crates/."""
    if name.startswith(KERNEL_DIR):
        return not name.endswith(".cu") and not name.endswith(".md")
    return name.startswith("crates/") and name.endswith(HEADER_EXT)


def touched_cu(repo, base, head):
    """(touched .cu units under cu/, every unit needs comparing, kernel-capable inputs changed)."""
    names = git(repo, "diff", "--name-only", base, head, "--", "crates", *BUILD_INPUTS).split()
    cu = {n.split("crates/memra-engine/", 1)[1] for n in names if n.startswith(KERNEL_DIR) and n.endswith(".cu")}
    every = any(is_kernel_header(n) or n in BUILD_INPUTS for n in names)
    return cu, every, bool(cu) or every


def rev_command(args):
    repo = Path(args.repo).expanduser()
    base = git(repo, "rev-parse", args.base).strip()
    head = git(repo, "rev-parse", args.head).strip()
    WORK.mkdir(parents=True, exist_ok=True)
    trees, units = {}, {}
    for r in (base, head):
        trees[r], units[r] = build_rev(repo, r, args.arch)
    touched, every, kernel_inputs = touched_cu(repo, base, head)
    all_srcs = {s for r in units for s, _ in units[r]}
    # a touched .cu that another unit #includes acts as a header for it, whether or not it is a
    # unit itself; one no command compiles and nothing includes is not covered for this arch
    unbuilt = sorted(touched - all_srcs)
    included = sorted(s for s in touched if any(included_by(trees[r], s) for r in trees))
    if included:
        every = True
    wanted = all_srcs if every else touched
    files = set(args.files) if args.files else wanted
    report = {"base": base, "head": head, "arch": args.arch, "nvcc": nvcc_identity(), "files": {},
              "summary": collections.Counter(), "partial": bool(args.files) and not wanted <= set(args.files)}
    seen, uncovered, one_sided, host_units = set(), [], [], []
    report["included_sources"] = included
    for src in sorted(files | set(unbuilt)):
        variants = sorted({u for r in units for s, u in units[r] if s == src})
        if not variants:
            if src in included:  # not a unit; its includers are compared
                report["files"][src] = {"compiled": False, "note": "not a unit; included by other units (all compared)"}
                continue
            report["files"][src] = {"compiled": False, "note": f"not compiled for sm_{args.arch} on either side"}
            uncovered.append(src)
            continue
        for unit in variants:
            label = src if len(variants) == 1 else f"{src} [{unit}]"
            eb, eh = units[base].get((src, unit)), units[head].get((src, unit))
            entry = {"compiled": bool(eb and eh), "unit": unit}
            if not eb or not eh:
                entry["note"] = "compiled on " + ("head" if eh else "base") + " only"
                one_sided.append(label)
                report["files"][label] = entry
                continue
            flag_change = flags_of(eb) != flags_of(eh)
            if flag_change:
                entry["flags"] = {"base": flags_of(eb), "head": flags_of(eh)}
            objs = {r: compile_one(e, trees[r], WORK / "objs" / r[:12] / args.arch) for r, e in ((base, eb), (head, eh))}
            fb, fh = fingerprint(objs[base]), fingerprint(objs[head])
            seen |= set(fb[0]) | set(fh[0])
            rows = compare_objects(fb, fh, flag_change)
            entry["kernels"] = rows
            for row in rows:
                report["summary"][row["verdict"]] += 1
            if is_host_unit(eb) or is_host_unit(eh):
                hb, hh = host_digest(objs[base]), host_digest(objs[head])
                if hb is None or hh is None or hb != hh:
                    entry["host_code_changed"] = True
                    host_units.append(f"{src} (host code of {unit})")
            report["files"][label] = entry
    report["summary"] = dict(report["summary"])
    notes = sorted({e["capture_note"] for r in units for e in units[r].values() if e.get("capture_note")})
    if notes:
        report["capture_notes"] = notes
    host = host_changes(repo, base, head) + host_units
    report["host_changed"] = host
    report["uncovered"] = uncovered
    report["one_sided"] = one_sided
    if host:
        report["launch_mentions"] = launch_mentions(
            repo, base, head, [h for h in host if " (" not in h]
            + sorted({"crates/memra-engine/" + h.split(" (")[0] for h in host_units}), seen)
    report["verdict"] = change_verdict(report["summary"], one_sided, uncovered, host, kernel_inputs)
    return report


def change_verdict(summary, one_sided, uncovered, host, kernel_inputs=False):
    """The change-level verdict, the first that holds (see the module doc). Kernel-capable
    inputs that changed while nothing was compared is UNCOVERED, never an identity claim."""
    if summary.get("CHANGED") or summary.get("ADDED") or summary.get("REMOVED") or one_sided:
        return "CHANGED"
    if uncovered or (kernel_inputs and not summary):
        return "UNCOVERED"
    if host:
        return "KERNELS-IDENTICAL"
    return "IDENTICAL"


def parse_spec(spec):
    src, sep, kernel = spec.rpartition(":")
    if not sep:
        raise SystemExit(f"{spec!r}: expected FILE:KERNEL (FILE as build.rs names it, e.g. cu/kernels.cu)")
    return src, kernel


def twins_command(repo, rev, arch, spec_a, spec_b):
    """A twin claim between two kernels of one revision. A name that means several template
    instantiations compares every pair; a source built in several variants compares variant
    with variant (same source) or every variant pair (two sources)."""
    repo = Path(repo).expanduser()
    rev = git(repo, "rev-parse", rev).strip()
    tree, units = build_rev(repo, rev, arch)
    sides = []
    for spec in (spec_a, spec_b):
        src, name = parse_spec(spec)
        variants = sorted(u for s, u in units if s == src)
        if not variants:
            return {"rev": rev, "arch": arch, "rows": [], "note": f"{src} is not compiled for sm_{arch}"}
        side = []
        for unit in variants:
            e = units[(src, unit)]
            kernels, digest = fingerprint(compile_one(e, tree, WORK / "objs" / rev[:12] / arch))
            hits = find_kernels(name, kernels)
            if hits:
                side.append((src, unit, flags_of(e), digest, {k: kernels[k] for k in hits}))
        if not side:
            return {"rev": rev, "arch": arch, "rows": [], "note": f"{name!r} in {src}: no single match"}
        sides.append(side)
    rows = []
    for sa, ua, fa, da, ka in sides[0]:
        for sb, ub, fb, db, kb in sides[1]:
            if sa == sb and ua != ub:
                continue
            for na in sorted(ka):
                for nb in sorted(kb):
                    if na == nb and sa == sb:
                        continue
                    same_data = (sa, ua) == (sb, ub) or da == db
                    row = twin_compare(ka[na], kb[nb], fa, fb, same_data)
                    label = f"{na} vs {nb}"
                    if ua != ub or len(sides[0]) > 1:
                        label += f" [{ua} vs {ub}]"
                    rows.append({"kernel": label, **row})
    return {"rev": rev, "arch": arch, "nvcc": nvcc_identity(), "a": spec_a, "b": spec_b, "rows": rows}


# ------------------------------------------------------------------------------------- CLI

def short(report_rows):
    for r in report_rows:
        extra = ""
        if r["verdict"] == "RENAMED":
            extra = f" -> {r['to']}"
        if r["verdict"] == "CHANGED":
            extra = " " + ",".join(r["classes"])
            if r.get("fp_delta"):
                extra += " " + " ".join(f"{k}{v:+d}" for k, v in sorted(r["fp_delta"].items()))
            if r.get("census"):
                c = r["census"]
                extra += f"  [FFMA {c['ffma'][0]}/{c['ffma'][1]}, unfused FMUL->FADD {c['unfused_pairs'][0]}/{c['unfused_pairs'][1]}]"
            elif r.get("unfused_pairs"):
                extra += f"  [unfused FMUL->FADD {r['unfused_pairs'][0]} -> {r['unfused_pairs'][1]}]"
            if r.get("flags"):
                extra += f"  [flags a={' '.join(r['flags']['a']) or '-'} b={' '.join(r['flags']['b']) or '-'}]"
        print(f"  {r['verdict']:16s} {r['kernel']}{extra}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("rev")
    p.add_argument("base")
    p.add_argument("head")
    p.add_argument("--repo", default="~/projects/memra")
    p.add_argument("--arch", default="120a")
    p.add_argument("--files", nargs="*")
    p.add_argument("--out")
    p = sub.add_parser("twins")
    p.add_argument("rev")
    p.add_argument("a", help="FILE:KERNEL, FILE as build.rs names it (cu/kernels.cu)")
    p.add_argument("b")
    p.add_argument("--repo", default="~/projects/memra")
    p.add_argument("--arch", default="120a")
    p.add_argument("--out")
    p = sub.add_parser("objs")
    p.add_argument("base")
    p.add_argument("head")
    p = sub.add_parser("twin")
    p.add_argument("obj")
    p.add_argument("a")
    p.add_argument("b")
    p = sub.add_parser("sass")
    p.add_argument("obj")
    args = ap.parse_args()
    if args.cmd == "rev":
        report = rev_command(args)
        if args.out:
            Path(args.out).write_text(json.dumps(report, indent=1))
        print(f"numdiff {report['base'][:10]}..{report['head'][:10]} sm_{report['arch']}: {report['verdict']}"
              + (" PARTIAL (--files narrower than the change)" if report["partial"] else "")
              + f" {report['summary']}  [nvcc {report['nvcc']['version']}]")
        for src, entry in report["files"].items():
            print(f"{src}" + (f"  [{entry['note']}]" if entry.get("note") else "")
                  + ("  [flags changed]" if entry.get("flags") else "")
                  + ("  [host code changed]" if entry.get("host_code_changed") else ""))
            short(entry.get("kernels", []))
        if report["host_changed"]:
            print(f"host code changed (launch parameters and kernel choice not covered): {' '.join(report['host_changed'])}")
            for f, names in report.get("launch_mentions", {}).items():
                print(f"  {f} changed lines name: {' '.join(names)}")
        return 0 if report["verdict"] == "IDENTICAL" and not report["partial"] else 1
    if args.cmd == "twins":
        report = twins_command(args.repo, args.rev, args.arch, args.a, args.b)
        if args.out:
            Path(args.out).write_text(json.dumps(report, indent=1))
        print(f"numdiff twins {report['rev'][:10]} sm_{report['arch']}" + (f"  [{report['note']}]" if report.get("note") else ""))
        short(report["rows"])
        return 0 if report["rows"] and all(r["verdict"] == "IDENTICAL-STREAM" for r in report["rows"]) else 1
    if args.cmd == "objs":
        short(compare_objects(fingerprint(args.base), fingerprint(args.head)))
        return 0
    if args.cmd == "twin":
        kernels, _ = fingerprint(args.obj)
        def find(n):
            hits = find_kernels(n, kernels)
            if len(hits) != 1:
                raise SystemExit(f"{n!r} matches {len(hits) or 'no single'} kernel: {hits[:6] or sorted(k for k in kernels if n in k)[:6]}")
            return hits[0]
        a, b = find(args.a), find(args.b)
        row = twin_compare(kernels[a], kernels[b])
        short([{"kernel": f"{a} vs {b}", **row}])
        if row.get("first_fp_difference"):
            d = row["first_fp_difference"]
            print(f"  first float difference at op {d['fp_index']}: {d['base']} -> {d['head']}")
        return 0 if row["verdict"] == "IDENTICAL-STREAM" else 1
    if args.cmd == "sass":
        kernels, digest = fingerprint(args.obj)
        print(f"data {digest[:16]}")
        for k, v in sorted(kernels.items()):
            prof = " ".join(f"{o}:{n}" for o, n in sorted(v["profile"].items()))
            print(f"{v['fp']} {v['n']:6d} {k}  {prof}")
        return 0


if __name__ == "__main__":
    sys.exit(main())
