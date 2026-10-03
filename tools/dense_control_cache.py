#!/usr/bin/env python3
"""Conservative exact-input reuse for six CPU-control native executables.

Preprocessor output and dependency files bind compiler-reachable repository bytes.
Conservative toolchain inventories include device tools, headers and link inputs.
The original compiler recipe/environment stays unchanged. A restore never substitutes
for --check-controls execution.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import tempfile
import struct
import re
import shlex

NAMES = tuple("dsv4-dense-" + name for name in (
    "exact-tail-gate", "tc-gate", "tc-gate-r4", "tc-gate-r7-driver",
    "tc-gate-r8-driver", "tc-gate-r9-driver"))
FLAGS = ("-t", "2", "-std=c++17", "-O3", "-fmad=false",
         "-Xcompiler=-ffp-contract=off", "-arch=sm_120a")
SCHEMA = "memra-dense-control-cache-v1"
OVERRIDES = ("NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS", "NVCC_CCBIN", "CUDAHOSTCXX",
             "CC", "CXX", "CPATH", "C_INCLUDE_PATH", "CPLUS_INCLUDE_PATH",
             "COMPILER_PATH", "GCC_EXEC_PREFIX", "LIBRARY_PATH", "LD_PRELOAD",
             "LD_LIBRARY_PATH", "CFLAGS", "CXXFLAGS", "LDFLAGS", "CPPFLAGS",
             "CUDAFE_FLAGS", "PTXAS_FLAGS", "CICC_PATH", "NVVMIR_LIBRARY_DIR",
             "INCLUDES", "SYSTEM_INCLUDES", "LIBRARIES", "TOP",
             "DEPENDENCIES_OUTPUT", "SUNPRO_DEPENDENCIES", "GCC_COMPARE_DEBUG",
             "GCC_COMPARE_DEBUG_AUXBASE", "GNUTARGET", "LDEMULATION", "SOURCE_DATE_EPOCH")

# Stock nvcc/GNU compiler inputs. CI transport metadata is not a compiler input;
# process environment is passed through unchanged, never sanitized by this tool.
COMPILER_ENV = ("PATH", "HOME", "TMPDIR", "TMP", "TEMP", "LANG", "LANGUAGE", "LC_ALL",
                "LC_CTYPE", "LC_MESSAGES", "LC_COLLATE", "LC_NUMERIC", "LC_TIME",
                "SOURCE_DATE_EPOCH", "LD_LIBRARY_PATH", "GCC_COMPARE_DEBUG",
                "GCC_COMPARE_DEBUG_AUXBASE", "GNUTARGET", "LDEMULATION")
PROFILE = """
TOP = $(_HERE_)/..
CICC_PATH = $(TOP)/nvvm/bin
NVVMIR_LIBRARY_DIR = $(TOP)/nvvm/libdevice
LD_LIBRARY_PATH += $(TOP)/lib:
PATH += $(CICC_PATH):$(_HERE_):
INCLUDES += "-I$(TOP)/$(_TARGET_DIR_)/include" $(_SPACE_)
SYSTEM_INCLUDES += "-isystem" "$(TOP)/$(_TARGET_DIR_)/include/cccl" $(_SPACE_)
LIBRARIES =+ $(_SPACE_) "-L$(TOP)/$(_TARGET_DIR_)/lib$(_TARGET_SIZE_)/stubs" "-L$(TOP)/$(_TARGET_DIR_)/lib$(_TARGET_SIZE_)"
CUDAFE_FLAGS +=
PTXAS_FLAGS +=
"""


def admit_profile(profile):
    normalize = lambda text: [" ".join(line.split()) for line in text.splitlines() if line.strip()]
    if normalize(profile.read_text()) != normalize(PROFILE):
        raise Miss("unknown nvcc profile")


def compiler_environment(env):
    for name, value in env.items():
        if (name.startswith(("NVCC_", "LD_", "GCC_"))
                      and name not in (*OVERRIDES, *COMPILER_ENV)):
            raise Miss("unknown compiler environment override")
    return {name: digest(os.fsencode(env[name])) for name in COMPILER_ENV if name in env}


class Miss(Exception):
    pass


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def identity(path):
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode):
        raise Miss("non-regular payload/input")
    with path.open("rb") as reader:
        before = os.fstat(reader.fileno())
        value = hashlib.file_digest(reader, "sha256").hexdigest()
        after = os.fstat(reader.fileno())
    if (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
        raise Miss("input changed during hashing")
    return {"sha256": value, "bytes": before.st_size,
            "mode": stat.S_IMODE(before.st_mode)}


def inventory(roots):
    """Bind files/link targets and absence-sensitive search directory membership."""
    policies = roots if isinstance(roots, dict) else {Path(root): "tree" for root in roots}
    records, active = {}, set()
    def visit(path, policy="tree"):
        path = Path(path).absolute()
        spelling = str(path)
        if spelling in active:
            raise Miss("compiler input link cycle")
        info = path.lstat()
        mode = stat.S_IMODE(info.st_mode)
        if stat.S_ISLNK(info.st_mode):
            records[spelling] = {"link": os.readlink(path), "mode": mode}
            active.add(spelling)
            try:
                resolved = path.resolve(strict=True)
                # Tool/header/library links into unknown filesystems are a miss.
                if not any(resolved.is_relative_to(root) for root in (Path("/usr"), Path("/lib"), Path("/lib64"))) and not any(
                        resolved.is_relative_to(root.resolve()) for root in policies):
                    raise Miss("compiler input link escapes admitted roots")
                visit(resolved, policy)
            except (RuntimeError, OSError) as error:
                raise Miss("unresolved compiler input link") from error
            finally:
                active.remove(spelling)
        elif stat.S_ISDIR(info.st_mode):
            if records.get(spelling, {}).get("tree"):
                return
            records[spelling] = {"directory": True, "mode": mode, "tree": policy == "tree"}
            for child in sorted(path.iterdir()):
                if policy == "search" and child.is_dir():
                    # The linker searches direct leaves, not descendant folders.
                    records[str(child)] = {"search_subdirectory": True, "mode": stat.S_IMODE(child.lstat().st_mode)}
                    if child.is_symlink():
                        records[str(child)]["link"] = os.readlink(child)
                else:
                    visit(child, policy)
        elif stat.S_ISREG(info.st_mode):
            records[spelling] = identity(path)
        else:
            raise Miss("unsupported compiler input type")
    for root, policy in policies.items():
        visit(root, policy)
    return records


def visible_roots(repo, nvcc):
    toolkit = nvcc.parent.parent
    if not nvcc.is_relative_to(toolkit / "bin") or not (toolkit / "bin/nvcc.profile").is_file():
        raise Miss("unknown nvcc layout")
    admit_profile(toolkit / "bin/nvcc.profile")
    roots = {toolkit: "tree", Path("/usr/include"): "tree", Path("/usr/lib/gcc"): "tree"}
    host = Path(shutil.which("g++") or "").resolve()
    gcc = Path(shutil.which("gcc") or "").resolve()
    if not host.is_relative_to(Path("/usr/bin")) or not gcc.is_relative_to(Path("/usr/bin")):
        raise Miss("unknown host compiler")
    executables = [host, gcc]
    for name in ("cc1plus", "as", "ld", "collect2"):
        value = subprocess.check_output([str(host), "-print-prog-name=" + name], text=True).strip()
        executable = Path(value if "/" in value else shutil.which(value) or "").resolve()
        if not executable.is_file() or not executable.is_relative_to(Path("/usr")):
            raise Miss("unknown host tool input")
        executables.append(executable)
    for executable in executables:
        roots[executable] = "file"
        linked = subprocess.check_output(["/usr/bin/ldd", str(executable)], text=True)
        for value in re.findall(r"(?:=>\s*)?(/[^\s()]+)", linked):
            library = Path(value)
            if not library.resolve().is_relative_to(Path("/usr")):
                raise Miss("unknown dynamic compiler input")
            roots[library] = "file"
    searches = subprocess.check_output([str(host), "-print-search-dirs"], text=True)
    rows = [line for line in searches.splitlines() if line.startswith("libraries: =")]
    if len(rows) != 1:
        raise Miss("unknown GNU link search manifest")
    for spelling in rows[0].split("=", 1)[1].split(":"):
        path = Path(spelling).resolve()
        if path.is_dir():
            if not path.is_relative_to(Path("/usr")):
                raise Miss("unknown GNU link search root")
            roots.setdefault(path, "search")
    for name in ("/etc/ld.so.cache", "/etc/ld.so.conf", "/etc/ld.so.conf.d"):
        path = Path(name)
        if path.exists():
            roots[path] = "tree"
    roots[Path("/usr/bin/ldd")] = "file"
    return roots


def phase_commands(repo, nvcc, name):
    """Extract actual host/device preprocessing phases from the unchanged recipe."""
    dry = subprocess.run([*recipe(nvcc, name, Path("/output")), "--dryrun"], cwd=repo,
                         check=True, capture_output=True, text=True).stderr
    variables, phases, tools, counts = {}, [], {}, {}
    allowed = {"rm", "gcc", "g++", "cudafe++", "$CICC_PATH/cicc", "ptxas", "fatbinary", "nvlink"}
    for line in dry.splitlines():
        if not line.startswith("#$ "):
            if line.strip():
                raise Miss("unknown nvcc dryrun output")
            continue
        line = line[3:]
        match = re.fullmatch(r"([A-Z_][A-Z0-9_]*)=(.*)", line)
        if match:
            variables[match[1]] = match[2]
            continue
        tokens = shlex.split(line)
        if not tokens or tokens[0] not in allowed:
            raise Miss("unknown native compiler phase")
        counts[tokens[0]] = counts.get(tokens[0], 0) + 1
        program = tokens[0].replace("$CICC_PATH", variables.get("CICC_PATH", ""))
        executable = Path(program if "/" in program else shutil.which(program, path=variables.get("PATH", "")) or "").absolute()
        resolved = executable.resolve(strict=True)
        if not (resolved.is_relative_to(nvcc.parent.parent) or resolved.is_relative_to(Path("/usr/bin"))):
            raise Miss("unknown native compiler phase executable")
        tools[str(executable)] = "file"
        linked = subprocess.check_output(["/usr/bin/ldd", str(executable)], text=True,
                                        env=dict(os.environ, PATH=variables.get("PATH", ""),
                                                 LD_LIBRARY_PATH=variables.get("LD_LIBRARY_PATH", "")))
        for value in re.findall(r"(?:=>\s*)?(/[^\s()]+)", linked):
            library = Path(value)
            if not (library.resolve().is_relative_to(Path("/usr"))
                    or library.resolve().is_relative_to(nvcc.parent.parent)):
                raise Miss("unknown native phase dynamic library")
            tools[str(library)] = "file"
        if tokens[0] == "gcc" and "-E" in tokens:
            if " > " not in line or tokens.count("-o") != 1 or "tools/" + name + ".cu" not in tokens:
                raise Miss("unknown native preprocessing phase")
            tokens = tokens[:tokens.index(">")]
            index = tokens.index("-o")
            del tokens[index:index+2]
            tokens[0] = str(executable)
            phases.append(tokens)
    expected_counts = {"rm": 2, "gcc": 5, "g++": 1, "cudafe++": 1, "$CICC_PATH/cicc": 2,
                       "ptxas": 2, "fatbinary": 2, "nvlink": 1}
    if counts != expected_counts or len(phases) != 3 or not {"PATH", "LD_LIBRARY_PATH"} <= set(variables):
        raise Miss("incomplete host/device phase manifest")
    phase_env = dict(os.environ, PATH=variables["PATH"], LD_LIBRARY_PATH=variables["LD_LIBRARY_PATH"])
    return phases, phase_env, {Path(path): policy for path, policy in tools.items()}


def preprocess(repo, nvcc):
    records = {}
    for name in NAMES:
        commands, phase_env, tools = phase_commands(repo, nvcc, name)
        phases = []
        for command in commands:
            expanded = subprocess.check_output(command, cwd=repo, env=phase_env)
            if re.search(rb"\.(?:incbin|include)\b", expanded):
                raise Miss("external assembler input directive")
            dependency_command = [token for token in command if token != "-E"]
            dependencies = subprocess.check_output([*dependency_command, "-M", "-MT", "dense-control"], cwd=repo, env=phase_env)
            text = dependencies.decode().replace("\\\n", " ")
            if not text.startswith("dense-control:"):
                raise Miss("unknown dependency manifest")
            paths = text.split(":", 1)[1].split()
            if not paths or any("\\" in path for path in paths):
                raise Miss("unsupported dependency spelling")
            source = {}
            for spelling in paths:
                path = Path(spelling)
                if not path.is_absolute():
                    path = repo / path
                path = path.resolve(strict=True)
                source[str(path)] = identity(path)
            phases.append({"command": command, "expanded_sha256": digest(expanded), "dependencies": source})
        # nvcc -t 2 can report independent device phases in either order. Keep
        # every complete command/expanded-stream/dependency record, canonically.
        records[name] = {"phases": sorted(phases, key=canonical), "tools": inventory(tools)}
    return records


def context(repo, nvcc, roots, env, stub_dir=None):
    overrides = [name for name in OVERRIDES if name in env]
    stub = None
    if stub_dir is not None and env.get("LD_LIBRARY_PATH") == str(stub_dir) + ":":
        link = stub_dir / "libcuda.so.1"
        if not link.is_symlink() or link.resolve() != (nvcc.parent.parent / "lib64/stubs/libcuda.so").resolve():
            raise Miss("driver stub identity differs")
        if any(p.name not in {*NAMES, "libcuda.so.1"} for p in stub_dir.iterdir()):
            raise Miss("unknown loader input in control output")
        overrides.remove("LD_LIBRARY_PATH")
        stub = identity(link.resolve())
    if overrides:
        raise Miss("ambient compiler override")
    return {"schema": SCHEMA, "nvcc": str(nvcc), "repo": str(repo),
            "recipe": [list(recipe(nvcc, name, Path("/output"))) for name in NAMES],
            "environment": compiler_environment(env),
            "inputs": inventory(roots), "source": preprocess(repo, nvcc),
            "controller": identity(Path(__file__).resolve()), "driver_stub": stub}


def recipe(nvcc, name, out):
    toolkit = nvcc.resolve().parent.parent
    return [str(nvcc), *FLAGS, "tools/" + name + ".cu", "-lcublasLt", "-lcublas",
            "-ldl", "-L" + str(toolkit / "lib64/stubs"), "-lcuda", "-o", str(out / name)]


def payloads(folder):
    values = {}
    for name in NAMES:
        path = folder / name
        if path.lstat().st_size > 96 * 1024 * 1024:
            raise Miss("cache payload exceeds size ceiling")
        values[name] = identity(path)
        with path.open("rb") as reader:
            header = reader.read(64)
            if (len(header) != 64 or header[:7] != b"\x7fELF\x02\x01\x01"
                    or struct.unpack_from("<H", header, 16)[0] not in (2, 3)
                    or struct.unpack_from("<H", header, 18)[0] != 62
                    or not values[name]["mode"] & 0o111):
                raise Miss("payload is not an executable ELF64 x86_64")
    return values


def restore(entry, expected, out):
    if entry.is_symlink() or not entry.is_dir() or (entry / "manifest.json").is_symlink():
        raise Miss("cache entry/manifest is not regular")
    manifest = entry / "manifest.json"
    if not stat.S_ISREG(manifest.lstat().st_mode) or manifest.stat().st_size > 64 * 1024 * 1024:
        raise Miss("cache manifest type/size differs")
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise Miss("duplicate cache manifest key")
            result[key] = value
        return result
    def constant(value):
        raise Miss("non-finite cache manifest number")
    record = json.loads(manifest.read_text(), object_pairs_hook=unique, parse_constant=constant)
    if not isinstance(record, dict) or set(record) != {"context", "payloads"} or canonical(record["context"]) != canonical(expected):
        raise Miss("incomplete or changed cache manifest")
    if set(p.name for p in entry.iterdir()) != {*NAMES, "manifest.json"}:
        raise Miss("cache inventory differs")
    if canonical(payloads(entry)) != canonical(record["payloads"]):
        raise Miss("corrupt cache payload")
    with tempfile.TemporaryDirectory(prefix=".restore-", dir=out.parent) as temporary:
        stage = Path(temporary)
        for name in NAMES:
            shutil.copy2(entry / name, stage / name, follow_symlinks=False)
        if canonical(payloads(stage)) != canonical(record["payloads"]):
            raise Miss("copied cache payload differs")
        for name in NAMES:
            os.replace(stage / name, out / name)
    if canonical(payloads(out)) != canonical(record["payloads"]):
        raise Miss("restored payload differs")


def build(repo, nvcc, out, argv0=None):
    for name in NAMES:
        command = recipe(nvcc, name, out)
        if argv0 is not None:
            command[0] = argv0
        subprocess.run(command, cwd=repo, check=True)


def unrestricted_payloads(out):
    # Uncached builds retain original host/compiler semantics. Cache-only ELF
    # admission cannot veto a valid native binary on another host architecture.
    return {name: identity(out / name) for name in NAMES}


def differences(before, after, prefix=""):
    if isinstance(before, dict) and isinstance(after, dict):
        result = []
        for key in sorted(set(before) | set(after)):
            path = prefix + "/" + str(key)
            if key not in before or key not in after:
                result.append(path)
            else:
                result += differences(before[key], after[key], path)
        return result
    if isinstance(before, list) and isinstance(after, list) and len(before) == len(after):
        return [path for index, (a, b) in enumerate(zip(before, after))
                for path in differences(a, b, prefix + "/" + str(index))]
    return [] if canonical(before) == canonical(after) else [prefix]


def reason(error):
    # Miss messages are fixed diagnostics, never raw environment values.
    return str(error) if isinstance(error, Miss) else type(error).__name__


def run(repo, nvcc, out, cache):
    raw_nvcc = str(nvcc)
    repo, out, cache = (Path(p).resolve() for p in (repo, out, cache))
    nvcc = Path(nvcc).absolute()
    out.mkdir(parents=True, exist_ok=True)
    if (cache.is_relative_to(repo / "tools") or cache.is_relative_to(repo / "crates")
            or out.is_relative_to(cache) or cache.is_relative_to(out)):
        raise ValueError("cache/output overlaps compiler source or each other")
    env, expected, key, roots = dict(os.environ), None, None, None
    try:
        if not Path(raw_nvcc).is_absolute() or nvcc.resolve() != nvcc:
            raise Miss("noncanonical compiler spelling")
        roots = visible_roots(repo, nvcc)
        expected = context(repo, nvcc, roots, env, out)
        key = digest(canonical(expected))
        try:
            restore(cache / key, expected, out)
            after = context(repo, nvcc, roots, env, out)
            if canonical(after) != canonical(expected):
                print("dense-control changed inputs:", differences(expected, after)[:20], flush=True)
                raise Miss("compiler inputs changed during restore")
            print(json.dumps({"status": "restored", "key": key, "payloads": payloads(out)}), flush=True)
            return
        except (Miss, OSError, ValueError) as error:
            print("dense-control cache miss:", reason(error), flush=True)
    except (Miss, OSError, ValueError, subprocess.CalledProcessError) as error:
        expected = None
        print("dense-control cache unavailable:", reason(error), flush=True)

    # Never catch/retry an actual compiler failure. This is exactly one original
    # build attempt, outside cache/census failure handling, with original argv[0].
    build(repo, nvcc, out, raw_nvcc)
    status = "uncached"
    if expected is not None:
        try:
            after = context(repo, nvcc, roots, env, out)
            if canonical(after) != canonical(expected):
                print("dense-control changed inputs:", differences(expected, after)[:20], flush=True)
                raise Miss("compiler inputs changed during build")
            record = {"context": expected, "payloads": payloads(out)}
            cache.mkdir(parents=True, exist_ok=True)
            entry = cache / key
            with tempfile.TemporaryDirectory(prefix=".writer-", dir=cache) as temporary:
                stage = Path(temporary)
                for name in NAMES:
                    shutil.copy2(out / name, stage / name)
                (stage / "manifest.json").write_bytes(canonical(record))
                with tempfile.TemporaryDirectory(prefix=".verify-", dir=out.parent) as check:
                    restore(stage, expected, Path(check))
                if entry.exists() or entry.is_symlink():
                    shutil.rmtree(entry) if entry.is_dir() and not entry.is_symlink() else entry.unlink()
                stage.rename(entry)
            status = "built"
        except (Miss, OSError, ValueError, subprocess.CalledProcessError) as error:
            print("dense-control cache not stored:", reason(error), flush=True)
    print(json.dumps({"status": status, "key": key, "payloads": unrestricted_payloads(out)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=".")
    parser.add_argument("--nvcc", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--cache", required=True)
    args = parser.parse_args()
    try:
        run(args.repo, args.nvcc, args.out, args.cache)
    except subprocess.CalledProcessError as error:
        raise SystemExit(error.returncode if error.returncode > 0 else 128 - error.returncode)


if __name__ == "__main__":
    main()
