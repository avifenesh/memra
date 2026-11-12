#!/usr/bin/env python3
"""receipts.py: copy memra#1221's trimmed receipts into the PR tree. Per cell the small files
(ab.jsonl, guard.log, run.log, trace.txt, holder.log, kernel.log, kernel-oom-count.txt, gate
logs); any text file over 64 KB is gzipped. Idempotent: the destination is rewritten."""
import gzip
import pathlib
import shutil

D = pathlib.Path.home() / ".cache/memra-1221"
R = D / "wt/research/spec-capture-headroom-20261112"
KEEP = {"ab.jsonl", "guard.log", "run.log", "trace.txt", "holder.log", "kernel.log", "kernel-oom-count.txt",
        "campaign.log", "gap.txt", "split.txt", "pf.jsonl"}


# The repository is public: home paths become ~ and the two Spark units are named by number.
SUBS = [(str(pathlib.Path.home()), "~"), ("user", "user"), ("u2", "u2"), ("u1", "u1")]


def clean(t: str) -> str:
    for a, b in SUBS:
        t = t.replace(a, b)
    return t


def put(src: pathlib.Path, dst: pathlib.Path):
    dst = pathlib.Path(clean(str(dst.relative_to(R))))
    dst = R / dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    data = clean(src.read_text(errors="replace")).encode()
    if len(data) > 64 * 1024:
        with gzip.open(str(dst) + ".gz", "wb") as g:
            g.write(data)
    else:
        dst.write_bytes(data)


def tree(src: pathlib.Path, dst: pathlib.Path, names=KEEP):
    for p in sorted(src.rglob("*")):
        if p.is_file() and (names is None or p.name in names):
            put(p, dst / p.relative_to(src))


for sub in ("gb10", "rtx5090", "cpu", "tools"):
    shutil.rmtree(R / sub, ignore_errors=True)
for c in ("probe131k", "probe260k", "ih131k-u1", "zt131k", "fa260k-u1", "pa260k-u1", "qa260k-u1",
          "qa131k-u1", "short-u1"):
    if (D / "runs" / c).exists():
        tree(D / "runs" / c, R / "gb10" / c)
for c in ("nsys131k-B", "nsys260k-B"):
    tree(D / "nsys" / c, R / "gb10/nsys" / c)
g = D / "runs/gates"
if g.exists():
    for p in sorted(g.iterdir()):
        if p.is_file():
            put(p, R / "gb10/gates" / p.name)
        else:
            tree(p, R / "gb10/gates" / p.name, None)
for c in ("rtx28k", "rtx60k", "rtx100k", "rtx28k-v", "rtx28k-v2"):
    if (D / "runs" / c).exists():
        tree(D / "runs" / c, R / "rtx5090" / c)
for f in ("full-final.txt", "census-final.txt", "census-narrow-trim.txt", "build5090.log", "build5090-d75d.log"):
    if (D / "logs" / f).exists():
        put(D / "logs" / f, R / "cpu" / f)
for f in ("specpub_cell.sh", "specpub_ab.py", "specpub_tab.py", "serve.sh", "serve_short.sh", "fill_cache.sh",
          "short_cell.sh", "mem_holder.py", "short_req.py", "unit_cell.sh", "ab_run.sh", "rtx_cell.sh", "rtx_ab.sh",
          "probe_i.py", "probe_i782.py", "probe_p.py", "probe_h.py", "spark_build_probe.sh", "spark_build_arms.sh",
          "gate_run.sh", "rows.py", "tsrun.py", "pf_long_memra.py", "eos_ab.py", "specpub_nsys.sh", "specpub_gap.py",
          "specpub_split.py", "sizes-memra.json", "receipts.py"):
    put(D / "tools" / f, R / "tools" / f)
total = sum(p.stat().st_size for p in R.rglob("*") if p.is_file())
print(f"receipts: {sum(1 for p in R.rglob('*') if p.is_file())} files, {total / 1e6:.2f} MB")
