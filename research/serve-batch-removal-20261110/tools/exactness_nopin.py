#!/usr/bin/env python3
"""memra #1158: research/spec-gate-20260806/exactness.py without its `MEMRA_SPEC_K` pin.

The harness boots every arm with `MEMRA_SPEC_K=<k>`. On current main an operator K pin disables
the automatic demotion (`[spec-k] operator pin K=3: ... automatic demotion disabled`), so its
loaded DEMOTE arm never demotes; only the forced DEMOTE_SOLO arm does. This wrapper runs the
same harness, same arms and same verdict code, with the pin left out, so the automatic K (3 on
these prompts) runs and the load-triggered demotion fires too.

usage: exactness_nopin.py HARNESS_PATH [harness arguments...]
"""
import importlib.util
import os
import subprocess
import sys

path = sys.argv[1]
sys.argv = [path] + sys.argv[2:]
spec = importlib.util.spec_from_file_location("exactness", path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def boot(env_extra, log_path, addr, models, ctx, k):
    env = dict(os.environ)
    env.update({"MEMRA_MODELS": models, "MEMRA_ADDR": addr, "MEMRA_CTX": str(ctx),
                "MEMRA_TICK_TRACE": "1"})
    env.update(env_extra)
    lf = open(log_path, "wb")
    p = subprocess.Popen(["target/release/memra-server"], stdout=lf, stderr=subprocess.STDOUT,
                         env=env, preexec_fn=os.setsid)
    return p, lf


mod.boot = boot
mod.main()
