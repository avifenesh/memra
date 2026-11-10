#!/usr/bin/env python3
"""One answer-position cell on the GB10 (memra#1177), run under a sparklock lease.

1. Serve the chat request twice from main's memra-server, greedy, thinking off: once with the
   server's defaults (the MTP draft at one stream) and once with `MEMRA_SPEC_K=0` (plain
   decode, the served batch program). One boot per arm, cold prefix cache.
2. Take the plain arm's answer. Its ids: the server's `/v1/tokenize` of the answer text; the
   cell checks that `/v1/detokenize` of prompt ids + answer ids gives back the rendered prompt
   followed by the answer text. `<|im_end|>` is appended when the answer stopped on it.
3. Teacher-force prompt + answer through `kvq_rows_probe --ids`: the window is the 2,048 rows
   before the first answer position plus every answer position, under the f32-KV prefill (a),
   the prime over the quantized cache (b) and the served batch decode over a prime-written
   cache (c).

usage: answer_cell.py RUN_ROOT CELL MODEL_GGUF
RUN_ROOT holds bins/{memra-server,kvq_rows_probe} and requests/CELL.request.json; output goes
to RUN_ROOT/runs/CELL/. Exit 0 only if every step passed.
"""
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

PORT = 18177
IN_TURN_ROWS = 2048
IM_END = "<|im_end|>"
FLOOR_KB = 5 * 1024 * 1024


def log(d: str, msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {msg}"
    print(line, flush=True)
    with open(f"{d}/cell.log", "a") as f:
        f.write(line + "\n")


def post(path: str, body: dict, timeout: float) -> dict:
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}{path}",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def up() -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/models", timeout=2) as r:
            return r.status == 200
    except (urllib.error.URLError, OSError):
        return False


def mem_available_kb() -> int:
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1])
    return -1


def snapshot(path: str) -> None:
    with open(path, "w") as f:
        f.write(time.strftime("%Y-%m-%dT%H:%M:%SZ\n", time.gmtime()))
        f.write(subprocess.run(["free", "-m"], capture_output=True, text=True).stdout)
        f.write(
            subprocess.run(
                ["nvidia-smi", "--query-gpu=name,temperature.gpu,clocks.sm,power.draw", "--format=csv"],
                capture_output=True,
                text=True,
            ).stdout
        )
        f.write(
            subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv"],
                capture_output=True,
                text=True,
            ).stdout
        )


def serve(d: str, root: str, arm: str, model: str, req: dict) -> dict:
    if up():
        raise SystemExit(f"port {PORT} already serving; refusing to boot over it")
    env = dict(os.environ)
    env["MEMRA_MODELS"] = f"q38={model}"
    env["MEMRA_ADDR"] = f"127.0.0.1:{PORT}"
    env.pop("MEMRA_SPEC_K", None)
    if arm == "plain":
        env["MEMRA_SPEC_K"] = "0"
    out = open(f"{d}/server-{arm}.log", "w")
    t0 = time.time()
    p = subprocess.Popen([f"{root}/bins/memra-server"], stdout=out, stderr=subprocess.STDOUT, env=env, start_new_session=True)
    try:
        while not up():
            if p.poll() is not None:
                raise SystemExit(f"server ({arm}) exited {p.returncode} before it served")
            if time.time() - t0 > 900:
                raise SystemExit(f"server ({arm}) not up after 900 s")
            time.sleep(2)
        boot_s = time.time() - t0
        log(d, f"server {arm} up after {boot_s:.0f} s")
        msgs = {"model": "q38", "messages": req["messages"], "chat_template_kwargs": req["chat_template_kwargs"]}
        prompt_ids = post("/v1/tokenize", msgs, 600)["tokens"]
        rendered = post("/v1/detokenize", {"model": "q38", "tokens": prompt_ids}, 600)["prompt"]
        im_end = post("/v1/tokenize", {"model": "q38", "prompt": IM_END, "add_special_tokens": False}, 60)["tokens"]
        t1 = time.time()
        resp = post("/v1/chat/completions", dict(req, model="q38"), 3600)
        gen_s = time.time() - t1
        ch = resp["choices"][0]
        text = ch["message"]["content"]
        ans_ids = post("/v1/tokenize", {"model": "q38", "prompt": text, "add_special_tokens": False}, 60)["tokens"]
        joined = post("/v1/detokenize", {"model": "q38", "tokens": prompt_ids + ans_ids}, 600)["prompt"]
        res = {
            "arm": arm,
            "boot_s": round(boot_s, 1),
            "request_s": round(gen_s, 1),
            "prompt_ids": prompt_ids,
            "rendered_prompt_tail": rendered[-200:],
            "im_end_ids": im_end,
            "response": resp,
            "answer_text": text,
            "answer_ids": ans_ids,
            "roundtrip_ok": joined == rendered + text,
            "usage_prompt_matches": resp.get("usage", {}).get("prompt_tokens") == len(prompt_ids),
            "usage_completion_tokens": resp.get("usage", {}).get("completion_tokens"),
        }
        with open(f"{d}/served-{arm}.json", "w") as f:
            json.dump(res, f, ensure_ascii=False, indent=1)
        log(
            d,
            f"served {arm}: prompt {len(prompt_ids)} ids, answer {len(ans_ids)} ids "
            f"(usage {res['usage_completion_tokens']}), finish {ch.get('finish_reason')}, "
            f"roundtrip {res['roundtrip_ok']}, usage-prompt match {res['usage_prompt_matches']}, "
            f"{gen_s:.1f} s",
        )
        return res
    finally:
        if p.poll() is None:
            os.killpg(p.pid, signal.SIGTERM)
            try:
                p.wait(timeout=120)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
                p.wait()
        out.close()
        log(d, f"server {arm} stopped rc={p.returncode}")


def probe(d: str, root: str, model: str, ids_path: str, rows: int) -> int:
    cmd = [
        f"{root}/bins/kvq_rows_probe",
        model,
        ids_path,
        "--ids",
        "--rows",
        str(rows),
        "--decode",
        "batch",
        "--decode-from",
        "prime",
        "--out",
        f"{d}/rows.jsonl",
    ]
    t0 = time.time()
    low = 0
    floor_hit = False
    min_kb = mem_available_kb()
    with open(f"{d}/stdout.log", "w") as out, open(f"{d}/stderr.log", "w") as err, open(f"{d}/mem.tsv", "w") as mem:
        mem.write("t_s\tmem_available_kb\n")
        p = subprocess.Popen(cmd, stdout=out, stderr=err, start_new_session=True)
        while p.poll() is None:
            kb = mem_available_kb()
            min_kb = min(min_kb, kb)
            mem.write(f"{time.time() - t0:.1f}\t{kb}\n")
            mem.flush()
            low = low + 1 if kb < FLOOR_KB else 0
            if low >= 2:
                floor_hit = True
                os.killpg(p.pid, signal.SIGKILL)
                break
            time.sleep(1.0)
        rc = p.wait()
    res = {
        "cmd": cmd,
        "exit": rc,
        "wall_s": round(time.time() - t0, 1),
        "min_mem_available_kb": min_kb,
        "killed_at_memory_floor": floor_hit,
    }
    with open(f"{d}/result.json", "w") as f:
        json.dump(res, f, indent=1)
    log(d, f"probe exit {rc} in {res['wall_s']} s, min MemAvailable {min_kb} kB")
    return rc


def main() -> None:
    root, cell, model = sys.argv[1], sys.argv[2], sys.argv[3]
    d = f"{root}/runs/{cell}"
    os.makedirs(d, exist_ok=True)
    req = json.load(open(f"{root}/requests/{cell}.request.json"))
    snapshot(f"{d}/pre.txt")
    log(d, f"cell {cell} start")
    served = {arm: serve(d, root, arm, model, req) for arm in ("default", "plain")}
    pl, df = served["plain"], served["default"]
    if pl["prompt_ids"] != df["prompt_ids"]:
        raise SystemExit("the two boots tokenized the prompt differently")
    if not pl["roundtrip_ok"]:
        raise SystemExit("plain answer ids do not detokenize back to the answer text")
    if not pl["usage_prompt_matches"]:
        raise SystemExit("/v1/tokenize count differs from the served usage.prompt_tokens")
    ids = pl["prompt_ids"] + pl["answer_ids"]
    finish = pl["response"]["choices"][0].get("finish_reason")
    if finish == "stop":
        ids += pl["im_end_ids"]
    p_len, t = len(pl["prompt_ids"]), len(ids)
    if t > 65535:
        raise SystemExit(f"{t} ids: past the f32 reference's 65,535-row limit (memra#1200)")
    first = p_len - 1 - IN_TURN_ROWS
    rows = t - first
    meta = {
        "cell": cell,
        "prompt_tokens": p_len,
        "answer_tokens": len(pl["answer_ids"]),
        "im_end_appended": finish == "stop",
        "finish_reason": finish,
        "total_ids": t,
        "window_first_row": first,
        "window_rows": rows,
        "first_answer_row": p_len - 1,
        "default_equals_plain_text": df["answer_text"] == pl["answer_text"],
        "default_answer_ids": len(df["answer_ids"]),
        "default_finish_reason": df["response"]["choices"][0].get("finish_reason"),
    }
    with open(f"{d}/ids.json", "w") as f:
        f.write(json.dumps(ids))
    with open(f"{d}/meta.json", "w") as f:
        json.dump(meta, f, indent=1)
    log(d, "meta " + json.dumps(meta))
    rc = probe(d, root, model, f"{d}/ids.json", rows)
    snapshot(f"{d}/post.txt")
    log(d, f"CELL-DONE {cell} rc={rc}")
    sys.exit(rc)


if __name__ == "__main__":
    main()
