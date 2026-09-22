#!/usr/bin/env python3
"""WP-B day 30 cell (memra#423): N concurrent requests under ONE synthetic tenant budget, the
budget journal read back, the `balance_after_micro` chain walked.

Pre-registered in research/spill-b-20260919/DAY30.md section 2. Stdlib only. Boots the
deployment binary (darklanes-serve, which links memra-server through the metering seam) on
the local RTX 5090 under the canonical lock, on a scratch ledger with a synthetic tenant, a
synthetic key and a loopback admin listener. Nothing here touches a real tenant, key, box or
balance; every scratch file is deleted when the cell closes and only the journal, the request
ledger, the server log and the summary are copied into --out.

Verdict line:
  JOURNAL-ORDER CELL arm=<arm> N=<n> rows=<r> chain_violations=<v> order_violations=<o>
      conservation=<ok|FAIL> cancel_sum=<s> -> <PASS|FAIL>
PASS is chain_violations == 0 and conservation == ok. A REFUSED line (exit 2) is never a verdict.
"""

import argparse
import fcntl
import hashlib
import json
import os
import secrets
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

LOCK = "/tmp/memra-5090.lock"
TENANT = "cell-tenant"
ALIAS = "journal9"
DEFAULT_MODEL = "/data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf"

PROMPTS = [
    "Name one prime number below twenty and say why it is prime.",
    "Give one sentence about the colour of the sky at noon.",
    "State the boiling point of water at sea level in Celsius.",
    "Write one line of shell that lists files in the current directory.",
    "Name a mammal that lives in the ocean.",
    "What is seven times eight? Answer with the number only.",
    "Name the largest planet in the solar system.",
    "Give one synonym for the word quick.",
    "What year has 366 days? Describe the rule in one sentence.",
    "Name one noble gas.",
    "How many sides does a hexagon have?",
    "Give one sentence about why salt dissolves in water.",
    "Name a programming language that starts with the letter R.",
    "What is the capital letter that follows Q in the alphabet?",
    "Give a one-line definition of a mutex.",
    "Name one river that flows into the sea.",
]


def refused(why):
    print(f"REFUSED: {why}", flush=True)
    sys.exit(2)


def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout


def card_sample(path):
    text = sh("nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv 2>&1")
    text += sh("nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader 2>&1")
    with open(path, "w") as f:
        f.write(text)
    return text


def write_private(path, text):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(text)


def http(method, url, body=None, headers=None, timeout=300):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode() or "null")
    except urllib.error.HTTPError as e:
        raw = e.read().decode(errors="replace")
        try:
            return e.code, json.loads(raw)
        except Exception:
            return e.code, {"raw": raw}


def port_free(port):
    out = sh(f"ss -ltnH 'sport = :{port}'")
    return out.strip() == ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--bin", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=48)
    ap.add_argument("--port", type=int, default=8191)
    ap.add_argument("--admin-port", type=int, default=8192)
    ap.add_argument("--seed-micro", type=int, default=1_000_000_000)
    ap.add_argument("--credit-micro", type=int, default=1000)
    ap.add_argument("--boot-timeout", type=int, default=900)
    ap.add_argument("--lock-retries", type=int, default=30)
    args = ap.parse_args()

    if not os.path.isfile(args.model):
        refused(f"no model at {args.model}")
    if not os.access(args.bin, os.X_OK):
        refused(f"binary not executable: {args.bin}")
    for p in (args.port, args.admin_port):
        if not port_free(p):
            refused(f"port {p} is owned by a process this cell did not start")

    lock_fd = os.open(LOCK, os.O_RDWR | os.O_CREAT, 0o666)
    held = False
    for attempt in range(args.lock_retries):
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            held = True
            break
        except BlockingIOError:
            print(f"lock busy ({attempt + 1}/{args.lock_retries}); waiting 10 s", flush=True)
            time.sleep(10)
    if not held:
        refused("canonical GPU lock busy")

    os.makedirs(args.out, exist_ok=True)
    scratch = f"/tmp/spill-b-day30/cell-{args.arm}"
    shutil.rmtree(scratch, ignore_errors=True)
    os.makedirs(scratch, mode=0o700)

    api_key = "cell-" + secrets.token_hex(16)
    key_sha = hashlib.sha256(api_key.encode()).hexdigest()
    admin_token = "adm-" + secrets.token_hex(16)
    budgets_path = os.path.join(scratch, "budgets.toml")
    metadata_path = os.path.join(scratch, "metadata.toml")
    token_path = os.path.join(scratch, "admin.token")
    ledger_path = os.path.join(scratch, "requests.jsonl")
    journal_path = ledger_path + ".tenant-budget-journal.jsonl"
    write_private(
        budgets_path,
        f'[[budgets]]\ntenant = "{TENANT}"\ncurrency = "USD"\nbalance_micro = {args.seed_micro}\n',
    )
    write_private(
        metadata_path,
        f'[models."{ALIAS}"]\nowned_by = "cell"\nquantization = "nvfp4"\n'
        f'[models."{ALIAS}".pricing]\nprompt = "0.00000022"\ncached_prompt = "0.00000007"\n'
        f'completion = "0.0000015"\n',
    )
    write_private(token_path, admin_token + "\n")
    # The admin listener requires a FILE-backed keyring (inline keys cannot be provisioned).
    keys_path = os.path.join(scratch, "keys.toml")
    write_private(
        keys_path,
        f'[[keys]]\nprefix = "cell"\nsha256 = "{key_sha}"\ntenant = "{TENANT}"\nenabled = true\n',
    )

    base = f"http://127.0.0.1:{args.port}"
    admin = f"http://127.0.0.1:{args.admin_port}"
    env = dict(os.environ)
    env.update(
        {
            "MEMRA_COMPAT": "openai",
            "MEMRA_MODELS": f"{ALIAS}={args.model}",
            "MEMRA_ADDR": f"127.0.0.1:{args.port}",
            "MEMRA_REQUEST_LEDGER": ledger_path,
            "MEMRA_TENANT_BUDGETS": budgets_path,
            "MEMRA_MODEL_METADATA": metadata_path,
            "MEMRA_ADMIN_ADDR": f"127.0.0.1:{args.admin_port}",
            "MEMRA_ADMIN_TOKEN_FILE": token_path,
            "MEMRA_API_KEYS": keys_path,
        }
    )
    summary = {
        "arm": args.arm,
        "bin": args.bin,
        "bin_sha256": hashlib.sha256(open(args.bin, "rb").read()).hexdigest(),
        "model": args.model,
        "n": args.n,
        "max_tokens": args.max_tokens,
        "seed_micro": args.seed_micro,
        "credit_micro": args.credit_micro,
        "executed_not_qualified": True,
    }
    summary["card_before_boot"] = card_sample(os.path.join(args.out, "card-before.txt"))

    log = open(os.path.join(args.out, "server.log"), "w")
    server = subprocess.Popen([args.bin], env=env, stdout=log, stderr=subprocess.STDOUT, cwd=scratch)
    summary["server_pid"] = server.pid
    t0 = time.time()
    ready = False
    while time.time() - t0 < args.boot_timeout:
        if server.poll() is not None:
            break
        try:
            code, _ = http("GET", f"{base}/health", timeout=5)
            if code == 200:
                code2, body2 = http("GET", f"{admin}/admin/readiness", timeout=5,
                                    headers={"Authorization": f"Bearer {admin_token}"})
                if code2 == 200 and body2.get("ready"):
                    ready = True
                    break
        except Exception:
            pass
        time.sleep(2)
    summary["boot_s"] = round(time.time() - t0, 1)
    summary["card_after_boot"] = card_sample(os.path.join(args.out, "card-after-boot.txt"))

    def stop():
        if server.poll() is None:
            server.send_signal(signal.SIGTERM)
            try:
                server.wait(timeout=90)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()
        log.flush()
        summary["card_after_stop"] = card_sample(os.path.join(args.out, "card-after-stop.txt"))

    if not ready:
        stop()
        with open(os.path.join(args.out, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        shutil.rmtree(scratch, ignore_errors=True)
        refused(f"server not ready within {args.boot_timeout} s (exit={server.returncode})")

    ah = {"Authorization": f"Bearer {admin_token}"}
    code, bal0 = http("GET", f"{admin}/admin/tenants/{TENANT}/balance", headers=ah, timeout=10)
    summary["balance_before"] = {"status": code, "body": bal0}

    results = [None] * args.n
    credit_result = {}

    def one(i):
        body = {
            "model": ALIAS,
            "messages": [{"role": "user", "content": PROMPTS[i % len(PROMPTS)]}],
            "max_tokens": args.max_tokens,
            "temperature": 0,
            "stream": False,
        }
        s = time.time()
        try:
            code, resp = http("POST", f"{base}/v1/chat/completions", body,
                              headers={"Authorization": f"Bearer {api_key}"})
        except Exception as e:  # transport error: recorded verbatim, never a dead thread
            code, resp = 0, {"error": f"{type(e).__name__}: {e}"}
        results[i] = {
            "i": i,
            "status": code,
            "elapsed_s": round(time.time() - s, 3),
            "usage": resp.get("usage") if isinstance(resp, dict) else None,
            "request_id": (resp.get("id") if isinstance(resp, dict) else None),
            "error": (resp.get("error") if isinstance(resp, dict) and code != 200 else None),
        }

    def credit():
        time.sleep(0.5)
        code, resp = http(
            "POST", f"{admin}/admin/tenants/{TENANT}/credit",
            {"amount_micro": args.credit_micro, "idempotency_key": "cell-credit-1"},
            headers=ah, timeout=30,
        )
        credit_result.update({"status": code, "body": resp})

    threads = [threading.Thread(target=one, args=(i,)) for i in range(args.n)]
    threads.append(threading.Thread(target=credit))
    load_t0 = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    summary["load_s"] = round(time.time() - load_t0, 1)
    summary["requests"] = results
    summary["credit"] = credit_result
    time.sleep(1.0)
    code, bal1 = http("GET", f"{admin}/admin/tenants/{TENANT}/balance", headers=ah, timeout=10)
    summary["balance_after"] = {"status": code, "body": bal1}

    # The journal as its readers see it: sealed segments in sequence order, then the active file.
    d = os.path.dirname(journal_path)
    name = os.path.basename(journal_path)
    segments = sorted(
        (int(fn[len(name) + 1:]), fn) for fn in os.listdir(d)
        if fn.startswith(name + ".") and len(fn) == len(name) + 6 and fn[len(name) + 1:].isdigit()
    )
    rows = []
    for _, fn in segments + [(None, name)]:
        p = os.path.join(d, fn)
        if os.path.exists(p):
            with open(p) as f:
                rows.extend(json.loads(line) for line in f if line.strip())
    summary["journal_segments"] = len(segments)

    def signed(row):
        return row["amount_micro"] if row["kind"] == "credit" else -row["amount_micro"]

    chain, order = [], []
    prev_stamp = args.seed_micro
    prev_ms = None
    for n, row in enumerate(rows):
        stamp = row.get("balance_after_micro")
        expected = prev_stamp + signed(row)
        if stamp is None or stamp != expected:
            chain.append(
                f"CHAIN n={n} kind={row['kind']} amount={row['amount_micro']} prev={prev_stamp} "
                f"stamp={stamp} expected={expected} delta={None if stamp is None else stamp - expected}"
            )
        if prev_ms is not None and row["unix_ms"] < prev_ms:
            order.append(f"ORDER n={n} unix_ms={row['unix_ms']} prev_unix_ms={prev_ms}")
        prev_ms = row["unix_ms"]
        prev_stamp = stamp if stamp is not None else expected
    total = args.seed_micro + sum(signed(r) for r in rows)
    admin_bal = bal1.get("balance_micro") if isinstance(bal1, dict) else None
    last_stamp = rows[-1].get("balance_after_micro") if rows else args.seed_micro
    conservation = "ok" if (admin_bal == total and last_stamp == total) else "FAIL"
    cancel_sum = sum(int(v.rsplit("delta=", 1)[1]) for v in chain if not v.endswith("delta=None"))
    ok_requests = sum(1 for r in results if r and r["status"] == 200)
    verdict = "PASS" if (not chain and conservation == "ok") else "FAIL"
    line = (
        f"JOURNAL-ORDER CELL arm={args.arm} N={args.n} rows={len(rows)} chain_violations={len(chain)} "
        f"order_violations={len(order)} conservation={conservation} cancel_sum={cancel_sum} -> {verdict}"
    )
    summary.update(
        {
            "rows": len(rows),
            "ok_requests": ok_requests,
            "chain_violations": chain,
            "order_violations": order,
            "conservation": {
                "seed_plus_sum": total,
                "admin_balance_micro": admin_bal,
                "last_stamp": last_stamp,
                "verdict": conservation,
            },
            "cancel_sum": cancel_sum,
            "verdict_line": line,
        }
    )

    stop()
    for src, dst in ((journal_path, "journal.jsonl"), (ledger_path, "requests.jsonl"),
                     (budgets_path, "budgets.toml"), (metadata_path, "metadata.toml")):
        if os.path.exists(src):
            shutil.copyfile(src, os.path.join(args.out, dst))
    for _, fn in segments:
        shutil.copyfile(os.path.join(d, fn), os.path.join(args.out, fn))
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    shutil.rmtree(scratch, ignore_errors=True)
    fcntl.flock(lock_fd, fcntl.LOCK_UN)

    for v in chain:
        print(v)
    for v in order:
        print(v)
    print(f"requests ok={ok_requests}/{args.n} credit_status={credit_result.get('status')} "
          f"boot_s={summary['boot_s']} load_s={summary['load_s']}")
    print(line, flush=True)
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
