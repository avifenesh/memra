#!/usr/bin/env python3
"""Finite Qwen3.5 sampled-MTP experiment. No serving default or support promotion."""

import argparse
import ast
import concurrent.futures
import http.client
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import secrets
import socket
import subprocess
import threading
import time
import tomllib
import traceback

import numpy as np

spec = importlib.util.spec_from_file_location("serving_collector", Path(__file__).with_name("collect-serving-qualification.py"))
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
save, require, digest = base.save, base.require, base.digest

PROMPT = ("Continue with a random sequence of colors. Every item must be red, blue, green, "
          "or yellow. Sequence: red, blue,")
SAMPLES = 512
WIDTH = 8
PERMUTATIONS = 1999
ARMS = {
    "plain": {"MEMRA_SERVE_SPEC": "0"},
    "pmin0": {},
    "positive_graph": {"MEMRA_SPEC_PMIN": "0.5"},
    "positive_eager": {"MEMRA_SPEC_PMIN": "0.5", "MEMRA_SPEC_NOGRAPH": "1"},
    "zero_graph": {"MEMRA_SPEC_PMIN": "1.1", "MEMRA_SPEC_PMIN0": "1"},
    "zero_eager": {"MEMRA_SPEC_PMIN": "1.1", "MEMRA_SPEC_PMIN0": "1", "MEMRA_SPEC_NOGRAPH": "1"},
}
DIAGNOSTICS = {"MEMRA_DEBUG_SPEC": "1", "MEMRA_SKEY_PROBE": "1", "MEMRA_SPEC_STATS": "1",
               "MEMRA_SPEC_PHASE": "1", "MEMRA_SPEC_PHASE_SYNC": "1"}


class ExperimentalServer(base.Server):
    """Reuse the proven transport/lifetime methods, with explicitly recorded experiment settings."""
    def __init__(self, args, out, arm, cache_mb=0, openai=False):
        self.args, self.out = args, out
        self.key = secrets.token_hex(24)
        self.lines, self.condition, self.eof = [], threading.Condition(), False
        with socket.socket() as guard:
            guard.bind(("127.0.0.1", args.port))
        env = {k: v for k, v in os.environ.items() if not k.startswith("MEMRA_")}
        settings = {"MEMRA_ADDR": f"127.0.0.1:{args.port}", "MEMRA_MODELS": f"gate={args.model}",
                    "MEMRA_CTX": "2048", "MEMRA_MAX_SESSIONS": "4", "MEMRA_KV_HOST_MB": "0",
                    "MEMRA_PREFIX_CACHE_MB": str(cache_mb), **DIAGNOSTICS, **ARMS[arm]}
        # Auth alone defaults to OpenAI responses, which omit the native token IDs.
        settings["MEMRA_COMPAT"] = "openai" if openai else "native"
        if openai:
            require(bool(getattr(args, "vendor_metadata", None)), "vendor-profile cells require pinned model metadata")
            settings["MEMRA_MODEL_METADATA"] = str(Path(args.vendor_metadata).resolve())
        env.update(settings)
        env.update(MEMRA_API_KEY=self.key, CUDA_CACHE_PATH=args.private_cache)
        save(out / "environment.json", settings)
        self.process = subprocess.Popen([args.server], env=env, stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT, text=True, bufsize=1,
                                        pass_fds=(args.external_lock,))
        self.log = out / "server.log"
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()


def permutation_rows(a, b, seed=673):
    """Categorical TV statistic, independently at each token position; shuffled-label null."""
    require(a.shape == b.shape and a.ndim == 2, "sample arrays must have equal shape")
    n, width = a.shape
    rng = np.random.default_rng(seed)
    results = []
    for column in range(width):
        values, codes = np.unique(np.concatenate((a[:, column], b[:, column])), return_inverse=True)
        total = np.bincount(codes, minlength=len(values))
        observed_count = np.bincount(codes[:n], minlength=len(values))
        observed = int(np.abs(2 * observed_count - total).sum())
        extreme = 0
        for _ in range(PERMUTATIONS):
            selected = rng.permutation(2 * n)[:n]
            count = np.bincount(codes[selected], minlength=len(values))
            extreme += int(np.abs(2 * count - total).sum()) >= observed
        results.append({"position": column + 1, "categories": len(values), "tv": observed / (2 * n),
                        "p_value": (1 + extreme) / (1 + PERMUTATIONS)})
    return results


def native_tokens(document):
    require(document.get("model") == "gate" and not document.get("error"), "native response identity/error")
    tokens = document.get("tokens")
    require(isinstance(tokens, list) and all(type(t) is int and t >= 0 for t in tokens), "invalid native token ids")
    require(type(document.get("n_tokens")) is int and len(tokens) == document["n_tokens"]
            and 0 < len(tokens) <= WIDTH, "native token count mismatch")
    require(type(document.get("prompt_tokens")) is int and document["prompt_tokens"] > 0, "invalid prompt accounting")
    require(document.get("stop_reason") in ("MaxTokens", "MaxNew", "Length", "Eos", "Stop", "ContextFull"),
            f"unexpected stop reason {document.get('stop_reason')}")
    require(type(document.get("cached_tokens")) is int and document["cached_tokens"] == 0,
            "statistical cell must remain a cold prefix miss")
    return tokens + [-1] * (WIDTH - len(tokens))


def sample_arm(args, out, arm, arm_index):
    out.mkdir()
    template = {"model": "gate", "prompt": PROMPT, "max_tokens": WIDTH, "temperature": 0.8,
                "top_k": 20, "top_p": 0.95, "min_p": 0.0, "frequency_penalty": 0.0,
                "presence_penalty": 0.0, "repetition_penalty": 1.0}
    save(out / "request-template.json", {"request": template, "samples": SAMPLES,
                                         "seed_formula": f"{100000 * arm_index} + index + 1"})
    server = ExperimentalServer(args, out, arm)
    rows, times = [], []
    try:
        server.event("[server] listening on")
        connection = http.client.HTTPConnection("127.0.0.1", args.port, timeout=90)
        try:
            with (out / "responses.jsonl").open("w") as raw:
                for index in range(SAMPLES):
                    request = {**template, "seed": 100000 * arm_index + index + 1}
                    started = time.monotonic()
                    connection.request("POST", "/v1/completions", json.dumps(request),
                                       {"Content-Type": "application/json", "Authorization": "Bearer " + server.key})
                    response = connection.getresponse()
                    payload = response.read().decode()
                    elapsed = time.monotonic() - started
                    document = json.loads(payload)
                    raw.write(json.dumps({"index": index, "seed": request["seed"], "status": response.status,
                                          "wall_seconds": elapsed, "response": document}) + "\n")
                    raw.flush()
                    require(response.status == 200, f"sample {index} returned {response.status}")
                    rows.append(native_tokens(document))
                    times.append(elapsed)
                    if index in (0, 127, 255, 383, 511):
                        print(f"{arm}: {index + 1}/{SAMPLES} real samples", flush=True)
        finally:
            connection.close()
    finally:
        server.close()
    text = server.log.read_text()
    require("EXACTNESS q=0" not in text, "draft proposal had zero probability in reconstructed q")
    if arm != "plain":
        route = "eager" if "eager" in arm else "graph_s"
        require(f"[skey] chain={route} " in text, f"expected {route} path did not engage")
    require(len({tuple(row) for row in rows}) > 1, "sample distribution is degenerate")
    summary = {"samples": len(rows), "distinct_tapes": len({tuple(row) for row in rows}),
               "e2e_seconds_p50_p95_p99": np.quantile(times, [0.5, 0.95, 0.99]).tolist(),
               "request_seconds_total": sum(times)}
    save(out / "summary.json", summary)
    return np.array(rows, dtype=np.int64)


def probe_rounds(text):
    pattern = r"\[R(\d+)\] pos=(\d+) out_len=(\d+) last_tok=(\d+) draft=(\[[^\]]*\]) n_acc=(\d+) bonus=(\d+)"
    rows = []
    for match in re.finditer(pattern, text):
        r, pos, length, last, draft, accepted, bonus = match.groups()
        if int(r) == 0 and rows:
            break  # run-spec performs a seeded repeat; retain the first walk here.
        rows.append({"round": int(r), "pos": int(pos), "out_len": int(length), "last": int(last),
                     "draft": ast.literal_eval(draft), "accepted": int(accepted), "bonus": int(bonus)})
    return rows


def first_matched_cutoff(control, positive):
    """Stop comparing as soon as emitted prefixes differ; matching last tokens is insufficient."""
    if not control or not positive:
        return None
    a_prefix, b_prefix = [control[0]["last"]], [positive[0]["last"]]
    for a, b in zip(control, positive):
        if a_prefix != b_prefix or any(a[k] != b[k] for k in ("pos", "out_len", "last")):
            return None
        if len(b["draft"]) < len(a["draft"]) and a["draft"][:len(b["draft"])] == b["draft"]:
            return {"control": a, "positive": b, "shared_emitted_prefix": a_prefix}
        a_prefix += a["draft"][:a["accepted"]] + [a["bonus"]]
        b_prefix += b["draft"][:b["accepted"]] + [b["bonus"]]
    return None


def cli_probes(args, out):
    out.mkdir()
    reports = {}
    for arm in tuple(ARMS)[1:]:
        env = {k: v for k, v in os.environ.items() if not k.startswith("MEMRA_")}
        env.update(DIAGNOSTICS, **ARMS[arm], MEMRA_SPEC_TEMP="0.8", MEMRA_TOP_K="20", MEMRA_TOP_P="0.95",
                   MEMRA_MIN_P="0", MEMRA_SEED="7", MEMRA_SPEC_K="3", MEMRA_NGEN="64",
                   MEMRA_PROMPT=PROMPT, MEMRA_PRINT_TEXT="1", CUDA_CACHE_PATH=args.private_cache)
        log = out / (arm + ".log")
        started = time.monotonic()
        with log.open("w") as output:
            completed = subprocess.run([args.run_spec, args.model], env=env, stdout=output,
                                       stderr=subprocess.STDOUT, pass_fds=(args.external_lock,), timeout=180)
        text = log.read_text()
        rows = probe_rounds(text)
        require(rows and "=== SELF-CONSISTENCY PASS ===" in text, f"{arm}: sampled replay failed")
        expected_zero_warning = (arm.startswith("zero_") and completed.returncode == 1
                                 and 'speculative rewrite accepted no draft tokens' in text)
        require(completed.returncode == 0 or expected_zero_warning, f"{arm}: CLI failed")
        route = "eager" if "eager" in arm else "graph_s"
        require(f"[skey] chain={route} " in text, f"{arm}: route did not engage")
        tokens = [ast.literal_eval(x) for x in re.findall(r"sampled tokens: (\[[^\]]*\])", text)]
        require(len(tokens) == 1 and len(tokens[0]) == 64, "probe must capture 64 token ids")
        reports[arm] = {"exit_code": completed.returncode, "expected_zero_accept_warning": expected_zero_warning,
                        "wall_seconds": time.monotonic() - started, "rounds": rows, "tokens": tokens[0],
                        "phase_lines": [line for line in text.splitlines() if line.startswith("[spec-phase]")]}
    for graph, eager in (("positive_graph", "positive_eager"), ("zero_graph", "zero_eager")):
        require(reports[graph]["tokens"] == reports[eager]["tokens"], f"{graph}/{eager} token ids differ")
    control = reports["pmin0"]["rounds"]
    positive = reports["positive_graph"]["rounds"]
    cutoff = first_matched_cutoff(control, positive)
    require(cutoff, "positive-PMIN cutoff not observed at a matched control context")
    for arm in ("positive_graph", "positive_eager"):
        rows = reports[arm]["rounds"]
        require(any(r["accepted"] > 0 for r in rows), f"{arm}: acceptance not exercised")
        require(any(r["accepted"] < len(r["draft"]) for r in rows), f"{arm}: residual not exercised")
        require(any(r["accepted"] == len(r["draft"]) > 0 for r in rows), f"{arm}: full-accept bonus not exercised")
    for arm in ("zero_graph", "zero_eager"):
        require(any(not r["draft"] for r in reports[arm]["rounds"]), f"{arm}: no zero-draft round")
    reports["matched_cutoff"] = cutoff
    save(out / "summary.json", reports)


def chat_completion(document):
    choices = document.get("choices", [])
    require(isinstance(choices, list) and len(choices) == 1, "expected one chat completion choice")
    choice = choices[0]
    message = choice.get("message", {})
    content = message.get("content") or ""
    reasoning = message.get("reasoning") or message.get("reasoning_content") or ""
    normalized = {**document, "choices": [{"text": reasoning + content, "finish_reason": choice.get("finish_reason")}]}
    result = base.completion(normalized, 64)
    result.update(content_chars=len(content), reasoning_chars=len(reasoning), surface="chat/completions")
    return result


def vendor_trace(text, profile):
    require("[server] listening on" in text, "missing listener boundary")
    live = text.split("[server] listening on", 1)[1]
    bursts = [line for line in live.splitlines() if line.startswith("[skey] burst sampled=1 ")]
    require(bursts, "no actual HTTP sampled burst; startup canary is insufficient")
    expected = {"temp": profile["default_temperature"], "top_k": profile["default_top_k"],
                "top_p": profile["default_top_p"], "min_p": profile["default_min_p"]}
    expected["pen_on"] = int(profile.get("default_presence_penalty", 0) != 0
                             or profile.get("default_frequency_penalty", 0) != 0
                             or profile.get("default_repetition_penalty", 1) != 1)
    for line in bursts:
        for key, value in expected.items():
            match = re.search(r"\b" + key + r"=([^ ]+)", line)
            require(match is not None and math.isclose(float(match[1]), value, abs_tol=1e-6),
                    f"actual vendor burst {key} does not match pinned profile")
    return {"expected": expected, "burst_count": len(bursts), "first_http_burst": bursts[0]}


def vendor_cell(args, out, arm):
    out.mkdir()
    with Path(args.vendor_metadata).open("rb") as source:
        profile = tomllib.load(source)["models"]["gate"]
    server = ExperimentalServer(args, out, arm, cache_mb=512, openai=True)
    prompt = base.PROMPT
    request = {"model": "gate", "messages": [{"role": "user", "content": prompt}],
               "max_tokens": 64, "cache_salt": "vendor-default"}
    rows = []
    try:
        server.event("[server] listening on")
        for label in ("cold", "warm"):
            doc, receipt = server.request(label, request, "/v1/chat/completions")
            rows.append(chat_completion(doc))
        require(rows[0]["cached_tokens"] == 0, "vendor cold request must miss")
        require(rows[1]["cached_tokens"] == base.server_capture_len(rows[0]["prompt_tokens"]), "vendor warm cache accounting")
        require(rows[0]["spec"] is not None and rows[1]["spec"] is not None, "vendor sampled MTP did not engage")
        barrier = threading.Barrier(4)
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            pending = [pool.submit(server.request, f"concurrent-{i}", {**request, "cache_salt": f"vendor-c{i}"}, "/v1/chat/completions",
                                   barrier=barrier) for i in range(4)]
            replies = [p.result(timeout=180) for p in pending]
        wave = [chat_completion(doc) for doc, _ in replies]
        require(max(r["started_monotonic"] for _, r in replies) < min(r["ended_monotonic"] for _, r in replies),
                "vendor concurrent offers did not overlap")
        server.request("ready", route="/readyz")
    finally:
        server.close()
    text = server.log.read_text()
    trace = vendor_trace(text, profile)
    require(digest(args.vendor_metadata) in text, "server did not report the pinned metadata hash")
    live = text.split("[server] listening on", 1)[1]
    require("[skey] chain=graph_s " in live or "[skey] chain=eager " in live, "vendor HTTP sampled path absent")
    require("source=concurrency" in live, "vendor concurrent admission policy not observed")
    save(out / "summary.json", {"status": "passed", "cold_warm": rows, "concurrent": wave,
                                "profile_sha256": digest(args.vendor_metadata), "profile": profile, "trace": trace,
                                "request_has_no_sampling_overrides": True})


def main(args):
    require(not (args.statistics_only and args.vendor_only), "choose one partial scope")
    if not args.statistics_only:
        require(args.vendor_metadata and Path(args.vendor_metadata).is_file(),
                "vendor-profile HTTP requires --vendor-metadata; generic fallback does not qualify")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    proof = subprocess.check_output(["python3", "tools/tier-lock-proof.py", "--fd", str(args.external_lock),
                                     "--lock", "/tmp/memra-5090.lock", "--owner", "collector"],
                                    text=True, pass_fds=(args.external_lock,))
    (out / "LOCK.json").write_text(proof)
    save(out / "manifest.json", {"runtime_source": "2873dd4ca37faa15cd4261e7b398926cced30106",
                                 "collector_sha256": digest(__file__), "server_sha256": digest(args.server),
                                 "run_spec_sha256": digest(args.run_spec), "model_sha256": digest(args.model),
                                 "samples_per_arm": SAMPLES, "token_positions": WIDTH, "arms": ARMS,
                                 "permutations": PERMUTATIONS, "familywise_alpha": 0.05,
                                 "vendor_profile_sha256": digest(args.vendor_metadata) if args.vendor_metadata else None,
                                 "uncovered": ["multi-head MTP chain", "other model families/topologies", "independent confidence-read clock"]})
    errors = {}
    def checked(name, callback):
        try:
            return callback()
        except Exception as error:
            errors[name] = {"error": str(error), "traceback": traceback.format_exc()}
            save(out / "failures.json", errors)
            print(f"{name}: FAILED: {error}", flush=True)
            return None
    if not args.statistics_only and not args.vendor_only:
        checked("cli_probes", lambda: cli_probes(args, out / "cli-probes"))
    statistics, failures = {}, []
    if not args.vendor_only:
        samples = {arm: checked(arm, lambda arm=arm, index=index: sample_arm(args, out / arm, arm, index))
                   for index, arm in enumerate(ARMS)}
        threshold = 0.05 / ((len(ARMS) - 1) * WIDTH)
        statistics = ({arm: permutation_rows(samples["plain"], samples[arm]) for arm in tuple(ARMS)[1:]}
                      if all(value is not None for value in samples.values()) else {})
        # CPU power control for the issue's old 0.8 -> 0.96 bias, not native evidence.
        a = np.array([0] * 410 + [1] * 102, dtype=np.int64).reshape(-1, 1)
        b = np.array([0] * 492 + [1] * 20, dtype=np.int64).reshape(-1, 1)
        red = permutation_rows(a, b)[0]
        checked("distribution_red", lambda: require(red["p_value"] <= threshold, "distribution test missed old-rule bias"))
        failures = [(arm, row) for arm, rows in statistics.items() for row in rows if row["p_value"] <= threshold]
        save(out / "distribution.json", {"comparisons": statistics, "threshold": threshold, "red_biased_law": red,
                                         "failures": failures, "verdict": ("incomplete" if not statistics else
                                             "difference_detected" if failures else "no_detected_difference"),
                                         "limit": "Finite marginal test, not a proof of full autoregressive distribution equivalence."})
    if not args.statistics_only:
        for label, arm in (("vendor-default", "pmin0"), ("vendor-positive", "positive_graph"), ("vendor-rollback", "pmin0")):
            checked(label, lambda label=label, arm=arm: vendor_cell(args, out / label, arm))
    passed = not errors and not failures and (args.vendor_only or bool(statistics))
    save(out / "result.json", {"status": "passed" if passed else "failed",
                               "scope": "vendor-profile HTTP only" if args.vendor_only else
                                   "statistical marginals only" if args.statistics_only else "full composite experiment",
                               "errors": errors, "distribution_failures": failures})
    require(passed, "sampled cell failed; retain all evidence")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "server", "run-spec", "out", "private-cache"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--port", type=int, default=18121)
    p.add_argument("--external-lock", type=int, required=True)
    p.add_argument("--statistics-only", action="store_true", help="Run the fixed statistical arms only; no CLI or vendor-cell verdict")
    p.add_argument("--vendor-only", action="store_true", help="Run only the three fresh-process vendor-profile HTTP cells")
    p.add_argument("--vendor-metadata", help="Pinned model-specific sampling defaults in MEMRA_MODEL_METADATA TOML format")
    main(p.parse_args())
