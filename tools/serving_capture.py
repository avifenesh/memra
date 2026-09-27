"""Collect a bounded, owned-server request schedule into hash-bound raw evidence.

This module does not qualify a release. The independent serving policy and release
record verifier must select required scenarios and validate these observations.
Run only inside the physical-card lease on a non-serving host for native capture.

A group's optional ``metrics`` flag must be a bool (default False). True records
raw GET /metrics before and after its requests, inside the listener and health
brackets: health_before, metrics_before, requests, metrics_after, health_after.
Metrics bodies and statuses are not interpreted here. Transport/framing failures
fail capture after retaining the requests and closing observations; HTTP non-200
and malformed JSON remain raw observations for downstream policy to reject.
Counter values never establish a cache pass or a cache-tier attribution here.
Probe IDs are reserved across the complete plan. Startup uses startup-health,
then startup-health-2, etc.; that prefix namespace is unavailable to group probes
and requests. Group health IDs are <group>-before/after; optional metrics IDs are
<group>-metrics-before/after. Generated IDs must be disjoint from all planned IDs.
"""

import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import threading
import time

from serving_http import capture_request
from serving_lifecycle import account_generations
from serving_listener import ListenerOwnershipError, prove_listener
from serving_process import OwnedServer, ServingProcessError
from serving_release import WIRE_VALIDATORS, account_attempts, json_object, require


def encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def file_identity(path):
    path = Path(path)
    require(path.is_absolute(), "identity path must be absolute")
    digest, size = hashlib.sha256(), 0
    with path.open("rb") as source:
        before = os.fstat(source.fileno())
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
            size += len(block)
        after = os.fstat(source.fileno())
    require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) ==
            (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
            and size == after.st_size, "identity file changed while hashing")
    return {"bytes": size, "sha256": digest.hexdigest()}


class EvidenceStore:
    """Write exact bytes once; references use a canonical content-addressed path."""

    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(mode=0o700, parents=True, exist_ok=False)
        (self.root / "blobs").mkdir()
        self.payloads = {}

    def put(self, data):
        require(isinstance(data, bytes), "evidence must be exact bytes")
        digest = hashlib.sha256(data).hexdigest()
        name = f"blobs/{digest}"
        if name not in self.payloads:
            with (self.root / name).open("xb") as output:
                output.write(data)
            self.payloads[name] = digest
        return {"path": name, "sha256": digest}

    def obj(self, value):
        return self.put(encoded(value))

    def observation(self, value):
        record = dict(value)
        record["body"] = self.put(record["body"])
        return self.obj(record)

    def index(self, record):
        temporary = self.root / "capture.tmp"
        temporary.write_bytes(encoded({**record, "payloads": self.payloads}))
        os.replace(temporary, self.root / "capture.json")


def _positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def _group_probe_ids(group):
    probes = {"health_before": group["id"] + "-before", "health_after": group["id"] + "-after"}
    if group.get("metrics", False):
        probes.update(metrics_before=group["id"] + "-metrics-before",
                      metrics_after=group["id"] + "-metrics-after")
    return probes


def _reserved_startup_probe_id(value):
    return value == "startup-health" or value.startswith("startup-health-")


def collect_clients(requests, mode, invoke, persist, timeout, errors):
    """Own all client threads until cancellation settles and results are durable.

    Even serial requests run in a worker so an interrupt in the caller can set
    the HTTP cancellation event before joining. Invoke must honor that event and
    its own bounded I/O deadline (capture_request implements both).
    """
    cancel, release, lock = threading.Event(), threading.Event(), threading.Lock()
    results, recorded, threads, finished, failures, problem = {}, set(), [], {}, {}, None
    unstarted = set()

    def worker(request):
        release.wait()
        try:
            result = invoke(request, cancel)
            with lock:
                results[request["id"]] = result
        except BaseException as error:
            with lock:
                failures[request["id"]] = f"{type(error).__name__}: {error}"
        finally:
            finished[request["id"]].set()

    def flush():
        with lock:
            ready = [(r["id"], results[r["id"]]) for r in requests
                     if r["id"] in results and r["id"] not in recorded]
        for request_id, result in ready:
            persist(result)
            recorded.add(request_id)

    try:
        for request in requests:
            finished[request["id"]] = threading.Event()
            thread = threading.Thread(target=worker, args=(request,),
                                      name="serving-client-" + request["id"])
            # Register before native launch: start() can be interrupted after
            # creating the OS thread but before returning to its caller.
            threads.append(thread)
            try:
                thread.start()
            except RuntimeError:
                if thread.ident is None:
                    # A genuine start failure has no native worker to join.
                    unstarted.add(thread)
                    errors[request["id"]] = "client thread could not start"
                raise
            if mode == "serial":
                release.set()
                thread.join(timeout=timeout + 2)
                flush()
                require(finished[request["id"]].is_set(), "serial client exceeded capture deadline")
                require(not failures, "serial client failed")
        release.set()
        deadline = time.monotonic() + timeout + 2
        for thread in threads:
            thread.join(timeout=max(0, deadline - time.monotonic()))
            flush()
        require(all(finished[t.name.removeprefix("serving-client-")].is_set() for t in threads),
                "concurrent client exceeded capture deadline")
    except BaseException as error:
        problem = error
    finally:
        if problem is not None or not all(done.is_set() for done in finished.values()):
            cancel.set()
        release.set()
        deadline = time.monotonic() + timeout + 2
        for thread in threads:
            if thread in unstarted:
                continue
            done = finished[thread.name.removeprefix("serving-client-")]
            # An interrupted join on older CPython can mark a Thread stopped
            # before its worker actually returns. The worker's own final event
            # is the completion authority, followed by a bounded join.
            while not done.is_set() and time.monotonic() < deadline:
                try:
                    done.wait(timeout=max(0, deadline - time.monotonic()))
                except BaseException as error:
                    cancel.set()
                    problem = problem or error
            try:
                if thread.ident is not None:
                    thread.join(timeout=max(0, deadline - time.monotonic()))
            except BaseException as error:
                cancel.set()
                problem = problem or error
            if not done.is_set() or thread.is_alive():
                errors[thread.name] = "client failed bounded cancellation and join"
        with lock:
            errors.update(failures)
        # Every completed or cancelled wire observation survives every unwind.
        flush()
    if problem is not None:
        raise problem
    require(not errors and len(results) == len(requests), "incomplete client group")
    return [results[request["id"]] for request in requests]


def validate_plan(plan):
    require(isinstance(plan, dict) and plan.get("schema") == "memra-serving-capture-plan-v1",
            "invalid capture plan schema")
    server = plan.get("server")
    require(isinstance(server, dict), "missing server launch configuration")
    require(server.get("host") in ("127.0.0.1", "::1"), "capture server must bind literal loopback")
    require(type(server.get("port")) is int and 1 <= server["port"] <= 65535, "invalid server port")
    require(isinstance(server.get("argv"), list) and server["argv"], "missing server argv")
    require(isinstance(server.get("env"), dict), "missing explicit server environment")
    require(not any(k.startswith("MEMRA_API_KEY") for k in server["env"]),
            "loopback gate must use its isolated unauthenticated instance")
    for name in ("startup_timeout", "overall_timeout", "drain_timeout", "kill_timeout"):
        require(_positive(server.get(name)), f"missing finite positive {name}")
    identities = plan.get("identities")
    require(isinstance(identities, dict) and identities.get("server_binary") == server["argv"][0]
            and all(isinstance(k, str) and k and isinstance(v, str) and Path(v).is_absolute()
                    for k, v in identities.items()), "missing exact server/artifact identity paths")
    groups, ids, names, probe_ids = plan.get("groups"), set(), set(), set()
    require(isinstance(groups, list) and groups, "empty capture schedule")
    http = plan.get("http")
    require(isinstance(http, dict) and set(http) ==
            {"connect_timeout", "read_timeout", "wall_timeout", "max_body_bytes"},
            "HTTP limits must be explicit")
    require(all(_positive(http[k]) for k in ("connect_timeout", "read_timeout", "wall_timeout")),
            "invalid HTTP deadlines")
    require(type(http["max_body_bytes"]) is int and http["max_body_bytes"] > 0, "invalid body limit")
    for group in groups:
        require(isinstance(group, dict) and isinstance(group.get("id"), str) and group["id"]
                and group["id"] not in names, "duplicate or missing group id")
        names.add(group["id"])
        require(group.get("mode") in ("serial", "concurrent"), "unknown group mode")
        require(type(group.get("metrics", False)) is bool, "group metrics flag must be a bool")
        for probe_id in _group_probe_ids(group).values():
            require(not _reserved_startup_probe_id(probe_id) and probe_id not in probe_ids,
                    "generated probe IDs collide or use reserved startup namespace")
            probe_ids.add(probe_id)
        requests = group.get("requests")
        require(isinstance(requests, list) and requests and len(requests) <= 256,
                "group must schedule 1..256 requests")
        for request in requests:
            require(isinstance(request, dict) and isinstance(request.get("id"), str)
                    and request["id"] and request["id"] not in ids, "duplicate or missing request id")
            ids.add(request["id"])
            require(request.get("wire") in WIRE_VALIDATORS, "unknown response wire format")
            payload = request.get("payload")
            require(isinstance(payload, dict) and isinstance(request.get("model"), str)
                    and request["model"] and payload.get("model") == request["model"],
                    "request model and payload disagree")
            require(type(payload.get("stream")) is bool and
                    payload["stream"] == request["wire"].endswith("_sse"), "request stream mode differs")
            expected_path = "/v1/chat/completions" if request["wire"].startswith("chat_") else "/v1/completions"
            require(request.get("path") == expected_path, "request path differs from wire contract")
            encoded(payload)
    require(not (ids & probe_ids) and not any(_reserved_startup_probe_id(key) for key in ids),
            "planned request ID collides with a generated probe or reserved startup namespace")
    return ids


def collect(plan, output):
    """Run serial/concurrent groups with before/after health and ownership proof.

    Cancellation, overload and drain *qualification* need their separate required
    scenario schedules and predicates. This collector records group failures but
    does not retry requests or promote a completed capture to a passing release.
    """
    validate_plan(plan)
    store = EvidenceStore(output)
    state = {"schema": "memra-serving-capture-v1", "state": "collecting",
             "qualification": False, "plan": store.obj(plan), "groups": [],
             "startup_observations": [], "listener_observations": [], "errors": [], "clock": "monotonic_ns"}
    store.index(state)
    config = plan["server"]
    server = OwnedServer(argv=config["argv"], env=config["env"], cwd=config["cwd"],
                         evidence_dir=str(store.root / "process"),
                         **{k: config[k] for k in ("startup_timeout", "overall_timeout",
                                                 "drain_timeout", "kill_timeout")})

    def ownership():
        deadline = time.monotonic() + min(3.0, config["startup_timeout"])
        while True:
            try:
                proof = prove_listener(server.identity, config["host"], config["port"],
                                       timeout=max(0.001, deadline - time.monotonic()))
                return proof, store.obj(asdict(proof))
            except ListenerOwnershipError as error:
                # A compiler/helper can exit during the complete procfs walk.
                # Retry only this read-only proof, with one bounded total budget.
                # Never weaken its visibility/identity checks or retry requests.
                state["listener_observations"].append(store.obj({"error": str(error),
                                                                "observed_ns": time.monotonic_ns()}))
                store.index(state)
                if time.monotonic() >= deadline or server.receipt().get("state") == "finished":
                    raise
                time.sleep(min(.02, max(0, deadline - time.monotonic())))

    def probe(label, path="/health"):
        value = capture_request(request_id=label, host=config["host"], port=config["port"],
                                path=path, method="GET", body=b"",
                                headers={"X-Request-Id": label}, **plan["http"])
        identity = server.identity
        record = server.receipt()
        value["server_identity"] = {"pid": identity.pid,
                                    "start_identity": record["boot_id"] + ":" + identity.start_time}
        value["path"], value["method"] = path, "GET"
        return value, store.observation(value)

    try:
        state["identities_before"] = {k: file_identity(v) for k, v in plan["identities"].items()}
        store.index(state)
        server.start()
        deadline = time.monotonic() + config["startup_timeout"]
        startup_probes = 0
        while True:
            require(time.monotonic() < deadline, "server startup deadline exceeded")
            require(server.receipt().get("state") != "finished", "server exited during startup")
            try:
                proof, proof_ref = ownership()
            except ListenerOwnershipError as error:
                state["startup_observations"].append(store.obj({"ownership_error": str(error),
                                                               "observed_ns": time.monotonic_ns()}))
                store.index(state)
                time.sleep(0.05)
                continue
            startup_probes += 1
            label = "startup-health" if startup_probes == 1 else f"startup-health-{startup_probes}"
            observed, ref = probe(label)
            state["startup_observations"].append(ref)
            store.index(state)
            if observed["transport_error"] is None and observed["status"] == 200:
                body = json_object(observed["body"])
                require(body.get("status") == "ok", "startup health did not report ok")
                expected_models = {r["model"] for g in plan["groups"] for r in g["requests"]}
                require(isinstance(body.get("models"), list) and expected_models <= set(body["models"]),
                        "owned server does not advertise every scheduled model")
                require(type(body.get("worker", {}).get("generation")) is int,
                        "health lacks worker generation")
                # Health may take time: retain a second ownership observation at
                # the readiness decision as well as the one before the request.
                proof, state["startup_listener"] = ownership()
                server.mark_ready(proof)
                break
            time.sleep(0.05)

        for group in plan["groups"]:
            entry = {"id": group["id"], "schedule": store.obj(group), "observations": [],
                     "state": "collecting", "errors": []}
            state["groups"].append(entry)
            probe_ids = _group_probe_ids(group)
            metrics = []
            _, entry["listener_before"] = ownership()
            before, entry["health_before"] = probe(probe_ids["health_before"])
            store.index(state)
            if group.get("metrics", False):
                value, entry["metrics_before"] = probe(probe_ids["metrics_before"], "/metrics")
                metrics.append(value)
                store.index(state)
            entry["errors"] = {}

            def request_one(request, cancelled):
                result = capture_request(request_id=request["id"], host=config["host"],
                                             port=config["port"], path=request["path"],
                                             body=encoded(request["payload"]),
                                             headers={"Content-Type": "application/json",
                                                      "X-Request-Id": request["id"]},
                                             cancel_event=cancelled, **plan["http"])
                result["server_identity"] = dict(before["server_identity"])
                result["path"], result["method"] = request["path"], "POST"
                return result

            def persist(result):
                entry["observations"].append(store.observation(result))
                store.index(state)

            try:
                ordered = collect_clients(group["requests"], group["mode"], request_one, persist,
                                          plan["http"]["wall_timeout"], entry["errors"])
            except BaseException:
                entry["state"] = "failed"
                store.index(state)
                raise
            accounting = account_attempts(group["requests"], ordered)
            entry["wire_accounting"] = store.obj(accounting)
            if group.get("metrics", False):
                value, entry["metrics_after"] = probe(probe_ids["metrics_after"], "/metrics")
                metrics.append(value)
                store.index(state)
            after, entry["health_after"] = probe(probe_ids["health_after"])
            store.index(state)
            _, entry["listener_after"] = ownership()
            require(before["transport_error"] is None and after["transport_error"] is None,
                    "incomplete health observation cannot bracket generation accounting")
            entry["generation_accounting"] = store.obj(account_generations(accounting, ordered, [before, after]))
            # Metrics cannot erase request failures or short-circuit the planned
            # denominator. Retain both probe bodies, wire and generation records
            # before refusing a transport-incomplete metrics capture. Complete
            # error statuses/invalid JSON are policy inputs, not capture failures.
            for sample in metrics:
                if sample["transport_error"] is not None:
                    entry["errors"][sample["id"]] = "incomplete metrics observation: " + str(sample["transport_error"])
            if entry["errors"]:
                entry["state"] = "failed"
                store.index(state)
            require(not entry["errors"], "incomplete metrics observation cannot finish capture")
            entry["state"] = "captured"
            store.index(state)
        state["state"] = "captured"
    except BaseException as error:
        state["state"] = "failed"
        state["errors"].append(f"{type(error).__name__}: {error}")
    finally:
        if server._supervisor is not None:
            try:
                receipt = server.close(reason="capture_complete" if state["state"] == "captured" else "capture_error")
            except ServingProcessError as error:
                receipt = server.receipt()
                state["errors"].append(str(error))
                state["state"] = "failed"
            state["lifecycle"] = store.obj(receipt)
            expected_stop = "capture_complete" if state["state"] == "captured" else "capture_error"
            if (receipt.get("stop") or {}).get("reason") != expected_stop:
                state["errors"].append("unexpected server stop: " + str((receipt.get("stop") or {}).get("reason")))
                state["state"] = "failed"
            for name in ("output.log", "events.jsonl", "supervisor.log"):
                path = store.root / "process" / name
                if path.exists():
                    state[name] = store.put(path.read_bytes())
            if (receipt.get("errors") or not receipt.get("cleanup", {}).get("complete")
                    or receipt.get("cleanup", {}).get("escalated")
                    or (receipt.get("server_exit") or {}).get("returncode") != 0):
                state["state"] = "failed"
                state["errors"].append("server did not exit and clean up normally")
        try:
            state["identities_after"] = {k: file_identity(v) for k, v in plan["identities"].items()}
            if state.get("identities_before") != state["identities_after"]:
                state["state"] = "failed"
                state["errors"].append("server/artifact bytes changed")
        except Exception as error:
            state["state"] = "failed"
            state["errors"].append(f"identity check failed: {error}")
        store.index(state)
    return state


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    result = collect(json_object(args.plan.read_bytes()), args.out.resolve())
    print(json.dumps({"state": result["state"], "qualification": False, "errors": result["errors"],
                      "groups": len(result["groups"]), "capture": str(args.out.resolve() / "capture.json")}))
    raise SystemExit(0 if result["state"] == "captured" else 1)
