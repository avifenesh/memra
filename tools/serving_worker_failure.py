"""Narrow worker-panic/respawn predicate, not request-fault or native qualification.

The trusted program specifies victim chat streams, ONE short completion driver
(trigger), then recovery requests. MEMRA_PANIC_AFTER=1 fires during retirement:
its completed driver may succeed; admitted victims must close with the exact
worker-restart error plus DONE, never silent/truncated200. The first worker is
0, its one permitted respawn is1. Observed loading/warming phases stay observations,
not mandatory events for routes which skip calibration. A clean HTTP result alone
cannot establish recovery. Every planned request and every probe is accounted.
"""
import ipaddress
import math
import re
from pathlib import Path

from serving_capture import encoded
from serving_completion import _keys, _same
from serving_drain_capture import require_inflight_prefix
from serving_lifecycle import account_generations
from serving_policy import _identity, _range, _completed_response, evaluate_scenario
from serving_release import account_attempts, json_object, require, sse_events

SCHEMA = "memra-worker-failure-program-v1"
PROBE_PREFIX = "wf-probe-"
PANIC = "MEMRA_PANIC_AFTER=1 fault injection: deliberate worker panic after 1 completed request(s)"
DETAIL = "worker thread panicked: " + PANIC
PANIC_LINE = "[worker] PANIC in the GPU worker thread: " + PANIC
RESPAWN_LINE = "[worker] respawn attempt 1/1 in 2s (reloading weights)"
CLOSED_MESSAGE = "worker closed the stream without completing (worker restart in progress)"
REQUIREMENTS = {"injected_fault_observed":True,"typed_failure":True,"all_attempts_accounted":True,
                "generation_change_observed":True,"recovery_completion":True}
MAX_LOG = 64 * 1024 * 1024
MAX_PROBES = 512


def validate_program(required, program):
    _keys(required, {"id","scope","scenario","requirements"}, "worker failure cell")
    require(required["scenario"] == "worker_failure_recovery" and _same(required["requirements"], REQUIREMENTS),
            "unknown worker failure requirements")
    scope=required["scope"]
    _keys(scope, {"id","model","route","profile"}, "worker failure scope")
    require(all(type(v) is str and v.strip() for v in scope.values())
            and required["id"] == scope["id"] + "/worker_failure_recovery", "invalid worker failure scope")
    _keys(program, {"schema","mode","cell_id","scope","server_identity","endpoint","identities","http","timing","requests"}, "worker failure program")
    require(program["schema"] == SCHEMA and program["mode"] == "worker_failure_recovery"
            and program["cell_id"] == required["id"] and _same(program["scope"],scope), "program differs from required cell")
    _identity(program["server_identity"])
    _keys(program["endpoint"], {"host","port"}, "endpoint")
    require(type(program["endpoint"]["host"]) is str, "literal loopback required")
    try: require(ipaddress.ip_address(program["endpoint"]["host"]).is_loopback, "literal loopback required")
    except ValueError as error: raise ValueError("literal loopback required") from error
    require(type(program["endpoint"]["port"]) is int and 1<=program["endpoint"]["port"]<=65535, "invalid port")
    require(type(program["identities"]) is dict and "server_binary" in program["identities"]
            and 1<=len(program["identities"])<=32 and all(type(k) is str and k and type(v) is str
            and Path(v).is_absolute() for k,v in program["identities"].items()), "invalid artifact paths")
    _keys(program["http"], {"connect_timeout","read_timeout","wall_timeout","max_body_bytes"}, "HTTP limits")
    for key in ("connect_timeout","read_timeout","wall_timeout"):
        value=program["http"][key]
        require(type(value) in (int,float) and math.isfinite(value) and 0<value<=3600, "invalid HTTP deadline")
    require(type(program["http"]["max_body_bytes"]) is int and 0<program["http"]["max_body_bytes"]<=16*1024*1024, "invalid body cap")
    timing=program["timing"]
    _keys(timing, {"prefix_timeout_s","fault_timeout_s","recovery_timeout_s","poll_s","overall_s"}, "worker failure timing")
    require(all(type(v) in (int,float) and math.isfinite(v) and 0<v<=3600 for v in timing.values())
            and .01<=timing["poll_s"]<=1 and sum(timing[k] for k in ("prefix_timeout_s","fault_timeout_s","recovery_timeout_s"))<timing["overall_s"],
            "invalid worker failure timing budgets")
    requests=program["requests"];roles={"victim":[],"trigger":[],"recovery":[]};ids=set()
    require(type(requests) is list and 3<=len(requests)<=32, "worker failure needs3..32 requests")
    for row in requests:
        require(type(row) is dict and type(row.get("role")) is str and row["role"] in roles, "unknown request role")
        _keys(row, {"id","role","model","wire","path","payload"} | (set() if row["role"]=="victim" else {"prompt_tokens","completion_tokens"}), "worker failure request")
        require(type(row["id"]) is str and row["id"].strip() and row["id"] not in ids
                and not row["id"].startswith(PROBE_PREFIX), "duplicate/empty/reserved request ID")
        ids.add(row["id"])
        require(row["wire"] in ("chat_json","chat_sse") and (row["role"]!="victim" or row["wire"]=="chat_sse")
                and row["path"]=="/v1/chat/completions" and row["model"]==scope["model"], "unsupported request wire/model/path")
        payload=row["payload"]
        require(type(payload) is dict and payload.get("model")==row["model"]
                and type(payload.get("stream")) is bool and payload["stream"]==(row["wire"]=="chat_sse"), "sent payload differs from wire/model")
        if row["role"]!="victim":
            _range(row["prompt_tokens"],"prompt tokens");_range(row["completion_tokens"],"completion tokens")
        roles[row["role"]].append(row)
    require(roles["victim"] and len(roles["trigger"])==1 and roles["recovery"], "missing victim, unique driver or recovery")
    encoded(program)
    return roles


def fault_environment(launch):
    env=launch.get("env")
    require(type(env) is dict and env.get("MEMRA_PANIC_AFTER")=="1" and env.get("MEMRA_WORKER_RESPAWN")=="1"
            and "MEMRA_FAULT_INJECT_CACHE_SALT" not in env, "launch does not select the one-shot WORKER fault")
    require(not any(k.startswith("MEMRA_API_KEY") for k in env), "isolated unauthenticated loopback required")


def log_facts(baseline, raw):
    require(type(baseline) is bytes and type(raw) is bytes and 0<len(raw)<=MAX_LOG and raw.startswith(baseline)
            and (not baseline or baseline.endswith(b"\n")) and raw.endswith(b"\n"), "incomplete or changed worker log")
    text=raw.decode("utf-8");old=baseline.decode("utf-8")
    require("[worker] PANIC" not in old and "[worker] respawn attempt" not in old, "fault was already present before this cell")
    lines=text[len(old):].split("\n")
    panics=[i for i,l in enumerate(lines) if l.startswith("[worker] PANIC")]
    respawns=[i for i,l in enumerate(lines) if l.startswith("[worker] respawn attempt")]
    require(len(panics)==len(respawns)==1 and lines[panics[0]]==PANIC_LINE and lines[respawns[0]]==RESPAWN_LINE
            and panics[0]<respawns[0], "missing, repeated, wrong or reordered injected panic/respawn")
    require(not any(l.startswith(("[worker] FATAL", "[fault] request=")) for l in lines), "unrecovered worker or request-fault substitute")
    return {"panic_line":panics[0]+1,"respawn_line":respawns[0]+1,"panic":PANIC,"respawns":1}


def probe_facts(row, model):
    require(row.get("method")=="GET" and row.get("path") in ("/health","/readyz")
            and row.get("transport_error") is None, "incomplete or wrong health/readiness probe")
    payload=json_object(row["body"]);worker=payload.get("worker")
    require(type(worker) is dict and type(worker.get("generation")) is int and worker["generation"] in (0,1)
            and worker.get("phase") in ("idle","busy","dead","loading","warming"), "invalid worker generation/phase")
    require(type(payload.get("models")) is list and model in payload["models"], "probe does not identify requested model")
    if row["status"]==200:
        require(payload.get("status")==("ok" if row["path"]=="/health" else "ready")
                and worker["phase"] in ("idle","busy") and "detail" not in payload, "untruthful healthy probe")
    else:
        require(row["status"]==503 and payload.get("status")==("unhealthy" if row["path"]=="/health" else "not_ready")
                and payload.get("detail") in (DETAIL, "worker fault", "worker thread is gone",
                    "worker is (re)loading weights", "worker is warming: boot calibration probe in flight (readiness follows its completion)"),
                "probe does not report a source-defined worker-unavailable reason")
        hints={}
        for key,value in row.get("headers",[]):
            if key.lower() in ("retry-after","retry-after-ms","x-should-retry"):
                require(key.lower() not in hints, "duplicate worker retry hint");hints[key.lower()]=value
        require(hints=={"retry-after":"2","retry-after-ms":"2000"}, "worker retry hint differs from source")
    return {"id":row["id"],"path":row["path"],"status":row["status"],"generation":worker["generation"],"phase":worker["phase"],"detail":payload.get("detail"),
            "started_ns":row["started_ns"],"finished_ns":row["finished_ns"]}


def typed_victim(row, model):
    require(row["status"]==200 and row.get("transport_error") is None, "victim did not finish a complete error stream")
    error_seen=False;done=False;frames=0
    for kind,data in sse_events(row["body"]):
        require(kind=="message" and not done, "malformed/late worker error stream")
        if data=="[DONE]":
            require(error_seen, "victim DONE lacks typed failure");done=True;continue
        require(not error_seen, "data after worker error")
        value=json_object(data)
        if "error" in value:
            _keys(value,{"error"},"worker failure body")
            require(_same(value["error"],{"message":CLOSED_MESSAGE,"type":"server_error","param":None,"code":"overloaded"}), "wrong worker failure error")
            error_seen=True
        else:
            require(value.get("model")==model and type(value.get("choices")) is list and len(value["choices"])==1,
                    "foreign/malformed victim frame")
            choice=value["choices"][0]
            require(type(choice) is dict and type(choice.get("index")) is int and choice["index"]==0
                    and choice.get("finish_reason") is None and type(choice.get("delta")) is dict and "usage" not in value,
                    "victim supplied a successful terminal or malformed delta")
            delta=choice['delta']
            require(not delta.get('tool_calls') and all(delta.get(k) is None or type(delta[k]) is str
                    for k in ('content','reasoning','reasoning_content')), 'invalid victim text delta')
            frames+=1
    require(error_seen and done, "victim stream truncated before typed error/DONE")
    return {"id":row["id"],"error_code":"overloaded","message":CLOSED_MESSAGE,"pre_error_frames":frames}


def require_pre_error_prefix(body, end_offset):
    """Reconcile a prospective fragment with its completed SSE frame boundary.

    The first read may split any nonterminal frame, including inside UTF-8 or
    JSON. It must not contain even one byte of the later terminal error frame.
    Walk original bytes so CR/LF normalization and multibyte text cannot move
    the recorded prefix offset. The shared parser remains the syntax authority.
    """
    require(type(body) is bytes, "victim body must retain raw bytes")
    sse_events(body)  # Validate the complete stream before slicing its frames.
    start = 0
    for line in re.finditer(rb'[^\r\n]*(?:\r\n|\r|\n)', body):
        value = line.group().rstrip(b'\r\n')
        if line.start() == 0:
            value = value.removeprefix(b'\xef\xbb\xbf')
        if value:
            continue
        for _, data in sse_events(body[start:line.end()]):
            if data != '[DONE]' and 'error' in json_object(data):
                require(end_offset <= start, "pre-driver prefix already touches terminal worker-error frame")
                return start
        start = line.end()
    raise ValueError("victim has no complete terminal worker-error frame")


def evaluate_worker_failure_cell(required, program, *, launch, attempts, probes, prefixes, log, baseline_log=b""):
    roles=validate_program(required,program);fault_environment(launch)
    accounting=account_attempts(program["requests"],attempts)
    indexed={r["id"]:r for r in attempts};attempts=[indexed[r["id"]] for r in program["requests"]]
    for row in attempts:
        require(row.get("server_identity")==program["server_identity"] and row.get("path")=="/v1/chat/completions"
                and row.get("method")=="POST" and row["started_ns"]<row["finished_ns"], "foreign or invalid request capture")
    require(type(probes) is list and 6<=len(probes)<=MAX_PROBES, "missing/bounded probe census")
    facts=[];previous=-1;generation=0
    for i,row in enumerate(probes,1):
        require(row.get("id")==PROBE_PREFIX+str(i) and row.get("server_identity")==program["server_identity"]
                and type(row.get("started_ns")) is int and type(row.get("finished_ns")) is int
                and previous<row["started_ns"]<=row["finished_ns"], "probe IDs/owner/order differ")
        f=probe_facts(row,required["scope"]["model"])
        require(f["generation"]>=generation, "decreasing worker generation")
        previous=row["finished_ns"];generation=f["generation"];facts.append(f)
    require([(f["path"],f["status"],f["generation"]) for f in facts[:2]]==[("/health",200,0),("/readyz",200,0)], "cell needs fresh generation0 healthy/ready baseline")
    driver=indexed[roles["trigger"][0]["id"]]
    require(facts[1]["finished_ns"]<min(r["started_ns"] for r in attempts), "requests precede initial readiness")
    require(type(prefixes) is dict and set(prefixes)=={r["id"] for r in roles["victim"]}, "victim prefix denominator differs")
    for request in roles["victim"]:
        row=indexed[request["id"]];prefix=prefixes[request["id"]]
        _keys(prefix,{"observed_ns","end_offset","body"},"victim prefix")
        n=prefix["end_offset"]
        require(type(n) is int and 0<n<=min(65536,len(row["body"])) and prefix["body"]==row["body"][:n]
                and prefix["observed_ns"]==row.get("first_body_byte_ns") and row.get("chunks")
                and row["chunks"][0]=={"end_offset":n,"observed_ns":prefix["observed_ns"]}
                and row["started_ns"]<=prefix["observed_ns"]<driver["started_ns"]<row["finished_ns"], "victim was not observed open before the completion driver")
        require_inflight_prefix(request,prefix["body"])
        error_start = require_pre_error_prefix(row['body'], n)
        require(all(chunk['end_offset'] <= error_start for chunk in row['chunks']
                    if chunk['observed_ns'] <= driver['started_ns']),
                "terminal worker-error bytes were observed before the completion driver")
    require(all(f["finished_ns"]>=driver["started_ns"] for f in facts if f["status"]==503), "fault was observed before the completion driver")
    dead=[f for f in facts if f["status"]==503 and f["generation"]==0 and f["phase"]=="dead"
          and f["detail"]==DETAIL and f["started_ns"]>driver["started_ns"]]
    require(all(any(f["path"]==path for f in dead) for path in ("/health","/readyz")), "injected worker unavailability was not observed on both probes")
    require(driver["started_ns"]<min(f["started_ns"] for f in dead), "fault precedes completion driver")
    # Later dead samples belong to the recovery wait, not the first-observation
    # deadline. Retain all samples, and bound the first complete H/R pair.
    first_dead=[next(f for f in dead if f["path"]==path) for path in ("/health","/readyz")]
    require(max(f["finished_ns"] for f in first_dead)-driver["started_ns"]<=int(program["timing"]["fault_timeout_s"]*1e9), "fault observation deadline exceeded")
    pressure_end=max(indexed[r["id"]]["finished_ns"] for r in roles["victim"]+roles["trigger"])
    recovery_start=min(indexed[r["id"]]["started_ns"] for r in roles["recovery"])
    recovery_end=max(indexed[r["id"]]["finished_ns"] for r in roles["recovery"])
    require(pressure_end<recovery_start, "recovery starts before every pressure capture ends")
    for path in ("/health","/readyz"):
        require(any(f["path"]==path and f["status"]==200 and f["generation"]==1
                    and pressure_end<f["started_ns"]<=f["finished_ns"]<recovery_start for f in facts), "missing fresh recovered health/readiness before requests")
        require(any(f["path"]==path and f["status"]==200 and f["generation"]==1 and recovery_end<f["started_ns"] for f in facts), "missing post-recovery healthy/ready boundary")
    recovered=next(f for f in facts if f["status"]==200 and f["generation"]==1)
    first_ready=[next(f for f in facts if f["path"]==path and f["status"]==200 and f["generation"]==1)
                 for path in ("/health","/readyz")]
    require(max(f["finished_ns"] for f in first_ready)-min(f["started_ns"] for f in dead)<=int(program["timing"]["recovery_timeout_s"]*1e9), "worker recovery observation deadline exceeded")
    require(all(f["status"]==200 and f["generation"]==1 for f in facts if f["started_ns"]>=recovered["started_ns"]), "worker became unavailable after recovery")
    health=[r for r in probes if r["path"]=="/health"]
    generations=account_generations(accounting,attempts,health)
    require(generations["respawns_observed"]==1, "expected exactly one observed worker generation increment")
    wire={r["id"]:r for r in accounting["requests"]}
    completed_driver=_completed_response(roles["trigger"][0],driver,wire[driver["id"]])
    failures=[typed_victim(indexed[r["id"]],r["model"]) for r in roles["victim"]]
    rec_schedule=[{k:r[k] for k in ("id","model","wire","prompt_tokens","completion_tokens")} for r in roles["recovery"]]
    rec_rows=[indexed[r["id"]] for r in roles["recovery"]]
    rec_account=account_attempts(rec_schedule,rec_rows)
    rec_gen=account_generations(rec_account,rec_rows,health)
    recovery=evaluate_scenario({"scenario":"completed_group","server_identity":program["server_identity"],"requests":rec_schedule},
        accounting=rec_account,generations=rec_gen,attempts=rec_rows,health_samples=health)
    return {"id":required["id"],"scope":required["scope"],"scenario":"worker_failure_recovery","qualification":False,
            "planned":accounting["attempted"],"wire_counts":accounting["counts"],"accounting":accounting,
            "generation_observations":generations,"probes":facts,"driver_completion":completed_driver,
            "typed_failures":failures,"recovery":recovery,"log":log_facts(baseline_log,log),
            "limits":"Sampled worker epoch and client wire facts; no timing benchmark, native or release qualification."}
