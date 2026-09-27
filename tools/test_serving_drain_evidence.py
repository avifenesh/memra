"""Pure synthetic Linux-shaped bundles; no procfs, model or native qualification."""

import copy
import hashlib
import json
from pathlib import Path
import unittest

from serving_capture import encoded
from serving_drain_capture import policy_config, require_inflight_prefix, validate_drain_program
from serving_drain_evidence import read_drain_capture
from serving_policy import evaluate_scenario
from serving_release import ServingGateError, account_attempts
from test_serving_policy import BOOT, OWNER, PROCESS, drain


def ns(value): return int(value * 1_000_000_000)


class Packet:
    """Independent protocol example. The external expectations are test oracles."""
    def __init__(self):
        self.blobs = {}
        self.capture = {"payloads":{}}
        scope = {"id":"fixture","model":"gate","route":"plain","profile":"text-generation-v1"}
        manifest = json.loads((Path(__file__).parent/"serving-release.cells.json").read_text())
        requirements = next(c["requirements"] for c in manifest["profiles"]["text-generation-v1"] if c["id"] == "drain")
        self.required = {"id":"fixture/drain","scope":scope,"scenario":"drain","requirements":requirements}
        requests = [
            {"id":"work","role":"inflight","model":"gate","wire":"chat_sse","path":"/v1/chat/completions",
             "payload":{"model":"gate","messages":[{"role":"user","content":"fixture"}],"stream":True},
             "prompt_tokens":{"min":1,"max":32},"completion_tokens":{"min":1,"max":64}},
            {"id":"new","role":"new_admission","model":"gate","wire":"chat_json","path":"/v1/chat/completions",
             "payload":{"model":"gate","messages":[{"role":"user","content":"fixture"}],"stream":False}}]
        self.program = {"schema":"memra-drain-program-v1","mode":"drain","cell_id":self.required["id"],"scope":scope,
            "server_identity":dict(OWNER),"endpoint":{"host":"127.0.0.1","port":12345},
            "identities":{"server_binary":"/fixture/server","source":"/fixture/source.json"},
            "http":{"connect_timeout":.5,"read_timeout":4,"wall_timeout":6,"max_body_bytes":16384},
            "timing":{"trigger_timeout_s":2,"overall_s":10,"drain_timeout_ns":3_000_000_000,"retry_after_s":3},
            "requests":requests}
        self.launch = {"argv":["/fixture/server"],"cwd":"/fixture","env":{},
                       "timeouts":{"startup":1,"overall":10,"drain":3,"kill":1},"output_path":"/fixture/process/output.log"}
        self.identities = {name:{"bytes":len(name),"sha256":hashlib.sha256(name.encode()).hexdigest()} for name in self.program["identities"]}
        life = drain()["lifecycle"]
        life.update(argv=self.launch["argv"],cwd=self.launch["cwd"],env_keys=[],
            env_sha256=hashlib.sha256(b"{}").hexdigest(),output_path=self.launch["output_path"],
            started_monotonic=.2, ready={"method":"owned_output","monotonic":.3,"owner":dict(PROCESS),"details":{"synthetic_fixture":True}})
        life["ownership_observations"][0]["first_seen_monotonic"] = .25
        before_life=copy.deepcopy(life)
        before_life.update(state="ready",stop=None,server_exit=None,signals=[])
        before_life["ownership_observations"][0]["last_seen_monotonic"] = .3
        self.capture.update(schema="memra-drain-capture-v1",state="captured",qualification=False,clock="monotonic_ns",
            started_ns=ns(.4),finished_ns=ns(4.5),required=self.put_obj(self.required),program=self.put_obj(self.program),
            identities_before=copy.deepcopy(self.identities),identities_after=copy.deepcopy(self.identities),
            process_before=self.put_obj({"started_ns":ns(.45),"finished_ns":ns(.46),"receipt":before_life,"observed_identity":dict(PROCESS)}),
            lifecycle=self.put_obj(life),listeners=[],invocations=[],observations=[],probes=[],http_events=[],
            errors=[],client_errors={},unattempted_ids=[],unobserved_invoked_ids=[],
            request_denominator=[{"id":r["id"],"role":r["role"],"status":"captured"} for r in requests])
        initial=self.listener(.50,.51)
        self.capture["listeners"].append(self.put_obj({"stage":"initial","id":None,"proof":initial}))
        prefix=b'data: {"model":"gate","choices":[{"index":0,"delta":{"content":"working "},"finish_reason":null}]}\n\n'
        terminal=b'data: {"model":"gate","choices":[{"index":0,"delta":{"content":"done"},"finish_reason":"stop"}],"usage":{"prompt_tokens":12,"completion_tokens":2,"total_tokens":14}}\n\ndata: [DONE]\n\n'
        specs=[("drain-probe-before",.90,1.10,200,encoded({"status":"ok","worker":{"generation":7}}),"/health"),
               ("work",1.20,4.20,200,prefix+terminal,"/v1/chat/completions"),
               ("drain-probe-draining",2.20,2.30,200,encoded({"status":"draining","worker":{"generation":7}}),"/health"),
               ("drain-probe-ready",2.40,2.50,503,encoded({"status":"not_ready"}),"/readyz"),
               ("new",2.60,2.70,503,encoded({"error":{"code":"draining","type":"server_error","message":"draining"}}),"/v1/chat/completions")]
        self.rows = {}
        body_events = {}
        for i,(name,start,end,status,body,path) in enumerate(specs):
            method="POST" if name in ("work","new") else "GET"
            first=len(prefix) if name=="work" else len(body)
            headers=[["Content-Length",str(len(body))]]+([["Retry-After","3"]] if status==503 else [])
            body_time=start+.08
            chunks=[{"end_offset":first,"observed_ns":ns(body_time)}]
            if first<len(body):chunks.append({"end_offset":len(body),"observed_ns":ns(end-.01)})
            row={"id":name,"method":method,"path":path,"server_identity":dict(OWNER),"started_ns":ns(start),"finished_ns":ns(end),
                 "status":status,"headers":headers,"body":body,"transport_error":None,"first_body_byte_ns":ns(body_time),"chunks":chunks}
            self.rows[name]=row
            ref=self.observation(row)
            if name.startswith("drain-probe-"):
                self.capture["probes"].append({"role":name.removeprefix("drain-probe-"),"observation":ref})
            else:self.capture["observations"].append(ref)
            request=next((r for r in requests if r["id"]==name),None)
            self.capture["invocations"].append(self.put_obj({"id":name,"method":method,"path":path,"started_ns":ns(start)-1,
                "body":self.put(encoded(request["payload"]) if request else b"")}))
            check=self.put_obj({"started_ns":ns(start+.02),"finished_ns":ns(start+.03),"identity":dict(PROCESS)})
            proof=self.listener(start+.04,start+.05)
            self.capture["listeners"].append(self.put_obj({"stage":"connected","id":name,"proof":proof}))
            event={"id":name,"event":"connected","observed_ns":ns(start+.01),"local":["127.0.0.1",20000+i],
                   "peer":["127.0.0.1",12345],"owner_check":check,"listener":proof}
            self.capture["http_events"].append(self.put_obj(event))
            self.capture["http_events"].append(self.put_obj({"id":name,"event":"headers","observed_ns":ns(start+.06),"status":status,"headers":headers}))
            body_events[name]=self.put_obj({"id":name,"event":"first_body","observed_ns":ns(body_time),"end_offset":first,"body":self.put(body[:first])})
            self.capture["http_events"].append(body_events[name])
        self.capture["trigger"]=self.put_obj({"started_ns":ns(1.15),"observed_ns":ns(1.5),"inflight_ids":["work"],"first_body_events":{"work":body_events["work"]}})
        self.capture["stop_call"]=self.put_obj({"started_ns":ns(1.9),"finished_ns":ns(4.15),"reason":"drain_cell","error":None})
        self.capture["stop_observation"]=self.put_obj({"started_ns":ns(2.06),"finished_ns":ns(2.07),"signal":life["signals"][0]})
        baseline=b"READY\n"
        log=baseline+b"[server] SIGTERM: draining (1 in flight, deadline 3s)\n[server] drain complete in 2.0s; exiting\n"
        self.capture["log_before"],self.capture["log"]=self.put(baseline),self.put(log)
        self.capture["log_identity"]=self.put_obj({"path":self.launch["output_path"],"device":1,"inode":2})
        self.capture["log_reads"]=[]
        for data,a,b in [(baseline,.60,.61),(log,4.30,4.31)]:
            self.capture["log_reads"].append(self.put_obj({"started_ns":ns(a),"finished_ns":ns(b),"device":1,"inode":2,
                "named_device":1,"named_inode":2,"named_after_device":1,"named_after_inode":2,"bytes":len(data),"complete_bytes":len(data),
                "size_before":len(data),"size_after":len(data),"sha256":hashlib.sha256(data).hexdigest()}))
        self.rederive()

    def put(self, data):
        digest=hashlib.sha256(data).hexdigest();name="blobs/"+digest
        self.blobs[name]=data;self.capture["payloads"][name]=digest
        return {"path":name,"sha256":digest}

    def put_obj(self, value):return self.put(encoded(value))
    def raw(self, ref):return self.blobs[ref["path"]]
    def obj(self, ref):return json.loads(self.raw(ref))
    def observation(self, row):return self.put_obj({**row,"body":self.put(row["body"])})

    def listener(self, start, end):
        return self.put_obj({"method":"listener_identity","owner":dict(PROCESS),"details":{
            "schema":"memra-linux-listener-v1","clock":"monotonic_ns","endpoint":dict(self.program["endpoint"]),
            "started_ns":ns(start),"finished_ns":ns(end),"boot_id":BOOT,"owner_before":dict(PROCESS),"owner_after":dict(PROCESS),
            "samples":[{"inodes":[123],"primary_fds":[9]},{"inodes":[123],"primary_fds":[9]}]}})

    def change(self, ref, mutate):
        value=self.obj(ref);mutate(value);return self.put_obj(value)

    def rederive(self):
        attempts=[self.rows[r["id"]] for r in self.program["requests"]]
        accounting=account_attempts(self.program["requests"],attempts)
        self.capture["accounting"]=self.put_obj(accounting)
        self.capture["facts"]=self.put_obj(evaluate_scenario(policy_config(self.program),accounting=accounting,
            attempts=attempts,health_samples=[self.rows["drain-probe-before"],self.rows["drain-probe-draining"]],
            status_samples=[self.rows["drain-probe-ready"]],lifecycle=self.obj(self.capture["lifecycle"])))

    def evaluate(self, **overrides):
        data=encoded(self.capture)
        args={"expected_capture_sha256":hashlib.sha256(data).hexdigest(),"expected_required":self.required,
              "expected_program":self.program,"expected_server":self.launch,"expected_identities":self.identities}
        return read_drain_capture(data,self.blobs.__getitem__,**{**args,**overrides})


class DrainEvidenceTests(unittest.TestCase):
    def test_complete_synthetic_bundle_preserves_unknown_tail_and_after_exit_reads(self):
        packet=Packet();before=copy.deepcopy(packet.capture)
        result=packet.evaluate()
        self.assertFalse(result["qualification"])
        self.assertEqual(result["facts"]["wire_counts"],{"clean_success":1,"refused":1})
        self.assertIn("work",result["facts"]["client_reads_finished_after_exit_observation"])
        work=next(r for r in result["facts"]["generation_observations"]["requests"] if r["id"]=="work")
        self.assertEqual(work["scope"],"unobserved_tail")
        self.assertIsNone(work["generation_affected"])
        self.assertIsNone(work["clean_latency_ns"])
        self.assertEqual(packet.capture,before)

    def test_completion_order_is_not_the_derived_program_order(self):
        packet=Packet()
        packet.capture["observations"].reverse()  # new refusal finishes before work
        before=copy.deepcopy(packet.capture)
        result=packet.evaluate()
        self.assertEqual([row["id"] for row in result["attempts"]],["work","new"])
        self.assertEqual([row["id"] for row in result["facts"]["generation_observations"]["requests"]],["work","new"])
        self.assertEqual(packet.capture,before)
        self.assertFalse(result["qualification"])

    def test_complete_first_prefix_cannot_count_as_inflight_even_with_late_client_finish(self):
        packet=Packet();packet.evaluate()
        body=packet.rows["work"]["body"]
        for i,ref in enumerate(packet.capture["http_events"]):
            event=packet.obj(ref)
            if event["id"]=="work" and event["event"]=="first_body":
                event.update(end_offset=len(body),body=packet.put(body))
                first=packet.put_obj(event);packet.capture["http_events"][i]=first
        packet.capture["trigger"]=packet.change(packet.capture["trigger"],
            lambda trigger:trigger["first_body_events"].update(work=first))
        packet.rows["work"]["chunks"]=[{"end_offset":len(body),
            "observed_ns":packet.rows["work"]["first_body_byte_ns"]}]
        packet.capture["observations"]=[packet.observation(packet.rows[packet.obj(ref)["id"]])
            for ref in packet.capture["observations"]]
        packet.rederive()
        before=copy.deepcopy(packet.capture)
        with self.assertRaisesRegex(ServingGateError,"prefix already contains a complete response"):
            packet.evaluate()
        self.assertEqual(packet.capture,before)

    def test_shared_inflight_prefix_rule_preserves_partial_and_rejects_typed_error(self):
        packet=Packet();request=packet.program["requests"][0]
        first=packet.obj(packet.obj(packet.capture["trigger"])["first_body_events"]["work"])
        require_inflight_prefix(request,packet.raw(first["body"]))
        with self.assertRaisesRegex(ServingGateError,"typed server error"):
            require_inflight_prefix(request,
                b'data: {"error":{"code":"engine_error","message":"failed"}}\n\n')

    def test_plan_denominators_types_unknown_fields_and_scope_refuse(self):
        for change in (lambda p:p["requests"].pop(),lambda p:p["requests"].append(copy.deepcopy(p["requests"][0])),
                       lambda p:p.update(skip=True),lambda p:p["requests"][0].update(role=[]),
                       lambda p:p["requests"][0].update(id="drain-probe-before"),
                       lambda p:p["timing"].update(drain_timeout_ns=True),
                       lambda p:p["http"].update(wall_timeout=float("nan")),
                       lambda p:p["requests"][0]["payload"].update(model="other")):
            packet=Packet();change(packet.program)
            with self.assertRaises((ServingGateError,ValueError)):validate_drain_program(packet.required,packet.program)

    def test_external_source_binary_program_owner_and_capture_digest_cannot_be_substituted(self):
        for field in ("server_binary","source"):
            packet=Packet()
            packet.capture["identities_before"][field]["sha256"]="0"*64
            packet.capture["identities_after"][field]["sha256"]="0"*64
            with self.assertRaisesRegex(ServingGateError,"identity differs"):packet.evaluate()
        packet=Packet();bad=copy.deepcopy(packet.program);bad["requests"][0]["payload"]["messages"][0]["content"]="different"
        with self.assertRaises(ServingGateError):packet.evaluate(expected_program=bad)
        packet=Packet();packet.capture["process_before"]=packet.change(packet.capture["process_before"],lambda r:r["observed_identity"].update(start_time="999"))
        with self.assertRaises(ServingGateError):packet.evaluate()
        with self.assertRaisesRegex(ServingGateError,"external digest"):Packet().evaluate(expected_capture_sha256="0"*64)
        packet=Packet();packet.capture["finished_ns"]=ns(11)
        with self.assertRaisesRegex(ServingGateError,"overall deadline"):packet.evaluate()

    def test_all_blobs_and_requests_probes_events_are_required(self):
        for field in ("observations","probes","http_events","invocations","listeners"):
            packet=Packet();packet.capture[field].pop()
            with self.subTest(field=field),self.assertRaises(ServingGateError):packet.evaluate()
        packet=Packet();ref=packet.put(b"diagnostic");packet.blobs[ref["path"]]=b"changed"
        with self.assertRaises(ServingGateError):packet.evaluate()
        packet=Packet();packet.capture["state"]="failed"
        with self.assertRaises(ServingGateError):packet.evaluate()

    def test_terminal_body_failure_and_bodycap_cannot_be_relabelled_by_summaries(self):
        for body in (b"data: {}\n\n",b"",b" "*16385):
            packet=Packet();row=packet.obj(packet.capture["observations"][0]);row["body"]=packet.put(body)
            packet.capture["observations"][0]=packet.put_obj(row)
            with self.assertRaises(ServingGateError):packet.evaluate()
        for role in (0,1,2):
            packet=Packet();ref=packet.capture["probes"][role]["observation"]
            packet.capture["probes"][role]["observation"]=packet.change(ref,lambda r:r.update(body=packet.put(b"{")))
            with self.assertRaises(ServingGateError):packet.evaluate()

    def test_post_exit_connection_or_probe_and_late_capture_refuse(self):
        for field,value in (("started_ns",ns(4.01)),("finished_ns",ns(5.1))):
            packet=Packet();ref=packet.capture["probes"][2]["observation"]
            packet.capture["probes"][2]["observation"]=packet.change(ref,lambda r:r.update({field:value}))
            with self.assertRaises(ServingGateError):packet.evaluate()
        packet=Packet();ref=next(r for r in packet.capture["http_events"] if packet.obj(r)["event"]=="connected")
        at=packet.capture["http_events"].index(ref)
        packet.capture["http_events"][at]=packet.change(ref,lambda r:r.update(observed_ns=ns(4.01)))
        with self.assertRaises(ServingGateError):packet.evaluate()

    def test_missing_sent_signal_escalation_retirement_and_late_cleanup_refuse(self):
        changes=[lambda r:r["signals"][0].update(result="already_exited"),
                 lambda r:r["signals"][0].pop("sent_monotonic"),
                 lambda r:r["signals"][0].update(result="error"),
                 lambda r:r["cleanup"].update(escalated=True),
                 lambda r:r["cleanup"].update(finished_monotonic=5.01),
                 lambda r:r["server_exit"].update(returncode=7)]
        for change in changes:
            packet=Packet();packet.capture["lifecycle"]=packet.change(packet.capture["lifecycle"],change)
            with self.assertRaises(ServingGateError):packet.evaluate()

    def test_self_consistent_headers_cannot_hide_short_transport_body(self):
        packet=Packet()
        original=packet.obj(packet.capture["observations"][0])
        original["headers"][0][1]=str(len(packet.raw(original["body"]))+1)
        packet.capture["observations"][0]=packet.put_obj(original)
        for i,ref in enumerate(packet.capture["http_events"]):
            event=packet.obj(ref)
            if event["id"]=="work" and event["event"]=="headers":
                event["headers"]=original["headers"];packet.capture["http_events"][i]=packet.put_obj(event)
        with self.assertRaisesRegex(ServingGateError,"Content-Length"):packet.evaluate()

    def test_no_successful_signal_and_post_exit_probe_refuse_at_their_actual_boundary(self):
        packet=Packet()
        def unsent(receipt):
            receipt["signals"][0]["result"]="already_exited"
            receipt["signals"][0].pop("sent_monotonic")
        packet.capture["lifecycle"]=packet.change(packet.capture["lifecycle"],unsent)
        with self.assertRaisesRegex(ServingGateError,"exactly one owned primary SIGTERM"):packet.evaluate()
        packet=Packet()
        ref=packet.capture["probes"][2]["observation"]
        packet.capture["probes"][2]["observation"]=packet.change(ref,lambda row:row.update(started_ns=ns(4.01),finished_ns=ns(4.2)))
        with self.assertRaisesRegex(ServingGateError,"at or after owned process exit"):packet.evaluate()

    def test_zero_inflight_log_missing_start_and_deadline_path_refuse(self):
        for replacement in (b"0 in flight",b"[server] drain deadline",b"missing start"):
            packet=Packet();raw=packet.raw(packet.capture["log"])
            if replacement==b"0 in flight":raw=raw.replace(b"1 in flight",replacement)
            elif replacement==b"missing start":raw=raw.replace(b"[server] SIGTERM:",replacement)
            else:raw+=replacement+b" (3s) hit with 1 in flight; exiting\n"
            packet.capture["log"]=packet.put(raw)
            with self.assertRaises(ServingGateError):packet.evaluate()


if __name__=='__main__':unittest.main()
