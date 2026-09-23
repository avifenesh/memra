"""Actual POSIX HTTP/process drain controls. Linux proof is explicit, never faked."""

import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import serving_http
from serving_capture import file_identity
from serving_drain_capture import collect_drain_cell, validate_drain_program
from serving_process import OwnedServer, ReadinessEvidence
from serving_release import ServingGateError


def program_for(server, port, source):
    scope = {"id":"fixture","model":"gate","route":"plain","profile":"text-generation-v1"}
    manifest = json.loads((Path(__file__).parent/"serving-release.cells.json").read_text())
    declaration = next(c for c in manifest["profiles"]["text-generation-v1"] if c["id"] == "drain")
    required = {"id":"fixture/drain","scope":scope,"scenario":"drain","requirements":declaration["requirements"]}
    receipt = server.receipt(); owner = server.identity
    identity = {"pid":owner.pid,"start_identity":receipt.get("boot_id",owner.identity_source)+":"+owner.start_time}
    requests = []
    for name,role in (("work","inflight"),("new","new_admission")):
        item = {"id":name,"role":role,"model":"gate","wire":"chat_sse" if role=="inflight" else "chat_json",
                "path":"/v1/chat/completions","payload":{"model":"gate","messages":[{"role":"user","content":"fixture"}],
                    "stream":role=="inflight","max_tokens":64,"temperature":0}}
        if role == "inflight":item.update(prompt_tokens={"min":1,"max":32},completion_tokens={"min":1,"max":64})
        requests.append(item)
    return required, {"schema":"memra-drain-program-v1","mode":"drain","cell_id":required["id"],"scope":scope,
        "server_identity":identity,"endpoint":{"host":"127.0.0.1","port":port},
        "identities":{"server_binary":sys.executable,"source":str(source)},
        "http":{"connect_timeout":.5,"read_timeout":6,"wall_timeout":8,"max_body_bytes":16384},
        "timing":{"trigger_timeout_s":3,"overall_s":12,"drain_timeout_ns":3_000_000_000,"retry_after_s":3},
        "requests":requests}


class DrainProcessTests(unittest.TestCase):
    def run_case(self, mode="normal", buffered=False, multiple=False):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary).resolve()
            script=Path(__file__).parent/"fixtures/drain_http_server.py"
            with socket.socket() as sock:sock.bind(("127.0.0.1",0));port=sock.getsockname()[1]
            server=OwnedServer(argv=[sys.executable,str(script),str(port),mode],env={},cwd=str(root),
                evidence_dir=str(root/"process"),startup_timeout=4,overall_timeout=15,drain_timeout=3,kill_timeout=1)
            output=root/"capture"
            try:
                server.start();until=time.monotonic()+3
                while time.monotonic()<until and (not server.output_path.exists() or "READY" not in server.output_path.read_text()):time.sleep(.01)
                self.assertIn("READY",server.output_path.read_text())
                server.mark_ready(ReadinessEvidence(server.identity,"owned_output",{"CPU_fixture":True}))
                required,program=program_for(server,port,script.resolve())
                if multiple:
                    second=copy.deepcopy(program["requests"][0]);second["id"]="work-2"
                    program["requests"].insert(1,second)
                original=serving_http._StrictHTTPResponse.read1
                delayed=[]
                def read(response,*args,**kwargs):
                    data=original(response,*args,**kwargs)
                    if buffered and b"[DONE]" in data and response.getheader("X-Fixture-Role")=="inflight":
                        until=time.monotonic()+2
                        while time.monotonic()<until and server.receipt().get("server_exit") is None:time.sleep(.005)
                        if server.receipt().get("server_exit") is None:raise AssertionError("fixture did not exit before buffered client finish")
                        delayed.append(time.monotonic_ns())
                    return data
                # Darwin has no procfs listener proof. The fake is deliberately
                # labelled and does NOT alter policy or manufacture a Linux receipt:
                # portable IO completes but the final Linux drain predicate refuses.
                from contextlib import ExitStack
                with ExitStack() as stack:
                    if sys.platform != "linux":
                        stack.enter_context(patch("serving_listener.prove_listener",return_value=
                            ReadinessEvidence(server.identity,"listener_identity",{"CPU_fixture_only":True})))
                    if buffered:stack.enter_context(patch.object(serving_http._StrictHTTPResponse,"read1",read))
                    result=collect_drain_cell(required,program,server=server,output=output)
                raw=(output/"capture.json").read_bytes();index=json.loads(raw)
                for name,digest in index["payloads"].items():self.assertEqual(hashlib.sha256((output/name).read_bytes()).hexdigest(),digest)
                def obj(ref):return json.loads((output/ref["path"]).read_bytes())
                attempts=[]
                for ref in result["observations"]:
                    row=obj(ref);row["body"]=(output/row["body"]["path"]).read_bytes();attempts.append(row)
                probes=[]
                for row in result["probes"]:
                    if row["observation"] is not None:
                        value=obj(row["observation"]);value["body"]=(output/value["body"]["path"]).read_bytes();probes.append(value)
                receipt=obj(result["lifecycle"])
                self.assertFalse(result["qualification"])
                self.assertEqual(len(result["request_denominator"]),3 if multiple else 2)
                self.assertFalse(any(t.name.startswith(("drain-http-","drain-owned-close","serving-http-")) for t in threading.enumerate()))
                time.sleep(.06);self.assertEqual((output/"capture.json").read_bytes(),raw)
                if buffered:self.assertTrue(delayed)
                return result,attempts,probes,receipt
            finally:
                if server.receipt().get("state") != "finished":server.close(reason="fixture_cleanup")
                evidence=os.environ.get("MEMRA_DRAIN_TEST_EVIDENCE_DIR")
                if evidence:
                    destination=Path(tempfile.mkdtemp(prefix=mode+"-",dir=evidence))
                    shutil.copytree(root,destination/"fixture")

    def test_real_term_open_stream_probes_refusal_and_exit(self):
        result,attempts,probes,receipt=self.run_case()
        self.assertEqual({r["id"] for r in attempts},{"work","new"})
        self.assertEqual([r["status"] for r in probes],[200,200,503])
        self.assertEqual(receipt["server_exit"]["returncode"],0)
        self.assertTrue(receipt["cleanup"]["complete"])
        self.assertFalse(receipt["cleanup"]["escalated"])
        work=next(r for r in attempts if r["id"]=="work")
        self.assertIn(b"[DONE]",work["body"])
        self.assertLess(work["started_ns"],int(receipt["signals"][0]["monotonic"]*1e9))
        self.assertGreater(work["finished_ns"],int(receipt["signals"][0]["sent_monotonic"]*1e9))
        if sys.platform == "linux":self.assertEqual(result["state"],"captured",result["errors"])
        else:
            self.assertEqual(result["state"],"failed")
            self.assertTrue(any("Linux birth" in e for e in result["errors"]),result["errors"])

    def test_real_buffered_body_capture_finishes_after_process_exit(self):
        result,attempts,_,receipt=self.run_case(buffered=True)
        work=next(r for r in attempts if r["id"]=="work")
        self.assertGreater(work["finished_ns"],int(receipt["server_exit"]["observed_monotonic"]*1e9))
        self.assertIsNone(work["transport_error"])
        self.assertIn(b"[DONE]",work["body"])
        if sys.platform == "linux":self.assertEqual(result["state"],"captured",result["errors"])

    def test_every_real_inflight_stream_is_completed_and_retained(self):
        result,attempts,_,receipt=self.run_case(multiple=True)
        self.assertEqual({r["id"] for r in attempts},{"work","work-2","new"})
        for row in attempts:
            if row["id"].startswith("work"):
                self.assertIn(b"[DONE]",row["body"])
                self.assertIsNone(row["transport_error"])
        self.assertEqual(receipt["server_exit"]["returncode"],0)
        self.assertEqual(result["unattempted_ids"],[])
        if sys.platform == "linux":self.assertEqual(result["state"],"captured",result["errors"])

    def test_actual_failed_rows_do_not_disappear(self):
        for mode in ("truncated_stream","typed_error","malformed_ready","framing_failure"):
            with self.subTest(mode=mode):
                result,attempts,probes,receipt=self.run_case(mode)
                self.assertEqual(result["state"],"failed")
                self.assertEqual({r["id"] for r in attempts},{"work","new"})
                self.assertEqual(len(probes),3)
                self.assertEqual(result["unattempted_ids"],[])
                self.assertEqual(result["unobserved_invoked_ids"],[])

    def test_stuck_drain_escalation_is_failure_with_original_exit(self):
        result,attempts,_,receipt=self.run_case("stuck")
        self.assertEqual(result["state"],"failed")
        self.assertTrue(receipt["cleanup"]["escalated"])
        self.assertEqual(receipt["server_exit"]["returncode"],-9)
        self.assertEqual({r["id"] for r in attempts},{"work","new"})

    def test_fast_complete_cannot_be_used_as_an_inflight_drain(self):
        result,attempts,_,_=self.run_case("fast_complete")
        self.assertEqual(result["state"],"failed")
        self.assertIn("new",result["unattempted_ids"])
        self.assertEqual([r["id"] for r in attempts],["work"])

    @unittest.skipUnless(sys.platform=="linux","full owned listener/reaping replay requires actual Linux procfs")
    def test_full_linux_capture_immutable_replay(self):
        result,*_=self.run_case()
        self.assertEqual(result["state"],"captured",result["errors"])


if __name__ == '__main__':unittest.main()
