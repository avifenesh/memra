"""Content-addressed corruption tests; fixture ownership is synthetic, not native."""

import base64
import copy
import hashlib
import json
from pathlib import Path
import unittest

from serving_cancel_evidence import read_phase_capture
from serving_cancel import evaluate_cancel_wire
from serving_release import ServingGateError, account_attempts


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


class Packet:
    def __init__(self):
        fixture = json.loads((Path(__file__).parent / "fixtures/phase-cancel-capture.json").read_text())
        self.capture = fixture["capture"]
        self.expected = fixture["expected"]
        self.blobs = {name: base64.b64decode(data, validate=True) for name, data in fixture["blobs"].items()}

    def raw(self, ref):
        return self.blobs[ref["path"]]

    def obj(self, ref):
        return json.loads(self.raw(ref))

    def put(self, data):
        digest = hashlib.sha256(data).hexdigest()
        name = "blobs/" + digest
        self.blobs[name] = data
        self.capture["payloads"][name] = digest
        return {"path": name, "sha256": digest}

    def change(self, ref, mutate):
        value = self.obj(ref)
        mutate(value)
        return self.put(encoded(value))

    def root(self, key, mutate):
        self.capture[key] = self.change(self.capture[key], mutate)

    def row(self, key, index, mutate):
        self.capture[key][index] = self.change(self.capture[key][index], mutate)

    def rederive_wire(self):
        rows = {}
        for key in ("observations", "probes"):
            rows[key] = []
            for ref in self.capture[key]:
                value = self.obj(ref); value["body"] = self.raw(value["body"])
                rows[key].append(value)
        accounting = account_attempts(self.expected["program"]["requests"], rows["observations"])
        wire = evaluate_cancel_wire(self.expected["required"], self.expected["program"],
            attempts=rows["observations"], health_samples=rows["probes"],
            cancellation=self.obj(self.capture["trigger"])["cancellation"])
        self.capture["wire_accounting"] = self.put(encoded(accounting))
        self.capture["wire_facts"] = self.put(encoded(wire))

    def body_size(self, probes, size):
        key = "probes" if probes else "observations"
        index = 0 if probes else next(i for i, ref in enumerate(self.capture[key]) if self.obj(ref)["id"] == "peer")
        row = self.obj(self.capture[key][index]); original = self.raw(row["body"])
        body = original + b" " * (size - len(original)) if probes else b":" + b"x" * (size - len(original) - 3) + b"\n\n" + original
        row["body"] = self.put(body)
        row["headers"] = [[k, str(len(body)) if k.lower() == "content-length" else v] for k, v in row["headers"]]
        row["chunks"][-1]["end_offset"] = len(body)
        self.capture[key][index] = self.put(encoded(row))
        self.rederive_wire()

    def target_start(self, start):
        for i, ref in enumerate(self.capture["observations"]):
            row = self.obj(ref)
            if row["id"] != "target":
                continue
            row["started_ns"] = start
            for chunk in row["chunks"]:
                chunk["observed_ns"] = max(start + 1, chunk["observed_ns"])
            if row["first_body_byte_ns"] is not None:
                row["first_body_byte_ns"] = max(start + 1, row["first_body_byte_ns"])
            self.capture["observations"][i] = self.put(encoded(row))
        self.rederive_wire()

    def evaluate(self, **overrides):
        data = encoded(self.capture)
        args = dict(expected_capture_sha256=hashlib.sha256(data).hexdigest(),
                    expected_required=self.expected["required"], expected_program=self.expected["program"],
                    expected_server=self.expected["server"], expected_log_identity=self.expected["log_identity"])
        # Recompute the external index checksum after internal corruptions so
        # tests exercise semantic binding, not merely the outer checksum guard.
        return read_phase_capture(data, self.blobs.__getitem__, **{**args, **overrides})


class PhaseEvidenceTests(unittest.TestCase):
    def reject(self, packet, reason=None, **kwargs):
        with self.assertRaisesRegex(ServingGateError, reason or "."):
            packet.evaluate(**kwargs)

    def test_complete_fixture_replays_without_qualification_or_losing_denominator(self):
        p = Packet(); before = copy.deepcopy(p.capture)
        result = p.evaluate()
        self.assertFalse(result["qualification"])
        self.assertEqual(result["scope"], "phase_capture_consistency_only")
        self.assertEqual(len(result["attempts"]), 3)
        self.assertEqual(len(result["health_samples"]), 3)
        self.assertEqual(result["verified_payloads"], len(p.blobs))
        self.assertEqual(result["log"], p.raw(p.capture["log_final"]))
        self.assertEqual(p.capture, before)

    def test_concurrent_completion_order_does_not_become_dispatch_order(self):
        p = Packet(); p.capture["observations"][:2] = reversed(p.capture["observations"][:2])
        # The generation ledger retains capture order too. Keep its unchanged
        # request facts in the corresponding order; timestamps do not change.
        def reorder(value):
            value["generations"]["requests"][:2] = reversed(value["generations"]["requests"][:2])
        p.root("wire_facts", reorder)
        result = p.evaluate()
        self.assertEqual(len(result["attempts"]), 3)

    def test_external_index_and_every_blob_are_verified(self):
        self.reject(Packet(), "external digest", expected_capture_sha256="0" * 64)
        p = Packet(); ref = p.put(b"unreferenced diagnostic"); p.blobs[ref["path"]] = b"changed"
        self.reject(p, "payload hash")
        p = Packet(); p.capture["payloads"]["../../other"] = "0" * 64
        self.reject(p, "canonical")

    def test_capture_cannot_change_its_program_required_cell_or_log_owner(self):
        for key, mutate in [("program", lambda v: v["scope"].update(model="substituted")),
                            ("required", lambda v: v.update(id="other/cancel_decode")),
                            ("log_descriptor", lambda v: v.update(inode=v["inode"] + 1))]:
            with self.subTest(key=key):
                p = Packet(); p.root(key, mutate); self.reject(p)

    def test_failed_capture_and_partial_or_extra_attempts_cannot_supply_one_cell(self):
        for mutate in [lambda c: c.update(state="failed"), lambda c: c.update(qualification=True),
                       lambda c: c["observations"].pop(), lambda c: c["observations"].append(c["observations"][0]),
                       lambda c: c.update(captured_attempts=2), lambda c: c.update(unattempted_ids=["target"]),
                       lambda c: c["request_denominator"][0].update(status="invoked_without_observation")]:
            p = Packet(); mutate(p.capture); self.reject(p)

    def test_request_invocations_must_bind_actual_unique_attempts(self):
        for changes in ({"id": "foreign"}, {"id": []}, {"path": "/other"}, {"started_ns": 0},
                        {"stage": "after_capture_request"}):
            p = Packet(); p.row("request_invocations", 0, lambda v: v.update(changes)); self.reject(p)

    def test_server_launch_timeouts_and_controller_lineage_are_bound(self):
        for mutate in [lambda v: v["receipt"].update(argv=["/unrelated"]),
                       lambda v: v["receipt"].update(env_sha256="0" * 64),
                       lambda v: v["receipt"]["timeouts"].update(overall=999),
                       lambda v: v["receipt"]["supervisor"].update(ppid=999),
                       lambda v: v["receipt"].update(output_path="/different/log"),
                       lambda v: v["receipt"].update(state="finished")]:
            p = Packet(); p.row("process_observations", 0, mutate); self.reject(p)

    def test_nonlinux_dead_or_reborn_process_evidence_refuses(self):
        for changes in ({"identity_source": "ps_lstart_seconds"}, {"state": "Z"},
                        {"state": "invented"}, {"start_time": "90002"}, {"pgid": 1.0}):
            p = Packet(); p.row("process_observations", 1, lambda v: v["observed_identity"].update(changes))
            self.reject(p, "process evidence|ownership drift")

    def test_read_hash_inode_clock_and_watermark_cannot_be_summary_only(self):
        for changes in ({"sha256": "0" * 64}, {"inode": 1}, {"complete_bytes": 1},
                        {"started_ns": 0}, {"bytes": True}, {"size_before": 1}):
            p = Packet(); p.row("log_observations", 0, lambda v: v.update(changes)); self.reject(p)

    def test_append_chunk_sequence_and_per_read_growth_are_bound(self):
        p = Packet(); p.capture["log_chunks"][0]["offset"] = 1; self.reject(p)
        p = Packet(); p.capture["log_chunks"][0]["bytes"] += 1; self.reject(p)
        p = Packet(); pieces = p.capture["log_chunks"]; a, b = pieces[:2]
        combined = p.put(p.raw(a["raw"]) + p.raw(b["raw"]))
        pieces[:2] = [{"offset": 0, "bytes": a["bytes"] + b["bytes"], "raw": combined}]
        self.reject(p, "read watermark")

    def test_final_log_and_trigger_prefix_cannot_be_substituted(self):
        p = Packet(); p.capture["log_final"] = p.put(p.raw(p.capture["log_final"])[:-1]); self.reject(p)
        p = Packet(); p.root("trigger", lambda v: v.update(log_prefix=p.put(b"ordinary log\n")))
        self.reject(p, "prefix differs")

    def test_observed_stat_shrink_refuses_even_when_all_observed_bytes_are_retained(self):
        p = Packet(); trigger = p.obj(p.capture["trigger"])
        reads = [p.obj(ref) for ref in p.capture["log_observations"]]
        index = reads.index(trigger["log_read"])
        reads[index]["size_after"] = reads[index + 1]["bytes"] + 1
        p.capture["log_observations"][index] = p.put(encoded(reads[index]))
        trigger["log_read"] = reads[index]; p.capture["trigger"] = p.put(encoded(trigger))
        self.reject(p, "byte/size/hash")

    def test_trigger_must_match_raw_selected_phase_and_bounded_host_interval(self):
        for mutate in [lambda v: v["observed_record"]["record"].update(event="queued", phase="queued"),
                       lambda v: v.update(clock="trace_elapsed_ns"),
                       lambda v: v["cancellation"].update(set_before_ns=0),
                       lambda v: v["cancellation"].update(invoke_started_ns=0)]:
            p = Packet(); p.root("trigger", mutate); self.reject(p)

    def test_recorded_trace_wire_or_terminal_summary_does_not_override_raw_data(self):
        p = Packet(); p.root("trace_facts", lambda v: v["target"].update(request_id="other")); self.reject(p)
        p = Packet(); p.root("wire_facts", lambda v: v.update(qualification=True)); self.reject(p)
        p = Packet(); p.root("target_end_observation", lambda v: v["facts"].update(scope="native_pass")); self.reject(p)

    def test_probe_denominator_and_recovery_sequence_cannot_be_selected_after_capture(self):
        p = Packet(); p.capture["probes"].pop(); self.reject(p)
        p = Packet(); p.capture["probes"].reverse(); self.reject(p)
        p = Packet(); p.row("probes", 1, lambda v: v.update(path="/metrics")); self.reject(p)
        p = Packet(); p.row("probes", 1, lambda v: v.update(started_ns=0, finished_ns=1)); self.reject(p)

    def test_listener_requires_real_schema_owner_and_capture_brackets(self):
        for mutate in [lambda v: v["proof"].update(details={"synthetic_fixture_only": True}),
                       lambda v: v["proof"]["details"]["samples"][0].update(inodes=[1, 2]),
                       lambda v: v["proof"]["details"].update(started_ns=0, finished_ns=1)]:
            p = Packet(); p.row("listeners", 0, mutate); self.reject(p)

    def test_outer_capture_chronology_cannot_follow_process_and_request_records(self):
        p = Packet(); p.capture["started_ns"] = p.capture["finished_ns"] - 1; self.reject(p)
        p = Packet(); p.capture["finished_ns"] = p.capture["started_ns"] + 1; self.reject(p)
        p = Packet(); p.capture["finished_ns"] += int(p.expected["program"]["timing"]["overall_s"] * 1e9)
        self.reject(p, "deadline")

    def test_payload_substitution_cannot_be_hidden_by_resealing_its_reference(self):
        p = Packet()
        for i, ref in enumerate(p.capture["observations"]):
            row = p.obj(ref)
            if row["id"] == "peer":
                p.row("observations", i, lambda v: v.update(body=p.put(b'{"error":{"message":"broken"}}')))
                break
        self.reject(p)

    def test_trusted_body_cap_applies_to_requests_and_probes_including_exact_boundary(self):
        for probes in (False, True):
            with self.subTest(probes=probes):
                p = Packet(); expected = copy.deepcopy(p.expected)
                p.body_size(probes, p.expected["program"]["http"]["max_body_bytes"])
                self.assertFalse(p.evaluate()["qualification"])
                self.assertEqual(p.expected, expected)
                p = Packet()
                p.body_size(probes, p.expected["program"]["http"]["max_body_bytes"] + 1)
                self.reject(p, "trusted byte cap")

    def test_phase_read_cannot_predate_target_but_may_overlap_its_start(self):
        p = Packet(); read = p.obj(p.capture["trigger"])["log_read"]
        p.target_start(read["finished_ns"] + 1)
        self.reject(p, "phase observation predates")
        p = Packet(); p.target_start(read["started_ns"] + 1)
        self.assertFalse(p.evaluate()["qualification"])

    def test_rereading_a_phase_after_start_does_not_hide_earlier_trace_observation(self):
        p = Packet(); trigger = p.obj(p.capture["trigger"])
        reads = [p.obj(ref) for ref in p.capture["log_observations"]]
        index = reads.index(trigger["log_read"])
        repeated = copy.deepcopy(reads[index])
        repeated.update(started_ns=repeated["finished_ns"] + 10, finished_ns=repeated["finished_ns"] + 20)
        p.capture["log_observations"].insert(index + 1, p.put(encoded(repeated)))
        trigger["log_read"] = repeated; p.capture["trigger"] = p.put(encoded(trigger))
        p.target_start(reads[index]["finished_ns"] + 5)
        self.reject(p, "target trace was observed before")

    def test_missing_or_invalid_request_path_has_a_typed_refusal(self):
        p = Packet(); p.row("observations", 0, lambda row: row.pop("path"))
        self.reject(p, "method/path")
        p = Packet(); p.row("observations", 0, lambda row: row.update(path=[]))
        self.reject(p, "method/path")


if __name__ == "__main__":
    unittest.main()
