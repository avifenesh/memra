"""Real Linux owned-process/HTTP capture replay, with synthetic phase records."""
import copy
import hashlib
import json
import os
from pathlib import Path
import sys

root = Path(__file__).resolve().parent
sys.path.insert(0, str(root / "tools"))
import test_serving_cancel_phase as fixture
from serving_cancel_evidence import read_phase_capture

out = Path(sys.argv[1])
out.mkdir(exist_ok=False)
os.environ["MEMRA_PHASE_TEST_EVIDENCE_DIR"] = str(out)
original = fixture.collect_phase_cancel_cell
results = []


def capture(required, program, *, server, output):
    stat = server.output_path.stat()
    expected = {"required": copy.deepcopy(required), "program": copy.deepcopy(program),
        "server": {"argv": list(server._config["argv"]), "env": dict(server._config["env"]),
                   "cwd": server._config["cwd"], "timeouts": dict(server._config["timeouts"]),
                   "output_path": str(server.output_path)},
        "log_identity": {"path": str(server.output_path), "device": stat.st_dev,
                         "inode": stat.st_ino, "controller_pid": os.getpid()}}
    # Captured from the actual call and launch configuration before collection;
    # never reconstructed from capture.json's claims.
    (server.directory.parent / "external-expectations.json").write_text(json.dumps(expected, indent=2) + "\n")
    result = original(required, program, server=server, output=output)
    if result["state"] != "captured":
        raise RuntimeError("synthetic phase fixture failed: " + str(result["errors"]))
    raw = (output / "capture.json").read_bytes()
    decoded = read_phase_capture(raw, lambda name: (output / name).read_bytes(),
        expected_capture_sha256=hashlib.sha256(raw).hexdigest(),
        expected_required=expected["required"], expected_program=expected["program"],
        expected_server=expected["server"], expected_log_identity=expected["log_identity"])
    if decoded["qualification"] is not False or len(decoded["attempts"]) != len(program["requests"]):
        raise RuntimeError("reader changed qualification or denominator")
    summary = {"scenario": required["scenario"], "capture_sha256": hashlib.sha256(raw).hexdigest(),
               "requests": len(decoded["attempts"]), "health_samples": len(decoded["health_samples"]),
               "payloads_verified": decoded["verified_payloads"], "qualification": False,
               "scope": "Real Linux process/listener/socket IO with synthetic lifecycle producer; no model/GPU"}
    (server.directory.parent / "reader-result.json").write_text(json.dumps(summary, indent=2) + "\n")
    results.append(summary)
    return result


fixture.collect_phase_cancel_cell = capture
case = fixture.PhaseProcessTests()
for scenario in ("cancel_queued", "cancel_prime", "cancel_decode"):
    case.run_case(scenario, real_listener=True)
(out / "results.json").write_text(json.dumps(results, indent=2) + "\n")
print(json.dumps(results))
