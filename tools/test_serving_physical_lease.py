"""CPU simulations of closed receipts; no actual GPU or lock evidence."""

import copy
import unittest

import release_qualification as q


A = "GPU-00000000-0000-0000-0000-000000000001"
B = "GPU-00000000-0000-0000-0000-000000000002"


def fixture(indices=(0,), ids=(A,)):
    devices = [{"index": i, "uuid": uid, "name": "NVIDIA RTX PRO 6000 Blackwell"}
               for i, uid in zip(indices, ids)]
    lease = {"requested_uuids": list(ids), "lock_order": sorted(ids),
             "lock_files": {uid: f"/tmp/memra-gpu-locks/{uid}.lock" for uid in ids},
             "wrapper_pid": 10, "child_pid": 11, "state": "finished", "exit_code": 0,
             "child_exit_code": 0, "timed_out": False, "interrupted_signal": None,
             "lingering_compute": [], "started_unix": 1.0, "finished_unix": 4.0, "devices": devices}
    run = {"lease_owner": {k: copy.deepcopy(lease[k]) for k in
                           ("wrapper_pid", "child_pid", "requested_uuids")},
           "started_unix": 2.0, "finished_unix": 3.0,
           "hardware": {"devices": [{**d, "index": str(d["index"]), "compute_cap": "12.0",
                                      "driver_version": "CPU fixture"} for d in devices],
                        "headroom_query": {"nvml_index": indices[0], "uuid": ids[0]}},
           "numeric_environment": {"CUDA_VISIBLE_DEVICES": q.digest(",".join(ids).encode())}}
    return lease, run


class PhysicalLeaseTests(unittest.TestCase):
    def test_selected_nonzero_card_is_valid_without_forging_gpu0(self):
        lease, run = fixture((2,))
        before = copy.deepcopy((lease, run))
        q.validate_physical_lease(lease, run)
        self.assertEqual((lease, run), before)
        with self.assertRaisesRegex(q.GateError, "NVML GPU0"):
            q.validate_lease(lease, run)

    def test_cuda_order_can_differ_from_sorted_lock_acquisition_order(self):
        lease, run = fixture((3, 1), (B, A))
        q.validate_physical_lease(lease, run)
        with self.assertRaisesRegex(q.GateError, "one physical card"):
            q.validate_lease(lease, run)
        run["numeric_environment"]["CUDA_VISIBLE_DEVICES"] = q.digest((A + "," + B).encode())
        with self.assertRaisesRegex(q.GateError, "visibility"):
            q.validate_physical_lease(lease, run)

    def test_generic_profile_keeps_its_hardware_constraint(self):
        lease, run = fixture(); q.validate_lease(lease, run)
        lease["devices"][0]["name"] = run["hardware"]["devices"][0]["name"] = "CPU-simulated other GPU"
        q.validate_physical_lease(lease, run)
        with self.assertRaisesRegex(q.GateError, "wrong release rig"):
            q.validate_lease(lease, run)

    def test_incomplete_or_failed_lease_never_passes(self):
        for key, value in (("state", "running"), ("exit_code", 1), ("child_exit_code", 1),
                           ("timed_out", True), ("interrupted_signal", 15), ("lingering_compute", [123])):
            lease, run = fixture(); lease[key] = value
            with self.subTest(key=key), self.assertRaises(q.GateError):
                q.validate_physical_lease(lease, run)

    def test_wrong_lock_set_and_duplicate_devices_refuse(self):
        mutations = [lambda l: l.update(lock_order=[B, A]),
                     lambda l: l["lock_files"].update({A: "/tmp/wrong.lock"}),
                     lambda l: l["requested_uuids"].append(A),
                     lambda l: l.update(requested_uuids=[]),
                     lambda l: l["devices"][1].update(index=1)]
        for mutate in mutations:
            lease, run = fixture((1, 2), (A, B)); mutate(lease)
            with self.assertRaises(q.GateError): q.validate_physical_lease(lease, run)

    def test_wrong_process_device_or_visibility_identity_refuses(self):
        mutations = [lambda l, r: r["lease_owner"].update(child_pid=12),
                     lambda l, r: l.update(wrapper_pid=11),
                     lambda l, r: l.update(wrapper_pid=True),
                     lambda l, r: r["hardware"]["devices"][0].update(uuid=B),
                     lambda l, r: r["hardware"]["devices"][0].update(index="1"),
                     lambda l, r: r["hardware"]["devices"][0].update(name="different"),
                     lambda l, r: r["numeric_environment"].clear()]
        for mutate in mutations:
            lease, run = fixture(); mutate(lease, run)
            with self.assertRaises(q.GateError): q.validate_physical_lease(lease, run)

    def test_times_are_finite_numeric_and_enclosed_by_the_lease(self):
        for target, key, value in (("lease", "started_unix", False), ("lease", "started_unix", -1),
                                   ("lease", "started_unix", float("-inf")),
                                   ("lease", "finished_unix", float("inf")),
                                   ("run", "started_unix", float("nan")),
                                   ("run", "started_unix", 0.5), ("run", "finished_unix", 5),
                                   ("run", "finished_unix", 2)):
            lease, run = fixture(); (lease if target == "lease" else run)[key] = value
            with self.subTest(target=target, key=key, value=value), self.assertRaises(q.GateError):
                q.validate_physical_lease(lease, run)

    def test_boolean_exit_or_float_pid_is_not_an_integer_wait_status_or_identity(self):
        for target, key, value in (("lease", "exit_code", False), ("lease", "child_exit_code", False),
                                   ("owner", "wrapper_pid", 10.0), ("owner", "child_pid", 11.0)):
            lease, run = fixture()
            (lease if target == "lease" else run["lease_owner"])[key] = value
            with self.subTest(target=target, key=key), self.assertRaises(q.GateError):
                q.validate_physical_lease(lease, run)


if __name__ == "__main__":
    unittest.main()
