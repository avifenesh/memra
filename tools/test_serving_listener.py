"""Procfs fixtures plus optional real Linux loopback sockets; no remote/GPU work."""

from dataclasses import asdict, replace
import ipaddress
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import serving_listener as listener
from serving_process import OwnedServer, ProcessIdentity, ReadinessEvidence


HEADER = "  sl  local_address rem_address   st tx_queue rx_queue tr tm->when retrnsmt uid timeout inode\n"
BOOT = "f85f4631-c8e4-44f1-b5af-22a2ee64f9c3"


def encoded_ip(address):
    packed = ipaddress.ip_address(address).packed
    return "".join(f"{int.from_bytes(packed[i:i + 4], sys.byteorder):08X}"
                   for i in range(0, len(packed), 4))


def row(address="127.0.0.1", port=18001, inode=777, state="0A", extra=""):
    address_hex = encoded_ip(address)
    return (f"  0: {address_hex}:{port:04X} {'0' * len(address_hex)}:0000 {state} "
            f"00000000:00000000 00:00000000 00000000 1000 0 {inode} 1 0000000000000000 {extra}\n")


class ProcFixture:
    def __init__(self, root):
        self.root = root
        root.mkdir()
        self.owner = ProcessIdentity(41, 7, 41, "12345", "linux_proc_start_ticks", "S")
        self.process(7, ppid=1, pgid=7, start="100")
        self.process(41, ppid=7, pgid=41, start="12345")
        self.process(42, ppid=7, pgid=42, start="23456")
        (root / "self").symlink_to("7")
        (root / "thread-self").symlink_to("7")
        self.namespace_file = root / "namespace"
        self.namespace_file.touch()
        for pid in (7, 41, 42):
            self.namespace(pid, self.namespace_file)
        boot = root / "sys/kernel/random/boot_id"
        boot.parent.mkdir(parents=True)
        boot.write_text(BOOT + "\n")
        (root / "7/mountinfo").write_text(f"25 0 0:5 / {root} rw,nosuid - proc proc rw\n")
        self.tables([row()], [])
        self.fd(41, 3, "socket:[777]")
        self.fd(42, 9, "unrelated-private-target")

    def process(self, pid, *, ppid, pgid, start, state="S"):
        directory = self.root / str(pid)
        (directory / "fd").mkdir(parents=True, exist_ok=True)
        fields = [state, str(ppid), str(pgid), str(pgid)] + ["0"] * 15 + [start]
        (directory / "stat").write_text(f"{pid} (name with ) parens) " + " ".join(fields) + "\n")

    def namespace(self, pid, file):
        directory = self.root / str(pid) / "ns"
        directory.mkdir(exist_ok=True)
        name = f"net:[{file.stat().st_ino}]"
        target = directory / name
        if not target.exists():
            os.link(file, target)
        (directory / "net").unlink(missing_ok=True)
        (directory / "net").symlink_to(name)

    def fd(self, pid, number, target):
        path = self.root / str(pid) / "fd" / str(number)
        path.unlink(missing_ok=True)
        path.symlink_to(target)

    def tables(self, tcp, tcp6):
        directory = self.root / "41/net"
        directory.mkdir(exist_ok=True)
        (directory / "tcp").write_text(HEADER + "".join(tcp))
        (directory / "tcp6").write_text(HEADER + "".join(tcp6))

    def prove(self, host="127.0.0.1", port=18001, view=None):
        return listener._prove_listener(view or listener._ProcFS(self.root, 3),
                                        self.owner, ipaddress.ip_address(host), port)


class ListenerFixtureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="serving-listener-fixture-")
        self.addCleanup(self.temp.cleanup)
        self.fixture = ProcFixture(Path(self.temp.name) / "proc")

    def test_ipv4_evidence_is_selected_raw_and_contains_no_unrelated_rows(self):
        selected = row()
        self.fixture.tables([selected, row("10.1.2.3", 3333, 999, extra="PRIVATE_OTHER_ROW")], [])
        self.fixture.fd(41, 8, "socket:[777]")  # dup() in primary is still one listener
        proof = self.fixture.prove()
        self.assertIsInstance(proof, ReadinessEvidence)
        self.assertEqual(proof.method, "listener_identity")
        self.assertEqual(proof.owner.key, self.fixture.owner.key)
        self.assertEqual(proof.details["boot_id"], BOOT)
        self.assertLessEqual(proof.details["started_ns"], proof.details["finished_ns"])
        self.assertEqual(len(proof.details["samples"]), 2)
        for sample in proof.details["samples"]:
            self.assertEqual(sample["selected_rows"][0]["raw"], selected.rstrip("\n"))
            self.assertEqual(sample["inodes"], [777])
            self.assertEqual(sample["primary_fds"], [3, 8])
        serialized = json.dumps(asdict(proof))
        self.assertNotIn("PRIVATE_OTHER_ROW", serialized)
        self.assertNotIn("unrelated-private-target", serialized)

    def test_ipv6_mapped_and_definite_wildcard_bindings(self):
        for bound, query, family in [("::1", "::1", "tcp6"), ("::", "::1", "tcp6"),
                                     ("::ffff:127.0.0.1", "127.0.0.1", "tcp6"),
                                     ("0.0.0.0", "127.0.0.1", "tcp"),
                                     ("127.0.0.1", "::ffff:127.0.0.1", "tcp")]:
            with self.subTest(bound=bound, query=query):
                self.fixture.tables([row(bound)] if family == "tcp" else [],
                                    [row(bound)] if family == "tcp6" else [])
                self.assertEqual(self.fixture.prove(query).details["samples"][0]["inodes"], [777])

    def test_fixed_kernel_hex_layout_and_tick_zero_task(self):
        # Literal kernel encodings, independent of the fixture encoder above.
        for byteorder, v4, v6 in [
            ("little", "0100007F", "B80D0120000000004433221188776655"),
            ("big", "7F000001", "20010DB8000000001122334455667788"),
        ]:
            with self.subTest(byteorder=byteorder), patch.object(listener.sys, "byteorder", byteorder):
                self.assertEqual(str(listener._decode_address(v4, "tcp")), "127.0.0.1")
                self.assertEqual(str(listener._decode_address(v6, "tcp6")),
                                 "2001:db8::1122:3344:5566:7788")
        self.fixture.process(2, ppid=0, pgid=0, start="0", state="I")
        proof = self.fixture.prove()
        self.assertEqual(proof.details["samples"][0]["processes_checked"], 4)

    def test_foreign_reuseport_and_shared_inode_are_refused(self):
        self.fixture.fd(41, 3, "socket:[999]")
        with self.assertRaisesRegex(listener.ListenerOwnershipError, "owned primary"):
            self.fixture.prove()
        self.fixture.fd(41, 3, "socket:[777]")
        self.fixture.tables([row(), row(inode=888)], [])
        with self.assertRaisesRegex(listener.ListenerOwnershipError, "shared"):
            self.fixture.prove()
        self.fixture.tables([row()], [])
        self.fixture.fd(42, 10, "socket:[777]")
        # A holder can have moved namespaces after inheriting the socket.
        other = self.fixture.root / "other-net"
        other.touch()
        self.fixture.namespace(42, other)
        with self.assertRaisesRegex(listener.ListenerOwnershipError, "shared with another"):
            self.fixture.prove()

    def test_wildcard_competitor_and_unknown_ipv6_v6only_refuse(self):
        for tcp, tcp6 in [([row(), row("0.0.0.0", inode=888)], []),
                          ([row()], [row("::", inode=888)]), ([], [row("::")])]:
            with self.subTest(tcp=tcp, tcp6=tcp6):
                self.fixture.tables(tcp, tcp6)
                with self.assertRaises(listener.ListenerOwnershipError):
                    self.fixture.prove()

    def test_nonlistener_missing_inode_and_wrong_address_or_port_refuse(self):
        for selected in [row(state="01"), row(inode=0), row("127.0.0.2"), row(port=18002)]:
            with self.subTest(selected=selected):
                self.fixture.tables([selected], [])
                with self.assertRaises(listener.ListenerOwnershipError):
                    self.fixture.prove()

    def test_missing_namespace_table_or_primary_is_not_a_success(self):
        for relative in ["41/ns/net", "41/net/tcp6", "41/stat", "41/fd/3"]:
            with self.subTest(path=relative):
                path = self.fixture.root / relative
                backup = self.fixture.root / "saved-input"
                path.rename(backup)
                try:
                    with self.assertRaises(listener.ListenerOwnershipError):
                        self.fixture.prove()
                finally:
                    backup.rename(path)

    def test_birth_and_namespace_changes_refuse(self):
        self.fixture.process(41, ppid=7, pgid=41, start="99999")
        with self.assertRaisesRegex(listener.ListenerOwnershipError, "birth"):
            self.fixture.prove()
        self.fixture.process(41, ppid=7, pgid=41, start="12345")
        other = self.fixture.root / "other-net"
        other.touch()
        self.fixture.namespace(41, other)
        with self.assertRaisesRegex(listener.ListenerOwnershipError, "namespaces differ"):
            self.fixture.prove()

    def test_mid_observation_birth_socket_and_boot_changes_refuse(self):
        for change in ("birth", "socket", "namespace", "caller_namespace"):
            with self.subTest(change=change):
                original = listener._snapshot
                calls = 0

                def observe(*args):
                    nonlocal calls
                    result = original(*args)
                    calls += 1
                    if calls == 1:
                        if change == "birth":
                            self.fixture.process(41, ppid=7, pgid=41, start="88888")
                        elif change == "socket":
                            self.fixture.tables([row(inode=888)], [])
                            self.fixture.fd(41, 3, "socket:[888]")
                        else:
                            other = self.fixture.root / "moved-net"
                            other.touch()
                            self.fixture.namespace(7 if change == "caller_namespace" else 41, other)
                    return result

                with patch.object(listener, "_snapshot", side_effect=observe):
                    with self.assertRaises(listener.ListenerOwnershipError):
                        self.fixture.prove()
                self.fixture.process(41, ppid=7, pgid=41, start="12345")
                self.fixture.tables([row()], [])
                self.fixture.fd(41, 3, "socket:[777]")
                self.fixture.namespace(41, self.fixture.namespace_file)
                self.fixture.namespace(7, self.fixture.namespace_file)
        view = listener._ProcFS(self.fixture.root, 3)
        with patch.object(view, "boot_id", side_effect=[BOOT, "eb212f71-709a-4aeb-94e8-d458f80b121f"]):
            with self.assertRaisesRegex(listener.ListenerOwnershipError, "boot"):
                self.fixture.prove(view=view)

    def test_primary_fd_disappearing_during_scan_refuses(self):
        view = listener._ProcFS(self.fixture.root, 3)
        original = view.socket_fds

        def scan(pid, inode):
            result = original(pid, inode)
            if pid == 41 and result:
                (self.fixture.root / "41/fd/3").unlink()
            return result

        with patch.object(view, "socket_fds", side_effect=scan):
            with self.assertRaisesRegex(listener.ListenerOwnershipError, "descriptors changed"):
                self.fixture.prove(view=view)

    def test_listener_state_and_competitors_are_rechecked_after_fd_scan(self):
        for after in [[row(state="01")], [row(), row(inode=888)]]:
            with self.subTest(after=after):
                self.fixture.tables([row()], [])
                view = listener._ProcFS(self.fixture.root, 3)
                original = view.socket_fds

                def scan(pid, inode):
                    result = original(pid, inode)
                    if pid == 41:
                        self.fixture.tables(after, [])
                    return result

                with patch.object(view, "socket_fds", side_effect=scan):
                    with self.assertRaisesRegex(listener.ListenerOwnershipError, "listener changed"):
                        self.fixture.prove(view=view)

    def test_inaccessible_foreign_inventory_and_hidden_mount_refuse(self):
        original = os.scandir

        def scan(path):
            if Path(path) == self.fixture.root / "42/fd":
                raise PermissionError("not observable")
            return original(path)

        with patch.object(listener.os, "scandir", side_effect=scan):
            with self.assertRaisesRegex(listener.ListenerOwnershipError, "inventory is incomplete"):
                self.fixture.prove()
        (self.fixture.root / "7/mountinfo").write_text(
            f"25 0 0:5 / {self.fixture.root} rw - proc proc rw,hidepid=2\n")
        with self.assertRaisesRegex(listener.ListenerOwnershipError, "hidden"):
            self.fixture.prove()

    def test_observation_bounds_are_refusals(self):
        view = listener._ProcFS(self.fixture.root, 3)
        view.deadline = 0
        with self.assertRaisesRegex(listener.ListenerOwnershipError, "timed out"):
            self.fixture.prove(view=view)
        with patch.object(listener, "_MAX_TABLE_BYTES", 5):
            with self.assertRaisesRegex(listener.ListenerOwnershipError, "bound"):
                self.fixture.prove()

    def test_public_input_refusals_do_not_connect_or_resolve_dns(self):
        for host, port in [("localhost", 1), ("10.0.0.1", 1), ("0.0.0.0", 1),
                           ("::", 1), ("::1%lo", 1), ("127.0.0.1", True), ("::1", 0)]:
            with self.subTest(host=host, port=port), self.assertRaises(listener.ListenerOwnershipError):
                listener.prove_listener(self.fixture.owner, host, port)
        with self.assertRaises(listener.ListenerOwnershipError):
            listener.prove_listener(replace(self.fixture.owner, identity_source="ps_lstart_seconds"),
                                    "127.0.0.1", 18001)


SOCKET_SERVER = r'''
import json, os, socket, subprocess, sys, time
family, port, mode = sys.argv[1:]
s = socket.socket(socket.AF_INET6 if family == "6" else socket.AF_INET, socket.SOCK_STREAM)
if family == "6":
    s.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
if mode == "reuse":
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
s.bind(("::1" if family == "6" else "127.0.0.1", int(port)))
s.listen(8)
if mode == "shared":
    subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], pass_fds=(s.fileno(),))
print(json.dumps({"port": s.getsockname()[1], "pid": os.getpid()}), flush=True)
while True:
    time.sleep(.02)
'''


def stable_positive_proof(owner, host, port, timeout=3):
    """Retry only whole read-only observations, as the capture caller does.

    Real shared hosts can change an unrelated FD during the complete procfs walk.
    The production primitive must refuse that snapshot; the positive fixture needs
    a later complete proof within one total deadline, not an assumed quiet host.
    Negative ownership tests below still exercise the primitive directly.
    """
    deadline = time.monotonic() + timeout
    refusals = []
    while True:
        try:
            proof = listener.prove_listener(owner, host, port,
                                            timeout=max(.001, deadline - time.monotonic()))
            if refusals:
                print("positive listener snapshot refusals: " + json.dumps(refusals), file=sys.stderr)
            return proof
        except listener.ListenerOwnershipError as error:
            refusals.append(str(error))
            if time.monotonic() >= deadline:
                raise
            time.sleep(min(.02, max(0, deadline - time.monotonic())))


def require_live_refusal(owner, host, port, expected, timeout=3):
    """A live negative must reach its intended refusal, never merely an FD race.

    Only the observed unrelated descriptor-inventory race is retried. Unexpected
    acceptance fails immediately; an unrelated persistent error also fails the test.
    Deterministic procfs-fixture negatives continue to call the primitive directly.
    """
    deadline = time.monotonic() + timeout
    while True:
        try:
            listener.prove_listener(owner, host, port, timeout=max(.001, deadline - time.monotonic()))
        except listener.ListenerOwnershipError as error:
            if expected in str(error):
                return
            if (str(error) != "descriptor inventory is incomplete or changed"
                    or time.monotonic() >= deadline):
                raise
            print("live negative snapshot refused: " + str(error), file=sys.stderr)
            time.sleep(min(.02, max(0, deadline - time.monotonic())))
        else:
            raise AssertionError("listener accepted a fixture that requires refusal: " + expected)


class PositiveSnapshotRetryTests(unittest.TestCase):
    def test_changed_inventory_requires_a_new_complete_proof(self):
        expected = object()
        with patch.object(listener, "prove_listener", side_effect=[
                listener.ListenerOwnershipError("descriptor inventory changed"), expected]) as observed:
            self.assertIs(stable_positive_proof(None, "::1", 123), expected)
            self.assertEqual(observed.call_count, 2)

    def test_persistent_refusal_stays_a_failure_after_total_deadline(self):
        with patch.object(listener, "prove_listener", side_effect=listener.ListenerOwnershipError("shared")):
            with self.assertRaisesRegex(listener.ListenerOwnershipError, "shared"):
                stable_positive_proof(None, "::1", 123, timeout=.001)

    def test_live_negative_retries_inventory_race_but_requires_its_actual_reason(self):
        with patch.object(listener, "prove_listener", side_effect=[
                listener.ListenerOwnershipError("descriptor inventory is incomplete or changed"),
                listener.ListenerOwnershipError("listener socket is shared with another process")]) as observed:
            require_live_refusal(None, "::1", 123, "shared with another")
            self.assertEqual(observed.call_count, 2)
        with patch.object(listener, "prove_listener", side_effect=listener.ListenerOwnershipError(
                "descriptor inventory is incomplete or changed")):
            with self.assertRaises(listener.ListenerOwnershipError):
                require_live_refusal(None, "::1", 123, "shared with another", timeout=.001)

    def test_live_negative_never_retries_unexpected_acceptance_or_other_errors(self):
        with patch.object(listener, "prove_listener", side_effect=[object(),
                listener.ListenerOwnershipError("shared with another")]) as observed:
            with self.assertRaisesRegex(AssertionError, "listener accepted"):
                require_live_refusal(None, "::1", 123, "shared with another")
            self.assertEqual(observed.call_count, 1)
        with patch.object(listener, "prove_listener", side_effect=listener.ListenerOwnershipError("wrong owner")) as observed:
            with self.assertRaisesRegex(listener.ListenerOwnershipError, "wrong owner"):
                require_live_refusal(None, "::1", 123, "shared with another")
            self.assertEqual(observed.call_count, 1)


@unittest.skipUnless(sys.platform == "linux", "real Linux socket/procfs tests; parent runs on Linux")
class LinuxListenerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="serving-listener-linux-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.servers = []
        self.addCleanup(self.cleanup_servers)

    def cleanup_servers(self):
        for server in reversed(self.servers):
            server.close(reason="socket_test_cleanup")

    def server(self, family="4", port=0, mode="normal"):
        server = OwnedServer(argv=[sys.executable, "-u", "-c", SOCKET_SERVER, family, str(port), mode],
                             env={}, cwd=str(self.root), evidence_dir=str(self.root / str(len(self.servers))),
                             startup_timeout=10, overall_timeout=20, drain_timeout=.2, kill_timeout=2)
        self.servers.append(server)
        server.start()
        deadline = time.monotonic() + 4
        while time.monotonic() < deadline:
            if server.output_path.exists():
                lines = server.output_path.read_text().splitlines()
                if lines and lines[0].startswith('{"port":'):
                    return server, json.loads(lines[0])["port"]
            if server.receipt().get("state") == "finished":
                self.fail(server.output_path.read_text())
            time.sleep(.02)
        self.fail("owned listener did not publish its port")

    def test_real_ipv4_primary_proof_is_accepted_by_owned_server(self):
        server, port = self.server()
        proof = stable_positive_proof(server.identity, "127.0.0.1", port)
        self.assertEqual(proof.owner.key, server.identity.key)
        self.assertEqual(proof.details["samples"][0]["inodes"], proof.details["samples"][1]["inodes"])
        self.assertTrue(server.mark_ready(proof)["ready"])

    @unittest.skipUnless(socket.has_ipv6, "Python reports no IPv6 support")
    def test_real_ipv6_primary_listener(self):
        server, port = self.server(family="6")
        self.assertEqual(stable_positive_proof(server.identity, "::1", port).method, "listener_identity")

    def test_real_foreign_listener_and_reuseport_are_refused(self):
        owner, port = self.server(mode="reuse")
        foreign, foreign_port = self.server()
        require_live_refusal(owner.identity, "127.0.0.1", foreign_port, "owned primary")
        if not hasattr(socket, "SO_REUSEPORT"):
            self.skipTest("kernel/Python lacks SO_REUSEPORT constant")
        self.server(port=port, mode="reuse")
        require_live_refusal(owner.identity, "127.0.0.1", port, "shared")

    def test_real_inherited_listener_and_dead_pid_refuse(self):
        server, port = self.server(mode="shared")
        identity = server.identity
        require_live_refusal(identity, "127.0.0.1", port, "shared with another")
        server.close()
        with self.assertRaises(listener.ListenerOwnershipError):
            listener.prove_listener(identity, "127.0.0.1", port)


if __name__ == "__main__":
    unittest.main()
