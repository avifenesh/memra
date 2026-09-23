"""Linux procfs ownership evidence for an OwnedServer's TCP listener.

prove_listener(owner, literal_loopback_host, port) returns ReadinessEvidence for
mark_ready(). It makes no connection and checks no HTTP status, model or protocol.
The caller must perform those checks separately. The owner must originate from a
live OwnedServer in this boot; its ProcessIdentity contains Linux start ticks.

Two observations must identify one stable listening socket held only by the
primary PID in the observable procfs inventory. Both TCP tables and every visible
process FD directory must be readable; permission gaps, disappearing inventory,
or competing wildcard/reuseport listeners refuse instead of weakening the proof.
IPv6 wildcard sockets possibly covering IPv4 are conservatively ambiguous because
tcp6 does not expose IPV6_V6ONLY. Namespace handles are pinned while observing.

This is sampled evidence, not an atomic socket lease. It requires an unrestricted
host procfs PID view (not hidepid or a partial/container PID view of a host network
namespace). It cannot guarantee that an FD is not shared immediately afterward.
Keep process supervision active and recheck around qualification as appropriate.
Receipts contain only selected endpoint rows, primary FD numbers and identities;
unrelated process rows, descriptor targets and environment values are never saved.
"""

from contextlib import ExitStack
from dataclasses import asdict
import ipaddress
import math
import os
from pathlib import Path
import re
import sys
import time
import uuid

from serving_process import ProcessIdentity, ReadinessEvidence


class ListenerOwnershipError(ValueError):
    """Ownership is absent or ambiguous; not a readiness or protocol result."""


_MAX_TABLE_BYTES = 16 * 1024 * 1024
_MAX_PROCESSES = 32768
_MAX_FDS = 262144
_LISTENER_KEYS = ("family", "address", "port", "inode", "overlap")


def _require(condition, message):
    if not condition:
        raise ListenerOwnershipError(message)


def _identity(pid, raw):
    try:
        prefix, fields = raw.rsplit(")", 1)
        _require(int(prefix.split(" (", 1)[0]) == pid, "process stat PID differs")
        fields = fields.split()
        state, ppid, pgid, start = fields[0], int(fields[1]), int(fields[2]), fields[19]
        # Early kernel tasks can legitimately start at tick zero.
        _require(start.isdecimal(), "invalid process birth ticks")
        return ProcessIdentity(pid, ppid, pgid, start, "linux_proc_start_ticks", state)
    except (ValueError, IndexError) as error:
        raise ListenerOwnershipError("cannot parse process birth identity") from error


def _check_owner(expected, observed):
    _require(observed.key == expected.key, "primary PID birth changed or was reused")
    _require(observed.ppid == expected.ppid and observed.pgid == expected.pgid,
             "primary process ownership/session changed")
    _require(observed.state not in {"Z", "X", "x"}, "primary process has exited")


def _decode_address(value, family):
    width = 8 if family == "tcp" else 32
    _require(len(value) == width and re.fullmatch(r"[0-9A-Fa-f]+", value),
             "malformed TCP local address")
    # procfs prints each network-order u32 as a host-order hex integer.
    packed = b"".join(int(value[i:i + 8], 16).to_bytes(4, sys.byteorder)
                      for i in range(0, width, 8))
    return ipaddress.ip_address(packed)


def _overlap(bound, requested):
    """Return definite/possible or None; never ignore an overlapping wildcard."""
    requested = requested.ipv4_mapped if isinstance(requested, ipaddress.IPv6Address) and requested.ipv4_mapped else requested
    if isinstance(bound, ipaddress.IPv6Address) and bound.ipv4_mapped:
        bound = bound.ipv4_mapped
    if bound.version == requested.version:
        return "definite" if bound == requested or bound.is_unspecified else None
    if requested.version == 4 and bound.version == 6 and bound.is_unspecified:
        return "possible"  # procfs cannot distinguish a v6-only socket here
    return None


def _selected_rows(raw, family, host, port):
    lines = raw.splitlines()
    _require(lines and lines[0].split()[:2] == ["sl", "local_address"],
             f"missing {family} table header")
    result = []
    for line in lines[1:]:
        if not line.strip():
            continue
        fields = line.split()
        _require(len(fields) >= 10, f"malformed {family} row")
        try:
            state = int(fields[3], 16)
        except ValueError as error:
            raise ListenerOwnershipError(f"malformed {family} state") from error
        if state != 0x0A:  # TCP_LISTEN
            continue
        try:
            address, port_hex = fields[1].split(":")
            _require(len(port_hex) == 4, f"malformed {family} port")
            local_port = int(port_hex, 16)
        except ValueError as error:
            raise ListenerOwnershipError(f"malformed {family} endpoint") from error
        if local_port != port:
            continue
        bound = _decode_address(address, family)
        overlap = _overlap(bound, host)
        if overlap is None:
            continue
        _require(fields[9].isdecimal() and int(fields[9]) > 0,
                 "listener inode is missing or disappeared")
        result.append({"family": family, "address": str(bound), "port": port,
                       "inode": int(fields[9]), "overlap": overlap, "raw": line})
    return result


class _ProcFS:
    """Private I/O seam for CPU fixtures; the public entry point always uses /proc."""

    def __init__(self, root, timeout):
        self.root = Path(root)
        self.deadline = time.monotonic() + timeout
        self.fd_reads = 0

    def check_time(self):
        _require(time.monotonic() < self.deadline, "listener ownership observation timed out")

    def text(self, relative, limit=65536):
        self.check_time()
        try:
            with (self.root / relative).open("rb") as handle:
                value = handle.read(limit + 1)
        except OSError as error:
            raise ListenerOwnershipError(f"required procfs input is unavailable: {relative}") from error
        _require(len(value) <= limit, "procfs input exceeds observation bound")
        self.check_time()
        return value.decode("utf-8", errors="surrogateescape")

    def birth(self, pid):
        return _identity(pid, self.text(f"{pid}/stat"))

    def boot_id(self):
        value = self.text("sys/kernel/random/boot_id").strip()
        try:
            _require(str(uuid.UUID(value)) == value, "invalid boot identity")
        except ValueError as error:
            raise ListenerOwnershipError("invalid boot identity") from error
        return value

    def visibility(self):
        mounts = []
        for line in self.text("self/mountinfo", _MAX_TABLE_BYTES).splitlines():
            parts = line.split(" - ", 1)
            _require(len(parts) == 2, "cannot establish procfs visibility")
            left, right = parts[0].split(), parts[1].split()
            _require(len(left) >= 6 and len(right) >= 3, "malformed mount inventory")
            mountpoint = re.sub(r"\\([0-7]{3})", lambda m: chr(int(m[1], 8)), left[4])
            if mountpoint == str(self.root) and right[0] == "proc":
                mounts.append((left[3], set(left[5].split(",") + right[2].split(","))))
        _require(len(mounts) == 1 and mounts[0][0] == "/", "partial or ambiguous procfs mount")
        options = mounts[0][1]
        _require(not any(option.startswith("hidepid=") and option not in {"hidepid=0", "hidepid=off"}
                         for option in options), "hidden process inventory cannot prove ownership")
        _require("subset=pid" not in options, "partial procfs inventory cannot prove ownership")

    def namespace(self, who, stack):
        self.check_time()
        path = self.root / str(who) / "ns/net"
        try:
            fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC)
            stack.callback(os.close, fd)
            stat = os.fstat(fd)
            link = os.readlink(path)
        except OSError as error:
            raise ListenerOwnershipError("network namespace is unavailable") from error
        _require(link == f"net:[{stat.st_ino}]", "network namespace changed while opening")
        return {"device": stat.st_dev, "inode": stat.st_ino, "link": link}

    def pids(self):
        self.check_time()
        result = []
        try:
            with os.scandir(self.root) as entries:
                for entry in entries:
                    if entry.name.isdecimal():
                        result.append(int(entry.name))
                        _require(len(result) <= _MAX_PROCESSES, "process inventory exceeds bound")
                        self.check_time()
        except OSError as error:
            raise ListenerOwnershipError("process inventory is unavailable") from error
        return sorted(result)

    def socket_fds(self, pid, inode):
        self.check_time()
        matches = []
        try:
            with os.scandir(self.root / str(pid) / "fd") as entries:
                for entry in entries:
                    self.fd_reads += 1
                    _require(self.fd_reads <= _MAX_FDS, "descriptor inventory exceeds bound")
                    self.check_time()
                    _require(entry.name.isdecimal(), "malformed descriptor inventory")
                    target = os.readlink(entry.path)
                    if target == f"socket:[{inode}]":
                        matches.append(int(entry.name))
        except OSError as error:
            # A vanished descriptor could have held the socket too: caller may retry
            # a fresh proof, but this observation cannot claim exclusive ownership.
            raise ListenerOwnershipError("descriptor inventory is incomplete or changed") from error
        return sorted(matches)


def _snapshot(view, owner, host, port, namespace):
    before = view.birth(owner.pid)
    _check_owner(owner, before)
    with ExitStack() as pins:
        _require(view.namespace(owner.pid, pins) == namespace
                 and view.namespace("thread-self", pins) == namespace,
                 "primary and caller network namespaces differ or changed")
        rows = []
        for family in ("tcp", "tcp6"):
            rows.extend(_selected_rows(view.text(f"{owner.pid}/net/{family}", _MAX_TABLE_BYTES),
                                       family, host, port))
        _require(len(rows) == 1, "absent or ambiguous/shared same-endpoint listeners")
        selected = rows[0]
        _require(selected["overlap"] == "definite", "IPv6 listener coverage of IPv4 is ambiguous")
        inode = selected["inode"]
        primary_fds = view.socket_fds(owner.pid, inode)
        _require(primary_fds, "LISTEN socket is not held by the owned primary PID")
        pids = view.pids()
        _require(owner.pid in pids, "primary disappeared from process inventory")
        for pid in pids:
            if pid == owner.pid:
                continue
            # Do not filter by holder's current netns: a passed/inherited socket
            # retains its original namespace even after its holder changes netns.
            birth = view.birth(pid)
            shared = view.socket_fds(pid, inode)
            _require(view.birth(pid).key == birth.key, "PID changed during descriptor inventory")
            _require(not shared, "listener socket is shared with another process")
        _require(primary_fds == view.socket_fds(owner.pid, inode),
                 "primary listener descriptors changed or disappeared")
        # A socket FD can survive shutdown while no longer being in LISTEN.
        # Recheck endpoint state/competition after the descriptor walk as well.
        rows_after = []
        for family in ("tcp", "tcp6"):
            rows_after.extend(_selected_rows(view.text(f"{owner.pid}/net/{family}", _MAX_TABLE_BYTES),
                                             family, host, port))
        _require(len(rows_after) == 1
                 and all(rows_after[0][key] == selected[key] for key in _LISTENER_KEYS),
                 "listener changed, disappeared or became shared during descriptor inventory")
        after = view.birth(owner.pid)
        _check_owner(owner, after)
        _require(view.namespace(owner.pid, pins) == namespace
                 and view.namespace("thread-self", pins) == namespace,
                 "network namespace changed during listener observation")
    return {"owner_before": asdict(before), "owner_after": asdict(after),
            "selected_rows": rows, "selected_rows_after": rows_after,
            "inodes": [inode], "primary_fds": primary_fds,
            "processes_checked": len(pids)}


def _prove_listener(view, owner, host, port):
    started_ns = time.monotonic_ns()
    started_unix_ns = time.time_ns()
    view.visibility()
    boot = view.boot_id()
    before = view.birth(owner.pid)
    _check_owner(owner, before)
    with ExitStack() as pins:
        namespace = view.namespace("thread-self", pins)
        _require(view.namespace(owner.pid, pins) == namespace,
                 "primary and caller network namespaces differ")
        samples = [_snapshot(view, owner, host, port, namespace) for _ in range(2)]
        first, second = (sample["selected_rows"][0] for sample in samples)
        # Queue/counter fields may change during valid traffic, but ownership may not.
        _require(all(first[k] == second[k] for k in _LISTENER_KEYS)
                 and samples[0]["primary_fds"] == samples[1]["primary_fds"],
                 "listener socket changed between observations")
        after = view.birth(owner.pid)
        _check_owner(owner, after)
        _require(view.boot_id() == boot, "boot identity changed")
        view.visibility()
        _require(view.namespace(owner.pid, pins) == namespace
                 and view.namespace("thread-self", pins) == namespace,
                 "network namespace changed before proof completion")
    view.check_time()
    return ReadinessEvidence(after, "listener_identity", {
        "schema": "memra-linux-listener-v1", "endpoint": {"host": str(host), "port": port},
        "boot_id": boot, "network_namespace": namespace,
        "owner_before": asdict(before), "owner_after": asdict(after),
        "started_ns": started_ns, "finished_ns": time.monotonic_ns(),
        "clock": "monotonic_ns", "started_unix_ns": started_unix_ns,
        "finished_unix_ns": time.time_ns(), "samples": samples,
        "protocol_validation": "separate", "observation_scope": "visible procfs PID inventory"})


def prove_listener(owner, host, port, *, timeout=3.0):
    """Prove a literal loopback TCP endpoint is exclusively held by this primary.

    Inaccessible or unstable procfs inventories refuse. This function performs no
    network requests and must not be used through a partial/hidden PID inventory.
    The returned proof can be passed to OwnedServer.mark_ready after the caller's
    separate bounded protocol/model checks. No unrelated proc rows are returned.
    """
    _require(isinstance(owner, ProcessIdentity) and type(owner.pid) is int and owner.pid > 0
             and owner.identity_source == "linux_proc_start_ticks"
             and isinstance(owner.start_time, str) and owner.start_time.isdecimal(),
             "owner must be a Linux ProcessIdentity from OwnedServer")
    _require(isinstance(host, str) and "%" not in host, "host must be a literal loopback address")
    try:
        address = ipaddress.ip_address(host)
    except ValueError as error:
        raise ListenerOwnershipError("host must be a literal loopback address; no DNS") from error
    mapped = address.ipv4_mapped if isinstance(address, ipaddress.IPv6Address) else None
    _require(address.is_loopback or (mapped is not None and mapped.is_loopback),
             "only loopback endpoints are accepted")
    _require(type(port) is int and 1 <= port <= 65535, "port must be an integer from 1 to 65535")
    _require(type(timeout) in (int, float) and math.isfinite(timeout) and timeout > 0,
             "timeout must be finite and positive")
    _require(sys.platform == "linux", "listener ownership requires Linux procfs")
    return _prove_listener(_ProcFS("/proc", timeout), owner, address, port)
