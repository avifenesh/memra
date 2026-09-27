"""Bounded loopback HTTP capture for the serving release runner.

The caller must verify that this port belongs to its owned server. Successful
capture is not successful inference: serving_release validates the preserved wire
body, while lifecycle predicates check worker state and cleanup separately.
"""

import http.client
import ipaddress
import math
import socket
import threading
import time
from types import MappingProxyType

from serving_release import require


class _ObserverFailure(Exception):
    """A collector observation failed; it is not a server or wire failure."""


class _StrictHTTPResponse(http.client.HTTPResponse):
    """Tighten CPython's chunk hooks without changing payload read semantics.

    The standard client tolerates missing trailer termination and ignores the
    data-chunk delimiter bytes. A release collector must refuse those endings.
    These hooks are exercised by real-socket controls on supported Python hosts.
    """

    def _framing_line(self, kind):
        line = self.fp.readline(65537)
        if len(line) > 65536:
            raise http.client.LineTooLong(kind)
        if not line.endswith(b"\r\n"):
            raise http.client.IncompleteRead(b"")
        return line[:-2]

    def _read_next_chunk_size(self):
        line = self._framing_line("chunk size")
        size, extension, _ = line.partition(b";")
        if extension:
            size = size.rstrip(b" \t")
        if not size or any(byte not in b"0123456789abcdefABCDEF" for byte in size):
            raise http.client.HTTPException("invalid chunk size")
        return int(size, 16)

    def _read_and_discard_trailer(self):
        token = b"!#$%&'*+-.^_`|~0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
        for _ in range(101):
            line = self._framing_line("trailer line")
            if not line:
                return
            name, separator, _ = line.partition(b":")
            if not separator or not name or any(byte not in token for byte in name):
                raise http.client.HTTPException("invalid trailer field")
        raise http.client.HTTPException("too many trailer fields")

    def _get_chunk_left(self):
        remaining = self.chunk_left
        if not remaining:
            if remaining is not None and self._safe_read(2) != b"\r\n":
                raise http.client.HTTPException("invalid data chunk delimiter")
            remaining = self._read_next_chunk_size()
            if remaining == 0:
                self._read_and_discard_trailer()
                self._close_conn()
                remaining = None
            self.chunk_left = remaining
        return remaining


def capture_request(*, request_id, port, path, body, headers=None, host="127.0.0.1",
                    method="POST", connect_timeout=5.0, read_timeout=10.0,
                    wall_timeout=60.0, max_body_bytes=16 * 1024 * 1024,
                    cancel_event=None, observer=None):
    """Return every attempt, including partial bodies and transport failures.

    The watcher shuts down this request's socket at the absolute deadline. Socket
    inactivity timeouts alone do not bound a peer that drips headers or body bytes.
    Chunk timestamps mean client-observed body availability, not model TTFT.

    Optional observer receives an immutable mapping, synchronously and OUTSIDE
    internal locks, for connected (local/peer socket tuples), headers (status and
    immutable header pairs), and first_body (exact bounded kept bytes/end_offset).
    Each includes id/event/observed_ns. It must return promptly: only bounded,
    nonblocking work such as enqueueing the snapshot; no waiting on this capture,
    blocking IO, or deferred writes. Callbacks are never invoked after return.
    Exceptions fail capture with transport_error.kind=observer and retain partial
    body/header evidence. Deadline/cancellation still shut down the socket; a slow
    callback cannot extend the successful capture budget. An arbitrary unbounded
    Python callback cannot be forcibly stopped and violates this API contract.
    Omitting observer preserves the original result shape and collection behavior.
    """
    try:
        address = ipaddress.ip_address(host)
    except ValueError as error:
        raise ValueError("capture host must be a literal loopback address") from error
    require(address.is_loopback, "capture must target loopback")
    require(type(port) is int and 1 <= port <= 65535, "invalid capture port")
    require(isinstance(request_id, str) and request_id, "invalid capture request id")
    require(isinstance(path, str) and path.startswith("/") and not path.startswith("//"),
            "capture path must be origin-relative")
    require(isinstance(body, bytes), "request body must be exact bytes")
    require(method in ("GET", "POST"), "capture method must be GET or POST")
    require(headers is None or (isinstance(headers, dict)
            and all(isinstance(k, str) and isinstance(v, str) for k, v in headers.items())),
            "capture headers must map strings to strings")
    require(type(max_body_bytes) is int and max_body_bytes > 0, "invalid response body limit")
    for value in (connect_timeout, read_timeout, wall_timeout):
        require(type(value) in (int, float) and math.isfinite(value) and value > 0,
                "timeouts must be finite positive numbers")

    require(observer is None or callable(observer), "observer must be callable or None")
    observer_error = None

    def observe(event, observed_ns=None, **fields):
        nonlocal observer_error
        snapshot = MappingProxyType({"id": request_id, "event": event,
            "observed_ns": time.monotonic_ns() if observed_ns is None else observed_ns, **fields})
        try:
            observer(snapshot)
        except BaseException as error:
            try:
                message = str(error)
            except BaseException:
                message = "exception message unavailable"
            observer_error = f"{type(error).__name__}: {message[:1024]}"
            raise _ObserverFailure(observer_error) from error
        # A callback cannot authorize sending/reading after cancellation or a
        # deadline just because the watchdog has not received its next timeslice.
        if cancel_event is not None and cancel_event.is_set():
            interrupt("cancelled")
            raise InterruptedError("capture cancelled during observation")
        if time.monotonic() >= deadline:
            interrupt("timeout")
            raise TimeoutError("capture deadline expired during observation")

    started = time.monotonic_ns()
    deadline = time.monotonic() + wall_timeout
    connection = http.client.HTTPConnection(host, port, timeout=min(connect_timeout, wall_timeout))
    connection.response_class = _StrictHTTPResponse
    completed, lock = threading.Event(), threading.Lock()
    active_socket, interruption = [None], [None]

    def interrupt(kind):
        with lock:
            if completed.is_set():
                return
            if interruption[0] is None:
                interruption[0] = kind
            if active_socket[0] is not None:
                try:
                    active_socket[0].shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass

    def watch():
        while not completed.is_set():
            if cancel_event is not None and cancel_event.is_set():
                interrupt("cancelled")
                return
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                interrupt("timeout")
                return
            completed.wait(min(0.02, remaining))

    watcher = threading.Thread(target=watch, name=f"serving-http-{request_id}", daemon=True)
    result = dict(id=request_id, started_ns=started, status=None, headers=[], body=b"",
                  chunks=[], first_body_byte_ns=None, transport_error=None)
    chunks, phase, response = [], "connect", None
    watcher.start()
    try:
        if cancel_event is not None and cancel_event.is_set():
            interrupt("cancelled")
            raise InterruptedError("cancelled before connect")
        # Numeric loopback avoids an unbounded DNS resolver path. A failed connect
        # is bounded independently because it has no socket for the watcher yet.
        connection.connect()
        with lock:
            active_socket[0] = connection.sock
            interrupted = interruption[0]
        if interrupted is not None:
            raise InterruptedError(interrupted)
        if observer is not None:
            observe("connected", local=tuple(connection.sock.getsockname()), peer=tuple(connection.sock.getpeername()))
        connection.sock.settimeout(read_timeout)
        phase = "read"
        connection.request(method, path, body=body, headers=dict(headers or {}))
        response = connection.getresponse()
        result["status"], result["headers"] = response.status, response.getheaders()
        if observer is not None:
            observe("headers", status=result["status"], headers=tuple(tuple(pair) for pair in result["headers"]))
        size = 0
        while True:
            chunk = response.read1(min(65536, max_body_bytes - size + 1))
            if not chunk:
                if response.length not in (None, 0):
                    raise http.client.IncompleteRead(b"", response.length)
                break
            now = time.monotonic_ns()
            kept = chunk[:max_body_bytes - size]
            if kept:
                if result["first_body_byte_ns"] is None:
                    result["first_body_byte_ns"] = now
                chunks.append(kept)
                size += len(kept)
                result["chunks"].append({"end_offset": size, "observed_ns": now})
                if observer is not None and len(chunks) == 1:
                    observe("first_body", observed_ns=now, end_offset=size, data=bytes(kept))
            if len(kept) != len(chunk):
                raise ValueError("response exceeds capture byte limit")
    except _ObserverFailure:
        result["transport_error"] = {"kind": "observer", "message": observer_error}
    except (OSError, http.client.HTTPException, ValueError) as error:
        result["transport_error"] = {"kind": interruption[0] or
                                     ("timeout" if isinstance(error, TimeoutError) else phase),
                                     "message": f"{type(error).__name__}: {error}"}
    finally:
        # This protected observation is the final capture decision. Pending
        # cancellation cannot be lost between watcher polls; later cancellation
        # cannot rewrite a completed result. Keep any earlier interruption.
        with lock:
            if interruption[0] is None and cancel_event is not None and cancel_event.is_set():
                interruption[0] = "cancelled"
            if interruption[0] is None and time.monotonic() >= deadline:
                interruption[0] = "timeout"
            completed.set()
            if interruption[0] is not None:
                if observer_error is not None:
                    result["transport_error"] = {"kind": "observer",
                        "message": observer_error + f"; capture interrupted: {interruption[0]}"}
                else:
                    result["transport_error"] = {"kind": interruption[0],
                                                 "message": f"capture {interruption[0]}"}
        if response is not None:
            response.close()
        connection.close()
        watcher.join(timeout=1)
        require(not watcher.is_alive(), "HTTP deadline watcher did not stop")
        result["body"] = b"".join(chunks)
        result["finished_ns"] = time.monotonic_ns()
    return result
