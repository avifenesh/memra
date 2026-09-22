"""Real loopback socket controls, including slow headers and slow body peers."""

import contextlib
import socket
import threading
import time
import unittest

from serving_http import capture_request
from serving_release import ServingGateError


@contextlib.contextmanager
def peer(handler):
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    listener.settimeout(2)
    port = listener.getsockname()[1]
    failures = []

    def serve():
        try:
            with listener.accept()[0] as client:
                client.settimeout(1)
                data = b""
                while b"\r\n\r\n" not in data:
                    block = client.recv(4096)
                    if not block:
                        return
                    data += block
                header, _, received = data.partition(b"\r\n\r\n")
                length = 0
                for line in header.split(b"\r\n")[1:]:
                    name, _, value = line.partition(b":")
                    if name.lower() == b"content-length":
                        length = int(value.strip())
                # Closing with unread request bytes can produce RST instead of
                # the clean FIN that the incomplete-response controls require.
                while len(received) < length:
                    block = client.recv(length - len(received))
                    if not block:
                        raise AssertionError("test client ended its request body early")
                    received += block
                handler(client)
        except (BrokenPipeError, ConnectionResetError):
            pass  # The cancellation/limit controls intentionally close this peer.
        except Exception as error:
            failures.append(error)

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    try:
        yield port
    finally:
        listener.close()
        thread.join(timeout=3)
        if thread.is_alive():
            raise AssertionError("test peer was not reaped")
        if failures:
            raise failures[0]


class HttpCaptureTests(unittest.TestCase):
    def capture(self, port, **options):
        return capture_request(request_id="test", port=port, path="/v1/chat/completions",
                               body=b"{}", read_timeout=0.2, wall_timeout=0.4, **options)

    def test_complete_response_retains_headers_bytes_and_timestamps(self):
        def handler(client):
            client.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: 5\r\nX-Request-Id: test\r\n\r\nhello")
        with peer(handler) as port:
            result = self.capture(port)
        self.assertIsNone(result["transport_error"])
        self.assertEqual((result["status"], result["body"]), (200, b"hello"))
        self.assertIn(("X-Request-Id", "test"), result["headers"])
        self.assertLessEqual(result["started_ns"], result["first_body_byte_ns"])
        self.assertLessEqual(result["first_body_byte_ns"], result["finished_ns"])
        self.assertEqual(result["chunks"][-1]["end_offset"], 5)

    def test_slow_drip_body_cannot_reset_wall_deadline(self):
        def handler(client):
            client.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: 100\r\n\r\n")
            for _ in range(100):
                client.sendall(b"x")
                time.sleep(0.03)
        with peer(handler) as port:
            result = self.capture(port)
        self.assertEqual(result["transport_error"]["kind"], "timeout")
        self.assertGreater(len(result["body"]), 1)
        self.assertLess(len(result["body"]), 100)
        self.assertLess((result["finished_ns"] - result["started_ns"]) / 1e9, 1.5)

    def test_slow_drip_headers_are_also_bounded(self):
        def handler(client):
            for byte in b"HTTP/1.1 200 OK\r\nContent-Length: 1\r\n\r\nx":
                client.sendall(bytes([byte]))
                time.sleep(0.03)
        with peer(handler) as port:
            result = self.capture(port)
        self.assertEqual(result["transport_error"]["kind"], "timeout")
        self.assertLess((result["finished_ns"] - result["started_ns"]) / 1e9, 1.5)

    def test_early_content_length_eof_is_a_failure_with_partial_body(self):
        with peer(lambda client: client.sendall(
                b"HTTP/1.1 200 OK\r\nContent-Length: 20\r\n\r\npartial")) as port:
            result = self.capture(port)
        self.assertEqual(result["body"], b"partial")
        self.assertEqual(result["transport_error"]["kind"], "read")
        self.assertIn("IncompleteRead", result["transport_error"]["message"])

    def test_cancelled_socket_keeps_partial_output_without_claiming_worker_cleanup(self):
        cancel = threading.Event()
        def handler(client):
            client.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: 20\r\n\r\npartial")
            time.sleep(0.04)
            cancel.set()
            time.sleep(0.08)
        with peer(handler) as port:
            result = self.capture(port, cancel_event=cancel)
        self.assertEqual(result["body"], b"partial")
        self.assertEqual(result["transport_error"]["kind"], "cancelled")
        self.assertNotIn("worker_cancelled", result)

    def test_body_limit_refuses_and_preserves_bounded_prefix(self):
        with peer(lambda client: client.sendall(
                b"HTTP/1.1 200 OK\r\nContent-Length: 20\r\n\r\nabcdefghijklmnopqrst")) as port:
            result = self.capture(port, max_body_bytes=8)
        self.assertEqual(result["body"], b"abcdefgh")
        self.assertEqual(result["transport_error"]["kind"], "read")
        self.assertIn("byte limit", result["transport_error"]["message"])

    def test_chunked_body_retains_payload_and_requires_terminator(self):
        for suffix, succeeds in ((b"0\r\n\r\n", True), (b"", False)):
            with self.subTest(suffix=suffix):
                with peer(lambda client: client.sendall(
                        b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n5\r\nhello\r\n" + suffix)) as port:
                    result = self.capture(port)
                self.assertEqual(result["body"], b"hello")
                self.assertEqual(result["transport_error"] is None, succeeds)

    def test_non_loopback_and_invalid_budgets_refuse_before_connect(self):
        for options in ({"host": "192.0.2.1"}, {"host": "localhost"}, {"max_body_bytes": 0}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.capture(1, **options)
        with self.assertRaises(ServingGateError):
            capture_request(request_id="bad", port=1, path="/", body=b"", wall_timeout=float("inf"))

    def test_pre_cancel_does_not_open_a_connection(self):
        cancel = threading.Event(); cancel.set()
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0)); listener.listen(); listener.settimeout(0.05)
            result = self.capture(listener.getsockname()[1], cancel_event=cancel)
            with self.assertRaises(TimeoutError):
                listener.accept()
        self.assertEqual(result["transport_error"]["kind"], "cancelled")
        self.assertIsNone(result["status"])

    def test_cancel_before_fast_response_is_observed_at_completion(self):
        for attempt in range(10):
            with self.subTest(attempt=attempt):
                cancel = threading.Event()
                def handler(client):
                    cancel.set()
                    client.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: 5\r\n\r\nhello")
                with peer(handler) as port:
                    result = self.capture(port, cancel_event=cancel)
                self.assertIsNotNone(result["transport_error"])
                self.assertEqual(result["transport_error"]["kind"], "cancelled")
                self.assertTrue(b"hello".startswith(result["body"]))

    def test_later_cancel_cannot_mutate_completed_capture(self):
        cancel = threading.Event()
        with peer(lambda client: client.sendall(
                b"HTTP/1.1 200 OK\r\nContent-Length: 5\r\n\r\nhello")) as port:
            result = self.capture(port, cancel_event=cancel)
        cancel.set()
        self.assertEqual(result["body"], b"hello")
        self.assertIsNone(result["transport_error"])

    def test_chunk_lines_data_delimiters_and_trailers_must_complete(self):
        cases = [(b"5\r\nhello\r\n0\r\n", False),
                 (b"5\r\nhello\r\n0", False),
                 (b"5\r\nhelloXX0\r\n\r\n", False),
                 (b"5\r\nhello\r\n0\r\nX-Foo: bar\r\n", False),
                 (b"5;extension=yes\r\nhello\r\n0\r\nX-Foo: bar\r\n\r\n", True)]
        for wire, succeeds in cases:
            with self.subTest(wire=wire):
                with peer(lambda client: client.sendall(
                        b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n" + wire)) as port:
                    result = self.capture(port)
                self.assertEqual(result["body"], b"hello")
                self.assertEqual(result["transport_error"] is None, succeeds)


if __name__ == "__main__":
    unittest.main()
