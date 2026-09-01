"""Regression tests for the proxy's credential canonicalization boundary."""

import email.message
import importlib.util
import pathlib
import unittest
from unittest import mock
import urllib.error


MODULE_PATH = pathlib.Path(__file__).with_name("serve-proxy.py")
SPEC = importlib.util.spec_from_file_location("serve_proxy", MODULE_PATH)
serve_proxy = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(serve_proxy)


def headers(*pairs):
    message = email.message.Message()
    for name, value in pairs:
        message[name] = value
    return message


class Response:
    def __init__(self, status):
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc, _traceback):
        return False


def http_error(url, status):
    return urllib.error.HTTPError(url, status, "fixture", {}, None)


class CredentialBoundaryTests(unittest.TestCase):
    def test_missing_credential_is_distinct_from_one_canonical_credential(self):
        self.assertIsNone(serve_proxy.canonical_credential(headers()))
        self.assertEqual(
            serve_proxy.canonical_credential(
                headers(("Authorization", "Bearer one"))),
            ("Authorization", "Bearer one"),
        )

    def test_duplicate_same_name_credentials_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "exactly one"):
            serve_proxy.canonical_credential(
                headers(("Authorization", "Bearer one"),
                        ("Authorization", "Bearer two")))
        with self.assertRaisesRegex(ValueError, "exactly one"):
            serve_proxy.canonical_credential(
                headers(("X-Api-Key", "one"), ("X-Api-Key", "two")))

    def test_every_transfer_encoding_form_is_rejected(self):
        self.assertTrue(serve_proxy.has_transfer_encoding(
            headers(("Transfer-Encoding", "gzip, chunked"))))
        self.assertTrue(serve_proxy.has_transfer_encoding(
            headers(("Transfer-Encoding", "chunked"),
                    ("Transfer-Encoding", "identity"))))
        self.assertFalse(serve_proxy.has_transfer_encoding(
            headers(("Content-Length", "10"))))

    def test_forwarding_replaces_all_original_credentials_with_canonical_one(self):
        incoming = headers(("Authorization", "Bearer one"),
                           ("Authorization", "Bearer two"),
                           ("X-Request-Id", "request-1"))
        forwarded = list(serve_proxy.iter_forward_headers(
            incoming, ("Authorization", "Bearer one")))
        self.assertEqual(
            forwarded,
            [("X-Request-Id", "request-1"),
             ("Authorization", "Bearer one")],
        )

    def test_forwarding_has_no_cross_request_credential_state(self):
        authenticated = headers(("Authorization", "Bearer one"))
        unauthenticated = headers(("X-Request-Id", "request-2"))
        list(serve_proxy.iter_forward_headers(
            authenticated, ("Authorization", "Bearer one")))
        self.assertEqual(
            list(serve_proxy.iter_forward_headers(unauthenticated, None)),
            [("X-Request-Id", "request-2")],
        )

    def test_preflight_retries_a_legacy_replica_then_accepts_a_capable_one(self):
        candidates = [serve_proxy.Backend("http://old"), serve_proxy.Backend("http://new")]
        with mock.patch.object(
            serve_proxy.urllib.request,
            "urlopen",
            side_effect=[http_error("http://old/v1/auth/check", 404), Response(204)],
        ) as opener:
            result = serve_proxy.credential_preflight(
                candidates, "Authorization", "Bearer valid"
            )
        self.assertEqual(result, serve_proxy.PREFLIGHT_ACCEPTED)
        self.assertEqual(opener.call_count, 2)

    def test_preflight_all_legacy_degrades_to_origin_authentication(self):
        candidates = [serve_proxy.Backend("http://old-a"), serve_proxy.Backend("http://old-b")]
        with mock.patch.object(
            serve_proxy.urllib.request,
            "urlopen",
            side_effect=[
                http_error("http://old-a/v1/auth/check", 404),
                http_error("http://old-b/v1/auth/check", 404),
            ],
        ):
            result = serve_proxy.credential_preflight(
                candidates, "X-Api-Key", "legacy-key"
            )
        self.assertEqual(result, serve_proxy.PREFLIGHT_LEGACY)

    def test_preflight_real_denial_wins_over_legacy_fallback(self):
        candidates = [serve_proxy.Backend("http://new"), serve_proxy.Backend("http://old")]
        with mock.patch.object(
            serve_proxy.urllib.request,
            "urlopen",
            side_effect=[
                http_error("http://new/v1/auth/check", 401),
                http_error("http://old/v1/auth/check", 404),
            ],
        ):
            result = serve_proxy.credential_preflight(
                candidates, "Authorization", "Bearer invalid"
            )
        self.assertEqual(result, serve_proxy.PREFLIGHT_DENIED)

    def test_preflight_retries_transport_failure_and_reports_total_outage(self):
        candidates = [serve_proxy.Backend("http://down-a"), serve_proxy.Backend("http://down-b")]
        with mock.patch.object(
            serve_proxy.urllib.request,
            "urlopen",
            side_effect=[TimeoutError("slow"), ConnectionRefusedError("down")],
        ) as opener:
            result = serve_proxy.credential_preflight(
                candidates, "Authorization", "Bearer unknown"
            )
        self.assertEqual(result, serve_proxy.PREFLIGHT_UNAVAILABLE)
        self.assertEqual(opener.call_count, 2)

    def test_preflight_candidate_order_rotates_across_healthy_replicas(self):
        router = serve_proxy.Router(
            ["http://one", "http://two", "http://three"], 8, 16, 1.0
        )
        first = [backend.url for backend in router.preflight_candidates()]
        second = [backend.url for backend in router.preflight_candidates()]
        third = [backend.url for backend in router.preflight_candidates()]
        self.assertEqual(first, ["http://one", "http://two", "http://three"])
        self.assertEqual(second, ["http://two", "http://three", "http://one"])
        self.assertEqual(third, ["http://three", "http://one", "http://two"])


if __name__ == "__main__":
    unittest.main()
