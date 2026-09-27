import copy
import unittest

from serving_release import ServingGateError
from serving_trace import validate_cancel_trace
from test_serving_trace import fixture, options, peer, wire


class FramingRegression(unittest.TestCase):
    def test_ordinary_log_positive(self):
        rows = fixture()
        result = validate_cancel_trace(b"ordinary host message\n" + wire(rows), **options())
        self.assertEqual(len(result["decoded_log"]["records"]), len(rows))
        self.assertEqual(result["decoded_log"]["non_lifecycle_lines"], 1)

    def test_canonical_displaced_record_refuses(self):
        with self.assertRaises(ServingGateError):
            validate_cancel_trace(wire(fixture()) + b"host-prefix " + wire(peer()[:1]), **options())

    def test_malformed_displaced_lifecycle_record_must_refuse(self):
        raw = wire(fixture()) + b'host-prefix [request-lifecycle] {"schema":\n'
        with self.assertRaises(ServingGateError):
            validate_cancel_trace(raw, **options())

    def test_unknown_schema_displaced_lifecycle_record_must_refuse(self):
        rows = peer()[:1]
        rows[0]["schema"] = "broken-schema"
        raw = wire(fixture()) + b"host-prefix " + wire(rows)
        with self.assertRaises(ServingGateError):
            validate_cancel_trace(raw, **options())

    def test_displaced_post_end_record_must_not_disappear(self):
        rows = fixture()
        extra = copy.deepcopy(rows[-1])
        extra["seq"] += 1
        extra["at_ns"] += 10
        # Legal JSON spelling of the same schema, plus a displaced reserved marker.
        # With canonical framing this duplicate trace_end is rejected.
        encoded = wire([extra]).replace(b'"memra-request-lifecycle-v1"',
                                        b'"\\u006demra-request-lifecycle-v1"')
        raw = wire(rows) + b"host-prefix " + encoded
        with self.assertRaises(ServingGateError):
            validate_cancel_trace(raw, **options())

    def test_bare_lifecycle_JSON_with_escaped_schema_must_refuse(self):
        encoded = wire(peer()[:1], prefix=b"").replace(
            b'"memra-request-lifecycle-v1"', b'"\\u006demra-request-lifecycle-v1"')
        with self.assertRaises(ServingGateError):
            validate_cancel_trace(wire(fixture()) + encoded, **options())


if __name__ == "__main__":
    unittest.main()
