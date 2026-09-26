"""The prose/remaining-domain split must cover one unchanged native menu."""

import unittest

import eval as native_eval


class DomainSliceTest(unittest.TestCase):
    def test_validation_split_is_complete_and_disjoint(self):
        prose = native_eval.requested_domains(
            "validation", ["prose"],
        )
        remaining = native_eval.requested_domains(
            "validation", ["code", "math"],
        )
        self.assertEqual(set(prose) | set(remaining), set(
            native_eval.DOMAINS,
        ))
        self.assertFalse(set(prose) & set(remaining))
        self.assertEqual(
            native_eval.requested_domains("final", None),
            native_eval.DOMAINS,
        )

    def test_qualifier_cannot_be_sliced(self):
        with self.assertRaises(ValueError):
            native_eval.requested_domains(
                "qualification", ["prose"],
            )
        with self.assertRaises(ValueError):
            native_eval.requested_domains(
                "validation", ["code", "code"],
            )


if __name__ == "__main__":
    unittest.main()
