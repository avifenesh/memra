import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import analyze


class ReuseAuditorIdentityTests(unittest.TestCase):
    def test_recorded_imported_auditor_is_accepted(self):
        identity = {
            "audit_reuse_sha256": hashlib.sha256(Path(analyze.reuse_auditor.__file__).read_bytes()).hexdigest(),
        }
        analyze.verify_reuse_auditor(identity)

    def test_changed_imported_auditor_is_rejected(self):
        identity = {
            "audit_reuse_sha256": hashlib.sha256(Path(analyze.reuse_auditor.__file__).read_bytes()).hexdigest(),
        }
        with tempfile.TemporaryDirectory() as directory:
            changed = Path(directory) / "audit_reuse.py"
            changed.write_text("def audit_reuse(root): return {}\n")
            with patch.object(analyze.reuse_auditor, "__file__", str(changed)):
                with self.assertRaisesRegex(ValueError, "Cache auditor source differs"):
                    analyze.verify_reuse_auditor(identity)


if __name__ == "__main__":
    unittest.main()
