"""Schema shape tests only. Synthetic values are NOT a native execution receipt."""
import copy
import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("battery_schema_e", ROOT / "tools/tier-battery.py")
BATTERY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BATTERY)
SCHEMA = json.loads(Path(__file__).with_name("native-conformance.schema.json").read_text())


def example(schema):
    if "const" in schema:
        return schema["const"]
    if "enum" in schema:
        return schema["enum"][0]
    if "anyOf" in schema:
        return example(schema["anyOf"][0])
    kind = schema["type"]
    if kind == "object":
        return {key: example(value) for key, value in schema["properties"].items()}
    if kind == "array":
        return [example(schema["items"]) for _ in range(max(1, schema.get("minItems", 0)))]
    if kind == "string":
        if schema.get("pattern") == "[0-9a-f]{64}":
            return "a" * 64
        if schema.get("pattern") == "[0-9a-f]{40}":
            return "b" * 40
        return "structural-fixture"
    if kind in ("integer", "number"):
        return max(1, schema.get("minimum", 0))
    if kind == "boolean":
        return False
    if kind == "null":
        return None
    raise AssertionError(kind)


class NativeSchemaTests(unittest.TestCase):
    def setUp(self):
        self.value = example(SCHEMA)

    def validate(self, value):
        BATTERY.validate_schema(value, SCHEMA)

    def test_valid_shape_and_both_artifact_arms(self):
        self.validate(self.value)
        self.value["artifact"] = example(SCHEMA["properties"]["artifact"]["anyOf"][1])
        self.validate(self.value)
        self.value["hardware"]["devices"][0].update(
            power_limit_w=None, power_max_limit_w=None, unknown_reason="not observed")
        self.value["schedules"][0].update(verdict="HELD", executed=False, reason="no GPU")
        self.validate(self.value)

    def test_each_required_field_at_every_depth(self):
        def walk(value, schema):
            if "anyOf" in schema:
                schema = schema["anyOf"][0]
            if isinstance(value, dict):
                for key in schema.get("required", []):
                    bad = copy.deepcopy(value)
                    del bad[key]
                    with self.subTest(missing=key), self.assertRaises(ValueError):
                        BATTERY.validate_schema(bad, schema)
                for key, item in value.items():
                    walk(item, schema["properties"][key])
            elif isinstance(value, list):
                for item in value:
                    walk(item, schema["items"])
        walk(self.value, SCHEMA)

    def test_bad_identity_and_unknown_field(self):
        for field in ("commit", "repository"):
            bad = copy.deepcopy(self.value)
            bad["source"][field] = "wrong"
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.validate(bad)
        for sha in ("a" * 63, "G" * 64, "a" * 65, 123):
            bad = copy.deepcopy(self.value)
            bad["binary"]["sha256"] = sha
            with self.subTest(sha=sha), self.assertRaises(ValueError):
                self.validate(bad)
        self.value["qualification"] = True
        with self.assertRaises(ValueError):
            self.validate(self.value)
        self.value["qualification"] = False
        self.value["unregistered_field"] = 1
        with self.assertRaises(ValueError):
            self.validate(self.value)

    def test_evidence_path_escape_and_integer_type(self):
        for path in ("/absolute", "../escape", "a/../escape", "a/..", "a\\escape"):
            bad = copy.deepcopy(self.value)
            bad["collector"]["journal"]["path"] = path
            with self.subTest(path=path), self.assertRaises(ValueError):
                self.validate(bad)
        self.value["hardware"]["device_count"] = True
        with self.assertRaises(ValueError):
            self.validate(self.value)

    def test_verdict_and_schedule_coverage_shape(self):
        for schedules in ([], [{"name": "device_hand_back", "verdict": "SKIP"}]):
            self.value["schedules"] = schedules
            with self.assertRaises(ValueError):
                self.validate(self.value)


if __name__ == "__main__":
    unittest.main()
