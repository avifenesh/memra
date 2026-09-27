import copy
import json
from pathlib import Path
import unittest

from serving_manifest import required_cells, validate_cell_coverage
from serving_release import ServingGateError


class ServingManifestTests(unittest.TestCase):
    def setUp(self):
        self.manifest = (Path(__file__).parent / "serving-release.cells.json").read_bytes()
        self.scopes = [dict(id="own-plain", model="own-model", route="plain-eager", profile="text-generation-v1"),
                       dict(id="step-pp", model="step-model", route="pp3-fp8", profile="text-generation-v1")]
        self.cells = [dict(row, state="captured") for row in required_cells(self.manifest, self.scopes)]

    def test_each_required_model_route_gets_all_scenarios_without_hardware_veto(self):
        result = validate_cell_coverage(self.manifest, self.scopes, self.cells)
        self.assertEqual(len(result), 22)
        self.assertEqual(result[0]["requirements"]["prompt_tokens_max"], 32)
        self.assertTrue(all("gpu" not in cell["requirements"] for cell in result))
        self.assertTrue(all("state" not in cell for cell in result))

    def test_missing_duplicate_unknown_and_skipped_cannot_qualify(self):
        variants = [self.cells[:-1], self.cells + self.cells[:1],
                    [dict(self.cells[0], id="other/drain")] + self.cells[1:],
                    [dict(self.cells[0], state="skipped")] + self.cells[1:],
                    [dict(self.cells[0], skip=False)] + self.cells[1:]]
        for wrong in variants:
            with self.subTest(count=len(wrong)), self.assertRaises(ServingGateError):
                validate_cell_coverage(self.manifest, self.scopes, wrong)

    def test_receipt_cannot_replace_scope_or_weaken_returned_requirements(self):
        wrong = copy.deepcopy(self.cells);wrong[0]["scope"]["route"] = "unreviewed-route"
        with self.assertRaises(ServingGateError):
            validate_cell_coverage(self.manifest, self.scopes, wrong)
        wrong = copy.deepcopy(self.cells);wrong[0]["requirements"] = {"prompt_tokens_max": 1000000}
        result = validate_cell_coverage(self.manifest, self.scopes, wrong)
        self.assertEqual(result[0]["requirements"]["prompt_tokens_max"], 32)

    def test_empty_duplicate_scope_and_missing_policy_scenario_refuse(self):
        for scopes in ([], self.scopes * 2, [dict(self.scopes[0], profile="unknown")]):
            with self.assertRaises(ServingGateError):
                required_cells(self.manifest, scopes)
        policy = json.loads(self.manifest)
        policy["profiles"]["text-generation-v1"].pop()
        with self.assertRaises(ServingGateError):
            required_cells(json.dumps(policy), self.scopes)


if __name__ == "__main__":
    unittest.main()
