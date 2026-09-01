#!/usr/bin/env python3
"""CPU-only tests for the architecture gate scaffold generator."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parent.parent
TOOL = ROOT / "tools" / "generate-arch-gates.py"


def valid_spec() -> dict:
    return {
        "id": "38",
        "artifact_env": "MEMRA_Q38_GGUF",
        "chunk": {
            "label": "qwen38-swa",
            "prompts": ["research/prompts/qwen38-long.txt"],
            "chunks": [4096, 513, 512, 256, 64],
            "steps": 24,
            "seam": "MEMRA_Q38_CHUNK_LEGACY",
        },
        "tick": {
            "label": "qwen38-tick",
            "prompts": ["research/prompts/qwen38-long.txt"],
            "budgets": [0, 1024, 513, 512, 256, 64],
            "splits": [64, 256, 512],
            "steps": 24,
            "seam": "MEMRA_Q38_CALLLOCAL",
        },
        "batch": {
            "model_alias": "qwen38",
            "draft_path": "/data/models/qwen38-draft.gguf",
            "draft_env": "MEMRA_Q38_DRAFT_GGUF",
            "canary_env": {"MEMRA_Q38_BATCH": "0"},
            # A-10: the canary must name WHICH assertion the seam is guaranteed to break.
            # MEMRA_Q38_BATCH=0 has exactly one guaranteed consequence — no batched-walk line
            # can be emitted — so the batched-walk arm must be the red one.
            "canary_expect_regex": "FAIL: no batched-walk evidence",
            "required_gpus": 2,
            "pp_stages": 2,
            "pp_devices": [0, 1],
            "concurrency": [2, 4],
            # Inside the band reserved for generated gates. It was 8094 here and in
            # docs/ONBOARDING.md, which is tools/step35-b2-geometry-gate.sh's port (A-16).
            "port": 18300,
            "receipt_dir": "research/qwen38-batch/raw",
            "server_env": {
                "MEMRA_SERVE_B1FAST": "0",
                "MEMRA_SERVE_SPEC": "0",
            },
            "request": {
                "messages": [{"role": "user", "content": "Count to eight."}],
                "max_tokens": 48,
                "temperature": 0.0,
            },
            "liveness": {
                "cap_regex": "qwen38: decode chunk cap [0-9]+",
                "cap_min": 2,
                "walk_regex": "\\[qwen38-batch\\] first B>1",
            },
        },
        "mapping": [
            {
                "path_regex": (
                    "^crates/memra-engine/src/"
                    "(decode|decode_batch|forward|hybrid_forward)\\.rs$"
                ),
                "kernel_scope": "synthetic",
                "base_probes": ["g12", "q9", "q35"],
                "base_spec_probes": ["q35spec"],
                "gate_families": ["chunk", "tick", "batch"],
            },
            {
                "path_regex": "^crates/memra-server/",
                "kernel_scope": "none",
                "base_probes": ["sstress", "accept"],
                "base_spec_probes": [],
                "gate_families": ["tick", "batch"],
            },
        ],
    }


class ArchGateGeneratorTests(unittest.TestCase):
    def run_generator(
        self,
        directory: Path,
        spec: dict,
        *,
        extra: tuple[str, ...] = (),
    ) -> subprocess.CompletedProcess[str]:
        spec_path = directory / "spec.json"
        spec_path.write_text(json.dumps(spec), encoding="utf-8")
        return subprocess.run(
            [
                sys.executable,
                str(TOOL),
                "Qwen 3.8",
                "/data/models/Qwen3.8 27B.gguf",
                "--spec",
                str(spec_path),
                "--out-dir",
                "generated/qwen-3-8",
                *extra,
            ],
            cwd=directory,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

    def test_generates_scripts_and_registry_fragments(self) -> None:
        with tempfile.TemporaryDirectory(prefix="arch-gate-generator-") as temp:
            directory = Path(temp)
            result = self.run_generator(directory, valid_spec())
            self.assertEqual(result.returncode, 0, result.stderr)
            output = directory / "generated" / "qwen-3-8"

            scripts = sorted(output.glob("*.sh"))
            self.assertEqual(len(scripts), 3)
            for script in scripts:
                self.assertTrue(os.access(script, os.X_OK), script)
                content = script.read_text(encoding="utf-8")
                self.assertNotIn("{{", content)
                syntax = subprocess.run(
                    ["bash", "-n", str(script)],
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=False,
                )
                self.assertEqual(syntax.returncode, 0, syntax.stderr)

            # A missing artifact is REPORTED and FATAL (exit 77), not a silent pass. This
            # assertion used to read `assertEqual(returncode, 0)` — the fixture blessed the
            # defect, which is why the shape survived two audits.
            missing_artifact = subprocess.run(
                [str(output / "qwen-3-8-chunk-invariance-gate.sh")],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(missing_artifact.returncode, 77)
            self.assertIn("SKIP", missing_artifact.stdout)
            self.assertIn("a skip is not a pass", missing_artifact.stdout)
            self.assertIn(
                "MEMRA_ARCH_GATE_ALLOW_SKIP=1", missing_artifact.stdout
            )

            # ...and the escape hatch works, because a developer without artifacts still has to
            # be able to run the tree. It is explicit, it is printed, and it says the run
            # proves nothing.
            accounted = subprocess.run(
                [str(output / "qwen-3-8-chunk-invariance-gate.sh")],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                env={**os.environ, "MEMRA_ARCH_GATE_ALLOW_SKIP": "1"},
            )
            self.assertEqual(accounted.returncode, 0)
            self.assertIn("skip ACCOUNTED", accounted.stdout)

            # The skip is censused when a census is wired, so a battery can count it.
            census = directory / "skip-census.tsv"
            censused = subprocess.run(
                [str(output / "qwen-3-8-tick-invariance-gate.sh")],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                env={
                    **os.environ,
                    "MEMRA_ARCH_GATE_ALLOW_SKIP": "1",
                    "MEMRA_SKIP_CENSUS": str(census),
                },
            )
            self.assertEqual(censused.returncode, 0)
            rows = census.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(rows), 1, rows)
            self.assertIn("qwen-3-8-tick-invariance-gate", rows[0])
            self.assertEqual(rows[0].split("\t")[1], "gate")

            batch_text = (
                output / "qwen-3-8-b2-geometry-gate.sh"
            ).read_text(encoding="utf-8")
            # The port is the band value, the override is derived per gate, and the guard is
            # sourced and called BEFORE the server is started (asserted by position, not eye).
            self.assertIn("DEFAULT_PORT=18300", batch_text)
            self.assertIn("PORT_ENV=MEMRA_QWEN_3_8_B2GEO_PORT", batch_text)
            # Comments stripped first: the template now DOCUMENTS 8094 as the port it used to
            # collide with, and matching the documentation of a bug as the bug is its own blind
            # assertion (round 2 hit the same trap in its anti-pattern greps).
            code_only = "\n".join(
                l for l in batch_text.splitlines() if not l.lstrip().startswith("#")
            )
            self.assertNotIn("8094", code_only)
            lines = batch_text.splitlines()
            guard_line = next(
                i for i, l in enumerate(lines) if "memra_port_guard" in l
            )
            boot_line = next(
                i for i, l in enumerate(lines) if 'MEMRA_ADDR="127.0.0.1:$PORT"' in l
            )
            owned_line = next(
                i for i, l in enumerate(lines) if "memra_port_owned" in l
            )
            self.assertLess(guard_line, boot_line, "guard must precede the bind")
            self.assertLess(boot_line, owned_line, "ownership check is post-boot")
            self.assertIn("EXPECT_ASSERTS=9", batch_text)  # 1 + (2 + 4) + 2

            model_rows = [
                row
                for row in (output / "fast-gate-models.tsv")
                .read_text(encoding="utf-8")
                .splitlines()
                if row and not row.startswith("#")
            ]
            self.assertEqual(len(model_rows), 6)
            self.assertEqual(
                {row.split("\t")[0] for row in model_rows},
                {
                    "chunkinv38",
                    "chunkinv38c",
                    "tickinv38",
                    "tickinv38c",
                    "b2geo38",
                    "b2geo38c",
                },
            )
            self.assertTrue(all(len(row.split("\t")) == 6 for row in model_rows))
            self.assertTrue(
                all("generated/qwen-3-8/" in row for row in model_rows)
            )

            map_rows = [
                row
                for row in (output / "fast-gate-map.tsv")
                .read_text(encoding="utf-8")
                .splitlines()
                if row and not row.startswith("#")
            ]
            self.assertEqual(len(map_rows), 2)
            self.assertTrue(all(len(row.split("\t")) == 4 for row in map_rows))
            engine_probes = map_rows[0].split("\t")[2].split(",")
            self.assertTrue(
                {
                    "chunkinv38",
                    "chunkinv38c",
                    "tickinv38",
                    "tickinv38c",
                    "b2geo38",
                    "b2geo38c",
                }.issubset(engine_probes)
            )
            server_probes = map_rows[1].split("\t")[2].split(",")
            self.assertNotIn("chunkinv38", server_probes)
            self.assertIn("tickinv38", server_probes)
            self.assertIn("b2geo38", server_probes)

            normalized = json.loads(
                (output / "gate-spec.json").read_text(encoding="utf-8")
            )
            self.assertEqual(normalized["architecture"], "Qwen 3.8")
            self.assertEqual(
                normalized["artifact"], "/data/models/Qwen3.8 27B.gguf"
            )
            self.assertEqual(len(normalized["generated_sha256"]), 5)

    def test_rejects_invalid_scientific_inputs(self) -> None:
        cases = []
        missing_seam = valid_spec()
        del missing_seam["chunk"]["seam"]
        cases.append((missing_seam, "missing required keys: seam"))

        bad_cap = valid_spec()
        bad_cap["batch"]["liveness"]["cap_regex"] = "qwen38 cap"
        cases.append((bad_cap, "must include a '[0-9]+' capture"))

        bad_pp = valid_spec()
        bad_pp["batch"]["pp_devices"] = [0]
        cases.append((bad_pp, "length must equal"))

        bad_device = valid_spec()
        bad_device["batch"]["pp_devices"] = [0, 2]
        cases.append((bad_device, "index outside"))

        shadowed_canary = valid_spec()
        shadowed_canary["batch"]["server_env"]["MEMRA_Q38_BATCH"] = "1"
        cases.append((shadowed_canary, "conflicts with server"))

        bad_port = valid_spec()
        bad_port["batch"]["port"] = 70000
        cases.append((bad_port, "at most 65535"))

        # A-16, the finding round 2 listed against this very file: 8094 is
        # tools/step35-b2-geometry-gate.sh's port and was the documented example here.
        colliding_port = valid_spec()
        colliding_port["batch"]["port"] = 8094
        cases.append((colliding_port, "outside the band reserved for generated gates"))

        # In band, but a hand-written gate could still take it later, so the census is checked
        # independently of the band. 18099 is spec-on-cache-hit-gate.sh's.
        out_of_band_but_bound = valid_spec()
        out_of_band_but_bound["batch"]["port"] = 18099
        cases.append((out_of_band_but_bound, "outside the band"))

        missing_canary_expect = valid_spec()
        del missing_canary_expect["batch"]["canary_expect_regex"]
        cases.append(
            (missing_canary_expect, "missing required keys: canary_expect_regex")
        )

        # A pattern that can match a PASSING verdict line would let the canary certify a run in
        # which nothing broke — the A-10 shape moved one level up.
        weak_canary_expect = valid_spec()
        weak_canary_expect["batch"]["canary_expect_regex"] = "batched-walk"
        cases.append((weak_canary_expect, "has to start with 'FAIL'"))

        for spec, expected in cases:
            with self.subTest(expected=expected):
                with tempfile.TemporaryDirectory(
                    prefix="arch-gate-generator-invalid-"
                ) as temp:
                    result = self.run_generator(Path(temp), spec)
                    self.assertEqual(result.returncode, 2)
                    self.assertIn(expected, result.stderr)

    def test_refuses_overwrite_without_force(self) -> None:
        with tempfile.TemporaryDirectory(prefix="arch-gate-generator-force-") as temp:
            directory = Path(temp)
            first = self.run_generator(directory, valid_spec())
            self.assertEqual(first.returncode, 0, first.stderr)
            target = (
                directory
                / "generated"
                / "qwen-3-8"
                / "qwen-3-8-chunk-invariance-gate.sh"
            )
            target.write_text("sentinel\n", encoding="utf-8")

            refused = self.run_generator(directory, valid_spec())
            self.assertEqual(refused.returncode, 2)
            self.assertIn("refusing to overwrite", refused.stderr)
            self.assertEqual(target.read_text(encoding="utf-8"), "sentinel\n")

            forced = self.run_generator(directory, valid_spec(), extra=("--force",))
            self.assertEqual(forced.returncode, 0, forced.stderr)
            self.assertNotEqual(target.read_text(encoding="utf-8"), "sentinel\n")

    def test_rejects_output_path_with_whitespace(self) -> None:
        with tempfile.TemporaryDirectory(prefix="arch-gate-generator-path-") as temp:
            directory = Path(temp)
            spec_path = directory / "spec.json"
            spec_path.write_text(json.dumps(valid_spec()), encoding="utf-8")
            result = subprocess.run(
                [
                    sys.executable,
                    str(TOOL),
                    "Qwen 3.8",
                    "/data/models/qwen38.gguf",
                    "--spec",
                    str(spec_path),
                    "--out-dir",
                    "generated/has space",
                ],
                cwd=directory,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("must not contain whitespace", result.stderr)

    def test_port_census_sees_the_real_gates_and_refuses_a_vacuous_one(self) -> None:
        """The census must name specific ports, and refuse to certify from an empty scan.

        Counting alone is not enough: the first draft of the census used ``[A-Z_]+`` inside the
        parameter expansion and missed ``MEMRA_B2GEO_PORT``'s 8094 because of the digit in
        B2GEO. It still found 17 ports, which looks entirely plausible — so the assertion has to
        be by NAME. The non-vacuity floor covers the other direction, where the scan finds
        nothing at all and every port reads as free.
        """
        listing = subprocess.run(
            [sys.executable, str(TOOL), "--list-ports"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        self.assertEqual(listing.returncode, 0, listing.stderr)
        ports = {
            int(row.split("\t")[0])
            for row in listing.stdout.splitlines()
            if row and not row.startswith("#")
        }
        for expected in (8094, 8177, 8178, 8179, 8180, 8186, 8187, 18086, 18099):
            self.assertIn(expected, ports, f"census lost port {expected}")
        self.assertGreaterEqual(len(ports), 10)
        self.assertIn("band 18300-18399", listing.stdout)

        with tempfile.TemporaryDirectory(prefix="arch-gate-census-") as temp:
            empty_root = Path(temp)
            (empty_root / "tools").mkdir()
            (empty_root / "tools" / "a.sh").write_text("PORT=19001\n", encoding="utf-8")
            vacuous = subprocess.run(
                [
                    sys.executable,
                    str(TOOL),
                    "--list-ports",
                    "--census-root",
                    str(empty_root),
                ],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(vacuous.returncode, 2)
            self.assertIn("the scan is broken, not the tree", vacuous.stderr)

            # And the same floor blocks a GENERATION, not just the report: a port certified
            # free by a broken census is exactly the collision this check exists to stop.
            spec_path = empty_root / "spec.json"
            spec_path.write_text(json.dumps(valid_spec()), encoding="utf-8")
            refused = subprocess.run(
                [
                    sys.executable,
                    str(TOOL),
                    "Qwen 3.8",
                    "/data/models/q38.gguf",
                    "--spec",
                    str(spec_path),
                    "--out-dir",
                    "generated/q38",
                    "--census-root",
                    str(empty_root),
                ],
                cwd=empty_root,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(refused.returncode, 2)
            self.assertIn("non-vacuity floor", refused.stderr)

    def test_sibling_generated_gates_cannot_share_a_port(self) -> None:
        """Two architectures generated from one documented example collided with each other."""
        with tempfile.TemporaryDirectory(prefix="arch-gate-sibling-") as temp:
            root = Path(temp)
            # A census that clears the floor, so the sibling check is what fails.
            (root / "tools").mkdir()
            for index in range(12):
                (root / "tools" / f"g{index}.sh").write_text(
                    f"PORT=190{index:02d}\n", encoding="utf-8"
                )
            sibling = root / "tools" / "generated-arch-gates" / "other-arch"
            sibling.mkdir(parents=True)
            (sibling / "gate-spec.json").write_text(
                json.dumps({"spec": {"batch": {"port": 18300}}}), encoding="utf-8"
            )
            spec_path = root / "spec.json"
            spec_path.write_text(json.dumps(valid_spec()), encoding="utf-8")
            result = subprocess.run(
                [
                    sys.executable,
                    str(TOOL),
                    "Qwen 3.8",
                    "/data/models/q38.gguf",
                    "--spec",
                    str(spec_path),
                    "--out-dir",
                    str(root / "tools" / "generated-arch-gates" / "qwen-3-8"),
                    "--census-root",
                    str(root),
                ],
                cwd=root,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(result.returncode, 2, result.stdout)
            self.assertIn("already claimed by generated gate spec", result.stderr)


if __name__ == "__main__":
    unittest.main()
