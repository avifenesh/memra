#!/usr/bin/env python3
"""CPU contract for tools/crates_publish.py and the publish step of .github/workflows/publish.yml.

Nothing here talks to crates.io. The unit cases drive the index wait and the retry rules with a fake index, a
fake clock and a fake cargo. The workflow cases run the real "Publish to crates.io" step (its bash, extracted from
the workflow file) against a fake registry served on 127.0.0.1 and a fake cargo whose `publish` resolves internal
dependencies against that registry's index, the way the real one does: a crate is in the API the moment it is
uploaded but reaches the index only after a lag. That is the v0.139.0 tag run (memra-engine uploaded, missing
from the index when memra-server resolved it seconds later) and its recovery run (everything up to memra-lanes
live, memra-engine still lagging). PUBLISH_WORKFLOW_UNDER_TEST runs the same cases against another copy of the
workflow, the before/after check of the fix.
"""
import http.client
import http.server
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import tomllib
import unittest
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
ROOT = TOOLS.parent
sys.path.insert(0, str(TOOLS))
import crates_publish as cp  # noqa: E402

WORKFLOW = Path(os.environ.get("PUBLISH_WORKFLOW_UNDER_TEST", ROOT / ".github/workflows/publish.yml"))
VERSION = "0.139.0"


class FakeClock:
    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def clock(self):
        return self.now

    def sleep(self, s):
        self.sleeps.append(s)
        self.now += s


class FakeIndex:
    """get() for the index: answers, in order, from a script of (status, versions) or exceptions."""

    def __init__(self, *answers):
        self.answers = list(answers)
        self.urls = []

    def __call__(self, url):
        self.urls.append(url)
        answer = self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]
        if isinstance(answer, Exception):
            raise answer
        status, versions = answer
        return status, "".join(json.dumps({"name": "x", "vers": v, "deps": []}) + "\n" for v in versions)


class IndexWaitTests(unittest.TestCase):
    def setUp(self):
        os.environ.pop("CRATES_PUBLISH_REGISTRY", None)
        self.c = FakeClock()
        self.logs = []

    def wait(self, get, timeout=60, poll=10):
        return cp.wait_visible("memra-engine", VERSION, timeout=timeout, poll=poll, get=get, clock=self.c.clock,
                               sleep=self.c.sleep, log=self.logs.append)

    def test_index_paths_follow_the_registry_layout(self):
        self.assertEqual(cp.index_path("memra-engine"), "me/mr/memra-engine")
        self.assertEqual(cp.index_path("Memra-CLI"), "me/mr/memra-cli")
        self.assertEqual(cp.index_path("a"), "1/a")
        self.assertEqual(cp.index_path("ab"), "2/ab")
        self.assertEqual(cp.index_path("abc"), "3/a/abc")
        self.assertEqual(cp.index_path("abcd"), "ab/cd/abcd")

    def test_waits_until_the_exact_version_is_listed(self):
        get = FakeIndex((200, ["0.138.0"]), (200, ["0.138.0"]), (200, ["0.138.0", "0.139.0-rc1"]),
                        (200, ["0.138.0", VERSION]))
        self.assertEqual(self.wait(get), 30)
        self.assertEqual(self.c.sleeps, [10, 10, 10])
        self.assertEqual(get.urls[0], "https://index.crates.io/me/mr/memra-engine")
        self.assertIn("memra-engine@0.139.0 is in the crates.io index", self.logs[-1])

    def test_a_crate_new_to_the_index_and_a_failed_read_are_waited_out(self):
        get = FakeIndex((404, []), OSError("connection reset"), (503, []), (200, [VERSION]))
        self.assertEqual(self.wait(get), 30)

    def test_times_out_with_a_named_error(self):
        get = FakeIndex((200, ["0.138.0"]))
        with self.assertRaises(cp.PublishError) as err:
            self.wait(get, timeout=1200)
        msg = str(err.exception)
        for part in ("memra-engine@0.139.0", "after 1200 s", "https://index.crates.io/me/mr/memra-engine",
                     "Rerun the publish"):
            self.assertIn(part, msg)
        self.assertLessEqual(self.c.now, 1200 + 10, "the wait overran its bound")
        self.assertGreaterEqual(self.c.now, 1200)

    def test_the_default_bound_covers_the_index_cache_lifetime(self):
        self.assertGreaterEqual(cp.INDEX_WAIT_S, 600, "shorter than the index CDN's max-age=600")


LAG = 'failed to select a version for the requirement `memra-engine = "=0.139.0"`'


class PublishRetryTests(unittest.TestCase):
    def setUp(self):
        for name in ("CRATES_PUBLISH_INDEX_WAIT_S", "CRATES_PUBLISH_POLL_S", "CRATES_PUBLISH_LAG_RETRY_S",
                     "CRATES_PUBLISH_RATE_WAIT_S"):
            os.environ.pop(name, None)
        self.c = FakeClock()
        self.waited = []
        self.logs = []

    def publish(self, *outcomes, index_wait=0.0, cargo_lag=None):
        """OUTCOMES are cargo's answers in order; CARGO_LAG (s) instead makes cargo miss memra-engine until that
        much fake time has passed. The fake index wait takes INDEX_WAIT s."""
        outs = list(outcomes)
        calls = []

        def run(name):
            calls.append(self.c.now)
            self.c.now += 1  # a cargo run takes time: a retry loop that never sleeps still ends
            if len(calls) > 200:
                raise AssertionError("cargo ran 200 times: the retry does not back off")
            if cargo_lag is not None:
                return (0, "Uploaded") if self.c.now >= cargo_lag else (101, LAG)
            return outs.pop(0)

        def wait(n, v, timeout=None):
            self.waited.append((n, v, timeout))
            self.c.now += index_wait
            return index_wait

        result = cp.publish("memra-server", VERSION, run=run, wait=wait, sleep=self.c.sleep, clock=self.c.clock,
                            log=self.logs.append)
        return result, calls

    def test_rate_limit_waits_the_new_crate_window_and_retries(self):
        result, calls = self.publish((101, "error: 429 Too Many Requests"), (0, "Uploaded"))
        self.assertEqual((result, len(calls)), ("published", 2))
        self.assertEqual(self.c.sleeps, [cp.RATE_LIMIT_WAIT_S])

    def test_a_dependency_missing_from_the_index_is_waited_for_then_retried(self):
        result, calls = self.publish((101, LAG), (0, "Uploaded"), index_wait=40)
        self.assertEqual((result, len(calls)), ("published", 2))
        self.assertEqual(self.waited, [("memra-engine", VERSION, 1200)])
        self.assertEqual(self.c.sleeps, [], "the index wait already covered the lag; no extra back-off")

    def test_cargo_missing_a_dependency_the_index_lists_backs_off_until_it_resolves(self):
        # the index lists memra-engine here from the start; cargo is served an older copy for 100 s
        result, calls = self.publish(cargo_lag=100)
        self.assertEqual(result, "published")
        self.assertEqual(self.c.sleeps, [30, 60, 90], "no growing back-off between the retries")
        self.assertGreaterEqual(calls[-1], 100)
        timeouts = [t for _, _, t in self.waited]
        self.assertEqual(timeouts[0], 1200)
        self.assertTrue(all(b < a for a, b in zip(timeouts, timeouts[1:])),
                        f"the index wait does not share the lag window: {timeouts}")

    def test_a_crate_missing_from_cargos_index_copy_is_lag_too(self):
        # a workspace crate's first release: cargo's copy of the index has no entry for it at all
        result, calls = self.publish((101, "error: no matching package named `memra-reference` found"),
                                     (0, "Uploaded"), index_wait=20)
        self.assertEqual((result, len(calls)), ("published", 2))
        self.assertEqual(self.waited[0][:2], ("memra-reference", VERSION))

    def test_a_rate_limit_between_two_misses_starts_a_new_lag_window(self):
        os.environ["CRATES_PUBLISH_INDEX_WAIT_S"] = "100"
        self.addCleanup(os.environ.pop, "CRATES_PUBLISH_INDEX_WAIT_S", None)
        result, _ = self.publish((101, LAG), (101, "429 Too Many Requests"), (101, LAG), (0, "Uploaded"))
        self.assertEqual(result, "published", "the 620 s rate-limit wait used up the lag window")

    def test_cargo_missing_a_listed_dependency_is_bounded(self):
        with self.assertRaisesRegex(cp.PublishError, "cargo still cannot resolve memra-engine@0.139.0 .* after 1200 s"):
            self.publish(cargo_lag=10 ** 6)
        self.assertLessEqual(self.c.now, 1200 + 2, "a retry started after the lag window")  # + the last cargo run

    def test_a_dependency_at_another_version_is_a_real_failure(self):
        with self.assertRaisesRegex(cp.PublishError, "other than a rate limit"):
            self.publish((101, 'failed to select a version for the requirement `memra-engine = "=0.138.0"`'))
        self.assertEqual(self.waited, [])

    def test_an_already_uploaded_version_counts_as_live(self):
        result, _ = self.publish((101, "crate version `0.139.0` is already uploaded"))
        self.assertEqual(result, "live")
        result, _ = self.publish((101, "error: crate memra-server@0.139.0 already exists on crates.io index"))
        self.assertEqual(result, "live")

    def test_other_failures_stop_and_attempts_are_bounded(self):
        with self.assertRaisesRegex(cp.PublishError, "other than a rate limit"):
            self.publish((101, "error: failed to verify package tarball"))
        with self.assertRaisesRegex(cp.PublishError, "still rate-limited after 6 attempts"):
            self.publish(*[(101, "429 Too Many Requests")] * cp.ATTEMPTS)
        self.assertEqual(len(self.c.sleeps), cp.ATTEMPTS - 1, "slept after the last attempt")

    def test_an_unanswered_api_is_unknown_not_absent(self):
        self.assertIsNone(cp.api_live("memra-tier", VERSION, get=FakeIndex(OSError("down")), sleep=self.c.sleep))
        self.assertIsNone(cp.api_live("memra-tier", VERSION, get=FakeIndex((503, [])), sleep=self.c.sleep))
        self.assertIs(cp.api_live("memra-tier", VERSION, get=FakeIndex((404, [])), sleep=self.c.sleep), False)
        answers = [OSError("reset"), (200, json.dumps({"version": {"num": VERSION}}))]

        def flaky(_url):
            answer = answers.pop(0)
            if isinstance(answer, Exception):
                raise answer
            return answer
        self.assertIs(cp.api_live("memra-tier", VERSION, get=flaky, sleep=self.c.sleep), True,
                      "one failed query is not retried")

    def test_a_broken_http_read_is_a_failed_read(self):
        real = cp.urllib.request.urlopen

        def broken(*_a, **_k):
            raise http.client.IncompleteRead(b"partial")
        cp.urllib.request.urlopen = broken
        try:
            with self.assertRaises(OSError):
                cp.http_get("https://index.crates.io/me/mr/memra-engine")
        finally:
            cp.urllib.request.urlopen = real

    def test_a_long_wait_says_it_is_still_waiting(self):
        logs = []
        get = FakeIndex(*[(200, ["0.138.0"])] * 13, (200, [VERSION]))
        cp.wait_visible("memra-engine", VERSION, timeout=1200, poll=10, get=get, clock=self.c.clock,
                        sleep=self.c.sleep, log=logs.append)
        self.assertEqual(len([l for l in logs if l.startswith("waiting for memra-engine@0.139.0")]), 2, logs)


# --- the workflow step against a fake registry ----------------------------------------------------------------

def internal_deps():
    deps = {}
    for manifest in sorted(ROOT.glob("crates/*/Cargo.toml")):
        m = tomllib.loads(manifest.read_text())
        deps[m["package"]["name"]] = sorted(k for k in m.get("dependencies", {}) if k.startswith("memra-"))
    return deps


class Registry:
    """The fake crates.io: state.json holds {crate: upload time} and {crate: index lag in s}."""

    def __init__(self, state):
        self.state = state

    def load(self):
        return json.loads(self.state.read_text())

    def visible(self, name):
        s = self.load()
        return name in s["uploaded"] and time.time() >= s["uploaded"][name] + s["lag"].get(name, 0)


def serve(registry):
    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_GET(self):
            parts = self.path.strip("/").split("/")
            s = registry.load()
            body, code = "", 404
            if parts[:3] == ["api", "v1", "crates"] and s.get("api_down"):
                code = 503
            elif parts[:3] == ["api", "v1", "crates"] and len(parts) == 5:
                if parts[3] in s["uploaded"] and parts[4] == s["version"]:
                    body, code = json.dumps({"version": {"num": s["version"]}}), 200
            elif parts[0] == "index" and registry.visible(parts[-1]):
                body, code = json.dumps({"name": parts[-1], "vers": s["version"]}) + "\n", 200
            self.send_response(code)
            self.end_headers()
            self.wfile.write(body.encode())

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


FAKE_CARGO = textwrap.dedent("""\
    #!/usr/bin/env python3
    import json, os, sys, time, urllib.request
    state_path = os.environ["FAKE_REGISTRY_STATE"]
    s = json.load(open(state_path))
    if sys.argv[1] == "metadata":
        print(json.dumps({"packages": [{"name": "memra-server", "version": s["version"]}]}))
        sys.exit(0)
    assert sys.argv[1:3] == ["publish", "-p"] and sys.argv[4:] == ["--locked"], sys.argv
    name = sys.argv[3]
    base = os.environ["FAKE_REGISTRY_URL"]
    for dep in s["deps"][name]:  # cargo resolves each internal dependency against the index
        try:
            urllib.request.urlopen(f"{base}/index/x/y/{dep}", timeout=5).read()
            # cargo_lag: cargo is served an older copy than the poll sees, for that long after the upload
            assert time.time() >= s["uploaded"][dep] + s.get("cargo_lag", {}).get(dep, 0)
        except Exception:
            print("error: failed to prepare local package for uploading\\n\\nCaused by:\\n"
                  f"  failed to select a version for the requirement `{dep} = \\"={s['version']}\\"`")
            sys.exit(101)
    if name in s["uploaded"]:
        print(f"error: crate {name}@{s['version']} already exists on crates.io index")
        sys.exit(101)
    s["uploaded"][name] = time.time()
    s["order"].append(name)
    json.dump(s, open(state_path, "w"))
    print(f"   Uploaded {name} v{s['version']} to registry `crates-io`")
    """)

# The old step asked the real API through curl; this stand-in answers it from the same fake registry.
FAKE_CURL = textwrap.dedent("""\
    #!/usr/bin/env python3
    import os, sys, urllib.error, urllib.request
    url = next(a for a in sys.argv[1:] if a.startswith("https://crates.io/"))
    try:
        sys.stdout.write(urllib.request.urlopen(url.replace("https://crates.io", os.environ["FAKE_REGISTRY_URL"]),
                                                timeout=5).read().decode())
    except urllib.error.HTTPError:
        sys.exit(22)
    """)


def publish_step(workflow):
    """The bash of the workflow's "Publish to crates.io" step."""
    lines = workflow.read_text().splitlines()
    start = next(i for i, line in enumerate(lines) if "name: Publish to crates.io" in line)
    run = next(i for i in range(start, len(lines)) if lines[i].strip() == "run: |")
    body = []
    for line in lines[run + 1:]:
        if line.strip() and not line.startswith(" " * 10):
            break
        body.append(line[10:])
    return "\n".join(body) + "\n"


class WorkflowStepTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="crates-publish-test-"))
        self.state = self.tmp / "state.json"
        self.deps = internal_deps()
        self.registry = Registry(self.state)
        self.server = serve(self.registry)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        bin_dir = self.tmp / "bin"
        bin_dir.mkdir()
        for name, text in (("cargo", FAKE_CARGO), ("curl", FAKE_CURL)):
            (bin_dir / name).write_text(text)
            (bin_dir / name).chmod(0o755)
        self.env = dict(os.environ, PATH=f"{bin_dir}:{os.environ['PATH']}", CARGO_REGISTRY_TOKEN="fake",
                        FAKE_REGISTRY_STATE=str(self.state), FAKE_REGISTRY_URL=self.url,
                        CRATES_PUBLISH_REGISTRY=self.url, CRATES_PUBLISH_POLL_S="0.1",
                        CRATES_PUBLISH_INDEX_WAIT_S="30", CRATES_PUBLISH_RATE_WAIT_S="0",
                        CRATES_PUBLISH_API_RETRY_S="0")
        self.crates = cp.workflow_crates(WORKFLOW)

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        subprocess.run(["rm", "-rf", str(self.tmp)], check=False)

    def run_step(self, uploaded=(), lag=None, cargo_lag=None):
        now = time.time()
        self.state.write_text(json.dumps({"version": VERSION, "deps": self.deps, "order": [],
                                          "uploaded": {n: now for n in uploaded}, "lag": lag or {},
                                          "cargo_lag": cargo_lag or {}}))
        # the old step logged to /tmp; keep every byte of this test under its own directory
        script = publish_step(WORKFLOW).replace("/tmp/", f"{self.tmp}/")
        return subprocess.run(["bash", "-e", "-c", script], cwd=ROOT, env=self.env, capture_output=True, text=True,
                              timeout=300)

    def test_the_publish_list_is_every_published_member(self):
        self.assertEqual(sorted(self.crates), sorted(n for n in self.deps if n != "memra-probe"))

    def test_a_tag_run_waits_for_a_lagging_dependency_before_its_dependents(self):
        r = self.run_step(lag={"memra-engine": 3})
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        s = self.registry.load()
        self.assertEqual(s["order"], self.crates, "not every crate was uploaded, in order")
        self.assertGreaterEqual(s["uploaded"]["memra-server"], s["uploaded"]["memra-engine"] + 3,
                                "memra-server went up before memra-engine was in the index")
        self.assertNotIn("could not resolve", r.stdout, "the dependent ran before the wait, and only the retry saved it")

    def test_a_rerun_skips_live_crates_and_waits_for_the_lagging_one(self):
        live = self.crates[:self.crates.index("memra-lanes") + 1]
        r = self.run_step(uploaded=live, lag={"memra-engine": 3})
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual(self.registry.load()["order"], ["memra-server", "memra-cli"])
        self.assertNotIn("could not resolve", r.stdout, "the live crate was not waited for before its dependent")

    def test_cargo_served_an_older_index_than_the_poll_backs_off_and_finishes(self):
        self.env["CRATES_PUBLISH_LAG_RETRY_S"] = "0.5"
        r = self.run_step(cargo_lag={"memra-engine": 2})
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual(self.registry.load()["order"], self.crates)
        self.assertIn("is listed here but not to cargo", r.stdout)
        self.assertLessEqual(r.stdout.count("could not resolve"), 5, "the retry did not back off")

    def test_a_real_run_with_no_api_answer_tries_the_upload(self):
        self.state.write_text(json.dumps({"version": VERSION, "deps": self.deps, "order": [], "uploaded": {},
                                          "lag": {}, "api_down": True}))
        script = publish_step(WORKFLOW).replace("/tmp/", f"{self.tmp}/")
        r = subprocess.run(["bash", "-e", "-c", script], cwd=ROOT, env=self.env, capture_output=True, text=True,
                           timeout=300)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual(self.registry.load()["order"], self.crates)
        self.assertEqual(r.stdout.count("the crates.io API did not answer"), len(self.crates))

    def test_an_index_that_never_lists_the_crate_fails_by_name(self):
        self.env["CRATES_PUBLISH_INDEX_WAIT_S"] = "1"
        r = self.run_step(lag={"memra-engine": 3600})
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("memra-engine@0.139.0 is not in the crates.io index after 1 s", r.stdout)
        self.assertNotIn("memra-lanes", self.registry.load()["order"], "the publish went on past the missing crate")

    def test_a_dry_run_that_cannot_reach_the_api_fails(self):
        self.state.write_text(json.dumps({"version": VERSION, "deps": self.deps, "order": [], "uploaded": {},
                                          "lag": {}, "api_down": True}))
        r = subprocess.run([sys.executable, str(TOOLS / "crates_publish.py"), "--version", VERSION, "--dry-run",
                            "--workflow", str(WORKFLOW)], cwd=ROOT, env=self.env, capture_output=True, text=True,
                           timeout=120)
        self.assertEqual(r.returncode, 1, r.stdout)
        self.assertIn("the crates.io API did not answer for memra-tier@0.139.0", r.stdout)
        self.assertNotIn("would upload", r.stdout)

    def test_the_dry_run_uploads_nothing(self):
        now = time.time()
        self.state.write_text(json.dumps({"version": VERSION, "deps": self.deps, "order": [],
                                          "uploaded": {"memra-tier": now}, "lag": {}}))
        r = subprocess.run([sys.executable, str(TOOLS / "crates_publish.py"), "--version", VERSION, "--dry-run",
                            "--workflow", str(WORKFLOW)], cwd=ROOT, env=self.env, capture_output=True, text=True,
                           timeout=120)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("memra-tier@0.139.0 is in the crates.io index", r.stdout)
        self.assertIn("memra-cli@0.139.0 is not on crates.io; a publish would upload it", r.stdout)
        self.assertEqual(self.registry.load()["order"], [])


if __name__ == "__main__":
    unittest.main()
