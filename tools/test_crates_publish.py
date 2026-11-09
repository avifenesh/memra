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


class PublishRetryTests(unittest.TestCase):
    def setUp(self):
        self.c = FakeClock()
        self.waited = []
        self.logs = []

    def publish(self, *outcomes):
        outs = list(outcomes)
        calls = []

        def run(name):
            calls.append(name)
            return outs.pop(0)

        result = cp.publish("memra-server", VERSION, run=run, wait=lambda n, v: self.waited.append((n, v)),
                            sleep=self.c.sleep, log=self.logs.append)
        return result, calls

    def test_rate_limit_waits_the_new_crate_window_and_retries(self):
        result, calls = self.publish((101, "error: 429 Too Many Requests"), (0, "Uploaded"))
        self.assertEqual((result, len(calls)), ("published", 2))
        self.assertEqual(self.c.sleeps, [cp.RATE_LIMIT_WAIT_S])

    def test_a_dependency_missing_from_the_index_is_waited_for_then_retried(self):
        lag = 'failed to select a version for the requirement `memra-engine = "=0.139.0"`'
        result, calls = self.publish((101, lag), (0, "Uploaded"))
        self.assertEqual((result, len(calls)), ("published", 2))
        self.assertEqual(self.waited, [("memra-engine", VERSION)])

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
        with self.assertRaisesRegex(cp.PublishError, "after 6 attempts"):
            self.publish(*[(101, "429 Too Many Requests")] * cp.ATTEMPTS)
        self.assertEqual(len(self.c.sleeps), cp.ATTEMPTS)


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
            if parts[:3] == ["api", "v1", "crates"] and len(parts) == 5:
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
                        CRATES_PUBLISH_INDEX_WAIT_S="30", CRATES_PUBLISH_RATE_WAIT_S="0")
        self.crates = cp.workflow_crates(WORKFLOW)

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        subprocess.run(["rm", "-rf", str(self.tmp)], check=False)

    def run_step(self, uploaded=(), lag=None):
        now = time.time()
        self.state.write_text(json.dumps({"version": VERSION, "deps": self.deps, "order": [],
                                          "uploaded": {n: now for n in uploaded}, "lag": lag or {}}))
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

    def test_a_rerun_skips_live_crates_and_waits_for_the_lagging_one(self):
        live = self.crates[:self.crates.index("memra-lanes") + 1]
        r = self.run_step(uploaded=live, lag={"memra-engine": 3})
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual(self.registry.load()["order"], ["memra-server", "memra-cli"])

    def test_an_index_that_never_lists_the_crate_fails_by_name(self):
        self.env["CRATES_PUBLISH_INDEX_WAIT_S"] = "1"
        r = self.run_step(lag={"memra-engine": 3600})
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("memra-engine@0.139.0 is not in the crates.io index after 1 s", r.stdout)
        self.assertNotIn("memra-lanes", self.registry.load()["order"], "the publish went on past the missing crate")

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
