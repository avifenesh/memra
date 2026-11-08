#!/usr/bin/env python3
"""Red arms for the request-log tests. Applies to crates/memra-server/src/request_log.rs in place;
`git checkout -- crates/memra-server/src/request_log.rs` reverts.

    red_arm.py bind PATH   bind() binds nothing: the server as it was before the log could see
                           a request (middleware lines only, no session, prompt, token or tool
                           observation). request_log_two_turns_... must fail.
    red_arm.py both PATH   bind() cut, plus the streaming body wrapper ends every logged stream
                           after its first frame (a hook that alters what it observes).
                           request_log_on_and_off_serve_identical_bytes must fail too.
"""
import sys

BIND = ("    let Some(entry) = current() else {\n        return;\n    };\n    let session",
        "    let Some(entry) = current().filter(|_| false) else {\n        return;\n    };\n    let session")
BODY = ("        let next = self.inner.as_mut().poll_frame(cx);\n",
        "        if self.entry.is_none() {\n            return Poll::Ready(None);\n        }\n"
        "        let next = self.inner.as_mut().poll_frame(cx);\n"
        "        if matches!(&next, Poll::Ready(Some(Ok(_)))) {\n            self.end();\n        }\n")


def main() -> int:
    mode, path = sys.argv[1], sys.argv[2]
    src = open(path).read()
    for old, new in [BIND] + ([BODY] if mode == "both" else []):
        if new in src:
            continue
        assert src.count(old) == 1, f"anchor not found: {old[:40]!r}"
        src = src.replace(old, new)
    open(path, "w").write(src)
    print(f"red arm {mode} applied to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
