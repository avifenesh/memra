"""Shared input and response contract for the existing cache batteries.

Completed answers require nonempty final content and a stop terminator. A bounded
raw greedy tape may end at length for byte-exactness instrumentation only. Neither
errors nor reasoning-only/truncated chats are eligible for answer identity or for
continuation history. Mechanics receipts remain visible without becoming PASS.
"""

import hashlib
import http.client
import json
import os
from pathlib import Path
import time
import urllib.error
import urllib.request


class QualificationError(ValueError):
    pass


def load_prompt_pool(path, minimum=8, explicit=True):
    path = Path(path)
    origin = "explicit PROMPTS_JSON" if explicit else "default; set PROMPTS_JSON explicitly"
    try:
        raw = path.read_bytes()
        document = json.loads(raw)
    except (OSError, ValueError) as error:
        raise QualificationError(f"REFUSE: prompt pool {path} ({origin}): {error}") from error
    pool = document.get("decode") if isinstance(document, dict) else None
    if not isinstance(pool, list) or len(pool) < minimum:
        raise QualificationError(f"REFUSE: prompt pool {path} ({origin}) needs decode list with at least {minimum} entries")
    for index, entry in enumerate(pool):
        if not isinstance(entry, dict) or not isinstance(entry.get("text"), str) or not entry["text"].strip():
            raise QualificationError(f"REFUSE: prompt pool {path} decode[{index}] needs nonempty text")
    metadata = {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(),
                "n": len(pool), "chars": sum(len(x["text"]) for x in pool), "explicit": explicit}
    print("[pool] " + json.dumps(metadata), flush=True)
    return pool, metadata


def auth_headers():
    """A task-owned synthetic key file on qualification hosts, never a key in receipts."""
    headers = {"content-type": "application/json"}
    if "CACHE_BATTERY_KEY_FILE" in os.environ:
        path = Path(os.environ["CACHE_BATTERY_KEY_FILE"])
        try:
            key = path.read_text().strip()
        except OSError as error:
            raise QualificationError(f"REFUSE: cannot read cache battery key file {path}") from error
        if not key or any(c.isspace() for c in key):
            raise QualificationError(f"REFUSE: cache battery key file {path} needs one bearer token")
        headers["Authorization"] = "Bearer " + key
    return headers


def response_fields(document):
    if not isinstance(document, dict):
        raise QualificationError("response is not an object")
    if document.get("error") is not None or document.get("__error__") is not None:
        raise QualificationError("response contains an error")
    if "choices" in document:
        choices = document["choices"]
        if not isinstance(choices, list) or len(choices) > 1:
            raise QualificationError("expected one completion choice")
        choice = choices[0] if choices else {}
        if choice.get("error") is not None:
            raise QualificationError("choice contains an error")
        delta = choice.get("delta") or choice.get("message") or {}
        text = delta.get("content") or choice.get("text") or ""
        reasoning = delta.get("reasoning_content") or delta.get("reasoning") or ""
        finish = choice.get("finish_reason")
        usage = document.get("usage") or {}
        counts = (usage.get("prompt_tokens"), (usage.get("prompt_tokens_details") or {}).get("cached_tokens"), usage.get("completion_tokens"))
        spec = usage.get("spec") or {}
    else:
        if not any(k in document for k in ("text", "stop_reason")):
            raise QualificationError("unrecognized completion shape")
        text, reasoning = document.get("text") or "", ""
        native = document.get("stop_reason")
        finish = {"Eos": "stop", "eos": "stop", "Stop": "stop", "stop": "stop", "Callback": "stop",
                  "MaxTokens": "length", "MaxNew": "length", "ContextFull": "length", "Length": "length", "length": "length"}.get(native, native)
        counts = (document.get("prompt_tokens"), document.get("cached_tokens"), document.get("n_tokens"))
        spec = document.get("spec") or {}
    if not isinstance(text, str) or not isinstance(reasoning, str):
        raise QualificationError("completion text/reasoning is not a string")
    return text, reasoning, finish, counts, spec


def classify(text, reasoning, finish, counts, error=None, raw_tape=False):
    if error is None and (any(type(n) is not int or n < 0 for n in counts)
                          or counts[0] == 0 or counts[2] == 0 or counts[1] > counts[0]):
        error = "missing or invalid token accounting"
    if error is None and finish not in ("stop", "length"):
        error = "missing or invalid finish reason"
    completed = error is None and bool(text.strip()) and finish == "stop"
    tape = error is None and raw_tape and bool(text.strip())
    mechanics = error is None and bool(text.strip() or reasoning.strip())
    verdict = "completed" if completed else "raw_tape" if tape else "mechanics_only" if mechanics else "invalid"
    if not completed and not tape and error is None:
        error = "output budget exhausted before completed answer" if finish == "length" else "empty final content"
    return {"content": text, "reasoning": reasoning, "finish": finish,
            "prompt_tokens": counts[0], "cached_tokens": counts[1], "completion_tokens": counts[2],
            "verdict": verdict, "completed_answer": completed, "mechanics_observed": mechanics,
            "identity_eligible": completed or tape, "identity_kind": "raw_tape" if raw_tape else "completed_answer",
            "out_sha16": hashlib.sha256(text.encode()).hexdigest()[:16] if completed or tape else None,
            "error": error}


def same_identity(rows):
    """All requested rows must be eligible; dropping failures cannot create a pass."""
    return (len(rows) >= 2 and all(r["identity_eligible"] and r["out_sha16"] for r in rows)
            and len({r["identity_kind"] for r in rows}) == 1
            and len({r["out_sha16"] for r in rows}) == 1)


def append_answer(messages, row):
    if not row["completed_answer"]:
        raise QualificationError("cannot continue from an incomplete or invalid answer")
    messages.append({"role": "assistant", "content": row["content"]})


def completion(base, body, output, name, raw_tape=False):
    """Collect JSON or SSE and retain every raw byte before validating the verdict."""
    if raw_tape and ("messages" in body or body.get("stream") is not False
                     or body.get("temperature") != 0 or type(body.get("max_tokens")) is not int
                     or body["max_tokens"] <= 0):
        raise QualificationError("raw tape requires bounded non-streaming greedy /v1/completions")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / (name + "-request.json")).write_text(json.dumps(body, indent=2) + "\n")
    req = urllib.request.Request(base + ("/v1/chat/completions" if "messages" in body else "/v1/completions"),
                                 data=json.dumps(body).encode(), headers=auth_headers())
    text, reasoning, finish, counts, spec = "", "", None, (None, None, None), {}
    error, ttft, status = None, None, None
    start = time.monotonic()
    streamed = body.get("stream", False)
    raw_path = output / (name + (".sse" if streamed else ".json"))
    try:
        with urllib.request.urlopen(req, timeout=1800) as response, raw_path.open("wb") as raw:
            status = response.status
            if not 200 <= status < 300:
                raise QualificationError(f"HTTP {status}")
            if streamed:
                # Capture first, parse second: malformed/error frames must not hide
                # the rest of the bytes the server sent. Transport failures leave
                # the bytes received so far in the raw receipt, and fail the row.
                received = []
                for line in response:
                    raw.write(line)
                    received.append((line, time.monotonic() - start))
                data, done, native_terminal, protocol = [], False, False, None
                for line, arrived in received:
                    decoded = line.decode().rstrip("\r\n")
                    if decoded.startswith("data:"):
                        data.append(decoded[5:].lstrip())
                    elif not decoded and data:
                        payload = "\n".join(data)
                        data = []
                        if payload == "[DONE]":
                            if done or protocol != "openai":
                                raise QualificationError("unexpected stream terminator")
                            done = True
                            continue
                        if done or native_terminal:
                            raise QualificationError("data after stream terminator")
                        document = json.loads(payload)
                        shape = "openai" if isinstance(document, dict) and "choices" in document else "native"
                        if protocol is not None and protocol != shape:
                            raise QualificationError("mixed stream protocols or error frame")
                        protocol = shape
                        t, r, f, ns, sp = response_fields(document)
                        text += t
                        reasoning += r
                        if (t or r) and ttft is None:
                            ttft = round(arrived, 3)
                        if f is not None:
                            finish = f
                            native_terminal = shape == "native"
                        counts = tuple(new if new is not None else old for old, new in zip(counts, ns))
                        if sp:
                            spec = sp
                if data:
                    raise QualificationError("unterminated SSE event")
                if protocol == "openai" and not done:
                    raise QualificationError("OpenAI stream missing [DONE]")
                if protocol != "openai" and not native_terminal:
                    raise QualificationError("native stream missing stop_reason terminal frame")
            else:
                data = response.read()
                raw.write(data)
                text, reasoning, finish, counts, spec = response_fields(json.loads(data))
    except http.client.IncompleteRead as exc:
        with raw_path.open("ab") as raw:
            raw.write(exc.partial)
        error = "incomplete HTTP response body"
    except urllib.error.HTTPError as exc:
        status = exc.code
        raw_path.write_bytes(exc.read())
        error = f"HTTP {status}"
    except (OSError, http.client.HTTPException, ValueError, TypeError, KeyError, AttributeError) as exc:
        error = f"{type(exc).__name__}: {exc}"
    row = classify(text, reasoning, finish, counts, error, raw_tape)
    row.update(name=name, http_status=status, wall_s=round(time.monotonic() - start, 3), ttft_s=ttft, spec=spec)
    (output / (name + "-verdict.json")).write_text(json.dumps(row, indent=2) + "\n")
    return row
