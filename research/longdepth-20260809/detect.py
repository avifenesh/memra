#!/usr/bin/env python3
"""Mechanical long-generation corruption detector for the frozen Step-3.7 HTML task."""

from __future__ import annotations

import argparse
import bisect
import hashlib
from html.parser import HTMLParser
import json
import pathlib
import re
import shutil
import subprocess
import tempfile
import unicodedata


VOID_TAGS = {
    "area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta",
    "param", "source", "track", "wbr",
}


def line_starts(text: str) -> list[int]:
    starts = [0]
    starts.extend(match.end() for match in re.finditer("\n", text))
    return starts


def char_offset(starts: list[int], position: tuple[int, int]) -> int:
    line, column = position
    return starts[max(0, line - 1)] + column


class StrictHtml(HTMLParser):
    """A deliberately strict subset matching the frozen prompt's explicit-balance contract."""

    def __init__(self, source: str):
        super().__init__(convert_charrefs=False)
        self.source = source
        self.starts = line_starts(source)
        self.stack: list[tuple[str, int]] = []
        self.errors: list[dict[str, object]] = []
        self.declarations: list[str] = []
        self.scripts: list[dict[str, object]] = []
        self.styles: list[dict[str, object]] = []
        self._raw: dict[str, object] | None = None

    def here(self) -> int:
        return char_offset(self.starts, self.getpos())

    def handle_decl(self, decl: str) -> None:
        self.declarations.append(decl)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        if tag.lower() not in VOID_TAGS:
            self.errors.append({
                "kind": "non_void_self_close",
                "char_offset": self.here(),
                "detail": f"non-void <{tag}/> conflicts with explicit paired-tag contract",
            })

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        tag = tag.lower()
        at = self.here()
        if tag not in VOID_TAGS:
            self.stack.append((tag, at))
        if tag in {"script", "style"}:
            self._raw = {"tag": tag, "start": at, "text": "", "data_start": None}

    def handle_data(self, data: str) -> None:
        if self._raw is not None:
            if self._raw["data_start"] is None:
                self._raw["data_start"] = self.here()
            self._raw["text"] = str(self._raw["text"]) + data

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        at = self.here()
        if not self.stack:
            self.errors.append({
                "kind": "unmatched_end_tag",
                "char_offset": at,
                "detail": f"unmatched </{tag}>",
            })
        elif self.stack[-1][0] != tag:
            self.errors.append({
                "kind": "misnested_end_tag",
                "char_offset": at,
                "detail": f"saw </{tag}> while <{self.stack[-1][0]}> was open",
            })
            match = next((i for i in range(len(self.stack) - 1, -1, -1)
                          if self.stack[i][0] == tag), None)
            if match is not None:
                del self.stack[match:]
        else:
            self.stack.pop()
        if self._raw is not None and self._raw["tag"] == tag:
            target = self.scripts if tag == "script" else self.styles
            target.append(self._raw)
            self._raw = None


def javascript_parser_name() -> str:
    try:
        import tree_sitter  # noqa: F401
        import tree_sitter_javascript  # noqa: F401
        return "tree-sitter-javascript"
    except ImportError:
        if shutil.which("node"):
            return "node --check"
    raise RuntimeError(
        "no JavaScript parser available: install parser-requirements.txt or provide node")


def tree_sitter_script_error(text: str) -> tuple[int, str] | None:
    from tree_sitter import Language, Parser
    import tree_sitter_javascript

    language = Language(tree_sitter_javascript.language())
    parser = Parser(language)
    tree = parser.parse(text.encode())
    if not tree.root_node.has_error:
        return None

    def first_error(node):
        if node.is_error or node.is_missing:
            return node
        for child in node.children:
            if child.has_error or child.is_error or child.is_missing:
                found = first_error(child)
                if found is not None:
                    return found
        return None

    node = first_error(tree.root_node) or tree.root_node
    char_at = len(text.encode()[: node.start_byte].decode(errors="ignore"))
    return char_at, (
        f"tree-sitter-javascript syntax error node={node.type!r} "
        f"start={node.start_point} end={node.end_point} missing={node.is_missing}")


def script_error(block: dict[str, object]) -> dict[str, object] | None:
    text = str(block["text"])
    if javascript_parser_name() == "tree-sitter-javascript":
        failure = tree_sitter_script_error(text)
        if failure is None:
            return None
        relative, detail = failure
        return {
            "kind": "javascript_syntax",
            "char_offset": int(block.get("data_start") or block["start"]) + relative,
            "detail": detail,
        }
    with tempfile.NamedTemporaryFile("w", suffix=".js") as handle:
        handle.write(text)
        handle.flush()
        result = subprocess.run(
            ["node", "--check", handle.name], capture_output=True, text=True, check=False)
    if result.returncode == 0:
        return None
    match = re.search(r"\.js:(\d+)", result.stderr)
    line = int(match.group(1)) if match else 1
    relative = sum(len(part) + 1 for part in text.splitlines()[: max(0, line - 1)])
    return {
        "kind": "javascript_syntax",
        "char_offset": int(block.get("data_start") or block["start"]) + relative,
        "detail": result.stderr.strip()[-1000:],
    }


def balanced_css_error(block: dict[str, object]) -> dict[str, object] | None:
    text = str(block["text"])
    pairs = {"}": "{", "]": "[", ")": "("}
    stack: list[tuple[str, int]] = []
    quote = None
    escaped = False
    comment = False
    i = 0
    while i < len(text):
        char = text[i]
        nxt = text[i + 1] if i + 1 < len(text) else ""
        if comment:
            if char == "*" and nxt == "/":
                comment = False
                i += 2
                continue
        elif quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
        elif char == "/" and nxt == "*":
            comment = True
            i += 2
            continue
        elif char in {'"', "'"}:
            quote = char
        elif char in "{[(":
            stack.append((char, i))
        elif char in "}])":
            if not stack or stack[-1][0] != pairs[char]:
                return {
                    "kind": "css_unbalanced_delimiter",
                    "char_offset": int(block.get("data_start") or block["start"]) + i,
                    "detail": f"unexpected CSS delimiter {char!r}",
                }
            stack.pop()
        i += 1
    if quote or comment or stack:
        return {
            "kind": "css_incomplete",
            "char_offset": int(block.get("data_start") or block["start"]) + len(text),
            "detail": "complete <style> block ended with an open quote, comment, or delimiter",
        }
    return None


def forbidden_script(char: str) -> bool:
    if ord(char) < 128:
        return False
    name = unicodedata.name(char, "")
    category = unicodedata.category(char)
    return category.startswith(("L", "M")) and "LATIN" not in name


def excerpt(text: str, at: int, radius: int = 90) -> str:
    return text[max(0, at - radius): min(len(text), at + radius)].replace("\n", "\\n")


def analyze_text(text: str, stop_reason: str, doctype_prefilled: bool = False) -> dict[str, object]:
    think_end = text.find("</think>")
    search_from = think_end + len("</think>") if think_end >= 0 else 0
    candidates = [position for position in (
        text.lower().find("<!doctype html", search_from),
        text.lower().find("<html", search_from),
    ) if position >= 0]
    code_start = min(candidates) if candidates else None
    result: dict[str, object] = {
        "javascript_parser": javascript_parser_name(),
        "doctype_prefilled": doctype_prefilled,
        "think_end_char_offset": think_end if think_end >= 0 else None,
        "code_start_char_offset": code_start,
        "first_non_latin": None,
        "first_non_ascii": None,
        "parse_failure": None,
        "terminal_unclosed_tags": [],
    }
    if code_start is None:
        result["parse_failure"] = {
            "kind": "missing_html_start",
            "char_offset": search_from,
            "detail": "no <!doctype html> or <html start after the reasoning segment",
        }
        return result

    code = text[code_start:]
    for offset, char in enumerate(code):
        absolute = code_start + offset
        if ord(char) >= 128 and result["first_non_ascii"] is None:
            result["first_non_ascii"] = {
                "char": char,
                "codepoint": f"U+{ord(char):04X}",
                "name": unicodedata.name(char, "UNKNOWN"),
                "char_offset": absolute,
                "excerpt": excerpt(text, absolute),
            }
        if forbidden_script(char):
            result["first_non_latin"] = {
                "char": char,
                "codepoint": f"U+{ord(char):04X}",
                "name": unicodedata.name(char, "UNKNOWN"),
                "char_offset": absolute,
                "excerpt": excerpt(text, absolute),
            }
            break

    parser = StrictHtml(code)
    parse_errors: list[dict[str, object]] = []
    try:
        parser.feed(code)
        parser.close()
    except Exception as exc:
        parse_errors.append({
            "kind": "html_parser_exception",
            "char_offset": 0,
            "detail": f"{type(exc).__name__}: {exc}",
        })
    parse_errors.extend(parser.errors)
    if (not doctype_prefilled
            and not any(decl.lower() == "doctype html" for decl in parser.declarations)):
        parse_errors.append({
            "kind": "missing_html5_doctype",
            "char_offset": 0,
            "detail": "document does not contain an HTML5 doctype declaration",
        })
    for block in parser.styles:
        failure = balanced_css_error(block)
        if failure:
            parse_errors.append(failure)
    for block in parser.scripts:
        failure = script_error(block)
        if failure:
            parse_errors.append(failure)

    # A MaxNew response is expected to end mid-document. Only that terminal open stack/raw block
    # is forgiven; mismatched tags and syntax errors in completed blocks remain interior failures.
    result["terminal_unclosed_tags"] = [tag for tag, _ in parser.stack]
    if stop_reason.lower() not in {"maxnew", "length"} and parser.stack:
        tag, opened_at = parser.stack[-1]
        # A still-open element is not erroneous when it opens; it becomes a parse failure only
        # when a natural stop arrives without the close. Attribute the failure to the final
        # visible token rather than falsely calling the opening tag the first corrupt token.
        at = max(0, len(code) - 1)
        parse_errors.append({
            "kind": "unclosed_tag_at_stop",
            "char_offset": at,
            "detail": f"generation stopped with <{tag}> open (opened at code char {opened_at})",
        })
    if parser._raw is not None and stop_reason.lower() not in {"maxnew", "length"}:
        parse_errors.append({
            "kind": "unclosed_raw_text_element",
            "char_offset": max(0, len(code) - 1),
            "detail": f"generation stopped inside <{parser._raw['tag']}> "
                      f"(opened at code char {parser._raw['start']})",
        })

    if parse_errors:
        first = min(parse_errors, key=lambda item: int(item["char_offset"]))
        first["char_offset"] = code_start + int(first["char_offset"])
        first["excerpt"] = excerpt(text, int(first["char_offset"]))
        result["parse_failure"] = first
    return result


def token_spans(
    tok_span: pathlib.Path,
    model: pathlib.Path,
    tokens: pathlib.Path,
    completion: pathlib.Path,
    byte_offsets: list[int],
    allow_terminal_token_without_text: bool = False,
) -> tuple[dict[str, object], dict[int, dict[str, object]]]:
    def invoke(ids_path: pathlib.Path) -> subprocess.CompletedProcess[str]:
        command = [str(tok_span), str(model), str(ids_path), str(completion)]
        command.extend(map(str, byte_offsets))
        return subprocess.run(command, capture_output=True, text=True, check=False)

    run = invoke(tokens)
    terminal_token_without_text = None
    if run.returncode != 0 and allow_terminal_token_without_text:
        ids = [int(part) for part in tokens.read_text().split()]
        if ids:
            with tempfile.NamedTemporaryFile("w", suffix=".tokens.txt") as trimmed:
                trimmed.write(" ".join(map(str, ids[:-1])) + "\n")
                trimmed.flush()
                retry = invoke(pathlib.Path(trimmed.name))
            if retry.returncode == 0:
                terminal_token_without_text = ids[-1]
                run = retry
    if run.returncode != 0:
        detail = run.stderr.strip() or run.stdout.strip() or "no diagnostic output"
        raise RuntimeError(f"tok_span failed with exit {run.returncode}: {detail}")
    summary: dict[str, object] = {}
    offsets: dict[int, dict[str, object]] = {}
    for line in run.stdout.splitlines():
        fields = line.split("\t")
        if fields[0] == "summary":
            summary = {
                "token_count": int(fields[1]),
                "decoded_bytes": int(fields[2]),
                "decode_match": fields[3],
                "terminal_token_without_text": terminal_token_without_text,
            }
        elif fields[0] == "offset":
            byte_offset = int(fields[1])
            offsets[byte_offset] = {
                "completion_token_index_0based": int(fields[2]),
                "completion_token_index_1based": int(fields[2]) + 1,
                "token_id": int(fields[3]),
                "token_byte_start": int(fields[4]),
                "token_byte_end": int(fields[5]),
                "token_hex": fields[6],
                "token_text_debug": fields[7],
            }
    return summary, offsets


def run_analysis(response_path: pathlib.Path, model: pathlib.Path, tok_span_path: pathlib.Path,
                 out_path: pathlib.Path, label: str,
                 doctype_prefilled: bool = False) -> dict[str, object]:
    response = json.loads(response_path.read_text())
    text = response["text"]
    stop_reason = str(response.get("stop_reason", ""))
    analysis = analyze_text(text, stop_reason, doctype_prefilled)
    analysis.update({
        "label": label,
        "stop_reason": stop_reason,
        "server_n_tokens": response.get("n_tokens"),
        "prompt_tokens": response.get("prompt_tokens"),
        "cached_tokens": response.get("cached_tokens"),
        "completion_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "completion_bytes": len(text.encode()),
        "token_index_convention": "completion token index is 0-based; 1-based twin is included",
    })

    interesting: list[dict[str, object]] = []
    for key in ("first_non_latin", "first_non_ascii", "parse_failure"):
        value = analysis.get(key)
        if isinstance(value, dict) and isinstance(value.get("char_offset"), int):
            value["byte_offset"] = len(text[: int(value["char_offset"])].encode())
            interesting.append(value)
    unique_offsets = sorted({int(value["byte_offset"]) for value in interesting
                             if int(value["byte_offset"]) < len(text.encode())})
    summary, spans = token_spans(
        tok_span_path,
        model,
        response_path.with_name("tokens.txt"),
        response_path.with_name("completion.txt"),
        unique_offsets,
        stop_reason.lower() == "eos",
    )
    analysis["token_stream"] = summary
    for value in interesting:
        span = spans.get(int(value["byte_offset"]))
        if span:
            value.update(span)

    corruption_candidates = []
    for key in ("first_non_latin", "parse_failure"):
        value = analysis.get(key)
        if isinstance(value, dict) and "completion_token_index_0based" in value:
            corruption_candidates.append((int(value["completion_token_index_0based"]), key))
    if corruption_candidates:
        token_index, cause = min(corruption_candidates)
        analysis["first_corruption_token_index_0based"] = token_index
        analysis["first_corruption_token_index_1based"] = token_index + 1
        analysis["first_corruption_cause"] = cause
        analysis["corrupt"] = True
    else:
        analysis["first_corruption_token_index_0based"] = None
        analysis["first_corruption_token_index_1based"] = None
        analysis["first_corruption_cause"] = None
        analysis["corrupt"] = False

    out_path.write_text(json.dumps(analysis, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    return analysis


def self_test() -> None:
    clean = ("brief plan</think>\n<!doctype html><html><head><style>body { color: #111; }"
             "</style></head><body><section><h2>One</h2></section>"
             "<script>const x = 1;\nconsole.log(x);</script></body></html>")
    result = analyze_text(clean, "Eos")
    assert result["first_non_latin"] is None, result
    assert result["parse_failure"] is None, result

    unicode_bad = clean.replace("color: #111", "font-size\u5218\u5907? 14px")
    result = analyze_text(unicode_bad, "Eos")
    assert result["first_non_latin"]["char"] == "\u5218", result

    nesting_bad = clean.replace("</section>", "</div></section>")
    result = analyze_text(nesting_bad, "Eos")
    assert result["parse_failure"]["kind"] == "misnested_end_tag", result

    js_bad = clean.replace("const x = 1;", "const = 1;")
    result = analyze_text(js_bad, "Eos")
    assert result["parse_failure"]["kind"] == "javascript_syntax", result

    truncated = "plan</think>\n<!doctype html><html><body><script>const x ="
    result = analyze_text(truncated, "MaxNew")
    assert result["parse_failure"] is None, result

    eos_unclosed = "plan</think>\n<!doctype html><html><head><style>body { color: red; }"
    result = analyze_text(eos_unclosed, "Eos")
    assert result["parse_failure"]["kind"] == "unclosed_tag_at_stop", result
    assert result["parse_failure"]["char_offset"] == len(eos_unclosed) - 1, result

    continuation = clean[clean.index("<html"):]
    result = analyze_text(continuation, "Eos", doctype_prefilled=True)
    assert result["parse_failure"] is None, result
    print("detect.py self-test PASS "
          "(clean, unicode, HTML nesting, JavaScript, truncation, prefixed doctype)")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--response")
    parser.add_argument("--model")
    parser.add_argument("--tok-span")
    parser.add_argument("--out")
    parser.add_argument("--label", default="")
    parser.add_argument("--doctype-prefilled", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    required = (args.response, args.model, args.tok_span, args.out)
    if any(value is None for value in required):
        parser.error("--response, --model, --tok-span, and --out are required")
    result = run_analysis(
        pathlib.Path(args.response), pathlib.Path(args.model), pathlib.Path(args.tok_span),
        pathlib.Path(args.out), args.label, args.doctype_prefilled)
    print(json.dumps({
        "label": args.label,
        "n_tokens": result["server_n_tokens"],
        "stop_reason": result["stop_reason"],
        "corrupt": result["corrupt"],
        "first_corruption_token_index_0based": result["first_corruption_token_index_0based"],
        "cause": result["first_corruption_cause"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
