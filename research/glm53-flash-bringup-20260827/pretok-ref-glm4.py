#!/usr/bin/env python3
"""Reference + gate inputs for the `glm4` pre-tokenizer split (GLM-5.3-Flash).

Oracle: the CHECKPOINT'S OWN `tokenizer.json`, driven by HF `tokenizers` — the same engine
`transformers` delegates to, and the engine whose output defined the vocab at training time.
That is a stronger oracle than re-executing llama.cpp's algorithm in a second regex engine
(the method `research/step37-p2-20260806/pretok-ref-deepseek-v3.py` used for deepseek-v3),
and it is the same method as the step35 SKU gate (`research/step-sku-20260807/`).

The artifact is sha-pinned: tokenizer.json must hash to the sha256 recorded in this lane's
`inspect-receipts/artifact.lock` (19e77364…). The script refuses to run otherwise.

Three outputs:
  --rust        the Rust const corpus for `unicode.rs`'s `glm4_split_matches_reference`
  corpus.tsv    `<name>\t<hex of utf-8 bytes>` — hex transport, the corpus is full of
                newlines/NBSP/combining marks
  splits.tsv    `<name>\t<hex of each pre-token, space separated>` — the wide differential
                the Rust side is diffed against
  ref-ids.tsv   `<name>\t<ids add_special=true csv>\t<ids add_special=false csv>` for
                `tok-parity` (end-to-end token ids, which is the only thing that actually
                matters downstream)

Run: python3 pretok-ref-glm4.py <dir-with-tokenizer.json> [--rust]
"""
import hashlib
import pathlib
import random
import sys

from tokenizers import Tokenizer, pre_tokenizers
from tokenizers import Regex as TkRegex

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "parity-evidence"
EXPECT_SHA = "19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d"

GLM4_REGEX = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}"
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)

cases: list[tuple[str, str]] = []


def add(name: str, text: str) -> None:
    assert "\t" not in name and "\n" not in name, name
    cases.append((name, text))


# ---------------------------------------------------------------------- basics
add("empty", "")
add("single-letter", "x")
add("ascii-hello", "Hello, world!")
add("leading-space-word", " hello")
add("sentence", "The quick brown fox jumps over the lazy dog.")

# ---------------------------------------- the ONE atom: \p{N}{1,3} digit runs
# lengths 1..12 — this is the only alternative that differs from qwen2, so every
# remainder class (n%3 == 0,1,2) has to be pinned at several lengths.
for n in range(1, 13):
    add(f"digits-len{n}", "123456789012"[:n])
add("digits-14", "12345678901234")
add("digits-in-words", "abc123def4567gh89")
add("digit-letter-alternating", "a1b22c333d4444e")
add("decimal", "3.14159265358979")
add("negative", "-273.15 degrees")
add("thousands", "1,234,567.89")
add("version-string", "v2.7.18-rc3+build.4521")
add("date-iso", "2026-08-07T06:00:00Z")
add("phone", "+1 (555) 010-4477 ext. 42")
add("hex-literal", "0xDEADBEEF 0b1011 1e-9 6.022e23")
add("digits-then-newline", "123\n456")
add("space-then-digits", " 1234")
add("digits-then-space", "1234 ")
add("digits-hugging-punct", "(1234)[5678]{9012}")
# \p{N} is Nd + Nl + No, not just ASCII digits
add("arabic-indic-digits", "١٢٣٤٥٦٧")
add("fullwidth-digits", "１２３４５")
add("math-bold-digits", "\U0001d7ce\U0001d7cf\U0001d7d0\U0001d7d1")
add("roman-numerals-Nl", "ⅨⅨⅨⅨ")
add("circled-digits-No", "①②③④")
add("fractions-No", "½½½½")
add("superscript-No", "x²²²² + y³")
add("mixed-script-digits", "12٣٤ 5６")

# --------------------------------------------------------------- contractions
add("contractions-lower", "don't can't we're I've I'm you'll he'd")
add("contractions-upper", "DON'T CAN'T WE'RE I'VE I'M YOU'LL HE'D")
add("contractions-mixed", "We'Ve a'lL It'S tHeY'rE")
add("contraction-long-s", "'ſx and 'ſ")  # (?i:) folds U+017F onto 's'
add("apostrophe-not-contraction", "'q 'z '9 ' '")
add("quote-then-word", "'quoted' \"double\"")

# ------------------------------------------------- combining marks (\p{L} vs \p{L}\p{M})
# The alternative that separates glm4 from memra's qwen35 machine: `\p{L}+` stops at a
# combining mark and the mark falls through to ` ?[^\s\p{L}\p{N}]+`.
add("nfd-e-acute", "café")
add("nfc-e-acute", "café")
add("mark-leading", "́abc")
add("mark-interior", "x́y")
add("mark-runs", "áb́ć")
add("mark-double", "á̈b")
add("arabic-harakat", "مُحَمَّد")
add("hebrew-niqqud", "שָׁלוֹם")
add("devanagari-matras", "हिन्दी")
add("mark-then-digit", "á1234")
add("mark-then-space", "á b")

# ------------------------------------------------------------------ whitespace
add("leading-trailing-spaces", "   leading and trailing spaces   ")
add("interior-double-space", "a  b")
add("interior-triple-space", "a   b")
add("space-only-1", " ")
add("space-only-2", "  ")
add("space-only-8", "        ")
add("tabs", "tabs\tand\t\tspaces   x")
add("newline-single", "\n")
add("newline-run", "\n\n\n")
add("crlf", "line1\r\nline2\r\n\r\nline4")
add("cr-only", "\r\r\n\n")
add("ws-then-newline", "x  \n\n  y")
add("newline-then-ws", "\n\n  \n indented")
add("trailing-newlines", "trailing newlines\n\n\n")
add("space-before-eof", "end with space ")
add("nbsp-single", "a b")
add("nbsp-double", "a  b")
add("ideographic-space", "a　　b")
add("line-separator", "a  b")
add("mongolian-vowel-sep", "a᠎᠎b")  # NOT \s — falls to the complement run
add("zwsp", "a​b")
add("form-feed-vtab", "a\x0c\x0bb")

# --------------------------------------------------------- CJK / mixed scripts
add("cjk-sentence", "中文测试")
add("cjk-mixed-digits", "中1文2")
add("japanese", "日本語のテスト、カタカナ")
add("korean", "한국어 테스트")
add("cyrillic", "ЖИВЁТ русский")
add("greek", "Ελληνικά κείμενα")
add("arabic", "العربية نص")
add("mixed-scripts", "混合 English 中文 123 рус")
add("thai", "ภาษาไทย 123")

# ---------------------------------------------------------------- emoji / symbols
add("emoji-run", "\U0001f680\U0001f525✅")
add("emoji-zwj", "\U0001f636‍\U0001f32b️")
add("emoji-with-text", "emoji test \U0001f680\U0001f525✅ and math ∑∫√π≠≤")
add("skin-tone", "\U0001f44d\U0001f3fd")
add("regional-indicators", "\U0001f1fa\U0001f1f8\U0001f1e8\U0001f1f3")
add("symbols-spaced", "symbols ~ ^ | $ + = < >")
add("punct-run", "@#$%^&*()")
add("punct-heavy", "''''''```````\"\"\"\"......!!!!!!??????")
add("lower-eighth-block", "▁escaped▁space")
add("unassigned-tag-cpt", "a\U000e0001b")

# ---------------------------------------------------------------------- code
add("rust-fn", 'fn main() { let x: i32 = 42; println!("{}", x*2); }')
add("json", '{"key": [1, 2, 3], "n": 12345}')
add("path", "path/to/file-00042.gguf")
add("identifiers", "snake_case camelCase kebab-case SCREAMING_SNAKE_2")
add("markdown-fence", "```python\nprint(1234)\n```\n")
add("html", "<div class=\"x\" id=\"row-17\">text</div>")
add("chatml", "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n")

# --------------------------------------------------- llama.cpp's own chkhsh probe string
add(
    "llamacpp-chktxt",
    "\n \n\n \n\n\n \t \t\t \t\n  \n   \n    \n     \n\U0001f680 (normal) "
    "\U0001f636‍\U0001f32b️ (multiple emojis concatenated) ✅ "
    "\U0001f999\U0001f999 3 33 333 3333 33333 333333 3333333 33333333 3.3 3..3 3...3 "
    "កាន់តែពិសេសអាច"
    "\U0001f601 ?我想在apple工作1314151天～ ------======= "
    "нещо на Български "
    "''''''```````\"\"\"\"......!!!!!!?????? I've been 'told he's there, 'RE you sure? "
    "'M not sure I'll make it, 'D you like some tea? We'Ve a'lL",
)

# ------------------------------------------- special / control token literals
# Layer 2 of the step-sku method: every real server request is a rendered template full of
# these literals, and they take the `tokenizer_st_partition` branch — a different code path
# from the pre-tokenizer split. Each must resolve to ONE id on both sides, and the text
# around the boundary must split the same way.
add("st-endoftext", "<|endoftext|>")
add("st-gmask-sop", "[gMASK]<sop>")
add("st-role-turn", "<|system|>You are helpful.<|user|>hi 1234<|assistant|>")
add("st-think", "<think>reasoning 42</think>answer")
add("st-tool-call", "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value></tool_call>")
add("st-tool-response", "<tool_response>{\"temp\": 21}</tool_response>")
add("st-observation", "<|observation|>result 007<|assistant|>")
add("st-code-fim", "<|code_prefix|>def f(x):<|code_suffix|>return x<|code_middle|>")
add("st-nothink", "/nothink what is 2+2?")
add("st-box", "<|begin_of_box|>1234<|end_of_box|>")
add("st-nonspecial-mask", "[MASK][sMASK]<eop>")
add("st-glued", "a<|user|>1<|assistant|>2")
add("st-partial-literal", "<|user and |assistant|> and <|nope|>")
add("st-adjacent-digits", "<|user|>1234<|assistant|>5678")

# ---------------------------------------------------------------- long uniform runs
add("digits-3000", "7" * 3000)
add("letters-3000", "Z" * 3000)
add("spaces-3000", " " * 3000)
add("spaces-3000-then-x", " " * 3000 + "x")

# ------------------------------------------------------------------- fuzz layer
# Deterministic random mixing of every class that can start an alternative, so the
# differential is not limited to hand-picked shapes.
ALPHABET = list(
    "abcXYZ019 \t\n\r.,!?'\"-_/*"
    " 　​ُ́̈中日рα٣１①½"
    "\U0001f680▁ſ᠎"
)
rng = random.Random(20260827)
for i in range(400):
    n = rng.randint(1, 40)
    add(f"fuzz-{i:03d}", "".join(rng.choice(ALPHABET) for _ in range(n)))


def _add_template_renders() -> None:
    """Layer 2b: the checkpoint's own chat_template.jinja, rendered under the exact HF
    environment (trim_blocks / lstrip_blocks False, keep_trailing_newline True — jinja2's
    defaults, which is what `transformers` uses). Tokenizing a real render end-to-ends the
    serve path by composition: the rendered string is what a server actually encodes."""
    import jinja2

    tmpl_path = HERE / "chat_template.jinja"
    if not tmpl_path.exists():
        print(f"note: {tmpl_path} absent, skipping template renders", file=sys.stderr)
        return
    # `transformers` enables loopcontrols (break/continue) and do; the GLM template uses
    # {% break %}, so a bare Environment cannot compile it.
    env = jinja2.Environment(
        keep_trailing_newline=True,
        extensions=["jinja2.ext.loopcontrols", "jinja2.ext.do"],
    )
    # `transformers` overrides tojson so the template can pass ensure_ascii=False
    def _tojson(x, indent=None, ensure_ascii=False, **kw):
        import json as _json
        return _json.dumps(x, indent=indent, ensure_ascii=ensure_ascii, **kw)

    env.filters["tojson"] = _tojson
    tmpl = env.from_string(tmpl_path.read_text())
    convos = [
        (
            "tmpl-simple",
            {
                "messages": [
                    {"role": "user", "content": "What is 12345 divided by 3?"},
                ],
                "add_generation_prompt": True,
            },
        ),
        (
            "tmpl-multiturn",
            {
                "messages": [
                    {"role": "system", "content": "You are terse."},
                    {"role": "user", "content": "café résumé, 2026-08-28"},
                    {"role": "assistant", "content": "Noted: 1,234 items."},
                    {"role": "user", "content": "\u4e2d\u6587 and \U0001f680 too?"},
                ],
                "add_generation_prompt": True,
            },
        ),
        (
            "tmpl-tools",
            {
                "messages": [{"role": "user", "content": "weather in Paris?"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "description": "Get weather for a city",
                            "parameters": {
                                "type": "object",
                                "properties": {"city": {"type": "string"}},
                                "required": ["city"],
                            },
                        },
                    }
                ],
                "add_generation_prompt": True,
            },
        ),
    ]
    for name, kw in convos:
        try:
            add(name, tmpl.render(**kw))
        except Exception as e:  # a render failure must be LOUD, not a silently missing case
            print(f"TEMPLATE RENDER FAILED for {name}: {e}", file=sys.stderr)
            raise


_add_template_renders()


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    want_rust = "--rust" in sys.argv
    ref_dir = pathlib.Path(args[0] if args else "/tmp/glm53-hf-ref")
    tj = ref_dir / "tokenizer.json"
    sha = hashlib.sha256(tj.read_bytes()).hexdigest()
    if sha != EXPECT_SHA:
        print(f"REFUSE: {tj} sha256 {sha} != pinned {EXPECT_SHA}", file=sys.stderr)
        return 2

    tok = Tokenizer.from_file(str(tj))
    declared = tok.pre_tokenizer.__getstate__() if hasattr(tok.pre_tokenizer, "__getstate__") else None
    split = pre_tokenizers.Split(TkRegex(GLM4_REGEX), behavior="isolated", invert=False)

    names = [n for n, _ in cases]
    assert len(names) == len(set(names)), "duplicate case name"

    (OUT / "corpus.tsv").write_text(
        "".join(f"{n}\t{t.encode().hex()}\n" for n, t in cases)
    )

    rows = []
    for name, text in cases:
        words = [w for w, _ in split.pre_tokenize_str(text)]
        assert "".join(words) == text, f"{name}: reassembly"
        rows.append((name, words))
    (OUT / "splits.tsv").write_text(
        "".join(f"{n}\t{' '.join(w.encode().hex() for w in ws)}\n" for n, ws in rows)
    )

    idrows = []
    for name, text in cases:
        idrows.append(
            (
                name,
                tok.encode(text, add_special_tokens=True).ids,
                tok.encode(text, add_special_tokens=False).ids,
            )
        )
    (OUT / "ref-ids.tsv").write_text(
        "".join(
            f"{n}\t{','.join(map(str, s))}\t{','.join(map(str, p))}\n" for n, s, p in idrows
        )
    )

    print(f"tokenizer.json sha256 {sha} (pinned OK)")
    print(f"declared pre_tokenizer: {declared}")
    print(f"{len(cases)} cases -> corpus.tsv, splits.tsv, ref-ids.tsv")
    print(f"  total pre-tokens {sum(len(w) for _, w in rows)}, "
          f"total ids(plain) {sum(len(p) for _, _, p in idrows)}")

    if want_rust:
        # the hand-picked cases only (the fuzz layer stays in splits.tsv — it is a
        # differential input, not something to read in a source file)
        def lit(s: str) -> str:
            out = '"'
            for ch in s:
                if ch == '"':
                    out += '\\"'
                elif ch == "\\":
                    out += "\\\\"
                elif ch == "\n":
                    out += "\\n"
                elif ch == "\r":
                    out += "\\r"
                elif ch == "\t":
                    out += "\\t"
                elif ord(ch) < 0x20 or ord(ch) == 0x7F or not ch.isprintable():
                    out += "\\u{%x}" % ord(ch)
                else:
                    out += ch
            return out + '"'

        picked = [(n, ws) for n, ws in rows if not n.startswith(("fuzz-", "digits-3000", "letters-3000", "spaces-3000"))]
        lines = []
        for name, ws in picked:
            text = dict(cases)[name]
            lines.append(
                f"    // {name}\n    ({lit(text)}, &[{', '.join(lit(w) for w in ws)}]),"
            )
        out = OUT / "glm4-cases.rs"
        out.write_text(
            "const GLM4_CASES: &[(&str, &[&str])] = &[\n" + "\n".join(lines) + "\n];\n"
        )
        print(f"wrote {out} ({len(picked)} cases)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
