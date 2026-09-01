const GLM4_CASES: &[(&str, &[&str])] = &[
    // empty
    ("", &[]),
    // single-letter
    ("x", &["x"]),
    // ascii-hello
    ("Hello, world!", &["Hello", ",", " world", "!"]),
    // leading-space-word
    (" hello", &[" hello"]),
    // sentence
    ("The quick brown fox jumps over the lazy dog.", &["The", " quick", " brown", " fox", " jumps", " over", " the", " lazy", " dog", "."]),
    // digits-len1
    ("1", &["1"]),
    // digits-len2
    ("12", &["12"]),
    // digits-len3
    ("123", &["123"]),
    // digits-len4
    ("1234", &["123", "4"]),
    // digits-len5
    ("12345", &["123", "45"]),
    // digits-len6
    ("123456", &["123", "456"]),
    // digits-len7
    ("1234567", &["123", "456", "7"]),
    // digits-len8
    ("12345678", &["123", "456", "78"]),
    // digits-len9
    ("123456789", &["123", "456", "789"]),
    // digits-len10
    ("1234567890", &["123", "456", "789", "0"]),
    // digits-len11
    ("12345678901", &["123", "456", "789", "01"]),
    // digits-len12
    ("123456789012", &["123", "456", "789", "012"]),
    // digits-14
    ("12345678901234", &["123", "456", "789", "012", "34"]),
    // digits-in-words
    ("abc123def4567gh89", &["abc", "123", "def", "456", "7", "gh", "89"]),
    // digit-letter-alternating
    ("a1b22c333d4444e", &["a", "1", "b", "22", "c", "333", "d", "444", "4", "e"]),
    // decimal
    ("3.14159265358979", &["3", ".", "141", "592", "653", "589", "79"]),
    // negative
    ("-273.15 degrees", &["-", "273", ".", "15", " degrees"]),
    // thousands
    ("1,234,567.89", &["1", ",", "234", ",", "567", ".", "89"]),
    // version-string
    ("v2.7.18-rc3+build.4521", &["v", "2", ".", "7", ".", "18", "-rc", "3", "+build", ".", "452", "1"]),
    // date-iso
    ("2026-08-07T06:00:00Z", &["202", "6", "-", "08", "-", "07", "T", "06", ":", "00", ":", "00", "Z"]),
    // phone
    ("+1 (555) 010-4477 ext. 42", &["+", "1", " (", "555", ")", " ", "010", "-", "447", "7", " ext", ".", " ", "42"]),
    // hex-literal
    ("0xDEADBEEF 0b1011 1e-9 6.022e23", &["0", "xDEADBEEF", " ", "0", "b", "101", "1", " ", "1", "e", "-", "9", " ", "6", ".", "022", "e", "23"]),
    // digits-then-newline
    ("123\n456", &["123", "\n", "456"]),
    // space-then-digits
    (" 1234", &[" ", "123", "4"]),
    // digits-then-space
    ("1234 ", &["123", "4", " "]),
    // digits-hugging-punct
    ("(1234)[5678]{9012}", &["(", "123", "4", ")[", "567", "8", "]{", "901", "2", "}"]),
    // arabic-indic-digits
    ("١٢٣٤٥٦٧", &["١٢٣", "٤٥٦", "٧"]),
    // fullwidth-digits
    ("１２３４５", &["１２３", "４５"]),
    // math-bold-digits
    ("𝟎𝟏𝟐𝟑", &["𝟎𝟏𝟐", "𝟑"]),
    // roman-numerals-Nl
    ("ⅨⅨⅨⅨ", &["ⅨⅨⅨ", "Ⅸ"]),
    // circled-digits-No
    ("①②③④", &["①②③", "④"]),
    // fractions-No
    ("½½½½", &["½½½", "½"]),
    // superscript-No
    ("x²²²² + y³", &["x", "²²²", "²", " +", " y", "³"]),
    // mixed-script-digits
    ("12٣٤ 5６", &["12٣", "٤", " ", "5６"]),
    // contractions-lower
    ("don't can't we're I've I'm you'll he'd", &["don", "'t", " can", "'t", " we", "'re", " I", "'ve", " I", "'m", " you", "'ll", " he", "'d"]),
    // contractions-upper
    ("DON'T CAN'T WE'RE I'VE I'M YOU'LL HE'D", &["DON", "'T", " CAN", "'T", " WE", "'RE", " I", "'VE", " I", "'M", " YOU", "'LL", " HE", "'D"]),
    // contractions-mixed
    ("We'Ve a'lL It'S tHeY'rE", &["We", "'Ve", " a", "'lL", " It", "'S", " tHeY", "'rE"]),
    // contraction-long-s
    ("'ſx and 'ſ", &["'ſ", "x", " and", " '", "ſ"]),
    // apostrophe-not-contraction
    ("'q 'z '9 ' '", &["'q", " '", "z", " '", "9", " '", " '"]),
    // quote-then-word
    ("'quoted' \"double\"", &["'quoted", "'", " \"", "double", "\""]),
    // nfd-e-acute
    ("café", &["cafe", "́"]),
    // nfc-e-acute
    ("café", &["café"]),
    // mark-leading
    ("́abc", &["́abc"]),
    // mark-interior
    ("x́y", &["x", "́y"]),
    // mark-runs
    ("áb́ć", &["a", "́b", "́c", "́"]),
    // mark-double
    ("á̈b", &["a", "́̈", "b"]),
    // arabic-harakat
    ("مُحَمَّد", &["م", "ُح", "َم", "َّ", "د"]),
    // hebrew-niqqud
    ("שָׁלוֹם", &["ש", "ָׁ", "לו", "ֹם"]),
    // devanagari-matras
    ("हिन्दी", &["ह", "िन", "्द", "ी"]),
    // mark-then-digit
    ("á1234", &["a", "́", "123", "4"]),
    // mark-then-space
    ("á b", &["a", "́", " b"]),
    // leading-trailing-spaces
    ("   leading and trailing spaces   ", &["  ", " leading", " and", " trailing", " spaces", "   "]),
    // interior-double-space
    ("a  b", &["a", " ", " b"]),
    // interior-triple-space
    ("a   b", &["a", "  ", " b"]),
    // space-only-1
    (" ", &[" "]),
    // space-only-2
    ("  ", &["  "]),
    // space-only-8
    ("        ", &["        "]),
    // tabs
    ("tabs\tand\t\tspaces   x", &["tabs", "\tand", "\t", "\tspaces", "  ", " x"]),
    // newline-single
    ("\n", &["\n"]),
    // newline-run
    ("\n\n\n", &["\n\n\n"]),
    // crlf
    ("line1\r\nline2\r\n\r\nline4", &["line", "1", "\r\n", "line", "2", "\r\n\r\n", "line", "4"]),
    // cr-only
    ("\r\r\n\n", &["\r\r\n\n"]),
    // ws-then-newline
    ("x  \n\n  y", &["x", "  \n\n", " ", " y"]),
    // newline-then-ws
    ("\n\n  \n indented", &["\n\n  \n", " indented"]),
    // trailing-newlines
    ("trailing newlines\n\n\n", &["trailing", " newlines", "\n\n\n"]),
    // space-before-eof
    ("end with space ", &["end", " with", " space", " "]),
    // nbsp-single
    ("a\u{a0}b", &["a", "\u{a0}b"]),
    // nbsp-double
    ("a\u{a0}\u{a0}b", &["a", "\u{a0}", "\u{a0}b"]),
    // ideographic-space
    ("a\u{3000}\u{3000}b", &["a", "\u{3000}", "\u{3000}b"]),
    // line-separator
    ("a\u{2028}\u{2028}b", &["a", "\u{2028}", "\u{2028}b"]),
    // mongolian-vowel-sep
    ("a\u{180e}\u{180e}b", &["a", "\u{180e}\u{180e}", "b"]),
    // zwsp
    ("a\u{200b}b", &["a", "\u{200b}b"]),
    // form-feed-vtab
    ("a\u{c}\u{b}b", &["a", "\u{c}", "\u{b}b"]),
    // cjk-sentence
    ("中文测试", &["中文测试"]),
    // cjk-mixed-digits
    ("中1文2", &["中", "1", "文", "2"]),
    // japanese
    ("日本語のテスト、カタカナ", &["日本語のテスト", "、カタカナ"]),
    // korean
    ("한국어 테스트", &["한국어", " 테스트"]),
    // cyrillic
    ("ЖИВЁТ русский", &["ЖИВЁТ", " русский"]),
    // greek
    ("Ελληνικά κείμενα", &["Ελληνικά", " κείμενα"]),
    // arabic
    ("العربية نص", &["العربية", " نص"]),
    // mixed-scripts
    ("混合 English 中文 123 рус", &["混合", " English", " 中文", " ", "123", " рус"]),
    // thai
    ("ภาษาไทย 123", &["ภาษาไทย", " ", "123"]),
    // emoji-run
    ("🚀🔥✅", &["🚀🔥✅"]),
    // emoji-zwj
    ("😶\u{200d}🌫️", &["😶\u{200d}🌫️"]),
    // emoji-with-text
    ("emoji test 🚀🔥✅ and math ∑∫√π≠≤", &["emoji", " test", " 🚀🔥✅", " and", " math", " ∑∫√", "π", "≠≤"]),
    // skin-tone
    ("👍🏽", &["👍🏽"]),
    // regional-indicators
    ("🇺🇸🇨🇳", &["🇺🇸🇨🇳"]),
    // symbols-spaced
    ("symbols ~ ^ | $ + = < >", &["symbols", " ~", " ^", " |", " $", " +", " =", " <", " >"]),
    // punct-run
    ("@#$%^&*()", &["@#$%^&*()"]),
    // punct-heavy
    ("''''''```````\"\"\"\"......!!!!!!??????", &["''''''```````\"\"\"\"......!!!!!!??????"]),
    // lower-eighth-block
    ("▁escaped▁space", &["▁escaped", "▁space"]),
    // unassigned-tag-cpt
    ("a\u{e0001}b", &["a", "\u{e0001}b"]),
    // rust-fn
    ("fn main() { let x: i32 = 42; println!(\"{}\", x*2); }", &["fn", " main", "()", " {", " let", " x", ":", " i", "32", " =", " ", "42", ";", " println", "!(\"{}\",", " x", "*", "2", ");", " }"]),
    // json
    ("{\"key\": [1, 2, 3], \"n\": 12345}", &["{\"", "key", "\":", " [", "1", ",", " ", "2", ",", " ", "3", "],", " \"", "n", "\":", " ", "123", "45", "}"]),
    // path
    ("path/to/file-00042.gguf", &["path", "/to", "/file", "-", "000", "42", ".gguf"]),
    // identifiers
    ("snake_case camelCase kebab-case SCREAMING_SNAKE_2", &["snake", "_case", " camelCase", " kebab", "-case", " SCREAMING", "_SNAKE", "_", "2"]),
    // markdown-fence
    ("```python\nprint(1234)\n```\n", &["```", "python", "\n", "print", "(", "123", "4", ")\n", "```\n"]),
    // html
    ("<div class=\"x\" id=\"row-17\">text</div>", &["<div", " class", "=\"", "x", "\"", " id", "=\"", "row", "-", "17", "\">", "text", "</", "div", ">"]),
    // chatml
    ("<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n", &["<|", "im", "_start", "|>", "user", "\n", "What", " is", " ", "2", "+", "2", "?<|", "im", "_end", "|>\n", "<|", "im", "_start", "|>", "assistant", "\n"]),
    // llamacpp-chktxt
    ("\n \n\n \n\n\n \t \t\t \t\n  \n   \n    \n     \n🚀 (normal) 😶\u{200d}🌫️ (multiple emojis concatenated) ✅ 🦙🦙 3 33 333 3333 33333 333333 3333333 33333333 3.3 3..3 3...3 កាន់តែពិសេសអាច😁 ?我想在apple工作1314151天～ ------======= нещо на Български ''''''```````\"\"\"\"......!!!!!!?????? I've been 'told he's there, 'RE you sure? 'M not sure I'll make it, 'D you like some tea? We'Ve a'lL", &["\n \n\n \n\n\n \t \t\t \t\n  \n   \n    \n     \n", "🚀", " (", "normal", ")", " 😶\u{200d}🌫️", " (", "multiple", " emojis", " concatenated", ")", " ✅", " 🦙🦙", " ", "3", " ", "33", " ", "333", " ", "333", "3", " ", "333", "33", " ", "333", "333", " ", "333", "333", "3", " ", "333", "333", "33", " ", "3", ".", "3", " ", "3", "..", "3", " ", "3", "...", "3", " ក", "ាន", "់ត", "ែព", "ិស", "េសអ", "ាច", "😁", " ?", "我想在apple工作", "131", "415", "1", "天", "～", " ------=======", " нещо", " на", " Български", " ''''''```````\"\"\"\"......!!!!!!??????", " I", "'ve", " been", " '", "told", " he", "'s", " there", ",", " '", "RE", " you", " sure", "?", " '", "M", " not", " sure", " I", "'ll", " make", " it", ",", " '", "D", " you", " like", " some", " tea", "?", " We", "'Ve", " a", "'lL"]),
    // st-endoftext
    ("<|endoftext|>", &["<|", "endoftext", "|>"]),
    // st-gmask-sop
    ("[gMASK]<sop>", &["[gMASK", "]<", "sop", ">"]),
    // st-role-turn
    ("<|system|>You are helpful.<|user|>hi 1234<|assistant|>", &["<|", "system", "|>", "You", " are", " helpful", ".<|", "user", "|>", "hi", " ", "123", "4", "<|", "assistant", "|>"]),
    // st-think
    ("<think>reasoning 42</think>answer", &["<think", ">reasoning", " ", "42", "</", "think", ">answer"]),
    // st-tool-call
    ("<tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value></tool_call>", &["<tool", "_call", ">get", "_weather", "<arg", "_key", ">city", "</", "arg", "_key", "><", "arg", "_value", ">Paris", "</", "arg", "_value", "></", "tool", "_call", ">"]),
    // st-tool-response
    ("<tool_response>{\"temp\": 21}</tool_response>", &["<tool", "_response", ">{\"", "temp", "\":", " ", "21", "}</", "tool", "_response", ">"]),
    // st-observation
    ("<|observation|>result 007<|assistant|>", &["<|", "observation", "|>", "result", " ", "007", "<|", "assistant", "|>"]),
    // st-code-fim
    ("<|code_prefix|>def f(x):<|code_suffix|>return x<|code_middle|>", &["<|", "code", "_prefix", "|>", "def", " f", "(x", "):<|", "code", "_suffix", "|>", "return", " x", "<|", "code", "_middle", "|>"]),
    // st-nothink
    ("/nothink what is 2+2?", &["/nothink", " what", " is", " ", "2", "+", "2", "?"]),
    // st-box
    ("<|begin_of_box|>1234<|end_of_box|>", &["<|", "begin", "_of", "_box", "|>", "123", "4", "<|", "end", "_of", "_box", "|>"]),
    // st-nonspecial-mask
    ("[MASK][sMASK]<eop>", &["[MASK", "][", "sMASK", "]<", "eop", ">"]),
    // st-glued
    ("a<|user|>1<|assistant|>2", &["a", "<|", "user", "|>", "1", "<|", "assistant", "|>", "2"]),
    // st-partial-literal
    ("<|user and |assistant|> and <|nope|>", &["<|", "user", " and", " |", "assistant", "|>", " and", " <|", "nope", "|>"]),
    // st-adjacent-digits
    ("<|user|>1234<|assistant|>5678", &["<|", "user", "|>", "123", "4", "<|", "assistant", "|>", "567", "8"]),
    // tmpl-simple
    ("[gMASK]<sop><|system|>Reasoning Effort: Max<|user|>What is 12345 divided by 3?<|assistant|><think>", &["[gMASK", "]<", "sop", "><|", "system", "|>", "Reasoning", " Effort", ":", " Max", "<|", "user", "|>", "What", " is", " ", "123", "45", " divided", " by", " ", "3", "?<|", "assistant", "|><", "think", ">"]),
    // tmpl-multiturn
    ("[gMASK]<sop><|system|>Reasoning Effort: Max<|system|>You are terse.<|user|>café résumé, 2026-08-28<|assistant|><think></think>Noted: 1,234 items.<|user|>中文 and 🚀 too?<|assistant|><think>", &["[gMASK", "]<", "sop", "><|", "system", "|>", "Reasoning", " Effort", ":", " Max", "<|", "system", "|>", "You", " are", " terse", ".<|", "user", "|>", "café", " résumé", ",", " ", "202", "6", "-", "08", "-", "28", "<|", "assistant", "|><", "think", "></", "think", ">Noted", ":", " ", "1", ",", "234", " items", ".<|", "user", "|>", "中文", " and", " 🚀", " too", "?<|", "assistant", "|><", "think", ">"]),
    // tmpl-tools
    ("[gMASK]<sop><|system|>Reasoning Effort: Max<|system|>\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n\n{\"name\": \"get_weather\", \"description\": \"Get weather for a city\", \"parameters\": {\"type\": \"object\", \"properties\": {\"city\": {\"type\": \"string\"}}, \"required\": [\"city\"]}}\n\n\n</tools>\n\nFor each function call, output the function name and arguments within the following XML format:\n<tool_call>{function-name}<arg_key>{arg-key-1}</arg_key><arg_value>{arg-value-1}</arg_value><arg_key>{arg-key-2}</arg_key><arg_value>{arg-value-2}</arg_value>...</tool_call><|user|>weather in Paris?<|assistant|><think>", &["[gMASK", "]<", "sop", "><|", "system", "|>", "Reasoning", " Effort", ":", " Max", "<|", "system", "|>\n", "#", " Tools", "\n\n", "You", " may", " call", " one", " or", " more", " functions", " to", " assist", " with", " the", " user", " query", ".\n\n", "You", " are", " provided", " with", " function", " signatures", " within", " <", "tools", "></", "tools", ">", " XML", " tags", ":\n", "<tools", ">\n\n", "{\"", "name", "\":", " \"", "get", "_weather", "\",", " \"", "description", "\":", " \"", "Get", " weather", " for", " a", " city", "\",", " \"", "parameters", "\":", " {\"", "type", "\":", " \"", "object", "\",", " \"", "properties", "\":", " {\"", "city", "\":", " {\"", "type", "\":", " \"", "string", "\"}},", " \"", "required", "\":", " [\"", "city", "\"]}}\n\n\n", "</", "tools", ">\n\n", "For", " each", " function", " call", ",", " output", " the", " function", " name", " and", " arguments", " within", " the", " following", " XML", " format", ":\n", "<tool", "_call", ">{", "function", "-name", "}<", "arg", "_key", ">{", "arg", "-key", "-", "1", "}</", "arg", "_key", "><", "arg", "_value", ">{", "arg", "-value", "-", "1", "}</", "arg", "_value", "><", "arg", "_key", ">{", "arg", "-key", "-", "2", "}</", "arg", "_key", "><", "arg", "_value", ">{", "arg", "-value", "-", "2", "}</", "arg", "_value", ">...</", "tool", "_call", "><|", "user", "|>", "weather", " in", " Paris", "?<|", "assistant", "|><", "think", ">"]),
];
