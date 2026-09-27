//! Bounded lexical routing of the latest user instruction.
//!
//! The caller supplies measured depths and the runtime's legal/admitted ceiling.
//! This function does not change model state, sampling, or a running request.

pub const MAX_BYTES: usize = 256 * 1024;
pub const MAX_WORDS: usize = 512;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Kind {
    Prose,
    Code,
    Numeric,
    Mixed,
    Unknown,
}

impl Kind {
    pub fn name(self) -> &'static str {
        match self {
            Self::Prose => "prose",
            Self::Code => "code",
            Self::Numeric => "numeric",
            Self::Mixed => "mixed",
            Self::Unknown => "unknown",
        }
    }
}

#[derive(Clone, Copy)]
pub struct Profile {
    pub prose: u8,
    pub code: u8,
    pub numeric: u8,
    pub fallback: u8,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Decision {
    pub kind: Kind,
    pub k: u8,
}

/// Call once before drafting, then retain `k` for the complete request.
/// Explicit operator pins bypass this policy at the caller.
pub fn select_depth(instruction: &str, profile: Profile, ceiling: u8) -> Decision {
    let kind = classify(instruction);
    let k = match kind {
        Kind::Prose => profile.prose,
        Kind::Code => profile.code,
        Kind::Numeric => profile.numeric,
        Kind::Mixed | Kind::Unknown => profile.fallback,
    };
    Decision {
        kind,
        k: k.min(ceiling),
    }
}

const PROSE: u8 = 1;
const CODE: u8 = 2;
const NUMERIC: u8 = 4;

#[derive(Clone, Copy, PartialEq)]
enum Tag {
    Action(u8),
    Produce,
    Noun(u8),
    Hint(u8),
    Format,
    Negation,
    And,
    Next,
    Only,
    Content,
    In,
    Question,
    Quantity,
    Other,
}

fn tag(word: &str) -> Tag {
    // Normalizing one bounded token on the stack avoids a regex engine, heap
    // allocation, or an NLP model. Long identifiers carry no routing evidence.
    let mut lower = [0u8; 32];
    if word.len() > lower.len() {
        return Tag::Other;
    }
    for (dst, src) in lower.iter_mut().zip(word.bytes()) {
        *dst = src.to_ascii_lowercase();
    }
    match &lower[..word.len()] {
        b"explain" | b"summarize" | b"summarise" | b"describe" | b"discuss" | b"compare"
        | b"review" | b"translate" | b"paraphrase" | b"proofread" | b"recommend" | b"outline"
        | b"analyze" | b"analyse" | b"tell" | b"say" | b"respond" | b"reply" => Tag::Action(PROSE),
        b"implement" | b"refactor" | b"debug" => Tag::Action(CODE),
        b"calculate" | b"compute" | b"solve" | b"tabulate" => Tag::Action(NUMERIC),
        b"write" | b"generate" | b"create" | b"produce" | b"return" | b"output" | b"give"
        | b"provide" | b"make" | b"show" | b"build" | b"draft" | b"compose" | b"rewrite"
        | b"convert" | b"format" | b"fix" | b"need" | b"evaluate" => Tag::Produce,
        b"prose" | b"explanation" | b"summary" | b"essay" | b"email" | b"emails" | b"letter"
        | b"poem" | b"story" | b"paragraph" | b"paragraphs" | b"description" | b"documentation"
        | b"docstring" | b"analysis" | b"comparison" | b"plan" | b"instructions" | b"tutorial"
        | b"article" | b"report" | b"rationale" | b"advice" | b"checklist" | b"feedback" => {
            Tag::Noun(PROSE)
        }
        b"code" | b"function" | b"script" | b"class" | b"pseudocode" | b"regex" | b"regexp"
        | b"patch" | b"diff" | b"dockerfile" | b"parser" | b"implementation" | b"program"
        | b"endpoint" => Tag::Noun(CODE),
        b"number" | b"numbers" | b"sum" | b"quotient" | b"decimal" | b"decimals"
        | b"percentage" | b"percentages" | b"total" | b"totals" | b"mean" | b"median"
        | b"variance" | b"probability" | b"equation" | b"equations" | b"integral"
        | b"derivative" | b"factorial" => Tag::Noun(NUMERIC),
        b"python" | b"javascript" | b"typescript" | b"rust" | b"golang" | b"java" | b"sql"
        | b"html" | b"css" | b"bash" | b"shell" | b"powershell" | b"julia" | b"perl"
        | b"kotlin" | b"swift" => Tag::Hint(CODE),
        b"numeric" | b"numerical" | b"arithmetic" => Tag::Hint(NUMERIC),
        b"json" | b"yaml" | b"xml" | b"toml" | b"csv" => Tag::Format,
        b"no" | b"not" | b"don't" | b"dont" | b"never" | b"without" | b"avoid" | b"can't"
        | b"cannot" | b"don\xe2\x80\x99t" | b"can\xe2\x80\x99t" => Tag::Negation,
        b"and" => Tag::And,
        b"then" | b"also" => Tag::Next,
        b"only" | b"just" => Tag::Only,
        b"about" | b"that" | b"which" | b"because" | b"how" | b"why" | b"for" | b"with" | b"of"
        | b"to" | b"on" => Tag::Content,
        b"in" | b"as" => Tag::In,
        b"what" | b"find" | b"determine" => Tag::Question,
        b"many" | b"much" => Tag::Quantity,
        _ => Tag::Other,
    }
}

#[derive(Clone, Copy)]
enum Token<'a> {
    Word(&'a str),
    Boundary,
    Math,
}

#[derive(Clone)]
struct Tokens<'a> {
    text: &'a str,
    pos: usize,
    line_start: bool,
}

impl<'a> Tokens<'a> {
    fn new(text: &'a str) -> Self {
        Self {
            text,
            pos: 0,
            line_start: true,
        }
    }

    fn skip_quote(&mut self, delimiter: u8, count: usize) {
        let bytes = self.text.as_bytes();
        self.pos += count;
        let mut run = 0;
        while self.pos < bytes.len() {
            let byte = bytes[self.pos];
            self.pos += 1;
            if byte == b'\\' && delimiter != b'`' && delimiter != b'~' {
                self.pos = (self.pos + 1).min(bytes.len());
                run = 0;
            } else if byte == delimiter {
                run += 1;
                if run == count {
                    if matches!(delimiter, b'`' | b'~') {
                        while self.pos < bytes.len() && bytes[self.pos] == delimiter {
                            self.pos += 1;
                        }
                    }
                    break;
                }
            } else {
                run = 0;
            }
        }
    }
}

impl<'a> Iterator for Tokens<'a> {
    type Item = Token<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        let bytes = self.text.as_bytes();
        while self.pos < bytes.len() {
            let byte = bytes[self.pos];
            if byte == b'<' {
                let mut skipped = false;
                for (open, close) in [
                    ("<reference", "</reference>"),
                    ("<context", "</context>"),
                    ("<document", "</document>"),
                ] {
                    if bytes[self.pos..].starts_with(open.as_bytes())
                        && bytes
                            .get(self.pos + open.len())
                            .is_some_and(|b| *b == b'>' || b.is_ascii_whitespace())
                    {
                        // '<' is a UTF-8 boundary. Find the closing marker
                        // without tokenizing or copying the reference.
                        self.pos = self.text[self.pos + open.len()..]
                            .find(close)
                            .map_or(bytes.len(), |end| self.pos + open.len() + end + close.len());
                        skipped = true;
                        break;
                    }
                }
                if skipped {
                    continue;
                }
            }
            let curly_close = if bytes[self.pos..].starts_with("“".as_bytes()) {
                Some("”")
            } else if bytes[self.pos..].starts_with("‘".as_bytes()) {
                Some("’")
            } else {
                None
            };
            if let Some(close) = curly_close {
                self.pos += close.len();
                while self.pos < bytes.len() && !bytes[self.pos..].starts_with(close.as_bytes()) {
                    self.pos += 1;
                }
                self.pos = (self.pos + close.len()).min(bytes.len());
                continue;
            }
            if byte == b'>' && self.line_start {
                while self.pos < bytes.len() && bytes[self.pos] != b'\n' {
                    self.pos += 1;
                }
                continue;
            }
            if matches!(byte, b'`' | b'~') {
                let count = bytes[self.pos..].iter().take_while(|&&b| b == byte).count();
                if byte == b'`' || count >= 3 {
                    self.skip_quote(byte, count);
                    continue;
                }
            }
            if matches!(byte, b'"' | b'\'') {
                self.skip_quote(byte, 1);
                continue;
            }
            if byte.is_ascii_alphanumeric() {
                let start = self.pos;
                self.pos += 1;
                while self.pos < bytes.len() {
                    if bytes[self.pos].is_ascii_alphanumeric() || bytes[self.pos] == b'\'' {
                        self.pos += 1;
                    } else if bytes[self.pos..].starts_with("’".as_bytes()) {
                        self.pos += "’".len();
                    } else {
                        break;
                    }
                }
                self.line_start = false;
                return Some(Token::Word(&self.text[start..self.pos]));
            }
            if ["×", "÷", "−"]
                .iter()
                .any(|s| bytes[self.pos..].starts_with(s.as_bytes()))
            {
                self.pos += if byte == 0xc3 { 2 } else { 3 };
                return Some(Token::Math);
            }
            self.pos += 1;
            if matches!(byte, b'\n' | b'.' | b',' | b';' | b':' | b'?' | b'!') {
                self.line_start = byte == b'\n';
                return Some(Token::Boundary);
            }
            if matches!(byte, b'+' | b'-' | b'*' | b'/' | b'=' | b'%') {
                self.line_start = false;
                return Some(Token::Math);
            }
            if !byte.is_ascii_whitespace() {
                self.line_start = false;
            }
        }
        None
    }
}

#[derive(Default)]
struct Clause {
    active: bool,
    produce: bool,
    negated: bool,
    closed_object: bool,
    in_format: bool,
    only: bool,
    kind: u8,
    hint: u8,
}

impl Clause {
    fn finish(&mut self, result: &mut u8) {
        let kind = if self.negated {
            0
        } else {
            self.kind | self.hint
        };
        if self.active && kind != 0 {
            *result |= kind;
        }
        *self = Self::default();
    }
}

pub fn classify(instruction: &str) -> Kind {
    if instruction.len() > MAX_BYTES {
        return Kind::Unknown;
    }
    let mut tokens = Tokens::new(instruction).peekable();
    let mut clause = Clause::default();
    let (mut result, mut words, mut numbers, mut math, mut question) = (0, 0, 0, false, false);
    let mut saw_action = false;
    let mut previous_how = false;
    while let Some(token) = tokens.next() {
        let word = match token {
            Token::Boundary => {
                clause.finish(&mut result);
                previous_how = false;
                continue;
            }
            Token::Math => {
                math = true;
                previous_how = false;
                continue;
            }
            Token::Word(word) => word,
        };
        words += 1;
        if words > MAX_WORDS {
            return Kind::Unknown;
        }
        numbers += usize::from(word.bytes().all(|b| b.is_ascii_digit()));
        let quantity_question = previous_how;
        previous_how = word.eq_ignore_ascii_case("how");
        let token_tag = tag(word);
        let next_tag = match tokens.peek() {
            Some(Token::Word(next)) => tag(next),
            _ => Tag::Other,
        };
        let new_action = matches!(
            next_tag,
            Tag::Action(_) | Tag::Produce | Tag::Next | Tag::Only
        );
        let new_object = clause.produce
            && !clause.closed_object
            && matches!(next_tag, Tag::Noun(_) | Tag::Hint(_) | Tag::Format);
        if token_tag == Tag::Next || (token_tag == Tag::And && (new_action || new_object)) {
            let negated = token_tag == Tag::And && clause.negated;
            clause.finish(&mut result);
            clause.negated = negated;
            if new_object && !new_action {
                clause.active = true;
                clause.produce = true;
            }
            continue;
        }
        match token_tag {
            Tag::Negation if !clause.active => clause.negated = true,
            Tag::Negation => clause.closed_object = true,
            Tag::Only if !clause.active || clause.produce => {
                clause.only = true;
                if !clause.active && clause.hint != 0 {
                    clause.active = true;
                    clause.produce = true;
                }
            }
            Tag::Question if !clause.active => question = true,
            Tag::Quantity if !clause.active && quantity_question => {
                saw_action = true;
                clause.active = true;
                clause.kind = NUMERIC;
            }
            Tag::Action(kind) if !clause.active => {
                saw_action = true;
                clause.active = true;
                clause.kind = kind;
            }
            Tag::Produce if !clause.active => {
                saw_action = true;
                clause.active = true;
                clause.produce = true;
            }
            Tag::Noun(kind) | Tag::Hint(kind) if !clause.active => {
                clause.hint |= kind;
                if clause.only {
                    clause.active = true;
                    clause.produce = true;
                }
            }
            Tag::Format if !clause.active => {
                clause.hint |= CODE;
                if clause.only {
                    clause.active = true;
                    clause.produce = true;
                }
            }
            Tag::Format if clause.active && (clause.produce || clause.in_format) => {
                clause.kind = CODE;
                clause.hint = 0;
            }
            Tag::Noun(kind) if clause.produce && !clause.closed_object && clause.kind == 0 => {
                clause.kind = kind;
                clause.hint = 0;
            }
            Tag::Hint(kind) if clause.produce && !clause.closed_object && clause.kind == 0 => {
                clause.hint |= kind;
            }
            Tag::Content if clause.kind != 0 => clause.closed_object = true,
            _ => {}
        }
        clause.in_format = token_tag == Tag::In;
    }
    clause.finish(&mut result);
    match result {
        PROSE => Kind::Prose,
        CODE => Kind::Code,
        NUMERIC => Kind::Numeric,
        0 if !saw_action && numbers >= 2 && math && (question || words == numbers) => Kind::Numeric,
        0 => Kind::Unknown,
        _ => Kind::Mixed,
    }
}
