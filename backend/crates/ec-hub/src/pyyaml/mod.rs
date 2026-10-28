//! PyYAML 6.0.3 for the hub, in one module (PORTING.md §A9): `safe_load` ([`load`], with the Python objects it builds
//! in [`pyobj`]) and, in this file, `yaml.safe_dump(data, sort_keys=..., default_flow_style=False)` for JSON-like data,
//! shared by every caller: the TensorRT-LLM config (`sort_keys=True`), the compose file of a run spec
//! (`sort_keys=False`) and the profile export. First written as `trtllm::yaml` and merged
//! with the compose emitter of the lifecycle (§A9); pinned by both golden sets (tests/test_ws_a2_goldens.rs, 1300
//! cases with sort_keys=True; services/docker.rs `test_yaml_safe_dump_matches_pyyaml_vectors`, sort_keys=False).
//!
//! The port: the SafeRepresenter's scalars, the Resolver's implicit tags (which decide whether a string may stay plain) and the
//! Emitter's state machine (indentation, indentless sequences in mappings, simple vs complex keys, plain /
//! single-quoted / double-quoted styles, folding at 80 columns, escapes without allow_unicode), ported line by
//! line from `yaml/emitter.py`.

pub mod load;
pub mod pyobj;

pub use load::safe_load;

use std::sync::LazyLock;

use regex::Regex;

use serde_json::Value;

/// A Python value as `yaml.safe_dump` sees it: what `json.loads` builds (ints unbounded, as canonical digits) and
/// insertion-ordered mappings with str keys (a Python dict).
#[derive(Debug, Clone, PartialEq)]
pub enum Node {
    Null,
    Bool(bool),
    /// canonical decimal digits, with a leading `-` when negative
    Int(String),
    Float(f64),
    Str(String),
    Seq(Vec<Node>),
    Map(Vec<(String, Node)>),
}

impl Node {
    pub fn int(i: i64) -> Self {
        Self::Int(i.to_string())
    }

    /// The value `json.loads` gives for this JSON.
    pub fn from_json(v: &Value) -> Self {
        match v {
            Value::Null => Self::Null,
            Value::Bool(b) => Self::Bool(*b),
            Value::Number(n) if n.is_f64() => n.as_f64().map_or(Self::Null, Self::Float),
            Value::Number(n) => Self::Int(n.to_string()),
            Value::String(s) => Self::Str(s.clone()),
            Value::Array(a) => Self::Seq(a.iter().map(Self::from_json).collect()),
            Value::Object(o) => Self::Map(
                o.iter()
                    .map(|(k, x)| (k.clone(), Self::from_json(x)))
                    .collect(),
            ),
        }
    }
}

const BEST_WIDTH: usize = 80;
const BEST_INDENT: i64 = 2;

/// The implicit resolvers, keyed by the first characters they apply to (resolver.py, in registration order).
static RESOLVERS: LazyLock<Vec<(&'static str, Regex)>> = LazyLock::new(|| {
    let r = |p: &str| Regex::new(p).unwrap_or_else(|e| unreachable!("{e}"));
    vec![
        (
            "yYnNtTfFoO",
            r(
                r"^(?:yes|Yes|YES|no|No|NO|true|True|TRUE|false|False|FALSE|on|On|ON|off|Off|OFF)\n?\z",
            ),
        ),
        (
            "-+0123456789.",
            r(
                r"^(?:[-+]?(?:[0-9][0-9_]*)\.[0-9_]*(?:[eE][-+][0-9]+)?|\.[0-9][0-9_]*(?:[eE][-+][0-9]+)?|[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\.[0-9_]*|[-+]?\.(?:inf|Inf|INF)|\.(?:nan|NaN|NAN))\n?\z",
            ),
        ),
        (
            "-+0123456789",
            r(
                r"^(?:[-+]?0b[0-1_]+|[-+]?0[0-7_]+|[-+]?(?:0|[1-9][0-9_]*)|[-+]?0x[0-9a-fA-F_]+|[-+]?[1-9][0-9_]*(?::[0-5]?[0-9])+)\n?\z",
            ),
        ),
        ("<", r(r"^(?:<<)\n?\z")),
        ("~nN", r(r"^(?:~|null|Null|NULL|)\n?\z")),
        (
            "0123456789",
            r(
                r"^(?:[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]|[0-9][0-9][0-9][0-9]-[0-9][0-9]?-[0-9][0-9]?(?:[Tt]|[ \t]+)[0-9][0-9]?:[0-9][0-9]:[0-9][0-9](?:\.[0-9]*)?(?:[ \t]*(?:Z|[-+][0-9][0-9]?(?::[0-9][0-9])?))?)\n?\z",
            ),
        ),
        ("=", r(r"^(?:=)\n?\z")),
        ("!&*", r(r"^(?:!|&|\*)\n?\z")),
    ]
});

/// `resolve(ScalarNode, value, (True, False)) == str tag`: a plain scalar would read back as a string.
fn plain_reads_as_str(value: &str) -> bool {
    let Some(first) = value.chars().next() else {
        return false; // '' resolves to null
    };
    !RESOLVERS
        .iter()
        .any(|(firsts, re)| firsts.contains(first) && re.is_match(value))
}

/// One emitter event (anchors and tags never occur in this data).
enum Ev {
    Scalar { value: String, implicit0: bool },
    SeqStart,
    SeqEnd,
    MapStart,
    MapEnd,
}

/// The representer + serializer: data to events (mappings sorted by key).
fn events(v: &Node, sort_keys: bool, out: &mut Vec<Ev>) {
    match v {
        Node::Null => out.push(scalar("null", true)),
        Node::Bool(b) => out.push(scalar(if *b { "true" } else { "false" }, true)),
        Node::Int(d) => out.push(scalar(d, true)),
        Node::Float(f) => out.push(scalar(&represent_float(*f), true)),
        Node::Str(s) => out.push(scalar(s, plain_reads_as_str(s))),
        Node::Seq(a) => {
            out.push(Ev::SeqStart);
            for x in a {
                events(x, sort_keys, out);
            }
            out.push(Ev::SeqEnd);
        }
        Node::Map(d) => {
            out.push(Ev::MapStart);
            let mut items: Vec<&(String, Node)> = d.iter().collect();
            if sort_keys {
                // `sorted(mapping.items())`: str keys by code point (UTF-8 byte order is code point order)
                items.sort_by(|a, b| a.0.cmp(&b.0));
            }
            for (k, x) in items {
                out.push(scalar(k, plain_reads_as_str(k)));
                events(x, sort_keys, out);
            }
            out.push(Ev::MapEnd);
        }
    }
}

fn scalar(value: &str, implicit0: bool) -> Ev {
    Ev::Scalar {
        value: value.to_string(),
        implicit0,
    }
}

/// `SafeRepresenter.represent_float`.
fn represent_float(f: f64) -> String {
    if f.is_nan() {
        return ".nan".into();
    }
    if f.is_infinite() {
        return if f > 0.0 {
            ".inf".into()
        } else {
            "-.inf".into()
        };
    }
    let value = crate::pyjson::float_repr(f).to_lowercase();
    if !value.contains('.') && value.contains('e') {
        return value.replacen('e', ".0e", 1);
    }
    value
}

struct Analysis {
    empty: bool,
    multiline: bool,
    allow_flow_plain: bool,
    allow_block_plain: bool,
    allow_single_quoted: bool,
}

fn is_break(c: char) -> bool {
    matches!(c, '\n' | '\u{85}' | '\u{2028}' | '\u{2029}')
}

fn is_ws_or_end(c: Option<&char>) -> bool {
    c.is_none_or(|c| {
        matches!(
            c,
            '\0' | ' ' | '\t' | '\r' | '\n' | '\u{85}' | '\u{2028}' | '\u{2029}'
        )
    })
}

/// `Emitter.analyze_scalar` (allow_unicode False).
fn analyze(scalar: &[char]) -> Analysis {
    if scalar.is_empty() {
        return Analysis {
            empty: true,
            multiline: false,
            allow_flow_plain: false,
            allow_block_plain: true,
            allow_single_quoted: true,
        };
    }
    let (mut block_ind, mut flow_ind, mut line_breaks, mut special) = (false, false, false, false);
    let (mut leading_space, mut leading_break, mut trailing_space, mut trailing_break) =
        (false, false, false, false);
    let (mut break_space, mut space_break) = (false, false);
    let starts = |p: &str| scalar.iter().take(3).copied().eq(p.chars());
    if scalar.len() >= 3 && (starts("---") || starts("...")) {
        block_ind = true;
        flow_ind = true;
    }
    let mut preceded_ws = true;
    let mut followed_ws = scalar.len() == 1 || is_ws_or_end(scalar.get(1));
    let (mut prev_space, mut prev_break) = (false, false);
    let n = scalar.len();
    for (index, &ch) in scalar.iter().enumerate() {
        if index == 0 {
            if "#,[]{}&*!|>'\"%@`".contains(ch) {
                flow_ind = true;
                block_ind = true;
            }
            if ch == '?' || ch == ':' {
                flow_ind = true;
                if followed_ws {
                    block_ind = true;
                }
            }
            if ch == '-' && followed_ws {
                flow_ind = true;
                block_ind = true;
            }
        } else {
            if ",?[]{}".contains(ch) {
                flow_ind = true;
            }
            if ch == ':' {
                flow_ind = true;
                if followed_ws {
                    block_ind = true;
                }
            }
            if ch == '#' && preceded_ws {
                flow_ind = true;
                block_ind = true;
            }
        }
        if is_break(ch) {
            line_breaks = true;
        }
        if !(ch == '\n' || ('\x20'..='\x7e').contains(&ch)) {
            special = true;
        }
        if ch == ' ' {
            if index == 0 {
                leading_space = true;
            }
            if index == n - 1 {
                trailing_space = true;
            }
            if prev_break {
                break_space = true;
            }
            prev_space = true;
            prev_break = false;
        } else if is_break(ch) {
            if index == 0 {
                leading_break = true;
            }
            if index == n - 1 {
                trailing_break = true;
            }
            if prev_space {
                space_break = true;
            }
            prev_space = false;
            prev_break = true;
        } else {
            prev_space = false;
            prev_break = false;
        }
        preceded_ws = matches!(
            ch,
            '\0' | ' ' | '\t' | '\r' | '\n' | '\u{85}' | '\u{2028}' | '\u{2029}'
        );
        followed_ws = index + 2 >= n || is_ws_or_end(scalar.get(index + 2));
    }
    let (mut flow_plain, mut block_plain, mut single) = (true, true, true);
    if leading_space || leading_break || trailing_space || trailing_break {
        flow_plain = false;
        block_plain = false;
    }
    if break_space {
        flow_plain = false;
        block_plain = false;
        single = false;
    }
    if space_break || special {
        flow_plain = false;
        block_plain = false;
        single = false;
    }
    if line_breaks {
        flow_plain = false;
        block_plain = false;
    }
    if flow_ind {
        flow_plain = false;
    }
    if block_ind {
        block_plain = false;
    }
    Analysis {
        empty: false,
        multiline: line_breaks,
        allow_flow_plain: flow_plain,
        allow_block_plain: block_plain,
        allow_single_quoted: single,
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Style {
    Plain,
    Single,
    Double,
}

struct Emitter<'a> {
    events: &'a [Ev],
    pos: usize,
    out: String,
    indents: Vec<Option<i64>>,
    indent: Option<i64>,
    flow_level: i64,
    root_context: bool,
    mapping_context: bool,
    simple_key_context: bool,
    column: usize,
    whitespace: bool,
    indention: bool,
    open_ended: bool,
}

impl Emitter<'_> {
    fn peek(&self, k: usize) -> Option<&Ev> {
        self.events.get(self.pos + k)
    }

    fn increase_indent(&mut self, flow: bool, indentless: bool) {
        self.indents.push(self.indent);
        match self.indent {
            None => self.indent = Some(if flow { BEST_INDENT } else { 0 }),
            Some(i) if !indentless => self.indent = Some(i + BEST_INDENT),
            Some(_) => {}
        }
    }

    fn pop_indent(&mut self) {
        self.indent = self.indents.pop().flatten();
    }

    fn write(&mut self, data: &str) {
        self.column += data.chars().count();
        self.out.push_str(data);
    }

    fn write_indicator(
        &mut self,
        indicator: &str,
        need_ws: bool,
        whitespace: bool,
        indention: bool,
    ) {
        let data = if self.whitespace || !need_ws {
            indicator.to_string()
        } else {
            format!(" {indicator}")
        };
        self.whitespace = whitespace;
        self.indention = self.indention && indention;
        self.open_ended = false;
        self.write(&data);
    }

    fn write_indent(&mut self) {
        let indent = usize::try_from(self.indent.unwrap_or(0)).unwrap_or(0);
        if !self.indention || self.column > indent || (self.column == indent && !self.whitespace) {
            self.write_line_break(None);
        }
        if self.column < indent {
            self.whitespace = true;
            let pad = " ".repeat(indent - self.column);
            self.write(&pad);
        }
    }

    fn write_line_break(&mut self, data: Option<char>) {
        self.whitespace = true;
        self.indention = true;
        self.column = 0;
        self.out.push(data.unwrap_or('\n'));
    }

    // ---------------------------------------------------------------- nodes
    /// `expect_node` for the event at `pos` (consumes the whole node).
    fn node(&mut self, root: bool, mapping: bool, simple_key: bool, analysis: Option<Analysis>) {
        self.root_context = root;
        self.mapping_context = mapping;
        self.simple_key_context = simple_key;
        match self.peek(0) {
            Some(Ev::Scalar { .. }) => {
                let Some(Ev::Scalar { value, implicit0 }) = self.peek(0) else {
                    return;
                };
                let (value, implicit0) = (value.clone(), *implicit0);
                self.pos += 1;
                let chars: Vec<char> = value.chars().collect();
                let a = analysis.unwrap_or_else(|| analyze(&chars));
                let style = self.choose_style(&a, implicit0);
                self.increase_indent(true, false);
                let split = !self.simple_key_context;
                match style {
                    Style::Double => self.write_double_quoted(&chars, split),
                    Style::Single => self.write_single_quoted(&chars, split),
                    Style::Plain => self.write_plain(&chars, split),
                }
                self.pop_indent();
            }
            Some(Ev::SeqStart) => {
                self.pos += 1;
                if self.flow_level > 0 || matches!(self.peek(0), Some(Ev::SeqEnd)) {
                    self.flow_collection('[', ']');
                } else {
                    self.block_sequence();
                }
            }
            Some(Ev::MapStart) => {
                self.pos += 1;
                if self.flow_level > 0 || matches!(self.peek(0), Some(Ev::MapEnd)) {
                    self.flow_collection('{', '}');
                } else {
                    self.block_mapping();
                }
            }
            _ => {}
        }
    }

    fn choose_style(&self, a: &Analysis, implicit0: bool) -> Style {
        if implicit0
            && !(self.simple_key_context && (a.empty || a.multiline))
            && ((self.flow_level > 0 && a.allow_flow_plain)
                || (self.flow_level == 0 && a.allow_block_plain))
        {
            return Style::Plain;
        }
        if a.allow_single_quoted && !(self.simple_key_context && a.multiline) {
            return Style::Single;
        }
        Style::Double
    }

    /// An empty flow collection (the only flow collections block-style data has).
    fn flow_collection(&mut self, open: char, close: char) {
        self.write_indicator(&open.to_string(), true, true, false);
        self.flow_level += 1;
        self.increase_indent(true, false);
        // the end event
        self.pos += 1;
        self.pop_indent();
        self.flow_level -= 1;
        self.write_indicator(&close.to_string(), false, false, false);
    }

    fn block_sequence(&mut self) {
        let indentless = self.mapping_context && !self.indention;
        self.increase_indent(false, indentless);
        while !matches!(self.peek(0), Some(Ev::SeqEnd) | None) {
            self.write_indent();
            self.write_indicator("-", true, false, true);
            self.node(false, false, false, None);
        }
        self.pos += 1;
        self.pop_indent();
    }

    /// `check_simple_key` for the key at `pos`; returns its analysis when it is a scalar. PyYAML counts the prepared
    /// tag too: every scalar event carries its resolved tag (`!!str`, `!!int`, ...: 5 characters for each scalar this
    /// module emits), so a key is simple only when `5 + len(value) < 128` (a 123-character key is complex: `? key`).
    fn simple_key(&self) -> (bool, Option<Analysis>) {
        const PREPARED_TAG_LEN: usize = 5;
        match self.peek(0) {
            Some(Ev::Scalar { value, .. }) => {
                let chars: Vec<char> = value.chars().collect();
                let a = analyze(&chars);
                (
                    PREPARED_TAG_LEN + chars.len() < 128 && !a.empty && !a.multiline,
                    Some(a),
                )
            }
            _ => (false, None),
        }
    }

    fn block_mapping(&mut self) {
        self.increase_indent(false, false);
        while !matches!(self.peek(0), Some(Ev::MapEnd) | None) {
            self.write_indent();
            let (simple, analysis) = self.simple_key();
            if simple {
                self.node(false, true, true, analysis);
                self.write_indicator(":", false, false, false);
                self.node(false, true, false, None);
            } else {
                self.write_indicator("?", true, false, true);
                self.node(false, true, false, analysis);
                self.write_indent();
                self.write_indicator(":", true, false, true);
                self.node(false, true, false, None);
            }
        }
        self.pos += 1;
        self.pop_indent();
    }

    // ---------------------------------------------------------------- scalar writers
    fn write_plain(&mut self, text: &[char], split: bool) {
        if self.root_context {
            self.open_ended = true;
        }
        if text.is_empty() {
            return;
        }
        if !self.whitespace {
            self.write(" ");
        }
        self.whitespace = false;
        self.indention = false;
        let (mut spaces, mut breaks) = (false, false);
        let (mut start, mut end) = (0usize, 0usize);
        while end <= text.len() {
            let ch = text.get(end).copied();
            if spaces {
                if ch != Some(' ') {
                    if start + 1 == end && self.column > BEST_WIDTH && split {
                        self.write_indent();
                        self.whitespace = false;
                        self.indention = false;
                    } else {
                        let data: String = text[start..end].iter().collect();
                        self.write(&data);
                    }
                    start = end;
                }
            } else if breaks {
                if !ch.is_some_and(is_break) {
                    if text[start] == '\n' {
                        self.write_line_break(None);
                    }
                    for &br in &text[start..end] {
                        self.write_line_break(Some(br));
                    }
                    self.write_indent();
                    self.whitespace = false;
                    self.indention = false;
                    start = end;
                }
            } else if ch.is_none_or(|c| c == ' ' || is_break(c)) {
                let data: String = text[start..end].iter().collect();
                self.write(&data);
                start = end;
            }
            if let Some(c) = ch {
                spaces = c == ' ';
                breaks = is_break(c);
            }
            end += 1;
        }
    }

    fn write_single_quoted(&mut self, text: &[char], split: bool) {
        self.write_indicator("'", true, false, false);
        let (mut spaces, mut breaks) = (false, false);
        let (mut start, mut end) = (0usize, 0usize);
        while end <= text.len() {
            let ch = text.get(end).copied();
            if spaces {
                if ch != Some(' ') {
                    if start + 1 == end
                        && self.column > BEST_WIDTH
                        && split
                        && start != 0
                        && end != text.len()
                    {
                        self.write_indent();
                    } else {
                        let data: String = text[start..end].iter().collect();
                        self.write(&data);
                    }
                    start = end;
                }
            } else if breaks {
                if !ch.is_some_and(is_break) {
                    if text[start] == '\n' {
                        self.write_line_break(None);
                    }
                    for &br in &text[start..end] {
                        self.write_line_break(Some(br));
                    }
                    self.write_indent();
                    start = end;
                }
            } else if ch.is_none_or(|c| c == ' ' || is_break(c) || c == '\'') && start < end {
                let data: String = text[start..end].iter().collect();
                self.write(&data);
                start = end;
            }
            if ch == Some('\'') {
                self.write("''");
                start = end + 1;
            }
            if let Some(c) = ch {
                spaces = c == ' ';
                breaks = is_break(c);
            }
            end += 1;
        }
        self.write_indicator("'", false, false, false);
    }

    fn escape(c: char) -> String {
        let short = match c {
            '\0' => Some('0'),
            '\x07' => Some('a'),
            '\x08' => Some('b'),
            '\x09' => Some('t'),
            '\x0a' => Some('n'),
            '\x0b' => Some('v'),
            '\x0c' => Some('f'),
            '\x0d' => Some('r'),
            '\x1b' => Some('e'),
            '"' => Some('"'),
            '\\' => Some('\\'),
            '\u{85}' => Some('N'),
            '\u{a0}' => Some('_'),
            '\u{2028}' => Some('L'),
            '\u{2029}' => Some('P'),
            _ => None,
        };
        if let Some(s) = short {
            return format!("\\{s}");
        }
        let n = c as u32;
        if n <= 0xff {
            format!("\\x{n:02X}")
        } else if n <= 0xffff {
            format!("\\u{n:04X}")
        } else {
            format!("\\U{n:08X}")
        }
    }

    fn write_double_quoted(&mut self, text: &[char], split: bool) {
        self.write_indicator("\"", true, false, false);
        let (mut start, mut end) = (0usize, 0usize);
        while end <= text.len() {
            let ch = text.get(end).copied();
            let needs_escape = ch.is_none_or(|c| {
                matches!(
                    c,
                    '"' | '\\' | '\u{85}' | '\u{2028}' | '\u{2029}' | '\u{feff}'
                ) || !('\x20'..='\x7e').contains(&c)
            });
            if needs_escape {
                if start < end {
                    let data: String = text[start..end].iter().collect();
                    self.write(&data);
                    start = end;
                }
                if let Some(c) = ch {
                    self.write(&Self::escape(c));
                    start = end + 1;
                }
            }
            if 0 < end
                && end + 1 < text.len()
                && (ch == Some(' ') || start >= end)
                && self.column + end.saturating_sub(start) > BEST_WIDTH
                && split
            {
                let mut data: String = text[start.min(end)..end].iter().collect();
                data.push('\\');
                if start < end {
                    start = end;
                }
                self.write(&data);
                self.write_indent();
                self.whitespace = false;
                self.indention = false;
                if text.get(start) == Some(&' ') {
                    self.write("\\");
                }
            }
            end += 1;
        }
        self.write_indicator("\"", false, false, false);
    }
}

/// `yaml.safe_dump(data, sort_keys=True, default_flow_style=False)`.
pub fn safe_dump(data: &Node, sort_keys: bool) -> String {
    let mut evs = Vec::new();
    events(data, sort_keys, &mut evs);
    let mut e = Emitter {
        events: &evs,
        pos: 0,
        out: String::new(),
        indents: Vec::new(),
        indent: None,
        flow_level: 0,
        root_context: false,
        mapping_context: false,
        simple_key_context: false,
        column: 0,
        whitespace: true,
        indention: true,
        open_ended: false,
    };
    // the document start is implicit for a collection root (and for any non-empty scalar)
    e.node(true, false, false, None);
    e.write_indent();
    if e.open_ended {
        e.write_indicator("...", true, false, false);
        e.write_indent();
    }
    e.out
}

#[cfg(test)]
mod tests {
    use super::{Node, safe_dump};

    #[test]
    fn a_key_is_simple_only_below_128_with_its_tag() {
        // PyYAML: len("!!str") + len(key) < 128, so 122 characters stay `key: v` and 123 become `? key` / `: v`
        // (a first port missed the tag; the compose vectors pinned it). Same with and without sort_keys.
        let (k122, k123) = ("k".repeat(122), "k".repeat(123));
        let doc = Node::Map(vec![
            (k122.clone(), Node::int(1)),
            (k123.clone(), Node::int(2)),
        ]);
        let want = format!("{k122}: 1\n? {k123}\n: 2\n");
        assert_eq!(safe_dump(&doc, false), want);
        assert_eq!(safe_dump(&doc, true), want);
    }

    #[test]
    fn sort_keys_orders_by_code_point_and_false_keeps_insertion_order() {
        let doc = Node::Map(vec![
            ("b".into(), Node::Null),
            ("a".into(), Node::Bool(true)),
            ("\u{e9}".into(), Node::Str("x".into())),
        ]);
        assert_eq!(safe_dump(&doc, true), "a: true\nb: null\n\"\\xE9\": x\n");
        assert_eq!(safe_dump(&doc, false), "b: null\na: true\n\"\\xE9\": x\n");
    }
}
