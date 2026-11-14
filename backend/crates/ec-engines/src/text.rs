//! Python text helpers the adapters share with the hub.

/// `str.splitlines()` (the line boundaries Python recognises, without keeping them).
pub fn py_splitlines(s: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut start = 0;
    let mut chars = s.char_indices().peekable();
    while let Some((i, c)) = chars.next() {
        let is_break = matches!(
            c,
            '\n' | '\r'
                | '\u{b}'
                | '\u{c}'
                | '\u{1c}'
                | '\u{1d}'
                | '\u{1e}'
                | '\u{85}'
                | '\u{2028}'
                | '\u{2029}'
        );
        if is_break {
            out.push(&s[start..i]);
            let mut end = i + c.len_utf8();
            if c == '\r'
                && let Some(&(j, '\n')) = chars.peek()
            {
                chars.next();
                end = j + 1;
            }
            start = end;
        }
    }
    if start < s.len() {
        out.push(&s[start..]);
    }
    out
}
