//! The part of Python's `http.client` the node uses (httpproxy.py `request`, discovery.py `http_get`):
//! one HTTP/1.1 exchange over a plain TCP connection to 127.0.0.1, never a redirect. Parsing follows CPython 3.12:
//! the status line (`_read_status`: at most 65536 bytes, "HTTP/" version, status 100-999, interim 100 Continue
//! responses skipped), at most 100 header lines of at most 65536 bytes (decoded as ISO-8859-1 and split like the
//! `email` parser does), then a body framed by `Transfer-Encoding: chunked`, `Content-Length` or the end of the
//! connection, read with `read1` semantics. Errors carry the Python exception class name the node puts in messages.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{Shutdown, SocketAddr, TcpStream};
use std::time::Duration;

use crate::util::py;

const MAXLINE: usize = 65536;
const MAXHEADERS: usize = 100;

/// A failure, as the Python exception it would be.
#[derive(Debug)]
pub enum HttpError {
    /// an OSError (TimeoutError, ConnectionRefusedError, BrokenPipeError, ...)
    Io(std::io::Error),
    /// an http.client.HTTPException subclass (RemoteDisconnected, BadStatusLine, IncompleteRead, LineTooLong, ...)
    Proto(&'static str),
}

impl From<std::io::Error> for HttpError {
    fn from(e: std::io::Error) -> Self {
        HttpError::Io(e)
    }
}

impl HttpError {
    /// `except TimeoutError`
    pub fn is_timeout(&self) -> bool {
        matches!(self, HttpError::Io(e) if matches!(e.kind(), std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut))
    }

    /// `except ConnectionRefusedError`
    pub fn is_refused(&self) -> bool {
        matches!(self, HttpError::Io(e) if e.kind() == std::io::ErrorKind::ConnectionRefused)
    }

    /// `type(e).__name__`
    pub fn class_name(&self) -> String {
        match self {
            HttpError::Proto(n) => (*n).to_string(),
            HttpError::Io(_) if self.is_timeout() => "TimeoutError".into(),
            HttpError::Io(e) => match e.raw_os_error() {
                Some(n) => py::os_error_name(n).to_string(),
                None => match e.kind() {
                    std::io::ErrorKind::ConnectionRefused => "ConnectionRefusedError".into(),
                    std::io::ErrorKind::ConnectionReset => "ConnectionResetError".into(),
                    std::io::ErrorKind::ConnectionAborted => "ConnectionAbortedError".into(),
                    std::io::ErrorKind::BrokenPipe => "BrokenPipeError".into(),
                    _ => "OSError".into(),
                },
            },
        }
    }
}

fn dur(s: f64) -> Duration {
    Duration::from_secs_f64(if s.is_finite() && s > 0.0 { s } else { 0.001 })
}

/// A connection to 127.0.0.1:<port> (`HTTPConnection("127.0.0.1", port, timeout=connect_s).connect()`).
pub struct Conn {
    w: TcpStream,
    r: Option<BufReader<TcpStream>>,
}

impl Conn {
    pub fn connect(port: u16, connect_s: f64) -> Result<Self, HttpError> {
        let addr = SocketAddr::from(([127, 0, 0, 1], port));
        let w = TcpStream::connect_timeout(&addr, dur(connect_s))?;
        w.set_read_timeout(Some(dur(connect_s)))?;
        w.set_write_timeout(Some(dur(connect_s)))?;
        let r = BufReader::with_capacity(65536, w.try_clone()?);
        Ok(Self { w, r: Some(r) })
    }

    /// A second handle on the socket (for a cancel hook's shutdown).
    pub fn handle(&self) -> Result<TcpStream, HttpError> {
        Ok(self.w.try_clone()?)
    }

    /// `sock.settimeout(t)`: every later send and receive.
    pub fn set_timeout(&self, t: f64) -> Result<(), HttpError> {
        self.w.set_read_timeout(Some(dur(t)))?;
        self.w.set_write_timeout(Some(dur(t)))?;
        Ok(())
    }

    /// `sock.sendall(data)`
    pub fn send(&mut self, data: &[u8]) -> Result<(), HttpError> {
        self.w.write_all(data)?;
        Ok(())
    }

    /// `conn.getresponse()`; `head`: the request was a HEAD.
    pub fn get_response(&mut self, head: bool) -> Result<Response, HttpError> {
        let mut r = self.r.take().ok_or(HttpError::Proto("ResponseNotReady"))?;
        let (status, version) = loop {
            let (version, status) = read_status(&mut r)?;
            if status != 100 {
                break (status, version);
            }
            read_header_lines(&mut r)?; // the interim response's headers
        };
        if !(version == "HTTP/1.0" || version == "HTTP/0.9" || version.starts_with("HTTP/1.")) {
            return Err(HttpError::Proto("UnknownProtocol"));
        }
        let lines = read_header_lines(&mut r)?;
        let headers = parse_header_block(&lines);
        let get = |name: &str| {
            headers
                .iter()
                .find(|(k, _)| k.to_lowercase() == name)
                .map(|(_, v)| v.clone())
        };
        let chunked = get("transfer-encoding")
            .is_some_and(|t| !t.is_empty() && t.to_lowercase() == "chunked");
        let mut length: Option<u64> = None;
        if let Some(l) = get("content-length").filter(|l| !l.is_empty())
            && !chunked
        {
            length = py::parse_int(&l).and_then(|n| u64::try_from(n).ok());
        }
        if status == 204 || status == 304 || (100..200).contains(&status) || head {
            length = Some(0);
        }
        let body = if head {
            Body::Done
        } else if chunked {
            Body::Chunked(None)
        } else {
            match length {
                Some(n) => Body::Length(n),
                None => Body::Close,
            }
        };
        Ok(Response {
            status,
            headers,
            r: Some(r),
            body,
        })
    }

    /// `conn.close()` without waiting: the other side sees the connection end.
    pub fn shutdown(&self) {
        let _ = self.w.shutdown(Shutdown::Both);
    }
}

/// `sock.shutdown(SHUT_RDWR)` on a handle from `Conn::handle`.
pub fn shutdown(s: &TcpStream) {
    let _ = s.shutdown(Shutdown::Both);
}

/// `fp.readline(limit)`: through the next b"\n", at most `limit` bytes.
fn readline(r: &mut BufReader<TcpStream>, limit: usize) -> Result<Vec<u8>, HttpError> {
    let mut line = Vec::new();
    while line.len() < limit {
        let avail = match r.fill_buf() {
            Ok(b) => b,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(e.into()),
        };
        if avail.is_empty() {
            break;
        }
        let room = limit - line.len();
        let window = &avail[..avail.len().min(room)];
        if let Some(i) = window.iter().position(|c| *c == b'\n') {
            line.extend_from_slice(&window[..=i]);
            r.consume(i + 1);
            return Ok(line);
        }
        let n = window.len();
        line.extend_from_slice(window);
        r.consume(n);
    }
    Ok(line)
}

/// ISO-8859-1: one character per byte.
fn latin1(b: &[u8]) -> String {
    b.iter().map(|c| char::from(*c)).collect()
}

fn read_status(r: &mut BufReader<TcpStream>) -> Result<(String, i64), HttpError> {
    let raw = readline(r, MAXLINE + 1)?;
    if raw.len() > MAXLINE {
        return Err(HttpError::Proto("LineTooLong"));
    }
    if raw.is_empty() {
        return Err(HttpError::Proto("RemoteDisconnected"));
    }
    let line = latin1(&raw);
    let parts: Vec<&str> = line.split(py::isspace).filter(|p| !p.is_empty()).collect();
    let (version, status) = match parts.as_slice() {
        [v, s, ..] => (*v, *s),
        _ => ("", ""),
    };
    if !version.starts_with("HTTP/") {
        return Err(HttpError::Proto("BadStatusLine"));
    }
    match py::parse_int(status) {
        Some(s) if (100..=999).contains(&s) => Ok((version.to_string(), s)),
        _ => Err(HttpError::Proto("BadStatusLine")),
    }
}

/// `http.client._read_headers`: the raw lines up to the blank one.
fn read_header_lines(r: &mut BufReader<TcpStream>) -> Result<Vec<Vec<u8>>, HttpError> {
    let mut lines = Vec::new();
    loop {
        let line = readline(r, MAXLINE + 1)?;
        if line.len() > MAXLINE {
            return Err(HttpError::Proto("LineTooLong"));
        }
        if line == b"\r\n" || line == b"\n" || line.is_empty() {
            return Ok(lines);
        }
        lines.push(line);
        if lines.len() > MAXHEADERS {
            return Err(HttpError::Proto("HTTPException"));
        }
    }
}

/// `email.parser` over the header block (compat32): (name, value) in order; folded lines joined, a line that is
/// not a header ends the block.
fn parse_header_block(raw: &[Vec<u8>]) -> Vec<(String, String)> {
    let text: String = raw.iter().map(|l| latin1(l)).collect();
    // the email parser's universal-newline split, line endings kept
    let mut lines: Vec<String> = Vec::new();
    let mut cur = String::new();
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        cur.push(c);
        if c == '\n' || (c == '\r' && chars.peek() != Some(&'\n')) {
            lines.push(std::mem::take(&mut cur));
        }
    }
    if !cur.is_empty() {
        lines.push(cur);
    }
    let is_header_line = |l: &str| {
        if l.starts_with("From ") || l.starts_with(' ') || l.starts_with('\t') {
            return true;
        }
        match l.find(':') {
            Some(i) => l[..i]
                .chars()
                .all(|c| ('\x21'..='\x7e').contains(&c) && c != ':'),
            None => false,
        }
    };
    let mut out: Vec<(String, String)> = Vec::new();
    let mut last: Vec<String> = Vec::new();
    let flush = |last: &mut Vec<String>, out: &mut Vec<(String, String)>| {
        if last.is_empty() {
            return;
        }
        let (name, first) = last[0].split_once(':').unwrap_or((last[0].as_str(), ""));
        let mut value = first.trim_start_matches([' ', '\t']).to_string();
        for cont in &last[1..] {
            value.push_str(cont);
        }
        out.push((
            name.to_string(),
            value.trim_end_matches(['\r', '\n']).to_string(),
        ));
        last.clear();
    };
    let n = lines.len();
    for (lineno, line) in lines.iter().enumerate() {
        if !is_header_line(line) {
            break;
        }
        if line.starts_with(' ') || line.starts_with('\t') {
            if !last.is_empty() {
                last.push(line.clone());
            }
            continue;
        }
        flush(&mut last, &mut out);
        if line.starts_with("From ") {
            if lineno + 1 == n {
                break;
            }
            continue;
        }
        match line.find(':') {
            Some(0) | None => continue,
            Some(_) => last.push(line.clone()),
        }
    }
    flush(&mut last, &mut out);
    out
}

enum Body {
    Length(u64),
    /// bytes left in the current chunk: None before the first, Some(0) at a chunk's end
    Chunked(Option<u64>),
    Close,
    Done,
}

/// `HTTPResponse`
pub struct Response {
    pub status: i64,
    /// `getheaders()`: (name, value) as received
    pub headers: Vec<(String, String)>,
    r: Option<BufReader<TcpStream>>,
    body: Body,
}

/// `BufferedReader.read1(n)`: buffered bytes if any, else one read.
fn fp_read1(r: &mut BufReader<TcpStream>, n: usize) -> Result<Vec<u8>, HttpError> {
    if n == 0 {
        return Ok(Vec::new());
    }
    loop {
        match r.fill_buf() {
            Ok(b) => {
                let take = b.len().min(n);
                let out = b[..take].to_vec();
                r.consume(take);
                return Ok(out);
            }
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
            Err(e) => return Err(e.into()),
        }
    }
}

impl Response {
    fn close(&mut self) {
        self.r = None;
    }

    /// `resp.read1(n)`: at most n bytes, empty at the end of the body.
    pub fn read1(&mut self, n: usize) -> Result<Vec<u8>, HttpError> {
        let Some(r) = self.r.as_mut() else {
            return Ok(Vec::new());
        };
        match self.body {
            Body::Done => Ok(Vec::new()),
            Body::Chunked(_) => self.read1_chunked(n),
            Body::Length(left) => {
                let want = usize::try_from(left).map_or(n, |l| n.min(l));
                let got = fp_read1(r, want)?;
                if got.is_empty() && want > 0 {
                    self.close();
                } else {
                    let left = left - got.len() as u64;
                    self.body = Body::Length(left);
                    if left == 0 {
                        self.close();
                    }
                }
                Ok(got)
            }
            Body::Close => {
                let got = fp_read1(r, n)?;
                if got.is_empty() && n > 0 {
                    self.close();
                }
                Ok(got)
            }
        }
    }

    fn chunk_left(&mut self) -> Result<Option<u64>, HttpError> {
        let Body::Chunked(left) = self.body else {
            return Ok(None);
        };
        if let Some(l) = left
            && l > 0
        {
            return Ok(Some(l));
        }
        let Some(r) = self.r.as_mut() else {
            return Ok(None);
        };
        if left.is_some() {
            // the CRLF at the end of the chunk
            let mut crlf = [0u8; 2];
            let mut got = 0;
            while got < 2 {
                match r.read(&mut crlf[got..]) {
                    Ok(0) => return Err(HttpError::Proto("IncompleteRead")),
                    Ok(k) => got += k,
                    Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
                    Err(e) => return Err(e.into()),
                }
            }
        }
        let line = readline(r, MAXLINE + 1)?;
        if line.len() > MAXLINE {
            return Err(HttpError::Proto("LineTooLong"));
        }
        let line = match line.iter().position(|c| *c == b';') {
            Some(i) => &line[..i],
            None => &line[..],
        };
        let Some(size) = py::parse_int_hex(&latin1(line)) else {
            self.close();
            return Err(HttpError::Proto("IncompleteRead"));
        };
        if size == 0 {
            // the trailer, up to the blank line
            loop {
                let t = readline(r, MAXLINE + 1)?;
                if t.len() > MAXLINE {
                    return Err(HttpError::Proto("LineTooLong"));
                }
                if t.is_empty() || t == b"\r\n" || t == b"\n" {
                    break;
                }
            }
            self.close();
            self.body = Body::Chunked(None);
            return Ok(None);
        }
        self.body = Body::Chunked(Some(size));
        Ok(Some(size))
    }

    fn read1_chunked(&mut self, n: usize) -> Result<Vec<u8>, HttpError> {
        let Some(left) = self.chunk_left()? else {
            return Ok(Vec::new());
        };
        if n == 0 {
            return Ok(Vec::new());
        }
        let want = usize::try_from(left).map_or(n, |l| n.min(l));
        let Some(r) = self.r.as_mut() else {
            return Ok(Vec::new());
        };
        let got = fp_read1(r, want)?;
        self.body = Body::Chunked(Some(left - got.len() as u64));
        if got.is_empty() {
            return Err(HttpError::Proto("IncompleteRead"));
        }
        Ok(got)
    }
}

/// The request line and headers, as http.client sends them (ISO-8859-1 header values).
pub fn request_head(method: &str, path: &str, headers: &[(String, String)]) -> Vec<u8> {
    let mut out = format!("{method} {path} HTTP/1.1\r\n").into_bytes();
    for (k, v) in headers {
        out.extend_from_slice(k.as_bytes());
        out.extend_from_slice(b": ");
        out.extend(
            v.chars()
                .map(|c| u8::try_from(u32::from(c)).unwrap_or(b'?')),
        );
        out.extend_from_slice(b"\r\n");
    }
    out.extend_from_slice(b"\r\n");
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lines(text: &str) -> Vec<Vec<u8>> {
        text.split_inclusive('\n')
            .map(|l| l.as_bytes().to_vec())
            .collect()
    }

    #[test]
    fn header_block_like_the_email_parser() {
        assert_eq!(
            parse_header_block(&lines("A: 1\r\nB:  two \r\n  folded\r\nC:3\r\n")),
            vec![
                ("A".to_string(), "1".to_string()),
                ("B".to_string(), "two \r\n  folded".to_string()),
                ("C".to_string(), "3".to_string())
            ]
        );
        // a line that is no header ends the block
        assert_eq!(
            parse_header_block(&lines("A: 1\r\nBad Header: x\r\nC: 3\r\n")),
            vec![("A".to_string(), "1".to_string())]
        );
    }
}
