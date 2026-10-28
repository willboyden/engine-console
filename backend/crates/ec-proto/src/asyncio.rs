//! tokio readers (the hub side). Same rules and errors as the blocking ones.

use tokio::io::{AsyncBufRead, AsyncBufReadExt, AsyncRead, AsyncReadExt};

use crate::{Frame, MAGIC, MAX_PREAMBLE, ProtoError, ReadError, decode_body, parse_header};

/// Next frame and its size on the wire (5 + body); `None` on a clean end between frames. A stream that ends inside
/// a frame is `ProtoError::Truncated` (the hub's `_Truncated`).
pub async fn read_frame_async<R: AsyncRead + Unpin>(
    r: &mut R,
) -> Result<Option<(Frame, usize)>, ReadError> {
    let mut hdr = [0u8; 5];
    let mut got = 0;
    while got < 5 {
        let n = r.read(&mut hdr[got..]).await?;
        if n == 0 {
            if got == 0 {
                return Ok(None);
            }
            return Err(ProtoError::Truncated("stream ended inside a frame header".into()).into());
        }
        got += n;
    }
    let (kind, n) = parse_header(&hdr)?;
    let mut body = vec![0u8; n];
    if let Err(e) = r.read_exact(&mut body).await {
        if e.kind() == std::io::ErrorKind::UnexpectedEof {
            return Err(ProtoError::Truncated("stream ended inside a frame".into()).into());
        }
        return Err(e.into());
    }
    Ok(Some((decode_body(kind, &body)?, 5 + n)))
}

/// Skip shell noise up to and including the MAGIC line. `line_limit` is the longest line accepted (the Python hub's
/// StreamReader limit, MAX_DATA + 4096); a longer line is an error, as is more than MAX_PREAMBLE bytes of noise.
pub async fn find_magic_async<R: AsyncBufRead + Unpin>(
    r: &mut R,
    line_limit: usize,
) -> Result<usize, ReadError> {
    let mut skipped = 0usize;
    loop {
        let mut line = Vec::new();
        loop {
            let buf = r.fill_buf().await?;
            if buf.is_empty() {
                return Err(ProtoError::invalid(
                    "the far side closed before speaking the ecnode protocol",
                )
                .into());
            }
            let (done, used) = match buf.iter().position(|&b| b == b'\n') {
                Some(i) => {
                    line.extend_from_slice(&buf[..=i]);
                    (true, i + 1)
                }
                None => {
                    line.extend_from_slice(buf);
                    (false, buf.len())
                }
            };
            r.consume(used);
            if line.len() > line_limit {
                return Err(ProtoError::invalid(
                    "the far side printed something other than the ecnode protocol",
                )
                .into());
            }
            if done {
                break;
            }
        }
        if line.ends_with(MAGIC) {
            return Ok(skipped + line.len() - MAGIC.len());
        }
        skipped += line.len();
        if skipped > MAX_PREAMBLE {
            return Err(ProtoError::invalid(
                "the far side printed something other than the ecnode protocol",
            )
            .into());
        }
    }
}
