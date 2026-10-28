//! Python: node/src/ecnode/gguf.py.
//!
//! A minimal, strictly bounded GGUF header reader. It reads only what a model library needs:
//! general.architecture / name / file_type, a few `{arch}.*` sizes and the total tensor element count
//! (parameters). Every length and count in a GGUF header is untrusted: before anything is read or skipped it is
//! checked against a fixed byte budget, so a truncated or hostile file can neither fail out of this module nor
//! make it allocate more than MAX_KEY (a key) or MAX_KEEP_STR (a kept string) bytes per value.

use std::collections::HashMap;
use std::io::{BufReader, Read};
use std::os::fd::OwnedFd;
use std::path::Path;

use ec_proto::Obj;
use nix::fcntl::OFlag;
use nix::sys::stat::Mode;
use serde_json::{Value, json};

pub const MAGIC: &[u8] = b"GGUF";
/// bytes of header this reader will consume, at most
pub const MAX_HEADER_BYTES: u64 = 16 * 1024 * 1024;
pub const MAX_KV: u64 = 1_000_000;
pub const MAX_TENSORS: u64 = 1_000_000;
pub const MAX_DIMS: u32 = 8;
pub const MAX_KEY: u64 = 65_535;
pub const MAX_KEEP_STR: u64 = 4096;
pub const MAX_DEPTH: usize = 3;

// value types
const U8: u32 = 0;
const I8: u32 = 1;
const U16: u32 = 2;
const I16: u32 = 3;
const U32: u32 = 4;
const I32: u32 = 5;
const F32: u32 = 6;
const BOOL: u32 = 7;
const STR: u32 = 8;
const ARR: u32 = 9;
const U64: u32 = 10;
const I64: u32 = 11;
const F64: u32 = 12;

/// `struct.calcsize` of each fixed-size type.
fn fixed_size(t: u32) -> Option<u64> {
    match t {
        U8 | I8 | BOOL => Some(1),
        U16 | I16 => Some(2),
        U32 | I32 | F32 => Some(4),
        U64 | I64 | F64 => Some(8),
        _ => None,
    }
}

pub const ARCH_KEYS: &[&str] = &[
    "context_length",
    "block_count",
    "embedding_length",
    "attention.head_count",
    "attention.head_count_kv",
    "expert_count",
    "expert_used_count",
];
pub const GENERAL_KEYS: &[&str] = &[
    "general.architecture",
    "general.name",
    "general.file_type",
    "general.size_label",
    "general.basename",
    "split.count",
    "split.no",
];

/// llama.cpp LLAMA_FTYPE values (general.file_type) -> the quant name people use
pub fn file_type_name(ft: i128) -> Option<&'static str> {
    Some(match ft {
        0 => "F32",
        1 => "F16",
        2 => "Q4_0",
        3 => "Q4_1",
        7 => "Q8_0",
        8 => "Q5_0",
        9 => "Q5_1",
        10 => "Q2_K",
        11 => "Q3_K_S",
        12 => "Q3_K_M",
        13 => "Q3_K_L",
        14 => "Q4_K_S",
        15 => "Q4_K_M",
        16 => "Q5_K_S",
        17 => "Q5_K_M",
        18 => "Q6_K",
        19 => "IQ2_XXS",
        20 => "IQ2_XS",
        21 => "Q2_K_S",
        22 => "IQ3_XS",
        23 => "IQ3_XXS",
        24 => "IQ1_S",
        25 => "IQ4_NL",
        26 => "IQ3_S",
        27 => "IQ3_M",
        28 => "IQ2_S",
        29 => "IQ2_M",
        30 => "IQ4_XS",
        31 => "IQ1_M",
        32 => "BF16",
        36 => "TQ1_0",
        37 => "TQ2_0",
        38 => "MXFP4_MOE",
        _ => return None,
    })
}

/// Out of budget, out of file, or malformed: stop reading and keep what was found.
struct Stop;

type R<T> = Result<T, Stop>;

/// A value read from the header (the Python object `struct.unpack` / `_read_str` gives). Floats are kept as
/// values although no fact uses them (Python keeps them too).
#[derive(Debug, Clone, PartialEq)]
#[allow(dead_code)]
enum Kv {
    Int(i128),
    Float(f64),
    Bool(bool),
    Str(String),
    /// a string longer than MAX_KEEP_STR, or an array: skipped
    None,
}

struct Reader<'a, T: Read> {
    f: &'a mut T,
    left: u64,
}

impl<T: Read> Reader<'_, T> {
    fn take(&mut self, n: u64) -> R<Vec<u8>> {
        if n > self.left {
            return Err(Stop);
        }
        let len = usize::try_from(n).map_err(|_| Stop)?;
        let mut b = vec![0u8; len];
        self.f.read_exact(&mut b).map_err(|_| Stop)?;
        self.left -= n;
        Ok(b)
    }

    fn skip(&mut self, n: u64) -> R<()> {
        if n > self.left {
            return Err(Stop);
        }
        self.left -= n;
        // read, not seek: a seek past the end would not tell us the file is truncated
        let got =
            std::io::copy(&mut (&mut *self.f).take(n), &mut std::io::sink()).map_err(|_| Stop)?;
        if got != n { Err(Stop) } else { Ok(()) }
    }

    fn arr<const N: usize>(&mut self) -> R<[u8; N]> {
        let b = self.take(N as u64)?;
        <[u8; N]>::try_from(b.as_slice()).map_err(|_| Stop)
    }

    fn u32(&mut self) -> R<u32> {
        Ok(u32::from_le_bytes(self.arr::<4>()?))
    }

    fn u64(&mut self) -> R<u64> {
        Ok(u64::from_le_bytes(self.arr::<8>()?))
    }

    /// a count or length: u64 in v2+, u32 in v1
    fn len(&mut self, wide: bool) -> R<u64> {
        if wide {
            self.u64()
        } else {
            self.u32().map(u64::from)
        }
    }

    fn fixed(&mut self, t: u32) -> R<Kv> {
        Ok(match t {
            U8 => Kv::Int(i128::from(self.arr::<1>()?[0])),
            I8 => Kv::Int(i128::from(i8::from_le_bytes(self.arr::<1>()?))),
            U16 => Kv::Int(i128::from(u16::from_le_bytes(self.arr::<2>()?))),
            I16 => Kv::Int(i128::from(i16::from_le_bytes(self.arr::<2>()?))),
            U32 => Kv::Int(i128::from(u32::from_le_bytes(self.arr::<4>()?))),
            I32 => Kv::Int(i128::from(i32::from_le_bytes(self.arr::<4>()?))),
            F32 => Kv::Float(f64::from(f32::from_le_bytes(self.arr::<4>()?))),
            BOOL => Kv::Bool(self.arr::<1>()?[0] != 0),
            U64 => Kv::Int(i128::from(u64::from_le_bytes(self.arr::<8>()?))),
            I64 => Kv::Int(i128::from(i64::from_le_bytes(self.arr::<8>()?))),
            F64 => Kv::Float(f64::from_le_bytes(self.arr::<8>()?)),
            _ => return Err(Stop),
        })
    }

    fn read_str(&mut self, wide: bool, keep: u64) -> R<Option<String>> {
        let n = self.len(wide)?;
        if n > keep {
            self.skip(n)?;
            return Ok(None);
        }
        let b = self.take(n)?;
        Ok(Some(String::from_utf8_lossy(&b).into_owned()))
    }

    fn skip_value(&mut self, vtype: u32, wide: bool, depth: usize) -> R<()> {
        if let Some(n) = fixed_size(vtype) {
            return self.skip(n);
        }
        match vtype {
            STR => {
                let n = self.len(wide)?;
                self.skip(n)
            }
            ARR => {
                if depth >= MAX_DEPTH {
                    return Err(Stop);
                }
                let etype = self.u32()?;
                let count = self.len(wide)?;
                if let Some(size) = fixed_size(etype) {
                    // skip() refuses anything beyond the budget
                    self.skip(count.checked_mul(size).ok_or(Stop)?)
                } else if etype == STR || etype == ARR {
                    if count > self.left / 4 {
                        // every element costs at least 4 bytes
                        return Err(Stop);
                    }
                    for _ in 0..count {
                        self.skip_value(etype, wide, depth + 1)?;
                    }
                    Ok(())
                } else {
                    Err(Stop)
                }
            }
            _ => Err(Stop),
        }
    }

    fn read_value(&mut self, vtype: u32, wide: bool) -> R<Kv> {
        if fixed_size(vtype).is_some() {
            return self.fixed(vtype);
        }
        if vtype == STR {
            return Ok(self.read_str(wide, MAX_KEEP_STR)?.map_or(Kv::None, Kv::Str));
        }
        self.skip_value(vtype, wide, 0)?;
        Ok(Kv::None)
    }
}

fn int_value(i: i128) -> Value {
    if let Ok(v) = i64::try_from(i) {
        json!(v)
    } else if let Ok(v) = u64::try_from(i) {
        json!(v)
    } else {
        Value::Null
    }
}

/// `_s`: a string, capped at 256 characters
fn s_of(v: Option<&Kv>) -> Value {
    match v {
        Some(Kv::Str(s)) => Value::String(s.chars().take(256).collect()),
        _ => Value::Null,
    }
}

/// `_i`: an int (not a bool) from 0 to 2**53
fn i_of(v: Option<&Kv>) -> Value {
    match v {
        Some(Kv::Int(i)) if (0..=(1i128 << 53)).contains(i) => int_value(*i),
        _ => Value::Null,
    }
}

fn read_body<T: Read>(
    r: &mut Reader<'_, T>,
    wide: bool,
    tensors: bool,
    kv: &mut HashMap<String, Kv>,
    out: &mut Obj,
) -> R<()> {
    let n_tensors = r.len(wide)?;
    let n_kv = r.len(wide)?;
    if n_kv > MAX_KV || n_tensors > MAX_TENSORS {
        return Err(Stop);
    }
    for _ in 0..n_kv {
        let klen = r.len(wide)?;
        if klen > MAX_KEY {
            return Err(Stop);
        }
        let key = String::from_utf8_lossy(&r.take(klen)?).into_owned();
        let vtype = r.u32()?;
        if GENERAL_KEYS.contains(&key.as_str())
            || ARCH_KEYS.iter().any(|s| key.ends_with(&format!(".{s}")))
        {
            let v = r.read_value(vtype, wide)?;
            kv.insert(key, v);
        } else {
            r.skip_value(vtype, wide, 0)?;
        }
    }
    if tensors {
        let mut total: u128 = 0;
        for _ in 0..n_tensors {
            r.read_str(wide, 0)?;
            let n_dims = r.u32()?;
            if n_dims > MAX_DIMS {
                return Err(Stop);
            }
            let mut count: u128 = 1;
            for _ in 0..n_dims {
                count = count.saturating_mul(u128::from(r.u64()?));
            }
            r.skip(4 + 8)?; // type, offset
            total = total.saturating_add(count);
        }
        out.insert("tensors".into(), json!(n_tensors));
        out.insert(
            "params".into(),
            match u64::try_from(total) {
                Ok(t) if total <= 1u128 << 53 => json!(t),
                _ => Value::Null,
            },
        );
    }
    out.insert("complete".into(), json!(true));
    Ok(())
}

/// Facts from a GGUF header, or None if this is not a GGUF file. Never fails for bad input.
/// Result keys: version, complete, (tensors, params), arch, name, size_label, basename, file_type, quant,
/// split_count, context, block_count, embedding_length, head_count, head_count_kv, expert_count,
/// expert_used_count.
pub fn read_header<T: Read>(f: &mut T, budget: u64, tensors: bool) -> Option<Obj> {
    let mut r = Reader { f, left: budget };
    let magic = r.take(4).ok()?;
    if magic != MAGIC {
        return None;
    }
    let version = r.u32().ok()?;
    let mut out = Obj::new();
    out.insert("version".into(), json!(version));
    out.insert("complete".into(), json!(false));
    if !(1..=3).contains(&version) {
        return Some(out);
    }
    let wide = version >= 2;
    let mut kv: HashMap<String, Kv> = HashMap::new();
    let _ = read_body(&mut r, wide, tensors, &mut kv, &mut out);
    let arch = match kv.get("general.architecture") {
        Some(Kv::Str(s)) => Some(s.clone()),
        _ => None,
    };
    let ft = kv.get("general.file_type");
    out.insert(
        "arch".into(),
        arch.clone().map_or(Value::Null, Value::String),
    );
    out.insert("name".into(), s_of(kv.get("general.name")));
    out.insert("size_label".into(), s_of(kv.get("general.size_label")));
    out.insert("basename".into(), s_of(kv.get("general.basename")));
    out.insert(
        "file_type".into(),
        match ft {
            Some(Kv::Int(i)) => int_value(*i),
            _ => Value::Null,
        },
    );
    // FILE_TYPES.get(ft) for any int, bools included (True == 1)
    let ft_int = match ft {
        Some(Kv::Int(i)) => Some(*i),
        Some(Kv::Bool(b)) => Some(i128::from(*b)),
        _ => None,
    };
    out.insert(
        "quant".into(),
        ft_int
            .and_then(file_type_name)
            .map_or(Value::Null, |q| Value::String(q.into())),
    );
    out.insert("split_count".into(), i_of(kv.get("split.count")));
    for (suffix, field) in [
        ("context_length", "context"),
        ("block_count", "block_count"),
        ("embedding_length", "embedding_length"),
        ("attention.head_count", "head_count"),
        ("attention.head_count_kv", "head_count_kv"),
        ("expert_count", "expert_count"),
        ("expert_used_count", "expert_used_count"),
    ] {
        let v = match &arch {
            Some(a) => i_of(kv.get(&format!("{a}.{suffix}"))),
            None => Value::Null,
        };
        out.insert(field.into(), v);
    }
    Some(out)
}

/// read_header on a file path (or an already opened fd, which is closed afterwards). Only a regular file is read
/// (a FIFO, socket or device gives None: node hardening); the reader consumes at most `budget` bytes.
pub fn read_file(path: &Path, fd: Option<OwnedFd>, budget: u64, tensors: bool) -> Option<Obj> {
    let fd = match fd {
        Some(fd) => fd,
        None => nix::fcntl::open(
            path,
            OFlag::O_RDONLY | OFlag::O_CLOEXEC | OFlag::O_NONBLOCK,
            Mode::empty(),
        )
        .ok()?,
    };
    let st = nix::sys::stat::fstat(std::os::fd::AsFd::as_fd(&fd)).ok()?;
    if !crate::util::fsread::is_regular(&st) {
        return None;
    }
    let mut f = BufReader::with_capacity(1 << 16, std::fs::File::from(fd));
    read_header(&mut f, budget, tensors)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn not_gguf_and_short_files() {
        assert_eq!(read_header(&mut &b"NOPE"[..], MAX_HEADER_BYTES, true), None);
        assert_eq!(read_header(&mut &b"GG"[..], MAX_HEADER_BYTES, true), None);
        assert_eq!(
            read_header(&mut &b"GGUF\x01"[..], MAX_HEADER_BYTES, true),
            None
        );
    }

    #[test]
    fn bool_file_type_maps_like_python() {
        // general.file_type stored as BOOL true: file_type None, quant F16 (FILE_TYPES.get(True))
        let mut blob = b"GGUF".to_vec();
        blob.extend(3u32.to_le_bytes());
        blob.extend(0u64.to_le_bytes());
        blob.extend(1u64.to_le_bytes());
        let key = b"general.file_type";
        blob.extend((key.len() as u64).to_le_bytes());
        blob.extend(key);
        blob.extend(BOOL.to_le_bytes());
        blob.push(1);
        let h = read_header(&mut blob.as_slice(), MAX_HEADER_BYTES, true).unwrap_or_default();
        assert_eq!(h["file_type"], Value::Null);
        assert_eq!(h["quant"], json!("F16"));
        assert_eq!(h["complete"], json!(true));
    }
}
