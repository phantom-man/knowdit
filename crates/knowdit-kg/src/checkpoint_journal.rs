//! Append-only per-project/stage checkpoint journal: the FAST resume
//! path ("B" in the owner's TCP/IP-redundancy framing, 2026-10-31).
//!
//! One record per extraction chunk. Records are length-prefixed and
//! CRC32-stamped; the authoritative staging table ("A") is
//! cross-checked by the caller, and a torn or corrupt tail is healed
//! by truncating at the last good record, then re-filled from A.
//!
//! Layout: file magic, then records
//!   [chunk_idx u32 LE][hash_len u8][hash utf8][payload_len u32 LE]
//!   [payload][crc32 u32 LE]      (crc32 over hash bytes || payload)

use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};

pub const MAGIC: &[u8] = b"KDCP1\n";

/// Guard against corrupt length fields steering huge allocations.
const MAX_PAYLOAD_LEN: u32 = 64 * 1024 * 1024;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChunkRecord {
    pub chunk_idx: u32,
    pub hash: String,
    pub payload: Vec<u8>,
}

/// Filename-safe component: keep `[A-Za-z0-9._-]`, replace the rest.
pub fn safe_key(key: &str) -> String {
    key.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '.' || c == '_' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

/// Standard CRC32 (reflected, poly 0xEDB88320); check value
/// `crc32(b"123456789") == 0xCBF43926` is pinned by a test.
pub fn crc32(data: &[u8]) -> u32 {
    let mut crc: u32 = 0xFFFF_FFFF;
    for &b in data {
        crc ^= b as u32;
        for _ in 0..8 {
            let mask = (crc & 1).wrapping_neg();
            crc = (crc >> 1) ^ (0xEDB8_8320 & mask);
        }
    }
    !crc
}

pub struct Journal {
    path: PathBuf,
}

impl Journal {
    pub fn new(dir: &Path, project_key: &str, stage: &str) -> io::Result<Self> {
        fs::create_dir_all(dir)?;
        let name = format!("{}.{}.journal", safe_key(project_key), safe_key(stage));
        Ok(Self {
            path: dir.join(name),
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Append one record; writes the magic first when the file is new.
    pub fn append(&self, rec: &ChunkRecord) -> io::Result<()> {
        let mut f = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)?;
        if f.metadata()?.len() == 0 {
            f.write_all(MAGIC)?;
        }
        let hash_bytes = rec.hash.as_bytes();
        f.write_all(&rec.chunk_idx.to_le_bytes())?;
        f.write_all(&[hash_bytes.len() as u8])?;
        f.write_all(hash_bytes)?;
        f.write_all(&(rec.payload.len() as u32).to_le_bytes())?;
        f.write_all(&rec.payload)?;
        let mut crc_data = Vec::with_capacity(hash_bytes.len() + rec.payload.len());
        crc_data.extend_from_slice(hash_bytes);
        crc_data.extend_from_slice(&rec.payload);
        f.write_all(&crc32(&crc_data).to_le_bytes())?;
        f.flush()
    }

    /// All good records plus the byte offset just past the last good
    /// one. A torn/corrupt tail (short read, bad crc, implausible
    /// length) stops the walk. A missing file or bad magic yields
    /// `(empty, 0)` — the caller heals and rebuilds from A.
    pub fn load(&self) -> io::Result<(Vec<ChunkRecord>, u64)> {
        let bytes = match fs::read(&self.path) {
            Ok(b) => b,
            Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok((Vec::new(), 0)),
            Err(e) => return Err(e),
        };
        if bytes.len() < MAGIC.len() || &bytes[..MAGIC.len()] != MAGIC {
            return Ok((Vec::new(), 0));
        }
        let mut recs = Vec::new();
        let mut pos = MAGIC.len();
        loop {
            let start = pos;
            if pos + 4 > bytes.len() {
                return Ok((recs, start as u64));
            }
            let chunk_idx = u32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap());
            pos += 4;
            if pos >= bytes.len() {
                return Ok((recs, start as u64));
            }
            let hash_len = bytes[pos] as usize;
            pos += 1;
            if pos + hash_len + 4 > bytes.len() {
                return Ok((recs, start as u64));
            }
            let hash = match std::str::from_utf8(&bytes[pos..pos + hash_len]) {
                Ok(h) => h.to_string(),
                Err(_) => return Ok((recs, start as u64)),
            };
            pos += hash_len;
            let payload_len = u32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap());
            pos += 4;
            if payload_len > MAX_PAYLOAD_LEN {
                return Ok((recs, start as u64));
            }
            let pl = payload_len as usize;
            if pos + pl + 4 > bytes.len() {
                return Ok((recs, start as u64));
            }
            let payload = bytes[pos..pos + pl].to_vec();
            pos += pl;
            let stored_crc = u32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap());
            pos += 4;
            let mut crc_data = Vec::with_capacity(hash_len + pl);
            crc_data.extend_from_slice(hash.as_bytes());
            crc_data.extend_from_slice(&payload);
            if crc32(&crc_data) != stored_crc {
                return Ok((recs, start as u64));
            }
            recs.push(ChunkRecord {
                chunk_idx,
                hash,
                payload,
            });
        }
    }

    /// Self-heal primitive: drop everything from `offset` onward.
    pub fn truncate_to(&self, offset: u64) -> io::Result<()> {
        let f = fs::OpenOptions::new().write(true).open(&self.path)?;
        f.set_len(offset)
    }

    /// Success path: remove this journal entirely.
    pub fn reset(&self) -> io::Result<()> {
        match fs::remove_file(&self.path) {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(()),
            Err(e) => Err(e),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// std-only temp dir (the crate has no `tempfile` dev-dependency).
    fn tmpdir() -> PathBuf {
        let p = std::env::temp_dir().join(format!(
            "kdckpt-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = fs::remove_dir_all(&p);
        fs::create_dir_all(&p).unwrap();
        p
    }

    fn rec(idx: u32, hash: &str, payload: &[u8]) -> ChunkRecord {
        ChunkRecord {
            chunk_idx: idx,
            hash: hash.to_string(),
            payload: payload.to_vec(),
        }
    }

    #[test]
    fn crc32_pins_standard_check_value() {
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
    }

    #[test]
    fn safe_key_sanitizes() {
        assert_eq!(safe_key("c4-1007"), "c4-1007");
        assert_eq!(safe_key("a:b/c d"), "a_b_c_d");
    }

    #[test]
    fn roundtrip_two_records() {
        let td = tmpdir();
        let j = Journal::new(&td, "c4-1007", "semantics").unwrap();
        j.append(&rec(0, "h0", b"one")).unwrap();
        j.append(&rec(1, "h1", b"two")).unwrap();
        let (recs, last) = j.load().unwrap();
        assert_eq!(recs, vec![rec(0, "h0", b"one"), rec(1, "h1", b"two")]);
        assert!(last > 0);
        assert_eq!(
            last as usize,
            fs::metadata(j.path()).unwrap().len() as usize
        );
    }

    #[test]
    fn torn_tail_truncates_and_heals() {
        let td = tmpdir();
        let j = Journal::new(&td, "p", "semantics").unwrap();
        j.append(&rec(0, "h0", b"one")).unwrap();
        j.append(&rec(1, "h1", b"two")).unwrap();
        let (_, last) = j.load().unwrap();
        // tear the tail by 3 bytes
        let f = fs::OpenOptions::new().write(true).open(j.path()).unwrap();
        f.set_len(last - 3).unwrap();
        drop(f);
        let (recs, last2) = j.load().unwrap();
        assert_eq!(recs, vec![rec(0, "h0", b"one")]);
        // end of rec 0 = MAGIC(6) + 4+1+2+4+3+4 = 24
        assert_eq!(last2, (MAGIC.len() + 18) as u64);
        j.truncate_to(last2).unwrap();
        j.append(&rec(2, "h2", b"three")).unwrap();
        let (recs3, _) = j.load().unwrap();
        assert_eq!(recs3.len(), 2);
        assert_eq!(recs3[1], rec(2, "h2", b"three"));
    }

    #[test]
    fn crc_mismatch_rejects_record() {
        let td = tmpdir();
        let j = Journal::new(&td, "p", "semantics").unwrap();
        j.append(&rec(0, "h0", b"one")).unwrap();
        j.append(&rec(1, "h1", b"two")).unwrap();
        let mut bytes = fs::read(j.path()).unwrap();
        let n = bytes.len();
        bytes[n - 1] ^= 0xFF; // flip a crc byte of the LAST record
        fs::write(j.path(), &bytes).unwrap();
        let (recs, _) = j.load().unwrap();
        assert_eq!(recs, vec![rec(0, "h0", b"one")]);
    }

    #[test]
    fn missing_file_and_bad_magic_yield_zero_good() {
        let td = tmpdir();
        let j = Journal::new(&td, "p", "semantics").unwrap();
        assert_eq!(j.load().unwrap(), (Vec::new(), 0));
        fs::write(j.path(), b"garbage-not-magic").unwrap();
        assert_eq!(j.load().unwrap(), (Vec::new(), 0));
    }

    #[test]
    fn reset_removes_and_is_idempotent() {
        let td = tmpdir();
        let j = Journal::new(&td, "p", "semantics").unwrap();
        j.append(&rec(0, "h0", b"one")).unwrap();
        assert!(j.path().exists());
        j.reset().unwrap();
        assert!(!j.path().exists());
        j.reset().unwrap(); // idempotent
    }

    #[test]
    fn implausible_length_rejected() {
        let td = tmpdir();
        let j = Journal::new(&td, "p", "semantics").unwrap();
        let mut bytes = MAGIC.to_vec();
        bytes.extend_from_slice(&0u32.to_le_bytes()); // chunk_idx
        bytes.push(2); // hash_len
        bytes.extend_from_slice(b"h0");
        bytes.extend_from_slice(&(MAX_PAYLOAD_LEN + 1).to_le_bytes());
        fs::write(j.path(), &bytes).unwrap();
        assert_eq!(j.load().unwrap(), (Vec::new(), MAGIC.len() as u64));
    }
}
