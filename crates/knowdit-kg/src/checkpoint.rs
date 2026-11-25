//! Checkpoint sink: wraps the journal (layer B) for use by the
//! extraction loops. Provides stage-scoped load/write/clear for a
//! single project.
//!
//! HARD INVARIANT: `None` (no sink) must produce byte-identical learn
//! behavior. Every checkpoint path is behind an `Option`.

use std::collections::HashMap;
use std::io;
use std::path::PathBuf;

use super::checkpoint_journal::{self, ChunkRecord, Journal};

/// Stage identifiers for the two extraction phases.
pub mod stage {
    pub const SEMANTIC_EXTRACT: &str = "semantic-extract";
    pub const FINDING_EXTRACT: &str = "finding-extract";
}

/// Per-project checkpoint sink. Manages one journal file per stage.
pub struct CheckpointSink {
    dir: PathBuf,
    project_key: String,
}

impl CheckpointSink {
    pub fn new(dir: impl Into<PathBuf>, project_key: impl Into<String>) -> Self {
        Self {
            dir: dir.into(),
            project_key: project_key.into(),
        }
    }

    /// Stable identifier for logging.
    pub fn project_key(&self) -> &str {
        &self.project_key
    }

    fn journal(&self, stage: &str) -> io::Result<Journal> {
        Journal::new(&self.dir, &self.project_key, stage)
    }

    /// Load all good checkpoint records for a stage.
    /// Torn tails are self-healed (truncated at the last good record).
    /// Returns `chunk_idx -> (hash, payload)` for every intact record.
    pub fn load_stage(&self, stage: &str) -> io::Result<HashMap<u32, (String, Vec<u8>)>> {
        let journal = self.journal(stage)?;
        let (records, last_good_offset) = journal.load()?;

        // Self-heal: if the file has bytes past the last good record,
        // a torn tail from a crash mid-append is present — truncate it
        // NOW so the next append lands after the last good record
        // (gate_20261101-002643 finding 1: the previous double-load
        // comparison was dead code; compare against actual file length)
        let file_len = match std::fs::metadata(journal.path()) {
            Ok(m) => m.len(),
            Err(e) if e.kind() == io::ErrorKind::NotFound => {
                // no journal file (cleared or first run): nothing to heal
                return Ok(records
                    .into_iter()
                    .map(|r| (r.chunk_idx, (r.hash, r.payload)))
                    .collect());
            }
            Err(e) => return Err(e),
        };
        if file_len > last_good_offset {
            journal.truncate_to(last_good_offset)?;
            tracing::debug!(
                "Checkpoint journal self-healed: truncated {} torn byte(s) past offset {}",
                file_len - last_good_offset,
                last_good_offset,
            );
        }

        Ok(records
            .into_iter()
            .map(|r| (r.chunk_idx, (r.hash, r.payload)))
            .collect())
    }

    /// Checkpoint a completed chunk for a stage.
    pub fn write_chunk(
        &self,
        stage: &str,
        chunk_idx: u32,
        hash: &str,
        payload: &[u8],
    ) -> io::Result<()> {
        let journal = self.journal(stage)?;
        journal.append(&ChunkRecord {
            chunk_idx,
            hash: hash.to_string(),
            payload: payload.to_vec(),
        })
    }

    /// Clear all checkpoints for this project (after successful commit).
    /// Removes every stage's journal file. Idempotent.
    pub fn clear(&self) -> io::Result<()> {
        let mut last_err = None;
        for stage in [stage::SEMANTIC_EXTRACT, stage::FINDING_EXTRACT] {
            match self.journal(stage) {
                Ok(j) => {
                    if let Err(e) = j.reset() {
                        last_err = Some(e);
                    }
                }
                Err(e) => last_err = Some(e),
            }
        }
        match last_err {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }
}

/// Chunk hash for resume validation: CRC32 + byte length.
/// Detects source-text changes between runs (the same text yields the
/// same hash; any edit invalidates the checkpoint for that chunk).
pub fn chunk_hash(text: &str) -> String {
    format!(
        "{:08x}-{}",
        checkpoint_journal::crc32(text.as_bytes()),
        text.len()
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmpdir() -> PathBuf {
        let d = std::env::temp_dir().join(format!(
            "ckpt-sink-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn write_load_roundtrip() {
        let dir = tmpdir();
        let sink = CheckpointSink::new(&dir, "test-project");
        sink.write_chunk(stage::SEMANTIC_EXTRACT, 0, "abc-100", b"payload0")
            .unwrap();
        sink.write_chunk(stage::SEMANTIC_EXTRACT, 1, "def-200", b"payload1")
            .unwrap();

        let loaded = sink.load_stage(stage::SEMANTIC_EXTRACT).unwrap();
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded[&0].1, b"payload0");
        assert_eq!(loaded[&1].1, b"payload1");

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn clear_removes_all_stages() {
        let dir = tmpdir();
        let sink = CheckpointSink::new(&dir, "test-project");
        sink.write_chunk(stage::SEMANTIC_EXTRACT, 0, "h-1", b"s")
            .unwrap();
        sink.write_chunk(stage::FINDING_EXTRACT, 0, "h-2", b"f")
            .unwrap();
        sink.clear().unwrap();

        let s = sink.load_stage(stage::SEMANTIC_EXTRACT).unwrap();
        assert!(s.is_empty());
        let f = sink.load_stage(stage::FINDING_EXTRACT).unwrap();
        assert!(f.is_empty());

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn chunk_hash_deterministic_and_sensitive() {
        let a = chunk_hash("hello world");
        let b = chunk_hash("hello world");
        let c = chunk_hash("hello worle");
        assert_eq!(a, b, "same text must produce the same hash");
        assert_ne!(a, c, "one-char change must produce a different hash");
    }

    #[test]
    fn missing_stage_yields_empty_map() {
        let dir = tmpdir();
        let sink = CheckpointSink::new(&dir, "no-such-project");
        let loaded = sink.load_stage(stage::SEMANTIC_EXTRACT).unwrap();
        assert!(loaded.is_empty());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn torn_tail_self_heals_on_load() {
        let dir = tmpdir();
        let sink = CheckpointSink::new(&dir, "torn-test");
        sink.write_chunk(stage::SEMANTIC_EXTRACT, 0, "h-0", b"good")
            .unwrap();

        // Simulate a torn tail: append garbage bytes after the good record
        let journal = Journal::new(&dir, "torn-test", stage::SEMANTIC_EXTRACT).unwrap();
        use std::io::Write;
        let mut f = std::fs::OpenOptions::new()
            .append(true)
            .open(journal.path())
            .unwrap();
        f.write_all(&[0xDE, 0xAD, 0xBE, 0xEF]).unwrap();

        // Load must return the good record AND truncate the garbage
        let loaded = sink.load_stage(stage::SEMANTIC_EXTRACT).unwrap();
        assert_eq!(loaded.len(), 1, "good record must survive");
        assert_eq!(loaded[&0].1, b"good");

        // Write a new chunk: must land after the good record, not after
        // the (now-truncated) garbage
        sink.write_chunk(stage::SEMANTIC_EXTRACT, 1, "h-1", b"new")
            .unwrap();
        let loaded2 = sink.load_stage(stage::SEMANTIC_EXTRACT).unwrap();
        assert_eq!(loaded2.len(), 2, "both records must be visible");
        assert_eq!(loaded2[&1].1, b"new");

        std::fs::remove_dir_all(&dir).ok();
    }
}
