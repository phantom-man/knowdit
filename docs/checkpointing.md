# Extraction checkpointing — design (2026-10-31)

Owner order: "I do want A + B, A is the ultimate backup and B is the
speedy resume. I think of it like the built-in redundancy of tcp/ip."

## Goal

A killed `learn` run currently redoes the whole in-memory contest. This
feature checkpoints **per extraction chunk** so a re-run resumes the
extraction phase instead of re-paying it (worst real case: a Sherlock
behemoth cost 6h20m / $12-15 on one contest).

## Two layers (TCP-style redundancy)

- **A — authoritative store**: a `checkpoint_chunk` staging table in
  the KG (`project_key, stage, chunk_idx, chunk_hash, payload JSON,
  created`). Written per chunk, deleted when the contest's final write
  transaction succeeds. This layer is the backup.
- **B — fast resume path**: an append-only per-project/stage journal
  file (`checkpoint_journal.rs`), one length-prefixed CRC32-stamped
  record per chunk.
- **Redundancy rule**: resume reads **B** first (fast, no SQLite
  contention); every B record is **validated against A** (coverage +
  hash). A torn/corrupt B tail is truncated at the last good record
  and re-filled from A (self-heal). If B is unusable, A alone
  reconstructs the state. B never overrules A.

## Hard invariant (must ship tested)

**No sink configured ⇒ byte-identical behavior to today.** The
checkpoint parameter is optional everywhere; `None` skips every
new code path. The checkpoint adds writes and a resume branch — it
never changes what a FROM-SCRATCH run extracts, merges, or writes.

## Scope: extraction ONLY (deliberate)

- **Checkpointed**: semantics-extraction chunks and findings-extraction
  chunks. Their outputs are KB-independent (function of project source
  + prompts only), so replaying a stored chunk output equals having
  run it.
- **NOT checkpointed — merges.** Merge decisions are LLM judgments
  against the LIVE knowledge base, which changes between crash and
  resume in multi-shard runs (other contests add semantics). Replaying
  stale merge decisions would be semantically wrong; merges re-run
  against the current KB by design. `categorize` is a single small
  call — also re-run.

## Resume key

Chunk boundaries are deterministic given the same source text and
token budget (`TokenCursor.next_chunk`), so the key is
`(project_key, stage, chunk_idx)` **plus a content hash of the chunk
text**: a staged record is reused only when the hash matches the chunk
about to be extracted. A corpus edit therefore invalidates exactly the
affected chunk — never silently feeds a stale extraction.

## Flow (per stage)

```
staged = sink.load_staged(stage)          # A, reconciled with B
while let Some(chunk) = cursor.next_chunk(budget) {
    if let Some(rec) = staged.get(chunk_idx) and rec.hash == hash(chunk) {
        extend(acc, decode(rec.payload));  # "resumed chunk N"
    } else {
        items = extract(chunk).await?;     # LLM
        extend(acc, items);
        sink.put(stage, chunk_idx, hash(chunk), encode(items));
    }
}
```

After the contest's final `write_project_completed_txn` commits:
`sink.clear(project_key)` (A rows deleted, B journal removed).
Staging rows only ever live for in-flight contests.

## Config

Wired at the command layer (`cmd/learn/learn.rs`) from
`KNOWDIT_CHECKPOINT_DIR` (unset ⇒ feature off ⇒ invariant holds).
The sink threads into `categorize_and_extract` →
`extract_semantics` / `extract_findings` as an `Option<&CheckpointSink>`.

## Acceptance test

Run `learn` on one contest; kill mid-extraction; re-run; logs show
`resumed chunk N` for the completed chunks, only the missing chunks hit
the LLM, and the final KG rows match a from-scratch run's shape.

## Status

- [x] design (this doc)
- [x] layer B: journal module + tests
- [ ] layer A: `checkpoint_chunk` entity + sink (A ops)
- [ ] wiring: sink param through `categorize_and_extract` /
      `extract_semantics` / `extract_findings`; env construction at
      `cmd/learn/learn.rs` + `cmd/workflow/learn.rs`; clear-after-commit
- [ ] kill/resume acceptance run