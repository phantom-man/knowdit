use crate::cmd::learn::finding_link_args::FindingLinkCliArgs;
use crate::cmd::learn::merge_args::MergeCliArgs;
use clap::{Args, ValueEnum};
use color_eyre::eyre::{Result, WrapErr};
use knowdit_kg::db::HistoricalDatabase;
use knowdit_kg::error::KgError;
use knowdit_kg::learn::{ExtractResult, FindingLinkOptions};
use knowdit_kg::project_loader::{MovePlatform, ProjectData};
use knowdit_kg_model::db::operation_history::OperationType;
use llmy::clap::OpenAISetup;
use std::collections::HashMap;
use std::path::PathBuf;
use tokio::sync::mpsc;
use tokio::task::JoinSet;

#[derive(Args)]
pub struct LearnArgs {
    #[command(flatten)]
    pub llm: OpenAISetup,

    /// Project directories to learn. Format: "name:path" or "name:path:platform_id"
    #[arg(long = "project", short = 'p')]
    pub projects: Vec<String>,

    /// Number of projects to categorize+extract concurrently. Merge has its
    /// own knob: `--merge-concurrency` (see `MergeCliArgs`).
    #[arg(long, default_value_t = 1)]
    pub concurrency: usize,

    /// Run finding-to-semantic linking after all projects are written
    #[arg(long)]
    pub link: bool,

    #[command(flatten)]
    pub merge: MergeCliArgs,

    #[command(flatten)]
    pub finding_link: FindingLinkCliArgs,
}

#[derive(Args)]
pub struct LearnC4Args {
    #[command(flatten)]
    pub llm: OpenAISetup,

    /// Code4rena data directory (expects audits/ and contracts/ subdirs)
    #[arg(long)]
    pub c4_dir: PathBuf,

    /// Specific Code4rena contest IDs to process
    #[arg(long, value_delimiter = ',')]
    pub c4_ids: Vec<u32>,

    /// Code4rena contest IDs to skip. Applied after `--c4-ids`
    /// selection (or the full discovered set when `--c4-ids` is
    /// empty) and before `--limit` truncates. Useful for resuming
    /// past projects that previously failed to load or were already
    /// learned out-of-band.
    #[arg(long, value_delimiter = ',')]
    pub skip_ids: Vec<u32>,

    /// Maximum number of C4 projects to process (when no --c4-ids given)
    #[arg(long, default_value_t = 5)]
    pub limit: usize,

    /// Number of projects to categorize+extract concurrently. Merge has its
    /// own knob: `--merge-concurrency` (see `MergeCliArgs`).
    #[arg(long, default_value_t = 1)]
    pub concurrency: usize,

    /// Run finding-to-semantic linking after all projects are written
    #[arg(long)]
    pub link: bool,

    #[command(flatten)]
    pub merge: MergeCliArgs,

    #[command(flatten)]
    pub finding_link: FindingLinkCliArgs,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum MovePlatformArg {
    Aptos,
    Sui,
}

impl From<MovePlatformArg> for MovePlatform {
    fn from(value: MovePlatformArg) -> Self {
        match value {
            MovePlatformArg::Aptos => MovePlatform::Aptos,
            MovePlatformArg::Sui => MovePlatform::Sui,
        }
    }
}

#[derive(Args)]
pub struct LearnMovesArgs {
    #[command(flatten)]
    pub llm: OpenAISetup,

    /// Move dataset directory (expects _codebase_apt/_codebase_sui and vulnerability dirs)
    #[arg(long, default_value = "moves")]
    pub moves_dir: PathBuf,

    /// Specific Move project commit hashes to process
    #[arg(long, value_delimiter = ',')]
    pub commits: Vec<String>,

    /// Restrict the dataset to specific Move ecosystems
    #[arg(long, value_enum, value_delimiter = ',')]
    pub platforms: Vec<MovePlatformArg>,

    /// Number of projects to categorize+extract concurrently. Merge has its
    /// own knob: `--merge-concurrency` (see `MergeCliArgs`).
    #[arg(long, default_value_t = 1)]
    pub concurrency: usize,

    /// Run finding-to-semantic linking after all projects are written
    #[arg(long)]
    pub link: bool,

    #[command(flatten)]
    pub merge: MergeCliArgs,

    #[command(flatten)]
    pub finding_link: FindingLinkCliArgs,
}

impl LearnArgs {
    pub async fn run(self, db: &HistoricalDatabase) -> Result<()> {
        self.merge.validate()?;
        self.finding_link.validate()?;

        let llm = self.llm.clone().to_llm().await;
        let mut all_projects = Vec::new();

        for spec in &self.projects {
            match ProjectData::from_path_spec(spec).await {
                Ok(data) => all_projects.push(data),
                Err(e) => tracing::error!("Failed to load project '{}': {}", spec, e),
            }
        }

        run_pipeline(
            db,
            &llm,
            all_projects,
            self.concurrency,
            self.link,
            self.merge.to_agent_options(),
            self.merge.to_chunking_options(),
            self.finding_link.to_options(self.concurrency),
        )
        .await
    }
}

impl LearnC4Args {
    pub async fn run(self, db: &HistoricalDatabase) -> Result<()> {
        self.merge.validate()?;
        self.finding_link.validate()?;

        let llm = self.llm.clone().to_llm().await;
        let mut all_projects = Vec::new();

        let contest_ids: Vec<u32> = if !self.c4_ids.is_empty() {
            self.c4_ids
        } else {
            let mut all = knowdit_kg::project_loader::list_contest_ids(&self.c4_dir)?;
            all.sort_unstable_by(|a, b| b.cmp(a));
            all.into_iter().collect()
        };

        let skip: std::collections::HashSet<u32> = self.skip_ids.iter().copied().collect();
        let contest_ids: Vec<u32> = contest_ids
            .into_iter()
            .filter(|id| {
                if skip.contains(id) {
                    tracing::info!("Skipping c4 contest {} (in --skip-ids)", id);
                    false
                } else {
                    true
                }
            })
            .collect();

        for c4id in &contest_ids {
            match ProjectData::from_c4(&self.c4_dir, *c4id).await {
                Ok(data) => all_projects.push(data),
                Err(e) => tracing::error!("Failed to load c4 project {}: {}", c4id, e),
            }
        }

        // Captured before the pipeline runs, recorded only after it succeeds:
        // a killed run leaves no operation_history row (see `record_operation`).
        let merge_args = serde_json::to_value(&self.merge)
            .wrap_err("failed to serialize c4learn merge args for operation_history")?;
        run_pipeline(
            db,
            &llm,
            all_projects,
            self.concurrency,
            self.link,
            self.merge.to_agent_options(),
            self.merge.to_chunking_options(),
            self.finding_link.to_options(self.concurrency),
        )
        .await?;
        db.record_operation(OperationType::C4Learn, merge_args)
            .await?;
        Ok(())
    }
}

impl LearnMovesArgs {
    pub async fn run(self, db: &HistoricalDatabase) -> Result<()> {
        self.merge.validate()?;
        self.finding_link.validate()?;

        let llm = self.llm.clone().to_llm().await;
        let platforms: Vec<MovePlatform> = self.platforms.iter().copied().map(Into::into).collect();
        let audit_reports =
            knowdit_kg::project_loader::load_move_audit_reports(&self.moves_dir, &platforms)?;
        let discovered =
            knowdit_kg::project_loader::list_move_projects(&self.moves_dir, &platforms)?;
        let mut all_projects = Vec::new();

        if !self.commits.is_empty() {
            let mut by_commit: HashMap<String, knowdit_kg::project_loader::MoveProjectDescriptor> =
                discovered
                    .into_iter()
                    .map(|project| (project.commit_hash.clone(), project))
                    .collect();

            for commit_hash in &self.commits {
                match by_commit.remove(commit_hash) {
                    Some(project) => {
                        let report = audit_reports.get(&project.commit_hash).cloned();
                        all_projects.push(
                            ProjectData::from_move_snapshot(
                                &project.name,
                                &project.root_dir,
                                &project.commit_hash,
                                report,
                            )
                            .await?,
                        );
                    }
                    None => tracing::error!(
                        "Move project commit {} not found under {}",
                        commit_hash,
                        self.moves_dir.display()
                    ),
                }
            }
        } else {
            for project in discovered.into_iter() {
                let report = audit_reports.get(&project.commit_hash).cloned();
                match ProjectData::from_move_snapshot(
                    &project.name,
                    &project.root_dir,
                    &project.commit_hash,
                    report,
                )
                .await
                {
                    Ok(project) => {
                        all_projects.push(project);
                    }
                    Err(e) => {
                        tracing::warn!("Skipping {} due to {}", project.root_dir.display(), e);
                    }
                }
            }
        }

        run_pipeline(
            db,
            &llm,
            all_projects,
            self.concurrency,
            self.link,
            self.merge.to_agent_options(),
            self.merge.to_chunking_options(),
            self.finding_link.to_options(self.concurrency),
        )
        .await
    }
}

#[derive(Args)]
pub struct LearnSherlockArgs {
    #[command(flatten)]
    pub llm: OpenAISetup,

    /// sherlock-scrape output directory (expects metadata/, source/, reports/)
    #[arg(long)]
    pub sherlock_dir: PathBuf,

    /// Specific Sherlock contest IDs to process (default: all discovered)
    #[arg(long, value_delimiter = ',')]
    pub sherlock_ids: Vec<u32>,

    /// Contest IDs to skip, applied after --sherlock-ids / discovery
    #[arg(long, value_delimiter = ',')]
    pub skip_ids: Vec<u32>,

    /// Max contests to process (0 = no limit). Applied after skip.
    #[arg(long, default_value_t = 0)]
    pub limit: usize,

    /// Projects to categorize+extract concurrently (merge is always serial)
    #[arg(long, default_value_t = 1)]
    pub concurrency: usize,

    /// Run finding-to-semantic linking after all projects are written
    #[arg(long)]
    pub link: bool,

    #[command(flatten)]
    pub merge: MergeCliArgs,

    #[command(flatten)]
    pub finding_link: FindingLinkCliArgs,
}

impl LearnSherlockArgs {
    pub async fn run(self, db: &HistoricalDatabase) -> Result<()> {
        self.merge.validate()?;
        self.finding_link.validate()?;

        let llm = self.llm.clone().to_llm().await;

        let mut ids: Vec<u32> = if !self.sherlock_ids.is_empty() {
            self.sherlock_ids.clone()
        } else {
            // Discover contests by their `metadata/<id>.json` files.
            let meta_dir = self.sherlock_dir.join("metadata");
            let mut all = Vec::new();
            for entry in std::fs::read_dir(&meta_dir)
                .wrap_err_with(|| format!("failed to read {}", meta_dir.display()))?
            {
                let path = entry?.path();
                if path.extension().and_then(|ext| ext.to_str()) == Some("json")
                    && let Some(id) = path
                        .file_stem()
                        .and_then(|stem| stem.to_str())
                        .and_then(|stem| stem.parse::<u32>().ok())
                {
                    all.push(id);
                }
            }
            all.sort_unstable();
            all
        };
        let skip: std::collections::HashSet<u32> = self.skip_ids.iter().copied().collect();
        ids.retain(|id| !skip.contains(id));
        if self.limit > 0 && ids.len() > self.limit {
            ids.truncate(self.limit);
        }

        let mut all_projects = Vec::new();
        for id in &ids {
            match ProjectData::from_sherlock(&self.sherlock_dir, *id).await {
                Ok(Some(data)) => all_projects.push(data),
                Ok(None) => {
                    tracing::info!("Skipping sherlock contest {} (no scope / unsupported)", id)
                }
                Err(e) => tracing::error!("Failed to load sherlock contest {}: {}", id, e),
            }
        }
        tracing::info!(
            "Loaded {} ingestible sherlock contest(s)",
            all_projects.len()
        );

        // Captured before the pipeline runs, recorded only after it succeeds:
        // a killed run leaves no operation_history row (see `record_operation`).
        let merge_args = serde_json::to_value(&self.merge)
            .wrap_err("failed to serialize sherlock learn merge args for operation_history")?;
        run_pipeline(
            db,
            &llm,
            all_projects,
            self.concurrency,
            self.link,
            self.merge.to_agent_options(),
            self.merge.to_chunking_options(),
            self.finding_link.to_options(self.concurrency),
        )
        .await?;
        db.record_operation(OperationType::SherlockLearn, merge_args)
            .await?;
        Ok(())
    }
}

/// Decide the pipeline's outcome from the projects that were LOST.
///
/// A run that lost a project must NOT report success: the caller keys on the
/// exit code, and `tranche_runner.classify_result()` keys on the
/// `"fully processed and saved"` marker that only a successful merge emits.
/// Returning `Ok(())` after a swallowed failure made a failed contest
/// indistinguishable from a completed one (2026-11-06 c4-1052: the final
/// finding-merge chunk hit the output ceiling, the process logged
/// "Learning complete." and exited 0 with the KG unchanged).
///
/// `lost` carries one entry per LOST PROJECT, keyed by project id, so the
/// counts are derived from project identity and never from the number of
/// diagnostic strings. Counting strings was a real defect: two worker panics
/// plus one aggregate entry reported "2 of 5 succeeded" when zero projects
/// had been written (gate_20261107-013907).
fn pipeline_outcome(total: usize, lost: Vec<(String, String)>) -> Result<()> {
    if lost.is_empty() {
        return Ok(());
    }
    let succeeded = total.saturating_sub(lost.len());
    let detail = lost
        .iter()
        .map(|(id, why)| format!("{id}: {why}"))
        .collect::<Vec<_>>()
        .join("; ");
    Err(color_eyre::eyre::eyre!(
        "{} of {} project(s) failed; {} succeeded. Failures: {}",
        lost.len(),
        total,
        succeeded,
        detail
    ))
}

/// Shared pipeline: categorize+extract concurrently, then merge+write each
/// project serially as soon as extraction finishes.
async fn run_pipeline(
    db: &HistoricalDatabase,
    llm: &llmy::client::client::LLM,
    all_projects: Vec<ProjectData>,
    concurrency: usize,
    link: bool,
    agent_options: knowdit_kg::agent_runner::AgentRunOptions,
    merge_chunking: knowdit_kg::agents::MergeChunkingOptions,
    link_options: FindingLinkOptions,
) -> Result<()> {
    if all_projects.is_empty() {
        tracing::warn!("No projects to process.");
        return Ok(());
    }

    // Per-project failures are COLLECTED, not swallowed: a run that lost a
    // project must not report success. The caller keys on the exit code (and
    // on the "fully processed and saved" marker), so returning Ok(()) here
    // made a failed contest indistinguishable from a completed one
    // (2026-11-06 c4-1052: merge/write failed, process exited 0).
    //
    // Keyed by PROJECT ID, not by diagnostic string: the counts in the final
    // report must reflect projects, and a diagnostic entry (a worker panic)
    // must never be mistaken for a project (gate_20261107-013907).
    //
    // Declared before the completion filter so a project lost to a DB read
    // error is counted too - it is dropped from `pending`, and without this
    // it would vanish from the run with no trace in the outcome.
    let mut lost: Vec<(String, String)> = Vec::new();

    // Filter out already-completed projects
    let mut pending = Vec::new();
    for p in all_projects {
        match p.is_completed(db).await {
            Ok(true) => {
                tracing::info!("Project {} already completed, skipping", p.display_id());
            }
            Ok(false) => pending.push(p),
            Err(e) => {
                tracing::error!("Error checking project {}: {}", p.display_id(), e);
                lost.push((p.display_id(), format!("is_completed: {e}")));
            }
        }
    }

    // Projects we intended to learn: the ones still pending plus the ones
    // already lost above. Already-completed projects are not part of the run.
    let total = pending.len() + lost.len();
    // Captured before the branch consumes `pending`: the accounting invariant
    // compares these ids against what actually completed or was named lost.
    let pending_ids: Vec<String> = pending.iter().map(|p| p.display_id()).collect();

    if pending.is_empty() {
        if lost.is_empty() {
            tracing::info!("All projects already completed.");
            if link {
                link_options.link_pending_findings(db, llm).await?;
            }
            return Ok(());
        }
        // Every project was lost before the pipeline could start - fail closed
        // rather than reporting a clean run.
        pipeline_outcome(total, lost)?;
        return Ok(());
    }

    tracing::info!(
        "Will process {} projects (concurrency={}): {:?}",
        pending.len(),
        concurrency,
        pending.iter().map(|p| p.display_id()).collect::<Vec<_>>()
    );

    // Checkpoint enable: KNOWDIT_CHECKPOINT_DIR=<dir> activates resumable
    // extraction; unset = no checkpointing (byte-identical behavior).
    // Sequential path clears per-project after merge; concurrent workers
    // create per-project sinks (clear-after-merge is a follow-up for
    // the concurrent path — the tranche_runner uses sequential children).
    let checkpoint_dir = std::env::var("KNOWDIT_CHECKPOINT_DIR").ok();
    if checkpoint_dir.is_some() && concurrency <= 1 {
        tracing::info!(
            "Checkpointing ACTIVE (dir={}, sequential path)",
            checkpoint_dir.as_deref().unwrap_or("?"),
        );
    } else if checkpoint_dir.is_some() {
        tracing::warn!(
            "Checkpointing INACTIVE: KNOWDIT_CHECKPOINT_DIR is set but concurrency={} (concurrent path does not yet checkpoint; use --concurrency 1 for resumable extraction)",
            concurrency,
        );
    }

    // `failures` and `total` are declared above the completion filter so a
    // project lost to a DB read error is counted as well.

    if concurrency <= 1 {
        for project in pending {
            let sink = checkpoint_dir
                .as_ref()
                .map(|dir| knowdit_kg::checkpoint::CheckpointSink::new(dir, project.display_id()));
            match project
                .categorize_and_extract(llm, &agent_options, None, sink.as_ref())
                .await
            {
                Ok(extract) => {
                    match project
                        .merge_and_write(db, llm, &extract, &agent_options, merge_chunking)
                        .await
                    {
                        Ok(_) => {
                            // merge succeeded: clear this project's
                            // checkpoint (the data is now in the KG)
                            if let Some(sink) = &sink {
                                if let Err(e) = sink.clear() {
                                    tracing::warn!(
                                        "Checkpoint clear failed for {}: {}",
                                        project.display_id(),
                                        e,
                                    );
                                }
                            }
                        }
                        Err(e) => {
                            tracing::error!(
                                "Failed to merge/write project {}: {}",
                                project.display_id(),
                                e,
                            );
                            lost.push((project.display_id(), format!("merge/write: {e}")));
                        }
                    }
                }
                Err(e) => {
                    tracing::error!(
                        "Skipping merge for project {} (extract failed: {})",
                        project.display_id(),
                        e,
                    );
                    lost.push((project.display_id(), format!("extract: {e}")));
                }
            }
        }
    } else {
        let (tx, rx) = async_channel::bounded::<ProjectData>(pending.len() + 1);
        // The Result travels WITH its project so a failed extraction can be
        // attributed to the project that caused it (and counted in
        // `failures`) instead of being dropped by an unjoined JoinSet.
        // Spelled out because this file's bare `Result` is color_eyre's,
        // while `categorize_and_extract` returns `knowdit_kg::Result`.
        let (out_tx, mut out_rx) = mpsc::channel::<(
            ProjectData,
            std::result::Result<ExtractResult, KgError>,
        )>(concurrency + 1);

        let mut handles = JoinSet::new();
        for _ in 0..concurrency {
            let rx = rx.clone();
            let out = out_tx.clone();
            let llm = llm.clone();
            let extract_opts = agent_options.clone();
            handles.spawn(async move {
                while let Ok(project) = rx.recv().await {
                    // An extraction failure is REPORTED, not returned: the
                    // JoinSet is never joined, so `?` here dropped the error
                    // on the floor and the project vanished from the run
                    // (gate_20261107-004223 finding 1). The worker also keeps
                    // draining the queue so one bad project cannot strand the
                    // rest.
                    let res = project
                        .categorize_and_extract(&llm, &extract_opts, None, None)
                        .await;
                    if out.send((project, res)).await.is_err() {
                        break;
                    }
                }
                Ok::<_, KgError>(())
            });
        }
        drop(out_tx);
        drop(rx);

        for project in pending {
            if tx.send(project).await.is_err() {
                break;
            }
        }

        drop(tx);
        let mut completed_ids: Vec<String> = Vec::new();
        while let Some((project, extract_res)) = out_rx.recv().await {
            match extract_res {
                Ok(extract_res) => {
                    // A merge failure is RECORDED and the loop continues: a
                    // `?` here abandoned every already-extracted project still
                    // buffered in out_rx, discarding paid work and skipping
                    // the aggregated report (gate_20261107-005525). The
                    // sequential branch already behaves this way.
                    match project
                        .merge_and_write(db, llm, &extract_res, &agent_options, merge_chunking)
                        .await
                    {
                        Ok(_) => completed_ids.push(project.display_id()),
                        Err(e) => {
                            tracing::error!(
                                "Failed to merge/write project {}: {}",
                                project.display_id(),
                                e,
                            );
                            lost.push((project.display_id(), format!("merge/write: {e}")));
                        }
                    }
                }
                Err(e) => {
                    tracing::error!(
                        "Skipping merge for project {} (extract failed: {})",
                        project.display_id(),
                        e,
                    );
                    lost.push((project.display_id(), format!("extract: {e}")));
                }
            }
        }

        // A worker that PANICKED never sends its project and never returns an
        // Err, so the receive loop above simply ends early and the project
        // vanishes (gate_20261107-010622). Joining the set is what makes the
        // panic observable at all - the JoinSet was previously spawned into
        // and never joined. The panic is a DIAGNOSTIC, not a project: the
        // projects it stranded are named individually by the accounting
        // invariant below, so this entry must not be counted as one.
        while let Some(joined) = handles.join_next().await {
            if let Err(e) = joined {
                tracing::error!("extraction worker terminated abnormally: {}", e);
            }
        }

        // Accounting invariant: every project we intended to learn is either
        // completed or named in `lost`. Anything else is a loss path nobody
        // has thought of yet, and it must not pass as success.
        //
        // Derived from PROJECT IDENTITY, never from a count of diagnostic
        // strings: two worker panics plus one aggregate entry used to report
        // "2 of 5 succeeded" when zero projects had been written
        // (gate_20261107-013907).
        let accounted: std::collections::HashSet<&str> = lost
            .iter()
            .map(|(id, _)| id.as_str())
            .chain(completed_ids.iter().map(|id| id.as_str()))
            .collect();
        // Collected into an owned Vec first: pushing to `lost` while the set
        // still borrows it is a borrow error.
        let unaccounted: Vec<String> = pending_ids
            .iter()
            .filter(|id| !accounted.contains(id.as_str()))
            .cloned()
            .collect();
        drop(accounted);
        for id in unaccounted {
            lost.push((
                id,
                "unaccounted for - no completion and no recorded failure".to_string(),
            ));
        }
    }

    if link {
        link_options.link_pending_findings(db, llm).await?;
    }

    // Checked AFTER the link stage so the projects that succeeded still get
    // their links; the run then fails closed instead of printing the
    // completion marker over a lost project.
    pipeline_outcome(total, lost)?;

    tracing::info!("Learning complete.");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::pipeline_outcome;

    fn lost(id: &str, why: &str) -> (String, String) {
        (id.to_string(), why.to_string())
    }

    /// The 2026-11-06 c4-1052 regression: one project's merge/write failed,
    /// the pipeline swallowed it, printed "Learning complete." and exited 0.
    /// A lost project must surface as an Err.
    #[test]
    fn a_failed_project_fails_the_run() {
        let err = pipeline_outcome(
            1,
            vec![lost("c4-1052", "merge/write: reach output length limit")],
        )
        .expect_err("a run that lost its only project must not report success");
        let msg = err.to_string();
        assert!(
            msg.contains("1 of 1 project(s) failed"),
            "error must state the loss count, got: {msg}"
        );
        assert!(
            msg.contains("c4-1052"),
            "error must name the lost project, got: {msg}"
        );
        assert!(
            msg.contains("reach output length limit"),
            "error must carry the underlying cause, got: {msg}"
        );
    }

    /// A clean run stays a clean run - the fix must not turn success into
    /// failure.
    #[test]
    fn a_clean_run_succeeds() {
        assert!(pipeline_outcome(3, Vec::new()).is_ok());
    }

    /// Partial loss is still a failure, and the message distinguishes the
    /// survivors from the losses.
    #[test]
    fn partial_loss_reports_both_counts() {
        let err = pipeline_outcome(
            3,
            vec![
                lost("c4-1", "extract: boom"),
                lost("c4-2", "merge/write: bang"),
            ],
        )
        .expect_err("2 of 3 lost must fail");
        let msg = err.to_string();
        assert!(msg.contains("2 of 3 project(s) failed"), "got: {msg}");
        assert!(msg.contains("1 succeeded"), "got: {msg}");
        assert!(msg.contains("c4-1") && msg.contains("c4-2"), "got: {msg}");
    }

    /// Every failure is named, not just the first - an operator needs the
    /// whole list to know what to re-run.
    #[test]
    fn every_failure_is_named() {
        let lost_projects: Vec<(String, String)> = (0..5)
            .map(|i| lost(&format!("c4-{i}"), &format!("merge/write: e{i}")))
            .collect();
        let msg = pipeline_outcome(5, lost_projects).unwrap_err().to_string();
        for i in 0..5 {
            assert!(msg.contains(&format!("c4-{i}")), "missing c4-{i} in: {msg}");
        }
    }

    /// gate_20261107-004223 finding 2: a project whose `is_completed` check
    /// errored is dropped from `pending`. It must still be counted, or the
    /// run reports success having never learned it.
    #[test]
    fn a_project_lost_to_a_completion_check_error_fails_the_run() {
        let err = pipeline_outcome(2, vec![lost("c4-9", "is_completed: database is locked")])
            .expect_err("a project lost before the pipeline must fail the run");
        let msg = err.to_string();
        assert!(msg.contains("1 of 2 project(s) failed"), "got: {msg}");
        assert!(msg.contains("c4-9"), "got: {msg}");
        assert!(msg.contains("database is locked"), "got: {msg}");
    }

    /// gate_20261107-004223 finding 1: an extraction failure in the
    /// concurrent path used to be dropped by an unjoined JoinSet. It now
    /// reaches `lost` with the project's identity attached.
    #[test]
    fn a_concurrent_extraction_failure_fails_the_run() {
        let err = pipeline_outcome(3, vec![lost("c4-77", "extract: reach output length limit")])
            .expect_err("a concurrent extraction loss must fail the run");
        let msg = err.to_string();
        assert!(msg.contains("1 of 3 project(s) failed"), "got: {msg}");
        assert!(msg.contains("c4-77"), "got: {msg}");
        assert!(msg.contains("reach output length limit"), "got: {msg}");
    }

    /// gate_20261107-005525: the concurrent branch used to `?` on the first
    /// merge failure, abandoning every already-extracted project still
    /// buffered. Multiple losses must all be reported in ONE aggregated
    /// error, not just the first.
    #[test]
    fn a_concurrent_merge_failure_is_aggregated_not_short_circuited() {
        let err = pipeline_outcome(
            4,
            vec![
                lost("c4-10", "merge/write: boom"),
                lost("c4-11", "merge/write: bang"),
                lost("c4-12", "extract: crash"),
            ],
        )
        .expect_err("3 of 4 lost must fail");
        let msg = err.to_string();
        assert!(msg.contains("3 of 4 project(s) failed"), "got: {msg}");
        assert!(msg.contains("1 succeeded"), "got: {msg}");
        for id in ["c4-10", "c4-11", "c4-12"] {
            assert!(msg.contains(id), "missing {id} in: {msg}");
        }
    }

    /// gate_20261107-010622: a worker panic is invisible to the receive loop,
    /// so the project simply vanishes. The accounting invariant names each
    /// stranded project individually.
    #[test]
    fn an_unaccounted_project_fails_the_run() {
        let err = pipeline_outcome(
            3,
            vec![
                lost("c4-20", "merge/write: boom"),
                lost(
                    "c4-21",
                    "unaccounted for - no completion and no recorded failure",
                ),
            ],
        )
        .expect_err("an unaccounted project must fail the run");
        let msg = err.to_string();
        assert!(msg.contains("2 of 3 project(s) failed"), "got: {msg}");
        assert!(msg.contains("c4-21"), "got: {msg}");
        assert!(msg.contains("unaccounted"), "got: {msg}");
    }

    /// gate_20261107-013907: the counts must come from PROJECT IDENTITY, not
    /// from the number of diagnostic strings. Two worker panics plus one
    /// aggregate entry used to report "2 of 5 succeeded" when zero projects
    /// had been written. With one entry per lost project the arithmetic is
    /// exact.
    #[test]
    fn counts_come_from_projects_not_diagnostic_strings() {
        // 5 intended, 0 completed, 5 individually named losses.
        let lost_projects: Vec<(String, String)> = (0..5)
            .map(|i| lost(&format!("c4-{i}"), "unaccounted for"))
            .collect();
        let msg = pipeline_outcome(5, lost_projects).unwrap_err().to_string();
        assert!(
            msg.contains("5 of 5 project(s) failed; 0 succeeded"),
            "zero projects were written, so zero may be reported as succeeded: {msg}"
        );
    }
}
