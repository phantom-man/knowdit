use crate::agent_runner::AgentRunOptions;
use crate::agents::{
    AggregatedFindingMergeDecision, AggregatedSemanticMergeDecision, CategorizeRunner,
    FindingChunkExtractor, FindingMerger, MergeChunkingOptions, SemanticChunkExtractor,
    SemanticMerger,
};
use crate::category::DeFiCategory;
use crate::db::HistoricalDatabase;
use crate::error::{KgError, Result};
pub use crate::link::{FindingLinkOptions, PendingFindingForLinking, PersistedFindingLinkResult};
use crate::project_loader::ProjectData;
use crate::prompts;
use crate::vulnerability::{VulnerabilityCategory, resolve_taxonomy_entry};
use itertools::Itertools;
pub use knowdit_kg_model::{ExtractedFinding, ExtractedFunction, ExtractedSemantic};
use llmy::client::client::LLM;
use llmy::client::context::TokenCursor;
use knowdit_repo_model::cg::CallGraph;
use knowdit_cg_static::shard_pack;
use llmy::client::model::OpenAIModel;
use serde::Deserialize;
use std::collections::{HashMap, HashSet};
use std::time::Instant;

// ── LLM response types ──────────────────────────────────────────────
//
// The categorize / extract semantic / extract finding / semantic merge /
// finding merge phases all use llmy `Agent` + tool-calls now; their tool
// arguments + tool implementations live in [`crate::agents`]. Only the
// in-project linking phase still uses the JSON-mode prompt+parse pattern,
// because it's a single LLM call producing a small, simply-shaped index map.

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct InProjectLinkEntry {
    finding_index: usize,
    semantic_indices: Vec<usize>,
    #[serde(default)]
    reasoning: String,
}

#[derive(Debug, Deserialize)]
struct InProjectLinkResponse {
    links: Vec<InProjectLinkEntry>,
}

fn render_semantics_for_in_project_link(semantics: &[ExtractedSemantic]) -> String {
    semantics
        .iter()
        .enumerate()
        .map(|(idx, s)| {
            format!(
                "S{idx}\nname: {name}\ncategory: {category}\ndefinition: {definition}\ndescription: {description}\n\n",
                idx = idx,
                name = s.name,
                category = s.category.as_str(),
                definition = s.definition.trim(),
                description = s.description.trim()
            )
        })
        .collect::<String>()
}

fn render_findings_for_in_project_link(findings: &[ExtractedFinding]) -> String {
    findings
        .iter()
        .enumerate()
        .map(|(idx, f)| {
            format!(
                "F{idx}\ntitle: {title}\nseverity: {severity}\ncategory: {category}\nsubcategory: {subcategory}\nroot_cause: {root_cause}\ndescription: {description}\npatterns: {patterns}\n\n",
                idx = idx,
                title = f.title,
                severity = f.severity,
                category = f.category,
                subcategory = f.subcategory,
                root_cause = f.root_cause.trim(),
                description = f.description.trim(),
                patterns = f.patterns.trim(),
            )
        })
        .collect::<String>()
}

// ── Public merge types (used by HistoricalDatabase) ────────────────────────

#[derive(Debug, Clone)]
pub enum MergeAction {
    /// Admit the new raw as a fresh canonical (no existing matches).
    New,
    /// Fold the new raw into one *or more* existing canonicals. Each
    /// `target_id` becomes one row in `semantic_merge`. The canonical's
    /// `name` and `definition` are stable identity and are NEVER touched
    /// by a merge. `updated_description`, when present, REPLACES each
    /// merged-into canonical's description with a generalization; `None`
    /// keeps the existing description.
    Merge {
        target_ids: Vec<i32>,
        updated_description: Option<String>,
        /// One-or-two-sentence note (how this raw extends the canonical)
        /// written to every `semantic_merge` edge this fold creates.
        appended_description: Option<String>,
    },
}

#[derive(Debug, Clone)]
pub struct MergeResult {
    pub semantic: ExtractedSemantic,
    pub action: MergeAction,
}

#[derive(Debug, Clone)]
pub enum FindingMergeAction {
    /// Admit the new raw finding as a fresh canonical.
    New,
    /// Fold the new raw into one *or more* existing canonical findings.
    /// Canonical's `title`, `severity`, and `root_cause` are stable
    /// identity and are NEVER touched. The `updated_*` fields, when present,
    /// REPLACE the canonical's description / patterns / exploits with
    /// generalizations at write time; `None` keeps the existing value.
    Merge {
        target_ids: Vec<i32>,
        updated_description: Option<String>,
        updated_patterns: Option<String>,
        updated_exploits: Option<String>,
        /// One-or-two-sentence notes (how this raw extends the canonical's
        /// description / patterns / exploits) written to every `finding_merge`
        /// edge this fold creates.
        appended_description: Option<String>,
        appended_patterns: Option<String>,
        appended_exploits: Option<String>,
    },
}

#[derive(Debug, Clone)]
pub struct FindingMergeResult {
    pub finding: ExtractedFinding,
    pub action: FindingMergeAction,
}

// ── ProjectData learning pipeline ───────────────────────────────────

/// Index-based links produced by the in-project linking step. Each
/// `(finding_index, semantic_index)` pair refers to positions in the
/// parent `ExtractResult.findings` and `ExtractResult.semantics` arrays.
/// The atomic admission step translates these positional indices into
/// concrete row ids for `semantic_finding_link` after all raw inserts.
#[derive(Debug, Clone, Default)]
pub struct InProjectLinks {
    pub edges: Vec<(usize, usize)>,
}

impl InProjectLinks {
    pub fn is_empty(&self) -> bool {
        self.edges.is_empty()
    }
}

/// Intermediate result from the categorize + extract + in-project link phase.
/// Can be computed concurrently across projects (no DB writes happen here).
pub struct ExtractResult {
    pub categories: Vec<DeFiCategory>,
    pub semantics: Vec<ExtractedSemantic>,
    pub findings: Vec<ExtractedFinding>,
    /// In-project semantic↔finding links: every finding has ≥1 entry
    /// (LLM-enforced). Indices are positional in `semantics` / `findings`.
    pub in_project_links: InProjectLinks,
}

impl ProjectData {
    /// Phase 1: Categorize the project and extract semantics.
    /// Safe to run concurrently across multiple projects.
    pub async fn categorize_and_extract(
        &self,
        llm: &LLM,
        agent_options: &AgentRunOptions,
        chunk_input_budget: Option<usize>,
        sink: Option<&crate::checkpoint::CheckpointSink>,
    ) -> Result<ExtractResult> {
        let pid = self.display_id();

        tracing::info!(
            "Processing project {}: {} ({} source files)",
            pid,
            self.name(),
            self.source_files().len()
        );

        if self.source_files().is_empty() {
            tracing::warn!("No source files found for project {}", pid);
            return Ok(ExtractResult {
                categories: vec![],
                semantics: vec![],
                findings: vec![],
                in_project_links: InProjectLinks::default(),
            });
        }

        let categories = self.categorize(llm, agent_options).await?;
        tracing::info!("Project {} categorized as: {:?}", pid, categories);

        let (all_semantics, all_findings) = tokio::try_join!(
            self.extract_semantics(llm, &categories, agent_options, chunk_input_budget, sink),
            self.extract_findings(llm, &categories, agent_options, chunk_input_budget, sink)
        )?;

        tracing::info!(
            "Extracted {} raw semantics from project {}",
            all_semantics.len(),
            pid
        );

        tracing::info!(
            "Extracted {} raw findings from project {}",
            all_findings.len(),
            pid
        );

        let deduped = Self::dedup_semantics(all_semantics);
        let deduped_findings = Self::dedup_findings(all_findings);
        tracing::info!(
            "After intra-project dedup: {} semantics for project {}",
            deduped.len(),
            pid
        );

        tracing::info!(
            "After intra-project dedup: {} findings for project {}",
            deduped_findings.len(),
            pid
        );

        // Run the in-project linking step before any cross-project merge.
        // Every finding must claim ≥1 same-project semantic; this is what
        // grounds raw findings to raw semantics inside the atomic admission
        // transaction even when both later get merged into canonicals.
        let in_project_links = if deduped_findings.is_empty() || deduped.is_empty() {
            InProjectLinks::default()
        } else {
            self.link_findings_in_project(llm, &categories, &deduped, &deduped_findings)
                .await?
        };
        tracing::info!(
            "In-project linking for project {}: {} edge(s) over {} finding(s)",
            pid,
            in_project_links.edges.len(),
            deduped_findings.len()
        );

        Ok(ExtractResult {
            categories,
            semantics: deduped,
            findings: deduped_findings,
            in_project_links,
        })
    }

    /// Categorize the project and extract only project semantics.
    /// This is used by consumers that need semantic context but do not need audit findings.
    pub async fn categorize_and_extract_semantics(
        &self,
        llm: &LLM,
        agent_options: &AgentRunOptions,
        chunk_input_budget: Option<usize>,
        sink: Option<&crate::checkpoint::CheckpointSink>,
    ) -> Result<ExtractResult> {
        let pid = self.display_id();

        tracing::info!(
            "Processing project {} for semantics only: {} ({} source files)",
            pid,
            self.name(),
            self.source_files().len()
        );

        if self.source_files().is_empty() {
            tracing::warn!("No source files found for project {}", pid);
            return Ok(ExtractResult {
                categories: vec![],
                semantics: vec![],
                findings: vec![],
                in_project_links: InProjectLinks::default(),
            });
        }

        let categories = self.categorize(llm, agent_options).await?;
        tracing::info!("Project {} categorized as: {:?}", pid, categories);

        let all_semantics = self
            .extract_semantics(llm, &categories, agent_options, chunk_input_budget, sink)
            .await?;
        tracing::info!(
            "Extracted {} raw semantics from project {}",
            all_semantics.len(),
            pid
        );

        let deduped = Self::dedup_semantics(all_semantics);
        tracing::info!(
            "After intra-project dedup: {} semantics for project {}",
            deduped.len(),
            pid
        );

        Ok(ExtractResult {
            categories,
            semantics: deduped,
            findings: Vec::new(),
            in_project_links: InProjectLinks::default(),
        })
    }

    /// Graph-aware variant: cycle-coherent shards + cycle-signature anchors + emit gates.
    /// Uses the stored static call graph so mechanisms are not split across shards.
    pub async fn categorize_and_extract_semantics_with_graph(
        &self,
        llm: &LLM,
        agent_options: &AgentRunOptions,
        chunk_input_budget: Option<usize>,
        sink: Option<&crate::checkpoint::CheckpointSink>,
        call_graph: &CallGraph,
    ) -> Result<ExtractResult> {
        if self.source_files().is_empty() {
            return Ok(ExtractResult { categories: vec![], semantics: vec![], findings: vec![], in_project_links: InProjectLinks::default() });
        }
        let categories = self.categorize(llm, agent_options).await?;
        let all_semantics = self
            .extract_semantics_with_graph(llm, &categories, agent_options, chunk_input_budget, sink, call_graph)
            .await?;
        let deduped = Self::dedup_semantics(all_semantics);
        Ok(ExtractResult { categories, semantics: deduped, findings: Vec::new(), in_project_links: InProjectLinks::default() })
    }

    /// Phase 2: Merge extracted semantics with existing KB and write to DB.
    /// MUST be run serially (one project at a time) to avoid merge conflicts.
    /// Commit one project's full merge output to the historical KG.
    ///
    /// Begins its own transaction via
    /// [`HistoricalDatabase::write_project_completed`] under the
    /// hood and discards the new-canonical id list. Use this when
    /// the caller doesn't need to compose additional writes
    /// (`pending_semantic` enqueue etc.) inside the same
    /// transaction. Bulk learn paths (`learn moves` / `learn c4`
    /// / `learn projects`) go through here because they never run
    /// retro-link.
    pub async fn merge_and_write(
        &self,
        db: &HistoricalDatabase,
        llm: &LLM,
        extract: &ExtractResult,
        agent_options: &AgentRunOptions,
        merge_chunking: MergeChunkingOptions,
    ) -> Result<()> {
        let txn = db.begin().await?;
        self.merge_and_write_txn(&txn, db, llm, extract, agent_options, merge_chunking)
            .await?;
        txn.commit().await?;
        Ok(())
    }

    /// Transaction-scoped variant of [`Self::merge_and_write`].
    /// Performs the merge-LLM passes (against the LIVE DB — these
    /// are reads, not writes, so don't depend on the txn) and then
    /// writes the resulting rows through
    /// [`HistoricalDatabase::write_project_completed_txn`] using
    /// the supplied transaction. Returns the canonical semantic
    /// ids this project newly introduced, so the caller can chain
    /// [`HistoricalDatabase::enqueue_pending_canonical_semantics_txn`]
    /// in the same transaction when needed (incremental
    /// `workflow learn` flow).
    pub async fn merge_and_write_txn(
        &self,
        conn: &sea_orm::DatabaseTransaction,
        db: &HistoricalDatabase,
        llm: &LLM,
        extract: &ExtractResult,
        agent_options: &AgentRunOptions,
        merge_chunking: MergeChunkingOptions,
    ) -> Result<Vec<i32>> {
        let pid = self.display_id();

        if extract.semantics.is_empty() && extract.findings.is_empty() {
            let new_canonicals = db
                .write_project_completed_txn(
                    conn,
                    self.name(),
                    self.platform_id(),
                    &extract.categories,
                    &[],
                    &[],
                    &InProjectLinks::default(),
                )
                .await?;
            tracing::info!("Project {} written (no semantics or findings)", pid);
            return Ok(new_canonicals);
        }

        let semantic_merge_results = self
            .merge_with_existing(db, llm, extract, agent_options, merge_chunking)
            .await?;
        let finding_merge_results = self
            .merge_findings_with_existing(db, llm, extract, agent_options, merge_chunking)
            .await?;

        let new_canonicals = db
            .write_project_completed_txn(
                conn,
                self.name(),
                self.platform_id(),
                &extract.categories,
                &semantic_merge_results,
                &finding_merge_results,
                &extract.in_project_links,
            )
            .await?;

        tracing::info!("Project {} fully processed and saved", pid);
        Ok(new_canonicals)
    }

    /// Check if this project is already completed in the DB.
    pub async fn is_completed(&self, db: &HistoricalDatabase) -> Result<bool> {
        if let Some(pid) = self.platform_id() {
            db.is_project_completed(pid).await
        } else {
            Ok(db
                .get_project_by_name(self.name())
                .await?
                .map(|p| p.status == "completed")
                .unwrap_or(false))
        }
    }

    fn build_project_prompt_body(&self) -> String {
        let mut content = prompts::project_user_prefix();
        content.push_str("## Source Files\n\n");

        // README FIRST (2026-11-13): the categorize cap bounds this body to
        // N tokens, and the README is the most explicitly descriptive
        // categorization signal. Appending it after every source file meant
        // any project above the cap lost it entirely, so the label came from
        // only the leading slice of the file list - and that label then
        // propagates into every extraction prompt via
        // `extract_semantics_user_suffix(categories)`. Front-loading it keeps
        // the best signal inside any budget and is strictly better even at
        // the full window.
        if let Some(readme) = self.load_readme() {
            content.push_str("## README\n\n");
            content.push_str(&readme);
            content.push_str("\n\n");
        }

        let lang = self.source_language().code_fence().to_string();

        for file in self.source_files() {
            let path = &file.relative_path;

            // Pre-filter: skip test/mock/script files entirely and strip
            // boilerplate lines from remaining files — reduces extraction
            // LLM input tokens 30-40% without removing audit-relevant code
            if crate::source_filter::should_skip_file(path) {
                content.push_str(&crate::source_filter::skip_marker(path));
                continue;
            }

            let filtered =
                crate::source_filter::filter_content(&file.content, &lang);
            content.push_str(&format!(
                "### {}\n```{}\n{}\n```\n\n",
                path.display(),
                lang,
                filtered
            ));
        }

        content
    }

    fn build_report_prompt_body(&self) -> Option<String> {
        let report = self.audit_report()?.render();
        let mut content = prompts::report_user_prefix();
        content.push_str(&report);
        content.push_str("\n\n");
        Some(content)
    }

    fn load_readme(&self) -> Option<String> {
        for name in &["README.md", "readme.md", "Readme.md"] {
            let readme_path = self.root_dir().join(name);
            if readme_path.exists()
                && let Ok(readme) = std::fs::read_to_string(&readme_path)
            {
                return Some(readme);
            }
        }

        None
    }

    fn id_slug(&self) -> String {
        sanitize_prompt_prefix(&self.display_id())
    }

    fn debug_key(&self, stage: &str) -> String {
        format!("{}-{}", sanitize_prompt_prefix(stage), self.id_slug())
    }

    // ── Private pipeline steps ──────────────────────────────────────

    /// Categorize the project via an `Agent` with two tools
    /// (`set_project_categories` + `finalize_categorization`). Fills the
    /// context window with the README and as many source files as fit.
    async fn categorize(
        &self,
        llm: &LLM,
        agent_options: &AgentRunOptions,
    ) -> Result<Vec<DeFiCategory>> {
        let started_at = Instant::now();
        let model = &llm.model;
        let user_suffix = prompts::CATEGORIZE_USER_SUFFIX;
        let sys_tokens = model
            .config
            .count_tokens_lossy(prompts::GENERAL_ROLE_SYSTEM);
        let suffix_tokens = model.config.count_tokens_lossy(user_suffix);
        let window_budget = get_context_budget(model, agent_options.context_window_utilization)
            .saturating_sub(sys_tokens + suffix_tokens);
        // Categorize picks ONE label from a fixed enum, so the whole context
        // window is waste: measured median 27,894 tokens across the r2 shards.
        // A bounded sample is enough. `None` keeps the old window budget.
        let budget = match agent_options.categorize_token_budget {
            Some(cap) => window_budget.min(cap),
            None => window_budget,
        };
        let content = self.build_project_prompt_body();
        tracing::info!(
            "categorize preparing {}: source_files={}, body_chars={}, budget={}",
            self.display_id(),
            self.source_files().len(),
            content.len(),
            budget,
        );

        let Some(mut cursor) = TokenCursor::new(content, model.clone()) else {
            return Err(KgError::other(
                "Failed to initialize TokenCursor for categorization",
            ));
        };
        let user_prompt = format!("{}{}", cursor.next_chunk(budget).unwrap_or(""), user_suffix,);
        tracing::info!(
            "Categorization prompt: ~{} tokens (budget: {})",
            sys_tokens + model.config.count_tokens_lossy(&user_prompt),
            sys_tokens + budget,
        );

        let label = format!("categorize-{}", self.display_id());
        let local_options = agent_options.scoped(&self.debug_key("categorize"));
        let runner = CategorizeRunner {
            llm: llm.clone(),
            options: local_options,
            system_prompt: prompts::GENERAL_ROLE_SYSTEM.to_string(),
            user_prompt,
            label,
        };
        let record = runner.run().await?;
        tracing::info!(
            "categorize finished for {} in {:?}: categories={:?} ({})",
            self.display_id(),
            started_at.elapsed(),
            record.categories,
            record.reasoning,
        );
        Ok(record.categories)
    }

    /// Extract semantics from the project's source files. Splits the source
    /// text into chunks that fit the context window and runs one Agent per
    /// chunk; each chunk's semantics are emitted via `emit_semantic` tool
    /// calls and the chunk is closed with `finalize_semantic_extraction`.
    pub async fn extract_semantics(
        &self,
        llm: &LLM,
        categories: &[DeFiCategory],
        agent_options: &AgentRunOptions,
        chunk_input_budget: Option<usize>,
        sink: Option<&crate::checkpoint::CheckpointSink>,
    ) -> Result<Vec<ExtractedSemantic>> {
        self.extract_semantics_inner(llm, categories, agent_options, chunk_input_budget, sink, None).await
    }

    /// Graph-aware variant: co-locates call-graph cycles in one shard (H1 proxy) and
    /// enables the emit_semantic anchoring gate. See docs/SEMANTIC_EXTRACTION_REVIEW_AND_MATH.md.
    pub async fn extract_semantics_with_graph(
        &self,
        llm: &LLM,
        categories: &[DeFiCategory],
        agent_options: &AgentRunOptions,
        chunk_input_budget: Option<usize>,
        sink: Option<&crate::checkpoint::CheckpointSink>,
        call_graph: &CallGraph,
    ) -> Result<Vec<ExtractedSemantic>> {
        self.extract_semantics_inner(llm, categories, agent_options, chunk_input_budget, sink, Some(call_graph)).await
    }

    /// Extract over MATERIALIZED shard files (sequential + manifest.json), resuming by
    /// stable shard index + content hash. Shards must be produced first via
    /// `knowdit static solidity repack-shards --out-dir <dir>` (whole functions, cycles
    /// never split). Pass the same `CallGraph` used to pack the shards to enable the
    /// emit_semantic anchoring/hyperbolic gates; agents continue where they left off.
    pub async fn extract_semantics_from_shards(
        &self,
        shard_dir: &std::path::Path,
        llm: &LLM,
        categories: &[DeFiCategory],
        agent_options: &AgentRunOptions,
        sink: Option<&crate::checkpoint::CheckpointSink>,
        call_graph: Option<&CallGraph>,
    ) -> Result<Vec<ExtractedSemantic>> {
        let manifest: Vec<serde_json::Value> =
            serde_json::from_str(&std::fs::read_to_string(shard_dir.join("manifest.json"))?)?;
        let system_prompt = prompts::GENERAL_ROLE_SYSTEM;
        let user_suffix = prompts::extract_semantics_user_suffix(categories);
        let checkpointed: std::collections::HashMap<u32, (String, Vec<u8>)> = match sink {
            Some(s) => s.load_stage(crate::checkpoint::stage::SEMANTIC_EXTRACT).unwrap_or_default(),
            None => std::collections::HashMap::new(),
        };
        let mut all = Vec::new();
        for entry in manifest {
            let idx = entry["index"].as_u64().unwrap_or(0) as u32;
            let file = entry["file"].as_str().unwrap_or("");
            let content = std::fs::read_to_string(shard_dir.join(file))?;
            let hash = crate::checkpoint::chunk_hash(&content);
            if let Some((stored_hash, payload)) = checkpointed.get(&idx) {
                if *stored_hash == hash {
                    if let Ok(stored) = serde_json::from_slice::<Vec<ExtractedSemantic>>(payload) {
                        tracing::info!("Shard {} resumed from checkpoint ({} semantic(s))", idx, stored.len());
                        all.extend(stored);
                        continue;
                    }
                }
            }
            let mut user_prompt = format!("{}{}", content, user_suffix);
            if let Some(cg) = call_graph {
                let ids: Vec<i32> = entry["function_ids"].as_array().map(|a| a.iter().filter_map(|v| v.as_i64().map(|x| x as i32)).collect()).unwrap_or_default();
                user_prompt.push_str(&shard_pack::graph_features(cg, Some(&ids)));
            }
            let extractor = SemanticChunkExtractor {
                llm: llm.clone(),
                options: agent_options.scoped(&format!("shard-{}", idx)),
                system_prompt: system_prompt.to_string(),
                user_prompt,
                label: format!("semantic-shard-{}", idx),
                call_graph: call_graph.map(|c| std::sync::Arc::new(c.clone())), 
                storage_deps: std::sync::Arc::new(std::collections::BTreeMap::new()),
                existing_nodes: std::sync::Arc::new(Vec::new()),
            };
            let sems = extractor.run().await?;
            if let Some(s) = sink {
                s.write_chunk(crate::checkpoint::stage::SEMANTIC_EXTRACT, idx, &hash, &serde_json::to_vec(&sems)?)?;
            }
            all.extend(sems);
        }
        Ok(all)
    }

    /// Reorder the project body's `### path` source sections so contracts belonging to the
    /// same call-graph shard group (cycle) are contiguous, keeping cycles inside one shard.
    fn graph_ordered_body(&self, cg: &CallGraph) -> String {
        let body = self.build_project_prompt_body();
        let mut order: Vec<String> = Vec::new();
        for g in shard_pack::shard_groups(cg) {
            for cid in &g {
                if let Some(c) = cg.contracts.get(cid) {
                    let p = c.relative_file_path.to_string_lossy().replace('\\', "/");
                    if !order.contains(&p) { order.push(p); }
                }
            }
        }
        let Some(idx) = body.find("### ") else { return body; };
        let header = body[..idx].to_string();
        let rest = &body[idx..];
        let mut sections: Vec<(String, String)> = Vec::new();
        for part in rest.split("### ").skip(1) {
            let path = part.lines().next().unwrap_or("").trim().to_string();
            sections.push((path, format!("### {}", part)));
        }
        let rank = |p: &String| -> usize {
            let n = p.replace('\\', "/");
            order.iter().position(|o| n.ends_with(o) || o.ends_with(&n)).unwrap_or(usize::MAX)
        };
        sections.sort_by_key(|(p, _)| rank(p));
        let mut out = header;
        for (_, s) in sections { out.push_str(&s); }
        out
    }

    async fn extract_semantics_inner(
        &self,
        llm: &LLM,
        categories: &[DeFiCategory],
        agent_options: &AgentRunOptions,
        chunk_input_budget: Option<usize>,
        sink: Option<&crate::checkpoint::CheckpointSink>,
        call_graph: Option<&CallGraph>,
    ) -> Result<Vec<ExtractedSemantic>> {
        let system_prompt = prompts::GENERAL_ROLE_SYSTEM;
        let model = &llm.model;
        let debug_key = self.debug_key("extract");
        let sys_tokens = model.config.count_tokens_lossy(system_prompt);
        let total_budget = get_context_budget(model, agent_options.context_window_utilization);
        let user_suffix = prompts::extract_semantics_user_suffix(categories);
        let suffix_tokens = model.config.count_tokens_lossy(&user_suffix);

        // Caller-overridable per-chunk input budget; default = ~80% of
        // model max input minus the fixed system + suffix overhead.
        let chunk_budget = match chunk_input_budget {
            Some(cap) => cap.min(total_budget.saturating_sub(sys_tokens + suffix_tokens)),
            None => total_budget.saturating_sub(sys_tokens + suffix_tokens),
        };

        let all_files = match call_graph {
            Some(cg) => self.graph_ordered_body(cg),
            None => self.build_project_prompt_body(),
        };
        let Some(mut cursor) = TokenCursor::new(all_files, model.clone()) else {
            return Err(KgError::other(
                "Failed to initialize TokenCursor for extraction",
            ));
        };

        // Checkpoint resume: load already-completed chunks for this
        // stage. Empty when no sink is configured (hard invariant:
        // None produces byte-identical behavior).
        let checkpointed: std::collections::HashMap<
            u32,
            (String, Vec<u8>),
        > = match sink {
            Some(s) => s
                .load_stage(crate::checkpoint::stage::SEMANTIC_EXTRACT)
                .unwrap_or_default(),
            None => std::collections::HashMap::new(),
        };
        if !checkpointed.is_empty() {
            tracing::info!(
                "Checkpoint resume: {} semantic-extract chunk(s) already completed for {}",
                checkpointed.len(),
                self.display_id(),
            );
        }

        let mut all_semantics = Vec::new();
        let mut chunk_idx = 0usize;
        while let Some(chunk) = cursor.next_chunk(chunk_budget) {
            let hash = crate::checkpoint::chunk_hash(&chunk);

            // Resume from checkpoint when the hash matches (same source
            // text, same budget => same deterministic chunk boundaries)
            if let Some((stored_hash, payload)) =
                checkpointed.get(&(chunk_idx as u32))
            {
                if *stored_hash == hash {
                    match serde_json::from_slice::<Vec<ExtractedSemantic>>(
                        payload,
                    ) {
                        Ok(stored) => {
                            tracing::info!(
                                "Chunk {} resumed from checkpoint ({} semantic(s))",
                                chunk_idx,
                                stored.len(),
                            );
                            all_semantics.extend(stored);
                            chunk_idx += 1;
                            continue;
                        }
                        Err(e) => {
                            tracing::warn!(
                                "Checkpoint chunk {} deserialize failed ({}); re-extracting",
                                chunk_idx,
                                e,
                            );
                        }
                    }
                } else {
                    tracing::warn!(
                        "Checkpoint chunk {} hash mismatch (source changed); re-extracting",
                        chunk_idx,
                    );
                }
            }

            let mut user_prompt = format!("{}{}", chunk, user_suffix);
            if let Some(cg) = call_graph {
                let cycles = shard_pack::cycle_groups(cg);
                if !cycles.is_empty() {
                    user_prompt.push_str("\n## Call-graph cycle anchors (keep each cycle's mechanism in one semantic)\n");
                    for g in &cycles {
                        user_prompt.push_str(&format!("- {}\n", shard_pack::cycle_signature(cg, g)));
                    }
                }
                user_prompt.push_str(&shard_pack::graph_features(cg, None));
            }
            tracing::info!(
                "Extracting semantics from chunk {} (~{} tokens, done={})",
                chunk_idx,
                sys_tokens + model.config.count_tokens_lossy(&user_prompt),
                cursor.is_done(),
            );
            let chunk_label = format!("semantic-extract-{}-chunk{}", self.display_id(), chunk_idx);
            let chunk_debug = format!("{}-chunk{}", debug_key, chunk_idx);
            let local_options = agent_options.scoped(&chunk_debug);
            let extractor = SemanticChunkExtractor {
                llm: llm.clone(),
                options: local_options,
                system_prompt: system_prompt.to_string(),
                user_prompt,
                label: chunk_label,
                call_graph: call_graph.map(|c| std::sync::Arc::new(c.clone())), 
                storage_deps: std::sync::Arc::new(std::collections::BTreeMap::new()),
                existing_nodes: std::sync::Arc::new(Vec::new()),
            };
            let chunk_semantics = extractor.run().await?;
            tracing::info!(
                "Chunk {} produced {} semantic(s)",
                chunk_idx,
                chunk_semantics.len(),
            );

            // Checkpoint this chunk (best-effort: extraction already
            // succeeded; a checkpoint write failure is logged, not fatal)
            if let Some(sink) = sink {
                let payload = serde_json::to_vec(&chunk_semantics)
                    .map_err(|e| {
                        KgError::other(&format!(
                            "checkpoint serialize: {}",
                            e
                        ))
                    })?;
                if let Err(e) = sink.write_chunk(
                    crate::checkpoint::stage::SEMANTIC_EXTRACT,
                    chunk_idx as u32,
                    &hash,
                    &payload,
                ) {
                    tracing::warn!(
                        "Checkpoint write failed for chunk {}: {}",
                        chunk_idx,
                        e,
                    );
                }
            }

            all_semantics.extend(chunk_semantics);
            chunk_idx += 1;
        }
        Ok(all_semantics)
    }

    /// LLM-driven in-project linking. Returns positional `(finding_idx,
    /// semantic_idx)` edges. Validates that every finding got at least one
    /// link (the LLM is instructed to enforce this; we re-check and bail
    /// on violations rather than silently produce an unlinked finding).
    /// FGW structural+feature coupling (docs/SEMANTIC_EXTRACTION_REVIEW_AND_MATH.md §3.3).
    /// Proposes additional mechanism<->finding edges: feature = text distance, structure =
    /// semantic function names mentioned in the finding prose. Augments (never removes) LLM links.
    fn fgw_structural_edges(semantics: &[ExtractedSemantic], findings: &[ExtractedFinding]) -> Vec<(usize, usize)> {
        let mut idmap: std::collections::BTreeMap<String, i32> = std::collections::BTreeMap::new();
        let mut next = 0i32;
        for s in semantics {
            for fn_ in &s.functions {
                let k = format!("{}::{}", fn_.contract, fn_.name);
                if !idmap.contains_key(&k) { idmap.insert(k, next); next += 1; }
            }
        }
        let sem: Vec<(String, Vec<i32>)> = semantics.iter().map(|s| {
            let ids = s.functions.iter().filter_map(|fn_| idmap.get(&format!("{}::{}", fn_.contract, fn_.name)).copied()).collect();
            (format!("{} {}", s.name, s.definition), ids)
        }).collect();
        let find: Vec<(String, Vec<i32>)> = findings.iter().map(|fd| {
            let text = format!("{} {} {} {}", fd.title, fd.root_cause, fd.patterns, fd.description).to_lowercase();
            let ids = idmap.iter().filter(|(k, _)| {
                let fname = k.rsplit("::").next().unwrap_or("").to_lowercase();
                fname.len() > 3 && text.contains(&fname)
            }).map(|(_, v)| *v).collect();
            (format!("{} {}", fd.title, fd.root_cause), ids)
        }).collect();
        knowdit_cg_static::fgw::link_semantics_findings(&sem, &find, 0.5)
            .into_iter().enumerate().map(|(s_idx, f_idx)| (f_idx, s_idx)).collect()
    }

    async fn link_findings_in_project(
        &self,
        llm: &LLM,
        categories: &[DeFiCategory],
        semantics: &[ExtractedSemantic],
        findings: &[ExtractedFinding],
    ) -> Result<InProjectLinks> {
        debug_assert!(!findings.is_empty() && !semantics.is_empty());

        let semantics_block = render_semantics_for_in_project_link(semantics);
        let findings_block = render_findings_for_in_project_link(findings);
        let user_msg =
            prompts::in_project_link_user_message(categories, &semantics_block, &findings_block);

        let debug_key = self.debug_key("in-project-link");

        let parsed: InProjectLinkResponse = llm
            .prompt_json_with_retry(
                prompts::GENERAL_ROLE_SYSTEM,
                &user_msg,
                Some(&debug_key),
                None,
                None,
            )
            .await?;

        let mut edges: Vec<(usize, usize)> = Vec::new();
        let mut seen: HashSet<(usize, usize)> = HashSet::new();
        let mut covered: HashSet<usize> = HashSet::new();
        for entry in &parsed.links {
            let f_idx = entry.finding_index;
            if f_idx >= findings.len() {
                return Err(KgError::other(format!(
                    "in-project link response references unknown finding_index {}",
                    f_idx
                )));
            }
            if entry.semantic_indices.is_empty() {
                return Err(KgError::other(format!(
                    "in-project link response left finding_index {} unlinked (rule violation)",
                    f_idx
                )));
            }
            for s_idx in &entry.semantic_indices {
                if *s_idx >= semantics.len() {
                    return Err(KgError::other(format!(
                        "in-project link response references unknown semantic_index {} for finding {}",
                        s_idx, f_idx
                    )));
                }
                if seen.insert((f_idx, *s_idx)) {
                    edges.push((f_idx, *s_idx));
                }
            }
            covered.insert(f_idx);
        }
        if covered.len() != findings.len() {
            let missing: Vec<usize> = (0..findings.len())
                .filter(|i| !covered.contains(i))
                .collect();
            return Err(KgError::other(format!(
                "in-project link response missing {} finding(s): {:?}",
                missing.len(),
                missing
            )));
        }

        for (f_idx, s_idx) in Self::fgw_structural_edges(semantics, findings) {
            if seen.insert((f_idx, s_idx)) {
                edges.push((f_idx, s_idx));
            }
        }
        Ok(InProjectLinks { edges })
    }

    /// Extract audit findings from the project's report. Same chunked
    /// agent pattern as [`Self::extract_semantics`]: one Agent per chunk,
    /// `emit_finding` tool per finding, terminated by
    /// `finalize_finding_extraction`.
    async fn extract_findings(
        &self,
        llm: &LLM,
        categories: &[DeFiCategory],
        agent_options: &AgentRunOptions,
        chunk_input_budget: Option<usize>,
        sink: Option<&crate::checkpoint::CheckpointSink>,
    ) -> Result<Vec<ExtractedFinding>> {
        let Some(report_body) = self.build_report_prompt_body() else {
            tracing::warn!("No audit report found for project {}", self.display_id());
            return Ok(Vec::new());
        };

        let system_prompt = prompts::GENERAL_ROLE_SYSTEM;
        let model = &llm.model;
        let debug_key = self.debug_key("finding-extract");
        let sys_tokens = model.config.count_tokens_lossy(system_prompt);
        let total_budget = get_context_budget(model, agent_options.context_window_utilization);
        let user_suffix = prompts::extract_findings_user_suffix(categories);
        let suffix_tokens = model.config.count_tokens_lossy(&user_suffix);
        let chunk_budget = match chunk_input_budget {
            Some(cap) => cap.min(total_budget.saturating_sub(sys_tokens + suffix_tokens)),
            None => total_budget.saturating_sub(sys_tokens + suffix_tokens),
        };

        let Some(mut cursor) = TokenCursor::new(report_body, model.clone()) else {
            return Err(KgError::other(
                "Failed to initialize TokenCursor for finding extraction",
            ));
        };

        // Checkpoint resume: load already-completed chunks
        let checkpointed: std::collections::HashMap<
            u32,
            (String, Vec<u8>),
        > = match sink {
            Some(s) => s
                .load_stage(crate::checkpoint::stage::FINDING_EXTRACT)
                .unwrap_or_default(),
            None => std::collections::HashMap::new(),
        };
        if !checkpointed.is_empty() {
            tracing::info!(
                "Checkpoint resume: {} finding-extract chunk(s) already completed for {}",
                checkpointed.len(),
                self.display_id(),
            );
        }

        let mut all_findings = Vec::new();
        let mut chunk_idx = 0usize;
        while let Some(chunk) = cursor.next_chunk(chunk_budget) {
            let hash = crate::checkpoint::chunk_hash(&chunk);

            // Resume from checkpoint when the hash matches
            if let Some((stored_hash, payload)) =
                checkpointed.get(&(chunk_idx as u32))
            {
                if *stored_hash == hash {
                    match serde_json::from_slice::<Vec<ExtractedFinding>>(
                        payload,
                    ) {
                        Ok(stored) => {
                            tracing::info!(
                                "Chunk {} resumed from checkpoint ({} finding(s))",
                                chunk_idx,
                                stored.len(),
                            );
                            // Re-canonicalize: the stored payload is the
                            // RAW findings; canonicalize is deterministic
                            for finding in stored {
                                all_findings
                                    .push(Self::canonicalize_finding(finding)?);
                            }
                            chunk_idx += 1;
                            continue;
                        }
                        Err(e) => {
                            tracing::warn!(
                                "Checkpoint chunk {} deserialize failed ({}); re-extracting",
                                chunk_idx,
                                e,
                            );
                        }
                    }
                } else {
                    tracing::warn!(
                        "Checkpoint chunk {} hash mismatch (source changed); re-extracting",
                        chunk_idx,
                    );
                }
            }

            let user_prompt = format!("{}{}", chunk, user_suffix);
            tracing::info!(
                "Extracting findings from chunk {} (~{} tokens, done={})",
                chunk_idx,
                sys_tokens + model.config.count_tokens_lossy(&user_prompt),
                cursor.is_done(),
            );
            let chunk_label = format!("finding-extract-{}-chunk{}", self.display_id(), chunk_idx);
            let chunk_debug = format!("{}-chunk{}", debug_key, chunk_idx);
            let local_options = agent_options.scoped(&chunk_debug);
            let extractor = FindingChunkExtractor {
                llm: llm.clone(),
                options: local_options,
                system_prompt: system_prompt.to_string(),
                user_prompt,
                label: chunk_label,
            };
            let raw_findings = extractor.run().await?;
            tracing::info!(
                "Chunk {} produced {} finding(s)",
                chunk_idx,
                raw_findings.len(),
            );

            // Checkpoint the RAW findings (pre-canonicalize) so the
            // resume path can re-apply the same deterministic transform
            if let Some(sink) = sink {
                let payload = serde_json::to_vec(&raw_findings)
                    .map_err(|e| {
                        KgError::other(&format!(
                            "checkpoint serialize: {}",
                            e
                        ))
                    })?;
                if let Err(e) = sink.write_chunk(
                    crate::checkpoint::stage::FINDING_EXTRACT,
                    chunk_idx as u32,
                    &hash,
                    &payload,
                ) {
                    tracing::warn!(
                        "Checkpoint write failed for chunk {}: {}",
                        chunk_idx,
                        e,
                    );
                }
            }

            for finding in raw_findings {
                all_findings.push(Self::canonicalize_finding(finding)?);
            }
            chunk_idx += 1;
        }

        Ok(all_findings)
    }

    /// Deduplicate semantics by name (case-insensitive). Keeps the longer
    /// description and merges function lists.
    fn dedup_semantics(semantics: Vec<ExtractedSemantic>) -> Vec<ExtractedSemantic> {
        let mut by_name: HashMap<String, ExtractedSemantic> = HashMap::new();

        for sem in semantics {
            let key = sem.name.to_lowercase().trim().to_string();
            if let Some(existing) = by_name.get_mut(&key) {
                for func in sem.functions {
                    let already_has = existing
                        .functions
                        .iter()
                        .any(|f| f.name == func.name && f.contract == func.contract);
                    if !already_has {
                        existing.functions.push(func);
                    }
                }
                if sem.description.len() > existing.description.len() {
                    existing.description = sem.description;
                    existing.definition = sem.definition;
                }
            } else {
                by_name.insert(key, sem);
            }
        }

        by_name.into_values().collect()
    }

    fn dedup_findings(findings: Vec<ExtractedFinding>) -> Vec<ExtractedFinding> {
        let mut by_title: HashMap<String, ExtractedFinding> = HashMap::new();

        for finding in findings {
            let key = finding.title.to_lowercase().trim().to_string();
            if let Some(existing) = by_title.get_mut(&key) {
                existing.severity = existing.severity.max(finding.severity);

                if finding.description.len() > existing.description.len() {
                    existing.category = finding.category;
                    existing.subcategory = finding.subcategory.clone();
                    existing.description = finding.description.clone();
                }

                if finding.root_cause.len() > existing.root_cause.len() {
                    existing.root_cause = finding.root_cause.clone();
                }

                if finding.patterns.len() > existing.patterns.len() {
                    existing.patterns = finding.patterns.clone();
                }

                if finding.exploits.len() > existing.exploits.len() {
                    existing.exploits = finding.exploits.clone();
                }
            } else {
                by_title.insert(key, finding);
            }
        }

        by_title.into_values().collect()
    }

    fn canonicalize_finding(mut finding: ExtractedFinding) -> Result<ExtractedFinding> {
        finding.title = finding.title.trim().to_string();
        finding.root_cause = finding.root_cause.trim().to_string();
        finding.description = finding.description.trim().to_string();
        finding.patterns = finding.patterns.trim().to_string();
        finding.exploits = finding.exploits.trim().to_string();

        let Some(entry) = resolve_taxonomy_entry(finding.category, &finding.subcategory) else {
            return Err(KgError::other(format!(
                "Unknown vulnerability subcategory '{}' for category '{}'",
                finding.subcategory, finding.category
            )));
        };

        finding.subcategory = entry.subcategory.to_string();
        Ok(finding)
    }
}

impl ProjectData {
    /// Merge newly-extracted semantics against the historical KB. Chunks
    /// the existing canonicals (with their merged-away raw children
    /// rendered alongside, so the LLM's `updated_*` generalizations can
    /// take prior merges into account) by token budget, runs one merge
    /// agent per chunk in parallel via [`SemanticMerger`], and unions the
    /// per-chunk decisions into the final per-raw [`MergeAction`].
    ///
    /// A new raw can merge into multiple canonicals; that's encoded in
    /// `MergeAction::Merge { target_ids, .. }`.
    async fn merge_with_existing(
        &self,
        db: &HistoricalDatabase,
        llm: &LLM,
        extract: &ExtractResult,
        agent_options: &AgentRunOptions,
        merge_chunking: MergeChunkingOptions,
    ) -> Result<Vec<MergeResult>> {
        if extract.semantics.is_empty() {
            return Ok(Vec::new());
        }
        let semantic_categories: Vec<DeFiCategory> = extract
            .semantics
            .iter()
            .map(|sem| sem.category)
            .unique()
            .collect();
        let candidates = db
            .canonical_semantics_with_children_for_categories(&semantic_categories)
            .await?;
        let merger = SemanticMerger {
            new_semantics: extract.semantics.clone(),
            candidates,
            llm: llm.clone(),
            agent_options: agent_options.clone(),
            chunking: merge_chunking,
            debug_key_root: self.debug_key("merge"),
            label_root: format!("semantic-merge-{}", self.display_id()),
        };
        let aggregated = merger.run().await?;
        Ok(Self::semantic_merge_results_from_aggregated(
            &extract.semantics,
            aggregated,
        ))
    }

    /// Convert the orchestrator's `(name → AggregatedSemanticMergeDecision)`
    /// list into the persistence layer's `MergeResult` (one per raw, in the
    /// same order as `extracted`).
    fn semantic_merge_results_from_aggregated(
        extracted: &[ExtractedSemantic],
        aggregated: Vec<AggregatedSemanticMergeDecision>,
    ) -> Vec<MergeResult> {
        let by_name: HashMap<String, AggregatedSemanticMergeDecision> = aggregated
            .into_iter()
            .map(|d| (d.new_semantic_name.to_lowercase(), d))
            .collect();
        extracted
            .iter()
            .map(|sem| {
                let action = match by_name.get(&sem.name.to_lowercase()) {
                    Some(d) if !d.merge_target_ids.is_empty() => MergeAction::Merge {
                        target_ids: d.merge_target_ids.clone(),
                        updated_description: d.updated_description.clone(),
                        appended_description: d.appended_description.clone(),
                    },
                    _ => MergeAction::New,
                };
                MergeResult {
                    semantic: sem.clone(),
                    action,
                }
            })
            .collect()
    }

    /// Merge newly-extracted findings against the historical KB. Same
    /// chunked + parallel design as [`Self::merge_with_existing`]; uses
    /// [`FindingMerger`] under the hood. Multi-target merges supported.
    async fn merge_findings_with_existing(
        &self,
        db: &HistoricalDatabase,
        llm: &LLM,
        extract: &ExtractResult,
        agent_options: &AgentRunOptions,
        merge_chunking: MergeChunkingOptions,
    ) -> Result<Vec<FindingMergeResult>> {
        if extract.findings.is_empty() {
            return Ok(Vec::new());
        }
        let finding_categories: Vec<VulnerabilityCategory> = extract
            .findings
            .iter()
            .map(|finding| finding.category)
            .unique()
            .collect();
        let candidates = db
            .canonical_findings_with_children_for_categories(&finding_categories)
            .await?;
        let merger = FindingMerger {
            new_findings: extract.findings.clone(),
            candidates,
            llm: llm.clone(),
            agent_options: agent_options.clone(),
            chunking: merge_chunking,
            debug_key_root: self.debug_key("finding-merge"),
            label_root: format!("finding-merge-{}", self.display_id()),
        };
        let aggregated = merger.run().await?;
        // Coverage check (2026-11-06): a model that under-emits merge
        // decisions would otherwise silently turn every undecided finding
        // into a NEW canonical. Log loudly with the exact shortfall so a
        // model swap (e.g. a small fast model on this stage) cannot
        // corrupt the KG unnoticed.
        let (results, missing) =
            Self::finding_merge_results_with_coverage(&extract.findings, aggregated);
        if !missing.is_empty() {
            tracing::warn!(
                "finding-merge COVERAGE SHORTFALL for {}: {}/{} findings received no \
                 decision and were defaulted to NEW: {:?}",
                self.display_id(),
                missing.len(),
                extract.findings.len(),
                missing,
            );
        }
        Ok(results)
    }

    /// Coverage-aware variant of [`Self::finding_merge_results_from_aggregated`].
    ///
    /// MEASURED DEFECT (2026-11-06): the plain variant maps a MISSING
    /// decision to `FindingMergeAction::New`. A model that under-emits
    /// decisions is therefore silently tolerated - measured on
    /// meta-llama/llama-3.1-8b-instruct, 3 of 12 decisions were returned
    /// without an explicit count in the prompt, which would have created
    /// 9 duplicate canonicals with no error anywhere.
    ///
    /// This variant returns the titles of every finding that received NO
    /// decision, so the caller can fail closed (or at minimum log loudly)
    /// instead of silently duplicating the knowledge graph.
    fn finding_merge_results_with_coverage(
        extracted: &[ExtractedFinding],
        aggregated: Vec<AggregatedFindingMergeDecision>,
    ) -> (Vec<FindingMergeResult>, Vec<String>) {
        let decided: std::collections::HashSet<String> = aggregated
            .iter()
            .map(|d| d.new_finding_title.to_lowercase())
            .collect();
        let missing: Vec<String> = extracted
            .iter()
            .filter(|f| !decided.contains(&f.title.to_lowercase()))
            .map(|f| f.title.clone())
            .collect();
        (
            Self::finding_merge_results_from_aggregated(extracted, aggregated),
            missing,
        )
    }

    fn finding_merge_results_from_aggregated(
        extracted: &[ExtractedFinding],
        aggregated: Vec<AggregatedFindingMergeDecision>,
    ) -> Vec<FindingMergeResult> {
        let by_title: HashMap<String, AggregatedFindingMergeDecision> = aggregated
            .into_iter()
            .map(|d| (d.new_finding_title.to_lowercase(), d))
            .collect();
        extracted
            .iter()
            .map(|finding| {
                let action = match by_title.get(&finding.title.to_lowercase()) {
                    Some(d) if !d.merge_target_ids.is_empty() => FindingMergeAction::Merge {
                        target_ids: d.merge_target_ids.clone(),
                        updated_description: d.updated_description.clone(),
                        updated_patterns: d.updated_patterns.clone(),
                        updated_exploits: d.updated_exploits.clone(),
                        appended_description: d.appended_description.clone(),
                        appended_patterns: d.appended_patterns.clone(),
                        appended_exploits: d.appended_exploits.clone(),
                    },
                    _ => FindingMergeAction::New,
                };
                FindingMergeResult {
                    finding: finding.clone(),
                    action,
                }
            })
            .collect()
    }
}

/// Default fraction of a model's max input a single prompt may fill, used
/// where no CLI override is threaded in (categorize / extract). Kept well
/// below 1.0 on purpose: over-long prompts dilute the model's attention, so
/// we trade packing efficiency (more, smaller chunks/batches) for sharper
/// per-prompt focus — and 0.2 (~200K on a ~1M-token window registration) also
/// keeps every prompt, plus multi-step agent growth, inside 256K-window
/// deployments (0.4 on a 922K-window registration produced real 290K+
/// requests). Overridable per-command — e.g. `merge-kg`'s
/// `--context-window-utilization` threads a value into the merge + link
/// budgets.
pub const DEFAULT_CONTEXT_WINDOW_UTILIZATION: f64 = 0.2;

/// LOCAL PATCH (Nexusmill fork, 2026-10-31): llmy's registry does not
/// know OpenRouter-native model slugs (e.g.
/// `deepseek/deepseek-v4.1-flash-1017`) and reports `max_input() == 0`
/// for them, which collapsed every window-derived prompt budget to 1
/// token and hard-failed the finding-link pass. An exact-zero registry
/// window means "unknown" — fall back to a conservative default
/// (override: `KNOWDIT_DEFAULT_CONTEXT_WINDOW`). Non-zero windows are
/// returned verbatim, so genuinely small real windows keep their true
/// budget (gate finding 2, gate_20261031-022743).
pub fn effective_max_input(model: &OpenAIModel) -> u64 {
    let registry_max = model.config.max_input();
    if registry_max > 0 {
        return registry_max;
    }
    std::env::var("KNOWDIT_DEFAULT_CONTEXT_WINDOW")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .filter(|v| *v > 0) // gate_20261031-023513: a zero override must not
                            // silently reintroduce the zero-budget failure
        .unwrap_or(131_072)
}

/// Token budget for a single prompt: `utilization` × the model's max input.
/// `utilization` is clamped to a sane `(0, 1]` band so a mis-typed CLI value
/// can't zero out (or overflow) the budget.
pub(crate) fn get_context_budget(model: &OpenAIModel, utilization: f64) -> usize {
    let utilization = utilization.clamp(0.05, 1.0);
    (effective_max_input(model) as f64 * utilization) as _
}

pub(crate) fn sanitize_prompt_prefix(value: &str) -> String {
    let mut out = String::new();
    let mut last_was_dash = false;

    for ch in value.chars() {
        if ch.is_ascii_alphanumeric() {
            out.push(ch.to_ascii_lowercase());
            last_was_dash = false;
        } else if !last_was_dash {
            out.push('-');
            last_was_dash = true;
        }

        if out.len() >= 48 {
            break;
        }
    }

    let trimmed = out.trim_matches('-');
    if trimmed.is_empty() {
        "project".to_string()
    } else {
        trimmed.to_string()
    }
}

// Merge-response validation lived here in the JSON-mode era: the prompt
// returned a flat array of decisions which we cross-checked against the
// existing canonical id set. With the agent-tool form, validation is
// inlined into `project_*_merge_results`: decisions referencing an
// unknown id are downgraded to `New` (and logged) rather than rejected.

#[cfg(test)]
mod merge_coverage_tests {
    use super::*;
    use knowdit_kg_model::audit_finding::FindingSeverity;
    use knowdit_kg_model::finding_category::VulnerabilityCategory;

    fn finding(title: &str) -> ExtractedFinding {
        ExtractedFinding {
            title: title.to_string(),
            severity: FindingSeverity::High,
            category: VulnerabilityCategory::Arithmetic,
            subcategory: "s".to_string(),
            root_cause: "rc".to_string(),
            description: "d".to_string(),
            patterns: "p".to_string(),
            exploits: "e".to_string(),
        }
    }

    fn decision(title: &str, targets: Vec<i32>) -> AggregatedFindingMergeDecision {
        AggregatedFindingMergeDecision {
            new_finding_title: title.to_string(),
            merge_target_ids: targets,
            updated_description: None,
            updated_patterns: None,
            updated_exploits: None,
            appended_description: None,
            appended_patterns: None,
            appended_exploits: None,
        }
    }

    /// MEASURED DEFECT (2026-11-06): a model that under-emits merge
    /// decisions is SILENTLY tolerated. `finding_merge_results_from_aggregated`
    /// maps a missing decision to `FindingMergeAction::New`, so a model
    /// returning 3 of 12 decisions produces 9 duplicate canonicals with
    /// no error anywhere. Measured on meta-llama/llama-3.1-8b-instruct:
    /// 3 of 12 without an explicit count in the prompt.
    ///
    /// The aggregation must therefore REPORT the shortfall so the caller
    /// can fail closed instead of silently duplicating the KG.
    #[test]
    fn missing_decisions_are_reported_not_silently_new() {
        let extracted = vec![finding("a"), finding("b"), finding("c")];
        // only ONE decision supplied for THREE findings
        let aggregated = vec![decision("a", vec![7])];
        let (results, missing) =
            ProjectData::finding_merge_results_with_coverage(&extracted, aggregated);
        assert_eq!(results.len(), 3, "every finding still gets a result");
        assert_eq!(
            missing,
            vec!["b".to_string(), "c".to_string()],
            "the two undecided findings must be reported by title"
        );
    }

    /// Full coverage reports nothing missing.
    #[test]
    fn full_coverage_reports_no_shortfall() {
        let extracted = vec![finding("a"), finding("b")];
        let aggregated = vec![decision("a", vec![7]), decision("b", vec![])];
        let (results, missing) =
            ProjectData::finding_merge_results_with_coverage(&extracted, aggregated);
        assert_eq!(results.len(), 2);
        assert!(missing.is_empty(), "no shortfall when every finding is decided");
    }

    /// Title matching is case-insensitive (the existing behaviour), so a
    /// decision differing only in case must NOT count as missing.
    #[test]
    fn coverage_matching_is_case_insensitive() {
        let extracted = vec![finding("Missing Slippage Check")];
        let aggregated = vec![decision("missing slippage check", vec![7])];
        let (_results, missing) =
            ProjectData::finding_merge_results_with_coverage(&extracted, aggregated);
        assert!(missing.is_empty(), "case-only difference is not a shortfall");
    }
}


#[cfg(test)]
mod readme_order_tests {
    use super::*;

    /// REGRESSION (2026-11-13): the categorize cap bounds the prompt body
    /// to N tokens. The README used to be appended AFTER every source
    /// file, so any project above the cap lost it entirely - and the
    /// README is the most explicitly descriptive categorization signal.
    /// The label then came from only the leading slice of the file list,
    /// and that label propagates into every extraction prompt via
    /// `extract_semantics_user_suffix(categories)`.
    ///
    /// The README must therefore appear BEFORE the first source file.
    #[tokio::test]
    async fn readme_precedes_source_files_in_the_prompt_body() {
        let dir = std::env::temp_dir().join(format!(
            "knowdit-readme-order-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("contracts")).expect("mkdir");
        std::fs::write(
            dir.join("README.md"),
            "# Vault protocol\n\nA lending market with oracle-priced collateral.\n",
        )
        .expect("write readme");
        std::fs::write(
            dir.join("contracts").join("Vault.sol"),
            "pragma solidity ^0.8.0;\ncontract Vault { uint256 s_total; }\n",
        )
        .expect("write source");

        let project = ProjectData::from_dir("readme-order", &dir, None)
            .await
            .expect("load project");
        let body = project.build_project_prompt_body();

        let readme_at = body.find("## README").expect(
            "the README must be present in the body; a missing README means \
             the fixture did not load and this test would pass vacuously"
        );
        let first_source_at = body.find("### contracts").expect(
            "the source file must be present in the body"
        );
        assert!(
            readme_at < first_source_at,
            "the README must precede the source files so a bounded \
             categorize budget keeps the best signal; got README at {} and \
             first source at {}",
            readme_at,
            first_source_at
        );

        let _ = std::fs::remove_dir_all(&dir);
    }
}
