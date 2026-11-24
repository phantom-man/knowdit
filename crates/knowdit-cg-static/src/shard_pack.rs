//! Call-graph-guided shard packing (H1 / persistent-homology proxy) + anchoring gate.
//!
//! Implements the pre-chunking intervention from
//! `docs/SEMANTIC_EXTRACTION_REVIEW_AND_MATH.md`: token-budget chunking can slice a
//! multi-contract mechanism across shards. Here we compute strongly-connected components
//! (SCCs) over the function call graph as a deterministic proxy for high-persistence H1
//! cycles, co-locate co-cyclical functions in one shard, and expose a cycle call-chain
//! signature as a structural anchor. A post-extraction anchoring gate rejects semantics
//! whose functions do not lie in the claimed contract neighborhood.

use std::collections::{BTreeMap, BTreeSet};

use knowdit_repo_model::cg::CallGraph;
use knowdit_repo_model::storage::StorageGraph;

/// Function-level adjacency (caller -> callees) and id->name, over contract functions.
fn adjacency(cg: &CallGraph) -> (BTreeMap<i32, Vec<i32>>, BTreeMap<i32, String>) {
    let mut adj: BTreeMap<i32, Vec<i32>> = BTreeMap::new();
    let mut names: BTreeMap<i32, String> = BTreeMap::new();
    for c in cg.contracts.values() {
        for f in &c.functions {
            adj.entry(f.id).or_default();
            names.insert(f.id, f.name.clone());
            for call in &f.calls {
                adj.entry(f.id).or_default().push(call.to_id);
                adj.entry(call.to_id).or_default();
            }
        }
    }
    (adj, names)
}

/// Tarjan SCC over the call graph. Returns components in deterministic order.
fn scc(adj: &BTreeMap<i32, Vec<i32>>) -> Vec<Vec<i32>> {
    let mut index = 0i32;
    let mut indices: BTreeMap<i32, i32> = BTreeMap::new();
    let mut low: BTreeMap<i32, i32> = BTreeMap::new();
    let mut on_stack: BTreeSet<i32> = BTreeSet::new();
    let mut stack: Vec<i32> = Vec::new();
    let mut out: Vec<Vec<i32>> = Vec::new();

    fn strongconnect(
        v: i32,
        adj: &BTreeMap<i32, Vec<i32>>,
        index: &mut i32,
        indices: &mut BTreeMap<i32, i32>,
        low: &mut BTreeMap<i32, i32>,
        on_stack: &mut BTreeSet<i32>,
        stack: &mut Vec<i32>,
        out: &mut Vec<Vec<i32>>,
    ) {
        indices.insert(v, *index);
        low.insert(v, *index);
        *index += 1;
        stack.push(v);
        on_stack.insert(v);
        for &w in adj.get(&v).unwrap_or(&Vec::new()) {
            if !indices.contains_key(&w) {
                strongconnect(w, adj, index, indices, low, on_stack, stack, out);
                let lw = *low.get(&w).unwrap();
                let lv = *low.get(&v).unwrap();
                low.insert(v, lv.min(lw));
            } else if on_stack.contains(&w) {
                let iw = *indices.get(&w).unwrap();
                let lv = *low.get(&v).unwrap();
                low.insert(v, lv.min(iw));
            }
        }
        if low.get(&v) == indices.get(&v) {
            let mut comp = Vec::new();
            while let Some(w) = stack.pop() {
                on_stack.remove(&w);
                comp.push(w);
                if w == v {
                    break;
                }
            }
            comp.sort();
            out.push(comp);
        }
    }

    for &v in adj.keys() {
        if !indices.contains_key(&v) {
            strongconnect(v, adj, &mut index, &mut indices, &mut low, &mut on_stack, &mut stack, &mut out);
        }
    }
    out.sort();
    out
}

/// SCCs with more than one node = call cycles (H1 proxy).
pub fn cycle_groups(cg: &CallGraph) -> Vec<Vec<i32>> {
    let (adj, _) = adjacency(cg);
    scc(&adj).into_iter().filter(|c| c.len() > 1).collect()
}

/// Shard groups: cycle groups kept intact, plus singletons for acyclic functions.
pub fn shard_groups(cg: &CallGraph) -> Vec<Vec<i32>> {
    let (adj, _) = adjacency(cg);
    let comps = scc(&adj);
    let mut groups: Vec<Vec<i32>> = comps.into_iter().filter(|c| c.len() > 1).collect();
    let in_cycle: BTreeSet<i32> = groups.iter().flatten().copied().collect();
    for &v in adj.keys() {
        if !in_cycle.contains(&v) {
            groups.push(vec![v]);
        }
    }
    groups.sort();
    groups
}

fn tokens_est(cg: &CallGraph, ids: &[i32]) -> usize {
    let mut total = 0usize;
    for c in cg.contracts.values() {
        for f in &c.functions {
            if ids.contains(&f.id) {
                total += f.content.as_deref().map(|s| s.len() / 4).unwrap_or(8);
            }
        }
    }
    total
}

/// Call-chain signature for a cycle group (structural anchor for the prompt).
pub fn cycle_signature(cg: &CallGraph, ids: &[i32]) -> String {
    let (_, names) = adjacency(cg);
    let chain: Vec<String> = ids.iter().map(|i| names.get(i).cloned().unwrap_or_default()).collect();
    let mut s = chain.join(" -> ");
    if let Some(first) = chain.first() {
        s.push_str(" -> ");
        s.push_str(first);
    }
    s
}

#[derive(Debug, Clone)]
pub struct Shard {
    pub function_ids: Vec<i32>,
    pub tokens_est: usize,
    pub signature: Option<String>,
}

/// Pack shard groups into budget-sized shards, never splitting a cycle group.
pub fn pack_shards(cg: &CallGraph, budget_tokens: usize) -> Vec<Shard> {
    let cycles: BTreeSet<Vec<i32>> = cycle_groups(cg).into_iter().collect();
    let mut shards: Vec<Shard> = Vec::new();
    let mut cur: Vec<i32> = Vec::new();
    let mut cur_tok = 0usize;
    let mut cur_sig: Option<String> = None;
    for g in shard_groups(cg) {
        let gt = tokens_est(cg, &g);
        let is_cycle = cycles.contains(&g);
        if !cur.is_empty() && cur_tok + gt > budget_tokens {
            shards.push(Shard { function_ids: cur.clone(), tokens_est: cur_tok, signature: cur_sig.clone() });
            cur.clear();
            cur_tok = 0;
            cur_sig = None;
        }
        cur.extend_from_slice(&g);
        cur_tok += gt;
        if is_cycle {
            cur_sig = Some(cycle_signature(cg, &g));
        }
    }
    if !cur.is_empty() {
        shards.push(Shard { function_ids: cur, tokens_est: cur_tok, signature: cur_sig });
    }
    shards
}

/// Anchoring gate: every emitted function must belong to the claimed contract.
pub fn anchored(cg: &CallGraph, fn_ids: &[i32], container_id: i32) -> bool {
    match cg.contracts.get(&container_id) {
        Some(c) => {
            let member: BTreeSet<i32> = c.functions.iter().map(|f| f.id).collect();
            fn_ids.iter().all(|i| member.contains(i))
        }
        None => false,
    }
}

/// Anchoring gate for an emitted semantic: every (function, contract) anchor must
/// resolve to a real function, and all anchored contracts must lie in ONE shard group
/// (cycle-coherent). Unresolved or cross-shard anchors => unanchored (reject).
pub fn functions_anchored(cg: &CallGraph, funcs: &[(String, String)]) -> bool {
    if funcs.is_empty() {
        return false;
    }
    let groups = shard_groups(cg);
    let mut anchored_ids: BTreeSet<i32> = BTreeSet::new();
    for (fname, cpath) in funcs {
        let norm = cpath.replace('\\', "/");
        let mut found = None;
        for c in cg.contracts.values() {
            let cp = c.relative_file_path.to_string_lossy().replace('\\', "/");
            if cp == norm || cp.ends_with(&format!("/{}", norm)) || norm.ends_with(&format!("/{}", cp)) {
                if let Some(fn_) = c.functions.iter().find(|f| &f.name == fname) {
                    found = Some(fn_.id);
                    break;
                }
            }
        }
        match found {
            Some(id) => { anchored_ids.insert(id); }
            None => return false,
        }
    }
    // The anchored FUNCTIONS themselves must all lie in ONE shard group (cycle-coherent).
    groups.iter().any(|g| {
        let ids: BTreeSet<i32> = g.iter().copied().collect();
        anchored_ids.iter().all(|fid| ids.contains(fid))
    })
}

/// Render a shard's source as sequential `### path` + fenced-solidity sections (deterministic).
pub fn render_shard(cg: &CallGraph, shard: &Shard) -> String {
    let ids: BTreeSet<i32> = shard.function_ids.iter().copied().collect();
    let mut out = String::from("## Source Files\n\n");
    for c in cg.contracts.values() {
        let funcs: Vec<&knowdit_repo_model::cg::Function> = c.functions.iter().filter(|f| ids.contains(&f.id)).collect();
        if funcs.is_empty() { continue; }
        out.push_str(&format!("### {}\n", c.relative_file_path.to_string_lossy()));
        out.push_str("```solidity\n");
        for f in funcs {
            if let Some(content) = &f.content { out.push_str(content); out.push('\n'); }
        }
        out.push_str("```\n\n");
    }
    out
}

/// Materialize shards as SEQUENTIAL files (shard-0000.sol ...) + manifest.json BEFORE
/// extraction, so agents can resume by stable shard index+content-hash. Returns the
/// manifest entries (index, file, function_ids, signature, tokens_est).
pub fn materialize_shards(cg: &CallGraph, budget: usize, out_dir: &std::path::Path) -> std::io::Result<Vec<serde_json::Value>> {
    std::fs::create_dir_all(out_dir)?;
    let shards = pack_shards(cg, budget);
    let mut manifest = Vec::new();
    for (i, s) in shards.iter().enumerate() {
        let fname = format!("shard-{:04}.sol", i);
        let body = render_shard(cg, s);
        std::fs::write(out_dir.join(&fname), &body)?;
        manifest.push(serde_json::json!({
            "index": i, "file": fname, "function_ids": s.function_ids,
            "signature": s.signature, "tokens_est": s.tokens_est,
        }));
    }
    std::fs::write(out_dir.join("manifest.json"), serde_json::to_string_pretty(&manifest).unwrap())?;
    Ok(manifest)
}

/// Structural graph context to accompany source in the extraction prompt (graph-guided,
/// never replaces source). Reports cycle signatures, hub functions (in-degree), and
/// internal/external call edges so the model reasons about structure AND code.
/// If `ids` is Some, restrict to that shard's functions; else whole graph.
pub fn graph_features(cg: &CallGraph, ids: Option<&[i32]>) -> String {
    let scope: BTreeSet<i32> = match ids {
        Some(v) => v.iter().copied().collect(),
        None => cg.contracts.values().flat_map(|c| c.functions.iter().map(|f| f.id)).collect(),
    };
    let mut indeg: BTreeMap<i32, usize> = BTreeMap::new();
    let mut internal = 0usize;
    let mut external = 0usize;
    let name = |id: i32| -> String {
        cg.contracts.values().find_map(|c| c.functions.iter().find(|f| f.id == id).map(|f| f.name.clone())).unwrap_or_default()
    };
    for c in cg.contracts.values() {
        for f in &c.functions {
            if !scope.contains(&f.id) { continue; }
            for call in &f.calls {
                if scope.contains(&call.to_id) {
                    internal += 1;
                    *indeg.entry(call.to_id).or_insert(0) += 1;
                } else {
                    external += 1;
                }
            }
        }
    }
    let mut hubs: Vec<(i32, usize)> = indeg.into_iter().collect();
    hubs.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    let mut out = String::from("## Graph context (structure alongside the source)\n");
    out.push_str(&format!("- functions: {} | internal call edges: {} | external calls: {}\n", scope.len(), internal, external));
    if !hubs.is_empty() {
        let top: Vec<String> = hubs.iter().take(5).map(|(id, d)| format!("{} (in-degree {})", name(*id), d)).collect();
        out.push_str(&format!("- hub functions: {}\n", top.join(", ")));
    }
    if ids.is_none() {
        for g in cycle_groups(cg) {
            out.push_str(&format!("- cycle: {}\n", cycle_signature(cg, &g)));
        }
    }
    out
}

/// Build state-variable -> [function names] read/write map from the static call + storage
/// graphs, for the get_storage_dependencies agent tool (group coupled functions).
pub fn storage_dependencies(cg: &CallGraph, storage: &StorageGraph) -> BTreeMap<String, Vec<String>> {
    let fname: BTreeMap<i32, String> = cg.contracts.values()
        .flat_map(|c| c.functions.iter().map(|f| (f.id, f.name.clone()))).collect();
    let mut out: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for fsv in &storage.function_state_variables {
        let Some(var) = storage.state_variables.get(&fsv.state_variable_id) else { continue };
        let Some(fn_) = fname.get(&fsv.function_id) else { continue };
        let e = out.entry(var.name.clone()).or_default();
        if !e.contains(fn_) { e.push(fn_.clone()); }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use knowdit_repo_model::cg::{Contract, FileChunk, FileLocation, Function, FunctionCall};
    use std::path::PathBuf;

    fn loc() -> FileLocation {
        FileLocation { start_line: 1, start_column: 0, end_line: 2, end_column: 0 }
    }
    fn func(id: i32, name: &str, calls: Vec<i32>) -> Function {
        Function {
            id,
            name: name.to_string(),
            args: String::new(),
            relative_file_path: PathBuf::from("a.sol"),
            loc: loc(),
            content: Some("x".repeat(40)),
            calls: calls.into_iter().enumerate().map(|(i, to)| FunctionCall { id: i as i32, from_id: id, to_id: to, description: None }).collect(),
            description: None,
        }
    }
    fn graph() -> CallGraph {
        let mut cg = CallGraph { contracts: BTreeMap::new(), interfaces: BTreeMap::new() };
        cg.contracts.insert(1, Contract {
            id: 1, name: "Cyc".into(), relative_file_path: PathBuf::from("a.sol"),
            chunk: FileChunk { loc: loc(), content: String::new() },
            functions: vec![func(1, "A", vec![2]), func(2, "B", vec![3]), func(3, "C", vec![1])],
            description: None,
        });
        cg.contracts.insert(2, Contract {
            id: 2, name: "Solo".into(), relative_file_path: PathBuf::from("b.sol"),
            chunk: FileChunk { loc: loc(), content: String::new() },
            functions: vec![func(4, "D", vec![])],
            description: None,
        });
        cg
    }

    #[test]
    fn cycle_co_located() {
        let cg = graph();
        let cyc = cycle_groups(&cg);
        assert_eq!(cyc, vec![vec![1, 2, 3]]);
        let shards = pack_shards(&cg, 1000);
        // the 3-cycle must appear together in one shard
        assert!(shards.iter().any(|s| [1, 2, 3].iter().all(|i| s.function_ids.contains(i))));
        assert!(shards.iter().any(|s| s.signature.is_some()));
    }

    #[test]
    fn budget_never_splits_cycle() {
        let cg = graph();
        // tiny budget: cycle (30 tok) alone exceeds it -> own shard, never split
        let shards = pack_shards(&cg, 5);
        for s in &shards {
            let has = |i: i32| s.function_ids.contains(&i);
            assert!(!(has(1) ^ has(2)) && !(has(2) ^ has(3)), "cycle split across shards");
        }
    }

    #[test]
    fn graph_features_reports_edges_and_hubs() {
        let cg = graph();
        let s = graph_features(&cg, None);
        assert!(s.contains("internal call edges: 3"));
        assert!(s.contains("hub functions:"));
        assert!(s.contains("cycle:"));
        let scoped = graph_features(&cg, Some(&[1, 2]));
        assert!(scoped.contains("functions: 2"));
    }

    #[test]
    fn function_appears_in_exactly_one_shard() {
        let cg = graph();
        let shards = pack_shards(&cg, 5); // tiny budget forces multiple shards
        let mut seen = std::collections::BTreeMap::new();
        for s in &shards {
            for id in &s.function_ids {
                *seen.entry(*id).or_insert(0) += 1;
            }
        }
        assert!(seen.values().all(|c| *c == 1), "a function was split across shards");
        assert_eq!(seen.len(), 4);
    }

    #[test]
    fn materialize_writes_sequential_files_and_manifest() {
        let cg = graph();
        let dir = std::env::temp_dir().join(format!("knowdit_shard_test_{}", std::process::id()));
        let man = materialize_shards(&cg, 1000, &dir).unwrap();
        assert!(!man.is_empty());
        assert!(dir.join("manifest.json").exists());
        assert!(dir.join("shard-0000.sol").exists());
        let body = std::fs::read_to_string(dir.join("shard-0000.sol")).unwrap();
        assert!(body.contains("### "));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn anchored_functions_must_share_one_group() {
        // contract1: A<->B cycle + C singleton; contract2: D singleton.
        let mut cg = graph();
        cg.contracts.get_mut(&1).unwrap().functions.push(func(5, "C", vec![]));
        // A(id1) in cycle group, D(id4) in singleton group -> must reject.
        assert!(!functions_anchored(&cg, &[("A".into(), "a.sol".into()), ("D".into(), "b.sol".into())]));
        assert!(functions_anchored(&cg, &[("A".into(), "a.sol".into()), ("B".into(), "a.sol".into())]));
    }

    #[test]
    fn functions_anchored_co_shard() {
        let cg = graph();
        assert!(functions_anchored(&cg, &[("A".into(), "a.sol".into()), ("B".into(), "a.sol".into())]));
        assert!(!functions_anchored(&cg, &[("A".into(), "a.sol".into()), ("D".into(), "b.sol".into())]));
        assert!(!functions_anchored(&cg, &[("Z".into(), "a.sol".into())]));
    }

    #[test]
    fn anchoring_gate() {
        let cg = graph();
        assert!(anchored(&cg, &[1, 2], 1));
        assert!(!anchored(&cg, &[1, 4], 1));
        assert!(!anchored(&cg, &[1], 99));
    }
}