//! Proactive deterministic tools for the extraction agent (docs: graph-native tooling).
//!
//! Gives the agent on-demand structural grounding, cross-shard graph context, and
//! self-correction gates INSTEAD of guessing cross-file context from a 30k-token shard.
//! Backed by the pre-computed static CallGraph + shard_pack + hyperbolic gates.
//! Tools return JSON strings; validation tools return pass/fail + reason so the agent
//! can self-correct in-loop before emit_semantic is accepted.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::sync::Arc;

use knowdit_repo_model::cg::CallGraph;
use llmy::agent::{LLMYError, tool};
use schemars::JsonSchema;
use serde::Deserialize;

use knowdit_cg_static::{hyperbolic, shard_pack};

fn find_function<'a>(cg: &'a CallGraph, contract: &str, symbol: &str) -> Option<(&'a knowdit_repo_model::cg::Contract, &'a knowdit_repo_model::cg::Function)> {
    let cn = contract.replace('\\', "/");
    for c in cg.contracts.values() {
        let cp = c.relative_file_path.to_string_lossy().replace('\\', "/");
        let cmatch = contract.is_empty() || c.name == contract || cp == cn || cp.ends_with(&format!("/{}", cn)) || cn.ends_with(&cp);
        if !cmatch { continue; }
        if let Some(f) = c.functions.iter().find(|f| f.name == symbol) {
            return Some((c, f));
        }
    }
    None
}

fn adjacency(cg: &CallGraph) -> BTreeMap<i32, Vec<i32>> {
    let mut adj: BTreeMap<i32, Vec<i32>> = BTreeMap::new();
    for c in cg.contracts.values() {
        for f in &c.functions {
            adj.entry(f.id).or_default();
            for call in &f.calls {
                adj.entry(f.id).or_default().push(call.to_id);
                adj.entry(call.to_id).or_default();
            }
        }
    }
    adj
}

fn fname(cg: &CallGraph, id: i32) -> String {
    cg.contracts.values().find_map(|c| c.functions.iter().find(|f| f.id == id).map(|f| f.name.clone())).unwrap_or_default()
}

// ---- 1. inspect_symbol ----
#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct InspectSymbolArgs {
    /// Contract name or file path (empty = search all).
    pub contract: String,
    /// Function / symbol name.
    pub symbol: String,
}
#[derive(Debug, Clone)]
#[tool(arguments = InspectSymbolArgs, invoke = invoke, name = "inspect_symbol",
    description = "Return the exact definition, parameter list and source body of a function/contract, even if outside the current shard. Use instead of guessing cross-file behavior.")]
pub struct InspectSymbolTool { pub cg: Arc<CallGraph> }
impl InspectSymbolTool {
    async fn invoke(&self, args: InspectSymbolArgs) -> std::result::Result<String, LLMYError> {
        match find_function(&self.cg, &args.contract, &args.symbol) {
            Some((c, f)) => Ok(serde_json::json!({
                "contract": c.name, "file": c.relative_file_path.to_string_lossy(),
                "name": f.name, "args": f.args, "content": f.content,
            }).to_string()),
            None => Ok(format!("error: symbol {} not found in {}", args.symbol, args.contract)),
        }
    }
}

// ---- 2. trace_call_graph ----
#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct TraceCallGraphArgs {
    /// Function name to start from.
    pub function: String,
    /// "downstream" (callees) or "upstream" (callers).
    pub direction: String,
    /// Max traversal depth.
    pub depth: usize,
}
#[derive(Debug, Clone)]
#[tool(arguments = TraceCallGraphArgs, invoke = invoke, name = "trace_call_graph",
    description = "Return the exact static call cone (upstream/downstream, bounded depth) with call edges, so you see the true multi-contract execution loop before categorizing.")]
pub struct TraceCallGraphTool { pub cg: Arc<CallGraph> }
impl TraceCallGraphTool {
    async fn invoke(&self, args: TraceCallGraphArgs) -> std::result::Result<String, LLMYError> {
        let start = self.cg.contracts.values().find_map(|c| c.functions.iter().find(|f| f.name == args.function).map(|f| f.id));
        let Some(start) = start else { return Ok(format!("error: function {} not found", args.function)); };
        let adj = adjacency(&self.cg);
        let rev = {
            let mut r: BTreeMap<i32, Vec<i32>> = BTreeMap::new();
            for (a, bs) in &adj { for b in bs { r.entry(*b).or_default().push(*a); } }
            r
        };
        let g = if args.direction == "upstream" { &rev } else { &adj };
        let mut seen: BTreeSet<i32> = BTreeSet::new();
        let mut q: VecDeque<(i32, usize)> = VecDeque::new();
        let mut edges: Vec<(String, String)> = Vec::new();
        q.push_back((start, 0));
        seen.insert(start);
        while let Some((n, d)) = q.pop_front() {
            if d >= args.depth { continue; }
            for m in g.get(&n).unwrap_or(&Vec::new()) {
                edges.push((fname(&self.cg, n), fname(&self.cg, *m)));
                if seen.insert(*m) { q.push_back((*m, d + 1)); }
            }
        }
        Ok(serde_json::json!({ "start": args.function, "direction": args.direction, "edges": edges }).to_string())
    }
}

// ---- 3. check_h1_loops ----
#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct CheckH1Args { pub function: String }
#[derive(Debug, Clone)]
#[tool(arguments = CheckH1Args, invoke = invoke, name = "check_h1_loops",
    description = "Return the closed state-mutating call cycle (H1/SCC signature) a function participates in, so you extract the end-to-end mechanism rather than isolated helpers.")]
pub struct CheckH1LoopsTool { pub cg: Arc<CallGraph> }
impl CheckH1LoopsTool {
    async fn invoke(&self, args: CheckH1Args) -> std::result::Result<String, LLMYError> {
        let id = self.cg.contracts.values().find_map(|c| c.functions.iter().find(|f| f.name == args.function).map(|f| f.id));
        let Some(id) = id else { return Ok(format!("error: function {} not found", args.function)); };
        for g in shard_pack::cycle_groups(&self.cg) {
            if g.contains(&id) {
                return Ok(serde_json::json!({ "function": args.function, "cycle": shard_pack::cycle_signature(&self.cg, &g) }).to_string());
            }
        }
        Ok(serde_json::json!({ "function": args.function, "cycle": null }).to_string())
    }
}

// ---- 4. validate_semantic ----
#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct AnchorArg { pub name: String, pub contract: String }
#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct ValidateSemanticArgs {
    pub name: String,
    pub category: String,
    pub functions: Vec<AnchorArg>,
}
#[derive(Debug, Clone)]
#[tool(arguments = ValidateSemanticArgs, invoke = invoke, name = "validate_semantic",
    description = "Self-correction gate BEFORE emit: checks every cited function exists and is co-located in one call-graph shard (anchoring) and passes the hyperbolic radial/geodesic cone. Returns pass/fail + reason.")]
pub struct ValidateSemanticTool { pub cg: Arc<CallGraph> }
impl ValidateSemanticTool {
    async fn invoke(&self, args: ValidateSemanticArgs) -> std::result::Result<String, LLMYError> {
        let anchors: Vec<(String, String)> = args.functions.iter().map(|a| (a.name.clone(), a.contract.clone())).collect();
        let anchored = shard_pack::functions_anchored(&self.cg, &anchors);
        let hyper = hyperbolic::hyperbolic_anchored(&self.cg, &anchors);
        let pass = anchored && hyper;
        let reason = if !anchored { "anchors unresolved or span multiple shards" } else if !hyper { "fails hyperbolic radial/geodesic cone" } else { "ok" };
        Ok(serde_json::json!({ "pass": pass, "anchored": anchored, "hyperbolic": hyper, "reason": reason }).to_string())
    }
}

// ---- 5. get_storage_dependencies ----
#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct StorageDepArgs { pub state_var: String }
#[derive(Debug, Clone)]
#[tool(arguments = StorageDepArgs, invoke = invoke, name = "get_storage_dependencies",
    description = "List every function that reads/writes a storage slot, to group coupled functions (deposit/withdraw sharing totalAssets) into one mechanism.")]
pub struct StorageDepTool { pub deps: Arc<BTreeMap<String, Vec<String>>> }
impl StorageDepTool {
    async fn invoke(&self, args: StorageDepArgs) -> std::result::Result<String, LLMYError> {
        match self.deps.get(&args.state_var) {
            Some(fns) => Ok(serde_json::json!({ "state_var": args.state_var, "functions": fns }).to_string()),
            None => Ok(serde_json::json!({ "state_var": args.state_var, "functions": [], "note": "storage analysis not loaded for this project" }).to_string()),
        }
    }
}

// ---- 6. lookup_existing_kg_nodes ----
#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct LookupKgArgs { pub contract: String }
#[derive(Debug, Clone)]
#[tool(arguments = LookupKgArgs, invoke = invoke, name = "lookup_existing_kg_nodes",
    description = "Query already-emitted semantic nodes for a contract (adjacent shards) to reuse canonical mechanisms instead of creating duplicates.")]
pub struct LookupKgTool { pub existing: Arc<std::sync::Mutex<Vec<(String, String)>>> }
impl LookupKgTool {
    async fn invoke(&self, args: LookupKgArgs) -> std::result::Result<String, LLMYError> {
        let guard = self.existing.lock().unwrap();
        let hits: Vec<&(String, String)> = guard.iter().filter(|(c, _)| c == &args.contract || c.ends_with(&args.contract)).collect();
        Ok(serde_json::json!({ "contract": args.contract, "existing": hits }).to_string())
    }
}