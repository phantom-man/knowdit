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

/// Resolve an (contract filter, symbol) pair to exactly one function.
/// Two rules from audit 20261124-072027:
/// - F18: the reverse path match must be SLASH-BOUNDED. `cn.ends_with(&cp)`
///   made "src/MyToken.sol" match a different contract "Token.sol".
/// - F17: overloaded / same-named functions are ambiguous. Returning the
///   first `find` result let the agent ground on the wrong implementation.
fn find_function<'a>(
    cg: &'a CallGraph,
    contract: &str,
    symbol: &str,
) -> std::result::Result<
    (
        &'a knowdit_repo_model::cg::Contract,
        &'a knowdit_repo_model::cg::Function,
    ),
    String,
> {
    let cn = contract.replace('\\', "/");
    let mut found: Vec<(
        &knowdit_repo_model::cg::Contract,
        &knowdit_repo_model::cg::Function,
    )> = Vec::new();
    for c in cg.contracts.values() {
        let cp = c.relative_file_path.to_string_lossy().replace('\\', "/");
        let reverse_match = !cp.is_empty() && cn.ends_with(&format!("/{}", cp));
        let cmatch = contract.is_empty()
            || c.name == contract
            || cp == cn
            || (!cn.is_empty() && cp.ends_with(&format!("/{}", cn)))
            || reverse_match;
        if !cmatch {
            continue;
        }
        for f in c.functions.iter().filter(|f| f.name == symbol) {
            found.push((c, f));
        }
    }
    match found.len() {
        0 => Err(format!("symbol {} not found in {}", symbol, contract)),
        1 => Ok(found[0]),
        n => {
            let sites: Vec<String> = found
                .iter()
                .map(|(c, f)| {
                    format!(
                        "{}:{} (args={}, file={})",
                        c.name,
                        f.name,
                        f.args,
                        c.relative_file_path.to_string_lossy()
                    )
                })
                .collect();
            Err(format!(
                "symbol {} is AMBIGUOUS ({} matches): {}",
                symbol,
                n,
                sites.join(" | ")
            ))
        }
    }
}

/// Resolve a bare function name to exactly one id, or an error string naming
/// the ambiguity. Used by trace_call_graph (F15) and check_h1_loops (F16).
fn unique_function_id(cg: &CallGraph, name: &str) -> std::result::Result<i32, String> {
    let ids: Vec<i32> = cg
        .contracts
        .values()
        .flat_map(|c| c.functions.iter().filter(|f| f.name == name).map(|f| f.id))
        .collect();
    match ids.len() {
        0 => Err(format!("function {} not found", name)),
        1 => Ok(ids[0]),
        n => Err(format!(
            "function {} is AMBIGUOUS ({} overloads/contracts share this name); pass a unique name",
            name, n
        )),
    }
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
    cg.contracts
        .values()
        .find_map(|c| {
            c.functions
                .iter()
                .find(|f| f.id == id)
                .map(|f| f.name.clone())
        })
        .unwrap_or_default()
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
pub struct InspectSymbolTool {
    pub cg: Arc<CallGraph>,
}
impl InspectSymbolTool {
    async fn invoke(&self, args: InspectSymbolArgs) -> std::result::Result<String, LLMYError> {
        match find_function(&self.cg, &args.contract, &args.symbol) {
            Ok((c, f)) => Ok(serde_json::json!({
                "contract": c.name, "file": c.relative_file_path.to_string_lossy(),
                "name": f.name, "args": f.args, "content": f.content,
            })
            .to_string()),
            Err(e) => Ok(format!("error: {}", e)),
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
pub struct TraceCallGraphTool {
    pub cg: Arc<CallGraph>,
}
impl TraceCallGraphTool {
    async fn invoke(&self, args: TraceCallGraphArgs) -> std::result::Result<String, LLMYError> {
        // F14: a near-miss direction ("callers", "Upstream", "upstream ")
        // silently selected downstream and echoed the bad value back.
        let direction = args.direction.trim();
        if direction != "upstream" && direction != "downstream" {
            return Ok(format!(
                "error: direction must be 'upstream' or 'downstream', got '{}'",
                args.direction
            ));
        }
        // F15: a bare name shared by overloads/contracts must be rejected,
        // not resolved by iteration order.
        let start = match unique_function_id(&self.cg, &args.function) {
            Ok(id) => id,
            Err(e) => return Ok(format!("error: {}", e)),
        };
        let adj = adjacency(&self.cg);
        let rev = {
            let mut r: BTreeMap<i32, Vec<i32>> = BTreeMap::new();
            for (a, bs) in &adj {
                for b in bs {
                    r.entry(*b).or_default().push(*a);
                }
            }
            r
        };
        let g = if direction == "upstream" { &rev } else { &adj };
        let mut seen: BTreeSet<i32> = BTreeSet::new();
        let mut q: VecDeque<(i32, usize)> = VecDeque::new();
        let mut edges: Vec<(String, String)> = Vec::new();
        q.push_back((start, 0));
        seen.insert(start);
        while let Some((n, d)) = q.pop_front() {
            if d >= args.depth {
                continue;
            }
            for m in g.get(&n).unwrap_or(&Vec::new()) {
                edges.push((fname(&self.cg, n), fname(&self.cg, *m)));
                if seen.insert(*m) {
                    q.push_back((*m, d + 1));
                }
            }
        }
        Ok(
            serde_json::json!({ "start": args.function, "direction": direction, "edges": edges })
                .to_string(),
        )
    }
}

// ---- 3. check_h1_loops ----
#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct CheckH1Args {
    pub function: String,
}
#[derive(Debug, Clone)]
#[tool(arguments = CheckH1Args, invoke = invoke, name = "check_h1_loops",
    description = "Return the closed state-mutating call cycle (H1/SCC signature) a function participates in, so you extract the end-to-end mechanism rather than isolated helpers.")]
pub struct CheckH1LoopsTool {
    pub cg: Arc<CallGraph>,
}
impl CheckH1LoopsTool {
    async fn invoke(&self, args: CheckH1Args) -> std::result::Result<String, LLMYError> {
        let id = match unique_function_id(&self.cg, &args.function) {
            Ok(id) => id,
            Err(e) => return Ok(format!("error: {}", e)),
        };
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
pub struct AnchorArg {
    pub name: String,
    pub contract: String,
}
#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct ValidateSemanticArgs {
    pub name: String,
    pub category: String,
    pub functions: Vec<AnchorArg>,
}
#[derive(Debug, Clone)]
#[tool(arguments = ValidateSemanticArgs, invoke = invoke, name = "validate_semantic",
    description = "Self-correction gate BEFORE emit: checks every cited function exists and is co-located in one call-graph shard (anchoring) and passes the hyperbolic radial/geodesic cone. Returns pass/fail + reason.")]
pub struct ValidateSemanticTool {
    pub cg: Arc<CallGraph>,
}
impl ValidateSemanticTool {
    async fn invoke(&self, args: ValidateSemanticArgs) -> std::result::Result<String, LLMYError> {
        let anchors: Vec<(String, String)> = args
            .functions
            .iter()
            .map(|a| (a.name.clone(), a.contract.clone()))
            .collect();
        let anchored = shard_pack::functions_anchored(&self.cg, &anchors);
        let hyper = hyperbolic::hyperbolic_anchored(&self.cg, &anchors);
        let pass = anchored && hyper;
        let reason = if !anchored {
            "anchors unresolved or span multiple shards"
        } else if !hyper {
            "fails hyperbolic radial/geodesic cone"
        } else {
            "ok"
        };
        Ok(serde_json::json!({ "pass": pass, "anchored": anchored, "hyperbolic": hyper, "reason": reason }).to_string())
    }
}

// ---- 5. get_storage_dependencies ----
#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct StorageDepArgs {
    pub state_var: String,
}
#[derive(Debug, Clone)]
#[tool(arguments = StorageDepArgs, invoke = invoke, name = "get_storage_dependencies",
    description = "List every function that reads/writes a storage slot, to group coupled functions (deposit/withdraw sharing totalAssets) into one mechanism.")]
pub struct StorageDepTool {
    pub deps: Arc<BTreeMap<String, Vec<String>>>,
}
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
pub struct LookupKgArgs {
    pub contract: String,
}
#[derive(Debug, Clone)]
#[tool(arguments = LookupKgArgs, invoke = invoke, name = "lookup_existing_kg_nodes",
    description = "Query already-emitted semantic nodes for a contract (adjacent shards) to reuse canonical mechanisms instead of creating duplicates.")]
pub struct LookupKgTool {
    pub existing: Arc<std::sync::Mutex<Vec<(String, String)>>>,
}
impl LookupKgTool {
    async fn invoke(&self, args: LookupKgArgs) -> std::result::Result<String, LLMYError> {
        let guard = self.existing.lock().unwrap();
        let hits: Vec<&(String, String)> = guard
            .iter()
            .filter(|(c, _)| c == &args.contract || c.ends_with(&args.contract))
            .collect();
        Ok(serde_json::json!({ "contract": args.contract, "existing": hits }).to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use knowdit_repo_model::cg::{
        CallGraph, Contract, FileChunk, FileLocation, Function, FunctionCall,
    };
    use std::collections::BTreeMap;
    use std::path::PathBuf;

    fn loc() -> FileLocation {
        FileLocation {
            start_line: 1,
            start_column: 0,
            end_line: 2,
            end_column: 0,
        }
    }
    fn func(id: i32, name: &str, args: &str) -> Function {
        Function {
            id,
            name: name.into(),
            args: args.into(),
            relative_file_path: PathBuf::from("Token.sol"),
            loc: loc(),
            content: Some("x".repeat(20)),
            calls: Vec::<FunctionCall>::new(),
            description: None,
        }
    }
    fn contract(id: i32, name: &str, path: &str, funcs: Vec<Function>) -> Contract {
        Contract {
            id,
            name: name.into(),
            relative_file_path: PathBuf::from(path),
            chunk: FileChunk {
                loc: loc(),
                content: String::new(),
            },
            functions: funcs,
            description: None,
        }
    }
    fn graph() -> CallGraph {
        let mut cg = CallGraph {
            contracts: BTreeMap::new(),
            interfaces: BTreeMap::new(),
        };
        cg.contracts.insert(
            1,
            contract(1, "Token", "src/Token.sol", vec![func(1, "withdraw", "()")]),
        );
        cg.contracts.insert(
            2,
            contract(
                2,
                "MyToken",
                "src/MyToken.sol",
                vec![func(2, "withdraw", "(uint256)")],
            ),
        );
        cg
    }

    // F17: two same-named functions must be reported ambiguous, not first.
    #[test]
    fn overloaded_symbol_is_ambiguous() {
        let cg = graph();
        let r = find_function(&cg, "", "withdraw");
        assert!(r.is_err(), "duplicate name must be ambiguous");
        assert!(r.unwrap_err().contains("AMBIGUOUS"));
    }

    // F17: a unique symbol still resolves.
    #[test]
    fn unique_symbol_resolves() {
        let mut cg = graph();
        cg.contracts
            .get_mut(&1)
            .unwrap()
            .functions
            .push(func(3, "deposit", "()"));
        let (c, f) = find_function(&cg, "", "deposit").expect("unique");
        assert_eq!(c.name, "Token");
        assert_eq!(f.name, "deposit");
    }

    // F18: "src/MyToken.sol" must NOT match a different contract "Token.sol"
    // via the old unbounded reverse suffix check.
    #[test]
    fn reverse_path_match_is_slash_bounded() {
        let mut cg = graph();
        // Give contract 2 a SHORT stored path "Token.sol" and request "src/MyToken.sol".
        // A different symbol avoids F17 ambiguity so only F18 is under test.
        cg.contracts.get_mut(&2).unwrap().relative_file_path = PathBuf::from("Token.sol");
        cg.contracts.get_mut(&2).unwrap().functions = vec![func(2, "onlyHere", "()")];
        assert!(
            find_function(&cg, "src/MyToken.sol", "onlyHere").is_err(),
            "src/MyToken.sol must not match stored Token.sol without a slash boundary"
        );
    }

    // F15/F16: bare name shared by two contracts is ambiguous in both resolvers.
    #[test]
    fn unique_function_id_rejects_duplicates() {
        let cg = graph();
        assert!(unique_function_id(&cg, "withdraw").is_err());
        assert_eq!(unique_function_id(&cg, "missing").is_err(), true);
        let mut cg2 = graph();
        cg2.contracts.get_mut(&1).unwrap().functions = vec![func(1, "uniqueFn", "()")];
        assert_eq!(unique_function_id(&cg2, "uniqueFn").unwrap(), 1);
    }
}
