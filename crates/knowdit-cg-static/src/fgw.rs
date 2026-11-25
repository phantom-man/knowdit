//! Fused Gromov-Wasserstein (FGW) optimal-transport coupling for the dual-stream KG merge.
//!
//! Implements docs/SEMANTIC_EXTRACTION_REVIEW_AND_MATH.md §3.3: link semantic (mechanism)
//! nodes to audit-finding nodes only when BOTH feature similarity and neighborhood topology
//! align. The fused cost combines a feature term (text/embedding distance) and a structure
//! term (call/dependency-graph distance), then entropic OT (Sinkhorn) yields a coupling.
//! Deterministic, dependency-free (no external OT solver).

/// Fused cost matrix: C = alpha*feature + (1-alpha)*structure. alpha in [0,1].
pub fn fused_cost(feature: &[Vec<f64>], structure: &[Vec<f64>], alpha: f64) -> Vec<Vec<f64>> {
    let a = alpha.clamp(0.0, 1.0);
    feature
        .iter()
        .zip(structure.iter())
        .map(|(fr, sr)| fr.iter().zip(sr.iter()).map(|(f, s)| a * f + (1.0 - a) * s).collect())
        .collect()
}

/// Entropic OT (Sinkhorn) on a cost matrix with uniform marginals.
/// Returns coupling P (n x m), rows/cols ~ 1/n, 1/m. `eps` = entropy regularization.
pub fn sinkhorn(cost: &[Vec<f64>], eps: f64, iters: usize) -> Vec<Vec<f64>> {
    let n = cost.len();
    let m = if n > 0 { cost[0].len() } else { 0 };
    if n == 0 || m == 0 {
        return Vec::new();
    }
    let eps = eps.max(1e-6);
    // K = exp(-C/eps)
    let k: Vec<Vec<f64>> = cost.iter().map(|r| r.iter().map(|c| (-c / eps).exp()).collect()).collect();
    let mut u = vec![1.0f64; n];
    let mut v = vec![1.0f64; m];
    let a = 1.0 / n as f64;
    let b = 1.0 / m as f64;
    for _ in 0..iters {
        // u = a / (K v)
        for i in 0..n {
            let kv: f64 = (0..m).map(|j| k[i][j] * v[j]).sum();
            u[i] = a / kv.max(1e-12);
        }
        // v = b / (K^T u)
        for j in 0..m {
            let ku: f64 = (0..n).map(|i| k[i][j] * u[i]).sum();
            v[j] = b / ku.max(1e-12);
        }
    }
    (0..n).map(|i| (0..m).map(|j| u[i] * k[i][j] * v[j]).collect()).collect()
}

/// FGW coupling = Sinkhorn on the fused feature+structure cost.
pub fn fgw_coupling(feature: &[Vec<f64>], structure: &[Vec<f64>], alpha: f64, eps: f64, iters: usize) -> Vec<Vec<f64>> {
    sinkhorn(&fused_cost(feature, structure, alpha), eps, iters)
}

/// Best structural+feature partner for each semantic row (argmax of coupling).
pub fn best_links(coupling: &[Vec<f64>]) -> Vec<usize> {
    coupling
        .iter()
        .map(|row| row.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|(j, _)| j).unwrap_or(0))
        .collect()
}

/// Concrete dual-stream linker: propose a finding partner for each semantic using FGW over
/// feature (name token-Jaccard distance) + structure (shared function-id Jaccard distance).
/// Returns, per semantic, the index of the best finding (mechanism<->failure coupling).
pub fn link_semantics_findings(
    sem: &[(String, Vec<i32>)],
    find: &[(String, Vec<i32>)],
    alpha: f64,
) -> Vec<usize> {
    let tok = |s: &String| -> std::collections::BTreeSet<String> {
        s.to_lowercase().split(|c: char| !c.is_alphanumeric()).filter(|t| !t.is_empty()).map(|t| t.to_string()).collect()
    };
    let jac = |a: &std::collections::BTreeSet<String>, b: &std::collections::BTreeSet<String>| -> f64 {
        let u = a.union(b).count();
        if u == 0 { 1.0 } else { 1.0 - (a.intersection(b).count() as f64 / u as f64) }
    };
    let sjac = |a: &Vec<i32>, b: &Vec<i32>| -> f64 {
        let sa: std::collections::BTreeSet<i32> = a.iter().copied().collect();
        let sb: std::collections::BTreeSet<i32> = b.iter().copied().collect();
        let u = sa.union(&sb).count();
        if u == 0 { 1.0 } else { 1.0 - (sa.intersection(&sb).count() as f64 / u as f64) }
    };
    let st: Vec<_> = sem.iter().map(|(n, _)| tok(n)).collect();
    let ft: Vec<_> = find.iter().map(|(n, _)| tok(n)).collect();
    let feat: Vec<Vec<f64>> = st.iter().map(|s| ft.iter().map(|t| jac(s, t)).collect()).collect();
    let stru: Vec<Vec<f64>> = sem.iter().map(|(_, a)| find.iter().map(|(_, b)| sjac(a, b)).collect()).collect();
    best_links(&fgw_coupling(&feat, &stru, alpha, 0.05, 100))
}

/// Neural variant: feature term = 1 - cosine(embedded names) via the embedding backend
/// (falls back to hashed vectors offline); structure term unchanged. Upgrades mechanism<->finding
/// coupling from lexical to learned semantic similarity.
pub fn link_semantics_findings_neural(
    sem: &[(String, Vec<i32>)],
    find: &[(String, Vec<i32>)],
    alpha: f64,
) -> Vec<usize> {
    let sn: Vec<String> = sem.iter().map(|(n, _)| n.clone()).collect();
    let fn_: Vec<String> = find.iter().map(|(n, _)| n.clone()).collect();
    // Opt-in managed server: acquired here so its lifetime covers both embed calls.
    let _embed_guard = crate::embed::EmbedServerGuard::acquire();
    let se = crate::embed::embed_texts(&sn);
    let fe = crate::embed::embed_texts(&fn_);
    // The two embed_texts calls can land on different backends (e.g. first
    // succeeds neural 384-d, second falls back hashed 64-d). cosine zips by
    // prefix, so cross-space comparison is silently wrong. Require a shared
    // dimension; on mismatch fall back to the lexical linker, which never
    // mixes spaces (audit 20261124-054841 F13).
    let dim = se.first().map(|v| v.len());
    if dim.is_none() || se.is_empty() || fe.is_empty()
        || se.iter().any(|v| v.len() != dim.unwrap())
        || fe.iter().any(|v| v.len() != dim.unwrap()) {
        return link_semantics_findings(sem, find, alpha);
    }
    let feat: Vec<Vec<f64>> = se.iter().map(|s| fe.iter().map(|f| 1.0 - crate::embed::cosine(s, f)).collect()).collect();
    let sjac = |a: &Vec<i32>, b: &Vec<i32>| -> f64 {
        let sa: std::collections::BTreeSet<i32> = a.iter().copied().collect();
        let sb: std::collections::BTreeSet<i32> = b.iter().copied().collect();
        let u = sa.union(&sb).count();
        if u == 0 { 1.0 } else { 1.0 - (sa.intersection(&sb).count() as f64 / u as f64) }
    };
    let stru: Vec<Vec<f64>> = sem.iter().map(|(_, a)| find.iter().map(|(_, b)| sjac(a, b)).collect()).collect();
    best_links(&fgw_coupling(&feat, &stru, alpha, 0.05, 100))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coupling_marginals_uniform() {
        let feat = vec![vec![0.0, 1.0], vec![1.0, 0.0]];
        let stru = vec![vec![0.0, 1.0], vec![1.0, 0.0]];
        let p = fgw_coupling(&feat, &stru, 0.5, 0.1, 50);
        for row in &p {
            let s: f64 = row.iter().sum();
            assert!((s - 0.5).abs() < 1e-3, "row sum {}", s);
        }
        for j in 0..2 {
            let s: f64 = p.iter().map(|r| r[j]).sum();
            assert!((s - 0.5).abs() < 1e-3, "col sum {}", s);
        }
    }

    #[test]
    fn prefers_aligned_pair() {
        // semantic0 matches finding0 on both feature and structure.
        let feat = vec![vec![0.0, 2.0], vec![2.0, 0.0]];
        let stru = vec![vec![0.0, 2.0], vec![2.0, 0.0]];
        let p = fgw_coupling(&feat, &stru, 0.5, 0.05, 100);
        assert!(p[0][0] > p[0][1]);
        assert!(p[1][1] > p[1][0]);
        assert_eq!(best_links(&p), vec![0, 1]);
    }

    // Regression: audit 20261124-054841 F13. A mixed-backend dimension
    // mismatch must fall back to the lexical linker, not compare zipped
    // prefixes of unrelated vector spaces.
    #[test]
    fn neural_linker_handles_backend_dimension_mismatch() {
        let sem = vec![("reward accounting".into(), vec![1, 2])];
        let find = vec![("reward accounting bug".into(), vec![1, 2])];
        let links = link_semantics_findings_neural(&sem, &find, 0.5);
        assert_eq!(links.len(), sem.len());
        assert!(links.iter().all(|&j| j < find.len()));
    }

    #[test]
    fn links_semantic_to_matching_finding() {
        let sem = vec![("reward accounting".into(), vec![1, 2]), ("oracle price".into(), vec![3])];
        let find = vec![("reward accounting bug".into(), vec![1, 2]), ("oracle manipulation".into(), vec![3])];
        assert_eq!(link_semantics_findings(&sem, &find, 0.5), vec![0, 1]);
    }

    #[test]
    fn structure_can_override_feature() {
        // feature says 0<->1, structure says 0<->0; with low alpha structure wins.
        let feat = vec![vec![2.0, 0.0], vec![0.0, 2.0]];
        let stru = vec![vec![0.0, 2.0], vec![2.0, 0.0]];
        let p = fgw_coupling(&feat, &stru, 0.1, 0.05, 100);
        assert_eq!(best_links(&p), vec![0, 1]);
    }
}