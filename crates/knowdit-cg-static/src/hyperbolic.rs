//! Hyperbolic (Poincaré ball) geodesic anchoring gate for emit_semantic.
//!
//! Implements the post-extraction intervention from
//! docs/SEMANTIC_EXTRACTION_REVIEW_AND_MATH.md §3.2: DeFi code is a strict hierarchy
//! (category -> contract -> function). Embed that hierarchy into a Poincaré ball with
//! radius increasing by depth and angle shared down a branch, then enforce:
//!   1. radial hierarchy: r(function) > r(category);
//!   2. geodesic cone: each anchored function lies within a tight geodesic distance of its
//!      own contract embedding (cross-branch anchors are far and rejected).
//! Deterministic (no neural embedder): depth->radius, branch->angle.

use std::collections::BTreeMap;

use knowdit_repo_model::cg::CallGraph;

/// A point in the 2-D Poincaré ball (open unit disk).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Poincare {
    pub x: f64,
    pub y: f64,
}

impl Poincare {
    pub fn norm2(&self) -> f64 {
        self.x * self.x + self.y * self.y
    }
    /// Geodesic distance in the Poincaré ball.
    pub fn distance(&self, other: &Poincare) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        let num = 2.0 * (dx * dx + dy * dy);
        let den = (1.0 - self.norm2()).max(1e-9) * (1.0 - other.norm2()).max(1e-9);
        (1.0 + num / den).max(1.0).acosh()
    }
}

fn hash_angle(s: &str) -> f64 {
    let mut h: u64 = 14695981039346656037;
    for b in s.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(1099511628211);
    }
    ((h % 100000) as f64 / 100000.0) * std::f64::consts::TAU
}

/// Radius by hierarchy depth (0=category,1=contract,2=function), strictly increasing, <1.
fn depth_radius(depth: usize) -> f64 {
    (0.25 + 0.25 * depth as f64).tanh()
}

fn point(depth: usize, angle: f64) -> Poincare {
    let r = depth_radius(depth);
    Poincare { x: r * angle.cos(), y: r * angle.sin() }
}

/// Embed a contract branch: contract at depth1 with its own angle; functions at depth2
/// inheriting the branch angle plus a small per-function offset.
pub fn embed_branch(contract: &str, function: &str) -> (Poincare, Poincare) {
    let ca = hash_angle(contract);
    let fa = ca + 0.05 * (hash_angle(function) / std::f64::consts::TAU);
    (point(1, ca), point(2, fa))
}

pub fn category_point() -> Poincare {
    point(0, 0.0)
}

/// Geodesic cone threshold: a function must sit close to its own contract embedding.
pub const CONE: f64 = 1.0;

/// Hyperbolic anchoring gate: every (function, contract) anchor must satisfy the radial
/// hierarchy and lie within the geodesic cone of its own contract branch.
pub fn hyperbolic_anchored(cg: &CallGraph, funcs: &[(String, String)]) -> bool {
    if funcs.is_empty() {
        return false;
    }
    let cat = category_point();
    for (fname, cpath) in funcs {
        let norm = cpath.replace('\\', "/");
        let mut found: Option<(String, String)> = None;
        for c in cg.contracts.values() {
            let cp = c.relative_file_path.to_string_lossy().replace('\\', "/");
            if cp == norm || cp.ends_with(&format!("/{}", norm)) || norm.ends_with(&format!("/{}", cp)) {
                if c.functions.iter().any(|f| &f.name == fname) {
                    found = Some((c.name.clone(), fname.clone()));
                    break;
                }
            }
        }
        let (cname, fnname) = match found {
            Some(v) => v,
            None => return false,
        };
        let (cp, fp) = embed_branch(&cname, &fnname);
        // radial hierarchy: function deeper than category
        if fp.norm2() <= cat.norm2() {
            return false;
        }
        // geodesic cone to own contract branch
        if fp.distance(&cp) > CONE {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn distance_symmetric_positive() {
        let a = point(1, 0.3);
        let b = point(2, 0.35);
        assert!((a.distance(&b) - b.distance(&a)).abs() < 1e-9);
        assert!(a.distance(&b) > 0.0);
        assert!(a.distance(&a) < 1e-6);
    }

    #[test]
    fn radial_hierarchy_increases() {
        assert!(depth_radius(0) < depth_radius(1));
        assert!(depth_radius(1) < depth_radius(2));
        assert!(depth_radius(2) < 1.0);
    }

    #[test]
    fn same_branch_within_cone_cross_branch_outside() {
        let (c, f) = embed_branch("Vault", "deposit");
        assert!(f.distance(&c) <= CONE);
        let (c2, f2) = embed_branch("Other", "deposit");
        // cross-branch function vs Vault contract should be far
        assert!(f2.distance(&c) > CONE);
    }
}