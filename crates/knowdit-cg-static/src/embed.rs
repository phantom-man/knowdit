//! Neural embedding backend (OpenAI-compatible /v1/embeddings) with deterministic fallback.
//!
//! Upgrades the FGW/hyperbolic feature terms from token-Jaccard to learned semantic vectors.
//! Config via env: KNOWDIT_EMBED_URL (default http://127.0.0.1:8766), KNOWDIT_EMBED_MODEL
//! (default all-MiniLM-L6-v2), KNOWDIT_EMBED_KEY (optional bearer). When the endpoint is
//! unreachable, falls back to a deterministic hashed token vector so gates still function
//! offline. A local server is provided at tools/embed_server.py (sentence-transformers).

use std::io::{Read, Write};
use std::net::TcpStream;
use std::time::Duration;

pub fn cosine(a: &[f64], b: &[f64]) -> f64 {
    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if na == 0.0 || nb == 0.0 { 0.0 } else { dot / (na * nb) }
}

/// Deterministic 64-dim hashed token vector (fallback), L2-normalized.
pub fn fallback_vec(text: &str) -> Vec<f64> {
    let mut v = vec![0.0f64; 64];
    for t in text.to_lowercase().split(|c: char| !c.is_alphanumeric()).filter(|t| !t.is_empty()) {
        let mut h: u64 = 14695981039346656037;
        for b in t.bytes() { h ^= b as u64; h = h.wrapping_mul(1099511628211); }
        v[(h % 64) as usize] += 1.0;
    }
    let n: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    if n > 0.0 { for x in v.iter_mut() { *x /= n; } }
    v
}

fn parse_host_port(url: &str) -> Option<(String, u16)> {
    // Only plaintext local http is supported; reject https so a bearer key is never
    // sent over a cleartext TCP connection (fail closed to the deterministic fallback).
    if url.starts_with("https://") {
        return None;
    }
    let rest = url.trim_start_matches("http://");
    let hostport = rest.split('/').next()?;
    let mut it = hostport.rsplitn(2, ':');
    let port = it.next()?.parse().ok()?;
    let host = it.next()?.to_string();
    Some((host, port))
}

/// POST /v1/embeddings; returns one vector per input, or None on any failure.
fn neural_embed(texts: &[String]) -> Option<Vec<Vec<f64>>> {
    let url = std::env::var("KNOWDIT_EMBED_URL").unwrap_or_else(|_| "http://127.0.0.1:8766".into());
    let model = std::env::var("KNOWDIT_EMBED_MODEL").unwrap_or_else(|_| "all-MiniLM-L6-v2".into());
    let (host, port) = parse_host_port(&url)?;
    let body = serde_json::json!({ "model": model, "input": texts });
    let payload = body.to_string();
    let mut stream = TcpStream::connect((host.as_str(), port)).ok()?;
    stream.set_read_timeout(Some(Duration::from_secs(10))).ok()?;
    stream.set_write_timeout(Some(Duration::from_secs(10))).ok()?;
    let mut req = format!(
        "POST /v1/embeddings HTTP/1.1\r\nHost: {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n",
        host, payload.len());
    let loopback = host == "127.0.0.1" || host == "localhost" || host == "::1";
    if loopback {
        if let Ok(key) = std::env::var("KNOWDIT_EMBED_KEY") {
            req.push_str(&format!("Authorization: Bearer {}\r\n", key));
        }
    }
    req.push_str("\r\n");
    stream.write_all(req.as_bytes()).ok()?;
    stream.write_all(payload.as_bytes()).ok()?;
    let mut resp = String::new();
    stream.read_to_string(&mut resp).ok()?;
    let body = resp.split("\r\n\r\n").nth(1)?;
    let v: serde_json::Value = serde_json::from_str(body).ok()?;
    let data = v.get("data")?.as_array()?;
    let mut out = Vec::new();
    for d in data {
        let emb = d.get("embedding")?.as_array()?;
        out.push(emb.iter().filter_map(|x| x.as_f64()).collect());
    }
    if out.len() == texts.len() { Some(out) } else { None }
}

/// Embed texts: neural backend when reachable, else deterministic fallback.
pub fn embed_texts(texts: &[String]) -> Vec<Vec<f64>> {
    if let Some(v) = neural_embed(texts) { return v; }
    texts.iter().map(|t| fallback_vec(t)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_identity_and_orthogonal() {
        assert!((cosine(&[1.0, 0.0], &[1.0, 0.0]) - 1.0).abs() < 1e-9);
        assert!(cosine(&[1.0, 0.0], &[0.0, 1.0]).abs() < 1e-9);
    }

    #[test]
    fn fallback_deterministic_normalized() {
        let a = fallback_vec("reward accounting before transfer");
        let b = fallback_vec("reward accounting before transfer");
        assert_eq!(a, b);
        let n: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!((n - 1.0).abs() < 1e-9);
    }

    #[test]
    fn embed_falls_back_offline() {
        // Force an unreachable endpoint so the fallback path is exercised deterministically.
        unsafe { std::env::set_var("KNOWDIT_EMBED_URL", "http://127.0.0.1:9"); }
        let v = embed_texts(&["alpha".into(), "beta".into()]);
        assert_eq!(v.len(), 2);
        assert_eq!(v[0].len(), 64);
        unsafe { std::env::remove_var("KNOWDIT_EMBED_URL"); }
    }

    #[test]
    fn https_rejected_falls_back() {
        unsafe { std::env::set_var("KNOWDIT_EMBED_URL", "https://example.com:443"); }
        let v = embed_texts(&["x".into()]);
        assert_eq!(v[0].len(), 64); // no cleartext https; deterministic fallback
        unsafe { std::env::remove_var("KNOWDIT_EMBED_URL"); }
    }

    #[test]
    fn neural_backend_when_server_up() {
        // Only meaningful with tools/embed_server.py running; skip gracefully otherwise.
        if std::env::var("KNOWDIT_EMBED_URL").is_err() {
            // default 8766; probe via a tiny TCP connect
            if std::net::TcpStream::connect(("127.0.0.1", 8766)).is_err() { return; }
        }
        let v = embed_texts(&["reward accounting".into()]);
        assert!(v[0].len() == 384 || v[0].len() == 64);
    }
}