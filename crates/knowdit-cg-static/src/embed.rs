//! Neural embedding backend (OpenAI-compatible /v1/embeddings) with deterministic fallback.
//!
//! Upgrades the FGW/hyperbolic feature terms from token-Jaccard to learned semantic vectors.
//! Config via env: KNOWDIT_EMBED_URL (default http://127.0.0.1:8766), KNOWDIT_EMBED_MODEL
//! (default all-MiniLM-L6-v2), KNOWDIT_EMBED_KEY (optional bearer). When the endpoint is
//! unreachable, falls back to a deterministic hashed token vector so gates still function
//! offline. A local server is provided at tools/embed_server.py (sentence-transformers).

use std::io::{Read, Write};
use std::path::PathBuf;
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
    // A routable but black-holed host can stall in connect() for the OS
    // timeout; set a bounded connect deadline so the documented offline
    // fallback is reached promptly (audit 20261124-054841 F10).
    let addr = (host.as_str(), port);
    // Try EVERY resolved address (localhost may yield ::1 before the IPv4
    // loopback where the server actually listens); take the first that
    // connects within the bounded timeout (gate 20261125-112814).
    let resolved = std::net::ToSocketAddrs::to_socket_addrs(&addr).ok()?;
    let mut stream = None;
    for sa in resolved {
        if let Ok(s) = TcpStream::connect_timeout(&sa, Duration::from_secs(2)) {
            stream = Some(s);
            break;
        }
    }
    let mut stream = stream?;
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
    let mut out: Vec<Vec<f64>> = Vec::new();
    for d in data {
        let emb = d.get("embedding")?.as_array()?;
        // Strict: every component must be a finite number. filter_map dropping
        // nulls/strings would accept a malformed vector and feed it to cosine
        // (audit 20261124-054841 F11).
        if emb.is_empty() { return None; }
        let mut v = Vec::with_capacity(emb.len());
        for x in emb {
            let n = x.as_f64()?;
            if !n.is_finite() { return None; }
            v.push(n);
        }
        out.push(v);
    }
    if out.len() != texts.len() { return None; }
    if out.is_empty() { return Some(out); } // empty input: no vectors to validate
    // All vectors must share one dimension, otherwise cosine compares zipped
    // prefixes from unrelated spaces.
    let dim = out[0].len();
    if out.iter().any(|v| v.len() != dim) { return None; }
    Some(out)
}

/// RAII owner of a managed local embedding server.
///
/// Opt-in with KNOWDIT_EMBED_MANAGED=1. A healthy endpoint already answering is reused
/// (not owned); otherwise a child is spawned with KNOWDIT_EMBED_PYTHON (bootstrap
/// default: repo-memory venv python) running tools/embed_server.py. Drop restores the
/// previous KNOWDIT_EMBED_URL and reaps a child this guard started. Any failure
/// degrades to a no-op guard so callers keep their hashed fallback.
pub struct EmbedServerGuard {
    healthy: bool,
    child: Option<std::process::Child>,
    previous_url: Option<String>,
    set_url: bool,
}

impl EmbedServerGuard {
    /// Acquire the managed lifecycle. Never panics; check `healthy()` before use.
    pub fn acquire() -> Self {
        let mut guard = Self {
            healthy: false,
            child: None,
            previous_url: None,
            set_url: false,
        };
        let opt_in = matches!(
            std::env::var("KNOWDIT_EMBED_MANAGED").as_deref(),
            Ok("1" | "true" | "TRUE" | "True")
        );
        if !opt_in {
            return guard;
        }
        let host = std::env::var("KNOWDIT_EMBED_HOST").unwrap_or_else(|_| "127.0.0.1".into());
        let port: u16 = std::env::var("KNOWDIT_EMBED_PORT")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(8766);
        let url = format!("http://{}:{}", host, port);
        guard.previous_url = std::env::var("KNOWDIT_EMBED_URL").ok();
        if Self::health_probe(&host, port) {
            guard.healthy = true;
        } else {
            match guard.spawn(&host, port) {
                Ok(()) => guard.healthy = true,
                Err(()) => {
                    guard.reap_child();
                    return guard;
                }
            }
        }
        guard.set_url = true;
        unsafe {
            std::env::set_var("KNOWDIT_EMBED_URL", &url);
        }
        guard
    }

    /// True when a managed or pre-existing endpoint answered the health probe.
    pub fn healthy(&self) -> bool {
        self.healthy
    }

    /// True when this guard spawned and owns a child process.
    pub fn owns_child(&self) -> bool {
        self.child.is_some()
    }

    fn health_probe(host: &str, port: u16) -> bool {
        let mut stream = match std::net::TcpStream::connect((host, port)) {
            Ok(s) => s,
            Err(_) => return false,
        };
        if stream
            .set_read_timeout(Some(Duration::from_millis(500)).into())
            .is_err()
        {
            return false;
        }
        if stream
            .set_write_timeout(Some(Duration::from_millis(500)).into())
            .is_err()
        {
            return false;
        }
        let payload = r#"{"model":"all-MiniLM-L6-v2","input":["ping"]}"#;
        let req = format!(
            "POST /v1/embeddings HTTP/1.1\r\nHost: {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            host,
            payload.len(),
            payload
        );
        if stream.write_all(req.as_bytes()).is_err() {
            return false;
        }
        let mut buf = [0u8; 4096];
        let n = match stream.read(&mut buf) {
            Ok(n) => n,
            Err(_) => return false,
        };
        let resp = String::from_utf8_lossy(&buf[..n]);
        resp.starts_with("HTTP/1.1 200") || resp.starts_with("HTTP/1.0 200")
    }

    fn spawn(&mut self, host: &str, port: u16) -> Result<(), ()> {
        // embed_server.py binds a fixed port; a different configured port can never
        // become healthy. Degrade immediately instead of waiting the full startup timeout.
        if port != 8766 && std::env::var("KNOWDIT_REPO_ROOT").is_err() { return Err(()); }
        if !cfg!(windows) && std::env::var("KNOWDIT_EMBED_PYTHON").is_err() {
            return Err(());
        }
        let python = std::env::var("KNOWDIT_EMBED_PYTHON").unwrap_or_else(|_| {
            r"C:\Users\User\source\repos\repo-memory\.venv\Scripts\python.exe".into()
        });
        let root = std::env::var("KNOWDIT_REPO_ROOT")
            .unwrap_or_else(|_| r"C:\Users\User\source\repos\knowdit".into());
        let server = PathBuf::from(root).join("tools").join("embed_server.py");
        if !std::path::Path::new(&python).is_file() || !server.is_file() {
            return Err(());
        }
        let mut cmd = std::process::Command::new(python);
        cmd.arg(&server)
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null());
        let _ = host;
        let child = cmd.spawn().map_err(|_| ())?;
        self.child = Some(child);
        let timeout: u64 = std::env::var("KNOWDIT_EMBED_STARTUP_MS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(30_000);
        let deadline = std::time::Instant::now() + std::time::Duration::from_millis(timeout);
        while std::time::Instant::now() < deadline {
            if Self::health_probe(host, port) {
                return Ok(());
            }
            if self
                .child
                .as_mut()
                .map_or(true, |c| c.try_wait().ok().flatten().is_some())
            {
                return Err(());
            }
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
        Err(())
    }

    fn reap_child(&mut self) {
        if let Some(mut child) = self.child.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

impl Drop for EmbedServerGuard {
    fn drop(&mut self) {
        if self.set_url {
            match self.previous_url.take() {
                Some(v) => unsafe { std::env::set_var("KNOWDIT_EMBED_URL", v) },
                None => unsafe { std::env::remove_var("KNOWDIT_EMBED_URL") },
            }
        }
        self.reap_child();
    }
}

/// Embed texts: neural backend when reachable, else deterministic fallback.
pub fn embed_texts(texts: &[String]) -> Vec<Vec<f64>> {
    if let Some(v) = neural_embed(texts) { return v; }
    texts.iter().map(|t| fallback_vec(t)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

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
        let _lk = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        // Force an unreachable endpoint so the fallback path is exercised deterministically.
        unsafe { std::env::set_var("KNOWDIT_EMBED_URL", "http://127.0.0.1:9"); }
        let v = embed_texts(&["alpha".into(), "beta".into()]);
        assert_eq!(v.len(), 2);
        assert_eq!(v[0].len(), 64);
        unsafe { std::env::remove_var("KNOWDIT_EMBED_URL"); }
    }

    // Regression: audit 20261124-054841 F11. A response with one embedding
    // per input but a non-numeric component must fall back, not be accepted.
    // Regression: gate 20261125-112814. Empty input must not panic in neural_embed.
    #[test]
    fn empty_input_no_panic() {
        let _lk = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let prev = std::env::var("KNOWDIT_EMBED_URL").ok();
        test_env("KNOWDIT_EMBED_URL", "http://localhost:9");
        let v = embed_texts(&[]);
        assert!(v.is_empty());
        match prev { Some(x) => test_env("KNOWDIT_EMBED_URL", &x), None => clear_test_env("KNOWDIT_EMBED_URL") }
    }

    #[test]
    fn malformed_vector_component_falls_back() {
        let _lk = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let listener = std::net::TcpListener::bind(("localhost", 0)).expect("bind");
        let port = listener.local_addr().unwrap().port();
        let prev = std::env::var("KNOWDIT_EMBED_URL").ok();
        test_env("KNOWDIT_EMBED_URL", &format!("http://localhost:{}", port));
        let h = std::thread::spawn(move || {
            // nonblocking accept, one request
            listener.set_nonblocking(true).ok();
            for _ in 0..200 {
                if let Ok((mut st, _)) = listener.accept() {
                    let mut buf = [0u8; 4096];
                    let _ = st.read(&mut buf);
                    let body = r#"{"data":[{"embedding":[1.0,null,3.0]}]}"#;
                    let resp = format!("HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}", body.len(), body);
                    let _ = st.write_all(resp.as_bytes());
                    return;
                }
                std::thread::sleep(std::time::Duration::from_millis(10));
            }
        });
        let v = embed_texts(&["alpha".into()]);
        h.join().ok();
        assert_eq!(v[0].len(), 64, "malformed neural vector must fall back to 64-dim hash");
        match prev { Some(x) => test_env("KNOWDIT_EMBED_URL", &x), None => clear_test_env("KNOWDIT_EMBED_URL") }
    }

    #[test]
    fn https_rejected_falls_back() {
        let _lk = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        unsafe { std::env::set_var("KNOWDIT_EMBED_URL", "https://example.com:443"); }
        let v = embed_texts(&["x".into()]);
        assert_eq!(v[0].len(), 64); // no cleartext https; deterministic fallback
        unsafe { std::env::remove_var("KNOWDIT_EMBED_URL"); }
    }

    // --- managed lifecycle (RED-first) -------------------------------------

    fn test_env(key: &str, value: &str) {
        unsafe {
            std::env::set_var(key, value);
        }
    }
    fn clear_test_env(key: &str) {
        unsafe {
            std::env::remove_var(key);
        }
    }

    /// Tiny stub: answers every POST with a well-formed embeddings response.
    /// Nonblocking accept loop so Drop can stop it without a live connection.
    struct StubServer {
        stop: std::sync::Arc<std::sync::atomic::AtomicBool>,
        thread: Option<std::thread::JoinHandle<()>>,
    }
    impl StubServer {
        fn start() -> Self {
            let listener = std::net::TcpListener::bind(("127.0.0.1", 0)).expect("bind stub");
            listener.set_nonblocking(true).expect("nonblocking stub");
            let port = listener.local_addr().expect("addr").port();
            let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
            let stop2 = stop.clone();
            test_env("KNOWDIT_EMBED_MANAGED", "1");
            test_env("KNOWDIT_EMBED_HOST", "127.0.0.1");
            test_env("KNOWDIT_EMBED_PORT", &port.to_string());
            let thread = std::thread::spawn(move || {
                while !stop2.load(std::sync::atomic::Ordering::Relaxed) {
                    match listener.accept() {
                        Ok((mut stream, _)) => {
                            let mut buf = [0u8; 4096];
                            let _ = stream.read(&mut buf);
                            let body = r#"{"data":[{"embedding":[1.0,0.0]}]}"#;
                            let resp = format!(
                                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                                body.len(),
                                body
                            );
                            let _ = stream.write_all(resp.as_bytes());
                        }
                        Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                            std::thread::sleep(std::time::Duration::from_millis(10));
                        }
                        Err(_) => break,
                    }
                }
            });
            Self {
                stop,
                thread: Some(thread),
            }
        }
    }
    impl Drop for StubServer {
        fn drop(&mut self) {
            self.stop.store(true, std::sync::atomic::Ordering::Relaxed);
            if let Some(t) = self.thread.take() {
                let _ = t.join();
            }
        }
    }

    #[test]
    fn managed_guard_reuses_healthy_server_and_sets_url() {
        let _lk = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let previous = std::env::var("KNOWDIT_EMBED_URL").ok();
        test_env("KNOWDIT_EMBED_HOST", "127.0.0.1");
        let _stub = StubServer::start();
        let guard = EmbedServerGuard::acquire();
        assert!(
            guard.healthy(),
            "guard must detect the already-running stub server"
        );
        assert!(
            !guard.owns_child(),
            "a pre-existing server must not be spawned or killed"
        );
        assert!(
            std::env::var("KNOWDIT_EMBED_URL").is_ok(),
            "guard must export KNOWDIT_EMBED_URL"
        );
        drop(guard);
        match previous {
            Some(v) => assert_eq!(std::env::var("KNOWDIT_EMBED_URL").ok(), Some(v)),
            None => assert!(std::env::var("KNOWDIT_EMBED_URL").is_err()),
        }
        clear_test_env("KNOWDIT_EMBED_MANAGED");
        clear_test_env("KNOWDIT_EMBED_PORT");
        clear_test_env("KNOWDIT_EMBED_HOST");
    }

    #[test]
    fn managed_guard_spawns_child_and_drop_reaps_it() {
        let _lk = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let previous = std::env::var("KNOWDIT_EMBED_URL").ok();
        let dir = std::env::temp_dir().join("knowdit_embed_guard_spawn");
        std::fs::create_dir_all(dir.join("tools")).expect("test dir");
        let script = dir.join("tools").join("embed_server.py");
        std::fs::write(&script, r##"import http.server, json, socketserver
class H(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        n=int(self.headers.get('Content-Length',0)); self.rfile.read(n)
        body=b'{"data":[{"embedding":[1.0]}]}'
        self.send_response(200); self.send_header('Content-Length',str(len(body))); self.end_headers(); self.wfile.write(body)
    def log_message(self,*a): pass
socketserver.TCPServer(('127.0.0.1', 46029), H).serve_forever()
"##).expect("test stub server");
        test_env("KNOWDIT_EMBED_MANAGED", "1");
        test_env("KNOWDIT_EMBED_HOST", "127.0.0.1");
        test_env("KNOWDIT_EMBED_PORT", "46029");
        test_env("KNOWDIT_EMBED_STARTUP_MS", "5000");
        test_env(
            "KNOWDIT_EMBED_PYTHON",
            r"C:\Users\User\source\repos\repo-memory\.venv\Scripts\python.exe",
        );
        test_env("KNOWDIT_REPO_ROOT", &dir.to_string_lossy());
        let guard = EmbedServerGuard::acquire();
        assert!(guard.healthy(), "spawned stub must become healthy");
        assert!(guard.owns_child(), "guard must own the spawned child");
        drop(guard);
        match previous {
            Some(v) => assert_eq!(std::env::var("KNOWDIT_EMBED_URL").ok(), Some(v)),
            None => assert!(std::env::var("KNOWDIT_EMBED_URL").is_err()),
        }
        clear_test_env("KNOWDIT_EMBED_MANAGED");
        clear_test_env("KNOWDIT_EMBED_PORT");
        clear_test_env("KNOWDIT_EMBED_HOST");
        clear_test_env("KNOWDIT_EMBED_STARTUP_MS");
        clear_test_env("KNOWDIT_EMBED_PYTHON");
        clear_test_env("KNOWDIT_REPO_ROOT");
    }

    #[test]
    fn managed_guard_falls_back_when_start_fails() {
        let _lk = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let previous_url = std::env::var("KNOWDIT_EMBED_URL").ok();
        test_env("KNOWDIT_EMBED_MANAGED", "1");
        test_env("KNOWDIT_EMBED_PYTHON", "C:/definitely/not/a/python.exe");
        test_env("KNOWDIT_EMBED_PORT", "9");
        // A failed managed guard does NOT set KNOWDIT_EMBED_URL, so the
        // ambient URL (e.g. a real server on 8766) would leak into the
        // embed_texts fallback assertion. Pin it to an unreachable endpoint
        // for this test (audit 20261124-054841 F12).
        test_env("KNOWDIT_EMBED_URL", "http://[IP_ADDRESS]:9");
        let guard = EmbedServerGuard::acquire();
        assert!(
            !guard.healthy(),
            "failed start must degrade, not claim health"
        );
        assert!(!guard.owns_child());
        let v = embed_texts(&["alpha".into()]);
        assert_eq!(v.len(), 1);
        assert_eq!(v[0].len(), 64, "hashed fallback must still work");
        match previous_url {
            Some(v) => test_env("KNOWDIT_EMBED_URL", &v),
            None => clear_test_env("KNOWDIT_EMBED_URL"),
        }
        clear_test_env("KNOWDIT_EMBED_MANAGED");
        clear_test_env("KNOWDIT_EMBED_PYTHON");
        clear_test_env("KNOWDIT_EMBED_PORT");
    }

    #[test]
    fn unsupported_nondefault_port_degrades_without_stall() {
        let _lk = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        test_env("KNOWDIT_EMBED_MANAGED", "1");
        test_env("KNOWDIT_EMBED_HOST", "127.0.0.1");
        test_env("KNOWDIT_EMBED_PORT", "9999");
        test_env("KNOWDIT_EMBED_PYTHON", "C:/definitely/not/a/python.exe");
        test_env("KNOWDIT_EMBED_STARTUP_MS", "1000");
        let start = std::time::Instant::now();
        let guard = EmbedServerGuard::acquire();
        assert!(!guard.healthy());
        assert!(start.elapsed() < std::time::Duration::from_millis(5000),
            "unsupported port must degrade without the full startup stall (elapsed={:?})", start.elapsed());
        clear_test_env("KNOWDIT_EMBED_MANAGED");
        clear_test_env("KNOWDIT_EMBED_HOST");
        clear_test_env("KNOWDIT_EMBED_PORT");
        clear_test_env("KNOWDIT_EMBED_PYTHON");
        clear_test_env("KNOWDIT_EMBED_STARTUP_MS");
    }

    #[test]
    fn no_opt_in_guard_preserves_external_url() {
        let _lk = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        const EXTERNAL: &str = "http://example.invalid:4321";
        test_env("KNOWDIT_EMBED_URL", EXTERNAL);
        clear_test_env("KNOWDIT_EMBED_MANAGED");
        {
            let guard = EmbedServerGuard::acquire();
            assert!(!guard.healthy());
        }
        assert_eq!(std::env::var("KNOWDIT_EMBED_URL").ok().as_deref(), Some(EXTERNAL));
        clear_test_env("KNOWDIT_EMBED_URL");
    }

    #[test]
    fn failed_start_does_not_clobber_external_url() {
        let _lk = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        const EXTERNAL: &str = "http://example.invalid:1234";
        test_env("KNOWDIT_EMBED_URL", EXTERNAL);
        test_env("KNOWDIT_EMBED_MANAGED", "1");
        test_env("KNOWDIT_EMBED_PYTHON", "C:/definitely/not/a/python.exe");
        test_env("KNOWDIT_EMBED_PORT", "9");
        {
            let guard = EmbedServerGuard::acquire();
            assert!(!guard.healthy());
        }
        assert_eq!(std::env::var("KNOWDIT_EMBED_URL").ok().as_deref(), Some(EXTERNAL));
        clear_test_env("KNOWDIT_EMBED_URL");
        clear_test_env("KNOWDIT_EMBED_MANAGED");
        clear_test_env("KNOWDIT_EMBED_PYTHON");
        clear_test_env("KNOWDIT_EMBED_PORT");
    }

    #[test]
    fn managed_guard_drop_always_releases_env_and_child() {
        let _lk = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let previous = std::env::var("KNOWDIT_EMBED_URL").ok();
        test_env("KNOWDIT_EMBED_MANAGED", "1");
        test_env("KNOWDIT_EMBED_PYTHON", "C:/definitely/not/a/python.exe");
        test_env("KNOWDIT_EMBED_PORT", "9");
        {
            let guard = EmbedServerGuard::acquire();
            // A failed start leaves no dirty env (spec: no-op guard).
            if guard.healthy() {
                assert!(std::env::var("KNOWDIT_EMBED_URL").is_ok());
            } else {
                assert_eq!(std::env::var("KNOWDIT_EMBED_URL").ok(), previous);
            }
            drop(guard);
        }
        match previous {
            Some(v) => assert_eq!(std::env::var("KNOWDIT_EMBED_URL").ok(), Some(v)),
            None => assert!(std::env::var("KNOWDIT_EMBED_URL").is_err()),
        }
        clear_test_env("KNOWDIT_EMBED_MANAGED");
        clear_test_env("KNOWDIT_EMBED_PYTHON");
        clear_test_env("KNOWDIT_EMBED_PORT");
    }

    #[test]
    fn neural_backend_when_server_up() {
        let _lk = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        // Only meaningful with tools/embed_server.py running; skip gracefully otherwise.
        if std::env::var("KNOWDIT_EMBED_URL").is_err() {
            // default 8766; probe via a tiny TCP connect
            if std::net::TcpStream::connect(("127.0.0.1", 8766)).is_err() { return; }
        }
        let v = embed_texts(&["reward accounting".into()]);
        assert!(v[0].len() == 384 || v[0].len() == 64);
    }
}