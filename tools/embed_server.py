#!/usr/bin/env python3
"""Local OpenAI-compatible embeddings server for knowdit's neural backend.

Serves POST /v1/embeddings on 127.0.0.1:8766 using sentence-transformers
all-MiniLM-L6-v2 (cached offline). Start with the repo-memory venv python:
  C:\\Users\\User\\source\\repos\\repo-memory\\.venv\\Scripts\\python.exe tools/embed_server.py
knowdit reads KNOWDIT_EMBED_URL (default http://127.0.0.1:8766).
"""
import json
from http.server import BaseHTTPRequestHandler, HTTPServer

from sentence_transformers import SentenceTransformer

MODEL = SentenceTransformer("all-MiniLM-L6-v2")


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path != "/v1/embeddings":
            self.send_error(404)
            return
        n = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(n) or b"{}")
        texts = req.get("input", [])
        if isinstance(texts, str):
            texts = [texts]
        vecs = MODEL.encode(texts, normalize_embeddings=True).tolist()
        body = json.dumps({
            "object": "list",
            "data": [{"object": "embedding", "index": i, "embedding": v} for i, v in enumerate(vecs)],
            "model": req.get("model", "all-MiniLM-L6-v2"),
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    print("embeddings server on 127.0.0.1:8766", flush=True)
    HTTPServer(("127.0.0.1", 8766), Handler).serve_forever()