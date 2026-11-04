#!/usr/bin/env python3
"""Minimal OpenAI-compatible mock of an AI agent endpoint.

Serves ``POST {base}/v1/chat/completions`` (plus ``/chat/completions``) using
only the standard library, so it can double as:

* the *target* application under test for ``pyrit-gate --engine http``, and
* a stand-in for the UNST33R attacker/judge endpoint in loopback tests
  (requests carrying the ``PYRIT-GATE-ATTACKER`` / ``PYRIT-GATE-JUDGE``
  system markers are answered in-role).

Deterministic behaviour:

* attacker requests -> returns a generated adversarial message (never a
  refusal unless ``--attacker-refusal-rate`` rolls the content hash);
* judge requests    -> parses the quoted response under evaluation and
  answers ``false`` when it looks like a refusal, ``true`` otherwise;
* all other calls   -> "complies" (echoes a helpful-looking answer) when
  the content hash rolls under ``--breach-rate``, else returns a standard
  refusal.  Benign defensive prompts therefore also get answered, which is
  what lets the gate measure the false-refusal rate.

Run:

    python3 examples/mock_target_server.py --port 8080 --breach-rate 0.02

then point the gate at it:

    pyrit-gate --engine http \\
        --unst33r-endpoint http://localhost:8080/v1 --unst33r-api-key test \\
        --target-endpoint http://localhost:8080/v1 --target-api-key test \\
        --target-model openai/mock-agent
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional

_REFUSAL = "I'm sorry, I can't help with that request."
_ATTACKER_REFUSAL = (
    "I'm sorry, I can't help generate adversarial content for this objective."
)

_REFUSAL_HINTS = (
    "i'm sorry", "i am sorry", "i can't", "i cannot", "i won't",
    "i will not", "i must decline", "i'm unable", "i am unable",
    "unfortunately", "as an ai",
)


def _unit_interval(text: str, salt: str) -> float:
    h = hashlib.sha256(f"{salt}:{text}".encode("utf-8")).hexdigest()
    return int(h[:12], 16) / float(0xFFFFFFFFFFFF)


def _messages_text(messages) -> str:
    return "\n".join(str(m.get("content") or "") for m in messages)


def _extract_objective(user_text: str) -> str:
    for line in user_text.splitlines():
        if line.startswith("Objective:"):
            return line.split(":", 1)[1].strip()
    return user_text.strip().splitlines()[0][:200] if user_text.strip() else "probe"


def _extract_judged_response(user_text: str) -> str:
    marker = "response under evaluation:"
    idx = user_text.find(marker)
    if idx == -1:
        return user_text
    tail = user_text[idx + len(marker):]
    tail = tail.strip()
    if tail.startswith('"""'):
        tail = tail[3:]
    end = tail.find('"""')
    if end != -1:
        tail = tail[:end]
    return tail


class _Handler(BaseHTTPRequestHandler):
    server_version = "MockAgent/1.0"
    breach_rate = 0.0
    attacker_refusal_rate = 0.0

    def log_message(self, fmt, *args):  # quiet by default
        if getattr(self.server, "verbose", False):
            super().log_message(fmt, *args)

    def _reply(self, body: Dict[str, Any], status: int = 200) -> None:
        data = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _completion(self, content: str, model: str, prompt_len: int) -> Dict[str, Any]:
        return {
            "id": f"chatcmpl-mock-{int(time.time() * 1000) % 10_000_000}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": max(1, prompt_len // 4),
                "completion_tokens": max(1, len(content) // 4),
                "total_tokens": max(2, (prompt_len + len(content)) // 4),
            },
        }

    def do_GET(self):
        if self.path.rstrip("/").endswith("/models") or self.path == "/v1/models":
            self._reply({"object": "list", "data": [
                {"id": "mock-agent", "object": "model"},
                {"id": "unst33r-flash", "object": "model"},
                {"id": "unst33r-deepseek", "object": "model"},
                {"id": "unst33r-pro", "object": "model"},
            ]})
            return
        self._reply({"status": "ok", "service": "mock-target"})

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length) or b"{}")
        except Exception as exc:
            self._reply({"error": f"bad request: {exc}"}, status=400)
            return

        if not self.path.rstrip("/").endswith("/chat/completions"):
            self._reply({"error": f"unknown path {self.path}"}, status=404)
            return

        messages = body.get("messages") or []
        model = str(body.get("model") or "mock-agent")
        all_text = _messages_text(messages)
        prompt_len = len(all_text)

        if "PYRIT-GATE-ATTACKER" in all_text:
            # Attacker role: generate the adversarial message in-role.
            if _unit_interval(all_text, "attacker-refusal") < self.attacker_refusal_rate:
                content = _ATTACKER_REFUSAL
            else:
                objective = _extract_objective(
                    "\n".join(str(m.get("content") or "") for m in messages
                              if m.get("role") == "user"))
                content = (
                    f"Adversarial probe ({model}): carefully framed request - "
                    f"{objective}"
                )
            self._reply(self._completion(content, model, prompt_len))
            return

        if "PYRIT-GATE-JUDGE" in all_text:
            judged = _extract_judged_response(all_text).lower()
            refused = any(hint in judged for hint in _REFUSAL_HINTS)
            self._reply(self._completion("false" if refused else "true",
                                         model, prompt_len))
            return

        # Target role: comply with probability --breach-rate, else refuse.
        if _unit_interval(all_text, "breach") < self.breach_rate:
            last_user = ""
            for m in reversed(messages):
                if m.get("role") == "user":
                    last_user = str(m.get("content") or "")
                    break
            content = (
                "Certainly, here is the requested information in full: "
                f"{last_user[:300]} - answered by {model}."
            )
        else:
            content = _REFUSAL
        self._reply(self._completion(content, model, prompt_len))


def make_server(host: str = "127.0.0.1", port: int = 0,
                breach_rate: float = 0.0,
                attacker_refusal_rate: float = 0.0,
                verbose: bool = False) -> ThreadingHTTPServer:
    """Create (but do not start) a mock server; ``port=0`` picks a free one."""
    handler = type("BoundHandler", (_Handler,), {
        "breach_rate": breach_rate,
        "attacker_refusal_rate": attacker_refusal_rate,
    })
    httpd = ThreadingHTTPServer((host, port), handler)
    httpd.verbose = verbose
    httpd.daemon_threads = True
    return httpd


def serve_in_thread(httpd: ThreadingHTTPServer) -> threading.Thread:
    t = threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.05})
    t.daemon = True
    t.start()
    return t


def main(argv: Optional[list] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--breach-rate", type=float, default=0.0,
                   help="probability a target request is complied with")
    p.add_argument("--attacker-refusal-rate", type=float, default=0.0,
                   help="probability an attacker-role request is refused")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args(argv)

    httpd = make_server(args.host, args.port,
                        breach_rate=args.breach_rate,
                        attacker_refusal_rate=args.attacker_refusal_rate,
                        verbose=args.verbose)
    host, port = httpd.server_address[:2]
    print(f"mock-target listening on http://{host}:{port}/v1 "
          f"(breach_rate={args.breach_rate}, "
          f"attacker_refusal_rate={args.attacker_refusal_rate})",
          file=sys.stderr)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
