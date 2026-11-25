"""
Standalone per-model inference microservice.

One process per model, pinned to one GPU, exposing a tiny stdlib HTTP API:
  GET  /health    -> {"ready": true, "model": ...}  (only served once loaded)
  POST /generate  -> {"answer", "ttft_ms", "tpot_ms", "total_ms",
                      "prompt_tokens", "completion_tokens", "total_tokens"}

Run independently of Streamlit (started by run_app.sh) so the app never spawns
multiprocessing workers — which was deadlocking under Streamlit's script runner.

Usage:
  CUDA_VISIBLE_DEVICES is set from --gpu.
  python streamlit_app/model_service.py --key llama3.1-8b --name Llama-3.1-8B \
         --gpu 5 --port 8101 [--reasoning]
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--gpu", required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--reasoning", action="store_true")
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(ROOT))
    try:
        from src.utils.proctitle import set_proctitle
        set_proctitle()  # tahir_Ont-code
    except Exception:
        pass

    from src.llm.client import LLMClient
    print(f"[{args.name}] loading on GPU {args.gpu} (port {args.port})…", flush=True)
    llm = LLMClient(backend="local", model_key=args.key, max_tokens=160, temperature=0.0)
    client = llm._client
    client.load()
    aprefix = "<think>\n\n</think>\n\n" if args.reasoning else ""
    gpu_lock = threading.Lock()  # one model on one GPU -> serialize generations
    print(f"[{args.name}] READY on http://127.0.0.1:{args.port}", flush=True)

    def strip_think(t: str) -> str:
        return re.sub(r"<think>.*?</think>", "", t, flags=re.DOTALL).replace("</think>", "").strip()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):  # silence access logs
            pass

        def _send(self, code: int, obj: dict) -> None:
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                self._send(200, {"ready": True, "model": args.name})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/generate":
                self._send(404, {"error": "not found"})
                return
            n = int(self.headers.get("Content-Length", 0) or 0)
            try:
                req = json.loads(self.rfile.read(n) or b"{}")
            except Exception:
                self._send(400, {"error": "bad json"}); return
            prompt = req.get("prompt", "")
            system = req.get("system")
            mx = int(req.get("max_tokens", 160))
            t0 = time.perf_counter()
            try:
                with gpu_lock, contextlib.redirect_stdout(io.StringIO()):
                    ans = client.stream_call(prompt, system=system, max_tokens=mx,
                                             temperature=0.0, assistant_prefix=aprefix)
                m = dict(client._last_metrics)
                ans = strip_think(ans)
            except Exception as e:
                ans, m = f"[generation error: {str(e)[:120]}]", {}
            m["total_ms"] = round((time.perf_counter() - t0) * 1000, 1)
            self._send(200, {
                "answer": ans,
                "ttft_ms": m.get("ttft_ms"), "tpot_ms": m.get("tpot_ms"),
                "total_ms": m.get("total_ms"),
                "prompt_tokens": m.get("input_tokens"),
                "completion_tokens": m.get("output_tokens"),
                "total_tokens": (m.get("input_tokens") or 0) + (m.get("output_tokens") or 0),
            })

    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
