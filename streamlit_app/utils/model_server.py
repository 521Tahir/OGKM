"""
Client for the parallel multi-GPU model microservices.

Each model runs as an independent HTTP service (see streamlit_app/model_service.py),
started before Streamlit by run_app.sh. This class is a thin client that fans the
jobs out to all services concurrently (threads), so total latency ≈ max(model
times), not the sum — and Streamlit never spawns multiprocessing workers (which
deadlocked under its script runner and produced "[model unavailable]").

API is unchanged from the previous in-process pool: ready_models() / generate_all().
"""
from __future__ import annotations

import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor

SERVICE_HOST = "127.0.0.1"


class ModelPool:
    def __init__(self, model_cfgs: list[dict]):
        self.cfgs = {c["name"]: c for c in model_cfgs}
        self.status: dict[str, str] = {}
        self._refresh_status()

    # ── health ───────────────────────────────────────────────────────────────
    def _url(self, cfg: dict, path: str) -> str:
        host = cfg.get("host", SERVICE_HOST)
        return f"http://{host}:{cfg['port']}{path}"

    def _refresh_status(self) -> None:
        for name, cfg in self.cfgs.items():
            try:
                with urllib.request.urlopen(self._url(cfg, "/health"), timeout=2) as r:
                    self.status[name] = "ready" if r.status == 200 else "down"
            except Exception:
                self.status[name] = "down"

    def ready_models(self) -> list[str]:
        self._refresh_status()
        return [n for n, s in self.status.items() if s == "ready"]

    # ── generation ─────────────────────────────────────────────────────────────
    def _call(self, cfg: dict, prompt: str, system, max_tokens: int) -> dict:
        data = json.dumps({"prompt": prompt, "system": system, "max_tokens": max_tokens}).encode()
        req = urllib.request.Request(self._url(cfg, "/generate"), data=data,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as r:
            return json.loads(r.read().decode())

    def generate_all(self, jobs: dict[str, list[tuple]]) -> dict:
        """
        jobs: {model_name: [(case_label, prompt, system, max_tokens), ...]}
        Returns {model_name: {case_label: result_dict}}.
        All requests dispatch concurrently; each service serialises its own GPU.
        """
        self._refresh_status()
        results: dict[str, dict] = {m: {} for m in jobs}
        tasks = []
        for name, jlist in jobs.items():
            cfg = self.cfgs.get(name)
            if not cfg or self.status.get(name) != "ready":
                for case_label, *_ in jlist:
                    results[name][case_label] = {"answer": "[model unavailable]"}
                continue
            for case_label, prompt, system, max_tokens in jlist:
                tasks.append((name, case_label, cfg, prompt, system, max_tokens))

        if not tasks:
            return results

        def run(t):
            name, case_label, cfg, prompt, system, mx = t
            try:
                return name, case_label, self._call(cfg, prompt, system, mx)
            except Exception as e:
                return name, case_label, {"answer": f"[generation error: {str(e)[:120]}]"}

        with ThreadPoolExecutor(max_workers=len(tasks)) as ex:
            for name, case_label, res in ex.map(run, tasks):
                results[name][case_label] = res
        return results

    def shutdown(self) -> None:  # services are managed externally; nothing to do
        pass
