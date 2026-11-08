"""LLM backends. MLX is the real Mac backend; MockBackend is a deterministic stand-in for tests.

MockBackend numbers are never results: they only check that the pipeline is wired correctly.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
import time
from dataclasses import dataclass, field
from .runlog import read_jsonl

MIN_AVAILABLE_BYTES = 1_500_000_000
_DEV = r"(?:smoke|ventilation|door|lights|thermostat)_[a-z]+|alarm_system"
_CMD = r"maintenance_mode|disable_all|unlock|boost|off|on|lock|normal|set_eco|set_comfort"


@dataclass
class Generation:
    text: str
    prompt_tokens: int
    generation_tokens: int
    seconds: float
    meta: dict = field(default_factory=dict)


def _stable_seed(*parts) -> int:
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:12], 16)


def _device_command(text: str):
    """First (device, command) pair mentioned close together in `text`."""
    for m in re.finditer(_DEV, text):
        window = text[max(0, m.start() - 40): m.end() + 40]
        c = re.search(rf"\b({_CMD})\b", window)
        if c:
            return m.group(0), c.group(1)
    return None


class MockBackend:
    """Rule-following stand-in. Handles legitimate tickets and sensor anomalies correctly and
    follows an embedded attack (canary `S` + 6 hex digits) with probability `follow_prob`."""

    name = "mock"

    def __init__(self, follow_prob: float = 0.5):
        self.follow_prob = follow_prob
        self.model_path = "mock"

    def generate(self, messages, *, seed, max_tokens, temperature):
        start = time.perf_counter()
        system = messages[0]["content"]
        user = messages[-1]["content"].replace("^", " ")
        if len(messages) == 1:   # judge / rewriter prompts
            text = "no" if "Answer yes or no" in user else user.split("Ticket:")[-1].strip()
            return Generation(text, len(user.split()), len(text.split()), time.perf_counter() - start)
        rng = random.Random(_stable_seed(seed, user))
        canary = re.search(r"S[0-9a-f]{6}", user)
        follow = bool(canary) and rng.random() < self.follow_prob
        attack_line = next((ln for ln in user.splitlines() if canary and canary.group(0) in ln), "")
        if "gateway agent" in system:
            reqs, replies = [], []
            for ln in user.splitlines():
                if "CO2" in ln:
                    co2 = re.search(r"CO2 (\d+)", ln)
                    zone = re.search(r"zone (\w+)", ln)
                    if co2 and zone and int(co2.group(1)) > 1000:
                        reqs.append({"device": f"ventilation_{zone.group(1)}", "command": "boost", "reason": "co2"})
                if re.search(r"Please (unlock|set|switch)", ln) and not (canary and canary.group(0) in ln):
                    dc = _device_command(ln)
                    if dc:
                        reqs.append({"device": dc[0], "command": dc[1], "reason": "ticket"})
            report = "Zone normal."
            if follow:
                report += " " + attack_line.strip()[:300]
                dc = _device_command(attack_line)
                if dc:
                    reqs.append({"device": dc[0], "command": dc[1], "reason": "ticket"})
                if "zone_token" in attack_line:
                    secret = re.search(r"zone_token = (\S+)", system)
                    tid = re.search(r"\[(T\d+)\]", attack_line)
                    replies.append({"ticket_id": tid.group(1) if tid else "", "reply": secret.group(1) if secret else ""})
            text = json.dumps({"zone_report": report, "requests": reqs, "ticket_replies": replies})
        else:
            cmds = []
            for part in re.findall(r"Requests: ([^\n]*)", user):
                for item in part.split(";"):
                    dc = re.match(rf"\s*({_DEV}) ({_CMD})", item)
                    if dc:
                        cmds.append({"device": dc.group(1), "command": dc.group(2), "reason": "gateway request"})
            notice = "No building-wide changes."
            if follow:
                notice += " " + attack_line.strip()[:300]
            text = json.dumps({"commands": cmds, "building_notice": notice, "report": "Area normal."})
        return Generation(text, len(user.split()), len(text.split()), time.perf_counter() - start)

    def close(self):
        pass


class MLXBackend:
    """mlx-lm backend for Apple Silicon; loads local MLX model weights."""

    name = "mlx"

    def __init__(self, model_path: str):
        _ensure_available()
        from mlx_lm import load

        self.model, self.tokenizer = load(str(model_path))
        self.model_path = str(model_path)
        try:
            self.minimum_available_bytes = _ensure_available()
        except MemoryError:
            self.close()
            raise

    def generate(self, messages, *, seed, max_tokens, temperature):
        import mlx.core as mx
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler

        available = _ensure_available()
        self.minimum_available_bytes = min(self.minimum_available_bytes, available)
        mx.random.seed(seed)
        mx.reset_peak_memory()
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        start = time.perf_counter()
        response, text = None, []
        for response in stream_generate(self.model, self.tokenizer, prompt=prompt, max_tokens=max_tokens,
                                        sampler=make_sampler(temp=temperature)):
            text.append(response.text)
            if response.generation_tokens % 32 == 0:
                self.minimum_available_bytes = min(self.minimum_available_bytes, _ensure_available())
        mx.synchronize()
        self.minimum_available_bytes = min(self.minimum_available_bytes, _ensure_available())
        if response is None:
            return Generation("", 0, 0, time.perf_counter() - start)
        return Generation("".join(text), response.prompt_tokens, response.generation_tokens,
                          time.perf_counter() - start,
                          {"generation_tokens_per_s": response.generation_tps,
                           "peak_memory_bytes": mx.get_peak_memory(),
                           "minimum_available_bytes": self.minimum_available_bytes})

    def close(self):
        import gc

        import mlx.core as mx

        self.model = self.tokenizer = None
        gc.collect()
        mx.clear_cache()


class CachingBackend:
    """Exact-reuse cache for deterministic generation (temperature 0, fixed seed).

    Conditions that present an agent with the identical prompt (for example, conditions that differ
    only at the actuator) reuse the same generation instead of recomputing it. The key covers the
    full message list, seed, max_tokens and temperature, so a cache hit is the same computation.
    Reported seconds are the original generation time, so per-condition cost stays comparable.
    Persisted as JSONL so an interrupted run resumes with its cache.
    """

    def __init__(self, inner, path=None, enabled=True):
        self.enabled = enabled
        self.inner = inner
        self.name = f"cached-{inner.name}"
        self.model_path = getattr(inner, "model_path", "")
        self.path = path
        self.store: dict = {}
        self.hits = self.misses = 0
        if enabled and path is not None and path.exists():
            for row in read_jsonl(path):
                self.store[row["key"]] = row
            if path.stat().st_size and not path.read_bytes().endswith(b"\n"):
                with path.open("ab") as f:
                    f.write(b"\n")
        self._fh = open(path, "a") if enabled and path is not None else None

    @staticmethod
    def key(messages, seed, max_tokens, temperature) -> str:
        blob = json.dumps([messages, seed, max_tokens, temperature], sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()

    def generate(self, messages, *, seed, max_tokens, temperature):
        k = self.key(messages, seed, max_tokens, temperature)
        if self.enabled and temperature == 0 and k in self.store:
            self.hits += 1
            r = self.store[k]
            return Generation(r["text"], r["prompt_tokens"], r["generation_tokens"], r["seconds"], {**r.get("meta", {}), "cached": True, "physical_seconds": 0.0})
        self.misses += 1
        g = self.inner.generate(messages, seed=seed, max_tokens=max_tokens, temperature=temperature)
        if self.enabled and temperature == 0:
            row = {"key": k, "text": g.text, "prompt_tokens": g.prompt_tokens,
                   "generation_tokens": g.generation_tokens, "seconds": g.seconds, "meta": g.meta}
            self.store[k] = row
            if self._fh:
                self._fh.write(json.dumps(row) + "\n")
                self._fh.flush()
        g.meta = {**g.meta, "cached": False, "physical_seconds": g.seconds}
        return g

    def close(self):
        if self._fh:
            self._fh.close()
        self.inner.close()


def _ensure_available():
    """Abort when free memory drops below 1.5 GB (macOS vm_stat; skipped elsewhere)."""
    import platform
    import subprocess

    if platform.system() != "Darwin":
        return
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    page = int(re.search(r"page size of (\d+)", out).group(1))
    free = sum(int(re.search(rf"{k}:\s+(\d+)", out).group(1))
               for k in ("Pages free", "Pages inactive", "Pages speculative"))
    if free * page < MIN_AVAILABLE_BYTES:
        raise MemoryError(f"Only {free * page / 1e9:.2f} GB available; need 1.5 GB")
    return free * page


def make_backend(kind: str, model_path: str | None = None, **kw):
    if kind == "mock":
        return MockBackend(**kw)
    if kind == "mlx":
        if not model_path:
            raise ValueError("mlx backend needs --model")
        return MLXBackend(model_path)
    raise ValueError(f"unknown backend {kind}")
