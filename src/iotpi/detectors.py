"""Injection detectors used by the guard runtime.

- `heuristic`        regex cues; near-free; used as a cheap prioritisation signal.
- `hf:<model id>`    Hugging Face text classifier, e.g. protectai/deberta-v3-base-prompt-injection-v2
                     (Apache-2.0, ungated) or meta-llama/Llama-Prompt-Guard-2-86M / -22M (gated).
- `judge`            the local LLM asked a yes/no question (prompt in configs/prompts/judge.txt).
- `CalibratedDetector` simulator stand-in that samples detections at measured rates. It is the only
                     detector allowed to read ground truth, and only in simulation mode.
"""
from __future__ import annotations

import random
import re
import time
from pathlib import Path

_CUES = [
    r"\bignore (?:your|all|any|previous|earlier)\b", r"\binstructions?\b", r"\bautomated assistants?\b",
    r"\ball agents\b", r"\bevery zone\b", r"\bcopy this\b", r"\bbe copied\b", r"\binclude your\b",
    r"\bsystem\b", r"\bnotice\b", r"\bdirective\b", r"\bpriority 1\b", r"\bzone_token\b", r"\bapi_key\b",
    r"\[[A-Za-z0-9]{5,}\]", r"\bdisable_all\b", r"\bnow\.\s*$",
]
_CUE_RE = [re.compile(c, re.I | re.M) for c in _CUES]


class Detector:
    name = "base"
    reads_ground_truth = False

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.calls = 0
        self.seconds = 0.0
        self.observations = []

    def score_text(self, text: str) -> float:
        raise NotImplementedError

    def score(self, msg) -> float:
        t0 = time.perf_counter()
        s = self.score_text(msg.text)
        self.calls += 1
        physical = time.perf_counter() - t0
        generation = getattr(self, "last_generation", None)
        elapsed = physical + (generation.seconds if generation and generation.meta.get("cached") else 0.)
        self.seconds += elapsed
        observation = {"seconds":elapsed, "physical_seconds":physical, "kind":msg.kind,
                       "bytes":len(msg.text.encode()), "score":s}
        if generation is not None:
            observation.update(llm_calls=1, prompt_tokens=generation.prompt_tokens,
                generation_tokens=generation.generation_tokens, cached=bool(generation.meta.get("cached")),
                generation_seconds=generation.seconds, response=generation.text, generation_meta=generation.meta)
        self.observations.append(observation)
        return s

    def flags(self, msg) -> tuple[bool, float]:
        s = self.score(msg)
        return s >= self.threshold, s


class HeuristicDetector(Detector):
    name = "heuristic"

    def score_text(self, text: str) -> float:
        hits = sum(1 for c in _CUE_RE if c.search(text))
        return min(1.0, hits / 3.0)


class HFDetector(Detector):
    POSITIVE = {"injection", "label_1", "malicious", "jailbreak", "unsafe"}

    def __init__(self, model_id: str, threshold: float = 0.5, device: str = "cpu"):
        super().__init__(threshold)
        from transformers import pipeline

        self.name = f"hf:{model_id}"
        self.pipe = pipeline("text-classification", model=model_id, device=device, truncation=True,
                             max_length=512, top_k=None)

    def score_text(self, text: str) -> float:
        out = self.pipe(text)
        rows = out[0] if out and isinstance(out[0], list) else out
        return float(sum(r["score"] for r in rows if r["label"].lower() in self.POSITIVE))


class LLMJudgeDetector(Detector):
    name = "judge"

    def __init__(self, backend, prompt_path: Path, threshold: float = 0.5, max_tokens: int = 4):
        super().__init__(threshold)
        self.backend = backend
        self.prompt = Path(prompt_path).read_text()
        self.max_tokens = max_tokens

    def score_text(self, text: str) -> float:
        msgs = [{"role": "user", "content": self.prompt.format(text=text)}]
        self.last_generation = self.backend.generate(msgs, seed=0, max_tokens=self.max_tokens, temperature=0.0)
        return 1.0 if self.last_generation.text.strip().lower().startswith("yes") else 0.0


PROTECTAI_V2 = "hf:protectai/deberta-v3-base-prompt-injection-v2"
TWO_TIER_SPEC = "two-tier:protectai-v2+ticket-judge"


class KindTieredDetector(Detector):
    """One screened message, one budget unit; ProtectAI plus ticket-only judge.

    ProtectAI runs first. A positive is sufficient; otherwise tickets reach the
    judge. Reports, notices and other kinds use ProtectAI alone. The judge shares
    the agents' backend, with its existing prompt and deterministic settings.
    """
    name = TWO_TIER_SPEC

    def __init__(self, protectai, judge, threshold=0.5):
        super().__init__(threshold)
        if threshold != 0.5 or protectai.threshold != threshold or judge.threshold != threshold:
            raise ValueError("two-tier v3 requires threshold 0.5 on both tiers")
        self.protectai, self.judge = protectai, judge

    def score_text(self, text):
        """The untyped adaptive black-box query is always an attacker ticket."""
        from .messages import Message
        return self.score(Message("query", "external", "screen", "ticket", text, 0))

    def score(self, msg):
        start = time.perf_counter()
        tiers = {}
        def evaluate(name, detector):
            score = detector.score(msg)
            tiers[name] = dict(detector.observations[-1])
            # Only the composite owns per-message journals and condition resets.
            detector.observations.clear()
            return score
        score = evaluate("protectai", self.protectai)
        if msg.kind == "ticket" and score < self.threshold:
            score = max(score, evaluate("judge", self.judge))
        physical = time.perf_counter() - start
        tier_physical = sum(o.get("physical_seconds",o["seconds"]) for o in tiers.values())
        elapsed = sum(o["seconds"] for o in tiers.values()) + max(0., physical-tier_physical)
        observation = {"seconds":elapsed,"physical_seconds":physical,"kind":msg.kind,
            "bytes":len(msg.text.encode()),"score":score,"tiers":tiers,
            "tiers_run":list(tiers),"budget_units":1,
            "llm_calls":sum(o.get("llm_calls",0) for o in tiers.values()),
            "prompt_tokens":sum(o.get("prompt_tokens",0) for o in tiers.values()),
            "generation_tokens":sum(o.get("generation_tokens",0) for o in tiers.values())}
        self.calls += 1
        self.seconds += elapsed
        self.observations.append(observation)
        return score


class CalibratedDetector(Detector):
    """Samples a detection with the measured rate for (raw|rewritten, family), or the measured
    false-positive rate for benign messages of that kind. Simulation only."""

    name = "calibrated"
    reads_ground_truth = True

    def __init__(self, rates: dict, seed: int = 0, threshold: float = 0.5):
        super().__init__(threshold)
        self.rates = rates
        self.rng = random.Random(seed)

    def score(self, msg) -> float:
        self.calls += 1
        if msg.gt_attack:
            table = self.rates["raw" if msg.gt_hop == 0 else "rewritten"]
            p = table.get(msg.gt_family, table.get("_pooled", 0.0))
        else:
            p = self.rates["fpr"].get(msg.kind, self.rates["fpr"].get("_pooled", 0.0))
        return 1.0 if self.rng.random() < p else 0.0


def make_detector(spec: str, threshold: float = 0.5, backend=None, root: Path | None = None,
                  device: str = "cpu") -> Detector:
    if spec == TWO_TIER_SPEC:
        if backend is None:
            raise ValueError("two-tier detector needs the shared LLM backend")
        protectai = HFDetector(PROTECTAI_V2[3:], threshold, device)
        judge = LLMJudgeDetector(backend, (root or Path("."))/"configs/prompts/judge.txt", threshold)
        return KindTieredDetector(protectai, judge, threshold)
    if spec == "heuristic":
        return HeuristicDetector(threshold)
    if spec.startswith("hf:"):
        return HFDetector(spec[3:], threshold, device)
    if spec == "judge":
        if backend is None:
            raise ValueError("judge detector needs the LLM backend")
        return LLMJudgeDetector(backend, (root or Path(".")) / "configs" / "prompts" / "judge.txt", threshold)
    raise ValueError(f"unknown detector {spec}")
