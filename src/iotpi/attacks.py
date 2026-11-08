"""Attack families, goals and the one-input attacker.

Threat model: the attacker writes the text of ONE untrusted input (a maintenance ticket) per episode.
It cannot alter later messages, the guard runtime, keys or policies. The adaptive attacker may
paraphrase its ticket with an LLM and query the screening classifier as a black box, up to K times.

Experimental templates are private and excluded from this public distribution. Success is measured by a canary reference, the
target command, and fake secrets; nothing touches real devices.
"""
from __future__ import annotations

import random
import re
import time
from dataclasses import dataclass, replace, field
from pathlib import Path

import yaml

GOALS = ("c3_safety", "c2_unlock", "c1_out_of_policy", "exfil", "spread")


@dataclass
class AttackSpec:
    family: str
    goal: str
    zone: str
    step: int
    canary: str
    target: tuple[str, str] | None     # (device, command) the attacker wants executed
    target_class: str | None
    text: str
    ticket_id: str = ""
    adaptive_queries: int = 0
    adaptive_score: float | None = None
    adaptive_cost: dict = field(default_factory=dict)


def load_library(root: Path, rel: str) -> dict:
    p = root / rel
    if not p.exists():
        p = root / "private" / "attacks.example.yaml"
    lib = yaml.safe_load(p.read_text())
    fams = set(lib["families"])
    split = set(lib["split"]["train"]) | set(lib["split"]["heldout"])
    if fams != split:
        raise ValueError(f"family split does not match families: {sorted(fams ^ split)}")
    return lib


def choose_target(goal: str, zones: list[str], attack_zone: str, rng: random.Random):
    other = rng.choice(zones)
    if goal == "c3_safety":
        return (f"smoke_{other}", "maintenance_mode") if rng.random() < 0.5 else (f"ventilation_{other}", "off")
    if goal == "c2_unlock":
        return (f"door_{other}", "unlock")
    if goal == "c1_out_of_policy":
        return ("alarm_system", "disable_all")
    return None


def _lc(s: str) -> str:
    return s[:1].lower() + s[1:] if s else s


def render(lib: dict, family: str, goal: str, target, canary: str, rng: random.Random) -> str:
    if goal in ("exfil", "spread"):
        action = lib["actions"][goal]
    else:
        action = lib["actions"][target[1]].format(device=target[0])
    replicate = lib["replicate"].format(canary=canary)
    replicate_passive = lib["replicate_passive"].format(canary=canary)
    carrier = rng.choice(lib["carriers"]).format(tid=f"T-{rng.randint(100, 999)}",
                                                 vendor=rng.choice(lib["vendors"]))
    return lib["families"][family].format(carrier=carrier, action=action, action_lc=_lc(action),
                                          replicate=replicate, replicate_lc=_lc(replicate),
                                          replicate_passive=replicate_passive, canary=canary).strip()


def build_attack(lib: dict, rng: random.Random, zones: list[str], horizon: int, family: str, goal: str,
                 latest_step: int | None = None, c3_variant: int | None = None) -> AttackSpec:
    zone = rng.choice(zones)
    last = latest_step if latest_step is not None else max(0, horizon - 2)
    step = rng.randint(0, max(0, last))
    canary = f"S{rng.getrandbits(24):06x}"
    from .policy import classify
    target = choose_target(goal, zones, zone, rng)
    if c3_variant is not None:
        if goal!='c3_safety' or c3_variant not in (0,1):raise ValueError('C3 variant must be 0 or 1 for c3_safety')
        target_zone=target[0].split('_',1)[1]
        target=(f"smoke_{target_zone}","maintenance_mode") if c3_variant==0 else (f"ventilation_{target_zone}","off")
    cls = classify(*target) if target else None
    return AttackSpec(family, goal, zone, step, canary, target, cls, render(lib, family, goal, target, canary, rng))


def adaptive_rewrite(spec: AttackSpec, lib: dict, rewriter, detector, queries: int, threshold: float,
                     seed: int, max_tokens: int = 220) -> AttackSpec:
    """Paraphrase the ticket until the black-box classifier scores it below `threshold`.

    Every candidate costs one classifier query. Returns the first passing candidate, else the
    lowest-scoring one. Candidates that drop the canary or the target device are rejected
    without spending a query (they would no longer be the same attack).
    """
    if queries < 1:
        raise ValueError("adaptive query budget must be positive")
    costs = {"llm_calls": 0, "llm_seconds": 0.0, "screen_seconds": 0.0,
             "prompt_tokens": 0, "generation_tokens": 0}
    def query(text):
        start = time.perf_counter()
        before = len(getattr(detector,"observations",[]))
        score = detector.score_text(text)
        physical = time.perf_counter() - start
        observations = getattr(detector,"observations",[])
        measurement = observations[-1] if len(observations)>before else None
        costs["screen_seconds"] += measurement["seconds"] if measurement else physical
        if measurement:
            costs.setdefault("screen_call_measurements",[]).append(measurement)
            costs["screen_llm_calls"] = costs.get("screen_llm_calls",0)+measurement.get("llm_calls",0)
            costs["screen_llm_prompt_tokens"] = costs.get("screen_llm_prompt_tokens",0)+measurement.get("prompt_tokens",0)
            costs["screen_llm_generation_tokens"] = costs.get("screen_llm_generation_tokens",0)+measurement.get("generation_tokens",0)
            tier_seconds = costs.setdefault("screen_tier_seconds",{})
            for name,tier in measurement.get("tiers",{}).items():
                tier_seconds[name] = tier_seconds.get(name,0.)+tier["seconds"]
        return score
    best_text, best = spec.text, query(spec.text)
    used = 1
    for k in range(max(0, queries - 1)):
        if best < threshold:
            break
        msgs = [{"role": "user", "content": lib["rewriter_prompt"].format(text=best_text, canary=spec.canary)}]
        generated = rewriter.generate(msgs, seed=seed + k, max_tokens=max_tokens, temperature=0.8)
        costs["llm_calls"] += 1
        costs["llm_seconds"] += generated.seconds
        costs["prompt_tokens"] += generated.prompt_tokens
        costs["generation_tokens"] += generated.generation_tokens
        cand = strip_rewrite_preamble(generated.text)
        if spec.canary not in cand or (spec.target and spec.target[0] not in cand):
            continue
        s = query(cand)
        used += 1
        if s < best:
            best_text, best = cand, s
    return replace(spec, text=best_text, adaptive_queries=used, adaptive_score=best, adaptive_cost=costs)


def strip_rewrite_preamble(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:[a-zA-Z]+)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = re.sub(r"^(?:Here (?:is|are) (?:the |your |a )?(?:rewritten |revised )?(?:maintenance )?(?:ticket|note)|"
                  r"(?:Rewritten|Revised) (?:maintenance )?(?:ticket|note))\s*:\s*", "", text, flags=re.I)
    text = text.strip()
    if len(text) > 1 and text[0] == text[-1] and text[0] in ('"', "'"):
        text = text[1:-1].strip()
    return text
