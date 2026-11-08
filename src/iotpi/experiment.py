"""Glue shared by the scripts: prompts, agents, risk weights, episode plans and one-condition runs."""
from __future__ import annotations

import random
from pathlib import Path

from .agents import CalibratedAgent, LLMAgent
from .engine import run_episode
from .guard import GuardNetwork
from .risk import flow_matrices, risk_weights_dv
from .scenario import make_scenario

PROMPT_FILES = ["role_gateway", "task_gateway", "role_coordinator", "task_coordinator", "spotlight_note"]


def load_prompts(root: Path) -> dict:
    return {k: (root / "configs" / "prompts" / f"{k}.txt").read_text() for k in PROMPT_FILES}


def llm_agents(graph, backend, prompts, gen, prompt_defence="none") -> dict:
    return {n: LLMAgent(n, graph.role[n], backend, prompts, gen, prompt_defence) for n in graph.agents}


def calibrated_agents(graph, rates, seed) -> dict:
    rng = random.Random(seed)
    return {n: CalibratedAgent(n, graph.role[n], rates, rng) for n in graph.agents}


def risk_weights(graph, rates, h: int) -> dict:
    """Distance-vector risk weights (sensitivity of the execution bound to each node carrying)."""
    nodes, Q, a, persist = flow_matrices(graph, rates)
    out_edges = {nodes[j]: {nodes[i]: Q[i, j] for i in range(len(nodes)) if Q[i, j] > 0} for j in range(len(nodes))}
    R, _ = risk_weights_dv(out_edges, {n: a[k] for k, n in enumerate(nodes)}, h,
                           {n: persist[k] for k, n in enumerate(nodes)})
    return R


def episode_plan(cfg: dict, lib: dict, which: str = "evaluation") -> list[dict]:
    """Deterministic list of episodes: every (family, goal) cell, then benign episodes.
    Contexts cycle through the requested split so cells are spread over buildings."""
    lo, hi = cfg["split"][f"{which}_contexts"]
    ctxs = list(range(lo, hi))
    p = cfg["pilot"]
    plan, k = [], 0
    for fam in sorted(lib["families"]):
        for goal in p["goals"]:
            for _ in range(p["episodes_per_cell"]):
                plan.append({"context": ctxs[k % len(ctxs)], "episode": k, "family": fam, "goal": goal,
                             "heldout_family": fam in lib["split"]["heldout"]})
                k += 1
    for _ in range(p["benign_episodes"]):
        plan.append({"context": ctxs[k % len(ctxs)], "episode": k, "family": None, "goal": None,
                     "heldout_family": False})
        k += 1
    return plan


def build_scenario(cfg, lib, item):
    w = cfg["world"]
    return make_scenario(item["context"], item["episode"], w["n_zones"], w["horizon"], lib,
                         item["family"], item["goal"])


def run_condition(sc, graph, agents, gcfg, screen, cheap, rw, seed, reset_every=None, keep_prompts=False):
    if screen is not None:
        screen.calls, screen.seconds = 0, 0.0
        screen.observations = []
    if cheap is not None:
        cheap.calls, cheap.seconds = 0, 0.0
        cheap.observations = []
    guard = GuardNetwork(gcfg, graph, screen if gcfg.screen != "none" else None, cheap, rw, seed=seed)
    return run_episode(sc, graph, agents, guard, seed, reset_every=reset_every, keep_prompts=keep_prompts)
