"""Finite-horizon risk analysis (primary) and the spectral threshold (supporting only).

Why finite-horizon: the spectral condition rho(diag(gamma)^-1 M) < 1 only says that infection
eventually dies out. It says nothing about how many agents are hit first. On a directed out-tree
with certain transmission and one-step recovery, rho = 0 yet every node is infected
(see tests/test_risk.py::test_tree_counterexample). Unsafe actuation happens inside that transient.

Bound used here (union bound over space-time transmission chains, assuming transmission events on
distinct (edge, step) pairs are independent or negatively correlated):

    W_0 = p0,   W_t = persist * W_{t-1} + Q W_{t-1}
    E[# agents ever carrying by H]  <=  sum_i min(1, p0_i + sum_{t=1..H} (Q W_{t-1})_i)
    P(unauthorised execution by H) <=  min(1, sum_{t=0..H} a . W_t)

Q[i, j] is the per-step probability that a carrying j makes i carry (message rate x pass rate x
(1 - screening detection)); a_i is the per-step probability that a carrying i gets the target
executed. The independence assumption can fail (shared payloads, shared models, memory), so the
pilot tests the bound against held-out multi-hop episodes rather than assuming it.

Risk weights for budget allocation are the bound's sensitivity to node i carrying,
R = sum_{t=0..h} ((persist I + Q)^T)^t a, computed by distance-vector rounds in which every node only
uses its downstream neighbours' current values.
"""
from __future__ import annotations

import numpy as np


def finite_horizon_bound(Q: np.ndarray, p0: np.ndarray, horizon: int, persist: float | np.ndarray = 0.0,
                         a: np.ndarray | None = None) -> dict:
    n = Q.shape[0]
    persist = np.broadcast_to(np.asarray(persist, dtype=float), (n,))
    W = p0.astype(float).copy()
    ever = p0.astype(float).copy()
    exec_mass = float(a @ W) if a is not None else 0.0
    per_step = [W.copy()]
    for _ in range(horizon):
        new = Q @ W
        ever += new
        W = persist * W + new
        per_step.append(W.copy())
        if a is not None:
            exec_mass += float(a @ W)
    return {"ever_bound": float(np.minimum(1.0, ever).sum()), "ever_raw": ever,
            "exec_bound": min(1.0, exec_mass) if a is not None else None, "W": per_step}


def risk_weights_matrix(Q: np.ndarray, a: np.ndarray, h: int, persist: float | np.ndarray = 0.0) -> np.ndarray:
    n = Q.shape[0]
    P = np.diag(np.broadcast_to(np.asarray(persist, dtype=float), (n,))) + Q
    R = a.astype(float).copy()
    for _ in range(h):
        R = a + P.T @ R
    return R


def risk_weights_dv(out_edges: dict, a: dict, h: int, persist: dict | float = 0.0) -> tuple[dict, int]:
    """Distance-vector form. out_edges[i] = {j: Q[j, i]} (i's downstream neighbours and the chance
    i passes a carry to j). Each round, every node reads its downstream neighbours' last value.
    Returns (weights, messages exchanged)."""
    pers = persist if isinstance(persist, dict) else {i: persist for i in a}
    R = dict(a)
    msgs = 0
    for _ in range(h):
        new = {}
        for i in a:
            downstream = sum(q * R[j] for j, q in out_edges.get(i, {}).items())
            msgs += len(out_edges.get(i, {}))
            new[i] = a[i] + pers.get(i, 0.0) * R[i] + downstream
        R = new
    return R, msgs


def spectral_value(M: np.ndarray, gamma: float | np.ndarray) -> float:
    """rho(diag(gamma)^-1 M); below 1 means eventual extinction (supporting analysis only)."""
    g = np.broadcast_to(np.asarray(gamma, dtype=float), (M.shape[0],))
    if np.any(g <= 0):
        return float("inf") if M.any() else 0.0
    return float(max(abs(np.linalg.eigvals(M / g[:, None]))))


# ------------------------------------------------------------------ flow-graph helpers

def _val(rates, role, key, family=None):
    v = rates[role][key]
    if isinstance(v, dict):
        return v.get(family, v.get("_pooled", 0.0))
    return float(v)


def flow_matrices(graph, rates: dict, detect: dict | None = None, target_zone: str | None = None,
                  p_exec_given_issue: float = 1.0, family: str | None = None):
    """Q, a and persistence for a FlowGraph from calibrated per-role rates (all conditional on exposure).

    Q[dst, src]  = carry rate of dst's role on agent-written input x (1 - detect[dst])
    a[src]       = per-step chance that src carrying gets the target executed: src reports to the
                   coordinator that commands `target_zone`, which issues the target with its 'target'
                   rate after passing screening, and the actuator executes it with p_exec_given_issue
                   (1 for an unguarded C3 target, 0 for C1, the evidence pass rate for C2).
    detect[node] = chance the guard at node screens and catches an attack-carrying input.
    """
    nodes = list(graph.agents)
    idx = {n: k for k, n in enumerate(nodes)}
    n = len(nodes)
    Q, a, persist = np.zeros((n, n)), np.zeros(n), np.zeros(n)
    detect = detect or {}
    commander = graph.commander_of(target_zone) if target_zone else None
    for src in nodes:
        for dst, _kind in graph.out_edges(src):
            Q[idx[dst], idx[src]] = _val(rates, graph.role[dst], "carry_rewritten", family) * (1 - detect.get(dst, 0.0))
            if dst == commander or (commander is None and graph.commands_for.get(dst)):
                a[idx[src]] += (_val(rates, "coordinator", "target", family) * (1 - detect.get(dst, 0.0))
                                * p_exec_given_issue)
        persist[idx[src]] = _val(rates, graph.role[src], "persist", family)
    # a commanding coordinator that carries the payload in memory is exposed again next step
    for c in ([commander] if commander else [n for n in nodes if graph.commands_for.get(n)]):
        a[idx[c]] += _val(rates, "coordinator", "target", family) * p_exec_given_issue
    return nodes, Q, a, persist


def entry_vector(graph, nodes, rates: dict, zone: str, detect_raw: float = 0.0, family: str | None = None):
    """p0: the attacked zone's gateway carries after reading the raw ticket."""
    p0 = np.zeros(len(nodes))
    p0[nodes.index(graph.gateway(zone))] = _val(rates, "gateway", "carry_raw", family) * (1 - detect_raw)
    return p0
