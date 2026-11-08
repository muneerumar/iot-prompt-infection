import random

import numpy as np

from iotpi.risk import (finite_horizon_bound, flow_matrices, risk_weights_dv, risk_weights_matrix,
                        spectral_value)
from iotpi.topologies import campus, pilot


def tree7():
    Q = np.zeros((7, 7))
    for j, i in [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)]:
        Q[i, j] = 1.0
    return Q


def test_tree_counterexample():
    """Audit item 2: spectral value 0 ('dies out') while all 7 nodes are infected first."""
    Q = tree7()
    assert spectral_value(Q, 1.0) == 0.0
    p = np.zeros(7)
    p[0] = 1
    ever = p.copy()
    for _ in range(5):   # the plan's recurrence with gamma = 1
        p = (1 - p) * (1 - np.prod(1 - Q * p[None, :], axis=1)) + p * 0.0
        ever = np.maximum(ever, p)
    assert ever.sum() == 7
    b = finite_horizon_bound(Q, np.eye(7)[0], horizon=5, persist=0.0)
    assert b["ever_bound"] == 7.0      # the finite-horizon bound sees the full cascade


def _mc(Q, p0, H, persist, a, reps, rng):
    n = Q.shape[0]
    ever_tot, exec_tot = 0.0, 0
    for _ in range(reps):
        carrying = np.array([rng.random() < p for p in p0])
        ever = carrying.copy()
        executed = bool(any(rng.random() < a[i] for i in range(n) if carrying[i]))
        for _ in range(H):
            new = np.zeros(n, dtype=bool)
            for j in np.flatnonzero(carrying):
                for i in range(n):
                    if Q[i, j] > 0 and rng.random() < Q[i, j]:
                        new[i] = True
            keep = np.array([carrying[i] and rng.random() < persist for i in range(n)])
            carrying = keep | new
            ever |= new
            if any(rng.random() < a[i] for i in range(n) if carrying[i]):
                executed = True
        ever_tot += ever.sum()
        exec_tot += executed
    return ever_tot / reps, exec_tot / reps


def test_bound_dominates_independent_simulation():
    rng = random.Random(1)
    n = 6
    Q = np.zeros((n, n))
    for j in range(n):
        for i in range(n):
            if i != j and rng.random() < 0.4:
                Q[i, j] = rng.uniform(0.05, 0.4)
    a = np.zeros(n)
    a[n - 1] = 0.3
    p0 = np.zeros(n)
    p0[0] = 0.8
    b = finite_horizon_bound(Q, p0, 4, 0.3, a)
    ever, exe = _mc(Q, p0, 4, 0.3, a, 4000, rng)
    assert b["ever_bound"] >= ever - 0.05
    assert b["exec_bound"] >= exe - 0.02


def test_distance_vector_matches_matrix():
    rng = np.random.default_rng(3)
    Q = rng.uniform(0, 0.3, (5, 5)) * (rng.random((5, 5)) < 0.5)
    np.fill_diagonal(Q, 0)
    a = rng.uniform(0, 0.5, 5)
    nodes = list("abcde")
    out = {nodes[j]: {nodes[i]: Q[i, j] for i in range(5) if Q[i, j] > 0} for j in range(5)}
    R_dv, msgs = risk_weights_dv(out, dict(zip(nodes, a)), 6, 0.2)
    R_m = risk_weights_matrix(Q, a, 6, 0.2)
    assert np.allclose([R_dv[x] for x in nodes], R_m)
    assert msgs == 6 * int((Q > 0).sum())


def test_flow_matrices_shapes():
    rates = {r: {"carry_raw": 0.5, "carry_rewritten": 0.3, "persist": 0.2, "target": 0.4} for r in ("gateway", "coordinator")}
    g = pilot(["north", "south"])
    nodes, Q, a, persist = flow_matrices(g, rates, target_zone="north")
    assert Q.shape == (3, 3) and a[nodes.index("gw_north")] > 0 and a[nodes.index("coord")] > 0
    c = campus([f"z{i:02d}" for i in range(12)], 4)
    nodes, Q, a, _ = flow_matrices(c, rates, target_zone="z05")
    assert a[nodes.index("area1")] > 0 and a[nodes.index("area0")] == 0
