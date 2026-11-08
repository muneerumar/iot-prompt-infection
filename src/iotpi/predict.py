"""Predict held-out multi-hop pilot outcomes from Stage A calibration, with calibration uncertainty.

For every pilot episode the SAME scenario is replayed with CalibratedAgents and a CalibratedDetector
inside the SAME engine and guard code. Calibration uncertainty is propagated by redrawing every rate
by resampling original carrier-source blocks for v2 (legacy: Beta(k + 1, n - k + 1)) for each of B parameter draws (parametric bootstrap); each draw is
simulated `reps` times. The result is a predictive distribution for each condition's harm rate,
which is compared with the observed pilot rate. The analytic finite-horizon bound is reported for the
unscreened conditions, where its inputs are fully determined by calibration.
"""
from __future__ import annotations

import ast
import random
from collections import defaultdict

import numpy as np

from .detectors import CalibratedDetector, HeuristicDetector
from .experiment import calibrated_agents, run_condition
from .risk import entry_vector, finite_horizon_bound, flow_matrices

ROLE_KEYS = ("carry_raw", "carry_rewritten", "persist", "target", "exfil")


def _beta(rng, k, n):
    return rng.betavariate(k + 1, n - k + 1)


def _draw_rate(rng, row):
    if rng is None:
        return row["k"]/row["n"] if row["n"] else 0.
    groups = row.get("source_counts")
    if not groups:
        return _beta(rng,row["k"],row["n"])
    keys = sorted(groups)
    samples = [groups[rng.choice(keys)] for _ in keys]
    return sum(k for k,n in samples)/sum(n for k,n in samples)


def _pooled_rows(rows):
    groups = defaultdict(lambda:[0,0])
    for row in rows:
        for key,(k,n) in row.get("source_counts",{}).items():
            groups[key][0]+=k;groups[key][1]+=n
    out = {"k":sum(r["k"] for r in rows),"n":sum(r["n"] for r in rows)}
    if groups:
        if sum(n for k,n in groups.values())!=out["n"]:
            raise ValueError("mixed source provenance in prediction")
        out["source_counts"]=dict(groups)
    return out


def sample_rates(cal: dict, rng: random.Random | None) -> tuple[dict, dict]:
    """(agent rates, screen detector rates); point estimates when rng is None."""
    ac = cal["counts"]["agent_counts"]
    fams = sorted(cal.get("training_families", {ast.literal_eval(k)[2] for k in ac} - {"_pooled"}))
    rates = {"gateway": {}, "coordinator": {}}
    for role in rates:
        for key in ROLE_KEYS:
            table, K, N, pooled = {}, 0, 0, []
            for fam in fams:
                row = ac.get(str((role, key, fam)))
                if row and row["n"]:
                    table[fam] = _draw_rate(rng,row)
                    pooled.append(row)
                    K, N = K + row["k"], N + row["n"]
            table["_pooled"] = _draw_rate(rng,_pooled_rows(pooled)) if N else 0.0
            rates[role][key] = table
        lg = ac.get(str((role, "legit", "_pooled")))
        rates[role]["legit"] = (_draw_rate(rng,lg) if lg and lg["n"] else 0.0)
    rates["coordinator"]["carry_raw"] = rates["coordinator"]["carry_rewritten"]

    dc = cal["counts"].get("training_detector_counts", cal["counts"]["detector_counts"])["screen"]
    det = {"raw": {}, "rewritten": {}, "fpr": {}}
    agg = defaultdict(list)
    for key, row in dc.items():
        kind, sub = ast.literal_eval(key)
        if kind != "benign" and sub not in fams:
            continue
        b = {"raw": "raw", "rewritten": "rewritten", "benign": "fpr"}[kind]
        det[b][sub or "_none"] = _draw_rate(rng,row)
        agg[b].append(row)
    for b, rows in agg.items():
        det[b]["_pooled"] = _draw_rate(rng,_pooled_rows(rows))
    return rates, det


def simulate_episode(sc, graph, gcfg, rates, det_rates, rw, reps: int, seed: int, return_samples: bool = False):
    harms = []
    for r in range(reps):
        agents = calibrated_agents(graph, rates, seed * 1009 + r)
        screen = CalibratedDetector(det_rates, seed=seed * 2003 + r)
        res = run_condition(sc, graph, agents, gcfg, screen, HeuristicDetector(0.5), rw, seed=seed * 3001 + r)
        harms.append(bool(res["attack"]["harm"]))
    return harms if return_samples else sum(harms) / reps


def analytic_bound(sc, graph, rates, gcfg, det_rates) -> float | None:
    """Finite-horizon bound for command goals under unscreened guard configs; None elsewhere."""
    atk = sc.attack
    if not atk.target or gcfg.screen != "none" or gcfg.prompt_defence != "none":
        return None
    cls = atk.target_class
    if not gcfg.policy:
        p_exec = 1.0
    elif cls == "C3":
        p_exec = 0.0 if gcfg.taint_block_c3 else 1.0
    else:
        p_exec = 0.0   # C1 is always blocked; the attacked door never has a booking (scenario.py)
    zone = atk.target[0].split("_", 1)[1] if "_" in atk.target[0] else None
    nodes, Q, a, persist = flow_matrices(graph, rates, None, zone if zone in sc.zones else None, p_exec, atk.family)
    p0 = entry_vector(graph, nodes, rates, atk.zone, 0.0, atk.family)
    hops = 2 * (sc.horizon - atk.step)
    return finite_horizon_bound(Q, p0, hops, persist, a)["exec_bound"]


def brier(pred: list[float], obs: list[bool]) -> float:
    return float(np.mean([(p - float(o)) ** 2 for p, o in zip(pred, obs)])) if pred else float("nan")


def predictive_cohort_means(rows: list[dict]) -> np.ndarray:
    """Replicated cohorts preserve both parameter draws and binary episode variability.

    Each parameter draw is shared across the cohort. A replication uses one actual
    simulated binary outcome per observed scenario, rather than averaging away its noise.
    """
    outcomes = np.asarray([r["predictive_outcomes"] for r in rows], dtype=float)
    if outcomes.ndim != 3 or not np.all((outcomes == 0) | (outcomes == 1)):
        raise ValueError("predictive outcomes must have shape episodes x draws x replications and be binary")
    return outcomes.mean(axis=0).ravel()
