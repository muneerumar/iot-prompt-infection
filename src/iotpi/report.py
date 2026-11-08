"""Summaries computed only from saved episode rows (AGENTS.md rule 1)."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from .stats import cluster_bootstrap, holm, mcnemar_exact
from .runlog import read_jsonl


def load_rows(run_dir: Path) -> list[dict]:
    p = Path(run_dir) / "episodes.jsonl"
    return read_jsonl(p)


def _rate(rows, key_fn):
    vals = [float(key_fn(r)) for r in rows]
    return cluster_bootstrap(vals, [r["context_id"] for r in rows])


def summarise(rows: list[dict], reference: str = "trustgate") -> dict:
    by_cond = defaultdict(list)
    for r in rows:
        attack = r.get("attack")
        if attack:
            expected = (attack["exfiltrated"] if attack["goal"] == "exfil" else
                        bool(attack["spread_gateways"]) if attack["goal"] == "spread" else attack["executed"])
            if bool(attack["harm"]) != bool(expected):
                raise ValueError("harm label disagrees with the goal-specific outcome")
        by_cond[r["condition"]].append(r)
    out = {"conditions": {}, "paired": {}}
    for cond, rs in sorted(by_cond.items()):
        atk = [r for r in rs if r["attack"]]
        ben = [r for r in rs if not r["attack"]]
        d = {"n_attack": len(atk), "n_benign": len(ben)}
        if atk:
            d["harm"] = _rate(atk, lambda r: r["attack"]["harm"])
            d["generated"] = _rate(atk, lambda r: r["attack"]["generated"])
            d["spread"] = _rate(atk, lambda r: bool(r["attack"]["spread_gateways"]))
            for split in ("train", "heldout", "adaptive"):
                sub = [r for r in atk if r["split"] == split]
                if sub:
                    d[f"harm_{split}"] = _rate(sub, lambda r: r["attack"]["harm"])
            goals = sorted({r["attack"]["goal"] for r in atk})
            d["harm_by_goal"] = {g: _rate([r for r in atk if r["attack"]["goal"] == g], lambda r: r["attack"]["harm"])
                                 for g in goals}
        allr = [r for r in rs if r["utility"]["expected"]]
        if allr:
            d["legit_done_rate"] = _rate(allr, lambda r: r["utility"]["done"] / r["utility"]["expected"])
        d["false_drops_per_episode"] = sum(r["utility"]["false_drops"] for r in rs) / len(rs)
        d["false_quarantines_per_episode"] = sum(r["utility"]["false_quarantines"] for r in rs) / len(rs)
        d["screens_per_episode"] = sum(r["cost"].get("screens", 0) for r in rs) / len(rs)
        d["screen_seconds_per_episode"] = sum(r["cost"].get("screen_seconds", 0) for r in rs) / len(rs)
        d["llm_seconds_per_episode"] = sum(r["cost"].get("seconds", 0) for r in rs) / len(rs)
        d["alert_bytes_per_episode"] = sum(r["cost"].get("alert_bytes", 0) for r in rs) / len(rs)
        out["conditions"][cond] = d

    if reference in by_cond:
        ref = {(r["plan_index"]): r for r in by_cond[reference] if r["attack"]}
        pvals = {}
        for cond, rs in by_cond.items():
            if cond == reference:
                continue
            pairs = [(ref[r["plan_index"]]["attack"]["harm"], r["attack"]["harm"])
                     for r in rs if r["attack"] and r["plan_index"] in ref]
            if pairs:
                res = mcnemar_exact([a for a, _ in pairs], [b for _, b in pairs])
                res["n_pairs"] = len(pairs)
                out["paired"][cond] = res
                pvals[cond] = res["p"]
        for cond, p in holm(pvals).items():
            out["paired"][cond]["p_holm"] = p
    return out


def markdown(summary: dict) -> str:
    def f(t):
        if not t or t[0] is None:
            return "TBD"
        est, lo, hi = t
        return f"{est:.3f}" + (f" [{lo:.3f}, {hi:.3f}]" if lo is not None else "")
    lines = ["| Condition | Harm (all) | Harm (held-out families) | Harm (adaptive) | Legit work done | "
             "False drops/ep | Screens/ep | vs TrustGate (Holm p) |", "|" + "---|" * 8]
    for cond, d in summary["conditions"].items():
        pr = summary["paired"].get(cond, {})
        lines.append(f"| {cond} | {f(d.get('harm'))} | {f(d.get('harm_heldout'))} | {f(d.get('harm_adaptive'))} | "
                     f"{f(d.get('legit_done_rate'))} | {d['false_drops_per_episode']:.2f} | "
                     f"{d['screens_per_episode']:.1f} | {pr.get('p_holm', float('nan')):.3g} |")
    lines.append("\nIntervals: 95% percentile bootstrap resampling contexts. Paired test: exact McNemar on harm, "
                 "same episodes, Holm-adjusted across conditions.")
    return "\n".join(lines)
