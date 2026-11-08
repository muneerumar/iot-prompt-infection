"""Does single-hop calibration predict held-out multi-hop outcomes? (RQ2)

    python scripts/predict_multihop.py --calibration results/<cal>/calibration.json \
        --pilot results/<pilot_run> --draws 10 --reps 20 --name pred_llama

Calibration contexts and pilot (evaluation) contexts are disjoint by construction (configs/default.yaml).
Outputs per condition and split: observed harm rate, predicted mean with a 95% predictive interval
(parametric bootstrap over calibration counts), Brier score, and the analytic bound where defined.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from iotpi.attacks import load_library  # noqa: E402
from iotpi.conditions import conditions  # noqa: E402
from iotpi.experiment import build_scenario, risk_weights  # noqa: E402
from iotpi.predict import analytic_bound, brier, sample_rates, simulate_episode, predictive_cohort_means  # noqa: E402
from iotpi.report import load_rows  # noqa: E402
from iotpi.runlog import Run  # noqa: E402
from iotpi.scenario import set_attack_text  # noqa: E402
from iotpi.stats import cluster_bootstrap  # noqa: E402
from iotpi.topologies import pilot  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibration", required=True)
    ap.add_argument("--pilot", required=True)
    ap.add_argument("--draws", type=int, default=10)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--conditions", nargs="*")
    ap.add_argument("--name", default="prediction")
    a = ap.parse_args(argv)

    cal = json.loads(Path(a.calibration).read_text())
    pilot_dir = Path(a.pilot)
    cfg = yaml.safe_load((pilot_dir / "config.yaml").read_text())
    rows = [r for r in load_rows(pilot_dir) if r["attack"]]
    traces = {(json.loads(line)["plan_index"]): json.loads(line)["attack_text"]
              for line in (pilot_dir / "traces.jsonl").read_text().splitlines() if line.strip()}
    conds = conditions(cfg["pilot"]["budget_per_node"], cfg["pilot"].get("cache_distance"))
    keep = set(a.conditions or [c for c in cfg["pilot"]["conditions"]
                                if conds[c].prompt_defence == cal.get("prompt_defence", "none")])
    rows = [r for r in rows if r["condition"] in keep]
    run = Run(ROOT, a.name, {"calibration": a.calibration, "pilot": str(pilot_dir), "draws": a.draws,
                             "reps": a.reps, "conditions": sorted(keep), "pilot_config": cfg})
    if cal.get("backend") != "mlx" or cfg["resolved"]["backend"] != "mlx":
        run.log("WARNING: mock calibration or pilot; wiring check only.")
    lib = load_library(ROOT, cfg["attacks"])
    point_rates, point_det = sample_rates(cal, None)
    rng = random.Random(cfg["seed"])
    draws = [sample_rates(cal, rng) for _ in range(a.draws)]

    per_row = []
    for i, r in enumerate(rows):
        item = {k: r[k] for k in ("context", "episode", "family", "goal")}
        sc = build_scenario(cfg, lib, item)
        if r.get("adaptive") and traces.get(r["plan_index"]):
            sc.attack.text = traces[r["plan_index"]]
            set_attack_text(sc, sc.attack)
        graph = pilot(sc.zones)
        rw = risk_weights(graph, cfg["resolved"].get("risk_rates", cfg["risk"]["prior_rates"]),
                          cfg["risk"]["horizon_hops"])
        gcfg = conds[r["condition"]]
        samples = [simulate_episode(sc, graph, gcfg, rates, det, rw, a.reps, seed=i * 31 + d, return_samples=True)
                   for d, (rates, det) in enumerate(draws)]
        preds = [float(np.mean(sample)) for sample in samples]
        bound = analytic_bound(sc, graph, point_rates, gcfg, point_det)
        row = {"plan_index": r["plan_index"], "condition": r["condition"], "split": r["split"],
               "goal": r["attack"]["goal"], "context_id": r["context_id"], "observed": bool(r["attack"]["harm"]),
               "pred_mean": float(np.mean(preds)), "pred_draws": preds, "predictive_outcomes": samples, "bound": bound}
        per_row.append(row)
        run.record("predictions", row)
        if (i + 1) % 25 == 0:
            run.log(f"{i + 1}/{len(rows)} episodes predicted")

    summary = {}
    groups = defaultdict(list)
    for row in per_row:
        groups[(row["condition"], "all")].append(row)
        groups[(row["condition"], row["split"])].append(row)
    for (cond, split), rs in sorted(groups.items()):
        obs = cluster_bootstrap([float(x["observed"]) for x in rs], [x["context_id"] for x in rs])
        draw_means = np.mean(np.array([x["pred_draws"] for x in rs]), axis=0)   # one mean per parameter draw
        cohort_means = predictive_cohort_means(rs)
        brs = [x for x in rs if x["bound"] is not None]
        bounds = [x["bound"] for x in brs]
        obs_b = (sum(x["observed"] for x in brs) / len(brs)) if brs else None
        summary[f"{cond}|{split}"] = {
            "n": len(rs), "observed": obs, "predicted_mean": float(np.mean([x["pred_mean"] for x in rs])),
            "predicted_95": [float(np.percentile(cohort_means, 2.5)), float(np.percentile(cohort_means, 97.5))],
            "parameter_mean_95": [float(np.percentile(draw_means, 2.5)), float(np.percentile(draw_means, 97.5))],
            "predictive_replications": len(cohort_means),
            "brier": brier([x["pred_mean"] for x in rs], [x["observed"] for x in rs]),
            "mean_bound": float(np.mean(bounds)) if bounds else None,
            "observed_on_bound_episodes": obs_b,
            "observed_le_bound": (obs_b <= float(np.mean(bounds))) if bounds else None,
        }
    run.save_json("prediction_summary.json", summary)
    run.save_json("metrics.json", summary)
    lines = ["| Condition | Split | n | Observed harm [95% CI] | Predicted [95% predictive] | Brier | Bound, command goals |",
             "|---|---|---|---|---|---|---|"]
    for key, s in summary.items():
        cond, split = key.split("|")
        o = s["observed"]
        ci = f" [{o[1]:.3f}, {o[2]:.3f}]" if o[1] is not None else ""
        mb = ("n/a" if s["mean_bound"] is None
              else f"{s['mean_bound']:.3f} (observed {s['observed_on_bound_episodes']:.3f})")
        lines.append(f"| {cond} | {split} | {s['n']} | {o[0]:.3f}{ci} | {s['predicted_mean']:.3f} "
                     f"[{s['predicted_95'][0]:.3f}, {s['predicted_95'][1]:.3f}] | {s['brier']:.3f} | {mb} |")
    (run.dir / "prediction_summary.md").write_text("\n".join(lines))
    run.log(f"done -> {run.dir / 'prediction_summary.md'}")
    run.close()
    return run.dir


if __name__ == "__main__":
    main()
