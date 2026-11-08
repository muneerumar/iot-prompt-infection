"""Phase 2 (run only after the pilot passes): campus-scale episodes with CalibratedAgents.

    python scripts/run_scale.py --calibration results/<cal>/calibration.json --budgets 0.5 1 2 --name scale_v1

Same engine, scenario generator and guard code as the pilot; only the agents and the screen detector
are replaced by calibrated stand-ins. Claims from this script are conditional on the pilot's
prediction check (RQ2) holding; say so in the paper.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from iotpi.attacks import load_library  # noqa: E402
from iotpi.conditions import conditions  # noqa: E402
from iotpi.detectors import CalibratedDetector, HeuristicDetector  # noqa: E402
from iotpi.experiment import calibrated_agents, risk_weights, run_condition  # noqa: E402
from iotpi.predict import sample_rates  # noqa: E402
from iotpi.report import markdown, summarise  # noqa: E402
from iotpi.runlog import Run  # noqa: E402
from iotpi.scenario import make_scenario  # noqa: E402
from iotpi.topologies import campus  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    ap.add_argument("--calibration", required=True)
    ap.add_argument("--budgets", nargs="+", type=float, default=[1.0])
    ap.add_argument("--conditions", nargs="*", default=["policy_only", "classifier_random", "classifier_taint",
                                                         "classifier_cache", "trustgate", "tg_no_risk_split"])
    ap.add_argument("--episodes", type=int)
    ap.add_argument("--name", default="scale")
    a = ap.parse_args(argv)

    cfg = yaml.safe_load(Path(a.config).read_text())
    s = cfg["scale"]
    if a.episodes:
        s["episodes"] = a.episodes
    cal = json.loads(Path(a.calibration).read_text())
    run = Run(ROOT, a.name, {**cfg, "resolved": {"calibration": a.calibration, "budgets": a.budgets,
                                                 "conditions": a.conditions}})
    if cal.get("backend") != "mlx":
        run.log("WARNING: mock calibration; wiring check only.")
    lib = load_library(ROOT, cfg["attacks"])
    rates, det_rates = sample_rates(cal, None)
    fams = sorted(lib["families"])
    rng = random.Random(cfg["seed"])
    lo, hi = cfg["split"]["evaluation_contexts"]
    for budget in a.budgets:
        conds = conditions(budget, cfg["pilot"].get("cache_distance"))
        rows = []
        for i in range(s["episodes"]):
            fam = fams[i % len(fams)]
            goal = cfg["pilot"]["goals"][(i // len(fams)) % len(cfg["pilot"]["goals"])]
            sc = make_scenario(lo + i % (hi - lo), 200_000 + i, s["zones"], s["horizon"], lib, fam, goal)
            graph = campus(sc.zones, s["zones_per_area"])
            rw = risk_weights(graph, rates, cfg["risk"]["horizon_hops"])
            for name in a.conditions:
                agents = calibrated_agents(graph, rates, rng.randrange(10 ** 9))
                screen = CalibratedDetector(det_rates, seed=rng.randrange(10 ** 9))
                r = run_condition(sc, graph, agents, conds[name], screen, HeuristicDetector(0.5), rw, seed=i)
                r.pop("trace")
                row = {"plan_index": i, "condition": name, "budget": budget, "split": "scale", **r}
                rows.append(row)
                run.record("episodes", row)
            if (i + 1) % 50 == 0:
                run.log(f"budget {budget}: {i + 1}/{s['episodes']} episodes")
        summ = summarise(rows)
        run.save_json(f"summary_budget_{budget}.json", summ)
        (run.dir / f"summary_budget_{budget}.md").write_text(markdown(summ))
        run.log(f"budget {budget} done")
    run.close()
    return run.dir


if __name__ == "__main__":
    main()
