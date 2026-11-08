"""Pilot: real LLM agents in the smart-building workflow, every condition on the same episodes.

    python scripts/run_pilot.py --backend mlx --model llama-4bit --name pilot_llama
    python scripts/run_pilot.py --resume results/<run_dir>          # continue after an interruption
    python scripts/run_pilot.py --backend mock --limit 10 --name smoke  # wiring check only

Rows go to episodes.jsonl (labels + costs) and traces.jsonl (every agent output, prompts with
--keep-prompts). summary.json / summary.md are recomputed from the saved rows at the end.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from dataclasses import replace

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from iotpi.attacks import adaptive_rewrite, load_library  # noqa: E402
from iotpi.backends import CachingBackend, make_backend  # noqa: E402
from iotpi.conditions import conditions  # noqa: E402
from iotpi.detectors import make_detector  # noqa: E402
from iotpi.experiment import (build_scenario, episode_plan, llm_agents, load_prompts, risk_weights,  # noqa: E402
                              run_condition)
from iotpi.report import load_rows, markdown, summarise  # noqa: E402
from iotpi.runlog import Run, read_jsonl  # noqa: E402
from iotpi.scenario import set_attack_text  # noqa: E402
from iotpi.topologies import pilot  # noqa: E402


def adaptive_items(cfg, lib, start_index):
    a = cfg["pilot"]["adaptive"]
    lo, hi = cfg["split"]["evaluation_contexts"]
    ctxs = list(range(lo, hi))
    items, k = [], 0
    for fam in a["families"]:
        for i in range(a["episodes_per_family"]):
            goal = cfg["pilot"]["goals"][i % len(cfg["pilot"]["goals"])]
            items.append({"context": ctxs[k % len(ctxs)], "episode": 70_000 + k, "family": fam, "goal": goal,
                          "heldout_family": fam in lib["split"]["heldout"], "adaptive": True})
            k += 1
    return items


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    ap.add_argument("--backend", choices=["mock", "mlx"], default="mock")
    ap.add_argument("--model")
    ap.add_argument("--conditions", nargs="*")
    ap.add_argument("--limit", type=int, help="first N plan items only (timing / smoke)")
    ap.add_argument("--no-adaptive", action="store_true")
    ap.add_argument("--keep-prompts", action="store_true")
    ap.add_argument("--screen", help="override detectors.screen, e.g. heuristic for a no-download smoke run")
    ap.add_argument("--mock-follow-prob", type=float, default=0.6)
    ap.add_argument("--split", choices=["calibration", "evaluation"], default="evaluation")
    ap.add_argument("--attack-only", action="store_true")
    ap.add_argument("--disable-cache", action="store_true", help="disable exact generation reuse")
    ap.add_argument("--risk-rates", help="measured calibration.json for risk allocation")
    ap.add_argument("--resume", help="existing run directory to continue")
    ap.add_argument("--name", default="pilot")
    a = ap.parse_args(argv)

    if a.resume:
        run_dir = Path(a.resume)
        cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
        run = Run.reopen(run_dir)
    else:
        cfg = yaml.safe_load(Path(a.config).read_text())
        if a.screen:
            cfg["detectors"]["screen"] = a.screen
        if a.conditions:
            cfg["pilot"]["conditions"] = a.conditions
        if a.no_adaptive:
            cfg["pilot"]["adaptive"]["enabled"] = False
        cfg["resolved"] = {"backend": a.backend, "model_key": a.model or "mock",
                           "model_path": cfg["models"].get(a.model) if a.backend == "mlx" else "mock",
                           "limit": a.limit, "keep_prompts": a.keep_prompts}
        cfg["resolved"].update(split=a.split, attack_only=a.attack_only, disable_cache=a.disable_cache, mock_follow_prob=a.mock_follow_prob)
        if a.split == "calibration":
            cfg["pilot"]["adaptive"]["enabled"] = False
        if a.risk_rates:
            rate_path = Path(a.risk_rates).resolve()
            cal = json.loads(rate_path.read_text())
            if a.backend == "mlx" and (cal.get("backend") != "mlx" or cal.get("model") != a.model):
                raise ValueError("real pilot requires calibration from the same real model")
            cfg["resolved"].update(risk_rates_path=str(rate_path),
                risk_rates_sha256=hashlib.sha256(rate_path.read_bytes()).hexdigest(),
                risk_rates=cal.get("training_rates", cal["rates"]), risk_source="calibration")
        else:
            used = any(conditions(cfg["pilot"]["budget_per_node"])[n].screen == "risk"
                       for n in cfg["pilot"]["conditions"])
            if a.backend == "mlx" and used and a.name != "timing" and a.split != "calibration":
                raise ValueError("real risk-screened pilot requires --risk-rates; priors are smoke/timing only")
            cfg["resolved"].update(risk_rates=cfg["risk"]["prior_rates"], risk_source="assumed-smoke-prior")
        run = Run(ROOT, f"{a.name}_{a.model or 'mock'}", cfg)
    res = cfg["resolved"]
    run.recover_episodes()
    done = {(r["plan_index"], r["condition"]) for r in load_rows(run.dir)}
    saved_attacks = {r["plan_index"]: r for r in read_jsonl(run.dir / "adaptive_attacks.jsonl")}
    if done:
        run.log(f"resuming: {len(done)} (episode, condition) rows already saved")

    lib = load_library(ROOT, cfg["attacks"])
    prompts = load_prompts(ROOT)
    backend = (make_backend("mlx", str((ROOT / res["model_path"]).resolve())) if res["backend"] == "mlx"
               else make_backend("mock", follow_prob=res.get("mock_follow_prob", a.mock_follow_prob)))
    if res["backend"] == "mock":
        run.log("MOCK BACKEND: wiring check only; these numbers are not results.")
    backend = CachingBackend(backend, run.dir / "llm_cache.jsonl", enabled=not res.get("disable_cache", False))
    det = cfg["detectors"]
    screen = make_detector(det["screen"], det["screen_threshold"], backend, ROOT, det["device"])
    cheap = make_detector(det["cheap"], 0.5)
    conds = conditions(cfg["pilot"]["budget_per_node"], cfg["pilot"].get("cache_distance"))
    names = cfg["pilot"]["conditions"]
    plan = episode_plan(cfg, lib, res.get("split", "evaluation"))
    if res.get("split") == "calibration":
        plan = [p for p in plan if p["family"] is None or p["family"] in lib["split"]["train"]]
        # Interleave training cells so a short strength check covers families and goals.
        cells = {}
        for item in plan:
            cells.setdefault((item["family"], item["goal"]), []).append(item)
        plan = [items[i] for i in range(max(map(len, cells.values()), default=0))
                for items in cells.values() if i < len(items)]
    if res.get("attack_only"):
        plan = [p for p in plan if p["family"] is not None]
    if cfg["pilot"]["adaptive"]["enabled"]:
        plan += adaptive_items(cfg, lib, len(plan))
    if res.get("limit"):
        plan = plan[: res["limit"]]
    run.log(f"{len(plan)} episodes x {len(names)} conditions; screen={det['screen']}")

    try:
        for idx, item in enumerate(plan):
            if all((idx, c) in done for c in names):
                continue
            sc = build_scenario(cfg, lib, item)
            if item.get("adaptive") and sc.attack:
                if idx in saved_attacks:
                    saved = saved_attacks[idx]
                    spec = replace(sc.attack, text=saved["text"], adaptive_queries=saved["queries"],
                                   adaptive_score=saved["score"], adaptive_cost=saved.get("cost", {}))
                else:
                    spec = adaptive_rewrite(sc.attack, lib, backend, screen, cfg["pilot"]["adaptive"]["queries"],
                                            det["screen_threshold"], seed=cfg["seed"] + idx)
                    saved = {"plan_index": idx, "text": spec.text, "queries": spec.adaptive_queries,
                             "score": spec.adaptive_score, "cost": spec.adaptive_cost}
                    run.record("adaptive_attacks", saved)
                    saved_attacks[idx] = saved
                set_attack_text(sc, spec)
            graph = pilot(sc.zones)
            rw = risk_weights(graph, res.get("risk_rates", cfg["risk"]["prior_rates"]), cfg["risk"]["horizon_hops"])
            split = "adaptive" if item.get("adaptive") else ("heldout" if item["heldout_family"] else "train")
            if not sc.attack:
                split = "benign"
            for name in names:
                if (idx, name) in done:
                    continue
                gcfg = conds[name]
                agents = llm_agents(graph, backend, prompts, cfg["generation"], gcfg.prompt_defence)
                r = run_condition(sc, graph, agents, gcfg, screen, cheap, rw, seed=cfg["seed"] * 7919 + idx,
                                  reset_every=cfg["world"]["reset_every"], keep_prompts=res.get("keep_prompts", False))
                trace = r.pop("trace")
                run.record_episode(
                    {"plan_index": idx, "condition": name, "split": split, **item, **r},
                    {"plan_index": idx, "condition": name, "trace": trace,
                     "condition_seed": cfg["seed"] * 7919 + idx,
                     "scenario_seed": sc.seed, "context_id": sc.context_id,
                     "attack_text": sc.attack.text if sc.attack else None})
                done.add((idx, name))
                h = r["attack"]["harm"] if r["attack"] else None
                run.log(f"[{idx + 1}/{len(plan)}] {split:8s} {str(item['family']):10s} {str(item['goal']):16s} "
                        f"{name:18s} harm={h} legit={r['utility']['done']}/{r['utility']['expected']} "
                        f"llm_s={r['cost'].get('seconds', 0):.1f}")
    finally:
        backend.close()
        summary = summarise(load_rows(run.dir))
        summary["backend"] = res["backend"]
        complete = all((i, c) in done for i in range(len(plan)) for c in names)
        if complete:
            target = run.dir / "summary.json"
            if target.exists():
                if json.loads(target.read_text()) != json.loads(json.dumps(summary)):
                    raise ValueError("completed summary differs; preserving the original")
            else:
                run.save_json("summary.json", summary)
                (run.dir / "summary.md").write_text(markdown(summary))
            if not (run.dir / "metrics.json").exists():
                run.save_json("metrics.json", summary)
        else:
            partial = f"partial_summary_{len(list(run.dir.glob('partial_summary_*.json')))}.json"
            run.save_json(partial, summary)
        run.log(f"complete={complete}; LLM cache hits {backend.hits}, misses {backend.misses}")
        run.close()
    return run.dir


if __name__ == "__main__":
    main()
