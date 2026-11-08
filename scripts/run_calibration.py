"""Stage A: single-hop calibration on calibration contexts (never used by the pilot evaluation).

    python scripts/run_calibration.py --backend mlx --model llama-4bit --name cal_llama
Writes calibration.json: per-role agent rates (per family + pooled, with counts and source-clustered CIs (Wilson diagnostics retained)) and
detector rates (raw / rewritten / false-positive), consumed by predict_multihop.py and run_scale.py.
"""
from __future__ import annotations

import argparse
import json
import hashlib
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from iotpi.attacks import load_library  # noqa: E402
from iotpi.backends import make_backend  # noqa: E402
from iotpi.calibrate import MeasuredAgent, detector_rates, rates_from_counts, run_units  # noqa: E402
from iotpi.calibration_resume import CalibrationJournal, ResumableDetector, import_legacy
from iotpi.detectors import make_detector  # noqa: E402
from iotpi.experiment import llm_agents, load_prompts  # noqa: E402
from iotpi.runlog import Run, read_jsonl  # noqa: E402
from iotpi.topologies import pilot  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    ap.add_argument("--backend", choices=["mock", "mlx"], default="mock")
    ap.add_argument("--model")
    ap.add_argument("--units", type=int, help="override calibration.units_per_cell")
    ap.add_argument("--screen", help="override detectors.screen")
    ap.add_argument("--prompt-defence", default=None, help="calibrate under a prompt-level defence")
    ap.add_argument("--mock-follow-prob", type=float, default=0.6)
    ap.add_argument("--name", default="calibration")
    group=ap.add_mutually_exclusive_group()
    group.add_argument("--resume",help="append to a checkpoint-format calibration directory")
    group.add_argument("--resume-from",help="read-only import of an aborted pre-checkpoint run into a new directory")
    a = ap.parse_args(argv)

    source=Path(a.resume or a.resume_from).resolve() if (a.resume or a.resume_from) else None
    if source and any(x is not None for x in (a.units,a.screen,a.prompt_defence,a.model)):
        raise ValueError("resume uses the saved model, units, detector and prompts; no overrides")
    cfg = yaml.safe_load((source/"config.yaml" if source else Path(a.config)).read_text())
    if source:
        if a.resume and (source/"calibration.json").exists() and (source/"metrics.json").exists():
            return source
        if a.resume and any("records" not in row for row in read_jsonl(source/"units.jsonl")):
            raise ValueError("legacy calibration is read-only; use --resume-from into a new directory")
        a.backend=cfg["resolved"]["backend"];a.model=cfg["resolved"]["model_key"]
        a.mock_follow_prob=cfg["resolved"].get("mock_follow_prob",a.mock_follow_prob)
    else:
        if a.units:cfg["calibration"]["units_per_cell"] = a.units
        if a.screen:cfg["detectors"]["screen"] = a.screen
        if a.prompt_defence:cfg["calibration"]["prompt_defence"] = a.prompt_defence
        cfg["resolved"] = {"backend":a.backend,"model_key":a.model or "mock",
            "model_path":cfg["models"].get(a.model) if a.backend=="mlx" else "mock",
            "mock_follow_prob":a.mock_follow_prob}
    inputs=sorted((ROOT/"configs/prompts").glob("*"))+[ROOT/cfg["attacks"]]
    if not inputs[-1].exists():inputs[-1]=ROOT/"private/attacks.example.yaml"
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    previous=cfg["resolved"].get("calibration_input_hashes")
    if previous is not None and previous!=hashes:raise ValueError("calibration prompt or attack inputs changed")
    if not a.resume:cfg["resolved"]["calibration_input_hashes"]=hashes
    raw=Run.reopen(source) if a.resume else Run(ROOT,f"{a.name}_{a.model or 'mock'}",cfg)
    run=CalibrationJournal(raw)
    lib = load_library(ROOT, cfg["attacks"])
    if a.resume_from:
        imported=import_legacy(source,run,cfg["calibration"]["units_per_cell"],sorted(lib["families"]))
        run.save_json("resume_import.json",imported)
        run.log(f"Imported {imported['completed_units_imported']} completed units from read-only source {source}")
    prompts = load_prompts(ROOT)
    backend = (make_backend("mlx", str((ROOT / cfg["models"][a.model]).resolve())) if a.backend == "mlx"
               else make_backend("mock", follow_prob=a.mock_follow_prob))
    if a.backend == "mock":
        run.log("MOCK BACKEND: wiring check only; these numbers are not results.")
    det = cfg["detectors"]
    detectors = {"screen": make_detector(det["screen"], det["screen_threshold"], backend, ROOT, det["device"]),
                 "cheap": make_detector(det["cheap"], 0.5)}
    detectors={name:ResumableDetector(name,d,run) for name,d in detectors.items()}

    def agents_for(sc):
        g = pilot(sc.zones)
        agents = llm_agents(g, backend, prompts, cfg["generation"], cfg["calibration"]["prompt_defence"])
        return g, {n: MeasuredAgent(agent, run, sc) for n, agent in agents.items()}

    try:
        counts = run_units(cfg, lib, agents_for, pilot, detectors, run, cfg["calibration"]["units_per_cell"], cfg["seed"])
        fams = sorted(lib["families"])
        out = {"backend": a.backend, "model": a.model or "mock", "prompt_defence": cfg["calibration"]["prompt_defence"],
               "counts": counts, "rates": rates_from_counts(counts, fams),
               "training_families": list(lib["split"]["train"]),
               "training_rates": rates_from_counts(counts, list(lib["split"]["train"])),
               "detectors": {name: detector_rates(counts, name) for name in detectors},
               "training_detectors": {name: detector_rates({"detector_counts": counts["training_detector_counts"]}, name) for name in detectors},
               "detector_specs": {"screen": det["screen"], "cheap": det["cheap"]}}
        for filename in ("calibration.json","metrics.json"):
            path=run.dir/filename
            if path.exists():
                if json.loads(path.read_text())!=out:raise ValueError("existing final calibration differs")
            else:run.save_json(filename,out)
        run.log(f"Resume checkpoint replay: {run.replayed_units} completed units skipped; {len(run.completed)} total committed units")
        for role, table in out["rates"].items():
            run.log(f"{role}: " + ", ".join(f"{k}={v['_pooled'] if isinstance(v, dict) else v}" for k, v in table.items()))
        run.log(f"screen detector: {out['detectors']['screen']}")
    finally:
        backend.close()
        run.close()
    return run.dir


if __name__ == "__main__":
    main()
