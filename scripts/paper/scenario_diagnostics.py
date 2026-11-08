#!/usr/bin/env python3
"""Secondary diagnostics that need the episode scenarios (expected-work windows, benign tickets).

Run from the project root on the machine that holds private/attacks.yaml:
    PYTHONPATH=src python scripts/paper/scenario_diagnostics.py --root .

This script requires the withheld private attack file; the public synthetic fixture is not a substitute.
The committed data/paper/scenario_diagnostics.json is its saved aggregate output.

Scenarios are rebuilt deterministically from the frozen plan (no model is loaded). The script
checks that every rebuilt scenario reproduces the saved expected-work count, then writes
data/paper/scenario_diagnostics.json. No attack text is written.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

RUNS = {
    "llama": "data/runs/20261106-122410_round2c_20261106-122409_confirm_llama-4bit",
    "qwen": "data/runs/20261107-023314_round2c_20261106-122409_confirm_qwen-4bit",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    a = ap.parse_args()
    root = Path(a.root).resolve()
    sys.path.insert(0, str(root / "src"))
    from iotpi.attacks import load_library
    from iotpi.round2 import build_round2_scenario, confirmatory_plan

    cfg = yaml.safe_load((root / "configs/round2.yaml").read_text())
    private_input = root / "private/attacks.yaml"
    if not private_input.is_file():
        ap.error("requires the withheld private/attacks.yaml; use the committed aggregate JSON for public number regeneration")
    lib = load_library(root, "private/attacks.yaml")
    plan = confirmatory_plan(cfg, lib)
    horizon = cfg["world"]["horizon"]
    scen = {i: build_round2_scenario(cfg, lib, item) for i, item in enumerate(plan)}

    # benign tickets that reach a gateway in the attack-free episodes (identical for both models)
    benign_tickets = sum(1 for i, item in enumerate(plan) if item["stratum"] == "benign"
                         for t in scen[i].tickets if t.step < horizon)

    out = {"benign_episode_ticket_messages": benign_tickets, "models": {}}
    for m, d in RUNS.items():
        mismatches = 0
        hold = {"events": 0, "events_in_window": 0, "episodes": 0}
        benign_ticket_flags_screen_all = 0
        for line in (root / d / "episodes.jsonl").open():
            r = json.loads(line)
            sc = scen[r["plan_index"]]
            if len(sc.expected) != r["utility"]["expected"]:
                mismatches += 1
            if r["condition"] == "classifier_all" and r["stratum"] == "benign":
                benign_ticket_flags_screen_all += sum(1 for e in r["guard_events"]
                                                      if e[0] == "detect" and str(e[4]).startswith("ticket"))
            if r["condition"] != "trustgate":
                continue
            pairs = {(e.device, e.command) for e in sc.expected}
            hits = [c for c in r["commands"] if c["reason"] == "alert_hold" and (c["device"], c["command"]) in pairs]
            hold["events"] += len(hits)
            hold["episodes"] += bool(hits)
            hold["events_in_window"] += sum(
                1 for c in hits if any(e.device == c["device"] and e.command == c["command"]
                                       and e.due <= c["step"] <= e.deadline for e in sc.expected))
        if mismatches:
            raise SystemExit(f"{m}: {mismatches} rebuilt scenarios disagree with saved expected work")
        out["models"][m] = {"trustgate_alert_hold_on_expected_pairs": hold,
                            "screen_all_benign_ticket_flags": benign_ticket_flags_screen_all}
    dest = root / "data/paper/scenario_diagnostics.json"
    dest.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out))


if __name__ == "__main__":
    main()
