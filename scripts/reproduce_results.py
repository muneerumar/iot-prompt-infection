"""Recompute archived primary results, then secondary tables, without model loading."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import yaml
from iotpi.attacks import load_library
from iotpi.round2 import confirmatory_plan
from iotpi.round2_analysis import both_models_primary
from round2_confirm_support import validate_rows
from analyze_round2 import secondary_tables

RUNS = {
    'llama-4bit': '20261106-122410_round2c_20261106-122409_confirm_llama-4bit',
    'qwen-4bit': '20261107-023314_round2c_20261106-122409_confirm_qwen-4bit',
}
ANALYSIS = '20261107-183408_round2_confirmatory_analysis'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_exports():
    index = json.loads((ROOT / 'data/provenance/exported_files.json').read_text())
    seen = set()
    for entry in index:
        relative = Path(entry['path'])
        if relative.is_absolute() or '..' in relative.parts or str(relative) in seen:
            raise ValueError('Invalid or duplicate data path')
        seen.add(str(relative))
        path = ROOT / relative
        if not path.is_file() or digest(path) != entry['export_sha256']:
            raise ValueError('Data digest mismatch: ' + str(relative))
    if not index:
        raise ValueError('Empty data manifest')
    return len(index)


def load_completed():
    cfg = yaml.safe_load((ROOT / 'configs/round2.yaml').read_text())
    # Only family identifiers/splits are needed to rebuild the registered plan.
    # Synthetic fixture text is never used to generate experimental outcomes here.
    plan = confirmatory_plan(cfg, load_library(ROOT, cfg['attacks']))
    model_rows, health = {}, {}
    for model, folder in RUNS.items():
        path = ROOT / 'data/runs' / folder / 'episodes.jsonl'
        rows = [json.loads(line) for line in path.open() if line.strip()]
        health[model] = validate_rows(rows, plan, complete=True)
        if health[model]['budget_violations']:
            raise ValueError('Screening budget violation: ' + model)
        model_rows[model] = rows
    return model_rows, health


def reproduce(output=None):
    verified_files = validate_exports()
    model_rows, health = load_completed()
    archived = ROOT / 'data/runs' / ANALYSIS
    primary = both_models_primary(model_rows)
    if primary != json.loads((archived / 'primary.json').read_text()):
        raise ValueError('Primary results differ from the archived analysis')
    target = Path(output) if output is not None else None
    if target is not None:
        target.mkdir(parents=True, exist_ok=False)
        (target / 'primary.json').write_text(json.dumps(primary, indent=2) + '\n')
    # The primary result has been computed, checked and optionally persisted first.
    secondary = secondary_tables(model_rows)
    if secondary != json.loads((archived / 'secondary.json').read_text()):
        raise ValueError('Secondary tables differ from the archived analysis')
    receipt = {
        'verified_data_files': verified_files,
        'health': health,
        'primary_matches_archive': True,
        'secondary_matches_archive': True,
        'cross_model_win': primary['cross_model_win'],
        'model_wins': {key: value['wins'] for key, value in primary['models'].items()},
        'model_generations': 0,
    }
    if target is not None:
        (target / 'secondary.json').write_text(json.dumps(secondary, indent=2) + '\n')
        (target / 'verification.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--verify', action='store_true')
    group.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(reproduce(args.output), indent=2))


if __name__ == '__main__':
    main()
