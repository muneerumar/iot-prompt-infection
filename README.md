# TrustGate

Budgeted local containment of prompt-injection cascades in simulated multi-agent IoT workflows.

This repository contains the simulator, containment runtime, detector routing, experiment harness, tests and numeric data for a two-model confirmatory evaluation. The unpublished manuscript is not included. This is a portable public snapshot; the original private experiment checkout and its Git history are not distributed.

## Install and test

Python 3.11, 3.12 and 3.13 are supported by the test matrix. Re-analysis and mock tests require no model files.

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m pytest -q
python scripts/reproduce_results.py --verify
```

`requirements.lock` records the original full Apple Silicon environment after removal of its machine-local editable-install entry. It includes platform-specific packages; use the package extras for installation on other systems. For local Apple Silicon generation, the optional extras are `.[mac,classifier,dev]`. Model weights are not included; configure locally available, appropriately licensed models under `models/` or update the model paths.

## Reproduce the archived analysis

```sh
python scripts/reproduce_results.py --verify
python scripts/reproduce_results.py --output results/public_reanalysis
```

Both commands validate the exported-file digests and paired row coverage. The registered primary analysis is computed before the secondary tables. `--verify` compares both outputs with the archived JSON without creating result files. `--output` writes derived outputs into a new directory. Neither command loads a language model or needs the private attack corpus.

Python 3.11 with the versions constrained by `requirements.lock` reproduces the archived JSON exactly. Newer stacks agree within `math.isclose(rel_tol=1e-9, abs_tol=1e-12)` for floats. Dictionary keys, list lengths, integer counts, strings, nulls and every Boolean decision (`holds`, `wins`, `cross_model_win`) must match exactly. An integer and a float may match only when their values are exactly equal; paired NaNs are equal. The receipt reports `max_abs_float_deviation` separately for primary and secondary outputs. Dataset and generator-file digests remain exact SHA-256 checks.

The cross-model confirmatory criterion was not met. The data and scripts retain that outcome; secondary observations must not be presented as replacing the registered decision. See the historical [pre-registration](docs/preregistration_round2.md) and [data documentation](docs/DATA.md).

## Regenerate the manuscript numbers

```sh
python -m pip install -e ".[dev,paper]"
python scripts/paper/make_numbers.py --root . --out results/paper_numbers
```

`--out` is required and must be new or empty. The script reads public numeric data and the saved aggregate diagnostics, independently verifies the primary results, recounts the exploratory pilot, and produces nine TeX files, a derived-data JSON and three vector figures. The scripts contain table captions, but the manuscript itself is not included. Generated outputs are ignored by version control and must not be committed. Figure PDF bytes can vary with Matplotlib versions.

`scripts/paper/scenario_diagnostics.py` requires the withheld private `private/attacks.yaml` to rebuild scenarios; the public synthetic test fixture cannot substitute for that input. Its saved output is committed as `data/paper/scenario_diagnostics.json`, which contains aggregate counts only. Public number regeneration uses that saved output and needs neither private templates nor a model. Source and export hashes for both scripts and the aggregate JSON are recorded in the provenance manifests.

The locked validation environment uses Python 3.11 and `requirements.lock` constraints, with Matplotlib 3.11.2 added for figures:

```sh
python -m pip install -c requirements.lock -e ".[dev,paper]" "matplotlib==3.11.2"
```

## Code and data

| Path | Contents |
| --- | --- |
| `src/iotpi/` | Simulated scenarios and topology, agents, authorization, egress floor, screening, caches, alerts, quarantine, lineage holds, calibration, risk and statistics |
| `scripts/` | Experiment, resume, analysis and historical workflow implementations; `reproduce_results.py` is the public re-analysis entry point |
| `scripts/paper/` | Number/figure generator and private-input scenario diagnostic; manuscript text is excluded |
| `configs/` | Portable settings and benign role/task/judge prompts |
| `tests/` | Complete existing regression suite and labelled synthetic test fixtures |
| `data/runs/` | Both confirmatory datasets and supporting calibration, selection, exploratory and analysis records |
| `data/paper/` | Saved aggregate scenario diagnostics used by the number generator |
| `data/provenance/` | Exact original/export hashes for data and generator files, and historical pre-registration identifiers |
| `PUBLIC_EXPORT.json` | Distribution scope and source adaptations |
| `SHA256SUMS` | Checksums for every distributed file other than this checksum file itself |

## Mock smoke test

```sh
python scripts/run_pilot.py --backend mock --screen heuristic --limit 6 --no-adaptive --name smoke
```

The supplied library is an explicitly synthetic software fixture. Mock outputs are wiring checks, not research findings. Re-generating the original experiments requires the withheld private corpus, model weights and original provenance checkout. Historical controllers require those original run histories; do not use them as the public re-analysis entry point.

## Runtime and limitations

Tickets use ProtectAI v2 or the local judge; reports and notices use ProtectAI only, at threshold 0.5. One screened message costs one budget unit, regardless of which tiers run. The judge shares the agents' model, so failures may be correlated. Agent-reworded C3 requests can resemble legitimate requests, which limits screening; downstream containment uses lineage holds. These are simulated commands and observed finite-horizon outcomes, not a physical-safety guarantee.

See [distribution scope](docs/EXPORT_SCOPE.md) for what is withheld and how public paths differ from the historical checkout. No open-source license has been selected for this snapshot. Third-party models and dependencies retain their own licenses.
