# TrustGate

Budgeted local containment of prompt-injection cascades in simulated multi-agent IoT workflows.

This repository contains the simulator, containment runtime, detector routing, experiment harness, tests and numeric data for a two-model confirmatory evaluation. The unpublished manuscript is not included. This is a portable public snapshot; the original private experiment checkout and its Git history are not distributed.

## Install and test

Python 3.11 is the tested interpreter. Re-analysis and mock tests require no model files.

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

Both commands validate the dataset digests and paired row coverage. The registered primary analysis is computed before the secondary tables. `--verify` compares both outputs with the archived JSON without creating result files. `--output` writes derived outputs into a new directory. Neither command loads a language model or needs the private attack corpus.

The cross-model confirmatory criterion was not met. The data and scripts retain that outcome; secondary observations must not be presented as replacing the registered decision. See the historical [pre-registration](docs/preregistration_round2.md) and [data documentation](docs/DATA.md).

## Code and data

| Path | Contents |
| --- | --- |
| `src/iotpi/` | Simulated scenarios and topology, agents, authorization, egress floor, screening, caches, alerts, quarantine, lineage holds, calibration, risk and statistics |
| `scripts/` | Experiment, resume, analysis and historical workflow implementations; `reproduce_results.py` is the public re-analysis entry point |
| `configs/` | Portable settings and benign role/task/judge prompts |
| `tests/` | Complete existing regression suite and labelled synthetic test fixtures |
| `data/runs/` | Both confirmatory datasets and supporting calibration, selection, exploratory and analysis records |
| `data/provenance/` | Original/export hashes and historical pre-registration identifiers |
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
