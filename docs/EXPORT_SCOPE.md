# Distribution scope

This is a public snapshot, not the original pre-registration checkout. Source algorithms and numeric outcomes are retained. Report locations and model paths are made portable; test configurations point to explicitly synthetic fixtures. `PUBLIC_EXPORT.json` records each changed source/configuration file and its original and export digest.

All 28 core modules, experiment scripts and the complete existing test suite are included. The registered primary-analysis implementation is unchanged. The preregistration document is retained byte for byte; its historical hashes identify the original inputs, not the portable configuration copies. Original and exported data digests are in `data/provenance/exported_files.json`.

The unpublished manuscript and every manuscript artifact are excluded. Private experimental templates, raw agent messages/generations, caches, internal development records, original Git history and machine-local state are excluded. Result exports retain structured outcomes, utility, commands, costs, screening budgets and detector measurements. Free-text message bodies are omitted where present; structured episode records, including classifier yes/no response labels, are preserved; row counts and removed field names are recorded. The two confirmatory episode files retain their original bytes.

The archived result files support re-analysis without model loading. Generating the original experiments also requires the private attack corpus, licensed model files and the original tagged provenance checkout. The synthetic fixtures cannot substitute for that corpus and must never be presented as producing the archived experimental results.

Historical workflow, repair and development scripts are included for implementation completeness. They refer to original private run histories and require that complete working checkout; they are not the public re-analysis entry point. Public re-analysis uses `scripts/reproduce_results.py`. The public checksum manifest validates this distribution, not unpublished original inputs. No new pre-registration or original Git tag is claimed by this export.

No open-source license has been selected for this snapshot. Third-party model and dependency licenses remain applicable; model weights are not redistributed.
