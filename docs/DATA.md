# Numeric data

`data/runs/` contains the completed confirmatory records for both model families, the registered primary and secondary analysis outputs, and supporting detector-selection, calibration, exploratory pilot, quarantine-probe, prediction, prompt-selection and timing records. Every exported file has an original/source digest and an export digest in `data/provenance/exported_files.json`.

Each confirmatory model has 320 paired scenarios under six conditions, or 1,920 episode-condition rows: 240 C3 main scenarios, 30 sanity scenarios, 20 adaptive C3 scenarios and 30 benign scenarios. Episodes share `plan_index`, seeds and building IDs across conditions. Do not treat all condition rows as independent replicates.

Primary harm uses only `stratum == "c3_main"` and `attack.executed`. Utility uses all paired episodes with nonzero `utility.expected`. `utility.done_rate` is the per-episode rate. The registered method uses exact two-sided McNemar tests with the specified within-model Holm family, followed by paired building-clustered utility bounds. The implementation and historical registration contain the complete decision rule. Costs retain logical versus physical time and per-tier screening measurements; cache reuse limits independent latency interpretation.

`commands` records simulated device-command events, not independent tasks or physical device actions. `utility.overreach_exec` and `utility.high_impact_overreach` overlap. The latter is a subset, not an additional disjoint count. No actual device or real secret is used.

Calibration messages can share `root_source_id` across message kinds. Merge by that ID before counting distinct pooled sources or resampling. The pooled benign detector data contain 743 measurements from 267 distinct root IDs; summing per-kind source counts gives 591 and overcounts shared roots. A percentile bootstrap of all-zero observations is degenerate; it supplies no informative upper uncertainty bound. The saved calibration allocation constant is preserved unchanged.

Original free-text traces, private templates and model caches are withheld. Numeric summaries are not a replacement for independent inspection of the private text when evaluating semantic labels. Structured episode records preserve judge yes/no response labels alongside scores, flags and cost. Raw message text is omitted from calibration exports.
