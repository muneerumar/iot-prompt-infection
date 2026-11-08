# Development

Run `python -m pytest -q` before proposing a change. Keep tests deterministic, with fixed seeds and synthetic fixtures. Keep experimental results separate from mock smoke tests. Do not tune on held-out families or evaluation buildings. Never expose evaluator ground truth to real guards or real detectors. All budgeted conditions must use the same screening budget.

Record configuration, revision, environment and per-trial measurements for new experiments. Preserve the historical pre-registration and result records. Do not commit private templates, model weights, credentials, raw message traces, local caches or unpublished manuscript files. Release one model before loading another on memory-limited systems.
