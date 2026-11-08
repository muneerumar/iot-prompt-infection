"""Traceable run directories: results/<timestamp>_<name>/ with configuration, revision, environment and measurements."""
from __future__ import annotations

import json
import os
import shutil
import platform
import subprocess
import sys
import time
from pathlib import Path

import yaml


class Run:
    def __init__(self, root: Path, name: str, config: dict, argv: list[str] | None = None):
        stamp = time.strftime("%Y%m%d-%H%M%S")
        results_root = Path(os.environ.get("IOTPI_RESULTS_DIR", str(root / "results")))
        base = results_root / f"{stamp}_{name}"
        self.dir, k = base, 1
        while self.dir.exists():
            self.dir, k = base.with_name(f"{base.name}-{k}"), k + 1
        self.dir.mkdir(parents=True, exist_ok=False)
        (self.dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
        meta = {
            "name": name,
            "project_root": str(root.resolve()),
            "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "argv": argv or sys.argv,
            "git_commit": _git(root),
            "python": sys.version,
            "platform": platform.platform(),
        }
        (self.dir / "meta.json").write_text(json.dumps(meta, indent=2))
        uv = shutil.which("uv")
        command = [uv, "pip", "freeze", "--python", sys.executable] if uv else [sys.executable, "-m", "pip", "freeze"]
        freeze = subprocess.run(command, capture_output=True, text=True)
        if freeze.returncode:
            from importlib.metadata import distributions
            freeze.stdout = "\n".join(sorted(f"{d.metadata['Name']}=={d.version}" for d in distributions())) + "\n"
        (self.dir / "pip_freeze.txt").write_text(freeze.stdout or freeze.stderr)
        self._log = open(self.dir / "run.log", "a")
        self._jsonl = {}

    @classmethod
    def reopen(cls, run_dir: Path) -> "Run":
        """Continue an interrupted run in place; appends to its logs and JSONL files."""
        self = cls.__new__(cls)
        self.dir = Path(run_dir)
        meta = json.loads((self.dir / "meta.json").read_text())
        self._log = open(self.dir / "run.log", "a")
        self._jsonl = {}
        self.record("resume_events", {"at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "argv": sys.argv,
                    "git_commit": _git(Path(meta.get("project_root", self.dir.parents[1])))})
        return self

    def log(self, msg: str):
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        self._log.write(line + "\n")
        self._log.flush()

    def record(self, stream: str, row: dict):
        if stream not in self._jsonl:
            path = self.dir / f"{stream}.jsonl"
            # Preserve crash fragments; start the next record on a fresh line.
            if path.exists() and path.stat().st_size:
                with path.open("rb") as f:
                    f.seek(-1, 2)
                    tail = f.read(1)
                if tail != b"\n":
                    with path.open("ab") as f:
                        f.write(b"\n")
            self._jsonl[stream] = open(path, "a")
        self._jsonl[stream].write(json.dumps(row) + "\n")
        self._jsonl[stream].flush()
        os.fsync(self._jsonl[stream].fileno())

    def save_json(self, filename: str, obj):
        path = self.dir / filename
        with path.open("x") as f:
            json.dump(obj, f, indent=2)

    def record_episode(self, episode: dict, trace: dict):
        """Durable canonical transaction, followed by append-only projections."""
        self.record("transactions", {"episode": episode, "trace": trace})
        self.record("traces", trace)
        self.record("episodes", episode)

    def recover_episodes(self):
        key = lambda row: (row["plan_index"], row["condition"])
        episodes = {key(r) for r in read_jsonl(self.dir / "episodes.jsonl")}
        traces = {key(r) for r in read_jsonl(self.dir / "traces.jsonl")}
        for transaction in read_jsonl(self.dir / "transactions.jsonl"):
            episode, trace = transaction["episode"], transaction["trace"]
            k = key(episode)
            if k not in traces:
                self.record("traces", trace)
                traces.add(k)
            if k not in episodes:
                self.record("episodes", episode)
                episodes.add(k)

    def close(self):
        for f in self._jsonl.values():
            f.close()
        self._log.close()


def _git(root: Path) -> str:
    r = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else "not-a-git-repo"


def read_jsonl(path: Path):
    """Read complete append-only records, warning about preserved crash fragments."""
    import warnings
    if not path.exists():
        return []
    rows = []
    for n, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            warnings.warn(f"Preserved incomplete JSONL record: {path}:{n}", RuntimeWarning)
    return rows
