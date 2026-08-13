"""Append-only experiment log.

Every training or evaluation run in this project must be recorded here —
including failures, dead ends, and hyperparameter-search trials. The total
entry count is the denominator for the multiple-testing adjustment in the
final report, so an unlogged run silently biases the headline result.

The log is a JSON-lines file (one entry per line) so entries are append-only
and diff-friendly. Entries are never edited or deleted; corrections are new
entries with a note.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

DEFAULT_LOG_PATH = Path(__file__).resolve().parents[2] / "experiments.jsonl"

_REQUIRED_FIELDS = (
    "timestamp",
    "git_sha",
    "config_hash",
    "data_snapshot_hash",
    "seed",
    "metrics",
    "notes",
)


def config_hash(config: Mapping[str, Any]) -> str:
    """Deterministic hash of a run configuration.

    Canonical JSON (sorted keys, no whitespace variance) so logically equal
    configs hash equal regardless of dict ordering.
    """
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _current_git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).resolve().parent,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


@dataclass(frozen=True)
class RunRecord:
    timestamp: str
    git_sha: str
    config_hash: str
    data_snapshot_hash: str
    seed: int
    metrics: dict[str, Any]
    notes: str
    config: dict[str, Any]

    @classmethod
    def from_json(cls, line: str) -> "RunRecord":
        raw = json.loads(line)
        missing = [f for f in _REQUIRED_FIELDS if f not in raw]
        if missing:
            raise ValueError(f"experiment log entry missing fields: {missing}")
        return cls(**{k: raw[k] for k in (*_REQUIRED_FIELDS, "config")})


def log_run(
    config: Mapping[str, Any],
    metrics: Mapping[str, Any],
    *,
    seed: int,
    data_snapshot_hash: str,
    notes: str = "",
    log_path: Path = DEFAULT_LOG_PATH,
    git_sha: str | None = None,
) -> RunRecord:
    """Append one run to the experiment log and return the record written.

    ``data_snapshot_hash`` is mandatory: a run that can't name the exact data
    it saw is not reproducible and must not be logged as if it were. Pass the
    content hash from the snapshot manifest (Phase 1).
    """
    if not data_snapshot_hash:
        raise ValueError("data_snapshot_hash is required — see CLAUDE.md Reproducibility")
    record = RunRecord(
        timestamp=_dt.datetime.now(_dt.timezone.utc).isoformat(),
        git_sha=git_sha if git_sha is not None else _current_git_sha(),
        config_hash=config_hash(config),
        data_snapshot_hash=data_snapshot_hash,
        seed=int(seed),
        metrics=dict(metrics),
        notes=notes,
        config=dict(config),
    )
    line = json.dumps(record.__dict__, sort_keys=True, default=str)
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")
    return record


def read_runs(log_path: Path = DEFAULT_LOG_PATH) -> list[RunRecord]:
    if not Path(log_path).exists():
        return []
    with open(log_path, encoding="utf-8") as fh:
        return [RunRecord.from_json(line) for line in fh if line.strip()]


def run_count(log_path: Path = DEFAULT_LOG_PATH) -> int:
    """Total logged runs — the multiple-testing denominator."""
    return len(read_runs(log_path))
