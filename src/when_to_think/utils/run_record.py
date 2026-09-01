"""Run records: capture everything needed to reproduce a run (AGENTS.md §9).

A RunRecord bundles the fully-resolved config snapshot with the runtime facts that
config alone cannot capture: git commit + dirty flag, timestamp, a unique run id,
and resolved values discovered at load time (e.g. the model's actual commit hash,
device, and dataset split sizes).

Every §9 field lives here: model_name / tokenizer_name / dataset_name /
dataset_split / seed / generation_config / reasoning_budget / decision_interval /
reward_config / lambda_compute come from `config`; model_revision (resolved),
timestamp, git_commit, and run_id come from the record itself; training_config is
added to `runtime` once M4 exists.
"""

from __future__ import annotations

import json
import subprocess
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from when_to_think.config import ExperimentConfig, config_to_dict


def get_git_commit(repo_dir: str | Path | None = None) -> str | None:
    """Current HEAD commit hash, or None if not in a git repo."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def get_git_dirty(repo_dir: str | Path | None = None) -> bool | None:
    """True if the working tree has uncommitted changes; None if not a git repo.

    Recorded because a result produced from a dirty tree is not reproducible from
    the commit hash alone — the run record must say so.
    """
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            check=True,
        )
        return bool(out.stdout.strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def generate_run_id(name: str) -> str:
    """Human-readable, unique run id: `<name>-<utc-timestamp>-<short-uuid>`."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{name}-{timestamp}-{uuid.uuid4().hex[:8]}"


@dataclass
class RunRecord:
    run_id: str
    timestamp: str
    git_commit: str | None
    git_dirty: bool | None
    config: dict[str, Any]
    runtime: dict[str, Any] = field(default_factory=dict)


def create_run_record(
    cfg: ExperimentConfig,
    *,
    run_id: str | None = None,
    repo_dir: str | Path | None = None,
    runtime: dict[str, Any] | None = None,
) -> RunRecord:
    """Build a RunRecord from a resolved config plus runtime facts."""
    return RunRecord(
        run_id=run_id or generate_run_id(cfg.name),
        timestamp=datetime.now(timezone.utc).isoformat(),
        git_commit=get_git_commit(repo_dir),
        git_dirty=get_git_dirty(repo_dir),
        config=config_to_dict(cfg),
        runtime=runtime or {},
    )


def run_dir_for(record: RunRecord, base_output_dir: str | Path) -> Path:
    """Per-run output directory `<base_output_dir>/<run_id>` (not created)."""
    return Path(base_output_dir) / record.run_id


def write_run_record(record: RunRecord, base_output_dir: str | Path) -> Path:
    """Write `run_record.json` into the run directory; return that directory."""
    directory = run_dir_for(record, base_output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "run_record.json").write_text(json.dumps(asdict(record), indent=2))
    return directory
