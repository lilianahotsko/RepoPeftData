"""In-process job store + generation worker for the hosted API.

The default :class:`JobStore` is an in-memory, thread-safe dict suitable for a
single-node deployment. For multi-node SaaS, swap it for a queue/DB-backed
implementation with the same interface and run :func:`run_generation_job` from
external workers (e.g. Celery / RQ / a Kubernetes job).
"""

from __future__ import annotations

import threading
import time
import traceback
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Optional

from ..config import C2LConfig, load_config
from ..pipeline import AdapterGenerator, resolve_repo
from ..registry import AdapterRegistry


@dataclass
class Job:
    job_id: str
    repo: str
    task: str
    status: str = "queued"        # queued | running | done | error
    message: str = ""
    fingerprint: str = ""
    adapter_path: str = ""
    repo_id: str = ""
    n_commits_walked: int = 0
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def touch(self, **kw) -> "Job":
        for k, v in kw.items():
            setattr(self, k, v)
        self.updated_at = time.time()
        return self

    def to_dict(self) -> dict:
        return asdict(self)


class JobStore:
    def __init__(self):
        self._jobs: Dict[str, Job] = {}
        self._lock = threading.Lock()

    def create(self, repo: str, task: str) -> Job:
        job = Job(job_id=uuid.uuid4().hex[:16], repo=repo, task=task)
        with self._lock:
            self._jobs[job.job_id] = job
        return job

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)


# Process-wide generator + registry (lazy, shared across jobs).
_GEN_LOCK = threading.Lock()
_GENERATOR: Optional[AdapterGenerator] = None
_REGISTRY: Optional[AdapterRegistry] = None


def _shared(config: C2LConfig):
    global _GENERATOR, _REGISTRY
    with _GEN_LOCK:
        if _GENERATOR is None:
            _GENERATOR = AdapterGenerator(config)
        if _REGISTRY is None:
            _REGISTRY = AdapterRegistry(config=config)
    return _GENERATOR, _REGISTRY


def run_generation_job(job: Job, config: Optional[C2LConfig] = None,
                       work_dir: Optional[str] = None) -> Job:
    """Resolve the repo, generate + register an adapter, update the job."""
    cfg = config or load_config()
    gen, reg = _shared(cfg)
    try:
        job.touch(status="running", message="resolving repository")
        wd = Path(work_dir).expanduser() if work_dir else Path(
            "~/.cache/c2l/api_repos").expanduser()
        wd.mkdir(parents=True, exist_ok=True)

        def _p(msg, frac):
            job.touch(message=f"{msg} ({frac * 100:.0f}%)")

        # Generation is serialized: one heavy model context per process.
        with _GEN_LOCK:
            repo_dir, repo_id = resolve_repo(job.repo, wd, progress=_p)
            adapter = gen.generate(repo_dir, repo_id, task=job.task, progress=_p)
        fp = adapter.fingerprint()
        cached = reg.lookup(fp)
        path = cached if cached is not None else reg.put(adapter)
        job.touch(status="done", message="ready", fingerprint=fp,
                  adapter_path=str(path), repo_id=repo_id,
                  n_commits_walked=adapter.n_commits_walked)
    except Exception as e:
        job.touch(status="error",
                  message=f"{e}\n{traceback.format_exc()[-1500:]}")
    return job


__all__ = ["Job", "JobStore", "run_generation_job"]
