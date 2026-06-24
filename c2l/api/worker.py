"""Standalone generation worker.

Use this as the task body for an external queue (Celery / RQ / k8s Job) or run
it directly to generate + register one adapter without the HTTP layer::

    python -m c2l.api.worker --repo https://github.com/psf/cachecontrol \
            --task assert_rhs

It shares the same :func:`run_generation_job` path the API uses, so results are
identical and registry-cached.
"""

from __future__ import annotations

import argparse
import json
import sys

from ..config import load_config
from .jobs import Job, run_generation_job


def generate_and_register(repo: str, task: str = "assert_rhs",
                          config=None) -> dict:
    """Generate + register one adapter; return the finished job as a dict."""
    cfg = config or load_config()
    job = Job(job_id="worker", repo=repo, task=task)
    run_generation_job(job, cfg)
    return job.to_dict()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--task", default="assert_rhs")
    ap.add_argument("--config")
    args = ap.parse_args(argv)
    result = generate_and_register(args.repo, args.task,
                                   config=load_config(args.config))
    json.dump(result, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0 if result.get("status") == "done" else 1


if __name__ == "__main__":
    raise SystemExit(main())
