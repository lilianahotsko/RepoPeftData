"""Hosted C2L service (FastAPI).

Thin HTTP surface over the SDK:

* ``POST /adapters``            -- enqueue async generation for a repo + task
* ``GET  /adapters/{job_id}``   -- poll job status / result fingerprint
* ``GET  /adapters/by-fp/{fp}`` -- adapter metadata by fingerprint
* ``GET  /adapters/{fp}/download`` -- download the exported PEFT adapter (zip)
* ``POST /predict``             -- run a cached adapter server-side (SaaS)
* ``GET  /tasks`` / ``GET /healthz``

Generation is registry-cached and content-addressed, so the same
repo + commit + task is generated at most once.
"""

from __future__ import annotations

__all__ = ["create_app"]


def __getattr__(name):
    if name == "create_app":
        from .app import create_app
        return create_app
    raise AttributeError(name)
