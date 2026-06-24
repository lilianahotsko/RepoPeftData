"""FastAPI application for the hosted C2L service.

Run::

    pip install -e .[api]
    uvicorn c2l.api.app:app --host 0.0.0.0 --port 8000

Endpoints are documented in :mod:`c2l.api`. Heavy inference is optional; the
core product is *generate an adapter, return a tiny portable artifact* that the
user can run anywhere (including locally on a quantized base).
"""

from __future__ import annotations

import io
import os
import time
import uuid
import zipfile
from pathlib import Path
from typing import List, Optional

from ..config import load_config
from ..registry import AdapterRegistry
from .jobs import JobStore, run_generation_job


def create_app():
    from fastapi import BackgroundTasks, FastAPI, HTTPException
    from fastapi.responses import StreamingResponse
    from pydantic import BaseModel

    cfg = load_config()
    app = FastAPI(title="Code2LoRA", version="0.1.0")
    jobs = JobStore()
    registry = AdapterRegistry(config=cfg)

    # ---- OpenAI-compatible bridge (so Cursor / any OpenAI client can use C2L) ----
    # Configured via env so the same server can serve a chosen adapter:
    #   C2L_OPENAI_BACKEND   hf | 4bit | 8bit          (default: 4bit)
    #   C2L_OPENAI_ADAPTER   path to an exported PEFT adapter dir (optional)
    #   C2L_OPENAI_FP        adapter fingerprint to look up in the registry (optional)
    #   C2L_OPENAI_MODEL     advertised model id        (default: c2l-coder-4bit)
    oai_backend = os.environ.get("C2L_OPENAI_BACKEND", "4bit")
    oai_model_id = os.environ.get("C2L_OPENAI_MODEL", f"c2l-coder-{oai_backend}")
    _backend_cache: dict = {}

    def _resolve_adapter_dir() -> Optional[str]:
        d = os.environ.get("C2L_OPENAI_ADAPTER")
        if d:
            return d
        fp = os.environ.get("C2L_OPENAI_FP")
        if fp:
            p = registry.lookup(fp)
            return str(p) if p else None
        return None

    def _get_backend(backend: str, adapter_dir: Optional[str]):
        key = (backend, adapter_dir or "")
        be = _backend_cache.get(key)
        if be is None:
            from ..infer import make_backend
            be = make_backend(backend, adapter_dir=adapter_dir, config=cfg)
            _backend_cache[key] = be
        return be

    def _messages_to_prefix(messages: List[dict]) -> str:
        """Flatten an OpenAI chat history into a single completion prefix.

        C2L is a code-completion model, not a chat model, so we render the
        conversation as plain text and let it continue from the last turn.
        """
        parts: List[str] = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if isinstance(content, list):  # OpenAI "parts" form
                content = "".join(
                    p.get("text", "") for p in content
                    if isinstance(p, dict) and p.get("type") == "text")
            if role == "system":
                parts.append(str(content))
            elif role == "assistant":
                parts.append(str(content))
            else:
                parts.append(str(content))
        return "\n".join(p for p in parts if p).rstrip() + "\n"

    class AdaptRequest(BaseModel):
        repo: str
        task: str = "assert_rhs"

    class PredictRequest(BaseModel):
        prefix: str
        fingerprint: Optional[str] = None
        adapter_path: Optional[str] = None
        backend: str = "hf"           # hf | 4bit | 8bit
        max_new_tokens: int = 32
        use_adapter: bool = True

    @app.get("/healthz")
    def healthz():
        return {"status": "ok", "base_model": cfg.base_model}

    @app.get("/tasks")
    def list_tasks():
        from .. import tasks as T
        return {"tasks": [
            {"id": t, "index": T.task_index(t),
             "description": T.get_task(t).description}
            for t in T.list_tasks()]}

    @app.post("/adapters")
    def create_adapter(req: AdaptRequest, bg: BackgroundTasks):
        from .. import tasks as T
        if req.task not in T.list_tasks():
            raise HTTPException(400, f"unknown task {req.task!r}")
        job = jobs.create(req.repo, req.task)
        bg.add_task(run_generation_job, job, cfg)
        return {"job_id": job.job_id, "status": job.status}

    @app.get("/adapters/{job_id}")
    def get_job(job_id: str):
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "unknown job")
        return job.to_dict()

    @app.get("/adapters/by-fp/{fingerprint}")
    def get_by_fp(fingerprint: str):
        path = registry.lookup(fingerprint)
        if path is None:
            raise HTTPException(404, "adapter not found")
        meta = Path(path) / "c2l_adapter.json"
        import json
        return json.loads(meta.read_text()) if meta.exists() else {"path": str(path)}

    @app.get("/adapters/{fingerprint}/download")
    def download(fingerprint: str):
        path = registry.lookup(fingerprint)
        if path is None:
            raise HTTPException(404, "adapter not found")
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in Path(path).iterdir():
                if f.is_file():
                    zf.write(f, arcname=f.name)
        buf.seek(0)
        return StreamingResponse(
            buf, media_type="application/zip",
            headers={"Content-Disposition":
                     f"attachment; filename={fingerprint}.zip"})

    @app.post("/predict")
    def predict(req: PredictRequest):
        from ..infer import make_backend

        adapter_dir = req.adapter_path
        if adapter_dir is None and req.fingerprint:
            p = registry.lookup(req.fingerprint)
            if p is None:
                raise HTTPException(404, "adapter not found")
            adapter_dir = str(p)
        if req.backend not in ("hf", "4bit", "8bit"):
            raise HTTPException(400, "predict supports hf/4bit/8bit backends")
        be = make_backend(req.backend, adapter_dir=adapter_dir, config=cfg)
        text = be.generate(req.prefix, max_new_tokens=req.max_new_tokens,
                           use_adapter=req.use_adapter)
        return {"prediction": text, "adapter": adapter_dir or "(base only)"}

    # ---------------- OpenAI-compatible endpoints ----------------

    @app.get("/v1/models")
    def list_models():
        return {"object": "list", "data": [
            {"id": oai_model_id, "object": "model", "created": int(time.time()),
             "owned_by": "code2lora"}]}

    class ChatMessage(BaseModel):
        role: str
        content: object = ""

    class ChatRequest(BaseModel):
        model: Optional[str] = None
        messages: List[ChatMessage]
        max_tokens: Optional[int] = None
        temperature: Optional[float] = None
        stream: bool = False

    @app.post("/v1/chat/completions")
    def chat_completions(req: ChatRequest):
        import json

        adapter_dir = _resolve_adapter_dir()
        be = _get_backend(oai_backend, adapter_dir)
        prefix = _messages_to_prefix([m.model_dump() for m in req.messages])
        max_new = int(req.max_tokens or 64)
        text = be.generate(prefix, max_new_tokens=max_new)

        cid = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())
        model_name = req.model or oai_model_id

        if not req.stream:
            return {
                "id": cid, "object": "chat.completion", "created": created,
                "model": model_name,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0,
                          "total_tokens": 0},
            }

        def _sse():
            head = {"id": cid, "object": "chat.completion.chunk",
                    "created": created, "model": model_name,
                    "choices": [{"index": 0, "delta": {"role": "assistant"},
                                 "finish_reason": None}]}
            yield f"data: {json.dumps(head)}\n\n"
            # Emit the generated text as a single content delta (C2L returns the
            # whole continuation at once; we chunk it for client compatibility).
            body = {"id": cid, "object": "chat.completion.chunk",
                    "created": created, "model": model_name,
                    "choices": [{"index": 0, "delta": {"content": text},
                                 "finish_reason": None}]}
            yield f"data: {json.dumps(body)}\n\n"
            tail = {"id": cid, "object": "chat.completion.chunk",
                    "created": created, "model": model_name,
                    "choices": [{"index": 0, "delta": {},
                                 "finish_reason": "stop"}]}
            yield f"data: {json.dumps(tail)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(_sse(), media_type="text/event-stream")

    return app


# Importing this module gives uvicorn an ``app`` target. We build it lazily so
# that ``import c2l.api.app`` does not require fastapi unless the app is created.
class _LazyApp:
    _app = None

    def __getattr__(self, item):
        if _LazyApp._app is None:
            _LazyApp._app = create_app()
        return getattr(_LazyApp._app, item)


app = _LazyApp()

__all__ = ["create_app", "app"]
