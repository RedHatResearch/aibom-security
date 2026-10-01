"""FastAPI app: fire-and-forget verify submissions.

``POST /verify`` accepts a Hugging Face model id, queues a verify run, and
returns ``202`` with the run id immediately. Results are not served over
HTTP — they land in the server's runs directory (see
:mod:`aibom_server.jobs`).
"""

from __future__ import annotations

import queue
import re
import secrets

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, field_validator

from aibom_server.jobs import JobRunner

_MODEL_ID_RE = re.compile(r"[\w.-]+(?:/[\w.-]+)?")


class VerifyRequest(BaseModel):
    model_id: str = Field(min_length=1, max_length=200)

    @field_validator("model_id")
    @classmethod
    def _validate_model_id(cls, value: str) -> str:
        value = value.strip()
        if not _MODEL_ID_RE.fullmatch(value):
            raise ValueError("model_id must be a Hugging Face repo id, e.g. org/model")
        return value


def create_app(*, runner: JobRunner, token: str | None = None) -> FastAPI:
    """Build the app around a started :class:`~aibom_server.jobs.JobRunner`."""
    app = FastAPI(title="aibom verify server", version="0.1.0")

    @app.post("/verify", status_code=202)
    def verify(payload: VerifyRequest, request: Request) -> dict[str, str]:
        _authorize(request, token)
        try:
            job = runner.submit(payload.model_id)
        except queue.Full:
            raise HTTPException(
                status_code=503, detail="verify queue is full; retry later"
            ) from None
        return {"run_id": job.run_id, "status": "queued"}

    @app.get("/healthz")
    def healthz() -> dict[str, object]:
        return {"status": "ok", "pending": runner.pending()}

    return app


def _authorize(request: Request, token: str | None) -> None:
    if token is None:
        return
    scheme, _, credentials = (request.headers.get("authorization") or "").partition(" ")
    if scheme != "Bearer" or not secrets.compare_digest(credentials.encode(), token.encode()):
        raise HTTPException(status_code=401, detail="missing or invalid bearer token")
