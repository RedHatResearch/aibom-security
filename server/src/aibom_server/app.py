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

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from starlette.types import ASGIApp, Receive, Scope, Send

from aibom_server.jobs import JobRunner

_MODEL_ID_RE = re.compile(r"[A-Za-z0-9][\w.-]{0,95}(?:/[A-Za-z0-9][\w.-]{0,95})?")

# model_id is capped at 200 chars by VerifyRequest; 4KB leaves generous room
# for JSON framing while bounding how much body an unauthenticated client can
# make the server buffer.
MAX_REQUEST_BODY_BYTES = 4096


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
    # Auth and the body cap must run before FastAPI reads the request body,
    # so they live in a pure-ASGI middleware instead of the endpoint.
    app.add_middleware(_VerifyGate, token=token, max_body_bytes=MAX_REQUEST_BODY_BYTES)

    @app.post("/verify", status_code=202)
    def verify(payload: VerifyRequest) -> dict[str, str]:
        """Queue a verify run; auth + body cap happen in ``_VerifyGate``."""
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


def _header(scope: Scope, name: bytes) -> bytes | None:
    for key, value in scope["headers"]:
        if key == name:
            return value
    return None


class _VerifyGate:
    """Pure-ASGI gate for ``POST /verify``.

    Enforces the request-body cap and bearer auth *before* the request body
    is read, so oversized or unauthenticated requests are rejected without
    buffering any body bytes. All other requests (including ``GET /healthz``,
    which stays unauthenticated) pass through unchanged.
    """

    def __init__(self, app: ASGIApp, token: str | None, max_body_bytes: int) -> None:
        self.app = app
        self.token = token
        self.max_body_bytes = max_body_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["method"] != "POST" or scope["path"] != "/verify":
            await self.app(scope, receive, send)
            return
        # Requests without a Content-Length header (e.g. chunked transfer
        # encoding) pass through; anything unparseable is treated as too large.
        content_length = _header(scope, b"content-length")
        if content_length is not None:
            try:
                declared = int(content_length)
            except ValueError:
                declared = self.max_body_bytes + 1
            if declared > self.max_body_bytes:
                response = JSONResponse(
                    status_code=413, content={"detail": "request body too large"}
                )
                await response(scope, receive, send)
                return
        if self.token is not None:
            scheme, _, credentials = (_header(scope, b"authorization") or b"").partition(b" ")
            if scheme != b"Bearer" or not secrets.compare_digest(credentials, self.token.encode()):
                response = JSONResponse(
                    status_code=401,
                    content={"detail": "missing or invalid bearer token"},
                )
                await response(scope, receive, send)
                return
        await self.app(scope, receive, send)
