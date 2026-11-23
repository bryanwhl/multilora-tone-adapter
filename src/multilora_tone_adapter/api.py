"""A bounded, local-only rewrite API. Request bodies are never logged here."""

from __future__ import annotations

import secrets
import threading
from typing import Literal
from urllib.parse import urlsplit

from fastapi import FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, field_validator
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .labeling import validate_meaning


class ContextMessage(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    speaker: str = Field(min_length=1, max_length=80)
    text: str = Field(min_length=1, max_length=4000)


class RewriteRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    task: Literal["rewrite"] = "rewrite"
    profile: Literal["telegram"] = "telegram"
    draft: str = Field(min_length=1, max_length=8000)
    context: list[ContextMessage] = Field(default_factory=list, max_length=8)
    max_output_tokens: int = Field(default=128, ge=1, le=512)

    @field_validator("draft")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Draft must contain text.")
        return value


class RewriteResponse(BaseModel):
    output: str
    adapter_version: str
    meaning_validation: str = "Heuristic checks passed; semantic review is still required."


def create_app(engine, adapter_version: str = "base", token: str | None = None) -> FastAPI:
    app = FastAPI(title="Local Telegram tone rewrite", docs_url=None, redoc_url=None)
    app.add_middleware(
        TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1", "[::1]", "testserver"]
    )
    lock = threading.Lock()

    def authorize(request: Request, authorization: str | None) -> None:
        origin = request.headers.get("origin")
        if origin:
            parsed = urlsplit(origin)
            if parsed.scheme not in {"http", "https"} or parsed.hostname not in {
                "localhost",
                "127.0.0.1",
                "::1",
            }:
                raise HTTPException(status_code=403, detail="Browser origin is not local.")
        if token is not None:
            expected = "Bearer " + token
            if not secrets.compare_digest((authorization or "").encode(), expected.encode()):
                raise HTTPException(status_code=401, detail="Authentication required.")

    @app.get("/health")
    def health(request: Request, authorization: str | None = Header(default=None)):
        authorize(request, authorization)
        # Enumerated fields prevent an engine implementation leaking private paths.
        return {
            "status": "ok",
            "loaded": bool(engine.loaded),
            "backend": "mlx",
            "adapter_version": adapter_version,
            "busy": lock.locked(),
        }

    @app.post("/v1/rewrite", response_model=RewriteResponse)
    def rewrite(
        body: RewriteRequest, request: Request, authorization: str | None = Header(default=None)
    ):
        authorize(request, authorization)
        if not lock.acquire(blocking=False):
            raise HTTPException(
                status_code=429, detail="Model is busy; retry later.", headers={"Retry-After": "1"}
            )
        try:
            try:
                output = engine.rewrite(
                    body.draft,
                    [row.model_dump() for row in body.context],
                    max_output_tokens=body.max_output_tokens,
                )
            except ValueError:
                raise HTTPException(
                    status_code=422, detail="Input or output exceeds local model limits."
                )
            except Exception:  # noqa: BLE001 - sanitize backend errors at HTTP boundary
                raise HTTPException(
                    status_code=503, detail="Local model unavailable or memory guard stopped it."
                )
            if not isinstance(output, str) or len(output) > 16000:
                raise HTTPException(status_code=422, detail="Model returned invalid output.")
            failures = validate_meaning(body.draft, output)
            if failures:
                raise HTTPException(
                    status_code=422, detail={"reason": "Meaning checks failed.", "checks": failures}
                )
            return RewriteResponse(output=output, adapter_version=adapter_version)
        finally:
            lock.release()

    return app
