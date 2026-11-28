"""A bounded, local-only rewrite API. Request bodies are never logged here."""

from __future__ import annotations

import copy
import secrets
import threading
from typing import Literal
from urllib.parse import urlsplit

from fastapi import FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, field_validator
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .labeling import validate_meaning
from .rewrite_review import fallback, finalize, make_request


class RewriteRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    task: Literal["rewrite"] = "rewrite"
    profile: Literal["telegram"] = "telegram"
    draft: str = Field(min_length=1, max_length=8000)
    max_output_tokens: int = Field(default=512, ge=1, le=512)

    @field_validator("draft")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Draft must contain text.")
        return value


class RewriteResponse(BaseModel):
    output: str
    adapter_version: str
    status: Literal["rewritten", "fallback_original"]
    reason: str
    meaning_validation: str


class CandidateResponse(BaseModel):
    candidate: str
    adapter_version: str
    request_hash: str
    language_scope: Literal["english", "multilingual"]
    status: Literal["requires_semantic_review"] = "requires_semantic_review"
    protected_check_failures: list[str]


def create_app(
    engine, adapter_version: str = "base", token: str | None = None, *, semantic_reviewer=None,
    language_scope: str = "english",
) -> FastAPI:
    if language_scope not in {"english", "multilingual"}:
        raise ValueError("Invalid qualification language scope")
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
            "semantic_reviewer_configured": semantic_reviewer is not None,
            "language_scope": language_scope,
        }

    def generate(body: RewriteRequest) -> str:
        if not lock.acquire(blocking=False):
            raise HTTPException(
                status_code=429, detail="Model is busy; retry later.", headers={"Retry-After": "1"}
            )
        try:
            try:
                output = engine.rewrite(
                    body.draft,
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
            if not isinstance(output, str) or not output.strip() or len(output) > 16000:
                raise HTTPException(status_code=422, detail="Model returned invalid output.")
            return output
        finally:
            lock.release()

    @app.post("/v1/rewrite/candidate", response_model=CandidateResponse)
    def candidate(
        body: RewriteRequest, request: Request, authorization: str | None = Header(default=None)
    ):
        authorize(request, authorization)
        output = generate(body)
        review = make_request(body.draft, output, adapter_version, language_scope=language_scope)
        return CandidateResponse(
            candidate=output, adapter_version=adapter_version,
            request_hash=review["request_hash"],
            language_scope=language_scope,
            protected_check_failures=validate_meaning(body.draft, output),
        )

    @app.post("/v1/rewrite", response_model=RewriteResponse)
    def rewrite(
        body: RewriteRequest, request: Request, authorization: str | None = Header(default=None)
    ):
        authorize(request, authorization)
        if semantic_reviewer is None:
            return RewriteResponse(**fallback(body.draft, adapter_version, "review_required"))
        output = generate(body)
        if validate_meaning(body.draft, output):
            return RewriteResponse(
                **fallback(body.draft, adapter_version, "protected_check_failed")
            )
        review = make_request(body.draft, output, adapter_version, language_scope=language_scope)
        try:
            result = finalize(review, semantic_reviewer(copy.deepcopy(review)))
        except Exception:  # noqa: BLE001 - reviewer details may contain private text
            result = fallback(body.draft, adapter_version, "review_unavailable")
        return RewriteResponse(**result)

    return app
