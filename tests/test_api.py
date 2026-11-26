import threading
from concurrent.futures import ThreadPoolExecutor

from fastapi.testclient import TestClient

from multilora_tone_adapter.api import create_app


class Engine:
    loaded = False

    def __init__(self):
        self.calls = []
        self.output = "hello there"

    def rewrite(self, draft, *, max_output_tokens=128):
        self.calls.append((draft, max_output_tokens))
        self.loaded = True
        return self.output


def test_health_is_lazy_and_request_uses_only_draft():
    engine = Engine()
    client = TestClient(create_app(engine, adapter_version="candidate-1"))
    assert client.get("/health").json()["loaded"] is False
    response = client.post(
        "/v1/rewrite",
        json={
            "task": "rewrite",
            "profile": "telegram",
            "draft": "hello",
            "max_output_tokens": 40,
        },
    )
    assert response.status_code == 200
    assert engine.calls == [("hello", 40)]
    assert response.json()["adapter_version"] == "candidate-1"
    assert "semantic review" in response.json()["meaning_validation"]


def test_meaning_failure_never_returns_unsafe_output():
    engine = Engine()
    engine.output = "meet at 9"
    response = TestClient(create_app(engine)).post("/v1/rewrite", json={"draft": "meet at 8"})
    assert response.status_code == 422
    assert "changed_numbers" in response.json()["detail"]["checks"]
    assert engine.output not in response.text


def test_context_is_rejected_even_when_empty_or_null():
    engine = Engine()
    client = TestClient(create_app(engine))
    for context in ([], None, [{"speaker": "other", "text": "arrive at 8"}]):
        response = client.post("/v1/rewrite", json={"draft": "hello", "context": context})
        assert response.status_code == 422
    assert not engine.calls
    schema = client.get("/openapi.json").json()["components"]["schemas"]["RewriteRequest"]
    assert "context" not in schema["properties"]
    assert schema["additionalProperties"] is False


def test_strict_request_bounds_and_no_arbitrary_paths():
    engine = Engine()
    client = TestClient(create_app(engine))
    for payload in (
        {"draft": " "},
        {"draft": "x", "task": "send"},
        {"draft": "x", "profile": "other"},
        {"draft": "x", "model_path": "/tmp/model"},
        {"draft": "x", "max_output_tokens": 513},
        {"draft": "x", "context": [{"role": "user", "content": "x"}]},
    ):
        assert client.post("/v1/rewrite", json=payload).status_code == 422
    assert not engine.calls


def test_auth_origin_and_host():
    client = TestClient(create_app(Engine(), token="local-secret"))
    assert client.get("/health").status_code == 401
    headers = {"Authorization": "Bearer local-secret"}
    assert client.get("/health", headers=headers).status_code == 200
    assert (
        client.post(
            "/v1/rewrite",
            json={"draft": "hello"},
            headers={**headers, "Origin": "https://evil.example"},
        ).status_code
        == 403
    )
    assert client.get("/health", headers={**headers, "Host": "evil.example"}).status_code == 400


def test_busy_request_is_rejected_without_a_second_generation():
    started, finish = threading.Event(), threading.Event()

    class BlockingEngine(Engine):
        def rewrite(self, *args, **kwargs):
            started.set()
            assert finish.wait(5)
            return super().rewrite(*args, **kwargs)

    engine = BlockingEngine()
    client = TestClient(create_app(engine))
    with ThreadPoolExecutor(max_workers=1) as pool:
        first = pool.submit(client.post, "/v1/rewrite", json={"draft": "hello"})
        assert started.wait(5)
        try:
            second = client.post("/v1/rewrite", json={"draft": "hello again"})
            assert second.status_code == 429
        finally:
            finish.set()
        assert first.result().status_code == 200
    assert len(engine.calls) == 1


def test_engine_exception_is_sanitized_and_lock_released():
    class BrokenEngine(Engine):
        def rewrite(self, *args, **kwargs):
            raise RuntimeError("/private/path plus prompt secret")

    client = TestClient(create_app(BrokenEngine()))
    for _ in range(2):
        response = client.post("/v1/rewrite", json={"draft": "hello"})
        assert response.status_code == 503
        assert "/private" not in response.text
        assert "prompt secret" not in response.text
