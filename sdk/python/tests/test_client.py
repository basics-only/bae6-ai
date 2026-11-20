import asyncio
import json
import traceback
from datetime import UTC, datetime

import httpx
import pytest

from bae6_ai import (
    Ai,
    AiApiError,
    AiAuthError,
    AiBudgetError,
    AiError,
    AiTimeoutError,
    AiTransportError,
)

MESSAGES = [{"role": "user", "content": "hi"}]
USAGE = {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}
TOKEN = "SECRET\nX"


def run(handler, call, agent_runs_url="http://worker/", eval_runs_url="http://evals/"):
    async def go():
        async with Ai(
            "http://gw",
            "svc-token",
            transport=httpx.MockTransport(handler),
            agent_runs_url=agent_runs_url,
            eval_runs_url=eval_runs_url,
        ) as ai:
            return await call(ai)

    return asyncio.run(go())


def sse(events: list[str]):
    body = "".join(f"data: {e}\n\n" for e in events)
    return lambda _: httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})


async def collect(ai: Ai) -> list[dict]:
    return [c async for c in ai.chat_stream(model="m", messages=MESSAGES)]


def test_sends_credentials_end_user_token_and_idempotency_key() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"choices": [], "usage": USAGE})

    run(
        handler,
        lambda ai: ai.chat(
            model="m", messages=MESSAGES, end_user_token="user", idempotency_key="key-00001"
        ),
    )
    headers = seen[0].headers
    assert headers["authorization"] == "Bearer svc-token"
    assert headers["bae6-end-user-token"] == "user"
    assert headers["idempotency-key"] == "key-00001"


def test_streams_chunks_and_requires_the_stream_to_finish() -> None:
    chunk = json.dumps({"choices": [{"index": 0, "delta": {"content": "hel"}}]})
    last = json.dumps({"choices": [], "usage": USAGE})
    chunks = run(sse([chunk, last, "[DONE]"]), collect)
    assert chunks[-1]["usage"] == USAGE
    with pytest.raises(AiTransportError):
        run(sse([chunk]), collect)


def test_mid_stream_error_surfaces_as_api_error() -> None:
    error = json.dumps({"error": {"code": "timeout", "message": "late", "requestId": "r1"}})
    with pytest.raises(AiApiError) as raised:
        run(sse([error]), collect)
    assert raised.value.code == "timeout"
    assert raised.value.request_id == "r1"


@pytest.mark.parametrize(
    ("status", "error_type"), [(402, AiBudgetError), (401, AiAuthError), (429, AiApiError)]
)
def test_maps_error_statuses(status: int, error_type: type) -> None:
    body = {"code": "x", "message": "no", "requestId": "r", "retryAfterSeconds": 2}
    with pytest.raises(error_type) as raised:
        run(
            lambda _: httpx.Response(status, json=body),
            lambda ai: ai.chat(model="m", messages=MESSAGES),
        )
    assert raised.value.retry_after_seconds == 2


def test_transport_error_does_not_carry_a_token() -> None:
    async def call() -> None:
        async with Ai("http://127.0.0.1:9", "svc") as ai:
            await ai.chat(model="m", messages=MESSAGES, end_user_token=TOKEN)

    with pytest.raises(AiTransportError) as raised:
        asyncio.run(call())
    assert "SECRET" not in "".join(traceback.format_exception(raised.value))


def test_reads_a_last_stream_line_sent_without_a_newline() -> None:
    last = json.dumps({"choices": [], "usage": USAGE})
    body = f"data: {last}\n\ndata: [DONE]"
    handler = lambda _: httpx.Response(200, text=body)
    assert run(handler, collect)[-1]["usage"] == USAGE


def test_a_stalled_stream_times_out() -> None:
    async def stall(_reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        writer.write(b"HTTP/1.1 200 OK\r\ntransfer-encoding: chunked\r\n\r\n")
        await asyncio.sleep(5)

    async def go() -> None:
        server = await asyncio.start_server(stall, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        try:
            async with Ai(f"http://127.0.0.1:{port}", "svc", timeout=0.2) as ai:
                await collect(ai)
        finally:
            server.close()

    with pytest.raises(AiTimeoutError):
        asyncio.run(go())


def test_usage_rejects_a_naive_datetime_and_read_calls_send_the_end_user_token() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"data": []})

    with pytest.raises(ValueError):
        run(handler, lambda ai: ai.usage(start=datetime(2026, 1, 1)))
    assert seen == []
    run(
        handler,
        lambda ai: ai.usage(start=datetime(2026, 1, 1, tzinfo=UTC), end_user_token="user"),
    )
    assert seen[0].headers["bae6-end-user-token"] == "user"
    assert seen[0].url.params["from"] == "2026-01-01T00:00:00+00:00"


def test_submits_feedback_and_pages_through_the_listing() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.method == "POST":
            if json.loads(request.content)["requestId"] == "chatcmpl-led_y":
                return httpx.Response(404, json={"code": "not_found", "message": "no"})
            return httpx.Response(200, json={"requestId": "chatcmpl-led_x", "verdict": "negative"})
        return httpx.Response(200, json={"data": [], "nextCursor": "7"})

    ack = run(
        handler,
        lambda ai: ai.feedback(request_id="chatcmpl-led_x", verdict="negative", label="unsafe"),
    )
    assert ack["verdict"] == "negative"
    assert json.loads(seen[0].content) == {
        "requestId": "chatcmpl-led_x",
        "verdict": "negative",
        "label": "unsafe",
    }

    page = run(handler, lambda ai: ai.list_feedback(limit=2, cursor="9"))
    assert page["nextCursor"] == "7"
    assert dict(seen[1].url.params) == {"limit": "2", "cursor": "9"}

    with pytest.raises(AiApiError) as raised:
        run(handler, lambda ai: ai.feedback(request_id="chatcmpl-led_y", verdict="positive"))
    assert raised.value.status == 404


def test_agent_runs_go_to_the_worker_and_accept_its_202() -> None:
    seen: list[httpx.Request] = []
    made = {"id": "run_00000000-0000-0000-0000-000000000000", "status": "queued", "receipts": []}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.method == "GET" and request.url.path == "/v1/agent-runs":
            return httpx.Response(200, json={"data": [], "nextCursor": None})
        return httpx.Response(202 if request.method == "POST" else 200, json=made)

    run_ = run(
        handler,
        lambda ai: ai.create_agent_run(task="t", model="m", idempotency_key="key-00001"),
    )
    assert run_["status"] == "queued"
    assert str(seen[0].url) == "http://worker/v1/agent-runs"
    assert seen[0].headers["idempotency-key"] == "key-00001"
    assert json.loads(seen[0].content) == {"task": "t", "model": "m"}

    run(handler, lambda ai: ai.cancel_agent_run(made["id"]))
    assert seen[1].url.path == f"/v1/agent-runs/{made['id']}/cancel"

    run(handler, lambda ai: ai.list_agent_runs(status="finished", limit=5))
    assert dict(seen[2].url.params) == {"status": "finished", "limit": "5"}

    def full(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"code": "queue_full", "retryAfterSeconds": 5})

    with pytest.raises(AiApiError) as raised:
        run(full, lambda ai: ai.create_agent_run(task="t", model="m"))
    assert raised.value.retry_after_seconds == 5

    with pytest.raises(AiError, match="agent_runs_url is not configured"):
        run(handler, lambda ai: ai.get_agent_run(made["id"]), agent_runs_url=None)


def test_eval_runs_go_to_the_eval_worker_and_accept_its_202() -> None:
    seen: list[httpx.Request] = []
    made = {"id": "eval_00000000-0000-0000-0000-000000000000", "status": "queued", "scorers": []}
    cases = [{"id": "a", "input": "1+1", "expected": "2"}]

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.method == "GET" and request.url.path == "/v1/eval-runs":
            return httpx.Response(200, json={"data": [], "nextCursor": None})
        return httpx.Response(202 if request.method == "POST" else 200, json=made)

    created = run(
        handler,
        lambda ai: ai.create_eval_run(
            name="smoke",
            dataset_version="1",
            model="m",
            cases=cases,
            idempotency_key="key-00001",
        ),
    )
    assert created["status"] == "queued"
    assert str(seen[0].url) == "http://evals/v1/eval-runs"
    assert seen[0].headers["idempotency-key"] == "key-00001"
    assert json.loads(seen[0].content) == {
        "name": "smoke",
        "datasetVersion": "1",
        "model": "m",
        "cases": cases,
    }

    run(handler, lambda ai: ai.get_eval_run(made["id"]))
    assert seen[1].url.path == f"/v1/eval-runs/{made['id']}"

    run(handler, lambda ai: ai.cancel_eval_run(made["id"]))
    assert seen[2].url.path == f"/v1/eval-runs/{made['id']}/cancel"

    run(handler, lambda ai: ai.list_eval_runs(status="finished", limit=5))
    assert dict(seen[3].url.params) == {"status": "finished", "limit": "5"}

    with pytest.raises(AiError, match="eval_runs_url is not configured"):
        run(handler, lambda ai: ai.get_eval_run(made["id"]), eval_runs_url=None)
