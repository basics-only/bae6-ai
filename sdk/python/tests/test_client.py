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
    AiTimeoutError,
    AiTransportError,
)

MESSAGES = [{"role": "user", "content": "hi"}]
USAGE = {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}
TOKEN = "SECRET\nX"


def run(handler, call):
    async def go():
        async with Ai(
            "http://gw", "svc-token", transport=httpx.MockTransport(handler)
        ) as ai:
            return await call(ai)

    return asyncio.run(go())


def sse(events: list[str]):
    body = "".join(f"data: {e}\n\n" for e in events)
    return lambda _: httpx.Response(
        200, text=body, headers={"content-type": "text/event-stream"}
    )


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
            model="m",
            messages=MESSAGES,
            end_user_token="user",
            idempotency_key="key-00001",
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
    error = json.dumps(
        {"error": {"code": "timeout", "message": "late", "requestId": "r1"}}
    )
    with pytest.raises(AiApiError) as raised:
        run(sse([error]), collect)
    assert raised.value.code == "timeout"
    assert raised.value.request_id == "r1"


@pytest.mark.parametrize(
    ("status", "error_type"),
    [(402, AiBudgetError), (401, AiAuthError), (429, AiApiError)],
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
    async def stall(
        _reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
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


def test_usage_rejects_a_naive_datetime_and_read_calls_send_the_end_user_token() -> (
    None
):
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"data": []})

    with pytest.raises(ValueError):
        run(handler, lambda ai: ai.usage(start=datetime(2026, 1, 1)))
    assert seen == []
    run(
        handler,
        lambda ai: ai.usage(
            start=datetime(2026, 1, 1, tzinfo=UTC), end_user_token="user"
        ),
    )
    assert seen[0].headers["bae6-end-user-token"] == "user"
    assert seen[0].url.params["from"] == "2026-01-01T00:00:00+00:00"


def test_reads_credential_status_and_rejects_a_malformed_answer() -> None:
    data = [{"provider": "bedrock", "source": "web-identity", "lastRefreshAgeSeconds": 3}]
    assert run(lambda _: httpx.Response(200, json={"data": data}), lambda ai: ai.credentials()) == data
    with pytest.raises(AiApiError):
        run(lambda _: httpx.Response(200, json={}), lambda ai: ai.credentials())


def test_submits_feedback_and_pages_through_the_listing() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.method == "POST":
            if json.loads(request.content)["requestId"] == "chatcmpl-led_y":
                return httpx.Response(404, json={"code": "not_found", "message": "no"})
            return httpx.Response(
                200, json={"requestId": "chatcmpl-led_x", "verdict": "negative"}
            )
        return httpx.Response(200, json={"data": [], "nextCursor": "7"})

    ack = run(
        handler,
        lambda ai: ai.feedback(
            request_id="chatcmpl-led_x", verdict="negative", label="unsafe"
        ),
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
        run(
            handler,
            lambda ai: ai.feedback(request_id="chatcmpl-led_y", verdict="positive"),
        )
    assert raised.value.status == 404


def test_sends_tag_headers_on_chat_stream_and_embeddings() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.path == "/v1/embeddings":
            return httpx.Response(200, json={"data": []})
        if request.content and b'"stream"' in request.content:
            return sse(["[DONE]"])(request)
        return httpx.Response(200, json={"choices": [], "usage": USAGE})

    tags = {"task_type": "summarization", "domain": "legal", "run_id": "run-1"}

    async def call(ai: Ai) -> None:
        await ai.chat(model="m", messages=MESSAGES, **tags)
        [c async for c in ai.chat_stream(model="m", messages=MESSAGES, **tags)]
        await ai.embeddings(model="m", input="x", **tags)
        await ai.chat(model="m", messages=MESSAGES)

    run(handler, call)
    for request in seen[:3]:
        assert request.headers["bae6-task-type"] == "summarization"
        assert request.headers["bae6-domain"] == "legal"
        assert request.headers["bae6-run-id"] == "run-1"
    assert "bae6-task-type" not in seen[3].headers


def test_usage_summary_sends_group_runs_and_range() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"data": []})

    start = datetime(2026, 11, 1, tzinfo=UTC)
    run(
        handler,
        lambda ai: ai.usage_summary(group_by="run", start=start, run_ids=["a", "b"]),
    )
    query = seen[0].url.params
    assert query["groupBy"] == "run"
    assert query.get_list("runId") == ["a", "b"]
    assert query["from"] == start.isoformat()
    with pytest.raises(ValueError):
        run(
            handler,
            lambda ai: ai.usage_summary(group_by="model", end=datetime(2026, 11, 2)),
        )


def test_usage_distribution_sends_measure_and_narrowing() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"users": 0, "total": "0", "data": []})

    run(
        handler,
        lambda ai: ai.usage_distribution(
            measure="cost", group_by="taskType", group="summarization"
        ),
    )
    assert seen[0].url.path == "/v1/usage/distribution"
    query = seen[0].url.params
    assert query["measure"] == "cost"
    assert query["groupBy"] == "taskType"
    assert query["group"] == "summarization"


def test_usage_concentration_sends_measure_to_its_path() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"data": []})

    run(handler, lambda ai: ai.usage_concentration(measure="cost"))
    assert seen[0].url.path == "/v1/usage/concentration"
    assert seen[0].url.params["measure"] == "cost"


def test_manages_task_types_and_domains() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.method == "DELETE":
            return httpx.Response(204)
        if request.method == "PUT":
            return httpx.Response(
                200, json={"id": "legal", "name": "Legal", "updatedAt": "t"}
            )
        return httpx.Response(200, json={"data": [{"id": "legal"}]})

    async def call(ai: Ai):
        assert await ai.list_domains() == [{"id": "legal"}]
        assert (await ai.put_task_type("legal", "Legal"))["id"] == "legal"
        assert await ai.delete_domain("legal") is None

    run(handler, call)
    assert [(r.method, r.url.path) for r in seen] == [
        ("GET", "/v1/domains"),
        ("PUT", "/v1/task-types/legal"),
        ("DELETE", "/v1/domains/legal"),
    ]
    assert json.loads(seen[1].content) == {"name": "Legal"}


def test_encodes_tag_ids_in_the_path() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(204)

    async def call(ai: Ai):
        await ai.delete_domain("a/b?c")

    run(handler, call)
    assert seen[0].url.raw_path == b"/v1/domains/a%2Fb%3Fc"
