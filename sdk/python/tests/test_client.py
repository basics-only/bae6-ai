import asyncio
import json
import traceback

import httpx
import pytest
from bae6_ai import Ai, AiApiError, AiAuthError, AiBudgetError, AiTransportError

MESSAGES = [{"role": "user", "content": "hi"}]
USAGE = {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}
TOKEN = "SECRET\nX"


def run(handler, call):
    async def go():
        async with Ai("http://gw", "svc-token", transport=httpx.MockTransport(handler)) as ai:
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
