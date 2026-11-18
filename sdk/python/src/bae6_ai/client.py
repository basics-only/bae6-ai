import asyncio
import json
from collections.abc import AsyncIterator
from datetime import datetime
from types import TracebackType
from typing import Any, Self

import httpx

MAX_RESPONSE_BYTES = 64 * 1024 * 1024
MAX_SSE_LINE = 1024 * 1024


class AiError(Exception):
    def __init__(self, message: str, request_id: str | None = None) -> None:
        super().__init__(message)
        self.request_id = request_id


class AiApiError(AiError):
    """The gateway answered with an error."""

    def __init__(
        self,
        status: int,
        code: str | None,
        message: str,
        request_id: str | None = None,
        retry_after_seconds: float | None = None,
    ) -> None:
        super().__init__(message, request_id)
        self.status = status
        self.code = code
        self.retry_after_seconds = retry_after_seconds


class AiAuthError(AiApiError):
    """401 or 403: bad workload or end-user credentials, a missing scope, or no budget."""


class AiValidationError(AiApiError):
    """400, 409 or 413: the gateway rejected the request; retrying it unchanged will not help."""


class AiBudgetError(AiApiError):
    """402: the organization's budget cannot cover this request."""


class AiTransportError(AiError):
    """The gateway was not reached, did not answer, or a stream ended early."""


class AiTimeoutError(AiTransportError):
    pass


def _str(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _http_error(status: int, parsed: object) -> AiApiError:
    body = parsed if isinstance(parsed, dict) else {}
    error_type = (
        AiBudgetError
        if status == 402
        else AiAuthError
        if status in (401, 403)
        else AiValidationError
        if status in (400, 409, 413)
        else AiApiError
    )
    retry = body.get("retryAfterSeconds")
    return error_type(
        status,
        _str(body.get("code")),
        _str(body.get("message")) or f"the gateway returned HTTP {status}",
        _str(body.get("requestId")),
        retry if isinstance(retry, int | float) else None,
    )


def _invalid_response() -> AiApiError:
    return AiApiError(200, "invalid_response", "the gateway returned an unusable response")


def _parse(text: bytes) -> object:
    try:
        return json.loads(text)
    except (ValueError, RecursionError):
        return None


class Ai:
    """Async gateway client. Use `async with Ai(...) as ai` so connections are closed.

    Responses are the gateway's OpenAI-shaped JSON, as dicts.
    """

    def __init__(
        self,
        base_url: str,
        credentials: str,
        timeout: float = 120.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._timeout = timeout
        self._http = httpx.AsyncClient(
            base_url=base_url,
            headers={"authorization": f"Bearer {credentials}"},
            # Non-streaming deadlines are asyncio.timeout's, so they cover the whole request.
            timeout=None,
            # Credentials must not follow a redirect to another host.
            follow_redirects=False,
            transport=transport,
        )

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._http.aclose()

    async def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        max_completion_tokens: int | None = None,
        temperature: float | None = None,
        end_user_token: str | None = None,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        payload = _chat_payload(model, messages, max_completion_tokens, temperature)
        body = await self._call(
            "POST", "/v1/chat/completions", end_user_token, idempotency_key, payload
        )
        if not isinstance(body.get("choices"), list) or not isinstance(body.get("usage"), dict):
            raise _invalid_response()
        return body

    async def chat_stream(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        max_completion_tokens: int | None = None,
        temperature: float | None = None,
        end_user_token: str | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Yields chunks as the model produces them; the last carries `usage`.

        Closing the iterator early cancels the model call. A stream that ends early raises
        AiTransportError.
        """
        payload = _chat_payload(model, messages, max_completion_tokens, temperature)
        payload["stream"] = True
        payload["stream_options"] = {"include_usage": True}
        request = self._http.build_request(
            "POST",
            "/v1/chat/completions",
            json=payload,
            headers=_headers(end_user_token, None),
            # Bounds connecting, the wait for the answer, and each gap between chunks.
            timeout=httpx.Timeout(self._timeout),
        )
        try:
            response = await self._http.send(request, stream=True)
        except httpx.HTTPError as e:
            raise _transport(e) from None
        try:
            if response.status_code != 200:
                raise _http_error(response.status_code, _parse(await _read(response)))
            async for line in _lines(response):
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    return
                event = _parse(data.encode())
                if not isinstance(event, dict):
                    raise _invalid_response()
                if isinstance(event.get("error"), dict):
                    raise _http_error(502, event["error"])
                yield event
            raise AiTransportError("the stream ended before completion")
        except httpx.HTTPError as e:
            raise _transport(e) from None
        finally:
            await response.aclose()

    async def embeddings(
        self,
        *,
        model: str,
        input: str | list[str],
        end_user_token: str | None = None,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        body = await self._call(
            "POST",
            "/v1/embeddings",
            end_user_token,
            idempotency_key,
            {"model": model, "input": input},
        )
        if not isinstance(body.get("data"), list):
            raise _invalid_response()
        return body

    async def models(self, *, end_user_token: str | None = None) -> list[dict[str, Any]]:
        body = await self._call("GET", "/v1/models", end_user_token, None, None)
        if not isinstance(body.get("data"), list):
            raise _invalid_response()
        return body["data"]

    async def usage(
        self,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
        end_user_token: str | None = None,
    ) -> list[dict[str, Any]]:
        """Usage of this application, by model and outcome, in [start, end) (at most 93 days).

        Datetimes must be timezone-aware; a naive one has no unambiguous instant.
        """
        params = {}
        for key, value in (("from", start), ("to", end)):
            if value is None:
                continue
            if value.utcoffset() is None:
                raise ValueError(f"{key} must be a timezone-aware datetime")
            params[key] = value.isoformat()
        body = await self._call("GET", "/v1/usage", end_user_token, None, None, params)
        if not isinstance(body.get("data"), list):
            raise _invalid_response()
        return body["data"]

    async def budget(self, *, end_user_token: str | None = None) -> dict[str, Any]:
        """Money values are decimal strings in the price book's currency."""
        body = await self._call("GET", "/v1/budgets", end_user_token, None, None)
        if not isinstance(body.get("limit"), str) or not isinstance(body.get("spent"), str):
            raise _invalid_response()
        return body

    async def _call(
        self,
        method: str,
        path: str,
        end_user_token: str | None,
        idempotency_key: str | None,
        payload: dict[str, Any] | None,
        params: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        try:
            async with asyncio.timeout(self._timeout):
                request = self._http.build_request(
                    method,
                    path,
                    json=payload,
                    params=params,
                    headers=_headers(end_user_token, idempotency_key),
                )
                response = await self._http.send(request, stream=True)
                try:
                    raw = await _read(response)
                finally:
                    await response.aclose()
        except TimeoutError as e:
            raise AiTimeoutError(f"the gateway did not answer within {self._timeout} s") from e
        except httpx.HTTPError as e:
            raise _transport(e) from None
        parsed = _parse(raw)
        if response.status_code != 200:
            raise _http_error(response.status_code, parsed)
        if not isinstance(parsed, dict):
            raise _invalid_response()
        return parsed


def _chat_payload(
    model: str,
    messages: list[dict[str, Any]],
    max_completion_tokens: int | None,
    temperature: float | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"model": model, "messages": messages}
    if max_completion_tokens is not None:
        payload["max_completion_tokens"] = max_completion_tokens
    if temperature is not None:
        payload["temperature"] = temperature
    return payload


def _headers(end_user_token: str | None, idempotency_key: str | None) -> dict[str, str]:
    headers = {}
    if end_user_token:
        headers["bae6-end-user-token"] = end_user_token
    if idempotency_key:
        headers["idempotency-key"] = idempotency_key
    return headers


def _transport(e: httpx.HTTPError) -> AiTransportError:
    # Raised `from None`: the cause can quote a header value, i.e. a token.
    if isinstance(e, httpx.TimeoutException):
        return AiTimeoutError(f"the gateway did not answer in time ({type(e).__name__})")
    return AiTransportError(f"could not reach the gateway ({type(e).__name__})")


async def _read(response: httpx.Response) -> bytes:
    chunks = []
    size = 0
    async for chunk in response.aiter_bytes():
        size += len(chunk)
        if size > MAX_RESPONSE_BYTES:
            raise AiTransportError("the gateway response is too large")
        chunks.append(chunk)
    return b"".join(chunks)


async def _lines(response: httpx.Response) -> AsyncIterator[str]:
    """Lines without endings, including a last unterminated one."""
    oversized = "the gateway sent an oversized stream event"
    buffered = ""
    async for text in response.aiter_text():
        buffered += text
        *complete, buffered = buffered.split("\n")
        for line in complete:
            if len(line) > MAX_SSE_LINE:
                raise AiTransportError(oversized)
            yield line.removesuffix("\r")
        if len(buffered) > MAX_SSE_LINE:
            raise AiTransportError(oversized)
    if buffered:
        yield buffered.removesuffix("\r")
