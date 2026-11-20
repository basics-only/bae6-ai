import asyncio
import json
from collections.abc import AsyncIterator
from datetime import datetime
from types import TracebackType
from typing import Any, Self
from urllib.parse import quote

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
    return AiApiError(
        200, "invalid_response", "the gateway returned an unusable response"
    )


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
        task_type: str | None = None,
        domain: str | None = None,
        run_id: str | None = None,
    ) -> dict[str, Any]:
        """`task_type` and `domain` must be ids the organization has defined; `run_id` groups calls."""
        payload = _chat_payload(model, messages, max_completion_tokens, temperature)
        body = await self._call(
            "POST",
            "/v1/chat/completions",
            end_user_token,
            idempotency_key,
            payload,
            tags=_tags(task_type, domain, run_id),
        )
        if not isinstance(body.get("choices"), list) or not isinstance(
            body.get("usage"), dict
        ):
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
        task_type: str | None = None,
        domain: str | None = None,
        run_id: str | None = None,
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
            headers=_headers(end_user_token, None, _tags(task_type, domain, run_id)),
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
        task_type: str | None = None,
        domain: str | None = None,
        run_id: str | None = None,
    ) -> dict[str, Any]:
        body = await self._call(
            "POST",
            "/v1/embeddings",
            end_user_token,
            idempotency_key,
            {"model": model, "input": input},
            tags=_tags(task_type, domain, run_id),
        )
        if not isinstance(body.get("data"), list):
            raise _invalid_response()
        return body

    async def models(
        self, *, end_user_token: str | None = None
    ) -> list[dict[str, Any]]:
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
        body = await self._call(
            "GET", "/v1/usage", end_user_token, None, None, _range(start, end)
        )
        if not isinstance(body.get("data"), list):
            raise _invalid_response()
        return body["data"]

    async def budget(self, *, end_user_token: str | None = None) -> dict[str, Any]:
        """Money values are decimal strings in the price book's currency."""
        body = await self._call("GET", "/v1/budgets", end_user_token, None, None)
        if not isinstance(body.get("limit"), str) or not isinstance(
            body.get("spent"), str
        ):
            raise _invalid_response()
        return body

    async def feedback(
        self,
        *,
        request_id: str,
        verdict: str,
        label: str | None = None,
        comment: str | None = None,
        end_user_token: str | None = None,
    ) -> dict[str, Any]:
        """Rates one chat completion (its `id`) of this organization; a second rating replaces it.

        Needs `feedback:write`. `comment` is at most 1000 UTF-8 bytes.
        """
        payload = {"requestId": request_id, "verdict": verdict}
        if label is not None:
            payload["label"] = label
        if comment is not None:
            payload["comment"] = comment
        body = await self._call("POST", "/v1/feedback", end_user_token, None, payload)
        if not isinstance(body.get("requestId"), str):
            raise _invalid_response()
        return body

    async def list_feedback(
        self,
        *,
        limit: int | None = None,
        cursor: str | None = None,
        end_user_token: str | None = None,
    ) -> dict[str, Any]:
        """One page of the organization's feedback, newest first; needs `feedback:read`.

        Comments are untrusted caller text: never put one in a prompt.
        """
        params = {}
        if limit is not None:
            params["limit"] = str(limit)
        if cursor is not None:
            params["cursor"] = cursor
        body = await self._call(
            "GET", "/v1/feedback", end_user_token, None, None, params
        )
        if not isinstance(body.get("data"), list):
            raise _invalid_response()
        return body

    async def usage_summary(
        self,
        *,
        group_by: str,
        start: datetime | None = None,
        end: datetime | None = None,
        organization: str | None = None,
        run_ids: list[str] | None = None,
        end_user_token: str | None = None,
    ) -> dict[str, Any]:
        """Spend, tokens, latency and active users per group; needs `ai:usage:read`.

        `group_by` is organization, taskType, domain, model or run (`run_ids` then names 1 to 50
        runs). Grouping by organization, or naming another one, needs `ai:usage:platform`.
        """
        params = _range(start, end)
        params["groupBy"] = group_by
        if organization is not None:
            params["organization"] = organization
        body = await self._call(
            "GET",
            "/v1/usage/summary",
            end_user_token,
            None,
            None,
            [*params.items(), *(("runId", r) for r in run_ids or [])],
        )
        if not isinstance(body.get("data"), list):
            raise _invalid_response()
        return body

    async def usage_series(
        self,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
        group_by: str | None = None,
        group: str | None = None,
        organization: str | None = None,
        end_user_token: str | None = None,
    ) -> dict[str, Any]:
        """One point per UTC day, for everything or for one `group` of `group_by`."""
        params = _range(start, end)
        for key, value in (
            ("groupBy", group_by),
            ("group", group),
            ("organization", organization),
        ):
            if value is not None:
                params[key] = value
        body = await self._call(
            "GET", "/v1/usage/series", end_user_token, None, None, params
        )
        if not isinstance(body.get("data"), list):
            raise _invalid_response()
        return body

    async def usage_distribution(
        self,
        *,
        measure: str | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
        group_by: str | None = None,
        group: str | None = None,
        organization: str | None = None,
        end_user_token: str | None = None,
    ) -> dict[str, Any]:
        """How concentrated usage is across end users; carries no user ids."""
        params = _range(start, end)
        for key, value in (
            ("measure", measure),
            ("groupBy", group_by),
            ("group", group),
            ("organization", organization),
        ):
            if value is not None:
                params[key] = value
        body = await self._call(
            "GET", "/v1/usage/distribution", end_user_token, None, None, params
        )
        if not isinstance(body.get("data"), list):
            raise _invalid_response()
        return body

    async def usage_concentration(
        self,
        *,
        measure: str | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
        group_by: str | None = None,
        group: str | None = None,
        organization: str | None = None,
        end_user_token: str | None = None,
    ) -> dict[str, Any]:
        """How the top users' share of usage moves day by day; carries no user ids."""
        params = _range(start, end)
        for key, value in (
            ("measure", measure),
            ("groupBy", group_by),
            ("group", group),
            ("organization", organization),
        ):
            if value is not None:
                params[key] = value
        body = await self._call(
            "GET", "/v1/usage/concentration", end_user_token, None, None, params
        )
        if not isinstance(body.get("data"), list):
            raise _invalid_response()
        return body

    async def list_task_types(
        self, *, end_user_token: str | None = None
    ) -> list[dict[str, Any]]:
        return await self._list_tags("task-types", end_user_token)

    async def list_domains(
        self, *, end_user_token: str | None = None
    ) -> list[dict[str, Any]]:
        return await self._list_tags("domains", end_user_token)

    async def put_task_type(
        self, id: str, name: str, *, end_user_token: str | None = None
    ) -> dict[str, Any]:
        """Creates or renames a task type; needs `ai:usage:admin`. At most 50 per organization."""
        return await self._put_tag("task-types", id, name, end_user_token)

    async def put_domain(
        self, id: str, name: str, *, end_user_token: str | None = None
    ) -> dict[str, Any]:
        return await self._put_tag("domains", id, name, end_user_token)

    async def delete_task_type(
        self, id: str, *, end_user_token: str | None = None
    ) -> None:
        """Past usage keeps the id; new calls tagged with it are refused."""
        await self._call(
            "DELETE",
            f"/v1/task-types/{quote(id, safe='')}",
            end_user_token,
            None,
            None,
            empty=True,
        )

    async def delete_domain(
        self, id: str, *, end_user_token: str | None = None
    ) -> None:
        await self._call(
            "DELETE",
            f"/v1/domains/{quote(id, safe='')}",
            end_user_token,
            None,
            None,
            empty=True,
        )

    async def _list_tags(
        self, kind: str, end_user_token: str | None
    ) -> list[dict[str, Any]]:
        body = await self._call("GET", f"/v1/{kind}", end_user_token, None, None)
        if not isinstance(body.get("data"), list):
            raise _invalid_response()
        return body["data"]

    async def _put_tag(
        self, kind: str, id: str, name: str, end_user_token: str | None
    ) -> dict[str, Any]:
        body = await self._call(
            "PUT",
            f"/v1/{kind}/{quote(id, safe='')}",
            end_user_token,
            None,
            {"name": name},
        )
        if not isinstance(body.get("id"), str):
            raise _invalid_response()
        return body

    async def _call(
        self,
        method: str,
        path: str,
        end_user_token: str | None,
        idempotency_key: str | None,
        payload: dict[str, Any] | None,
        params: dict[str, str] | list[tuple[str, str]] | None = None,
        tags: dict[str, str] | None = None,
        empty: bool = False,
    ) -> dict[str, Any]:
        try:
            async with asyncio.timeout(self._timeout):
                request = self._http.build_request(
                    method,
                    path,
                    json=payload,
                    params=params,
                    headers=_headers(end_user_token, idempotency_key, tags),
                )
                response = await self._http.send(request, stream=True)
                try:
                    raw = await _read(response)
                finally:
                    await response.aclose()
        except TimeoutError as e:
            raise AiTimeoutError(
                f"the gateway did not answer within {self._timeout} s"
            ) from e
        except httpx.HTTPError as e:
            raise _transport(e) from None
        if empty and response.status_code == 204:
            return {}
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


def _tags(
    task_type: str | None, domain: str | None, run_id: str | None
) -> dict[str, str]:
    tags = {"bae6-task-type": task_type, "bae6-domain": domain, "bae6-run-id": run_id}
    return {name: value for name, value in tags.items() if value is not None}


def _range(start: datetime | None, end: datetime | None) -> dict[str, str]:
    params = {}
    for key, value in (("from", start), ("to", end)):
        if value is None:
            continue
        if value.utcoffset() is None:
            raise ValueError(f"{key} must be a timezone-aware datetime")
        params[key] = value.isoformat()
    return params


def _headers(
    end_user_token: str | None,
    idempotency_key: str | None,
    tags: dict[str, str] | None = None,
) -> dict[str, str]:
    headers = dict(tags or {})
    if end_user_token:
        headers["bae6-end-user-token"] = end_user_token
    if idempotency_key:
        headers["idempotency-key"] = idempotency_key
    return headers


def _transport(e: httpx.HTTPError) -> AiTransportError:
    # Raised `from None`: the cause can quote a header value, i.e. a token.
    if isinstance(e, httpx.TimeoutException):
        return AiTimeoutError(
            f"the gateway did not answer in time ({type(e).__name__})"
        )
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
