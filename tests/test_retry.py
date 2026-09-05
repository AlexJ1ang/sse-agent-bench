"""用例执行器重试逻辑单元测试。

只测 ``_should_retry`` 的判定逻辑（纯函数）与 execute 的重试编排，
不发起真实网络请求。
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import httpx

from agent_harness.config import HarnessConfig
from agent_harness.core.runner import CaseRunner
from agent_harness.models import (
    SSEEvent,
    SSEStreamResult,
)
from agent_harness.models import (
    TestCase as HarnessTestCase,
)


def _make_runner(**eval_kwargs) -> CaseRunner:
    config = HarnessConfig()
    for k, v in eval_kwargs.items():
        setattr(config.eval, k, v)
    return CaseRunner(config)


# ── _should_retry 判定 ─────────────────────────────────────────


def test_should_retry_timeout():
    assert _make_runner()._should_retry("timeout: deadline exceeded") is True


def test_should_retry_request_error():
    assert _make_runner()._should_retry("request error: conn refused") is True


def test_should_retry_stream_interrupted():
    assert _make_runner()._should_retry("stream interrupted") is True


def test_should_not_retry_http_error():
    assert _make_runner()._should_retry("http 500: internal") is False


def test_should_not_retry_http_404():
    assert _make_runner()._should_retry("http 404: not found") is False


def test_should_not_retry_empty_error():
    assert _make_runner()._should_retry("") is False


# ── execute 重试编排 ───────────────────────────────────────────


class TestExecuteRetry:
    def test_retries_then_succeeds(self):
        """前两次超时、第三次成功，最终 success=True 且 retries=2。"""
        runner = _make_runner(retry_count=2, retry_delay_ms=0)

        calls = {"n": 0}

        async def fake_request(client, url, payload, started_at):
            calls["n"] += 1
            if calls["n"] < 3:
                raise httpx.TimeoutException("timeout")
            stream = SSEStreamResult(
                events=[SSEEvent(event="answer", payload={"content": "ok"})],
                answer_parts=["ok"],
            )
            # 用假的 LatencyMetrics-like 对象
            from agent_harness.models import LatencyMetrics
            return stream, LatencyMetrics(request_total_ms=10.0)

        case = HarnessTestCase(id="c1", question="q")
        runner._execute_sse_request = fake_request  # type: ignore[assignment]

        result = asyncio.run(runner.execute(case))
        assert result.success is True
        assert result.retries == 2
        assert calls["n"] == 3

    def test_http_error_not_retried(self):
        """4xx/5xx 直接失败，不重试。"""
        runner = _make_runner(retry_count=3, retry_delay_ms=0)

        async def fake_request(client, url, payload, started_at):
            raise httpx.HTTPStatusError(
                "boom", request=MagicMock(), response=MagicMock(status_code=500)
            )

        case = HarnessTestCase(id="c1", question="q")
        runner._execute_sse_request = fake_request  # type: ignore[assignment]

        result = asyncio.run(runner.execute(case))
        assert result.success is False
        assert result.retries == 0

    def test_retries_exhausted(self):
        """重试次数用尽仍失败，success=False。"""
        runner = _make_runner(retry_count=1, retry_delay_ms=0)

        async def fake_request(client, url, payload, started_at):
            raise httpx.TimeoutException("timeout")

        case = HarnessTestCase(id="c1", question="q")
        runner._execute_sse_request = fake_request  # type: ignore[assignment]

        result = asyncio.run(runner.execute(case))
        assert result.success is False
        assert result.retries == 1

    def test_no_retry_by_default(self):
        """默认 retry_count=0，一次失败即失败。"""
        runner = _make_runner()

        calls = {"n": 0}

        async def fake_request(client, url, payload, started_at):
            calls["n"] += 1
            raise httpx.TimeoutException("timeout")

        case = HarnessTestCase(id="c1", question="q")
        runner._execute_sse_request = fake_request  # type: ignore[assignment]

        result = asyncio.run(runner.execute(case))
        assert result.success is False
        assert result.retries == 0
        assert calls["n"] == 1
