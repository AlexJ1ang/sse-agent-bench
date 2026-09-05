"""用例执行器。

批量执行测试用例，采集 SSE 流，支持并发控制、超时、重试。
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any

import httpx

from agent_harness.config import HarnessConfig
from agent_harness.core.payload import build_payload
from agent_harness.core.sse_collector import collect_sse_stream
from agent_harness.models import (
    CaseResult,
    LatencyMetrics,
    RepeatResult,
    RunResult,
    SSEStreamResult,
    TestCase,
)

logger = logging.getLogger(__name__)


class CaseRunner:
    """执行单条测试用例。"""

    def __init__(self, config: HarnessConfig) -> None:
        self.config = config
        self.target = config.target

    async def execute(
        self,
        case: TestCase,
        client: httpx.AsyncClient | None = None,
    ) -> CaseResult:
        """执行单条用例，返回完整结果。

        对网络层瞬时错误做透明重试（见 ``_should_retry``），重试不产生
        新的 RepeatResult，因此不会污染 flaky 统计。
        """
        result = CaseResult(case=case)
        result.started_at = time.perf_counter()

        payload = build_payload(self.config, case)
        result.request_payload = payload
        url = f"{self.target.base_url}{self.target.chat_path}"
        timeout = httpx.Timeout(
            connect=self.target.connect_timeout,
            read=self.target.read_timeout,
            write=self.target.connect_timeout,
            pool=self.target.read_timeout,
        )

        own_client = client is None
        if own_client:
            client = httpx.AsyncClient(timeout=timeout)
        assert client is not None

        max_attempts = self.config.eval.retry_count + 1
        for attempt in range(1, max_attempts + 1):
            result.retries = attempt - 1
            try:
                stream_result, latency = await self._execute_sse_request(
                    client, url, payload, result.started_at
                )
            except httpx.TimeoutException as exc:
                result.error = f"timeout: {exc}"
                result.success = False
            except httpx.HTTPStatusError as exc:
                result.error = f"http {exc.response.status_code}: {exc.response.text[:500]}"
                result.success = False
            except httpx.RequestError as exc:
                result.error = f"request error: {exc}"
                result.success = False
            except Exception as exc:
                result.error = f"unexpected: {exc}"
                result.success = False
                logger.exception("Case %s failed", case.id)
            else:
                result.stream = stream_result
                result.latency = latency
                result.success = (
                    stream_result.event_count > 0
                    and not stream_result.error_events
                )
                # SSE 流中断（提前截断）视为瞬时错误，可重试
                if stream_result.stream_interrupted:
                    result.success = False
                    result.error = "stream interrupted"

            # 判断是否需要重试
            if result.success or attempt >= max_attempts:
                break
            if not self._should_retry(result.error):
                break
            delay = self.config.eval.retry_delay_ms / 1000
            logger.warning(
                "Case %s 第 %d 次执行失败(%s)，%.0fms 后重试",
                case.id, attempt, result.error[:80], delay * 1000,
            )
            await asyncio.sleep(delay)
            # 重置错误，准备下一次执行
            result.error = ""

        if own_client:
            await client.aclose()

        result.completed_at = time.perf_counter()
        return result

    @staticmethod
    def _should_retry(error: str) -> bool:
        """判断错误是否值得重试。

        只对网络层瞬时错误重试：超时、连接类请求错误、SSE 流中断。
        HTTP 4xx/5xx 是服务端明确响应，重试同一请求大概率得到同样结果，不重试。
        """
        if not error:
            return False
        return (
            error.startswith("timeout")
            or error.startswith("request error")
            or error == "stream interrupted"
        )

    async def _execute_sse_request(
        self,
        client: httpx.AsyncClient,
        url: str,
        payload: dict[str, Any],
        started_at: float,
    ) -> tuple[SSEStreamResult, LatencyMetrics]:
        """发起 SSE 请求并采集流。"""
        headers = {"Accept": "text/event-stream", **self.target.request_adapter.extra_headers}
        async with client.stream(
            self.target.method, url, json=payload, headers=headers
        ) as response:
            response.raise_for_status()
            return await collect_sse_stream(
                response.aiter_bytes(), started_at, self.target.sse_mapping
            )


class BatchRunner:
    """批量执行测试用例。"""

    def __init__(self, config: HarnessConfig) -> None:
        self.config = config
        self.runner = CaseRunner(config)

    async def run(
        self,
        cases: list[TestCase],
        concurrency: int | None = None,
    ) -> RunResult:
        """批量执行用例，返回运行结果。

        当 eval.repeat_count > 1 时，每条用例重复执行 N 次以采集统计分布。
        """
        run_id = uuid.uuid4().hex[:12]
        concurrency = concurrency or self.config.eval.concurrency
        repeat_count = self.config.eval.repeat_count
        semaphore = asyncio.Semaphore(concurrency)
        started = datetime.now(timezone.utc).isoformat()

        timeout = httpx.Timeout(
            connect=self.config.target.connect_timeout,
            read=self.config.target.read_timeout,
            write=self.config.target.connect_timeout,
            pool=self.config.target.read_timeout,
        )

        async with httpx.AsyncClient(timeout=timeout) as client:
            async def execute_single(case: TestCase) -> CaseResult:
                async with semaphore:
                    return await self.runner.execute(case, client)

            async def execute_with_repeats(case: TestCase) -> CaseResult:
                """执行单条用例（含重复采样）。

                每次执行只负责采集 SSE 流并存入 RepeatResult；分数聚合
                在评分阶段（EvalPipeline）完成后统一进行，因为执行期
                尚不持有 eval_score。
                """
                # 首次执行（代表值）
                primary = await execute_single(case)
                primary.repeat_count = repeat_count
                primary.repeats.append(
                    RepeatResult(
                        attempt=1,
                        success=primary.success,
                        stream=primary.stream,
                        latency_ms=primary.latency.request_total_ms,
                        answer_preview=primary.stream.full_answer[:80],
                        error=primary.error,
                    )
                )

                if repeat_count <= 1:
                    return primary

                # 后续重复执行（仅采集快照，不覆盖 primary）
                for attempt in range(2, repeat_count + 1):
                    r = await execute_single(case)
                    primary.repeats.append(
                        RepeatResult(
                            attempt=attempt,
                            success=r.success,
                            stream=r.stream,
                            latency_ms=r.latency.request_total_ms,
                            answer_preview=r.stream.full_answer[:80],
                            error=r.error,
                        )
                    )

                return primary

            tasks = [execute_with_repeats(c) for c in cases]
            case_results = await asyncio.gather(*tasks)

        completed = datetime.now(timezone.utc).isoformat()
        succeeded = sum(1 for r in case_results if r.success)
        failed = len(case_results) - succeeded

        return RunResult(
            run_id=run_id,
            mode="eval",
            started_at=started,
            completed_at=completed,
            config_summary={
                "concurrency": concurrency,
                "repeat_count": repeat_count,
                "base_url": self.config.target.base_url,
                "total_cases": len(cases),
            },
            case_results=list(case_results),
            summary={
                "total": len(case_results),
                "succeeded": succeeded,
                "failed": failed,
                "success_rate": round(succeeded / max(len(case_results), 1), 4),
            },
        )
