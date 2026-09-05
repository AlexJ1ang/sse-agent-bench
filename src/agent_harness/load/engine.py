"""性能压测引擎。

基于 asyncio 的并发压测，支持 QPS 控制、延迟分位数统计。
"""

from __future__ import annotations

import asyncio
import logging
import statistics
import time
import uuid
from datetime import datetime, timezone

import httpx

from agent_harness.config import HarnessConfig
from agent_harness.core.payload import build_payload
from agent_harness.core.sse_collector import collect_sse_stream
from agent_harness.models import (
    LoadSample,
    LoadTestResult,
    TestCase,
)

logger = logging.getLogger(__name__)


def _percentile_value(sorted_values: list[float], p: float) -> float:
    """对已排序列表计算第 p 分位数（线性插值，H&L 法 rank = p*(n-1)）。

    相比 int(n*p) 下取整，避免 p99 等高分位在小样本下被系统性低估。
    """
    if not sorted_values:
        return 0.0
    n = len(sorted_values)
    if n == 1:
        return sorted_values[0]
    rank = p * (n - 1)
    lo = int(rank)
    hi = min(lo + 1, n - 1)
    frac = rank - lo
    return sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac


class LoadEngine:
    """压测引擎。"""

    def __init__(self, config: HarnessConfig) -> None:
        self.config = config
        self.target = config.target

    async def run(
        self,
        cases: list[TestCase],
        concurrency: int | None = None,
        duration_seconds: int | None = None,
    ) -> LoadTestResult:
        """执行压测。

        策略：从用例列表中随机选取，按指定并发数持续发送请求，
        直到达到持续时间上限。
        """
        if not cases:
            raise ValueError("压测用例集为空，无法执行；请提供至少一条用例")

        run_id = uuid.uuid4().hex[:12]
        concurrency = concurrency or self.config.load.concurrency
        duration = duration_seconds or self.config.load.duration_seconds
        ramp_up = self.config.load.ramp_up_seconds
        ramp_up = min(ramp_up, duration)  # 渐进加压不得超过总时长
        started = datetime.now(timezone.utc).isoformat()

        samples: list[LoadSample] = []
        end_time = time.perf_counter() + duration

        timeout = httpx.Timeout(
            connect=self.target.connect_timeout,
            read=self.target.read_timeout,
            write=self.target.connect_timeout,
            pool=self.target.read_timeout,
        )

        async with httpx.AsyncClient(timeout=timeout) as client:
            async def worker(worker_id: int) -> None:
                # 渐进加压：第 i 个 worker 在 ramp_up * i / concurrency 时刻才启动
                if ramp_up > 0 and concurrency > 1:
                    delay = ramp_up * worker_id / concurrency
                    await asyncio.sleep(delay)
                case_idx = worker_id % len(cases)
                while time.perf_counter() < end_time:
                    case = cases[case_idx % len(cases)]
                    sample = await self._execute_one(client, case)
                    samples.append(sample)
                    case_idx += 1
                    if self.config.load.think_time_ms > 0:
                        await asyncio.sleep(self.config.load.think_time_ms / 1000)

            tasks = [worker(i) for i in range(concurrency)]
            await asyncio.gather(*tasks, return_exceptions=True)

        completed = datetime.now(timezone.utc).isoformat()

        return self._build_result(
            run_id, started, completed, samples, duration
        )

    async def _execute_one(
        self,
        client: httpx.AsyncClient,
        case: TestCase,
    ) -> LoadSample:
        """执行单条压测请求。"""
        started_at = time.perf_counter()
        sample = LoadSample(
            case_id=case.id,
            started_at=started_at,
            completed_at=started_at,
        )

        try:
            payload = build_payload(self.config, case)
            url = f"{self.target.base_url}{self.target.chat_path}"
            headers = {
                "Accept": "text/event-stream",
                **self.target.request_adapter.extra_headers,
            }

            async with client.stream(
                self.target.method, url, json=payload, headers=headers
            ) as response:
                response.raise_for_status()
                stream_result, latency = await collect_sse_stream(
                    response.aiter_bytes(), started_at, self.target.sse_mapping
                )
                sample.latency = latency
                sample.sse_event_count = stream_result.event_count
                sample.answer_length = len(stream_result.full_answer)
                sample.stream_interrupted = stream_result.stream_interrupted
                sample.success = True

        except Exception as exc:
            sample.success = False
            sample.error = str(exc)[:200]

        sample.completed_at = time.perf_counter()
        return sample

    def _build_result(
        self,
        run_id: str,
        started: str,
        completed: str,
        samples: list[LoadSample],
        duration: float,
    ) -> LoadTestResult:
        """汇总压测结果。"""
        successful = [s for s in samples if s.success]
        failed = [s for s in samples if not s.success]

        latencies = [
            s.latency.request_total_ms
            for s in successful
            if s.latency.request_total_ms is not None
        ]
        ttfts = [
            s.latency.request_to_first_answer_ms
            for s in successful
            if s.latency.request_to_first_answer_ms is not None
        ]

        def percentiles(values: list[float]) -> dict[str, float]:
            if not values:
                return {}
            s = sorted(values)

            return {
                "p50": round(_percentile_value(s, 0.50), 2),
                "p90": round(_percentile_value(s, 0.90), 2),
                "p95": round(_percentile_value(s, 0.95), 2),
                "p99": round(_percentile_value(s, 0.99), 2),
                "avg": round(statistics.mean(values), 2),
                "min": round(min(values), 2),
                "max": round(max(values), 2),
            }

        interrupted = sum(1 for s in samples if s.stream_interrupted)

        # ── 节点级延迟分解 ──
        # 聚合所有成功样本的 workflow_node_timings_ms，给出每个工作流节点的
        # 平均耗时与发生次数，用于把总延迟拆解到具体 Agent 节点。
        node_aggregate: dict[str, list[float]] = {}
        for s in successful:
            for node, ms in s.latency.workflow_node_timings_ms.items():
                node_aggregate.setdefault(node, []).append(ms)

        node_breakdown: dict[str, dict[str, float]] = {}
        for node, ms_list in node_aggregate.items():
            sorted_ms = sorted(ms_list)
            node_breakdown[node] = {
                "avg_ms": round(statistics.mean(ms_list), 2),
                "p95_ms": round(_percentile_value(sorted_ms, 0.95), 2)
                if ms_list else 0.0,
                "count": float(len(ms_list)),
            }

        # ── 事件吞吐：SSE 事件总吞吐量（事件/秒）与单请求事件数 ──
        total_events = sum(s.sse_event_count for s in samples)
        event_counts = [s.sse_event_count for s in successful]

        latency_breakdown = {
            "event_throughput_per_sec": round(total_events / duration, 2)
            if duration > 0 else 0.0,
            "avg_events_per_request": round(statistics.mean(event_counts), 2)
            if event_counts else 0.0,
            "node_breakdown": node_breakdown,
        }

        return LoadTestResult(
            run_id=run_id,
            scenario_name=f"load-{self.config.load.concurrency}concurrent",
            started_at=started,
            completed_at=completed,
            duration_seconds=duration,
            total_requests=len(samples),
            successful_requests=len(successful),
            failed_requests=len(failed),
            samples=samples,
            latency_percentiles=percentiles(latencies),
            ttft_percentiles=percentiles(ttfts),
            throughput_rps=round(len(samples) / duration, 2) if duration > 0 else 0,
            error_rate=round(len(failed) / max(len(samples), 1), 4),
            interruption_rate=round(interrupted / max(len(samples), 1), 4),
            latency_breakdown=latency_breakdown,
        )
