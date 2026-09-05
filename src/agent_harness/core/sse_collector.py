"""SSE 流式事件采集器。

解析 Agent 服务的 SSE 响应，采集所有事件并计算延迟指标。
事件名与 payload 字段名通过 SSEEventMapping 配置注入，框架不预设业务语义。
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, AsyncIterator

from agent_harness.config import SSEEventMapping
from agent_harness.models import (
    LatencyMetrics,
    SSEEvent,
    SSEStreamResult,
    StageTiming,
    ToolCallRecord,
)

logger = logging.getLogger(__name__)


def parse_sse_line(
    line: str,
    event_name: str,
    data_lines: list[str],
) -> tuple[str, list[str], tuple[str, Any] | None]:
    """解析单行 SSE 文本。"""
    if line == "":
        if not data_lines:
            return "message", [], None
        raw = "\n".join(data_lines)
        try:
            payload: Any = json.loads(raw)
        except json.JSONDecodeError:
            payload = raw
        item = (event_name, payload)
        return "message", [], item

    if line.startswith("event:"):
        event_name = line.removeprefix("event:").strip() or "message"
    elif line.startswith("data:"):
        data_lines.append(line.removeprefix("data:").strip())
    return event_name, data_lines, None


class SSECollector:
    """采集 SSE 流并计算延迟指标。

    所有事件名、payload 字段名均从 SSEEventMapping 读取，支持任意被测 Agent。
    """

    def __init__(
        self,
        started_at: float | None = None,
        mapping: SSEEventMapping | None = None,
    ) -> None:
        self.mapping = mapping or SSEEventMapping()
        self.started_at = started_at or time.perf_counter()
        self.result = SSEStreamResult()
        self._first_answer_at: float | None = None
        self._last_answer_at: float | None = None
        self._answering_started_at: float | None = None
        self._done_at: float | None = None
        self._tool_starts: dict[str, ToolCallRecord] = {}
        self._last_step_at: float = self.started_at

    def consume(self, event: str, payload: Any, at: float | None = None) -> None:
        """消费一条 SSE 事件。"""
        at = at or time.perf_counter()
        self.result.events.append(SSEEvent(event=event, payload=payload, received_at=at))

        if not isinstance(payload, dict):
            return

        m = self.mapping

        if event == m.answer_event:
            content = str(payload.get(m.answer_content_field) or "")
            if content.strip():
                if self._first_answer_at is None:
                    self._first_answer_at = at
                self._last_answer_at = at
                self.result.answer_parts.append(content)

        elif event == m.status_event:
            stage = str(payload.get(m.status_stage_field) or "")
            if stage == "answering" and self._answering_started_at is None:
                self._answering_started_at = at

            tool_name = payload.get("tool_name")
            phase = str(payload.get("phase") or "")
            if tool_name and phase != "started":
                self._complete_tool(payload, at)
            elif phase in {"completed", "failed", "skipped"} and not tool_name:
                node = str(payload.get("node") or stage or "status")
                self.result.stage_timings.append(
                    StageTiming(
                        stage=node,
                        phase=phase,
                        step_duration_ms=round((at - self._last_step_at) * 1000, 2),
                        elapsed_ms=round((at - self.started_at) * 1000, 2),
                    )
                )

        elif event == m.tool_call_event:
            call_id = str(
                payload.get(m.tool_call_id_field)
                or f"{payload.get(m.tool_name_field, 'unknown')}:{len(self._tool_starts)}"
            )
            record = ToolCallRecord(
                call_id=call_id,
                name=str(payload.get(m.tool_name_field, "unknown")),
                started_at=at,
                arguments=payload.get(m.tool_args_field) or {},
            )
            self._tool_starts[call_id] = record
            self.result.tool_calls.append(record)

        elif event == m.error_event:
            self.result.error_events.append(
                str(payload.get(m.error_message_field) or payload.get("content") or payload)
            )

        elif event == m.done_event:
            self._done_at = at
            self.result.trace_id = str(payload.get("trace_id") or "")
            timing = payload.get("timing")
            if isinstance(timing, dict):
                nodes = timing.get("workflow_nodes_ms")
                if isinstance(nodes, dict):
                    self.result.workflow_node_timings_ms = {
                        str(k): round(float(v), 2)
                        for k, v in nodes.items()
                        if isinstance(v, (int, float))
                    }

        self._last_step_at = at

    def _complete_tool(self, payload: dict[str, Any], at: float) -> None:
        call_id = str(payload.get(self.mapping.tool_call_id_field) or "")
        tool_name = str(payload.get("tool_name") or "")
        record = self._tool_starts.get(call_id)
        if record is None:
            for r in reversed(self.result.tool_calls):
                if r.name == tool_name and r.completed_at is None:
                    record = r
                    break
        if record is None:
            return
        record.completed_at = at
        record.duration_ms = round((at - record.started_at) * 1000, 2)

    def latency_report(self) -> LatencyMetrics:
        def elapsed(start: float | None, end: float | None) -> float | None:
            if start is None or end is None:
                return None
            return round((end - start) * 1000, 2)

        return LatencyMetrics(
            request_total_ms=elapsed(self.started_at, self._done_at),
            request_to_first_answer_ms=elapsed(self.started_at, self._first_answer_at),
            request_to_answer_complete_ms=elapsed(self.started_at, self._last_answer_at),
            answer_output_duration_ms=elapsed(self._first_answer_at, self._last_answer_at),
            answer_generation_total_ms=elapsed(self._answering_started_at, self._last_answer_at),
            workflow_node_timings_ms=self.result.workflow_node_timings_ms,
            tool_timings=[
                {
                    "name": r.name,
                    "duration_ms": r.duration_ms,
                    "completed_elapsed_ms": elapsed(self.started_at, r.completed_at),
                }
                for r in self.result.tool_calls
                if r.duration_ms is not None
            ],
        )


async def collect_sse_stream(
    async_byte_stream: AsyncIterator[bytes],
    started_at: float | None = None,
    mapping: SSEEventMapping | None = None,
) -> tuple[SSEStreamResult, LatencyMetrics]:
    """从 httpx 异步字节流中采集 SSE 事件。

    若流在收到 ``done`` 事件之前就结束（连接被关闭或异常中断），
    则将 ``stream_interrupted`` 置为 True，用于压测的流中断率统计。
    """
    collector = SSECollector(started_at, mapping)
    event_name = "message"
    data_lines: list[str] = []

    buffer = ""
    try:
        async for chunk in async_byte_stream:
            buffer += chunk.decode("utf-8", errors="replace")
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                line = line.rstrip("\r")
                event_name, data_lines, flushed = parse_sse_line(
                    line, event_name, data_lines
                )
                if flushed is not None:
                    collector.consume(flushed[0], flushed[1])
    except Exception:
        # 读取流过程中被异常打断（连接重置、超时等）
        logger.warning("SSE stream interrupted by exception", exc_info=True)
        collector.result.stream_interrupted = True

    if data_lines:
        raw = "\n".join(data_lines)
        try:
            payload: Any = json.loads(raw)
        except json.JSONDecodeError:
            payload = raw
        collector.consume(event_name, payload)

    # 流正常结束但从未收到 done 事件，视为中断（提前结束/截断）
    if collector._done_at is None and not collector.result.error_events:
        collector.result.stream_interrupted = True

    return collector.result, collector.latency_report()
