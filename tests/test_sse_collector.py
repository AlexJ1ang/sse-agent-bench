"""SSE 采集器单元测试：事件解析、延迟指标、流中断检测。"""

from __future__ import annotations

import asyncio
from typing import AsyncIterator

from agent_harness.config import SSEEventMapping
from agent_harness.core.sse_collector import (
    SSECollector,
    collect_sse_stream,
    parse_sse_line,
)


class TestParseLine:
    def test_event_line(self):
        event, data, flushed = parse_sse_line("event: answer", "message", [])
        assert event == "answer"

    def test_data_line(self):
        event, data, flushed = parse_sse_line('data: {"x":1}', "answer", [])
        assert data == ['{"x":1}']

    def test_blank_line_flushes(self):
        event, data, flushed = parse_sse_line(
            "", "answer", ['{"content": "hi"}']
        )
        assert flushed == ("answer", {"content": "hi"})


class TestSSECollector:
    def test_answer_accumulation_and_latency(self):
        collector = SSECollector(started_at=1000.0)
        collector.consume("answer", {"content": "你好"}, at=1001.0)
        collector.consume("answer", {"content": "世界"}, at=1002.0)
        collector.consume("done", {"trace_id": "abc"}, at=1003.0)

        assert collector.result.full_answer == "你好世界"
        latency = collector.latency_report()
        assert latency.request_total_ms == 3000.0
        assert latency.request_to_first_answer_ms == 1000.0
        assert collector.result.trace_id == "abc"

    def test_tool_call_duration(self):
        collector = SSECollector(
            started_at=1000.0, mapping=SSEEventMapping()
        )
        collector.consume(
            "tool_call", {"call_id": "t1", "name": "query_device"}, at=1001.0
        )
        collector.consume(
            "status",
            {"tool_name": "query_device", "call_id": "t1", "phase": "completed"},
            at=1003.0,
        )
        record = collector.result.tool_calls[0]
        assert record.duration_ms == 2000.0


def _chunks_from_lines(*lines: str) -> AsyncIterator:
    async def gen():
        for line in lines:
            yield (line + "\n").encode("utf-8")

    return gen()


class TestCollectStream:
    def test_complete_stream_not_interrupted(self):
        async def run():
            stream = _chunks_from_lines(
                "event: answer",
                'data: {"content": "hi"}',
                "",
                "event: done",
                'data: {"trace_id": "t"}',
                "",
            )
            result, _ = await collect_sse_stream(stream, started_at=0.0)
            return result

        result = asyncio.run(run())
        assert result.full_answer == "hi"
        assert result.stream_interrupted is False

    def test_stream_without_done_is_interrupted(self):
        async def run():
            stream = _chunks_from_lines(
                "event: answer",
                'data: {"content": "hi"}',
                "",
            )
            result, _ = await collect_sse_stream(stream, started_at=0.0)
            return result

        result = asyncio.run(run())
        assert result.stream_interrupted is True
