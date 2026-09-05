"""Trace 持久化与三维 diff 单元测试。"""

from __future__ import annotations

import json
from pathlib import Path

from agent_harness.core.trace_store import (
    build_trace_diff,
    list_case_ids,
    list_run_ids,
    load_trace,
    save_run_traces,
)
from agent_harness.models import (
    CaseResult,
    EvalScore,
    RunResult,
    SSEStreamResult,
    StageTiming,
    ToolCallRecord,
)
from agent_harness.models import (
    TestCase as HarnessTestCase,
)


def _make_case_result(case_id: str, answer: str) -> CaseResult:
    case = HarnessTestCase(id=case_id, question="q")
    stream = SSEStreamResult(
        answer_parts=[answer],
        tool_calls=[
            ToolCallRecord(call_id="1", name="search", arguments={"q": "x"}, duration_ms=10.0)
        ],
        stage_timings=[StageTiming(stage="plan", phase="completed", step_duration_ms=20.0)],
    )
    return CaseResult(
        case=case,
        stream=stream,
        success=True,
        request_payload={"query": "q"},
        eval_score=EvalScore(case_id=case_id, overall_score=0.5, overall_passed=True),
    )


def _roundtrip(tmp_path: Path) -> dict:
    run = RunResult(
        run_id="run1",
        mode="eval",
        case_results=[_make_case_result("c1", "hello world")],
    )
    save_run_traces(run, tmp_path)
    return load_trace(tmp_path, "run1", "c1") or {}


def test_save_and_load_roundtrip(tmp_path):
    trace = _roundtrip(tmp_path)
    assert trace["case_id"] == "c1"
    assert trace["stream"]["full_answer"] == "hello world"
    assert trace["payload"]["query"] == "q"
    assert trace["stream"]["tool_calls"][0]["name"] == "search"
    assert trace["stream"]["stage_timings"][0]["stage"] == "plan"


def test_list_run_ids_and_case_ids(tmp_path):
    run = RunResult(
        run_id="run1",
        mode="eval",
        case_results=[
            _make_case_result("c1", "a"),
            _make_case_result("c2", "b"),
        ],
    )
    save_run_traces(run, tmp_path)
    assert list_run_ids(tmp_path) == ["run1"]
    assert list_case_ids(tmp_path, "run1") == ["c1", "c2"]


def test_load_missing_trace_returns_none(tmp_path):
    assert load_trace(tmp_path, "nope", "c1") is None


def test_self_diff_is_unchanged(tmp_path):
    trace = _roundtrip(tmp_path)
    diff = build_trace_diff(trace, trace)
    assert diff["changed"] is False
    assert diff["answer_changed"] is False
    assert diff["tool_changed"] is False
    assert diff["latency_changed"] is False


def test_answer_diff_detected(tmp_path):
    trace = _roundtrip(tmp_path)
    other = json.loads(json.dumps(trace))
    other["stream"]["full_answer"] = "hello WORLD"
    diff = build_trace_diff(trace, other)
    assert diff["answer_changed"] is True
    assert "+hello WORLD" in diff["answer_diff"]
    assert "-hello" in diff["answer_diff"]


def test_tool_sequence_diff_detected(tmp_path):
    trace = _roundtrip(tmp_path)
    other = json.loads(json.dumps(trace))
    other["stream"]["tool_calls"] = [
        {"name": "search", "arguments": {"q": "x"}},
        {"name": "calc", "arguments": {}},
    ]
    diff = build_trace_diff(trace, other)
    assert diff["tool_changed"] is True
    op_tags = {op["tag"] for op in diff["tool_opcodes"]}
    assert "insert" in op_tags


def test_tool_argument_diff_detected(tmp_path):
    trace = _roundtrip(tmp_path)
    other = json.loads(json.dumps(trace))
    other["stream"]["tool_calls"][0]["arguments"] = {"q": "y"}
    diff = build_trace_diff(trace, other)
    assert diff["tool_changed"] is True
    assert len(diff["tool_arg_diffs"]) == 1
    assert diff["tool_arg_diffs"][0]["tool"] == "search"


def test_latency_diff_detected(tmp_path):
    trace = _roundtrip(tmp_path)
    other = json.loads(json.dumps(trace))
    other["stream"]["stage_timings"][0]["step_duration_ms"] = 99.0
    diff = build_trace_diff(trace, other)
    assert diff["latency_changed"] is True
