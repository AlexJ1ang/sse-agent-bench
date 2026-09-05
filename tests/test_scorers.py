"""评分器单元测试：intent / tool / fact 三个确定性维度。"""

from __future__ import annotations

from agent_harness.eval.fact_scorer import FactScorer
from agent_harness.eval.intent_scorer import IntentScorer
from agent_harness.eval.tool_scorer import ToolScorer
from agent_harness.models import (
    ExpectationSpec,
    SSEEvent,
    SSEStreamResult,
    ToolCallRecord,
)
from agent_harness.models import (
    TestCase as HarnessTestCase,
)


def make_case(**kwargs) -> HarnessTestCase:
    kw = {
        "id": "c1",
        "question": "q",
        **kwargs,
    }
    return HarnessTestCase(**kw)


def make_stream(
    answer_parts: list[str] | None = None,
    tool_names: list[str] | None = None,
    events: list[SSEEvent] | None = None,
) -> SSEStreamResult:
    return SSEStreamResult(
        answer_parts=answer_parts or [],
        tool_calls=[
            ToolCallRecord(call_id=f"t{i}", name=n)
            for i, n in enumerate(tool_names or [])
        ],
        events=events or [],
    )


# ── IntentScorer ───────────────────────────────────────────────


class TestIntentScorer:
    def test_skip_when_no_expectation(self):
        case = make_case()
        result = IntentScorer().score(case, make_stream())
        assert result.skipped is True

    def test_missing_intent_event(self):
        case = make_case(
            expectations=ExpectationSpec(expected_intents={"main": "device_query"})
        )
        result = IntentScorer().score(case, make_stream())
        assert result.passed is False

    def test_match_from_status_detail_dict(self):
        case = make_case(
            expectations=ExpectationSpec(expected_intents={"main": "device_query"})
        )
        stream = make_stream(
            events=[
                SSEEvent(
                    event="status",
                    payload={"stage": "intent_recognition", "detail": {"main": "device_query"}},
                )
            ]
        )
        result = IntentScorer().score(case, stream)
        assert result.passed is True

    def test_mismatch(self):
        case = make_case(
            expectations=ExpectationSpec(expected_intents={"main": "fleet_analysis"})
        )
        stream = make_stream(
            events=[
                SSEEvent(
                    event="status",
                    payload={"stage": "intent_recognition", "detail": {"main": "device_query"}},
                )
            ]
        )
        result = IntentScorer().score(case, stream)
        assert result.passed is False
        assert result.score == 0.0


# ── ToolScorer ────────────────────────────────────────────────


class TestToolScorer:
    def test_skip_when_no_expectation(self):
        case = make_case()
        result = ToolScorer().score(case, make_stream(tool_names=["query_device"]))
        assert result.skipped is True

    def test_exact_match(self):
        case = make_case(
            expectations=ExpectationSpec(expected_tools=["query_device"])
        )
        result = ToolScorer().score(case, make_stream(tool_names=["query_device"]))
        assert result.passed is True
        assert result.score == 1.0

    def test_missing_tool(self):
        case = make_case(
            expectations=ExpectationSpec(expected_tools=["query_device"])
        )
        result = ToolScorer().score(case, make_stream(tool_names=["other_tool"]))
        assert result.passed is False
        assert result.score < 1.0


# ── FactScorer ────────────────────────────────────────────────


class TestFactScorer:
    def test_empty_answer(self):
        case = make_case(
            expectations=ExpectationSpec(expected_keywords=["工时"])
        )
        result = FactScorer().score(case, make_stream())
        assert result.passed is False

    def test_keyword_hit(self):
        case = make_case(
            expectations=ExpectationSpec(expected_keywords=["工作时长", "累计"])
        )
        result = FactScorer().score(
            case, make_stream(answer_parts=["本月累计工作时长 39.72 小时"])
        )
        assert result.passed is True

    def test_keyword_miss(self):
        case = make_case(
            expectations=ExpectationSpec(expected_keywords=["油耗"])
        )
        result = FactScorer().score(
            case, make_stream(answer_parts=["本月累计工作时长 39.72 小时"])
        )
        assert result.passed is False

    def test_pattern_match(self):
        case = make_case(
            expectations=ExpectationSpec(expected_patterns=[r"\d+\.?\d*\s*小时"])
        )
        result = FactScorer().score(
            case, make_stream(answer_parts=["累计 39.72 小时"])
        )
        assert result.passed is True

    def test_reference_number_coverage(self):
        case = make_case(
            expectations=ExpectationSpec(reference_answer="工时 39.72 小时")
        )
        result = FactScorer().score(
            case, make_stream(answer_parts=["累计工时 39.72 小时"])
        )
        assert result.passed is True
