"""工具调用评分器（通用版）。

校验 Agent 调用的工具是否与用例期望一致。
从用例 expectations.expected_tools 读取期望，与实际 tool_call 事件对比。
"""

from __future__ import annotations

from agent_harness.models import DimensionScore, ScorerType, SSEStreamResult, TestCase


class ToolScorer:
    """基于 tool_call 事件评估工具选择的正确性。"""

    def score(self, case: TestCase, stream: SSEStreamResult) -> DimensionScore:
        expected = case.expectations.expected_tools
        actual_tool_names = [r.name for r in stream.tool_calls]

        if not expected:
            return DimensionScore(
                scorer=ScorerType.TOOL,
                score=1.0,
                skipped=True,
                details=f"用例未标注期望工具，实际调用: {actual_tool_names}",
            )

        expected_set = set(expected)
        actual_set = set(actual_tool_names)

        missing = expected_set - actual_set
        extra = actual_set - expected_set

        if not missing and not extra:
            return DimensionScore(
                scorer=ScorerType.TOOL,
                score=1.0,
                details=f"工具调用完全匹配: {sorted(expected_set)}",
            )

        coverage = len(expected_set & actual_set) / max(len(expected_set), 1)
        precision = len(expected_set & actual_set) / max(len(actual_set), 1)
        score = round((coverage + precision) / 2, 4)

        issues: list[str] = []
        if missing:
            issues.append(f"缺少期望的工具调用: {sorted(missing)}")
        if extra:
            issues.append(f"存在未期望的工具调用: {sorted(extra)}")

        return DimensionScore(
            scorer=ScorerType.TOOL,
            score=score,
            passed=not missing,
            details=f"期望: {sorted(expected_set)}, 实际: {sorted(actual_set)}",
            issues=issues,
        )
