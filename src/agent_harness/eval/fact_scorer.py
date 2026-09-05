"""事实一致性评分器（通用版）。

校验 Agent 的回答是否满足用例 expectations 中声明的确定性期望：
关键词存在、正则匹配、参考答案中的数值覆盖。
不含任何领域特定的硬编码规则。
"""

from __future__ import annotations

import re

from agent_harness.models import DimensionScore, ScorerType, SSEStreamResult, TestCase

_NUMBER_PATTERN = re.compile(r"-?\d+(?:\.\d+)?")


class FactScorer:
    """基于 expectations 的通用事实校验。"""

    def score(self, case: TestCase, stream: SSEStreamResult) -> DimensionScore:
        answer = stream.full_answer
        if not answer:
            return DimensionScore(
                scorer=ScorerType.FACT,
                score=0.0,
                passed=False,
                details="回答为空",
                issues=["回答为空"],
            )

        exp = case.expectations
        issues: list[str] = []
        checks_passed = 0
        checks_total = 0

        # 1. 关键词存在性校验
        if exp.expected_keywords:
            checks_total += 1
            missing_kw = [kw for kw in exp.expected_keywords if kw not in answer]
            if not missing_kw:
                checks_passed += 1
            else:
                issues.append(f"回答缺少关键词: {missing_kw[:5]}")

        # 2. 正则模式匹配校验
        if exp.expected_patterns:
            checks_total += 1
            failed_patterns: list[str] = []
            for pattern in exp.expected_patterns:
                try:
                    if not re.search(pattern, answer):
                        failed_patterns.append(pattern)
                except re.error:
                    failed_patterns.append(f"[invalid regex: {pattern}]")
            if not failed_patterns:
                checks_passed += 1
            else:
                issues.append(f"回答未匹配正则: {failed_patterns[:3]}")

        # 3. 参考答案数值覆盖校验
        if exp.reference_answer:
            checks_total += 1
            ref_numbers = set(_NUMBER_PATTERN.findall(exp.reference_answer))
            answer_numbers = set(_NUMBER_PATTERN.findall(answer))
            missing = ref_numbers - answer_numbers
            if not missing or len(missing) <= len(ref_numbers) * 0.5:
                checks_passed += 1
            else:
                issues.append(
                    f"回答缺少参考答案中的关键数值: {sorted(missing)[:5]}"
                )

        if checks_total == 0:
            return DimensionScore(
                scorer=ScorerType.FACT,
                score=1.0,
                skipped=True,
                details="用例未声明事实期望，跳过",
            )

        score = round(checks_passed / checks_total, 4)
        return DimensionScore(
            scorer=ScorerType.FACT,
            score=score,
            passed=not issues,
            details=f"通过 {checks_passed}/{checks_total} 项事实校验",
            issues=issues,
        )
