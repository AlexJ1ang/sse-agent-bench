"""意图识别评分器（通用版）。

校验 Agent 输出的意图/路由分类是否与用例期望一致。
意图标签的提取方式：从 SSE 事件中按可配置的 stage 名匹配，
支持任意层级的 intent key（如 main/sub/intent/route 等）。
"""

from __future__ import annotations

import json
import logging
from typing import Any

from agent_harness.models import DimensionScore, ScorerType, SSEStreamResult, TestCase

logger = logging.getLogger(__name__)


class IntentScorer:
    """基于 SSE 事件中的意图分类结果进行评分。

    从用例 expectations.expected_intents 读取期望，
    从 SSE status 事件中提取实际值，做宽松匹配。
    """

    def __init__(self, intent_stage: str = "intent_recognition") -> None:
        self.intent_stage = intent_stage

    def score(self, case: TestCase, stream: SSEStreamResult) -> DimensionScore:
        expected = case.expectations.expected_intents
        if not expected:
            return DimensionScore(
                scorer=ScorerType.INTENT,
                score=1.0,
                skipped=True,
                details="用例未标注意图期望，跳过",
            )

        detected = self._extract_intent(stream)
        if detected is None:
            return DimensionScore(
                scorer=ScorerType.INTENT,
                score=0.0,
                passed=False,
                details="SSE 流中未检测到意图识别结果",
                issues=["未找到意图事件"],
            )

        matched = 0
        total = len(expected)
        issues: list[str] = []

        for key, expected_val in expected.items():
            actual_val = str(detected.get(key, ""))
            if self._match(actual_val, expected_val):
                matched += 1
            else:
                issues.append(
                    f"意图 '{key}' 不匹配：期望 '{expected_val}'，实际 '{actual_val}'"
                )

        score = round(matched / max(total, 1), 4)
        return DimensionScore(
            scorer=ScorerType.INTENT,
            score=score,
            passed=matched == total,
            details=json.dumps(detected, ensure_ascii=False),
            issues=issues,
        )

    def _extract_intent(self, stream: SSEStreamResult) -> dict[str, Any] | None:
        """从 SSE 事件中提取意图识别结果。

        查找 status 事件中 stage 匹配的 payload，
        支持 detail 为 dict 或 JSON 字符串两种形式。
        """
        for event in stream.events:
            if not isinstance(event.payload, dict):
                continue
            payload = event.payload
            if payload.get("stage") != self.intent_stage:
                continue
            detail = payload.get("detail")
            if isinstance(detail, dict):
                return detail
            if isinstance(detail, str):
                try:
                    parsed = json.loads(detail)
                    if isinstance(parsed, dict):
                        return parsed
                except json.JSONDecodeError:
                    continue
            # 有些 Agent 直接把 intent 字段平铺在 status payload 里
            intent_keys = {
                k: v for k, v in payload.items()
                if k not in {"stage", "phase", "node", "message"}
            }
            if intent_keys:
                return intent_keys
        return None

    @staticmethod
    def _match(detected: str, expected: str) -> bool:
        """宽松匹配：精确或包含。"""
        if not detected or not expected:
            return not expected
        d, e = detected.strip().lower(), expected.strip().lower()
        return d == e or e in d or d in e
