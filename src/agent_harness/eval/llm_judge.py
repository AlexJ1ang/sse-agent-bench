"""LLM-as-Judge 评分器。

对开放式回答进行多维度质量评分。
包含评分 Rubric、缓存、结果解析。
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import httpx

from agent_harness.config import JudgeConfig
from agent_harness.models import DimensionScore, ScorerType, SSEStreamResult, TestCase

logger = logging.getLogger(__name__)

DEFAULT_JUDGE_SYSTEM_PROMPT = """\
你是一个 LLM Agent 回答质量评审员。请根据以下维度对用户问题的回答进行评分。

## 评分维度（每项 0-5 分）

1. **事实准确性 (accuracy)**：回答中的数值、实体、日期是否与工具返回的原始数据一致。出现任何编造的数据记 0 分。
2. **完整性 (completeness)**：是否回答了用户的全部问题，关键信息是否遗漏。
3. **逻辑性 (coherence)**：回答结构是否清晰，推理过程是否合理。
4. **格式规范 (format)**：是否使用了合适的展示方式（表格/图表/文字），格式是否整洁。
5. **语言质量 (language)**：语言是否流畅自然，是否符合用户偏好。

## 推理过程要求（Chain-of-Thought）

评分前，请先在 `reasoning` 字段中逐维度分析，**必须引用回答原文的关键片段或工具返回数据作为佐证**，
再给出该维度的分数。禁止只给分数而不给理由。

## 评分规则

- 每个维度给出 0-5 的整数分数
- 每个维度给出简短的一行评语说明扣分原因（没有扣分则留空）
- 总分 = 各维度加权平均：accuracy 权重 0.35，completeness 0.25，coherence 0.15，format 0.10，language 0.15

## 输出格式

严格按以下 JSON 格式输出，不要输出其他内容：
```json
{
  "reasoning": {
    "accuracy": "引用原文片段并说明准确性判断",
    "completeness": "引用原文片段并说明完整性判断",
    "coherence": "引用原文片段并说明逻辑性判断",
    "format": "引用原文片段并说明格式判断",
    "language": "引用原文片段并说明语言质量判断"
  },
  "dimensions": {
    "accuracy": {"score": 5, "comment": ""},
    "completeness": {"score": 4, "comment": "缺少部分数据"},
    "coherence": {"score": 5, "comment": ""},
    "format": {"score": 4, "comment": "表格缺少单位"},
    "language": {"score": 5, "comment": ""}
  },
  "weighted_total": 4.65,
  "summary": "一句话总评"
}
```
"""

# 自一致性校验用的变体 prompt：维度顺序打乱 + 措辞模板替换，
# 用于检测 Judge 的 position bias（评分是否随维度排序位置漂移）。
JUDGE_VARIANT_PROMPT_SUFFIX = """\

## 重要：本轮评分顺序说明

请按以下**重新排序后的维度顺序**逐一评估（顺序不代表重要性）：
format → coherence → accuracy → language → completeness

评分规则与权重不变。请仍然先输出 reasoning（按上面的新顺序），再给 dimensions。
"""

JUDGE_PROMPT_VERSION_V1 = "v1"
JUDGE_PROMPT_VERSION_COT = "v2-cot"
JUDGE_PROMPT_VERSION_COT_VARIANT = "v2-cot-variant"

JUDGE_USER_TEMPLATE = """\
## 用户问题

{question}

## 用户上下文

{context}

## Agent 调用的工具及返回数据

{tool_context}

## Agent 的回答

{answer}

## 参考答案（如有）

{reference_answer}

请按系统提示中的评分维度和输出格式进行评分。
"""

DIMENSION_WEIGHTS = {
    "accuracy": 0.35,
    "completeness": 0.25,
    "coherence": 0.15,
    "format": 0.10,
    "language": 0.15,
}


class JudgeCache:
    """Judge 评分结果缓存，按 (用例ID + 回答hash + prompt版本) 去重。"""

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_key(self, case_id: str, answer: str, prompt_version: str) -> str:
        content = f"{case_id}:{answer}:{prompt_version}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def get(
        self, case_id: str, answer: str, prompt_version: str = "v1"
    ) -> dict[str, Any] | None:
        key = self._cache_key(case_id, answer, prompt_version)
        path = self.cache_dir / f"{key}.json"
        if not path.exists():
            return None
        try:
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return None

    def put(
        self,
        case_id: str,
        answer: str,
        result: dict[str, Any],
        prompt_version: str = "v1",
    ) -> None:
        key = self._cache_key(case_id, answer, prompt_version)
        path = self.cache_dir / f"{key}.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)


class LLMJudge:
    """LLM-as-Judge 评分器。"""

    def __init__(self, config: JudgeConfig, cache_dir: Path | None = None) -> None:
        self.config = config
        self.cache = (
            JudgeCache(cache_dir or Path("harness_output/judge_cache"))
            if config.cache_enabled
            else None
        )

    async def score(
        self,
        case: TestCase,
        stream: SSEStreamResult,
        client: httpx.AsyncClient | None = None,
    ) -> DimensionScore:
        """对一条用例的回答进行 LLM 评分。

        支持 Chain-of-Thought 评分与 position-bias 自一致性校验：
        - cot_enabled 时使用 CoT prompt（先 reasoning 再 scores）
        - consistency_check 时评两次（第二次维度顺序打乱 + 换措辞），
          两次 weighted_total 差异 > 阈值则标注 low_confidence
        """
        if not self.config.enabled:
            return DimensionScore(
                scorer=ScorerType.LLM_JUDGE,
                score=0.0,
                max_score=5.0,
                details="LLM Judge 已禁用",
            )

        answer = stream.full_answer
        if not answer:
            return DimensionScore(
                scorer=ScorerType.LLM_JUDGE,
                score=0.0,
                max_score=5.0,
                passed=False,
                details="回答为空",
                issues=["回答为空"],
            )

        tool_context = self._build_tool_context(stream)
        user_prompt = JUDGE_USER_TEMPLATE.format(
            question=case.question,
            context=json.dumps(case.context, ensure_ascii=False) if case.context else "（无）",
            tool_context=tool_context,
            answer=answer,
            reference_answer=case.expectations.reference_answer or "（无参考答案）",
        )

        system_prompt = self.config.system_prompt or DEFAULT_JUDGE_SYSTEM_PROMPT
        prompt_version = (
            JUDGE_PROMPT_VERSION_COT if self.config.cot_enabled else JUDGE_PROMPT_VERSION_V1
        )

        # 首次评分（带缓存）
        judge_result = await self._call_judge_cached(
            case.id, answer, system_prompt, user_prompt, prompt_version, client
        )
        if judge_result is None:
            return DimensionScore(
                scorer=ScorerType.LLM_JUDGE,
                score=0.0,
                max_score=5.0,
                passed=False,
                details="LLM Judge 调用失败",
                issues=["Judge 调用失败"],
            )

        dimension_score = self._parse_judge_result(judge_result)

        # ── 自一致性校验 ──
        if self.config.consistency_check:
            variant_system = system_prompt + JUDGE_VARIANT_PROMPT_SUFFIX
            variant_result = await self._call_judge_cached(
                case.id, answer, variant_system, user_prompt,
                JUDGE_PROMPT_VERSION_COT_VARIANT, client,
            )
            if variant_result is not None:
                primary_total = self._weighted_total(judge_result)
                variant_total = self._weighted_total(variant_result)
                diff = abs(primary_total - variant_total)
                if diff > self.config.consistency_threshold:
                    dimension_score.low_confidence = True
                    dimension_score.details += (
                        f" | low_confidence(两次评分差 {diff:.2f} > "
                        f"{self.config.consistency_threshold})"
                    )

        return dimension_score

    async def _call_judge_cached(
        self,
        case_id: str,
        answer: str,
        system_prompt: str,
        user_prompt: str,
        prompt_version: str,
        client: httpx.AsyncClient | None = None,
    ) -> dict[str, Any] | None:
        """带缓存的 Judge 调用（缓存 key 含 prompt 版本号）。"""
        if self.cache:
            cached = self.cache.get(case_id, answer, prompt_version)
            if cached:
                return cached

        result = await self._call_judge(system_prompt, user_prompt, client)
        if result is not None and self.cache:
            self.cache.put(case_id, answer, result, prompt_version)
        return result

    @staticmethod
    def _weighted_total(result: dict[str, Any]) -> float:
        """从 judge 原始 JSON 中取加权总分（缺失则按维度权重计算）。"""
        total = result.get("weighted_total", 0.0)
        if total:
            return float(total)
        dimensions = result.get("dimensions", {})
        total = 0.0
        weight_sum = 0.0
        for dim_name, weight in DIMENSION_WEIGHTS.items():
            dim_data = dimensions.get(dim_name, {})
            total += dim_data.get("score", 0) * weight
            weight_sum += weight
        return total / max(weight_sum, 0.01)

    async def _call_judge(
        self,
        system_prompt: str,
        user_prompt: str,
        client: httpx.AsyncClient | None = None,
    ) -> dict[str, Any] | None:
        """调用 Judge LLM 进行评分。"""
        own_client = client is None
        if own_client:
            client = httpx.AsyncClient(timeout=httpx.Timeout(60.0))
        assert client is not None

        try:
            url = f"{self.config.api_base}{self.config.chat_path}"
            headers = {"Authorization": f"Bearer {self.config.api_key}"}
            payload = {
                "model": self.config.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
                "stream": False,
            }

            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            body = response.json()

            content = self._extract_content(body)
            if not content:
                logger.warning("Judge returned empty content")
                return None

            return self._parse_json_from_content(content)

        except Exception:
            logger.exception("LLM Judge call failed")
            return None
        finally:
            if own_client:
                await client.aclose()

    @staticmethod
    def _extract_content(body: dict[str, Any]) -> str:
        """从 LLM 响应中提取文本内容。"""
        choices = body.get("choices", [])
        if not choices:
            return ""
        message = choices[0].get("message", {})
        return str(message.get("content") or "")

    @staticmethod
    def _parse_json_from_content(content: str) -> dict[str, Any] | None:
        """从 LLM 输出中解析 JSON。"""
        try:
            start = content.index("```json")
            end = content.index("```", start + 7)
            return json.loads(content[start + 7:end].strip())
        except (ValueError, json.JSONDecodeError):
            pass
        try:
            start = content.index("{")
            end = content.rindex("}") + 1
            return json.loads(content[start:end])
        except (ValueError, json.JSONDecodeError):
            logger.warning("Failed to parse Judge JSON: %s", content[:200])
            return None

    @staticmethod
    def _build_tool_context(stream: SSEStreamResult) -> str:
        """构建工具调用上下文描述。"""
        if not stream.tool_calls:
            return "（无工具调用）"
        lines: list[str] = []
        for tc in stream.tool_calls:
            duration = f"{tc.duration_ms:.0f}ms" if tc.duration_ms else "N/A"
            lines.append(f"- 工具: {tc.name} | 耗时: {duration}")
        return "\n".join(lines)

    def _parse_judge_result(
        self,
        result: dict[str, Any],
        from_cache: bool = False,
    ) -> DimensionScore:
        """将 Judge 的原始 JSON 结果转换为 DimensionScore。"""
        dimensions = result.get("dimensions", {})
        reasoning = result.get("reasoning", {})
        summary = result.get("summary", "")
        weighted_total = self._weighted_total(result)
        weighted_total = round(weighted_total, 2)

        normalized_score = round(weighted_total / 5.0, 4)

        issues: list[str] = []
        for dim_name, dim_data in dimensions.items():
            comment = dim_data.get("comment", "")
            score = dim_data.get("score", 5)
            if comment and score < 4:
                issues.append(f"{dim_name}({score}/5): {comment}")

        details_parts = [f"weighted_total={weighted_total}/5"]
        if reasoning:
            # 附上逐维度推理摘要（截断，避免 details 过长）
            reasoning_short = " | ".join(
                f"{k}: {str(v)[:60]}" for k, v in list(reasoning.items())[:5]
            )
            details_parts.append(f"[reasoning] {reasoning_short}")
        if summary:
            details_parts.append(f"总结: {summary}")
        if from_cache:
            details_parts.append("(cached)")

        return DimensionScore(
            scorer=ScorerType.LLM_JUDGE,
            score=normalized_score,
            max_score=5.0,
            passed=weighted_total >= self.config.pass_threshold,
            details=" | ".join(details_parts),
            issues=issues,
        )
