"""评测管线。

按配置编排启用的评分维度，生成完整的评测报告。
"""

from __future__ import annotations

import logging
from statistics import mean, stdev
from typing import Any

import httpx

from agent_harness.config import HarnessConfig
from agent_harness.core.stats import wilson_interval
from agent_harness.eval.fact_scorer import FactScorer
from agent_harness.eval.intent_scorer import IntentScorer
from agent_harness.eval.llm_judge import LLMJudge
from agent_harness.eval.tool_scorer import ToolScorer
from agent_harness.models import (
    CaseResult,
    DimensionScore,
    EvalScore,
    RunResult,
    ScorerType,
    SSEStreamResult,
    TestCase,
)

logger = logging.getLogger(__name__)


class EvalPipeline:
    """编排多维度评分的管线。"""

    def __init__(self, config: HarnessConfig) -> None:
        self.config = config
        self.enabled = set(config.eval.scorers_enabled)
        self.intent_scorer = IntentScorer()
        self.tool_scorer = ToolScorer()
        self.fact_scorer = FactScorer()
        self.llm_judge = LLMJudge(
            config.eval.judge,
            cache_dir=config.output_dir / "judge_cache",
        )

    async def evaluate(self, run_result: RunResult) -> RunResult:
        """对运行结果中的每条用例进行评分。"""
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(60.0)
        ) as judge_client:
            for case_result in run_result.case_results:
                case_result.eval_score = await self._score_case(
                    case_result, judge_client
                )
                # 重复采样：对每次执行的流独立评分，回填分数并聚合统计
                if case_result.repeat_count > 1:
                    await self._score_repeats(case_result, judge_client)

        run_result.summary.update(self._build_eval_summary(run_result))
        return run_result

    async def _score_case(
        self,
        case_result: CaseResult,
        judge_client: httpx.AsyncClient,
    ) -> EvalScore:
        """对单条用例进行多维度评分（基于代表 stream）。"""
        eval_score = await self._score_stream(
            case_result.case, case_result.stream, judge_client
        )
        # 代表 stream 的评分额外携带延迟与执行错误信息
        return eval_score.model_copy(
            update={
                "latency": case_result.latency,
                "error": case_result.error,
            }
        )

    async def _score_stream(
        self,
        case: TestCase,
        stream: SSEStreamResult,
        judge_client: httpx.AsyncClient,
    ) -> EvalScore:
        """对（case, stream）组合进行多维度评分。"""
        dimensions: list[DimensionScore] = []

        if ScorerType.INTENT.value in self.enabled:
            dimensions.append(self.intent_scorer.score(case, stream))

        if ScorerType.TOOL.value in self.enabled:
            dimensions.append(self.tool_scorer.score(case, stream))

        if ScorerType.FACT.value in self.enabled:
            dimensions.append(self.fact_scorer.score(case, stream))

        if (
            ScorerType.LLM_JUDGE.value in self.enabled
            and self.config.eval.judge.enabled
            and case.expectations.is_open_ended
        ):
            judge_score = await self.llm_judge.score(case, stream, judge_client)
            dimensions.append(judge_score)

        overall = self._compute_overall_score(dimensions)
        all_passed = all(d.passed for d in dimensions)

        return EvalScore(
            case_id=case.id,
            dimensions=dimensions,
            overall_score=overall,
            overall_passed=all_passed,
        )

    async def _score_repeats(
        self,
        case_result: CaseResult,
        judge_client: httpx.AsyncClient,
    ) -> None:
        """对每个 repeat 的 stream 独立评分并回填，然后聚合统计。

        评分前 repeat 只持有 stream，无分数；此方法补齐 overall_score /
        overall_passed，再计算 mean/std/pass_rate/flaky。
        """
        primary_score = case_result.eval_score
        scores: list[float] = []
        passes: list[bool] = []
        evaluated: list[tuple[float, bool]] = []

        for repeat in case_result.repeats:
            if repeat.attempt == 1:
                # 首次执行与 primary 共享 stream/分数
                score = primary_score.overall_score if primary_score else 0.0
                passed = primary_score.overall_passed if primary_score else False
            else:
                eval_score = await self._score_stream(
                    case_result.case, repeat.stream, judge_client
                )
                score = eval_score.overall_score
                passed = eval_score.overall_passed
            repeat.overall_score = round(score, 4)
            repeat.overall_passed = passed
            scores.append(score)
            passes.append(passed)
            evaluated.append((score, passed))

        case_result.mean_score = round(mean(scores), 4) if scores else 0.0
        case_result.std_score = (
            round(stdev(scores), 4) if len(scores) >= 2 else 0.0
        )
        case_result.pass_rate = round(sum(passes) / max(len(passes), 1), 4)
        case_result.flaky = len({p for _, p in evaluated}) > 1

    def _compute_overall_score(self, dimensions: list[DimensionScore]) -> float:
        """按 config.eval.dimension_weights 计算综合得分。"""
        if not dimensions:
            return 0.0
        weights = self.config.eval.dimension_weights
        total = 0.0
        weight_sum = 0.0
        for d in dimensions:
            w = weights.get(d.scorer.value, 0.1)
            normalized = d.score / d.max_score if d.max_score > 0 else 0
            total += normalized * w
            weight_sum += w
        return round(total / max(weight_sum, 0.01), 4)

    @staticmethod
    def _build_eval_summary(run_result: RunResult) -> dict[str, Any]:
        """构建评测汇总统计（含 Wilson 置信区间与 flaky 检测）。"""
        scores = [
            cr.eval_score
            for cr in run_result.case_results
            if cr.eval_score is not None
        ]
        if not scores:
            return {}

        overall_scores = [s.overall_score for s in scores]
        avg_score = sum(overall_scores) / len(overall_scores)
        passed = sum(1 for s in scores if s.overall_passed)

        # Wilson 置信区间
        _, pass_lower, pass_upper = wilson_interval(passed, len(scores))

        # Flaky 检测
        flaky_cases = [
            {"case_id": cr.case.id, "pass_rate": cr.pass_rate}
            for cr in run_result.case_results
            if cr.flaky
        ]

        # Judge 分歧率（low_confidence 标注的 llm_judge 维度占比）
        judge_total = 0
        judge_low_conf = 0
        for s in scores:
            for d in s.dimensions:
                if d.scorer == ScorerType.LLM_JUDGE:
                    judge_total += 1
                    if d.low_confidence:
                        judge_low_conf += 1
        judge_disagreement_rate = (
            round(judge_low_conf / judge_total, 4) if judge_total else None
        )

        dimension_stats: dict[str, dict[str, float]] = {}
        for s in scores:
            for d in s.dimensions:
                key = d.scorer.value
                if key not in dimension_stats:
                    dimension_stats[key] = {"total": 0, "count": 0, "avg": 0.0}
                dimension_stats[key]["total"] += d.score
                dimension_stats[key]["count"] += 1
        for key, stats in dimension_stats.items():
            stats["avg"] = round(stats["total"] / max(stats["count"], 1), 4)

        return {
            "eval_avg_score": round(avg_score, 4),
            "eval_pass_rate": round(passed / len(scores), 4),
            "eval_pass_rate_ci": [round(pass_lower, 4), round(pass_upper, 4)],
            "eval_passed": passed,
            "eval_failed": len(scores) - passed,
            "flaky_count": len(flaky_cases),
            "flaky_cases": flaky_cases,
            "judge_disagreement_rate": judge_disagreement_rate,
            "judge_low_confidence_count": judge_low_conf,
            "judge_total": judge_total,
            "dimension_stats": dimension_stats,
        }
