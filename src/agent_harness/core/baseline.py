"""基线管理。

保存和加载评测基线，支持跨版本回归对比。
对比时引入 McNemar 检验、paired t-test、Wilson 置信区间，
将「裸 delta 阈值」升级为「三档统计显著性判断」：

- ``significant_improvement``   显著提升（p < 0.05 且方向为正）
- ``no_significant_change``     无显著变化
- ``significant_regression``    显著回归（p < 0.05 且方向为负）

只有 ``significant_regression`` 才会计入 ``--gate-max-regressions``，
从而过滤掉随机噪声导致的假回归。
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

from agent_harness.core.stats import (
    describe_change,
    mcnemar_p_value,
    paired_t_test,
    wilson_interval,
)
from agent_harness.models import RunResult

# 显著性水平
ALPHA = 0.05
# 方向判定阈值（delta 绝对值过小视为无变化，避免浮点抖动）
DELTA_EPS = 0.01


def _significance_verdict(p_value: float, direction: float) -> str:
    """根据 p 值与方向给出三档结论。

    Args:
        p_value: 显著性检验 p 值
        direction: 差异方向（current - baseline 的均值差）
    """
    if p_value < ALPHA:
        return "significant_regression" if direction < 0 else "significant_improvement"
    return "no_significant_change"


class BaselineStore:
    """评测基线的持久化存储。"""

    def __init__(self, baseline_dir: Path) -> None:
        self.baseline_dir = baseline_dir
        self.baseline_dir.mkdir(parents=True, exist_ok=True)

    def save(self, name: str, run_result: RunResult) -> Path:
        """保存一次运行结果为基线。"""
        path = self.baseline_dir / f"{name}.json"
        data = self._serialize_run(run_result)
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return path

    def load(self, name: str) -> dict[str, Any] | None:
        """加载基线。"""
        path = self.baseline_dir / f"{name}.json"
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def list_baselines(self) -> list[str]:
        """列出所有基线名称。"""
        return sorted(p.stem for p in self.baseline_dir.glob("*.json"))

    def compare(
        self,
        current: RunResult,
        baseline_name: str = "latest",
    ) -> dict[str, Any]:
        """将当前运行结果与基线对比，输出回归报告（含统计显著性检验）。"""
        baseline = self.load(baseline_name)
        if baseline is None:
            return {"error": f"baseline '{baseline_name}' not found"}

        current_scores = self._extract_scores(current)
        baseline_scores = self._extract_scores_from_dict(baseline)
        # 重复采样分布（用于 per-case 显著性检验）
        current_dist = self._extract_repeat_distributions(current)
        baseline_dist = self._extract_repeat_distributions_from_dict(baseline)

        regressions: list[dict[str, Any]] = []
        improvements: list[dict[str, Any]] = []
        new_cases: list[str] = []
        removed_cases: list[str] = []

        # 配对统计（aggregate 层面）
        b_pairs: list[int] = []  # baseline pass -> current fail
        c_pairs: list[int] = []  # baseline fail -> current pass
        paired_current: list[float] = []
        paired_baseline: list[float] = []

        for case_id, current_score in current_scores.items():
            if case_id not in baseline_scores:
                new_cases.append(case_id)
                continue
            base_score = baseline_scores[case_id]
            base_passed = base_score >= 0.5  # 阈值判断
            current_passed = current_score >= 0.5

            if base_passed and not current_passed:
                b_pairs.append(1)
            elif not base_passed and current_passed:
                c_pairs.append(1)

            paired_current.append(current_score)
            paired_baseline.append(base_score)

            delta = current_score - base_score
            entry: dict[str, Any] = {
                "case_id": case_id,
                "baseline_score": round(base_score, 4),
                "current_score": round(current_score, 4),
                "delta": round(delta, 4),
            }

            # per-case 显著性检验：双方都存在重复采样分布（≥2 个样本）时可做
            base_vals = baseline_dist.get(case_id, [])
            curr_vals = current_dist.get(case_id, [])
            if len(base_vals) >= 2 and len(curr_vals) >= 2:
                _, per_p = paired_t_test(base_vals, curr_vals)
                entry["p_value"] = round(per_p, 4)
                entry["significance"] = _significance_verdict(per_p, delta)
            else:
                entry["p_value"] = None
                entry["significance"] = None

            if delta < -DELTA_EPS:
                regressions.append(entry)
            elif delta > DELTA_EPS:
                improvements.append(entry)

        for case_id in baseline_scores:
            if case_id not in current_scores:
                removed_cases.append(case_id)

        # ── aggregate 统计显著性检验 ──
        mcnemar_p = mcnemar_p_value(sum(b_pairs), sum(c_pairs))
        t_stat, t_p = paired_t_test(paired_baseline, paired_current)
        overall_direction = (
            (mean(paired_current) - mean(paired_baseline))
            if paired_current
            else 0.0
        )
        # 整体结论：连续分数以 paired t-test 为准
        overall_verdict = _significance_verdict(t_p, overall_direction)

        # Wilson 区间
        current_pass = sum(1 for s in paired_current if s >= 0.5)
        total = len(paired_current) if paired_current else 1
        _, curr_lower, curr_upper = wilson_interval(current_pass, total)
        base_pass = sum(1 for s in paired_baseline if s >= 0.5)
        _, base_lower, base_upper = wilson_interval(base_pass, total)

        avg_current = sum(paired_current) / max(len(paired_current), 1)
        avg_baseline = sum(paired_baseline) / max(len(paired_baseline), 1)

        # ── 显著回归计数 ──
        any_entries = regressions + improvements
        has_per_case_sig = any(
            e.get("significance") is not None for e in any_entries
        )
        if has_per_case_sig:
            # 有重复采样：只统计 per-case 判定为显著的回归
            sig_regressions = [
                r
                for r in regressions
                if r.get("significance") == "significant_regression"
            ]
            regression_count = len(sig_regressions)
        else:
            # 单次采样：仅当整体结论为显著回归时才把方向性回归计入
            regression_count = (
                len(regressions)
                if overall_verdict == "significant_regression"
                else 0
            )

        return {
            "baseline_name": baseline_name,
            "compared_at": datetime.now(timezone.utc).isoformat(),
            "overall": {
                "current_avg_score": round(avg_current, 4),
                "baseline_avg_score": round(avg_baseline, 4),
                "delta": round(avg_current - avg_baseline, 4),
                "current_pass_rate_ci": [round(curr_lower, 4), round(curr_upper, 4)],
                "baseline_pass_rate_ci": [round(base_lower, 4), round(base_upper, 4)],
                "verdict": overall_verdict,
            },
            "significance": {
                "mcnemar_p_value": round(mcnemar_p, 4),
                "mcnemar_conclusion": describe_change(mcnemar_p),
                "paired_t_p_value": round(t_p, 4),
                "paired_t_statistic": round(t_stat, 4),
                "paired_t_conclusion": describe_change(t_p),
                "verdict": overall_verdict,
            },
            "regressions": regressions,
            "improvements": improvements,
            "new_cases": new_cases,
            "removed_cases": removed_cases,
            "summary": {
                "total_compared": len(paired_current),
                "regression_count": regression_count,
                "improvement_count": len(improvements),
                "new_count": len(new_cases),
                "removed_count": len(removed_cases),
                "verdict": overall_verdict,
            },
        }

    @staticmethod
    def _serialize_run(run_result: RunResult) -> dict[str, Any]:
        """将运行结果序列化为可存储的字典。

        除单分数外，额外保留重复采样分布（repeat 分数列表），
        供跨版本 per-case 显著性检验使用。
        """
        scores: dict[str, float] = {}
        repeat_scores: dict[str, list[float]] = {}
        for cr in run_result.case_results:
            if cr.eval_score:
                scores[cr.case.id] = cr.eval_score.overall_score
            # 重复采样分布：优先取 repeats 的分数；无 repeats 则退化为单一代表分
            if cr.repeats and any(r.overall_score is not None for r in cr.repeats):
                repeat_scores[cr.case.id] = [
                    r.overall_score for r in cr.repeats
                ]
            elif cr.eval_score is not None:
                repeat_scores[cr.case.id] = [cr.eval_score.overall_score]
        return {
            "run_id": run_result.run_id,
            "mode": run_result.mode,
            "started_at": run_result.started_at,
            "completed_at": run_result.completed_at,
            "scores": scores,
            "repeat_scores": repeat_scores,
            "summary": run_result.summary,
        }

    @staticmethod
    def _extract_scores(run_result: RunResult) -> dict[str, float]:
        scores: dict[str, float] = {}
        for cr in run_result.case_results:
            if cr.eval_score:
                scores[cr.case.id] = cr.eval_score.overall_score
        return scores

    @staticmethod
    def _extract_repeat_distributions(
        run_result: RunResult,
    ) -> dict[str, list[float]]:
        dist: dict[str, list[float]] = {}
        for cr in run_result.case_results:
            vals = [r.overall_score for r in cr.repeats]
            # 过滤 None，避免重复采样未评分阶段残留
            vals = [v for v in vals if v is not None]
            if not vals and cr.eval_score is not None:
                vals = [cr.eval_score.overall_score]
            if vals:
                dist[cr.case.id] = vals
        return dist

    @staticmethod
    def _extract_scores_from_dict(data: dict[str, Any]) -> dict[str, float]:
        return data.get("scores", {})

    @staticmethod
    def _extract_repeat_distributions_from_dict(
        data: dict[str, Any],
    ) -> dict[str, list[float]]:
        return data.get("repeat_scores", {})
