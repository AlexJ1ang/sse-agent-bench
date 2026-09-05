"""baseline 三档统计显著性判定 + gate 联动单元测试。"""

from __future__ import annotations

from agent_harness.core.baseline import (
    BaselineStore,
    _significance_verdict,
)
from agent_harness.core.gate import GateRule, evaluate_gate
from agent_harness.models import (
    CaseResult,
    EvalScore,
    RepeatResult,
    RunResult,
    SSEStreamResult,
)
from agent_harness.models import (
    TestCase as HarnessTestCase,
)


def _run(case_scores: dict[str, list[float]]) -> RunResult:
    """按 case_id -> 分数列表（repeat 分数）构造 RunResult。

    分数列表长度 >1 时写入 repeats，触发 per-case 显著性检验。
    """
    case_results: list[CaseResult] = []
    for cid, scores in case_scores.items():
        case = HarnessTestCase(id=cid, question="q")
        rep_score = scores[0]
        cr = CaseResult(
            case=case,
            stream=SSEStreamResult(answer_parts=["a"]),
            success=True,
            eval_score=EvalScore(
                case_id=cid, overall_score=rep_score, overall_passed=rep_score >= 0.5
            ),
            repeats=[
                RepeatResult(attempt=i + 1, success=True, overall_score=s)
                for i, s in enumerate(scores)
            ],
            repeat_count=len(scores),
        )
        case_results.append(cr)
    return RunResult(run_id="r", mode="eval", case_results=case_results)


def _no_repeat_run(case_scores: dict[str, float]) -> RunResult:
    """单次采样（无 repeats）。"""
    case_results: list[CaseResult] = []
    for cid, score in case_scores.items():
        case = HarnessTestCase(id=cid, question="q")
        cr = CaseResult(
            case=case,
            stream=SSEStreamResult(answer_parts=["a"]),
            success=True,
            eval_score=EvalScore(case_id=cid, overall_score=score, overall_passed=score >= 0.5),
        )
        case_results.append(cr)
    return RunResult(run_id="r", mode="eval", case_results=case_results)


# ── 三档判定函数 ───────────────────────────────────────────────


def test_significance_verdict_regression():
    assert _significance_verdict(0.01, -0.3) == "significant_regression"


def test_significance_verdict_improvement():
    assert _significance_verdict(0.01, 0.3) == "significant_improvement"


def test_significance_verdict_not_significant():
    assert _significance_verdict(0.5, -0.3) == "no_significant_change"


# ── baseline 对比 ──────────────────────────────────────────────


def test_compare_new_and_removed_cases(tmp_path):
    store = BaselineStore(tmp_path / "baselines")
    store.save("v1", _no_repeat_run({"c1": 0.9}))
    cmp = store.compare(_no_repeat_run({"c2": 0.9}), "v1")
    assert cmp["summary"]["new_count"] == 1
    assert cmp["summary"]["removed_count"] == 1


def test_compare_missing_baseline(tmp_path):
    store = BaselineStore(tmp_path / "baselines")
    cmp = store.compare(_no_repeat_run({"c1": 0.9}), "missing")
    assert "error" in cmp


def test_compare_per_case_significance_with_repeats(tmp_path):
    """重复采样下，per-case paired t-test 能判定显著回归。"""
    store = BaselineStore(tmp_path / "baselines")
    # 基线高分、当前低分：差异应显著（多次采样方差小）
    store.save("v1", _run({"c1": [0.9, 0.88, 0.92]}))
    cmp = store.compare(_run({"c1": [0.2, 0.18, 0.22]}), "v1")
    entry = cmp["regressions"][0]
    assert entry["significance"] == "significant_regression"
    assert cmp["summary"]["regression_count"] == 1


def test_compare_single_sample_uses_overall_verdict(tmp_path):
    """单次采样（无 repeats）：只有整体显著回归才计入 regression_count。"""
    store = BaselineStore(tmp_path / "baselines")
    store.save("v1", _no_repeat_run({"c1": 0.9, "c2": 0.9}))
    cmp = store.compare(_no_repeat_run({"c1": 0.2, "c2": 0.2}), "v1")
    # 两个 case 都大幅下降，整体 paired-t 应显著回归
    assert cmp["summary"]["regression_count"] == 2
    assert cmp["significance"]["paired_t_conclusion"] == "significant"


def test_compare_overall_verdict_field(tmp_path):
    store = BaselineStore(tmp_path / "baselines")
    store.save("v1", _run({"c1": [0.4, 0.4, 0.4]}))
    cmp = store.compare(_run({"c1": [0.4, 0.4, 0.4]}), "v1")
    # 无变化 → 不显著
    assert cmp["overall"]["verdict"] == "no_significant_change"


# ── gate 联动 ──────────────────────────────────────────────────


def test_gate_counts_only_significant_regressions(tmp_path):
    store = BaselineStore(tmp_path / "baselines")
    store.save("v1", _run({"c1": [0.9, 0.9, 0.9]}))
    cmp = store.compare(_run({"c1": [0.2, 0.2, 0.2]}), "v1")
    summary = {"eval_avg_score": 0.9, "eval_pass_rate": 0.9}
    result = evaluate_gate(summary, cmp, GateRule(max_regressions=0))
    checks = {c.name: c for c in result.checks}
    assert checks["max_regressions"].passed is False
    assert result.passed is False


def test_gate_skips_regression_without_baseline():
    summary = {"eval_avg_score": 0.9, "eval_pass_rate": 0.9}
    result = evaluate_gate(summary, None, GateRule(max_regressions=0))
    checks = {c.name: c for c in result.checks}
    # 无基线可对比 → 跳过该项，不判失败
    assert checks["max_regressions"].passed is True
